#!/usr/bin/env python3
"""Isolate the cnn scalar-mode slowdown (bf16/fp16 vs fp32) for the R6 kernel.

Variants all use the production tiling (R=4096 planes of L=3136, P=2 planes per
tile, 40 persistent cores):

    copy1d  flat 8192-element 1D copy  y <- a          (control: 2N DMA)
    copy2d  (P,L) 2D copy              y <- a          (tiling cost alone)
    vbrc    copy + vbrc(P,1 -> P,L)                    (broadcast pass)
    add     copy + vbrc + vadd                         (production scalar mode)

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python3 r6_scalar_diag.py --variant copy2d --dtype bfloat16
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

R, L, PLANES, B_LEN = 4096, 3136, 2, 256
CHUNK = B_LEN // PLANES
GROUPS = R // PLANES
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@tilelang.jit(out_idx=[1], target="npuir")
def kernel(variant, dtype, block_size=8192, num_cores=None):
    n_total = R * L
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if variant == "copy1d":
        num_full = n_total // block_size
        nk = min(num_full, vector_cores)
        nlt = T.ceildiv(num_full, nk)

        @T.prim_func
        def main(a: T.Tensor((n_total,), dtype), y: T.Tensor((n_total,), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                for i in T.Pipelined(nlt, num_stages=2):
                    block_id = i * nk + cid
                    if block_id < num_full:
                        off = block_id * block_size
                        T.copy(a[off : off + block_size], a_ub)
                        T.copy(a_ub, y_ub)
                        T.copy(y_ub, y[off : off + block_size])

        return main

    nk = min(GROUPS, vector_cores)
    nlt = T.ceildiv(GROUPS, nk)

    @T.prim_func
    def main(
        a: T.Tensor((R, L), dtype),
        b: T.Tensor((CHUNK, PLANES, 1), dtype),
        y: T.Tensor((R, L), dtype),
    ):
        with T.Kernel(nk, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((PLANES, L), dtype)
            b_ub = T.alloc_ub((PLANES, 1), dtype)
            b_tile = T.alloc_ub((PLANES, L), dtype)
            y_ub = T.alloc_ub((PLANES, L), dtype)
            for i in T.Pipelined(nlt, num_stages=2):
                group = i * nk + cid
                if group < GROUPS:
                    T.copy(a[group * PLANES : (group + 1) * PLANES, 0:L], a_ub)
                    if variant == "copy2d":
                        T.copy(a_ub, y_ub)
                    else:
                        T.copy(b[group % CHUNK, 0:PLANES, 0:1], b_ub)
                        T.vbrc(b_ub, b_tile)
                        if variant == "vbrc":
                            T.copy(a_ub, y_ub)
                        else:
                            T.vadd(a_ub, b_tile, y_ub)
                    T.copy(y_ub, y[group * PLANES : (group + 1) * PLANES, 0:L])

    return main


def run(args):
    dtype = DTYPES[args.dtype]
    if args.variant == "copy1d":
        prog = kernel("copy1d", args.dtype)
        a = torch.randn(R * L, dtype=dtype, device="npu")
        for _ in range(args.launches):
            prog(a)
    else:
        prog = kernel(args.variant, args.dtype)
        a = torch.randn(R, L, dtype=dtype, device="npu")
        b = torch.randn(CHUNK, PLANES, 1, dtype=dtype, device="npu")
        for _ in range(args.launches):
            prog(a, b)
    torch.npu.synchronize()
    print("[diag] done", file=sys.stderr)


def measure(args):
    out_dir = args.output
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "msprof",
        "op",
        "--kernel-name=main",
        f"--launch-count={args.launch_count}",
        "--warm-up=5",
        "--aic-metrics=Default,Roofline",
        f"--output={out_dir}",
        sys.executable,
        os.path.abspath(__file__),
        "--variant",
        args.variant,
        "--dtype",
        args.dtype,
        "--launches",
        str(args.launches),
        "--no-msprof",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"msprof failed rc={proc.returncode}")
    durations = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        raise SystemExit("no OpBasicInfo rows")
    nbytes = R * L * (4 if args.dtype == "float32" else 2)
    med = statistics.median(durations)
    print(
        f"[diag] variant={args.variant:7s} dtype={args.dtype:9s} n={len(durations)} "
        f"median={med:.3f}us 2N_bw={2 * nbytes / (med * 1e-6) / 1e12:.2f}TB/s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="add", choices=["copy1d", "copy2d", "vbrc", "add"])
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--output", default="")
    ap.add_argument("--no-msprof", action="store_true")
    args = ap.parse_args()
    if args.output:
        measure(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
