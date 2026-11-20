#!/usr/bin/env python3
"""R5 broadcast-path feasibility probe (measurement only -- NOT wired to the Op).

``linear-bias`` is ``input [1024, 4096] + other [1, 4096]``.  The Op layer
resolves the broadcast before dispatch (``AddFwdOp._expand_flat``:
``expand -> contiguous -> view(-1)``), so today the kernel reads a
*materialized* ``[1024, 4096]`` copy of ``other``: 3 x N x 2 B = 24 MB traffic
for 4.19 M bf16 elements, of which 8 MB is a re-read of the same 8 KB row.

This probe measures the alternative *without touching the Op interface*: the
kernel takes ``other`` as a single row ``(row,)``, stages it in UB once, and
re-broadcasts it across the tile (``block_size`` is a multiple of ``row``, and
every tile offset is a multiple of ``row``, so the per-tile pattern is just
``row`` repeated ``block_size // row`` times).

Expected: traffic 24 MB -> 16 MB (a + y), i.e. ~2/3 of the kernel time.

    msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
        python3 r5_bcast_probe.py --dtype bfloat16 --block-size 8192 --launches 25
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

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
N_ROWS = 1024
ROW = 4096


@tilelang.jit(out_idx=[2], target="npuir")
def add_row_broadcast_kernel(N, row, dtype, block_size=8192):
    """y[r, :] = a[r, :] + other_row[:] for a flattened (N,) layout."""
    if block_size % row != 0:
        raise ValueError("block_size must be a multiple of row")
    reps = block_size // row
    num_full = N // block_size
    tail = N - num_full * block_size
    cores = NPUUtils.get().get_aicore_num() * 2
    nk = min(num_full + (1 if tail else 0), cores)
    nlt = T.ceildiv(num_full, nk)

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((row,), dtype),
        y: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(nk, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((block_size,), dtype)
            b_ub = T.alloc_ub((block_size,), dtype)
            row_ub = T.alloc_ub((row,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)

            # Stage the single row in UB once per core, then broadcast it
            # inside the tile (block_size / row copies).
            T.copy(b[0:row], row_ub)
            for r in T.serial(reps):
                T.copy(row_ub[0:row], b_ub[r * row : (r + 1) * row])

            if num_full > 0:
                for i in T.Pipelined(nlt, num_stages=2):
                    block_id = i * nk + cid
                    if block_id < num_full:
                        off = block_id * block_size
                        T.copy(a[off : off + block_size], a_ub)
                        T.vadd(a_ub, b_ub, y_ub)
                        T.copy(y_ub, y[off : off + block_size])

    return main


def measure(module_args):
    """Run this file under msprof and print the Task Duration median."""
    out_dir = module_args.output
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir], check=False)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op",
        "--kernel-name=main",
        f"--launch-count={module_args.launch_count}",
        f"--warm-up=5",
        "--aic-metrics=Default,Roofline",
        f"--output={out_dir}",
        sys.executable, os.path.abspath(__file__),
        "--dtype", module_args.dtype,
        "--block-size", str(module_args.block_size),
        "--launches", str(module_args.launches),
        "--no-msprof",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:], file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"msprof failed rc={proc.returncode}")
    durations = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("Task Duration(us)"):
                    durations.append(float(row["Task Duration(us)"]))
    if not durations:
        raise SystemExit("no OpBasicInfo rows")
    print(f"[probe] n={len(durations)} median={statistics.median(durations):.3f}us "
          f"min={min(durations):.3f} max={max(durations):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(_TORCH_DTYPE))
    ap.add_argument("--block-size", type=int, default=8192)
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--output", default="")
    ap.add_argument("--no-msprof", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify vs torch golden")
    args = ap.parse_args()

    if args.output:
        measure(args)
        return

    dtype = _TORCH_DTYPE[args.dtype]
    a = torch.randn(N_ROWS, ROW, dtype=dtype, device="npu")
    b = torch.randn(1, ROW, dtype=dtype, device="npu")
    prog = add_row_broadcast_kernel(N_ROWS * ROW, ROW, args.dtype, block_size=args.block_size)
    with torch.no_grad():
        if args.check:
            y = prog(a.view(-1), b.view(-1), N_ROWS * ROW)
            ref = torch.add(a.float(), b.float().expand(N_ROWS, ROW)).to(dtype)
            got = y.view(N_ROWS, ROW).cpu()
            eq = torch.equal(got.view(torch.int16), ref.cpu().view(torch.int16))
            md = (got.float() - ref.cpu().float()).abs().max().item()
            print(f"[probe] check bitwise_equal={eq} max_diff={md:.3e}")
        for _ in range(args.launches):
            prog(a.view(-1), b.view(-1), N_ROWS * ROW)
        torch.npu.synchronize()
    print("[probe] done")


if __name__ == "__main__":
    main()
