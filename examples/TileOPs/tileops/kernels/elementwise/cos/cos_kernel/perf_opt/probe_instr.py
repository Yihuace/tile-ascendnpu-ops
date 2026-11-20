"""Instruction-cost probe for the cos kernel (Stage 4, round 5).

Why: the C3 kernel is NOT stall-bound -- msprof reports ``aiv_vec_ratio = 0.967``
and ``aiv_vec_fp32_ratio = 0.945``, so the vector pipe is essentially always
busy; ratio is 53-56% only because the metric divides by a 64-fp32-ops/cycle
peak while the measured retirement rate is ~34.8 ops/cycle.  The remaining
levers are (a) the per-instruction cycle cost mix and (b) how many element-ops
the profiler credits per instruction.

Method: every variant runs the same shell --

    1 vcast(fp16->fp32) + K chained vmul(fp32) + 1 <special> + 1 vcast(fp32->fp16)

so ``duration(special variant) - duration(mul reference with the same slot
count)`` = ``cost(special) - cost(vmul)``, and the ``mul8/12/16/20`` slope gives
``cost(vmul)`` directly.

Two toolchain constraints measured while writing this probe (recorded in
opt_log.md as traps):

  1. A Python-level ``if/elif`` chain that assigns a buffer alias inside the
     ``T.prim_func`` body does not compile: TVMScript parses the body statically
     -> "Undefined variable: src".
  2. Several ``@T.prim_func`` definitions inside branches of one
     ``@tilelang.jit`` factory do not compile either: the jit factory reports
     ``script.ir_builder.tir.Arg ... expected Object but got str``.  Hence one
     top-level factory per variant below.

Run one variant per invocation, under msprof:

    msprof op --kernel-name=main --output=<dir> --launch-count=15 --warm-up=5 \
        --dump=off --aic-metrics=BasicInfo,ArithmeticUtilization,PipeUtilization \
        python probe_instr.py --variant sin --case elementwise-16m-float16
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

BLOCK_SIZE = 4096
NUM_CORES = 48
_MUL = 0.999999


@tilelang.jit(out_idx=[1], target="npuir")
def probe_mul(N, dtype, k_mul, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """K chained vmul between the in/out casts."""
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    src, dst = a_ub, b_ub
                    for _ in range(k_mul):
                        T.vmul(src, _MUL, dst)
                        src, dst = dst, src
                    T.vcast(dst, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def probe_dual(N, dtype, k_pairs, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """Two INDEPENDENT interleaved vmul chains (same op count as probe_mul 2*k).

    Purpose: tell whether the ~93 cycles/instruction measured on a single
    dependent chain is issue-rate (throughput) or dependency latency.  If
    interleaving two chains drops the per-op cost, ILP is available and a
    2-stream cos kernel would also raise ratio.
    """
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            z_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            c_ub = T.alloc_ub((block_size,), compute_dtype)
            d_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    T.vcast(x_ub, c_ub, round_mode="rint")
                    sa, sb = a_ub, b_ub
                    sc, sd = c_ub, d_ub
                    for _ in range(k_pairs):
                        T.vmul(sa, _MUL, sb)
                        T.vmul(sc, _MUL, sd)
                        sa, sb = sb, sa
                        sc, sd = sd, sc
                    T.vcast(sa, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])
                    T.vcast(sc, z_ub, round_mode="rint")
                    T.copy(z_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def probe_sin(N, dtype, k_mul, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            c_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    src, dst = a_ub, b_ub
                    for _ in range(k_mul):
                        T.vmul(src, _MUL, dst)
                        src, dst = dst, src
                    T.vsin(src, c_ub)
                    T.vcast(c_ub, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def probe_cos(N, dtype, k_mul, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            c_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    src, dst = a_ub, b_ub
                    for _ in range(k_mul):
                        T.vmul(src, _MUL, dst)
                        src, dst = dst, src
                    T.vcos(src, c_ub)
                    T.vcast(c_ub, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def probe_cast_rint(N, dtype, k_mul, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """One extra in-place same-dtype quantising vcast (round_mode=rint)."""
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    src, dst = a_ub, b_ub
                    for _ in range(k_mul):
                        T.vmul(src, _MUL, dst)
                        src, dst = dst, src
                    T.vcast(src, src, round_mode="rint")
                    T.vcast(src, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def probe_cast_floor(N, dtype, k_mul, block_size=BLOCK_SIZE, num_cores=NUM_CORES):
    """One extra in-place same-dtype quantising vcast (round_mode=floor)."""
    compute_dtype = "float32"
    total_blocks = (N + block_size - 1) // block_size
    iters = (total_blocks + num_cores - 1) // num_cores

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_cores, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), dtype)
            a_ub = T.alloc_ub((block_size,), compute_dtype)
            b_ub = T.alloc_ub((block_size,), compute_dtype)
            for it in T.serial(iters):
                blk = it * num_cores + cid
                if blk < total_blocks:
                    offset = blk * block_size
                    tail = T.min(block_size, shape - offset)
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcast(x_ub, a_ub, round_mode="rint")
                    src, dst = a_ub, b_ub
                    for _ in range(k_mul):
                        T.vmul(src, _MUL, dst)
                        src, dst = dst, src
                    T.vcast(src, src, round_mode="floor")
                    T.vcast(src, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


# variant name -> (factory, number of chained vmul)
VARIANTS = {
    "mul8": (probe_mul, 8),
    "mul12": (probe_mul, 12),
    "mul16": (probe_mul, 16),
    "mul20": (probe_mul, 20),
    "mul15": (probe_mul, 15),
    "sin": (probe_sin, 15),
    "cos": (probe_cos, 15),
    "cast_rint": (probe_cast_rint, 15),
    "cast_floor": (probe_cast_floor, 15),
    "single20": (probe_mul, 20),
    "dual10": (probe_dual, 10),
}


def run_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--case", required=True)
    ap.add_argument(
        "--inventory",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workload_inventory.json"),
    )
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    args = ap.parse_args()

    with open(args.inventory, encoding="utf-8") as handle:
        inventory = json.load(handle)
    case = next(item for item in inventory["workloads"] if item["workload_id"] == args.case)
    dtype = DTYPES[case["dtype"]]
    n_total = int(case["params"]["n_total"])

    factory, k_mul = VARIANTS[args.variant]
    kernel = factory(n_total, str(dtype).replace("torch.", ""), k_mul)
    x = torch.randn(n_total, dtype=dtype, device="npu")

    for _ in range(args.warm_up):
        kernel(x, n_total)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        kernel(x, n_total)
    torch.npu.synchronize()
    print(f"[probe] {args.variant} {args.case} done")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
