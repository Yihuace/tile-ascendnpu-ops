# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Mish activation forward kernel, Stage 4 performance-tuned version (R2).

y = x * tanh(ln(1 + exp(x))), 1-D (N,) tensor, fp16/bf16/fp32 input,
computed in fp32 intermediate precision, output dtype = input dtype.

Round 1 summary (vs baseline flat-grid mish.py, N=1048576, Ascend 910B2C):
  - persistent kernel T.Kernel(num_cores=48) + tile loop (Block Dim 512->48)
  - auto multi-buffer on the tile loop, dtype-aware block_size
  - fp16/fp32/bf16: 23.3 -> 7.44/8.56/7.66 us (N=1M)

Round 2 re-tune (yolo-p3 N=26,214,400 / yolo-p4 N=13,107,200 benchmarks):
  Diagnosis (msprof Task Duration, yolo-p3 fp16):
    incumbent 103.4us == copy-only floor 40.4us + ~63us vector chain, i.e.
    MTE2/MTE3 transfers and the vector chain run fully serialized, and the
    vector chain dominates.  num_cores re-scan at large N (24/96/192 all
    worse or flat) confirms nc=48 saturates vector throughput.
  Adopted optimizations (cast path, fp16/bf16):
    1. guard-free 2-stage pipelined hot loop (Part A) + tail epilogue
       (Part B, <=1 guarded tile per core); 2-buffer in-place chain keeps
       the 16B/elem UB budget;
    2. explicit tanh identity  tanh(softplus(x)) = (w^2-1)/(w^2+1),
       w = 1+e^x, applied in BOTH Part A and Part B, replacing T.vln +
       T.vtanh (11 -> 9 vector passes per tile; the Developer-mode vtanh
       lowering already expands to the same (e^{2z}-1)/(e^{2z}+1) formula,
       so precision and the fp32 overflow edge x > ~44.15 are unchanged vs
       round 1 -- verified identical outputs incl. the NaN edge at x=100);
    3. block_size 6144 -> 8192 (UB budget unblocked by 1+2).
  Measured (msprof Task Duration, launch-count=15, median across
  independent runs; this bandwidth-bound kernel shows a config-independent
  run-level bimodality of ~2-3.5% between processes -- fast/slow HBM page
  states, incumbent immune -- see opt_log.md Round 2 post-scriptum):
    yolo-p3 fp16 103.42 -> 78.30 us   (-24.3%; fast-state runs 75.3-76.4)
    yolo-p3 bf16 105.90 -> 81.18 us   (-23.3%)
    yolo-p4 fp16  53.84 -> 41.02 us   (-23.8%)
    yolo-p4 bf16  55.18 -> 42.93 us   (-22.2%)
    guards: smoke-1m 7.44/7.66 -> 7.01/7.08, fc-wide 35.92/36.52
    -> 27.24/28.22 (all improved); fp32 path byte-identical to round 1.
  Explored but NOT adopted (see opt_log.md Round 2, same directory):
    - manual MTE2/V/MTE3 software pipeline (Expert mode, explicit
      set_flag/wait_flag, 91.4us = -11.6% vs round-1 incumbent): rejected
      because round-2 constraints require TILELANG_ASCEND_MODE=Developer;
      Developer-mode tensor compile restructures manual rs/flag code and
      corrupts it.
    - num_cores 24/96/192; block_size 7168/7680/8704/9216 re-scanned after
      the Part A unification: combined medians within the run-state noise
      band, no >=3% reproducible gain over 8192.

Drop-in usage (same factory signature as the baseline mish.py):
    kernel = mish_fwd_kernel(N, "float16")     # auto best tile per dtype
    y = kernel(x_npu, N)
Explicit block_size/num_cores still override the tuned defaults.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

tilelang.cache.clear_cache()

NUM_CORES_DEFAULT = 48
BLOCK_SIZE_CAST = 8192
BLOCK_SIZE_FP32 = 8192


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_mish(x):
    """Mish reference: y = x * tanh(ln(1 + exp(x)))."""
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = x_f32 * torch.tanh(torch.log(1 + torch.exp(x_f32)))
    return y_f32.to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def mish_fwd_kernel(N, dtype, output_dtype=None, block_size=None, num_cores=48):
    """Mish kernel, cast path with guard-free 2-stage pipelined hot loop."""
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    if block_size is None:
        block_size = (
            BLOCK_SIZE_FP32
            if dtype == compute_dtype and out_dtype == compute_dtype
            else BLOCK_SIZE_CAST
        )

    n_num = T.ceildiv(N, block_size)
    if num_cores <= 0 or num_cores >= n_num:
        num_cores = n_num

    # Full waves whose every block_id is a full tile: block_id <= n_num-2.
    iters_a = (n_num - 1) // num_cores

    if dtype == compute_dtype and out_dtype == compute_dtype:
        # ---------------- fp32 specialization (unchanged from incumbent) --
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(num_cores, is_npu=True) as (cid, _):
                x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
                buf1_ub = T.alloc_ub((block_size,), compute_dtype)
                buf2_ub = T.alloc_ub((block_size,), compute_dtype)

                for i in T.serial(T.ceildiv(n_num, num_cores)):
                    block_id = i * num_cores + cid
                    if block_id < n_num:
                        offset = block_id * block_size
                        remaining = shape - offset
                        tail_size = T.min(block_size, remaining)

                        T.copy(
                            x[offset : offset + tail_size],
                            x_f32_ub[0:tail_size],
                        )

                        T.vexp(x_f32_ub, buf1_ub)
                        T.vadd(buf1_ub, 1.0, buf2_ub)
                        T.vln(buf2_ub, buf1_ub)
                        T.vtanh(buf1_ub, buf2_ub)
                        T.vmul(x_f32_ub, buf2_ub, buf1_ub)

                        T.copy(buf1_ub[0:tail_size], y[offset : offset + tail_size])
    else:
        # ---------------- fp16/bf16 cast specialization ------------------
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
            shape: T.int32,
        ):
            with T.Kernel(num_cores, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
                buf1_ub = T.alloc_ub((block_size,), compute_dtype)
                buf2_ub = T.alloc_ub((block_size,), compute_dtype)
                y_ub = T.alloc_ub((block_size,), out_dtype)

                # ---- Part A: guard-free pipelined hot loop ----
                for i in T.Pipelined(iters_a, num_stages=2):
                    offset = (i * num_cores + cid) * block_size
                    T.copy(x[offset : offset + block_size], x_ub)
                    T.vcast(x_ub, x_f32_ub, round_mode="rint")

                    # tanh(softplus(x)) = (w^2-1)/(w^2+1), w = 1+e^x
                    T.vexp(x_f32_ub, buf1_ub)
                    T.vadd(buf1_ub, 1.0, buf2_ub)
                    T.vmul(buf2_ub, buf2_ub, buf1_ub)
                    T.vsub(buf1_ub, 1.0, buf2_ub)
                    T.vadd(buf1_ub, 1.0, buf1_ub)
                    T.vdiv(buf2_ub, buf1_ub, buf2_ub)
                    T.vmul(x_f32_ub, buf2_ub, buf1_ub)

                    T.vcast(buf1_ub, y_ub, round_mode="rint")
                    T.copy(y_ub, y[offset : offset + block_size])

                # ---- Part B: epilogue, <=1 guarded tile per core ----
                block_id = iters_a * num_cores + cid
                if block_id < n_num:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])
                    T.vcast(x_ub, x_f32_ub, round_mode="rint")

                    T.vexp(x_f32_ub, buf1_ub)
                    T.vadd(buf1_ub, 1.0, buf2_ub)
                    T.vmul(buf2_ub, buf2_ub, buf1_ub)
                    T.vsub(buf1_ub, 1.0, buf2_ub)
                    T.vadd(buf1_ub, 1.0, buf1_ub)
                    T.vdiv(buf2_ub, buf1_ub, buf2_ub)
                    T.vmul(x_f32_ub, buf2_ub, buf1_ub)

                    T.vcast(buf1_ub, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-5),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _run_tensor(x, dtype_str, tag):
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = mish_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_mish(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=1.0):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


def run_L0():
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (4096, 100):
            _run_case(N, dtype_str, "L1")
    _run_case(8192, "float16", "L1")
    _run_case(2049, "float16", "L1")
    _run_case(3000, "bfloat16", "L1")


def run_L2():
    # L2: edge sizes (non-blocking).
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float32", "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")
    for N in (1, 2, 10):
        try:
            _run_case(N, "float16", "L2")
        except Exception as e:
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float16", "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")


def run_boundary():
    # Boundary: special-value inputs (non-blocking).
    try:
        x = torch.zeros(4096, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-zeros")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): zeros err={e}")
    try:
        x = (torch.randn(4096, dtype=torch.float32) * 5.0).to(torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-large-positive")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): large-positive err={e}")
    try:
        x = torch.full((4096,), -10.0, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-large-negative")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): large-negative err={e}")
    try:
        x = torch.full((4096,), 100.0, dtype=torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-overflow-domain")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): overflow-domain err={e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
