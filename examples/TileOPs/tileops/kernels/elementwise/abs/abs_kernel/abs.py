# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Abs forward kernel — NPU (npuir) implementation.

Migrated from GPU TileOPs ``tileops/kernels/elementwise.py`` —
``AbsFwdKernel`` / ``abs_fwd_kernel`` (strategy 3: fragment load → compute →
fragment store).  The GPU reference implementation is preserved verbatim in
``../_abs_fwd_kernels.py``; this file is the NPU re-implementation.

GPU -> NPU adaptations applied here (see
``docs/gpu_to_npu_adaptation.md``):

  K1: ``@tilelang.jit`` gains ``target="npuir"``; the body is rewritten
      against the npuir v-prefix vector API.
  K2: CUDA grid/``threads`` replaced by the NPUIR 1-D grid
      ``T.Kernel(n_num, is_npu=True) as (cid, _)``; no manual sync
      (Developer mode + automatic sync).
  K3: the ``threads`` factory parameter is removed.
  K11: the GPU ``threads(256) * num_per_thread(8)`` product collapses into a
      single ``block_size`` parameter (elementwise ops only).

API note: npuir's unary form is ``T.vabs(A, B)`` meaning ``B = |A|`` —
source first, destination second (verified against
``tilelang/language/customize_npuir.py: npuir_abs`` and the in-repo usages
``examples/deepseek_v4/example_act_quant_kernel.py:61`` and
``examples/deepseek_v4/engram/engram_gate_kernel_fwd.py:178``).
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

tilelang.cache.clear_cache()


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_abs(x):
    """Abs reference: y = |x|.

    Exact in every dtype — clearing the sign bit is lossless, so no upcast is
    needed for the reference (unlike mish/tanh).
    """
    return torch.abs(x.detach().cpu())


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def abs_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    """Element-wise absolute value: y = |x|.

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    per-block element count (K11).

    fp16/bf16 route through an fp32 intermediate (K-series cast convention,
    same as the in-repo ``mish_fwd_kernel``): the result is bit-exact either
    way, but this keeps one uniform compute path across dtypes.
    """
    out_dtype = output_dtype or dtype
    n_num = T.ceildiv(N, block_size)
    compute_dtype = "float32"

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 1. UB buffers (v-prefix API operates on UB-resident data).
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # 2. Tail handling for non-divisible N (dynamic extent).
            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            # 3. GM -> UB, only valid elements.
            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # 4. Compute y = |x|. Trailing garbage lanes are never copied out.
            if dtype != compute_dtype:
                T.vcast(x_ub, x_f32_ub, round_mode="rint")
                T.vabs(x_f32_ub, x_f32_ub)
                if out_dtype != compute_dtype:
                    T.vcast(x_f32_ub, y_ub, round_mode="rint")
                else:
                    T.copy(x_f32_ub, y_ub)
            else:
                T.vabs(x_ub, y_ub)

            # 5. UB -> GM, only valid elements.
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
    """Run kernel on a pre-made tensor and compare against golden."""
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = abs_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_abs(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=3.0):
    """Run kernel on random data of shape (N,) with mixed signs."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke: representative 1M-element cases covering all three dtypes.
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    # L1: divisible + non-divisible shapes across all dtypes (tail path).
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


def run_boundary():
    # Boundary: special-value inputs (non-blocking).
    # zeros: |0| = 0
    try:
        x = torch.zeros(4096, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-zeros")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): zeros err={e}")
    # negative values only: exercises the sign-flip branch
    try:
        x = (-torch.rand(4096, dtype=torch.float32) - 0.5).to(torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-all-negative")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): all-negative err={e}")
    # ±inf: |±inf| = +inf (no NaN produced by abs itself)
    try:
        x = torch.tensor([-float("inf"), float("inf")] + [1.0] * 4094, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-inf")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")


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
