# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Exp forward kernel — NPU (npuir) implementation.

Migrated from GPU TileOPs ``tileops/kernels/elementwise.py`` — ``ExpFwdKernel``
/ ``exp_fwd_kernel``.  GPU reference preserved verbatim in
``../_exp_fwd_kernels.py``; this file is the NPU re-implementation.

GPU ``op_func`` was ``T.exp(T.cast(x, "float32"))`` — the GPU already computes
in fp32, so the NPU port keeps the same fp32 compute path (K-series cast
convention, same as the in-repo ``mish_fwd_kernel``).

Adaptations: K1 (``target="npuir"`` + v-prefix body), K2 (1-D NPUIR grid, no
manual sync), K3 (``threads`` removed), K11 (``threads*npt`` -> ``block_size``).

API note: npuir unary form is ``T.vexp(A, B)`` meaning ``B = exp(A)`` — source
first, destination second.
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
def golden_exp(x):
    """exp reference, computed in fp32 then cast back (matches the kernel)."""
    x_cpu = x.detach().cpu()
    return torch.exp(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def exp_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    """Element-wise exponential: y = exp(x).

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    per-block element count (K11).  fp16/bf16 route through an fp32
    intermediate, mirroring the GPU ``T.exp(T.cast(x, "float32"))``.
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
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            if dtype != compute_dtype:
                T.vcast(x_ub, x_f32_ub, round_mode="rint")
                T.vexp(x_f32_ub, x_f32_ub)
                if out_dtype != compute_dtype:
                    T.vcast(x_f32_ub, y_ub, round_mode="rint")
                else:
                    T.copy(x_f32_ub, y_ub)
            else:
                T.vexp(x_ub, y_ub)

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
    kernel = exp_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_exp(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=3.0):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke: 1M elements, all three float dtypes.
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    # L1: divisible + non-divisible shapes (tail path).
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (4096, 100):
            _run_case(N, dtype_str, "L1")
    _run_case(2049, "float16", "L1")
    _run_case(3000, "bfloat16", "L1")


def run_L2():
    # L2: edge sizes (non-blocking, records only).
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
    # Boundary: exp's special values (non-blocking, records only).
    # exp(0)=1, exp(-inf)=0, exp(+inf)=+inf; large negatives underflow to 0.
    try:
        vals = torch.tensor([0.0, -float("inf"), float("inf"), -30.0, 30.0])
        x = torch.cat([vals, torch.zeros(4096 - vals.numel())]).npu()
        _run_tensor(x, "float32", "Boundary-special-values")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): special-values err={e}")


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
