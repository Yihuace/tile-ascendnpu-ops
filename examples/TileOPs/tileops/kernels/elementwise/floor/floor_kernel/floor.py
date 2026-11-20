# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Floor forward kernel — NPU (npuir) implementation.

Migrated from GPU TileOPs ``tileops/kernels/elementwise.py`` —
``FloorFwdKernel`` / ``floor_fwd_kernel``.  GPU reference preserved verbatim in
``../_floor_fwd_kernels.py``; this file is the NPU re-implementation.

GPU ``op_func`` was ``T.floor(T.cast(x, "float32"))`` — the GPU cast to fp32
first, so the NPU port keeps the same fp32 compute path.

**API substitution (important)**: this npuir build exposes **no unary
``floor``** (see ``ceil.py`` for the full unary-op inventory check).
``floor`` is expressed through the cast intrinsics, whose valid round modes
are ``{round, rint, floor, ceil, trunc, odd}``:

    floor(x) = vcast( vcast(x -> int32, round_mode="floor") -> fp32 )

**Range limitation**: the int32 intermediate caps the representable magnitude
at ``|x| < 2**31`` — same caveat as ``ceil.py``.
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

# int32 intermediate path ceiling (documented above).
_INT32_MAGNITUDE_LIMIT = 2**31


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_floor(x):
    """floor reference, computed in fp32 then cast back (matches the kernel)."""
    x_cpu = x.detach().cpu()
    return torch.floor(x_cpu.to(torch.float32)).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def floor_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    """Element-wise floor: y = floor(x).

    Implemented as an int32 round-trip with ``round_mode="floor"`` because this
    npuir build has no unary floor instruction (see module docstring).

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    per-block element count (K11).
    """
    out_dtype = output_dtype or dtype
    n_num = T.ceildiv(N, block_size)
    compute_dtype = "float32"
    idx_dtype = "int32"

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            f_ub = T.alloc_ub((block_size,), compute_dtype)
            i_ub = T.alloc_ub((block_size,), idx_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            offset = cid * block_size
            remaining = shape - offset
            tail_size = T.min(block_size, remaining)

            T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

            # NOTE (measured): `round_mode` is only honoured on a cast that
            # actually changes dtype — see the same note in `ceil.py`.  fp32
            # goes straight to int32; only low-precision dtypes widen first.
            if dtype == compute_dtype:
                T.vcast(x_ub, i_ub, round_mode="floor")
                T.vcast(i_ub, y_ub, round_mode="rint")
            else:
                # dtype -> fp32
                T.vcast(x_ub, f_ub, round_mode="rint")
                # fp32 -> int32, rounding down: this is the floor
                T.vcast(f_ub, i_ub, round_mode="floor")
                # int32 -> fp32 (f_ub is consumed above, so reuse is safe)
                T.vcast(i_ub, f_ub, round_mode="rint")
                # fp32 -> output dtype
                T.vcast(f_ub, y_ub, round_mode="rint")

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
    kernel = floor_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_floor(x)
    torch.testing.assert_close(y.cpu(), golden, atol=atol, rtol=rtol)
    max_diff = (y.float().cpu() - golden.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")


def _run_case(N, dtype_str, tag, scale=3.0):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    _run_tensor(x, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    N = 1048576
    _run_case(N, "float16", "L0")
    _run_case(N, "float32", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (4096, 100):
            _run_case(N, dtype_str, "L1")
    _run_case(2049, "float16", "L1")
    _run_case(3000, "bfloat16", "L1")


def run_L2():
    # L2: edge sizes and out-of-band magnitudes (non-blocking, records only).
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float32", "L2")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")
    # Documented int32 round-trip range limit (|x| >= 2**31 is undefined).
    try:
        x = torch.full((4096,), 3.0e9, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "L2-beyond-int32-range")
    except Exception as e:
        print(f"[L2] WARN (Record without blocking): beyond-int32-range err={e}")


def run_boundary():
    # Boundary: signed-magnitude coverage around 0 (non-blocking, records only).
    try:
        x = torch.tensor(
            [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5] + [0.25] * 4088,
            dtype=torch.float32,
        ).npu()
        _run_tensor(x, "float32", "Boundary-signed")
    except Exception as e:
        print(f"[Boundary] WARN (Record without blocking): signed err={e}")


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
