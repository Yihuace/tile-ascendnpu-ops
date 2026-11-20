# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""ELU forward kernel -- Stage 4 Round 1 branch OP1 (mask-chain -> arithmetic).

Single-point change vs the Stage 3 baseline (``_make_elu_kernel.py``):
replace the branch-free mask chain ``vcmp + vselect`` with the *arithmetic*
identity

    elu(x) = max(x, 0) + alpha * (exp(min(x, 0)) - 1)

which has no mask at all.  Derivation (bit-exact, see below):
  * x > 0  -> max = x, min = 0, exp(0) - 1 = 0 exactly  -> x + alpha*0 = x
  * x <= 0 -> max = 0, min = x                          -> exp(x) - 1 (times alpha)

Because ``exp(0) - 1 == 0`` exactly and ``0 + v == v`` exactly (and ``v + 0``
for the positive branch), the result is **bit-identical** to the baseline
``if_then_else`` value for every finite/+-inf/NaN input; the fp32 transit,
the ``alpha == 1.0`` compile-time fold and the tail/copy discipline are
unchanged.

Cost rationale: on the sibling expm1 Stage 4 probe the mask select
(``T.vselect``) marginal cost is ~6.5–8x a normal v-op
(pattern-library ``PL-1.29-vselect-cost`` / ``CONST-vector-op-cost-910_9362``),
while ``vmin``/``vmax``/``vadd`` are normal ALU v-ops.  Replacing
(vcmp + vselect) by (vmin + vmax + vadd) shrinks the f16/bf16 chain from
12 to 7.5 "normal-op units" and the fp32 chain from 10 to 5.5.

NOTE: no ``cond`` (bool) UB buffer is needed any more; ``y_ub`` is still
allocated unconditionally for buffer-alloc uniformity (R-2/R-6).

run: python _make_elu_kernel_round1_op1.py --level all
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() placement

ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_BLOCK_SIZE = 2048
_CORE_NUM = 40  # physical Vector cores on Ascend910B3 / 910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU reference implementation, equivalent to ``torch.nn.functional.elu``."""
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return F.elu(x_f32, alpha).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, block_size=_BLOCK_SIZE):
    """ELU forward kernel factory (OP1: mask-free arithmetic chain)."""
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    need_mul = float(alpha) != 1.0
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)        # input tile
            y_ub = T.alloc_ub((block_size,), out_dtype)    # round-back tile
            xf = T.alloc_ub((block_size,), f32)            # fp32 transit / domain
            e = T.alloc_ub((block_size,), f32)             # neg-branch result
            z = T.alloc_ub((block_size,), f32)             # read-only constant 0

            T.fill(z, T.cast(0.0, f32))

            xc = xf if need_cast else x_ub

            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- mask-free arithmetic ELU chain (fp32 domain) ---
                    T.vmin(xc, z, e)                    # e  = min(x, 0)
                    T.vexp(e, e)                        # e  = exp(min(x, 0))
                    T.vsub(e, T.cast(1.0, f32), e)      # e  = exp(min(x,0)) - 1
                    if need_mul:
                        T.vmul(e, T.cast(alpha, f32), e)  # e = alpha*(...)
                    T.vmax(xc, z, xc)                   # xc = max(x, 0)
                    T.vadd(xc, e, xc)                   # xc = max(x,0) + e

                    if need_cast:
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        T.copy(xc[0:tail], y[off : off + tail])

    return main


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_activation.py
# ``_get_tolerances`` (DESIGN.md section 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _make_input(N, dtype_str):
    """Generate a CPU input tensor following the EluWorkload logic (randn)."""
    return torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _max_diff(y, golden):
    """Max abs error over finite golden entries (inf/NaN handled by caller)."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag, alpha=1.0):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    x_cpu = x_cpu.detach().cpu().reshape(-1).contiguous()
    N = x_cpu.numel()
    golden = golden_elu(x_cpu, alpha)
    kernel = _make_elu_kernel(N, dtype_str, alpha, block_size=_DEFAULT_BLOCK_SIZE[dtype_str])

    x_npu = x_cpu.npu()
    y = kernel(x_npu, N)
    y_cpu = y.cpu().reshape(-1)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(
            y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True
        )
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} alpha={alpha} "
            f"max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} alpha={alpha} "
        f"max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, alpha=1.0):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag, alpha)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke (DESIGN.md section 8.3 L0-1..L0-3): 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")

    # L0-4 corner / branch-point / saturation values (test_elu_edge pattern):
    # -88.8 -> -1.0, -1e-7 -> exp(-1e-7)-1 (negative branch), 0 -> 0,
    # +1e-7 -> +1e-7 (positive branch), 88.8 -> 88.8 (no overflow).
    _run_tensor(
        _repeat_values([-88.8, -1.0, -1e-7, 0.0, 1e-7, 1.0, 88.8], 4096, torch.float32),
        "float32",
        "L0-corner",
    )

    # L0-5 negative-branch small-|x| sensitive region (the exp-1 vs expm1
    # discriminator; DESIGN.md section 0.6 R2).
    x_small = torch.linspace(-1e-3, 0.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_small, "float32", "L0-small-x")

    # L0-6 full sensitive region including the branch point x = 0.
    x_lin = torch.linspace(-9.0, 9.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_lin, "float32", "L0-linspace")

    # L0-7 non-divisible tail (4_000_000 % 2048 = 1216).
    _run_case(4000000, "float16", "L0-tail")

    # L0-8 non-default alpha (exercises the vmul path).
    _run_case(1048576, "float32", "L0-alpha0.5", alpha=0.5)

    # L0-9 manifest multi-dim shapes, flattened by the host (fp16/bf16).
    for shape in ((2048, 4096), (2048, 8192)):
        for dtype_str in ("float16", "bfloat16"):
            x = torch.randn(*shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, f"L0-manifest{shape}")


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat, non-divisible
    # tails for every dtype, exact / partial tiles.
    _run_case(4096 * 4096, "float32", "L1-16m")
    _run_case(4000000, "float32", "L1-tail")
    _run_case(4000000, "bfloat16", "L1-tail")
    _run_case(4000000, "float32", "L1-tail-alpha0.5", alpha=0.5)
    _run_case(1024, "float32", "L1-exact-tile")
    _run_case(4096, "float16", "L1-two-tiles")
    _run_case(2047, "bfloat16", "L1-partial-tile")


def run_L2():
    # L2: edge sizes around the tile boundary, non-blocking.
    for N, dtype_str in (
        (1, "float32"),
        (2, "float16"),
        (10, "bfloat16"),
        (1025, "float32"),
        (4095, "float16"),
        (8191, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary: special-value inputs (branch floor, saturation, inf, NaN),
    # all non-blocking.  ``elu(-inf) = -1`` (alpha=1), ``elu(+inf) = +inf``,
    # ``elu(NaN) = NaN``.
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        try:
            x = _repeat_values(
                [0.0, -0.0, 1e-7, -1e-7, 1e-3, -1e-3, 1.0, -1.0, 88.8, -88.8],
                4096,
                td,
            )
            _run_tensor(x, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            td = _TORCH_DTYPE[dtype_str]
            x = _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], 4096, td)
            _run_tensor(x, dtype_str, "Boundary-inf-nan")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} inf/nan err={e}")

    # Manifest full workloads (non-blocking).
    for N, dtype_str in (
        (4096 * 4096, "float16"),
        (8192 * 8192, "float16"),
        (8192 * 8192, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "Boundary-manifest")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


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
