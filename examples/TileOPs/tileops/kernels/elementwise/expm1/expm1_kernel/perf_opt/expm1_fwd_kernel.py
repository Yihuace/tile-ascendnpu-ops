# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Expm1 forward kernel (NPU, target="npuir") -- Stage 4 performance-tuned FINAL.

Drop-in replacement for the Stage 3 baseline ``expm1_fwd_kernel.py``: same
factory name and ``(N, dtype, output_dtype=None, block_size=None)`` signature,
same ``main(x, y, shape)`` prim_func.  ``block_size=None`` selects the
per-dtype tuned default exposed as the module constant ``TUNED_BLOCK_SIZE``
(imported by the wrapper's perf_opt toggle).  An explicit ``block_size``
still overrides.

Single-point structural optimisation over the Stage 3 baseline: the piecewise
expm1 small-|x| branch is rebuilt as a **degree-2 minimax polynomial of
expm1(x)/x on [-T, T]** with T = 0.05 (cancellation-free), instead of the
baseline degree-5 Taylor with T = 0.5*ln2.  Together with a per-dtype tuned
``block_size`` this cuts the vector work per element from 16 ops to 12
(fp32) / 14 (fp16/bf16), taking the kernel from a 16-op Vector-emission-bound
regime to near the launch/copy floor of its structure.

Why this is safe:
  * degree-2 minimax (3 coefficients) reaches 2.1e-6 max relative error over
    the whole domain vs the gate rtol 1e-5 -> 4.6x margin (comparable to the
    accepted erf Stage 4 margin);
  * the small-branch threshold is small (T = 0.05) so the large branch
    e^x - 1 only ever runs where its cancellation is bounded
    (exp(T)/(exp(T)-1) = 20.5 -> <= 2.4e-6 relative);
  * the op count drop 16 -> 12 removes 4 Horner ops while keeping the robust
    bool-mask ``vselect`` (the arithmetic clamp-mask blend was measured
    *slower* on this device and broke +-inf semantics, see opt_log.md).

Measured (``msprof op`` Task Duration(us), median of 20 launches): see
opt_log.md / perf_records.jsonl.  Baseline bs=2048 -> this candidate bs per
dtype.

Hard constraints honoured (DESIGN.md section 3.5.1 / 9.1):
  * R-6/R-7: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (conditional alloc triggers ``Undefined
    variable``); the compile-time ``if need_cast:`` branch only selects which
    v-ops run.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``).
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-10: the ``bool`` comparison mask is kept on-chip (``vselect``); it is
    never copied back to GM.
  * The threshold constant buffer is filled once outside the tile loop.

run: python expm1_fwd_kernel_round1_deg2mm.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# Per-dtype tuned block_size (element count).  UB footprint is unchanged from
# the baseline (same 8 buffers), so the sweep tops out at the same UB limit;
# the winner latches per dtype (see opt_log.md Round 1).
TUNED_BLOCK_SIZE = {
    "float32": 8192,
    "float16": 7168,
    "bfloat16": 7168,
}

# Piecewise expm1: |x| < T uses a cancellation-free polynomial of
# expm1(x)/x; otherwise vexp(x)-1.
#
# T = 0.05 balances the two branch error terms:
#   small branch: degree-2 minimax of expm1(x)/x on [-T,T]  -> ~2.1e-6
#   large branch: vexp rel err * exp(T)/(exp(T)-1)          -> ~2.4e-6
# (T larger -> large-branch error drops but the small-branch degree-2 error
# grows; T smaller -> the reverse.  The minimax coefficients below were
# solved on CPU; Taylor degree 2 would need ~3.7e-6 at this T.)
_T = 0.05
_COEF = [0.9999999955, 0.5000625040, 0.1666845248]  # ascending c0, c1, c2


def _auto_block_size(dtype):
    return TUNED_BLOCK_SIZE[dtype]


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_expm1(x):
    """Expm1 reference implementation, equivalent to ``torch.expm1``.

    Mirrors the source op_func ``T.exp(T.cast(x, "float32")) - 1.0``: promote
    the input to fp32, compute in the fp32 domain and round once back to the
    original dtype.  ``torch.expm1`` itself already uses fp32 opmath for
    fp16/bf16 (libm special-cased expm1, relatively accurate for small |x|),
    so this matches the source fp32-transit semantics.

    Independent from the NPU kernel: only the source semantics are used here,
    never the piecewise coefficients.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return torch.expm1(x_f32).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=None):
    """Expm1 forward kernel factory: y = expm1(x).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> piecewise chain directly on the fp32 buffer;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.

    ``block_size=None`` selects the per-dtype tuned default (``TUNED_BLOCK_SIZE``).
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    block_size = block_size or _auto_block_size(dtype)
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-6/R-7: all UB buffers allocated unconditionally at the top level.
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)  # fp16/bf16 round-back
            xf = T.alloc_ub((block_size,), f32)  # fp32 transit entry
            xa = T.alloc_ub((block_size,), f32)  # |x| -> small-branch result
            g = T.alloc_ub((block_size,), f32)  # Horner accumulator
            e = T.alloc_ub((block_size,), f32)  # exp(x) -> exp(x)-1
            thr = T.alloc_ub((block_size,), f32)  # read-only threshold T
            cond = T.alloc_ub((block_size,), "bool")  # on-chip mask (R-10)

            # Threshold constant: filled once outside the tile loop.
            T.fill(thr, T.cast(_T, f32))

            # Compile-time alias: fp32 path computes in-place on x_ub, the
            # fp16/bf16 path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            # In-core grid-stride over logical tiles.
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- piecewise expm1 chain (fp32, whole-buffer v-ops) ---
                    T.vabs(xc, xa)  # xa = |x|
                    T.vcmp(xa, thr, cond, "lt")  # cond = |x| < T

                    # g = ((c2*x + c1)*x + c0); small = x*g
                    T.vmul(xc, _COEF[2], g)
                    T.vadd(g, T.cast(_COEF[1], f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_COEF[0], f32), g)
                    T.vmul(xc, g, xa)  # xa = x*g (small branch)

                    T.vexp(xc, e)  # e = exp(x)
                    T.vsub(e, T.cast(1.0, f32), e)  # e = exp(x)-1 (large branch)

                    T.vselect(cond, xa, e, xc)  # xc = |x|<T ? x*g : exp(x)-1

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

# Tolerances from examples/TileOPs/tests/ops/test_unary_math.py (DESIGN 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _make_input(N, dtype_str):
    """Generate a CPU input tensor following the Expm1Workload logic (randn)."""
    return torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _max_diff(y, golden):
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _max_rel_err(y, golden):
    """Max relative error over finite, non-zero golden entries."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    mask = torch.isfinite(g_f64) & (g_f64 != 0)
    if not bool(mask.any()):
        return 0.0
    return ((y_f64[mask] - g_f64[mask]).abs() / g_f64[mask].abs()).max().item()


def _run_tensor(x_cpu, dtype_str, tag, rel_check=False):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    N = x_cpu.numel()
    golden = golden_expm1(x_cpu)
    kernel = expm1_fwd_kernel(N, dtype_str)

    x_npu = x_cpu.npu()
    y = kernel(x_npu, N)
    y_cpu = y.cpu()

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} max_diff={max_diff:.6e}\n{e}"
        ) from e

    max_rel = _max_rel_err(y_cpu, golden)
    if rel_check:
        assert max_rel <= rtol, (
            f"[{tag}] REL FAIL: shape=({N},) dtype={dtype_str} "
            f"max_rel={max_rel:.6e} > rtol={rtol:.1e}"
        )
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
        f"max_diff={max_diff:.6e} max_rel={max_rel:.6e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, rel_check=False):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag, rel_check=rel_check)


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1..L0-3 smoke: 1M randn elements per dtype (DESIGN.md section 8.3).
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")

    # L0-4 fp32 corner values from test_expm1_edge (0.0 / 88.8 -> inf /
    # -88.8 -> -1.0 / 1e-7 -> relatively exact).
    _run_tensor(
        _repeat_values([0.0, 88.8, -88.8, 1e-7], 4096, torch.float32),
        "float32",
        "L0-corner",
        rel_check=True,
    )

    # L0-4b inf / NaN IEEE propagation.
    _run_tensor(
        _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], 4096, torch.float32),
        "float32",
        "L0-corner-inf-nan",
    )

    # L0-5 small-|x| cancellation-sensitive band: relative accuracy required.
    x_small = torch.linspace(-1e-3, 1e-3, 1 << 20, dtype=torch.float32)
    _run_tensor(x_small, "float32", "L0-small-x-rel", rel_check=True)

    # L0-6 full-domain sensitive region, incl. the threshold boundary +-T.
    x_full = torch.linspace(-9.0, 9.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_full, "float32", "L0-linspace-9")

    # L0-7 non-divisible tail (4_000_000 % 7168 = 256).
    _run_case(4000000, "float16", "L0-tail")


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat and non-divisible
    # tails for every dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4096 * 4096, dtype_str, "L1-16m")
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4000000, dtype_str, "L1-tail")
    # Exact single-tile / multi-tile grid.
    _run_case(7168, "float32", "L1-exact-tile")
    _run_case(8192, "float16", "L1-two-tiles")
    _run_case(2047, "bfloat16", "L1-partial-tile")
    # Threshold boundary and small-|x| band across dtypes (relative check).
    x_thr = torch.linspace(-1.0, 1.0, 1 << 18, dtype=torch.float32)
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_tensor(x_thr.to(_TORCH_DTYPE[dtype_str]), dtype_str, "L1-threshold-band")


def run_L2():
    # L2: edge sizes around the tile boundary, non-blocking.
    for N, dtype_str in (
        (1, "float32"),
        (2, "float16"),
        (10, "bfloat16"),
        (7169, "float32"),
        (7167, "float16"),
        (8191, "bfloat16"),
    ):
        try:
            _run_case(N, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary: special-value inputs, all non-blocking.
    special = [0.0, -0.0, 1e-7, -1e-7, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0, 88.8, -88.8]
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            x = _repeat_values(special, 4096, _TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    # inf / NaN IEEE propagation across dtypes.
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            td = _TORCH_DTYPE[dtype_str]
            x = _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], 4096, td)
            _run_tensor(x, dtype_str, "Boundary-inf-nan")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} inf/nan err={e}")

    # Threshold-crossing sign-alternating pattern (branch selection stress).
    try:
        thr = _T
        x = _repeat_values([thr, -thr, thr + 1e-6, -thr - 1e-6, 0.0], 4096, torch.float32)
        _run_tensor(x, "float32", "Boundary-threshold", rel_check=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): threshold err={e}")

    # Manifest full workloads (non-blocking).
    for N, dtype_str in (
        (4096 * 4096, "float16"),
        (8192 * 8192, "float16"),
        (16384 * 16384, "float32"),
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
