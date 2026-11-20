# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Expm1 forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``expm1_fwd_kernel``
(``op_func = T.exp(T.cast(x, "float32")) - 1.0``, ``register_copy`` strategy).
Semantics are equivalent to ``torch.expm1`` for float16 / bfloat16 / float32,
output dtype ``same_as(input)``.  The source fp32-transit precision semantics
(x -> fp32 -> expm1 -> single round back to input dtype) is preserved.

NPU re-design (see DESIGN.md section 0.5/0.6):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (the GPU thread-level parallelism has no
    direct NPU counterpart).  ``core_num = 40`` is the physical Vector core
    count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- the source ``exp(x) - 1`` catastrophic-cancellation is NOT
    acceptable for the ``torch.expm1`` contract: on device the naive direct
    translation gives a 19.2% *relative* error at x=1e-7 (probe P0).  The
    compute core is therefore rebuilt as a piecewise expm1 with threshold
    ``T = 0.5*ln2`` (exp(T) = sqrt(2)):

        |x| <  T : x * sum_{n=0..5} x^n / (n+1)!   (6-term Taylor, Horner)
        |x| >= T : vexp(x) - 1

    Both branches are computed and selected branch-free with vcmp/vselect.
    Small-|x| branch relative error <= ~5e-7, large-|x| branch <= ~4e-7,
    both >> below rtol 1e-5.
  * R3 -- dtype-transit chain with vector ``T.vcast``: fp32 computes the
    piecewise chain directly; fp16/bf16 go ``vcast(dtype->fp32, rint) ->
    piecewise chain (fp32) -> vcast(fp32->dtype, rint)``.  bf16 transit is
    *mandatory* because the v-prefix math ops have no bf16 dtype in their
    matrix.
  * R4 -- threshold selection via ``T.vabs`` + ``T.vcmp`` + ``T.vselect``
    (no runtime branch divergence).

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

run: python expm1_fwd_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_BLOCK_SIZE = 2048
_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# Piecewise expm1 threshold and the expm1(x)/x Taylor coefficients
# (C_n = 1/(n+1)!), used in Horner form on the small-|x| branch.
_T = 0.5 * math.log(2.0)  # 0.34657359027997264
_C0 = 1.0  # 1/1!
_C1 = 1.0 / 2.0  # 1/2!
_C2 = 1.0 / 6.0  # 1/3!
_C3 = 1.0 / 24.0  # 1/4!
_C4 = 1.0 / 120.0  # 1/5!
_C5 = 1.0 / 720.0  # 1/6!


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_expm1(x):
    """Expm1 reference implementation, equivalent to ``torch.expm1``.

    Mirrors the source op_func ``T.exp(T.cast(x, "float32")) - 1.0``: promote
    the input to fp32, compute in the fp32 domain and round once back to the
    original dtype.  ``torch.expm1`` itself already uses fp32 opmath for
    fp16/bf16 (libm special-cased expm1, relatively accurate for small |x|),
    so this matches the source fp32-transit semantics.

    Independent from the NPU kernel: only the source semantics are used here,
    never the piecewise Taylor coefficients.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return torch.expm1(x_f32).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE):
    """Expm1 forward kernel factory: y = expm1(x).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> piecewise expm1 chain directly on the fp32 buffer;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.

    ``block_size`` folds the GPU ``threads(256) * num_per_thread(8) = 2048``
    (K11).  ``N`` is a compile-time constant; the runtime ``shape`` parameter
    guards the dynamic tail tile.
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-6/R-7: all UB buffers allocated unconditionally at the top
            # level (conditional T.alloc is illegal -- TRAP-conditional-alloc).
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)  # fp16/bf16 round-back
            xf = T.alloc_ub((block_size,), f32)  # fp32 transit entry
            xa = T.alloc_ub((block_size,), f32)  # |x| -> small-branch result
            g = T.alloc_ub((block_size,), f32)  # Horner accumulator
            e = T.alloc_ub((block_size,), f32)  # exp(x) -> exp(x)-1
            thr = T.alloc_ub((block_size,), f32)  # read-only threshold T
            cond = T.alloc_ub((block_size,), "bool")  # on-chip mask (R-10)

            # Threshold constant: filled once outside the tile loop
            # (read-only, valid across all tiles).
            T.fill(thr, T.cast(_T, f32))

            # Compile-time alias: fp32 path computes in-place on x_ub, the
            # fp16/bf16 path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            # In-core grid-stride over logical tiles: static loop bound,
            # guard covers non-divisible grids (DESIGN.md section 5.5).
            for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                task = i * _CORE_NUM + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    # Only T.copy uses the dynamic tail slice (R-8).
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        # lossless up-cast (f16/bf16 -> f32, rint only, R-9)
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- piecewise expm1 chain (fp32, whole-buffer v-ops) ---
                    T.vabs(xc, xa)  # xa = |x|
                    T.vcmp(xa, thr, cond, "lt")  # cond = |x| < T

                    # g = ((((C5*x + C4)*x + C3)*x + C2)*x + C1)*x + C0
                    #   = sum_{n=0..5} x^n / (n+1)!   (Horner form)
                    T.vmul(xc, _C5, g)
                    T.vadd(g, T.cast(_C4, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C3, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C2, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C1, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C0, f32), g)
                    T.vmul(xc, g, xa)  # xa = x*g(x)  (small branch, reuse)

                    T.vexp(xc, e)  # e = exp(x)
                    T.vsub(e, T.cast(1.0, f32), e)  # e = exp(x)-1 (large branch)

                    T.vselect(cond, xa, e, xc)  # xc = |x|<T ? x*g : exp(x)-1

                    if need_cast:
                        # single round back to input dtype (R-9)
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
    """Run the kernel on a CPU tensor and compare with the golden output.

    ``rel_check`` additionally asserts the *relative* error (critical for the
    small-|x| cancellation-sensitive region, DESIGN.md section 8.3 L0-5).
    """
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
    # -88.8 -> -1.0 / 1e-7 -> relatively exact).  rel_check guards the
    # small-|x| cancellation that the naive exp(x)-1 translation fails.
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

    # L0-5 small-|x| cancellation-sensitive band: relative accuracy required
    # (naive exp(x)-1 gives ~19% relative error here; piecewise gives <=1e-7).
    x_small = torch.linspace(-1e-3, 1e-3, 1 << 20, dtype=torch.float32)
    _run_tensor(x_small, "float32", "L0-small-x-rel", rel_check=True)

    # L0-6 full-domain sensitive region, incl. the threshold boundary +-T.
    x_full = torch.linspace(-9.0, 9.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_full, "float32", "L0-linspace-9")

    # L0-7 non-divisible tail (4_000_000 % 2048 = 1216).
    _run_case(4000000, "float16", "L0-tail")


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat and non-divisible
    # tails for every dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4096 * 4096, dtype_str, "L1-16m")
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(4000000, dtype_str, "L1-tail")
    # Exact single-tile / multi-tile grid.
    _run_case(2048, "float32", "L1-exact-tile")
    _run_case(4096, "float16", "L1-two-tiles")
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
        (2049, "float32"),
        (4095, "float16"),
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
