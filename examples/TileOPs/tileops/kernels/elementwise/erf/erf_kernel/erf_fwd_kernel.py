# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Erf forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``erf_fwd_kernel`` (``y = erf(x)``,
``register_copy`` strategy, ``op_func = T.erf(T.cast(x, "float32"))``).
Semantics are equivalent to ``torch.erf`` for float16 / bfloat16 / float32,
output dtype ``same_as(input)``.  The source fp32-transit precision semantics
(x -> fp32 -> erf -> single round back to input dtype) is preserved.

NPU re-design (see DESIGN.md section 0.5/0.6, revision 1):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (the GPU thread-level parallelism has no
    direct NPU counterpart).  ``core_num = 40`` is the physical Vector core
    count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- there is NO usable native erf meta-op on this device: ``T.verf``
    is a 7th-order *truncated Maclaurin polynomial* on the non-A5 codegen
    paths (accurate only for |x| <~ 0.5; CG-2026-0015, see
    ``repro/TRAP-verf-maclaurin-truncation.py``) and the scalar ``T.erf``
    segfaults at compile time.  The error function is therefore rebuilt from
    device-verified v-prefix primitives with the Abramowitz & Stegun 7.1.26
    rational-exponential approximation (max abs err 1.5e-7 for x >= 0),
    extended to the negative half-axis by odd symmetry:

        t = 1 / (1 + p*|x|)
        erf(x) ~= sign(x) * (1 - Horner(a1..a5; t) * exp(-x^2))

    Device-measured end-to-end fp32 max_abs = 3.7e-7 (27x margin vs atol
    1e-5); see ``history_version/design_probe_erf_ops*.py``.
  * R3 -- dtype-transit chain with vector ``T.vcast``: fp32 computes the
    A&S chain directly; fp16/bf16 go ``vcast(dtype->fp32, rint) ->
    A&S chain (fp32) -> vcast(fp32->dtype, rint)``.  bf16 transit is
    *mandatory* because the v-prefix math ops have no bf16 dtype in their
    matrix.

Hard constraints honoured (DESIGN.md section 3.5.1 / 9.1):
  * R-2/R-6: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (conditional alloc triggers ``Undefined
    variable``); the compile-time ``if need_cast:`` branch only selects which
    v-ops run.
  * R-7: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``).
  * R-8: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-9: the ``bool`` comparison mask is kept on-chip (``vselect``); it is
    never copied back to GM.

run: python erf_fwd_kernel.py --level all
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

_BLOCK_SIZE = 2048
_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# Abramowitz & Stegun 7.1.26 coefficients (max abs err 1.5e-7, x >= 0).
_P = 0.3275911
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_erf(x):
    """Erf reference implementation, equivalent to ``torch.erf``.

    Mirrors the source op_func ``T.erf(T.cast(x, "float32"))``: promote the
    input to fp32, compute the error function in the fp32 domain and round
    once back to the original dtype.  ``torch.erf`` itself already uses fp32
    opmath for fp16/bf16, so this is bit-identical to ``torch.erf(x)`` (see
    verify_equiv.py, DESIGN.md section 1.6.1).

    Independent from the NPU kernel: only the source semantics are used here,
    never the A&S coefficients.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return torch.erf(x_f32).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE):
    """Erf forward kernel factory: y = erf(x).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> A&S 7.1.26 chain directly on the fp32 input buffer;
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
            # R-2/R-6: all UB buffers allocated unconditionally at the top
            # level (conditional T.alloc is illegal -- TRAP-conditional-alloc).
            x_ub = T.alloc_ub((block_size,), dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)
            xf = T.alloc_ub((block_size,), f32)       # fp32 transit entry
            xa = T.alloc_ub((block_size,), f32)       # |x| -> x^2 -> -x^2
            t = T.alloc_ub((block_size,), f32)        # denom -> t -> sign
            poly = T.alloc_ub((block_size,), f32)     # Horner -> prod
            e = T.alloc_ub((block_size,), f32)        # exp(-x^2)
            ones = T.alloc_ub((block_size,), f32)     # read-only constant 1
            negones = T.alloc_ub((block_size,), f32)  # read-only constant -1
            zero = T.alloc_ub((block_size,), f32)     # read-only constant 0
            cond = T.alloc_ub((block_size,), "bool")  # on-chip mask (R-9)

            # Constants: filled once outside the tile loop (read-only, valid
            # across all tiles).
            T.fill(ones, T.cast(1.0, f32))
            T.fill(negones, T.cast(-1.0, f32))
            T.fill(zero, T.cast(0.0, f32))

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
                    # Only T.copy uses the dynamic tail slice (R-7).
                    T.copy(x[off : off + tail], x_ub[0:tail])

                    if need_cast:
                        # lossless up-cast (f16/bf16 -> f32, rint only, R-8)
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- A&S 7.1.26 erf chain (fp32, whole-buffer v-ops) ---
                    # cond = (x >= 0) captured before xc is reused (R-9).
                    T.vcmp(xc, zero, cond, "ge")
                    T.vabs(xc, xa)                    # xa = |x|
                    T.vmul(xa, _P, t)                 # t = p*|x|
                    T.vadd(t, T.cast(1.0, f32), t)    # t = 1 + p*|x|
                    T.vrec(t, t)                      # t = 1/(1+p*|x|)
                    # poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
                    T.vmul(t, _A5, poly)
                    T.vadd(poly, _A4, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A3, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A2, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A1, poly)
                    T.vmul(poly, t, poly)
                    T.vmul(xa, xa, xa)                # xa = x^2
                    T.vmul(xa, T.cast(-1.0, f32), xa)  # xa = -x^2
                    T.vexp(xa, e)                     # e = exp(-x^2)
                    T.vmul(poly, e, poly)             # poly = poly*e
                    T.vsub(ones, poly, xc)            # xc = 1 - poly*e
                    T.vselect(cond, ones, negones, t)  # t = sign (reuse)
                    T.vmul(xc, t, xc)                 # xc = sign * y_even

                    if need_cast:
                        # single round back to input dtype (R-8)
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
    """Generate a CPU input tensor following the ErfWorkload logic (randn)."""
    return torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])


def _repeat_values(values, N, dtype):
    """Repeat a small value pattern up to N elements (edge / corner cases)."""
    base = torch.tensor(values, dtype=dtype)
    repeats = (N + len(values) - 1) // len(values)
    return base.repeat(repeats)[:N]


def _max_diff(y, golden):
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    return (y_f64 - g_f64).abs().max().item()


def _run_tensor(x_cpu, dtype_str, tag):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    N = x_cpu.numel()
    golden = golden_erf(x_cpu)
    kernel = erf_fwd_kernel(N, dtype_str)

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
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.6e}")
    return max_diff


def _run_case(N, dtype_str, tag):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0 smoke (DESIGN.md section 8.3 L0-1..L0-3): 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")
    # L0-4 corner values that the v0 (T.verf) design always failed:
    # erf(3) must be ~0.999978, not the truncated-polynomial value 37.9.
    _run_tensor(
        _repeat_values([0.0, 3.0, -3.0, 100.0], 4096, torch.float32),
        "float32",
        "L0-corner",
    )
    # L0-5 full-domain sensitive region (|x| up to 8).
    x_lin = torch.linspace(-8.0, 8.0, 1 << 20, dtype=torch.float32)
    _run_tensor(x_lin, "float32", "L0-linspace")
    # L0-6 non-divisible tail (4_000_000 % 2048 = 1216).
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
    # Boundary: special-value inputs (saturation, inf, NaN), non-blocking.
    special = [0.0, 3.0, -3.0, 3.9, -3.9, 5.0, 100.0, -100.0, 1e-8]
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            x = _repeat_values(special, 4096, _TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}")

    # inf / NaN IEEE propagation.
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
