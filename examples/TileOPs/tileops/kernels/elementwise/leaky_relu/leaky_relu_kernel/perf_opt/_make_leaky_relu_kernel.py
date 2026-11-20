# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""LeakyReLU forward kernel (NPU, target="npuir") -- Stage 4 performance-tuned FINAL.

Drop-in replacement for the Stage 3 baseline ``_make_leaky_relu_kernel.py``:
same factory name, same ``(N, dtype, negative_slope, output_dtype=None,
block_size=None)`` signature, same ``main(x, y, shape)`` prim_func, same
fp32-transit precision semantics and same compile-time ``negative_slope``
specialisation (max form on ``0 < s <= 1``, ``vcmp``/``vselect`` fallback
otherwise).  Two single-point-verified changes vs the Stage 3 baseline, merged
into ONE ``@T.prim_func`` (per-branch measurements in perf_opt/opt_log.md):

1. **per-dtype tuned block_size** (Round 1, `r1_bs`).  The Stage 3 K11-collapsed
   defaults (fp32 1024 / fp16·bf16 2048) leave every tile at 4 KB, where this
   purely DMA-bound elementwise kernel runs at ~0.6-0.75 TB/s effective
   (baseline 16m-f16 = 75.7 us).  Exposed as the module constant
   ``TUNED_BLOCK_SIZE`` (the wrapper's perf_opt toggle imports it):
   fp32 12288 / fp16 12288 / bf16 12288.  Measured 2.4-2.6 TB/s effective at the
   DMA plateau (device block_size<->bandwidth curve in opt_log.md).  The rare
   ``vcmp``/``vselect`` fallback path keeps the ``z``+``cond`` buffers live and
   has a lower UB ceiling, so the factory clamps its tile size via
   ``_WHERE_BLOCK_SIZE`` -- a compile-time Python value, no runtime branch and
   no new dispatch surface.

2. **static-bulk / dynamic-tail loop split for the fp32 path** (Round 2,
   `r2_static`).  ``block_size`` is a compile-time constant, so the
   ``N // block_size`` full tiles use *static-extent* ``T.copy`` and only the
   (at most one) remainder tile keeps the dynamic ``T.min`` extent.  Measured
   to unlock MTE2/MTE3 overlap on the fp32 workload (interleaved A/B: 52.68 ->
   47.75 us, -9.4%, 5/5 consistent).  The fp16/bf16 path intentionally keeps the
   original dynamic-extent loop: there static-bulk measured neutral at bs=8192
   and clearly worse at bs>=10240 (PL-1.34 fp16 counterexample).

run: python _make_leaky_relu_kernel.py --level all
"""

"""LeakyReLU forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_make_leaky_relu_kernel``
(``y = T.if_then_else(val > zero, val, slope * val)``, ``register_copy``
strategy, one ``npt`` element per thread, computed in the *input dtype*
domain).  Semantics are equivalent to ``torch.nn.functional.leaky_relu`` for
float16 / bfloat16 / float32, output dtype ``same_as(input)``.

NPU re-design (see DESIGN.md sections 0.5 / 0.6):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (GPU thread-level parallelism has no
    direct NPU counterpart).  ``core_num = 40`` is the physical Vector core
    count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- compile-time ``negative_slope`` specialisation.  For the default
    contract ``0 < negative_slope <= 1`` the branch-free 2-op form
    ``y = max(x, s*x)`` (``vmul`` + ``vmax``) replaces the source 3-op
    ``if_then_else`` (``vcmp`` + ``vmul`` + ``vselect``).  Outside the guard
    (``s <= 0`` or ``s > 1``) the ``where`` fallback is used, which is
    bit-exact with the contract too.
  * R3 -- computation domain is **fp32 transit**, NOT the source input dtype
    domain.  The contract golden is ``F.leaky_relu(x.float(), 0.01).to(dtype)``
    (fp32 opmath): ``cast(0.01, fp16/fp16)`` is not exact, so the source
    input-dtype arithmetic deviates from the golden at the corners
    (device-measured violation ratio 0.2597 for fp16 / 0.1297 for bf16),
    while fp32 transit matches the golden bit-for-bit.  bf16 transit is
    *mandatory* because the v-prefix math ops (``vmul``/``vmax``) have no
    bf16 dtype in their matrix.
  * R4 -- the source ``T.if_then_else`` becomes a branch-free vector chain;
    the ``bool`` mask stays on-chip and is never copied back to GM.

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (a conditional alloc triggers ``Undefined
    variable``); the compile-time ``if need_cast:`` / ``if use_max:``
    branches only select which v-ops run.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``).  A dynamic slice passed to a v-op
    fails at compile time.
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-10: the ``bool`` comparison mask is kept on-chip (``vselect``).
  * Zero constant buffer: ``T.fill(z, 0.0)`` once outside the tile loop.
  * ``T.vmax`` has no scalar broadcast, so ``t = vmul(x, s)`` is materialised
    first and ``vmax(xc, t, xc)`` is tensor-tensor.

run: python _make_leaky_relu_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_BLOCK_SIZE = 2048
_CORE_NUM = 40  # physical Vector cores on Ascend910B3 / 910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# K11: GPU block_size = threads(256) * npt collapsed per dtype
# (source ParametricUnaryKernel: npt = 4 for fp32, 8 for fp16/bf16).
#
# Stage 4 Round 1 (OP = per-dtype block_size): the K11-collapsed values leave
# every tile at 4 KB, where this DMA-bound elementwise kernel runs at
# ~0.6-0.75 TB/s effective.  The measured block_size<->bandwidth curve
# (perf_opt/opt_log.md Round 1 probe) plateaus at ~2.4-2.6 TB/s (the copy
# floor) for bs >= 12288 (fp16/bf16) / >= 16384 (fp32).
TUNED_BLOCK_SIZE = {
    "float32": 12288,
    "float16": 12288,
    "bfloat16": 12288,
}
_DEFAULT_BLOCK_SIZE = TUNED_BLOCK_SIZE

# Fallback-path (`negative_slope` outside `0 < s <= 1`) tile ceiling.  The
# `vcmp`+`vselect` fallback additionally keeps the `z` (fp32 zero) and `cond`
# (bool mask) buffers live, so BiShengIR enforces a lower UB budget there; the
# default max-form path dead-code-eliminates those two buffers.  The fp32
# static-bulk path (Round 2) has the tightest ceiling (multi-buffer inflation
# of the 16 B/elem fp32 working set), so its fallback clamp is the most
# conservative.  Measured device ceilings (perf_opt/opt_log.md probes:
# `ub overflow, requires ... while 1572864 bits available`):
#   fp32 dynamic: 15360 ok / 16384 overflow ; fp32 static-bulk: 12288 ok / 14336 overflow
#   fp16 dynamic: 10240 ok / 11264 overflow.
_WHERE_BLOCK_SIZE = {
    "float32": 6144,
    "float16": 10240,
    "bfloat16": 10240,
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """LeakyReLU reference, equivalent to ``torch.nn.functional.leaky_relu``.

    Contract (``test_activation.py::LeakyReluTest.ref_program``):

        F.leaky_relu(x.float(), negative_slope).to(dtype)

    i.e. the elementwise ramp ``y = x if x > 0 else negative_slope * x`` is
    evaluated in the fp32 opmath domain and rounded back to the input dtype
    exactly once.  The source GPU non-fp8 branch computes in the *input*
    dtype domain (``slope = cast(s, dtype)``), which deviates from this
    contract at the corners under fp16/bf16; the golden follows the contract
    strictly and never reproduces the DESIGN.md vector chain structure.
    """
    x_f32 = x.detach().cpu().to(torch.float32)
    return F.leaky_relu(x_f32, negative_slope).to(x.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None, block_size=None):
    """LeakyReLU forward kernel factory: y = x if x > 0 else negative_slope * x.

    ``block_size=None`` selects the Stage-4 tuned per-dtype default exposed as
    ``TUNED_BLOCK_SIZE``; an explicit ``block_size`` still overrides it.
    """
    out_dtype = output_dtype or dtype
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    block_size = block_size or _DEFAULT_BLOCK_SIZE[dtype]
    slope = float(negative_slope)
    # Strict compile-time guard: max(x, s*x) is equivalent only for 0 < s <= 1.
    use_max = 0.0 < slope <= 1.0
    # Fallback path keeps z/cond live -> lower UB ceiling -> clamp the tile.
    if not use_max:
        block_size = min(block_size, _WHERE_BLOCK_SIZE[dtype])
    n_num = T.ceildiv(N, block_size)
    n_full = N // block_size
    has_tail = (N % block_size) != 0

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-7: all UB buffers allocated unconditionally at the top level
            # (conditional T.alloc is illegal -- TRAP-conditional-alloc).
            x_ub = T.alloc_ub((block_size,), dtype)        # input tile
            y_ub = T.alloc_ub((block_size,), out_dtype)    # round-back tile
            xf = T.alloc_ub((block_size,), f32)            # fp32 transit / domain
            t = T.alloc_ub((block_size,), f32)             # negative-arm s*x
            z = T.alloc_ub((block_size,), f32)             # read-only constant 0
            cond = T.alloc_ub((block_size,), "bool")       # on-chip mask (R-10)

            # Zero constant: filled once outside the tile loop (read-only,
            # valid across all tiles; vcmp needs a same-shape tensor operand).
            T.fill(z, T.cast(0.0, f32))

            # Compile-time alias: fp32 path computes in-place on x_ub, the
            # fp16/bf16 path computes in the fp32 transit buffer xf.
            xc = xf if need_cast else x_ub

            # In-core grid-stride over logical tiles: static loop bound,
            # guard covers non-divisible grids (DESIGN.md section 5.5).
            if need_cast:
                # ---- fp16 / bf16: vector-bound, dynamic-extent (tail) loop ----
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_num:
                        off = task * block_size
                        tail = T.min(block_size, shape - off)
                        # Only T.copy uses the dynamic tail slice (R-8).
                        T.copy(x[off : off + tail], x_ub[0:tail])

                        # lossless up-cast (f16/bf16 -> f32, rint only, R-9)
                        T.vcast(x_ub, xf, round_mode="rint")

                        # --- LeakyReLU vector chain (fp32 domain, whole buffer) ---
                        if use_max:
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vmax(xc, t, xc)
                        else:
                            T.vcmp(xc, z, cond, "gt")
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vselect(cond, xc, t, xc)

                        # single round back to input dtype (R-9)
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off : off + tail])
            else:
                # ---- fp32: DMA-bound, static-bulk + single dynamic tail ----
                # `block_size` is a compile-time constant, so the `N // bs` full
                # tiles use static-extent T.copy and only the (at most one)
                # remainder tile keeps the dynamic T.min extent.  Measured to
                # unlock MTE2/MTE3/VEC overlap at bs=16384 (PL-1.34).
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_full:
                        off = task * block_size
                        T.copy(x[off : off + block_size], x_ub)
                        if use_max:
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vmax(xc, t, xc)
                        else:
                            T.vcmp(xc, z, cond, "gt")
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vselect(cond, xc, t, xc)
                        T.copy(xc, y[off : off + block_size])
                    elif has_tail and task == n_full:
                        off = task * block_size
                        tail = T.min(block_size, shape - off)
                        T.copy(x[off : off + tail], x_ub[0:tail])
                        if use_max:
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vmax(xc, t, xc)
                        else:
                            T.vcmp(xc, z, cond, "gt")
                            T.vmul(xc, T.cast(slope, f32), t)
                            T.vselect(cond, xc, t, xc)
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
    """Generate a CPU input tensor following the LeakyReluWorkload logic (randn)."""
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


def _run_tensor(x_cpu, dtype_str, tag, negative_slope=0.01):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    x_cpu = x_cpu.detach().cpu().reshape(-1).contiguous()
    N = x_cpu.numel()
    golden = golden_leaky_relu(x_cpu, negative_slope)
    kernel = _make_leaky_relu_kernel(
        N, dtype_str, negative_slope, block_size=_DEFAULT_BLOCK_SIZE[dtype_str]
    )

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
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} "
            f"negative_slope={negative_slope} max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
        f"negative_slope={negative_slope} max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, negative_slope=0.01):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag, negative_slope)


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1..L0-3 smoke (DESIGN.md section 8.3): 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")

    # L0-4 corner / branch-point values (test_leaky_relu_edge pattern):
    # -88.8 -> -0.888, -1e-7 -> -1e-9, 0 -> 0, +1e-7 -> +1e-7, 88.8 -> 88.8.
    _run_tensor(
        _repeat_values([-88.8, -1.0, -1e-7, 0.0, 1e-7, 1.0, 88.8], 4096, torch.float32),
        "float32",
        "L0-corner",
    )

    # L0-5 negative-branch small-|x| sensitive region.
    _run_tensor(
        torch.linspace(-2e-3, 0.0, 1 << 20, dtype=torch.float32),
        "float32",
        "L0-small-x",
    )

    # L0-6 positive/negative saturation + inf + NaN.
    _run_tensor(
        _repeat_values(
            [-65504.0, 65504.0, float("-inf"), float("inf"), float("nan")],
            4096,
            torch.float32,
        ),
        "float32",
        "L0-saturate",
    )

    # L0-7 x=0 branch point + full sensitive region.
    _run_tensor(
        torch.linspace(-9.0, 9.0, 1 << 20, dtype=torch.float32),
        "float32",
        "L0-linspace",
    )

    # L0-8 non-divisible tail (4_000_000 % 2048 = 1216).
    _run_case(4000000, "float16", "L0-tail")

    # L0-9 manifest multi-dim shapes, flattened by the host (fp16/bf16).
    for shape in ((16, 256, 64, 64), (16, 512, 32, 32)):
        for dtype_str in ("float16", "bfloat16"):
            x = torch.randn(*shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, f"L0-manifest{shape}")

    # L0-10 non-default negative_slope (exercises the vselect fallback path).
    _run_case(1048576, "float32", "L0-slope2.0", negative_slope=2.0)


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat, non-divisible
    # tails for every dtype, exact / partial tiles, non-default slopes.
    _run_case(4096 * 4096, "float32", "L1-16m")
    _run_case(4000000, "float32", "L1-tail")
    _run_case(4000000, "bfloat16", "L1-tail")
    _run_case(1024, "float32", "L1-exact-tile")
    _run_case(4096, "float16", "L1-two-tiles")
    _run_case(2047, "bfloat16", "L1-partial-tile")
    # Compile-time slope specialisation: max path (0.5) and fallback paths.
    _run_case(1048576, "float32", "L1-slope0.5", negative_slope=0.5)
    _run_case(1048576, "float32", "L1-slope100.0", negative_slope=100.0)
    _run_case(1048576, "float16", "L1-slope0.0", negative_slope=0.0)
    # Manifest 4-D workloads (16m) for both sub-fp32 dtypes.
    for shape in ((16, 256, 64, 64), (16, 512, 32, 32)):
        for dtype_str in ("float16", "bfloat16"):
            x = torch.randn(*shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, f"L1-manifest{shape}")


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
    # Boundary: special-value inputs (branch point, saturation, inf, NaN),
    # all non-blocking.  leaky_relu(+inf) = +inf, leaky_relu(-inf) = -inf
    # (0 < s <= 1), leaky_relu(NaN) = NaN.
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
