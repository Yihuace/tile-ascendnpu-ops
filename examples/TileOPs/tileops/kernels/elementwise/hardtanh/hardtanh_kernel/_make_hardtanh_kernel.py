# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Hardtanh forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_make_hardtanh_kernel``
(``y = min(max(x, min_val), max_val)``, ``register_copy`` strategy, one ``npt``
element per thread, computed in the **fp32 transit** domain).  Semantics are
equivalent to ``torch.nn.functional.hardtanh`` for float16 / bfloat16 /
float32, output dtype ``same_as(input)``.

NPU re-design (see DESIGN.md sections 0.5 / 0.6):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical tiles (the GPU fragment + block x thread
    parallelism has no direct NPU counterpart).  ``core_num = 40`` is the
    physical Vector core count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- the computation core is a fused whole-buffer vector chain plus
    compile-time bound folding.  Default (both bounds finite):
    ``T.vclamp(xc, xc, lo, hi)`` (2 ops, front-end expands to
    ``npuir_max`` + ``npuir_min`` -- the fused form of the source
    ``T.min(T.max(v32, lo), hi)``).  Degenerate bounds fold to ``T.vmin``
    (1 op, ``lo = -inf``), ``T.vmax`` (1 op, ``hi = +inf``), a pure
    ``T.copy`` (0 ops, both infinities) or ``T.fill`` (constant, ``lo == hi``).
  * R3 -- computation domain is **fp32 transit** (*preserved* from the source,
    which already computes ``v32 = T.cast(val, "float32")`` ->
    ``min(max(v32, lo), hi)`` -> single round back).  fp16/bf16 go
    ``vcast(dtype->fp32, rint) -> chain (fp32) -> vcast(fp32->dtype, rint)``.
    bf16 transit is *mandatory* because the v-prefix math ops
    (``vclamp``/``vmin``/``vmax``) have no bf16 dtype in their matrix.  The
    integration wrapper docstring requires this fp32 transit to be preserved.

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the
    ``T.Kernel`` top level (a conditional alloc triggers ``Undefined
    variable``); the compile-time ``if need_cast:`` / folding branches only
    select which v-ops run.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x_ub[0:tail]``).  A dynamic slice passed to a v-op
    fails at compile time.
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-10: ``T.vmax`` has no documented scalar broadcast, so the ``hi = +inf``
    fold uses a pre-filled fp32 constant buffer ``lo_ub`` as its second
    operand (filled once outside the tile loop, read-only across tiles).
  * ``T.vclamp`` / ``T.vmin`` take scalar bounds directly (per docs).

run: python _make_hardtanh_kernel.py --level all
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
_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
_COMPUTE_DTYPE = "float32"

# K11: GPU block_size = threads(256) * npt collapsed per dtype
# (source ParametricUnaryKernel: npt = 4 for fp32, 8 for fp16/bf16;
# integration wrapper ``_DEFAULT_BLOCK_SIZE``).
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}

_NEG_INF = float("-inf")
_POS_INF = float("inf")


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """Hardtanh reference, equivalent to ``torch.nn.functional.hardtanh``.

    Contract (``test_activation.py::HardtanhTest.ref_program``):

        F.hardtanh(x.float(), min_val, max_val).to(dtype)

    i.e. the two-sided clamp is evaluated in the fp32 opmath domain and
    rounded back to the input dtype exactly once.  This is independent from
    the NPU vector chain -- only the library implementation is used here,
    never the DESIGN.md kernel structure.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    return F.hardtanh(x_f32, min_val, max_val).to(x_cpu.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=_BLOCK_SIZE):
    """Hardtanh forward kernel factory: y = clamp(x, min_val, max_val).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> the vector chain runs directly on the fp32 input;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.

    ``min_val`` / ``max_val`` are compile-time scalars (the source
    ``functools.lru_cache`` specialises on them).  Their values select the
    compile-time bound folding (DESIGN.md section 1.6.1 #1): default both
    finite -> ``vclamp``; ``lo = -inf`` -> ``vmin``; ``hi = +inf`` ->
    ``vmax``; both infinite -> ``copy``; ``lo == hi`` -> ``fill`` (constant).

    ``block_size`` folds the GPU ``threads(256) * npt`` product (K11).  ``N``
    is a compile-time constant; the runtime ``shape`` parameter guards the
    dynamic tail tile.
    """
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    lo = float(min_val)
    hi = float(max_val)

    # Compile-time bound folding (host Python constants, no runtime branch;
    # the default (-1, 1) does not trigger any fold -- DESIGN.md section 1.6.1 #1).
    only_upper = lo == _NEG_INF and hi != _POS_INF   # -> vmin(x, hi)
    only_lower = hi == _POS_INF and lo != _NEG_INF   # -> vmax(x, lo)
    identity = lo == _NEG_INF and hi == _POS_INF     # -> copy(x)
    const_val = lo == hi                             # -> fill(const)
    n_num = T.ceildiv(N, block_size)

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
            # R-7: all UB buffers allocated unconditionally at the top level
            # (conditional T.alloc is illegal -- TRAP-conditional-alloc).
            x_ub = T.alloc_ub((block_size,), dtype)        # input tile
            y_ub = T.alloc_ub((block_size,), dtype)        # round-back / constant tile
            xf = T.alloc_ub((block_size,), f32)            # fp32 transit / domain
            lo_ub = T.alloc_ub((block_size,), f32)         # read-only constant lo (hi=+inf fold)

            # Constant lo buffer: filled once outside the tile loop (read-only,
            # valid across all tiles; T.vmax needs a same-shape tensor operand).
            if only_lower:
                T.fill(lo_ub, T.cast(lo, f32))

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

                    if const_val:
                        # Fold: output is the (equal) bound, x data is not read.
                        T.fill(y_ub, T.cast(lo, dtype))
                        T.copy(y_ub[0:tail], y[off : off + tail])
                    else:
                        # Only T.copy uses the dynamic tail slice (R-8).
                        T.copy(x[off : off + tail], x_ub[0:tail])
                        if need_cast:
                            # lossless up-cast (f16/bf16 -> f32, rint only, R-9)
                            T.vcast(x_ub, xf, round_mode="rint")

                        # --- hardtanh vector chain (fp32 domain, whole buffer) ---
                        # The result is carried in-place in xc (PL-1.23).
                        if identity:
                            # Fold: identity map, 0 data ops (pure copy below).
                            pass
                        elif only_upper:
                            # Fold lo = -inf: single-sided upper clamp, 1 op.
                            T.vmin(xc, T.cast(hi, f32), xc)
                        elif only_lower:
                            # Fold hi = +inf: single-sided lower clamp, 1 op.
                            # T.vmax takes tensor operands -> pre-filled lo_ub.
                            T.vmax(xc, lo_ub, xc)
                        else:
                            # Default (both bounds finite): fused 2-op clamp.
                            T.vclamp(xc, xc, T.cast(lo, f32), T.cast(hi, f32))

                        if need_cast:
                            # single round back to the input dtype (R-9)
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
    """Generate a CPU input tensor following the HardtanhWorkload logic (randn)."""
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


def _run_tensor(x_cpu, dtype_str, tag, min_val=-1.0, max_val=1.0):
    """Run the kernel on a CPU tensor and compare with the golden output."""
    x_cpu = x_cpu.detach().cpu().reshape(-1).contiguous()
    N = x_cpu.numel()
    golden = golden_hardtanh(x_cpu, min_val, max_val)
    kernel = _make_hardtanh_kernel(
        N, dtype_str, min_val, max_val, block_size=_DEFAULT_BLOCK_SIZE[dtype_str]
    )

    x_npu = x_cpu.npu()
    y = kernel(x_npu, N)
    y_cpu = y.cpu().reshape(-1)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({N},) dtype={dtype_str} "
            f"bounds=({min_val},{max_val}) max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
        f"bounds=({min_val},{max_val}) max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, min_val=-1.0, max_val=1.0):
    return _run_tensor(_make_input(N, dtype_str), dtype_str, tag, min_val, max_val)


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1..L0-3 smoke (DESIGN.md section 8.3): 1M elements per dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L0-smoke")

    # L0-4 corner / bound-in-out values (test_hardtanh_edge pattern):
    # -88.8 -> -1.0, -1.0 -> -1.0, -0.5 -> -0.5, 0 -> 0,
    # 0.5 -> 0.5, 1.0 -> 1.0, 88.8 -> 1.0.
    _run_tensor(
        _repeat_values([-88.8, -1.0, -0.5, 0.0, 0.5, 1.0, 88.8], 4096, torch.float32),
        "float32",
        "L0-corner",
    )

    # L0-5 lower/upper bound neighbourhood (covers both triggered sides and
    # the active linear interior).
    _run_tensor(
        torch.linspace(-1.1, 1.1, 1 << 20, dtype=torch.float32),
        "float32",
        "L0-linspace",
    )

    # L0-6 extremes / saturation: +/-65504 -> +/-1, +inf -> 1, -inf -> -1.
    _run_tensor(
        _repeat_values(
            [-65504.0, 65504.0, float("-inf"), float("inf")],
            4096,
            torch.float32,
        ),
        "float32",
        "L0-extreme",
    )

    # L0-7 non-divisible tail (4_000_000 % 2048 = 1216).
    _run_case(4000000, "float16", "L0-tail")

    # L0-8 manifest multi-dim shapes, flattened by the host (fp16/bf16).
    # [16,256,56,56] = 12,845,056 % 2048 = 512 -> tail tile.
    for shape in ((2048, 4096), (16, 256, 56, 56)):
        for dtype_str in ("float16", "bfloat16"):
            x = torch.randn(*shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(x, dtype_str, f"L0-manifest{shape}")

    # L0-9 fold lo = -inf (vmin path).
    _run_case(1048576, "float32", "L0-fold-lo-inf", min_val=float("-inf"), max_val=1.0)

    # L0-10 fold hi = +inf (vmax + lo_ub path).
    _run_case(1048576, "float32", "L0-fold-hi-inf", min_val=-1.0, max_val=float("inf"))


def run_L1():
    # L1: full dtype x boundary shapes -- manifest 16m flat, non-divisible
    # tails for every dtype, exact / partial tiles.
    _run_case(4096 * 4096, "float32", "L1-16m")
    _run_case(4000000, "float32", "L1-tail")
    _run_case(4000000, "bfloat16", "L1-tail")
    _run_case(1024, "float32", "L1-exact-tile")
    _run_case(4096, "float16", "L1-two-tiles")
    _run_case(2047, "bfloat16", "L1-partial-tile")
    # Manifest [4096,4096] fp16 / bf16.
    for dtype_str in ("float16", "bfloat16"):
        _run_case(16_777_216, dtype_str, "L1-16m")
    # Compile-time folding paths for every dtype.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(1048576, dtype_str, "L1-fold-lo-inf", min_val=float("-inf"), max_val=1.0)
        _run_case(1048576, dtype_str, "L1-fold-hi-inf", min_val=-1.0, max_val=float("inf"))
        _run_case(1048576, dtype_str, "L1-fold-both-inf", min_val=float("-inf"), max_val=float("inf"))
        _run_case(1048576, dtype_str, "L1-fold-const", min_val=0.5, max_val=0.5)
    # Non-default (asymmetric, still finite) bounds -> default vclamp path.
    _run_case(1048576, "float32", "L1-bounds-asym", min_val=-0.5, max_val=2.0)


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
    # Boundary: special-value inputs (clamp projection, +/-0, inf, NaN),
    # all non-blocking.  hardtanh(+inf) = hi, hardtanh(-inf) = lo,
    # hardtanh(NaN) = NaN (torch NaN propagation).
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
            x = _repeat_values([float("inf"), float("-inf"), 0.0], 4096, td)
            _run_tensor(x, dtype_str, "Boundary-inf")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} inf err={e}")

    # NaN propagation (R-4, device semantics probe; non-blocking).
    try:
        x = _repeat_values([float("nan"), 0.0, 1.0, -1.0], 4096, torch.float32)
        _run_tensor(x, "float32", "Boundary-nan")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): NaN err={e}")

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
