# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_simple_reduce_kernel (AminFwdOp resident path) -- NPU Developer mode.

Migrated from the GPU TileOPs ``_simple_reduce_kernel`` (``op_kind="amin"``)
to ``target="npuir"``.  amin is the exact dual of the already-frozen sibling
``examples/amax/_simple_reduce_kernel`` (min<->max, +inf<->-inf,
reduce_min<->reduce_max).

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = min_{0 <= j < N} x[i, j]          # torch.amin(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  out: (M,) same_as(x)

NPU redesign (DESIGN.md section 0.6, frozen):
    R1: raw-N boundary -- kernel receives the original (M, N); no 256-aligned
        pad and no masked load. The M tail uses ``real_m = min(bm, M - off_m)``
        plus explicit src/dst slices on both ``T.copy`` calls.
    R2: persistent one-dimensional core split -- ``num_kernels =
        min(num_logical, vector_cores)`` with an in-core ``T.serial``
        grid-stride loop and an if-guard for out-of-range row blocks
        (static loop bound, dynamic guard only).
    R3: dtype domain -- fp16/fp32 reduce in the ORIGINAL domain (min is an
        exact selection, no rounding); bf16 is promoted with
        ``T.vcast(..., round_mode="rint")`` to fp32 and cast back on output
        (``T.reduce_min`` does not support bf16).
    R4: block_m ladder from a probe-calibrated UB budget; bm * N * B(dtype)
        <= 64KB manual budget (x2 for auto-multi-buffer fits 192KB UB).
    R5: min carried by a DIRECT ``T.reduce_min(dim=1, clear=True)`` -- the
        source GPU kernel instead negates, calls reduce_max and negates back
        (min(x) = -max(-x)); the negate bridge is replaced by the native NPU
        min reduce (E1).  ``clear=True`` initializes dst to +inf, the min
        identity element (E4).

Known toolchain constraints enforced here (DESIGN.md section 9.1):
    - ``T.reduce_min`` src/dst MUST be the same dtype (mixed dtype silently
      corrupts, probe-backed) -> bf16 uses a dedicated fp32 work buffer.
    - unused ``alloc`` is NOT dead-code-eliminated -> fp16/fp32 and bf16 use
      separate prim_funcs (no conditional alloc).
    - ``T.Parallel`` may not read GM -> all GM traffic goes through ``T.copy``.
    - ``T.vcast`` f32->f32 is non-identity -> same-dtype uses plain copy.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# Probe-calibrated race-safe manual UB budget per block (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Large-buffer bytes per element (DESIGN.md section 4.5 / 3.2 dtype matrix):
# fp16 x_ub=2; fp32 x_ub=4; bf16 x_ub=2 + x_work fp32=4.
_B_ELEM = {"float16": 2, "float32": 4, "bfloat16": 6}

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")

_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


# ---------- block_m ladder (DESIGN.md section 5.2) ----------
def select_block_m(M, N, dtype, ub_budget=UB_MANUAL_BUDGET):
    """Factory-period block_m ladder.

    Basic ladder {1,2,4,8}; extended ladder {16..2048} when the logical block
    count would otherwise exceed 2x the vector core count (small-N/large-M).
    Returns None when even p=1 cannot fit -> caller routes to the sibling tiled
    migration unit ``_simple_reduce_kernel_tiled`` (DESIGN.md section 5.2).
    """
    b = _B_ELEM[dtype]
    cand = [p for p in (1, 2, 4, 8) if p * N * b <= ub_budget]
    block_m = max(cand) if cand else None
    if block_m is None:
        return None

    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if _ceildiv(M, block_m) > 2 * vector_cores:
        ext = [p for p in (16, 32, 64, 128, 256, 512, 1024, 2048) if p * N * b <= ub_budget]
        block_m = max(ext + [block_m])

    # Never allocate more rows than exist: trailing UB rows would be
    # uninitialized and (for narrow N) can pollute valid rows (DESIGN R6).
    return min(block_m, M)


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amin(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amin(x, dim=-1)`` on CPU.

    Computed in fp32 then cast back to the input dtype, matching
    ``examples/TileOPs/tests/ops/test_reduce.py::ReduceTest.ref_program``.
    It does NOT reproduce the NPU redesign (no padding removal / original-domain
    reduce / direct reduce_min / persistent split), guaranteeing validation
    independence (DESIGN.md section 8.1).

    Must use the explicit ``torch`` module function form (``torch.amin(x, ...)``),
    not the equivalent tensor-method chain ``x.float().amin(dim=-1)``, so the
    Stage 3 golden gate (S3-GOLDEN-TORCH) recognizes a real torch API call.
    """
    return torch.amin(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel factory ----------
def _simple_reduce_kernel(M, N, op_kind, dtype):
    """Build the resident amin kernel factory.

    Mirrors the GPU source structure:
      ``_simple_reduce_kernel(M, N, op_kind, dtype) -> _func(block_m) -> main``.
    K9: the ``threads`` parameter is removed (NPU has no CUDA threads concept).

    ``op_kind`` is kept for sibling-reduce interface compatibility; this
    migration unit implements the ``amin`` branch only.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "amin":
        raise ValueError(f"this unit implements op_kind='amin' only, got {op_kind!r}")
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")

    vector_cores = NPUUtils.get().get_aicore_num() * 2  # pure Vector cores x2
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)            # logical row blocks
        num_kernels = min(num_logical, vector_cores)  # persistent grid
        num_local_tasks = _ceildiv(num_logical, num_kernels)  # static bound

        if is_bf16:
            # bf16 path: vcast to fp32 -> reduce_min(fp32) -> cast back to bf16.
            # Separate prim_func so the fp32 work buffer is only allocated when
            # actually used (unused alloc is not DCE'd).
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_shared((block_m, N), "float32")
                    m_ub = T.alloc_shared((block_m, 1), "float32")
                    out_ub = T.alloc_shared((block_m,), dtype)

                    for s in T.serial(num_local_tasks):     # static loop bound
                        task = s * num_kernels + cid        # grid-stride mapping
                        if task < num_logical:              # dynamic guard only
                            off_m = task * block_m
                            # M tail: real_m rows only; src/dst both sliced so
                            # no out-of-bounds read of the last partial block.
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            # bf16 reduce_min is unsupported -> promote to fp32
                            # with rint (exact, order/equality preserving).
                            T.vcast(x_ub, x_work, round_mode="rint")
                            # clear=True initializes dst to +inf (min identity).
                            T.reduce_min(x_work, m_ub, dim=1, clear=True)
                            for i in T.Parallel(block_m):
                                # (bm,1) -> (bm,): shape contraction + fp32->bf16.
                                out_ub[i] = T.cast(m_ub[i, 0], dtype)
                            # src truncated to real_m so garbage rows are not written.
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:
            # fp16 / fp32 path: original-domain reduce_min, src/dst same dtype.
            # min is an exact selection op (no arithmetic / no rounding), so the
            # native-domain result is bit-identical to an fp32-promoted baseline.
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    m_ub = T.alloc_shared((block_m, 1), dtype)
                    out_ub = T.alloc_shared((block_m,), dtype)

                    for s in T.serial(num_local_tasks):     # static loop bound
                        task = s * num_kernels + cid        # grid-stride mapping
                        if task < num_logical:              # dynamic guard only
                            off_m = task * block_m
                            # M tail: explicit double slice (raw-N contract, no
                            # padding and no masked load).
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            # Same-dtype reduce_min: src/dst MUST match dtype.
                            # clear=True initializes dst to +inf (min identity);
                            # direct min (no negate -> reduce_max -> negate bridge).
                            T.reduce_min(x_ub, m_ub, dim=1, clear=True)
                            for i in T.Parallel(block_m):
                                # (bm,1) -> (bm,) identity (result already target dtype).
                                out_ub[i] = m_ub[i, 0]
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------- precision comparison ----------
def _max_diff(y, ref):
    """Finite-only max abs diff.

    Non-finite rows (inf/nan) would make ``(y - ref).abs().max()`` report
    ``nan`` (inf - inf / nan - nan) even when the gate comparison passes; the
    gate stays ``torch.testing.assert_close`` and this helper only makes the
    printed number meaningful.
    """
    yf = y.float()
    rf = ref.float()
    both_finite = torch.isfinite(yf) & torch.isfinite(rf)
    if not bool(both_finite.any()):
        return float("nan")
    return (yf[both_finite] - rf[both_finite]).abs().max().item()


def _run_case(M, N, dtype_str, block_m, tag, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel(M, N, "amin", dtype_str)(block_m)
    y = kernel(x)
    ref = golden_amin(x.cpu())
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = _max_diff(y.cpu(), ref)
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.2e}"
    )
    return max_diff


def _run_case_from_values(x, dtype_str, tag, equal_nan=False):
    """Run a pre-built CPU tensor through the kernel; check against golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    block_m = min(select_block_m(M, N, dtype_str) or 1, M)
    x_npu = x.to(dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel(M, N, "amin", dtype_str)(block_m)
    y = kernel(x_npu)
    ref = golden_amin(x.to(dtype=torch_dtype))
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = _max_diff(y.cpu(), ref)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} max_diff={max_diff:.2e}")
    return max_diff


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.3 (manifest workloads + R1/R2/R3/R4 risk coverage).
def run_L0():
    max_diff = 0.0
    # L0-1: smoke, all three dtypes.
    max_diff = max(max_diff, _run_case(32, 256, "float16", 8, "L0-1a"))
    max_diff = max(max_diff, _run_case(32, 256, "bfloat16", 8, "L0-1b"))
    max_diff = max(max_diff, _run_case(32, 256, "float32", 8, "L0-1c"))
    # L0-2/3: manifest hidden-state (2048,4096) -> persistent multi-wave.
    max_diff = max(max_diff, _run_case(2048, 4096, "float16", 8, "L0-2"))
    max_diff = max(max_diff, _run_case(2048, 4096, "bfloat16", 2, "L0-3"))
    # L0-4: manifest 3d-multidim reshaped to (128,16384) fp16.
    max_diff = max(max_diff, _run_case(128, 16384, "float16", 2, "L0-4"))
    # L0-5: harness ReduceBasic fp32.
    max_diff = max(max_diff, _run_case(128, 512, "float32", 8, "L0-5"))
    # L0-6: N non-aligned (300 not a 32/256 multiple).
    max_diff = max(max_diff, _run_case(128, 300, "float16", 8, "L0-6a"))
    max_diff = max(max_diff, _run_case(128, 300, "bfloat16", 8, "L0-6b"))
    # L0-7: M tail (129 % 8 == 1).
    max_diff = max(max_diff, _run_case(129, 512, "float16", 8, "L0-7"))
    # L0-8: single row.
    max_diff = max(max_diff, _run_case(1, 512, "float16", 1, "L0-8"))
    # L0-9: boundary-value semantics (verify_equiv corner set replayed on NPU),
    # includes the +inf min-identity rows.
    for dtype_str in ("float16", "float32"):
        max_diff = max(max_diff, _run_boundary_rows(dtype_str, "L0-9"))
    # L0-11: NaN propagation (device probe replayed on NPU).
    for dtype_str in ("float16", "float32"):
        max_diff = max(max_diff, _run_nan_rows(dtype_str, "L0-11"))
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


def _boundary_rows(dtype_str):
    """Construct (8, 257) rows covering the E1/E4 corner set for amin."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    finfo = torch.finfo(torch_dtype)
    rows = [
        torch.full((257,), float("-inf")),                             # all -inf -> -inf
        torch.full((257,), float("inf")),                              # all +inf -> +inf (min identity)
        torch.cat([torch.tensor([float("inf")]), torch.zeros(256)]),   # +inf amid finite -> 0
        torch.cat([torch.tensor([float("-inf")]), torch.zeros(256)]),  # -inf amid finite -> -inf
        torch.full((257,), -3.0),                                      # repeated extreme
        torch.arange(257, dtype=torch.float32) * 0.5,                  # monotone
        torch.zeros(257),                                              # all zeros
        torch.cat([torch.tensor([finfo.min]), torch.zeros(256)]),      # most-negative dtype extreme
    ]
    x = torch.stack(rows, dim=0).to(torch.float32)
    # clamp finite sentinels into the dtype range for fp16.
    if dtype_str == "float16":
        x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return x


def _run_boundary_rows(dtype_str, tag):
    x = _boundary_rows(dtype_str).to(_DTYPE_MAP[dtype_str][0])
    return _run_case_from_values(x, dtype_str, tag)


def _nan_rows(dtype_str):
    """NaN rows mirroring the device probe CONST-reduce-min-nan-propagation."""
    x = torch.full((8, 64), 3.0)
    x[0, :] = -3.0                       # finite row (must stay clean)
    x[1, 7] = float("nan")               # NaN amid finite (centered)
    x[2, :] = float("nan")               # all NaN
    x[3, 0] = float("nan")               # NaN at first position
    x[4, 63] = float("nan")              # NaN at last position
    x[5, :] = float("inf")               # all +inf
    x[6, :] = -2.0
    x[6, 32] = float("inf")              # +inf amid finite
    x[7, :] = 2.0
    x[7, 32] = float("-inf")             # -inf amid finite
    return x.to(_DTYPE_MAP[dtype_str][0])


def _run_nan_rows(dtype_str, tag):
    """NaN input: torch.amin propagates NaN; check pointwise equality on NPU."""
    return _run_case_from_values(_nan_rows(dtype_str), dtype_str, tag, equal_nan=True)


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(130, 256, "float16", 8, "L1-1"))
    max_diff = max(max_diff, _run_case(64, 1024, "bfloat16", 4, "L1-2"))
    max_diff = max(max_diff, _run_case(8, 512, "float16", 8, "L1-3"))
    max_diff = max(max_diff, _run_case(256, 1024, "float32", 8, "L1-4"))
    max_diff = max(max_diff, _run_case(3, 300, "bfloat16", 2, "L1-5"))
    max_diff = max(max_diff, _run_case(32, 32, "float32", 8, "L1-6"))
    max_diff = max(max_diff, _run_case(255, 777, "float16", 4, "L1-7"))
    # NaN propagation is a documented L0/L1 semantics item (DESIGN R1, section 9.2).
    max_diff = max(max_diff, _run_nan_rows("float16", "L1-8a-nan"))
    max_diff = max(max_diff, _run_nan_rows("float32", "L1-8b-nan"))
    max_diff = max(max_diff, _run_nan_rows("bfloat16", "L1-8c-nan"))
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", 1),      # minimal
        (8, 256, "float16", 8),    # exact divide M == block_m
        (9, 256, "float16", 8),    # M == block_m + 1
        (32, 1, "float16", 4),     # N == 1 single-element reduce
        (16, 33, "bfloat16", 4),   # bf16 small odd N
    ]
    for M, N, dtype_str, block_m in cases:
        try:
            _run_case(M, N, dtype_str, block_m, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # mixed-sign zeros (IEEE: min(-0.0, +0.0) selection preserved).
    try:
        x = torch.zeros(32, 256, dtype=torch.float16)
        x[:, 0] = -0.0
        _run_case_from_values(x, "float16", "Boundary-zeros-fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros: {e}")

    # subnormal extremes.
    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((32, 256), finfo.tiny / 2, dtype=torch.float16)
        _run_case_from_values(x, "float16", "Boundary-subnormal-fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")

    # all -inf / all +inf.
    for val, name in ((float("-inf"), "all-neg-inf"), (float("inf"), "all-pos-inf")):
        try:
            x = torch.full((32, 256), val, dtype=torch.float16)
            _run_case_from_values(x, "float16", f"Boundary-{name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {e}")

    # large M persistent multi-wave regression.
    try:
        _run_case(1024, 2048, "float16", 4, "Boundary-large-M")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large M (1024,2048): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
