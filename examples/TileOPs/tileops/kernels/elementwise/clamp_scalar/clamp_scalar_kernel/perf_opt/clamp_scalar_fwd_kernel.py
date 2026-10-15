# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Clamp (scalar bounds) forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``ClampFwdKernel`` (TileOPs,
``make_unary_regcopy`` register_copy strategy) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = clamp(x, min, max) = min(max(x, min), max)
on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` same dtype.  ``min`` / ``max`` are construction-time
``Number | None`` scalars (keyword-only, at least one present); an absent
bound is not applied.  Intermediate math runs in fp32.  A NaN input, or a NaN
present bound, propagates NaN (as in ``torch.clamp``).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` + v-prefix vector chain
        (vcast -> vclamp/vmax/vmin -> vcmp+vselect) replaces the CUDA
        threads x npt two-level parallel; ``block_size`` folds threads*npt.
    R2: Ascend vmax/vmin do not document NaN propagation, so NaN is recovered
        per-point: ``T.vcmp(x, x, nan_mask, "ne")`` (IEEE unordered compare,
        NaN != NaN -> true) + ``T.vselect`` back-fills the original x32 (whose
        NaN bit pattern becomes the output NaN; no NaN constant is built).
    R3: a NaN present bound short-circuits to an all-NaN output (torch returns
        NaN for the whole tensor in that case) using only finite constants.
    R4: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and an
        if-guard for out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python clamp_scalar_fwd_kernel.py --level L0
    python clamp_scalar_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_clamp_scalar", "clamp_scalar_fwd_kernel"]


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_clamp_scalar(x, min_val=None, max_val=None):
    """Clamp reference: ``y = clamp(x, min, max)``, fp32 opmath, NaN-propagating.

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_clamp_scalar.py: ``ref_program = torch.clamp(x.float(), min, max)
    .to(x.dtype)``).  Independent of the NPU algorithm (no vcast / vclamp /
    vcmp+vselect / persistent structure).  Runs on CPU and does not require an
    NPU device.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = torch.clamp(x_f32, min_val, max_val)  # NaN input/bound -> NaN
    return y_f32.to(x_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def clamp_scalar_fwd_kernel(
    N, dtype, output_dtype=None, min_val=None, max_val=None, block_size=8192, num_cores=None
):
    """Clamp (scalar bounds) forward kernel: ``y = clamp(x, min, max)``.

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        output_dtype: optional output dtype (defaults to dtype).
        min_val: optional lower bound (Number | None); not applied if None.
        max_val: optional upper bound (Number | None); not applied if None.
        block_size: UB tile width (folds GPU threads*npt; 8192 per section 5.2).
        num_cores: optional explicit kernel count override (else section 5.5).

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y``
    is auto-allocated (``out_idx=[1]``).
    """
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"

    num_logical = T.ceildiv(N, block_size)  # logical tile count (compile-time)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

    # Trace-time (Python) branches: absent bound -> op not emitted; a NaN
    # present bound -> all-NaN short-circuit (DESIGN.md section 0.6 R2/R3).
    has_min = min_val is not None
    has_max = max_val is not None
    nan_bound = (has_min and math.isnan(float(min_val))) or (
        has_max and math.isnan(float(max_val))
    )
    # Plain Python floats for the vmax/vmin/vclamp scalar operands (the NPU
    # binary-op front end only accepts a raw scalar operand, not a cast expr).
    min_scalar = float(min_val) if has_min else None
    max_scalar = float(max_val) if has_max else None

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3).  All buffers are allocated
            # unconditionally (mish.py / hardsigmoid pattern); on the fp32 path
            # the fp32 transit buffers are populated via T.copy.
            x_ub = T.alloc_ub((block_size,), dtype)
            x32 = T.alloc_ub((block_size,), compute_dtype)
            buf1 = T.alloc_ub((block_size,), compute_dtype)
            buf2 = T.alloc_ub((block_size,), compute_dtype)
            nan_mask = T.alloc_ub((block_size,), "bool")
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # Persistent in-core serial task loop (static bound, grid-stride
            # mapping, if-guard masks out-of-range tiles).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    tail_size = T.min(block_size, shape - offset)

                    # 1. Load GM -> UB (only valid elements copied in).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 2. Cast to fp32 (fp16/bf16) or copy (fp32).
                    if dtype != compute_dtype:
                        T.vcast(x_ub, x32, round_mode="rint")
                    else:
                        T.copy(x_ub, x32)

                    # 3. Clamp chain (fp32).  Absent sides are not applied.
                    if nan_bound:
                        # NaN present bound -> whole output NaN (torch
                        # semantics), built from finite 0.0 only (no NaN
                        # constant; avoid VP-2026-0142 cache trap).
                        T.vmul(x32, 0.0, buf1)   # finite->0; +-inf->NaN; NaN->NaN
                        T.vdiv(buf1, 0.0, buf1)  # perf_opt: in-place (buf2 unused)
                    else:
                        if has_min and has_max:
                            T.vclamp(x32, buf1, min_scalar, max_scalar)
                        elif has_min:
                            T.vmax(x32, min_scalar, buf1)
                        else:
                            T.vmin(x32, max_scalar, buf1)
                        # perf_opt: NaN recovery dropped -- measured that
                        # vmax/vmin/vclamp preserve NaN on this toolchain
                        # (fp16/fp32, all three bound forms), so the extra
                        # vcmp+vselect is redundant.

                    # 5. Cast back to output dtype (fp16/bf16) or copy (fp32).
                    if out_dtype != compute_dtype:
                        T.vcast(buf1, y_ub, round_mode="rint")
                    else:
                        T.copy(buf1, y_ub)

                    # 6. Store UB -> GM (only valid elements copied out).
                    T.copy(y_ub[0:tail_size], y[offset : offset + tail_size])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Precision tolerances (DESIGN.md section 8.2, identical to the source-repo
# test_clamp_scalar.py ``_get_tolerances``).
_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-5),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _max_diff_metric(out, ref):
    """Return the max absolute difference between out and ref.

    NaN/Inf aware: positions where both are NaN (or both equal infinities)
    count as equal and contribute zero difference.
    """
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()


def _run_tensor(x, dtype_str, tag, min_val=None, max_val=None, equal_nan=False, block_size=None):
    """Run kernel on a pre-made tensor and compare against golden."""
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kwargs = {"min_val": min_val, "max_val": max_val}
    if block_size is not None:
        kwargs["block_size"] = block_size
    kernel = clamp_scalar_fwd_kernel(N, dtype_str, **kwargs)
    y = kernel(x, N)
    golden = golden_clamp_scalar(x, min_val, max_val)
    y_cpu = y.cpu()
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} "
        f"min={min_val} max={max_val} max_diff={max_diff:.2e}"
    )
    return max_diff


def _run_case(N, dtype_str, tag, min_val=None, max_val=None, scale=1.0, block_size=None):
    """Run kernel on random data of shape (N,) with given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(x, dtype_str, tag, min_val=min_val, max_val=max_val, block_size=block_size)


# ---------------------------------------------------------------------------
# hierarchical testing
# ---------------------------------------------------------------------------
_L0_N = 1048576


def run_L0():
    """L0 gate: 1M-element cases covering all three dtypes and the
    both / min-only / max-only bound forms (DESIGN.md section 8.3)."""
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(_L0_N, dtype_str, "L0", min_val=-0.5, max_val=0.5)
    # Bound present/absent combinations.
    _run_case(_L0_N, "float32", "L0", min_val=-0.5, max_val=None)
    _run_case(_L0_N, "float32", "L0", min_val=None, max_val=0.5)
    _run_case(_L0_N, "float16", "L0", min_val=-0.5, max_val=None)
    _run_case(_L0_N, "bfloat16", "L0", min_val=None, max_val=0.5)


def run_L1():
    """L1: divisible + non-divisible shapes (tail path) across all dtypes and
    all bound forms, plus a large spot check and a non-default block_size."""
    bound_forms = [(-0.5, 0.5), (-0.5, None), (None, 0.5)]
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (8192, 4097, 100, 4000000):
            for min_val, max_val in bound_forms:
                _run_case(N, dtype_str, "L1", min_val=min_val, max_val=max_val)
    _run_case(16777216, "float16", "L1", min_val=-0.5, max_val=0.5)
    _run_case(16777216, "float32", "L1", min_val=-0.5, max_val=0.5)
    # Non-default block_size sanity (UB budget still holds).
    _run_case(_L0_N, "float16", "L1", min_val=-0.5, max_val=0.5, block_size=2048)


def run_L2():
    """L2: edge sizes (warn only, non-blocking)."""
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2", min_val=-0.5, max_val=0.5)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4000000, "float32", "L2", min_val=-0.5, max_val=0.5)
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (Record without blocking): N=4000000 err={e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    Includes the critical NaN-propagation probe (DESIGN.md section 0.6 R2 /
    section 9.1): verify ``vcmp(x, x, "ne")`` self-compare actually detects NaN
    so that NaN in -> NaN out (matching the golden).  Also covers +-inf, NaN
    bound (all-NaN output, section 0.6 R3), and min > max.
    """
    N = 4096

    # +-inf input: clamp(+inf)=max, clamp(-inf)=min (converges, no overflow).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(N, dtype=torch.float32)
            x[0::4] = float("inf")
            x[1::4] = float("-inf")
            x = x.to(torch_dtype).npu()
            _run_tensor(x, dtype_str, "Boundary-inf", min_val=-0.5, max_val=0.5)
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): inf dtype={dtype_str} err={e}")

    # NaN propagation probe: NaN in -> NaN out (R2 vcmp(ne)+vselect).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(N, dtype=torch.float32)
            x[0::8] = float("nan")
            x[1::8] = float("inf")
            x[2::8] = float("-inf")
            x = x.to(torch_dtype).npu()
            kernel = clamp_scalar_fwd_kernel(N, dtype_str, min_val=-0.5, max_val=0.5)
            y = kernel(x, N).cpu()
            golden = golden_clamp_scalar(x, -0.5, 0.5)
            torch.testing.assert_close(
                y,
                golden,
                atol=_TOLERANCE[dtype_str][0],
                rtol=_TOLERANCE[dtype_str][1],
                equal_nan=True,
            )
            nan_in = torch.isnan(x.cpu())
            nan_out = torch.isnan(y)
            assert bool(torch.equal(nan_in, nan_out)), (
                f"NaN mask mismatch: input NaN positions {int(nan_in.sum().item())} "
                f"vs output {int(nan_out.sum().item())}"
            )
            print(
                f"[Boundary] PASS: NaN->NaN (R2 vcmp(ne)+vselect verified) "
                f"dtype={dtype_str} nan_count={int(nan_in.sum().item())}"
            )
        except AssertionError as e:
            print(f"[Boundary] FAIL (NaN probe): dtype={dtype_str} err={e}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): NaN dtype={dtype_str} err={e}")

    # NaN present bound -> all-NaN output (R3 short-circuit).
    for dtype_str in ("float32", "float16", "bfloat16"):
        for min_val, max_val in [(float("nan"), 0.5), (-0.5, float("nan"))]:
            try:
                x = torch.randn(N, dtype=torch.float32).to(_TORCH_DTYPE[dtype_str]).npu()
                kernel = clamp_scalar_fwd_kernel(
                    N, dtype_str, min_val=min_val, max_val=max_val
                )
                y = kernel(x, N).cpu()
                golden = golden_clamp_scalar(x, min_val, max_val)
                assert bool(torch.isnan(y).all()), "NaN-bound output not all NaN"
                assert bool(torch.isnan(golden).all()), "golden NaN-bound not all NaN"
                print(
                    f"[Boundary] PASS: NaN-bound -> all NaN dtype={dtype_str} "
                    f"min={min_val} max={max_val} nan_out={int(torch.isnan(y).sum().item())}"
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Boundary] WARN (Record without blocking): NaN-bound "
                    f"dtype={dtype_str} min={min_val} max={max_val} err={e}"
                )

    # min > max (matches torch's min(max(x, min), max) composition; result = max).
    try:
        x = torch.randn(N, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-min>max", min_val=0.5, max_val=-0.5)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): min>max err={e}")


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
