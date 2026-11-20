# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Clamp (scalar bounds) forward kernel (NPU, Developer mode) -- Stage-4 R2 final.

Semantics (frozen by DESIGN.md section 0.1)::

    y = clamp(x, min, max) = min(max(x, min), max)

on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` same dtype.  ``min`` / ``max`` are construction-time
``Number | None`` scalars (at least one present); an absent bound is not
applied.  A NaN input, or a NaN present bound, propagates NaN (as in
``torch.clamp``).

Stage-4 R2 redesign (see ``perf_opt/opt_log.md`` R2 section):

    N1 (``r2-native``): the clamp comparison runs in the *input* dtype for
        fp16 / fp32 instead of the ``vcast -> clamp -> vcast`` fp32 transit
        chain.  Legal because clamp only selects values that already exist in
        the input / bound set and rounding is monotone:
        ``rn(clamp(x, lo, hi)) == clamp(x, rn(lo), rn(hi))`` on the dtype grid,
        so the fp32-opmath golden and the native path agree (measured
        ``max_diff = 0`` for every L0/L1 case, NaN mask exact on 586-point
        probes).  The scalar bounds are pre-rounded to the dtype grid at trace
        time (``struct`` round-trip, round-half-even).  bf16 keeps the fp32
        transit: the NPU vector compare ops document no bf16 support
        (``docs/Tilelang.language/比较操作/T.vclamp.md`` / ``T.vmax.md`` /
        ``T.vmin.md`` section 2.2.1).
    N2 (``r2-a4``): the task space is split into static-extent *full* tiles
        (hot loop, ``T.Pipelined(num_stages=2)``) plus one dynamic tail tile
        owned by core 0, so the hot loop no longer recomputes ``T.min`` /
        dynamic copy extents per tile (pattern-library PL-1.22).
    N3: the two-bound case uses ``vmax`` then ``vmin`` with ``dst == src1``
        only (never ``dst == src2``), avoiding the undocumented src2-alias
        performance tax (pattern-library PL-1.15).
    NaN present bound keeps the R1 short-circuit (``vmul 0`` then ``vdiv 0``)
    in the compute dtype; R1's redundant per-point NaN recovery stays removed
    (pattern-library PL-1.21).

Run the embedded hierarchical tests::

    python ClampScalarFwdOp.py --level L0
    python ClampScalarFwdOp.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse
import math
import struct

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_clamp_scalar", "clamp_scalar_fwd_kernel", "TUNED_BLOCK_SIZE"]

# Stage-4 R2 tuned default tile width (module-level so the Stage-5 wrapper
# switch block can pair its ``_DEFAULT_BLOCK_SIZE`` entry with this constant).
# The R2 measurement kept the family value: the wrapper pins block_size per
# dtype (fp32 4096, fp16/bf16 8192) and a 16384 sweep showed no gain
# (see perf_opt/opt_log.md, config_no_gain).
TUNED_BLOCK_SIZE = 8192

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

_L0_N = 1048576

_STUCT_FMT = {"float16": "e", "float32": "f"}


def _round_bound(value, dtype_str):
    """Round a Python scalar bound onto the ``dtype_str`` grid (round-half-even)."""
    fmt = _STUCT_FMT.get(dtype_str)
    if fmt is None:
        return float(value)
    return float(struct.unpack(fmt, struct.pack(fmt, float(value)))[0])


@tilelang.jit(out_idx=[1], target="npuir")
def clamp_scalar_fwd_kernel(
    N, dtype, output_dtype=None, min_val=None, max_val=None, block_size=8192, num_cores=None
):
    """Clamp (scalar bounds) forward kernel: native clamp + A4 tile structure."""
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    native = dtype in ("float16", "float32") and out_dtype == dtype

    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)

    has_min = min_val is not None
    has_max = max_val is not None
    nan_bound = (has_min and math.isnan(float(min_val))) or (
        has_max and math.isnan(float(max_val))
    )
    if native:
        min_scalar = _round_bound(min_val, dtype) if has_min else None
        max_scalar = _round_bound(max_val, dtype) if has_max else None
    else:
        min_scalar = float(min_val) if has_min else None
        max_scalar = float(max_val) if has_max else None

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            x32 = T.alloc_ub((block_size,), compute_dtype)
            buf1 = T.alloc_ub((block_size,), compute_dtype)
            y_ub = T.alloc_ub((block_size,), out_dtype)

            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(x[offset : offset + block_size], x_ub)

                        if nan_bound:
                            if native:
                                T.vmul(x_ub, 0.0, y_ub)
                                T.vdiv(y_ub, 0.0, y_ub)
                            else:
                                T.vcast(x_ub, x32, round_mode="rint")
                                T.vmul(x32, 0.0, buf1)
                                T.vdiv(buf1, 0.0, buf1)
                                T.vcast(buf1, y_ub, round_mode="rint")
                        elif native:
                            if has_min and has_max:
                                T.vmax(x_ub, min_scalar, y_ub)
                                T.vmin(y_ub, max_scalar, y_ub)
                            elif has_min:
                                T.vmax(x_ub, min_scalar, y_ub)
                            else:
                                T.vmin(x_ub, max_scalar, y_ub)
                        else:
                            T.vcast(x_ub, x32, round_mode="rint")
                            if has_min and has_max:
                                T.vmax(x32, min_scalar, buf1)
                                T.vmin(buf1, max_scalar, buf1)
                            elif has_min:
                                T.vmax(x32, min_scalar, buf1)
                            else:
                                T.vmin(x32, max_scalar, buf1)
                            T.vcast(buf1, y_ub, round_mode="rint")

                        T.copy(y_ub, y[offset : offset + block_size])

            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(x[offset : offset + tail], x_ub[0:tail])

                    if nan_bound:
                        if native:
                            T.vmul(x_ub, 0.0, y_ub)
                            T.vdiv(y_ub, 0.0, y_ub)
                        else:
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vmul(x32, 0.0, buf1)
                            T.vdiv(buf1, 0.0, buf1)
                            T.vcast(buf1, y_ub, round_mode="rint")
                    elif native:
                        if has_min and has_max:
                            T.vmax(x_ub, min_scalar, y_ub)
                            T.vmin(y_ub, max_scalar, y_ub)
                        elif has_min:
                            T.vmax(x_ub, min_scalar, y_ub)
                        else:
                            T.vmin(x_ub, max_scalar, y_ub)
                    else:
                        T.vcast(x_ub, x32, round_mode="rint")
                        if has_min and has_max:
                            T.vmax(x32, min_scalar, buf1)
                            T.vmin(buf1, max_scalar, buf1)
                        elif has_min:
                            T.vmax(x32, min_scalar, buf1)
                        else:
                            T.vmin(x32, max_scalar, buf1)
                        T.vcast(buf1, y_ub, round_mode="rint")

                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


def golden_clamp_scalar(x, min_val=None, max_val=None):
    """Clamp reference: ``y = clamp(x, min, max)``, fp32 opmath, NaN-propagating."""
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = torch.clamp(x_f32, min_val, max_val)  # NaN input/bound -> NaN
    return y_f32.to(x_cpu.dtype)


def _max_diff_metric(out, ref):
    """Return the max absolute difference between out and ref (NaN/Inf aware)."""
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    return diff.max().item()



def _run_tensor(x, dtype_str, tag, min_val=None, max_val=None, equal_nan=False,
                block_size=None):
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kwargs = {"min_val": min_val, "max_val": max_val}
    if block_size is not None:
        kwargs["block_size"] = block_size
    kernel = clamp_scalar_fwd_kernel(N, dtype_str, **kwargs)
    y = kernel(x, N)
    golden = golden_clamp_scalar(x, min_val, max_val)
    y_cpu = y.cpu()
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol,
                               equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    print(f"[{tag}] PASS: N={N} dtype={dtype_str} min={min_val} "
          f"max={max_val} max_diff={max_diff:.2e}")
    return max_diff

def _run_case(N, dtype_str, tag, min_val=None, max_val=None, scale=1.0,
              block_size=None):
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(x, dtype_str, tag, min_val=min_val, max_val=max_val,
                       block_size=block_size)

def run_L0():
    """L0 gate: 1M-element cases, all three dtypes / bound forms."""
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(_L0_N, dtype_str, "L0", min_val=-0.5, max_val=0.5)
    _run_case(_L0_N, "float32", "L0", min_val=-0.5, max_val=None)
    _run_case(_L0_N, "float32", "L0", min_val=None, max_val=0.5)
    _run_case(_L0_N, "float16", "L0", min_val=-0.5, max_val=None)
    _run_case(_L0_N, "bfloat16", "L0", min_val=None, max_val=0.5)

def run_L1():
    """L1: divisible + non-divisible shapes (tail path), all dtypes/forms."""
    bound_forms = [(-0.5, 0.5), (-0.5, None), (None, 0.5)]
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (8192, 4097, 100, 4000000):
            for min_val, max_val in bound_forms:
                _run_case(N, dtype_str, "L1", min_val=min_val,
                          max_val=max_val)
    _run_case(16777216, "float16", "L1", min_val=-0.5, max_val=0.5)
    _run_case(16777216, "float32", "L1", min_val=-0.5, max_val=0.5)
    _run_case(_L0_N, "float16", "L1", min_val=-0.5, max_val=0.5,
              block_size=2048)

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
    """Boundary: special values.  FAIL lines make main() exit non-zero."""
    N = 4096
    failed = []

    # +-inf input: clamp(+inf)=max, clamp(-inf)=min.
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(N, dtype=torch.float32)
            x[0::4] = float("inf")
            x[1::4] = float("-inf")
            x = x.to(torch_dtype).npu()
            _run_tensor(x, dtype_str, "Boundary-inf", min_val=-0.5,
                        max_val=0.5)
        except Exception as e:  # noqa: BLE001
            failed.append(f"inf/{dtype_str}: {e}")
            print(f"[Boundary] FAIL (inf): dtype={dtype_str} err={e}")

    # NaN propagation: NaN in -> NaN out, mask-exact.
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
            nan_in = torch.isnan(x.cpu())
            nan_out = torch.isnan(y)
            assert bool(torch.equal(nan_in, nan_out)), (
                f"NaN mask mismatch: in {int(nan_in.sum().item())} vs "
                f"out {int(nan_out.sum().item())}"
            )
            torch.testing.assert_close(
                y, golden, atol=_TOLERANCE[dtype_str][0],
                rtol=_TOLERANCE[dtype_str][1], equal_nan=True,
            )
            print(f"[Boundary] PASS: NaN->NaN dtype={dtype_str} "
                  f"nan_count={int(nan_in.sum().item())}")
        except Exception as e:  # noqa: BLE001
            failed.append(f"nan/{dtype_str}: {e}")
            print(f"[Boundary] FAIL (NaN probe): dtype={dtype_str} err={e}")

    # NaN present bound -> all-NaN output.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for min_val, max_val in [(float("nan"), 0.5), (-0.5, float("nan"))]:
            try:
                x = torch.randn(N, dtype=torch.float32).to(
                    _TORCH_DTYPE[dtype_str]).npu()
                kernel = clamp_scalar_fwd_kernel(N, dtype_str, min_val=min_val,
                                         max_val=max_val)
                y = kernel(x, N).cpu()
                golden = golden_clamp_scalar(x, min_val, max_val)
                assert bool(torch.isnan(y).all()), "NaN-bound out not all NaN"
                assert bool(torch.isnan(golden).all()), "golden not all NaN"
                print(f"[Boundary] PASS: NaN-bound -> all NaN "
                      f"dtype={dtype_str} min={min_val} max={max_val}")
            except Exception as e:  # noqa: BLE001
                failed.append(f"nanbound/{dtype_str}/{min_val}: {e}")
                print(f"[Boundary] FAIL (NaN-bound): dtype={dtype_str} "
                      f"min={min_val} max={max_val} err={e}")

    # min > max (torch composition: result == max).
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            x = torch.randn(N, dtype=torch.float32).to(
                _TORCH_DTYPE[dtype_str]).npu()
            _run_tensor(x, dtype_str, "Boundary-min>max", min_val=0.5,
                        max_val=-0.5)
        except Exception as e:  # noqa: BLE001
            failed.append(f"minmax/{dtype_str}: {e}")
            print(f"[Boundary] FAIL (min>max): dtype={dtype_str} err={e}")

    # Non-representable bounds (rounding to the compute dtype).
    for dtype_str, mn, mx in [
        ("float16", -0.3333333, 1.0000001),
        ("float16", -7e-8, 6.1e-5),
        ("bfloat16", -0.3333333, 1.0000001),
        ("float32", -0.3333333, 1.0000001),
    ]:
        try:
            x = torch.randn(N, dtype=torch.float32).to(
                _TORCH_DTYPE[dtype_str]).npu()
            _run_tensor(x, dtype_str, "Boundary-round-bound", min_val=mn,
                        max_val=mx)
        except Exception as e:  # noqa: BLE001
            failed.append(f"roundbound/{dtype_str}: {e}")
            print(f"[Boundary] FAIL (round-bound): dtype={dtype_str} "
                  f"min={mn} max={mx} err={e}")

    if failed:
        print(f"[Boundary] {len(failed)} FAILURES")
    return failed



# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    run_L0()
    failed = []
    if args.level == "all":
        run_L1()
        run_L2()
        failed = run_boundary()
    if failed:
        print(f"\033[91m{failed} boundary case(s) failed\033[0m")
        raise SystemExit(1)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
