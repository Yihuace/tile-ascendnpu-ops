# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Shared L0/L1/L2/Boundary harness for the R2 clamp experiment branches.

The final deliverable ``perf_opt/ClampScalarFwdOp.py`` carries its own inlined
copy of this harness (gate 4 requires the golden + ``run_*`` entry points to be
defined in that file).  The throw-away branch kernels import it from here so
that each branch file stays a thin, single-change variant of the builder.

Usage inside a branch file::

    from _r2_harness import make_harness, main
    run_L0, run_L1, run_L2, run_boundary = make_harness(lambda: clamp_scalar_fwd_kernel)

The harness resolves the kernel through the getter, so ``python <branch>.py
--level all`` always exercises the branch's own builder.
"""

import argparse
import math

import torch

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


def make_harness(kernel_getter):
    """Build ``(run_L0, run_L1, run_L2, run_boundary)`` for one kernel getter."""

    def _run_tensor(x, dtype_str, tag, min_val=None, max_val=None, equal_nan=False,
                    block_size=None):
        N = x.shape[0]
        atol, rtol = _TOLERANCE[dtype_str]
        kwargs = {"min_val": min_val, "max_val": max_val}
        if block_size is not None:
            kwargs["block_size"] = block_size
        kernel = kernel_getter()(N, dtype_str, **kwargs)
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
                kernel = kernel_getter()(N, dtype_str, min_val=-0.5, max_val=0.5)
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
                    kernel = kernel_getter()(N, dtype_str, min_val=min_val,
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

    return run_L0, run_L1, run_L2, run_boundary


def main(kernel_getter, argv=None):
    run_L0, run_L1, run_L2, run_boundary = make_harness(kernel_getter)
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args(argv)
    run_L0()
    failed = []
    if args.level == "all":
        run_L1()
        run_L2()
        failed = run_boundary()
    if failed:
        raise SystemExit(1)
    print("\033[92mAll check passed!\033[0m")


__all__ = ["golden_clamp_scalar", "make_harness", "main", "_TOLERANCE",
           "_max_diff_metric"]


def _unused():  # pragma: no cover - keeps math import referenced
    return math.nan
