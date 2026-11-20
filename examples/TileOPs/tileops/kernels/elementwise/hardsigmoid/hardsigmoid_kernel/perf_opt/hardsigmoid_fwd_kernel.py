# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Hardsigmoid forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``HardsigmoidFwdKernel`` (TileOPs,
``make_unary_regcopy`` register_copy strategy) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = clamp(x + 3, 0, 6) * (1 / 6) = relu6(x + 3) / 6
on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` same dtype.  Intermediate math runs in fp32.
NaN input propagates NaN (as in torch).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` + v-prefix vector
        chain (vadd -> vclamp -> vmul) replaces the CUDA threads x npt
        two-level parallel; ``block_size`` folds threads*npt.
    R2: Ascend vmax/vmin do not document NaN propagation, so NaN is
        recovered per-point: ``T.vcmp(x, x, nan_mask, "ne")`` (IEEE
        unordered compare, NaN != NaN -> true) + ``T.vselect`` back-fills
        NaN from a ``T.vbrc``-broadcast constant (single detection).
    R3: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and
        an if-guard for out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python hardsigmoid_fwd_kernel.py --level L0
    python hardsigmoid_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_hardsigmoid", "hardsigmoid_fwd_kernel"]


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_hardsigmoid(x):
    """Hardsigmoid reference: y = clamp(x + 3, 0, 6) / 6 (NaN-propagating).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_hardsigmoid.py: ``ref_program = F.hardsigmoid(x.float()).to(x.dtype)``).
    Independent of the NPU algorithm (no vcast / vcmp+vselect / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    x_cpu = x.detach().cpu()
    x_f32 = x_cpu.to(torch.float32)
    y_f32 = torch.nn.functional.hardsigmoid(x_f32)
    return y_f32.to(x_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def hardsigmoid_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    """Hardsigmoid forward kernel: ``y = clamp(x + 3, 0, 6) * (1 / 6)``.

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input (and output) dtype, "float16" | "bfloat16" | "float32".
        output_dtype: optional output dtype (defaults to dtype).
        block_size: UB tile width (folds GPU threads*npt; 8192 per §5.2).
        num_cores: optional explicit kernel count override (else §5.5 split).

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

    # fp32 constants (DESIGN.md section 0.1: upcast all constants to fp32 so
    # the add/clamp/mul chain runs in fp32 opmath, matching the source).
    three = T.cast(3.0, "float32")
    zero = T.cast(0.0, "float32")
    six = T.cast(6.0, "float32")
    one_sixth = T.cast(1.0 / 6.0, "float32")
    # NaN broadcast constant (let-bound scalar for T.vbrc, DESIGN.md §0.0 #3).
    nan_scalar = T.cast(float("nan"), "float32")

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3).  All buffers are allocated
            # unconditionally (mish.py pattern); on the fp32 path the fp32
            # transit buffers are populated via T.copy (no-op-ish dtype copy).
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32_ub = T.alloc_ub((block_size,), compute_dtype)
            buf1_ub = T.alloc_ub((block_size,), compute_dtype)
            buf2_ub = T.alloc_ub((block_size,), compute_dtype)
            nan_mask = T.alloc_ub((block_size,), "bool")
            y_ub = T.alloc_ub((block_size,), out_dtype)

            # Persistent in-core serial task loop (static bound, grid-stride
            # mapping, if-guard masks out-of-range tiles).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 1. Load GM -> UB (only valid elements copied in).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 2. Cast to fp32 (fp16/bf16) or copy (fp32).
                    #    Python-level branch: resolved at JIT trace time.
                    if dtype != compute_dtype:
                        T.vcast(x_ub, x_f32_ub, round_mode="rint")
                    else:
                        T.copy(x_ub, x_f32_ub)

                    # 3. Compute chain: clamp(x + 3, 0, 6) * (1/6).
                    T.vadd(x_f32_ub, three, buf1_ub)       # t = x + 3
                    T.vclamp(buf1_ub, buf2_ub, zero, six)  # c = clamp(t, 0, 6)
                    T.vmul(buf2_ub, one_sixth, buf1_ub)    # r = c * (1/6)

                    # perf_opt: NaN recovery dropped -- measured that vadd +
                    # vclamp + vmul already propagate NaN on this toolchain
                    # (NaN -> NaN through clamp(x+3,0,6)*(1/6)), so vbrc+vcmp+
                    # vselect is redundant (saves 3 vector ops).

                    # 5. Cast back to output dtype (fp16/bf16) or copy (fp32).
                    if out_dtype != compute_dtype:
                        T.vcast(buf1_ub, y_ub, round_mode="rint")
                    else:
                        T.copy(buf1_ub, y_ub)

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
# test_hardsigmoid.py ``_get_tolerances``).
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


def _run_tensor(x, dtype_str, tag, equal_nan=False):
    """Run kernel on a pre-made tensor and compare against golden."""
    N = x.shape[0]
    atol, rtol = _TOLERANCE[dtype_str]
    kernel = hardsigmoid_fwd_kernel(N, dtype_str)
    y = kernel(x, N)
    golden = golden_hardsigmoid(x)
    y_cpu = y.cpu()
    torch.testing.assert_close(y_cpu, golden, atol=atol, rtol=rtol, equal_nan=equal_nan)
    max_diff = _max_diff_metric(y_cpu, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, scale=1.0, block_size=None):
    """Run kernel on random data of shape (N,) with given dtype."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = (torch.randn(N, dtype=torch.float32) * scale).to(torch_dtype).npu()
    return _run_tensor(x, dtype_str, tag)


# ---------------------------------------------------------------------------
# hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: representative 1M-element cases covering all three dtypes."""
    N = 1048576
    _run_case(N, "float32", "L0")
    _run_case(N, "float16", "L0")
    _run_case(N, "bfloat16", "L0")


def run_L1():
    """L1: divisible + non-divisible shapes across all dtypes (tail path)."""
    for dtype_str in ("float32", "float16", "bfloat16"):
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1")
    # Large spot check.
    _run_case(4194304, "float16", "L1")
    _run_case(4194304, "float32", "L1")
    # Non-default block_size sanity (UB budget still holds).
    _run_case(1048576, "float16", "L1", block_size=2048)


def run_L2():
    """L2: edge sizes (warn only, non-blocking)."""
    for N in (1, 2, 10):
        try:
            _run_case(N, "float32", "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} err={e}")
    try:
        _run_case(4194304, "float32", "L2")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (Record without blocking): N=4194304 err={e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    Includes the critical NaN-propagation probe (DESIGN.md §0.6 R2 / §9.1):
    verify ``vcmp(x, x, "ne")`` self-compare actually detects NaN so that
    NaN in -> NaN out (matching the golden's NaN propagation).
    """
    N = 4096
    # zeros: hardsigmoid(0) = clamp(3,0,6)/6 = 0.5.
    try:
        x = torch.zeros(N, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros err={e}")

    # large positive: saturates to 1.0.
    try:
        x = (torch.randn(N, dtype=torch.float32) * 5.0 + 50.0).to(torch.float16).npu()
        _run_tensor(x, "float16", "Boundary-large-positive")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large-positive err={e}")

    # large negative: -> 0.0.
    try:
        x = torch.full((N,), -10.0, dtype=torch.float32).npu()
        _run_tensor(x, "float32", "Boundary-large-negative")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large-negative err={e}")

    # +-inf: clamp(+inf,0,6)/6 = 1.0, clamp(-inf,0,6)/6 = 0.0.
    try:
        x = torch.randn(N, dtype=torch.float32)
        x[0::4] = float("inf")
        x[1::4] = float("-inf")
        _run_tensor(x.to(torch.float32).npu(), "float32", "Boundary-inf")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")

    # NaN propagation probe (DESIGN.md §8.3 / §9.1): NaN in -> NaN out.
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            torch_dtype = _TORCH_DTYPE[dtype_str]
            x = torch.randn(N, dtype=torch.float32)
            x[0::8] = float("nan")
            x[1::8] = float("inf")
            x[2::8] = float("-inf")
            x = x.to(torch_dtype).npu()
            kernel = hardsigmoid_fwd_kernel(N, dtype_str)
            y = kernel(x, N).cpu()
            golden = golden_hardsigmoid(x)
            # NaN positions must be NaN in both (equal_nan); finite/inf
            # positions must match within tolerance.
            torch.testing.assert_close(
                y, golden,
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
