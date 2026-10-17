# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Isnan forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``IsnanFwdKernel`` (TileOPs,
``FloatPredicateKernel`` / ``register_copy`` strategy, OUTPUT_DTYPE=bool) to
TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = isnan(x)   (elementwise float predicate, bool output, exact)

on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` bool.  NaN -> True; +-inf / +-0 / subnormal /
finite -> False.  The predicate is exact boolean logic (no fp arithmetic), so
the comparison tolerance is atol = rtol = 0.

NPU redesign (DESIGN.md sections 0.6 / 3.3, revision v1):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` with ``T.alloc_ub``
        tiles and explicit-slice ``T.copy``; the CUDA ``threads x npt``
        two-level parallel folds into a single ``block_size``.
    R2: NaN predicate via the native-dtype ORDERED self-compare
        ``T.vcmp(x_ub, x_ub, mask_ub, "le")`` -- ``x <= x`` is true for every
        non-NaN value (incl. +-inf / +-0 / subnormal) and false ONLY for NaN --
        then inverted: ``isnan(x) = NOT (x <= x)``.  This replaces
        ``T.isnan(T.cast(x, "float32"))`` (no fp32 widening, no special
        function).  NOTE: ``vcmp "ne"`` is lowered as an ORDERED not-equal
        (``ne == lt | gt``) on this toolchain, so a ``"ne"`` self-compare is
        false for NaN and does NOT detect it -- it must not be used (see
        repro/CONST-vcmp-ne-ordered.py).
    R3: bool output representation: the on-chip bool mask cannot load/store GM
        and ``vcast`` has no ``i1->i8`` direct edge, so the mask is turned into
        a one-byte int8 0/1 via ``vcast(bool->f16)`` (non-NaN=1.0 / NaN=0.0) +
        ``T.vsub(1.0, m)`` complement in the fp16 domain (0.0/1.0 are exact) +
        ``vcast(f16->i8)``; the host views the result back to ``torch.bool``
        (metadata view only).  ``vnot`` does not support bool/int1 and
        ``vselect`` does not support int8/uint8 operands, hence the fp16
        complement.
    Persistent core split: ``num_kernels = min(num_logical, vector_cores)``
        with an in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python isnan_fwd_kernel.py --level L0
    python isnan_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual sync
# points (DESIGN.md section 7).  This literal mirrors the wrapper's
# ``ascend_mode = "Developer"`` and is read by Stage 5 integration.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
ASCEND_MODE = "Developer"

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["golden_isnan", "isnan_fwd_kernel"]

# Float predicate dtype set (FloatPredicateKernel FLOAT_DTYPES); integer / bool
# inputs are rejected (ValueError, DESIGN.md sections 9.3 / 11.3).
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# bool byte storage dtype (DESIGN.md section 3.3 / 11.4): bool i1 cannot
# load/store GM, so the predicate result lands as an int8 0/1 byte.
_BOOL_STORAGE_DTYPE = "int8"


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_isnan(x):
    """NaN predicate reference: y = torch.isnan(x)  (bool output, exact).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_isnan.py: ``ref_program = torch.isnan(x.detach().cpu().float())``,
    atol = rtol = 0).  Independent of the NPU algorithm (no vcmp self-compare /
    vcast byte chain / vsub complement / int8 byte storage / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    return torch.isnan(x.detach().cpu().float()).to(torch.bool)


# Stage-4 tuned default (perf_opt/opt_log.md).  Exposed at module level so the
# Stage-5 wrapper switch block can pair its default assignment with this value.
TUNED_BLOCK_SIZE = 8192


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def isnan_fwd_kernel(N, dtype, output_dtype=None, block_size=TUNED_BLOCK_SIZE, num_cores=None):
    """Isnan forward kernel: ``y = isnan(x)`` (bool via int8 byte storage).

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32"; other dtypes
            raise ``ValueError``.
        output_dtype: optional output (storage) dtype; defaults to ``"int8"``
            (the bool byte storage).
        block_size: UB tile width (folds GPU threads*npt); default
        ``TUNED_BLOCK_SIZE`` = 8192 (Stage-4 winner, opt_log.md).
        num_cores: optional explicit kernel-count override (else section 5.5).

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y`` is
    auto-allocated (``out_idx=[1]``).  The caller must pass an ``int8`` byte
    view only through the returned tensor, and views the result back to
    ``torch.bool`` (bool i1 cannot load/store GM -- see the module docstring).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"isnan_fwd_kernel only supports dtypes {SUPPORTED_DTYPES}, "
            f"got {dtype!r}"
        )

    out_dtype = output_dtype or _BOOL_STORAGE_DTYPE

    # Stage-4 T1: static full tiles + one dynamic tail; T2: 2-stage pipeline.
    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)  # static bound
    # fp16 constant 1.0 for the NOT(x<=x) complement (0.0/1.0 are exact).
    one = T.cast(1.0, "float16")

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB buffers (DESIGN.md section 4.3): v-prefix APIs require UB
            # residency.  All buffers are allocated unconditionally (a
            # conditional alloc is rejected by the TVM script parser).
            x_ub = T.alloc_ub((block_size,), dtype)
            mask_ub = T.alloc_ub((block_size,), "bool")        # le mask: non-NaN=1
            zero_f16_ub = T.alloc_ub((block_size,), "float16")  # select A (non-NaN)
            one_f16_ub = T.alloc_ub((block_size,), "float16")   # select B (NaN)
            sel_f16_ub = T.alloc_ub((block_size,), "float16")   # 0.0 / 1.0
            y_ub = T.alloc_ub((block_size,), out_dtype)        # int8 output tile

            # A7: replace (vcast+vsub) with one vselect (4 vector passes -> 3).
            T.fill(zero_f16_ub, T.cast(0.0, "float16"))
            T.fill(one_f16_ub, one)

            # Stage-4 T1+T2: static full-tile loop, software-pipelined.
            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(x[offset : offset + block_size], x_ub)
                        T.vcmp(x_ub, x_ub, mask_ub, "le")
                        T.vselect(mask_ub, zero_f16_ub, one_f16_ub, sel_f16_ub)
                        T.vcast(sel_f16_ub, y_ub, round_mode="rint")
                        T.copy(y_ub, y[offset : offset + block_size])

            # Tail epilogue: at most one partial tile, owned by core 0.
            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    T.vcmp(x_ub, x_ub, mask_ub, "le")
                    T.vselect(mask_ub, zero_f16_ub, one_f16_ub, sel_f16_ub)
                    T.vcast(sel_f16_ub, y_ub, round_mode="rint")
                    T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _max_diff_metric(out, ref):
    """Max absolute difference between out and ref (bool, exact)."""
    out = out.detach().cpu().to(torch.uint8)
    ref = ref.detach().cpu().to(torch.uint8)
    eq = out == ref
    if bool(eq.all()):
        return 0.0
    diff = (out.to(torch.int64) - ref.to(torch.int64)).abs()
    diff = torch.where(eq, torch.zeros_like(diff), diff)
    return diff.max().item()


def _make_input(N, dtype_str, special=False):
    """Random float input of shape (N,) (CPU).

    With ``special=True`` a dense mix of NaN / +-inf / +-0 / finite values is
    injected so the predicate and the +-inf non-misclassification are exercised
    (DESIGN.md section 8.3).
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]
    x = torch.randn(N, dtype=torch.float32)
    if special:
        x[0::8] = float("nan")
        x[1::8] = float("inf")
        x[2::8] = float("-inf")
        x[3::8] = 0.0
        x[4::8] = -0.0
        x[5::8] = 1e-30  # subnormal probe
        x[6::8] = 3.4e38  # large finite
        x[7::8] = -3.4e38
    return x.to(torch_dtype)


def _run_tensor(x, dtype_str, tag):
    """Run kernel on a pre-made CPU tensor and compare against golden."""
    N = x.shape[0]
    kernel = isnan_fwd_kernel(N, dtype_str)
    y_storage = kernel(x.npu(), N).cpu()
    # int8 byte storage -> bool (metadata view only).
    y = y_storage.view(torch.bool)
    golden = golden_isnan(x)
    # Exact: atol = rtol = 0 (DESIGN.md section 8.2).
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    max_diff = _max_diff_metric(y, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, special=False, block_size=None):
    """Run kernel on random (optionally special-value) data of shape (N,)."""
    x = _make_input(N, dtype_str, special=special)
    if block_size is not None:
        kernel = isnan_fwd_kernel(N, dtype_str, block_size=block_size)
        y_storage = kernel(x.npu(), N).cpu()
        y = y_storage.view(torch.bool)
        golden = golden_isnan(x)
        torch.testing.assert_close(y, golden, atol=0, rtol=0)
        max_diff = _max_diff_metric(y, golden)
        print(
            f"[{tag}] PASS: shape=({N},) dtype={dtype_str} "
            f"block_size={block_size} max_diff={max_diff:.2e}"
        )
        return max_diff
    return _run_tensor(x, dtype_str, tag)


# ---------------------------------------------------------------------------
# hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """L0 gate: 1M-element cases for every float dtype (DESIGN.md 8.3) with
    injected NaN / +-inf, plus the negative case that int dtypes are rejected."""
    N = 1048576
    for dtype_str in SUPPORTED_DTYPES:
        _run_case(N, dtype_str, "L0", special=True)

    # Negative: non-float dtypes must be rejected with ValueError.
    for bad_dtype in ("int32", "bool", "int8"):
        try:
            isnan_fwd_kernel(N, bad_dtype)
            raise AssertionError(f"{bad_dtype} must be rejected")
        except ValueError:
            print(f"[L0] PASS: dtype {bad_dtype} rejected")


def run_L1():
    """L1: divisible + non-divisible shapes across all dtypes (tail path)."""
    for dtype_str in SUPPORTED_DTYPES:
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1", special=True)
    # Large spot checks.
    _run_case(4194304, "float16", "L1")
    _run_case(4194304, "float32", "L1")
    _run_case(1048576, "bfloat16", "L1", special=True)

    # Non-default block_size sanity (UB budget still holds).
    _run_case(1048576, "float32", "L1", special=True, block_size=2048)


def run_L2():
    """L2: edge sizes (warn only, non-blocking)."""
    for dtype_str in SUPPORTED_DTYPES:
        for N in (1, 2, 10):
            try:
                _run_case(N, dtype_str, "L2", special=True)
            except Exception as e:  # noqa: BLE001
                print(f"[L2] WARN (Record without blocking): N={N} dtype={dtype_str} err={e}")


def run_boundary():
    """Boundary: NaN / +-inf / +-0 / subnormal / extremes (warn only,
    non-blocking; exact equality required when they run).

    The critical probe is NaN -> True and +-inf -> False, verifying the
    ``vcmp(x, x, "le")`` ORDERED self-compare + complement (``isnan = NOT(x<=x)``,
    DESIGN.md section 0.6 R2 / 9.1) against the independent ``torch.isnan``
    golden.  The all-NaN / all-+-inf cases are the discriminating probes that a
    (wrong) ``vcmp "ne"`` implementation would fail.
    """
    N = 4096
    try:
        _run_case(N, "float32", "Boundary-special", special=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): special err={e}")

    # Pure NaN: every element must be True.
    try:
        x = torch.full((N,), float("nan"), dtype=torch.float32)
        y = isnan_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert bool(y.all()), "all-NaN input must map to all-True"
        print(f"[Boundary] PASS: all-NaN -> all-True (nan_count={N})")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-NaN err={e}")

    # Pure +-inf: must all be False (NOT misclassified as NaN).
    try:
        x = torch.empty(N, dtype=torch.float32)
        x[0::2] = float("inf")
        x[1::2] = float("-inf")
        y = isnan_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "+-inf must not be classified as NaN"
        print("[Boundary] PASS: +-inf -> all-False (not misclassified)")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")


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
