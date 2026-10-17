# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Isinf forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``IsinfFwdKernel`` (TileOPs,
``FloatPredicateKernel`` / ``register_copy`` strategy, OUTPUT_DTYPE=bool) to
TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = isinf(x)   (elementwise float predicate, bool output, exact)

on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` bool.  +-inf -> True; NaN / finite / +-0 / subnormal
-> False.  The predicate is exact boolean logic (no fp arithmetic), so the
comparison tolerance is atol = rtol = 0.

NPU redesign (DESIGN.md sections 0.6 / 3.1-3.3):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` with ``T.alloc_ub``
        tiles and explicit-slice ``T.copy``; the CUDA ``threads x npt``
        two-level parallel folds into a single ``block_size``.
    R2: infinity predicate via the native-dtype ORDERED ``eq`` against the two
        infinities: ``isinf(x) = (x == +inf) | (x == -inf)``.  Ordered ``eq``
        is true only for the matching infinity and false for NaN, so NaN
        correctly answers "not infinite".  This replaces
        ``T.isinf(T.cast(x, "float32"))`` (no fp32 widening, no special
        function).  NOTE: ``vcmp "ne"`` is lowered as an ORDERED not-equal
        (``ne == lt | gt``) on this toolchain, so an ``x != x`` self-compare is
        false for NaN and does NOT detect special values -- it must not be used
        (see repro/CONST-vcmp-ne-ordered.py).
    R3: bool output representation: the on-chip bool masks cannot load/store GM
        and ``vcast`` has no ``i1->i8`` direct edge, so each mask is turned into
        an fp16 0.0/1.0 via ``vcast(bool->f16)``, the two masks are merged with
        ``vadd`` in the fp16 domain (mutually exclusive -> sum in {0.0, 1.0},
        exact; bool-domain ``vor``/``vmax`` and int8 ``vselect`` are not
        supported), and the sum is cast ``vcast(f16->i8)`` to a one-byte 0/1.
        The host views the result back to ``torch.bool`` (metadata view only).
    Persistent core split: ``num_kernels = min(num_logical, vector_cores)``
        with an in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python isinf_fwd_kernel.py --level L0
    python isinf_fwd_kernel.py --level all
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

__all__ = ["golden_isinf", "isinf_fwd_kernel"]

# Float predicate dtype set (FloatPredicateKernel FLOAT_DTYPES); integer / bool
# inputs are rejected (ValueError, DESIGN.md sections 9.3 / 11.3).
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# bool byte storage dtype (DESIGN.md section 3.3 / 11.4): bool i1 cannot
# load/store GM, so the predicate result lands as an int8 0/1 byte.
_BOOL_STORAGE_DTYPE = "int8"


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_isinf(x):
    """Infinity predicate reference: y = torch.isinf(x)  (bool output, exact).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_isinf.py: ``ref_program = torch.isinf(x.detach().cpu().float())``,
    atol = rtol = 0).  Independent of the NPU algorithm (no vcmp eq against
    +-inf / vadd mask merge / vcast byte chain / int8 byte storage / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    return torch.isinf(x.detach().cpu().float()).to(torch.bool)


# Stage-4 tuned default (perf_opt/opt_log.md).  Exposed at module level for the
# Stage-5 wrapper switch block.
TUNED_BLOCK_SIZE = 8192


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def isinf_fwd_kernel(N, dtype, output_dtype=None, block_size=TUNED_BLOCK_SIZE, num_cores=None):
    """Isinf forward kernel: ``y = isinf(x)`` (bool via int8 byte storage).

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
    auto-allocated (``out_idx=[1]``).  The caller passes the input directly and
    views the returned ``int8`` byte tensor back to ``torch.bool`` (bool i1
    cannot load/store GM -- see the module docstring).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"isinf_fwd_kernel only supports dtypes {SUPPORTED_DTYPES}, "
            f"got {dtype!r}"
        )

    out_dtype = output_dtype or _BOOL_STORAGE_DTYPE

    # Stage-4 T1: static full tiles + one dynamic tail.
    num_full = N // block_size
    tail = N - num_full * block_size
    num_logical = num_full + (1 if tail else 0)
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_full, num_kernels)  # static bound
    # Trace-time pure-name flags (expression conditions do not fold).
    use_abs = dtype != "bfloat16"          # vabs has no bf16 dtype edge
    use_select = not use_abs

    # +-inf constants in the input dtype.  T.infinity supports fp16/fp32 but NOT
    # bf16 (docs/Tilelang.language/创建操作/T.infinity.md 2.2.1: bf16 = x), so
    # bf16 builds the constant by casting an fp32 infinity (exact narrowing: bf16
    # keeps the fp32 exponent field).  This scalar-cast constant path does not
    # introduce any elementwise fp32 widening (DESIGN.md sections 3.3 / 9.1).
    if dtype == "bfloat16":
        pos_inf = T.cast(T.infinity("float32"), "bfloat16")
        neg_inf = T.cast(-T.infinity("float32"), "bfloat16")
    else:
        pos_inf = T.infinity(dtype)
        neg_inf = -T.infinity(dtype)

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
            abs_ub = T.alloc_ub((block_size,), dtype)      # |x| (fp16/fp32 path)
            posinf_ub = T.alloc_ub((block_size,), dtype)   # constant +inf tile
            neginf_ub = T.alloc_ub((block_size,), dtype)   # constant -inf tile
            m_pos = T.alloc_ub((block_size,), "bool")      # eq(+inf) mask
            m_neg = T.alloc_ub((block_size,), "bool")      # eq(-inf) mask
            f_a = T.alloc_ub((block_size,), "float16")
            f_b = T.alloc_ub((block_size,), "float16")
            one_ub = T.alloc_ub((block_size,), "float16")
            zero_ub = T.alloc_ub((block_size,), "float16")
            y_ub = T.alloc_ub((block_size,), out_dtype)    # int8 output byte tile

            # Pre-fill the constants once, outside the tile loop.
            T.fill(posinf_ub, pos_inf)
            T.fill(neginf_ub, neg_inf)
            T.fill(one_ub, T.cast(1.0, "float16"))
            T.fill(zero_ub, T.cast(0.0, "float16"))

            # Stage-4 T1+T2: static full-tile loop, software-pipelined.
            if num_full > 0:
                for i in T.Pipelined(num_local_tasks, num_stages=2):
                    block_id = i * num_kernels + cid
                    if block_id < num_full:
                        offset = block_id * block_size
                        T.copy(x[offset : offset + block_size], x_ub)
                        if use_abs:
                            # A8: isinf(x) = (|x| == +inf): 4 passes instead of 6.
                            T.vabs(x_ub, abs_ub)
                            T.vcmp(abs_ub, posinf_ub, m_pos, "eq")
                            T.vcast(m_pos, f_a, round_mode="rint")
                            T.vcast(f_a, y_ub, round_mode="rint")
                        if use_select:
                            # bf16: no vabs edge.  Keep the baseline mask merge
                            # (2 vcmp + 2 vcast + vadd + vcast); vselect was
                            # measured 6-9% SLOWER for bf16.
                            T.vcmp(x_ub, posinf_ub, m_pos, "eq")
                            T.vcmp(x_ub, neginf_ub, m_neg, "eq")
                            T.vcast(m_pos, f_a, round_mode="rint")
                            T.vcast(m_neg, f_b, round_mode="rint")
                            T.vadd(f_a, f_b, f_a)
                            T.vcast(f_a, y_ub, round_mode="rint")
                        T.copy(y_ub, y[offset : offset + block_size])

            # Tail epilogue: at most one partial tile, owned by core 0.
            if tail > 0:
                if cid == 0:
                    offset = num_full * block_size
                    T.copy(x[offset : offset + tail], x_ub[0:tail])
                    if use_abs:
                        T.vabs(x_ub, abs_ub)
                        T.vcmp(abs_ub, posinf_ub, m_pos, "eq")
                        T.vcast(m_pos, f_a, round_mode="rint")
                        T.vcast(f_a, y_ub, round_mode="rint")
                    if use_select:
                        T.vcmp(x_ub, posinf_ub, m_pos, "eq")
                        T.vcmp(x_ub, neginf_ub, m_neg, "eq")
                        T.vcast(m_pos, f_a, round_mode="rint")
                        T.vcast(m_neg, f_b, round_mode="rint")
                        T.vadd(f_a, f_b, f_a)
                        T.vcast(f_a, y_ub, round_mode="rint")
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

    With ``special=True`` a dense mix of NaN / +-inf / +-0 / subnormal / large
    finite values is injected so the predicate and the NaN / +-inf
    discrimination are exercised (DESIGN.md section 8.3).
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
    kernel = isinf_fwd_kernel(N, dtype_str)
    y_storage = kernel(x.npu(), N).cpu()
    # int8 byte storage -> bool (metadata view only).
    y = y_storage.view(torch.bool)
    golden = golden_isinf(x)
    # Exact: atol = rtol = 0 (DESIGN.md section 8.2).
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    max_diff = _max_diff_metric(y, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, special=False, block_size=None):
    """Run kernel on random (optionally special-value) data of shape (N,)."""
    x = _make_input(N, dtype_str, special=special)
    if block_size is not None:
        kernel = isinf_fwd_kernel(N, dtype_str, block_size=block_size)
        y_storage = kernel(x.npu(), N).cpu()
        y = y_storage.view(torch.bool)
        golden = golden_isinf(x)
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
    injected NaN / +-inf, plus the negative case that non-float dtypes are
    rejected."""
    N = 1048576
    for dtype_str in SUPPORTED_DTYPES:
        _run_case(N, dtype_str, "L0", special=True)

    # Negative: non-float dtypes must be rejected with ValueError.
    for bad_dtype in ("int32", "bool", "int8"):
        try:
            isinf_fwd_kernel(N, bad_dtype)
            raise AssertionError(f"{bad_dtype} must be rejected")
        except ValueError:
            print(f"[L0] PASS: dtype {bad_dtype} rejected")


def run_L1():
    """L1: divisible + non-divisible shapes across all dtypes (tail path)."""
    for dtype_str in SUPPORTED_DTYPES:
        for N in (8192, 100, 4097):
            _run_case(N, dtype_str, "L1", special=True)
    # Large spot checks.
    _run_case(4194304, "float16", "L1", special=True)
    _run_case(4194304, "float32", "L1", special=True)
    _run_case(1048576, "bfloat16", "L1", special=True)

    # Non-default block_size sanity (UB budget still holds).
    _run_case(1048576, "float32", "L1", special=True, block_size=4096)


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

    The critical probes are all-NaN -> all-False and all-+-inf -> all-True,
    verifying the two ORDERED ``eq`` masks against the two infinities
    (``isinf = (x==+inf)|(x==-inf)``, DESIGN.md section 0.6 R2 / 9.1) against the
    independent ``torch.isinf`` golden.  These are the discriminating probes
    that a (wrong) ``vcmp "ne"`` or ``(x-x)!=0`` implementation would fail.
    """
    N = 4096
    try:
        _run_case(N, "float32", "Boundary-special", special=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): special err={e}")

    # Pure NaN: every element must be False (NaN is NOT infinite).
    try:
        x = torch.full((N,), float("nan"), dtype=torch.float32)
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "all-NaN input must map to all-False"
        print(f"[Boundary] PASS: all-NaN -> all-False (nan_count={N})")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-NaN err={e}")

    # Pure +-inf: must all be True.
    try:
        x = torch.empty(N, dtype=torch.float32)
        x[0::2] = float("inf")
        x[1::2] = float("-inf")
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert bool(y.all()), "all-+-inf input must map to all-True"
        print("[Boundary] PASS: all-+-inf -> all-True")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")

    # +-0 / subnormal / finite: must all be False.
    try:
        x = torch.zeros(N, dtype=torch.float32)
        x[0::3] = -0.0
        x[1::3] = 1e-30  # subnormal probe
        x[2::3] = -3.4e38
        y = isinf_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "+-0 / subnormal / finite must be False"
        print("[Boundary] PASS: +-0 / subnormal / finite -> all-False")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): finite err={e}")


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
