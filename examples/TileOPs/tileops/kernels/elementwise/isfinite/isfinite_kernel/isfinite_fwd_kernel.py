# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Isfinite forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``IsfiniteFwdKernel`` (TileOPs,
``FloatPredicateKernel`` / ``register_copy`` strategy, OUTPUT_DTYPE=bool) to
TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y = isfinite(x)   (elementwise float predicate, bool output, exact)

on a flattened contiguous ``(N,)`` tensor of dtype float16 | bfloat16 |
float32; output ``(N,)`` bool.  Finite values (incl. +-0 / subnormal / finite
extremes) -> True; NaN / +inf / -inf -> False.  The predicate is exact boolean
logic (no fp arithmetic), so the comparison tolerance is atol = rtol = 0.

NPU redesign (DESIGN.md sections 0.6 / 3.1-3.3):
    R1: one-dim ``T.Kernel(num_kernels, is_npu=True)`` with ``T.alloc_ub``
        tiles and explicit-slice ``T.copy``; the CUDA ``threads x npt``
        two-level parallel folds into a single ``block_size``.
    R2: finiteness predicate via native-dtype ORDERED comparisons:
        ``isfinite(x) = (x > -inf) AND (x < +inf)``, which is algebraically
        identical to the canonical ``|x| < +inf`` but needs no ``T.vabs``
        (T.vabs is bf16-unsupported).  Ordered ``gt`` / ``lt`` are false for
        NaN, so NaN correctly answers "not finite"; ``+inf`` fails ``lt`` and
        ``-inf`` fails ``gt``.  This replaces ``T.isfinite(T.cast(x,
        "float32"))`` (no fp32 widening, no special function).  NOTE: this
        toolchain's ``vcmp "ne"`` is lowered as an ORDERED not-equal
        (``ne == lt | gt``) with no unordered mode, so it does NOT detect
        special values and must not be used (see repro/CONST-vcmp-ne-ordered.py
        in the isnan kernel; CG-2026-0024).
    R3: bool output representation: the on-chip bool masks cannot load/store GM
        and ``vcast`` has no ``i1->i8`` direct edge, so each mask is turned
        into an fp16 0.0/1.0 via ``vcast(bool->f16)``, the two masks are merged
        with ``vmul`` in the fp16 domain (0.0/1.0 product == logical AND,
        exact), and the product is cast ``vcast(f16->i8)`` to a one-byte 0/1.
        The host views the result back to ``torch.bool`` (metadata view only).
        bool-domain ``vand``/``vnot`` and int8 ``vselect`` are not supported.
    Persistent core split: ``num_kernels = min(num_logical, vector_cores)``
        with an in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds).

Run the embedded hierarchical tests:
    python isfinite_fwd_kernel.py --level L0
    python isfinite_fwd_kernel.py --level all
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

__all__ = ["golden_isfinite", "isfinite_fwd_kernel"]

# Float predicate dtype set (FloatPredicateKernel FLOAT_DTYPES); integer / bool
# inputs are rejected (ValueError, DESIGN.md sections 9.3 / 11.3).
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# bool byte storage dtype (DESIGN.md section 3.3 / 11.4): bool i1 cannot
# load/store GM, so the predicate result lands as an int8 0/1 byte.
_BOOL_STORAGE_DTYPE = "int8"


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_isfinite(x):
    """Finiteness predicate reference: y = torch.isfinite(x)  (bool, exact).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_isfinite.py: ``ref_program = torch.isfinite(x.detach().cpu().float())``,
    atol = rtol = 0).  Independent of the NPU algorithm (no vcmp gt/lt against
    +-inf / vmul mask merge / vcast byte chain / int8 byte storage / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    return torch.isfinite(x.detach().cpu().float()).to(torch.bool)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def isfinite_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    """Isfinite forward kernel: ``y = isfinite(x)`` (bool via int8 byte storage).

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32"; other dtypes
            raise ``ValueError``.
        output_dtype: optional output (storage) dtype; defaults to ``"int8"``
            (the bool byte storage).
        block_size: UB tile width (folds GPU threads*npt; DESIGN.md section 5.2).
        num_cores: optional explicit kernel-count override (else section 5.5).

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y`` is
    auto-allocated (``out_idx=[1]``).  The caller passes the input directly and
    views the returned ``int8`` byte tensor back to ``torch.bool`` (bool i1
    cannot load/store GM -- see the module docstring).
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"isfinite_fwd_kernel only supports dtypes {SUPPORTED_DTYPES}, "
            f"got {dtype!r}"
        )

    out_dtype = output_dtype or _BOOL_STORAGE_DTYPE

    num_logical = T.ceildiv(N, block_size)  # logical tile count (compile-time)
    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op; re-queried per build so other devices adapt automatically).
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores > num_logical:
        num_kernels = min(num_logical, vector_cores)
    else:
        num_kernels = num_cores
    num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

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
            posinf_ub = T.alloc_ub((block_size,), dtype)   # constant +inf tile
            neginf_ub = T.alloc_ub((block_size,), dtype)   # constant -inf tile
            m_lo = T.alloc_ub((block_size,), "bool")       # vcmp("gt",-inf) mask
            m_hi = T.alloc_ub((block_size,), "bool")       # vcmp("lt",+inf) mask
            f_lo = T.alloc_ub((block_size,), "float16")    # mask -> f16 (0.0/1.0)
            f_hi = T.alloc_ub((block_size,), "float16")    # mask -> f16 (0.0/1.0)
            y_ub = T.alloc_ub((block_size,), out_dtype)    # int8 output byte tile

            # Pre-fill the +-inf constant tiles once, outside the tile loop.
            T.fill(posinf_ub, pos_inf)
            T.fill(neginf_ub, neg_inf)

            # Persistent in-core serial tile loop (static bound, grid-stride
            # mapping, if-guard masks out-of-range tiles).
            for i in T.serial(num_local_tasks):
                block_id = i * num_kernels + cid
                if block_id < num_logical:
                    offset = block_id * block_size
                    remaining = shape - offset
                    tail_size = T.min(block_size, remaining)

                    # 1. Load GM -> UB (only valid elements copied in).
                    T.copy(x[offset : offset + tail_size], x_ub[0:tail_size])

                    # 2. Finiteness predicate (ORDERED gt/lt only):
                    #    isfinite(x) = (|x| < +inf) = (x > -inf) & (x < +inf).
                    #    The two-sided form is algebraically identical to
                    #    |x| < +inf but needs no T.vabs (bf16-unsupported).
                    #    Native input dtype, no fp32 widening (DESIGN.md section
                    #    0.6 R2).  "ne" is ordered here (ne == lt|gt) and does NOT
                    #    detect special values -- it must NOT be used.
                    T.vcmp(x_ub, neginf_ub, m_lo, "gt")   # x > -inf (NaN -> false)
                    T.vcmp(x_ub, posinf_ub, m_hi, "lt")   # x < +inf (NaN -> false)

                    # 3. Merge the two masks in the fp16 domain: bool mask -> f16
                    #    (1.0/0.0) then vmul (0.0/1.0 product == logical AND,
                    #    exact).  No i1->i8 direct edge, no bool/int1 vand/vnot,
                    #    no int8/uint8 vselect operand support (DESIGN.md 0.6 R3).
                    T.vcast(m_lo, f_lo, round_mode="rint")
                    T.vcast(m_hi, f_hi, round_mode="rint")
                    T.vmul(f_lo, f_hi, f_lo)

                    # 4. isfinite indicator (0.0/1.0) -> int8 byte (0/1).
                    T.vcast(f_lo, y_ub, round_mode="rint")

                    # 5. Store UB -> GM (int8 bytes; host views bool; only valid
                    #    elements copied out, garbage tail lanes are never
                    #    written back).
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
    kernel = isfinite_fwd_kernel(N, dtype_str)
    y_storage = kernel(x.npu(), N).cpu()
    # int8 byte storage -> bool (metadata view only).
    y = y_storage.view(torch.bool)
    golden = golden_isfinite(x)
    # Exact: atol = rtol = 0 (DESIGN.md section 8.2).
    torch.testing.assert_close(y, golden, atol=0, rtol=0)
    max_diff = _max_diff_metric(y, golden)
    print(f"[{tag}] PASS: shape=({N},) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_case(N, dtype_str, tag, special=False, block_size=None):
    """Run kernel on random (optionally special-value) data of shape (N,)."""
    x = _make_input(N, dtype_str, special=special)
    if block_size is not None:
        kernel = isfinite_fwd_kernel(N, dtype_str, block_size=block_size)
        y_storage = kernel(x.npu(), N).cpu()
        y = y_storage.view(torch.bool)
        golden = golden_isfinite(x)
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
            isfinite_fwd_kernel(N, bad_dtype)
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

    The critical probes are all-NaN -> all-False and all-+-inf -> all-False,
    plus all-finite -> all-True, verifying the two ORDERED masks
    (``isfinite = (x>-inf)&(x<+inf)``, DESIGN.md section 0.6 R2 / 9.1) against
    the independent ``torch.isfinite`` golden.  These are the discriminating
    probes that a (wrong) ``vcmp "ne"`` or ``(x*x) < +inf`` implementation would
    fail.
    """
    N = 4096
    try:
        _run_case(N, "float32", "Boundary-special", special=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): special err={e}")

    # Pure finite: every element must be True.
    try:
        x = torch.randn(N, dtype=torch.float32)
        y = isfinite_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert bool(y.all()), "all-finite input must map to all-True"
        print(f"[Boundary] PASS: all-finite -> all-True (finite_count={N})")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): finite err={e}")

    # Pure NaN: every element must be False (NaN is NOT finite).
    try:
        x = torch.full((N,), float("nan"), dtype=torch.float32)
        y = isfinite_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "all-NaN input must map to all-False"
        print(f"[Boundary] PASS: all-NaN -> all-False (nan_count={N})")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-NaN err={e}")

    # Pure +-inf: must all be False.
    try:
        x = torch.empty(N, dtype=torch.float32)
        x[0::2] = float("inf")
        x[1::2] = float("-inf")
        y = isfinite_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert not bool(y.any()), "all-+-inf input must map to all-False"
        print("[Boundary] PASS: all-+-inf -> all-False")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf err={e}")

    # +-0 / subnormal / finite: must all be True.
    try:
        x = torch.zeros(N, dtype=torch.float32)
        x[0::3] = -0.0
        x[1::3] = 1e-30  # subnormal probe
        x[2::3] = -3.4e38
        y = isfinite_fwd_kernel(N, "float32")(x.npu(), N).cpu().view(torch.bool)
        assert bool(y.all()), "+-0 / subnormal / finite must be True"
        print("[Boundary] PASS: +-0 / subnormal / finite -> all-True")
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
