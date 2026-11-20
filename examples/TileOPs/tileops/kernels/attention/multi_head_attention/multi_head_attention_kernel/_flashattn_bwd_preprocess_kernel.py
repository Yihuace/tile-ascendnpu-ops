# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""FlashAttention backward preprocess kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_flashattn_bwd_preprocess_kernel``
(``/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`` lines 26-96,
carried by ``FlashAttnBwdPreprocessKernel`` in TileOPs
``MultiHeadAttentionBwdOp``) to TileLang ``target="npuir"``.

Semantics (DESIGN.md section 0.1/1.3, frozen):
    delta[b, h, s] = sum_d fp32(o[b, s, h, d]) * fp32(do[b, s, h, d])
    o, do : (B, S, H, D) float16 / bfloat16   (BSHD, D contiguous)
    delta : (B, H, S) float32                 (out_idx=[2])

Products and accumulation are carried out in fp32 (the source accumulates
into a fp32 fragment); the fp16/bf16 -> fp32 upcast is bit-exact (mantissa
11/8 bits <= fp32's 24), so the emitted product is bit-identical to the
source GPU "fp32 FMA" product (DESIGN.md section 1.6.1 P1/P2, machine
verified by ``verify_equiv.py``).  No normalisation, no mask, no scale, and
no final round-back to the storage dtype.

The grouped-head path uses the default element budget with D=128,
S divisible by 8 and H divisible by 16. It reads adjacent heads together
and writes the reduced values in BHS layout.

Single-head fallback (DESIGN.md section 0.6):
    R1: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with the
        logical unit ``(b, h, s_block)`` unflattened from ``uid`` and a
        persistent in-core ``T.serial`` grid-stride loop guarded by
        ``if uid < num_logical`` (static bounds, host-folded constants).
    R2: UB budget driven residence tile ``(blk_s, 1, D)``;  ``blk_s`` is
        derived from ``dim`` and the element budget (10 declared B/element:
        stage 2 + acc32 4 + aux32 4).
    R3: the row reduction uses a three-dimensional flat declaration
        ``(blk_s, 1, D)`` with ``T.reduce_sum(dim=2)`` -- the 2-D
        "reduction axis == innermost axis" form is scalarised on this
        toolchain (DESIGN.md section 1.6.2 item 4).
    R4: explicit ``T.vcast(..., round_mode="rint")`` fp16/bf16 -> fp32
        upcasts; bf16 has no vector arithmetic, so the fp32 transit chain is
        mandatory.  The ``stage`` buffer carries ``o`` first and is then
        reused for ``do``.
    R5: tail blocks use the dynamic row bound ``real_s = min(blk_s, S - s0)``
        with explicit slices on both sides; padded rows never reach GM.
    R6: the source ``T.Parallel(blk, blk)`` element loop becomes
        ``T.vmul`` over the whole buffer.
    R7 (adopted optimisation O1): the product is written in place into the
        fp32 carrier of ``o`` (``T.vmul(acc32, aux32, acc32)``), bit-exact
        and saving one 4 B/element buffer.

Implementation note (Stage 3, forced by the toolchain): the frozen design's
final store ``T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])``
(region ``(real_s, 1, 1)`` into region ``(1, 1, real_s)``) is rejected by
the codegen (``bufferization.materialize_in_destination`` op source /
destination shapes are incompatible).  The reduction result is therefore
staged through a one-dimensional UB buffer (``mid1``) before the S-contiguous
GM store: ``T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])`` then
``T.copy(mid1[0:real_s], delta[bz, bx, s0:s0+real_s])``.  The maths,
the element set and the memory ordering are unchanged (an extra UB -> UB
copy of at most ``blk_s`` fp32 values per task).

Alignment note (Stage 3): the BishengIR reduction lowering materialises about
96 B of temporaries per tile element when the reduction axis is not a multiple
of the vector width (8), against roughly 12 B/element when it is (measured on
Ascend910B3 + CANN 9.0.0 by back-solving the "ub overflow" message).  The
factory therefore caps the effective element budget per alignment band; the
manifest shapes ``D in {64, 128}`` always sit in the cheap band and keep the
default 4096-element budget untouched.

Factory contract (source interface unchanged):
    _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype,
                                     blk_elems=4096) -> compiled kernel
    main(o, do) -> delta          (out_idx=[2], delta auto-allocated)

Run the embedded hierarchical tests:
    python _flashattn_bwd_preprocess_kernel.py --level L0
    python _flashattn_bwd_preprocess_kernel.py --level all
"""

import os

# Developer mode: T.alloc_shared maps to UB and the compiler inserts all
# synchronisation automatically (DESIGN.md sections 2.1 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse  # noqa: E402
import time  # noqa: E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

# Programming mode actually verified in Stage 3 (Stage 5 syncs the wrapper's
# Kernel.ascend_mode from this literal).
ASCEND_MODE = "Developer"

__all__ = [
    "ASCEND_MODE",
    "_flashattn_bwd_preprocess_kernel",
    "golden_flashattn_bwd_preprocess",
]


# ---------------------------------------------------------------------------
# Constants (DESIGN.md sections 4.3 / 4.5 / 1.6.3)
# ---------------------------------------------------------------------------
# Unified Buffer capacity per Vector core (queried below, 196608 B on 910B3).
_UB_CAPACITY_BYTES = 196_608

# Conservative persistent-unit element budget: min of the two budgeting
# accounts in DESIGN.md section 4.5 (declared x multi-buffer inflation, and
# declared + reduction-chain lowering unit price).
_UB_ELEMS_BUDGET = 4_096

# Declared bytes per element: stage (input dtype, 2 B) + acc32 (4 B) +
# aux32 (4 B); dlta / mid1 are O(blk_s) and negligible.
_UB_DECLARED_BYTES_PER_ELEM = 10

# Auto multi-buffer inflation factor of the declared buffers (conservative).
_UB_MULTIBUFFER_FACTOR = 2

# ---------------------------------------------------------------------------
# Measured UB cost of the reduction chain (Ascend910B3 + CANN 9.0.0, this
# toolchain, Stage 3 back-solved from the BishengIR
# "ub overflow, requires N bits while 1572864 bits available" message):
# the lowering materialises ~96 B per tile element when the reduction axis is
# NOT a multiple of the fp16/bf16 vector width (8), versus ~11-12 B/element
# when it is (declared buffers included).  Only the manifest shapes
# D in {64, 128} (both multiples of 8) are in the fast band; the unit element
# budget is capped for the remaining (defensive) shapes so that they still
# compile, at the price of more in-core rounds.
# ---------------------------------------------------------------------------
_UB_REDUCE_B_PER_ELEM_ALIGNED = 12.0
_UB_REDUCE_B_PER_ELEM_UNALIGNED = 96.5
_UB_BUDGET_SAFETY = 0.85

# Supported input dtypes (the source op only accepts fp16 / bf16).
_SUPPORTED_DTYPES = ("float16", "bfloat16")

# fp16/bf16 vector width in elements (alignment granularity of the tail axis).
_VECTOR_WIDTH = 8

# Precision tolerances for delta (DESIGN.md section 8.2): delta is an fp32
# intermediate whose only error source is the association order of the D-term
# fp32 sum.  Both input dtypes share the same gate because the upcast and the
# product are bit-exact.
_TOLERANCE = {
    "float16": (torch.float16, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-4, 1e-4),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_flashattn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Row-wise ``delta = rowsum(o * do)`` reference (DESIGN.md section 8.1).

    Parameters
    ----------
    o, do : torch.Tensor
        ``(B, S, H, D)`` fp16/bf16 (BSHD).  Device independent: CPU is fine.

    Returns
    -------
    torch.Tensor
        ``(B, H, S)`` fp32.

    Independent of the NPU algorithm: no persistent structure, no flat
    three-dimensional declaration, no explicit fp32 transit, no in-place
    product.  ``torch.sum(..., dim=-1)`` uses torch's own reduction order,
    which differs from the kernel's reduction tree -- that difference is
    exactly the tolerance source validated in DESIGN.md section 1.6.1 P2.

    The reference is written with explicit ``torch.*`` namespace calls
    (``torch.mul`` / ``torch.sum`` / ``torch.transpose``) rather than the
    equivalent tensor methods, so that the module-level torch API usage is
    unambiguous to static gate checks.  The operator semantics and the
    resulting values are bit-identical to the tensor-method form.
    """
    o32 = o.detach().float()
    do32 = do.detach().float()
    prod = torch.mul(o32, do32)
    reduced = torch.sum(prod, dim=-1)
    return torch.transpose(reduced, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _tiled_preprocess(batch, heads, seq_len, dim, dtype):
    """Build the grouped-head preprocess kernel.

    Read a contiguous group of heads at each sequence position. After
    reducing over D, transpose the row sums in UB for an S-contiguous store.
    """
    seq_tile = 8
    head_tile = 16
    head_blocks = heads // head_tile
    seq_blocks = seq_len // seq_tile
    tasks = batch * seq_blocks * head_blocks
    cores = min(tasks, NPUUtils.get().get_aivector_core_num())
    rounds = (tasks + cores - 1) // cores

    @T.prim_func
    def flash_bwd_prep(
        o: T.Tensor((batch, seq_len, heads, dim), dtype),
        do: T.Tensor((batch, seq_len, heads, dim), dtype),
        delta: T.Tensor((batch, heads, seq_len), "float32"),
    ):
        with T.Kernel(cores, is_npu=True) as (core, _):
            shape = (seq_tile, head_tile, dim)
            data = T.alloc_shared(shape, dtype)
            out = T.alloc_shared(shape, "float32")
            grad = T.alloc_shared(shape, "float32")
            summed = T.alloc_shared((seq_tile, head_tile, 1), "float32")
            transposed = T.alloc_shared((head_tile, seq_tile, 1), "float32")
            result = T.alloc_shared((head_tile, seq_tile), "float32")

            for i in T.Pipelined(rounds, num_stages=1):
                task = i * cores + core
                if task < tasks:
                    b = task // (seq_blocks * head_blocks)
                    s = task // head_blocks % seq_blocks * seq_tile
                    h = task % head_blocks * head_tile
                    T.copy(o[b, s : s + seq_tile, h : h + head_tile, :], data)
                    T.vcast(data, out, round_mode="rint")
                    T.copy(do[b, s : s + seq_tile, h : h + head_tile, :], data)
                    T.vcast(data, grad, round_mode="rint")
                    T.vmul(out, grad, out)
                    T.reduce_sum(out, summed, dim=2)
                    T.transpose(summed, transposed, permutation=[1, 0, 2])
                    T.reshape(transposed, result)
                    T.copy(result, delta[b, h : h + head_tile, s : s + seq_tile])

    return flash_bwd_prep


@tilelang.jit(out_idx=[2], target="npuir")
def _flashattn_bwd_preprocess_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dtype: str,
    blk_elems: int = _UB_ELEMS_BUDGET,
):
    """Build the FlashAttention backward preprocess kernel (NPU, npuir).

    The first five positional parameters keep the source factory signature
    (``batch, heads, seq_len, dim, dtype``); ``blk_elems`` is an optional
    element budget used by the Stage 4 parameter-level A/B study
    (DESIGN.md sections 1.6.3 / 4.5) and does not affect the interface.

    ``delta`` is the third parameter, hence ``out_idx=[2]``: the runtime
    allocates ``(batch, heads, seq_len)`` float32 automatically, exactly as
    the source contract does.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if min(batch, heads, seq_len, dim) < 1:
        raise ValueError(
            f"batch/heads/seq_len/dim must be positive, got {(batch, heads, seq_len, dim)}"
        )
    if blk_elems < 1:
        raise ValueError(f"blk_elems must be positive, got blk_elems={blk_elems}")

    if blk_elems == _UB_ELEMS_BUDGET and dim == 128 and seq_len % 8 == 0 and heads % 16 == 0:
        return _tiled_preprocess(batch, heads, seq_len, dim, dtype)

    # ---- host-side compile-time constants (folded to IntImm while tracing) --
    # UB guards (DESIGN.md sections 4.5 / 5.4 / 9.3).  The declared buffers
    # alone (x multi-buffer inflation) bound the requested budget; the
    # reduction chain then bounds the *effective* budget, with an
    # alignment-dependent unit price (see the constants above).  The second
    # bound is the automatic form of the DESIGN.md section 4.5 ladder.
    if blk_elems * _UB_DECLARED_BYTES_PER_ELEM * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"blk_elems={blk_elems} needs more than the declared-buffer UB budget "
            f"({_UB_CAPACITY_BYTES} B); lower blk_elems (DESIGN.md section 4.5 ladder)"
        )

    if dim % _VECTOR_WIDTH == 0 or dim == 1:
        reduce_b_per_elem = _UB_REDUCE_B_PER_ELEM_ALIGNED
    else:
        reduce_b_per_elem = _UB_REDUCE_B_PER_ELEM_UNALIGNED
    elem_budget_cap = int(_UB_CAPACITY_BYTES * _UB_BUDGET_SAFETY / reduce_b_per_elem)
    eff_elems = min(blk_elems, elem_budget_cap)

    raw = max(1, eff_elems // dim)  # resident elements per single row budget
    blk_s = raw if raw < 8 else (raw // 8) * 8  # >=8: align to vector; <8: keep capacity
    nblk = (seq_len + blk_s - 1) // blk_s
    num_logical = batch * heads * nblk
    # Pure Vector operator: the physical AI cores are doubled for vector work.
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

    # Defensive guard (unreachable by construction, kept as a contract check).
    if blk_s * dim > eff_elems:
        raise ValueError(
            f"unit tile {blk_s}x{dim} exceeds the effective element budget {eff_elems}"
        )

    @T.prim_func
    def flash_bwd_prep(
        o: T.Tensor((batch, seq_len, heads, dim), dtype),  # type: ignore
        do: T.Tensor((batch, seq_len, heads, dim), dtype),  # type: ignore
        delta: T.Tensor((batch, heads, seq_len), "float32"),  # out_idx=[2]
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            stage = T.alloc_shared((blk_s, 1, dim), dtype)  # o load, then do load
            acc32 = T.alloc_shared((blk_s, 1, dim), "float32")  # o carrier -> product
            aux32 = T.alloc_shared((blk_s, 1, dim), "float32")  # do carrier
            dlta = T.alloc_shared((blk_s, 1, 1), "float32")  # row reduction result
            mid1 = T.alloc_shared((blk_s,), "float32")  # store staging (1-D segment)

            for i in T.serial(num_local_tasks):
                uid = i * num_kernels + cid
                if uid < num_logical:
                    bz = uid // (heads * nblk)
                    rem = uid % (heads * nblk)
                    bx = rem // nblk
                    by = rem % nblk
                    s0 = by * blk_s
                    real_s = T.min(blk_s, seq_len - s0)

                    T.copy(o[bz, s0 : s0 + real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])
                    T.vcast(stage, acc32, round_mode="rint")  # bit-exact upcast
                    T.copy(do[bz, s0 : s0 + real_s, bx, 0:dim], stage[0:real_s, 0:1, 0:dim])
                    T.vcast(stage, aux32, round_mode="rint")  # stage reused
                    T.vmul(acc32, aux32, acc32)  # in-place product (O1)
                    T.reduce_sum(acc32, dlta, dim=2)  # (blk_s,1,D) -> (blk_s,1,1)
                    # 1-D transit so that the store source and the S-contiguous
                    # GM region have compatible shapes (see module docstring).
                    T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])
                    T.copy(mid1[0:real_s], delta[bz, bx, s0 : s0 + real_s])

    return flash_bwd_prep


# ---------------------------------------------------------------------------
# Precision comparing helper + hierarchical testing
# ---------------------------------------------------------------------------
# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _build(batch, heads, seq_len, dim, dtype_str, blk_elems=None):
    """Build (or fetch from the JIT cache) the kernel for one shape/dtype.

    ``blk_elems=None`` uses the factory default (DESIGN.md section 4.5), which
    is what the wrapper does; an explicit value drives the L0/L1 ladder.
    """
    if blk_elems is None:
        return _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype_str)
    return _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype_str, blk_elems)


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between ``out`` and ``ref``.

    NaN/Inf aware: positions where both are NaN count as equal when
    ``equal_nan`` is set, and equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _run_case(
    batch,
    heads,
    seq_len,
    dim,
    dtype_str,
    tag="L0",
    blk_elems=None,
    o=None,
    do=None,
    atol=None,
    rtol=None,
    equal_nan=False,
):
    """Run one kernel case and compare against the golden.

    Inputs are generated on CPU and moved to the NPU (device-independent data
    generation); special-value cases pass pre-built CPU tensors via ``o`` /
    ``do``.  Returns (max_diff, violations) and raises AssertionError on any
    tolerance violation.
    """
    torch_dtype, def_atol, def_rtol = _TOLERANCE[dtype_str]
    atol = def_atol if atol is None else atol
    rtol = def_rtol if rtol is None else rtol

    if o is None:
        o = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)
        do = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)
    else:
        assert o.dtype == torch_dtype, f"input dtype {o.dtype} != expected {torch_dtype}"
        assert do.dtype == torch_dtype, f"input dtype {do.dtype} != expected {torch_dtype}"
        assert o.shape == (batch, seq_len, heads, dim), f"o shape {tuple(o.shape)}"
        assert do.shape == (batch, seq_len, heads, dim), f"do shape {tuple(do.shape)}"

    kernel = _build(batch, heads, seq_len, dim, dtype_str, blk_elems)
    out = kernel(o.npu(), do.npu()).cpu()

    ref = golden_flashattn_bwd_preprocess(o, do)
    assert out.dtype == torch.float32, f"delta dtype {out.dtype} != float32"
    assert out.shape == ref.shape, f"delta shape {tuple(out.shape)} != {tuple(ref.shape)}"

    max_diff, violations = _compare(out, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff and max_diff >= 0.0:
        best = _STATS_MAX_DIFF.get(dtype_str, -1.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    if violations:
        raise AssertionError(
            f"shape=({batch},{seq_len},{heads},{dim}) dtype={dtype_str} "
            f"blk_elems={blk_elems} max_diff={max_diff:.3e} "
            f"violations={violations}/{out.numel()} atol={atol} rtol={rtol}"
        )
    print(
        f"[{tag}] PASS: (B,S,H,D)=({batch},{seq_len},{heads},{dim}) dtype={dtype_str} "
        f"blk_elems={blk_elems} max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case`` collecting (not re-raising) per-case failures."""
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_iface_case(failures):
    """L0-14: interface contract (5 positional args + out_idx=[2])."""
    case_id = "L0-14/interface-contract"
    try:
        batch, heads, seq_len, dim, dtype_str = 1, 4, 33, 64, "float16"
        torch_dtype = torch.float16
        o = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)
        do = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)

        # Wrapper call form: exactly five positional arguments.
        kernel = _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype_str)
        delta = kernel(o.npu(), do.npu()).cpu()
        assert delta.shape == (batch, heads, seq_len), (
            f"out_idx allocation shape {tuple(delta.shape)} != {(batch, heads, seq_len)}"
        )
        assert delta.dtype == torch.float32, f"out_idx dtype {delta.dtype}"
        ref = golden_flashattn_bwd_preprocess(o, do)
        max_diff, violations = _compare(delta, ref, 1e-4, 1e-4)
        assert violations == 0, f"max_diff={max_diff:.3e} violations={violations}"
        print(
            f"[L0-14] PASS: 5-positional-arg call + out_idx=[2] -> "
            f"{tuple(delta.shape)} float32, max_diff={max_diff:.3e}"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def _run_ub_ladder(failures):
    """L0-10: UB element-budget ladder on the representative smoke shape.

    The design default (4096) and the any-price fallback (2048) are gating;
    the stretch factor (8192, needs reduction-chain unit price <= 24 B/elem)
    is recorded only, since a compile-time ``ub overflow`` there is expected
    data for the Stage 4 back-solve, not a correctness failure.
    """
    batch, heads, seq_len, dim, dtype_str = 1, 1024, 8, 64, "float16"
    torch_dtype = torch.float16
    o = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)
    do = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32).to(torch_dtype)

    stretch = 8192
    try:
        _run_case(
            batch,
            heads,
            seq_len,
            dim,
            dtype_str,
            tag="L0-10/stretch",
            blk_elems=stretch,
            o=o,
            do=do,
        )
        print(f"[L0-10] recorded: blk_elems={stretch} compiles and is correct")
    except Exception as exc:  # noqa: BLE001
        print(f"[L0-10] recorded: blk_elems={stretch} rejected ({exc!r})")

    for blk_elems in (4096, 2048):
        _try_case(
            failures,
            f"L0-10/ladder-{blk_elems}",
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim,
            dtype_str=dtype_str,
            tag=f"L0-10/ladder{blk_elems}",
            blk_elems=blk_elems,
            o=o,
            do=do,
        )


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, all 14 items).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        # id, (B, S, H, D), dtype, tag
        ("L0-1/smoke-fp16", (1, 1024, 8, 64), "float16", "L0-1"),
        ("L0-2/smoke-bf16", (1, 1024, 8, 64), "bfloat16", "L0-2"),
        ("L0-3/full-fp16-large", (16, 2048, 16, 128), "float16", "L0-3"),
        ("L0-4/full-bf16", (4, 4096, 16, 128), "bfloat16", "L0-4"),
        ("L0-5/tail-1", (1, 1025, 8, 64), "float16", "L0-5"),
        ("L0-6/tail-bf16", (2, 1000, 3, 128), "bfloat16", "L0-6"),
        ("L0-7/dim63", (1, 64, 4, 63), "float16", "L0-7"),
        ("L0-8/dim1", (1, 64, 4, 1), "float16", "L0-8"),
        ("L0-9/dim4096", (1, 512, 2, 4096), "float16", "L0-9"),
    ]
    for case_id, (b, s, h, d), dtype_str, tag in cases:
        _try_case(
            failures,
            case_id,
            batch=b,
            heads=h,
            seq_len=s,
            dim=d,
            dtype_str=dtype_str,
            tag=tag,
        )

    _run_ub_ladder(failures)

    # L0-11 zeros: delta must be bit-exact +0.0 on both sides.
    _try_case(
        failures,
        "L0-11/zeros",
        batch=1,
        heads=4,
        seq_len=128,
        dim=64,
        dtype_str="float16",
        tag="L0-11",
        o=torch.zeros(1, 128, 4, 64, dtype=torch.float16),
        do=torch.zeros(1, 128, 4, 64, dtype=torch.float16),
        atol=0.0,
        rtol=0.0,
    )
    # L0-12 constant ones: delta == D exactly (64.0 in fp32).
    _try_case(
        failures,
        "L0-12/ones",
        batch=1,
        heads=4,
        seq_len=128,
        dim=64,
        dtype_str="float16",
        tag="L0-12",
        o=torch.full((1, 128, 4, 64), 1.0, dtype=torch.float16),
        do=torch.full((1, 128, 4, 64), 1.0, dtype=torch.float16),
        atol=0.0,
        rtol=0.0,
    )
    # L0-13 inf / NaN IEEE propagation, row isolation.
    o13 = torch.randn(1, 16, 2, 128, dtype=torch.float32).to(torch.float16)
    do13 = torch.randn(1, 16, 2, 128, dtype=torch.float32).to(torch.float16)
    o13[0, 0, 0, :] = float("inf")
    do13[0, 0, 0, :] = 1.0
    o13[0, 1, 0, :] = float("nan")
    o13[0, 2, 1, :] = float("inf")
    do13[0, 2, 1, :] = -1.0
    _try_case(
        failures,
        "L0-13/inf-nan",
        batch=1,
        heads=2,
        seq_len=16,
        dim=128,
        dtype_str="float16",
        tag="L0-13",
        o=o13,
        do=do13,
        equal_nan=True,
    )

    _run_iface_case(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(c for c, _ in failures))
    else:
        print("[L0] ALL PASS (14 items)")
    return failures


def run_L1():
    """L1: full functional coverage (DESIGN.md section 8.3 "L1 suggestions").

    Covers the UB element-budget scan, the degenerate ``blk_s < 8`` shapes,
    ``seq_len < blk_s``, single-logical-task grids, non-multiple-of-40 grids
    (persistent guard path) and the factory contract guards.

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        # blk_elems scan on a tail-carrying bf16 shape (300 = 32*9 + 12).
        ("L1/bf16/budget8192", (2, 300, 4, 128), "bfloat16", 8192),
        ("L1/bf16/budget4096", (2, 300, 4, 128), "bfloat16", 4096),
        ("L1/bf16/budget2048", (2, 300, 4, 128), "bfloat16", 2048),
        # Degenerate blk_s < vector width (D = 1024 -> blk_s = 4; D = 2048 -> 2).
        ("L1/fp16/blk_s4", (1, 64, 2, 1024), "float16", None),
        ("L1/fp16/blk_s2", (1, 64, 2, 2048), "float16", None),
        # seq_len < blk_s (single sequence block per (b, h)).
        ("L1/fp16/seq-lt-blk", (1, 10, 4, 64), "float16", None),
        # Single logical task (num_kernels == num_logical == 1).
        ("L1/fp16/single-task", (1, 7, 1, 64), "float16", None),
        ("L1/bf16/single-task", (1, 5, 1, 128), "bfloat16", None),
        # num_logical not a multiple of the 40 vector cores (guard path).
        ("L1/fp16/guard-195", (3, 777, 5, 64), "float16", None),
        ("L1/bf16/guard-385", (5, 333, 7, 128), "bfloat16", None),
        # heads == 1 and small batch (delta H axis of length 1).
        ("L1/fp16/heads1", (2, 100, 1, 64), "float16", None),
        ("L1/fp16/batch1", (1, 129, 8, 64), "float16", None),
    ]
    for case_id, (b, s, h, d), dtype_str, blk_elems in cases:
        _try_case(
            failures,
            case_id,
            batch=b,
            heads=h,
            seq_len=s,
            dim=d,
            dtype_str=dtype_str,
            tag="L1",
            blk_elems=blk_elems,
        )

    # Factory contract guards.
    guard_cases = [
        ("L1/guard-dtype", (1, 8, 4, 64, "float32", None), ValueError),
        ("L1/guard-batch0", (0, 8, 4, 64, "float16", None), ValueError),
        ("L1/guard-dim0", (1, 8, 4, 0, "float16", None), ValueError),
        ("L1/guard-blk_elems0", (1, 8, 4, 64, "float16", 0), ValueError),
        # Needs 16384 * 10 B * x2 > 196608 B -> the declared-buffer UB guard fires.
        ("L1/guard-ub-budget", (1, 8, 4, 64, "float16", 16384), ValueError),
    ]
    for case_id, (b, h, s, d, dtype_str, blk_elems), exc_type in guard_cases:
        try:
            _flashattn_bwd_preprocess_kernel(
                b, h, s, d, dtype_str, blk_elems
            ) if blk_elems is not None else _flashattn_bwd_preprocess_kernel(b, h, s, d, dtype_str)
        except exc_type as exc:
            print(f"[L1] PASS: {case_id} rejected ({exc})")
        except Exception as exc:  # noqa: BLE001
            failures.append((case_id, exc))
            print(f"[L1] {case_id} raised the wrong exception: {exc!r}")
        else:
            failures.append((case_id, AssertionError(f"{case_id} not rejected")))
            print(f"[L1] {case_id} was NOT rejected")

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(c for c, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme shapes (warn only, non-blocking)."""
    cases = [
        (1, 3, 1, 3, "bfloat16"),
        (2, 5, 3, 7, "float16"),
        (1, 8192, 4, 128, "float16"),
        (1, 257, 8, 256, "float16"),
        (8, 17, 2, 64, "float16"),
        (1, 64, 2, 4096, "float16"),
        (16, 33, 3, 64, "bfloat16"),
    ]
    for b, s, h, d, dtype_str in cases:
        try:
            _run_case(
                batch=b,
                heads=h,
                seq_len=s,
                dim=d,
                dtype_str=dtype_str,
                tag="L2",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record only): (B,S,H,D)=({b},{s},{h},{d}) {dtype_str}: {exc!r}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    batch, heads, seq_len, dim = 1, 4, 32, 128

    # fp16 saturation: products up to 4.29e9, fp32 accumulation holds them.
    try:
        _run_case(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim,
            dtype_str="float16",
            tag="Boundary-saturate",
            o=torch.full((batch, seq_len, heads, dim), 65504.0, dtype=torch.float16),
            do=torch.full((batch, seq_len, heads, dim), 65504.0, dtype=torch.float16),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): saturate: {exc!r}")

    # fp16 subnormal magnitudes (~1e-5).
    try:
        _run_case(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim,
            dtype_str="float16",
            tag="Boundary-subnormal",
            o=(torch.randn(batch, seq_len, heads, dim, dtype=torch.float32) * 1e-5).to(
                torch.float16
            ),
            do=(torch.randn(batch, seq_len, heads, dim, dtype=torch.float32) * 1e-5).to(
                torch.float16
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {exc!r}")

    # Mixed-sign cancellation at large magnitude (worst case for the sum order).
    try:
        o_b = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32) * 256.0
        do_b = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32) * 256.0
        _run_case(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim,
            dtype_str="float16",
            tag="Boundary-cancel",
            o=o_b.to(torch.float16),
            do=do_b.to(torch.float16),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): cancellation: {exc!r}")

    # inf / NaN natural IEEE propagation (rows must stay isolated).
    try:
        o_b = torch.randn(2, 16, 4, 128, dtype=torch.float32).to(torch.float16)
        do_b = torch.randn(2, 16, 4, 128, dtype=torch.float32).to(torch.float16)
        o_b[0, 0, 1, :] = float("inf")
        o_b[0, 3, 2, :] = float("nan")
        do_b[1, 7, 0, :] = float("inf")
        _run_case(
            batch=2,
            heads=4,
            seq_len=16,
            dim=128,
            dtype_str="float16",
            tag="Boundary-inf-nan",
            o=o_b,
            do=do_b,
            equal_nan=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): inf-nan: {exc!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(c for c, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
