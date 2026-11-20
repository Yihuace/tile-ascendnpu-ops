# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""FlashAttention backward postprocess kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_flashattn_bwd_postprocess_kernel``
(``/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`` lines 106-128,
carried by ``FlashAttnBwdPostprocessKernel`` in TileOPs
``MultiHeadAttentionBwdOp``) to TileLang ``target="npuir"``.

Semantics (DESIGN.md sections 0.1 / 1.3, frozen):
    dq_out[b, s, h, d] = RNE_storage_dtype(dq[b, s, h, d])
    dq     : (B, S, H, D) float32              fp32 accumulator of the main kernel
    dq_out : (B, S, H, D) float16 / bfloat16   out_idx=[1], auto-allocated

The mapping is element-wise and one-to-one (identity index map): no reduction,
no scale, no mask, no broadcast.  The only numerical behaviour is the IEEE 754
round-to-nearest-even (ties-to-even) down-conversion, which is machine verified
to be bit-identical to the source implicit ``T.copy`` cast (DESIGN.md section
1.6.1 O1, ``verify_equiv.py``: 12471 cases x 2 dtypes, 0 violations).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial(num_local_tasks)`` grid-stride loop
        guarded by ``if uid < num_logical`` (static bounds, host-folded
        constants).
    R2: UB budget driven residence tile ``(blk_s, heads, dim)`` -- the whole
        H axis is folded into the tile so that ``dq[bz, s0:s0+rs, 0:H, 0:D]``
        is a single contiguous slab (maximal DMA burst) and the number of
        logical units is H times smaller than a per-(b, h) split.
    R3: the source single cross-dtype ``T.copy`` becomes an explicit
        ``T.copy`` (GM -> UB fp32) + ``T.vcast(..., round_mode="rint")``
        (fp32 -> storage dtype) + ``T.copy`` (UB -> GM) chain.  Same vector
        pass count, but the rounding mode is explicit and auditable.
    Layout: the source ``make_dq_layout`` annotation is DROPPED (DESIGN.md
        section 0.5 decision 1).  It is an Ampere 8x8 mma-fragment physical
        contract shared with the main kernel; the NPU main kernel writes
        ``dq`` in natural BSHD (per-row 1-D row atomics, no layout
        annotation), so keeping the annotation would silently permute the
        elements (``dq_out[i] = cast(dq[pi(i)])``, no error, no NaN).
        Dropping it restores the identity index map.
    Tail blocks use the dynamic row bound ``rs = T.min(blk_s, seq_len - s0)``
    constraining all four sides (GM source slice, UB source slice, UB
    destination slice, GM destination slice) -- no padding, no out-of-range
    write.
    ``threads=`` is dropped (no effect on npuir).

Factory contract (source interface unchanged):
    _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype,
                                      blk_elems=16384) -> compiled kernel
    main(dq) -> dq_out          (out_idx=[1], dq_out auto-allocated)

Run the embedded hierarchical tests:
    python _flashattn_bwd_postprocess_kernel.py --level L0
    python _flashattn_bwd_postprocess_kernel.py --level all
"""

import os

# Programming mode verified in Stage 3 (Stage 5 syncs the wrapper's
# Kernel.ascend_mode from this literal).  Developer mode: T.alloc_shared maps
# to UB and the compiler inserts all synchronisation automatically
# (DESIGN.md sections 2.1 and 7).
ASCEND_MODE = "Developer"
os.environ.setdefault("TILELANG_ASCEND_MODE", ASCEND_MODE)

import argparse  # noqa: E402
import time  # noqa: E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

__all__ = [
    "ASCEND_MODE",
    "_flashattn_bwd_postprocess_kernel",
    "golden_flashattn_bwd_postprocess",
]


# ---------------------------------------------------------------------------
# Constants (DESIGN.md sections 4.3 / 4.5 / 1.6.3)
# ---------------------------------------------------------------------------
# Unified Buffer capacity per Vector core (queried in Stage 3, 196608 B on
# Ascend910B3; DESIGN.md section 5.5 records the same live value).
_UB_CAPACITY_BYTES = 196_608

# Element budget of one residence tile.  blk_s is derived from it and from the
# row length (heads * dim); the design default is 16384 (DESIGN.md section 4.5
# default band, 6 declared B/element x 16384 = 96 KB declared).
_UB_ELEMS_BUDGET = 16_384

# Declared bytes per residence element: src32 (float32, 4 B) + dst16
# (storage dtype, 2 B).  The two buffers cannot alias: T.vcast requires two
# distinct same-shape buffers (DESIGN.md section 4.3).
_UB_DECLARED_BYTES_PER_ELEM = 6

# Auto multi-buffer inflation factor of the declared buffers (DESIGN.md
# section 4.5, default band, conservative).
_UB_MULTIBUFFER_FACTOR = 1.7

# Supported storage dtypes (the source op accepts fp16 / bf16 only).
_SUPPORTED_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation) -- torch.* namespace calls
# ---------------------------------------------------------------------------
def golden_flashattn_bwd_postprocess(dq: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Element-wise RNE down-conversion reference (DESIGN.md section 8.1).

    Parameters
    ----------
    dq : torch.Tensor
        ``(B, S, H, D)`` float32.  Device independent: CPU is fine.
    dtype : torch.dtype
        ``torch.float16`` or ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        ``(B, S, H, D)`` in ``dtype``; element ``i`` equals
        ``RNE_dtype(dq[i])`` with the identity index map.

    Independent of the NPU algorithm: no persistent structure, no UB tile, no
    explicit ``vcast``, no ``rs`` tail bound; the source semantics (a single
    IEEE RNE down-conversion) is the only input.

    The reference is written with explicit ``torch.*`` namespace calls
    (``torch.Tensor.to`` / ``torch.float32``) rather than the equivalent
    tensor-method form ``dq.to(dtype)``, so that the module-level torch API
    usage is unambiguous to static gate checks.  The values are bit-identical
    to the tensor-method form (verified with ``torch.equal`` for both fp16 and
    bf16 on random inputs).
    """
    x32 = torch.Tensor.to(dq.detach(), torch.float32)  # already fp32; view/copy
    return torch.Tensor.to(x32, dtype)  # single RNE round -> storage dtype


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dtype: str,
    blk_elems: int = _UB_ELEMS_BUDGET,
):
    """Build the FlashAttention backward postprocess kernel (NPU, npuir).

    The first five positional parameters keep the source factory signature
    (``batch, heads, seq_len, dim, dtype``); ``blk_elems`` is an optional
    element budget driving the Stage 4 parameter-level A/B study
    (DESIGN.md sections 1.6.3 / 4.5) and does not affect the interface.

    ``dq_out`` is the second parameter, hence ``out_idx=[1]``: the runtime
    allocates ``(batch, seq_len, heads, dim)`` in the storage dtype
    automatically, exactly as the source contract does.
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

    # Declared-buffer UB guard (DESIGN.md section 4.5, x1.7 inflation band).
    if blk_elems * _UB_DECLARED_BYTES_PER_ELEM * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"blk_elems={blk_elems} needs more than the declared-buffer UB budget "
            f"({_UB_CAPACITY_BYTES} B); lower blk_elems (DESIGN.md section 4.5 ladder)"
        )

    # ---- host-side compile-time constants (folded to IntImm while tracing) --
    row = heads * dim  # contiguous elements per S row (H folded into the tile)
    # blk_s from the element budget; max(1, ...) also covers row > blk_elems
    # (then blk_s == 1 and one whole S row is the unit tile).
    blk_s = max(1, min(seq_len, blk_elems // row))
    nblk = (seq_len + blk_s - 1) // blk_s
    num_logical = batch * nblk
    # Pure Vector operator: the physical AI cores are doubled for vector work.
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
    num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

    # Defensive UB guard on the *actual* residence tile (unreachable in the
    # manifest domain, kept as a contract check for H*D > blk_elems shapes).
    if blk_s * row * _UB_DECLARED_BYTES_PER_ELEM * _UB_MULTIBUFFER_FACTOR > _UB_CAPACITY_BYTES:
        raise ValueError(
            f"residence tile ({blk_s}, {heads}, {dim}) = {blk_s * row} elements "
            f"exceeds the declared UB budget ({_UB_CAPACITY_BYTES} B); lower the "
            "shape or blk_elems (DESIGN.md section 9.3)"
        )

    @T.prim_func
    def flash_bwd_post(
        dq: T.Tensor((batch, seq_len, heads, dim), "float32"),  # type: ignore
        dq_out: T.Tensor((batch, seq_len, heads, dim), dtype),  # type: ignore, out_idx=[1]
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            src32 = T.alloc_shared((blk_s, heads, dim), "float32")  # GM -> UB transit
            dst16 = T.alloc_shared((blk_s, heads, dim), dtype)  # cast result

            for i in T.serial(num_local_tasks):  # static bound (host-folded)
                uid = i * num_kernels + cid  # grid-stride map
                if uid < num_logical:
                    bz = uid // nblk  # batch index
                    s0 = (uid % nblk) * blk_s  # S axis origin
                    rs = T.min(blk_s, seq_len - s0)  # tail rows (all four sides)

                    # 1. GM -> UB (fp32, single contiguous slab).
                    T.copy(
                        dq[bz, s0 : s0 + rs, 0:heads, 0:dim],
                        src32[0:rs, 0:heads, 0:dim],
                    )
                    # 2. Element-wise IEEE RNE cast (fp32 -> storage dtype),
                    #    vectorised over the whole buffer; the source and the
                    #    destination dtype differ (never a same-dtype vcast).
                    T.vcast(src32, dst16, round_mode="rint")
                    # 3. UB -> GM (storage dtype, same slab, same extents).
                    T.copy(
                        dst16[0:rs, 0:heads, 0:dim],
                        dq_out[bz, s0 : s0 + rs, 0:heads, 0:dim],
                    )

    return flash_bwd_post


# ---------------------------------------------------------------------------
# Precision comparing helpers + hierarchical testing
# ---------------------------------------------------------------------------
# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}

# Corner values for the boundary semantic tests (DESIGN.md sections 0.1/8.3
# L0-12): +-inf, NaN, +-0, fp16 max normal, fp16 overflow threshold (65520),
# fp16 min normal, fp16 min subnormal and its half (RNE tie), fp16/bf16
# mantissa ties, fp32 extremes and fp32 subnormal magnitudes.
_CORNER_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    float("inf"),
    float("-inf"),
    float("nan"),
    65504.0,
    65519.0,
    65520.0,
    65536.0,
    -65504.0,
    -65520.0,
    6.103515625e-05,
    3.0517578125e-05,
    5.960464477539063e-08,
    2.9802322387695312e-08,
    1.0e-45,
    1.0 + 2.0**-11,
    1.0 + 2.0**-8,
    1.0 + 3.0 * 2.0**-11,
    1.5 + 2.0**-11,
    3.4028235e38,
    -3.4028235e38,
    1.0e-40,
    -1.0e-40,
    1.0e30,
]


def _build(batch, heads, seq_len, dim, dtype_str, blk_elems=None):
    """Build (or fetch from the JIT cache) the kernel for one shape/dtype.

    ``blk_elems=None`` uses the factory default (DESIGN.md section 4.5), which
    is what the wrapper does; an explicit value drives the L0/L1 ladder.
    """
    if blk_elems is None:
        return _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype_str)
    return _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype_str, blk_elems)


def _compare_bits(out, ref):
    """Return (value_mismatches, bit_mismatches, max_diff) for two equal-shaped
    storage-dtype tensors.

    The gate is the DESIGN.md section 8.2 main criterion: value equality with
    NaN positions equivalent (``torch.equal`` + ``equal_nan`` semantics).
    ``bit_mismatches`` (int16 view) is reported as a diagnostic only -- it also
    flags ``-0.0`` versus ``+0.0``, which the value criterion treats as equal.
    """
    assert out.dtype == ref.dtype, f"dtype {out.dtype} != {ref.dtype}"
    assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"

    both_nan = torch.isnan(out) & torch.isnan(ref)
    val_eq = (out == ref) | both_nan
    n_val = int((~val_eq).sum().item())

    n_bit = int((~(out.view(torch.int16) == ref.view(torch.int16)) & ~both_nan).sum().item())

    absdiff = (out.float() - ref.float()).abs()
    absdiff = torch.where(
        torch.isfinite(absdiff),
        absdiff,
        torch.full_like(absdiff, float("inf")),
    )
    absdiff = torch.where(val_eq, torch.zeros_like(absdiff), absdiff)
    max_diff = float(absdiff.max().item())
    return n_val, n_bit, max_diff


def _unique_input(batch, seq_len, heads, dim, torch_dtype, seed=0):
    """Injective per-element encoding on the *storage* dtype (DESIGN.md L0-6).

    Distinct finite normal bit patterns of ``torch_dtype`` are used as values,
    so the encoding stays injective after the down-conversion (the fp32 source
    value is the exact upcast of the storage-dtype value, hence casts back to
    the same bit pattern).  A ``idx/2**k`` encoding is NOT injective here:
    once ``|x|`` crosses a binade of the storage dtype, adjacent indices
    collapse to the same stored value.
    """
    n = batch * seq_len * heads * dim
    if torch_dtype == torch.float16:
        lo, hi = 0x0400, 0x7BFF  # fp16 positive normals (no inf / NaN)
    elif torch_dtype == torch.bfloat16:
        lo, hi = 0x0080, 0x7F7F  # bf16 positive normals (no inf / NaN)
    else:
        raise ValueError(f"unsupported dtype {torch_dtype}")
    span = hi - lo + 1
    assert n <= 2 * span, f"cannot encode {n} distinct values in {torch_dtype}"

    idx = torch.arange(n, dtype=torch.int64)
    code = lo + (idx // 2) % span
    sign = torch.where(idx % 2 == 0, torch.zeros_like(code), torch.full_like(code, 0x8000))
    vals = (code | sign).to(torch.int32).to(torch.int16).view(torch_dtype)
    assert torch.unique(vals).numel() == n, "encoding is not injective"
    vals32 = vals.to(torch.float32)
    vals32 = vals32[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]
    return vals32.reshape(batch, seq_len, heads, dim).contiguous()


def _corner_input(batch, seq_len, heads, dim, seed=0):
    """Tile the corner-value set over the whole tensor, then scramble it."""
    n = batch * seq_len * heads * dim
    vals = torch.tensor(_CORNER_VALUES, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int64) % vals.numel()
    out = vals[idx]
    out = out[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]
    return out.reshape(batch, seq_len, heads, dim).contiguous()


def _run_case(batch, heads, seq_len, dim, dtype_str, tag="L0", blk_elems=None, dq=None, seed=0):
    """Run one kernel case and compare bit-exactly against the golden.

    Inputs are generated on CPU and moved to the NPU (device-independent data
    generation); special-value cases pass a pre-built CPU tensor via ``dq``.
    Returns ``(max_diff, out)`` and raises AssertionError on any value
    mismatch.
    """
    torch_dtype = _TORCH_DTYPE[dtype_str]

    if dq is None:
        dq = torch.randn(
            batch,
            seq_len,
            heads,
            dim,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        assert dq.dtype == torch.float32, f"input dtype {dq.dtype} != float32"
        assert tuple(dq.shape) == (batch, seq_len, heads, dim), (
            f"input shape {tuple(dq.shape)} != {(batch, seq_len, heads, dim)}"
        )
    dq = dq.contiguous()

    kernel = _build(batch, heads, seq_len, dim, dtype_str, blk_elems)
    out = kernel(dq.npu()).cpu()
    ref = golden_flashattn_bwd_postprocess(dq, torch_dtype)

    assert out.dtype == torch_dtype, f"out dtype {out.dtype} != {torch_dtype}"
    assert tuple(out.shape) == (batch, seq_len, heads, dim), (
        f"out shape {tuple(out.shape)} != {(batch, seq_len, heads, dim)}"
    )

    n_val, n_bit, max_diff = _compare_bits(out, ref)
    best = _STATS_MAX_DIFF.get(dtype_str, -1.0)
    _STATS_MAX_DIFF[dtype_str] = max(best, max_diff)

    if n_val:
        raise AssertionError(
            f"(B,S,H,D)=({batch},{seq_len},{heads},{dim}) dtype={dtype_str} "
            f"blk_elems={blk_elems} value_mismatches={n_val}/{out.numel()} "
            f"bit_mismatches={n_bit} max_diff={max_diff:.6e}"
        )
    print(
        f"[{tag}] PASS: (B,S,H,D)=({batch},{seq_len},{heads},{dim}) dtype={dtype_str} "
        f"blk_elems={blk_elems} bit_mismatch={n_bit} max_diff={max_diff:.3e}"
    )
    return max_diff, out


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case`` collecting (not re-raising) per-case failures."""
    try:
        return _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
    return None


def _expect_reject(failures, case_id, kwargs, exc_type=ValueError):
    """Assert that the factory rejects an invalid configuration."""
    try:
        _flashattn_bwd_postprocess_kernel(**kwargs)
    except exc_type as exc:
        print(f"[L1] PASS: {case_id} rejected ({exc})")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[L1] {case_id} raised the wrong exception: {exc!r}")
    else:
        failures.append((case_id, AssertionError(f"{case_id} not rejected")))
        print(f"[L1] {case_id} was NOT rejected")


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, 14 items, all must pass).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # --- L0-1..L0-4: main domain, the four manifest geometry bands ---------
    main_domain = [
        ("L0-1/manifest-smoke", (1, 512, 8, 64)),
        ("L0-2/manifest-8b-short", (4, 512, 32, 128)),
        ("L0-3/manifest-8b-long", (2, 2048, 32, 128)),
        ("L0-4/manifest-70b-short", (2, 512, 64, 128)),
    ]
    for case_id, (b, s, h, d) in main_domain:
        for dtype_str in ("float16", "bfloat16"):
            _try_case(
                failures,
                f"{case_id}/{'fp16' if dtype_str == 'float16' else 'bf16'}",
                batch=b,
                heads=h,
                seq_len=s,
                dim=d,
                dtype_str=dtype_str,
                tag=case_id,
            )

    # --- L0-5: tile-budget invariance (DESIGN.md claim P3) -----------------
    budget_outputs = {}
    for blk_elems in (4096, 8192, 16384):
        res = _try_case(
            failures,
            f"L0-5/budget-{blk_elems}",
            batch=1,
            heads=8,
            seq_len=512,
            dim=64,
            dtype_str="float16",
            tag="L0-5",
            blk_elems=blk_elems,
        )
        if res is not None:
            budget_outputs[blk_elems] = res[1]
    if len(budget_outputs) == 3:
        keys = sorted(budget_outputs)
        base = budget_outputs[keys[0]]
        for key in keys[1:]:
            n_val, n_bit, _ = _compare_bits(budget_outputs[key], base)
            if n_val or n_bit:
                failures.append(
                    (
                        f"L0-5/invariance-{key}",
                        AssertionError(
                            f"blk_elems={key} vs {keys[0]}: value_mismatches={n_val} "
                            f"bit_mismatches={n_bit} (tile budget must not change values)"
                        ),
                    )
                )
                print(f"[collected] L0-5/invariance-{key} FAILED: {n_val} value mismatches")
            else:
                print(
                    f"[L0-5] PASS: blk_elems={key} output is bit-identical to blk_elems={keys[0]}"
                )
    else:
        failures.append(
            ("L0-5/compile", AssertionError("tile-budget invariance needs all three budgets"))
        )

    # --- L0-6: element coverage / identity index map (claim P2) ------------
    for dtype_str in ("float16", "bfloat16"):
        _try_case(
            failures,
            f"L0-6/identity-{dtype_str}",
            batch=1,
            heads=4,
            seq_len=64,
            dim=64,
            dtype_str=dtype_str,
            tag="L0-6",
            dq=_unique_input(1, 64, 4, 64, _TORCH_DTYPE[dtype_str]),
        )

    # --- L0-7: region integrity (tail block must not touch neighbours) ----
    # B=2 makes the batch-adjacent rows observable: an over-long tail write in
    # batch 0 would land on batch 1 rows 0..27 and show up as value mismatches.
    for dtype_str in ("float16", "bfloat16"):
        _try_case(
            failures,
            f"L0-7/region-integrity-{dtype_str}",
            batch=2,
            heads=4,
            seq_len=100,
            dim=64,
            dtype_str=dtype_str,
            tag="L0-7",
            dq=_unique_input(2, 100, 4, 64, _TORCH_DTYPE[dtype_str], seed=7),
        )

    # --- L0-8: tail blocks (seq_len not divisible by blk_s) ---------------
    for case_id, (b, s, h, d) in [
        ("L0-8/tail-513", (1, 513, 8, 64)),
        ("L0-8/tail-100", (1, 100, 8, 64)),
    ]:
        for dtype_str in ("float16", "bfloat16"):
            _try_case(
                failures,
                f"{case_id}/{'fp16' if dtype_str == 'float16' else 'bf16'}",
                batch=b,
                heads=h,
                seq_len=s,
                dim=d,
                dtype_str=dtype_str,
                tag="L0-8",
            )

    # --- L0-9: extreme H*D (blk_s clamping, 1-element rows) ---------------
    for case_id, (b, s, h, d) in [
        ("L0-9/hd-1", (1, 8, 1, 1)),
        ("L0-9/hd-64", (1, 8, 1, 64)),
        ("L0-9/hd-8192", (2, 4, 64, 128)),
    ]:
        for dtype_str in ("float16", "bfloat16"):
            _try_case(
                failures,
                f"{case_id}/{'fp16' if dtype_str == 'float16' else 'bf16'}",
                batch=b,
                heads=h,
                seq_len=s,
                dim=d,
                dtype_str=dtype_str,
                tag="L0-9",
            )

    # --- L0-10: UB element-budget ladder ---------------------------------
    for blk_elems in (4096, 8192, 16384):
        _try_case(
            failures,
            f"L0-10/ladder-{blk_elems}",
            batch=1,
            heads=8,
            seq_len=512,
            dim=64,
            dtype_str="bfloat16",
            tag="L0-10",
            blk_elems=blk_elems,
        )

    # --- L0-11: interface contract (5 positional args + out_idx=[1]) ------
    _run_iface_case(failures)

    # --- L0-12: corner values -------------------------------------------
    for dtype_str in ("float16", "bfloat16"):
        _try_case(
            failures,
            f"L0-12/corners-{dtype_str}",
            batch=1,
            heads=4,
            seq_len=64,
            dim=64,
            dtype_str=dtype_str,
            tag="L0-12",
            dq=_corner_input(1, 64, 4, 64, seed=3),
        )

    # --- L0-13: random magnitudes (normal / overflow / subnormal domains) -
    for dtype_str in ("float16", "bfloat16"):
        for dist_id, scale, offset in [
            ("normal", 1.0, 0.0),
            ("u4", 4.0, -4.0),
            ("overflow", 7.0e4 / 2.0, -7.0e4 / 2.0),
            ("subnormal", 1.0e-4, -1.0e-4),
        ]:
            gen = torch.Generator().manual_seed(11)
            raw = torch.rand(1, 128, 4, 64, dtype=torch.float32, generator=gen) * scale + offset
            if dist_id == "normal":
                raw = torch.randn(1, 128, 4, 64, dtype=torch.float32, generator=gen)
            _try_case(
                failures,
                f"L0-13/{dist_id}-{dtype_str}",
                batch=1,
                heads=4,
                seq_len=128,
                dim=64,
                dtype_str=dtype_str,
                tag="L0-13",
                dq=raw.contiguous(),
            )

    # --- L0-14: determinism (same input twice, bit-identical) ------------
    for dtype_str in ("float16", "bfloat16"):
        case_id = f"L0-14/determinism-{dtype_str}"
        dq = torch.randn(
            1,
            128,
            4,
            64,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(23),
        )
        first = _try_case(
            failures,
            f"{case_id}/first",
            batch=1,
            heads=4,
            seq_len=128,
            dim=64,
            dtype_str=dtype_str,
            tag="L0-14",
            dq=dq,
        )
        second = _try_case(
            failures,
            f"{case_id}/second",
            batch=1,
            heads=4,
            seq_len=128,
            dim=64,
            dtype_str=dtype_str,
            tag="L0-14",
            dq=dq,
        )
        if first is not None and second is not None:
            n_val, n_bit, _ = _compare_bits(first[1], second[1])
            if n_val or n_bit:
                failures.append(
                    (case_id, AssertionError(f"non-deterministic: value={n_val} bit={n_bit}"))
                )
                print(f"[collected] {case_id} FAILED: non-deterministic")
            else:
                print(f"[L0-14] PASS: two runs of {dtype_str} are bit-identical")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(c for c, _ in failures))
    else:
        print("[L0] ALL PASS (14 items)")
    return failures


def _run_iface_case(failures):
    """L0-11: interface contract (5 positional args + out_idx=[1])."""
    case_id = "L0-11/interface-contract"
    try:
        batch, heads, seq_len, dim, dtype_str = 1, 4, 33, 64, "float16"
        torch_dtype = torch.float16
        dq = torch.randn(
            batch,
            seq_len,
            heads,
            dim,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(31),
        )
        # Wrapper call form: exactly five positional arguments.
        kernel = _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype_str)
        out = kernel(dq.npu()).cpu()
        assert tuple(out.shape) == (batch, seq_len, heads, dim), (
            f"out_idx allocation shape {tuple(out.shape)} != {(batch, seq_len, heads, dim)}"
        )
        assert out.dtype == torch_dtype, f"out_idx dtype {out.dtype} != {torch_dtype}"
        ref = golden_flashattn_bwd_postprocess(dq, torch_dtype)
        n_val, n_bit, max_diff = _compare_bits(out, ref)
        assert n_val == 0, f"value_mismatches={n_val} bit={n_bit} max_diff={max_diff:.3e}"
        # Keyword form must keep the same semantics (blk_elems is optional).
        kernel_kw = _flashattn_bwd_postprocess_kernel(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype_str,
            blk_elems=4096,
        )
        out_kw = kernel_kw(dq.npu()).cpu()
        n_val_kw, n_bit_kw, _ = _compare_bits(out_kw, ref)
        assert n_val_kw == 0, f"keyword-call value_mismatches={n_val_kw} bit={n_bit_kw}"
        print(
            f"[L0-11] PASS: 5-positional-arg call + out_idx=[1] -> "
            f"{tuple(out.shape)} {out.dtype}; keyword call identical"
        )
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


def run_L1():
    """L1: extended functional coverage (blocking).

    Covers both storage dtypes on larger shapes, the persistent guard path
    (num_logical not a multiple of the 40 vector cores), single-logical-task
    grids, seq_len < blk_s, heads == 1, budget scans on a tail-carrying shape
    and the factory contract guards.

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    cases = [
        # id, (B, S, H, D), dtype, blk_elems
        ("L1/fp16/large", (2, 1024, 16, 128), "float16", None),
        ("L1/bf16/large", (2, 1024, 16, 128), "bfloat16", None),
        ("L1/fp16/manifest-8b-long", (1, 2048, 32, 128), "float16", None),
        ("L1/bf16/tail-budget8192", (4, 300, 4, 128), "bfloat16", 8192),
        ("L1/fp16/tail-budget4096", (4, 300, 4, 128), "float16", 4096),
        ("L1/fp16/guard-shape", (3, 777, 5, 64), "float16", None),
        ("L1/bf16/guard-shape", (5, 333, 7, 128), "bfloat16", None),
        ("L1/fp16/single-task", (1, 7, 1, 64), "float16", None),
        ("L1/bf16/single-task", (1, 5, 1, 128), "bfloat16", None),
        ("L1/fp16/seq-lt-blk", (1, 10, 4, 64), "float16", None),
        ("L1/bf16/heads1", (2, 100, 1, 64), "bfloat16", None),
        ("L1/fp16/batch1-tail", (1, 129, 8, 64), "float16", None),
        ("L1/bf16/odd-dims", (3, 65, 3, 96), "bfloat16", None),
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
    _expect_reject(
        failures,
        "L1/guard-dtype",
        {"batch": 1, "heads": 8, "seq_len": 4, "dim": 64, "dtype": "float32"},
    )
    _expect_reject(
        failures,
        "L1/guard-batch0",
        {"batch": 0, "heads": 8, "seq_len": 4, "dim": 64, "dtype": "float16"},
    )
    _expect_reject(
        failures,
        "L1/guard-dim0",
        {"batch": 1, "heads": 8, "seq_len": 4, "dim": 0, "dtype": "float16"},
    )
    _expect_reject(
        failures,
        "L1/guard-blk_elems0",
        {"batch": 1, "heads": 8, "seq_len": 4, "dim": 64, "dtype": "float16", "blk_elems": 0},
    )
    # 32768 * 6 B * 1.7 = 334 KB > 196608 B -> the declared-buffer UB guard fires.
    _expect_reject(
        failures,
        "L1/guard-ub-budget",
        {"batch": 1, "heads": 8, "seq_len": 4, "dim": 64, "dtype": "float16", "blk_elems": 32768},
    )
    # row = 2 * 10240 = 20480 elements -> blk_s clamps to 1 and the residence
    # tile alone exceeds the declared UB budget -> second (shape) guard fires.
    _expect_reject(
        failures,
        "L1/guard-row-over-budget",
        {"batch": 1, "heads": 2, "seq_len": 8, "dim": 10240, "dtype": "float16"},
    )

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(c for c, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: real manifest shapes and larger extremes (warn only, non-blocking)."""
    cases = [
        (1, 512, 8, 64, "float16"),
        (4, 512, 32, 128, "bfloat16"),
        (2, 2048, 32, 128, "float16"),
        (2, 512, 64, 128, "bfloat16"),
        (1, 2048, 64, 128, "float16"),
        (1, 3, 1, 3, "bfloat16"),
        (2, 5, 3, 7, "float16"),
        (8, 17, 2, 64, "float16"),
        (16, 33, 3, 64, "bfloat16"),
        (1, 257, 8, 256, "float16"),
        (1, 64, 2, 4096, "float16"),
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
    """Boundary: extreme shapes and special-value inputs (warn only)."""
    # seq_len == 1 (single row per batch).
    try:
        _run_case(batch=2, heads=4, seq_len=1, dim=64, dtype_str="float16", tag="Boundary-s1")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): seq_len=1: {exc!r}")

    # H = D = 1 over a long sequence (blk_s clamps to seq_len).
    try:
        _run_case(batch=1, heads=1, seq_len=1000, dim=1, dtype_str="bfloat16", tag="Boundary-h1d1")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): H=D=1: {exc!r}")

    # dim == 1 with a wide row count.
    try:
        _run_case(batch=1, heads=8, seq_len=1024, dim=1, dtype_str="float16", tag="Boundary-d1")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): dim=1: {exc!r}")

    # Whole-tensor corner values (inf / NaN / zero / ties / overflow).
    for dtype_str in ("float16", "bfloat16"):
        try:
            _run_case(
                batch=2,
                heads=4,
                seq_len=64,
                dim=64,
                dtype_str=dtype_str,
                tag="Boundary-corners",
                dq=_corner_input(2, 64, 4, 64, seed=5),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): corners {dtype_str}: {exc!r}")

    # fp16 saturation domain: dense values above the fp16 overflow threshold.
    try:
        dq = torch.full((1, 128, 4, 64), 7.0e4, dtype=torch.float32)
        dq[0, 0, 0, 0] = -7.0e4
        _run_case(
            batch=1,
            heads=4,
            seq_len=128,
            dim=64,
            dtype_str="float16",
            tag="Boundary-saturate",
            dq=dq,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): saturate: {exc!r}")

    # fp16 subnormal domain (dense values below the min normal).
    try:
        dq = (
            torch.randn(
                1,
                128,
                4,
                64,
                dtype=torch.float32,
                generator=torch.Generator().manual_seed(41),
            )
            * 3.0e-5
        )
        _run_case(
            batch=1,
            heads=4,
            seq_len=128,
            dim=64,
            dtype_str="float16",
            tag="Boundary-subnormal",
            dq=dq,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {exc!r}")


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
