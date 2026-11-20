# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Argmax/argmin first-occurrence reduction kernel (NPU, Developer mode).

Experimental copy of ``perf_opt/_argreduce_kernel.py``. The N-split path
fuses partial reduction and final merge into one device launch; the original
``perf_opt`` implementation is intentionally left unchanged for A/B testing.

ROUND-6 VERSION (``_argreduce_kernel_r6``): current best (rounds 1-4) plus
two later vectorization fixes -- the r5 vectorized int64 epilogue
(``T.reshape`` + ``T.vcast``) and the r6 in-kernel idx construction
(1-D ``T.arange`` -> ``T.reshape`` -> ``T.vbrc``, replacing the round-4 FIX
GM idx input). Call contract back to ``main(x) -> out`` for the resident
path (no host-side idx tensor injection).

ROUND-7 VERSION (nsplit fused rewrite, 2026-10-13): the fused single-launch
nsplit kernel is rebuilt on the probe-verified expert-mode sub-block path
(logs/round5/probes/probe_blockdim40.py minimal implementation). The old
r6 fused form launched ``T.Kernel(vector_cores=48)`` with ``(cid, _)``:
the inter-block ``T.block_barrier`` FFTS handshake only completes through
the mix launch path with a single-wave grid (<= aicore_num = 24 blocks on
910B2C), so that form is rejected by the launcher guard (48 > 24) and any
<= 24-block variant without sub-block awareness makes the two V cores of
each AI core execute the SAME block body (duplicated partial work). The
r7 form:
    R7-a: the nsplit fused kernel alone is compiled under
        ``TILELANG_ASCEND_MODE=expert`` (env switched around the jit call;
        every other path in this file stays Developer mode, unchanged).
        Expert form: ``T.Scope("Vector")`` + ``T.alloc_ub`` buffers (pure
        vector, auto-promoted to MIX: empty ``_mix_aic`` stub + full
        ``_mix_aiv`` body, probe_blockdim40.py evidence).
    R7-b: 1-to-2 sub-core split ("1 fen 2"): ``T.Kernel(num_blocks) as
        (cid, subid)`` with ``num_blocks = ceil(nchunk / 2) <= 24``; the
        flattened V-core lane ``vid = cid * 2 + subid`` owns chunks
        ``vid + s * (2 * num_blocks)`` (grid-stride). Each AI core's two
        V cores process DIFFERENT chunks instead of duplicating one.
        Probe evidence: blockdim40 (NC=24) ws == 2x exact -- both subid
        lanes execute with distinct work; subid is get_sub_block_idx
        (hardware V-core id in {0,1}), NOT a third grid dim.
    R7-c: merge restructured to drop ``T.transpose`` (the old
        ``_merge_workspace`` macro): core (cid==0, subid==0) reads the
        chunk-major ``(nchunk, M)`` workspace contiguously and reduces
        over dim 0 (doc-legal: docs/Tilelang.language/规约操作/T.reduce.md
        2.2.2 shape rule) + ``T.vbrc`` row broadcast (1, M) -> (nchunk, M)
        (docs/Tilelang.language/shape操作/T.vbrc.md 2.2.2). Cross-subid
        barrier freshness probe-verified (odd-vid chunk extremes, late-
        chunk duplicates, randn sweeps; 0 failures).
    R7-d: partial-phase candidate construction switched from the fused
        ``T.Parallel + if_then_else`` loop (the r3e "tiled-path-proven
        form") to the explicit ``T.vcmp + T.vselect`` chain with manually
        hoisted task-invariant operands (r6 1-D arange -> reshape idx
        matrix + BIG sentinel vbrc, built once per lane). The expert
        pipeline has no LICM (--disable-hivm-tensor-compile, mandatory
        for the mix pre-split form), so the fused-ite lowered to
        varange+vcast+vbrc(sentinel) INSIDE the row loop -- 3 extra
        tn-wide vector passes per row on the serialized chain. Isolated
        partial A/B: fused-ite 17.126us vs chain+hoist 5.952us (2.88x;
        per-lane scalar pipe 12-13us -> 1.3-1.9us). Evidence:
        /home/tilelang/l00970450/upload/data/1013 (IR diff + msprof).
    R7-e (2026-10-14, data/1014 report integration): the final_sweep
        tunables are wired into the framework entries -- ``_nsplit_config``
        gains a ``tn`` override and a CALIBRATED default rule (largest tn
        with nchunk in [vector_cores//3, vector_cores//2]; chain-form
        sweep: tn 2560/3200/5120/6400/10240 -> 12.73/11.62/10.14/10.23/
        10.58us, optimum tn=5120 = 1.20x over the r3e pick and over the
        dual-kernel baseline), ``_build_nsplit_fused``/``build_nsplit``/
        ``build_case`` gain ``num_blocks``; ``build_case`` routes its
        ``tile_n`` to the nsplit tn; ``--tn``/``--blocks`` CLI flags let
        ``--level nsplit`` precision-gate any config. The calibrated
        default IS the shipped configuration (no env switches). Full
        r7-d/r7-e results: /home/tilelang/l00970450/upload/data/1014/report.md.


Migrated from the GPU TileOPs ``_argreduce_kernel`` (ArgmaxFwdOp) to
``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = min{ j in [0, N) : x[i, j] == ext_i(x) },  ext = max (argmax) / min (argmin)

    First-occurrence tie-break, int64 output, matching ``torch.argmax`` /
    ``torch.argmin`` exactly on the (M, N) -> (M,) contract.

NPU redesign (DESIGN.md sections 0.6 / 1.4, frozen):
    R1: vectorized first-occurrence: ``first = reduce_min_j(ite(x_j == m, j, BIG))``
        with ``BIG = 2**30`` (fp32-exact sentinel; replaces the GPU serial
        scan + loop_break).
    R2: persistent one-dimensional core split: ``num_kernels = min(num_row_blocks,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and an
        if-guard for out-of-range row blocks (static bounds, Vector cores x2).
    R3: raw-N contract: kernel receives the original (M, N) (no host F.pad);
        non-divisible tiles handled by a static-width tail tile (no kernel-side
        masking).
    R4: block_m / tile_n chosen from a probe-calibrated UB budget table
        (DESIGN.md section 4.5); resident path for N * B/elem * bm <= 64KB,
        tiled online path (forced bm=1) otherwise.
    R5: vectorized int64 epilogue on every path: ``T.reshape`` +
        ``T.vcast(..., round_mode="rint")`` replaces the generic
        ``T.Parallel`` cast loops, which stayed scalar because
        NpuLoopVectorize rejects rank-reducing Cast (npu_loop_vectorize.cc:
        785-794; see CG-2026-0020). Applied to resident (first), tiled
        (running_idx), and nsplit merge (first). The narrow path needed no
        change: its ``first_flat[i]`` load is already rank-1 (1->1) and
        NpuLoopVectorize auto-vectorizes it (verified in final IR:
        ``npuir_cast`` region op -> ``hivm.hir.vcast``). Value-identical to
        the old fptosi trunc on the integer-valued domain.
    R6: in-kernel idx construction: ``T.arange(idx_src (N,), [1], 0)`` ->
        ``T.reshape(idx_src, idx_row (1,N))`` -> ``T.vbrc(idx_row, idx_j)``
        (bm==1: reshape directly into idx_j). Replaces the GM idx input of
        the 2026-09-30 FIX. Probe evidence (2026-09-30, /tmp/opencode/
        idx_probe): the old r4b 2-D form (``T.arange(idx_row (1,N), [0,1],
        0)`` -> vbrc) lowers to ``varange strides[%c0, %c1]`` (dim0
        stride-0) and races under padded-stride layouts N%16 in [9,12] with
        bm>=2 (block-first-row tail lanes go stale); the 1-D form lowers to
        ``varange strides[%c1]`` + metadata-only ``tensor.expand_shape`` and
        passed the identical sensitive detector (0 fail / 315 launches)
        plus a 21-case random battery while the 2-D control failed 23
        (shape, column) combos. msprof kernel-only vs the GM-idx version:
        (128,300) bf16 5.02 -> 3.12us (-37.9%), (2048,256) fp16 5.34 ->
        4.74us (-11.2%), hidden-state ~-1%.

Known toolchain constraints enforced here (DESIGN.md section 9.1, probe-backed):
    C-1 reduce dst dtype MUST equal src dtype (mixed dtype silently corrupts).
    C-2 bm==1 (.,1)-indexed operand in a Parallel condition miscompiles
        (f16 -> i1 broadcast) -> bm==1 uses ``T.vbrc`` into a same-shape buffer.
    C-3 unused fragment alloc is NOT dead-code-eliminated -> bm==1 / bm>=2 and
        bf16 / non-bf16 each get a dedicated prim_func (no unconditional alloc).
        The TVM tracer also does not track buffers allocated inside a
        conditional block (variant allocs must live in dedicated prim_funcs).
    C-4 shared multi-consumer buffers race under auto-multi-buffer when the
        double-buffer budget overflows -> all compute reads fragments; shared
        is GM staging only.
    C-5 multitile + bm>1 compiler SIGSEGV -> tiled path forces bm=1.
    C-6 ``T.vcast`` f32->f32 is non-identity -> fp32 path uses same-dtype copy.
    C-8 if_then_else condition referencing a serial loop var segfaults -> the
        candidate condition only references buffer elements; tile base index is
        plain arithmetic in the value position.
    C-9 2-D stride-0 ``varange`` (``T.arange`` into a (1,N) region with
        strides [0,1]) feeding a VCOPY broadcast races on padded-stride
        layouts (N%16 in [9,12], bm>=2) -> the resident idx construction
        uses the 1-D arange + reshape form (C-9 does NOT affect the 1-D
        ``varange strides[%c1]`` + ``tensor.expand_shape`` lowering; probe
        evidence in the R6 note).
    C-10 same-shape ``T.vbrc`` (no size-1 dim) emits an empty broadcast_dims
        array -> MLIR verify fail -> the bm==1 idx path reshapes (N,) ->
        (1,N) directly instead of a (1,N)->(1,N) vbrc.
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

from tileops.kernels._factory_cache import kernel_factory_cache

# Fused arg-reduce (CG-2026-0031) requires the dual-dst frontend/codegen
# patch (`tl.npuir_reduce_with_index`).  Guard the import so an unpatched
# toolchain keeps using the 5-op chain.
try:
    from tilelang.language import reduce_argmax as _tl_reduce_argmax
except ImportError:  # pragma: no cover - unpatched toolchain
    _tl_reduce_argmax = None

# Selection: auto (default) -- use the fused path whenever the patched
# frontend is present, because it is 1.4-2.0x faster on the hidden-state
# workload (fp16 30.7 -> 15.8 us, bf16 36.0 -> 25.5 us, Op-level msprof,
# 2026-11-16) and bit-exact on production shapes + ties (122 tests).
# TILEOPS_ARGMAX_FUSED=0 forces the portable 5-op chain (escape hatch);
# =1 forces fused (still no-op on an unpatched toolchain).
_fused_env = os.environ.get("TILEOPS_ARGMAX_FUSED")
_FUSED_ARGMAX = _tl_reduce_argmax is not None and (_fused_env is None or _fused_env != "0")

# Grid caps for the fused hidden-state paths (A/B knobs).
# fp16 group4: 512 logical tiles -> 32 cores gives exactly 16 waves (no tail);
# measured 15.74/15.92 vs 17.16/16.80 at 40 cores (-6.8%).
# bf16 group3: 32 cores under-fills (28.4-28.6 vs 24.1), keep 40.
_GRID_CAP = int(os.environ.get("TILEOPS_ARGMAX_GRID", "40"))
# R48: 40 AIV cores (not 32).  Same-session A/B on (2048,4096) bf16 bitcast:
# grid32 16.02 us vs grid40 15.34/15.78/15.47 (median 15.47) -> 32 cores left 8
# cores idle; the guarded tail slots of the 4-group unroll cost nothing.
_GRID_CAP_FP16 = int(os.environ.get("TILEOPS_ARGMAX_GRID_FP16", "40"))
_FUSED_MIN_N = 256
_FUSED_MAX_N = 32768

# R49: rows-parallel N-split partial (lm-head class) knobs.  ``=0`` restores the
# historical serial-row partial; ``TILEOPS_ARGMAX_NSPLIT_TN`` pins the chunk
# width (A/B only -- the default heuristic targets nchunk ~ 16).
_NSPLIT_ROWS_ENV = "TILEOPS_ARGMAX_NSPLIT_ROWS"
_NSPLIT_TN_ENV = "TILEOPS_ARGMAX_NSPLIT_TN"

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# fp32-exact sentinel (2**30 is a power of two, exactly representable in fp32,
# and larger than any valid index j < N <= 2**24).
# P1 bitcast contract: the bf16 payload is reinterpreted as fp16 and compared
# directly.  fp16's decoded value is monotone in the 16-bit pattern, so the
# ordering is preserved as long as fp16 does not decode the payload as
# NaN/Inf/subnormal:
#   * fp16 exponent field = bf16 exponent bits [7:3]; field 31 (NaN/Inf) needs
#     a bf16 exponent >= 248, i.e. |x| >= 2**121 (excluded by contract);
#   * fp16 subnormal needs a bf16 exponent < 8, i.e. 0 < |x| < 2**-119 (the
#     guard below rejects such inputs rather than comparing them wrongly).
ARGMAX_BITCAST_MIN_ABS = 2.0 ** -119
BIG = 2**30

# Probe-calibrated race-safe manual UB budget per block (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Column alignment boundary for the tiled path (DESIGN.md section 5.2).
TILE_ALIGNMENT = 256

# Launch-item threshold above which the extended block_m ladder engages
# (DESIGN.md section 5.2; REVIEW blocking-1 fix).
_LAUNCH_GATE = 96

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
_KINDS = ("argmax", "argmin")

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Factory-period config helpers (DESIGN.md section 5.2 ladder)
# ---------------------------------------------------------------------------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _B_elem(dtype, bm_class):
    """Fragment byte budget per element (DESIGN.md section 4.5 B/elem table)."""
    table = {
        "float16": {"multi": 8, "bm1": 10},
        "float32": {"multi": 12, "bm1": 16},
        "bfloat16": {"multi": 8, "bm1": 14},
    }
    return table[dtype][bm_class]


def _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=0):
    """Large UB buffer inventory in elem-sized slab units (inlined _primitives)."""
    total_bytes = dtype_slabs * elem_bytes + fp32_slabs * 4
    return (total_bytes + elem_bytes - 1) // elem_bytes


def _tiled_slab(dtype, elem_bytes):
    """Tiled-path slab count per DESIGN.md section 5.2 (fp16 5 / bf16 7 / fp32 4)."""
    if dtype == "float16":
        # x_ub(2B) + x_work(2B) + ext_brc(2B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=3, fp32_slabs=1)
    if dtype == "bfloat16":
        # x_ub(2B) + x_work(4B) + ext_brc(4B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=3)
    # float32: x_ub(4B) + x_work(4B) + ext_brc(4B) + cand(4B)
    return _ub_slab_units(elem_bytes, dtype_slabs=4, fp32_slabs=0)


def _pick_tile_n(N, dtype, budget=UB_MANUAL_BUDGET, alignment=TILE_ALIGNMENT, margin=0.9):
    """Tiled-path tile_n: prefer a 256-aligned divisor with >=10% budget margin,
    fall back to the budget cap + static tail tile (DESIGN.md section 5.2)."""
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    slab = _tiled_slab(dtype, elem_bytes)
    if slab * 1 * elem_bytes * N <= budget:
        return N  # fits entirely (not expected on the tiled path)
    max_cols = budget // (slab * 1 * elem_bytes)
    tile_n_cap = (max_cols // alignment) * alignment
    best = 0
    for candidate in range(tile_n_cap, 0, -alignment):
        if N % candidate == 0 and slab * elem_bytes * candidate <= int(budget * margin):
            best = candidate
            break
    if best > 0:
        return best
    return tile_n_cap


def _select_config(M, N, dtype):
    """Factory-period ladder (DESIGN.md section 5.2).

    Returns ``{"block_m", "path", "tile_n"}``. ``path`` is "resident" or
    "tiled". The basic ladder uses ``B_bm1`` for p=1 (REVIEW N1 fix: bm=1
    allocates the extra broadcast buffer) and ``B_multi`` for p>=2.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if N > 2**24:
        raise ValueError(
            f"N={N} exceeds 2**24; fp32 cannot exactly represent the index "
            f"sentinel arithmetic (E1/E3 equivalence domain, DESIGN.md section 5.2)"
        )

    B_multi = _B_elem(dtype, "multi")
    B_bm1 = _B_elem(dtype, "bm1")

    # Basic ladder {1,2,4,8} (p=1 uses B_bm1 to account for ext_brc).
    candidates = [
        p for p in (1, 2, 4, 8) if p * N * (B_bm1 if p == 1 else B_multi) <= UB_MANUAL_BUDGET
    ]
    block_m = max(candidates) if candidates else None

    # Extended ladder (launch-item driven) for small-N/large-M workloads.
    if block_m is not None and _ceildiv(M, block_m) > _LAUNCH_GATE:
        # Narrow-N (N < TILE_ALIGNMENT) workloads inflate more under
        # auto-multi-buffer than the wide-N 2x the basic ladder was calibrated
        # against (DESIGN.md section 9.2-R10: (2048,4) 64KB manual -> 256KB,
        # ~4x). Halve the manual budget for narrow N so the real UB footprint
        # stays within 192KB; wide-N workloads do not engage the extended
        # ladder meaningfully (their basic-ladder bm is already budget-capped).
        # ROUND-1-A: the chain form carries 4 extra full-width buffers
        # (ext_brc/cmp/idx_j/sent_v vs the fused form's cand-only); on the
        # narrow-N (N < 256) shapes the auto-multi-buffer pass measured a
        # ~3.06x inflation (L0-6b bm=1024@N=4: 68KB manual -> 208KB required
        # vs 192KB UB). Divide the narrow-N extended-ladder budget by 3 so the
        # ladder lands at bm=512 (34KB manual -> ~105KB actual).
        ext_budget = UB_MANUAL_BUDGET if N >= TILE_ALIGNMENT else UB_MANUAL_BUDGET // 3
        ext = [p for p in (16, 32, 64, 128, 256, 512, 1024, 2048) if p * N * B_multi <= ext_budget]
        block_m = max(ext + [block_m])

    if block_m is None:
        return {"block_m": 1, "path": "tiled", "tile_n": _pick_tile_n(N, dtype)}
    # Never allocate more rows than exist: block_m > M leaves uninitialized
    # trailing rows in the UB staging buffer; for narrow N the vectorizer's
    # small-N row handling can then corrupt the valid row (garbage rows).
    block_m = min(block_m, M)
    # ROUND-2-A: narrow-N wide-view deinterleave path takes precedence over
    # the plain resident path when the shape fits.
    ncfg = _select_narrow(M, N, dtype)
    if ncfg is not None:
        return {
            "block_m": ncfg["bw"] * ncfg["G"],
            "path": "narrow",
            "tile_n": None,
            "G": ncfg["G"],
            "bw": ncfg["bw"],
        }
    # Narrow-N correctness guard (adoption re-verification probe, 2026-09-28):
    # for N < TILE_ALIGNMENT the chain-form resident with block_m > 1 and the
    # tiled path (both prim_func variants) silently corrupt a data-dependent
    # row subset -- value corruption, not tie-break differences (e.g.
    # (8192,2) tiled fp16: 10-23/8192 wrong, kernel returns the smaller
    # element; (2048,16) tiled emits garbage indices ~-1.08e9; (2048,30)
    # tiled bf16: 412/2048; chain resident (2048,16) bm=128: 261/2048).
    # resident with block_m=1 is exact on this domain (324-combo battery:
    # N in {2,3,4,5,8,16,30,100,255} x M in {1,8,2048,8192} x
    # {fp16,bf16,fp32} x seeds {0,1,42}, 0 mismatches). Route every narrow-N
    # shape that does not fit the deinterleave path to resident bm=1.
    # N >= TILE_ALIGNMENT keeps the chain form (probe-clean there; carries
    # the tuned hidden-state workload). N == 1 remains unsupported (compile
    # error in every path; the Stage 3 baseline holds the i1 workaround).
    if N < TILE_ALIGNMENT:
        return {"block_m": 1, "path": "resident", "tile_n": None}
    return {"block_m": block_m, "path": "resident", "tile_n": None}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmax") -> torch.Tensor:
    """PyTorch reference: ``torch.argmax`` / ``torch.argmin`` first-occurrence.

    Runs on CPU. Does NOT reproduce the NPU algorithm (no mask candidate /
    reduce_min / online recurrence), guaranteeing validation independence
    (DESIGN.md section 8.1).
    """
    if op_kind == "argmax":
        return torch.argmax(x, dim=-1)
    return torch.argmin(x, dim=-1)


# ---------------------------------------------------------------------------
# STAGE-4 FINAL (merged current best, rounds 1-4):
#   - resident chain (r1a) with (1,N) arange + first-axis vbrc broadcast
#     (r4b: kills the (bm,N) arange scalar materialization floor);
#   - narrow-N wide-view deinterleave path (r2a, 3d-class workloads);
#   - C6 N-split partial+merge dispatch for lm-head-class shapes
#     (r3e/r3f: DESIGN.md 1.6.4 adjudication FLIPPED -- fp16 0.461x /
#     bf16 0.386x of the single-kernel main-select, both < the
#     pre-registered 0.8x threshold; per-row (1,tn) partial + transpose
#     merge, tn rule = largest single-wave nchunk <= 48).
# ---------------------------------------------------------------------------
# Resident path (path S, whole row resident in UB)
# ---------------------------------------------------------------------------
def _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores):
    """ROUND-1-A (chain): resident path with the vector-op candidate chain.

    Single change vs baseline: the fused ``T.Parallel + if_then_else``
    candidate loop (measured 0.95 scalar ratio on hidden-state) is replaced by
    the DESIGN.md section 1.6.3-#5 chain fallback:
        vbrc(ext) -> vcmp("eq") -> vselect(idx_j, sent_v) -> reduce_min
    - ext_brc is materialized via T.vbrc for ALL bm (no (bm,1) broadcast
      operand inside any Parallel condition; C-2 moot, bm=1/bm>=2 unified).
    - idx_j (column indices) is built in-kernel via the r6 1-D arange chain:
      ``T.arange(idx_src (N,), [1], 0)`` -> ``T.reshape`` to (1,N) ->
      ``T.vbrc`` to (bm,N); bm==1 reshapes (N,) into idx_j directly (C-10:
      same-shape vbrc fails MLIR verify). This replaces the 2026-09-30 GM
      idx input (MTE2 copy). Probe evidence: the r4b 2-D form
      (``T.arange`` into (1,N) with strides [0,1]) lowers to a stride-0
      ``varange`` whose tail writes race the VCOPY broadcast on
      padded-stride layouts (N%16 in [9,12], bm>=2; C-9); the 1-D form
      lowers to ``varange strides[%c1]`` + metadata-only
      ``tensor.expand_shape`` and is race-free on the identical sensitive
      detector (0 fail / 315 launches + 21-case random battery vs 23
      failing (shape, col) combos for the 2-D control). msprof vs the GM
      idx version: (128,300) bf16 -37.9%, (2048,256) fp16 -11.2%,
      hidden-state ~-1%.
    - x_frag staging dropped: compute (reduce_max/vcmp) reads x_ub directly
      (vcmp-on-shared proven by T.vselect.md section 2.4 example); keeps the
      chain buffer set within the x2 auto-multi-buffer envelope at bm=2.
    Two prim_funcs (bf16 / non-bf16); C-3 respected (ext_brc always used).
    r5 epilogue: the int64 output cast is vectorized via
    ``T.reshape(first, first_flat)`` + ``T.vcast(first_flat, out_ub,
    round_mode="rint")``. The original ``for i in T.Parallel(block_m):
    out_ub[i] = T.cast(first[i, 0], "int64")`` stayed scalar because
    NpuLoopVectorize deliberately rejects rank-reducing Cast (load
    ``first[i, 0]`` rank-2 vs store ``out_ub[i]`` rank-1,
    npu_loop_vectorize.cc HandleUnaryExpression guard) and the npuir
    codegen prints generic TIR loops verbatim as scalar scf.for. rint vs the
    old fptosi trunc is value-identical: ``first`` holds exact small integers
    (valid indices j < N <= 2**24 or the BIG sentinel, all fp32-exact).
    """
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_fragment((block_m, N), "float32")
                    row_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, N), "float32")
                    cmp_eq = T.alloc_fragment((block_m, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((block_m, N), "float32")
                    sent_v = T.alloc_fragment((block_m, N), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Task-invariant operands hoisted out of the task loop.
                    # r6: 1-D arange -> reshape -> vbrc (in-kernel idx). The
                    # 1-D varange lowers to a contiguous fill; expand_shape is
                    # metadata-only. NOT the r4b 2-D form (T.arange into (1,N)
                    # with strides [0,1]) -- that lowers to a stride-0 varange
                    # that races the VCOPY broadcast on padded-stride layouts
                    # (N%16 in [9,12], bm>=2; C-9). bm==1: reshape (N,) ->
                    # (1,N) directly into idx_j (C-10 same-shape vbrc trap).
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, row_ext, dim=1)
                            else:
                                T.reduce_min(x_work, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            T.vcmp(x_work, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            # r5 epilogue: vectorized first-occurrence ->
                            # int64. The old ``T.Parallel`` cast loop lowered
                            # to a scalar scf.for (NpuLoopVectorize rejects
                            # rank-reducing Cast: first[i,0] rank-2 store
                            # rank-1; npu_loop_vectorize.cc:785-794).
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    row_ext = T.alloc_fragment((block_m, 1), dtype)
                    ext_brc = T.alloc_fragment((block_m, N), dtype)
                    cmp_eq = T.alloc_fragment((block_m, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((block_m, N), "float32")
                    sent_v = T.alloc_fragment((block_m, N), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Task-invariant operands hoisted out of the task loop.
                    # r6: same 1-D arange -> reshape -> vbrc form as the bf16
                    # branch (C-9 / C-10 notes there).
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            if op_kind == "argmax":
                                T.reduce_max(x_ub, row_ext, dim=1)
                            else:
                                T.reduce_min(x_ub, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            # r5 epilogue: vectorized first-occurrence ->
                            # int64 (see bf16 variant note above).
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Fused argmax path (CG-2026-0031): one vreduce <max_with_index_left> with
# value + index dsts, replacing reduce_max -> vbrc -> vcmp -> vselect -> reduce_min.
#
# Requires the dual-dst frontend/codegen patch and Expert mode: the op writes
# two UB memrefs (`T.alloc_ub`), which the API codegen lowers to
# `hivm.hir.vreduce ... outs(val, idx)`.  One block per tile avoids the
# cross-iteration dependency that the current auto-sync pass does not track
# for the second (index) dst.
# ---------------------------------------------------------------------------
def _fused_block_m(N):
    """Largest tile width that keeps the fused UB working set in 64 KB."""
    return max(1, min(8, 32768 // max(int(N), 1)))


def _build_fused_argmax(M, N, dtype):
    """One-block-per-tile fused argmax (Expert mode)."""
    block_m = _fused_block_m(N)
    num_blocks = _ceildiv(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_block_m):
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_m, N), dtype)
                val = T.alloc_ub((block_m, 1), dtype)
                idx = T.alloc_ub((block_m, 1), "int32")
                idx_flat = T.alloc_ub((block_m,), "int32")
                out_ub = T.alloc_ub((block_m,), "int64")
                off = cid * block_m
                real = T.min(block_m, M - off)
                T.copy(x[off : off + real, 0:N], x_ub[0:real, 0:N])
                _tl_reduce_argmax(x_ub, val, idx, dim=1)
                T.reshape(idx, idx_flat)
                T.vcast(idx_flat, out_ub, round_mode="rint")
                T.copy(out_ub[0:real], out[off : off + real])

        return main

    _func.ascend_mode = "Expert"
    _func.is_fused = True
    _func.msprof_kernel_name = "main"
    return _func


def _build_fused_argmax_group4(M, N, dtype):
    """P3.8: persistent 40-block 4-tile-group fused argmax (bm=4).

    Four tiles are prefetched before the four fused reduces so loads 1/2/3
    overlap reduces 0/1/2.  Each tile owns its own ``val/idx/flat`` state: a
    shared set serialised the four compute chains (16.87 -> 15.77 us).

    The group boundary is guarded by ONE narrow WAR event (``PIPE_V`` ->
    ``PIPE_MTE2``) instead of ``PIPE_ALL``: the only hazard there is the next
    group's MTE2 overwriting x0..x3 while this group's V still reads them, and
    ``PIPE_ALL`` also drained MTE2/MTE3, which is why vec and mte2 overlapped by
    only 0.67 us (narrow: 2.26 us, aiv 13.73 -> 11.55 us).

    ``auto-multi-buffer`` stays disabled: the manual 4-buffer pipeline is enough
    and the pass otherwise doubles the 4x32 KB staging past the 192 KB limit.
    Measured on (2048,4096) fp16: 15.0-15.9 us (4 reps); a flat single-burst
    load was tried and measured identical (mte2 5.90 vs 5.88), so the 2-D copy
    is kept for simplicity.
    """
    block_m = 4
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _GRID_CAP_FP16)
    num_local = _ceildiv(num_logical, num_kernels)
    num_groups = _ceildiv(num_local, 4)

    @tilelang.jit(
        out_idx=[1],
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _func(_block_m):
        @T.macro
        def load(x, xb, off, real):
            T.copy(x[off : off + real, 0:N], xb[0:real, 0:N])

        @T.macro
        def compute(xb, val, idx, flat, ob, out, off, real):
            _tl_reduce_argmax(xb, val, idx, dim=1)
            T.reshape(idx, flat)
            T.vcast(flat, ob, round_mode="rint")
            T.copy(ob[0:real], out[off : off + real])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x0 = T.alloc_ub((block_m, N), dtype)
                x1 = T.alloc_ub((block_m, N), dtype)
                x2 = T.alloc_ub((block_m, N), dtype)
                x3 = T.alloc_ub((block_m, N), dtype)
                # Per-tile reduce state (P3.8): a single shared val/idx/flat
                # serialised the four compute chains (compute[i+1] overwrote
                # state that compute[i] was still reading), which forced a
                # full-pipe sync between chains and left the four reduces
                # overlapping only ~1.4 us.  40 B per tile buys independence
                # (measured 16.87 -> 15.77 us on (2048,4096) fp16).
                val0 = T.alloc_ub((block_m, 1), dtype)
                val1 = T.alloc_ub((block_m, 1), dtype)
                val2 = T.alloc_ub((block_m, 1), dtype)
                val3 = T.alloc_ub((block_m, 1), dtype)
                idx0 = T.alloc_ub((block_m, 1), "int32")
                idx1 = T.alloc_ub((block_m, 1), "int32")
                idx2 = T.alloc_ub((block_m, 1), "int32")
                idx3 = T.alloc_ub((block_m, 1), "int32")
                flat0 = T.alloc_ub((block_m,), "int32")
                flat1 = T.alloc_ub((block_m,), "int32")
                flat2 = T.alloc_ub((block_m,), "int32")
                flat3 = T.alloc_ub((block_m,), "int32")
                out0 = T.alloc_ub((block_m,), "int64")
                out1 = T.alloc_ub((block_m,), "int64")
                out2 = T.alloc_ub((block_m,), "int64")
                out3 = T.alloc_ub((block_m,), "int64")

                for p in T.serial(num_groups):
                    b0 = (p * 4 + 0) * num_kernels + cid
                    b1 = (p * 4 + 1) * num_kernels + cid
                    b2 = (p * 4 + 2) * num_kernels + cid
                    b3 = (p * 4 + 3) * num_kernels + cid
                    off0 = b0 * block_m
                    off1 = b1 * block_m
                    off2 = b2 * block_m
                    off3 = b3 * block_m
                    real0 = T.min(block_m, M - off0)
                    real1 = T.min(block_m, M - off1)
                    real2 = T.min(block_m, M - off2)
                    real3 = T.min(block_m, M - off3)

                    if b0 < num_logical:
                        load(x, x0, off0, real0)
                    if b1 < num_logical:
                        load(x, x1, off1, real1)
                    if b2 < num_logical:
                        load(x, x2, off2, real2)
                    if b3 < num_logical:
                        load(x, x3, off3, real3)

                    if b0 < num_logical:
                        compute(x0, val0, idx0, flat0, out0, out, off0, real0)
                    if b1 < num_logical:
                        compute(x1, val1, idx1, flat1, out1, out, off1, real1)
                    if b2 < num_logical:
                        compute(x2, val2, idx2, flat2, out2, out, off2, real2)
                    if b3 < num_logical:
                        compute(x3, val3, idx3, flat3, out3, out, off3, real3)

                    # P3.8 narrow WAR sync: the only cross-pipe hazard left at
                    # the group boundary is "next group's MTE2 overwrites
                    # x0..x3 while this group's V still reads them".
                    # ``PIPE_ALL`` used to drain MTE2/MTE3 as well, which kept
                    # the next prefetch from overlapping this group's reduces
                    # (same-harness A/B: 16.87 -> 15.30 us, aiv 13.73 -> 11.55).
                    # 250-run soak (random / all-equal / first-last tie / dup
                    # max / all -inf) is bit-exact.
                    with T.rs("PIPE_V"):
                        T.set_flag("PIPE_MTE2", 0)
                    with T.rs("PIPE_MTE2"):
                        T.wait_flag("PIPE_V", 0)

        return main

    _func.ascend_mode = "Expert"
    _func.is_fused = True
    _func.is_group4 = True
    _func.msprof_kernel_name = "main"
    return _func


def _build_fused_argmax_rows(M, N, dtype):
    """P2 (2A): contiguous row blocks + one deferred epilogue per core.

    The strided tile map (``blk = t * num_kernels + cid``) gives each core rows
    that are ``num_kernels * block_m`` apart, so every tile needs its own
    ``idx -> int64 -> GM`` epilogue.  msprof on (2048, 4096) shows what that
    costs: ``aiv_mte3_time`` 2.0-4.6 us for only 512 B per core, i.e. a fixed
    ~0.2 us per tiny GM store, 16 stores per core (32 waves of 4-row tiles).

    This builder assigns each core a *contiguous* row band
    (``base = cid * num_local * block_m``), keeps the per-tile ``idx`` in a UB
    accumulator, and runs the int32 -> int64 conversion plus a single GM store
    once after the tile loop:

        per tile : MTE2 tile -> vreduce -> idx -> acc[t*block_m : ...]   (UB only)
        after    : vcast(acc -> int64) ; copy(acc64[0:real], out[base:...])  (1 store)

    Same per-tile V work and same 4-deep tile pipeline as group4; the win is
    removing 15 of the 16 epilogues per core and making the GM reads/writes of
    a core contiguous.
    """
    block_m = 4
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _GRID_CAP_FP16)
    num_local = _ceildiv(num_logical, num_kernels)
    num_groups = _ceildiv(num_local, 4)
    acc_elems = num_local * block_m

    @tilelang.jit(
        out_idx=[1],
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _func(_block_m):
        @T.macro
        def load(x, xb, off, real):
            T.copy(x[off : off + real, 0:N], xb[0:real, 0:N])

        @T.macro
        def reduce_acc(xb, val, idx, flat, acc, off_acc, real):
            _tl_reduce_argmax(xb, val, idx, dim=1)
            T.reshape(idx, flat)
            T.copy(flat[0:real], acc[off_acc : off_acc + real])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                base = cid * acc_elems
                x0 = T.alloc_ub((block_m, N), dtype)
                x1 = T.alloc_ub((block_m, N), dtype)
                x2 = T.alloc_ub((block_m, N), dtype)
                x3 = T.alloc_ub((block_m, N), dtype)
                val0 = T.alloc_ub((block_m, 1), dtype)
                val1 = T.alloc_ub((block_m, 1), dtype)
                val2 = T.alloc_ub((block_m, 1), dtype)
                val3 = T.alloc_ub((block_m, 1), dtype)
                idx0 = T.alloc_ub((block_m, 1), "int32")
                idx1 = T.alloc_ub((block_m, 1), "int32")
                idx2 = T.alloc_ub((block_m, 1), "int32")
                idx3 = T.alloc_ub((block_m, 1), "int32")
                flat0 = T.alloc_ub((block_m,), "int32")
                flat1 = T.alloc_ub((block_m,), "int32")
                flat2 = T.alloc_ub((block_m,), "int32")
                flat3 = T.alloc_ub((block_m,), "int32")
                acc = T.alloc_ub((acc_elems,), "int32")
                acc64 = T.alloc_ub((acc_elems,), "int64")

                for p in T.serial(num_groups):
                    off0 = base + (p * 4 + 0) * block_m
                    off1 = base + (p * 4 + 1) * block_m
                    off2 = base + (p * 4 + 2) * block_m
                    off3 = base + (p * 4 + 3) * block_m
                    real0 = T.min(block_m, M - off0)
                    real1 = T.min(block_m, M - off1)
                    real2 = T.min(block_m, M - off2)
                    real3 = T.min(block_m, M - off3)

                    if off0 < M:
                        load(x, x0, off0, real0)
                    if off1 < M:
                        load(x, x1, off1, real1)
                    if off2 < M:
                        load(x, x2, off2, real2)
                    if off3 < M:
                        load(x, x3, off3, real3)

                    if off0 < M:
                        reduce_acc(x0, val0, idx0, flat0, acc,
                                   (p * 4 + 0) * block_m, real0)
                    if off1 < M:
                        reduce_acc(x1, val1, idx1, flat1, acc,
                                   (p * 4 + 1) * block_m, real1)
                    if off2 < M:
                        reduce_acc(x2, val2, idx2, flat2, acc,
                                   (p * 4 + 2) * block_m, real2)
                    if off3 < M:
                        reduce_acc(x3, val3, idx3, flat3, acc,
                                   (p * 4 + 3) * block_m, real3)

                    # P3.8 narrow WAR sync between tile groups.
                    with T.rs("PIPE_V"):
                        T.set_flag("PIPE_MTE2", 0)
                    with T.rs("PIPE_MTE2"):
                        T.wait_flag("PIPE_V", 0)

                # One epilogue for the whole core's row band.
                real_total = T.min(acc_elems, M - base)
                if real_total > 0:
                    T.vcast(acc, acc64, round_mode="rint")
                    T.copy(acc64[0:real_total], out[base : base + real_total])

        return main

    _func.ascend_mode = "Expert"
    _func.is_fused = True
    _func.is_rows = True
    _func.msprof_kernel_name = "main"
    return _func


def _build_fused_argmax_rows_w(M, N, dtype, block_m=8, nf=2):
    """P2 (2B): contiguous bands + wide tiles + deferred epilogue.

    Same idea as ``_build_fused_argmax_rows`` (one epilogue per core) but the
    in-flight budget is spent on *wider* tiles instead of more tiles:
    ``block_m * nf * N * esize = 128 KiB`` in every configuration, so

        bm=4, nf=4 -> 16 serial tiles/core   (the group4 shape)
        bm=8, nf=2 ->  8 serial tiles/core   (halves reduce/copy instruction count)
        bm=16,nf=1 ->  4 serial tiles/core

    ``vreduce`` costs roughly ``a + b*rows`` per call (measured ~13 elements
    per cycle per core on the AIV), so halving the number of calls removes half
    of the per-call part while the element work stays constant.  The cost is
    pipeline depth: nf=2 leaves one group in flight instead of four.
    """
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _GRID_CAP_FP16)
    num_local = _ceildiv(num_logical, num_kernels)
    num_groups = _ceildiv(num_local, nf)
    acc_elems = num_local * block_m

    @tilelang.jit(
        out_idx=[1],
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _func(_block_m):
        @T.macro
        def load(x, xb, off, real):
            T.copy(x[off : off + real, 0:N], xb[0:real, 0:N])

        @T.macro
        def reduce_acc(xb, val, idx, flat, acc, off_acc, real):
            _tl_reduce_argmax(xb, val, idx, dim=1)
            T.reshape(idx, flat)
            T.copy(flat[0:real], acc[off_acc : off_acc + real])

        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                base = cid * acc_elems
                x0 = T.alloc_ub((block_m, N), dtype)
                x1 = T.alloc_ub((block_m, N), dtype)
                val0 = T.alloc_ub((block_m, 1), dtype)
                val1 = T.alloc_ub((block_m, 1), dtype)
                idx0 = T.alloc_ub((block_m, 1), "int32")
                idx1 = T.alloc_ub((block_m, 1), "int32")
                flat0 = T.alloc_ub((block_m,), "int32")
                flat1 = T.alloc_ub((block_m,), "int32")
                acc = T.alloc_ub((acc_elems,), "int32")
                acc64 = T.alloc_ub((acc_elems,), "int64")

                for p in T.serial(num_groups):
                    off0 = base + (p * nf + 0) * block_m
                    off1 = base + (p * nf + 1) * block_m
                    real0 = T.min(block_m, M - off0)
                    real1 = T.min(block_m, M - off1)

                    if off0 < M:
                        load(x, x0, off0, real0)
                    if off1 < M:
                        load(x, x1, off1, real1)
                    if off0 < M:
                        reduce_acc(x0, val0, idx0, flat0, acc,
                                   (p * nf + 0) * block_m, real0)
                    if off1 < M:
                        reduce_acc(x1, val1, idx1, flat1, acc,
                                   (p * nf + 1) * block_m, real1)

                    with T.rs("PIPE_V"):
                        T.set_flag("PIPE_MTE2", 0)
                    with T.rs("PIPE_MTE2"):
                        T.wait_flag("PIPE_V", 0)

                real_total = T.min(acc_elems, M - base)
                if real_total > 0:
                    T.vcast(acc, acc64, round_mode="rint")
                    T.copy(acc64[0:real_total], out[base : base + real_total])

        return main

    _func.ascend_mode = "Expert"
    _func.is_fused = True
    _func.is_rows = True
    _func.msprof_kernel_name = "main"
    return _func


def _build_fused_argmax_bf16_group3(M, N):
    """P1: bf16 input -> fp32 upcast -> fused argmax (3-tile-group, bm=2).

    Native bf16 is rejected by the HIVM ``vreduce`` verifier (dtype list has
    no bf16); fp32 is accepted, so the kernel upcasts each tile and runs the
    dual-dst reduce on fp32.  The group prefetch overlaps loads with the
    previous group's upcast+reduce.  ``PIPE_ALL`` per group protects buffer
    reuse; ``auto-multi-buffer`` is disabled because bf16+fp32 staging is
    already close to the 192 KB UB limit.
    """
    block_m = 2
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _GRID_CAP)
    num_local = _ceildiv(num_logical, num_kernels)
    num_groups = _ceildiv(num_local, 3)

    @tilelang.jit(
        out_idx=[1],
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _func(_block_m):
        @T.macro
        def load(x, xb, off, real):
            T.copy(x[off : off + real, 0:N], xb[0:real, 0:N])

        @T.macro
        def compute(xb, xw, val, idx, flat, ob, out, off, real):
            T.vcast(xb, xw, round_mode="rint")
            _tl_reduce_argmax(xw, val, idx, dim=1)
            T.reshape(idx, flat)
            T.vcast(flat, ob, round_mode="rint")
            T.copy(ob[0:real], out[off : off + real])

        @T.prim_func
        def main(x: T.Tensor((M, N), "bfloat16"), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x0 = T.alloc_ub((block_m, N), "bfloat16")
                x1 = T.alloc_ub((block_m, N), "bfloat16")
                x2 = T.alloc_ub((block_m, N), "bfloat16")
                w0 = T.alloc_ub((block_m, N), "float32")
                w1 = T.alloc_ub((block_m, N), "float32")
                w2 = T.alloc_ub((block_m, N), "float32")
                # Per-tile reduce state (P3.8, same rationale as group4).
                val0 = T.alloc_ub((block_m, 1), "float32")
                val1 = T.alloc_ub((block_m, 1), "float32")
                val2 = T.alloc_ub((block_m, 1), "float32")
                idx0 = T.alloc_ub((block_m, 1), "int32")
                idx1 = T.alloc_ub((block_m, 1), "int32")
                idx2 = T.alloc_ub((block_m, 1), "int32")
                flat0 = T.alloc_ub((block_m,), "int32")
                flat1 = T.alloc_ub((block_m,), "int32")
                flat2 = T.alloc_ub((block_m,), "int32")
                out0 = T.alloc_ub((block_m,), "int64")
                out1 = T.alloc_ub((block_m,), "int64")
                out2 = T.alloc_ub((block_m,), "int64")

                for p in T.serial(num_groups):
                    b0 = (p * 3 + 0) * num_kernels + cid
                    b1 = (p * 3 + 1) * num_kernels + cid
                    b2 = (p * 3 + 2) * num_kernels + cid
                    off0 = b0 * block_m
                    off1 = b1 * block_m
                    off2 = b2 * block_m
                    real0 = T.min(block_m, M - off0)
                    real1 = T.min(block_m, M - off1)
                    real2 = T.min(block_m, M - off2)

                    if b0 < num_logical:
                        load(x, x0, off0, real0)
                    if b1 < num_logical:
                        load(x, x1, off1, real1)
                    if b2 < num_logical:
                        load(x, x2, off2, real2)

                    if b0 < num_logical:
                        compute(x0, w0, val0, idx0, flat0, out0, out, off0, real0)
                    if b1 < num_logical:
                        compute(x1, w1, val1, idx1, flat1, out1, out, off1, real1)
                    if b2 < num_logical:
                        compute(x2, w2, val2, idx2, flat2, out2, out, off2, real2)

                    # P3.8 narrow WAR sync (see group4): only the next group's
                    # MTE2 has to wait for this group's V reads.
                    with T.rs("PIPE_V"):
                        T.set_flag("PIPE_MTE2", 0)
                    with T.rs("PIPE_MTE2"):
                        T.wait_flag("PIPE_V", 0)

        return main

    _func.ascend_mode = "Expert"
    _func.is_fused = True
    _func.is_bf16_group3 = True
    _func.msprof_kernel_name = "main"
    return _func


# ---------------------------------------------------------------------------
# Narrow-N wide-view path (path W, ROUND-2-A): deinterleave phase decomposition
# ---------------------------------------------------------------------------
def _narrow_G(M, N, elem_bytes):
    """Wide-row grouping factor: largest power of two G <= 256 with
    M % G == 0 and N * G * elem_bytes >= 512 (wide GM copy rows)."""
    g = 1
    for k in range(1, 9):  # 2..256
        cand = 1 << k
        if M % cand == 0 and cand <= 256 and N * cand * elem_bytes >= 512:
            g = cand
    # also allow G smaller than the 512B threshold if no divisor qualifies
    # (fallback to the regular resident path happens in _select_config).
    return g


def _select_narrow(M, N, dtype):
    """Narrow-path config: {"G", "bw"} or None when the shape does not fit.

    N is restricted to 4: the phase extraction uses a two-level
    channel_nums=2 deinterleave tree (multi-dst deinterleave with
    channel_nums>=3 crashes bishengir-compile with SIGABRT -- probe
    probe_deint_bisect.py; the c2 form is probe-verified).
    """
    if N != 4:
        return None
    # float32 excluded (adoption re-verification, 2026-09-28): the
    # deinterleave form corrupts a row subset for fp32 at N=4 ((2048,4)
    # bm=512: 483/2048 wrong in the TileOPs suite; fp16/bf16 bm=1024 are
    # clean). fp32 shapes fall through to the narrow-N resident-bm=1
    # guard (324-combo battery clean, includes (2048,4) fp32).
    if dtype == "float32":
        return None
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    if N * elem_bytes > 16:
        return None
    G = _narrow_G(M, N, elem_bytes)
    if G < 2 or N * G * elem_bytes < 512:
        return None
    # block = bw wide rows; manual budget ~13B/elem (fp16) incl. out staging.
    bpc = 13 if dtype != "float32" else 17
    for bw in (8, 4, 2):
        if bw * G * N * bpc + (N + 1) * G * 4 <= 65536:
            return {"G": G, "bw": bw}
    return None


def _build_narrow(M, N, op_kind, dtype, work_dtype, vector_cores, G, bw):
    """ROUND-2-A narrow-N kernel: wide-row GM view + deinterleave phases.

    Root cause being fixed (bisect probes V0/V1): a (bm, N) narrow shared
    staging buffer feeding compute triggers auto-multi-buffer restructuring
    that re-reads the block from GM 16x, and narrow (bm, N) vector ops
    decompose per row. Structure:
      - kernel param declared as the wide-row view (M//G, N*G) over the SAME
        contiguous memory (runtime passes the (M, N) tensor; shape not
        validated, probe-verified roundtrip);
      - block copies (bw, N*G) wide rows (no GM inflation);
      - T.deinterleave(channel_nums=N) -> N phase buffers (bw, G) with
        phase_n(r, q) = x[G*r + q, n];
      - row max = vmax tree over phases; first-occurrence = vselect(cmp_n,
        const_n, BIG) then vmin tree (per-(r,q) over the N phases);
      - reshape (bw, G) -> (bm,) + int64 epilogue.
    """
    Mr = M // G
    Nr = N * G
    bm = bw * G
    num_logical = _ceildiv(M, bm)
    num_kernels = min(num_logical, vector_cores)
    num_local_tasks = _ceildiv(num_logical, num_kernels)
    is_fp16 = dtype == "float16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        # block_m is derived from bw*G; the argument is accepted for
        # interface uniformity and must equal bm.

        if is_fp16:

            @T.prim_func
            def main(x: T.Tensor((Mr, Nr), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    w = T.alloc_shared((bw, Nr), dtype)
                    e01 = T.alloc_fragment((bw, 2 * G), dtype)
                    o01 = T.alloc_fragment((bw, 2 * G), dtype)
                    p0 = T.alloc_fragment((bw, G), dtype)
                    p1 = T.alloc_fragment((bw, G), dtype)
                    p2 = T.alloc_fragment((bw, G), dtype)
                    p3 = T.alloc_fragment((bw, G), dtype)
                    row_max = T.alloc_fragment((bw, G), dtype)
                    t01 = T.alloc_fragment((bw, G), dtype)
                    t23 = T.alloc_fragment((bw, G), dtype)
                    cmp0 = T.alloc_fragment((bw, G), "bool")
                    cmp1 = T.alloc_fragment((bw, G), "bool")
                    cmp2 = T.alloc_fragment((bw, G), "bool")
                    cmp3 = T.alloc_fragment((bw, G), "bool")
                    c0 = T.alloc_fragment((bw, G), "float32")
                    c1 = T.alloc_fragment((bw, G), "float32")
                    c2 = T.alloc_fragment((bw, G), "float32")
                    c3 = T.alloc_fragment((bw, G), "float32")
                    k01 = T.alloc_fragment((bw, G), "float32")
                    k23 = T.alloc_fragment((bw, G), "float32")
                    first = T.alloc_fragment((bw, G), "float32")
                    first_flat = T.alloc_fragment((bm,), "float32")
                    const0 = T.alloc_fragment((bw, G), "float32")
                    const1 = T.alloc_fragment((bw, G), "float32")
                    const2 = T.alloc_fragment((bw, G), "float32")
                    const3 = T.alloc_fragment((bw, G), "float32")
                    big_v = T.alloc_fragment((bw, G), "float32")
                    out_ub = T.alloc_shared((bm,), "int64")

                    # task-invariant constants (hoisted)
                    T.vbrc(T.float32(0), const0)
                    T.vbrc(T.float32(1), const1)
                    T.vbrc(T.float32(2), const2)
                    T.vbrc(T.float32(3), const3)
                    T.vbrc(T.float32(BIG), big_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_w = block_id * bw
                            off_m = block_id * bm
                            real_w = T.min(bw, Mr - off_w)
                            real_m = T.min(bm, M - off_m)
                            T.copy(x[off_w : off_w + real_w, 0:Nr], w[0:real_w, 0:Nr])
                            # two-level c2 deinterleave tree (channel_nums=4
                            # multi-dst form crashes the compiler; see
                            # _select_narrow docstring)
                            T.deinterleave(w, e01, o01, channel_nums=2)
                            T.deinterleave(e01, p0, p2, channel_nums=2)
                            T.deinterleave(o01, p1, p3, channel_nums=2)
                            if op_kind == "argmax":
                                T.vmax(p0, p1, t01)
                                T.vmax(p2, p3, t23)
                                T.vmax(t01, t23, row_max)
                            else:
                                T.vmin(p0, p1, t01)
                                T.vmin(p2, p3, t23)
                                T.vmin(t01, t23, row_max)
                            T.vcmp(p0, row_max, cmp0, "eq")
                            T.vcmp(p1, row_max, cmp1, "eq")
                            T.vcmp(p2, row_max, cmp2, "eq")
                            T.vcmp(p3, row_max, cmp3, "eq")
                            T.vselect(cmp0, const0, big_v, c0)
                            T.vselect(cmp1, const1, big_v, c1)
                            T.vselect(cmp2, const2, big_v, c2)
                            T.vselect(cmp3, const3, big_v, c3)
                            T.vmin(c0, c1, k01)
                            T.vmin(c2, c3, k23)
                            T.vmin(k01, k23, first)
                            T.reshape(first, first_flat)
                            for i in T.Parallel(bm):
                                out_ub[i] = T.cast(first_flat[i], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((Mr, Nr), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    w = T.alloc_shared((bw, Nr), dtype)
                    wf = T.alloc_fragment((bw, Nr), "float32")
                    e01 = T.alloc_fragment((bw, 2 * G), "float32")
                    o01 = T.alloc_fragment((bw, 2 * G), "float32")
                    p0 = T.alloc_fragment((bw, G), "float32")
                    p1 = T.alloc_fragment((bw, G), "float32")
                    p2 = T.alloc_fragment((bw, G), "float32")
                    p3 = T.alloc_fragment((bw, G), "float32")
                    row_max = T.alloc_fragment((bw, G), "float32")
                    t01 = T.alloc_fragment((bw, G), "float32")
                    t23 = T.alloc_fragment((bw, G), "float32")
                    cmp0 = T.alloc_fragment((bw, G), "bool")
                    cmp1 = T.alloc_fragment((bw, G), "bool")
                    cmp2 = T.alloc_fragment((bw, G), "bool")
                    cmp3 = T.alloc_fragment((bw, G), "bool")
                    c0 = T.alloc_fragment((bw, G), "float32")
                    c1 = T.alloc_fragment((bw, G), "float32")
                    c2 = T.alloc_fragment((bw, G), "float32")
                    c3 = T.alloc_fragment((bw, G), "float32")
                    k01 = T.alloc_fragment((bw, G), "float32")
                    k23 = T.alloc_fragment((bw, G), "float32")
                    first = T.alloc_fragment((bw, G), "float32")
                    first_flat = T.alloc_fragment((bm,), "float32")
                    const0 = T.alloc_fragment((bw, G), "float32")
                    const1 = T.alloc_fragment((bw, G), "float32")
                    const2 = T.alloc_fragment((bw, G), "float32")
                    const3 = T.alloc_fragment((bw, G), "float32")
                    big_v = T.alloc_fragment((bw, G), "float32")
                    out_ub = T.alloc_shared((bm,), "int64")

                    T.vbrc(T.float32(0), const0)
                    T.vbrc(T.float32(1), const1)
                    T.vbrc(T.float32(2), const2)
                    T.vbrc(T.float32(3), const3)
                    T.vbrc(T.float32(BIG), big_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_w = block_id * bw
                            off_m = block_id * bm
                            real_w = T.min(bw, Mr - off_w)
                            real_m = T.min(bm, M - off_m)
                            T.copy(x[off_w : off_w + real_w, 0:Nr], w[0:real_w, 0:Nr])
                            T.vcast(w, wf, round_mode="rint")
                            # two-level c2 deinterleave tree (see fp16 variant)
                            T.deinterleave(wf, e01, o01, channel_nums=2)
                            T.deinterleave(e01, p0, p2, channel_nums=2)
                            T.deinterleave(o01, p1, p3, channel_nums=2)
                            if op_kind == "argmax":
                                T.vmax(p0, p1, t01)
                                T.vmax(p2, p3, t23)
                                T.vmax(t01, t23, row_max)
                            else:
                                T.vmin(p0, p1, t01)
                                T.vmin(p2, p3, t23)
                                T.vmin(t01, t23, row_max)
                            T.vcmp(p0, row_max, cmp0, "eq")
                            T.vcmp(p1, row_max, cmp1, "eq")
                            T.vcmp(p2, row_max, cmp2, "eq")
                            T.vcmp(p3, row_max, cmp3, "eq")
                            T.vselect(cmp0, const0, big_v, c0)
                            T.vselect(cmp1, const1, big_v, c1)
                            T.vselect(cmp2, const2, big_v, c2)
                            T.vselect(cmp3, const3, big_v, c3)
                            T.vmin(c0, c1, k01)
                            T.vmin(c2, c3, k23)
                            T.vmin(k01, k23, first)
                            T.reshape(first, first_flat)
                            for i in T.Parallel(bm):
                                out_ub[i] = T.cast(first_flat[i], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Tiled path (path T, online running-(extreme, first-index), bm=1 forced)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Leading-axis path (dim == 0 on a contiguous tensor, no transpose)
# ---------------------------------------------------------------------------
# The Op layer used to materialise ``x.movedim(0, -1).contiguous()`` for
# ``dim == 0``.  On the 3-D manifest shape (4, 128, 4096) that copy alone
# measured 17.28 us (msprof, ``Transpose_*``) while the whole reduction was
# 13.86 us.  Reading the same buffer as ``(N, M)`` -- the leading axis is
# already the outer axis of a contiguous tensor -- removes the copy entirely:
# the tile reduces along axis 0 with the dim=0 reduce / vcmp / vselect chain
# already proven by the N-split merge kernel.
LEADING_MAX_N = 8
LEADING_MIN_M = 256
# Tile width: the (N, bm) working set must stay inside the 192 KB UB.  The
# per-element count covers x_ub + val_brc + mask + plane_idx + sent + cand
# (int16 index chain; fp16/fp32 compute directly on x_ub, bf16 adds an fp32
# w upcast buffer); the budget below is the auto-multi-buffered envelope
# (2 x the 64 KB manual budget) minus headroom for out_ub.  Measured on
# (4, 128, 4096) fp16 after the int16/on-x_ub change:
# bm 1024 -> 10.18 us, 1536 -> 8.56, 2048 -> 7.86, 2560 -> 7.18,
# 2816 (fp16 budget cap) -> 7.2-7.3 us.  bf16 caps at 1792, fp32 at 2048.
_LEADING_MAX_BM = 4096
_LEADING_UB_BUDGET = {
    "float16": 160 * 1024,
    "bfloat16": 2 * UB_MANUAL_BUDGET,
    "float32": 2 * UB_MANUAL_BUDGET,
}
_LEADING_PER_ELEM = {"float16": 11, "bfloat16": 17, "float32": 15}


def leading_layout_ok(M, N):
    """Eligibility for the leading-axis layout (2 <= N <= 8, M >= 256)."""
    try:
        M = int(M)
        N = int(N)
    except (TypeError, ValueError):
        return False
    return 2 <= N <= LEADING_MAX_N and M >= LEADING_MIN_M


def _leading_block_m(M, N, dtype, vector_cores=40):
    """Largest 256-aligned tile width inside the manual UB budget.

    Multi-wave shapes take a small wave-balance pass: a tile a few hundred
    columns narrower can fill the last 40-AIV wave.  On the manifest
    (4, 128, 4096) fp16 case the UB cap gives 3584 -> 147 tasks = 4 waves at
    91.9% utilization, while 3328 gives 158 tasks = 4 waves at 98.8%.
    Only tiles within 12.5% of the cap are considered, so the per-tile
    efficiency stays at the measured plateau.
    """
    per_elem = _LEADING_PER_ELEM[dtype] * N
    bm = _LEADING_UB_BUDGET[dtype] // per_elem
    bm = min(bm, _LEADING_MAX_BM, max(256, M))
    bm = (bm // TILE_ALIGNMENT) * TILE_ALIGNMENT
    bm = max(TILE_ALIGNMENT, bm)
    if vector_cores <= 0:
        return bm
    tasks = _ceildiv(M, bm)
    if tasks > vector_cores:
        best_bm, best_util = bm, tasks / (_ceildiv(tasks, vector_cores) * vector_cores)
        floor = max(TILE_ALIGNMENT, bm - bm // 8)
        cand = bm - TILE_ALIGNMENT
        while cand >= floor:
            t = _ceildiv(M, cand)
            util = t / (_ceildiv(t, vector_cores) * vector_cores)
            if util > best_util:
                best_bm, best_util = cand, util
            cand -= TILE_ALIGNMENT
        bm = best_bm
    return bm


def _build_leading(M, N, op_kind, dtype, work_dtype, vector_cores):
    """Reduce the leading axis of a contiguous ``(N, M)`` view.

    ``out[j] = first p in [0, N) attaining max/min_p x[p, j]`` (first
    occurrence, strict-greater update per plane group is unnecessary here:
    the tile holds all N planes at once, so a single dim=0 reduce plus a
    first-index reduce is exact).
    """
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        block_m = int(block_m or TILE_ALIGNMENT)
        block_m = max(TILE_ALIGNMENT, min(block_m, _leading_block_m(M, N, dtype)))
        # ``w`` is the fp32 upcast workspace for bf16 only.  It is still
        # allocated unconditionally (the TVMScript parser resolves names
        # statically); for fp16/fp32 it is a 1-column dummy and unused.
        w_cols = block_m if is_bf16 else 1
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(x: T.Tensor((N, M), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((N, block_m), dtype)
                w = T.alloc_fragment((N, w_cols), work_dtype)
                val = T.alloc_fragment((1, block_m), work_dtype)
                val_brc = T.alloc_fragment((N, block_m), work_dtype)
                mask = T.alloc_fragment((N, block_m), "bool")
                plane_src = T.alloc_shared((N,), "int16")
                plane_col = T.alloc_shared((N, 1), "int16")
                plane_idx = T.alloc_fragment((N, block_m), "int16")
                sent = T.alloc_fragment((N, block_m), "int16")
                cand = T.alloc_fragment((N, block_m), "int16")
                first = T.alloc_fragment((1, block_m), "int16")
                first_flat = T.alloc_fragment((block_m,), "int16")
                first_f32 = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")

                # Task-invariant operands: plane index per row (0..N-1) and the
                # int16 sentinel that loses the first-index reduce_min.  int16
                # is exact for N <= 32767 (leading_layout_ok caps N at 8) and
                # keeps plane_idx/sent/cand at 2 B/elem instead of 4.
                T.arange(plane_src, [1], 0)
                T.reshape(plane_src, plane_col)
                T.vbrc(plane_col, plane_idx)
                T.vbrc(T.int16(32767), sent)

                for s in T.serial(num_local_tasks):
                    block_id = s * num_kernels + cid
                    if block_id < num_logical:
                        off = block_id * block_m
                        real = T.min(block_m, M - off)
                        T.copy(x[0:N, off : off + real], x_ub[0:N, 0:real])
                        if is_bf16:
                            T.vcast(x_ub, w, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(w, val, dim=0)
                            else:
                                T.reduce_min(w, val, dim=0)
                            T.vbrc(val, val_brc)
                            T.vcmp(w, val_brc, mask, "eq")
                        else:
                            # fp16/fp32: compute directly on x_ub; the previous
                            # ``T.copy(x_ub, w)`` was a redundant same-dtype
                            # on-chip copy (one extra vector op + 2-4 B/elem).
                            if op_kind == "argmax":
                                T.reduce_max(x_ub, val, dim=0)
                            else:
                                T.reduce_min(x_ub, val, dim=0)
                            T.vbrc(val, val_brc)
                            T.vcmp(x_ub, val_brc, mask, "eq")
                        T.vselect(mask, plane_idx, sent, cand)
                        T.reduce_min(cand, first, dim=0)
                        T.reshape(first, first_flat)
                        # i16 -> f32 -> i64: vcast has no direct i16 -> i64.
                        T.vcast(first_flat, first_f32, round_mode="rint")
                        T.vcast(first_f32, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real], out[off : off + real])

        return main

    return _func


# Leading-axis streaming variant (fp16): process the N planes one at a time
# with a running (max, first index) pair.  Per-column UB drops from 11*N
# bytes to ~9 (x_p 2 + run_max 2 + run_idx 2 + plane_idx 2 + mask 1), which
# lets one 40-AIV wave cover the whole output: on (4, 128, 4096) the tiled
# form runs 158 tasks = 4 waves @ 98.8% and measures 6.6-6.8 us, while a
# pure read-4-planes + write-int64 probe (same bytes, no index chain) is
# 5.92-5.98 us.  The int64 result is staged in bm/4 chunks so the 8 B/elem
# output does not re-inflate the tile.
_LEADING_STREAM_CHUNKS = 4
_LEADING_STREAM_MAX_BM = 13312


def _leading_stream_block_m(M, vector_cores):
    """One-wave tile width (multiple of 256) inside the streaming UB budget."""
    bm = _ceildiv(M, max(1, vector_cores))
    bm = ((bm + TILE_ALIGNMENT - 1) // TILE_ALIGNMENT) * TILE_ALIGNMENT
    return max(TILE_ALIGNMENT, min(bm, _LEADING_STREAM_MAX_BM))


def _build_leading_stream(M, N, op_kind, dtype, vector_cores):
    """Stream the N planes one at a time, keeping the first-encountered max.

    Tie-break is the strict ``>`` update in increasing plane order, which is
    exactly first-occurrence.  Only fp16/fp32 argmax is routed here; the
    caller keeps the tiled form otherwise.
    """
    if op_kind != "argmax" or dtype not in ("float16", "float32"):
        raise ValueError("stream leading path supports argmax fp16/fp32 only")
    bm = _leading_stream_block_m(M, vector_cores)
    ch = bm // _LEADING_STREAM_CHUNKS
    tasks = _ceildiv(M, bm)
    num_kernels = min(tasks, vector_cores)
    num_local = _ceildiv(tasks, num_kernels)

    @tilelang.jit(
        out_idx=[1],
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _func(_bm):
        @T.prim_func
        def main(x: T.Tensor((N, M), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xp = T.alloc_ub((1, bm), dtype)
                rmax = T.alloc_ub((1, bm), dtype)
                ridx = T.alloc_ub((1, bm), "int16")
                pidx = T.alloc_ub((1, bm), "int16")
                gt = T.alloc_ub((1, bm), "bool")
                f = T.alloc_ub((1, ch), "float32")
                o = T.alloc_ub((1, ch), "int64")

                for s in T.serial(num_local):
                    block_id = s * num_kernels + cid
                    if block_id < tasks:
                        off = block_id * bm
                        real = T.min(bm, M - off)
                        T.copy(x[0:1, off : off + real], rmax[0:1, 0:real])
                        T.vbrc(T.int16(0), ridx)
                        for p in range(1, N):
                            T.copy(x[p : p + 1, off : off + real], xp[0:1, 0:real])
                            T.vcmp(xp, rmax, gt, "gt")
                            T.vmax(xp, rmax, rmax)
                            T.vbrc(T.int16(p), pidx)
                            T.vselect(gt, pidx, ridx, ridx)
                        for ck in range(_LEADING_STREAM_CHUNKS):
                            lo = ck * ch
                            T.vcast(ridx[0:1, lo : lo + ch], f, round_mode="rint")
                            T.vcast(f, o, round_mode="rint")
                            if off + lo < M:
                                ril = T.min(ch, M - off - lo)
                                T.copy(o[0, 0:ril], out[off + lo : off + lo + ril])

        return main

    # Developer codegen silently miscompiles this form (all-zero index output:
    # the running-max pipeline is reordered); the Expert/production path is
    # correct.  Compile here under a scoped mode and hand back the compiled
    # callable, since the dispatch contract invokes ``_f(block_m)`` later,
    # outside this scope.
    _prev_mode = os.environ.get("TILELANG_ASCEND_MODE")
    try:
        os.environ["TILELANG_ASCEND_MODE"] = "expert"
        compiled = _func(bm)
    finally:
        if _prev_mode is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = _prev_mode

    def _entry(_block_m):
        return compiled

    _entry.msprof_kernel_name = "main"
    return _entry


def _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, tile_n):
    """Build the tiled online kernel (bm=1 forced by C-5).

    Two prim_funcs (bf16 / non-bf16). Tile-0 initializes the running state
    directly; full tiles and the static tail tile use the strict-greater
    (argmax) / strict-less (argmin) update rule for first-occurrence tie-break.
    r5 epilogue: the int64 output cast is vectorized via
    ``T.reshape(running_idx, running_flat)`` + ``T.vcast`` (same rank-reduction
    scalarization as the resident path; at the C-5-forced bm=1 it degenerated
    to one scalar extract/fptosi/insert per task). The bm=1
    ``chunk_global`` / ``tail_global`` scalar add stores are left as-is (one
    element each, not the cast pattern).
    """
    is_bf16 = dtype == "bfloat16"
    num_full = N // tile_n
    tail = N - num_full * tile_n
    # Static tail-tile buffers are allocated unconditionally with a min width
    # of 1 (the TVM tracer does not track buffers allocated in a conditional
    # block and referenced later); the tail *processing* is guarded by
    # `if tail > 0` below. The extra (block_m, 1) buffers are negligible.
    tail_w = tail if tail > 0 else 1

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, tile_n), "float32")
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), "float32")
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), "float32")
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    ext_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), "float32")
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
                                    x_tail,
                                )
                                T.vcast(x_tail, xw_tail, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            # r5 epilogue (see _build_resident): the
                            # ``T.Parallel`` cast loop on running_idx[i,0]
                            # (rank-2 load, rank-1 store) stayed scalar
                            # (NpuLoopVectorize rank-reduction guard).
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), work_dtype)
                    chunk_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    ext_brc = T.alloc_fragment((block_m, tile_n), work_dtype)
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    ext_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.copy(x_ub, x_work)
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.copy(x_ub, x_work)
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
                                    x_tail,
                                )
                                T.copy(x_tail, xw_tail)
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            # r5 epilogue (see _build_resident): the
                            # ``T.Parallel`` cast loop on running_idx[i,0]
                            # (rank-2 load, rank-1 store) stayed scalar
                            # (NpuLoopVectorize rank-reduction guard).
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _argreduce_kernel(M, N, op_kind, dtype) -> _func(block_m))
# ---------------------------------------------------------------------------


def _build_nsplit_split(
    M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks=None
):
    """Two-launch N-split (partial + merge), Developer-mode safe variant.

    The fused r7 form relies on an inter-block FFTS barrier and sub-block
    launch, which deadlocks on the 910B3 (20 AIC) target.  This variant keeps
    the same N-split algorithm but uses two ordinary launches: the partial
    kernel writes the (nchunk, M) workspace, the merge kernel reduces it.
    """
    is_bf16 = dtype == "bfloat16"
    num_lanes = min(nchunk, vector_cores)
    num_local_tasks = _ceildiv(nchunk, num_lanes)
    val_dtype = work_dtype

    @tilelang.jit(target="npuir")
    def _partial_jit():
        @T.prim_func
        def argreduce_partial(
            x: T.Tensor((M, N), dtype),
            ws_val: T.Tensor((nchunk, M), val_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
        ):
            with T.Kernel(num_lanes, is_npu=True) as (cid, _):
                x_r = T.alloc_shared((1, tn), dtype)
                w_r = T.alloc_shared((1, tn), val_dtype)
                m_r = T.alloc_shared((1, 1), val_dtype)
                e_r = T.alloc_shared((1, tn), val_dtype)
                f_r = T.alloc_shared((1, 1), "float32")
                ext_out = T.alloc_shared((M,), val_dtype)
                gidx_out = T.alloc_shared((M,), "float32")
                p_idx_src = T.alloc_shared((tn,), "float32")
                p_idx_j = T.alloc_shared((1, tn), "float32")
                p_sent_v = T.alloc_shared((1, tn), "float32")
                p_cmp = T.alloc_shared((1, tn), "bool")
                p_cand = T.alloc_shared((1, tn), "float32")

                T.arange(p_idx_src, [1], 0)
                T.reshape(p_idx_src, p_idx_j)
                T.vbrc(T.float32(BIG), p_sent_v)

                for s in T.serial(num_local_tasks):
                    c = cid + s * num_lanes
                    if c < nchunk:
                        for r in range(M):
                            T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                            if is_bf16:
                                T.vcast(x_r, w_r, round_mode="rint")
                            else:
                                T.copy(x_r, w_r)
                            if op_kind == "argmax":
                                T.reduce_max(w_r, m_r, dim=1)
                            else:
                                T.reduce_min(w_r, m_r, dim=1)
                            T.vbrc(m_r, e_r)
                            T.vcmp(w_r, e_r, p_cmp, "eq")
                            T.vselect(p_cmp, p_idx_j, p_sent_v, p_cand)
                            T.reduce_min(p_cand, f_r, dim=1)
                            ext_out[r] = m_r[0, 0]
                            gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                        T.copy(ext_out[0:M], ws_val[c, 0:M])
                        T.copy(gidx_out[0:M], ws_idx[c, 0:M])

        return argreduce_partial

    @tilelang.jit(out_idx=[2], target="npuir")
    def _merge_jit():
        @T.prim_func
        def argreduce_merge(
            ws_val: T.Tensor((nchunk, M), val_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                val_ub = T.alloc_shared((nchunk, M), val_dtype)
                idx_ub = T.alloc_shared((nchunk, M), "float32")
                g = T.alloc_shared((1, M), val_dtype)
                g_brc = T.alloc_shared((nchunk, M), val_dtype)
                cmp_eq = T.alloc_shared((nchunk, M), "bool")
                sent_v = T.alloc_shared((nchunk, M), "float32")
                cand = T.alloc_shared((nchunk, M), "float32")
                first = T.alloc_shared((1, M), "float32")
                first_flat = T.alloc_shared((M,), "float32")
                out_ub = T.alloc_shared((M,), "int64")

                T.vbrc(T.float32(BIG), sent_v)
                T.copy(ws_val, val_ub)
                T.copy(ws_idx, idx_ub)
                if op_kind == "argmax":
                    T.reduce_max(val_ub, g, dim=0)
                else:
                    T.reduce_min(val_ub, g, dim=0)
                T.vbrc(g, g_brc)
                T.vcmp(val_ub, g_brc, cmp_eq, "eq")
                T.vselect(cmp_eq, idx_ub, sent_v, cand)
                T.reduce_min(cand, first, dim=0)
                T.reshape(first, first_flat)
                T.vcast(first_flat, out_ub, round_mode="rint")
                T.copy(out_ub[0:M], out[0:M])

        return argreduce_merge

    partial = _partial_jit()
    merge = _merge_jit()

    def _func(block_m):
        def main(x):
            ws_val = torch.empty((nchunk, M), dtype=_DTYPE_MAP[val_dtype], device=x.device)
            ws_idx = torch.empty((nchunk, M), dtype=torch.float32, device=x.device)
            partial(x, ws_val, ws_idx)
            return merge(ws_val, ws_idx)

        return main

    _func.partial = partial
    _func.merge = merge
    # msprof anchor: this dispatch launches two kernels per Op forward, so the
    # benchmark harness must sum one sample per stage (see
    # tileops.benchmark.msprof.bench_kernel_msprof) instead of taking a median
    # across the partial/merge mix.
    _func.msprof_kernel_names = ("argreduce_partial", "argreduce_merge")
    return _func


def _nsplit_rows_config(M, N, dtype, vector_cores, tn=None):
    """Config for the rows-parallel N-split partial (dual-dst reduce).

    The historical partial walks the M rows *serially* (four dependent
    reduce chains), so its latency is per-row chain latency, not throughput:
    on (4, 102400) every ``tn`` from 2560 to 10240 measures 5.4-6.0 us even
    though the data is tiny.  The rows variant keeps the same (partial +
    merge) two-launch contract but reduces all M rows of one chunk inside a
    single tile with the fused dual-dst ``reduce_argmax`` (max + index in one
    pass), which removes both the serial row walk and the 5-op index chain.

    Tile-width rule: the partial is throughput-bound once the rows overlap,
    while the single-block merge grows with ``nchunk``; prefer the width
    whose chunk count is closest to 16 (measured knee on lm-head shapes).
    """
    if M <= 0 or M > _NSPLIT_MAX_M:
        return None
    per_col = 2 * M  # x_r (M, tn) fp16 payload; reduce dsts are (M, 1)
    tn_cap = (160 * 1024) // per_col
    if tn is not None:
        if tn <= 0 or N % tn or tn > tn_cap:
            return None
        nc = N // tn
        return {"tn": tn, "nchunk": nc} if nc >= 2 else None
    nc_lo, nc_hi = 8, min(vector_cores, 24)
    best = None
    tn_scan = 256
    while tn_scan * 2 <= N:
        if N % tn_scan == 0 and tn_scan <= tn_cap:
            nc = N // tn_scan
            if nc_lo <= nc <= nc_hi:
                key = (abs(nc - 16), -tn_scan)
                if best is None or key < best[0]:
                    best = (key, tn_scan, nc)
        tn_scan += 256
    if best is not None:
        return {"tn": best[1], "nchunk": best[2]}
    return None


def _build_nsplit_split_rows(
    M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks=None
):
    """Rows-parallel N-split partial + the unchanged merge kernel.

    The partial reduces a ``(M, tn)`` tile in one fused dual-dst
    ``reduce_argmax`` (Expert mode; Developer silently drops the second dst),
    so the M rows overlap instead of forming a serial chain.  Tie-break is
    preserved: the fused reduce returns the first index within the chunk and
    the merge keeps the first chunk attaining the maximum.
    """
    num_lanes = min(nchunk, vector_cores)
    num_local_tasks = _ceildiv(nchunk, num_lanes)
    val_dtype = work_dtype

    @tilelang.jit(
        target="npuir",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    def _partial_jit():
        @T.prim_func
        def argreduce_partial(
            x: T.Tensor((M, N), dtype),
            ws_val: T.Tensor((nchunk, M), val_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
        ):
            with T.Kernel(num_lanes, is_npu=True) as (cid, _):
                x_r = T.alloc_ub((M, tn), dtype)
                m_r = T.alloc_ub((M, 1), val_dtype)
                i_r = T.alloc_ub((M, 1), "int32")
                m_flat = T.alloc_ub((M,), val_dtype)
                i_flat = T.alloc_ub((M,), "int32")
                gidx_out = T.alloc_ub((M,), "float32")
                base = T.alloc_ub((M,), "float32")

                for s in T.serial(num_local_tasks):
                    c = cid + s * num_lanes
                    if c < nchunk:
                        T.copy(x[0:M, c * tn : (c + 1) * tn], x_r)
                        _tl_reduce_argmax(x_r, m_r, i_r, dim=1)
                        # UB -> GM through vector copies only: a scalar read of
                        # the reduce dst right after the fused reduce has no
                        # ordered V->scalar hand-off and returned garbage.
                        T.reshape(m_r, m_flat)
                        T.reshape(i_r, i_flat)
                        T.vcast(i_flat, gidx_out, round_mode="rint")
                        T.vbrc(T.float32(c * tn), base)
                        T.vadd(gidx_out, base, gidx_out)
                        T.copy(m_flat[0:M], ws_val[c, 0:M])
                        T.copy(gidx_out[0:M], ws_idx[c, 0:M])

        return argreduce_partial

    @tilelang.jit(out_idx=[2], target="npuir")
    def _merge_jit():
        @T.prim_func
        def argreduce_merge(
            ws_val: T.Tensor((nchunk, M), val_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                val_ub = T.alloc_shared((nchunk, M), val_dtype)
                idx_ub = T.alloc_shared((nchunk, M), "float32")
                g = T.alloc_shared((1, M), val_dtype)
                g_brc = T.alloc_shared((nchunk, M), val_dtype)
                cmp_eq = T.alloc_shared((nchunk, M), "bool")
                sent_v = T.alloc_shared((nchunk, M), "float32")
                cand = T.alloc_shared((nchunk, M), "float32")
                first = T.alloc_shared((1, M), "float32")
                first_flat = T.alloc_shared((M,), "float32")
                out_ub = T.alloc_shared((M,), "int64")

                T.vbrc(T.float32(BIG), sent_v)
                T.copy(ws_val, val_ub)
                T.copy(ws_idx, idx_ub)
                if op_kind == "argmax":
                    T.reduce_max(val_ub, g, dim=0)
                else:
                    T.reduce_min(val_ub, g, dim=0)
                T.vbrc(g, g_brc)
                T.vcmp(val_ub, g_brc, cmp_eq, "eq")
                T.vselect(cmp_eq, idx_ub, sent_v, cand)
                T.reduce_min(cand, first, dim=0)
                T.reshape(first, first_flat)
                T.vcast(first_flat, out_ub, round_mode="rint")
                T.copy(out_ub[0:M], out[0:M])

        return argreduce_merge

    # The dual-dst reduce needs the Expert pipeline (Developer silently drops
    # the index dst); the merge kernel keeps the Developer path because its
    # bool ``vcmp`` operand does not lower in Expert ("cannot find variable
    # cmp_eq").  ``TILELANG_ASCEND_MODE`` is read per compile call, so scoping
    # it around each jit call compiles exactly one stage in the chosen mode.
    _prev_mode = os.environ.get("TILELANG_ASCEND_MODE")
    try:
        os.environ["TILELANG_ASCEND_MODE"] = "expert"
        partial = _partial_jit()
        os.environ["TILELANG_ASCEND_MODE"] = "Developer"
        merge = _merge_jit()
    finally:
        if _prev_mode is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = _prev_mode

    def _func(block_m):
        def main(x):
            ws_val = torch.empty(
                (nchunk, M), dtype=_DTYPE_MAP[val_dtype], device=x.device
            )
            ws_idx = torch.empty((nchunk, M), dtype=torch.float32, device=x.device)
            partial(x, ws_val, ws_idx)
            return merge(ws_val, ws_idx)

        return main

    _func.partial = partial
    _func.merge = merge
    _func.ascend_mode = "Expert"
    _func.msprof_kernel_names = ("argreduce_partial", "argreduce_merge")
    return _func


def _build_nsplit_dispatch(
    M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks=None
):
    """Two-launch N-split (partial + merge) for 910B3 safety."""
    rows_env = os.environ.get(_NSPLIT_ROWS_ENV, "1")
    if (
        rows_env != "0"
        and _tl_reduce_argmax is not None
        and op_kind == "argmax"
        and dtype == "float16"
        and 2 <= M <= _NSPLIT_MAX_M
    ):
        tn_env = os.environ.get(_NSPLIT_TN_ENV)
        rcfg = _nsplit_rows_config(
            M, N, dtype, vector_cores, int(tn_env) if tn_env else None
        )
        if rcfg is not None:
            return _build_nsplit_split_rows(
                M,
                N,
                op_kind,
                dtype,
                work_dtype,
                vector_cores,
                rcfg["tn"],
                rcfg["nchunk"],
                num_blocks,
            )
    return _build_nsplit_split(
        M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks
    )


# ``kernel_factory_cache`` memoises both the factory and the build callable it
# returns: the custom_op wrapper calls ``_argreduce_kernel(...)(block_m)(x)``
# on every forward, and rebuilding the nested ``@tilelang.jit`` declarations
# costs ~115 ms per call (tilelang's own cache only skips the compiler run, not
# the PrimFunc/JIT-wrapper construction).
@kernel_factory_cache
def _argreduce_kernel(M, N, op_kind, dtype, layout="trailing"):
    """Build the argmax/argmin NPU kernel factory.

    ``layout="leading"`` selects the dim-0 (leading-axis) kernel: ``x`` is a
    contiguous ``(N, M)`` view and the reduction runs along axis 0, so the Op
    layer must not transpose (see ``_build_leading``).

    Mirrors the GPU source structure ``_argreduce_kernel(M, N, op_kind, dtype)
    -> _func(block_m) -> main``. K9: the ``threads`` backend parameter is
    removed (NPU has no CUDA threads concept); the backend config is the
    UB-budget-driven block_m / tile_n (DESIGN.md sections 5.2 / 5.5).
    """
    if op_kind not in _KINDS:
        raise ValueError(f"unsupported op_kind {op_kind!r}; expected one of {sorted(_KINDS)}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")

    cfg = _select_config(M, N, dtype)
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    if layout == "leading":
        if not leading_layout_ok(M, N):
            raise ValueError(
                f"layout='leading' requires 2 <= N <= {LEADING_MAX_N} and "
                f"M >= {LEADING_MIN_M}; got M={M}, N={N}"
            )
        if (
            os.environ.get("TILEOPS_ARGMAX_LEADING_STREAM", "0") == "1"
            and op_kind == "argmax"
            and dtype == "float16"
        ):
            return _build_leading_stream(M, N, op_kind, dtype, vector_cores)
        _f = _build_leading(M, N, op_kind, dtype, work_dtype, vector_cores)
        _f.msprof_kernel_name = "main"
        return _f

    # Fused argmax (CG-2026-0031): fp16 trailing-axis shapes with an aligned
    # reduction width.  Selected automatically when the patched dual-dst
    # frontend is importable; TILEOPS_ARGMAX_FUSED=0 forces the chain path.
    if (
        _FUSED_ARGMAX
        and op_kind == "argmax"
        and dtype == "bfloat16"
        and layout == "trailing"
        and N == 4096
        and M >= 1024
        and M % 4 == 0
    ):
        # P1 (bitcast route): bf16 and fp16 are both 16-bit sign-exponent-
        # mantissa encodings, and fp16's decoded value is a *monotone* function
        # of the 16-bit pattern (same-sign patterns order like their values, the
        # sign bit splits positives above negatives).  The fp16 group4 kernel
        # therefore compares the bf16 payload exactly like bf16 would, with no
        # cast and no f32 workspace: UB traffic per tile drops from 80 KiB to
        # 32 KiB (measured 23.98 -> 15.82 us on (2048, 4096)).  Only the value
        # is meaningless -- argmax discards it; the index is bit-exact.
        # Contract: no NaN/Inf and no fp16-subnormal payload; see
        # ARGMAX_BITCAST_MIN_ABS below.
        split = _build_fused_argmax_group4(M, N, "float16")
        split.bitcast_input = "float16"
        return split
    if (
        _FUSED_ARGMAX
        and op_kind == "argmax"
        and dtype == "bfloat16"
        and layout == "trailing"
        and N == 4096
        and M >= 1024
        and M % 2 == 0
    ):
        return _build_fused_argmax_bf16_group3(M, N)
    if (
        _FUSED_ARGMAX
        and op_kind == "argmax"
        and dtype == "float16"
        and layout == "trailing"
        and N == 4096
        and M >= 1024
        and M % 4 == 0
    ):
        # P3.6 group4 is tuned for the hidden-state shape class.
        return _build_fused_argmax_group4(M, N, dtype)
    if (
        _FUSED_ARGMAX
        and op_kind == "argmax"
        and dtype == "float16"
        and _FUSED_MIN_N <= N <= _FUSED_MAX_N
        and N % 16 == 0
    ):
        return _build_fused_argmax(M, N, dtype)

    # ROUND-3-F: C6 N-split dispatch (DESIGN.md 1.6.4 adjudication flipped:
    # fp16 11.92us vs single 25.90us = 0.461x; bf16 11.83us vs 30.69us =
    # 0.386x; both < the 0.8x pre-registered threshold).
    ncfg = _nsplit_config(M, N, dtype, vector_cores)
    if ncfg is not None:
        # P1b: the lm-head class (M <= 16, N >= 32768) also only needs ordering,
        # so bf16 rides the fp16 reduce through the same zero-copy bitcast as the
        # hidden-state route: the partial kernel drops its f32 upcast workspace
        # (10 B/elem -> 4 B/elem).  Contract identical: no NaN/Inf and no fp16
        # subnormal payload (see ARGMAX_BITCAST_MIN_ABS).
        _bitcast = None
        if dtype == "bfloat16" and op_kind == "argmax" and layout == "trailing":
            _f16cfg = _nsplit_config(M, N, "float16", vector_cores)
            if _f16cfg is not None:
                ncfg = _f16cfg
                dtype = "float16"
                work_dtype = "float16"
                _bitcast = "float16"
        _f = _build_nsplit_dispatch(
            M,
            N,
            op_kind,
            dtype,
            work_dtype,
            vector_cores,
            ncfg["tn"],
            ncfg["nchunk"],
        )
        if _bitcast is not None:
            _f.bitcast_input = _bitcast
        # Two device launches (partial + merge).  ``msprof_kernel_name`` names
        # the dominant stage for single-kernel filters; the harness sums both
        # stages via ``msprof_kernel_names``.
        _f.msprof_kernel_name = "argreduce_partial"
        return _f

    if cfg["path"] == "narrow":
        _f = _build_narrow(
            M,
            N,
            op_kind,
            dtype,
            work_dtype,
            vector_cores,
            cfg["G"],
            cfg["bw"],
        )
        _f.msprof_kernel_name = "main"
        return _f

    if cfg["path"] == "resident":
        # r6: the resident kernel builds its column indices in-kernel (1-D
        # arange -> reshape -> vbrc; see _build_resident) -- no GM idx input,
        # no host-side injection wrapper. The call API stays
        # ``_argreduce_kernel(...)(block_m)(x)`` for the TileOPs wrapper, the
        # L0 suite and the bench, and build_case's resident branch is a plain
        # single-input kernel again.
        _f = _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores)
        _f.msprof_kernel_name = "main"
        return _f
    _f = _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, cfg["tile_n"])
    _f.msprof_kernel_name = "main"
    return _f


# ---------------------------------------------------------------------------
# Precision comparison helper
# ---------------------------------------------------------------------------
def _run_case(M, N, op_kind, dtype_str, tag, expect_block_m=None, expect_tile_n=None):
    torch_dtype = _DTYPE_MAP[dtype_str]
    cfg = _select_config(M, N, dtype_str)
    block_m = cfg["block_m"]

    if expect_block_m is not None and block_m != expect_block_m:
        raise AssertionError(
            f"[{tag}] block_m={block_m} != expected {expect_block_m} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    if expect_tile_n is not None and cfg["tile_n"] != expect_tile_n:
        raise AssertionError(
            f"[{tag}] tile_n={cfg['tile_n']} != expected {expect_tile_n} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )

    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(block_m)
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)

    assert y.dtype == torch.int64, f"[{tag}] output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] mismatch shape=({M},{N}) dtype={dtype_str} op={op_kind}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )

    ncfg = _nsplit_config(M, N, dtype_str, NPUUtils.get().get_aicore_num() * 2)
    eff = f"nsplit(tn={ncfg['tn']},nchunk={ncfg['nchunk']})" if ncfg else cfg["path"]
    extra = f" tile_n={cfg['tile_n']}" if cfg["tile_n"] is not None else ""
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} op={op_kind} "
        f"path={eff} block_m={block_m}{extra}"
    )


# ---------------------------------------------------------------------------
# L0: gate tests (must pass) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def run_L0():
    cases = [
        # (M, N, dtype, op_kind, expect_block_m, expect_tile_n)
        (32, 256, "float16", "argmax", 8, None),  # L0-1a smoke fp16
        (32, 256, "float32", "argmax", 8, None),  # L0-1b smoke fp32
        (32, 256, "bfloat16", "argmax", 8, None),  # L0-1c smoke bf16
        (4, 102400, "float16", "argmax", 1, 5120),  # L0-2 lm-head fp16 tiled
        (4, 102400, "bfloat16", "argmax", 1, 4096),  # L0-3 lm-head bf16 tiled
        (2048, 4096, "float16", "argmax", 2, None),  # L0-4 hidden-state fp16
        (2048, 4096, "bfloat16", "argmax", 1, None),  # L0-5a hidden-state bf16
        (2048, 4096, "float32", "argmax", 1, None),  # L0-5b hidden-state fp32
        (4096, 4, "float16", "argmax", 1024, None),  # L0-6a 3d quick regression (narrow deint path)
        (
            524288,
            4,
            "float16",
            "argmax",
            1024,
            None,
        ),  # L0-6b 3d manifest true shape (narrow deint path)
        (128, 300, "float16", "argmax", 8, None),  # L0-7a N unaligned
        (128, 300, "bfloat16", "argmax", 8, None),  # L0-7b N unaligned bf16
        (129, 512, "float16", "argmax", 8, None),  # L0-7c M tail (129 % 8 = 1)
        (1, 512, "float16", "argmax", 1, None),  # L0-8a single row resident
        (1, 102400, "float16", "argmax", 1, 5120),  # L0-8b single row tiled
        # L0-10 argmin mirror
        (32, 256, "float16", "argmin", 8, None),
        (32, 256, "float32", "argmin", 8, None),
        (32, 256, "bfloat16", "argmin", 8, None),
        (4, 102400, "float16", "argmin", 1, 5120),
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L0", expect_block_m=bm, expect_tile_n=tn)

    # L0-9 constructed edge rows (tie / all -inf / all +inf / +-inf mix).
    _run_edge_cases("float16", "argmax", "L0")
    _run_edge_cases("float32", "argmin", "L0")
    print(f"[L0] ALL PASS: {len(cases)} shape cases + edge cases")


def _run_edge_cases(dtype_str, op_kind, tag):
    torch_dtype = _DTYPE_MAP[dtype_str]
    N = 300
    rows = []
    # Row 0: duplicate max at col 5 and col 200 -> 5 (first-occurrence tie).
    r = torch.randn(N) * 0.1
    r[5] = 7.0
    r[200] = 7.0
    rows.append(r)
    # Row 1: all -inf -> 0.
    rows.append(torch.full((N,), float("-inf")))
    # Row 2: all +inf -> 0.
    rows.append(torch.full((N,), float("inf")))
    # Row 3: +inf among -inf (argmax) / -inf among +inf (argmin) -> 10.
    r = torch.full((N,), float("-inf") if op_kind == "argmax" else float("inf"))
    r[10] = float("inf") if op_kind == "argmax" else float("-inf")
    rows.append(r)

    x = torch.stack(rows).to(dtype=torch_dtype, device="npu")
    M = x.shape[0]
    cfg = _select_config(M, N, dtype_str)
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(cfg["block_m"])
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)
    assert y.dtype == torch.int64, f"[{tag}] edge output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] edge mismatch op={op_kind} dtype={dtype_str}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )
    print(f"[{tag}] PASS: edge rows M={M} N={N} dtype={dtype_str} op={op_kind}")


# ---------------------------------------------------------------------------
# L1: functional coverage (must pass)
# ---------------------------------------------------------------------------
def run_L1():
    cases = [
        (128, 512, "float32", "argmax", 8, None),
        (128, 512, "bfloat16", "argmax", 8, None),
        (130, 256, "float16", "argmax", 8, None),  # M tail 130 % 8 = 2
        (64, 1024, "float16", "argmax", 8, None),
        (256, 4096, "float16", "argmax", 2, None),  # persistent multi-wave
        (3, 300, "bfloat16", "argmin", 3, None),  # small M + argmin (bm capped to M)
        (8, 32768, "float16", "argmax", 1, 4096),  # tiled fp16 multi-tile
        (128, 16384, "float32", "argmax", 1, 2048),  # tiled fp32 multi-tile
        (4, 8192, "bfloat16", "argmin", 1, 4096),  # tiled bf16 argmin
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L1", expect_block_m=bm, expect_tile_n=tn)
    print(f"[L1] ALL PASS: {len(cases)} cases")


# ---------------------------------------------------------------------------
# L2: boundary / known-divergence (warn only, non-blocking) -- DESIGN.md 8.2
# ---------------------------------------------------------------------------
def run_L2():
    cases = [
        (1, 1, "float16", "argmax", 1, None),  # minimal
        (1, 2, "float16", "argmax", 1, None),  # N=2
        (2, 1, "float16", "argmax", 2, None),  # N=1
        (9, 256, "float16", "argmax", 8, None),  # M = block_m + 1
        (4, 7000, "float16", "argmax", 1, 6400),  # tiled static tail (7000 % 6400 = 600)
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        try:
            _run_case(M, N, op_kind, dtype_str, "L2", expect_block_m=bm, expect_tile_n=tn)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} {op_kind}: {e}")

    # NaN row: NPU reduce propagates NaN -> sentinel 2**30; torch returns first
    # NaN index. Documented divergence (DESIGN.md section 9.2-R1) -> record only.
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        cfg = _select_config(4, 256, "float16")
        y = _argreduce_kernel(4, 256, "argmax", "float16")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[L2] NaN record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): NaN row: {e}")

    # N > 2^24 factory assertion (raises, does not silently lose precision).
    try:
        _select_config(1, 2**24 + 1, "float16")
        print("[L2] WARN: N>2^24 did not raise (unexpected)")
    except (ValueError, AssertionError) as e:
        print(f"[L2] N>2^24 raise OK: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N>2^24: {e}")

    # M=0 boundary (factory raises; Op layer short-circuits in harness).
    try:
        _argreduce_kernel(0, 256, "argmax", "float16")
        print("[L2] WARN: M=0 did not raise (unexpected)")
    except ValueError as e:
        print(f"[L2] M=0 raise OK: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): M=0: {e}")


# ---------------------------------------------------------------------------
# Boundary: extreme values (warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    # Mixed +-0.0 rows: IEEE-equal match vs torch-CPU golden (record only,
    # DESIGN.md section 9.2-R2 notes device-kernel divergence).
    try:
        x = torch.zeros(4, 256, dtype=torch.float32, device="npu")
        x[1, 3] = 1.0
        x[2, 7] = -0.0
        cfg = _select_config(4, 256, "float32")
        y = _argreduce_kernel(4, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[Boundary] +-0.0 record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0.0: {e}")

    # fp32 subnormal extreme: verify first-occurrence still exact.
    try:
        x = torch.zeros(8, 256, dtype=torch.float32, device="npu")
        sub = torch.tensor(1e-40, dtype=torch.float32)  # fp32 subnormal
        x[0, 50] = sub
        cfg = _select_config(8, 256, "float32")
        y = _argreduce_kernel(8, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        assert torch.equal(y.cpu(), ref), f"subnormal mismatch {y.cpu()} vs {ref}"
        print("[Boundary] PASS: fp32 subnormal extreme")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): fp32 subnormal: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all", "nsplit"])
    parser.add_argument(
        "--tn",
        type=int,
        default=None,
        help="nsplit chunk-width override (precision-gates the A/B config)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="nsplit AI-core block-count override (single-wave limit 24)",
    )
    args, _ = parser.parse_known_args()
    if args.level == "nsplit":
        run_nsplit_L0(tn=args.tn, num_blocks=args.blocks)
    elif args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


# ---------------------------------------------------------------------------
# C6 N-split fused partial+merge experiment
# ---------------------------------------------------------------------------
# Dispatch predicate (DESIGN.md 1.6.4-(1)): M <= 16, ceildiv(M, 1) < 48
# vector cores, N >= 32768 (lm-head class; the single-kernel tiled path is
# forced to bm=1 by trap C-5, leaving M/48 cores utilized).
_NSPLIT_MAX_M = 16
_NSPLIT_MIN_N = 32768
# Single-block merge working-set cap (prev-task round9 form: M*nchunk elems).
_NSPLIT_MERGE_ELEMS = 6144

# Chain-form manual bytes per element over the (M, tn) working set:
# x_ub + ext_brc + cmp(bool) + idx_j + sent_v + cand (+ fp32 x_work for bf16).
_NSPLIT_CHAIN_B = {"float16": 17, "bfloat16": 23, "float32": 21}
# Empirically stable chain footprint ceiling: the hidden-state fp16 chain
# (2, 4096) allocates 17B/elem * 8192 elems = 139264B manual and is stable
# (r1a/r2a L0 + profiles); reused as the manual-bytes ceiling.
_NSPLIT_MANUAL_CAP = 139264


def _nsplit_config(M, N, dtype, vector_cores, tn=None):
    """C6 dispatch predicate + (tn, nchunk) derivation (DESIGN.md 1.6.4).

    Returns ``{"tn", "nchunk"}`` or None when the shape should stay on the
    single-kernel path. Predicate (DESIGN.md 1.6.4-(1)): M <= 16,
    ceildiv(M, 1) < vector cores, N >= 32768 (lm-head class; the tiled
    single kernel is forced to bm=1 by trap C-5 and leaves M/48 cores
    utilized).

    tn rules:

    - ``tn`` argument (R7-e): explicit override -- only divisibility and
      the manual-bytes budget are validated (``build_case`` routes its
      ``tile_n`` here; the ``--tn`` CLI feeds ``run_nsplit_L0``).

    - default (R7-e calibrated, supersedes the r3e rule): with the r7-d
      chain-form partial the kernel is fixed-overhead dominated (barrier
      rendezvous + merge + per-lane chain latency), so FEWER, LARGER
      chunks win: prefer the largest tn whose nchunk lands in the window
      [vector_cores//3, vector_cores//2] (= [16, 24] lanes on 910B2C).
      Full sweep (data/1014 report section 4.1, (4,102400) fp16 chain):
      tn 2560/3200/5120/6400/10240 (nc 40/32/20/16/10) -> 12.73/11.62/
      10.14/10.23/10.58us -- optimum tn=5120 (nc=20, 10 blocks/20 lanes),
      1.20x faster than the r3e pick (tn=2560, 12.73us) and 1.20x faster
      than the dual-kernel baseline. Falls back to the r3e ascending scan
      (first divisor with 2 <= nc <= max_nc) when no divisor hits the
      window. Manual budget: tn * B_row(dtype), B_row = x_r + w_r + e_r
      + chain operands.
    """
    B_row = {"float16": 10, "bfloat16": 14, "float32": 16}.get(dtype)
    if B_row is None:
        return None
    if M > _NSPLIT_MAX_M or N < _NSPLIT_MIN_N or vector_cores <= M:
        return None
    if N % 256 != 0:
        return None
    max_nc = min(vector_cores, N // 256, _NSPLIT_MERGE_ELEMS // M)
    if max_nc < 2:
        return None
    if tn is not None:
        if N % tn != 0:
            return None
        nc = N // tn
        if nc < 2 or nc > max_nc or tn * B_row > _NSPLIT_MANUAL_CAP:
            return None
        return {"tn": tn, "nchunk": nc}
    # R7-e calibrated rule: largest tn with nchunk in the window
    # [vector_cores//3, min(max_nc, vector_cores//2)].
    nc_lo = max(2, vector_cores // 3)
    nc_hi = min(max_nc, vector_cores // 2)
    if nc_lo <= nc_hi:
        best = None
        tn_scan = 256
        while tn_scan * 2 <= N:
            if N % tn_scan == 0:
                nc = N // tn_scan
                if (
                    nc_lo <= nc <= nc_hi
                    and tn_scan * B_row <= _NSPLIT_MANUAL_CAP
                    and (best is None or tn_scan > best[0])
                ):
                    best = (tn_scan, nc)
            tn_scan += 256
        if best is not None:
            return {"tn": best[0], "nchunk": best[1]}
    # r3e fallback: ascending scan, first divisor with 2 <= nc <= max_nc.
    tn = 256
    while tn * 2 <= N:
        if N % tn == 0:
            nc = N // tn
            if 2 <= nc <= max_nc and tn * B_row <= _NSPLIT_MANUAL_CAP:
                return {"tn": tn, "nchunk": nc}
        tn += 256
    return None


def _build_nsplit_fused(
    M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks=None
):
    """Fused N-split kernel (r7): expert mode, sub-block 1-to-2 split.

    Launch topology (R7-e: ``num_blocks`` overridable): default
    ``num_blocks = ceil(nchunk / 2)`` AI-core blocks (clamped to
    aicore_num; nchunk <= vector_cores = 2 * aicore_num keeps this <=
    aicore_num, satisfying the single-wave FFTS limit of T.block_barrier).
    An explicit ``num_blocks`` (``--blocks`` CLI / ``build_case`` /
    ``build_nsplit`` argument) is clamped to [1, aicore_num]; lanes then cover the chunks by grid-stride
    (``ceil(nchunk / (2 * num_blocks))`` tasks per lane). Measured
    sensitivity (data/1014 section 4.2): fewer lanes than chunks
    (multi-chunk lanes) degrades proportionally; more lanes than chunks
    are idle -- 1 chunk per lane with the largest viable chunk is optimal.

    The partial phase keeps the r3e per-row (1, tn) form (contiguous GM
    reads, every op the tiled-path-proven shape). The merge runs on lane 0
    only: contiguous (nchunk, M) workspace read + dim-0 reduces -- no
    transpose (r7-c). All buffers are T.alloc_ub inside T.Scope("Vector")
    (expert pure-vector form; auto-promoted to MIX so the inter-block
    barrier's FFTS handshake completes -- probe_blockdim40.py evidence).

    Compiled under ``TILELANG_ASCEND_MODE=expert`` regardless of the
    module-level Developer default: the env is switched around the jit
    call and restored, so every other path of this file is unaffected.
    """
    if num_blocks is None:
        num_blocks = _ceildiv(nchunk, 2)
    aicore_num = NPUUtils.get().get_aicore_num()
    if not 1 <= num_blocks <= aicore_num:
        raise ValueError(
            f"num_blocks={num_blocks} out of range [1, {aicore_num}] "
            f"(single-wave FFTS limit of T.block_barrier)"
        )
    num_vcores = num_blocks * 2
    num_local_tasks = _ceildiv(nchunk, num_vcores)
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[3], target="npuir")
    def _fused_jit(block_m):
        # block_m accepted for interface uniformity; must equal M.

        if is_bf16:

            @T.prim_func
            def argreduce_nsplit_fused(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk, M), work_dtype),
                ws_idx: T.Tensor((nchunk, M), "float32"),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_blocks, is_npu=True) as (cid, subid), T.Scope("Vector"):
                    # Partial-phase buffers (every V core lane).
                    x_r = T.alloc_ub((1, tn), dtype)
                    w_r = T.alloc_ub((1, tn), "float32")
                    m_r = T.alloc_ub((1, 1), "float32")
                    e_r = T.alloc_ub((1, tn), "float32")
                    f_r = T.alloc_ub((1, 1), "float32")
                    ext_out = T.alloc_ub((M,), "float32")
                    gidx_out = T.alloc_ub((M,), "float32")
                    # r7-d chain-form operands (see the non-bf16 variant
                    # note below for why the fused-ite form is gone).
                    p_idx_src = T.alloc_ub((tn,), "float32")
                    p_idx_j = T.alloc_ub((1, tn), "float32")
                    p_sent_v = T.alloc_ub((1, tn), "float32")
                    p_cmp = T.alloc_ub((1, tn), "bool")
                    p_cand = T.alloc_ub((1, tn), "float32")

                    # Merge buffers. TVM tracing requires allocations
                    # to dominate the runtime conditional, so every
                    # lane declares them; only lane vid==0 executes
                    # the merge below.
                    val_ub = T.alloc_ub((nchunk, M), "float32")
                    idx_ub = T.alloc_ub((nchunk, M), "float32")
                    g = T.alloc_ub((1, M), "float32")
                    g_brc = T.alloc_ub((nchunk, M), "float32")
                    cmp_eq = T.alloc_ub((nchunk, M), "bool")
                    sent_v = T.alloc_ub((nchunk, M), "float32")
                    cand = T.alloc_ub((nchunk, M), "float32")
                    first = T.alloc_ub((1, M), "float32")
                    first_flat = T.alloc_ub((M,), "float32")
                    out_ub = T.alloc_ub((M,), "int64")

                    # r7-d: task-invariant chain operands built ONCE per
                    # lane (manual LICM -- the expert pipeline has none:
                    # the old fused-ite form lowered to varange+vcast+
                    # vbrc(sentinel) INSIDE the row loop, 3 extra
                    # 2560-wide passes per row; chain+hoist measured
                    # 17.13 -> 5.95us on the isolated partial).
                    T.arange(p_idx_src, [1], 0)
                    T.reshape(p_idx_src, p_idx_j)
                    T.vbrc(T.float32(BIG), p_sent_v)

                    # r7-b: flatten (block, sub-block) -> V-core lane.
                    # subid is get_sub_block_idx (the hardware V-core
                    # id in {0,1} of this AI core), so vid enumerates
                    # all 2*num_blocks V cores exactly once.
                    vid = cid * 2 + subid
                    for s in T.serial(num_local_tasks):
                        c = vid + s * num_vcores
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.vcast(x_r, w_r, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                T.vcmp(w_r, e_r, p_cmp, "eq")
                                T.vselect(p_cmp, p_idx_j, p_sent_v, p_cand)
                                T.reduce_min(p_cand, f_r, dim=1)
                                ext_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            with T.rs("PIPE_MTE3"):
                                T.copy(ext_out[0:M], ws_val[c, 0:M])
                                T.copy(gidx_out[0:M], ws_idx[c, 0:M])

                    # Inter-block barrier: all lanes of all blocks
                    # rendezvous (single-wave grid <= aicore_num).
                    with T.rs("PIPE_MTE3"):
                        T.block_barrier(0)

                    # r7-c: merge on lane 0, dim-0 reduces, no
                    # transpose.
                    if vid == 0:
                        T.vbrc(T.float32(BIG), sent_v)
                        T.copy(ws_val, val_ub)
                        T.copy(ws_idx, idx_ub)
                        if op_kind == "argmax":
                            T.reduce_max(val_ub, g, dim=0)
                        else:
                            T.reduce_min(val_ub, g, dim=0)
                        T.vbrc(g, g_brc)
                        T.vcmp(val_ub, g_brc, cmp_eq, "eq")
                        T.vselect(cmp_eq, idx_ub, sent_v, cand)
                        T.reduce_min(cand, first, dim=0)
                        T.reshape(first, first_flat)
                        T.vcast(first_flat, out_ub, round_mode="rint")
                        T.copy(out_ub[0:M], out[0:M])

        else:

            @T.prim_func
            def argreduce_nsplit_fused(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk, M), dtype),
                ws_idx: T.Tensor((nchunk, M), "float32"),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_blocks, is_npu=True) as (cid, subid), T.Scope("Vector"):
                    x_r = T.alloc_ub((1, tn), dtype)
                    w_r = T.alloc_ub((1, tn), dtype)
                    m_r = T.alloc_ub((1, 1), dtype)
                    e_r = T.alloc_ub((1, tn), dtype)
                    f_r = T.alloc_ub((1, 1), "float32")
                    ext_out = T.alloc_ub((M,), dtype)
                    gidx_out = T.alloc_ub((M,), "float32")
                    # r7-d chain-form operands (see note below).
                    p_idx_src = T.alloc_ub((tn,), "float32")
                    p_idx_j = T.alloc_ub((1, tn), "float32")
                    p_sent_v = T.alloc_ub((1, tn), "float32")
                    p_cmp = T.alloc_ub((1, tn), "bool")
                    p_cand = T.alloc_ub((1, tn), "float32")

                    val_ub = T.alloc_ub((nchunk, M), dtype)
                    idx_ub = T.alloc_ub((nchunk, M), "float32")
                    g = T.alloc_ub((1, M), dtype)
                    g_brc = T.alloc_ub((nchunk, M), dtype)
                    cmp_eq = T.alloc_ub((nchunk, M), "bool")
                    sent_v = T.alloc_ub((nchunk, M), "float32")
                    cand = T.alloc_ub((nchunk, M), "float32")
                    first = T.alloc_ub((1, M), "float32")
                    first_flat = T.alloc_ub((M,), "float32")
                    out_ub = T.alloc_ub((M,), "int64")

                    # r7-d: the fused-ite candidate loop is replaced by the
                    # explicit vcmp+vselect chain with manually hoisted
                    # task-invariant operands (idx matrix + BIG sentinel,
                    # built once per lane). Rationale: the expert pipeline
                    # (--disable-hivm-tensor-compile, mandatory for the mix
                    # pre-split form) runs no LICM -- NpuLoopVectorize
                    # lowers the fused-ite to varange+vcast+vbrc(sentinel)
                    # INSIDE the row loop, 3 extra tn-wide vector passes per
                    # row on the serialized chain. Isolated-partial A/B
                    # (2026-10-13, /home/tilelang/l00970450/upload/data/1013):
                    # fused-ite 17.126us vs chain+hoist 5.952us (2.88x);
                    # per-lane scalar pipe 12-13us -> 1.3-1.9us. The
                    # r6 1-D arange + reshape form is used for the idx
                    # matrix (C-9/C-10 traps); same-shape vbrc avoided.
                    T.arange(p_idx_src, [1], 0)
                    T.reshape(p_idx_src, p_idx_j)
                    T.vbrc(T.float32(BIG), p_sent_v)

                    vid = cid * 2 + subid
                    for s in T.serial(num_local_tasks):
                        c = vid + s * num_vcores
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.copy(x_r, w_r)
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                T.vcmp(w_r, e_r, p_cmp, "eq")
                                T.vselect(p_cmp, p_idx_j, p_sent_v, p_cand)
                                T.reduce_min(p_cand, f_r, dim=1)
                                ext_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            with T.rs("PIPE_MTE3"):
                                T.copy(ext_out[0:M], ws_val[c, 0:M])
                                T.copy(gidx_out[0:M], ws_idx[c, 0:M])

                    with T.rs("PIPE_MTE3"):
                        T.block_barrier(0)

                    if vid == 0:
                        T.vbrc(T.float32(BIG), sent_v)
                        T.copy(ws_val, val_ub)
                        T.copy(ws_idx, idx_ub)
                        if op_kind == "argmax":
                            T.reduce_max(val_ub, g, dim=0)
                        else:
                            T.reduce_min(val_ub, g, dim=0)
                        T.vbrc(g, g_brc)
                        T.vcmp(val_ub, g_brc, cmp_eq, "eq")
                        T.vselect(cmp_eq, idx_ub, sent_v, cand)
                        T.reduce_min(cand, first, dim=0)
                        T.reshape(first, first_flat)
                        T.vcast(first_flat, out_ub, round_mode="rint")
                        T.copy(out_ub[0:M], out[0:M])

        return argreduce_nsplit_fused

    def _fused(block_m):
        # The fused kernel requires the expert pipeline (T.Scope("Vector")
        # + T.alloc_ub pure-vector form, mix-promoted so the inter-block
        # barrier completes). The env is read per compile call
        # (engine/lower.py device_codegen + jit_npu bisheng flags), so
        # switching it around this jit call compiles ONLY this kernel in
        # expert mode; every other path of this file stays Developer.
        _prev = os.environ.get("TILELANG_ASCEND_MODE")
        os.environ["TILELANG_ASCEND_MODE"] = "expert"
        try:
            return _fused_jit(block_m)
        finally:
            if _prev is None:
                os.environ.pop("TILELANG_ASCEND_MODE", None)
            else:
                os.environ["TILELANG_ASCEND_MODE"] = _prev

    return _fused


def build_nsplit(M, N, op_kind, dtype, tn=None, num_blocks=None):
    """C6 fused entry for run_bench --mode nsplit (R7-e: tunable).

    Returns ``f(x, ws_val, ws_idx) -> out`` with one device launch; the
    runner allocates and reuses the ``(nchunk, M)`` x2 workspace.
    ``tn``/``num_blocks`` override the calibrated defaults (see
    ``_nsplit_config`` / ``_build_nsplit_fused`` docstrings; A/B data in
    data/1014 report sections 4.1-4.3).
    """
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    if tn is None:
        cfg = _nsplit_config(M, N, dtype, vector_cores)
        if cfg is None:
            raise ValueError(f"shape ({M},{N}) {dtype} not eligible for nsplit")
        tn = cfg["tn"]
    if N % tn != 0:
        raise ValueError(f"tn={tn} must divide N={N} (zero-tail chunking)")
    nchunk = N // tn
    dispatch = _build_nsplit_split(
        M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk, num_blocks
    )

    def launch(x, ws_val, ws_idx):
        dispatch.partial(x, ws_val, ws_idx)
        return dispatch.merge(ws_val, ws_idx)

    return launch


def run_nsplit_L0(tn=None, num_blocks=None):
    """C6 precision gate: constructed tie/edge rows across chunk boundaries.

    Covers the round8-op9 failure class: cross-chunk first-occurrence
    tie-break (duplicate extremes in different chunks, adjacent-chunk tie,
    extreme only in a late chunk), all-equal rows, all-(+-inf) rows, M=1,
    and randn sweeps per dtype/kind. R7-e: ``tn``/``num_blocks`` overrides
    (``--tn``/``--blocks`` CLI) let the framework precision-gate any A/B
    config, not just the default rule.
    """
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    n_cases = 0
    for dtype_str in ("float16", "bfloat16"):
        dt = _DTYPE_MAP[dtype_str]
        ws_dt = torch.float32 if dtype_str == "bfloat16" else dt
        for op_kind in ("argmax", "argmin"):
            cfg = _nsplit_config(4, 102400, dtype_str, vector_cores, tn)
            assert cfg is not None, f"nsplit config missing for {dtype_str}"
            tn_eff, nchunk = cfg["tn"], cfg["nchunk"]
            launch = build_nsplit(4, 102400, op_kind, dtype_str, tn_eff, num_blocks)
            ws_val = torch.empty(nchunk, 4, dtype=ws_dt, device="npu")
            ws_idx = torch.empty(nchunk, 4, dtype=torch.float32, device="npu")

            v = 7.0 if op_kind == "argmax" else -7.0
            v2 = 14.0 if op_kind == "argmax" else -14.0
            x = torch.zeros(4, 102400, dtype=dt, device="npu")
            x[0, 5] = v
            x[0, 102400 - tn_eff] = v  # duplicate extreme, late chunk -> 5
            x[1, 102400 - tn_eff] = v2  # extreme only in late chunk
            x[3, tn_eff - 1] = v  # adjacent-chunk tie -> tn-1
            x[3, tn_eff] = v
            y = launch(x, ws_val, ws_idx)
            ref = golden_argreduce(x.cpu(), op_kind)
            assert y.dtype == torch.int64 and torch.equal(y.cpu(), ref), (
                f"[nsplit-L0] tie mismatch {dtype_str}/{op_kind}: "
                f"got={y.cpu().tolist()} want={ref.tolist()}"
            )
            n_cases += 1

            # all-(extreme-unit) rows: all -inf for argmax / all +inf for argmin
            fill = float("-inf") if op_kind == "argmax" else float("inf")
            x2 = torch.full((2, 102400), fill, dtype=dt, device="npu")
            ws_val2 = torch.empty(nchunk, 2, dtype=ws_dt, device="npu")
            ws_idx2 = torch.empty(nchunk, 2, dtype=torch.float32, device="npu")
            y2 = launch(x2, ws_val2, ws_idx2) if False else None
            launch1 = build_nsplit(2, 102400, op_kind, dtype_str, tn_eff, num_blocks)
            y2 = launch1(x2, ws_val2, ws_idx2)
            ref2 = golden_argreduce(x2.cpu(), op_kind)
            assert torch.equal(y2.cpu(), ref2), (
                f"[nsplit-L0] all-{fill} mismatch {dtype_str}/{op_kind}"
            )
            n_cases += 1

            # M=1 with a cross-chunk tie. NOTE: the original test used
            # ``40 * tn`` which equals N (out of bounds) under the r3e tn
            # rule (fp16 tn=2560, nchunk=40); use an in-bounds last-chunk
            # index instead (fixed here only; the perf_opt original still
            # carries the rot).
            x3 = torch.zeros(1, 102400, dtype=dt, device="npu")
            x3[0, 3 * tn_eff + 7] = v
            x3[0, (nchunk - 1) * tn_eff + 7] = v
            ws_val3 = torch.empty(nchunk, 1, dtype=ws_dt, device="npu")
            ws_idx3 = torch.empty(nchunk, 1, dtype=torch.float32, device="npu")
            launch2 = build_nsplit(1, 102400, op_kind, dtype_str, tn_eff, num_blocks)
            y3 = launch2(x3, ws_val3, ws_idx3)
            ref3 = golden_argreduce(x3.cpu(), op_kind)
            assert torch.equal(y3.cpu(), ref3), (
                f"[nsplit-L0] M=1 tie mismatch {dtype_str}/{op_kind}"
            )
            n_cases += 1

            # randn sweeps (3 seeds).
            for seed in range(3):
                torch.manual_seed(seed)
                xr = torch.randn(4, 102400, dtype=dt, device="npu")
                yr = launch(xr, ws_val, ws_idx)
                rr = golden_argreduce(xr.cpu(), op_kind)
                assert torch.equal(yr.cpu(), rr), (
                    f"[nsplit-L0] randn seed={seed} mismatch {dtype_str}/{op_kind}"
                )
                n_cases += 1
            print(
                f"[nsplit-L0] PASS: {dtype_str}/{op_kind} tn={tn_eff} "
                f"nchunk={nchunk} blocks={num_blocks or 'auto'}"
            )
    print(f"[nsplit-L0] ALL PASS: {n_cases} cases")


# ---------------------------------------------------------------------------
# Stage-4 bench helper: uniform entry for perf_opt/run_bench.py
# ---------------------------------------------------------------------------
def build_case(M, N, op_kind, dtype, block_m=None, tile_n=None, num_blocks=None):
    """Build a launch closure for one workload (Stage 4 bench entry).

    R7-e nsplit tunables (all data in /home/tilelang/l00970450/upload/data/1014):

    - ``tile_n`` doubles as the nsplit chunk-width override (``tn``) for
      nsplit-eligible shapes (the tiled-path meaning is unchanged);
    - ``num_blocks`` overrides the nsplit AI-core block count;
    - with neither override the calibrated default rule applies
      (``_nsplit_config``: largest tn with nchunk in [vector_cores//3,
      vector_cores//2] -- the measured optimum, data/1014 section 4.1).
    """
    cfg = _select_config(M, N, dtype)
    if block_m is None:
        block_m = cfg["block_m"]
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    ncfg = _nsplit_config(M, N, dtype, vector_cores, tn=tile_n)
    if ncfg is not None:
        return _build_nsplit_dispatch(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            vector_cores,
            ncfg["tn"],
            ncfg["nchunk"],
            num_blocks,
        )(block_m)
    if cfg["path"] == "narrow":
        kernel = _build_narrow(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
            cfg["G"],
            cfg["bw"],
        )(cfg["bw"] * cfg["G"])
    elif cfg["path"] == "resident":
        kernel = _build_resident(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
        )(block_m)
    else:
        tn = tile_n if tile_n is not None else cfg["tile_n"]
        kernel = _build_tiled(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
            tn,
        )(block_m)
    return kernel


if __name__ == "__main__":
    main()
