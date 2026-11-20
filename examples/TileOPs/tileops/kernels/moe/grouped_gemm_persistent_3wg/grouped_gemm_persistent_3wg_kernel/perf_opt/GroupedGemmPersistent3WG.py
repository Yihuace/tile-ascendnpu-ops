# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GroupedGemmPersistent3WG -- MoE 3-WG persistent grouped GEMM (NT),
NPUIR Stage 4 final (Stage 3 design + D2 double-buffered prefetch).

This is the Stage 4 `perf_opt` drop-in for the Stage 3 module
``_persistent_grouped_gemm_v2_kernel.py`` (same factory signature, same
schedule contract, unchanged golden); it is shared by three consumer ops:
``FusedMoEExpertsNopadPersistent3WGFwdOp`` / ``FusedMoeFwdOp`` /
``FusedMoeFwdCbFwdOp``.

Stage-4 changes vs the Stage 3 module (three items, all compile-time constant
folds -- no runtime branch, no new kernel, no algorithm-semantics change)
--------------------------------------------------------------------------
R8 strided flat task index (round 8, the dominant item):

  * ``tile_id = t * nums_kernel + core_id`` instead of
    ``tile_id = core_id * num_local_tasks + t``.
  * Why: ``kernel_plan`` sizes the static in-core bound ``num_local_tasks``
    from the *upper bound* ``max_tiles = numel // block_m + num_experts``
    (every expert pays its own per-expert M-tail rounding), but only
    ``total_tiles * num_pid_n`` flat slots are live.  With the core-contiguous
    index every dead slot lands on the tail cores -- measured 3 idle cores on
    qwen3-prefill and 5 on deepseek-prefill (80.7% / 73.1% core occupancy).
    The strided index spreads the same dead tail over all 20 cores, so every
    core gets ``ceil(live / 20)`` live tiles, and 20 consecutive ``tile_id``s
    (one instant across the grid) cover 1.25 ``bx`` values, i.e. the 16
    N-segments of one A block run *simultaneously* on different cores.
  * Measured (kernel calibre, ``msprof op`` ``Task Duration``, same-session
    A/B): qwen-prefill-gate-up 21426 -> 13241 us (-38.2%, Ratio 56.15 ->
    65.58%), qwen-prefill-down 9941 -> 6847 (-31.1%), ds-prefill-gate-up
    26178 -> 18135 (-30.7%), ds-prefill-down 12371 -> 9069 (-26.7%),
    qwen-decode-gate-up 7826 -> 6913 (-11.7%), qwen-decode-down 3946 -> 3323
    (-15.8%), ds-decode-gate-up 14307 -> 13609 (-4.9%), ds-decode-down
    7038 -> 6407 (-9.0%).
  * This also closes the "A re-read" question the v2 reports left open: the
    GM read (L2 read miss) of qwen3-prefill drops 21.39 -> 7.99 GB = exactly
    the incompressible lower bound (weights once + activation once;
    ``lb_gb`` in ``probe_r8_discriminate.py``).  The MTE2 port bytes are
    bit-identical (26.11 GB both ways) -- the whole gain is (a) core
    occupancy and (b) the port idle that goes away with it.  Byte-level
    accounting is in opt_log sec 8.1.

D2 手工 2 深双缓冲预取 (round 7):

  * Two L1 buffer pairs ``a0/b0`` and ``a1/b1``; the K loop is walked in
    pairs (``k_iters // 2`` iterations, ``k_iters = K // block_k``): both
    copy segments are issued first, then the two ``T.gemm`` calls consume
    them in order, so ``copy(chunk k+1)`` overlaps ``gemm(chunk k)``.
  * Measured (kernel calibre, ``msprof op`` ``Task Duration``) on 4 shapes:
    ds-decode 18262 -> 14298 us (-21.7%, Ratio 46.98 -> 60.04%),
    qwen-prefill 30210 -> 24361 (-19.4%), ds-decode-down 8909 -> 7040
    (-21.0%), kimi-medium 12845 -> 9703 (-24.4%).  GM traffic is bit-identical
    (15.44 GB both ways): the gain is pure pipeline overlap, and MTE2 busy
    fraction rises 0.79 -> 0.95.
  * Compile-time guards (both are hard requirements, see opt_log sec 6.3.3):
      - ``K % (2 * block_k) == 0`` -- the paired loop has no odd tail.
      - ``N % block_n == 0`` -- combining a larger ``block_n`` with an N tail
        makes ``bishengir-compile`` SIGSEGV deterministically (10/10 on the
        probe), so the pair body is only built for aligned N.
    Failing either guard falls through to the Stage-3 ``T.serial`` body
    below, which keeps the full dynamic-extent K/N tail handling.  The
    dispatch is a plain Python ``if`` on compile-time integers in the factory
    (both segments define ``main``; exactly one is built), because
    ``if <python int>:`` inside a ``@T.prim_func`` body is parsed as a runtime
    boolean and rejected.  Nothing is dispatched at the op / wrapper level.
  * ``_PASS_CFG`` is intentionally unchanged: the gain was measured with
    ``NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`` (same config as the Stage 3
    module), so no buffer-planner switch is touched.

``block_n`` shape selection (secondary, independently revertable):

  * ``block_n`` 128 -> 256 when the routing is heavy enough that the A
    re-read dominates (mean rows per expert ``numel // num_experts >= 128``),
    and only when ``N % 256 == 0``.  Measured: qwen-prefill -6.7%,
    kimi-medium -0.4%, ds-decode +2.4%, ds-decode-down +1.4% (so the bump is
    gated on the shape family, not applied globally).  Combined with D2:
    qwen-prefill -29.1% (Ratio 40.58 -> 56.03%).
  * L1 budget: D2 + bn=128 -> 2*(64+64) KB = 256 KB; D2 + bn=256 ->
    2*(64+128) KB = 384 KB.  Both <= 512 KB.

Harness confirmation (the three consumer ops, `msprof op` same calibre as the
v1 report; run dirs in opt_log sec 7): qwen3-decode 10221 -> 7849 us,
qwen3-prefill 30199 -> 21428, deepseek-decode 18199 -> 14332, deepseek-prefill
38133 -> 26215.  No shape regressed.  deepseek-decode lands on the 60% line
(Ratio 59.84% in the authoritative run, 60.32% in a second run of the same
candidate), so treat it as borderline rather than a robust pass; the rest are
54-56%.

Compile-side note (CG-2026-0033): the flaky `bishengir-compile` SIGSEGV that
the Stage-3 module already documents gets *more frequent* as the in-core serial
bound grows.  Measured on the ten real GEMM shapes at the shipped policy, a
clean build needs 1-3 fresh compiles, but a build that sits behind many other
compiles in one long-lived process can burn a small retry budget (observed
twice: `L2-3` in `--level all`, and `qwen3-prefill` inside a full harness
session).  The default `PGG_COMPILE_ATTEMPTS` budget is therefore 30, which
covers those runs and matches the sibling `MOE_COMPILE_ATTEMPTS` default; set
the environment variable to override it.  The crash is compile-time only,
never a wrong number.

Design: examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/DESIGN.md
        (frozen, REVIEW.md verdict = pass)
Migration source: Hopper warp-specialized persistent grouped GEMM
        (`grouped_gemm_persistent_3wg.py::_persistent_grouped_gemm_v2_kernel`).

Semantics (source-op contract, unchanged by the migration)
----------------------------------------------------------
For every expert e in [0, E) with s_e = true_sizes[e] tight rows located at
true_offsets[e] in A, and B[e] of shape (N, K):

    C[o_e + r, :] = A[o_e + r, :] @ B[e].T          r in [0, s_e)

i.e. a per-expert NT matmul with fp32 accumulation and one cast back to the
input dtype, over a tight (no-pad) row layout.  C has shape (numel, N).

Structure actually implemented (DESIGN sec 0.6 R1..R7 / sec 3.3)
---------------------------------------------------------------
Expert, single ``T.Scope("Cube")``.  The Hopper design's kernel-internal
static-wave scheduler is NOT reproducible inside an NPU Cube scope (CG-2026-0010
lowering segfault / CG-2026-0027 unsynchronised scalar-of-vector consume /
CG-2026-0026 uncompilable cross-engine in-block handoff), so the schedule is
materialised in GM by a separate producer unit and consumed here as three int32
tables:

    tile_expert_ids[max_tiles]    expert id per M-tile (dead tile = -1)
    tile_row_offsets[max_tiles]   in-expert row offset (already * block_m)
    total_tiles[1]                number of live M-tiles (stays on device)

This kernel therefore only does:

    read GM tables/scalars -> dynamic-extent T.copy (GM->L1, implicit ND2NZ) ->
    T.gemm(size=[tile_m, tile_k, tile_n], b_transpose=True, initC=(k == 0)) ->
    single-path T.store_fixpipe([tile_m, tile_n], enable_nz2nd=True)

Deliberately NOT used (each one is a measured toolchain blocker):
  * ``T.sync_block_set/wait`` (any Vector<->Cube in-block handoff -> the
    ``syn_instr_mode`` attribute mismatch, CG-2026-0026);
  * the Developer programming mode (persistent + T.gemm + integer indexing
    segfaults at lowering, CG-2026-0010);
  * any scalar read of a vector-produced value (CG-2026-0027).

Launch geometry: persistent grid = AI-core count (queried at import time,
DESIGN sec 0.5 O13 / sec 5.5), with an in-core static-bound ``T.serial``
multi-tile loop; the dead-tile guard ``if bx < total_tiles[0]`` sits inside the
serial loop and OUTSIDE the K loop (CG-2026-0001).

Dynamic extents (DESIGN sec 0.6 R7 / sec 5.4) clip M, N and K exactly
(``tile_m/tile_n/tile_k``), which is both the no-pad correctness requirement
(never write into the next expert's rows) and the sole K-tail mechanism --
``tile_k`` MUST be passed into ``T.gemm(size=...)``: without it the operand
buffers' stale ``[tile_k, block_k)`` tail would be reduced over a static
``block_k`` and silently corrupt the accumulation (DESIGN sec 1.6.1 M1).

Golden is the source-op semantics, written with explicit ``torch`` module ops
(per-expert NT matmul, fp32 accumulate then cast) and is independent of this
NPU algorithm (DESIGN sec 8.1).

Toolchain note (CG-2026-0033): the bundled ``bishengir-compile`` intermittently
SIGSEGVs on the IR form that combines the in-core serial task loop with a
dynamic column offset (``n_start = by * block_n``, i.e. ``num_pid_n > 1``
together with ``num_local_tasks > 1``).  The bit-identical IR compiles on retry,
so the harness raises ``RLIMIT_STACK`` and retries a bounded number of fresh
compiles.  Accuracy is unaffected: the crash is compile-time, never a wrong
number.  Sibling repro:
``examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-BISHENGIR-FLAKY-SIGSEGV.py``.

Run:
    TILELANG_ASCEND_MODE=Expert ASCEND_RT_VISIBLE_DEVICES=1 \
        python GroupedGemmPersistent3WG.py --level all
"""

import argparse
import importlib.util
import os
import resource
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

# Programming mode actually implemented and validated by this file: Expert
# (single T.Scope("Cube")).  Must match TILELANG_ASCEND_MODE when the kernel is
# exercised standalone.
ASCEND_MODE = "Expert"

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _query_aicore_num(default=20):
    """Query the physical AI-core (Cube) count -- DESIGN sec 5.5.

    A Cube operator uses the queried value directly (no x2; the x2 rule applies
    to pure-Vector kernels only).  The default is only a fallback for
    environments where the utility is unavailable.
    """
    try:
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num())
    except Exception:  # noqa: BLE001 - fall back to the measured constant
        return default


#: Physical Cube core count of this box (measured; DESIGN sec 5.5).
PHYS_CORES = _query_aicore_num()

# --------------------------------------------------------------------------
# Stage-4 tuned defaults / compile-time shape policy
#
# Exposed as module constants so the wrapper's baseline/perf_opt switch block
# can reference the same numbers instead of re-hardcoding them.
# --------------------------------------------------------------------------
TUNED_DEFAULT_CONFIG = {
    "block_m": 128,
    "block_n": 128,
    "block_k": 256,
    "num_stages": 3,
}

# The L0/L1/L2/boundary suites below build through ``run_case``/``build_schedule``
# and must exercise the SHIPPED tiling, not the Stage-3 one: the harness-only
# defaults now track ``TUNED_DEFAULT_CONFIG`` (the Stage-3 module left them at
# 64/256/64, so its suites never covered the tuned geometry).
_BLOCK_M = TUNED_DEFAULT_CONFIG["block_m"]
_BLOCK_N = TUNED_DEFAULT_CONFIG["block_n"]
_BLOCK_K = TUNED_DEFAULT_CONFIG["block_k"]

#: ``block_n`` shape policy: bump 128 -> 256 only when the A re-read dominates.
#: A/B read ratio = numel / (num_experts * block_n), so ``mean_rows`` (=
#: numel // num_experts) >= this threshold means the bump pays back.
BN_BUMP_MIN_MEAN_ROWS = 128
BN_BUMP_TARGET = 256


def _select_block_n(numel, num_experts, n, block_n):
    """Resolve ``block_n`` for this shape (compile-time constant).

    Measured (same-run A/B/A/B, opt_log sec 6.5): ``block_n`` 128 -> 256 gives
    -6.7% on the A-reread-dominated prefill shape (A/B = 2.0), -0.4% at
    A/B = 0.5, and +1.4% / +2.4% on the decode shapes (A/B = 0.125) where
    there is no A traffic left to save and only a bigger L1/L0C working set.
    Hence the bump is conditional on the routing density, never global.

    The guard ``n % BN_BUMP_TARGET == 0`` is mandatory (see the module
    docstring: a ``block_n`` tail combined with the paired body makes
    ``bishengir-compile`` SIGSEGV deterministically).
    """
    if block_n >= BN_BUMP_TARGET:
        return block_n
    if num_experts <= 0:
        return block_n
    if numel // num_experts < BN_BUMP_MIN_MEAN_ROWS:
        return block_n
    if n % BN_BUMP_TARGET != 0:
        return block_n
    return BN_BUMP_TARGET


def _use_prefetch_pair(K, N, block_k, block_n):
    """Compile-time gate for the D2 paired (double-buffered) K loop.

    Both conditions are hard: ``K % (2 * block_k) == 0`` because the paired
    loop has no odd tail, and ``N % block_n == 0`` because the paired body
    combined with an N tail is the deterministic bishengir SIGSEGV form.
    Anything else keeps the Stage-3 ``T.serial`` body (which handles both
    tails through dynamic extents).
    """
    return (block_k > 0 and K % (2 * block_k) == 0
            and block_n > 0 and N % block_n == 0)


def _ceildiv(x, y):
    return (x + y - 1) // y


def _is_a5():
    """True when the compile target is the Ascend 950 (A5) family."""
    name = os.environ.get("TILELANG_ASCEND_DEVICE_NAME", "").strip()
    if not name:
        try:
            from tilelang.utils.npu_arch import get_ascend_device_name

            name = str(get_ascend_device_name() or "")
        except Exception:  # noqa: BLE001
            name = ""
    return name.startswith("Ascend950")


def _pick_block_k(k_dim, block_k):
    """Choose a K block that divides ``k_dim`` (A5 static-size gemm requirement).

    On A5/CANN 9.2 a dynamic-extent K copy combined with a static-size
    ``T.gemm`` is numerically wrong (the tail columns keep the previous
    iteration's data and get folded into the reduction), so the reduction
    extent must be exact: ``block_k`` has to divide K.  Prefer the largest
    multiple-of-16 divisor <= ``block_k`` (16 = fp16/bf16 fractal height).
    Returns None when no such divisor exists -- the caller must not fall back
    to the static path, because a non-dividing ``block_k`` would read past
    the end of A/B.
    """
    if k_dim % block_k == 0:
        return block_k
    start = min(block_k, k_dim) // 16 * 16
    for cand in range(start, 0, -16):
        if k_dim % cand == 0:
            return cand
    return None


def _pick_block_n(n_dim, block_n):
    """Choose an N block that avoids a 16-aligned *static partial* L1 copy.

    On A5/CANN 9.2 a B copy whose compile-time extent is a multiple of 16 but
    smaller than the L1 buffer aborts bishengir with
    ``[hivm-bind-sub-block] revert main_mix_aiv`` + ``LLVM ERROR: Unknown
    PIPE!``.  Dynamic partial copies (N > block_n) and full copies are fine.
    Rounding a sub-block N up leaves stale B columns past N, but those only
    feed discarded output columns (the fixpipe writes [t_m, t_n]), so the
    visible result is unaffected.
    """
    if n_dim % block_n == 0:
        return block_n
    if n_dim < block_n:
        return n_dim if n_dim % 16 == 0 else _ceildiv(n_dim, 16) * 16
    return block_n


def max_tiles_of(numel, num_experts, block_m=_BLOCK_M):
    """Compile-time M-tile upper bound (DESIGN sec 4.1 contract, shared)."""
    return numel // block_m + num_experts


def kernel_plan(max_tiles, n, block_n=_BLOCK_N):
    """Compile-time launch geometry: (nums_kernel, num_local_tasks).

    nums_kernel = min(logical tiles, PHYS_CORES) -- persistent grid = AI-core
    count; num_local_tasks is the static in-core serial bound (DESIGN sec 5.5).
    """
    num_pid_n = _ceildiv(n, block_n)
    num_logical = max_tiles * num_pid_n
    nums_kernel = min(num_logical, PHYS_CORES)
    num_local_tasks = _ceildiv(num_logical, nums_kernel)
    return nums_kernel, num_local_tasks


def build_schedule(sizes_l, block_m=_BLOCK_M):
    """Host-built schedule fixture == producer-kernel output (DESIGN sec 4.1).

    Contract: experts clustered in order, each contributing
    ceil(s_e / block_m) consecutive M-tiles; unused slots carry the (-1, 0)
    sentinel; ``tile_row_offsets`` is in ROWS (already multiplied by block_m).
    Returns (tile_expert_ids, tile_row_offsets, total_tiles) as python lists.
    """
    ids, rows = [], []
    for e, s in enumerate(sizes_l):
        for j in range(_ceildiv(s, block_m)):
            ids.append(e)
            rows.append(j * block_m)
    total = len(ids)
    max_tiles = max_tiles_of(sum(sizes_l), len(sizes_l), block_m)
    assert total <= max_tiles, f"schedule overflow: {total} > {max_tiles}"
    while len(ids) < max_tiles:
        ids.append(-1)
        rows.append(0)
    return ids, rows, total


# Expert form: the explicit L1/L0C allocations must not be re-scoped/reordered
# by the buffer planners (DESIGN sec 7.3 / PL-1.16).
_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}


# --------------------------------------------------------------------------
# Kernel (DESIGN sec 4.7 factory; sec 3.3 Expert single-Cube body)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir", pass_configs=_PASS_CFG)
def _persistent_grouped_gemm_v2_kernel(numel, max_tiles, num_experts, N, K,
                                       dtype="float16", block_m=128,
                                       block_n=128, block_k=256,
                                       num_stages=3):
    """Factory: binds the compile-time geometry (num_stages is reserved for a
    Stage-4 ``T.Pipelined`` K loop; the validated body uses ``T.serial``).

    STAGE-4 TUNED DEFAULTS: ``block_m=128 / block_n=128 / block_k=256``
    (baseline was 64/256/64 formals, effectively 128/256/64 through the
    wrapper config).  Measured on the qwen3-235b shapes:
      gate_up (numel=4096,E=128,N=4096,K=7168): 14037.7 -> 10201.3 us (-27.3%)
      down    (numel=4096,E=128,N=7168,K=2048):  6911.8 ->  4969.7 us (-28.1%)
    ``T.Pipelined`` on the K loop was also tried (see
    ``_persistent_grouped_gemm_v2_kernel_r1_pipelined.py``): latency-neutral
    (10200.0 vs 10201.3 us) -- but that experiment was void (the auto
    multi-buffer pass was disabled, so no second buffer was ever materialised;
    pipe occupancy was bit-identical).  The real overlap gain comes from the
    manual D2 pair body below.

    STAGE-4 (round 7) -- ``block_n`` is re-resolved from the shape
    (:func:`_select_block_n`) and the K loop body is chosen at compile time
    (:func:`_use_prefetch_pair`): the D2 paired / double-buffered body when
    ``K % (2*block_k) == 0`` and ``N % block_n == 0``, else the Stage-3
    ``T.serial`` body.  Both segments define ``main`` and only one is built.
    """
    accum_dtype = "float32"
    a5 = _is_a5()
    if a5:
        # A5/CANN 9.2 constraints (measured on Ascend950PR; see the sibling
        # moe_grouped_gemm_nopad kernel for the isolation experiments):
        #  - mmadL1 size must be a compile-time constant, and the K reduction
        #    extent must exactly equal block_k -> block_k must divide K;
        #  - a 16-aligned static partial B copy crashes bishengir;
        #  - the in-core serial task loop (scf.for carrying buffer iter_args)
        #    fails verification, so one logical tile per block.
        picked_k = _pick_block_k(K, block_k)
        if picked_k is None:
            raise ValueError(
                f"A5/CANN 9.2 requires block_k to divide K (16-aligned divisor "
                f"<= {block_k}); K={K} has no such divisor"
            )
        block_k = picked_k
        block_n = _pick_block_n(N, block_n)
    else:
        # Compile-time shape policy (constant fold): a larger block_n pays back
        # only on A-reread-dominated shapes, and only with an aligned N.
        block_n = _select_block_n(numel, num_experts, N, block_n)
    num_pid_n = _ceildiv(N, block_n)
    num_logical = max_tiles * num_pid_n
    if a5:
        nums_kernel = num_logical
        num_local_tasks = 1
        if nums_kernel > 65535:
            raise ValueError(
                f"A5/CANN 9.2 runs one logical tile per block (in-core serial "
                f"task loops fail); grid would be {nums_kernel} > 65535 "
                f"(max_tiles={max_tiles}, num_pid_n={num_pid_n}). Use a larger "
                f"block_m/block_n or the tile-scheduler kernel."
            )
    else:
        nums_kernel = min(num_logical, PHYS_CORES)            # persistent grid = AI cores
        num_local_tasks = _ceildiv(num_logical, nums_kernel)  # static bound

    if _use_prefetch_pair(K, N, block_k, block_n):
        k_iters = K // block_k        # paired loop bound (even, guards hold)

        @T.prim_func
        def main(
            A: T.Tensor((numel, K), dtype),
            B: T.Tensor((num_experts, N, K), dtype),
            C: T.Tensor((numel, N), dtype),
            tile_expert_ids: T.Tensor((max_tiles,), "int32"),
            tile_row_offsets: T.Tensor((max_tiles,), "int32"),
            true_offsets: T.Tensor((num_experts,), "int32"),
            true_sizes: T.Tensor((num_experts,), "int32"),
            total_tiles: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(nums_kernel, is_npu=True) as (core_id, _subid):
                with T.Scope("Cube"):
                    # D2：两份 a/b L1 缓冲（ping-pong）。同一任务内先发两段 copy，
                    # 再依次 gemm —— copy(chunk k+1) 与 gemm(chunk k) 在时间上重叠，
                    # 把单缓冲下 MTE2 的 ~20% 空转吃掉（实测 mte2 .79 -> .95）。
                    a0 = T.alloc_L1((block_m, block_k), dtype)
                    b0 = T.alloc_L1((block_n, block_k), dtype)
                    a1 = T.alloc_L1((block_m, block_k), dtype)
                    b1 = T.alloc_L1((block_n, block_k), dtype)
                    c_l0c = T.alloc_L0C((block_m, block_n), accum_dtype)
                    for t in T.serial(num_local_tasks):
                        # R8：strided flat 任务序。`num_local_tasks` 由**上界**
                        # max_tiles*num_pid_n 定出，而实际 live tile 只有
                        # total_tiles*num_pid_n；core-contiguous 索引把这些
                        # 死槽全堆到尾部核上（qwen3-prefill 3 核空转、
                        # deepseek-prefill 5 核空转），strided 索引让 20 核
                        # 均匀分摊死槽 => 满核 + 跨核同时段共享同一 A 块。
                        tile_id = t * nums_kernel + core_id
                        bx = tile_id // num_pid_n
                        by = tile_id % num_pid_n
                        # 死 tile 守卫在 K 循环之外（CG-2026-0001）。
                        if bx < total_tiles[0]:
                            expert_id = tile_expert_ids[bx]
                            row = tile_row_offsets[bx]
                            n_start = by * block_n
                            m_start = true_offsets[expert_id] + row
                            tile_m = T.min(block_m, true_sizes[expert_id] - row)
                            tile_n = T.min(block_n, N - n_start)
                            # 成对走 K：K % (2*block_k) == 0 由编译期守卫保证，
                            # 故每个成对 chunk 的 tile_k 恒为 block_k（无 K 尾块），
                            # size= 里可直接用静态 block_k。
                            for k2 in T.serial(k_iters // 2):
                                k0 = 2 * k2 * block_k
                                k1 = (2 * k2 + 1) * block_k
                                T.copy(
                                    A[m_start:m_start + tile_m,
                                      k0:k0 + block_k],
                                    a0[0:tile_m, 0:block_k],
                                )
                                T.copy(
                                    B[expert_id, n_start:n_start + tile_n,
                                      k0:k0 + block_k],
                                    b0[0:tile_n, 0:block_k],
                                )
                                T.copy(
                                    A[m_start:m_start + tile_m,
                                      k1:k1 + block_k],
                                    a1[0:tile_m, 0:block_k],
                                )
                                T.copy(
                                    B[expert_id, n_start:n_start + tile_n,
                                      k1:k1 + block_k],
                                    b1[0:tile_n, 0:block_k],
                                )
                                # 首块 initC=True 覆盖初始化 L0C，其后一律累加。
                                # A5: static mmadL1 size; stale M rows / N
                                # columns are discarded by the [tile_m, tile_n]
                                # fixpipe write.  Non-A5 keeps dynamic sizes.
                                gemm_size = ([block_m, block_k, block_n] if a5
                                             else [tile_m, block_k, tile_n])
                                T.gemm(a0, b0, c_l0c,
                                       size=gemm_size,
                                       b_transpose=True, initC=(k2 == 0))
                                T.gemm(a1, b1, c_l0c,
                                       size=gemm_size,
                                       b_transpose=True, initC=False)
                            with T.rs("PIPE_FIX"):
                                T.store_fixpipe(c_l0c, C[m_start, n_start],
                                                size=[tile_m, tile_n],
                                                enable_nz2nd=True)
        return main

    @T.prim_func
    def main(
        A: T.Tensor((numel, K), dtype),
        B: T.Tensor((num_experts, N, K), dtype),
        C: T.Tensor((numel, N), dtype),
        tile_expert_ids: T.Tensor((max_tiles,), "int32"),
        tile_row_offsets: T.Tensor((max_tiles,), "int32"),
        true_offsets: T.Tensor((num_experts,), "int32"),
        true_sizes: T.Tensor((num_experts,), "int32"),
        total_tiles: T.Tensor((1,), "int32"),
    ):
        # Stage-3 body kept verbatim as the fallback: full dynamic-extent K/N
        # tail handling (used whenever the paired-body guards do not hold,
        # e.g. the K=256 smoke shape or any N tail).
        with T.Kernel(nums_kernel, is_npu=True) as (core_id, _subid):
            # 单一 Cube scope：只读 GM 调度表 + NT GEMM + 写回；无 Vector 段、
            # 无跨引擎同步（规避 CG-2026-0026）／无核内调度状态（规避 CG-2026-0027）。
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((block_m, block_k), dtype)
                b_l1 = T.alloc_L1((block_n, block_k), dtype)
                c_l0c = T.alloc_L0C((block_m, block_n), accum_dtype)
                # 固定核数（=AI core 数）+ 核内串行多任务；循环边界全静态（§5.5）。
                for t in T.serial(num_local_tasks):
                    # 专家聚簇 + M-major 任务序（§0.6 R5）：flat task -> (bx, by)。
                    # R8：strided 索引（同 D2 体，见上）。
                    tile_id = t * nums_kernel + core_id
                    bx = tile_id // num_pid_n
                    by = tile_id % num_pid_n
                    # 死 tile 守卫：单比较，位于 T.serial 体内、K 循环之外
                    # （规避 CG-2026-0001：Pipelined 体内条件构造崩溃）。
                    if bx < total_tiles[0]:
                        expert_id = tile_expert_ids[bx]   # GM 标量读（运行时下标）
                        row = tile_row_offsets[bx]        # GM 标量读；单位=行（已乘 block_m）
                        n_start = by * block_n
                        m_start = true_offsets[expert_id] + row
                        # 动态 extent：M/N 尾块精确裁剪（no-pad 关键，§5.4）。
                        tile_m = T.min(block_m, true_sizes[expert_id] - row)
                        tile_n = T.min(block_n, N - n_start)
                        for k in T.serial(T.ceildiv(K, block_k)):
                            # A5: full static K block (block_k divides K);
                            # non-A5 keeps the dynamic tail extent.
                            tile_k = block_k if a5 else T.min(block_k, K - k * block_k)
                            # 动态 extent 取数：GM->L1（隐式 ND2NZ）。
                            T.copy(
                                A[m_start:m_start + tile_m,
                                  k * block_k:k * block_k + tile_k],
                                a_l1[0:tile_m, 0:tile_k],
                            )
                            T.copy(
                                B[expert_id, n_start:n_start + tile_n,
                                  k * block_k:k * block_k + tile_k],
                                b_l1[0:tile_n, 0:tile_k],
                            )
                            # size=[tile_m, tile_k, tile_n] 把归约范围限定到有效
                            # tile_k（正确性必需：否则缓冲 [tile_k, block_k) 的陈旧
                            # 尾值会被按静态 block_k 计入累加，§3.3 注 1 / §1.6.1 M1）。
                            # A5: static mmadL1 size; stale M rows / N columns
                            # are discarded by the [tile_m, tile_n] fixpipe
                            # write.  Non-A5 keeps dynamic sizes.
                            gemm_size = ([block_m, block_k, block_n] if a5
                                         else [tile_m, tile_k, tile_n])
                            # initC=(k==0)：首块覆盖初始化 L0C，其后累加（§1.6.1 M2）。
                            T.gemm(a_l1, b_l1, c_l0c,
                                   size=gemm_size,
                                   b_transpose=True, initC=(k == 0))
                        # 动态 extent 写回：只写有效 [tile_m, tile_n]，杜绝越界写
                        # 坏下一专家行（no-pad 关键正确性约束，§5.4）。
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_l0c, C[m_start, n_start],
                                            size=[tile_m, tile_n],
                                            enable_nz2nd=True)

    return main


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source-op semantics, explicit torch module ops)
# --------------------------------------------------------------------------
def golden_persistent_grouped_gemm(a, b, true_sizes, true_offsets):
    """Per-expert NT matmul over the tight (no-pad) MoE layout.

    a: (numel, K); b: (E, N, K); true_sizes: (E,) int32; true_offsets: (E,) int32
    Returns c: (numel, N), same dtype as ``a`` (fp32 accumulate, cast back).
    """
    numel = a.shape[0]
    num_experts, n, _ = b.shape
    c = torch.zeros(numel, n, dtype=a.dtype, device=a.device)
    sizes = true_sizes.tolist()
    offsets = true_offsets.tolist()
    for e in range(num_experts):
        size_e = sizes[e]
        if size_e == 0:
            continue
        off_e = offsets[e]
        a_e = a[off_e:off_e + size_e].to(torch.float32)
        b_e = b[e].to(torch.float32)
        c[off_e:off_e + size_e] = torch.matmul(
            a_e, torch.transpose(b_e, 0, 1)
        ).to(a.dtype)
    return c


# --------------------------------------------------------------------------
# Reference inputs (mirrors the equivalent op's manifest workload distribution)
# --------------------------------------------------------------------------
def make_sizes_offsets(numel, num_experts, distribution):
    if distribution == "uniform":
        base = max(1, numel // num_experts)
        sizes = [base] * num_experts
        sizes[-1] = numel - base * (num_experts - 1)
    elif distribution == "skewed":
        sizes = [1] * num_experts
        extra = numel - num_experts
        top = max(1, num_experts // 5)
        per = extra // top
        for i in range(top):
            sizes[i] += per
        sizes[0] += extra - per * top
    elif distribution == "empty":
        # Some experts get zero tokens -> no tile at all (dead by construction).
        sizes = [0] * num_experts
        active = [e for e in range(num_experts) if e % 3 != 0]
        if not active:
            active = [num_experts - 1]
        base = numel // len(active)
        rem = numel - base * len(active)
        for i, e in enumerate(active):
            sizes[e] = base + (1 if i < rem else 0)
    elif distribution == "single":
        sizes = [0] * num_experts
        sizes[0] = numel
    else:
        raise ValueError(distribution)
    offsets = [0] * num_experts
    for e in range(1, num_experts):
        offsets[e] = offsets[e - 1] + sizes[e - 1]
    assert sum(sizes) == numel, (sizes, numel)
    return sizes, offsets


# --------------------------------------------------------------------------
# Flaky bishengir-compile mitigation (CG-2026-0033)
# --------------------------------------------------------------------------
def _raise_stack_limit():
    """Raise RLIMIT_STACK to the hard limit.

    bishengir-compile walks nested scf regions (task loop -> dead-tile guard ->
    K loop) with deep recursion; a symbol-less backtrace repeating one PC is a
    stack overflow.  A larger stack lowers the crash rate; the reliable net is
    the bounded fresh-compile retry in ``_run_gemm``.  Children inherit the
    limit across exec.
    """
    try:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ValueError, OSError, AttributeError):
        pass


_raise_stack_limit()


def _is_flaky_compile_error(exc):
    msg = str(exc)
    return ("NPU IR opt failed" in msg or "SIGSEGV" in msg
            or "segmentation" in msg.lower() or "-11" in msg)


# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------
def _run_gemm(build, call_args, attempts=None):
    """Invoke the JIT kernel, retrying flaky bishengir-compile SIGSEGVs.

    Only compile-time crash signatures are retried (fresh factory each time),
    so deterministic shape-dependent errors are never masked.  The retry
    budget is controlled by PGG_COMPILE_ATTEMPTS (default 30, matching the
    sibling MOE_COMPILE_ATTEMPTS default -- the crash rate rises with
    ``num_local_tasks``, and a run was observed needing 5 consecutive retries).
    """
    if attempts is None:
        attempts = int(os.environ.get("PGG_COMPILE_ATTEMPTS", "30"))
    last = None
    for i in range(attempts):
        try:
            return build()(*call_args)
        except RuntimeError as exc:  # noqa: PERF203
            if not _is_flaky_compile_error(exc):
                raise
            last = exc
            print(f"  [warn] fresh-compile attempt {i + 1}/{attempts} hit the "
                  f"flaky bishengir-compile crash, retrying...")
            time.sleep(1.0)
    raise last


def make_inputs(numel, num_experts, n, k, dtype, distribution, seed=42):
    """Reference A/B plus the true_sizes/true_offsets pair (host side)."""
    torch.manual_seed(seed)
    tdt = _DTYPE_MAP[dtype]
    sizes_l, offsets_l = make_sizes_offsets(numel, num_experts, distribution)
    a = torch.randn(numel, k, dtype=tdt) * 0.02
    b = torch.randn(num_experts, n, k, dtype=tdt) * 0.02
    sizes = torch.tensor(sizes_l, dtype=torch.int32)
    offsets = torch.tensor(offsets_l, dtype=torch.int32)
    return a, b, sizes, offsets, sizes_l, offsets_l


def _compare_and_report(c, ref, tag, extra=""):
    cc = c.cpu().float()
    rr = ref.float()
    if cc.numel() == 0:  # degenerate no-work case: shape equality is the check
        assert cc.shape == rr.shape, (cc.shape, rr.shape)
        max_diff = 0.0
    else:
        max_diff = (cc - rr).abs().max().item()
    torch.testing.assert_close(cc, rr, atol=1e-2, rtol=1e-2)
    print(f"[{tag}] PASS  {extra} max_diff={max_diff:.3e}")
    return max_diff


def run_case(numel, num_experts, n, k, dtype, distribution, tag,
             block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K, seed=42):
    """Single-case check: host-built schedule fixture -> GEMM kernel -> golden."""
    a, b, sizes, offsets, sizes_l, offsets_l = make_inputs(
        numel, num_experts, n, k, dtype, distribution, seed)
    max_tiles = max_tiles_of(numel, num_experts, block_m)
    teids_l, troffs_l, total = build_schedule(sizes_l, block_m)
    # Report the geometry the factory will actually build (block_n is re-resolved
    # from the shape inside the factory, so the print must use the same value).
    bn_eff = _select_block_n(numel, num_experts, n, block_n)
    nk, nlt = kernel_plan(max_tiles, n, bn_eff)
    body = ("d2_pair" if _use_prefetch_pair(k, n, block_k, bn_eff)
            else "serial")

    c = _run_gemm(
        lambda: _persistent_grouped_gemm_v2_kernel(
            numel, max_tiles, num_experts, n, k, dtype,
            block_m, block_n, block_k),
        (
            a.npu(), b.npu(),
            torch.tensor(teids_l, dtype=torch.int32).npu(),
            torch.tensor(troffs_l, dtype=torch.int32).npu(),
            offsets.npu(), sizes.npu(),
            torch.tensor([total], dtype=torch.int32).npu(),
        ),
    )
    torch.npu.synchronize()
    ref = golden_persistent_grouped_gemm(a, b, sizes, offsets)
    return _compare_and_report(
        c, ref, tag,
        extra=f"shape=({numel},{num_experts},{n},{k}) dtype={dtype} "
              f"dist={distribution} max_tiles={max_tiles} total={total} "
              f"cores={nk} local_tasks={nlt} "
              f"bn={bn_eff} body={body}")


# --- optional two-kernel integration check (DESIGN sec 8.3 L0-6) -----------
_SIBLING_SCHED = None
_SIBLING_TRIED = False


def _load_sibling_scheduler():
    """Load the sibling producer unit `_tile_scheduler_kernel` (DESIGN sec 4.1).

    The producer kernel is an independent unit with its own DESIGN; it is not
    part of this unit.  When present, the two real kernels are chained in one
    stream to validate the GM-table handoff contract end to end.
    """
    global _SIBLING_SCHED, _SIBLING_TRIED
    if _SIBLING_TRIED:
        return _SIBLING_SCHED
    _SIBLING_TRIED = True
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(root, "examples", "moe_grouped_gemm_nopad",
                        "_tile_scheduler_kernel", "_tile_scheduler_kernel.py")
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_pgg_sibling_sched", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SIBLING_SCHED = mod
    except Exception as exc:  # noqa: BLE001 - integration check is optional
        print(f"[L0-6] WARN sibling scheduler unavailable: {exc}")
        _SIBLING_SCHED = None
    return _SIBLING_SCHED


def run_case_two_kernel(numel, num_experts, n, k, dtype, distribution, tag,
                        block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K):
    """Producer kernel -> GEMM kernel, one stream, no CPU round-trip."""
    mod = _load_sibling_scheduler()
    if mod is None:
        print(f"[{tag}] SKIP  producer unit not available")
        return None
    a, b, sizes, offsets, sizes_l, offsets_l = make_inputs(
        numel, num_experts, n, k, dtype, distribution)
    max_tiles = max_tiles_of(numel, num_experts, block_m)
    nk, nlt = kernel_plan(max_tiles, n, block_n)

    cum, ids, rows, total = mod._run_scheduler(
        num_experts, max_tiles, block_m, sizes)

    # Contract check (DESIGN sec 4.1): table dtype/shape/sentinel/row unit.
    g_ids, g_rows, g_total = build_schedule(sizes_l, block_m)
    assert total.dtype == torch.int32 and tuple(total.shape) == (1,)
    assert int(total.cpu().item()) == g_total
    assert ids.dtype == torch.int32 and tuple(ids.shape) == (max_tiles,)
    assert rows.dtype == torch.int32 and tuple(rows.shape) == (max_tiles,)
    assert bool((ids.cpu() == torch.tensor(g_ids, dtype=torch.int32)).all())

    c = _run_gemm(
        lambda: _persistent_grouped_gemm_v2_kernel(
            numel, max_tiles, num_experts, n, k, dtype,
            block_m, block_n, block_k),
        (a.npu(), b.npu(), ids, rows, offsets.npu(), sizes.npu(), total),
    )
    torch.npu.synchronize()
    ref = golden_persistent_grouped_gemm(a, b, sizes, offsets)
    return _compare_and_report(
        c, ref, tag,
        extra=f"two-kernel shape=({numel},{num_experts},{n},{k}) dtype={dtype} "
              f"total={g_total} cores={nk} local_tasks={nlt}")


# --------------------------------------------------------------------------
# Hierarchical suites
# --------------------------------------------------------------------------
def run_L0():
    """L0 gate suite (DESIGN sec 8.3)."""
    print(f"[env] PHYS_CORES (queried AI-core count) = {PHYS_CORES}, "
          f"ASCEND_MODE = {ASCEND_MODE}")
    cases = [
        (64, 4, 128, 64, "float16", "uniform", "L0-1 smoke fp16, N<block_n"),
        (64, 4, 128, 64, "float16", "skewed", "L0-2 smoke fp16 skewed, dead tiles"),
        (128, 8, 384, 128, "float16", "uniform", "L0-3 K-aligned, num_pid_n>1"),
        (233, 5, 130, 70, "bfloat16", "skewed", "L0-4 odd shape, K tail (70)"),
        (8192, 128, 4096, 1024, "bfloat16", "uniform",
         "L0-5 representative large (extreme core split)"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    try:
        d = run_case_two_kernel(64, 4, 128, 64, "float16", "uniform",
                                "L0-6 two-kernel handoff")
        if d is not None:
            worst = max(worst, d)
    except Exception as exc:  # noqa: BLE001 - integration check is optional
        print(f"[L0-6] WARN two-kernel check failed (non-blocking): {exc}")
    print(f"L0 done ({len(cases)} cases + two-kernel), worst max_diff={worst:.3e}")
    return worst


def run_L1():
    cases = [
        (256, 8, 512, 128, "float16", "uniform", "L1-1 medium uniform, 2 pid_n"),
        (128, 16, 384, 192, "bfloat16", "uniform", "L1-2 E=16 bf16, K tail"),
        (300, 12, 200, 64, "float16", "skewed", "L1-3 skewed non-divisible"),
        (384, 10, 300, 100, "bfloat16", "empty", "L1-4 empty experts + K tail"),
        (512, 4, 768, 256, "float16", "uniform", "L1-5 multi local task"),
        (1024, 32, 640, 320, "float16", "skewed", "L1-6 large E skewed"),
        # --- Stage 4 (round 7) body coverage: the two compile-time guards ------
        # D2 paired body + block_n bump (numel/E = 128 -> bn=256).
        (2048, 16, 1024, 1024, "bfloat16", "skewed",
         "L1-7 D2 pair + bn=256 (mean_rows=128), K=2*2*bk"),
        # D2 paired body + dense experts (E > cores) + empty experts.
        (600, 300, 1024, 2048, "bfloat16", "empty",
         "L1-8 D2 pair, E>cores, empty experts"),
        # D2 paired body, M tail inside the last expert tile (skewed small M).
        (513, 9, 512, 1024, "float16", "skewed",
         "L1-9 D2 pair + M tail (513 rows, 9 experts)"),
        # Paired-K but N tail -> must fall back to the serial body (guard).
        (512, 8, 300, 1024, "float16", "uniform",
         "L1-10 K paired but N=300 tail -> serial fallback"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L1 done ({len(cases)} cases), worst max_diff={worst:.3e}")
    return worst


def run_L2():
    """Representative workload shapes (advisory; failures are recorded)."""
    cases = [
        (4096, 64, 2048, 1024, "bfloat16", "uniform", "L2-1 large uniform"),
        (8192, 128, 2048, 512, "bfloat16", "skewed", "L2-2 large skewed"),
        (32768, 256, 4096, 1024, "bfloat16", "uniform",
         "L2-3 prefill-scale numel/E/N (K reduced for harness budget)"),
    ]
    failures = []
    for c in cases:
        try:
            run_case(*c)
        except Exception as exc:  # noqa: BLE001 - L2 is advisory
            failures.append((c[6], str(exc)))
            print(f"[{c[6]}] WARN (non-blocking): {exc}")
    print(f"L2 done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def run_boundary():
    cases = [
        (64, 4, 16, 32, "float16", "uniform", "B-1 tiny N, K"),
        (70, 7, 70, 40, "bfloat16", "skewed", "B-2 all non-divisible"),
        (33, 4, 128, 33, "float16", "uniform", "B-3 odd numel/K"),
        (128, 8, 48, 16, "float16", "empty", "B-4 empty experts, tiny N/K"),
        (64, 1, 64, 64, "bfloat16", "uniform", "B-5 single expert"),
        (64, 4, 300, 64, "float16", "uniform", "B-6 N > block_n, N tail"),
        (64, 4, 256, 64, "float16", "single", "B-7 all tokens on expert 0"),
        (1, 1, 64, 64, "float16", "uniform", "B-8 numel=1 single row"),
        (600, 300, 128, 64, "bfloat16", "skewed",
         "B-9 E > cores, many one-token experts"),
        (0, 4, 64, 64, "float16", "empty", "B-10 total_tiles=0 (no work)"),
        # Stage 4: K is NOT a multiple of 2*block_k -> serial fallback, even
        # though N is aligned (proves the pair guard, not the N guard, fires).
        (600, 6, 512, 1100, "bfloat16", "uniform",
         "B-11 K=1100 (K % 2*bk != 0) -> serial fallback"),
    ]
    failures = []
    for c in cases:
        try:
            run_case(*c)
        except Exception as exc:  # noqa: BLE001 - Boundary is advisory
            failures.append((c[6], str(exc)))
            print(f"[{c[6]}] WARN (non-blocking): {exc}")
    print(f"Boundary done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L0()
        run_L1()
    else:
        run_L0()
        run_L1()
        l2_fail = run_L2()
        bd_fail = run_boundary()
        if l2_fail or bd_fail:
            # L2 / Boundary are advisory: recorded, never blocking.
            print(f"\033[93mAdvisory warns: L2={len(l2_fail)}, "
                  f"Boundary={len(bd_fail)} (non-blocking)\033[0m")
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
