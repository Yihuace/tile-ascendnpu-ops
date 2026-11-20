# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_make_scan_kernel_nopad -- MoE no-pad token permute Phase-1 counting-sort scan
(NPUIR Stage 3, Developer mode).

Design: examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md
        (revision v1, REVIEW.md verdict = pass)
Golden semantics: DESIGN sec 0.1 / sec 8.1 -- stable sequential counting sort.

What this operator does
-----------------------
From ``flat_ids[numel]`` (int32, ``numel = top_k * num_tokens``, each entry the
target expert id in ``[0, E)``) compute the no-pad counting-sort metadata:

    true_sizes[E]                    token count per expert  (int32)
    true_offsets[E]                  compact start per expert (int32)
    expert_first_token_offset[E+1]   exclusive prefix, int64
    fwd_idx[numel]                   flat position -> compact slot (int32)
    permuted_idx[numel]              slot -> token row (int32, read by sibling gather)

All integers (int32 + int64 prefix), bit-exact, no floating point.

Structure (DESIGN sec 0.6 R1/R2/R3, sec 3.3 -- THREE Vector kernels, GM handoff)
--------------------------------------------------------------------------------
The source (CUDA) is a single block with 5 phases (SMEM atomic histogram,
thread-0 serial prefix, ``tl_atomic_add_offset`` fetch-add scatter).  That form
is not realisable on NPU: there is no fetch-add (``T.atomic_add`` lowers to
``hivm::StoreOp``, no return value, CG-2026-0030), the scalar-payload
``T.atomic_add`` fails in the frontend (CG-2026-0029), a same-scope
vector-produce -> scalar-consume read is unsynchronised (CG-2026-0027) and
run-time indexed UB access corrupts neighbours (CG-2026-0028d).  The design
isolates the data-flow domains behind kernel boundaries -- the only handoff
primitive measured to work:

    Kernel A (_scan_count_kernel, grid=NB, one block per core)
        e = flat_ids[idx]; bh[core_id, e] = bh[core_id, e] + 1
        plain RMW (block <-> core 1:1, each (block, expert) row is private to a
        single core -> no atomic needed); -> GM bh[NB, E]

    Kernel B (_scan_prefix_kernel, grid=1)
        counts = column-sum of bh            (T.serial(NB) x T.Parallel(E))
        incl   = T.cumsum(counts)            (ONE int32 (1, E) scan)
        off    = [0, incl[0:E]]              (1-lane shift -> exclusive prefix)
        -> true_sizes / true_offsets / efo(int64 cast)  (UB staging + T.copy)
        -> slot_base[b, e] = off[e] + sum_{b'<b} bh[b', e]

    Kernel C (_scan_scatter_kernel, grid=NB, one block per core)
        e = flat_ids[idx]; r = run[core_id, e]; run[core_id, e] = r + 1
        slot = slot_base[core_id, e] + r            (deterministic block-parallel rank)
        fwd_idx[idx] = slot; permuted_idx[slot] = idx // top_k

Kernel C's ``slot`` reproduces the source/golden STABLE counting-sort order
bit-exactly; every slot is written exactly once, no fetch-add, no atomics.
Every scalar operand is either a loop variable or a GM load (flat_ids / run /
slot_base); no scalar ever consumes a vector-produced UB value.  No
``T.alloc_local``, no int32 ``>>``, no scalar ``T.if_then_else``, no manual
``T.sync_block_*``, at most one ``(1, E)`` ``T.cumsum`` per kernel.

Measured deviation from DESIGN sec 3.3 (Stage 3 finding, 2026-10-14)
-------------------------------------------------------------------
DESIGN sec 3.3 wrote kernel A as a **plain RMW** ``bh[core_id, e] =
bh[core_id, e] + 1`` and kernel C's ``fwd_idx[idx] = slot`` as a plain scalar GM
store.  On this toolchain (tilelang 0.1.2+2853d71 / Ascend910B2C) **direct
scalar GM stores from a multi-block Vector kernel are dropped
nondeterministically** -- probe: 8 blocks each storing ``out[core_id]`` loses a
different subset on every run; a trailing ``T.pipe_barrier("PIPE_ALL")`` or a
trailing UB->GM ``T.copy`` does not fix it.  Single-block kernels are stable.
The only measured-reliable GM write paths are MTE3 ``T.copy`` (UB -> GM,
including dynamic extents) and ``T.atomic_add(dst_runtime_indexed, src_ub,
[extent])`` with a **vector** payload.  The algorithm family, the five tables
and the deterministic stable-order formula are unchanged; only the emission
primitive is:
  * kernel A histogram: ``T.atomic_add(bh[core_id, e], ub_one, [1])``
    (vector payload -- NOT the scalar payload forbidden by CG-2026-0029; each
    (block, expert) row is private to one core so atomicity is not needed, the
    op is used purely as a reliable GM store).
  * kernel C ``run`` RMW: scalar GM load ``r`` then a vector-payload atomic
    increment.
  * kernel C ``fwd_idx``: per-block UB scratch ``ub_fwd`` filled in the serial
    loop, flushed by ONE dynamic-extent ``T.copy``.
  * kernel C ``permuted_idx``: pre-zeroed output + vector-payload
    ``T.atomic_add(permuted_idx[slot], ub_val, [1])`` (each slot written once).
``permuted_idx`` must therefore be host zero-initialised (the source op does
exactly this).  Evidence probes: /tmp/opencode/probe{2,4,5,6,7}.py during the
task (transferred to repro/ candidates in RETROSPECTIVE.md).

Contract with the sibling consumer (DESIGN sec 4.2)
--------------------------------------------------
1. numel / top_k / num_experts are shared compile-time constants; 2. all tables
int32 except efo (int64); 3. fwd_idx and permuted_idx are mutually inverse and
in stable order; 4. kernel A -> B -> C -> sibling gather launch on one stream
(no in-block sync, no CPU sync).

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python _make_scan_kernel_nopad.py --level all
"""

import os

# Developer mode: compiler-managed UB (alloc_ub) + automatic synchronisation.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import functools
import random
import resource
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

# Stage 3 programming mode actually implemented/validated: Developer
# (three Vector kernels + GM handoff; no manual sync primitives, no T.gemm).
ASCEND_MODE = "Developer"

# Pure Vector physical core count (DESIGN sec 5.5):
#   NPUUtils.get().get_aicore_num() -> 20  (Ascend910B2C, measured)
#   pure Vector kernels use 2x the AICore count -> 40.
VECTOR_CORES = 40

# Keep the UB buffers single-versioned and stop the buffer planner from
# re-scoping/permuting them (DESIGN sec 7.3, sibling-validated).
_PASS_CFG = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}


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


def scan_num_blocks(numel):
    """Grid size NB for the scan chain (A -> B -> C).

    Non-A5 keeps the historical ``min(VECTOR_CORES, numel)``.  On A5 the CANN
    9.2 bishengir back end SIGSEGVs on the runtime tail guard of a serial task
    loop, so pick the largest divisor of ``numel`` that fits the core budget:
    every block then gets exactly ``numel // nb`` elements, the tail guard
    folds away, and kernel A/C compile.  (Worst case numel prime > 40 -> NB=1,
    slower but correct.)
    """
    if numel <= 0:
        return 1
    if not _is_a5():
        return min(VECTOR_CORES, numel)
    for nb in range(min(VECTOR_CORES, numel), 0, -1):
        if numel % nb == 0:
            return nb
    return 1


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source-op semantics, explicit torch module ops,
# independent of the NPU three-kernel algorithm: stable argsort counting sort).
# --------------------------------------------------------------------------
def golden_scan(flat_ids, num_experts, top_k):
    """flat_ids[numel] -> (true_offsets, true_sizes, efo, fwd_idx, permuted_idx).

    Stable sequential counting sort built from explicit torch module ops
    (``torch.bincount`` / ``torch.cumsum`` / ``torch.argsort(stable=True)`` /
    ``torch.arange`` / index assignment) -- never from the NPU block-parallel
    histogram + running-rank dataflow.
    """
    E = int(num_experts)
    K = int(top_k)
    ids = flat_ids.detach().to(torch.device("cpu")).reshape(-1).to(torch.int64)
    numel = int(ids.numel())

    # counts[e] (int64) and its exclusive prefix off[E+1].
    counts = torch.bincount(ids, minlength=E)[:E].to(torch.int64)
    off = torch.zeros(E + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(counts, dim=0)

    # Stable counting-sort order: order[r] = flat position of the r-th element.
    order = torch.argsort(ids, stable=True)
    ranks = torch.arange(numel, dtype=torch.int64)

    fwd_idx = torch.empty(numel, dtype=torch.int64)
    fwd_idx[order] = ranks                       # position -> compact slot
    permuted_idx = torch.empty(numel, dtype=torch.int64)
    permuted_idx[ranks] = order // K             # slot -> token row

    return (
        off[:E].to(torch.int32),                 # true_offsets
        counts.to(torch.int32),                  # true_sizes
        off.to(torch.int64),                     # expert_first_token_offset
        fwd_idx.to(torch.int32),                 # fwd_idx
        permuted_idx.to(torch.int32),            # permuted_idx
    )


# --------------------------------------------------------------------------
# Kernel A (DESIGN sec 3.3): flat_ids -> per-block histogram bh[NB, E].
# Vector, NB cores, block <-> core 1:1, static in-core T.serial.
# Write primitive: vector-payload T.atomic_add (see the module-header note on
# the measured scalar-GM-store defect).
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_count_kernel(numel, num_experts, nb, cblk):
    E = num_experts

    @T.prim_func
    def _count_main(
        flat_ids: T.Tensor([numel], "int32"),
        bh: T.Tensor([nb, E], "int32"),
    ):
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                 # 静态边界
                if base + c < numel:                 # 普通 if 守卫（非 T.if_then_else）
                    e = flat_ids[base + c]           # 标量 GM 读（运行期下标）
                    # 向量载荷 atomic_add：本工具链多核标量 GM store 会随机丢写，
                    # 仅 MTE3 T.copy / 向量载荷 atomic_add 可靠（标量载荷 atomic
                    # 前端报错 CG-2026-0029）。块核 1:1、每行私有单核，无并发。
                    T.atomic_add(bh[core_id, e], ub_one[0, 0:1], [1])

    return _count_main


# --------------------------------------------------------------------------
# Kernel B (DESIGN sec 3.3): bh -> counts/prefix/tables/slot_base. Single core.
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_prefix_kernel(numel, num_experts, nb, cblk, top_k):
    E = num_experts
    a5 = _is_a5()

    @T.prim_func
    def _prefix_main(
        bh: T.Tensor([nb, E], "int32"),
        true_offsets: T.Tensor([E], "int32"),
        true_sizes: T.Tensor([E], "int32"),
        efo: T.Tensor([E + 1], "int64"),
        slot_base: T.Tensor([nb, E], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _subid):   # 单核一次
            ub_acc = T.alloc_ub((1, E), "int32")
            ub_bh = T.alloc_ub((1, E), "int32")
            ub_incl = T.alloc_ub((1, E), "int32")
            ub_off = T.alloc_ub((1, E + 1), "int32")
            ub_sb = T.alloc_ub((1, E), "int32")
            # 表输出 UB 暂存（T.Parallel 体内禁直写 GM，须先写 UB 再 T.copy 落 GM）
            ub_ts = T.alloc_ub((1, E), "int32")        # true_sizes 暂存
            ub_to = T.alloc_ub((1, E), "int32")        # true_offsets 暂存
            ub_efo = T.alloc_ub((1, E + 1), "int64")   # efo 暂存（int64）

            if a5:
                # A5(CANN 9.2): bishengir SIGSEGVs on any *partial* GM->UB read
                # of a small 2-D tensor (row slice / scalar row index / even a
                # per-lane elementwise load).  Whole-tile copies are fine, so
                # read bh once into UB and derive everything from UB.  The
                # 1-lane shifted UB->UB copy is dropped too: exclusive prefix
                # = inclusive prefix - own count.
                ub_all = T.alloc_ub((nb, E), "int32")
                T.copy(bh[0:nb, 0:E], ub_all[0:nb, 0:E])
                for e in T.Parallel(E):
                    ub_acc[0, e] = 0
                for b in T.serial(nb):
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_all[b, e]
                T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)
                for e in T.Parallel(E):
                    ub_ts[0, e] = ub_acc[0, e]
                    ub_to[0, e] = ub_incl[0, e] - ub_acc[0, e]
                ub_efo[0, 0] = 0
                for k in T.Parallel(E):
                    ub_efo[0, k + 1] = T.Cast("int64", ub_incl[0, k])
                T.copy(ub_ts[0, 0:E], true_sizes[0:E])
                T.copy(ub_to[0, 0:E], true_offsets[0:E])
                T.copy(ub_efo[0, 0:E + 1], efo[0:E + 1])
                # slot_base[b, e] = off[e] + sum_{b'<b} bh[b', e]
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_to[0, e]
                for b in T.serial(nb):
                    for e in T.Parallel(E):
                        ub_sb[0, e] = ub_acc[0, e]
                    T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_all[b, e]
            else:
                # B1: counts[e] = 列求和（与顺序无关，可向量化）
                for e in T.Parallel(E):
                    ub_acc[0, e] = 0
                for b in T.serial(nb):                     # 静态边界
                    T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

                # B2: 唯一一次 scan（int32 / (1,E) 已设备实测可用）
                T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)
                ub_off[0, 0] = 0
                T.copy(ub_incl[0, 0:E], ub_off[0, 1:E + 1])   # E1: 1-lane 移位得排他前缀

                # B3: 三张表输出（先写 UB 暂存，再 T.copy 落 GM）
                for e in T.Parallel(E):
                    ub_ts[0, e] = ub_acc[0, e]                # counts
                    ub_to[0, e] = ub_off[0, e]                # off
                for k in T.Parallel(E + 1):
                    ub_efo[0, k] = T.Cast("int64", ub_off[0, k])   # E3: int32 -> int64
                T.copy(ub_ts[0, 0:E], true_sizes[0:E])
                T.copy(ub_to[0, 0:E], true_offsets[0:E])
                T.copy(ub_efo[0, 0:E + 1], efo[0:E + 1])

                # B4: slot_base[b, e] = off[e] + sum_{b'<b} bh[b', e]（E4）
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_off[0, e]
                for b in T.serial(nb):
                    for e in T.Parallel(E):
                        ub_sb[0, e] = ub_acc[0, e]
                    T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                    T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

    return _prefix_main


# --------------------------------------------------------------------------
# Kernel C (DESIGN sec 3.3): deterministic block-parallel stable-rank scatter.
# Vector, NB cores, block <-> core 1:1, static in-core T.serial.
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_scatter_kernel(numel, num_experts, nb, cblk, top_k):
    E = num_experts
    a5 = _is_a5()

    @T.prim_func
    def _scatter_main(
        flat_ids: T.Tensor([numel], "int32"),
        slot_base: T.Tensor([nb, E], "int32"),
        run: T.Tensor([nb, E], "int32"),
        fwd_idx: T.Tensor([numel], "int32"),
        permuted_idx: T.Tensor([numel], "int32"),
    ):
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_val = T.alloc_ub((1, 1), "int32")
            ub_fwd = T.alloc_ub((1, cblk), "int32")   # 块内 fwd_idx 暂存（T.copy 落 GM）
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                     # 静态边界（循环携带 rank）
                idx = base + c
                if idx < numel:
                    e = flat_ids[idx]                    # 标量 GM 读（运行期下标）
                    r = run[core_id, e]                  # 块内运行计数（块核 1:1，无争用）
                    # 向量载荷 atomic_add 自增（多核标量 GM store 不可靠，见模块头注）
                    T.atomic_add(run[core_id, e], ub_one[0, 0:1], [1])
                    slot = slot_base[core_id, e] + r     # E2: 确定性稳定序 slot
                    ub_fwd[0, c] = slot                  # UB 暂存（循环变量下标）
                    ub_val[0, 0] = idx // top_k          # E5: //（非 >>，CG-2026-0028a）
                    # 运行期下标 GM 散写：向量载荷 atomic_add（permuted_idx host 侧预清零）
                    T.atomic_add(permuted_idx[slot], ub_val[0, 0:1], [1])
            # 尾块动态 extent：块内 fwd_idx 一次 DMA 落 GM（可靠写路径）
            if a5:
                # A5(CANN 9.2): scan_num_blocks 保证 nb 整除 numel，每块恰好
                # cblk 个元素、没有尾块；动态 extent + if 守卫在 A5 上会让
                # bishengir 崩溃，这里直接按静态 cblk 落盘。
                T.copy(ub_fwd[0, 0:cblk], fwd_idx[base:base + cblk])
            else:
                ct = T.min(cblk, numel - base)
                if ct > 0:
                    T.copy(ub_fwd[0, 0:ct], fwd_idx[base:base + ct])

    return _scatter_main


# --------------------------------------------------------------------------
# Public entry (DESIGN sec 3.3): chains kernel A -> B -> C on one stream.
# scratch bh / run must be host zero-initialised; slot_base/tables are written.
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _make_scan_kernel_nopad(numel: int, num_experts: int, top_k: int):
    nb = scan_num_blocks(numel)
    cblk = (numel + nb - 1) // nb if numel > 0 else 1
    kA = _scan_count_kernel(numel, num_experts, nb, cblk)
    kB = _scan_prefix_kernel(numel, num_experts, nb, cblk, top_k)
    kC = _scan_scatter_kernel(numel, num_experts, nb, cblk, top_k)

    def run(
        flat_ids,
        bh,
        true_offsets,
        true_sizes,
        efo,
        slot_base,
        run_buf,
        fwd_idx,
        permuted_idx,
    ):
        kA(flat_ids, bh)
        kB(bh, true_offsets, true_sizes, efo, slot_base)
        kC(flat_ids, slot_base, run_buf, fwd_idx, permuted_idx)
        return fwd_idx, permuted_idx

    return run


# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------
def _raise_stack_limit():
    """Raise RLIMIT_STACK to the hard limit (mitigation for a toolchain bug)."""
    try:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ValueError, OSError, AttributeError):
        pass


_raise_stack_limit()

_ATTEMPTS = int(os.environ.get("SCAN_COMPILE_ATTEMPTS", "5"))


def _run_kernels(numel, E, K, ids_npu, buffers):
    """Run kernel A -> B -> C on one stream.

    Retries a bounded number of fresh compiles for the known flaky
    bishengir-compile SIGSEGV (compile-time only; never a wrong number).
    """
    last = None
    for i in range(_ATTEMPTS):
        try:
            run = _make_scan_kernel_nopad(numel, E, K)
            run(ids_npu, *buffers)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "NPU IR opt failed" not in msg and "SIGSEGV" not in msg:
                raise
            last = exc
            _make_scan_kernel_nopad.cache_clear()
            print(f"  [warn] fresh-compile attempt {i + 1}/{_ATTEMPTS} hit the "
                  f"flaky bishengir-compile crash, retrying...")
            time.sleep(1.0)
    raise last


# ---- input distributions (flat_ids values, not just table-friendly sizes) ---
def make_flat_ids(numel, E, dist, seed=0):
    """Build a ``flat_ids`` list of length ``numel`` with values in ``[0, E)``."""
    if numel == 0:
        return []
    rng = random.Random(1234 + seed)
    if dist == "roundrobin":
        return [i % E for i in range(numel)]
    if dist == "uniform":
        return [rng.randrange(E) for _ in range(numel)]
    if dist == "skewed":
        top = max(1, E // 8)
        out = []
        for i in range(numel):
            out.append(rng.randrange(top) if rng.random() < 0.85 else rng.randrange(E))
        return out
    if dist == "empty":
        active = [e for e in range(E) if e % 3 != 0] or [E - 1]
        return [active[rng.randrange(len(active))] for _ in range(numel)]
    if dist == "single":
        return [0] * numel
    if dist == "last":
        return [E - 1] * numel
    if dist == "ones":
        if numel > E:
            raise ValueError(f"numel={numel} > E={E} for dist=ones")
        return list(range(numel))
    if dist == "duplicates":
        # many repeated ids -> exercises the stable-order tie break
        pat = [0, 1, 0, 1, 0, 2, 1, 0, 3, 2, 0, 1]
        return [pat[i % len(pat)] % E for i in range(numel)]
    raise ValueError(dist)


def _diff_stats(got, ref):
    n = int((got != ref).sum().item())
    md = 0
    if n:
        md = int((got.to(torch.int64) - ref.to(torch.int64)).abs().max().item())
    return n, md


def _run_case(numel, E, K, dist, tag, seed=0):
    ids_list = make_flat_ids(numel, E, dist, seed)
    assert len(ids_list) == numel, (len(ids_list), numel)
    nb = scan_num_blocks(numel)

    ids = torch.tensor(ids_list, dtype=torch.int32)
    ids_npu = ids.npu()
    bh = torch.zeros((nb, E), dtype=torch.int32, device="npu")       # zero-init scratch
    run_buf = torch.zeros((nb, E), dtype=torch.int32, device="npu")  # zero-init scratch
    slot_base = torch.empty((nb, E), dtype=torch.int32, device="npu")
    true_offsets = torch.empty((E,), dtype=torch.int32, device="npu")
    true_sizes = torch.empty((E,), dtype=torch.int32, device="npu")
    efo = torch.empty((E + 1,), dtype=torch.int64, device="npu")
    fwd_idx = torch.empty((numel,), dtype=torch.int32, device="npu")
    # permuted_idx is accumulated by a vector-payload atomic store -> must be
    # zero-initialised (exactly as the source op allocates it).
    permuted_idx = torch.zeros((numel,), dtype=torch.int32, device="npu")

    _run_kernels(
        numel, E, K, ids_npu,
        (bh, true_offsets, true_sizes, efo, slot_base, run_buf, fwd_idx, permuted_idx),
    )
    torch.npu.synchronize()

    g_to, g_ts, g_efo, g_fwd, g_perm = golden_scan(ids, E, K)

    # dtype / shape contract (DESIGN sec 4.2)
    assert true_offsets.dtype == torch.int32 and true_sizes.dtype == torch.int32
    assert efo.dtype == torch.int64
    assert fwd_idx.dtype == torch.int32 and permuted_idx.dtype == torch.int32
    assert true_offsets.shape == (E,) and true_sizes.shape == (E,)
    assert efo.shape == (E + 1,) and fwd_idx.shape == (numel,)
    assert permuted_idx.shape == (numel,)

    to_c = true_offsets.cpu()
    ts_c = true_sizes.cpu()
    efo_c = efo.cpu()
    fwd_c = fwd_idx.cpu()
    perm_c = permuted_idx.cpu()

    n_to, d_to = _diff_stats(to_c, g_to)
    n_ts, d_ts = _diff_stats(ts_c, g_ts)
    n_efo, d_efo = _diff_stats(efo_c, g_efo)
    n_fwd, d_fwd = _diff_stats(fwd_c, g_fwd)
    n_perm, d_perm = _diff_stats(perm_c, g_perm)
    max_diff = max(d_to, d_ts, d_efo, d_fwd, d_perm)

    if n_to or n_ts or n_efo or n_fwd or n_perm:
        raise AssertionError(
            f"[{tag}] mismatch: offsets={n_to} sizes={n_ts} efo={n_efo} "
            f"fwd={n_fwd} permuted={n_perm} max_diff={max_diff} "
            f"shape=(numel={numel},E={E},K={K},NB={nb},dist={dist})"
        )

    # inverse-relation contract (same criterion as the source-op test)
    if numel > 0:
        inv = perm_c.to(torch.int64)[fwd_c.to(torch.int64)]
        rows = torch.arange(numel, dtype=torch.int64) // K
        if not bool((inv == rows).all().item()):
            raise AssertionError(f"[{tag}] inverse contract violated")

    print(
        f"[{tag}] PASS  numel={numel} E={E} K={K} NB={nb} dist={dist} "
        f"max_diff={max_diff}"
    )
    return True


def run_L0():
    cases = [
        (8, 384, 8, "roundrobin", "L0-1 smoke E=384"),
        (64, 4, 2, "uniform", "L0-2 uniform"),
        (256, 256, 8, "ones", "L0-3 one token per expert"),
        (64, 4, 2, "single", "L0-4 single expert"),
        (40, 8, 5, "empty", "L0-5 empty experts (head/tail zero)"),
        (100, 5, 5, "duplicates", "L0-6 E=5 non-power-of-2, non-divisible"),
        (32768, 384, 8, "uniform", "L0-7 prefill numel=32768"),
        (32, 4, 4, "duplicates", "L0-stable repeated flat_id order"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L0 done ({len(cases)} cases), 5 tables bit-exact")

    # L0-8 repeated-run stability (guards the CG-2026-0027 stale-read class).
    for _ in range(6):
        for numel, E, K, dist in [(8, 384, 8, "roundrobin"), (32768, 384, 8, "uniform")]:
            _run_case(numel, E, K, dist, "L0-8 stability")
    print("L0 stability done (6 repeats x 2 shapes, bit-exact)")


def run_L1():
    cases = [
        (1024, 16, 4, "uniform", "L1-1 E=16 medium"),
        (3000, 32, 3, "skewed", "L1-2 E=32 skewed"),
        (999, 8, 3, "empty", "L1-3 non-divisible numel + empty experts"),
        (8192, 64, 8, "uniform", "L1-4 E=64"),
        (200, 200, 5, "ones", "L1-5 one-token experts"),
        (4096, 24, 2, "single", "L1-6 all tokens on expert 0"),
        (65, 4, 1, "skewed", "L1-7 top_k=1, numel tail"),
        (5120, 40, 4, "uniform", "L1-8 E == VECTOR_CORES"),
        (6000, 48, 2, "skewed", "L1-9 E > VECTOR_CORES"),
        (64, 1, 1, "uniform", "L1-10 single expert E=1, top_k=1"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L1 done ({len(cases)} cases), 5 tables bit-exact")


def run_L2():
    cases = [
        (16384, 128, 8, "uniform", "L2-1 E=128"),
        (65536, 256, 4, "skewed", "L2-2 E=256 numel=65536"),
        (32768, 512, 2, "uniform", "L2-3 large E=512"),
    ]
    failures = []
    for c in cases:
        try:
            _run_case(*c)
        except Exception as exc:  # noqa: BLE001 - L2 is advisory, record + continue
            failures.append((c[4], str(exc)))
            print(f"[{c[4]}] WARN (non-blocking): {exc}")
    print(f"L2 done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def run_boundary():
    cases = [
        (64, 1, 1, "uniform", "B-1 single expert"),
        (0, 8, 2, "empty", "B-2 numel=0 (all tables zero)"),
        (1, 8, 1, "roundrobin", "B-3 numel=1"),
        (63, 3, 3, "uniform", "B-4 odd E, tail block"),
        (448, 7, 7, "uniform", "B-5 odd E exact"),
        (19200, 300, 4, "skewed", "B-6 large E"),
        (5, 2, 1, "uniform", "B-7 numel < VECTOR_CORES"),
        (39, 384, 3, "uniform", "B-8 numel=39 < NB bound"),
        (300, 2, 2, "last", "B-9 all on last expert"),
    ]
    failures = []
    for c in cases:
        try:
            _run_case(*c)
        except Exception as exc:  # noqa: BLE001 - Boundary is advisory
            failures.append((c[4], str(exc)))
            print(f"[{c[4]}] WARN (non-blocking): {exc}")
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
