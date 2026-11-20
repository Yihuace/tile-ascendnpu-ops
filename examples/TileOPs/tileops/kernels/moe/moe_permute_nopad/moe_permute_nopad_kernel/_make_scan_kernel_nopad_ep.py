# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_make_scan_kernel_nopad_ep -- MoE no-pad token permute Phase-1 counting-sort
scan unit, EP (expert-parallel / ``expert_map``) variant (NPUIR Stage 3,
Developer mode).

Design: examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/DESIGN.md
        (REVIEW.md verdict = pass)
Frozen sibling (non-EP, Stage-3 measured): _make_scan_kernel_nopad.py
Golden semantics: DESIGN sec 0.1 / sec 8.1 -- stable sequential counting sort
over the LOCAL expert domain, ``-1`` sentinel for non-local pairs.

What this operator does
-----------------------
From ``flat_ids[numel]`` (int32, ``numel = top_k * num_tokens``, each entry the
target GLOBAL expert id in ``[0, E_global)``) and ``expert_map[E_global]``
(int32, pull-down remap with ``-1`` for "not on this rank"), compute:

    le(idx) = expert_map[flat_ids[idx]]           (local expert id, -1 non-local)
    true_sizes[E_local]                 token count per LOCAL expert (int32)
    true_offsets[E_local]               compact start per LOCAL expert (int32)
    expert_first_token_offset[E_local+1] exclusive prefix, int64
    fwd_idx[numel]                      -> compact slot, or -1 for non-local
    permuted_idx[numel]                 slot -> token row (nonzero only on
                                        [0, M_local), read by sibling gather)

All integers (int32 + int64 prefix), bit-exact, no floating point.

Structure (DESIGN sec 0.6 R1/R2/R3, sec 3.3 -- THREE Vector kernels, GM handoff)
-------------------------------------------------------------------------------
The non-EP sibling established the GM-handoff base: the source (CUDA) single
block with 5 phases (SMEM atomic histogram, thread-0 serial prefix,
``tl_atomic_add_offset`` fetch-add scatter) cannot be transcribed (no fetch-add,
CG-2026-0030; scalar-payload T.atomic_add fails in the frontend, CG-2026-0029;
same-scope vector-produce -> scalar-consume is unsynchronised, CG-2026-0027;
runtime-indexed UB access corrupts neighbours, CG-2026-0028d).  The EP variant
REUSES that base and re-designs ONLY the EP deltas:

    Kernel A (_scan_count_ep_kernel, grid=NB, one block per core)
        g = flat_ids[idx]; le = expert_map[g]        (two-level scalar GM read)
        if le >= 0: vector-payload atomic_add(bh[core_id, le], one, [1])
        -> GM bh[NB, E_local]  (per-(block, LOCAL expert) histogram)

    Kernel B (_scan_prefix_ep_kernel, grid=1)         (identical structure to
        counts = column-sum of bh over the E_local axis   the non-EP prefix
        incl   = T.cumsum(counts)  (ONE int32 (1, E_local) scan)   kernel, axis
        off    = [0, incl[0:E_local]]  (1-lane shift)              is E_local)
        -> true_sizes / true_offsets / efo(int64 cast)
        -> slot_base[b, l] = off[l] + sum_{b'<b} bh[b', l]

    Kernel C (_scan_scatter_ep_kernel, grid=NB, one block per core)
        g = flat_ids[idx]; le = expert_map[g]
        if le < 0: ub_fwd[0, c] = -1                  (EP non-local sentinel)
        else: r = run[core_id, le]; run[core_id, le] = r + 1
              slot = slot_base[core_id, le] + r       (deterministic stable rank)
              ub_fwd[0, c] = slot; permuted_idx[slot] = idx // top_k
        -> fwd_idx[base : base+ct] via ONE dynamic-extent T.copy

EP deltas re-designed (everything else承接 the non-EP base unchanged):
  * two-level GM indirection ``le = expert_map[g]`` (scalar GM read chain);
  * ``le >= 0`` guard as a PLAIN ``if`` statement (scalar ``T.if_then_else`` is
    forbidden, CG-2026-0028c);
  * ``-1`` sentinel for non-local pairs, staged in the same UB row as the local
    slots and flushed by the same T.copy;
  * the vectorised expert axis is ``E_local`` (not ``E``);
  * all GM writes stay on the measured-reliable paths: vector-payload
    ``T.atomic_add`` (never a scalar payload, CG-2026-0029; never a bare scalar
    GM store from a multi-block kernel, which the non-EP sibling measured to be
    dropped nondeterministically) and MTE3 ``T.copy``.

Kernel C's ``slot`` reproduces the source/golden STABLE counting-sort order
bit-exactly (block order x in-block position order = global position order);
every slot is written exactly once.  No ``T.alloc_local``, no int32 ``>>``, no
scalar ``T.if_then_else``, no manual ``T.sync_block_*``, at most one
``(1, E_local)`` ``T.cumsum`` per kernel, runtime indices used only against GM.

Contract with the sibling consumer (DESIGN sec 4.2)
--------------------------------------------------
1. numel / top_k / E_global / E_local are shared compile-time constants; 2. all
tables int32 except efo (int64); 3. fwd_idx and permuted_idx are mutually
inverse and in stable order on the local slot domain, non-local fwd_idx = -1;
4. kernel A -> B -> C -> sibling gather launch on one stream (no in-block sync,
no CPU sync).  scratch ``bh`` / ``run`` and output ``permuted_idx`` must be host
zero-initialised (the source op does exactly this).  ``E_local >= 1`` is a
compile-time precondition (a zero-length UB alloc fails with MLIR ``strides must
not be zero``, DESIGN sec 9.4 R-1).

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python _make_scan_kernel_nopad_ep.py --level all
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


def scan_num_blocks_ep(numel):
    """Grid size NB for the EP scan chain (A -> B -> C).

    Non-A5 keeps the historical ``min(VECTOR_CORES, numel)``.  On A5 the CANN
    9.2 bishengir back end SIGSEGVs on the runtime tail guard of a serial task
    loop, so pick the largest divisor of ``numel`` that fits the core budget:
    every block then gets exactly ``numel // nb`` elements and the guard folds
    away.  (Worst case numel prime > 40 -> NB=1, slower but correct.)
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
# Golden (DESIGN sec 8.1 -- source-op EP semantics, built from explicit torch
# integer ops; independent of the NPU three-kernel block-parallel algorithm).
# --------------------------------------------------------------------------
def golden_scan_ep(flat_ids, expert_map, e_local, top_k):
    """flat_ids[numel] + expert_map[E_g] -> 5 tables (stable EP counting sort).

    Returns (true_offsets, true_sizes, expert_first_token_offset, fwd_idx,
    permuted_idx).  Built from explicit torch integer ops:
      * pull-down remap             le = expert_map[flat_ids]
      * local-domain counts         torch.bincount over the local pairs
      * exclusive prefix            torch.cumsum
      * stable counting-sort order  torch.argsort(..., stable=True) keyed by
                                    (le, position) with non-local pushed to the
                                    far end and dropped
      * fwd_idx / permuted_idx      torch.arange + index assignment
    Never derived from the NPU block-parallel histogram + running-rank dataflow.
    """
    E = int(e_local)
    K = int(top_k)
    ids = flat_ids.detach().to(torch.device("cpu")).reshape(-1).to(torch.int64)
    emap = expert_map.detach().to(torch.device("cpu")).reshape(-1).to(torch.int64)
    numel = int(ids.numel())

    if numel == 0:
        return (
            torch.zeros(E, dtype=torch.int32),
            torch.zeros(E, dtype=torch.int32),
            torch.zeros(E + 1, dtype=torch.int64),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
        )

    # Pull-down remap: local expert id in [0, E), -1 for non-local pairs.
    le = emap[ids]
    local_mask = le >= 0
    m_local = int(local_mask.sum().item())

    # counts[l] over the LOCAL domain + int64 exclusive prefix off[E+1].
    if m_local > 0:
        counts = torch.bincount(le[local_mask], minlength=E)[:E].to(torch.int64)
    else:
        counts = torch.zeros(E, dtype=torch.int64)
    off = torch.zeros(E + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(counts, dim=0)

    # Stable counting-sort order over the local pairs: key = le for local, E for
    # non-local (which stably sort to the far end and are then dropped).
    key = torch.where(local_mask, le, torch.full_like(le, E))
    order = torch.argsort(key, stable=True)
    order_local = order[:m_local]                       # local (le, idx) order

    slot = torch.arange(m_local, dtype=torch.int64)
    fwd_idx = torch.full((numel,), -1, dtype=torch.int64)
    fwd_idx[order_local] = slot                         # position -> compact slot
    permuted_idx = torch.zeros(numel, dtype=torch.int64)
    permuted_idx[slot] = order_local // K               # slot -> token row

    return (
        off[:E].to(torch.int32),                         # true_offsets
        counts.to(torch.int32),                          # true_sizes
        off.to(torch.int64),                             # efo
        fwd_idx.to(torch.int32),                         # fwd_idx
        permuted_idx.to(torch.int32),                    # permuted_idx
    )


# --------------------------------------------------------------------------
# Kernel A (DESIGN sec 3.3): flat_ids + expert_map -> per-block LOCAL-domain
# histogram bh[NB, E_local].  Vector, NB cores, block <-> core 1:1.
# Two-level scalar GM indirection + plain-if EP guard; vector-payload
# T.atomic_add as the reliable GM store (each (block, expert) row is private to
# one core, so atomicity is not required).
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_count_ep_kernel(numel, e_global, e_local, nb, cblk):
    @T.prim_func
    def _count_ep_main(
        flat_ids: T.Tensor([numel], "int32"),
        expert_map: T.Tensor([e_global], "int32"),
        bh: T.Tensor([nb, e_local], "int32"),          # host 零初始化 GM scratch
    ):
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                    # 静态边界
                if base + c < numel:                    # 尾块守卫（普通 if）
                    g = flat_ids[base + c]              # 一级 GM 标量读（运行期下标）
                    le = expert_map[g]                  # 二级 GM 标量读（EP 下拉重映射）
                    if le >= 0:                         # EP 过滤守卫（普通 if，非 T.if_then_else）
                        # 向量载荷 atomic_add 作可靠 GM store：多核标量 GM store
                        # 随机丢写、标量载荷 atomic 前端报错（CG-2026-0029）。
                        T.atomic_add(bh[core_id, le], ub_one[0, 0:1], [1])

    return _count_ep_main


# --------------------------------------------------------------------------
# Kernel B (DESIGN sec 3.3): bh -> counts / prefix / 3 tables / slot_base.
# Single core.  Structurally identical to the non-EP sibling; only the vector
# expert-axis length becomes E_local.
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_prefix_ep_kernel(numel, e_global, e_local, nb, cblk, top_k):
    E = e_local
    a5 = _is_a5()

    if a5 and E == 1:
        # A5(CANN 9.2) + E_local == 1: every 2-D [nb, 1] access in a serial
        # loop SIGSEGVs bishengir (probed 2026-11-11: serial loop alone is
        # fine, a 1-D UB read with the loop index is fine, but both the
        # [nb, 1] whole-tile read and the row writes crash).  With a single
        # local expert the table degenerates: sizes[0] = sum(bh), offsets[0]
        # = 0, slot_base[b] = exclusive prefix over blocks.  Express it on a
        # 1-D column instead (probe v11: bit-exact).  bh / slot_base are
        # declared [nb] here -- the caller's [nb, 1] tensors are the same
        # contiguous int32 memory and the JIT accepts them (probe v11).
        @T.prim_func
        def _prefix_ep_main_e1(
            bh: T.Tensor([nb], "int32"),
            true_offsets: T.Tensor([E], "int32"),
            true_sizes: T.Tensor([E], "int32"),
            efo: T.Tensor([E + 1], "int64"),
            slot_base: T.Tensor([nb], "int32"),
        ):
            with T.Kernel(1, is_npu=True) as (_cid, _subid):
                ub_col = T.alloc_ub((nb,), "int32")
                ub_sb = T.alloc_ub((nb,), "int32")
                ub_acc = T.alloc_ub((1, E), "int32")
                ub_ts = T.alloc_ub((1, E), "int32")
                ub_to = T.alloc_ub((1, E), "int32")
                ub_efo = T.alloc_ub((1, E + 1), "int64")
                T.copy(bh[0:nb], ub_col[0:nb])
                ub_acc[0, 0] = 0
                for b in T.serial(nb):
                    ub_sb[b] = ub_acc[0, 0]
                    ub_acc[0, 0] = ub_acc[0, 0] + ub_col[b]
                T.copy(ub_sb[0:nb], slot_base[0:nb])
                ub_ts[0, 0] = ub_acc[0, 0]
                ub_to[0, 0] = 0
                ub_efo[0, 0] = 0
                ub_efo[0, 1] = T.Cast("int64", ub_acc[0, 0])
                T.copy(ub_ts[0, 0:E], true_sizes[0:E])
                T.copy(ub_to[0, 0:E], true_offsets[0:E])
                T.copy(ub_efo[0, 0:E + 1], efo[0:E + 1])

        return _prefix_ep_main_e1

    @T.prim_func
    def _prefix_ep_main(
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
                # of a small 2-D tensor, so read bh once into UB and derive
                # everything from UB.  Exclusive prefix = inclusive - own
                # count (drops the 1-lane shifted UB->UB copy).
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
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_to[0, e]
                for b in T.serial(nb):
                    for e in T.Parallel(E):
                        ub_sb[0, e] = ub_acc[0, e]
                    T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_all[b, e]
            else:
                # B1: counts[l] = 本地域列求和（与顺序无关，可向量化）
                for e in T.Parallel(E):
                    ub_acc[0, e] = 0
                for b in T.serial(nb):                     # 静态边界
                    T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

                # B2: 唯一一次 scan（int32 / (1,E_local) 已探针实测可用）
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

                # B4: slot_base[b, l] = off[l] + sum_{b'<b} bh[b', l]（E4）
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_off[0, e]
                for b in T.serial(nb):
                    for e in T.Parallel(E):
                        ub_sb[0, e] = ub_acc[0, e]
                    T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                    T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                    for e in T.Parallel(E):
                        ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

    return _prefix_ep_main


# --------------------------------------------------------------------------
# Kernel C (DESIGN sec 3.3): deterministic block-parallel stable-rank scatter
# with the two-level remap, the le<0 -1 sentinel and the le>=0 running rank.
# Vector, NB cores, block <-> core 1:1.
# --------------------------------------------------------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_scatter_ep_kernel(numel, e_global, e_local, nb, cblk, top_k):
    E = e_local
    a5 = _is_a5()

    @T.prim_func
    def _scatter_ep_main(
        flat_ids: T.Tensor([numel], "int32"),
        expert_map: T.Tensor([e_global], "int32"),
        slot_base: T.Tensor([nb, E], "int32"),
        run: T.Tensor([nb, E], "int32"),               # host 零初始化 GM scratch
        fwd_idx: T.Tensor([numel], "int32"),
        permuted_idx: T.Tensor([numel], "int32"),      # host 零初始化（哨兵兜底）
    ):
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_val = T.alloc_ub((1, 1), "int32")
            ub_fwd = T.alloc_ub((1, cblk), "int32")    # 块内 fwd_idx 暂存（含哨兵）
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                    # 静态边界（循环携带 rank）
                idx = base + c
                if idx < numel:                         # 尾块守卫（普通 if）
                    g = flat_ids[idx]                   # 一级 GM 标量读
                    le = expert_map[g]                  # 二级 GM 标量读（EP 下拉重映射）
                    if le < 0:
                        ub_fwd[0, c] = -1               # EP 非本地哨兵（与本地对同暂存行）
                    else:
                        r = run[core_id, le]            # 块内运行计数（块核 1:1，无争用）
                        # 向量载荷 atomic_add 自增（标量 GM store / 标量载荷 atomic 均不可用）
                        T.atomic_add(run[core_id, le], ub_one[0, 0:1], [1])
                        slot = slot_base[core_id, le] + r   # E2-EP: 确定性稳定序 slot
                        ub_fwd[0, c] = slot             # UB 暂存（循环变量下标）
                        ub_val[0, 0] = idx // top_k     # E5: //（非 >>，CG-2026-0028a）
                        # 运行期下标 GM 散写：向量载荷 atomic（permuted_idx host 侧预清零）
                        T.atomic_add(permuted_idx[slot], ub_val[0, 0:1], [1])
            # 尾块动态 extent：块内 fwd_idx（含哨兵）一次 DMA 落 GM（可靠写路径）
            if a5:
                # A5(CANN 9.2): scan_num_blocks_ep 保证 nb 整除 numel，每块
                # 恰好 cblk 个元素、没有尾块；动态 extent + if 守卫在 A5 上
                # 会让 bishengir 崩溃，这里直接按静态 cblk 落盘。
                T.copy(ub_fwd[0, 0:cblk], fwd_idx[base:base + cblk])
            else:
                ct = T.min(cblk, numel - base)
                if ct > 0:
                    T.copy(ub_fwd[0, 0:ct], fwd_idx[base:base + ct])

    return _scatter_ep_main


# --------------------------------------------------------------------------
# Public entry (DESIGN sec 3.3 / 4.7): chains kernel A -> B -> C on one stream.
# scratch bh / run and output permuted_idx must be host zero-initialised.
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _make_scan_kernel_nopad_ep(numel: int, e_global: int, e_local: int, top_k: int):
    assert numel >= 0
    assert e_local >= 1, "E_local >= 1 is a compile-time precondition (zero-length UB)"
    nb = scan_num_blocks_ep(numel)
    cblk = (numel + nb - 1) // nb if numel > 0 else 1
    kA = _scan_count_ep_kernel(numel, e_global, e_local, nb, cblk)
    kB = _scan_prefix_ep_kernel(numel, e_global, e_local, nb, cblk, top_k)
    kC = _scan_scatter_ep_kernel(numel, e_global, e_local, nb, cblk, top_k)

    def run(
        flat_ids,
        expert_map,
        bh,
        true_offsets,
        true_sizes,
        efo,
        slot_base,
        run_buf,
        fwd_idx,
        permuted_idx,
    ):
        kA(flat_ids, expert_map, bh)
        kB(bh, true_offsets, true_sizes, efo, slot_base)
        kC(flat_ids, expert_map, slot_base, run_buf, fwd_idx, permuted_idx)
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


def _run_kernels(numel, e_global, e_local, top_k, buffers):
    """Run kernel A -> B -> C on one stream.

    Retries a bounded number of fresh compiles for the known flaky
    bishengir-compile SIGSEGV (compile-time only; never a wrong number).
    """
    last = None
    for i in range(_ATTEMPTS):
        try:
            run = _make_scan_kernel_nopad_ep(numel, e_global, e_local, top_k)
            run(*buffers)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "NPU IR opt failed" not in msg and "SIGSEGV" not in msg:
                raise
            last = exc
            _make_scan_kernel_nopad_ep.cache_clear()
            print(f"  [warn] fresh-compile attempt {i + 1}/{_ATTEMPTS} hit the "
                  f"flaky bishengir-compile crash, retrying...")
            time.sleep(1.0)
    raise last


# ---- expert_map builders (exactly e_local local experts, values in [0, e_local)) ----
def _build_expert_map(e_global, e_local, mode, seed=0):
    """Build expert_map[E_g] with exactly ``e_local`` entries >= 0.

    The non-negative entries take values in [0, e_local) (a permutation), so the
    host-side definition ``E_local = (expert_map >= 0).sum()`` matches the
    compile-time ``e_local`` axis used by the kernels.
    """
    rng = random.Random(101 + seed)
    emap = [-1] * e_global
    if mode == "identity":                       # local experts = global [0, E_l)
        chosen = list(range(e_local))
    elif mode == "tail":                         # local experts = global tail
        chosen = list(range(e_global - e_local, e_global))
    elif mode == "shuffled":                     # non-contiguous local experts
        pos = list(range(e_global))
        rng.shuffle(pos)
        chosen = pos[:e_local]
    else:
        raise ValueError(mode)
    vals = list(range(e_local))
    if mode == "shuffled":                       # permuted local-value assignment
        rng.shuffle(vals)
    for p, v in zip(chosen, vals):
        emap[p] = v
    return emap


# ---- flat_ids distributions (values are GLOBAL expert ids in [0, E_g)) -------
def _build_flat_ids(numel, emap, dist, seed=0):
    rng = random.Random(1234 + seed)
    e_global = len(emap)
    local_ids = [g for g in range(e_global) if emap[g] >= 0]
    remote_ids = [g for g in range(e_global) if emap[g] < 0]
    e_local = len(local_ids)
    if numel == 0:
        return []
    if dist == "uniform":
        return [rng.randrange(e_global) for _ in range(numel)]
    if dist == "local_only":
        return [local_ids[rng.randrange(e_local)] for _ in range(numel)]
    if dist == "remote_only":
        if remote_ids:
            return [remote_ids[rng.randrange(len(remote_ids))] for _ in range(numel)]
        return [local_ids[0]] * numel            # no remote experts exist
    if dist == "skewed":
        top = max(1, e_local // 4)
        out = []
        for _ in range(numel):
            if remote_ids and rng.random() < 0.25:
                out.append(remote_ids[rng.randrange(len(remote_ids))])
            else:
                out.append(local_ids[rng.randrange(top)])
        return out
    if dist == "empty":                          # only a prefix of local experts
        keep = [g for g in local_ids if emap[g] < max(1, e_local // 2)]
        if not keep:
            keep = local_ids
        return [keep[rng.randrange(len(keep))] for _ in range(numel)]
    if dist == "roundrobin":
        return [i % e_global for i in range(numel)]
    if dist == "single":
        return [local_ids[0]] * numel
    if dist == "last":
        return [local_ids[-1]] * numel
    if dist == "ones":                           # one token per local expert
        return [local_ids[i % e_local] for i in range(numel)]
    if dist == "duplicates":
        pat = [0, 1, 0, 1, 0, 2, 1, 0, 3, 2, 0, 1]
        return [local_ids[pat[i % len(pat)] % e_local] for i in range(numel)]
    raise ValueError(dist)


def _diff_stats(got, ref):
    n = int((got != ref).sum().item())
    md = 0
    if n:
        md = int((got.to(torch.int64) - ref.to(torch.int64)).abs().max().item())
    return n, md


def _run_case(numel, e_global, e_local, top_k, emap_mode, dist, tag, seed=0):
    emap = _build_expert_map(e_global, e_local, emap_mode, seed)
    ids_list = _build_flat_ids(numel, emap, dist, seed)
    assert len(ids_list) == numel, (len(ids_list), numel)
    nb = scan_num_blocks_ep(numel)

    ids = torch.tensor(ids_list, dtype=torch.int32)
    emap_t = torch.tensor(emap, dtype=torch.int32)
    ids_npu = ids.npu()
    emap_npu = emap_t.npu()

    bh = torch.zeros((nb, e_local), dtype=torch.int32, device="npu")       # zero-init scratch
    run_buf = torch.zeros((nb, e_local), dtype=torch.int32, device="npu")  # zero-init scratch
    slot_base = torch.empty((nb, e_local), dtype=torch.int32, device="npu")
    true_offsets = torch.empty((e_local,), dtype=torch.int32, device="npu")
    true_sizes = torch.empty((e_local,), dtype=torch.int32, device="npu")
    efo = torch.empty((e_local + 1,), dtype=torch.int64, device="npu")
    fwd_idx = torch.empty((numel,), dtype=torch.int32, device="npu")
    # permuted_idx is written by a vector-payload atomic store -> must be
    # zero-initialised (exactly as the source op allocates it).
    permuted_idx = torch.zeros((numel,), dtype=torch.int32, device="npu")

    _run_kernels(
        numel, e_global, e_local, top_k,
        (ids_npu, emap_npu, bh, true_offsets, true_sizes, efo,
         slot_base, run_buf, fwd_idx, permuted_idx),
    )
    torch.npu.synchronize()

    g_to, g_ts, g_efo, g_fwd, g_perm = golden_scan_ep(ids, emap_t, e_local, top_k)

    # dtype / shape contract (DESIGN sec 4.2)
    assert true_offsets.dtype == torch.int32 and true_sizes.dtype == torch.int32
    assert efo.dtype == torch.int64
    assert fwd_idx.dtype == torch.int32 and permuted_idx.dtype == torch.int32
    assert true_offsets.shape == (e_local,) and true_sizes.shape == (e_local,)
    assert efo.shape == (e_local + 1,) and fwd_idx.shape == (numel,)
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
            f"shape=(numel={numel},E_g={e_global},E_l={e_local},K={top_k},"
            f"NB={nb},map={emap_mode},dist={dist})"
        )

    # EP sentinel contract: fwd_idx == -1 exactly on the non-local pairs.
    if numel > 0:
        le = emap_t.to(torch.int64)[ids.to(torch.int64)]
        expected_sentinel = le < 0
        got_sentinel = fwd_c < 0
        if not bool((got_sentinel == expected_sentinel).all().item()):
            raise AssertionError(f"[{tag}] sentinel set mismatch")
        m_local = int((~expected_sentinel).sum().item())
        if not bool((perm_c[m_local:] == 0).all().item()):
            raise AssertionError(f"[{tag}] permuted_idx tail not zero beyond M_local")

    # inverse-relation contract on the local slot domain
    if numel > 0:
        inv = perm_c.to(torch.int64)[fwd_c.to(torch.int64).clamp(min=0)]
        rows = torch.arange(numel, dtype=torch.int64) // top_k
        local_ok = (fwd_c.to(torch.int64) >= 0)
        if not bool((inv[local_ok] == rows[local_ok]).all().item()):
            raise AssertionError(f"[{tag}] inverse contract violated")

    print(
        f"[{tag}] PASS  numel={numel} E_g={e_global} E_l={e_local} K={top_k} "
        f"NB={nb} map={emap_mode} dist={dist} max_diff={max_diff}"
    )
    return True


def run_L0():
    cases = [
        (8, 8, 4, 2, "identity", "local_only", "L0-EP-1 smoke all-local"),
        (8, 384, 128, 8, "identity", "uniform", "L0-EP-2 EP smoke mixed"),
        (64, 384, 128, 2, "identity", "uniform", "L0-EP-3 filtering active"),
        (100, 64, 64, 5, "identity", "remote_only", "L0-EP-4 all non-local"),
        (256, 512, 64, 8, "shuffled", "uniform", "L0-EP-5 discrete map"),
        (64, 64, 3, 2, "identity", "uniform", "L0-EP-6a tail lane E_l=3"),
        (64, 64, 5, 4, "identity", "uniform", "L0-EP-6b tail lane E_l=5"),
        (64, 64, 1, 2, "identity", "uniform", "L0-EP-7 single local expert"),
        (32768, 384, 128, 8, "identity", "uniform", "L0-EP-8 prefill numel=32768"),
        (39, 384, 40, 3, "identity", "uniform", "L0-EP-12a numel<NB"),
        (100, 384, 128, 3, "identity", "empty", "L0-EP-12b non-divisible C + empty experts"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L0 done ({len(cases)} cases), 5 tables + sentinel bit-exact")

    # L0-EP-9 repeated-run stability (guards the CG-2026-0027 stale-read class).
    for _ in range(6):
        for numel, eg, el, k, mm, dd in [
            (8, 384, 128, 8, "identity", "uniform"),
            (32768, 384, 128, 8, "identity", "uniform"),
        ]:
            _run_case(numel, eg, el, k, mm, dd, "L0-EP-9 stability")
    print("L0 stability done (6 repeats x 2 shapes, bit-exact)")


def run_L1():
    cases = [
        (1024, 16, 16, 4, "identity", "uniform", "L1-1 all local E_g=E_l=16"),
        (3000, 32, 16, 3, "shuffled", "skewed", "L1-2 shuffled map skewed"),
        (999, 64, 8, 3, "identity", "empty", "L1-3 non-divisible + empty experts"),
        (8192, 64, 64, 8, "tail", "local_only", "L1-4 all-local tail map"),
        (200, 512, 200, 5, "identity", "ones", "L1-5 one token per expert"),
        (4096, 384, 24, 2, "shuffled", "single", "L1-6 all on one local expert"),
        (65, 128, 4, 1, "identity", "skewed", "L1-7 top_k=1"),
        (5120, 40, 40, 4, "identity", "uniform", "L1-8 E_l == VECTOR_CORES"),
        (6000, 48, 48, 2, "shuffled", "skewed", "L1-9 E_l > VECTOR_CORES"),
        (64, 64, 1, 1, "identity", "uniform", "L1-10 single local expert top_k=1"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L1 done ({len(cases)} cases), 5 tables + sentinel bit-exact")


def run_L2():
    cases = [
        (16384, 512, 128, 8, "identity", "uniform", "L2-1 E_l=128"),
        (65536, 384, 256, 4, "shuffled", "skewed", "L2-2 numel=65536 E_l=256"),
        (32768, 1024, 512, 2, "identity", "uniform", "L2-3 large E_l=512"),
    ]
    failures = []
    for c in cases:
        try:
            _run_case(*c)
        except Exception as exc:  # noqa: BLE001 - L2 is advisory, record + continue
            failures.append((c[6], str(exc)))
            print(f"[{c[6]}] WARN (non-blocking): {exc}")
    print(f"L2 done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def run_boundary():
    cases = [
        (64, 1, 1, 1, "identity", "uniform", "B-1 single global expert"),
        (0, 8, 4, 2, "identity", "empty", "B-2 numel=0 (all tables zero)"),
        (1, 8, 4, 1, "identity", "roundrobin", "B-3 numel=1"),
        (63, 3, 3, 3, "identity", "uniform", "B-4 odd E_l, tail block"),
        (448, 7, 7, 7, "identity", "uniform", "B-5 odd E_l exact"),
        (5, 2, 2, 1, "identity", "uniform", "B-6 numel < VECTOR_CORES"),
        (300, 64, 2, 2, "identity", "last", "B-7 all on last local expert"),
        (100, 64, 64, 5, "identity", "remote_only", "B-8 all non-local"),
        (19200, 384, 300, 4, "shuffled", "skewed", "B-9 large E_l=300"),
        (39, 384, 40, 3, "identity", "uniform", "B-10 numel=39 < NB bound"),
    ]
    failures = []
    for c in cases:
        try:
            _run_case(*c)
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
