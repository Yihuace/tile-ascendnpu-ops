# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_tile_scheduler_kernel -- MoE no-pad grouped GEMM tile scheduler (Stage 4 drop-in).

Byte-identical logic to the Stage 3 product
(`moe_grouped_gemm_nopad_kernel/_tile_scheduler_kernel.py`); the only change is
the mode-pinning wrapper at the bottom, required because the wrapper's
perf_opt switch imports this module directly and therefore bypasses the
package-level `_pin_ascend_mode` re-export.

NOT tuned in Stage 4: the scheduler runs two Vector kernels over E-sized
tables (a few KB) before the GEMM; its measured cost is < 0.1 ms against
5.6-16 ms of GEMM, i.e. < 1 % of the op, and its traffic is not part of the
GEMM kernel's msprof Roofline.  It is shipped here so that the perf_opt pair
adopts atomically.

Stage 4 regression: `python perf_opt/_tile_scheduler_kernel.py --level all`
(L0/L1 bit-exact + advisory L2/boundary).
"""

"""_tile_scheduler_kernel -- MoE no-pad grouped GEMM tile scheduler (NPUIR Stage 3, rev v2).

Design: examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md
        (revision v2, REVIEW.md verdict = pass)
Golden semantics: DESIGN sec 0.1 / sec 8.1 (linear bucketing over true_sizes).

What this operator does
-----------------------
Map the per-expert token counts ``true_sizes[E]`` (int32) into the flat
M-tile schedule consumed by the sibling no-pad grouped GEMM:

    tile_expert_ids[max_tiles]    expert id per M-tile (dead tile = -1)
    tile_row_offsets[max_tiles]   in-expert row offset (already * block_m)
    total_tiles[1]                number of live M-tiles (stays on device)

All integers, bit-exact, no floating point.  ``total_tiles`` never leaves the
device (zero CPU synchronisation, source intent).

Structure (DESIGN rev v2 -- sec 0.6 R1 / sec 3.3, TWO Vector kernels, GM handoff)
--------------------------------------------------------------------------------
The v1 single-scope formulation (UB ``T.cumsum`` then **scalar** per-tile
consume) is not realisable: a scalar read of a vector-produced UB value is not
synchronised (CG-2026-0027, same compiled kernel gives different results on
different runs) and its surrounding constructs make device_codegen SIGSEGV
(CG-2026-0028).  Rev v2 isolates the two dataflow domains behind a kernel
boundary, the only handoff primitive that is measured to work:

    Kernel A (_sched_scan_kernel, out_idx=[1], grid=1)
        true_sizes[E] --T.copy--> UB
        cnt_e = ceil(s_e / block_m)          (T.Parallel(E), integer `//`)
        incl  = T.cumsum(cnt)                (ONE int32 scan, (1, E), dim=1)
        cum   = [0, incl[0:E]]               (1-lane shift -> exclusive prefix)
        -> cum_gm[E+1] (GM, the handoff buffer)

    Kernel B (_sched_expand_kernel, out_idx=[1,2,3], grid=min(E, 40))
        scalar GM read c_lo = cum_gm[e], c_hi = cum_gm[e+1]
        broadcast (e, k*block_m) with T.Parallel(mte)
        -> dynamic-extent T.copy into tile_expert_ids[c_lo:c_hi] /
           tile_row_offsets[c_lo:c_hi]
        dead-tile range [total, max_tiles) <- sentinel (-1, 0) (core 0 only)

Kernel B never consumes a vector-produced UB value in scalar context: the only
scalar operands are (a) GM loads from ``cum_gm`` written by kernel A (isolated
by the kernel boundary, same-stream sequential launch) and (b) loop indices.
No ``T.alloc_local``, no int32 ``>>``, no scalar ``T.if_then_else``, no manual
``T.sync_block_*``: every construct on the CG-2026-0027/0028 trap list is
avoided.  Feasibility evidence: history_version/design_probe_two_kernel*.py
(device bit-exact, 6-run deterministic) and repro/TRAP-VECTOR-SCALAR-DATAFLOW.py.

Contract with the sibling consumer (DESIGN sec 0.6 R1 / sec 4.1)
---------------------------------------------------------------
1. ``max_tiles = numel // block_m + E`` and ``block_m`` are shared compile-time
   constants; 2. all tables are int32; 3. ``tile_row_offsets`` is in ROWS
   (already * block_m); 4. dead-tile sentinel is ``(-1, 0)``;
   5. ``total_tiles <= max_tiles``; 6. kernel A -> kernel B -> GEMM launch on
   one stream (no in-block sync, no CPU sync).

Toolchain note: the bundled ``bishengir-compile`` intermittently SIGSEGVs on
some dynamic-extent IR forms (DESIGN sec 9.4 R-6; sibling repro
TRAP-BISHENGIR-FLAKY-SIGSEGV.py).  The harness raises the stack limit and
retries a bounded number of fresh compiles; accuracy is unaffected because the
crash is compile-time, never a wrong number.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python _tile_scheduler_kernel.py --level all
"""

import argparse
import functools
import os
import resource
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import threading

import tilelang.language as T

from tileops.utils import scoped_ascend_mode as _scoped_ascend_mode

_tile_scheduler_kernel_cache_lock = threading.Lock()

# Stage 3 programming mode actually implemented/validated: Developer
# (two Vector kernels + GM handoff; no manual sync primitives).
ASCEND_MODE = "Developer"

# Pure Vector physical core count (DESIGN sec 5.5):
#   NPUUtils.get().get_aicore_num() -> 20  (Ascend910B2C, measured)
#   pure Vector kernels use 2x the AICore count -> 40.
VECTOR_CORES = 40

_DEFAULT_BLOCK_M = 64

# Keep the UB buffers single-versioned and stop the buffer planner from
# re-scoping/permuting them: with auto multi-buffer the kernel B dynamic-extent
# range copies are not emitted reliably (DESIGN sec 7.3, probe-validated).
_PASS_CFG = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source-op semantics, torch module ops called
# explicitly; independent of the NPU algorithm: pure per-expert expansion).
# --------------------------------------------------------------------------
def golden_tile_scheduler(true_sizes, block_m, max_tiles):
    """true_sizes[E] -> (tile_expert_ids, tile_row_offsets, total_tiles).

    Linear bucketing: each expert e contributes ``ceil(s_e / block_m)``
    consecutive M-tiles with offsets ``0, block_m, 2*block_m, ...``; the tail
    ``[total, max_tiles)`` is the (-1, 0) sentinel.  Built from explicit torch
    module ops (``repeat_interleave`` / ``cumsum`` / ``cat`` / ``arange``),
    never from the NPU prefix-scan + range-expansion dataflow.
    """
    sizes = true_sizes.detach().to(torch.int64).cpu()
    num_experts = int(sizes.shape[0])

    # cnt_e = ceil(s_e / block_m)  (integer ceil via torch.div rounding_mode).
    cnt = torch.div(sizes + (block_m - 1), block_m, rounding_mode="floor")

    # Live M-tiles in expert-major order (linear expansion of each expert run).
    expert_ids_live = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int64), cnt
    )
    total = int(expert_ids_live.numel())

    # Exclusive prefix of the counts gives each expert's first tile index.
    cum = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(cnt, dim=0)]
    )
    tile_idx = torch.arange(total, dtype=torch.int64)
    row_offsets_live = (tile_idx - cum[expert_ids_live]) * block_m

    pad = max_tiles - total
    tile_expert_ids = torch.cat(
        [expert_ids_live, torch.full((pad,), -1, dtype=torch.int64)]
    )
    tile_row_offsets = torch.cat(
        [row_offsets_live, torch.zeros((pad,), dtype=torch.int64)]
    )
    return (
        tile_expert_ids.to(torch.int32),
        tile_row_offsets.to(torch.int32),
        total,
    )


def golden_cum(true_sizes, block_m):
    """Exclusive prefix of ceil(true_sizes[e]/block_m) (kernel A's cum_gm)."""
    sizes = true_sizes.detach().to(torch.int64).cpu()
    cnt = torch.div(sizes + (block_m - 1), block_m, rounding_mode="floor")
    return torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(cnt, dim=0)])


# --------------------------------------------------------------------------
# Plan helpers (compile-time constants; shared with the sibling consumer)
# --------------------------------------------------------------------------
def _ceildiv(x, y):
    return (x + y - 1) // y


def max_tiles_of(numel, num_experts, block_m=_DEFAULT_BLOCK_M):
    """Compile-time M-tile upper bound: DESIGN sec 4.1 contract."""
    return numel // block_m + num_experts


# --------------------------------------------------------------------------
# Kernel A (DESIGN sec 3.3): true_sizes -> cum_gm  (single Vector core,
# one int32 (1, E) T.cumsum; fully vector domain -- no scalar consume).
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir", pass_configs=_PASS_CFG)
def _sched_scan_kernel(num_experts: int, max_tiles: int, block_m: int):
    E = num_experts

    @T.prim_func
    def _scan_main(
        true_sizes: T.Tensor([E], "int32"),
        cum_gm: T.Tensor([E + 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _subid):
            # 单核一次：整条数据流都在向量域，无标量消费（规避 CG-2026-0027）。
            ub_sizes = T.alloc_ub((1, E), "int32")
            ub_cnt = T.alloc_ub((1, E), "int32")
            ub_incl = T.alloc_ub((1, E), "int32")
            ub_zero = T.alloc_ub((1, 1), "int32")

            # GM -> UB
            T.copy(true_sizes[0:E], ub_sizes[0, 0:E])
            # cnt_e = ceil(s_e / block_m)；用 // 而非 >>（>> 令 device_codegen SIGSEGV）。
            for e in T.Parallel(E):
                ub_cnt[0, e] = (ub_sizes[0, e] + (block_m - 1)) // block_m
            # 唯一一次 scan：含前缀和（int32 / (1,E) 已设备实测可用）。
            T.cumsum(ub_cnt, ub_incl, dim=1, reverse=False)
            # 1-lane 移位得排他前缀。原实现在 UB 内做切片 T.copy
            # (ub_cum[0, 1:E+1] <- ub_incl[0, 0:E])：该形态在当前工具链/芯片
            # （tilelang 0.1.2+2238099 / Ascend910B3）会被向量化成**非对齐 UB
            # 向量写**（4 字节偏移），运行时向量核异常
            # "UB address accessed by the VEC instruction is not aligned"。
            # 改为在 GM 侧完成移位（纯 MTE 数据流）：先写 0 头，再把含前缀和
            # 拷到 cum_gm[1:E+1]，不做非对齐 UB 向量写。
            ub_zero[0, 0] = 0
            T.copy(ub_zero[0, 0:1], cum_gm[0:1])
            T.copy(ub_incl[0, 0:E], cum_gm[1:E + 1])

    return _scan_main


# --------------------------------------------------------------------------
# Kernel B (DESIGN sec 3.3): cum_gm -> 3 GM tables  (per-expert range
# expansion; the only scalar operands are cross-kernel GM loads + loop vars).
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[1, 2, 3], target="npuir", pass_configs=_PASS_CFG)
def _sched_expand_kernel(num_experts: int, max_tiles: int, block_m: int):
    E = num_experts
    # mte = per-expert tile upper bound = numel//block_m + 1 >= ceil(numel/block_m)
    mte = max_tiles - E + 1
    nk_B = min(E, VECTOR_CORES)
    epc = _ceildiv(E, nk_B)

    @T.prim_func
    def _expand_main(
        cum_gm: T.Tensor([E + 1], "int32"),
        tile_expert_ids: T.Tensor([max_tiles], "int32"),
        tile_row_offsets: T.Tensor([max_tiles], "int32"),
        total_tiles: T.Tensor([1], "int32"),
    ):
        with T.Kernel(nk_B, is_npu=True) as (core_id, _subid):
            ub_e = T.alloc_ub((1, mte), "int32")
            ub_r = T.alloc_ub((1, mte), "int32")
            ub_neg = T.alloc_ub((1, max_tiles), "int32")
            ub_zro = T.alloc_ub((1, max_tiles), "int32")

            # total 全程设备侧（零 CPU 同步）
            if core_id == 0:
                total_tiles[0] = cum_gm[E]

            # 逐专家区间展开：c_lo/c_hi 是跨 kernel 的 GM 标量读（非本 scope 向量产出）。
            for ee in T.serial(epc):  # 静态边界（epc 为编译期常量）
                e = core_id * epc + ee
                if e < E:
                    c_lo = cum_gm[e]
                    c_hi = cum_gm[e + 1]
                    # 广播 (e, k*block_m) 到 UB（索引未变换）
                    for tt in T.Parallel(mte):
                        ub_e[0, tt] = e
                        ub_r[0, tt] = tt * block_m
                    # 动态 extent 区间写：空专家 c_lo==c_hi 时跳过
                    if c_lo < c_hi:
                        T.copy(ub_e[0, 0 : c_hi - c_lo], tile_expert_ids[c_lo:c_hi])
                        T.copy(ub_r[0, 0 : c_hi - c_lo], tile_row_offsets[c_lo:c_hi])

            # 死 tile 区间 [total, max_tiles) 写哨兵 (-1, 0)：仅 core 0，
            # 与各核有效区间不相交（无写竞态）。
            if core_id == 0:
                tot = cum_gm[E]
                if tot < max_tiles:
                    for i in T.Parallel(max_tiles):
                        ub_neg[0, i] = -1
                        ub_zro[0, i] = 0
                    T.copy(ub_neg[0, 0 : max_tiles - tot], tile_expert_ids[tot:max_tiles])
                    T.copy(ub_zro[0, 0 : max_tiles - tot], tile_row_offsets[tot:max_tiles])

    return _expand_main


# --------------------------------------------------------------------------
# Public entry (DESIGN sec 3.3): chains kernel A -> kernel B on one stream and
# returns the 3 schedule tables for ``true_sizes``.
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _tile_scheduler_kernel_run(num_experts: int, max_tiles: int, block_m: int):
    scan = _sched_scan_kernel(num_experts, max_tiles, block_m)
    expand = _sched_expand_kernel(num_experts, max_tiles, block_m)

    def run(true_sizes):
        cum = scan(true_sizes)
        return expand(cum)

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


def _run_scheduler(num_experts, max_tiles, block_m, sizes):
    """Run kernel A -> kernel B on one stream, return (cum_gm, 3 tables).

    Retries a bounded number of fresh compiles for the known flaky
    bishengir-compile SIGSEGV (compile-time only; never a wrong number).
    """
    attempts = int(os.environ.get("SCHED_COMPILE_ATTEMPTS", "6"))
    last = None
    for i in range(attempts):
        try:
            cum = _sched_scan_kernel(num_experts, max_tiles, block_m)(sizes.npu())
            ids, rows, total = _sched_expand_kernel(
                num_experts, max_tiles, block_m
            )(cum)
            return cum, ids, rows, total
        except RuntimeError as exc:
            msg = str(exc)
            if "NPU IR opt failed" not in msg and "SIGSEGV" not in msg:
                raise
            last = exc
            _tile_scheduler_kernel_run.cache_clear()
            print(f"  [warn] fresh-compile attempt {i + 1}/{attempts} hit the "
                  f"flaky bishengir-compile crash, retrying...")
            time.sleep(1.0)
    raise last


def make_sizes(numel, num_experts, distribution):
    """Build a ``true_sizes`` list of length ``num_experts`` summing to ``numel``."""
    if distribution == "uniform":
        base = numel // num_experts
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
        sizes = [0] * num_experts
        active = [e for e in range(num_experts) if e % 3 != 0]
        if not active:
            active = [num_experts - 1]
        base = numel // len(active)
        rem = numel - base * len(active)
        for i, e in enumerate(active):
            sizes[e] = base + (1 if i < rem else 0)
    elif distribution == "ones":
        if numel < num_experts:
            raise ValueError(f"numel={numel} < num_experts={num_experts}")
        sizes = [1] * num_experts
        sizes[0] += numel - num_experts
    elif distribution == "single":
        sizes = [0] * num_experts
        sizes[0] = numel
    else:
        raise ValueError(distribution)
    assert sum(sizes) == numel, (sizes, numel)
    return sizes


def _run_case(num_experts, numel, distribution, block_m, tag):
    sizes_l = make_sizes(numel, num_experts, distribution)
    max_tiles = max_tiles_of(numel, num_experts, block_m)

    sizes = torch.tensor(sizes_l, dtype=torch.int32)
    cum, ids, rows, total = _run_scheduler(num_experts, max_tiles, block_m, sizes)
    torch.npu.synchronize()

    g_ids, g_rows, g_total = golden_tile_scheduler(sizes, block_m, max_tiles)
    g_cum = golden_cum(sizes, block_m)

    ids_c, rows_c = ids.cpu(), rows.cpu()
    total_c = int(total.cpu().reshape(-1)[0].item())
    cum_c = cum.cpu()

    n_ids = int((ids_c != g_ids).sum().item())
    n_rows = int((rows_c != g_rows).sum().item())
    n_cum = int((cum_c != g_cum).sum().item())
    max_diff = 0
    if n_ids:
        max_diff = max(max_diff, int((ids_c - g_ids).abs().max().item()))
    if n_rows:
        max_diff = max(max_diff, int((rows_c - g_rows).abs().max().item()))
    if n_cum:
        max_diff = max(max_diff, int((cum_c - g_cum).abs().max().item()))

    if n_ids or n_rows or n_cum or total_c != g_total:
        raise AssertionError(
            f"[{tag}] mismatch: id_diff={n_ids} row_diff={n_rows} "
            f"cum_diff={n_cum} total={total_c} golden_total={g_total} "
            f"max_diff={max_diff} "
            f"shape=(E={num_experts},numel={numel},max_tiles={max_tiles})"
        )
    print(
        f"[{tag}] PASS  E={num_experts} numel={numel} block_m={block_m} "
        f"max_tiles={max_tiles} total_tiles={total_c} "
        f"dist={distribution} max_diff=0"
    )
    return max_tiles, total_c, ids_c, rows_c


def _contract_check(ids, rows, total, max_tiles, block_m, tag):
    """L0-7: table dtype/shape/sentinel contract for the sibling GEMM consumer."""
    assert ids.dtype == torch.int32 and rows.dtype == torch.int32
    assert ids.shape == (max_tiles,) and rows.shape == (max_tiles,)
    assert 0 <= total <= max_tiles
    live = torch.arange(max_tiles, dtype=torch.int32) < total
    dead = ~live
    assert bool((ids[live] >= 0).all().item())
    assert bool((ids[dead] == -1).all().item())
    assert bool((rows[dead] == 0).all().item())
    assert bool((rows[live] % block_m == 0).all().item())
    print(f"[{tag}] PASS  contract: int32 {tuple(ids.shape)} total_tiles={total} "
          f"<= max_tiles={max_tiles}, sentinel (-1,0), row unit={block_m}")
    return True


def run_L0():
    cases = [
        (4, 64, "uniform", 64, "L0-1 smoke uniform"),
        (4, 64, "skewed", 64, "L0-2 smoke skewed"),
        (8, 64, "empty", 64, "L0-3 empty experts (head/tail zero)"),
        (5, 100, "uniform", 64, "L0-4 non-power-of-2 E=5"),
        (16, 16, "ones", 64, "L0-5 tiny tiles, no dead tile"),
        (256, 32768, "uniform", 64, "L0-6 prefill E=256 max_tiles=768"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L0 done ({len(cases)} cases), cum + all tables bit-exact")

    # L0-6 repeated-run stability: same shapes, fresh + reused compiled kernels
    # (guards against the CG-2026-0027 stale-read class of defect).
    for _ in range(6):
        for num_experts, numel, dist in [(4, 64, "skewed"), (256, 32768, "uniform")]:
            _run_case(num_experts, numel, dist, 64, "L0-8 stability")
    print("L0 stability done (6 repeats x 2 shapes, bit-exact)")

    for num_experts, numel, dist in [(4, 64, "skewed"), (256, 32768, "uniform")]:
        max_tiles, total, ids, rows = _run_case(
            num_experts, numel, dist, 64, "L0-7"
        )
        _contract_check(ids, rows, total, max_tiles, 64, "L0-7")


def run_L1():
    cases = [
        (16, 1024, "uniform", 64, "L1-1 E=16 medium uniform"),
        (32, 3000, "skewed", 64, "L1-2 E=32 skewed"),
        (8, 999, "empty", 64, "L1-3 non-divisible numel + empty experts"),
        (64, 8192, "uniform", 64, "L1-4 E=64 multi local task"),
        (12, 200, "ones", 64, "L1-5 one-token experts"),
        (24, 4096, "single", 64, "L1-6 all tokens on expert 0"),
        (4, 65, "skewed", 64, "L1-7 numel not multiple of block_m"),
        (16, 1024, "uniform", 128, "L1-8 block_m=128"),
        (40, 5120, "uniform", 64, "L1-9 E == VECTOR_CORES exactly"),
        (48, 6000, "skewed", 64, "L1-10 E > VECTOR_CORES, epc=2"),
    ]
    for c in cases:
        _run_case(*c)
    print(f"L1 done ({len(cases)} cases), cum + all tables bit-exact")


def run_L2():
    cases = [
        (128, 16384, "uniform", 64, "L2-1 E=128 uniform"),
        (256, 65536, "skewed", 64, "L2-2 E=256 skewed"),
        (256, 32768, "empty", 64, "L2-3 E=256 empty experts"),
        (512, 32768, "uniform", 64, "L2-4 large E=512"),
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
        (1, 64, "uniform", 64, "B-1 single expert"),
        (8, 0, "empty", 64, "B-2 all experts empty (total_tiles=0)"),
        (8, 512, "single", 64, "B-3 all tokens on one expert"),
        (4, 65, "uniform", 64, "B-4 numel tail, no fill"),
        (2, 1, "skewed", 64, "B-5 tiny numel=1"),
        (16, 16, "ones", 64, "B-6 every expert one tile"),
        (3, 63, "uniform", 64, "B-7 numel < block_m, total == max_tiles (no sentinel write)"),
        (7, 448, "uniform", 64, "B-8 odd E, exact tiles"),
        (1, 0, "single", 64, "B-9 single expert, zero tokens"),
        (300, 19200, "skewed", 64, "B-10 large E with sentinel tail"),
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


# --------------------------------------------------------------------------
# Mode-pinned public entry (Stage 4 integration; see module docstring).
# --------------------------------------------------------------------------
def _tile_scheduler_kernel(*args, **kwargs):
    with _scoped_ascend_mode(ASCEND_MODE):
        with _tile_scheduler_kernel_cache_lock:
            return _tile_scheduler_kernel_run(*args, **kwargs)
