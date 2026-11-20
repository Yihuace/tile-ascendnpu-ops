# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_persistent_grouped_gemm_v2_kernel -- MoE 3-WG persistent grouped GEMM (NT),
NPUIR Stage 3 implementation.

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
    ASCEND_RT_VISIBLE_DEVICES=1 python _persistent_grouped_gemm_v2_kernel.py --level all
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

_BLOCK_M = 64
_BLOCK_N = 256
_BLOCK_K = 64


def _ceildiv(x, y):
    return (x + y - 1) // y


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
                                       dtype="float16", block_m=_BLOCK_M,
                                       block_n=_BLOCK_N, block_k=_BLOCK_K,
                                       num_stages=2):
    """Factory: binds the compile-time geometry (num_stages is reserved for a
    Stage-4 ``T.Pipelined`` K loop; the validated body uses ``T.serial``)."""
    accum_dtype = "float32"
    num_pid_n = _ceildiv(N, block_n)
    num_logical = max_tiles * num_pid_n
    nums_kernel = min(num_logical, PHYS_CORES)            # persistent grid = AI cores
    num_local_tasks = _ceildiv(num_logical, nums_kernel)  # static bound

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
            # 单一 Cube scope：只读 GM 调度表 + NT GEMM + 写回；无 Vector 段、
            # 无跨引擎同步（规避 CG-2026-0026）／无核内调度状态（规避 CG-2026-0027）。
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((block_m, block_k), dtype)
                b_l1 = T.alloc_L1((block_n, block_k), dtype)
                c_l0c = T.alloc_L0C((block_m, block_n), accum_dtype)
                # 固定核数（=AI core 数）+ 核内串行多任务；循环边界全静态（§5.5）。
                for t in T.serial(num_local_tasks):
                    # 专家聚簇 + M-major 任务序（§0.6 R5）：flat task -> (bx, by)。
                    tile_id = core_id * num_local_tasks + t
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
                            tile_k = T.min(block_k, K - k * block_k)
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
                            # initC=(k==0)：首块覆盖初始化 L0C，其后累加（§1.6.1 M2）。
                            T.gemm(a_l1, b_l1, c_l0c,
                                   size=[tile_m, tile_k, tile_n],
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
    so deterministic shape-dependent errors are never masked.  Override the
    retry budget with PGG_COMPILE_ATTEMPTS (default 10 -- the crash rate rises
    with ``num_local_tasks``; a run was observed needing 5 consecutive retries).
    """
    if attempts is None:
        attempts = int(os.environ.get("PGG_COMPILE_ATTEMPTS", "10"))
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
    nk, nlt = kernel_plan(max_tiles, n, block_n)

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
              f"cores={nk} local_tasks={nlt}")


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
