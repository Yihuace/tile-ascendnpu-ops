# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""MoeGroupedGemmNopadFwdOp -- MoE no-pad grouped GEMM (NT), Stage 4 final drop-in.

Round 12 (2026-10-26) re-attack notes
-------------------------------------
* This file is the gate-contract artifact: it carries the tuned GEMM body and
  re-exports the sibling scheduler factory, so a single import adopts the
  tuned pair.  `_tile_scheduler_kernel.py` keeps its Stage 4 name/contents.
* The kernel body is unchanged from round 11 (r10_border): every round-12
  structural candidate was measured and rejected or blocked -- see
  `opt_log.md` round 12.  Net: nothing beat the round-11 winner on this
  toolchain.
* TOOLCHAIN CHANGED between round 11 and round 12 (CANN 8.5.2 -> 9.0.0,
  tilelang 6c987c0a -> 39a8cc68).  The same artifact is measurably slower on
  the new toolchain (prefill-gate-up 15.95 -> 18.18 ms, decode-down
  5.68 -> 6.00 ms); all round-12 numbers below and in opt_log.md are measured
  on the new toolchain.

Baseline: `_moe_grouped_gemm_kernel.py` (Stage 3, design rev 2) --
          examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md
Tuning log: `perf_opt/opt_log.md`; structured records: `perf_opt/perf_records.jsonl`.

Semantics are unchanged: for every expert e in [0, E), with s_e true rows at
true_offsets[e] in A, C[o_e + r, :] = A[o_e + r, :] @ B[e].T (fp32 accumulate,
cast back).  The single ``T.Scope("Cube")`` structure, the GM schedule-table
contract and the dynamic-extent no-pad trimming are all preserved verbatim.

Stage 4 changes (each measured with ``msprof op`` on the four tune workloads)
---------------------------------------------------------------------------
1. ``block_k`` 64 -> 256.  The old 64 (=128 B of bf16) made every GM read a
   128 B fragment of a 4-14 KB-strided row; L1 fills quadrupled per useful byte
   (msprof reported only 1/4 of the true GM traffic, i.e. it was not even
   measurable) and MTE2 stalled.  256 gives 512 B contiguous per row.
2. ``T.serial`` -> ``T.Pipelined(..., num_stages=2)`` **with both L1 buffer
   planners re-enabled** (``TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`` and
   ``NPUIR_ENABLE_AUTO_MULTI_BUFFER``).  With the planners off the pipeline
   pragma is inert (measured: identical latency); with them on the a_l1/b_l1
   buffers are double-versioned, MTE2 reaches 100 % busy and the latency drops
   ~20-30 % on every shape.  The Stage 3 note that these must stay disabled
   applies to the Stage 3 *single-buffer* body only.
3. Tile -> core mapping ``core_id * num_local_tasks + t`` -> strided
   ``t * nums_kernel + core_id``.  The old mapping gave every core a
   *contiguous* block of the logical (M-tile, N-tile) grid, so the dead-tile
   suffix (``bx >= total_tiles``) of low-expert shapes idled whole cores
   (decode: 4/20 idle, prefill: 10/20 idle).  The strided mapping spreads the
   live range over all 20 cores (the dead suffix stays a contiguous *t* tail,
   so no per-iteration interleaving is introduced).
4. ``block_m`` 64 -> 128.  An expert run shorter than block_m is trimmed by
   ``tile_m`` (no-pad contract), but the Cube still issues the *static*
   block_m rows of MMA and B is re-read once per M-tile of an expert.  For the
   prefill shapes (128 tokens/expert) block_m=128 makes B single-pass; the
   decode shapes keep one M-tile per expert either way.
5. The B copy is issued before the A copy inside the K loop (measured ~4 % on
   the largest shape; A and B traffic are otherwise interleaved unfavourably).

Correctness is unchanged and re-validated (`--level all`: L0/L1/L2/boundary,
plus a device-scheduler end-to-end check against the golden).
"""

"""MoE no-pad grouped GEMM (NT) -- NPUIR Stage 3 implementation (design rev 2).

Design: examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md
        (revision 2, REVIEW.md verdict = pass)

Semantics
---------
For every expert e in [0, E), with s_e = true_sizes[e] tight rows at
true_offsets[e] in A:

    C[o_e + r, :] = A[o_e + r, :] @ B[e].T          r in [0, s_e)

i.e. a per-expert NT matmul with fp32 accumulation and a cast back to the
input dtype, on a tight (no-pad) row layout.  C has shape (numel, N).

Structure (DESIGN rev 2 -- sec 0.6 R1 / sec 3.3, single Cube scope)
-------------------------------------------------------------------
The tile schedule is NOT computed inside this kernel.  It is materialised in
GM by a separate producer kernel (`_tile_scheduler_kernel`, sibling unit) and
consumed here as three int32 tables:

    tile_expert_ids[max_tiles]    expert id per M-tile (dead tile = -1)
    tile_row_offsets[max_tiles]   in-expert row offset (already * block_m)
    total_tiles[1]                number of live M-tiles (stays on device)

This kernel therefore has a SINGLE ``T.Scope("Cube")`` and only:

    read GM tables/scalars -> dynamic-extent T.copy (GM->L1, ND2NZ) ->
    T.gemm(size=[tile_m, tile_k, tile_n], b_transpose=True, initC=) ->
    T.store_fixpipe([tile_m, tile_n], enable_nz2nd=True)

No Vector scope and no ``T.sync_block_set/wait`` anywhere: the current
toolchain cannot compile any in-block Vector<->Cube handoff (CG-2026-0026,
`syn_instr_mode` attribute mismatch, reproduced by the repo's own
``examples/mixcv/mixcv_mixkernel.py``), and the Developer persistent+gemm
combination segfaults at lowering (CG-2026-0010).  Kernel-to-kernel ordering
is provided by the same-stream sequential launch, with no CPU synchronisation.

Stage 3 validates this GEMM core against a host-built schedule fixture (the
sibling producer kernel is not implemented yet); the fixture only supplies
auxiliary table inputs and never mutates A/B (ascend-constraints sec 4).
Validated base evidence: ``repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py``
``positive_control()`` (max_diff 9.537e-07).

Toolchain note: the bundled ``bishengir-compile`` intermittently SIGSEGVs on
the IR form that combines the serial task loop with a dynamic column offset
(``n_start = by * block_n``, i.e. ``num_pid_n > 1`` together with
``num_local_tasks > 1``).  The same IR compiles when retried, so the harness
raises the stack limit and retries a bounded number of times (see
``_raise_stack_limit`` / ``_run_gemm``); accuracy results are unaffected
(this is a compile-time crash, never a wrong number).  Repro:
``repro/TRAP-BISHENGIR-FLAKY-SIGSEGV.py``.
"""

import argparse
import os
import resource
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

from tileops.utils import scoped_ascend_mode as _scoped_ascend_mode

# Stage 3 programming mode actually implemented/validated: Expert single Cube scope.
ASCEND_MODE = "Expert"

#: Stage 4 tuned launch geometry -- single source of truth.
#:
#: The wrapper passes block_m/block_n/block_k/num_stages explicitly to the
#: factory, so adopting the perf_opt import is NOT sufficient on its own:
#: ``MoeGroupedGemmNopadKernel.default_config`` must return
#: ``dict(TUNED_DEFAULT_CONFIG)`` (see perf_opt/workload_inventory.json ->
#: ``wrapper_pairing_required``).  block_m is also the shared constant with the
#: sibling scheduler (max_tiles = numel // block_m + E), so both call sites and
#: the schedule tables must use the same value.
TUNED_DEFAULT_CONFIG = {
    "block_m": 128,
    "block_n": 256,
    "block_k": 256,
    "num_stages": 2,
}

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Physical AICore count, measured on this box (Ascend910B2C):
#   NPUUtils.get().get_aicore_num() -> 20   (DESIGN sec 5.5)
PHYS_CORES = 20


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source-op semantics, torch API called explicitly)
# --------------------------------------------------------------------------
def golden_moe_grouped_gemm_nopad(a, b, true_sizes, true_offsets):
    """Per-expert NT matmul over the tight (no-pad) permute layout.

    a: (numel, K); b: (E, N, K); true_sizes: (E,) int32; true_offsets: (E,) int32
    Returns c: (numel, N), same dtype as ``a``.
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
# Plan helpers (shared by the kernel factory and the standalone test harness)
# --------------------------------------------------------------------------
def _ceildiv(x, y):
    return (x + y - 1) // y


def max_tiles_of(numel, num_experts, block_m=64):
    """Compile-time M-tile upper bound: DESIGN sec 4.1 contract (shared constant)."""
    return numel // block_m + num_experts


def kernel_plan(numel, max_tiles, num_experts, n, block_m=64, block_n=256):
    """Return the compile-time launch geometry (nums_kernel, num_local_tasks).

    nums_kernel = min(logical tiles, PHYS_CORES); num_local_tasks = static
    in-core serial bound (DESIGN sec 5.5).
    """
    num_pid_n = _ceildiv(n, block_n)
    num_logical = max_tiles * num_pid_n
    nums_kernel = min(num_logical, PHYS_CORES)
    num_local_tasks = _ceildiv(num_logical, nums_kernel)
    return nums_kernel, num_local_tasks


def build_schedule(sizes_l, block_m=64):
    """Host-built schedule fixture (equivalent of `_tile_scheduler_kernel`).

    Returns (tile_expert_ids, tile_row_offsets, total_tiles) as python lists.
    Contract (DESIGN sec 4.1): experts are clustered in order, each producing
    ceil(s_e/block_m) M-tiles; unused slots use the (-1, 0) sentinel;
    tile_row_offsets is in ROWS (already multiplied by block_m).
    """
    ids = []
    rows = []
    for e, s in enumerate(sizes_l):
        n_tiles_e = (s + block_m - 1) // block_m  # ceil; s >= 0
        for j in range(n_tiles_e):
            ids.append(e)
            rows.append(j * block_m)
    total = len(ids)
    max_tiles = max_tiles_of(sum(sizes_l), len(sizes_l), block_m)
    assert total <= max_tiles, f"schedule overflow: total={total} > max_tiles={max_tiles}"
    while len(ids) < max_tiles:
        ids.append(-1)
        rows.append(0)
    return ids, rows, total


# Expert form: the explicit L1/L0C allocations must not be re-scoped by the
# buffer planners (DESIGN sec 7.3 / PL-1.16).
_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: True,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
}


# --------------------------------------------------------------------------
# Kernel (DESIGN sec 4.7 factory signature; sec 3.3 Expert single-Cube body)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir", pass_configs=_PASS_CFG)
def _moe_grouped_gemm_kernel_impl(numel, max_tiles, num_experts, N, K, dtype="float16",
                             block_m=TUNED_DEFAULT_CONFIG["block_m"],
                             block_n=TUNED_DEFAULT_CONFIG["block_n"],
                             block_k=TUNED_DEFAULT_CONFIG["block_k"],
                             num_stages=TUNED_DEFAULT_CONFIG["num_stages"]):
    accum_dtype = "float32"
    num_pid_n = _ceildiv(N, block_n)
    num_logical = max_tiles * num_pid_n
    nums_kernel = min(num_logical, PHYS_CORES)
    num_local_tasks = _ceildiv(num_logical, nums_kernel)

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
        with T.Kernel(nums_kernel, is_npu=True) as (core_id, subid):
            # 单一 Cube scope：只读 GM 调度表 + NT GEMM + 写回；无 Vector 段、
            # 无跨引擎同步（规避 CG-2026-0026）。
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((block_m, block_k), dtype)
                b_l1 = T.alloc_L1((block_n, block_k), dtype)
                c_l0c = T.alloc_L0C((block_m, block_n), accum_dtype)
                # 固定核数 + 核内串行多任务（循环边界静态，DESIGN sec 5.5）。
                for t in T.serial(num_local_tasks):
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
                        # 动态 extent：M/N 尾块精确裁剪（no-pad 关键，DESIGN sec 5.4）。
                        tile_m = T.min(block_m, true_sizes[expert_id] - row)
                        tile_n = T.min(block_n, N - n_start)
                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            tile_k = T.min(block_k, K - k * block_k)
                            # 动态 extent 取数：GM->L1（隐式 ND2NZ）。
                            T.copy(
                                B[expert_id, n_start:n_start + tile_n,
                                  k * block_k:k * block_k + tile_k],
                                b_l1[0:tile_n, 0:tile_k],
                            )
                            T.copy(
                                A[m_start:m_start + tile_m,
                                  k * block_k:k * block_k + tile_k],
                                a_l1[0:tile_m, 0:tile_k],
                            )
                            # size=[tile_m, tile_k, tile_n] 把归约范围限定到有效
                            # tile_k（正确性必需：否则缓冲 [tile_k, block_k) 的陈旧
                            # 尾值会被按静态 block_k 计入累加，DESIGN sec 3.3 注 1）。
                            T.gemm(a_l1, b_l1, c_l0c,
                                   size=[tile_m, tile_k, tile_n],
                                   b_transpose=True, initC=(k == 0))
                        # 动态 extent 写回：只写有效 [tile_m, tile_n]，杜绝越界写
                        # 坏下一专家行（DESIGN sec 5.4）。
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_l0c, C[m_start, n_start],
                                            size=[tile_m, tile_n],
                                            enable_nz2nd=True)

    return main


# --------------------------------------------------------------------------
# Mode-pinned public factory (Stage 4 integration).
#
# The wrapper's perf_opt switch imports THIS module directly
# (``from .moe_grouped_gemm_nopad_kernel.perf_opt._moe_grouped_gemm_kernel
# import _moe_grouped_gemm_kernel``), which bypasses the per-function
# ``_pin_ascend_mode`` re-export in the package ``__init__``.  Pinning the
# compile-time mode here keeps the Expert/Developer split of the two sibling
# units working through either import path.
# --------------------------------------------------------------------------
def _moe_grouped_gemm_kernel(*args, **kwargs):
    with _scoped_ascend_mode(ASCEND_MODE):
        return _moe_grouped_gemm_kernel_impl(*args, **kwargs)


# --------------------------------------------------------------------------
# Reference inputs (mirrors examples/TileOPs work-load distributions)
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
        # Some experts get zero tokens -> cnt=0 -> no tile (dead-by-construction).
        sizes = [0] * num_experts
        active = [e for e in range(num_experts) if e % 3 != 0]
        if not active:
            active = [num_experts - 1]
        base = numel // len(active)
        rem = numel - base * len(active)
        for i, e in enumerate(active):
            sizes[e] = base + (1 if i < rem else 0)
    else:
        raise ValueError(distribution)
    offsets = [0] * num_experts
    for e in range(1, num_experts):
        offsets[e] = offsets[e - 1] + sizes[e - 1]
    assert sum(sizes) == numel, (sizes, numel)
    return sizes, offsets


def _raise_stack_limit():
    """Raise RLIMIT_STACK to the hard limit (a mitigation for a toolchain bug).

    The bundled bishengir-compile walks the nested ``scf`` regions (task loop ->
    dead-tile guard -> K loop) with deep recursion and dies with SIGSEGV, whose
    symbol-less backtrace repeats the same PC (recursion).  A larger stack
    reduces the crash rate (an 8 MB default is clearly marginal) but does NOT
    eliminate it -- the reliable net is the bounded retry in ``_run_gemm``.
    Child processes inherit the raised limit across exec.
    """
    try:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ValueError, OSError, AttributeError):
        pass


_raise_stack_limit()


# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------
def _run_gemm(build, call_args, attempts=None):
    """Invoke the JIT kernel, retrying on flaky bishengir-compile SIGSEGVs.

    Primary robustness comes from ``_raise_stack_limit()`` above (the compiler
    stack overflow on the nested-loop IR form); this retry is a safety net in
    case the hard RLIMIT_STACK is itself capped.  The retry rebuilds the
    factory so a failed compile is not reused; it does NOT mask deterministic
    shape-dependent errors.  Override with MOE_COMPILE_ATTEMPTS (default 6).
    """
    if attempts is None:
        attempts = int(os.environ.get("MOE_COMPILE_ATTEMPTS", "6"))
    last = None
    for i in range(attempts):
        try:
            return build()(*call_args)
        except RuntimeError as exc:  # noqa: PERF203
            msg = str(exc)
            if "NPU IR opt failed" not in msg and "SIGSEGV" not in msg:
                raise
            last = exc
            print(f"  [warn] fresh-compile attempt {i + 1}/{attempts} hit the "
                  f"flaky bishengir-compile crash, retrying...")
            time.sleep(1.0)
    raise last


def run_case(numel, num_experts, n, k, dtype, distribution, tag):
    torch.manual_seed(42)
    tdt = _DTYPE_MAP[dtype]
    sizes_l, offsets_l = make_sizes_offsets(numel, num_experts, distribution)
    block_m = TUNED_DEFAULT_CONFIG["block_m"]
    block_n = TUNED_DEFAULT_CONFIG["block_n"]
    max_tiles = max_tiles_of(numel, num_experts, block_m)
    teids_l, troffs_l, total = build_schedule(sizes_l, block_m)
    sizes = torch.tensor(sizes_l, dtype=torch.int32)
    offsets = torch.tensor(offsets_l, dtype=torch.int32)
    a = (torch.randn(numel, k, dtype=tdt) * 0.02)
    b = (torch.randn(num_experts, n, k, dtype=tdt) * 0.02)

    nk, nlt = kernel_plan(numel, max_tiles, num_experts, n, block_m, block_n)
    c = _run_gemm(
        lambda: _moe_grouped_gemm_kernel(numel, max_tiles, num_experts, n, k, dtype),
        (
            a.npu(), b.npu(),
            torch.tensor(teids_l, dtype=torch.int32).npu(),
            torch.tensor(troffs_l, dtype=torch.int32).npu(),
            offsets.npu(), sizes.npu(),
            torch.tensor([total], dtype=torch.int32).npu(),
        ),
    )
    torch.npu.synchronize()

    ref = golden_moe_grouped_gemm_nopad(a, b, sizes, offsets)
    cc = c.cpu().float()
    rr = ref.float()
    max_diff = (cc - rr).abs().max().item()
    torch.testing.assert_close(cc, rr, atol=1e-2, rtol=1e-2)
    print(f"[{tag}] PASS  shape=({numel},{num_experts},{n},{k}) dtype={dtype} "
          f"dist={distribution} max_tiles={max_tiles} total_tiles={total} "
          f"cores={nk} local_tasks={nlt} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    cases = [
        (64, 4, 128, 64, "float16", "uniform", "L0-1 smoke fp16 uniform, N<block_n"),
        (64, 4, 128, 64, "float16", "skewed", "L0-2 smoke fp16 skewed, dead tiles"),
        (128, 8, 256, 96, "bfloat16", "uniform", "L0-3 K non-aligned (96)"),
        (233, 5, 130, 70, "bfloat16", "skewed", "L0-4 odd shape skewed"),
        (256, 6, 192, 64, "float16", "empty", "L0-5 empty experts (cnt=0)"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L0 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L1():
    cases = [
        (256, 8, 512, 128, "float16", "uniform", "L1-1 medium uniform, 2 pid_n"),
        (128, 16, 384, 192, "bfloat16", "uniform", "L1-2 E=16"),
        (300, 12, 200, 64, "float16", "skewed", "L1-3 skewed non-divisible"),
        (384, 10, 300, 100, "bfloat16", "empty", "L1-4 empty experts + K tail"),
        (512, 4, 768, 256, "float16", "uniform", "L1-5 multi local task"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L1 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L2():
    cases = [
        (4096, 64, 512, 256, "bfloat16", "uniform", "L2-1 large uniform"),
        (8192, 128, 1024, 512, "bfloat16", "skewed", "L2-2 large skewed"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L2 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_boundary():
    cases = [
        (64, 4, 16, 32, "float16", "uniform", "B-1 tiny N, K"),
        (70, 7, 70, 40, "bfloat16", "skewed", "B-2 all non-divisible"),
        (33, 4, 128, 33, "float16", "uniform", "B-3 odd numel/K, max_tiles small"),
        (128, 8, 48, 16, "float16", "empty", "B-4 empty experts, tiny N/K"),
        (64, 1, 64, 64, "bfloat16", "uniform", "B-5 single expert"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"Boundary done ({len(cases)} cases), worst max_diff={worst:.3e}")


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
        run_L2()
        run_boundary()


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Sibling scheduler re-export (gate contract for the final artifact).
#
# The tuned pair is `MoeGroupedGemmNopadFwdOp.py` (this module) +
# `_tile_scheduler_kernel.py`.  The relative import is deliberately lazy /
# guarded: the perf harness loads this file standalone via
# `importlib.util.spec_from_file_location` (no package context), where a
# top-level relative import would raise.
# --------------------------------------------------------------------------
try:  # package context (wrapper / pytest import path)
    from ._tile_scheduler_kernel import _tile_scheduler_kernel  # noqa: F401
except ImportError:  # standalone file load (perf_opt/prof_entry.py)
    _tile_scheduler_kernel = None

__all__ = [
    "_moe_grouped_gemm_kernel",
    "_tile_scheduler_kernel",
    "TUNED_DEFAULT_CONFIG",
    "_run_gemm",
    "max_tiles_of",
    "kernel_plan",
    "build_schedule",
    "golden_moe_grouped_gemm_nopad",
]
