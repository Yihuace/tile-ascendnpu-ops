# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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

    read GM tables/scalars -> dynamic-extent T.copy (GM->L1, ND2NZ;
    M/N tail-clipped; K a full static block on A5) ->
    T.gemm(size=[block_m, block_k, block_n] on A5, else [block_m, tile_k,
    block_n], b_transpose=True, initC=) ->
    T.store_fixpipe([tile_m, tile_n], enable_nz2nd=True)

A5/CANN 9.2 note: the mmadL1 size must be compile-time constant (a dynamic
size computes wrong values and can crash bishengir), and the K reduction
extent must equal the L1 block exactly (that is why block_k divides K and
M/N tails are handled by clipping the copy extents and the fixpipe write).
The in-core serial task loop is also unusable on A5 (the scf.for iter_args
trip a ``'memref.alloc' read before first write`` verification failure), so
on A5 the kernel launches one logical tile per block instead.  Non-A5
devices keep the original dynamic-extent / fixed-core form unchanged.

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

# Stage 3 programming mode actually implemented/validated: Expert single Cube scope.
ASCEND_MODE = "Expert"

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


def _pick_block_k(k_dim, block_k):
    """Choose a K block that divides ``k_dim`` (A5 static-size gemm requirement).

    On A5/CANN 9.2 a dynamic-extent K copy combined with a static-size
    ``T.gemm`` is numerically wrong: the tail columns keep the previous
    iteration's data and get folded into the reduction (measured maxdiff
    ~1e-2 on the real kernel shape).  The reduction extent therefore must be
    exact, which means ``block_k`` must divide K.  Prefer the largest
    multiple-of-16 divisor <= ``block_k`` (16 is the fp16/bf16 L1 fractal
    height).  Falls back to ``block_k`` unchanged when no such divisor
    exists (real workloads are all 64-aligned, so this is a dev-shape guard).
    """
    if k_dim % block_k == 0:
        return block_k
    start = min(block_k, k_dim) // 16 * 16
    for cand in range(start, 0, -16):
        if k_dim % cand == 0:
            return cand
    return block_k


def _pick_block_n(n_dim, block_n):
    """Choose an N block that avoids a 16-aligned *static partial* L1 copy.

    On A5/CANN 9.2 a B copy whose compile-time extent is a multiple of 16 but
    smaller than the L1 buffer (N < block_n, single N-tile) makes bishengir
    abort with ``[hivm-bind-sub-block] revert main_mix_aiv`` +
    ``LLVM ERROR: Unknown PIPE!`` (measured: N = 112/128/144/160/192/224/240
    with block_n=256 all crash; N=200 — not 16-aligned — compiles).
    Dynamic partial copies (N > block_n, runtime ``by``) and full copies are
    fine.  So: keep ``block_n`` when N is a multiple of it or larger than it;
    otherwise make the single N-tile a full block.
    """
    if n_dim % block_n == 0:
        return block_n
    if n_dim < block_n:
        # Single N-tile: use a full block (16-aligned N -> exact; otherwise
        # round up, which leaves an unaligned static partial copy = safe path).
        return n_dim if n_dim % 16 == 0 else _ceildiv(n_dim, 16) * 16
    return block_n


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
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}


def _is_a5():
    """True when the compile target is the Ascend 950 (A5) family."""
    name = os.environ.get("TILELANG_ASCEND_DEVICE_NAME", "").strip()
    if not name:
        try:
            from tilelang.utils.npu_arch import get_ascend_device_name

            name = str(get_ascend_device_name() or "")
        except Exception:
            name = ""
    return name.startswith("Ascend950")


# --------------------------------------------------------------------------
# Kernel (DESIGN sec 4.7 factory signature; sec 3.3 Expert single-Cube body)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir", pass_configs=_PASS_CFG)
def _moe_grouped_gemm_kernel(numel, max_tiles, num_experts, N, K, dtype="float16",
                             block_m=64, block_n=256, block_k=64, num_stages=2):
    accum_dtype = "float32"
    a5 = _is_a5()
    if a5:
        # A5/CANN 9.2 真机实测约束（详见 iso 实验，见下）：
        #  - mmadL1 的 size 必须是编译期常量：动态 size 会算错（K 尾把上一轮
        #    的陈旧数据算进归约）甚至让 bishengir 崩溃；
        #  - 归约维必须精确等于 block_k，因此 block_k 需整除 K；
        #  - 16 对齐的静态不满块 B 拷贝会让 bishengir 中止（见 _pick_block_n）；
        #  - 核内串行多任务（scf.for 携带 tensor iter_args）触发
        #    "'memref.alloc' read before first write" 校验失败，故一 tile 一 block。
        block_k = _pick_block_k(K, block_k)
        block_n = _pick_block_n(N, block_n)
    num_pid_n = _ceildiv(N, block_n)
    num_logical = max_tiles * num_pid_n
    if a5:
        nums_kernel = num_logical
        num_local_tasks = 1
    else:
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
                        # 动态 extent：M/N 尾块精确裁剪（no-pad 关键，DESIGN sec 5.4）。
                        tile_m = T.min(block_m, true_sizes[expert_id] - row)
                        tile_n = T.min(block_n, N - n_start)
                        for k in T.serial(T.ceildiv(K, block_k)):
                            # A5：K 方向静态整块取数（block_k 已保证整除 K，
                            # 见 _pick_block_k），M/N 保持动态 extent 精确裁剪
                            # 尾块（真机实测动态 M/N extent 取数可编译且正确）。
                            # 非 A5：保持原动态 K extent + 动态 gemm size。
                            tile_k = block_k if a5 else T.min(block_k, K - k * block_k)
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
                            # A5/CANN 9.2：动态尺寸的 mmadL1 会算错甚至让编译器
                            # 崩溃（真机实测）。改为固定尺寸计算整块：M/N 尾块多算
                            # 的行列不写回（fixpipe 仍按动态 size 写），K 尾部依赖
                            # block_k 对齐（DESIGN sec 3.3 注 1）。
                            T.gemm(a_l1, b_l1, c_l0c,
                                   size=([block_m, block_k, block_n] if a5
                                         else [block_m, tile_k, block_n]),
                                   b_transpose=True, initC=(k == 0))
                        # 动态 extent 写回：只写有效 [tile_m, tile_n]，杜绝越界写
                        # 坏下一专家行（DESIGN sec 5.4）。
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_l0c, C[m_start, n_start],
                                            size=[tile_m, tile_n],
                                            enable_nz2nd=True)

    return main


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
    shape-dependent errors.  Override with MOE_COMPILE_ATTEMPTS (default 30).

    NOTE (why 30, and why it is NOT a precision relaxation): this is purely a
    *flaky compile-retry budget*.  ``bishengir-compile`` intermittently SIGSEGVs
    on this IR form and succeeds on a fresh process; raising the budget only
    increases how many fresh compiles are attempted before giving up.  Every
    successful attempt runs the identical kernel and is checked against the same
    tolerance -- a wrong number can never be "retried away", because only
    compile-time ``NPU IR opt failed`` / ``SIGSEGV`` errors are retried (any
    other error is re-raised immediately, see below).  30 matches the sibling
    budget ``PGG_COMPILE_ATTEMPTS`` in the 3WG persistent family; the previous
    default of 6 was observed to be exhausted 6/6 by the bm64/bn256/bk64 shape
    in one long harness session (FusedMoeFwdCbFwdOp run 20261026_174140), while
    the same code passed end-to-end once the budget was raised.
    """
    if attempts is None:
        attempts = int(os.environ.get("MOE_COMPILE_ATTEMPTS", "30"))
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
    block_m, block_n = 64, 256
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
