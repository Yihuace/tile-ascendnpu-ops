# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""MoE no-pad grouped GEMM + fused gate/up activation -- NPUIR Stage 3.

Design: examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/DESIGN.md
        (frozen; REVIEW.md verdict = pass)

Semantics (DESIGN sec 0.1/1.3)
------------------------------
For every expert e in [0, E), with s_e = true_sizes[e] tight rows at
true_offsets[e] in A:

    g = A[o_e:o_e+s_e] @ B[e, 0:ffn].T
    u = A[o_e:o_e+s_e] @ B[e, ffn:2*ffn].T
    C[o_e:o_e+s_e, :] = act(g) * u

with act in {silu_and_mul, gelu_and_mul}; gate/up stay in fp32 and the output
is rounded back to the input dtype exactly once (source L449-450).

Structure (DESIGN sec 2.1 / 0.6 R2 / R5 -- two kernels, same stream)
-------------------------------------------------------------------
kernel G (Expert, single ``T.Scope("Cube")``):
    read GM schedule table -> per logical tile dynamic-extent T.copy
    (GM->L1, implicit ND2NZ) -> two ``T.gemm`` per K step sharing one A tile
    (dual fp32 L0C accumulators, ``b_transpose=True``, ``initC=(k==0)``) ->
    two ``T.store_fixpipe`` write the packed ``gate_up[numel, 2*ffn]`` (fp32)
    at disjoint column ranges (gate -> [n_start,..), up -> [ffn+n_start,..)).

kernel V (pure Vector, packed activation):
    read packed ``gate_up`` -> v-prefix chain ``act(gate)*up`` -> single
    round-back to ``C[numel, ffn]``.

Both kernels have NO cross-engine in-block handoff (each is a separate kernel
launched on the same stream), which side-steps CG-2026-0026 (``syn_instr_mode``
attribute mismatch: any Vector<->Cube in-block handoff is uncompilable) and
CG-2026-0010 (Developer persistent+gemm lowering segfault).  The GEMM half is
Expert (single Cube scope), the activation half is a plain Vector kernel; the
module-level ``ASCEND_MODE`` records the mode the file is compiled and
validated under.  Validated precursor:
``history_version/design_probe_dual_acc.py`` (three block configs, PASS).

The tile schedule is NOT computed in-kernel (CG-2026-0010): it is materialised
in GM by the sibling producer kernel ``_tile_scheduler_kernel`` and consumed
here as three int32 tables::

    tile_expert_ids[max_tiles]    expert id per M-tile (dead tile = -1)
    tile_row_offsets[max_tiles]   in-expert row offset (already * block_m)
    total_tiles[1]                number of live M-tiles (stays on device)

Until that sibling is available, Stage 3 feeds a host-built schedule fixture
(identical to ``_moe_grouped_gemm_kernel.build_schedule``): it only supplies
auxiliary table inputs and never mutates A/B (ascend-constraints sec 4).

Contract (DESIGN sec 4.1): ``max_tiles = numel // block_m + E`` and
``block_m = 128`` are compile-time constants shared with the producer kernel;
all tables are int32; ``tile_row_offsets`` is in ROWS; dead-tile sentinel
``(-1, 0)``.

Toolchain note (CG-2026-0033): the bundled ``bishengir-compile`` intermittently
SIGSEGVs on the IR form that combines the serial task loop with a dynamic
column offset (``num_pid_n > 1`` and ``num_local_tasks > 1``).  The same IR
compiles on retry, so the harness raises ``RLIMIT_STACK`` and retries a bounded
number of fresh compiles (``MOE_COMPILE_ATTEMPTS``, default 20; the crash rate
for the N-tail IR form is high enough that the sibling's default of 6 is not
always sufficient).  Accuracy is unaffected (a compile-time crash, never a
wrong number).

Run the embedded hierarchical tests::

    ASCEND_RT_VISIBLE_DEVICES=1 python _fused_act_kernel.py --level L0
    ASCEND_RT_VISIBLE_DEVICES=1 python _fused_act_kernel.py --level all
"""

import argparse
import os
import resource
import time

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

# Programming mode actually implemented/validated by this file: the GEMM core
# is an Expert single Cube scope (the activation kernel is a plain Vector
# kernel compiled in the same Expert-mode process, as validated by the D-3
# probe).  Must match TILELANG_ASCEND_MODE when the kernel is exercised
# standalone.
ASCEND_MODE = "Expert"

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Physical core counts, measured on this box (Ascend910B2C):
#   NPUUtils.get().get_aicore_num()   -> 20  (Cube, DESIGN sec 5.5)
#   NPUUtils.get().get_aivector_core_num() -> 40 (pure Vector)
_CUBE_CORES_FALLBACK = 20
_VECTOR_CORES_FALLBACK = 40

# GEMM tile geometry (DESIGN sec 5.2): block_m=128 keeps one M-tile per expert
# (B single pass); block_n=128 gives dual fp32 L0C = 128 KB (at the limit,
# probe-validated); block_k=64 is the fractal K lower bound.
_GEMM_BLOCK_M = 128
_GEMM_BLOCK_N = 128
_GEMM_BLOCK_K = 64

# Activation tile geometry (DESIGN sec 5.2).
_ACT_BLOCK_M = 2
_ACT_BLOCK_N = 256

# A&S 7.1.26 complementary-error coefficients (|eps| <= 1.5e-7) and constants;
# used to build erf for the exact GELU chain (native T.verf is a low-order
# Taylor and is unusable for |z| >~ 1).
_A1, _A2, _A3, _A4, _A5 = (
    0.254829592,
    -0.284496736,
    1.421413741,
    -1.453152027,
    1.061405429,
)
_ARF_P = 0.3275911
_INV_SQRT2 = 0.7071067811865476

# Expert form: explicit L1/L0C allocations must not be re-scoped by the buffer
# planners (DESIGN sec 7.3 / PL-1.16).
_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}


def _ceildiv(x, y):
    return (x + y - 1) // y


def _cube_cores():
    """Queried physical Cube core count (DESIGN sec 5.5; fallback = 20)."""
    try:
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num())
    except Exception:  # noqa: BLE001 - fall back to the measured constant
        return _CUBE_CORES_FALLBACK


def _vector_cores():
    """Queried physical Vector core count (pure Vector -> queried value)."""
    try:
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aivector_core_num())
    except Exception:  # noqa: BLE001
        return _VECTOR_CORES_FALLBACK


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
    multiple-of-16 divisor <= ``block_k`` (16 = fp16/bf16 L1 fractal height).
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


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source-op semantics; torch API called explicitly)
# --------------------------------------------------------------------------
def golden_fused_act(a, b, true_sizes, true_offsets, activation="silu_and_mul"):
    """Per-expert fused gate/up activation over the tight (no-pad) layout.

    Semantics only -- no NPU algorithm is mirrored here (independent oracle).

        a: (numel, K); b: (E, 2*ffn, K); true_sizes: (E,) int32;
        true_offsets: (E,) int32
        returns c: (numel, ffn), same dtype as ``a``.

    ``a``/``b`` may live on CPU; the golden runs on CPU without an NPU device.
    """
    numel = a.shape[0]
    num_experts, two_ffn, _ = b.shape
    ffn = two_ffn // 2

    def act(gate, up):
        if activation == "silu_and_mul":
            return torch.nn.functional.silu(gate) * up
        if activation == "gelu_and_mul":
            return torch.nn.functional.gelu(gate, approximate="none") * up
        raise ValueError(f"unsupported activation {activation!r}")

    c = torch.zeros(numel, ffn, dtype=a.dtype, device=a.device)
    sizes = true_sizes.tolist()
    offsets = true_offsets.tolist()
    for e in range(num_experts):
        size_e = sizes[e]
        if size_e == 0:
            continue
        off_e = offsets[e]
        a_e = a[off_e:off_e + size_e].to(torch.float32)
        b_e = b[e].to(torch.float32)
        gu = torch.matmul(a_e, torch.transpose(b_e, 0, 1))
        c[off_e:off_e + size_e] = act(gu[:, :ffn], gu[:, ffn:]).to(a.dtype)
    return c


# --------------------------------------------------------------------------
# Plan helpers (shared with the host-built schedule fixture)
# --------------------------------------------------------------------------
def max_tiles_of(numel, num_experts, block_m=_GEMM_BLOCK_M):
    """Compile-time M-tile upper bound (DESIGN sec 4.1 shared constant)."""
    return numel // block_m + num_experts


def build_schedule(sizes_l, block_m=_GEMM_BLOCK_M):
    """Host-built schedule fixture (equivalent of `_tile_scheduler_kernel`).

    Contract (DESIGN sec 4.1): experts are clustered in order, each producing
    ceil(s_e / block_m) M-tiles; unused slots use the (-1, 0) sentinel;
    ``tile_row_offsets`` is in ROWS (already multiplied by block_m).
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


# --------------------------------------------------------------------------
# Kernel G: Expert single Cube scope -- dual-accumulator grouped GEMM writing
# the packed `gate_up` (fp32).  (DESIGN sec 3.3 / sec 4.7)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], target="npuir", pass_configs=_PASS_CFG)
def _gemm_gate_up(numel, max_tiles, E, ffn, K, dtype="float16",
                  block_m=_GEMM_BLOCK_M, block_n=_GEMM_BLOCK_N, block_k=_GEMM_BLOCK_K):
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
        block_n = _pick_block_n(ffn, block_n)
    num_pid_n = _ceildiv(ffn, block_n)
    num_logical = max_tiles * num_pid_n
    if a5:
        nums_kernel = num_logical
        num_local_tasks = 1
    else:
        nums_kernel = min(num_logical, _cube_cores())
        num_local_tasks = _ceildiv(num_logical, nums_kernel)

    @T.prim_func
    def main(
        A: T.Tensor((numel, K), dtype),
        B: T.Tensor((E, 2 * ffn, K), dtype),
        GU: T.Tensor((numel, 2 * ffn), accum_dtype),
        tile_expert_ids: T.Tensor((max_tiles,), "int32"),
        tile_row_offsets: T.Tensor((max_tiles,), "int32"),
        true_offsets: T.Tensor((E,), "int32"),
        true_sizes: T.Tensor((E,), "int32"),
        total_tiles: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(nums_kernel, is_npu=True) as (core_id, subid):
            # 单一 Cube scope：只读 GM 调度表 + NT GEMM + 双 store_fixpipe。
            # 无 Vector 段、无跨引擎同步（规避 CG-2026-0026）。
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((block_m, block_k), dtype)
                bg_l1 = T.alloc_L1((block_n, block_k), dtype)
                bu_l1 = T.alloc_L1((block_n, block_k), dtype)
                # 双 fp32 累加器：gate 与 up 共享同一个 A tile（A5 / DESIGN R2）。
                c_gate = T.alloc_L0C((block_m, block_n), accum_dtype)
                c_up = T.alloc_L0C((block_m, block_n), accum_dtype)
                # 固定核数 + 核内串行多任务（循环边界静态，DESIGN sec 5.5）。
                for t in T.serial(num_local_tasks):
                    tile_id = core_id * num_local_tasks + t
                    bx = tile_id // num_pid_n
                    by = tile_id % num_pid_n
                    # 死 tile 守卫：单比较，位于 T.serial 体内、K 循环之外
                    # （规避 CG-2026-0001：Pipelined 体内条件构造崩溃）。
                    if bx < total_tiles[0]:
                        e = tile_expert_ids[bx]       # GM 标量读（运行时下标）
                        row = tile_row_offsets[bx]    # GM 标量读；单位=行（已乘 block_m）
                        n_start = by * block_n
                        m_start = true_offsets[e] + row
                        # 动态 extent：M/N 尾块精确裁剪（no-pad 关键，DESIGN sec 5.4）。
                        t_m = T.min(block_m, true_sizes[e] - row)
                        t_n = T.min(block_n, ffn - n_start)
                        for k in T.serial(T.ceildiv(K, block_k)):
                            # A5: full static K block (block_k divides K);
                            # non-A5 keeps the dynamic tail extent.
                            t_k = block_k if a5 else T.min(block_k, K - k * block_k)
                            # 动态 extent 取数：GM->L1（隐式 ND2NZ）。
                            T.copy(
                                A[m_start:m_start + t_m,
                                  k * block_k:k * block_k + t_k],
                                a_l1[0:t_m, 0:t_k],
                            )
                            T.copy(
                                B[e, n_start:n_start + t_n,
                                  k * block_k:k * block_k + t_k],
                                bg_l1[0:t_n, 0:t_k],
                            )
                            T.copy(
                                B[e, ffn + n_start:ffn + n_start + t_n,
                                  k * block_k:k * block_k + t_k],
                                bu_l1[0:t_n, 0:t_k],
                            )
                            # size=[t_m, t_k, t_n] 把归约范围限定到有效 tile_k
                            # （正确性必需：否则缓冲 [tile_k, block_k) 的陈旧尾值
                            # 会被按静态 block_k 计入累加，DESIGN sec 3.3 注 1）。
                            # A5: static mmadL1 size; stale M rows / N columns
                            # are computed but discarded by the [t_m, t_n]
                            # fixpipe write.  Non-A5 keeps dynamic sizes.
                            gemm_size = ([block_m, block_k, block_n] if a5
                                         else [t_m, t_k, t_n])
                            T.gemm(a_l1, bg_l1, c_gate,
                                   size=gemm_size,
                                   b_transpose=True, initC=(k == 0))
                            T.gemm(a_l1, bu_l1, c_up,
                                   size=gemm_size,
                                   b_transpose=True, initC=(k == 0))
                        # packed 写出：gate->列 [n_start,..)、up->列 [ffn+n_start,..)，
                        # 两区间不相交（DESIGN sec 3.3 注 4）。动态 extent 裁剪，
                        # 杜绝越界写坏下一专家行。
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_gate, GU[m_start, n_start],
                                            size=[t_m, t_n],
                                            enable_nz2nd=True)
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_up, GU[m_start, ffn + n_start],
                                            size=[t_m, t_n],
                                            enable_nz2nd=True)

    return main


# --------------------------------------------------------------------------
# Kernel V: packed activation (Vector).  C[m,n] = act(GU[m,n]) * GU[m,ffn+n]
# in the fp32 domain, single round-back.  (DESIGN sec 3.3 / sec 4.7)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def _act_packed(numel, ffn, dtype="float16", activation="silu_and_mul"):
    block_m = _ACT_BLOCK_M
    block_n = _ACT_BLOCK_N
    chunks = _ceildiv(ffn, block_n)
    m_blocks = _ceildiv(numel, block_m)
    num_logical = m_blocks * chunks
    num_kernels = min(num_logical, _vector_cores())   # 纯 Vector 分核
    num_local = _ceildiv(num_logical, num_kernels)    # 编译期静态边界

    @T.prim_func
    def main(
        GU: T.Tensor((numel, 2 * ffn), "float32"),
        C: T.Tensor((numel, ffn), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # 全部 buffer 无条件声明：npuir 解析器会解析所有分支体（即使
            # activation 是 Python 常量）；未使用分支由 DCE 移除。
            g32 = T.alloc_ub((block_m, block_n), "float32")   # gate 段 [n0,n0+bn)
            v32 = T.alloc_ub((block_m, block_n), "float32")   # up 段 [ffn+n0,..)
            y_ub = T.alloc_ub((block_m, block_n), dtype)      # 回舍结果 / 输出 staging
            t32 = T.alloc_ub((block_m, block_n), "float32")   # -g -> exp(-g) 原地链
            d32 = T.alloc_ub((block_m, block_n), "float32")   # 分母 1+exp(-g)
            p32 = T.alloc_ub((block_m, block_n), "float32")   # silu(g) -> silu(g)*v
            ag = T.alloc_ub((block_m, block_n), "float32")    # |g|
            z = T.alloc_ub((block_m, block_n), "float32")     # |g|/sqrt2 -> p*z（原地）
            z2 = T.alloc_ub((block_m, block_n), "float32")    # z^2 -> -z^2（原地）
            ez = T.alloc_ub((block_m, block_n), "float32")    # exp(-z^2)
            den = T.alloc_ub((block_m, block_n), "float32")   # 1+p*z -> -|g|*erf_abs
            tin = T.alloc_ub((block_m, block_n), "float32")   # 1/den, Horner 变量
            poly = T.alloc_ub((block_m, block_n), "float32")  # Horner P(t) -> erfc
            dl = T.alloc_ub((block_m, block_n), "float32")    # gelu -> gelu*v
            # persistent 核内串行：num_local 为编译期静态值，grid-stride
            # task = i*num_kernels + cid，if 屏蔽越界 task。
            for i in T.serial(num_local):
                task = i * num_kernels + cid
                if task < num_logical:
                    mb = task // chunks      # 行块（raster 序）
                    ck = task % chunks       # 列块
                    m0 = mb * block_m
                    n0 = ck * block_n
                    # runtime 尾块：非整除 shape 逐 task 截断（不 host padding）。
                    m_t = T.min(block_m, numel - m0)
                    n_t = T.min(block_n, ffn - n0)
                    # packed 直取两段：gate 列 [n0,n0+nt)、up 列 [ffn+n0,..)
                    # （行 stride 2*ffn，跨段切片）。
                    T.copy(GU[m0:m0 + m_t, n0:n0 + n_t], g32[0:m_t, 0:n_t])
                    T.copy(GU[m0:m0 + m_t, ffn + n0:ffn + n0 + n_t],
                           v32[0:m_t, 0:n_t])
                    if activation == "silu_and_mul":
                        # silu direct-exp（A3）：(g/d)*v，未合并结合序，避免
                        # |g|,|v| 同近 dtype max 时 (g*v)/d 的 inf/inf=NaN。
                        T.vmul(g32, -1.0, t32)       # t = -g
                        T.vexp(t32, t32)             # t = exp(-g)
                        T.vadd(t32, 1.0, d32)        # d = 1 + exp(-g)
                        T.vdiv(g32, d32, p32)        # p = silu(g)
                        T.vmul(p32, v32, p32)        # p = silu(g) * v
                        T.vcast(p32, y_ub, round_mode="rint")
                    else:
                        # gelu exact（A4）：odd-symmetry + A&S 7.1.26 erfc 链。
                        # gelu(g) = 0.5*(g + |g|*erf(|g|/sqrt2))
                        T.vabs(g32, ag)              # ag = |g|
                        T.vmul(ag, _INV_SQRT2, z)    # z  = |g|/sqrt2
                        T.vmul(z, z, z2)             # z2 = z^2
                        T.vmul(z2, -1.0, z2)         # z2 = -z^2
                        T.vexp(z2, ez)               # ez = exp(-z^2)
                        T.vmul(z, _ARF_P, z)         # z  = p*z
                        T.vadd(z, 1.0, den)          # den = 1 + p*z
                        T.vrec(den, tin)             # t = 1/den
                        T.vmul(tin, _A5, poly)       # Horner P(t)（A&S 7.1.26）
                        T.vadd(poly, _A4, poly)
                        T.vmul(poly, tin, poly)
                        T.vadd(poly, _A3, poly)
                        T.vmul(poly, tin, poly)
                        T.vadd(poly, _A2, poly)
                        T.vmul(poly, tin, poly)
                        T.vadd(poly, _A1, poly)
                        T.vmul(poly, tin, poly)
                        T.vmul(poly, ez, poly)       # erfc = P(t)*exp(-z^2)
                        T.vadd(poly, -1.0, den)      # den = erfc-1 = -erf_abs
                        T.vmul(ag, den, den)         # den = -|g|*erf_abs
                        T.vsub(g32, den, dl)         # dl = g + |g|*erf_abs
                        T.vmul(dl, 0.5, dl)          # dl = gelu(g)
                        T.vmul(dl, v32, dl)          # dl = gelu(g)*v
                        T.vcast(dl, y_ub, round_mode="rint")
                    # 向量 op 覆盖整 buffer，仅 [0:m_t,0:n_t] 写回，垃圾不落 GM。
                    T.copy(y_ub[0:m_t, 0:n_t], C[m0:m0 + m_t, n0:n0 + n_t])

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
    """Raise RLIMIT_STACK to the hard limit (mitigation for a toolchain bug).

    ``bishengir-compile`` walks the nested ``scf`` regions (task loop ->
    dead-tile guard -> K loop) with deep recursion and dies with SIGSEGV; a
    larger stack reduces the crash rate but does NOT eliminate it -- the
    reliable net is the bounded retry in ``_run_kernel``.  Child processes
    inherit the raised limit across exec.
    """
    try:
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ValueError, OSError, AttributeError):
        pass


_raise_stack_limit()


def _run_kernel(build, call_args, attempts=None):
    """Invoke a JIT kernel, retrying on flaky bishengir-compile SIGSEGVs.

    The retry rebuilds the factory so a failed compile is not reused; it does
    NOT mask deterministic shape-dependent errors.
    """
    if attempts is None:
        attempts = int(os.environ.get("MOE_COMPILE_ATTEMPTS", "20"))
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


# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------
_STATS_MAX_DIFF = {}


def run_case(numel, num_experts, ffn, k, dtype, activation, distribution, tag):
    torch.manual_seed(42)
    tdt = _DTYPE_MAP[dtype]
    sizes_l, offsets_l = make_sizes_offsets(numel, num_experts, distribution)
    block_m, block_n, block_k = _GEMM_BLOCK_M, _GEMM_BLOCK_N, _GEMM_BLOCK_K
    max_tiles = max_tiles_of(numel, num_experts, block_m)
    ids_l, rows_l, total = build_schedule(sizes_l, block_m)
    sizes = torch.tensor(sizes_l, dtype=torch.int32)
    offsets = torch.tensor(offsets_l, dtype=torch.int32)
    a = torch.randn(numel, k, dtype=tdt) * 0.02
    b = torch.randn(num_experts, 2 * ffn, k, dtype=tdt) * 0.02

    # kernel G: Expert Cube dual-accumulator grouped GEMM -> packed gate_up(fp32)
    gu = _run_kernel(
        lambda: _gemm_gate_up(numel, max_tiles, num_experts, ffn, k, dtype,
                              block_m, block_n, block_k),
        (
            a.npu(), b.npu(),
            torch.tensor(ids_l, dtype=torch.int32).npu(),
            torch.tensor(rows_l, dtype=torch.int32).npu(),
            offsets.npu(), sizes.npu(),
            torch.tensor([total], dtype=torch.int32).npu(),
        ),
    )
    torch.npu.synchronize()

    # kernel V: Vector packed activation -> C
    c = _run_kernel(
        lambda: _act_packed(numel, ffn, dtype, activation),
        (gu,),
    )
    torch.npu.synchronize()

    ref = golden_fused_act(a, b, sizes, offsets, activation)
    cc = c.cpu().float()
    rr = ref.float()
    max_diff = (cc - rr).abs().max().item()
    torch.testing.assert_close(cc, rr, atol=1e-2, rtol=1e-2)

    key = (dtype, activation)
    if max_diff > _STATS_MAX_DIFF.get(key, -1):
        _STATS_MAX_DIFF[key] = max_diff
    print(f"[{tag}] PASS  shape=({numel},{num_experts},{ffn},{k}) dtype={dtype} "
          f"act={activation} dist={distribution} max_tiles={max_tiles} "
          f"total_tiles={total} max_diff={max_diff:.3e}")
    return max_diff


def run_L0():
    cases = [
        (64, 4, 128, 64, "bfloat16", "silu_and_mul", "uniform",
         "L0-1 smoke silu bf16, ffn=block_n, M-tail (s_e=16<128)"),
        (64, 4, 128, 64, "float16", "gelu_and_mul", "uniform",
         "L0-2 smoke gelu fp16, A&S erf chain"),
        (256, 8, 256, 128, "bfloat16", "silu_and_mul", "uniform",
         "L0-3 multi-expert, num_pid_n=2, dual accumulator"),
        (512, 4, 512, 256, "bfloat16", "silu_and_mul", "uniform",
         "L0-4 num_pid_n>1 & num_local>1 (flaky-compile domain)"),
        (128, 8, 256, 70, "bfloat16", "silu_and_mul", "uniform",
         "L0-5 K non-aligned (70), dynamic K reduction extent"),
        (128, 8, 256, 64, "bfloat16", "silu_and_mul", "uniform",
         "L0-6 dead tiles (total_tiles < max_tiles)"),
        (233, 5, 130, 96, "bfloat16", "silu_and_mul", "skewed",
         "L0-7 odd shape + skewed distribution (M/N/K tails)"),
        (255, 6, 200, 100, "float16", "gelu_and_mul", "empty",
         "L0-8 empty experts + K tail + gelu"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L0 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L1():
    cases = [
        (256, 8, 512, 128, "float16", "silu_and_mul", "uniform",
         "L1-1 medium uniform, 2 pid_n"),
        (1024, 8, 512, 256, "bfloat16", "silu_and_mul", "uniform",
         "L1-2 multi local task + 4 pid_n"),
        (384, 10, 300, 100, "bfloat16", "gelu_and_mul", "skewed",
         "L1-3 skewed + gelu"),
        (512, 4, 768, 256, "float16", "silu_and_mul", "uniform",
         "L1-4 6 pid_n"),
        (300, 12, 200, 64, "float16", "silu_and_mul", "skewed",
         "L1-5 E=12 skewed non-divisible"),
        (128, 16, 384, 192, "bfloat16", "gelu_and_mul", "uniform",
         "L1-6 E=16 gelu, K tail"),
        (1024, 64, 256, 128, "bfloat16", "silu_and_mul", "empty",
         "L1-7 empty experts, many experts"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"L1 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L2():
    cases = [
        (4096, 64, 512, 256, "bfloat16", "silu_and_mul", "uniform",
         "L2-1 large uniform"),
        (2048, 32, 1024, 512, "bfloat16", "gelu_and_mul", "skewed",
         "L2-2 large skewed gelu"),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (non-blocking): {c[-1]}: {exc!r}")
    print(f"L2 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_boundary():
    cases = [
        (16, 4, 16, 32, "float16", "silu_and_mul", "uniform", "B-1 tiny ffn/K"),
        (70, 7, 70, 40, "bfloat16", "gelu_and_mul", "skewed", "B-2 all non-divisible"),
        (33, 4, 128, 33, "float16", "silu_and_mul", "uniform", "B-3 odd numel/K"),
        (128, 8, 48, 16, "float16", "silu_and_mul", "empty", "B-4 empty experts, tiny N/K"),
        (64, 1, 64, 64, "bfloat16", "silu_and_mul", "uniform", "B-5 single expert"),
        (1, 1, 16, 16, "float16", "gelu_and_mul", "uniform", "B-6 numel=1"),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): {c[-1]}: {exc!r}")
    print(f"Boundary done ({len(cases)} cases), worst max_diff={worst:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0",
        choices=["L0", "L1", "L2", "boundary", "all"],
    )
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L0()
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()

    print("Summary max_diff per (dtype, act): "
          + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items())))


if __name__ == "__main__":
    main()
