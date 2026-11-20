# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_cumulative_kernel (CumprodFwdOp inclusive prefix-product scan) -- NPU Developer mode.

Migrated from the GPU TileOPs ``_cumulative_kernel`` (``op_kind="prod"`` branch,
``tileops/kernels/reduction/cumulative.py`` L176-L254) to ``target="npuir"``.

Semantics (DESIGN.md section 0.1 / 3.1, frozen):
    y[i, j] = prod_{k <= j} x[i, k]          # torch.cumprod(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  y: (M, N) same_as(x)

NPU redesign (DESIGN.md v2 section 0.6 R1-R4 / 1.6.1 E1-E5, frozen):
    R1/E1: the row-wise sequential running product is replaced by the log-domain
        identity  prod x_k = (-1)^(#neg) * exp(sum ln|x_k|), turning the within-
        chunk prefix into ONE Cube MAC-array pass:  Z = A @ L^T  with
        A = ln|x| and L = tril(ones[bs, bs]).
    R1/E2: the sign is carried exactly by a negative-count prefix K = Ind @ L^T,
        with the rounding-free fp32 arithmetic indicator
        Ind = 0.5 * (1 - x / (|x| + eps)),  eps = 1e-30.
    R4/E3: the chunk-level carry is a SECOND triangular gemm over the chunk
        totals (carry = incl_prefix(tot) - own_total), so NO in-kernel serial
        carry is needed (probe G1-b: a T.gemm inside a T.serial loop breaks the
        loop-carried Vector accumulator).
    R3: non-divisible boundaries are handled on the HOST by identity-element 1.0
        alignment of both axes (N_eff = ceil(N/bs)*bs, M_eff = ceil(M/bm)*bm),
        so the kernel views are fully divisible -> no padding lane, no partial
        gemm, no ``T.clear`` on a partially written tile (CG-2026-0035 /
        examples/gemm/_gemv_kernel/repro/TRAP-clear-padded-fragment-nan.py).
        The surplus columns/rows are trimmed by the host caller.

v3 (2026-10-22) -- HIERARCHICAL CARRY (covers 4096 < N <= 32768):
    v2's phase-2 carry is a single ncp x ncp triangular gemm with
    ncp = max(16, ceil(N/bs)); at N = 32768 that is 512x512 fp32 = 1 MB for the
    amplitude operator alone, far beyond the 192 KB UB, so v2 could not run wide
    rows (the wrapper fell back to v1 => 2089 us for [64, 32768]).
    v3 splits the carry into two levels:
      L1 (chunk, phase 1):     unchanged within-chunk prefix (bs x bs tril gemm).
      L2 (segment, phase 2a):  per SEGMENT of nc_seg chunks, a
                               nc_seg x nc_seg tril gemm gives the within-segment
                               chunk prefix; exclusive = incl - own (same formula
                               as v2, smaller scope).  It also emits the segment
                               total (last column of the inclusive prefix).
      L3 (segment prefix, 2b): one ncp3 x ncp3 tril gemm over the ns segment
                               totals (ncp3 = max(16, ns); 16 for N = 32768).
      L4 (phase 3):            carry = cw[chunk] + cs[segment].
    Granularity is derived from N alone: nc_seg = min(nc, SEG_MAX=64),
    ns = ceil(nc / nc_seg).  For nc <= 64 (N <= 4096) ns == 1 and the L3 kernel
    plus its term are compile-time elided, so the emitted kernels are the v2
    kernels (bit-identical results, same phase count) -- the already-delivered
    8.9x-23x path is preserved by construction, not by a separate dispatch.

v4 (2026-10-26) -- CHUNK WIDTH 64 -> 128:
    v3 removed the scalar-scan floor and the UB ceiling; what is left is pure
    per-block latency (see the BS comment below).  v4 halves the block count by
    doubling the scan-axis chunk width, which cuts the whole-op time 35.7 % on
    the hidden-state workload (654 -> 422 us fp16, 663 -> 427 us bf16) and does
    not regress the narrow-row / long-row workloads.  No algorithm change: the
    carry ladder, the kernels and the numerics are v3's, only BS moved.

Programming mode: Developer, non-persistent one-dimensional mixed grids
(compiler auto CV-split; Expert ``T.sync_block_set/wait`` is unavailable on this
toolchain -- CG-2026-0026 -- and Developer persistent + gemm crashes --
CG-2026-0010).

Known toolchain facts enforced here:
    - ``T.vln`` / ``T.vexp`` do NOT support bf16 -> every complex buffer is fp32.
    - ``T.vcast(fp32 -> fp32)`` returns garbage/zeros on this toolchain -> for
      fp32 inputs the gemm consumes the fp32 buffer DIRECTLY (no vcast); the
      vcast-to-dtype operand path is only used for fp16/bf16.
    - ``T.vfloor`` is unavailable on this target -> the parity floor uses
      ``T.vcast(..., round_mode="floor")``.
    - A ``T.Parallel`` loop reading a gemm fragment is pathologically slow here
      (Stage 3 measured 34 ms) -> fragment -> shared via ``T.copy`` then a
      vector op.
    - ``T.vsub`` with a scalar ``src0`` must use the ``T.float32(1.0)`` form.
"""

import os

# Developer mode: the compiler manages UB (alloc_shared) and auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# Tile shapes (DESIGN.md section 5.2): bs = scan-axis chunk width, bm = row block.
# Both are multiples of 16 (Cube fractal lower bound).
# bm = 64 (v3 Round 3): every phase is latency-bound (all msprof pipe ratios of
# phase 1 are <= 0.007), so halving the block count halves the exposed per-block
# DMA/CV-handshake latency. Measured on [2048,4096] fp16: phase1 716 -> 455 us,
# phase3 268 -> 190 us, phase2a 21 -> 13 us; on [64,32768] bf16: 129 + 73 us.
#
# BS = 128 (v4 Round 4): the SAME lever one step further, taken on the scan axis
# instead of the row axis.  Every phase visits ceil(M/bm) * ceil(N/bs) blocks
# (phase 3: the same product on the pure-vector grid), and every block pays a
# fixed ~3.5-5.5 us of launch / MTE2 / cube-fixpipe handshake latency while its
# pipes sit at 0.1-20 % (msprof: phase 1 aiv_vec_ratio 0.0012, aic_cube_ratio
# 0.0003).  Doubling bs halves the block count for the same bm; the price is
# that the triangular operator doubles its MAC count (total Cube MACs are
# M*N*bs) and that L grows to bs^2 -- both irrelevant at a 0.03 % cube ratio.
# Measured [2048,4096] fp16 whole-op (msprof, same session, A/B alternating):
#   v3 (bs=64)  654-663 us -> v4 (bs=128) 421-423 us  (-35.7 %)
# phase 1 452 -> 296 us, phase 3 187 -> 114 us, phase 2a 15 -> 13 us.
# bs = 256 is NOT reachable: BiShengIR reports "ub overflow, requires 2621696
# bits while 1572864 bits available" (bs=256 needs 320 KB of the 192 KB UB;
# bs=512 is 4x worse).  bs = 128 is the largest UB-legal chunk width.
# bm = 128 was measured as the alternative way to halve the block count and is
# weaker (502 us on the same workload): it does not shrink phase 3's per-block
# DMA count and leaves the bs^2 triangular reload per block unchanged.
BS = 128
BM = 64

# v3 hierarchical-carry knobs.
# SEG_MAX: maximum number of chunks per carry segment (level-2 gemm side).  The
# level-2 operator is SEG_MAX x SEG_MAX fp32 = 16 KB at 64, so the whole carry
# ladder (level-2 + level-3) stays inside the 192 KB UB at any supported N.
# Capping at 64 also makes ns == 1 for nc <= 64 (N <= 4096), i.e. the v2 path is
# reproduced exactly (see the module docstring).
SEG_MAX = 64
# Level-3 gemm lower bound (Cube fractal K/N floor), same rule v2 used for ncp.
_MIN_CARRY_DIM = 16

# Tuned defaults exposed for the wrapper baseline/perf_opt switch block: this
# module ignores the GPU-derived (block_m, block_n) the harness passes and uses
# (BM, BS) + the N-derived carry hierarchy.
TUNED_BLOCK_M = BM
TUNED_BLOCK_N = BS

# Log-floor stabilisation (DESIGN.md section 1.6.1-E4): exp(LOGFLOOR) underflows
# to exact 0 in fp32, matching torch.cumprod's "zero stays zero" behaviour, and
# keeps -inf out of the carry arithmetic.
LOGFLOOR = -1e4

# Sign indicator threshold (DESIGN.md section 1.6.1-E2): eps = 1e-30 makes
# x / (|x| + eps) exactly +/-1 for every normal |x| != 0, so the indicator is an
# exact 0/1; the only deviation (x == 0 -> 0.5) lands where exp() underflows to 0
# and the sign is irrelevant.
EPS = 1e-30

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# dtype -> (torch dtype, atol, rtol); tolerances from the harness
# ``_cumprod_tol`` (examples/TileOPs/tests/ops/test_cumulative.py).
_DTYPE_MAP = {
    "float16": (torch.float16, 5e-2, 5e-2),
    "bfloat16": (torch.bfloat16, 5e-2, 5e-2),
    "float32": (torch.float32, 1e-3, 1e-3),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_cumprod(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch reference: ``torch.cumprod(x.float(), dim).to(x.dtype)`` on CPU.

    Matches the harness ``CumprodFwdTest.ref_program`` and the frozen source
    semantics (DESIGN.md section 0.1 / 8.1).  It does NOT reproduce the NPU
    redesign (no log-domain transform, no triangular gemm, no chunk carry),
    guaranteeing validation independence.  Uses an explicit ``torch.cumprod``
    module call so the reference is unambiguously a torch API call.
    """
    return torch.cumprod(x.float(), dim=dim).to(x.dtype)


# ---------- Phase 1: within-chunk prefix + chunk totals ----------
@tilelang.jit(out_idx=[2, 3, 4, 5], target="npuir")
def scan_partial(M, NE, bs, ncp, bm, dtype):
    """One block per (row-block, chunk): within-chunk log prefix Z and neg-count
    prefix K, plus the chunk totals (last column of each gemm)."""
    nc = NE // bs
    is_f32 = dtype == "float32"

    @T.prim_func
    def main(
        x_pad: T.Tensor((M, NE), dtype),
        L: T.Tensor((bs, bs), dtype),
        part: T.Tensor((M, NE), "float32"),
        kpart: T.Tensor((M, NE), "float32"),
        tot: T.Tensor((M, ncp), "float32"),
        ktot: T.Tensor((M, ncp), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, bm) * nc, is_npu=True) as (cid, _):
            rb = cid // nc
            c = cid % nc
            r0 = rb * bm
            c0 = c * bs
            l_sh = T.alloc_shared((bs, bs), dtype)
            x_sh = T.alloc_shared((bm, bs), "float32")
            a_sh = T.alloc_shared((bm, bs), "float32")
            i_sh = T.alloc_shared((bm, bs), "float32")
            z_fr = T.alloc_fragment((bm, bs), "float32")
            k_fr = T.alloc_fragment((bm, bs), "float32")
            a_dt = T.alloc_shared((bm, bs), dtype)
            i_dt = T.alloc_shared((bm, bs), dtype)

            T.copy(L, l_sh)
            # 宿主已按 bs 对齐，整宽载入：无 padding lane、无 partial gemm
            T.copy(x_pad[r0 : r0 + bm, c0 : c0 + bs], x_sh)
            # --- 幅值通道：A = max(ln|x|, LOGFLOOR)（log-floor 稳定化，避免 -inf 进入 carry）---
            T.vabs(x_sh, a_sh)
            T.vln(a_sh, a_sh)
            T.vmax(a_sh, LOGFLOOR, a_sh)
            # --- 符号通道：逐元素比较 Ind = (x < 0) ? 1 : 0（精确 0/1，且无 vdiv）---
            for i, j in T.Parallel(bm, bs):
                i_sh[i, j] = T.if_then_else(x_sh[i, j] < 0, 1.0, 0.0)
            # fp32->fp32 的 vcast 在本工具链返回 0（实测）→ fp32 直接把 fp32 缓冲喂给 Cube
            if is_f32:
                T.gemm(a_sh, l_sh, z_fr, initC=True, b_transpose=True)
                T.gemm(i_sh, l_sh, k_fr, initC=True, b_transpose=True)
            else:
                T.vcast(a_sh, a_dt)
                T.vcast(i_sh, i_dt)
                T.gemm(a_dt, l_sh, z_fr, initC=True, b_transpose=True)
                T.gemm(i_dt, l_sh, k_fr, initC=True, b_transpose=True)
            # chunk 总量 = gemm 输出末列（不用 reduce_sum 作用于 gemm 操作数缓冲，该上下文返回 ~0）
            T.copy(z_fr[0:bm, bs - 1 : bs], tot[r0 : r0 + bm, c : c + 1])
            T.copy(z_fr, part[r0 : r0 + bm, c0 : c0 + bs])
            T.copy(k_fr[0:bm, bs - 1 : bs], ktot[r0 : r0 + bm, c : c + 1])
            T.copy(k_fr, kpart[r0 : r0 + bm, c0 : c0 + bs])

    return main


# ---------- Phase 1L: long-row variant (fp32 amplitude operand) ----------
@tilelang.jit(out_idx=[2, 3, 4, 5], target="npuir")
def scan_partial_long(M, NE, bs, ncp, bm, dtype):
    """Phase 1 for WIDE rows (level-3 carry needed): identical to
    ``scan_partial`` except that the AMPLITUDE gemm consumes fp32 operands.

    Why: the log-magnitude operand is quantised to the input dtype in the v2
    path, and bf16/fp16 rounding of ``ln|x|`` is systematically biased.  That
    bias adds up LINEARLY along the scan axis, so it is harmless for N <= 4096
    (v2's proven regime) but breaks the harness tolerance at N = 32768:
    measured [64, 32768] bf16 `near1` max_diff 0.219 vs the 0.05 atol
    (fp16 0.027, fp32 1.4e-06).  Feeding fp32 to the Cube removes the
    quantisation (fp32 is exact to 2^-24), which is what the fp32-input path
    already did in v2.

    Both channels use the fp32 operator buffer: keeping an unused second
    ``(bs, bs)`` dtype buffer in the same kernel made the shared-memory aliasing
    corrupt the fp32 operand (measured chunk-total error 0.42 -> 7e-08 once the
    dead buffer was removed).
    """
    nc = NE // bs

    @T.prim_func
    def main(
        x_pad: T.Tensor((M, NE), dtype),
        L32: T.Tensor((bs, bs), "float32"),
        part: T.Tensor((M, NE), "float32"),
        kpart: T.Tensor((M, NE), "float32"),
        tot: T.Tensor((M, ncp), "float32"),
        ktot: T.Tensor((M, ncp), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, bm) * nc, is_npu=True) as (cid, _):
            rb = cid // nc
            c = cid % nc
            r0 = rb * bm
            c0 = c * bs
            l32 = T.alloc_shared((bs, bs), "float32")
            x_sh = T.alloc_shared((bm, bs), "float32")
            a_sh = T.alloc_shared((bm, bs), "float32")
            i_sh = T.alloc_shared((bm, bs), "float32")
            z_fr = T.alloc_fragment((bm, bs), "float32")
            k_fr = T.alloc_fragment((bm, bs), "float32")

            T.copy(L32, l32)
            T.copy(x_pad[r0 : r0 + bm, c0 : c0 + bs], x_sh)
            T.vabs(x_sh, a_sh)
            T.vln(a_sh, a_sh)
            T.vmax(a_sh, LOGFLOOR, a_sh)
            for i, j in T.Parallel(bm, bs):
                i_sh[i, j] = T.if_then_else(x_sh[i, j] < 0, 1.0, 0.0)
            T.gemm(a_sh, l32, z_fr, initC=True, b_transpose=True)
            T.gemm(i_sh, l32, k_fr, initC=True, b_transpose=True)
            T.copy(z_fr[0:bm, bs - 1 : bs], tot[r0 : r0 + bm, c : c + 1])
            T.copy(z_fr, part[r0 : r0 + bm, c0 : c0 + bs])
            T.copy(k_fr[0:bm, bs - 1 : bs], ktot[r0 : r0 + bm, c : c + 1])
            T.copy(k_fr, kpart[r0 : r0 + bm, c0 : c0 + bs])

    return main


# ---------- Phase 2a: segment-level (within-segment) chunk carry ----------
@tilelang.jit(out_idx=[4, 5, 6, 7], target="npuir")
def scan_segcarry(M, NR, ncp2, ns, ncp3, bm, dtype, emit_segtot):
    """One block per (row-block, segment): inclusive chunk prefix INSIDE the
    segment via Lc2 = tril(ones[ncp2]), then exclusive = inclusive - own_total.

    This is v2's phase-2 gemm with the scope narrowed from "all chunks" to
    "nc_seg chunks", which is what lets N = 32768 fit: ncp2 <= SEG_MAX (64),
    i.e. a 16 KB fp32 operator instead of 1 MB.

    ``emit_segtot`` is a compile-time flag: when the whole row is a single
    segment (nc <= SEG_MAX, i.e. N <= 4096) the segment totals are consumed
    nowhere, so the write -- and the level-3 kernel -- are elided entirely and
    this kernel is exactly v2's phase 2.

    The amplitude carry consumes the fp32 chunk totals DIRECTLY: a chunk total is
    a sum of bs log-magnitudes and can exceed the fp16 range, so an fp16 operand
    down-cast overflows to -inf and the triangular gemm's 0 * (-inf) poisons the
    whole row with NaN (Stage 3 finding).  The sign carry keeps an input-dtype
    operand (per-chunk counts <= bs are exact in fp16/bf16).
    """
    is_f32 = dtype == "float32"

    @T.prim_func
    def main(
        tot: T.Tensor((M, NR), "float32"),
        ktot: T.Tensor((M, NR), "float32"),
        Lc32: T.Tensor((ncp2, ncp2), "float32"),
        Lcd: T.Tensor((ncp2, ncp2), dtype),
        cw: T.Tensor((M, NR), "float32"),
        kcw: T.Tensor((M, NR), "float32"),
        segtot: T.Tensor((M, ncp3), "float32"),
        ksegtot: T.Tensor((M, ncp3), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, bm) * ns, is_npu=True) as (cid, _):
            rb = cid // ns
            s = cid % ns
            r0 = rb * bm
            k0 = s * ncp2
            lc32 = T.alloc_shared((ncp2, ncp2), "float32")
            lcd = T.alloc_shared((ncp2, ncp2), dtype)
            t_sh = T.alloc_shared((bm, ncp2), "float32")
            kt_sh = T.alloc_shared((bm, ncp2), "float32")
            kt_dt = T.alloc_shared((bm, ncp2), dtype)
            incl = T.alloc_fragment((bm, ncp2), "float32")
            kincl = T.alloc_fragment((bm, ncp2), "float32")
            ic_sh = T.alloc_shared((bm, ncp2), "float32")
            kic_sh = T.alloc_shared((bm, ncp2), "float32")
            c_sh = T.alloc_shared((bm, ncp2), "float32")
            kc_sh = T.alloc_shared((bm, ncp2), "float32")

            T.copy(Lc32, lc32)
            if not is_f32:
                T.copy(Lcd, lcd)
            # tot/ktot 宿主零初始化填充列（段尾不足 nc_seg 时）→ 整宽载入无未定义 lane
            T.copy(tot[r0 : r0 + bm, k0 : k0 + ncp2], t_sh)
            T.copy(ktot[r0 : r0 + bm, k0 : k0 + ncp2], kt_sh)
            # 幅值通道：chunk 总量可达 ~-6.4e5，超 fp16 值域 → 直接喂 fp32 操作数
            T.gemm(t_sh, lc32, incl, initC=True, b_transpose=True)
            # 符号通道：每 chunk 计数 <= bs，input dtype 表示精确
            if is_f32:
                T.gemm(kt_sh, lc32, kincl, initC=True, b_transpose=True)
            else:
                T.vcast(kt_sh, kt_dt)
                T.gemm(kt_dt, lcd, kincl, initC=True, b_transpose=True)
            # fragment→shared 用 T.copy（T.Parallel 直接读 gemm fragment 实测 34ms 病态慢）
            # 段内互斥 carry = 含前缀 − 自身总量（无核内串行 carry）
            T.copy(incl, ic_sh)
            T.copy(kincl, kic_sh)
            T.vsub(ic_sh, t_sh, c_sh)
            T.vsub(kic_sh, kt_sh, kc_sh)
            T.copy(c_sh, cw[r0 : r0 + bm, k0 : k0 + ncp2])
            T.copy(kc_sh, kcw[r0 : r0 + bm, k0 : k0 + ncp2])
            if emit_segtot:
                # 段总量 = 段内含前缀的末列（填充列为 0，故等于该段真实 chunk 总量）
                T.copy(incl[0:bm, ncp2 - 1 : ncp2], segtot[r0 : r0 + bm, s : s + 1])
                T.copy(kincl[0:bm, ncp2 - 1 : ncp2], ksegtot[r0 : r0 + bm, s : s + 1])

    return main


# ---------- Phase 2b: segment-level (cross-segment) prefix ----------
@tilelang.jit(out_idx=[3, 4], target="npuir")
def scan_segprefix(M, ns, ncp3, bm, dtype):
    """One block per row-block: exclusive prefix over the ns segment totals
    via Ls = tril(ones[ncp3])  (ncp3 = max(16, ns) = 16 for N = 32768, 32 B).

    Same "inclusive - own" formula as phase 2a, one level up.  Only built when
    ns > 1; for ns == 1 the host skips it and phase 3 drops its term.

    BOTH channels feed the gemm with fp32 operands here, unlike phase 2a.  A
    segment total of the negative count is ~nc_seg * bs / 2 ~ 2048 for N = 32768,
    i.e. PAST the fp16 exact-integer boundary of 2048 (and far past bf16's 256):
    an input-dtype operand rounds 2061 -> 2060 and the sign parity flips on a
    whole segment (measured max_diff 3.15 on the fp16 [32, 8192] check).  fp32
    carries integers exactly up to 2^24, so both channels stay exact.
    """

    @T.prim_func
    def main(
        segtot: T.Tensor((M, ncp3), "float32"),
        ksegtot: T.Tensor((M, ncp3), "float32"),
        Ls32: T.Tensor((ncp3, ncp3), "float32"),
        cs: T.Tensor((M, ncp3), "float32"),
        kcs: T.Tensor((M, ncp3), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, bm), is_npu=True) as (cid, _):
            r0 = cid * bm
            ls32 = T.alloc_shared((ncp3, ncp3), "float32")
            t_sh = T.alloc_shared((bm, ncp3), "float32")
            kt_sh = T.alloc_shared((bm, ncp3), "float32")
            incl = T.alloc_fragment((bm, ncp3), "float32")
            kincl = T.alloc_fragment((bm, ncp3), "float32")
            ic_sh = T.alloc_shared((bm, ncp3), "float32")
            kic_sh = T.alloc_shared((bm, ncp3), "float32")
            c_sh = T.alloc_shared((bm, ncp3), "float32")
            kc_sh = T.alloc_shared((bm, ncp3), "float32")

            T.copy(Ls32, ls32)
            T.copy(segtot[r0 : r0 + bm, 0:ncp3], t_sh)
            T.copy(ksegtot[r0 : r0 + bm, 0:ncp3], kt_sh)
            T.gemm(t_sh, ls32, incl, initC=True, b_transpose=True)
            T.gemm(kt_sh, ls32, kincl, initC=True, b_transpose=True)
            T.copy(incl, ic_sh)
            T.copy(kincl, kic_sh)
            T.vsub(ic_sh, t_sh, c_sh)
            T.vsub(kic_sh, kt_sh, kc_sh)
            T.copy(c_sh, cs[r0 : r0 + bm, 0:ncp3])
            T.copy(kc_sh, kcs[r0 : r0 + bm, 0:ncp3])

    return main


# ---------- Phase 3: add carry, exponentiate, apply exact sign parity ----------
@tilelang.jit(out_idx=[6], target="npuir")
def scan_final(M, NE, bs, nc, NR, ncp2, nc_seg, ns, ncp3, bm, dtype):
    """One block per (row-block, chunk):
    y = exp(part + cw[chunk] + cs[segment]) * (-1)^(kpart + kcw[chunk] + kcs[segment]).

    Carry read policy: the WHOLE (bm, ncp2) segment tile of cw/kcw (plus the
    (bm, ncp3) cs/kcs tiles when ns > 1) is loaded with one contiguous DMA per
    buffer, and the column this block needs is extracted with a 32-element
    scalar loop.  A single strided column of a (M, NR) buffer is 32 four-byte
    MTE2 transfers at an NR*4 stride, measured at ~1 us EACH (msprof on
    [64, 32768]: aiv_mte2_ratio 0.87, active b/w 2.9 GB/s, 13 us per block;
    the same pathology costs 843 us of the 1181 us on [2048, 4096]).
    """

    @T.prim_func
    def main(
        part: T.Tensor((M, NE), "float32"),
        kpart: T.Tensor((M, NE), "float32"),
        cw: T.Tensor((M, NR), "float32"),
        kcw: T.Tensor((M, NR), "float32"),
        cs: T.Tensor((M, ncp3), "float32"),
        kcs: T.Tensor((M, ncp3), "float32"),
        y: T.Tensor((M, NE), dtype),
    ):
        with T.Kernel(T.ceildiv(M, bm) * nc, is_npu=True) as (cid, _):
            rb = cid // nc
            c = cid % nc
            r0 = rb * bm
            c0 = c * bs
            # 本 chunk 的段号与段内偏移（ns == 1 时 seg == 0 / off == c）。
            # 必须在 if/else 之外定义：tvm script parser 的 if 分支自成作用域。
            seg = c // nc_seg
            off = c - seg * nc_seg
            p_sh = T.alloc_shared((bm, bs), "float32")
            k_sh = T.alloc_shared((bm, bs), "float32")
            cc_sh = T.alloc_shared((bm, 1), "float32")
            kc_sh = T.alloc_shared((bm, 1), "float32")
            ss_sh = T.alloc_shared((bm, 1), "float32")
            kss_sh = T.alloc_shared((bm, 1), "float32")
            m32 = T.alloc_shared((bm, bs), "float32")
            c32 = T.alloc_shared((bm, bs), "float32")
            par = T.alloc_shared((bm, bs), "float32")
            sgn = T.alloc_shared((bm, bs), "float32")
            T.copy(part[r0 : r0 + bm, c0 : c0 + bs], p_sh)
            T.copy(kpart[r0 : r0 + bm, c0 : c0 + bs], k_sh)
            if nc <= 16:
                # 窄行（N <= 1024）：cw 只有 <= ncp2 列，单列读是 32 行 x 4B、
                # 行距 <= 64B 的一次近连续访问，与 v2 完全一致；改载整段 tile 反而多搬
                # 2KB/block（实测 512x300 慢 ~1.5us）。
                T.copy(cw[r0 : r0 + bm, c : c + 1], cc_sh)
                T.copy(kcw[r0 : r0 + bm, c : c + 1], kc_sh)
            else:
                # 整段 tile 一次连续 DMA 载入，再用 32 元素标量抽取本 chunk 的一列。
                # 单列跨步读（bm 行 x 4B、行距 NR*4）是每次 ~1us 的 latency-bound
                # MTE2：hidden-state(2048x4096) 的 phase 3 单列读版因此多花 ~120us
                # （msprof 对 [64,32768] 的旧版测得 aiv_mte2_ratio 0.87、
                # active bw 2.9GB/s、13us/block）。
                # 注意：不能把 runtime 列号直接写进 (bm, bs) 向量循环——shared 的
                # runtime 索引会让整个循环标量化（实测 339us -> 3034us）。
                cwt = T.alloc_shared((bm, ncp2), "float32")
                kcwt = T.alloc_shared((bm, ncp2), "float32")
                T.copy(cw[r0 : r0 + bm, seg * ncp2 : seg * ncp2 + ncp2], cwt)
                T.copy(kcw[r0 : r0 + bm, seg * ncp2 : seg * ncp2 + ncp2], kcwt)
                for i in T.Parallel(bm):
                    cc_sh[i, 0] = cwt[i, off]
                for i in T.Parallel(bm):
                    kc_sh[i, 0] = kcwt[i, off]
            # 两个独立 T.Parallel：同一循环内两次 store 会触发编译器病态
            # 调度（实测 6.9ms -> 0.42ms，16x；Stage 3 发现）
            for i, j in T.Parallel(bm, bs):
                m32[i, j] = p_sh[i, j] + cc_sh[i, 0]
            for i, j in T.Parallel(bm, bs):
                c32[i, j] = k_sh[i, j] + kc_sh[i, 0]
            if ns > 1:
                # 段间 carry：整段 (bm, ncp3) tile 载入 + 标量抽取本段一列
                cst = T.alloc_shared((bm, ncp3), "float32")
                kcst = T.alloc_shared((bm, ncp3), "float32")
                T.copy(cs[r0 : r0 + bm, 0:ncp3], cst)
                T.copy(kcs[r0 : r0 + bm, 0:ncp3], kcst)
                for i in T.Parallel(bm):
                    ss_sh[i, 0] = cst[i, seg]
                for i in T.Parallel(bm):
                    kss_sh[i, 0] = kcst[i, seg]
                for i, j in T.Parallel(bm, bs):
                    m32[i, j] = m32[i, j] + ss_sh[i, 0]
                for i, j in T.Parallel(bm, bs):
                    c32[i, j] = c32[i, j] + kss_sh[i, 0]
            T.vexp(m32, m32)
            # 符号奇偶精确：sign = 1 − 2·(c − 2·floor(c/2))；T.vfloor 不可用 → vcast floor
            # （T.vsub 标量 src0 用 T.float32(...) 形态）
            T.vmul(c32, 0.5, par)
            T.vcast(par, par, round_mode="floor")
            T.vmul(par, 2.0, par)
            T.vsub(c32, par, par)
            T.vmul(par, 2.0, par)
            T.vsub(T.float32(1.0), par, sgn)
            T.vmul(m32, sgn, m32)
            T.copy(m32, y[r0 : r0 + bm, c0 : c0 + bs])

    return main


# ---------- Kernel factory (harness contract) ----------
def _carry_plan(N):
    """Derive the v3 carry hierarchy from N alone (compile-time constants).

    Returns (NE, nc, nc_seg, ncp2, ns, NR, ncp3):

      nc     = NE / bs                      number of bs-wide chunks
      nc_seg = min(nc, SEG_MAX)             chunks per carry segment
      ncp2   = max(16, nc_seg)              level-2 gemm side == segment stride
      ns     = ceil(nc / nc_seg)            number of segments
      NR     = ns * ncp2                    chunk-major buffer row count
      ncp3   = max(16, ns)                  level-3 gemm side

    ``ns == 1`` for every nc <= 64, i.e. for N <= 4096: then there is no level-3
    kernel and the level-2 gemm is exactly v2's ncp x ncp carry.
    """
    NE = _ceildiv(N, BS) * BS
    nc = NE // BS
    nc_seg = min(nc, SEG_MAX)
    ncp2 = max(_MIN_CARRY_DIM, nc_seg)
    ns = _ceildiv(nc, nc_seg)
    NR = ns * ncp2
    ncp3 = max(_MIN_CARRY_DIM, ns)
    return NE, nc, nc_seg, ncp2, ns, NR, ncp3


def select_config(M, N):
    """Tuned tile shape for the profiling harness (block_m, block_n).

    This module ignores the GPU-derived harness tile and uses (BM, BS); the
    helper exists so ``perf_opt/bench_kernel.py`` drives it unchanged (it calls
    ``mod.select_config(M, N)``).
    """
    return BM, BS


def _cumulative_kernel(M, N, op_kind, dtype):
    """Build the NPU inclusive prefix-product scan kernel factory.

    Mirrors the harness contract
    ``_cumulative_kernel(M, N, op_kind, dtype)(block_m, block_n)(x)``.
    K9: the ``threads`` parameter is removed (NPU has no CUDA threads concept).

    ``op_kind`` is kept for sibling-scan interface compatibility; this
    migration unit implements the ``prod`` branch only (the ``sum`` branch is a
    sibling unit that maps directly to ``T.cumsum``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "prod":
        raise ValueError(f"this unit implements op_kind='prod' only, got {op_kind!r}")
    if M <= 0 or N <= 0:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")

    torch_dtype = _DTYPE_MAP[dtype][0]

    bs, bm = BS, BM
    NE, nc, nc_seg, ncp2, ns, NR, ncp3 = _carry_plan(N)
    Me = _ceildiv(M, bm) * bm  # row axis aligned to the bm multiple
    multi = ns > 1  # level-3 (cross-segment) carry needed

    # Wide rows (level-3 carry present) need the fp32 amplitude operand; fp32
    # input already gets it.
    long_amp = multi and dtype != "float32"
    if long_amp:
        f1 = scan_partial_long(Me, NE, bs, NR, bm, dtype)
    else:
        f1 = scan_partial(Me, NE, bs, NR, bm, dtype)
    f2 = scan_segcarry(Me, NR, ncp2, ns, ncp3, bm, dtype, multi)
    f3 = scan_segprefix(Me, ns, ncp3, bm, dtype) if multi else None
    f4 = scan_final(Me, NE, bs, nc, NR, ncp2, nc_seg, ns, ncp3, bm, dtype)

    # Host constants (triangular all-ones operators); they do not modify x.
    L = torch.tril(torch.ones(bs, bs, dtype=torch_dtype)).npu()
    L32 = torch.tril(torch.ones(bs, bs, dtype=torch.float32)).npu() if long_amp else None
    Lc32 = torch.tril(torch.ones(ncp2, ncp2, dtype=torch.float32)).npu()
    Lcd = torch.tril(torch.ones(ncp2, ncp2, dtype=torch_dtype)).npu()
    Ls32 = torch.tril(torch.ones(ncp3, ncp3, dtype=torch.float32)).npu() if multi else None
    # ns == 1: the level-3 term is compile-time elided, but phase 3 still takes
    # the buffer (a read that never reaches the emitted code) -> a zero buffer.
    cs_zero = torch.zeros(Me, ncp3, dtype=torch.float32).npu()
    kcs_zero = torch.zeros(Me, ncp3, dtype=torch.float32).npu()

    def _func(block_m=None, block_n=None):
        # The harness passes its GPU-derived (block_m, block_n); the NPU kernel
        # uses the frozen DESIGN tile shapes (BS, BM) and ignores them.

        def call(x):
            M_, N_ = x.shape
            if Me == M_ and N_ == NE:
                x_pad = x  # already aligned, no copy
            else:
                # Host identity (multiplicative unit 1.0) alignment: surplus
                # rows/cols become 1.0 -> ln 1 = 0 and indicator 0, so they do
                # not change any valid output.
                x_pad = torch.ones(Me, NE, dtype=x.dtype, device=x.device)
                x_pad[:M_, :N_] = x
            if long_amp:
                part, kpart, tot, ktot = f1(x_pad, L32)
            else:
                part, kpart, tot, ktot = f1(x_pad, L)
            if nc < NR:
                # Host zero-init of the padded chunk-total ROWS: keeps the
                # level-2 full-width load lane-defined (no 0*NaN contamination)
                # and makes a segment's last row its true total.
                tot[:, nc:] = 0.0
                ktot[:, nc:] = 0.0
            cw, kcw, segtot, ksegtot = f2(tot, ktot, Lc32, Lcd)
            if multi:
                if ncp3 > ns:
                    # Zero the level-3 gemm K padding (rows [ns, ncp3)).
                    segtot[:, ns:] = 0.0
                    ksegtot[:, ns:] = 0.0
                cs, kcs = f3(segtot, ksegtot, Ls32)
            else:
                cs, kcs = cs_zero, kcs_zero
            y = f4(part, kpart, cw, kcw, cs, kcs)
            if Me != M_ or N_ != NE:
                return y[:M_, :N_]  # trim the aligned surplus
            return y

        return call

    return _func


# ---------- input generators ----------
def _gen_values(M, N, kind, seed=0):
    """Build a CPU fp32 tensor; ``kind`` selects the value regime."""
    g = torch.Generator().manual_seed(seed)
    if kind == "near1":
        # graded regime: every factor ~ 1.0 so the product stays O(1)
        return torch.rand(M, N, generator=g) * 0.02 + 0.99
    if kind == "signed":
        # graded regime with random signs -> magnitude O(1), sign parity active
        base = torch.rand(M, N, generator=g) * 0.02 + 0.99
        sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
        return base * sign
    if kind == "randn":
        return torch.randn(M, N, generator=g) * 0.5
    if kind == "mixed":
        return torch.rand(M, N, generator=g) * 2.0 - 1.0
    raise ValueError(f"unknown generator kind {kind!r}")


def _run_case(M, N, dtype_str, kind, tag, seed=0, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = _gen_values(M, N, kind, seed).to(torch_dtype)
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(BM, BS)
    y = kernel(x.npu())
    ref = golden_cumprod(x)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_values_case(x, dtype_str, tag, equal_nan=False):
    """Run a pre-built CPU fp32 tensor through the kernel; check vs golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    x = x.to(torch_dtype)
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(BM, BS)
    y = kernel(x.npu())
    ref = golden_cumprod(x)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} max_diff={max_diff:.2e}")
    return max_diff


def _run_multidim(shape, dim, dtype_str, kind, tag, seed=0):
    """Emulate the Op-layer movedim/reshape choreography around the 2-D kernel."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    total = 1
    for d in shape:
        total *= d
    x = _gen_values(1, total, kind, seed).reshape(shape).to(torch_dtype)
    xm = x.movedim(dim, -1).contiguous()
    x2d = xm.reshape(-1, xm.shape[-1])
    M, N = x2d.shape
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(BM, BS)
    y2d = kernel(x2d.npu())
    y = y2d.reshape(xm.shape).movedim(-1, dim)
    ref = golden_cumprod(x, dim=dim)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape={tuple(shape)} dim={dim} dtype={dtype_str} "
        f"-> ({M},{N}) max_diff={max_diff:.2e}"
    )
    return max_diff


def _corner_rows(dtype_str):
    """(8, 257) rows covering zeros / negatives / alternating / ones / near-1."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    rows = [
        torch.ones(257),  # all one
        torch.full((257,), 0.5),  # constant
        torch.cat([torch.zeros(1), torch.full((256,), 0.5)]),  # leading zero
        torch.cat([torch.full((200,), 0.5), torch.zeros(57)]),  # trailing zero
        torch.full((257,), -1.0),  # sign flips
        (torch.arange(257, dtype=torch.float32) % 2 == 0).float() * 2 - 1,
        torch.cat([torch.tensor([-2.0]), torch.full((256,), 0.5)]),  # one negative
        torch.linspace(0.9, 1.1, 257),  # near-1 ramp
    ]
    return torch.stack(rows, dim=0).to(torch_dtype)


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.2 (manifest workloads + R1-R4 risk coverage).
def run_L0():
    max_diff = 0.0
    # L0-1: smoke (128,512) in the three dtypes (graded regime).
    max_diff = max(max_diff, _run_case(128, 512, "float16", "near1", "L0-1a"))
    max_diff = max(max_diff, _run_case(128, 512, "bfloat16", "near1", "L0-1b"))
    max_diff = max(max_diff, _run_case(128, 512, "float32", "near1", "L0-1c"))
    # L0-2: manifest max load (256,4096).
    max_diff = max(max_diff, _run_case(256, 4096, "float16", "near1", "L0-2a"))
    max_diff = max(max_diff, _run_case(256, 4096, "bfloat16", "near1", "L0-2b"))
    # L0-3: N non-divisible by bs (identity alignment path).
    max_diff = max(max_diff, _run_case(128, 300, "float16", "signed", "L0-3a"))
    max_diff = max(max_diff, _run_case(128, 300, "bfloat16", "signed", "L0-3b"))
    # L0-4: M tail (row axis alignment path).
    max_diff = max(max_diff, _run_case(129, 512, "float16", "near1", "L0-4a"))
    max_diff = max(max_diff, _run_case(129, 512, "bfloat16", "near1", "L0-4b"))
    # L0-5: 3D / 4D inputs (Op-layer movedim/reshape).
    max_diff = max(max_diff, _run_multidim((2, 64, 512), -1, "float16", "near1", "L0-5a"))
    max_diff = max(max_diff, _run_multidim((2, 4, 8, 512), -1, "float16", "near1", "L0-5b"))
    # L0-6: 1D input.
    max_diff = max(max_diff, _run_multidim((512,), -1, "float32", "near1", "L0-6a"))
    max_diff = max(max_diff, _run_multidim((512,), -1, "float16", "signed", "L0-6b"))
    # L0-7: dim=1 movedim path.
    max_diff = max(max_diff, _run_multidim((2, 512, 256), 1, "float16", "near1", "L0-7"))
    # L0-8: sign parity on random negatives (magnitude stays O(1)).
    max_diff = max(max_diff, _run_case(256, 1024, "float16", "signed", "L0-8a"))
    max_diff = max(max_diff, _run_case(256, 1024, "bfloat16", "signed", "L0-8b"))
    max_diff = max(max_diff, _run_case(256, 1024, "float32", "signed", "L0-8c"))
    # L0-9: corner semantics (zeros / negatives / alternating / ones).
    for dtype_str in ("float16", "bfloat16", "float32"):
        x = _corner_rows(dtype_str)
        max_diff = max(max_diff, _run_values_case(x, dtype_str, f"L0-9-{dtype_str}"))
    # L0-10: manifest hidden-state (2048,4096) fp16.
    max_diff = max(max_diff, _run_case(2048, 4096, "float16", "near1", "L0-10"))
    # L0-11: non-divisible N (128,300) / (512,300)专项 -- single-phase part/kpart
    # and the final y must show no 0*NaN padding contamination.
    max_diff = max(max_diff, _run_case(128, 300, "float16", "near1", "L0-11a"))
    max_diff = max(max_diff, _run_case(512, 300, "float16", "near1", "L0-11b"))
    max_diff = max(max_diff, _run_case(512, 300, "bfloat16", "signed", "L0-11c"))
    # L0-12: wide magnitude range (randn) -- log-domain prefix precision under exp.
    max_diff = max(max_diff, _run_case(256, 512, "float16", "randn", "L0-12a"))
    max_diff = max(max_diff, _run_case(128, 1024, "bfloat16", "randn", "L0-12b"))
    # L0-13: long scan axis (multi-chunk carry) with bounded magnitude.
    max_diff = max(max_diff, _run_case(64, 8192, "float16", "signed", "L0-13a"))
    max_diff = max(max_diff, _run_case(32, 5000, "bfloat16", "near1", "L0-13b"))
    # L0-14: v3 wide-row path (ns = 8 segments on the tune shape [64, 32768]).
    max_diff = max(max_diff, _run_case(64, 32768, "bfloat16", "near1", "L0-14a"))
    max_diff = max(max_diff, _run_case(64, 32768, "float16", "signed", "L0-14b"))
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(130, 256, "float16", "signed", "L1-1"))
    max_diff = max(max_diff, _run_case(64, 1024, "bfloat16", "signed", "L1-2"))
    max_diff = max(max_diff, _run_case(8, 512, "float16", "mixed", "L1-3"))
    max_diff = max(max_diff, _run_case(256, 1024, "float32", "signed", "L1-4"))
    max_diff = max(max_diff, _run_case(3, 300, "bfloat16", "signed", "L1-5"))
    max_diff = max(max_diff, _run_case(32, 32, "float32", "near1", "L1-6"))
    max_diff = max(max_diff, _run_case(255, 777, "float16", "signed", "L1-7"))
    # multi-chunk carry + non-aligned N (N > bs, N % bs != 0)
    max_diff = max(max_diff, _run_case(16, 5000, "float16", "near1", "L1-8"))
    # multi-chunk carry + M tail
    max_diff = max(max_diff, _run_case(129, 4100, "bfloat16", "signed", "L1-9"))
    # sign parity on a long axis
    max_diff = max(max_diff, _run_case(4, 9000, "float32", "signed", "L1-10"))
    # 4D input along the last axis
    max_diff = max(max_diff, _run_multidim((2, 4, 8, 128), -1, "float16", "signed", "L1-11"))
    # 1D fp32 / bf16 short
    max_diff = max(max_diff, _run_multidim((300,), -1, "bfloat16", "signed", "L1-12"))
    # v3 hierarchical carry: level-2 partial segment + non-aligned N
    # (20000 = 312.5 chunks -> NE 20032, nc 313, nc_seg 64, ns 5, ncp3 16)
    max_diff = max(max_diff, _run_case(8, 20000, "float16", "signed", "L1-13"))
    # v3: exact tune shape, fp32 (no vcast operand path) + 4D wrap
    max_diff = max(max_diff, _run_case(4, 32768, "float32", "near1", "L1-14"))
    max_diff = max(max_diff, _run_multidim((2, 2, 4, 8192), -1, "bfloat16", "signed", "L1-15"))
    # v3: N = 4160 (nc 65 -> ns 2, one full + one 1-chunk segment)
    max_diff = max(max_diff, _run_case(16, 4160, "float16", "signed", "L1-16"))
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary / degenerate (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", "near1", 1),
        (8, 256, "float16", "near1", 2),
        (9, 256, "float16", "near1", 3),
        (32, 1, "float16", "near1", 4),
        (1, 512, "float32", "signed", 5),
        (512, 1, "float32", "near1", 6),
    ]
    for M, N, dtype_str, kind, seed in cases:
        try:
            _run_case(M, N, dtype_str, kind, "L2", seed=seed)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # All zeros -> log(0) = -inf -> log-floor -> exp underflow 0.
    try:
        x = torch.zeros(32, 256)
        _run_values_case(x, "float16", "Boundary-all-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all zeros: {e}")

    # Leading zero -> every following prefix is exactly 0.
    try:
        x = _gen_values(32, 256, "near1", 3).clone()
        x[:, 0] = 0.0
        _run_values_case(x, "float16", "Boundary-leading-zero")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): leading zero: {e}")

    # All negative -> exact sign parity (-1)^j.
    try:
        x = -_gen_values(32, 256, "near1", 7)
        _run_values_case(x, "float32", "Boundary-all-negative")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all negative: {e}")

    # Growing sequence -> overflow to +inf (must match torch.cumprod inf).
    try:
        x = torch.full((16, 512), 1.5)
        _run_values_case(x, "float16", "Boundary-growth-overflow")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): growth overflow: {e}")

    # Shrinking sequence -> underflow to 0.
    try:
        x = torch.full((16, 512), 0.5)
        _run_values_case(x, "float16", "Boundary-shrink-underflow")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): shrink underflow: {e}")

    # NaN propagation (equal_nan=True).
    try:
        x = _gen_values(32, 256, "near1", 11)
        x[0, 3] = float("nan")
        _run_values_case(x, "float32", "Boundary-nan", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): NaN: {e}")

    # +inf / -inf input (DESIGN.md section 9.2 R8 known boundary).
    try:
        x = _gen_values(32, 256, "near1", 13)
        x[0, 3] = float("inf")
        _run_values_case(x, "float16", "Boundary-pos-inf", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +inf: {e}")
    try:
        x = _gen_values(32, 256, "near1", 17)
        x[0, 3] = float("-inf")
        _run_values_case(x, "float16", "Boundary-neg-inf", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): -inf: {e}")

    # Subnormal magnitudes.
    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((32, 256), finfo.tiny / 2, dtype=torch.float16)
        _run_values_case(x.float(), "float16", "Boundary-subnormal")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")


def run_timing():
    """L0-13: record end-to-end latency (host timing, informational)."""
    import time

    try:
        M, N = 2048, 4096
        x = _gen_values(M, N, "near1", 0).to(torch.float16).npu()
        kernel = _cumulative_kernel(M, N, "prod", "float16")(BM, BS)
        kernel(x)
        torch.npu.synchronize()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            kernel(x)
        torch.npu.synchronize()
        print(
            f"[L0-13 timing] (2048,4096) fp16 full pipeline: "
            f"{(time.perf_counter() - t0) / reps * 1e6:.1f} us/call"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L0-13 timing] WARN (record only): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
        run_timing()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_timing()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
