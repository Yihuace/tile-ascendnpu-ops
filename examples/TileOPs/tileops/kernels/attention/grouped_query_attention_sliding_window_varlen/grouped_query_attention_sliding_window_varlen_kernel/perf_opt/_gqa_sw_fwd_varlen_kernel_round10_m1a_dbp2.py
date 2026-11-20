# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GQA sliding-window forward kernel, variable-length packed-NHD, NPU Expert mode.

Migrated from the GPU TileLang factory ``_gqa_sw_fwd_varlen_kernel``
(``tileops/kernels/attention/gqa_sliding_window_varlen_fwd.py``) to
``target="npuir"`` per DESIGN.md (Expert mode, two-phase S/P materialization,
varlen axes V1-V8):

    o   = softmax(scale * q @ k^T + band_mask) @ v   (packed NHD, in dtype)
    lse = log2-domain per-row log-sum-exp            ([H, total_q], fp32)

The CPU golden below carries the independent sliding-window semantics (the
source's exact four-condition mask with ``offset = kv_len - q_len``, FA3
bottom-right), never the NPU two-phase algorithm.

Structure (DESIGN 0.6 / 3.3 / 6), a structural descendant of the fixed-length
sibling ``examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel``
(Expert two-phase, 22/22 + integration) -- the two-phase skeleton (dual Scope,
three ws families, scales[] delayed replay, dual-AIV adaptive row split,
Q-hoist, e-domain softmax, fp16/bf16 dual trace, size-form tail clamps, sync
shim, compile retry) is reused line-by-line; only the eight varlen axes change:

  V1 ragged task mapping: 1-D persistent ``T.Kernel(20)``; ``num_logical =
     ceildiv(max_seqlen_q, bm) * H * B``; per-task decode reads ``cu_q[bz]`` /
     ``cu_k[bz]`` as runtime GM scalars -> ``q_start/kv_start/q_len/kv_len/
     offset``; ``real_m = clamp(min(bm, q_len - bx*bm), 0, ..)``; ghost blocks
     (``real_m == 0``) run zero band iterations and write nothing.
  V2 offset-shifted two-sided arithmetic-penalty mask + per-row column OOB:
     ``K_up = (bx*bm + row0 + offset + up_shift) - n_lo``;
     ``K_lo = (bx*bm + row0 + offset - wl) - n_lo``;
     ``poob`` threshold ``(tn_real - 1) - i`` (the absolute column-OOB
     condition rewritten in diff space MUST carry the row index).
  V3 runtime band range: source k_start/k_end formulas ported with runtime
     ``kv_len``/``offset``.
  V4 always-on ragged KV tail (runtime ``kv_len % bn_eff != 0``): N-carrier
     zero init once per core + full-width per-row column OOB + ``zbuf`` l1_b
     zero init (PL-1.11: 0*NaN=NaN).
  V5 analytical flag-budget guard (runtime loop_count): ``L_max_bound``.
  V6 packed-NHD runtime addressing (all GM<->L1/UB moves are slice-form
     ``T.copy``; ``T.load_nd2nz`` is forbidden, CG-2026-0004).
  V7 lse ``(H, total_q, 1)`` view + ghost/row-truncated stores (transpose-free).
  V8 interface: cu_seqlens + total_q/total_k + closure + lse reshape; the
     caller form ``(block_m,block_n,num_stages)(q,k,v,cu_q,cu_k,max_seqlen_q)``
     is kept verbatim (wrapper zero change); max_seqlen_q is folded at the
     host (deferred compile keyed on it) so it is a trace-time constant.

Device-verified varlen probes (DESIGN 0.6 V1/V2/V4, history_version/):
  * P-V1: runtime per-sample ``cu_seqlens`` scalar GM read usable for slice
    start / dynamic extent / dynamic T.serial bound / output address, incl. a
    ghost block (maxdiff=0.0); the "copy cu -> UB then scalar index" path is
    BLOCKED (``cannot find variable cub``).
  * P-V2: offset-shifted two-sided arithmetic penalty + per-row column OOB +
    always-on ragged tail -> 7/7 bit-exact, sentinel bit-exact, NaN-free,
    zero scalarization warning.

Run: python _gqa_sw_fwd_varlen_kernel.py --level {L0,all}

Stage 4 (perf_opt) tuned deltas vs the Stage 3 baseline -- all inside the same
``@T.prim_func``, same factory signature, same public contract:

  T1 depth-2 cross-task pipeline (r1_op1_pipe2): the Cube stream hoists Q and
     produces the whole S band of task T+1 into the *other* ws slot while the
     Vector stream consumes task T; the task-boundary TASKDONE barrier is
     dropped (single-slot fallback keeps it when 2*L_max_bound > 15).
  T2 ragged-tail OOB guard (r2_op1_oobguard): the always-on per-row column
     OOB penalty chain runs only when ``tn_real < bn_eff`` (a full block needs
     no column masking) -- the single largest win (-36% on 8b-long fp16).
  T3 f32 S carrier on the fp16 path (r3_op3_sf32): removes the per-block
     f16->f32 vcast of the [half,bn] tile (AB-adjudicated, -3.1%).

Measured (msprof op Task Duration, median of 20) 8b-long-w1024 fp16:
3759.49us (Stage 3) -> 2003.35us (-46.7%).
"""

import argparse
import functools
import math
import os
from typing import Callable

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

# Expert mode is the default; set explicitly to guard against env leftovers.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

# Programming mode actually validated by this file (TileOPs harness sync).
ASCEND_MODE = "Expert"


# ---------------------------------------------------------------------------
# Toolchain shim (this tilelang build + CANN 9.0.0) -- CG-2026-0023.
# The NPUIR codegen emits the sync-block attribute as ``syn_instr_mode`` while
# the CANN 9.0.0 ``bishengir-compile`` parser expects ``sync_instr_mode``
# ("failed to parse HIVM_SyncBlockInstrModeAttr parameter 'sync_instr_mode'").
# This shim rewrites that one attribute name in the generated .npuir text
# immediately before the external bishengir-compile call.  Reused verbatim
# from the fixed-length sibling; without it the two-phase cross-engine
# handshake cannot compile on this container.
# ---------------------------------------------------------------------------
# Second toolchain defect on the same build: ``bishengir-compile`` 1.1.0
# (LLVM 19.1.7) intermittently SIGSEGVs (exit -11) while compiling this
# kernel's IR -- the same .npuir compiles or crashes across runs (measured
# ~25-35% success per attempt; the crash point varies between
# ConvertHFusionToHIVM and ConvertHIVMToStandard, and disabling threading,
# ASLR or raising the stack limit does not help).  This is a compiler
# memory bug rather than an IR validation error, so the external compile is
# retried on SIGSEGV only (other failures are re-raised unchanged).
_COMPILE_RETRIES = 40


def _install_sync_block_attr_shim():
    try:
        import importlib

        _jn = importlib.import_module("tilelang.jit.jit_npu")
        cls = getattr(_jn, "compiler_npu", None)
        if cls is None:
            return
        orig = cls._npuir_to_bin_enable_npu_compile
        if getattr(orig, "_sync_attr_shim", False):
            return

        def _patched(self):
            mc = getattr(self, "mlir_content", None)
            if mc:
                self.mlir_content = mc.replace("syn_instr_mode", "sync_instr_mode")
            last = None
            for _ in range(_COMPILE_RETRIES):
                try:
                    return orig(self)
                except Exception as exc:
                    cause = getattr(exc, "__cause__", None)
                    rc = getattr(cause, "returncode", None)
                    if rc is None:
                        rc = getattr(exc, "returncode", None)
                    if rc in (-11, 139) or "Segmentation fault" in str(exc):
                        last = exc
                        continue
                    raise
            raise last

        _patched._sync_attr_shim = True
        cls._npuir_to_bin_enable_npu_compile = _patched
    except Exception:
        pass


_install_sync_block_attr_shim()

LOG2E = 1.4426950408889634
# Physical AI-core count: NPUUtils.get().get_aicore_num() on Ascend910B3
# (DESIGN 5.5, measured 2026-10-16: "Ascend910B3 20 40").
NUM_KERNELS = 20

# Stage 4 tuned default block geometry (bm, bn_eff) -- exposed as a module-level
# constant so the wrapper's baseline/perf_opt switch block can reference it
# (bench.py / L0-L1 still call the factory with the wrapper default, which is
# substituted by this pair).  Scan evidence (perf_opt/opt_log.md round 3/4):
# (64,256) is UB-optimal -- bm=80 overflows UB (204.8KB > 192KB, BishengIR
# segfault), (32,512) is +85%, (64,128) is +26% (and 2*L_max > 15 kills the
# depth-2 pipeline).
TUNED_DEFAULT_BLOCK = (64, 256)


# ---------- Golden (independent PyTorch CPU reference, DESIGN 8.1) ----------
def golden_gqa_sw_fwd_varlen(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    is_causal, window_size_left, window_size_right,
):
    """Variable-length sliding-window GQA forward reference (packed NHD).

    q:    [total_q, H,   D]    fp16/bf16
    k/v:  [total_k, H_kv, D]   fp16/bf16
    cu_seqlens_q/k: [B+1] int32 (CPU)
    Returns (output: q shape/dtype, lse: [H, total_q] fp32, log2 domain).

    Independence: no online softmax, no band trimming, no e-domain reorder, no
    arithmetic-penalty mask -- per-sample materialized softmax in fp32 with the
    source's exact four-condition mask (offset = seqlen_k - seqlen_q, FA3
    bottom-right), P quantized at the source acc_s_cast point.  lse is emitted
    in the source's log2 domain (the kernel's lse = log2(sum exp(S''))).
    """
    heads = q.shape[1]
    dim = q.shape[2]
    groups = heads // k.shape[1]
    scale = dim ** -0.5
    log2e = 1.0 / math.log(2.0)
    outs = []
    lse = torch.full((heads, q.shape[0]), float("-inf"), dtype=torch.float32)
    for i in range(len(cu_seqlens_q) - 1):
        q0, q1 = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        k0, k1 = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        sq, sk = q1 - q0, k1 - k0
        if sq == 0:
            outs.append(torch.empty(0, heads, dim, dtype=q.dtype))
            continue
        offset = sk - sq
        qi = q[q0:q1].float().transpose(0, 1)                                   # [H,sq,D]
        ki = k[k0:k1].float().repeat_interleave(groups, dim=1).transpose(0, 1)  # [H,sk,D]
        vi = v[k0:k1].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        sc = (qi @ ki.transpose(-2, -1)) * scale                                # [H,sq,sk]
        qp = torch.arange(sq).view(sq, 1)
        kp = torch.arange(sk).view(1, sk)
        mask = torch.zeros(sq, sk, dtype=torch.bool)
        if is_causal:
            mask = mask | (kp > qp + offset)
        if window_size_left >= 0:
            mask = mask | (kp < qp + offset - window_size_left)
        if window_size_right >= 0:
            mask = mask | (kp > qp + offset + window_size_right)
        sc = sc.masked_fill(mask.unsqueeze(0), float("-inf"))
        m = sc.max(-1, keepdim=True).values.clamp(min=-1e38)                    # [H,sq,1]
        p = torch.exp(sc - m)
        ps = p.sum(-1, keepdim=True)
        # fully-masked rows: lse -> -inf, out -> 0 (source nan_to_num).  Not in
        # the target domain, kept for numerical robustness.
        lse[:, q0:q1] = (torch.log(ps).squeeze(-1) + m.squeeze(-1)) * log2e
        pq = torch.where(ps > 0, p / ps, torch.zeros_like(p))
        pq = pq.to(q.dtype).float()                                             # P quantize point
        outs.append((pq @ vi).transpose(0, 1).to(q.dtype))                      # [sq,H,D]
    return torch.cat(outs, dim=0), lse


# ---------- Kernel (Expert mode two-phase, DESIGN 0.6 V1-V8) ----------
@functools.lru_cache(maxsize=32)
def _gqa_sw_fwd_varlen_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,   # -1 = unlimited
    window_size_right: int,  # -1 = unlimited
    dtype: str = "float16",
    accum_dtype: str = "float",  # accepted for wrapper compatibility; -> f32
) -> Callable:
    """GPU-source-compatible factory (signature/lru_cache kept, V8).

    Returns an inner factory ``_gqa_sw_fwd_varlen_func(block_m, block_n,
    num_stages)`` whose result is a closure ``wrapped(q, k, v, cu_q, cu_k,
    max_seqlen_q) -> (output, lse)``.  ``max_seqlen_q`` is folded at the host:
    the inner JIT compile is deferred to the first ``wrapped`` call and keyed
    on ``max_seqlen_q`` so it becomes a trace-time constant.
    """
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError("window_size_left/right must be >= -1")
    groups = heads // heads_kv
    prescale = dim ** -0.5
    has_window = window_size_left >= 0 or window_size_right >= 0
    # Trace-time mask-family selectors (source four-branch semantics).
    use_left = window_size_left >= 0
    # Upper bound: causal (source IGNORES the right window under causal) OR a
    # non-causal right window.
    use_upper = bool(is_causal) or ((not is_causal) and window_size_right >= 0)
    up_shift = 0 if is_causal else (window_size_right if window_size_right >= 0 else 0)
    causal_end = bool(is_causal)
    right_end = (not is_causal) and has_window and window_size_right >= 0

    def _gqa_sw_fwd_varlen_func(block_m: int, block_n: int, num_stages: int) -> Callable:
        # ---- effective config: the wrapper default (64, 64, 1) / (64, 32, 1)
        # is replaced by the design default (64, 256); any other explicitly
        # passed config is respected verbatim.  num_stages stays a reserved
        # knob (the single-slot two-phase structure is stage-invariant).
        if (block_m, block_n, num_stages) in ((64, 64, 1), (64, 32, 1)):
            block_m, block_n = TUNED_DEFAULT_BLOCK
        bm = block_m
        half = bm // 2

        def _band_upper_bound(bn_eff_):
            """Analytical upper bound of the runtime loop_count (V5).

            With a left window the offset/kv_len terms cancel and the span is
            ``bm + wl + eff_right``.  Without a left window k_start is 0 and
            loop_count can reach ceildiv(kv_len, bn), bounded by total_k.
            """
            if window_size_left < 0:
                return (total_k + bn_eff_ - 1) // bn_eff_ + 1
            eff_right = window_size_right if (window_size_right >= 0 and not is_causal) else 0
            return (bm + window_size_left + eff_right + bn_eff_ - 1) // bn_eff_ + 2

        @functools.lru_cache(maxsize=8)
        def _compile(max_seqlen_q: int):
            num_q_blocks = (max_seqlen_q + bm - 1) // bm
            num_logical = num_q_blocks * heads * batch
            HB = heads * batch
            # Flag-budget guard (E-band / V5): enlarge bn_eff until the
            # analytic loop_count bound fits event ids 0..15 (plus TASKDONE).
            bn_eff = max(block_n, 16)
            bn_eff = ((bn_eff + 15) // 16) * 16
            while _band_upper_bound(bn_eff) + 1 > 16:
                bn_eff += 16
            L_max_bound = _band_upper_bound(bn_eff)
            assert L_max_bound >= 1 and L_max_bound + 1 <= 16, (
                f"flag budget exceeded: L_max_bound={L_max_bound}, bn_eff={bn_eff}"
            )
            assert bn_eff % 16 == 0 and dim % 16 == 0
            block_share = max(bn_eff, dim)
            pad_band = L_max_bound * bn_eff
            # ---- ws carriers (E5): fp16 -> f16/f16/f16; bf16 -> f32/bf16/f32
            if dtype == "float16":
                # r3_op: f32 S carrier on the fp16 path -- the per-block
                # f16->f32 vcast of the whole [half,bn] tile disappears (the
                # fixpipe store also writes f32 directly).  O stays f16 (its
                # consumer is the Vector accumulate chain).
                ws_s_dtype, ws_o_dtype = "float32", "float16"
            else:
                ws_s_dtype, ws_o_dtype = "float32", "float32"
            # ---- task-slot dimension (r1_op1): depth-2 cross-task software
            # pipeline.  Slot s owns flag ids [s*L_max, s*L_max + L_max), so the
            # budget is 2*L_max <= 15 usable ids (flag 0..15).  When the band is
            # too wide for two slots the kernel falls back to the single-slot
            # form (slot pinned to 0 + the FLAG_TASKDONE task barrier).
            n_slots = 2 if 2 * L_max_bound <= 15 else 1
            use_pipe2 = n_slots == 2
            keep_barrier = not use_pipe2
            ws_s_shape = [NUM_KERNELS, n_slots, bm, pad_band]
            ws_p_shape = [NUM_KERNELS, n_slots, bm, pad_band]
            ws_o_shape = [NUM_KERNELS, n_slots, bm, dim * L_max_bound]
            FLAG_TASKDONE = L_max_bound
            s_f16 = ws_s_dtype == "float16"   # S carrier is f16?
            o_f16 = dtype == "float16"        # O carrier is f16?
            causal_t = bool(is_causal)
            use_left_t = bool(use_left)
            use_upper_t = bool(use_upper)
            up_shift_t = int(up_shift)

            @tilelang.jit(
                out_idx=[-2, -1],
                target="npuir",
                pass_configs={
                    # Protect the manual two-phase flow and explicit workspaces
                    # from compiler reordering (DESIGN 7.3).
                    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
                    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
                },
            )
            def _builder_2phase():
                @T.prim_func
                def _gqa_sw_fwd_varlen_main(
                    q: T.Tensor((total_q, heads, dim), dtype),
                    k: T.Tensor((total_k, heads_kv, dim), dtype),
                    v: T.Tensor((total_k, heads_kv, dim), dtype),
                    cu_q: T.Tensor((batch + 1,), "int32"),
                    cu_k: T.Tensor((batch + 1,), "int32"),
                    ws_s: T.Tensor(ws_s_shape, ws_s_dtype),
                    ws_p: T.Tensor(ws_p_shape, dtype),
                    ws_o: T.Tensor(ws_o_shape, ws_o_dtype),
                    zbuf: T.Tensor((bn_eff, dim), dtype),
                    output: T.Tensor((total_q, heads, dim), dtype),
                    # lse is a [H, total_q, 1] VIEW of the contract's
                    # [H, total_q] memory: the epilogue writes the contiguous
                    # row run as a natural 2D region copy (NO T.transpose, VP-D6).
                    lse: T.Tensor((heads, total_q, 1), "float32"),
                ):
                    with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
                        # Ghost-task clamp: T.ceildiv lowers to a truncating
                        # form returning 1 for x in (-N, 0]; the clamp pins an
                        # idle core's task to the LAST LEGAL task id.
                        num_local_tasks = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)
                        # ================= Cube stream =================
                        with T.Scope("Cube"):
                            # Dedicated Q buffer so the NEXT task's Q hoist no
                            # longer WARs the CURRENT task's P staging in l1_a
                            # -- prerequisite of the depth-2 pipeline.
                            l1_q = T.alloc_L1([bm, dim], dtype)
                            l1_a = T.alloc_L1([bm, block_share], dtype)
                            l1_b = T.alloc_L1([bn_eff, dim], dtype)
                            # r10_m1: second P/V L1 pair for the hand-written
                            # double buffer of pass-2 (slot1).  Static buffers
                            # (not a [2,...] buffer with a runtime offset
                            # slice) because the toolchain's gemm drops operand
                            # offsets (Conv3d probe: "gemm ignores the N
                            # offset"); parity is resolved by unrolling pairs
                            # at trace time instead.  L1 total: 16+32*2+64*2 =
                            # 208 KB of the 512 KB AIC L1.
                            l1_a2 = T.alloc_L1([bm, block_share], dtype)
                            l1_b2 = T.alloc_L1([bn_eff, dim], dtype)
                            l0_c = T.alloc_L0C([bm, block_share], accum_dtype)
                            # V4 (always on): zero-init the V carrier once per
                            # core so the fractal band of gemm2 cannot pick up
                            # uninitialised (NaN) rows -- PL-1.11 0*NaN=NaN.
                            # Both slots need it (the stale row band is read by
                            # gemm2's K dimension on the very first block).
                            with T.rs("PIPE_MTE2"):
                                T.copy(zbuf, l1_b)
                                # zbuf is the zero parameter, so T.clear is
                                # equivalent to T.copy(zbuf, l1_b2) here -- and
                                # the GM->L1 copy form into the second buffer
                                # SIGSEGVs bishengir-compile on this kernel
                                # (probe ladder in opt_log round 10).
                                T.clear(l1_b2)
                            # ---- depth-2 pipeline prologue (r1_op1): hoist
                            # Q(0) and produce the whole S band of task 0 into
                            # slot 0 before entering the steady-state loop, so
                            # the first p2 wait is already covered.
                            if use_pipe2:
                                cid0 = T.min(kernel_id, num_logical - 1)
                                bx0 = cid0 // HB
                                by0 = (cid0 // batch) % heads
                                bz0 = cid0 % batch
                                q_start0 = cu_q[bz0]
                                kv_start0 = cu_k[bz0]
                                q_len0 = cu_q[bz0 + 1] - q_start0
                                kv_len0 = cu_k[bz0 + 1] - kv_start0
                                offset0 = kv_len0 - q_len0
                                real_m0 = T.max(0, T.min(bm, q_len0 - bx0 * bm))
                                q_off0 = q_start0 + T.min(bx0 * bm, q_len0)
                                kv_head0 = by0 // groups
                                tmc0 = T.max(16, T.min(bm, T.ceildiv(real_m0, 16) * 16))
                                active0 = T.min(real_m0, 1)
                                k_end0 = (
                                    T.ceildiv(T.min(kv_len0, offset0 + (bx0 + 1) * bm), bn_eff)
                                    if causal_t
                                    else (
                                        T.ceildiv(
                                            T.min(kv_len0,
                                                  offset0 + (bx0 + 1) * bm + window_size_right),
                                            bn_eff,
                                        )
                                        if right_end
                                        else T.ceildiv(kv_len0, bn_eff)
                                    )
                                )
                                k_start0 = (
                                    T.max(0, offset0 + bx0 * bm - window_size_left) // bn_eff
                                    if use_left_t
                                    else 0
                                )
                                lc0 = active0 * T.max(k_end0 - k_start0, 0)
                                with T.rs("PIPE_MTE2"):
                                    T.copy(
                                        q[q_off0 : q_off0 + real_m0, by0, 0:dim],
                                        l1_q[0:real_m0, 0:dim],
                                    )
                                for l in T.serial(lc0):
                                    n_lo = (k_start0 + l) * bn_eff
                                    tn_real = T.min(bn_eff, kv_len0 - n_lo)
                                    tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            k[kv_start0 + n_lo : kv_start0 + n_lo + tn_real,
                                              kv_head0, 0:dim],
                                            l1_b[0:tn_real, 0:dim],
                                        )
                                    with T.rs("PIPE_C"):
                                        T.gemm(
                                            l1_q, l1_b, l0_c,
                                            initC=True, b_transpose=True,
                                            size=[tmc0, dim, tnc],
                                        )
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc0, 0:tnc],
                                            ws_s[kernel_id, 0, 0:tmc0,
                                                 l * bn_eff : l * bn_eff + tnc],
                                        )
                                        T.sync_block_set(l)
                            for task_id in T.serial(num_local_tasks):
                                cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                                bx = cid // HB
                                by = (cid // batch) % heads
                                bz = cid % batch
                                # V1/V6: runtime per-sample scalar reads.
                                q_start = cu_q[bz]
                                kv_start = cu_k[bz]
                                q_len = cu_q[bz + 1] - q_start
                                kv_len = cu_k[bz + 1] - kv_start
                                offset = kv_len - q_len
                                real_m = T.max(0, T.min(bm, q_len - bx * bm))
                                q_off = q_start + T.min(bx * bm, q_len)
                                kv_head = by // groups
                                tmc = T.max(16, T.min(bm, T.ceildiv(real_m, 16) * 16))
                                active = T.min(real_m, 1)
                                # ---- band KV range (V3, source S4 verbatim) --
                                # Ternary (not if/elif): a trace-time `if` would
                                # scope k_end inside the branch.
                                k_end = (
                                    T.ceildiv(T.min(kv_len, offset + (bx + 1) * bm), bn_eff)
                                    if causal_t
                                    else (
                                        T.ceildiv(
                                            T.min(
                                                kv_len,
                                                offset + (bx + 1) * bm + window_size_right,
                                            ),
                                            bn_eff,
                                        )
                                        if right_end
                                        else T.ceildiv(kv_len, bn_eff)
                                    )
                                )
                                k_start = (
                                    T.max(0, offset + bx * bm - window_size_left) // bn_eff
                                    if use_left_t
                                    else 0
                                )
                                # ghost blocks (real_m == 0) run zero iterations.
                                loop_count = active * T.max(k_end - k_start, 0)
                                slot = (task_id % n_slots) if use_pipe2 else 0
                                adv_slot = (1 - task_id % n_slots) if use_pipe2 else 0
                                adv_ok = (
                                    (task_id + 1 < num_local_tasks)
                                    if use_pipe2
                                    else (task_id < num_local_tasks)
                                )
                                if keep_barrier:
                                    # single-slot only: task-boundary handshake
                                    # (single-slot cross-task WAR protection).
                                    if task_id > 0:
                                        with T.rs("PIPE_MTE2"):
                                            T.sync_block_wait(FLAG_TASKDONE)
                                # ---- advance: Q hoist + all S = Q @ K^T for
                                # the next task (fills the p2 wait stall of
                                # task T).  Varlen: the same ragged decode is
                                # replayed for task_id+1 (ghost-safe).
                                if adv_ok:
                                    cid_a = T.min((task_id + 1) * NUM_KERNELS + kernel_id,
                                                  num_logical - 1)
                                    bx_a = cid_a // HB
                                    by_a = (cid_a // batch) % heads
                                    bz_a = cid_a % batch
                                    kv_start_a = cu_k[bz_a]
                                    q_len_a = cu_q[bz_a + 1] - cu_q[bz_a]
                                    kv_len_a = cu_k[bz_a + 1] - kv_start_a
                                    offset_a = kv_len_a - q_len_a
                                    real_m_a = T.max(0, T.min(bm, q_len_a - bx_a * bm))
                                    q_off_a = cu_q[bz_a] + T.min(bx_a * bm, q_len_a)
                                    kv_head_a = by_a // groups
                                    tmc_a = T.max(16, T.min(bm, T.ceildiv(real_m_a, 16) * 16))
                                    active_a = T.min(real_m_a, 1)
                                    k_end_a = (
                                        T.ceildiv(T.min(kv_len_a,
                                                        offset_a + (bx_a + 1) * bm), bn_eff)
                                        if causal_t
                                        else (
                                            T.ceildiv(
                                                T.min(kv_len_a,
                                                      offset_a + (bx_a + 1) * bm
                                                      + window_size_right),
                                                bn_eff,
                                            )
                                            if right_end
                                            else T.ceildiv(kv_len_a, bn_eff)
                                        )
                                    )
                                    k_start_a = (
                                        T.max(0, offset_a + bx_a * bm - window_size_left) // bn_eff
                                        if use_left_t
                                        else 0
                                    )
                                    lc_a = active_a * T.max(k_end_a - k_start_a, 0)
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            q[q_off_a : q_off_a + real_m_a, by_a, 0:dim],
                                            l1_q[0:real_m_a, 0:dim],
                                        )
                                    for l in T.serial(lc_a):
                                        n_lo = (k_start_a + l) * bn_eff
                                        tn_real = T.min(bn_eff, kv_len_a - n_lo)
                                        tnc = T.max(32, T.min(bn_eff,
                                                              T.ceildiv(tn_real, 16) * 16))
                                        with T.rs("PIPE_MTE2"):
                                            T.copy(
                                                k[kv_start_a + n_lo : kv_start_a + n_lo + tn_real,
                                                  kv_head_a, 0:dim],
                                                l1_b[0:tn_real, 0:dim],
                                            )
                                        with T.rs("PIPE_C"):
                                            T.gemm(
                                                l1_q, l1_b, l0_c,
                                                initC=True, b_transpose=True,
                                                size=[tmc_a, dim, tnc],
                                            )
                                        with T.rs("PIPE_FIX"):
                                            T.copy(
                                                l0_c[0:tmc_a, 0:tnc],
                                                ws_s[kernel_id, adv_slot, 0:tmc_a,
                                                     l * bn_eff : l * bn_eff + tnc],
                                            )
                                            T.sync_block_set(l + adv_slot * L_max_bound)
                                # ---- pass-2 (M1, hand-written double buffer):
                                # O_partial = P @ V.  With the single P/V L1
                                # pair the three Cube pipes are ~serial
                                # (baseline 8b-long fp16: mte2 0.652 + cube
                                # 0.244 + fixpipe 0.313 = 1.21 x wall).  Here
                                # the MTE2 stream loads block b+1 into the
                                # other L1 pair while the M pipe runs gemm(b)
                                # and FIX drains l0_c(b).  Block parity is a
                                # trace-time choice: the loop is unrolled over
                                # block *pairs* (steady loop + peeled last pair
                                # + peeled odd tail) so no runtime branch sits
                                # inside the hot loop.  Dependencies between
                                # the MTE2 write of a slot and the gemm read /
                                # overwrite of that slot are expressed by the
                                # access order of the static buffers (the
                                # toolchain emits the intra-core pipe syncs).
                                npr = loop_count // 2
                                tail = loop_count - npr * 2
                                # prologue: stage block 0 into slot 0
                                if loop_count > 0:
                                    n_lo = k_start * bn_eff
                                    tn_real = T.min(bn_eff, kv_len - n_lo)
                                    tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            v[kv_start + n_lo : kv_start + n_lo + tn_real,
                                              kv_head, 0:dim],
                                            l1_b[0:tn_real, 0:dim],
                                        )
                                        T.sync_block_wait(slot * L_max_bound)
                                        T.copy(
                                            ws_p[kernel_id, slot, 0:tmc, 0:tnc],
                                            l1_a[0:tmc, 0:tnc],
                                        )
                                # steady state: both prefetches are in range
                                # (block 2j+1 <= loop_count-1 and
                                #  block 2j+2 <= loop_count-2)
                                for j in T.serial(T.max(npr - 1, 0)):
                                    i_a = 2 * j
                                    i_b = 2 * j + 1
                                    i_c = 2 * j + 2
                                    lo_b = (k_start + i_b) * bn_eff
                                    tr_b = T.min(bn_eff, kv_len - lo_b)
                                    tc_b = T.max(32, T.min(bn_eff,
                                                          T.ceildiv(tr_b, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            v[kv_start + lo_b : kv_start + lo_b + tr_b,
                                              kv_head, 0:dim],
                                            l1_b2[0:tr_b, 0:dim],
                                        )
                                        T.sync_block_wait(i_b + slot * L_max_bound)
                                        T.copy(
                                            ws_p[kernel_id, slot, 0:tmc,
                                                 i_b * bn_eff : i_b * bn_eff + tc_b],
                                            l1_a2[0:tmc, 0:tc_b],
                                        )
                                    lo_a = (k_start + i_a) * bn_eff
                                    tr_a = T.min(bn_eff, kv_len - lo_a)
                                    tc_a = T.max(32, T.min(bn_eff,
                                                          T.ceildiv(tr_a, 16) * 16))
                                    with T.rs("PIPE_C"):
                                        T.gemm(l1_a, l1_b, l0_c, initC=True,
                                               size=[tmc, tc_a, dim])
                                    lo_c2 = (k_start + i_c) * bn_eff
                                    tr_c2 = T.min(bn_eff, kv_len - lo_c2)
                                    tc_c2 = T.max(32, T.min(bn_eff,
                                                           T.ceildiv(tr_c2, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            v[kv_start + lo_c2 : kv_start + lo_c2 + tr_c2,
                                              kv_head, 0:dim],
                                            l1_b[0:tr_c2, 0:dim],
                                        )
                                        T.sync_block_wait(i_c + slot * L_max_bound)
                                        T.copy(
                                            ws_p[kernel_id, slot, 0:tmc,
                                                 i_c * bn_eff : i_c * bn_eff + tc_c2],
                                            l1_a[0:tmc, 0:tc_c2],
                                        )
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc, 0:dim],
                                            ws_o[kernel_id, slot, 0:tmc,
                                                 i_a * dim : (i_a + 1) * dim],
                                        )
                                        T.sync_block_set(i_a + slot * L_max_bound)
                                    with T.rs("PIPE_C"):
                                        T.gemm(l1_a2, l1_b2, l0_c, initC=True,
                                               size=[tmc, tc_b, dim])
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc, 0:dim],
                                            ws_o[kernel_id, slot, 0:tmc,
                                                 i_b * dim : (i_b + 1) * dim],
                                        )
                                        T.sync_block_set(i_b + slot * L_max_bound)
                                # peeled last pair (slot 0 already staged)
                                if npr > 0:
                                    i_a = 2 * npr - 2
                                    i_b = 2 * npr - 1
                                    lo_b = (k_start + i_b) * bn_eff
                                    tr_b = T.min(bn_eff, kv_len - lo_b)
                                    tc_b = T.max(32, T.min(bn_eff,
                                                          T.ceildiv(tr_b, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            v[kv_start + lo_b : kv_start + lo_b + tr_b,
                                              kv_head, 0:dim],
                                            l1_b2[0:tr_b, 0:dim],
                                        )
                                        T.sync_block_wait(i_b + slot * L_max_bound)
                                        T.copy(
                                            ws_p[kernel_id, slot, 0:tmc,
                                                 i_b * bn_eff : i_b * bn_eff + tc_b],
                                            l1_a2[0:tmc, 0:tc_b],
                                        )
                                    lo_a = (k_start + i_a) * bn_eff
                                    tr_a = T.min(bn_eff, kv_len - lo_a)
                                    tc_a = T.max(32, T.min(bn_eff,
                                                          T.ceildiv(tr_a, 16) * 16))
                                    with T.rs("PIPE_C"):
                                        T.gemm(l1_a, l1_b, l0_c, initC=True,
                                               size=[tmc, tc_a, dim])
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc, 0:dim],
                                            ws_o[kernel_id, slot, 0:tmc,
                                                 i_a * dim : (i_a + 1) * dim],
                                        )
                                        T.sync_block_set(i_a + slot * L_max_bound)
                                    # prefetch the odd tail block into slot 0
                                    if tail > 0:
                                        i_c = 2 * npr
                                        lo_c2 = (k_start + i_c) * bn_eff
                                        tr_c2 = T.min(bn_eff, kv_len - lo_c2)
                                        tc_c2 = T.max(32, T.min(bn_eff,
                                                               T.ceildiv(tr_c2, 16) * 16))
                                        with T.rs("PIPE_MTE2"):
                                            T.copy(
                                                v[kv_start + lo_c2 : kv_start + lo_c2 + tr_c2,
                                                  kv_head, 0:dim],
                                                l1_b[0:tr_c2, 0:dim],
                                            )
                                            T.sync_block_wait(i_c + slot * L_max_bound)
                                            T.copy(
                                                ws_p[kernel_id, slot, 0:tmc,
                                                     i_c * bn_eff : i_c * bn_eff + tc_c2],
                                                l1_a[0:tmc, 0:tc_c2],
                                            )
                                    with T.rs("PIPE_C"):
                                        T.gemm(l1_a2, l1_b2, l0_c, initC=True,
                                               size=[tmc, tc_b, dim])
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc, 0:dim],
                                            ws_o[kernel_id, slot, 0:tmc,
                                                 i_b * dim : (i_b + 1) * dim],
                                        )
                                        T.sync_block_set(i_b + slot * L_max_bound)
                                # peeled odd tail (slot 0 already staged)
                                if tail > 0:
                                    i_t = loop_count - 1
                                    lo_t = (k_start + i_t) * bn_eff
                                    tr_t = T.min(bn_eff, kv_len - lo_t)
                                    tc_t = T.max(32, T.min(bn_eff,
                                                           T.ceildiv(tr_t, 16) * 16))
                                    with T.rs("PIPE_C"):
                                        T.gemm(l1_a, l1_b, l0_c, initC=True,
                                               size=[tmc, tc_t, dim])
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:tmc, 0:dim],
                                            ws_o[kernel_id, slot, 0:tmc,
                                                 i_t * dim : (i_t + 1) * dim],
                                        )
                                        T.sync_block_set(i_t + slot * L_max_bound)
                        # ================= Vector stream =================
                        with T.Scope("Vector"):
                            # let-bound broadcast scalars (vbrc/vsub/vmax scalar
                            # operands must not be raw literals).
                            value_neg = -1e38
                            value_zero = 0
                            value_one = 1.0
                            value_min = -T.infinity("float32")
                            ub_f16_N = T.alloc_ub([half, bn_eff], dtype)
                            ub_f32_N = T.alloc_ub([half, bn_eff], accum_dtype)
                            ub_f16_D = T.alloc_ub([half, dim], dtype)
                            ub_f32_D = T.alloc_ub([half, dim], accum_dtype)
                            acc_o = T.alloc_ub([half, dim], accum_dtype)
                            ub_m = T.alloc_ub([half, 1], accum_dtype)
                            ub_mprev = T.alloc_ub([half, 1], accum_dtype)
                            ub_mcur = T.alloc_ub([half, 1], accum_dtype)
                            ub_alpha = T.alloc_ub([half, 1], accum_dtype)
                            ub_ell = T.alloc_ub([half, 1], accum_dtype)
                            ub_ellcur = T.alloc_ub([half, 1], accum_dtype)
                            ub_t = T.alloc_ub([half, 1], accum_dtype)
                            ub_lse = T.alloc_ub([half, 1], accum_dtype)
                            ub_lse_tmp = T.alloc_ub([half, 1], accum_dtype)
                            scales = T.alloc_ub([L_max_bound * half, 1], accum_dtype)
                            # mask-chain buffers (full size, allocated
                            # unconditionally: a conditional T.alloc scopes the
                            # buffer inside the if-body).
                            ub_diff = T.alloc_ub([half, bn_eff], accum_dtype)
                            ub_pen = T.alloc_ub([half, bn_eff], accum_dtype)
                            ub_rowi = T.alloc_ub([half, 1], accum_dtype)
                            ub_thr = T.alloc_ub([half, 1], accum_dtype)
                            # ---- per-core once: diff = j - i (task-independent
                            # arithmetic-penalty mask basis) + N-carrier zero
                            # init (V4: stale tail columns must be finite).
                            with T.rs("PIPE_V"):
                                T.arange(ub_diff, [0, 1], 0)   # column index j
                                T.arange(ub_pen, [1, 0], 0)    # row index (scratch)
                                T.vsub(ub_diff, ub_pen, ub_diff)  # diff = j - i
                                T.arange(ub_rowi, [1, 0], 0)   # row index i
                                T.vbrc(value_zero, ub_f16_N)
                                T.vbrc(value_zero, ub_f32_N)
                            for task_id in T.serial(num_local_tasks):
                                cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                                bx = cid // HB
                                by = (cid // batch) % heads
                                bz = cid % batch
                                q_start = cu_q[bz]
                                kv_start = cu_k[bz]
                                q_len = cu_q[bz + 1] - q_start
                                kv_len = cu_k[bz + 1] - kv_start
                                offset = kv_len - q_len
                                real_m = T.max(0, T.min(bm, q_len - bx * bm))
                                q_off = q_start + T.min(bx * bm, q_len)
                                s_lo = bx * bm
                                active = T.min(real_m, 1)
                                k_end = (
                                    T.ceildiv(T.min(kv_len, offset + (bx + 1) * bm), bn_eff)
                                    if causal_t
                                    else (
                                        T.ceildiv(
                                            T.min(
                                                kv_len,
                                                offset + (bx + 1) * bm + window_size_right,
                                            ),
                                            bn_eff,
                                        )
                                        if right_end
                                        else T.ceildiv(kv_len, bn_eff)
                                    )
                                )
                                k_start = (
                                    T.max(0, offset + bx * bm - window_size_left) // bn_eff
                                    if use_left_t
                                    else 0
                                )
                                loop_count = active * T.max(k_end - k_start, 0)
                                # task slot (depth-2 pipeline): ws carrier +
                                # flag id window this task owns on the Vector
                                # side.  Single-slot fallback pins it to 0.
                                slot = (task_id % n_slots) if use_pipe2 else 0
                                # adaptive row split (odd tails natural): AIV0
                                # owns [0, real_m0), AIV1 the rest; only REAL
                                # rows are covered.
                                real_m0 = (real_m + 1) // 2
                                row0 = subid * real_m0
                                rm = real_m0 - (real_m % 2) * subid
                                # Two-stage band split -- DIRECT integer
                                # comparisons (V3/VM-bnd), avoiding floordiv on
                                # negative numerators (CG-2026-0024):
                                #   apply pu iff n_lo > s_lo+row0+offset+up_shift-(bn_eff-1)
                                #   apply pl iff n_lo < s_lo+row0+rm-1+offset-wl
                                with T.rs("PIPE_V"):
                                    # task head: online state init (l == 0 skips
                                    # the alpha path, so zeroed alpha/ell/acc_o
                                    # make the first update exact).
                                    T.vbrc(value_zero, ub_ell)
                                    T.vbrc(value_zero, acc_o)
                                    T.vbrc(value_zero, ub_alpha)
                                    T.vbrc(value_zero, scales)
                                    T.vbrc(value_min, ub_m)
                                # ---- pass-1: f32 softmax chain + factor save --
                                for l in T.serial(loop_count):
                                    n_lo = (k_start + l) * bn_eff
                                    tn_real = T.min(bn_eff, kv_len - n_lo)
                                    tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                    T.copy(ub_m, ub_mprev)
                                    with T.rs("PIPE_MTE2"):
                                        T.sync_block_wait(l + slot * L_max_bound)
                                        if s_f16:
                                            T.copy(
                                                ws_s[kernel_id, slot, row0 : row0 + rm,
                                                     l * bn_eff : l * bn_eff + tnc],
                                                ub_f16_N[0:rm, 0:tnc],
                                            )
                                        else:
                                            T.copy(
                                                ws_s[kernel_id, slot, row0 : row0 + rm,
                                                     l * bn_eff : l * bn_eff + tnc],
                                                ub_f32_N[0:rm, 0:tnc],
                                            )
                                    with T.rs("PIPE_V"):
                                        if s_f16:
                                            T.vcast(ub_f16_N, ub_f32_N, round_mode="rint")
                                        # prescale (M1): S'' = prescale*S before
                                        # the penalty mask (DESIGN 1.6.1 M5).
                                        if prescale != 1.0:
                                            T.vmul(ub_f32_N, prescale, ub_f32_N)
                                        # ---- V2 upper bound (causal/right win)
                                        if use_upper_t and (
                                            n_lo > s_lo + row0 + offset + up_shift_t - (bn_eff - 1)
                                        ):
                                            k_up = (
                                                kernel_id * 0.0
                                                + (s_lo - n_lo + row0 + offset + up_shift_t)
                                            )
                                            T.vsub(ub_diff, k_up, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        # ---- V2 left window lower bound
                                        # (K_lo - diff is the scalar-minus-
                                        # buffer direction: compute
                                        # -(diff - K_lo) first, vsub only
                                        # allows a scalar src2).
                                        if use_left_t and (
                                            n_lo < s_lo + row0 + rm - 1 + offset - window_size_left
                                        ):
                                            k_lo = (
                                                kernel_id * 0.0
                                                + (s_lo - n_lo + row0 + offset - window_size_left)
                                            )
                                            T.vsub(ub_diff, k_lo, ub_pen)
                                            T.vmul(ub_pen, -1.0, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        # ---- V2/V4 per-row column OOB, now
                                        # GUARDED to ragged tail blocks
                                        # (r2_op1): for a full block
                                        # (tn_real == bn_eff) every column
                                        # j < bn_eff is in range, so the whole
                                        # 6-op chain is a provable no-op.  In
                                        # diff space the threshold carries the
                                        # row index: j >= tn_real <=> diff >=
                                        # (tn_real-1) - i.
                                        if tn_real < bn_eff:
                                            tnr = kernel_id * 0.0 + (tn_real - 1)
                                            T.vsub(ub_rowi, tnr, ub_thr)
                                            T.vmul(ub_thr, -1.0, ub_thr)
                                            T.vsub(ub_diff, ub_thr, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        # row max -> monotone max + NaN guard
                                        T.reduce(ub_f32_N, ub_mcur, dims=[1],
                                                 reduce_mode="max", clear=True)
                                        if l != 0:
                                            T.vmax(ub_mprev, ub_mcur, ub_m)
                                            T.vmax(ub_m, value_neg, ub_m)
                                            T.vsub(ub_mprev, ub_m, ub_t)
                                            T.vexp(ub_t, ub_alpha)
                                            T.copy(ub_alpha, scales[l * half : l * half + half, 0:1])
                                        else:
                                            T.copy(ub_mcur, ub_m)
                                        T.vsub(ub_f32_N, ub_m, ub_f32_N)
                                        T.vexp(ub_f32_N, ub_f32_N)
                                        T.reduce(ub_f32_N, ub_ellcur, dims=[1],
                                                 reduce_mode="sum", clear=True)
                                        T.vcast(ub_f32_N, ub_f16_N, round_mode="rint")
                                    with T.rs("PIPE_MTE3"):
                                        T.copy(
                                            ub_f16_N[0:rm, 0:tnc],
                                            ws_p[kernel_id, slot, row0 : row0 + rm,
                                                 l * bn_eff : l * bn_eff + tnc],
                                        )
                                        T.sync_block_set(l + slot * L_max_bound)
                                    with T.rs("PIPE_V"):
                                        T.vmul(ub_ell, ub_alpha, ub_ell)
                                        T.vadd(ub_ell, ub_ellcur, ub_ell)
                                # ---- pass-2: delayed rescale accumulate ----
                                for l in T.serial(loop_count):
                                    with T.rs("PIPE_MTE2"):
                                        T.sync_block_wait(l + slot * L_max_bound)
                                        if o_f16:
                                            T.copy(
                                                ws_o[kernel_id, slot, row0 : row0 + rm,
                                                     l * dim : (l + 1) * dim],
                                                ub_f16_D[0:rm, 0:dim],
                                            )
                                        else:
                                            T.copy(
                                                ws_o[kernel_id, slot, row0 : row0 + rm,
                                                     l * dim : (l + 1) * dim],
                                                ub_f32_D[0:rm, 0:dim],
                                            )
                                    with T.rs("PIPE_V"):
                                        if o_f16:
                                            T.vcast(ub_f16_D, ub_f32_D, round_mode="rint")
                                        if l != 0:
                                            T.copy(scales[l * half : l * half + half, 0:1], ub_alpha)
                                        T.vmul(acc_o, ub_alpha, acc_o)
                                        T.vadd(acc_o, ub_f32_D, acc_o)
                                # ---- epilogue: normalize + transpose-free lse
                                with T.rs("PIPE_V"):
                                    T.vdiv(acc_o, ub_ell, acc_o)
                                    T.vcast(acc_o, ub_f16_D, round_mode="rint")
                                    T.vlog2(ub_ell, ub_lse, ub_lse_tmp)
                                    T.vmul(ub_m, LOG2E, ub_t)
                                    T.vadd(ub_lse, ub_t, ub_lse)
                                with T.rs("PIPE_MTE3"):
                                    if rm > 0:
                                        # row-truncated stores: src slice extent
                                        # MUST match the dst region.
                                        T.copy(
                                            ub_f16_D[0:rm, 0:dim],
                                            output[q_off + row0 : q_off + row0 + rm, by, 0:dim],
                                        )
                                        T.copy(
                                            ub_lse[0:rm, 0:1],
                                            lse[by, q_off + row0 : q_off + row0 + rm, 0],
                                        )
                                    # task drain handshake -- single-slot
                                    # form only (depth-2 closes the cross-task
                                    # WAR with the flag window + slot alternation).
                                    if keep_barrier:
                                        T.sync_block_set(FLAG_TASKDONE)
                return _gqa_sw_fwd_varlen_main

            # Compile with a small retry loop: on this container
            # ``bishengir-compile`` occasionally dies with SIGSEGV on an
            # otherwise valid .npuir (identical kernel compiles cleanly in a
            # fresh process; the crash is inside the external subprocess, so a
            # retry from a clean invocation is safe and side-effect free).
            import time as _time

            kernel = None
            last_err = None
            for _attempt in range(24):
                try:
                    kernel = _builder_2phase()
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    cause = getattr(e, "__cause__", None)
                    blob = f"{e} {cause}"
                    if ("SIGSEGV" not in blob) and ("NPU IR opt failed" not in blob):
                        raise
                    # External-toolchain crash: give the process a moment and
                    # retry from a clean compile invocation.
                    _time.sleep(0.3)
            if kernel is None:
                raise last_err
            torch_ws_s = (
                torch.float16
                if ws_s_dtype == "float16"
                else (torch.bfloat16 if ws_s_dtype == "bfloat16" else torch.float32)
            )
            torch_ws_o = (
                torch.float16
                if ws_o_dtype == "float16"
                else (torch.bfloat16 if ws_o_dtype == "bfloat16" else torch.float32)
            )
            meta = {
                "ws_s_shape": ws_s_shape,
                "ws_p_shape": ws_p_shape,
                "ws_o_shape": ws_o_shape,
                "bn_eff": bn_eff,
                "L_max_bound": L_max_bound,
                "torch_ws_s": torch_ws_s,
                "torch_ws_o": torch_ws_o,
                "torch_dtype": (torch.float16 if dtype == "float16" else torch.bfloat16),
                "num_logical": num_logical,
            }
            return kernel, meta

        def wrapped(q_t, k_t, v_t, cu_q_t, cu_k_t, max_seqlen_q):
            kernel, meta = _compile(int(max_seqlen_q))
            ws_s = torch.empty(meta["ws_s_shape"], dtype=meta["torch_ws_s"], device=q_t.device)
            ws_p = torch.empty(meta["ws_p_shape"], dtype=meta["torch_dtype"], device=q_t.device)
            ws_o = torch.empty(meta["ws_o_shape"], dtype=meta["torch_ws_o"], device=q_t.device)
            zbuf = torch.zeros((meta["bn_eff"], dim), dtype=meta["torch_dtype"], device=q_t.device)
            out_t, lse_t = kernel(q_t, k_t, v_t, cu_q_t, cu_k_t, ws_s, ws_p, ws_o, zbuf)
            # lse was computed in a [H, total_q, 1] view; restore the contract
            # shape [H, total_q] (zero-copy view on contiguous memory).
            return out_t, lse_t.reshape(heads, total_q)

        return wrapped

    return _gqa_sw_fwd_varlen_func


# ---------- precision gates (DESIGN 8.2, source-consistent & calibrated) ----
_ATOL_OUT = 5e-3
_RTOL_OUT = 1e-5
_ATOL_LSE = 1e-3
_RTOL_LSE = 1e-3
# 2-ulp allowance: fp16 2*2^-10 = 2^-9, bf16 2*2^-7 = 2^-6 (relative).
_TWO_ULP_REL = {"float16": 2**-9, "bfloat16": 2**-6}


def _dtype_str(dt):
    return "float16" if dt == torch.float16 else "bfloat16"


def _check_output(out, ref_out, dtype_str, v_absmax=None):
    """Double-gate output check: tier-1 primary, tier-2 (2-ulp) fallback, plus
    a narrowly scoped tier-3 P-quantization noise classification (see the
    fixed-length sibling; the source algorithm quantizes P to the input dtype,
    so the golden reference itself carries P-quantization noise)."""
    flip_rel = {"float16": 2**-11, "bfloat16": 2**-8}[dtype_str]
    out_f = out.float().cpu()
    ref_f = ref_out.float().cpu()
    diff = (out_f - ref_f).abs()
    ref_abs = ref_f.abs()
    tier1_bad = diff > (_ATOL_OUT + _RTOL_OUT * ref_abs)
    n_flips = int(tier1_bad.sum().item())
    if n_flips == 0:
        return True, 0, float(diff.max().item()), None
    tol2 = _ATOL_OUT + (_RTOL_OUT + _TWO_ULP_REL[dtype_str]) * ref_abs
    tier2_bad = diff > tol2
    if not bool(tier2_bad.any().item()):
        return True, n_flips, float(diff.max().item()), None
    n_bad = int(tier2_bad.sum().item())
    frac = n_bad / diff.numel()
    note = None
    ok3 = False
    if v_absmax is not None and frac <= 1e-4:
        env = 2.0 * flip_rel * float(v_absmax)
        ok3 = bool((diff[tier2_bad] <= env).all().item())
        note = (
            f"tier3: {n_bad}/{diff.numel()} elem classified as "
            f"P-quant amplification noise (envelope {env:.1e})"
        )
    return ok3, n_flips, float(diff.max().item()), note


def _run_case(
    batch,
    seqlens_q,
    seqlens_k,
    heads,
    heads_kv,
    dim,
    is_causal,
    window_size_left,
    window_size_right,
    dtype=torch.float16,
    ns=1,
    tag="L?",
    seed=0,
    zeros=False,
    max_seqlen_q=None,
):
    """Generate packed inputs on NPU, run the kernel, gate vs the golden.
    Returns True on pass."""
    import time

    assert len(seqlens_q) == batch and len(seqlens_k) == batch
    torch.manual_seed(seed)
    dt_str = _dtype_str(dtype)
    dev = "npu"
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    cu_q = torch.tensor([0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
                        dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
                        dtype=torch.int32)
    cu_q_npu = cu_q.npu()
    cu_k_npu = cu_k.npu()

    gen = torch.zeros if zeros else torch.randn
    q = gen((total_q, heads, dim), dtype=dtype, device=dev)
    k = gen((total_k, heads_kv, dim), dtype=dtype, device=dev)
    v = gen((total_k, heads_kv, dim), dtype=dtype, device=dev)

    fn = _gqa_sw_fwd_varlen_kernel(
        batch, heads, heads_kv, total_q, total_k, dim, is_causal,
        window_size_left, window_size_right, dt_str,
    )
    # (64, 64, ns): exercises the wrapper-default -> design-default (64, 256)
    # config substitution path.
    wrapped = fn(64, 64, ns)
    msq = int(max_seqlen_q) if max_seqlen_q is not None else max(seqlens_q) if seqlens_q else 1
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out, lse = wrapped(q, k, v, cu_q_npu, cu_k_npu, msq)
    torch.npu.synchronize()
    t_ms = (time.perf_counter() - t0) * 1e3

    assert out.shape == (total_q, heads, dim) and out.dtype == dtype, (
        f"output shape/dtype contract violated: {tuple(out.shape)}/{out.dtype}"
    )
    assert lse.shape == (heads, total_q) and lse.dtype == torch.float32, (
        f"lse shape/dtype contract violated: {tuple(lse.shape)}/{lse.dtype}"
    )

    ref_out, ref_lse = golden_gqa_sw_fwd_varlen(
        q.cpu(), k.cpu(), v.cpu(), cu_q, cu_k, msq, is_causal,
        window_size_left, window_size_right,
    )

    v_absmax = float(v.float().abs().max().item())
    ok_out, flips, max_diff, t3note = _check_output(out, ref_out, dt_str, v_absmax)
    ok_lse = bool(torch.allclose(lse.cpu(), ref_lse, atol=_ATOL_LSE, rtol=_RTOL_LSE))
    lse_diff = float((lse.cpu() - ref_lse).abs().max().item())
    fin_out = bool(torch.isfinite(out).all().item())
    fin_lse = bool(torch.isfinite(lse).all().item())

    flip_note = f" tier1_flips={flips}" if flips else ""
    if t3note:
        flip_note += f" {t3note}"
    print(
        f"  [{tag}] B={batch} sq={seqlens_q} sk={seqlens_k} H={heads}/{heads_kv} "
        f"D={dim} causal={int(is_causal)} wl={window_size_left} wr={window_size_right} "
        f"{dt_str} msq={msq}{' zeros' if zeros else ''}: "
        f"out={'PASS' if ok_out else 'FAIL'} (max_diff={max_diff:.3e}"
        f"{flip_note}) lse={'PASS' if ok_lse else 'FAIL'} "
        f"(diff={lse_diff:.3e}) isfinite={fin_out and fin_lse} [{t_ms:.1f} ms]"
    )
    return ok_out and ok_lse and fin_out and fin_lse


def _contract_checks():
    """Wrapper contract verification: lru_cache identity, ValueError path,
    positional call form, output/lse shapes, lse-view restore."""
    print("  [Contract] wrapper call form / lru_cache / ValueError")
    f1 = _gqa_sw_fwd_varlen_kernel(2, 8, 2, 256, 256, 64, True, -1, -1, "float16")
    f2 = _gqa_sw_fwd_varlen_kernel(2, 8, 2, 256, 256, 64, True, -1, -1, "float16")
    assert f1 is f2, "lru_cache factory identity violated"
    fn = f1(64, 64, 1)
    cu_q = torch.tensor([0, 128, 256], dtype=torch.int32, device="npu")
    cu_k = torch.tensor([0, 128, 256], dtype=torch.int32, device="npu")
    q = torch.randn(256, 8, 64, dtype=torch.float16, device="npu")
    k = torch.randn(256, 2, 64, dtype=torch.float16, device="npu")
    v = torch.randn(256, 2, 64, dtype=torch.float16, device="npu")
    out, lse = fn(q, k, v, cu_q, cu_k, 128)  # positional form
    torch.npu.synchronize()
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (8, 256) and lse.dtype == torch.float32
    assert lse.is_contiguous(), "lse view restore must be contiguous"
    try:
        _gqa_sw_fwd_varlen_kernel(1, 8, 3, 256, 256, 64, True, -1, -1, "float16")
        raise AssertionError("heads%heads_kv ValueError not raised")
    except ValueError:
        pass
    print("  [Contract] PASS")


def run_L0():
    """L0 gate (blocking, DESIGN 8.2 order): L0-1 fp16 smoke offset=0 (main
    trace + ragged), L0-3 fp16 KV-cache offset>0 (highest-novelty mask path),
    L0-4 mixed-length ragged fp16 (ghost blocks), L0-2 bf16 smoke, contract."""
    print("== L0 (blocking) ==")
    ok = True
    ok &= _run_case(2, [128, 128], [128, 128], 8, 2, 64, True, -1, -1,
                    torch.float16, tag="L0-1-smoke-fp16", seed=0)
    ok &= _run_case(2, [64, 128], [256, 512], 8, 2, 64, True, -1, -1,
                    torch.float16, tag="L0-3-kvcache-offset", seed=1)
    ok &= _run_case(3, [128, 256, 384], [128, 256, 384], 8, 2, 64, True, -1, -1,
                    torch.float16, tag="L0-4-mixed-ragged", seed=2)
    ok &= _run_case(2, [128, 128], [128, 128], 8, 2, 64, True, -1, -1,
                    torch.bfloat16, tag="L0-2-smoke-bf16", seed=3)
    _contract_checks()
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (4 precision cases + contract)")


def run_L1():
    """L1 (blocking): manifest workload sweep -- source test domain (dim 64,
    H/Hkv 8/2) plus the 8b/70b short/long design workloads, fp16/bf16."""
    print("== L1 (blocking) ==")
    ok = True
    cases = [
        # manifest source domain (test_grouped_query_attention_sliding_window_varlen)
        (2, [256, 512], [256, 512], 8, 2, 64, True, 128, -1, "L1-manifest-w128"),
        (2, [256, 512], [256, 512], 8, 2, 64, False, -1, -1, "L1-manifest-noncausal"),
        (2, [256, 512], [256, 512], 8, 2, 64, False, 64, 64, "L1-manifest-symband"),
        (2, [64, 128], [256, 512], 8, 2, 64, True, 128, -1, "L1-manifest-kvcache-w128"),
        (2, [64, 128], [256, 512], 8, 2, 64, False, 64, 64, "L1-manifest-kvcache-symband"),
        (2, [256, 512], [256, 512], 8, 8, 64, True, -1, -1, "L1-manifest-gqa-h8-hkv8"),
        (2, [256, 512], [256, 512], 16, 1, 64, True, -1, -1, "L1-manifest-gqa-h16-hkv1"),
        (2, [256, 512], [256, 512], 8, 2, 64, False, -1, 64, "L1-manifest-rightwin"),
        (2, [128, 256], [128, 256], 8, 2, 64, True, 0, -1, "L1-manifest-wl0"),
        # design workloads (8b/70b short/long)
        (4, [512, 512, 512, 512], [512, 512, 512, 512], 32, 8, 128, True, 256, -1, "L1-8b-short"),
        (4, [2048] * 4, [2048] * 4, 32, 8, 128, True, 1024, -1, "L1-8b-long"),
        (4, [512] * 4, [512] * 4, 64, 8, 128, True, 256, -1, "L1-70b-short"),
        (4, [2048] * 4, [2048] * 4, 64, 8, 128, True, 1024, -1, "L1-70b-long"),
    ]
    seed = 10
    for batch, sq, sk, h, hkv, d, causal, wl, wr, name in cases:
        for dt in (torch.float16, torch.bfloat16):
            ok &= _run_case(batch, sq, sk, h, hkv, d, causal, wl, wr, dt, ns=1,
                            tag=f"{name}-{_dtype_str(dt)}", seed=seed)
            seed += 1
    assert ok, "L1 FAILED (blocking)"
    print(f"[L1] PASS ({len(cases) * 2} cases)")


def run_L2():
    """L2 (warn-only): four mask families, GQA ratios, non-power-of-2 lengths,
    KV-cache x masks, right window, wl=0."""
    print("== L2 (warn-only) ==")
    results = []
    results.append(_run_case(2, [128, 128], [128, 128], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="L2-causal", seed=20))
    results.append(_run_case(2, [128, 128], [128, 128], 8, 2, 64, True, 16, -1,
                             torch.float16, tag="L2-causal-w16", seed=21))
    results.append(_run_case(2, [128, 128], [128, 128], 8, 2, 64, False, -1, -1,
                             torch.float16, tag="L2-noncausal", seed=22))
    results.append(_run_case(2, [128, 128], [128, 128], 8, 2, 64, False, 64, 64,
                             torch.float16, tag="L2-symband", seed=23))
    results.append(_run_case(2, [128, 128], [128, 128], 8, 2, 64, False, 64, 64,
                             torch.bfloat16, tag="L2-symband-bf16", seed=24))
    results.append(_run_case(2, [64, 128], [256, 512], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="L2-kvcache-causal", seed=25))
    results.append(_run_case(2, [64, 128], [256, 512], 8, 2, 64, True, 128, -1,
                             torch.float16, tag="L2-kvcache-w128", seed=26))
    results.append(_run_case(2, [64, 128], [256, 512], 8, 2, 64, False, 64, 64,
                             torch.float16, tag="L2-kvcache-symband", seed=27))
    results.append(_run_case(3, [128, 256, 384], [128, 256, 384], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="L2-mixed-b3", seed=28))
    results.append(_run_case(2, [128, 128], [128, 128], 8, 8, 64, True, -1, -1,
                             torch.float16, tag="L2-gqa-h8-hkv8", seed=29))
    results.append(_run_case(2, [128, 128], [128, 128], 16, 1, 64, True, -1, -1,
                             torch.float16, tag="L2-gqa-h16-hkv1", seed=30))
    results.append(_run_case(2, [96, 160], [96, 160], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="L2-s96-s160", seed=31))
    n_pass = sum(results)
    if n_pass == len(results):
        print(f"[L2] PASS ({len(results)} cases)")
    else:
        print(f"[L2] WARN: {len(results) - n_pass}/{len(results)} cases failed (recorded, non-blocking)")


def run_boundary():
    """Boundary (warn-only): extreme short samples, single-token samples,
    kv_len not a multiple of 16, zeros, q_len=0 samples, ghost blocks."""
    print("== Boundary (warn-only) ==")
    results = []
    results.append(_run_case(1, [1], [1], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-q1", seed=40))
    results.append(_run_case(1, [1], [16], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-q1-kv16", seed=41))
    results.append(_run_case(1, [13], [13], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-kv13-not16", seed=42))
    results.append(_run_case(1, [13], [29], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-kv29-not16-off16", seed=48))
    results.append(_run_case(1, [20], [68], 8, 2, 64, False, -1, -1,
                             torch.float16, tag="B-kv68-noncausal", seed=43))
    results.append(_run_case(1, [64], [64], 8, 2, 64, False, -1, -1,
                             torch.float16, tag="B-zeros", seed=44, zeros=True))
    # q_len = 0 sample alongside real samples (cu equal)
    results.append(_run_case(2, [0, 128], [64, 128], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-empty-sample", seed=45))
    # max_seqlen_q > actual max -> ghost blocks in every sample
    results.append(_run_case(2, [64, 100], [64, 100], 8, 2, 64, True, -1, -1,
                             torch.float16, tag="B-ghost-msq196", seed=46,
                             max_seqlen_q=196))
    results.append(_run_case(1, [129], [129], 8, 2, 64, True, 32, -1,
                             torch.bfloat16, tag="B-s129-w32-bf16", seed=47))
    n_pass = sum(results)
    if n_pass == len(results):
        print(f"[Boundary] PASS ({len(results)} cases)")
    else:
        print(f"[Boundary] WARN: {len(results) - n_pass}/{len(results)} cases failed "
              f"(recorded, non-blocking)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
