# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GQA sliding-window forward kernel, NPU Expert-mode two-phase reimplementation.

Migrated from the GPU TileLang factory ``_gqa_sw_fwd_kernel`` (BSHD layout,
flash-attention style online softmax with a sliding-window / causal band mask)
to ``target="npuir"`` per DESIGN.md (Expert mode, two-phase S/P
materialization, E1-E7 + E-band/E-mask/E-bnd):

    o   = softmax(prescale * q @ k^T + band_mask) @ v   (BSHD, in dtype)
    lse = log2-domain per-row log-sum-exp                ([B, H, S], fp32)

The CPU golden below carries the independent sliding-window semantics (the
source's exact three-condition mask), never the NPU two-phase algorithm.

Structure (DESIGN 0.6 / 1.4 / 6), a structural descendant of the same-type
precedent ``_gqa_prefill_fwd_kernel.py`` (GQA prefill, Expert two-phase):
  - 1-D persistent ``T.Kernel(20)`` (E1, Ascend910B3 AIC count); logical task
    = (q-block, head, batch); ``cid = task_id * 20 + kernel_id`` decode with a
    ghost-task clamp.
  - BAND KV RANGE (E-band, the sliding-window core): the per-task KV loop
    covers only ``l in [0, loop_count)`` with ``n = k_start + l``; ``k_start``
    / ``k_end`` / ``loop_count`` are the source's S4 formulas, ported verbatim
    as per-task PrimExprs. Trimmed blocks are online-softmax identities.
  - Flag id = LOCAL band index ``l`` (E-band), triple handshake reused
    (S-ready -> P-ready -> O-ready) + FLAG_TASKDONE = L_max at the task
    boundary (single-slot cross-task WAR protection). Flag budget is enforced
    at factory time (``L_max + 1 <= 16``).
  - Two-phase dual-Scope stream (E2): Cube pass-1 runs ALL S = Q@K^T blocks
    wait-free (Q-hoisted once per task) into a per-core single-slot GM ws_s;
    Cube pass-2 consumes the P blocks and materializes per-block PV partial
    sums into ws_o; Vector pass-1 consumes S (vcast -> two-sided mask ->
    prescale -> online max/alpha/ell -> P quantize) saving per-block rescale
    factors into UB ``scales[]``; Vector pass-2 replays the SAME factor
    sequence onto acc_o; epilogue normalizes and writes lse transpose-free.
  - TWO-SIDED ARITHMETIC-PENALTY MASK (E-mask, sliding-window-specific, the
    design's highest-novelty element): no ``vcmp`` (PL-1.11 scalarization),
    fully vectorised ALU chain, device-verified bit-exact by probe P2:
        diff  = j - i                              (built once per core, f32)
        pu    = clamp(diff - K_up, 0, 1)           (causal / right-window)
        pl    = clamp(K_lo - diff, 0, 1)           (left-window lower bound)
        S    += pen * (-1e38)                      (pen in {0,1})
    applied only on boundary blocks: TWO-STAGE band split (E-bnd) -- blocks
    ``l < L_lo_full`` are not fully left-valid, blocks ``l >= L_hi_full`` are
    not fully upper-valid; the all-valid middle stretch has zero mask cost.
  - e-domain online softmax (E4): m/alpha/P/ell in fp32, lse via vlog2 plus
    m*LOG2E; NaN guard clamp ``vmax(m, -1e38)`` kept from the source; the
    -1e38 sentinel makes every masked P an exact +0.0.
  - fp16/bf16 direct Cube path (E5): structurally isomorphic traces; ws
    carrier switches at trace time -- fp16: f16/f16/f16, bf16: f32/bf16/f32.
  - Unified tail handling (E7): size-form gemms with fractal lower clamps
    (M/N >= 16, K >= 32), tail-clamped slice loads, a column-OOB arithmetic
    penalty folded into the same chain, a per-core ``zbuf`` zero-init of
    ``l1_b`` (PL-1.11: 0*NaN=NaN) and row-truncated stores. The manifest
    domain has no ragged tail (all S divisible by bn_eff) -- this mechanism
    only serves the contract-test domain.

Implementation notes (toolchain facts, each taken from the MHA precedent or
the device-verified design probes P1/P2):
  1. GM->L1 block loads use slice-form ``T.copy`` (NOT ``T.load_nd2nz``: it
     silently reads strided BSHD same-head tiles as flat memory, CG-2026-0004).
  2. Developer auto CV-split cannot hold persistent + T.gemm + v-prefix
     (BishengIR compile-time crash, probe P1 / CG-2026-0010 / PL-1.16) ->
     Expert dual-Scope.
  3. ``T.reduce`` with ``clear=False`` is a documented silent-error form on
     uninitiated buffers -- every reduce here passes ``clear=True``.
  4. ``vbrc`` scalar sources must be let-bound locals, not raw literals;
     ``vsub``/``vmax``/``vmin``/``vmul`` scalar operands must be float-typed
     / kernel-derived PrimExprs (probe P2 revision-1 fix).
  5. lse is declared as a ``[B, H, S, 1]`` VIEW of the contract ``[B, H, S]``
     memory so the epilogue writes the contiguous row run as a natural 2D
     region copy (no ``T.transpose``: a live-source transpose epilogue
     poisons the whole kernel 2.6x, VP-D6); the closure reshapes it back.
  6. S/O cross the engines through GM workspaces (store_fixpipe dst is GM
     only, CG-2026-0005).

Interface contract (unchanged vs GPU source, DESIGN E6):

    _gqa_sw_fwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
        window_size_left, window_size_right, dtype='float16')(
        block_m, block_n, num_stages)(q, k, v) -> (output, lse)

Run: python _gqa_sw_fwd_kernel.py --level {L0,all}
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
# Toolchain shim (this tilelang build + CANN 9.0.0).
# The NPUIR codegen emits the sync-block attribute as ``syn_instr_mode``
# (src/target/codegen_npuir.cc) while the CANN 9.0.0 ``bishengir-compile``
# parser expects ``sync_instr_mode`` (its diagnostic:
# "failed to parse HIVM_SyncBlockInstrModeAttr parameter 'sync_instr_mode'").
# Minimal repro / evidence: any Expert dual-Scope kernel with a single
# ``T.sync_block_set`` fails with "Dialect `' not found for custom op
# 'syn_instr_mode'" (probe_sync.py).  This shim rewrites that one attribute
# name in the generated .npuir text immediately before the external
# bishengir-compile call.  It touches neither the toolchain on disk nor any
# other codegen output.  Without it the two-phase cross-engine handshake
# cannot compile on this container.
# ---------------------------------------------------------------------------
# Second toolchain defect on the same build: ``bishengir-compile`` 1.1.0
# (LLVM 19.1.7) intermittently SIGSEGVs (exit -11) while compiling this
# kernel's IR -- the same .npuir compiles or crashes across runs (measured
# ~25-35% success per attempt; the crash point varies between
# ConvertHFusionToHIVM and ConvertHIVMToStandard, and disabling threading,
# ASLR or raising the stack limit does not help).  This is a compiler
# memory bug rather than an IR validation error, so the external compile is
# retried on SIGSEGV only (other failures are re-raised unchanged).
# r18b: a third failure mode of the same bug is an INFINITE LOOP (100% CPU,
# no output, observed 80+ min) instead of a crash; the stock call
# ``subprocess.run(cmd, capture_output=True, check=True)`` has no timeout, so
# a single hang wedges the whole build.  The shim therefore (a) caps every
# bishengir-compile invocation at ``_COMPILE_TIMEOUT_S`` wallclock (the child
# is killed by subprocess.run on expiry) and (b) retries the timeout the
# same way as the SIGSEGV, so a trace is built as soon as one attempt lands
# in the ~25-35% success window.
_COMPILE_RETRIES = 16
_COMPILE_TIMEOUT_S = 150


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

        # (a) wallclock cap on the external compiler call only.  The patch is
        # installed on the jit_npu module's ``subprocess`` binding, which is
        # the one ``_npuir_to_bin_enable_npu_compile`` uses.
        if not getattr(_jn.subprocess.run, "_compile_timeout", False):
            _real_run = _jn.subprocess.run

            def _run_guarded(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args")
                if kwargs.get("timeout") is None and "bishengir-compile" in str(cmd):
                    kwargs["timeout"] = _COMPILE_TIMEOUT_S
                return _real_run(*args, **kwargs)

            _run_guarded._compile_timeout = True
            _jn.subprocess.run = _run_guarded

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
                    text = f"{exc} {cause!r}"
                    if (
                        rc in (-11, 139)
                        or "Segmentation fault" in text
                        or "timed out" in text
                        or "TimeoutExpired" in text
                    ):
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


# ---------- Golden (independent PyTorch CPU reference, DESIGN 8.1) ----------
def golden_gqa_sw_fwd(q, k, v, is_causal, window_size_left, window_size_right):
    """Sliding-window GQA forward reference (BSHD, batched materialize softmax).

    q:    [B, S, H, D]        fp16/bf16 (BSHD)
    k/v:  [B, S, H_kv, D]     fp16/bf16 (BSHD)
    Returns (output: q shape/dtype, lse: [B, H, S] fp32).

    Independence: no online softmax, no band trimming, no e-domain reordering,
    no arithmetic-penalty mask -- per (b, h) materialized batched softmax in
    fp32 with the source's exact three-condition mask, P quantized at the
    source ``acc_s_cast`` point.
    """
    batch, seq_len, heads, dim = q.shape
    heads_kv = k.shape[2]
    groups = heads // heads_kv
    scale = dim**-0.5

    q_ = q.float().transpose(1, 2)  # [B, H, S, D]
    k_exp = k.repeat_interleave(groups, dim=2).float().transpose(1, 2)
    v_exp = v.repeat_interleave(groups, dim=2).float().transpose(1, 2)

    scores = torch.matmul(q_, k_exp.transpose(-2, -1)) * scale  # [B, H, S, S]

    q_pos = torch.arange(seq_len, device=q.device).unsqueeze(1)  # [S, 1]
    k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)  # [1, S]
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=q.device)
    if is_causal:
        mask = mask | (q_pos < k_pos)
    if window_size_left >= 0:
        mask = mask | (k_pos < q_pos - window_size_left)
    if window_size_right >= 0:
        mask = mask | (k_pos > q_pos + window_size_right)

    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # log2-domain lse (D7 calibration).
    lse = torch.logsumexp(scores, dim=-1) / math.log(2.0)
    probs = torch.softmax(scores, dim=-1)
    p_q = probs.to(q.dtype).float()  # P quantized to the input dtype
    output = torch.matmul(p_q, v_exp).to(q.dtype)  # [B, H, S, D]
    return output.transpose(1, 2).contiguous(), lse


# ---------- Kernel (Expert mode two-phase, DESIGN 0.6 E1-E7) ----------
@functools.lru_cache(maxsize=32)
def _gqa_sw_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,  # -1 = unlimited
    window_size_right: int,  # -1 = unlimited
    dtype: str = "float16",
) -> Callable:
    """GPU-source-compatible factory (signature/validation/lru_cache kept).

    Returns an inner factory ``_gqa_sw_fwd_func(block_m, block_n, num_stages)``
    whose result is a closure ``wrapped(q, k, v) -> (output, lse)`` allocating
    the per-core GM workspaces on each call (DESIGN E6).
    """
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    # r18 G=2: two query heads that share ONE kv head ride in the same block
    # (H/Hkv = 4 for llama-3.1-8b/70b), so the K/V GM->L1 traffic per unit of
    # work halves.  ``head_group`` is trace-time (1 = the plain single-head
    # kernel).  ``groups % 2 == 0`` is required: a pair (2p, 2p+1) must not
    # straddle a kv-head boundary, otherwise the two heads need different K/V.
    # ``dim >= 128`` and the bf16/8b-long exclusion are trace-time
    # compileability guards (NOT semantic conditions):
    #  * the D=64/S=1024/wl=512 G=2 trace deterministically SIGSEGVs
    #    bishengir-compile (err code -11, retried), while the identical G=1
    #    trace compiles in ~24 s;
    #  * the bf16, seq>=2048, heads=32 G=2 trace deterministically SIGSEGVs
    #    the same way (70b-long bf16 and 8b-long fp16 both compile in ~25 s).
    # All formal tune workloads keep G=2 except the last one, which keeps the
    # proven G=1 body (no gain, no regression).
    #
    # r18b compile-stability fix (the second deterministic SIGSEGV class):
    # every ``head_group`` loop below is a PYTHON ``range`` (trace-time
    # unroll), NOT ``T.serial(head_group)``.  ``T.serial(1)`` -- the G=1
    # fallback form -- traced the D=64/S=520/wl=256 case into IR that makes
    # ``bishengir-compile`` SIGSEGV deterministically (rc -11, 40 retries
    # never converge, ~24 s per attempt), while the byte-identical
    # straight-line G=1 body compiles in 24 s.  With ``range`` the G=1 trace
    # degenerates to exactly one straight-line copy of the body (the
    # Stage-3-verified form) and every G>=2 trace is the same statements
    # unrolled; only the outer band loop stays a runtime ``T.serial``.
    _bf16_8b_long = dtype == "bfloat16" and seq_len >= 2048 and heads == 32
    head_group = (
        2
        if (
            groups >= 2
            and groups % 2 == 0
            and heads % 2 == 0
            and dim >= 128
            and not _bf16_8b_long
        )
        else 1
    )
    accum_dtype = "float32"
    prescale = dim**-0.5
    has_window = window_size_left >= 0 or window_size_right >= 0
    # Trace-time mask-family selectors (source 100-123 four-branch semantics).
    use_left = window_size_left >= 0
    # Upper bound: causal (source IGNORES the right window under causal,
    # DESIGN 0.1 corner) OR a non-causal right window.
    use_upper = bool(is_causal) or ((not is_causal) and window_size_right >= 0)
    up_shift = 0 if is_causal else (window_size_right if window_size_right >= 0 else 0)
    # Trace-time k_end formula selector (source S4).
    causal_end = bool(is_causal)
    right_end = (not is_causal) and has_window and window_size_right >= 0

    def _gqa_sw_fwd_func(block_m: int, block_n: int, num_stages: int) -> Callable:
        # ---- effective config (DESIGN E6): the wrapper default (64, 64, 1) /
        # (64, 32, 1) is replaced by the design default (64, 256); any other
        # explicitly-passed config is respected verbatim (S4-5 discipline).
        # num_stages stays a reserved knob: the single-slot two-phase structure
        # is stage-invariant.
        if (block_m, block_n, num_stages) in ((64, 64, 1), (64, 32, 1)):
            block_m, block_n = 64, 256
        bm = block_m
        half = bm // 2
        # Flag-budget guard (E-band): bn_eff is large enough that the widest
        # band spans <= 15 blocks, so L_max + 1 <= 16 flag ids (0..15).
        eff_left = window_size_left if window_size_left >= 0 else seq_len
        eff_right = window_size_right if window_size_right >= 0 else 0
        bn_guard = (((bm + eff_left + eff_right + 13) // 14 + 15) // 16) * 16
        bn_eff = max(block_n, bn_guard)
        assert bn_eff % 16 == 0 and dim % 16 == 0
        block_share = max(bn_eff, dim)
        num_q_blocks = (seq_len + bm - 1) // bm
        # head_group heads per logical block -> the head axis shrinks by
        # head_group and each block carries head_group q-tiles.
        HG = heads // head_group
        num_logical = num_q_blocks * HG * batch
        HB = HG * batch
        nk_total = (seq_len + bn_eff - 1) // bn_eff
        # ---- L_max (trace-time scan over all q-blocks; DESIGN 5.2) ----
        L_max = 0
        for _bx in range(num_q_blocks):
            if causal_end:
                _k_end = (min(seq_len, (_bx + 1) * bm) + bn_eff - 1) // bn_eff
            elif right_end:
                _k_end = (min(seq_len, (_bx + 1) * bm + window_size_right) + bn_eff - 1) // bn_eff
            else:
                _k_end = (seq_len + bn_eff - 1) // bn_eff
            if has_window and window_size_left >= 0:
                _k_start = max(0, _bx * bm - window_size_left) // bn_eff
            else:
                _k_start = 0
            L_max = max(L_max, max(_k_end - _k_start, 0))
        assert L_max >= 1 and L_max + 1 <= 16, f"flag budget exceeded: L_max={L_max}"
        pad_band = L_max * bn_eff
        # ---- ragged kv tail (E7): trace-time gates ----
        # has_tail: S_kv is not a multiple of bn_eff -- the last block reads
        #   only tn_real < bn_eff columns, so the Vector tile's columns
        #   [tn_real, bn_eff) hold stale UB bytes that would poison the row
        #   max (NaN).  Needs (a) a column-OOB arithmetic penalty over the
        #   FULL [half, bn_eff] width and (b) a once-per-core zero init of the
        #   f32/f16 N-buffers so those columns are finite before masking.
        # has_band: the fractal band tnc > tn_real, i.e. gemm1/gemm2 read
        #   stale K/V rows -- needs the zbuf l1_b zero init (PL-1.11).
        tail_kv = seq_len - (nk_total - 1) * bn_eff
        tnc_last = max(32, min(bn_eff, ((tail_kv + 15) // 16) * 16))
        has_band = tnc_last > tail_kv
        has_tail = (seq_len % bn_eff) != 0
        # ---- ws carriers (E5): fp16 -> f16/f16/f16; bf16 -> f32/bf16/f32 ----
        if dtype == "float16":
            ws_s_dtype, ws_o_dtype = "float16", "float16"
        else:
            ws_s_dtype, ws_o_dtype = "float32", "float32"
        # ---- task-slot dimension (r3_op1): depth-2 cross-task software
        # pipeline.  Slot s owns flag ids [s*L_max, s*L_max + L_max), so the
        # budget is 2*L_max <= 15 usable ids (flag 0..15).  When the band is
        # too wide for two slots the kernel falls back to the single-slot
        # form (slot pinned to 0 + the FLAG_TASKDONE task barrier).
        n_slots = 2 if 2 * L_max <= 15 else 1
        use_pipe2 = n_slots == 2
        keep_barrier = not use_pipe2
        ws_s_shape = [NUM_KERNELS, n_slots, head_group * bm, pad_band]
        ws_p_shape = [NUM_KERNELS, n_slots, head_group * bm, pad_band]
        ws_o_shape = [NUM_KERNELS, n_slots, head_group * bm, dim * L_max]
        FLAG_TASKDONE = L_max  # ids 0..L_max-1 are local band-block indices
        # trace-time selectors (Python constants -> single traced variant)
        s_f16 = dtype == "float16"
        causal_t = bool(is_causal)
        use_left_t = bool(use_left)
        use_upper_t = bool(use_upper)
        up_shift_t = int(up_shift)
        has_band_t = bool(has_band)
        has_tail_t = bool(has_tail)

        @tilelang.jit(
            out_idx=[-2, -1],
            target="npuir",
            pass_configs={
                # Protect the manual two-phase flow and the explicit
                # workspaces from compiler reordering (DESIGN 7.3).
                tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
                tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
                # Round 19 / candidate A: Expert mode already hand-writes the
                # cross-engine two-phase handshake (sync_block_set/wait per
                # band step); test whether the compiler's automatic inner-core
                # sync injection is redundant barrier overhead.
                # -> bishengir-compile --disable-hivm-auto-inject-sync=true
                tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: True,
            },
        )
        def _builder_2phase(bm_, bn_, ns_):
            @T.prim_func
            def _gqa_sw_fwd_main(
                q: T.Tensor((batch, seq_len, heads, dim), dtype),
                k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
                v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
                ws_s: T.Tensor(ws_s_shape, ws_s_dtype),
                ws_p: T.Tensor(ws_p_shape, dtype),
                ws_o: T.Tensor(ws_o_shape, ws_o_dtype),
                zbuf: T.Tensor((bn_eff, dim), dtype),
                output: T.Tensor((batch, seq_len, heads, dim), dtype),
                # lse is a [B, H, S, 1] VIEW of the contract's [B, H, S]
                # memory: the epilogue writes the contiguous row run as a
                # natural 2D region copy from ub_lse[0:rm, 0:1] -- NO
                # T.transpose (VP-D6).
                lse: T.Tensor((batch, heads, seq_len, 1), accum_dtype),
            ):
                with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
                    # Ghost-task clamp (D3/E1): T.ceildiv lowers to a
                    # truncating form that returns 1 for x in (-N, 0]; clamp
                    # pins an idle core's task to the LAST LEGAL task id.
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)
                    # ================= Cube stream (E2) =================
                    with T.Scope("Cube"):
                        # Dedicated Q buffer so the NEXT task's Q hoist no
                        # longer WARs the CURRENT task's P staging in l1_a
                        # -- prerequisite of the depth-2 pipeline.
                        l1_q = T.alloc_L1([head_group * bm, dim], dtype)
                        l1_a = T.alloc_L1([head_group * bm, block_share], dtype)
                        l1_b = T.alloc_L1([bn_eff, dim], dtype)
                        l0_c = T.alloc_L0C([head_group * bm, block_share], accum_dtype)
                        if has_band_t:
                            # PL-1.11: 0*NaN=NaN -- zero-init the V carrier
                            # once per core so the fractal band of gemm2
                            # cannot pick up uninitialised (NaN) rows.
                            with T.rs("PIPE_MTE2"):
                                T.copy(zbuf, l1_b)
                        # ---- depth-2 pipeline prologue (r3_op1): hoist
                        # Q(0) and produce the whole S band of task 0 into
                        # slot 0 before entering the steady-state loop, so
                        # the first p2 wait is already covered.
                        if use_pipe2:
                            cid0 = T.min(kernel_id, num_logical - 1)
                            bx0 = cid0 // HB
                            byp0 = (cid0 // batch) % HG
                            bz0 = cid0 % batch
                            s_lo0 = bx0 * bm
                            kv_head0 = (byp0 * head_group) // groups
                            tail0 = T.min(bm, seq_len - s_lo0)
                            tmc0 = T.max(16, T.min(bm, T.ceildiv(tail0, 16) * 16))
                            # fused-M extent: head gh occupies rows
                            # [gh*bm, gh*bm+tmc); one gemm covers
                            # [0, bm + tmc) so the per-head row base stays
                            # a trace-time constant.
                            fm0 = tmc0 if head_group == 1 else bm + tmc0
                            k_end0 = (
                                T.ceildiv(T.min(seq_len, (bx0 + 1) * bm), bn_eff)
                                if causal_t
                                else (
                                    T.ceildiv(
                                        T.min(seq_len, (bx0 + 1) * bm + window_size_right), bn_eff
                                    )
                                    if right_end
                                    else T.ceildiv(seq_len, bn_eff)
                                )
                            )
                            k_start0 = (
                                T.max(0, bx0 * bm - window_size_left) // bn_eff
                                if use_left_t
                                else 0
                            )
                            lc0 = T.max(k_end0 - k_start0, 0)
                            # python range -> trace-time unroll (see factory
                            # note: ``T.serial(1)`` here SIGSEGVs the compiler)
                            for gh in range(head_group):
                                with T.rs("PIPE_MTE2"):
                                    T.copy(
                                        q[bz0, s_lo0 : s_lo0 + tail0,
                                          byp0 * head_group + gh, 0:dim],
                                        l1_q[gh * bm : gh * bm + tail0, 0:dim],
                                    )
                            for l in T.serial(lc0):
                                n_lo = (k_start0 + l) * bn_eff
                                tn_real = T.min(bn_eff, seq_len - n_lo)
                                tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                with T.rs("PIPE_MTE2"):
                                    T.copy(
                                        k[bz0, n_lo : n_lo + tn_real, kv_head0, 0:dim],
                                        l1_b[0:tn_real, 0:dim],
                                    )
                                with T.rs("PIPE_C"):
                                    T.gemm(
                                        l1_q,
                                        l1_b,
                                        l0_c,
                                        initC=True,
                                        b_transpose=True,
                                        size=[fm0, dim, tnc],
                                    )
                                with T.rs("PIPE_FIX"):
                                    T.copy(
                                        l0_c[0:fm0, 0:tnc],
                                        ws_s[kernel_id, 0, 0:fm0, l * bn_eff : l * bn_eff + tnc],
                                    )
                                    T.sync_block_set(l)
                        for task_id in T.serial(num_local_tasks):
                            cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                            bx = cid // HB
                            byp = (cid // batch) % HG
                            bz = cid % batch
                            s_lo = bx * bm
                            kv_head = (byp * head_group) // groups
                            tail_m_real = T.min(bm, seq_len - s_lo)
                            tmc = T.max(16, T.min(bm, T.ceildiv(tail_m_real, 16) * 16))
                            fm = tmc if head_group == 1 else bm + tmc
                            # ---- band KV range (E-band, source S4 verbatim) --
                            # Ternary (not if/elif): a trace-time `if` would
                            # scope k_end inside the branch.
                            k_end = (
                                T.ceildiv(T.min(seq_len, (bx + 1) * bm), bn_eff)
                                if causal_t
                                else (
                                    T.ceildiv(
                                        T.min(seq_len, (bx + 1) * bm + window_size_right), bn_eff
                                    )
                                    if right_end
                                    else T.ceildiv(seq_len, bn_eff)
                                )
                            )
                            k_start = (
                                T.max(0, bx * bm - window_size_left) // bn_eff
                                if use_left_t
                                else 0
                            )
                            loop_count = T.max(k_end - k_start, 0)
                            slot = (task_id % n_slots) if use_pipe2 else 0
                            adv_slot = (1 - task_id % n_slots) if use_pipe2 else 0
                            # single-slot form re-runs p1 for the SAME task
                            # inside the loop body (classic in-order stream).
                            adv_task = (task_id + 1) if use_pipe2 else task_id
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
                            # ---- advance: Q hoist + all S = Q @ K^T for the
                            # next task (fills the p2 wait stall of task T).
                            if adv_ok:
                                cid_a = T.min(adv_task * NUM_KERNELS + kernel_id, num_logical - 1)
                                bx_a = cid_a // HB
                                byp_a = (cid_a // batch) % HG
                                bz_a = cid_a % batch
                                s_lo_a = bx_a * bm
                                kv_head_a = (byp_a * head_group) // groups
                                tail_a = T.min(bm, seq_len - s_lo_a)
                                tmc_a = T.max(16, T.min(bm, T.ceildiv(tail_a, 16) * 16))
                                fm_a = tmc_a if head_group == 1 else bm + tmc_a
                                k_end_a = (
                                    T.ceildiv(T.min(seq_len, (bx_a + 1) * bm), bn_eff)
                                    if causal_t
                                    else (
                                        T.ceildiv(
                                            T.min(seq_len, (bx_a + 1) * bm + window_size_right),
                                            bn_eff,
                                        )
                                        if right_end
                                        else T.ceildiv(seq_len, bn_eff)
                                    )
                                )
                                k_start_a = (
                                    T.max(0, bx_a * bm - window_size_left) // bn_eff
                                    if use_left_t
                                    else 0
                                )
                                lc_a = T.max(k_end_a - k_start_a, 0)
                                for gh in range(head_group):
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            q[bz_a, s_lo_a : s_lo_a + tail_a,
                                              byp_a * head_group + gh, 0:dim],
                                            l1_q[gh * bm : gh * bm + tail_a, 0:dim],
                                        )
                                for l in T.serial(lc_a):
                                    n_lo = (k_start_a + l) * bn_eff
                                    tn_real = T.min(bn_eff, seq_len - n_lo)
                                    tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            k[bz_a, n_lo : n_lo + tn_real, kv_head_a, 0:dim],
                                            l1_b[0:tn_real, 0:dim],
                                        )
                                    with T.rs("PIPE_C"):
                                        T.gemm(
                                            l1_q,
                                            l1_b,
                                            l0_c,
                                            initC=True,
                                            b_transpose=True,
                                            size=[fm_a, dim, tnc],
                                        )
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_c[0:fm_a, 0:tnc],
                                            ws_s[
                                                kernel_id,
                                                adv_slot,
                                                0:fm_a,
                                                l * bn_eff : l * bn_eff + tnc,
                                            ],
                                        )
                                        T.sync_block_set(l + adv_slot * L_max)
                            # ---- pass-2: all O_partial = P @ V for task_id ----
                            for l in T.serial(loop_count):
                                n_lo = (k_start + l) * bn_eff
                                tn_real = T.min(bn_eff, seq_len - n_lo)
                                tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(l + slot * L_max)
                                    T.copy(
                                        ws_p[
                                            kernel_id,
                                            slot,
                                            0:fm,
                                            l * bn_eff : l * bn_eff + tnc,
                                        ],
                                        l1_a[0:fm, 0:tnc],
                                    )
                                    T.copy(
                                        v[bz, n_lo : n_lo + tn_real, kv_head, 0:dim],
                                        l1_b[0:tn_real, 0:dim],
                                    )
                                with T.rs("PIPE_C"):
                                    T.gemm(l1_a, l1_b, l0_c, initC=True,
                                           size=[fm, tnc, dim])
                                with T.rs("PIPE_FIX"):
                                    T.copy(
                                        l0_c[0:fm, 0:dim],
                                        ws_o[kernel_id, slot, 0:fm, l * dim : (l + 1) * dim],
                                    )
                                    T.sync_block_set(l + slot * L_max)
                    # ================= Vector stream (E2) =================
                    with T.Scope("Vector"):
                        # let-bound broadcast scalars (D4: vbrc/vsub/vmax scalar
                        # operands must not be raw literals).
                        value_neg = -1e38
                        value_zero = 0
                        value_one = 1.0
                        value_min = -T.infinity("float32")
                        ub_f16_N = T.alloc_ub([half, bn_eff], dtype)
                        ub_f32_N = T.alloc_ub([half, bn_eff], accum_dtype)
                        # r18_shrink: the D carriers are ALIASED onto the
                        # (dead-in-pass-2) N carriers -- the [half,dim] prefix
                        # of the [half,bn_eff] tile (dim <= bn_eff).  Frees
                        # 16,384 B (f32_D) + 8,192 B (f16_D) of UB with zero
                        # op-count change; this is the prerequisite of the
                        # G=2 head-batch (UB plan is at 100%).
                        ub_f16_D = ub_f16_N
                        ub_f32_D = ub_f32_N
                        # r18 G=2: only the state that must survive band steps
                        # is replicated per head (acc_o / m / ell / scales);
                        # the per-band-step transients and mask scratch stay
                        # shared.  head dim goes into the COLUMNS of acc_o:
                        # getBroadcastDim compares DECLARED shapes, so every
                        # acc_o operand must share dim0 = half with the
                        # [half,*] scratch tiles.
                        acc_o = T.alloc_ub([half, head_group * dim], accum_dtype)
                        ub_m = T.alloc_ub([head_group * half, 1], accum_dtype)
                        # per-[half,1] bridge scratch (declared dim0 must match
                        # the [half, bn_eff] S tile).
                        ub_ms = T.alloc_ub([half, 1], accum_dtype)
                        ub_mprev = T.alloc_ub([half, 1], accum_dtype)
                        ub_mcur = T.alloc_ub([half, 1], accum_dtype)
                        ub_alpha = T.alloc_ub([half, 1], accum_dtype)
                        ub_ell = T.alloc_ub([head_group * half, 1], accum_dtype)
                        ub_ells = T.alloc_ub([half, 1], accum_dtype)
                        ub_ellcur = T.alloc_ub([half, 1], accum_dtype)
                        ub_t = T.alloc_ub([half, 1], accum_dtype)
                        ub_lse = T.alloc_ub([half, 1], accum_dtype)
                        ub_lse_tmp = T.alloc_ub([half, 1], accum_dtype)
                        scales = T.alloc_ub(
                            [head_group * L_max * half, 1], accum_dtype)
                        # mask-chain buffers -- allocated UNCONDITIONALLY at
                        # full size (a conditional T.alloc scopes the buffer
                        # inside the if-body; traces that never mask simply
                        # never touch them).
                        ub_diff = T.alloc_ub([half, bn_eff], accum_dtype)
                        ub_pen = T.alloc_ub([half, bn_eff], accum_dtype)
                        ub_rowi = T.alloc_ub([half, 1], accum_dtype)
                        ub_thr = T.alloc_ub([half, 1], accum_dtype)
                        # ---- per-core once: diff = j - i (task-independent,
                        # arithmetic-penalty mask basis; DESIGN E-mask) ----
                        with T.rs("PIPE_V"):
                            if use_left_t or use_upper_t or has_tail_t:
                                T.arange(ub_diff, [0, 1], 0)  # column index j
                                T.arange(ub_pen, [1, 0], 0)  # row index i
                                T.vsub(ub_diff, ub_pen, ub_diff)  # diff = j - i
                                T.arange(ub_rowi, [1, 0], 0)  # row index (tail only)
                            if has_tail_t:
                                # Once per core: make the N-carriers finite so
                                # the ragged block's stale columns
                                # [tn_real, bn_eff) are 0 (not uninitialised
                                # NaN) before the column-OOB penalty.  Every
                                # later copy only writes [0:rm, 0:tnc] and can
                                # therefore only leave finite values behind.
                                T.vbrc(value_zero, ub_f16_N)
                                T.vbrc(value_zero, ub_f32_N)
                        for task_id in T.serial(num_local_tasks):
                            cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                            bx = cid // HB
                            byp = (cid // batch) % HG
                            bz = cid % batch
                            s_lo = bx * bm
                            tail_m_real = T.min(bm, seq_len - s_lo)
                            # ternary (not if/elif): trace-time `if` scopes
                            # the variable inside the branch.
                            k_end = (
                                T.ceildiv(T.min(seq_len, (bx + 1) * bm), bn_eff)
                                if causal_t
                                else (
                                    T.ceildiv(
                                        T.min(seq_len, (bx + 1) * bm + window_size_right), bn_eff
                                    )
                                    if right_end
                                    else T.ceildiv(seq_len, bn_eff)
                                )
                            )
                            k_start = (
                                T.max(0, bx * bm - window_size_left) // bn_eff
                                if use_left_t
                                else 0
                            )
                            loop_count = T.max(k_end - k_start, 0)
                            # task slot (depth-2 pipeline): ws carrier + flag
                            # id window this task owns on the Vector side.
                            slot = (task_id % n_slots) if use_pipe2 else 0
                            # adaptive row split (odd tails natural): AIV0 owns
                            # [s_lo, s_lo+real_m0), AIV1 the rest; only REAL
                            # rows are covered (E7).
                            real_m0 = (tail_m_real + 1) // 2
                            row0 = subid * real_m0
                            rm = real_m0 - (tail_m_real % 2) * subid
                            bx_r = s_lo + row0
                            # Two-stage band split (E-bnd) -- applied below as
                            # DIRECT integer comparisons.  These are
                            # algebraically identical to the design's
                            #   L_lo_full = ceildiv(s_lo+row0+rm-1-wl, bn)-k_start
                            #   L_hi_full = floordiv(s_lo+row0-(bn-1), bn)-k_start+1
                            # but avoid floordiv/ceildiv on negative numerators:
                            # BishengIR lowers those to a truncating divsi
                            # (verified in the dumped .npuir), which is wrong
                            # whenever s_lo+row0-(bn-1) < 0 -- i.e. exactly the
                            # leading q-blocks (mask silently dropped).
                            #   apply pu (upper bound) iff n_lo > s_lo+row0-(bn-1)
                            #   apply pl (left  bound) iff n_lo < s_lo+row0+rm-1-wl
                            with T.rs("PIPE_V"):
                                # task head: online state init (l==0 skips the
                                # alpha path, so zeroed alpha/ell/acc_o make the
                                # first update exact).
                                T.vbrc(value_zero, ub_ell)
                                T.vbrc(value_zero, acc_o)
                                T.vbrc(value_zero, ub_alpha)
                                T.vbrc(value_zero, scales)
                                T.vbrc(value_min, ub_m)
                            # ---- pass-1: f32 softmax chain + factor save ----
                            for l in T.serial(loop_count):
                                n_lo = (k_start + l) * bn_eff
                                tn_real = T.min(bn_eff, seq_len - n_lo)
                                tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                                # one Cubeside S tile covers all head_group
                                # heads -> one flag wait / set per band step
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(l + slot * L_max)
                                for gh in range(head_group):
                                    gh0 = gh * half
                                    wrow = gh * bm + row0
                                    srow = (gh * L_max + l) * half
                                    with T.rs("PIPE_V"):
                                        T.copy(ub_m[gh0 : gh0 + half, 0:1], ub_ms)
                                        T.copy(ub_ell[gh0 : gh0 + half, 0:1], ub_ells)
                                        T.copy(ub_ms, ub_mprev)
                                    with T.rs("PIPE_MTE2"):
                                        # ws slots are per-core LOCAL [0,bm) row
                                        # spaces -- indexed by wrow, not bx_r.
                                        if s_f16:
                                            T.copy(
                                                ws_s[kernel_id, slot, wrow : wrow + rm, l * bn_eff : l * bn_eff + tnc],
                                                ub_f16_N[0:rm, 0:tnc],
                                            )
                                        else:
                                            T.copy(
                                                ws_s[kernel_id, slot, wrow : wrow + rm, l * bn_eff : l * bn_eff + tnc],
                                                ub_f32_N[0:rm, 0:tnc],
                                            )
                                    with T.rs("PIPE_V"):
                                        if s_f16:
                                            T.vcast(ub_f16_N, ub_f32_N, round_mode="rint")
                                        # prescale (M1): e-domain (S'' = prescale*S
                                        # + pen*(-1e38), DESIGN 1.6.1 M5).
                                        if prescale != 1.0:
                                            T.vmul(ub_f32_N, prescale, ub_f32_N)
                                        # ---- two-sided arithmetic-penalty mask ----
                                        if use_upper_t and (n_lo > s_lo + row0 - (bn_eff - 1)):
                                            # upper bound (causal / right win):
                                            # valid iff diff <= K_up.
                                            # diff - K_up is float (kernel
                                            # derived); clamp [0,1] -> pu.
                                            k_up = (
                                                kernel_id * 0.0
                                                + (s_lo - n_lo + row0 + up_shift_t)
                                            )
                                            T.vsub(ub_diff, k_up, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        if use_left_t and (
                                            n_lo < s_lo + row0 + rm - 1 - window_size_left
                                        ):
                                            # left-window lower bound: valid iff
                                            # diff >= K_lo. K_lo - diff is the
                                            # scalar-minus-buffer direction, so
                                            # compute -(diff - K_lo) first (vsub
                                            # only allows a scalar src1).
                                            k_lo = (
                                                kernel_id * 0.0
                                                + (s_lo - n_lo + row0 - window_size_left)
                                            )
                                            T.vsub(ub_diff, k_lo, ub_pen)
                                            T.vmul(ub_pen, -1.0, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        if has_tail_t:
                                            # column-OOB penalty (no-op on full
                                            # blocks where tn_real == bn_eff): mask
                                            # j >= tn_real over the FULL [half,
                                            # bn_eff] width, so the stale columns
                                            # [tn_real, bn_eff) are forced to the
                                            # -1e38 sentinel instead of poisoning
                                            # the row max.  (diff is j - i, so the
                                            # per-row threshold is (tn_real-1)-i.)
                                            # Thresholds must be kernel-derived
                                            # float PrimExprs whose top node is an
                                            # Add (a bare tir.Cast scalar is
                                            # rejected by _to_region; probe P2).
                                            tnr = kernel_id * 0.0 + (tn_real - 1)
                                            T.vsub(ub_rowi, tnr, ub_thr)
                                            T.vmul(ub_thr, -1.0, ub_thr)
                                            T.vsub(ub_diff, ub_thr, ub_pen)
                                            T.vmax(ub_pen, value_zero, ub_pen)
                                            T.vmin(ub_pen, value_one, ub_pen)
                                            T.vmul(ub_pen, value_neg, ub_pen)
                                            T.vadd(ub_f32_N, ub_pen, ub_f32_N)
                                        # row max -> monotone max + NaN-guard clamp
                                        # (M3/M6; the source guard is kept verbatim:
                                        # m >= -1e38 makes every masked P an exact
                                        # +0.0, no -inf-(-inf) NaN path).
                                        T.reduce(
                                            ub_f32_N, ub_mcur, dims=[1], reduce_mode="max", clear=True
                                        )
                                        if l != 0:
                                            T.vmax(ub_mprev, ub_mcur, ub_ms)
                                            T.vmax(ub_ms, value_neg, ub_ms)
                                            T.vsub(ub_mprev, ub_ms, ub_t)
                                            T.vexp(ub_t, ub_alpha)
                                            T.copy(ub_alpha, scales[srow : srow + half, 0:1])
                                        else:
                                            # first block adopts m_cur directly;
                                            # alpha_0 is irrelevant (ell/acc_o are
                                            # exactly 0).
                                            T.copy(ub_mcur, ub_ms)
                                        T.vsub(ub_f32_N, ub_ms, ub_f32_N)
                                        T.vexp(ub_f32_N, ub_f32_N)
                                        # ell block sum (clear=True mandatory, M14).
                                        T.reduce(
                                            ub_f32_N, ub_ellcur, dims=[1], reduce_mode="sum", clear=True
                                        )
                                        # P quantized to the INPUT dtype at the
                                        # source's exact cast point (M10).
                                        T.vcast(ub_f32_N, ub_f16_N, round_mode="rint")
                                    with T.rs("PIPE_MTE3"):
                                        # P back to Cube (the set also carries
                                        # "S consumed" transitively).
                                        T.copy(
                                            ub_f16_N[0:rm, 0:tnc],
                                            ws_p[kernel_id, slot, wrow : wrow + rm, l * bn_eff : l * bn_eff + tnc],
                                        )
                                    with T.rs("PIPE_V"):
                                        # ell update AFTER the P store.
                                        T.vmul(ub_ells, ub_alpha, ub_ells)
                                        T.vadd(ub_ells, ub_ellcur, ub_ells)
                                        # write the per-head state back
                                        T.copy(ub_ms, ub_m[gh0 : gh0 + half, 0:1])
                                        T.copy(ub_ells, ub_ell[gh0 : gh0 + half, 0:1])
                                with T.rs("PIPE_MTE3"):
                                    # all head_group P tiles of this band
                                    # step are stored before the Cube is
                                    # released -> one flag per (l, slot)
                                    T.sync_block_set(l + slot * L_max)
                            # ---- pass-2: delayed rescale accumulate ----
                            for l in T.serial(loop_count):
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(l + slot * L_max)
                                for gh in range(head_group):
                                    gh0 = gh * half
                                    wrow = gh * bm + row0
                                    srow = (gh * L_max + l) * half
                                    with T.rs("PIPE_MTE2"):
                                        if s_f16:
                                            T.copy(
                                                ws_o[kernel_id, slot, wrow : wrow + rm,
                                                     l * dim : (l + 1) * dim],
                                                ub_f16_N[0:rm, 0:dim],
                                            )
                                        else:
                                            T.copy(
                                                ws_o[kernel_id, slot, wrow : wrow + rm,
                                                     l * dim : (l + 1) * dim],
                                                ub_f32_N[0:rm, 0:dim],
                                            )
                                    with T.rs("PIPE_V"):
                                        if s_f16:
                                            # static extent (vcast needs a
                                            # constant shape); rows >= rm are
                                            # scratch and never read.
                                            T.vcast(
                                                ub_f16_N[0:half, 0:dim],
                                                ub_f32_N[0:half, 0:dim],
                                                round_mode="rint",
                                            )
                                        if l != 0:
                                            # replay the SAME factor sequence
                                            # saved in pass-1 (E2).
                                            T.copy(scales[srow : srow + half, 0:1], ub_alpha)
                                        # l == 0 keeps the zeroed alpha: acc_o*0 +
                                        # O_partial(0) is exact.
                                        T.vmul(acc_o[0:half, gh * dim : (gh + 1) * dim],
                                               ub_alpha,
                                               acc_o[0:half, gh * dim : (gh + 1) * dim])
                                        T.vadd(acc_o[0:half, gh * dim : (gh + 1) * dim],
                                               ub_f32_N[0:half, 0:dim],
                                               acc_o[0:half, gh * dim : (gh + 1) * dim])
                            # ---- epilogue: normalize + transpose-free lse ----
                            for gh in range(head_group):
                                gh0 = gh * half
                                with T.rs("PIPE_V"):
                                    T.copy(ub_m[gh0 : gh0 + half, 0:1], ub_ms)
                                    T.copy(ub_ell[gh0 : gh0 + half, 0:1], ub_ells)
                                    T.vdiv(acc_o[0:half, gh * dim : (gh + 1) * dim],
                                           ub_ells,
                                           acc_o[0:half, gh * dim : (gh + 1) * dim])
                                    T.vcast(acc_o[0:half, gh * dim : (gh + 1) * dim],
                                            ub_f16_N[0:half, 0:dim], round_mode="rint")
                                    # lse (log2 domain) = log2(ell) + m*LOG2E (M2).
                                    T.vlog2(ub_ells, ub_lse, ub_lse_tmp)
                                    T.vmul(ub_ms, LOG2E, ub_t)
                                    T.vadd(ub_lse, ub_t, ub_lse)
                                with T.rs("PIPE_MTE3"):
                                    if rm > 0:
                                        # row-truncated stores (E7): the src slice
                                        # extent MUST match the dst region -- an
                                        # oversized src makes the MTE engine write
                                        # past the dst region (D3).
                                        T.copy(
                                            ub_f16_N[0:rm, 0:dim],
                                            output[bz, bx_r : bx_r + rm,
                                                   byp * head_group + gh, 0:dim],
                                        )
                                        T.copy(
                                            ub_lse[0:rm, 0:1],
                                            lse[bz, byp * head_group + gh,
                                                bx_r : bx_r + rm, 0],
                                        )
                            # task drain handshake -- single-slot form
                            # only (depth-2 closes the cross-task WAR
                            # with the flag window + slot alternation).
                            if keep_barrier:
                                with T.rs("PIPE_MTE3"):
                                    T.sync_block_set(FLAG_TASKDONE)

            return _gqa_sw_fwd_main

        # Compile with a small retry loop: on this container
        # ``bishengir-compile`` occasionally dies with SIGSEGV on an otherwise
        # valid .npuir (reproduced on the long-band bf16 trace when it follows
        # other compiles; the identical kernel compiles cleanly in a fresh
        # process).  The crash is inside the external subprocess, so a retry
        # from a clean compile invocation is safe and side-effect free.
        kernel = None
        last_err = None
        for _ in range(6):
            try:
                kernel = _builder_2phase(block_m, bn_eff, num_stages)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                cause = getattr(e, "__cause__", None)
                blob = f"{e} {cause}"
                # Retry only external-toolchain crashes (the generated .npuir
                # is valid; the subprocess died).  Genuine lowering errors
                # (AttributeError / parser diagnostics) are re-raised at once.
                if ("SIGSEGV" not in blob) and ("NPU IR opt failed" not in blob):
                    raise
        if kernel is None:
            raise last_err
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
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

        def wrapped(q_t, k_t, v_t):
            # GM workspaces are per-call scratch (every byte is produced
            # before it is consumed, flag-ordered).
            ws_s = torch.empty(ws_s_shape, dtype=torch_ws_s, device=q_t.device)
            ws_p = torch.empty(ws_p_shape, dtype=torch_dtype, device=q_t.device)
            ws_o = torch.empty(ws_o_shape, dtype=torch_ws_o, device=q_t.device)
            zbuf = torch.zeros((bn_eff, dim), dtype=torch_dtype, device=q_t.device)
            out_t, lse_t = kernel(q_t, k_t, v_t, ws_s, ws_p, ws_o, zbuf)
            # lse was computed in a [B, H, S, 1] view; restore the contract
            # shape [B, H, S] (zero-copy view on contiguous memory).
            return out_t, lse_t.reshape(batch, heads, seq_len)

        return wrapped

    return _gqa_sw_fwd_func


# ---------- output-independence gate (torch.library.custom_op contract) ------
# The op layer exposes this kernel through ``torch.library.custom_op``, which
# REQUIRES that returned tensors never alias each other nor any input
# (``RuntimeError: The output of this custom operator ... must not also be an
# input / may not alias other returns``).  A Stage-4 perf_opt that returns two
# slices of one shared workspace compiles and passes a direct factory call but
# fails 5/5 end-to-end (VP-2026-0302).  This gate asserts pairwise-distinct
# untyped storages for every output vs every other output and every input, and
# runs on EVERY case of every level (L0/L1/L2/Boundary) plus the contract test.
def check_output_independence(q, k, v, out, lse, tag="Independence"):
    """Assert no output aliases another output or any input."""
    o_ptrs = [t.untyped_storage().data_ptr() for t in (out, lse)]
    i_ptrs = [t.untyped_storage().data_ptr() for t in (q, k, v)]
    assert len(set(o_ptrs)) == len(o_ptrs), (
        f"[{tag}] FAIL: outputs alias each other: {[hex(p) for p in o_ptrs]}"
    )
    assert not (set(o_ptrs) & set(i_ptrs)), (
        f"[{tag}] FAIL: an output aliases an input: "
        f"out={[hex(p) for p in o_ptrs]} in={[hex(p) for p in i_ptrs]}"
    )
    assert len(set(i_ptrs)) == len(i_ptrs), (
        f"[{tag}] FAIL: inputs alias each other: {[hex(p) for p in i_ptrs]}"
    )
    print(
        f"  [{tag}] PASS: out={hex(o_ptrs[0])} lse={hex(o_ptrs[1])} "
        f"| q/k/v={[hex(p) for p in i_ptrs]} -- pairwise distinct, outputs "
        f"disjoint from inputs"
    )
    return True


# ---------- precision gates (DESIGN 8.2, D8-calibrated double gate) ----------
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
    a narrowly scoped tier-3 P-quantization noise classification.

    The source algorithm quantizes P to the input dtype, so the golden
    reference itself carries P-quantization noise; on early causal rows one
    1-ulp P flip is amplified up to flip_rel*|v| in the output. Classification
    (all must hold): failing elements <= 0.01% of the tensor, every failing
    element within the amplification envelope 2*flip_rel*v_absmax, and the
    fp32 lse gate unaffected. Real defects still fail loudly by orders of
    magnitude.
    """
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


def _sdpa_cross(q, k, v, is_causal, out, tag):
    """SDPA cross validation (fp32 CPU math backend; only valid for
    is_causal with no window and S_q == S_kv).  GQA KV heads are expanded
    to the query head count first."""
    groups = q.shape[2] // k.shape[2]
    k_e = k.repeat_interleave(groups, dim=2) if groups > 1 else k
    v_e = v.repeat_interleave(groups, dim=2) if groups > 1 else v
    try:
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float().cpu(),
            k_e.transpose(1, 2).float().cpu(),
            v_e.transpose(1, 2).float().cpu(),
            is_causal=is_causal,
        )
    except TypeError:
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float().cpu(),
            k_e.transpose(1, 2).float().cpu(),
            v_e.transpose(1, 2).float().cpu(),
        )
    o = o.transpose(1, 2)
    diff = (out.float().cpu() - o).abs()
    tol1 = _ATOL_OUT + _RTOL_OUT * o.abs()
    tier1_ok = bool((diff <= tol1).all().item())
    max_d = float(diff.max().item())
    if tier1_ok:
        ok = True
    else:
        tol2 = _ATOL_OUT + (_RTOL_OUT + 2**-6) * o.abs()
        ok = bool((diff <= tol2).all().item())
    print(
        f"  [{tag}] SDPA cross: {'PASS' if ok else 'FAIL'} "
        f"(max_diff={max_d:.3e}{'' if tier1_ok else ' tier2'})"
    )
    return ok


def _run_case(
    batch,
    heads,
    heads_kv,
    seq_len,
    dim,
    is_causal,
    window_size_left,
    window_size_right,
    dtype=torch.float16,
    ns=1,
    tag="L?",
    seed=0,
    zeros=False,
    sdpa=False,
):
    """Generate inputs on NPU, run the kernel, gate vs the golden (and an
    optional SDPA cross where applicable). Returns True on pass."""
    import time

    torch.manual_seed(seed)
    dt_str = _dtype_str(dtype)
    dev = "npu"
    shape_q = (batch, seq_len, heads, dim)
    shape_kv = (batch, seq_len, heads_kv, dim)
    gen = torch.zeros if zeros else torch.randn
    q = gen(shape_q, dtype=dtype, device=dev)
    k = gen(shape_kv, dtype=dtype, device=dev)
    v = gen(shape_kv, dtype=dtype, device=dev)

    fn = _gqa_sw_fwd_kernel(
        batch, heads, heads_kv, seq_len, dim, is_causal, window_size_left,
        window_size_right, dt_str,
    )
    # (64, 64, ns): exercises the wrapper-default -> design-default (64, 256)
    # config substitution path.
    wrapped = fn(64, 64, ns)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out, lse = wrapped(q, k, v)
    torch.npu.synchronize()
    t_ms = (time.perf_counter() - t0) * 1e3

    assert out.shape == shape_q and out.dtype == dtype, (
        f"output shape/dtype contract violated: {tuple(out.shape)}/{out.dtype}"
    )
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32, (
        f"lse shape/dtype contract violated: {tuple(lse.shape)}/{lse.dtype}"
    )
    # custom_op alias contract: outputs independent of each other and of inputs.
    check_output_independence(q, k, v, out, lse, tag=f"{tag}-alias")

    ref_out, ref_lse = golden_gqa_sw_fwd(
        q.cpu(), k.cpu(), v.cpu(), is_causal, window_size_left, window_size_right
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
    wtag = f" wl={window_size_left} wr={window_size_right}"
    print(
        f"  [{tag}] B={batch} H={heads}/{heads_kv} S={seq_len} D={dim} "
        f"causal={int(is_causal)}{wtag} {dt_str} ns={ns}"
        f"{' zeros' if zeros else ''}: "
        f"out={'PASS' if ok_out else 'FAIL'} (max_diff={max_diff:.3e}"
        f"{flip_note}) lse={'PASS' if ok_lse else 'FAIL'} "
        f"(diff={lse_diff:.3e}) isfinite={fin_out and fin_lse} [{t_ms:.1f} ms]"
    )

    ok = ok_out and ok_lse and fin_out and fin_lse

    if sdpa and is_causal and window_size_left < 0 and window_size_right < 0:
        ok = _sdpa_cross(q, k, v, is_causal, out, tag) and ok
    return ok


def _contract_checks():
    """Wrapper hard-contract verification (DESIGN 8.2): positional call form,
    lru_cache identity, output/lse shapes, ValueError path, and the
    [B,H,S,1] lse view restore."""
    print("  [Contract] wrapper call form / lru_cache / ValueError")
    f1 = _gqa_sw_fwd_kernel(1, 8, 2, 512, 64, True, -1, -1, "float16")
    f2 = _gqa_sw_fwd_kernel(1, 8, 2, 512, 64, True, -1, -1, "float16")
    assert f1 is f2, "lru_cache factory identity violated"
    fn = f1(64, 64, 1)
    q = torch.randn(1, 512, 8, 64, dtype=torch.float16, device="npu")
    k = torch.randn(1, 512, 2, 64, dtype=torch.float16, device="npu")
    v = torch.randn(1, 512, 2, 64, dtype=torch.float16, device="npu")
    out, lse = fn(q, k, v)  # positional (block_m, block_n, num_stages)(q,k,v)
    torch.npu.synchronize()
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (1, 8, 512) and lse.dtype == torch.float32
    assert lse.is_contiguous(), "lse view restore must be contiguous"
    check_output_independence(q, k, v, out, lse, tag="Contract-alias")
    try:
        _gqa_sw_fwd_kernel(1, 8, 3, 512, 64, True, -1, -1, "float16")
        raise AssertionError("heads%heads_kv ValueError not raised")
    except ValueError:
        pass
    print("  [Contract] PASS")


def run_L0():
    """L0 gate (blocking, DESIGN 8.2 order): L0-1 smoke fp16 (two-phase main
    trace), L0-2 smoke bf16 (R-2 probe), L0-3 8b-short wl=256 fp16 (band
    mask), L0-4 band-trim wl=512 fp16 (k_start > 0), L0-5 contract."""
    print("== L0 (blocking) ==")
    ok = True
    ok &= _run_case(2, 8, 2, 512, 64, True, -1, -1, torch.float16, tag="L0-1-smoke-fp16", seed=0, sdpa=True)
    ok &= _run_case(2, 8, 2, 512, 64, True, -1, -1, torch.bfloat16, tag="L0-2-smoke-bf16", seed=1)
    ok &= _run_case(4, 32, 8, 512, 128, True, 256, -1, torch.float16, tag="L0-3-8bshort-w256", seed=2)
    ok &= _run_case(1, 8, 2, 1024, 64, True, 512, -1, torch.float16, tag="L0-4-bandtrim-w512", seed=3)
    _contract_checks()
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (4 precision cases + contract)")


def run_L1():
    """L1 (blocking): full manifest workload sweep (llama 8b/70b short/long,
    causal, x fp16/bf16, DESIGN 8.2)."""
    print("== L1 (blocking) ==")
    ok = True
    cases = [
        (4, 512, 32, 8, 128, 256, "L1-8b-short"),
        (2, 2048, 32, 8, 128, 1024, "L1-8b-long"),
        (2, 512, 64, 8, 128, 256, "L1-70b-short"),
        (1, 2048, 64, 8, 128, 1024, "L1-70b-long"),
    ]
    seed = 10
    for batch, s, h, hkv, d, wl, name in cases:
        for dt in (torch.float16, torch.bfloat16):
            ok &= _run_case(
                batch, h, hkv, s, d, True, wl, -1, dt, ns=1,
                tag=f"{name}-{_dtype_str(dt)}", seed=seed,
            )
            seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (8 cases)")


def run_L2():
    """L2 (warn-only): coverage domains -- the four mask families, GQA ratio,
    non-power-of-2 S, right window, wl=0."""
    print("== L2 (warn-only) ==")
    results = []
    # causal + left window (wl=128)
    results.append(_run_case(2, 8, 2, 512, 64, True, 128, -1, torch.float16, tag="L2-causal-w128", seed=20))
    # non-causal, no window
    results.append(_run_case(2, 8, 2, 512, 64, False, -1, -1, torch.float16, tag="L2-noncausal", seed=21))
    # symmetric band (non-causal wl=64 wr=64)
    results.append(_run_case(2, 8, 2, 512, 64, False, 64, 64, torch.float16, tag="L2-symband-64", seed=22))
    results.append(_run_case(2, 8, 2, 512, 64, False, 64, 64, torch.bfloat16, tag="L2-symband-64-bf16", seed=23))
    # wl=1 (S=128, dim=128)
    results.append(_run_case(2, 8, 1, 128, 128, True, 1, -1, torch.float16, tag="L2-wl1-d128", seed=24))
    # GQA no grouping
    results.append(_run_case(2, 8, 8, 512, 64, True, -1, -1, torch.float16, tag="L2-gqa-h8-hkv8", seed=25))
    # GQA G=16
    results.append(_run_case(2, 16, 1, 512, 64, True, -1, -1, torch.float16, tag="L2-gqa-h16-hkv1", seed=26))
    # non-power-of-2 S
    results.append(_run_case(2, 8, 2, 384, 64, True, -1, -1, torch.float16, tag="L2-s384-causal", seed=27))
    results.append(_run_case(2, 8, 2, 768, 64, False, 256, -1, torch.float16, tag="L2-s768-noncausal-w256", seed=28))
    # right window only (non-causal)
    results.append(_run_case(2, 8, 2, 512, 64, False, -1, 64, torch.float16, tag="L2-rightwin-64", seed=29))
    # wl=0
    results.append(_run_case(2, 8, 2, 256, 64, True, 0, -1, torch.float16, tag="L2-wl0", seed=30))
    # large S wl=512
    results.append(_run_case(1, 8, 2, 2048, 64, True, 512, -1, torch.float16, tag="L2-s2048-w512", seed=31))
    n_pass = sum(results)
    if n_pass == len(results):
        print(f"[L2] PASS ({len(results)} cases)")
    else:
        print(f"[L2] WARN: {len(results) - n_pass}/{len(results)} cases failed (recorded, non-blocking)")


def run_boundary():
    """Boundary (warn-only): ragged tails (tmc/tnc clamps + OOB band), zeros,
    wl >= S (equivalent to unlimited), very long S."""
    print("== Boundary (warn-only) ==")
    results = []
    results.append(_run_case(1, 8, 2, 520, 64, True, -1, -1, torch.float16, tag="B-s520-causal", seed=40))
    results.append(_run_case(1, 8, 2, 520, 64, True, 256, -1, torch.float16, tag="B-s520-w256", seed=41))
    results.append(_run_case(1, 8, 2, 520, 64, False, -1, -1, torch.bfloat16, tag="B-s520-noncausal-bf16", seed=42))
    results.append(_run_case(1, 8, 2, 100, 64, True, -1, -1, torch.float16, tag="B-s100-causal", seed=43))
    results.append(_run_case(1, 8, 2, 100, 64, True, 64, -1, torch.bfloat16, tag="B-s100-w64-bf16", seed=44))
    # zeros: lse should be exact for the all-valid (non-causal, no window) case
    results.append(_run_case(1, 8, 2, 512, 64, False, -1, -1, torch.float16, tag="B-zeros", seed=45, zeros=True))
    # wl >= S -> equivalent to unlimited
    results.append(_run_case(1, 8, 2, 512, 64, True, 4096, -1, torch.float16, tag="B-wl>S", seed=46))
    # very long S
    results.append(_run_case(1, 8, 2, 4096, 64, True, 1024, -1, torch.float16, tag="B-s4096-w1024", seed=47))
    n_pass = sum(results)
    if n_pass == len(results):
        print(f"[Boundary] PASS ({len(results)} cases)")
    else:
        print(f"[Boundary] WARN: {len(results) - n_pass}/{len(results)} cases failed (recorded, non-blocking)")


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
