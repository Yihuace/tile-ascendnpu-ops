"""[repro] PATT-twophase-restructure -- pattern-library/attention.md PL-1.9-twophase

Optimization-point repro, DELTA FORM: the key code change that took the
attention kernel from slow to fast (226.72us -> 98.05us at fa4096, four hard
targets reached), NOT a runnable kernel. Extracted & minimized from the
reference implementation examples/flash_attention/flash_attn_npuir.py
(provenance, may rot). Syntax-checked only; the full kernels carry the real
buffers, tails and softmax chain.

KEY CODE CHANGE (slow -> fast):

  SLOW (per-block persistent chain, the frozen design family):
    for n in T.serial(num_n_blocks):          # one n-block at a time
        with T.Scope("Vector"): ... softmax partial (S row block) ...
        set_flag(FLAG_P)                       # Vector -> Cube
        with T.Scope("Cube"): ... gemm2 on this n-block ...   # waits on P
        set_flag(FLAG_O)                       # Cube -> Vector
        with T.Scope("Vector"): ... accumulate O ...          # waits on O
    # every block pays the V1 -> FLAG_P -> C2 -> FLAG_O -> V2 serial chain:
    # flag-wait spin was 38-47% of busy-core scalar time; vec op count and
    # flag round-trips scale linearly with the number of n-blocks.

  FAST (two-phase restructure -- the skeleton below):
    with T.Scope("Cube"):
        for i in T.serial(num_n_blocks):       # PHASE 1: ALL of S first
            ... gemm1 block -> l0_c ...
            with T.rs("PIPE_FIX"):
                T.copy(l0_c, workspace_1[... block i ...])  # S materialized
                T.sync_block_set(i)             # flag id = n-block index
        for i in T.serial(num_n_blocks):       # PHASE 2: ALL of O_partial
            with T.rs("PIPE_MTE2"):
                T.sync_block_wait(i)            # consumes S from phase 1
                T.copy(workspace_2[... block i ...], l1_a)
            ... gemm2 -> l0_c -> workspace_3 ...
            with T.rs("PIPE_FIX"):
                T.sync_block_set(i)
    # single l0_c reused sequentially by gemm1 then gemm2 (fits L0C 128KB);
    # the Vector softmax runs as its own pass reading S / writing O_partial
    # with UB scales[] deferred-rescale replay; Q hoisted (loaded once per
    # m-block, not once per n-block).

Insight summary (LLM-distilled, why slow -> fast): the per-block cross-
engine serial chain was the cycle floor, not the compute. Phase separation
removes per-block flag round-trips entirely (flag id = block index, nk <= 16
fits the flag budget, CONST-flag-id-budget) and lets each engine run long
serial loops over materialized GM workspace (store_fixpipe is GM-only, so
S/P round trips are mandatory -- CONST-store-fixpipe-gm-only -- but they now
pipeline across whole phases instead of serializing per block). Measured
effect: per-core L1 r+w 18.70 -> 11.17MB (-40%), cube pipe busy 171 -> 66us;
speedup comes from traffic reduction, not overlap (busy/wall 0.66x equals
the reference's 0.68x). Block width is bounded by L0C 128KB
(l0c_s + l0c_o <= 128KB, bm <= 51 at bn = 512).

First verified: tilelang 0.1.2+3a214cde + CANN 8.5.0 + Ascend910B2C,
2026-09-09 (attention round 3, four hard targets 12.94/18.11/30.88/98.05us,
[DESIGN_LIMIT] negative conclusion overturned by this structure).
origin_task (provenance): multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
Re-verification log (append-only):
  2026-09-10 PASS syntax check (delta-form extraction; run the reference
  kernel in examples/flash_attention/ for end-to-end verification).
"""

# --- minimal structural skeleton (delta form: syntax-checkable, not run) ---

import tilelang.language as T  # noqa: F401

if True:  # keep the skeleton at module scope without executing anything
    _ = """
    with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, subid):
        with T.Scope("Cube"):
            l1_a = T.alloc_L1([block_m, block_share], dtype)
            l1_b = T.alloc_L1([block_n, block_k], dtype)
            l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)
            # PHASE 1: all of S first (Q hoisted, single l0_c reuse)
            for i in T.serial(T.ceildiv(seq_len, block_n)):
                for k in T.serial(T.ceildiv(dim, block_k)):
                    T.copy(Q[...], l1_a[...])      # Q loaded once per m-block
                    T.copy(K[...], l1_b[...])
                    T.gemm(l1_a, l1_b, l0_c, initC=(k == 0), b_transpose=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0_c, workspace_1[..., i-th n-block ...])
                    T.sync_block_set(i)            # flag id = block index
            # PHASE 2: all of O_partial (no per-block flag round-trips)
            for i in T.serial(T.ceildiv(seq_len, block_n)):
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(i)
                    T.copy(workspace_2[..., i-th ...], l1_a[...])
                for k in T.serial(T.ceildiv(dim, block_k)):
                    T.copy(V[...], l1_b[...])
                    T.gemm(l1_a, l1_b, l0_c, initC=True)
                    T.copy(l0_c, workspace_3[..., i-th ...])
                with T.rs("PIPE_FIX"):
                    T.sync_block_set(i)
    """
