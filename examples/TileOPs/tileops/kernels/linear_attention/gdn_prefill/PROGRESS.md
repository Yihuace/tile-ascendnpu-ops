# GDN prefill — migration progress (resume point)

> Op: `GatedDeltaNetPrefillFwdOp` (baseline `GDNFwdOp`). Toolchain: CANN **9.0.0**,
> tilelang 0.1.2 (dev root), Ascend910B3 × 1. Mode: **Developer** for the Vector
> kernels; the Cube-containing ones will need **Expert** (see `examples/gdn_prefill/DESIGN.md`).

## Status: 2 / 6 kernels implemented and verified

| # | kernel | status | test evidence |
|---|---|---|---|
| 1 | `prefill_chunk_local_cumsum_bthd_tl` | ✅ **done** | L0 0.0e+00 / L0b 0.0e+00 / L1 4.77e-07 / L2 4.77e-07, 2.38e-07 |
| 2 | `prefill_blocksolve_A_bthd` | ⬜ pending | needs Cube (`T.gemm`) + block solve — hardest |
| 3 | `get_warmup_chunks` | ✅ **done** | L0/L0b/L1/boundary all bit-exact vs golden |
| 4 | `correct_initial_states` | ⬜ pending | uses `T.gemm` in the fallback path → Expert mode |
| 5 | `fused_gdr_h` | ⬜ pending | large (prepare-h pipeline) |
| 6 | `fused_gdr_fwd` | ⬜ pending | large (chunk recurrence) |

## Workflow state (statectl)

```
stage_plan      = [0, 1, 2, 3, 5]
stage 0/1/2/3   = completed    (gate passed each time)
current_stage   = 5 (INTEGRATE)  ← op-level; per-function progress lives in .migration_state.json
migration funcs : prefill_chunk_local_cumsum_bthd_tl = done, get_warmup_chunks = done, others pending
```

> Note: the state machine does not allow re-entering Stage 1/2/3 after completion, so the
> per-function loop is tracked with `statectl migration func <name> --status ...`.

## How to resume

```bash
cd /mnt/workspace/npuir-migration/examples/TileOPs
export TILELANG_ASCEND_MODE=Developer

# kernel 1
python3 tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py --level all

# kernel 3
python3 -c "from tileops.kernels.linear_attention.gdn_prefill.gdn_prefill import run_L0_warmup, run_L1_warmup, test_boundary_warmup; run_L0_warmup(); run_L1_warmup(); test_boundary_warmup()"
```

## Next kernel

`prefill_blocksolve_A_bthd` (function 2). Source: `prefill_prepare.py:158`.
Needs: Blocked Neumann solve `A = (I + S_sys)^{-1}`-style, Cube `T.gemm`, and a
fp16 scratch round-trip. The structure was validated in
`examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py` (err 9.7e-3)
and `14_two_loop_handshake.py` (err 9.8e-4) — both run in **Expert** mode.

---

## Kernel 2 `prefill_blocksolve_A_bthd` — implementation plan (read from source 2026-10-17)

Source: `prefill_prepare.py:158` (`_prefill_blocksolve_A_bthd_tl`).

**Contract**

```
k        [1, total_tokens, key_head, dim_k]   T        (dim_k ∈ {64,128})
g        [1, total_tokens, head]              float32
beta     [1, total_tokens, head]              T
cu_seqlens [num_sequences+1]                  int32
A        [1, total_tokens, head, chunk_size]  T        (chunk_size = 64, out)
k_rnorm  [1, rnorm_tokens, key_head]          float32   (only when l2norm; out of scope)
```

**Structure (the hard part)**

- Parameters: `block_t=64`, `block_c=16`, `block_k=64`, `accum=fp32`, `solve_dtype=dtype`.
- A chunk of 64 rows is solved as a **4×4 grid of 16×16 sub-blocks**:
  `G_ij = β_i · G_strict_ij ⊙ (k_i k_jᵀ)` for `i ≥ j` (10 accumulators `G00..G33`,
  strictly-lower sub-blocks only), then **block forward substitution** over the 4
  diagonal 16×16 blocks (each an in-place small solve on `a_s`/`i_s`/`work_s`),
  producing the 4×16 block of `A` per chunk.
- `group = head // key_head`: the solve reads the key at head `hid // group`.

**NPU porting notes**

1. Needs **Cube** (`T.gemm` for the 10 `k_i k_jᵀ` products) → `ascend_mode = "Expert"`.
   The single-Cube + single-Vector scope rule applies (`02_gm_handoff_sync.py`,
   `14_two_loop_handshake.py` both pass in Expert).
2. The 16×16 diagonal solves are small and sequential: implement them on the Vector side
   (UB) via explicit forward substitution, or fold into the Cube pipeline with a scratch
   round-trip (`13_l0c_fp16_scratch_cann9.py` validated: fp32→fp16 scratch, err 9.7e-3).
3. Keep the accumulator in fp32 (the source does); only the scratch round-trip goes fp16.
4. `l2norm` / `uniform_seq_len == 0` (varlen) paths are out of scope — assert them off.
5. Suggested L0: `total_tokens=64, key_head=1, head=1, dim_k=64`, compare `A` against a
   pure-torch reference that does the same 4×4 block solve in fp64.

**Reference for the golden**: solve `(I + S_sys) X = rhs` directly in fp64 with
`torch.linalg.solve` on the 64×64 chunk matrix, then compare to the blocked result —
that also cross-checks the block-solve algebra.
