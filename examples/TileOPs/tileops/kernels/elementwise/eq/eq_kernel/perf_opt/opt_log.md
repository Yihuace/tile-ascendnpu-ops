# Stage 4 Optimization Log — `eq_fwd_kernel` (EqFwdOp)

- project: `eq`
- operator: `eq_fwd_kernel`
  (`examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/`)
- benchmark source: `examples/TileOPs/benchmarks/ops/bench_eq.py::test_eq_manifest_bench`
- baseline (read-only, never modified): `../eq_fwd_kernel.py`
  (sha256 `da68c2379e05e7843c21fbfd0bdb68058288f2c1463ded198c3313b0b8509ce4`)
- deliverable: `perf_opt/eq_fwd_kernel.py`
  (sha256 `5cdfae467e4906d1106569fc343298d85ca4789feb1377d138f4d98ad019d52a`)
- primary metric: `msprof op` `Task Duration(us)` (kernel-only, median of the
  `--launch-count=15` launches in one `msprof op` run). No other timing口径 is used.
- Roofline Ratio = the repo's single-operator report formula, reproduced by
  `perf_opt/roofline_ratio.py` (mirrors
  `tileops/benchmark/msprof_roofline.compute_from_csvs`, GM-bounded branch):
  `ratio = (Σ aiv/aic r0+r1 read_cache_miss_allocate × 128) / (Task Duration(s) × 1.8e12)`,
  BW = 1.8 TB/s. **The numerator is GM *read* traffic only** (writes are not in the report
  formula), which bounds `ratio ≤ read/(read+write) · B_eff/1.8`.
- AICore: Ascend910, 40 physical Vector cores (`NPUUtils.get().get_aicore_num()*2`).
- Sister reference reused: `BitwiseAndFwdOp`
  (`../bitwise_and/bitwise_and_kernel/perf_opt/opt_log.md`) — same elementwise/broadcast
  source shape; its per-dtype `block_size` + outer-broadcast-operand hoist were carried over.

> Pre-existing partial artifacts were read first (block_size scan logs `logs/scan/*`,
> `eq_fwd_kernel_round1_op1.py`/`_op2.py`, `workload_inventory.json`, the round-1 candidate
> profiles). The run was resumed from r1_op2 (block_size + hoist, unmeasured), which is the
> deliverable.

## Workload inventory (from the migrated benchmark)

`workload_inventory.json` — 9 tune cases (3 shape classes × 3 dtypes) + 3 smoke cases
(`smoke-1k` fp32/fp16/bf16, precision-only). No skipped cases. The fp16/bf16 smoke ids carry a
benchmark `full` mark conflicting with the independent `smoke` token; excluded from performance
tuning and recorded as such.

## Baseline — Performance Test Data (round 0, `phase=baseline`, bs=1024)

The shipped default (`EqFwdKernel.default_config`) is `block_size = threads*npt = 256*4 = 1024`
for the float dtypes, so the baseline is measured at bs=1024 (the config the benchmark/wrapper
actually uses).

| kernel_id | workload_id | dtype | shape (a \| b) | Task Duration(us) | AICore mte2/mte3/vec/scalar | GM read BW | Roofline Ratio | raw profile |
|---|---|---|---|---:|---|---:|---:|---|
| …::main | hidden-state-prefill-float16 | fp16 | (2048,4096) \| = | 83.658 | 0.806/0.332/0.165/0.124 | 9.7 GB/s | 22.28 % | `profiles/baseline/round0/baseline_8ba8496a/hidden-state-prefill-float16_f48f537e` |
| …::main | hidden-state-prefill-bfloat16 | bf16 | (2048,4096) \| = | 91.198 | 0.724/0.303/0.261/0.144 | 8.9 GB/s | 20.44 % | `…/hidden-state-prefill-bfloat16_274d79ca` |
| …::main | hidden-state-prefill-float32 | fp32 | (2048,4096) \| = | 94.438 | 0.806/0.305/0.169/0.110 | 17.2 GB/s | 39.48 % | `…/hidden-state-prefill-float32_dac1dbdc` |
| …::main | cnn-feat-broadcast-float16 | fp16 | (16,256,56,56) \| (256,1,1) | 699.806 | 0.943/0.165/0.046/0.034 | 1.2 GB/s | 1.77 % | `…/cnn-feat-broadcast-float16_f1cddcf9` |
| …::main | cnn-feat-broadcast-bfloat16 | bf16 | (16,256,56,56) \| (256,1,1) | 578.868 | 0.896/0.182/0.089/0.050 | 1.4 GB/s | 2.13 % | `…/cnn-feat-broadcast-bfloat16_f5a51caa` |
| …::main | cnn-feat-broadcast-float32 | fp32 | (16,256,56,56) \| (256,1,1) | 406.852 | 0.883/0.234/0.097/0.066 | 4.0 GB/s | 6.73 % | `…/cnn-feat-broadcast-float32_2ff59659` |
| …::main | row-broadcast-float32 | fp32 | (4096,4096) \| (1,4096) | 185.296 | 0.823/0.321/0.178/0.084 | 18.1 GB/s | 20.13 % | `…/row-broadcast-float32_7e902357` |
| …::main | row-broadcast-float16 | fp16 | (4096,4096) \| (1,4096) | 183.876 | 0.836/0.338/0.168/0.088 | 9.9 GB/s | 10.14 % | `…/row-broadcast-float16_410291d8` |
| …::main | row-broadcast-bfloat16 | bf16 | (4096,4096) \| (1,4096) | 166.257 | 0.727/0.329/0.283/0.116 | 9.6 GB/s | 11.22 % | `…/row-broadcast-bfloat16_00d04600` |

Coalesced shapes (compile-time, from `coalesce_broadcast_dims`):
`hidden` → (8388608,) R=1 L=8.39 M; `cnn-feat` → (16,256,3136) R=4096 L=3136 b_stride-inner=0;
`row-broadcast` → (4096,4096) R=4096 L=4096 a_stride-inner=b_stride-inner=1 (b outer stride 0).

### Baseline bottleneck ranking (by measured burden)

1. `cnn-feat-broadcast-*` (407–700 µs; 1.8–6.7 %): smallest per-task granularity — one 3136-elem
   row per task with a per-row scalar-b (`T.vbrc`), tiny MTE2 bursts (1–4 GB/s).
2. `row-broadcast-*` (166–185 µs; 10–20 %): the (1,4096) b row is re-read for every one of the
   4096 rows (64 MB fp32 / 33 MB fp16 of redundant GM→UB traffic; MTE2-bound, mte2_ratio≈0.82–0.92).
3. `hidden-state-prefill-*` (84–94 µs; 20–39 %): flat contiguous stream, but bs=1024 gives only
   1024-elem tiles → SMALL bursts; the healthiest class but far from the UB-filling tile.

### Design estimate vs measured (D-2 backfill)

`DESIGN.md` §1.6.0 estimate (flow/emit terms at `B_eff ≈ 1.05 TB/s`, 40-core scaling) vs baseline:

| workload | design estimate (lower bound) | measured baseline | deviation |
|---|---:|---:|---|
| hidden-state-prefill fp32 | ≈77 µs (emit-dominated) | 94.4 µs | 1.23× slower |
| cnn-feat-broadcast fp32 | ≈205 µs (emit-dominated) | 406.9 µs | 1.98× slower |
| row-broadcast fp32 | ≈154 µs (emit) / 144 µs (b-reread) | 185.3 µs | 1.20× slower |

All deviations <2× (recorded, not a design contradiction per `hardware-cost-model.md` §3). The
cnn-feat 1.98× gap confirms the design's "emit-dominated" diagnosis (the per-task scalar-b DMA +
`vbrc` was under-counted); hidden/row sit on the flow term. The design's `B_eff ≈ 1.05 TB/s`
scaling constant is conservative — the tuned hidden fp32 reaches 67.1 MB / 48.7 µs ≈ 1.38 TB/s
read and row-fp32 reaches ≈1.1 TB/s read.

## Round 1 — per-dtype `block_size` + outer-broadcast-operand hoist

Base = baseline. Two single-change branches derived from the same base:

- `r1_op1` (`eq_fwd_kernel_round1_op1.py`): per-dtype `block_size` module constants
  (`TUNED_BLOCK_BY_DTYPE`: fp16/bf16 8192, fp32 4096) replacing the single 1024 float default.
- `r1_op2` (`eq_fwd_kernel_round1_op2.py`): `r1_op1` **plus** the outer-broadcast-operand hoist —
  when an operand is broadcast across *every* outer dim (all outer strides 0) and the row fits one
  chunk (`chunks == 1`), its GM→UB materialisation is hoisted out of the per-task `T.serial` loop
  and executed once per core. The hoist is provably inert for `hidden` (R=1) and `cnn-feat`
  (b outer stride 1), so only `row-broadcast` changes.

### Candidate vs current best — round 1 (B2 comparison table)

Current best = baseline (`../eq_fwd_kernel.py`). AICore util columns = mte2/mte3/vec/scalar.

| candidate | workload_id | Task Duration(us) | AICore mte2/mte3/vec/scalar | mem (main read BW / GM→UB KB) | L0 | vs current best |
|---|---|---:|---|---|:--:|---:|
| baseline | hidden-state-prefill-float16 | 83.658 | 0.806/0.332/0.165/0.124 | 9.7 GB/s / 820 | pass | — |
| r1_op1 | hidden-state-prefill-float16 | 26.699 | 0.789/0.223/0.255/0.084 | 35.2 GB/s / — | pass | +68.1 % |
| **r1_op2** | hidden-state-prefill-float16 | **26.159** | 0.791/0.227/0.257/0.081 | 35.5 GB/s / — | pass | **+68.7 %** |
| baseline | hidden-state-prefill-bfloat16 | 91.198 | 0.724/0.303/0.261/0.144 | 8.9 GB/s / 820 | pass | — |
| r1_op1 | hidden-state-prefill-bfloat16 | 28.359 | 0.579/0.240/0.518/0.089 | 32.4 GB/s / — | pass | +68.9 % |
| **r1_op2** | hidden-state-prefill-bfloat16 | **27.739** | 0.569/0.245/0.523/0.090 | 32.7 GB/s / — | pass | **+69.6 %** |
| baseline | hidden-state-prefill-float32 | 94.438 | 0.806/0.305/0.169/0.110 | 17.2 GB/s / 1640 | pass | — |
| r1_op1 | hidden-state-prefill-float32 | 51.039 | 0.813/0.175/0.201/0.066 | 35.9 GB/s / — | pass | +45.9 % |
| **r1_op2** | hidden-state-prefill-float32 | **49.619** | 0.814/0.175/0.199/0.066 | 35.5 GB/s / — | pass | **+47.5 %** |
| baseline | cnn-feat-broadcast-float16 | 699.806 | 0.943/0.165/0.046/0.034 | 1.2 GB/s / 869 | pass | — |
| r1_op1 | cnn-feat-broadcast-float16 | 156.677 | 0.865/0.264/0.183/0.038 | 4.0 GB/s / — | pass | +77.6 % |
| **r1_op2** | cnn-feat-broadcast-float16 | **167.717** | 0.873/0.258/0.170/0.036 | 3.7 GB/s / — | pass | **+76.0 %** |
| baseline | cnn-feat-broadcast-bfloat16 | 578.868 | 0.896/0.182/0.089/0.050 | 1.4 GB/s / 869 | pass | — |
| r1_op1 | cnn-feat-broadcast-bfloat16 | 134.717 | 0.626/0.236/0.427/0.051 | 4.6 GB/s / — | pass | +76.7 % |
| **r1_op2** | cnn-feat-broadcast-bfloat16 | **146.857** | 0.656/0.233/0.392/0.047 | 4.2 GB/s / — | pass | **+74.6 %** |
| baseline | cnn-feat-broadcast-float32 | 406.852 | 0.883/0.234/0.097/0.066 | 4.0 GB/s / 1687 | pass | — |
| r1_op1 | cnn-feat-broadcast-float32 | 83.798 | 0.738/0.334/0.289/0.073 | 15.1 GB/s / — | pass | +79.4 % |
| **r1_op2** | cnn-feat-broadcast-float32 | **89.558** | 0.756/0.323/0.269/0.067 | 14.1 GB/s / — | pass | **+78.0 %** |
| baseline | row-broadcast-float32 | 185.296 | 0.823/0.321/0.178/0.084 | 18.1 GB/s / 3280 | pass | — |
| r1_op1 | row-broadcast-float32 | 175.076 | 0.917/0.264/0.101/0.021 | 18.0 GB/s / — | pass | +5.5 % |
| **r1_op2** | row-broadcast-float32 | **61.299** | 0.747/0.290/0.310/0.064 | 28.0 GB/s / — | pass | **+66.9 %** |
| baseline | row-broadcast-float16 | 183.876 | 0.836/0.338/0.168/0.088 | 9.9 GB/s / 1640 | pass | — |
| r1_op1 | row-broadcast-float16 | 183.156 | 0.919/0.237/0.125/0.022 | 8.6 GB/s / — | pass | +0.4 % |
| **r1_op2** | row-broadcast-float16 | **51.739** | 0.710/0.313/0.465/0.077 | 16.2 GB/s / — | pass | **+71.9 %** |
| baseline | row-broadcast-bfloat16 | 166.257 | 0.727/0.329/0.283/0.116 | 9.6 GB/s / 1640 | pass | — |
| r1_op1 | row-broadcast-bfloat16 | 183.316 | 0.822/0.210/0.279/0.026 | 8.6 GB/s / — | pass | −10.3 % |
| **r1_op2** | row-broadcast-bfloat16 | **79.578** | 0.437/0.197/0.668/0.060 | 10.3 GB/s / — | pass | **+52.1 %** |

- Memory metrics: `hidden-fp32` main read BW 17.2→35.5 GB/s (+2.1×, larger tile → larger MTE2
  bursts); `cnn-fp32` 4.0→14.1 GB/s (+3.5×); `row-*` read BW roughly unchanged but the hoist
  deleted 4096 redundant per-row b `T.copy` calls, so the same MTE2 budget serves ~3× more work.
- **Noise**: `r1_op1` vs `r1_op2` on `cnn-feat` differ by 4–13 % even though the hoist is inert for
  cnn — this is the measured run-to-run spread of the shared host (≈7–13 %). All hoist gains for
  `row-broadcast` (52–72 %) are far above that band.
- **Winner / merge:** `r1_op2` (block_size constants **+** hoist) is the round-1 winner and was
  merged into `perf_opt/eq_fwd_kernel.py`. `block_size` values are exposed at module level
  (`TUNED_BLOCK_BY_DTYPE` / `TUNED_BLOCK_SIZE`) for the Op-layer wrapper switch block.
  `r1_op1` is retained as the isolated single-change evidence (it fixes `hidden`/`cnn` but leaves
  `row-broadcast` unchanged).

### Design-estimate adjudication (D-2, hidden fp32)

Design "≈77 µs emit-dominated" vs r1_op2 49.6 µs → the emit term is ~1.55× over-estimated for
this device (larger tiles amortise the ~0.5 µs/op launch constant better than modelled).

## Round 2 — DESIGN §1.6.3 alternative #4 (2-D tile) adjudication

`DESIGN.md` §1.6.3 freezes an experiment-adjudication plan: alternative #4 (`[R_blk, L]` 2-D tile,
recording-plan threshold "备选 latency < 主选 × 0.97 → flip; broadcast class routes #4, same-shape
routes #1") must be A/B tested on the representative shapes `row-broadcast fp32` and
`cnn-feat fp32`. This round executes that plan.

- `r2_op1` (`eq_fwd_kernel_round2_op1.py`): base = r1_op2. Adds a compile-time 2-D-tile path for
  the outer-broadcast row class. When b is periodic with period `L` (all outer b strides 0) and a is
  contiguous across rows (`a_strides[0] == L`), a tile of `block_size = R_blk·L` consecutive
  elements is exactly `R_blk` rows: a/y stay flat-contiguous and only b repeats, so the op is
  re-interpreted as a flat stream of `R_blk·L`-element tiles whose b operand is the period-L row
  repeated `R_blk` times. `fp32` block → 8192 (`R_blk=2`); fp16/bf16 stay 8192 (`R_blk=2`). The
  condition is inert for `hidden` (R=1) and `cnn-feat` (b not outer-broadcast), so it is a
  shape-specialised compile-time dispatch inside the same kernel (no wrapper/factory runtime or
  shape dispatch on the op path).

### Candidate vs current best — round 2 (B2 comparison table)

Current best = r1_op2 (`perf_opt/eq_fwd_kernel.py`).

| candidate | workload_id | Task Duration(us) | AICore mte2/mte3/vec/scalar | mem (main read BW / GM→UB KB) | L0 | vs current best |
|---|---|---:|---|---|:--:|---:|
| r1_op2 | hidden-state-prefill-float16 | 26.159 | 0.791/0.227/0.257/0.081 | 35.5 GB/s / — | pass | — |
| r2_op1 | hidden-state-prefill-float16 | 29.139 | 0.784/0.228/0.263/0.087 | 36.4 GB/s / — | pass | −11.4 % (noise) |
| r1_op2 | hidden-state-prefill-bfloat16 | 27.739 | 0.569/0.245/0.523/0.090 | 32.7 GB/s / — | pass | — |
| r2_op1 | hidden-state-prefill-bfloat16 | 27.979 | 0.560/0.247/0.525/0.093 | 32.8 GB/s / — | pass | −0.9 % |
| r1_op2 | hidden-state-prefill-float32 | 49.619 | 0.814/0.175/0.199/0.066 | 35.5 GB/s / — | pass | — |
| r2_op1 | hidden-state-prefill-float32 | 49.599 | 0.844/0.130/0.180/0.042 | 36.6 GB/s / — | pass | +0.0 % |
| r1_op2 | cnn-feat-broadcast-float16 | 167.717 | 0.873/0.258/0.170/0.036 | 3.7 GB/s / — | pass | — |
| r2_op1 | cnn-feat-broadcast-float16 | 146.437 | 0.855/0.274/0.196/0.040 | 4.2 GB/s / — | pass | +12.7 % (inert path → noise) |
| r1_op2 | cnn-feat-broadcast-bfloat16 | 146.857 | 0.656/0.233/0.392/0.047 | 4.2 GB/s / — | pass | — |
| r2_op1 | cnn-feat-broadcast-bfloat16 | 177.596 | 0.716/0.216/0.323/0.039 | 3.5 GB/s / — | pass | −20.9 % (noise) |
| r1_op2 | cnn-feat-broadcast-float32 | 89.558 | 0.756/0.323/0.269/0.067 | 14.1 GB/s / — | pass | — |
| r2_op1 | cnn-feat-broadcast-float32 | 92.498 | 0.623/0.289/0.458/0.066 | 13.7 GB/s / — | pass | −3.3 % (block 8192 for cnn) |
| r1_op2 | row-broadcast-float32 | 61.299 | 0.747/0.290/0.310/0.064 | 28.0 GB/s / — | pass | — |
| **r2_op1** | row-broadcast-float32 | **52.019** | 0.745/0.248/0.336/0.047 | 34.9 GB/s / — | pass | **+15.1 %** |
| r1_op2 | row-broadcast-float16 | 51.739 | 0.710/0.313/0.465/0.077 | 16.2 GB/s / — | pass | — |
| **r2_op1** | row-broadcast-float16 | **33.939** | 0.745/0.317/0.372/0.073 | 26.3 GB/s / — | pass | **+34.4 %** |
| r1_op2 | row-broadcast-bfloat16 | 79.578 | 0.437/0.197/0.668/0.060 | 10.3 GB/s / — | pass | — |
| **r2_op1** | row-broadcast-bfloat16 | **46.539** | 0.508/0.209/0.572/0.059 | 18.3 GB/s / — | pass | **+41.5 %** |

- **Adjudication result (shape A, `row-broadcast`):** alternative #4 beats the frozen main choice by
  15–42 % (52.0 vs 61.3 µs fp32, 33.9 vs 51.7 µs fp16, 46.5 vs 79.6 µs bf16) — far past the
  `< 0.97` flip threshold → **#4 wins for the outer-broadcast row class**. A 3-row tile
  (block 12288) was also probed and is worse (row fp16 55.7 µs, bf16 91.5 µs) — `R_blk=2` is the
  optimum (UB/pipeline-depth limit at `R_blk=3`).
- **Shape B (`cnn-feat`):** #4 does not apply — b varies per row (per-channel) so the periodic-b
  re-interpretation is invalid; the cnn differences above are the noise band of the inert path
  (plus the fp32 block 4096→8192 side-effect). The block-size sweep stands as its adjudication:
  per-row tile 4096 (71.2 µs fp32, 39.7 %) beats 8192/9216/3136.
- **Merge decision:** #4 is validated but **not merged** into the deliverable. `row-broadcast`
  fp32 already meets the frozen ≥60 % target with r1_op2 (60.98 %), the conductor task plan
  specifies the block_size + hoist route, and the `phase=final` records for the deliverable were
  already collected (see process note below). `r2_op1` is retained as the validated alternative and
  `perf_opt/eq_fwd_kernel_round2_op1.py` is the drop-in candidate if the deliverable is re-opened.

### Process note (frozen-final sequencing)

The `phase=final` records were collected on `eq_fwd_kernel.py` (= r1_op2) **before** this round-2
adjudication. `perf_records.jsonl` is append-only, so a subsequent re-designation of the final
candidate is not possible without rewriting history; the deliverable is therefore frozen at r1_op2.
Round-2 candidate records are appended (not `final`). This is recorded as a skill-retrospective
process issue: run the DESIGN-mandated adjudication **before** the Phase-3 final collection.

## Autotune Log

Not used. The only tunable execution parameter is `block_size`; the per-dtype choice was fixed by
direct `msprof op` sweeps (see `logs/scan/*`, `profiles/scan/*`), not a search space:
fp16/bf16 ∈ {4096 → 30.9, 8192 → 27.3, **12288 → 25.2 (best for hidden)**, 14336 → 25.4};
fp32 ∈ {2048 → 59.6, 4096 → 50.1, 8192 → 48.9, **9216 → 47.8**}; cnn fp32 ∈ {3136 → 73.2,
**4096 → 71.2**, 8192 → 79.8, 9216 → 92.2}; cnn fp16 ∈ {3136 → 157, 4096 → 146, 8192 → 130}.
The deliverable keeps the shared per-dtype value that serves both the flat and row classes
(fp16/bf16 8192, fp32 4096), because the wrapper binds one `block_size` per dtype.

## Final Performance Test Data (round 3, superseded 2026-10-22)

Final deliverable `perf_opt/eq_fwd_kernel.py` (sha256
`5cdfae467e4906d1106569fc343298d85ca4789feb1377d138f4d98ad019d52a`), candidate_id `final`
(= r1_op2 content). `final_us` is the single `phase=final` record per tune workload, measured on
the final file with `msprof op` `Task Duration(us)`. `baseline_us` is the round-0 `phase=baseline`
record at the shipped bs=1024.

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-float16 | 83.658 | 26.460 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.198 | 29.000 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-float32 | 94.438 | 49.920 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-float16 | 699.806 | 43.760 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 578.868 | 41.040 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-float32 | 406.852 | 49.020 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-float32 | 185.296 | 52.620 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-float16 | 183.876 | 32.840 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-bfloat16 | 166.257 | 44.600 | final |

Smoke (`smoke-1k` fp32/fp16/bf16): precision-only, `precision_pass = true` (L0/L1 on the final
file); no performance records (per contract). No skipped workloads.

## Final Summary

Deliverable = `perf_opt/eq_fwd_kernel.py` (candidate `final` = r1_op2: per-dtype `block_size`
constants **+** outer-broadcast-operand hoist). Ratios via `roofline_ratio.py`.

| kernel_id + workload_id | baseline (µs) | final (µs) | improvement | baseline Ratio | **final Ratio** | ≥60 %? |
|---|---:|---:|---:|---:|---:|:--:|
| …::main / hidden-state-prefill-float16 | 83.66 | 26.00 | +68.9 % | 22.28 % | **71.70 %** | ✓ |
| …::main / hidden-state-prefill-bfloat16 | 91.20 | 27.74 | +69.6 % | 20.44 % | **67.20 %** | ✓ |
| …::main / hidden-state-prefill-float32 | 94.44 | 48.74 | +48.4 % | 39.48 % | **76.49 %** | ✓ |
| …::main / row-broadcast-float32 | 185.30 | 61.16 | +67.0 % | 20.13 % | **60.98 %** | ✓ (marginal) |
| …::main / row-broadcast-float16 | 183.88 | 51.50 | +72.0 % | 10.14 % | **36.21 %** | ✗ (read-limited; #4 → 54.9 % unmerged) |
| …::main / row-broadcast-bfloat16 | 166.26 | 79.48 | +52.2 % | 11.22 % | **23.46 %** | ✗ (Vector-bound bf16; #4 → 40.0 % unmerged) |
| …::main / cnn-feat-broadcast-float32 | 406.85 | 83.90 | +79.4 % | 6.73 % | **33.68 %** | ✗ (small-burst / scalar-b granularity) |
| …::main / cnn-feat-broadcast-float16 | 699.81 | 157.20 | +77.5 % | 1.77 % | **8.73 %** | ✗ (read-limited + small bursts) |
| …::main / cnn-feat-broadcast-bfloat16 | 578.87 | 153.32 | +73.5 % | 2.13 % | **8.95 %** | ✗ (read-limited + small bursts) |

**≥60 % target: 4 / 9 workloads** (all three `hidden-state-prefill` dtypes + `row-broadcast-float32`).
As required by the task's simplification mandate, the remaining cases were given at most one round
of effort and are reported honestly rather than maximised.

### Why ≥60 % is not reached for the remaining cases

The report formula's numerator is **GM read traffic only** while every eq case also writes the
1-byte bool carrier, so the ratio ceiling is `read/(read+write) · B_eff/1.8`:

- `cnn-feat` fp16/bf16: read ≈ 25.7 MB vs write 12.8 MB → ceiling ≈ `0.667·B_eff/1.8` ≈ 70 % at
  `B_eff≈1.9`. Reaching it needs ~7× more effective BW than measured; the class is limited by the
  3136-element row granularity (one row per task, per-row scalar-b `vbrc`) → tiny MTE2 bursts
  (4–5 GB/s). This is a kernel-granularity limitation, not a DESIGN algorithm flaw; a future
  2-D/`R_blk` tiling with per-row-group b materialisation would be needed (out of the ≤1-round
  budget set for this class).
- `cnn-feat` fp32 (33.7 %): same granularity limit; its larger read fraction (4/5) gives a higher
  but still unmet ceiling.
- `row-broadcast` fp16/bf16: read ≈ 33.6 MB vs write 16.8 MB → ceiling ≈ `0.667·B_eff/1.8` ≈ 70 %.
  Even the r1_op2 hoist leaves it at 36 % / 23 % because the element count (16.8 M) is high while
  the read bytes are halved; bf16 is additionally **Vector-bound** (`vec_ratio = 0.67` — the
  `vcmp` + 2×`vcast` chain). Alternative #4 (2-D tile) lifts them to 54.9 % / 40.0 % but still not
  ≥60 % (see round 2).

Per `_shared/standards/perf-feedback.md` §1, this is **not** filed as a `[DESIGN_LIMIT]`: the
shortfall is a report-metric + kernel-granularity limit, not a contradiction of a specific
DESIGN.md assumption (the design correctly predicted emit-domination for cnn/row), and no
structural `>2×` acceleration of the class is demonstrated as *unreachable* — alternative #4 shows
the row class is still improvable. No `perf_feedback.md` is produced.

## Skill Retrospective

- **Process issue (R, strong).** The DESIGN-mandated §1.6.3 alternative-#4 adjudication was run
  *after* the Phase-3 `phase=final` records had already been collected. Because
  `perf_records.jsonl` is append-only, the validated winner (#4, row-broadcast −15…−42 %) could not
  be promoted to the deliverable. **Rule to distil:** execute all mandatory adjudication/A-B
  experiments (skill Phase 0 step 4 / Phase 2) *before* Phase 3 final collection; only freeze and
  record `phase=final` once no further structural candidate is pending. A `final` designation
  should be reversible (or Phase 3 should assert "no un-adjudicated branch remains").
  - evidence: `perf_records.jsonl` (r2_op1 candidate rows vs `phase=final` rows on r1_op2 content);
    round-2 B2 table above.
  - repro: n/a (process rule).
  - toolchain_stamp: tilelang npuir, Ascend910 40-vector-core, CANN as measured 2026-10-14.
- **Process issue (R).** The round-1 runner re-measured the *baseline file* under
  `--phase candidate` with `bench.py`'s fallback `block_size=4096` (not the shipped 1024), producing
  9 `candidate/baseline` rows that do not correspond to the baseline config. They are harmless to
  gate 4 but must be read carefully. Rule: candidate-phase runs of the baseline file must pin the
  baseline `block_size`, or the baseline file should be excluded from the candidate list.
- **Metric caveat (D, carried from the sister operator).** The single-operator Roofline ratio is
  **read-only**, so eq's ≥60 % target is only meaningful for read-heavy classes; the 1-byte bool
  output makes eq *more* favourable than the int32 bitwise family (read fraction 0.8 vs 0.5).
- **Validated optimization (P, positive).** Outer-broadcast-operand hoist (load the fully
  outer-broadcast operand once per core instead of once per row) → row-broadcast 61–72 % faster
  (185→61 µs fp32, 184→52 µs fp16, 166→79 µs bf16). Reproduced here from the bitwise family.
  - evidence: round-1 B2 table; `profiles/candidate/round1/r1_op2_*`.
  - repro: `perf_opt/eq_fwd_kernel_round1_op2.py` (diff vs `eq_fwd_kernel_round1_op1.py`).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive).** Outer-broadcast-row 2-D tile (re-interpret as a flat
  `R_blk·L`-element stream with a period-L b operand; `R_blk = block_size/L`) → row-broadcast
  −15…−42 % on top of the hoist. Counter-intuitive `R_blk=2` optimum (`R_blk=3` regresses:
  row fp16 33.9→55.7 µs) — UB/pipeline-depth limit.
  - evidence: round-2 B2 table; `profiles/candidate/round2/r2_op1_*`.
  - repro: `perf_opt/eq_fwd_kernel_round2_op1.py` (`perc2d` block in the factory).
  - toolchain_stamp: same as above.
- **Block-size constants (D).** UB-budget-feasible optima on this 40-Vector-core Ascend910:
  hidden fp16/bf16 12288 (25.2 µs) > 8192 (27.3) ; hidden fp32 9216 (47.8) ; row needs a
  2-row tile (`R_blk=2`); cnn per-row 4096. A single per-dtype value is a compromise because the
  wrapper binds one `block_size` per dtype.
- **BP proposal (R):** `BP_stage4_final_freeze` — freeze `phase=final` only after all
  DESIGN-mandated adjudications are resolved; otherwise append-only records orphan a validated
  winner. (vp_type=R)
- No existing `BP_xxx` was invalidated.

### Measurement environment

Shared host; the residual run-to-run spread on the (otherwise idle) NPU is ≈7–13 % (e.g. cnn-feat
fp16 146–168 µs and hidden fp16 26.0–29.1 µs across identical-config re-runs). Single-point
comparisons inside that band (r1_op1 vs r1_op2 on cnn) are not treated as real differences; the
hoist and #4 effects (15–72 %) are well above it. `row-broadcast-float32` at 60.98 % is within ~1
point of the ≥60 % line; alternative #4 gives it a robust 71.7 % margin but was left unmerged (see
process note).

---

# Round 4–7 (2026-10-22) — 2-D `[R_blk, L]` tile ladder and the col-broadcast stall

Session resources: exclusive chip0 (`ASCEND_RT_VISIBLE_DEVICES=0`), private
`TILEOPS_MSPROF_OUTPUT_DIR=/tmp/opencode/msprof-out-eqge`. Every number below is
`msprof op` `Task Duration(us)` (median of the per-launch CSVs, `--launch-count=15`),
A/B taken in **alternating order with even pairs** (`perf_opt/run_ab.py`, 2 reps per
branch). Raw data: `perf_opt/profiles/candidate/r{4,5,6,7}/`, `perf_opt/profiles/final/r8/`;
stdout: `perf_opt/logs/r4,r5,final/`. Records appended to `perf_opt/perf_records.jsonl`
(rounds 4–7 = `candidate`, round 8 = `final`).

## Round 4 — measured UB budget replaces the ×1.7 inflation

**现象 (base = round-3 final).** The 2-D tile budget was
`UB // ((2·cmp_bytes + 4) · 1.7)`; `r_blk` must divide `run` (a power of two for both
broadcast classes), so the inflation cost a whole tile step: cnn-feat fp32 → `r_blk=2`
and row-broadcast fp16/bf16 → `r_blk=2`, i.e. the same 2× tile step the gt/le siblings
had already recovered with a 1.5 inflation factor.

**候选优化点 (single lever).** Replace the guessed inflation by the UB demand that
BishengIR itself reports for this exact body. `perf_opt/probe_ub_demand.py` compiles the
body at forced `r_blk` and reads back
`ub overflow, requires N bits while 1572864 bits available`:

| dtype | fit | `demand(r_blk)` | fixed | B/elem |
|---|---|---:|---:|---:|
| fp16/bf16, L=3136 | r_blk 4 ok / 8 ✗ (1 606 656 bit = 200 832 B) | 25 088·r_blk + 128 | 128 B | 8 (= 2·2+4) |
| fp32, L=3136 | 4 ok / 6 ✗ (1 808 640 bit = 226 080 B) | 37 632·r_blk + 288 | 288 B | 12 (= 2·4+4) |

So the five logical tiles are already liveness-compacted and the ×1.7 reserve was pure
under-utilisation. The budget became
`max_2d_elems = (196 608 − reserve) // (2·cmp_bytes + 4)` with
`reserve = 512 B (+ 2·L·elem_bytes on the row path for its `(1,L)` staging)`.
Derived `r_blk` (identical for both orientiation classes):

| case | r_blk before | r_blk after |
|---|:--:|:--:|
| cnn-feat fp16 / bf16 | 4 | 4 |
| cnn-feat fp32 | **2** | **4** |
| row fp16 / bf16 | **2** | **4** |
| row fp32 | 2 | 2 (3 would fit but does not divide 4096) |

**候选 vs current best（`eq_fwd_kernel_round4_op1.py`，round 4）**

| workload | current best (us) | r4op1 (us) | Δ | mte2 base→cand | vec base→cand | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|
| cnn-feat-broadcast-float16 | 37.62 | 37.69 | −0.2 % (tie) | 0.750→0.748 | 0.315→0.314 | PASS |
| cnn-feat-broadcast-bfloat16 | 38.58 | 38.50 | +0.2 % (tie) | 0.481→0.416 | 0.527→0.593 | PASS |
| cnn-feat-broadcast-float32 | 43.75 | **39.38** | **+10.0 %** | 0.611→0.613 | 0.439→0.456 | PASS |
| row-broadcast-float32 | 51.26 | 51.76 | −1.0 % (tie) | 0.736→0.734 | 0.347→0.346 | PASS |
| row-broadcast-float16 | 32.04 | **27.48** | **+14.2 %** | 0.737→0.705 | 0.388→0.486 | PASS |
| row-broadcast-bfloat16 | 44.10 | **34.15** | **+22.6 %** | 0.495→0.393 | 0.587→0.723 | PASS |

`memory` axis: `gm_to_ub_datas_kB` per core is unchanged within 0.4 % on every case
(640.25 → 640.25 cnn-fp16, 824 → 840 row-fp16 = 1 extra broadcast row per core), i.e. the
gain is pure MTE2 efficiency (`aiv_mte2_active_bw` 20.5 → 24.1 GB/s on cnn-fp16,
36.7 → 50.2 GB/s on row-fp16), not a traffic reduction. `L0` = `--level L0` on the branch
file (12 items) in both runs.

The two "tie" rows are expected: for those cases `r_blk` is unchanged, so the emitted
kernel is identical and the residual spread is the ±3–5 % session noise (bimodal:
cnn-bf16 shows 40.66/36.50 on the *same* binary).

## Round 5 — resident b column in UB (rejected: dynamic UB sub-view scalarises)

**现象.** A geometry-matched diagnostic added to `bench.py` isolates the broadcast
orientation at 12.845 M elements and L=3136 (`diag-row-3136-*`, timing only, not a
manifest workload):

| variant | Task Duration | mte2 | vec | mte2 active BW |
|---|--:|--:|--:|--:|
| row-broadcast (diag) | 20.86 / 20.91 | 0.696 | 0.431 | 47.0 GB/s |
| cnn-feat fp16 (col) | 34.36 | 0.739 | 0.325 | 25.2 GB/s |

Same bytes, same tile size, same `r_blk`: the col path is 1.65× slower and MTE2 moves half
the bytes per cycle. Two timing-only probes (wrong output on purpose, `--check` removed,
never recorded) price the two candidates:

| probe (cnn-feat fp16, 3 reps) | Task Duration | mte2 | vec | scalar |
|---|--:|--:|--:|--:|
| baseline r4op1 | 40.38 / 43.80 / 40.44 | 0.78 | 0.27 | 0.046 |
| P1: per-task GM scalar load removed | **22.38 / 22.18 / 22.34** | 0.70 | 0.57 | 0.085 |
| P2: per-task `T.vbrc` removed | 22.10 / 22.08 / 22.24 | 0.72 | 0.41 | 0.078 |

⇒ the 18 µs is the single-buffered `copy(8 B GM) → vbrc → vcmp` chain of the col path
(≈0.7 µs per tile), not the tile geometry.

**候选优化点.** Keep the whole b column (`run` scalars, 512 B) resident in UB, loaded once
per core, and broadcast from a sub-view.

**实验分支.** `r5op1` = `T.vbrc(b_col[sub*r_blk : sub*r_blk+r_blk, 0:1], b2)`; `r5op2` =
`T.copy(...)` of the sub-view into the static `(r_blk,1)` staging then `T.vbrc`.

**结果：both rejected.** L0 passes, but the duration explodes:

| branch | cnn-fp16 | cnn-bf16 | cnn-fp32 | vec | scalar | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|
| r5op1 | 3625.5 | 3635.2 | 3631.0 | 0.003 | 0.55–0.99 | PASS |
| r5op2 | 3619.6 | 3629.3 | 3622.5 | 0.27 | 0.55 | PASS |

A **dynamic-offset** UB sub-view feeding `T.vbrc` scalarises the whole `r_blk×L`
broadcast (12.8 M scalar ops → 3.6 ms). Recorded as a compiler trap; `T.vbrc` keeps its
vector form only with a static-shape operand.

## Round 6 — broadcast-group-major task order (predicated `vbrc` rejected on precision)

**候选优化点.** Leave the a-tile contiguous, but enumerate tiles *broadcast-group major*
and hand each core a contiguous block: `task = cid·A + i·B`, `sub = task // R_lead`,
`lead = task % R_lead`. A core then walks `R_lead` consecutive tiles that all share the
SAME b row block, so the 8-byte GM scalar load runs once per group instead of once per
tile.

**实验分支 `r6op1`** (both the load *and* the `T.vbrc` inside `if i == 0 or lead == 0`):
**rejected — precision.** `L0-3/float16/cnn-feat` → `max_diff=1`, 2174 violating
elements; `L0-3/float32/cnn-feat` → `ub overflow, requires 1 606 400 bits`
(b2 double-buffered: +50 KB). A predicated write to a buffer read across loop iterations
is not safe under the auto-multi-buffer transform.

**实验分支 `r6op2`** (only the GM scalar load predicated; `T.vbrc` unconditional, so b2
keeps one static producer): L0 PASS. `r_blk` footprint returns to the round-4 value.

| workload | current best (us) | r6op2 (us) | Δ | mte2 | vec | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|
| cnn-feat-broadcast-float16 | 41.84 | **23.36** | **+44.2 %** | 0.777→0.600 | 0.278→0.501 | PASS |
| cnn-feat-broadcast-bfloat16 | 42.62 | **32.01** | **+24.9 %** | 0.480→0.388 | 0.527→0.680 | PASS |
| cnn-feat-broadcast-float32 | 39.85 | 38.65 | +3.0 % | 0.610→0.620 | 0.456→0.444 | PASS |

`gm_to_ub_datas_kB` 640.25 → 637.31 (−0.5 %, i.e. the b re-reads that disappeared) —
the whole 18 µs band is MTE2 stall removal.

## Round 7 — merge (`r_blk` ladder + group-major col order) and precision fix

`r7op1` = round-4 budget + round-6 col order, with the row/1-D paths **left on the
original cyclic mapping** (`task_a = 1, task_b = num_kernels`) so their code is
unchanged. Row re-check (must not regress):

| workload | current best (us) | r7op1 (us) | Δ | L0 |
|---|--:|--:|--:|:--:|
| row-broadcast-float16 | 27.34 | 28.37 | −3.8 % (noise band) | PASS |
| row-broadcast-bfloat16 | 34.09 | 34.16 | −0.2 % (tie) | PASS |
| row-broadcast-float32 | 52.96 | 52.59 | +0.7 % (tie) | PASS |

`--level all` on the merged file then found one真 regression introduced by the
predicated load: `L1/bool-vbrc/(16,256,56,56)` (int8-carrier col broadcast). Root cause:
the `i1→f16` cast was inside the predicate as well, so its dst (`b_c`) became a
conditionally-written buffer. Fix (in the final module): predicate **only** the GM
scalar load, keep `T.vcast` + `T.vbrc` unconditional → `--level all` PASS.
`--level all` on `perf_opt/EqFwdOp.py`: L0 + L1 + boundary all pass, `max_diff=0` for
every dtype.

## Final Performance Test Data

Final deliverable `perf_opt/EqFwdOp.py` (sha256
`c189e7a10f733a14…`, see `perf_records.jsonl`), `candidate_id = final`, content =
`eq_fwd_kernel_round7_op1.py` + the L1 bool fix. One `phase=final` record per tune
workload, measured on that file in a single session (round 8); `baseline_us` is the
round-0 `phase=baseline` record. Ratio = report口径 Roofline
(`roofline_ratio.py`: Σ L2 read-miss × 128 B ÷ (duration × 1.8 TB/s)), computed from the
same raw profile as the duration.

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-float16 | 83.658 | 27.860 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.198 | 27.020 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | hidden-state-prefill-float32 | 94.438 | 48.680 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-float16 | 699.806 | 23.040 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 578.868 | 31.940 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | cnn-feat-broadcast-float32 | 406.852 | 38.280 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-float32 | 185.296 | 52.280 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-float16 | 183.876 | 26.440 | final |
| examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py::eq_fwd_kernel::main | row-broadcast-bfloat16 | 166.257 | 33.980 | final |

Roofline Ratio before → after (before = round-3 record scaled by the identical read
traffic; the round-3 record is now `phase=merged`):

| case | duration before → after (us) | Ratio before → after | ≥60 % |
|---|---|--:|:--:|
| hidden fp32 | 49.92 → 48.68 | 74.69 → **76.59** | ✅ |
| hidden fp16 | 26.46 → 27.86 | 70.45 → **66.91** | ✅ |
| hidden bf16 | 29.00 → 27.02 | 64.28 → **68.99** | ✅ |
| row fp32 | 52.62 → 52.28 | 70.88 → **71.33** | ✅ |
| row fp16 | 32.84 → 26.44 | 56.78 → **70.52** | ✅ **新达标** |
| cnn fp32 | 49.02 → 38.28 | 58.23 → **74.57** | ✅ **新达标** |
| cnn fp16 | 43.76 → 23.04 | 32.62 → **61.95** | ✅ **新达标** |
| row bf16 | 44.60 → 33.98 | 41.81 → 54.87 | ❌ |
| cnn bf16 | 41.04 → 31.94 | 34.78 → 44.69 | ❌ |

**7/9 达标 (was 4/9).** The three newly passing cases are exactly the three levers
above: row fp16 (tile 2→4), cnn fp32 (tile 2→4), cnn fp16 (col stall removed).
`hidden fp16` moved −5 % (26.46 → 27.86) but stays at 66.9 %: the 1-D path is untouched,
it is session noise on the same binary (the round-3 log records 26.0–29.1 µs spread for
this exact case).

### Why the two bfloat16 cases still miss 60 %

- **row bf16 (54.87 %).** `vec=0.725, mte2=0.390`: purely Vector-bound. Every element
  costs `vcmp(bf16)` + two casts (`i1→f16`, then `f16→i8, rint`); the documented
  `T.vcast` pair table (2.2.1) has no `i1→i8`, and the gt/le sibling measured that the
  direct form does compile bit-exactly but with zero benefit (BiShengIR still
  materialises a 2 B/elem intermediate), so it is not a lever. bf16 `vcmp` costs ~2× the
  fp16 one. `r_blk` cannot grow: at L=4096, `r_blk=8` needs 8·4096·8 B = 256 KB > 192 KB,
  and even a hypothetical 6 B/elem aliasing would land at 196 736 B > 196 608 B.
  The gt sibling lands at 54.65 % on the same case with the same structure — this is the
  family ceiling for the read-only-Roofline口径 (ceiling = 2/3·B_eff/1.8 ≈ 62 %).
- **cnn bf16 (44.69 %).** Same Vector bound (`vec=0.683, mte2=0.386`), plus L=3136 caps
  `r_blk` at 4 (8 would need 200 832 B, measured). Comparing in fp16 would be faster but
  is not exact for the bf16 exponent range → rejected as semantically unsafe.

Both are hardware/口径 ceilings, not tiling parameters — **no `perf_feedback.md` is
produced**: neither [DESIGN_LIMIT] trigger is met (the residual gap is not attributable to
a DESIGN.md assumption, and there is no >2× structural candidate left; the gt/le siblings
confirm the same ceiling from outside this task).

## Skill Retrospective (2026-10-22 r_blk round)

**流程问题**

1. **The ×1.7 auto-multi-buffer inflation was carried as folklore for three rounds** while
   `perf_opt/probe_ub_demand.py` could have reverse-engineered the exact demand in one
   compile. It cost a whole tile step on 4 of 6 broadcast cases (row fp16/bf16 + cnn fp32
   for both ops). Rule: **before choosing a tile size, read the compiler's own overflow
   number** — the pattern-library already says this (`CONST-2dtile-ub-demand-k`), but the
   sibling entries kept quoting the inflated formula.
2. **The col-broadcast class was never profiled separately from the row class.** The
   `[R_blk,L]`-tile entry treats both orientations as one pattern; the 1.65× gap at
   identical geometry (20.9 vs 34.4 µs) was invisible because only absolute durations were
   compared across shapes. A geometry-matched *diagnostic* case (same element count, same
   L, only the broadcast axis changed) was what exposed it.
3. **Bimodal runs (+10 %/−10 % on the same binary) still appear** when the neighbouring
   chip is busy (the gt/le agent shares HBM). Alternating A/B pairs handled it for
   decisions, but single-shot `final` records inherit the wobble (row fp16 25.0–28.4 µs
   across sessions). `ab_test.py`-style protocols should also be used for the final table
   when a case sits within 5 % of the acceptance line (cnn fp16 lands at 61.6–62.0 %).

**value points**

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| CONST-2dtile-ub-demand-k (delta) | D | For the *cmp* body (`a2+b2+cond+yf+y8`) the UB demand is exactly `(2·cmp_bytes+4)·r_blk·L + 128…288 B` — the five tiles are already liveness-compacted, so **no ×1.7 inflation applies** (contrast: the vxor body needs k=6·E). Max r_blk at L=3136: fp16 7, fp32 5 (divisible → 4); at L=4096: fp16 5, fp32 3 | probe_ub_demand.py, 6 compile points | `perf_opt/probe_ub_demand.py` | tilelang 0.1.2+eb362ab / CANN 9.0.0 / Ascend910 40-core / 2026-10-22 |
| PL-1.21 delta (col) | P | **col-broadcast (per-row scalar) costs 1.65× vs row-broadcast at identical tile geometry**: the single-buffered `8 B GM→UB copy → vbrc → vcmp` chain stalls ≈0.7 µs per tile (mte2 busy 0.78 but at half the active BW). Fix = **broadcast-group-major task order + predicate only the GM scalar load** (1 load per `R_lead` tiles): cnn fp16 41.8→23.4 µs, bf16 42.6→32.0, fp32 39.9→38.7 | eq perf_records round 6/7; diag-row-3136 | `perf_opt/probe_ub_demand.py` + `bench.py::diag-row-3136-*` | same |
| TRAP (new) | D | `T.vbrc` scalarises completely when its src is a **dynamic-offset UB sub-view** (34 → 3625 µs, `scalar_ratio 0.55–0.99`); a `T.copy` of the same sub-view into a static staging buffer does not help. Also: **a predicated `T.vbrc` (inside a runtime `if`) is both incorrect across loop iterations** (max_diff=1 on 2174 elements) **and UB-doubling** (+50 KB → overflow). Only the GM load may be predicated | eq perf_records round 5/6; L0-3 log | `perf_opt/probe_ub_demand.py --hoist direct|ubcopy` | same |
| R | 通用规则 | 在为任何 `[R_blk,L]` tile 选 `r_blk` 之前先做一次 UB-demand 反解；广播**方向**（row vs col）必须单独做同几何对照，不能只比跨 shape 的绝对时延 | 本轮回溯 | — | — |
