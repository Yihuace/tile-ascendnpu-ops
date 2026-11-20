# Stage 4 Optimization Log — `ge_fwd_kernel` (GeFwdOp)

- project: `ge`
- operator: `ge_fwd_kernel`
  (`examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/`)
- benchmark source: `examples/TileOPs/benchmarks/ops/bench_ge.py::test_ge_manifest_bench`
- baseline (read-only, never modified): `../ge_fwd_kernel.py`
- deliverable: `perf_opt/ge_fwd_kernel.py`
  (sha256 `747ac65c20426426d9ed1787b9d9adc64e7c66880c4d9276f893a7591bf500e5`)
- primary metric: `msprof op` `Task Duration(us)` (kernel-only, median of the
  `--launch-count=15` launches in one `msprof op` run). No other timing口径 is used.
- Roofline Ratio = the repo single-operator report formula, reproduced by
  `perf_opt/roofline_ratio.py` (mirrors
  `tileops/benchmark/msprof_roofline.compute_from_csvs`, GM-bounded branch):
  `ratio = (Σ aiv/aic r0+r1 read_cache_miss_allocate × 128) / (Task Duration(s) × 1.8e12)`,
  BW = 1.8 TB/s. The numerator is GM **read** traffic only (writes are not in the report
  formula), which bounds `ratio ≤ read/(read+write) · B_eff/1.8`.
- AICore: Ascend910, 40 physical Vector cores (`NPUUtils.get().get_aicore_num()*2`).
- Task target: every case Roofline Ratio ≥ 60 % ("达标即停"); hard classes get a bounded,
  honestly-reported effort.
- Sister reference reused: `eq_fwd_kernel` (`../eq/eq_kernel/perf_opt/opt_log.md`) and
  `floor_divide_fwd_kernel` — same elementwise/broadcast family. Their per-dtype `block_size`
  and outer-broadcast-operand hoist were carried over; the 2-D `[R_blk, L]` tile that eq could
  not promote (append-only freeze ordering) is merged here.

> Pre-existing partial artifacts (round-1/2 branch files, `perf_records.jsonl` baseline/candidate
> rows, the final `phase=final` rows, `profiles/`, `run_experiments.py`, `probe_2d.py`) were read
> first. The run was resumed from `r2_op2` (= the deliverable) and finalised in this session: the
> `tuning_status` fields, this log, and a round-3 negative-control experiment were added.

## Workload inventory (from the migrated benchmark)

`perf_opt/workload_inventory.json` — 9 tune cases (3 shape classes × 3 dtypes) + 3 smoke cases
(`smoke-1k` fp32/fp16/bf16, precision-only). No skipped cases. The fp16/bf16 smoke ids carry a
benchmark `full` mark that conflicts with the independent `smoke` token; they are excluded from
performance tuning (recorded with reason) and used for precision regression only. All 9 tune
workloads resolved as `winner_merged`.

Coalesced shapes (compile-time, from `coalesce_broadcast_dims`):
- `hidden-state-prefill` → `(8388608,)`, R=1, L=8.39 M → **1-D path** (per-dtype `block_size`).
- `cnn-feat-broadcast` → `(16,256,3136)`, R=4096, L=3136, b inner stride 0, b outer stride 1
  → **2-D path**, `r_blk=4` (fp16/bf16) / `r_blk=2` (fp32).
- `row-broadcast` → `(4096,4096)`, R=4096, L=4096, b fully outer-broadcast (strides all 0)
  → **2-D path**, `r_blk=2`, b row materialised once per core.

## Baseline — Performance Test Data (round 0, `phase=baseline`)

Shipped default config (`block_size=4096`; the op-layer default is `threads*npt=1024` for float,
but the Stage-4 driver measures at the recommended `4096` unless a tuned constant is present).

| kernel_id | workload_id | dtype | shape (a \| b) | Task Duration(us) | AICore mte2/mte3/vec/scalar | GM read BW | Roofline Ratio | raw profile |
|---|---|---|---|---:|---|---:|---:|---|
| …::main | hidden-state-prefill-float16 | fp16 | (2048,4096) \| = | 82.578 | 0.804/0.337/0.167/0.126 | 9.9 GB/s | 22.57 % | `profiles/baseline/round0/baseline_8ba8496a/hidden-state-prefill-float16_f48f537e` |
| …::main | hidden-state-prefill-bfloat16 | bf16 | (2048,4096) \| = | 91.338 | 0.728/0.302/0.261/0.141 | 8.9 GB/s | 20.41 % | `…/hidden-state-prefill-bfloat16_274d79ca` |
| …::main | hidden-state-prefill-float32 | fp32 | (2048,4096) \| = | 95.118 | 0.808/0.302/0.167/0.108 | 17.0 GB/s | 39.20 % | `…/hidden-state-prefill-float32_dac1dbdc` |
| …::main | cnn-feat-broadcast-float16 | fp16 | (16,256,56,56) \| (256,1,1) | 528.969 | 0.924/0.198/0.061/0.045 | 1.6 GB/s | 2.34 % | `…/cnn-feat-broadcast-float16_f1cddcf9` |
| …::main | cnn-feat-broadcast-bfloat16 | bf16 | (16,256,56,56) \| (256,1,1) | 661.607 | 0.909/0.167/0.078/0.044 | 1.3 GB/s | 1.87 % | `…/cnn-feat-broadcast-bfloat16_f5a51caa` |
| …::main | cnn-feat-broadcast-float32 | fp32 | (16,256,56,56) \| (256,1,1) | 352.613 | 0.863/0.266/0.113/0.077 | 4.7 GB/s | 7.76 % | `…/cnn-feat-broadcast-float32_2ff59659` |
| …::main | row-broadcast-float32 | fp32 | (4096,4096) \| (1,4096) | 177.676 | 0.826/0.323/0.177/0.081 | 18.0 GB/s | 20.99 % | `…/row-broadcast-float32_7e902357` |
| …::main | row-broadcast-float16 | fp16 | (4096,4096) \| (1,4096) | 184.136 | 0.856/0.305/0.146/0.078 | 8.6 GB/s | 10.13 % | `…/row-broadcast-float16_410291d8` |
| …::main | row-broadcast-bfloat16 | bf16 | (4096,4096) \| (1,4096) | 177.196 | 0.729/0.325/0.281/0.114 | 9.6 GB/s | 10.52 % | `…/row-broadcast-bfloat16_00d04600` |

Smoke / skip: `smoke-1k-*` (fp32/fp16/bf16) are precision-only (no perf profile), verified by the
kernel self-test (`--level L0` in round verification, `--level all` in Phase 3). No skipped cases.

### Baseline bottleneck ranking (by measured burden)

1. `cnn-feat-broadcast-*` (353–662 µs; 1.9–7.8 %) — per-row 1-D granularity: one 3136-elem row per
   task with a per-row scalar-`b` `vbrc`; tiny MTE2 bursts (1.3–4.7 GB/s ≪ 1.8 TB/s) ⇒ emission
   (per-task op count) dominated.
2. `row-broadcast-*` (177–184 µs; 10–21 %) — same per-row granularity; the fully-broadcast `b`
   row is re-materialised every task.
3. `hidden-state-prefill-*` (83–95 µs; 20–39 %) — flat same-shape stream, largest read volume;
   MTE2 window is already near-saturated (mte2 0.73–0.81) but `block_size=4096` is small.

### Design estimate vs measured (D-2)

DESIGN §1.6.0 estimated the baseline lower bound (BW=1.05 TB/s, 40-core scaled) as
`hidden fp32 ≈ 78 µs`, `cnn-feat fp32 ≈ 206 µs`, `row-broadcast fp32 ≈ 155 µs` (b-reread upper
bound ≈ 144 µs). Measured baseline: 95.1 / 352.6 / 177.7 µs → ratio to estimate 1.22× / 1.71× /
1.15×, all **< 2×** (no失准 flag). Note the design's emission model **over**-predicted cnn/row by
≈ 1.7× and was beaten by ≈ 4× by the 2-D alternative (§1.6.3 #4) actually selected in round 2 —
the per-task op cost is lower than the design's 0.5 µs/op constant assumed.

## Iteration Log

### Round 1 — per-dtype `block_size` and outer-broadcast hoist

Branches derived from `baseline`: `r1_op1` (per-dtype `block_size`), `r1_op2` (= `r1_op1` +
outer-broadcast-operand hoist).

**Candidate vs current best (`baseline`) — Task Duration(us)**

| workload_id | baseline (current best) | r1_op1 | r1_op2 | winner |
|---|---:|---:|---:|---|
| hidden-state-prefill-float16 | 82.578 | 26.90 | 26.70 | r1_op2 |
| hidden-state-prefill-bfloat16 | 91.338 | 29.32 | 27.56 | r1_op2 |
| hidden-state-prefill-float32 | 95.118 | 49.78 | 51.10 | r1_op1 |
| cnn-feat-broadcast-float16 | 528.969 | 146.52 | 167.82 | r1_op1 |
| cnn-feat-broadcast-bfloat16 | 661.607 | 155.06 | 135.18 | r1_op2 |
| cnn-feat-broadcast-float32 | 352.613 | 90.56 | 89.56 | r1_op2 |
| row-broadcast-float32 | 177.676 | 183.26 | 60.72 | r1_op2 |
| row-broadcast-float16 | 184.136 | 182.48 | 51.52 | r1_op2 |
| row-broadcast-bfloat16 | 177.196 | 183.64 | 79.40 | r1_op2 |

`r1_op2` won 7/9 (all except hidden fp32 by <3 % and cnn fp16) and was merged as the round-1
current best. Optimization points:

- **Per-dtype `block_size` (`r1_op1`)**: `float16/bf16 → 8192`, `float32 → 4096`, exposed as
  `TUNED_BLOCK_BY_DTYPE` / `TUNED_BLOCK_SIZE` module constants (the wrapper source-selection block
  can bind the same values). Largest UB-fitting tile for the flat same-shape stream (MTE2-bound).
- **Outer-broadcast-operand hoist (`r1_op2`)**: an operand that is contiguous in the innermost dim
  and broadcast across *every* outer dim (`b` of row-broadcast) is materialised **once per core**
  instead of once per task (`hoist_a`/`hoist_b`, provably inert for hidden fp16/bf16 and cnn).
  row-broadcast fp32 −66 %, fp16 −72 %, bf16 −55 %.

All branches passed L0. `r1_op2` became the current best.

### Round 2 — DESIGN §1.6.3 adjudication: 2-D `[R_blk, L]` tile (alternative #4)

Branches derived from `r1_op2`: `r2_op1` (2-D tile, shared UB budget), `r2_op2` (2-D tile with a
dtype-aware element budget `2·cmp_bytes + 4`, letting fp16/bf16 admit a wider `r_blk`).

**Candidate vs current best (`r1_op2`) — Task Duration(us)**

| workload_id | r1_op2 (current best) | r2_op1 | r2_op2 | winner |
|---|---:|---:|---:|---|
| hidden-state-prefill-float16 | 26.70 | 27.18 | 25.86 | r2_op2 |
| hidden-state-prefill-bfloat16 | 27.56 | 26.12 | 27.92 | r2_op1 |
| hidden-state-prefill-float32 | 51.10 | 49.04 | 50.02 | r2_op1 |
| cnn-feat-broadcast-float16 | 167.82 | 74.12 | 38.50 | r2_op2 |
| cnn-feat-broadcast-bfloat16 | 135.18 | 85.80 | 44.68 | r2_op2 |
| cnn-feat-broadcast-float32 | 89.56 | 45.54 | 44.36 | r2_op2 |
| row-broadcast-float32 | 60.72 | 55.18 | 53.06 | r2_op2 |
| row-broadcast-float16 | 51.52 | 32.64 | 32.78 | r2_op1 |
| row-broadcast-bfloat16 | 79.40 | 44.24 | 44.58 | r2_op1 |

`r2_op2` won the broadcast classes decisively (cnn −77 %, row −14…−44 %) and the per-workload
differences on hidden are <3 % (noise). `r2_op2` was merged and became the current best.

- **2-D `[R_blk, L]` tile (`r2_op1`)**: DESIGN §1.6.3 alternative #4. `a` is loaded as a
  `[r_blk, L]` contiguous tile; `b` is materialised to the full tile shape via `T.vbrc`
  (`(1,L)→(r_blk,L)` for row-broadcast, `(r_blk,1)→(r_blk,L)` for cnn-feat); `T.vcmp`/`T.vcast`
  and the `bool→int8` chain run on the rank-2 tile. The per-task emit cost is amortised over
  `r_blk` rows, which is exactly the design's emission-dominated class. **Adjudication result:
  alternative #4 wins** — cnn-feat fp32 89.6→45.5 µs, row-broadcast fp32 60.7→55.2 µs; the ≥0.97×
  threshold of §1.6.3 is met, so #4 replaces #1 for the broadcast classes (per-shape dispatch at
  factory-build time; the same-shape class stays on the 1-D path).
- **dtype-aware 2-D budget (`r2_op2`)**: `max_2d_elems = UB / ((2·cmp_bytes + 4)·1.7)`, so
  fp16/bf16 get `r_blk=4` on cnn-feat L=3136 (vs `r_blk=2` for fp32). cnn fp16 74.1→38.5 µs,
  cnn bf16 85.8→44.7 µs.

All branches passed L0. `r2_op2` became the current best and was frozen as the deliverable.

### Round 3 — negative control: `r_blk` cap at 2

The round-2 winner picks the **largest** UB-fitting `r_blk` (cnn fp16/bf16 → 4). A hypothesis that
the tight tile (≈170 KB inflated) starves auto multi-buffer was tested by capping `r_blk` at 2
(`ge_fwd_kernel_round3_op1.py`) — matching cnn fp32's `r_blk=2`. The affected workloads were
re-measured.

**Candidate vs current best (`r2_op2`) — Task Duration(us)**

| workload_id | r2_op2 (current best) | r3_op1 | Δ | winner |
|---|---:|---:|---:|---|
| cnn-feat-broadcast-float16 | 38.50 | 68.279 | +77 % | r2_op2 |
| cnn-feat-broadcast-bfloat16 | 44.68 | 85.678 | +92 % | r2_op2 |

`r3_op1` passed L0 but **regressed** both workloads ~2×; the hypothesis is falsified and the
branch is discarded (recorded in `perf_records.jsonl` for auditability). This confirms `r_blk=4`
is the genuine optimum for the cnn-feat fp16/bf16 class under this toolchain — the design space
here is exhausted. Current best unchanged.

## Autotune Log

No autotune framework was used. The parameter searches were hand/scanned:
`block_size ∈ {4096, 8192}` per dtype (round 1) and `r_blk` ∈ {2, 4} for the 2-D tile (rounds 2–3,
`r_blk` derived from the UB budget and capped once as a control). `bench.py` defaults to
`TUNED_BLOCK_BY_DTYPE` when the impl module exposes it, so the tuned `block_size` values are active
in all candidate/final measurements.

## Final Performance Test Data (round 3, superseded 2026-10-22)

Deliverable `perf_opt/ge_fwd_kernel.py` (candidate `final` = `r2_op2`), all 9 tune workloads
measured in one `phase=final`, round-3 session. `baseline_us` = round-0 `phase=baseline`; both
columns are the same-candidate records in `perf_records.jsonl`.

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-float16 | 82.578 | 26.819 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.338 | 27.999 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-float32 | 95.118 | 50.259 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-float16 | 528.969 | 40.619 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 661.607 | 39.159 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-float32 | 352.613 | 44.919 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-float32 | 177.676 | 50.139 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-float16 | 184.136 | 32.219 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-bfloat16 | 177.196 | 44.319 | final |

### Roofline Ratio (report口径, `roofline_ratio.py`, BW=1.8 TB/s) and target

| workload_id | baseline µs | final µs | improvement | baseline Ratio | final Ratio | ≥60 %? | raw final profile |
|---|---:|---:|---:|---:|---:|:--:|---|
| hidden-state-prefill-float16 | 82.578 | 26.819 | +67.5 % | 22.57 % | **69.51 %** | ✓ | `profiles/final/round3/final_2443630b/hidden-state-prefill-float16_f48f537e` |
| hidden-state-prefill-bfloat16 | 91.338 | 27.999 | +69.3 % | 20.41 % | **66.58 %** | ✓ | `…/hidden-state-prefill-bfloat16_274d79ca` |
| hidden-state-prefill-float32 | 95.118 | 50.259 | +47.2 % | 39.20 % | **74.18 %** | ✓ | `…/hidden-state-prefill-float32_dac1dbdc` |
| cnn-feat-broadcast-float16 | 528.969 | 40.619 | +92.3 % | 2.34 % | **35.14 %** | ✗ | `…/cnn-feat-broadcast-float16_f1cddcf9` |
| cnn-feat-broadcast-bfloat16 | 661.607 | 39.159 | +94.1 % | 1.87 % | **36.45 %** | ✗ | `…/cnn-feat-broadcast-bfloat16_f5a51caa` |
| cnn-feat-broadcast-float32 | 352.613 | 44.919 | +87.3 % | 7.76 % | **63.55 %** | ✓ | `…/cnn-feat-broadcast-float32_2ff59659` |
| row-broadcast-float32 | 177.676 | 50.139 | +71.8 % | 20.99 % | **74.38 %** | ✓ | `…/row-broadcast-float32_7e902357` |
| row-broadcast-float16 | 184.136 | 32.219 | +82.5 % | 10.13 % | **57.87 %** | ✗ | `…/row-broadcast-float16_410291d8` |
| row-broadcast-bfloat16 | 177.196 | 44.319 | +75.0 % | 10.52 % | **42.07 %** | ✗ | `…/row-broadcast-bfloat16_00d04600` |

**≥60 % target: 5 / 9 tune workloads.** The task-provided baseline reference table
(hidden 22.45/20.25/39.02 %, cnn 2.01/2.84/8.02 %, row 22.55/11.63/11.17 %) differs from the
round-0 session numbers here only by normal cross-session measurement variance; the ratios above
are recomputed from the same `self-profiled` CSVs via `roofline_ratio.py`.

## Smoke precision and regression

- Smoke `smoke-1k` fp32/fp16/bf16 — precision-only, no perf profile (per the skill rule smoke is
  never performance-tuned). Final kernel self-test passes.
- Phase-3 regression on `perf_opt/ge_fwd_kernel.py`:
  `python3 perf_opt/ge_fwd_kernel.py --level all` → **all PASS** (L0 fourteen items + L1 +
  boundary; `Summary max_diff per dtype: all 0`). Log: `perf_opt/logs/verify/final_all.log`.

## Why ≥60 % is not reached for the remaining 4 workloads

The report ratio numerator is **GM read traffic only**, while every ge case also writes the 1-byte
bool carrier, so the ceiling is `read/(read+write) · B_eff/1.8`. For the fp16/bf16 broadcast
classes `read = a`, `write = N`, i.e. read fraction 2/3 → ceiling ≈ `0.667·B_eff/1.8`. The best
total bandwidth observed on this device is ≈ 1.67 TB/s (row-broadcast fp32), so the practical
ceiling for a 2/3-read workload is ≈ 62 % — i.e. ≥60 % requires essentially the maximum device BW
with almost no slack.

- **cnn-feat fp16/bf16 (35.1 / 36.5 %)**: 1024 `[4,3136]` tasks, read 25.7 MB. The per-task
  scalar-`b` load + `T.vbrc((r_blk,1)→(r_blk,L))` adds a vector op and a serial b→vbrc→vcmp
  dependency on top of the minimal `vcmp + bool→f16 + f16→i8` chain. `r_blk` is UB-capped (the
  `r_blk=2` control regressed ~2×, round 3), and the cast chain cannot be shortened: `T.vcmp`
  emits bool and bool cannot be stored to GM (CG-2026-0022), while `T.vcast` has no `i1→i8` pair
  (`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1) — so `bool→f16→i8` is the ISA
  minimum. bf16 is additionally Vector-bound (`vec_ratio 0.55`, above mte2 0.46).
- **row-broadcast fp16 (57.9 %)**: read 33.6 MB; the 2-D tile is forced to `r_blk=2` (r_blk=4
  needs ≈ 223 KB > 192 KB UB at the 1.7× buffer inflation), leaving it 3.6 % short of the 60 %
  cutoff — the class is at the pipeline-overhead floor for this design (b is already hoisted once
  per core).
- **row-broadcast bf16 (42.1 %)**: same bytes as fp16 but **Vector-bound** (`vec_ratio 0.586` vs
  fp16 0.386) — the bf16 `vcmp`/`vcast` path is the hardware cost, not a tiling parameter.

Per `_shared/standards/perf-feedback.md` §1 this is **not** filed as `[DESIGN_LIMIT]`: the
shortfall is a report-metric (read-only numerator) + hardware-cost (bf16 Vector throughput,
per-task broadcast dependency) limit, not a contradiction of a specific DESIGN.md assumption; the
design correctly predicted emission-domination, and no structural `>2×` *unreachable* acceleration
is demonstrated (the 2-D alternative already delivered the reachable structural win). No
`perf_feedback.md` is produced.

## Final Summary

- **Deliverable**: `perf_opt/ge_fwd_kernel.py` (candidate `final` = `r2_op2`).
- **Merged optimization points**: (1) per-dtype `block_size` (fp16/bf16 8192, fp32 4096);
  (2) outer-broadcast-operand hoist; (3) DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D tile for
  the broadcast classes with a dtype-aware UB budget (`r_blk=4` fp16/bf16, `r_blk=2` fp32).
- **Result**: 9 / 9 tune workloads improved 47–94 % in `Task Duration`; **5 / 9 reach the ≥60 %
  Roofline Ratio target** (hidden fp16/bf16/fp32, cnn-feat fp32, row-broadcast fp32).
- **Stop reason**: `success` for the reachable classes + `plateau` for the fp16/bf16 broadcast
  classes (round-3 negative control confirmed the tiling space is exhausted; structural candidate
  list empty).
- **Leftover**: cnn-feat fp16/bf16 and row-broadcast fp16/bf16 — bounded by the read-only report
  metric ceiling (~62 %) and the bf16/additional-vector-op hardware cost; honestly reported, not
  forced. All artifacts are gate-4 compliant.

## Skill Retrospective

- **Process issue (R, strong).** The DESIGN-mandated §1.6.3 alternative-#4 adjudication
  (`r2_op1`/`r2_op2`) was already run, but the prior session froze `phase=final` records
  mid-way and never wrote `opt_log.md` / `tuning_status`; the deliverable was left unmaterialised
  even though the run data existed. **Rule to distil:** Phase 3 must emit `opt_log.md` +
  `workload_inventory.json` `tuning_status` + the Final table in the same session as the
  `phase=final` collection — otherwise an interrupt orphans a completed run.
  - evidence: gate-4 failure list on resume (`S4-OPTLOG`, `S4-WORKLOAD-INVENTORY`,
    `S4-OPTLOG-COMPTABLE`, `S4-PERF-RECORDS-RECON`) despite complete profiles/records.
  - repro: n/a (process rule).
  - toolchain_stamp: tilelang npuir `0.1.2+eb362ab`, Ascend910 40-vector-core, CANN 9.0.0.
- **Validated optimization (P, positive).** The DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D tile
  is a decisive win on the emission-dominated broadcast classes and is the *first* time it is
  merged for this family (the sister `eq` run froze it out): cnn-feat fp16 168→38.5 µs,
  cnn-feat bf16 135→44.7 µs, cnn-feat fp32 89.6→44.4 µs, row-broadcast fp16 51.5→32.8 µs.
  Mechanism: per-task emit cost (3–4 vector ops + per-task index divmod) is amortised over
  `r_blk` rows; `T.vbrc` materialises the broadcast operand to the full tile shape because
  `T.vcmp` has no auto-broadcast (CG-2026-0021).
  - evidence: round-2 B2 table; `profiles/candidate/round2/r2_op2_*`.
  - repro: `perf_opt/ge_fwd_kernel_round2_op2.py` (2-D `main2` block in the factory).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive).** Outer-broadcast-operand hoist (materialise a fully
  outer-broadcast operand once per core, not once per task) → row-broadcast fp32 177→60.7 µs,
  fp16 184→51.5 µs, bf16 177→79.4 µs (round-1 `r1_op2`).
  - evidence: round-1 B2 table; `profiles/candidate/round1/r1_op2_*`.
  - repro: `perf_opt/ge_fwd_kernel_round1_op2.py` (`hoist_a`/`hoist_b`).
  - toolchain_stamp: same as above.
- **Negative control (D).** Capping the 2-D `r_blk` at 2 for cnn-feat fp16/bf16 (to test an
  auto-multi-buffer-starvation hypothesis) **regresses** both ~2× (38.5→68.3 µs; 44.7→85.7 µs).
  The largest UB-fitting `r_blk` (4 for fp16/bf16, 2 for fp32) is the real optimum — the tight
  tile does not starve the pipeline; tile granularity dominates.
  - evidence: round-3 B2 table; `profiles/candidate/round3/r3_op1_*`.
  - repro: `perf_opt/ge_fwd_kernel_round3_op1.py` (`_R_BLK_CAP = 2`).
  - toolchain_stamp: same as above.
- **Metric caveat (D).** The single-operator Roofline ratio is **read-only**, so the ≥60 % target
  is only reachable for read-heavy classes; with a 1-byte bool output the read fraction is 2/3,
  capping the practical ratio at ≈ 62 % (best observed `B_eff ≈ 1.67 TB/s`). The bf16 broadcast
  classes are additionally Vector-throughput bound.

---

# Round 4 / 7 (2026-10-22) — 2-D `[R_blk, L]` tile ladder and the col-broadcast stall

Session resources: exclusive chip0 (`ASCEND_RT_VISIBLE_DEVICES=0`), private
`TILEOPS_MSPROF_OUTPUT_DIR=/tmp/opencode/msprof-out-eqge`. Every number is `msprof op`
`Task Duration(us)` (median of the per-launch CSVs, `--launch-count=15`), A/B taken in
**alternating order with even pairs** (`perf_opt/run_ab.py`, 2 reps/branch). Raw data
`perf_opt/profiles/candidate/r{4,7}/`, `perf_opt/profiles/final/r8/`; stdout
`perf_opt/logs/r4,r7,final/`. Rounds 5/6 (UB-resident b column, predicated `vbrc`) were
run on the eq sibling — same structure, same compiler, same conclusion — and are recorded
there; this file records the two rounds that produced changes here.

## Round 4 — measured UB budget replaces the ×1.7 inflation

**现象 (base = round-3 final).** The 2-D budget was `UB // ((2·cmp_bytes+4)·1.7)` and
`r_blk` must divide `run` (a power of two), so the inflation cost a whole tile step:
cnn-feat fp32 → `r_blk=2`, row fp16/bf16 → `r_blk=2`. The gt/le siblings had already
recovered that step with 1.5.

**候选优化点.** Use the UB demand the compiler reports
(`perf_opt/probe_ub_demand.py`, `ub overflow, requires N bits while 1572864 bits
available`): `demand(r_blk) = (2·cmp_bytes+4)·r_blk·L + 128…288 B` — the five logical
tiles are liveness-compacted, the ×1.7 reserve was pure under-utilisation. Budget:
`max_2d_elems = (196 608 − reserve)//(2·cmp_bytes+4)`, `reserve = 512 B (+2·L·elem_bytes`
on the row path). Derived `r_blk`: cnn fp16/bf16 4→4, **cnn fp32 2→4**, **row fp16/bf16
2→4**, row fp32 2→2 (3 fits but does not divide 4096).

**候选 vs current best（`ge_fwd_kernel_round4_op1.py`, round 4）**

| workload | current best (us) | r4op1 (us) | Δ | mte2 base→cand | vec base→cand | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|
| cnn-feat-broadcast-float16 | 37.16 | 37.24 | −0.2 % (tie) | 0.739→0.740 | 0.325→0.325 | PASS |
| cnn-feat-broadcast-bfloat16 | 37.96 | 38.08 | −0.3 % (tie) | 0.397→0.396 | 0.612→0.612 | PASS |
| cnn-feat-broadcast-float32 | 44.74 | **39.00** | **+12.8 %** | 0.612→0.612 | 0.440→0.453 | PASS |
| row-broadcast-float32 | 50.41 | 50.25 | +0.3 % (tie) | 0.734→0.733 | 0.348→0.348 | PASS |
| row-broadcast-float16 | 32.16 | **26.59** | **+17.3 %** | 0.735→0.705 | 0.387→0.487 | PASS |
| row-broadcast-bfloat16 | 44.28 | **34.19** | **+22.8 %** | 0.495→0.389 | 0.588→0.724 | PASS |

`memory` axis: `gm_to_ub_datas_kB`/core unchanged within 2 % (640.25 → 640.25 cnn-fp16,
824 → 840 row-fp16 = one extra broadcast row per core); the gain is MTE2 efficiency
(`aiv_mte2_active_bw` 20.5 → 25.2 GB/s cnn-fp16, 36.7 → 50.9 GB/s row-fp16). The three
ties are cases whose `r_blk` did not change (identical binary; ±3–5 % session noise,
bimodal: cnn-bf16 shows 40.68/35.24 on the same file).

## Round 7 — broadcast-group-major task order for the col-broadcast path

**现象.** A geometry-matched diagnostic (12.845 M elements, L=3136, only the broadcast
axis changed) shows the col path pays 1.65× vs the row path at identical tile size
(34.4 vs 20.9 µs), and two timing-only probes on the eq sibling price it as the
single-buffered `8 B GM→UB copy → vbrc → vcmp` chain (≈0.7 µs/tile): removing just that
load gives 22.4 µs, removing just the `vbrc` 22.1 µs.

**候选优化点 (single lever).** Enumerate tiles *broadcast-group major* and hand each core
a contiguous block (`task = cid·A + i·B`, `sub = task // R_lead`, `lead = task % R_lead`)
so a core walks `R_lead` consecutive tiles sharing the SAME b row block; then predicate
**only** the GM scalar load (`if i == 0 or lead == 0`). `T.vbrc` stays unconditional —
predicating it too is incorrect (max_diff=1 on 2174 elements) and doubles b2's UB
(+50 KB). The row path keeps the original cyclic mapping.

| workload | current best (us) | r7op1 (us) | Δ | mte2 | vec | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|
| cnn-feat-broadcast-float16 | 37.09 | **23.37** | **+37.0 %** | 0.742→0.597 | 0.325→0.505 | PASS |
| cnn-feat-broadcast-bfloat16 | 38.05 | **31.82** | **+16.4 %** | 0.396→0.386 | 0.611→0.684 | PASS |
| cnn-feat-broadcast-float32 | 43.40 | **39.11** | **+9.9 %** | 0.610→0.618 | 0.448→0.446 | PASS |
| row-broadcast-float16 | 32.19 | **25.71** | **+20.1 %** | 0.734→0.710 | 0.389→0.482 | PASS |
| row-broadcast-bfloat16 | 43.89 | **34.05** | **+22.4 %** | 0.494→0.390 | 0.590→0.724 | PASS |
| row-broadcast-float32 | 49.93 | 50.13 | −0.4 % (tie) | 0.736→0.734 | 0.345→0.346 | PASS |

`--level all` on the merged `perf_opt/GeFwdOp.py`: L0 + L1 + boundary all pass,
`max_diff = 0` for every dtype (`perf_opt/logs/final/ge_level_all.log`). The L1
`bool-vbrc/(16,256,56,56)` case that failed in the first merge attempt (predicated
`i1→f16` cast) is fixed in the final module by predicating only the GM load.

## Final Performance Test Data

Final deliverable `perf_opt/GeFwdOp.py` (sha256 `e5ed07856f0a7c2e…`, see
`perf_records.jsonl`), `candidate_id = final`, content = `ge_fwd_kernel_round7_op1.py` +
the L1 bool fix. One `phase=final` record per tune workload, measured on that file in a
single session (round 8). Ratio = report口径 Roofline (`roofline_ratio.py`).

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|--:|--:|---|
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-float16 | 82.578 | 25.920 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.338 | 28.240 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | hidden-state-prefill-float32 | 95.118 | 48.460 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-float16 | 528.969 | 23.160 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 661.607 | 31.900 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | cnn-feat-broadcast-float32 | 352.613 | 38.320 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-float32 | 177.676 | 50.420 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-float16 | 184.136 | 28.220 | final |
| examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py::ge_fwd_kernel::main | row-broadcast-bfloat16 | 177.196 | 34.280 | final |

Roofline Ratio before → after (before = round-3 record scaled by identical read traffic;
those rows are now `phase=merged`):

| case | duration before → after (us) | Ratio before → after | ≥60 % |
|---|---|--:|:--:|
| hidden fp32 | 50.26 → 48.46 | 75.11 → **76.94** | ✅ |
| hidden fp16 | 26.82 → 25.92 | 69.82 → **71.92** | ✅ |
| hidden bf16 | 28.00 → 28.24 | 69.20 → **66.01** | ✅ |
| row fp32 | 50.14 → 50.42 | 72.98 → **73.96** | ✅ |
| cnn fp32 | 44.92 → 38.32 | 63.27 → **74.49** | ✅ |
| row fp16 | 32.22 → 28.22 | 57.69 → **66.07** | ✅ **新达标** |
| cnn fp16 | 40.62 → 23.16 | 38.53 → **61.63** | ✅ **新达标** |
| row bf16 | 44.32 → 34.28 | 42.17 → 54.39 | ❌ |
| cnn bf16 | 39.16 → 31.90 | 30.41 → 44.74 | ❌ |

**7/9 达标 (was 5/9).** `row fp16` overtook its own round-7 A/B value (28.22 vs 25.71,
−9 %) purely from session wobble — the same binary; the final record is kept as measured
(66.07 %, still above the 60 % line, and it is what `perf_records.jsonl` contains).

### Why the two bfloat16 cases still miss 60 %

- **row bf16 (54.39 %).** `vec=0.722, mte2=0.390` → Vector-bound. Per element:
  `vcmp(bf16)` + two casts (`i1→f16`, then `f16→i8, rint`). `T.vcast` 2.2.1 does not
  document `i1→i8`; the gt/le sibling measured that the direct form compiles bit-exactly
  but yields zero benefit (BiShengIR materialises a 2 B/elem intermediate anyway), so it
  is not a lever. bf16 `vcmp` costs ≈2× fp16's. `r_blk` is
  capped at 4: `r_blk=8` at L=4096 needs 256 KB of UB (measured 8·3136 case: 200 832 B,
  i.e. 25 088·r_blk + 128), and even a hypothetical 6 B/elem alias would need 196 736 B
  > 196 608 B. The gt sibling lands at 54.65 % with the same structure → family ceiling
  for the read-only-Roofline口径 (ceiling 2/3·B_eff/1.8 ≈ 62 %).
- **cnn bf16 (44.74 %).** Same Vector bound (`vec=0.684`), L=3136 caps `r_blk` at 4.
  Comparing in fp16 would be cheaper but is not exact over the bf16 exponent range →
  rejected as semantically unsafe.

Both are hardware/口径 ceilings, not tiling parameters — **no `perf_feedback.md` is
produced**: neither [DESIGN_LIMIT] trigger holds (no DESIGN.md assumption is contradicted
and no >2× structural candidate remains; gt/le confirm the same ceiling independently).

## Skill Retrospective (2026-10-22 r_blk round)

**流程问题**

1. 三轮调优把 ×1.7 auto-multi-buffer 膨胀当常数沿用，而 `probe_ub_demand.py` 一次编译
   就能反解出精确足迹（`(2·cmp+4)·r_blk·L + 128…288 B`，无膨胀）。4 个广播 case 因此各
   少走一档 tile。规则：**选 tile 前先读编译器自己的 overflow 报文**。
2. 广播**方向**从未单独对照：row 与 col 在同几何下差 1.65×（20.9 vs 34.4 µs），只比跨
   shape 的绝对时延看不出来。新增诊断 case（同元素数、同 L、只改广播轴）是发现它的关键。
3. 同二进制仍呈 ±10 % 双峰（邻居 chip 占用 HBM）。交替 A/B 对决策有效，但 `final` 单点
   记录会继承抖动（本算子 row fp16 25.0–28.4 µs 跨 session）。逼近判据线（本站 cnn fp16
    ≈61.6–62.0 %）的 case 建议 final 也走 `ab_test.py` 协议。

**value points**（与 eq 同批，见 `eq/eq_kernel/perf_opt/opt_log.md` 的对应表：
CONST-2dtile-ub-demand-k delta = D；PL-1.21 col-broadcast delta = P；dynamic-UB-sub-view
vbrc scalarisation + predicated vbrc = D(trap)；UB-demand-first 规则 = R）。本算子的独立
证据：round 4 / round 7 的 `perf_records.jsonl`（两 op 同向、量级一致）。
