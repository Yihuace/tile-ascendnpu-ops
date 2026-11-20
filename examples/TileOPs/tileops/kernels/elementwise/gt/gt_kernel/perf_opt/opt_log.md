# gt_fwd_kernel Stage 4 Tuning Log

- operator: `gt_fwd_kernel` (`GtFwdOp`, elementwise `a > b`, bool output carrier = int8)
- project: `gt`; scenario `optimize`; mode `full`
- baseline kernel (read-only): `gt_fwd_kernel/gt_fwd_kernel.py`
- deliverable: `perf_opt/gt_fwd_kernel.py` (candidate `final`)
- benchmark source: `examples/TileOPs/benchmarks/ops/bench_gt.py::test_gt_manifest_bench`
- **primary metric**: `msprof op` `Task Duration(us)` (kernel-only, median of a
  `--launch-count=15 / --warm-up=5` run). No other timing basis is used.
- Roofline Ratio (report口径, BW = 1.8 TB/s):
  `ratio = (Σ aiv/aic r0+r1 read_cache_miss_allocate × 128) / (Task Duration(s) × 1.8e12)`
  — recomputed from the same `L2Cache_*`/`OpBasicInfo_*` CSVs by `roofline_ratio.py`
  (matches `tileops/benchmark/msprof_roofline.compute_from_csvs`).
- toolchain: `tilelang 0.1.2+302ad3c` (HEAD `302ad3ce`), CANN 9.0.0, Ascend910
  (40 Vector cores, `UB = 192 KB`), `TILELANG_ASCEND_MODE=Developer`.
- Stage-4 target: **≥ 60 %** Roofline Ratio per case ("reach target, not optimum").
- Resume note: rounds 1–3 and the `hidden` `phase=final` records were produced in an
  earlier session interrupted before `opt_log.md` / inventory `tuning_status` were written;
  this session verified the deliverable, collected the remaining `phase=final` records and
  produced this log (per the Phase-3 process rule distilled in the sister `ge` retrospective).

## Performance Test Data (baseline)

Workload inventory: `perf_opt/workload_inventory.json`, expanded from
`bench_gt.py::test_gt_manifest_bench` (2-input manifest; 4 manifest entries × 3 dtypes = 12
cases). Classification: 3 `smoke` (`smoke-1k` = manifest idx 0 fp32 + its fp16/bf16 siblings,
precision-only), 9 `tune`. `skipped`: none.

Baseline = the Stage-3 kernel (`../gt_fwd_kernel.py`, sha `8c1dd7ee…`) at its shipped
`block_size=1024` (`phase=baseline`, round 0). The `ge`-family block-size scan
(`bs=None`→4096) was run first as an exploration only; the **official** baseline set is the
`block_size=1024` re-run that matches the task reference table.

| kernel (id suffix `::gt_fwd_kernel::main`) | workload_id | dtype | shape (a \| b) | Task Duration(us) | mte2/mte3/vec/scalar | Roofline Ratio |
|---|---|---|---:|---:|---:|---:|
| `…/gt_fwd_kernel.py` | hidden-state-prefill-float16 | fp16 | [2048,4096] \| [2048,4096] | 83.078 | .806/.334/.166/.124 | 22.44 % |
| `…/gt_fwd_kernel.py` | hidden-state-prefill-bfloat16 | bf16 | [2048,4096] \| [2048,4096] | 91.198 | .725/.303/.261/.142 | 20.44 % |
| `…/gt_fwd_kernel.py` | hidden-state-prefill-float32 | fp32 | [2048,4096] \| [2048,4096] | 94.398 | .806/.303/.168/.110 | 39.50 % |
| `…/gt_fwd_kernel.py` | cnn-feat-broadcast-float16 | fp16 | [16,256,56,56] \| [256,1,1] | 511.890 | .921/.201/.063/.047 | 2.42 % |
| `…/gt_fwd_kernel.py` | cnn-feat-broadcast-bfloat16 | bf16 | [16,256,56,56] \| [256,1,1] | 578.868 | .896/.182/.089/.050 | 2.14 % |
| `…/gt_fwd_kernel.py` | cnn-feat-broadcast-float32 | fp32 | [16,256,56,56] \| [256,1,1] | 345.593 | .860/.269/.116/.078 | 7.92 % |
| `…/gt_fwd_kernel.py` | row-broadcast-float32 | fp32 | [4096,4096] \| [1,4096] | 177.496 | .819/.328/.182/.085 | 21.01 % |
| `…/gt_fwd_kernel.py` | row-broadcast-float16 | fp16 | [4096,4096] \| [1,4096] | 183.536 | .836/.334/.168/.088 | 10.16 % |
| `…/gt_fwd_kernel.py` | row-broadcast-bfloat16 | bf16 | [4096,4096] \| [1,4096] | 177.316 | .764/.308/.272/.108 | 10.52 % |

Raw profiles: `profiles/baseline/round0/baseline_8ba8496a/<workload_id>_<hash>/`.

**Design estimate vs measured (D-2 deviation, DESIGN.md §1.6.0).** The design-time emission-term
lower bound (block_size 4096, fp32 representative, 40 Vector cores) vs the measured baseline:

| representative | design emission-term bound | measured baseline | deviation |
|---|---:|---:|---:|
| hidden-state-prefill fp32 | ≈ 78 µs | 94.398 µs | 1.21× |
| cnn-feat-broadcast fp32 | ≈ 206 µs | 345.593 µs | 1.68× |
| row-broadcast fp32 | ≈ 155 µs (b-row re-read upper ≈ 144 µs) | 177.496 µs | 1.15× |

The estimate correctly identified **emission domination** (every task = 3–4 Vector ops:
`vcmp` + `bool→f16` + `f16→i8`, plus a per-task `vbrc` on the broadcast operand); the residuals
are the per-task index divmod / `vbrc` serial dependency the model did not price. This is the
quantified motivation for the DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D tile executed below.

## Iteration Log

### Round 1 — per-dtype block_size (`r1_op1`) and outer-broadcast hoist (`r1_op2`)

Base = `baseline` (current best). Two independent single-point branches.

- `r1_op1` — per-dtype `block_size` (fp16/bf16 8192, fp32/int32 4096; shipped default 1024),
  exposed as `TUNED_BLOCK_BY_DTYPE`. The flat same-shape stream (`hidden-*`) is MTE2-bound and
  wants the largest UB-fitting tile; broadcast rows already have `chunks == 1` at 4096/8192.
- `r1_op2` — `r1_op1` + materialise a fully outer-broadcast operand **once per core**
  (`hoist_a`/`hoist_b` when `chunks == 1` and all outer strides are 0) instead of re-loading
  it every task.

**Candidate vs current best (`baseline`) — Task Duration(us)**

| workload_id | baseline (current best) | r1_op1 | r1_op2 | winner |
|---|---:|---:|---:|---|
| hidden-state-prefill-float16 | 83.078 | 26.639 | 27.019 | r1_op2 (tie; O(1)) |
| hidden-state-prefill-bfloat16 | 91.198 | 27.839 | 27.479 | r1_op2 |
| hidden-state-prefill-float32 | 94.398 | 49.499 | 49.939 | r1_op1 |
| cnn-feat-broadcast-float16 | 511.890 | 129.837 | 177.156 | r1_op1 |
| cnn-feat-broadcast-bfloat16 | 578.868 | 153.397 | 153.177 | tie |
| cnn-feat-broadcast-float32 | 345.593 | 81.978 | 95.198 | r1_op1 |
| row-broadcast-float32 | 177.496 | 183.416 | 63.339 | r1_op2 |
| row-broadcast-float16 | 183.536 | 174.717 | 51.439 | r1_op2 |
| row-broadcast-bfloat16 | 177.316 | 183.176 | 79.478 | r1_op2 |

All branches L0-pass. The block-size win is decisive on the flat stream (hidden 83→27 µs); the
hoist is decisive on `row-broadcast` (a fully outer-broadcast `b` → per-core load instead of
per-task, 177→63 µs) but slightly regresses the `cnn` col-broadcast (its `b` varies per channel
so it is not hoistable — the extra branch is pure code bloat there). `r1_op2` merged and became
the current best (the row win dominates; the cnn overlap is recovered by the 2-D tile next round).

### Round 2 — `[R_blk, L]` 2-D tile (`r2_op1`) and dtype-aware UB budget (`r2_op2`)

Base = `r1_op2` (current best). This is the DESIGN §1.6.3 **alternative-#4 adjudication** — a
2-D `[R_blk, L]` row tile per task so the per-task emit chain (`vcmp` + 2 `vcast`, plus the
`vbrc`) is amortised over `R_blk` rows; `T.vcmp` has no auto-broadcast (CG-2026-0021), so the
broadcast operand is materialised via `T.vbrc`.

- `r2_op1` — add the 2-D tile (`main2`), `R_blk` from the conservative 1.7× UB budget.
- `r2_op2` — dtype-aware 2-D budget: `a2 + b2 (cmp) + cond(1) + yf(2) + y8(1) = 2·E + 4` B/elem,
  so fp16/bf16 (8 B/elem) admit a wider tile (`R_blk=4` on cnn L=3136) than fp32 (12 B/elem).

**Candidate vs current best (`r1_op2`) — Task Duration(us)**

| workload_id | r1_op2 (current best) | r2_op1 | r2_op2 | winner |
|---|---:|---:|---:|---|
| hidden-state-prefill-float16 | 27.019 | 26.579 | 25.219 | r2_op2 |
| hidden-state-prefill-bfloat16 | 27.479 | 27.499 | 27.319 | r2_op2 |
| hidden-state-prefill-float32 | 49.939 | 50.439 | 47.839 | r2_op2 |
| cnn-feat-broadcast-float16 | 177.156 | 78.278 | 34.499 | r2_op2 |
| cnn-feat-broadcast-bfloat16 | 153.177 | 90.018 | 46.719 | r2_op2 |
| cnn-feat-broadcast-float32 | 95.198 | 45.439 | 44.579 | r2_op2 |
| row-broadcast-float32 | 63.339 | 50.039 | 50.119 | r2_op1 (tie) |
| row-broadcast-float16 | 51.439 | 32.339 | 32.359 | tie |
| row-broadcast-bfloat16 | 79.478 | 44.219 | 43.999 | r2_op2 |

All branches L0-pass. The 2-D tile is the decisive structural win for the broadcast classes
(cnn fp16 177→34.5 µs; row fp16 51→32 µs) and the dtype-aware budget adds another 2× on
cnn fp16/bf16. `r2_op2` merged and became the current best.

### Round 3 — relaxed 2-D inflation for the native fp16/bf16 path (`r3_op1`)

Base = `r2_op2` (current best). On the native fp16/bf16 2-D path the staging copy goes straight
into `a2` (`a_contig_plain`), so the real tile footprint is exactly `2·E + 4` B/elem with no
extra staging buffer → relax the conservative 1.7× auto-multi-buffer inflation to 1.5× for
`E ≤ 2` (integer/upcast paths keep 1.7×). This admits `R_blk=4` at L=4096 (row-broadcast fp16/bf16)
where 1.7× only gave `R_blk=2`.

**Candidate vs current best (`r2_op2`) — Task Duration(us)**

| workload_id | r2_op2 (current best) | r3_op1 | Δ | winner |
|---|---:|---:|---:|---|
| hidden-state-prefill-float16 | 25.219 | 26.879 | +6.6 % | r2_op2 |
| hidden-state-prefill-bfloat16 | 27.319 | 26.539 | −2.9 % | r3_op1 |
| hidden-state-prefill-float32 | 47.839 | 51.099 | +6.8 % | r2_op2 |
| cnn-feat-broadcast-float16 | 34.499 | 41.099 | +19.1 % | r2_op2 |
| cnn-feat-broadcast-bfloat16 | 46.719 | 39.379 | −15.7 % | r3_op1 |
| cnn-feat-broadcast-float32 | 44.579 | 45.319 | +1.7 % | r2_op2 |
| row-broadcast-float32 | 50.119 | 49.599 | −1.0 % | r3_op1 |
| row-broadcast-float16 | 32.359 | 28.459 | −12.1 % | r3_op1 |
| row-broadcast-bfloat16 | 43.999 | 34.399 | −21.8 % | r3_op1 |

All branches L0-pass. `r3_op1` gives a large, *structural* gain on the priority `row-broadcast`
class (fp16 −12 %, bf16 −22 %) and on cnn bf16 (−16 %); the apparent cnn fp16 "+19 %" is not a
code difference — for cnn-feat L=3136 both budgets select `R_blk=4` (1.7× → `max_2d_r=4`; 1.5× →
`max_2d_r=5→4`), so the two kernels are identical there and the spread is cross-run measurement
variance (the same file later measured 41.1→46.0 µs). Following the "priority class wins +
no credible regression" rule, `r3_op1` merged and was frozen as the deliverable
(`gt_fwd_kernel.py`, sha `d2aefb9e…`).

Per workload, the current best after the merge is `r3_op1` (identical to the deliverable); the
final current best is `final`.

### Round 4 (20261022) -- real-footprint UB budget + the r_blk 5/6/7 ladder

**Phenomenon (from `perf_records.jsonl` `phase=final` of round 3 + a same-session
re-measure `r3_op1_ref`).** 6/9 tune workloads are >= 60 %; the three missing ones are
structurally different from each other, which is why they get different candidates:

| workload_id | Task Duration(us) | mte2 | vec | mm_read_bw GB/s (per core) | diagnosis |
|---|---:|---:|---:|---:|---|
| row-broadcast-bfloat16 | 33.92 | 0.389 | **0.725** | 25.0 | Vector-bound (bf16 vcmp/vcast throughput) |
| cnn-feat-broadcast-float16 | 40.90 | **0.782** | 0.271 | 15.4 | MTE2-issue-bound; achieved read BW is 2.3x below row-fp16 (35.6) |
| cnn-feat-broadcast-bfloat16 | 36.36 | 0.416 | 0.594 | 17.7 | both pipes ~sum to 1.0 -> no DMA/Vector overlap |
| cnn-feat-broadcast-float32 | 43.84 | 0.605 | 0.446 | 30.2 | already >= 60 %, but `r_blk=2` is far from the UB ceiling |
| row-broadcast-float16 | 26.92 | 0.706 | 0.489 | 35.6 | already >= 60 % (69.3 %) |
| row-broadcast-float32 | 49.40 | 0.734 | 0.348 | 35.8 | already >= 60 % (75.5 %) |

**Hypothesis under test (conductor-supplied, cross-op evidence from ge/eq).** The 2-D
budget still divides by a blanket 1.5x (fp16/bf16) / 1.7x (fp32) auto-multi-buffer factor.
On the *native* path the tile footprint is exactly `2*cmp_bytes + 4` B/elem
(`a2 + b2 + cond + yf + y8`), so with inflation 1.0 the tile ladder should open up to
`r_blk` 5/6/7. Mechanised arithmetic first (UB = 196,608 B, `run` = `outer[-1]`):

| class | `run` | shipped `r_blk` (1.5x/1.7x) | budget 1.0 + 16 KB reserve | note |
|---|---:|---:|:--:|---|
| cnn-feat fp16/bf16 (L=3136) | 256 | 4 (`max_2d_r`=5, 256%5!=0) | 7 | 256 = 36x7 + 4 -> needs a masked tail |
| row-broadcast fp16/bf16 (L=4096) | 4096 | 4 | 5 | 4096 = 819x5 + 1 -> needs a masked tail |
| cnn-feat fp32 (L=3136) | 256 | **2** | **4** | 256%4 == 0 -> no tail needed |
| row-broadcast fp32 (L=4096) | 4096 | 2 | 3 | tail 1 |

**Branches (all from the same base, `r3_op1_ref`; one main point each).**

- `r4_op1` -- replace the 1.5x/1.7x divisor with the real footprint:
  `max_2d_elems = (UB - staging - 16 KB reserve) / (2*cmp_bytes + 4)`, staging = the one
  live `b_in` (1,L) carrier in row mode (0 in col mode). No tail.
- `r4_op2` -- `r4_op1` + **static masked tail** (`groups_per_lead = ceildiv(run, r_blk)`,
  tail rows copied through `a2[0:tail_n,0:L]` / `y8[0:tail_n,0:L]`, no cross-core overlap),
  `r_blk` cap 6 -> cnn 6, row 5, fp32-row 3.
- `r4_op3` -- `r4_op1` + **per-core `b` preload** for the col class (`b` is only 256 elems,
  load once per core, take the per-task `vbrc` source from UB instead of a tiny 8-12 B GM
  read inside the `b -> vbrc -> vcmp` chain).
- `r4_op4` -- `r4_op1` + **col `r_blk` cap 2** (50 KB tile): is the missing DMA/Vector
  overlap recoverable by leaving UB room for the compiler's auto multi-buffer?

**Candidate vs current best (`r3_op1_ref`) -- Task Duration(us)** (single msprof run,
`--launch-count=15 --warm-up=5`, plus the interleaved A/B verdict where it decides):

| workload_id | ref | r4_op1 | r4_op2 | A/B verdict (3 pairs) | L0 |
|---|---:|---:|---:|---|:--:|
| cnn-feat-broadcast-float32 | 43.84 | **38.72** | 39.86 | op1 vs ref **B_better -8.21 %** (43.56 -> 39.59 median) | pass |
| cnn-feat-broadcast-float16 | 40.90 | 40.90 | 45.92 | op2 vs ref **tie** (+0.85 %) | pass |
| cnn-feat-broadcast-bfloat16 | 36.36 | 44.66 | 46.72 | op2 vs ref **tie** (+0.98 %) | pass |
| row-broadcast-bfloat16 | 33.92 | 33.94 | 33.94 | op2 vs ref **tie** (+0.12 %) | pass |
| row-broadcast-float16 | 26.92 | 26.28 | 29.80 | op2 vs ref **inconclusive** (both arms bimodal 25.3/31.2) | pass |
| row-broadcast-float32 | 49.40 | 48.54 | 51.58 | not A/B'd (code identical for op1; no claim) | pass |

The single-run spread on `cnn-*` (36.4 vs 44.7 us for the *same* `r_blk=4` binary) is
machine-state bimodality, so only the interleaved `ab_test.py` verdicts are used for
accept/reject; the JSONs are in `profiles/ab/round4/` and `logs/r4/ab_*.json`.

**Verdicts.**

1. `r4_op1` **adopted**. It changes exactly one class (`cnn-feat fp32`, `r_blk` 2 -> 4,
   tile 128 KB -> 150 KB) and that class is a consistent A/B win (-8.2 %, three pairs, all
   in the same direction). Every other class keeps a byte-identical kernel: cnn fp16/bf16
   and row fp16/bf16 stay at `r_blk=4`, row fp32 stays at 2 (`max_2d_r`=3 and 4096%3!=0),
   hidden uses the 1-D path. `g2u` for cnn fp32 rises 1255.9 -> 1277.3 KB/core (the
   per-task `b` load is amortised over 4 rows instead of 2).
2. `r4_op2` **rejected (config_no_gain)**. The masked tail is correct (L0 14/14 twice,
   `bench.py --check` exact on the target shapes) but the larger tile buys nothing:
   cnn fp16/bf16 and row bf16 are ties, row fp16 is inconclusive. The tail mechanism
   itself is fine -- it is simply not the lever. Kept only as an experiment artifact.
3. `r4_op3` **rejected (implementation trap)**. A UB->UB `T.copy` whose source is a
   *runtime row-offset* UB slice (`b_all[sub*r_blk : sub*r_blk+r_blk, 0:1]`) compiles and
   is exact, but BiShengIR scalarises it: cnn fp16 40.9 -> **3619.4 us** (88x), bf16
   36.4 -> **3629.7 us**. Fatal; recorded so nobody re-derives it.
4. `r4_op4` **rejected (hypothesis falsified)**. Halving the col tile to 2 rows (50 KB,
   room for 2x buffering) costs **+94 %** (cnn fp16 43.8 -> 85.2 us) and **+92 %**
   (bf16 44.5 -> 85.4 us). The DMA/Vector serialisation is therefore not recoverable via
   multi-buffering: tile bytes dominate monotonically, confirming PL-1.21's negative
   control a third time.

**Why the ladder 5/6/7 does not help (measured, not inferred).** `cnn-feat fp16` is
issued-bound, not byte-bound: the same kernel moves 25.7 MB of reads at `r_blk=4` in
40.4 us and *twice* that (51.4 MB, fp32) in 39.8 us. Adding rows per task cannot fix an
issue-rate limit, which is why `r_blk` 4 -> 6 is a tie while 4 -> 2 costs 2x. For
`cnn-feat bf16` the 4th Vector pass (`T.vbrc` materialisation, which `T.vcmp` requires --
CG-2026-0021) plus the 1.85x slower bf16 Vector path is the cost, and neither scales with
tile size. `row-broadcast bf16` is purely Vector-bound (`vec_ratio` 0.725) with 3 passes.

### Round 5 (20261022) -- output-chain pass reduction and the overlap control

Two residual ideas, each one point, each from the round-4 baseline:

- `r5_op1` -- **direct `T.vcast(bool -> int8)`**, dropping the `bool -> f16 -> i8` hop.
  `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 lists only `i1 -> f16` /
  `i1 -> f32`, but that table is stale on this toolchain: `probe_cast_i1_i8.py` (both
  arms, exact compare against `torch.gt`) compiles and is bit-exact. A 3 -> 2 pass output
  chain would be the largest remaining win on the Vector-bound bf16 cases.
- `r5_op2` -- **row `r_blk` cap 2** control (64 KB tile) for cross-task overlap.

| workload_id | base (current best) | r5_op1 | r5_op2 | A/B verdict (3 pairs) | L0 |
|---|---:|---:|---:|---|:--:|
| row-broadcast-bfloat16 | 34.04 | 34.12 | 44.20 | op1 vs base **tie** (+0.41 %); op2 vs base **+29.9 % slower** | pass |
| row-broadcast-float16 | 28.06 | - | 32.31 | op2 vs base **+15.5 % slower** | pass |
| cnn-feat-broadcast-float16 | 34.32 | 35.14 | - | op1 vs base **inconclusive** (bimodal 34.3/35.4 in both arms) | pass |

**Verdicts.** Both rejected.

- `r5_op1` (no gain, and the mechanism is now measured): the direct `i1 -> i8` cast does
  **not** reduce the tile footprint -- BiShengIR still materialises 2 B/elem for the cast
  output, proven by its own `ub overflow` report on the relaxed budget: `r_blk=8` on
  cnn-feat fp16 asks **200,832 B == 8 x 3136 x 8 B** and fp32 row `r_blk=4` asks
  **262,240 B == 4 x 4096 x 16 B**, i.e. exactly `2*cmp_bytes + 4` B/elem again. The
  backend pads the cast back to the same width, so both the pass count and the per-element
  cost are unchanged (A/B tie). The vcast doc table is nonetheless *wrong* (supported, just
  not profitable) -- recorded as a capability-note correction.
- `r5_op2` (hypothesis falsified again): halving the row tile costs +30 % (bf16) / +16 %
  (fp16). At 128 KB the row tile is already the optimum; the `vec_ratio 0.72` of bf16 is
  not an overlap artefact.

**Merge and non-regression check (this round's merged candidate = `r4_op1`).** Both
branches were re-measured on the merged file across every tune workload
(`phase=final`, `perf_records.jsonl` round 4, candidate `r4_final`): no class regresses
outside the measurement band, and the only r_blk change (cnn fp32 2 -> 4) is the A/B win.
Smoke `smoke-1k` fp32/fp16/bf16 re-ran inside the same `--level L0` pass (14/14) and in
the final `--level all` regression; all `max_diff = 0`.

### Round 6 (20261022, second session) -- column-broadcast group-major order + predicated GM scalar load

**Phenomenon (re-measured this session, not inherited).** Round 4/5 removed every
parameter lever; the remaining headline miss was cnn-feat fp16 (35.35 %). The eq/ge
siblings had just isolated a *separate* structural cost: at **identical** tile geometry
the column-broadcast class is 1.65× the row-broadcast class, and the whole delta is the
per-tile single-buffered `8 B GM -> UB copy -> T.vbrc -> T.vcmp` dependency chain
(PL-1.21 §11). Reproduced on this operator with `stall_probe.py --tag r6diag` (2 reps,
msprof op median-of-15) against the **round-4 deliverable**, same element count
(12.845 M), same `L = 3136`, same `r_blk = 4`:

| dtype | col (`cnn-feat-broadcast`) | row (`diag-row-3136`) | ratio | col mte2 active BW | row mte2 active BW |
|---|--:|--:|--:|--:|--:|
| fp16 | 34.32 | 21.84 | **1.57×** | 25.1 GB/s | 46.2 GB/s |
| bf16 | 35.22 | 29.25 | **1.20×** | 46.0 GB/s | 52.2 GB/s |

Same signature as eq (1.65×): identical bytes moved, but the col arm's MTE2 is *busy and
slow* (`mte2_ratio 0.74` with `vec_wait 0.71`), i.e. a serialized scalar path rather than
a bandwidth ceiling. `diag-row-3136-*` are timing-only diagnostic cases added to
`bench.py` (not manifest workloads, never written to `perf_records.jsonl`).

**Candidate optimisation point ②** (ported from the eq/ge fix, PL-1.21 §11 `r6op2`):
① enumerate tiles **broadcast-group major** and hand each core a *contiguous* block
(`task = cid * num_local_tasks + i`, `sub = task // R_lead`, `lead = task % R_lead`), so
one core walks `R_lead` consecutive tiles that share the same b row block; ② predicate
**only** the 8 B GM scalar load (`if i == 0 or lead == 0`) and leave the `T.vcast` and the
`T.vbrc` unconditional. The two forms the sibling had to reject were not re-run here
(they are a documented precondition, PL-1.21 §12): predicated `T.vbrc` is both wrong
across iterations and UB-doubling (cnn fp32 +50 KB -> `ub overflow`), predicated cast
breaks `L1/bool-vbrc`.

**Branch `r6_op1`** = round-4 deliverable + the above. Col path only: the row/1-D paths
keep their loop shape (`task_a = 1`, `task_b = num_kernels`); the row path's only textual
change is the `(sub, lead)` tie-break feeding `row0`, which is still a bijection onto the
same tile set.

**Candidate vs current best** (interleaved A/B, 2 pairs per arm, alternating order,
`profiles/candidate/r6ab6/`, JSON rows with `rep` in `perf_records.jsonl`):

| workload_id | current best (us) | r6_op1 (us) | Δ | mte2 | vec | vec_wait | GM→UB (KB) | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|--:|:--:|
| cnn-feat-broadcast-float16 | 34.36 | **23.42** | **+31.8 %** | 0.741→0.600 | 0.325→0.503 | 0.706→0.544 | 640.25→637.31 | pass |
| cnn-feat-broadcast-bfloat16 | 35.40 | **31.93** | **+9.8 %** | 0.396→0.385 | 0.611→0.682 | 0.413→0.345 | 640.25→637.31 | pass |
| cnn-feat-broadcast-float32 | 40.40 | **38.85** | **+3.8 %** | 0.613→0.619 | 0.454→0.443 | 0.570→0.579 | 1277.25→1274.31 | pass |

`gm_to_ub_datas` falls by 2.94 KB on cnn (fp16/bf16: 640.25→637.31, i.e. −0.46 %): the
b re-reads that disappeared. The 11 µs band on fp16 is MTE2 stall removal, not traffic.

**Non-regression A/B** (same protocol; the row and 1-D paths must stay put):

| workload_id | current best (us) | r6_op1 (us) | Δ | L0 |
|---|--:|--:|--:|:--:|
| row-broadcast-float16 | 27.18 | 27.17 | +0.0 % (tie) | pass |
| row-broadcast-bfloat16 | 34.13 | 34.20 | −0.2 % (tie) | pass |
| row-broadcast-float32 | 53.43 | 52.67 | +1.4 % (tie) | pass |
| hidden-state-prefill-float16 | 25.64 | 26.28 | −2.5 % (noise band) | pass |
| hidden-state-prefill-bfloat16 | 27.59 | 27.86 | −1.0 % (tie) | pass |
| hidden-state-prefill-float32 | 49.46 | 49.33 | +0.3 % (tie) | pass |

The hidden rows use the round-4 deliverable as arm A (`gt_fwd_kernel_round4_op1.py`,
byte-identical to the pre-round `GtFwdOp.py`); they are the control proving the 1-D path
was not touched — the `phase=final` hidden numbers move by −4.3 %/+2.3 %/−2.9 % purely
from session state.

**Verdict: ADOPT.** All three cnn cases improve (fp16 +31.8 %, bf16 +9.8 %, fp32 +3.8 %,
the last one ≥3 % but read as a tie-band win on a case that is already at 71.8 %), and no
row/1-D case regresses beyond the noise band. `perf_opt/GtFwdOp.py` := `r6_op1`
(sha256 `5eb4e9bd…`), then re-measured on every tune workload (`phase=final`, round 6,
candidate `r6_final`) plus smoke precision.

**Bimodality caveat (read with the table).** The cnn-feat class is bimodal on this
machine (documented in round 4/5: 34.3 / 40.4 / 44.5 µs for the same binary). This
session's `cur` arm measured 34.3–34.4 µs at fp16 (fast state); the Le sister arm in the
*next* session measured the identical binary at 43.8–45.9 µs. `r6_op1` measured
23.0–23.6 µs in both sessions and in the final collection, i.e. it is stable, and the
verdict holds against both states. No accept/reject decision above uses a cross-session
absolute.

### Autotune Log

No `T.Pipelined`/autotune search was used: this is a single-pass pure-Vector elementwise op with
one execution knob (`block_size`) that folds to a compile-time constant, and the decisive gains
were structural (hoist / 2-D tile), not a parameter walk. `block_size` was set by a small manual
per-dtype scan (round 1); all other candidates were explicit single-point branches.

## Final Performance Test Data

Deliverable `perf_opt/GtFwdOp.py` (candidate `r6_final` = `r6_op1`, sha256
`5eb4e9bd7c0661af…`), all 9 tune workloads, `phase=final` round 6, one msprof run each
on that single file in this session. Both columns come from this same
`perf_records.jsonl` (`phase=baseline` block_size=1024 / `phase=final`). The round-4
`phase=final` rows are demoted to `phase=merged` with a `note` when this set was
collected (data unchanged; pre-round text in
`history_version/20261022_pre_r6/opt_log.md`).

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | hidden-state-prefill-float16 | 83.078 | 25.420 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.198 | 27.960 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | hidden-state-prefill-float32 | 94.398 | 48.900 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | cnn-feat-broadcast-float16 | 511.890 | 23.200 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 578.868 | 31.840 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | cnn-feat-broadcast-float32 | 345.593 | 38.400 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | row-broadcast-float32 | 177.496 | 51.900 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | row-broadcast-float16 | 183.536 | 27.840 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py::gt_fwd_kernel::main | row-broadcast-bfloat16 | 177.316 | 34.140 | r6_final |

### Roofline Ratio (report-口径, `roofline_ratio.py`, BW = 1.8 TB/s, GM-read numerator)

| workload_id | baseline us | final us | improvement | ratio before (r4) | ratio now | >=60 %? | raw final profile |
|---|---:|---:|---:|---:|---:|:--:|---|
| hidden-state-prefill-float16 | 83.08 | 25.42 | +69.4 % | 76.59 % | **73.33 %** | yes | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-float16_f48f537e/OPPROF_20261022170002_YNTSXDJKPTXKSOVH/main` |
| hidden-state-prefill-bfloat16 | 91.20 | 27.96 | +69.3 % | 68.18 % | **66.67 %** | yes | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-bfloat16_274d79ca/OPPROF_20261022170027_DQULIGFQFADTMQNQ/main` |
| hidden-state-prefill-float32 | 94.40 | 48.90 | +48.2 % | 78.42 % | **76.24 %** | yes | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-float32_dac1dbdc/OPPROF_20261022170052_OPGLLUIDCOFGFHJP/main` |
| cnn-feat-broadcast-float16 | 511.89 | 23.20 | +95.5 % | 35.35 % | **61.52 %** | **yes (new)** | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-float16_f1cddcf9/OPPROF_20261022170116_YNYOMUTBQIUFVSAX/main2` |
| cnn-feat-broadcast-bfloat16 | 578.87 | 31.84 | +94.5 % | 36.48 % | **44.83 %** | no | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-bfloat16_f5a51caa/OPPROF_20261022170138_RHSOIYMPXWKQEBNK/main2` |
| cnn-feat-broadcast-float32 | 345.59 | 38.40 | +88.9 % | 71.79 % | **74.34 %** | yes | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-float32_2ff59659/OPPROF_20261022170201_ZQXYKRRXJBDFRCEV/main2` |
| row-broadcast-float32 | 177.50 | 51.90 | +70.8 % | 71.96 % | **71.85 %** | yes | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-float32_7e902357/OPPROF_20261022170225_MCMMBKJKFSVBWCJJ/main2` |
| row-broadcast-float16 | 183.54 | 27.84 | +84.8 % | 68.25 % | **66.98 %** | yes | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-float16_410291d8/OPPROF_20261022170248_SHCREGVPEYHKIEIJ/main2` |
| row-broadcast-bfloat16 | 177.32 | 34.14 | +80.7 % | 54.33 % | **54.62 %** | no | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-bfloat16_00d04600/OPPROF_20261022170312_CHVDAZJWMOPXKRLY/main2` |

**>= 60 % target: 7 / 9 tune workloads** (was 6/9). **Newly passing: cnn-feat fp16,
35.35 % -> 61.52 %** (lever ②, the col-broadcast stall). The two remaining misses are
cnn-feat bf16 (44.83 %, was 36.48 %) and row-broadcast bf16 (54.62 %, unchanged) — both
purely Vector-bound on bf16 `vcmp`/`vcast` throughput, analysed below. The hidden-class
deltas (73.33 vs 76.59 etc.) are session state on an **unmodified** 1-D branch; the
interleaved control A/B above measures them as ties (−2.5 %/−1.0 %/+0.3 %).


**Measurement-state note (must be read with the table).** `cnn-feat` is bimodal on this
machine: the *same* binary measures 34.3 / 40.4 / 44.5 us depending on session state
(this session even measured the two sisters' identical binaries at 34.3 us in one arm and
43.9-45.9 us in the next). The final `cnn` rows above are single msprof runs, while every
accept/reject decision in rounds 4/5/6 was made with the interleaved A/B protocol
(alternating order, even pair count; `profiles/ab/round4|round5/` and
`profiles/candidate/r6ab6|r6hid6/`, per-rep rows in `perf_records.jsonl`). Where the two
disagree the A/B verdict is authoritative. `r6_op1` is the first candidate on this
operator that measured inside a narrow band (23.0-23.6 us) in *every* session and arm --
consistent with the sibling finding that removing the col scalar stall also removes the
slow bimodal state.


## Smoke precision and regression

- Smoke `smoke-1k` fp32/fp16/bf16 — precision-only (per the skill, smoke is never
  performance-tuned); no perf profile recorded. `perf_opt/workload_inventory.json`
  `precision_pass: true` for all three.
- Phase-3 regression on the deliverable `perf_opt/GtFwdOp.py` (round 6, sha `5eb4e9bd…`):
  `python3 perf_opt/GtFwdOp.py --level all` → **All check passed** (L0 fourteen items,
  L1 functional matrix + UB-guard/dtype-reject + `bool-vbrc` int8-carrier col broadcast,
  L2, boundary; `Summary max_diff per dtype: all 0`). Log: `perf_opt/logs/r6/final_all.log`.
  The same `--level L0` gate ran on every round-4/5/6 candidate (`logs/r4/L0_*.log`,
  `logs/r5/L0_*.log`, `logs/r6/L0_gt_r6_op1.log`) and on the final file inside
  `run_experiments.py --phase final` (`logs/r6/`), and the three smoke cases were re-checked
  bit-exactly with `bench.py --check` (`logs/r6/smoke_check.log`).

## Why >= 60 % is not reached for the remaining 2 workloads

The report ratio numerator is **GM read traffic only** (`read_cache_miss_allocate x 128`), while
every gt case also writes the 1-byte bool carrier, so the ceiling is
`read/(read+write) x B_eff/1.8`. For the fp16/bf16 classes `read = a`, `write = N` -> read
fraction 2/3 -> practical ceiling ~ `0.667 x B_eff/1.8`. With the best total bandwidth observed
on this device ~ 1.67 TB/s the 2/3-read ceiling is ~ 62 %; cnn-feat fp16 now sits at 61.52 %,
i.e. essentially *at* that ceiling.

Round 6 closed the fp16 case by removing the column-broadcast stall (`r6_op1`, +31.8 % A/B).
Both remaining misses are bfloat16, and both are Vector-throughput bound, not schedule bound:

- **cnn-feat bf16 (44.83 %, was 36.48 %)** -- the same lever gave it only +9.8 % (35.40 ->
  31.93 us A/B) because its schedule was already partly hiding the scalar chain
  (`mte2_ratio` 0.40 vs fp16 0.74), and it now runs at `vec_ratio 0.682` with
  `mte2 active BW 52.7 GB/s`. bf16 Vector ops on this part are ~1.85x slower than fp16 for
  identical work (round 4 measurement), so the residual gap is `vcmp`/`vcast` throughput on
  the same 4-pass chain. The two pass-reduction routes are closed by measurement: `i1 -> i8`
  direct cast is supported and exact but the backend re-materialises 2 B/elem (round 5, A/B
  tie), and dropping the broadcast materialisation is blocked by CG-2026-0021 (`T.vcmp` has
  no broadcast) -- an open compiler capability, not a design choice.
- **row-broadcast bf16 (54.62 %, unchanged)** -- purely Vector-bound (`vec_ratio 0.725` vs
  fp16 0.485 at the same bytes and the same 3 passes). Round 6 leaves it untouched by
  construction (row path), and the A/B control confirms the tie (34.13 -> 34.20 us, -0.2 %).
  Half tile was falsified in round 5 (+30 %), `r_blk=8` is UB-impossible
  (8 x 4096 x 8 B = 256 KB > 192 KB), and row fp16 on the same schedule reaches 66.98 %,
  isolating the gap to bf16 Vector throughput.

**Why this is still not `[DESIGN_LIMIT]`.** Per `_shared/standards/perf-feedback.md` §1 the
shortfall is a report-metric (read-only numerator) + hardware-cost (bf16 Vector throughput,
mandatory broadcast materialisation) boundary; it does not contradict a specific DESIGN.md
assumption, and no structural >2x unreachable acceleration is demonstrated (the largest
conceivable change -- removing the broadcast pass -- is blocked by CG-2026-0021, an open
compiler capability gap). No `perf_feedback.md` is produced.

## Skill Retrospective

- **Process rule (R, strong) — Phase 3 must materialise `opt_log.md` + inventory `tuning_status`
  + the Final table in the same session as the `phase=final` collection.** The earlier session
  measured every round and even wrote 3 `phase=final` records, then stopped without the log /
  `tuning_status`; on resume the gate failed (`S4-OPTLOG`, `S4-OPTLOG-COMPTABLE`,
  `S4-WORKLOAD-INVENTORY`, `S4-PERF-RECORDS-RECON`) despite complete profiles. This is the
  second occurrence of the same orphaning failure (identical to the `ge` retrospective) →
  `recurring`.
  - evidence: gate-4 failure list on resume; `ge` opt_log Skill Retrospective.
  - repro: n/a (process rule).
  - toolchain_stamp: tilelang `0.1.2+302ad3c`, Ascend910 40-Vector-core, CANN 9.0.0.
- **Resume discipline (R, weak) — append-only `perf_records.jsonl` + "exactly one `phase=final`
  per workload" requires a **workload-subset** runner entry.** The `--all`-only final path cannot
  top up the workloads still missing a final record without duplicating the ones already done;
  fixed by adding `--workloads <csv>` to `run_experiments.py`.
  - evidence: this session's `logs/final_round3_remaining.log`; gate rule `S4-PERF-RECORDS-RECON`
    ("必须恰有一条当前最终版本的 final 记录").
  - repro: `perf_opt/run_experiments.py` `--workloads`.
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive) — DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D tile +
  dtype-aware UB budget for the broadcast classes.** Decisive: cnn fp16 177→34.5 µs,
  cnn bf16 153→46.7 µs, cnn fp32 95→44.6 µs, row fp16 51→32.4 µs, row bf16 79→44.0 µs
  (round 2). Mechanism: the per-task emit chain (`vcmp` + 2 `vcast`, plus the per-task
  broadcast load + `vbrc`) is amortised over `R_blk` rows; `T.vbrc` materialises the broadcast
  operand because `T.vcmp` has no auto-broadcast (CG-2026-0021).
  - evidence: round-2 "Candidate vs current best" table; `profiles/candidate/round2/r2_op2_*`.
  - repro: `perf_opt/gt_fwd_kernel_round2_op2.py` (`main2` block + dtype-aware budget).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive) — outer-broadcast-operand hoist.** Materialise a fully
  outer-broadcast operand once per core (not once per task) → row-broadcast fp32 177→63.3 µs,
  fp16 184→51.4 µs, bf16 177→79.5 µs (round-1 `r1_op2`).
  - evidence: round-1 "Candidate vs current best" table; `profiles/candidate/round1/r1_op2_*`.
  - repro: `perf_opt/gt_fwd_kernel_round1_op2.py` (`hoist_a`/`hoist_b`).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive) — relaxed (1.5×) 2-D UB inflation for the native
  fp16/bf16 path.** The staging copy lands directly in `a2` (no separate staging buffer), so the
  real footprint is `2·E + 4` B/elem; relaxing 1.7×→1.5× admits `R_blk=4` at L=4096
  (row-broadcast fp16 32.4→28.5 µs, bf16 44.0→34.4 µs; round-3 `r3_op1`).
  - evidence: round-3 table; `profiles/candidate/round3/r3_op1_*`.
  - repro: `perf_opt/gt_fwd_kernel_round3_op1.py` (`_infl_2d`).
  - toolchain_stamp: same as above.
- **Measurement caveat (D) — same-file cross-run spread on the small broadcast kernels is
  ~10–20 %.** Evidence: `r2_op2` and `r3_op1` produce *identical* `R_blk` on cnn-feat L=3136
  (both 4), yet cnn fp16 measured 34.5 µs (r2_op2) vs 41.1 µs (r3_op1) vs 46.0 µs (`final`);
  cnn fp32 `final` straddles the 60 % line (44.9 µs → 63.5 % earlier vs 48.9 µs → 58.4 % here).
  Consequence: per-workload winners among near-equal candidates on these kernels are within
  noise and should not override the priority-class decision; the ≥60 % verdict on cnn fp32 is
  reported honestly as borderline rather than claimed either way.
  - evidence: perf_records `candidate`/`final` rows for cnn-feat fp16/fp32; `roofline_ratio.py`.
  - repro: re-run `bench.py` on the same file (n/a minimal).
  - toolchain_stamp: same as above.

- **D (measured constant, strong) -- 2-D `[r_blk,L]` tile footprint and the right budget
  formula (this round).** The tile is exactly `2*cmp_bytes + 4` B/elem
  (`a2 + b2 + cond` + the 2 B/elem output-cast intermediate), and the shipped blanket
  1.5x/1.7x "auto-multi-buffer" divisor was the wrong knob: replacing it with
  `(UB - staging - reserve) / (2*cmp_bytes + 4)`, staging = the one live `b_in` (1,L) carrier,
  gives cnn-feat **fp32 `r_blk` 2 -> 4 at 150,528 B (77 % of UB)** and a consistent
  **-8.2 % A/B win** (43.56 -> 39.59 us), while every fp16/bf16 class keeps a byte-identical
  kernel. Independent compiler cross-check: at the relaxed budget BiShengIR demands 200,832 B
  for `r_blk=8` (== 8 x 3136 x 8 B) and 262,240 B for fp32 row `r_blk=4`
  (== 4 x 4096 x 16 B) -- exactly 2E+4 again.
  evidence: `perf_records.jsonl` round 4 (`r3_op1_ref` vs `r4_op1`),
  `logs/r4/ab_op1_cnn-feat-broadcast-float32.json`, compiler messages in
  `logs/r5/L0_r5_op1.log`.
- **D (measured, overturns a doc-bound negative) -- `T.vcast(bool -> int8)` DOES exist on this
  toolchain.** `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 lists only
  `i1 -> f16/f32`, but `perf_opt/probe_cast_i1_i8.py` compiles and is bit-exact on tilelang
  `0.1.2+a79a1316`. It is however **not profitable**: the backend re-materialises the same
  2 B/elem intermediate (identical `ub overflow` footprint to the explicit f16 hop) and the A/B
  is a tie (row bf16 34.04 vs 34.12 us). Treat the doc table as *incomplete but
  cost-equivalent*; replace the old "no i1->i8 pair" claim with this one.
  evidence: `probe_cast_i1_i8.py` (both arms exact), `logs/r5/ab_r5op1_*.json`.
- **D (compiler trap, strong) -- a UB->UB `T.copy` whose source is a runtime row-offset UB slice
  scalarises.** `T.copy(b_all[sub*r_blk : sub*r_blk+r_blk, 0:1], b_in)` compiles, passes L0
  (exact) and costs **88x** (cnn fp16 40.9 -> 3619.4 us), so it cannot be used to hoist a tiny
  broadcast vector into UB. Correct-but-slow: no precision gate catches it, only the A/B
  measurement does.
  evidence: `logs/r4/ab_op3_*.json`; the 3-pair merged median is recorded in
  `perf_records.jsonl` round 4 (`candidate_id=r4_op3`, `duration_us` = 3619.4 / 3629.5 us).
- **P (pattern, negative) -- the `r_blk` ladder above the divisor-4 plateau is exhausted for this
  family.** Seven independent branches in rounds 4/5 (inflation 1.25/1.0, masked-tail `r_blk` 6
  and row 5, col cap 2, row cap 2, per-core broadcast preload, direct i1->i8) produced **zero**
  A/B wins on cnn-fp16 / cnn-bf16 / row-bf16. The only accepts were `r_blk` 2 -> 4 (fp32) and
  the footprint-derived budget. Third reproduction of PL-1.21's "tile size is monotone, fill UB,
  then no parameter is left" law, and the first that also falsifies the *inflation-factor*
  sub-hypothesis at 1.0.
  evidence: `opt_log.md` rounds 4/5 tables + `logs/r4|r5/ab_*.json`.
- **R (process, weak) -- `ab_test.py` needs a pre-created 0700 `--workdir` subtree.** msprof
  refuses a group-writable output dir; `ab_test.py` creates its `pair{N}{arm}` dirs with the
  default umask, so the first run of every A/B dies with "is writable by any other users or group
  users". Pre-create `pair1A..pair3B` with `mkdir -m 700`, or patch the tool.
  evidence: `logs/r4/ab_op2_cnn-feat-broadcast-float16.err` (first failed attempt) vs the JSON of
  the same name after the fix.
- **P (pattern, positive, cross-operator replicated) -- the col-broadcast stall is a *separate*
  structural lever from the tile ladder, and it is the same fix on all four sibling operators.**
  Measured here at identical geometry (`stall_probe.py --tag r6diag`, same 12.845 M elements,
  same `L=3136`, same `r_blk=4`): col 34.32 us vs row 21.84 us (fp16, 1.57x; mte2 active BW
  25.1 vs 46.2 GB/s) and 35.22 vs 29.25 us (bf16). Fix = broadcast-group-major task order
  (`task = cid*num_local_tasks + i`, `sub = task // R_lead`, `lead = task % R_lead`) so a core
  walks `R_lead` tiles sharing one b row block, **plus** predicating *only* the 8 B GM scalar
  load (`if i == 0 or lead == 0`), with `T.vcast`/`T.vbrc` unconditional. A/B 2 pairs, alternating:
  cnn fp16 34.36 -> 23.42 us (+31.8 %, ratio 35.35 % -> 61.52 %), bf16 35.40 -> 31.93 (+9.8 %),
  fp32 40.40 -> 38.85 (+3.8 %); row/1-D control ties (-0.2 % .. +1.4 %, -2.5 % .. +0.3 %).
  Origin: PL-1.21 §11 (eq/ge), now replicated on gt (this round) and le (same session).
  evidence: `perf_records.jsonl` round 6 (`r6` arm, per-rep), `profiles/candidate/r6ab6*`.
  repro: `perf_opt/stall_probe.py` (diagnosis) + `perf_opt/gt_fwd_kernel_round6_op1.py` (fix).
  toolchain_stamp: tilelang 0.1.2+a79a1316 / CANN 9.0.0 / Ascend910 40-vector-core.
- **D (measured, strong) -- the cnn-feat bimodality disappears once the col scalar chain is
  predicated away.** The same binary swings 34.3 / 40.4 / 44.5 us across sessions (this session:
  gt at 34.3-34.4 us, the le sister's identical binary at 43.8-45.9 us 30 min later), but
  `r6_op1` measured 23.0-23.6 us in *every* arm, session and the final collection. So the slow
  bimodal state is the *stall amplified*, not generic machine noise, and a candidate that only
  wins in the fast state should be treated as unproven.
  evidence: `perf_records.jsonl` round 6 `cur` vs `r6` arms (incl. `aiv_mte2_active_bw` 17-18
  GB/s for the le slow state vs 48 GB/s for `r6`).
- **R (process, weak, second occurrence) -- a final-record collection silently makes the previous
  `phase=final` rows invalid, and the only gate-compatible repair is a phase demotion.** After
  changing `perf_opt/GtFwdOp.py`, the old 9 `phase=final` rows fail both the "exactly one final
  per workload" count and the artifact-hash check. This round therefore re-labelled them to
  `phase=merged` with a `note` (durations/hashes untouched) exactly as round 4 did. Doing this
  *before* the new collection (and archiving `opt_log.md` to `history_version/<date>/`) keeps the
  append-only record honest and the gate green.
  evidence: `perf_records.jsonl` rows with `note = "relabelled to merged ..."`; gate-4 run.
