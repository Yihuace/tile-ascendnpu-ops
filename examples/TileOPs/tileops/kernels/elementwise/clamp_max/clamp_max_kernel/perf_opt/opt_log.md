# _make_clamp_tensor_kernel Stage 4 Tuning Log

- operator: `_make_clamp_tensor_kernel` (`ClampMaxFwdOp`, elementwise `y = min(x, hi)`, broadcasting bound)
- project: `clamp_max`; scenario `optimize`; mode `full`
- baseline kernel (read-only): `clamp_max_kernel/_make_clamp_tensor_kernel.py`
- deliverable: `perf_opt/_make_clamp_tensor_kernel.py` (candidate `final` = `r1_bs8192`)
- benchmark source: `examples/TileOPs/benchmarks/ops/bench_clamp_max.py::test_clamp_max_bench`
- **primary metric**: `msprof op` `Task Duration(us)` (kernel-only, median of a `--launch-count=15 / --warm-up=5` run). No other timing basis is used.
- Roofline Ratio (report口径, BW = 1.8 TB/s):
  `ratio = (Σ aiv/aic r0+r1 read_cache_miss_allocate × 128) / (Task Duration(s) × 1.8e12)` — re-derived from the same `L2Cache_*`/`OpBasicInfo_*` CSVs the single-operator report reads, by `roofline_ratio.py`.
- toolchain: `tilelang 0.1.2+b9ece659`, Ascend910 (20 AI cores → **40 Vector cores**, `UB = 192 KB`), `TILELANG_ASCEND_MODE=Developer`, CANN 9.0.0.
- Stage-4 target: **≥ 60 %** Roofline Ratio per case ("reach the target, not the optimum; stop once reached").
- Quick verdict: **7 / 9 tune workloads ≥ 60 %**; the 3 fp32 cases are all improved (cnn-feat fp32 crosses from 57.4 % to 62.7 %), the two remaining fp32 16M cases sit ~1.5–2.5 % short of 60 % in this session (see "Why ≥60 % is not reached for 2 fp32 workloads").

## Path fidelity (which kernel the benchmark actually dispatches)

`ClampMaxFwdOp._eager_forward` broadcasts **both** operands with `_expand_flat`
(`expand` + `contiguous` + `view(-1)`) and calls the frozen factory as
`_make_clamp_tensor_kernel(N_total, dtype, has_min, has_max)` — i.e. with
`coalesced_shape=None`. Therefore **every** benchmark case (including the two
manifest broadcast labels) enters the kernel's **same-shape flat path**: the
kernel reads two full-length operands and writes one (`3·N·E` bytes) and the
DESIGN §0.6 R3 in-kernel broadcast path is used only by this module's own
L0-2/L1 cases. `perf_opt/bench.py` reproduces exactly that call form.

## Performance Test Data (baseline)

Workload inventory: `perf_opt/workload_inventory.json`, expanded from
`bench_clamp_max.py::test_clamp_max_bench` (4 manifest entries × 3 dtypes = 12
cases): **9 `tune`** + **3 `smoke`** (`smoke-1k` = manifest idx 0 fp32 plus its
fp16/bf16 siblings — precision-only; the benchmark's `full` mark on those two
conflicts with the independent smoke label and is recorded in the inventory
`reason`). `skipped`: none.

Baseline = the Stage-3 kernel (`../_make_clamp_tensor_kernel.py`, sha
`2fce52e4…`) at its shipped K11 `block_size` (2048 fp32 / 4096 fp16,bf16),
`phase=baseline` round 0. All 9 tune workloads, `msprof op` median of 15 launches:

| workload_id | dtype | shape (input \| max) | N_total | Task Duration(us) | mte2 / mte3 / vec / scalar | Roofline Ratio |
|---|---|---|---:|---:|---:|---:|
| elementwise-16M-float16 | fp16 | [4096,4096] \| [4096,4096] | 16.78 M | 58.039 | .967 / .324 / .095 / .063 | 64.24 % |
| elementwise-16M-bfloat16 | bf16 | [4096,4096] \| [4096,4096] | 16.78 M | 59.319 | .906 / .357 / .502 / .088 | 62.85 % |
| elementwise-16M-float32 | fp32 | [4096,4096] \| [4096,4096] | 16.78 M | 136.117 | .974 / .282 / .077 / .047 | 54.78 % |
| cnn-feat-broadcast-float16 | fp16 | [16,256,56,56] \| [256,1,1] | 12.85 M | 43.219 | .962 / .342 / .101 / .073 | 66.05 % |
| cnn-feat-broadcast-bfloat16 | bf16 | [16,256,56,56] \| [256,1,1] | 12.85 M | 43.179 | .884 / .375 / .541 / .097 | 66.11 % |
| cnn-feat-broadcast-float32 | fp32 | [16,256,56,56] \| [256,1,1] | 12.85 M | 99.478 | .971 / .287 / .080 / .052 | 57.39 % |
| row-broadcast-float32 | fp32 | [4096,4096] \| [1,4096] | 16.78 M | 137.377 | .973 / .280 / .077 / .047 | 54.28 % |
| row-broadcast-float16 | fp16 | [4096,4096] \| [1,4096] | 16.78 M | 59.839 | .966 / .315 / .093 / .063 | 62.31 % |
| row-broadcast-bfloat16 | bf16 | [4096,4096] \| [1,4096] | 16.78 M | 56.039 | .895 / .368 / .531 / .092 | 66.53 % |

Raw profiles: `profiles/baseline/round0/baseline_8ba8496a/<workload_id>_<hash>[/main]/OPPROF_*/`.

Every row is `mte2_ratio ≈ 0.89–0.97` with `vec_ratio ≤ 0.10` (bf16 ≤ 0.55, see
below): the kernel is **MTE2/HBM-bandwidth-bound**, not emission- or
compute-bound. Achieved total traffic bandwidth
(`3·N·E / duration`) is ≈ 1.74 TB/s for the fp16 classes but only ≈ 1.48–1.55
TB/s for the fp32 classes — the fp32 16M streams are the only ones under target.

**Interpretation of the ratio metric.** The numerator counts **GM read misses
only** (`read_cache_miss_allocate × 128`), so for a same-shape 3-stream kernel
`ratio = ⅔ · (achieved_total_BW / 1.8 TB/s)`; the 60 % target is therefore
equivalent to `achieved_total_BW ≥ 1.62 TB/s`. Measured fp16 classes already run
at ≈ 1.74 TB/s (≈ 64–67 %), cnn-feat fp32 at ≈ 1.74 TB/s, but the fp32 **16M**
streams only at ≈ 1.56 TB/s — a device bandwidth degradation with footprint, not
a tiling artifact (see round 1/4 and the closing analysis).

### Design estimate vs measured (D-2 deviation, DESIGN.md §1.6.0)

DESIGN §1.6.0 estimated the flow-dominated lower bound at `3·N·E ÷ 1.05 TB/s`
(40/48-core-scaled `CONST-mte2-degradation`), with an emission term of
`~0.5 µs/vmin` per task:

| representative | design bound | measured baseline | deviation |
|---|---:|---:|---:|
| elementwise-16M fp16 | ≈ 96 µs | 58.04 µs | 0.60× (estimate conservative) |
| elementwise-16M fp32 | ≈ 192 µs | 136.12 µs | 0.71× (estimate conservative) |
| cnn-feat fp16 | ≈ 51 µs (in-kernel broadcast) | 43.22 µs | 0.85× |
| row-broadcast fp16 | ≈ 64 µs | 59.84 µs | 0.94× |

The estimate is conservative in every case (< 2× in the other direction), so it
does **not** mis-attribute the bottleneck: it correctly identified a
flow/bandwidth-dominated bound. The residual is that the design's 1.05 TB/s
effective-bandwidth constant (scaled from a 48-core/256 M measurement) understates
this device's 16M achievable bandwidth (~1.7 TB/s mixed R/W). No
traffic/emission/capacity term was mis-valued.

## Iteration Log

Each round starts from the current best; **every comparison table below is
candidate vs current best**, all durations from `msprof op` (median of 15
launches). Only the fp32 workloads were re-measured per branch (the fp16/bf16
classes are identical in structure and unchanged by a fp32-only parameter — they
were re-measured at merge and final). L0 = the branch file's own `--level L0`
suite (all branches pass; no rollback).

### Round 1 — base = `baseline` (fp32 block_size 2048); per-dtype block_size scan

| branch | optimization point | ew16M fp32 µs | cnn-feat fp32 µs | row fp32 µs | L0 |
|---|---|---:|---:|---:|:--:|
| `baseline` | shipped K11 (fp32 2048) | 136.117 | 99.478 | 137.377 | pass |
| `r1_bs4096` | fp32 block_size 4096 | 129.817 | 90.418 | 130.677 | pass |
| `r1_bs6144` | fp32 block_size 6144 | 134.137 | 89.638 | 132.937 | pass |
| `r1_bs8192` | fp32 block_size 8192 | **128.877** | **88.738** | **127.677** | pass |
| `r1_bs9216` | fp32 block_size 9216 (UB cap 9637) | 131.317 | 92.538 | 134.497 | pass |

- **Observation**: the fp32 stream is MTE2-saturated (`mte2 0.98`), and one
  32 KB contiguous tile per operand beats the shipped 8 KB tile on all three
  fp32 workloads; 6144/9216 are worse (non-monotonic ⇒ part of the ordering is
  run bimodality, see the A/B below).
- **Winner**: `r1_bs8192` (fp32 8192; fp16/bf16 keep 4096 — larger fp16 tiles
  are not needed and the fp16/bf16 classes already pass).
- **L0**: all branches pass. No rollback.

### Round 2 — base = `r1_bs8192`; drop `y_ub` (in-place `vmin`, 2 live UB buffers)

| branch | optimization point | ew16M fp32 | cnn-feat fp32 | row fp32 | L0 |
|---|---|---:|---:|---:|:--:|
| `r2_inplace_bs8192` | result written in place into `x_ub` (UB cap 14455) | 127.817 | 91.198 | 127.877 | pass |
| `r2_inplace_bs14336` | in-place + larger tile 14336 (2-buffer UB cap) | 131.397 | 89.998 | 130.257 | pass |

- **Observation**: releasing a third of the UB budget changes nothing —
  multi-buffer depth is not the limiter.
- **A/B adjudication** (`ab_test.py`, elementwise-16M-float32, 3 interleaved
  pairs): A=`r1_bs8192` merged median **130.097 µs**, B=`r2_inplace_bs8192`
  **128.757 µs**, `b_rel_change = -1.03 %`, `sign_test_p = 1.0`,
  `direction_consistent = false` → **verdict `tie`**.
- **Winner**: none — `r1_bs8192` retained (simpler structure, no `vmin`
  dst==src0 aliasing).

### Round 3 — base = `r1_bs8192`; DESIGN §1.6.3 alternative-#2 adjudication (rank-2 tile)

DESIGN §1.6.3 froze 主选 #1 (rank-1 `(block_size,)` tile) + 备选 #2 (rank-2
`[R_blk, block]` tile) with the rule "alternative latency < primary × 0.97 flips
it". The alternative was implemented as a rank-2 in-kernel re-tile of the flat
stream (`main2`: `x/hi/y` viewed as `(L/col, col)`, tile `[2, col]`,
`col = block_size/2`, byte-identical footprint, 2 DMA descriptors instead of 1).

| branch | optimization point | ew16M fp32 | row fp32 | ew16M fp16 | L0 |
|---|---|---:|---:|---:|:--:|
| `r3_2d` | rank-2 `[2, col]` tile (`col = block_size/2`) | 129.837 | 125.257 | 52.239 | pass |

- **Observation**: first reading suggested a large fp16 win (58.04 → 52.24 µs),
  but the same file's fp32 rows were −0.7 % / +1.9 %.
- **A/B adjudication** (`ab_test.py`, elementwise-16M-float16, 3 interleaved
  pairs): A=`r1_bs8192` **55.599 µs**, B=`r3_2d` **55.119 µs**,
  `b_rel_change = -0.86 %`, `sign_test_p = 1.0`, `direction_consistent = false`
  → **verdict `tie`**. The apparent fp16 win was run bimodality (the same file
  measured 52.2 µs in round 3 and 55.1 µs under the interleaved protocol).
- **Winner**: 主选 #1 retained by measurement (adjudication resolved: alternative
  #2 does **not** beat the primary by the 0.97 rule). Structurally this is
  expected: for the contiguous same-shape stream the integrated Op layer
  dispatches, a `[R_blk, col]` tile with contiguous row stride is a
  descriptor-rank-only re-formulation of the rank-1 tile of the same byte size —
  the round-1 `block_size` scan is the same axis.

### Round 4 — base = `r1_bs8192`; tile-height sweep of the alternative-#2 form (fp32 8192)

| branch | tile | ew16M fp32 | cnn-feat fp32 | row fp32 | L0 |
|---|---|---:|---:|---:|:--:|
| `r4_2d_r2` | `[2, 4096]` | 128.397 | 86.218 | 131.457 | pass |
| `r4_2d_r4` | `[4, 2048]` | 132.537 | 88.678 | 130.877 | pass |
| `r4_2d_r8` | `[8, 1024]` | 130.997 | 91.278 | 126.837 | pass |

- **Observation**: no tile height wins consistently; every spread is inside the
  ±3–5 % same-file cross-run band (e.g. `[2,4096]` measured 125.257 µs for
  row-broadcast in round 3 and 131.457 µs in round 4). `mte2_bw` stays at
  24.5–26.0 GB/s/core regardless of descriptor rank.
- **Winner**: none — #1 retained.

### Round 5 — base = `r1_bs8192`; static fast path and explicit 4-stream loads

| branch | optimization point | ew16M fp32 | cnn-feat fp32 | row fp32 | L0 |
|---|---|---:|---:|---:|:--:|
| `r5_exact` | exact/static full-tile fast path (no divmod, no `T.min` tail, no dynamic slice) | 130.337 | 89.198 | 129.357 | pass |
| `r5_4stream` | 4 explicit outstanding loads per task (2 x-halves + 2 hi-halves, in-place vmin) | 126.037 | 86.378 | 129.977 | pass |

- **Observation**: removing per-task scalar work is a tie/slight loss — expected,
  since `mte2_ratio ≥ 0.98` means scalar work is already hidden. The 4-stream
  variant raises `mte3_ratio` to 0.83–0.86 (more write/read interleave) without a
  bandwidth gain.
- **A/B adjudication** (`ab_test.py`, elementwise-16M-float32, 3 interleaved
  pairs): A=`r1_bs8192` **125.917 µs**, B=`r5_4stream` **128.737 µs**
  (`b_rel_change = +2.22 %`, `sign_test_p = 0.25`, `direction_consistent = true`)
  → **verdict `tie`** (B not better) → rejected.
- **Winner**: none — `r1_bs8192` retained.

### Round 6 — merge, non-regression and final measurement

`r1_bs8192` is the single merged change (per-dtype `block_size`); the delivered
`@tilelang.jit` factory body is **unchanged** from the Stage-3 kernel (verified
by `diff`: only the docstring, the new module-level `TUNED_BLOCK_BY_DTYPE`
constant and `_default_block_size` differ). All 9 tune workloads were re-measured
on the final file (`phase=final`, candidate `final`), all 3 smoke cases were
re-checked for precision, and the full `--level all` regression was re-run. No
`merge_blocked`; no workload regressed structurally.

## Autotune Log

No library autotune was used. The only tunable is `block_size`, set by a small
manual per-dtype scan (round 1: fp32 4096/6144/8192/9216; the UB budget guard
caps fp32 at 9637) plus a 3-point tile-height sweep of the round-3 rank-2 form
(round 4). The decisive change is the fp32 `block_size` (2048 → 8192); the
structural candidates (rounds 2–5) were all adjudicated as ties by the
interleaved A/B protocol and therefore rejected.

## Roofline Ratio (report口径, `roofline_ratio.py`, BW = 1.8 TB/s) and target

| workload_id | baseline µs | final µs | Δ Task Duration | baseline Ratio | final Ratio | ≥60 %? |
|---|---:|---:|---:|---:|---:|:--:|
| elementwise-16M-float16 | 58.039 | 56.459 | +2.7 % | 64.24 % | **66.04 %** | ✓ |
| elementwise-16M-bfloat16 | 59.319 | 55.199 | +6.9 % | 62.85 % | **67.54 %** | ✓ |
| elementwise-16M-float32 | 136.117 | 129.757 | +4.7 % | 54.78 % | **57.47 %** | ✗ |
| cnn-feat-broadcast-float16 | 43.219 | 42.359 | +2.0 % | 66.05 % | **67.39 %** | ✓ |
| cnn-feat-broadcast-bfloat16 | 43.179 | 42.339 | +1.9 % | 66.11 % | **67.42 %** | ✓ |
| cnn-feat-broadcast-float32 | 99.478 | 91.118 | +8.4 % | 57.39 % | **62.65 %** | ✓ |
| row-broadcast-float32 | 137.377 | 129.197 | +6.0 % | 54.28 % | **57.71 %** | ✗ |
| row-broadcast-float16 | 59.839 | 57.179 | +4.4 % | 62.31 % | **65.20 %** | ✓ |
| row-broadcast-bfloat16 | 56.039 | 57.379 | −2.4 % | 66.53 % | **64.98 %** | ✓ |

**≥ 60 % target: 7 / 9 tune workloads.** Reading the table correctly:

- Only the **fp32** classes changed configuration (`block_size` 2048 → 8192);
  the fp16/bf16 classes are byte-for-byte the same execution parameters as the
  baseline. Therefore the fp16/bf16 deltas above (+1.9 % … +6.9 %, −2.4 %) are
  **run-to-run bimodality of the same kernel** (±3–5 % on this device, BP
  `run_state_bimodality`), not a tuning effect — the fp16/bf16 cases already
  passed at baseline.
- The **fp32** deltas (+4.7 % / +6.0 % / +8.4 %) are single-run medians; because
  they are the only real change they were re-verified with the interleaved A/B
  protocol (below), which reports **−10.5 %** (elementwise-16M-float32) and
  **−8.5 %** (row-broadcast-float32), `verdict = B_better`,
  `direction_consistent = true` — the fp32 `block_size` win is protocol-backed.
- The `final` column ratios use the same run as the Final table; across the
  session the two short fp32 cases oscillate between ≈ 57.5 % and ≈ 59.2 %
  (A/B medians: elementwise-16M-float32 125.9–126.3 µs ⇒ 59.0–59.2 %;
  row-broadcast-float32 126.3–128.0 µs ⇒ 58.3–59.0 %).

### A/B protocol evidence (interleaved A/B/A/B, `msprof op`, 3 pairs each)

| workload | A (baseline) | B (final) | Δ | verdict |
|---|---:|---:|---:|---|
| elementwise-16M-float32 | 141.157 µs | 126.337 µs | **−10.5 %** | `B_better` |
| row-broadcast-float32 | 139.857 µs | 127.957 µs | **−8.5 %** | `B_better` |
| elementwise-16M-float16 (2-D candidate vs #1) | 55.599 µs | 55.119 µs | −0.9 % | `tie` |
| elementwise-16M-float32 (in-place candidate vs #1) | 130.097 µs | 128.757 µs | −1.0 % | `tie` |
| elementwise-16M-float32 (4-stream candidate vs #1) | 125.917 µs | 128.737 µs | +2.2 % | `tie` |

Raw A/B artefacts: `logs/ab/ab_final_ew16M_fp32`, `logs/ab/ab_final_row_fp32`,
`logs/ab/ab_r4_ew16M_fp16`, `logs/ab/ab_r2_ew16M_fp32`, `logs/ab/ab_r5_4stream_ew_fp32`.

## Smoke precision and regression

- Smoke `smoke-1k` fp32/fp16/bf16 — precision-only (per the skill, smoke is
  never performance-tuned); no perf profile recorded. `bench.py --check` on the
  final file: all three `CHECK PASSED` (`logs/verify/smoke_precision.log`);
  inventory `precision_pass: true`.
- Phase-3 regression on `perf_opt/_make_clamp_tensor_kernel.py`:
  `python _make_clamp_tensor_kernel.py --level all` → exit 0,
  **`[L0] ALL PASS (eight items)`, `[L1] ALL PASS`, `[L2]`/`[Boundary]` warn-only
  clean, `Summary max_diff per dtype: bfloat16=0.0, float16=0.0, float32=0.0`**
  (`logs/verify/final_all.log`). L0/L1 include the in-kernel broadcast path
  (L0-2/L0-6 cnn-feat / row-broadcast / scalar bound), NaN/±Inf/±0/subnormal
  corners, non-divisible tails, the persistent-split boundary and the UB-guard /
  dtype-reject cases, so the unchanged R3 structure stays validated.

## Why ≥60 % is not reached for 2 fp32 workloads

`elementwise-16M-float32` (57.47 %) and `row-broadcast-float32` (57.71 %) are the
only cases below target, and they share the same N = 16.78 M fp32 stream:

1. **The kernel is at the device MTE2/HBM limit for this footprint.**
   `mte2_ratio = 0.99`, per-core `mte2_active_bw = 25.2 GB/s` (40 cores ⇒ total
   ≈ 1.56 TB/s for the 3·N·E = 201 MB of traffic), i.e. ≈ 87 % of the 1.8 TB/s
   reference. The identical kernel structure reaches 28–30 GB/s/core on the
   12.85 M cnn-feat stream and on all fp16 classes (128–154 MB footprints), so
   the shortfall is the documented **MTE2 effective-bandwidth degradation with
   total flow** (`PL-1.6-copy-floor`: 33.9 → 30.2 → 20.9 GB/s/core as traffic
   grows 1 M → 16 M → 64 M), not a tiling/parameter choice.
2. **The report metric's ceiling for a 3-stream same-shape kernel is ⅔ of the
   achieved-bandwidth ratio** (numerator = GM read misses only). Reaching 60 %
   needs `achieved_total_BW ≥ 1.62 TB/s` on a 201 MB fp32 flow whose measured
   ceiling here is ≈ 1.56–1.62 TB/s.
3. **Every plausible structural lever was measured and adjudicated as a tie:**
   in-place output (2 UB buffers), rank-2 `[R_blk, col]` re-tile (DESIGN §1.6.3
   alternative #2) at 3 tile heights, an exact/static tail-free fast path, and
   4 explicit outstanding loads per task. Larger tiles are UB-capped (fp32
   `max_block = 9637`; 9216 measured worse than 8192). Removing the last-iteration
   imbalance is impossible: 40 Vector cores and `N = 2²⁴` mean
   `num_logical = N/block_size` can never be a multiple of 40 (no factor 5), so
   `ceil()/mean ≈ 1.6 %` of the wall time is structural.

Per `_shared/standards/perf-feedback.md` §1 this is **not** filed as
`[DESIGN_LIMIT]` and **no `perf_feedback.md` is produced**: (a) the ceiling is a
device-bandwidth/report-metric limit, not a contradiction of a specific
DESIGN.md assumption — the design's §1.6.0 flow bound predicted this regime and
was conservative (measured 0.60–0.94× of its estimate); (b) no structural
candidate with > 2× headroom exists (the best alternative measured is a tie), and
the design's own reachable structural win (§1.6.3 alternative #2) was
implemented and measured. Both conditions of the `[DESIGN_LIMIT]` trigger are
therefore unmet.

## Final Performance Test Data

Deliverable `perf_opt/_make_clamp_tensor_kernel.py` (candidate `final` =
`r1_bs8192`, sha `424a48ec…`), all 9 tune workloads. Both columns come from
`perf_records.jsonl`: `baseline` = round-0 `phase=baseline` (sha `2fce52e4…`),
`final` = round-6 `phase=final` on the delivered file. Raw final profiles:
`profiles/final/round6/final_2443630b/<workload_id>_<hash>/`.

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | elementwise-16M-float16 | 58.039 | 56.459 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | elementwise-16M-bfloat16 | 59.319 | 55.199 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | elementwise-16M-float32 | 136.117 | 129.757 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | cnn-feat-broadcast-float16 | 43.219 | 42.359 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | cnn-feat-broadcast-bfloat16 | 43.179 | 42.339 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | cnn-feat-broadcast-float32 | 99.478 | 91.118 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | row-broadcast-float32 | 137.377 | 129.197 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | row-broadcast-float16 | 59.839 | 57.179 | final |
| examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py::_make_clamp_tensor_kernel::main | row-broadcast-bfloat16 | 56.039 | 57.379 | final |

Stop reason: **plateau** — rounds 2–5 produced no valid branch beating the round-1
winner beyond the 3 % noise threshold (all near-candidates adjudicated `tie` by
the interleaved A/B protocol), and the remaining structural candidates are
exhausted. Target status: 7/9 ≥ 60 %, 2 fp32 cases ≈ 1.5–2.5 % short (analysis
above); no `merge_blocked`; smoke precision passes.

## Deliverable change summary

`perf_opt/_make_clamp_tensor_kernel.py` = the Stage-3 kernel + per-dtype
`block_size` defaults only (no kernel-body change):

- new module-level `TUNED_BLOCK_BY_DTYPE = {"float16": 4096, "bfloat16": 4096, "float32": 8192}`;
- `_default_block_size(dtype)` now resolves through `TUNED_BLOCK_BY_DTYPE`
  (K11 fallback 2048/4096 for unknown dtypes).

**Wiring requirement for the conductor's wrapper flip** (same convention as the
sister `le`/`gt` operators): the wrapper's source-selection block must be toggled
**as a pair** —

```python
# --- perf_opt (Stage 4 tuned) ---
from .clamp_max_kernel.perf_opt._make_clamp_tensor_kernel import _make_clamp_tensor_kernel
from .clamp_max_kernel.perf_opt._make_clamp_tensor_kernel import TUNED_BLOCK_BY_DTYPE

def _default_block_size(dtype: torch.dtype) -> int:
    return TUNED_BLOCK_BY_DTYPE[Kernel.dtype_to_str(dtype)]
```

Otherwise the wrapper keeps passing the shipped K11 `block_size` (2048 fp32) and
the measured fp32 gain does not reach `bench_clamp_max.py`. This agent does not
modify the wrapper (`clamp_max.py`) or the op layer.

## Skill Retrospective

- **Validated optimization (P, positive) — per-dtype `block_size` retune of the
  MTE2-bound fp32 flat elementwise stream (2048 → 8192, UB cap 9637).** One 32 KB
  contiguous tile per operand instead of 8 KB: elementwise-16M-float32
  141.16 → 126.34 µs (−10.5 %) and row-broadcast-float32 139.86 → 127.96 µs
  (−8.5 %), both `verdict = B_better` under the interleaved A/B protocol;
  cnn-feat-broadcast-float32 99.48 → 91.12 µs (62.7 % report ratio, crosses 60 %).
  Mechanism: the kernel is MTE2-saturated (`mte2_ratio 0.99`) with a fixed
  per-task DMA/issue cost amortised over the tile, so tile bytes — not compute,
  not UB depth — set the throughput.
  - evidence: round-1 table; `logs/ab/ab_final_ew16M_fp32` (+ `row`) JSON verdicts;
    `perf_records.jsonl` `phase=baseline`/`final` pairs.
  - repro: `perf_opt/_make_clamp_tensor_kernel_round1_bs8192.py` (`TUNED_BLOCK_BY_DTYPE`
    fp32=8192) vs `../_make_clamp_tensor_kernel.py`; driver `perf_opt/bench.py`.
  - toolchain_stamp: tilelang `0.1.2+b9ece659`, Ascend910 40-Vector-core, CANN 9.0.0, Developer.
- **Negative result (D, boundary) — for this 1-`vmin`-per-task MTE2-bound kernel,
  four structural variants are ties within the ±3–5 % run band**: in-place output
  (drop `y_ub`), rank-2 `[R_blk, col]` re-tile (DESIGN §1.6.3 alternative #2) at
  `R_blk` ∈ {2,4,8}, exact static tail-free fast path, and 4 explicit outstanding
  loads per task. Each was first rejected by interleaved A/B (`tie`), and the
  DESIGN §1.6.3 adjudication rule (alternative < primary × 0.97) is therefore
  resolved in favour of the primary for the contiguous same-shape stream.
  - evidence: rounds 2–5 tables; `logs/ab/ab_r2_*`, `ab_r4_*`, `ab_r5_*` JSON.
  - repro: `perf_opt/_make_clamp_tensor_kernel_round{2,3,4,5}_*.py`.
  - toolchain_stamp: same as above.
- **Measurement trap (D, strong) — `bench --check` must be NaN-aware, otherwise a
  false "CHECK FAILED" aborts the profiled child and msprof emits no per-launch
  OpBasicInfo.** `elementwise-16M-bfloat16` failed 3/3 under `msprof op` with
  `no OpBasicInfo rows`; the child log showed `[bench] CHECK FAILED` + exit 768
  and the mismatching rows were `got=nan ref=nan` (NaN appears in the *input*
  under msprof interception, and `NaN != NaN` made `torch.equal` false). Fix:
  compare with `(got != ref) & ~(isnan(got) & isnan(ref))` in `perf_opt/bench.py`.
  Second trap in the same failure: msprof sometimes writes the CSV set **flat**
  into `OPPROF_*/` (no `main/<n>/` sub-directory), so the collector globs must be
  recursive (`**/OpBasicInfo_*.csv`) and `roofline_ratio.py` must accept both
  layouts — a non-recursive glob silently reports "no data" for a valid profile.
  - evidence: `logs/profiles_baseline_round0_baseline_8ba8496a_elementwise-16M-bfloa_b8693102.log`
    (exit 768 / no CSVs), the fixed re-run (`CHECK PASSED`, 15 per-launch rows).
  - repro: `perf_opt/bench.py` (NaN-aware check) + `perf_opt/roofline_ratio.py` (`_launch_dirs`).
  - toolchain_stamp: same as above.
- **Case index (C) — the integrated `ClampMaxFwdOp` never exercises the DESIGN
  R3 in-kernel broadcast path.** `_eager_forward` materialises the bound with
  `_expand_flat` and calls the frozen factory with `coalesced_shape=None`, so all
  benchmark cases are same-shape streams reading `2·N·E`. Reported for the
  conductor as an integration-layer observation, **not** as a design defect: the
  kernel implements and passes R3 (L0-2/L0-6), and for the report metric the
  materialisation actually *helps* (the numerator counts reads only, and
  materialising doubles them). No wrapper/op change was made (out of scope).
  - evidence: `tileops/ops/elementwise/clamp_max.py::_eager_forward`; row-broadcast
    fp32 baseline ≈ elementwise fp32 baseline (137.4 vs 136.1 µs for the same N).
  - repro: `perf_opt/bench.py` (same-shape call form) vs the module's L0-2
    (`coalesce=True`).
  - toolchain_stamp: same as above.
- **Process note (R, weak) — pattern-library write-back was not performed.** The
  skill's Phase 4.6 asks the optimizer to append new P/D entries plus minimal
  repros to `.agents/skills/tilelang-op-optimize/references/pattern-library/`;
  this task's explicit artifact-scope constraint ("产物只写
  `.../clamp_max_kernel/perf_opt/`") takes precedence, so the four value points
  above are recorded here with evidence/repro/toolchain stamps for the terminal
  `tilelang-skill-evolver` to distil. No `Optimize.md` was requested by the task.
