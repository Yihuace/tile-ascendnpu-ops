# Stage 4 tuning log — `_max_pool2d_kernel` (MaxPool2dFwdOp, TileOPs `pool` family)

Operator dir: `examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/`
Stage 4 artifacts live in `perf_opt/`. Baseline kernel (read-only) is `../_max_pool2d_kernel.py`;
the tuned artifact is `perf_opt/_max_pool2d_kernel.py`.

## 1. Toolchain and measurement setup

| item | value |
|---|---|
| repo HEAD | `b3a7b21` |
| tilelang | 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025 (`target="npuir"`) |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post8 |
| python | 3.10.12 |
| bishengir-compile | CANN 9.0.0 (`TILELANG_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile`) |
| device | Ascend 910B4 (npu-smi 26.1.1), 20 AI cores x 2 = **40 vector cores**, UB budget **192 KB** |
| latency metric | `msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off --aic-metrics=PipeUtilization,Memory`, **median Task Duration(us)** — the only latency metric used here |
| precision gate per measurement | `bench.py --check` (bit-exact vs the CPU golden; `max_diff == 0.0`, NaN mask equal) |

Round-4 toolchain re-stamp (measured, not copied): repo HEAD `b192197`, tilelang
`0.1.2+b192197db223c30d8f8f75611b83f8ec4754f067` reporting the shared libs built
2026-10-13 (`build/libtilelang{,_module,_ir}.so`), CANN 9.0.0
(`bishengir-compile` 428ab8f path above), `npu-smi 26.1.1`, Ascend 910B4, 40
vector cores, UB 192 KB. The python-side tree advanced past the round-3 stamp
(`b3a7b21`) but the compiled pipeline is the same Oct-13 build, so the round-3
compiler findings (conditional `vdeinterleave` dropped, unit-level store) were
re-verified this round through the IR diff rather than assumed.

Experiments used: rounds 1-3 = **12 (bh, ch) grid configs + 9 baseline + 9 phase_v1
- 9 final = 39 `msprof op` runs**; round 4 = **9 candidate + 6 interleaved A/B + 9
final = 24 `msprof op` runs** (at the `max_stage4_experiments=24` cap). All runs
have `l0_pass = true` (0 precision failures, 0 discarded runs).
`perf_opt/perf_records.jsonl` has 36 rows (9 baseline, 9 `phase_v1` candidate,
9 `r4_noperl` candidate, 9 `r4_noperl` final — exactly one final row per tune
workload as gate 4 requires); the superseded `merged_v3` finals are preserved in
`logs/round5/perf_records_merged_v3_archive.jsonl`. `perf_opt/tuning_sweep.jsonl`
holds the 12 grid rows.

## 2. Workload inventory (from the migrated benchmark)

Source of truth: `benchmarks/ops/bench_pool.py::test_max_pool2d_bench`, manifest
entry `MaxPool2dFwdOp` (`tileops/manifest/pool.yaml`). Full detail in
`perf_opt/workload_inventory.json` (12 items).

| workload_id | shape (N,C,H,W) | k / s / p / d | ceil | dtype | kind |
|---|---|---|---|---|---|
| smoke-1x8x16x16-k3s2p1-float16 | 1,8,16,16 | 3x3 / 2x2 / 1x1 / 1 | no | fp16 | smoke |
| smoke-1x8x16x16-k3s2p1-bfloat16 | 1,8,16,16 | 3x3 / 2x2 / 1x1 / 1 | no | bf16 | smoke |
| smoke-1x8x16x16-k3s2p1-float32 | 1,8,16,16 | 3x3 / 2x2 / 1x1 / 1 | no | fp32 | smoke |
| resnet-stem-float16 | 32,64,112,112 | 3x3 / 2x2 / 1x1 / 1 | no | fp16 | tune |
| resnet-stem-bfloat16 | 32,64,112,112 | 3x3 / 2x2 / 1x1 / 1 | no | bf16 | tune |
| resnet-stem-float32 | 32,64,112,112 | 3x3 / 2x2 / 1x1 / 1 | no | fp32 | tune |
| vgg-block-float16 | 16,128,56,56 | 2x2 / 2x2 / 0 / 1 | no | fp16 | tune |
| vgg-block-bfloat16 | 16,128,56,56 | 2x2 / 2x2 / 0 / 1 | no | bf16 | tune |
| vgg-block-float32 | 16,128,56,56 | 2x2 / 2x2 / 0 / 1 | no | fp32 | tune |
| alexnet-ceil-float16 | 32,64,55,55 | 3x3 / 2x2 / 0 / 1 | yes | fp16 | tune |
| alexnet-ceil-bfloat16 | 32,64,55,55 | 3x3 / 2x2 / 0 / 1 | yes | bf16 | tune |
| alexnet-ceil-float32 | 32,64,55,55 | 3x3 / 2x2 / 0 / 1 | yes | fp32 | tune |

Smoke items are precision-only (never profiled); all three report `precision_pass: true`
from `bench.py --check` (`smoke_check.json`). Output sizes: 56x56 / 28x28 / 27x27.
DESIGN.md shapes were used as design reference only; they did not add or remove
a measured case.

## 3. Step 0 — DMA calibration (what a copy alone costs)

Measured with the standalone copy-floor kernels (`run_copy_floor*.py`,
`copy_floor_calibration.jsonl`) on the same device, same 40 cores:

| variant | intent | duration_us | read GB/s | write GB/s | mte2_us | mte3_us |
|---|---|---|---|---|---|---|
| rd_c1_plane | 1-channel full plane read | 71.34 | 720.2 | 3.2 | 58.28 | 6.41 |
| rd_c1_r28 | 1-channel, 28-row units | 113.59 | 452.3 | 8.1 | 72.26 | 33.90 |
| rd_c1_r8 | 1-channel, 8-row units | 328.73 | 156.3 | 9.8 | 198.57 | 117.51 |
| rd_c1_r1 | 1-channel, 1-row units | 2070.79 | 24.8 | 12.4 | 1126.30 | 879.26 |
| rd_c32_r8 | 32-channel, 8-row units | 74.88 | 686.2 | 42.9 | 70.36 | 32.81 |
| rd_c32_r1 | 32-channel, 1-row units | 142.96 | 359.4 | 179.7 | 125.43 | 110.88 |
| wr_c32_r8 | 32-channel, 8-row stores | 11.52 | — | 1115.0 | 0.02 | 8.52 |
| wr_c32_r1 | 32-channel, 1-row stores | 42.80 | — | 300.1 | 0.11 | 38.85 |
| wr_c1_r1 | 1-channel, 1-row stores | 512.07 | — | 25.1 | 3.48 | 493.23 |
| rw_c32_bh1..bh4 | read+write mixed | 100.72 / 91.52 / 89.64 | 765.2 / 701.7 / 644.8 | 127.5 / 140.3 / 143.3 | 93.8 / 83.2 / 84.9 | 60.0 / 60.2 / 51.8 |

Conclusions used by the tuning:
- **Rows per DMA matter more than bytes**: the same 25.1 MB plane read costs
  71 µs at 1 unit/plane, 114 µs at 28-row units and 2071 µs at 1-row units. The
  per-copy fixed cost dominates until the unit holds a full row block, which is
  why the phase path loads whole rows and keeps `bh` rows per unit.
- **Store width matters**: a 1-channel 1-row store moves 25 GB/s, a 32-channel
  8-row store 1115 GB/s. Narrow (64 B) lane stores are the expensive case.

## 3B. Device peak bandwidth and the roofline target

Independent peak probes (`peak.py`, `run_peak2.py`, `peak_recheck.jsonl`, copy-floor
variants): **read 753.3–778.7 GB/s**, **write 1057–1115 GB/s**, **mixed read+write
682–1065 GB/s** aggregate.

The DESIGN.md roofline assumed "1.75 TB/s HBM peak x 70 % = 1.23 TB/s effective".
That is **unreachable on this device/toolchain**: the best single-direction read
ever measured is 778.7 GB/s = 44.5 % of 1.75 TB/s, and the best mixed figure is
1065 GB/s = 61 %. The tuning therefore reports efficiency against the *measured*
peaks (`/1065 GB/s` mixed, plus `/1750` for continuity with the design text) and
treats 1.23 TB/s as an invalid target rather than an unmet one. `[DESIGN_LIMIT]`
is **not** raised for this: the gap is a measurement-derived hardware fact, not a
2x structural opportunity, and the operator still lands 2.0–9.6x over baseline.

## 4. Rounds — candidate vs current best

### Round 1 (already recorded): structural rewrite, baseline -> `phase_v1`

`phase_v1` replaced the per-lane window (`k_h*k_w*out_w` narrow 128 B ops) with one
full-row sliding max + one `T.deinterleave` decimation, and kept the old planner
rule "largest ch, then largest bh".

| workload_id | baseline µs | current best (`phase_v1`) µs | delta | L0 |
|---|---|---|---|---|
| resnet-stem-float16 | 1512.06 | 230.42 | -84.8 % | pass |
| resnet-stem-bfloat16 | 1513.56 | 216.19 | -85.7 % | pass |
| resnet-stem-float32 | 2880.14 | 300.99 | -89.6 % | pass |
| vgg-block-float16 | 124.55 | 55.08 | -55.8 % | pass |
| vgg-block-bfloat16 | 124.39 | 62.31 | -49.9 % | pass |
| vgg-block-float32 | 206.41 | 76.86 | -62.8 % | pass |
| alexnet-ceil-float16 | 217.67 | 66.65 | -69.4 % | pass |
| alexnet-ceil-bfloat16 | 219.21 | 70.27 | -67.9 % | pass |
| alexnet-ceil-float32 | 395.95 | 103.19 | -73.9 % | pass |

### Round 2 — (bh, ch) grid, then the structural selection rule

Twelve `(bh, ch)` points were measured on the three fp16 shapes
(`tuning_sweep.jsonl`, every point bit-exact):

| workload | bh | ch | Task Duration µs | vec µs | mte2 µs | mte3 µs | scalar µs | u2g KB | vs current best (auto) |
|---|---|---|---|---|---|---|---|---|---|
| resnet-stem-f16 | 4 | 64 | **230.42** | 125.5 | 59.4 | 25.5 | 107.7 | 24500 | = auto, grid optimum |
| resnet-stem-f16 | 2 | 64 | 238.82 | 137.0 | 63.2 | 24.4 | 115.7 | 25088 | +3.6 % |
| resnet-stem-f16 | 8 | 32 | 241.68 | 120.1 | 67.7 | 30.8 | 108.0 | 25088 | +4.9 % |
| resnet-stem-f16 | 4 | 32 | 243.91 | 126.4 | 72.0 | 30.6 | 114.5 | 25088 | +5.9 % |
| resnet-stem-f16 | 2 | 32 | 256.90 | 140.7 | 75.7 | 31.1 | 126.5 | 25088 | +11.5 % |
| resnet-stem-f16 | 1 | 64 | 265.12 | 162.3 | 86.1 | 28.1 | 142.5 | 25088 | +15.0 % |
| vgg-block-f16 | 1 | 64 | **49.30** | 14.5 | 33.4 | 11.2 | 7.2 | 9856 | **-10.5 %** |
| vgg-block-f16 | 2 | 64 | 55.31 | 14.4 | 28.5 | 11.7 | 6.7 | 9856 | +0.4 % |
| vgg-block-f16 | 4 | 64 | 55.87 | 15.7 | 31.2 | 12.1 | 7.5 | 9856 | +1.4 % |
| alexnet-ceil-f16 | 3 | 64 | **61.12** | 17.6 | 31.6 | 11.5 | 8.4 | 9720 | **-8.3 %** |
| alexnet-ceil-f16 | 1 | 64 | 63.35 | 22.0 | 44.0 | 11.9 | 9.5 | 9720 | -5.0 % |
| alexnet-ceil-f16 | 9 | 64 | 66.65 | 13.9 | 27.9 | 10.2 | — | 9720 | = auto |
| resnet-stem-f16 | 8 | 64 | n/a | — | — | — | — | — | rejected: UB 221 KB x1.7 > 192 KB |

The grid falsified the old rule ("largest ch, then largest bh"): for `out_h = 28`
and `out_h = 27` the largest `bh` loses by 8–11 %. The rule was replaced by a
structural cost model in `_phase_plan` (no workload labels, shape/dtype/core-count
only):

```
_phase_score = num_local(bh, ch) * rows(bh) * ch * (1 + core_split_tail)
core_split_tail = (num_kernels*num_local - num_logical) / num_logical
```
plus the lane pre-filter `ch * elem_bytes >= 128 B` (narrower lanes are used only
when UB rules out every wide one) and the UB footprint guard `_phase_bytes * 1.7
<= 192 KB`.

`plan_probe.py --auto` after the change, and the measured result per workload:

| workload_id | new (bh, ch) | current best (`phase_v1`) µs | candidate (`merged_v2`) µs | delta | vec | mte2 | mte3 | scalar | u2g KB | L0 |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet-stem-float16 | 4, 64 | 230.42 | 231.47 | +0.5 % | 125.5 | 59.2 | 25.7 | 107.7 | 25088 | pass |
| resnet-stem-bfloat16 | 7, 32 | 216.19 | 216.47 | +0.1 % | 117.7 | 70.5 | 30.6 | 38.6 | 25088 | pass |
| resnet-stem-float32 | 4, 32 | 300.99 | 301.17 | +0.1 % | 120.1 | 116.2 | 48.3 | 39.4 | 35840 | pass |
| vgg-block-float16 | 1, 64 | 55.08 | 49.50 | **-10.1 %** | 14.5 | 33.5 | 11.3 | 7.2 | 9856 | pass |
| vgg-block-bfloat16 | 1, 64 | 62.31 | 63.06 | +1.2 % | 31.7 | 29.2 | 10.7 | 11.5 | 9856 | pass |
| vgg-block-float32 | 1, 32 | 76.86 | 73.27 | **-4.7 %** | 30.4 | 45.9 | 20.3 | 13.9 | 12544 | pass |
| alexnet-ceil-float16 | 3, 64 | 66.65 | 61.26 | **-8.1 %** | 17.6 | 31.4 | 11.5 | 8.4 | 9720 | pass |
| alexnet-ceil-bfloat16 | 3, 64 | 70.27 | 70.46 | +0.3 % | 36.9 | 28.0 | 11.4 | 12.7 | 9720 | pass |
| alexnet-ceil-float32 | 9, 32 | 103.19 | 95.00 | **-7.9 %** | 35.3 | 57.1 | 17.2 | 16.1 | 12528 | pass |

Three workloads regressed by <= 1.2 % (resnet f16 +0.5 %, vgg bf16 +1.2 %, alexnet
bf16 +0.3 %); two of those are the phase_v1 optimum under the same plan, i.e. msprof
noise, and vgg bf16 is the one real trade: `bh = 1` triples the unit count for a
1.2 % loss on bf16 while buying 10 % on f16/f32 with the same rule. Net across the
nine tune workloads: **-3.3 % mean, -10.1 % best** against `phase_v1`, on top of the
6.5–9.6x already gained over baseline. Because every kernel is a single `kernel_id`,
the per-kernel step is closed here: `merged_v2` is the merged winner for all nine.

### Round 3 (session 9) — row DMA / vector software pipeline (ping-pong `src`)

**Phenomenon (round-2 final profile).** `Task Duration ≈ vec + mte2 + mte3` on every
workload (resnet f16 231.5 ≈ 125.5 + 59.2 + 25.7 µs; alexnet f16 61.3 ≈ 17.6 + 31.4
- 11.5), i.e. the three pipes never overlap. Cause: the row copy is consumed
immediately by `T.deinterleave`, so the MTE2 wait sits in front of every row's
vector block.

**Candidate (single optimization point).** Ping-pong row scratch: row *r* is consumed
from one `src` while row *r+1* is prefetched into `src2`, so the GM->UB copy of the
next row overlaps the deinterleave/transpose/vmax of the current one. Parity is
resolved at **trace** time (the row pair loop is unrolled by two), the `-inf` tail
padding is armed once per buffer, and the store block stays at unit level. The
dispatcher is a structural UB predicate: the pipeline is used only when
`_phase_bytes(..., pipe=True) * 1.7 <= 192 KB`, so a shape that cannot afford the
second scratch keeps the serial row loop untouched (same arithmetic, same taps).

Enabled: vgg-block (all dtypes, 8.2 KB extra) and alexnet-ceil (all dtypes).
Not enabled: resnet-stem — `acc` (bh·out_w·ch) + `et` and the odd tile leave no room
(102 KB + 14 KB -> x1.7 = 197 KB > 192 KB), so it keeps the frozen serial body and
the frozen (bh, ch) plan.

| workload | (bh, ch) | pipe | `merged_v2` µs | `merged_v3` µs | delta | vec µs | mte2 µs | mte3 µs | L0 |
|---|---|---|---|---|---|---|---|---|---|
| resnet-stem-float16 | 4, 64 | off | 231.47 | **231.13** | -0.1% | 125.51 | 59.35 | 25.80 | pass |
| resnet-stem-bfloat16 | 7, 32 | off | 216.47 | **216.74** | +0.1% | 117.66 | 70.60 | 30.75 | pass |
| resnet-stem-float32 | 4, 32 | off | 301.17 | **303.67** | +0.8% | 120.14 | 118.14 | 48.94 | pass |
| vgg-block-float16 | 1, 64 | on | 49.50 | **46.03** | -7.0% | 16.17 | 35.70 | 11.39 | pass |
| vgg-block-bfloat16 | 1, 64 | on | 63.06 | **47.30** | -25.0% | 33.07 | 29.83 | 11.07 | pass |
| vgg-block-float32 | 1, 32 | on | 73.27 | **62.21** | -15.1% | 33.34 | 42.03 | 23.00 | pass |
| alexnet-ceil-float16 | 3, 64 | on | 61.26 | **54.70** | -10.7% | 19.15 | 39.94 | 15.48 | pass |
| alexnet-ceil-bfloat16 | 3, 64 | on | 70.46 | **62.60** | -11.2% | 39.46 | 38.37 | 16.24 | pass |
| alexnet-ceil-float32 | 9, 32 | on | 95.00 | **87.73** | -7.7% | 36.62 | 66.64 | 18.47 | pass |

The plan of the nine workloads is byte-identical to `merged_v2` (`plan_probe.py
--auto`, `plan_same=True` for all nine), so every delta above is the row pipeline
alone. Six workloads improve by 7.0–25.0 %; resnet stays inside ±0.8 % (noise
band = 3 %). Mean over the six pipelined workloads: **-12.8 %**; mean over all
nine: **-8.6 %**. Winner: `merged_v3` (per-kernel merge is a no-op: one
`kernel_id`, one body).

**Two falsifications while porting (both cost real time, both recorded).**

1. `T.deinterleave` inside a runtime `if/else` parity branch is **silently
   dropped** by the npuir pipeline: probe `probe_pipe.py` (parity via `r % 2`)
   compiled, ran, and produced garbage — the generated hivm IR has no
   `vdeinterleave` at all (the buffer it should write is read uninitialised).
   Fix: unroll the pair loop at trace time so both deinterleaves are straight-line
   code. Verified bit-exact against the serial wrapper over the whole 448-unit
   probe workload (`max_diff = 0.0`).
2. The store loop must stay at **unit** level. The first port moved it into the row
   body (as it reads in the frozen source at a glance): resnet f16 went to 486 µs
   with `u2g` 5544 KB/block instead of 616 (9x write amplification). Restored to the
   frozen unit-level store; the control run then reproduced **229.63 µs** vs the
   frozen 231.47 µs, i.e. the macro refactor itself is neutral.

**Probe evidence for the mechanism** (`probe_pipe.py`, exact resnet-f16 row body,
no per-row store): serial 201.0 µs (vec 105.2 + mte2 76.5 + mte3 30.2 = 211.9, fully
serialised) -> ping-pong 156.6 µs (vec 108.5, mte2 98.9): the DMA is now largely
hidden under the vector work. The remaining gap (156.6 vs max(vec) + mte3) is the
one-row pipeline depth; two rows of prefetch would need a third scratch and does not
fit the UB budget on any of these shapes.

### Round 4 (session 10) — transpose straight out of the `acc` slice (drop `perl`)

**Phenomenon (round-3 final profile).** The six vgg/alexnet shapes were pipelined
(`dur < Σ pipes`), but the three resnet shapes were still fully serialised
(`vec 125.5 + mte2 59.4 + mte3 25.8 = 210.7` of a 231.1 µs Task Duration), because
the structural UB predicate refused the ping-pong scratch:
`_phase_bytes(pipe=True) * 1.7 = 197 KB > 192 KB`. The footprint included a
`perl` staging tile (`out_w * ch * acc_bytes` = 7 KB on resnet) that existed only
so the store could do `copy(acc row -> perl)` then `transpose(perl -> st)`.

**Candidate (single optimization point).** Drop `perl`; transpose the output row
straight out of its `acc` slice:

```
- T.copy(acc[o*out_w:(o+1)*out_w, 0:ch], perl); T.transpose(perl, st, [1,0])
+ T.transpose(acc[o*out_w:(o+1)*out_w, 0:ch], st, [1,0])
```

and charge only `st` in `_phase_bytes` (`2 * out_w*ch*acc_bytes` →
`1 * out_w*ch*acc_bytes`). The arithmetic is untouched: same `-inf` lead-in, same
`vmax` tap sequence (still [ow, c] x [rows, c] -> [ow, c]), same NaN semantics
(bf16 still widens to f32), same trailing-row-crossing guard
`(out_w-1)*s_w+(k_w-1)*d_w-p_w <= w_in-1` (not modified). The dispatcher stays
structural — no workload name, no dtype name, only the UB predicate.

**Plan effect (host arithmetic, `plan_probe.py --auto`).** All nine `(bh, ch)`
pairs are byte-identical to round 3; only the `pipe` flag flips on the three
resnet shapes:

| workload | (bh, ch) | pad_rows | UB before ×1.7 | UB now ×1.7 | pipe r3 → r4 |
|---|---|---|---|---|---|
| resnet-stem-float16 | 4, 64 | 72 | 197.4 KB (reject) | **185.3 KB** (accept) | off → **on** |
| resnet-stem-bfloat16 | 7, 32 | 72 | 202.9 KB (reject) | **191.2 KB** (accept, 0.8 KB margin) | off → **on** |
| resnet-stem-float32 | 4, 32 | 72 | 197.4 KB (reject) | **185.3 KB** (accept) | off → **on** |
| vgg-block ×3 / alexnet-ceil ×3 | unchanged | 48 | already accepted | -7 KB | on → on |

**Candidate vs current best (`merged_v3`) — `msprof op` Task Duration, all nine
tune workloads re-measured, `l0_pass = true` everywhere:**

| workload | (bh,ch) | pipe | current best µs | candidate µs | delta | vec µs | mte2 µs | mte3 µs | Σpipes/dur | L0 |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet-stem-float16 | 4, 64 | no → **yes** | 231.13 | **162.42** | **-29.7 %** | 118.67 | 74.05 | 27.03 | 0.91 → **1.35** | pass |
| resnet-stem-bfloat16 | 7, 32 | no → **yes** | 216.74 | **145.52** | **-32.9 %** | 106.84 | 73.79 | 35.53 | 1.01 → **1.49** | pass |
| resnet-stem-float32 | 4, 32 | no → **yes** | 303.67 | **228.13** | **-24.9 %** | 100.68 | 145.03 | 67.08 | 0.95 → **1.37** | pass |
| vgg-block-float16 | 1, 64 | yes | 46.03 | 45.69 | -0.7 % | 16.17 | 35.52 | 11.41 | 1.37 → 1.38 | pass |
| vgg-block-bfloat16 | 1, 64 | yes | 47.30 | 47.39 | +0.2 % | 33.07 | 29.93 | 11.09 | 1.56 → 1.56 | pass |
| vgg-block-float32 | 1, 32 | yes | 62.21 | 61.89 | -0.5 % | 33.35 | 41.64 | 22.96 | 1.58 → 1.58 | pass |
| alexnet-ceil-float16 | 3, 64 | yes | 54.70 | 54.05 | -1.2 % | 19.15 | 39.72 | 15.19 | 1.36 → 1.37 | pass |
| alexnet-ceil-bfloat16 | 3, 64 | yes | 62.60 | 62.33 | -0.4 % | 39.46 | 38.25 | 15.44 | 1.50 → 1.49 | pass |
| alexnet-ceil-float32 | 9, 32 | yes | 87.73 | 87.89 | +0.2 % | 36.63 | 66.72 | 18.38 | 1.39 → 1.39 | pass |

`Σpipes/dur = (vec + mte2 + mte3) / Task Duration` is the overlap signature:
`< 1` means the wall clock exceeds the summed pipe times (serialised, with
waiting gaps) and `> 1` means the pipes run concurrently. The three resnet shapes
crossed from `< 1` to `≈1.4` (DMA now sits under the vector work); the six small
shapes were already `> 1` and stayed there, which is exactly why they are a tie.
AICore (cube) is unused by this AIV-only kernel (`block_dim = 40` vector cores),
so the utilisation figure that matters is the vector-pipe share: resnet f16
`vec/dur` went 54.3 % → **73.1 %**, resnet f32 39.6 % → **44.1 %**, resnet bf16
54.3 % → **73.4 %**. Memory columns (sum over the 40 block rows, from the final
rows): resnet f16 `g2u 76.5 MB / u2g 24.5 MB`, resnet f32 `g2u 125 MB / u2g
35 MB` — read traffic is unchanged, so the whole resnet gain is pipe overlap.

**IR diff (hivm-level MLIR, `TILELANG_DUMP_IR=TRUE`, resnet f16).**
`logs/round5/ir_frozen_resnet_f16.mlir` (141 lines, frozen) vs
`logs/round5/ir_cand_r4_resnet_f16.mlir` (240 lines, candidate):

| effectful op | frozen | candidate | note |
|---|---|---|---|
| `hivm.hir.vdeinterleave` | 1 | 3 | row pair unrolled at trace time (2 bodies) + odd-row epilogue; none inside a conditional |
| `hivm.hir.vtranspose` | 3 | 7 | 2 per row body × 3 + 1 in the store loop |
| `hivm.hir.vmax` | 3 | 9 | 3 taps × 3 row bodies |
| `hivm.hir.vbrc` | 4 | 5 | +1 for the `src2` arm of the ping-pong buffers |
| GM store (`bufferization.materialize_in_destination` inside `scf.for 0..bh`) | 1 op / bh iter | 1 op / bh iter | **store granularity unchanged: one write per output row, `bh` per unit — no per-input-row store, so the historical 9x write amplification is not reintroduced** |

No effectful op was dropped or moved into a runtime branch.

**Inputs and falsifications.**

1. **Lever 2 (`pad_rows` 72 → 64, the measured `slack = 7` form) — falsified
   before spending a profile.** `probe_padrows.py` (host arithmetic,
   `logs/round5/probe_padrows.log`) enumerates the full `(bh, ch, pipe)` grid for
   all nine cases under `pad_rows` and `pad_rows - 8`. Only two pairs become
   feasible (`vgg bf16 (7,64,serial)`, `alexnet f32 (3,64,pipe)`), and **neither
   beats the auto plan on `_phase_score`**, so the 2 KB saving has no consumer:
   every plan is already the structural optimum and every shape is already
   pipelined. Not implemented (it would only remove a safety margin that
   `dbg6.py` found necessary at `slack = 1`).
2. **The six vgg/alexnet shapes are a tie, not a gain — interleaved A/B.**
   Their deltas (+0.2 … -1.2 %) are inside the 3 % noise band, so the worst-looking
   one (vgg bf16, +0.8 % in the single run) went through
   `.agents/tools/ab_test.py` (3 alternating pairs, `launch-count 20`):
   A (frozen) merged median **47.38 µs**, B (candidate) **47.17 µs**,
   `b_rel_change = -0.44 %`, exact paired sign test `p = 1.0`,
   `direction_consistent = false`, verdict **tie**
   (`logs/round5/ab_vgg_bf16/`, raw msprof dirs under it). The structural change
   only *removes* one `vcopy` per output row, so a tie is the expected result.
3. **Lever 3 (vgg two-row prefetch) — deferred, with the reason measured.** vgg's
   phase tile has `rows = 2` (`k=2, s=2`), i.e. a unit only holds two input rows,
   so a deeper *row* pipeline cannot extend beyond one row of look-ahead; the
   remaining MTE2 exposure is per-copy latency (7 KB per row copy → 352 KB g2u
   summed at 394 GB/s, not bandwidth-limited). The lever that could still move it
   is a **cross-unit** pipeline (prefetch the next unit's row 0 during the current
   unit's row 1), which restructures the unit loop and needs its own round with
   its own nine-case re-measure. Not attempted here (single-variable discipline,
   and this round already reached the 24-run cap).

**Round-4 verdict.** `r4_noperl` is the merged winner for all nine workloads:
one `kernel_id`, one body, no per-shape dispatch. Mean over the nine tune
workloads **-10.0 %**, mean over the three resnet workloads **-29.2 %**; the
six-shape tie is confirmed by A/B. Final artifact
`perf_opt/_max_pool2d_kernel.py`, sha256
`0dbff0d56220a527737371b0c336f4732f6f5a08ebf8215332cbda47ec608e5a`.

## 5. Structural probes and falsifications

1. **Decimation primitive** (`probe_prims.py`). `T.deinterleave(channel_nums=2)` is
   the only primitive that performs the stride-2 even/odd split in one op; the
   alternatives (two strided `T.copy` gathers, `vcast`-based reshape, a `vbrc`
   trick) were probed and either compile-failed or produced one narrow op per
   element pair (the per-op overhead the rewrite exists to avoid). Adopted.
2. **Write amplification is per channel-row burst, ~100–118 B.** Traffic summed over
   the 40 block rows shows `u2g` is bit-identical across tilings with the same
   (row count x row length) — 25,088 KB for five different resnet tilings — so the
   staging path does not over-write; the excess is a fixed per-burst cost
   (56 B row -> 168 B, 108 B -> 221 B, 112 B -> 214 B, 224 B -> 305 B).
   Consequence measured: vgg 3.0x, alexnet 3.18x, resnet f16 1.91x write amp.
   **Falsified as a lever for resnet**: its limiter is the vector pipe
   (vec 125.5 µs of 231.5 µs Task Duration, mte3 only 25.7 µs = 11 %), so the
   designed block-store fix (-35 % predicted write bytes) cannot move it, and at
   `bh = 4, ch = 64` the extra staging tile pushes the UB footprint to ~123 KB
   (x1.7 = 210 KB > 192 KB), i.e. the planner would drop the current-best config.
   Not implemented; kept as a documented follow-up for the narrow-row shapes only.
3. **vgg / alexnet tail balance** (the rule change). Hypothesis "bigger `bh` is
   always better" is false: `out_h = 27` with `bh = 9` puts 96 units on 40 cores
   (25 % of the last wave idle) and loses 8 % to `bh = 3` (288 units, 11 % tail),
   even though `bh = 9` reads fewer halo rows. Encoded as the `core_split_tail`
   term; it reproduces all twelve measured points (rank order and the 8–11 % gaps).
4. **Lane width.** A 64 B lane (`ch = 32` fp16) does 6 % less element work than a
   128 B lane yet takes 4.9 % longer (resnet 241.68 vs 230.42 µs) i.e. ~11 %
   throughput loss per element. Encoded as the `>= 128 B` pre-filter.
5. **Trailing row-crossing (correctness probe, found here).** `probe_outw_floor.py`
   sweeps `w_in = 11/13/15/17/27` with `k=3 s=2 p=1` and shows the phase path
   loses 5–9 of 960 elements **in the last output column only** whenever that
   column's window needs right padding (the phase tile only has a -inf lead-in
   slot for the left side). All nine benchmark shapes have their last tap inside
   the row, which is why `bench.py --check` never saw it. Fixed with a structural
   applicability guard `(out_w-1)*s_w + (k_w-1)*d_w - p_w <= w_in - 1`; those shapes
   now fall back to the general grid and are bit-exact (verified). After the guard
   `probe_outw_floor.py` reports EXACT for all five, and the Boundary suite's
   large / subnormal / pm-inf / all-NaN rows went from MISMATCH to PASS
   (`logs/round3/L0_all_v5.log`, `max_diff = 0.0` for all three dtypes).
6. **L1 path tags.** Four L1 rows still asserted `expect_path="general"` for shapes
   the phase path has handled since round 1. They were re-run with
   `expect_path=None` (`probe_l1_paths.py`, `max_diff = 0.0` on vgg/alexnet bf16 and
   fp32, and the fp16 disjoint shape) before the tags were updated to `"phase"`.
   No tolerance, case or assertion was relaxed; the tag now documents the actual
   (bit-exact) dispatcher branch.

Gate suites on the frozen artifact: `python _max_pool2d_kernel.py --level all`
-> exit 0, **L0 ALL PASS** (manifest shapes x dtypes, C4/C4b/C4x, guards, NaN/Inf,
validation), **L1 ALL PASS** (25 rows), L2 7/7 PASS, Boundary 7/7 PASS,
`max_diff = 0.0` for bf16/f16/f32 (`logs/round3/L0_all_v5.log`).

## 6. Skill Retrospective

- **What worked.** (a) The "one measurement, one append" discipline: 39 `msprof op`
  runs across 8 interrupted sessions with no lost data and no rewritten rows.
  (b) Grid-first, rule-second: twelve (bh, ch) points were enough to *falsify* the
  inherited planner rule, and the replacement is nine lines of arithmetic that
  needs no device access. (c) Cost-model in structural terms (per-core element
  count, core-split tail, lane width) rather than per-workload dispatch — the same
  rule then improved shapes it was never tuned on (vgg bf16/f32, alexnet f32).
- **What cost time.** (a) The L1 suite asserts the *path tag* before it compares
  numbers, so a stale tag hides a real numeric bug in exactly the way it did here
  (round 1–7 read "L1 red = test bookkeeping" when one of the six was a genuine
  wrong-last-column bug). Suggested change to the skill: when a Stage 4 rewrite
  introduces a new dispatcher path, require `--level all` immediately and treat a
  path-tag mismatch as a *numeric* investigation, not a label update.
  (b) `bench.py --check` with `randn` data cannot see padding/NaN edges; a
  special-value variant of `bench.py` would have caught the trailing-column bug at
  round 1 instead of the last micro-step. (c) The write-amplification thread
  consumed a full session before the vec-bound profile showed it could not help
  resnet; profile-first would have skipped it.
- **Process proposals.** `P`: add a "path-tag mismatch => re-run with
  `expect_path=None` and report max_diff" rule to the optimize skill's gate
  checklist. `D`: record the measured device peaks (read 753–779, write 1057–1115,
  mixed 682–1065 GB/s) in the hardware cost model so later stages stop comparing
  against 1.75 TB/s. `R`: when a plan-level constant (`_PHASE_LANE_BYTES`,
  `_UB_MULTIBUFFER_FACTOR`) is tuned, require the probe script that justified it to
  be kept in `perf_opt/` — both are re-derivable in one host run today.

- **Session 9 addendum (row pipeline).** (a) The round-2 profile already *said*
  "`dur = vec + mte2 + mte3`", but the conclusion "no lever left" was wrong: the sum
  is a *serialisation* signature, and reading it as a roofline bound cost a session.
  Proposed rule `P`: when `dur ≈ Σ pipe times` (within ~10 %), require a
  software-pipeline hypothesis to be evaluated before closing a round. (b) Two
  compiler behaviours were only discoverable by measuring: `vdeinterleave` is
  silently dropped inside a runtime `if/else`, and store loops are only cheap at unit
  level. Proposed rule `R`: any Stage 4 body restructuring must (i) diff the
  generated hivm IR against the frozen version and check that every *effectful* op
  still exists, and (ii) keep the store block at the granularity where the frozen
  version had it (one store pass per unit, not per input row). (c) Value point `D`:
  ping-pong row DMA is worth -7…-25 % on the four DMA-comparable shapes here and 0 %
  where `vec >> mte2` but UB is tight — the predicate, not the pipeline, is the
  reusable knowledge.

- **Session 10 addendum (round 4 — UB headroom as the real lever).** (a) The
  round-3 note already contained the answer ("free ~7 KB by transposing straight
  from the `acc` slice"), and it was worth **-24.9…-32.9 %** on the three resnet
  shapes while being a no-op elsewhere — i.e. the win came from *flipping a
  structural predicate*, not from the removed copy. Proposed rule `P`: when a
  structural optimisation is already implemented but gated by a resource
  predicate, the first candidate of the next round should be **footprint
  reduction to flip that predicate**, and the round must show the predicate table
  (before/after UB and the accept/reject decision) as evidence.
  (b) A "free UB" change must be charged honestly in `_phase_bytes`: dropping
  `perl` changes the model term from `2 * out_w*ch*acc_bytes` to
  `1 * out_w*ch*acc_bytes`; if the model had kept the old term the pipeline would
  still have been rejected and the change would have looked like a 0 % result.
  Proposed rule `R`: any allocation change must be accompanied by the matching
  footprint-model edit in the same commit, and the plan probe re-run to prove
  which shapes changed plan/predicate.
  (c) The 6 "unchanged" shapes were correctly reported as a tie, and the A/B
  protocol (3 alternating pairs, sign test `p = 1.0`) is what turns "+0.8 %
  regression" into a defensible statement instead of a spurious rollback.
  Value point `D`: on this toolchain, `Σpipes/dur < 1` identifies serialisation
  and `> 1` identifies a pipelined body (resnet f16 0.91 → 1.35 with essentially
  the same pipes); the ratio is a cheaper diagnostic than reconstructing an
  overlap model, and it is the metric that made the round-3 mis-reading visible.

## Round-4 Final Performance Test Data (historical)

> Round-4 record, kept verbatim. Its `final` rows were measured on the round-4
> artifact (sha256 `0dbff0d5…`); they no longer reconcile with the current
> `perf_opt/_max_pool2d_kernel.py` (round 5 changed the file) and are archived,
> row for row, in `logs/round5/perf_records_round4_full_history.jsonl`. The
> canonical, currently-valid table is the round-5 `### Final Performance Test
> Data` section at the end of this log.

All nine rows share `kernel_id = _max_pool2d_kernel` and
`candidate_id = r4_noperl`:

| workload_id (round 4) | r4_baseline_us | r4_final_us | r4_candidate_id |
|---|---|---|---|
| resnet-stem-float16 | 1512.06 | 162.42 | r4_noperl |
| resnet-stem-bfloat16 | 1513.56 | 145.52 | r4_noperl |
| resnet-stem-float32 | 2880.14 | 228.13 | r4_noperl |
| vgg-block-float16 | 124.55 | 45.69 | r4_noperl |
| vgg-block-bfloat16 | 124.39 | 47.39 | r4_noperl |
| vgg-block-float32 | 206.41 | 61.89 | r4_noperl |
| alexnet-ceil-float16 | 217.67 | 54.05 | r4_noperl |
| alexnet-ceil-bfloat16 | 219.21 | 62.33 | r4_noperl |
| alexnet-ceil-float32 | 395.95 | 87.89 | r4_noperl |

Final artifact `perf_opt/_max_pool2d_kernel.py`, sha256
`0dbff0d56220a527737371b0c336f4732f6f5a08ebf8215332cbda47ec608e5a` (all nine final
rows were measured on that file, same `candidate_id = r4_noperl`, `l0_pass = true`,
raw profiles under `profiles/final_v4/`). The round-3 artifact is kept as
`_max_pool2d_kernel_round3_frozen.py` (sha256 `738bf413…`) and its nine final rows
are archived in `logs/round5/perf_records_merged_v3_archive.jsonl`; the session-8
artifact remains as `_max_pool2d_kernel_v2_frozen.py` (sha256 `7385d4a2…`). The
nine `phase="candidate"` `r4_noperl` rows are the same-candidate pre-final
measurements and differ from the final rows only by msprof run-to-run noise
(≤1.7 %).

Smoke workloads (precision only, `bench.py --check`, `max_diff = 0.0`, NaN mask
equal, all three dtypes): `smoke-1x8x16x16-k3s2p1-{float16,bfloat16,float32}`
→ `precision_pass = true` (`logs/round5/smoke_r4.log`, `smoke_check.json`).
No tune workload was skipped, no case or tolerance was relaxed.

Achieved bandwidth, `bytes = (N*C*H*W + N*C*outH*outW) * elem_bytes`,
`achieved = bytes / Task Duration`:

| workload_id | bytes MB | final_us | achieved GB/s | / 1.75 TB/s | / measured peak (1065 GB/s) | speedup vs baseline |
|---|---|---|---|---|---|---|
| resnet-stem-float16 | 64.23 | 162.42 | 395.4 | 22.6 % | 37.1 % | 9.31x |
| resnet-stem-bfloat16 | 64.23 | 145.52 | 441.4 | 25.2 % | 41.4 % | 10.40x |
| resnet-stem-float32 | 128.45 | 228.13 | 563.1 | 32.2 % | 52.9 % | 12.62x |
| vgg-block-float16 | 16.06 | 45.69 | 351.5 | 20.1 % | 33.0 % | 2.73x |
| vgg-block-bfloat16 | 16.06 | 47.39 | 338.8 | 19.4 % | 31.8 % | 2.62x |
| vgg-block-float32 | 32.11 | 61.89 | 518.9 | 29.7 % | 48.7 % | 3.34x |
| alexnet-ceil-float16 | 15.38 | 54.05 | 284.5 | 16.3 % | 26.7 % | 4.03x |
| alexnet-ceil-bfloat16 | 15.38 | 62.33 | 246.7 | 14.1 % | 23.2 % | 3.52x |
| alexnet-ceil-float32 | 30.75 | 87.89 | 349.9 | 20.0 % | 32.9 % | 4.51x |

Remaining headroom after round 4: the limiter has moved. resnet f32 is now
**MTE2-bound** (`mte2 145.0 µs` of a 228.1 µs Task Duration; `g2u` 125 MB at
~862 GB/s, i.e. above the copy-floor read peak, so the DMA engine is saturated —
the only lever left is *less* read traffic, which needs a larger `bh` the UB
budget cannot afford at `ch = 32, f32`). resnet f16/bf16 are vector-bound with the
DMA mostly hidden (`vec 118.7` / `106.8` of 162.4 / 145.5 µs; `vec/dur` 73 %).
The six small shapes still pay per-copy latency (vgg f16: `mte2 35.5 µs` for
14 MB moved in 7 KB row copies, 394 GB/s, i.e. latency- not bandwidth-limited);
the next structural candidate there is a **cross-unit** pipeline (prefetch the
next unit's row 0 while finishing the current unit's last row), which needs its
own round and its own nine-case re-measure. `pad_rows` 72 → 64 was falsified this
round (no plan consumer, see Round 4).

---

### Round 5 (session 11) — dedicated disjoint-window span path (`stride == kernel`)

**Toolchain stamp (re-measured this round).** tilelang dev-root build
`0.1.2+16f1ee4`, repo HEAD `16f1ee4`, CANN 9.0.0 (`bishengir-compile
1.1.0-428ab8f`), torch 2.9.0+cpu / torch_npu 2.9.0.post8, **Ascend910B2**
(`NPUUtils.get_aicore_num()` = **24** → 48 pure-Vector cores, UB 196 608 B —
queried in-process this session, not copied from another board). `kb_stale_check.py`
still reports 407 stale entries library-wide (toolchain stamp mismatch), so every
behaviour this round depends on was re-probed here (`probe_span2d.py`); the
entries cited below are used as *hypotheses* only.

#### Scope and trigger (single-shape)

A dedicated path for **disjoint windows** — `stride == kernel`, `padding == 0`,
`dilation == 1`, `k*out == in` on both axes — enabled for exactly one signature:

```python
_SPAN_SCOPE_SHAPES = ((16, 128, 56, 56),)          # vgg-block
def _span_scope(n, c_in, h_in, w_in, dtype):
    return (n, c_in, h_in, w_in) in _SPAN_SCOPE_SHAPES and dtype == "float16"
```

`_span_plan` itself carries the full algebraic guard (`s==k`, `p==0`, `d==1`,
`k*out==in`, `dtype != bfloat16`, `ch | C`, `ch*eb % 32 == 0`, `bh | out_h`,
live-UB budget) so the body is general, but the **call site** gates it to the vgg
fp16 signature: all 8 other full workloads (bf16/fp32 twins, resnet-stem,
alexnet-ceil) and all smoke cases keep their round-4 plan and kernel body
**byte-for-byte** (`_plan` prints `path=phase`/`general` for every one of them,
identical `(bh, ch)` to round 4 — see the plan dump below). The comment in the
source marks it *"round-5 single-shape scope; promote only if >3%"*.

#### Phenomenon (round-4 final profile, re-measured in-session)

`vgg-block-float16` baseline = **36.86 µs** median `Task Duration`
(min 35.30 / max 38.88, 20 launches, `--warm-up=5 --dump=off`), `vec 12.77`,
`mte2 27.75`, `mte3 8.91`, `scalar 6.21`, Σpipes/dur = 1.29. This reproduces the
official 10-23 report (36.00 µs / Ratio 14.53 %) within 2.4 %.

The diagnosis in the task brief was **confirmed and quantified**: the amplifier
is not "bandwidth" but *bytes moved per useful byte*:

| metric | round-4 phase path | round-5 span path | note |
|---|---|---|---|
| `GM_to_UB` (KB, all 48 blocks) | 14 336 | 12 544 | useful read = 12 544 KB |
| DRAM read (`read_main_memory_datas`) | 25 094 KB (1.53×) | 12 550 KB (**1.00×**) | 112 B row / 6272 B stride read amplification |
| DRAM write (`write_main_memory_datas`) | 9 856 KB (3.00×) | 4 480 KB (**1.37×**) | 56 B store rows (PL-1.19) |
| total DRAM traffic | 34.9 MB | 17.0 MB | **2.05× less** |
| DRAM throughput | 0.95 TB/s | 0.95 TB/s | same engine rate |

So the phase path was paying 2.05× the necessary traffic at the *same* delivered
DRAM rate: 112 B bursts with a 6272 B stride (CONST-dma-burst-rowstride-floor
predicts ≈0.38 TB/s for that shape) and 3× write amplification. Every input
element was read 1.53× and every output element written 3×.

#### Candidate optimization points (one per branch, all from the same base)

| id | main change | structure |
|---|---|---|
| **r5-T** | one long-burst DMA + **two wide middle-axis reduces**, channel-innermost | `span(ch,R,w) → reshape(ch,R·w) → transpose(R·w,ch) → reshape(R·out_w,k_w,ch) → reduce_max(dim=1) → reshape(bh,k_h,out_w·ch) → reduce_max(dim=1) → reshape(bh·out_w,ch) → transpose → store` |
| r5-W | one long-burst DMA + `deinterleave` W-decimation + 4-D H reduce, **no transpose** | `span(ch,R,w) → deinterleave → eo/oo(ch,R,out_w) → vmax → reshape(ch·bh,k_h,out_w) → reduce_max(dim=1) → store` |
| r5-TF | r5-T + flattened 2-D store region | same as r5-T, `out` declared `(n,c,out_h·out_w)` |

Per unit the winner's runtime op list is exactly
`{1 memref.copy (DMA), 2 vtranspose, 2 vreduce<max>, 1 store}` — verified in the
hivm census (below), down from ~10 ops per *input row* in the phase path.

#### Candidate vs current best — `msprof op` Task Duration, vgg-block-float16

| candidate | tiling | Task Duration µs | vec µs | mte2 µs | mte3 µs | scalar µs | Σpipes/dur | L0 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **current best (r4_noperl, phase)** | bh 1, ch 64, pipe | **36.86** | 12.77 | 27.75 | 8.91 | 6.21 | 1.29 | pass | baseline |
| r5-W | bh 4, ch 64 | 73.60 | 40.19 | 28.39 | 9.96 | 31.25 | 1.49 | pass (bit-exact) | **rejected** — 2× slower |
| r5-T | bh 2, ch 128 | 22.56 | 6.82 | 11.36 | 6.93 | 4.46 | 1.34 | pass | rejected (short burst 448 B) |
| r5-T | bh 2, ch 64 | 21.56 | 7.01 | 10.50 | 6.68 | 7.37 | 1.42 | pass | rejected (448 B) |
| r5-T | bh 7, ch 32 | 21.34 | 8.57 | 7.07 | 4.76 | 9.51 | 1.24 | pass | rejected (64 B lane) |
| r5-TF | bh 4, ch 64 | 17.89 | 7.36 | 6.69 | 5.19 | 6.06 | 1.39 | pass | within noise of r5-T, no write gain |
| r5-T | bh 1, ch 128 | 27.06 | 6.55 | 14.90 | 8.32 | 5.57 | 1.24 | pass | rejected (224 B burst) |
| **r5-T (winner, merged)** | **bh 4, ch 64** | **18.20 / 18.24 / 18.17** | 7.36 | 6.76 | 5.29 | 6.06 | 1.38 | pass | **adopted** |

The winner was measured 3× independently (`profiles/round5/final_span_r5_rep{1,2,3}`,
**18.20 / 18.24 / 18.17 µs**, median **18.20 µs**, spread ±0.2 %); the baseline was
measured on the same device session, within the hour before the candidate runs
(same 20-launch protocol), and the two distributions do not overlap (baseline min
35.30 µs vs candidate max 19.26 µs). The candidate beats the current best by
**−50.6 % (2.03×)**; the 3 % gate (34.92 µs) is cleared by a wide margin.
`r5-TF`'s flat store changed nothing (`u2g` identical at 4480 KB), so it was
dropped instead of carrying a changed output signature.

(Measurement provenance: the first final pass on this artifact family measured
17.97 / 18.14 / 17.99 µs; a pure host-side signature cleanup — two unused
parameters dropped from `_span_bytes`, zero IR change — then changed the artifact
hash, so the whole final pass was re-run on the final file and *those* numbers are
the ones recorded in `perf_records.jsonl`. Same IR, same pipes (`vec 7.358`,
`mte2 6.76`, `mte3 5.29`, `scalar 6.06`); the 1.2 % shift is session drift.)

**Why r5-W failed (own falsification, kept as a negative data point).**
`deinterleave + 4-D reduce` is bit-exact but 2× slower (`scalar 31.2`, `vec
40.2`): its H reduce runs with a 56 B innermost lane (`(ch·bh, k_h, out_w)`) and
the `vreduce` over that form scalarises, exactly the failure mode
VP-2026-0367 records for narrow innermost lanes. Input structure must put the
**widest** available axis innermost — the channel axis (128 B) for the W reduce
and `out_w·ch` (3584 B) for the H reduce.

#### Tiling + UB byte accounting (winner: bh 4, ch 64, rows 8)

Selection rule (`_span_score`, shape-derived only): work proxy × core-split tail,
divided by `min(1, burst/1 KiB)` (CONST-dma-burst-rowstride-floor: 126 B → 0.38
TB/s, 1 KiB → 1.03 TB/s) and multiplied by a lane-width penalty (128 B = 1.0,
64 B = 1.11 as measured on the phase path, 32 B = 1.25 extrapolated). It
reproduces the measured order of all five swept configs and picks `(bh 4, ch 64)`
→ `num_logical = 224`, `num_kernels = 48`, `num_local = 5` (7.1 % tail).

UB accounting (fp16, per core). The naive sum over every `alloc_ub` over-counts
`T.reshape` targets, which alias their source (they lower to
`collapse_shape`/`expand_shape`, i.e. no allocation at all — confirmed in the IR):

| tile | shape | bytes | live with |
|---|---|---|---|
| `src3` | (64, 8, 56) | 57 344 | `tr` (the transpose) |
| `src2` | (64, 448) | *aliases `src3`* | — |
| `tr` | (448, 64) | 57 344 | `src3` |
| `w3` | (224, 2, 64) | *aliases `tr`* | — |
| `wacc3` | (224, 1, 64) | 28 672 | `tr` |
| `wacc2`/`h3` | (224, 64) / (4, 2, 1792) | *aliases `wacc3`* | — |
| `hacc3` | (4, 1, 1792) | 14 336 | `wacc3` |
| `hacc2`/`st2` | (112, 64) / (64, 112) | 14 336 | — |
| `st3` | (64, 4, 28) | *aliases `st2`* | — |
| **IR liveness peak** | `src3 + tr` | **114 688 B (58.3 % of 192 KB)** | |
| plan guard (`_span_bytes`, worst case) | `2·span + wacc + 3·tail` | **186 368 B** (budget 240 000, same convention as the max_pool1d group path) | |

Naive sum of all 11 declarations = 387 072 B; it is *not* the footprint, because
7 of the 11 are reshape views. The IR liveness peak (114 688 B) is what the
backend must materialise and it leaves a 78 KB safety margin; the measurement
(no device fault, bit-exact) is the empirical confirmation.

#### IR evidence (store granularity, no scalarisation)

`TILELANG_DUMP_IR=TRUE` on the production file (`logs/round5/ir_final_vgg_f16_span.mlir`,
69 lines). Census of the `====== final npuir ======` module:

| op | count | reading |
|---|---|---|
| `memref.copy` (GM→UB) | 1 | one DMA per unit, `(64, 8, 56)` = 64 runs of **896 B** at 3136 B stride |
| `hivm.hir.vtranspose` | 2 | 1 full-span + 1 output tile, both 2-axis swaps |
| `hivm.hir.vreduce <max>` | 2 | `224×2×64` (`dim=1`, inner 128 B) and `4×2×1792` (`dim=1`, inner 3584 B) |
| `bufferization.materialize_in_destination` | 1 | unit-level store of `(64, 4, 28)`; per-channel run = 4·28 = **112 elements (224 B)** vs 28 elements (56 B) before → write amplification 3.00× → 1.37× |
| `hivm.hir.vmax / vcast / vbrc / vselect / vdeinterleave` | 0 / 0 / 0 / 0 / 0 | no narrow per-lane op anywhere |
| `scf.for` | 1 | the persistent unit loop only — **no scalarised inner loop**, no 1-element vector op |

Stop-condition audit: (1) no scalarisation — `aiv_scalar` 6.21 → 6.05 µs and the
census above; (2) store granularity did **not** degrade (224 B rows, 1 store per
unit, `u2g` amplification 3.00× → 1.37×); (3) UB peak 114 688 B ≤ 192 KB with a
78 KB margin (table above); (4) gain 50.6 % ≫ 3 %. None fired.

#### Precision evidence

Bit-exactness argument: max is associative and commutative, every window element
is read exactly once (the `k·s` partition is exact, no halo/clamp/padding), and
the two reduces are plain maxima over disjoint tap sets — so the reduction order
cannot change the value, and a tie keeps the same element (hence the same value,
identical to `F.max_pool2d`'s "first max" tie rule for a *value* output).

NaN is the only order-sensitive part, and this path does not rely on the
undocumented part of the API blindly — it was **measured**: `vtranspose` and
`vreduce<max>` propagate NaN on this toolchain, and the output NaN mask equals
PyTorch's bit-for-bit. The same NaN-mask construction as the existing suite
(corner / interior / full-column / bottom-right NaN, plus an all-NaN 2×2 window,
plus `{+inf,−inf}` mixes) is now a gate case:

```
[L0-16] PASS: span path NaN mask equal (32 NaN outputs), finite max_diff=0, {+inf,-inf} -> +inf
[L0-3]  PASS: shape=(16,128,56,56) out=(28,28) dtype=float16 path=span CH=64 max_diff=0.000e+00
[L0] ALL PASS (manifest shapes x dtypes + C4/C4b + guards + NaN/Inf + validation)
Summary max_diff per dtype: bfloat16=0.000e+00, float16=0.000e+00, float32=0.000e+00
All check passed!
```

`--level all` (L0 + L1 + L2 + Boundary) is green: 79 PASS lines, `[L1] ALL PASS`,
`Summary max_diff per dtype: 0.000e+00` for all three dtypes
(`logs/round5/L1_all_r5_final.log`, run on the final artifact `04013b96…`). `bench.py --check` on the final artifact:
`max_diff=0.000e+00 exact=True nan_ok=True`. Smoke (precision only, all three
dtypes): `max_diff=0.000e+00 exact=True nan_ok=True` each.

#### Diff shape vs the frozen round-4 file

`diff -u history_version/_max_pool2d_kernel_round4_pre_round5_20261023.py perf_opt/_max_pool2d_kernel.py`
→ **294 added lines, 2 removed lines** (of the additions, ~180 are the new span
section + `_func_span` body + dispatch, ~50 are the new `L0-16` gate case, the
rest are comments), and the two removals are exactly the two dispatch/test tag
lines that had to change:

```
-    if path == "phase":                    ->  if path == "span":  (+ elif chain)
-         "L0-3/vgg-block", "phase"),       ->  "L0-3/vgg-block", "span"),
```

Everything else is additive: the span section (`_span_scope` / `_span_bytes` /
`_span_score` / `_span_plan` + constants), the `path == "span"` body
(`_func_span`), the dispatch insertion in `_plan`, the `span` key in the plan
dict, and the new `L0-16` gate case. **No edit inside `_phase_plan` / the `_plan`
general-or-phase branches, no edit to `_func_phase` / `_func_global` /
`_func_identity`, no change to any `_PHASE_*` constant, no change to the wrapper.**
Non-regression sentinel on the unchanged paths: vgg bf16 (same shape, phase path)
37.74 µs and resnet-stem f16 126.35 µs in the same session, i.e. the same
performance class as their round-4 code (the whole device measured ~20 % faster
today than in the 10-19 session — that offset applies to both files and is why
this round re-measured its own baseline instead of reusing 45.69 µs).

#### Official benchmark entry (final confirmation, not a hand-rolled timer)

```
cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend \
  /workspace/venv/bin/python -m pytest benchmarks/ops/bench_pool.py \
  -k "vgg-block and float16" -sv
```

```
=== [msprof] latency_us: 18.440001, roofline: {'Ratio(%)': 29.2885, 'Bandwidth(TB/s)': 1.75,
    'AI(Ops/Byte)': 1.403377652168274, 'Perf(TOps/s)': 0.7192988991737366,
    'Computility(TOps/s)': 5.063262462615967}
2 passed, 34 deselected  (vgg-block-float16 18.44 µs; vgg-block-bfloat16 38.72 µs — unchanged path)
```

Official Ratio 14.53 % → **29.29 %**, `performance_tops` 0.5354 → **0.7193** on the
same entry (`logs/round5/official_bench_vgg_f16.log`, run on the final artifact
`04013b96…`). The 18.44 µs is the `ManifestBenchmark` protocol
(`--launch-count=10`), 1.3 % above the 18.20 µs of the 20-launch protocol used for
the tuning decisions — same file, same direction. (The pre-cleanup artifact
measured 18.9 µs / 28.57 % on the same entry, i.e. the two runs bracket the same
value.)

#### 是否推广：**否**（不推广到 bf16/fp32 `max_pool2d`，也不动 `max_pool1d`/sincnet）

理由摘要（逐条见下）：①本轮任务范围只允许单 shape 落地，推广明确不在本轮范围；②bf16 代数可用但 `hivm.hir.vreduce` 无 bf16 形态；③其余 shape 的前置条件清单未逐项复验。

Reasons, in order of weight:

1. **The single shape already delivers 50.6 % (2.03×), far past the 3 % bar, but
   the promotion conditions were not part of this round's scope** — the task fixes
   one workload only, and the requirement is explicit that bf16/fp32 twins and
   `max_pool1d`/sincnet must not be touched this round. Recording the plan only.
2. **bf16 is algebraically eligible but structurally blocked**: `hivm.hir.vreduce`
   has no bfloat16 form on this toolchain — documented, not just measured:
   `docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1 lists bf16 as `×`
   (fp16/fp32/int16/int32/int64 only), and every `vreduce` in this round's IR
   census is `f16`. The path therefore returns `None` for bfloat16. A bf16
   promotion would have to widen to f32 first (`vcast` of the whole span), which
   needs re-measuring — it is not a free dispatch flip.
3. **Which dtypes/shapes are algebraically eligible**: any fp16/fp32 shape with
   `s_h == k_h`, `s_w == k_w`, `p == 0`, `d == 1`, `k*out == in` on both axes, and
   a feasible `(ch, bh)`. Prerequisites that must be re-checked per shape before
   promoting: (a) `ch | C` with `ch*elem_bytes % 32 == 0` (16 B / 48 B lanes fault
   the device); (b) the live-UB budget (`2·ch·rows·w·eb + …` ≤ 240 000 B, and the
   measured `bl ≤ 96`-style ceiling: the largest tile that ran bit-exactly here is
   `ch·rows = 512`, a 2.2× larger tile was not attempted); (c) `bh | out_h`;
   (d) the innermost lane of *each* reduce must be wide (`ch ≥ 64` fp16 for the W
   reduce, `out_w·ch` for the H reduce) — this is what killed r5-W;
   (e) the `k*out == in` exact-partition identity (resnet-stem `112→56` with
   `k 3, s 2, p 1` is *not* disjoint, so it can never use this path; alexnet-ceil
   `55→27` has a padded last window, likewise excluded); (f) for a 1-D analogue
   (`max_pool1d`/sincnet `3·1365 = 4095 ≤ 4096`) the existing group path already
   occupies this design point, so the 2-D span path brings nothing new there.

---

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---|---|---|
| _max_pool2d_kernel | vgg-block-float16 | 36.86 | 18.20 | r5_span |

Baseline = the round-4 artifact measured in this session
(`profiles/round5/baseline_r5`, sha256 `e4dcd670…`, byte-identical to
`history_version/_max_pool2d_kernel_round4_pre_round5_20261023.py`).
Final = the round-5 artifact `perf_opt/_max_pool2d_kernel.py`, sha256
`04013b960556ed222a3489843c026dc64e6148b1d8f38b858d6edc4398f432ba`, measured 3×
independently on that file (`18.20 / 18.24 / 18.17 µs`; the table and
`perf_records.jsonl` use the median run `profiles/round5/final_span_r5_rep1`).
Delta **−50.6 % (2.03×)**, official-entry Ratio 14.53 % → 28.57 %.

Achieved rates (useful bytes = `(N·C·H·W + N·C·outH·outW)·2` = 16.06 MB):

| artifact | Task Duration µs | useful GB/s | DRAM traffic MB | DRAM GB/s | DRAM bytes per useful byte |
|---|---|---|---|---|---|
| round-4 phase path | 36.86 | 435.7 | 34.9 | 947 | 2.17 |
| round-5 span path | **18.20** | **882.4** | **17.0** | 934 | **1.06** |

Smoke (precision only, current artifact): `smoke-1x8x16x16-k3s2p1-{float16,bfloat16,float32}`
→ `max_diff = 0.0`, `exact = True`, `nan_ok = True` (`precision_pass = true` in
`workload_inventory.json`, `logs/round5/verify_r5.stdout`).
Skipped this round: the 8 other full workloads (unchanged path, round-4 records
archived in `logs/round5/perf_records_round4_full_history.jsonl`) — no performance
claim is made for them in this round.

### Skill Retrospective (round 5)

1. **The proxy must price the mechanism the profile names.** Round 5's first
   plan-selection attempt reused `_phase_score` verbatim (work × tail) and picked
   `(bh 1, ch 64)` — the *worst* measured config (its 19 shallow waves minimise
   the tail term). The measured curve is dominated by DMA *burst length*
   (2→4→8 rows = 27→22→18 µs), so the score needed a `1/min(1, burst/1 KiB)` term
   (value point **P**, evidence: the 5-point sweep in the table above; repro:
   `probe_span2d.py` + `logs/round5/sweep_span2d.sh`; stamp: 910B2 / `16f1ee4`).
   Generalisation: for any DMA-shaped kernel, the tiling objective must carry the
   burst-length term *explicitly* — a wave/tail-only proxy inverts the ranking.
2. **Two structures were probed before the production file was touched**, and the
   one that "looked" cheaper (r5-W: no transpose at all, 129 KB of UB) lost by 2×
   because its reduce innermost lane was 56 B. Same lesson as VP-2026-0367 but on
   a new axis: it is not "transposes are expensive" — it is "a narrow innermost
   *reduce* lane is expensive" (value point **D**: 73.60 vs 17.97 µs, `scalar`
   31.2 vs 6.1 µs, same shape/dtype/stamp).
3. **Traffic amplification, not peak bandwidth, was the lever.** The delivered
   DRAM rate is identical (947 vs 946 GB/s) while the time halved — because bytes
   went from 2.17× to 1.06× the useful traffic. Future bottleneck rankings should
   read `read_main_memory_datas / useful_bytes` *first* (value point **P**;
   evidence in the two tables above; stamp as above). This also explains why the
   brief's "0.407 TB/s vs 0.753 TB/s ceiling" framing understated the headroom:
   the useful-byte rate was 435 GB/s while the *engine* was already at ~0.95 TB/s.
4. **Gap filed (compiler capability):** there is still no vector primitive for a
   cross-stride "gather a 2×2 window into a 4-element group axis"; the whole
   design exists because the group axis must be *manufactured* from
   reshape+transpose+two reduces. Already registered as CG-2026-0031/0033 — this
   round adds a data point: the 2-D analogue costs 2 transposes + 2 reduces per
   unit, and the "one reshape + one reduce" ideal remains unreachable
   (`TRAP-window-decimation-primitives`).
5. **Process observation:** the round-4 log's `## Final Performance Test Data`
   table could not be reused after the artifact changed, which forced the
   archive-and-reopen dance. The skill's rule (one `final` row per tune workload,
   reconciled against the *current* artifact) is right, but the log would be
   easier to audit if each round's final table were dated and stamped with the
   artifact hash *in its header* (value point **R**, proposed).

### Round-5 Bookkeeping Corrections (gate 4, 2026-10-23)

No kernel edit, no re-measurement, no latency value changed in this step. Two
mechanical-reconciliation defects were repaired so that `statectl gate 4`
returns `pass: true`:

1. `workload_inventory.json` — the 8 round-5 `kind: "skipped"` full workloads
   (`resnet-stem-{float16,bfloat16,float32}`, `vgg-block-{bfloat16,float32}`,
   `alexnet-ceil-{float16,bfloat16,float32}`) carried only `marks: ["bench"]`,
   while the gate derives `expected_kind = "skipped" if "skip" in marks`. The
   literal mark `"skip"` was **appended** to each of those 8 lists (the `bench`
   mark and every other field are byte-identical — reverting only the 8 inserted
   `"skip"` entries reproduces the previous file hash `fff3d8a8…` exactly).
   Their existing skip `reason` text was kept. The 3 smoke entries stay
   `kind: "smoke"` with `marks: ["bench"]`, and `vgg-block-float16` stays the
   single `kind: "tune"` workload (`marks: ["bench"]`). File hash is now
   `5f5b612168d87bfdba5857021be95a062845b5dc826ac7ae7a4f4f7ce1155a45`.
2. `perf_records.jsonl` — both rows spelled `artifact_path` as
   `perf_opt/_max_pool2d_kernel.py`, but the gate resolves a relative
   `artifact_path` against the record file's own directory (`perf_opt/`), so the
   `final` row resolved to the non-existent
   `perf_opt/perf_opt/_max_pool2d_kernel.py` and failed `S4-PERF-RECORDS-RECON`
   ("final 未测最终 perf_opt 文件"). The two spellings were normalised to
   `_max_pool2d_kernel.py` (the convention the archived round-4 rows use). Only
   the path string changed: `duration_us` (36.86 / 18.20), `candidate_id`
   (`baseline` / `r5_span`), `artifact_sha256` (`e4dcd670…` / `04013b96…`),
   `msprof_raw_path`, timestamps and all bandwidth counters are untouched, and
   the final hash still equals the current kernel file hash
   `04013b960556ed222a3489843c026dc64e6148b1d8f38b858d6edc4398f432ba`.
   Pre-fix copies of both artefacts are kept under
   `logs/round5/bookkeep/`.

Gate after the corrections: `{"action": "gate", "failures": [], "ok": true,
"pass": true, "stage": 4, "warnings": []}`.
