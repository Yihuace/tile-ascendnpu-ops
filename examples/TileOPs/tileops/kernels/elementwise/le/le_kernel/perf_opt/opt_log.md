# le_fwd_kernel Stage 4 Tuning Log

- operator: `le_fwd_kernel` (`LeFwdOp`, elementwise `a <= b`, bool output carrier = int8)
- project: `le`; scenario `optimize`; mode `full`
- baseline kernel (read-only): `le_fwd_kernel/le_fwd_kernel.py`
- deliverable: `perf_opt/le_fwd_kernel.py` (candidate `final` = `r3_op1`)
- benchmark source: `examples/TileOPs/benchmarks/ops/bench_le.py::test_le_manifest_bench`
- **primary metric**: `msprof op` `Task Duration(us)` (kernel-only, median of a
  `--launch-count=15 / --warm-up=5` run). No other timing basis is used.
- Roofline Ratio (report口径, BW = 1.8 TB/s):
  `ratio = (Σ aiv/aic r0+r1 read_cache_miss_allocate × 128) / (Task Duration(s) × 1.8e12)`
  — recomputed from the same `L2Cache_*`/`OpBasicInfo_*` CSVs by `roofline_ratio.py`.
- toolchain: `tilelang 0.1.2+302ad3c`, CANN 9.0.0, Ascend910 (40 Vector cores,
  `UB = 192 KB`), `TILELANG_ASCEND_MODE=Developer`.
- Stage-4 target: **≥ 60 %** Roofline Ratio per case ("reach target, not optimum").
- Resume note: rounds 1–3 were produced in an earlier session interrupted before
  `opt_log.md` / inventory `tuning_status` were written. This session verified the
  deliverable, collected the missing round-2/3 candidate records and all 9
  `phase=final` records, re-ran the L0/L1/L2 regression, and produced this log
  (same resume pattern as the sister `gt`/`ge` retros).

## Performance Test Data (baseline)

Workload inventory: `perf_opt/workload_inventory.json`, expanded from
`bench_le.py::test_le_manifest_bench` (2-input manifest; 3 manifest entries × 3 dtypes =
9 tune + 3 smoke = 12 cases). Classification: 3 `smoke`
(`smoke-1k` = manifest idx 0 fp32 + its fp16/bf16 siblings, precision-only), 9 `tune`.
`skipped`: none.

Baseline = the Stage-3 kernel (`../le_fwd_kernel.py`, sha `b48ad6a0…`) at its shipped
`block_size=1024` (`phase=baseline`, round 0).

| kernel (id suffix `::le_fwd_kernel::main`) | workload_id | dtype | shape (a \| b) | Task Duration(us) | mte2/mte3/vec/scalar | Roofline Ratio |
|---|---|---|---:|---:|---:|---:|
| `…/le_fwd_kernel.py` | hidden-state-prefill-float16 | fp16 | [2048,4096] \| [2048,4096] | 84.198 | .809/.329/.163/.122 | 22.14 % |
| `…/le_fwd_kernel.py` | hidden-state-prefill-bfloat16 | bf16 | [2048,4096] \| [2048,4096] | 91.078 | .726/.303/.262/.142 | 20.47 % |
| `…/le_fwd_kernel.py` | hidden-state-prefill-float32 | fp32 | [2048,4096] \| [2048,4096] | 94.738 | .807/.302/.168/.109 | 39.35 % |
| `…/le_fwd_kernel.py` | cnn-feat-broadcast-float16 | fp16 | [16,256,56,56] \| [256,1,1] | 611.248 | .934/.180/.053/.039 | 2.02 % |
| `…/le_fwd_kernel.py` | cnn-feat-broadcast-bfloat16 | bf16 | [16,256,56,56] \| [256,1,1] | 512.190 | .882/.197/.101/.057 | 2.42 % |
| `…/le_fwd_kernel.py` | cnn-feat-broadcast-float32 | fp32 | [16,256,56,56] \| [256,1,1] | 361.793 | .867/.256/.110/.074 | 7.57 % |
| `…/le_fwd_kernel.py` | row-broadcast-float32 | fp32 | [4096,4096] \| [1,4096] | 177.416 | .819/.329/.182/.085 | 21.02 % |
| `…/le_fwd_kernel.py` | row-broadcast-float16 | fp16 | [4096,4096] \| [1,4096] | 176.656 | .850/.313/.153/.081 | 10.55 % |
| `…/le_fwd_kernel.py` | row-broadcast-bfloat16 | bf16 | [4096,4096] \| [1,4096] | 176.716 | .765/.307/.272/.108 | 10.55 % |

Raw profiles: `profiles/baseline/round0/baseline_8ba8496a/<workload_id>_<hash>/`.

**Design estimate vs measured (D-2 deviation, DESIGN.md §1.6.0).** Design-time
lower bound (block_size 4096, fp32 representative, 40 Vector cores) vs measured baseline:

| representative | design bound | measured baseline | deviation |
|---|---:|---:|---:|
| hidden-state-prefill fp32 | ≈ 78 µs | 94.738 µs | 1.21× |
| cnn-feat-broadcast fp32 | ≈ 206 µs | 361.793 µs | 1.76× |
| row-broadcast fp32 | ≈ 155 µs | 177.416 µs | 1.14× |

The estimate correctly identified **emission domination** (every task = 3–4 Vector ops:
`vcmp` + `bool→f16` + `f16→i8`, plus a per-task broadcast `vbrc`); the residuals are the
per-task index divmod / `vbrc` serial dependency the model did not price. This is the
quantified motivation for the DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D tile.

## Final candidate result: DESIGN §1.6.3 adjudication

DESIGN §1.6.3 froze 主选 = #1 (per-row `(block_size,)` 1-D tile) and 备选 = #4
(`[R_blk, L]` 2-D tile for the broadcast classes), with the explicit adjudication rule
"备选 latency < 主选 × 0.97 才翻转". This Stage executed the adjudication:
**#4 wins decisively on every broadcast workload** (cnn-feat fp16 134→34.5 µs, row
fp16 51.9→32.4 µs at the round-2 candidate), while same-shape `hidden-state-prefill`
(no broadcast) keeps #1 (the 2-D path is not selected: `b_2d_col`/`b_2d_row` both false).
Result is written back to §1.6.3 (adjudication resolved in favour of #4 for the broadcast
class; #1 retained for the flat same-shape stream).

## Iteration Log

Each round starts from the current best; **every comparison table below is
candidate vs current best**, all durations from `msprof op` (median of 15 launches).

### Round 1 — base = `baseline` (block_size 1024)

| branch | optimization point | hidden fp16/bf16/fp32 | cnn fp16/bf16/fp32 | row fp32/fp16/bf16 |
|---|---|---:|---:|---:|
| `r1_op1` | per-dtype `block_size` (fp16/bf16 8192, fp32 4096) | 26.08 / 27.66 / 49.50 | 134.24 / 146.98 / 73.52 | 175.18 / 174.76 / 183.60 |
| `r1_op2` | r1_op1 + outer-broadcast operand hoist | 26.04 / 27.48 / 49.40 | 154.54 / 155.52 / 83.26 | 61.46 / 51.86 / 79.68 |

Task Duration(us), candidate vs baseline (84.20 / 91.08 / 94.74 | 611.25 / 512.19 / 361.79
| 177.42 / 176.66 / 176.72):

| branch | workload_id | Task Duration(us) | vs baseline | L0 |
|---|---|---:|---:|:--:|
| r1_op1 | hidden-state-prefill-float16 | 26.08 | +69.0 % | pass |
| r1_op1 | cnn-feat-broadcast-float16 | 134.24 | +78.0 % | pass |
| r1_op1 | row-broadcast-float32 | 175.18 | +1.3 % | pass |
| r1_op2 | hidden-state-prefill-float16 | 26.04 | +69.1 % | pass |
| r1_op2 | cnn-feat-broadcast-float16 | 154.54 | +74.7 % | pass |
| r1_op2 | row-broadcast-float32 | 61.46 | +65.4 % | pass |
| r1_op2 | row-broadcast-bfloat16 | 79.68 | +54.9 % | pass |

- **Observation**: the flat same-shape stream is MTE2-bound (`mte2 0.81–0.93`, low
  `vec`) and wants the largest UB-fitting tile → per-dtype `block_size` is the decisive
  round-1 win (hidden-state 84→26 µs). The broadcast classes are emission-bound
  (`mte2 0.88–0.93` but `mm_read_bw` only 1.4–4.6 GB/s) → the 1-D path stays at one row
  per task.
- **Winner**: `r1_op1` for the flat stream; `r1_op2` for row-broadcast. Conflict on
  cnn-feat (hoist slightly hurts). Both become the base for round 2 (2-D tile).
- **L0**: both branches pass all 9 tune workloads. No rollback.

### Round 2 — base = `r1_op2`; alternative #4 `[R_blk, L]` 2-D tile

`T.vcmp` has no auto-broadcast (CG-2026-0021), so the broadcast operand is materialised
to the full `(R_blk, L)` shape via `T.vbrc`; both `(1,L)→(r_blk,L)` (row) and
`(r_blk,1)→(r_blk,L)` (col) orientations are used.

| branch | optimization point | hidden fp16/bf16/fp32 | cnn fp16/bf16/fp32 | row fp32/fp16/bf16 |
|---|---|---:|---:|---:|
| `r2_op1` | #4 2-D tile, fixed UB budget (`_MAX_2D_ELEMS`, r_blk: fp16 2 / fp32 2) | 26.10 / 27.44 / 49.36 | 68.30 / 78.82 / 44.82 | 52.48 / 32.50 / 43.88 |
| `r2_op2` | #4 + dtype-aware 2-D budget (fp16/bf16 r_blk=4 on cnn L=3136) | 25.78 / 27.68 / 48.96 | 34.48 / 41.18 / 44.70 | 51.88 / 32.42 / 44.42 |

candidate vs current best (`r1_op2`), Task Duration(us):

| branch | workload_id | current best | candidate | Δ | L0 |
|---|---|---:|---:|---:|:--:|
| r2_op1 | cnn-feat-broadcast-float16 | 154.54 | 68.30 | +55.8 % | pass |
| r2_op1 | row-broadcast-float16 | 51.86 | 32.50 | +37.3 % | pass |
| r2_op1 | row-broadcast-bfloat16 | 79.68 | 43.88 | +44.9 % | pass |
| r2_op2 | cnn-feat-broadcast-float16 | 154.54 | 34.48 | +77.7 % | pass |
| r2_op2 | cnn-feat-broadcast-bfloat16 | 155.52 | 41.18 | +73.5 % | pass |
| r2_op2 | row-broadcast-float32 | 61.46 | 51.88 | +15.6 % | pass |

- **Observation**: the 2-D tile amortises the per-task emit chain and the broadcast
  load/`vbrc` over `R_blk` rows — the decisive structural win. Hidden-state is unchanged
  (no broadcast → 1-D path). `mte2_active_bw` on cnn-feat rises 1.5→24 GB/s.
- **Winner**: `r2_op2` (dtype-aware budget admits `r_blk=4` for fp16/bf16 at L=3136);
  it is the merged round-2 best.
- **L0**: all branches pass. No rollback.

### Round 3 — base = `r2_op2`; relaxed 1.5× 2-D UB inflation (native fp16/bf16)

| branch | optimization point | hidden fp16/bf16/fp32 | cnn fp16/bf16/fp32 | row fp32/fp16/bf16 |
|---|---|---:|---:|---:|
| `r3_op1` | staging copy lands in `a2` directly → relax `1.7×→1.5×` for native fp16/bf16 (row L=4096 admits `r_blk=4`) | 25.38 / 26.62 / 49.26 | 41.04 / 46.70 / 45.42 | 50.86 / 29.42 / 34.24 |

candidate vs current best (`r2_op2`), Task Duration(us):

| branch | workload_id | current best | candidate | Δ | L0 |
|---|---|---:|---:|---:|:--:|
| r3_op1 | row-broadcast-float16 | 32.42 | 29.42 | +9.3 % | pass |
| r3_op1 | row-broadcast-bfloat16 | 44.42 | 34.24 | +22.9 % | pass |
| r3_op1 | hidden-state-prefill-bfloat16 | 27.68 | 26.62 | +3.8 % | pass |
| r3_op1 | cnn-feat-broadcast-float16 | 34.48 | 41.04 | -19.0 % | pass |

- **Observation**: the 1.5× relaxation is a genuine `r_blk` unlock for row-broadcast
  (L=4096: `r_blk` 2→4), giving a large win; the apparent cnn regression is
  **measurement noise** — for cnn L=3136 `r3_op1` and `r2_op2` compute the *identical*
  `r_blk=4` and emit the identical `main2` (verified: both `max_2d_r≥4`, `256 % 4 == 0`),
  so the two files' cnn kernels are structurally the same. The same-file cross-run spread
  on these small broadcast kernels is ~10–20 % (see Retrospective, D).
- **Winner**: `r3_op1` — structurally ≥ `r2_op2` everywhere (identical for cnn/fp32,
  strictly larger `r_blk` for the fp16/bf16 row/broadcast cases). It is the merged
  current best and the final deliverable.
- **L0**: pass. No rollback.

### Merge / non-regression check

`r3_op1` = `perf_opt/le_fwd_kernel.py` (sha `62cde013…`) is the single merged
`@T.prim_func` factory: per-dtype `block_size` + broadcast hoist + alternative-#4 2-D
tile + dtype-aware 2-D budget + 1.5× native-fp16/bf16 inflation. All 9 tune workloads
re-measured on the merged file (`phase=final`), all smoke/`--level all` precision passes.
No `merge_blocked`.

### Round 4 (20261022) -- real-footprint UB budget + the r_blk 5/6/7 ladder

**Phenomenon (same-session re-measure of the shipped deliverable, `r3_op1_ref`).**

| workload_id | Task Duration(us) | mte2 | vec | mm_read_bw GB/s (per core) | diagnosis |
|---|---:|---:|---:|---:|---|
| row-broadcast-bfloat16 | 34.26 | 0.392 | **0.724** | 25.0 | Vector-bound (bf16 vcmp/vcast throughput) |
| cnn-feat-broadcast-float16 | 43.84 | **0.799** | 0.252 | 14.4 | MTE2-issue-bound; achieved read BW is 2.5x below row-fp16 (35.4) |
| cnn-feat-broadcast-bfloat16 | 41.70 | 0.495 | 0.514 | 15.3 | both pipes ~sum to 1.0 -> no DMA/Vector overlap |
| cnn-feat-broadcast-float32 | 44.24 | 0.613 | 0.436 | 29.5 | already >= 60 %, but `r_blk=2` is far from the UB ceiling |
| row-broadcast-float16 | 28.94 | 0.708 | 0.487 | 35.4 | already >= 60 % |
| row-broadcast-float32 | 50.92 | 0.735 | 0.345 | 35.5 | already >= 60 % |

**Hypothesis under test (conductor-supplied, cross-op evidence from ge/eq).** The 2-D budget
still divides by a blanket 1.5x (fp16/bf16) / 1.7x (fp32) auto-multi-buffer factor. The native
tile footprint is exactly `2*cmp_bytes + 4` B/elem (`a2 + b2 + cond + yf + y8`), so at inflation
1.0 the tile ladder should open up to `r_blk` 5/6/7. Mechanised arithmetic (UB = 196,608 B,
`run = outer[-1]`): cnn-feat fp16/bf16 (run=256) 4 -> 7 (needs a masked tail, 256 = 36x7+4);
row fp16/bf16 (run=4096) 4 -> 5 (tail 1); **cnn-feat fp32 (run=256) 2 -> 4** (256 % 4 == 0, no
tail); row fp32 2 -> 3.

**Branches.**

- `r4_op1` -- replace the 1.5x/1.7x divisor with the real footprint:
  `max_2d_elems = (UB - staging - 16 KB reserve) / (2*cmp_bytes + 4)`, staging = the one live
  `b_in` (1,L) carrier (row mode only). Adopted below.
- The four *structural* probes (masked-tail `r_blk` 6 / row 5, per-core `b` preload, col
  `r_blk` cap 2, row `r_blk` cap 2) and the round-5 pair (direct `T.vcast(bool -> i8)`, row cap
  2 again) were executed on the **sister kernel `gt`**, whose 2-D scheduling block is
  byte-identical to this one: `diff` of the `run = outer[-1] .. main2` region between
  `gt/.../perf_opt/GtFwdOp.py` and `le/.../perf_opt/LeFwdOp.py` yields exactly one line
  (`T.vcmp(..., "gt")` vs `"le"`). All six were falsified there (A/B ties or worse) and are
  recorded in `gt/.../perf_opt/opt_log.md` rounds 4/5 with their `profiles/ab/` evidence; they
  are not re-measured here because the generated machine code differs only in the compare
  opcode. This kernel's own A/B re-verification covers the adopted change and the most
  noise-prone class.

**Candidate vs current best (`r3_op1_ref`) -- Task Duration(us)** (single msprof run,
`--launch-count=15 --warm-up=5`, plus the interleaved A/B verdict where it decides):

| workload_id | ref | r4_op1 | A/B verdict (3 pairs) | L0 |
|---|---:|---:|---|:--:|
| cnn-feat-broadcast-float32 | 44.24 | **41.66** | op1 vs ref **B_better -7.8 %** (44.88 -> 40.99 median, all pairs same direction) | pass |
| cnn-feat-broadcast-bfloat16 | 41.70 | 36.26 | **inconclusive** -- both arms bimodal 35.1/41.1 us (identical kernel) | pass |
| cnn-feat-broadcast-float16 | 43.84 | 40.90 | not A/B'd (identical `r_blk=4` kernel; no claim) | pass |
| row-broadcast-float16 | 28.94 | 27.86 | not A/B'd (identical kernel) | pass |
| row-broadcast-bfloat16 | 34.26 | 34.02 | not A/B'd (identical kernel) | pass |
| row-broadcast-float32 | 50.92 | 51.10 | ref vs final re-checked: **tie** (52.32 vs 52.31 median) | pass |

**Verdict.** `r4_op1` **adopted**: it changes exactly one class (`cnn-feat fp32`, `r_blk` 2 -> 4,
tile 128 -> 150 KB) and that class is a consistent A/B win; every other class keeps a
byte-identical kernel. The masked-tail / preload / half-tile probes belong to the sister
kernel's round and were rejected there (tie, 88x, +94 %, +30 % respectively).

### Round 5 (20261022) -- output-chain pass reduction

The single remaining idea was a 3 -> 2 pass output chain via a direct `T.vcast(bool -> int8)`
(the doc's cast table lists only `i1 -> f16/f32`). It was measured on the sister kernel
(identical scheduling block): the cast **does** compile and is bit-exact on tilelang
`0.1.2+a79a1316`, but BiShengIR re-materialises the same 2 B/elem intermediate, so the tile
footprint and the pass count are unchanged and the A/B is a tie (row bf16 34.04 vs 34.12 us;
cnn fp16 inconclusive in the bimodal band). Rejected; the doc-bound "no `i1 -> i8` pair" claim
is corrected in the sister log and in the family pattern-library entry.

**Merge and non-regression check (merged candidate = `r4_op1` = `r4_final`).** Every tune
workload was re-measured on the final file (`phase=final`, round 4) and smoke re-ran inside the
same `--level L0` pass; the passing classes were re-checked with the interleaved A/B protocol
(`row fp32`: tie) so the apparent cross-session drift on `cnn-*` is not attributed to the code
change. See "Final Performance Test Data" for the measurement-state note.

### Round 6 (20261022, second session) -- column-broadcast group-major order + predicated GM scalar load

**Phenomenon (re-measured this session).** Rounds 4/5 exhausted the tile ladder; cnn-feat fp16
was the headline miss (34.84 %). The eq/ge siblings had just isolated a *separate* structural
cost: at identical tile geometry the column-broadcast class is 1.65x the row-broadcast class and
the whole delta is the per-tile single-buffered `8 B GM -> UB copy -> T.vbrc -> T.vcmp` chain
(PL-1.21 §11). Reproduced here with `stall_probe.py --tag r6diag` (2 reps, msprof op
median-of-15) on the **round-4 deliverable**, same element count (12.845 M), same `L = 3136`,
same `r_blk = 4`:

| dtype | col (`cnn-feat-broadcast`) | row (`diag-row-3136`) | ratio | col mte2 active BW | row mte2 active BW |
|---|--:|--:|--:|--:|--:|
| fp16 | 34.38 | 22.05 | **1.56x** | 25.1 GB/s | 46.1 GB/s |
| bf16 | 35.24 | 29.34 | **1.20x** | 46.2 GB/s | 51.6 GB/s |

Same signature as the sibling (gt: 1.57x / 1.20x; eq: 1.65x): identical bytes, but the col arm's
MTE2 is *busy and slow* (`mte2 0.74` + `vec_wait 0.71` at fp16), i.e. a serialized scalar path.

**Candidate optimisation point ②** (ported from eq/ge, PL-1.21 §11 `r6op2`): ① broadcast-group
**major** tile enumeration with contiguous per-core blocks (`task = cid * num_local_tasks + i`,
`sub = task // R_lead`, `lead = task % R_lead`) so a core walks `R_lead` tiles sharing the same
b row block; ② predicate **only** the 8 B GM scalar load (`if i == 0 or lead == 0`), leaving
`T.vcast` and `T.vbrc` unconditional (predicating those is a documented double fault: wrong
across iterations + UB doubling, PL-1.21 §12).

**Branch `r6_op1`** = round-4 deliverable + the above (col path only; `task_a = 1,
task_b = num_kernels` for row).

**Candidate vs current best** (interleaved A/B, 2 pairs per arm, alternating order, run twice =
4 pairs per case, `profiles/candidate/r6ab6/` and `r6ab26/`; per-rep rows in
`perf_records.jsonl`):

| workload_id | current best (us) | r6_op1 (us) | Δ | mte2 | vec | vec_wait | GM→UB (KB) | L0 |
|---|--:|--:|--:|:--:|:--:|:--:|--:|:--:|
| cnn-feat-broadcast-float16 | 44.89 | **23.10** | **+48.5 %** | 0.806→0.599 | 0.240→0.503 | 0.783→0.543 | 640.25→637.31 | pass |
| cnn-feat-broadcast-bfloat16 | 45.58 | **31.93** | **+29.9 %** | 0.552→0.385 | 0.456→0.682 | 0.563→0.345 | 640.25→637.31 | pass |
| cnn-feat-broadcast-float32 | 40.60 | **38.35** | **+5.5 %** | 0.611→0.619 | 0.456→0.445 | 0.568→0.579 | 1277.25→1274.31 | pass |

**Bimodality caveat -- the one place where this operator reads worse than its sister.** The
`cur` arm above sat in the *slow* bimodal state for all 4 pairs (fp16 43.84-45.94 us,
`mte2 active BW 17-18 GB/s`), while the same binary measured 34.38 us in the diagnosis probe
40 min earlier (`profiles/diag/r6diag/`) and the gt sister measured 34.36 us in the same
session. So the honest improvement against the *fast* state is `34.4 -> 23.1 us (+33 %)`, not
+48.5 %; both are quoted. `r6_op1` measured 22.98-23.18 us in every arm, session and in the
final collection -> the candidate is stable and wins against both states. (Consistent with the
sibling finding that the slow state is the col stall amplified.)

**Non-regression A/B** (same protocol, 2 pairs):

| workload_id | current best (us) | r6_op1 (us) | Δ | L0 |
|---|--:|--:|--:|:--:|
| row-broadcast-float16 | 27.19 | 27.46 | -1.0 % (tie) | pass |
| row-broadcast-bfloat16 | 34.29 | 34.39 | -0.3 % (tie) | pass |
| row-broadcast-float32 | 50.28 | 50.55 | -0.5 % (tie) | pass |
| hidden-state-prefill-float16 | 26.03 | 26.21 | -0.7 % (tie) | pass |
| hidden-state-prefill-bfloat16 | 27.97 | 27.64 | +1.2 % (tie) | pass |
| hidden-state-prefill-float32 | 49.70 | 49.11 | +1.2 % (tie) | pass |

The hidden rows use the round-4 deliverable as arm A (`le_fwd_kernel_round4_op1.py`, byte
identical to the pre-round `LeFwdOp.py`; diff = the 45 round-6 lines only) and are the control
proving the 1-D branch was untouched: the `phase=final` hidden fp16 number (27.42 vs 24.36)
moves purely with session state.

**Verdict: ADOPT.** Col cases +33 %/+30 %/+5.5 % (fast-state basis), no row/1-D regression
outside the noise band. `perf_opt/LeFwdOp.py` := `r6_op1` (sha256 `478a247e…`), then
re-measured on every tune workload (`phase=final`, round 6, `r6_final`) + smoke.

## Autotune Log

No library autotune was used. The only tunable was `block_size`, set by a small manual
per-dtype scan in round 1 (one knob that folds to a compile-time constant); the decisive
gains were the structural 2-D tile + hoist, not a parameter walk.

## Roofline Ratio (report口径, `roofline_ratio.py`, BW = 1.8 TB/s) and target

| workload_id | baseline µs | final µs | improvement | ratio before (round 4) | ratio now (round 6) | ≥60 %? | raw final profile |
|---|---:|---:|---:|---:|---:|:--:|---|
| hidden-state-prefill-float16 | 84.198 | 27.420 | +67.4 % | 76.52 % | **67.98 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-float16_f48f537e/OPPROF_20261022172303_KRAWFBNQQRMRUNQR/main` |
| hidden-state-prefill-bfloat16 | 91.078 | 27.160 | +70.2 % | 67.79 % | **68.64 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-bfloat16_274d79ca/OPPROF_20261022172328_YXFLMRLRVYTLGIUI/main` |
| hidden-state-prefill-float32 | 94.738 | 49.440 | +47.8 % | 72.70 % | **75.41 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/hidden-state-prefill-float32_dac1dbdc/OPPROF_20261022172353_ZBQWEWFVQSPNGKHC/main` |
| cnn-feat-broadcast-float16 | 611.248 | 22.980 | +96.2 % | 34.84 % | **62.11 %** | ✓ **(new)** | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-float16_f1cddcf9/OPPROF_20261022172417_DPUDBWHYNBYKLOEH/main2` |
| cnn-feat-broadcast-bfloat16 | 512.190 | 31.860 | +93.8 % | 32.09 % | **44.80 %** | ✗ | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-bfloat16_f5a51caa/OPPROF_20261022172440_NZKUOCPRMRNJTRSB/main2` |
| cnn-feat-broadcast-float32 | 361.793 | 38.300 | +89.4 % | 69.42 % | **74.53 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/cnn-feat-broadcast-float32_2ff59659/OPPROF_20261022172503_INCGTDJLEVESFUOH/main2` |
| row-broadcast-float32 | 177.416 | 51.840 | +70.8 % | 69.32 % | **71.94 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-float32_7e902357/OPPROF_20261022172527_YIXUGENGZGWTJJUB/main2` |
| row-broadcast-float16 | 176.656 | 27.940 | +84.2 % | 70.36 % | **66.74 %** | ✓ | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-float16_410291d8/OPPROF_20261022172549_NWQADQVEKEWCPCDW/main2` |
| row-broadcast-bfloat16 | 176.716 | 34.300 | +80.6 % | 54.55 % | **54.36 %** | ✗ | `profiles/final/round6/r6_final_76cb60b1/row-broadcast-bfloat16_00d04600/OPPROF_20261022172612_KGJFZJDPNHJSKCQC/main2` |

**≥ 60 % target: 7 / 9 tune workloads** (was 6/9). **Newly passing: cnn-feat fp16,
34.84 % → 62.11 %** (lever ②, the col-broadcast stall). The two misses are bf16 cases
(cnn-feat 44.80 %, row-broadcast 54.36 %), both Vector-throughput bound (see next section).
The hidden fp16 ratio reads 67.98 % here vs 76.52 % in round 4 on a **byte-identical** 1-D
branch; the interleaved control A/B measures that class as a tie (-0.7 %), so this is session
state, not a regression.

**Measurement-state note.** `cnn-feat` is bimodal on this machine: the *same* binary measures
34.3 / 40.4 / 44.5 µs depending on session state. The round-4 and round-6 `final` rows are
single msprof runs; every accept/reject decision was made with the interleaved A/B protocol
(alternating order, even pair count; `profiles/ab/round4/`, `profiles/candidate/r6ab6/`,
`r6ab26/`, `r6hid6/`, per-rep rows in `perf_records.jsonl`). Where the two disagree the A/B
verdict is authoritative. Round 6 is the first candidate on this operator whose col arm stayed
inside 22.98-23.18 µs across every session and arm.

## Smoke precision and regression

- Smoke `smoke-1k` fp32/fp16/bf16 — precision-only (per the skill, smoke is never
  performance-tuned); no perf profile recorded. `workload_inventory.json`
  `precision_pass: true` for all three.
- Phase-3 regression on the deliverable `perf_opt/LeFwdOp.py` (round 6, sha `478a247e…`):
  `python3 perf_opt/LeFwdOp.py --level all` → **All check passed!**
  (L0 fourteen items, L1 functional matrix + UB-guard/dtype-reject + `bool-vbrc` int8-carrier
  col broadcast, L2, Boundary; `Summary max_diff per dtype: all 0`).
  Log: `perf_opt/logs/r6/final_all.log`. The same `--level L0` gate ran on the round-6 candidate
  (`logs/r6/L0_le_r6_op1.log`) and on the final file inside `run_experiments.py --phase final`,
  and the three smoke cases were re-checked bit-exactly with `bench.py --check`
  (`logs/r6/smoke_check.log`).

## Why ≥60 % is not reached for the remaining 2 workloads

The report ratio numerator is **GM read traffic only** (`read_cache_miss_allocate × 128`),
while every le case also writes the 1-byte bool carrier, so the ceiling is
`read/(read+write) · B_eff/1.8`. For the fp16/bf16 classes `read ≈ a`, `write = N` → read
fraction 2/3 → practical ceiling ≈ `0.667 · B_eff/1.8`. With the best total bandwidth seen
on this device ≈ 1.67 TB/s, the 2/3-read ceiling is ≈ 62 %; cnn-feat fp16 now sits at 62.11 %,
i.e. at that ceiling.

Round 6 removed the column-broadcast stall (cnn fp16 34.84 % → 62.11 %). Both remaining misses
are bfloat16 and Vector-throughput bound:

- **cnn-feat bf16 (44.80 %, was 32.09 %)**: the same lever gave it +29.9 % (45.58 → 31.93 µs
  A/B) but the schedule was already partially hiding the scalar chain (`mte2 0.40` vs fp16
  0.74), and it now runs at `vec_ratio 0.682` with `mte2 active BW 52.5 GB/s`. bf16 Vector ops
  are ~1.85x slower than fp16 for identical work, so the residual is `vcmp`/`vcast` throughput
  on the same 4-pass chain (`vcmp` + `bool→f16` + `f16→i8`; the bool carrier cannot be stored to
  GM directly, CG-2026-0022). Dropping the broadcast materialisation is blocked by CG-2026-0021
  (`T.vcmp` has no broadcast), an open compiler capability, and the `i1→i8` shortcut was
  measured as a tie in round 5.
- **row-broadcast bf16 (54.36 %)**: purely Vector-bound (`vec_ratio 0.723` vs fp16 0.487 at the
  same bytes and passes). Round 6 leaves it untouched by construction and the control A/B
  confirms the tie (34.29 → 34.39 µs, -0.3 %). Half tile was falsified in round 5 (+30 %),
  `r_blk=8` is UB-impossible (8×4096×8 B = 256 KB > 192 KB), and row fp16 on the same schedule
  reaches 66.74 %, isolating the gap to bf16 Vector throughput.

Per `_shared/standards/perf-feedback.md` §1 this is **not** filed as `[DESIGN_LIMIT]`: the
shortfall is a report-metric (read-only numerator) + hardware-cost (bf16 Vector throughput,
mandatory broadcast materialisation) boundary, not a contradiction of a specific DESIGN.md
assumption, and no structural `>2×` unreachable acceleration is demonstrated. No
`perf_feedback.md` produced.

## Final Performance Test Data

Deliverable `perf_opt/LeFwdOp.py` (candidate `r6_final` = `r6_op1`, sha256 `478a247e3a0f79f5…`),
all 9 tune workloads, `phase=final` round 6, one msprof run each on that single file in this
session. Both columns come from this same `perf_records.jsonl` (`phase=baseline`
block_size=1024 / `phase=final`). The round-4 `phase=final` rows are demoted to `phase=merged`
with a `note` when this set was collected (data unchanged; pre-round text in
`history_version/20261022_pre_r6/opt_log.md`).

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | hidden-state-prefill-float16 | 84.198 | 27.420 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | hidden-state-prefill-bfloat16 | 91.078 | 27.160 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | hidden-state-prefill-float32 | 94.738 | 49.440 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | cnn-feat-broadcast-float16 | 611.248 | 22.980 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | cnn-feat-broadcast-bfloat16 | 512.190 | 31.860 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | cnn-feat-broadcast-float32 | 361.793 | 38.300 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | row-broadcast-float32 | 177.416 | 51.840 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | row-broadcast-float16 | 176.656 | 27.940 | r6_final |
| examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py::le_fwd_kernel::main | row-broadcast-bfloat16 | 176.716 | 34.300 | r6_final |

## Skill Retrospective

- **Process rule (R, strong) — Phase 3 must materialise `opt_log.md` + inventory
  `tuning_status` + the Final table in the same session as the `phase=final` collection.**
  The earlier session measured rounds 1–2 and wrote candidate files for round 3, then
  stopped without the log / `tuning_status`; on resume the gate would have failed
  (`S4-OPTLOG`, `S4-OPTLOG-COMPTABLE`, `S4-PERF-RECORDS-RECON`) despite complete profiles.
  This is the same orphaning failure as the sister `ge`/`gt` retros → `recurring`.
  - evidence: this session's resume; `gt`/`ge` opt_log Skill Retrospective.
  - repro: n/a (process rule).
  - toolchain_stamp: tilelang `0.1.2+302ad3c`, Ascend910 40-Vector-core, CANN 9.0.0.
- **Validated optimization (P, positive) — DESIGN §1.6.3 alternative-#4 `[R_blk, L]` 2-D
  tile + dtype-aware UB budget for the broadcast classes.** Decisive: cnn fp16 134→34.5 µs,
  cnn bf16 147→41.2 µs, cnn fp32 73.5→44.7 µs, row fp16 51.9→32.4 µs, row bf16 79.7→44.4 µs
  (round 2). Mechanism: the per-task emit chain (`vcmp` + 2 `vcast`, plus the per-task
  broadcast load + `vbrc`) is amortised over `R_blk` rows; `T.vbrc` materialises the
  broadcast operand because `T.vcmp` has no auto-broadcast (CG-2026-0021).
  - evidence: round-2 candidate-vs-current-best table; `profiles/candidate/round2/r2_op2_*`.
  - repro: `perf_opt/le_fwd_kernel_round2_op2.py` (`main2` block + dtype-aware budget).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive) — outer-broadcast-operand hoist.** Materialise a
  fully outer-broadcast operand once per core (not once per task) → row-broadcast
  fp32 177→61.5 µs, fp16 184→51.9 µs, bf16 177→79.7 µs (round-1 `r1_op2`).
  - evidence: round-1 table; `profiles/candidate/round1/r1_op2_*`.
  - repro: `perf_opt/le_fwd_kernel_round1_op2.py` (`hoist_a`/`hoist_b`).
  - toolchain_stamp: same as above.
- **Validated optimization (P, positive) — relaxed (1.5×) 2-D UB inflation for the native
  fp16/bf16 path.** The staging copy lands directly in `a2` (no separate staging buffer),
  so the real footprint is `2·E + 4` B/elem; relaxing 1.7×→1.5× admits `R_blk=4` at L=4096
  (row-broadcast fp16 32.4→29.4 µs, bf16 44.4→34.2 µs; round-3 `r3_op1`).
  - evidence: round-3 table; `profiles/candidate/round3/r3_op1_*`.
  - repro: `perf_opt/le_fwd_kernel_round3_op1.py` (`_infl_2d`).
  - toolchain_stamp: same as above.
- **Measurement caveat (D) — same-file cross-run spread on the small broadcast kernels is
  ~10–20 %.** Evidence: `r2_op2` and `r3_op1` produce *identical* `r_blk` on cnn-feat
  L=3136 (both 4), yet cnn fp16 measured 34.5 µs (r2_op2) vs 41.0 µs (r3_op1) vs 35.6 µs
  (`final`); cnn bf16 41.2 vs 46.7 vs 36.8 µs. Consequence: per-workload winners among
  near-equal candidates on these kernels are within noise and should not override the
  priority-class decision; the ≥60 % verdict on cnn fp32 is reported as near-line rather
  than asserted.
  - evidence: perf_records `candidate`/`final` rows for cnn-feat fp16/bf16; `roofline_ratio.py`.
  - repro: re-run `bench.py` on the same file (n/a minimal).
  - toolchain_stamp: same as above.

- **D (measured constant, strong) — 2-D `[r_blk,L]` tile footprint and the right budget formula.**
  The tile is exactly `2*cmp_bytes + 4` B/elem; replacing the blanket 1.5x/1.7x
  "auto-multi-buffer" divisor with `(UB - staging - reserve) / (2*cmp_bytes + 4)` (staging = the
  one live `b_in` (1,L) carrier) gives cnn-feat **fp32 `r_blk` 2 -> 4 (150,528 B, 77 % of UB)** and
  a consistent **-7.8 % A/B win** (44.88 -> 40.99 µs, three pairs, same direction), while every
  fp16/bf16 class keeps a byte-identical kernel. Compiler cross-check on the sister kernel:
  BiShengIR demands exactly `2E+4` B/elem at the relaxed budget (200,832 B for 8x3136 fp16,
  262,240 B for 4x4096 fp32).
  evidence: `perf_records.jsonl` round 4 (`r3_op1_ref` vs `r4_op1`),
  `logs/r4/ab_op1_cnn-feat-broadcast-float32.json`.
- **D (measured, overturns a doc-bound negative) — `T.vcast(bool -> int8)` exists but is not
  profitable.** The vcast doc lists only `i1 -> f16/f32`, yet the direct cast compiles and is
  bit-exact on tilelang `0.1.2+a79a1316`; BiShengIR still materialises the same 2 B/elem
  intermediate, so the pass count and footprint are unchanged and the A/B is a tie. Measured on
  the byte-identical sister kernel (`gt`), see its `probe_cast_i1_i8.py` and logs.
- **P (pattern, negative) — the `r_blk` ladder above the divisor-4 plateau is exhausted.** Six
  structural branches (masked-tail `r_blk` 6/5, per-core broadcast preload, col cap 2, row cap 2,
  direct i1->i8) were measured on the byte-identical sister kernel and produced zero A/B wins on
  cnn-fp16 / cnn-bf16 / row-bf16; the only accept is `r_blk` 2 -> 4 for fp32 plus the
  footprint-derived budget. Cross-op transfer is legitimate here because the 2-D scheduling block
  of the two kernels differs by exactly one token (`"gt"` vs `"le"`).
  evidence: `gt/.../perf_opt/opt_log.md` rounds 4/5, `gt/.../perf_opt/profiles/ab/`.
- **R (process, weak) — same-session `cnn-feat` bimodality must be handled with `ab_test.py`, not
  single runs.** This session's `final` run landed in the slow mode (cnn bf16 44.48 vs 36.26 µs
  for the same binary); only the interleaved A/B proves the code did not change behaviour. Report
  the A/B verdict alongside any single-run table.
  evidence: `logs/r4/ab_op1_cnn-feat-broadcast-bfloat16.json`, this log's measurement-state note.
- **P (pattern, positive, cross-operator replicated) -- the col-broadcast stall is a separate
  structural lever from the tile ladder.** Measured at identical geometry (`stall_probe.py
  --tag r6diag`, same 12.845 M elements, same `L=3136`, same `r_blk=4`): col 34.38 us vs row
  22.05 us (fp16, 1.56x; mte2 active BW 25.1 vs 46.1 GB/s) and 35.24 vs 29.34 us (bf16).
  Fix = broadcast-group-major task order + **only** the 8 B GM scalar load predicated
  (`if i == 0 or lead == 0`), `T.vcast`/`T.vbrc` unconditional. A/B (4 pairs at fp16):
  cnn fp16 44.89 -> 23.10 us, cnn bf16 45.58 -> 31.93, cnn fp32 40.60 -> 38.35; ratio
  34.84 % -> 62.11 % (new pass), 32.09 % -> 44.80 %, 69.42 % -> 74.53 %.
  evidence: `perf_records.jsonl` round 6 (`r6` arm, per-rep), `profiles/candidate/r6ab6*`.
  repro: `perf_opt/stall_probe.py` + `perf_opt/le_fwd_kernel_round6_op1.py`.
  toolchain_stamp: tilelang 0.1.2+a79a1316 / CANN 9.0.0 / Ascend910 40-vector-core.
- **D (measured, strong) -- cross-session arm asymmetry is large enough to invert a naive
  reading: quote both bases.** In this operator's session the control arm sat in the slow
  bimodal state for 4/4 pairs (fp16 43.84-45.94 us, `mte2 active BW 17-18 GB/s`) while the
  *same binary* measured 34.38 us 40 minutes earlier and the gt sister measured 34.36 us at
  the same time. Reporting only the paired median would have claimed +48.5 %; the fast-state
  basis (+33 %) is the honest number. Rule: when the control arm's `aiv_mte2_active_bw`
  differs by >2x from the candidate's, report both bases and state which one the final table
  uses.
  evidence: `perf_records.jsonl` round 6 `cur` vs `r6` rows; `profiles/diag/r6diag/`.
- **R (process, weak, second occurrence) -- collecting a new `phase=final` set requires
  demoting the previous one to `phase=merged` (with a `note`) *before* the collection**, else
  gate 4 fails on both "exactly one final per workload" and the artifact-hash check. Same
  repair as round 4; archived the pre-round log to `history_version/20261022_pre_r6/`.
  evidence: `perf_records.jsonl` rows with `note = "relabelled to merged ..."`; gate-4 run.
