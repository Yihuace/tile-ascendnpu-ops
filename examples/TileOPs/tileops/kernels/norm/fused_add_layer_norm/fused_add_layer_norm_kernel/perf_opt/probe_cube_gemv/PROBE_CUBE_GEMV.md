# PROBE_CUBE_GEMV — Cube ones-vector GEMV row-reduce feasibility probe (FALN headroom)

Scope: bounded feasibility probe for moving FALN's row reductions off the
Vector dependency chain onto the Cube. **No delivery kernel / wrapper was
modified**; everything here is additive.

- Repository: `/mnt/workspace/tile-ascendnpu-ops`
- Toolchain: tilelang `0.1.2+a2d5db161aa9342c1ef05e71714f0921f35ae55b` (dev root
  `/mnt/workspace/tilelang-mlir-ascend`) + CANN 9.0.0 + Ascend 910B (device 0)
- Mode: `TILELANG_ASCEND_MODE=Expert` (both probe paths, one process)
- Timing discipline: `msprof op --launch-count=20 --warm-up=5`, median of
  `Task Duration(us)`; A and B kernels alternate **inside one python process**
  (two `msprof` invocations over the same process, one per kernel name), so
  neither path sees a different device state.
- Date: 2026-10-21

---

## 1. Verdict

| question | answer |
|---|---|
| Q1 Can `T.gemm` do a `(block_m, N) x (N, 1)` row-reduce in Expert mode (persistent `T.Kernel` + in-core `T.serial`, explicit `T.Scope("Cube")`) on CANN 9.0.0? | **YES — compiles, runs, numerically correct.** Even the *strict* form `size=[M, N, 1]` with `M in {1,2}` and a 1-wide GEMV output works (no 16-row / 16-col padding required). `M=1/2` GEMV does **not** hit CG-2026-0010 (that gap is Developer-mode only). |
| Q2 Is the Cube GEMV faster than `T.reduce_sum` (msprof Task Duration)? | **NO — it is slower in every measured configuration.** Per-tile marginal cost at `block_m=2, N=4096`: **1.17 µs (Cube) vs 0.637 µs (Vector) = 1.8x slower**; at `block_m=1, N=8192` with mandatory K-chunking: **4.27 µs vs 0.637 µs = 6.7x slower**. Single-shot (one launch, one row-block) it is 1.14x slower at N=4096 and 1.45x slower at N=8192. |
| Q3 Worth integrating into FALN? | **NO.** Verdict for the headroom question: **INFEASIBLE** (the API path is proven feasible; the *performance* premise is falsified). Details and the one remaining theoretical variant in §5. |

**One-line takeaway:** the per-row reduction is **not** Cube-bound work at
`block_m in {1,2}` — a 1-wide GEMV wastes 15/16 of the 16-row Cube fractal and
is bound by MTE1 (L1→L0, ratio 0.74) with the Cube pipe only 32% busy. The
Vector `T.reduce_sum` already runs at 0.637 µs per `(2,4096)` tile, i.e. 26% of
FALN's 2.41 µs/tile — and the Cube cannot even reach that without first
materialising its input in GM.

---

## 2. What was actually built and run

`probe_cube_gemv.py` — one file, four kernels:

| kernel | path | structure |
|---|---|---|
| `vec_reduce_main` | A (baseline) | `T.Kernel(1, is_npu=True)`, `T.Scope("Vector")`, fp16 GM→UB, `T.vcast`→fp32, `T.reduce_sum(dim=1)`, (M,1) fp32→GM. Semantics identical to the delivered FALN reduce step. |
| `cube_gemv_main` | B (Cube) | `T.Scope("Cube")`, `T.load_nd2nz` x→L1 and ones→L1, `T.gemm(..., size=[M,N,PN])` into `T.alloc_L0C([M,PN],"float32")`, `T.store_fixpipe(..., enable_nz2nd=True)` → GM. |
| `vec_reduce_multi_main` | A, persistent | same, wrapped in `for i in T.serial(ntasks)` (persistent grid-stride form, single core) — measures the **marginal** per-tile cost after the ~4 µs launch floor. |
| `cube_gemv_multi_main` | B, persistent | same, with `kc` (K-chunk) parameter for the L1 budget. |

Correctness reference: `x.cpu().double().sum(dim=1)` (float64).
`run_probe.py` drives msprof and parses `OpBasicInfo_*.csv`.

### Compile evidence (success)

```
COMPILE vec  M=2 N=4096: OK
COMPILE cube M=2 N=4096 PN=1 KC=0 bt=0: OK
COMPILE vec  M=1 N=8192: OK
COMPILE cube M=1 N=8192 PN=1 KC=0 bt=0: OK
```

### Compile / run evidence (failures — L1 budget, not API)

Whole-row L1 tiles at `N=8192` overflow the 512 KB L1 because the **NZ layout
pads every tile to 16 rows**: one `(bm, 8192)` fp16 tile occupies
`16 x 8192 x 2 B = 256 KB` regardless of `bm`, and the `ones` tile is another
256 KB. Two different diagnostics for the same root cause:

- `bm=1` → clean capacity diagnostic:
  `error: cbuf overflow, requires 6291456 bits while 4194304 bits available!
  (possible reason: tiling basic block is too large or block number is more than
  what user expect due to multi-buffer feature is enabled and some ops need
  extra local buffer.)`
  (= 768 KB = 256+256 KB payload × 1.5 auto-multi-buffer — matches the 16-row
  padding hypothesis exactly; log `logs/err_bm1_n8192.log`)
- `bm=2` → **compiler crash, no diagnostic**:
  `RuntimeError: NPU IR opt failed: PLEASE submit a bug report to
  https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.`
  (`bishengir-compile`, `--enable-auto-multi-buffer=true`; log
  `logs/err_bm2_n8192.log`). No pass name in the output, so it
  cannot be attributed to a specific pass from the CLI text.

Workaround proven: K-chunking (`KC in {512, 1024}`) makes `N=8192` compile and
run correctly (`CHK multi M=128 N=8192 bm=16 ntasks=8 kc=512: cube_err=9.2e-05`).

---

## 3. A/B latency — single GEMV per launch (the "N=4096 bm=2" / "N=8192 bm=1" two-tier ask)

`msprof op --launch-count=20 --warm-up=5`, median µs, 20/20 samples, both
kernels alternating in one process. `probe_results.json`.

| variant (B) | 2 x 4096 | 1 x 8192 |
|---|---|---|
| **A: `T.reduce_sum` (Vector)** | **4.05** (3.95-4.12 over runs) | **4.13** (4.07-4.13) |
| B: ones as column `(N,1)`, `b_transpose=False` | 117.27 (**29x slower**) | 252.97 (**61x slower**) |
| B: ones as row `(1,N)`, `b_transpose=True` (PN=1) | 4.69 (1.14x) | 5.96 (1.45x) |
| B: PN=16 padded GEMV output | 6.26 (1.58x) | 8.39 (2.06x) |
| B: row-ones + K chunk 512 | 5.59 (1.41x) | 7.81 (1.90x) |

Two things fall out of this table:

1. **The ones layout is worth 25–40x.** A `(N,1)` column vector is fetched by
   ND2NZ as N single-element strided rows; the `(1,N)` row + `b_transpose=True`
   form is one contiguous burst. Any future GEMV row-reduce on this toolchain
   must use the row form (this alone would look like an "API is unusable"
   result if measured only in the naive layout).
2. At one-launch granularity both paths sit on the ~4 µs launch floor, so this
   table cannot separate the reduce cost from fixed overhead. Hence the
   persistent measurement below.

### Persistent (multi-tile) measurement — marginal per-tile cost

128 tasks per kernel, single core, `T.serial` in-core loop. `probe_results_multi.json`.

| rows/task (bm) | N | Cube K-chunk | A: Vector µs/tile | B: Cube µs/tile | Cube vs Vector |
|---|---|---|---|---|---|
| 2 | 4096 | whole row | 0.637 | **1.166** | **1.83x slower** |
| 2 | 4096 | 512 | 0.641 | 2.29 | 3.6x slower |
| 1 | 8192 | whole row | 0.637 | *compile fail (cbuf overflow)* | — |
| 1 | 8192 | 512 | 0.637 | 4.27 | 6.7x slower |
| 1 | 8192 | 1024 | 0.637 | 3.32 | 5.2x slower |
| 4 | 4096 | whole row | 0.913 (0.228/row) | 1.276 (0.319/row) | 1.4x slower |
| 8 | 4096 | whole row | *ub overflow (256 KB > 192 KB)* | 1.429 (0.179/row) | — |
| 16 | 4096 | whole row | not possible (UB budget) | 2.156 (0.135/row) | — |
| 16 | 8192 | 512 | not possible | 7.04 (0.44/row) | — |

Per-row view (µs/row): Vector `bm=2` 0.319, Vector `bm=4` 0.228,
Cube `bm=2` 0.319…0.60, Cube `bm=4` 0.319, Cube `bm=8` 0.179, Cube `bm=16` 0.135.

So the Cube only wins **per row** once ≥8 rows share one GEMV (amortising the
16-row fractal and the per-gemm MTE1 cost) — a regime FALN's `block_m in {1,2}`
never reaches.

### max_abs_err (vs float64 reference)

| case | A: Vector | B: Cube |
|---|---|---|
| M=2, N=4096 | 5.7e-06 | 5.2e-06 |
| M=1, N=8192 | 3.3e-06 | 4.2e-05 |
| M=256 (multi, bm=2, N=4096) | 1.1e-05 | 3.1e-05 |
| M=128 (multi, bm=16, N=4096 / 8192) | — | 2.7e-05 / 9.2e-05 |

All ≤ 1e-4, far inside the 1e-3 fp32-accumulation tolerance. **Numerics are not
the blocker** — the Cube accumulation is as accurate as the vector reduce.

### Pipe profile (msprof `PipeUtilization`, medians; `probe_pipe_metrics.json`)

| run | aic_time | aic_cube_ratio | aic_mte1_ratio | aic_mte2_ratio | aiv_time | aiv_vec_ratio | aiv_mte2_ratio |
|---|---|---|---|---|---|---|---|
| A vec bm=2 N=4096 | 0.37 | – | – | – | 79.3 | 0.590 | 0.589 |
| B cube bm=2 N=4096 | 152.0 | **0.316** | **0.744** | 0.512 | 0.38 | – | – |
| A vec bm=1 N=8192 | 0.34 | – | – | – | 78.8 | 0.608 | 0.586 |
| B cube bm=1 N=8192 KC=512 | 544.0 | 0.223 | 0.508 | **0.995** | 0.42 | – | – |
| B cube bm=16 N=4096 | 14.7 | 0.179 | 0.482 | 0.782 | 0.43 | – | – |
| B cube bm=16 N=8192 KC=512 | 53.7 | 0.136 | 0.310 | 0.948 | 0.42 | – | – |

Reading:

- **M=1/2 Cube utilisation is genuinely poor**: the Cube pipe is only 22–32%
  busy; the limiter is MTE1 (L1→L0A/L0B, 0.74) — i.e. the 16-row NZ padding
  moves 8x the useful data through L1. This is the mechanistic reason the GEMV
  loses at small `block_m`, not a tiling parameter.
- At `bm=1, N=8192, KC=512` the MTE2 pipe is **99.5% saturated** yet delivers
  ~4 GB/s effective — 1-row ND2NZ descriptors (512 elements per 16-row
  fractal) are a pathological transfer shape. K-chunking is what makes N=8192
  compile, and it is also what kills its bandwidth.
- The Vector path is balanced (vec 0.59 / MTE2 0.59) — no single pipe saturated,
  which is why it lands at 0.637 µs/tile.

---

## 4. Why it loses (mechanism, quantified)

Delivered FALN (`perf_opt/FusedAddLayerNormFwdOp.py`, llama-8b-prefill fp16):
2048x4096, `block_m=2`, 62.65 µs total, 1024 logical tiles over 40 Vector cores
→ **2.41 µs per tile**. The isolated reduce is **0.637 µs = 26% of the tile**.
Even a *free* Cube reduce could therefore only buy ≤26% — and it is not free:

1. **Per-tile GEMV overhead.** 1.17 µs vs 0.637 µs → **+0.53 µs/tile (+21% of
   tile time)**: the probe's B path would make FALN ~21% *slower* before any
   integration cost.
2. **Cube cannot read UB.** `docs/Tilelang.language/内存操作/` documents exactly
   two data paths — `T.load_nd2nz` (GM→L1) and `T.store_fixpipe` (L0C→GM). There
   is **no documented UB→L1 path**, so FALN's `s` (which lives in UB after the
   residual add) must be written to a GM workspace and re-read into L1.
3. **The centred two-pass variance needs the mean first.** FALN computes
   `sum(s)` → mean → `d = s - mean` → `sum(d^2)`. The second reduce's input
   `d^2` exists only in UB, so the Cube pass needs *another* GM round-trip of
   4 B/elem: 32 KB written + 32 KB read = 64 KB per 2-row tile, ≈1.0 µs at the
   measured ~60 GB/s per-core MTE2 — **~2x the entire theoretical saving**
   (0.37-0.53 µs/tile), before the fixpipe/L0C round-trip and the extra sync
   points are counted.
4. **`sum(x) + sum(residual)` trick only fixes the first reduce.** The first
   summation *can* be done Cube-side directly from GM (two accumulating GEMMs
   over `K=2N`, or one GEMM over a concatenated `[x | residual]` axis), avoiding
   the GM materialisation of `s` — but the second reduce still needs `d^2`.

Net: **the headroom direction is falsified at `block_m ≤ 4`**; no integration
should be attempted on this basis.

---

## 5. If integration were still wanted: the only remaining variant, and its cost

Not recommended, recorded for completeness:

- **Row-batched GEMV (`bm = 16` on the Cube side, `bm = 2` on the Vector side,
  8 logical tiles per Cube call).** Per-row cost 0.135 µs vs 0.319 µs
  (`bm=2`) → save ~0.18 µs/row ≈ 0.37 µs per 2-row tile ≈ **15% of FALN tile
  time** — *if* the GM round-trip of `d^2` were free, which it is not (§4.3).
  It also breaks FALN's persistent structure: the Vector pass would have to run
  in two phases (rows 0-15 statistics → normalise rows 0-15), i.e. a
  `block_m=16` residency for `stage`/`x_f32`/`sq_f32` = 16x4096x(2+4+4) B =
  640 KB ≫ 192 KB UB. **UB-blocked, not Cube-blocked.**
- Budgets if someone retries: L1 512 KB, but usable per-tile payload is
  `16 x N x 2 B` per NZ tile (N=4096 → 128 KB per tile, N=8192 → 256 KB per
  tile, so N=8192 *must* be K-chunked); L0C `(bm, PN)` fp32, one 16x16 fractal
  = 1 KB; UB unchanged (≈192 KB effective / 1,572,864 bits).

**Cheaper alternative the probe did surface (recommended to the parent session,
no Cube involved):** the whole win is *row batching*, and the pure Vector path
gets it too — `bm=4, N=4096` measures **0.228 µs/row vs 0.319 µs/row at bm=2
(−29% reduce cost)** and needs only 96 KB UB (fp16 stage 32 KB + fp32 64 KB).
The current `block_m` cap of 2 comes from the conservative
`20 B/elem` multi-buffered budget (`_UB_MAX_ELEMS = 9216`), not from the
hardware. Re-deriving that budget (or fusing buffer roles so `sq_f32` is not a
second full plane) is a higher-expected-value direction than the Cube GEMV.
Caveat: this is a reduce-only measurement; FALN also holds `w1/b1` and a `stage`
plane, so a full-kernel `bm=4` needs the UB diet to actually fit
(full-kernel `bm=4` needs roughly 4x4096x(4+4+2) B = 160 KB of data planes
plus 32 KB of `w1/b1` affine buffers ≈ 192 KB, i.e. exactly at the 192 KB
capacity — *plausible but unproven*, and it must survive the ×1.10-1.12
BishengIR base-allocation overhead recorded in CG-2026-0008, so a UB diet is
almost certainly required).

---

## 6. Capability-gap bookkeeping (recommendation, no edits made)

`.agents/evolution/capability-gaps.md` was **not modified** by this probe. Two
observations and what I recommend the parent do:

1. **Not CG-2026-0010.** CG-2026-0010 is the *Developer*-mode persistent+gemm
   runtime crash. Expert mode is confirmed healthy here: persistent `T.Kernel`
   + in-core `T.serial` + `T.gemm` + `T.load_nd2nz` + `T.store_fixpipe` all run
   on CANN 9.0.0 with no crash. This probe is a *positive* datapoint for the
   documented workaround, not a new occurrence.
2. **New (undocumented) behaviour worth one line of evidence**, currently *no
   existing entry covers it* (checked `cbuf`, `overflow`, `bishengir-compile`,
   `crash backtrace` hits: CG-2026-0008 is UB-side, CG-2026-0002 is L1→L1 copy,
   CG-2026-0003 is the `ConvertHFusionToHIVM` `getResult(0)` crash family):
   **the L1 (cbuf) capacity model counts the NZ 16-row padding (`16 x N x 2 B`
   per tile, 8x–16x the logical payload), and the same overflow is reported
   cleanly for `bm=1` but as an unattributed `bishengir-compile` LLVM crash for
   `bm=2`.** Suggested home: a new entry (or a bullet under CG-2026-0008's
   "allocation visibility" theme extended to cbuf). I deliberately did not file
   it because the crash cannot be attributed to a pass from the CLI text, and
   because it is a *diagnostics/L1-budget* issue with a working kernel-level
   workaround (K-chunk), not an algorithm blocker.

---

## 7. Artifacts and reproduction

```
perf_opt/probe_cube_gemv/
  probe_cube_gemv.py        # A/B + persistent variants, --mode compile|check|bench|multi
  run_probe.py              # msprof op driver + CSV parser -> probe_results*.json
  probe_results.json        # single-shot A/B (4 layouts x 2 shapes)
  probe_results_multi.json  # persistent per-tile A/B (rebuilt from profiles/)
  probe_pipe_metrics.json   # PipeUtilization medians
  profiles/                 # raw msprof dirs (one per tag/shape/path)
  logs/                     # msprof + kernel logs (incl. err_bm1_n8192.log)
  PROBE_CUBE_GEMV.md        # this report
```

Reproduce (all NPU commands through the workspace lock):

```bash
cd /mnt/workspace/tile-ascendnpu-ops && source /mnt/workspace/tileops_env.sh
P=examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/perf_opt/probe_cube_gemv

# compile only (both paths, both shapes)
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 1800 -- \
    python $P/probe_cube_gemv.py --mode compile

# correctness vs float64 reference
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 1800 -- \
    python $P/probe_cube_gemv.py --mode check --pn 1 --bt 1

# single-shot A/B latency (msprof, alternating in one process)
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 3000 -- \
    python $P/run_probe.py --configs base,bt1,pad16,kc512

# persistent per-tile A/B latency
.agents/tools/npu_lock.sh --wait 900 --wait-idle 60 --timeout 3000 -- \
    python $P/run_probe.py --mode multi
```

### Probe caveats

- One core only (`T.Kernel(1)`); the probe isolates the reduce, it does not
  model the 40-core persistent split. Per-tile numbers are therefore the right
  comparison and the wrong absolute kernel time.
- `msprof op` Task Duration includes the ~4 µs single-launch floor, which is why
  the single-shot table compresses every variant to 4–8 µs. The persistent
  table (§3, 128 tasks) is the decision-grade one.
- The strict (unpadded) `size=[M,N,1]` form was verified at **one GEMV per
  launch** and in the persistent loop; it was *not* exercised with
  `block_m > 2` on the strict form (only the `bm=4/8/16` padded-`bm` form).
- `PN=16` and `KC` sweeps show the shape sensitivity is real (1.14x → 2.06x),
  so an "it compiles" result with a different ones layout would be misleading;
  the layout is called out explicitly in every table.
