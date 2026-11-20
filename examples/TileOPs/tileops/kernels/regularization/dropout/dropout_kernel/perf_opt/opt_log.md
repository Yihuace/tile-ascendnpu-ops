# Dropout Stage 4 — Round 9 (declared-roofline acceptance, bit-exact instruction fusion)

- project: TileOPs
- operator: `_make_dropout_kernel` (DropoutOp, `tileops/ops/regularization/dropout.py`)
- kernel dir: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel`
- final candidate: `perf_opt/_make_dropout_kernel.py`
  sha256 `1c969ba4e4084ec0383d23e378dda44a00174d40d841213e423e766c724c5937`
- pre-round9 candidate (baseline of this round): `perf_opt/_r9_base.py`
  sha256 `c3f71a95854ca1711d7ab8f34908f5e4a8a54d19dc406b0566ff2a9731962572`
  (byte-identical to the R8 final; kept so the round-9 baseline rows have a stable artifact)
- measurement protocol (single source of truth): `msprof op --kernel-name=main
  --aic-metrics=Default,Roofline --launch-count=20 --warm-up=5 --dump=off`,
  `Task Duration(us)` from `OpBasicInfo_*.csv`. Everything below is kernel-only.
- predecessor round artifacts: `perf_opt/history/round9_opt_log.md`,
  `perf_opt/history/round9_perf_records.jsonl`,
  `perf_opt/history/round9_workload_inventory.json` (moved verbatim, not edited)

## 1. Acceptance criterion used in this round

The user switched acceptance to the **declared roofline** at the start of this round.

- Primary (B): `Ratio_B = ideal_declared / duration`,
  `ideal_declared = max(bytes / 1.75e12, flops / peak)` with
  `flops = N`, `bytes = 2 * N * elem_bytes`
  (`DropoutOp.eval_roofline`, `examples/TileOPs/tileops/ops/regularization/dropout.py`,
  lines 167-198). The numerator is implementation-independent; `peak` is the
  `computility_tops` reported by the same msprof Roofline dump.
- Secondary (A, diagnostics only): the pre-existing msprof Roofline ratio,
  `perf_tops / computility_tops`, whose flops count is inferred by the tool from
  the emitted instruction mix. A and B move in opposite directions when the op
  count is reduced, so both are reported side by side below.

## 2. Workload inventory (unchanged from R8; see `perf_opt/workload_inventory.json`)

benchmark source `examples/TileOPs/benchmarks/ops/bench_dropout.py::test_dropout_bench`;
six cases: three `smoke` (1 Mi elements, fp32/fp16/bf16 — precision only, no
performance records) and three `tune`:

| workload_id | dtype | N | block_size |
|---|---|---|---|
| `tokens-1k-hidden-4k-fp16-float16` (W1) | float16 | 4194304 | 8192 |
| `tokens-1k-hidden-10k-bf16-bfloat16` (W2) | bfloat16 | 10485760 | 8192 |
| `tokens-1k-hidden-4k-fp32-float32` (W3) | float32 | 4194304 | 8192 |

Tuning order (by measured burden): W2 (58 us) -> W1 (22 us) -> W3 (22.5 us).
All three were re-measured in this round; none was skipped.

## 3. Baseline (round 9, `perf_opt/_r9_base.py`, sha c3f71a95854ca1711d7ab8f34908f5e4a8a54d19dc406b0566ff2a9731962572)

| kernel_id | workload_id | Task Duration(us) | aiv_vec_time(us) | vec_ratio | mte2_ratio | mte3_ratio | scalar_time(us) | Ratio_A | Ratio_B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 21.896 | 19.054 | 0.9513 | 0.2476 | 0.1430 | 1.412 | 54.08 % | 43.78 % |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 58.151 | 54.468 | 0.9739 | 0.2152 | 0.0995 | 2.723 | 63.06 % | 41.22 % |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 22.491 | 19.603 | 0.9538 | 0.2723 | 0.1764 | 1.241 | 57.11 % | 85.25 % |

Mean baseline Ratio_B = (43.78 + 41.22 + 85.25) / 3 = **56.75 %**
(the reference values 43.8 / 41.2 / 85.1 % in the task statement reproduce).

## 4. Round 9 — iteration log

Phenomenon carried in from R8: `aiv_vec_ratio` 0.95-0.97 and DMA ratios
0.10-0.27, i.e. the Vector pipe is the serial critical path and DMA is fully
hidden. The only remaining lever is the number/cost of emitted vector ops.

### 4.1 Candidate branches (all derived from the same base `_r9_base.py`)

| branch | single change | file | outcome |
|---|---|---|---|
| A | fold `T.fill(h, off+seed)` into the counter `T.vadd` | `perf_opt/_r9_a_fillfold.py` | compiles, L0 pass, valid msprof, **winner** |
| B | drop the f32 staging cast on the fp16 storage-domain path (int32 -> fp16 in one `vcast`) | `perf_opt/_r9_b_cast1.py` | **blocked**: frontend rejects the cast (see 4.3) |

### 4.2 Branch A — how it was landed (deviation from the literal recipe, disclosed)

The literal recipe `T.vadd(lane, T.cast(off + seed, "int32"), h)` is **not
accepted**: `AscendBinaryOp.buildTirCall` (`tilelang/language/customize_npuir.py`,
lines 100-113) only treats `int`/`float` literals and `tir.expr.Var` as scalar
operands; anything else is routed to `_to_region`, which raises
`<class 'tilelang.tvm.tir.expr.Add'> has no attribute buffer`. A let-bound
`tir.Var` scalar (`with T.let(s, off + seed)`) **segfaults** the frontend
(rc=139), reproducing the earlier `probes/_v_idx1.py` finding.

The accepted equivalent is a one-element UB scalar operand:

```python
sbase = T.alloc_var("int32")          # allocated once, outside the tile loop
sbase[0] = off + seed                 # scalar store (1 lane), run-time value
T.vadd(lane, sbase, h)                # h[i] = lane[i] + (off + seed)
```

Probe: `probes/r9_probe_vadd4.py` (not committed) produced `bit_exact=True`
against `arange(N)+42`. Generated NPUIR per device module (see 4.4) now has
`vbrc` = 1 (the hoisted `tp_b` fill) instead of 2: the full-tile broadcast fill
is gone; the run-time base is a scalar store that emits no `hivm.hir.v*` op.

### 4.3 Branch B — blocked by the toolchain (evidence)

```text
error: 'hivm.hir.vcast' op currently don't support cast int32_t_to_half_rintmode
```

checked for every round mode exposed by the frontend:

| call | result |
|---|---|
| `T.vcast(h, md, round_mode="rint")` | `don't support cast int32_t_to_half_rintmode` |
| `T.vcast(h, md, round_mode="round")` | `don't support cast int32_t_to_half_roundmode` |
| `T.vcast(h, md, round_mode="floor")` | `don't support cast int32_t_to_half_floormode` |
| `T.vcast(h, md)` | `don't support cast int32_t_to_half_rintmode` |

So the two-step int32->f32->fp16 cast cannot be collapsed, and the f32 staging
buffer stays on the fp16 path. Branch B keeps the base behaviour and produced no
performance record (blocked branches are not written to `perf_records.jsonl`).
Compile log: `perf_opt/logs/round9/bitexact_c.log`.

Consequence for the 11 -> 9 target: only fusion (a) is implementable, so the
kernel lands at **10 logical vector ops** (11 physical: `vclamp` expands to
`vmin` + `vmax`), not 9. The measured latency gain is nevertheless 8.5-9.1 %
because the removed op is a full-tile broadcast (`vbrc`) that writes 32 KiB of
int32 into UB per tile, i.e. costlier than an average op.

### 4.4 Instruction-level evidence (TILELANG_DUMP_IR=TRUE, fp16, N=8192, bs=8192)

Per device module, `hivm.hir.*` op census
(`perf_opt/logs/round9/ir__r9_base.py.log`,
`perf_opt/logs/round9/ir__make_dropout_kernel.py.log`):

| op | base | final |
|---|---:|---:|
| `vbrc` (full-tile broadcast/fill) | 2 | 1 |
| `vadd` | 1 | 1 |
| `vmul` | 4 | 4 |
| `vshl` | 1 | 1 |
| `vxor` | 1 | 1 |
| `vmin` / `vmax` (from `vclamp`) | 1 / 1 | 1 / 1 |
| `vcast` | 2 | 2 |
| `varange` | 1 | 1 |
| vector ops total (excl. `vbrc`, `varange`) | 12 | 12 |
| full-tile `vbrc` count | 2 | 1 |

The `sbase[0] = off + seed` staging does not appear as a `hivm.hir.v*` op.

### 4.5 Candidate vs current best (B2 comparison table)

All numbers from `perf_opt/perf_records.jsonl` (round 9) and the same msprof
dumps; "current best" = the pre-round9 candidate (`r9_base`).

| workload_id | candidate | Task Duration(us) | delta | aiv_vec_time(us) | vec_ratio | mte2_ratio | mte3_ratio | scalar_time(us) | Ratio_A | Ratio_B | L0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W1 fp16 4M | current best `r9_base` | 21.896 | — | 19.054 | 0.9513 | 0.2476 | 0.1430 | 1.412 | 54.08 % | 43.78 % | pass |
| W1 fp16 4M | A `r9_final` | 19.907 | **-9.09 %** | 17.111 | 0.9521 | 0.2466 | 0.1527 | 1.266 | 55.31 % | **48.16 %** | pass |
| W2 bf16 10M | current best `r9_base` | 58.151 | — | 54.468 | 0.9739 | 0.2152 | 0.0995 | 2.723 | 63.06 % | 41.22 % | pass |
| W2 bf16 10M | A `r9_final` | 53.212 | **-8.49 %** | 49.665 | 0.9723 | 0.2451 | 0.1155 | 2.608 | 64.47 % | **45.04 %** | pass |
| W3 fp32 4M | current best `r9_base` | 22.491 | — | 19.603 | 0.9538 | 0.2723 | 0.1764 | 1.241 | 57.11 % | 85.25 % | pass |
| W3 fp32 4M | A `r9_final` | 20.536 | **-8.69 %** | 17.662 | 0.9517 | 0.3101 | 0.2280 | 1.184 | 58.47 % | **93.37 %** | pass |

Branch B produced no row (compile blocked). `vclamp` count and `mte` ratios are
unchanged in kind; `vec_time` drops by 10.2 / 8.8 / 9.9 % which is the mechanism
behind the Task Duration drop.

Winner for **every** tune workload: branch A (merged as the final file). No
workload regressed, so a single unconditional kernel is kept — there is no
shape/dtype conditional path, and no Python-level dispatch was added.

### 4.6 Merge / non-regression check

The winner is the whole-kernel change (one edit, no per-workload specialisation),
so "merge" is the identity. Re-measured on the final file for all three tune
workloads (table above, phase=final) and all three `smoke` workloads
(1 Mi elements, fp32/fp16/bf16) pass precision as part of
`perf_opt/logs/round9/L0_L1_all_final.log`.

### 4.7 Bit-exactness of the fusion (hard constraint)

`perf_opt/probes/round9_bitexact.py` runs the *same* tensor through
`_r9_base.py` and the final file and requires `torch.equal` on the raw output,
plus exact equality of the recovered 0/1 mask (`round(y / scale)`):

| case | input | out_equal | mask_equal | ones drop rate |
|---|---|---|---|---|
| W1 fp16 N=4194304 bs=8192 | ones | True | True | 0.499870 |
| W1 fp16 N=4194304 bs=8192 | randn | True | True | — |
| W2 bf16 N=10485760 bs=8192 | ones | True | True | 0.499928 |
| W2 bf16 N=10485760 bs=8192 | randn | True | True | — |
| W3 fp32 N=4194304 bs=8192 | ones | True | True | 0.499870 |
| fp16 p=0.75 N=1048576 (non-signed mask path) | ones | True | True | 0.749927 |
| fp16 tail N=4000000 bs=8192 | ones | True | True | 0.499869 |

`ALL_BIT_EXACT= True` (`perf_opt/logs/round9/bitexact_a.log`). Statistics are
therefore unchanged by construction: p=0.5 ones-ratio 0.49987/0.49993, exact
mask equality implies identical lag1-4 autocorrelation and identical block
chi-square (the hash itself is untouched — only how its counter base is
materialised changed).

### 4.8 Precision regression

| level | command | result |
|---|---|---|
| L0 | `python perf_opt/_make_dropout_kernel.py --level L0` | 22/22 pass |
| L0+L1+L2+Boundary | `python perf_opt/_make_dropout_kernel.py --level all` | L0 22/22, L1 12/12, L2 14/14, Boundary 9/9, `All check passed` |

Log: `perf_opt/logs/round9/L0_L1_all_final.log`.

### 4.9 Stop reason for this round

`success` for the declared-roofline target: mean Ratio_B = **62.19 %** >= 60 %.
Budget used: 3 experiments (A, B, A-as-final); well inside the 8-branch budget.

## Final Performance Test Data

Same final candidate for every row (`r9_final`, artifact
`perf_opt/_make_dropout_kernel.py`, sha
`1c969ba4e4084ec0383d23e378dda44a00174d40d841213e423e766c724c5937`).

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 21.896 | 19.907 | r9_final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 58.151 | 53.212 | r9_final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 22.491 | 20.536 | r9_final |

### Both acceptance views (final)

| workload_id | duration(us) | ideal_declared(us) | Ratio_B (declared, primary) | Ratio_A (msprof count, diagnostic) |
|---|---:|---:|---:|---:|
| tokens-1k-hidden-4k-fp16-float16 | 19.907 | 9.587 | 48.16 % | 55.31 % |
| tokens-1k-hidden-10k-bf16-bfloat16 | 53.212 | 23.968 | 45.04 % | 64.47 % |
| tokens-1k-hidden-4k-fp32-float32 | 20.536 | 19.174 | 93.37 % | 58.47 % |
| mean | — | — | **62.19 %** | 59.42 % |

Ratio_A rose slightly on all three workloads despite the op count dropping:
A is computed from the tool-inferred flops, and the counter-base materialisation
changed form. Ratio_A is reported for completeness only, as instructed.

### Excluded workloads

- `smoke-dropout-1m-float32` / `-float16` / `-bfloat16` (1 Mi elements):
  manifest label carries the `smoke` token -> precision-only. Not profiled.
  Precision: pass (final output is bit-identical to the R8 candidate, which
  passed these cases; also covered by L0/L1 at 1M-10M elements).

## 6. Skill Retrospective

Flow observations from this round:

- **P** (pattern, positive): replacing a full-tile `T.fill` + `T.vadd` pair by a
  runtime scalar staged in a **one-element UB buffer** and a broadcasting
  `T.vadd` is worth **8.5-9.1 %** kernel time on this device, i.e. far more than
  its 1/N share of the op count, because the removed `vbrc` writes a full tile
  (32 KiB int32 at bs=8192) into UB. evidence: `perf_records.jsonl` round 9
  (`r9_base` vs `r9_final`), NPUIR census `vbrc` 2 -> 1, `aiv_vec_time`
  19.054 -> 17.111 us; repro: `perf_opt/probes/round9_bitexact.py` plus the
  three-line diff in `perf_opt/_make_dropout_kernel.py`; toolchain: tilelang
  dev build at `/mnt/workspace/tilelang-ascend-npu-ops/build`, CANN 9.0.0,
  Ascend910B3, 2026-11-02.
- **D** (measured data, negative): `hivm.hir.vcast` has no int32->fp16 lowering
  (rint/round/floor all rejected), so the "f32 staging cast can always be
  removed when the mask is 0/1" assumption is false on CANN 9.0.0 / 910B3.
  evidence: `perf_opt/logs/round9/bitexact_c.log`; repro: the four rejected
  `T.vcast` calls listed in 4.3; toolchain stamp as above.
- **R** (process): the recipe in the task prompt (`T.vadd` with a `tir.Cast`
  expression operand) contradicts the frontend's operand contract; a 30-second
  probe before writing the branch would have avoided one compile cycle. The
  skill's existing `TRAP-expert-v-operands` entry already records the literal
  family restriction — it should also record the **accepted workaround**
  (one-element UB buffer + broadcasting vadd), which the entry currently lacks.
- **C** (case index): dropout, `examples/TileOPs/tileops/kernels/regularization/
  dropout/dropout_kernel`, round 9 — "declared-roofline acceptance flips the
  sign of op-count reductions": under Ratio_A removing a vector op can look
  neutral/negative while Ratio_B rises; report both.

No skill process change is proposed beyond the two pattern-library updates
above; the Stage-4 flow itself (baseline -> branch -> L0 -> msprof -> merge ->
final) ran as specified.

## 7. G3 admission check — "remove one PRNG op" direction (2026-11-03)

按 `_shared/standards/compiler-change-admission.md` 的顺序先做 G3（收益可测性），
避免又走"先动手后验证"。这一轮**没有产出候选 kernel**：结论是负收益，方向关闭。

**A/B 构造**（同一 workload `tokens-1k-hidden-4k-fp16` [1024,4096] fp16、bs=8192、
仅 kernel 变体不同；app `probes/prng_ab_app.py`，变体 `probes/_make_dropout_kernel_minus_op.py`）：
baseline = r9_final 的 5 个 int32 PRNG op；变体 = 删掉最后一个 `T.vmul(h, _B_I32, h)`
（**仅计时用**，掩码分布不合法，不得作为候选）。

| 变体 | Task Duration | fops | computility | Perf | **提交 Ratio(GM)** |
|---|---:|---:|---:|---:|---:|
| baseline（5 PRNG op） | 20.008 µs | 44.50 M | 4.0267 TOps/s | 2.2243 | **55.24 %** |
| minus_op（4 PRNG op） | 19.084 µs | 40.31 M | 3.9746 TOps/s | 2.1122 | **53.14 %** |

复算管线自检：baseline 复算 55.24% vs 报告 `ratio_pct` 55.31% ⇒ 复算可信
（`probes/g3_ab_ratio.py`，纯 CSV 解析，秒级）。

**判定**：

- Δ时延 **−4.62 %**（kernel 确实更快）——1 个向量 op 在本 workload 上的时延权重量级成立；
- Δfops **−9.42 %**（分子掉得约为时延的近 2 倍）；
- Δcomputility **−1.29 %**（指令混合向 MISC 偏移，峰值略降）；
- 净 ΔRatio **−2.10 个百分点**（55.24 → 53.14）⇒ **提交口径负收益**。

**结论**：在算力分支（AI×bw > computility）下，减向量 op 会让 fops 的跌幅大于时延的跌幅，
**提交 Ratio 反而下降**；这与 R9 `Skill Retrospective` 记的 "declared-roofline acceptance
flips the sign of op-count reductions" 是同一机制，本轮给出了该机制在第一档 workload 上的
单 op 量化值（≈ −2.1 点/op）。叠加 dropout 已贴向量地板（`aiv_vec_ratio` 0.952）——
"降时延但不减 fops"的路径也不存在 ⇒ **该方向关闭，不再排工**。
仅当将来出现**原生** i32→half 单指令（CG-2026-0027）时，才可能同时改善时延与 fops 口径。

