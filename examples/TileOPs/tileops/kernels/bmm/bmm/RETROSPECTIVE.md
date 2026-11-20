# RETROSPECTIVE — BmmFwdOp (op_slug=bmm)

> migration-harness op-level retrospective. Stage 5 (integrate) is op-scoped (integration
> problems do not belong to a single function); Stage 1–3 function-scoped retrospectives live at
> `examples/bmm/_bmm_kernel/RETROSPECTIVE.md`.

## Stage 5 (integrate) Retrospective — attempt_1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` rewrites only the wrapper **import** (`from ._bmm_fwd_kernels import ...` -> baseline/perf_opt block), never the wrapper **call site**. The Stage 0 scaffold invokes the GPU nested factory shape `_bmm_kernel(dims, dtype)(block_m, block_n, block_k, num_stages)(a, b)`, while the Stage 3 npuir kernel is a **flat, already-`@tilelang.jit`-decorated** factory `_bmm_kernel(batch, m, n, k, block_M, block_N, block_K, num_stages, dtype) -> compiled(a, b)`. Import smoke (the only gate the script runs) passes, and the mismatch surfaces only at report time as a Python `TypeError` (`dtype` binds to `block_M`). | Pre-fix wrapper snapshot: `examples/TileOPs/tileops/kernels/bmm/bmm/bmm_kernel/history_version/bmm_s5_attempt1.py` L162 (`self.kernel = _bmm_kernel(batch, m, n, k, self.dtype_str)`) and L88-89 (custom-op body); npuir factory signature: `bmm_kernel/_bmm_kernel.py` L67-78; failure at `_bmm_kernel.py:97` recorded in `integration_log.md` 调试历史. | Add a call-shape/signature compatibility precheck to `integrate_kernel.py` (compare the wrapper's `_{op}_wrapped_kernel` call against the integrated factory's `inspect.signature`), or document in the add-npu-op integrate step that the Stage 0 scaffold assumes the GPU nested shape and the integrator must reconcile it when the Stage 3 kernel exposes a flat factory. | R |
| Integrate-flow | The Stage 0 scaffold's eager `self.kernel = _bmm_kernel(dims, dtype)` is doubly wrong for a flat npuir factory: (a) wrong positional bind, and (b) even with correct args it would compile at construction time, **outside** `Kernel.__call__`'s `scoped_ascend_mode(ascend_mode)` guard (kernel_base.py L89-97), so the first compile would read the ambient `TILELANG_ASCEND_MODE` rather than the class's `Developer`. The scaffold intends "deferred compile", which no longer holds once the outer factory is itself jitted. | `tileops/kernels/kernel_base.py` L55-97 (`scoped_ascend_mode` only wraps `forward`); scaffold comment "does not compile yet -- compilation is deferred to forward()" in `history_version/bmm_s5_attempt1.py` L160-162. | Scaffolder should not eagerly invoke the factory; for a flat jitted factory it should store the factory handle (or defer) and let `forward` compile inside the mode scope. Note this in the scaffold template for cube/`@tilelang.jit`-at-top factories. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Flat-jitted npuir factory needs the wrapper to store the factory handle in `Kernel.kernel`, not the compiled result** — storing the factory *function* (a `FunctionType`) both avoids unscoped eager compilation and keeps msprof kernel-name derivation working: `msprof._derive_kernel_name_from_prim_func` Pass 2 walks `vars(kernel_instance)`, hits the function, reads its module file, and regex-finds the single `@T.prim_func` name `_bmm_main`. The run.json confirms `msprof.kernel_name_resolved="_bmm_main"`, `captured_op_names=["_bmm_main"]`, `kernel_filter_fallback_count=0`. | `tileops/benchmark/msprof.py` L312-411 (Pass 2 static scan); fixed wrapper `bmm.py` `self.kernel = _bmm_kernel`; run.json `performance.cases[*].msprof.kernel_name_resolved`. | `cd examples/TileOPs && source ~/.tilelang-ascend.env && python -m tileops.reporting.cli run --op BmmFwdOp --prof-mode msprof` | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | pattern-library（npnir 集成：flat jitted factory 的 wrapper 胶水） |
| **Batched-GEMM Developer-mode cube migration integrates with only wrapper-call glue; kernel/wrapper produce identical golden-contract results** — the Stage 3 kernel (folded 1D grid + slice-form `T.copy` epilogue + `initC=(k==0)`) reaches 17/17 correctness through the frozen Op layer with zero kernel edits; only the wrapper call shape changed. max avg abs err 7.10e-3, max 0.0625. | run.json `summary.correctness_passed=true`, `correctness_tests=17`, `operators[0].avg_max_abs_err=7.1e-3`; final wrapper `tileops/kernels/bmm/bmm/bmm.py`. | report command above | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | pattern-library（bmm/cube Developer-mode 集成基线） |
| **msprof Roofline `Ratio(%)` can silently degrade to N/A while latency stays valid** — `BenchmarkBase._build_result` calls `_parse_msprof_roofline`; when the `GM Read + Write` roofline entry is absent from `visualize_data.bin` it logs "msprof roofline parsing failed; falling back to theoretical metrics" and records only theoretical `tflops`/`bandwidth_tbs`. All 4 valid BmmFwdOp cases had `roofline_utilization_percent=null`. Integrators must not assume Ratio (%) exists in run.json. | `tileops/benchmark/benchmark_base.py` L307-357 (fallback path); run.json `performance.cases[*].roofline_utilization_percent=null`, `computility_tops=null`. | report command above; `python -c "import json;d=json.load(open('<run>/run.json'));print([c['roofline_utilization_percent'] for c in d['performance']['cases']])"` | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | Integrate-flow docs（run.json roofline 字段可空性） |
| **Large-shape msprof (`square-b4-4k` bf16) exceeds the 600 s per-case subprocess budget** — the profiler `TimeoutExpired` fails that benchmark test while the other 4 cases pass; the report status becomes `partial` and CLI exits 1, but correctness is intact. A bigger `--timeout` (runner) or smaller `--launch-count`/`--warm-up` would be needed; not an integration defect. | run.json `benchmark_tests.cases[4].message` (`TimeoutExpired ... timed out after 600 seconds`); `metadata.exit_codes.benchmark=1`. | report command above | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | bottleneck-patterns（msprof 单 case 超时 vs 报告 partial） |

### Transferable Lessons（可迁移教训）

- **When integrating an op, diff the wrapper's factory-call shape against the Stage 3 kernel's
  actual signature before trusting the import smoke.** `integrate_kernel.py` only checks imports.
  A flat `@tilelang.jit`-at-top factory (`f(dims, block_*, stages, dtype) -> compiled`) is NOT
  interchangeable with the GPU nested shape (`f(dims, dtype)(block_*)(inputs)`); the wrapper call
  site must be adapted (wrapper glue is in-scope for Stage 5).
- **Do not eagerly invoke a jitted factory in `Kernel.__init__`.** Compilation must be reachable
  from `Kernel.__call__` so it runs under `scoped_ascend_mode(ascend_mode)`; store the factory
  handle in `Kernel.kernel` instead — this also keeps `msprof` kernel-name derivation working via
  the Pass-2 `@T.prim_func` module scan.
- **`run.json` roofline `Ratio(%)` may be absent** (the msprof Roofline parse can fall back to
  theoretical metrics). Read latency + theoretical TOps/s + theoretical bandwidth, and treat
  `roofline_utilization_percent=null` as "N/A", not zero.
- **A `partial` report with `correctness_passed=true` is an integration success**, not a failure:
  `partial` here came from a single large-workload profiler timeout plus the missing roofline
  ratio; benchmark findings are report-only and must not trigger kernel/wrapper repair.

## Stage 5 (integrate) Retrospective — report re-run — 2026-10-14

> 集成脚本未再运行（幂等产物未变）；本次仅重跑单算子报告，验证 msprof Roofline `Ratio(%)`
> 容错修复（commit `b192f3d`）的效果。本机（另一台机器修复后已 pull）重跑结论：**report
> `status=passed`，correctness 17/17，benchmark 5/5 有效，Ratio 全部恢复**。本机重跑同时**推翻**
> 上一章节中两条 D 类结论（见下表与 Transferable Lessons）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Cube 算子 msprof Roofline `Ratio(%)` 全 N/A 的根因是 `parse_bin_file` 裸 `float()` 解析 `ratio="NA"`**（FIXP 记录），异常放弃整个 `visualize_data.bin` 解析，连 GM Read+Write 记录一起丢失；改用已有 `safe_float`（NA/空/None→0.0）后 5/5 case 恢复非空。修复后 `roofline_utilization_percent` 范围 6.79%–10.41%。 | D | 修复前：`reports/tileops/20261014_152027_623162_BmmFwdOp/run.json` 4/4 `roofline_utilization_percent=null`；修复后：`reports/tileops/20261014_155549_027687_BmmFwdOp/run.json` 5/5 非空；代码 `examples/TileOPs/tileops/benchmark/msprof_roofline.py` L142 `safe_float`, L176-180；commit `b192f3d`；capability-gap `CG-2026-0031`。 | `cd examples/TileOPs && source ~/.tilelang-ascend.env && python -m tileops.reporting.cli run --op BmmFwdOp --prof-mode msprof` 然后 `python -c "import json;d=json.load(open('<run>/run.json'));print([c['roofline_utilization_percent'] for c in d['performance']['cases']])"` | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | capability-gaps / msprof-roofline 文档 |
| **`Bandwidth(TB/s)` 列在 roofline 修复后口径改变**：修复前为 N/A 回退路径理论推得的“达成带宽”（随 latency 变化，如 0.0892 TB/s）；修复后为该 roofline 条目的硬件**峰值**带宽（本机恒为 1.75 TB/s），`Computility(TOps/s)` 同理恒为 294.9120（峰值）。跨修复前后版本对比该列会得到错误结论。 | D | 修复前 `.../20261014_152027_623162_BmmFwdOp/run.json` `performance.cases[*].bandwidth_tb_s≈0.0892/0.1436`；修复后 `.../20261014_155549_027687_BmmFwdOp/run.json` 全部 `bandwidth_tb_s=1.75`, `computility_tops=294.9119873046875`。 | 同上一行 repro；对比两个 run.json 的 `performance.cases[*].bandwidth_tb_s` | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | bottleneck-patterns（run.json roofline 字段口径） |
| **`square-b4-4k-bfloat16` 的 msprof 600 s 超时是**非确定性**的，重跑未复现**：本次同 shape 成功产出有效 profile，`latency_us=13748.58`、`profiler_valid=true`。上一章节把单次 `TimeoutExpired` 描述为“exceeds the 600 s per-case subprocess budget”属于对偶发现象的过度泛化。 | D | 本次 `reports/tileops/20261014_155549_027687_BmmFwdOp/run.json` `performance.cases[4]`（`label=square-b4-4k`, bf16, b=4 m=n=k=4096, `latency_us=13748.580078`, `profiler_valid=true`）；上次 `reports/tileops/20261014_152027_623162_BmmFwdOp/run.json` 同 case `TimeoutExpired after 600 s`、`profiler_valid=false`。 | 同第一节 repro | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | bottleneck-patterns（msprof 超时的非确定性） |
| **BmmFwdOp 集成基线性能（修复后口径）**：tag=tileops，msprof Ratio 6.79%–10.41%，latency 175.6–13748.58 us，AI 1023.97–2848.29 Ops/Byte；`benchmark.md` 同附 `torch-bmm`（`BatchMatMulV2`）对照 Ratio 46.1%–88.9%。Ratio 全区 <11% 说明当前 64/64/64/stages=2 配置离 cube 上限仍有较大优化空间（Stage 4 输入，非集成缺陷）。 | D | `reports/tileops/20261014_155549_027687_BmmFwdOp/run.json` `operators[0].ratio_range={min:6.786,max:10.4074}`；`benchmark.md` torch-bmm 表。 | 同第一节 repro | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 ×1 / driver 25.5.0 | bottleneck-patterns（bmm 迁移基线） |

### Transferable Lessons（可迁移教训）

- **修复依赖已验证后的报告重跑，必须区分“修复前/后”两次 run.json，不要用旧报告验证新结论**：
  本任务是 pull 到容错修复（`safe_float`）后的重跑；先确认修复代码在本地（`grep safe_float
  msprof_roofline.py`）再重跑，结论只引本次时间戳目录的 run.json。
- **msprof 单 case 超时不是确定性缺陷**：同一 shape 重跑可能成功。单次 `TimeoutExpired` 不得
  写成“该 shape 必然超时”；如实记录“本次失败/本次成功”并标注非确定性即可，不触发修复。
- **`run.json` 的 `Bandwidth(TB/s)` / `Computility(TOps/s)` 语义随 roofline 解析是否成功而变**：
  解析成功→硬件**峰值**；解析失败回退→理论**达成值**。跨版本/跨修复对比这两列前先确认
  `roofline_utilization_percent` 是否为 null，否则会得出错误性能结论。
- **集成期报告重跑不需要重跑 `integrate_kernel.py`**（幂等产物未变时）：只需更新集成包内
  `integration_report.json` 指向本次 run.json，并在 `integration_log.md` 追加本次结论。
