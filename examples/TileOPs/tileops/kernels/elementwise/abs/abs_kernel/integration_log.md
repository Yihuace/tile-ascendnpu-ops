# Integration Log — `AbsFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/elementwise/abs/.migration_meta.json`
- **integrated**:`abs_fwd_kernel` → `tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py`(源自 `examples/abs/abs_fwd_kernel/abs_fwd_kernel.py`)
- **design_docs**:`abs_fwd_kernel` → `tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel_DESIGN.md`(源自 `examples/abs/abs_fwd_kernel/DESIGN.md`,Stage 1 交付件快照;**未缺失**)
- **wrapper**:`tileops/kernels/elementwise/abs/abs.py` —— `[wrapper] already integrated`;`[mode] Developer (unchanged)`
- **ascend_mode**:`Developer`(从 Stage 3 产物的 `ASCEND_MODE` 字面量读取,非猜测)
- **attempts**:1(首次即完成,未进调试闭环)
- **集成脚本摘要**:
  ```json
  {"op_slug": "abs", "family": "elementwise", "functions": ["abs_fwd_kernel"],
   "target_dir": "tileops/kernels/elementwise/abs/abs_kernel",
   "wrapper": "tileops/kernels/elementwise/abs/abs.py",
   "wrapper_rewritten": false, "ascend_mode": "Developer",
   "wrapper_mode_updated": false, "reexported": ["abs_fwd_kernel"],
   "smoke": "pass", "timestamp": "2026-10-15T09:05:46+0000"}
  ```

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261015_090559_382649_AbsFwdOp/run.json`
- **report_md**:`examples/TileOPs/reports/tileops/20261015_090559_382649_AbsFwdOp/report.md`
- **report_status**:`partial`(CLI 退出码 1 —— 按规范**不凭退出码判定**,以 `run.json` 为准)
- **correctness**:**pass** —— `summary.correctness_passed = true`,`correctness_tests = 9`,`correctness_failed = 0`,`pass_rate = 1.0`
  - pytest 侧原文:`9 passed, 14 warnings in 36.01s`(`tests/ops/test_abs.py`,覆盖 smoke-1m / elementwise-16m / elementwise-64m 等 workload × dtype)
- **benchmark**:**partial** —— `summary.benchmark_passed = false`(10 例中 **2 例失败**)

### benchmark 有效数值(run.json 摘要,摘自 `benchmark.md`)

| workload | dtype | Latency (µs) | 说明 |
|---|---|---|---|
| smoke-1m `[1048576]` | float32 / float16 / bfloat16 | 23.00 / 23.16 / 23.16 | 有效 |
| elementwise-16m `[4096,4096]` | float16 / bfloat16 / float32 | 348.19 / 347.71 / 354.47 | 有效 |
| elementwise-64m `[8192,8192]` | float16 / bfloat16 | 1389.74 / 1389.82 | 有效 |

### benchmark 失败明细(2 例,**不与 kernel 性能结论混同**)

| 用例 | 报错 |
|---|---|
| `bench_abs.py::test_abs_bench[elementwise-256m-float16]` | `RuntimeError: elapsed_time ... NPUEvent.cpp:178 NPU function error` / `rtFree execution failed`, `rtEventElapsedTime execution failed, reason=event recorder null` |
| `bench_abs.py::test_abs_bench[elementwise-256m-bfloat16]` | 同上 |

- **失败定位**:两条都是 manifest 的最大 workload `[16384,16384]`(= 268 435 456 元素)。
- **性质**:设备侧资源/事件记录失败,属**超大规模用例的资源问题**,不是 kernel 正确性或 kernel 性能问题(同 kernel 在 16m/64m 上数值正常)。
- **前置披露**:该量级在 Stage 3 的分层套件中已被实测会触发进程 OOM(`Invalid_Argument(EL0003)` + 核心转储),当时已按「告警层规模须保证失败可捕获」口径收敛并记录于 `examples/abs/abs_fwd_kernel/RETROSPECTIVE.md`。本次 report 使用的是 manifest 原规模,故在 benchmark 阶段复现同类失败。
- **处置**:按 integrator 规范「benchmark 失败只记录、不修复、不重试」,本 Stage **未**对 kernel / wrapper / tests / benchmarks / manifest 做任何修改。

## 调试历史

无 —— attempt 1 首次集成即完成(wrapper 已集成、smoke 通过、正确性 9/9),未进入调试闭环。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`,NPU count 1
- CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`
- profiling mode:msprof(`profiler_fallback_count = 0`、`invalid_profile_count = 0`)

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**。harness.md §7.1 要求基于 `opencode export <session_id> > examples/abs/full.json` 生成该文件,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**,无法导出会话数据。按「无法确认时如实标注、不得伪造」的口径,**未伪造任何导出数据或耗时数字**,该钩子交由 conductor 处置。
