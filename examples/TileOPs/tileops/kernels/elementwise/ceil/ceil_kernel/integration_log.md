# Integration Log — `CeilFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/elementwise/ceil/.migration_meta.json`
- **integrated**:`ceil_fwd_kernel` → `tileops/kernels/elementwise/ceil/ceil_kernel/ceil_fwd_kernel.py`(源自 `examples/ceil/ceil_fwd_kernel/ceil_fwd_kernel.py`)
- **design_docs**:`ceil_fwd_kernel` → `tileops/kernels/elementwise/ceil/ceil_kernel/ceil_fwd_kernel_DESIGN.md`(源自 `examples/ceil/ceil_fwd_kernel/DESIGN.md`,Stage 1 交付件快照;**未缺失**)
- **wrapper**:`tileops/kernels/elementwise/ceil/ceil.py` —— `[wrapper] already integrated`;`[mode] Developer (unchanged)`
- **ascend_mode**:`Developer`(从 Stage 3 产物的 `ASCEND_MODE` 字面量读取,非猜测)
- **attempts**:1(首次即完成,未进调试闭环)
- **集成脚本摘要**:
  ```json
  {"op_slug": "ceil", "family": "elementwise", "functions": ["ceil_fwd_kernel"],
   "sources": {"ceil_fwd_kernel": ".../examples/ceil/ceil_fwd_kernel/ceil_fwd_kernel.py"},
   "design_docs": {"ceil_fwd_kernel": ".../ceil_kernel/ceil_fwd_kernel_DESIGN.md"},
   "target_dir": "tileops/kernels/elementwise/ceil/ceil_kernel",
   "wrapper": "tileops/kernels/elementwise/ceil/ceil.py",
   "wrapper_rewritten": false, "ascend_mode": "Developer",
   "wrapper_mode_updated": false, "reexported": ["ceil_fwd_kernel"],
   "smoke": "pass", "timestamp": "2026-10-15T13:47:21+0000"}
  ```

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261015_134734_293891_CeilFwdOp/run.json`
- **report_md**:`examples/TileOPs/reports/tileops/20261015_134734_293891_CeilFwdOp/report.md`
- **report_status**:`partial`(CLI 退出码非零 —— 按规范**不凭退出码判定**,以 `run.json` 为准)
- **correctness**:**pass** —— `summary.correctness_passed = true`,`correctness_tests = 8`,`correctness_failed = 0`,`pass_rate = 1.0`(pytest 侧原文:`8 passed`,覆盖 smoke-1m / elementwise-16m / elementwise-64m 等 workload × dtype)
- **benchmark**:**partial** —— `summary.benchmark_passed = false`(10 例中 **2 例失败**)

### benchmark 有效数值(run.json 摘要,摘自 `benchmark.md`)

| workload | shape | dtype | Latency (µs) | Bandwidth (TB/s) |
|---|---|---|---|---|
| elementwise-16m | [4096,4096] | float16 | 348.2739 | 0.1927 |
| elementwise-16m | [4096,4096] | bfloat16 | 348.0139 | 0.1928 |
| elementwise-16m | [4096,4096] | float32 | 348.0339 | 0.3856 |
| elementwise-64m | [8192,8192] | float16 | 1390.4756 | 0.1931 |
| elementwise-64m | [8192,8192] | bfloat16 | 1390.8556 | 0.1930 |

### benchmark 失败明细(2 例,**不与 kernel 性能结论混同**)

| 用例 | 报错 |
|---|---|
| `bench_ceil.py::test_ceil_bench[elementwise-256m-float16]` | `RuntimeError: aclrtEventElapsedTime(...) error 507000` / `rtFree execution failed` / `rtEventElapsedTime execution failed, reason=event recorder null` |
| `bench_ceil.py::test_ceil_bench[elementwise-256m-bfloat16]` | 同上 |

- **失败定位**:两条都是 manifest 的最大 workload `[16384,16384]`(= 268 435 456 元素),与 `AbsFwdOp` 的 benchmark 失败**同型**(该轮已在 `examples/abs/RETROSPECTIVE.md` 登记为采集侧能力边界)。
- **性质**:**采集侧**(profiling 事件记录)在超大规模下的设备失败,**不是 kernel 缺陷** —— 同 kernel 在 16m/64m 数值正常且与 abs 的实测带宽同量级。
- **处置**:按 integrator 规范「benchmark 失败只记录、不修复、不重试」,本 Stage **未**对 kernel / wrapper / tests / benchmarks / manifest 做任何修改。

### 与既有实现的差异(本次集成的实质变更)

既有手工实现 `ceil_kernel/ceil.py` 走 **C1(int32 round-trip)路径**,在下列三类边界上不正确(实测):

| 输入 | 既有实现(实测) | 本次集成产物 |
|---|---|---|
| `\|x\| >= 2^31` | `±2.147e9` ❌ | `x` ✅ |
| `±inf` | `±2.147e9` ❌ | `±inf` ✅ |
| `NaN` | `0` ❌ | `NaN` ✅ |

其测试 `tests/ops/test_ceil.py` **不含这三类用例**,故此前未暴露 —— 本次集成用 Stage 3 的 C2+NaN 实现取代了它。**但 tests/ 目录不在本 Agent 的可改范围内**(规范禁止修改 `tests/`),故集成侧测试**仍未覆盖**这三类;该缺口已在 `examples/ceil/RETROSPECTIVE.md` 的 Stage 5 章节登记为需 conductor 决策的事项。

## 调试历史

无 —— attempt 1 首次集成即完成(wrapper 已集成、smoke 通过、正确性 8/8),未进入调试闭环。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`
- CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`
- profiling mode:msprof

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**。harness.md §7.1 要求基于 `opencode export <session_id> > examples/ceil/full.json` 生成该文件,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**,无法导出会话数据。按「无法确认时如实标注、不得伪造」的口径,**未伪造任何导出数据或耗时数字**,该钩子交由 conductor 处置。
