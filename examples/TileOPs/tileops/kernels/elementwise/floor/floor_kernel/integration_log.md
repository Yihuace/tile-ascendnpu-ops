# Integration Log — `FloorFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/elementwise/floor/.migration_meta.json`
- **integrated**:`floor_fwd_kernel` → `tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py`(源自 `examples/floor/floor_fwd_kernel/floor_fwd_kernel.py`)
- **design_docs**:`floor_fwd_kernel` → `tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel_DESIGN.md`(源自 `examples/floor/floor_fwd_kernel/DESIGN.md`,Stage 1 交付件快照;**未缺失**)
- **wrapper**:`tileops/kernels/elementwise/floor/floor.py` —— `wrapper_rewritten: false`(已集成);`ascend_mode: Developer`
- **attempts**:1(首次即完成,未进调试闭环)
- **集成脚本摘要**:
  ```json
  {"op_slug": "floor", "family": "elementwise", "functions": ["floor_fwd_kernel"],
   "sources": {"floor_fwd_kernel": ".../examples/floor/floor_fwd_kernel/floor_fwd_kernel.py"},
   "design_docs": {"floor_fwd_kernel": ".../floor_kernel/floor_fwd_kernel_DESIGN.md"},
   "target_dir": "tileops/kernels/elementwise/floor/floor_kernel",
   "wrapper": "tileops/kernels/elementwise/floor/floor.py",
   "wrapper_rewritten": false, "ascend_mode": "Developer",
   "wrapper_mode_updated": false, "reexported": ["floor_fwd_kernel"],
   "smoke": "pass", "timestamp": "2026-10-15T14:01:12+0000"}
  ```

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261015_140127_875207_FloorFwdOp/run.json`
- **report_md**:`examples/TileOPs/reports/tileops/20261015_140127_875207_FloorFwdOp/report.md`
- **report_status**:`partial`(CLI 退出码非零 —— 按规范**不凭退出码判定**,以 `run.json` 为准)
- **correctness**:**pass** —— `summary.correctness_passed = true`,`correctness_tests = 8`,`correctness_failed = 0`,`pass_rate = 1.0`
- **benchmark**:**partial** —— `summary.benchmark_passed = false`(`benchmark_tests = 10`,`benchmark_failed = 2`)

### benchmark 失败明细(2 例,**不与 kernel 性能结论混同**)

| 用例 | 报错 |
|---|---|
| `bench_floor.py::test_floor_bench[elementwise-256m-float16]` | 采集侧设备失败(`aclrtEventElapsedTime` / `rtFree execution failed` 同型) |
| `bench_floor.py::test_floor_bench[elementwise-256m-bfloat16]` | 同上 |

- **失败定位**:两条都是 manifest 的最大 workload `[16384,16384]`(= 268 435 456 元素)。
- **性质**:**采集侧**(profiling 事件记录)在超大规模下的设备失败,**不是 kernel 缺陷**。
- **系统性**:该失败在 `AbsFwdOp`、`CeilFwdOp`、`FloorFwdOp` **三个算子上完全同型**(均恰好 2 条、均为最大 workload、报错同类)→ 属 harness 能力边界,**应在 Stage 0 的 workload 规模规则层面修**,不应每个算子各踩一次。详见 `examples/floor/RETROSPECTIVE.md` 的 Stage 5 章节。
- **处置**:按 integrator 规范「benchmark 失败只记录、不修复、不重试」,本 Stage **未**修改 kernel / wrapper / tests / benchmarks / manifest。

### 与既有实现的差异(本次集成的实质变更)

既有手工实现 `floor_kernel/floor.py` 走 **C1 路径**,其 docstring 已显式记录范围限制(`|x| < 2**31`),但**未处理 NaN**;故在下列三类边界上不正确(实测同 ceil 的 C1 路径):

| 输入 | 既有实现 | 本次集成产物 |
|---|---|---|
| `\|x\| >= 2^31` | `±2.147e9` ❌ | `x` ✅ |
| `±inf` | `±2.147e9` ❌ | `±inf` ✅ |
| `NaN` | `0` ❌ | `NaN` ✅ |

其测试 `tests/ops/test_floor.py` **不含这三类用例**(独立核对:0 条命中),故此前未暴露。**tests/ 目录不在本 Agent 的可改范围内**,故集成侧测试**仍未覆盖**这三类;该缺口已登记于 `examples/floor/RETROSPECTIVE.md` 的 Stage 5 章节。

## 调试历史

无 —— attempt 1 首次集成即完成(wrapper 已集成、smoke 通过、正确性 8/8),未进入调试闭环。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`;CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`;profiling mode:msprof

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**。harness.md §7.1 要求基于 `opencode export <session_id>` 生成,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**。按「无法确认时如实标注、不得伪造」的口径,**未伪造任何导出数据或耗时数字**,该钩子交由 conductor 处置。
