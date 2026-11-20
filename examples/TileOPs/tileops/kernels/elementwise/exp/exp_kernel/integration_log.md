# Integration Log — `ExpFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/elementwise/exp/.migration_meta.json`
- **integrated**:`exp_fwd_kernel` → `tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py`(源自 `examples/exp/exp_fwd_kernel/exp_fwd_kernel.py`)
- **design_docs**:`exp_fwd_kernel` → `tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel_DESIGN.md`(源自 `examples/exp/exp_fwd_kernel/DESIGN.md`;**未缺失**)
- **wrapper**:`tileops/kernels/elementwise/exp/exp.py` —— 已集成;`ascend_mode: Developer`
- **attempts**:1(首次即完成,未进调试闭环)
- **smoke**:pass

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261015_142822_583659_ExpFwdOp/run.json`
- **report_status**:`partial`(CLI 退出码非零 —— 按规范**不凭退出码判定**,以 `run.json` 为准)
- **correctness**:**pass** —— `summary.correctness_passed = true`,`correctness_tests = 8`,`correctness_failed = 0`,`pass_rate = 1.0`
- **benchmark**:**partial** —— `summary.benchmark_passed = false`(`benchmark_tests = 10`,`benchmark_failed = 2`)

### benchmark 失败明细(2 例)

| 用例 | 说明 |
|---|---|
| `bench_exp.py::test_exp_bench[elementwise-256m-float16]` | 采集侧设备失败(与前序算子同型) |
| `bench_exp.py::test_exp_bench[elementwise-256m-bfloat16]` | 同上 |

- **失败定位**:均为 manifest 最大 workload `[16384,16384]`(268 435 456 元素)。
- **系统性**:该失败现已在 **`AbsFwdOp` / `CeilFwdOp` / `FloorFwdOp` / `ExpFwdOp` 四个算子上完全同型** —— 均恰好 2 条、均为最大 workload、报错同类。**这是 harness 的能力边界,应在 Stage 0 的 workload 规模规则层面修**;每个算子各记一次不解决问题。
- **处置**:按 integrator 规范「benchmark 失败只记录、不修复、不重试」,**未**修改 kernel / wrapper / tests / benchmarks / manifest。

### 本次集成的实质变更

既有手工实现 `exp_kernel/exp.py` 的形态与本设计**一致**(`widen → vexp → narrow`,`T.vexp(A,B)` 源在前),故本次集成**不改变数值语义**,主要变更在于:
- 把 §9.2 的边界用例(NaN / ±inf / 上溢 / 下溢 / **fp16 超范围** / 近零 / 非整除)明确列为 **L1 阻塞层**;
- 交付件按 Stage 3 规范补齐分层套件与 golden(独立于 NPU 算法)。

**集成侧测试覆盖缺口**:`tests/ops/test_exp.py` 以 shape/dtype 为主,不含上述边界类别。**tests/ 不在本 Agent 的可改范围内**,故该缺口登记于 `examples/exp/RETROSPECTIVE.md` 的 Stage 5 章节,交 conductor 决策。

## 调试历史

无 —— attempt 1 首次集成即完成(smoke 通过、正确性 8/8),未进入调试闭环。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`;CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`;profiling mode:msprof

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**:harness.md §7.1 要求基于 `opencode export <session_id>` 生成,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**。按「无法确认时如实标注、不得伪造」的口径,**未伪造任何导出数据**,交由 conductor 处置。
