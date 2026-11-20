# RETROSPECTIVE.md — `exp`(op 级)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见 `examples/exp/exp_fwd_kernel/RETROSPECTIVE.md`。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark `partial`,已明确披露)
- **attempts_used**:1 · **smoke**:pass
- **正确性**:**8/8 通过**(`run.json`:`correctness_passed = true`,`pass_rate = 1.0`)
- **benchmark**:8 通过 / **2 失败**(均为最大 workload `[16384,16384]`)

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **268M workload 的 benchmark 失败已连续在 4 个算子上复现,Stage 0 的 workload 规模规则至今未改** —— 逐算子登记显然不收敛 | 本任务:`benchmark_failed = 2`、失败用例均为 `elementwise-256m-{float16,bfloat16}`、`report_status = partial`;与 `AbsFwdOp`/`CeilFwdOp`/`FloorFwdOp` **完全同型**(四份 run.json 一致) | 该 R 类提案**已被同一现象触发 4 次**,远超"三次即提升"的门槛:应直接在 `add-npu-op` 的 S1 workload 规则里加入 **NPU 侧单 workload 元素数上限判据**(纯逐元素算子实测 ~64M 可稳定画像、268M 不可) | R |

### Transferable Lessons（可迁移教训）

- **同一现象触发同一条 R 类提案 4 次 —— 逐算子登记是无效流程**:268M workload 的 benchmark 失败在 `abs`/`ceil`/`floor`/`exp` 上完全同型(四份 `run.json` 的 `benchmark_failed` 恒为 2、失败标签恒为 `elementwise-256m-*`)。修法在 Stage 0 的 workload 规模规则里,**不在每个算子的复盘里**。
- **集成侧的实质变更不总是数值性的**:`exp` 的既有手工实现与本设计同形态,故集成不改变数值语义,主要价值在于把边界用例落进 L1 阻塞层 —— **"集成完成"不等于"语义被更强地约束"**,后者需要在复盘里显式说明。
