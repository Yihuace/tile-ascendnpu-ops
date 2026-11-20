# RETROSPECTIVE.md — `floor`(op 级)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见
> `examples/floor/floor_fwd_kernel/RETROSPECTIVE.md`。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark `partial`,已明确披露)
- **attempts_used**:1(未进调试闭环)
- **集成**:`[ok] integration complete`;`smoke: pass`;`ascend_mode = Developer`
- **正确性**:**8/8 通过**(`run.json` 的 `summary.correctness_passed = true`,`pass_rate = 1.0`)
- **benchmark**:8 通过 / **2 失败**(均为 manifest 最大 workload `[16384,16384]` = 268M 元素)

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **268M workload 的 benchmark 失败已在 3 个算子上复现,但流程仍无"提升到 Stage 0 规则"的机制** —— 每个算子都在自己的 op 级复盘里记一次,而 Stage 0 的 workload 规模规则**至今未改**,于是第 4 个算子还会再踩 | 本任务:`AbsFwdOp` / `CeilFwdOp` / `FloorFwdOp` 三者 `summary.benchmark_failed` **均为 2**、失败用例**均为** `elementwise-256m-{float16,bfloat16}`、报错**同类**(`aclrtEventElapsedTime` / `event recorder null`);而 `examples/abs/RETROSPECTIVE.md`(首次登记)与 `examples/ceil/RETROSPECTIVE.md`(二次登记)已各提过一次同一条 R 类提案 | 该 R 类提案**已被同一现象触发 3 次**,建议 conductor/evolver 按"**三次触发即自动提升为 Tier 1 修订**"处理:直接在 `add-npu-op` 的 S1 workload 规则里加入 NPU 侧单 workload 元素数上限判据(纯逐元素算子实测 ~64M 可画像、268M 不可) | R |
| Integrate-flow | 集成侧 `tests/` 的边界覆盖缺口(同 ceil)在本算子**第 2 次出现**,机制性原因相同:设计 §9.2 的 L1 是 **Stage 3 自套件**的,而 integrator 无权改集成侧 `tests/` | 本任务:独立核对 `tests/ops/test_floor.py` 中 NaN/inf/超范围 命中数 = **0**;而 `examples/floor/floor_fwd_kernel/floor_fwd_kernel.py` 的 L1 已含这三类且全过 | 同 ceil 的提案(Stage 5 须提供"集成侧测试覆盖缺口"的登记与升级路径);该提案现亦已触发 2 次 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **268M workload 的 benchmark 失败是 harness 的稳定能力边界(3/3 算子复现)**:`abs`/`ceil`/`floor` 三者均在**恰好 2 条** `elementwise-256m-{float16,bfloat16}` 上失败,`benchmark_failed` 恒为 2、`report_status` 恒为 `partial`,报错类别一致 | D | 三份 `run.json`:`reports/tileops/20261015_090559_382649_AbsFwdOp/`、`..._115517_595927_CeilFwdOp/`、`..._140127_875207_FloorFwdOp/` | 对任一算子跑 `python -m tileops.reporting.cli run --op <Op> --prof-mode msprof`,观察 benchmark 阶段失败的两条 workload 标签 | tilelang `0.1.2+npuir+7fc4a3c8` / Ascend910B2C / CANN 9.0.0 / torch_npu 2.10.0 | Stage 0 workload 规模规则(`add-npu-op` S1)+ `tilelang-op-integrator` SKILL |

### Transferable Lessons（可迁移教训）

- **同一现象触发同一条 R 类提案 3 次,就该提升为 Tier 1 修订而不是继续登记**:268M workload 的 benchmark 失败在 `abs`/`ceil`/`floor` 上完全同型(各提过一次 R 类提案),说明"逐算子登记"不解决问题 —— 修法在 Stage 0 的 workload 规模规则里。**重复登记同一 R 类提案本身就是流程信号**。
- **"集成侧 `tests/` 未覆盖设计声明的边界语义"是结构性缺口,不是偶发**:本任务与 `ceil` 上重复出现,原因是设计 §9.2 的 L1 属于 Stage 3 自套件,而 integrator 无权修改集成侧 `tests/`。`correctness_passed=true` **不能**当作该缺口已闭环的证据。
- **集成包内 `{func}_DESIGN.md` 是快照,其与最终设计的一致性依赖集成时机**:本算子设计只经一轮检视即通过,快照与最终设计一致;但机制上仍应确保集成发生在**最后一次设计修订之后**。
