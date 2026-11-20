# RETROSPECTIVE.md — `ceil`(op 级)

> 追加式日志:每次 Stage 执行追加一个章节,不覆盖历史章节。
> 本文件为 **op 级单份**(Stage 5 集成复盘落点);逐函数的 Stage 1/2/3 复盘见
> `examples/ceil/ceil_fwd_kernel/RETROSPECTIVE.md`。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark `partial`,已明确披露)
- **attempts_used**:1(未进调试闭环)
- **集成**:`[ok] integration complete`;smoke pass;`ascend_mode = Developer`(从 Stage 3 产物字面量读取)
- **正确性**:**8/8 通过**(`run.json` 的 `summary.correctness_passed = true`,`pass_rate = 1.0`)
- **benchmark**:8 通过 / **2 失败**(均为 manifest 最大 workload `[16384,16384]` = 268M 元素)

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **集成侧 `tests/` 与 Stage 3 自带的 L1 套件是两个测试面,前者不会因 kernel 改进而自动补齐。** 设计的 §9.2 把 boundary 语义(本算子:NaN / ±inf / 超 int32)声明为"必须保持"并落 L1 —— 但那是 **Stage 3 自套件**的 L1;集成侧 `tests/ops/test_ceil.py` 仍只有 8 个不含这三类的用例。而 integrator 被规范**禁止修改 `tests/`**,故本次虽用正确的实现取代了旧的 C1 实现,集成侧覆盖缺口**依然存在**且**无法由本 Agent 关闭** | 本任务:集成后 `tests/ops/test_ceil.py` 仍 8 用例、不含 NaN/±inf/超范围;而 `examples/ceil/ceil_fwd_kernel/ceil_fwd_kernel.py` 的 L1 已含这三类且全过。runner 的 `correctness_tests = 8` 与设计 §9.2 的 L1 类别数不匹配 | 在 harness 的 Stage 5 规范里补一条**升级路径**:当设计把某边界语义声明为"必须保持"而集成侧 `tests/` 未覆盖时,integrator 须在 `integration_log.md` 明确登记「集成侧测试覆盖缺口 + 建议补的用例清单」,并**交回 conductor 决策**(而非静默通过) | R |
| Integrate-flow | **268M workload 的 benchmark 失败是系统性、可复现的跨算子问题**,但当前只在各算子自己的 RETROSPECTIVE 里逐条登记,没有回归到 Stage 0 的 workload 规模规则 —— 于是每个算子集成时都会重复踩一次 | 本任务与 `abs` 的失败**完全同型**:都是恰好 manifest 最大 workload 的 2 条(`256m` × {fp16,bf16}),报错同为 `aclrtEventElapsedTime error 507000` / `rtFree execution failed` / `event recorder null`;`benchmark_failed = 2`、`report_status = partial` 在两者上一致 | 在 Stage 0(`add-npu-op` 的 S1 workload 规则)增加**NPU 侧单 workload 元素数上限判据**:纯逐元素算子实测 ~64M 可稳定画像、268M 不可 → 超出者拆成更小 workload 或在 manifest 中标注「benchmark 不可画像」,从源头上消除每个算子都出现的 `partial` | R |
| Integrate-flow | 集成包内 `{func}_DESIGN.md` 是 Stage 1 交付件的**快照**,与 `examples/{op_slug}/{func}/DESIGN.md` 同源但**不随设计修订更新**;本算子的设计经 round 1/2/3 三轮修订,若快照在早期轮次生成就会与最终设计不一致 | 本任务:集成脚本在 Stage 2 round 3 **之后**才运行,故快照与最终设计一致(`design_docs` 映射有记录);但这一致性**依赖集成时机**而非机制保证 —— 若在 Stage 1 早期已集成过一次,快照就会陈旧 | 在 `integrate_kernel.py` 的输出里打印快照源文件的 hash/时间戳,或在 integration_log 里记录「快照源 = 最终冻结版」的核对结论 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **manifest 最大 workload(268M 元素)的 benchmark 失败是 harness 能力边界,跨算子可复现**:`abs` 与 `ceil` 均在**恰好 2 条** `elementwise-256m-{float16,bfloat16}` 上失败,报错与 `benchmark_failed=2` 完全同型 | D | `abs`:`reports/tileops/20261015_090559_382649_AbsFwdOp/`;`ceil`:`reports/tileops/20261015_134734_293891_CeilFwdOp/`;两者 `run.json` 的 `summary.benchmark_failed = 2`,`report_status = partial` | `python -m tileops.reporting.cli run --op CeilFwdOp --prof-mode msprof`(观察 benchmark 阶段失败的两条 workload 标签) | tilelang `0.1.2+npuir+7fc4a3c8` / Ascend910B2C / CANN 9.0.0 / torch_npu 2.10.0 | `tilelang-op-integrator` SKILL 的 report 判读段 + Stage 0 workload 规模规则 |
| **「读 `run.json` 而非 CLI 退出码」在第二个算子上再次被验证必要**:`report_status=partial` + 非零退出码,而 `summary.correctness_passed=true`、`pass_rate=1.0` —— 单看退出码会把它误判为集成失败 | D | `examples/TileOPs/reports/tileops/20261015_134734_293891_CeilFwdOp/run.json`:`status=partial`, `correctness_passed=true`, `correctness_tests=8`, `correctness_failed=0`, `benchmark_passed=false`, `benchmark_failed=2` | 同上命令后 `echo $?` 与同目录 `run.json` 对照 | 同上 | `tilelang-op-integrator` SKILL 的 report 判读段(已有,作为二次实证) |

### Transferable Lessons（可迁移教训）

- **集成侧 `tests/` 与 Stage 3 自带的分层套件是两个测试面**:kernel 改进后,集成侧测试**不会**自动补齐。若设计把某边界语义声明为"必须保持",除 Stage 3 的 L1 之外还必须有人把用例补进集成侧 `tests/` —— 而 integrator 被禁止修改 `tests/`,所以遇到「设计 L1 已覆盖、集成侧未覆盖」时必须**显式登记缺口并交回 conductor**,不能因为 `correctness_passed=true` 就当它已闭环。
- **manifest 的最大 workload 可能超出 benchmark 的稳定画像能力,且这是跨算子复现的系统性问题**:`abs` 与 `ceil` 都在 `[16384,16384]`(268M 元素)的 2 条上失败,报错同为事件记录类(`aclrtEventElapsedTime 507000` / `event recorder null`)。遇到「唯一失败的都是最大 workload、报设备/事件类错误」时先怀疑规模而非 kernel;修法应在 Stage 0 的 workload 规模规则里做,而不是每个算子集成时各踩一次。
- **`report_status=partial` + CLI 非零退出码 ≠ 集成失败**:判据是本次 `run.json` 的 `summary.correctness_passed`。本任务与 `abs` 两次验证了这一点。
- **`{func}_DESIGN.md` 快照与最终设计的一致性依赖"集成时机"而非机制**:设计经多轮修订时,须确保集成发生在**最后一次设计修订之后**,或在 integration_log 里记录该核对结论。
