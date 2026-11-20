# RETROSPECTIVE.md — `abs`(op 级)

> 追加式日志:每次 Stage 执行追加一个章节,不覆盖历史章节。
> 本文件为 **op 级单份**(Stage 5 集成复盘落点);逐函数的 Stage 1/2/3 复盘见
> `examples/abs/abs_fwd_kernel/RETROSPECTIVE.md`。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark `partial`,已明确披露)
- **attempts_used**:1(未进调试闭环)
- **集成**:`[ok] integration complete`;smoke pass;`ascend_mode = Developer`(从 Stage 3 产物字面量读取)
- **正确性**:**9/9 通过**(`run.json` 的 `summary.correctness_passed = true`,`pass_rate = 1.0`)
- **benchmark**:8 通过 / **2 失败**(均为 manifest 最大 workload `[16384,16384]`)

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `benchmark.json` 的 `records` **不带 shape / dtype 字段**(本次 8 条记录全部 `shape=None`),失败用例只能从 `pytest/benchmark.log` 的 `FAILED ...::test_abs_bench[elementwise-256m-float16]` 行反查;suggested 的「从 run.json 摘要引用性能数字」在**失败归因**上不成立 | 本任务:`reports/tileops/20261015_090559_382649_AbsFwdOp/benchmark.json` 的 `records` 为 8 条、字段无 shape;失败名仅出现在 `pytest/benchmark.log:686-687` | 在 `tileops` 的 benchmark 记录结构里补 `shape` / `dtype` / `workload_label` 字段(与 `benchmark.md` 表格同源),使 `run.json` 能独立支撑失败归因 | R |
| Integrate-flow | harness 的 manifest workload 规模与 report 的 benchmark 能力之间无护栏:本算子 manifest 声明 `[16384,16384]`(268M),而 benchmark 在该量级上**设备侧失败**,导致每次集成都会出现 2 条 `benchmark_failed`,report 恒为 `partial`,`passed` 不可达 | 本任务:2 条失败均为 `elementwise-256m-*`,报 `rtFree execution failed` / `rtEventElapsedTime execution failed, reason=event recorder null`;同 kernel 在 16m/64m 数值正常 | 在 Stage 0(`add-npu-op`)的 S1 workload 规则里补一条 NPU 侧上限判据:单个 workload 的元素数须落在 report benchmark 可稳定画像的范围内(纯逐元素算子本次实测 ~64M 可用、268M 不可用),超出者拆成更小 workload 或标注为「benchmark 不可画像」 | R |
| Integrate-flow | harness.md §7.1 的 `SESSION_TIMING_ANALYSIS.md` 钩子绑定 `opencode export <session_id>`,在非 OpenCode 宿主(本次为 AtomCode)上**无对应命令**,该钩子恒不可执行且规范未给降级口径 | 本任务:宿主无 `opencode` CLI;按「无法确认时如实标注」处理,未伪造导出数据,已在 `integration_log.md` 的「已知未执行项」披露 | 在 harness.md §7.1 增加宿主无关的降级条款:「宿主不提供 `opencode export` 时,以宿主自身的会话导出能力替代;均不可得则写 `total_duration_s: null` 并在正文披露原因,不得伪造」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **report 的 `partial` + CLI 退出码 1 不等于集成失败**:判定须读本次 `run.json` 的 `summary.correctness_passed`;本次 correctness 9/9 通过、仅 benchmark 2 例设备侧失败 | D | `run.json`:`status=partial`、`summary.correctness_passed=true`、`correctness_failed=0`、`benchmark_passed=false`、`benchmark_failed=2`;CLI `REPORT_EXIT=1` | `env -u PYTHONPATH <venv>/bin/python -u -m tileops.reporting.cli run --op AbsFwdOp --prof-mode msprof; echo $?` 后读同目录 `run.json` | tilelang `0.1.2+npuir+7fc4a3c8` / Ascend910B2C / CANN 9.0.0 / torch_npu 2.10.0 | `tilelang-op-integrator` SKILL.md 的 report 判读段 |
| **纯逐元素算子在 268M 元素规模上 benchmark 设备侧失败**:`rtFree execution failed` + `rtEventElapsedTime execution failed, reason=event recorder null`,同 kernel 在 64M 及以下数值正常 —— 属超大规模资源/事件记录问题,非 kernel 缺陷 | D | `pytest/benchmark.log:686-687` 的两条 `FAILED ...[elementwise-256m-float16] / [elementwise-256m-bfloat16]`;`benchmark.md` 表格显示 16m/64m 数值正常(348µs / 1390µs) | `env -u PYTHONPATH <venv>/bin/python -u -m tileops.reporting.cli run --op AbsFwdOp --prof-mode msprof`(观察 benchmark 阶段 2 例失败的 workload 标签) | 同上 | `pattern-library/traps-runtime.md`(大规模 workload 类) |

### Transferable Lessons（可迁移教训）

- **集成期判定必须读 `run.json`,不能凭 CLI 退出码**:tileops report 在 `partial`(正确性通过但 benchmark 有问题)时也返回非零退出码,单看退出码会把"正确性全过"误判为集成失败。判据是本次 `run.json` 的 `summary.correctness_passed`。
- **manifest 声明的最大 workload 可能超出 report benchmark 的稳定画像能力**:纯逐元素算子在 268M 元素(16384×16384)上会设备侧失败(`rtFree execution failed` / `event recorder null`),使 report 恒为 `partial`;同 kernel 在 64M 及以下正常。遇到「唯一失败的都是最大 workload、报的是设备/事件类错误」时,先怀疑规模而非 kernel。
- **`benchmark.json` 的 `records` 不带 shape/dtype**:失败归因必须回到 `pytest/benchmark.log` 的 `FAILED ...::[用例名]` 行;`benchmark.md` 的表格有 shape 但结构化 json 没有,不要以为能在 json 里定位失败用例。
- **OpenCode 专属的 harness 钩子在非 OpenCode 宿主上不可执行**:harness.md §7.1 的 `opencode export` 钩子在 AtomCode 上无对应命令。不要伪造导出数据或耗时数字 —— 如实披露「无法执行 + 原因」,把处置交回 conductor。
