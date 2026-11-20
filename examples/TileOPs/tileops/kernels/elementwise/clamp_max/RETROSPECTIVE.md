## Stage 5 (integrate) Retrospective — harness re-run (attempt_index=2) — 2026-10-15

> 结果：`INTEGRATE_COMPLETED`。correctness 15/15 全通过（含此前因 frozen 测试 bug 失败的 0-dim
> 上界用例）；benchmark 已运行（11 tests passed，9/9 tileops 用例），但 tileops 侧 msprof 采集回退
> events → report `partial`（仅记录，不修复）；`gate 5 --migration-dir examples/clamp_max` pass
> （仅 `S5-REPORT-BENCH` warn）。集成重跑 1 次（幂等）+ 报告重跑 2 次（第二次因重跑刷新了集成文件
> mtime、触发 S5-REPORT-STALE 而必须重跑）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 缺陷的 Stage 0 scaffold 测试可**硬阻塞** Stage 5 集成，且 harness 无「scaffold 测试缺陷」升级通道：集成器按冻结文件规则不能改 `tests/`，只能返回 `[INTEGRATE_FAIL]`，需外部（用户/conductor）介入修复一行 fixture 代码才能继续 | 上一轮 integration_log「issues」段落；本次 integration_log「issues / 调试历史」；缺陷点 `tests/ops/test_clamp_max.py:121`（修前 `torch.randn(*max_shape, ...)`） | `.agents/skills/_shared/standards/gate-and-retry.md` 或 harness 流程文档新增「scaffold 测试缺陷」升级路径（Stage 0 owner 重生成 / 窄范围 fixture-only 修复并 review） | R |
| Integrate-flow | `integrate_kernel.py` 幂等重跑的 `--extracted-module {op_slug}_kernel` 要求被同族 Stage 5 日志（le）断言为「必须」，但当前脚本版本下**不带该 flag 的重跑同样成功**：`rewrite_wrapper_import` 一见到 `from .{op_slug}_kernel import` 即短路返回（已集成守卫），`parse_wrapper_imports` 空返回时回退 `list(functions)` | 本次对照重跑（无 flag）：`wrapper_rewritten=false, smoke=pass`，`__init__.py` 字节一致；`integrate_kernel.py` L316–L328 / L436–L437 | `integrate_kernel.py` docstring 或 add-npu-op skill 说明**明确该 flag 在何种条件下才必需**（避免流程文档与实现不一致） | R |
| Integrate-flow | 集成脚本幂等重跑会重写 kernel/DESIGN/`__init__.py`（字节相同）并刷新 mtime，从而**静默使已产出的 report 失效**（gate `S5-REPORT-STALE` 要求 run.json mtime ≥ 集成文件 mtime），迫使一次昂贵的全量 report 重跑 | `statectl gate 5` 首跑输出 `S5-REPORT-STALE`（报告早于当前集成文件）；本次 integration_log「验证结果」superseded 说明 | 集成脚本可加 `--touch-only-if-changed`（内容相同则不重写文件/不刷新 mtime），或在 skill 中提示「重跑集成后必须重跑 report」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| TileOPs `msprof op` 采集对经 Op 层调用的 npuir 投影 kernel（clamp_max）**产不出 `OpBasicInfo*.csv`**（`--kernel-name=main` 与无过滤两次候选均失败 → RuntimeError → `_bench_with_mode` 回退 events）；同一次运行的 `torch.minimum` 参照却能正常采集（`captured_op_count=1`）——即设备/msprof 本身健康，问题定位在 tileops 侧 kernel 的采集（kernel-name 解析 / launch 流） | D | 本轮 run.json `summary.profiler_fallback_count=9, msprof_case_count=0`；`reports/tileops/20261015_121037_823687_ClampMaxFwdOp/pytest/benchmark.log` L18–L21（tileops 两条 msprof 命令后无 `=== [msprof]` 输出）；`benchmark.json` torch 记录 `prof_mode="msprof"` | `ASCEND_RT_VISIBLE_DEVICES=1 python3 -m tileops.reporting.cli run --op ClampMaxFwdOp --prof-mode msprof`（观察 tileops 侧回落 events、torch 侧 msprof 有效） | tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910_9362 / torch_npu 2.7.1.post4 | tileops reporting/benchmark infra（benchmark_base.py `_bench_with_mode` / msprof.py 采集层）；或 `.agents/evolution/capability-gaps.md` |
| 迁移算子的**单一 kernel 文件可同时被 `Kernel` 类与集成聚合 `__init__` 复用**，`integrate_kernel.py` 的 baseline/perf_opt 切换块只需一次性改写 wrapper import；同族（le/ge/eq/gt/clamp_max）重跑一致性由 `from .{op_slug}_kernel import` 已集成守卫保证 | P | 本次 `wrapper_rewritten=false`，baseline 激活；`integration_log.md` wrapper 段 | `python examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta> --extracted-module clamp_max_kernel`（第二次调用输出 `already integrated`） | tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910_9362 | add-npu-op skill（集成脚本幂等语义） |

### Transferable Lessons（可迁移教训）

- **集成前先核实 frozen scaffold 测试 fixture 本身是否可运行**：`torch.randn(*shape, ...)` 在
  `shape==()` 时退化为 `torch.randn(dtype=..., device=...)` 抛 `TypeError`，会在 op 被调用前失败，
  被误读为 kernel 缺陷。排查 elementwise 迁移的「1 例失败」时，先看测试输入生成行（`*` 解包）再
  怀疑 kernel；修复须由测试 owner 做（Stage 5 不可改 `tests/`）。
- **report `partial` + CLI 非零 ≠ 集成失败**：正确性通过、benchmark 已实跑即满足
  `INTEGRATE_COMPLETED`；msprof 采集回退 events 属「仅记录」范畴，gate 5 只给 warn。判定必须读
  本次 `run.json` 的 `summary`（`correctness_passed` / `benchmark_requested` / `benchmark_passed`），
  不得凭退出码。
- **幂等集成重跑会刷新集成文件 mtime，使此前 report 失效**：`gate 5` 的 `S5-REPORT-STALE` 会要求
  run.json 不早于 wrapper/kernel/`__init__`。因此「重跑集成」与「跑 report」必须**同轮、report 在后**；
  任何在 report 之后触碰集成文件的动作都会强制一次全量 report 重跑。
- **msprof 采集失败时优先看「参照组」**：同一次 benchmark 里 tileops 与 torch 两条记录并排，
  若 torch 有效而 tileops 回退 events，则问题在 tileops 侧采集路径而非设备，可快速排除环境因素。
