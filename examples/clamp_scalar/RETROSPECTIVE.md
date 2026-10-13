# RETROSPECTIVE.md — clamp_scalar (op 级)

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。
> 本文件为 migration-harness Stage 5（集成复盘，op 级单份）；Stage 1/2/3 复盘在各函数目录 `clamp_scalar_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — first_integrate（migration-harness）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 wrapper 脚手架把 NPU `default_config.block_size` 硬编码为 GPU `threads×npt` 折叠值（`_DEFAULT_BLOCK_SIZE={fp32:1024, fp16:2048, bf16:2048}`），而 Stage 1 DESIGN §5.2 明确 NPU `block_size=8192` 并注明「以本设计为准」——集成前两者不一致，集成期须人工核对 wrapper config 胶水（本次已改）。脚手架未把「DESIGN 覆盖的 block_size」纳入集成校验 | wrapper 原文件 `clamp_scalar.py#_DEFAULT_BLOCK_SIZE`（备份 `clamp_scalar_kernel/history_version/clamp_scalar_s5_glue0.py`）vs `clamp_scalar_fwd_kernel_DESIGN.md#§5.2`；集成后 run.json `performance.cases[].config.block_size == 8192` | `integrate_kernel.py` 或 Stage 0 脚手架增加一条校验：若 DESIGN §5.2 声明了 `block_size` 且与 wrapper `_DEFAULT_BLOCK_SIZE` 不一致，输出 `[warn]` 并把 wrapper config 同步为 DESIGN 值（或明确要求 integrator 修正） | R |
| Integrate-flow | `integrate_kernel.py` 的 `write_integration_log` 以 `# Integration Log` 起头新建 `integration_log.md`，会覆盖 integrator 已写的 narrative（本次先把 narrative 写在日志中，脚本运行是首次写入故未覆盖；但重跑时会 append 机器段，说明脚本假定日志已存在时只 append）——流程上 integrator 的 narrative 应在脚本之后补写 | `integration_log.md` 含脚本机器段（`## integrate_kernel.py run`）+ integrator narrative 两段 | 在 integrator skill 中明确顺序：先运行 integrate_kernel.py（生成机器段）→ 再补写 narrative + debug history（追加，不覆盖机器段） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **clamp 标量边界逐元素算子（fp32 中转 + 免常数单点 NaN 恢复 + NaN-bound 退化）经 TileOPs 端到端集成实证**：正确性 11/11（max_abs_err 全 0.00e+00）、benchmark 14/14、msprof 11 case 全有效（fallback=0/invalid=0）；16M fp16 both-bounds 42.68µs / ratio 44.93%，256M fp16 878.06µs / 63.32%，16M fp32 max-only 53.40µs / 71.81%（ratio 区间 44.93–71.81%） | D | `reports/tileops/20261013_141507_226988_ClampScalarFwdOp/run.json`（summary/performance）+ `benchmark.json` | `python -m tileops.reporting.cli run --op ClampScalarFwdOp --prof-mode msprof --kernel-name main`（从 examples/TileOPs/） | tilelang 0.1.2+417dae99 dev-root + Ascend 910B3 x1（实查 AI Core 20）+ CANN 8.5.2 / 2026-10-13 | pattern-library/elementwise.md（clamp 族集成基线） |
| **`block_size=8192`（同族 elementwise 实测最优）在 fp32 全 buffer 显式分配+DEPENDENT 向量链下可编译且 msprof 有效**：wrapper config 中 `block_size=8192` 生效（run.json config 字段），fp16/bf16/fp32 三 dtype 基准全 valid，无 profiler fallback / invalid profile | D | `run.json#summary`（msprof_case_count=11, profiler_fallback_count=0, invalid_profile_count=0）+ `benchmark.md` config 列 | 同上（集成 report 即实证） | tilelang 0.1.2+417dae99 dev-root + Ascend 910B3 x1 + CANN 8.5.2 / 2026-10-13 | pattern-library/elementwise.md；佐证 CONST-capacity-910B2C（UB 预算） |

### Transferable Lessons（可迁移教训）

- **集成期必须核对 wrapper `default_config` 与 DESIGN 的 `block_size`**：migration-harness 的 Stage 0 wrapper 是 GPU 移植件，常保留 GPU `threads×npt` 折叠值（本任务 fp32=1024 / fp16=2048）；Stage 1 DESIGN §5.2 会按 NPU 实测给出新值（本任务 8192）。集成后从 run.json `performance.cases[].config` 反查实际生效的 `block_size`，若与 DESIGN 不符即改 wrapper（胶水层允许，仅 config，勿动 kernel/测试）。
- **wrapper import 改写后 `clamp_scalar_fwd_kernel.__module__` 必须指向 baseline 子包**：`tileops.kernels.{family}.{op_slug}.{op_slug}_kernel.clamp_scalar_fwd_kernel`（非 perf_opt）；这是 baseline/perf_opt 切换块是否正确的机械判据。
- **report 退出码不可作为集成判定**：本任务 report_status=passed 正常置 0，但 harness 规范明确 `partial`（benchmark 无效）也会返回非零；一切结论以本次 `run.json` 的 `summary.correctness_passed` / `status` 为准。
- **性能数字只报告不修复**：benchmark 数值（如 ratio 44.93% 的 16M fp16）低不代表集成失败——benchmark 仅报告，调优属 optimize 场景。
