# tile-ascendnpu-ops

TileOPs GPU→Ascend NPU 算子迁移工作区（`npuir-migration`）。

## 目录

- `.agents/` — 方法与工具：skills、`tools/`（statectl 状态机 / kb_search / standards_check …）、`evolution/`（提案队列 queue.md、能力缺口 capability-gaps.md、统计 stats.md）
- `.opencode/` — 编排配置：conductor 与各子 agent、`conductor-scenarios/`（migration / harness / new-op / optimize）
- `examples/TileOPs/` — 集成侧 NPU TileOPs 工程（manifest / ops / kernels / tests / benchmarks / reports）
- `examples/<op_slug>/` — 逐算子迁移工作区（`.migration_state.json`、逐函数 `.stage_state.json` / DESIGN / REVIEW / kernel / 报告）
- `AGENTS.md` — 仓库总纲

## 迁移状态

- 已完成：`HardsigmoidFwdOp`（含迁移报告 `examples/hardsigmoid/MIGRATION_REPORT.md`）
- 批次计划见《TileOPs 迁移任务书》：批 1a `ClampScalarFwdOp`/`HardswishFwdOp`、批 1b elementwise、批 1c 谓词、批 2 `FusedAddLayerNormFwdOp`、批 3 高风险

> 说明：`.gitignore` 排除了可再生的缓存、`node_modules`、msprof 原始 dump（`reports/**/artifacts/`）与会话导出 `full.json`。
