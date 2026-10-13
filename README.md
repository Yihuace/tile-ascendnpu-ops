# tile-ascendnpu-ops

TileOPs GPU→Ascend NPU 算子迁移工作区（`npuir-migration`）。目标：多算子迁移 + 全流程记录，主要使用 Agent 进行开发与优化。

## 目录角色

| 路径 | 角色 |
|---|---|
| `.agents/` | **工具与知识**：`skills/`（设计/开发/调优/评审/标准）、`tools/`（`statectl` 状态机 / `kb_search` / `standards_check` …）、`evolution/`（提案 `queue.md`、能力缺口 `capability-gaps.md`、`stats.md`） |
| `.opencode/` | **编排配置**：conductor 与各子 agent、`conductor-scenarios/`（migration / harness / new-op / optimize）、`command/` |
| `examples/TileOPs/` | **集成侧工程（成品）**：NPU TileOPs 工程（manifest / ops / kernels / tests / benchmarks / reports），含其专属 skill `.agents/skills/add-npu-op/` |
| `examples/<op_slug>/` | **逐算子任务工作区（过程）**：`.migration_state.json`、逐函数 `{func}/`（`.stage_state.json` / DESIGN / REVIEW / kernel / `history_version/`）与报告 |
| `MIGRATION_INDEX.md` | 迁移总索引（状态 / 正确性 / 性能 / 报告指针 / 版本戳） |
| `TileOPs迁移任务书.md` | 本仓迁移 spec（算子清单 §5、批次与出口标准 §6） |
| `AGENTS.md` | 仓库总纲（API 约定、能力缺口规则、docs 路由） |

> - `examples/TileOPs/` 同时是 `{NPU_TILEOPS}`（集成目标），其子目录 `tileops/kernels/{family}/{op_slug}/` 才是 `{OP_DIR}`；`examples/<op_slug>/` 是 conductor 的过程工作区。二者一对：过程在工作区，成果落集成侧。
> - 报告两种口径：`examples/TileOPs/reports/report.md` = 多算子统一 run；`examples/TileOPs/reports/tileops/<run_id>/` = 单算子 run。

## 迁移状态

见 [`MIGRATION_INDEX.md`](MIGRATION_INDEX.md)。已完成：`HardsigmoidFwdOp`（迁移报告 `examples/hardsigmoid/MIGRATION_REPORT.md`）。批次：1a `ClampScalarFwdOp`/`HardswishFwdOp` → 1b elementwise → 1c 谓词 → 2 `FusedAddLayerNormFwdOp` → 3 高风险。

## 约定

- 迁移（baseline）与优化（Stage 4）分离：迁移报告不写“性能已达水位”，优化另起 `optimize` 场景。
- 记录可追溯：`statectl` 维护 `.stage_state.json` / `.task_timeline.jsonl`；结论绑定版本戳。
- `.gitignore` 排除可再生大件：缓存、`.opencode/node_modules/`、`reports/**/artifacts/`（msprof 原始 dump）、`examples/hardsigmoid/full.json`。

## 已知缺口

- ~~顶层 `docs/` 缺失~~ → **已修复**：已从 tilelang 源仓拷入 `docs/`（含 `开发指南.md`、`Tilelang.language/`）与 `examples/flash_attention/flash_attn_npuir.py`，`standards_check.py check` 现为 `ok`（0 failures）。
- `kb_stale_check.py`：修正后（优先取 tilelang 包所在仓的 commit）报告 `stale_count=132`——**真实结果**（KB 条目跨多个更早的工具链 commit，当前为 `417dae9908`），非阻塞。
- **Stage 0 提取（已大幅打通）**：修复 `extract_tl_kernel.py` 的 src-layout 内部导入解析（`_resolve_abs`）+ builder 解析起点（`_build_kernel` 所在文件）+ 工厂式 op-body 绑定剔除 + 合并函数辅助依赖发现/发出。现状：16 个待迁算子**全部可提取、0 报错**——5 个走 Pattern A（自包含合并 kernel：hardsigmoid/hardswish/gelu*(2)/fused_add_layer_norm），11 个走 Pattern B（原始 `@tilelang.jit` builder，op-body 未内联）。`gen_source_map.py`（源映射生成器，覆盖 172 算子）并入 `scripts/`。
  - **残余**：Pattern B 的动态 op-body 族（`ScalarParamUnaryKernel` 的 `_make_op_func`、binary 的 `_combine`、谓词等）暂只给原始 builder 参考，Stage 0 仍需轻度人工合并 op-body。
