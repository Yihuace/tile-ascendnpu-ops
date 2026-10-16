# 交付清单索引（D1–D11 / §4 报告骨架）

任务书 §3 的 D1–D11 为**逐算子**证据项，§4 为**每算子一页六节报告**。下表为**本批**的落位索引（按算子族/公共件），便于逐项核对。

## D1–D11 证据落位
| 项 | 内容 | 落位 |
|---|---|---|
| D1 | 路由与判定（scenario/mode/stage_plan） | 各算子 `kernels/**/.migration_meta.json`（harness 算子）；薄接口见 `docs/thin_interface_status.md` |
| D2 | 流程与时间线（`statectl show/verify`、`.task_timeline.jsonl`） | ⚠️ 未产出（本批未走 statectl/.task_timeline）——**待补**（见下） |
| D3 | Stage 0 产物（7 文件 + `.migration_meta.json`） | 各 `kernels/<family>/<op>/.migration_meta.json`（14 个 harness 算子）；4 个直接实现见 A1 |
| D4 | Tier 1 结构校验 | `pytest tests/ops/test_<op>.py --collect-only`；`python -c "import tileops"` |
| D5 | 设计与检视（DESIGN.md / REVIEW.md + `结论:`) | 部分：`kernels/**/DESIGN.md`（mish/logsumexp 等）；本批多数 kernel 设计记录在 `perf_opt/opt_log.md` 与集成 `integration_log.md` |
| D6 | 精度门禁（L0/L1 实测） | 各 kernel 文件内嵌 `--level all`；`perf_opt/*.py --level all` |
| D7 | 集成护栏（wrapper_rewritten / already integrated / `__module__`） | `kernels/**/integration_log.md`（harness 算子） |
| D8 | 正确性事实 | `reports/tileops/<run>/report.md|run.json`（`correctness_*`）；`tests/ops/test_<op>.py` |
| D9 | 性能事实（msprof per-shape 表 + 计数 + 用例构成） | `reports/tileops/<run>/report.md|run.json`；`reports/横向对照_SOL.md`；优化：`reports/optimize/<Op>/` |
| D10 | 版本戳 | 报告 `environment`（含 tilelang+CANN+设备+torch，已补 tilelang build/commit）；见 `docs/cann9_adaptation.md` |
| D11 | 未决项与风险（gap） | `.agents/evolution/capability-gaps.md`（CG-2026-0018..0026） |

## §4 报告骨架（每算子一页六节）
- 每算子的 **§1–§4 节**（身份/流程/正确性/性能）见 `reports/tileops/<run>/report.md`（含 setup、correctness、per-shape 性能表、warnings）。
- **本批统一横评**：`reports/迁移与优化总报告.md` + `reports/横向对照_SOL.md`。
- **§5 未决项**：`capability-gaps.md`；**§6 复核入口**：各 `report.md` 末尾 + 各 `perf_opt/opt_log.md`。

## 待补（若要严格对齐 §3/§4）
1. **D2**：为每个 harness 算子生成 `.task_timeline.jsonl` 或补 `statectl show/verify` 片段；
2. **D5**：为直接实现的算子（conv3d/gla_decode 等）补 `DESIGN.md`/`REVIEW.md`；
3. **D9**：benchmark 用例构成表（收集数 vs 进 msprof 数）——`msprof_case_count` 已在 run.json，可补入报告节。
