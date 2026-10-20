# 当前工作状态

> 更新：2026-10-20。**本文件是状态快照，不承载数据表**——19 接口的正确性/ratio/达标一律以
> [`FINAL_DELIVERY_20261018.md`](examples/TileOPs/reports/FINAL_DELIVERY_20261018.md) §2 为准
> （机器可读 `perf_data/canonical.json`，由 `data_surface_lint.py` 校验）。

## 结论

| 项 | 值 |
|---|---|
| 接口交付 | **19 / 19**（GDN in-scope 主路径已交付；K5/CP 与小 shape fallback 待续） |
| 正确性 | **19 / 19**（GDN `test_gdn.py` 10/10；Conv1d / Conv1dBias 27 passed + 1 skipped、0 failed） |
| 全 shape 严格达标 | **1 / 19**（Sub 15/15） |
| shape 级达标 | **40 / 136（29.4%）** |
| 有生效调优 kernel | 15 / 18 |
| 单算子优化报告 | 16 份 |

## 去哪看

| 想了解 | 看这里 |
|---|---|
| 交付数据（表） | `examples/TileOPs/reports/FINAL_DELIVERY_20261018.md` §2 ｜ `perf_data/canonical.json` |
| 口径陷阱 / 天花板 / 能力缺口 | `examples/TileOPs/reports/optimize/STAGE4_SUMMARY.md` |
| 19 算子台账（批次 / 状态 / 数据源 / 版本戳） | `MIGRATION_INDEX.md` |
| 剩余空间（哪些还有杠杆） | `examples/TileOPs/reports/optimize/HEADROOM_ASSESSMENT.md` |
| 逐算子逐轮过程 | `examples/TileOPs/reports/optimize/<Op>/OPTIMIZE_REPORT.md` |
| 根因与工具链排查 | `docs/cann9_adaptation.md` |

## 必须记住的四件事

1. **CG-2026-0003（recurring）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取 `getResult(0)` 越界；
   9.0.0 与 9.2.0 官方包均带，上一代工具链无。Conv1d 族已用**形状路由**绕开；GDN 用 K2 split 绕开。
2. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降（GeluAndMul −43.6% 时延、
   ratio 仅 43.2%→46.0%）；BatchNormBwd 因 subid 去双计 ratio 不可比。这两类**看 Task Duration**。
3. **口径统一 CANN 9.0.0**：上一代工具链数据只在 `MIGRATION_INDEX.md` §B 历史存档，不参与本批 19 接口结论；
   跨版本数字不横比。
4. **clamp 族 9.0 回退已定性**：是工具链/环境整体下移，不是调优失效（同链路 tuned 仍快 7.6–40%）；
   保留 perf_opt，不重优化。

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）
- **所有 agent/subagent 一律 `deepseek/deepseek-flash`**（`AGENTS.md` Model Policy，禁止 pro）
- 新派发的 optimizer 必须在返回前自查 `statectl gate 4` 直到 pass
- 当前唯一默认栈：CANN 9.0.0 系统 `bishengir-compile 1.1.0` + `hivmc 0.2.0`（9.2 已物理删除）
