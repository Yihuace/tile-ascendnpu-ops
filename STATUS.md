# 当前工作状态

> 更新：2026-11-05（**Gelu 家族收口**：P0 tile 宽度 padding + P1-3 bf16 in-place 7296 tile；
> Gelu 3-shape avg 54.38→**63.94%**、GeluTanh 54.29→**63.54%**；19-op shape 达标 67/128 → **71/126**）。
> **FALN bm=2 原型**：89.20µs（−23.4%）/ 75.6GB/s，但 fp32 affine 差 16KB UB，未采纳；现网仍 bm=1，5-shape avg **45.73%**（1/5）。
> **本文件是状态快照，不承载数据表**——19 接口的正确性/ratio/达标一律以
> [`perf_data/canonical.json`](perf_data/canonical.json) 为准（由 `data_surface_lint.py` 校验）；
> [`FINAL_DELIVERY_20261018.md`](examples/TileOPs/reports/FINAL_DELIVERY_20261018.md) §2 仅为 10-18 批次上游。

## 结论

| 项 | 值 |
|---|---|
| 接口交付 | **19 / 19**（GDN in-scope 主路径已交付；K5/CP 与小 shape fallback 待续） |
| 正确性 | **19 / 19**（GDN `test_gdn.py` 10/10；Conv1d / Conv1dBias 27 passed + 1 skipped、0 failed） |
| 全 shape 严格达标 | **3 / 19**（Sub 15/15、ClampMin 7/7、Hardswish 6/6） |
| shape 级达标 | **71 / 126（56.3%）**（10 个微型 shape 已豁免，不计入分母；Hardsigmoid 60.26% 未达 70，BN large-spatial 60.02% 为 stats-only 不计） |
| 有生效调优 kernel | 16 / 18 |
| 单算子优化报告 | 17 份 |

## 去哪看

| 想了解 | 看这里 |
|---|---|
| 交付数据（表） | `perf_data/canonical.json` ｜ `perf_data/delivery.json` ｜ 10-18 批次上游 `FINAL_DELIVERY_20261018.md` §2 |
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

## 统计口径（用户决策 2026-10-22）

- **不更改统计口径**：交付/达标继续使用现有 `msprof` Ratio（L2 read-miss × cacheline / Task Duration / 1.75 TB/s），
  不切换到 `eval_roofline` 字节模型。
- 因此：**Add broadcast 复用保持 `BCAST_ENABLED=False`**（启用会使官方 ratio 下降、10/11→9/11）；
  Lerp/Sub 的同类广播复用暂不推广。涉及“删除冗余读”的优化不再以 ratio 提升为目标，只作为可选性能路径留档。
- 贴线 shape（Clamp 16M-min-only bf16 69.62%、Add 69.37%、Lerp 66.12%）如需复核，只在**同一口径**下增加采样，
  不改变量定义。

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）
- **所有 agent/subagent 一律 `deepseek/deepseek-flash`**（`AGENTS.md` Model Policy，禁止 pro）
- 新派发的 optimizer 必须在返回前自查 `statectl gate 4` 直到 pass
- 当前唯一默认栈：CANN 9.0.0 系统 `bishengir-compile 1.1.0` + `hivmc 0.2.0`（9.2 已物理删除）
