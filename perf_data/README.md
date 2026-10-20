# perf_data —— 19 算子性能数据的唯一事实源

## 这是什么

`canonical.json` 是 **cann9.0 分支上 19 个算子的唯一可信性能数据集**：每个算子只有一条记录，
交付态只有一份口径（**CANN 9.0.0**）。它由 `.agents/tools/gen_perf_data.py` 从下面两类输入合并而成：

| 输入 | 作用 |
|---|---|
| `delivery.json` | **交付态（权威）**：19 接口最新 9.0 实测，逐行转录自 `examples/TileOPs/reports/FINAL_DELIVERY_20261018.md` §2（2026-10-18 端到端重测表）+ GDN 的 2026-10-19 per-kernel 汇总 |
| `manual.json` | **迁移态补充**：机器记录里没有 baseline 的算子（见下） |

文档（`FINAL_DELIVERY_20261018.md` §2、`MIGRATION_INDEX.md` §A）
与仓库外交付件 `../19_operators_ratio.xlsx`（由 `.agents/tools/gen_ratio_xlsx.py` 生成）
的数字必须与它一致，由 `.agents/tools/data_surface_lint.py` 一起校验。

## 口径

| 项 | 值 |
|---|---|
| 工具链 | CANN 9.0.0 / Ascend910B3 |
| ratio 定义 | `msprof op` Roofline **GM Read+Write SOL / 实测 Task Duration**（端到端 `tileops-report`） |
| 达标线 | 访存 70% / 算力 60% / Mix 40%（GDN 未定义达标线） |
| 可调 shape | Task Duration > 5 µs（微型 shape 豁免，不计入分母） |
| 达标 | 交付态 ratio ≥ 达标线 |

两个列：**迁移态** = 优化前；**交付态** = 当前交付。两列都是 9.0.0，但**可能来自不同轮次**，
区间覆盖的 case 集合也可能不同——**不可直接横比**，判断收益要看同 case。

## 交付态从哪来（为什么是快照而不是 run.json）

2026-10-18 那批端到端重测的 per-run 报告目录**被 `.gitignore` 忽略、未入库**，仓库里没有对应
`run.json`；唯一完整记录是 `FINAL_DELIVERY_20261018.md` §2 的表。因此 `delivery.json` 是逐行转录的
机器快照，`source.kind = "delivery_20261018"`。GDN 例外：它没有 msprof op ratio，取
`reports/optimize/GatedDeltaNetPrefillFwdOp/OPTIMIZE_REPORT.md` §3 的 per-kernel 汇总。

## 迁移态从哪来

`manual.json` 的 `migrated` 段，**只补缺口、不覆盖交付态**，登记四类补充来源：

| `kind` | 用于 | 说明 |
|---|---|---|
| `stage4_bench_log` | Conv1d / Conv1dBias | 优化前 benchmark 全量记录（`perf_opt/logs/bench_all.json`，2026-10-17T15:38）；只取与交付态同 case 且有 ratio 的部分 |
| `identity` | Hardswish / Lerp | 无 perf_opt 内核、wrapper 未切换 → 迁移态 ≡ 交付态 |
| `migrated_bench_run` | ClampScalar / ClampMin | 2026-10-20 补测：wrapper 临时切到 Stage-3 baseline，只跑 benchmark，跑完按字节还原（`reports/tileops/*_migrated/PROVENANCE.md`）；排除 smoke 用例 |
| `official_baseline_run` | Conv3dBias | 2026-10-18 官方迁移态 run（`git_commit 8109d785`，wrapper 仍 import `_conv3d_kernel`）；正确性 17/17、benchmark 4/4，3 个 case 均为 msprof Roofline ratio |

其余 11 个算子的迁移态来自各自 `perf_records.jsonl` 的 `phase == "baseline"`（每个 workload 取最后一轮）。
Conv3dBias 的迁移态来自 `20261018_101545_935539_Conv3dBiasFwdOp` 官方 run：r3d **1.2111%**、video **6.291%**、unet **0.757%**，0.76–6.29%。早期 0.25% 自算 SOL 已废弃。

## 溯源等级（如实记录）

| 数据 | 溯源到什么级别 |
|---|---|
| **迁移态** | **逐 case**：11 个来自已入库的 `perf_records.jsonl`（`phase=baseline`）；Conv1d/Conv1dBias 来自已入库的 `bench_all.json`；ClampScalar/ClampMin 来自 `manual.json`；Hardswish/Lerp 为恒等；Conv3dBias 来自其官方 baseline run |
| **交付态 · 15 行** | **聚合快照**（`FINAL_DELIVERY_20261018.md` §2 的 min–max + 达标 a/b）；10-18 那批 per-run 目录未入库、工作区已删除，**逐 case 不可恢复** |
| **交付态 · 4 行含微型的** | **逐 case**：取自各自 `reports/optimize/<Op>/OPTIMIZE_REPORT.md` §3 的 final SOL，**已剔除 ≤5µs 微型 shape**（Hardsigmoid / FusedAddLayerNorm / GeluAndMul / GeluTanhAndMul） |
| 逐 case 补充 | 17/19 的 `OPTIMIZE_REPORT.md` §3 有优化轮逐 case 表（ClampMin 由 ClampScalar 覆盖、Lerp 未优化）；它们与 10-18 交付轮不是同一轮，仅作过程证据 |

> 用户口径（2026-10-20）：**不接受重测**。因此交付值保持 10-18 的实测结果：
> 15 行用聚合快照；4 行含微型的用报告里的逐 case 值重算区间（只含可调 shape）。
> 新一批实测落地时，请把逐 case 存成 `cases.json` 一并入库，避免再次不可溯。

## 特殊条目

- **GDN**：`mode = self_sol`。交付态 0.357% / 迁移态 0.027% 是 **per-kernel msprof Task Duration 汇总 +
  op-level 理论 roofline**，**不是 msprof op 的 ratio**，与其它行不可横比；迁移态由 per-head H1×16 重建（估算）。
- **Conv3dBias**：迁移态 run 中 r3d 捕获 `_conv3d_bias_main`，video/unet 捕获 `_conv3d_bias_main_base`；两者都属于 Stage-3 baseline kernel，不是优化态。
- **BatchNormBwd**：ratio 因 subid 去双计下降，与优化前不可比，以 Task Duration 为准。
- **微型 shape**：Hardsigmoid / FusedAddLayerNorm / GeluAndMul / GeluTanhAndMul 的区间含微型 shape，
  下限很低（0.4%~4%），但达标数只按可调 shape 计。

## 用法

```bash
python3 .agents/tools/gen_perf_data.py --table   # 打印 §2 风格表
python3 .agents/tools/gen_perf_data.py           # 重新生成 perf_data/canonical.json
python3 .agents/tools/gen_perf_data.py --check   # CI：文件过期则非零退出
python3 .agents/tools/gen_ratio_xlsx.py          # 由 SSOT 重建交付件 19_operators_ratio.xlsx
python3 .agents/tools/data_surface_lint.py       # 校验文档数字与 SSOT 一致、且不含历史版本号
```

## 纪律

- **不要手改 `canonical.json`**：它是生成物。交付态要改就更新 `delivery.json`（并注明新的实测来源），
  迁移态要改就更新 `manual.json`，然后重跑生成器。
- **一个算子一份数据**：不允许同一算子在文档里出现两套口径。
- **新一批实测落地时**：把新的端到端表更新到 `delivery.json`，同时检查 `FINAL_DELIVERY` /
  `MIGRATION_INDEX` / `STATUS` / 总报告 / xlsx，跑一遍 lint。
- **机器记录也不许带旧版本戳**：`perf_records.jsonl` 属数据面（lint 会扫）。某轮测量若发生在工具链升级前，
  `toolchain` 字段写「上一代工具链（…）」并加 `superseded_by` 指向 9.0.0 的官方报告。
