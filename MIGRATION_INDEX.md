# 迁移索引（MIGRATION_INDEX）

> TileOPs GPU→NPU 迁移总览。算子清单见《TileOPs迁移任务书.md》§5/§6；此处记录**状态 / 正确性 / 性能 / 报告指针 / 版本戳**。
> 维护约定：每完成一个算子（Stage 5 通过）更新本表一行；一批完成跑一次 `tileops-report run --all` 生成统一横评。

## A. 本批待迁（任务书 §5/§6，去重后 16 个）

| 算子 | family | 批次 | 状态 | 正确性 | 性能 ratio | 报告指针 |
|---|---|---|---|---|---|---|
| `HardsigmoidFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 7/7, max_abs_err 5.96e-08 | 0.46%–55.07% (6 shapes) | `examples/hardsigmoid/MIGRATION_REPORT.md`；run `reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/` |
| `ClampScalarFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`） | 11/11, max_abs_err 0.00e+00 | 44.93%–71.81% (11 shapes) | `examples/TileOPs/reports/tileops/20261013_141507_226988_ClampScalarFwdOp/` |
| `HardswishFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`） | 7/7, max_abs_err 4.77e-07 | 49.89%–60.61% (6 shapes) | `examples/TileOPs/reports/tileops/20261013_150641_447235_HardswishFwdOp/` |
| `AddFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 24/24, max_abs_err 0.00e+00 | 66.12%–82.16% (11 shapes) | `examples/TileOPs/reports/tileops/20261013_162043_457669_AddFwdOp/` |
| `SubFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 24/24, max_abs_err 0.00e+00 | 68.43%–82.70% (15 shapes) | `examples/TileOPs/reports/tileops/20261013_170400_275674_SubFwdOp/` |
| `GeluAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`；1 轮设计修订） | 11/11, max_abs_err 1.43e-06(fp32)/0.0312(bf16) | 4.77%–43.99% (5 shapes) | `examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/` |
| `GeluTanhAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 11/11, max_abs_err 0(fp32)/1.56e-02(bf16) | 2.46%–49.73% (5 shapes) | `examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/` |
| `IsnanFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`；1 设计修订） | 9/9, max_abs_err 0.00e+00 | 51.91%–74.40% (5 shapes) | `examples/TileOPs/reports/tileops/20261014_093134_398669_IsnanFwdOp/` |
| `IsinfFwdOp` | elementwise | 1c | ⬜ 未开始（同上） | — | — | — |
| `IsfiniteFwdOp` | elementwise | 1c | ⬜ 未开始（同上） | — | — | — |
| `FusedAddLayerNormFwdOp` | norm | 2 | ⬜ 未开始（复用 row-reduction 模板） | — | — | — |
| `BatchNormBwdOp` | norm | 3 | ⬜ 未开始（高风险，先探针） | — | — | — |
| `Conv1dFwdOp` | convolution | 3 | ⬜ 未开始（高风险） | — | — | — |
| `Conv3dFwdOp` | convolution | 3 | ⬜ 未开始（高风险） | — | — | — |
| `GLARecurrentFwdOp` | linear_attention | 3 | ⬜ 未开始（高风险） | — | — | — |
| `GDNFwdOp` | linear_attention | 3 | ⬜ 未开始（高风险） | — | — | — |

**别名 / 不单列**（任务书 §5 标 ❌）：`ClampMinFwdOp`→`ClampScalarFwdOp`；`LerpFwdOp`→`LerpTensorFwdOp`（已迁，见 B）；`Conv1dBiasFwdOp`→`Conv1dFwdOp`；`Conv3dBiasFwdOp`→`Conv3dFwdOp`；`GLADecodeOp`→`GLARecurrentFwdOp`；`GatedDeltaNetPrefillFwdOp`→`GDNFwdOp`。

## B. 此前已完成（不在本批清单）

来源：`examples/TileOPs/reports/report.md`（统一 run `20261009_022152_829230_all`，环境 910B2C / CANN 8.5.0 / git `84519be`）。

| 算子 | 正确性 | avg max_abs_err | shapes | ratio 范围 |
|---|---|---|---|---|
| `AdaLayerNormFwdOp` | 22/22 | 1.13e-02 | 5 | 0.41%–69.38% |
| `ArgmaxFwdOp` | 42/42 | 0.00e+00 | 5 | 5.44%–41.39% |
| `LerpTensorFwdOp` | 8/8 | 5.21e-08 | 7 | 67.38%–80.61% |
| `LogSumExpFwdOp` | 23/23 | 3.67e-08 | 6 | 3.23%–36.61% |
| `MishFwdOp` | 7/7 | 2.45e-05 | 6 | 54.01%–64.87% |
| `MultiHeadAttentionFwdOp` | 6/6 | 8.54e-04 | 8 | 14.34%–23.44% |
| `SSDChunkScanFwdOp` | 4/4 | 2.19e-04 | 11 | 0.13%–20.30% |

## C. 版本戳与状态文件

- 当前工作区版本戳：tilelang `0.1.2`（commit `417dae9`）/ torch `2.7.1+cpu` / torch_npu `2.7.1.post4` / CANN `8.5.2` / Ascend910B3。
- 状态机：`python3 .agents/tools/statectl.py show --dir examples/<op_slug>`；op 级 `migration show --dir examples/<op_slug>`。
- 报告口径：`examples/TileOPs/reports/report.md`（多算子统一 run）vs `examples/TileOPs/reports/tileops/<run_id>/`（单算子）。
- 待办缺口：顶层 `docs/` 缺失（skill 引用 `docs/开发指南.md`/`docs/Tilelang.language/` 不存在）→ `standards_check` 存量 SC-REF-PATH；补入后应转绿。
