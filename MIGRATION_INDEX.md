# 迁移索引（MIGRATION_INDEX）

> TileOPs GPU→NPU 迁移总览。算子清单见《TileOPs迁移任务书.md》§5/§6；此处记录**状态 / 正确性 / 性能 / 报告指针 / 版本戳**。
> 维护约定：每完成一个算子（Stage 5 通过）更新本表一行；一批完成跑一次 `tileops-report run --all` 生成统一横评。

## A. 本批待迁（任务书 §5/§6，**19 接口全量交付**；交付名 = 任务书原名）

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
| `IsinfFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 41.21%–75.58% (5 shapes) | `examples/TileOPs/reports/tileops/20261014_101823_741698_IsinfFwdOp/` |
| `IsfiniteFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 40.78%–72.06% (5 shapes) | `examples/TileOPs/reports/tileops/20261014_105757_589000_IsfiniteFwdOp/` |
| `FusedAddLayerNormFwdOp` | norm | 2 | ✅ **已完成**（harness, `phase=DONE`） | 14/14, max_abs_err 1.56e-02(bf16) | 27.52%–45.08% (7 shapes) | `examples/TileOPs/reports/tileops/20261014_115534_389335_FusedAddLayerNormFwdOp/` |
| `BatchNormBwdOp` | norm | 3 | ✅ **已完成**（harness, `phase=DONE`；3 kernel: wide/split/general） | 12/12, avg max_abs_err 2.29e-03 | ratio 15.80%–29.98%（msprof；ratio 解析缺口已修） | `examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/` |
| `Conv1dFwdOp` | convolution | 3 | ✅ **已完成**（harness, `phase=DONE`；5 kernel） | 28/28, avg max_abs_err 3.91e-02 | ratio 0.07%–0.46%（msprof；极低，待 optimize） | `examples/TileOPs/reports/tileops/20261015_162520_967795_Conv1dFwdOp/` |
| `ClampMinFwdOp` | elementwise | 1c | ✅ **已完成**（薄接口；复用 ClampScalar kernel，max=None；fp32 block_size 修正） | 14/14, max_abs_err 0.00e+00 | benchmark 10/10（fp32 UB 溢出已修，block_size 8192→4096） | `examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/` |
| `LerpFwdOp`（标量权重） | elementwise | 1c | ✅ **已完成**（薄接口；新 npuir `lerp_scalar` 二元 kernel） | 36/36, fp16≤1.95e-3 / bf16≤1.56e-2 | 见 `git show 5fd2613` | `tests/ops/test_lerp_scalar.py` |
| `Conv1dBiasFwdOp` | convolution | 3 | ✅ **已完成**（薄接口；复用已迁 Conv1d 5 kernel 的 has_bias 路径） | 28/28, max_abs_err 2.50e-01(bf16) | 见 `git show b2b4a4b` | `tests/ops/test_conv1d_bias.py` |
| `Conv3dBiasFwdOp` | convolution | 3 | ✅ **已完成**（新增 dense NCDHW implicit-GEMM + MixCV bias kernel；stride/dilation/pad 通用，W 轴 polyphase） | 17/17, max_abs_err 1.56e-02(fp16 r3d-stem) / 5.00e-01(bf16 unet-encoder, rtol) | ratio 0.58%–6.23%（msprof；r3d-stem 6.40ms） | `examples/TileOPs/reports/tileops/20261015_140650_934253_Conv3dBiasFwdOp/` |
| `GLADecodeOp` | linear_attention | 4 | ✅ **已完成**（Vector kernel；K-tile 流式 state，bf16 fp32 中转） | 11/11, worst max_abs_err 1.51e-02(bf16) | ratio 2.47%–14.12%（msprof，5 shapes） | `examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/` |
| `GatedDeltaNetPrefillFwdOp` | linear_attention | 4 | ⛔ **探针编译失败**（bishengir backend `error pointer address space cast`，CG-2026-0026）；按批 4「不可行只交付 plain 产物」口径暂不交付，待 Expert 模式重构 | — | — | `examples/gdn_prefill/probe/`；`.agents/evolution/capability-gaps.md` CG-2026-0026 |

> 基线名映射（**交付名 = 任务书原名**；基线 commit `1238a9f` 注册名为映射参考，非交付名）：`ClampMinFwdOp`→`ClampScalarFwdOp`(max=None)；`LerpFwdOp`→`LerpScalar`/`LerpTensor`；`Conv1dBiasFwdOp`/`Conv3dBiasFwdOp`→基线 conv1d/conv3d（bias 可选）；`GLADecodeOp`→`GLARecurrentFwdOp`；`GatedDeltaNetPrefillFwdOp`→`GDNFwdOp`。裸 `Conv3dFwdOp` **不在 19 内**（任务书 §5）。

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
