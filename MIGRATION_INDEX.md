# 迁移索引（MIGRATION_INDEX）

> TileOPs GPU→NPU 迁移总览。算子清单见《TileOPs迁移任务书.md》§5/§6；此处记录**状态 / 正确性 / 性能 / 报告指针 / 版本戳**。
> 维护约定：每完成一个算子（Stage 5 通过）更新本表一行；一批完成跑一次 `tileops-report run --all` 生成统一横评。

## A. 本批待迁（任务书 §5/§6，**19 接口全量交付**；交付名 = 任务书原名）

> **口径（2026-10-18 统一）**：本节所有状态 / 正确性 / 性能均为 **CANN 9.0.0** 端到端 `tileops-report` 数据；8.5.x 数据只保留在 §B 历史存档。

| 算子 | family | 批次 | 状态 | 正确性 | 性能 ratio（9.0 端到端） | 报告指针 |
|---|---|---|---|---|---|---|
| `HardsigmoidFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 7/7, max_abs_err 5.96e-08 | 0.37%–61.52% (6 shapes；下限为微型 shape) | `examples/hardsigmoid/MIGRATION_REPORT.md`；run `reports/tileops/20261018_003016_986321_HardsigmoidFwdOp/` |
| `ClampScalarFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`；9.0 下 1/11 ≥70%；tuned 仍比 baseline 快 7.6–40%） | 11/11, max_abs_err 0.00e+00 | 58.21%–70.16% (11 shapes；baseline 45.35–64.37%) | `examples/TileOPs/reports/tileops/20261018_131331_663371_ClampScalarFwdOp/` |
| `HardswishFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`；compute-bound，2/6） | 7/7, max_abs_err 4.77e-07 | 50.20%–60.95% (6 shapes) | `examples/TileOPs/reports/tileops/20261018_133632_996783_HardswishFwdOp/` |
| `AddFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`；perf_opt 已生效，10/11 ≥70%；`linear-bias bf16` 复测 63.7–69.4%） | 24/24, max_abs_err 0.00e+00 | 69.37%–83.48% (11 shapes) | `examples/TileOPs/reports/tileops/20261018_143726_708597_AddFwdOp/` |
| `SubFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`；perf_opt 已生效，15/15 ≥70%） | 24/24, max_abs_err 0.00e+00 | 70.70%–84.00% (15 shapes) | `examples/TileOPs/reports/tileops/20261018_145645_852546_SubFwdOp/` |
| `GeluAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`；1 轮设计修订） | 11/11, max_abs_err 6.25e-02(bf16) | 4.05%–47.71% (5 shapes；compute-bound) | `examples/TileOPs/reports/tileops/20261018_023257_847304_GeluAndMulFwdOp/` |
| `GeluTanhAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 11/11, max_abs_err 1.56e-02(bf16) | 2.22%–47.63% (5 shapes；compute-bound) | `examples/TileOPs/reports/tileops/20261018_013231_555821_GeluTanhAndMulFwdOp/` |
| `IsnanFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`；1 设计修订；不采纳 perf_opt） | 9/9, max_abs_err 0.00e+00 | 52.91%–79.33% (5 shapes；1/5 ≥70%) | `examples/TileOPs/reports/tileops/20261018_152024_774098_IsnanFwdOp/` |
| `IsinfFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 41.45%–78.13% (5 shapes；1/5 ≥70%) | `examples/TileOPs/reports/tileops/20261018_031842_098578_IsinfFwdOp/` |
| `IsfiniteFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 42.29%–78.29% (5 shapes；1/5 ≥70%) | `examples/TileOPs/reports/tileops/20261018_035243_081880_IsfiniteFwdOp/` |
| `FusedAddLayerNormFwdOp` | norm | 2 | ✅ **已完成**（harness, `phase=DONE`） | 14/14, max_abs_err 1.56e-02(bf16) | 0.54%–43.89% (7 shapes；下限为微型 decode) | `examples/TileOPs/reports/tileops/20261018_045246_337991_FusedAddLayerNormFwdOp/` |
| `BatchNormBwdOp` | norm | 3 | ✅ **已完成**（harness, `phase=DONE`；3 kernel: wide/split/general） | 12/12, avg max_abs_err 2.29e-03 | 11.91%–24.44%（msprof；ratio 因 subid 去双计不可比） | `examples/TileOPs/reports/tileops/20261018_060023_410172_BatchNormBwdOp/` |
| `Conv1dFwdOp` | convolution | 3 | ✅ **已完成**（5 kernel；通用 `conv1d` taps-in-K；report **partial**：1 例 CG-2026-0003 skip + 2 例 flaky bench） | **27 passed + 1 skipped**（CG-2026-0003），0 failed | 0.04%–1.79%（端到端 SOL；Task Duration wav2vec2 **17.7×/17.2×**、encodec-init **4.2×/3.4×**） | `examples/TileOPs/reports/tileops/20261018_180016_718940_Conv1dFwdOp/` |
| `ClampMinFwdOp` | elementwise | 1c | ✅ **已完成**（薄接口；复用 ClampScalar tuned kernel，max=None；9.0 下 0/7） | 14/14, max_abs_err 0.00e+00 | 61.77%–68.84% (7 shapes) | `examples/TileOPs/reports/tileops/20261018_153708_107121_ClampMinFwdOp/` |
| `LerpFwdOp`（标量权重） | elementwise | 1c | ✅ **已完成**（薄接口；新 npuir `lerp_scalar` 二元 kernel；未优化） | 36/36, max_abs_err 1.56e-02(bf16) | 60.22%–82.03% (11 shapes；9/11 ≥70%) | `examples/TileOPs/reports/tileops/20261018_135215_533806_LerpFwdOp/` |
| `Conv1dBiasFwdOp` | convolution | 3 | ✅ **已完成**（薄接口；复用 Conv1d 5 kernel 的 has_bias 路径；perf_opt 已生效） | **27 passed + 1 skipped**（CG-2026-0003），0 failed | 0.03%–1.73%（Task Duration wav2vec2-bias **8.3×/8.9×**、encodec-init-bias 1.15×/1.20×） | `examples/TileOPs/reports/tileops/20261018_180716_376489_Conv1dBiasFwdOp/` |
| `Conv3dBiasFwdOp` | convolution | 3 | ✅ **已完成**（新增 dense NCDHW implicit-GEMM + MixCV bias kernel；stride/dilation/pad 通用，W 轴 polyphase） | 17/17（run.json 未聚合 max_abs_err） | 0.76%–6.29%（msprof，3 shapes） | `examples/TileOPs/reports/tileops/20261018_101545_935539_Conv3dBiasFwdOp/` |
| `GLADecodeOp` | linear_attention | 4 | ✅ **已完成**（Vector kernel；K-tile 流式 state，bf16 fp32 中转） | 11/11（run.json 未聚合；历史 worst 1.51e-02 bf16） | 3.16%–18.24%（msprof，5 shapes） | `examples/TileOPs/reports/tileops/20261018_000805_527785_GLADecodeOp/` |
| `GatedDeltaNetPrefillFwdOp` | linear_attention | 4 | ⛔ **未交付**（6 kernel 完成 2；blocksolve 卡 CG-2026-0003 `ConvertHFusionToHIVM` SIGSEGV；早期探针为 CG-2026-0026） | — | — | `tileops/kernels/linear_attention/gdn_prefill/PROGRESS.md`；`.agents/evolution/capability-gaps.md` CG-2026-0003 / CG-2026-0026 |

> 基线名映射（**交付名 = 任务书原名**；基线 commit `1238a9f` 注册名为映射参考，非交付名）：`ClampMinFwdOp`→`ClampScalarFwdOp`(max=None)；`LerpFwdOp`→`LerpScalar`/`LerpTensor`；`Conv1dBiasFwdOp`/`Conv3dBiasFwdOp`→基线 conv1d/conv3d（bias 可选）；`GLADecodeOp`→`GLARecurrentFwdOp`；`GatedDeltaNetPrefillFwdOp`→`GDNFwdOp`。裸 `Conv3dFwdOp` **不在 19 内**（任务书 §5）。

## B. 此前已完成（不在本批清单，**历史存档**）

来源：`examples/TileOPs/reports/report.md`（统一 run `20261009_022152_829230_all`，环境 910B2C / CANN 8.5.0 / git `84519be`）。

> ⚠️ **本节为 8.5.0 历史存档，不参与本批 19 接口的 9.0 口径对照。**

> ⚠️ **2026-10-17 决策：表 B 退出本次 CANN 9.0 重定基线范围。**
> 这 7 个算子**不属于本批迁移/优化目标**，后续阶段（②正确性 / ③性能 / ④记录）**一律不再纳入**，除非另行下令；
> 本节数据**维持 8.5.0 口径不动**（不做 9.0 重测）。仅存档观察到的事实：
> `SSDChunkScanFwdOp`（test_mamba）在 9.0 下 `bishengir-compile` **挂死**（CPU 空转 >19min）、
> `MultiHeadAttentionFwdOp` 按此决策跳过、`MishFwdOp` 7/7 ✅。
> 详见 `docs/cann9_adaptation.md` §11.2。

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

- 当前工作区版本戳：tilelang `0.1.2`（dev root `/home/developer/tilelang-mlir-ascend`，无 `.git`）/ torch `2.7.1+cpu` / torch_npu `2.7.1.post4` / **CANN `9.0.0`**（`bishengir 1.1.0` + `hivmc 0.2.0`）/ Ascend910B3（可见设备 6）。
  - **2026-10-16 重定基线**：工作区自 8.5.2 升级至 CANN 9.0.0。工作流① 知识库结论重验已完（19/19 repro 探针存活，见 `docs/cann9_adaptation.md` §10）。
  - **2026-10-18 口径统一**：表 A 的 18 个已交付接口全部完成 **9.0.0 端到端重测**（Add / Sub / Isnan / ClampMin 为最后 4 个 8.5 残留）；表 B 仍为 8.5.0 历史存档，不参与本批对照。
- 状态机：`python3 .agents/tools/statectl.py show --dir examples/<op_slug>`；op 级 `migration show --dir examples/<op_slug>`。
- 报告口径：`examples/TileOPs/reports/report.md`（多算子统一 run）vs `examples/TileOPs/reports/tileops/<run_id>/`（单算子）。
- 待办缺口：顶层 `docs/` 缺失（skill 引用 `docs/开发指南.md`/`docs/Tilelang.language/` 不存在）→ `standards_check` 存量 SC-REF-PATH；补入后应转绿。
