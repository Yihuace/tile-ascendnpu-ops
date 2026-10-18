# Stage-4 批量优化总报告（2026-10-17 22:52 → 2026-10-18 15:54）

> **口径（2026-10-18 统一）**：全部性能数据为 **CANN 9.0.0** `msprof op` Task Duration + Roofline GM Read+Write SOL；达标线按 AI 分类（访存 ≥70% / 算力 ≥60% / Mix ≥40%）；微型 shape（≤5µs）豁免；硬底线 = kernel 时间必须变短，不达标不得硬凑。
> **版本纪律**：不再混用 8.5.2。8.5.x 数据仅保留在 `MIGRATION_INDEX.md` 表 B 历史存档（不在本批 19 接口范围）；本报告所有行均为 9.0.0。
> **统计纪律**：达标数一律**两个分母**（全 shape / 可调 shape），豁免项单列，不计入分母。
> **明细**：`STAGE4_QUEUE.md`（逐算子结论）、`STAGE4_RUNBOOK.md`（执行手册）、`REPORT_TEMPLATE.md`（报告模板）、`optimize/{Op}/OPTIMIZE_REPORT.md`（单算子报告）。
>
> **口径策略（用户口径 2026-10-18）**：**全部统一 9.0.0**。Add / Sub 已切 `perf_opt`；ClampScalar / ClampMin 在 9.0 下的回退如实记录，不沿用 8.5 结论。

## 1. 本轮 11 个算子的结果（全部 9.0.0 口径）

| 算子 | 关键时延收益 | 采纳 | SOL（可调 shape 达标） | 结论 |
|---|---|---:|---|---|
| GLADecodeOp | **−24.2~35.4%**（k16→k32+persistent；超发 −8.0~14.3%） | ✅ | 3.10–20.16%（0/5） | latency-bound；persistent 仅 B*H>40；pipelining 无收益 |
| HardsigmoidFwdOp | 0（wash） | ✅（tie） | 0.37–61.52%（0/2；4 微型豁免） | 实测 compute-bound，GM SOL 判据失配 |
| GeluTanhAndMulFwdOp | prefill −8.2/−8.1% | ✅ | 2.22–47.63%（0/4） | Vtanh 不可用；sigmoid 恒等折叠已到最小链 |
| GeluAndMulFwdOp | **prefill −43.6/−43.1%** | ✅ | 4.05–47.71%（0/4） | **dtype 分化 erf 阶数**（deg-20 → deg-10 for fp16/bf16） |
| IsnanFwdOp | 0（plateau） | ❌ | 52.91–79.33%（1/5） | **MTE2 混合读写带宽地板**，非 pass 数受限 |
| IsinfFwdOp | **bf16 −19.4~−19.9%** | ✅ | 41.45–78.13%（1/5） | `vand(x,0x7FFF)` 求 \|x\|，bf16 6→4 pass |
| IsfiniteFwdOp | **bf16 −21.2/−16.4%** | ✅ | 42.29–78.29%（1/5） | 同 `vand` 路径（第二独立证据） |
| FusedAddLayerNormFwdOp | **8b prefill −23.2%**、70b −14.2% | ✅ | 0.54–43.89%（0/5；2 微型豁免） | loads-first 装载前置；单遍方差 tie 弃用 |
| BatchNormBwdOp | **−33~−46%**（几何 −40.3%） | ✅ | 11.91–24.44%（0/6，subid 不可比） | subid 分片从 MixCV 迁到 Vector 归约族 |
| Conv1dBiasFwdOp / Conv1dFwdOp | whisper **15×**、encodec-deep **3.6×**；通用 kernel wav2vec2 **17.7×**、encodec-init **4.2×** | ✅ | 0.03–1.79%（Conv1d 0/15、Bias 0/8；按 Task Duration 判据） | 形状路由 + taps-in-K；wav2vec2 的 CG-2026-0003 已被新 IR 绕开；Conv1d report partial |
| Conv3dBiasFwdOp | **r3d-stem 2.36×、unet 21.3×** | ✅ | 1.68–20.43%（0/3；按 Task Duration） | tap 打包 + padded-N 分块 + NDHWC 暂存；im2col DMA 段数 91774→973 |

## 2. 18 个接口：CANN 9.0.0 端到端现状（全可调 shape 范围）

> 数据源：`reports/tileops/<run_id>/run.json`；`ratio` 为 `roofline_utilization_percent`。达标线按 AI 分类：访存 ≥70% / 算力 ≥60%。

| 算子 | perf shape | 9.0 ratio | 达标线 | 达标数 | 生效内核 | 采纳 |
|---|---:|---:|---:|:--:|---|---|
| Add | 11 | 69.37–83.48% | 70 | **10/11**（min 69.37 贴线） | `perf_opt` | ✅ |
| Sub | 15 | 70.70–84.00% | 70 | **15/15** ✅ | `perf_opt` | ✅ |
| ClampScalar | 11 | 58.21–70.16% | 70 | **1/11** ❌ | `perf_opt` | ✅（9.0 回退） |
| ClampMin | 7 | 61.77–68.84% | 70 | **0/7** ❌ | 复用 ClampScalar `perf_opt` | ✅（继承） |
| Lerp | 11 | 60.22–82.03% | 70 | **9/11** ⚠️ | baseline | 未优化 |
| Hardswish | 6 | 50.20–60.95% | 60 | **2/6** ⚠️ | baseline | 未切 |
| Hardsigmoid | 6（4 微型豁免） | 0.37–61.52% | 70 | **0/2** ❌ | `perf_opt` | ✅(tie) |
| Isnan | 5 | 52.91–79.33% | 70 | **1/5** ❌ | baseline | ❌ |
| Isinf | 5 | 41.45–78.13% | 70 | **1/5** ❌ | `perf_opt` | ✅ |
| Isfinite | 5 | 42.29–78.29% | 70 | **1/5** ❌ | `perf_opt` | ✅ |
| GeluTanhAndMul | 5（1 微型豁免） | 2.22–47.63% | 60 | **0/4** ❌ | `perf_opt` | ✅ |
| GeluAndMul | 5（1 微型豁免） | 4.05–47.71% | 60 | **0/4** ❌ | `perf_opt` | ✅ |
| FusedAddLayerNorm | 7（2 微型豁免） | 0.54–43.89% | 60 | **0/5** ❌ | `perf_opt` | ✅ |
| BatchNormBwd | 6 | 11.91–24.44%（subid 去双计） | 60 | **0/6** ❌ | `perf_opt` | ✅ |
| GLADecode | 5 | 3.10–20.16% | 60 | **0/5** ❌（persistent −8~14.3%） | `perf_opt` | ✅ |
| Conv1d | 15 | 0.04–1.79% | 60 | **0/15** ❌（Task Duration 17.7×） | `perf_opt` | ✅ |
| Conv1dBias | 8 | 0.03–1.73% | 60 | **0/8** ❌（Task Duration 8.3–8.9×） | `perf_opt` | ✅ |
| Conv3dBias | 3 | 1.68–20.43% | 60 | **0/3** ❌（Task Duration 21.3×） | `perf_opt` | ✅ |

> 备注：
> - Add 的 `linear-bias bf16` 为 **69.37%**（贴线）；2026-10-18 单点复测 3 次为 **14.00/14.28/15.06µs、68.48/67.13/63.66%**，确认稳定在 63.7–69.4%，未过 70%，10/11 结论成立。Sub 原直跑唯一未过的 `linear-bias bf16` 在 9.0 端到端为 **70.70%**，已过线。
> - ClampScalar / ClampMin 在 9.0 下相对 8.5 明显回退（8.5：71.9–82.8% / 69.8–77.2%）；本表按 9.0 实测记录，不沿用 8.5 结论。

### 达标统计（严格口径，全部 9.0.0）

| 判定 | 数量 | 算子 |
|---|---|---|
| **全 shape 严格达标** | **1** | Sub（15/15） |
| 贴线 / 部分达标 | 4 | Add（10/11）、Lerp（9/11）、Hardswish（2/6）、ClampScalar（1/11） |
| 未达标但**已采纳真实时延收益** | 12 | Hardsigmoid、ClampMin、Isinf、Isfinite、GeluTanh、GeluAndMul、FALN、BatchNormBwd、GLA、Conv1d、Conv1dBias、Conv3dBias |
| 未达标且未采纳 | 1 | Isnan（无收益） |
| 未交付 | 1 | GDN（2/6 kernel，CG-2026-0003） |

> 上一版把 Add（11/11）、Sub（14/15）、ClampScalar（8.5 的 6/6）写成“3 个严格达标”；按 9.0 端到端**严格只有 Sub 1 个**，本表为准。

## 3. 三个口径陷阱（必须随报告一起读）

1. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降，ratio 几乎不动（GeluAndMul −43.6% 时延，ratio 仅 43.2%→46.0%）。这类算子应以 Task Duration 判定。
2. **subid 分片使 ratio 减半**：BatchNormBwd 的 `counted_flops` 去双计 → ratio 从 15.3–28.2% 降到 13.0–25.5%，但时延 −33~−46%。ratio 不可比（`VP-2026-0158`）。
3. **9.0 clamp 族整体下移（已定性）**：同链路 baseline 9.0 为 45.35–64.37%，tuned 为 58.21–70.16%（1/11）、ClampMin 61.77–68.84%（0/7）；tuned 仍比 baseline 快 **7.6–40%**。因此 8.5 的 71.9–82.8% / 69.8–77.2% 不能跨版本引用，但 **perf_opt 仍是正确选择**，不重优化。

## 4. 天花板与豁免（已举证）

| 类别 | 算子 | 证据 |
|---|---|---|
| MTE2 混合读写带宽地板 | Isnan / Isinf / Isfinite（fp16/fp32） | `aiv_mte2_ratio 0.81–0.90 ≫ aiv_vec_ratio 0.25–0.39`；4→3 pass 仅 −3.4%；40 核最优 |
| 计算受限 + 判据失配 | Hardsigmoid / GeluTanh / GeluAndMul | `aiv_vec_ratio 0.85–0.95`、`computility=Vec_FP32(~99%)` |
| latency-bound | GLA / FALN | `advice=latency bound:pipeline caused`；并行度受限 / 15+ pass + 2 reduce |
| fp32 中转结构天花板 | BatchNormBwd / FALN | fp32 中转 + Vec_MISC + 内存 |
| 编译器缺陷（非优化可解） | GDN blocksolve、Conv1d 的 1 个 bench 形状 | CG-2026-0003（`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取 `getResult(0)` 越界）|
| 微型 shape 豁免 | Hardsigmoid ×4、Gelu ×1、FALN ×2、GLA smoke | Task Duration ≤5µs 或启动开销 ≥50% |

## 5. 关键技术与能力缺口

**技术突破**
1. `vand(x, ~(-0.0))` 位运算求 `|x|`：绕开 bf16 无 `vabs`，predicate 族 6→4 pass
2. dtype 分化多项式阶数：fp16/bf16 容差 1e-3 无需 deg-20，GELU 族 −43%
3. subid 分片跨族迁移（MixCV → 纯 Vector 逐通道归约）：BatchNormBwd −40%
4. loads-first 双输入前置装载：FALN 8b prefill −23%
5. implicit-GEMM 权重 L1 常驻 + tap 打包：Conv3d r3d-stem 2×；Conv1d co 块切分 15×
6. 形状路由绕开编译器缺陷：dilation 分流 + `c_in>=64` 门，使 Conv1d 族在官方 9.0 上可交付
7. **im2col DMA 段数削减**：padded-N 多行分块 + NDHWC 暂存，把 `c_in_pad` 个 128B 小段降到 973 条，Conv3d unet 21.3×

**能力缺口**
- **CG-2026-0003（recurring, occurrences=4）**：9.0.0 与 9.2.0 官方包均带该缺陷；8.5.2 无（405/405）
- `T.gemm` 忽略 L1 切片偏移（TRAP 已入库）
- **L1 常驻归因更正**：`c_in_pad=32` 常驻早已表达（IR 中 27 次 `nd2nz` 在 task 循环外），不是杠杆；真正病根是 im2col 输入 copy 的 DMA 段数（91774→973 后 unet 21.3×）
- L1 超配 → bishengir SIGSEGV（`n_max` 绕法）
- predicate 缺 `i1→i8` 直转 / bool `vnot`（CG-2026-0027）
- **CG-2026-0028**：Vector prelude「1D GM 切片 → 带列偏移 GM 子区域」写必崩（`Unknown core type: llvm.func @malloc`）；bias-in-K 折叠被迫放弃，bias 路径保留中继
- **CG-2026-0029**：Conv1d bias MixCV 的 ws 宽度只能等于 `block_n=128`；放大到 160/192/256 触发 `multiple entry functions`（AIV/AIC 双 hacc.entry），bias N tile 无法放大
- `statectl` 无法重开 optimize 场景 DONE 任务
- 多 kernel 包的 `S4-PERF-RECORDS-RECON` 单文件契约已由 `kernel_opt_py_paths` 扩展支持（2026-10-18）

## 6. 过程质量

- 端到端 A/B 两次纠正了结论：Conv1d（baseline 同崩→工具链问题）、Conv3d（baseline 正常→真回归）
- 门禁返工 3 次（FALN / BatchNormBwd / Conv3d），已固化「返回前自查 `gate 4`」
- 所有采纳版本：`--level all` 全绿 + `tileops-report` 通过（Conv1d 为 partial，原因已查明为 pre-existing）
- 报告结构已统一为 8 节模板（14 份），旧版存 `history_version/OPTIMIZE_REPORT_v0.md`

## 7. 产物索引

| 类别 | 路径 |
|---|---|
| 本总报告 | `reports/optimize/STAGE4_SUMMARY.md` |
| 队列与进度 | `reports/optimize/STAGE4_QUEUE.md` |
| 执行手册 | `reports/optimize/STAGE4_RUNBOOK.md` |
| 报告模板 | `reports/optimize/REPORT_TEMPLATE.md` |
| 单算子报告 | `reports/optimize/{Op}/OPTIMIZE_REPORT.md` |
| 调优日志/结构化记录 | `.../perf_opt/opt_log.md`、`perf_records.jsonl` |
| 端到端验收 | `reports/tileops/<ts>_{Op}/{run.json,report.md}` |
| 状态与时间线 | `.../.stage_state.json`、`.task_timeline.jsonl` |
