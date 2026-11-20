# 性能反馈（[DESIGN_LIMIT]）

- 算子: `ErfFwdOp` / `erf_fwd_kernel`（project=`erf`，`examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/`）
- 任务: `erf-erf_fwd_kernel-20261013T131205Z`（scenario=`optimize`，Stage 4）
- 时延口径: `msprof op` Task Duration(us)（median of 20 launches，warm-up 5）

## 触发判定

- 设计层归因: `erf_fwd_kernel_DESIGN.md` §1.6.0 算法调研 R1 候选表 —— 采纳候选 2「自建 A&S 7.1.26 有理-指数逼近」（21-op v-prefix 链），并在 R4 中把候选 7「tanh 型逼近」淘汰（理由原文：「容差内等价但误差 ~1e-3，达不到 fp32 门禁」，公式限定为 `erf(x)≈tanh(αx+βx³)`，仅 2 个系数）；调研结论原文「选定候选 2（自建 A&S 7.1.26 有理-指数逼近），**无更优可行替代**」。§1.6.0 R3 复杂度表据此把本算子的 op 数下界定为 21（fp32）/23（fp16·bf16），§5.2/§9 亦把「21-op 链的发射成本」登记为固有代价。
- 结构性加速估计: **2.15x – 2.75x**（依据 = 本次同会话实测外推，非估算）。同一 `msprof op` 口径下 baseline → final：16m f16 256.45→117.05（2.19x）、16m bf16 256.72→119.25（2.15x）、16m f32 308.10→112.08（2.75x）、64m f16 1015.60→457.77（2.22x）、64m bf16 1016.85→466.69（2.18x）、256m f16 4058.63→1822.15（2.23x）、256m bf16 4060.51→1857.83（2.19x）。全部 > 2x 门槛。
- 实测与设计假设直接矛盾: 对 §1.6.0 R4 候选 7 的淘汰依据直接证伪——tanh 型 `erf(x)≈tanh(x·Q(x²))` 在把 Q 提升到 **k=4（5 系数，P 为 9 次奇多项式）** 时，设备实测 fp32 `max_abs=2.98e-6`（门禁 atol/rtol 1e-5 → **3.4x 余量**，通过）；提升到 **k=6（7 系数，P 为 13 次奇多项式）** 时 `max_abs=6.2e-8`（**>160x 余量**）。该族 op 数仅 12（fp32）/14（fp16·bf16），远低于设计下界的 21/23。即：候选 7 的错误结论是**系数个数不足（2）导致**，而非 tanh 族本身精度不足。

## 参照锚定

- 参照实现: 本仓**未找到**比冻结设计更优的 erf kernel 参照——`examples/erf/erf_fwd_kernel/`（迁移工作区）、`examples/TileOPs/tileops/kernels/elementwise/erf/`（集成包）与 pattern-library `CASE-elementwise-erf-migration` 记录的**唯一实现即为被评估的 A&S 21-op 基线本身**；`testing/npuir/` 与 `examples/elementwise/` 无 erf 元素级实现；`torch.erf` / `torch_npu` 无可在 TileLang npuir kernel 内调用的设备元算（`T.verf` 为 7 阶 Maclaurin 截断，`repro/TRAP-verf-maclaurin-truncation.py` + CG-2026-0015），故无法作为可移植参照。本条的「参照」即被推翻的冻结设计族自身；结论限定于实测过的结构族（op-count ≥ 12 的纯 Vector 一维逐元素 erf 链）。
- 不可移植论证: 无可移植的更优参照；反向论证（为何设计族之外的 tanh-奇多项式族反而更优、且不违反设计已核实的硬约束）为实测结论：① 设计 R4 已核实 `T.vtanh` 在本设备 Developer 模式为**真 tanh**（`max_abs=1.192e-7`），所以 vtanh 路线不被设备能力否决——被否决的只是「2 系数的 tanh 拟合精度」；② 设计 §1.6.1 #1 为 A&S 链采纳的 sign 处理（`vcmp`+`vselect`+`vmul`）在本次逐 op 消融实测中，`vselect` 单项即占 16m-f16 链时延约 59µs / 23%（约 6.5x 单个 vmul 的代价，见「实测」），而 tanh-奇多项式因 `tanh(odd)=odd` 天然消掉全部 sign/abs 处理；③ 设计 §1.6.0 R3 的「全部候选访存同为下界 2N·elem_bytes、差异只在 op 数」判断成立，本优化正是沿该判断把 op 数从 21 降到 12，方向与设计一致，属 op 数下界的重新标定而非破坏设计前提。**未找到可直接移植参照，天花板/下界结论置信度如实降级为「本设备实测外推」**。

## 假设

`erf_fwd_kernel_DESIGN.md` §1.6.0（算法调研）与 §1.6.1（数学等价优化）：

1. 候选表 R1 #7：`tanh 型逼近`公式 = `erf(x)≈tanh(αx+βx³)`（2 系数），R4 结论「逼近误差 ~1e-3 超 fp32 门禁（模型知识；未文档化假设 + 数学分析）」→ 淘汰。
2. R3 复杂度表把采纳候选 2 的 op 数记为 **21**（fp32）/23（fp16·bf16），并注明候选 2/3 支配、其余候选 op 更多或精度不足。
3. 调研结论：「选定候选 2 …**无更优可行替代**」；§1.6.0 不确定性声明与 §9 风险 R-3 明确「唯一根本解是 CG-2026-0015 编译器层修复（候选 8）；block_size 是摊薄发射的第一杠杆，但因 UB 预算放大空间有限」。

## 实测

全部为 `msprof op` Task Duration(us) 口径（median of 20 launches，warm-up 5，`--aic-metrics=BasicInfo,PipeUtilization,...`；raw profile 见 `perf_opt/profiles/`）。

**(a) baseline 瓶颈判读（16m f16 与 256m f16）**：AIV `aiv_vec_ratio ≈ 0.79`、`aiv_mte2_ratio ≈ 0.25`、`aiv_mte3_ratio ≈ 0.12`、`aiv_scalar_ratio ≈ 0.11` —— kernel 是**纯 Vector 计算/发射受限**，非 MTE2 搬运墙（与调度背景提示一致）。

**(b) 逐 op 代价消融（同结构探针，16m f16，`perf_opt/probe_chain.py`，raw `profiles/probe/`）**，用于定位高代价 op：

| 变体 | Task Duration(us) | 差分归因 |
|---|---:|---|
| full（A&S 21/23-op 链） | 256.30 | — |
| no_exp（`vexp`→`vmul`） | 251.66 | `vexp` ≈ 13.8µs |
| no_rec（`vrec`→`vmul`） | 242.60 | `vrec` ≈ 22.9µs |
| short_horner（9-op Horner→1 op） | 183.08 | 普通 op ≈ **9.15µs/op** |
| no_sign（去 `vselect`+`vmul`） | 197.16 | **`vselect` ≈ 59.1µs（≈6.5x 普通 op）** |
| only_copy（仅搬入/搬出+cast） | 76.60 | 搬运地板 |

结论：**`vselect`（bool 掩码选择）是链上最贵的单 op**，A&S 的 sign 处理（`vcmp`+`vselect`+`vmul`）是设计层可消除的结构性成本；`vrec` 次之；`vexp` 便宜。

**(c) 采纳候选（tanh-奇多项式，`perf_opt/erf_fwd_kernel.py`）与 baseline 同会话对账**：

| workload_id | dtype | baseline(us) | final(us) | 加速 | final `aiv_vec_ratio` |
|---|---|---:|---:|---:|---:|
| elementwise-16m | float16 | 256.45 | 117.05 | 2.19x | 0.972 |
| elementwise-16m | bfloat16 | 256.72 | 119.25 | 2.15x | 0.972 |
| elementwise-16m | float32 | 308.10 | 112.08 | 2.75x | 0.962 |
| elementwise-64m | float16 | 1015.60 | 457.77 | 2.22x | 0.982 |
| elementwise-64m | bfloat16 | 1016.85 | 466.69 | 2.18x | 0.982 |
| elementwise-256m | float16 | 4058.63 | 1822.15 | 2.23x | 0.984 |
| elementwise-256m | bfloat16 | 4060.51 | 1857.83 | 2.19x | 0.985 |

**(d) 精度（设备实测，`torch.erf` fp64 对账）**：fp32 `max_abs` 5.90e-7（baseline）→ 2.98e-6（tanh k=4），门禁 1e-5 → 3.4x 余量；fp16 4.88e-4 / bf16 1.95e-3 与 baseline **完全相同**（受输出域舍入支配，1 ulp）；L0/L1/L2/Boundary 全层级通过，±inf/NaN 语义与 `torch.erf` 逐值一致。若需与设计同量级精度余量，k=6 变体（16 op，仍为 2.0x vs baseline）给出 fp32 `max_abs=6.2e-8`（>160x 余量）。

## 影响面

- **受影响算子族**：所有「无原生元算、须自建超越函数逼近链」的逐元素算子——erf 本身的后续 optimize/再设计；同族 `expm1`（已用分段 `vcmp`+`vselect` 选路，同样承担 vselect 代价）、`tanh` 类、以及任何用 `|x|`+符号掩码重构奇/偶函数的超越算子。
- **受影响设计环节**：`DESIGN.md` §1.6.0 算法调研的**候选枚举完整性与淘汰依据的量化门槛**——本次证明「同一函数族（tanh 型）在系数个数变化下精度可有 3 个数量级差异（~1e-3 → 6e-8）」，故淘汰任何函数族前须给出该族的**最小可达误差（按系数预算扫描）**，不能以单一低阶形态的误差定论。
- **改设计预期影响**：若采纳附录补记，只影响 DESIGN.md 附录（不改既有章节）；若后续再有同族算子任务，算法调研应把「tanh-奇多项式（系数预算扫描）」列为候选族并复核 op/精度权衡。不重开 Stage 1（`scenario=optimize`，`stage_plan=[4]`，无 Stage 1 可修订）。Stage 4 与集成工作量不变（本优化已落在 `perf_opt/`）。

## 反馈结论

设计 §1.6.0 对 tanh 型逼近（候选 7）的淘汰依据（「误差 ~1e-3 达不到 fp32 门禁」）仅在其采用的 2 系数形态下成立；放宽到 5–7 系数的奇多项式 `erf(x)≈tanh(x·Q(x²))` 后，设备实测在通过 fp32 门禁（3.4x–160x 余量）的同时把 op 数从 21 降到 12（fp16·bf16 从 23 降到 14），结构性加速 **2.15x–2.75x**（全部 > 2x），A&S 链中昂贵的 `vselect`/`vrec` 与全部 sign 处理被消除——设计「无更优可行替代」的结论需修订。

建议路由: 附录补记
