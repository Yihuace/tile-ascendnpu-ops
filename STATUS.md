# 当前工作状态（每轮更新）

> 本窗口：**Stage 4 批量优化（11 个未达标算子）已完成**，报告结构已统一。
> 更新：2026-10-18 13:40 (+08:00) · 会话：Stage4 批量优化（主会话编排 + 真派 optimizer）

## 总览（严格口径）

**19 个交付接口**，其中 18 个有优化/性能数据：

| 判定 | 数量 | 算子 |
|---|---|---|
| **全 shape 严格达标** | **3** | Add（11/11）、Sub（14/15）、ClampScalar（6/6，8.5 口径） |
| 单点达标、未按全 shape 验证 | 2 | Lerp、ClampMin（8.5 口径，仅最大 shape） |
| 边界 | 1 | Hardswish（2/6，compute-bound） |
| 未达标但**已采纳真实时延收益** | 11 | Hardsigmoid、Isinf、Isfinite、GeluTanh、GeluAndMul、FALN、BatchNormBwd、GLA、Conv1d、Conv1dBias、Conv3dBias |
| 未达标且未采纳 | 1 | Isnan（无收益，MTE2 带宽地板） |
| 未进入优化流程 | 1 | GatedDeltaNetPrefillFwdOp（migration 未完成，2/6 函数） |

> 早前曾把「最大 shape 达标」当作「算子达标」写成 6 个，**严格按全 shape 只有 3 个**。
> 详见 `examples/TileOPs/reports/optimize/STAGE4_SUMMARY.md` §2。

## 本轮 11 个算子的结果（9.0.0 口径）

| 算子 | 收益 | 采纳 | 状态 |
|---|---|---:|---|
| GLADecodeOp | −21.6~28.3% | ✅ | DONE（latency-bound） |
| HardsigmoidFwdOp | 0（wash，判据失配） | ✅ | DONE |
| GeluTanhAndMulFwdOp | prefill −8.2% | ✅ | DONE |
| GeluAndMulFwdOp | prefill **−43.6%** | ✅ | DONE |
| IsnanFwdOp | 0（MTE2 地板） | ❌ | DONE |
| IsinfFwdOp | bf16 **−19.4~−19.9%** | ✅ | DONE |
| IsfiniteFwdOp | bf16 **−21.2%** | ✅ | DONE |
| FusedAddLayerNormFwdOp | 8b prefill **−23.2%** | ✅ | DONE |
| BatchNormBwdOp | **−33~−46%** | ✅ | DONE |
| Conv1dBiasFwdOp / Conv1dFwdOp | whisper **15×**、encodec-deep 3.6× | ✅ | DONE（形状路由解封） |
| Conv3dBiasFwdOp | r3d-stem **1.99×** | ✅ | DONE |

## 必须记住的四件事

1. **CANN 9.0 编译器缺陷（CG-2026-0003, recurring, occurrences=4）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取 `getResult(0)` 越界；**9.0.0 与 9.2.0 官方包均有**，8.5.2 无（405/405）。
   - Conv1d/Conv1dBias 已用**形状路由**绕开（`dilation==1` 才走 unit-stride；unit-stride 仅 `c_in>=64` 用调优版）
   - GDN blocksolve 仍阻塞（触发点在算法本体的 L0C→GM→L1 回喂）
2. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降（GeluAndMul −43.6% 时延，ratio 43.2%→46.0%）。这类算子看 Task Duration，不看 ratio。
3. **BatchNormBwd 的 ratio 因 subid 去双计而下降**（15.3–28.2% → 13.0–25.5%），不可比，时延 −40%。
4. **口径策略：能衔接 8.5 就衔接**。ClampScalar / Hardswish / Lerp / ClampMin 直接沿用 8.5.2 报告，不重测；仅 ClampScalar 抽测差异大（9.0 下 58.2–70.2%、时延 +4~33%）在汇总表加备注，不改结论、不重优化。

## 报告规范

- 模板：`examples/TileOPs/reports/optimize/REPORT_TEMPLATE.md`
- 14 份单算子报告已按 8 节统一（0 身份 / 1 结论 / 2 优化方法与过程〔2.1 诊断 + 2.2 方法 + 2.3 逐轮〕/ 3 结果 / 4 精度 / 5 时间成本 / 6 产物 / 7 天花板豁免缺口 / 8 建议）
- 旧版报告存各目录 `history_version/OPTIMIZE_REPORT_v0.md`

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）
- **所有 agent/subagent 一律 `deepseek/deepseek-flash`**（`AGENTS.md` Model Policy，禁止 pro）
- 新派发的 optimizer 必须在返回前自查 `statectl gate 4` 直到 pass

## 后续可选（按收益排序）

1. **Conv1d 通用 kernel**（wav2vec2 0.33% / encodec-init 0.41%）：仍是逐核冗余搬运，可能 2–5×
2. **Conv3d unet 路径**：权重 L1 常驻在 `c_in_pad=32` 未生效，可能 2×
3. **FALN 设计层**：两次 row-reduce 改 Cube GEMV 卸载，20–40%
4. **GDN prefill**：先做判定性探针（换算法形态能否编译），再决定是否投入 1–2 天
6. Tier 2 进化提案（`VP-2026-0147`~`0162`）待人工审批
