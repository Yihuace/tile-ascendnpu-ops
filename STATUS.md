# 当前工作状态（每轮更新）

> 本窗口：**19 算子统一 CANN 9.0.0 口径**。四个 8.5 残留（Add / Sub / Isnan / ClampMin）已完成端到端重测；Add / Sub 同时切到 `perf_opt`。
> 更新：2026-10-18 16:00 (+08:00) · 会话：9.0 口径统一

## 总览（严格口径，全部 CANN 9.0.0）

**19 个交付接口，18 个已交付；1 个（GDN）未交付。**

| 判定 | 数量 | 算子 |
|---|---|---|
| **全 shape 严格达标** | **1** | Sub（15/15） |
| 贴线 / 部分达标 | 4 | Add（10/11；`linear-bias bf16` 复测 63.7–69.4%，确认贴线未过）、Lerp（9/11）、Hardswish（2/6）、ClampScalar（1/11） |
| 未达标但**已采纳真实时延收益** | 12 | Hardsigmoid、ClampMin、Isinf、Isfinite、GeluTanh、GeluAndMul、FALN、BatchNormBwd、Conv1d、Conv1dBias、Conv3dBias、GLA |
| 未达标且未采纳 | 1 | Isnan（无收益，MTE2 带宽地板） |
| 未交付 | 1 | GDN（2/6 kernel，CG-2026-0003） |

> 全 shape 达标从上一版“3 个”修正为“1 个”：Add 在 9.0 端到端下 10/11（`linear-bias bf16` 69.37%，复测 63.7–69.4%，确认未过线），ClampScalar 在 9.0 下仅 1/11（8.5 的 71.9–82.8% 不再引用）。
> 所有数字来自 `reports/tileops/<run_id>/run.json`；8.5.2 数据仅保留在 `MIGRATION_INDEX.md` 表 B 历史存档（不在本批范围）。

## 本轮 9.0 统一重测（4 个 8.5 残留）

| 算子 | 生效内核 | 正确性 | ratio（9.0 端到端） | 达标 | run_id |
|---|---|---|---:|---:|---|
| AddFwdOp | `perf_opt`（本轮切换） | 24/24 | 69.37–83.48% | 10/11 | `20261018_143726_708597` |
| SubFwdOp | `perf_opt`（本轮切换） | 24/24 | 70.70–84.00% | 15/15 | `20261018_145645_852546` |
| IsnanFwdOp | baseline（不采纳） | 9/9 | 52.91–79.33% | 1/5 | `20261018_152024_774098` |
| ClampMinFwdOp | 复用 ClampScalar `perf_opt` | 14/14 | 61.77–68.84% | 0/7 | `20261018_153708_107121` |

## 上一轮 11 个算子的结果（9.0.0 口径，仍有效）

| 算子 | 关键时延收益 | 采纳 | 状态 |
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

## 必须记住的五件事

1. **CANN 9.0 编译器缺陷（CG-2026-0003, recurring, occurrences=4）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取 `getResult(0)` 越界；**9.0.0 与 9.2.0 官方包均有**，8.5.2 无（405/405）。
   - Conv1d/Conv1dBias 已用**形状路由**绕开（`dilation==1` 才走 unit-stride；unit-stride 仅 `c_in>=64` 用调优版）
   - GDN blocksolve 仍阻塞（触发点在算法本体的 L0C→GM→L1 回喂）
2. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降（GeluAndMul −43.6% 时延，ratio 43.2%→46.0%）。这类算子看 Task Duration，不看 ratio。
3. **BatchNormBwd 的 ratio 因 subid 去双计而下降**（15.3–28.2% → 13.0–25.5%），不可比，时延 −40%。
4. **口径策略：全部统一 CANN 9.0.0**（2026-10-18 用户口径）。8.5.2 数据仅在表 B 历史存档中保留，不再用于本批 19 接口的结论；跨版本数字不横比。
5. **Clamp 族 9.0 回退已定性：工具链/环境整体下移，不是调优失效**。同链路 baseline vs tuned（`20261018_160303` vs `20261018_131331`）：tuned 仍快 **7.6–40%**（16M 30.7–32.9µs vs 38.3–43.1µs；256M 803–810µs vs 864–874µs）。9.0 baseline 自身只有 **45.35–64.37%**，所以 8.5 的 71.9–82.8% 不能跨版本引用；**保留 perf_opt**，不再重优化。

## 报告规范

- 模板：`examples/TileOPs/reports/optimize/REPORT_TEMPLATE.md`
- 14 份单算子报告已按 8 节统一（0 身份 / 1 结论 / 2 优化方法与过程〔2.1 诊断 + 2.2 方法 + 2.3 逐轮〕/ 3 结果 / 4 精度 / 5 时间成本 / 6 产物 / 7 天花板豁免缺口 / 8 建议）
- 旧版报告存各目录 `history_version/OPTIMIZE_REPORT_v0.md`
- 缺失报告：Conv1dFwdOp、GLADecodeOp（待补），ClampMin/Lerp 无独立报告（分别由 ClampScalar / 未优化覆盖）

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
5. Tier 2 进化提案（`VP-2026-0147`~`0162`）待人工审批
