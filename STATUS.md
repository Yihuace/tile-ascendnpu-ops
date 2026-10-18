# 当前工作状态（每轮更新）

> 本窗口：**Stage 4 批量优化（11 个未达标算子）已完成**。
> 更新：2026-10-18 10:25 (+08:00) · 会话：Stage4 批量优化（真派 optimizer）

## 总览

**19 个交付接口**：6 个 SOL 已达标（Add/Sub/Lerp/ClampMin/ClampScalar/Hardswish，均为早期完成）；本轮把其余 **11 个未达标算子**全部跑完 Stage 4；1 个（GDN prefill）仍处 migration `INTEGRATE`，未进入优化流程。

完整汇总：`examples/TileOPs/reports/optimize/STAGE4_SUMMARY.md`
队列与逐算子结论：`examples/TileOPs/reports/optimize/STAGE4_QUEUE.md`
执行手册：`examples/TileOPs/reports/optimize/STAGE4_RUNBOOK.md`

## 本轮结果（11 个）

| 算子 | 收益 | 采纳 | 状态 |
|---|---|---:|---|
| GLADecodeOp | −21.6~28.3% | ✅ | DONE |
| HardsigmoidFwdOp | 0（wash，compute-bound 判据失配） | ✅ | DONE |
| GeluTanhAndMulFwdOp | prefill −8.2% | ✅ | DONE |
| GeluAndMulFwdOp | prefill **−43.6%** | ✅ | DONE |
| IsnanFwdOp | 0（MTE2 带宽地板） | ❌ | DONE |
| IsinfFwdOp | bf16 **−19.4~−19.9%** | ✅ | DONE |
| IsfiniteFwdOp | bf16 **−21.2%** | ✅ | DONE |
| FusedAddLayerNormFwdOp | 8b prefill **−23.2%** | ✅ | DONE |
| BatchNormBwdOp | **−33~−46%** | ✅ | DONE |
| Conv1dBiasFwdOp / Conv1dFwdOp | whisper **15×**、encodec-deep 3.6× | ✅ | DONE（形状路由绕开 9.0 缺陷） |
| Conv3dBiasFwdOp | r3d-stem **1.99×** | ✅ | DONE |

## 必须记住的三件事

1. **CANN 9.0 编译器缺陷（CG-2026-0003, recurring, occurrences=4）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 调 `getResult(0)` 越界；8.5.2 无此问题。已用两处形状路由在官方 9.0 上绕开：`dilation≠1` 走通用 kernel、unit-stride 仅 `c_in>=64` 用调优版；残留 1 个 bench 形状（wav2vec2-layer1）仍崩且与 Stage-3 逐字节相同 ⇒ pre-existing。
2. **GM SOL 对 compute-bound 算子判据失配**：Hardsigmoid / GeluTanh / GeluAndMul 实测 `aiv_vec_ratio ≥0.85`，达标判定改按向量利用率 + Task Duration。
3. **predicate 族（Isnan/Isinf/Isfinite）的 fp16/fp32 已达 MTE2 带宽地板**，bf16 通过 `vand` 位运算从 6-pass 降到 4-pass（−19~−21%）后也到地板。

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）
- **所有 agent/subagent 一律 `deepseek/deepseek-flash`**（`AGENTS.md` Model Policy，禁止 pro）
- 子 agent 返回前必须自查 `statectl gate 4` 直到 pass

## 后续可选

- GDN prefill（19 个接口里唯一未进入优化）若需要，先完成其 migration `INTEGRATE` 再进 Stage 4
- Conv1d/Conv1dBias 等 CANN 9.0 修复后，直接复用 `perf_opt/` 成果重跑端到端验收
- Tier 2 进化提案（`VP-2026-0147`~`0162`）待人工审批
