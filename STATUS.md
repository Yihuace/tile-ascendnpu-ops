# 当前工作状态（每轮更新）

> 本窗口只做 **Stage 4 优化**，执行 `optimize` 工作流。
> 更新：2026-10-17 21:10 (+08:00) · 会话：Stage4 优化：Add → Sub → Isnan → Isinf/Isfinite

## 已完成

| 算子 | 结果 | 门禁/提交 |
|---|---|---|
| **AddFwdOp** | **11/11 ≥70%**（linear-bias fp16 72.30 / bf16 70.80）；整体 −1.16%；A6 fp16 原生 | gate pass，`68490cd` |
| **SubFwdOp** | **14/15 ≥70%**（linear-bias fp16 73.97、llama-hidden fp16 73.29）；唯一未过 bf16 经 A/B 确认为漂移 | gate pass，`38021c0` |
| **IsnanFwdOp** | **1/7 ≥70%**（16m fp32 78.45）；其余 55–65%；收益 0–6.2%；**pipeline-latency 上限（CG-2026-0027）** | gate pass |

**共同结论**
- 有效结构：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核；
- fp16 `alpha==1` 原生 add/sub 是位等价快路径（单次舍入），Add/Sub 的小 shape 由此过线；
- bf16 原生可编译且 max_diff=0，但时延方向不一致，不采纳（保持 fp32 中转）；
- 小 shape 的判定受 ±5% 进程双态支配，<5% 差异一律走交错 A/B，不做单次快照结论。

## 优化范围修订（2026-10-17 21:30，用户口径）

**已有优化报告且不含 Cube 的算子不再重测**；只有含 Cube 的算子需要按 CANN 9.0 重测。

| 类别 | 算子 | 处理 |
|---|---|---|
| 已有报告 + 纯 Vector | ClampScalar、GeluAndMul、GeluTanhAndMul、Hardswish、Hardsigmoid | **跳过重测**（沿用 8.5 报告） |
| 无报告 + 纯 Vector | FusedAddLayerNorm、GLADecode | 仍需 Stage 4 |
| 已有报告 + 含 Cube | BatchNormBwd、Conv1d/Bias、Conv3dBias | 按 9.0 重测（Cube lowering 变更） |
| predicate 族 | Isfinite（进行中） | 完成收尾 |

## 下一步：Isfinite → FusedAddLayerNorm → GLA → BatchNormBwd → Conv

- 目录：`elementwise/isinf/`、`elementwise/isfinite/`
- 复用 Isnan 的 3-pass predicate 结构与 `perf_opt/` 骨架；判据仍按 AI 分类（访存 70%）
- 预期同样受 CG-2026-0027 的 3-pass 下限约束，报告按实际 ceiling 如实写

## 已挂起

- **Conv1dFwdOp**：R1-A 否决，R1-B 分区重排数值错；`fb94e72`。恢复入口 `.../conv1d/conv1d_kernel/perf_opt/opt_log.md` §4.3/§4.4。

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`。
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）。
- 序：Add ✅ → Sub ✅ → Isnan ✅ → **Isinf/Isfinite** → ClampScalar → Gelu/Hardswish → Norm → GLA → Conv。
