# 当前工作状态（每轮更新）

> 本窗口只做 **Stage 4 优化**，执行 `optimize` 工作流。
> 更新：2026-10-17 18:56 (+08:00) · 会话：Stage4 优化：Add → Sub → Isnan/Isinf/Isfinite

## 已完成

### AddFwdOp（`phase=DONE`，commit `5a4367e`）

| 项 | 内容 |
|---|---|
| 产物 | `.../elementwise/add/add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE=8192`） |
| 报告 | `perf_opt/opt_log.md` + `add_kernel/Optimize.md` + `reports/optimize/AddFwdOp/OPTIMIZE_REPORT.md` |
| 结果 | 11 tune 中 9 个 ≥70%；hidden bf16 −3.85%、llama fp32 −1.86%；`linear-bias fp16/bf16` 66.9/68.7%（噪声内，未稳健达标） |
| 门禁 | gate 4 pass |

### SubFwdOp（`phase=DONE`）

| 项 | 内容 |
|---|---|
| 产物 | `.../elementwise/sub/sub_kernel/perf_opt/SubFwdOp.py`（`TUNED_BLOCK_SIZE=8192`，Add winner 同构移植） |
| 报告 | `perf_opt/opt_log.md` + `sub_kernel/Optimize.md` + `reports/optimize/SubFwdOp/OPTIMIZE_REPORT.md` |
| 结果 | 15 tune 中 **13 个 ≥70%**；整体 −0.51%；llama fp16 −3.06%、llama-bias fp16 −2.77%、linear fp32 −2.33%；`linear-bias fp16` 69.27%、`llama-hidden fp16` 68.38%（噪声内） |
| 门禁 | gate 4 pass |

**关键方法论结论**：Sub 首轮 baseline 处于慢态，曾把 `llama-hidden` 收益伪造成 −13.5%；相邻交错复测只剩 −0.6%。**跨轮 baseline 比较不可作为收益证据**，已回写 pattern-library PL-1.22 边界。

## 下一步：Isnan / Isinf / Isfinite（elementwise 1c）

- 目录：`examples/TileOPs/tileops/kernels/elementwise/{isnan,isinf,isfinite}/`
- 现状：已有 Stage 3 产物 + 迁移报告；无 `perf_opt/`
- 入口：`statectl init --scenario optimize` → 同 Add/Sub 流程（baseline 与候选**同批相邻交错**）
- 注意：`MIGRATION_INDEX` 显示这三个算子最大 shape SOL 仅 40–75%，属**算力/访存混合受限**，
  判据按 AI 分类（≥70% 仅适用访存受限；算力受限线为 60%），不能一律套 70%。

## 已挂起

- **Conv1dFwdOp**：R1-A 否决（whisper +0.2%），R1-B 分区重排数值错；已收尾并提交（`fb94e72`）。
  恢复入口：`.../conv1d/conv1d_kernel/perf_opt/opt_log.md` §4.3/§4.4。

## 环境纪律

- 所有上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 必须 `source perf_opt/env.sh`。
- **长测试运行期间禁止编辑被测源文件**（TileLang JIT 的 `inspect.getsource` 行号映射会失效，伪造编译失败）。
- 序：Add ✅ → Sub ✅ → Isnan/Isinf/Isfinite → ClampScalar → Gelu/Hardswish → Norm → GLA → Conv。
