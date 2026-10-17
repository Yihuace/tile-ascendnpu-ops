# 当前工作状态（每轮更新）

> 本窗口只做 **Stage 4 优化**，执行 `optimize` 工作流。
> 更新：2026-10-17 17:48 (+08:00) · 会话：Stage4 优化：Add → Sub

## 已完成：AddFwdOp（`phase=DONE`）

| 项 | 内容 |
|---|---|
| 基线 kernel | `tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py` |
| 调优产物 | `.../add_kernel/perf_opt/AddFwdOp.py`（`TUNED_BLOCK_SIZE=8192`） |
| 报告 | `.../add_kernel/perf_opt/opt_log.md` + `.../add_kernel/Optimize.md` |
| 达标线 | 访存受限 ≥ 70%（微缩 2–4µs 豁免） |
| 门禁 | `statectl gate 4` → pass（0 failures / 0 warnings） |

**结论（如实）**
- 11 个 tune case：**9 个 ≥70%**；明确收益 `hidden-state-prefill bf16 −3.85%`、`llama-7b-ffn fp32 −1.86%`；
- `linear-bias fp16/bf16` 官方单次 final **66.85% / 68.67%（未达标）**，但差异 <±5% 噪声、
  池化中位数反而快 2.4–2.9%（≈70.0/70.5%）→ 判定为「**噪声内，70% 未被稳健达成**」，不宣称达标；
- 采纳结构：满块静态 extent + 尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核；
- 否决：block 12288/16384（负载不均 +5~7%）、UB 输出别名（trace 作用域编译失败）、
  等元素连续分块（−3% vs grid-stride）。

## 下一步：SubFwdOp

- 目录：`examples/TileOPs/tileops/kernels/elementwise/sub/sub_kernel/`
- 现状：只有 Stage 3 产物（`sub_fwd_kernel.py` + DESIGN），**无 `perf_opt/`、无 `.stage_state.json`**
- 入口：`statectl init --scenario optimize` → Stage 4（同 Add 的采数/门禁/报告流程）

## 已挂起

- **Conv1dFwdOp**：R1-A 否决（whisper +0.2%），R1-B 分区重排数值错；已收尾并提交（`fb94e72`）。
  恢复入口：`.../conv1d/conv1d_kernel/perf_opt/opt_log.md` §4.3/§4.4。

## 环境纪律

- 所有上机命令经 `.agents/tools/npu_lock.sh`（互斥 + 空闲等待 + 超时回收）。
- 非交互 shell 必须 `source perf_opt/env.sh`（tilelang dev root + CANN 9.0.0）。
- **长测试运行期间禁止编辑被测源文件**（TileLang JIT 的 `inspect.getsource` 行号映射会失效，伪造编译失败）。
- 原计划：Add ✅ → Sub → Isnan/Isinf/Isfinite → ClampScalar → Gelu/Hardswish → Norm → GLA → Conv。
