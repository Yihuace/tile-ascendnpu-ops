# Stage 4 进度报告 — ErfFwdOp / `erf_fwd_kernel`

- 任务: `erf-erf_fwd_kernel-20261013T131205Z`（scenario=`optimize`，mode=`full`，best_effort）
- 状态: **TUNING_COMPLETED**（gate 4 通过；`stop_reason=success`）
- 迭代: `perf_iteration.count=3`，`last_improvement=+54.4%`（16m-f16，Round 1），`consecutive_no_improvement=2`（Round 2–3 无 >3% 提升）

## 结果一览（msprof op Task Duration(us)，median of 20）

| workload | dtype | baseline | final | 提升 | 加速 |
|---|---|---:|---:|---:|---:|
| elementwise-16m | float16 | 256.45 | 117.05 | -54.4% | 2.19x |
| elementwise-16m | bfloat16 | 256.72 | 119.25 | -53.5% | 2.15x |
| elementwise-16m | float32 | 308.10 | 112.08 | -63.6% | 2.75x |
| elementwise-64m | float16 | 1015.60 | 457.77 | -54.9% | 2.22x |
| elementwise-64m | bfloat16 | 1016.85 | 466.69 | -54.1% | 2.18x |
| elementwise-256m | float16 | 4058.63 | 1822.15 | -55.1% | 2.23x |
| elementwise-256m | bfloat16 | 4060.51 | 1857.83 | -54.2% | 2.19x |

- 全部 7 个 tune workload 均 `winner_merged`；无 `merge_blocked`；smoke 3 项精度回归通过。
- 有效优化点（单点验证）：**奇 tanh-多项式逼近**（`erf(x)≈tanh(x·Q(x²))`，Q 4 次）+ **per-dtype block_size**（f32 9216 / f16 10240 / bf16 10240）。
- 未采纳：静态整块+动态尾块 T.copy（UB overflow）、连续核内分片（tie）。

## 瓶颈轨迹

| 阶段 | AIV vec 占比 | MTE2 占比 | 判读 |
|---|---:|---:|---|
| baseline | 62.6–80.1% | 21.7–24.8% | A&S 21/23-op 链 Vector 发射受限（`vselect` 单 op ≈6.5x 普通 op） |
| final | 96.2–98.5% | 14.6–26.9% | Vector 饱和；op 数 12/14，接近该算法族结构下界 |

## 精度

- fp32: baseline 5.90e-7 → final 2.98e-6（门禁 1e-5，3.4x 余量）；fp16 4.88e-4 / bf16 1.95e-3 与 baseline 逐值相同（输出域舍入支配）。
- L0/L1/L2/Boundary 全层级通过；`inf/-inf/nan` 与 `torch.erf` 逐值一致。
- 遗留取舍：如需 ≥10x fp32 余量，可切 `k=6`（7 系数）变体（16 op，设备 `max_abs` 1.79e-7，仍为 baseline 的 ~2.0x）。

## 信号与工件

- `design_limit_signal`: **`[DESIGN_LIMIT]`** → `perf_opt/perf_feedback.md`（设计 §1.6.0 对 tanh 型候选 7 的淘汰依据仅在其 2 系数形态下成立；实测结构性加速 2.15x–2.75x > 2x；建议路由 = **附录补记**）
- `perf_opt/erf_fwd_kernel.py`（final，导出 `TUNED_BLOCK_SIZE` 供 wrapper 切换块成对引用）
- `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`workload_inventory.json`、`perf_feedback.md`、`profiles/`、`logs/`
- pattern-library 回写：`elementwise.md` PL-1.29 / PL-1.30、`traps-runtime.md` TRAP-vtanh-large-arg-nan、`constants.md` CONST-vector-op-cost-910_9362（repro ×3 实跑 PASS；`kb_lint` 0 failures）

## 未决风险

1. `[DESIGN_LIMIT]` 属非阻塞信号，路由（附录补记/不处理）由 conductor 按标准执行；本 Agent 不改 DESIGN.md。
2. fp32 精度余量由 16.9x 降至 3.4x（确定性误差上界），已在 opt_log / perf_feedback 记录与备选。
3. `perf_records.jsonl` 含一行 `round2_static`（`--skip-l0`）非权威记录，已在 opt_log append-only 更正说明中留痕，未参与 winner 判定。
