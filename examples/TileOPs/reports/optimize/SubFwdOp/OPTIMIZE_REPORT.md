# 优化报告：SubFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `SubFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a - alpha * b` |
| kernel | `tileops/kernels/elementwise/sub/sub_kernel/` |
| 调优产物 | `.../sub_kernel/perf_opt/SubFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| 工具链 | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17 |
| benchmark | `benchmarks/ops/bench_sub.py::test_sub_manifest_bench`（15 tune + 1 smoke） |
| 过程日志 | `.../sub_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/` |

## 1. 结论

- **15 个 tune case 中 13 个 ≥70% SOL**；未达标 2 个都在噪声内：
  `linear-bias fp16` **69.27%**（Δ +0.58%）、`llama-7b-hidden fp16` **68.38%**（Δ −0.57%）。
- 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核**，
  由 `AddFwdOp` 的 Stage-4 winner 同构移植（仅 `vadd→vsub`）。
- 整体 15 case 时延和 **506.151 → 503.591 µs（−0.51%）**，中位 −0.40%；无 >5% 的可信回退。
- 稳定收益：`llama-7b-ffn fp16 −3.06%`、`llama-7b-ffn-bias fp16 −2.77%`、`linear-bias fp32 −2.33%`。
- **重要方法论修正**：首轮 baseline 处于慢态，曾把 `llama-7b-hidden` 的收益伪造成 −13.5%
  （15.68→13.56 µs）；baseline/final 相邻交错复测只剩 −0.6%（13.98→13.90 µs），
  说明**跨轮 baseline 比较不能作为收益证据**。

## 2. 所做改动

仅在 `perf_opt/` 内（基准 kernel 与 wrapper 未动）：

1. 满块静态 extent（`num_full = N // block_size`）+ 单尾块 epilogue，去掉热循环里的 `T.min`/动态 copy extent；
2. `T.Pipelined(num_local_tasks, num_stages=2)` 重叠下一 tile 的 MTE2 与当前 tile 的 VEC 链；
3. `TUNED_BLOCK_SIZE = 8192` 模块级常量，供 Stage-5 wrapper 成对引用；
4. 语义未变：fp32 中转、`alpha==1` 折叠（`vsub(a,b)`）、`alpha==0` 不折叠。

## 3. 优化结果（msprof，baseline/final 相邻交错，各 1 次独立进程）

| label | dtype | base µs | opt µs | base SOL% | opt SOL% | 判定 |
|---|---|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast | bfloat16 | 36.971 | 36.411 | 78.50 | 80.26 | ✅ |
| cnn-feat-broadcast | float16 | 36.841 | 36.381 | 79.78 | 79.69 | ✅ |
| cnn-feat-broadcast | float32 | 70.651 | 71.101 | 82.96 | 83.53 | ✅ |
| hidden-state-prefill | bfloat16 | 25.111 | 25.011 | 76.76 | 78.26 | ✅ |
| hidden-state-prefill | float32 | 46.561 | 46.901 | 83.40 | 81.62 | ✅ |
| linear-bias | bfloat16 | 13.410 | 13.510 | 71.44 | 70.70 | ✅ |
| **linear-bias** | **float16** | 13.900 | 13.980 | 71.12 | **69.27** | ❌ 噪声内 |
| linear-bias | float32 | 24.900 | 24.320 | 76.69 | 80.36 | ✅ |
| llama-7b-ffn | bfloat16 | 32.711 | 32.991 | 79.18 | 77.14 | ✅ |
| llama-7b-ffn-bias | float16 | 33.621 | 32.691 | 77.09 | 79.92 | ✅ |
| llama-7b-ffn | float16 | 33.621 | 32.591 | 75.42 | 79.67 | ✅ |
| llama-7b-ffn | float32 | 62.881 | 62.401 | 82.39 | 83.60 | ✅ |
| **llama-7b-hidden** | **float16** | 13.980 | 13.900 | 68.09 | **68.38** | ❌ 噪声内 |
| phi-2-ffn-bias | float16 | 30.511 | 30.881 | 78.58 | 76.13 | ✅ |
| phi-2-ffn | float16 | 30.481 | 30.521 | 77.46 | 78.53 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

### A/B 复核（5 对/例，`ab_test.py`）

| case | base µs | 候选 µs | 判定 |
|---|---:|---:|---|
| hidden-state-prefill bf16 | 25.891 | 25.051（−3.2%） | inconclusive（推翻 R1 的 +5.8%） |
| cnn-feat-broadcast fp32 | 71.011 | 72.521（+2.1%） | inconclusive（<5%，不做特判） |
| linear-bias fp16 | 13.880 | 13.880 | tie |

### 精度

- `python perf_opt/SubFwdOp.py --level all` → **L0 / L1 / L2 / Boundary 全 PASS**；
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192` → smoke + linear-bias PASS，max_diff=0。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 17:48 → 18:55 |
| 墙钟时长 | **≈ 67 min** |
| token input/output/reasoning | 22,993 / 16,494 / 12,862 |
| cache-read / cost | 17,749,760 / **$0.0743** |

（来源：`.start_utc`；`opencode.db` 同会话窗口求和。）

## 5. 产物

- `.../sub_kernel/perf_opt/SubFwdOp.py`（final，sha256 见 `perf_records.jsonl` final 行）
- `.../sub_kernel/perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`（含 `ab_sub_*.json`）
- 基线报告（迁移期）：`reports/tileops/20261013_170400_275674_SubFwdOp/`
- gate 4：`pass`；state `phase=DONE`
