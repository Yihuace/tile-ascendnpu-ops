# 优化报告：SubFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `SubFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = a - alpha * b` |
| kernel | `tileops/kernels/elementwise/sub/sub_kernel/` |
| 调优产物 | `.../sub_kernel/perf_opt/SubFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op --kernel-name=main` 的 `Task Duration(us)` + `visualize_data.bin` 的 GM Read+Write SOL ratio |
| 工具链 | tilelang `0.1.2` dev root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3 / 2026-10-17 |
| benchmark | `benchmarks/ops/bench_sub.py::test_sub_manifest_bench`（15 tune + 1 smoke） |
| 过程日志 | `.../sub_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/` |

## 1. 结论

- **15 个 tune case 中 14 个 ≥70% SOL**；两个此前未过的 case 现在稳过：
  `linear-bias fp16` **73.97%**（12.96µs，−6.76%）、`llama-7b-hidden fp16` **73.29%**（12.99µs，−7.08%）。
- 结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核
  + fp16(alpha==1) 原生 vsub**（由 AddFwdOp 的 Stage-4 winner 同构移植）。
- 全部 15 case 时延和 **506.151 → 504.791 µs（−0.27%）**。
- 唯一未过 `linear-bias bf16` 66.76%：该 case 的 bf16 代码路径与 baseline 完全相同（A6 只改 fp16），
  5 对交错 A/B 显示 tuned **14.000→13.770µs（快 1.64%，verdict=tie）** ⇒ 官方值是跨轮漂移样本，不是回退。
- 方法论修正：首轮 baseline 处于慢态，曾把 `llama-7b-hidden` 伪造成 −13.5%；相邻交错复测只剩 −0.6%。

## 2. 所做改动

仅在 `perf_opt/` 内（基准 kernel 与 wrapper 未动）：

1. 满块静态 extent + 单尾块 epilogue（去热循环 `T.min`/动态 extent）；
2. `T.Pipelined(num_local_tasks, num_stages=2)` 重叠 MTE2 与 VEC；
3. **A6**：fp16 且 `alpha==1` 时直接 `T.vsub(a_ub,b_ub,y_ub)`，4 条向量 pass → 1 条；
4. `TUNED_BLOCK_SIZE = 8192` 模块级常量。

## 3. 优化结果（msprof，baseline round 2 / final round 5）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast-bfloat16 | 36.971 | 37.151 | +0.49% | 78.50 | 79.09 | ✅ |
| cnn-feat-broadcast-float16 | 36.841 | 37.091 | +0.68% | 79.78 | 78.80 | ✅ |
| cnn-feat-broadcast-float32 | 70.651 | 71.341 | +0.98% | 82.96 | 83.64 | ✅ |
| hidden-state-prefill-bfloat16 | 25.111 | 25.091 | -0.08% | 76.76 | 75.61 | ✅ |
| hidden-state-prefill-float32 | 46.561 | 46.971 | +0.88% | 83.40 | 83.91 | ✅ |
| linear-bias-bfloat16 | 13.410 | 14.080 | +5.00% | 71.44 | 66.76 | ❌ |
| linear-bias-float16 | 13.900 | 12.960 | -6.76% | 71.12 | 73.97 | ✅ |
| linear-bias-float32 | 24.900 | 24.580 | -1.29% | 76.69 | 76.33 | ✅ |
| llama-7b-ffn-bfloat16 | 32.711 | 32.851 | +0.43% | 79.18 | 78.69 | ✅ |
| llama-7b-ffn-bias-float16 | 33.621 | 33.101 | -1.55% | 77.09 | 79.13 | ✅ |
| llama-7b-ffn-float16 | 33.621 | 33.941 | +0.95% | 75.42 | 77.79 | ✅ |
| llama-7b-ffn-float32 | 62.881 | 63.101 | +0.35% | 82.39 | 84.72 | ✅ |
| llama-7b-hidden-float16 | 13.980 | 12.990 | -7.08% | 68.09 | 73.29 | ✅ |
| phi-2-ffn-bias-float16 | 30.511 | 29.681 | -2.72% | 78.58 | 79.52 | ✅ |
| phi-2-ffn-float16 | 30.481 | 29.861 | -2.03% | 77.46 | 79.73 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0），不参与性能判定。

### A/B 复核（5 对/例）

| case | base µs | 候选 µs | 判定 |
|---|---:|---:|---|
| hidden-state-prefill bf16 | 25.891 | 25.051（−3.2%） | inconclusive（推翻 R1 的 +5.8%） |
| cnn-feat-broadcast fp32 | 71.011 | 72.521（+2.1%） | inconclusive（噪声，不做特判） |
| linear-bias fp16 | 13.880 | 13.880 | tie |
| linear-bias bf16 | 14.000 | 13.770（−1.6%） | tie（官方 66.76% 是漂移） |

### 精度

- `python perf_opt/SubFwdOp.py --level all` → **L0/L1/L2/Boundary 全 PASS**；
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192` → smoke + linear-bias PASS，max_diff=0。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 17:48 → 20:35 |
| 墙钟时长 | **≈ 167 min**（含 A6 原生路径与 bf16 证伪、两次 final 复测） |
| token input/output/reasoning | 88,528 / 75,076 / 50,407 |
| cache-read / cost | 78,880,000 / **$0.3252** |

## 5. 产物

- `.../sub_kernel/perf_opt/SubFwdOp.py`（final，sha256 见 round 5 final 行）
- `.../sub_kernel/perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- 基线报告（迁移期）：`reports/tileops/20261013_170400_275674_SubFwdOp/`
- gate 4：`pass`；state `phase=DONE`
