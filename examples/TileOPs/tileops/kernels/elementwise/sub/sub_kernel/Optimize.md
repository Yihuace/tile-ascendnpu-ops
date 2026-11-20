# SubFwdOp Stage-4 调优摘要（Optimize.md）

> 本页只摘要 `perf_opt/opt_log.md`；完整过程与证据以该文件为准。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root · Ascend910B3（40 AIV）· 2026-10-17。

## 结论

- **15 个 tune case 中 14 个 ≥70% SOL**；两个此前未过的 case 现在稳过：
  `linear-bias fp16` **73.97%**、`llama-7b-hidden fp16` **73.29%**。
- 采纳：**满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核
  + fp16(alpha==1) 原生 vsub**（`perf_opt/SubFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`）。
- 全部 15 case 时延和 **506.151 → 504.791 µs（−0.27%）**。
- 唯一未过 `linear-bias bf16` 66.76%：bf16 代码路径与 baseline 完全相同，5 对 A/B 显示
  tuned **快 1.64%**（verdict=tie）⇒ 官方值是跨轮漂移样本，不是回退。

## 最终对账（官方 msprof）

| workload | baseline µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
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

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0）。

## 有效优化点

| 优化点 | 实测 |
|---|---|
| 满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` | 大 shape −2~−3%（phi-2、llama-bias） |
| fp16(alpha==1) 原生 vsub | `linear-bias fp16 −6.76%`、`llama-7b-hidden fp16 −7.08%`，`--level all` max_diff=0 |

## 否决项

| 候选 | 原因 |
|---|---|
| bf16 原生 vsub | 可编译且 max_diff=0，但性能方向不一致（Add 上 −2%~+3.8%），不采纳 |
| 首轮 baseline 慢态结论 | 曾伪造 llama-hidden −13.5%，相邻交错复测仅 −0.6% |

## 精度

- `python perf_opt/SubFwdOp.py --level all`：**L0/L1/L2/Boundary 全 PASS**
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192`：smoke + linear-bias PASS，max_diff=0

## 遗留

1. `linear-bias bf16` 的 70% 判定仍在噪声线附近；bf16 原生收益不可靠，已按“噪声内”处理。
2. Stage 5 wrapper 切换：`from .sub_kernel.perf_opt.SubFwdOp import sub_fwd_kernel`，
   默认 block 用 `SubFwdOp.TUNED_BLOCK_SIZE`。

## 证据

`perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`（含 `ab_sub_lb_bf16.json`）、
`reports/optimize/SubFwdOp/OPTIMIZE_REPORT.md`
