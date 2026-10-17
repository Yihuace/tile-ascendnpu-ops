# SubFwdOp Stage-4 调优摘要（Optimize.md）

> 本页只摘要 `perf_opt/opt_log.md`；完整过程与证据以该文件为准。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root · Ascend910B3（40 AIV）· 2026-10-17。

## 结论

- **15 个 tune case 中 13 个 ≥70% SOL**；未达标 2 个均在噪声内：
  `linear-bias fp16 69.27%`（Δ +0.58%）、`llama-7b-hidden fp16 68.38%`（Δ −0.57%）。
- 采纳结构：**满块静态 extent + 单尾块 epilogue + `T.Pipelined(num_stages=2)` + grid-stride 40 核**
  （`perf_opt/SubFwdOp.py`，`TUNED_BLOCK_SIZE = 8192`），由 AddFwdOp 的 winner 同构移植。
- 整体 15 case 时延和 **506.151 → 503.591 µs（−0.51%）**，中位 −0.40%；无 >5% 的可信回退。
- 稳定收益：`llama-7b-ffn fp16 −3.06%`、`llama-7b-ffn-bias fp16 −2.77%`、`linear-bias fp32 −2.33%`。
- **方法论教训**：首轮 baseline 处于慢态，曾把 llama-7b-hidden 的收益伪造成 −13.5%；
  相邻交错复测只剩 −0.6%。跨轮 baseline 比较不可作为收益证据。

## 最终对账（官方单次 msprof，15 个 tune）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| cnn-feat-broadcast bf16 | 36.971 | 36.411 | −1.51% | 78.50 | 80.26 | ✅ |
| cnn-feat-broadcast fp16 | 36.841 | 36.381 | −1.25% | 79.78 | 79.69 | ✅ |
| cnn-feat-broadcast fp32 | 70.651 | 71.101 | +0.64% | 82.96 | 83.53 | ✅ |
| hidden-state-prefill bf16 | 25.111 | 25.011 | −0.40% | 76.76 | 78.26 | ✅ |
| hidden-state-prefill fp32 | 46.561 | 46.901 | +0.73% | 83.40 | 81.62 | ✅ |
| linear-bias bf16 | 13.410 | 13.510 | +0.75% | 71.44 | 70.70 | ✅ |
| **linear-bias fp16** | 13.900 | 13.980 | +0.58% | 71.12 | **69.27** | ❌ |
| linear-bias fp32 | 24.900 | 24.320 | −2.33% | 76.69 | 80.36 | ✅ |
| llama-7b-ffn bf16 | 32.711 | 32.991 | +0.86% | 79.18 | 77.14 | ✅ |
| llama-7b-ffn-bias fp16 | 33.621 | 32.691 | −2.77% | 77.09 | 79.92 | ✅ |
| llama-7b-ffn fp16 | 33.621 | 32.591 | −3.06% | 75.42 | 79.67 | ✅ |
| llama-7b-ffn fp32 | 62.881 | 62.401 | −0.76% | 82.39 | 83.60 | ✅ |
| **llama-7b-hidden fp16** | 13.980 | 13.900 | −0.57% | 68.09 | **68.38** | ❌ |
| phi-2-ffn-bias fp16 | 30.511 | 30.881 | +1.21% | 78.58 | 76.13 | ✅ |
| phi-2-ffn fp16 | 30.481 | 30.521 | +0.13% | 77.46 | 78.53 | ✅ |

smoke：`hidden-state-prefill fp16`（N=8,388,608）精度 PASS（max_diff=0）。

## 精度与验收

- `python perf_opt/SubFwdOp.py --level all` → **L0/L1/L2/Boundary 全 PASS**（`All check passed!`）
- `python perf_opt/validate_final.py SubFwdOp.py --block-size 8192` → smoke + linear-bias 全 PASS，max_diff=0

## 遗留与建议

1. `linear-bias fp16` / `llama-7b-hidden fp16` 的 70% 判定在噪声内（差 0.7~1.6 点），
   需要 ≥7 对交错 A/B 或更多独立 run 才能定论；当前不做 shape 特判（证据不足）。
2. 同族 elementwise（Isnan/Isinf/Isfinite/ClampScalar/Gelu/Hardswish）可直接复用该结构作为起点，
   但必须按本任务的方法（同批相邻 baseline）复测，禁止跨轮比较。
3. Stage 5 切换 wrapper 时：`from .sub_kernel.perf_opt.SubFwdOp import sub_fwd_kernel`，
   默认 block_size 用 `SubFwdOp.TUNED_BLOCK_SIZE`（8192）。

## 证据

- `perf_opt/opt_log.md`、`perf_opt/perf_records.jsonl`、`perf_opt/logs/`（含 3 组 A/B JSON）
- `reports/optimize/SubFwdOp/OPTIMIZE_REPORT.md`
