# FusedAddLayerNormFwdOp Stage-4 调优摘要（Optimize.md）

- **0/7 ≥ 线**：prefill SOL 27–45%（算力受限线 60%），decode 微型豁免（4.1–4.7µs）。
- 采纳：**weight/bias 循环不变量外提**（每 core 一次，而非每 tile 一次）。
- 时延收益：70b prefill **−14.0~−14.5%**、8b prefill −6.4~−7.1%、decode −3.1%、13b −0.9%。
- 指标提醒：SOL ratio 同步不动（fops 同降），所以报告以 Task Duration 为主、Ratio 为辅。
- 下一轮方向：单遍方差（E[x²]−E[x]²）+ 减少 fp32 中转 + UB 瘦身后的 Pipelined。

## 最终对账

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% |
|---|---:|---:|---:|---:|---:|
| llama-13b-prefill-float32 | 107.562 | 106.622 | -0.87% | 43.80 | 45.53 |
| llama-70b-decode-bfloat16 | 4.620 | 4.660 | +0.87% | 0.81 | 0.78 |
| llama-70b-prefill-bfloat16 | 168.653 | 145.113 | -13.96% | 33.05 | 33.40 |
| llama-70b-prefill-float16 | 164.883 | 141.033 | -14.46% | 29.45 | 29.19 |
| llama-8b-decode-bfloat16 | 4.250 | 4.120 | -3.06% | 0.52 | 0.56 |
| llama-8b-prefill-bfloat16 | 84.642 | 79.252 | -6.37% | 30.97 | 30.77 |
| llama-8b-prefill-float16 | 82.312 | 76.492 | -7.07% | 27.53 | 27.07 |

## 精度
`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → `All check passed!`。
