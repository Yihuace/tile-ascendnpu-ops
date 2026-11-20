# TileOPs Evaluation Report: GroupedQueryAttentionSlidingWindowVarlenFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T15:50:06.039214+08:00 |
| evaluation_scope | GroupedQueryAttentionSlidingWindowVarlenFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_154414_151142_GroupedQueryAttentionSlidingWindowVarlenFwdOp |
| git_commit | e3abcbd7a12428c5e1965588305aab764477ce68 |
| correctness_target | tests/ops/test_grouped_query_attention_sliding_window_varlen.py |
| benchmark_target | benchmarks/ops/bench_grouped_query_attention_sliding_window_varlen.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 26 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GroupedQueryAttentionSlidingWindowVarlenFwdOp | 100.0% (26/26) | 1.42e-04 | 10 | 0.01% – 0.02% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupedQueryAttentionSlidingWindowVarlenFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| gqa-sw-varlen-fwd-smoke | 3.6400 | 0.01 | {"batch": 2, "dim": 64, "heads": 8, "heads_kv": 2, "is_causal": true, "seqlens_k": [128, 128], "seqlens_q": [128, 128], "wl": -1, "wr": -1} | float16 | msprof | N/A | 1.7500 |
| gqa-sw-varlen-fwd-smoke | 3.3800 | 0.01 | {"batch": 2, "dim": 64, "heads": 8, "heads_kv": 2, "is_causal": true, "seqlens_k": [128, 128], "seqlens_q": [128, 128], "wl": -1, "wr": -1} | bfloat16 | msprof | N/A | 1.7500 |
| llama-3.1-8b-short-w256 | 3.6000 | 0.02 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | float16 | msprof | N/A | 1.7500 |
| llama-3.1-8b-short-w256 | 3.6800 | 0.01 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | bfloat16 | msprof | N/A | 1.7500 |
| llama-3.1-8b-long-w1024 | 3.7400 | 0.01 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | float16 | msprof | N/A | 1.7500 |
| llama-3.1-8b-long-w1024 | 4.1800 | 0.01 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | bfloat16 | msprof | N/A | 1.7500 |
| llama-3.1-70b-short-w256 | 3.2600 | 0.02 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | float16 | msprof | N/A | 1.7500 |
| llama-3.1-70b-short-w256 | 3.2800 | 0.01 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | bfloat16 | msprof | N/A | 1.7500 |
| llama-3.1-70b-long-w1024 | 3.5400 | 0.02 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | float16 | msprof | N/A | 1.7500 |
| llama-3.1-70b-long-w1024 | 4.2600 | 0.01 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | bfloat16 | msprof | N/A | 1.7500 |

## Warnings

- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-c126a6672340: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-4e0b9ce1a8e8: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-b0e4e6267c86: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-c42935f41bd4: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-87e2aca33b14: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-78a67f644b76: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-4f6f5ee9cf7f: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-f76d12797acf: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-1575d6eebd41: invalid tileops profile: unfiltered capture contains 6 distinct ops
- GroupedQueryAttentionSlidingWindowVarlenFwdOp/GroupedQueryAttentionSlidingWindowVarlenFwdOp-2bf46b6c5bbf: invalid tileops profile: unfiltered capture contains 6 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
