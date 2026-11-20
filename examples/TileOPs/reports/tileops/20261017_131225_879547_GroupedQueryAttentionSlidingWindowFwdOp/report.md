# TileOPs Evaluation Report: GroupedQueryAttentionSlidingWindowFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T13:17:13.922912+08:00 |
| evaluation_scope | GroupedQueryAttentionSlidingWindowFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_131225_879547_GroupedQueryAttentionSlidingWindowFwdOp |
| git_commit | cebcec9dd203208e30657c667cdc87d6526d1829 |
| correctness_target | tests/ops/test_grouped_query_attention_sliding_window.py |
| benchmark_target | benchmarks/ops/bench_grouped_query_attention_sliding_window.py |

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
| 100.0% | 1 | 22 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GroupedQueryAttentionSlidingWindowFwdOp | 100.0% (22/22) | 1.63e-04 | 10 | 3.19% – 21.37% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupedQueryAttentionSlidingWindowFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| gqa-sw-fwd-smoke-s512-h8-kv2-d64 | 85.6800 | 3.28 | {"batch": 2, "dim": 64, "heads": 8, "heads_kv": 2, "is_causal": true, "seq": 512, "wl": -1, "wr": -1} | float16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| gqa-sw-fwd-smoke-s512-h8-kv2-d64 | 88.0600 | 3.19 | {"batch": 2, "dim": 64, "heads": 8, "heads_kv": 2, "is_causal": true, "seq": 512, "wl": -1, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-8b-short-w256 | 348.5200 | 12.71 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seq": 512, "wl": 256, "wr": -1} | float16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-8b-short-w256 | 347.4200 | 12.74 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seq": 512, "wl": 256, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-8b-long-w1024 | 1056.9000 | 20.80 | {"batch": 2, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seq": 2048, "wl": 1024, "wr": -1} | float16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-8b-long-w1024 | 1027.9800 | 21.37 | {"batch": 2, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seq": 2048, "wl": 1024, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-70b-short-w256 | 349.8200 | 12.66 | {"batch": 2, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seq": 512, "wl": 256, "wr": -1} | float16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-70b-short-w256 | 350.3400 | 12.64 | {"batch": 2, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seq": 512, "wl": 256, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-70b-long-w1024 | 1047.9600 | 20.98 | {"batch": 1, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seq": 2048, "wl": 1024, "wr": -1} | float16 | msprof | _gqa_sw_fwd_main | 1.7500 |
| llama-3.1-70b-long-w1024 | 1057.2000 | 20.78 | {"batch": 1, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seq": 2048, "wl": 1024, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_main | 1.7500 |
