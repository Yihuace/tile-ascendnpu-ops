# TileOPs Evaluation Report: GroupedQueryAttentionBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T01:38:24.990886+08:00 |
| evaluation_scope | GroupedQueryAttentionBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_010918_087690_GroupedQueryAttentionBwdOp |
| git_commit | 5a90b9918747fe9ed4870e6cd5c39760c68c33e8 |
| correctness_target | tests/ops/test_grouped_query_attention_bwd.py |
| benchmark_target | benchmarks/ops/bench_grouped_query_attention_bwd.py |

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
| 100.0% | 1 | 5 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GroupedQueryAttentionBwdOp | 100.0% (5/5) | 1.53e-03 | 9 | 0.11% – 1.54% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupedQueryAttentionBwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| gqa-bwd-smoke | 198.2200 | 0.11 | {"batch": 1, "causal": true, "dim": 64, "heads": 8, "heads_kv": 2, "seq_len": 128} | float16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-8b-short | 5870.3198 | 1.42 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "heads_kv": 8, "seq_len": 512} | float16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-8b-short | 5889.5200 | 1.42 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "heads_kv": 8, "seq_len": 512} | bfloat16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-8b-long | 39857.4375 | 1.53 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "heads_kv": 8, "seq_len": 2048} | float16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-8b-long | 39811.2578 | 1.54 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "heads_kv": 8, "seq_len": 2048} | bfloat16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-70b-short | 6200.7397 | 1.34 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "heads_kv": 8, "seq_len": 512} | float16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-70b-short | 6211.4600 | 1.35 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "heads_kv": 8, "seq_len": 512} | bfloat16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-70b-long | 40054.7773 | 1.53 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "heads_kv": 8, "seq_len": 2048} | float16 | msprof | _gqa_bwd_main | 1.7500 |
| llama-3.1-70b-long | 40100.6016 | 1.53 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "heads_kv": 8, "seq_len": 2048} | bfloat16 | msprof | _gqa_bwd_main | 1.7500 |
