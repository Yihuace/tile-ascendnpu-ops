# TileOPs Evaluation Report: GroupedQueryAttentionSlidingWindowVarlenFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-08T02:53:40.144982+08:00 |
| evaluation_scope | GroupedQueryAttentionSlidingWindowVarlenFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp |
| git_commit | 9cfae94c0d10200c1795c16cd9c2e5a504fd63db |
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
| GroupedQueryAttentionSlidingWindowVarlenFwdOp | 100.0% (26/26) | 1.42e-04 | 8 | 14.05% – 25.15% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupedQueryAttentionSlidingWindowVarlenFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-3.1-8b-short-w256 | 314.8263 | 14.05 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | float16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-8b-short-w256 | 308.9062 | 14.33 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-8b-long-w1024 | 1774.7355 | 24.73 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | float16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-8b-long-w1024 | 1787.2557 | 24.58 | {"batch": 4, "dim": 128, "heads": 32, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-70b-short-w256 | 570.7314 | 15.50 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | float16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-70b-short-w256 | 569.8914 | 15.54 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [512, 512, 512, 512], "seqlens_q": [512, 512, 512, 512], "wl": 256, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-70b-long-w1024 | 3490.6899 | 25.15 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | float16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
| llama-3.1-70b-long-w1024 | 3508.1501 | 25.05 | {"batch": 4, "dim": 128, "heads": 64, "heads_kv": 8, "is_causal": true, "seqlens_k": [2048, 2048, 2048, 2048], "seqlens_q": [2048, 2048, 2048, 2048], "wl": 1024, "wr": -1} | bfloat16 | msprof | _gqa_sw_fwd_varlen_main | 1.7500 |
