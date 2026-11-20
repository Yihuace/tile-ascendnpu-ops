# TileOPs Evaluation Report: MultiHeadAttentionDecodePagedWithKVCacheFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T07:47:41.739893+08:00 |
| evaluation_scope | MultiHeadAttentionDecodePagedWithKVCacheFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_074433_562940_MultiHeadAttentionDecodePagedWithKVCacheFwdOp |
| git_commit | 75e21fe37e8acccc29e6162aa89da8324d2d12d0 |
| correctness_target | tests/ops/test_mha_decode_paged.py |
| benchmark_target | benchmarks/ops/bench_mha_decode_paged.py |

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
| MultiHeadAttentionDecodePagedWithKVCacheFwdOp | 100.0% (5/5) | 3.51e-04 | 4 | 0.01% – 0.02% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MultiHeadAttentionDecodePagedWithKVCacheFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| single-token-page128 | 2.5200 | 0.02 | {"batch": 1, "dim": 128, "heads": 16, "is_causal": false, "page_size": 128, "seqlen_kv": 512, "seqlen_q": 1} | float16 | msprof | N/A | 1.7500 |
| batch2-page256 | 2.3800 | 0.01 | {"batch": 2, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 1024, "seqlen_q": 1} | float16 | msprof | N/A | 1.7500 |
| longer-cache | 2.3200 | 0.01 | {"batch": 1, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 1024, "seqlen_q": 1} | float16 | msprof | N/A | 1.7500 |
| shorter-cache | 1.8200 | 0.02 | {"batch": 1, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 512, "seqlen_q": 1} | float16 | msprof | N/A | 1.7500 |

## Warnings

- MultiHeadAttentionDecodePagedWithKVCacheFwdOp/MultiHeadAttentionDecodePagedWithKVCacheFwdOp-3a63069a7739: invalid tileops profile: unfiltered capture contains 8 distinct ops
- MultiHeadAttentionDecodePagedWithKVCacheFwdOp/MultiHeadAttentionDecodePagedWithKVCacheFwdOp-e227385bc4ce: invalid tileops profile: unfiltered capture contains 9 distinct ops
- MultiHeadAttentionDecodePagedWithKVCacheFwdOp/MultiHeadAttentionDecodePagedWithKVCacheFwdOp-886077985f85: invalid tileops profile: unfiltered capture contains 8 distinct ops
- MultiHeadAttentionDecodePagedWithKVCacheFwdOp/MultiHeadAttentionDecodePagedWithKVCacheFwdOp-5d2b4d56f611: invalid tileops profile: unfiltered capture contains 8 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
