# TileOPs Evaluation Report: MultiHeadAttentionDecodePagedWithKVCacheFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-19T16:14:27.267616+08:00 |
| evaluation_scope | MultiHeadAttentionDecodePagedWithKVCacheFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261119_160151_272035_MultiHeadAttentionDecodePagedWithKVCacheFwdOp |
| git_commit | 163fe0dd626aef01a905f84527f380b0c2eb8a31 |
| correctness_target | tests/ops/test_mha_decode_paged.py |
| benchmark_target | benchmarks/ops/bench_mha_decode_paged.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.3.0-beta.1 |
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
| MultiHeadAttentionDecodePagedWithKVCacheFwdOp | 100.0% (5/5) | 3.36e-04 | 4 | 13.62% – 33.96% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MultiHeadAttentionDecodePagedWithKVCacheFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| single-token-page128 | 7.0400 | 33.96 | {"batch": 1, "dim": 128, "heads": 16, "is_causal": false, "page_size": 128, "seqlen_kv": 512, "seqlen_q": 1} | float16 | msprof | decode_vector_packed | 1.7500 |
| batch2-page256 | 7.6200 | 31.24 | {"batch": 2, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 1024, "seqlen_q": 1} | float16 | msprof | decode_vector_packed | 1.7500 |
| longer-cache | 5.5000 | 21.64 | {"batch": 1, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 1024, "seqlen_q": 1} | float16 | msprof | decode_vector_packed | 1.7500 |
| shorter-cache | 4.3400 | 13.62 | {"batch": 1, "dim": 64, "heads": 8, "is_causal": false, "page_size": 256, "seqlen_kv": 512, "seqlen_q": 1} | float16 | msprof | decode_vector_packed | 1.7500 |
