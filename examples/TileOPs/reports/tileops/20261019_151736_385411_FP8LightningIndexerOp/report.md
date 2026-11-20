# TileOPs Evaluation Report: FP8LightningIndexerOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-19T15:20:19.186625+08:00 |
| evaluation_scope | FP8LightningIndexerOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261019_151736_385411_FP8LightningIndexerOp |
| git_commit | c840ae1763725f068619af474dc5eac630fa6209 |
| correctness_target | tests/ops/test_fp8_lightning_indexer.py |
| benchmark_target | benchmarks/ops/bench_fp8_lightning_indexer.py |

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
| 100.0% | 1 | 4 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FP8LightningIndexerOp | 100.0% (4/4) | 0.00e+00 | 3 | 2.77% – 9.04% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FP8LightningIndexerOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| lightning-indexer-smoke-s1k-h8-d64 | 134.9400 | 2.77 | {"batch": 1, "clean_logits": true, "heads": 8, "index_dim": 64, "kv_group": 1, "seq_len": 1024, "seq_len_kv": 1024} | N/A | msprof | _fp8_lightning_indexer_main | 1.7500 |
| lightning-indexer-bf16-s8k-h32-d64 | 10519.7002 | 9.04 | {"batch": 1, "clean_logits": true, "heads": 32, "index_dim": 64, "kv_group": 1, "seq_len": 8192, "seq_len_kv": 8192} | N/A | msprof | _fp8_lightning_indexer_main | 1.7500 |
| lightning-indexer-bf16-s4k-h32-d64 | 5288.1602 | 8.99 | {"batch": 1, "clean_logits": true, "heads": 32, "index_dim": 64, "kv_group": 1, "seq_len": 4096, "seq_len_kv": 8192} | N/A | msprof | _fp8_lightning_indexer_main | 1.7500 |
