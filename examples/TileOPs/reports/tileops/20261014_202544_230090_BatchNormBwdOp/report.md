# TileOPs Evaluation Report: BatchNormBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T20:29:27.904054+08:00 |
| evaluation_scope | BatchNormBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_202544_230090_BatchNormBwdOp |
| git_commit | b1d4bae7ab245be7f52ac7fa7c501d4d81359a58 |
| correctness_target | tests/ops/test_batch_norm_bwd.py |
| benchmark_target | benchmarks/ops/bench_batch_norm_bwd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BatchNormBwdOp | 100.0% (12/12) | 2.29e-03 | 6 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BatchNormBwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| resnet-stage1 | 20.7800 | N/A | {"c": 64, "n": 8, "s": 1024} | float16 | msprof | main | 0.1514 |
| resnet-stage1 | 21.1000 | N/A | {"c": 64, "n": 8, "s": 1024} | bfloat16 | msprof | main | 0.1491 |
| resnet-stage2 | 24.9800 | N/A | {"c": 128, "n": 4, "s": 1024} | float16 | msprof | main | 0.1260 |
| resnet-stage3 | 40.4200 | N/A | {"c": 256, "n": 4, "s": 784} | float16 | msprof | main | 0.1192 |
| resnet-stage3 | 41.5000 | N/A | {"c": 256, "n": 4, "s": 784} | float32 | msprof | main | 0.2323 |
| large-spatial | 2085.3401 | N/A | {"c": 128, "n": 4, "s": 262144} | float16 | msprof | N/A | 0.3862 |

## Warnings

- BatchNormBwdOp/BatchNormBwdOp-20ebc8b7ce60: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- correctness passed, but the benchmark stage did not produce valid records
