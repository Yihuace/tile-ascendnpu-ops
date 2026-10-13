# TileOPs Evaluation Report: BitwiseNotFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T18:38:22.208951+08:00 |
| evaluation_scope | BitwiseNotFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_181824_602023_BitwiseNotFwdOp |
| git_commit | d3a1c3749e537473d36b171196b105bb3f103c2f |
| correctness_target | tests/ops/test_bitwise_not.py |
| benchmark_target | benchmarks/ops/bench_bitwise_not.py |

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
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BitwiseNotFwdOp | 100.0% (11/11) | 0.00e+00 | 7 | 32.50% – 76.60% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseNotFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16m | 50.0600 | 76.60 | [4096, 4096] | int32 | msprof | main | 1.7500 |
| 16m | 129.7000 | 72.43 | [4096, 4096] | int64 | msprof | main | 1.7500 |
| 256m | 1610.9399 | 72.50 | [16384, 16384] | int32 | msprof | main | 1.7500 |
| 16m-int-small | 29.5000 | 32.50 | [4096, 4096] | bool | msprof | main | 1.7500 |
| 16m-int-small | 23.6400 | 40.55 | [4096, 4096] | uint8 | msprof | main | 1.7500 |
| 16m-int-small | 23.0000 | 41.68 | [4096, 4096] | int8 | msprof | main | 1.7500 |
| 16m-int-small | 29.5200 | 64.95 | [4096, 4096] | int16 | msprof | main | 1.7500 |
