# TileOPs Evaluation Report: ClampScalarFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T20:50:40.416326+08:00 |
| evaluation_scope | ClampScalarFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_202825_275014_ClampScalarFwdOp |
| git_commit | 88d028d251378cbb6b366ed1fab8cfda70ad4938 |
| correctness_target | tests/ops/test_clamp_scalar.py |
| benchmark_target | benchmarks/ops/bench_clamp_scalar.py |

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
| ClampScalarFwdOp | 100.0% (11/11) | 0.00e+00 | 11 | 71.86% – 82.83% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampScalarFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16M | 26.1000 | 73.46 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M | 25.9200 | 73.97 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M | 46.3000 | 82.83 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256M | 751.3800 | 73.94 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256M | 773.3200 | 71.86 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 26.1600 | 73.30 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-min-only | 24.7400 | 77.50 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 49.4200 | 77.60 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 16M-max-only | 26.1600 | 73.30 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-max-only | 25.5800 | 74.96 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-max-only | 47.5800 | 80.96 | [4096, 4096] | float32 | msprof | main | 1.7500 |
