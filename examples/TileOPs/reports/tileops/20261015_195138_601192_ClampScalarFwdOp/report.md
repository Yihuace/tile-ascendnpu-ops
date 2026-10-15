# TileOPs Evaluation Report: ClampScalarFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T20:14:22.792714+08:00 |
| evaluation_scope | ClampScalarFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_195138_601192_ClampScalarFwdOp |
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
| ClampScalarFwdOp | 100.0% (11/11) | 0.00e+00 | 11 | 45.54% – 62.85% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampScalarFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16M | 42.1000 | 45.54 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M | 42.2800 | 51.74 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M | 68.4800 | 56.00 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256M | 891.4000 | 62.36 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256M | 884.6200 | 62.85 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 39.3400 | 48.74 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-min-only | 38.7000 | 49.55 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 63.9600 | 59.96 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 16M-max-only | 38.6000 | 49.67 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-max-only | 37.5000 | 51.13 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-max-only | 63.2600 | 60.62 | [4096, 4096] | float32 | msprof | main | 1.7500 |
