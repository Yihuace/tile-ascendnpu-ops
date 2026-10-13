# TileOPs Evaluation Report: ClampScalarFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T14:37:04.052723+08:00 |
| evaluation_scope | ClampScalarFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_141507_226988_ClampScalarFwdOp |
| git_commit | 66293a38ea6790a5c5dddb48f97fcc1ede4f73ca |
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
| ClampScalarFwdOp | 100.0% (11/11) | 0.00e+00 | 11 | 44.93% – 71.81% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampScalarFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16M | 42.6800 | 44.93 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M | 43.0000 | 50.87 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M | 55.9800 | 68.50 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256M | 878.0600 | 63.32 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256M | 874.2600 | 63.60 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 39.8400 | 48.13 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-min-only | 39.6200 | 48.39 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 54.8600 | 69.90 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 16M-max-only | 39.3800 | 48.69 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-max-only | 39.0000 | 49.16 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-max-only | 53.4000 | 71.81 | [4096, 4096] | float32 | msprof | main | 1.7500 |
