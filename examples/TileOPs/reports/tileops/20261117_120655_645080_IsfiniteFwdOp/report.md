# TileOPs Evaluation Report: IsfiniteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-17T12:23:09.707766+08:00 |
| evaluation_scope | IsfiniteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261117_120655_645080_IsfiniteFwdOp |
| correctness_target | tests/ops/test_isfinite.py |
| benchmark_target | benchmarks/ops/bench_isfinite.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| IsfiniteFwdOp | 100.0% (9/9) | 0.00e+00 | 5 | 41.32% – 79.10% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### IsfiniteFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16m | 25.8400 | 74.20 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16m | 46.4000 | 41.32 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16m | 48.4800 | 79.10 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256m | 587.2200 | 71.79 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256m | 744.3600 | 56.67 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
