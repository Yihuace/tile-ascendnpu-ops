# TileOPs Evaluation Report: HardswishFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T15:21:33.875979+08:00 |
| evaluation_scope | HardswishFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_150641_447235_HardswishFwdOp |
| git_commit | ab5724276f59687988244301b03d541b5f4285de |
| correctness_target | tests/ops/test_hardswish.py |
| benchmark_target | benchmarks/ops/bench_hardswish.py |

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
| 100.0% | 1 | 7 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardswishFwdOp | 100.0% (7/7) | 9.54e-08 | 6 | 49.89% – 60.61% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardswishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mbv3-stage2 | 22.9400 | 54.97 | [32, 96, 56, 56] | float16 | msprof | main | 1.7500 |
| mbv3-stage2 | 24.3400 | 60.40 | [32, 96, 56, 56] | bfloat16 | msprof | main | 1.7500 |
| mbv3-stage3 | 15.8000 | 49.89 | [32, 240, 28, 28] | float16 | msprof | main | 1.7500 |
| mbv3-stage3 | 16.2000 | 56.72 | [32, 240, 28, 28] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 20.1200 | 54.58 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 21.1200 | 60.61 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
