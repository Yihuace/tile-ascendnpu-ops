# TileOPs Evaluation Report: IsnanFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T09:47:28.496493+08:00 |
| evaluation_scope | IsnanFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_093134_398669_IsnanFwdOp |
| git_commit | dd6ca892fc378ba9e3e08a21f0916cbcab407781 |
| correctness_target | tests/ops/test_isnan.py |
| benchmark_target | benchmarks/ops/bench_isnan.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| IsnanFwdOp | 100.0% (9/9) | 0.00e+00 | 5 | 51.91% – 74.40% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### IsnanFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16m | 30.9600 | 61.93 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16m | 36.9400 | 51.91 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16m | 51.5400 | 74.40 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256m | 643.3200 | 65.58 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256m | 658.3200 | 64.04 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
