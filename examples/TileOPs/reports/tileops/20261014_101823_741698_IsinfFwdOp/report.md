# TileOPs Evaluation Report: IsinfFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T10:34:15.422607+08:00 |
| evaluation_scope | IsinfFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_101823_741698_IsinfFwdOp |
| git_commit | af6c1f6b8bb3f1a89b4818778a42a7a275166365 |
| correctness_target | tests/ops/test_isinf.py |
| benchmark_target | benchmarks/ops/bench_isinf.py |

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
| IsinfFwdOp | 100.0% (9/9) | 0.00e+00 | 5 | 41.21% – 75.58% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### IsinfFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16m | 37.9000 | 50.59 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16m | 58.3600 | 41.21 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16m | 50.7400 | 75.58 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256m | 685.6200 | 61.54 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256m | 902.2400 | 46.75 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
