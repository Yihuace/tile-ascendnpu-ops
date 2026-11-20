# TileOPs Evaluation Report: CosFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.0.1.dev2 |
| date | 2026-11-03T08:45:25.987849+00:00 |
| evaluation_scope | CosFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261103_084040_855173_CosFwdOp |
| correctness_target | tests/ops/test_cos.py |
| benchmark_target | benchmarks/ops/bench_cos.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 1 |
| cpu | x86_64 |
| cann | 9.0.0 |
| driver | 26.0.rc1 |
| pytorch | 2.10.0+cpu |
| pytorch_npu | 2.10.0 |
| python | 3.11.15 |
| os | Linux-5.15.0-113-generic-x86_64-with-glibc2.38 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| CosFwdOp | 100.0% (9/9) | 2.93e-04 | 7 | 53.44% – 67.98% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### CosFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 97.0439 | 62.66 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 98.6039 | 64.75 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 108.1443 | 53.44 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 371.2748 | 65.49 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 377.9351 | 67.54 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1473.9790 | 65.97 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1501.9000 | 67.98 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |
