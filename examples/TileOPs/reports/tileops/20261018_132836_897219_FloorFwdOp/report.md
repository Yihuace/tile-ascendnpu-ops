# TileOPs Evaluation Report: FloorFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T13:35:48.308174+00:00 |
| evaluation_scope | FloorFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_132836_897219_FloorFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_floor.py |
| benchmark_target | benchmarks/ops/bench_floor.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 1 |
| cpu | x86_64 |
| cann | 9.0.0 |
| driver | 26.0.rc1 |
| pytorch | 2.10.0+cpu |
| pytorch_npu | 2.10.0 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-113-generic-x86_64-with-glibc2.38 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FloorFwdOp | 100.0% (8/8) | 0.00e+00 | 7 | 34.50% – 46.93% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FloorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 71.2829 | 34.50 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 70.3828 | 39.26 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 79.5432 | 46.87 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 280.8712 | 34.99 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 279.3112 | 39.53 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1151.7261 | 46.93 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1181.0072 | 45.76 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |
