# TileOPs Evaluation Report: ExpFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T13:43:04.280578+00:00 |
| evaluation_scope | ExpFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_133552_071886_ExpFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_exp.py |
| benchmark_target | benchmarks/ops/bench_exp.py |

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
| ExpFwdOp | 100.0% (8/8) | 0.00e+00 | 7 | 64.14% – 76.46% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 28.4411 | 65.54 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 29.0612 | 64.14 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 48.7620 | 76.46 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 134.5854 | 69.27 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 139.3656 | 67.58 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 779.1111 | 69.32 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 786.5715 | 68.58 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |
