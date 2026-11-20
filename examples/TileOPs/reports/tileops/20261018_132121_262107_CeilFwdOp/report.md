# TileOPs Evaluation Report: CeilFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T13:28:33.089808+00:00 |
| evaluation_scope | CeilFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_132121_262107_CeilFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_ceil.py |
| benchmark_target | benchmarks/ops/bench_ceil.py |

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
| CeilFwdOp | 100.0% (8/8) | 0.00e+00 | 7 | 35.08% – 46.91% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### CeilFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 70.1028 | 35.08 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 69.7428 | 39.62 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 79.4832 | 46.91 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 278.4912 | 35.29 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 278.6912 | 39.61 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1162.4666 | 46.49 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1180.8672 | 45.78 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |
