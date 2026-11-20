# TileOPs Evaluation Report: ClampMaxFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T13:59:04.312804+08:00 |
| evaluation_scope | ClampMaxFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_135036_077346_ClampMaxFwdOp |
| git_commit | 2238099280b481491f96f99482c86246a09c8c68 |
| correctness_target | tests/ops/test_clamp_max.py |
| benchmark_target | benchmarks/ops/bench_clamp_max.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910_9362 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.5 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2220_156.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 15 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ClampMaxFwdOp | 100.0% (15/15) | 0.00e+00 | 9 | 63.32% – 69.15% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampMaxFwdOp

| Label | Latency (us) | Ratio (%) | Achieved BW (TB/s) | BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| elementwise-16M | 55.1389 | 68.14 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16M | 54.5789 | 68.84 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16M | 129.8774 | 63.92 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[4096, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 45.9991 | 63.32 | N/A | N/A | input_shape=[16, 256, 56, 56]; max_shape=[256, 1, 1] | float16 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 41.2792 | 69.15 | N/A | N/A | input_shape=[16, 256, 56, 56]; max_shape=[256, 1, 1] | bfloat16 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 93.6381 | 63.63 | N/A | N/A | input_shape=[16, 256, 56, 56]; max_shape=[256, 1, 1] | float32 | msprof | main | 1.8000 |
| row-broadcast | 128.1574 | 63.42 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[1, 4096] | float32 | msprof | main | 1.8000 |
| row-broadcast | 57.5988 | 65.94 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[1, 4096] | float16 | msprof | main | 1.8000 |
| row-broadcast | 60.3188 | 63.45 | N/A | N/A | input_shape=[4096, 4096]; max_shape=[1, 4096] | bfloat16 | msprof | main | 1.8000 |
