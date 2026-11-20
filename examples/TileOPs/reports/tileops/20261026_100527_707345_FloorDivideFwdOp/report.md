# TileOPs Evaluation Report: FloorDivideFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-26T10:18:13.692210+08:00 |
| evaluation_scope | FloorDivideFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261026_100527_707345_FloorDivideFwdOp |
| git_commit | 04f9f09f1e6c683f8fc2ccce95b28cf4fb436a4f |
| correctness_target | tests/ops/test_floor_divide.py |
| benchmark_target | benchmarks/ops/bench_floor_divide.py |

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
| 100.0% | 1 | 16 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FloorDivideFwdOp | 100.0% (16/16) | 0.00e+00 | 9 | 29.95% – 63.28% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FloorDivideFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 29.4600 | 63.28 | 63.28 | 33.87 | 94.92 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-prefill | 30.5000 | 61.12 | 61.12 | 39.23 | 91.68 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| hidden-state-prefill | 72.5400 | 51.73 | 51.73 | 25.70 | 77.09 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 29.8200 | 47.86 | 47.86 | 41.06 | 95.72 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main3 | 1.8000 |
| cnn-feat-broadcast | 31.8400 | 48.01 | 44.83 | 48.01 | 89.65 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main3 | 1.8000 |
| cnn-feat-broadcast | 89.6000 | 32.85 | 32.85 | 32.67 | 63.72 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 132.2600 | 29.95 | 29.95 | 28.25 | 56.38 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float32 | msprof | main | 1.8000 |
| row-broadcast | 35.1000 | 53.12 | 53.12 | 45.65 | 106.23 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float16 | msprof | main2 | 1.8000 |
| row-broadcast | 37.5200 | 53.29 | 49.70 | 53.29 | 99.38 | input_shape=[4096, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main2 | 1.8000 |
