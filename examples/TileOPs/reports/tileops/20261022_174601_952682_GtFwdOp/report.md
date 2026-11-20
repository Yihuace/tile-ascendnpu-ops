# TileOPs Evaluation Report: GtFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T17:59:05.773593+08:00 |
| evaluation_scope | GtFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_174601_952682_GtFwdOp |
| git_commit | a79a1316be3962aec3c034a50b2b510275347fe2 |
| correctness_target | tests/ops/test_gt.py |
| benchmark_target | benchmarks/ops/bench_gt.py |

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
| 100.0% | 1 | 18 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GtFwdOp | 100.0% (18/18) | 0.00e+00 | 9 | 44.80% – 74.80% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GtFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 26.5800 | 70.13 | 70.13 | 11.45 | 87.67 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-prefill | 27.0600 | 68.89 | 68.89 | 25.98 | 86.11 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| hidden-state-prefill | 49.9400 | 74.66 | 74.66 | 6.22 | 83.99 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 23.1600 | 61.63 | 61.63 | 26.80 | 92.44 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 31.8600 | 44.80 | 44.80 | 38.72 | 67.20 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 38.1600 | 74.80 | 74.80 | 20.17 | 93.50 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 50.4400 | 73.93 | 73.93 | 12.22 | 92.41 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 26.9400 | 69.21 | 69.21 | 22.70 | 103.81 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float16 | msprof | main2 | 1.8000 |
| row-broadcast | 34.1400 | 54.62 | 54.62 | 41.32 | 81.92 | input_shape=[4096, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main2 | 1.8000 |
