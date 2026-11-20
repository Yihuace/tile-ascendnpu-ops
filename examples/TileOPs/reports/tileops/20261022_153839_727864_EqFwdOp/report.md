# TileOPs Evaluation Report: EqFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T15:54:08.520320+08:00 |
| evaluation_scope | EqFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_153839_727864_EqFwdOp |
| git_commit | a79a1316be3962aec3c034a50b2b510275347fe2 |
| correctness_target | tests/ops/test_eq.py |
| benchmark_target | benchmarks/ops/bench_eq.py |

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
| EqFwdOp | 100.0% (18/18) | 0.00e+00 | 9 | 44.08% – 78.92% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EqFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 23.6200 | 78.92 | 78.92 | 12.88 | 98.65 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-prefill | 26.9200 | 69.25 | 69.25 | 26.12 | 86.56 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| hidden-state-prefill | 49.9400 | 74.66 | 74.66 | 6.22 | 83.99 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 23.0200 | 62.00 | 62.00 | 26.97 | 93.00 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 32.3800 | 44.08 | 44.08 | 38.10 | 66.12 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 37.9800 | 75.16 | 75.16 | 20.27 | 93.95 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 52.6400 | 70.88 | 70.88 | 11.71 | 88.55 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 30.3400 | 61.46 | 61.46 | 20.16 | 92.18 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float16 | msprof | main2 | 1.8000 |
| row-broadcast | 34.3800 | 54.23 | 54.23 | 41.03 | 81.35 | input_shape=[4096, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main2 | 1.8000 |
