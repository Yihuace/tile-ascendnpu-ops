# TileOPs Evaluation Report: BitwiseOrFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T10:38:30.314542+08:00 |
| evaluation_scope | BitwiseOrFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_102732_749504_BitwiseOrFwdOp |
| git_commit | dd451a57ff06bf78bc6886439cf06a45158c82bf |
| correctness_target | tests/ops/test_bitwise_or.py |
| benchmark_target | benchmarks/ops/bench_bitwise_or.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BitwiseOrFwdOp | 100.0% (14/14) | 0.00e+00 | 8 | 7.61% – 73.36% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseOrFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 14.9403 | 62.39 | 62.39 | 3.34 | 93.58 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bool | msprof | main | 1.8000 |
| hidden-state-prefill | 51.3010 | 73.36 | 73.36 | 3.88 | 109.01 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int32 | msprof | main | 1.8000 |
| hidden-state-prefill | 126.8625 | 62.76 | 62.76 | 3.13 | 88.16 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int64 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 133.8227 | 7.61 | 4.83 | 7.61 | 10.67 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bool | msprof | main | 1.8000 |
| cnn-feat-broadcast | 45.2409 | 65.44 | 65.44 | 6.76 | 126.19 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int32 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 194.8639 | 35.28 | 35.28 | 12.33 | 58.59 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int64 | msprof | main | 1.8000 |
| row-broadcast | 65.2213 | 61.51 | 61.51 | 6.09 | 114.34 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int32 | msprof | main2 | 1.8000 |
| row-broadcast | 169.0834 | 58.94 | 58.94 | 4.70 | 88.21 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int64 | msprof | main2 | 1.8000 |
