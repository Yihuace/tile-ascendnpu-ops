# TileOPs Evaluation Report: BitwiseAndFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T10:24:46.867419+08:00 |
| evaluation_scope | BitwiseAndFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_101351_003780_BitwiseAndFwdOp |
| git_commit | dd451a57ff06bf78bc6886439cf06a45158c82bf |
| correctness_target | tests/ops/test_bitwise_and.py |
| benchmark_target | benchmarks/ops/bench_bitwise_and.py |

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
| BitwiseAndFwdOp | 100.0% (14/14) | 0.00e+00 | 8 | 6.96% – 76.08% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseAndFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 14.6003 | 63.84 | 63.84 | 3.42 | 95.76 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bool | msprof | main | 1.8000 |
| hidden-state-prefill | 49.4810 | 76.08 | 76.08 | 4.02 | 113.02 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int32 | msprof | main | 1.8000 |
| hidden-state-prefill | 127.2825 | 62.88 | 62.88 | 3.12 | 87.87 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int64 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 146.2029 | 6.96 | 4.42 | 6.96 | 9.76 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bool | msprof | main | 1.8000 |
| cnn-feat-broadcast | 39.6008 | 72.74 | 72.74 | 7.72 | 144.16 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int32 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 196.8239 | 34.97 | 34.97 | 12.20 | 58.01 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int64 | msprof | main | 1.8000 |
| row-broadcast | 60.8812 | 65.09 | 65.09 | 6.53 | 122.49 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int32 | msprof | main2 | 1.8000 |
| row-broadcast | 167.4433 | 59.29 | 59.29 | 4.75 | 89.07 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int64 | msprof | main2 | 1.8000 |
