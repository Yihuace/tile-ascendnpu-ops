# TileOPs Evaluation Report: BitwiseXorFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T16:50:17.381291+08:00 |
| evaluation_scope | BitwiseXorFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_164300_363709_BitwiseXorFwdOp |
| git_commit | 3d212b67b10f1c528ecaf74abc45b1bbf2423c47 |
| correctness_target | tests/ops/test_bitwise_xor.py |
| benchmark_target | benchmarks/ops/bench_bitwise_xor.py |

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
| BitwiseXorFwdOp | 100.0% (14/14) | 0.00e+00 | 8 | 29.17% – 69.53% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseXorFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 16.1997 | 57.54 | 57.54 | 12.27 | 86.30 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bool | msprof | main | 1.8000 |
| hidden-state-prefill | 54.4589 | 69.53 | 69.53 | 14.59 | 102.69 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int32 | msprof | main | 1.8000 |
| hidden-state-prefill | 129.3974 | 62.35 | 62.35 | 12.28 | 86.44 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | int64 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 22.1996 | 32.15 | 32.15 | 27.82 | 64.29 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bool | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 48.1590 | 61.09 | 61.09 | 31.77 | 118.54 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int32 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 232.8553 | 29.17 | 29.17 | 20.55 | 49.03 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | int64 | msprof | main | 1.8000 |
| row-broadcast | 64.2987 | 61.22 | 61.22 | 24.84 | 115.98 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int32 | msprof | main2 | 1.8000 |
| row-broadcast | 170.3166 | 58.59 | 58.59 | 18.66 | 87.57 | input_shape=[4096, 4096]; other_shape=[1, 4096] | int64 | msprof | main | 1.8000 |
