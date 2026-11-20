# TileOPs Evaluation Report: GeFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T16:10:44.814559+08:00 |
| evaluation_scope | GeFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_155422_135963_GeFwdOp |
| git_commit | a79a1316be3962aec3c034a50b2b510275347fe2 |
| correctness_target | tests/ops/test_ge.py |
| benchmark_target | benchmarks/ops/bench_ge.py |

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
| GeFwdOp | 100.0% (18/18) | 0.00e+00 | 9 | 44.77% – 74.43% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 25.7000 | 72.53 | 72.53 | 11.84 | 90.67 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-prefill | 28.7800 | 64.77 | 64.77 | 24.43 | 80.96 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| hidden-state-prefill | 50.2400 | 74.21 | 74.21 | 6.18 | 83.49 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 23.2600 | 61.36 | 61.36 | 26.69 | 92.04 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 31.8800 | 44.77 | 44.77 | 38.69 | 67.15 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 38.4600 | 74.22 | 74.22 | 20.01 | 92.78 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 50.1000 | 74.43 | 74.43 | 12.30 | 93.04 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 29.5000 | 63.21 | 63.21 | 20.73 | 94.80 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float16 | msprof | main2 | 1.8000 |
| row-broadcast | 34.1000 | 54.68 | 54.68 | 41.37 | 82.01 | input_shape=[4096, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main2 | 1.8000 |
