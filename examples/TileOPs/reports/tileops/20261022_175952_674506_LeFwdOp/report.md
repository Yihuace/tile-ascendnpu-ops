# TileOPs Evaluation Report: LeFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T18:13:04.638309+08:00 |
| evaluation_scope | LeFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_175952_674506_LeFwdOp |
| git_commit | a79a1316be3962aec3c034a50b2b510275347fe2 |
| correctness_target | tests/ops/test_le.py |
| benchmark_target | benchmarks/ops/bench_le.py |

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
| LeFwdOp | 100.0% (18/18) | 0.00e+00 | 9 | 44.49% – 75.09% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LeFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 28.0600 | 66.43 | 66.43 | 10.84 | 83.04 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-prefill | 26.8800 | 69.35 | 69.35 | 26.15 | 86.69 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| hidden-state-prefill | 50.3600 | 74.03 | 74.03 | 6.17 | 83.29 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.8000 |
| cnn-feat-broadcast | 23.0200 | 62.00 | 62.00 | 26.97 | 93.00 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 32.0800 | 44.49 | 44.49 | 38.45 | 66.74 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main2 | 1.8000 |
| cnn-feat-broadcast | 38.2000 | 74.72 | 74.72 | 20.15 | 93.41 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 49.6600 | 75.09 | 75.09 | 12.41 | 93.86 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float32 | msprof | main2 | 1.8000 |
| row-broadcast | 27.5200 | 67.75 | 67.75 | 22.22 | 101.62 | input_shape=[4096, 4096]; other_shape=[1, 4096] | float16 | msprof | main2 | 1.8000 |
| row-broadcast | 34.2200 | 54.49 | 54.49 | 41.22 | 81.73 | input_shape=[4096, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main2 | 1.8000 |
