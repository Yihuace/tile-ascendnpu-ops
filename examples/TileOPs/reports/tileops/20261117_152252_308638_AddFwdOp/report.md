# TileOPs Evaluation Report: AddFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-17T15:42:00.108597+08:00 |
| evaluation_scope | AddFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261117_152252_308638_AddFwdOp |
| correctness_target | tests/ops/test_add.py |
| benchmark_target | benchmarks/ops/bench_add.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 25 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AddFwdOp | 100.0% (25/25) | 0.00e+00 | 11 | 51.26% – 83.06% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AddFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 46.8200 | 81.91 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.7500 |
| hidden-state-prefill | 25.0600 | 76.51 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 42.2400 | 69.51 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 22.7000 | 68.01 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 23.4000 | 65.98 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| llama-7b-ffn | 62.0400 | 83.06 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float32 | msprof | main | 1.7500 |
| llama-7b-ffn | 33.9200 | 75.96 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float16 | msprof | main | 1.7500 |
| llama-7b-ffn | 32.6400 | 78.94 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | bfloat16 | msprof | main | 1.7500 |
| linear-bias | 15.8400 | 60.58 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float32 | msprof | main | 1.7500 |
| linear-bias | 8.6800 | 55.28 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float16 | msprof | main | 1.7500 |
| linear-bias | 9.3600 | 51.26 | input_shape=[1024, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main | 1.7500 |
