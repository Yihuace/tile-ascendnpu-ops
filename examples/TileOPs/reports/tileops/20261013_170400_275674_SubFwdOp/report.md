# TileOPs Evaluation Report: SubFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T17:26:43.938868+08:00 |
| evaluation_scope | SubFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_170400_275674_SubFwdOp |
| git_commit | 676c6e951b73cae4e530ca2ac1b66a9ada0ff0c0 |
| correctness_target | tests/ops/test_sub.py |
| benchmark_target | benchmarks/ops/bench_sub.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 24 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| SubFwdOp | 100.0% (24/24) | 0.00e+00 | 15 | 68.43% – 82.70% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SubFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 47.4400 | 80.83 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.7500 |
| hidden-state-prefill | 25.7400 | 74.49 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 71.0000 | 82.70 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 39.0000 | 75.28 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 37.7600 | 77.75 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| llama-7b-ffn | 64.2000 | 80.27 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float32 | msprof | main | 1.7500 |
| llama-7b-ffn | 33.1000 | 77.84 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float16 | msprof | main | 1.7500 |
| llama-7b-ffn | 32.5200 | 79.23 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | bfloat16 | msprof | main | 1.7500 |
| linear-bias | 28.0200 | 68.43 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float32 | msprof | main | 1.7500 |
| linear-bias | 13.7200 | 69.88 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float16 | msprof | main | 1.7500 |
| linear-bias | 13.8200 | 69.37 | input_shape=[1024, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main | 1.7500 |
| llama-7b-hidden | 13.7000 | 69.98 | input_shape=[1024, 4096]; other_shape=[1024, 4096] | float16 | msprof | main | 1.7500 |
| phi-2-ffn | 31.1600 | 76.92 | input_shape=[1024, 10240]; other_shape=[1024, 10240] | float16 | msprof | main | 1.7500 |
| phi-2-ffn-bias | 31.6000 | 75.85 | input_shape=[1024, 10240]; other_shape=[1, 10240] | float16 | msprof | main | 1.7500 |
| llama-7b-ffn-bias | 33.1200 | 77.79 | input_shape=[1024, 11008]; other_shape=[1, 11008] | float16 | msprof | main | 1.7500 |
