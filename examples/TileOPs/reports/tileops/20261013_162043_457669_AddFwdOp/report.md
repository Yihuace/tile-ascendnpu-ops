# TileOPs Evaluation Report: AddFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T16:38:38.808831+08:00 |
| evaluation_scope | AddFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_162043_457669_AddFwdOp |
| git_commit | 8e1cfb5d4a717dfba57ccb27d64b53ebd85ae2dc |
| correctness_target | tests/ops/test_add.py |
| benchmark_target | benchmarks/ops/bench_add.py |

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
| AddFwdOp | 100.0% (24/24) | 0.00e+00 | 11 | 66.12% – 82.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AddFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 48.4000 | 79.23 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.7500 |
| hidden-state-prefill | 24.9800 | 76.76 | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 74.8600 | 78.44 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 37.5400 | 78.21 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 38.9200 | 75.44 | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| llama-7b-ffn | 62.7200 | 82.16 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float32 | msprof | main | 1.7500 |
| llama-7b-ffn | 34.1000 | 75.56 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float16 | msprof | main | 1.7500 |
| llama-7b-ffn | 33.5800 | 76.73 | input_shape=[1024, 11008]; other_shape=[1024, 11008] | bfloat16 | msprof | main | 1.7500 |
| linear-bias | 24.4000 | 78.58 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float32 | msprof | main | 1.7500 |
| linear-bias | 13.0200 | 73.63 | input_shape=[1024, 4096]; other_shape=[1, 4096] | float16 | msprof | main | 1.7500 |
| linear-bias | 14.5000 | 66.12 | input_shape=[1024, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main | 1.7500 |
