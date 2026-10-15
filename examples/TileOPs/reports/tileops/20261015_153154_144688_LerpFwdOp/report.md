# TileOPs Evaluation Report: LerpFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T15:50:27.313687+08:00 |
| evaluation_scope | LerpFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_153154_144688_LerpFwdOp |
| git_commit | c258aa41e8c40ff6a6e28d3a642ead0f7218fdf7 |
| correctness_target | tests/ops/test_lerp_scalar.py |
| benchmark_target | benchmarks/ops/bench_lerp_scalar.py |

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
| 100.0% | 1 | 36 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| LerpFwdOp | 100.0% (36/36) | 1.21e-03 | 11 | 61.93% – 80.97% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LerpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-prefill | 47.3600 | 80.97 | end_shape=[2048, 4096]; input_shape=[2048, 4096] | float32 | msprof | main | 1.7500 |
| hidden-state-prefill | 25.9000 | 74.03 | end_shape=[2048, 4096]; input_shape=[2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 72.6400 | 80.84 | end_shape=[256, 1, 1]; input_shape=[16, 256, 56, 56] | float32 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 38.5000 | 76.26 | end_shape=[256, 1, 1]; input_shape=[16, 256, 56, 56] | float16 | msprof | main | 1.7500 |
| cnn-feat-broadcast | 43.2600 | 67.87 | end_shape=[256, 1, 1]; input_shape=[16, 256, 56, 56] | bfloat16 | msprof | main | 1.7500 |
| llama-7b-ffn | 63.9000 | 80.64 | end_shape=[1024, 11008]; input_shape=[1024, 11008] | float32 | msprof | main | 1.7500 |
| llama-7b-ffn | 33.6400 | 76.59 | end_shape=[1024, 11008]; input_shape=[1024, 11008] | float16 | msprof | main | 1.7500 |
| llama-7b-ffn | 33.8200 | 76.18 | end_shape=[1024, 11008]; input_shape=[1024, 11008] | bfloat16 | msprof | main | 1.7500 |
| linear-bias | 25.9200 | 73.97 | end_shape=[1, 4096]; input_shape=[1024, 4096] | float32 | msprof | main | 1.7500 |
| linear-bias | 14.1800 | 67.61 | end_shape=[1, 4096]; input_shape=[1024, 4096] | float16 | msprof | main | 1.7500 |
| linear-bias | 15.4800 | 61.93 | end_shape=[1, 4096]; input_shape=[1024, 4096] | bfloat16 | msprof | main | 1.7500 |
