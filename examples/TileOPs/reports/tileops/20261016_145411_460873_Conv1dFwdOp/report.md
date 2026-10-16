# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T15:02:36.459036+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_145411_460873_Conv1dFwdOp |
| git_commit | 167c4dc013c24bd64bfe9d36fd36bf57ebefc4e1 |
| correctness_target | tests/ops/test_conv1d.py |
| benchmark_target | benchmarks/ops/bench_conv1d.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b (/home/developer/tilelang-mlir-ascend@417dae9) |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 28 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 1.02% – 14.24% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| whisper-conv1 | 43.6800 | 14.24 | 2132.3911 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1 | 43.6800 | 14.24 | 2133.0212 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 134.7400 | 1.31 | 1254.9081 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 131.7000 | 1.34 | 1273.2422 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 113.5400 | 1.02 | 924.3535 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 114.0200 | 1.02 | 955.2310 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 22.7400 | 5.74 | 542.4230 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 22.8800 | 5.71 | 541.4541 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 52.1000 | 11.97 | 2138.9683 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 53.3600 | 11.72 | 2152.7644 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 140.4600 | 1.26 | 1266.1898 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 135.9200 | 1.31 | 1269.2885 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 97.5200 | 1.20 | 944.1223 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 100.0400 | 1.18 | 959.1071 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 26.7600 | 4.87 | 545.7041 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 26.0400 | 5.00 | 544.9828 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
