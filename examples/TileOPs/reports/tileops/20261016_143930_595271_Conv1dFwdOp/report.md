# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T14:47:03.908244+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_143930_595271_Conv1dFwdOp |
| git_commit | ad6e103d477ac68ea14060c29f4a80b2e0c3851f |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 0.63% – 11.65% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| whisper-conv1 | 53.3800 | 11.65 | 2335.9360 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1 | 54.4800 | 11.41 | 2335.9360 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 217.7000 | 0.81 | 1827.7644 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 222.7800 | 0.79 | 1795.7124 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 175.4200 | 0.66 | 1306.7671 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 183.8800 | 0.63 | 1291.0759 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 21.0800 | 6.19 | 553.0122 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 21.4600 | 6.08 | 552.5082 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 67.1000 | 9.29 | 2349.4519 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 67.2000 | 9.31 | 2361.6748 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 213.7200 | 0.83 | 1870.5247 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 221.6600 | 0.80 | 1822.7463 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 184.5200 | 0.63 | 1328.1423 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 181.5600 | 0.65 | 1353.7233 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 24.9000 | 5.23 | 552.9313 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 25.9800 | 5.01 | 553.2969 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
