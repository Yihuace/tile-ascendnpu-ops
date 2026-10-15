# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T16:32:42.308530+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_162520_967795_Conv1dFwdOp |
| git_commit | c258aa41e8c40ff6a6e28d3a642ead0f7218fdf7 |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 0.07% – 0.46% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| whisper-conv1 | 5457.2002 | 0.12 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 |
| whisper-conv1 | 5523.8003 | 0.11 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 |
| wav2vec2-layer1 | 618.5600 | 0.31 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_main | 1.7500 |
| wav2vec2-layer1 | 606.9800 | 0.32 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_main | 1.7500 |
| encodec-init | 283.6200 | 0.41 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_main | 1.7500 |
| encodec-init | 251.6600 | 0.46 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_main | 1.7500 |
| encodec-deep | 1848.4800 | 0.07 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 |
| encodec-deep | 1891.1600 | 0.07 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 |
| whisper-conv1-bias | 5637.1201 | 0.11 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 |
| whisper-conv1-bias | 5925.2197 | 0.11 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 609.1400 | 0.32 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 609.6600 | 0.32 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 |
| encodec-init-bias | 287.9000 | 0.41 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_bias_main | 1.7500 |
| encodec-init-bias | 255.2000 | 0.46 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 |
| encodec-deep-bias | 1852.4399 | 0.07 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 |
| encodec-deep-bias | 1760.6400 | 0.08 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 |
