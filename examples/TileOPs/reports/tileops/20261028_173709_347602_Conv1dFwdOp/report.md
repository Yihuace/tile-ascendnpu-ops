# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-28T17:46:34.860917+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261028_173709_347602_Conv1dFwdOp |
| git_commit | fc068f4d5a754554f2ec2c3390d394ba0ea37b01 |
| correctness_target | tests/ops/test_conv1d.py |
| benchmark_target | benchmarks/ops/bench_conv1d.py |

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
| 100.0% | 1 | 29 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| Conv1dFwdOp | 100.0% (29/29) | 3.24e-02 | 16 | 8.64% – 27.45% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| whisper-conv1 | 6200.6846 | 27.11 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_direct_main | 1.7500 |
| whisper-conv1 | 6131.6890 | 27.45 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 |
| wav2vec2-layer1 | 336.8741 | 9.14 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_direct_main | 1.7500 |
| wav2vec2-layer1 | 365.5404 | 8.67 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 |
| encodec-init | 88.1794 | 8.64 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_direct_main | 1.7500 |
| encodec-init | 85.6989 | 9.29 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 |
| encodec-deep | 4144.6318 | 10.85 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_direct_main | 1.7500 |
| encodec-deep | 4133.5093 | 10.88 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 |
| whisper-conv1-bias | 6463.2217 | 26.04 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| whisper-conv1-bias | 6445.2383 | 26.15 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 334.6536 | 9.51 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 338.1544 | 9.68 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-init-bias | 83.7984 | 9.54 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-init-bias | 80.8778 | 10.30 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-deep-bias | 4147.1123 | 10.85 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-deep-bias | 4136.1899 | 10.88 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
