# TileOPs Evaluation Report: Conv1dBiasFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-27T09:52:57.404058+08:00 |
| evaluation_scope | Conv1dBiasFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261027_094816_102558_Conv1dBiasFwdOp |
| git_commit | e12640b0d4c561986800e7e2b64f7bb6c12e63bc |
| correctness_target | tests/ops/test_conv1d_bias.py |
| benchmark_target | benchmarks/ops/bench_conv1d_bias.py |

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
| 100.0% | 1 | 28 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| Conv1dBiasFwdOp | 100.0% (28/28) | 3.84e-02 | 8 | 1.82% – 2.90% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dBiasFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| whisper-conv1-bias | 59178.4805 | 2.90 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| whisper-conv1-bias | 59330.4766 | 2.89 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 1141.3711 | 2.70 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| wav2vec2-layer1-bias | 1164.7362 | 2.71 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-init-bias | 519.5343 | 1.90 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-init-bias | 559.8832 | 1.82 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-deep-bias | 14533.0381 | 2.29 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 |
| encodec-deep-bias | 14658.9443 | 2.27 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 |
