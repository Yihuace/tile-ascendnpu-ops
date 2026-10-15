# TileOPs Evaluation Report: Conv3dBiasFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T14:09:07.017983+08:00 |
| evaluation_scope | Conv3dBiasFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_140650_934253_Conv3dBiasFwdOp |
| git_commit | 8d7dcc67252d5a42bfbf4d4f6f507eb7235d4803 |
| correctness_target | tests/ops/test_conv3d_bias.py |
| benchmark_target | benchmarks/ops/bench_conv3d_bias.py |

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
| 100.0% | 1 | 17 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| Conv3dBiasFwdOp | 100.0% (17/17) | N/A | 3 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv3dBiasFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| r3d-stem-bias | 6403.6001 | N/A | input_shape=[1, 3, 16, 112, 112]; weight_shape=[64, 3, 3, 3, 3] | float16 | msprof | _conv3d_bias_main | 0.0042 |
| video-down-bias | 85.0800 | N/A | input_shape=[1, 64, 8, 56, 56]; weight_shape=[128, 64, 3, 3, 3] | float16 | msprof | _conv3d_bias_main | 0.0524 |
| unet-encoder-bias | 6414.4600 | N/A | input_shape=[1, 32, 32, 64, 64]; weight_shape=[64, 32, 3, 3, 3] | bfloat16 | msprof | _conv3d_bias_main | 0.0039 |
