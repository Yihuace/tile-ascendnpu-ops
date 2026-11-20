# TileOPs Evaluation Report: Conv3dBiasFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T18:44:12.654411+08:00 |
| evaluation_scope | Conv3dBiasFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_184106_697102_Conv3dBiasFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_conv3d_bias.py |
| benchmark_target | benchmarks/ops/bench_conv3d_bias.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+60cfa4e4e811e9deaa895ae56181d9bc5be488c4 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| Conv3dBiasFwdOp | 100.0% (17/17) | N/A | 3 | 1.64% – 20.43% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv3dBiasFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| r3d-stem-bias | 2649.6799 | 1.64 | 4907.0322 | ? | input_shape=[1, 3, 16, 112, 112]; weight_shape=[64, 3, 3, 3, 3] | float16 | msprof | _conv3d_bias_main_r2 | 1.7500 | latency bound:memory caused |
| video-down-bias | 87.6000 | 6.09 | 1049.7212 | ? | input_shape=[1, 64, 8, 56, 56]; weight_shape=[128, 64, 3, 3, 3] | float16 | msprof | _conv3d_bias_main_base | 1.7500 | latency bound:memory caused |
| unet-encoder-bias | 307.0200 | 20.43 | 1667.3169 | ? | input_shape=[1, 32, 32, 64, 64]; weight_shape=[64, 32, 3, 3, 3] | bfloat16 | msprof | _conv3d_bias_main_r3 | 1.7500 | latency bound:pipeline caused |
