# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T14:37:28.039701+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_142945_037507_Conv1dFwdOp |
| git_commit | 405814d4b1e029ab505f242eb9020ca08f8e7552 |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 0.31% – 11.68% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| whisper-conv1 | 53.2200 | 11.68 | 2335.9360 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1 | 53.4600 | 11.63 | 2335.9360 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 617.1400 | 0.31 | 993.8912 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 605.2400 | 0.32 | 993.8912 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 262.8400 | 0.44 | 1187.5110 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 250.1200 | 0.46 | 1223.6926 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 22.9200 | 5.70 | 552.8105 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 20.0400 | 6.51 | 552.2062 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 67.4200 | 9.25 | 2350.9722 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 66.3000 | 9.43 | 2361.6748 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 619.0400 | 0.31 | 1002.2609 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 614.0800 | 0.32 | 1007.9410 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:pipeline caused |
| encodec-init-bias | 269.8200 | 0.43 | 1237.2916 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:pipeline caused |
| encodec-init-bias | 249.0200 | 0.47 | 1223.7848 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:pipeline caused |
| encodec-deep-bias | 27.1800 | 4.79 | 552.4281 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 25.5200 | 5.10 | 553.4985 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
