# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T15:11:31.796198+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_150416_704663_Conv1dFwdOp |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 0.97% – 14.33% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| whisper-conv1 | 43.4000 | 14.33 | 2136.1770 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1 | 43.8400 | 14.18 | 2132.0762 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 130.6800 | 1.35 | 1263.4238 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1 | 128.4800 | 1.37 | 1256.4479 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 98.5800 | 1.17 | 949.4949 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-init | 106.3800 | 1.09 | 947.4868 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 22.2800 | 5.86 | 543.2007 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 22.2800 | 5.86 | 541.2607 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 54.2400 | 11.49 | 2143.7012 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| whisper-conv1-bias | 51.3200 | 12.18 | 2163.2952 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 137.1600 | 1.29 | 1273.9889 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| wav2vec2-layer1-bias | 144.2200 | 1.24 | 1282.1885 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 105.9400 | 1.10 | 960.7676 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-init-bias | 121.2200 | 0.97 | 966.8663 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 27.4000 | 4.75 | 540.5541 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
| encodec-deep-bias | 26.6000 | 4.90 | 544.2020 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_unit_stride_bias_main | 1.7500 | latency bound:memory caused |
