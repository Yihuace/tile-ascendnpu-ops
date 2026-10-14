# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T17:55:52.425765+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_174100_695290_Conv1dFwdOp |
| git_commit | 25106f5215b45b18e663a01b823e1cb7809e6390 |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.91e-02 | 16 | 1.08% – 5.59% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| whisper-conv1 | 2.9000 | 1.33 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | N/A | 1.7500 |
| whisper-conv1 | 2.8800 | 4.78 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | N/A | 1.7500 |
| wav2vec2-layer1 | 592.9600 | 1.16 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | N/A | 1.7500 |
| wav2vec2-layer1 | 615.4200 | 3.93 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | N/A | 1.7500 |
| encodec-init | 3.1000 | 1.45 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | N/A | 1.7500 |
| encodec-init | 3.1600 | 5.41 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | N/A | 1.7500 |
| encodec-deep | 2.9600 | 1.31 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | N/A | 1.7500 |
| encodec-deep | 2.9400 | 4.99 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | N/A | 1.7500 |
| whisper-conv1-bias | 3.0200 | 1.21 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | N/A | 1.7500 |
| whisper-conv1-bias | 2.9600 | 4.65 | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | N/A | 1.7500 |
| wav2vec2-layer1-bias | 634.9200 | 1.08 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | N/A | 1.7500 |
| wav2vec2-layer1-bias | 628.5200 | 3.98 | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | N/A | 1.7500 |
| encodec-init-bias | 3.0600 | 1.47 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | N/A | 1.7500 |
| encodec-init-bias | 3.0600 | 5.59 | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | N/A | 1.7500 |
| encodec-deep-bias | 3.0000 | 1.30 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | N/A | 1.7500 |
| encodec-deep-bias | 2.9400 | 4.99 | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | N/A | 1.7500 |

## Warnings

- Conv1dFwdOp/Conv1dFwdOp-f5eeceafac0d: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-641990cb6e14: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-5029ac522ba4: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-4242f72164d2: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-c9aa57bb875b: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-b61fa207198d: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-59751a88257f: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-bafe9f39ea61: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-3696fca1b6c6: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-e5284771eae5: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-59c2812dbf96: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-ddd8f3a71b5d: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-ac8047fef70f: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-802571d618fd: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-70d6f0e84fa3: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- Conv1dFwdOp/Conv1dFwdOp-71d05debf21d: requested kernel filter 'main' produced no data; msprof retried without a kernel filter
- correctness passed, but the benchmark stage did not produce valid records
