# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-26T18:09:31.688936+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261026_180218_350993_Conv1dFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
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
| tilelang | 0.1.2+60cfa4e4e811e9deaa895ae56181d9bc5be488c4 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| Conv1dFwdOp | 100.0% (28/28) | 3.24e-02 | 16 | 1.52% – 2.92% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| whisper-conv1 | 58756.0000 | 2.91 | 11389.9893 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| whisper-conv1 | 58745.3789 | 2.92 | 11395.7207 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| wav2vec2-layer1 | 1067.3800 | 2.72 | 1197.5098 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| wav2vec2-layer1 | 1061.6000 | 2.80 | 1230.7134 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| encodec-init | 593.4400 | 1.52 | 776.1064 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| encodec-init | 588.1400 | 1.59 | 808.1064 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:memory caused |
| encodec-deep | 14575.9395 | 2.28 | 5058.1113 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| encodec-deep | 14604.2393 | 2.28 | 5063.9556 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_direct_main | 1.7500 | latency bound:compute caused |
| whisper-conv1-bias | 58710.5039 | 2.92 | 11395.8252 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| whisper-conv1-bias | 58927.2969 | 2.91 | 11411.1240 | ? | input_shape=[1, 80, 3000]; weight_shape=[1280, 80, 3] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| wav2vec2-layer1-bias | 1137.0200 | 2.71 | 1260.6469 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| wav2vec2-layer1-bias | 1135.1200 | 2.78 | 1293.7645 | ? | input_shape=[1, 1, 16000]; weight_shape=[512, 1, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| encodec-init-bias | 526.9400 | 1.87 | 837.8780 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| encodec-init-bias | 535.3200 | 1.90 | 869.7931 | ? | input_shape=[1, 1, 24000]; weight_shape=[32, 1, 7] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| encodec-deep-bias | 14846.6807 | 2.24 | 5064.8408 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | float16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
| encodec-deep-bias | 14770.9004 | 2.25 | 5061.3540 | ? | input_shape=[1, 128, 600]; weight_shape=[256, 128, 10] | bfloat16 | msprof | _conv1d_direct_bias_main | 1.7500 | latency bound:compute caused |
