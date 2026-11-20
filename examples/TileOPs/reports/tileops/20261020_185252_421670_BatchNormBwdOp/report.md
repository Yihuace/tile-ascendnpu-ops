# TileOPs Evaluation Report: BatchNormBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-20T18:56:05.184296+08:00 |
| evaluation_scope | BatchNormBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261020_185252_421670_BatchNormBwdOp |
| git_commit | 278f6a6b37ce19ddc49376b5eb8caf6e8a9414a0 |
| correctness_target | tests/ops/test_batch_norm_bwd.py |
| benchmark_target | benchmarks/ops/bench_batch_norm_bwd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+278f6a6b37ce19ddc49376b5eb8caf6e8a9414a0 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BatchNormBwdOp | 100.0% (12/12) | 2.39e-03 | 6 | 10.67% – 28.71% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BatchNormBwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| resnet-stage1 | 12.2200 | 11.81 | 2.7134 | ? | {"c": 64, "n": 8, "s": 1024} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| resnet-stage1 | 12.1400 | 12.82 | 2.9634 | ? | {"c": 64, "n": 8, "s": 1024} | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
| resnet-stage2 | 14.2600 | 10.67 | 2.8033 | ? | {"c": 128, "n": 4, "s": 1024} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| resnet-stage3 | 20.4400 | 11.83 | 3.6281 | ? | {"c": 256, "n": 4, "s": 784} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| resnet-stage3 | 19.7800 | 16.79 | 1.5245 | ? | {"c": 256, "n": 4, "s": 784} | float32 | msprof | main | 1.7500 | latency bound:compute caused |
| large-spatial | 1091.5200 | 28.71 | 2.3433 | ? | {"c": 128, "n": 4, "s": 262144} | float16 | msprof | stats | 1.7500 | latency bound:compute caused |
