# TileOPs Evaluation Report: BatchNormBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T16:04:18.371885+08:00 |
| evaluation_scope | BatchNormBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_155843_127784_BatchNormBwdOp |
| git_commit | c258aa41e8c40ff6a6e28d3a642ead0f7218fdf7 |
| correctness_target | tests/ops/test_batch_norm_bwd.py |
| benchmark_target | benchmarks/ops/bench_batch_norm_bwd.py |

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
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BatchNormBwdOp | 100.0% (12/12) | 2.29e-03 | 6 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BatchNormBwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| resnet-stage1 | 19.9800 | N/A | {"c": 64, "n": 8, "s": 1024} | float16 | msprof | main | 0.1575 |
| resnet-stage1 | 21.1200 | N/A | {"c": 64, "n": 8, "s": 1024} | bfloat16 | msprof | main | 0.1490 |
| resnet-stage2 | 24.3600 | N/A | {"c": 128, "n": 4, "s": 1024} | float16 | msprof | main | 0.1292 |
| resnet-stage3 | 41.0400 | N/A | {"c": 256, "n": 4, "s": 784} | float16 | msprof | main | 0.1174 |
| resnet-stage3 | 40.8400 | N/A | {"c": 256, "n": 4, "s": 784} | float32 | msprof | main | 0.2360 |
| large-spatial | 2092.8000 | N/A | {"c": 128, "n": 4, "s": 262144} | float16 | msprof | stats | 0.3848 |
