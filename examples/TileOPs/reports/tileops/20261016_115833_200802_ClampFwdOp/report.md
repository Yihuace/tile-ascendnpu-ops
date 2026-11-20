# TileOPs Evaluation Report: ClampFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T12:22:14.985857+08:00 |
| evaluation_scope | ClampFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_115833_200802_ClampFwdOp |
| git_commit | f8e4950c2a578892265fabd277c0cc9705296b27 |
| correctness_target | tests/ops/test_special_elementwise.py |
| benchmark_target | benchmarks/ops/bench_independent_elementwise.py |

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
| 100.0% | 1 | 15 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ClampFwdOp | 100.0% (15/15) | 0.00e+00 | 10 | 51.80% – 83.85% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 10.4000 | 69.14 | input_shape=[1024, 1024]; max_shape=[1024, 1024]; min_shape=[1024, 1024] | float32 | msprof | main | 1.7500 |
| smoke-1m | 6.5600 | 54.80 | input_shape=[1024, 1024]; max_shape=[1024, 1024]; min_shape=[1024, 1024] | float16 | msprof | main | 1.7500 |
| smoke-1m | 6.9400 | 51.80 | input_shape=[1024, 1024]; max_shape=[1024, 1024]; min_shape=[1024, 1024] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 68.6000 | 83.85 | input_shape=[4096, 4096]; max_shape=[4096, 4096]; min_shape=[4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 72.9800 | 78.82 | input_shape=[4096, 4096]; max_shape=[4096, 4096]; min_shape=[4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 156.7200 | 79.55 | input_shape=[4096, 4096]; max_shape=[4096, 4096]; min_shape=[4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 367.1400 | 75.38 | input_shape=[8192, 8192]; max_shape=[8192, 8192]; min_shape=[8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 377.0400 | 73.59 | input_shape=[8192, 8192]; max_shape=[8192, 8192]; min_shape=[8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 1659.4600 | 72.18 | input_shape=[16384, 16384]; max_shape=[16384, 16384]; min_shape=[16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 1671.7400 | 71.65 | input_shape=[16384, 16384]; max_shape=[16384, 16384]; min_shape=[16384, 16384] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- AlibiFwdOp/AlibiFwdOp-06b6aa7f5bac: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-d7e12bbb50aa: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-72adcff99f7e: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-21b230887791: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-3800428ecbce: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-1255f123aa86: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-b0ea5273f702: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-58e182f00846: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-91d669d759e4: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-a39f3cf339d4: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-a109a398eaf9: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-4cb35ff72750: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-cfb7daf46e52: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-3602c4f14cbf: missing tileops candidate record
