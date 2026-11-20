# TileOPs Evaluation Report: LeakyReluFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T17:39:20.733276+08:00 |
| evaluation_scope | LeakyReluFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_171037_790791_LeakyReluFwdOp |
| git_commit | d5877ec1a4b53e46f58b86e1234a46c81e4e8b4e |
| correctness_target | tests/ops/test_activation.py |
| benchmark_target | benchmarks/ops/bench_elementwise_manifest.py |

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
| 100.0% | 1 | 24 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| LeakyReluFwdOp | 100.0% (12/12) | 0.00e+00 | 10 | 24.56% – 81.07% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LeakyReluFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 5.9600 | 40.22 | [1048576] | float32 | msprof | main | 1.7500 |
| smoke-1m | 4.7000 | 25.50 | [1048576] | float16 | msprof | main | 1.7500 |
| smoke-1m | 4.8800 | 24.56 | [1048576] | bfloat16 | msprof | main | 1.7500 |
| gan-feat | 29.3800 | 65.26 | [16, 256, 64, 64] | float16 | msprof | main | 1.7500 |
| gan-feat | 27.6400 | 69.37 | [16, 256, 64, 64] | bfloat16 | msprof | main | 1.7500 |
| gan-feat-deep | 15.9000 | 60.30 | [16, 512, 32, 32] | float16 | msprof | main | 1.7500 |
| gan-feat-deep | 15.0400 | 63.74 | [16, 512, 32, 32] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 27.3600 | 70.08 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 27.4200 | 69.93 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 47.3000 | 81.07 | [4096, 4096] | float32 | msprof | main | 1.7500 |

## Warnings

- EluFwdOp/EluFwdOp-56e20096e3a6: missing tileops candidate record
- EluFwdOp/EluFwdOp-6b2fe425a545: missing tileops candidate record
- EluFwdOp/EluFwdOp-1311a885a673: missing tileops candidate record
- EluFwdOp/EluFwdOp-ba2c0250501d: missing tileops candidate record
- EluFwdOp/EluFwdOp-80ce4e7e2007: missing tileops candidate record
- EluFwdOp/EluFwdOp-bb3ca4c1f9f8: missing tileops candidate record
- EluFwdOp/EluFwdOp-c9f36d02e036: missing tileops candidate record
- EluFwdOp/EluFwdOp-6206b85729f4: missing tileops candidate record
- EluFwdOp/EluFwdOp-14ebb0c26ee5: missing tileops candidate record
- EluFwdOp/EluFwdOp-301e1bd4d0d7: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-848fa5de93f3: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-5354e770054f: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-044918fca260: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-23449d445d6e: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-9027cece7fdd: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-c69579ddda07: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-6f5011bf776c: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-e0d0b8c71f3d: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-b76faef45eb7: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-49abf903bbc9: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-5f94f407ee4e: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-27f03f64b9e3: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-2ea8c7d89eeb: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-9f3b387d7f57: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-41c70939584f: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-f8e00ef361f5: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-67a38adc6305: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-b824517e81d3: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-00d84572c449: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-833e03f709c5: missing tileops candidate record
