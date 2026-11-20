# TileOPs Evaluation Report: EluFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T13:45:25.298450+08:00 |
| evaluation_scope | EluFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_131559_786633_EluFwdOp |
| git_commit | c144c4b2aaefb9eeca259cd438fc4cfed69c77c6 |
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
| EluFwdOp | 100.0% (12/12) | 0.00e+00 | 10 | 27.34% – 77.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EluFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 5.9400 | 40.35 | [1048576] | float32 | msprof | main | 1.7500 |
| smoke-1m | 5.8800 | 27.34 | [1048576] | float16 | msprof | main | 1.7500 |
| smoke-1m | 6.0200 | 30.51 | [1048576] | bfloat16 | msprof | main | 1.7500 |
| mlp-hidden | 20.9800 | 60.78 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| mlp-hidden | 22.2200 | 65.58 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| mlp-hidden-wide | 38.9400 | 65.49 | [2048, 8192] | float16 | msprof | main | 1.7500 |
| mlp-hidden-wide | 40.8800 | 71.29 | [2048, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 38.8400 | 65.66 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 41.5200 | 70.19 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 49.7000 | 77.16 | [4096, 4096] | float32 | msprof | main | 1.7500 |

## Warnings

- EluFwdOp/EluFwdOp-52de7ddee008: missing tileops candidate record
- EluFwdOp/EluFwdOp-8257c4c71676: missing tileops candidate record
- EluFwdOp/EluFwdOp-6aee09873187: missing tileops candidate record
- EluFwdOp/EluFwdOp-4539a67085b5: missing tileops candidate record
- EluFwdOp/EluFwdOp-8f1ce858bc27: missing tileops candidate record
- EluFwdOp/EluFwdOp-baf3b23eb442: missing tileops candidate record
- EluFwdOp/EluFwdOp-cb3cf8a03031: missing tileops candidate record
- EluFwdOp/EluFwdOp-baa7a5419687: missing tileops candidate record
- EluFwdOp/EluFwdOp-1a0284c4d257: missing tileops candidate record
- EluFwdOp/EluFwdOp-3c8c30fb1d72: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-11cfc537faf1: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-e62625ea29ca: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-0ddaeb25de05: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-12ac28ef2bd5: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-2e68a13f47c5: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-62b3c0c262cf: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-1ab1485ea04b: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-f95388776a30: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-573b639a6ec2: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-f119d874c002: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-368675d8c1f6: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-a07ac2a7c012: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-2bede7568b0c: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-68ce06a9b5aa: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-f1dbb1e02cf0: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-9bf4c17235d7: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-8e6b81187c1d: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-920a6738b8e8: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-331d45b188fd: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-ff6121b1c7ec: missing tileops candidate record
