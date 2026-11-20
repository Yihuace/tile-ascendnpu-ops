# TileOPs Evaluation Report: ErfFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T19:00:42.218791+08:00 |
| evaluation_scope | ErfFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_182649_897387_ErfFwdOp |
| git_commit | a2357253662babf317ada194080078f928d155b3 |
| correctness_target | tests/ops/test_unary_math.py |
| benchmark_target | benchmarks/ops/bench_unary_elementwise.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ErfFwdOp | 100.0% (8/8) | 8.14e-04 | 10 | 40.66% – 68.53% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ErfFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 10.0400 | 40.98 | [1048576] | float32 | msprof | main | 1.7500 |
| smoke-1m | 10.7200 | 40.66 | [1048576] | float16 | msprof | main | 1.7500 |
| smoke-1m | 10.8400 | 42.32 | [1048576] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 107.4400 | 64.54 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 109.4400 | 66.69 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 102.0200 | 64.41 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 419.9000 | 66.03 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 427.9800 | 68.19 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 1670.4600 | 66.39 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 1703.3199 | 68.53 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- BitwiseNotFwdOp/BitwiseNotFwdOp-9bfed89d2b79: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-63c39395e622: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-c7e33273db5a: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-f86d5c70d383: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-0a8cef24e60c: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-78d97708945b: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-2b782df9f1d8: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-35112914837c: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-7156cebfb6e6: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-5e0dc4377b3d: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-80410955b266: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-62790e69d139: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-2e3527e90771: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-ea15ff59010e: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-af32aa3c35de: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-7b7a9f33ee5a: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-d2589b75089c: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-dfbee14899ad: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-3f6002712c87: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-ee39b6d3b663: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-eee2ba2c5d20: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-fcca72210e52: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c7288c4961c8: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c455aa1e1aa8: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-f76dab5dbe26: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-6e8d00dc72de: missing tileops candidate record
