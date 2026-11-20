# TileOPs Evaluation Report: BitwiseNotFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T16:09:35.315521+08:00 |
| evaluation_scope | BitwiseNotFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_153609_241917_BitwiseNotFwdOp |
| git_commit | a2357253662babf317ada194080078f928d155b3 |
| correctness_target | tests/ops/test_bitwise.py |
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
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BitwiseNotFwdOp | 100.0% (11/11) | 0.00e+00 | 6 | 43.11% – 75.93% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseNotFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 5.5600 | 43.11 | [1048576] | int32 | msprof | main | 1.7500 |
| smoke-1m | 8.2600 | 58.03 | [1048576] | int64 | msprof | main | 1.7500 |
| elementwise-16m | 51.0400 | 75.13 | [4096, 4096] | int32 | msprof | main | 1.7500 |
| elementwise-16m | 124.6000 | 75.93 | [4096, 4096] | int64 | msprof | main | 1.7500 |
| elementwise-64m | 334.0000 | 74.41 | [8192, 8192] | int32 | msprof | main | 1.7500 |
| elementwise-256m | 1591.6600 | 73.49 | [16384, 16384] | int32 | msprof | main | 1.7500 |

## Warnings

- BitwiseNotFwdOp/BitwiseNotFwdOp-96f551bea3b4: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-ee1298461afb: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-4142e521a038: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-6c24d7fc9855: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-0699102292f8: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-0df42eefd0ea: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-d3c80f2184db: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-bf5e9bf46db0: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-4bd7dbbf5eda: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-1602d692ac1c: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-84255480e21e: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-dfe4a6ee4528: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-479a4d25476e: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-9577b4f53f7b: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-7ef918611f2f: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-aff1a8467d36: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-2473758cb91e: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c1e9fd333cd1: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c1a471fb24ca: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-191ac0dcaa35: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-851cfe51200c: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-02f7925aa18c: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-adc415958990: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-9a9842609f1e: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-19e05f8954c2: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-0ade67ffdf65: missing tileops candidate record
