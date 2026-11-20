# TileOPs Evaluation Report: Expm1FwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T06:13:03.681710+08:00 |
| evaluation_scope | Expm1FwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_053857_371455_Expm1FwdOp |
| git_commit | 180c5d20623b1f9171484e2f5bca0e489c978daf |
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
| Expm1FwdOp | 100.0% (8/8) | 2.04e-05 | 10 | 26.11% – 56.02% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Expm1FwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 9.9000 | 26.11 | [1048576] | float32 | msprof | main | 1.7500 |
| smoke-1m | 8.6200 | 32.70 | [1048576] | float16 | msprof | main | 1.7500 |
| smoke-1m | 8.4200 | 36.19 | [1048576] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 85.7800 | 51.21 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 86.1000 | 55.25 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 103.2000 | 39.02 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 339.3600 | 51.71 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 339.2400 | 56.02 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 1371.2600 | 51.17 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 1367.1801 | 55.59 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- BitwiseNotFwdOp/BitwiseNotFwdOp-a43dfd6b5fbd: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-1945bca55b38: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-bf681964d0a6: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-2971c7a0e00f: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-c4e08f781313: missing tileops candidate record
- BitwiseNotFwdOp/BitwiseNotFwdOp-50fa1077275a: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-787544db3a0e: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-c8752746c6c6: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-81537e29a889: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-b3634a54a051: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-65138467f079: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-fe85ea43e8fe: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-214b2b26ee7d: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-e102ebbce906: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-d75ac4a08140: missing tileops candidate record
- ErfFwdOp/ErfFwdOp-ba0bb7c2f6cd: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c059b0996834: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-a033f0328427: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-159a0d5765f3: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-e38b1ac9418d: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-250088b6620b: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-97b843e030e6: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-798932aa9de4: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-e3f78dff44af: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-9b2851e718e1: missing tileops candidate record
- Expm1FwdOp/Expm1FwdOp-c2f2925f6a9a: missing tileops candidate record
