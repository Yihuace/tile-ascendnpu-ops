# TileOPs Evaluation Report: LerpTensorFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-25T23:17:48.239676+08:00 |
| evaluation_scope | LerpTensorFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261025_230253_045981_LerpTensorFwdOp |
| git_commit | da8b5dc6f533d93c5dba6b51c1eeafbc293cbd10 |
| correctness_target | tests/ops/test_lerp_tensor.py |
| benchmark_target | benchmarks/ops/bench_lerp_tensor.py |

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
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| LerpTensorFwdOp | 100.0% (8/8) | 5.21e-08 | 7 | 66.48% – 80.47% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LerpTensorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 71.4800 | 80.47 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 74.4200 | 77.29 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 152.6400 | 80.37 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 378.1200 | 73.46 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 381.9800 | 72.87 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 1772.8000 | 67.58 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 1802.3800 | 66.48 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
