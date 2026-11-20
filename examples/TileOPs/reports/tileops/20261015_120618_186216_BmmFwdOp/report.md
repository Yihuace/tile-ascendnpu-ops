# TileOPs Evaluation Report: BmmFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T12:11:23.743138+08:00 |
| evaluation_scope | BmmFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_120618_186216_BmmFwdOp |
| git_commit | b192f3d5b0b92c2b46b3cd95037ed9dd4c005cd8 |
| correctness_target | tests/ops/test_bmm.py |
| benchmark_target | benchmarks/ops/bench_bmm.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.4+ubuntu.20.4.cann852 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 17 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| BmmFwdOp | 100.0% (17/17) | 7.10e-03 | 5 | 43.16% – 83.17% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BmmFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| square-b8-1k | 102.7600 | 56.69 | {"batch": 8, "dtype_str": "float16", "k": 1024, "m": 1024, "n": 1024} | float16 | msprof | _bmm_main | 1.7500 |
| square-b8-1k | 102.7000 | 56.72 | {"batch": 8, "dtype_str": "bfloat16", "k": 1024, "m": 1024, "n": 1024} | bfloat16 | msprof | _bmm_main | 1.7500 |
| square-b16-512 | 33.7400 | 43.16 | {"batch": 16, "dtype_str": "float16", "k": 512, "m": 512, "n": 512} | float16 | msprof | _bmm_main | 1.7500 |
| square-b16-512 | 33.2600 | 43.79 | {"batch": 16, "dtype_str": "bfloat16", "k": 512, "m": 512, "n": 512} | bfloat16 | msprof | _bmm_main | 1.7500 |
| square-b4-4k | 2241.3799 | 83.17 | {"batch": 4, "dtype_str": "bfloat16", "k": 4096, "m": 4096, "n": 4096} | bfloat16 | msprof | _bmm_main | 1.7500 |
