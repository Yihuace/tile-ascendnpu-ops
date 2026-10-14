# TileOPs Evaluation Report: IsfiniteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T11:13:40.913967+08:00 |
| evaluation_scope | IsfiniteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_105757_589000_IsfiniteFwdOp |
| git_commit | 96567b38810ce26ac11c78e7a1e219235ae19075 |
| correctness_target | tests/ops/test_isfinite.py |
| benchmark_target | benchmarks/ops/bench_isfinite.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| IsfiniteFwdOp | 100.0% (9/9) | 0.00e+00 | 5 | 40.78% – 72.06% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### IsfiniteFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16m | 37.8200 | 50.70 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16m | 58.9800 | 40.78 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16m | 53.2200 | 72.06 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 256m | 683.4800 | 61.73 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256m | 916.7800 | 46.00 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
