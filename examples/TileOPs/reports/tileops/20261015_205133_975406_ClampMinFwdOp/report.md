# TileOPs Evaluation Report: ClampMinFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T21:09:34.593399+08:00 |
| evaluation_scope | ClampMinFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_205133_975406_ClampMinFwdOp |
| git_commit | 88d028d251378cbb6b366ed1fab8cfda70ad4938 |
| correctness_target | tests/ops/test_clamp_min.py |
| benchmark_target | benchmarks/ops/bench_clamp_min.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ClampMinFwdOp | 100.0% (14/14) | 0.00e+00 | 7 | 69.77% – 77.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampMinFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16M-min-only | 27.4800 | 69.77 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-min-only | 25.2400 | 75.97 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 16M-min-only | 49.7000 | 77.16 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| 64M-min-only | 123.0400 | 76.68 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| 64M-min-only | 126.7200 | 75.24 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| 256M-min-only | 756.6000 | 73.46 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256M-min-only | 759.0000 | 73.20 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
