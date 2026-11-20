# TileOPs Evaluation Report: IsfiniteFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T17:35:42.281840+08:00 |
| evaluation_scope | IsfiniteFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_172545_424339_IsfiniteFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_isfinite.py |
| benchmark_target | benchmarks/ops/bench_isfinite.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+60cfa4e4e811e9deaa895ae56181d9bc5be488c4 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| IsfiniteFwdOp | 100.0% (9/9) | 0.00e+00 | 5 | 42.48% – 78.33% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### IsfiniteFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| 16m | 35.4200 | 54.13 | 2.0124 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16m | 45.1400 | 42.48 | 2.3961 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16m | 48.9600 | 78.33 | 0.8824 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 256m | 656.5800 | 64.23 | 1.4622 | ? | [16384, 16384] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 256m | 752.1800 | 56.08 | 1.7376 | ? | [16384, 16384] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
