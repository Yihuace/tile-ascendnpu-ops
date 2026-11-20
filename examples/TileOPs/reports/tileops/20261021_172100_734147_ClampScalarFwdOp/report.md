# TileOPs Evaluation Report: ClampScalarFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T17:34:08.259223+08:00 |
| evaluation_scope | ClampScalarFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_172100_734147_ClampScalarFwdOp |
| git_commit | c378b4818bab3eca433877b27cfd6454549f7857 |
| correctness_target | tests/ops/test_clamp_scalar.py |
| benchmark_target | benchmarks/ops/bench_clamp_scalar.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+c378b4818bab3eca433877b27cfd6454549f7857 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| ClampScalarFwdOp | 100.0% (11/11) | 0.00e+00 | 11 | 69.62% – 80.53% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampScalarFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| 16M | 25.0400 | 76.57 | 1.0001 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M | 27.5400 | 69.62 | 2.0001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M | 47.6200 | 80.53 | 0.5001 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | memory bound |
| 256M | 764.1800 | 72.73 | 0.5520 | ? | [16384, 16384] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 256M | 764.2200 | 72.72 | 1.1040 | ? | [16384, 16384] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 24.4400 | 78.45 | 0.5002 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 25.5600 | 75.02 | 1.5001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 50.2800 | 76.27 | 0.2501 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-max-only | 26.2400 | 73.07 | 0.5002 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-max-only | 25.6600 | 74.72 | 1.5001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-max-only | 47.8600 | 80.13 | 0.2501 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | memory bound |
