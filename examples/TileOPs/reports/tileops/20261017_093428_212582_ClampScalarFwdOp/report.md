# TileOPs Evaluation Report: ClampScalarFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T09:56:39.175823+08:00 |
| evaluation_scope | ClampScalarFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_093428_212582_ClampScalarFwdOp |
| git_commit | 7ca0bb7e33b332f00e67c1ee6e2daec64c096949 |
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
| tilelang | 0.1.2 (/home/developer/tilelang-mlir-ascend) |
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
| ClampScalarFwdOp | 100.0% (11/11) | 0.00e+00 | 11 | 58.81% – 68.90% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampScalarFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| 16M | 32.6007 | 58.81 | 1.5001 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M | 30.9406 | 61.97 | 2.0001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M | 55.6611 | 68.90 | 0.5001 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
| 256M | 815.1363 | 68.20 | 0.8278 | ? | [16384, 16384] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 256M | 808.7762 | 68.74 | 1.1037 | ? | [16384, 16384] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 30.6806 | 62.50 | 1.0001 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M-min-only | 30.4806 | 62.91 | 1.5001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M-min-only | 56.7411 | 67.58 | 0.2501 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-max-only | 31.5406 | 60.79 | 1.0001 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M-max-only | 31.6406 | 60.60 | 1.5001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| 16M-max-only | 57.8012 | 66.34 | 0.2501 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
