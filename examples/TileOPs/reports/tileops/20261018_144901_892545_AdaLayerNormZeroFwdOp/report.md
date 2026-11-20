# TileOPs Evaluation Report: AdaLayerNormZeroFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T15:00:06.581525+00:00 |
| evaluation_scope | AdaLayerNormZeroFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_144901_892545_AdaLayerNormZeroFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_ada_layer_norm_zero.py |
| benchmark_target | benchmarks/ops/bench_ada_layer_norm.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 1 |
| cpu | x86_64 |
| cann | 9.0.0 |
| driver | 26.0.rc1 |
| pytorch | 2.10.0+cpu |
| pytorch_npu | 2.10.0 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-113-generic-x86_64-with-glibc2.38 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaLayerNormZeroFwdOp | 100.0% (14/14) | 1.39e-02 | 5 | 0.55% – 69.97% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaLayerNormZeroFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| dit-xl-2 | 11.1404 | 46.69 | {"m": 1024, "n": 1152} | float16 | msprof | main | 1.8000 |
| dit-xl-2 | 10.8404 | 47.98 | {"m": 1024, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 54.1022 | 68.91 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 53.2821 | 69.97 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-decode | 3.3601 | 0.55 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
