# TileOPs Evaluation Report: GeluTanhAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T22:57:00.289888+08:00 |
| evaluation_scope | GeluTanhAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_224427_350020_GeluTanhAndMulFwdOp |
| git_commit | f010d584f03451db13fd121f81a75d21f959415e |
| correctness_target | tests/ops/test_gelu_tanh_and_mul.py |
| benchmark_target | benchmarks/ops/bench_gelu_tanh_and_mul.py |

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
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GeluTanhAndMulFwdOp | 100.0% (11/11) | 3.26e-03 | 5 | 2.26% – 50.36% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluTanhAndMulFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-ffn-prefill | 200.1400 | 48.35 | {"M": 2048, "N": 14336} | float16 | msprof | main | 1.7500 |
| llama-8b-ffn-prefill | 204.8200 | 50.36 | {"M": 2048, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| llama-8b-ffn-decode | 2.2400 | 2.26 | {"M": 1, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 5.7000 | 22.76 | {"M": 32, "N": 12288} | float16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 5.8400 | 23.67 | {"M": 32, "N": 12288} | bfloat16 | msprof | main | 1.7500 |
