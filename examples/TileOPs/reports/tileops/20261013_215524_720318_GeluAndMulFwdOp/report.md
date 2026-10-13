# TileOPs Evaluation Report: GeluAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T22:07:25.340818+08:00 |
| evaluation_scope | GeluAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_215524_720318_GeluAndMulFwdOp |
| git_commit | c9a7ce917d03d0853124a1746c9ebb84803f6153 |
| correctness_target | tests/ops/test_gelu_and_mul.py |
| benchmark_target | benchmarks/ops/bench_gelu_and_mul.py |

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
| GeluAndMulFwdOp | 100.0% (11/11) | 5.86e-03 | 5 | 4.77% – 43.99% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluAndMulFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-ffn-prefill | 878.7600 | 43.21 | {"M": 2048, "N": 14336} | float16 | msprof | main | 1.7500 |
| llama-8b-ffn-prefill | 877.6400 | 43.99 | {"M": 2048, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| llama-8b-ffn-decode | 3.9600 | 4.77 | {"M": 1, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 15.1200 | 33.64 | {"M": 32, "N": 12288} | float16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 14.9400 | 34.62 | {"M": 32, "N": 12288} | bfloat16 | msprof | main | 1.7500 |
