# TileOPs Evaluation Report: GeluAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-17T10:39:50.255757+08:00 |
| evaluation_scope | GeluAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261117_102709_565655_GeluAndMulFwdOp |
| correctness_target | tests/ops/test_gelu_and_mul.py |
| benchmark_target | benchmarks/ops/bench_gelu_and_mul.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GeluAndMulFwdOp | 100.0% (14/14) | 1.95e-02 | 5 | 2.21% – 94.96% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluAndMulFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-ffn-prefill | 91.5000 | 94.96 | {"M": 2048, "N": 14336} | float16 | msprof | main | 1.7500 |
| llama-8b-ffn-prefill | 142.4600 | 68.49 | {"M": 2048, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| llama-8b-ffn-decode | 2.3600 | 2.21 | {"M": 1, "N": 14336} | bfloat16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 4.4400 | 31.86 | {"M": 32, "N": 12288} | float16 | msprof | main | 1.7500 |
| qwen3-8b-swiglu-b32 | 5.4200 | 25.03 | {"M": 32, "N": 12288} | bfloat16 | msprof | main | 1.7500 |
