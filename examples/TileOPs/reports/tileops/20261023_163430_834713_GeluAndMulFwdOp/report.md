# TileOPs Evaluation Report: GeluAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T16:42:17.521362+08:00 |
| evaluation_scope | GeluAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_163430_834713_GeluAndMulFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
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
| tilelang | 0.1.2+60cfa4e4e811e9deaa895ae56181d9bc5be488c4 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| GeluAndMulFwdOp | 100.0% (14/14) | 1.47e-02 | 5 | 3.86% – 47.75% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluAndMulFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| llama-8b-ffn-prefill | 494.3800 | 46.88 | 8.7657 | ? | {"M": 2048, "N": 14336} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| llama-8b-ffn-prefill | 498.8000 | 47.75 | 8.7059 | ? | {"M": 2048, "N": 14336} | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
| llama-8b-ffn-decode | 3.1000 | 3.86 | 9.2918 | ? | {"M": 1, "N": 14336} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| qwen3-8b-swiglu-b32 | 9.9000 | 31.56 | 8.9504 | ? | {"M": 32, "N": 12288} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| qwen3-8b-swiglu-b32 | 9.6600 | 33.22 | 9.2004 | ? | {"M": 32, "N": 12288} | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
