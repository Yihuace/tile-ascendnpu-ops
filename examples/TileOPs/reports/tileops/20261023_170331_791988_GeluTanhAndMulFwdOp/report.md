# TileOPs Evaluation Report: GeluTanhAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T17:11:11.633519+08:00 |
| evaluation_scope | GeluTanhAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_170331_791988_GeluTanhAndMulFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_gelu_tanh_and_mul.py |
| benchmark_target | benchmarks/ops/bench_gelu_tanh_and_mul.py |

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
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GeluTanhAndMulFwdOp | 100.0% (11/11) | 3.26e-03 | 5 | 2.38% – 47.67% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluTanhAndMulFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| llama-8b-ffn-prefill | 197.3200 | 45.62 | 3.4543 | ? | {"M": 2048, "N": 14336} | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| llama-8b-ffn-prefill | 202.2000 | 47.67 | 3.6362 | ? | {"M": 2048, "N": 14336} | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
| llama-8b-ffn-decode | 2.1200 | 2.38 | 3.8976 | ? | {"M": 1, "N": 14336} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| qwen3-8b-swiglu-b32 | 5.7000 | 21.50 | 3.5446 | ? | {"M": 32, "N": 12288} | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| qwen3-8b-swiglu-b32 | 5.7600 | 22.75 | 3.7946 | ? | {"M": 32, "N": 12288} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
