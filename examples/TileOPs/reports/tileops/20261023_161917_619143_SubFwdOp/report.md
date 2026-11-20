# TileOPs Evaluation Report: SubFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T16:33:37.321725+08:00 |
| evaluation_scope | SubFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_161917_619143_SubFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_sub.py |
| benchmark_target | benchmarks/ops/bench_sub.py |

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
| 100.0% | 1 | 24 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| SubFwdOp | 100.0% (24/24) | 0.00e+00 | 15 | 64.95% – 84.13% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### SubFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| hidden-state-prefill | 47.8000 | 80.23 | 0.1251 | ? | input_shape=[2048, 4096]; other_shape=[2048, 4096] | float32 | msprof | main | 1.7500 | memory bound |
| hidden-state-prefill | 25.2200 | 76.03 | 1.0001 | ? | input_shape=[2048, 4096]; other_shape=[2048, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| cnn-feat-broadcast | 70.7600 | 82.99 | 0.1250 | ? | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float32 | msprof | main | 1.7500 | memory bound |
| cnn-feat-broadcast | 34.9000 | 84.13 | 0.2501 | ? | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | float16 | msprof | main | 1.7500 | memory bound |
| cnn-feat-broadcast | 37.2200 | 78.88 | 1.0001 | ? | input_shape=[16, 256, 56, 56]; other_shape=[256, 1, 1] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| llama-7b-ffn | 63.1800 | 81.56 | 0.1251 | ? | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float32 | msprof | main | 1.7500 | memory bound |
| llama-7b-ffn | 32.5800 | 79.08 | 0.2501 | ? | input_shape=[1024, 11008]; other_shape=[1024, 11008] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| llama-7b-ffn | 32.5200 | 79.23 | 1.0001 | ? | input_shape=[1024, 11008]; other_shape=[1024, 11008] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| linear-bias | 24.7200 | 77.56 | 0.1252 | ? | input_shape=[1024, 4096]; other_shape=[1, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
| linear-bias | 13.2000 | 72.63 | 0.2503 | ? | input_shape=[1024, 4096]; other_shape=[1, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| linear-bias | 14.7600 | 64.95 | 1.0003 | ? | input_shape=[1024, 4096]; other_shape=[1, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| llama-7b-hidden | 13.6200 | 70.39 | 0.2503 | ? | input_shape=[1024, 4096]; other_shape=[1024, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| phi-2-ffn | 31.7000 | 75.61 | 0.2501 | ? | input_shape=[1024, 10240]; other_shape=[1024, 10240] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| phi-2-ffn-bias | 29.7400 | 80.59 | 0.2501 | ? | input_shape=[1024, 10240]; other_shape=[1, 10240] | float16 | msprof | main | 1.7500 | memory bound |
| llama-7b-ffn-bias | 33.7600 | 76.32 | 0.2501 | ? | input_shape=[1024, 11008]; other_shape=[1, 11008] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
