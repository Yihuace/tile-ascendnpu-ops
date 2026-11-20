# TileOPs Evaluation Report: FusedAddLayerNormFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T10:03:10.146351+08:00 |
| evaluation_scope | FusedAddLayerNormFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_095838_127621_FusedAddLayerNormFwdOp |
| git_commit | 3c4e9cf9a9b4fd41dcb7dfeafc45b8a8b70254f6 |
| correctness_target | tests/ops/test_fused_add_layer_norm.py |
| benchmark_target | benchmarks/ops/bench_fused_add_layer_norm.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+3c4e9cf9a9b4fd41dcb7dfeafc45b8a8b70254f6 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| FusedAddLayerNormFwdOp | 100.0% (14/14) | 3.29e-03 | 7 | 0.53% – 71.37% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FusedAddLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| llama-8b-prefill | 63.2600 | 32.80 | 2.8083 | ? | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-8b-prefill | 64.7400 | 37.68 | 3.3081 | ? | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-8b-decode | 4.7800 | 0.53 | 3.4695 | ? | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-70b-prefill | 117.5600 | 35.04 | 2.7927 | ? | {"m": 2048, "n": 8192} | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-70b-prefill | 121.2600 | 39.97 | 3.2924 | ? | {"m": 2048, "n": 8192} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-70b-decode | 4.6600 | 0.81 | 1.8721 | ? | {"m": 1, "n": 8192} | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| llama-13b-prefill | 67.2000 | 71.37 | 1.0183 | ? | {"m": 2048, "n": 5120} | float32 | msprof | main | 1.7500 | latency bound:pipeline caused |
