# TileOPs Evaluation Report: FusedAddLayerNormFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T12:00:13.357798+08:00 |
| evaluation_scope | FusedAddLayerNormFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_115534_389335_FusedAddLayerNormFwdOp |
| git_commit | 4daf47b8231b98b364e053f82921f42610d94a45 |
| correctness_target | tests/ops/test_fused_add_layer_norm.py |
| benchmark_target | benchmarks/ops/bench_fused_add_layer_norm.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FusedAddLayerNormFwdOp | 100.0% (14/14) | 3.29e-03 | 7 | 0.58% – 45.08% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FusedAddLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-8b-prefill | 81.9600 | 27.52 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.7500 |
| llama-8b-prefill | 83.5000 | 31.38 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.7500 |
| llama-8b-decode | 4.4000 | 0.58 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.7500 |
| llama-70b-prefill | 164.5800 | 29.40 | {"m": 2048, "n": 8192} | float16 | msprof | main | 1.7500 |
| llama-70b-prefill | 168.2600 | 33.08 | {"m": 2048, "n": 8192} | bfloat16 | msprof | main | 1.7500 |
| llama-70b-decode | 4.7600 | 0.79 | {"m": 1, "n": 8192} | bfloat16 | msprof | main | 1.7500 |
| llama-13b-prefill | 106.3800 | 45.08 | {"m": 2048, "n": 5120} | float32 | msprof | main | 1.7500 |
