# TileOPs Evaluation Report: MoeGroupedGemmNopadFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T15:04:38.518519+08:00 |
| evaluation_scope | MoeGroupedGemmNopadFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_145659_919898_MoeGroupedGemmNopadFwdOp |
| git_commit | 0d7f496ec0ef9d3b75347e5dd7f702f40031cbd9 |
| correctness_target | tests/ops/test_moe_grouped_gemm_nopad.py |
| benchmark_target | benchmarks/ops/bench_moe_grouped_gemm_nopad.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.4+ubuntu.20.4.cann852 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 3 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| MoeGroupedGemmNopadFwdOp | 100.0% (3/3) | N/A | 4 | 10.14% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MoeGroupedGemmNopadFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| deepseek-v3-decode-gate-up | 28550.9378 | N/A | {"dtype_str": "bfloat16", "k": 7168, "n": 4096, "num_experts": 256, "numel": 4096} | bfloat16 | events | N/A | 0.5297 |
| deepseek-v3-prefill-gate-up | 67656.4125 | N/A | {"dtype_str": "bfloat16", "k": 7168, "n": 4096, "num_experts": 256, "numel": 32768} | bfloat16 | events | N/A | 0.2331 |
| deepseek-v3-decode-down | 14539.3479 | N/A | {"dtype_str": "bfloat16", "k": 2048, "n": 7168, "num_experts": 256, "numel": 4096} | bfloat16 | events | N/A | 0.5221 |
| deepseek-v3-prefill-down | 31932.6992 | 10.14 | {"dtype_str": "bfloat16", "k": 2048, "n": 7168, "num_experts": 256, "numel": 32768} | bfloat16 | msprof | main_mix_aic | 1.7500 |

## Warnings

- MoeGroupedGemmNopadFwdOp/MoeGroupedGemmNopadFwdOp-a3b449f13a09: requested msprof but recorded events
- MoeGroupedGemmNopadFwdOp/MoeGroupedGemmNopadFwdOp-cfc0a433d6d1: requested msprof but recorded events
- MoeGroupedGemmNopadFwdOp/MoeGroupedGemmNopadFwdOp-392fc6891e46: requested msprof but recorded events
- correctness passed, but the benchmark stage did not produce valid records
