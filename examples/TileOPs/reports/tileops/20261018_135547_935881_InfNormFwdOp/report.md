# TileOPs Evaluation Report: InfNormFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T14:09:41.089883+00:00 |
| evaluation_scope | InfNormFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_135547_935881_InfNormFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_vector_norm.py |
| benchmark_target | benchmarks/ops/bench_vector_norm.py |

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
| 100.0% | 1 | 28 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| InfNormFwdOp | 100.0% (12/12) | 0.00e+00 | 4 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### InfNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-inf | 276.3180 | N/A | [2048, 4096] | float16 | events | N/A | 0.0607 |
| hidden-state-inf | 407.2620 | N/A | [2048, 4096] | bfloat16 | events | N/A | 0.0412 |
| long-row-inf | 252.1060 | N/A | [64, 8192] | bfloat16 | events | N/A | 0.0042 |
| 3d-multidim-reduce | 291.3300 | N/A | [4, 128, 512] | float16 | events | N/A | 0.0018 |

## Warnings

- InfNormFwdOp/InfNormFwdOp-d5050a3c4cca: requested msprof but recorded events
- InfNormFwdOp/InfNormFwdOp-e72fbf1d554a: requested msprof but recorded events
- InfNormFwdOp/InfNormFwdOp-075e534768f4: requested msprof but recorded events
- InfNormFwdOp/InfNormFwdOp-650ac584badf: requested msprof but recorded events
- correctness passed, but the benchmark stage did not produce valid records
