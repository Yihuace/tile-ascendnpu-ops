# TileOPs Evaluation Report: AdaptiveMaxPool2dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T01:35:19.984651+00:00 |
| evaluation_scope | AdaptiveMaxPool2dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_012441_829945_AdaptiveMaxPool2dFwdOp |
| git_commit | 19315978eaa21e1f926cd483d4e6f3236a33115b |
| correctness_target | tests/ops/test_pool.py |
| benchmark_target | benchmarks/ops/bench_pool.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 26.1.1 |
| pytorch | 2.9.0 |
| pytorch_npu | 2.9.0.post8 |
| python | 3.10.12 |
| os | Linux-5.10.0-216.0.0.115.oe2203sp4.aarch64-aarch64-with-glibc2.35 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 75 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaptiveMaxPool2dFwdOp | 100.0% (44/44) | 0.00e+00 | 3 | 3.44% – 11.68% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaptiveMaxPool2dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| global-1x1 | 17.5000 | 3.44 | [8, 2048, 7, 7] | float16 | msprof | main | 1.7500 |
| spp-6x6 | 17.7600 | 11.68 | [2, 128, 56, 56] | float16 | msprof | main | 1.7500 |
| nondiv-7x7 | 15.7400 | 10.97 | [2, 64, 55, 57] | bfloat16 | msprof | main | 1.7500 |
