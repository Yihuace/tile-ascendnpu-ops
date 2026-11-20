# TileOPs Evaluation Report: ArgminFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T03:20:01.379294+00:00 |
| evaluation_scope | ArgminFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_031649_438396_ArgminFwdOp |
| git_commit | 19315978eaa21e1f926cd483d4e6f3236a33115b |
| correctness_target | tests/ops/test_argmin.py |
| benchmark_target | benchmarks/ops/bench_argmin.py |

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
| 100.0% | 1 | 42 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ArgminFwdOp | 100.0% (42/42) | 0.00e+00 | 6 | 4.93% – 35.25% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ArgminFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| lm-head-argmin | 8.6800 | 5.39 | [4, 102400] | float16 | msprof | argreduce_partial | 1.7500 |
| lm-head-argmin | 8.9400 | 5.72 | [4, 102400] | bfloat16 | msprof | argreduce_partial | 1.7500 |
| hidden-state-argmin | 27.2000 | 35.25 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| hidden-state-argmin | 31.4600 | 30.81 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| non-aligned-argmin | 3.0400 | 4.93 | [256, 300] | float16 | msprof | main | 1.7500 |
| non-aligned-argmin | 3.2600 | 6.31 | [256, 300] | bfloat16 | msprof | main | 1.7500 |
