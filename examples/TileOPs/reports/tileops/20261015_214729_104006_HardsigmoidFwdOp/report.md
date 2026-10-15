# TileOPs Evaluation Report: HardsigmoidFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T22:03:05.370491+08:00 |
| evaluation_scope | HardsigmoidFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_214729_104006_HardsigmoidFwdOp |
| git_commit | f51ee3d5d46096ce27a2340a4a3b75b5315ea44b |
| correctness_target | tests/ops/test_hardsigmoid.py |
| benchmark_target | benchmarks/ops/bench_hardsigmoid.py |

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
| 100.0% | 1 | 7 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardsigmoidFwdOp | 100.0% (7/7) | 1.19e-08 | 6 | 0.36% – 64.64% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardsigmoidFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mbv3-se-gate | 2.4800 | 0.36 | [32, 240, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate | 2.6400 | 0.41 | [32, 240, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.0800 | 1.72 | [32, 960, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.0600 | 2.08 | [32, 960, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 17.2000 | 55.74 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 16.9000 | 64.64 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
