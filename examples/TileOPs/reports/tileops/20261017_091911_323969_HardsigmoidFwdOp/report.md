# TileOPs Evaluation Report: HardsigmoidFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T09:34:14.366953+08:00 |
| evaluation_scope | HardsigmoidFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_091911_323969_HardsigmoidFwdOp |
| git_commit | 7ca0bb7e33b332f00e67c1ee6e2daec64c096949 |
| correctness_target | tests/ops/test_hardsigmoid.py |
| benchmark_target | benchmarks/ops/bench_hardsigmoid.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2 (/home/developer/tilelang-mlir-ascend) |
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
| HardsigmoidFwdOp | 100.0% (7/7) | 1.19e-08 | 6 | 0.37% – 60.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardsigmoidFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| mbv3-se-gate | 2.4200 | 0.37 | 2.6098 | ? | [32, 240, 1, 1] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| mbv3-se-gate | 2.9201 | 0.37 | 3.1301 | ? | [32, 240, 1, 1] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| mbv3-se-gate-deep | 1.9800 | 1.81 | 2.6584 | ? | [32, 960, 1, 1] | float16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| mbv3-se-gate-deep | 2.2400 | 1.91 | 3.1884 | ? | [32, 960, 1, 1] | bfloat16 | msprof | main | 1.7500 | latency bound:pipeline caused |
| fc-wide | 17.4603 | 54.91 | 2.5003 | ? | [2048, 4096] | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| fc-wide | 18.1604 | 60.16 | 3.0003 | ? | [2048, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
