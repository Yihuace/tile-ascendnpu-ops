# TileOPs Evaluation Report: HardsigmoidFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-12T22:29:21.180124+08:00 |
| evaluation_scope | HardsigmoidFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261012_222741_907268_HardsigmoidFwdOp |
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
| 28.6% | 1 | 7 | 5 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardsigmoidFwdOp | 28.6% (2/7) | N/A | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardsigmoidFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_hardsigmoid::test_hardsigmoid_op[1048576-dtype0] | failed | AttributeError: 'int' object has no attribute 'device' |
| Correctness | tests.ops.test_hardsigmoid::test_hardsigmoid_op[1048576-dtype1] | failed | AttributeError: 'int' object has no attribute 'device' |
| Correctness | tests.ops.test_hardsigmoid::test_hardsigmoid_op[1048576-dtype2] | failed | AttributeError: 'int' object has no attribute 'device' |
| Correctness | tests.ops.test_hardsigmoid::test_hardsigmoid_op[4000000-dtype3] | failed | AttributeError: 'int' object has no attribute 'device' |
| Correctness | tests.ops.test_hardsigmoid::test_hardsigmoid_op[4000000-dtype4] | failed | AttributeError: 'int' object has no attribute 'device' |
