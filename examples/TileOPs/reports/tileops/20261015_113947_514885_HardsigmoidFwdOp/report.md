# TileOPs Evaluation Report: HardsigmoidFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T11:53:52.973680+08:00 |
| evaluation_scope | HardsigmoidFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_113947_514885_HardsigmoidFwdOp |
| git_commit | 8d7dcc67252d5a42bfbf4d4f6f507eb7235d4803 |
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
| HardsigmoidFwdOp | 100.0% (7/7) | 1.19e-08 | 6 | 0.50% – 56.98% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardsigmoidFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mbv3-se-gate | 2.8600 | 0.50 | [32, 240, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate | 2.7600 | 0.58 | [32, 240, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.4200 | 2.37 | [32, 960, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.6800 | 2.40 | [32, 960, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 28.9800 | 50.50 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 28.8800 | 56.98 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- correctness passed, but the benchmark stage did not produce valid records
- benchmark pytest reported 1 failed/error cases

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Benchmark | benchmarks.ops.bench_hardsigmoid::test_hardsigmoid_bench[smoke-1m-float32] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpigd3bn__/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpigd3bn__/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpigd3bn__/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpigd3bn__/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpigd3bn__/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpigd3bn__/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpigd3bn__/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpigd3bn__/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpigd3bn__/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpigd3bn__/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
