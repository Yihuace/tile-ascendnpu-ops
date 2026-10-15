# TileOPs Evaluation Report: HardsigmoidFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T21:46:52.515511+08:00 |
| evaluation_scope | HardsigmoidFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_213240_149053_HardsigmoidFwdOp |
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
| HardsigmoidFwdOp | 100.0% (7/7) | 1.19e-08 | 6 | 0.36% – 65.42% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardsigmoidFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mbv3-se-gate | 2.4800 | 0.36 | [32, 240, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate | 2.3600 | 0.45 | [32, 240, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.0800 | 1.72 | [32, 960, 1, 1] | float16 | msprof | main | 1.7500 |
| mbv3-se-gate-deep | 2.1000 | 2.04 | [32, 960, 1, 1] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 15.9200 | 60.22 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 16.7000 | 65.42 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- correctness passed, but the benchmark stage did not produce valid records
- benchmark pytest reported 1 failed/error cases

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Benchmark | benchmarks.ops.bench_hardsigmoid::test_hardsigmoid_bench[smoke-1m-float32] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpgku5voak/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpgku5voak/kernel.npuir":2:3): error: ub overflow, requires 1835008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpgku5voak/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpgku5voak/kernel.npuir":2:3): error: ub overflow, requires 1835008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpgku5voak/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpgku5voak/kernel.npuir":2:3): error: ub overflow, requires 1835008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpgku5voak/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpgku5voak/kernel.npuir":2:3): error: ub overflow, requires 1835008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpgku5voak/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpgku5voak/kernel.npuir":2:3): error: ub overflow, requires 1835008 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
