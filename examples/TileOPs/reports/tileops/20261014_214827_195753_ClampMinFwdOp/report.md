# TileOPs Evaluation Report: ClampMinFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T22:02:32.870907+08:00 |
| evaluation_scope | ClampMinFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_214827_195753_ClampMinFwdOp |
| git_commit | f4e51304d80d91a029a76d06d7d8e80e7a90a4ab |
| correctness_target | tests/ops/test_clamp_min.py |
| benchmark_target | benchmarks/ops/bench_clamp_min.py |

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
| ClampMinFwdOp | 100.0% (14/14) | 0.00e+00 | 6 | 49.75% – 67.04% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampMinFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| 16M-min-only | 38.5400 | 49.75 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| 16M-min-only | 38.4800 | 49.83 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| 64M-min-only | 164.3200 | 58.51 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| 64M-min-only | 163.0000 | 58.17 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| 256M-min-only | 829.2600 | 67.04 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| 256M-min-only | 857.2800 | 64.84 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |

## Warnings

- correctness passed, but the benchmark stage did not produce valid records
- benchmark pytest reported 2 failed/error cases

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Benchmark | benchmarks.ops.bench_clamp_min::test_clamp_min_bench[smoke-1m-float32] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmp91g3fwmr/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp91g3fwmr/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp91g3fwmr/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp91g3fwmr/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp91g3fwmr/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp91g3fwmr/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp91g3fwmr/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp91g3fwmr/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp91g3fwmr/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp91g3fwmr/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
| Benchmark | benchmarks.ops.bench_clamp_min::test_clamp_min_bench[16M-min-only-float32] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmp9tedc32q/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp9tedc32q/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp9tedc32q/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp9tedc32q/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp9tedc32q/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp9tedc32q/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp9tedc32q/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp9tedc32q/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmp9tedc32q/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmp9tedc32q/kernel.npuir":2:3): error: ub overflow, requires 1843200 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
