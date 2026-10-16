# TileOPs Evaluation Report: Conv1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T15:03:58.174646+08:00 |
| evaluation_scope | Conv1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_150257_246576_Conv1dFwdOp |
| git_commit | 167c4dc013c24bd64bfe9d36fd36bf57ebefc4e1 |
| correctness_target | tests/ops/test_conv1d.py |
| benchmark_target | benchmarks/ops/bench_conv1d.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b (/home/developer/tilelang-mlir-ascend@417dae9) |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 82.1% | 1 | 28 | 5 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| Conv1dFwdOp | 82.1% (23/28) | 4.21e-02 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Conv1dFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| - | No performance records | - | - | - | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_conv1d::test_conv1d[full-conformer-depthwise-k31-fp16] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpvkplrnao/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpvkplrnao/kernel.npuir":6:3): error: ub overflow, requires 1804288 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpvkplrnao/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpvkplrnao/kernel.npuir":6:3): error: ub overflow, requires 1804288 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpvkplrnao/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpvkplrnao/kernel.npuir":6:3): error: ub overflow, requires 1804288 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpvkplrnao/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpvkplrnao/kernel.npuir":6:3): error: ub overflow, requires 1804288 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpvkplrnao/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpvkplrnao/kernel.npuir":6:3): error: ub overflow, requires 1804288 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
| Correctness | tests.ops.test_conv1d::test_conv1d_bias_matches_torch | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpfbwcl1vf/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpfbwcl1vf/kernel.npuir":55:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpfbwcl1vf/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpfbwcl1vf/kernel.npuir":55:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpfbwcl1vf/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpfbwcl1vf/kernel.npuir":55:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpfbwcl1vf/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpfbwcl1vf/kernel.npuir":55:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpfbwcl1vf/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpfbwcl1vf/kernel.npuir":55:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
| Correctness | tests.ops.test_conv1d::test_conv1d_dilation_matches_torch[bias] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpzdtlely_/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpzdtlely_/kernel.npuir":51:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpzdtlely_/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpzdtlely_/kernel.npuir":51:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpzdtlely_/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpzdtlely_/kernel.npuir":51:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpzdtlely_/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpzdtlely_/kernel.npuir":51:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpzdtlely_/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpzdtlely_/kernel.npuir":51:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
| Correctness | tests.ops.test_conv1d::test_conv1d_same_padding_even_kernel_matches_torch[bias] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpmg19gkr7/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpmg19gkr7/kernel.npuir":48:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpmg19gkr7/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpmg19gkr7/kernel.npuir":48:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpmg19gkr7/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpmg19gkr7/kernel.npuir":48:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpmg19gkr7/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpmg19gkr7/kernel.npuir":48:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpmg19gkr7/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpmg19gkr7/kernel.npuir":48:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
| Correctness | tests.ops.test_conv1d::test_conv1d_grouped_matches_torch[bias] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpraqfd76h/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpraqfd76h/kernel.npuir":54:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpraqfd76h/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpraqfd76h/kernel.npuir":54:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpraqfd76h/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpraqfd76h/kernel.npuir":54:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpraqfd76h/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpraqfd76h/kernel.npuir":54:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) loc("/tmp/tmpraqfd76h/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline  loc("/tmp/tmpraqfd76h/kernel.npuir":54:3): error: ub overflow, requires 2116608 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.) [ERROR] Failed to run BiShengIR pipeline |
