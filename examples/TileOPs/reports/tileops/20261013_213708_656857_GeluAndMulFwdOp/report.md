# TileOPs Evaluation Report: GeluAndMulFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-13T21:38:55.740322+08:00 |
| evaluation_scope | GeluAndMulFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261013_213708_656857_GeluAndMulFwdOp |
| git_commit | c9a7ce917d03d0853124a1746c9ebb84803f6153 |
| correctness_target | tests/ops/test_gelu_and_mul.py |
| benchmark_target | benchmarks/ops/bench_gelu_and_mul.py |

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
| 90.9% | 1 | 11 | 1 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GeluAndMulFwdOp | 90.9% (10/11) | 1.87e-02 | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GeluAndMulFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Correctness | tests.ops.test_gelu_and_mul::test_gelu_and_mul_op[16-128-dtype0] | failed | AssertionError: Tensor-likes are not close!  Mismatched elements: 1154 / 2048 (56.3%) Greatest absolute difference: 0.0007715225219726562 at index (15, 113) (up to 1e-05 allowed) Greatest relative difference: 0.22231249511241913 at index (7, 117) (up to 1e-05 allowed) |
