# TileOPs Evaluation Report: GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-19T09:26:11.185038+00:00 |
| evaluation_scope | GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | events |
| run_id | 20261019_092548_565779_GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp |
| git_commit | 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b |
| correctness_target | tests/ops/test_gqa_fwd.py |
| benchmark_target | benchmarks/ops/bench_gqa_fwd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 1 |
| cpu | x86_64 |
| cann | 9.0.0 |
| driver | 26.0.rc1 |
| pytorch | 2.10.0+cpu |
| pytorch_npu | 2.10.0 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-113-generic-x86_64-with-glibc2.38 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 6 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp | 100.0% (6/6) | N/A | 0 | N/A |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| - | No performance records | - | - | - | - | - | - |
