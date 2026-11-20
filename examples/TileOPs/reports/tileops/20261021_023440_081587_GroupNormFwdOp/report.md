# TileOPs Evaluation Report: GroupNormFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T02:38:20.321069+00:00 |
| evaluation_scope | GroupNormFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_023440_081587_GroupNormFwdOp |
| git_commit | 19315978eaa21e1f926cd483d4e6f3236a33115b |
| correctness_target | tests/ops/test_group_norm.py |
| benchmark_target | benchmarks/ops/bench_group_norm.py |

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
| 90.5% | 1 | 21 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GroupNormFwdOp | 90.5% (19/21) | 9.20e-03 | 4 | 11.30% – 18.93% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GroupNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| image-g32 | 7.0600 | 16.98 | {"c": 128, "n": 8, "num_groups": 32, "spatial": [32, 32]} | float16 | msprof | main | 1.7500 |
| image-g32 | 7.1800 | 18.93 | {"c": 128, "n": 8, "num_groups": 32, "spatial": [32, 32]} | bfloat16 | msprof | main | 1.7500 |
| wider-channel-g32 | 6.0600 | 15.14 | {"c": 256, "n": 4, "num_groups": 32, "spatial": [28, 28]} | float16 | msprof | main | 1.7500 |
| tail-spatial-g16 | 4.6200 | 11.30 | {"c": 128, "n": 4, "num_groups": 16, "spatial": [30, 30]} | float16 | msprof | main | 1.7500 |
