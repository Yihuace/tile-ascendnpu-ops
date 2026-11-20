# TileOPs Evaluation Report: RopeLongRopeOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-25T23:57:05.685146+08:00 |
| evaluation_scope | RopeLongRopeOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261025_234635_085088_RopeLongRopeOp |
| git_commit | dc3684f1c2dc423d58cf39e7e669398470e9b31e |
| correctness_target | tests/ops/test_rope_longrope.py |
| benchmark_target | benchmarks/ops/bench_rope_longrope.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| RopeLongRopeOp | 100.0% (11/11) | 0.00e+00 | 3 | 15.29% – 54.74% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### RopeLongRopeOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| longrope-1d-8k-d128-bf16 | 6.3000 | 38.04 | [8192, 128] | bfloat16 | msprof | main | 1.7500 |
| longrope-2d-b1-s8k-h32-d128-fp16 | 81.5800 | 54.74 | [1, 8192, 32, 128] | float16 | msprof | main | 1.7500 |
| longrope-1d-2k-d128-fp16 | 3.9200 | 15.29 | [2048, 128] | float16 | msprof | main | 1.7500 |
