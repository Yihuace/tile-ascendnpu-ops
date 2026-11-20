# TileOPs Evaluation Report: GemmW4A16Op

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T11:09:09.908937+08:00 |
| evaluation_scope | GemmW4A16Op |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_110340_042869_GemmW4A16Op |
| git_commit | 6c987c0a520c559f830962a4cf081afe16fc7326 |
| correctness_target | tests/ops/test_gemm_w4a16.py |
| benchmark_target | benchmarks/ops/bench_gemm_w4a16.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.4+ubuntu.20.4.cann852 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 6 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GemmW4A16Op | 100.0% (6/6) | 8.78e-03 | 4 | 0.72% – 0.75% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GemmW4A16Op

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| decode-l2-resident-ish | 4141.4399 | 0.72 | 0.12 | 0.72 | 0.49 | {"group_size": 128, "k": 8192, "m": 1, "n": 8192} | float16 | msprof | main | 1.7500 |
| decode-hbm-streaming-threshold | 8073.4004 | 0.74 | 0.13 | 0.74 | 0.50 | {"group_size": 128, "k": 16384, "m": 1, "n": 8192} | float16 | msprof | main | 1.7500 |
| decode-non-power2-low-cta | 8649.4199 | 0.75 | 0.13 | 0.75 | 0.51 | {"group_size": 128, "k": 20480, "m": 1, "n": 7168} | float16 | msprof | main | 1.7500 |
| decode-long-k-pressure | 39821.8203 | 0.75 | 0.13 | 0.75 | 0.50 | {"group_size": 128, "k": 81920, "m": 1, "n": 8192} | float16 | msprof | main | 1.7500 |
