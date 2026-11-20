# TileOPs Evaluation Report: CumprodFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T10:34:23.237595+08:00 |
| evaluation_scope | CumprodFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_102951_600995_CumprodFwdOp |
| git_commit | 147b71bcf2f0d4b613eaf815fc0cd45dc22b4048 |
| correctness_target | tests/ops/test_cumulative.py |
| benchmark_target | benchmarks/ops/bench_cumulative.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910_9362 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.5 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2220_156.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 20 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| CumprodFwdOp | 100.0% (20/20) | 1.71e-03 | 5 | 6.21% – 13.25% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### CumprodFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-scan | 185.1363 | 12.24 | 10.38 | 12.24 | 10.07 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-scan | 185.9563 | 13.25 | 10.34 | 13.25 | 10.02 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| long-seq-scan | 73.1785 | 9.96 | 6.47 | 9.96 | 6.37 | [64, 32768] | bfloat16 | msprof | main | 1.8000 |
| non-aligned-cumprod | 9.6398 | 6.21 | 4.44 | 6.21 | 3.54 | [512, 300] | float16 | msprof | main | 1.8000 |
| non-aligned-cumprod | 10.3998 | 6.99 | 4.65 | 6.99 | 3.28 | [512, 300] | bfloat16 | msprof | main | 1.8000 |

## Warnings

- CumprodFwdOp/CumprodFwdOp-d5050a3c4cca: invalid tileops profile: unfiltered capture contains 2 distinct ops
- CumprodFwdOp/CumprodFwdOp-e72fbf1d554a: invalid tileops profile: unfiltered capture contains 2 distinct ops
- CumprodFwdOp/CumprodFwdOp-5524ce71b00c: invalid tileops profile: unfiltered capture contains 2 distinct ops
- CumprodFwdOp/CumprodFwdOp-b56215dc606f: invalid tileops profile: unfiltered capture contains 2 distinct ops
- CumprodFwdOp/CumprodFwdOp-5109eaf34008: invalid tileops profile: unfiltered capture contains 2 distinct ops
- correctness passed, but the benchmark stage did not produce valid records
