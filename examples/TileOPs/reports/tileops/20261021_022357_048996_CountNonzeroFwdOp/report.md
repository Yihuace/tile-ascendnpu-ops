# TileOPs Evaluation Report: CountNonzeroFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T02:27:33.900766+00:00 |
| evaluation_scope | CountNonzeroFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_022357_048996_CountNonzeroFwdOp |
| git_commit | 19315978eaa21e1f926cd483d4e6f3236a33115b |
| correctness_target | tests/ops/test_count_nonzero.py |
| benchmark_target | benchmarks/ops/bench_count_nonzero.py |

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
| 100.0% | 1 | 37 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| CountNonzeroFwdOp | 100.0% (37/37) | 0.00e+00 | 4 | 15.77% – 60.22% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### CountNonzeroFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| sparsity-hidden | 15.9200 | 60.22 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| sparsity-hidden | 21.4800 | 44.63 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| sparsity-seq | 3.8000 | 15.77 | [32, 16384] | float16 | msprof | main | 1.7500 |
| 3d-multidim-reduce | 7.5200 | 31.87 | [4, 128, 4096] | float16 | msprof | main | 1.7500 |
