# TileOPs Evaluation Report: HardtanhFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T00:21:43.485491+08:00 |
| evaluation_scope | HardtanhFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_235308_236667_HardtanhFwdOp |
| git_commit | cf91d889ce344b4feb4a13af00e9aa25a8de4c4d |
| correctness_target | tests/ops/test_activation.py |
| benchmark_target | benchmarks/ops/bench_elementwise_manifest.py |

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
| 100.0% | 1 | 24 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardtanhFwdOp | 100.0% (12/12) | 0.00e+00 | 10 | 24.46% – 78.45% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardtanhFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 6.0600 | 39.55 | [1048576] | float32 | msprof | main | 1.7500 |
| smoke-1m | 4.0000 | 29.96 | [1048576] | float16 | msprof | main | 1.7500 |
| smoke-1m | 4.9000 | 24.46 | [1048576] | bfloat16 | msprof | main | 1.7500 |
| bounded-hidden | 13.5000 | 71.02 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| bounded-hidden | 14.7600 | 64.95 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| bounded-conv-feat | 19.5800 | 74.98 | [16, 256, 56, 56] | float16 | msprof | main | 1.7500 |
| bounded-conv-feat | 21.7000 | 67.65 | [16, 256, 56, 56] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 25.6000 | 74.90 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 26.5800 | 72.14 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 48.8800 | 78.45 | [4096, 4096] | float32 | msprof | main | 1.7500 |

## Warnings

- EluFwdOp/EluFwdOp-6d5215d9b069: missing tileops candidate record
- EluFwdOp/EluFwdOp-e742d1f472ff: missing tileops candidate record
- EluFwdOp/EluFwdOp-c76693ca7157: missing tileops candidate record
- EluFwdOp/EluFwdOp-54ce5ca74486: missing tileops candidate record
- EluFwdOp/EluFwdOp-97bebc3bc5ee: missing tileops candidate record
- EluFwdOp/EluFwdOp-0cc872f295a9: missing tileops candidate record
- EluFwdOp/EluFwdOp-295448b24fd1: missing tileops candidate record
- EluFwdOp/EluFwdOp-36a1cd4baa11: missing tileops candidate record
- EluFwdOp/EluFwdOp-8adf02b9b756: missing tileops candidate record
- EluFwdOp/EluFwdOp-927dc5963e40: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-f44cf9834ffd: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-6b48ec7231f6: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-21321cdbdc38: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-6be4b94245b1: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-18ffc010ed48: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-f2a247f21583: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-e2ab320b5e9e: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-ea54725d3904: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-c141b5df0376: missing tileops candidate record
- LeakyReluFwdOp/LeakyReluFwdOp-c4bb6382daf3: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-7d1c965936fa: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-a9b7f6419af5: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-43c1dbe05fdf: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-e3ed2a6baba0: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-44051ab42ccc: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-470e4ffc4d0c: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-517231bd39e4: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-4ebb2bbb7f7e: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-ff256d6f1282: missing tileops candidate record
- HardtanhFwdOp/HardtanhFwdOp-fda6d963b54c: missing tileops candidate record
