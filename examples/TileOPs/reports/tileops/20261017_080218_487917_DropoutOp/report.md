# TileOPs Evaluation Report: DropoutOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T08:16:23.675656+08:00 |
| evaluation_scope | DropoutOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_080218_487917_DropoutOp |
| git_commit | 90fb7ae37f0be5e908d6317eb2b1e09aeb1ddc4b |
| correctness_target | tests/ops/test_dropout.py |
| benchmark_target | benchmarks/ops/bench_dropout.py |

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
| 100.0% | 1 | 28 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| DropoutOp | 100.0% (28/28) | N/A | 6 | 26.35% – 49.88% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### DropoutOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-dropout-1m | 16.1800 | 26.35 | [1024, 1024] | float32 | msprof | main | 1.7500 |
| smoke-dropout-1m | 11.0400 | 36.08 | [1024, 1024] | float16 | msprof | main | 1.7500 |
| smoke-dropout-1m | 11.9000 | 39.21 | [1024, 1024] | bfloat16 | msprof | main | 1.7500 |
| tokens-1k-hidden-4k-fp16 | 35.8800 | 44.04 | [1024, 4096] | float16 | msprof | main | 1.7500 |
| tokens-1k-hidden-10k-bf16 | 92.7600 | 49.88 | [1024, 10240] | bfloat16 | msprof | main | 1.7500 |
| tokens-1k-hidden-4k-fp32 | 56.3000 | 30.16 | [1024, 4096] | float32 | msprof | main | 1.7500 |

## Warnings

- DropoutOp/DropoutOp-a2d97090703b: missing tileops candidate record
- DropoutOp/DropoutOp-b4bd76ea5cf8: missing tileops candidate record
- DropoutOp/DropoutOp-1098851143a0: missing tileops candidate record
- DropoutOp/DropoutOp-880534da4c5b: missing tileops candidate record
- DropoutOp/DropoutOp-433f1d70eeb7: missing tileops candidate record
- DropoutOp/DropoutOp-0fa1737438f2: missing tileops candidate record
