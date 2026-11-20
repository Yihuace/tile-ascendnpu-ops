# TileOPs Evaluation Report: AlibiFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T21:05:32.818538+08:00 |
| evaluation_scope | AlibiFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_205040_770919_AlibiFwdOp |
| git_commit | 00aeb862c8cad637350900a64d4f9e0b840427c7 |
| correctness_target | tests/ops/test_special_elementwise.py |
| benchmark_target | benchmarks/ops/bench_independent_elementwise.py |

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
| 100.0% | 1 | 15 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AlibiFwdOp | 100.0% (10/10) | N/A | 4 | 28.69% – 74.97% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AlibiFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-prefill-2k-fp16 | 133.6400 | 28.69 | {"num_heads": 32, "seq_len": 2048} | float16 | msprof | main | 1.7500 |
| llama-prefill-2k-bf16 | 127.9400 | 47.72 | {"num_heads": 32, "seq_len": 2048} | bfloat16 | msprof | main | 1.7500 |
| llama-prefill-4k-fp16 | 665.0000 | 74.97 | {"num_heads": 32, "seq_len": 4096} | float16 | msprof | main | 1.7500 |
| llama-prefill-4k-bf16 | 667.8000 | 74.65 | {"num_heads": 32, "seq_len": 4096} | bfloat16 | msprof | main | 1.7500 |

## Warnings

- AlibiFwdOp/AlibiFwdOp-3d2c5c30d2ae: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-ea62e821c966: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-d126e0c7ba94: missing tileops candidate record
- AlibiFwdOp/AlibiFwdOp-00f9b454e905: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-0e5a5a523415: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-7c0b38b71b25: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-d6b10af60ee9: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-c74c8fc43c07: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-6c7953e0cce5: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-8192a2618c50: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-b8b74b19c293: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-2a8e47e5eb30: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-a6dd62231226: missing tileops candidate record
- ClampFwdOp/ClampFwdOp-2e98ada8467c: missing tileops candidate record
