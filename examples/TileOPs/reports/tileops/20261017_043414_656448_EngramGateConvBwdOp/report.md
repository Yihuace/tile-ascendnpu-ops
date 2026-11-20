# TileOPs Evaluation Report: EngramGateConvBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T04:42:33.031914+08:00 |
| evaluation_scope | EngramGateConvBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_043414_656448_EngramGateConvBwdOp |
| git_commit | 2b03a4f47cd19eca87f1e5cba083481c1ff0a5c8 |
| correctness_target | tests/ops/test_engram_gate_conv_bwd.py |
| benchmark_target | benchmarks/ops/bench_engram_gate_conv_bwd.py |

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
| 100.0% | 1 | 5 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| EngramGateConvBwdOp | 100.0% (5/5) | 1.30e-03 | 4 | 0.06% – 0.86% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EngramGateConvBwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| bwd-smoke-b1-s16-d256-fp16 | 51.9000 | 0.06 | {"M": 1, "d": 256, "seq_len": 16} | float16 | msprof | k1 | 1.7500 |
| bwd-b1-s32-d256-fp16 | 50.8000 | 0.12 | {"M": 1, "d": 256, "seq_len": 32} | float16 | msprof | k1 | 1.7500 |
| bwd-b2-s64-d512-fp16 | 59.3200 | 0.86 | {"M": 2, "d": 512, "seq_len": 64} | float16 | msprof | k1 | 1.7500 |
| bwd-b1-s128-d256-bf16 | 225.0800 | 0.59 | {"M": 1, "d": 256, "seq_len": 128} | bfloat16 | msprof | k1 | 1.7500 |
