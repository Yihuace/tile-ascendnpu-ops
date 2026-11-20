# TileOPs Evaluation Report: EngramGateConvFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T01:03:33.348538+08:00 |
| evaluation_scope | EngramGateConvFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_005814_531237_EngramGateConvFwdOp |
| git_commit | 90908d5694f131fa5538eeb3546d740c4b50bec4 |
| correctness_target | tests/ops/test_engram_gate_conv_fwd.py |
| benchmark_target | benchmarks/ops/bench_engram_gate_conv_fwd.py |

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
| 100.0% | 1 | 4 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| EngramGateConvFwdOp | 100.0% (4/4) | 1.14e-02 | 4 | 0.99% – 9.16% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EngramGateConvFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| fwd-smoke-b1-s16-d256-fp16 | 4.4600 | 0.99 | {"M": 1, "d": 256, "seq_len": 16} | float16 | msprof | main | 1.7500 |
| fwd-b1-s32-d256-fp16 | 4.9600 | 1.78 | {"M": 1, "d": 256, "seq_len": 32} | float16 | msprof | main | 1.7500 |
| fwd-b2-s64-d512-fp16 | 7.4600 | 9.16 | {"M": 2, "d": 512, "seq_len": 64} | float16 | msprof | main | 1.7500 |
| fwd-b1-s128-d256-bf16 | 6.3800 | 5.77 | {"M": 1, "d": 256, "seq_len": 128} | bfloat16 | msprof | main | 1.7500 |
