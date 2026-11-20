# TileOPs Evaluation Report: RopeNeoxOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T17:48:21.221493+08:00 |
| evaluation_scope | RopeNeoxOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_172054_008103_RopeNeoxOp |
| git_commit | e81800c554c192a7331cb7784a2b8306213efa59 |
| correctness_target | tests/ops/test_rope.py |
| benchmark_target | benchmarks/ops/bench_rope.py |

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
| 100.0% | 1 | 38 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| RopeNeoxOp | 100.0% (16/16) | 0.00e+00 | 6 | 1.38% – 48.91% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### RopeNeoxOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| neox-1d-256-d64-smoke | 3.0600 | 1.38 | [256, 64] | float16 | msprof | main | 1.7500 |
| neox-1d-256-d64-smoke | 3.1000 | 1.43 | [256, 64] | bfloat16 | msprof | main | 1.7500 |
| neox-1d-256-d64-smoke | 3.1200 | 2.02 | [256, 64] | float32 | msprof | main | 1.7500 |
| neox-1d-2k-d64-fp16 | 3.5200 | 8.35 | [2048, 64] | float16 | msprof | main | 1.7500 |
| neox-1d-4k-d128-bf16 | 4.8200 | 24.86 | [4096, 128] | bfloat16 | msprof | main | 1.7500 |
| neox-2d-b2-s2k-h32-d128-fp16 | 45.4800 | 48.91 | [2, 2048, 32, 128] | float16 | msprof | main | 1.7500 |
