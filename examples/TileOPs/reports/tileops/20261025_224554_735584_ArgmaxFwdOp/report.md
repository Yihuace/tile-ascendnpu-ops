# TileOPs Evaluation Report: ArgmaxFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-25T22:50:22.304456+08:00 |
| evaluation_scope | ArgmaxFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261025_224554_735584_ArgmaxFwdOp |
| git_commit | da8b5dc6f533d93c5dba6b51c1eeafbc293cbd10 |
| correctness_target | tests/ops/test_argmax.py |
| benchmark_target | benchmarks/ops/bench_argmax.py |

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
| 100.0% | 1 | 42 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ArgmaxFwdOp | 100.0% (42/42) | 0.00e+00 | 3 | 21.79% – 31.11% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ArgmaxFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-argmax | 30.8200 | 31.11 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| hidden-state-argmax | 52.7200 | 21.79 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| 3d-non-last-axis-argmax | 13.7200 | 24.60 | [4, 128, 4096] | float16 | msprof | main | 1.7500 |

## Warnings

- correctness passed, but the benchmark stage did not produce valid records
- benchmark pytest reported 2 failed/error cases

## Failed Cases / 失败用例

| Stage | Test | Outcome | Message |
|---|---|---|---|
| Benchmark | benchmarks.ops.bench_argmax::test_argmax_bench[lm-head-argmax-float16] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmpoprx1n3c/kernel.npuir":87:71): error: custom op 'syn_instr_mode' is unknown (tried 'func.syn_instr_mode' as well) [ERROR] Failed to parse input file:  /tmp/tmpoprx1n3c/kernel.npuir |
| Benchmark | benchmarks.ops.bench_argmax::test_argmax_bench[lm-head-argmax-bfloat16] | failed | RuntimeError: NPU IR opt failed: loc("/tmp/tmppw2m_x7c/kernel.npuir":87:71): error: custom op 'syn_instr_mode' is unknown (tried 'func.syn_instr_mode' as well) [ERROR] Failed to parse input file:  /tmp/tmppw2m_x7c/kernel.npuir |
