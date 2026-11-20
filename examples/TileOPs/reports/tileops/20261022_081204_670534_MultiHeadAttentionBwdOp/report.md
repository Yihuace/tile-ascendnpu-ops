# TileOPs Evaluation Report: MultiHeadAttentionBwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T08:15:09.397132+00:00 |
| evaluation_scope | MultiHeadAttentionBwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_081204_670534_MultiHeadAttentionBwdOp |
| git_commit | 16f1ee4578a66e9123bb32783956e272df17ecad |
| correctness_target | tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/bwd_only/test_mha_bwd_only.py |
| benchmark_target | tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/bwd_only/bench_mha_bwd_only.py |

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
| 100.0% | 1 | 6 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| MultiHeadAttentionBwdOp | 100.0% (6/6) | 1.16e-03 | 8 | 23.41% – 32.90% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MultiHeadAttentionBwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-3.1-8b-short | 30.9200 | 31.01 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | float16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-8b-short | 30.2600 | 31.68 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | bfloat16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-8b-long | 59.6800 | 32.13 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | float16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-8b-long | 58.2800 | 32.90 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | bfloat16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-70b-short | 32.6400 | 29.37 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | float16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-70b-short | 32.0800 | 29.88 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | bfloat16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-70b-long | 81.9200 | 23.41 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | float16 | msprof | flash_bwd_prep | 1.7500 |
| llama-3.1-70b-long | 72.1800 | 26.56 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | bfloat16 | msprof | flash_bwd_prep | 1.7500 |

## Warnings

- bwd-only, NON-MANIFEST entry: o/lse are produced by torch SDPA (fp32), not by the NPU fwd kernel. MultiHeadAttentionFwdOp is EXCLUDED because of the external compiler block (bishengir-compile SIGSEGV in convert-hfusion-to-hivm; perf_opt/REPORT_BLOCKER.md + perf_opt/issue_bishengir_hfusion_hivm/). This report is NOT a formal acceptance: formal status remains BLOCKED_ENVIRONMENT.
- msprof captured a single device kernel only (flash_bwd_prep). The op total is measured with NPU events around one op(...) call and recorded as op_wall_us (benchmark.json); 8/8 tune cases, see OP_WALL.md. op_wall includes host launch gaps and is not comparable with the device-only -47.8% figure (perf_records.jsonl).

## Op Wall / 总时延补充（8/8 tune case，非正式验收）

`msprof` 单 kernel 行只捕获 `flash_bwd_prep`。下表并列给出**一次 `op(...)` 调用的 NPU-event 总时延**
（`op wall`，含 host 下发间隙）与 `msprof` 单 kernel / 全 kernel 求和（device-only）两个数；
详细口径、分解与限制见 [`OP_WALL.md`](OP_WALL.md)。`Bandwidth 1.75 TB/s` 仅为 roofline 分母。

| workload | dtype | **op wall total (us)** | op wall min–max | msprof captured (us) | device sum (us) | per-kernel sdw/pds/mul (us/launch) | n_chunks |
|---|---|---:|---:|---:|---:|---|---:|
| llama-3.1-70b-long | bfloat16 | **12959.0** | 12873–15200 | 72.18 | 7545.28 | 123.6 / 80.2 / 266.4 | 16 |
| llama-3.1-70b-long | float16 | **12760.1** | 12389–13631 | 81.92 | 7525.84 | 123.7 / 77.9 / 269.1 | 16 |
| llama-3.1-70b-short | bfloat16 | **2326.0** | 2300–2447 | 32.08 | 1020.28 | 132.0 / 83.8 / 292.7 | 2 |
| llama-3.1-70b-short | float16 | **2302.5** | 2272–2378 | 32.64 | 1015.72 | 133.5 / 81.1 / 293.9 | 2 |
| llama-3.1-8b-long | bfloat16 | **12121.6** | 11819–14788 | 58.28 | 7531.92 | 124.1 / 79.8 / 267.2 | 16 |
| llama-3.1-8b-long | float16 | **12068.3** | 11996–12165 | 59.68 | 7518.62 | 123.8 / 78.1 / 266.2 | 16 |
| llama-3.1-8b-short | bfloat16 | **2516.3** | 2482–2737 | 30.26 | 1028.34 | 133.3 / 83.8 / 291.5 | 2 |
| llama-3.1-8b-short | float16 | **2382.4** | 2328–4010 | 30.92 | 1016.78 | 133.3 / 81.2 / 293.3 | 2 |

> 两个数不可互换：`op wall` 是端到端（含每次 launch 的 host 间隙），`device sum` 是器件 Task Duration
> 之和。round 11 的 −47.8% 属后者；本表不把 op wall 冒充验收数字。fwd 因外部编译器阻塞排除，
> 正式验收仍为 **BLOCKED_ENVIRONMENT**。`Ratio (%)` 与 `Bandwidth 1.75 TB/s` 均由上表 msprof**单 kernel** 数字/roofline 分母得到，不是 op 级实测占比或带宽。
