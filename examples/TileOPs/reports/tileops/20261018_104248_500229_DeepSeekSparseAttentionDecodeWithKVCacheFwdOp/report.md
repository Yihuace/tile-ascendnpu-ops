# TileOPs Evaluation Report: DeepSeekSparseAttentionDecodeWithKVCacheFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-18T10:58:55.331952+08:00 |
| evaluation_scope | DeepSeekSparseAttentionDecodeWithKVCacheFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp |
| git_commit | 5a90b9918747fe9ed4870e6cd5c39760c68c33e8 |
| correctness_target | tests/ops/test_deepseek_dsa_decode.py |
| benchmark_target | benchmarks/ops/bench_deepseek_dsa_decode.py |

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
| DeepSeekSparseAttentionDecodeWithKVCacheFwdOp | 100.0% (5/5) | 2.26e-03 | 5 | 0.35% – 0.35% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### DeepSeekSparseAttentionDecodeWithKVCacheFwdOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| dsa-decode-smoke | 8961.1602 | 0.35 | {"batch": 1, "dim": 512, "dim_tail": 64, "heads": 128, "heads_kv": 1, "q_start_index_s": 128, "seq_len_kv": 256, "seq_len_q": 128, "sm_scale": null, "stride_kv": 1, "topk": 256} | float16 | msprof | main | 1.7500 |
| dsa-decode-smoke | 9007.2197 | 0.35 | {"batch": 1, "dim": 512, "dim_tail": 64, "heads": 128, "heads_kv": 1, "q_start_index_s": 128, "seq_len_kv": 256, "seq_len_q": 128, "sm_scale": null, "stride_kv": 1, "topk": 256} | bfloat16 | msprof | main | 1.7500 |
| single-batch-mainstream | 567489.5625 | 0.35 | {"batch": 1, "dim": 512, "dim_tail": 64, "heads": 128, "heads_kv": 1, "q_start_index_s": 1024, "seq_len_kv": 2048, "seq_len_q": 1024, "sm_scale": null, "stride_kv": 1, "topk": 2048} | float16 | msprof | main | 1.7500 |
| longer-kv-lower-topk | 142168.1562 | 0.35 | {"batch": 1, "dim": 512, "dim_tail": 64, "heads": 128, "heads_kv": 1, "q_start_index_s": 512, "seq_len_kv": 4096, "seq_len_q": 512, "sm_scale": null, "stride_kv": 1, "topk": 1024} | float16 | msprof | main | 1.7500 |
| long-kv-low-topk | 35664.9609 | 0.35 | {"batch": 1, "dim": 512, "dim_tail": 64, "heads": 128, "heads_kv": 1, "q_start_index_s": 256, "seq_len_kv": 8192, "seq_len_q": 256, "sm_scale": null, "stride_kv": 1, "topk": 512} | float16 | msprof | main | 1.7500 |
