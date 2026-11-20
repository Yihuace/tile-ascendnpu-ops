# NPU Benchmark Report
Generated: 2026-11-08 02:44:02

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-long-w1024 | 1787.2557 | 24.5835 | 2266.0259 | 1.7500 | 299.4350 | 73.6117 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_fhts9p67 | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 1787.255737, 'max_latency_us': 1787.255737, 'min_latency_us': 1787.255737, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [2048, 2048, 2048, 2048] | [2048, 2048, 2048, 2048] | 1024 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |
