# NPU Benchmark Report
Generated: 2026-11-08 02:39:16

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
| llama-3.1-8b-short-w256 | 308.9062 | 14.3315 | 1047.9650 | 1.7500 | 299.4177 | 42.9112 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_8wrccqc_ | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 308.906189, 'max_latency_us': 308.906189, 'min_latency_us': 308.906189, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [512, 512, 512, 512] | [512, 512, 512, 512] | 256 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |
