# NPU Benchmark Report
Generated: 2026-10-18 10:58:49

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## DeepSeekSparseAttentionDecodeWithKVCacheFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | dim_tail | heads | heads_kv | q_start_index_s | seq_len_kv | seq_len_q | sm_scale | stride_kv | topk | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsa-decode-smoke | 8961.1602 | 0.3493 | 625.2284 | 1.7500 | 299.3019 | 1.0455 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/artifacts/msprof/main_mix_aic_vemq5zo0 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 8961.160156, 'max_latency_us': 8961.160156, 'min_latency_us': 8961.160156, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 1 | 512 | 64 | 128 | 1 | 128 | 256 | 128 | None | 1 | 256 | {'block_N': 64, 'block_heads': 32, 'num_stages': 2} | float16 |
| dsa-decode-smoke | 9007.2197 | 0.3484 | 626.8918 | 1.7500 | 299.3215 | 1.0429 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/artifacts/msprof/main_mix_aic_fjymly38 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 9007.219727, 'max_latency_us': 9007.219727, 'min_latency_us': 9007.219727, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 1 | 512 | 64 | 128 | 1 | 128 | 256 | 128 | None | 1 | 256 | {'block_N': 64, 'block_heads': 32, 'num_stages': 2} | bfloat16 |
| single-batch-mainstream | 567489.5625 | 0.3515 | 607.0604 | 1.7500 | 299.4311 | 1.0524 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/artifacts/msprof/main_mix_aic_w6rypi7b | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 567489.5625, 'max_latency_us': 567489.5625, 'min_latency_us': 567489.5625, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 1 | 512 | 64 | 128 | 1 | 1024 | 2048 | 1024 | None | 1 | 2048 | {'block_N': 64, 'block_heads': 32, 'num_stages': 2} | float16 |
| longer-kv-lower-topk | 142168.1562 | 0.3510 | 370.8241 | 1.7500 | 299.4092 | 1.0508 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/artifacts/msprof/main_mix_aic_cehp7wdu | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 142168.15625, 'max_latency_us': 142168.15625, 'min_latency_us': 142168.15625, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 1 | 512 | 64 | 128 | 1 | 512 | 4096 | 512 | None | 1 | 1024 | {'block_N': 64, 'block_heads': 32, 'num_stages': 2} | float16 |
| long-kv-low-topk | 35664.9609 | 0.3502 | 345.0673 | 1.7500 | 299.3694 | 1.0484 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261018_104248_500229_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/artifacts/msprof/main_mix_aic_0sqs6wep | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 35664.960938, 'max_latency_us': 35664.960938, 'min_latency_us': 35664.960938, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 1 | 512 | 64 | 128 | 1 | 256 | 8192 | 256 | None | 1 | 512 | {'block_N': 64, 'block_heads': 32, 'num_stages': 2} | float16 |
