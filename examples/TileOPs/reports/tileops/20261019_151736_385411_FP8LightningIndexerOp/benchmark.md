# NPU Benchmark Report
Generated: 2026-10-19 15:20:13

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## FP8LightningIndexerOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | clean_logits | heads | index_dim | kv_group | seq_len | seq_len_kv | config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightning-indexer-smoke-s1k-h8-d64 | 134.9400 | 2.7660 | 930.7040 | 1.7500 | 298.7836 | 8.2642 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261019_151736_385411_FP8LightningIndexerOp/artifacts/msprof/_fp8_lightning_indexer_main_mix_aic_xsiu1vjf | {'captured_op_count': 1, 'captured_op_names': ['_fp8_lightning_indexer_main_mix_aic'], 'kernel_name_attempts': ['_fp8_lightning_indexer_main'], 'kernel_name_requested': '_fp8_lightning_indexer_main', 'kernel_name_resolved': '_fp8_lightning_indexer_main', 'latency_us': 134.940002, 'max_latency_us': 134.940002, 'min_latency_us': 134.940002, 'op_name': '_fp8_lightning_indexer_main_mix_aic', 'sample_count': 1} | msprof | 1 | True | 8 | 64 | 1 | 1024 | 1024 | {'block_N': 64, 'block_Q': 8, 'num_stages': 2} |
| lightning-indexer-bf16-s8k-h32-d64 | 10519.7002 | 9.0403 | 1502.1758 | 1.7500 | 299.0944 | 27.0389 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261019_151736_385411_FP8LightningIndexerOp/artifacts/msprof/_fp8_lightning_indexer_main_mix_aic_0amx7lr8 | {'captured_op_count': 1, 'captured_op_names': ['_fp8_lightning_indexer_main_mix_aic'], 'kernel_name_attempts': ['_fp8_lightning_indexer_main'], 'kernel_name_requested': '_fp8_lightning_indexer_main', 'kernel_name_resolved': '_fp8_lightning_indexer_main', 'latency_us': 10519.700195, 'max_latency_us': 10519.700195, 'min_latency_us': 10519.700195, 'op_name': '_fp8_lightning_indexer_main_mix_aic', 'sample_count': 1} | msprof | 1 | True | 32 | 64 | 1 | 8192 | 8192 | {'block_N': 64, 'block_Q': 8, 'num_stages': 2} |
| lightning-indexer-bf16-s4k-h32-d64 | 5288.1602 | 8.9919 | 4514.8408 | 1.7500 | 299.0944 | 26.8941 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261019_151736_385411_FP8LightningIndexerOp/artifacts/msprof/_fp8_lightning_indexer_main_mix_aic_g4yvm4jr | {'captured_op_count': 1, 'captured_op_names': ['_fp8_lightning_indexer_main_mix_aic'], 'kernel_name_attempts': ['_fp8_lightning_indexer_main'], 'kernel_name_requested': '_fp8_lightning_indexer_main', 'kernel_name_resolved': '_fp8_lightning_indexer_main', 'latency_us': 5288.160156, 'max_latency_us': 5288.160156, 'min_latency_us': 5288.160156, 'op_name': '_fp8_lightning_indexer_main_mix_aic', 'sample_count': 1} | msprof | 1 | True | 32 | 64 | 1 | 4096 | 8192 | {'block_N': 64, 'block_Q': 8, 'num_stages': 2} |
