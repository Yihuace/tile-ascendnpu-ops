# NPU Benchmark Report
Generated: 2026-11-19 16:14:22

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## MultiHeadAttentionDecodePagedWithKVCacheFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | is_causal | page_size | seqlen_kv | seqlen_q | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single-token-page128 | 7.0400 | 33.9626 | 1.5425 | 1.7500 | 4.9128 | 0.9168 | /mnt/workspace/tilelang-ascend-npu-ops-20261119/examples/TileOPs/reports/tileops/20261119_160151_272035_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/artifacts/msprof/decode_vector_packed_34o6iq9a | {'captured_op_count': 1, 'captured_op_names': ['decode_vector_packed'], 'kernel_name_attempts': ['decode_vector_packed'], 'kernel_name_requested': 'decode_vector_packed', 'kernel_name_resolved': 'decode_vector_packed', 'kernel_names': None, 'latency_us': 7.04, 'max_latency_us': 7.04, 'min_latency_us': 7.04, 'multi_launch': False, 'op_name': 'decode_vector_packed', 'sample_count': 1} | msprof | 1 | 128 | 16 | False | 128 | 512 | 1 | {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2} | float16 |
| batch2-page256 | 7.6200 | 31.2441 | 1.8335 | 1.7500 | 5.1911 | 1.0025 | /mnt/workspace/tilelang-ascend-npu-ops-20261119/examples/TileOPs/reports/tileops/20261119_160151_272035_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/artifacts/msprof/decode_vector_packed_dtysyl5m | {'captured_op_count': 1, 'captured_op_names': ['decode_vector_packed'], 'kernel_name_attempts': ['decode_vector_packed'], 'kernel_name_requested': 'decode_vector_packed', 'kernel_name_resolved': 'decode_vector_packed', 'kernel_names': None, 'latency_us': 7.62, 'max_latency_us': 7.62, 'min_latency_us': 7.62, 'multi_launch': False, 'op_name': 'decode_vector_packed', 'sample_count': 1} | msprof | 2 | 64 | 8 | False | 256 | 1024 | 1 | {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2} | float16 |
| longer-cache | 5.5000 | 21.6450 | 1.8299 | 1.7500 | 5.2113 | 0.6932 | /mnt/workspace/tilelang-ascend-npu-ops-20261119/examples/TileOPs/reports/tileops/20261119_160151_272035_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/artifacts/msprof/decode_vector_packed_yj8vyi4r | {'captured_op_count': 1, 'captured_op_names': ['decode_vector_packed'], 'kernel_name_attempts': ['decode_vector_packed'], 'kernel_name_requested': 'decode_vector_packed', 'kernel_name_resolved': 'decode_vector_packed', 'kernel_names': None, 'latency_us': 5.5, 'max_latency_us': 5.5, 'min_latency_us': 5.5, 'multi_launch': False, 'op_name': 'decode_vector_packed', 'sample_count': 1} | msprof | 1 | 64 | 8 | False | 256 | 1024 | 1 | {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2} | float16 |
| shorter-cache | 4.3400 | 13.6208 | 1.8302 | 1.7500 | 5.2400 | 0.4363 | /mnt/workspace/tilelang-ascend-npu-ops-20261119/examples/TileOPs/reports/tileops/20261119_160151_272035_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/artifacts/msprof/decode_vector_packed_4nhrhwim | {'captured_op_count': 1, 'captured_op_names': ['decode_vector_packed'], 'kernel_name_attempts': ['decode_vector_packed'], 'kernel_name_requested': 'decode_vector_packed', 'kernel_name_resolved': 'decode_vector_packed', 'kernel_names': None, 'latency_us': 4.34, 'max_latency_us': 4.34, 'min_latency_us': 4.34, 'multi_launch': False, 'op_name': 'decode_vector_packed', 'sample_count': 1} | msprof | 1 | 64 | 8 | False | 256 | 512 | 1 | {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2} | float16 |
