# NPU Benchmark Report
Generated: 2026-10-15 16:04:12

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## BatchNormBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | c | n | s | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet-stage1 | 19.9800 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/main_mix_aic__3wix7p2 | 0.1575 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 19.98, 'max_latency_us': 19.98, 'min_latency_us': 19.98, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.2362 | 64 | 8 | 1024 | float16 |
| resnet-stage1 | 21.1200 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/main_mix_aic__0tftf36 | 0.1490 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 21.120001, 'max_latency_us': 21.120001, 'min_latency_us': 21.120001, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.2234 | 64 | 8 | 1024 | bfloat16 |
| resnet-stage2 | 24.3600 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/main_mix_aic_dlvvphl7 | 0.1292 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 24.360001, 'max_latency_us': 24.360001, 'min_latency_us': 24.360001, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1937 | 128 | 4 | 1024 | float16 |
| resnet-stage3 | 41.0400 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/main_mix_aic_3zq1vy3q | 0.1174 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 41.040001, 'max_latency_us': 41.040001, 'min_latency_us': 41.040001, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1761 | 256 | 4 | 784 | float16 |
| resnet-stage3 | 40.8400 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/main_mix_aic_ittrea0l | 0.2360 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 40.84, 'max_latency_us': 40.84, 'min_latency_us': 40.84, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1769 | 256 | 4 | 784 | float32 |
| large-spatial | 2092.8000 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_155843_127784_BatchNormBwdOp/artifacts/msprof/stats_mix_aic_avmbh2tr | 0.3848 | {'captured_op_count': 1, 'captured_op_names': ['stats_mix_aic'], 'kernel_name_attempts': ['stats'], 'kernel_name_requested': 'stats', 'kernel_name_resolved': 'stats', 'latency_us': 2092.800049, 'max_latency_us': 2092.800049, 'min_latency_us': 2092.800049, 'op_name': 'stats_mix_aic', 'sample_count': 1} | msprof | 0.5772 | 128 | 4 | 262144 | float16 |
