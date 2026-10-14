# NPU Benchmark Report
Generated: 2026-10-14 20:29:22

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
| resnet-stage1 | 20.7800 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/main_mix_aic_pqev2z1z | 0.1514 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 20.780001, 'max_latency_us': 20.780001, 'min_latency_us': 20.780001, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.2271 | 64 | 8 | 1024 | float16 |
| resnet-stage1 | 21.1000 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/main_mix_aic_p848yyo1 | 0.1491 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 21.1, 'max_latency_us': 21.1, 'min_latency_us': 21.1, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.2236 | 64 | 8 | 1024 | bfloat16 |
| resnet-stage2 | 24.9800 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/main_mix_aic_utm40gav | 0.1260 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 24.98, 'max_latency_us': 24.98, 'min_latency_us': 24.98, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1889 | 128 | 4 | 1024 | float16 |
| resnet-stage3 | 40.4200 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/main_mix_aic_ybuk49lr | 0.1192 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 40.419998, 'max_latency_us': 40.419998, 'min_latency_us': 40.419998, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1788 | 256 | 4 | 784 | float16 |
| resnet-stage3 | 41.5000 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/main_mix_aic_6j13d2av | 0.2323 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 41.5, 'max_latency_us': 41.5, 'min_latency_us': 41.5, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 0.1741 | 256 | 4 | 784 | float32 |
| large-spatial | 2085.3401 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/artifacts/msprof/apply_mix_aic_vjlg9aza | 0.3862 | {'captured_op_count': 3, 'captured_op_names': ['apply_mix_aic', 'finalize_mix_aic', 'stats_mix_aic'], 'kernel_name_attempts': ['main', None], 'kernel_name_requested': 'main', 'kernel_name_resolved': None, 'latency_us': 2085.340088, 'max_latency_us': 2784.47998, 'min_latency_us': 16.719999, 'op_name': 'apply_mix_aic', 'sample_count': 3} | msprof | 0.5793 | 128 | 4 | 262144 | float16 |
