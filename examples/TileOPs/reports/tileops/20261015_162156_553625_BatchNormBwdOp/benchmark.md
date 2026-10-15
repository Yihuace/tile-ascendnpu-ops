# NPU Benchmark Report
Generated: 2026-10-15 16:25:03

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## BatchNormBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | c | n | s | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet-stage1 | 19.9400 | 19.3670 | 6.9988 | 1.7500 | 3.8016 | 0.7363 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/main_mix_aic_t4yun7le | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 19.940001, 'max_latency_us': 19.940001, 'min_latency_us': 19.940001, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 64 | 8 | 1024 | float16 |
| resnet-stage1 | 19.9600 | 20.4876 | 7.4987 | 1.7500 | 3.8465 | 0.7881 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/main_mix_aic_d5dkfub4 | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 19.959999, 'max_latency_us': 19.959999, 'min_latency_us': 19.959999, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 64 | 8 | 1024 | bfloat16 |
| resnet-stage2 | 24.3200 | 16.7928 | 7.2507 | 1.7500 | 3.7242 | 0.6254 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/main_mix_aic_5nvr1el_ | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 24.32, 'max_latency_us': 24.32, 'min_latency_us': 24.32, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 128 | 4 | 1024 | float16 |
| resnet-stage3 | 41.1000 | 15.8021 | 9.3367 | 1.7500 | 3.6915 | 0.5833 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/main_mix_aic_g9v1g1fv | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 41.099998, 'max_latency_us': 41.099998, 'min_latency_us': 41.099998, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 256 | 4 | 784 | float16 |
| resnet-stage3 | 41.6000 | 16.4156 | 3.9591 | 1.7500 | 3.3860 | 0.5558 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/main_mix_aic_865lq63b | {'captured_op_count': 1, 'captured_op_names': ['main_mix_aic'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 41.599998, 'max_latency_us': 41.599998, 'min_latency_us': 41.599998, 'op_name': 'main_mix_aic', 'sample_count': 1} | msprof | 256 | 4 | 784 | float32 |
| large-spatial | 2092.2000 | 29.9808 | 4.6876 | 1.7500 | 3.8461 | 1.1531 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/artifacts/msprof/stats_mix_aic_m50hal9a | {'captured_op_count': 1, 'captured_op_names': ['stats_mix_aic'], 'kernel_name_attempts': ['stats'], 'kernel_name_requested': 'stats', 'kernel_name_resolved': 'stats', 'latency_us': 2092.199951, 'max_latency_us': 2092.199951, 'min_latency_us': 2092.199951, 'op_name': 'stats_mix_aic', 'sample_count': 1} | msprof | 128 | 4 | 262144 | float16 |
