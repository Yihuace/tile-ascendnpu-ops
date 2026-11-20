# NPU Benchmark Report
Generated: 2026-11-17 14:10:20

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
| resnet-stage1 | 6.7800 | 17.6794 | 2.2934 | 1.7500 | 5.7945 | 0.7096 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/main_b2x3bu_6 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 6.78, 'max_latency_us': 6.78, 'min_latency_us': 6.78, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 64 | 8 | 1024 | float16 |
| resnet-stage1 | 8.0200 | 19.4070 | 2.9631 | 1.7500 | 3.9935 | 0.7750 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/main_3oz2z8k3 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 8.02, 'max_latency_us': 8.02, 'min_latency_us': 8.02, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 64 | 8 | 1024 | bfloat16 |
| resnet-stage2 | 9.8000 | 12.2313 | 2.3383 | 1.7500 | 5.6712 | 0.5005 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/main_bs0_o4uv | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 9.8, 'max_latency_us': 9.8, 'min_latency_us': 9.8, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 128 | 4 | 1024 | float16 |
| resnet-stage3 | 10.3200 | 17.6046 | 2.4787 | 1.7500 | 5.7965 | 0.7636 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/main_bue148bo | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 10.32, 'max_latency_us': 10.32, 'min_latency_us': 10.32, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 256 | 4 | 784 | float16 |
| resnet-stage3 | 14.7600 | 24.8681 | 1.2552 | 1.7500 | 3.6371 | 0.5463 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/main_zwkfv8o3 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 14.76, 'max_latency_us': 14.76, 'min_latency_us': 14.76, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 256 | 4 | 784 | float32 |
| large-spatial | 371.6400 | 82.5489 | 1.3003 | 1.7500 | 4.5427 | 1.8784 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_140629_142480_BatchNormBwdOp/artifacts/msprof/stats_mix_aic_af4upism | {'captured_op_count': 1, 'captured_op_names': ['stats_mix_aic'], 'kernel_name_attempts': ['stats'], 'kernel_name_requested': 'stats', 'kernel_name_resolved': 'stats', 'kernel_names': None, 'latency_us': 371.640015, 'max_latency_us': 371.640015, 'min_latency_us': 371.640015, 'multi_launch': False, 'op_name': 'stats_mix_aic', 'sample_count': 1} | msprof | 128 | 4 | 262144 | float16 |
