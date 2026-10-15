# NPU Benchmark Report
Generated: 2026-10-15 11:53:46

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## HardsigmoidFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1m | 6.6400 | 27.5669 | 3.7608 | 1.7500 | 4.3090 | 1.1879 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_uxtgmi1d | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.64, 'max_latency_us': 6.64, 'min_latency_us': 6.64, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [1048576] |
| smoke-1m | 6.8800 | 29.9128 | 4.2607 | 1.7500 | 4.3420 | 1.2988 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_rlpyqvym | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.88, 'max_latency_us': 6.88, 'min_latency_us': 6.88, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [1048576] |
| mbv3-se-gate | 2.8600 | 0.5007 | 3.9167 | 1.7500 | 4.3063 | 0.0216 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_ls0hysz_ | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.86, 'max_latency_us': 2.86, 'min_latency_us': 2.86, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [32, 240, 1, 1] |
| mbv3-se-gate | 2.7600 | 0.5832 | 4.4370 | 1.7500 | 4.3396 | 0.0253 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main__ropk0ml | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.76, 'max_latency_us': 2.76, 'min_latency_us': 2.76, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [32, 240, 1, 1] |
| mbv3-se-gate-deep | 2.4200 | 2.3668 | 3.9896 | 1.7500 | 4.3063 | 0.1019 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_qrwvb_ky | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.42, 'max_latency_us': 2.42, 'min_latency_us': 2.42, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [32, 960, 1, 1] |
| mbv3-se-gate-deep | 2.6800 | 2.4026 | 4.5197 | 1.7500 | 4.3396 | 0.1043 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_84eap7hk | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.68, 'max_latency_us': 2.68, 'min_latency_us': 2.68, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [32, 960, 1, 1] |
| fc-wide | 28.9800 | 50.5030 | 3.7599 | 1.7500 | 4.3101 | 2.1767 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_pmy6p4jk | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 28.98, 'max_latency_us': 28.98, 'min_latency_us': 28.98, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [2048, 4096] |
| fc-wide | 28.8800 | 56.9814 | 4.2599 | 1.7500 | 4.3430 | 2.4747 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/artifacts/msprof/main_x1osn0np | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 28.879999, 'max_latency_us': 28.879999, 'min_latency_us': 28.879999, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [2048, 4096] |
