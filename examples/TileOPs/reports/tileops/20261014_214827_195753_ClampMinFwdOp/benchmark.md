# NPU Benchmark Report
Generated: 2026-10-14 22:02:26

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## ClampMinFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1m | 5.3800 | 22.2759 | 2.0043 | 1.7500 | 4.5979 | 0.7813 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_odb505sq | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.38, 'max_latency_us': 5.38, 'min_latency_us': 5.38, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | float16 | [1048576] |
| smoke-1m | 5.2600 | 22.7841 | 2.5042 | 1.7500 | 4.5999 | 0.9985 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_j8zd9kjn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.26, 'max_latency_us': 5.26, 'min_latency_us': 5.26, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | bfloat16 | [1048576] |
| 16M-min-only | 38.5400 | 49.7510 | 2.0021 | 1.7500 | 4.6032 | 1.7431 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_1s8iy7ja | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 38.540001, 'max_latency_us': 38.540001, 'min_latency_us': 38.540001, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | float16 | [4096, 4096] |
| 16M-min-only | 38.4800 | 49.8286 | 2.5021 | 1.7500 | 4.6041 | 2.1818 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_k__5jwiq | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 38.48, 'max_latency_us': 38.48, 'min_latency_us': 38.48, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | bfloat16 | [4096, 4096] |
| 64M-min-only | 164.3200 | 58.5086 | 1.5971 | 1.7500 | 4.6034 | 1.6352 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_7zaktwch | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 164.320007, 'max_latency_us': 164.320007, 'min_latency_us': 164.320007, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | float16 | [8192, 8192] |
| 64M-min-only | 163.0000 | 58.1670 | 2.0239 | 1.7500 | 4.6043 | 2.0602 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_iewwwul7 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 163.0, 'max_latency_us': 163.0, 'min_latency_us': 163.0, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | bfloat16 | [8192, 8192] |
| 256M-min-only | 829.2600 | 67.0422 | 1.1047 | 1.7500 | 4.6035 | 1.2961 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_j76f_664 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 829.26001, 'max_latency_us': 829.26001, 'min_latency_us': 829.26001, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | float16 | [16384, 16384] |
| 256M-min-only | 857.2800 | 64.8406 | 1.3808 | 1.7500 | 4.6044 | 1.5669 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/artifacts/msprof/main_9yl78jgf | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 857.280029, 'max_latency_us': 857.280029, 'min_latency_us': 857.280029, 'op_name': 'main', 'sample_count': 1} | msprof | {'min': -0.5} | {'block_size': 8192} | bfloat16 | [16384, 16384] |
