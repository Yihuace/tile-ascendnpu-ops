# NPU Benchmark Report
Generated: 2026-11-17 10:39:44

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GeluAndMulFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | N | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 2.6000 | 1.1282 | 7.3798 | 1.7500 | 4.1542 | 0.0469 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_4_x6t1au | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.6, 'max_latency_us': 2.6, 'min_latency_us': 2.6, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 2.1200 | 0.4193 | 8.4848 | 1.7500 | 4.0320 | 0.0169 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_euc6xyw6 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.12, 'max_latency_us': 2.12, 'min_latency_us': 2.12, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 91.5000 | 94.9565 | 3.8350 | 1.7500 | 5.3694 | 5.0986 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_h0y3xg7c | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 91.5, 'max_latency_us': 91.5, 'min_latency_us': 91.5, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 142.4600 | 68.4916 | 3.7889 | 1.7500 | 4.6011 | 3.1514 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_h2im6jpf | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 142.460007, 'max_latency_us': 142.460007, 'min_latency_us': 142.460007, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 2.3600 | 2.2081 | 4.0813 | 1.7500 | 4.4410 | 0.0981 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_dcm8zfzm | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.36, 'max_latency_us': 2.36, 'min_latency_us': 2.36, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 4.4400 | 31.8619 | 4.7177 | 1.7500 | 5.2457 | 1.6714 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_0t2bucbb | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 4.44, 'max_latency_us': 4.44, 'min_latency_us': 4.44, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 5.4200 | 25.0266 | 3.9615 | 1.7500 | 4.5461 | 1.1377 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_102709_565655_GeluAndMulFwdOp/artifacts/msprof/main_nz5i2hcb | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 5.42, 'max_latency_us': 5.42, 'min_latency_us': 5.42, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
