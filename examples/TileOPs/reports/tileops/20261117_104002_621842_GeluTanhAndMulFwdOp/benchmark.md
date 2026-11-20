# NPU Benchmark Report
Generated: 2026-11-17 10:52:33

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GeluTanhAndMulFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | N | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 2.0800 | 0.4536 | 1.7984 | 1.7500 | 3.9304 | 0.0143 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_liytmaa9 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.08, 'max_latency_us': 2.08, 'min_latency_us': 2.08, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 2.1000 | 0.4233 | 8.4848 | 1.7500 | 4.0320 | 0.0171 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_8gv7umyv | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.1, 'max_latency_us': 2.1, 'min_latency_us': 2.1, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 88.8200 | 97.8217 | 3.8912 | 1.7500 | 5.3694 | 5.2524 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_4b_w32mg | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 88.82, 'max_latency_us': 88.82, 'min_latency_us': 88.82, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 141.9400 | 68.7425 | 3.8571 | 1.7500 | 4.6011 | 3.1629 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_htuaaxpw | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 141.940002, 'max_latency_us': 141.940002, 'min_latency_us': 141.940002, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 2.3800 | 2.1895 | 4.0813 | 1.7500 | 4.4410 | 0.0972 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_r_5pytzl | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 2.38, 'max_latency_us': 2.38, 'min_latency_us': 2.38, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 4.4600 | 31.7190 | 4.7177 | 1.7500 | 5.2457 | 1.6639 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_6dbg53wq | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 4.46, 'max_latency_us': 4.46, 'min_latency_us': 4.46, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 5.3800 | 25.2127 | 3.9615 | 1.7500 | 4.5461 | 1.1462 | /mnt/workspace/reports/2026-11/optsweep_1117/20261117_104002_621842_GeluTanhAndMulFwdOp/artifacts/msprof/main_41qakbb2 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'kernel_names': None, 'latency_us': 5.38, 'max_latency_us': 5.38, 'min_latency_us': 5.38, 'multi_launch': False, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
