# NPU Benchmark Report
Generated: 2026-10-14 22:11:03

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## FFTC2COp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | batch_shape | n | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fft-64-c64-smoke | 7.1800 | 0.0102 | 0.1000 | 1.7500 | 2.3040 | 0.0000 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261014_220803_114070_FFTC2COp/artifacts/msprof/main_kvst5pt9 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.18, 'max_latency_us': 7.18, 'min_latency_us': 7.18, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | [] | 64 | {'block_size': 1024} | complex64 | [64] |
| fft-4k-c64-unbatched | 717.8000 | 0.0033 | 0.0991 | 1.7500 | 2.3040 | 0.0000 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261014_220803_114070_FFTC2COp/artifacts/msprof/main_gsoxfqmv | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 717.799988, 'max_latency_us': 717.799988, 'min_latency_us': 717.799988, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | [] | 4096 | {'block_size': 1024} | complex64 | [4096] |
| fft-4k-c64-b64 | 1496.6801 | 0.0804 | 0.1265 | 1.7500 | 2.3040 | 0.0002 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261014_220803_114070_FFTC2COp/artifacts/msprof/main_9o1wd2vh | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1496.680054, 'max_latency_us': 1496.680054, 'min_latency_us': 1496.680054, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | [64] | 4096 | {'block_size': 1024} | complex64 | [64, 4096] |

### torch-fft

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | batch_shape | n | result_bl | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fft-64-c64-smoke | 269.7880 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0000 | 0.0000 | [] | 64 | {'bandwidth_tbs': 3.7955728622427433e-06, 'latency_us': 269.78799700737, 'prof_mode': 'events', 'tflops': 7.1166991167051445e-06} | {'block_size': 1024} | complex64 | [64] |
| fft-4k-c64-unbatched | 272.6920 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0002 | 0.0009 | [] | 4096 | {'bandwidth_tbs': 0.00024032975117762247, 'latency_us': 272.69199788570404, 'prof_mode': 'events', 'tflops': 0.0009012365669160843} | {'block_size': 1024} | complex64 | [4096] |
| fft-4k-c64-b64 | 290.4280 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0144 | 0.0542 | [64] | 4096 | {'bandwidth_tbs': 0.014441803097690498, 'latency_us': 290.42800068855286, 'prof_mode': 'events', 'tflops': 0.054156761616339366} | {'block_size': 1024} | complex64 | [64, 4096] |
