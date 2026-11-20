# NPU Benchmark Report
Generated: 2026-10-17 08:16:18

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## DropoutOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-dropout-1m | 16.1800 | 26.3521 | 3.7225 | 1.7500 | 3.6620 | 0.9650 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_neuplzru | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 16.18, 'max_latency_us': 16.18, 'min_latency_us': 16.18, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 1024} | float32 | [1024, 1024] |
| smoke-dropout-1m | 11.0400 | 36.0819 | 7.3966 | 1.7500 | 3.8943 | 1.4051 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_muy7f816 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.04, 'max_latency_us': 11.04, 'min_latency_us': 11.04, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 2048} | float16 | [1024, 1024] |
| smoke-dropout-1m | 11.9000 | 39.2110 | 8.3966 | 1.7500 | 3.7740 | 1.4798 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_fs4io9ny | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 11.9, 'max_latency_us': 11.9, 'min_latency_us': 11.9, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 2048} | bfloat16 | [1024, 1024] |
| tokens-1k-hidden-4k-fp16 | 35.8800 | 44.0416 | 7.3453 | 1.7500 | 3.8993 | 1.7173 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_3qpyfj3c | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 35.880001, 'max_latency_us': 35.880001, 'min_latency_us': 35.880001, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 2048} | float16 | [1024, 4096] |
| tokens-1k-hidden-10k-bf16 | 92.7600 | 49.8772 | 8.3350 | 1.7500 | 3.7781 | 1.8844 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_l82qmx_4 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 92.760002, 'max_latency_us': 92.760002, 'min_latency_us': 92.760002, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 2048} | bfloat16 | [1024, 10240] |
| tokens-1k-hidden-4k-fp32 | 56.3000 | 30.1571 | 3.7080 | 1.7500 | 3.6641 | 1.1050 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_080218_487917_DropoutOp/artifacts/msprof/main_nl6sntj6 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 56.299999, 'max_latency_us': 56.299999, 'min_latency_us': 56.299999, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | {'p': 0.5, 'training': True} | {'block_size': 1024} | float32 | [1024, 4096] |

### torch

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | op_params | result_bl | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-dropout-1m | 302.6380 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0277 | 0.0035 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.0277182904759952, 'latency_us': 302.63800024986267, 'prof_mode': 'events', 'tflops': 0.0034647863094994} | {'block_size': 1024} | float32 | [1024, 1024] |
| smoke-dropout-1m | 313.2680 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0134 | 0.0033 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.013388868449406371, 'latency_us': 313.2679969072342, 'prof_mode': 'events', 'tflops': 0.003347217112351593} | {'block_size': 2048} | float16 | [1024, 1024] |
| smoke-dropout-1m | 395.5260 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0106 | 0.0027 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.010604369774186335, 'latency_us': 395.52600383758545, 'prof_mode': 'events', 'tflops': 0.002651092443546584} | {'block_size': 2048} | bfloat16 | [1024, 1024] |
| tokens-1k-hidden-4k-fp16 | 386.4520 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0434 | 0.0109 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.043413453861027414, 'latency_us': 386.45199835300446, 'prof_mode': 'events', 'tflops': 0.010853363465256853} | {'block_size': 2048} | float16 | [1024, 4096] |
| tokens-1k-hidden-10k-bf16 | 303.7060 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.1381 | 0.0345 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.13810408648917172, 'latency_us': 303.7060022354126, 'prof_mode': 'events', 'tflops': 0.03452602162229293} | {'block_size': 2048} | bfloat16 | [1024, 10240] |
| tokens-1k-hidden-4k-fp32 | 306.2780 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.1096 | 0.0137 | {'p': 0.5, 'training': True} | {'bandwidth_tbs': 0.10955547600112846, 'latency_us': 306.27799928188324, 'prof_mode': 'events', 'tflops': 0.013694434500141057} | {'block_size': 1024} | float32 | [1024, 4096] |
