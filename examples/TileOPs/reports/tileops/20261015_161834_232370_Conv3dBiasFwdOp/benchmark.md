# NPU Benchmark Report
Generated: 2026-10-15 16:21:38

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## Conv3dBiasFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | @py_assert1 | @py_assert2 | c_in | c_in_g | c_out | d_in | dilation | groups | h_in | input_shape | kd | kh | kw | n | padding | stride | use_bias | w_in | weight_shape | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-k3-s1-bias | 189.4600 | 0.0595 | 676.1080 | 1.7500 | 298.7095 | 0.1777 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_161834_232370_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_2jtkpixx | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 189.460007, 'max_latency_us': 189.460007, 'min_latency_us': 189.460007, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | None | None | 8 | 8 | 16 | 8 | 1 | 1 | 16 | [1, 8, 8, 16, 16] | 3 | 3 | 3 | 1 | 1 | 1 | True | 16 | [16, 8, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
| r3d-stem-bias | 6396.1201 | 0.5840 | 5268.0547 | 1.7500 | 298.3249 | 1.7423 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_161834_232370_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_l7b2nwq0 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 6396.120117, 'max_latency_us': 6396.120117, 'min_latency_us': 6396.120117, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | None | None | 3 | 3 | 64 | 16 | 1 | 1 | 112 | [1, 3, 16, 112, 112] | 3 | 3 | 3 | 1 | 1 | 1 | True | 112 | [64, 3, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | float16 |
| video-down-bias | 85.6000 | 6.2308 | 1049.7212 | 1.7500 | 298.3239 | 18.5881 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_161834_232370_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_was6s6m4 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 85.599998, 'max_latency_us': 85.599998, 'min_latency_us': 85.599998, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | None | None | 64 | 64 | 128 | 8 | 1 | 1 | 56 | [1, 64, 8, 56, 56] | 3 | 3 | 3 | 1 | 1 | 2 | True | 56 | [128, 64, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | float16 |
| unet-encoder-bias | 6425.1001 | 0.7597 | 4887.7935 | 1.7500 | 298.7112 | 2.2694 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_161834_232370_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_i1rgb779 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 6425.100098, 'max_latency_us': 6425.100098, 'min_latency_us': 6425.100098, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | None | None | 32 | 32 | 64 | 32 | 1 | 1 | 64 | [1, 32, 32, 64, 64] | 3 | 3 | 3 | 1 | 1 | 1 | True | 64 | [64, 32, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
