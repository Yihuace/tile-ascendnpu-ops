# NPU Benchmark Report
Generated: 2026-11-08 02:32:04

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa-sw-varlen-fwd-smoke | 58.9612 | 0.4869 | 653.4956 | 1.7500 | 299.2613 | 1.4570 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_o4ihs737 | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 58.961178, 'max_latency_us': 58.961178, 'min_latency_us': 58.961178, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 2 | 64 | 8 | 2 | True | [128, 128] | [128, 128] | -1 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | float16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:34:28

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa-sw-varlen-fwd-smoke | 58.1212 | 0.4961 | 656.4557 | 1.7500 | 299.2717 | 1.4847 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_3iow1d7_ | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 58.121162, 'max_latency_us': 58.121162, 'min_latency_us': 58.121162, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 2 | 64 | 8 | 2 | True | [128, 128] | [128, 128] | -1 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:36:51

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-short-w256 | 314.8263 | 14.0492 | 1046.9857 | 1.7500 | 299.4118 | 42.0649 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic__c1txjgg | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 314.826294, 'max_latency_us': 314.826294, 'min_latency_us': 314.826294, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [512, 512, 512, 512] | [512, 512, 512, 512] | 256 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | float16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:39:16

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-short-w256 | 308.9062 | 14.3315 | 1047.9650 | 1.7500 | 299.4177 | 42.9112 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_8wrccqc_ | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 308.906189, 'max_latency_us': 308.906189, 'min_latency_us': 308.906189, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [512, 512, 512, 512] | [512, 512, 512, 512] | 256 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:41:40

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-long-w1024 | 1774.7355 | 24.7343 | 2351.4312 | 1.7500 | 299.4277 | 74.0612 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_djj7v1kb | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 1774.735474, 'max_latency_us': 1774.735474, 'min_latency_us': 1774.735474, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [2048, 2048, 2048, 2048] | [2048, 2048, 2048, 2048] | 1024 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | float16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:44:02

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-long-w1024 | 1787.2557 | 24.5835 | 2266.0259 | 1.7500 | 299.4350 | 73.6117 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_fhts9p67 | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 1787.255737, 'max_latency_us': 1787.255737, 'min_latency_us': 1787.255737, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 32 | 8 | True | [2048, 2048, 2048, 2048] | [2048, 2048, 2048, 2048] | 1024 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:46:25

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-70b-short-w256 | 570.7314 | 15.4992 | 1258.9797 | 1.7500 | 299.4133 | 46.4066 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_oxala2to | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 570.731384, 'max_latency_us': 570.731384, 'min_latency_us': 570.731384, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 64 | 8 | True | [512, 512, 512, 512] | [512, 512, 512, 512] | 256 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | float16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:48:48

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-70b-short-w256 | 569.8914 | 15.5362 | 1260.1573 | 1.7500 | 299.4192 | 46.5185 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_53pu_yuj | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 569.891418, 'max_latency_us': 569.891418, 'min_latency_us': 569.891418, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 64 | 8 | True | [512, 512, 512, 512] | [512, 512, 512, 512] | 256 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:51:10

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-70b-long-w1024 | 3490.6899 | 25.1507 | 1579.6118 | 1.7500 | 299.4279 | 75.3082 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic_kcgmepup | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 3490.689941, 'max_latency_us': 3490.689941, 'min_latency_us': 3490.689941, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 64 | 8 | True | [2048, 2048, 2048, 2048] | [2048, 2048, 2048, 2048] | 1024 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | float16 |


# NPU Benchmark Report
Generated: 2026-11-08 02:53:34

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GroupedQueryAttentionSlidingWindowVarlenFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | dim | heads | heads_kv | is_causal | seqlens_k | seqlens_q | wl | wr | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-70b-long-w1024 | 3508.1501 | 25.0485 | 1571.2197 | 1.7500 | 299.4352 | 75.0040 | /mnt/workspace/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/artifacts/msprof/_gqa_sw_fwd_varlen_main_mix_aic__5d3fwty | {'captured_op_count': 1, 'captured_op_names': ['_gqa_sw_fwd_varlen_main_mix_aic'], 'kernel_name_attempts': ['_gqa_sw_fwd_varlen_main'], 'kernel_name_requested': '_gqa_sw_fwd_varlen_main', 'kernel_name_resolved': '_gqa_sw_fwd_varlen_main', 'latency_us': 3508.150146, 'max_latency_us': 3508.150146, 'min_latency_us': 3508.150146, 'op_name': '_gqa_sw_fwd_varlen_main_mix_aic', 'sample_count': 1} | msprof | 4 | 128 | 64 | 8 | True | [2048, 2048, 2048, 2048] | [2048, 2048, 2048, 2048] | 1024 | -1 | {'block_m': 64, 'block_n': 64, 'num_stages': 1} | bfloat16 |
