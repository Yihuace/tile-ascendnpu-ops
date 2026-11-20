# Integration Log -- BatchNormBwdOp

- op: BatchNormBwdOp (batch_norm_bwd, family=norm)
- meta: tileops/kernels/norm/batch_norm_bwd/.migration_meta.json
- integrated:
  - _batch_norm_bwd_wide_kernel -> batch_norm_bwd_kernel/_batch_norm_bwd_wide_kernel.py
  - _batch_norm_bwd_split_kernel -> batch_norm_bwd_kernel/_batch_norm_bwd_split_kernel.py
  - _batch_norm_bwd_kernel -> batch_norm_bwd_kernel/_batch_norm_bwd_kernel.py
- design_docs:
  - _batch_norm_bwd_wide_kernel -> _batch_norm_bwd_wide_kernel_DESIGN.md
  - _batch_norm_bwd_split_kernel -> _batch_norm_bwd_split_kernel_DESIGN.md
  - _batch_norm_bwd_kernel -> _batch_norm_bwd_kernel_DESIGN.md
- ascend_mode: Expert (all three source products declare `ASCEND_MODE = "Expert"`)
- attempts: 0

## Stage 5 glue (wrapper)

The wrapper `tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` is the
integration glue; it was adapted before running `integrate_kernel.py`:

- Dropped `VECTOR_ACCESS_BYTES` from the GPU extracted-file import and defined it
  locally (`VECTOR_ACCESS_BYTES = 16`); the migrated npuir kernels own their
  vector width internally and do not export the GPU helper constant.
- `_BatchNormBwdBase.ascend_mode`: `"Developer"` -> `"Expert"` (all three
  kernels are Expert-form: `T.alloc_ub` + single `T.Scope("Vector")`). The
  integrate script additionally pinned `BatchNormBwdKernel.ascend_mode = "Expert"`.
- `BatchNormBwdWideKernel.forward`: the NPU wide kernel uses
  `out_idx=[7, 5, 6]` and returns `(grad_x, grad_weight, grad_bias)`; the ported
  GPU call passed preallocated channel grads and took a single output, so the
  call was corrected to `grad_x, grad_weight, grad_bias = program(grad_out, x,
  weight, mean, rstd)`.
- `BatchNormBwdSplitKernel.forward` / `BatchNormBwdKernel.forward` already match
  the npuir factory contracts (split triple `(stats, finalize, apply)`, general
  `out_idx=[-1]` with host-preallocated channel grads) and were left unchanged.

## Dispatch logic (by call config)

`select_bwd_kernel(n, c, spatial, dtype)` (host, ported holder policy):

- holder == "thread" or "block" -> `bwd_wide` (`BatchNormBwdWideKernel`,
  register-held one channel per block).
- holder == "split" (block does not hold the channel and `c < 1024`,
  `N*S >= 65536`) -> `bwd_split` (`BatchNormBwdSplitKernel`, channel cut across
  blocks: stats -> finalize -> apply).
- otherwise ("tiled") -> `bwd_kernel` (`BatchNormBwdKernel`, general serial
  two-scan path).

The op `BatchNormBwdOp.default_kernel_map` maps those keys to the three classes;
`BatchNormBwdOp.forward` selects the key from the `(N, C, S)` call shape and
caches the kernel per `(key, n, c, s, dtype)`.

## integrate_kernel.py run (2026-10-14T20:25:27+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Expert
- functions: _batch_norm_bwd_wide_kernel, _batch_norm_bwd_split_kernel, _batch_norm_bwd_kernel
- smoke: pass

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_202544_230090_BatchNormBwdOp/report.md
- correctness: pass (12/12 cases, 0 failed; avg max_abs_err 2.29e-03, worst
  case max_abs_err 1.56e-02 on the bf16 resnet-stage2 shape)
- benchmark: partial. 6 msprof workloads ran with 0 pytest failures; 5/6
  produced valid per-shape latencies (wide/general kernels named `main`), 1/6
  (large-spatial `[4,128,262144]`, the split path) triggered a kernel-name
  filter fallback: `--kernel-name main` matched no device op because the split
  prim_funcs are named `stats`/`finalize`/`apply`, so msprof retried without a
  filter and captured all three ops (`apply_mix_aic` + `finalize_mix_aic` +
  `stats_mix_aic`, sample_count=3). That record is flagged invalid
  (`kernel_filter_fallback_count=1`, `invalid_profile_count=1`). Root cause is
  the harness kernel-name filter + a 3-stage split kernel, not a kernel defect
  (CG-2026-0025).

| workload | dtype | kernel | latency_us (valid) |
|---|---|---:|---:|
| resnet-stage1 (8,64,1024) | float16 | wide (`main_mix_aic`) | 20.78 |
| resnet-stage1 (8,64,1024) | bfloat16 | wide (`main_mix_aic`) | 21.10 |
| resnet-stage2 (4,128,1024) | float16 | wide (`main_mix_aic`) | 24.98 |
| resnet-stage3 (4,256,784) | float16 | wide (`main_mix_aic`) | 40.42 |
| resnet-stage3 (4,256,784) | float32 | wide (`main_mix_aic`) | 41.50 |
| large-spatial (4,128,262144) | float16 | split (3 ops) | 2085.34 (invalid profile) |

## 调试历史（若有）

- none (attempts=0; no kernel/wrapper fix was needed after the pre-run glue
  corrections above).

## Guardrails

1. wrapper_rewritten: true. No GPU CUDA execution knob is baked into the NPU
   wrapper: `default_config` returns `{}` for all three classes, the factories
   are invoked with no `threads` / `num_per_thread` / `block_l` argument, and
   no `_require_cuda` / `vector_aligned` GPU helper remains. `_block_launch`
   computes a `(threads, num_per_thread)` tuple only as the host holder-policy
   heuristic (`select_bwd_kernel`); that tuple is never forwarded to a kernel.
2. already integrated: true. A second run of `integrate_kernel.py` reported
   `[wrapper] ... already integrated`, `[mode] ... Expert (unchanged)`,
   `[smoke] pass`; the baseline import appears exactly once, so there is no
   double-registration of the kernel source block.
3. module lineage: correct. The wrapper no longer imports the GPU extracted
   file `._batch_norm_bwd_kernels` (only a historical docstring mention remains)
   and instead imports `from .batch_norm_bwd_kernel import` the three migrated
   factories; the integrated modules carry `target="npuir"` `@tilelang.jit`
   builders; the op imports `select_bwd_kernel` from
   `tileops.kernels.norm.batch_norm_bwd.batch_norm_bwd`.
