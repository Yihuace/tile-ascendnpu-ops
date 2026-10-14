# Integration Log

## integrate_kernel.py run (2026-10-14T17:38:57+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Expert
- functions: _conv1d_pointwise_kernel, _conv1d_kernel, _conv1d_unit_stride_kernel, _conv1d_group_kernel, _conv1d_direct_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-14T17:58:59+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Expert
- functions: _conv1d_pointwise_kernel, _conv1d_kernel, _conv1d_unit_stride_kernel, _conv1d_group_kernel, _conv1d_direct_kernel
- smoke: pass

## Stage 5 verification (2026-10-14)

- meta: tileops/kernels/convolution/conv1d/.migration_meta.json
- integrated:
    _conv1d_pointwise_kernel      -> conv1d_kernel/_conv1d_pointwise_kernel.py
    _conv1d_kernel                -> conv1d_kernel/_conv1d_kernel.py
    _conv1d_unit_stride_kernel    -> conv1d_kernel/_conv1d_unit_stride_kernel.py
    _conv1d_group_kernel          -> conv1d_kernel/_conv1d_group_kernel.py
    _conv1d_direct_kernel         -> conv1d_kernel/_conv1d_direct_kernel.py
- design_docs (all present, no `missing`):
    _conv1d_pointwise_kernel_DESIGN.md, _conv1d_kernel_DESIGN.md,
    _conv1d_unit_stride_kernel_DESIGN.md, _conv1d_group_kernel_DESIGN.md,
    _conv1d_direct_kernel_DESIGN.md
- wrapper glue (Stage 5 edits, conv1d.py):
    * default_config now carries NPU block tiling (block_m/n/k; taps for
      unit-stride; block_n for depthwise-direct) + `num_stages` placeholder; no
      CUDA `threads`/`enable_rasterization`.
    * forward() builds the npuir workspace operands (w_flat / x_pad|x_gpad /
      xdec / poly_idx / fp32 bias relay ws) from each module's own planner and
      calls the factory; native (c_out, c_in, k) weight (GPU host `permute`
      dropped).
- attempts: 0 (integration verifier passed on the first full report)

### 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261014_174100_695290_Conv1dFwdOp/run.json
- report_md:       examples/TileOPs/reports/tileops/20261014_174100_695290_Conv1dFwdOp/report.md
- correctness: pass (28/28 cases, 0 failed; avg max_abs_err 3.91e-02)
- benchmark: partial — 16 manifest workloads profiled; 0 pytest failures, but
  every msprof profile flagged invalid (`benchmark_passed=false`):
    * `--kernel-name main` matched no device op (device names are
      `_conv1d*_main_mix_aic`), so msprof retried unfiltered
      (`kernel_filter_fallback_count=16`); the unfiltered capture also contains
      the `ZerosLike` ops from the wrapper's workspace allocation
      (`captured_op_count=2..3`), so `_profile_validity` rejects each case
      (`invalid_profile_count=16`). Root cause: harness kernel-name filter +
      workspace-allocation ops, NOT a kernel correctness/perf defect.
    * True per-kernel latency is recoverable from the kept msprof artifacts
      (OpBasicInfo CSVs of the `_conv1d*_main_mix_aic` op), e.g. whisper-conv1
      `_conv1d_unit_stride_main` ~5.5ms, wav2vec2-layer1 `_conv1d_main`
      ~0.59-0.62ms. Reported (invalid) latency column shows the ZerosLike op.
- per-kernel embedded L0 smoke (independent, all PASS):
    _conv1d_pointwise_kernel      7 cases
    _conv1d_kernel               11 cases
    _conv1d_unit_stride_kernel    9 cases
    _conv1d_group_kernel         12 cases
    _conv1d_direct_kernel        10 cases
- guardrails:
    wrapper_rewritten:         true (baseline/perf_opt selection block; GPU
                               `from ._conv1d_fwd_kernels import` replaced)
    already integrated:        true (re-run -> `already integrated`, no duplicate
                               registration/exports)
    module lineage:            correct (wrapper imports `.conv1d_kernel`;
                               integrated modules import only torch/tilelang,
                               no GPU-repo import; GPU `_conv1d_fwd_kernels.py`
                               retained as unused reference)

### 调试历史

- none (correctness and per-kernel smoke passed on the first integrated build).
