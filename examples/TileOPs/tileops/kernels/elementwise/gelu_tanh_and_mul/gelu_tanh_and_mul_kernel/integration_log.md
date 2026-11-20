# Integration Log -- GeluTanhAndMulFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/.migration_meta.json
- integrated: gelu_tanh_and_mul_fwd_kernel -> gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel.py
- design_docs: gelu_tanh_and_mul_fwd_kernel -> gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel_DESIGN.md
- aggregated __init__.py: gelu_tanh_and_mul_kernel/__init__.py (re-exports gelu_tanh_and_mul_fwd_kernel)
- wrapper: gelu_tanh_and_mul.py rewritten to baseline/perf_opt kernel-source selection block
- attempts: 0 (first integration passed; debug loop not entered)

## Validation Result

- report_run_json: examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/report.md
- report_status: passed
- correctness: pass (11/11 cases, 0 failed; from run.json summary.correctness_passed=true)
- benchmark: pass (7/7; status=passed)
- msprof_case_count=5, profiler_fallback_count=0, kernel_filter_fallback_count=0, invalid_profile_count=0
- max_abs_err: 0.0156 (operator aggregate; bf16 2048x14336=`1.56e-02`, fp16 2048x14336=`3.91e-03`, fp32=0.0)

### Correctness cases (max_abs_err from run.json)

| case | max_abs_err |
|---|---|
| test_gelu_tanh_and_mul_op[16-128-dtype0] (fp32) | 4.77e-07 |
| test_gelu_tanh_and_mul_op[16-128-dtype1] (fp16) | 3.05e-05 |
| test_gelu_tanh_and_mul_op[16-128-dtype2] (bf16) | 0.00e+00 |
| test_gelu_tanh_and_mul_op[2048-14336-dtype3] (fp16) | 3.91e-03 |
| test_gelu_tanh_and_mul_op[2048-14336-dtype4] (bf16) | 1.56e-02 |
| test_gelu_tanh_and_mul_op[1-14336-dtype5] (bf16) | 0.00e+00 |
| test_gelu_tanh_and_mul_rejects_non_float_dtype | n/a (negative) |
| test_gelu_tanh_and_mul_kernel_rejects_non_float_dtype | n/a (negative) |
| test_gelu_tanh_and_mul_rejects_dtype_mismatch | n/a (negative) |
| test_gelu_tanh_and_mul_rejects_shape_mismatch | n/a (negative) |
| test_gelu_tanh_and_mul_eval_roofline | n/a |

### Per-shape performance (msprof, kernel_name=main)

| label | M | N | dtype | latency_us | bandwidth_tb_s | roofline_util_% | profiler_valid |
|---|---|---|---|---|---|---|---|
| llama-8b-ffn-prefill | 2048 | 14336 | float16 | 215.62 | 1.75 | 47.83 | true |
| llama-8b-ffn-prefill | 2048 | 14336 | bfloat16 | 220.22 | 1.75 | 49.73 | true |
| llama-8b-ffn-decode | 1 | 14336 | bfloat16 | 2.18 | 1.75 | 2.46 | true |
| qwen3-8b-swiglu-b32 | 32 | 12288 | float16 | 5.86 | 1.75 | 23.59 | true |
| qwen3-8b-swiglu-b32 | 32 | 12288 | bfloat16 | 6.02 | 1.75 | 24.38 | true |

## Debug History

- none (attempt 1 passed; 0 fixes).

## Guardrails (§2.1) -- raw results

```
$ grep -n "wrapper_rewritten" tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/integration_log.md
5:- wrapper_rewritten: true
$ grep -n "already integrated\|wrapper_rewrite_action" .../integration_log.md
6:- wrapper_rewrite_action: rewritten
$ python -c "import importlib; w=importlib.import_module('tileops.kernels.elementwise.gelu_tanh_and_mul.gelu_tanh_and_mul'); print(w.gelu_tanh_and_mul_fwd_kernel.__module__)"
tileops.kernels.elementwise.gelu_tanh_and_mul.gelu_tanh_and_mul_kernel.gelu_tanh_and_mul_fwd_kernel
```

- Guardrail 1 `wrapper_rewritten`: **true** (PASS).
- Guardrail 2 idempotent-rerun marker: `already integrated` NOT hit (`wrapper_rewrite_action: rewritten`);
  fresh integration, no manual review needed (PASS).
- Guardrail 3 baseline lineage: `__module__` =
  `tileops.kernels.elementwise.gelu_tanh_and_mul.gelu_tanh_and_mul_kernel.gelu_tanh_and_mul_fwd_kernel`;
  `perf_opt` not in path -> baseline (not perf_opt) lineage (PASS).

> Note: the §2.1 command imports `tileops.kernels.<family>.<op_slug>` and reads `w.<func>.__module__`;
> here the function is re-exported only through the `{op_slug}_kernel` subpackage / wrapper, so the
> literal command raises `AttributeError`. The intended check is run against the wrapper module
> (shown above) -- still resolves to the baseline module. No glue/package change made.

## Wrapper glue

- No manual glue fix required. Stage-3 factory signature
  `gelu_tanh_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=2048, num_cores=None)` is
  invoked by the wrapper as `gelu_tanh_and_mul_fwd_kernel(self.M, self.N, self.dtype_str)(x)`
  (single `[M, 2N]` tensor input, `out_idx=[1]` output). `default_config={}` (K11) confirmed.

## integrate_kernel.py run (2026-10-13T22:45:18+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: gelu_tanh_and_mul_fwd_kernel
- smoke: pass
