# Integration Log -- EqFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/eq/.migration_meta.json
- op: EqFwdOp (op_slug=eq, family=elementwise)
- integrated:
  - eq_fwd_kernel -> tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel.py
- design_docs:
  - eq_fwd_kernel -> tileops/kernels/elementwise/eq/eq_kernel/eq_fwd_kernel_DESIGN.md
- ascend_mode: Developer (wrapper `EqFwdKernel.ascend_mode = "Developer"` matches the
  source literal `ASCEND_MODE = "Developer"` / `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`;
  integrate_kernel.py reported `updated=false`)
- wrapper: tileops/kernels/elementwise/eq/eq.py
  - import already in baseline/perf_opt switch-block form (from a prior partial assembly);
    baseline (Stage 3) import `from .eq_kernel import coalesce_broadcast_dims, eq_fwd_kernel`
    is ACTIVE, perf_opt imports are COMMENTED and point at
    `.eq_kernel.perf_opt.eq_fwd_kernel` (no perf_opt/ dir exists yet, so baseline is the
    only valid source). `integrate_kernel.py` reported `wrapper_rewritten=false` (idempotent).
- attempts: 1 (no debug loop needed; correctness + benchmark passed on the first report run)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "eq", "family": "elementwise", "functions": ["eq_fwd_kernel"],
 "sources": {"eq_fwd_kernel": "examples/eq/eq_fwd_kernel/eq_fwd_kernel.py"},
 "design_docs": {"eq_fwd_kernel": ".../eq_kernel/eq_fwd_kernel_DESIGN.md"},
 "target_dir": ".../tileops/kernels/elementwise/eq/eq_kernel",
 "wrapper": ".../tileops/kernels/elementwise/eq/eq.py",
 "wrapper_rewritten": false, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["coalesce_broadcast_dims", "eq_fwd_kernel"], "smoke": "pass",
 "timestamp": "2026-10-13T18:36:39+0800"}
```

### Integration note -- integrate_kernel.py re-run dropped a re-export (R-type, no frozen file touched)
- First invocation `integrate_kernel.py --meta <meta>` (as prescribed) regenerated
  `eq_kernel/__init__.py` from `wrapper_names = parse_wrapper_imports(wrapper, "_eq_fwd_kernels")`.
  Because the wrapper had *already* been rewritten in a prior partial assemble (its import now
  points at `.eq_kernel`, not the original `._eq_fwd_kernels`), the parse returned nothing and the
  script fell back to `functions = ["eq_fwd_kernel"]`. The regenerated `__init__.py` therefore
  stopped re-exporting `coalesce_broadcast_dims`, and the import smoke test FAILED
  (`ImportError: cannot import name 'coalesce_broadcast_dims'`).
- Resolution (within scope: only the generated integration-package `__init__.py` was regenerated,
  no frozen file touched): re-ran
  `integrate_kernel.py --meta <meta> --extracted-module eq_kernel`.
  `--extracted-module eq_kernel` makes `parse_wrapper_imports` read the wrapper's *current*
  baseline import (`from .eq_kernel import coalesce_broadcast_dims, eq_fwd_kernel`), so the
  regenerated `__init__.py` re-exports both names and `[smoke]` passes
  (`reexported=["coalesce_broadcast_dims","eq_fwd_kernel"]`).
- Root cause / proposal: `integrate_kernel.py` is not fully idempotent when the wrapper import has
  already been rewritten -- the fallback `wrapper_names = list(functions)` silently loses the
  helper re-exports that the first run captured. Suggested fix: when `parse_wrapper_imports`
  finds nothing, also scan the existing `.{op_slug}_kernel` import line (or the currently generated
  `__init__.py`) before falling back to `functions`. Reported in `issues` below; no frozen file
  was modified by this Stage 5.

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261013_183657_078339_EqFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_183657_078339_EqFwdOp/report.md
- report_status: passed (run.json `status="passed"`)
- correctness: PASS -- 18/18 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`
  - dtypes covered: float32/float16/bfloat16 same-shape, 3 broadcast patterns, edge case,
    uint8/int8/int16/int32/int64 integer union, torch.bool fast path, unsupported-dtype
    rejection (complex64/fp8), roofline formula
- benchmark: PASS -- `summary.benchmark_requested=true`, `benchmark_passed=true`;
  `benchmark_tests=11`, `benchmark_failed=0`, `msprof_case_count=9`, `case_count=9`,
  `profiler_fallback_count=0`, `invalid_profile_count=0` (9/9 msprof profiles valid)
- avg max abs error: 0.00e+00 (bool output, exact compare)

### Benchmark numbers (msprof, from run.json `performance.cases`)
| label | dtype | input_shape | other_shape | block_size | latency (us) | ratio (%) | bandwidth (TB/s) |
|---|---|---|---:|---:|---:|---:|---:|
| hidden-state-prefill | float16  | [2048,4096]      | [2048,4096] | 1024 | 83.098  | 22.43 | 1.8000 |
| hidden-state-prefill | bfloat16 | [2048,4096]      | [2048,4096] | 1024 | 90.918  | 20.50 | 1.8000 |
| hidden-state-prefill | float32  | [2048,4096]      | [2048,4096] | 1024 | 94.598  | 39.41 | 1.8000 |
| cnn-feat-broadcast   | float16  | [16,256,56,56]   | [256,1,1]   | 1024 | 580.468 | 2.13  | 1.8000 |
| cnn-feat-broadcast   | bfloat16 | [16,256,56,56]   | [256,1,1]   | 1024 | 661.727 | 2.66  | 1.8000 |
| cnn-feat-broadcast   | float32  | [16,256,56,56]   | [256,1,1]   | 1024 | 365.893 | 7.48  | 1.8000 |
| row-broadcast        | float32  | [4096,4096]      | [1,4096]    | 1024 | 166.077 | 22.46 | 1.8000 |
| row-broadcast        | float16  | [4096,4096]      | [1,4096]    | 1024 | 161.017 | 11.58 | 1.8000 |
| row-broadcast        | bfloat16 | [4096,4096]      | [1,4096]    | 1024 | 185.336 | 10.06 | 1.8000 |

All 9 msprof cases valid (`profiler_valid=true`, `invalid_reason=null`); roofline 1.8000 TB/s
(device peak). Performance reported only, not tuned (Stage 4).

## 调试历史
- none. The single report run (attempt 1) passed both the full correctness suite and the msprof
  benchmark; no kernel/wrapper edit or backup was required.
- Pre-report integration iteration (not a report failure, resolved before the report run):
  `integrate_kernel.py` re-run re-export loss, fixed by re-running with `--extracted-module eq_kernel`
  as documented above. No frozen file (`tests/`, `ops/`, `workloads/`, `manifest/`) was modified.

## issues
- `integrate_kernel.py` idempotency defect (R-type): re-running the script after the wrapper import
  has been rewritten silently drops helper re-exports from the regenerated `{op_slug}_kernel/__init__.py`
  (details above). Worked around with `--extracted-module eq_kernel`; suggestion: fall back to the
  existing `.{op_slug}_kernel` import line / current `__init__.py` instead of `functions`.
- No frozen-file (tests/ops/workloads/manifest) problem was observed for EqFwdOp.
