# Integration Log -- FloorDivideFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/floor_divide/.migration_meta.json
- op: FloorDivideFwdOp (op_slug=floor_divide, family=elementwise)
- integrated:
  - floor_divide_fwd_kernel -> tileops/kernels/elementwise/floor_divide/floor_divide_kernel/floor_divide_fwd_kernel.py
- design_docs:
  - floor_divide_fwd_kernel -> tileops/kernels/elementwise/floor_divide/floor_divide_kernel/floor_divide_fwd_kernel_DESIGN.md
- ascend_mode: Developer (wrapper `FloorDivideFwdKernel.ascend_mode = "Developer"` synced by
  integrate_kernel.py; matches the source literal `ASCEND_MODE = "Developer"` /
  `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`; script reported
  `wrapper_mode_updated=true`)
- wrapper: tileops/kernels/elementwise/floor_divide/floor_divide.py
  - import rewritten to the baseline/perf_opt switch block; baseline (Stage 3) import
    `from .floor_divide_kernel import coalesce_broadcast_dims, floor_divide_fwd_kernel`
    is ACTIVE, the perf_opt imports are COMMENTED and point at
    `.floor_divide_kernel.perf_opt.floor_divide_fwd_kernel` (no perf_opt/ dir exists yet,
    so baseline is the only valid source). `integrate_kernel.py` reported
    `wrapper_rewritten=true` (first integration; scaffold had no `floor_divide_kernel/` yet).
- attempts: 1 (no debug loop needed; correctness + benchmark passed on the first report run)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "floor_divide", "family": "elementwise", "functions": ["floor_divide_fwd_kernel"],
 "sources": {"floor_divide_fwd_kernel": "examples/floor_divide/floor_divide_fwd_kernel/floor_divide_fwd_kernel.py"},
 "design_docs": {"floor_divide_fwd_kernel": ".../floor_divide_kernel/floor_divide_fwd_kernel_DESIGN.md"},
 "target_dir": ".../tileops/kernels/elementwise/floor_divide/floor_divide_kernel",
 "wrapper": ".../tileops/kernels/elementwise/floor_divide/floor_divide.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": true,
 "reexported": ["coalesce_broadcast_dims", "floor_divide_fwd_kernel"], "smoke": "pass",
 "timestamp": "2026-10-14T10:30:21+0800"}
```

- First integration run captured BOTH names the wrapper imported from
  `._floor_divide_fwd_kernels` (the helper `coalesce_broadcast_dims` plus the kernel
  factory `floor_divide_fwd_kernel`), so the regenerated
  `floor_divide_kernel/__init__.py` re-exports both and the import smoke test passed.
  No idempotency re-run (`--extracted-module floor_divide_kernel`) was needed because this
  was a clean first integration (no pre-existing rewritten wrapper import).

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_103032_909083_FloorDivideFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_103032_909083_FloorDivideFwdOp/report.md
- report_status: passed (run.json `status="passed"`)
- correctness: PASS -- 16/16 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`
  - tests: `test_floor_divide_op` [4096 fp32/fp16/bf16 + 16384 fp32] (4),
    `test_floor_divide_broadcast` [bias-add / row-broadcast / scalar] (3),
    `test_floor_divide_edge_case` [fp32, atol=1e-5] (1),
    `test_floor_divide_rejects_integer_dtype` [int64/int32/int16/int8/uint8/bool] (6),
    `test_floor_divide_rejects_runtime_dtype_mismatch` (1),
    `test_floor_divide_eval_roofline` (1)
- benchmark: PASS -- `summary.benchmark_requested=true`, `benchmark_passed=true`;
  `benchmark_tests=11`, `benchmark_failed=0`, `case_count=9`, `msprof_case_count=9`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
  (9/9 msprof profiles valid, `profiler_valid=true`, `invalid_reason=null`)
- avg max abs error: 0.00e+00 (fp32/fp16/bf16 same-shape cases exact on this run)

### Benchmark numbers (msprof, from run.json `performance.cases`)
| label | dtype | input_shape | other_shape | block_size | latency (us) | ratio (%) | bandwidth (TB/s) |
|---|---|---|---|---:|---:|---:|---:|
| hidden-state-prefill | float16  | [2048,4096]    | [2048,4096] | 1024 | 131.997 | 14.12 | 1.8000 |
| hidden-state-prefill | bfloat16 | [2048,4096]    | [2048,4096] | 1024 | 132.057 | 14.12 | 1.8000 |
| hidden-state-prefill | float32  | [2048,4096]    | [2048,4096] | 1024 | 84.218  | 44.87 | 1.8000 |
| cnn-feat-broadcast   | float16  | [16,256,56,56] | [256,1,1]   | 1024 | 661.647 | 2.70  | 1.8000 |
| cnn-feat-broadcast   | bfloat16 | [16,256,56,56] | [256,1,1]   | 1024 | 511.730 | 4.27  | 1.8000 |
| cnn-feat-broadcast   | float32  | [16,256,56,56] | [256,1,1]   | 1024 | 385.792 | 7.41  | 1.8000 |
| row-broadcast        | float32  | [4096,4096]    | [1,4096]    | 1024 | 163.837 | 24.01 | 1.8000 |
| row-broadcast        | float16  | [4096,4096]    | [1,4096]    | 1024 | 183.136 | 10.85 | 1.8000 |
| row-broadcast        | bfloat16 | [4096,4096]    | [1,4096]    | 1024 | 158.757 | 15.01 | 1.8000 |

All 9 msprof cases valid (`profiler_valid=true`, `invalid_reason=null`); roofline 1.8000 TB/s
(device peak). Performance reported only, not tuned (Stage 4).

## 调试历史
- none. The single report run (attempt 1) passed both the full correctness suite (16/16) and
  the msprof benchmark (9/9 valid); no kernel/wrapper edit or backup was required.

## issues
- none. The clean first integration captured both wrapper re-exports and the report passed on
  the first attempt; no frozen file (`tests/` / `ops/` / `workloads/` / `manifest/`) was
  modified by this Stage 5.
- Note (not a defect): the residual prose comment `# Baseline source (the GPU-extracted
  reference). ...` above the generated baseline/perf_opt block is the wrapper's original
  comment text left in place by `rewrite_wrapper_import`; it is cosmetic and was not edited
  (wrapper edits are restricted to the baseline/perf_opt toggle block).
