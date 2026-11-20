# Integration Log -- BitwiseOrFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/bitwise_or/.migration_meta.json
- op: BitwiseOrFwdOp (op_slug=bitwise_or, family=elementwise)
- integrated:
  - bitwise_or_fwd_kernel -> tileops/kernels/elementwise/bitwise_or/bitwise_or_kernel/bitwise_or_fwd_kernel.py
- design_docs:
  - bitwise_or_fwd_kernel -> tileops/kernels/elementwise/bitwise_or/bitwise_or_kernel/bitwise_or_fwd_kernel_DESIGN.md
- ascend_mode: Developer (wrapper `BitwiseOrFwdKernel.ascend_mode` already "Developer"; source `ASCEND_MODE="Developer"` literal read by integrate_kernel.py)
- wrapper: tileops/kernels/elementwise/bitwise_or/bitwise_or.py
  - import rewritten to baseline/perf_opt switch block; baseline (Stage 3) import ACTIVE,
    perf_opt import COMMENTED, pointing at `.bitwise_or_kernel.perf_opt.bitwise_or_fwd_kernel`
    (no perf_opt/ dir exists yet, so baseline is the only valid source)
  - wrapper already carried the 1-byte storage adapters (`_kernel_dtype_str` /
    `_as_kernel_input` / `_restore_public_dtype`) from Stage 0, so no glue edit was needed
    at Stage 5.
- attempts: 1

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "bitwise_or", "family": "elementwise", "functions": ["bitwise_or_fwd_kernel"],
 "sources": {"bitwise_or_fwd_kernel": "examples/bitwise_or/bitwise_or_fwd_kernel/bitwise_or_fwd_kernel.py"},
 "design_docs": {"bitwise_or_fwd_kernel": ".../bitwise_or_kernel/bitwise_or_fwd_kernel_DESIGN.md"},
 "target_dir": ".../bitwise_or_kernel", "wrapper": ".../bitwise_or.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["bitwise_or_fwd_kernel", "coalesce_broadcast_dims"], "smoke": "pass"}
```

## 验证结果
- report_run_json: examples/TileOPs/reports/verification/20261013_085747_281389_BitwiseOrFwdOp/run.json
- report_md: examples/TileOPs/reports/verification/20261013_085747_281389_BitwiseOrFwdOp/report.md
- report_status: passed
- correctness: PASS -- 14/14 cases passed (0 failed, 0 errors, 0 skipped)
  run.json `summary.correctness_passed=true`, `pass_rate=1.0`
- benchmark: PASS -- `benchmark_tests` 9/9 passed; 8 msprof-valid cases
  (`msprof_case_count=8`, `invalid_profile_count=0`, `profiler_fallback_count=0`)
- avg max abs error: 0.00e+00 (all integer/bool cases exact)

### Benchmark numbers (msprof, from run.json `performance.cases`)
| dtype | input_shape | other_shape | label | block_size | latency (us) | ratio (%) |
|---|---|---|---|---:|---:|---:|
| bool  | [2048,4096]    | [2048,4096] | hidden-state-prefill | 4096 | 21.900  | 42.56 |
| int32 | [2048,4096]    | [2048,4096] | hidden-state-prefill | 2048 | 59.379  | 64.21 |
| int64 | [2048,4096]    | [2048,4096] | hidden-state-prefill | 2048 | 132.897 | 60.94 |
| bool  | [16,256,56,56] | [256,1,1]   | cnn-feat-broadcast    | 4096 | 153.977 | 4.20  |
| int32 | [16,256,56,56] | [256,1,1]   | cnn-feat-broadcast    | 2048 | 178.876 | 16.19 |
| int64 | [16,256,56,56] | [256,1,1]   | cnn-feat-broadcast    | 2048 | 234.275 | 27.93 |
| int32 | [4096,4096]    | [1,4096]    | row-broadcast         | 2048 | 183.556 | 21.75 |
| int64 | [4096,4096]    | [1,4096]    | row-broadcast         | 2048 | 204.296 | 48.52 |

All bandwidth 1.8000 TB/s (device roofline). Performance reported only, not tuned (Stage 4).

## 调试历史
- None. First-attempt clean pass: integration script succeeded (smoke=pass) and the first
  full correctness+benchmark report returned `status=passed` with 14/14 correctness and
  9/9 benchmark tests. No attempt consumed beyond attempt 1.

## Issues / 建议（交回 conductor）
- R-INTEG-1 (proposal, same as BitwiseAnd R-INTEG-1): the wrapper carries two dtype adapters
  (`_as_kernel_input` / `_restore_public_dtype`) purely for the npuir uint8-GM limitation
  (CG-2026-0018). bitwise_or's manifest/test declare a plain `uint8` direct-input param
  (test `test_bitwise_or_op[4096-dtype2]`), unlike bitwise_and where the gap first appeared
  on the bool storage path, so the plain-uint8 restore is exercised here too. A pattern-library
  note on elementwise 1-byte unsigned/bool storage adapters (carry as int8 at the kernel
  boundary, restore public dtype on return) would cover both ops.
- No in-scope issue remains. Kernel unchanged during Stage 5 (integration copy only; wrapper
  glue pre-supplied by Stage 0, no edit needed).

## Report path fix (S5-REPORT-REF, mechanical)
- Prior report was written under `examples/TileOPs/reports/verification/...`, outside the
  gate-required `examples/TileOPs/reports/tileops/` root. Re-ran the single-op report from
  `examples/TileOPs/` with the CLI default `--reports-dir` (reports/tileops); no kernel/wrapper/
  test/manifest change.
- New report: `examples/TileOPs/reports/tileops/20261013_144638_038276_BitwiseOrFwdOp/run.json`
  (status=passed, correctness 14/14, benchmark 9/9).
- `integration_report.json` run_json updated to the above. `statectl gate 5` pass.
