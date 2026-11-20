# Integration Log -- BitwiseAndFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/bitwise_and/.migration_meta.json
- op: BitwiseAndFwdOp (op_slug=bitwise_and, family=elementwise)
- integrated:
  - bitwise_and_fwd_kernel -> tileops/kernels/elementwise/bitwise_and/bitwise_and_kernel/bitwise_and_fwd_kernel.py
- design_docs:
  - bitwise_and_fwd_kernel -> tileops/kernels/elementwise/bitwise_and/bitwise_and_kernel/bitwise_and_fwd_kernel_DESIGN.md
- ascend_mode: Developer (wrapper Kernel.ascend_mode synced; source ASCEND_MODE="Developer" literal read by integrate_kernel.py)
- wrapper: tileops/kernels/elementwise/bitwise_and/bitwise_and.py
  - import rewritten to baseline/perf_opt switch block; baseline (Stage 3) import ACTIVE,
    perf_opt import COMMENTED, pointing at `.bitwise_and_kernel.perf_opt.bitwise_and_fwd_kernel`
    (no perf_opt/ dir exists yet, so baseline is the only valid source)
- attempts: 2

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "bitwise_and", "family": "elementwise", "functions": ["bitwise_and_fwd_kernel"],
 "sources": {"bitwise_and_fwd_kernel": "examples/bitwise_and/bitwise_and_fwd_kernel/bitwise_and_fwd_kernel.py"},
 "design_docs": {"bitwise_and_fwd_kernel": ".../bitwise_and_kernel/bitwise_and_fwd_kernel_DESIGN.md"},
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["bitwise_and_fwd_kernel", "coalesce_broadcast_dims"], "smoke": "pass"}
```

## 验证结果
- report_run_json: examples/TileOPs/reports/verification/20261013_084022_118814_BitwiseAndFwdOp/run.json
- report_md: examples/TileOPs/reports/verification/20261013_084022_118814_BitwiseAndFwdOp/report.md
- report_status: passed
- correctness: PASS -- 14/14 cases passed (0 failed, 0 errors, 0 skipped)
  run.json `summary.correctness_passed=true`, `pass_rate=1.0`
- benchmark: PASS -- `benchmark_tests` 9/9 passed; 8 msprof-valid cases
  (`msprof_case_count=8`, `invalid_profile_count=0`, `profiler_fallback_count=0`)
- avg max abs error: 0.00e+00 (all integer/bool cases exact)

### Benchmark numbers (msprof, from run.json `performance.cases`)
| dtype | input_shape | other_shape | label | block_size | latency (us) | ratio (%) |
|---|---|---|---:|---:|---:|---:|
| bool  | [2048,4096] | [2048,4096] | hidden-state-prefill | 4096 | 22.300 | 41.80 |
| int32 | [2048,4096] | [2048,4096] | hidden-state-prefill | 2048 | 53.739 | 69.92 |
| int64 | [2048,4096] | [2048,4096] | hidden-state-prefill | 2048 | 128.397 | 62.38 |
| bool  | [16,256,56,56] | [256,1,1] | cnn-feat-broadcast | 4096 | 152.057 | 4.26 |
| int32 | [16,256,56,56] | [256,1,1] | cnn-feat-broadcast | 2048 | 168.937 | 16.90 |
| int64 | [16,256,56,56] | [256,1,1] | cnn-feat-broadcast | 2048 | 233.275 | 29.03 |
| int32 | [4096,4096] | [1,4096] | row-broadcast | 2048 | 162.837 | 24.40 |
| int64 | [4096,4096] | [1,4096] | row-broadcast | 2048 | 204.416 | 49.02 |

All bandwidth 1.8000 TB/s (device roofline). Performance reported only, not tuned (Stage 4).

## 调试历史
### attempt 1 — wrapper dtype adapter (completed in prior session)
- 症状: `test_bitwise_and_bool_fast_path[a_shape0]` / `[a_shape1]` failed with
  `ValueError: uint8 GM tensors are not supported on npuir (CG-2026-0018)`.
- 分类: wrapper 胶水错误 (dtype 转换) -- the frozen Op bool-storage fast path views `torch.bool`
  as `torch.uint8` and hands the kernel `torch.uint8`; the integrated NPU kernel rejects uint8.
- 修复: DESIGN.md section 0.6 R3 (a declared Stage-3/5 integration item): carry the 1-byte
  bool/uint8 storage as `int8` at the kernel boundary.
  wrapper `bitwise_and.py`: added `_kernel_dtype_str()` (uint8/bool -> "int8") and
  `_as_kernel_input()` (`tensor.view(torch.int8)`); `BitwiseAndFwdKernel.__init__` passes
  `self.kernel_dtype_str`; `forward` reinterprets uint8/bool as int8 (bit-identical).
  - backup: bitwise_and_kernel/history_version/bitwise_and_s5_attempt1.py
- 结果: bool fast-path both shapes PASS (9/14 -> 11/14).

### attempt 2 — uint8/int8 output restore (wrapper only)
- 症状: `test_bitwise_and_op[4096-dtype2]` (dtype=uint8) failed
  `AssertionError: output does not exactly match reference`
  (`int8` output `-127` vs `uint8` reference `129` -- identical bit pattern 0x81, different dtype).
- 分类: wrapper 胶水错误 (dtype 转换) -- after attempt 1, the kernel returned the int8 carrier
  directly; the plain uint8 path has no later `view(torch.uint8)`, so the frozen Op's
  same-dtype-in/out contract (and `exact_compare`) saw int8 instead of uint8.
- 修复 (wrapper glue only): added `_restore_public_dtype(tensor, dtype)` and applied it in
  `BitwiseAndFwdKernel.forward`, reinterpreting the int8 carrier back to the public 1-byte
  dtype (`view`, bit-identical). The bool fast path is unaffected: Op still does its own
  `result.view(torch.bool)` on the restored uint8.
  - backup: bitwise_and_kernel/history_version/bitwise_and_s5_attempt2.py (pre-edit state)
- 结果 (this run): 14/14 correctness PASS, benchmark PASS.
- Note: the Stage 0 test/workload dtype defect (`tests/ops/test_bitwise_and.py` +
  `tileops/workloads/elementwise.py` not wiring `dtype` into input generation) was fixed
  out-of-band (frozen files, 2026-10-12 19:26); with it and attempt 2, all dtype params
  (int32/int64/uint8/int16) pass. No frozen file was touched by this Stage 5.

## Issues / 建议（交回 conductor）
- R-TEST-1 (RESOLVED out-of-band): `tests/ops/test_bitwise_and.py::test_bitwise_and_op` now
  wires its `dtype` fixture param into `BitwiseTest(n_total, dtype)`.
- R-TEST-2 (RESOLVED out-of-band): `tileops/workloads/elementwise.py::BitwiseWorkload` is now
  dtype-parametrized (generates `self.dtype` tensors; integer window clamped to the dtype range).
- R-INTEG-1 (proposal): the four `test_bitwise_and_op` dtype params are dominated by the
  uint8/int8 boundary cases; the wrapper carries two adapters (`_as_kernel_input` /
  `_restore_public_dtype`) purely for the npuir uint8-GM limitation (CG-2026-0018). Worth a
  pattern-library note on elementwise 1-byte unsigned/bool storage adapters.
- No in-scope issue remains. Kernel unchanged during Stage 5 (only wrapper glue edited).

## Report path fix (S5-REPORT-REF, mechanical)
- Prior report was written under `examples/TileOPs/reports/verification/...`, outside the
  gate-required `examples/TileOPs/reports/tileops/` root. Re-ran the single-op report from
  `examples/TileOPs/` with the CLI default `--reports-dir` (reports/tileops); no kernel/wrapper/
  test/manifest change.
- New report: `examples/TileOPs/reports/tileops/20261013_143914_610878_BitwiseAndFwdOp/run.json`
  (status=passed, correctness 14/14, benchmark 9/9).
- `integration_report.json` run_json updated to the above. `statectl gate 5` pass.
