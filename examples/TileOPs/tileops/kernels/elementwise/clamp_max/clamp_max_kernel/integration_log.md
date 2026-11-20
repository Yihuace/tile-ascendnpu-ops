# Integration Log -- ClampMaxFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/clamp_max/.migration_meta.json
- op: ClampMaxFwdOp (op_slug=clamp_max, family=elementwise)
- integrated:
  - _make_clamp_tensor_kernel -> tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel.py
    (copied from examples/clamp_max/_make_clamp_tensor_kernel/_make_clamp_tensor_kernel.py, 35582 B;
    `cmp` byte-identical)
- design_docs:
  - _make_clamp_tensor_kernel -> tileops/kernels/elementwise/clamp_max/clamp_max_kernel/_make_clamp_tensor_kernel_DESIGN.md
    (snapshot of examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md, byte-identical, 95889 B;
    verified by `cmp`)
- ascend_mode: Developer. The Stage 3 kernel carries both the module-level literal
  `ASCEND_MODE = "Developer"` (line 98) and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`
  (line 76); integrate_kernel.py reported `ascend_mode="Developer"`,
  `wrapper_mode_updated=false` (the wrapper `ClampTensorFwdKernel.ascend_mode = "Developer"` already
  matched). Single extracted function -> no mixed-mode conflict.
- wrapper: tileops/kernels/elementwise/clamp_max/clamp_max.py
  - Already in the baseline/perf_opt source-selection form from the previous Stage 5 run
    (`wrapper_rewritten=false` on this re-run). Baseline (Stage 3) is ACTIVE:
    `from .clamp_max_kernel import _make_clamp_tensor_kernel`; the perf_opt import is COMMENTED and
    points at `.clamp_max_kernel.perf_opt._make_clamp_tensor_kernel` (no perf_opt/ dir exists yet, so
    baseline is the only valid source, as prescribed). No wrapper edit on this run.
- attempts: 1 in the editable-scope debug loop (attempt_index=2 of max_attempts=5 for the harness).
  No kernel/wrapper edit was needed: with the frozen Stage 0 test fixture repaired out-of-band, the
  single re-run report passed full correctness; the benchmark ran (msprof fell back to events —
  recorded, not fixed, per the Stage 5 rule).

## integrate_kernel.py summary (`[json]` line, re-run 2026-10-15)
```json
{"op_slug": "clamp_max", "family": "elementwise", "functions": ["_make_clamp_tensor_kernel"],
 "sources": {"_make_clamp_tensor_kernel": "/mnt/workspace/tilelang-mlir-ascend/examples/clamp_max/_make_clamp_tensor_kernel/_make_clamp_tensor_kernel.py"},
 "design_docs": {"_make_clamp_tensor_kernel": "<target>/clamp_max_kernel/_make_clamp_tensor_kernel_DESIGN.md"},
 "target_dir": "examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max_kernel",
 "wrapper": "examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max.py",
 "wrapper_rewritten": false, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["_make_clamp_tensor_kernel"], "smoke": "pass",
 "timestamp": "2026-10-15T11:53:43+0800"}
```
- Idempotent re-run invoked as
  `python examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta> --extracted-module clamp_max_kernel`.
  The `--extracted-module clamp_max_kernel` flag is REQUIRED on a re-run: the wrapper no longer imports
  from `._clamp_max_fwd_kernels` (it was rewritten in the previous Stage 5 run), so without the flag
  `parse_wrapper_imports` would find no names. With it, `wrapper_names` resolves from the active
  baseline import `from .clamp_max_kernel import _make_clamp_tensor_kernel`.
- `[smoke] pass` (import of `tileops.kernels.elementwise.clamp_max.clamp_max_kernel` + re-export
  `_make_clamp_tensor_kernel` + `ClampTensorFwdKernel` attribute check). No `[error]`/`[warn]`.
- Kernel + DESIGN copies overwritten byte-identically (`cmp` clean); aggregation `__init__.py`
  regenerated; wrapper left intact.

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261015_121037_823687_ClampMaxFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261015_121037_823687_ClampMaxFwdOp/report.md
- (superseded earlier run of this Stage 5: 20261015_115446_584731_ClampMaxFwdOp — same verdict;
  superseded only because the idempotent integration re-run bumped the kernel/`__init__` mtimes, so a
  fresh report was required by gate S5-REPORT-STALE)
- report_status: partial (run.json `status="partial"`; correctness passed, benchmark ran but its
  msprof profiles were invalid, so `summary.benchmark_passed=false`). The CLI exits nonzero for
  `partial`; this does NOT mean integration failed.
- correctness: PASS -- 15/15 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`, `correctness_tests=15`, `correctness_failed=0`.
  - Includes the previously-failing `tests.ops.test_clamp_max::test_clamp_max_broadcast[input_shape3-max_shape3]`
    = `_BROADCAST_PATTERNS[3] = ((4, 8), ())` ("0-dim upper bound"), which now PASSES after the
    out-of-band frozen-test fix.
  - Full case set: 8 `test_clamp_max_op` (`input_shape0..3` x `max_shape0..3` x dtype, float32/
    float16/bfloat16 same-shape across workloads), 4 `test_clamp_max_broadcast` (`input_shape0..3`),
    `test_clamp_max_init_signature_pytorch_aligned`, 2 `test_clamp_max_nan_propagation`,
    3 `test_clamp_max_rejects_unsupported_dtype`, `test_clamp_max_eval_roofline`.
  - Avg max abs error: 0.00e+00 (bit-exact on the passing elementwise cases).
- benchmark: PARTIAL -- `summary.benchmark_requested=true`, `benchmark_passed=false`;
  `benchmark_tests=11`, `benchmark_failed=0`, `case_count=9`, `msprof_case_count=0`,
  `profiler_fallback_count=9`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`.
  - All 9 tileops benchmark records fell back from msprof to **events** timing (`prof_mode="events"`,
    `profiler_valid=true` for events). Root cause is in the profiler capture layer, NOT the kernel:
    the generated `msprof op` commands for the tileops functor did not produce an `OpBasicInfo*.csv`
    (both `--kernel-name=main` and the no-filter retry failed -> `msprof op failed to produce
    OpBasicInfo CSV data` -> `RuntimeError` -> `_bench_with_mode` events fallback). The paired
    `torch.minimum` reference records on the SAME run got valid msprof captures
    (`prof_mode="msprof"`, `captured_op_count=1`), so the device/profiler itself is healthy.
  - Reported only (Stage 5 rule: benchmark/profile invalidity is recorded, never fixed/retried).
    Wall time 479.24s for the benchmark stage (11 passed, 1 deselected).

### Benchmark numbers (tileops event-mode, from run.json `performance.cases`, tag=tileops)
| label | dtype | input_shape | max_shape | block_size | latency (us) | bandwidth (TB/s) | prof_mode | profiler_valid |
|---|---|---|---:|---:|---:|---:|---|---|
| elementwise-16M     | float16  | [4096, 4096]    | [4096, 4096] | 4096 | 257.924 | 0.3903 | events | true |
| elementwise-16M     | bfloat16 | [4096, 4096]    | [4096, 4096] | 4096 | 257.172 | 0.3914 | events | true |
| elementwise-16M     | float32  | [4096, 4096]    | [4096, 4096] | 2048 | 357.714 | 0.5628 | events | true |
| cnn-feat-broadcast  | float16  | [16, 256, 56, 56] | [256, 1, 1] | 4096 | 249.390 | 0.3090 | events | true |
| cnn-feat-broadcast  | bfloat16 | [16, 256, 56, 56] | [256, 1, 1] | 4096 | 253.894 | 0.3036 | events | true |
| cnn-feat-broadcast  | float32  | [16, 256, 56, 56] | [256, 1, 1] | 2048 | 356.352 | 0.4326 | events | true |
| row-broadcast       | float32  | [4096, 4096]    | [1, 4096]    | 2048 | 401.766 | 0.5011 | events | true |
| row-broadcast       | float16  | [4096, 4096]    | [1, 4096]    | 4096 | 279.148 | 0.3606 | events | true |
| row-broadcast       | bfloat16 | [4096, 4096]    | [1, 4096]    | 4096 | 266.312 | 0.3780 | events | true |

Torch `torch.minimum` reference (valid msprof, same run, from benchmark.json tag=torch) — recorded
for context only:
| label | dtype | latency (us) | prof_mode |
|---|---|---:|---|
| elementwise-16M    | float16  | 54.339  | msprof |
| elementwise-16M    | bfloat16 | 65.859  | msprof |
| elementwise-16M    | float32  | 135.197 | msprof |
| cnn-feat-broadcast | float16  | 38.679  | msprof |
| cnn-feat-broadcast | bfloat16 | 69.479  | msprof |
| cnn-feat-broadcast | float32  | 69.559  | msprof |
| row-broadcast      | float32  | 80.358  | msprof |
| row-broadcast      | float16  | 38.019  | msprof |
| row-broadcast      | bfloat16 | 72.799  | msprof |

(Different measurement methods — tileops device events include launch overhead, torch rows are
kernel-only msprof. No cross-method ratio computed; performance reported only, not tuned.)

## gate 5 self-check
- `python .agents/tools/statectl.py gate 5 --migration-dir examples/clamp_max` -> `pass: true`,
  `failures: []`, one warning `S5-REPORT-BENCH` ("正确性通过，但 benchmark 未得到有效结果；仅记录，
  不阻断集成") — expected for this partial benchmark.

## 调试历史
- attempt 1 (re-run): no fix required. Integration re-run idempotent, full report re-run gave
  correctness 15/15. The previous-stage blocker (frozen Stage 0 test fixture bug at
  `tests/ops/test_clamp_max.py` lines 120/121) was repaired out-of-band by the user BEFORE this run;
  re-verified in-tree that lines 120/121 now read
  `torch.randn(input_shape, dtype=torch.float16, device=device)` /
  `torch.randn(max_shape, dtype=torch.float16, device=device)` (no `*` unpack).
- No kernel/wrapper/debug-loop edit was made in this Stage 5; nothing to back up.

## issues
- Prior blocker (RESOLVED out-of-band, no longer blocking): the Stage 0 fixture
  `test_clamp_max_broadcast` used `torch.randn(*max_shape, ...)`; for the 0-dim `max_shape=()` case
  this expanded to `torch.randn(dtype=..., device=...)` -> `TypeError` before the op ran. Fixed by the
  user (drop the `*`); this run confirms the case now passes. No frozen-file edit was made by this
  Stage 5.
- R-type (process/infra, informational, non-blocking): the tileops-side `msprof op` capture produced
  no `OpBasicInfo*.csv` for this kernel (both kernel-name candidates failed), while the torch
  reference on the same run captured msprof fine. Result: benchmark reported in events mode and the
  report is `partial`. This is a profiler-capture layer issue, out of the integrator's editable scope
  and explicitly record-only in Stage 5. Suggested follow-up (for the reporting/benchmark infra
  owner, NOT applied here): investigate why the npuir-projected kernel launched through the TileOPs
  op path is not captured by `msprof op` (kernel-name resolution / launch stream), or make the
  benchmark treat a torch-reference-valid / tileops-fallback run explicitly.
- Integration mechanics: no `[error]`/`[warn]` from `integrate_kernel.py`; smoke pass; no
  frozen-file (tests/ops/workloads/manifest) modification in this Stage 5.
- R-type (integrate-flow, informational): the documented "re-run must pass `--extracted-module
  {op_slug}_kernel`" idempotency workaround was supplied as instructed, but a control re-run WITHOUT
  the flag also completed (`wrapper_rewritten=false`, `smoke=pass`, byte-identical `__init__.py`):
  the current script's `rewrite_wrapper_import` short-circuits as soon as it sees
  `from .{op_slug}_kernel import` (already-integrated guard), and `parse_wrapper_imports` returning
  empty falls back to `list(functions)`. So with this script version the flag is harmless but not
  strictly required on a re-run. Suggested follow-up (integrate_kernel.py owner): document the exact
  condition under which the flag matters, since the sibling-stage log (le) asserts it is mandatory.
  This control re-run bumped the integrated-file mtimes and therefore required a fresh report
  (see the superseded run above) — a harmless but non-obvious coupling.
