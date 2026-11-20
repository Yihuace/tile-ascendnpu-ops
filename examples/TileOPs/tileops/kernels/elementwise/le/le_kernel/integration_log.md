# Integration Log -- LeFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/le/.migration_meta.json
- op: LeFwdOp (op_slug=le, family=elementwise)
- integrated:
  - le_fwd_kernel -> tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel.py
    (copied from examples/le/le_fwd_kernel/le_fwd_kernel.py, 51970 B; `cmp` byte-identical)
- design_docs:
  - le_fwd_kernel -> tileops/kernels/elementwise/le/le_kernel/le_fwd_kernel_DESIGN.md
    (snapshot of examples/le/le_fwd_kernel/DESIGN.md, byte-identical, 131448 B; verified by `cmp`)
- ascend_mode: Developer. Source carries both the module-level literal `ASCEND_MODE = "Developer"`
  and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`; integrate_kernel.py reported
  `ascend_mode="Developer"`, `wrapper_mode_updated=false` (wrapper `LeFwdKernel.ascend_mode = "Developer"`
  already matched). No mixed-mode conflict (single extracted function).
- wrapper: tileops/kernels/elementwise/le/le.py
  - `wrapper_rewritten=true`: the pre-integration import
    `from ._le_fwd_kernels import coalesce_broadcast_dims, le_fwd_kernel` was replaced by the
    baseline/perf_opt source-selection block. Baseline (Stage 3) is ACTIVE:
    `from .le_kernel import coalesce_broadcast_dims, le_fwd_kernel`; the perf_opt imports are
    COMMENTED and point at `.le_kernel.perf_opt.le_fwd_kernel` (no perf_opt/ dir exists yet, so
    baseline is the only valid source, as prescribed).
  - Note (cosmetic, same as the ge/eq/gt siblings): the Stage 0 scaffold already carried a
    `Kernel source selection` comment block + commented perf_opt lines; the rewrite inserted one
    generated copy in addition, so the file now holds two adjacent (identical, comment-only) header
    blocks. Purely cosmetic; no code impact; gate 5 `S5-WRAPPER-SWITCH` passes (perf_opt reference +
    commented perf_opt import present).
  - Pre-integration wrapper backed up to `le_kernel/history_version/le_wrapper_pre_s5.py` (safety;
    not used, no revert needed).
- attempts: 1 (no debug loop; correctness + benchmark passed on the first report run)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "le", "family": "elementwise", "functions": ["le_fwd_kernel"],
 "sources": {"le_fwd_kernel": "/mnt/workspace/tilelang-mlir-ascend/examples/le/le_fwd_kernel/le_fwd_kernel.py"},
 "design_docs": {"le_fwd_kernel": "<target>/le_kernel/le_fwd_kernel_DESIGN.md"},
 "target_dir": "examples/TileOPs/tileops/kernels/elementwise/le/le_kernel",
 "wrapper": "examples/TileOPs/tileops/kernels/elementwise/le/le.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["coalesce_broadcast_dims", "le_fwd_kernel"], "smoke": "pass",
 "timestamp": "2026-10-14T17:27:23+0800"}
```
- `[smoke] pass` (import of `tileops.kernels.elementwise.le.le_kernel` + re-exports +
  `LeFwdKernel` attribute check). First invocation is the canonical path (wrapper still imported from
  `._le_fwd_kernels`), so `parse_wrapper_imports` captured both `coalesce_broadcast_dims` and
  `le_fwd_kernel`; the eq/ge idempotency pitfall did not trigger (this was the first run; the
  `--extracted-module le_kernel` re-run workaround is only needed on a *re-run* after the wrapper has
  been rewritten).

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_172740_241358_LeFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_172740_241358_LeFwdOp/report.md
- report_status: passed (run.json `status="passed"`, CLI exit 0)
- correctness: PASS -- 18/18 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`, `correctness_tests=18`
  - covers: float32/float16/bfloat16 same-shape (smoke + full), 3 broadcast patterns, edge case
    (known-equal positions), bool fast path (uint8-backed bool storage), integer union
    (uint8/int8/int16/int32/int64), unsupported-dtype rejection (complex64/fp8_e4m3fn/fp8_e5m2),
    roofline formula
- benchmark: PASS -- `summary.benchmark_requested=true`, `benchmark_passed=true`;
  `benchmark_tests=11`, `benchmark_failed=0`, `case_count=9`, `msprof_case_count=9`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
  (9/9 tileops msprof profiles valid)
- avg max abs error: 0.00e+00 (bool output; exact compare)

### Benchmark numbers (msprof, from run.json `performance.cases`, tag=tileops)
| label | dtype | input_shape | other_shape | block_size | latency (us) | roofline util (%) | bandwidth (TB/s) | profiler_valid |
|---|---|---|---|---:|---:|---:|---:|---|
| hidden-state-prefill | float16  | [2048,4096]    | [2048,4096] | 1024 | 82.098  | 22.71 | 1.8000 | true |
| hidden-state-prefill | bfloat16 | [2048,4096]    | [2048,4096] | 1024 | 92.098  | 20.24 | 1.8000 | true |
| hidden-state-prefill | float32  | [2048,4096]    | [2048,4096] | 1024 | 95.258  | 39.14 | 1.8000 | true |
| cnn-feat-broadcast   | float16  | [16,256,56,56] | [256,1,1]   | 1024 | 610.048 | 2.04  | 1.8000 | true |
| cnn-feat-broadcast   | bfloat16 | [16,256,56,56] | [256,1,1]   | 1024 | 529.309 | 3.33  | 1.8000 | true |
| cnn-feat-broadcast   | float32  | [16,256,56,56] | [256,1,1]   | 1024 | 374.812 | 7.30  | 1.8000 | true |
| row-broadcast        | float32  | [4096,4096]    | [1,4096]    | 1024 | 165.017 | 22.60 | 1.8000 | true |
| row-broadcast        | float16  | [4096,4096]    | [1,4096]    | 1024 | 160.997 | 11.58 | 1.8000 | true |
| row-broadcast        | bfloat16 | [4096,4096]    | [1,4096]    | 1024 | 164.957 | 11.30 | 1.8000 | true |

All 9 tileops msprof cases valid (`profiler_valid=true`, `invalid_reason=null`); each is paired in
`benchmark.json` with a `tag="torch"` torch_npu reference record used as the comparison baseline
(run.json keeps only the tileops-side cases). Roofline 1.8000 TB/s (device peak) reached on all
shapes (memory-bound elementwise). Performance reported only, not tuned (Stage 4).

## 调试历史
- none. The single report run (attempt 1) passed both the full correctness suite and the msprof
  benchmark; no kernel/wrapper edit was required.
- Pre-report safety backup: `le_kernel/history_version/le_wrapper_pre_s5.py` (pre-integration wrapper
  snapshot) and `history_version/le_fwd_kernel_s5_pre.py` (integrated kernel snapshot). Not used
  (no revert needed).

## issues
- R-type (cosmetic, non-blocking, shared with ge/eq/gt): `integrate_kernel.py` rewrite appends a
  second copy of the source-selection comment header when the Stage 0 scaffold already contains the
  template. Functionally harmless and gate-5 clean. Suggested fix: detect and replace the
  pre-existing template instead of duplicating it.
- No frozen-file (tests/ops/workloads/manifest) problem was observed for LeFwdOp.
