# Integration Log -- GeFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/ge/.migration_meta.json
- op: GeFwdOp (op_slug=ge, family=elementwise)
- integrated:
  - ge_fwd_kernel -> tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel.py
- design_docs:
  - ge_fwd_kernel -> tileops/kernels/elementwise/ge/ge_kernel/ge_fwd_kernel_DESIGN.md
    (snapshot of examples/ge/ge_fwd_kernel/DESIGN.md, byte-identical, 126382 B)
- ascend_mode: Developer. Source literal `ASCEND_MODE = "Developer"` +
  `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`; integrate_kernel.py reported
  `ascend_mode="Developer"`, `wrapper_mode_updated=false` (wrapper `GeFwdKernel.ascend_mode = "Developer"`
  already matched).
- wrapper: tileops/kernels/elementwise/ge/ge.py
  - `wrapper_rewritten=true`: the pre-integration import
    `from ._ge_fwd_kernels import coalesce_broadcast_dims, ge_fwd_kernel` was replaced by the
    baseline/perf_opt source-selection block. Baseline (Stage 3) is ACTIVE:
    `from .ge_kernel import coalesce_broadcast_dims, ge_fwd_kernel`; the perf_opt imports are
    COMMENTED and point at `.ge_kernel.perf_opt.{ge_fwd_kernel}` (no perf_opt/ dir exists yet, so
    baseline is the only valid source, as prescribed).
  - Note: the Stage 0 scaffold already carried a `Kernel source selection` comment block +
    commented perf_opt lines; the rewrite inserted one generated copy in addition, so the file now
    holds two adjacent (identical, comment-only) header blocks. Purely cosmetic; no code impact;
    gate 5 `S5-WRAPPER-SWITCH` passes (perf_opt reference + commented perf_opt import present).
- attempts: 1 (no debug loop; correctness + benchmark passed on the first report run)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "ge", "family": "elementwise", "functions": ["ge_fwd_kernel"],
 "sources": {"ge_fwd_kernel": "examples/ge/ge_fwd_kernel/ge_fwd_kernel.py"},
 "design_docs": {"ge_fwd_kernel": "<target>/ge_kernel/ge_fwd_kernel_DESIGN.md"},
 "target_dir": "examples/TileOPs/tileops/kernels/elementwise/ge/ge_kernel",
 "wrapper": "examples/TileOPs/tileops/kernels/elementwise/ge/ge.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["coalesce_broadcast_dims", "ge_fwd_kernel"], "smoke": "pass",
 "timestamp": "2026-10-14T13:31:29+0800"}
```
- `[smoke] pass` (import of `tileops.kernels.elementwise.ge.ge_kernel` + re-exports +
  `GeFwdKernel` attribute check). The `[warn]`-free run needed no `--extracted-module` re-run:
  the wrapper import had not yet been rewritten when the script parsed it, so
  `parse_wrapper_imports` captured both `coalesce_broadcast_dims` and `ge_fwd_kernel` correctly on
  the first invocation (the eq idempotency pitfall did not trigger here).

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_133155_744031_GeFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_133155_744031_GeFwdOp/report.md
- report_status: passed (run.json `status="passed"`, CLI exit 0)
- correctness: PASS -- 18/18 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`, `correctness_tests=18`
  - covers: float32/float16/bfloat16 same-shape + 3 broadcast patterns, bool fast path
    (uint8 storage), integer union (uint8/int8/int16/int32/int64), unsupported-dtype rejection,
    roofline formula
- benchmark: PASS -- `summary.benchmark_requested=true`, `benchmark_passed=true`;
  `benchmark_tests=11`, `benchmark_failed=0`, `case_count=9`, `msprof_case_count=9`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
  (9/9 tileops msprof profiles valid)
- avg max abs error: 0.00e+00 (bool output; exact compare)

### Benchmark numbers (msprof, from run.json `performance.cases`, tag=tileops)
| label | dtype | input_shape | other_shape | block_size | latency (us) | roofline util (%) | bandwidth (TB/s) | profiler_valid |
|---|---|---|---:|---:|---:|---:|---:|---|
| hidden-state-prefill | float16  | [2048,4096]    | [2048,4096] | 1024 | 83.018  | 22.45 | 1.8000 | true |
| hidden-state-prefill | bfloat16 | [2048,4096]    | [2048,4096] | 1024 | 92.078  | 20.25 | 1.8000 | true |
| hidden-state-prefill | float32  | [2048,4096]    | [2048,4096] | 1024 | 95.538  | 39.02 | 1.8000 | true |
| cnn-feat-broadcast   | float16  | [16,256,56,56] | [256,1,1]   | 1024 | 620.268 | 2.01  | 1.8000 | true |
| cnn-feat-broadcast   | bfloat16 | [16,256,56,56] | [256,1,1]   | 1024 | 621.368 | 2.84  | 1.8000 | true |
| cnn-feat-broadcast   | float32  | [16,256,56,56] | [256,1,1]   | 1024 | 341.193 | 8.02  | 1.8000 | true |
| row-broadcast        | float32  | [4096,4096]    | [1,4096]    | 1024 | 166.657 | 22.55 | 1.8000 | true |
| row-broadcast        | float16  | [4096,4096]    | [1,4096]    | 1024 | 160.357 | 11.63 | 1.8000 | true |
| row-broadcast        | bfloat16 | [4096,4096]    | [1,4096]    | 1024 | 166.897 | 11.17 | 1.8000 | true |

All 9 tileops msprof cases valid; each is paired in `benchmark.json` with a `tag="torch"`
reference record (torch_npu `GreaterEqual_*_high_performance_*`, e.g. 26.399 us for fp16
hidden-state-prefill) that the report uses as the comparison baseline; run.json keeps only the
tileops-side cases. Roofline 1.8000 TB/s (device peak) reached on all shapes (memory-bound
elementwise). Performance reported only, not tuned (Stage 4).

## 调试历史
- none. The single report run (attempt 1) passed both the full correctness suite and the msprof
  benchmark; no kernel/wrapper edit or backup was required.
- Pre-report backup taken for safety: `history_version/ge_wrapper_pre_s5.py` (pre-integration
  wrapper snapshot). Not used (no revert needed).

## issues
- R-type (cosmetic, non-blocking): `integrate_kernel.py` rewrite appends a second copy of the
  source-selection comment header when the Stage 0 scaffold already contains the template
  (baseline pointed at `._ge_fwd_kernels` + commented perf_opt lines). The wrapper ends up with
  two adjacent comment-only header blocks. Functionally harmless and gate-5 clean, but the
  generated block could detect and replace the pre-existing template instead of duplicating it.
- No frozen-file (tests/ops/workloads/manifest) problem was observed for GeFwdOp.
