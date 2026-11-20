# Integration Log -- GtFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/gt/.migration_meta.json
- op: GtFwdOp (op_slug=gt, family=elementwise)
- integrated:
  - gt_fwd_kernel -> tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel.py
    (copied from examples/gt/gt_fwd_kernel/gt_fwd_kernel.py, 51861 B)
- design_docs:
  - gt_fwd_kernel -> tileops/kernels/elementwise/gt/gt_kernel/gt_fwd_kernel_DESIGN.md
    (snapshot of examples/gt/gt_fwd_kernel/DESIGN.md, byte-identical, 131573 B; verified by cmp)
- ascend_mode: Developer. Source carries both the module-level literal `ASCEND_MODE = "Developer"`
  and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`; integrate_kernel.py reported
  `ascend_mode="Developer"`, `wrapper_mode_updated=false` (wrapper `GtFwdKernel.ascend_mode = "Developer"`
  already matched).
- wrapper: tileops/kernels/elementwise/gt/gt.py
  - `wrapper_rewritten=true`: the pre-integration import
    `from ._gt_fwd_kernels import coalesce_broadcast_dims, gt_fwd_kernel` was replaced by the
    baseline/perf_opt source-selection block. Baseline (Stage 3) is ACTIVE:
    `from .gt_kernel import coalesce_broadcast_dims, gt_fwd_kernel`; the perf_opt imports are
    COMMENTED and point at `.gt_kernel.perf_opt.gt_fwd_kernel` (no perf_opt/ dir exists yet, so
    baseline is the only valid source, as prescribed).
  - Note (cosmetic, same as the ge/eq siblings): the Stage 0 scaffold already carried a
    `Kernel source selection` comment block + commented perf_opt lines; the rewrite inserted one
    generated copy in addition, so the file now holds two adjacent (identical, comment-only) header
    blocks. Purely cosmetic; no code impact; gate 5 `S5-WRAPPER-SWITCH` passes (perf_opt reference +
    commented perf_opt import present).
  - Pre-integration wrapper backed up to `history_version/gt_wrapper_pre_s5.py` (safety; not used,
    no revert needed).
- attempts: 1 (no debug loop; correctness + benchmark passed on the first report run)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "gt", "family": "elementwise", "functions": ["gt_fwd_kernel"],
 "sources": {"gt_fwd_kernel": "examples/gt/gt_fwd_kernel/gt_fwd_kernel.py"},
 "design_docs": {"gt_fwd_kernel": "<target>/gt_kernel/gt_fwd_kernel_DESIGN.md"},
 "target_dir": "examples/TileOPs/tileops/kernels/elementwise/gt/gt_kernel",
 "wrapper": "examples/TileOPs/tileops/kernels/elementwise/gt/gt.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["coalesce_broadcast_dims", "gt_fwd_kernel"], "smoke": "pass",
 "timestamp": "2026-10-14T15:30:19+0800"}
```
- `[smoke] pass` (import of `tileops.kernels.elementwise.gt.gt_kernel` + re-exports +
  `GtFwdKernel` attribute check). First invocation is the canonical path (wrapper still imported from
  `._gt_fwd_kernels`), so `parse_wrapper_imports` captured both `coalesce_broadcast_dims` and
  `gt_fwd_kernel`; the eq/ge idempotency pitfall did not trigger (this was the first run; the
  `--extracted-module gt_kernel` re-run workaround is only needed on a *re-run* after the wrapper has
  been rewritten).

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_153035_879724_GtFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_153035_879724_GtFwdOp/report.md
- report_status: passed (run.json `status="passed"`, CLI exit 0)
- correctness: PASS -- 18/18 cases passed (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `summary.correctness_passed=true`, `correctness_tests=18`
  - covers: float32/float16/bfloat16 same-shape + 3 broadcast patterns + edge case, bool fast path
    (uint8-backed bool storage), integer union (uint8/int8/int16/int32/int64, incl. the gt-specific
    unsigned-order uint8 path), unsupported-dtype rejection (complex64/fp8), roofline formula
- benchmark: PASS -- `summary.benchmark_requested=true`, `benchmark_passed=true`;
  `benchmark_tests=11`, `benchmark_failed=0`, `case_count=9`, `msprof_case_count=9`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
  (9/9 tileops msprof profiles valid)
- avg max abs error: 0.00e+00 (bool output; exact compare)

### Benchmark numbers (msprof, from run.json `performance.cases` / report.md, tag=tileops)
| label | dtype | input_shape | other_shape | latency (us) | ratio (%) | bandwidth (TB/s) | profiler_valid |
|---|---|---|---:|---:|---:|---:|
| hidden-state-prefill | float16  | [2048,4096]    | [2048,4096] | 82.618  | 22.56 | 1.8000 | true |
| hidden-state-prefill | bfloat16 | [2048,4096]    | [2048,4096] | 91.278  | 20.42 | 1.8000 | true |
| hidden-state-prefill | float32  | [2048,4096]    | [2048,4096] | 94.758  | 39.35 | 1.8000 | true |
| cnn-feat-broadcast   | float16  | [16,256,56,56] | [256,1,1]   | 512.130 | 2.42  | 1.8000 | true |
| cnn-feat-broadcast   | bfloat16 | [16,256,56,56] | [256,1,1]   | 603.208 | 2.92  | 1.8000 | true |
| cnn-feat-broadcast   | float32  | [16,256,56,56] | [256,1,1]   | 385.332 | 7.11  | 1.8000 | true |
| row-broadcast        | float32  | [4096,4096]    | [1,4096]    | 184.756 | 20.18 | 1.8000 | true |
| row-broadcast        | float16  | [4096,4096]    | [1,4096]    | 183.996 | 10.13 | 1.8000 | true |
| row-broadcast        | bfloat16 | [4096,4096]    | [1,4096]    | 185.776 | 10.04 | 1.8000 | true |

All 9 tileops msprof cases valid (`profiler_valid=true`, `invalid_reason=null`); each is paired in
`benchmark.json` with a `tag="torch"` torch_npu reference record used as the comparison baseline
(run.json keeps only the tileops-side cases). Roofline 1.8000 TB/s (device peak) reached on all
shapes (memory-bound elementwise). Performance reported only, not tuned (Stage 4).

## 调试历史
- none. The single report run (attempt 1) passed both the full correctness suite and the msprof
  benchmark; no kernel/wrapper edit was required.
- Pre-report safety backup: `history_version/gt_wrapper_pre_s5.py` (pre-integration wrapper
  snapshot). Not used (no revert needed).

## issues
- R-type (cosmetic, non-blocking, shared with ge/eq): `integrate_kernel.py` rewrite appends a second
  copy of the source-selection comment header when the Stage 0 scaffold already contains the template.
  Functionally harmless and gate-5 clean. Suggested fix: detect and replace the pre-existing template
  instead of duplicating it.
- No frozen-file (tests/ops/workloads/manifest) problem was observed for GtFwdOp.
