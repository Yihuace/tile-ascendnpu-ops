# Integration Log -- RopeNeoxOp

- meta: `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/.migration_meta.json`
- scenario: migration (migration_mode=harness), attempt_index=1, max_attempts=5
- target dir: `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/`
- integrated:
  - `_make_rope_neox_1d` -> `rope_neox_kernel/_make_rope_neox_1d.py`
  - `_make_rope_neox_2d` -> `rope_neox_kernel/_make_rope_neox_2d.py`
- design_docs:
  - `_make_rope_neox_1d` -> `rope_neox_kernel/_make_rope_neox_1d_DESIGN.md`
  - `_make_rope_neox_2d` -> `rope_neox_kernel/_make_rope_neox_2d_DESIGN.md`
- aggregated `__init__.py`: re-exports `_make_rope_neox_1d`, `_make_rope_neox_2d`
- wrapper: `tileops/kernels/position_encoding/rope_neox/rope_neox.py`
  - import rewritten to the baseline/perf_opt kernel-source selection block
  - **baseline active**: `from .rope_neox_kernel import _make_rope_neox_1d, _make_rope_neox_2d`
  - **perf_opt commented placeholder**:
    `# from .rope_neox_kernel.perf_opt._make_rope_neox_1d import _make_rope_neox_1d` /
    `# from .rope_neox_kernel.perf_opt._make_rope_neox_2d import _make_rope_neox_2d`
  - `ascend_mode`: `Developer` (both Stage 3 products declare `ASCEND_MODE = "Developer"`; wrapper
    scaffold already `Developer` -> `sync_wrapper_ascend_mode` reported *unchanged*, mode re-verified)
- import smoke: pass (`[smoke] ... warnings.warn(Permission mismatch...)` is the torch_npu env warning only)
- attempts: 0 (integration script + full single-op report both passed on the first run)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261014_195632_006487_RopeNeoxOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_195632_006487_RopeNeoxOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_195632_006487_RopeNeoxOp/report.html`
- benchmark_json: `examples/TileOPs/reports/tileops/20261014_195632_006487_RopeNeoxOp/benchmark.json`
- run status: `passed`
- correctness: **pass** -- 14/14 cases, `correctness_failed=0`, `pass_rate=1.0`,
  `max_abs_err=0.00e+00` on all parametrized cases (tests/ops/test_rope.py:
  `test_rope_neox_1d` x4, `test_rope_neox_2d` x4, `test_rope_neox_edge` x3, plus 3
  validation regression tests)
- benchmark: **pass** -- 6/6 cases, `benchmark_failed=0`, `invalid_profile_count=0`,
  `profiler_fallback_count=0`, `msprof_case_count=6` (all real msprof)

| workload (layout) | shape | dtype | latency (us) |
|---|---|---|---|
| neox-1d-256-d64-smoke | [256, 64] | float16 | 4.20 |
| neox-1d-256-d64-smoke | [256, 64] | bfloat16 | 4.06 |
| neox-1d-256-d64-smoke | [256, 64] | float32 | 3.16 |
| neox-1d-2k-d64-fp16 | [2048, 64] | float16 | 5.06 |
| neox-1d-4k-d128-bf16 | [4096, 128] | bfloat16 | 7.24 |
| neox-2d-b2-s2k-h32-d128-fp16 | [2, 2048, 32, 128] | float16 | 107.40 |

(torch-ref baseline latency ~345-360 us for all workloads; recorded in report only,
performance is report-only per Stage 5 contract.)

## 调试历史

- none -- `attempts=0`. The integration script completed with `[ok] integration complete`
  and the single-op report returned `status=passed` on the first run; no backup/modify/rerun
  cycle was needed.

## issues

- (info, non-blocking) After the import rewrite, the wrapper's preceding comment
  (`# Part A: GPU TileLang kernel factories extracted by extract_tl_kernel.py ...`) and the
  pre-integration module `tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py`
  are now stale/dead (the wrapper no longer imports them). Integration does not touch them, and
  they are outside the Stage 5 edit scope (the wrapper comment block is not the import-switch
  region; the extracted module is a scaffolder product). Reported for conductor awareness only --
  no correctness/benchmark impact.
- (none) No test/benchmark/manifest/workloads/ops file was modified; no tolerance was weakened.
