# Integration Log -- AnyFwdOp

- meta: `examples/TileOPs/tileops/kernels/reduction/any/.migration_meta.json`
- op: `AnyFwdOp` (op_slug=`any`, family=`reduction`, scenario=migration-harness)
- integrated: `_logical_reduce_kernel` -> `tileops/kernels/reduction/any/any_kernel/_logical_reduce_kernel.py`
- design_docs: `_logical_reduce_kernel` -> `tileops/kernels/reduction/any/any_kernel/_logical_reduce_kernel_DESIGN.md` (present)
- wrapper: `tileops/kernels/reduction/any/any.py` rewritten to baseline/perf_opt kernel-source selection block (baseline active, perf_opt commented placeholder)
- ascend_mode: `Developer` (synced onto `LogicalReduceKernel`; unchanged -- already `Developer`)
- attempts: 1 (no debug loop -- integration + report passed on first attempt)

## 集成步骤

- Ran from `examples/TileOPs/` with the full-dependency interpreter and repo-root on
  `PYTHONPATH` (repo root provides the dev-build `tilelang` package):
  ```
  cd examples/TileOPs
  PYTHONPATH=/workspace/tilelang-mlir-ascend /workspace/venv/bin/python \
    .agents/skills/add-npu-op/scripts/integrate_kernel.py \
    --meta tileops/kernels/reduction/any/.migration_meta.json
  ```
- Result: `[ok] integration complete`, `[smoke] smoke-ok ['_logical_reduce_kernel']`,
  `[json] {... "wrapper_rewritten": true, "ascend_mode": "Developer", "smoke": "pass" ...}`.
- Generated aggregation `any_kernel/__init__.py` re-exports `_logical_reduce_kernel`.
- `wrapper_import_names` was already persisted in `.migration_meta.json` (idempotent; unchanged).
- Copied products verified byte-identical to conductor sources
  (`_logical_reduce_kernel.py` and `_logical_reduce_kernel_DESIGN.md` both `diff -q` clean).

## Wrapper switch block

`from ._any_fwd_kernels import _logical_reduce_kernel` ->
`from .any_kernel import _logical_reduce_kernel` (baseline active;
`# from .any_kernel.perf_opt._logical_reduce_kernel import _logical_reduce_kernel`
commented placeholder). No Stage 4 was run for this task, so `any_kernel/perf_opt/`
does not exist -- expected, baseline remains active.

### Cosmetic cleanup of the rewritten switch block (not a debug attempt)

The Stage 0 scaffolder pre-seeded `any.py` with a full
"Kernel source selection: baseline vs perf_opt" header block (this template postdates the
`count_nonzero`/`all` scaffolders, whose wrappers carried only a plain extracted import).
`integrate_kernel.py::rewrite_wrapper_import` replaces only the inner
`from .{extracted_module} import ...` span and injects its own generated block, so the
pre-seeded template produced a **duplicated** header block and a duplicated perf_opt
comment pair (functionally harmless: still exactly one active baseline import, all
perf_opt lines commented).

- Evidence: pre-cleanup wrapper preserved at
  `any_kernel/history_version/any_wrapper_s5_preseed_dup.py`; post-cleanup file matches the
  clean single-header pattern of the sibling `all_kernel` integration.
- Action: removed the duplicate header (integrate-flow artifact, not a code defect) from the
  wrapper glue file only; the active baseline import and perf_opt placeholder are unchanged.
  Re-verified with the import smoke
  (`smoke-ok ['_logical_reduce_kernel'] ascend_mode= Developer`).
- Reported as an **R-class** flow issue in
  `examples/any/RETROSPECTIVE.md` (Stage 5 section) so `integrate_kernel.py` can be made
  idempotent against a pre-seeded wrapper block.

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261016_071759_121958_AnyFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_071759_121958_AnyFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_071759_121958_AnyFwdOp/report.html`
- report_status: `passed` (CLI exit 0; `summary.benchmark_passed=true`, `invalid_profile_count=0`)
- correctness: **pass** (42/42 cases passed, 0 failed, 0 errors, 0 skipped; pass_rate 1.0;
  `avg max abs error = 0.00e+00`; per-case `max_abs_err` property `0.00e+00`).
  Consistent with the conductor's independent double-check (`torch.equal`, 101 exact
  comparisons, 0 differences) -- logical OR over 0/1 values has no floating-point error.
- benchmark: **pass** (6/6 benchmark tests passed, 0 failed; `msprof_case_count=3`,
  `invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`;
  `profiler_valid=true` for all 3 cases; profiled kernel name `main`):

  | label | shape | dtype | dim | block_m | latency (us) | roofline ratio | bandwidth |
  |---|---|---|---|---:|---:|---:|---:|
  | mask-validation-4k | [32, 4096] | bool | -1 | 8 | 2.7401 | 10.94% | 1.75 TB/s |
  | mask-validation-32k | [32, 32768] | bool | -1 | 1 | 6.6801 | 35.88% | 1.75 TB/s |
  | 3d-multidim-reduce | [4, 128, 4096] | bool | [0, 2] | 2 | 11.1002 | 43.18% | 1.75 TB/s |

- environment: Ascend910B3 x1, CANN 9.0.0, driver 26.1.1, PyTorch 2.9.0,
  torch_npu 2.9.0.post8, python 3.10.12, git_commit 7fe5a890cbba3784f27ea742af219984eda202ba.
- Note: `_resolve_block_m` + `_select_config` in the kernel fall back `block_m` by UB element
  budget as designed; the GPU-era wrapper `default_config` (SMEM heuristic) was left unchanged
  and produced no UB overflow at compile time (no wrapper config edit needed).

## 调试历史

- 无（attempt 1 一次通过；无需备份/修改集成 kernel）。唯一对 wrapper 的改动是上述
  切换块重复头的**外观清理**（备份保留于 `history_version/`），非失败驱动的调试，不计为 attempt。

## 共享文件 / 姊妹算子

- 未修改任何共享或姊妹文件。`tileops/ops/reduction/__init__.py`、
  `tileops/kernels/reduction/__init__.py`、`tileops/kernels/reduction/_primitives.py`、
  `tileops/manifest/reduction.yaml`、`tileops/workloads/reduction.py`、以及
  `count_nonzero/`、`all/` 整棵子树（含 `all/_all_fwd_kernels.py` 孤儿文件）均由本次
  集成脚本未触碰；集成仅写入 `kernels/reduction/any/` 包内文件。
- `kernels/reduction/any/.migration_meta.json`：`wrapper_import_names` 已存在且在本次为
  不变（幂等），未产生内容变更。

## issues

- none（功能性）。外观层面记录一项 integrate_kernel.py 流程缺陷（见上，R 类，已进
  `examples/any/RETROSPECTIVE.md`）。
