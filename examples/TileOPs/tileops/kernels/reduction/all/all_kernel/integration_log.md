# Integration Log -- AllFwdOp

- meta: examples/TileOPs/tileops/kernels/reduction/all/.migration_meta.json
- op: AllFwdOp (op_slug=all, family=reduction)
- integrated: `_logical_reduce_kernel` -> `tileops/kernels/reduction/all/all_kernel/_logical_reduce_kernel.py`
- design_docs: `_logical_reduce_kernel` -> `tileops/kernels/reduction/all/all_kernel/_logical_reduce_kernel_DESIGN.md` (present)
- wrapper: `tileops/kernels/reduction/all/all.py` rewritten to baseline/perf_opt kernel-source selection block (baseline active)
- ascend_mode: Developer (synced onto `LogicalReduceKernel`; unchanged)
- attempts: 1 (no debug loop — integration + report passed on first attempt)

## 集成步骤
- Ran from `examples/TileOPs/`:
  `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/reduction/all/.migration_meta.json`
- First invocation failed only in the import smoke test because `tilelang` was not on
  `PYTHONPATH` (`ModuleNotFoundError: No module named 'tilelang'`). Re-ran with
  `PYTHONPATH=/workspace/tilelang-mlir-ascend` (repo root provides the `tilelang` package);
  integration then completed: `smoke-ok ['_logical_reduce_kernel']`.
  Not counted as an attempt (environment path, not a code defect).
- Generated aggregation `all_kernel/__init__.py` re-exports `_logical_reduce_kernel`.
- Wrapper rewrite: `from ._all_fwd_kernels import _logical_reduce_kernel` ->
  `from .all_kernel import _logical_reduce_kernel` (baseline active;
  `# from .all_kernel.perf_opt._logical_reduce_kernel import _logical_reduce_kernel` commented placeholder).
- Pre-existing scaffolding-only kernel copy `all/_all_fwd_kernels.py` is now orphaned
  (no longer imported); left untouched.

## 验证结果
- report_run_json: `examples/TileOPs/reports/tileops/20261016_033406_263435_AllFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_033406_263435_AllFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_033406_263435_AllFwdOp/report.html`
- report_status: passed (CLI exit 0)
- correctness: pass (42/42 cases passed, 0 failed; pass_rate 1.0; avg max abs error 0.00e+00)
- benchmark: pass (6/6 benchmark tests passed, 0 failed; msprof_case_count=3,
  invalid_profile_count=0, profiler_fallback_count=0)
  - mask-validation-4k   [32, 4096]       bool  dim=-1     block_m=8  latency 2.8601 us  ratio 10.48%  bw 1.75 TB/s
  - mask-validation-32k  [32, 32768]      bool  dim=-1     block_m=1  latency 6.7201 us  ratio 35.67%  bw 1.75 TB/s
  - 3d-multidim-reduce   [4, 128, 4096]   bool  dim=[0,2]  block_m=2  latency 10.9002 us ratio 43.98%  bw 1.75 TB/s
  - profiler: msprof, kernel `main`, profiler_valid=true for all cases.
- environment: Ascend910B3 x1, CANN 9.0.0, PyTorch 2.9.0, torch_npu 2.9.0.post8,
  tilelang git_commit 1bc8dbc3a1c80345a12522d815a2268e555270fa

## 调试历史
- 无（attempt 1 一次通过；无需备份/修改集成 kernel 或 wrapper）。

## 共享文件
- 未修改任何共享文件。`tileops/ops/reduction/__init__.py`、
  `tileops/kernels/reduction/__init__.py`、`manifest/reduction.yaml`、`workloads/reduction.py`、
  `_primitives.py` 与 `count_nonzero/` 均由 Stage 0 脚手架预先写入本算子条目，
  集成脚本未触碰它们（仅写入 `kernels/reduction/all/` 包内文件）。
- `kernels/reduction/all/.migration_meta.json` 被集成脚本更新了 `wrapper_import_names`
  字段（幂等持久化，追加语义）。

## issues
- none
