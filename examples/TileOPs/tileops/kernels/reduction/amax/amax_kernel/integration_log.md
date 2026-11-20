# Integration Log -- AmaxFwdOp

- meta: examples/TileOPs/tileops/kernels/reduction/amax/.migration_meta.json
- op_slug / family: amax / reduction
- wrapper: examples/TileOPs/tileops/kernels/reduction/amax/amax.py (`ReduceKernel`, ascend_mode=Developer)
- integrated:
  - `_simple_reduce_kernel` -> tileops/kernels/reduction/amax/amax_kernel/_simple_reduce_kernel.py
  - `_simple_reduce_kernel_tiled` -> tileops/kernels/reduction/amax/amax_kernel/_simple_reduce_kernel_tiled.py
- design_docs:
  - `_simple_reduce_kernel` -> tileops/kernels/reduction/amax/amax_kernel/_simple_reduce_kernel_DESIGN.md
  - `_simple_reduce_kernel_tiled` -> tileops/kernels/reduction/amax/amax_kernel/_simple_reduce_kernel_tiled_DESIGN.md
  - (none missing)
- attempts: 2

## Deterministic integration

`python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/reduction/amax/.migration_meta.json`

- First run FAILED its own import smoke (`[error] import smoke test FAILED`). Two independent glue defects were exposed (see 调试历史 attempt 1).
- After the wrapper glue fix, re-run completed: `[smoke] pass`, `[mode] Developer (unchanged)`, reexported = `["_simple_reduce_kernel", "_simple_reduce_kernel_tiled"]`, `mixed_modes=false`, `wrapper_rewritten=false` (idempotent).

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261015_135122_905437_AmaxFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261015_135122_905437_AmaxFwdOp/report.md
- report_status: partial
- correctness: **pass** -- 14/14 (`tests/ops/test_amax.py`), avg max abs error 0.00e+00
- benchmark: **partial** -- 3 of 4 manifest workloads produced valid msprof records; `long-seq-reduce` (tiled path) errored.
  - hidden-state-reduce | float16 | (2048, 4096) | 15.50 us | Ratio 61.852% | BW 1.75 TB/s
  - hidden-state-reduce | bfloat16 | (2048, 4096) | 19.56 us | Ratio 49.014% | BW 1.75 TB/s
  - 3d-multidim-reduce | float16 | (4, 128, 4096) dim=[0,2] | 8.66 us | Ratio 27.677% | BW 1.75 TB/s
  - smoke-amax | float16 | (32, 256) | 1.68 us | Ratio 0.566% | BW 1.75 TB/s   (recorded; not shown in report.md Operator Details)
  - smoke-amax | bfloat16 | (32, 256) | 2.10 us | Ratio 0.460% | BW 1.75 TB/s  (recorded)
  - smoke-amax | float32 | (32, 256) | 1.62 us | Ratio 1.165% | BW 1.75 TB/s   (recorded)
  - long-seq-reduce | bfloat16 | (64, 32768) | n/a | n/a | n/a -- `AssertionError: tile_n=16384 x B=6 exceeds the x2 UB envelope; lower tile_n`
  - Bandwidth(TB/s)=1.75 is the reference/peak constant used by the reporter for Ratio (identical across every record); achieved BW = bytes/latency (e.g. hidden-state fp16 ≈ 1.08 TB/s).

## 调试历史

- attempt 1 (integration script smoke failure -- wrapper glue):
  - 症状: `ImportError: cannot import name '_welford_reduce_kernel' from 'tileops.kernels.reduction.amax.amax_kernel'`, then `AssertionError: missing exports: ['_welford_reduce_kernel', '_welford_reduce_kernel_tiled']`.
  - 分类: wrapper 胶水 / import 路径错误（脚本 `integrate_kernel.py` 缺陷触发）。
  - 根因: `parse_wrapper_imports` (integrate_kernel.py) walks the *whole* wrapper AST and collects names from **every** `from ._amax_fwd_kernels import ...` statement, including the `try/except` **optional** Welford import (std/var, out of this migration unit). `rewrite_wrapper_import` then emits all four names into the baseline `from .amax_kernel import (...)`, but the generated aggregator correctly ships only the two migrated builders -> ImportError. The post-integration smoke test additionally asserts `hasattr(amax_kernel, n)` for all collected names, so the optional names also fail the smoke. Two secondary defects: generated `amax_kernel/__init__.py` `__all__` lists the two Welford names that are never imported (`all_names.append` is unconditional in `gen_init_py`).
  - 修复（仅 wrapper 胶水，`amax.py`；备份 history_version/amax_s5_attempt1.py）:
    1. baseline 块只保留两个已迁移 builder：`from .amax_kernel import _simple_reduce_kernel, _simple_reduce_kernel_tiled`；
    2. Welford 可选导入改为从集成包解析：`try: from .amax_kernel import _welford_reduce_kernel, _welford_reduce_kernel_tiled` / `except ImportError: ... = None`。这让 `parse_wrapper_imports` 不再收集到 out-of-unit 名称，smoke 只校验实际打包的两个 builder。
  - 结果: 脚本 smoke pass。
- attempt 2 (correctness failed 14/14 -- cross-op custom_op collision):
  - 症状: `TypeError: _simple_reduce_kernel.<locals>._func() missing 1 required positional argument: 'threads'`; report 0.0% (0/14); traceback routes through `tileops/kernels/reduction/amin/amin.py:92` (NOT amax).
  - 分类: wrapper 胶水 / 运行时注册冲突。
  - 根因: `amax.py` and `amin.py` both register the same `torch.library.custom_op` names (`npub::reduce_simple_fwd`, `npub::reduce_simple_tiled_fwd`, ...); `tileops.kernels.reduction.__init__` imports amax then amin, so amin's registration (whose kernel is the un-migrated `_amin_fwd_kernels` still requiring `threads`) hijacks amax's dispatch. Confirmed the repo convention is unique per-op names (argmax=`npub::argreduce_fwd`, logsumexp=`npub::logsumexp_fwd`).
  - 修复（仅 wrapper 胶水，`amax.py`；备份 history_version/amax_s5_attempt2.py): namespace all four amax custom ops to `npub::amax_reduce_*`. Verified single case `test_amax_op[128-512-dtype0]` passes, then full report.
  - 结果: correctness 14/14 pass.

## Issues (recorded, not fixed)

- **[benchmark-only, tiled path] `long-seq-reduce` (64, 32768) bfloat16 raises `AssertionError: tile_n=16384 x B=6 exceeds the x2 UB envelope; lower tile_n`.** The wrapper's `ReduceKernel._tile_n_for_block_m` derives `tile_n` from the shared GPU-derived `_primitives.compute_tile_n` using `element_size` (bf16 = 2 bytes), while the migrated tiled kernel budgets bf16 as B = 6 (x_ub 2 + fp32 work 4) and asserts `2 * tile_n * b <= 184*1024` in `_simple_reduce_kernel_tiled._simple_reduce_kernel_tiled` (L154). The wrapper therefore selects `tile_n=16384` for N=32768 bf16 and the kernel rejects it. The tiled migration unit's own `select_tile_n`/`needs_tiling` (64KB manual budget, B=6) is not consulted by the wrapper. Not fixed per Stage 5 policy (benchmark failures are report-only) -- flagged for conductor / Stage 4 (wrapper config-selection vs migrated tiled-kernel constraint mismatch).
- **[tooling] `integrate_kernel.py` defects** (see attempt 1): (a) `parse_wrapper_imports` collects optional/co-located import names and `rewrite_wrapper_import` emits them into the required baseline import; (b) the smoke test asserts every collected name is re-exported by the integration package; (c) `gen_init_py` appends unrecognized names to `__all__` unconditionally. Suggested: restrict collected/emitted names to those actually integrated (or to the non-optional import), and only assert integrated names in the smoke.

## 变更文件（集成范围内）

- tileops/kernels/reduction/amax/amax.py (wrapper glue; 2 attempt fixes)
- tileops/kernels/reduction/amax/amax_kernel/ (integrated kernels + DESIGN snapshots + generated __init__.py)
- tileops/kernels/reduction/amax/amax_kernel/history_version/amax_s5_attempt{1,2}.py (backups)

未修改：tests/、benchmarks/、tileops/manifest/、tileops/ops/、tileops/workloads/、examples/amax/（既有产物）。
