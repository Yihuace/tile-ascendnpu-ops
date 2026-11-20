# RETROSPECTIVE — CountNonzeroFwdOp (op-level)

> Op-level retrospective. Stage 5 integration appends here (see
> `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`).
> Per-function retrospectives live under `_logical_reduce_kernel/RETROSPECTIVE.md`.

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py`'s in-script import smoke test runs a `subprocess` with `cwd=tileops_root` and inherits the caller's env; when the tilelang repo root is importable only via cwd (not pip-installed), the smoke exits non-zero with `ModuleNotFoundError: No module named 'tilelang'` even though every integration artifact was written correctly | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` L398-422 (`smoke_test`) + this task's first run transcript; re-run with `PYTHONPATH=/workspace/tilelang-mlir-ascend` printed `[smoke] smoke-ok ['_logical_reduce_kernel']` | add-npu-op SKILL.md: state the required `PYTHONPATH=<tilelang_repo_root>` (or an editable install) for the integration smoke step, and/or have `smoke_test` inject `tilelang_root` into the subprocess env | R |
| Integrate-flow | the known-repair catalogue lists the wrapper `default_config` mis-wiring, and the integrator prompt flagged it, yet the wrapper ships with the GPU-era SMEM heuristic until Stage 5; a naive first report would fail to compile, not fail correctness | `tileops/kernels/reduction/count_nonzero/count_nonzero.py` (pre-edit, backup `count_nonzero_kernel/history_version/count_nonzero_s5_attempt1.py`); `examples/count_nonzero/_logical_reduce_kernel/DESIGN.md` §5.6; same defect class as `tileops/kernels/reduction/argmin/argmin_kernel/integration_log.md` attempt 1 | add-npu-op SKILL.md: make the Stage 5 pre-report checklist explicitly require re-wiring `Kernel.default_config` to the kernel module's `_select_config` before the first report run (kernel module = tiling single source of truth) | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Integrate-flow pattern: TileOPs wrapper `default_config` must delegate to the kernel's `_select_config(M, N, dtype)` instead of the inherited GPU SMEM heuristic | P | `count_nonzero.py` `default_config` -> `{"block_m": _select_config(self.M, self.N, self.dtype_to_str(self._kernel_dtype))["block_m"]}`; cross-check vs DESIGN.md §5.2 frozen table (all 7 probed shapes match); report `20261016_015623_851978_CountNonzeroFwdOp` passed 37/37 + 7/7 | `cwd=examples/TileOPs; PYTHONPATH=<tilelang_repo_root> python -m tileops.reporting.cli run --op CountNonzeroFwdOp --prof-mode msprof` | tilelang build `/workspace/tilelang-mlir-ascend/build`, git b39cc306, Ascend910B3, CANN 9.0.0, driver 26.1.1, torch 2.9.0+cpu, torch_npu 2.9.0.post8, Python 3.10.12 | pattern-library / bottleneck-patterns |
| Element-budget config values reproduce DESIGN.md §5.2 exactly through the wrapper path | D | `(32,256)->8`, `(2048,4096)->2`, `(256,4096)->2`, `(128,300)->8`, `(32,16384)->C bm=1`, `(128,16384)->C bm=1`, bool->f32 bm=8; report run.json `performance.cases[].config` confirms `block_m=2` for sparsity-hidden and `block_m=1` for the two path-C workloads | `python -c "from tileops.kernels.reduction.count_nonzero.count_nonzero import LogicalReduceKernel as K; print(K(2048,4096,'count_nonzero',__import__('torch').float16).config)"` | same as above | pattern-library |

### Transferable Lessons（可迁移教训）

- In this harness the tilelang repo root is **not pip-installed**; run integration and the
  report with `PYTHONPATH=/workspace/tilelang-mlir-ascend` (or an editable install) from
  `examples/TileOPs/`, otherwise the `tilelang` import fails even though tileops resolves.
- A migrated TileOPs wrapper keeps its GPU-era `default_config` after scaffolding; before the
  first Stage 5 report, re-wire `default_config` to the NPU kernel's own `_select_config`
  (kernel module = tiling single source of truth) so the per-block element budget
  (`E_CAP` / UB hard limit) is honored; the kernel-side defensive fallback is **not** a
  substitute for the wrapper wiring.
- The report's `Kernel` column showing `main` even when a factory sets `msprof_kernel_name`
  is a known cosmetic reporting-layer behaviour, not an integration defect.
