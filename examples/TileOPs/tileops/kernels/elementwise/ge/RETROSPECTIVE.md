# RETROSPECTIVE — GeFwdOp (op_slug=ge)

> migration-harness op-level retrospective. Stage 5 (integrate) is op-scoped (integration
> problems do not belong to a single function); Stage 1–3 function-scoped retrospectives live at
> `examples/ge/ge_fwd_kernel/RETROSPECTIVE.md`.

## Stage 5 (integrate) Retrospective — attempt_1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` `rewrite_wrapper_import()` inserts the generated baseline/perf_opt source-selection block at the position of the old `from .{extracted_module} import ...` statement **without removing the scaffold's pre-existing version of that same block**. When Stage 0 already emitted the template (baseline pointing at `._ge_fwd_kernels` + commented perf_opt lines), the integrated wrapper ends up with two adjacent, comment-only copies of the header. Reproduced on this op and visible in the sibling: `ge.py` L67–L82, `eq.py` L68–L89 both show a duplicated `# Kernel source selection:` header. Functionally harmless and gate-5 clean, but pollutes every migrated wrapper. | `examples/TileOPs/tileops/kernels/elementwise/ge/ge.py` (post-integration, two header blocks) vs `history_version/ge_wrapper_pre_s5.py` (one scaffold block); `eq.py` L68/L79 (same duplication) | Make `rewrite_wrapper_import()` detect a pre-existing `# Kernel source selection` / `# --- baseline (Stage 3)` template region and replace it instead of inserting alongside; or make the Stage 0 scaffold emit only a plain `from ._…_kernels import …` without the template. | R |
| Integrate-flow | No debug loop was needed (report passed on attempt 1), so the known-fix catalogue's idempotency trap (`--extracted-module` re-run) did **not** trigger here: the wrapper import was still `._ge_fwd_kernels` at parse time, so `parse_wrapper_imports` captured both re-exports on the first invocation. This is the opposite of the `eq` case and confirms the eq defect is order-dependent, not universal. | `integration_log.md` `[json]` (single run, `wrapper_rewritten=true`, `reexported=["coalesce_broadcast_dims","ge_fwd_kernel"]`); `eq_kernel/integration_log.md` §Integration note | Document in the integrate step that a **fresh scaffold** (import not yet rewritten) integrates correctly in one pass, and the `--extracted-module` re-run is only needed when a *prior* rewrite already happened (re-integration/partial assembly). | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Ordered-comparison (`ge`) bool-output migration integrates through the frozen Op layer unchanged** — the Stage 3 kernel emits a 1-byte `int8` 0/1 carrier (not `torch.bool`), and the frozen `GeFwdOp._eager_forward` restores the public bool result two ways: non-bool dtypes via `result.to(torch.bool)`, the `torch.bool` storage fast path via `result.view(torch.bool)` after the Op views operands as `uint8`. No wrapper dtype glue, no Op edit was required; end-to-end green. The uint8 unsigned-correction chain and the int16→fp32 domain (Stage 3 R4/R5) survive the wrapper→Op indirection. | `run.json` `summary.correctness_passed=true`, `correctness_tests=18` (incl. `test_ge_bool_storage`, integer union uint8/int8/int16/int32/int64); `tileops/ops/elementwise/ge.py` L133–L151; kernel int8 output at `ge_fwd_kernel.py` `out_dtype="int8"` | `cd examples/TileOPs && ASCEND_RT_VISIBLE_DEVICES=1 flock /tmp/tilelang-msprof.lock python3 -m tileops.reporting.cli run --op GeFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910_9362 ×1 / driver 25.5.5 | pattern-library（elementwise 比较类迁移模式：bool 输出的 op 层还原契约） |
| **Default wrapper `block_size` for the comparison family stays inside the kernel's UB budget on the int8-carrier path** — `GeFwdKernel.default_config` returns `_THREADS*_NPT` = 1024 (float) / 2048 (int dtypes) and `GeBoolStorageFwdKernel` returns 4096 (npt=16); the Stage 3 `_max_block` guard for the `need_ufix=True` uint8 path is 6086, so all three defaults compile. The wrapper's GPU-derived heuristic (K11 `threads*npt`) therefore needs no NPU re-tuning to *pass*, only the guard value respects it. | `ge.py` `_default_block_size` / `GeBoolStorageFwdKernel.default_config`; report 9/9 valid msprof cases all at `block_size=1024`; PREV: `design_calc_check.py`/Stage-3 `_max_block` = 6086 for uint8 | report command above; `benchmark.json` `records[*].config.block_size` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910_9362 ×1 / driver 25.5.5 | pattern-library/traps-runtime.md（UB budget vs wrapper default block_size 一致性） |
| **`benchmark.json` pairs each measured case with a `tag="torch"` reference; only the `tag="tileops"` records land in `run.json`** — for a `--prof-mode msprof` run the report records 2 entries per workload (the TileLang `main` kernel and the torch_npu `GreaterEqual_*_high_performance_*` reference) plus the smoke cases, so `benchmark.json` has ~2× the `run.json case_count`. Integrators reading `benchmark.json` must filter `tag=="tileops"` (or read `run.json`) or they will mix reference and measured latencies. | `benchmark.json` 22 records = 9 tileops + 9 torch + 4 smoke; `run.json` `performance.cases` = 9, all `tag="tileops"`, `msprof.op_name="main"` | report command above; `python3 -c "import json;print(len(json.load(open('.../benchmark.json'))['records']))"` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / CANN 9.0.0 / Ascend910_9362 ×1 / driver 25.5.5 | Integrate-flow docs（TileOPs report 读取口径） |

### Transferable Lessons（可迁移教训）

- **A Stage-5 integration report can pass on attempt 1 when the kernel already owns the carrier
  contract.** `ge` needed no wrapper dtype glue because Stage 3 hard-coded the 1-byte `int8` 0/1
  output and left bool restoration to the frozen Op layer (`view(torch.bool)` on the bool-storage
  path, `.to(torch.bool)` otherwise). Before adding glue to a wrapper, check whether the Stage 3
  kernel already declares its output carrier and whether the Op layer already restores the public
  dtype.
- **"Integrate passed" does not mean "re-integrate will pass the same way".**
  `integrate_kernel.py`'s `--extracted-module` re-run is an *order-dependent* workaround: it is only
  needed when a previous run already rewrote the wrapper's import. On a fresh scaffold (import still
  `from ._{op}_kernels import …`), a single default invocation correctly captures all re-exports.
  Do not preemptively add `--extracted-module`; run the prescribed command and only re-run if the
  import smoke test fails on a missing helper.
- **`integration_report.json` must point at the timestamped run, not `latest.json`.** The gate
  (`S5-REPORT-REF`) resolves the recorded path relative to the repo root and rejects anything
  outside `examples/TileOPs/reports/tileops/`; using `latest.json` would also be overwritten by the
  next op's run.
- **Read benchmark numbers from `run.json`/`report.md`, not from raw `benchmark.json`.** The raw
  file interleaves the torch_npu reference records; the `tag` field is the discriminator.
