# RETROSPECTIVE — FloorDivideFwdOp (op_slug=floor_divide)

> Stage 5 (integrate) retrospective is written at op level per the canonical schema
> (`.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md` §1):
> migration-harness Stage 1/2/3 per-function retrospectives live in
> `examples/floor_divide/floor_divide_fwd_kernel/RETROSPECTIVE.md`; the integration
> retrospective is a single op-level entry (integration issues do not belong to one function).

## Stage 5 (integrate) Retrospective — first_integration, attempt 1 — 2026-10-14

> Result: `INTEGRATE_COMPLETED`. `integrate_kernel.py --meta` (clean first run) copied the
> Stage 3 kernel + DESIGN snapshot, generated the aggregate `__init__.py`, rewrote the
> wrapper import into the baseline/perf_opt switch block (baseline active) and synced
> `FloorDivideFwdKernel.ascend_mode = "Developer"`; import smoke passed. Single-op report
> `status="passed"`: correctness 16/16, benchmark 11/11 with 9/9 msprof-valid cases. No
> debug-loop attempt, no kernel/wrapper edit.

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `rewrite_wrapper_import` replaces the `from ._<slug>_fwd_kernels import ...` statement in place but leaves the wrapper's original prose comment (which describes that import as "the GPU-extracted reference") dangling immediately above the generated `# Kernel source selection` block; the stale comment now contradicts the active baseline import (the integrated NPU kernel). Cosmetic only — the generated block and smoke are correct. | `tileops/kernels/elementwise/floor_divide/floor_divide.py` L57–L60 (stale "Baseline source (the GPU-extracted reference)" comment) directly above the generated block | In `integrate_kernel.py::rewrite_wrapper_import`, normalise or drop the immediately-preceding comment span when replacing the import, or emit an explicit "baseline = integrated NPU kernel" header in `gen_kernel_source_block`. | R |
| Integrate-flow | No new defect: the prescribed first run captured BOTH wrapper re-exports (`coalesce_broadcast_dims` helper + `floor_divide_fwd_kernel` factory) and smoke passed. The eq-precedent idempotency workaround (`--extracted-module <op_slug>_kernel`) is only needed when the wrapper import was already rewritten before the run — i.e. a re-run, not a clean scaffold. | `[json]` summary `reexported=["coalesce_broadcast_dims","floor_divide_fwd_kernel"], smoke="pass"`; eq precedent in `tileops/kernels/elementwise/eq/eq_kernel/integration_log.md` (S5 idempotency note) | Keep the eq-precedent note in the integrator guidance: clean first run = no flag; any re-run after a rewrite must pass `--extracted-module <op_slug>_kernel`. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **FloorDivideFwdOp is a first-attempt, drop-in TileOPs harness integration**: the frozen Developer scaffold (R1 outer-row divmod + innermost vectorisation + `T.vbrc`; R2 persistent `min(num_logical, aicore*2)`; R4 `T.vdiv` + `T.vcast(round_mode="floor")` fp32 chain, fp16/bf16 via fp32 transit, fp32 skipping identity casts; factory `floor_divide_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, ...)`) needed **zero** wrapper/kernel change to pass the TileOPs single-op report (correctness 16/16, benchmark 11/11, 9/9 msprof valid). Reusable as the floordiv/division-family harness baseline. | P | `integration_log.md` (attempts=1, `[json]` summary); `run.json` `status="passed"`, `summary.correctness_passed=true`, `benchmark_passed=true`, `invalid_profile_count=0` | `cd examples/TileOPs && ASCEND_RT_VISIBLE_DEVICES=1 python3 -m tileops.reporting.cli run --op FloorDivideFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / Ascend910_9362 x1 / CANN 9.0.0 / driver 25.5.5 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 | pattern-library elementwise（floordiv 迁移模板：Stage 5 集成通过实证） |
| **FloorDivideFwdOp baseline msprof profile (harness run, block_size=1024, untuned)**: 9/9 msprof cases valid, all pinned at the 1.8000 TB/s device roofline bandwidth; latency range 84.2–661.6 us across the 3 manifest workloads × {fp16,bf16,fp32}; roofline-utilization (`ratio`) 2.70%–44.87% (worst: cnn-feat-broadcast fp16 661.6 us / 2.70%; best: hidden-state-prefill fp32 84.2 us / 44.87%). Recorded as the Stage 4 tuning baseline; performance not tuned here. | D | `run.json` `performance.cases` (9 cases); `report.md` Operator Details table | `cd examples/TileOPs && ASCEND_RT_VISIBLE_DEVICES=1 python3 -m tileops.reporting.cli run --op FloorDivideFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / Ascend910_9362 x1 / CANN 9.0.0 / torch 2.7.1+cpu | pattern-library bottleneck-patterns（elementwise broadcast 基线） / 待定 |
| **A green TileOPs harness report does not certify bit-exactness for a division→floor op**: the frozen correctness suite runs N=4096/16384 with `atol=1.0` (fp16/bf16) / `1e-5` (fp32), while the known `T.vdiv` ≤1 ulp non-correctly-rounded gap flips the floor only when the quotient is within 1 ulp of an integer (~1/16.7M rate, Stage 3 D). At harness sizes the expected divergence count is ~0, so `max_abs_err=0.00e+00` here is a small-sample result, not proof of bit-exactness. Treat "integration passed" as a gate pass and carry the bounded risk forward (Stage 4 must not assume exactness either). | C | `run.json` correctness cases (`max_abs_err=0.00e+00`, N=4096/16384); Stage 3 retrospective D (`T.vdiv` mismatch 1/16777216, `repro/TRAP-vdiv-fp32-not-correctly-rounded.py`); `tests/ops/test_floor_divide.py` L68–L70 (tolerance) | `python examples/floor_divide/floor_divide_fwd_kernel/repro/TRAP-vdiv-fp32-not-correctly-rounded.py` (the gap needs N≈2^24 to surface) | tilelang 0.1.1.10+ubuntu.20.4.cann900 / Ascend910_9362 / CANN 9.0.0 (harness) vs Stage 3 stamp tilelang 0.1.2+417dae9908ad / CANN 8.5.0 | pattern-library traps-runtime（division→discontinuous-function 家族的 tolerance 语义） |

### Transferable Lessons（可迁移教训）

- **First-integration cleanliness depends on the wrapper not having been pre-rewritten.** Run
  `integrate_kernel.py --meta <meta>` on a scaffold whose wrapper still imports
  `._<slug>_fwd_kernels` (helper + kernel names) so the regenerated `{slug}_kernel/__init__.py`
  re-exports all of them; if the wrapper is already in baseline/perf_opt switch-block form,
  re-run with `--extracted-module <slug>_kernel` or the helper re-export is silently dropped and
  smoke fails (eq precedent).
- **A green single-op harness report for a division→floor family op is a gate pass, not a
  bit-exactness certificate.** The frozen correctness sizes (N=4096/16384) and tolerances
  (`atol=1.0` half / `1e-5` fp32) cannot surface the `T.vdiv` ≤1 ulp boundary gap (needs
  ~1/16.7M-scale boundary samples); report the measured `max_abs_err` as a small-sample result
  and carry the bounded risk into later stages.
- **`integrate_kernel.py` reads the programming mode from the Stage 3 product's own literal, not
  from the design doc or the op name.** The `ASCEND_MODE = "Developer"` literal in
  `floor_divide_fwd_kernel.py` drove both the wrapper sync (`wrapper_mode_updated=true`) and the
  mode printed in the `[json]` summary — always confirm that summary line before the report run.
- **Performance numbers in the integration report are Stage-4 inputs, recorded verbatim.** Do not
  tune during Stage 5: capture `run.json` `performance.cases` (latency / ratio / bandwidth) as the
  baseline and leave optimisation to Stage 4.
