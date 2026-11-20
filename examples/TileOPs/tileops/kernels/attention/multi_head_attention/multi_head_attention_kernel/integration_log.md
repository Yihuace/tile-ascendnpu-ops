# Integration Log -- MultiHeadAttentionBwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/multi_head_attention/.migration_meta.json`
- op: MultiHeadAttentionBwdOp (`op_slug=multi_head_attention`, `family=attention`)
- integrated (same-dir append into the existing fwd package):
  - `_flashattn_bwd_preprocess_kernel.py` -> `multi_head_attention_kernel/_flashattn_bwd_preprocess_kernel.py`
  - `_gqa_bwd_kernel.py`              -> `multi_head_attention_kernel/_gqa_bwd_kernel.py`
  - `_flashattn_bwd_postprocess_kernel.py` -> `multi_head_attention_kernel/_flashattn_bwd_postprocess_kernel.py`
- design_docs:
  - `_flashattn_bwd_preprocess_kernel.md` -> `multi_head_attention_kernel/_flashattn_bwd_preprocess_kernel_DESIGN.md`
  - `_gqa_bwd_kernel` -> `multi_head_attention_kernel/_gqa_bwd_kernel_DESIGN.md`
  - `_flashattn_bwd_postprocess_kernel` -> `multi_head_attention_kernel/_flashattn_bwd_postprocess_kernel_DESIGN.md`
- attempts: 1 (single integration attempt; blocker is out of the in-scope edit set, see below)

## Integration steps

1. `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta ...` (from `examples/TileOPs/`).
   - copied the 3 bwd `{func}.py` + the 3 Stage-1 `{func}_DESIGN.md` snapshots (unmodified) into the
     existing `multi_head_attention_kernel/` package alongside the fwd artifacts (add-only).
   - regenerated `multi_head_attention_kernel/__init__.py`, rewrote/checked the wrapper and synced
     `GQABwdKernel.ascend_mode`.
   - script result: `[wrapper] ... already integrated` (the `rewrite_wrapper_import` guard is tripped by the
     wrapper's commented fwd baseline line -> trap B), `[mode] Developer (updated)`, then
     `[error] import smoke test FAILED` (`missing exports: ['make_dq_layout']`, plus the regenerated
     `__init__.py` dropped the fwd export -> trap A).
2. Applied the three anticipated append traps (all inside the allowed edit set), each backed up under
   `multi_head_attention_kernel/history_version/`:
   - **Trap A** `multi_head_attention_kernel/__init__.py`: rewritten as the fwd + bwd union
     (`_gqa_prefill_fwd_kernel` + the 3 bwd factories) plus a `make_dq_layout` re-export (the persisted
     `wrapper_import_names` contract name; the NPU kernels discard it, but the script's smoke check reads it
     from the package). Backup: `history_version/__init___pre_s5.bak` +
     `history_version/__init___s5_attempt1.pre_fix.bak`.
   - **Trap B** `multi_head_attention.py` bwd import block: replaced the Stage-0 GPU-extracted import
     (`from ._multi_head_attention_bwd_kernels import ...`) with the baseline/perf_opt source-selection
     block importing the 3 bwd factories from `.multi_head_attention_kernel` (fwd block + `GQAPrefillFwdKernel`
     left byte-for-byte unchanged). Backup: `history_version/multi_head_attention_pre_s5.bak` +
     `history_version/multi_head_attention_s5_attempt1.pre_fix.bak`.
   - **Trap C** `ascend_mode`: all three bwd Kernel subclasses now declare `ascend_mode = "Developer"`
     (`GQABwdKernel` was set by the script; `FlashAttnBwdPreprocessKernel` / `FlashAttnBwdPostprocessKernel`
     were added manually, matching their sources' `ASCEND_MODE = "Developer"`). The fwd
     `GQAPrefillFwdKernel` stays `ascend_mode = "Expert"`.
3. Import smoke (replicating the script's check): pass — package exports
   `[_flashattn_bwd_postprocess_kernel, _flashattn_bwd_preprocess_kernel, _gqa_bwd_kernel,
   _gqa_prefill_fwd_kernel, make_dq_layout]`; wrapper exposes all 4 Kernel classes with
   `FlashAttnBwdPostprocessKernel/FlashAttnBwdPreprocessKernel/GQABwdKernel = Developer` and
   `GQAPrefillFwdKernel = Expert`.
4. Single-op report: `python -m tileops.reporting.cli run --op MultiHeadAttentionBwdOp --prof-mode msprof`.

## Verification result

- report_run_json: `examples/TileOPs/reports/tileops/20261016_065819_471539_MultiHeadAttentionBwdOp/run.json`
- report_md:       `examples/TileOPs/reports/tileops/20261016_065819_471539_MultiHeadAttentionBwdOp/report.md`
- correctness: **fail** — `summary.correctness_passed=false`, `4/12` (`operators` rollup: MultiHeadAttentionBwdOp 4 passed / 8 failed)
- benchmark: not run (report skips benchmark because correctness did not pass)
- per-operator rollup (`operators`): 12 cases / 4 passed / 8 failed (the `--op` scope selects the whole
  `tests/ops/test_multi_head_attention.py`, so fwd + bwd cases are both present)

### Root cause of the 8 failures (NOT the bwd integration)

**The three integrated bwd kernels all compile successfully in this report** — the log shows
`AscendNPU IR compile success` for `FlashAttnBwdPreprocessKernel`, `GQABwdKernel` and
`FlashAttnBwdPostprocessKernel`. The 8 failures come from the **forward** Expert kernel:

- `test_mha_fwd[*]` (4 cases) fail compiling the fwd `GQAPrefillFwdKernel`:
  `RuntimeError: NPU IR opt failed: ... error: Dialect`' not found for custom op 'syn_instr_mode'`.
  The fwd Expert (`tilelang_npuir_apis`) codegen emits
  `hivm.hir.sync_block_set[...] flag = %N syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>`
  (from the fwd's explicit `T.sync_block_set`), and the installed
  `bishengir-compile 1.1.0 (/usr1/BiSheng/AscendNPU-IR/bishengir 428ab8fdab46 2026-04-30, CANN 9.0.0)`
  (env `TILELANG_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile`) cannot parse it.
- `test_mha_bwd[*]` (4 cases) fail **downstream of the same fwd failure**: `MhaBwdWorkload.gen_inputs`
  (`tileops/workloads/attention.py:144`) constructs and runs `MultiHeadAttentionFwdOp` to produce the
  backward input `o`; because the fwd cannot compile, the bwd cases never reach the bwd kernels.

**Pre-existing / independent of this integration (evidence):**
- Reverting the wrapper `__init__.py` + `multi_head_attention.py` to their pre-S5 state and re-running
  `test_mha_fwd[smoke-*]` reproduces the identical `syn_instr_mode` failure -> a prior-artifact/toolchain
  issue, not caused by the Stage-5 edits.
- No `MultiHeadAttentionFwdOp` report exists under `reports/tileops/`; the fwd DESIGN/opt_log toolchain
  stamps are CANN 8.5.0 / Ascend910B2C, i.e. the fwd Expert kernel has never been validated on the current
  CANN 9.0.0 / Ascend910B3 build.
- The tilelang build's HIVM syntax (from `3rdparty/AscendNPU-IR` @31f690369, `.td` carries
  `syn_instr_mode`) is ahead of the CANN 9.0.0 `bishengir-compile` binary; the self-built
  `3rdparty/AscendNPU-IR/build/install/bin/bishengir-compile` parses the same `.npuir` past the parse stage
  (fails later only on CANN-version hivmc flag mismatch, as documented in `env.sh`).

### In-scope resolution attempts

None possible: the blocker is in
(i) the fwd Stage-4 artifact `multi_head_attention_kernel/{,_perf_opt/}_gqa_prefill_fwd_kernel.py`
    (must be preserved - add-only integration),
(ii) the workload coupling `tileops/workloads/attention.py` (`MhaBwdWorkload.gen_inputs`, forbidden to edit), and
(iii) the toolchain/compiler version skew (environment).
Per the failure-classification table this is an **environment problem** (BLOCKED_ENVIRONMENT); no allowed
edit can clear it, so the debug attempt budget is not consumed on unmodifiable files.

## Confirmatory check (bwd kernels, independent of the fwd op)

`python examples/multi_head_attention/_gqa_bwd_kernel/_gqa_bwd_kernel.py --level L0` (the kernel's own
golden harness, no TileOPs fwd dependency) was run in this environment to confirm the bwd kernels'
compile+numerics independently of the fwd op.

- **`[L0] ALL PASS (16 items)`, elapsed 844.7s.** 16/16 L0 cases pass (incl. causal / GQA / tail / dim128 /
  strided-write / determinism / reject-guards); no tier-2 relaxation. So the three integrated bwd kernels
  compile and are numerically correct on the current toolchain (CANN 9.0.0 / Ascend910B3).
- This corroborates the report log: the bwd kernels are not the failure source; the report's 8 failures are
  the fwd Expert compile failure + the fwd-dependent bwd `gen_inputs`.

## Debug history

- attempt 1: NOT a precision/compile failure of the integrated kernels. Symptom = report correctness fail on
  *forward* cases (`syn_instr_mode` codegen/compiler mismatch) + bwd cases failing on the fwd-dependent
  `gen_inputs`. Classification = environment/toolchain (BLOCKED_ENVIRONMENT) + test/workload coupling.
  No in-scope fix; terminated without consuming further attempts.

---

## Stage 5 Retry #1 (2026-10-16, new state) — PASS

Prior run (terminal `[INTEGRATE_FAIL]`): correctness **4/12**, root cause = toolchain version skew — the fwd
Expert kernel's codegen emitted `hivm.hir.sync_block_set[...] syn_instr_mode = <...>` which the installed
CANN 9.0.0 `bishengir-compile` could not parse (`Dialect '' not found for custom op 'syn_instr_mode'`);
the bwd cases failed downstream via `MhaBwdWorkload.gen_inputs` running the (uncompilable) fwd op.

- attempts: 1 (this retry). Integration was already complete and stable, so no write-mode re-integration was
  performed (see "Integration steps" below).
- toolchain change (external): `tilelang/jit/jit_npu.py::_npuir_to_bin_enable_npu_compile` now applies the
  version-compat shim `linalg.replace("syn_instr_mode", "sync_instr_mode")` before invoking
  `bishengir-compile`. This clears the parse failure. The shim is **not** modified here (out of scope).

### Integration steps (this retry)

1. `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta ... --dry-run` — plan inspected only.
   The plan re-confirms the known append traps: it would re-copy the 3 bwd `{func}.py` + 3 `{func}_DESIGN.md`
   (harmless, idempotent) **and regenerate `multi_head_attention_kernel/__init__.py`**, which would drop the
   fwd `_gqa_prefill_fwd_kernel` export and the `make_dq_layout` re-export. Write-mode was therefore **not**
   run (per task guidance); the existing stable integrated state was kept.
2. Verified the integrated package is byte-identical to the Stage 1 conductor products (no drift):
   `_flashattn_bwd_preprocess_kernel.py`, `_gqa_bwd_kernel.py`, `_flashattn_bwd_postprocess_kernel.py` and
   their `{func}_DESIGN.md` snapshots all `diff -q` == IDENTICAL to
   `examples/multi_head_attention/{func}/`.
3. Import smoke (re-verify): package exports
   `[_flashattn_bwd_postprocess_kernel, _flashattn_bwd_preprocess_kernel, _gqa_bwd_kernel,
   _gqa_prefill_fwd_kernel, make_dq_layout]`; wrapper Kernel classes
   `FlashAttnBwdPreprocessKernel/FlashAttnBwdPostprocessKernel/GQABwdKernel = Developer`,
   `GQAPrefillFwdKernel = Expert`. Wrapper fwd block + bwd baseline/perf_opt block unchanged.
4. Single-op report re-run (after all edits, so not stale):
   `python -m tileops.reporting.cli run --op MultiHeadAttentionBwdOp --prof-mode msprof`.

### Verification result (this retry)

- report_run_json: `examples/TileOPs/reports/tileops/20261016_082438_849619_MultiHeadAttentionBwdOp/run.json`
- report_md:       `examples/TileOPs/reports/tileops/20261016_082438_849619_MultiHeadAttentionBwdOp/report.md`
- report status: **partial** (CLI exit 1) — but correctness fully passed and benchmark ran.
- correctness: **pass — 12/12, 0 failed** (`summary.correctness_passed=true`, `pass_rate=1.0`;
  pytest: `12 passed in 8.12s`). Per-operator rollup (`operators`): MultiHeadAttentionBwdOp 8/8 passed,
  `avg_max_abs_err=1.16e-03`, `max_abs_err=1.95e-03`. bwd parametrized cases all PASSED
  (smoke-bwd-fp16/bf16, full-bwd-fp16-large, full-bwd-bf16) + fwd cases + dispatch/roofline/reject checks.
- benchmark: **partial** — 20 benchmark tests requested, 18 valid msprof records (8 for
  MultiHeadAttentionBwdOp incl. 7 report rows), 2 failed; 2 fwd msprof cases degraded to `prof_mode=events`.
  - bwd perf (7 rows, kernel `flash_bwd_prep`): latency **35.2–79.7 us**, ratio **24.06%–28.90%**,
    bandwidth **1.75 TB/s** (msprof). Labels: llama-3.1-8b-short/long, llama-3.1-70b-short/long, fp16+bf16.
  - 2 failures (record-only, NOT fixed): `bench_multi_head_attention::test_mha_fwd_bench` and
    `test_mha_bwd_bench` for `llama-3.1-70b-long-float16` (batch=1, heads=64, dim=128, seq_len=2048, fp16)
    both hit a `bishengir-compile` **segfault** (`NPU IR opt failed: PLEASE submit a bug report ... crash
    backtrace`). Shape/dtype-specific compiler crash, independent of the integrated bwd kernels (the same
    shape's bf16 case and all other shapes profile+build fine). Per task policy benchmark failures are
    reported, not repaired.

### Known structural issue (not fixable in the allowed edit set)

- gate 5 `S5-ASCEND-MODE` requires exactly one `Kernel` subclass in the wrapper with an `ascend_mode` equal
  to the integrated kernel mode. The same-file append design puts **4** Kernel classes in
  `multi_head_attention.py` (fwd `GQAPrefillFwdKernel = Expert` + 3 bwd = Developer). Conductor has already
  confirmed this as a structural conflict (enqueued Tier 2 `VP-2026-0289`); the fwd artifacts are
  add-only/preserved and their mode is not to be altered. Reported as-is.

### Debug history (this retry)

- attempt 1: **no debug attempt consumed** — the retry succeeds at the first report run; the prior
  environment blocker was cleared externally by the toolchain shim. The only residual is a shape-specific
  benchmark compiler crash (record-only, out of scope).
