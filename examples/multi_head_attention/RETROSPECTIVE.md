# RETROSPECTIVE — multi_head_attention (op-level)

> Op-level retrospective for the migration-harness `multi_head_attention` task.
> Per-function retrospectives live under `examples/multi_head_attention/{func}/RETROSPECTIVE.md`.
> This file follows `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md` (append-only).

## Stage 5 (integrate) Retrospective — retry #1 — 2026-10-16

Mode: migration-harness, Stage 5 re-run after an external toolchain fix (user authorized terminal-state reset,
Stage-5-only re-run). Prior terminal state was `[INTEGRATE_FAIL]` (BLOCKED_ENVIRONMENT: fwd Expert kernel's
`syn_instr_mode` keyword not parsed by CANN 9.0.0 `bishengir-compile`). The fix landed in
`tilelang/jit/jit_npu.py` (`_npuir_to_bin_enable_npu_compile` now does
`linalg = linalg.replace("syn_instr_mode", "sync_instr_mode")`), out of this agent's edit scope. This retry
produced: correctness **12/12 pass**, benchmark **partial** (18/20 valid records incl. 8 bwd), verdict
`INTEGRATE_COMPLETED` (with benchmark-partial warning).

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py::gen_init_py` **regenerates** `multi_head_attention_kernel/__init__.py` from only the current batch's function names, silently dropping previously-integrated siblings. For the "same-dir append" integration pattern (fwd package already exists, bwd appended) it destroys the fwd `_gqa_prefill_fwd_kernel` export and the persisted `make_dq_layout` contract name; the script's own import smoke then fails (`missing exports: ['make_dq_layout']`). | `multi_head_attention_kernel/integration_log.md` §"Integration steps" (dry-run plan re-confirms `[init] .../__init__.py` rewrite); `--dry-run` output line `[init] .../multi_head_attention_kernel/__init__.py`; `history_version/__init___s5_attempt1.pre_fix.bak` | Make `gen_init_py` **merge** into an existing `__init__.py` (union existing exports + this batch) instead of overwriting; or require an explicit `--append`/`--merge-init` flag for same-dir append integrations | R |
| Integrate-flow | The add-npu-op known-fix catalog has no entry for the **`__init__.py` append/clobber trap** nor for the **same-dir fwd+bwd multi-`ascend_mode`** structural case, so every integrator must rediscover them. | `multi_head_attention_kernel/integration_log.md` (traps A–C); gate 5 `S5-ASCEND-MODE` conflict noted below | Add "same-dir append integration" section to the add-npu-op reference (init merge, `make_dq_layout` contract re-export, multi-mode wrapper caveat) | R |
| Constraint-doc | gate 5 `S5-ASCEND-MODE` requires exactly one `Kernel` subclass per wrapper with `ascend_mode == integrated mode`; the same-file append contract (fwd `Expert` + bwd `Developer`) is structurally unsatisfiable without editing preserved fwd artifacts. | `multi_head_attention.py` (4 classes: lines 173/342/396/450; `ascend_mode` at 214/366/420/480) | Document the "one op-file appends kernels of a different mode" case explicitly in the integrator constraint doc; track resolution (conductor-enqueued Tier 2 `VP-2026-0289`) | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Toolchain version-skew shim: rewrite `syn_instr_mode`→`sync_instr_mode` in emitted `.npuir` before `bishengir-compile` unblocks CANN 9.0.0 parse failure on 3rdparty-pinned AscendNPU-IR | P | `tilelang/jit/jit_npu.py#_npuir_to_bin_enable_npu_compile` (shim `linalg.replace(...)`); prior `[INTEGRATE_FAIL]` = `Dialect '' not found for custom op 'syn_instr_mode'`; after shim: `12 passed in 8.12s` | `cd examples/TileOPs && python -m tileops.reporting.cli run --op MultiHeadAttentionBwdOp --prof-mode msprof` → `summary.correctness_passed=true` | tilelang g`a3010a66ba5d`, Ascend910B3 x1, CANN 9.0.0, bishengir-compile 428ab8f (2026-04-30), driver 26.1.1, torch 2.9.0 | pattern-library / toolchain-compat |
| Same-dir append integration survives only if `__init__.py` is manually kept as the union (fwd+bwd+`make_dq_layout`); do NOT run `integrate_kernel.py` write mode after a sibling package already exists | P | `multi_head_attention_kernel/__init__.py` (union exports, `__all__` 5 names); `--dry-run` plan shows regen; integration_log §Integration steps | `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta ... --dry-run` (inspect `[init]` line) then verify `diff -q` of package vs `examples/multi_head_attention/{func}/` | same as above | add-npu-op skill reference |
| Case index: `bishengir-compile` **segfault** (not a graceful error) on shape `llama-3.1-70b-long-float16` = batch1/heads64/dim128/seq2048/fp16; bf16 same shape and all other shapes compile — shape+dtype-specific, record-only | C | `.../20261016_082438_849619_MultiHeadAttentionBwdOp/report.md` §Failed Cases (2 rows); `run.json` `summary.benchmark_failed=2` | `python -m tileops.reporting.cli run --op MultiHeadAttentionBwdOp --prof-mode msprof` then inspect `benchmark.json` records | same as above | bottleneck-patterns / compiler-crash index |
| D: MHA-bwd integrated perf baseline (kernel `flash_bwd_prep`): latency 35.2–79.7 us, ratio 24.06%–28.90%, bandwidth 1.75 TB/s over 7 shapes (llama-3.1-8b/70b short/long, fp16+bf16) | D | `.../20261016_082438_849619_MultiHeadAttentionBwdOp/report.md` §Operator Details; `run.json` `operators[0].ratio_range` | `python -m tileops.reporting.cli run --op MultiHeadAttentionBwdOp --prof-mode msprof` | same as above | pattern-library (perf baseline) |

### Transferable Lessons（可迁移教训）

- **`integrate_kernel.py` write mode is not append-safe**: if the target `{op_slug}_kernel/` package already
  exists from a previous function/op batch, its `gen_init_py` regenerates `__init__.py` and drops the earlier
  exports (and any persisted `wrapper_import_names` contract name such as `make_dq_layout`). For harness
  "same-dir append" integration, run `--dry-run` first and keep the union `__init__.py`; never let the
  generator overwrite it.
- **Schema/version skew between the 3rdparty AscendNPU-IR pin and the installed `bishengir-compile` binary is
  a first-class blocker class**: emit-then-parse mismatches (e.g. `syn_instr_mode` vs `sync_instr_mode`)
  surface as `Dialect '' not found for custom op ...` at the compiler stage. Check the toolchain stamp parity
  before blaming the integrated kernel; a tiny string-shim in `jit_npu.py` can clear it.
- **A failing *forward* op inside a shared kernel package blocks *backward* cases too**: if the bwd workload's
  `gen_inputs` runs the fwd op (`MhaBwdWorkload.gen_inputs` → `MultiHeadAttentionFwdOp`), the fwd compile
  failure is reported as bwd case failures. Always attribute per-case failures to the operator actually in the
  failure stack before blaming the integrated unit.
- **Do not count `summary.correctness_tests` as the integrated op's score** when `--op` selects a whole test
  file that also exercises a sibling op: read the `operators` per-operator rollup instead.
- **Report `status=partial` is not integration failure**: it can mean correctness fully passed while the
  benchmark stage had invalid/fallback records (here: 2 msprof→events fallbacks + 2 compiler-crash cases).
  Always read `run.json` `summary`/`operators` to separate the two.
