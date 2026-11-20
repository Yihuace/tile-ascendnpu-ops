# RETROSPECTIVE -- gt (op-level, migration-harness)

> Op-level integration retrospective. Stage 1/2/3 per-function retrospectives live in
> `examples/gt/gt_fwd_kernel/RETROSPECTIVE.md`. This file accumulates Stage 5 (integrate) entries only.

## Stage 5 (integrate) Retrospective — GtFwdOp — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` appends a second copy of the source-selection comment header when the Stage 0 scaffold wrapper already carries the same template (baseline pointed at `._gt_fwd_kernels` + commented perf_opt lines); the rewritten wrapper ends up with two adjacent comment-only header blocks. Cosmetic only (gate 5 clean, no code impact). Same defect observed for the ge and eq siblings. | `examples/TileOPs/tileops/kernels/elementwise/gt/gt.py` (two adjacent `# Kernel source selection:` blocks) | `rewrite_wrapper_import` / `gen_kernel_source_block`: detect and replace the pre-existing template block instead of inserting a new one. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| integrate_kernel.py first run must use the meta `extracted_file` stem (`_gt_fwd_kernels`), NOT `--extracted-module gt_kernel`; only a *re-run* after the wrapper has been rewritten needs `--extracted-module {op_slug}_kernel`, else `parse_wrapper_imports` returns empty and the regenerated `{op_slug}_kernel/__init__.py` silently drops helper re-exports (e.g. `coalesce_broadcast_dims`) | R | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L436-L438` (fallback `wrapper_names = list(functions)`), `#L319` (`rewrite_wrapper_import` early-return when `.```{op_slug}_kernel``` import` already present); first-run log `reexported=["coalesce_broadcast_dims","gt_fwd_kernel"]` | first run: `python integrate_kernel.py --meta .../.migration_meta.json` -> smoke pass + both re-exports; re-run without extra flag would drop helpers | tilelang 0.1.1.10+ubuntu.20.4.cann900, Ascend910_9362 x1, CANN 9.0.0, driver 25.5.5, git 2853d71c99 | .agents/skills/add-npu-op/SKILL.md (integration section) |

### Transferable Lessons（可迁移教训）

- On the **first** `integrate_kernel.py` run for a fresh scaffold op, do NOT pass `--extracted-module {op_slug}_kernel`: the scaffold wrapper still imports from `._{func}_kernels`, and that stem is how the script captures the helper re-exports (`coalesce_broadcast_dims`, ...). Reserve `--extracted-module {op_slug}_kernel` for re-runs after the wrapper import has already been rewritten.
- A clean comparison-family (bool-output) integration can be a true one-shot: the Stage 0 scaffold wrapper's factory contract (`kernel(block_size)`, `output_dtype=` bool carrier, `unsigned`/uint8 R5 handled by the Op boundary `view`) matched the Stage 3 kernel unchanged, so no wrapper glue edit was needed beyond the generated switch block.
