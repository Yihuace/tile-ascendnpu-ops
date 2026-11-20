## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` re-run is not name-preserving for the package `__init__.py`: when the wrapper import has already been rewritten to `.{op_slug}_kernel` (e.g. a partially-assembled package from a prior session), `parse_wrapper_imports(wrapper, "_eq_fwd_kernels")` finds nothing and the script falls back to `functions`, so the regenerated `{op_slug}_kernel/__init__.py` silently drops helper re-exports (here `coalesce_broadcast_dims`) and the import smoke test fails. | examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/integration_log.md#Integration note -- integrate_kernel.py re-run dropped a re-export; run.json smoke evidence: first run `[error] import smoke test FAILED` -> `ImportError: cannot import name 'coalesce_broadcast_dims'` | `integrate_kernel.py`: when `parse_wrapper_imports` returns empty, scan the existing `.{op_slug}_kernel[.mod]` import line (or the current `__init__.py`) before falling back to `functions` | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| harness Stage 5 `integrate_kernel.py` drops helper re-exports on re-run; recover with `--extracted-module {op_slug}_kernel` | R | examples/TileOPs/tileops/kernels/elementwise/eq/eq_kernel/integration_log.md#Integration note (`[json] "reexported": ["coalesce_broadcast_dims", "eq_fwd_kernel"]`, `"smoke": "pass"` after workaround) | on an integration package whose wrapper import was already rewritten: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta>` -> smoke fails importing a helper; recover: same command `--extracted-module eq_kernel` | tilelang 0.1.1.10+ubuntu.20.4.cann900, Ascend910_9362 x1, CANN 9.0.0, git 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b | .agents/skills/add-npu-op/SKILL.md (S5 integration step) |

### Transferable Lessons（可迁移教训）

- In harness Stage 5, re-running `integrate_kernel.py` on an integration package whose wrapper import was **already** rewritten to `.{op_slug}_kernel` is not idempotent for `{op_slug}_kernel/__init__.py`: helper re-exports captured on the first run can be dropped. Pass `--extracted-module {op_slug}_kernel` in that situation, and validate against `[json].reexported` + the smoke result instead of trusting a zero exit code.
- After ANY regeneration of `{op_slug}_kernel/__init__.py`, verify that every name the wrapper imports from the package is still re-exported; a missing helper export surfaces as an `ImportError` in the integrate smoke test (the kernel module itself is fine -- only the aggregation `__init__.py` loses the name).
