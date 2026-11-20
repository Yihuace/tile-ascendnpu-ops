# Conv1dBiasFwdOp — Retrospective

## What worked

- Treating the bias variant as a **thin contract wrapper** was the right call.
  All five migrated Conv1d kernels already support `has_bias=True`; adding a
  second kernel family would have duplicated code and split test coverage.
- Keeping `_select_kernel_key` / `kernel_map` identical to `Conv1dFwdOp`
  means the manifest and reviewer expectations stay comparable across the two
  entries.
- The shared Conv1d Vector fallback (from the Conv1dFwdOp migration) removed
  the CANN 9.0 `bishengir-compile` SIGSEGV class (CG-2026-0003) from the bias
  suite: the full bias suite is now `28 passed` with no compiler crashes.

## What to watch

- A thin variant still needs a complete evidence package. The evidence lives
  under `conv1d_bias_evidence/`, while the kernels and their `has_bias` paths
  remain documented in `../DESIGN.md` and `../REVIEW.md`.
- Any future change to Conv1d dispatch or to the Vector fallback must rerun
  both `test_conv1d.py` and `test_conv1d_bias.py`.
- The manifest intentionally duplicates the Conv1d workload shapes with
  `bias: true`; keep the two blocks in sync when the workload set changes.

## Follow-ups

- If a future toolchain fixes CG-2026-0003, remove the Vector fallback and
  re-benchmark both Conv1d and Conv1dBias together.
- If the GPU baseline later merges the two entries, collapse this wrapper and
  the manifest entry accordingly.
