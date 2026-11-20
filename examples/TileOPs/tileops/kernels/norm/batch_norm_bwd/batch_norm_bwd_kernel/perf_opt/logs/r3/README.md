# R3 (structure deep-dive) archive notes

- `perf_records_before_r3_ab.jsonl` — byte copy of `perf_records.jsonl` taken right
  before the same-session A/B run (`run_r3_ab.py`).
- **Phase relabel disclosure (append-only exception, gate contract)**: gate 4
  requires *exactly one* `phase=final` row per tune workload.  Before the R3
  final measurement, the 12 pre-existing `phase=final` rows (6 from R2 round 2
  `final_wide_subidsplit` / `final_split_subidsplit`, 6 from the R3 merged-file
  repeat-1 run `r3_final_w2s3` at 18:28) were relabelled to `phase=merged`.
  No measurement value, candidate id, artifact path/hash or timestamp was
  altered; every row is still present (see the backup above for the original
  phase labels).
- The authoritative R3 final rows are the 6 `r3_final_w2s3` rows with
  `msprof_raw_path` = `profiles/ab-r0-r3final-*` (same NPU session, paired
  against the R2 best).
