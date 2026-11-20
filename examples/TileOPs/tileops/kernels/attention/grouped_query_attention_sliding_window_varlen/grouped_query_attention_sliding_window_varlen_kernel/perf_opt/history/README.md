# history/ — superseded Stage-4 records (append-only, never deleted)

| file | content | why superseded |
|---|---|---|
| `perf_records_round0-9_final_sha1216193a.jsonl` | round 0–9 records of the previous Stage-4 session (41 rows). Its `phase=final` rows carry `artifact_sha256 = 1216193a…`, but the surviving `perf_opt/_gqa_sw_fwd_varlen_kernel.py` hashes to `d98cc867…` (the earlier `perf_opt/profiles/`, `logs/`, `bench.py` and `workload_inventory.json` were lost together with it, so the final records could no longer be reproduced or reconciled). | Round 10 (this session) re-measured baseline and final on all 8 tune workloads with the surviving files and wrote a self-consistent `perf_records.jsonl` (single `candidate_id`, sha of the file actually on disk). The old rows are kept verbatim here for provenance. |

Note on the two round-0 duplicate `8b-short` rows documented in `opt_log.md` §2:
both are in the archived file, unchanged.
