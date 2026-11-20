# Integration Log -- FP8LightningIndexerOp

- meta: examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/.migration_meta.json
- integrated (kernel):
  - `_fp8_lightning_indexer_kernel` -> fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py
  - `clean_logits_` -> fp8_lightning_indexer_kernel/clean_logits_.py
- design_docs:
  - `_fp8_lightning_indexer_kernel` -> fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel_DESIGN.md
  - `clean_logits_` -> fp8_lightning_indexer_kernel/clean_logits__DESIGN.md
  - (none missing)
- aggregated `__init__.py`: fp8_lightning_indexer_kernel/__init__.py (re-exports both functions)
- wrapper import rewritten: tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py
  - baseline (active): `from .fp8_lightning_indexer_kernel import _fp8_lightning_indexer_kernel, clean_logits_`
  - perf_opt (commented placeholder): `# from .fp8_lightning_indexer_kernel.perf_opt._fp8_lightning_indexer_kernel import _fp8_lightning_indexer_kernel` / `# from .fp8_lightning_indexer_kernel.perf_opt.clean_logits_ import clean_logits_`
  - `ascend_mode`: "Developer" (synced from Stage 1-3 product `ASCEND_MODE`; unchanged across re-integration)
- attempts: 4 (round 1 = Op adapter; round 2 = Op adapter fp8 cast; round 3 = workload window + wrapper UB tiling; round 4 = test comparator NPU fp64 fidelity)

## 验证结果（本轮 = attempt 4 最终态, INTEGRATE_COMPLETED）

- report_run_json: examples/TileOPs/reports/tileops/20261015_164952_078449_FP8LightningIndexerOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261015_164952_078449_FP8LightningIndexerOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261015_164952_078449_FP8LightningIndexerOp/report.html
- report status: **passed** (`summary.status` = passed; `correctness_passed=true`, `benchmark_passed=true`)
- correctness: **pass 4/4** (all cases, incl. the previously failing `[1-4096-32-64-8192-2-True-None]`); `avg max_abs_err = 0.00e+00`
- benchmark: **pass 3/3** (msprof), `profiler_fallback_count=0`, `invalid_profile_count=0`

| nodeid | outcome |
|---|---|
| test_indexer[1-4096-32-64-8192-1-True-None] | passed |
| test_indexer[1-8192-32-64-8192-1-True-None] | passed (was F1) |
| test_indexer[1-4096-32-64-8192-2-True-None] | **passed** (was the comparator artifact) |
| test_indexer_rejects_bf16_inputs_with_external_scale | passed |

Progress: 1/4 (round 1) -> 2/4 (round 2) -> 3/4 (round 3) -> **4/4** (round 4).

### Benchmark 数值（本次 run.json / benchmark.md）

| Label | Latency (us) | Ratio (%) | AchievedBW (TB/s) | config | Shape |
|---|---:|---:|---:|---|---|
| lightning-indexer-smoke-s1k-h8-d64 | 244.0200 | 1.5305 | 1.7500 | {'block_N': 64, 'block_Q': 4, 'num_stages': 2} | {"batch":1,"heads":8,"index_dim":64,"kv_group":1,"seq_len":1024,"seq_len_kv":1024} |
| lightning-indexer-bf16-s8k-h32-d64 | 16413.4609 | 5.7950 | 1.7500 | {'block_N': 64, 'block_Q': 4, 'num_stages': 2} | {"batch":1,"heads":32,"index_dim":64,"kv_group":1,"seq_len":8192,"seq_len_kv":8192} |
| lightning-indexer-bf16-s4k-h32-d64 | 8293.8799 | 5.7341 | 1.7500 | {'block_N': 64, 'block_Q': 4, 'num_stages': 2} | {"batch":1,"heads":32,"index_dim":64,"kv_group":1,"seq_len":4096,"seq_len_kv":8192} |

(Kernel `_fp8_lightning_indexer_main`; msprof captured 1 op/sample, `kernel_name_resolved=_fp8_lightning_indexer_main`, sample_count=1.)

## 授权修复摘要（历史 rounds）

### F1 -- `tileops/workloads/attention_indexing.py` (workload window generator, round 3)

Backup: `fp8_lightning_indexer_kernel/history_version/fp8_lightning_indexer_workload_s5_attempt3.py`

Root cause: `generate_random_cu_seqlens` modelled context parallelism over a global axis
`total_seqlen = seq_len * cp_size`; for the hard-coded `cp_rank=3` every window bound for
`seq_len=8192` landed in `[16384, 20480)` while `seq_len_kv=8192` (all-empty windows; op and
ref both all `-inf`). Fix: single-rank local-axis generation + hardened segmentation + final
`ks/ke = clamp_(0, self.seq_len_kv)`. No assertion / comparator / tolerance / ref semantics
changed. No in-scope assertion weakened.

### F2 -- wrapper `default_config` (round 3)

Backup: `fp8_lightning_indexer_kernel/history_version/fp8_lightning_indexer_wrapper_s5_attempt3.py`

Root cause: grouped path (`kv_group>1`) stages full Q tile in UB; GPU-default `block_Q=4` at
`heads=32, index_dim=64` overflows UB. Fix: adaptive UB-safe `block_Q` (`<=4096` elements),
single-group keeps `block_Q=4`. Baseline/perf_opt import switching block untouched.

## attempt 4 — test comparator NPU fp64 fidelity（本轮, 授权修复）

Backup: `fp8_lightning_indexer_kernel/history_version/test_fp8_lightning_indexer_comparator_s5_attempt4.py`

**Failure**: `test_indexer[1-4096-32-64-8192-2-True-None]`:
`AssertionError: outputs is not close to outputs_ref, difference: -1.1920928955078125e-07; assert 0 <= tensor(-1.19e-07)`.

**Root cause (re-verified this round)**: `_compute_correlation` did `a.data.double(), b.data.double()`
**on NPU tensors**. Ascend910B3 has no fp64, so `.double()` is a silent no-op
(observed `device.double().dtype == torch.float32`); the ~3.2M-element correlation is therefore
reduced in fp32 with ~1e-7 relative rounding whose **sign is data-dependent**. The op matches the
test's own NPU `ref_program` to fp64 precision, so the one-sided `0 <= 1 - corr` assertion flips
on a rounding coin-toss for a numerically correct op.

**Fix (single line + explanatory comment) — restores the original GPU fp64 semantics:**

```diff
     @staticmethod
     def _compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
-        a, b = a.data.double(), b.data.double()
+        # Ascend has no fp64 -> `.double()` on NPU is a silent no-op (stays fp32).
+        # Move to CPU first so the reduction runs in true fp64, matching the GPU comparator.
+        a, b = a.data.cpu().double(), b.data.cpu().double()
         norm_sum = (a * a + b * b).sum()
         return 2 * (a * b).sum() / norm_sum
```

**Hard constraints honored**: `tolerance` unchanged (1e-3); assertion form `0 <= difference <= tolerance`
unchanged; no case skipped/deleted; `ref_program` semantics unchanged; `_validate_tensor_match`
isfinite / exact-equality checks unchanged. This restores the original fp64 reduction, it does NOT
weaken the judgment criterion.

**Evidence — same case, fp32-on-device (before) vs true fp64 (after)** (reproduced this round,
`get_device_backend` seed 1235, script `/tmp/opencode/evidence_fp8_comparator.py`):

| case | seed | `device.double().dtype` | `1-corr` fp32-on-device | verdict | `1-corr` true fp64 | verdict |
|---|---|---|---|---|---|---|
| s4096/skv8192 **g=2** (failing) | 1235 | torch.float32 | **-1.192093e-07** | **FAIL** (`0 <= diff` false) | **+1.887379e-15** | **PASS** |
| s4096/skv8192 g=1 (control) | 1235 | torch.float32 | +5.960464e-08 | PASS | +4.440892e-15 | PASS |

Both true-fp64 values are ~12 orders of magnitude inside the 1e-3 tolerance; the fp32 value's sign
is the sole cause of the failure. After the fix the full report passes 4/4.

## 调试历史

- attempt 1 (round 2): Op adapter fp8-cast blocker -> bf16 carrier pass-through with identity
  fp32 scale. Backup `history_version/fp8_lightning_indexer_op_s5_attempt2.py`. 1/4 -> 2/4.
- attempt 2 (round 3, F1): workload window generator produced all-empty windows for `seq_len=8192`.
  Backup `fp8_lightning_indexer_workload_s5_attempt3.py`. `[1-8192-...-1]` passes.
- attempt 3 (round 3, F2): wrapper `default_config` UB overflow for grouped path; adaptive
  `block_Q`. Backup `fp8_lightning_indexer_wrapper_s5_attempt3.py`. Grouped case complies; exposed
  the comparator artifact.
- attempt 4 (round 4, comparator): NPU `.double()` no-op -> reduce on CPU in true fp64.
  Backup `test_fp8_lightning_indexer_comparator_s5_attempt4.py`. Result: **4/4 + benchmark pass**.
- attempts used: 4.

## issues

- [已修复-F1] workload window generator OOB / degenerate windows (round 3).
- [已修复-F2] grouped-path UB overflow -> adaptive `block_Q` (round 3).
- [已修复-比较器] `_compute_correlation` fp32-on-NPU `.double()` no-op -> `.cpu().double()`
  true-fp64 reduction (round 4); this restores original GPU semantics, no assertion weakened.
- [已知限制-承接] fp8 index-carrier path remains device-restricted on NPU (CG-2026-0022);
  honestly preserved, not faked. Manifest fp8 workload still labelled `status: restricted`.
- [已知告警-非阻塞] `clean_logits_` `block_N=64` non-manifest advisory; not triggered this run.
- [非阻塞/外观] duplicate scaffold comment header above the generated wrapper import block
  (comments only, left untouched).
