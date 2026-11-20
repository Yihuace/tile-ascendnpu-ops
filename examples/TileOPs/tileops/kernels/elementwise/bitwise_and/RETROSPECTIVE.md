# RETROSPECTIVE — BitwiseAndFwdOp (op_slug=bitwise_and)

## Stage 5 (integrate) Retrospective — attempt2 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 test/workload dtype defect (test/workload not wiring `dtype` into input generation) blocked the end-to-end report for a full Stage-5 attempt; the failure surfaced only at integration, not at Stage 0 gate time. | `bitwise_and_kernel/integration_log.md#调试历史 attempt 1` (11/14, ValueError int64/uint8/int16 vs int32); fixed out-of-band 2026-10-12 19:26 | Add to the migration harness Stage-0/stage-gate checklist: for ops whose tests carry a `dtype` fixture axis, assert the workload `gen_inputs()` emits the fixture dtype (a 1-line smoke that would have caught R-TEST-1/2 before Stage 5). | R |
| Integrate-flow | An interrupted prior Stage-5 session left the wrapper already edited (attempt-2 `_restore_public_dtype`) but `integration_log.md`/`integration_report.json` still pointing at the old failed run; restart had to reconstruct state from `git status` + `history_version/`. | `.../history_version/bitwise_and_s5_attempt2.py` vs current `bitwise_and.py`; `integration_report.json` originally pointed at `20261012_192240...` | Integration skill: require the log/report to be updated in the same step as each wrapper edit (or write an in-flight marker), so a re-dispatch never re-runs against a stale verdict pointer. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir cannot carry uint8 GM tensors; elementwise uint8/bool 1-byte storage is bridged by viewing `int8` in and viewing back out (bit-identical, same itemsize) — two wrapper adapters needed: `_as_kernel_input` (input) and `_restore_public_dtype` (output, to satisfy the frozen Op's same-dtype-in/out contract). | P | `examples/TileOPs/tileops/kernels/elementwise/bitwise_and/bitwise_and.py#94-138,231-245`; DESIGN.md §0.6 R3 | `ASCEND_RT_VISIBLE_DEVICES=1 flock /tmp/tilelang-msprof.lock python3 -m tileops.reporting.cli run --op BitwiseAndFwdOp --prof-mode msprof --reports-dir reports/verification` (uint8 param `test_bitwise_and_op[4096-dtype2]` fails with int8-vs-uint8 `torch.equal` before the restore is added) | tilelang 0.1.1.10+ubuntu.20.4.cann900; Ascend910_9362; CANN 9.0.0; commit 417dae9908ad | pattern-library (elementwise 1-byte dtype adapter) |
| BitwiseAndFwdOp integrated baseline benchmark (msprof, 8 valid cases): util 4.26%–69.92%, BW 1.8 TB/s; int32 [2048,4096] 53.74us/69.92%, bool 22.30us/41.80%. | D | `examples/TileOPs/reports/verification/20261013_084022_118814_BitwiseAndFwdOp/run.json#performance.cases`; `integration_log.md#Benchmark numbers` | `python3 -m tileops.reporting.cli run --op BitwiseAndFwdOp --prof-mode msprof --reports-dir reports/verification` | tilelang 0.1.1.10+ubuntu.20.4.cann900; Ascend910_9362; CANN 9.0.0; commit 417dae9908ad | bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- npuir rejects `uint8` GM tensors (output silently remapped to float32, CG-2026-0018). For 1-byte `uint8`/`bool` elementwise paths, carry the storage as `int8` at the kernel boundary: `view(torch.int8)` on inputs AND `view(public_dtype)` on the kernel output before returning. Bitwise/byte-wise ops are bit-exact under the reinterpretation; the output restore is required because the frozen Op layer assumes same-dtype-in/same-dtype-out.
- A dtype-parametrized correctness test must have its parameter threaded into the workload's `gen_inputs()`; if the fixture only feeds the Op constructor, non-default dtypes fail in the frozen Op dtype guard at Stage 5, not at Stage 0. Verify this before spending a Stage-5 attempt.
- Integration-stage edits are glue-only (wrapper import/config/dtype); both fixes here landed in the wrapper and the kernel file was left byte-identical to the Stage-3 deliverable.
