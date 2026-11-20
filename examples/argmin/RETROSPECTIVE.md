# RETROSPECTIVE -- argmin (migration-harness, op-level)

> Op-level retrospective for the `argmin` migration unit. Per
> `retrospective-schema.md` §1, migration-harness Stage 1/2/3 function-level
> retrospectives live under `examples/argmin/_argreduce_kernel/RETROSPECTIVE.md`;
> Stage 5 (integrate) is op-level and is recorded here.

---

## Stage 5 (integrate) Retrospective — integration — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` runs a per-function kernel/mode/import integration but never reconciles the Stage-0 wrapper's `default_config` with the NPU kernel's own config selector. For the shared `ArgreduceKernel` the inherited GPU 48 KiB SMEM heuristic is guaranteed to over-select `block_m` on NPU, so integration fails until the wrapper glue is edited by hand. | `tileops/kernels/reduction/argmin/argmin.py` (pre-fix `default_config`) vs `argmin_kernel/_argreduce_kernel.py#_select_config`; failure `examples/TileOPs/reports/tileops/20261014_160039_216067_ArgminFwdOp/report.md` (ub overflow). | Add a wrapper `default_config` delegation check/rewrite to the `add-npu-op` integrate step, or extend the known-fix catalog (CASE-argmax-argreduce-integration defect ②) with the exact delegation code. | R |
| Integrate-flow | The aggregation `argmin_kernel/__init__.py` re-exports only the kernel factory (`_argreduce_kernel`), not the config selector, so the wrapper's delegation must reach into the private module `.argmin_kernel._argreduce_kernel`. | `argmin_kernel/__init__.py` (`__all__=["_argreduce_kernel"]`); wrapper import `from .argmin_kernel._argreduce_kernel import _select_config`. | Have `integrate_kernel.py` re-export (or expose) `_select_config` from the aggregate `__init__.py` so glue does not depend on a private module path. | R |
| Info-source | `CASE-argmax-argreduce-integration` was flagged **stale** by `kb_stale_check` (toolchain stamp differs), yet its defect ② reproduced byte-for-byte on this different toolchain/device. A "stale" marker should not discourage applying a shared-kernel case when the shared class is identical. | knowledge pre-injection block; reproduced on 910B3 / CANN 9.0.0 (same as design's toolchain stamp). | Add a re-verification note to the case entry (bump its toolchain stamp to 910B3/CANN 9.0.0) so it is no longer stale. | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Shared `ArgreduceKernel` NPU integration: wrapper `default_config` must delegate `block_m` to the kernel module's `_select_config` (GPU 48 KiB SMEM heuristic over-selects → BiShengIR `ub overflow`) | P | `argmin.py#default_config` (fixed) + `argmin_kernel/_argreduce_kernel.py#_select_config`; failing report `reports/tileops/20261014_160039_216067_ArgminFwdOp`; passing report `reports/tileops/20261014_160246_645105_ArgminFwdOp` | `cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend /workspace/venv/bin/python -m tileops.reporting.cli run --op ArgminFwdOp --prof-mode msprof` | tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce (dev root build) + CANN 9.0.0 + torch_npu 2.9.0.post8 + Ascend910B3 (20 AICore / 40 Vector) | pattern-library/cases.md (extend CASE-argmax-argreduce-integration) |
| Signature of the wrapper-contract defect: compile-time `ub overflow` on exactly one N_padded (4096) while all smaller shapes pass; plus a hard `ValueError` ("single row ... exceeds 49152-byte budget") at N_padded=102400 (49152//204800==0) | P | failing report the 2 failures are `test_argmin_op[256-4096-*]` only (40/42 pass); lm-head construction error | same command as above, inspect `reports/.../run.json` `correctness`/`failed cases` | same | pattern-library/traps-compiler.md |
| Measured msprof Task Duration for ArgminFwdOp on 910B3 (calibration anchor replacing 910B2C-era numbers): smoke[32,256] 1.80/1.86/1.88 µs (fp16/bf16/fp32, bm=8); lm-head[4,102400] 26.08/31.68 µs (fp16/bf16, tiled bm=1 tn=5120); hidden-state[2048,4096] 31.16/54.10 µs (fp16 bm=2 / bf16 bm=1); non-aligned[256,300] 3.10/3.32 µs (fp16/bf16, bm=8) | D | `examples/TileOPs/reports/tileops/20261014_160246_645105_ArgminFwdOp/benchmark.json` + `run.json#performance.cases` | same command; values in `benchmark.json` (all 9) and `run.json` (6 non-smoke) | pattern-library/constants.md |
| `tileops.reporting.analyzer._is_smoke_record` silently drops any benchmark case whose label contains "smoke" from `run.json#performance.cases`, even with a valid msprof profile; the numbers remain in `benchmark.json` | R | `analyzer.py:27-35,54-56`; smoke-argmin present in `benchmark.json` (9 records) but absent from `run.json` (6 cases) | compare `benchmark.json` (9 records) with `run.json#performance.cases` (6) for the passing run | same | pattern-library / reporting docs |

### Transferable Lessons（可迁移教训）

- For migration-harness integration of a shared reduction kernel (`ArgreduceKernel`, argmax/argmin), the Stage-0 wrapper's `default_config` keeps the GPU 48 KiB SMEM heuristic (`budget // (N_padded*elem)`). On NPU it over-selects `block_m` and the wrapper passes it as an override into the factory, causing BiShengIR compile-time `ub overflow` at `N_padded=4096` and a hard budget `ValueError` at `N_padded=102400`. Always make the wrapper delegate `block_m` to the kernel module's `_select_config`.
- A compile-time `ub overflow` that triggers on exactly one `N_padded` (here 4096) while all smaller shapes pass is the fingerprint of a wrapper config override disagreeing with the kernel's validated selector — not of a kernel algorithm bug. Check the wrapper's `default_config` before touching the kernel.
- `tileops.reporting.analyzer._is_smoke_record` filters any benchmark case whose label contains `smoke` out of `run.json#performance.cases` and `report.md`, even when its msprof profile is valid; quote those values from `benchmark.json`.
