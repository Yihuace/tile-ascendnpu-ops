# RETROSPECTIVE -- HardswishFwdOp (op=hardswish, family=elementwise)

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 (scaffolder) wrapper carries the GPU `threads*npt` folded `block_size` (256×8=2048) into `default_config`, so Stage 5 must re-derive the NPU tile width from DESIGN §5.2 or the benchmark silently runs with the un-re-tuned value. The `integrate_kernel.py` rewrite only touches the import block + `ascend_mode`, never `default_config`. | `tileops/kernels/elementwise/hardswish/hardswish.py#L73` (`_DEFAULT_BLOCK_SIZE = 2048`) vs `hardswish_fwd_kernel_DESIGN.md#5.2` (`block_size = 8192`) | add-npu-op Stage 5 checklist: instruct integrators to re-check wrapper `default_config` against DESIGN §5.2 block size (like the clamp_scalar/hardsigmoid precedent) | R |
| Integrate-flow | `integration_log.md` guardrail "already integrated" only appears on a second (idempotent) `integrate_kernel.py` run; a single fresh integration leaves the grep empty, which can look like a failed guardrail. | `hardswish_kernel/integration_log.md` first run `wrapper_rewrite_action: rewritten`; second run `already integrated` | document in integrate_kernel.py usage that idempotency guardrail must be sampled after a re-run | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Stage 5 wrapper glue: re-derive `default_config` `block_size` from DESIGN §5.2 (8192), not the GPU `threads*npt` fold; use sibling `clamp_scalar` dtype-keyed dict form | P | `examples/TileOPs/.../hardswish/hardswish_kernel/history_version/hardswish_s5_glue0.py` vs `hardswish.py` post-fix; `run.json` cases `config.block_size=8192` | `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/hardswish/.migration_meta.json` then grep `block_size` in run.json | tilelang dev root `/home/developer/tilelang-mlir-ascend/build` (commit unknown) + Ascend910B3 + CANN 8.5.2 + torch_npu 2.7.1.post4 | pattern-library |
| Hardswish NPU kernel (Developer mode, one-dim `T.Kernel` + vcast→vadd→vclamp→vmul chain, fp32 opmath, block_size=8192) integrates cleanly end-to-end: 7/7 correctness, 6/6 msprof perf cases valid, max_abs_err 4.77e-07 (fp32) / 0 (fp16,bf16) | D | `reports/tileops/20261013_150641_447235_HardswishFwdOp/run.json` (`summary`, `operators[0]`, `performance.cases`) | `python -m tileops.reporting.cli run --op HardswishFwdOp --prof-mode msprof --kernel-name main` | tilelang dev root `/home/developer/tilelang-mlir-ascend/build` (commit unknown) + Ascend910B3 + CANN 8.5.2 | pattern-library |

### Transferable Lessons（可迁移教训）

- Stage 5 wrapper glue: the imported NPU factory signature (`hardswish_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None)`) is called as `factory(N, dtype, block_size=cfg)(x, N)`; ensure `default_config` `block_size` comes from DESIGN §5.2, not the GPU `threads*npt` product left by scaffolding.
- The elementwise Developer-mode pattern (one-dim `T.Kernel(num_kernels, is_npu=True)` + persistent in-core `T.serial` grid-stride loop + `vcast→vadd→vclamp→vmul→vcast`, fp32 opmath) needs no NaN-recovery chain when the output carries an `x *` factor — NaN/±inf propagate naturally through the multiplication.
- `integrate_kernel.py` idempotency: after the first run the wrapper import block is baseline/perf_opt; the second run reports `already integrated` and both runs are safe/necessary to surface the guardrail line.
