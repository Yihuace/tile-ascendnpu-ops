# RETROSPECTIVE — le (op-level)

> Op-level migration retrospective for `le`. Stage 1/2/3 entries live in the
> function-level file `le_fwd_kernel/RETROSPECTIVE.md`; this op-level file holds
> the Stage 5 (integrate) retrospective (integration issues do not belong to a
> single function). Append-only.

## Stage 5 (integrate) Retrospective — first_integration — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py`'s wrapper rewrite inserts the full generated `Kernel source selection` header even when the Stage 0 scaffold already contains an identical template header + commented perf_opt lines, leaving two adjacent (identical, comment-only) header blocks in the wrapper. Observed again for `le` (4th op: ge/eq/gt/le). Cosmetic only; no code impact; gate 5 `S5-WRAPPER-SWITCH` still passes. | `examples/TileOPs/tileops/kernels/elementwise/le/le.py#68-89` (scaffold header) and `#79-90` (generated header): two "Kernel source selection" headers back-to-back; only line 90 `from .le_kernel import ...` is active code | In `integrate_kernel.py::rewrite_wrapper_import`, when the scaffold template header already exists, replace that existing block in place instead of inserting a second generated copy (match the `# Kernel source selection` sentinel and consume up to the existing baseline/perf_opt lines). | R |
| Integrate-flow | No idempotency trap triggered because this was a first run: `parse_wrapper_imports` read the still-unrewritten `from ._le_fwd_kernels import coalesce_broadcast_dims, le_fwd_kernel` and captured both names. On a *re-run* the wrapper no longer imports `._le_fwd_kernels`, so the script must be passed `--extracted-module le_kernel`. This pitfall is real (documented for eq/ge) but was not hit here. | `examples/TileOPs/tileops/kernels/elementwise/le/le_kernel/integration_log.md#integrate_kernel.py summary`; wrapper `le.py#90` post-rewrite import | Keep the documented `--extracted-module {op_slug}_kernel` re-run workaround visible in the integrator prompt/skill (already present). Optional hardening: make `rewrite_wrapper_import` short-circuit when `from .{op_slug}_kernel ... import` is already present (already returns False) AND make `parse_wrapper_imports` fall back to names parsed from the active baseline import rather than needing the flag. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Comparison-family integration needs no wrapper logic edit beyond the source-import switch: the Stage-3 NPU factory `le_kernel/le_fwd_kernel.py::le_fwd_kernel` is call-compatible with the wrapper's original GPU-factory call site (same positional shape/stride args + `output_dtype=` kwarg; extra `block_size=`/`unsigned=` are defaulted), so `integrate_kernel.py` only swaps `._le_fwd_kernels` -> `.le_kernel`. | P | `examples/TileOPs/tileops/kernels/elementwise/le/le.py#90,190-200` (call site unchanged); `examples/le/le_fwd_kernel/le_fwd_kernel.py#323-334` (integrated factory signature) | `python integrate_kernel.py --meta tileops/kernels/elementwise/le/.migration_meta.json` then `run --op LeFwdOp` -> 18/18 correctness | tilelang 0.1.2+d2c9fd059f8bba4dc00ab80f5cdccb026bc27d42; Ascend 910B2C; CANN version not captured in task artifacts | pattern-library / add-npu-op SKILL S5 |

### Transferable Lessons（可迁移教训）

- After integration, the wrapper's baseline source MUST resolve to the integrated NPU module (`from .{op_slug}_kernel import ...`), not the extracted GPU module (`._*_kernels`). Both define a same-named factory (`{func}`) and helper (`coalesce_broadcast_dims`), so a stale import silently selects the GPU factory (which takes `threads`/`num_per_thread` and cannot lower on `target="npuir"`). `integrate_kernel.py` performs this swap automatically — do not hand-edit the import beyond confirming the baseline line is active.
- For the comparison family the op layer hands `torch.bool` operands to the kernel already viewed as `uint8`, while the npuir kernel declares an `int8` carrier; the same-width reinterpretation is accepted by the runtime (verified by the passing `test_le_bool_dtype` / `test_le_integer_dtype[uint8]` cases), so no wrapper-side `.view(torch.int8)` is required.
- The op-level Stage 5 report is the single source of truth for integration verdict: read `summary.correctness_passed` / `summary.benchmark_passed` / `status` from the timestamped `run.json`; gate 5 requires the referenced `run.json` be newer than the wrapper/kernel/`__init__.py` (mtime) or it flags `S5-REPORT-STALE`.
