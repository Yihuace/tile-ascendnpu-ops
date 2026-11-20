# Integration Log -- FusedTopKOp

- meta: `examples/TileOPs/tileops/kernels/moe/fused_topk/.migration_meta.json`
- scope: **integration only** (Stage 5 report / benchmark intentionally NOT run -- conductor task scope)
- integrated:
  - `_fused_topk_kernel` -> `tileops/kernels/moe/fused_topk/fused_topk_kernel/_fused_topk_kernel.py`
    (from `examples/fused_topk/_fused_topk_kernel/_fused_topk_kernel.py`, byte-identical)
- design_docs:
  - `_fused_topk_kernel` -> `_fused_topk_kernel_DESIGN.md`
    (from `examples/fused_topk/_fused_topk_kernel/DESIGN.md`, byte-identical snapshot)
  - missing: none
- ascend_mode: `Developer`. The Stage 3 kernel declares the module-level literal
  `ASCEND_MODE = "Developer"` (line 58) and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`
  (line 44). `integrate_kernel.py` reported `ascend_mode="Developer"`,
  `function_modes={"_fused_topk_kernel": "Developer"}`, `mixed_modes=false`,
  `wrapper_mode_updated=true` (wrapper `FusedTopKKernel.ascend_mode` inserted as `"Developer"`).
- attempts: 0 (no debugging loop needed)

## Integration run (integrate_kernel.py)

Command (cwd `examples/TileOPs/`, `ASCEND_RT_VISIBLE_DEVICES=1`):

```
python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
  --meta tileops/kernels/moe/fused_topk/.migration_meta.json
```

Result summary (`[json]`):

```json
{"op_slug": "fused_topk", "family": "moe", "functions": ["_fused_topk_kernel"],
 "target_dir": ".../tileops/kernels/moe/fused_topk/fused_topk_kernel",
 "wrapper": ".../tileops/kernels/moe/fused_topk/fused_topk.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer",
 "function_modes": {"_fused_topk_kernel": "Developer"}, "mixed_modes": false,
 "wrapper_mode_updated": true, "reexported": ["_fused_topk_kernel"], "smoke": "pass"}
```

Second (idempotency) run: `wrapper_rewritten=false` ("already integrated"),
`wrapper_mode_updated=false` ("unchanged"), `smoke=pass` -- re-run safe.

### Import rewrite (wrapper `fused_topk.py`)

Original: `from ._fused_top_k_kernels import _fused_topk_kernel`
(GPU-extracted file).

Rewritten to a baseline/perf_opt kernel-source selection block (baseline active):

```python
# --- baseline (Stage 3) ------------------------------------------------------
from .fused_topk_kernel import _fused_topk_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .fused_topk_kernel.perf_opt._fused_topk_kernel import _fused_topk_kernel
```

### Generated aggregation `__init__.py`

`fused_topk_kernel/__init__.py` re-exports `_fused_topk_kernel`
(`__all__ = ["_fused_topk_kernel"]`).

## Verification result

- smoke: **pass** (manual re-run)
  - `import tileops.kernels.moe.fused_topk.fused_topk_kernel` -> exports
    `['_fused_topk_kernel']`
  - `import tileops.kernels.moe.fused_topk.fused_topk` -> `FusedTopKKernel` present,
    `FusedTopKKernel.ascend_mode == "Developer"`
- report: **not run** (per task scope). No `run.json` produced in this session;
  therefore `integration_report.json` is intentionally omitted. Correctness /
  benchmark remain UNVERIFIED here -- Stage 5 report gate must be run separately.
- correctness: not evaluated
- benchmark: not evaluated

## Deliverable integrity check

- `diff examples/fused_topk/_fused_topk_kernel/_fused_topk_kernel.py <integrated>` -> IDENTICAL
- `diff examples/fused_topk/_fused_topk_kernel/DESIGN.md <_fused_topk_kernel_DESIGN.md>` -> IDENTICAL
- DESIGN snapshot present: yes (`_fused_topk_kernel_DESIGN.md`, 90163 bytes)

## Debug history

none (integration succeeded on first run; smoke passed)

## Issues

none. No `[error]` / `[warn] no DESIGN.md` emitted by the integration script.
