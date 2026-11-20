# Integration Log -- MoeUnpermuteFwdOp

- meta: examples/TileOPs/tileops/kernels/moe/moe_unpermute/.migration_meta.json
- op_slug: moe_unpermute (family=moe); op_class: MoeUnpermuteFwdOp; kernel_class: MoeUnpermuteKernel
- integrated: `_make_unpermute_kernel` -> `moe_unpermute_kernel/_make_unpermute_kernel.py` (byte-identical copy of the Stage 3 product)
- design_docs: `_make_unpermute_kernel` -> `moe_unpermute_kernel/_make_unpermute_kernel_DESIGN.md` (byte-identical copy of `examples/moe_unpermute/_make_unpermute_kernel/DESIGN.md`)
- target_dir: examples/TileOPs/tileops/kernels/moe/moe_unpermute/moe_unpermute_kernel/
- wrapper: examples/TileOPs/tileops/kernels/moe/moe_unpermute/moe_unpermute.py
- attempts: 0 (integration + import smoke only; no debug closed-loop, report not run per conductor instruction)

## 集成结果（integrate_kernel.py）

Source: `examples/moe_unpermute/_make_unpermute_kernel/_make_unpermute_kernel.py`
(Stage 3 validated, `ASCEND_MODE = "Developer"` + `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`).

- generated `moe_unpermute_kernel/__init__.py`:
  `from ._make_unpermute_kernel import _make_unpermute_kernel` (`__all__ = ["_make_unpermute_kernel"]`).
- wrapper import rewritten -> baseline/perf_opt selection block, **baseline active**:
  ```python
  # --- baseline (Stage 3) ------------------------------------------------------
  from .moe_unpermute_kernel import _make_unpermute_kernel
  # --- perf_opt (Stage 4 tuned) ------------------------------------------------
  # from .moe_unpermute_kernel.perf_opt._make_unpermute_kernel import _make_unpermute_kernel
  ```
  (replaces the former `from ._moe_unpermute_fwd_kernels import _make_unpermute_kernel`).
- wrapper `MoeUnpermuteKernel.ascend_mode = "Developer"` inserted by the script
  (`wrapper_mode_updated=true`; matches the single Stage 3 source mode).
- `mixed_modes=false`, `function_modes={"_make_unpermute_kernel": "Developer"}`.

Script summary (`[json]`):

```json
{"op_slug": "moe_unpermute", "family": "moe", "functions": ["_make_unpermute_kernel"],
 "sources": {"_make_unpermute_kernel": ".../examples/moe_unpermute/_make_unpermute_kernel/_make_unpermute_kernel.py"},
 "design_docs": {"_make_unpermute_kernel": ".../moe_unpermute_kernel/_make_unpermute_kernel_DESIGN.md"},
 "target_dir": ".../moe_unpermute_kernel", "wrapper": ".../moe_unpermute.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer",
 "function_modes": {"_make_unpermute_kernel": "Developer"}, "mixed_modes": false,
 "wrapper_mode_updated": true, "reexported": ["_make_unpermute_kernel"], "smoke": "pass"}
```

Idempotency re-run: `wrapper_rewritten=false`, `wrapper_mode_updated=false`, smoke `pass` (safe to re-run).

## Import smoke

Command (from `examples/TileOPs/`):

```bash
python -c "
import tileops.kernels.moe.moe_unpermute.moe_unpermute_kernel as m
missing = [n for n in ['_make_unpermute_kernel'] if not hasattr(m, n)]
assert not missing, f'missing exports: {missing}'
import tileops.kernels.moe.moe_unpermute.moe_unpermute as w
assert hasattr(w, 'MoeUnpermuteKernel'), 'missing MoeUnpermuteKernel'
print('smoke-ok', m.__all__)
"
```

Result: `smoke-ok ['_make_unpermute_kernel']`; `MoeUnpermuteKernel.ascend_mode = Developer`. PASS.

## 验证结果

- report_run_json: **not_run** — per conductor instruction this stage performs integration + import smoke only; no `tileops.reporting.cli` run, therefore no `run.json` / `report.md` was produced and no `integration_report.json` is written (writing one without a real run would violate the "verdict must come from this run's run.json" rule).
- correctness: not_run (no report)
- benchmark: not_run (no report)

## Issues / risks (not fixed here — report not run)

1. **Wrapper factory-call convention vs integrated kernel factory topology (static analysis, unverified).**
   The GPU-ported wrapper expects a two-level factory: it stores
   `self._unpermute_fn = _make_unpermute_kernel(...)` (`moe_unpermute.py:136`) and then calls
   `fn = self._unpermute_fn()` with **no arguments** (`moe_unpermute.py:196`), mirroring the GPU
   extracted factory (`_make_unpermute_kernel` plain, returning a jit-decorated 0-arg inner).
   The Stage 3 NPU kernel instead applies `@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)`
   **directly to the outer shape-parameterised factory** (`_make_unpermute_kernel.py:147`), whose
   Stage 3 usage is one-step: `kernel = _make_unpermute_kernel(...)` then `kernel(mm2, idx, w, out)`
   (`_make_unpermute_kernel.py:293-300`). Per `tilelang/jit/__init__.py:126-173`, the decorated
   factory already returns a compiled `JITKernel` (`tilelang/jit/kernel.py:177-194`), so the extra
   `()` in `forward` would call the kernel with zero tensor args and raise. This is a wrapper-glue
   mismatch (allowed edit scope) that the (deferred) report correctness run is expected to surface;
   it is recorded rather than fixed because no run could verify a fix under this task's scope.
2. Stale GPU-scaffold file `_moe_unpermute_fwd_kernels.py` remains in the package after the wrapper
   import rewrite. It is no longer imported (harmless); `integrate_kernel.py` does not remove it.
3. `[warn] no DESIGN.md` did not occur — the co-located `DESIGN.md` was found and integrated.

DESIGN snapshot: present and byte-identical to `examples/moe_unpermute/_make_unpermute_kernel/DESIGN.md`
(sha verified by `diff -q`).
