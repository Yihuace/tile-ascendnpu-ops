# Integration Log -- FusedMoeFwdCbFwdOp (fused_topk_cb)

- meta: `examples/TileOPs/tileops/kernels/moe/fused_topk/.migration_meta.json`
  (manifest op = `FusedTopKOp`; this task additionally integrates the correction-bias
  factory `_fused_topk_kernel_cb` so the composed manifest op `FusedMoeFwdCbFwdOp`
  runs end-to-end. The Cb function is **not** listed in that meta's
  `extracted_functions` -- see "Integration approach" below.)
- op: `FusedMoeFwdCbFwdOp` (op_slug `fused_topk_cb`, family `moe`)
- integrated:
  - `_fused_topk_kernel_cb` -> `tileops/kernels/moe/fused_topk/fused_topk_cb_kernel/_fused_topk_kernel_cb.py`
    (byte-identical copy of `examples/fused_topk_cb/_fused_topk_kernel_cb/_fused_topk_kernel_cb.py`)
  - wrapper `tileops/kernels/moe/fused_topk/fused_topk.py` (import + dispatch only)
- design_docs:
  - `_fused_topk_kernel_cb` -> `fused_topk_cb_kernel/_fused_topk_kernel_cb_DESIGN.md`
    (byte-identical snapshot of `examples/fused_topk_cb/_fused_topk_kernel_cb/DESIGN.md`)
  - missing: none
- ascend_mode: `Developer`. Stage 3 kernel declares the module-level literal
  `ASCEND_MODE = "Developer"` (line 61) and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`
  (line 46); the wrapper `FusedTopKKernel.ascend_mode` is already `"Developer"`
  (unchanged by this task).
- attempts: 0 (no debugging loop needed)

## Integration approach

`integrate_kernel.py` was **not** run for this function, by design:

- The `.migration_meta.json` for `fused_topk` describes only the standard
  `_fused_topk_kernel` (`extracted_functions: ["_fused_topk_kernel"]`), which is
  already integrated. The `_fused_topk_kernel_cb` factory is a new sibling
  function of the same Tier-2 task.
- The script hardcodes `<family>/<op_slug>/<op_slug>_kernel` as its target and
  always (re)writes that package's `__init__.py`, so running it again would
  overwrite `fused_topk_kernel/__init__.py` (dropping the standard re-export)
  and had no knowledge of the Cb function.
- Per the conductor's explicit Stage 5 instructions, the Cb factory was placed
  in a sibling package `fused_topk_cb_kernel/` mirroring `fused_topk_kernel/`
  (kernel module + `{func}_DESIGN.md` snapshot + aggregating `__init__.py`), and
  the wrapper was edited to dispatch on `with_correction_bias`.

### Wrapper dispatch (interface contract preserved)

`FusedTopKKernel.__init__` now selects the factory by the pinned flag:

```python
self._kernel_fn = (
    _fused_topk_kernel_cb(...)  if with_correction_bias
    else _fused_topk_kernel(...)
)
```

Both factories keep the harness signature
`factory(num_tokens, num_experts, top_k, scoring_func, with_correction_bias)`
-> `fn(TOKENS_PER_BLOCK)` -> `fn(logits_f32[, bias_f32])` -> `(topk_weights, topk_ids)`.
The Cb kernel uses `@tilelang.jit(out_idx=[2, 3])` for its 4-arg signature
(`gating_output=0, correction_bias=1, topk_weights=2, topk_ids=3`), so
`fn(logits_f32, bias_f32)` returns exactly `(topk_weights, topk_ids)` -- matching
the wrapper's existing Cb call `fn(logits_f32, bias_f32)`. The standard factory
keeps `out_idx=[1, 2]`. No GEMM / main-kernel structure touched.

## Verification result

- report command (cwd `examples/TileOPs`, `ASCEND_RT_VISIBLE_DEVICES=1`,
  `TILEOPS_MSPROF_TMPDIR=/mnt/workspace/tmp`):

  ```
  python3 -m tileops.reporting.cli run --op FusedMoeFwdCbFwdOp \
      --prof-mode msprof --reports-dir reports/tileops --timeout 1200
  ```

- report_run_json: `examples/TileOPs/reports/tileops/20261016_173133_701785_FusedMoeFwdCbFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_173133_701785_FusedMoeFwdCbFwdOp/report.md`
- run status: `passed`
- correctness: **pass** -- 8/8 cases (1 file, `tests/ops/test_moe_fused_moe_cb.py`);
  `correctness_failed=0`, `warnings=[]`. Includes the Cb routing anchors
  (`smoke-bias-bf16`, `smoke-bias-scale-bf16/fp16`, `kimi-k2-*`, `..._op_identity`).
- benchmark: **passed** -- 5/5 benchmark cases ran; `case_count=3` performance
  cases (the 2 smoke cases are reported as benchmark tests but excluded from the
  performance table); all `profiler_valid=true`, `invalid_profile_count=0`.
  - `kimi-k2-decode` (T=512,E=256,K=8,H=7168,F=2048,bf16): latency **18138.26 us**,
    prof_mode msprof, valid, **Ratio 34.65%**
  - `kimi-k2-medium` (T=2048): latency **27183.72 us**, valid, **Ratio 34.49%**
  - `kimi-k2-prefill` (T=4096): latency **38333.89 us**, valid, **Ratio 37.63%**
  - (smoke, not in perf table) `fused-moe-cb-smoke` bf16 7.9998 us / fp16 7.6198 us
- report CLI exit code: 0 (`status=passed`).

### Capacity note (no relaxation applied)

`kimi-k2-*` use E=256, H=7168, F=2048, bf16 -> inherent weight volume
`3*E*F*H*2B = 3*256*2048*7168*2 = 22.5 GiB` (`w_gate_up` 15.0 GiB + `w_down` 7.5 GiB).
This is identical to the already-passing `deepseek-v3-prefill` case of
`FusedMoeFwdOp`, and the msprof path offloads the parent's device copies before the
profiling subprocess, so device peak stays well inside the 61.28 GiB card. All
three big workloads ran without OOM. `inputs.pt` (~22.6 GiB) was written under
`TILEOPS_MSPROF_TMPDIR=/mnt/workspace/tmp` (196 GiB partition) and removed by
`bench_kernel_msprof` after each profile. No shape was shrunk and no case skipped.

## Debug history

- attempt 0 (initial integration): FusedTopKOp Cb direct check on tiny shape
  (T=32,E=8,K=2): `ids_exact=True`, weights == unbiased `sigmoid(logits)` gathered
  at the returned ids (renorm=True and False), `maxdiff=5.96e-08`. No fix needed.
- The only anomalies in the benchmark log were two known toolchain races
  (`bishengir-compile ... err code: -11`, and one msprof `--kernel-name` pass);
  both retried successfully within the harness (documented environment flakiness,
  not a code defect). No source change was made in response.
- No integration debug loop was entered.

## File deltas (all within the allowed edit scope)

| file | change | reason |
|------|--------|--------|
| `tileops/kernels/moe/fused_topk/fused_topk_cb_kernel/_fused_topk_kernel_cb.py` | new (byte-identical copy) | integrate the Stage 3 Cb kernel product |
| `tileops/kernels/moe/fused_topk/fused_topk_cb_kernel/_fused_topk_kernel_cb_DESIGN.md` | new (byte-identical copy) | Stage 1 DESIGN.md snapshot for the Cb function |
| `tileops/kernels/moe/fused_topk/fused_topk_cb_kernel/__init__.py` | new | aggregate re-export of `_fused_topk_kernel_cb` |
| `tileops/kernels/moe/fused_topk/fused_topk.py` | edit (import + dispatch) | dispatch `with_correction_bias=True` to the Cb factory |
| `tileops/kernels/moe/fused_topk/history_version/fused_topk_s5_attempt1.py` | new (backup) | pre-edit snapshot of the wrapper |

Not modified: `tests/`, `benchmarks/`, `tileops/manifest/`, `tileops/ops/`,
`tileops/workloads/`, `examples/{op_slug}/` conductor products, and the integrated
`_fused_topk_kernel_cb_DESIGN.md` snapshot.

## Issues

none. No `[error]` / `[warn] no DESIGN.md`. Correctness and benchmark both pass;
report status `passed`.
