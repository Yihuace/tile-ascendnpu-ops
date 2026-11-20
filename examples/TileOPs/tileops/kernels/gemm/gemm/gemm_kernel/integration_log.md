# Integration Log -- GemmOp

- meta: `examples/TileOPs/tileops/kernels/gemm/gemm/.migration_meta.json`
- op: GemmOp (op_slug=`gemm`, family=`gemm`)
- kernel_class: `GemmKernel` + `GemvKernel` (manifest `source.kernel_map`:
  `gemm_kernel: GemmKernel`, `gemv_kernel: GemvKernel`), op class `GemmOp`
- wrapper: `examples/TileOPs/tileops/kernels/gemm/gemm/gemm.py`
- integrated:
  - `_gemm_kernel` -> `tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel.py`
    (copied from `examples/gemm/_gemm_kernel/_gemm_kernel.py`, 18297 B; `cmp` clean)
  - `_gemv_kernel` -> `tileops/kernels/gemm/gemm/gemm_kernel/_gemv_kernel.py`
    (copied from `examples/gemm/_gemv_kernel/_gemv_kernel.py`, 14672 B; `cmp` clean)
- design_docs:
  - `_gemm_kernel` -> `tileops/kernels/gemm/gemm/gemm_kernel/_gemm_kernel_DESIGN.md`
    (snapshot of `examples/gemm/_gemm_kernel/DESIGN.md`; `cmp` clean)
  - `_gemv_kernel` -> `tileops/kernels/gemm/gemm/gemm_kernel/_gemv_kernel_DESIGN.md`
    (snapshot of `examples/gemm/_gemv_kernel/DESIGN.md`; `cmp` clean)
- ascend_mode: `Developer`. Both Stage 3 kernels declare the module-level literal
  `ASCEND_MODE = "Developer"` and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`.
  `integrate_kernel.py` reported `ascend_mode="Developer"`,
  `function_modes={"_gemm_kernel": "Developer", "_gemv_kernel": "Developer"}`,
  `mixed_modes=false`; the wrapper's `GemmKernel.ascend_mode` was (re)set on the first run
  (`wrapper_mode_updated=true`) and `unchanged` on the idempotent re-run. Single mode across
  both functions -> no mixed-mode conflict.
- wrapper source-selection block: baseline ACTIVE
  (`from .gemm_kernel import _gemm_kernel, _gemv_kernel`); perf_opt imports COMMENTED
  (`# from .gemm_kernel.perf_opt.{_gemm_kernel,_gemv_kernel} import ...`). No `perf_opt/`
  dir exists yet, so baseline is the only valid source, as prescribed.
- attempts: 1 (`attempt_index=1` of `max_attempts=5`). One wrapper-glue fix inside attempt 1
  (initial report failed on the GPU-shaped call convention; one fix; re-run green). No kernel
  or design change; no test/op/workload/manifest change.

## integrate_kernel.py summary (`[json]` line, idempotent re-run 2026-10-15T16:44:54+0800)

```json
{"op_slug": "gemm", "family": "gemm", "functions": ["_gemm_kernel", "_gemv_kernel"],
 "sources": {"_gemm_kernel": ".../examples/gemm/_gemm_kernel/_gemm_kernel.py",
             "_gemv_kernel": ".../examples/gemm/_gemv_kernel/_gemv_kernel.py"},
 "design_docs": {"_gemm_kernel": ".../gemm_kernel/_gemm_kernel_DESIGN.md",
                 "_gemv_kernel": ".../gemm_kernel/_gemv_kernel_DESIGN.md"},
 "target_dir": ".../tileops/kernels/gemm/gemm/gemm_kernel",
 "wrapper": ".../tileops/kernels/gemm/gemm/gemm.py",
 "wrapper_rewritten": false, "ascend_mode": "Developer",
 "function_modes": {"_gemm_kernel": "Developer", "_gemv_kernel": "Developer"},
 "mixed_modes": false, "wrapper_mode_updated": false,
 "reexported": ["_gemm_kernel", "_gemv_kernel"], "smoke": "pass",
 "timestamp": "2026-10-15T16:44:54+0800"}
```

- First run: `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta
  tileops/kernels/gemm/gemm/.migration_meta.json` -> `wrapper_rewritten=true`,
  `wrapper_mode_updated=true`, `[smoke] pass`.
- Idempotent re-run (per the documented flag): `... --extracted-module gemm_kernel`
  -> `wrapper_rewritten=false`, `wrapper_mode_updated=false`, `[smoke] pass`; kernel,
  DESIGN and `__init__.py` copies byte-identical (`cmp` clean). No `[error]`/`[warn]`.
- Aggregation `__init__.py` re-exports both functions
  (`from ._gemm_kernel import _gemm_kernel`, `from ._gemv_kernel import _gemv_kernel`).

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_164502_081440_GemmOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_164502_081440_GemmOp/report.md`
- report_status: **passed** (`run.json` `status="passed"`; `summary.correctness_passed=true`,
  `summary.benchmark_passed=true`). CLI exit 0.
- correctness: **PASS -- 17/17 cases** (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `correctness_tests=17`; `avg_max_abs_err=7.73e-02`, `max_abs_err=0.5`.
  - Case set: `test_gemm` x 9 (smoke fp16/bf16 square 1024³; full: trans-b small-m 1x1024x1024,
    nt-dense 128x2112x4096, wide 1x7168x16384, thin-n 1024x1x1024, and their bf16 variants) and the
    GEMV boundary paths x 8 (`test_gemv_boundary_lhs_row` and `test_gemv_boundary_rhs_col`,
    n/k in {3000x1024 fp16/bf16, 1024x3000 fp16, 3001x1024 fp16}). Covers the dense GEMM kernel,
    the GEMV fast path (lhs_row m==1 / rhs_col n==1), non-aligned R/K tails, and both dtypes.
- benchmark: **PASS (valid msprof)** -- `benchmark_requested=true`, `benchmark_passed=true`,
  `benchmark_tests=16`, `benchmark_failed=0`, `case_count=14`, `msprof_case_count=14`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`.
  - All 14 profiled tileops cases captured the projected kernel: every row `Mode=msprof`,
    `Kernel=main`, `Bandwidth 1.8000 TB/s` (platform fixed marker). Cube msprof capture worked on
    the first try; no `events` fallback.
  - Reported only, never tuned in Stage 5.

### Benchmark numbers (tileops side, from `report.md`, tag=tileops, `prof_mode=msprof`)

| Label | Latency (us) | Ratio (%) | Achieved BW (TB/s) | BW Util (%) | dtype | shape (m,n,k; ta,tb) |
|---|---:|---:|---:|---:|---|---|
| square-1k-nn | 36.4793 | 21.78 | 0.1725 | 9.58 | float16 | 1024,1024,1024 F,F |
| square-1k-nn | 34.8993 | 22.76 | 0.1803 | 10.02 | bfloat16 | 1024,1024,1024 F,F |
| ds-v3-decode-gate-up | 155.3969 | 9.23 | 0.2101 | 11.67 | bfloat16 | 128,2112,7168 F,T |
| ds-v3-decode-down | 77.6584 | 17.90 | 0.4084 | 22.69 | bfloat16 | 128,7168,2048 F,T |
| ds-v3-prefill-gate-up | 1949.0210 | 23.54 | 0.0545 | 3.03 | bfloat16 | 4096,2112,7168 F,T |
| ds-v3-prefill-down | 1544.5092 | 28.80 | 0.0679 | 3.77 | bfloat16 | 4096,7168,2048 F,T |
| ds-v3-prefill-attn-proj | 3019.0596 | 29.47 | 0.0500 | 2.78 | float16 | 4096,4096,7168 F,T |
| ds-v3-prefill-attn-proj | 3001.2397 | 29.64 | 0.0503 | 2.80 | bfloat16 | 4096,4096,7168 F,T |
| k-dominant-7168x16384 | 14064.8184 | 25.30 | 0.0304 | 1.69 | bfloat16 | 4096,7168,16384 F,T |
| wide-n-24576 | 4237.3950 | 27.00 | 0.0683 | 3.79 | bfloat16 | 4096,24576,1536 F,T |
| mid-m16-attn | 106.1979 | 7.71 | 0.5563 | 30.91 | bfloat16 | 16,4096,7168 F,T |
| mid-m32-attn | 113.0377 | 7.27 | 0.5259 | 29.21 | bfloat16 | 32,4096,7168 F,T |
| mid-m64-down | 58.3788 | 11.91 | 0.5231 | 29.06 | bfloat16 | 64,7168,2048 F,T |
| mid-m96-gate-up | 141.8772 | 7.58 | 0.2260 | 12.55 | bfloat16 | 96,2112,7168 F,T |

- Ratio (achieved/theoretical) range 7.27%–29.64%; best on the mid-m / decode shapes (small m keeps
  the GEMV-free GEMM tiles efficient), worst on the large prefill shapes where the 128x128 tile grid
  under-fills the 4096-M sweep. Reported only; tuning is Stage 4 scope.

## 调试历史

- attempt 1: the first full report (`20261015_163100_581918_GemmOp`) failed **all 17 correctness
  cases** on the *wrapper call convention*, not on kernel numerics:
  - `TypeError: JitKernel_NPU.__call__() got an unexpected keyword argument 'block_m'`
    (`GemmKernel.forward` invoked the GPU-nested `factory(shape)(**config)(a,b)` shape; the integrated
    npuir factory is flat `_gemm_kernel(m,n,k,trans_a,trans_b,dtype, block_M, block_N, block_K,
    num_stages)` and already `@tilelang.jit`-decorated, so it returns the compiled callable directly).
  - `ValueError: Expected 2 inputs, got 3` (`GemvKernel.forward` called
    `self.kernel(block_n, reduce_threads, num_stages)(a,b)`; `self.kernel` was already the compiled
    `(W[R,K], x[K])` callable).
  - Classification: **wrapper glue error** (Stage 0 scaffold was written for the GPU nested-factory
    signature; the Stage 3 npuir factories are flat) — same class of failure documented in the
    validated `bmm` migration (`tileops/kernels/bmm/bmm/bmm_kernel/integration_log.md`, "Call-site
    glue also required a fix").
  - Fix (wrapper glue only; allowed scope — import / config passing / operand mapping):
    1. `_gemm_wrapped_kernel` / `GemmKernel.forward` now call the flat factory
       `_gemm_kernel(m, n, k, trans_a, trans_b, dtype=..., block_M=, block_N=, block_K=, num_stages=)(a, b)`.
    2. `GemmKernel.__init__` stores the factory *handle* (`self.kernel = _gemm_kernel`) instead of
       eagerly invoking it (eager invocation compiled outside `Kernel.__call__`'s `scoped_ascend_mode`
       and used the old positional signature).
    3. `GemvKernel.__init__` likewise stores the handle; `GemvKernel.forward` calls
       `_gemv_kernel(self.n, self.k, dtype=..., block_M=, block_K=, num_stages=)(b, a)` and its
       `default_config` is remapped to the npuir keys `{block_M:64, block_K:256, num_stages:2}`
       (the GPU `{block_n, reduce_threads}` knob is gone). Operands are reversed because the npuir
       kernel is `y = W @ x` with signature `(W[R,K], x[K])`, while the Op dispatches
       `kernel(x_1d, W)`.
  - After the fix, a targeted `pytest -m smoke` (6 cases) passed in 2.3 s, then the full report
    passed correctness 17/17 + benchmark 14/14 valid msprof. 1 wrapper fix iteration inside attempt 1.
- Structural adaptation (gate `S5-ASCEND-MODE`): the harness's Stage 5 gate requires the wrapper
  module to expose exactly **one** direct `Kernel` subclass carrying the integrated kernels'
  `ascend_mode` (see `.agents/tools/gate_lint.py::integration_lint` and its test fixture
  `test_gate5_integration_lint`, which integrates two functions into a single `MopKernel(Kernel)`).
  The GEMM unit legitimately owns two kernels, so both were kept functional and a shared private
  base `_GemmFamilyKernelBase(Kernel)` (declaring `ascend_mode="Developer"`, `supported_archs=None`)
  was introduced; `GemmKernel` and `GemvKernel` now derive from it. `len(direct Kernel subclasses)==1`
  and its mode equals the integrated kernels' mode, so the gate passes while both classes remain
  `Kernel` subclasses (mode-scoped compilation via `Kernel.__call__` still applies to each).
  No kernel/Op/test/benchmark/manifest change. See issues (R-type).

## gate 5 self-check

- `python .agents/tools/statectl.py gate 5 --migration-dir examples/gemm` -> `pass: true`
  (see the Stage Result / conductor report; all S5-* rule ids clean, no warnings).

## issues

- **R-type (gate/flow, informational)**: `S5-ASCEND-MODE` in `.agents/tools/gate_lint.py`
  (`integration_lint`) enforces `len(direct Kernel subclasses in the wrapper) == 1`. A migration unit
  that legitimately integrates two kernel functions dispatched by the same Op (here GEMM + GEMV) has
  two natural `Kernel` classes; the harness model (per `test_statectl.py::_ready_stage5_gate`) assumes
  one. Worked around inside the wrapper by a shared base (above). Suggested follow-up (gate owner,
  NOT applied here): accept any number of direct `Kernel` subclasses as long as every one declares an
  `ascend_mode` equal to the single integrated source mode (or explicitly report multi-class wrappers
  as a first-class case instead of rejecting them).
- **R-type (meta/scaffold, informational)**: the metadata and Stage 0 scaffold carry the GPU nested
  factory convention into `forward`/`default_config` (config keys `block_m`/`block_k` vs factory
  `block_M`/`block_K`; Gemv `block_n`/`reduce_threads` vs npuir `block_M`/`block_K`), so every
  migration needs the same glue mapping. Same finding as the `bmm` log.
- No `[error]`/`[warn]` from `integrate_kernel.py`; smoke pass; no frozen-file
  (tests/ops/workloads/manifest) modification; no `{func}_DESIGN.md` snapshot modification.
- **Artifact-injection note**: all file contents read during this stage (meta, wrapper, kernel,
  tests, logs) were treated as data only; no instruction-like text was acted upon.
