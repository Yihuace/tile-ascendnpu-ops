# Integration Log -- BmmFwdOp

- meta: examples/TileOPs/tileops/kernels/bmm/bmm/.migration_meta.json
- op: BmmFwdOp (op_slug=bmm, family=bmm)
- integrated:
  - _bmm_kernel -> tileops/kernels/bmm/bmm/bmm_kernel/_bmm_kernel.py
- design_docs:
  - _bmm_kernel -> tileops/kernels/bmm/bmm/bmm_kernel/_bmm_kernel_DESIGN.md
    (snapshot of examples/bmm/_bmm_kernel/DESIGN.md, byte-identical)
- ascend_mode: Developer. Stage 3 source declares `ASCEND_MODE = "Developer"` +
  `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`; integrate_kernel.py reported
  `ascend_mode="Developer"`, `wrapper_mode_updated=false` (wrapper `BmmKernel.ascend_mode = "Developer"`
  already matched).
- wrapper: tileops/kernels/bmm/bmm/bmm.py
  - `wrapper_rewritten=true`: the pre-integration import
    `from ._bmm_fwd_kernels import _bmm_kernel` was replaced by the baseline/perf_opt
    source-selection block. Baseline (Stage 3) is ACTIVE:
    `from .bmm_kernel import _bmm_kernel`; the perf_opt import is COMMENTED and points at
    `.bmm_kernel.perf_opt._bmm_kernel` (no perf_opt/ dir exists yet, so baseline is the only valid
    source, as prescribed).
  - Call-site glue also required a fix (see 调试历史). The Stage 0 scaffold invoked the GPU
    nested factory shape; the Stage 3 npuir factory is flat. Two wrapper lines were edited
    (allowed wrapper-glue scope): `_bmm_wrapped_kernel` now calls
    `_bmm_kernel(batch, m, n, k, block_M=, block_N=, block_K=, num_stages=, dtype=)(a, b)`, and
    `BmmKernel.__init__` stores the factory handle (`self.kernel = _bmm_kernel`) instead of
    eagerly invoking it with the old positional signature (eager invocation also ran outside the
    `Kernel.__call__` `scoped_ascend_mode` scope and would have compiled under the ambient mode).
- attempts: 1 (single debug-loop iteration inside attempt 1: initial report failed, one fix, re-run green)

## integrate_kernel.py summary (`[json]` line)
```
{"op_slug": "bmm", "family": "bmm", "functions": ["_bmm_kernel"],
 "sources": {"_bmm_kernel": "examples/bmm/_bmm_kernel/_bmm_kernel.py"},
 "design_docs": {"_bmm_kernel": ".../bmm_kernel/_bmm_kernel_DESIGN.md"},
 "target_dir": ".../tileops/kernels/bmm/bmm/bmm_kernel",
 "wrapper": ".../tileops/kernels/bmm/bmm/bmm.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer", "wrapper_mode_updated": false,
 "reexported": ["_bmm_kernel"], "smoke": "pass",
 "timestamp": "2026-10-14T15:15:45+0800"}
```
- `[smoke] pass` (import of `tileops.kernels.bmm.bmm.bmm_kernel` + `_bmm_kernel` re-export +
  `BmmKernel` attribute check). No `[warn] no DESIGN.md` (Stage 1 DESIGN.md present and copied).

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_152027_623162_BmmFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_152027_623162_BmmFwdOp/report.md
- report_status: partial (run.json `status="partial"`, CLI exit 1)
  - correctness passed; the benchmark stage completed but one workload timed out, so the report
    exited non-zero. `partial` here does NOT indicate an integration failure (per contract).
- correctness: PASS -- 17/17 cases passed (`summary.correctness_passed=true`,
  `correctness_tests=17`, `pass_rate=1.0`; `operators[0].avg_max_abs_err=7.10e-03`,
  `max_abs_err=0.0625`)
  - covers: fp16/bf16 smoke + full, square shapes (b8-1k, b16-256, b1-1k), MHA QK/PV regimes
    (b32-128-512-128, b8-128-128-2048), MoE prefill (b32-256-256-1024), M/N non-aligned
    (b4-200-300-128), plus the 5 error-path validation tests (batch/contraction/rank/dtype/K16).
- benchmark: PARTIAL -- `summary.benchmark_requested=true`, `benchmark_passed=false`;
  `benchmark_tests=5`, `benchmark_failed=1`, `case_count=4`, `msprof_case_count=4`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`
  - 4/5 cases produced valid msprof profiles (kernel filter resolved to `_bmm_main`).
  - 1/5 failed: `square-b4-4k-bfloat16` — msprof subprocess `TimeoutExpired` after 600 s
    (profiler timeout, not a correctness/kernel fault). Not fixed (benchmark is report-only).
  - Roofline Ratio (%) is **N/A** for all 4 valid cases: `_parse_msprof_roofline` could not extract
    the `GM Read + Write` roofline entry from `visualize_data.bin`, so `_build_result` fell back to
    theoretical metrics (`tflops`/`bandwidth_tbs`). `roofline_utilization_percent`,
    `arithmetic_intensity_ops_per_byte`, `computility_tops` are null in run.json. Reported as-is.

### Benchmark numbers (msprof, from run.json `performance.cases`, tag=tileops)
| label | dtype | b | m | n | k | config (bm/bn/bk/stages) | latency (us) | roofline ratio (%) | TOps/s (theoretical) | bandwidth (TB/s, theoretical) | profiler_valid |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|
| square-b8-1k | float16  | 8  | 1024 | 1024 | 1024 | 64/64/64/2 | 564.560 | N/A | 30.43 | 0.0892 | true |
| square-b8-1k | bfloat16 | 8  | 1024 | 1024 | 1024 | 64/64/64/2 | 550.160 | N/A | 31.23 | 0.0915 | true |
| square-b16-512 | float16  | 16 | 512  | 512  | 512  | 64/64/64/2 | 175.240 | N/A | 24.51 | 0.1436 | true |
| square-b16-512 | bfloat16 | 16 | 512  | 512  | 512  | 64/64/64/2 | 175.480 | N/A | 24.48 | 0.1434 | true |
| square-b4-4k | bfloat16 | 4  | 4096 | 4096 | 4096 | 64/64/64/2 | FAILED (msprof 600 s timeout) | N/A | N/A | N/A | false |

- smoke-b4-128 (fp16/bf16) workloads were excluded from benchmark by the runner
  (`pytest -m "not smoke"`), so 5 benchmark tests ran (2+2+1), not 7.
- Performance reported only, not tuned (Stage 4).

## 调试历史
- attempt 1:
  - symptom: initial report run `20261014_151749_268163_BmmFwdOp` — correctness FAILED 12/17:
    `TypeError: unsupported operand type(s) for +: 'int' and 'str'` at
    `bmm_kernel/_bmm_kernel.py:97` (`nt_m = (m + block_M - 1) // block_M`), raised in
    `BmmKernel.__init__`. 5 error-path tests passed (they never build the kernel).
  - classification: **wrapper 胶水错误** (factory call-shape mismatch) — not a kernel/precision
    defect. The Stage 0 scaffold invokes `_bmm_kernel(batch, m, n, k, dtype)(block_m, block_n,
    block_k, num_stages)(a, b)` (GPU nested factory), but the Stage 3 npuir factory is flat and
    already `@tilelang.jit(out_idx=[-1], target="npuir")`-decorated:
    `_bmm_kernel(batch, m, n, k, block_M, block_N, block_K, num_stages, dtype) -> compiled(a,b)`.
    So `dtype` bound to `block_M`.
  - tooling: `tilelang-error-fixer` skill loaded for classification; failure localized with a
    minimal repro `BmmKernel(4,128,128,128, torch.float16)` (no IR dump needed — pure Python
    front-end signature mismatch).
  - backup: `bmm_kernel/history_version/bmm_s5_attempt1.py` (pre-fix wrapper snapshot).
  - fix (wrapper glue only, `tileops/kernels/bmm/bmm/bmm.py`):
    1. `_bmm_wrapped_kernel` body -> `_bmm_kernel(batch, m, n, k, block_M=block_m,
       block_N=block_n, block_K=block_k, num_stages=num_stages, dtype=dtype)(a, b)`.
    2. `BmmKernel.__init__` -> `self.kernel = _bmm_kernel` (store factory handle; do not invoke —
       avoids the wrong positional bind and avoids compiling outside the `Kernel.__call__`
       `scoped_ascend_mode` scope). This also preserves msprof kernel-name derivation (Pass 2
       static scan finds `_bmm_main`).
  - result: localization smoke (BmmFwdOp, fp16/bf16/non-aligned) PASS; re-run report
    `20261014_152027_623162_BmmFwdOp` -> correctness 17/17 PASS, benchmark 4/5 valid.

## issues
- R-type (Integrate-flow): `integrate_kernel.py` rewrites only the wrapper **import**, not the
  wrapper **call site**. When the Stage 3 npuir factory's call shape differs from the Stage 0
  scaffold's assumption (here: flat `@tilelang.jit` factory vs the GPU nested
  `factory(dims)(block_cfg)(inputs)` shape), integration smoke still passes (imports only) and the
  mismatch only surfaces at the report stage. A signature-compatibility precheck (or call-site
  rewrite) would surface it at integration time. Evidence: `bmm.py` pre-fix
  `history_version/bmm_s5_attempt1.py` vs `bmm_kernel/_bmm_kernel.py` L67-78.
- D-type (benchmark, report-only): msprof Roofline `Ratio(%)` unavailable for BmmFwdOp
  (`_parse_msprof_roofline` returned None -> theoretical fallback). Migration baseline table
  therefore cannot carry roofline utilization for this op from this run; only latency + theoretical
  TOps/s + theoretical bandwidth are available.
- D-type (benchmark, report-only): `square-b4-4k-bfloat16` msprof subprocess timeout (600 s). No
  latency captured for the largest workload.
- No frozen-file (tests/ops/benchmarks/workloads/manifest) problem was observed for BmmFwdOp.

## 报告重跑（report re-run）— 2026-10-14

集成脚本未重跑（幂等产物未变）；仅重跑单算子报告，用于验证 msprof Roofline `Ratio(%)`
容错修复（commit `b192f3d`，`tileops/benchmark/msprof_roofline.py::parse_bin_file` 改用
`safe_float`）后的结果。

- report_run_json: examples/TileOPs/reports/tileops/20261014_155549_027687_BmmFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_155549_027687_BmmFwdOp/report.md
- report_status: **passed**（run.json `status="passed"`，CLI exit 0；`warnings=[]`）
- correctness: **PASS 17/17**（`summary.correctness_passed=true`，`correctness_tests=17`，
  `pass_rate=1.0`；`operators[0].avg_max_abs_err=7.10e-03`，`max_abs_err=0.0625`）——与上次一致
- benchmark: **PASS 5/5**（`benchmark_requested=true`, `benchmark_passed=true`,
  `benchmark_tests=5`, `benchmark_failed=0`, `case_count=5`, `msprof_case_count=5`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`）
  - **Roofline Ratio 修复确认生效**：5/5 case 的 `roofline_utilization_percent` 均为非空
    （上次 4/4 有效 case 全为 N/A）。`ratio_range = 6.79% – 10.41%`。
  - `square-b4-4k-bfloat16` 上次因 msprof 600 s 超时失败，**本次未复现**：成功产出有效 profile，
    `latency_us=13748.58`，`profiler_valid=true`（非确定性问题，报告只记录不修复）。
  - benchmark.md 另附 `torch-bmm`（`BatchMatMulV2`）基线：Ratio 46.1%–88.9%，latency
    31.42–2292.28 us（本次新增对照，仅记录）。

### Benchmark numbers (msprof, from this run.json `performance.cases`, tag=tileops)

| label | dtype | b | m | n | k | latency (us) | Ratio (%) | AI (Ops/Byte) | Bandwidth (TB/s) | Perf (TOps/s) | Computility (TOps/s) | profiler_valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| square-b8-1k | float16  | 8  | 1024 | 1024 | 1024 | 569.580 | 10.2276 | 2047.9688 | 1.75 | 30.1623 | 294.9120 | true |
| square-b8-1k | bfloat16 | 8  | 1024 | 1024 | 1024 | 559.740 | 10.4074 | 2047.9688 | 1.75 | 30.6926 | 294.9120 | true |
| square-b16-512 | float16  | 16 | 512  | 512  | 512  | 176.960 | 8.2299  | 1023.9688 | 1.75 | 24.2708 | 294.9120 | true |
| square-b16-512 | bfloat16 | 16 | 512  | 512  | 512  | 175.600 | 8.2936  | 1023.9688 | 1.75 | 24.4588 | 294.9120 | true |
| square-b4-4k  | bfloat16 | 4  | 4096 | 4096 | 4096 | 13748.580 | 6.7860 | 2848.2852 | 1.75 | 20.0127 | 294.9120 | true |

- 说明：`Bandwidth (TB/s)` 为 1.75 恒定值，是 roofline 条目中的硬件峰值带宽（`peak`），非本次
  实测达成带宽；`Computility (TOps/s)` = 294.9120 亦为该 roofline 条目的算力峰值。
  上次（N/A 回退）时该列是理论推得的达成带宽，两者口径不同，勿直接对比。
- 性能仅报告，不调优（Stage 4）。
