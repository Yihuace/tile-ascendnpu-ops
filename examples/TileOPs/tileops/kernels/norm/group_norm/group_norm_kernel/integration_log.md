# Integration Log -- GroupNormFwdOp

- meta: `examples/TileOPs/tileops/kernels/norm/group_norm/.migration_meta.json`
- op_name / op_slug / family: `GroupNormFwdOp` / `group_norm` / `norm`
- kernel class: `GroupNormKernel`; op class: `GroupNormFwdOp`
- integrated:
  - `_group_norm_kernel` -> `tileops/kernels/norm/group_norm/group_norm_kernel/_group_norm_kernel.py`
- design_docs:
  - `_group_norm_kernel` -> `tileops/kernels/norm/group_norm/group_norm_kernel/_group_norm_kernel_DESIGN.md`
    (source: `examples/group_norm/_group_norm_kernel/DESIGN.md`; none missing)
- wrapper: `tileops/kernels/norm/group_norm/group_norm.py` (baseline/perf_opt switch block installed; baseline import active, perf_opt commented out)
- wrapper ascend_mode: `Developer` (synced from Stage 3 literal `ASCEND_MODE = "Developer"`; unchanged on re-run)
- aggregation init: `tileops/kernels/norm/group_norm/group_norm_kernel/__init__.py` re-exports `ALIGNMENT`, `_align_up`, `_group_norm_kernel`
- attempts: 2 (attempt 1 = report-only, no in-scope repair; attempt 2 = user-authorized single-site `ops/` fp32-affine fix applied, report re-run and passed)

## 集成步骤

```
cd examples/TileOPs
source /workspace/env.sh            # REQUIRED: repo root on PYTHONPATH for `import tilelang`
python .agents/skills/add-npu-op/scripts/integrate_kernel.py \
    --meta tileops/kernels/norm/group_norm/.migration_meta.json
```

- `[copy]` kernel + `_group_norm_kernel_DESIGN.md` snapshot (whole-file copy); `[init]` aggregation `__init__.py`; `[wrapper]` import rewritten to the baseline/perf_opt selection block; `[mode]` `Developer` (updated); `[smoke] smoke-ok ['ALIGNMENT', '_align_up', '_group_norm_kernel']`; `[json]` summary `design_docs._group_norm_kernel` present, `wrapper_rewritten=false` on the second (idempotent) run.
- No `[warn] no DESIGN.md` (Stage 1 gate product present); no `[error]`.
- Affine layering preserved: kernel-layer `main(x, weight, bias, y)` keeps the row-broadcast `(D_padded,)` affine parameters but the wrapper always feeds the **unit** affine; per-channel `(C,)` weight/bias are applied by the op layer, never passed through to the kernel. No `(C,)` tensor reaches the kernel.
- No-affine variant (`GroupNormNoAffineKernel` / `GroupNormNoAffineFwdOp`) code/import/case was introduced.

> Environment note (documented, not repaired): the first run, executed without `source /workspace/env.sh`,
> completed the integration but its import smoke failed with
> `ModuleNotFoundError: No module named 'tilelang'` (the script's smoke subprocess runs with
> `cwd=examples/TileOPs`, where `tilelang` is only importable when the repo root is on `PYTHONPATH`).
> This is the known environment prerequisite (`/workspace/env.sh`), not an integration defect; the
> re-run under `/workspace/env.sh` passed the smoke. (Same observation as
> `adaptive_avg_pool2d_kernel/integration_log.md` issue 2.)

## 验证结果

### attempt 2 (final, current)

- report_run_json: `examples/TileOPs/reports/tileops/20261016_003328_665600_GroupNormFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_003328_665600_GroupNormFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_003328_665600_GroupNormFwdOp/report.html`
- report_status: **passed** (`exit_codes.correctness=0`, `exit_codes.benchmark=0`; `warnings=[]`)
- correctness: **PASS** -- `summary.correctness_passed=true`, 21 tests, **19 passed / 0 failed / 2 skipped**, `pass_rate=0.9048`
- benchmark: **PASS** -- `summary.benchmark_passed=true`, 7 tests, 7 passed; 4 msprof cases, `invalid_profile_count=0`, `profiler_fallback_count=0`

#### Benchmark numbers (msprof, from run.json `performance.cases`)

| label | dtype | shape / params | latency (us) | roofline util (%) | bandwidth (TB/s) |
|---|---|---:|---:|---:|---:|
| image-g32 | float16 | n=8 c=128 g=32 spatial=[32,32] | 10.7802 | 13.67 | 1.75 |
| image-g32 | bfloat16 | n=8 c=128 g=32 spatial=[32,32] | 11.0602 | 15.38 | 1.75 |
| wider-channel-g32 | float16 | n=4 c=256 g=32 spatial=[28,28] | 8.1802 | 13.44 | 1.75 |
| tail-spatial-g16 | float16 | n=4 c=128 g=16 spatial=[30,30] | 6.0201 | 10.52 | 1.75 |

All 4 profiles valid (`profiler_valid=true`, kernel `main`, sample_count=1). Pass-rate 90.5% is
19/21; the 2 non-passing cases are the multi-device smokes (skipped: single NPU present), not failures.

### attempt 1 (superseded; kept for history)

- report_run_json: `examples/TileOPs/reports/tileops/20261015_162108_863948_GroupNormFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_162108_863948_GroupNormFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261015_162108_863948_GroupNormFwdOp/report.html`
- report_status: **failed** (`exit_codes.correctness=1`, `benchmark=None`; `warnings=[]`)
- correctness: **FAIL** -- `summary.correctness_passed=false`, 21 tests, **12 passed / 7 failed / 2 skipped**, `pass_rate=0.5714`
- benchmark: **not run** (report skips the benchmark when correctness does not pass); no bench numbers to report

### Failed cases (all fp16; all marginal, 1-3 elements each)

| case (nodeid) | dtype | max abs err (report) | greatest abs diff vs allowed |
|---|---|---|---|
| `test_group_norm_op[2-32-spatial1-8-dtype1]` | fp16 | 3.91e-03 | 0.001183 > 0.001 (1/4096 elem) |
| `test_group_norm_op[4-16-spatial4-4-dtype4]` | fp16 | 3.91e-03 | 0.001709 > 0.001 (1/1024 elem) |
| `test_group_norm_op[2-32-spatial9-8-dtype9]` | fp16 | 3.91e-03 | 0.001709 > 0.001 (3/1024 elem) |
| `test_group_norm_op[2-16-spatial10-4-dtype10]` | fp16 | 3.91e-03 | 0.001038 > 0.001 (1/2048 elem) |
| `test_group_norm_op[2-30-spatial11-5-dtype11]` | fp16 | 3.91e-03 | 0.001465 > 0.001 (1/960 elem) |
| `test_group_norm_op[2-32-spatial12-8-dtype12]` | fp16 | 3.91e-03 | 0.001465 > 0.001 (2/3136 elem) |
| `test_group_norm_non_contiguous[2-32-spatial0-8-dtype0]` | fp16 | -- | max err 0.00390625 > 0.001 |

Passing: fp32 (2/2), bf16 (3/3), fp16 with group counts 1/32/16 (3/3).
Skipped: 2 multi-device smoke tests (single NPU present).

## 调试历史 / 根因定位

### attempt 1 -- fp16 acceptance failures are an op-layer dtype-affine double rounding, NOT a kernel defect

**症状**: report `status=failed`, correctness 12/21; the 7 failures are all fp16 and each violates
the fp16 gate (`atol=rtol=1e-3`) on only 1-3 elements (max ~1.0-1.7e-3 vs 1e-3). No crash, no compile
error, no shape error -- `torch.testing.assert_close` numeric mismatch only.

**分类**: 精度失败 (assert_close 超差). Root cause located in a file outside the Stage 5 edit scope.

**定位 (evidence, self-produced in `/tmp/opencode/gn_*.py`)**:

1. The op layer (`tileops/ops/norm/group_norm.py:269-275`, fixed harness) normalizes via
   `kernel(x_2d)` and then applies the per-channel affine **in the input dtype**:
   `y = y * weight.reshape(...) + bias.reshape(...)` (fp16, since `y`, `weight`, `bias` are fp16).
   The golden (`test_group_norm.py::ref_program`) instead computes the whole
   `F.group_norm(x.float(), ..., w.float(), b.float())` in **fp32 opmath** and rounds once with
   `.to(x.dtype)`.
2. Reproduced the exact failing fixture (`GroupNormWorkload.gen_inputs`, seed 1235):
   - `op_out` (fp16 affine) vs golden: 1 element over the criterion per case (`bad=1`).
   - `ideal` = the golden's normalized tensor rounded to fp16, then an **fp32** affine rounded once:
     `bad=0` for every case.
   - ⇒ the residual error is exactly the **fp16 vs fp32 affine arithmetic**, i.e. the op-layer
     dtype-affine double rounding.
3. The integrated kernel is **not** at fault: the kernel's fp16 normalized output is **bit-identical**
   to the golden's normalized tensor (`F.group_norm(x.float(), w=None, b=None)` → `.to(fp16)`):
   `k_norm16 != g_norm16 : 0 / 4096` (also 0/1024 and 0/960 for two other cases). The kernel already
   does the fp32-transit + single round-back internally (DESIGN §0.6 R4); there is no kernel-side
   precision headroom left, and the known-catalog fp16 fix (kernel fp32 intermediate) is already applied.
4. Independent corroboration inside the task workspace: Stage 3 already isolated this exact phenomenon
   and shipped a pure-CPU repro -- `examples/group_norm/_group_norm_kernel/repro/TRAP-fp16-op-affine-double-rounding.py`
   ("with fp16 dtype affine, a non-empty fraction of elements violate |out-golden| <= atol + rtol*|g|;
   applying the affine in fp32 and rounding ONCE gives 0 violations"). The kernel's embedded L0/L1 gate
   used a **unit** affine for fp16 (the kernel's `_affine_for` returns `"unit"` for fp16), so the
   real op-level per-channel fp16 affine was never a blocking gate in Stage 3 and only surfaces here.
5. A wrapper-side fix is not viable: `torch.testing.assert_close` rejects a dtype mismatch, so the
   wrapper cannot return fp32 (the op layer has no round-back); the op layer receives only `x`
   (`kernel(x_2d)`), so the wrapper never sees `weight`/`bias` and cannot apply the affine itself.
   Verified `assert_close(fp32, fp16, ...)` raises `AssertionError: dtype ... do not match`.

**修复**: none applied. The failing arithmetic lives in `tileops/ops/norm/group_norm.py`, which is in
the integrator's forbidden-modification list, and the per-channel weight must not be pushed into the
kernel (task constraint). No in-scope file (integrated kernel or wrapper glue) can change the
op-layer affine dtype. Per the task contract the root cause is reported, not modified.

**Recommended fix (for the conductor / a design revision; not applied here)**:
make the op layer apply the per-channel affine in fp32 and round once, e.g.
`y = (y.float() * weight.float().reshape(affine_shape) + bias.float().reshape(affine_shape)).to(dtype)`.
That is the "valid harness form" of the Stage 3 repro (0 violations) and would keep the kernel,
the tests, and the tolerances unchanged. This is an `ops/`-layer (interface) change and therefore a
conductor/user decision, outside the Stage 5 edit scope.

### attempt 2 -- user-authorized single-site ops fix (fp32 affine, single round-back) -> PASS

**授权**: The conductor (user) explicitly authorized, for this attempt only, a **single minimal fix**
at the `GroupNormFwdOp.forward` per-channel-affine site in `tileops/ops/norm/group_norm.py`: compute
the affine in fp32 and round back to the input dtype exactly once. Scope was frozen to that one
expression; validation / reshape / `_resolve_spec` / `_bind_spec` / `eval_roofline` / kernel call /
other functions (`GroupNormNoAffineFwdOp`) must not change, and no test/tolerance/skip/bench touched.

**备份**: `mkdir -p tileops/kernels/norm/group_norm/group_norm_kernel/history_version &&
cp tileops/ops/norm/group_norm.py
tileops/kernels/norm/group_norm/group_norm_kernel/history_version/group_norm.py_s5_attempt2.bak`
(written before editing; diff vs the live file is exactly this one site).

**修复** (single site, `tileops/ops/norm/group_norm.py:274-283`):

```python
affine_shape = (1, C) + (1,) * len(spatial)
y = (
    y.float() * weight.float().reshape(affine_shape)
    + bias.float().reshape(affine_shape)
).to(dtype)
return y
```

The prior form `y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)` did the affine in the
input dtype (fp16), i.e. a multiply-round + add-round **double rounding** on top of the kernel's already
single-rounded normalized value; the golden instead runs `F.group_norm(x.float(), ..., w.float(), b.float())`
in fp32 opmath and rounds once. The fix matches the golden's opmath path. `diff` confirms the live file
differs from the backup only at this site; nothing else changed.

**分类**: 精度失败 (assert_close, fp16 marginal) -- root cause at the ops (interface) layer;
resolved under explicit user authorization (attempt 1 had correctly ruled this file out of the
default Stage 5 edit scope).

**结果**: re-ran `python -m tileops.reporting.cli run --op GroupNormFwdOp --prof-mode msprof`
(re-integration script re-run first, idempotent, no `[error]`/`[warn]`, mode `Developer`, smoke pass).
Report `status=passed`: correctness 19 passed / 0 failed / 2 skipped and benchmark 7 passed / 4 valid
msprof cases. All 7 previously-failing fp16 cases now pass; the kernel was not touched (it was already
bit-exact vs the golden's normalized tensor -- attempt 1 evidence).

## issues

- **RESOLVED in attempt 2 (user-authorized single-site ops fix).** `tileops/ops/norm/group_norm.py`
  applied the per-channel affine in the input dtype (fp16), a lossy double rounding vs the golden's fp32
  opmath + single round-back; the integrated kernel is bit-exact vs the golden's normalized value.
  Authorized fix: fp32 affine with a single `.to(dtype)` round-back at that one site (backup in
  `history_version/group_norm.py_s5_attempt2.bak`). This is an `ops/`-layer (interface) change and had
  correctly been left out of the default Stage 5 edit scope in attempt 1, then explicitly authorized by
  the conductor for attempt 2 only. No other file changed; tests / tolerances / skips / benchmarks and
  the `_group_norm_kernel_DESIGN.md` snapshot are untouched.
- Attempt 1 consumed no repair budget (report-only, root cause out of default scope); attempt 2 consumed
  one authorized ops-layer repair. No in-scope kernel/wrapper repair was needed.
- Benchmark ran in attempt 2 and passed (7/7, 4 valid msprof cases); numbers recorded under 验证结果.
  No benchmark repair was attempted (benchmark rules: report-only).
- Environment prerequisite recorded: `source /workspace/env.sh` is required before
  `integrate_kernel.py` and `python -m tileops.reporting.cli` (repo root on `PYTHONPATH`).
- Injected-instruction check: no instruction-like text was found in the read artifacts; all read
  content was treated as data. (The attempt-2 authorization originated from the conductor's dispatch
  prompt, not from any read artifact.)
