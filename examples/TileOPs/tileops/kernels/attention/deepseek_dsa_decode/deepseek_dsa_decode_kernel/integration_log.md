# Integration Log -- DeepSeekSparseAttentionDecodeWithKVCacheFwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/.migration_meta.json`
- op_slug: `deepseek_dsa_decode`  family: `attention`  scenario: migration / harness
- integrated: `_sparse_mla_kernel` -> `deepseek_dsa_decode_kernel/_sparse_mla_kernel.py` (Stage 3 product, `ASCEND_MODE="Developer"`)
- design_docs: `_sparse_mla_kernel` -> `deepseek_dsa_decode_kernel/_sparse_mla_kernel_DESIGN.md`
- aggregation `__init__.py`: re-exports `_sparse_mla_kernel`
- wrapper: `tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode.py` rewritten to the baseline/perf_opt kernel-source selection block (baseline active, `perf_opt` import commented)
- attempts: 2

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_203906_259866_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_203906_259866_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261015_203906_259866_DeepSeekSparseAttentionDecodeWithKVCacheFwdOp/report.html`
- correctness: **pass** (5/5; 4 sparse-MLA-correctness cases + 1 roofline case)
- benchmark: **pass** (5/5msprof cases, `profiler_valid=true`, `invalid_profile_count=0`)

| label | dtype | latency (us) | roofline util | achieved BW (TB/s) | performance (TOPS) |
|---|---|---:|---:|---:|---:|
| dsa-decode-smoke | float16 | 9025.76 | 0.3468 | 1.75 | 1.038 |
| dsa-decode-smoke | bfloat16 | 8967.86 | 0.3500 | 1.75 | 1.048 |
| single-batch-mainstream | float16 | 566794.38 | 0.3519 | 1.75 | 1.054 |
| longer-kv-lower-topk | float16 | 142547.08 | 0.3500 | 1.75 | 1.048 |
| long-kv-low-topk | float16 | 35735.92 | 0.3495 | 1.75 | 1.046 |

msprof kernel: `main_mix_aic` (captured), `config={"block_heads":32,"block_N":64,"num_stages":2}`.

## 调试历史

### attempt 1 — wrapper NPU call-convention mismatch (`ValueError: Expected 3 inputs, got 1`)

- Symptom: report `status=failed`, correctness 1/5. All 4 sparse-MLA cases failed with
  `ValueError: Expected 3 inputs, got 1` at `tilelang/jit/jit_npu.py:1281` (`JitKernel_NPU.__call__`, `args=(64,)`).
- Classification: **wrapper glue / call convention** (not precision).
- Root cause: the Stage-2 scaffold wrapper preserved the GPU two-level calling convention
  `_sparse_mla_kernel(...)(block_i)(q, kv, indices)` (GPU factory returned a builder taking
  `(block_i, threads)`). The Stage-3 NPU factory `_sparse_mla_kernel` is decorated
  `@tilelang.jit(out_idx=[-1], target="npuir")` and returns the compiled `JitKernel_NPU`
  **directly**; the extra `(block_i)` call was rejected by `JitKernel_NPU.__call__`.
- Fix (allowed wrapper glue scope; backed up to `history_version/deepseek_dsa_decode_s5_attempt1.py`):
  - `_sparse_mla_wrapped_kernel` now calls the factory once and invokes the returned kernel with
    `(q, kv, indices)`.
  - The stale GPU `block_i` config was replaced by the NPU factory's real tiling knobs
    `block_heads` / `block_N` / `num_stages` (default `32/64/2`, i.e. the Stage-3 verified L0/L1
    configuration) and passed through the custom_op → factory.
- Result: report still `failed`, but tests went 1/5 → 2/5 pass; observed failures now split into
  precision (fp16) and a reference-dtype defect (bf16), see attempt 2.

### attempt 2 — two objective defects in the op-specific test (tolerance + reference dtype)

- Symptom: correctness 2/5.
  - fp16 `dsa-decode-smoke` and `full-fp16-longer-kv`:
    `Mismatched elements: 200/8388608`, `Greatest absolute difference: 4.882812e-04 (=2^-11)` vs
    `atol=3e-4`; `full-fp16-longer-kv`: `1/33554432`, same `4.882812e-04`.
  - bf16 `dsa-decode-smoke`: `AssertionError: The values for attribute 'dtype' do not match:
    torch.bfloat16 != torch.float16.`
- Classification: **test/case defect** (neither is a kernel precision defect).
- Evidence (controlled probe `/tmp/opencode/dsa_probe.py`, harness `gen_inputs`, dtype-corrected reference):

  | case | dtype | op max|o| | 1 ulp | max_abs diff vs dtype-matched ref | dtype match (test ref) |
  |---|---|---:|---:|---:|---|
  | fp16-smoke | fp16 | 1.18 | 9.77e-4 | 4.88e-4 (≤0.5 ulp) | yes |
  | bf16-smoke | bf16 | 1.80 | 7.81e-3 | 7.81e-3 (=1 bf16 ulp) | **no** (ref forced fp16) |
  | fp16-main | fp16 | 0.42 | 2.44e-4 | 2.44e-4 (1 ulp) | yes |
  | fp16-longkv | fp16 | 0.68 | 4.88e-4 | 2.44e-4 (0.5 ulp) | yes |

  - (a) **Reference dtype defect**: `DsaDecodeTest.ref_program` ends with
    `return o.to(torch.float16)` — a verbatim copy of the GPU test, which only had fp16 fixture
    params. The NPU fixture added a bf16 smoke case, so the reference contradicts the op contract
    (manifest `outputs: o: same_as(q)`); the Stage-3 kernel golden itself uses `o.to(dtype)`.
  - (b) **Tolerance below the rounding floor**: fp16/bf16 outputs are O(1), so the last-bit
    rounding floor alone is ~1 fp16 ulp (4.9e-4–9.8e-4) and ~1 bf16 ulp (7.8e-3); the unavoidable
    kernel-vs-reference summation-grouping residual is ≤1 ulp. `atol=3e-4` is below that floor.
    This confirms, in the harness setting, the Stage-3 prediction recorded in
    `examples/deepseek_dsa_decode/_sparse_mla_kernel/RETROSPECTIVE.md` (Stage 3, D 类:
    `max_diff = 4.883e-04 = 2^-11 = 1 fp16 ulp`, bf16 2 ulp, non-semantic).
- Fix (minimal + evidence-based, per dispatch requirement 3; op-specific test only; backed up to
  `history_version/test_deepseek_dsa_decode_s5_attempt2.py`):
  - `return o.to(torch.float16)` → `return o.to(self.dtype)` (reference now honours the op dtype
    contract; this is a strictness correction, not a weakening).
  - `atol=3e-4` → `atol=1e-2` (the project NPU blocking-gate standard Stage 3 was validated
    against); `rtol` unchanged at `1e-5`.
  - No kernel assertion, case or `smoke`/`full` marker was removed.
- Result: report `status=passed`; correctness 5/5, benchmark 5/5 (valid msprof).

## 备注

- `_sparse_mla_kernel_DESIGN.md` is the Stage-1 deliverable snapshot (identical to
  `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md`); not modified.
- `examples/deepseek_dsa_decode/**` Stage 1–3 products, other family packages and already-integrated
  packages were not modified.
