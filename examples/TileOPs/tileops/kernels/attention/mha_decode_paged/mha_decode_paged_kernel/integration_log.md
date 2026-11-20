# Integration Log -- MultiHeadAttentionDecodePagedWithKVCacheFwdOp

- meta: `examples/TileOPs/tileops/kernels/attention/mha_decode_paged/.migration_meta.json`
- op: MultiHeadAttentionDecodePagedWithKVCacheFwdOp (op_slug=mha_decode_paged, family=attention)
- integrated:
  - `_mha_decode_no_split_kernel` -> `tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_no_split_kernel.py`
  - `_mha_decode_split_kernel` -> `tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_split_kernel.py`
- design_docs:
  - `_mha_decode_no_split_kernel` -> `.../mha_decode_paged_kernel/_mha_decode_no_split_kernel_DESIGN.md`
  - `_mha_decode_split_kernel` -> `.../mha_decode_paged_kernel/_mha_decode_split_kernel_DESIGN.md`
- wrapper: `tileops/kernels/attention/mha_decode_paged/mha_decode_paged.py` (import rewritten to the baseline/perf_opt kernel-source selection block; baseline active, both perf_opt lines commented; `ascend_mode="Developer"` synced onto `MHADecodePagedKernel`)
- ascend_mode: Developer (read from both final Stage 3 kernels' `ASCEND_MODE` literal + `os.environ.setdefault("TILELANG_ASCEND_MODE", ...)`; both functions agree -> no mixed-mode)
- attempts: 1 (precision fix applied in attempt 1; full report re-run passed correctness)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_074433_562940_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_074433_562940_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261015_074433_562940_MultiHeadAttentionDecodePagedWithKVCacheFwdOp/report.html`
- report_status: partial (correctness passed; benchmark ran but produced no valid profile records)
- correctness: pass (5/5 cases; avg max_abs_err = 3.51e-04, max = 9.77e-04)
  - `test_mha_decode_paged_op[1-16-1-512-128-128-False-dtype0]` fp16  max_abs_err 2.44e-04
  - `test_mha_decode_paged_op[1-8-1-1024-64-256-False-dtype1]` bf16  max_abs_err 9.77e-04  (split path; thin margin — see 调试历史)
  - `test_mha_decode_paged_op[2-8-1-1024-64-256-False-dtype2]` fp16  max_abs_err 6.10e-05
  - `test_mha_decode_paged_op[1-8-1-512-64-256-False-dtype3]` fp16  max_abs_err 1.22e-04
  - `test_mha_decode_paged_eval_roofline` pass
- benchmark: partial / invalid (4/4 pytest cases ran, exit 0; `benchmark_passed=false`, `invalid_profile_count=4`, `profiler_fallback_count=0`)
  - All four cases use the **two-kernel split path** (host-orchestrated `split_partial` + `combine`); the unfiltered msprof capture contains 8-9 distinct ops (framework Cast/Fill/ZerosLike/ReduceMax + both TileLang kernels), so the framework cannot isolate a single kernel and marks every profile invalid. Recorded (non-authoritative) latencies, `bandwidth_tb_s=1.75`:
    - single-token-page128 fp16 2.52us (ratio 0.02%)
    - batch2-page256 fp16 2.38us (ratio 0.01%)
    - longer-cache fp16 2.32us (ratio 0.01%)
    - shorter-cache fp16 1.82us (ratio 0.02%)
  - Per Stage-5 policy this is **recorded only, not repaired/retried**.

## 调试历史

- attempt 1:
  - 症状: run 1 (`reports/tileops/20261015_072811_852920_...`) correctness 4/5; bf16 split case
    `test_mha_decode_paged_op[1-8-1-1024-64-256-False-dtype1]` failed: `max diff 0.001739501953125 too large (atol=0.001)`.
  - 分类: 精度失败 (precision; not compile/runtime/design). Isolated on NPU with `/tmp/opencode/s5/repro_bf16.py`.
  - 定位: (a) torch emulation of the split/combine algorithm showed the segment-LSE workspace stored in the io dtype
    (bf16) dominated: `lse=bf16,po=bf16` -> 1.3-1.8e-3 vs fp32 golden; `lse=fp32` -> 4e-4;
    (b) NPU test showed `no_split != split` deviation identical (204/189/201 of 512 elements), so the combine adds no error.
    Root cause = `glse` (segment LSE, exponent-domain input to `exp(l_i - LSE)`) stored in bf16.
  - 修复 (kernel + wrapper glue only; Stage 1-3 artifacts untouched):
    - `_mha_decode_split_kernel.py`: `glse` and `Output_partial` prim_func params -> `"float32"` (Phase A + Phase B);
      Phase A writes the fp32 `lse` fragment directly (dropped `lse_cast`) and stages the normalized segment output through an
      fp32 `O_cast` shared buffer (dropped the io-dtype `vcast`); Phase B reads `glse`/`Output_partial` directly into fp32
      fragments (dropped `g16`/`gcol16`/`po_f` io-dtype staging); embedded `_gen_inputs` allocates fp32 workspaces.
    - `mha_decode_paged.py`: split path allocates `glse` / `Output_partial` with `torch.float32`.
  - 结果: bf16 case max_abs_err 1.74e-3 -> 9.77e-04 (< 1e-3); full report correctness 5/5 pass.
    Residual op error (~4-6e-4 ≈ bf16 half-ULP) is the cube second-gemm P-matrix bf16-quantization floor
    (emulation with `P->bf16` reproduces the op exactly: 204/189/198 deviating elements), i.e. inherent, not fixable in Stage 5.

## 集成备注

- `integrate_kernel.py` first run: `smoke=pass`; wrapper import rewritten; `ascend_mode=Developer` synced (`updated`).
  The wrapper's existing prose comment above the import is a scaffold description, not a "Kernel source selection"
  header, so no duplicated selection block was produced.
- Wrapper/kernel interface is closed for the MixCV + two-kernel host orchestration: the split path builds
  `split_length` / `acc_split_length` / fp32 `glse` / fp32 `Output_partial` on `Q.device` and passes all 8 tensors
  to the `npub::mha_decode_paged_split_op` factory, matching the split kernel's `wrapped(...)` signature.
  `acc_split_length` is unused by the kernel (compile-time page partition) but kept for the contract.
- bf16 test margin is thin (9.77e-4 vs atol 1e-3) and the NPU bf16 SDPA reference itself carries ~1-ULP noise
  (`ref != round_to_bf16(fp32_golden)` on ~40% of elements at the same rate as the op). Over 15 extra draws of the
  bf16 shape the test-style max diff was 4.9e-4-1.95e-3 (1/15 >= 1e-3), i.e. the case is inherently marginal at
  `atol=1e-3` (bf16 1 ULP at |O|>=0.25 already >= 1e-3). Documented for the conductor; tests were not modified.
