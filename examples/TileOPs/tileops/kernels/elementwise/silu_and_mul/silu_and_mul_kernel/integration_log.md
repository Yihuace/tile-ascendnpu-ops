# Integration Log -- SiluAndMulFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/.migration_meta.json
- op: SiluAndMulFwdOp (op_slug=silu_and_mul, family=elementwise)
- integrated:
  - `_make_fused_gated_explicit` -> tileops/kernels/elementwise/silu_and_mul/silu_and_mul_kernel/_make_fused_gated_explicit.py
    (verbatim from examples/silu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py;
    md5 5e609a6f51aa5d85632154c6ad2b548d on both sides)
- design_docs:
  - `_make_fused_gated_explicit` -> tileops/kernels/elementwise/silu_and_mul/silu_and_mul_kernel/_make_fused_gated_explicit_DESIGN.md
    (md5 ceeefb4e9f86de37b9c71baad04e029a on both sides; present)
- ascend_mode: Developer (read from product literal `ASCEND_MODE = "Developer"`; synced to wrapper `Kernel.ascend_mode`, already correct)
- attempts: 1 (one wrapper-glue import fix, see 调试历史)

## 验证结果
- report_run_json: (not run -- report explicitly out of scope for this Stage 5 invocation per task instruction)
- report_md: (not run)
- correctness: not run (per task scope)
- benchmark: not run (per task scope)
- import smoke (integrate_kernel.py, own): PASS
  - `[smoke] smoke-ok ['_make_fused_gated_explicit']`
  - independent re-check: `from tileops.kernels.elementwise.silu_and_mul import SiluAndMulFwdKernel` OK;
    `SiluAndMulFwdKernel.ascend_mode == "Developer"`; wrapper exposes `activation_func`; kernel factory resolved.

## wrapper import 改写（before -> after）
- before (S3 scaffold, in `_silu_and_mul_fwd_kernels.py` module):
  `from ._silu_and_mul_fwd_kernels import ( _make_fused_gated_explicit, activation_func )`
- after (integrate_kernel.py mechanical rewrite, baseline active):
  `from .silu_and_mul_kernel import _make_fused_gated_explicit, activation_func`
  `# from .silu_and_mul_kernel.perf_opt._make_fused_gated_explicit import _make_fused_gated_explicit`
  `# from .silu_and_mul_kernel.perf_opt.activation_func import activation_func`
  -> BROKEN: `activation_func` is not exported by the NPU kernel package (see 调试历史)
- after (Stage 5 wrapper-glue fix + re-run, baseline active):
  `from .silu_and_mul_kernel import _make_fused_gated_explicit`
  `# from .silu_and_mul_kernel.perf_opt._make_fused_gated_explicit import _make_fused_gated_explicit`
  `from . import _silu_and_mul_fwd_kernels as _stage0_contract`
  `activation_func = _stage0_contract.activation_func`

## 调试历史
- attempt 1:
  - 症状: `integrate_kernel.py` 末行 `[error] import smoke test FAILED` (exit 1)。
    `ImportError: cannot import name 'activation_func' from 'tileops.kernels.elementwise.silu_and_mul.silu_and_mul_kernel'`.
  - 分类: wrapper 胶水错误（import 层）。NPU kernel 产品内联 SiLU v-prefix 链，`op_func` 仅作形态标识并 `del`，
    因此不导出 GPU-only helper `activation_func`；而 S3 wrapper 仍从集成包 import 它。
  - 定位: `silu_and_mul_kernel/__init__.py` 只有 `from ._make_fused_gated_explicit import _make_fused_gated_explicit`
    并带脚本自带的 `# [warn] activation_func: not found in integrated modules`；wrapper baseline import 仍带 `activation_func`。
  - 修复（仅 wrapper 胶水层）: 参照同族 `gelu_and_mul` 的 canonical 修复——
    baseline 只 import kernel factory；`activation_func` 从冻结的 Stage-0 契约模块
    (`_silu_and_mul_fwd_kernels.py`) 以模块对象方式导入，保持 wrapper `_effective_op_func` 契约且不破坏脚本幂等。
    备份: silu_and_mul_kernel/history_version/silu_and_mul_s5_attempt1.py。
  - 结果: 重跑 integrate_kernel.py -> `[wrapper] ... already integrated`，`__init__.py` 重新生成（仅导出 factory），
    `[smoke] smoke-ok`，`[ok] integration complete` (exit 0)。

## 备注
- 首次运行时 smoke 亦被同族并发集成的 `gelu_and_mul`（另一 writer，10:16:11）连带阻断（`elementwise/__init__` 传递导入）；
  该 writer 随后以同一 canonical 模式修复，故本 op 修复后 smoke 通过。
- 本阶段未运行 `tileops.reporting.cli`（任务显式排除），故不写 integration_report.json（无本次 run.json）。
