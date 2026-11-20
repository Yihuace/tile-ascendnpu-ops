# Integration Log -- FP8QuantOp

- meta: examples/TileOPs/tileops/kernels/quantization/fp8_quant/.migration_meta.json
- integrated:
  - `_fp8_quant_kernel` -> `tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/_fp8_quant_kernel.py`
- design_docs:
  - `_fp8_quant_kernel` -> `tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/_fp8_quant_kernel_DESIGN.md`
- ascend_mode: Developer (synced to wrapper `FP8QuantKernel.ascend_mode`, from Stage 3 literal `ASCEND_MODE = "Developer"`)
- attempts: 1

## 集成步骤
- `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/quantization/fp8_quant/.migration_meta.json`
  - kernel + DESIGN.md 快照整文件复制；聚合 `__init__.py` 生成（re-export `_fp8_quant_kernel`）；wrapper import 改写为
    baseline/perf_opt 双 import 切换块（baseline 激活，perf_opt 注释）；import 冒烟 `smoke-ok ['_fp8_quant_kernel']`。
  - 无 `[warn] no DESIGN.md`（Stage 1 门禁产物存在）。

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261014_171628_509361_FP8QuantOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_171628_509361_FP8QuantOp/report.md
- report_status: passed
- correctness: pass (8/8 用例，`summary.correctness_passed=true`；smoke 3 + full 5，含 fp32/fp16/bf16)
- benchmark: passed (3/3，msprof，有效 profile，`invalid_profile_count=0`)
  - kv-index-8k-d64-fp16: 238.4248 us (roofline 1.6194%, BW 1.75 TB/s, kernel `_fp8_quant_fwd_main`)
  - kv-index-8k-d64-bf16: 238.6248 us (roofline 1.6181%, BW 1.75 TB/s)
  - kv-index-4k-d128-fp32: 238.4448 us (roofline 1.5686%, BW 1.75 TB/s)

## 调试历史
- attempt 1:
  - 症状: 集成后直接调用 wrapper `FP8QuantKernel.forward` 报
    `ValueError: Expected 3 inputs, got 1`（`tilelang/jit/jit_npu.py:1281`）。集成 kernel 的编译后签名是显式输出
    `(X(R,D), Scale(R,), OutU8(R,D))`，无 `out_idx`（uint8 out_idx 静默回落 float32，CG-2026-0019）；原生 wrapper
    仍按「单输入 + 返回值」旧签名调用，参数个数不匹配。
  - 分类: wrapper 胶水错误（已知修复目录「wrapper dtype_str/签名不匹配 -> 胶水层转换」；Stage 3 交付说明已预告该签名差异）。
  - 修复: 仅改胶水层 `fp8_quant.py` 的 `_fp8_quant_wrapped_kernel` 实现体 —— 在 host 侧把输入 reshape 到 `(R, D)` 元
    数据视图，分配 fp32 `scale` 与 uint8 字节载体 `out_u8`，以显式输出参数调用 kernel，再用
    `reshape + .view(torch.float8_e4m3fn)` 还原对外契约（scale `[B,S,G]` fp32、out `[B,S,G,D]` fp8_e4m3fn）。
    未改 kernel、dtype/config 传递、测试、manifest 或 DESIGN 快照。
  - 备份: `fp8_quant_kernel/history_version/fp8_quant_s5_attempt1.py`
  - 结果: 修复后 wrapper 冒烟通过（scale [1,256,1] fp32，out [1,256,1,64] float8_e4m3fn，cos≈1.0）；完整单算子
    report 全量正确性 8/8 通过、benchmark 3/3 通过。

## issues
- none（无 `[warn]`/`[error]`；测试/用例/manifest/ops/workloads 均未修改）
