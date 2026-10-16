# CANN 9.0 适配记录

## 1. 背景
环境升级到 **CANN 9.0.0**（默认）；现有 tilelang dev-root（`/home/developer/tilelang-mlir-ascend`，`417dae9`）是**针对 CANN 8.5.2 编译**的，9.0 下编译报错。

## 2. 根因（已定位）
- **纯 Vector** kernel 在 9.0 下**编译通过**。
- **MixCV/Expert**（含 `sync_block_set/wait` 的 flag 握手）在 9.0 下报
  `error: Dialect '' not found for custom op 'syn_instr_mode'`。
- `syn_instr_mode` 是 **AscendNPU-IR 源码**（`bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:144`，**可选属性**）里定义、由旧 codegen 附在 sync op 上的；**CANN 9.0 的 bishengir 已不认识它** → 版本不匹配。

## 3. 绕法（小改动，**无需重建**）
在 tilelang 调用 bishengir 前，**删除该可选属性**：
`/home/developer/tilelang-mlir-ascend/tilelang/jit/jit_npu.py` `_npuir_to_bin_enable_npu_compile()` 中，
```python
linalg = self.mlir_content
import re as _re
linalg = _re.sub(r"\s+syn_instr_mode\s*=\s*<[^>]*>", "", linalg)   # CANN9 shim
```
（该文件游离于本仓外，属 kit/toolchain 本地补丁。）

## 4. 适配结果（CANN 9.0 + shim）
- `pytest tests/ops`：**398 passed / 7 failed**（8.5.2 下为 405 passed）。
- **仍失败的 7 个**：`test_conv1d_bias`×5、`test_mamba.py::test_ssd_chunk_scan_fwd`×1、`test_multi_head_attention.py::test_mha_fwd`×1。
  - 失败形态：**9.0 的 `bishengir-compile` SIGSEGV**（`died with <Signals.SIGSEGV: 11>`，LLVM crash backtrace）。
  - 疑因：删掉 `syn_instr_mode` 后这些 kernel 的 sync 语义不完整，或 9.0 后端自身缺陷。

## 5. 结论 / 待办
- **`syn_instr_mode` 类问题**：小 shim 即可解，**不需重建**。
- **剩余 7 个 SEGV**：需更深入处理——① 用**匹配 9.0 的 AscendNPU-IR 重建**（`3rdparty/AscendNPU-IR` 更新版本 + 重建，小时级）；② 或改这些 kernel 的 sync 构造规避；③ 或等 9.0 工具链修复。
- **性能结论**：9.0 下的性能数字需**在 8.5.2 之外单独复测**（结论绑版本）；"删 NaN 恢复"优化在 9.0 下需重验。
