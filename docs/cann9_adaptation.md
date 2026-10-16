# CANN 9.0 适配记录

## 1. 背景
环境升级到 **CANN 9.0.0**（默认）；现有 tilelang dev-root（`/home/developer/tilelang-mlir-ascend`，`417dae9`）是**针对 CANN 8.5.2 编译**的，9.0 下编译报错。

## 2. 根因（已定位）
- **纯 Vector** kernel 在 9.0 下**编译通过**。
- **MixCV/Expert**（含 `sync_block_set/wait` 的 flag 握手）在 9.0 下报
  `error: Dialect '' not found for custom op 'syn_instr_mode'`。
- `syn_instr_mode` 是 **AscendNPU-IR 源码**（`bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:144`，**可选属性**）里定义、由旧 codegen 附在 sync op 上的；**CANN 9.0 的 bishengir 已不认识它** → 版本不匹配。

## 3. 绕法（小改动，**无需重建**）
在 tilelang 调用 bishengir 前，**把 sync 属性关键字改写为 9.0 的拼写**：
`/home/developer/tilelang-mlir-ascend/tilelang/jit/jit_npu.py` `_npuir_to_bin_enable_npu_compile()` 中，
```python
linalg = self.mlir_content
import re as _re
linalg = _re.sub(r"\bsyn_instr_mode\b", "sync_instr_mode", linalg)   # CANN9 shim
```
（该文件游离于本仓外，属 kit/toolchain 本地补丁。）

> 早期版本曾用「删除属性」(`re.sub(... ``\s+syn_instr_mode\s*=\s*<[^>]*>``, "")`)。删除会让同步模式静默退回默认值 `INTRA_BLOCK_SYNCHRONIZATION`；**改名**保留了真实模式，语义更正确，两者对当前这批 kernel（模式恰为默认值）结果等价。详见 §7。

## 4. 适配结果（CANN 9.0 + shim）
- `pytest tests/ops`：**398 passed / 7 failed**（8.5.2 下为 405 passed）。
- **仍失败的 7 个**：`test_conv1d_bias`×5、`test_mamba.py::test_ssd_chunk_scan_fwd`×1、`test_multi_head_attention.py::test_mha_fwd`×1。
  - 失败形态：**9.0 的 `bishengir-compile` SIGSEGV**（`died with <Signals.SIGSEGV: 11>`，LLVM crash backtrace）。
  - 疑因：删掉 `syn_instr_mode` 后这些 kernel 的 sync 语义不完整，或 9.0 后端自身缺陷。

## 5. 结论 / 待办
- **`syn_instr_mode` 类问题**：小 shim 即可解，**不需重建**。
- **剩余 7 个 SEGV**：需更深入处理——① 用**匹配 9.0 的 AscendNPU-IR 重建**（`3rdparty/AscendNPU-IR` 更新版本 + 重建，小时级）；② 或改这些 kernel 的 sync 构造规避；③ 或等 9.0 工具链修复。
- **性能结论**：9.0 下的性能数字需**在 8.5.2 之外单独复测**（结论绑版本）；"删 NaN 恢复"优化在 9.0 下需重验。

## 6. 对剩余 7 个 SEGV 的进一步排查（2026-10-16）
- 崩溃发生在 **9.0 的 sync op lowering**（stripped npuir **解析通过**、9.0 backend 在 hivm.opt 之后 SEGV）。
- 尝试**改名**（非删除），均**解析失败**（`Dialect '' not found for custom op '<name>'`）：
  - `syn_instr_mode`（旧名）→ 解析失败；
  - `sync_instr_mode`（9.0 strings 里有）→ 解析失败；
  - `tsync_instr_mode`（9.0 strings 里有）→ 解析失败。
-⇒ 9.0 的 `hivm.hir.sync_block_set/wait` **op 定义结构性改版**（该 mode 的 text/属性/属性名与旧版不兼容），**无法靠字符串改名/删除解决**。
- **结论**：剩余 7 个（`conv1d_bias`×5、`ssd`、`mha`）需**匹配 9.0 的 AscendNPU-IR**（更新 submodule + 重建）才能正确生成；当前"strip"shim 只是**保 398/405** 的权宜之计（这 7 个在 strip 下从"解析错"变为"SEGV"，仍不可用）。

## 7. 受限问题（剩余 MixCV/Expert 失败）的最终定位（2026-10-16 深挖）

### 7.1 版本真相（关键发现）
- 本地 tilelang dev 树内 `3rdparty/AscendNPU-IR` = 上游 commit **`31f6903`（2026-10-12）**。
- **CANN 9.0.0 自带的 bishengir-compile / hivmc = 上游 commit `428ab8fdab46` = tag `v1.1.0-post2`（2026-04-30）**。
  （对应关系已在 gitcode 核实：`refs/tags/v1.1.0-post2` 恰好 = `428ab8fdab46`。）
- ⇒ 差异**不是"新旧 CANN"，而是 6 个月的 AscendNPU-IR 版本差**：树是 10 月版，容器里的 CANN 9.0 是 4 月版。

### 7.2 具体不兼容点（已逐一定位）
1. **sync 属性关键字改名**：4 月版 assembly keyword = `sync_instr_mode`，10 月版 = `syn_instr_mode`（枚举值/默认值完全相同）。tilelang 的 Expert codegen（`src/target/codegen_npuir.cc:1053`）**硬编码**输出 `syn_instr_mode`。→ 之前的"strip"或"改名"shim 即针对此点；**改名后解析通过**。
2. **编译器分层错配**：10 月版 `bishengir-compile` 会把 HIVM→LLVM 阶段**委托给外部 `hivmc`**（`runExternalHIVMC`），并向其转发 `--enable-lir-compile` / `--enable-cpu-trace-intrinsic` 等**旧 hivmc 不认识的参数**（版本探测把 4 月版 0.2.0 判为"已知新版"，未走 legacy 兼容分支）。用 wrapper 过滤参数后可继续，但随即在 **`hivm.hir.nd2nz` 合法化**处失败——旧 hivmc 无法 lower 新管线的 IR。
3. **hivmc 是外部（闭源）组件**：`bishengir-hivm-compile` 在 10 月版源码里**只有库、无可执行目标**（`lib/Tools/CMakeLists.txt` 未包含它；上游 tag 同样不含 `tools/bishengir-hivm-compile`）。⇒ **无法自行构建出匹配 10 月树的 hivmc**。

### 7.3 容器内实测结论
- 直接拿 CANN 9.0 的 `bishengir-compile` 编译（改名后、属性完全合规的）MixCV npuir → **确定性 SIGSEGV**（`--tb`/`ulimit -s`/多组参数矩阵均复现；崩溃帧为深层递归，符号被 strip）。即 **4 月版编译器自身在该类 kernel 上的缺陷**。
- 换成 10 月版编译器 → 不再 SEGV，但卡在旧 hivmc 的 `nd2nz` 合法化。
- ⇒ 两侧各缺一半：**4 月编译器崩、10 月编译器缺匹配 hivmc**。

### 7.4 结论 / 可行路径
当前容器**只有** CANN 9.0.0（= AscendNPU-IR `v1.1.0-post2`），而 tilelang 树按 **10 月版** AscendNPU-IR 构建；**匹配 10 月树的 hivmc 只存在于 kit 原环境（CANN 8.5.2 + `~/.local/toolchain`），二者均已不在本容器，kit payload 也不含**。因此：

- **重新 pin 到 `v1.1.0-post2` 重建**（文档早期建议）**不能修复**：该版本的 bishengir-compile 正是崩溃的那个。
- 真正的修复需**获得与 10 月树匹配的 hivmc**，途径仅：
  1. 安装 kit 原 CANN（8.5.2 内部版）+ `~/.local/toolchain`（离线不可得）；
  2. 由 CANN 侧提供更新（≥10 月）的 `hivmc`/`bishengir-compile` 对；
  3. 改这些 kernel 的 sync 构造以规避 4 月编译器的崩溃点（kernel 级 workaround，需逐个试）。
- **现状**：Vector/元素/规约/归一化族**完全可用**；`conv1d_bias`×5 / `ssd` / `mha` 仍受此工具链缺陷所限。

## 8. 落地方案：向 CANN 9.0 对齐（自建 Apr 版 AscendNPU-IR + 重编 tilelang）（2026-10-16 起）

### 8.1 思路
既然 CanN 9.0 只发**编译器二进制**、不发 AscendNPU-IR 开发件（头文件/`lib*.a`/`mlir_core`、`bishengir` python 绑定），而 tilelang 的 C++ 前端必须用这些开发件**编译链接**，那么"用合适的 tilelang"本身就**必须自建一套** AscendNPU-IR。做法：把 AscendNPU-IR 与 tilelang **一起对齐到 CANN 9.0 的 `v1.1.0-post2`（4 月版）**。

- 4 月版源码克隆到**旁路目录** `3rdparty/AscendNPU-IR-apr`（不动当前可用的 Oct 树）。
- 目标：产出 4 月版一致的 `bishengir-compile` / `mlir_core` / `bishengir` 绑定，再用它重编 tilelang，最终让生成的 npuir **与 CANN 9.0 自带编译器同源**，从根上绕开 §7 的错配。

### 8.2 构建前置
- cmake **≥3.28**（Apr 顶层要求）：容器系统 cmake 3.20.5 / ninja 1.10.0 太旧 → 用 `/tmp/opencode/bt-bin`（cmake 3.31.6 + ninja 1.13.2）。
- llvm-project 为 Apr 仓库的 submodule（pin `cd708029e0b2869e80abe31ddb175f7c35361f90`）。**注意**：本次先复用了 Oct 树里已打过补丁的 llvm-project（并跳过 `--apply-patches`），后续若再遇 API 缺失需换回 Apr 的 pin。
- `-DBISHENG_COMPILER_EXECUTABLE=$ASCEND_HOME/bin/bisheng`。

### 8.3 构建中踩到的坑（均已修复）
1. **pybind11 版本过新** → MLIR Python 绑定编译期 `static_assert`（`def_property` 不支持 `keep_alive`，见 `IRCore.cpp`）。
   - 环境 `python3` 默认命中 `/opt/buildtools/.../pybind11` = **3.1.0**（太新）。
   - Oct 树当初能编过，是因为彼时 `~/.local/.../site-packages/pybind11` 存在一个 **2.x** 副本被优先命中；该副本**现已不存在**（环境不可复现的隐患）。
   - 修复：装独立副本 **pybind11 2.12.0**（`/tmp/opencode/pyb212`），构建时注入 `-Dpybind11_DIR=...`（不改系统环境）。
2. **Apr 源码缺一条模板显式实例化** → 链接期 `undefined reference to mlir::tensor::reshape_utils::collapseAndReplace<CollapseShapeOp>`。
   - Apr 的 `bishengir/.../PropagateReshape/Utils.cpp` 只实例化了 `ExpandShapeOp`；而 `PropagateCollapseDown.cpp`（`handleBitcastOp`）会实例化 `CollapseShapeOp`。
   - **Oct 版同文件多这一条**（第 243 行）→ 上游/Oct 已修。
   - 修复：按 Oct 版**补上该显式实例化**（最小、语义无变化的改动）。

### 8.4 构建与切换脚本
- 构建：`build_npuir_apr_bg.sh` / `build_npuir_apr_resume.sh` / `build_npuir_apr_resume2.sh`（增量续编）。
- tilelang：`build_tilelang_apr_bg.sh`（旧 `build/` 自动备份为 `build.oct-bak`）。
- 环境切换：`env_switch_apr.sh`（改 `.bashrc` 指向 Apr 安装，先备份）与 `env_switch_oct.sh`（一键回退）。
- 全程**旁路、可回退**：当前可用环境不改，回退 = 还原 `.bashrc` + `build/` + 重跑 restore（`REBUILD=0`）。

## 9. 2026-10-16 晚：环境被换成 CANN 9.2.0 后的复核与最终根因

### 9.1 环境变更（发生在工作期间）
- 诊断时为 CANN 9.0.0；工作期间容器被替换为 **CANN 9.2.0**（`/home/developer/Ascend/cann-9.2.0` 于 17:04 安装，`cann-9.0.0` 已不存在，`cann -> cann-9.2.0`）。
- `ASCEND_HOME_PATH=.../cann-9.2.0`；`bishengir-compile 1.2.0 (7b16f3c124ac 2026-08-06)`；**`hivmc 0.3.0 (fd235e335e60 2026-08-06)` 存在**（§7.2.3 "hivmc 不可得"的前提已变化）。

### 9.2 §8「向 Apr 对齐」已建成，但对结果无影响
- Phase ①② 完成：自建 Apr `v1.1.0-post2` AscendNPU-IR + 重编 tilelang（0 error，`libtilelang_module.so` 产出）。
- 在 CANN 9.2.0 上与 Oct 原树对比实测，**结果逐项一致**：

| 用例 | Oct 底座 | Apr 底座 |
|---|---|---|
| `test_gelu_tanh_and_mul`（Vector 冒烟） | 11 passed | 11 passed |
| `test_conv1d_bias` | 21 failed / 7 passed | 21 failed / 7 passed |
| `test_multi_head_attention` | 4 failed / 2 passed | 4 failed / 2 passed |
| `test_mamba`（ssd_chunk_scan） | 4 failed | 4 failed |

⇒ **tilelang 底座版本无关；瓶颈在编译器侧。**

### 9.3 精确根因（定位到源码行）
1. **CANN 9.2.0 的 `bishengir-compile` 崩溃点**：pass `ConvertHFusionToHIVM`，
   `bishengir/lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp`（v1.2.0 分支 1349-1351 行）：
   ```cpp
   moduleOp->walk([&](hivm::MmadL1Op op) {
     utils::getAnnotateOpWithAttr(op.getResult(0), hivm::TileMixCubeNumAttr::name);
   ```
   而 tilelang 发出的是 **buffer 形态** IR：`hivm.hir.mmadL1 ins(...) outs(%m : memref<...>)` → **0 个 result**，
   `getResult(0)` 越界：CANN 二进制（含断言）→ `SIGABRT`（`resultNumber < getNumResults()`）；自建 Release 版 → 野指针 `SEGV`。
   - 实证：把该 op 从 IR 删掉 → pass 通过；只留该 op → 崩。**只影响 Cube 算子**（Vector 算子无 mmadL1，故正常）。
2. **上游修复状态**（用 gitcode 远端核对）：
   - `v1.2.0`（2026-06-30）**有**该 walk（即 bug）；
   - `feature/0725`（2026-07-25）**已删除**该 walk；Oct 树 10-12（主线）也无。
   - ⇒ CANN 9.2.0 的编译器版本号是 **`1.2.0`（发布分支）**，主线已修但**未回合到发布分支**，故带着该 bug。
3. **Oct 自建编译器可跑完全部 MLIR**（我们带符号栈实测）：Oct 的 `bishengir-compile` 顺利产出 `module.hivm.opt.mlir`，
   但随后委托的 **hivmc 0.3.0 无法 legalize `hivm.hir.nd2nz`**
   （`failed to legalize operation 'hivm.hir.nd2nz' that was explicitly marked illegal`；去掉 `{dst_continuous}`、增删参数均无效）。
   即 §7.2.2 同一个坎：**Oct 代 IR 与现役 hivmc 不同代**。
4. **参考环境基线**：`kit_extract/RESTORE_CANN9_ADAPTED.md:50` 记录 **CANN 8.5.2 下 `tests/ops` 405/405 全通过**，且用的是**同一棵 Oct tilelang 树**。
   ⇒ 能跑通的组合 = **Oct/主线代 tilelang + 与之同代的 hivmc**（8.5.2 内那份），该 hivmc **不在本容器**。
5. **试过且不成立的 IR 级绕法**：给 mmadL1 补 tensor result 让 `getResult(0)` 合法 →
   DPS 校验拒绝：`expected the number of tensor results (1) to be equal to the number of output tensors (0)`。
   ⇒ 该编译器要求 **tensor 形态** IR，而 tilelang 发 **buffer 形态**，根上不兼容。

### 9.4 可行路径
- **A（最稳）**：取得与 Oct 树同代的 hivmc/编译器对（原 CANN 8.5.2，或 ≥2026-10 的 CANN）。8.5.2 已验证 405/405。需外部提供安装包。
- **B（自建，可自控，推荐）**：按 **`v1.2.0` 发布分支**（与 CANN 9.2.0 自带编译器同代 ⇒ 与 hivmc 0.3.0 同代）自建 AscendNPU-IR，
  **补上主线已有的修复**（把 `op.getResult(0)` 的 walk 加 `numResults==0` 保护，或按 `feature/0725` 删掉该 walk），再用它重编 tilelang。
  预期：既绕过 mmadL1 崩溃，又让产出的 HIVM IR 与 hivmc 0.3.0 同代（`nd2nz` 可 lower）。
- **C**：向 CANN 侧报缺陷（`mmadL1.getResult(0)` 崩溃；`hivm.hir.nd2nz` 无法 lower），等修版工具链。

### 9.5 本次新增的工具/资产
- `switch_stack.sh {oct|apr}`：双底座切换（两份 `build/` 均保留，仅改名）。
- `verify_stack.sh <label>`：通用验证（日志 `/tmp/opencode/verify-<label>/`）。
- `.bashrc.apr-active` / `.bashrc.oct-orig`：两套环境变量快照。
- `/tmp/opencode/diag/`：崩溃复现与最小化素材（`kernel1.npuir`、`v_aic.mlir`、`keep_shim.mlir`、`bin/hivmc` wrapper）。

## 10. 2026-10-16 晚：定格 CANN 9.0.0 + 工作流①「知识库结论重验」

环境最终定格为 **CANN 9.0.0**（`bishengir-compile 1.1.0` + `hivmc 0.2.0`，April 同代对）× **oct 代 tilelang**（自建 AscendNPU-IR `31f6903`）——即文档记载的 398/405 配置（`cann` 软链指向 `cann-9.0.0`，`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0`，910B3 / 可见设备 6 / driver 25.5.0）。分支 `cann9.0` 承载本次重定基线的记录覆盖。

### 10.1 工作流①：KB 结论重验（覆盖旧 8.5.x 戳）

`kb_stale_check` 报 stale 137 条，但其中绝大多数是**版本戳失配的规则/案例条目**（其 `stale_count` 以 repo HEAD 为工具链戳，存在已知假阳性，见 `queue.md` VP-2026-0116）。真正**可上机复现**的为 19 个探针：

| 批次 | 集合 | 结果 |
|------|------|------|
| A | pattern-library repro 套件（`repro_runner --filter stale`，命中 9 条） | **9/9 PASS** |
| B | 算子本地 repro（`examples/*/repro/*.py`，11 次执行） | **11/11 EXIT=0** |

结论：**19/19 探针在 CANN 9.0.0 下全部存活**，无一条被推翻。回写内容：
- `pattern-library`：9 条（`CONST-copy-floor-method`、`PL-1.10`、`PL-1.13`、`TRAP-DEVELMODE-PERSIST-GEMM`、`TRAP-T-copy-region-semantics`、`TRAP-load-nd2nz-strided`、`TRAP-tvm-parser-rules`、`TRAP-vrsqrt-plain-precision`、`TRAP-zero-input-crash`）+ `CONST-vcmp-ne-nan-unordered`（维持 overturned）追加 9.0.0 重验戳（保留 8.5.x 溯源，追加式）。
- `capability-gaps.md`：CG-2026-0018/0019/0021/0023/0024 的 `toolchain_stamp` 追加 9.0.0 重验（复现仍成立，缺口未消失，`status: open` 维持）。

### 10.2 方法学结论（重要，已机械化）

repro 用 `os.environ.setdefault("TILELANG_ASCEND_MODE", ...)` 声明目标模式，但 `.bashrc` **导出** `TILELANG_ASCEND_MODE=Developer` 会使 `setdefault` 失效——Expert 模式的 MixCV/Cube repro 被强制跑在 Developer 模式，产生**假失败**（bishengir 报 `multiple entry functions detected (previous: 'main_mix_aiv', this: 'main_mix_aic')`）。首轮 A 组的 4 个 FAIL 即此因，清干净环境后全部转 PASS。

修复：`repro_runner.py` 已在子进程 `env` 中剥离 `TILELANG_ASCEND_MODE`（repro 自决模式）；`repro/README.md` 增补「环境卫生（强制）」小节。手工直跑用 `env -u TILELANG_ASCEND_MODE python3 repro/<ID>.py`。

### 10.3 待办（②/③/④ 未做）

- ② 19 算子正确性全量 `pytest tests/ops`（目标 ~398/405；已知残差 `conv1d_bias` fp16、`mha_fwd`、`ssd_chunk_scan_fwd`）。
- ③ 性能比全部失效，需 `tileops-report run --all` 重测（含 `NPUUtils.get().get_aicore_num()` 实查）。
- ④ `MIGRATION_INDEX.md §C` / `versions.txt` 版本戳收尾；复核「删 NaN 恢复」优化在 9.0 下是否仍有效。

