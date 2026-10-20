# CANN 9.0 适配记录

> 本文是**过程记录与根因归档**。交付口径（19 接口的正确性/性能数字）以
> `examples/TileOPs/reports/FINAL_DELIVERY_20261018.md` 与 `perf_data/canonical.json` 为准；
> 本文的数值是定位过程中测到的，**不作为交付数据**。
> 已废弃的尝试（自建 Apr 栈、CANN 9.2 栈、自编 `build-v11`）只保留结论，细节见 git 历史。

## 1. 背景

环境升级到 **CANN 9.0.0**（`bishengir-compile 1.1.0` + `hivmc 0.2.0`）；当时 tilelang dev-root
（`/home/developer/tilelang-mlir-ascend`，`417dae9`）是针对 CANN 8.5.2 编译的，9.0 下 MixCV/Expert 编译报错。

## 2. 根因（已定位）

- **纯 Vector** kernel 在 9.0 下编译通过；**MixCV/Expert**（含 `sync_block_set/wait` flag 握手）报
  `error: Dialect '' not found for custom op 'syn_instr_mode'`。
- `syn_instr_mode` 由旧 codegen 附在 sync op 上；**CANN 9.0 的 bishengir 已不认识它** → 版本不匹配。

## 3. 绕法（小改动，无需重建）

在 tilelang 调 bishengir 前把关键字改成 9.0 的拼写（`tilelang/jit/jit_npu.py`
`_npuir_to_bin_enable_npu_compile()`，文件在本仓外，属 toolchain 本地补丁）：

```python
linalg = _re.sub(r"\bsyn_instr_mode\b", "sync_instr_mode", self.mlir_content)   # CANN9 shim
```

早期试过「删除属性」，会让同步模式**静默退回默认值**；**改名**保留真实模式，语义更正确
（本批 kernel 模式恰为默认值，两者结果等价）。**这是权宜之计**，不是修复——见 §7。

## 4. 适配结果（CANN 9.0 + shim）

`pytest tests/ops`：**398 passed / 7 failed**（8.5.2 下 405 passed）。7 个失败为
`test_conv1d_bias`×5 + `test_mamba::ssd_chunk_scan` + `test_multi_head_attention::mha_fwd`，
形态均为 9.0 `bishengir-compile` **SIGSEGV**。

## 5. 结论 / 待办

- `syn_instr_mode` 类问题：小 shim 可解，**不需重建**。
- 剩余 7 个 SEGV：需匹配 9.0 的 AscendNPU-IR，或改 kernel sync 构造规避 → 见 §7/§9.3/§15。
- 性能数字**必须绑版本**（9.0 需单独复测）；「删 NaN 恢复」优化需在 9.0 重验 → 见 §12.5。

## 6. 对剩余 7 个 SEGV 的进一步排查（2026-10-16）

尝试用改名解决（`syn_instr_mode` / `sync_instr_mode` / `tsync_instr_mode` 三个拼写）**全部解析失败**
⇒ 9.0 的 `hivm.hir.sync_block_set/wait` op 定义**结构性改版**，字符串级改名/删除无效。
结论：需匹配 9.0 的 AscendNPU-IR，或改 kernel 构造。详细定位见 §7。

## 7. 受限问题的最终定位（2026-10-16 深挖）

### 7.1 版本真相（关键发现）

- 本地 tilelang 树内 `3rdparty/AscendNPU-IR` = 上游 `31f6903`（2026-10-12）。
- **CANN 9.0.0 自带的 bishengir-compile / hivmc = 上游 `428ab8fdab46` = tag `v1.1.0-post2`（2026-04-30）**。
- ⇒ 差异**不是「新旧 CANN」，而是 6 个月的 AscendNPU-IR 版本差**：树是 10 月版，容器里是 4 月版。

### 7.2 具体不兼容点

1. **sync 属性关键字改名**：4 月版 = `sync_instr_mode`，10 月版 = `syn_instr_mode`（枚举/默认值相同）。
   tilelang Expert codegen（`src/target/codegen_npuir.cc:1053`）硬编码输出后者 → 改名后解析通过。
2. **编译器分层错配**：10 月版 `bishengir-compile` 把 HIVM→LLVM 委托给外部 `hivmc`（`runExternalHIVMC`），
   并转发 `--enable-lir-compile` 等**旧 hivmc 不认识的参数**；过滤参数后可继续，但卡在
   **`hivm.hir.nd2nz` 合法化**——旧 hivmc 无法 lower 新管线的 IR。
3. **`hivmc` 是闭源外部组件**：10 月版源码里只有库、没有可执行目标 ⇒ **无法自行构建匹配 10 月树的 hivmc**。

### 7.3 容器内实测结论

- 用 CANN 9.0 的 `bishengir-compile` 编译（改名后、属性完全合规的）MixCV npuir → **确定性 SIGSEGV**
  （参数矩阵、`ulimit -s`、`--tb` 均复现）⇒ **4 月版编译器自身在该类 kernel 上的缺陷**。
- 换 10 月版编译器 → 不再 SEGV，但卡在旧 hivmc 的 `nd2nz` 合法化。
- ⇒ **两侧各缺一半：4 月编译器崩、10 月编译器缺匹配 hivmc。**

### 7.4 结论 / 可行路径

- 重新 pin 到 `v1.1.0-post2` 重建**不能修复**——该版本编译器正是崩溃的那个（§9.2 已实测验证）。
- 真正修复需拿到**与 10 月树匹配的 hivmc**：安装原 CANN 8.5.2 内部版（离线不可得）、或等 CANN 侧更新、
  或改 kernel 的 sync 构造规避。
- 现状：Vector/元素/规约/归一化族**完全可用**；Cube 族受限（后续在 §15 有部分解锁记录）。

## 8. 落地方案：自建 Apr 版 AscendNPU-IR + 重编 tilelang（**已废弃**）

思路：CANN 9.0 只发编译器二进制、不发开发件，故把 AscendNPU-IR 与 tilelang 一起对齐到 `v1.1.0-post2`，
在旁路目录 `3rdparty/AscendNPU-IR-apr` 自建后重编。构建要点：cmake ≥3.28、pybind11 **2.12.0**
（3.1.0 触发 `static_assert`）、需补 `collapseAndReplace<CollapseShapeOp>` 显式实例化。
**结果：建成但对结果无影响**（§9.2 逐项一致）⇒ 路线已放弃，脚本/副本不再维护。

## 9. 2026-10-16 晚：环境被换成 CANN 9.2.0 后的复核与最终根因

### 9.1 环境变更

诊断期是 CANN 9.0.0；工作期间容器被替换为 **CANN 9.2.0**
（`bishengir-compile 1.2.0`、**`hivmc 0.3.0` 存在**），§7.2.3「hivmc 不可得」的前提随之变化。

### 9.2 §8「向 Apr 对齐」已建成，但对结果无影响

在 CANN 9.2.0 上与 Oct 原树对比，**逐项一致**：

| 用例 | Oct 底座 | Apr 底座 |
|---|---|---|
| `test_gelu_tanh_and_mul`（Vector 冒烟） | 11 passed | 11 passed |
| `test_conv1d_bias` | 21 failed / 7 passed | 21 failed / 7 passed |
| `test_multi_head_attention` | 4 failed / 2 passed | 4 failed / 2 passed |
| `test_mamba`（ssd_chunk_scan） | 4 failed | 4 failed |

⇒ **tilelang 底座版本无关；瓶颈在编译器侧。**

### 9.3 精确根因（CG-2026-0003）

`bishengir/lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp`（v1.2.0 分支 1349–1351 行）：

```cpp
moduleOp->walk([&](hivm::MmadL1Op op) {
  utils::getAnnotateOpWithAttr(op.getResult(0), hivm::TileMixCubeNumAttr::name);
```

tilelang 发出的是 **buffer 形态** IR（`hivm.hir.mmadL1 ins(...) outs(%m : memref<...>)`）→ **0 个 result**，
`getResult(0)` 越界：CANN 二进制（带断言）→ `SIGABRT`；自建 Release → `SEGV`。
实证：删掉该 op → pass 通过；只留该 op → 崩。**只影响 Cube 算子**（Vector 无 mmadL1）。

上游状态：`v1.2.0`（2026-06-30）有该 walk（即 bug）；`feature/0725` 与 10 月主线**已删除**
⇒ CANN 发布分支未回合修复。本仓的处置是**工具链绑定 xfail**（§11.5）。

### 9.4 可行路径

- **A**：取得与 Oct 树同代的 hivmc/编译器对（原 CANN 8.5.2 或 ≥2026-10），需外部提供安装包。
- **B**：自建 v1.2.0 分支并补 `numResults==0` 守卫，与 hivmc 0.3.0 同代（已在 §14/§15 试过，见结论）。
- **C**：向 CANN 侧报缺陷，等修版工具链。

### 9.5 本次新增的工具/资产

`switch_stack.sh {oct|apr}`（双底座切换）、`verify_stack.sh`、`.bashrc.{apr-active,oct-orig}`、
`/tmp/opencode/diag/`（最小复现素材）。**均属本地临时资产，不入库。**

## 10. 2026-10-16 晚：定格 CANN 9.0.0 + 工作流①「知识库结论重验」

环境定格为 **CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）× oct 代 tilelang（自建 AscendNPU-IR `31f6903`）。
分支 `cann9.0` 承载重定基线记录。

### 10.1 工作流①：KB 结论重验（覆盖旧版本戳）

`kb_stale_check` 报 stale 137 条，多为版本戳失配（已知假阳性，见 `queue.md` VP-2026-0116）；
真正**可上机复现**的 19 个探针：

| 批次 | 集合 | 结果 |
|---|---|---|
| A | pattern-library repro 套件（`repro_runner --filter stale`，9 条） | **9/9 PASS** |
| B | 算子本地 repro（`examples/*/repro/*.py`，11 次执行） | **11/11 EXIT=0** |

结论：**19/19 在 CANN 9.0.0 下全部存活**，无一条被推翻；已按「追加式」给 9 条 pattern-library 条目与
5 条 CG（0018/0019/0021/0023/0024）补 9.0.0 重验戳（保留旧戳溯源）。

### 10.2 方法学结论（已机械化）

repro 用 `os.environ.setdefault("TILELANG_ASCEND_MODE", ...)` 声明模式，但 `.bashrc` **导出**同名变量会让
`setdefault` 失效 → Expert 的 MixCV/Cube repro 被强制跑在 Developer 模式，产生**假失败**。
修复：`repro_runner.py` 在子进程 env 中剥离该变量；手工直跑用
`env -u TILELANG_ASCEND_MODE python3 repro/<ID>.py`。

### 10.3 待办（均已完成）

② 19 算子正确性全量 → §11；③ 性能口径 → §12–§13（用细粒度快扫替代全量 `run --all`）；
④ 版本戳收尾 + T6「删 NaN 恢复」复核 → §12.5 的「T6 关闭」注记。

## 11. 2026-10-17：工作流② 正确性重跑（表 A 全量）

跑法：`pytest tests/ops/<file>.py` **逐文件串行**（独立进程 + 独立 summary，抗中断）；
环境 = CANN 9.0.0（§10 定格配置）。

### 11.1 表 A（19 交付算子）正确性结果

| 算子 | 测试文件 | 结果 |
|---|---|---|
| `HardsigmoidFwdOp` | `test_hardsigmoid` | ✅ 7/7 |
| `ClampScalarFwdOp` | `test_clamp_scalar` | ✅ 11/11 |
| `HardswishFwdOp` | `test_hardswish` | ✅ 7/7 |
| `AddFwdOp` | `test_add` | ✅ 24/24 |
| `SubFwdOp` | `test_sub` | ✅ 24/24 |
| `GeluAndMulFwdOp` | `test_gelu_and_mul` | ✅ 11/11 |
| `GeluTanhAndMulFwdOp` | `test_gelu_tanh_and_mul` | ✅ 11/11 |
| `IsnanFwdOp` | `test_isnan` | ✅ 9/9 |
| `IsinfFwdOp` | `test_isinf` | ✅ 9/9 |
| `IsfiniteFwdOp` | `test_isfinite` | ✅ 9/9 |
| `FusedAddLayerNormFwdOp` | `test_fused_add_layer_norm` | ✅ 14/14 |
| `BatchNormBwdOp` | `test_batch_norm_bwd` | ✅ 12/12 |
| `ClampMinFwdOp` | `test_clamp_min` | ✅ 14/14 |
| `LerpFwdOp` | `test_lerp_scalar` | ✅ 36/36 |
| `Conv3dBiasFwdOp` | `test_conv3d_bias` | ✅ 17/17 |
| `GLADecodeOp` | `test_gla_decode` | ✅ 11/11 |
| **`Conv1dFwdOp`** | `test_conv1d` | ❌ **1 failed** / 27 passed |
| **`Conv1dBiasFwdOp`** | `test_conv1d_bias` | ❌ **5 failed** / 23 passed |
| `GatedDeltaNetPrefillFwdOp` | `test_gdn`（当时整文件 skip） | — 未交付（CG-2026-0026） |

**合计：276 / 282 passed（6 failed）。**

> **后续更新（2026-10-19）**：本表是 2026-10-17 的存档。其后 GDN in-scope 主路径交付、
> `test_gdn.py` **10/10**；卷积族那 6 个失败在后续运行中**未复现**（Conv1d / Conv1dBias 均为
> 27 passed + 1 skipped、0 failed）。最新交付口径见 `FINAL_DELIVERY_20261018.md`。

失败的 6 个全在卷积族、形态全为 SIGSEGV（`bishengir-compile` LLVM crash backtrace；nodeid 清单见
`tests/conftest.py` 的 `_CG_0003_TESTS`）。根因 §9.3；**是工具链缺陷，不是算子缺陷**（同树在 8.5.2 下 405/405）。
注：并非「只有 fp16」，bf16 用例同样失败。

### 11.2 表 B 退出重定基线范围（决策记录）

**决策（2026-10-17）**：表 B 的 7 个算子（`AdaLayerNorm` / `Argmax` / `LerpTensor` / `LogSumExp` /
`Mish` / `MultiHeadAttention` / `SSDChunkScan`）**不属于本次迁移/优化目标**，后续阶段一律不再纳入，
除非另行下令。已观察到的事实仅存档：`SSDChunkScan` 第 2 例 `bishengir-compile` **挂死**
（`STAT=R`、CPU 99.8% 空转、`Time>19min`、`VmRSS` 仅 50 MB，非 SIGSEGV、非死锁）。

### 11.3 同一根因的第二种坏法（挂死）

§7.3/§9.3 记录的形态是确定性 SIGSEGV；本次新增 **CPU 空转挂死**。两者同源
（Cube 形态 IR × 4 月版编译器）——该编译器在这类 IR 上**要么崩、要么转不出来**。
后续遇到「编译不返回」应据此判定，勿误判为环境/死锁。

### 11.4 方法学：逐文件串行

全量 `pytest tests/ops` 单进程约 40+ min，**中断即丢全部结果**（首轮因此丢在 84%）。
改逐文件独立进程后：中断只丢当前文件、可逐文件对账、挂死文件可被 timeout 隔离。同类重跑沿用此法。

### 11.5 处置决策：工具链绑定 xfail（方案 A）

对 §11.1 的 6 个失败**不改算子、不改 kernel**，在 `examples/TileOPs/tests/conftest.py` 加
**工具链绑定 xfail**（`pytest_collection_modifyitems`）：

- 只在 `_cann_version() == "9.0.0"` 时生效 → 换工具链自动失效；
- `strict=False` → 编译器修好后变 **XPASS**（提醒摘标），不判红；
- 理由串带 CG 号与证据路径；**不是无条件 skip**，缺陷保持可见。

已验证：命中**恰好 6 项**（`--collect-only -m xfail` → 6/56）；摘掉 `ASCEND_TOOLKIT_HOME` 模拟其它工具链 → 0 项。
未采用：B（保持红）、C（自建修复版编译器）、D（报 issue，待批准）。

## 12. 2026-10-17：性能口径改为「细粒度快扫」

### 12.1 决策与理由

全量 T4（18 算子 × `tileops-report run`，约 4.5 h）与 optimize 阶段的第一步（采 baseline）**完全同源**
（`signal-registry.md` §5：`msprof op Task Duration(us)` 是唯一时延口径）⇒ 互为重复劳动。
且未被 optimize 的算子基线随时可补（基线是「后置」而非「丢失」）。
故放弃全量 T4，只做一次**粗粒度回归体检**，回答「9.0 相对 8.5.2 有没有把哪个算子打坏」这个唯一无法后补的问题。

### 12.2 快扫设计

- 范围：表 A 18 算子；每个取**元素量最小 + 最大**两个非 smoke workload、各带全部 dtype → **77 case**（31%）。
- 方法：**只跑 benchmark 阶段**（跳过 correctness），复用 runner 的环境变量（`TILEOPS_PROF_MODE=msprof`、
  `*_MSPROF_KEEP_OUTPUT=1`、`*_ARTIFACT_ROOT`、`*_BENCHMARK_REPORT_PATH`），用 **exact `<label>-<dtype>`
  token** 选 case（避开 `-k` 子串陷阱），逐算子独立 `timeout`。
- 产出 `bench_<Op>.json`（与 `run.json` 同 schema）。
- 口径自校验：`Hardsigmoid/mbv3-se-gate/fp16` 快扫 2.380 µs vs T4 全量 2.420 µs（噪声 1.7%）→ 可比。

### 12.3 回归判据（8.5.2 ↔ 9.0）

同芯片（环境戳均 `Ascend910B3 x 1`），按 `(op, label, dtype, shape)` join；告警阈值
**|Δlatency| > 15% 或 Δratio < −8 pt**。

### 12.4 顺带确认的事实：多个算子生效的是 perf_opt 变体

perf_opt 通过 loader 顶部「注释切换」生效；快扫/T4 测到的就是**当前生效内核**在 9.0 下的实测值。
（哪个算子在 perf_opt 上，以 `perf_data/delivery.json` 的 `effective_kernel` 为准。）

### 12.5 T6 复核点（已关闭）

`GeluAndMul` 的 Stage-4 优化 = **删除冗余 NaN 恢复**，依据「erf 多项式链 + `vclamp` 已传播 NaN」
——这是编译器行为假设，且**原测试不含 nan/inf 用例**，若假设失效会**静默算错**。

> **T6 关闭（2026-10-19）**
> - **增益仍在**：`opt_log.md §1` 的 9.0 同源 A/B：baseline 877.198 → `nan_drop` 800.076 µs（−8.8%，bf16 −8.2%）。
> - **假设成立**：`perf_opt/...py --level all` → `[Boundary] PASS: inf/nan`（fp16/bf16/fp32）+ 其余边界全 PASS。
> - **永久守卫**：新增 `test_gelu_and_mul.py::test_gelu_and_mul_special_values`（走生产 wrapper）；
>   并做反向对照（人为把 `gelu(-inf)` 改成 `-inf`）证明守卫**会失败**。
> - 结论已回写 `perf_opt/opt_log.md`。

### 12.6 事故记录：残留进程污染 + 重跑（2026-10-17）

`pkill -f run_perf_all.sh` **自匹配**了执行它的 shell（命令串含该字面量）→ shell 先收到 SIGTERM，
后续清理未执行；正在跑的 `--op HardswishFwdOp` 进程树被孤儿化（`ppid→1`）并**继续占用 NPU ~15 min**，
与快扫并发。处置：按 PID 杀掉孤儿树、`npu-smi info` 复核、作废污染窗口结果、10:12 干净重启。

**教训（已写入流程）**：① `pkill -f` 模式必须自匹配规避（如 `run_perf_[a]ll.sh`）；
② 杀父进程不终止已 fork 的子进程，须按 PID 逐层清理；③ 测量前必须确认 NPU 无其它进程。

## 13. 2026-10-17：18 算子「迁移态」偏差复核（9.0 ↔ 8.5.2）

### 13.1 方法纠正：先测迁移态，不是「当前生效态」

初版快扫测的是「当前生效态」；对 5 个算子（Hardsigmoid/ClampScalar/ClampMin/GeluAndMul/GeluTanhAndMul）
生效态是 **perf_opt**——那不是本阶段要验的迁移交付物。已纠正：4 个 loader 切回迁移态 →
补测迁移态正确性 54/54 → 全量 18 算子/77 用例迁移态重扫 → `git checkout` 还原、`git status` 确认干净。

### 13.2 8.5 基线口径：**「层」比「时间」重要**

8.5 报告本体不含「层」字段，两种「常识挑法」都会取错：

| 挑法 | 结果 | 结论 |
|---|---|---|
| 按 `git_commit` 分 | ❌ | 迁移态与优化态 run 的 `git_commit` **相同**（优化先在工作区、后提交） |
| 按时间戳分 | ❌ | 同理：优化态数字可能早于优化提交 |
| 按「最新 run」 | ❌ | 会把优化态当迁移态（Hardsigmoid 会得出 −42% 假偏差，真值是 −1.4%） |

**权威分层来源**：① `<kernel>/integration_report.json` 的 `run_json`（迁移态指针，14/18 有）；
② `MIGRATION_INDEX.md §A` 引用的 run（补齐后 18/19 明确）；③ `perf_opt/opt_log.md` / `perf_records.jsonl`（优化态）；
④ 兜底：数值签名反查，最后才回退「最新 run」。对比器默认 `--state=migrated` 即按此优先级取基线。

### 13.3 结果概览（18/18）

18/18 完成（EXIT=0 ×16，EXIT=1 ×2：Conv1d / Conv1dBias 的 fp16 命中 CG-2026-0003 崩溃）；
有效用例 **70**（剔除 2 崩溃 + 1 基线缺失）。

| \|Δlat\| | ≤2% | 2–5% | 5–10% | 10–15% | >15% |
|---|---|---|---|---|---|
| 用例数 | **39**（56%） | 15 | 9 | 5 | 2 |

**中位 1.64%，P90 10.13%**。

### 13.4 噪声底（判据标尺）

| 用例规模 | 同态重复 Δ% |
|---|---|
| 256M 档 | **±1.1 – 1.3%** |
| 16M 档 | **±2.0 – 6.5%**（中位 1.3%） |
| µs 级（3–15 µs） | ±7 – 15% |

### 13.5 >5% 用例三分类

**① 无效**：`Conv1d fp16` +199.8%、`Conv1dBias fp16` +238.7%（均为崩溃产物）；`ClampScalar 16M/fp32` +34.0%
——其中 **+22.3% 是 `block_size` 8192→4096 的配置差异**，工具链部分仅 **+9.6%**。

**② 已用复测定性**：`BatchNormBwd large-spatial/fp16` 首测 2567.61 / 复测 2571.39（+0.1%）⇒ **+22.7% 是真偏差**；
`resnet-stage1 fp16` 收窄至 +5.3%、`bf16` +3.9%。

**③ clamp 家族 fp32 确认 +7~8%**（`ClampMin` 真迁移态基线 64.0 µs，此前误用 49.70 的优化态）：

| 用例 | 8.5 迁移态 | 9.0 首测 | 9.0 复测 | 判定 |
|---|---:|---:|---:|---|
| ClampScalar 16M/fp32 | 68.48 | 75.02 | 72.96 | **+8.0%**（均值 74.0） |
| ClampMin 16M/fp32 | 64.0 | 68.40 | 69.26 | **+7.5%** |

→ 重复性良好（±1.3~2.7%）。但 `Isnan/Isinf/Isfinite` 的 fp32 为 −6.2%/+3.6%/−2.2%
⇒ **不是全面 fp32 回退**，是 clamp 家族特有（fp32 计算链 `vmax`/`vmin`/`vclamp` 与 9.0 调度交互）。

> **2026-10-20 补测**：ClampScalar / ClampMin 的迁移态已做全量补测（非折算），
> 结果见 `perf_data/canonical.json` 与 `reports/tileops/*_migrated/PROVENANCE.md`；
> 本节 74.0 / 68.83 µs 是当时的单点，已不作为交付数据。

### 13.6 阶段性结论

1. **迁移态在 9.0 上主体无偏差**：56% 用例 ≤2%，中位 1.64%（`Hardswish` 全档 −1.5%~+1.1%，可作无偏差锚点）。
2. **一处真异常**：`BatchNormBwd large-spatial/fp16` **+22.7%**（已复现、代码冻结）。
3. **一处真回退**：clamp 家族 fp32 **+7~8%**（复测复现，仅此家族）。
4. **8.5 报告可用性**：正确性可照用（conv 6 例受 CG-2026-0003 影响，已版本绑定 xfail）；
   大 shape + fp16/bf16 性能可照用（±2%）；clamp fp32 与 `BatchNormBwd` 需修正；µs 级小用例只作量级参考。
5. **用 8.5 报告前必须先定「层」**（§13.2），否则会得到 ±40% 的假偏差。

## 14. 2026-10-18：v1.2.0 补丁编译器 + CANN 9.2.0 hivmc 0.3.0（**已废弃**）

`v1.2.0` + `mmadL1` 守卫 + `hivmc 0.3.0`：conv1d/conv1d_bias/conv3d 的 SIGSEGV 消失
（23 passed / 1 xfailed / 4 xpassed），MHA/SSD/GDN blocksolve 仍因编译器与 hivmc 不同代失败。
**9.2 栈后来被物理删除**（释放 9.6G），默认栈切回 9.0；本路线不再维护，细节见 git 历史。

## 15. 2026-10-19：v1.1 补丁编译器 + CANN 9.0.0 hivmc 0.2.0（9.0 路线复验）

### 15.1 做了什么 / 实测

在 `3rdparty/AscendNPU-IR-v11` 检出 `release/v1.1.x @ 428ab8fd`，补两处（`mmadL1 getResult(0)` 守卫；
`collapseAndReplace<CollapseShapeOp>` 显式实例化）→ `bishengir-compile 1.1.0 (428ab8fd)` + 9.0 自带
`hivmc 0.2.0`（后者不认 `--enable-lir-compile` 等参数，需参数过滤 wrapper）。实测：

| 用例 | 结果 |
|---|---|
| `test_conv1d_bias` | **23 passed / 1 xfailed / 4 xpassed**（SIGSEGV 消失） |
| `test_conv1d` + `test_conv3d_bias` | 通过 |
| `test_multi_head_attention` | 4 failed / 2 passed |
| `test_mamba -k ssd` | 1 failed / 3 passed |
| GDN blocksolve L0 | `hivmc: Failed to compile BiShengLIR to binary` |

### 15.3 结论：9.0 路线只能覆盖 conv

- **19 批的 MixCV（conv1d / conv1d_bias / conv3d）在 9.0 原生栈上已解锁**，不必依赖 9.2。
- 复杂 kernel 必然 `Failed to compile BiShengLIR to binary`：`hivmc 0.2.0` 不支持 `--enable-lir-compile`，
  且闭源、无法本地重建；9.2 的 0.3.0 虽支持，但与 v1.2.0 仍无法 lower GDN blocksolve。
- 试过把守卫从「删除 walk」改为 buffer-form 取 `op.getC()` 并保留 `tile_mix_cube_num` 传播 → GDN 仍失败。
- **9.2 已按用户要求物理删除**；当前唯一默认栈为 9.0/v1.1。

### 15.4 环境层面还剩什么

1. **9.0 默认可用**：系统 `bishengir-compile` + `hivmc 0.2.0` 在**空 cache** 下可编译最小 Cube GEMM、K4、
   纯 Vector K1、conv1d_bias 与 conv3d。18 算子复测用的就是这套系统编译器。
2. **自编 `build-v11` 是回归源**：能过 HIR，但产出的 HIVM IR 与 `hivmc 0.2.0` 不兼容
   ⇒ 「9.0 不能编译任何 kernel」的结论**只对 build-v11 成立**，对系统编译器不成立。
3. **K2 仍被 CG-2026-0003 阻塞**：系统编译器在 GDN blocksolve 上 SIGSEGV；K2 三段拆分是出路
   → 最终在 10-19 打通（见 §15.6 与 GDN 的 `OPTIMIZE_REPORT`）。
4. **不属于环境**：GDN functions 4–6 端口、MHA/SSD kernel 重构、换工具链后的性能重基线。

### 15.5 复现命令

```bash
source /home/developer/tilelang-mlir-ascend/env_cann90.sh   # 系统 CANN 9.0 编译器 + hivmc 0.2.0
cd examples/TileOPs
python3 -m pytest tests/ops/test_conv1d_bias.py tests/ops/test_conv1d.py \
    tests/ops/test_conv3d_bias.py tests/ops/test_gdn.py -q
```

### 15.6 2026-10-19 K4/K2 编译探针（修正版）

探针目录 `examples/gdn_prefill/correct_initial_states/`（`probe_cube_min.py` / `probe_compile.py` /
`PROBE_20261019.md`）。

- **系统编译器 + 空 cache**：最小 GEMM / K4 / K1 均 `COMPILE_OK`；conv1d_bias 23 passed / 4 xfailed / 1 xpassed；
  conv3d `video-down` 1 passed（50s，真编译）。
- **自编 `build-v11` + 空 cache**：同一批全部 `Failed to compile BiShengLIR to binary`（此前的通过是缓存命中）。
- **影响**：K4 可进 Stage 3；K2 blocksolve 仍 SIGSEGV → 用三段拆分规避，最终打通（K1+K2 split+K6，
  `test_gdn.py` 10/10）。
