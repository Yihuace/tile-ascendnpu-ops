# GDN prefill → Expert 模式：已验证的构建块与设计（WIP）

> 目标算子 `GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`，`kernels/linear_attention/gdn/prefill.py:29`）。
> 工具链：tilelang `0.1.2+417dae99` / **CANN 8.5.2** / Ascend910B3。
> 状态：**WIP**。原 Developer 模式探针 `gdn_prefill_probe.py` 编译失败（gap CG-2026-0026）；本文件记录已实测排障结论。

## 1. 失败根因（已定位）
- 原探针用 **Developer 模式**（`T.alloc_shared`）同时承载 Cube（`T.gemm`）与 Vector，交 bh 后端崩溃：
  `fatal error: error in backend: ERROR: error pointer address space cast`（坏 IR 在 **AIC** 路径 `kernel.mix_aic.ll`）。
- 二分：删掉 Neumann 循环仍崩 → 与 Neumann 无关；崩溃来自 **Developer 模式下生成 Cube(AIC) 的代码**。
- 结论：必须改 **Expert 模式**（`T.Scope("Cube")` + `T.alloc_L1/L0C` + `T.gemm`；`T.Scope("Vector")` + `T.alloc_ub`）。

## 2. 已验证的构建块（实测）
1. **Expert `T.gemm` 可用**：`alloc_L1`/`alloc_L0C` + `T.gemm(..., b_transpose=True)`，KK/KS 对拍误差 ~1e-6。
2. **交错 `T.Scope("Cube")` → `T.Scope("Vector")` 能编译运行**（同一 `T.Kernel` 内顺序多段 scope）。
3. **Cube→Vector 经 GM 的交接需要显式 flag 握手**：
   - 无同步：Vector 读到旧数据（err 48.5 ✗）。
   - `Cube: T.sync_block_set(f)`（`with T.rs("PIPE_FIX")`）+ `Vector: T.sync_block_wait(f)`（`with T.rs("PIPE_MTE2")`）→ 正确（err 3.8e-6 ✓）。

## 3. Expert 版相位设计（每 chunk）
状态 `S[D,DV]` fp32 常驻；gemm 操作数用小写 fp16 视图。
| 相位 | 单元 | 内容 | GM 交接 |
|---|---|---|---|
| C1 | Cube | `KK=k·kᵀ`、`QK=q·kᵀ`、`KS=k·S16`、`QS=q·S16` | `kk/qk/ks/qs` → set flag |
| V1 | Vector | gate：`gc=cumsum(g)`、`exp(gc)`、`grev`/`gdb`、`G`(loose/strict)、`beta`；`S_sys=beta·G_strict·KK`→`sn16`；`rhs=beta·(v−exp(gc)·KS)`→`rhs` | `sn16/rhs/G` → set flag |
| C2+V2 | Cube↔Vector | Neumann `W=(I+S_sys)⁻¹rhs`（按 `pairs` 迭代；或纯 Cube 累加矩阵幂再 `W=M·rhs`，避免交错） | `w16` → set flag |
| C3 | Cube | `o2=(scale·G·QK)·W16`、`Supd=(k·grev)ᵀ·W16` | `o2/sup` → set flag |
| V3 | Vector | `o=scale·exp(gc)·QS+o2`；`S=gdb·S+Supd`（fp32） | 写 `o`；下一 chunk |

- 每交接点一组 `sync_block_set/wait`（flag 单调递增，取模复用）。
- Neumann 建议改为**纯 Cube**：先累加矩阵幂 `M=I+Sn+Sn²+…`（gemm 累加），再 `W=M·rhs`，以**减少 Cube/Vector 交错次数**。

## 4. 剩余工作（大）
- 逐相位写全 + 真机调通（`nch` chunk 循环、多 (b,h)）。
- fp16 路径（`_IN`），必要时 bf16。
- 再走 op 层（`GLADecodeOp` 同族）、manifest（`linear_attention.yaml` GLADecodeOp1 之外新增）、tests、bench、报告。
- 估：这是**多轮真机调试的内核**（GPU 源 ~2600 行），非一轮可收口。

## 5. 参考
- 验证最小件：`/tmp/opencode/{exp_gemm,sync_test}.py`（transient）。
- 算法/数学：`gdn_prefill_probe.py` 顶部注释 + 本文件 §3。

## 6. 分相位验证进度（真机对拍 torch）
| 相位 | 内容 | 状态 | 误差 |
|---|---|---|---|
| C1 | Cube：`KK=k·kᵀ`、`QK=q·kᵀ`、`KS=k·S16`、`QS=q·S16` | ✅ | ~1e-6 |
| V1(gate) | `gc=cumsum(g)`、`G_strict/G_incl`(arange+minmax 掩码)、`exp(gc)` | ✅ | gc 精确 / G 6e-8 |
| V1b | `S_sys=β·G_strict·KK`→`sn16`；`rhs=β·(v−exp(gc)·KS)` | ✅ | S_sys 1.2e-3(fp16 舍入) / rhs 2.4e-7 |
| C2 | Neumann `W=(I+S_sys)⁻¹rhs` | ⬜ 待做 | |
| C3/V3 | `o`、状态传递 | ⬜ 待做 | |

## 7. 关键 API 教训（踩坑）
- **向量 op 的 dst 必须是 UB**：`T.vmul(v, s, Gstrict)` 直接写 GM 张量 → 生成坏 IR → **bishengir SIGABRT**。必须写 UB，再 `T.copy(ub, GM)`。
- **`T.copy` 不做 dtype 转换**：fp16→fp32 需 `T.copy(fp16GM, fp16UB)` + `T.vcast(fp16UB, f32UB)`。
- **Cube→Vector 经 GM 交接必须显式 `sync_block_set/wait`**（`with T.rs(...)`），否则读旧数据。
- **非 out_idx 的工作区张量要显式传入**（kernel 报 "Expected N inputs"）。

## 8. 下一步
- C2（Neumann）：建议**纯 Cube** 实现——累加矩阵幂 `M=I+Σ Sn^j`（`gemm(P_j, I, M_l0c, initC=False)` 累加；`P_{j+1}=Sn@P_j`），再 `W=M@rhs`，避免 Cube/Vector 交错。
- C3/V3：`o=scale·exp(gc)·QS+(scale·G·QK)·W`；`S=gdb·S+(k·grev)ᵀ·W`。
- 再拼成完整 chunk 循环 → op 层/manifest/tests/报告。

## 9. 进度快照 / 恢复点（2026-10-16）
- **真机实验脚本**（可重跑）：`experiments/`
  - `01_expert_gemm.py` — Expert `T.gemm`（含 b_transpose）✅
  - `02_gm_handoff_sync.py` — Cube→Vector 经 GM + flag 握手 ✅
  - `03_c1_cube.py` — C1：KK/QK/KS/QS ✅ (~1e-6)
  - `04_v1_gate.py` — V1：gc / G_strict / G_incl ✅ (gc 精确, G 6e-8)
  - `05_v1b_ssys_rhs.py` — C1+V1b：S_sys→sn16 / rhs ✅ (1.2e-3 / 2.4e-7)
  - `06_c2_neumann_FAILED.py` — C2 Neumann 纯 Cube 幂累加 ❌（W err 50.5，环路中接 hazard）
- **恢复点**：修 `06` 的 L0C→GM→L1 环路同步（或改用 pairs 循环 + 已验证的 Cube/Vector 握手），通过后做 C3/V3，再拼整 kernel → op 层/manifest/tests/报告。
- **注意**：本任务未产出可用内核；`GatedDeltaNetPrefillFwdOp` 仍**未交付**（gap CG-2026-0026 更新为"进行中/WIP"）。

## 10. 追加工具链限制（2026-10-16，Expert 重写中实测）
在写 Neumann（`W=(I+S_sys)^{-1}rhs`）时又撞到两条 **Cube 侧工具链限制**，均以最小实验证实：
1. **`T.gemm(..., initC=False)` 不跨多次 gemm / 循环累加**：
   - 实验：`M=I` 后连续 3 次 `gemm(Sn,I,l0,initC=False)`（展开版）→ 期望 `I+3·Sn`，实测 err ≈ 10（量级=Sn），即**未累加**。
   - ⇒ "用 gemm 累加矩阵幂 `M=I+ΣSn^j`" 的路子**不可用**。
2. **环路携带的 Cube 中接（L0C→GM→L1）在 `T.serial` 循环里不正确**：
   - 实验：`term=Sn@term` 三次（L0C→GM(fp16 fixpipe)→L1 反馈），期望 `Sn³@rhs`，实测 err ≈ 193（错）。
   - 加 `with T.rs("PIPE_FIX")/("PIPE_MTE2")` 亦无效。
   - ⇒ Neumann 的"幂/项反馈"（本征是 loop-carried）在当前工具链下**不可靠**。

**影响**：Neumann 求逆（以及 chunk 循环里状态 S 的 loop-carried 传递）都依赖"环路携带的 Cube 依赖"，这在本工具链上是**硬伤**。
**可能绕法**（待试）：① 把 chunk/幂循环**在 Python 层展开**（不用 `T.serial`）；② 换逆的算法（如按块 forward-substitution，小块）；③ 把 Neumann 改成**单次大 gemm 的增广矩阵**形式（避免环路）。
**状态**：`GatedDeltaNetPrefillFwdOp` 仍**未交付**（CG-2026-0026，WIP，阻碍为上述工具链限制）。

## 11. 结论：Neumann 在本工具链不可实现（再证）
第 10 节的"绕法①（Python 展开）"已试——**展开 + 复用 L1 + 显式 `T.rs("PIPE_FIX"/"PIPE_MTE2")` 同步仍 err=192.6**（同环路版）。
⇒ 根因不是 `T.serial` 循环，而是 **`L0C → GM(fp16 fixpipe) → L1` 把 matmul 输出再作为 matmul 操作数** 这条链**本身在本工具链不可靠**。
- Neumann（`W=(I+S_sys)^{-1}rhs`）与 chunk 状态递归都**必须**做这种"matmul 输出回喂" ⇒ **被堵死**。
- 结合第 10 节（`initC=False` 不累加），**Cube 侧的环路携带数值流在两个维度上都不可用**。
- 另：GLADecode（同族）能交付是因为**单步、纯 Vector、无 Cube 环路**；GDN prefill（chunked + Cube + 环路）不满足该条件。

**最终状态**：`GatedDeltaNetPrefillFwdOp` **未交付**（18/19），阻碍为工具链对"Cube 环路携带数值流"的支持缺陷（CG-2026-0026，已升级说明）。可能的**长期**出路：工具链修复（L0C→L1 直连 / initC 累加），或纯 Vector 顺序实现（慢但无 Cube 环路，需另评估）。

---

## 12. CANN 9.0 复探（2026-10-17）——**修正版**

### 12.0 先记一个致命的测量陷阱（本轮最大教训）

`.bashrc` 导出 `TILELANG_ASCEND_MODE=Developer`，而本批探针用
`os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")` —— **静默无效**，
于是前 11 个探针全部跑在 **Developer** 模式。Developer 模式下出现的三类"限制"**全是假象**：

- `'hivm.hir.store' op only support store ub to gm currently!`
- `multiple entry functions detected (previous: 'main_mix_aiv', this: 'main_mix_aic')`
- 以及由前者派生的"L0C 存过之后不能再用"

**纪律（已落实到所有探针脚本）**：模式必须**强制赋值** `os.environ["TILELANG_ASCEND_MODE"]="Expert"`
或在命令行 `export`；**不得用 `setdefault`**。

### 12.1 Expert 模式下的真实结论（全部重测）

| 项 | 结论 | 探针 |
|---|---|---|
| **MixCV（同 kernel 内 Cube+Vector）** | ✅ **可用**（单次交接、双循环多轮握手均通过） | `02` / `17` / `14` |
| 交付内核佐证 | `Conv3dBiasFwdOp` 在 9.0 下 **17/17 通过**（今日实测） | `tests/ops/test_conv3d_bias.py` |
| **L0C "存后再用"** | ✅ **无此限制**（A/B/C/D 与 E/F/G 全通过） | `08` / `09` |
| **`initC=False` 跨 gemm 累加** | ✅ **可用**，`I+3P` 实测 **err = 0** | `10` K1 |
| 回喂经 **out_idx** 张量 | ⚠️ 能编译但**数值错**（err 9.7 / 23.0 / 32.5，3 步链） | `10` K2 |
| 回喂经 **独立 fp16 scratch**（fixpipe 转型） | ✅ **正确**，err **9.7e-3**（fp16 链式舍入量级） | `13` K5 |
| **双循环 + 双向多轮握手** | ✅ **正确**，err **9.8e-4**（对比 fp16 通路参考） | `14` R0 |

> `14` 对"全 fp32 参考"的误差是 3.36，但它**全部来自 fp16 中间量舍入**；
> 用等价的 fp16 通路参考对齐后误差 9.8e-4，因此结构本身是正确的。

### 12.2 对 GDN 设计的直接影响

1. **结构可行**：单 Cube 段 + 单 Vector 段，**各自跑 chunk 循环**，用 `sync_block_set/wait`
   双向多轮握手（`14` 已验证，含动态 chunk 索引与多轮）。
2. **C2 Neumann 可行**：`initC=False` 做累加（K1）+ **独立 fp16 scratch 回喂**（K5）。
3. **必须避免**：把中间积回喂到 `out_idx` 张量（K2 的数值错）。
4. **精度**：fp16 回喂有舍入累积（3 步 9.7e-3）；Neumann 累加器应留 fp32，只在回喂时降到 fp16。

### 12.3 探针清单（一律 Expert 强制模式）

`07`（C1，9.0 版）｜`08`/`09`（L0C store/reuse 形式矩阵）｜`10`（累加 + out_idx 回喂）｜
`11`（scratch 回喂，旧版）｜`12`（三段式 Cube→Vector→Cube，见 12.0 说明）｜
`13`（**K5：fp16 scratch 回喂，可行方案**）｜`14`（**R0：双循环握手，结构验证**）｜
`15`/`16`/`17`（握手形态与 conv3d 骨架对照）｜`_dump_ir.py`（保留临时 IR）

### 12.4 CG-2026-0026 状态

- **CANN 9.0 上不再阻塞**：结构、累加、回喂三条路径均有实测可用的实现。
- 8.5.2 的阻塞结论**保留为历史记录**（§10/§11 不删），但在 9.0 上已降级为"工程量大"。
- 剩余真实风险：**Neumann 链长的 fp16 舍入累积**（需在 L1 阶段用 fp32 累加器 + 分块求解控制）。
