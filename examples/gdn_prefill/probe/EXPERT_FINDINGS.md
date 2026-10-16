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
