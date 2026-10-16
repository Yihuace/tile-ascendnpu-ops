# GatedDeltaNetPrefillFwdOp — 迁移设计（WIP）

> 交付名 `GatedDeltaNetPrefillFwdOp`；基线 op 名 `GDNFwdOp`；kernel `kernels/linear_attention/gdn/prefill.py:29`（`GDNDensePrefillFwdKernel`）。工具链：tilelang 0.1.2+417dae99 / CANN 8.5.2 / Ascend910B3。
> 状态：**未交付**（18/19）。本文件汇总**已验证的算法/设计**与**工具链阻碍**，供后续收口。

## 1. 算法（chunked gated delta rule）
每 chunk（C token，state `S[D,DV]` fp32）：
```
gc[l]      = cumsum_{s<=l} g[s]
G[l,s]     = exp(gc[l]-gc[s]) if l>=s else 0
S_sys[l,s] = beta[l]*G[l,s]*(k_l·k_s) if l>s else 0     (strictly lower)
rhs[l]     = beta[l]*(v_l - exp(gc[l])*(k_l @ S))
W          = (I+S_sys)^{-1} rhs                          (Neumann series)
o[l]       = scale*exp(gc[l])*(q_l @ S) + Σ_s scale*G[l,s]*(q_l·k_s)*W[s]
S_new      = exp(gc_last)*S + Σ_l exp(gc_last-gc[l]) * k_l (x) W[l]
```
Golden = 独立 pure-torch **sequential** 递推（逐 token delta rule）。

## 2. 已验证的 Expert 构建块（真机对拍通过）
| 相位 | 内容 | 状态 | 误差 |
|---|---|---|---|
| C1 | Cube：`KK=k·kᵀ`/`QK`/`KS=k·S16`/`QS` | ✅ | ~1e-6 |
| V1 | Vector：`gc=cumsum(g)`、`G_strict/G_incl`、`exp(gc)` | ✅ | gc 精确 / G 6e-8 |
| V1b | `S_sys=β·G_strict·KK→sn16`；`rhs=β·(v−exp(gc)·KS)` | ✅ | 1.2e-3(fp16) / 2.4e-7 |
| C2 | Neumann `W=(I+S_sys)⁻¹rhs` | ❌ | 工具链阻塞（见 §3） |
| C3/V3 | `o`、状态传递 | ⬜ | — |

关键 API 规则（踩坑）：向量 op 的 dst 必须 UB；`T.copy` 不做 dtype 转换；Cube↔Vector 的 GM 交接需显式 `sync_block_set/wait`。

## 3. 工具链阻碍（CG-2026-0026，已升级）
1. 原 Developer 模式探针：Cube 路径 `error pointer address space cast` → 改 Expert。
2. **`T.gemm(initC=False)` 不跨 gemm/循环累加**（`I+3Sn` 实验失败）。
3. **环路携带的 `L0C→GM→L1` 回喂不可靠**（`Sn³@rhs` 实验 err≈193；展开 + `T.rs` 同步亦无效）。
⇒ Neumann（及 loop-carried chunk 递归）依赖的"Cube 环路携带数值流"在本工具链不可用。

## 4. 后续收口路径
1. **等/换工具链**：`L0C→L1` 直连 或 `initC` 累加支持（可能 CANN 9.0/新版 AscendNPU-IR 已改善）；
2. **纯 Vector 顺序实现**（逐 token 递推，无 Cube 环路）——能绕开但慢，需评估能否接受；
3. **改逆的算法**（如小块 forward-substitution 规避回喂）。

## 5. 参考
- 探针与实验：`examples/gdn_prefill/probe/`（`gdn_prefill_probe.py`、`EXPERT_FINDINGS.md`、`experiments/01..06`）。
- gap：`.agents/evolution/capability-gaps.md` CG-2026-0026。
