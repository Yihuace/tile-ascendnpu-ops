# GatedDeltaNetPrefillFwdOp — 迁移设计（rev.2，CANN 9.0）

> 交付名 `GatedDeltaNetPrefillFwdOp`；基线 op `GDNFwdOp`；GPU kernel `kernels/linear_attention/gdn/prefill.py:29`（`GDNDensePrefillFwdKernel`）。
> 工具链：**CANN 9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2（dev root）/ Ascend910B3。
> 本 rev.2 基于 rev.1 的 WIP + 2026-10-17 的 9.0 复探结论（`probe/EXPERT_FINDINGS.md` §12）。
> **rev.1 的"C2 不可实现"结论已被推翻**：`initC=False` 累加在 9.0 已修复，回喂有可行绕法。

---

## 1. 范围（本次迁移的 in-scope / out-of-scope）

上游 manifest 共 34 个 workload（含 decode、varlen、state_v_first、gate-in-kernel 等）。**本次只交付 prefill 主干**：

**In scope（L0/L1/L2）**

| 维度 | 取值 |
|---|---|
| `B` | 1（equal-length，无 `cu_seqlens`） |
| `S` | 4096 / 12288 / 32768 / 3000（非整 chunk） |
| `H = HV` | 16 |
| `K = V` | 128，另加 64（`prefill-narrow-state`） |
| `dtype T` | float16 / bfloat16 |
| `initial_state` | 有 / 无（`prefill-continued`） |
| 参数开关 | 全部默认：`state_v_first=False`、`use_gate_in_kernel=False`、`use_beta_sigmoid_in_kernel=False`、`use_qk_l2norm_in_kernel=False`、`allow_neg_eigval=False`、`scale=None` |

**Out of scope（本次不交付，显式记录）**

- `cu_seqlens` 变长（varlen-*、q35-*-NxM 全部）；
- `state_v_first=True`；
- `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel` / `use_qk_l2norm_in_kernel` / `allow_neg_eigval`（`prefill-raw-*`）；
- decode 路径（`S == 1`，由同族 `GLADecodeOp` 之外的 decode 内核承担；本 op 只做 prefill）。

> 判定依据：out-of-scope 项全部是"输入前处理 / 布局 / 变长索引"，与核心 delta-rule 数学正交，可在本 op 打通后按同一相位结构增量扩展。

## 2. I/O 契约

```
q,k         [1, S, H,  K]  T
v,g,beta    [1, S, HV, V] / [1,S,HV]  T
state_in    [1, HV, K, V]  float32   (optional)
o           [1, S, HV, V]  T
state_out   [1, HV, K, V]  float32
scale       float = K**-0.5（默认）
```

golden：**纯 torch 顺序递推**（逐 token delta rule），CPU 可独立运行。

## 3. 算法（每 chunk，C token）

```
gc[l]      = cumsum_{s<=l} g[s]
G[l,s]     = exp(gc[l]-gc[s])   (l>=s)，否则 0
S_sys      = diag(beta) · G_strict ⊙ KK,   KK = k·kᵀ        (严格下三角)
rhs        = diag(beta) · (v − diag(exp(gc)) · KS),  KS = k·S
W          = (I + S_sys)^{-1} · rhs
o          = scale·diag(exp(gc))·QS + scale·(G ⊙ QK)·W       QK=q·kᵀ, QS=q·S
S_new      = diag(exp(gc_last))·S + (diag(grev)·k)ᵀ · W
```

## 4. 9.0 工具链事实（Expert 模式实测，`EXPERT_FINDINGS.md` §12）

> **模式纪律**：`.bashrc` 导出 `TILELANG_ASCEND_MODE=Developer`。所有探针/内核必须
> **强制** `os.environ["TILELANG_ASCEND_MODE"]="Expert"`（`setdefault` 无效——本轮曾因此得出三条假限制）。

| 项 | 结论 | 证据 |
|---|---|---|
| MixCV（Cube+Vector 同 kernel） | ✅ 可用；交付内核 `Conv3dBiasFwdOp` 在 9.0 下 17/17 | `02`/`17`/tests |
| 双循环 + 双向多轮握手 | ✅ 正确（err 9.8e-4 vs fp16 通路参考） | `14` |
| L0C "存后再用" | ✅ 无限制 | `08`/`09` |
| `initC=False` 跨 gemm 累加 | ✅ 可用（err = 0） | `10` K1 |
| 回喂经 **out_idx** 张量 | ⚠️ 能编译、**数值错**（err 9.7~32） → **禁止** | `10` K2 |
| 回喂经 **独立 fp16 scratch**（fixpipe 转型） | ✅ 正确（9.7e-3，fp16 舍入） | `13` K5 |

## 5. Kernel 结构（两段并发 + 多轮握手）

```
T.Kernel(grid = (nchunk_total, HV), is_npu=True)
  T.Scope("Cube"):                     # 单段，跑全部 chunk
      for ck in serial(nch):           # 本 head 的 chunk 循环
          wait(V_flag[ck])             # 等 V1b 产出 S_sys/rhs
          C1: KK/QK/KS/QS 或直接复用
          C2: Neumann W（K5 回喂，独立 L0C）
          set(C_flag[ck])
          wait(o_flag[ck])             # 等 V3 写完 o/state
          C3: 输出 gemm
  T.Scope("Vector"):                   # 单段，跑全部 chunk
      for ck in serial(nch):
          V1 : gc / G_strict / exp(gc)
          V1b: S_sys=β·G⊙KK, rhs=β·(v−exp(gc)·KS)
          set(V_flag[ck])
          wait(C_flag[ck])
          V3 : o 收尾、state 更新、写回
```
- flag 单调递增、按 `% K` 复用；`set` 用产出侧的 pipe（`PIPE_FIX`/`PIPE_MTE3`），`wait` 用消费侧（`PIPE_MTE2`）。
- **R0 已验证（`14`）**：双循环 + 双向多轮握手 + 动态 chunk 索引在 9.0 上可正确运行（err 9.8e-4 vs fp16 通路参考）。
- **禁止**：把中间积回喂到 `out_idx` 张量（`10` K2 数值错）；回喂一律走**独立 fp16 scratch**。

## 6. C2：Neumann 求逆（K5 模式）

`(I+S_sys)^{-1} rhs`，`S_sys` 严格下三角 ⇒ 级数在 `S_sys^C = 0` 处截断。采用**迭代幂累加**：

```
t0 = rhs
t_{j+1} = S_sys @ t_j            (j = 0 .. C-2)
W = Σ_j (-1)^j t_j  ⇒  等价 W = (I+S_sys)^{-1} rhs
```

实现要点（全部实测）：
- 累加器用 `initC=False`（9.0 已支持）；
- 每个 `t_j` 用**独立 L0C**（R1）；
- 回喂走 **fp16 GM scratch**（R3）：`L0C(fp32) → gm16(fixpipe 转型) → L1(fp16)`；
- 精度：fp16 链式舍入实测 9.7e-3（3 级）；C=64 时链长 63 级，**必须把累加器留 fp32**，且按块 forward-substitution 可显著缩短链长（待 L1 阶段评估）。

## 7. 精度标准

- `float16`：`max_abs_err ≤ 2e-3`（相对 O(1) 输出）；`bfloat16`：`≤ 1.6e-2`（沿用同族 `GLADecodeOp` 的 bf16 口径）。
- state（fp32）：`max_abs_err ≤ 5e-3`。
- 逐层：L0（单 chunk、单 head）→ L1（多 chunk、非整 chunk 尾）→ L2（全 shape）。

## 8. L0 / L1 / L2 测试计划

| 层 | 用例 | 目的 |
|---|---|---|
| ~~R0 探针~~ | 双向多轮握手（2 chunk） | ✅ **已通过**（`14`，err 9.8e-4） |
| L0 | `S=64, C=64, H=HV=1, K=V=16` | 单 chunk 全相位数值 |
| L0b | 同上 + `state_in` | 状态注入路径 |
| L1 | `S=256, C=64, H=HV=1, K=V=64` | 多 chunk 传递 |
| L1b | `S=3000, C=64` | 非整 chunk 尾处理 |
| L2 | in-scope 全 shape（含 fp16/bf16） | 交付验证 |

## 9. 风险与未决

| 风险 | 影响 | 处置 |
|---|---|---|
| ~~R0 双向多轮握手未验证~~ | ✅ **已排除**（`14`：err 9.8e-4 vs fp16 参考） | — |
| Neumann 链长（C-1 级）×fp16 回喂的误差累积 | 精度不达标 | 累加器 fp32 + 分块 forward-substitution；必要时 C=32 |
| L0C 容量：每 chunk 需 4~6 个 `[C,C]` fp32 | C=64 时 4×16KB=64KB（<128KB，可行）；C=128 时需复核 | 固定 C=64 |
| `S_sys` 的 `G⊙KK` 是 Hadamard（非 gemm） | 必须在 Vector 段做 | 已在 §5 的 V1b 内 |
| 长序列 GM 中间量（KK/S_sys per chunk） | S=32768 时约 64MB | 按 chunk 逐块生产/消费，不整存 |
| **模式误用（Developer）** | 曾导致三条"假限制"，浪费一轮 | 所有脚本强制 `os.environ[...]="Expert"` |

## 10. 参考

- 探针与实测：`examples/gdn_prefill/probe/`（`EXPERT_FINDINGS.md` §12 + `experiments/07..13`）
- gap：`.agents/evolution/capability-gaps.md` CG-2026-0026（9.0 上已降级为"工程量大"，非"不可实现"）
- 上游参考：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill.py`

---

## 11. Stage 0 记录（2026-10-17，按迁移工作流执行）

### 11.1 状态机

```
statectl init --dir tileops/kernels/linear_attention/gdn_prefill \
   --scenario migration --migration-mode harness --stage-plan "0,1,2,3,5"
statectl migration init --functions <6 个函数>
statectl start 0        # ✅
statectl gate 0         # ✗ 缺 3 个脚手架文件（见 11.3）
```

### 11.2 关键事实：prefill 是 **6 个 TileLang 内核**的流水线

自动抽取器 `extract_tl_kernel.py --op-name GDNFwdOp` **只解析到 decode**
（`delta_decode_tl`），未跟到 prefill 的 import。人工枚举 prefill 实际需要迁移的 6 个内核
（`src/tileops/kernels/linear_attention/gdn/`）：

| # | 函数 | 源文件 | 行 |
|---|---|---|---|
| 1 | `prefill_chunk_local_cumsum_bthd_tl` | `prefill_prepare.py` | 22 |
| 2 | `prefill_blocksolve_A_bthd` | `prefill_prepare.py` | 654 |
| 3 | `get_warmup_chunks` | `prefill_prepare.py` | 772 |
| 4 | `correct_initial_states` | `prefill_prepare.py` | 998 |
| 5 | `fused_gdr_h` | `prefill_prepare.py` | 1469 |
| 6 | `fused_gdr_fwd` | `prefill_forward.py` | 567 |

合计约 2248 行 GPU TileLang → **6 ×（Stage 1 设计 → Stage 2 检视 → Stage 3 开发）**。

### 11.3 工具缺口（Stage 0）

1. `extract_tl_kernel.py` 的 `_resolve_targets` 只处理 `self.kernel` 可达的 builder；
   GDN prefill 的 6 个内核由 wrapper 直接 import（非 `self.kernel` 成员），**未被发现**。
2. 任务书提到的 scaffolder（生成 7 文件脚手架 + `.migration_meta.json`）**不在仓库中**
   → `.migration_meta.json` 已按既有算子 schema 手工生成；脚手架需手工补。
3. `gate 0` 当前失败项（即 Stage 0 待补工件）：
   - `tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py`（wrapper）
   - `tests/ops/test_gdn.py`
   - `benchmarks/ops/bench_gdn.py`
