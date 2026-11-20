# GLADecodeOp 设计说明

## 1. 语义

单 token（decode 阶段）gated linear attention，layout 为 `BHD`：

```text
state_new[b,h,dk,dv] = exp(gk[b,h,dk]) * state[b,h,dk,dv] + k[b,h,dk] * v[b,h,dv]
o[b,h,dv]            = scale * sum_dk q[b,h,dk] * state_new[b,h,dk,dv]
```

`scale` 在构造时固定；非正值表示 `DK ** -0.5`。`state` 以函数式返回为 `new_state`，不原地修改。

## 1.6 算法调研与优化分析（历史补记，2026-11-19）

> 本设计成稿早于 op-dev 的 §1.6 模板；此处按现有 `perf_opt/opt_log.md`（R6–R49）与官方 report 回填，
> 不改变已交付实现。

### 1.6.0 算法调研（四问）

| 问题 | 结论 | 依据 |
|---|---|---|
| 等价化简公式 | 无可化简公式：decode 是 `state_t = exp(g_t)·state_{t-1} + k_t⊗v_t`、`o_t = scale·q_t·state_t` 的单步递推，不存在减少乘加的代数恒等式；R40 代数变体 A/B 未找到等价更简式 | §1；`perf_opt/opt_log.md` R40 |
| 在线算法 | 已是在线形式（逐 token 递推，无需回看历史），无进一步在线化对象 | §1、R40 |
| 复杂度 | 每头每 token `O(DK·DV)`；state 读+写 2·DK·DV 元素，q/k/v/g 各 1 次 | §4 roofline |
| 硬件亲和 | 192 KB UB / 40 AIV；`k_tile` 只能取 32（64 装不下）；2-head 交错可共用 K/V 载入 | §3 分派、REVIEW 检查项 |

候选表（基线 → 采纳/否决）：

| 候选 | 结构 | 结论 |
|---|---|---|
| 基线 | 单头、`k_tile=128` | 基线（见 opt_log R6/R14） |
| R6 / R14 | 2-head 交错 + `k_tile` 64/128 按 shape 分流 | 采纳 |
| R45 / R46 | state 乒乓 + 复用死亡 `ou` 直接预取 | 采纳（R47 按 `B·H<=32` 分流小盘） |
| R49 | b1-h8 K 分片 + o-master（5 block/头） | 否决：3.34→4.85 µs（+45%） |
| 全 shape `k_tile=64` | — | 否决：超 192 KB UB |
| DV split | — | 否决：+25.7% |

复杂度四口径（每 head · 每 token）：

| 口径 | 值 |
|---|---|
| FLOPs | `≈5·DK·DV`（`q·state` 2·DK·DV + state 更新 3·DK·DV） |
| 访存 Bytes | `(3·DK + 2·DV + 2·DK·DV)·elem_bytes`（q/k/v/g + state 读+写） |
| 扫描遍数 | **1**（单遍 token 递推，无多 pass 重扫） |
| 中间缓冲峰值 | state 双 plane + k/v staging；3-plane 形态为 199,936 B，超过 196,608 B 上限，R46 删死平面后装下（≈192 KB 量级） |

### 1.6 数学等价优化

| 原式 / 原形态 | 优化后 | 等价论证 | 收益 |
|---|---|---|---|
| 单头顺序处理 | 2-head 交错（共用 K/V 载入） | 各 head 递推彼此独立，只合并调度，不动算术 | b8-h64 −9.7%（R10） |
| `state` 单平面回写 | `st0/st1` 乒乓 | 同一 state 的两个时步；写后读距离 1→2，仅内存别名变化 | 52.00→48.98 µs（R45/R46） |
| `ou→st` 拷贝落地 | 预取直接落另一平面 | 复用死亡 `ts` 平面后 UB 装得下；数值路径不变 | b8-h64 21.00 µs（R47） |
| fp32 state | fp16 state（bf16 输入） | **精度契约变更（非等价）**：容差 2e-2 内，5 shape×10 轮 max 8.28e-3 | 5-shape −6.3% |

### 1.6.1 等价性验证

- 采纳项均为**调度 / 内存别名 / 精度契约**变更，没有减少算术运算的等价变形 → 按标准「无优化空间」豁免
  `verify_equiv.py`（历史算子补记）。
- 机器可复核证据：官方 report 11/11 passed；R46/R47 5 shape × 10 轮随机输入共 50 次，max abs err 8.28e-3 < 2e-2
  （`reports/2026-11/2026-11-18-三算子9.3固化.md` §3.2）。

### 1.6.2 向量化替代分析

- 计算链全部为 v-prefix 向量指令（`vmul/vadd/vcast/vbrc`）；无逐元素标量循环，标量只用于分派与循环边界。

### 1.6.3 弃选行（佐证）

- `k_tile=64`：192 KB UB 装不下（REVIEW「tiling」检查项）；DV split：+25.7%；
  b1-h8 5-block：+45%（`perf_opt/opt_log.md` R49 负结果）。

## 2. 实现结构

- 单 kernel / 单 func，纯 Vector（AIV）。
- 输入 `q/k/v/gk/state`，输出 `o/new_state`；bf16 输入、fp32 累加中转。
- DK、DV 默认 128；K 方向按 `k_tile` 分块，tile 内做 `exp(gk)`、state decay、`k v^T` 更新与 `q^T S` readout。
- `k_tile` 默认 32：按 910B3 实测 UB 容量 192KB 计算，`k_tile=32` 可容纳（约 162KB 手工预算 × BishengIR 1.10–1.12 膨胀）；`k_tile=64` 需 294KB+，超 UB，否决。

## 3. 并行与分派

- 基础形态：每个 `(b,h)` 一个核，grid = `B*H`。
- 超发档（`B*H > 40`）：`T.Kernel(min(B*H, 40))` + 核内 grid-stride，把 block 数收到物理核数，避免超发。
- 欠订档（`B*H <= 40`）：保持 flat，避免 persistent 的额外循环开销。
- 编译期按 `B*H` 与整除规则分派，两档共享同一数学实现。

## 4. 访存与 roofline

- 每 token 的 GM 访问约为 `3*DK + 2*DV + 2*DK*DV` 个元素；AI ≈ 3.64 Ops/Byte（DK=DV=128）。
- `AI × GM_bw(1.75TB/s) ≈ 6.4 TOps/s`，远低于 Vector 算力 ⇒ 访存受限，但 decode 的实际瓶颈是 **latency-bound**（并行度仅 `B*H`，逐头 state 递归串行）。
- 因此本算子不以 70% GM SOL 作为可达目标；优化以 Task Duration 为准。

## 5. 精度与测试层级

- 参考实现 `golden_gla_decode`（fp32，CPU），容差：fp16 `atol=rtol=2e-3`，bf16 `2e-2`，fp32 `1e-4`。
- 测试覆盖 L0/L1/L2/boundary 与 manifest bench 共 11 个用例；无 skip、无降低容差。

## 6. 来源

- GPU 基线：`GLARecurrentFwdOp`（TileOPs，commit `1238a9f4`）。
- 迁移方式：direct implementation（由 `examples/gla_decode/probe` 直接实现，未走 extract/integrate harness）。
