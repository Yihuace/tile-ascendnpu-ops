# predicate 族性能模式（bool 输出：isnan / isinf / isfinite）

---
id: PL-1.23-predicate-pass-floor
kind: pattern
family: [elementwise]
apis: [T.vcmp, T.vcast, T.vselect, T.vsub, T.vmul, T.vadd, T.vabs, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: isnan-IsnanFwdOp-20261017T114928Z（+ isinf/isfinite 同日）
toolchain: tilelang 0.1.2 dev-root (AscendNPU-IR 31f6903) + CANN 9.0.0 (bishengir 1.1.0) / 910B3 / 2026-10-17
repro: repro-missing
---

### 1.23 predicate 族的 pass 下限与最短链 ✅ 已验证

bool 输出经 int8 字节存储（host `view(bool)`），**没有** `i1→i8` 直接 vcast、没有 bool/int1 的 `vnot`、
`T.vselect` 的 Out 不支持 int8（CG-2026-0027）⇒ 每条 predicate 至少 3 条向量 pass。实测最短链：

| 算子 | 链（fp16/fp32） | pass | bf16 |
|---|---|---:|---|
| `isnan` | `vcmp(le) → vselect(mask,0,1,f16) → vcast(f16→i8)` | **3** | 同左（vselect 可用；bf16 无 vabs 时的通用合并） |
| `isinf` | `vabs → vcmp(eq +inf) → vcast(bool→f16) → vcast(f16→i8)` | **4** | `vabs` 无 bf16 边 → 2×vcmp + 2×vcast + vadd + vcast（6）；**〔2026-10-18 二次深挖〕可 `vand(x, 0x7FFF)` 位运算符号位清除 → 4** |
| `isfinite` | `vabs → vcmp(lt +inf) → vcast(bool→f16) → vcast(f16→i8)` | **4** | `vabs` 无 bf16 边 → 2×vcmp + 2×vcast + vmul + vcast（6）；同 isinf 可 `vand(x, 0x7FFF)` → 4 |

- **收益**（相对基线 6-pass）：isinf fp16 16m **−11.8~−12.2%**；isfinite fp16 16m **−12.4~−13.8%**；
  isnan vselect −2.8~−3.0%（基线 4-pass）。
- **bf16 反例**：`vselect` 在 bf16 上比 `vcast+vadd` 合并慢 **6.6~8.6%**（isinf 实测）——bf16 的
  mask 合并必须用 f16 cast + 算术，不能用 vselect。
- **上限**：16m/256m fp16 只有 57–65% SOL（16m fp32 76–78%），roofline advice =
  `latency bound:pipeline caused`；block 16384 已 UB 超限。**这是 ISA 级上限，不是分块可解**。
- **〔2026-10-18 二次深挖更正，CANN 9.0.0 / 910B3，isnan 二次复核〕「ISA 级上限」归因不精确——真瓶颈是
  MTE2 带宽，不是 3-pass 向量链**。msprof PipeUtilization 实测（16m fp16 final：`aiv_mte2_ratio=0.81`
  vs `aiv_vec_ratio=0.39`；16m fp32：`0.84` vs `0.28`；256m fp16：`0.90` vs `0.25`）——fp16/fp32 的
  3-pass 向量链已完全隐藏于 MTE2 搬运窗口下。交叉验证：4-pass→3-pass 使 vec_time 12.95→10.06µs（fp16）
  但 Task Duration 仅 −3.4%（MTE2 21.2µs 不变）。**唯一例外 bf16**：baseline vec_t 20.2µs > mte2_t 20.1µs
  （向量受限），故 3-pass 收益最大（−5.6~−6.2%），3-pass 后翻转回 MTE2-bound（vec 0.52 < mte2 0.64）。
  有效带宽随 R:W 比与 N 退化：fp32(4:1) 1.70 TB/s > fp16(2:1) 1.63→1.24 TB/s(256m) ≈ bf16 1.45→1.18 TB/s。
  ⇒ 即便补齐 i1→i8 直出（消 1–2 pass），fp16/fp32 SOL 也几乎不动（vec 已非瓶颈）；CG-2026-0027 的性能
  影响评估须从「3-pass 钉死 SOL」修正为「MTE2 混合 R/W 带宽地板钉死 SOL，3-pass 已隐藏」。
- 结构侧仍可叠加：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)`（−1~−4%）。
- **〔2026-10-18 二次深挖，isinf 二次复核〕bf16 的 `|x|` 用 `vand(x, 0x7FFF)` 位运算符号位清除（1 op）——
  0x7FFF 掩码 `T.fill(-0.0) → T.vnot` 一次性构造（循环外），`T.vand`/`T.vnot` 均支持 bf16 按位级
  reinterpret（`docs/Tilelang.language/逻辑操作/` 2.2.1）。使 isinf/isfinite 的 bf16 从 6-pass 降到与
  fp16/fp32 相同的 4-pass。收益（isinf 实测）：16m bf16 **−19.4%**、16m-half bf16 −19.9%、256m bf16
  **−15.4%**（相对 round-4 bf16 6-pass 只有 −0.2~−0.9%）。pipe 证据：bf16 baseline `aiv_vec_ratio`
  0.72–0.76 ≫ `aiv_mte2_ratio` 0.36–0.40（向量受限），4-pass 后 vec_ratio 0.55–0.62 ≈ mte2 0.46–0.53
  （收敛均衡）。bit 级掩码天然保留 NaN mantissa，`vcmp eq` ordered 比较正确排除 NaN（比 `vmax(x,vsub(0,x))`
  2-op 更省且 NaN 语义无歧义）。这是 predicate 族目前最大单点收益，超过 fp16 vabs 的 −11.8~−12.2%。
- **〔2026-10-18 二次深挖，isfinite 二次复核 = 第二次独立证据〕** isfinite 复用同路径（`vand(x, 0x7FFF)`
  掩符号位 + `vcmp(lt +inf)`，ordered `lt` 同样正确排除 NaN），把 isfinite bf16 从 6-pass 降到 4-pass。
  实测：16m bf16 **−21.24%**、16m-half bf16 −21.29%、256m bf16 **−16.42%**（相对 round-2 bf16 6-pass
  只有 −2.2~−2.8%）。pipe 证据（256m bf16）：`aiv_vec_time` 665.2→419.7µs（−36.9%），`aiv_vec_ratio`
  0.72→0.55、`aiv_mte2_ratio` 0.39→0.53（收敛均衡）。与 isinf 双算子复现，vand 4-pass 是 bf16 predicate
  `|x|` 的普适路径（不再依赖 `vabs` bf16 边）。
