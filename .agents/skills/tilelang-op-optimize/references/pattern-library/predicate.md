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
| `isinf` | `vabs → vcmp(eq +inf) → vcast(bool→f16) → vcast(f16→i8)` | **4** | `vabs` 无 bf16 边 → 2×vcmp + 2×vcast + vadd + vcast（6） |
| `isfinite` | `vabs → vcmp(lt +inf) → vcast(bool→f16) → vcast(f16→i8)` | **4** | `vabs` 无 bf16 边 → 2×vcmp + 2×vcast + vmul + vcast（6） |

- **收益**（相对基线 6-pass）：isinf fp16 16m **−11.8~−12.2%**；isfinite fp16 16m **−12.4~−13.8%**；
  isnan vselect −2.8~−3.0%（基线 4-pass）。
- **bf16 反例**：`vselect` 在 bf16 上比 `vcast+vadd` 合并慢 **6.6~8.6%**（isinf 实测）——bf16 的
  mask 合并必须用 f16 cast + 算术，不能用 vselect。
- **上限**：16m/256m fp16 只有 57–65% SOL（16m fp32 76–78%），roofline advice =
  `latency bound:pipeline caused`；block 16384 已 UB 超限。**这是 ISA 级上限，不是分块可解**。
- 结构侧仍可叠加：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)`（−1~−4%）。
