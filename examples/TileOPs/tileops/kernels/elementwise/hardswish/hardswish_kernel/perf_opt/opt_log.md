# HardswishFwdKernel — Stage-4 optimize log

- 算子 `HardswishFwdOp`；纯 Vector（AIV）；单 kernel/单 func。
- 口径 `msprof op Task Duration`；工具链 tilelang 0.1.2+417dae99 / CANN 8.5.2 / Ascend910B3。

## 诊断
fc-wide fp16 `PipeUtilization`：**aiv_vec_ratio=0.939**、mte2=0.534、mte3=0.275 → **compute-bound（向量打满）**。
baseline 计算链已是最少：`vadd(x+3) → vclamp(0,6) → vmul(x,·) → vmul(1/6)`（+2 vcast）；
本 kernel **无 NaN 恢复链**（设计已注明），无可删除的冗余 op。

## 结论：无实质性能优化空间（接近计算天花板）
- 无冗余 op 可删；4 条计算 op 已是 hardswish 公式的最小形态（无 fused vmad 可用）。
- 因此"带宽利用率 60%"判据对本算子**不适用**（compute-bound；正确判据是向量利用率，已达 0.94）。

## 仅做的改动：fp32 健壮性修复
`_DEFAULT_BLOCK_SIZE` fp32 8192 → 4096（fp32 多缓冲 UB >192KB 编译失败；benchmark 无 fp32 用例故 baseline 未暴露）；
`__init__` 构造时传入 block_size。

## A/B（msprof，均在噪声内）
| label | dtype | base µs | opt µs | base% | opt% |
|---|---|---:|---:|---:|---:|
| fc-wide | fp16 | 20.12 | 19.88 | 54.6 | 55.2 |
| fc-wide | bf16 | 21.12 | 20.92 | 60.6 | 61.2 |
| mbv3-stage2/3 | fp16/bf16 | 15.8–24.3 | 15.7–23.9 | 49.9–60.4 | 50.3–61.6 |
