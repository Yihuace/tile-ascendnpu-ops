# 18 算子剩余优化空间评估（2026-10-18）

> **方法**：只基于**已有实测证据**（opt_log / perf_records / roofline advice / pipe ratio / 已试否决项）推断，**不新增测量**。
> **口径**：有 8.5 报告的算子沿用 8.5 值（用户口径：能衔接就衔接）；本轮新做的 11 个为 9.0 值。
> **「剩余空间」是估计值**，标注依据；实际能否拿到取决于实现与编译器。

## 总表

| # | 算子 | 当前 ratio | 瓶颈归因（证据） | 剩余空间 | 杠杆 | 类型 | 工作量 | 优先级 |
|---|---|---|---:|---|---|---|---|---|
| 1 | **Conv1d / Conv1dBias** | 0.25–1.8% | `conv1d` 通用 kernel 仍是「每核冗余搬运」（wav2vec2 0.33%、encodec-init 0.41%，从未优化） | **2–5×** | 复用 unit-stride 的 co 块切分到通用 kernel | 参数级 | 3–4h | **P0** |
| 2 | **Conv3dBias** | 0.25→0.51% | unet 路径权重 L1 常驻未生效（`c_in_pad=32`，编译器仍逐 task 从 L2 重载） | **~2×**（仅 unet） | 改 Cube 循环结构 / 换常驻形态 | 准设计级 | 4–6h | **P0** |
| 3 | **FusedAddLayerNorm** | 29.2–45.2% | 15+ 向量 pass + 2 次 row-reduce 全在 Vector（`advice=pipeline caused`） | **20–40%** | 两次 reduce 改 Cube GEMV 卸载（BatchNormBwd 有先例） | **设计级** | 1–2 天 | **P1** |
| 4 | **BatchNormBwd** | ratio 不可比（13.0–25.5%） | fp32 中转 + Vec_MISC；时延已 −40% | 10–20% | epilogue 融合 / 减 fp32 中转 | 参数级偏设计 | 4–6h | P1 |
| 5 | **GLADecode** | 3.2–18.3% | latency-bound：每 (b,h) 一核、并行度 = B×H | 10–20% | persistent 分核 `grid=min(B*H,40)`（上轮因欠订 workload 反噬未做） | 参数级 | 2–3h | P1 |
| 6 | **Lerp** | 8.5: 80.8%（单点）<br>9.0 抽测 60.2–82.0% | 访存受限；`linear-bias` 两档 66.5/60.2 掉线 | 5–10pp | 同族 A4 结构（静态尾块 + 流水） | 参数级 | 1–2h | P2 |
| 7 | **Sub** | 66.8–84.7% | 访存受限；仅 1 个 bf16 shape 66.76（漂移） | ~3pp（1 个 shape） | 该 shape 的交错复测 + 噪声裁定 | 参数级 | 0.5h | P2 |
| 8 | **ClampScalar** | 8.5: 71.9–82.8%<br>9.0 抽测 58.2–70.2% | 访存受限；9.0 抽测时延 +4~33%（codegen 回退，链路也不同） | 不确定 | 先定性「工具链 vs 链路」，再决定是否重做 | 诊断 | 1h 诊断 | P2 |
| 9 | **ClampMin** | 8.5: 73.2%（单点） | 与 ClampScalar 同 kernel | ≤5pp | 复用 ClampScalar 结论 | 参数级 | 0.5h | P3 |
| 10 | **Add** | 70.8–82.6% | 访存受限，已贴实测上限（同族最高 84.7%） | 0–3% | 无新杠杆 | — | — | P3 |
| 11 | **Hardswish** | 50.2–61.0% | 向量利用率 **0.94**，链已最短（无 fused vmad） | ~0 | 判据失配，非访存 | — | — | P3 |
| 12 | **Hardsigmoid** | 53.6–63.2% | 向量 0.85，fp32 链 5 op 已语义最小（CG-2026-0006：fp16 无吞吐收益） | 0–5% | 无 | — | — | P3 |
| 13 | **Isnan** | 55.3–78.5% | **MTE2 混合读写带宽地板**（mte2 0.81–0.90 ≫ vec 0.25–0.39）；4→3 pass 仅 −3.4% | ~0 | 需编译器补 `i1→i8`（CG-2026-0027） | 阻塞 | — | P3 |
| 14 | **Isinf** | 41.4–74.7% | 同上（bf16 已用 `vand` 4-pass） | ~0 | 同上 | 阻塞 | — | P3 |
| 15 | **Isfinite** | 41.7–78.0% | 同上 | ~0 | 同上 | 阻塞 | — | P3 |
| 16 | **GeluTanhAndMul** | 22.0–47.7% | 向量 0.95；Vtanh 不可用，sigmoid 恒等折叠已最小 | ~5% | 换 tanh 近似实现（设计级） | 设计级 | 1 天 | P3 |
| 17 | **GeluAndMul** | 32.7–47.7% | 向量 0.943；erf 阶数已 dtype 分化（本轮 −43.6%） | ~5% | 进一步降阶需精度冒险 | 设计级 | 1 天 | P3 |
| 18 | **GatedDeltaNetPrefill**<br>（19 接口之一，未完成） | 无 | blocksolve 卡 CG-2026-0003（Cube doubling + L0C→GM→L1 回喂） | 未知 | 先做「换算法形态能否编译」判定探针 | 设计级 | 探针 1–2h | P2 |

\* Conv1d 的 wav2vec2 bench 形状当前撞同一编译器缺陷（pre-existing），优化前需先确认可编译。

## 结论

**真正还有大空间的只有 3 个**（P0/P1）：

1. **Conv1d 通用 kernel**（wav2vec2 / encodec-init 两个家族）：SOL 0.33–0.41%，是当前最差的路径；unit-stride 的 co 块切分已证明能拿 15×，通用 kernel 大概率同构可复用。**但 wav2vec2 那个 bench 形状目前编译崩，需先解决/绕过。**
2. **Conv3d unet 路径**：权重常驻在 `c_in_pad=32` 不生效，已归因到编译器行为（`scalar_mte2_stall` 不变），需要改 Cube 循环结构。
3. **FALN 设计层**：把两次 row-reduce 改 Cube GEMV，是唯一能突破 45% 的路，但要回设计阶段。

**中间档 3 个**（P1）：BatchNormBwd（epilogue/中转）、GLA（persistent 分核）、Lerp（linear-bias 两档）。

**其余 9 个基本到顶**：Add、Hardswish、Hardsigmoid、Isnan、Isinf、Isfinite、Gelu×2、ClampMin——继续投入收益 <5%，或需要编译器补 API。

## 建议的开工顺序

| 顺序 | 任务 | 预计 | 前置 |
|---|---|---|---|
| 1 | Conv1d 通用 kernel 的「是否可编译」判定（先解决 wav2vec2 bench 崩溃） | 1h | 无 |
| 2 | 若可编译 → co 块切分移植（目标 2–5×） | 3h | 1 |
| 3 | Conv3d unet 权重常驻形态改造（目标 2×） | 4–6h | 无 |
| 4 | GLA persistent 分核（目标 10–20%） | 2–3h | 无 |
| 5 | FALN Cube 归约（目标 20–40%） | 1–2 天 | 需设计评审 |
