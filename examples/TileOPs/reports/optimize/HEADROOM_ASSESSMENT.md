# 18 算子剩余优化空间评估（2026-10-18）

> **方法**：只基于**已有实测证据**（opt_log / perf_records / roofline advice / pipe ratio / 已试否决项）推断，**不新增测量**。
> **口径**：全部为 **CANN 9.0.0** 数据（2026-10-18 统一，含 4 个 历史口径残留的端到端重测）；上一代工具链数据不再引用。
> **「剩余空间」是估计值**，标注依据；实际能否拿到取决于实现与编译器。

## 总表

| # | 算子 | 当前 ratio | 瓶颈归因（证据） | 剩余空间 | 杠杆 | 类型 | 工作量 | 优先级 |
|---|---|---|---:|---|---|---|---|---|
| 1 | **Conv1d / Conv1dBias** | 0.03–1.79% | 通用 kernel 已完成 taps-in-K（wav2vec2 **17.7×**、encodec-init **4.2×**）；bias 路径受 **CG-2026-0028 + CG-2026-0029** 双重阻塞（bias-in-K 不可表达；ws 宽度必须 128，N tile 放大会 dual-entry） | **blocked** | 独立 bias-add kernel / 等编译器修 MixCV entry 合并 | 准设计级 | 4–6h | P3 |
| 2 | **Conv3dBias** | 1.68–20.43% | unet 已完成 **21.3×**（im2col DMA 段数 91774→973）；L1 常驻早已生效但不是杠杆 | ~0（已到 L0C 上界） | 无 | — | — | P3 |
| 3 | **FusedAddLayerNorm** | 0.5–43.9%（微型豁免后 29.2–43.9%） | 15+ 向量 pass + 2 次 row-reduce 全在 Vector（`advice=pipeline caused`） | **20–40%** | 两次 reduce 改 Cube GEMV 卸载（BatchNormBwd 有先例） | **设计级** | 1–2 天 | **P1** |
| 4 | **BatchNormBwd** | 11.9–24.4%（subid 不可比） | fp32 中转 + Vec_MISC；时延已 −40% | 10–20% | epilogue 融合 / 减 fp32 中转 | 参数级偏设计 | 4–6h | P1 |
| 5 | **GLADecode** | 3.10–20.16% | latency-bound；persistent 已完成（B*H>40 才启用，超发 −8.0~14.3%；欠订 flat 零回退） | ~0 | 无（Pipelined 实测无收益） | — | — | P3 |
| 6 | **Lerp** | 60.2–82.0%（9/11） | 访存受限；`linear-bias` 两档 66.5/60.2 掉线 | 5–10pp | 同族 A4 结构（静态尾块 + 流水） | 参数级 | 1–2h | P2 |
| 7 | **Sub** | 70.7–84.0%（15/15） | 访存受限；9.0 端到端已全 shape 过线 | ~0 | 无新杠杆 | — | — | P3 |
| 8 | **ClampScalar** | 58.2–70.2%（1/11；baseline 45.4–64.4%） | 访存受限；9.0 整体下移，但 tuned 仍比 baseline 快 7.6–40% | ~0（保留 perf_opt） | 不改；上一代工具链 绝对值不可跨版本引用 | — | — | P3 |
| 9 | **ClampMin** | 61.8–68.8%（0/7） | 与 ClampScalar 同 kernel；9.0 整体下移，tuned 仍优于 baseline | ~0（保留 perf_opt） | 同 ClampScalar，不改 | — | — | P3 |
| 10 | **Add** | 69.4–83.5%（10/11） | 访存受限，已贴实测上限（同族最高 84.7%） | 0–3% | 无新杠杆 | — | — | P3 |
| 11 | **Hardswish** | 50.2–61.0%（2/6） | 向量利用率 **0.94**，链已最短（无 fused vmad） | ~0 | 判据失配，非访存 | — | — | P3 |
| 12 | **Hardsigmoid** | 0.4–61.5%（可调 53.6–61.5%，0/2） | 向量 0.85，fp32 链 5 op 已语义最小（CG-2026-0006：fp16 无吞吐收益） | 0–5% | 无 | — | — | P3 |
| 13 | **Isnan** | 52.9–79.3%（1/5） | **MTE2 混合读写带宽地板**（mte2 0.81–0.90 ≫ vec 0.25–0.39）；4→3 pass 仅 −3.4% | ~0 | 需编译器补 `i1→i8`（CG-2026-0027） | 阻塞 | — | P3 |
| 14 | **Isinf** | 41.5–78.1%（1/5） | 同上（bf16 已用 `vand` 4-pass） | ~0 | 同上 | 阻塞 | — | P3 |
| 15 | **Isfinite** | 42.3–78.3%（1/5） | 同上 | ~0 | 同上 | 阻塞 | — | P3 |
| 16 | **GeluTanhAndMul** | 2.2–47.6%（0/4） | 向量 0.95；Vtanh 不可用，sigmoid 恒等折叠已最小 | ~5% | 换 tanh 近似实现（设计级） | 设计级 | 1 天 | P3 |
| 17 | **GeluAndMul** | 4.1–47.7%（0/4） | 向量 0.943；erf 阶数已 dtype 分化（本轮 −43.6%） | ~5% | 进一步降阶需精度冒险 | 设计级 | 1 天 | P3 |
| 18 | **GatedDeltaNetPrefill**<br>（19 接口之一，未完成） | 无 | blocksolve 卡 CG-2026-0003（Cube doubling + L0C→GM→L1 回喂） | 未知 | 先做「换算法形态能否编译」判定探针 | 设计级 | 探针 1–2h | P2 |


## 结论

**真正还有大空间的只剩 1 个**（P1）：

1. **FALN 设计层**：把两次 row-reduce 改 Cube GEMV，是唯一能突破 45% 的路，但要回设计阶段。

> Conv1d 通用 kernel（wav2vec2 17.7×、encodec-init 4.2×）与 Conv3d unet（21.3×）已在 2026-10-18 完成；剩余空间集中在 **Conv1d bias 中继**（受 CG-2026-0028 阻塞，1.5–3×）。

**中间档 3 个**（P1）：BatchNormBwd（epilogue/中转）、GLA（persistent 分核）、Lerp（linear-bias 两档）。

**其余 9 个基本到顶**：Add、Hardswish、Hardsigmoid、Isnan、Isinf、Isfinite、Gelu×2、ClampMin——继续投入收益 <5%，或需要编译器补 API。

## 建议的开工顺序

| 顺序 | 任务 | 预计 | 前置 |
|---|---|---|---|
| 1 | GLA persistent 分核（目标 10–20%） | 2–3h | 无 |
| 2 | FALN Cube 归约（目标 20–40%） | 1–2 天 | 需设计评审 |
| 3 | Conv1d bias 中继绕 CG-2026-0028（目标 1.5–3×） | 4–6h | 需新缺口绕法 |
