# 逐算子横向对照（SOL 口径，最大 shape，CANN 9.0.0）

> 口径 = msprof Roofline SOL `Perf/min(AI×BW, 计算上限)`；按 AI 分瓶颈。数据取自各算子最新 `run.json` 的最大 shape（2026-10-18 统一 9.0.0）。
> 达标线（任务书 §3.1）：访存受限 ≥70%；算力受限 ≥60%；混合类 ≥40%。
> `GatedDeltaNetPrefillFwdOp` 未交付（2/6 kernel），不参与横评。

| 算子 | AI | 瓶颈 | SOL% | 带宽率% | 计算率% | 达标 |
|---|---:|:---:|---:|---:|---:|:---:|
| AddFwdOp | 0.125 | memory | 83.48 | 83.48 | — | ✅ |
| SubFwdOp | 0.125 | memory | 84.00 | 84.00 | — | ✅ |
| LerpFwdOp | 0.375 | memory | 82.03 | 82.03 | — | ✅ |
| ClampMinFwdOp | 0.828 | memory | 68.66 | 68.66 | — | ❌ |
| ClampScalarFwdOp | 0.828 | memory | 68.66 | 68.66 | — | ❌ |
| IsnanFwdOp | 1.64 | memory | 62.64 | 62.64 | — | ❌ |
| HardsigmoidFwdOp | 3.00 | memory | 61.52 | 61.52 | — | ❌ |
| IsinfFwdOp | 1.74 | memory | 56.86 | 56.86 | — | ❌ |
| IsfiniteFwdOp | 1.74 | memory | 57.01 | 57.01 | — | ❌ |
| HardswishFwdOp | 3.51 | compute | 60.95 | 46.1 | 60.95 | ✅ |
| GLADecodeOp | 3.64 | compute | 18.24 | 9.5 | 18.24 | ❌ |
| FusedAddLayerNormFwdOp | 3.29 | compute | 33.42 | 22.8 | 33.42 | ❌ |
| GeluTanhAndMulFwdOp | 3.77 | compute | 47.63 | 32.8 | 47.63 | ❌ |
| BatchNormBwdOp | 2.34 | compute | 24.44 | 14.1 | 24.44 | ❌ |
| GeluAndMulFwdOp | 9.01 | compute | 47.71 | 8.5 | 47.71 | ❌ |
| Conv1dFwdOp | 1.41e+03 | compute | 0.24 | 0.01 | 0.24 | ❌ |
| Conv1dBiasFwdOp | 1.41e+03 | compute | 0.24 | 0.01 | 0.24 | ❌ |
| Conv3dBiasFwdOp | 4.89e+03 | compute | 0.76 | 0.03 | 0.76 | ❌ |

> 说明：
> - 访存受限算子的 SOL = GM Read+Write 带宽利用率；Add/Sub/Lerp 已 82–84%，Clamp 族在 9.0 下降到 68.7%。
> - 算力受限算子看 Vector/Cube 计算单元利用率；Hardswish 是唯一过 60% 线的算力型算子。
> - 8.5.2 历史横评（Add 78.44%、ClampScalar 71.86% 等）已不再引用；如需追溯见 `MIGRATION_INDEX.md` 表 B。
