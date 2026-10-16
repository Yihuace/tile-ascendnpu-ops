# 逐算子横向对照（SOL 口径，最大 shape）

> 口径 = msprof Roofline SOL `Perf/min(AI×BW, 计算上限)`；按 AI 分瓶颈。数据取自各算子最大 shape 的 run.json。
> 达标线（任务书 §3.1）：访存受限 ≥70%；算力受限 ≥60%；混合类 ≥40%。

| 算子 | AI | 瓶颈 | SOL% | 带宽率% | 计算率% | 达标 |
|---|---:|:---:|---:|---:|---:|:---:|
| AddFwdOp | 0.125 | memory | 78.44 | 78.44 | 3.73 | ✅ |
| SubFwdOp | 0.125 | memory | 82.70 | 82.70 | 3.93 | ✅ |
| LerpFwdOp | 0.375 | memory | 80.84 | 80.84 | 11.52 | ✅ |
| ClampMinFwdOp | 0.828 | memory | 73.20 | 73.20 | 23.03 | ✅ |
| ClampScalarFwdOp | 1.1 | memory | 71.86 | 71.86 | 30.13 | ✅ |
| IsnanFwdOp | 1.64 | memory | 64.04 | 64.04 | 26.90 | ❌ |
| HardsigmoidFwdOp | 2.5 | memory | 55.74 | 55.74 | 52.93 | ❌ |
| IsinfFwdOp | 3.11 | memory | 46.75 | 46.75 | 42.53 | ❌ |
| IsfiniteFwdOp | 3.11 | memory | 46.00 | 46.00 | 41.86 | ❌ |
| HardswishFwdOp | 3.51 | compute | 61.61 | 46.14 | 61.61 | ✅ |
| GLADecodeOp | 3.73 | compute | 14.12 | 9.47 | 14.12 | ❌ |
| FusedAddLayerNormFwdOp | 3.78 | compute | 33.08 | 22.80 | 33.08 | ❌ |
| GeluTanhAndMulFwdOp | 3.9 | compute | 50.36 | 32.76 | 50.36 | ❌ |
| BatchNormBwdOp | 4.69 | compute | 29.98 | 14.06 | 29.98 | ❌ |
| GeluAndMulFwdOp | 14.2 | compute | 46.63 | 8.54 | 46.63 | ❌ |
| Conv1dFwdOp | 3.11e+03 | compute | 0.11 | 0.01 | 0.11 | ❌ |
| Conv1dBiasFwdOp | 3.11e+03 | compute | 0.11 | 0.01 | 0.11 | ❌ |
| Conv3dBiasFwdOp | 4.89e+03 | compute | 0.76 | 0.03 | 0.76 | ❌ |
