# BitwiseNot Stage 4 — 进度报告（perf_opt/progress_report.md）

口径: `msprof op` Task Duration(us)，median of 20 launches。噪声阈值 3%。`-` 表示该轮未测该 workload。

| round | candidate | 16m-int32 | 16m-int64 | 64m-int32 | 256m-int32 | 超噪声阈值 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| 0 | baseline | 93.62 | 183.46 | 484.17 | 2058.50 | — | 基准 |
| 1 | r1_bs4096 | 72.78 | 171.52 | 417.17 | 1803.76 | 是 | gain |
| 1 | r1_bs8192 | 64.38 | - | 393.17 | 1713.07 | 是 | winner(int32) |
| 1 | r1_bs3072 | - | 174.30 | - | - | 是(vs base) | gain |
| 1 | r1_bs6144 | - | - | 398.25 | - | 是(vs base) | gain |
| 2 | r2_2buf | 68.64 | 170.28 | 391.99 | 1710.61 | 否（相对 best 无有效增益） | int64 平区/ int32 回退 |
| 3 | r3_staticbulk | 63.48 | 171.10 | 389.35 | 1711.85 | 是 | winner(int32 小/中) |
| 4 | r4_nc20 | - | - | - | 1866.38 | 否 | 回退 |
| 4 | r4_pipe2 | - | - | - | 1717.21 | 否 | 平区 |
| 4 | r4_contig | - | - | - | 1669.39 | 是 | winner(256m) |
| merge | r4_merged | 63.72 | 171.14 | 384.59 | 1655.87 | 是 | 合并候选 |
| final | r4_merged (final file) | 63.94 | 171.60 | 382.23 | 1657.27 | 是 | 最终 |

最终 vs baseline 提升: 16m-int32 −31.70% / 16m-int64 −6.46% / 64m-int32 −21.06% / 256m-int32 −19.49%；全部 > 3% 噪声阈值。

收敛状态: 无 plateau（4 轮均产出 valid 提升）；无 `[DESIGN_LIMIT]`；stop_reason = success。
