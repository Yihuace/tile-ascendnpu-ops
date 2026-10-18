# OPTIMIZE_REPORT 统一模板（Stage-4 单算子优化报告）

> 适用：`examples/TileOPs/reports/optimize/{Op}/OPTIMIZE_REPORT.md`
> 规则：**只做结构与节名归一 + 补缺失节**；**不得新增、修改、删除任何测量数字与结论**；
> 原文没有的信息写「原文未记录」，不得编造。旧文件先备份到 `history_version/OPTIMIZE_REPORT_v0.md`。

## 固定结构（8 节，顺序不可变）

```markdown
# 优化报告：{Op}（Stage-4 optimize）

## 0. 身份
| 项 | 值 |
|---|---|
| 算子 | {Op}；{单/多 kernel}；{Vector/Cube/MixCV}；语义公式 |
| kernel（基准，只读） | 路径 |
| 调优产物 | perf_opt 路径 |
| 口径 | msprof op Task Duration + SOL ratio（写清 kernel-name 与数据来源） |
| **工具链版本戳** | tilelang commit / CANN 版本 + bishengir + hivmc / 设备 / 日期 |
| 达标线 | {访存 70% / 算力 60% / Mix 40%}（写 AI 与归类依据） |
| benchmark | 文件::函数（tune + smoke 数） |
| 过程日志 | opt_log.md / perf_records.jsonl / logs / candidates |
| 前置报告 | 本文件 v0 归档路径（如适用） |

## 1. 结论
- 一句话结论
- **达标数**：全 shape x/y、可调 shape a/b、豁免 n（写清豁免依据）
- 关键收益（逐 workload 或代表值，带 µs 与 %）
- **是否建议采纳 wrapper** + 理由（净收益/回退/噪声）

## 2. 优化方法与过程
### 2.1 诊断（决定用什么杠杆）
表格：观测 | 数值 | 结论（AI、瓶颈 pipe 占比、固定开销、噪声底、判据是否失配）

### 2.2 方法（做了什么、为什么这样做）
编号列出实际落地的改动，每条写清：
- 改了什么（代码层面）
- 为什么有效（访存/计算/流水/语义依据）
- 合法性或正确性依据（如位等价、精度容差、API 文档）

### 2.3 逐轮记录
| 轮次 | 候选 | 改动 | 结果 | 处置（✅采纳 / ❌否决+原因 / 噪声） |
保留被否决的方向与原因；区分「测试通过但无收益」与「测试失败」。

## 3. 优化结果（msprof 对比表）
| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
相邻交错/独立进程等测量口径说明写在表下方；噪声裁定单列。

## 4. 精度
`--level all` 结果（L0/L1/L2/Boundary）+ 关键边界用例（NaN/±inf/−0/subnormal）+ max_diff 依据。

## 5. 时间与成本
起止时间 / 墙钟 / token（input/output/reasoning）/ cost。

## 6. 产物
final kernel / 调优记录 / gate 4 状态 / 算子状态 / wrapper 切换 / 端到端 report 路径 / pattern-library 回写。

## 7. 天花板 / 豁免 / 缺口
- 豁免项与依据（微型 shape / 判据失配 / 结构上限）
- ceiling 证据（roofline advice、pipe ratio、CG 编号）
- 贴线风险（距达标线 <3pp 的 shape）

## 8. 建议
采纳建议；后续动作（如升级工具链需复核哪些 shape）；未解决项与原因。
```

## 写作要求

1. **方法必须可复现**：读到「方法」节的人应当能在不改代码的情况下理解每一步为什么这么做。
2. **数字必须可追溯**：每个关键数字要么在本表内，要么指向 `perf_records.jsonl` 的 round/行。
3. **评估必须双分母**：全 shape 与可调 shape 分开；豁免项单列，不计入达标分母。
4. **口径必须标版本**：跨 CANN 版本的数据不得混排；旧版本数据单独标注。
5. **否定结论要保留**：被否决的候选、失败的尝试、证伪的假设都留在「逐轮记录」里。
