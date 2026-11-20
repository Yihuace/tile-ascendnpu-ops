# 设计期硬件成本模型与 roofline 估算标准（D-2）⭐

> **单一事实源**：本文件定义**设计期定量性能估算协议**（何时算、怎么算、算完落哪）。硬件常数**数值表**的唯一事实源是 `tilelang-op-optimize` skill 的 [pattern-library/constants.md](../../tilelang-op-optimize/references/pattern-library/constants.md)（条目带版本戳与来源条目 ID，evolver 蒸馏新 D 类常数时追加进该表，Tier 0）——本文件不复制数值，只定义使用协议；两文件同步演进。
>
> **背景**：实测硬件常数（L1 端口带宽、发射开销、UB/L1/L0C 容量、fabric 带宽）此前只在 Stage 4 被动获得——attention 第三轮证明设计期结构认知错误的代价是两轮调优 + 一次错误 `[DESIGN_LIMIT]`（VP-2026-0042 证据链）。设计期 roofline 把这类判断前移到 Stage 1。

## 1. 估算协议（何时算）

| 触发位置 | 动作 |
|---------|------|
| Phase R R4（硬件亲和性逐候选评估） | 对 R3 存活候选计算**估算下界**，写入 §1.6.0 R4 列；违反容量/带宽量级的候选在此淘汰（淘汰理由引用具体常数条目 ID） |
| Phase 2 §1.6.3（布局决策逐候选评分） | 主选与备选的取舍分歧涉及搬运/发射代价时，用常数表量化（引用条目 ID）；判定裕度仍落在未实证常数内 → 走实验裁决模式（不纸面拍板） |
| Phase 4 §5（Tiling/分核） | 容量约束复算：UB/L1/L0C 驻留预算对照 `CONST-capacity-910B2C`（含 auto-multi-buffer ~1.7x 膨胀系数） |

## 2. 估算公式（怎么算）

**估算下界 = max(三项)**，全部按 dispatch 代表 workload 计算：

1. **流量项** = GM 读写总字节 ÷ 有效带宽（MTE2 曲线按总流量档位取值，`CONST-mte2-degradation`；混合读写流量禁用峰值参考值——3:1 R/W 地板见该条目）；
2. **发射项** = 向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`；Cube 段另按 L1 端口带宽折算操作数流地板，`CONST-L1-port-bw`）；
3. **容量项** = 容量约束决定的 tile 循环数 × 每轮跨引擎往返代价（`CONST-store-fixpipe-gm-only`：S/P 类中间矩阵无法直写 L1/UB，跨引擎传输强制 GM 往返）。

计算规则：
- 每项引用**常数条目 ID**（如 `CONST-L1-port-bw`）+ 该条目版本戳；常数缺失时显式标注「未实测假设 + 估算依据」，不得冒充实测值；
- 估算下界落入 DESIGN.md **§1.6.0 末尾的「设计期估算」行**（格式：`估算下界: <值> us（流量 <x> + 发射 <y> + 容量 <z>，引用 CONST-*）`）——轻量调研（单步逐元素/纯搬运类）可豁免容量项，但流量项必算（elementwise 主导项）；
- 硬件亲和性淘汰与 §1.6.3 弃选理由引用常数时**先查 constants.md**——库内已有的实测数字不得当作"未实证常数"重新假设（与 algorithm-research.md §3 R4 同口径）。

## 3. Stage 4 回填（实测 vs 估算对账）

1. optimizer Phase 1 baseline 采集完成后，在 `perf_opt/opt_log.md` Baseline 段追加**「设计估算 vs 实测」偏差行**：DESIGN.md §1.6.0 估算下界 vs baseline 实测 Task Duration，偏差 >2x 时注明偏差项（流量/发射/容量哪项失准）；
2. 该偏差行成为 `[DESIGN_LIMIT]` 归因的量化证据（perf-feedback.md §1 触发条件①的支撑材料——设计期估算系统性乐观/悲观是设计假设被实测推翻的直接形态）；
3. 调优中发现**新的硬件常数**（如新的端口带宽/发射代价/容量上限）→ Phase 4 回写 pattern-library/constants.md 新条目（Tier 0，含三件套 + repro，ED-C 责任前移），并在偏差行注明"估算失准根源 = 该常数设计期未知"。

## 4. 与现有机制的衔接

- **负向断言举证**（[negative-claim-evidence.md](negative-claim-evidence.md)）：以"违反容量/带宽量级"淘汰候选的论断，其证据 = constants.md 条目（含版本戳）；constants.md 未覆盖的量仍按「未文档化假设 + 估算依据」标注；
- **实验裁决模式**：roofline 估算不能替代实验裁决——判定裕度落在未实证常数的不确定区间内时仍须产出主选+备选+裁决计划三件套（设计 skill Phase 2 第 3 项）；
- **版本失效**：tilelang 重编译 / CANN 升级后，constants.md 相关条目自动待重验（`kb_stale_check.py` 检测，repro 套件重验）——设计期引用待重验常数须在估算行标注。
