# RETROSPECTIVE.md — `cos`(op 级)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见 `examples/cos/cos_fwd_kernel/RETROSPECTIVE.md`。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark `partial`,已明确披露)
- **attempts_used**:1(集成侧;上游 Stage 3 另有 1 次 `[DESIGN_ERROR]` 回退)
- **正确性**:**9/9 通过**(`run.json`:`correctness_passed = true`,`pass_rate = 1.0`)
- **benchmark**:8 通过 / **2 失败**(均为最大 workload,与前 5 个算子同型)

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **集成侧存在"已实测的精度分歧"时,流程没有强制的披露位点**:本算子与源算子在 $\lvert x\rvert\gtrsim500$ 处有量化分歧,但 `integration_log.md` 的模板字段里**没有"已知分歧"段** —— 本任务自行加了该段。若不加,分歧就会只存在于上游设计的 §10 里,集成侧记录看起来像"完全对齐" | 本任务:`integration_log.md` 的「本次集成的实质变更与已知分歧」段(含 4 档量化表);前序 5 个算子的 integration_log 均无此段(因为它们无此类分歧) | 在 `tilelang-op-integrator` 的 `integration_log.md` 模板里增加**条件段**「已知分歧(有则必填)」:凡上游设计 §10 存在"已在算子层无法消除、须契约层裁决"的风险项,集成日志必须复述其量化边界与对交付的影响判定 | R |

### Transferable Lessons（可迁移教训）

- **"集成的正确性通过"与"与源算子等价"是两件事**:本算子集成侧 `correctness 9/9` 通过,但它与源算子在 $\lvert x\rvert\gtrsim500$ 处有**实测量化的分歧**(704/8192 → 5930/8192 元素越界)。因为 manifest workload 在域内,集成判据看不到它。**集成日志必须显式记录这类"域外分歧"**,否则"9/9 通过"会被误读为全值域等价。
- **上游的 `[DESIGN_ERROR]` 回退应在集成日志的调试链里复述**:集成侧本身无调试,但交付件的形态是上游两次 attempt 的产物。集成日志应把该链写出,使"这个交付件是怎么来的"可被追溯。
- **同一 benchmark 失败在第 6 个算子上复现** —— `abs`/`ceil`/`floor`/`exp`/`cos` 五个算子的 `benchmark_failed` 恒为 2、失败标签恒为 `elementwise-256m-*`。**逐算子登记已累积 5 次,修法在 Stage 0 的 workload 规模规则里**。
- **`record_beyond_domain()` 这种"记录而非断言"的披露手法值得复用**:它在同一次运行里既给出本实现的越界数、又并列源算子的 `0`,使分歧**可见但不错位**为"测试失败"。

---

## Stage 4 性能优化复盘（2026-11-04）

### 有效杠杆（带实测数字）

| 杠杆 | 做法 | 数字 |
|---|---|---|
| 换掉贵指令 | `cos(x)=1−2sin²(x/2)`，按 π 归约让 sin² 吸收符号 ⇒ 无 `vsin`、无象限/符号分支 | 148.75 → 111.36 µs（−25%）；记账单元 26 → 20（`vsin` 一项记 9 个、周期成本 1215 cycles/tile） |
| **档位分档**（最大单项） | `block_size` 按 dtype：fp16/bf16 12288、fp32 8192 | 4096→12288 得 **+8.1 pp**（54.75 → 62.85） |
| fp32 双流 | 核内两 tile 交错、独立装载槽、19 步链交错 | fp32 16m 116.54 → 89.86 µs（−23%），49.53 → 64.31% |

### 被否决的杠杆（避免重复投入）

| 候选 | 结果 |
|---|---|
| deg-7 多项式填满空槽 | fp32 +17% 时间、fp16 无收益 |
| `T.Pipelined(num_stages=2)` | 与不加逐位相同（纯向量核在本后端拿不到重叠） |
| fp32 UB→UB 加粗 copy | 45.88%（更差） |
| fp16/bf16 `block_size` 16384 | `ub overflow`（12 B/元素 × 16384 > 192 KB） |
| fp32 `block_size` 12288 | `ub overflow, requires 2359296 bits`（双流实需 **24 B/元素**） |
| 3 缓冲瘦身（16384） | 16m/64m +1 pp，但 **256m bf16 回退 −8.11 pp** ⇒ 整体否决 |

### 可迁移教训

- **先判分支再挑杠杆**：本算子属"记账/算力"分支（`AI×1.8 > Computility`），所以「换贵指令」有效而「减少向量遍历」无效；
  反之在访存分支（如 L1/L2 归约）里减少向量遍历是纯收益、堆 fops 完全无用。
- **结构化记录必须能区分口径**：`perf_records.jsonl` 用 `phase` 区分「直跑（调优口径）」与「official（上报口径）」，
  否则会出现两套数字互相打架。本次实测：同一交付件 fp16 官方三次采样 61.64 / 66.10 / 56.90，**波动 ±10%**，
  单次采样不足以判定 1–10 pp 的差异。
- **测速必须与精度校验同一次运行**：本轮曾在未加 `--check` 的情况下把一条输出 `inf` 的路径测成"变快 25%"（在 L1/L2 那次复现），
  正确做法是 `msprof ... python bench.py ... --check` 一起跑。
- **相邻 tile 的双流映射存在性能双态**：相邻映射（2p / 2p+1）实测在 90 µs 与 108 µs 两档之间跳（fops 相同、周期差 21%），
  改为"两流相距半个 tile 域"后稳定；交付件采用后者。
