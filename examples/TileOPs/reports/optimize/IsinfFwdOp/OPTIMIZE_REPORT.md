# 优化报告：IsinfFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsinfFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = isinf(x)`，bool 经 int8 字节存储 |
| kernel（基准，只读） | `tileops/kernels/elementwise/isinf/isinf_kernel/` |
| 调优产物 | `.../isinf_kernel/perf_opt/IsinfFwdOp.py`（final 候选 `bf16vand-bs8192`，`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ GM Read+Write SOL ratio；数据来自 `perf_opt/perf_records.jsonl` round 5（含 round-5 candidate / cores 子轮） |
| **工具链版本戳** | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-17 → 2026-10-18（二次深挖）；bishengir 与 hivmc 版本原文未记录（opt_log 记 bishengir 1.1.0 / hivmc 0.2.0） |
| 达标线 | 访存受限 **≥70%**（AI 与归类依据原文未记录；opt_log 记 AI ≈ 2.0 ⇒ 访存受限） |
| benchmark | `benchmarks/ops/bench_isinf.py::test_isinf_bench`（7 tune + 3 smoke） |
| 过程日志 | `.../isinf_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`logs/`、`profiles/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**round 5 的 bf16 位运算 `|x| = vand(x, 0x7FFF)`（6→4 pass）把 bf16 从向量受限翻转到接近带宽地板，带来 −15~−20% 的单点跃迁；加上 fp16/fp32 的 vabs 4-pass 链，7 个 tune workload 全部正收益、无回退，建议采纳 wrapper。**
- **达标数（双分母）**：可调 shape（7 tune）**1/7 ≥70%**（16m fp32 74.72%）；全 shape（含 3 个 `smoke-1m`，仅精度不参与性能判定）性能判定口径同为 **1/7**；豁免 **3**（`smoke-1m`，依据：仅精度回归、不参与性能判定，原文明示）。
- 关键收益（vs 基线）：16m bf16 **−19.40%**、16m-half bf16 **−19.88%**、256m bf16 **−15.42%**（原文一位小数口径 −19.4% / −19.9% / −15.4%；round-4 的 bf16 保持 6-pass 时只有 −0.2~−0.9%，本轮为质的跃迁）；fp16 **−6.60~−13.72%**、fp32 **−2.43%**。
- 指标行为：bf16 `aiv_vec_ratio` **0.72–0.76 → 0.55–0.62**（≈ `aiv_mte2_ratio` 0.46–0.53），由向量受限翻转为收敛均衡；16m/256m fp16 仍 56–63%、bf16 41.4–54.5%，剩余 gap 到 70% 是 **MTE2 混合 R/W 带宽地板**（fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90）。
- 收益结构（vs 基线）：**bf16 −15.4~−19.9%**（round-5 新杠杆）、**fp16 −6.6~−13.7%**、**fp32 −2.4%**（round-4 vabs 链）——7 个 tune workload 全部正收益、无回退。
- **是否建议采纳 wrapper**：**是**（原文『二次深挖』采纳建议）——7 个 tune workload 全部正收益、无回退；此前 round-4「不建议采纳」的主要障碍（bf16 收益 ~0）已被 vand 4-pass 消除。是否最终翻转由 conductor 在回归通过后执行，本阶段不自行翻转 wrapper。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 基线向量链长度 | 6 条 pass（`vcmp(eq +inf)` + `vcmp(eq -inf)` + `vcast` + `vcast` + `vadd` + `vcast`） | 正负 inf 各判一次 → 有合并空间 |
| fp16/fp32 瓶颈 | `aiv_mte2_ratio` **0.81–0.90** ≫ `aiv_vec_ratio` 0.25–0.39 | **MTE2 带宽地板**为主瓶颈；减 pass 只能拿小头（fp32 −2.43%） |
| bf16 瓶颈（round 4 前） | `aiv_vec_ratio` **0.757 / 0.725** ＞ `aiv_mte2_ratio` **0.359 / 0.395** | **bf16 是 predicate 族唯一的「向量受限」例外**（`vec_t > mte2_t`）→ 减 pass 对它收益最大 |
| bf16 减 pass 后 | vec 0.622 / 0.546 vs mte2 0.458 / 0.534 | 翻转成收敛均衡，接近带宽地板 → 无进一步向量余量 |
| 核数 | 40 核 45.811µs / 774.206µs，20 核 85.132µs（+85.9%）/ 1351.637µs（+74.6%）（16m / 256m bf16） | 搬运并行主导，40 核已最优 |
| 判据是否失配 | roofline GM SOL 与 pipe 占比不一致（bf16「向量受限」但 GM SOL 41.4–54.5%） | **是**：按 pipe 占比判瓶颈（vec 受限 vs mte2 地板），不按 GM SOL 单一判据 |
| 噪声底 | 原文未记录显式阈值；同族 Isnan 记 ±3% 双态噪声，本算子 fp16/fp32 round 5 相对 round 4 的 Δ 归为机器噪声 | 单批小幅 Δ 不可分辨，故以 bf16 的 −15~−20% 作为裁决依据 |

### 2.2 方法（做了什么、为什么这样做）

仅在 `perf_opt/` 内改动（基准与 wrapper 未动）：

1. **结构层：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核**
   - 改了什么：热循环用编译期满块 extent、尾块单独处理；tile 循环 2-stage 流水；按 40 AIV 铺满。
   - 为什么有效：省掉循环内动态 extent 的标量开销，用下一 tile 的 MTE2 搬运覆盖当前 VEC 链；40 核是搬运并行上限（20 核实测严重回退）。
   - 正确性依据：满块 + 尾块覆盖全部元素，语义与基线一致；round 1 起 L0 全 PASS。

2. **fp16/fp32：`isinf(x) = (|x| == +inf)` 的 vabs 4-pass 链**（round 2/4 winner）
   - 改了什么：`vabs → vcmp(eq) → vcast(bool→f16) → vcast(f16→i8)`，把「正 inf / 负 inf 两次比较」合并为一次 → **6 pass → 4 pass**。
   - 为什么有效：`|+inf| = |−inf| = +inf`，一条 vabs 消掉一整套 `vcmp+vcast+vadd`；fp16/fp32 有 `T.vabs` 的 dtype 边可直接用。
   - 依据：数学等价（绝对值把 ±inf 折叠为单点）；`--level all` max_diff = 0；fp16 −6.60~−13.72%、fp32 −2.43%（后者受 MTE2 地板限制）。

3. **bf16：`|x| = vand(x, 0x7FFF)` 位运算符号位清除（6→4 pass）**（round 5 winner）
   - 改了什么：bf16 路径用 `vand(x, 0x7FFF)` 掩掉符号位得到 `|x|`，1 条 op 替代 `T.vabs`（`T.vabs` 无 bf16 边）；0x7FFF 掩码由 `T.fill(-0.0) → T.vnot` 一次性构造于**循环外**（`~0x8000 = 0x7FFF`）。
   - 为什么有效：bf16 原路径只能保留「两条 mask 合并」的 6-pass 链且处于**向量受限**（vec 0.72–0.76 ≫ mte2 0.36–0.40）；减 2 条 pass 后矢量占比降到 0.55–0.62，与 mte2 0.46–0.53 均衡 → 收益 −19.98% / −19.64% / −15.13%（bf16 三个 workload）。
   - 正确性依据：`T.vand` / `T.vnot` 支持 bf16 按位级 reinterpret（`docs/Tilelang.language/逻辑操作/` 2.2.1）；bit 级掩码保留 NaN mantissa，`vcmp eq` 的 ordered 比较正确排除 NaN；`--level all` max_diff = 0，且 bf16 −inf→+inf（仍判 False）、−0（仍判 True）、NaN（仍判 False）逐位验证通过。
   - 附带结论：6→4（删 2 pass）比 Isnan 的 4→3（删 1 pass）收益更大，与「bf16 是唯一向量受限例外」的 pipe 证据一致。

4. **`TUNED_BLOCK_SIZE = 8192`**：与同族 Isnan 保持一致，供 wrapper 成对引用。

**被否决的方向（保留）**：
- bf16 用 `vselect` 合并两条 mask：实测 **更慢（+6.6~8.6%，opt_log Round 2）**，已回退为基线 6-pass 链。
- 20 核、其他 tiling：见 §2.3 与 §7，实测严重回退。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | baseline（6-pass） | 静态尾块 + `T.Pipelined(2)` + 40 核 | 7 个 tune workload baseline（opt_log §1） | 参照 |
| R2 | `vabs` 4-pass（fp16/fp32） | `isinf(x) = (\|x\| == +inf)`，6 pass → 4 pass | fp16 −6.6~−13.7%、fp32 −2.4%（见 §1） | ✅ 采纳 |
| R2 | bf16 `vselect` 合并两条 mask | 两条 mask 合并 | **+6.6~8.6%（更慢，opt_log R2）** | ❌ 否决：回退为 6-pass 基线链 |
| R3 | bf16 回退验证 | bf16 保持 6-pass | −0.2~−0.9%（中性） | ❌ 否决：无收益（bf16 保留 6-pass 形态） |
| R4 | 合并 winner（`vabs-bs8192`） | fp16/fp32 4-pass + bf16 6-pass + 结构层 | 合并进 `perf_opt/IsinfFwdOp.py` | ✅ current best |
| R5 | `bf16vand-bs8192` | bf16 `vand(x,0x7FFF)` 位运算 \|x\|，6→4 pass；掩码循环外构造 | 16m bf16 −19.40%、16m-half bf16 −19.88%、256m bf16 −15.42% | ✅ 采纳为 final |
| R5 | 核数复核 40 / 20 | `num_cores` 40→20 | 40 核 45.811 / 774.206µs vs 20 核 85.132 / 1351.637µs（16m / 256m bf16；+85.9% / +74.6%） | ❌ 否决：40 核已最优 |

**合并与复测纪律**：fp16/fp32 代码路径在 round 5 字节一致、无回退（Δ 为机器噪声）；round-4 的 7 条旧 `phase=final` 记录已按「被取代即降级」纪律改为 `merged`（数据未改），最终 7 条 final 记录来自同一候选 `bf16vand-bs8192`。

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 46.001 | **-19.40%** | 原文未记录 | 41.36 | ❌ |
| 16m-float16 | 38.561 | 33.271 | -13.72% | 原文未记录 | 56.00 | ❌ |
| 16m-float32 | 52.761 | 51.481 | -2.43% | 原文未记录 | 74.72 | ✅ |
| 16m-half-bfloat16 | 57.251 | 45.871 | **-19.88%** | 原文未记录 | 41.92 | ❌ |
| 16m-half-float16 | 37.951 | 33.731 | -11.12% | 原文未记录 | 57.30 | ❌ |
| 256m-bfloat16 | 917.428 | 775.936 | **-15.42%** | 原文未记录 | 54.46 | ❌ |
| 256m-float16 | 709.204 | 662.403 | -6.60% | 原文未记录 | 63.01 | ❌ |

smoke：3 个 `smoke-1m` 仅精度回归 PASS，不参与性能判定。

**测量口径与噪声**：baseline = round-4 6-pass 基线、final = round-5 全 7 workload 同一候选，相邻交错各 1 次独立 `msprof op`。base SOL% 原文表未列（仅列 opt SOL%），故标「原文未记录」；round-5 候选 vs current best 的相邻交错对照见 §7（bf16 三个 workload）。

## 4. 精度

- `python perf_opt/IsinfFwdOp.py --level all` → **`All check passed!`**，max_diff = 0（NaN/±inf/−0/子正规 全覆盖）。
- vand 位运算的语义正确性已由 L0 special 输入逐位验证：若 0x7FFF 掩码构造错误，−inf 会误判为 False，L0 必失败——已通过。
- 3 个 `smoke-1m` 仅精度回归 PASS。
- 关键边界：NaN（判 False）、±inf（判 True）、−0（判 False）、子正规（判 False）全覆盖，max_diff = 0。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 21:05 → 21:35（round 1–4）；2026-10-18 02:57 → 03:15（round 5 二次深挖） |
| 墙钟时长 | 二次深挖 ≈ 18 min；round 1–4 墙钟原文未记录 |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../isinf_kernel/perf_opt/IsinfFwdOp.py`（`bf16vand-bs8192`，sha256 `783957dc…`） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only；round-4 旧 final 7 行降级 `merged`，数据未改）、`logs/`、`profiles/`、`candidates/` |
| round-5 备份 | `perf_opt/IsinfFwdOp_round5_bf16vand.py` |
| 能力缺口 | `.agents/evolution/capability-gaps.md` **CG-2026-0027** |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（建议采纳，由 conductor 执行） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | PL-1.23（bf16 不再「6-pass 死路」，有 vand 位运算 4-pass 路径） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：3 个 `smoke-1m`（仅精度回归、不参与性能判定，不计入达标分母）；无其他豁免。
- **候选 vs current best 的独立对照（bf16，相邻交错，msprof op）**——用于裁决 vand 是否真收益：

  | workload | current best µs | vand µs | Δ | 判定 |
  |---|---:|---:|---:|---|
  | 16m-bfloat16 | 57.001 | 45.611 | -19.98% | winner |
  | 16m-half-bfloat16 | 56.971 | 45.781 | -19.64% | winner |
  | 256m-bfloat16 | 912.198 | 774.135 | -15.13% | winner |

- **ceiling 证据（pipe 级归因，msprof PipeUtilization）**：

  | workload | 变体 | aiv_vec_ratio | aiv_mte2_ratio |
  |---|---|---:|---:|
  | 16m-bfloat16 | 6-pass baseline | **0.757** | 0.359 |
  | 16m-bfloat16 | vand 4-pass | 0.622 | 0.458 |
  | 256m-bfloat16 | 6-pass baseline | **0.725** | 0.395 |
  | 256m-bfloat16 | vand 4-pass | 0.546 | 0.534 |

  - bf16 由向量受限（`aiv_vec_ratio` 0.72–0.76 ≫ `aiv_mte2_ratio` 0.36–0.40）翻转为收敛均衡（0.55–0.62 vs 0.46–0.53）。
  - fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90（MTE2 地板），4-pass 向量链隐藏于搬运窗口下 → 剩余 56–63% / 74.72% 由带宽地板决定。
- **核数复核**：

  | num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
  |---:|---:|---:|---|
  | 40（winner） | **45.811** | **774.206** | 已最优 |
  | 20 | 85.132（+85.9%） | 1351.637（+74.6%） | 严重回退 |

  - 20 核印证「减半搬运并行 ≈ 减半有效带宽」，与 MTE2-bound 归因一致。
- **工具链约束**：`CG-2026-0027`（无 i1→i8 直连 / 无 bool vnot / `vselect` Out 不支持 int8）使 isinf 无法做到 3-pass；4-pass 已是本工具链下限。
- **贴线风险**：16m fp32 74.72% 为表中唯一达标项（高于线）；256m fp16 63.01%、16m fp16 56.00%、16m-half fp16 57.30% 均明显在线下，不属 <3pp 贴线项。bf16 三档 41.36–54.46% 为结构性 gap。
- **缺口**：`CG-2026-0027`（predicate 族 ISA 约束）；bf16 的 4-pass 下限已达成，无进一步可解项。
- **`[DESIGN_LIMIT]`**：不触发（无 >2× 结构加速、无设计假设矛盾），不产出 `perf_feedback.md`。

## 8. 建议

1. **建议采纳 `perf_opt/IsinfFwdOp.py`**（翻转 wrapper 的 baseline/perf_opt 双 import 切换块）：7 个 tune workload 全部正收益、无回退，精度无回归，签名 drop-in。
2. 后续动作：若工具链升级，优先复核（a）bf16 `vand` 路径的 bit 级语义（−inf/−0/NaN）与（b）`CG-2026-0027` 是否补齐（补齐后可评估 3-pass，但按 §7 地板归因 SOL 提升有限）。
3. **未解决项与原因**：fp16/bf16 到 70% 线的 gap 是 MTE2 混合 R/W 带宽地板，pass 数与 tiling 不可解；bf16 已从「6-pass 死路」走到 4-pass 下限。
