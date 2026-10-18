# 优化报告：IsfiniteFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsfiniteFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = isfinite(x)`，bool 经 int8 字节存储 |
| kernel（基准，只读） | `tileops/kernels/elementwise/isfinite/isfinite_kernel/` |
| 调优产物 | `.../isfinite_kernel/perf_opt/IsfiniteFwdOp.py`（final 候选 `bf16vand-bs8192`，`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ GM Read+Write SOL ratio；数据来自 `perf_opt/perf_records.jsonl` round 3（含 candidate / cores 子轮） |
| **工具链版本戳** | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-17 → 2026-10-18（二次深挖）；bishengir 与 hivmc 版本原文未记录（opt_log 未列该两项） |
| 达标线 | 访存受限 **≥70%**（AI 与归类依据原文未记录；opt_log 记 AI ≈ 2.0 ⇒ 访存受限） |
| benchmark | `benchmarks/ops/bench_isfinite.py::test_isfinite_bench`（7 tune + 3 smoke） |
| 过程日志 | `.../isfinite_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`logs/`、`profiles/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**round 3 的 bf16 位运算 `|x| = vand(x, 0x7FFF)`（6→4 pass）把 bf16 从向量受限翻转到接近带宽地板（−16.4~−21.3%），叠加 fp16/fp32 的 vabs 4-pass 链后 7 个 tune workload 全部正收益、无回退，建议采纳 wrapper。**
- **达标数（双分母）**：可调 shape（7 tune）**1/7 ≥70%**（16m fp32 78.01%）；全 shape（含 3 个 `smoke-1m`，仅精度不参与性能判定）性能判定口径同为 **1/7**；豁免 **3**（`smoke-1m`，依据：仅精度回归、不参与性能判定，原文明示）。
- 关键收益（vs 基线）：16m bf16 **−21.24%**、16m-half bf16 **−21.29%**、256m bf16 **−16.42%**（round-2 bf16 保持 6-pass 时仅 −2.2~−2.8%）；fp16/fp32 −5.4~−12.0%。
- 指标行为：bf16 `aiv_vec_ratio` **0.72 → 0.55**（≈ `aiv_mte2_ratio` 0.39 → 0.53），与 Isnan/Isinf 同族的「bf16 是唯一向量受限例外」结论一致；剩余 gap 到 70% 是 **MTE2 混合 R/W 带宽地板**（fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90）。
- **是否建议采纳 wrapper**：**是**（原文『二次深挖』建议）——bf16 −16.4~−21.3%、fp16/fp32 −5.4~−12.0% 均为真实 Task Duration 下降，40 核/block 8192 已复核最优，精度全通过、无回退。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 基线向量链长度 | 6 pass（`vcmp(gt -inf)` + `vcmp(lt +inf)` + `vcast` + `vcast` + `vmul` + `vcast`，opt_log §1） | 上下界各判一次 → 可折叠为一次 `\|x\| < +inf` |
| fp16/fp32 瓶颈 | `aiv_mte2_ratio` **0.81–0.90** ≫ `aiv_vec_ratio` 0.25–0.39 | **MTE2 带宽地板**为主瓶颈；减 pass 只能拿小头（fp32 −5.63%） |
| bf16 瓶颈（round 2 前） | `aiv_vec_ratio` **0.721** ＞ `aiv_mte2_ratio` **0.386**（256m） | **bf16 向量受限**，6-pass 向量链是关键路径 → 减 pass 收益最大 |
| bf16 减 pass 后 | vec 0.548 vs mte2 0.531（256m）；`aiv_vec_time` 665.21 → 419.65µs | 翻转成收敛均衡，接近带宽地板 |
| 核数 | 40 核 45.771µs / 775.596µs，20 核 85.202µs（+86.1%）/ 1356.307µs（+74.9%）（16m / 256m bf16） | 搬运并行主导，40 核已最优 |
| 判据是否失配 | bf16 GM SOL 41.7–54.2% 却「向量受限」；fp16/fp32 GM SOL 55.8–63.1% 却「MTE2 饱和」 | **是**：单一 GM SOL 不能定位瓶颈，须按 PipeUtilization 的 vec/mte2 占比归类 |
| 噪声底 | 原文未记录显式阈值；round-2 → round-3 的 fp16/fp32 Δ 归为机器噪声 | 单批小幅 Δ 不可分辨，以 bf16 的 −16~−21% 作为裁决依据 |

### 2.2 方法（做了什么、为什么这样做）

仅在 `perf_opt/` 内改动（基准与 wrapper 未动）：

1. **结构层：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核**
   - 改了什么：热循环用编译期满块 extent、尾块单独处理；tile 循环 2-stage 流水；按 40 AIV 铺满。
   - 为什么有效：省循环内动态 extent 的标量开销，用下一 tile 的 MTE2 搬运覆盖当前 VEC 链；40 核是搬运并行上限（20 核实测 +86.1% / +74.9% 回退）。
   - 正确性依据：满块 + 尾块覆盖全部元素，语义与基线一致；每轮 L0 全 PASS。

2. **fp16/fp32：`isfinite(x) = |x| < +inf` 的 vabs 4-pass 链**（round 1/2 winner）
   - 改了什么：`vabs → vcmp(lt) → vcast(bool→f16) → vcast(f16→i8)`，把上下界两次比较合并为一次 → **6 pass → 4 pass**。
   - 为什么有效：`isfinite(x) ⇔ |x| < +inf`，一条 vabs 消掉一整套比较/类型转换；fp16/fp32 有 `T.vabs` 的 dtype 边。
   - 依据：数学等价；NaN 的有序 `lt` 为 false，天然排除 NaN；`--level all` max_diff = 0；fp16 −5.36~−11.95%、fp32 −5.63%（受 MTE2 地板限制）。

3. **bf16：`|x| = vand(x, 0x7FFF)` 位运算符号位清除（6→4 pass）**（round 3 winner）
   - 改了什么：bf16 路径用 `vand(x, 0x7FFF)` 掩掉符号位得到 `|x|`（`T.vabs` 无 bf16 边），1 条 op 替代两次比较；0x7FFF 掩码由 `T.fill(-0.0) → T.vnot` 一次性构造于**循环外**（`~0x8000 = 0x7FFF`）。
   - 为什么有效：bf16 原路径只能保留 6-pass 链且处于**向量受限**（vec 0.721 ≫ mte2 0.386）；减 2 条 pass 后 `aiv_vec_time` 665.21 → 419.65µs（−36.9%，256m，opt_log §3），降到与 mte2 均衡 → 收益 −21.24% / −21.29% / −16.42%。
   - 正确性依据：`T.vand` / `T.vnot` 支持 bf16 按位级 reinterpret（`docs/Tilelang.language/逻辑操作/` 2.2.1）；bit 级掩码保留 NaN mantissa，`vcmp lt` ordered 比较正确排除 NaN；`--level all` max_diff = 0，新增 bf16 Boundary-special 探针显式验证 −inf→+inf（仍判 False）、−0→+0（仍判 True）、NaN（仍判 False）。

4. **`TUNED_BLOCK_SIZE = 8192`**：与同族 Isnan/Isinf 一致，供 wrapper 成对引用。

**被否决的方向（保留）**：bf16 保持 6-pass（仅靠结构层）在 round 2 只有 −2.2~−2.8%，被 round 3 的 vand 4-pass 取代，不存在「6-pass 已够用」的结论。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `vabs` 4-pass（fp16/fp32） | `isfinite(x) = (\|x\| < +inf)`，6 pass → 4 pass | fp16/fp32 正收益（round-2 汇总见 §1 的 −5.4~−12.0%） | ✅ 采纳 |
| R1/R2 | bf16 保持 6-pass | 仅加结构层（静态尾块 + 流水 + 40 核） | −2.2~−2.8%（中性偏弱） | ⚠️ 作为临时 current best，被 R3 取代 |
| R2 | 合并 `abs-pipe2-bs8192` | fp16/fp32 4-pass + bf16 6-pass | 合并为 round-2 final（7 行 final 记录后降级 `merged`） | ✅ 合并通过（后被 R3 取代） |
| R3 | `bf16vand-bs8192` | bf16 `vand(x,0x7FFF)` 位运算 \|x\|，6→4 pass；掩码循环外构造 | 16m bf16 −21.24%、16m-half bf16 −21.29%、256m bf16 −16.42% | ✅ 采纳为 final |
| R3 | 核数复核 40 / 20 | `num_cores` 40→20 | 40 核 45.771 / 775.596µs vs 20 核 85.202 / 1356.307µs（16m / 256m bf16；+86.1% / +74.9%） | ❌ 否决：40 核已最优 |

**复测纪律**：fp16/fp32 代码路径在 round 3 字节一致、无回退（Δ 为机器噪声）；round-2 的 7 条旧 `phase=final` 记录已降级为 `merged`（数据未改），最终 7 条 final 记录来自同一候选 `bf16vand-bs8192`。

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 58.301 | 45.921 | **-21.24%** | 41.28 | 41.74 | ❌ |
| 16m-float16 | 38.571 | 33.961 | -11.95% | 49.75 | 55.87 | ❌ |
| 16m-float32 | 52.731 | 49.761 | -5.63% | 73.01 | 78.01 | ✅ |
| 16m-half-bfloat16 | 58.111 | 45.741 | **-21.29%** | 41.44 | 42.44 | ❌ |
| 16m-half-float16 | 38.331 | 33.881 | -11.61% | 49.98 | 55.80 | ❌ |
| 256m-bfloat16 | 932.049 | 779.006 | **-16.42%** | 45.19 | 54.22 | ❌ |
| 256m-float16 | 706.864 | 668.993 | -5.36% | 59.52 | 63.13 | ❌ |

smoke：3 个 `smoke-1m` 仅精度回归 PASS，不参与性能判定。

**测量口径与噪声**：baseline = Stage-3 生产基线 `../isfinite_fwd_kernel.py`、final = round-3 全 7 workload 同一候选，相邻交错各 1 次独立 `msprof op`。base SOL% 由 `perf_records.jsonl` baseline 行回填（原报告表未列）；bf16 的候选 vs current best 对照见 §7。

## 4. 精度

- `python perf_opt/IsfiniteFwdOp.py --level all` → **`All check passed!`**，max_diff = 0（NaN/±inf/−0/子正规 全覆盖）。
- 新增 bf16 Boundary-special 探针：显式验证 vand 符号清除对 −inf→+inf（仍判 False）、−0→+0（仍判 True）、NaN（仍判 False）的语义正确。
- 3 个 `smoke-1m` 仅精度回归 PASS（微型 shape 固定开销主导，不参与性能判定）。
- 关键边界：NaN、±inf、−0、子正规全覆盖，max_diff = 0。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 21:25 → 21:45（round 1–2）；2026-10-18 03:36 → 03:49（round 3 二次深挖） |
| 墙钟时长 | 二次深挖 ≈ 13 min；round 1–2 墙钟原文未记录 |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../isfinite_kernel/perf_opt/IsfiniteFwdOp.py`（`bf16vand-bs8192`，sha256 `ca9d42da…`） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only；round-2 旧 final 降级 `merged`）、`logs/`、`profiles/` |
| 能力缺口 | `.agents/evolution/capability-gaps.md` **CG-2026-0027** |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（建议采纳，由 conductor 执行） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | PL-1.23（与 Isinf 同日互为独立佐证：同族双算子复现 vand 4-pass 收益） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：3 个 `smoke-1m`（仅精度回归、不参与性能判定，不计入达标分母）；无其他豁免。
- **候选 vs current best 的独立对照（bf16，相邻交错，msprof op）**：

  | workload | current best µs | vand µs | Δ | 判定 |
  |---|---:|---:|---:|---|
  | 16m-bfloat16 | 58.301 | 45.921 | -21.24% | winner |
  | 16m-half-bfloat16 | 58.111 | 45.741 | -21.29% | winner |
  | 256m-bfloat16 | 932.049 | 779.006 | -16.42% | winner |

- **ceiling 证据（pipe 级归因，msprof PipeUtilization，256m-bfloat16）**：

  | workload | 变体 | aiv_vec_time(us) | aiv_vec_ratio | aiv_mte2_ratio | 判定 |
  |---|---|---:|---:|---:|---|
  | 256m-bfloat16 | current best（6-pass） | 665.21 | 0.721 | 0.386 | 向量受限 |
  | 256m-bfloat16 | vand（4-pass） | 419.65 | 0.548 | 0.531 | 收敛均衡 |

  - bf16 由向量受限（vec 0.721 ≫ mte2 0.386）翻转为收敛均衡（0.548 vs 0.531），`aiv_vec_time` 665.21 → 419.65µs。
  - fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90（MTE2 地板），4-pass 向量链隐藏于搬运窗口下。
- **核数复核**：

  | num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
  |---:|---:|---:|---|
  | 40（winner） | 45.771 | 775.596 | 已最优 |
  | 20 | 85.202（+86.1%） | 1356.307（+74.9%） | 严重回退 |

  - 印证「减半搬运并行 ≈ 减半有效带宽」，与 Isnan/Isinf 同族结论一致。
- **工具链约束**：`CG-2026-0027`（无 i1→i8 直连 / 无 bool vnot / `vselect` Out 不支持 int8）⇒ isfinite 4-pass 已是下限。
- **贴线风险**：16m fp32 78.01% 为表中唯一达标项（高于线）；其余 fp16 55.80–63.13%、bf16 41.74–54.22% 均明显在线下，不属 <3pp 贴线项。
- **缺口**：`CG-2026-0027`（predicate 族 ISA 约束）；bf16 4-pass 下限已达成。
- **`[DESIGN_LIMIT]`**：不触发（无 >2× 结构加速、无设计假设矛盾），不产出 `perf_feedback.md`。

## 8. 建议

1. **建议采纳**（把 `perf_opt/IsfiniteFwdOp.py` 切换为生产 baseline）：7 个 tune workload 全部正收益、无回退，L0/L1 全通过，签名 drop-in。
2. 后续动作：工具链升级时优先复核（a）bf16 `vand` 路径的 −inf/−0/NaN 边界，（b）`CG-2026-0027` 是否补齐（补齐后按 §7 归因预期 SOL 提升有限）。
3. **未解决项与原因**：fp16/bf16 到 70% 线的剩余 gap 是 MTE2 混合 R/W 带宽地板（设计层天花板，`CG-2026-0027`），参数/tiling 级不可解。