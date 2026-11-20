# IsfiniteFwdOp — Stage 4 调优日志

> CANN 9.0.0 / tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）/ Ascend910B3 / 2026-10-17 / 2026-10-18（二次深挖）。
> 口径：msprof op Task Duration(us) + GM Read+Write SOL ratio；AI ≈ 2.0 → 访存受限，线 70%。

## 0. 对象与入口
- `IsfiniteFwdOp`（`y = isfinite(x)`，bool 经 int8 存储）；基线 `isfinite_kernel/isfinite_fwd_kernel.py`；
- 产物 `perf_opt/IsfiniteFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）；benchmark `bench_isfinite.py`（7 tune + 3 smoke）。

## 1. Baseline 与诊断
基线链 6 pass：`vcmp(gt -inf) + vcmp(lt +inf) + vcast + vcast + vmul + vcast`；
16m fp32 73.58%、fp16 48.7%、bf16 40.7%；256m fp16 59.1%、bf16 45.3%。

## 2. 迭代日志（候选 vs current best，Task Duration）

### Round 1 — `isfinite(x) = |x| < +inf`（fp16/fp32，4 pass）
`vabs → vcmp(lt) → vcast(bool→f16) → vcast(f16→i8)`；NaN 的有序 `lt` 为 false，天然排除。

| workload | baseline µs | abs µs | Δ |
|---|---:|---:|---:|
| 16m fp16 | 38.72 | 33.22 | **−14.20%** |
| 16m-half fp16 | 38.47 | 33.36 | **−13.28%** |
| 256m fp16 | 704.86 | 669.75 | −4.98% |

bf16：`T.vabs` 无 bf16 dtype 边，保留基线两条 mask 链（+静态尾块/流水），−2.0~−2.9%。

### Round 3 — 二次深挖（2026-10-18）：bf16 位运算 |x| 降到 4-pass ⭐ 新杠杆

**现象**：同族 Isinf 二次深挖（同日）证明 predicate 族 fp16/fp32 是 **MTE2 混合 R/W 带宽地板**
（`aiv_mte2_ratio 0.81–0.90 ≫ aiv_vec_ratio 0.25–0.39`），但 **bf16 是唯一的向量受限例外**
（baseline `aiv_vec_ratio 0.72 ≫ aiv_mte2_ratio 0.39`）。Isfinite 的 bf16 路径在 round-2 仍是
6-pass（`vabs` 无 bf16 边），是唯一非下限项。

**候选优化点**：用位运算 `vand(x, 0x7FFF)` 掩掉符号位，1 op 得到 |x|，把 bf16 从 6-pass 降到
4-pass，与 fp16/fp32 对齐。语义：`isfinite(x) = (|x| < +inf)`（`lt` 为 ordered 比较，NaN 天然
排除）。0x7FFF 掩码用 `T.fill(-0.0) → T.vnot` 一次性构造（`~0x8000 = 0x7FFF`），不在 tile
循环内。依据文档：`docs/Tilelang.language/逻辑操作/T.vand.md` 2.2.1（bf16 = √，按位级
reinterpret）、`T.vnot.md` 2.2.1（bf16 = √）。

**候选 vs current best（bf16，相邻交错，各 1 次 msprof op）**：

| workload | current best µs | vand µs | Δ Task Duration | current SOL% | vand SOL% | L0 |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 58.301 | 45.921 | **−21.24%** | 41.283 | 41.736 | pass |
| 16m-half-bfloat16 | 58.111 | 45.741 | **−21.29%** | 41.44 | 42.438 | pass |
| 256m-bfloat16 | 932.049 | 779.006 | **−16.42%** | 45.193 | 54.219 | pass |

**pipe 级归因（权威证据，PipeUtilization CSV，256m-bfloat16）**：

| workload | 变体 | aiv_vec_time(us) | aiv_vec_ratio | aiv_mte2_ratio | 判定 |
|---|---|---:|---:|---:|---|
| 256m-bfloat16 | current best（6-pass） | 665.21 | 0.721 | 0.386 | 向量受限 |
| 256m-bfloat16 | vand（4-pass） | 419.65 | 0.548 | 0.531 | 收敛均衡 |

bf16 baseline `aiv_vec_ratio ≈ 0.72 ≫ aiv_mte2_ratio ≈ 0.39`（**向量受限**，6-pass 向量链是关键
路径）；vand 4-pass 后 `aiv_vec_time` 665.2→419.7µs（−36.9%，256m），vec 占比降至 0.55，与
mte2 0.53 基本均衡——**bf16 从向量受限翻转为接近带宽地板**，与 Isnan/Isinf 的 bf16 例外结论
完全一致，且 6→4（删 2 pass）比 Isinf 的 6→4（删 2 pass）收益相当（本族 −16.4~−21.3% vs
Isinf −15.1~−20.0%）。

**核数复核（40 核已最优）**：

| num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
|---:|---:|---:|---|
| 40（winner） | **45.771** | **775.596** | 已最优 |
| 20 | 85.202（+86.1%） | 1356.307（+74.9%） | 严重回退 |

40 核已最优（20 核搬运并行减半 ≈ 有效带宽减半），与 Isnan/Isinf 同族结论一致。

**winner**：`bf16vand-bs8192`（bf16 = vand 4-pass；fp16/fp32 = vabs 4-pass 不变）。合并进
`perf_opt/IsfiniteFwdOp.py`（overwrite），fp16/fp32 代码路径字节一致、无回退（Δ 为机器噪声：
round-2 final 33.371 → round-3 final 33.961 等）。

## Final Performance Test Data

官方最终测量：`perf_opt/IsfiniteFwdOp.py`（round 3，`bf16vand-bs8192`，baseline=Stage-3 生产
基线 `../isfinite_fwd_kernel.py` 相邻交错；final=round-3 全 7 workload 同一候选）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---:|---:|---:|---|
| isfinite_fwd_kernel | 16m-bfloat16 | 58.301 | 45.921 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 16m-float16 | 38.571 | 33.961 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 16m-float32 | 52.731 | 49.761 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 16m-half-bfloat16 | 58.111 | 45.741 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 16m-half-float16 | 38.331 | 33.881 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 256m-bfloat16 | 932.049 | 779.006 | bf16vand-bs8192 |
| isfinite_fwd_kernel | 256m-float16 | 706.864 | 668.993 | bf16vand-bs8192 |

### SOL 对账

| workload | baseline µs | final µs | Δ | final SOL% | ≥70% |
|---|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 58.301 | 45.921 | -21.24% | 41.74 | ❌ |
| 16m-float16 | 38.571 | 33.961 | -11.95% | 55.87 | ❌ |
| 16m-float32 | 52.731 | 49.761 | -5.63% | 78.01 | ✅ |
| 16m-half-bfloat16 | 58.111 | 45.741 | -21.29% | 42.44 | ❌ |
| 16m-half-float16 | 38.331 | 33.881 | -11.61% | 55.80 | ❌ |
| 256m-bfloat16 | 932.049 | 779.006 | -16.42% | 54.22 | ❌ |
| 256m-float16 | 706.864 | 668.993 | -5.36% | 63.13 | ❌ |

**结论（如实）**：

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 78.01%）；fp16 55.8–63.1%、bf16 41.7–54.2%；
- **Round 3 新杠杆 = bf16 6→4 pass（vand 位运算 |x|）**：16m bf16 **−21.24%**、16m-half bf16
  −21.29%、256m bf16 **−16.42%**（相对 round-2 bf16 保持 6-pass 的 −2.2~−2.8% 是质的跃迁）；
- fp16/fp32 与 round-2 vabs 4-pass 链一致，round-3 无改动（Δ 为机器噪声）；
- bf16 由向量受限（`aiv_vec_ratio` 0.72）翻转为接近带宽地板（`aiv_vec_ratio` 0.55 ≈
  `aiv_mte2_ratio` 0.53）；
- 剩余 gap（fp16 55.8–63.1%、bf16 41.7–54.2% vs 70%）仍是 **MTE2 混合 R/W 带宽地板**，不是
  pass 数可解（isfinite 4-pass 已是下限：|x| + vcmp + vcast(bool→f16) + vcast(f16→i8)，
  CG-2026-0027 无 i1→i8 直连 / bool vnot / int8 vselect）。

## 精度回归
- `python perf_opt/IsfiniteFwdOp.py --level all` → **All check passed!**，max_diff=0（NaN/±inf/−0/子正规
  全覆盖；新增 bf16 Boundary-special 探针显式验证 vand 符号清除对 −inf→+inf（仍判 False）、
  −0→+0（仍判 True）、NaN（仍判 False）的语义正确）；
- 3 个 `smoke-1m` 仅精度回归 PASS（微型 shape 固定开销主导，不参与性能判定）。

## Skill Retrospective
- **D（实测）**：bf16 是 predicate 族唯一的向量受限例外，且 **vand(x, 0x7FFF) 位运算 |x| 可把
  isfinite bf16 从 6-pass 降到 4-pass**，16m −21.2~−21.3%、256m −16.4%。pipe 证据：bf16 baseline
  `aiv_vec_ratio` 0.72 ≫ `aiv_mte2_ratio` 0.39（向量受限），4-pass 后 vec_ratio 0.55 ≈ mte2 0.53
  （收敛均衡）。与 Isinf 同日结论互为独立佐证（同族双算子复现 vand 4-pass 收益）。
- **P（模式）**：`T.vand`/`T.vor`/`T.vnot` 支持 bf16 按位级 reinterpret（`逻辑操作/` 文档 2.2.1）；
  `|x| = vand(x, ~(-0.0))` 一次性掩码构造（`T.fill(-0.0) → T.vnot`，循环外），比 `vmax(x, vsub(0,x))`
  更省，且 NaN 传播语义明确（bit 级掩码保留 NaN mantissa，`vcmp lt` ordered 比较正确排除 NaN）。
  同族 predicate 及其他 bf16 无 vabs 边的逐元素算子可复用该路径。
- **R（流程）**：secondary deep-dive 改变 winner 时，旧 session 的 `phase=final` 记录会与新 final 的
  哈希/候选不一致，触发 gate 4 `S4-PERF-RECORDS-RECON`。正确收束动作：把被取代的旧 final 行
  **降级为 `merged`**（保留全部测量数据，仅更正已失效的 phase 标签），不得删除或伪造测量行。
  本轮将 round-2 的 7 条 `abs-pipe2-bs8192` final 行降级为 merged，与 Isinf 收束动作一致。
- **C（案例）**：isfinite bf16 6→4 pass 的完整 before/after 见
  `perf_opt/candidates/isfinite_fwd_kernel_abs.py`（before：bf16 6-pass）vs
  `perf_opt/IsfiniteFwdOp.py`（after：bf16 vand 4-pass，`use_vand` 分支）。

## 证据索引
`perf_records.jsonl`（round 1–3 全部测量，append-only；round-2 旧 final 7 行降级为 merged，
数据未改）、`profiles/`（round-3 candidate/final/cores 的 msprof raw）、`logs/`。
pipe 级证据：`profiles/*-r3-*/PipeUtilization_*.csv`（aiv_vec_ratio vs aiv_mte2_ratio）。
