# IsinfFwdOp — Stage 4 调优日志

> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）·
> Ascend910B3（40 AIV）· 2026-10-17 / 2026-10-18（二次深挖）。口径：msprof op Task Duration(us) + GM Read+Write SOL ratio。
> 判据：AI ≈ 2.0 → 访存受限，线 70%。

## 0. 对象与入口

- `IsinfFwdOp`（`y = isinf(x)`，bool 经 int8 存储）；基线 `isinf_kernel/isinf_fwd_kernel.py`；
- 产物 `perf_opt/IsinfFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`）；benchmark `bench_isinf.py`（7 tune + 3 smoke）。

## 1. Baseline（round 1，tune）

| workload | µs | SOL% |
|---|---:|---:|
| 16m fp32 | 52.77 | 73.94 |
| 16m fp16 | 38.98 | 50.19 |
| 16m bf16 | 57.38 | 42.37 |
| 16m-half fp16 | 38.34 | 50.46 |
| 16m-half bf16 | 57.23 | 42.45 |
| 256m fp16 | 711.45 | 59.01 |
| 256m bf16 | 918.79 | 46.02 |

基线链 6 条向量 pass：`vcmp(eq +inf) + vcmp(eq -inf) + vcast + vcast + vadd + vcast`。

## 2. 迭代日志

### Round 2 — vabs 合并正负 inf（fp16/fp32；候选 vs current best，Task Duration）

`isinf(x) = (|x| == +inf)`：`vabs → vcmp(eq) → vcast(bool→f16) → vcast(f16→i8)`，**6 pass → 4 pass**。

| workload | baseline µs | abs µs | Δ |
|---|---:|---:|---:|
| 16m fp16 | 38.02 | 33.52 | **−11.84%** |
| 16m-half fp16 | 38.83 | 33.71 | **−13.19%** |
| 256m fp16 | 706.20 | 671.25 | −4.95% |

- **bf16 不适用**：`T.vabs` dtype 表无 bf16。先试 `vselect` 合并两条 mask，实测 **+6.6~8.6%（更慢）**，
  已回退为基线的 `2×vcmp + 2×vcast + vadd + vcast`（Round 3 复测 −0.2~−0.9%，中性）。

### Round 3 — bf16 回退验证

| workload | baseline µs | 候选 µs | Δ |
|---|---:|---:|---:|
| 16m bf16 | 57.411 | 57.161 | −0.44% |
| 256m bf16 | 918.268 | 915.058 | −0.35% |

### 合并与 winner（round 4）

- fp16/fp32：**4-pass vabs 链**；bf16：**基线 6-pass 两条 mask 合并**；两者共用静态尾块 + Pipelined 结构；
- 合并进 `perf_opt/IsinfFwdOp.py`，基准与 wrapper 未改。

### Round 5 — 二次深挖（2026-10-18）：bf16 位运算 |x| 降到 4-pass ⭐ 新杠杆

**现象**：同族 Isnan 二次深挖（同日）证明 predicate 族 fp16/fp32 是 **MTE2 混合 R/W 带宽地板**
（`aiv_mte2_ratio 0.81–0.90 ≫ aiv_vec_ratio 0.25–0.39`），但 **bf16 是唯一的向量受限例外**（baseline
`vec_t > mte2_t`）。Isinf 的 bf16 路径仍是 6-pass（`vabs` 无 bf16 边），是唯一非下限项。

**候选优化点**：用位运算 `vand(x, 0x7FFF)` 掩掉符号位，1 op 得到 |x|，把 bf16 从 6-pass 降到 4-pass，
与 fp16/fp32 对齐。0x7FFF 掩码用 `T.fill(-0.0) → T.vnot` 一次性构造（`~0x8000 = 0x7FFF`），
不在 tile 循环内。依据文档：`docs/Tilelang.language/逻辑操作/T.vand.md` 2.2.1（bf16 = √，按位级
reinterpret）、`T.vnot.md` 2.2.1（bf16 = √）。

**候选 vs current best（bf16，相邻交错，各 1 次 msprof op）**：

| workload | current best µs | vand µs | Δ Task Duration | current SOL% | vand SOL% | L0 |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.001 | 45.611 | **−19.98%** | 41.67 | 41.27 | pass |
| 16m-half-bfloat16 | 56.971 | 45.781 | **−19.64%** | 42.42 | 41.72 | pass |
| 256m-bfloat16 | 912.198 | 774.135 | **−15.13%** | 46.16 | 53.95 | pass |

**pipe 级归因（权威证据，PipeUtilization CSV）**：

| workload | 变体 | aiv_vec_ratio | aiv_mte2_ratio | 判定 |
|---|---|---:|---:|---|
| 16m-bfloat16 | current best（6-pass） | **0.757** | 0.359 | 向量受限 |
| 16m-bfloat16 | vand（4-pass） | 0.622 | 0.458 | 收敛均衡 |
| 256m-bfloat16 | current best（6-pass） | **0.725** | 0.395 | 向量受限 |
| 256m-bfloat16 | vand（4-pass） | 0.546 | 0.534 | 收敛均衡 |

bf16 baseline `aiv_vec_ratio 0.72–0.76 ≫ aiv_mte2_ratio 0.36–0.40`（**向量受限**，6-pass 向量链是
关键路径）；vand 4-pass 后 `aiv_vec_time` 653.4→420.2µs（−35.7%，256m），vec 占比降至 0.55，与
mte2 0.53 基本均衡——**bf16 从向量受限翻转为接近带宽地板**，与 Isnan 的 bf16 例外结论完全一致，
且 6→4（删 2 pass）比 Isnan 的 4→3（删 1 pass）收益更大。

**核数复核（40 核已最优）**：

| num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
|---:|---:|---:|---|
| 40（winner） | **45.811** | **774.206** | 已最优 |
| 20 | 85.132（+85.9%） | 1351.637（+74.6%） | 严重回退 |

40 核已最优（20 核搬运并行减半 ≈ 有效带宽减半），与 Isnan 同族结论一致。

**winner**：`bf16vand-bs8192`（bf16 = vand 4-pass；fp16/fp32 = vabs 4-pass 不变）。合并进
`perf_opt/IsinfFwdOp.py`（overwrite），fp16/fp32 代码路径字节一致、无回退。

## Final Performance Test Data

官方最终测量：`perf_opt/IsinfFwdOp.py`（round 5，`bf16vand-bs8192`，baseline=round-4 相邻交错
6-pass baseline；final=round-5 全 7 workload 同一候选）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---:|---:|---:|---|
| isinf_fwd_kernel | 16m-bfloat16 | 57.071 | 46.001 | bf16vand-bs8192 |
| isinf_fwd_kernel | 16m-float16 | 38.561 | 33.271 | bf16vand-bs8192 |
| isinf_fwd_kernel | 16m-float32 | 52.761 | 51.481 | bf16vand-bs8192 |
| isinf_fwd_kernel | 16m-half-bfloat16 | 57.251 | 45.871 | bf16vand-bs8192 |
| isinf_fwd_kernel | 16m-half-float16 | 37.951 | 33.731 | bf16vand-bs8192 |
| isinf_fwd_kernel | 256m-bfloat16 | 917.428 | 775.936 | bf16vand-bs8192 |
| isinf_fwd_kernel | 256m-float16 | 709.204 | 662.403 | bf16vand-bs8192 |

### SOL 对账

| workload | baseline µs | final µs | Δ | final SOL% | ≥70% |
|---|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 46.001 | -19.40% | 41.36 | ❌ |
| 16m-float16 | 38.561 | 33.271 | -13.72% | 56.00 | ❌ |
| 16m-float32 | 52.761 | 51.481 | -2.43% | 74.72 | ✅ |
| 16m-half-bfloat16 | 57.251 | 45.871 | -19.88% | 41.92 | ❌ |
| 16m-half-float16 | 37.951 | 33.731 | -11.12% | 57.30 | ❌ |
| 256m-bfloat16 | 917.428 | 775.936 | -15.42% | 54.46 | ❌ |
| 256m-float16 | 709.204 | 662.403 | -6.60% | 63.01 | ❌ |

**结论（如实）**：

- **7 个 tune case 中 1 个 ≥70%**（16m fp32 74.72%）；fp16 56–63%、bf16 41.4–54.5%；
- **Round 5 新杠杆 = bf16 6→4 pass（vand 位运算 |x|）**：16m bf16 **−19.4%**、16m-half bf16 −19.9%、
  256m bf16 **−15.4%**（相对 round-4 bf16 保持 6-pass 的 −0.2~−0.9% 是质的跃迁）；
- fp16/fp32 与 round-4 vabs 链一致，round-5 无改动（Δ 为机器噪声）；
- bf16 由向量受限（vec_ratio 0.72–0.76）翻转为接近带宽地板（vec_ratio 0.55–0.62 ≈ mte2 0.46–0.53）；
- 剩余 gap（fp16 56–63%、bf16 41.4–54.5% vs 70%）仍是 **MTE2 混合 R/W 带宽地板**（fp16/fp32 已证
  mte2_ratio 0.81–0.90；bf16 4-pass 后也收敛到该地板），不是 pass 数可解（isinf 4-pass 已是下限：
  |x| + vcmp + vcast(bool→f16) + vcast(f16→i8)，CG-2026-0027 无 i1→i8 直连 / bool vnot / int8 vselect）。

## 精度回归

- `python perf_opt/IsinfFwdOp.py --level all` → **All check passed!**（NaN/±inf/−0/子正规 全覆盖，max_diff=0）；
- bf16 位运算 |x| 的 NaN/−inf/+inf/−0 语义在 L0 special 输入下逐位验证通过（若 0x7FFF 掩码构造错误，
  −inf 会误判为 False，L0 必失败——已通过）。

## 证据

`perf_records.jsonl`（round 1–5 全部测量，append-only；round-4 旧 final 7 行降级为 merged，
数据未改）、`profiles/`（round-5 candidate/final/cores 的 msprof raw）、`logs/`。
pipe 级证据：`profiles/*-r5-*/PipeUtilization_*.csv`（aiv_vec_ratio vs aiv_mte2_ratio）。

## Skill Retrospective

- **D（实测）**：bf16 是 predicate 族唯一的向量受限例外，且 **vand(x, 0x7FFF) 位运算 |x| 可把 isinf
  bf16 从 6-pass 降到 4-pass**，16m −19.4~−19.9%、256m −15.4%——这是 predicate 族目前最大的单点收益
  （超过 fp16 vabs 的 −11.8~−12.2%）。pipe 证据：bf16 baseline vec_ratio 0.72–0.76 ≫ mte2 0.36–0.40
  （向量受限），4-pass 后 vec_ratio 0.55–0.62 ≈ mte2 0.46–0.53（收敛均衡）。回写 PL-1.23（bf16 不再
  是「6-pass 死路」，有 vand 位运算 4-pass 路径）。
- **P（模式）**：`T.vand`/`T.vor`/`T.vnot` 支持 bf16 按位级 reinterpret（`逻辑操作/` 文档 2.2.1）；
  `|x| = vand(x, ~(-0.0))` 是一次性掩码构造（`T.fill(-0.0) → T.vnot`，循环外），比 `vmax(x, vsub(0,x))`
  （2 op）更省、且 NaN 传播语义明确（bit 级掩码天然保留 NaN 的 mantissa，`vcmp eq` ordered 比较正确
  排除 NaN）。同族 isfinite/其他 bf16 无 vabs 边的逐元素算子可复用该路径。
- **R（流程）**：secondary deep-dive 改变 winner 时，旧 session 的 `phase=final` 记录会与新 final 的
  哈希/候选不一致，触发 gate 4 `S4-PERF-RECORDS-RECON`（「必须恰有一条 final」+「哈希与采集时一致」）。
  正确收束动作：把被取代的旧 final 行**降级为 `merged`**（保留全部测量数据，仅更正已失效的 phase 标签），
  不得删除或伪造测量行。建议在 skill 里明确「winner 变更时旧 final → merged」的降级规范。
- **C（案例）**：isinf bf16 6→4 pass 的完整 before/after 见
  `perf_opt/candidates/isinf_fwd_kernel_abs.py`（before：bf16 6-pass）vs
  `perf_opt/IsinfFwdOp.py`（after：bf16 vand 4-pass）。

## 证据索引

`perf_records.jsonl`、`logs/l0_*.log`、`profiles/*-r5-*`、`candidates/isinf_fwd_kernel_abs.py`、
`IsinfFwdOp_round5_bf16vand.py`。
