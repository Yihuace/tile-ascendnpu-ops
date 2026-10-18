# 优化报告：IsinfFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsinfFwdOp`；纯 Vector，`y = isinf(x)`，bool 经 int8 字节存储 |
| kernel | `tileops/kernels/elementwise/isinf/isinf_kernel/` |
| 调优产物 | `.../isinf_kernel/perf_opt/IsinfFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | msprof op Task Duration(us) + GM Read+Write SOL ratio |
| 工具链 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-17 → 2026-10-18（二次深挖） |
| benchmark | `benchmarks/ops/bench_isinf.py::test_isinf_bench`（7 tune + 3 smoke） |

## 1. 结论

- **达标数**：可调 shape（7 tune）**1/7 ≥70%**（16m fp32 74.72%）；全 shape（含 3 个 `smoke-1m`，
  仅精度不参与性能判定）性能判定口径同为 **1/7**。16m/256m fp16 56–63%，bf16 41.4–54.5%。
- **Round 5（二次深挖）找到新杠杆 = bf16 位运算 |x| 降到 4-pass**：`vand(x, 0x7FFF)` 掩符号位，
  使 bf16 从 6-pass 降到与 fp16/fp32 相同的 4-pass。16m bf16 **−19.4%**、16m-half bf16 −19.9%、
  256m bf16 **−15.4%**（round-4 的 bf16 保持 6-pass 只有 −0.2~−0.9%）。
- fp16/fp32 沿用 round-4 vabs 4-pass 链（−2.4~−13.7%），round-5 无改动、无回退。
- **bf16 由向量受限翻转为接近带宽地板**：`aiv_vec_ratio` 0.72–0.76 → 0.55–0.62（≈ `aiv_mte2_ratio`
  0.46–0.53），与 Isnan 同族的「bf16 是唯一向量受限例外」结论一致。
- 剩余 gap 到 70% 是 **MTE2 混合 R/W 带宽地板**（fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90），
  不是 pass 数可解（isinf 4-pass 已是下限，CG-2026-0027 无 i1→i8 直连 / bool vnot / int8 vselect）。

## 2. 所做改动

仅在 `perf_opt/` 内（基准与 wrapper 未动）：

1. 满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核；
2. fp16/fp32：**vabs 4-pass 链**（round-4 winner）；
3. **Round 5：bf16 |x| = `vand(x, 0x7FFF)` 位运算符号位清除（4-pass）**，掩码 `T.fill(-0.0) →
   `T.vnot` 一次性构造（循环外）；
4. `TUNED_BLOCK_SIZE = 8192`。

## 3. 优化结果（round 5 final，全 7 workload 同一候选 `bf16vand-bs8192`）

| workload | base µs | opt µs | Δ | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 57.071 | 46.001 | **-19.40%** | 41.36 | ❌ |
| 16m-float16 | 38.561 | 33.271 | -13.72% | 56.00 | ❌ |
| 16m-float32 | 52.761 | 51.481 | -2.43% | 74.72 | ✅ |
| 16m-half-bfloat16 | 57.251 | 45.871 | **-19.88%** | 41.92 | ❌ |
| 16m-half-float16 | 37.951 | 33.731 | -11.12% | 57.30 | ❌ |
| 256m-bfloat16 | 917.428 | 775.936 | **-15.42%** | 54.46 | ❌ |
| 256m-float16 | 709.204 | 662.403 | -6.60% | 63.01 | ❌ |

### 精度

`python perf_opt/IsinfFwdOp.py --level all` → `All check passed!`，max_diff = 0（NaN/±inf/−0/子正规 全覆盖）；
3 个 `smoke-1m` 仅精度回归 PASS。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 21:05 → 21:35（round 1–4）；2026-10-18 02:57 → 03:15（round 5 二次深挖） |
| 二次深挖墙钟 | ≈ 18 min |

## 5. 产物

- `.../isinf_kernel/perf_opt/IsinfFwdOp.py`（final，`bf16vand-bs8192`，sha256 783957dc…）
- `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`logs/`、`profiles/`、`candidates/`
- 能力缺口：`.agents/evolution/capability-gaps.md` **CG-2026-0027**

## 6. 二次深挖（Round 5，2026-10-18，一次性复核）

> 目标「确认天花板 or 找到新杠杆」。结论：**找到新杠杆（bf16 位运算 |x| 4-pass），bf16 −15~−20%**。

### 6.1 候选 vs current best（bf16，相邻交错，msprof op）

| workload | current best µs | vand µs | Δ | 判定 |
|---|---:|---:|---:|---|
| 16m-bfloat16 | 57.001 | 45.611 | -19.98% | winner |
| 16m-half-bfloat16 | 56.971 | 45.781 | -19.64% | winner |
| 256m-bfloat16 | 912.198 | 774.135 | -15.13% | winner |

### 6.2 pipe 级归因（bf16 向量受限 → 收敛均衡）

| workload | 变体 | aiv_vec_ratio | aiv_mte2_ratio |
|---|---|---:|---:|
| 16m-bfloat16 | 6-pass baseline | **0.757** | 0.359 |
| 16m-bfloat16 | vand 4-pass | 0.622 | 0.458 |
| 256m-bfloat16 | 6-pass baseline | **0.725** | 0.395 |
| 256m-bfloat16 | vand 4-pass | 0.546 | 0.534 |

### 6.3 核数复核

| num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
|---:|---:|---:|---|
| 40（winner） | **45.811** | **774.206** | 已最优 |
| 20 | 85.132（+85.9%） | 1351.637（+74.6%） | 严重回退 |

### 6.4 ceiling 证据（判据复核）

- bf16 由向量受限（vec_ratio 0.72–0.76 ≫ mte2 0.36–0.40）翻转为收敛均衡（0.55–0.62 vs 0.46–0.53）；
  fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90（MTE2 地板），4-pass 向量链隐藏于搬运窗口下。
- 不触发 `[DESIGN_LIMIT]`（无 >2× 结构加速、无设计假设矛盾）；bf16 的 4-pass 下限已达成，
  CG-2026-0027 约束（无 i1→i8 直连 / bool vnot / int8 vselect）仍使 isinf 无法 3-pass。

### 6.5 采纳建议

**建议采纳 wrapper（翻转 perf_opt 版本）**：相对基线 7 个 tune workload 全部正收益、无回退——
bf16 −15.4~−19.9%（round-5 新杠杆）、fp16 −6.6~−13.7%、fp32 −2.4%（round-4 vabs 链）。此前
round-4「不建议采纳」的主要障碍（bf16 收益 ~0）已被 vand 4-pass 消除。是否最终翻转由 conductor
在回归通过后执行，本阶段不自行翻转 wrapper。
