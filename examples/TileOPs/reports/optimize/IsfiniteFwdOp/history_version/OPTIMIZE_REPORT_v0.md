# 优化报告：IsfiniteFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsfiniteFwdOp`；纯 Vector，`y = isfinite(x)`，bool 经 int8 字节存储 |
| kernel | `tileops/kernels/elementwise/isfinite/isfinite_kernel/` |
| 调优产物 | `.../isfinite_kernel/perf_opt/IsfiniteFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | msprof op Task Duration(us) + GM Read+Write SOL ratio |
| 工具链 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-17 → 2026-10-18（二次深挖） |
| benchmark | `benchmarks/ops/bench_isfinite.py::test_isfinite_bench`（7 tune + 3 smoke） |

## 1. 结论

- **达标数**：可调 shape（7 tune）**1/7 ≥70%**（16m fp32 78.01%）；全 shape（含 3 个 `smoke-1m`，
  仅精度不参与性能判定）性能判定口径同为 **1/7**。16m/256m fp16 55.8–63.1%，bf16 41.7–54.2%。
- **Round 3（二次深挖）找到新杠杆 = bf16 位运算 |x| 降到 4-pass**：`vand(x, 0x7FFF)` 掩符号位，
  使 bf16 从 6-pass 降到与 fp16/fp32 相同的 4-pass。16m bf16 **−21.24%**、16m-half bf16 −21.29%、
  256m bf16 **−16.42%**（round-2 的 bf16 保持 6-pass 只有 −2.2~−2.8%）。
- fp16/fp32 沿用 round-2 vabs 4-pass 链（−5.4~−12.0%），round-3 无改动、无回退。
- **bf16 由向量受限翻转为接近带宽地板**：`aiv_vec_ratio` 0.72 → 0.55（≈ `aiv_mte2_ratio`
  0.39 → 0.53），与 Isnan/Isinf 同族的「bf16 是唯一向量受限例外」结论一致。
- 剩余 gap 到 70% 是 **MTE2 混合 R/W 带宽地板**（fp16/fp32 已证 `aiv_mte2_ratio` 0.81–0.90），
  不是 pass 数可解（isfinite 4-pass 已是下限，CG-2026-0027 无 i1→i8 直连 / bool vnot / int8 vselect）。

## 2. 所做改动

仅在 `perf_opt/` 内（基准与 wrapper 未动）：

1. 满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核；
2. fp16/fp32：**vabs 4-pass 链**（round-2 winner）；
3. **Round 3：bf16 |x| = `vand(x, 0x7FFF)` 位运算符号位清除（4-pass）**，掩码 `T.fill(-0.0) →
   `T.vnot` 一次性构造（循环外）；
4. `TUNED_BLOCK_SIZE = 8192`。

## 3. 优化结果（round 3 final，全 7 workload 同一候选 `bf16vand-bs8192`）

| workload | base µs | opt µs | Δ | opt SOL% | ≥70% |
|---|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 58.301 | 45.921 | **-21.24%** | 41.74 | ❌ |
| 16m-float16 | 38.571 | 33.961 | -11.95% | 55.87 | ❌ |
| 16m-float32 | 52.731 | 49.761 | -5.63% | 78.01 | ✅ |
| 16m-half-bfloat16 | 58.111 | 45.741 | **-21.29%** | 42.44 | ❌ |
| 16m-half-float16 | 38.331 | 33.881 | -11.61% | 55.80 | ❌ |
| 256m-bfloat16 | 932.049 | 779.006 | **-16.42%** | 54.22 | ❌ |
| 256m-float16 | 706.864 | 668.993 | -5.36% | 63.13 | ❌ |

### 精度

`python perf_opt/IsfiniteFwdOp.py --level all` → `All check passed!`，max_diff = 0（NaN/±inf/−0/子正规
全覆盖，含 bf16 Boundary-special 探针）；3 个 `smoke-1m` 仅精度回归 PASS。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 21:25 → 21:45（round 1–2）；2026-10-18 03:36 → 03:49（round 3 二次深挖） |
| 二次深挖墙钟 | ≈ 13 min |

## 5. 产物

- `.../isfinite_kernel/perf_opt/IsfiniteFwdOp.py`（final，`bf16vand-bs8192`，sha256 ca9d42da…）
- `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only；round-2 旧 final 降级 merged）、`logs/`、`profiles/`
- 能力缺口：`.agents/evolution/capability-gaps.md` **CG-2026-0027**

## 6. 二次深挖（Round 3，2026-10-18，一次性复核）

> 目标「确认天花板 or 找到新杠杆」。结论：**找到新杠杆（bf16 位运算 |x| 4-pass），bf16 −16~−21%**。

### 6.1 候选 vs current best（bf16，相邻交错，msprof op）

| workload | current best µs | vand µs | Δ | 判定 |
|---|---:|---:|---:|---|
| 16m-bfloat16 | 58.301 | 45.921 | -21.24% | winner |
| 16m-half-bfloat16 | 58.111 | 45.741 | -21.29% | winner |
| 256m-bfloat16 | 932.049 | 779.006 | -16.42% | winner |

### 6.2 pipe 级归因（bf16 向量受限 → 收敛均衡）

| workload | 变体 | aiv_vec_time(us) | aiv_vec_ratio | aiv_mte2_ratio | 判定 |
|---|---|---:|---:|---:|---|
| 256m-bfloat16 | current best（6-pass） | 665.21 | 0.721 | 0.386 | 向量受限 |
| 256m-bfloat16 | vand（4-pass） | 419.65 | 0.548 | 0.531 | 收敛均衡 |

### 6.3 核数复核（40 核已最优）

| num_cores | 16m-bfloat16 µs | 256m-bfloat16 µs | 判定 |
|---:|---:|---:|---|
| 40（winner） | 45.771 | 775.596 | 已最优 |
| 20 | 85.202（+86.1%） | 1356.307（+74.9%） | 严重回退 |

### 6.4 建议

**建议采纳 wrapper**（把 `perf_opt/IsfiniteFwdOp.py` 切换为生产 baseline）：bf16 −16.4~−21.3%、
fp16/fp32 −5.4~−12.0% 均为真实 Task Duration 下降，40 核/block 8192 已复核最优，精度全通过、
无回退。剩余 gap 是 MTE2 混合 R/W 带宽地板（设计层天花板，CG-2026-0027），参数/tiling 级不可解。
