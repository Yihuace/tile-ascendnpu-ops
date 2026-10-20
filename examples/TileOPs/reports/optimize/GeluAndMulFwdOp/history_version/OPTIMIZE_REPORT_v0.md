# 优化报告：GeluAndMulFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GeluAndMulFwdOp`（`y = gelu(x[:,:N]) * x[:,N:]`，精确 erf GELU）；单 kernel/单 func/纯 Vector（AIV） |
| kernel | `tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/` |
| 口径 | `msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+ `visualize_data.bin` GM Read+Write SOL |
| 工具链 | tilelang 0.1.2 dev-root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **910B3**（20 AIC / 40 AIV） |
| benchmark | `benchmarks/ops/bench_gelu_and_mul.py::test_gelu_and_mul_manifest_bench`（5 tune + 3 smoke） |

## 1. 结论

- **有净收益，建议采纳到 wrapper**（prefill 主判 workload −43.6% / −43.1%，swiglu −35.4% / −33.2%）。
- **达标率（判据失配改判向量利用率）**：全 shape = **4/5 达标 + 1 豁免**（decode 微型豁免，不计分母）；
  可调 shape = **4/4 达标**（prefill fp16/bf16 + swiglu fp16/bf16，Task Duration 全部显著下降）。
- 硬底线「Task Duration 必须变短」满足：prefill fp16 877.20→494.38µs。
- **SOL 60% 目标不适用**：算子为 compute-bound（`aiv_vec_ratio=0.943`，GM SOL 仅 ~47%），
  按用户口径豁免规则 3 改判向量利用率。

## 2. 改动

上一轮（工具链升级前）已合入「删除冗余 NaN 恢复」（−9.3%）。本轮在 CANN 9.0.0 重采 + 追加：

| # | 优化 | 机制 | 效果 |
|---|---|---|---|
| **T1（本轮）** | **dtype 分化多项式阶数** | fp16/bf16 输出只需 ~1e-3 erf 精度，用 deg-10 minimax 替代 deg-20（fp32 保持 deg-20），Horner 从 40 减到 20 条向量 op | **prefill −36.8%**（对 nan_drop）、对基线 −42% |
| **T2（本轮）** | **loop-hoisted 常数 1 + 单条 vsub** | `w=1−u²` 用循环外 `vbrc(1.0)` 常数 buffer + 单条 `T.vsub`（替代 vmul-negate+vadd） | prefill −2.2%（A/B 方向一致，子阈值） |

- 否决项：deg-12（精度过剩，4 条 op 更多）；fold `1/√2` 进 erf 系数（**破坏精确饱和**——`c0=1/4` 与
  `t=±4` 的 fp32 精确性丢失 → `gelu(-inf)` 变 `-inf`，L1-4/Boundary 失败，见 opt_log §3 Round 2）。

## 3. msprof 对比表（最终候选 vs 基线）

| workload | dtype | baseline µs | final µs | Δlat | SOL% |
|---|---|---:|---:|---:|---:|
| llama-8b-ffn-prefill | fp16 | 877.198 | **494.380** | **−43.6%** | 46.86 |
| llama-8b-ffn-prefill | bf16 | 877.188 | **499.190** | **−43.1%** | 47.70 |
| qwen3-8b-swiglu-b32 | fp16 | 14.980 | 9.670 | −35.4% | 32.88 |
| qwen3-8b-swiglu-b32 | bf16 | 14.810 | 9.900 | −33.2% | 32.75 |
| llama-8b-ffn-decode | bf16 | 3.860 | 3.000 | 微型豁免 | 3.67 |

## 4. 精度

`python perf_opt/GeluAndMulFwdOp.py --level all` **全绿**（exit 0）：
L0/L1/L2/Boundary 全过，含 erf 饱和（±50 / 1e4 / ±inf）、大负值（−20）、signed-zero、NaN 传播、
L1-4 大 |gate| 全域（gate_scale 6/20）。max_diff 汇总：fp32 2.86e-6、fp16 7.81e-3（L1-4 大 gate）、
bf16 1.25e-1（rtol 支配）。唯一 WARN 为 L2-stress（scale-20 both halves，设计 §9.1 明言 out-of-spec、
非阻塞；deg-20 的 fp32 本身亦 WARN）。

## 5. 时间

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC+8） | ≈ 2026-10-18 01:35 → 02:30 |
| 墙钟时长 | **≈ 55 min**（含等待前序算子 GeluTanhAndMul benchmark 释放 NPU ~10 min） |
| 实测阶段 | baseline(r0) + 2 候选(r1) + fold/vsub(r2) + A/B + final(r3) + L0 全量 ≈ 45 min |

## 6. 产物

- `perf_opt/GeluAndMulFwdOp.py`（final kernel + golden + L0/L1/L2/Boundary + main --level）
- `perf_opt/gelu_and_mul_fwd_kernel.py`（同内容同步，保持当前 wrapper 的 perf_opt import 可运行）
- `perf_opt/opt_log.md`（逐轮候选 vs current best 对比表 + Final Performance Test Data + Skill Retrospective）
- `perf_opt/perf_records.jsonl`（29 条，signal-registry §5 字段契约；旧 schema 4 条归档于 `logs/legacy_perf_records_prior_run.jsonl`）
- `perf_opt/workload_inventory.json`、`perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`
- `perf_opt/run_experiments.py`、`bench_kernel.py`、`env.sh`、`experiments_r*.json`、实验分支 `*_round*.py`

## 7. ceiling / 缺口

- **compute-bound 地板**：`aiv_vec_ratio=0.943`（prefill fp16）——deg-10 多项式 erf 在 fp32 域展开为
  ~34 条向量 op，把 Vector FP32 单元打满；GM SOL 仅 47% 非瓶颈。残余 34 条向量 op 是
  「全 fp32 域 + 单次 cast + 精确 erf（无可用原生 erf）」精度契约下的 floor（fp16/bf16 可用 deg-10
  已达该 dtype 容差下的 op 数下限；fp32 仍需 deg-20）。
- **结构性加速估计 ~1.8×**（已实现的 deg-20→deg-10 + vsub）；进一步（deg-8 精度不足、tanh 近似语义错误、
  exp 型 minimax 引入 vexp/vdiv 更多 op）**远未达 2×** → 不触发 `[DESIGN_LIMIT]` 逆向反馈（perf-feedback
  两项触发条件未同时满足）。
- 未竟项：VP-2026-0033/0034/0035 的 `repro/` 最小形态待回填（时间受限，标 repro-missing）。
