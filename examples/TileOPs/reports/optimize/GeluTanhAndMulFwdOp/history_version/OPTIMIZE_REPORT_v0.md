# 优化报告：GeluTanhAndMulFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GeluTanhAndMulFwdOp`（`y = gelu_tanh(x[:,:N]) * x[:,N:]`）；单 kernel/单 func/纯 Vector（AIV） |
| kernel | `tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/` |
| 口径 | `msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+ `visualize_data.bin` GM Read+Write SOL |
| 工具链 | tilelang 0.1.2 dev-root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **910B3**（20 AIC / 40 AIV） |
| benchmark | `benchmarks/ops/bench_gelu_tanh_and_mul.py::test_gelu_tanh_and_mul_manifest_bench`（5 tune + 3 smoke） |

## 1. 结论

- **有净收益，建议采纳到 wrapper**（prefill 主判 workload −8.2% / −8.1%，swiglu −5.0%）。
- **达标率（判据失配改判向量利用率）**：全 shape = **4/5 达标 + 1 豁免**（decode 微型豁免，不计分母）；
  可调 shape = **4/4 达标**（prefill fp16/bf16 + swiglu fp16/bf16，Task Duration 全部下降）。
- 硬底线「Task Duration 必须变短」满足：prefill fp16 215.15→197.44µs。
- **SOL 60% 目标不适用**：算子为 compute-bound（roofline `Vec_FP32 = 99.09%`，GM SOL 仅 ~46%），
  按用户口径豁免规则 3 改判向量利用率。

## 2. 改动

上一轮（CANN 8.5.2）已合入「fold sqrt(2/π)×(−2) 为一次 vmul」。本轮在 CANN 9.0.0 重采 + 追加：

| # | 优化 | 机制 | 效果 |
|---|---|---|---|
| T1（上轮） | fold `−2u = (g+cg³)·(−2√(2/π))` | 合并两次连续 vmul 为一次 | prefill −5.2%（上轮，本轮复现） |
| **T2（本轮）** | **提升循环不变量常数缓冲填充** | `T.vbrc(1.0, one32)` 从逐 tile 提到循环外，每核只填一次（省 ~357 次/核） | **prefill −3.2%** |

- 否决项：`T.Pipelined(2)`（op1）、静态满块+流水（op3）——compute-bound 下 GM 搬运已被
  auto-multi-buffer 隐藏，流水无可重叠阶段，实测 tie；block_N∈{1024,1536,2560} 均劣于 2048（默认值维持最优）。

## 3. msprof 对比表（最终候选 vs 基线）

| workload | dtype | baseline µs | final µs | Δlat | SOL% |
|---|---|---:|---:|---:|---:|
| llama-8b-ffn-prefill | fp16 | 215.154 | **197.444** | **−8.2%** | 45.57 |
| llama-8b-ffn-prefill | bf16 | 220.114 | **202.274** | **−8.1%** | 47.71 |
| qwen3-8b-swiglu-b32 | fp16 | 5.980 | 5.680 | −5.0% | 22.04 |
| qwen3-8b-swiglu-b32 | bf16 | 6.060 | 5.760 | −5.0% | 22.99 |
| llama-8b-ffn-decode | bf16 | 2.160 | 2.140 | 微型豁免 | 2.43 |

## 4. 精度

`python perf_opt/GeluTanhAndMulFwdOp.py --level all` **全绿**（exit 0）：
L0/L1/L2/Boundary 全过，含 tanh 饱和（±50）、大负值（−20）、signed-zero、inf/nan 边界、
L1-4 大|gate|全域（T.vtanh 盲区 + Developer exp 溢出点）。max_diff 汇总：fp32 3.05e-5、
fp16 1.25e-1（L2-stress，rtol 支配，非阻塞）、bf16 2.50e-1（同上）。

## 5. 时间

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC+8） | ≈ 2026-10-18 00:30 → 01:30 |
| 墙钟时长 | **≈ 60 min**（含等待前序算子 Hardsigmoid benchmark 释放 NPU ~15 min） |
| 实测阶段 | baseline(r0) + 3 候选(r1) + block 扫(r2) + final(r3) + L0 全量 ≈ 35 min |

## 6. 产物

- `perf_opt/GeluTanhAndMulFwdOp.py`（final kernel + golden + L0/L1/L2/Boundary + main --level）
- `perf_opt/gelu_tanh_and_mul_fwd_kernel.py`（同内容同步，保持当前 wrapper 的 perf_opt import 可运行）
- `perf_opt/opt_log.md`（逐轮候选 vs current best 对比表 + Final Performance Test Data + Skill Retrospective）
- `perf_opt/perf_records.jsonl`（51 条，signal-registry §5 字段契约；旧 schema 3 条归档于 `logs/legacy_perf_records_prior_run.jsonl`）
- `perf_opt/workload_inventory.json`、`perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`
- `perf_opt/run_experiments.py`、`bench_kernel.py`、`env.sh`、`experiments_r*.json`

## 7. ceiling / 缺口

- **compute-bound 地板**：`Vec_FP32 = 99.09%`——sigmoid 形式 tanh（`vexp`+`vdiv`）在 fp32 域展开为
  FP32 多项式序列，把向量 FP32 单元打满；GM SOL 仅 46% 非瓶颈。残余 13 条向量 op 是
  「全 fp32 域 + 单次 cast」精度契约（fp32 需精确 sigmoid，无法用 `vrsqrt` 等近似指令）下的 floor。
- **结构性加速估计 ~10–20%**（替换 sigmoid 为 minimax 多项式/分段近似，代价是 fp32 精度与 ±inf 边界语义），
  **远未达 2×** → 不触发 `[DESIGN_LIMIT]` 逆向反馈（perf-feedback 两项触发条件未同时满足）。
- 未竟项：VP-2026-0030（hoist-vbrc）的 `repro/` 最小形态待回填（时间受限，标 repro-missing）。
