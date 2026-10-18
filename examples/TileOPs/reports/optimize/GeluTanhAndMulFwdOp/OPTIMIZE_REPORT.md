# 优化报告：GeluTanhAndMulFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `GeluTanhAndMulFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = gelu_tanh(x[:,:N]) * x[:,N:]` |
| kernel（基准，只读） | `tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/`（`gelu_tanh_and_mul_fwd_kernel.py`） |
| 调优产物 | `.../gelu_tanh_and_mul_kernel/perf_opt/GeluTanhAndMulFwdOp.py`（winner = fold + hoist-vbrc，候选 `op2-bs2048`） |
| 口径 | `msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+ `visualize_data.bin` GM Read+Write SOL；数据来自 `perf_opt/perf_records.jsonl`（final 行） |
| **工具链版本戳** | tilelang 0.1.2 dev-root（AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir 1.1.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **910B3**（20 AIC / 40 AIV）/ 2026-10-18 |
| 达标线 | 原文按**判据失配改判向量利用率**：算子 compute-bound（roofline `Vec_FP32 = 99.09%`），SOL 60% 目标不适用；硬底线 = Task Duration 必须变短 |
| benchmark | `benchmarks/ops/bench_gelu_tanh_and_mul.py::test_gelu_tanh_and_mul_manifest_bench`（5 tune + 3 smoke） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`（51 条）、`workload_inventory.json`、`profiles/`、`logs/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**compute-bound（Vec_FP32 99.09%）下唯一有效杠杆是「循环不变量常数缓冲提升」——把 `T.vbrc(1.0, one32)` 从逐 tile 提到循环外（每核只填一次），prefill −3.2%；叠加上一轮的 fold vmul 后 prefill −8.2% / −8.1%。**
- **达标数（双分母）**：全 shape = **4/5 达标 + 1 豁免**（decode 微型豁免，不计分母）；可调 shape = **4/4 达标**（prefill fp16/bf16 + swiglu fp16/bf16，Task Duration 全部下降）；豁免 **1**，依据：decode bf16 Task Duration 2.14µs ≤ 5µs 微型 shape。
- 硬底线「Task Duration 必须变短」满足：prefill fp16 **215.15→197.44µs**。
- 关键收益：prefill fp16 **−8.2%**、prefill bf16 **−8.1%**、swiglu fp16 **−5.0%**、swiglu bf16 **−5.0%**。
- **SOL 60% 目标不适用**：算子为 compute-bound（roofline `Vec_FP32 = 99.09%`，GM SOL 仅 ~46%），按用户口径豁免规则 3 改判向量利用率。
- **是否建议采纳 wrapper**：**是**——有净收益（prefill 主判 −8.2% / −8.1%），无精度回退，签名为同族 drop-in。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 瓶颈 pipe | roofline `Vec_FP32 ≈ 99.09%`、`Vec_MISC ≈ 0.9%`（全 workload） | **compute-bound**：sigmoid 形式 tanh（`vexp` + `vdiv`）展开为 FP32 多项式序列，把 Vector FP32 单元打满 |
| GM SOL | prefill GM Read+Write SOL ≈ 45–50%（表内 final 45.57 / 47.71） | **非正确判据**：实测向量利用率 ≥0.9 → 按豁免规则 3 改判向量利用率 |
| 链长构成 | 14 条向量 op（fp16/bf16：2 upcast + 5 多项式 + 1 vexp + 1 vadd + 1 vbrc + 1 vdiv + 2 vmul + 1 downcast） | 唯一可削的是**逐 tile 重复填充常数 1** 的 `vbrc`（循环不变量） |
| 残余链长（win 后） | 13 条向量 op | 已是「全 fp32 域 + 单次 cast」精度契约下的 floor |
| 设计估算 vs 实测 | DESIGN 估算 prefill 下界 ≈ 226µs（带宽口径），实测 baseline 215.15µs / final 197.44µs（**低于估算下界**） | **判据失配**：把 MTE2 带宽当地板，实为向量 FP32 打满 |
| 流水是否有空间 | compute-bound 下 GM→UB 搬运已被 auto-multi-buffer 隐藏 | 软件流水无可重叠阶段 → `T.Pipelined(2)` 预期 tie（实测确认） |
| 噪声阈 | 3%（A/B tie 判据；小 shape <5% 按交错 A/B 纪律） | prefill 主判 −3.2% 已超阈值，可作为 winner 依据 |

### 2.2 方法（做了什么、为什么这样做）

上一轮（CANN 8.5.2）已合入「fold `sqrt(2/π)×(−2)` 为一次 vmul」，作为本轮起点（current best `fold`）；本轮在 CANN 9.0.0 重采 + 追加。

1. **T1（上一轮已合入，本轮复现）：fold `−2u = (g+cg³)·(−2√(2/π))`**
   - 改了什么：把 `sqrt(2/π)` 与 `(−2)` 两次连续 vmul 合并为一次系数乘法。
   - 为什么有效：少 1 条向量 op；在 Vec_FP32 99% 的 compute-bound 下直接兑现在时延上（上轮 prefill **−5.2%**，本轮复现）。
   - 正确性依据：常数合并是精确的代数等价（系数在 trace-time 折叠，仍以 fp32 常量参与计算）；L0/L1/L2/Boundary 全过。

2. **T2（本轮主杠杆）：提升循环不变量常数缓冲填充**
   - 改了什么：`T.vbrc(1.0, one32)` 从 tile 循环内提升到循环外，**每核只填一次**。
   - 为什么有效：原实现逐 tile 重复填充常数 1 缓冲（prefill 每核 ~358 次，opt_log §3），而缓冲内容与 tile 无关——外提后省掉 ~357 次/核的向量 op，prefill **−3.2%**（bf16 −3.3%）。
   - 正确性依据：缓冲内容不变（恒为 1.0），数学严格等价；`--level all` 全绿。

3. **结构参数维持**：`block_N = 2048`（baseline 默认值）。

**被否决的方向（保留）**：
- **`T.Pipelined(2)`（op1）**：compute-bound 下 GM 搬运已被 auto-multi-buffer 隐藏，流水无可重叠阶段 → 实测 **tie**，❌ 否决。
- **静态满块 + 流水（op3）**：同 op1 原因，实测 **tie**，❌ 否决（印证 PL-1.6-copy-floor：≥16M 档向量链全隐藏）。
- **block_N 扫描**：block_N ∈ {1024, 1536, 2560} 均劣于 2048（tile 过小 → 发射开销↑；过大因 UB 超限 ~50B/elem × 4096 = 200KB > 192KB 不可编译）→ ❌ 否决，维持 2048。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | `fold`（上一轮，CANN 8.5.2） | fold `sqrt(2/π)×(−2)` 为一次 vmul（上轮 −5.2%），本轮在 9.0.0 重采为起点 | prefill fp16 204.024µs / bf16 209.044µs / swiglu 5.790µs | ✅ 起点 current best |
| R1 | `op1`（pipelined） | `T.serial` → `T.Pipelined(2)` | prefill 204.024 / bf16 208.954 / swiglu 5.810µs | ❌ 否决：tie（compute-bound 无可重叠阶段） |
| R1 | `op2`（hoist vbrc） | 常数 1 缓冲 `vbrc` 提到循环外 | prefill **197.494 / 202.234µs**（−3.2% / −3.3%）、swiglu 5.660µs，L0 PASS | ✅ **winner** |
| R1 | `op3`（static+pipe） | 整除 N 走静态满块 + `T.Pipelined(2)` | prefill 204.204 / bf16 208.794 / swiglu 5.720µs | ❌ 否决：tie（同 op1） |
| R2 | `op2-bs1024` | block_N 1024 | prefill fp16 246.75 / bf16 249.95µs（2rep 中位） | ❌ 否决：大幅回退（tile 过小、发射开销↑） |
| R2 | `op2-bs1536` | block_N 1536 | prefill 230.06 / 233.45µs | ❌ 否决：回退 |
| R2 | **`op2-bs2048`** | block_N 2048（默认） | prefill **197.48 / 202.22µs** | ✅ winner（默认值维持最优） |
| R2 | `op2-bs2560` | block_N 2560 | prefill 200.53 / 205.40µs | ❌ 否决：略回退 |
| R3 | merged | fold + op2 | prefill 197.444 / 202.274µs、swiglu 5.680 / 5.760µs、decode 2.140µs，`--level all` 全绿 | ✅ final |

**合并纪律**：op1/op3 无增益不合并；合并版 = `fold + op2`，并复测全部 tune workload（同一最终候选）。

## 3. 优化结果（msprof 对比表）

| workload | dtype | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---|---:|---:|---:|---:|---:|---|
| llama-8b-ffn-prefill | fp16 | 215.154 | **197.444** | **−8.2%** | 原文未记录 | 45.57 | ✅ |
| llama-8b-ffn-prefill | bf16 | 220.114 | **202.274** | **−8.1%** | 原文未记录 | 47.71 | ✅ |
| qwen3-8b-swiglu-b32 | fp16 | 5.980 | 5.680 | −5.0% | 原文未记录 | 22.04 | ✅ |
| qwen3-8b-swiglu-b32 | bf16 | 6.060 | 5.760 | −5.0% | 原文未记录 | 22.99 | ✅ |
| llama-8b-ffn-decode | bf16 | 2.160 | 2.140 | 微型豁免 | 原文未记录 | 2.43 | 豁免 |

smoke（16×128，fp32/fp16/bf16）：仅精度回归，不做性能采集。

**测量口径与噪声**：每点独立 `msprof op --kernel-name=main` 进程；原表只列单一 SOL%（即 final SOL%），base SOL% 原文未记录。decode 2.14µs ≤ 5µs → 微型 shape 豁免；swiglu-b32（5.68/5.76µs）为 small-shape（<20µs），差异 <5% 按交错 A/B 纪律处理（op2 在两 workload 上方向一致 −2.2% / −2.3%）。

## 4. 精度

- `python perf_opt/GeluTanhAndMulFwdOp.py --level all` **全绿**（exit 0）：L0/L1/L2/Boundary 全过。
- 覆盖：tanh 饱和（±50）、大负值（−20）、signed-zero、inf/nan 边界、L1-4 大 |gate| 全域（`T.vtanh` 盲区 + Developer exp 溢出点）。
- max_diff 汇总：fp32 **3.05e-5**、fp16 **1.25e-1**（L2-stress，rtol 支配，非阻塞）、bf16 **2.50e-1**（同上）。
- smoke：3 个 smoke 精度 PASS。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC+8） | ≈ 2026-10-18 00:30 → 01:30 |
| 墙钟时长 | **≈ 60 min**（含等待前序算子 Hardsigmoid benchmark 释放 NPU ~15 min） |
| 实测阶段 | baseline(r0) + 3 候选(r1) + block 扫(r2) + final(r3) + L0 全量 ≈ 35 min |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/GeluTanhAndMulFwdOp.py`（final kernel + golden + L0/L1/L2/Boundary + main --level） |
| wrapper 兼容同步 | `perf_opt/gelu_tanh_and_mul_fwd_kernel.py`（同内容同步，保持当前 wrapper 的 perf_opt import 可运行） |
| 调优记录 | `perf_opt/opt_log.md`（逐轮候选 vs current best 对比表 + Final Performance Test Data + Skill Retrospective） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（51 条，signal-registry §5 字段契约；旧 schema 3 条归档于 `logs/legacy_perf_records_prior_run.jsonl`） |
| 采集工件 | `perf_opt/workload_inventory.json`、`profiles/`（raw msprof）、`logs/`、`run_experiments.py`、`bench_kernel.py`、`env.sh`、`experiments_r*.json` |
| 能力缺口 | 原文未记录 |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（建议采纳，由 conductor 执行） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | 原文未记录（VP-2026-0030 的 `repro/` 最小形态待回填，标 repro-missing） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项与依据**：
  1. **decode bf16**：Task Duration 2.14µs ≤ 5µs → 微型 shape 豁免。
  2. **swiglu-b32**（5.68/5.76µs）：近微型，现场复核未达 ≤5µs 豁免线，但为 small-shape（<20µs），差异 <5% 按交错 A/B 纪律处理；prefill 主判 −3.2% 超出 3% 噪声阈决定 winner。
  3. **判据失配**：`Vec_FP32 = 99.09% ≥ 0.9` → compute-bound，改用向量利用率判据；硬底线 Task Duration 变短（已满足）。
- **ceiling 证据（compute-bound 地板）**：`Vec_FP32 = 99.09%`——sigmoid 形式 tanh（`vexp`+`vdiv`）在 fp32 域展开为 FP32 多项式序列，把向量 FP32 单元打满；GM SOL 仅 46% 非瓶颈。残余 13 条向量 op 是「全 fp32 域 + 单次 cast」精度契约下的 floor（fp32 需精确 sigmoid，无法用 `vrsqrt` 等近似指令）。
- **结构性加速估计 ~10–20%**（替换 sigmoid 为 minimax 多项式/分段近似，代价是 fp32 精度与 ±inf 边界语义），**远未达 2×** → 不触发 `[DESIGN_LIMIT]` 逆向反馈（perf-feedback 两项触发条件未同时满足）。
- **贴线风险**：无（判据已改判向量利用率，不存在 SOL 贴线项）。
- **缺口**：VP-2026-0030（hoist-vbrc）的 `repro/` 最小形态待回填（时间受限，标 repro-missing）。

## 8. 建议

1. **采纳 `perf_opt/GeluTanhAndMulFwdOp.py`**：prefill 主判 −8.2% / −8.1%，swiglu −5.0%，L0–Boundary 全绿，无回退。
2. 后续动作：若把 sigmoid 形式 tanh 换成 minimax 多项式近似，须先解决 fp32 精度与 ±inf 边界语义（预计仅 10–20% 收益）；工具链升级后复核 `Vec_FP32` 占比与 `vtanh` 展开形态。
3. **未解决项与原因**：残余 13 条向量 op 属精度契约 floor，参数/tiling/流水（op1、op3 已证 tie）均不可解；VP-2026-0030 的 repro 尚未回填。
