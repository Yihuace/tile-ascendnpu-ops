# GLADecodeOp Stage 4 调优日志（opt_log）

- 算子：GLADecodeOp（single-token gated linear attention decode，Vector 单元 / Developer 模式）
- 基准 kernel：`_gla_decode_kernel.py`（k_tile 默认 16）；最终 kernel：`perf_opt/GLADecodeOp.py`（k_tile 默认 32）
- 平台：Ascend910B3，CANN 9.0.0，tilelang dev-root build（7a4f77e184）；物理 AIV 核数 = `NPUUtils.get().get_aicore_num()*2` = 40
- 性能口径：`msprof op Task Duration(us)`（唯一时延口径）+ `--aic-metrics=Default,Roofline` 的 GM Read+Write SOL（Ratio×100）
- 性能目标（user_requirement）：Task Duration + GM Read+Write SOL；按 AI 分类判据（访存>=70% / 算力>=60% / Mix>=40%），不伪达标

---

## Performance Test Data

benchmark 入口：`benchmarks/ops/bench_gla_decode.py::test_gla_decode_manifest_bench`，经 manifest `tileops/manifest/linear_attention.yaml` `GLADecodeOp.workloads` 展开。

### tune workload（5 条，全 bfloat16，DK=DV=128）

| kernel_id | workload_id | baseline_us (k16) | final_us (k32) | 提速 | GM SOL baseline | GM SOL final |
|---|---|---|---:|---:|---:|---:|
| gla_decode_kernel | serving-b1-h8-bfloat16 | 9.01 | 6.82 | 24.3% | 2.52% | 3.18% |
| gla_decode_kernel | serving-b1-h32-bfloat16 | 11.03 | 8.63 | 21.8% | 8.23% | 10.15% |
| gla_decode_kernel | serving-b1-h64-bfloat16 | 18.77 | 14.71 | 21.6% | 9.53% | 11.68% |
| gla_decode_kernel | serving-b8-h32-bfloat16 | 57.59 | 41.31 | 28.3% | 12.30% | 16.77% |
| gla_decode_kernel | serving-b8-h64-bfloat16 | 106.31 | 76.44 | 28.1% | 13.45% | 18.25% |

（final 数值取 `phase=final` round 3，在最终 `GLADecodeOp.py` 上测得；round 2 复测同文件旧版本偏差 <2%，见下。）

### smoke / skip 排除

- `smoke-b2-h4-d64-float16` / `smoke-b2-h4-d64-bfloat16`：manifest idx 0 smoke 案例，仅精度回归，不采性能。最终 `precision_pass=true`（内嵌 `--level all` 9 案例全过，含这两条）。
- 无 `skipped` 项。

### 设计估算 vs 实测偏差（D-2 回填）

本算子无 DESIGN.md（direct-implementation 迁移），无 §1.6.0 设计期估算下界行可对照；跳过该偏差行。

---

## Iteration Log

### 度量管线

- `--aic-metrics=Default,Roofline` + `--kernel-name` 过滤（VP-2026-0081 规则）——Roofline 集产出 visualize_data.bin 预计算 Ratio；kernel-name 过滤避免把 ZerosLike 等辅助 op 当目标。
- **kernel-name 陷阱**：bf16/fp16 的 jitted kernel 名为 `_gla_decode_transit_main`（transit 变体）、fp32 为 `_gla_decode_main`，**不是** `main`。上一轮会话已把 `run_experiments.py` 的 `KERNEL_NAME` 修正为 `_gla_decode_transit_main`；本条在 Skill Retrospective 中登记。

### Round 1 — k_tile 扫描（候选：ktile8、ktile32；base：baseline k_tile=16）

现象：baseline advice 全为 `latency bound:pipeline caused`，GM SOL 仅 2.5~13.5%。k_tile 是唯一编译期参数，`_ceil_ktile_fit` 的 150KB 启发式把 k_tile 锁在 16，但实际 UB=192KB（实测 k_tile=32 可编译）。

候选 vs current best（Task Duration us；base=baseline k16）：

| workload | base(k16) | ktile8 | ktile32 | 结论 |
|---|---:|---:|---:|---|
| serving-b1-h8-bf16 | 9.01 | 12.94 | 6.77 | ktile32 胜 |
| serving-b1-h32-bf16 | 11.03 | 14.60 | 8.47 | ktile32 胜 |
| serving-b1-h64-bf16 | 18.77 | 26.72 | 14.44 | ktile32 胜 |
| serving-b8-h32-bf16 | 57.59 | 84.40 | 42.14 | ktile32 胜 |
| serving-b8-h64-bf16 | 106.31 | 156.32 | 77.06 | ktile32 胜 |

结论：k_tile=32 全 workload 胜（+23~27%）；k_tile=8 回退（+43~47%）。k_tile=32 把串行 K-tile 循环从 8 次降到 4 次，减少 loop 开销与每 tile 的 reduce/barrier。k_tile=64 不成立（fp32 缓冲 294KB+ 超 192KB UB）。

### Round 2 — 合并 winner 并 final 复测（winner_k32 = GLADecodeOp.py，k_tile 默认 32）

改动：`_ceil_ktile_fit` 的 UB 预算 150KB → 192KB（UB 实容量，constants.md），使 DK=DV=128 自动选 k_tile=32；暴露模块级常量 `TUNED_K_TILE=32` 供 wrapper 切换块引用；补齐内嵌分层测试（run_L0/L1/L2/boundary + main `--level`，gate 4 `S4-TEST-LEVELS` 要求）。语义不变（纯 tiling 变化 + 追加测试代码），L0/L1 9 案例全过。

candidate（final 复测）vs current best 对比表（Task Duration us；final=round 3）：

| workload | base(k16) | final(k32) | 提速 | GM SOL 变化 |
|---|---:|---:|---:|---|
| serving-b1-h8-bf16 | 9.01 | 6.82 | 24.3% | 2.52%→3.18% |
| serving-b1-h32-bf16 | 11.03 | 8.63 | 21.8% | 8.23%→10.15% |
| serving-b1-h64-bf16 | 18.77 | 14.71 | 21.6% | 9.53%→11.68% |
| serving-b8-h32-bf16 | 57.59 | 41.31 | 28.3% | 12.30%→16.77% |
| serving-b8-h64-bf16 | 106.31 | 76.44 | 28.1% | 13.45%→18.25% |

合并后全 workload 无回退（均提速），smoke 精度通过 → 采纳 winner_k32 为 current best。

### 未探索 / 已评估但未执行的方向（诚实披露）

- **persistent 分核（grid=min(B*H,40) 核内串行）**：对超发且非 40 整数倍的 workload（b1-h64 64 核负载不均 ~25%、b8-h32 256 核 ~9.4%）理论可回收 ~10-20%，但需运行时条件 guard（`head < B*H`）+ 静态循环边界，且对欠订 workload（b1-h8/b1-h32）会反噬（需按 shape 分派）。收益有限且风险高，未执行。
- **transit 路径消除（bf16/fp16 直接运算）**：bf16 无 Vector 算术，fp32 累加是 GPU 契约 + 精度要求，5 条 tune workload 全 bf16，不可消除。

---

## Final Summary

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| gla_decode_kernel | serving-b1-h8-bfloat16 | 9.01 | 6.82 | winner_k32 |
| gla_decode_kernel | serving-b1-h32-bfloat16 | 11.03 | 8.63 | winner_k32 |
| gla_decode_kernel | serving-b1-h64-bfloat16 | 18.77 | 14.71 | winner_k32 |
| gla_decode_kernel | serving-b8-h32-bfloat16 | 57.59 | 41.31 | winner_k32 |
| gla_decode_kernel | serving-b8-h64-bfloat16 | 106.31 | 76.44 | winner_k32 |

（final 时延与 perf_records.jsonl `phase=final` round 3 记录一致；全部来自同一 final 候选 `winner_k32` = `perf_opt/GLADecodeOp.py`）

### 性能目标达成情况（如实，不伪达标）

- **AI 分类**：AI ≈ 3.64 Ops/Byte（GM Read+Write 口径），`AI×GM_bw(1.75TB/s)=6.4TOps/s` 远低于 Vector 算力 → **访存型（memory-bound）**，判据阈值 **SOL ≥ 70%**。
- **GM Read+Write SOL 实测**：final 为 3.18% ~ 18.25%，**全部未达 70%**。msprof advice 全程 `latency bound:pipeline caused`（非 memory bound / compute bound）。
- **Task Duration**：+21.6% ~ +28.3%（k_tile=32 相对基线 k_tile=16），是本轮可证实的优化收益。
- **结论**：Task Duration 提速方向正确；SOL 70% 目标**未达成**。根因是算子为 latency-bound（每 (b,h) 一个核、decode 阶段并行度 = B×H 有限、逐头 state 递归存在串行依赖 + 小 per-head 工作集，流水线无法填满）。参数级（k_tile）已探到 UB 上限（32），分核级（persistent）收益有限且风险高，未见 >2x 的结构性加速方案（rank-1 外积 k⊗v 与 q·state matvec 不宜 Cube），故**不产出 [DESIGN_LIMIT]**（触发条件②「结构性加速 >2x」无实测/文献依据支撑）。

### 中止原因

`success`（Task Duration 主目标以参数级最优收束）——SOL 70% 副目标未达成已如实披露，不作为中止条件；未耗尽 max_rounds/max_experiments，未 plateau（仅 2 轮，k_tile=32 单点胜出后方向收敛）。

---

## Skill Retrospective

按 vp_type 标注，供任务终态 tilelang-skill-evolver 蒸馏。

- **D（实测数据）— `_ceil_ktile_fit` UB 预算过保守**：该 Vector 算子的 K-tile 启发式用 150KB 上限，实际 UB=192KB（constants.md「UB=192KB/AIV」），导致 DK=DV=128 被锁在 k_tile=16、错失 k_tile=32 的 ~22-28% 收益。证据：本任务 ktile32 全 workload 实测（perf_records.jsonl round1 candidate ktile32）+ 编译成功（162KB 手工 / ~182KB 经 1.10-1.12× BishengIR 膨胀 < 192KB）。toolchain_stamp：tilelang dev-root 7a4f77e184 + CANN 9.0.0 + Ascend910B3 / 2026-10-17。建议：向量算子 K-tile 容量启发式应以 192KB 为上限（并计入 ~1.12× 膨胀 + 小 buffer），勿用经验 150KB。
- **P（模式方法）— msprof kernel-name 与 transit 变体名**：`@tilelang.jit(out_idx=...)` 返回的 prim_func 名（`_gla_decode_transit_main` / `_gla_decode_main`）才是 msprof `--kernel-name` 应过滤的名字，不是 `main`。证据：本任务中断会话日志（`--kernel-name=main` 时 "Kernel skipped: not selected via --kernel-name" + "No profiling data dumped"）。与 VP-2026-0081（kernel-name 过滤缺一不可）同源，建议并入该条或新立 transit 变体名子条目。toolchain_stamp：同上。
- **C（案例索引）— GLADecodeOp 单 token decode 为 latency-bound**：逐头 state 递归 + decode 并行度=B×H 有限，GM SOL 上限 ~18%（k_tile=32），达 70% 访存 SOL 需结构性改变（非参数级可解）。证据：perf_records.jsonl 全部 advice="latency bound:pipeline caused"。
- 回写 pattern-library：本任务结论由 evolver 蒸馏合入，conductor 不直接改 skill 文件。
