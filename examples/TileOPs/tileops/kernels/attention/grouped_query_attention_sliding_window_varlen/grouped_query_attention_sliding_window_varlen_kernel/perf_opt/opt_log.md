# `_gqa_sw_fwd_varlen_kernel` Stage 4 调优日志（optimize · mode=full）

> 唯一时延口径：**`msprof op` `Task Duration(us)`**（kernel-only，median of 20 launches，warm-up=5）。不采集 NPU event / runner 端到端时间。
> 目标类型：`best_effort`（无硬性数值门槛）；噪声阈值 3%；最大迭代 10 轮；`perf_opt/` 为本任务唯一写域。

## 0. 任务与测量口径

| 项 | 值 |
|---|---|
| project / op | `grouped_query_attention_sliding_window_varlen` / `_gqa_sw_fwd_varlen_kernel` |
| 基准 kernel（只读） | `_gqa_sw_fwd_varlen_kernel.py`（Expert 模式 · 两相位 · 双 Scope · 1-D persistent `T.Kernel(20)`） |
| 工具链戳 | tilelang `0.1.2+cebcec9dd203208e30657c667cdc87d6526d1829`（commit `013dbbf`）+ CANN 9.0.0 + Ascend910B3 ×1（20 AIC / 40 AIV） |
| 采集入口（自建） | `perf_opt/bench.py`（直调 kernel 工厂 + 预分配 ws，**绕开 `Op.forward()`**，避免 host 侧校验算子污染采集窗口，DESIGN §12.1） |
| 采集/对账脚本 | `perf_opt/run_experiments.py`（T-4 批处理）、`perf_opt/parse_profile.py`、`perf_opt/probe_f16_chain.py` |
| captured op | `_gqa_sw_fwd_varlen_main_mix_aic`（每个 profile 目录下仅此一个 op 目录 ⇒ 采集有效，非框架小算子） |
| 原始 profile | `perf_opt/profiles/{baseline,candidate,merged,final,probe}/…/OPPROF_*` |
| 过程日志 | `perf_opt/logs/{baseline,round1..round5,final}/*.log` |

**基线的自建说明（DESIGN §12.1 落实）**：`examples/TileOPs/reports/…20261016_{220224,223115}_…` 两次 report 均为 `partial`（`invalid_reason = "unfiltered capture contains 6 distinct ops"`，3.26–4.30 µs 实为 host 侧 `GreaterEqual`），**不可作基线**。本轮以 `bench.py` 直调 kernel 工厂重测基线，profile 中只出现目标 kernel（无 host 校验算子），全部 `n=20` 条 `OpBasicInfo` 记录。
本轮结束后端到端 report 预计仍为 `partial`（评测框架口径问题，属 `tileops/reporting` 域，不在本任务写域）；按 optimize §4 如实报告，不作为失败。

## 1. Workload Inventory（由 benchmark 展开，非 DESIGN/manifest 代填）

入口：`examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_sliding_window_varlen.py::test_gqa_sliding_window_varlen_fwd_bench`
（`_manifest_params(load_workloads("GroupedQueryAttentionSlidingWindowVarlenFwdOp"))`，5 个 manifest workload × 2 dtype = **10 个参数化案例**，`id = "{label}-{dtype}"`；`q_lens/k_lens` 按 `total_q//batch` 均分，与 inlined `gqa_sliding_window_varlen_args` 一致）。

| workload_id | kind | batch | q_len×B (total_q) | k_len×B | H / H_kv / D | causal / wl / wr | 依据 |
|---|---|---|---|---|---:|---:|---|---|
| `gqa-sw-varlen-fwd-smoke-float16` / `-bfloat16` | **smoke** | 2 | 128×2 = 256 | 256 | 8 / 2 / 64 | 1 / −1 / −1 | label token `smoke`（独立 `-` 分隔词） |
| `llama-3.1-8b-short-w256-float16` / `-bfloat16` | **tune** | 4 | 512×4 = 2048 | 2048 | 32 / 8 / 128 | 1 / 256 / −1 | 非 smoke 的 full 案例 |
| `llama-3.1-8b-long-w1024-float16` / `-bfloat16` | **tune** | 4 | 2048×4 = 8192 | 8192 | 32 / 8 / 128 | 1 / 1024 / −1 | 同上 |
| `llama-3.1-70b-short-w256-float16` / `-bfloat16` | **tune** | 4 | 512×4 = 2048 | 2048 | 64 / 8 / 128 | 1 / 256 / −1 | 同上 |
| `llama-3.1-70b-long-w1024-float16` / `-bfloat16` | **tune** | 4 | 2048×4 = 8192 | 8192 | 64 / 8 / 128 | 1 / 1024 / −1 | 同上 |

- **tune = 8**（4 shape × fp16/bf16），**smoke = 2**（仅精度回归，不采性能 baseline、不选 winner），**skipped = 0**（benchmark 无显式 skip）。
- 全部案例调用同一个 Stage 3 kernel（`_gqa_sw_fwd_varlen_kernel` → 单 `@T.prim_func` `_gqa_sw_fwd_varlen_main`），故 inventory 只有一个 `kernel_id`。
- 完整机器可读清单：`perf_opt/workload_inventory.json`。

**调优顺序（按实测负担排序，只影响先后、不跳过任何 workload）**：
`70b-long-fp16`（7514.88 µs，最大计算量 206.4 GFLOP / 30720 执行块，关注 shape）→ `70b-long-bf16` → `8b-long-fp16/bf16` → `70b-short` → `8b-short`。
理由：长 shape 的计算量占总量 ~78%，且 §11 指出长域稳态画像才代表结构地板；短 shape 的每块摊薄更差（per-task 开销占比高），在结构定型后单独复核。

## 2. Performance Test Data（phase=baseline，round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir（相对 `perf_opt/`） |
|---|---|---|---|---|---:|---|---|
| `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main` | `llama-3.1-70b-long-w1024-float16` | baseline | `_gqa_sw_fwd_varlen_main` | `_gqa_sw_fwd_varlen_main_mix_aic` | 7514.88 | valid (n=20) | `profiles/baseline/round0/baseline_8ba8496a/llama-3.1-70b-long-w1024-float16_868712b3/` |
| 同上 | `llama-3.1-70b-long-w1024-bfloat16` | baseline | 同上 | 同上 | 7422.79 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-70b-long-w1024-bfloat16_327dd5…/` |
| 同上 | `llama-3.1-8b-long-w1024-float16` | baseline | 同上 | 同上 | 3759.49 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-8b-long-w1024-float16_698e82da/` |
| 同上 | `llama-3.1-8b-long-w1024-bfloat16` | baseline | 同上 | 同上 | 3755.83 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-8b-long-w1024-bfloat16_4ec7cd3…/` |
| 同上 | `llama-3.1-70b-short-w256-float16` | baseline | 同上 | 同上 | 1167.93 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-70b-short-w256-float16_0547c3d…/` |
| 同上 | `llama-3.1-70b-short-w256-bfloat16` | baseline | 同上 | 同上 | 1168.24 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-70b-short-w256-bfloat16_4ee474…/` |
| 同上 | `llama-3.1-8b-short-w256-float16` | baseline | 同上 | 同上 | 613.54 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-8b-short-w256-float16_240e8cea/` |
| 同上 | `llama-3.1-8b-short-w256-bfloat16` | baseline | 同上 | 同上 | 608.97 | valid (n=20) | `profiles/baseline/round0/…/llama-3.1-8b-short-w256-bfloat16_b46dc9d…/` |

命令模板（全部 tune workload 同形，见 `logs/baseline/round0/*.log` 首行 `# 命令`）：

```bash
msprof op --kernel-name=_gqa_sw_fwd_varlen_main \
  --output=profiles/baseline/round0/baseline_8ba8496a/<workload_id> \
  --launch-count=20 --warm-up=5 --dump=off --aic-metrics=Default \
  python bench.py ../_gqa_sw_fwd_varlen_kernel.py --case <case> --dtype <dtype> \
    --config 64,64,1 --msprof-loop 30
```

**记录说明（留痕）**：round-0 的 `8b-short fp16/bf16` 各有两行记录（`04:30:29` / `04:34:42`，612.17/608.49 与 613.54/608.97）。原因是后台 runner 被启动了两次（第一次 shell 超时但进程组存活），两次测量相差 ≤0.25%，本节与 Final Summary 采用**后一组（runner 自身汇总表那一组）**。append-only 契约下两行均保留、未改写。

**round 10 基线重测（记录迁移配套）**：上一 session 的 `profiles/`、`logs/`、`bench.py`、`workload_inventory.json` 已丢失且 `perf_records.jsonl` 的 final sha 与磁盘文件不符（见 §8 记录迁移段），本轮用同口径（`--launch-count=20 --warm-up=5`，median；`msprof op Task Duration`）对 **Stage 3 文件**（sha `0abfc081…`）与**现行 current best**（sha `d98cc867…`）在本 session 重测全部 8 个 tune workload：baseline 3777.40 / 3749.12 / 7566.28 / 7505.22 / 615.80 / 612.58 / 1181.98 / 1187.18 µs，final 2019.76 / 2036.34 / 4000.44 / 4041.12 / 346.88 / 350.88 / 642.48 / 647.30 µs（顺序：8b-long fp16/bf16、70b-long fp16/bf16、8b-short fp16/bf16、70b-short fp16/bf16）。与上表历史值差异 ≤1.0%（跨 session 漂移），结论不变；本节保留历史原始记录，**对账以 §8 Final Performance Test Data 为准**。

**smoke / skip 排除说明**：`gqa-sw-varlen-fwd-smoke-*`（2 例）仅作精度回归（`kind=smoke`），不采性能基线、不选 winner、不参与性能回退门禁；`skipped` 0 例。smoke 的最终精度结果见 §8。

## 3. Phase 0：上下文与首轮必查项

1. **版本失效核对（E-5）**：`kb_stale_check.py` 当前 `stale_count=266`（整库，`current_stamp = tilelang 0.1.2+cebcec9d…`）。本任务直接依赖的两条在清单内且已就地重验/重测：
   - `PL-1.9-blockwidth`（stale，910B2C/CANN8.5.0 口径）→ 本轮以 bm/bn 扫描（§5 round 3/4）**同口径重测**，结论见下；
   - `PL-1.11-causal-mask-scalartrap`（stale 清单内，但带 2026-10-16 跨 SoC 910B3 重验戳）→ 本轮**再次重验**：算术惩罚主选的收益在本 kernel 上被量化（§6 裁决），非仅"可编译"。
   - 同族预热条目 `CASE-gqa-sliding-window-migration` / `PL-1.12-task-pipeline-depth2` 的 origin_task = 定长版姊妹（同日、同 commit `cebcec9d`），其结构结论直接构成本轮 round 1 的候选来源（非同族泛化）。
2. **算子类型判定：MIX（Expert 双 Scope）**。`T.Scope("Cube")`（`T.gemm` ×2/块 + fixpipe 出数）+ `T.Scope("Vector")`（softmax/掩码/累加链）。依据 `AGENTS.md` 的 MixCV 触发规则与 DESIGN §2（P-V1 变体 B Developer BLOCKED / 变体 C Expert PASS）。**不退回 Developer**（`CG-2026-0022` / `VP-2026-0240/0246` 的适用域是 Developer auto CV-split，与本形态无关）。
3. **向量化轴与布局重估（首轮必查项）**：`per-baseline` msprof 显示 `aiv_scalar_ratio=0.324`（70b-long，未过 50% 强制换轴线）、`aiv_vec_ratio=0.567`，且 V-P 的向量链已在**列连续轴**（`[rm,bn]` 行主序，lane = 列轴，无 repack；DESIGN §1.6.3 方案 #1）。本轮**维持该布局**：`T.transpose` 的毒化税（VP-D6，2.6×）与 UB 硬上限否决了列主序/单 AIV 全行两条备选；实测到的真正瓶颈是**per-block 向量 op 数**与**跨引擎握手**，不是布局（round 2 的 −36% 只在 op 数变化时出现，布局未动）。
4. **编译器陷阱版本戳与 origin_task 核对**：`TRAP-npuir-single-primfunc-multi-kernel`（本算子单 `T.Kernel(20)` + `NUM_KERNELS` 逻辑分派，未新增第二 `T.Kernel`）、`TRAP-npuir-parallel-runtime-minmax` / `-vcast-shape-mismatch` / `-parallel-scalar-store-race`（2026-10-16 同工具链 verified）均未触发。
5. **同族既有实现检索**：定长版姊妹 `grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/perf_opt/`（同日 Stage 4 收束，`perf_records.jsonl` 21 行、winner `r3_op1_pipe2`），构成本轮 round 1 的直接先例；其 structure diff 见 §10 的 T-1 参照检查。

## 4. §11 ② 910B3 常数校准（本节为 §11 明确要求的产出）

以 baseline/final 的圆整画像（70b-long fp16）反解，并给出可复用判据：

| 常数 | 本任务实测/反解值 | 反解过程 | 与既有条目对照 |
|---|---|---|---|
| **向量 op 代价（[half,bn]=[32,256] f32 tile）** | **≈0.61 µs/op·per-AIV** | round 2：`r2_op2_minfold` 相对 merged base 在**全块**多跑 5 个 `[32,256]` f32 op → +1175 µs；blocks/AIV = 15360/40 = 384 ⇒ 1175/(384×5) = 0.61 µs | `CONST-vector-launch-overhead` 的 0.5 µs/op（910B2C）同量级；本任务把该常数扩展到 **910B3 + [32,256] f32 成块 op**（成块 op 也近似按"每 op 固定开销"计费，与条目"成块 vmul/vreduce 不适用 0.5µs"的收窄口径**部分冲突**，见 §10 回写） |
| **f16/f32 载体对同一 op 的影响** | f32 S→f16 S（同链）省 1 个 `[32,256]` vcast ⇒ −3.1%（AB 裁决）；O 载体 f32 化省 1 个 `[32,128]` vcast ⇒ +0.08%（噪声内） | round 3/4 | 与"代价 ∝ tile 字节数"一致：`[32,256]` 的 vcast 约为 `[32,128]` 的 2 倍 |
| **执行块密度（长域）** | 8b-long 15360 块 / 20 AIC = **768 块/AIC**；块均 **2.58 µs**（AIC 侧）/ 2.61 µs（AIV 侧） | final 2003 µs ÷ 768 | DESIGN §11 预期 15360 块 ✓（§1.6.0 R3 复算正确） |
| **AIC 侧标量管占比（长域）** | `aic_scalar_ratio` **0.955**（final，8b-long），`aic_cube_ratio` 0.242 | `parse_profile.py` | 印证 `PL-1.9-hardlimits`「flag 等待/标量自旋占据 AIC」——pipe2 后 Cube 侧仍是标量管饱和 |
| **fabric 有效带宽（ws 往返）** | 不作独立结论：`GM_to_L1 126.2 MB`、`L0C_to_GM 61.4 MB`、`GM_to_UB 30.7 MB` 的带宽占用率均 < 23%，长域未被 fabric 封顶 | final mem 段 | 与 `CONST-fabric-bw ≥1.73 TB/s` 不矛盾（本算子未触顶） |
| **向量发射率（整算子）** | 长域 `aiv_vec_ratio` 0.594、`aiv_scalar_wait_time` 1143 µs（57% 壁钟）——final 仍是"等 Cube 供数"为主 | final pipe 段 | pipe2 已把等待从 baseline 的 55% 摊薄到"仍 57% 但壁钟腰斩"，说明**等待是被压缩后的并行**，不是未解的重叠缺口 |

**校准结论（用于后续同类算子 roofline）**：910B3 上本结构族的经验式
`wall ≈ (blocks_per_AIV × Σ_block_ops × 0.6 µs) 与 AIC 标量侧 (blocks_per_AIC × ~2.5 µs) 取大者`，
其中"Σ_block_ops"只看 `[half,bn]` 级 op（`[half,1]` 级 tiny op 实测**不计费**：round 4 的 alphaslim 去掉 1 copy + 1 clamp 后 +0.01%）。此式直接解释了 round 2 的 −36%（去 6 op）与 round 3 的 −3.1%（去 1 个宽 tile vcast）。

## 5. Iteration Log

> 每轮的表 = **候选 vs current best** 对比（Task Duration / AICore 利用率 / memory 指标 / L0）。数据来自 `perf_records.jsonl` 与 `profiles/` 原始 profile（`parse_profile.py` 汇总，跨 launch 取均值）。所有候选均从**同一个 current best** 派生，每个分支只改一个主要点。

### Round 1（base = baseline）：跨任务深度 2 流水（pipe2）

**现象**（baseline，70b-long fp16）：`aiv_scalar_wait_time = 4140.8 µs = 55% 壁钟`、`aic_scalar_ratio 0.612`、`aiv_vec_ratio 0.567`。两相位虽有 pass-1 零等待，但**每个 task 内部** Vector 的 pass-1 必须等 Cube 把该 task 的整条 S 带写完（单槽 ws + 任务边界 `FLAG_TASKDONE` 屏障 ⇒ Cube(T+1) 的 Q-hoist/pass-1 被 Vector(T) 的 drain 挡住）。⇒ 瓶颈 = **跨引擎握手串行**，不是算力。

| # | 候选优化点 | 实现 | 预期 |
|---|---|---|---|
| op1 | 跨任务深度 2（Cube 预产生 S(T+1) 到另一槽，Vector 消费 T；标志 id = `l + slot·L_max`；独立 `l1_q` 解除 Q-hoist 与 P-staging 的 L1 WAR；`2·L_max > 15` 时自动回落单槽 + 保留 TASKDONE） | `_gqa_sw_fwd_varlen_kernel_round1_op1_pipe2.py` | 同族先例 −18~−27%（`PL-1.12` / 定长版 r3） |

**候选 vs current best（baseline）**

| candidate | workload | Task Duration(us) | vs current best | aiv_vec_ratio | aic_scalar_ratio | GM_to_L1(KB) | L0 | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| baseline (current best) | 8b-long fp16 | 3759.49 | — | 0.564 | 0.607 | 126220.8 | pass | base |
| r1_op1_pipe2 | 8b-long fp16 | **3237.05** | **−13.9%** | 0.656 | 0.670 | 126220.8 | pass | **winner** |
| r1_op1_pipe2 | 70b-long fp16 | **6434.69** | **−14.4%** | 0.662 | 0.676 | 252377.6 | pass | winner |
| r1_op1_pipe2 | 70b-short fp16 | 902.45 | −22.7% | 0.542 | 0.532 | 26278.4 | pass | winner |

**合并与复测**（`logs/round1/merged_round1.log`，8/8 tune workload，L0 已先跑 `logs/round1/pipe2_l0.log` = 4 精度例 + contract PASS）：
8b-short fp16 475.87（−22.4%）/ bf16 465.96（−23.5%）；8b-long fp16 3237.05（−13.9%）/ bf16 3150.80（−16.1%）；70b-short fp16 902.45（−22.7%）/ bf16 880.12（−24.7%）；70b-long fp16 6434.69（−14.4%）/ bf16 6242.21（−15.9%）。smoke 精度在最终轮统一复核。
**结论**：全域 > 噪声阈值，合并入 current best（`r2m` 的 base）。

### Round 2（base = r1 pipe2）：带状掩码链的向量 op 瘦身 + vselect 裁决

**现象**（round 1 profile，8b-long fp16）：`aiv_vec_ratio 0.656`、`aiv_scalar 0.329`、`aic_scalar 0.670`、`aiv_scalar_wait_time` 仍占 65%。逐块清点 Vector pass-1 的 `[32,256]` 级 op：**always-on 的 per-row 列 OOB 惩罚链占 6 个**（`vsub/vmul/vmax/vmin/vmul/vadd`），而全部 tune workload 的 `kv_len`（512/2048）都能被 `bn_eff=256` 整除 ⇒ `tn_real == bn_eff` 恒成立 ⇒ 该链**可证明是 no-op**，却仍在每个块上执行。

| # | 候选优化点（单点） | 实现 | 实测（8b-long fp16） | vs current best | L0 | verdict |
|---|---|---|---:|---:|---|---|
| op1 | **列 OOB 链加运行期守卫** `if tn_real < bn_eff:`（全块无需列掩码） | `…_round2_op1_oobguard.py` | **2065.79** | **−36.2%** | pass | **winner** |
| op2 | 上界 + 列 OOB **折叠**为单条惩罚链（`clamp(diff − min(K_up, tnr−i),0,1)`，数学恒等），无守卫 | `…_round2_op2_minfold.py` | 3240.16 | +0.1%（噪声内） | pass | 弃选 |
| op3 | **vselect 备选形态**（`vcmp + vselect`，折叠阈值 + 有限 −1e38 哨兵 + `T.copy` 回写） | `…_round2_op3_vselect.py` | **blocked** | — | 未完成 | 弃选（见 §6） |

**候选 vs current best 明细**

| candidate | workload | Task Duration(us) | vs current best | aiv_vec_ratio | aic_scalar_ratio | GM_to_UB(KB) | L0 | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| r1_op1_pipe2 (current best) | 8b-long fp16 | 3237.05 | — | 0.656 | 0.670 | 18432.0 | pass | base |
| r2_op1_oobguard | 8b-long fp16 | **2065.79** | **−36.2%** | 0.618 | 0.928 | 18432.0 | pass | winner |
| r2_op2_minfold | 8b-long fp16 | 3240.16 | +0.1% | — | — | — | pass | no_gain |

**op2 的反向教训（本轮最有价值的数据）**：把 6 个"仅尾块需要"的 op 换成 5 个"每块都跑"的 op 后完全无收益 ⇒ **op 代价 ∝ 每块无条件执行的 tile 级 op 数**，与"哪些块需要掩码"无关。这直接给出 §4 的 0.61 µs/op 常数，也解释了 op1 的 −36%（6 op × 384 块/AIV × 0.61 µs ≈ 1.4 ms）。

**合并与复测**（`logs/round2_merged_all.log`，8/8）：8b-short fp16 358.62（−41.6%）/ bf16 346.56（−43.1%）；8b-long fp16 2067.32（−45.0%）/ bf16 2016.07（−46.3%）；70b-short fp16 660.19（−43.5%）/ bf16 645.70（−44.7%）；70b-long fp16 4084.05（−45.7%）/ bf16 4004.17（−46.0%）。L0 已由 op1 分支跑过（4/4 + contract PASS）。合并后 `aic_scalar_ratio` 升至 0.928 ⇒ Cube 侧标量管成为新的近饱和管道（下一步的约束）。

### Round 3（base = r2m oobguard）：块几何扫描 + S 载体 dtype

**现象**（round 2 profile，8b-long fp16）：`aic_scalar_ratio 0.928`（Cube 侧标量近饱和）、`aic_mte2 0.577`、`cube 0.234`、`aiv_vec 0.618` ⇒ 两侧接近平衡（Cube 侧略高），剩余空间在两边的**每块固定成本**上。§11 ④ 要求的 bm/bn 扫描在本轮补全。

| # | 候选（单点） | config | 实测（8b-long fp16） | vs current best | L0 | verdict |
|---|---|---|---:|---:|---|---|
| op1 | **bm=32 / bn=512**（块数 −33%，L0C/UB 均满足：`L_max=5`、2·L≤15） | 32,512,1 | 3699.39 | **+79%** | n/a（config 分支，`bench --check` max\|dout\|=2.4e-4 pass） | 弃选 |
| op2 | **bm=80 / bn=256**（§11 提名的宽 bm 方向） | 80,256,1 | **blocked** | — | — | 弃选（UB 溢出） |
| op3 | **f32 S 载体**（fp16 路径去掉每块 `[32,256]` 的 f16→f32 vcast；fixpipe 直写 f32） | 64,256 | **2003.35** | **−3.1%**（AB 裁决） | pass | **winner** |
| — | 参考：current best（r2m） | 64,256 | 2067.32 | — | pass | base |

- **op2 blocked 证据**：`half=40` 使全部 `[half,bn]` 缓冲 ×1.25 ⇒ UB 需求 `(16,384+32,768+32,768+32,768)×1.25 + 53,760 ≈ 204.8 KB > 196,608 B/AIV` ⇒ `bishengir-compile` SIGSEGV（`logs/candidate/round3/*r3_cfg_bm80bn256*`，24 次重试全失败）。根因是本 kernel 的掩码链需要 `ub_diff`/`ub_pen` 两块 `[half,bn]` f32（定长版姊妹靠 UB diet 删掉 rowmat 才腾出 bm=80 的空间）⇒ **bm=64 是本结构的 UB 上限**。
- **op3 AB 裁决**（T-2，`<5%` 差异必须走交错协议）：`ab_test.py --pairs 3 --launch-count 20`（`logs/round3/ab_sf32.log`）
  A（r2m）= 2066.15 / 2066.61 / 2067.26，B（sf32）= 2001.70 / 2003.68 / 1999.12 ⇒ `a_merged=2066.61`、`b_merged=2001.70`、`b_rel_change=−0.0314`、3/3 同向、**verdict = `B_better`**（sign_test_p=0.25，n=3）。采纳。
- 合并与复测：round 4 的 base 采用 sf32（fp16 域生效；bf16 路径本来就已是 f32 S ⇒ 数值不变）。

### Round 4（base = r3 sf32）：O 载体 / α 递推瘦身 / bn=128

**现象**（round 3 profile，8b-long fp16）：`aic_scalar 0.955`、`aic_mte2 0.650`、`aiv_vec 0.594`、`aiv_scalar_wait 1143 µs`。Cube 侧标量管已近饱和 ⇒ 继续砍向量 op 的边际收益下降；本轮验证三个"最后一次低成本尝试"。

| # | 候选（单点） | 实测（8b-long fp16） | vs current best | L0 | verdict |
|---|---|---:|---:|---|---|
| op1 | f32 O 载体（fp16 路径再去 1 个 `[32,128]` vcast） | 2000.09 | +0.08% | pass | no_gain（噪声内，回滚） |
| op2 | α 递推改为 `exp(min(m_old−m_cur,0))`：去掉每块 `T.copy(ub_m,ub_mprev)` + NaN clamp（数学恒等） | 2001.41 | +0.01% | pass | no_gain（证明 tiny op 不计费，回滚） |
| op3 | bn=128（§11 扫描另一半） | 2528.45 | **+26%** | n/a（config，`bench` 未跑 check） | 弃选 |

**候选 vs current best 明细**

| candidate | Task Duration(us) | vs current best | aiv_vec_ratio | aic_scalar_ratio | GM_to_UB(KB) | L0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| r3_op3_sf32 (current best) | 2003.35 | — | 0.594 | 0.955 | 30720.0 | pass | base |
| r4_op1_of32 | 2000.09 | +0.08% | — | — | 36864.0 | pass | no_gain |
| r4_op2_alphaslim | 2001.41 | +0.01% | — | — | 30720.0 | pass | no_gain |
| r4_cfg_bn128 | 2528.45 | +26.2% | — | — | — | n/a | 弃选 |

- bn=128 的机理：`L_max = ceil((64+1024)/128)+2 = 11` ⇒ `2·L_max = 22 > 15` ⇒ pipe2 自动回落单槽（屏障回来）+ 块数近乎翻倍 ⇒ 双重回退。**这就是 (64,256) 同时是 UB 上限与 flag 预算上限的交点。**
- **本轮 3 个分支均未超过 3% 噪声阈值**（op1/op2 正向但 ≤0.1%，op3 大幅回退）。

### Round 5（base = r3 sf32）：#11 f16 softmax 链（DESIGN defer 项）——探针判定 blocked

**现象**：`aiv_vec 0.594` + `x` 的 f16/f32 载体差已由 round 3 证明"按字节计费"。f16 链理论上把 prescale/掩码/减 m/exp/reduce 五个 `[32,256]` op 的字节减半。

**执行顺序按证伪协议**：先写最小探针 `perf_opt/probe_f16_chain.py`（f16 tile + f32 标量阈值 + f16 vexp/vreduce + 有限哨兵）验证 API 合法性，再决定是否改造全 kernel。

| 探针项 | 结果 | 证据 |
|---|---|---|
| (b) `T.vmul(f16 tile, <python float>, f16 tile)` | 通过（不报错） | `logs/round5_probe_f16.log` |
| (a) `T.vsub(f16 tile, <f32 PrimExpr 标量>, f16 tile)` | **编译失败** | `InternalError: Check failed: (src_dtype == binaryOp.Src1().GetBuffer()->dtype) is false: The element types of two operands should be the same for "hivm.hir.vsub"`（`src/target/codegen_npuir_api.cc:2019`） |
| 绕过形态 `T.Cast("float16", <阈值>)` | 合法但**破坏契约域** | f16 整数精确上限 = 2048；掩码阈值 `K_up = s_lo − n_lo + row0 + offset + up_shift` 在 KV-cache 域可 > 2048（如 `q_len=1, kv_len=4096 ⇒ offset=4095`）⇒ 掩码边界错 1 列 = 精度 bug；该域由 L1/L2/Boundary 覆盖（`L1-manifest-kvcache-*`） |

**判定：blocked（family_no_gain）**。理由链完整可复现：合法形态已穷举（直接 f16 标量 / `T.Cast` 标量 / 哨兵有限化三条），直接形态编译器拒绝、`T.Cast` 形态与契约域冲突；剩余可行形态（掩码链留 f32 + 每次惩罚多一次 `vcast` 到 f16）按"op 数 × 字节"记账为净中性或净亏（+1 op vs 省半个 tile）。**不写入 `perf_records.jsonl`**（blocked 分支契约），证据留在本表与 `logs/round5_probe_f16.log`。

**T-1 参照结构 diff 检查（进入 plateau 判定前强制执行）**：检索同族已知最优实现 —— 定长版姊妹 `_gqa_sw_fwd_kernel/perf_opt/`（同日同 commit 收束，winner `r3_op1_pipe2`，8b-long fp16 = 1060.31 µs / 2048 tasks / 7680 块 / H·B=64）。本算子同 shape 为 4096 tasks / 15360 块 / H·B=128（**恰好 2× 单位工作量**）：`final 2002.73 / (1060.31×2) = 0.945` ⇒ **本 varlen 版本按单位工作量比同族定长版参考实现快 5.5%**，结构形态一致（两相位 + pipe2 + (64,256) + f16/f32 载体分派）。逐项 diff：grid 组织（1-D persistent 20 核，同）、数据流分相位（S/P 物化两相位，同）、片上驻留（L1 生命周期复用 + UB [half,bn]，同）、dtype 路径（S f32 / O f16，与姊妹的 f32 载体选择同向）、唯一差异 = 姊妹的 per-shape `bm=80` 分派（本算子被 UB 硬上限封死，round 3 op2 实测）与 nqb2 变体（姊妹自身也未采纳）。**未发现可迁移的结构差异** ⇒ 允许进入终止判定。

### Round 10（base = r3_op3_sf32 / 现行 current best）：M1 手写双缓冲重叠 —— 单点证伪实验

**本轮定位**：历史结论 §5 round 5 与 §8 写明「片内重叠只能手写（需 `l1_b`/`l0_c` 的双缓冲 + 动态偏移切片的 gemm 操作数，本轮未验证可行）」。Round 10 只验证这一件事，不动 bm/bn、不动参数空间、不回退 baseline。`BP_pipeline_overlap`（open：本结构族 pipeline overlap 无收益）是本轮要证伪的命题。

**现象（round 9 final 的 raw profile，同 session 重测，8b-long fp16，20 launch）**——Cube 侧四管道实测：
`aic_time 1959.12 µs`；`mte2 1291.31 (0.650)`、`mte1 609.24 (0.307)`、`cube 483.12 (0.243)`、`fixpipe 625.24 (0.315)`；
四管道之和 **3008.9 µs = 1.536 × 墙钟**（已部分重叠）；`aic_scalar_ratio 0.951`，其中跨引擎 flag 等待（`aic_scalar_wait_id*` 之和）**1073.4 µs**。
⇒ **mte2 是最大单管道（65% 墙钟），且单缓冲使 V(l+1) 必须等 gemm(l) 释放 `l1_b`。** 若 mte2 能完全藏进 cube+fixpipe，理论上界 = 3008.9/1959.12 → 墙钟 ≈ 1291 µs（**1.52×**）；Round 10 的采纳门槛只取 **1.10×**（拿到部分重叠即可）。

#### 候选优化点与实验分支（全部从同一个 current best 派生，每个分支只改 M1 一件事）

| # | 分支 | 实现要点 | 结果 |
|---|---|---|---|
| M1a | `_gqa_sw_fwd_varlen_kernel_round10_m1a_dbp2.py` | pass-2 双缓冲（P+V 各 2 块 L1，共 5 块活 L1 = 208 KB）；块 parity 用 trace-time 展开（稳定对 + 剥离末对 + 剥离奇尾） | **blocked**：`bishengir-compile` 一次 12 min @100% CPU 不收敛，随后 8 次 `SIGSEGV (err code -11)` / 15 min，0 成功 |
| M1b | `_gqa_sw_fwd_varlen_kernel_round10_m1b_3buf.py` | 3 块活 L1 形态：Q 与 P 共用 `l1_a`（两者在一次 task 迭代内生命周期不重叠），V 双缓冲 `l1_b`/`l1_b2`；稳定对 + 剥离末对 + 奇尾 | **blocked**：15 min 仅 1 次 SIGSEGV，其余时间编译不收敛 |
| M1d | `_gqa_sw_fwd_varlen_kernel_round10_m1d_lean.py` | 同 M1b 但只留**一个** pair body（前瞻索引钳到 `loop_count-1`，偶数块数时末对重复读一次末块，无消费者）——IR 体量减半 | **blocked**：同样 15 min 不收敛 |
| M1c | `_gqa_sw_fwd_varlen_kernel_round10_m1c_vfirst.py` | 零新增缓冲的**部分重叠**：把与跨引擎 gate 无关的 V 装载（GM→L1，64 KB）提到 `sync_block_wait` **之前**，让 MTE2 在 Cube 等 Vector 时不空转 | **编译通过（25 s）+ L0 通过 + 实测 −1.4%（回退）** |

#### M1 为什么 blocked：探针阶梯（`logs/r10_m1/PROBE_LADDER.md`，同一 session、同一工具链）

| 探针 | 相对 current best 的改动 | 活 L1 | 结果 |
|---|---|---|---|
| base | —（参照） | 3 块 / 112 KB | **26–42 s 编译通过，0 SIGSEGV** |
| A | +`l1_a2`(32K) +`l1_b2`(64K) + `T.copy(zbuf, l1_b2)` | 5 / 208 KB | 420 s，3 SIGSEGV，0 成功 |
| A1 | +`l1_b2`(64K) + `T.copy(zbuf, l1_b2)` | 4 / 176 KB | 400 s，5 SIGSEGV |
| A2 | +`l1_a2`(32K)，**从不使用** | 4 / 144 KB | **26 s 通过**（死分配被消除） |
| A3 | +`l1_pad`(4K) + `T.copy(zbuf[0:16], l1_pad)` | 4 / 116 KB | 72 次 SIGSEGV，0 成功 |
| C1 | +`l1_b2`(64K) + **`T.clear(l1_b2)`** | 4 / 176 KB | **26 s 通过** |
| OL | C1 + **一条** `T.copy(v…, l1_b2[…])` 放进原 pass-2 块循环 | 4 / 176 KB | 600 s，28 SIGSEGV |
| 4L1 | 独立最小 kernel：4 块活 L1 + 2 gemm + 4 条 GM→L1 copy | 4 | **37 s 通过** |

**阶梯结论（这是本轮最有价值的负面数据）**：
1. **不是 L1 容量**：116 KB（A3）失败而 176 KB（C1）通过；L1 规格 512 KB，且本前端对 cbuf 超量分配本就无容量诊断（`CONST-capacity` 条目）。
2. **不是「4 块 L1」通例**：独立最小 kernel（4L1）37 s 编译通过。
3. 触发点很窄且可复现：**本 kernel 内第一条「目的地址是第 4 块活 L1 缓冲」的 GM→L1 `T.copy`**（A1/A3/OL 合计 0/38 成功）；同一缓冲 `T.clear` 合法（C1）。
4. 因此「第二块可喂养的 L1 操作数缓冲」——任何手写双缓冲的前置条件——在本 kernel + 本工具链上不可用；改用 3 块活 L1 绕开第 4 块后，**成对展开的循环体本身**又让 `bishengir-compile` 不收敛（M1b/M1d，>15 min @100% CPU vs 基线 26–42 s）。

#### 唯一能编译的形态 M1c：实测为回退（候选 vs current best）

| candidate | workload | Task Duration(us) | vs current best | aic_mte2_ratio | aic_mte1_ratio | aic_cube_ratio | aic_fixpipe_ratio | aic_scalar_ratio | 四管道和/墙钟 | L0 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| current best (r9_final_sf32) | 8b-long fp16 | **2019.76** | — | 0.634 | 0.300 | 0.238 | 0.309 | 0.947 | 1.507 | pass | base |
| r10_m1c_vfirst | 8b-long fp16 | 2047.50 | **+1.4%** | 0.618 | 0.288 | 0.236 | 0.312 | 0.946 | 1.474 | pass | no_gain（回滚） |
| current best (r9_final_sf32) | 70b-long fp16 | **4000.44** | — | 0.648 | 0.303 | 0.241 | 0.312 | 0.962 | 1.519 | pass | base |
| r10_m1c_vfirst | 70b-long fp16 | 4058.26 | **+1.4%** | 0.624 | 0.290 | 0.237 | 0.312 | 0.952 | 1.486 | pass | no_gain（回滚） |

**M1c 的反向机理**：把 V 装载提到 gate 之前后，mte2 忙比 0.634→0.618（更闲）、四管道和/墙钟 1.507→1.474（重叠更差）、跨引擎等待 1191.7→1126.2 µs 却**总量不降**——被填的只是 MTE2 的空转，而 L1 端口上 MTE2 写与 MTE1 操作数读（0.300 忙比）互拖，最终墙钟 +1.4%（与 `PL-1.18` 的「L1 双缓冲端口互拖」同向）。两个 long 同向 +1.4% ⇒ **回滚，current best 不变**。

#### 是否采纳

**否**。采纳条件「long 的 5-run A/B 中位数 ≥1.10×，且 short 不回归、L0/L1 全过」：
- M1a/M1b/M1d 在编译器层 blocked，无时延可言（不造数据）；
- M1c 是唯一可编译形态，两个 long 同向 **−1.4%**（回退），按协议不进 A/B（无正向候选），short 也因此无需再测。
- current best 保持 `r9_final_sf32`（sha `d98cc867…`），本轮 `tuning_status = no_gain`。

**对下一步（nqb=2）的判据贡献**：Round 10 把「片内重叠」这条路的封堵点定位到 **编译器对「第 4 块活 L1 缓冲的 GM→L1 写」与「成对展开的 Cube 循环体」两个硬失败**，而不是「910B3 没有重叠空间」——四管道和/墙钟 1.50–1.54 说明已有 50% 重叠，且 mte2 单管道占 65% 墙钟仍有 1.5× 理论空间。nqb=2 的本质同样是**循环体翻倍**（历史已记 doubled body → bishengir 不收敛），因此本轮证据**下调** nqb=2 的可行性预期：不换到能收敛的代码形态（例如把 K/V 复用做成 trace-time 展开的更小增量）之前，nqb=2 大概率撞同一个墙。若仍要试，建议先只做「K/V 的 GM→L1 装载次数减半」这一点的**编译可行性探针**（不要求性能），成本约 15 min，可在下一轮 dispatch 的第一步完成。

### Round 13（base = `r9_final_sf32` / 现行 current best，sha `d98cc867…`）：G=2 head-group batching —— 采纳并转正

**轮次定位**：round 11/12 的探针（nqb=2、dualkernel、bn=128 复现）均未产生可合并 winner，其记录见 `logs/round11/`、`logs/round12/ROUND12_CONCLUSION.md`。Round 13 只验证一个结构假设：`heads / heads_kv = 4`（8b 与 70b 均为 32/8、64/8），因此**两个 query head 可以在块内共享同一个 KV head 的 K/V**；把每个 band step 的 K/V `GM→L1` 装载按 head 对摊薄。本轮定位为**可行性 → 原型 → A/B 过门槛 → 转正收尾**，不再派生第二个优化点。

**现象（round 9 final 的 raw profile，8b-long fp16，同 session）**：`mte2 1320.8 µs (ratio 0.667)`、`cube 482.6 (0.244)`、`fixpipe 632.0 (0.319)`、`scalar 1902.5 (0.960)`；`GM_to_L1 = 126,064 KB`，其中 K+V 占 ≈97.6 %。⇒ mte2 是最大单管道，且其字节量几乎全是每 head 重复的 K/V ⇒ **head 共享是唯一能移动该管道的参数级/结构级杠杆**。

#### 13.1 可行性前提：UB 账本（G=2 必须先瘦身才装得下）

| 项 | G=1（current best） | G=2 未瘦身 | G=2 + D→N 别名（采纳形态） |
|---|---:|---:|---:|
| user `T.alloc_ub` | 133,376 B | 151,040 B | 151,040 B |
| 编译器 staging | 38,656 B | 38,656 B | 38,656 B |
| plan total | 196,608 B（=192 KB，0 余量） | **214,272 B（超 17,664 B）** | **172,032 + 17,664 = 189,696 B（余量 3.5 %）** |

瘦身手段（`_round13_shrink.py`，已 L0 验证）：`ub_f16_D`/`ub_f32_D` 只在 pass-2 + epilogue 用，`ub_f16_N`/`ub_f32_N` 只在 pass-1 用 ⇒ D 声明为 N 的 Python 别名（`[half,dim]` 前缀），**−24,576 B，op 数不变，8b-long fp16 实测中性（−0.1 %）**。G=2 原型即建立在该形态上；`head_group = 2` 仅在 `groups >= 2 and groups % 2 == 0 and heads % 2 == 0` 时启用，否则退回单 head 体（`H/Hkv = 8/8` 的 L1 例覆盖该回退路径）。

#### 13.2 候选 vs current best 对比表（Task Duration + AICore 利用率 + memory + L0）

口径：`msprof op Task Duration(us)`（`--launch-count=20 --warm-up=5`，median）；AIC 比值取对应 raw profile 的 `PipeUtilization`/`Memory` CSV；memory 取 `GM_to_L1_datas(KB)` / `L0C_to_GM_datas(KB)`。

| 候选 | workload | Task Duration(us) | vs current best | aic_mte2_ratio | aic_cube_ratio | aic_fixpipe_ratio | aic_scalar_ratio | GM_to_L1 (KB) | L0C_to_GM (KB) | L0 | 裁决 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| current best `r9_final_sf32` | 8b-long fp16 | **2017.22**（5-run A/B base 中位数） | — | 0.667 | 0.244 | 0.319 | 0.960 | 126,064 | 61,360 | pass | base |
| `r13_g2_headgroup` | 8b-long fp16 | **1767.48**（5-run A/B g2 中位数） | **1.141×**（−12.4 %） | **0.433** | 0.261 | 0.296 | 0.869 | **77,280** | 61,600 | pass | **winner** |
| current best `r9_final_sf32` | 70b-long fp16 | **3991.16**（3-run） | — | — | — | — | — | — | — | pass | base |
| `r13_g2_headgroup` | 70b-long fp16 | **3487.04**（3-run） | **1.145×**（−12.6 %） | — | — | — | — | — | — | pass | **winner** |
| current best `r9_final_sf32` | 8b-short fp16 | **344.56**（3-run） | — | — | — | — | — | — | — | pass | base |
| `r13_g2_headgroup` | 8b-short fp16 | **312.60**（3-run） | **1.102×**（−9.3 %） | — | — | — | — | — | — | pass | **winner** |
| current best `r9_final_sf32` | 8b-long bf16 | 2036.34（r10_readout，A/B base 臂无效） | — | — | — | — | — | — | — | pass | base |
| `r13_g2_headgroup` | 8b-long bf16 | **1798.08**（1-run 补测） | **1.133×** | — | — | — | — | — | — | pass | **winner** |

（`—` = 该 workload 本轮只读 Task Duration；完整四管道分解只在 8b-long fp16 上导出。8b-long bf16 的 A/B 三对 run 中 base 臂与 2/3 个 g2 臂采集返回 NaN，故按"未测"处理，用 1-run msprof 补 final，baseline 沿用 r9 final 的 `r10_readout` 记录。）

**收益来源（8b-long fp16，AIC）**：`mte2_time 1320.8 → 757.3 µs（−42.7 %）`，`GM_to_L1 126,064 → 77,280 KB（−38.7 %）`；`cube +5.5 %`（fused `M = bm + tmc` gemm 使 L0B 操作数装载减半）、`fixpipe −9 %`、`scalar −11 %`。⇒ 增益归因为 **GM→L1 带宽**，不是算术量变化。

#### 13.3 分支与轮次编排

| # | 分支 | 单点改动 | 结果 |
|---|---|---|---|
| A | `_round13_ubprobe.py` | UB 分配器语义探针（4×64 KB 缓冲：disjoint vs concurrent 生命周期） | disjoint 编译通过且 4 号缓冲 alias 到 0 号（证明 `hivm-plan-memory` 按 live range 复用、192 KB 是硬门）；concurrent 报 `ub overflow, requires 2097152 bits while 1572864 bits available` |
| A2 | `_round13_shrink.py` | D→N 别名（−24,576 B） | L0 pass、8b-long fp16 中性（−0.1 %）；作为 G=2 的底座 |
| A3 | `_round13_g2.py` | head_group=2，fused gemm，Vector 侧运行期 `gh` 循环（非 trace-time 展开） | **A/B 过门槛 → 转正** |
| B | 因果全带运行期证明 | 跳过"整 tile 在带内"的掩码链 | **不建分支**：lowered IR 已有 3 个运行期 `scf.if` 守卫，8b-long 每 half 3.75 tile 中仅 1.5 次掩码链执行（40 %）且都跨边界，预期增益 ≈0 |
| C | V 转置 A/B | V 以 `[D,L]` 操作数读 | **跳过**：round 12 实测 net −0.28 µs（价值 ≤5 % 且符号为负） |

#### 13.4 转正与全量回归（本轮收尾）

- 旧 baseline 归档：`perf_opt/history/_gqa_sw_fwd_varlen_kernel_r9_final_sf32.py`（sha `d98cc867…`，逐字节保留）。
- 转正：`_round13_g2.py` **逐字节**拷贝为 `perf_opt/_gqa_sw_fwd_varlen_kernel.py`，`cmp` 一致，sha = `6befe38405368d894f2b619b62e7ad5f777422364ff3c930fe8131d73a3c0c79`。
- 全量回归 `python perf_opt/_gqa_sw_fwd_varlen_kernel.py --level all`（`logs/round13/final_all.log`）：**L0 4/4 + contract PASS、L1 26/26 PASS、L2 12/12 PASS、Boundary 9/9 PASS**（51/51 全部 PASS，0 FAIL，无告警层失败）。上一轮 dispatch 因设备被并发 benchmark 占用，L1 只跑完 6/26 + 12/12 定向子集；本轮设备空闲，**26/26 达成**。
- 转正前后 `perf_records.jsonl`：旧 18 行（round 10 的 8 baseline + 2 candidate + 8 r9 final）整份归档到 `perf_opt/history/perf_records_r9.jsonl`；新文件写 16 行（8 个 tune workload 各 1 条 `phase=baseline`，candidate `r9_final_sf32`、sha `d98cc867…`；各 1 条 `phase=final`，candidate `r13_g2_headgroup`、sha `6befe384…`）。4 个 A/B 已测 workload 用 A/B 中位数；其余 4 个（70b-short fp16/bf16、8b-short bf16、70b-long bf16）+ 8b-long bf16 各补 1-run msprof（`logs/round13/final_msprof_result.json`）。

## 6. 实验裁决：主选（算术惩罚式）vs 备选（`vselect`）—— §11 ③ 必办项

判定阈值（DESIGN §1.6.3）：**备选 latency < 主选 × 0.95 才翻转**。

| 步骤 | 做法 | 结果 |
|---|---|---|
| 1. 备选形态实现 | 在 `perf_opt/` 内实现折叠阈值版 vselect 形态：`vsub(diff−thr)` → `vbrc(0)` + `vcmp(...,"le"/"ge")` → `vbrc(−1e38)` + `vselect(cond, S, sentinel, out)` + `T.copy` 回写；`vand` 不可用（`docs/Tilelang.language/逻辑操作/T.vand.md` §2.2：bool 列在 Ascend 上不支持）⇒ 上界与左窗各一次 select | `_gqa_sw_fwd_varlen_kernel_round2_op3_vselect.py` |
| 2. UB 记账 | 备选需额外 `ub_sent` + `ub_sel`（`[half,bn]` f32，各 32 KB）+ `ub_c`（bool 8 KB）；三条比较的无折叠形态需 `ub_col` ⇒ 198.7 KB > 192 KB，**设计默认 config 下无折叠形态不可行**；折叠到两条比较后 165 KB 可行 | 见 §5 round 2 |
| 3. 实测 | `--level L0`（首例 smoke fp16, msq=128） | **未完成**：`bishengir-compile` 单进程 **99.9% CPU 连续 8 min 56 s 无输出**（PID 872277，`/tmp/tmp2ffns86s/kernel.npuir`），同结构算术版同 case ≈ 40 s 完成 4 例 ⇒ 判定为**编译器病态**（bool 掩码 `[32,256]` + `vselect` 形态），进程被强制终止 |
| 4. 裁决 | 阈值 `0.95` 的分子（备选时延）**无法取得**；且（a）备选在默认 config 下 UB 不可行，（b）在可行 config 下编译不终止，（c）`PL-1.29-vselect-cost`（verified：`vselect ≈ 6.5×` 普通 v-op）+ `PL-1.11`（vcmp 标量化，2026-10-16 已在 910B3 复现）共同指向备选更慢 | **主选保留**（备选未满足 `备选 < 主选 × 0.95`，翻转条件不成立） |

**归属**：该裁决把 DESIGN §1.6.3 的「主选 + 实验裁决」落地为实测 + 结构性否决的闭环；设计判定（算术惩罚式）**未被推翻**，属"以实测/文档固化设计判定"，不构成设计层天花板 ⇒ 不产出 `perf_feedback.md`。

## 7. 参数扫描 / Autotune Log

未使用 autotune 框架（参数空间只有 bm/bn/dtype 载体三类，且 flag 预算与 UB 把它压到极小）；采用等价的手动粗扫 + AB 裁决：

| 扫描 | 空间 | 实测（8b-long fp16） | 结论 |
|---|---|---:|---|
| bm × bn | (64,256) / (64,128) / (32,512) / (80,256) | 2003.35 / 2528.45 / 3699.39 / blocked | **(64,256) 唯一最优且是可行域边界**：`bn=256` 是 `2·L_max ≤ 15` 的上界（bn=128 时 L_max=11），`bm=64` 是 UB（≤192 KB）的上界（bm=80 → 204.8 KB） |
| num_stages | 仅 `1`（两相位单槽结构 stage-invariant，DESIGN §3.3 保留旋钮） | — | 未扫（结构不变式） |
| dtype 载体 | S ∈ {f16,f32} × O ∈ {f16,f32}（fp16 路径） | S-f32 −3.1%（AB）、O-f32 +0.08% | S=f32 / O=f16 |
| 掩码机制 | 算术惩罚（守卫式） / 折叠式 / vselect | 2065.79 / 3240.16 / blocked | 守卫式算术惩罚 |

## 8. Final Summary

**最终候选（Round 13 转正）**：`perf_opt/_gqa_sw_fwd_varlen_kernel.py`（`candidate_id = r13_g2_headgroup`，sha256 `6befe38405368d894f2b619b62e7ad5f777422364ff3c930fe8131d73a3c0c79`，= 磁盘上实际文件，`cmp` 与 `_round13_g2.py` 逐字节一致，已核）。上一版 `r9_final_sf32`（sha `d98cc867…`）归档在 `perf_opt/history/_gqa_sw_fwd_varlen_kernel_r9_final_sf32.py`，作为本轮 baseline。
**采纳的优化点（3 项，全部落在同一个 `@T.prim_func` 内，工厂签名与公开契约不变）**：
T1 跨任务深度 2 流水（含 TASKDONE 屏障移除 + `l1_q` 独立）；T2 列 OOB 惩罚链的运行期守卫；T3 fp16 路径的 f32 S 载体。
**弃选**：折叠式掩码（+0.1%）、O 载体 f32（+0.08%）、α 递推瘦身（+0.01%）、bm=32/bn=512（+79%）、bm=80（UB blocked）、bn=128（+26%）、vselect 备选（编译 blocked）、f16 softmax 链（探针 blocked）、**M1 手写双缓冲（round 10：4/5 缓冲形态 SIGSEGV，3 缓冲成对展开形态编译不收敛；唯一可编译形态 M1c 回退 −1.4%）**。

**记录迁移（round 10 交付整理）**：上一 session 的 `perf_records.jsonl`（round 0–9，41 行）里 `phase=final` 的 `artifact_sha256 = 1216193a…` 与磁盘上现存文件 `d98cc867…` 不一致，且它引用的 `profiles/`、`logs/`、`bench.py`、`workload_inventory.json` 已随上一 session 丢失，无法再对账。按 append-only 纪律**整份归档**到 `perf_opt/history/perf_records_round0-9_final_sha121693a.jsonl`（见 `history/README.md`），本轮在同 session 用现存文件对 8 个 tune workload 重测 baseline/final，生成自洽的新 `perf_records.jsonl`（final 单一 `candidate_id`、sha 与磁盘文件一致）。

### Final Performance Test Data

> 全部来自同一最终候选 `r13_g2_headgroup`（sha `6befe384…`，= `perf_opt/_gqa_sw_fwd_varlen_kernel.py`）的 `phase=final` 记录（`msprof op`，`--launch-count=20 --warm-up=5`，median）。baseline = 上一版 `r9_final_sf32`（sha `d98cc867…`，归档于 `perf_opt/history/_gqa_sw_fwd_varlen_kernel_r9_final_sf32.py`）。4 个 A/B 已测 workload 取交错 A/B 的按臂中位数（`logs/round13/ab13_result.json`，raw profile `profiles/round13/ab_*`）；其余取 `r10_readout` 记录或本轮 1-run 补测（`logs/round13/final_msprof_result.json`，raw profile `profiles/round13/final_*`）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---:|---|
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-8b-short-w256-float16 | 344.56 | 312.60 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-8b-short-w256-bfloat16 | 350.88 | 311.14 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-8b-long-w1024-float16 | 2017.22 | 1767.48 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-8b-long-w1024-bfloat16 | 2036.34 | 1798.08 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-70b-short-w256-float16 | 642.48 | 572.08 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-70b-short-w256-bfloat16 | 647.30 | 575.64 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-70b-long-w1024-float16 | 3991.16 | 3487.04 | r13_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/grouped_query_attention_sliding_window_varlen_kernel/_gqa_sw_fwd_varlen_kernel.py::_gqa_sw_fwd_varlen_kernel::_gqa_sw_fwd_varlen_main | llama-3.1-70b-long-w1024-bfloat16 | 4041.12 | 3533.76 | r13_g2_headgroup |

### 逐 workload 改进与证据

（同一最终候选 `r13_g2_headgroup` 的 `phase=final` 记录。）

| workload | baseline → final (µs) | 提升 | 加速 | 口径 |
|---|---|---:|---:|---|
| 8b-short fp16 | 344.56 → 312.60 | −9.3% | 1.102× | A/B 3-run |
| 8b-short bf16 | 350.88 → 311.14 | −11.3% | 1.128× | 1-run 补测 |
| 8b-long fp16 | 2017.22 → 1767.48 | −12.4% | 1.141× | A/B 5-run |
| 8b-long bf16 | 2036.34 → 1798.08 | −11.7% | 1.133× | 1-run 补测 |
| 70b-short fp16 | 642.48 → 572.08 | −11.0% | 1.123× | 1-run 补测 |
| 70b-short bf16 | 647.30 → 575.64 | −11.1% | 1.124× | 1-run 补测 |
| 70b-long fp16 | 3991.16 → 3487.04 | −12.6% | 1.145× | A/B 3-run |
| 70b-long bf16 | 4041.12 → 3533.76 | −12.6% | 1.144× | 1-run 补测 |

（8b-long fp16 的完整四管道与 memory 分解见 §5 Round 13.2：`mte2_ratio 0.667 → 0.433`，`GM_to_L1 126,064 → 77,280 KB`。8b-long bf16 的 A/B base 臂采集无效（NaN），故 baseline 用 `r10_readout` 的 r9 final 记录、final 用 1-run 补测。）
**8 个 tune workload 全部正向且远超 3% 噪声阈值（+9.3%–+12.6%，1.102×–1.145×）；相对上一版 r9 的几何平均加速 ≈ 1.128×。**

**smoke 排除与最终精度**（`kind=smoke`，仅精度、不采性能）：
- 最终文件 `--level all` 全绿（**round 13 转正后重跑**，`logs/round13/final_all.log`，exit 0，设备空闲、无并发作业）：**L0 4/4 PASS + contract PASS、L1 26/26 PASS、L2 12/12 PASS、Boundary 9/9 PASS**（51/51 全过，0 FAIL；L0/L1 为阻塞层，L2/Boundary 告警层本轮也无失败）。
- manifest smoke 案例（B=2，q_len=k_len=128，H8/Hkv2/D64，causal，wl=−1）就是 L0-1 / L0-2 的主 trace 例，直接用最终文件复核：

| smoke workload | max\|dout\| | max\|dlse\| | isfinite | 判据 | 结果 |
|---|---:|---:|---|---|---|
| `gqa-sw-varlen-fwd-smoke-float16`（L0-1） | 1.953e-03 | 9.537e-07 | True | atol_out 5e-3 / atol_lse 1e-3 | **PASS** |
| `gqa-sw-varlen-fwd-smoke-bfloat16`（L0-2） | 1.562e-02（tier1 flips 185 → tier2 2-ulp 门内） | 1.431e-06 | True | 同上 + bf16 2-ulp 回退门 | **PASS** |

**中止原因**：`success`（round 13 在既有 plateau 上拿到新的结构级增益并转正）。**round 10** 的 M1 手写双缓冲 4 条分支在 `bishengir-compile` 上不收敛/回退 1.4% ⇒ `no_gain`；**round 13** 改走 head 共享（G=2）——8b-long 1.141× / 70b-long 1.145× / 8b-short 1.102× / bf16 1.125–1.144×，L0/L1/L2/Boundary 51/51 全过 ⇒ `winner_merged`（详见 §5 Round 13、`logs/round13/ROUND13_CONCLUSION.md`）。**逐项枚举剩余候选为何不可继续**：
1. **UB 硬上限**：`bm ≥ 80` 需 204.8 KB/AIV > 192 KB（实证 `bishengir-compile` SIGSEGV）；`bn = 512` 需 272 KB。UB 被掩码链的两块 `[half,bn]` f32（`ub_diff`/`ub_pen`）占满 —— 它们是"精确整数 diff"与"算术惩罚"两条恒等性论证的载体，无法删除（round 2 op2 的折叠只减 op 数、不减缓冲）。
2. **flag id 预算**：`2·L_max_bound ≤ 15`。`bn=128 ⇒ L_max_bound=11 ⇒ 22 > 15`（pipe2 失效，实测 +26%）；`bn=512` 虽放宽 flag 但被 UB 拦住。`(64,256)` 是两条硬约束的唯一交点。
3. **编译器病态**：vselect 备选形态 `bishengir-compile` 8 min 56 s 未终止（§6）；f16 链在 `T.vsub(f16 tile, f32 scalar, f16 tile)` 被 codegen 拒绝（§5 round 5）。
4. **设计判定（已由 Stage 3 冻结，本轮不推翻）**：scale 折入 q（M11'，精度降级）、逐列裁剪（M14'，无 API）、KV-split（无核间原语且任务数 4096 ≫ 20 核无欠载）、host 块表（契约不允许 D2H）。
5. **op 数瘦身家族已穷尽**：tiny op 不计费（round 4 证）、O 载体 f32 无效（round 4 证）、折叠无效（round 2 证）、S 载体 f32 已采纳（round 3）。
并且 round 4/round 5 连续 2 轮无 valid 分支超过 3% 噪声阈值，T-1 参照结构 diff 检查（§5 round 5 末）确认与同族最终形态在单位工作量上持平（0.945×）。`last_improvement = 3.1`（round 3 sf32）、`consecutive_no_improvement = 2`。
**设计层天花板**：**未触发**。实测已落在 / 优于 DESIGN §11 的乐观区间（见 §9），且剩余空间被 UB 硬上限 + flag 预算 + 编译器限制封住，属"冻结设计族在工具链内的可达上限"，不满足 `perf-feedback.md` §1 的"结构性加速估计 > 2x"（round 13 后的累计加速 ≈1.83×1.128 ≈2.06× 相对 Stage 3 文件，但其来源是参数/op 数与 head 共享级优化，不是设计层归因；**剩余**结构候选 G=4 受 UB + L0C 容量封堵，预估另一层 1.15–1.20×，未达 2x 门槛）⇒ **不产出 `perf_feedback.md`**。

## 9. 设计估算（D-2）vs 实测偏差（Phase 1 要求的回填行）

| 口径 | 设计期估算（DESIGN §1.6.0 D-2 / §11，8b-long fp16） | 实测（baseline） | 偏差 | 失准项 |
|---|---|---:|---:|---|
| 朴素发射口径下界 | 3.3 ms | **3.759 ms** | +13.9% | **发射项**：设计按 ~17 op/块估，实测该结构每块还有 always-on 列 OOB 链（6 op）+ 握手；fabric 项也偏低（ws 往返实测占用率 <23% 未触顶，故非主因） |
| 定长版锚点外推（保守） | 2.56 ms（1279.54 µs × 2 任务比） | 3.759 ms | **+46.8%** | 定长版锚点未包含 varlen 新增的 per-task ragged 解码（Cube 与 Vector 各一遍）与 per-row 列 OOB 链 ⇒ varlen 的**每块固定成本**高于定长版 |
| 乐观区间 | 2.0–3.3 ms | 3.759 ms | 高于上界 | 同上 |
| **调优后** | — | **2.003 ms** | **低于乐观下界** | 优化后已进入 / 优于设计乐观区间 |

70b-long fp16：设计乐观区间 4.0–6.5 ms，实测 baseline 7.515 ms → **final 3.970 ms（优于乐观下界 0.8%）**；smoke（不调优）baseline 60.78 µs vs 设计 ~0.04–0.1 ms ✓ 落在区间内。

## 10. Skill Retrospective

### 10.1 本轮暴露的流程问题

1. **`risk: 后台 runner 被重复启动`**：`setsid nohup …&` 与 shell 超时叠加，导致同一 `--phase baseline` 起两个实例（`perf_records` round 0 出现 2 行重复、相差 0.25%）。建议：模板 runner 加 pidfile 互斥（`perf_opt/.runner.lock`，含 `--force` 逃生），并在 `phase in (baseline, final, merged)` 时拒绝并发的第二个实例。证据：§2 记录说明段 + `perf_records.jsonl` round0 前 4 行。
2. **`risk: benchmark 口径的 varlen 特有不兼容`**：`Op.forward()` 的 host 侧校验算子 + `wrapped()` 内 `torch.zeros` 会污染 msprof 采集窗口（report `invalid_profile_count=10`）。本轮以"闭包取 `_compile` + 预分配 ws"的自建 driver 绕过，**该形态可直接复用**为 varlen 类算子的标准 Stage 4 口径（建议进 `profile-collection.md`）。证据：`bench.py` 的 `_closure()`/`build_runner()`。
3. **`gap: 阻塞分支的 profile 目录残留`**：op3（vselect）的 msprof 目录被创建但无记录（编译未完成），`profiles/candidate/round2/` 下留下空壳目录；对账时需靠"无 OpBasicInfo 行"判空。建议 runner 在 `no OpBasicInfo rows` 时把该目录重命名为 `*.invalid`。

### 10.2 Value Point Proposals（vp_type + 证据三件套）

| id | vp_type | 提案 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| **VP-A（新，D 类）** | D | **910B3 · [32,256] f32 成块向量 op 代价 ≈ 0.61 µs/op**（每 AIV），且**tiny `[32,1]` op 不计费**；据此"op 数 × tile 字节"可预测壁钟到 ±10% | 本 opt_log §4（round 2 的 `+1175 µs / (384 块 × 5 op)`；round 4 的 alphaslim 去 2 个 tiny op → +0.01%） | `perf_opt/probe_f16_chain.py` 同目录可加极简 timer；本任务以 in-kernel A/B 反解（`logs/round2_long.log`、`logs/round4_long.log`） | tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-10-17 |
| **VP-B（新，P 类）** | P | **"可证明 no-op 的掩码子链必须加运行期守卫"**：带状掩码的 per-row 列 OOB 惩罚链在 `tn_real == bn_eff` 时恒为 0，但两相位 kernel 里它通常写成 always-on；加 `if tn_real < bn_eff:` 守卫在长域拿 **−36.2%**（同轮"折叠成更少的无条件 op"反而零收益）——判据 = **无条件执行的 tile 级 op 数**，不是"哪些块需要掩码" | 本 opt_log §5 round 2（op1 −36.2% vs op2 +0.1% vs base 3237.05 µs） | 最小 diff：`_gqa_sw_fwd_varlen_kernel_round2_op1_oobguard.py` 相对 `..._round1_op1_pipe2.py` 的 6 行缩进改动（before/after 完整对照已存于 perf_opt/） | 同上 |
| **VP-C（新，D 类）** | D | **f16 链的标量门槛**：`T.vsub(f16 tile, <f32 PrimExpr>, f16 tile)` 在本工具链**编译失败**（`codegen_npuir_api.cc:2019` "element types of two operands should be the same"），`T.Cast("f16", thr)` 虽合法但**整数精确上限 2048** ⇒ 掩码阈值类 f16 链在 KV-cache（大 offset）域不可用 | 本 opt_log §5 round 5 + `logs/round5_probe_f16.log` | `perf_opt/probe_f16_chain.py`（自包含：直接复现该 InternalError） | 同上 |
| **VP-D（更新既有 `PL-1.12`）** | P | 深度 2 任务流水的**varlen 形态**：ragged 解码需在 advance 里**重放**一次（ghost-safe：`real_m==0 ⇒ lc=0`），且 `2·L_max_bound ≤ 15` 的判据要用**解析上界**（含 `+2` 余量）而非实测 loop_count —— 本例 `bn=128` 时 L_max=11 直接封死 pipe2 | §5 round 1/round 4（bn128 +26%） | 已回写 `pattern-library/attention.md`（见 10.3） | 同上 |

### 10.2b Round 10 追加的 Value Point Proposals（同样带证据三件套）

| id | vp_type | 提案 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| **VP-E（新，D 类）** | D | **910B3 · GQA-SW-varlen 稳态的 Cube 四管道账**（8b-long fp16，20 launch）：`mte2 0.650 / mte1 0.307 / cube 0.243 / fixpipe 0.315`，四者之和 = **1.51–1.54 × 墙钟**（即结构本身已拿到 ~50% 重叠）；单管道（mte2）理论上界 **1.52×**。用于判断「重叠还有没有空间」时必须先看这个比值——它把「串行」与「已重叠但某管道独大」区分开 | 本 opt_log §5 Round 10；`profiles/r10_readout/{base,final}_*`（70b-long 同构：0.648/0.303/0.241/0.312，和 1.52×） | `parse_profile.py`（本目录，直接输出 aic_*_ratio 与四管道和/墙钟） | tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-04 |
| **VP-F（新，D 类）** | D | **零初始化的 `T.clear` 形态**：当 kernel 用「zero 参数 + `T.copy(zbuf, buf)`」做 NaN 防护的零初始化时，`T.clear(buf)` 是等价且**编译路径不同**的写法——在本 kernel 上，第 4 块活 L1 缓冲的 `T.copy(zbuf, buf)` 必 SIGSEGV 而 `T.clear(buf)` 26 s 通过。⇒ 零初始化优先用 `T.clear`，可绕开 copy 路径的编译器缺陷 | `logs/r10_m1/PROBE_LADDER.md`（A1 失败 / C1 通过，同一缓冲同一 init 语义） | `logs/r10_m1/probes/_probe_alloc_c1.py`（当前最优 + 1 块 L1 + `T.clear`，可跑） | 同上 |
| **VP-G（新，R 类）** | R | **Stage 4 工件丢失 / final sha 漂移的迁移纪律**：若 `perf_records.jsonl` 的 `phase=final` sha 与磁盘 `perf_opt/{op}.py` 不符（且其引用的 `profiles/`、`logs/`、`bench.py` 已丢失），处理 = ① **整份归档**到 `perf_opt/history/`（append-only，不改不删）并写 `history/README.md` 说明原因；② 同 session 用现存文件对**全部 tune workload** 重测 baseline + final（各一条），生成自洽新集（单一 `candidate_id`、sha 与磁盘一致）；③ opt_log §8 记录迁移段 + Final 表换新值。验收 = `statectl gate 4` 通过（本轮实测通过） | 本 opt_log §8「记录迁移」段；`perf_opt/history/README.md`；gate 4 JSON | `logs/r10_m1/run_all.sh`（16 次 msprof 全量重测的可复用脚本） | 同上 |
| **VP-H（新，P 类）** | P | **「重叠候选」的负向判据（可迁移）**：若一个核内重叠候选的唯一可编译形态是「把与跨引擎 flag 无关的装载提到 gate 之前」，且实测 **mte2 忙比下降而四管道和/墙钟也下降**（本例 0.634→0.618，1.507→1.474）、墙钟回退 ⇒ 该 kernel 的重叠缺口在 **L1 端口读写互拖**（MTE2 写 vs MTE1 操作数读），不是 MTE2 空转；此时继续加缓冲/加深流水无意义（与 `PL-1.18` 同向，跨到 910B3/attention 族复证） | 本 opt_log §5 Round 10 的 M1c 表（两长 workload 同向 +1.4%） | `_gqa_sw_fwd_varlen_kernel_round10_m1c_vfirst.py`（8 行 reorder diff，可直接 A/B） | 同上 |

**Round 10 的 skill 流程问题（过程复盘）**：

1. **`risk: 自伤式 pkill`**：本轮两次用 `pkill -f "bench.py launch"` 清理后台 runner，该 pattern 命中**当前 shell 自己的命令行**，导致正在执行的命令块整段被静默杀死（表现为「无输出」+ 后续文件不生成）。纪律：清理后台进程只用 `kill <pid>` 或 `pkill -x <exact-name>`，禁止用会出现在当前命令行里的 pattern；后台任务用 `setsid` + 独立日志文件，前台用 `timeout` 兜底。
2. **`gap: 编译探针缺「快速失败」通道」**：本 kernel 自带 `_COMPILE_RETRIES=40` × 外层 24 次的 SIGSEGV 重试，一次失败形态要烧 15–90 min。本轮用 `timeout 400~900` + `grep -c "err code"` 在 2 min 内就能判定「SIGSEGV 循环」，但「不收敛（100% CPU 无输出）」只能靠满超时发现。建议 kernel 模板提供 `--compile-probe`（重试上限 1、超时 10 min）供 Stage 4 的可行性探针使用。
3. **`gap: perf_opt 工件跨 session 丢失`**：上一 session 的 `bench.py`/`profiles/`/`logs/`/`workload_inventory.json` 全部消失（仅剩 3 个文件），导致 final 记录无法对账。本轮已把 `bench.py`、`parse_profile.py`、`run_all.sh` 一并落在 `perf_opt/`，并把探针与日志按 `logs/r10_*/` 归档，便于下次重建。

### 10.3 pattern-library 回写（Phase 4 ED-C，本轮已实际落盘，`kb_lint: 0 failures`）

| 落盘位置 | 内容 | repro（已跑通） |
|---|---|---|
| `pattern-library/constants.md` **新条目 `CONST-vector-op-cost-910B3`** | VP-A：910B3 成块 v-op `≈0.61 µs/op/AIV`（[32,256] f32）+ tiny `[32,1]` op 不计费 + 壁钟预测式（final 预测 2.06 ms / 实测 2.00 ms） | `repro/PL-1.11b-mask-guard.py`（always-on 7.018 vs guarded 5.193 µs/block） |
| `pattern-library/attention.md` `PL-1.11` 追记 | VP-B：仅边界块需要的掩码子链必须加**运行期守卫**（−36.2%），"折叠成更少 op" 无效 ⇒ 判据 = 无条件执行的 tile op 数 | 同上 |
| `pattern-library/attention.md` `PL-1.12` 追记 | VP-D：pipe2 的 varlen 形态（advance 重放 ragged 解码，ghost-safe）+ flag 预算判据改用解析上界 `2·L_max_bound ≤ 15` | `repro/PL-1.9b-gqa-sw-pipe2.py`（既有） |
| `pattern-library/traps-compiler.md` **新条目 `TRAP-vector-f16-scalar-dtype`** | VP-C：f16 tile + 运行期 f32 标量在 v-prefix op 上编译失败（两次复现）；`T.Cast` 亦不可用；f16 阈值 2048 精确上限 | `repro/TRAP-vector-f16-scalar.py`（kernel 尺度探针，直接复现 InternalError） |
| `pattern-library/INDEX.md` | 上述两个新条目登记 + PL-1.11/PL-1.12 一句话刷新 | — |

全部条目带 `origin_task = grouped_query_attention_sliding_window_varlen-_gqa_sw_fwd_varlen_kernel` + 工具链戳（tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-10-17）。

### 10.4 Round 13 追加的 Value Point Proposals（vp_type + 证据三件套）

| id | vp_type | 提案 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| **VP-I（新，D 类）** | D | **GQA head 共享（G=2）的收益账（910B3）**：`heads/heads_kv = 4` 时两个 query head 共享一个 KV head 的 K/V，8b-long fp16 `mte2_ratio 0.667→0.433`、`GM_to_L1 126,064→77,280 KB（−38.7 %）`、`Task Duration 2017.22→1767.48 µs（1.141×）`；70b-long 1.145×、8b-short 1.102×、bf16 1.125–1.144×。⇒ 当 K+V 占 mte2 字节 >95 % 时，head 共享是比"加深流水"更可靠的降 mte2 手段 | §5 Round 13.2/13.4；`profiles/round13/ab_8b-long_float16_{base,g2}_0` | `bench.py` + `logs/round13/ab13.py`（交错 A/B，直接可跑） | tilelang 0.1.2+cebcec9d / CANN 9.0.0 / Ascend910B3 / 2026-11-05 |
| **VP-J（新，P 类）** | P | **UB 瘦身的"别名声明"技巧**：两个生命周期不重叠的 UB tile（pass-1 的 N 与 pass-2 的 D）可把后者声明为前者的前缀 Python 别名（`[half,dim]` ⊂ `[half,bn_eff]`），**op 数不变、−24,576 B**；但切片必须用**静态 extent**（`[0:half,0:dim]`）才能满足 `vcast`/`vadd` 的形状推导，且 `getBroadcastDim` 比较的是**声明形状**而非 view 形状（`codegen_npuir_api.cc:157`）⇒ 跨 head 的维度要放在列上（`[half,G*dim]`），逐 head 的 `m`/`ell` 用 `[half,1]` scratch 桥接 | §5 Round 13.1；`_round13_shrink.py`（L0 pass、8b-long 中性） | `_gqa_sw_fwd_varlen_kernel_round13_shrink.py` / `_round13_g2.py` | 同上 |
| **VP-K（新，R 类）** | R | **`gate 4` 的 Final 表解析边界**：`S4-PERF-RECORDS-RECON` 从 `### Final Performance Test Data` 起**逐行解析所有 5 列表格，直到下一个 `## ` / `### ` 标题**；若在表后追加 4~5 列的「改进/证据」表（常见写法），会被当成 Final 表并报 "Final 表非法数值"。纪律：Final 表之后任何辅助表必须用 `### ` 子标题隔开（本轮实测） | `statectl gate 4` 第一次失败 JSON（9 条 `S4-PERF-RECORDS-RECON`）与修复后 pass | 无需 repro（规则可从 `gate_lint.py::_perf_records_workload_lint` 直接读出） | 同上 |

**Round 13 的 skill 流程问题（过程复盘）**：

1. **`risk: 设备争用导致 L1 只跑 6/26`**（上一 dispatch）：并发 benchmark 作业使 9 s 的用例变成 50 s。纪律：全量回归前先确认设备空闲（`npu-smi info` / 无 `bench_*` pytest 进程），否则只报「已完成项 + 未跑项」。本轮设备空闲，26/26 一次通过。
2. **`risk: 后台任务被 shell 超时连带杀死`**：`cd X && nohup ... &` 会把 `cd` 也放进后台子 shell，导致后续前台命令落在错误目录；且工具超时会连带终止进程组。本轮的可用写法是 `setsid nohup <cmd> > <abs-log> 2>&1 < /dev/null &`，日志与轮询一律用**绝对路径**（本轮两次踩坑，`final_all_aborted_1054.log` 为第一次被截断的日志）。
3. **`gap: 记录重写 vs append-only`**：转正时若旧 `phase=final` 行的 `candidate_id` 与最终文件不同，gate 的「同一 kernel 只保留一个 final candidate」规则要求**整份重建**；做法 = 完整归档旧文件到 `history/`（本轮 `history/perf_records_r9.jsonl`，18 行原样）后在原路径重建 16 行，并在 §8/§13.4 留痕。
