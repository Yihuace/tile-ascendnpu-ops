# `_gqa_sw_fwd_kernel` Stage 4 调优日志（GQA 滑动窗口 attention · Expert 两相位 · Ascend910B3）

- **场景**：`scenario=optimize`（`stage_plan=[4, 回归]`），算子目录
  `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/`
- **性能目标类型**：`best_effort`（无硬性数值门槛；以迁移基线 §12.1 为改进参照）
- **主指标（唯一时延口径）**：`msprof op` `Task Duration(us)`，`--launch-count=20 --warm-up=5`，取 20 次 launch 记录的 **median**；captured op = `_gqa_sw_fwd_main_mix_aic`，`--kernel-name=_gqa_sw_fwd_main`
- **噪声阈值**：3%（小 kernel 单 run 抖动实测 0.5%–3%）
- **最大迭代数**：10（本轮实际 5 轮）
- **工具链戳**：tilelang `0.1.2+cebcec9dd203208e30657c667cdc87d6526d1829`（`/mnt/workspace/Agentkj/tilelang`，git HEAD `cebcec9`）/ CANN `9.0.0` / Ascend910B3 **20 AIC / 40 AIV** / 1800 MHz（msprof `Current Freq` 实读）
- **回归纪律**：`python perf_opt/_gqa_sw_fwd_kernel.py --level all` 全绿为采纳前置（L0/L1 阻塞，L2/Boundary 告警不阻塞，本轮实测 32/32 全 PASS）
- **算子类型判断（Phase 0）**：**mix**（MixCV Expert：Cube `T.gemm`×2 + Vector v-prefix 链；DESIGN §5.1）

---

## Phase 0 摘要

- `_gqa_sw_fwd_kernel.py`（Stage 3 交付，**只读**，sha256 `d442506bf593…`）、DESIGN §0.6/§1.6/§2/§3/§4.5/§5/§6/§7/§9/§11/§12 分级读取完毕；硬件上下文与 pattern-library INDEX 已读。
- **版本失效核对（E-5）**：pattern-library 内条目戳（910B2C/24 AIC 时代）与本机（910B3/20 AIC/40 AIV）不同族；本轮把「跨 SoC 未重验」的硬件常数（fabric BW、向量发射率）一律当**待重验**处理，只以实测 profile 决策；`kb_stale_check.py` 报 `stale_count=268`，未引用任何 `stale/overturned` 条目作为依据。
- **首轮必查项**：
  - *向量化轴与布局重估*：DESIGN §1.6.3（无该小节；§1.6.2 向量化覆盖）已把 Vector 链锁在 `[half, bn]` 行主序连续轴（`bn_eff` 为最内连续维）。本机实测 `aiv_vec_ratio` 0.33–0.60、无标量化告警（`TILELANG_DUMP_IR` 未见 `scf.for` 元素级展开），布局路线**无需重估**；本轮不动布局。
  - *编译器陷阱版本戳与 origin_task 核对*：命中条目 `TRAP-npuir-parallel-elemcast-load-scalarize`（origin: engram_gate_conv_fwd/bwd）、`VP-2026-0283`（engram_decode）、`CG-2026-0005`（attention S/P 往返）、`PL-1.9-twophase`（fa 域）——本文件载入路径全部是**整块 `T.copy` + 整 tile `vcast`**（无元素级 `T.cast` 读 GM）、online softmax 的 running max/ell 是 `[half,1]` **行向量** `vmax`/`vadd` 而非 `T.Parallel` 内 loop-carried 累加，**两类陷阱在本实现中均不成立**（实测 `aiv_scalar_ratio` 0.33–0.37，远低于 `>50%` 的标量化判据）。
  - *DESIGN §1.6.3 实验裁决*：DESIGN 无「主选+备选+实验裁决计划」三件套（§1.6.3 不存在；掩码主选已在设计期 probe P2 裁决），**无强制 A/B 项**。
- **搜索同类既有实现（T-1 参照结构 diff，第 3 轮前执行）**：`examples/TileOPs/.../multi_head_attention/.../perf_opt/_gqa_prefill_fwd_kernel.py`（同族 MHA prefill 调优终态，DESIGN 的母本）。结构 diff 见表 R3 的「参照结构 diff」小节——**该 diff 直接产出了本轮唯一的有效结构改动**。

---

## Performance Test Data（baseline，`phase=baseline`，round 0）

`workload_inventory.json`：benchmark 参数化入口
`examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_sliding_window.py::test_gqa_sliding_window_fwd_bench`
展开 `load_workloads("GroupedQueryAttentionSlidingWindowFwdOp")` = **5 workload × 2 dtype = 10 case**：

- `smoke` ×2（label `gqa-sw-fwd-smoke-s512-h8-kv2-d64`，独立 `-` 分隔 token `smoke`）→ `kind=smoke`，**仅精度**，无性能记录；
- `tune` ×8 = {8b-short-w256, 8b-long-w1024, 70b-short-w256, 70b-long-w1024} × {fp16, bf16} → 全部参与 baseline / 迭代 / winner；
- `skipped`：0（benchmark 无 skip 标记）。

### Table A：baseline 性能与诊断（msprof op Task Duration median of 20；指标为 20 次 launch × 20 核均值）

| workload_id | Task Duration(us) | aic_cube% | aic_mte2% | aic_fix% | aiv_vec% | aiv_wait(us) | GM>L1(KB) | GM>UB(KB) | L0C>GM(KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b-short-w256-fp16 | 485.13 | 0.100 | 0.249 | 0.095 | 0.326 | 128.62 | 13107.2 | 1843.2 | 3686.4 |
| llama-3.1-8b-short-w256-bf16 | 490.24 | 0.099 | 0.255 | 0.126 | 0.305 | 124.70 | 13107.2 | 3686.4 | 7372.8 |
| llama-3.1-8b-long-w1024-fp16 | 1303.88 | 0.186 | 0.447 | 0.182 | 0.471 | 589.80 | 63078.4 | 9216.0 | 18432.0 |
| llama-3.1-8b-long-w1024-bf16 | 1299.17 | 0.186 | 0.481 | 0.245 | 0.447 | 566.13 | 63078.4 | 18432.0 | 36864.0 |
| llama-3.1-70b-short-w256-fp16 | 483.47 | 0.100 | 0.251 | 0.094 | 0.325 | 128.67 | 13107.2 | 1843.2 | 3686.4 |
| llama-3.1-70b-short-w256-bf16 | 491.19 | 0.099 | 0.255 | 0.126 | 0.304 | 124.70 | 13107.2 | 3686.4 | 7372.8 |
| llama-3.1-70b-long-w1024-fp16 | 1338.74 | 0.180 | 0.432 | 0.177 | 0.486 | 618.43 | 63078.4 | 9216.0 | 18432.0 |
| llama-3.1-70b-long-w1024-bf16 | 1336.36 | 0.181 | 0.466 | 0.240 | 0.460 | 590.63 | 63078.4 | 18432.0 | 36864.0 |

- raw profile：`profiles/baseline/round0/baseline_8ba8496a/<workload_id>_<hash>/OPPROF_*/`
- 与迁移基线（DESIGN §12.1，report `20261016_161519_904205`）逐项偏差 −1.6%…+2.0%，**采数链路一致** ✓（全部 8/8 valid：`OpBasicInfo` 记录数 = 20、captured op = `_gqa_sw_fwd_main_mix_aic`）。
- **smoke 排除说明**：`gqa-sw-fwd-smoke-s512-h8-kv2-d64`（fp16/bf16）按契约不做性能采集；其精度由 `--level all` 的 L0-1（fp16，含 SDPA 交叉校验）与 L0-2（bf16）覆盖，最终 `precision_pass=true`。

### 设计估算 vs 实测偏差（D-2 回填，hardware-cost-model §3）

| 口径 | 估算 | 实测 baseline | 偏差 | 失准项归因 |
|---|---:|---:|---|---|
| DESIGN §1.6.0 朴素发射下界 | 1.6–2.5 ms | 1.304 ms（8b-long fp16） | 实测**低于**下界 1.23× | 发射项常数 `0.5µs/op`（`CONST-vector-launch-overhead`）为 910B2C 跨 SoC 未重验值 → 高估发射成本；实机 op 数 ~17/块 × 7680 块 ÷40 AIV 的实际成本低于估算 |
| DESIGN §1.6.0 MHA 同族锚点外推 | ~0.99 ms | 1.304 ms | +32% | 锚点为**全 causal 9216 块**、且 MHA 母本调优终态已含 depth-2 跨任务流水；本 baseline 是**单槽**结构（未含该优化）→ 差距由结构项解释，非流量项 |
| 收敛区间判定 | 1.0–2.5 ms | 1.304 ms 在区间内（靠乐观端） | — | 设计期估算**未失准**（无 >2× 偏差），不构成 `[DESIGN_LIMIT]` 归因证据 |

### baseline 现象与调优排序（Phase 1 诊断）

1. **两引擎同时 >35% 空转**：8b-long fp16 `aiv_scalar_wait_time` = 589.8us（**45%** 时长）、Cube `aic_scalar_wait_time` 侧 flag 自旋占比 81%；`aic_cube_ratio` 仅 0.19。
2. **等待的分布指向「任务边界」**：baseline 的 flag 等待按 id 分解（`PipeUtilization` 单核行）为
   Cube `id0=303us / id1=143 / id2=123 / id3=103 / id4=103 / id5(TASKDONE)=306us`；
   Vector `id0=343us / id1..4=36/35/60/107us`。
   → **flag 0（任务首个 P 等待）与 `FLAG_TASKDONE`（任务排空栅栏）合计占 Cube 等待的 56%**，即**任务级流水填充/排空**是主因，而非块内 FLOPs 或带宽。
3. **不是带宽瓶颈**：`GM_to_L1_bw_usage_rate` 13.5%（short）/ 21.5%（long），`L0C_to_GM` 4.8%/7.9%，`read_main_memory` 36MB/core → HBM ≈ 0.55TB/s。约 100 GB/s/core 的 DMA 速率是**每核 MTE 发射率**，不是 fabric 上限。
4. 排序（负担 + 结构收益）：**8b-long / 70b-long（~1.3ms，绝对耗时最大、band 最长）→ 8b-short / 70b-short（~485us，band 短、固定开销占比高）→ bf16 与 fp16 交替复核**。

---

## Iteration Log

> 每轮固定 base = 当前 best；每个分支只改一个主要优化点；每分支先跑该文件自带 `--level L0`（阻塞），通过后 `msprof op` 采集。所有数值从 `perf_records.jsonl` + raw profile 直接生成（未手抄）。

### 参照结构 diff（T-1 morph-ladder 轻量版，第 3 轮前）

| 维度 | 参照实现（MHA `perf_opt/_gqa_prefill_fwd_kernel.py`，同族调优终态） | 本轮 base（baseline） | 差异是否可移植 |
|---|---|---|---|
| grid 组织 | 1-D persistent `T.Kernel(20)` | 同 | — |
| **数据流分相位** | 两相位 + **depth-2 跨任务流水**（`ws_*[kernel, 2, ...]`、flag id 偏移 `i + slot*nk_total`、每轮 `[hoist+p1(T+1)]; p2(T)`、**无 TASKDONE 栅栏**） | 两相位 + **单槽** + `FLAG_TASKDONE` 全栅栏 | ✅ **可移植（本轮 r3_op1 采纳）** |
| **Q 驻留** | 专用 `l1_q`（不占 `l1_a`） | Q 与 P 共用 `l1_a` | ✅ 可移植（depth-2 的前置条件） |
| 片上驻留 | `l1_b` 单缓冲 + zbuf 零初始化 | 同 | — |
| dtype 路径 | fp16 也把 `ws_s` 升 f32（省 vcast） | fp16 `ws_s`=f16 | ⚠️ 本轮实测（r2_op2）**无收益**，见下 |
| 片内重叠 | `auto_multi_buffer=False`、无 `T.Pipelined` | 同 | —（参照也未做，本轮独立验证两者均无效，见 r1/r4） |

**结论**：唯一显著结构差异 = **depth-2 跨任务流水**（参照实测 1191.6→989.3us，−17%）。本轮据此派生 r3_op1。

### Round 1 — `r1_op1_amb`：Cube/Vector 自动多缓冲（`NPUIR_ENABLE_AUTO_MULTI_BUFFER=True`）

- **依据**：baseline Cube 三管道（mte2 0.447 + cube 0.186 + fix 0.182 ≈ 0.815）接近串行，`NPUIR_ENABLE_AUTO_MULTI_BUFFER` 默认 False；打开后 BishengIR 可对 K/V 载入做双缓冲。
- **分支**：`_gqa_sw_fwd_kernel_round1_op1_amb.py`（单行 pass_config 改动，base = baseline）。

| round | candidate_id | workload_id | L0 | Task Duration(us) | current best(us) | delta | aic_cube% | aic_mte2% | aiv_vec% | aiv_wait(us) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | r1_op1_amb | llama-3.1-8b-long-w1024-fp16 | pass | 1300.93 | 1303.88 | +0.2% | 0.186 | 0.447 | 0.471 | 590 |

- **判定**：`config_no_gain`（0.2% « 3% 噪声）。`PipeUtilization` 各 ratio 与 baseline 逐项一致（mte2 0.4468→0.4468）→ **该 pass 在本 Expert 双 Scope + 显式 `T.rs` 结构上不产生任何双缓冲**。current best 保持 baseline。

### Round 2 — 静态 extent 特化（`r2_op1_static`）与 fp16 `ws_s` 升 f32（`r2_op2_sf32`）

- **r2_op1 依据**：Cube/Vector 的 `T.copy` extent 全是 `T.min(bn_eff, seq_len-n_lo)` / `T.ceildiv(...)` 运行时表达式；manifest 域 `seq_len % bn_eff == 0`（512/2048 vs 256）→ 全部可静态化（`has_tail=False`、`has_band=False`）。假设：动态 extent 阻碍 DMA 描述符优化与多缓冲。
- **r2_op2 依据**：参照实现在 fp16 也把 `ws_s` 升 f32（省一次 `[half,bn]` vcast，实测 −6.7%）。本算子 `prescale` 在 f32 域，vcast 路径相同。

| round | candidate_id | workload_id | L0 | Task Duration(us) | current best(us) | delta | aic_cube% | aic_mte2% | aiv_vec% | aiv_wait(us) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | r2_op1_static | llama-3.1-8b-long-w1024-fp16 | pass | 1312.16 | 1303.88 | −0.6% | 0.183 | 0.440 | 0.491 | 618 |
| 2 | r2_op2_sf32 | llama-3.1-8b-short-w256-fp16 | pass | 490.60 | 485.13 | −1.1% | 0.099 | 0.252 | 0.304 | 124 |
| 2 | r2_op2_sf32 | llama-3.1-8b-long-w1024-fp16 | pass | 1289.53 | 1303.88 | +1.1% | 0.188 | 0.482 | 0.451 | 555 |

- **判定**：两者均 `config_no_gain`（< 3%）。**假设证伪**：① 动态 extent 不是 MTE2 慢因（静态化后 mte2 0.447→0.440，无变化）→ 每核 ~100 GB/s 的 DMA 速率是**发射率**而非描述符开销；② `ws_s` f32 反而使 `L0C→GM` 翻倍（18432→30720KB），short 微降、long 微升，**净中性**（精度余量/UB 反而变差，不采纳）。current best 保持 baseline。

### Round 3 — `r3_op1_pipe2`：**depth-2 跨任务软件流水**（✅ 采纳）

- **目标现象**：Cube 等待的 56% 落在 `flag0`（任务首个 P 等待）与 `FLAG_TASKDONE`（任务排空栅栏）→ 任务级填充/排空。
- **优化点（单一主要改动）**：把 Cube 流改成 depth-2 跨任务流水，Vector 流按 slot 交替。具体：
  1. `ws_s / ws_p / ws_o` 增加任务槽维 `n_slots = 2`（flag 预算 `2*L_max ≤ 15`；本域 `L_max ≤ 5` → 10 个 id）；
  2. flag id 改为 `l + slot*L_max`（每 slot 独占一个 id 窗口）；
  3. Cube 每轮顺序改为 **[hoist Q(T+1) + pass-1(T+1)（写 slot_n）] → [pass-2(T)（读 slot）]**，prologue 先做 task 0 的 hoist+pass-1，**用 T+1 的 S 生产填掉 T 的 P 等待**；
  4. **删除 `FLAG_TASKDONE` 全栅栏**（跨任务 WAR 由「slot 交替 + flag id 窗口」闭合：`ws_s[slot]` 的下一次写入 p1(T+2) 在 p2(T+1) 之后，而 p2(T+1) 等 P-ready(T+1,!slot) → 蕴含 Vector p1(T+1) > Vector p1(T) 已消费）；
  5. 新增专用 `l1_q`（Q 不再与 P 共用 `l1_a`）——depth-2 的前置条件；
  6. `2*L_max > 15` 的宽 band 域自动回落到原单槽 + TASKDONE 形式（`keep_barrier`）。
- **分支**：`_gqa_sw_fwd_kernel_round3_op1_pipe2.py`（base = baseline），全部 8 个 tune workload 实测。

| round | candidate_id | workload_id | L0 | Task Duration(us) | current best(us) | delta | aic_cube% | aic_mte2% | aiv_vec% | aiv_wait(us) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | r3_op1_pipe2 | llama-3.1-8b-long-w1024-fp16 | pass | 1060.31 | 1303.88 | **+18.7%** | 0.228 | 0.570 | 0.603 | 640 |
| 3 | r3_op1_pipe2 | llama-3.1-8b-long-w1024-bf16 | pass | 1040.37 | 1299.17 | **+19.9%** | 0.232 | 0.632 | 0.578 | 605 |
| 3 | r3_op1_pipe2 | llama-3.1-70b-long-w1024-fp16 | pass | 1058.79 | 1338.74 | **+20.9%** | 0.229 | 0.572 | 0.604 | 640 |
| 3 | r3_op1_pipe2 | llama-3.1-70b-long-w1024-bf16 | pass | 1043.69 | 1336.36 | **+21.9%** | 0.232 | 0.633 | 0.577 | 607 |
| 3 | r3_op1_pipe2 | llama-3.1-8b-short-w256-fp16 | pass | 350.92 | 485.13 | **+27.7%** | 0.138 | 0.366 | 0.436 | 140 |
| 3 | r3_op1_pipe2 | llama-3.1-8b-short-w256-bf16 | pass | 343.30 | 490.24 | **+30.0%** | 0.141 | 0.383 | 0.422 | 128 |
| 3 | r3_op1_pipe2 | llama-3.1-70b-short-w256-fp16 | pass | 353.41 | 483.47 | **+26.9%** | 0.137 | 0.364 | 0.445 | 145 |
| 3 | r3_op1_pipe2 | llama-3.1-70b-short-w256-bf16 | pass | 346.14 | 491.19 | **+29.5%** | 0.140 | 0.382 | 0.431 | 133 |

- **机制核对**：cube 利用率 0.186→0.228、mte2 0.447→0.570、aiv_vec 0.471→0.603 —— 两引擎的**有效忙碌率同时上升**、`aiv_wait` 绝对量基本不变（590→640us，但分母缩小）→ 提升来自「等待被真实工作填满」而非减少算法工作。short 域收益更大（+27~30%），因为其 `loop_count ≤ 2`、任务级填充占比更高，正是 depth-2 针对的现象 ✓。
- **合并**：winner 先原样 `cp` 为 `perf_opt/_gqa_sw_fwd_kernel.py`（sha256 `992737ec643d…`，与分支文件逐字节相同 → 8 条 candidate 记录即合并后复测记录）；随后按本次任务的硬约束**只追加**输出独立性自检（`check_output_independence` + 两处调用，`diff` 计 34 行新增、**`@T.prim_func` 主体 0 行改动**，编译产物不变），最终文件 sha256 = `aae0ba6603dd…`。本 kernel 的 8/8 tune workload 均已在本分支上复测且**全部正向**，smoke 精度由 L0-1/L0-2 覆盖。current best ← `r3_op1_pipe2`。

### Round 4 — `r4_op1_pl_ns2`：块内 `T.Pipelined(num_stages=2)`

- **依据**：r3 后 Cube 三管道之和（mte2 0.57 + cube 0.23 + fix 0.23 ≈ 1.03 ≈ 100% 时长）仍接近串行；`T.Pipelined` 的软件流水若生效可把 K/V 载入与 gemm/fixpipe 重叠，理论上限 wall ≈ mte2（0.57）→ 潜在 −40%。
- **分支**：`_gqa_sw_fwd_kernel_round4_op1_pl.py`（把 4 处 `T.serial` 循环改为 `T.Pipelined(..., num_stages=ns_)`，base = r3），以 `--config 64,256,2` 采集。

| round | candidate_id | workload_id | L0 | Task Duration(us) | current best(us) | delta | aic_cube% | aic_mte2% | aiv_vec% | aiv_wait(us) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | r4_op1_pl_ns2 | llama-3.1-8b-long-w1024-fp16 | pass | 1060.81 | 1060.31 | −0.0% | 0.228 | 0.568 | 0.603 | 640 |

- **判定**：`config_no_gain`（逐项 ratio 与 r3 相同，说明 pass 未改变生成代码）。与 r1 合起来证明：**本工具链的 npuir Expert 双 Scope + 显式 `T.rs` + 手写 `sync_block_set/wait` 结构下，`auto_multi_buffer` 与 `T.Pipelined` 两个「自动重叠」旋钮都是惰性的**；片内重叠只能手写（需 l1_b/l0_c 的双缓冲 + 动态偏移切片的 gemm 操作数，本轮未验证可行）。current best 保持 `r3_op1_pipe2`。

### Round 5 — `r5_op1_nqb2`：2 个 q-block 共享一次 K/V 载入（**blocked**）

- **依据**：r3 后 Cube 的主要余量是 mte2（0.57 ≈ 604us，63MB/core）。按块计：K 64KB + V 64KB + P 32KB = 160KB/块，其中 **K/V 占 80%**，而一个 K/V tile 只服务 64 行 q。把 `nqb=2` 个相邻 q-block 编进同一 task（`bn_eff/bm = 4` 为偶数 ⇒ (2t,2t+1) 对的滑动窗口 band 完全相同，union 不增加块数），可把 Cube 的 GM 读降到 96KB/q-block（**−40%**，Cube mte2 578→~347us），且 flag 握手数减半。Vector 仍逐 q-block 处理（UB 只增 1 个 `acc_o` + 6 个 `[half,1]` 行向量）。
- **实现**：`_gqa_sw_fwd_kernel_round5_op1_nqb2.py`（在 r3 之上：`nqb=2` q-block 组解码 + union band + `l1_q`/`l1_q_b` 双 Q 缓冲 + Vector 侧 qb0/qb1 显式展开 + `ws_*` 行维 `nqb*bm`）。
- **结果**：`L0-1`(smoke fp16)、`L0-2`(smoke bf16) **PASS**；`L0-3`(4,32,8,512,128) 在 `bench.py --case 8b-short` 上 **600s 无任何输出（进程 CPU 仅 18s user / 16s sys，即阻塞等待）**，`--level L0` 全跑 2400s 超时未完成。判定为 **blocked：doubled prim_func 体（~2× 语句数）在本容器 `bishengir-compile` 上不收敛**（该外部编译器在本机本就概率性 SIGSEGV，产物内 6 次重试在此体量下无法收敛）。未写入 `perf_records.jsonl`（契约：blocked/invalid 分支不写记录），证据保存在 `logs/candidate/round5/` 与 `logs/final/verify_8bshort.log` 之外的本轮尝试日志。
- **判定**：**rollback**（不采纳）。current best 保持 `r3_op1_pipe2`。

### 中止判断

- success（best_effort）：8/8 tune workload 均有调优结论 + 最终全量复测 + smoke 精度通过 → **满足**。
- 结构候选是否穷尽：① depth-2（采纳）；② 自动多缓冲 / `T.Pipelined`（实测惰性）；③ 静态 extent（证伪）；④ `ws_s` 升 f32（中性）；⑤ `nqb=2` K/V 摊薄（编译不收敛，blocked）；⑥ GQA 头组批处理（与 ⑤ 同源收益、需 G 个 `acc_o`：`bm=64/bn=256` 下 UB 需 202KB > 192KB，**容量硬限否决**）。→ 按 `blocked` 收束（剩余候选受编译稳定性与 UB 容量硬限约束）。

---

## Autotune Log

未使用 autotune。理由：`bm`/`bn_eff` 的可行域由 UB 容量与 flag 预算**双向夹死**（`bm=64`：`half=32` 时 `ub_f32_N[32,256]`+`ub_diff`+`ub_pen` 已用满 154KB/192KB；`bn_eff=256` 是 `2*L_max ≤ 15` 与 UB 的交点，`bn≥512` 需 `[32,512]` f32 三块 = +96KB 溢出），搜索空间实为单点；`num_stages` 由 r4 证明为惰性旋钮。故无配置搜索，参数决策以 DESIGN §5 的容量推导 + 本轮实测复核为准。

---

## 输出独立性断言（`torch.library.custom_op` 别名契约）

- **文件内自检**：`perf_opt/_gqa_sw_fwd_kernel.py` 新增 `check_output_independence(q, k, v, out, lse, tag)`，对 `out`/`lse` 两个输出与 `q`/`k`/`v` 三个输入取 `untyped_storage().data_ptr()`，断言
  ① 输出两两互异、② 输出与任一输入互不相同、③ 输入两两互异。该断言在 `_run_case()`（每个 L0/L1/L2/Boundary case 都走）与 `_contract_checks()` 中各调用一次。
- **实测结果（`python perf_opt/_gqa_sw_fwd_kernel.py --level all`，32 case + contract）**：
  - `[<case>-alias] PASS` **33 次**（32 case + contract），**0 次 FAIL**；
  - 抽样：`[L2-rightwin-64-alias] PASS: out=0x12c041d80a00 lse=0x12c0427f7200 | q/k/v=['0x12c042400e00','0x12c044c00000','0x12c044c40200'] -- pairwise distinct, outputs disjoint from inputs`
  - 结论：`out`/`lse` 是 kernel 的两个独立输出张量（`out_idx=[-2,-1]`），`lse` 的 `reshape` 是同一 storage 的零拷贝视图（不与 `out` 或任何输入共享 storage）→ **不触发 VP-2026-0302 类别名失败**。
- **端到端 `custom_op` 路径复核**（额外，脚本 `perf_opt/check_custom_op.py`，以 monkeypatch 模拟 conductor 的 wrapper 采纳翻转后直调 `npub::gqa_sw_fwd_wrapped_kernel`）：
  ```
  [custom_op] smoke: out=(2,512,8,64) lse=(2,8,512) alias-free=True finite=True
  [custom_op] 8b-short: out=(4,512,32,128) lse=(4,32,512) alias-free=True finite=True
  [custom_op] 70b-short-bf16: out=(2,512,64,128) lse=(2,64,512) alias-free=True finite=True
  [custom_op] PASS
  ```
  → torch 的 custom_op 别名校验器未抛异常，输出 storage 与输入 storage 两两互异。

---

## Final Summary

- **最终候选**：`r18b_g2_headgroup`（= `perf_opt/_gqa_sw_fwd_kernel.py`，sha256 `6acdc11cd5a78b8da77ea65ef6fdfbca44a545757b912054c4279af75f752d71`）
- **关键有效优化点（两条，均结构性）**：
  1. **G=2 head-group batching**（r18 移植，r18b 转正）：`heads/heads_kv = 4` 时把共享同一 KV head 的两个 query head 放进同一 block，K/V 的 `GM→L1` 按 head 对摊薄（实测 −38.5%）；
  2. **编译器稳定性 shim**（r18b）：`head_group` 循环改为 trace-time 展开（`range`，消除 `T.serial(1)` 触发的确定性 SIGSEGV）+ 每次 `bishengir-compile` 150 s wallclock 上限并按 SIGSEGV/超时重试（消除无限循环假死）。
- **smoke 最终精度**：L0 4/4 + contract、L1 8/8、L2 12/12、Boundary 8/8 = **32/32 PASS**（`All check passed!`，`logs/round18b/level_all_final.log`——**转正后最终文件 sha `6acdc11c…` 上实测**，13m55s，期间 18 次 `bishengir-compile` 失败全部由 shim 重试消化；`level_all_v2.log` 为 shim 前的同内容版本）；官方 report correctness **22/22 PASS**。
- **官方 report**：`reports/tileops/20261106_105557_262003_GroupedQueryAttentionSlidingWindowFwdOp`（run_id，`status=passed`，benchmark **10/10**，见 R18b.4）。
- **中止原因**：`success`（G=2 收益达 1.10× 以上并已转正；R18 遗留的两处编译器 SIGSEGV 已消除）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-8b-short-w256-fp16 | 485.13 | 317.84 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-8b-short-w256-bf16 | 490.24 | 308.76 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-8b-long-w1024-fp16 | 1303.88 | 962.62 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-8b-long-w1024-bf16 | 1299.17 | 1051.60 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-70b-short-w256-fp16 | 483.47 | 311.72 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-70b-short-w256-bf16 | 491.19 | 300.20 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-70b-long-w1024-fp16 | 1338.74 | 965.50 | final_r18b_g2_headgroup |
| examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main | llama-3.1-70b-long-w1024-bf16 | 1336.36 | 918.28 | final_r18b_g2_headgroup |

逐项提升（同一最终候选、同一文件 sha256 `6acdc11c…`，`phase=final`，`msprof op` median of 20）：
`8b-short-fp16` **−34.5 %（1.53×）**、`8b-short-bf16` **−37.0 %（1.59×）**、`8b-long-fp16` **−26.2 %（1.35×）**、`8b-long-bf16` **−19.1 %（1.24×）**、
`70b-short-fp16` **−35.5 %（1.55×）**、`70b-short-bf16` **−38.9 %（1.64×）**、`70b-long-fp16` **−27.9 %（1.39×）**、`70b-long-bf16` **−31.3 %（1.46×）**。

> 说明：baseline 列沿用 round 0 的 8 条 `phase=baseline` 记录（Stage-3 交付文件 sha `d442506b…`）；final 列为 round 18b 用**转正后文件**（sha `6acdc11c…`）复测的 8 条 `phase=final` 记录（`perf_records.jsonl` 第 30–37 行）。round 9 的 8 条旧 `final` 行在 round 18b 被重新归类为 `phase=merged`（它们度量的是当时并入 perf_opt 的 r3 winner），逐字节副本保存在 `history/perf_records_round9_final_rows.jsonl`——gate 4 要求每个 `(kernel_id, workload_id)` 恰有一条 `final` 且其 artifact sha 等于当前 perf_opt 文件，故旧行不能继续占用该 phase。

**最终候选逐 workload 汇总（round 18b 实测，全部来自 `perf_records.jsonl` 同一最终候选）**

| workload_id | baseline(us) | final(us) | delta | speedup | L0 |
|---|---:|---:|---:|---:|---|
| llama-3.1-8b-short-w256-fp16 | 485.13 | 317.84 | −34.5 % | 1.53× | pass |
| llama-3.1-8b-short-w256-bf16 | 490.24 | 308.76 | −37.0 % | 1.59× | pass |
| llama-3.1-8b-long-w1024-fp16 | 1303.88 | 962.62 | −26.2 % | 1.35× | pass |
| llama-3.1-8b-long-w1024-bf16 | 1299.17 | 1051.60 | −19.1 % | 1.24× | pass |
| llama-3.1-70b-short-w256-fp16 | 483.47 | 311.72 | −35.5 % | 1.55× | pass |
| llama-3.1-70b-short-w256-bf16 | 491.19 | 300.20 | −38.9 % | 1.64× | pass |
| llama-3.1-70b-long-w1024-fp16 | 1338.74 | 965.50 | −27.9 % | 1.39× | pass |
| llama-3.1-70b-long-w1024-bf16 | 1336.36 | 918.28 | −31.3 % | 1.46× | pass |

**profile 有效性说明（如实披露）**：round 18b 只用 `bench.py`（`msprof op`、`--kernel-name=_gqa_sw_fwd_main`、`--launch-count=20 --warm-up=5`、median Task Duration）复测了 8 条最终记录，本轮 raw（`profiles/round18b/final_*`）的派生列（`aic_*_ratio` / `GM_to_L1`）在 `OpBasicInfo` 中为 `null`；因此本表**只报告实测的 `Task Duration`**，不引用未在本轮采集的派生指标。G=2 的机制证据（`GM→L1 63248→38912 KB`、`aic_mte2 0.571→0.367`）来自 R18 §18.2 的同参数对照 profile（`profiles/round18/`），该对照的 G=2 与转正文件为同一 trace。

---

## 设计层天花板判定（`[DESIGN_LIMIT]`）

**未触发，不产出 `perf_feedback.md`**。按 `_shared/standards/perf-feedback.md` §1 双门槛逐条核对：

- **门槛①（设计层归因）**：本轮唯一有效改动是**结构/流水层**（跨任务流水深度），属 DESIGN §6.3「流水线优化」与 §7.2「同步点说明」的**既定候选空间内**（DESIGN §11 度量计划④明列「双槽 id-offset 变体（R-7，L_max ≤ 7 域）」），**不是**对 DESIGN 算法/设计假设的否定；剩余未采纳项（`nqb=2`、GQA 头组批处理）的障碍是**工具链编译稳定性**与 **UB 192KB 容量**，属实现约束，DESIGN §4.5 已把 UB 154KB/192KB 列为设计约束并给出 diet 候选，非「设计层不可解」。
- **门槛②（结构性加速估计 > 2×，或实测与 DESIGN 假设直接矛盾）**：实测 baseline 1.304ms 落在 DESIGN §1.6.0 的**乐观–保守收敛区间 1.0–2.5ms 内**（偏差 < 2×，D-2 表见上），**未推翻**任何 DESIGN 假设；本轮已实现 1.36×（short）/ 1.25×（long），剩余理论余量（片内重叠 ~1.15× + K/V 摊薄 ~1.15×）**远小于 2×**。
- 结论：**两项门槛均不满足** → 参数/结构级不足留在迭代内解决，不做受控逆向反馈。

---

## Skill Retrospective

### 流程问题（本次暴露）

1. **compile 体量硬限会伪装成「hang」**：r5 的 `nqb=2` 变体把 prim_func 语句数翻倍后，`bishengir-compile` 600s 无输出且无 CPU 占用；`--level L0` 2400s 超时。产物内的「SIGSEGV 区分性重试（6 次）」在这种体量下反而把「一次明显失败」放大成「12 分钟假死」。**建议**：在 `run_experiments.py` 模板里给 `run_l0` 加「单 case 超时 + 无输出即判 blocked」的快速失败，并在 `bishengir-compile` 外部调用外包一层 wallclock 上限。
2. **`parse_profile` 需要 profile 有效性前置校验**：本次一条 final 记录的 per-launch 指标被同机残留污染（ratio > 1），先被写进分析表才发现。**建议** profile-collection 增加机械校验：任一 ratio ∉ [0,1] 或 `NA` 占比 > 50% → 标记 `metrics_invalid`（仅保留 Task Duration），并在 `perf_records.jsonl` 的 `msprof_raw_path` 旁记录该标记。
3. **驱动脚本对「单 workload 采数」支持不足**：模板的 `baseline/final` 阶段强制 `--all`，中断后无法续采（本次首次 baseline 因后台进程被杀而丢 7/8，需临时改脚本放开 `--workload`）。**建议**模板默认允许 `phase=baseline|final` + 单 `--workload`（幂等续采）。

### Value Point Proposals

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A1 | **D**（实测数据） | GQA 滑窗 attention（Expert 两相位，band 裁剪）域：**depth-2 跨任务软件流水**实测 short −26~30%、long −18.7~21.9%；`aic_cube_ratio` 0.186→0.228、`aiv_vec` 0.471→0.603（等待被真实工作填满） | 本日志 Round 3 表 + `perf_records.jsonl` round 3（8 行）+ `profiles/candidate/round3/` | `perf_opt/_gqa_sw_fwd_kernel_round3_op1_pipe2.py`（对 baseline 的最小 diff = 槽维 + flag id 窗口 + p1(T+1)/p2(T) 顺序 + 删 TASKDONE + l1_q） | tilelang 0.1.2+cebcec9 / CANN 9.0.0 / 910B3 20AIC |
| VP-A2 | **D**（实测数据，负向） | 本工具链 npuir **Expert 双 Scope + 显式 `T.rs` + 手写 `sync_block_set/wait`** 结构下，`NPUIR_ENABLE_AUTO_MULTI_BUFFER=True` 与 `T.Pipelined(num_stages=2)` **均不产生任何代码差异**（逐 ratio 相同，±0.2%）：片内重叠必须手写双缓冲 | Round 1 / Round 4 表 + raw `profiles/candidate/round1|round4/` | `_gqa_sw_fwd_kernel_round1_op1_amb.py` / `_gqa_sw_fwd_kernel_round4_op1_pl.py`（各为单点改动） | 同上 |
| VP-A3 | **P**（模式方法） | **`PL-1.9-twophase` 的 depth-2 流水是该族「任务级填充/排空」的通用解法**：判据 = `PipeUtilization` 的 `aic_scalar_wait_id*` 分解中「flag 0 + TASKDONE」占等待 > 50%，且任务 `loop_count` ≤ 5；flag 预算 `2*L_max ≤ 15`；WAR 闭合靠 slot 交替（无需栅栏） | Round 3 现象分析（flag 0 = 303us、TASKDONE = 306us，占 Cube 等待 56%） | 同 VP-A1 的 diff 骨架 | 同上 |
| VP-A4 | **R**（流程规则） | **doubled prim_func 体会让 `bishengir-compile` 假死**：显式展开（如 `nqb=2`）使语句数翻倍时，编译 600s 无 CPU 占用、重试循环 6 次放大为 12min 假死。规则：结构改动若使语句数 >1.7×，先以单 case + wallclock 上限探测编译可行性，再决定是否投入 | Round 5（`logs/` 中的 600s/2400s 超时记录） | `_gqa_sw_fwd_kernel_round5_op1_nqb2.py`（保留） | 同上 |
| VP-A5 | **C**（案例索引） | 案例：`grouped_query_attention_sliding_window` Stage 4 — 单槽→depth-2 流水（1.25–1.36×）；`nqb=2` K/V 摊薄与 GQA 头组批处理分别被编译稳定性与 UB 192KB 硬限阻断 | 本 opt_log 全文 | — | 同上 |

---

## Round 18（base = `perf_opt/_gqa_sw_fwd_kernel.py` sha `33f85319…`）：G=2 head-group batching 移植 —— **未采纳（no_gain / compile-blocked）**

**轮次定位**：把 Varlen sibling 已验证的 G=2 head-group batching（`_gqa_sw_fwd_varlen_kernel.py`，sha `6befe384…`）移植到定长 GQA-SW，验证 `heads/heads_kv = 4` 时两个 query head 共享同一 KV head 的 K/V、把每 band step 的 K/V `GM→L1` 按 head 对摊薄。候选文件：`perf_opt/_gqa_sw_fwd_kernel_round18_g2.py`（sha `ec9371de…`，未转正；baseline 保持 `33f85319…`）。

### 18.1 移植改动点（与 Varlen G=2 对齐）

| 层 | 改动 |
|---|---|
| 任务轴 | `head_group = 2 if (groups>=2 and groups%2==0 and heads%2==0 and dim>=128 and not bf16/8b-long) else 1`；`HG=heads//head_group`；`num_logical=num_q_blocks*HG*batch`；`HB=HG*batch` |
| ws | `ws_s/ws_p/ws_o` head 维 `bm → head_group*bm` |
| L1/L0C | `l1_q/l1_a/l0_c` M 维 `bm → head_group*bm` |
| 解码 | `by → byp=(cid//batch)%HG`；`kv_head=(byp*head_group)//groups` |
| Q hoist | `for gh in T.serial(head_group)` 装载 `byp*head_group+gh` → `l1_q[gh*bm ...]` |
| fused gemm | M extent `tmc → fm = tmc if G==1 else bm+tmc` |
| Vector | `acc_o=[half,G*dim]`、`ub_m/ub_ell=[G*half,1]`、`scales=[G*L_max*half,1]`；瞬态用 `[half,1]` 桥 |
| UB 瘦身 | `ub_f16_D=ub_f16_N`、`ub_f32_D=ub_f32_N`（R13 别名，−24,576 B；plan 196,608→189,696 B） |

### 18.2 候选 vs current best 对比表（B2，8b-long fp16，`msprof op` median of 20）

| 候选 | Task Duration(us) | vs current best | aic_mte2 | aic_cube | aic_fixpipe | aic_scalar | aiv_vec | GM→L1(KB) | L0 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| current best `r3_op1_pipe2` (sha `33f85319`) | **1069.92** | — | 0.571 | 0.224 | 0.234 | 0.901 | 0.591 | 63248 | pass | base |
| `r18_g2_headgroup` (sha `ec9371de`) | **965.32** | **1.108×**（−9.8 %） | **0.367** | 0.238 | 0.193 | 0.781 | 0.667 | **38912**（−38.5 %） | pass | 未采纳 |

⇒ 机制与 Varlen R13 一致：收益来自 **GM→L1 带宽**（K/V 按 head 对摊薄），非算术量变化。

### 18.3 A/B（交错，`--launch-count=20 --warm-up=5`，per-arm median）

| workload | base(us) | G=2(us) | speedup | 判定 |
|---|---:|---:|---:|---|
| 8b-long fp16 | 1069.92 | 962.66 | **1.111×** | 过 1.10× |
| 70b-long fp16 | 1067.74 | 964.14 | **1.107×** | 过 1.10× |
| 8b-short fp16 | 352.78 | 316.98 | **1.113×** | 不回归 |

raw profile：`profiles/round18/ab_*`、`profiles/round18/final_*`；结果 `logs/round18/ab18_result.json`、`final8_result.json`。

### 18.4 阻断证据：`bishengir-compile` 确定性 SIGSEGV

- `--level all`（single process）：**L0 4/4 PASS、L1 8/8 PASS、L2 12/12 PASS**；Boundary 第 2 例 `B-s520-w256`（D=64/S=520/wl=256）编译 **err code −11**，40 次重试仍失败（`logs/round18/final_all.log`、`final_all_v2.log`）。
- 官方 report（`logs/round18/official_report.log`）：correctness 通过；benchmark 前 4 例（smoke×2、8b-short fp16/bf16）PASS（8b-short Ratio 13.996 % / 14.355 %，baseline report 为 12.71 % / 12.74 %），第 5 例 `8b-long fp16` G=2 在同一进程内 **SIGSEGV 重试挂起**；54 min 未完成，已中止并删除未完成的 report 目录。
- 对照：G=1 baseline 同 trace 24 s 编译通过；D=128 正式 shape 单独进程 25 s 编译通过。⇒ 属 **多编译进程下的编译器稳定性缺陷**（与 Round 5 的 `bishengir-compile` 假死同族），不是设计层天花板。
- 两个 trace-time guard（`dim>=128`、bf16&8b-long 排除）已用于压掉两处确定性 SIGSEGV，但仍无法让 `B-s520-w256` 与 report 的多 case 进程稳定完成。

### 18.5 裁决与收尾

- **未采纳**：按任务「`--level all` 全绿」与「编译失败 → no_gain/blocked」的约定，`perf_opt/_gqa_sw_fwd_kernel.py` **恢复为 baseline sha `33f85319…`**；`perf_records.jsonl` 恢复为 29 行（r3 final，sha `aae0ba66…`）；未写新 records、未产出官方 report。
- 证据保留：候选 `_gqa_sw_fwd_kernel_round18_g2.py`、A/B/最终测量原始 profile 与日志均在 `perf_opt/` 下。
- **设计层天花板判定**：未触发（1.11× < 2×；阻断归因为工具链编译稳定性，非算法/设计层不可解）⇒ 不产出 `perf_feedback.md`。

### 18.6 Round 18 Value Point Proposals

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A6 | **D**（实测数据） | 定长 GQA-SW 的 G=2 head-group batching 在 910B3 上 8b-long fp16 `1069.92→962.66 µs`（1.111×），`GM_to_L1 63248→38912 KB`（−38.5 %）、`aic_mte2 0.571→0.367`；70b-long 1.107×、8b-short 1.113×、70b-short/bf16 1.12–1.16× | 本 Round 18 §18.2/18.3；`profiles/round18/` | `_gqa_sw_fwd_kernel_round18_g2.py`（对 baseline 的最小 diff） | tilelang 0.1.2+9cfae94c / CANN 9.0.0 / 910B3 |
| VP-A7 | **D**（负向/陷阱） | `for gh in T.serial(head_group)` 形态在定长 kernel 的 D=64/S=520/wl=256 trace 上使 `bishengir-compile` 确定性 SIGSEGV（rc −11，40 retry 不收敛）；同 trace 的原始直线 G=1 体可编译。规则：G=1 fallback 应保持原直线体，不要套 `T.serial(1)` | `logs/round18/final_all*.log`、`probe_Bs520w256.log` | `logs/round18/probe_Bs520w256.py` | 同上 |
| VP-A8 | **R**（流程规则） | 官方 report 在单 pytest 进程内连续编译 10 个 G=2 trace 时，第 5 例即触发 `bishengir-compile` SIGSEGV 重试挂起；G=2 移植的验收应改为「每 case 独立进程 + 单 case wallclock 上限」再汇总，否则会被 40×retry 的假死耗尽预算 | `logs/round18/official_report.log` | — | 同上 |

---

## Round 18b（收尾）：编译器稳定性修复 + 转正 —— **采纳（winner_merged）**

**轮次定位**：R18 的 G=2 移植性能已达标（A/B 1.107–1.113×），但被两处编译器稳定性问题阻断转正。本轮**不新增性能方向**，只做两件事：(1) 把两处确定性编译失败消除；(2) 用进程隔离拿到真实官方 report，然后转正。

### 18b.1 阻断 1：`B-s520-w256` 的形状与修复

- **形状定位**（`run_boundary()` 第 2 例 → `_run_case(...)`）：
  `batch=1, heads=8, heads_kv=2, seq=520, dim=64, is_causal=True, window_size_left=256, window_size_right=-1, dtype=float16`，block `(64, 64, 1)` → 生效 `(64, 256)`。
  `groups = 4`，但 `dim = 64 < 128` → `head_group = 1`（**R18 已有的 `dim>=128` guard 早已把它排除在 G=2 之外**）。⇒ 该例**根本不是 G=2 路径**，R18 里「加 guard 回退到 G=1」对它无效，这也是它没被两处 guard 覆盖的原因。
- **真实根因**：G=1 fallback 体在 R18 里被写成 `for gh in T.serial(head_group)`（即 `T.serial(1)`）。该 IR 形态使 `bishengir-compile` 对这条 trace **确定性 SIGSEGV**（rc −11）。对照实验：同一形状用原始直线体（循环展开）**24 s 编译成功**。
- **修复（不是 guard，是消除触发条件）**：把 5 处 `for gh in T.serial(head_group)` 全部改成 **Python `range(head_group)`**（trace-time 展开）。
  - `head_group == 1` → 退化成**一条直线体**，即 Stage-3 验证过的形态；
  - `head_group >= 2` → 同一条语句序列展开两次；外层 band 循环仍为运行时 `T.serial`。
  - 8 个正式 tune shape 的 `head_group` 判定逻辑**未改**（`dim>=128` 与 `bf16&seq>=2048&heads=32` 两处 guard 保留，属编译期可编译性 guard，不改变 tune 集合）。
- 验证：`logs/round18b/final_s520w256.log` → `TRACE+COMPILE ok 23.8 s`、`OUT max_abs_diff=1.953e-03`、`PASS`（此前 40 retry 不收敛）。

### 18b.2 阻断 1b：`bishengir-compile` 的第三种失败模式——无限循环

`--level all` 全绿后，冷 cache 单进程复现官方 report 时又出现**新的**失败模式：`bishengir-compile` 收到同一份 `.npuir` 后 **100 % CPU 死循环**（实测两个孤儿进程分别卡了 80 min / 53 min），而 stock 调用 `subprocess.run(cmd, capture_output=True, check=True)` **没有 timeout**，R18 的 40×retry 只覆盖 SIGSEGV ⇒ 一次死循环就把整条流水线钉死（官方 report 90 min 无进展即此因）。

修复（写进 `perf_opt/_gqa_sw_fwd_kernel.py` 的既有 shim，最小侵入）：
1. 在 `tilelang.jit.jit_npu` 模块的 `subprocess.run` 上装一个 wrapper：仅当命令里含 `bishengir-compile` 时补 `timeout=_COMPILE_TIMEOUT_S=150`（超时由 `subprocess.run` 负责 kill 子进程，孤儿进程问题一并消除）；
2. 重试判据从「仅 SIGSEGV」扩为「SIGSEGV **或** 超时」，`_COMPILE_RETRIES=16`（p(单次成功)≈0.3 ⇒ 连续 16 次失败 < 0.5 %）。
3. 冷 cache 实测（`TILELANG_CACHE_DIR=/tmp/opencode/coldc1`，8b-long fp16）**86.5 s** 完成编译并跑满 20 launch（median 968.62 us），此前该 trace 会卡死。

**残余风险（如实披露，未在本轮解决）**：shim 的安装是「kernel 模块 import 时 `importlib.import_module("tilelang.jit.jit_npu")` 成功」才生效（失败时被 `except Exception: pass` 静默吞掉）。实测在**同一进程里先通过 `tileops.*` 包路径导入、再二次导入 `tilelang.jit.jit_npu`** 的少数语境下会出现 `ImportError: cannot import name 'jit_npu' from 'jit' (unknown location)`。本轮所有 pytest/bench/`--level all` 进程里 shim 均生效（有重试计数与冷编译 86.5 s 为证），但 msprof 的 `prof_script` 子进程是否生效无法直接观测——这也是 18b.4 第 5 例 600 s 超时的唯一未解释点。若后续再次遇到单 case `msprof` 超时，按「单 case 复测 + `assemble_report.py` 重新汇总」处理，不要重跑整条 report。

### 18b.3 `--level all` 结果（转正后文件，sha `6acdc11c…`）

```
== L0 (blocking) ==            PASS (4 precision cases + contract)
== L1 (blocking) ==            PASS (8 cases)
== L2 (warn-only) ==          PASS (12 cases)
== Boundary (warn-only) ==    PASS (8 cases)
All check passed!              # logs/round18b/level_all_final.log (final file, 13m55s)
                               # logs/round18b/level_all_v2.log    (same content pre-shim, 3m14s)
```
R18 的 `B-s520-w256` 从「40 retry 不收敛」变为 **PASS**；32/32 无 WARN。

### 18b.4 阻断 2：官方 report 的进程隔离与结果

**结论：单进程官方流水线在本工具链上不可用**（实测两处，均为多 trace 同进程的累积状态问题，逐 case 独立进程则全部通过）：

| 失败点 | 现象 | 单进程 | 独立进程 |
|---|---|---|---|
| correctness 第 14 例 `2-256-8-2-64-True-0--1`（wl=0, G=1） | 卡住无输出 88 min（`correctness.log` 停在 07:33） | FAIL（deadlock/infinite-loop） | **PASS**（52 s） |
| benchmark 第 5 例 `8b-long fp16`（G=2） | `msprof op` 600 s 超时（profiling 子进程死锁） | TIMEOUT | **PASS**（128 s，967.10 us） |

**采用的方案**：保留官方 report 的全部代码路径（同一 `parse_junit_report` / `analyze_run` / `write_reports` / `setup_info`），只把 **pytest 粒度改成「每个 collected test id 一个进程」**，再把逐 case 的 JUnit XML 与 `BenchmarkReport` JSON 合并成 `run_operator` 同名布局，最后调用同一个 `analyze_run` + `write_reports` 生成 `run.json` / `report.md`。脚本：`logs/round18b/report_isolated.py`（+ `assemble_report.py`，用于单 case 复测后重新汇总）。**没有任何数据被合成**：每条 latency 都来自该 case 自己进程里 `msprof op` 的真实输出。

- **run_id**：`20261106_105557_262003_GroupedQueryAttentionSlidingWindowFwdOp`
- **run.json**：`examples/TileOPs/reports/tileops/<run_id>/run.json` → `status=passed`
- **correctness**：22/22 PASS（含 smoke、GQA ratio、非 2 次幂序列、wl=0、right-window、contract/validation 用例）
- **benchmark**：10/10（含 2 个 smoke 参数化用例；manifest 未给 smoke 打 marker，官方 `-m "not smoke"` 实际过滤不到，与 R18 观察一致）
- 第 5 例（`8b-long fp16`）首次跑 `msprof` 600 s 超时后被 kill；**单 case 复测一次即通过**（`967.10 us / Ratio 22.74 %`），随后用 `assemble_report.py` 重新汇总。超时发生在 msprof 的 `prof_script` 子进程里（`warm-up=0 + 1 次 measured launch`，冷编译与首启都在该子进程内完成），**无法从 msprof 的捕获日志区分是 prof_script 内的冷编译卡住还是设备侧 launch 死锁**——两种可能都在单 case 复测中消失（工具链抖动，非确定性）。

**8 个 tune 用例 Ratio（官方 report，msprof + Roofline）**

| workload | latency(us) | Ratio(%) |
|---|---:|---:|
| llama-3.1-8b-short-w256 fp16 | 315.22 | 14.06 |
| llama-3.1-8b-short-w256 bf16 | 310.32 | 14.27 |
| llama-3.1-8b-long-w1024 fp16 | 967.10 | 22.74 |
| llama-3.1-8b-long-w1024 bf16（G=1 guard） | 1032.34 | 21.28 |
| llama-3.1-70b-short-w256 fp16 | 312.50 | 14.18 |
| llama-3.1-70b-short-w256 bf16 | 299.20 | 14.80 |
| llama-3.1-70b-long-w1024 fp16 | 971.24 | 22.64 |
| llama-3.1-70b-long-w1024 bf16 | 909.18 | 24.16 |

**8 用例 Ratio 均值 = 18.52 %**（smoke 两例另计：88.08 / 87.38 us，Ratio 3.20 / 3.22 %）。R18 记录的 baseline report 同两例为 12.71 / 12.74 %（8b-short fp16/bf16），本轮 14.06 / 14.27 %，方向一致。

### 18b.5 候选 vs current best 对比表（B2，`msprof op` median of 20）

| 候选 | 8b-long fp16(us) | vs current best | 8b-short fp16(us) | 70b-long fp16(us) | L0/L1 | `--level all` | 裁决 |
|---|---:|---:|---:|---:|---|---|---|
| current best = R18 baseline `r3_op1_pipe2`（sha `33f85319`） | 1069.92 | — | 352.78 | 1067.74 | pass | pass | base |
| R18 候选 `r18_g2_headgroup`（sha `ec9371de`） | 962.66 | 1.111× | 316.98 | 964.14 | pass | **Boundary 失败**（`B-s520-w256` rc −11） | R18 未采纳 |
| **R18b 候选 `r18b_g2_headgroup`（sha `6acdc11c`）** | **962.62** | **1.111×** | **317.84** | **965.50** | pass | **pass 32/32** | **采纳** |

⇒ 稳定性修复**未改变** G=2 的性能（962.66 → 962.62 us，差值 0.004 %，远低于 3 % 噪声阈值）：`range()` 展开与 `T.serial(2)` 在设备端生成同一 schedule。

### 18b.6 转正与工件清单

```text
备份：perf_opt/history/_gqa_sw_fwd_kernel_baseline_33f85319_precise.py   (sha 33f85319…)
     （history/_gqa_sw_fwd_kernel_r3_op1_pipe2_33f85319.py 为既有同 sha 副本）
转正：perf_opt/_gqa_sw_fwd_kernel.py  ← 逐字节拷贝 _gqa_sw_fwd_kernel_round18b.py
      sha256 = 6acdc11cd5a78b8da77ea65ef6fdfbca44a545757b912054c4279af75f752d71
```

| 变更 | 说明 |
|---|---|
| `perf_opt/_gqa_sw_fwd_kernel.py` | 转正（G=2 + `range()` 展开 + 编译超时/重试 shim）；wrapper 的双 import 切换块未改（其 policy 本就是「perf_opt 存在即启用」） |
| `perf_opt/_gqa_sw_fwd_kernel_round18b.py` | 本轮候选（与转正文件逐字节相同） |
| `perf_opt/_gqa_sw_fwd_kernel_round18_g2.py` | R18 候选，保留为证据（未改） |
| `perf_opt/history/_gqa_sw_fwd_kernel_baseline_33f85319_precise.py` | 转正前 baseline 备份 |
| `perf_opt/workload_inventory.json` | **新建**：10 workload（8 tune + 2 smoke），`kernel_id` 沿用 `perf_records.jsonl` 既有记录的同一 kernel_id（`_gqa_sw_fwd_kernel.py::_gqa_sw_fwd_kernel::_gqa_sw_fwd_main`），tune 项 `tuning_status=winner_merged` |
| `perf_opt/perf_records.jsonl` | 37 行：追加 8 条 `phase=final`（round 18、candidate `final_r18b_g2_headgroup`、artifact sha `6acdc11c…`）；round-9 的 8 条旧 `final` 重归类为 `phase=merged`（逐字节副本见 `history/perf_records_round9_final_rows.jsonl`） |
| `perf_opt/logs/round18b/` | 本轮全部日志、probe、report 驱动脚本 |
| `perf_opt/profiles/round18b/` | 8 个 tune workload 的 final raw profile + 冷 cache 编译验证 |
| `reports/tileops/<run_id>/` | 官方 report（run.json / report.md / pytest 逐 case XML+JSON / msprof artifacts） |

### 18b.7 设计层天花板判定

**未触发，不产出 `perf_feedback.md`**：两项门槛均不满足——① 归因是**工具链编译稳定性**（`bishengir-compile` 的 SIGSEGV/死循环）与 trace 形态（`T.serial(1)`），不是 DESIGN 的算法/内存层级假设；② 本轮收益 1.24–1.64×（相对 round-0 baseline），相对 current best 为 1.111×，**远小于 2×**，未与任何 DESIGN 假设矛盾。

### 18b.8 Round 18b Value Point Proposals

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-A9 | **D**（负向/陷阱，已确认） | `T.serial(1)`（以及「用 `T.serial(head_group)` 写 G=1 fallback」）会让 `bishengir-compile` 对特定 trace（D=64/S=520/wl=256）**确定性 SIGSEGV**；改为 Python `range(head_group)` trace-time 展开即恢复，且 G=2 性能零损失（962.66→962.62 us）。规则：**G=1 fallback 必须是直线体** | `logs/round18b/p1_s520w256.log`（修复后 23.8 s PASS）、`logs/round18/probe_Bs520w256.log`（修复前 rc −11） | `_gqa_sw_fwd_kernel_round18b.py` 与 `_round18_g2.py` 的 5 处单行 diff | tilelang 0.1.2+9cfae94c / CANN 9.0.0 / 910B3 |
| VP-A10 | **P**（模式方法） | `bishengir-compile` 1.1.0 的失败是**三态**：rc −11 / 无限循环（100 % CPU）/ 成功（≈30 %）。因此重试必须**带 wallclock 上限**（`subprocess.run(timeout=)` 会 kill 子进程，顺带清理孤儿）；只重试 SIGSEGV 会被死循环钉死。建议所有 npuir kernel 的编译 shim 默认 150 s×16 | `logs/round18b/cfg_cold_8blong`（冷 cache 86.5 s 通过）；两个 80/53 min 的孤儿 `bishengir-compile` 进程 | 同 VP-A9 的 shim 段 | 同上 |
| VP-A11 | **R**（流程规则） | 本工具链上**官方 report 必须逐 case 独立进程**跑：单进程下 correctness 第 14 例（wl=0）与 benchmark 第 5 例（8b-long fp16）分别死锁/`msprof` 600 s 超时，逐 case 则 22/22 + 10/10 通过。汇总时复用官方 collector/analyzer/write_reports，只替换 pytest 粒度，可产出真 `run.json` | `reports/tileops/20261106_105557_262003_…/run.json`；`logs/round18b/official_report_isolated_v2.log` | `logs/round18b/report_isolated.py`、`assemble_report.py` | 同上 |
| VP-A12 | **C**（案例索引） | 案例：`grouped_query_attention_sliding_window` Stage 4 收尾——G=2 head-group batching（1.11×）在「`T.serial(1)` 触发 SIGSEGV」与「`bishengir-compile` 无限循环」两个工具链缺陷下完成转正；官方 report Ratio 8 用例均值 18.52 % | 本 Round 18b 全节 | — | 同上 |

### 18b.9 门禁校验（gate 4）

```text
$ python3 .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 4, "warnings": []}
exit=0
```

- `S4-WORKLOAD-INVENTORY`：`workload_inventory.json` 10 项（8 tune + 2 smoke），分类/类型/`tuning_status`/`precision_pass` 全部就位。
- `S4-PERF-RECORDS-{SCHEMA,COVERAGE,RECON}`：37 行全部合法；8 个 tune workload 各有 baseline + 恰一条 `final`（`final_r18b_g2_headgroup`），`artifact_sha256 = 6acdc11c…` 与磁盘文件一致。
- `S4-OPTLOG-COMPTABLE` / `S4-PERF-RECORDS-RECON`（Final 表对账）：`Final Performance Test Data` 的 8 行 baseline/final 与记录逐条一致（差值 0）。
- `S4-PERF-FEEDBACK-*`：不适用（未产出 `perf_feedback.md`，设计层天花板未触发）。

---

## Round 19（Tier-0 探针）：关闭自动同步注入 / cross_interval 批量同步

**任务**：验证 `NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC=True`（关闭编译器自动核内同步注入）与 cross_interval 批量跨核同步能否改善定长 GQA-SW 的 CV 重叠。
**预算**：1 次 dispatch、60–90 min、无全量回归 / 官方 report / `perf_records.jsonl` / 转正。**current best 未变**（`_gqa_sw_fwd_kernel.py`，sha `6acdc11c…`）。

### 19.1 口径与当日 baseline（`msprof op`，launch=20 / warm-up=5，median of 20）

| workload | 当日 baseline (us) | min/max | R18b 记录 (us) | 偏差 |
|---|---:|---|---:|---:|
| 8b-long fp16 | **966.52** | 956.60 / 971.86 | 962.62 | +0.4 % |
| 70b-long fp16 | **965.14** | 958.14 / 975.52 | 965.50 | −0.04 % |

环境/机器状态与 R18b 一致（差值在噪声内），baseline 可用作本轮唯一比较基准。

### 19.2 候选 A：关闭自动同步注入（`NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC=True`）

- key 在当前工具链**可用**（`tilelang/transform/pass_config.py:42` → `npuir.disable_hivm_auto_inject_sync`；`tilelang/jit/jit_npu.py:1897` 消费）。
- **编译命令实测**（`logs/round19/probeA_compile.log`，冷 cache 抓取）：
  `…… --disable-hivm-auto-inject-sync=true ……`（baseline 同命令为 `=false`，见 `logs/round19/B_level_L0.log` 的 3 条重试记录）。**编译成功**（51.0 s）。
- **L0：FAIL（阻塞）**，且在**第 1 个 case**（L0-1 smoke fp16）就死在设备侧，不是精度差：

```text
== L0 (blocking) ==
NPU function error: ... AclrtSynchronizeStreamWithTimeout, error code is 507015
[Error]: The aicore execution is abnormal.
errcode:(0x20000, 0, 0x10000) errorStr: L0C read/write conflict
  (vec read operation or cube write operation). When the FIXP reads l0c,
  the read and write operations occur at the same address. ...
  fixp_error0 info: 0x168050, fixp_error1 info: 0x9
```

**归因**：本 kernel 的 `T.sync_block_set/wait` 只覆盖 **Cube↔Vector 跨引擎**握手；同核内 `PIPE_C`(gemm 写 L0C) → `PIPE_FIX`(fixpipe 读 L0C) 的 WAR、`MTE2→V`、`V→MTE3` 等**同核内跨 pipe 依赖**完全依赖 `--disable-hivm-auto-inject-sync=false` 的自动注入。关掉它后第一条 trace 就在 L0C 上发生读写冲突。

⇒ **A 的结论是「关了会错」，不是「自动同步不是瓶颈」**：自动注入在本 kernel 里是**功能必需**（load-bearing），不是冗余 barrier。要吃到这条路必须补齐整套手写核内 pipe 同步（本轮预算内不做）。

### 19.3 候选 B：cross_interval 批量跨核同步（ci=2 / ci=4）

实现（`_gqa_sw_fwd_kernel_round19_B_ci2.py` / `_ci4.py`，各 5 处 band 循环，7 个 sync 点）：
- 生产者（Cube pass-1、Cube pass-2）只在**组末步** set：`if l % CI == CI-1: set(l)`，`else: if l == L-1: set(l)`；
- 消费者（Vector pass-1/pass-2、Cube pass-2）只在**组首步** wait：`if l % CI == 0: wait(min(l + CI-1, L-1) + slot*L_max)`；
- flag id 仍是 `组末步 + slot*L_max`（ci2 用 3 个 id/槽、ci4 用 2 个 id/槽，均 ≪ 16 预算）；workspace 布局不变（band 内 `l*bn_eff` 已逐块分离），因此**不需要加 slot/UB**。

| 候选 | L0 | 8b-long fp16 (us) | vs base | 70b-long fp16 (us) | vs base | 裁决 |
|---|---|---:|---:|---:|---:|---|
| baseline（sha `6acdc11c…`） | PASS | 966.52 | — | 965.14 | — | base |
| **B ci=2** | **PASS**（4 case + contract，max_diff 与 baseline 逐位相同） | 982.26 | **−1.6 %（慢）** | 983.28 | **−1.9 %（慢）** | no_gain |
| **B ci=4** | **PASS**（同上） | 1085.72 | **−12.3 %（慢）** | 1090.96 | **−13.0 %（慢）** | no_gain（明确回退） |

### 19.4 候选 vs current best 对比表（B2）

| 候选 | 8b-long fp16 (us) | 70b-long fp16 (us) | AICore 利用率 / memory 指标 | L0 | 编译命令 auto-inject | 裁决 |
|---|---:|---:|---|---|---|---|
| current best（R18b） | 966.52 | 965.14 | n/a（本机 `msprof op` 只产出 `OpBasicInfo`：Task Duration/Block Dim，`aic_*/aiv_*` 列为空） | PASS | `=false` | base |
| A：`DISABLE_HIVM_AUTO_INJECT_SYNC=True` | —（未测） | — | n/a | **FAIL**（L0C WAR，设备异常 507015） | `=true`（实测） | blocked |
| B：ci=2 | 982.26 | 983.28 | n/a（同上） | PASS | `=false` | no_gain |
| B：ci=4 | 1085.72 | 1090.96 | n/a（同上） | PASS | `=false` | no_gain |

> 说明：本轮 `perf_opt/bench.py` 的 `--aic-metrics=Default` 在此 msprof 版本下不落 pipe ratio 列（全部 `null`），因此 AICore 利用率/memory 指标不可得；裁决只用 `Task Duration`（与 Stage 4 唯一口径一致）。

### 19.5 结论

1. **A（关闭自动同步注入）：blocked / 「关了会错」**。关闭后功能上不成立（L0C WAR 立刻触发设备 AICore exception），说明自动注入在 Expert 双 Scope 结构里承担的是核内跨 pipe 顺序保证，而不是可省的冗余同步。
2. **B（cross_interval 批量同步）：family_no_gain，且随间隔单调变差**（ci2 −1.6 %/−1.9 %，ci4 −12.3 %/−13.0 %）。
   机制：本 kernel 的 Cube 已把整条 S band 提前产好（prologue/advance），真正的 CV 重叠来自 **逐 band step 的细粒度 `P`/`O` 握手**；把 2–4 步合成一次握手等于让消费者多等 1–3 个 band step 的 P/V 计算，重叠粒度直接变粗。因此 **「跨核同步次数」不是瓶颈**，相反细粒度 flag 是重叠的来源。
3. **保留 baseline**：`perf_opt/_gqa_sw_fwd_kernel.py`（sha `6acdc11c…`）不动；本轮不写 `perf_records.jsonl`、不改 wrapper/benchmark/manifest/tests、不产出 `perf_feedback.md`（设计层天花板未触发：A/B 都不涉及 DESIGN 的算法/内存层级假设，收益也不大于 2×）。
4. 两条剩余可选方向（本轮不做，供后续 Tier 级判断）：① 若要走 A，需要手写全套核内 pipe 同步（PIPE_FIX→PIPE_C、MTE2→V、V→MTE3、V→MTE2…），成本高、收益未知；② 本 kernel 的 `n_slots=2` 受 16 个 flag id 预算限制，加深跨任务流水（n_slots=3）需要 flag id 预算突破，属结构级改动。

### 19.6 工具链发现（本轮实证，已写入 pattern-library 建议项）

| id | 内容 | 证据 |
|---|---|---|
| VP-A13 (D/P) | **npuir 前端陷阱**：traced kernel 里用 `if (bool_expr_a) \| (bool_expr_b):`（PrimExpr 上按位或组合布尔条件）会让前端**进程内 SIGSEGV**（exit 139，~13 s 处，无 Python traceback）；改写成 `if a: … else: if b: …` 或 `T.if_then_else(...)==1` 均正常编译。注意：`if <单个 PrimExpr 比较>:` 本身完全正常（含包住 `T.sync_block_set/wait`） | `logs/round19/_T2_if_set_or_probe.log`（crash）vs `_T1/_T3/_T5/_T6_*.log`（均 `compile ok`）；`_B1_wait_uncond_bisect.py` 复现 |
| VP-A14 (D) | `msprof op --aic-metrics=Default` 在 CANN 9.0.0（本机）只产出 `OpBasicInfo`（Task Duration / Block Dim / Freq），**无 `aic_mte2/cube/fixpipe/scalar` 与 `aiv_*` 列**；R18 opt_log 中的 ratio 列不能由当前口径复现 | `profiles/round18b/final_final_8b-long_float16/**/OpBasicInfo_*.csv` 表头；`logs/round19/*_r19.json` 全 `null` |

### 19.7 本轮新增文件

```text
perf_opt/_gqa_sw_fwd_kernel_round19_A_nosync.py     候选 A（baseline + DISABLE_HIVM_AUTO_INJECT_SYNC=True）
perf_opt/_gqa_sw_fwd_kernel_round19_B_ci2.py        候选 B（cross_interval=2，5 处 band 循环 / 7 个 sync 点）
perf_opt/_gqa_sw_fwd_kernel_round19_B_ci4.py        候选 B（cross_interval=4）
perf_opt/logs/round19/run_r19.py                    本轮 bench 驱动（复用 bench.py 的 msprof 口径）
perf_opt/logs/round19/_probe_compile_cmd.py          冷 cache 抓取真实 bishengir-compile 命令行
perf_opt/logs/round19/_probe_trace_only.py           单 trace 探针（区分前端 crash vs 编译 crash）
perf_opt/logs/round19/_T1/_T3/_T5/_T6_*.py           前端条件写法探针（编译通过）
perf_opt/logs/round19/_T2_if_set_or.py               `|` 组合布尔条件 crash 复现
perf_opt/logs/round19/_B1_wait_uncond_bisect.py      二分探针（确认 set 侧条件为触发点）
perf_opt/logs/round19/{A,B_ci4}_level_L0.log         L0 日志（A FAIL / B PASS）
perf_opt/logs/round19/*_r19.json                     本轮全部 Task Duration 原始结果
perf_opt/logs/round19/{base,Bci2,Bci4}_*.log         每 case 的 msprof 原始 stdout/stderr
perf_opt/profiles/round19/                            6 个 raw `msprof op` 目录（base/B ci2/B ci4 × 2 shape）
```

未改动：`perf_opt/_gqa_sw_fwd_kernel.py`（sha 不变）、`perf_opt/perf_records.jsonl`、`perf_opt/workload_inventory.json`、wrapper、benchmark、manifest、tests、`task_operators.md`、`DESIGN.md`。


---

## Round 20（测量口径修复 + short 档诊断）

**任务（Tier-0 诊断轮，不重开探索）**：① 修 `perf_opt/bench.py` 的 `msprof op` 采集口径并扩展解析；② 在当前 best
（`_gqa_sw_fwd_kernel.py`，sha `6acdc11c…`，**未改 kernel**）上复采 3 个 workload；③ 用 pipe 证据定位 short 档
低 Ratio 的固定开销；④ 只有证据清晰且可低成本实现才做 1 个候选。
**预算**：`max_stage4_experiments=2`（本轮用 0）、`max_subagent_dispatches=2`、`max_wallclock_s=5400`。
**current best 未变**（sha `6acdc11c…`），本轮不写 `perf_records.jsonl`（无 valid candidate），Final Summary 保持 R18b。

### 20.1 口径修复（`perf_opt/bench.py`）

**R19 §19.4 / VP-A14 的「本机只产出 OpBasicInfo」是解析 bug，不是工具链限制**（本条推翻 VP-A14，见 20.6）：

| 事实 | 证据 |
|---|---|
| CANN 9.0.0 / msprof 25.5.0 的 `--aic-metrics=Default` **确实落盘** `PipeUtilization_*.csv`（60 行 = 20 block × 3 sub_block：`cube0`/`vector0`/`vector1`）以及 `ArithmeticUtilization_*` / `Memory_*` / `MemoryL0_*` / `MemoryUB_*` / `ResourceConflictRatio_*` | `profiles/round19/base_r19_8b-long_float16/OPPROF_*/_gqa_sw_fwd_main_mix_aic/6/`（R19 自己产出的目录里就有 `aiv_vec_ratio` 等列） |
| 旧 `_parse_op_basic()` 只 walk `OpBasicInfo*.csv`，而 `aic_*/aiv_*` 只存在于 `PipeUtilization*.csv` ⇒ `g("aic_mte2_ratio")` 恒为 `None` | `logs/round20_diag/bench_py_pre_round20.py.bak` L128-141 |

改动（`diff -u` 全文 `logs/round20_diag/bench_py_round20.diff`，+214/−48 行）：

1. `--aic-metrics=Default` → 显式列表 `BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline`
   （与同机同 CANN 上 GQA-**bwd** Stage 4 已实证可用的列表逐字一致；`cfg.metrics` 可覆盖）。
2. 新增 `_table_rows/_group_median/_summarize`：按 `sub_block_id` 分组、对 20 个 core 取 **median**，输出
   `pipe` / `arith` / `mem` 三张嵌套表 + 扁平键（`aic_*`←`cube0`，`aiv_*`←`vector0`，并保留 `aiv_vec_ratio_v1` 做
   `vector1` 一致性核对）；新增 `aic_flag_wait_sum(us)` / `aiv_flag_wait_sum(us)` 与 `*_flag_wait_by_id`（id0..id9）、
   GM 流量键（`gm_to_l1(KB)` / `gm_to_ub(KB)` / `l0c_to_gm(KB)` / `ub_to_gm(KB)` / `read_main_mem(KB)` + bw rate）。
3. 新增 `bench.py parse --cfg` 子命令：可对**既有** raw 目录重解析（不重新 profiling），本轮用它复算了 R19 的
   `profiles/round19/base_r19_*` 两条（得到与 R19 `Task Duration` 一致的 965.39/965.13 us）。
4. 新增 `probe-mid` case（仅诊断探针，**不是** manifest workload，不写入 inventory / records）。

**两个口径陷阱（本轮实测，写进 20.6）**：
- `aic_scalar_wait_time(us)` / `aiv_scalar_wait_time(us)` 的**聚合列不可直接当等待时间用**：cube0 的
  `aic_scalar_wait_time` 恒为 `0.000000`，真实值只在 `aic_scalar_wait_id0..9_time(us)` 里（逐 id 求和 = 744.3 us，
  与 `aic_scalar_time` 746.1 基本相等 ⇒ cube 的 scalar 时间几乎全是 flag 等待）；
- `aiv_scalar_wait_time`≈`aiv_vec_time`（短档 159.21 vs 159.35，长档 652.04 vs 637.71）⇒ 它是 **AIV scalar 等向量
  流水**（= 向量忙），**不是空闲/气泡**，不能当 idle 计。

### 20.2 复采（current best，未改 kernel；`msprof op`，launch=20 / warm-up=5，median of 20）

| workload (fp16) | 本轮 Task Duration(us) | min/max | R18b 记录(us) | 偏差 | 口径结论 |
|---|---:|---|---:|---:|---|
| 8b-short | **317.33** | 313.55 / 319.67 | 317.84 | −0.16 % | 一致 |
| 8b-long | **963.26** | 955.84 / 969.92 | 962.62 | +0.07 % | 一致 |
| 70b-short | **310.40** | 307.33 / 313.93 | 312.50 | −0.67 % | 一致 |

⇒ **显式 metrics 列表不改变时延口径**（差值 ≤0.7 % « 3 % 噪声），pipe 数据与 R19 `Default` 目录重解析逐项一致
（`aic_cube_ratio` 0.2361 vs 0.2367 等）。raw：`profiles/round20_diag/base_r20_<case>_float16/`；日志/JSON：
`logs/round20_diag/base_r20*.log|.json`。

结构账（trace-time 公式复算，`logs/round20_diag/model_fit.json`）：`bm=64, bn_eff=256, G=2 (head_group=2)`，三者均
`n_slots=2`：

| workload | tasks | band-steps | L_max | 平均 L | 每 core tasks | 每 core band-steps |
|---|---:|---:|---:|---:|---:|---:|
| 8b-short | 512 | 768 | 2 | **1.50** | 25.6 | 38.4 |
| 70b-short | 512 | 768 | 2 | **1.50** | 25.6 | 38.4 |
| 8b-long | 1024 | 3840 | 5 | **3.75** | 51.2 | 192.0 |

### 20.3 诊断表（每 core 中位数；us 为 20 次 launch 的每 core 量）

| 指标 | 8b-short | 8b-long | 70b-short |
|---|---:|---:|---:|
| **Task Duration(us)** | **317.33** | **963.26** | **310.40** |
| 墙钟/任务 · 墙钟/band-step | 12.40 · **8.26** | 18.81 · **5.02** | 12.13 · 8.08 |
| `aiv_scalar_wait_time`（µs / 占墙钟）※=向量忙 | 159.21 / **50.2 %** | 652.04 / **67.7 %** | 153.03 / 49.3 % |
| `aic_cube_ratio` | 0.1454 | 0.2367 | 0.1477 |
| `aic_mte2_ratio` | 0.2530 | 0.3584 | 0.2559 |
| `aic_mte1_ratio` | 0.1118 | 0.1813 | 0.1139 |
| `aic_fixpipe_ratio` | 0.1120 | 0.1904 | 0.1152 |
| `aiv_vec_ratio` | 0.5011 | 0.6626 | 0.4909 |
| `aiv_mte2_ratio` / `aiv_mte3_ratio` | 0.0911 / 0.0831 | 0.1552 / 0.1048 | 0.0941 / 0.0845 |
| `aic` flag 等待和（µs / 占墙钟） | **289.04 / 91.1 %** | 731.57 / 75.9 % | **282.92 / 91.2 %** |
| `aiv` flag 等待和（µs / 占墙钟） | 84.58 / 26.6 % | 135.11 / 14.0 % | 84.71 / 27.3 % |
| GM→L1（KB/core/launch） | 8320 | 38496 | 8320 |
| GM→UB（KB） | 1872 | 9216 | 1872 |
| L0C→GM（KB） | 3744 | 18432 | 3744 |
| UB→GM（KB） | 1670.5 | 6972.75 | 1670.5 |
| GM→L1 bw usage | 9.43 % | 14.45 % | 9.60 % |
| **每 band-step**：cube / mte1 / mte2 / fixpipe | 1.204 / 0.925 / 2.082 / 0.924 | 1.186 / 0.908 / 1.796 / 0.954 | 1.204 / 0.925 / 2.066 / 0.931 |
| **每 band-step**：aiv vec / mte2 / mte3 | 4.150 / 0.751 / 0.686 | 3.321 / 0.777 / 0.525 | 4.001 / 0.760 / 0.681 |
| GM→L1 / band-step（KB） | 216.7 | 200.5 | 216.7 |
| 每 core 时长极差（max/min，cube0 `aic_time`） | 1.068 | 1.034 | — |

**flag-id 分解**（id = `l + slot*L_max`；short `L_max=2` ⇒ id0/1=slot0 的 l0/l1、id2/3=slot1；long `L_max=5` ⇒ id0-4 / id5-9）：

| workload | cube `aic_scalar_wait_id*`(us) | vector `aiv_scalar_wait_id*`(us) |
|---|---|---|
| 8b-short | id0 **136.82**, id1 24.45, id2 **99.71**, id3 28.06 | id0 23.05, id1 19.89, id2 19.42, id3 22.22 |
| 8b-long | id0 **148.71**, id1 73.87, id2 61.85, id3 52.31, id4 53.08, id5 **103.03**, id6 71.55, id7 61.69, id8 52.87, id9 52.62 | id0 11.14, id1 10.61, id2 10.13, id3 8.19, id4 **28.38**, id5 10.33, id6 10.21, id7 10.27, id8 8.61, id9 **27.25** |

判读：cube 侧 **l=0 的等待是 l≥1 的 1.4–5.6 倍**（short：136.8/99.7 vs 24.5/28.1 —— l0 是 l1 的 4–5.6 倍；
long：148.7/103.0 vs 52–74 —— 2.0–2.8 倍）⇒ 等待集中在**任务首个 band-step**（任务头气泡），不是均匀分布。
vector 侧 short 四个 id 几乎平坦（19–23 us/任务窗），long 只有 **band 尾部 l=4** 突增（28.4/27.2 us）⇒ 任务**排空**项。
注意：S 交接与 O 交接**复用同一个 flag id**（`l + slot*L_max` 在 pass-1 前、pass-1 末、pass-2 前各用一次），因此单 id
等待无法再拆分 S-wait / O-wait —— 这是本 kernel 的 flag 预算（16 id）造成的测量上限。

### 20.4 根因判断：short 档低 Ratio = 每任务固定开销 ÷ 更短的 band（(a)，非 (b)(c)(d)）

**定量模型**（3 个 shape 的最小二乘两点拟合，`logs/round20_diag/model_fit.json`）：

```text
墙钟(每 core) = f·N_task + u·N_bandstep        f = 8.117 us/任务,  u = 2.852 us/band-step
  8b-short : 25.6·8.117 + 38.4·2.852 = 317.3 us   （实测 317.33）
  8b-long  : 51.2·8.117 + 192.0·2.852 = 963.1 us  （实测 963.26）
  70b-short（独立校验点，H 不同/HG=32）：25.6·8.117+38.4·2.852 = 317.3 vs 实测 310.40（−2.2 %）
```

**每 band-step 的真实工作量在 short/long 之间完全相同**（cube 1.204 vs 1.186、mte1 0.925 vs 0.908、fixpipe 0.924 vs
0.954，差 ≤2 %；mte2 +16 % 仅来自每任务 Q hoist 32 KB 摊到更少的步数）。把每 band-step 拆成「任务开销 / L + 真实工作」：

| | 每 band-step 墙钟 | = 任务开销分摊 | + 真实 band-step 工作 |
|---|---:|---:|---:|
| 8b-short (L=1.5) | 8.26 us | **5.41 us** | 2.85 us |
| 8b-long (L=3.75) | 5.02 us | **2.16 us** | 2.85 us |

⇒ short 比 long 慢 1.65× **100 % 由「f ÷ 平均 band 长度」解释**（5.41 vs 2.16），与官方 Ratio 14.06 % vs 22.74 %
（1.62×）一一对应。short 的任务里 **65 % 的墙钟是每任务固定项**（8.12 / 12.40），long 只有 43 %。

**四选项逐条裁决（每条附 pipe 证据）**：

| 假设 | 裁决 | 证据 |
|---|---|---|
| (a) 任务级填充/排空 | **成立（主因）** | cube flag 等待占墙钟 91 %（short）/76 %（long），且 **l=0 的等待是 l≥1 的 4–5.6 倍**（20.3 flag 表）；每 core 时长极差仅 6.8 % ⇒ 不是负载不均 |
| (b) Q-hoist / epilogue 固定成本 | **成立但只占小头，≤2.4 us/任务** | 去掉每步项后的向量侧固定项拟合约 2.4 us/任务（`vec+mte2+mte3` 两点拟合）；Q hoist 的 MTE2 流量仅 32 KB/任务 = GM→L1 的 10 %（819/8320 KB），且 mte2 有 75 % 余量、`aic_mte2_ratio` 仅 0.25 ⇒ 不是 MTE2 饱和 |
| (c) G=2 fused gemm 在短 band 下利用率低 | **否决** | 每 band-step 的 `aic_cube_time` short 1.204 us vs long 1.186 us（**+1.5 %，噪声内**）；`aic_cube_ratio` 从 0.147→0.237 的变化完全由「每任务 band 更短」解释，与 gemm 形状无关 |
| (d) 测量口径问题 | **否决（但口径确有 2 个陷阱已修）** | 显式 metrics 复采与 R18b 记录差 ≤0.7 %；pipe 数据自洽（cube 逐 id 等待和 744 us ≈ `aic_scalar_time` 746 us）；`aiv_scalar_wait_time`≈`aiv_vec_time` 说明它是「等向量」而非空闲 |

**机制**（把 (a) 说到可操作的粒度）：墙钟由**任务边界的一次串行往返**决定 —— cube 在 pass-2 的 `l=0` 上等 vector 的
`P(0)`（short 11.3 us/任务，long 14.3 us/任务但被更长的 band 摊薄），vector 在同一边界等 cube 的 `S(0)`（short
3.30 us/任务 vs long 2.64）。depth-2 流水的 `advance` 只有 **1 个任务**的前瞻：short 的 advance 只有 1–2 个 band-step
（≈ 4–9 us），不足以填掉对端一个任务的串行链；long 的 advance 有 3.75 步，覆盖更好 —— 这正好对应 vector 每步
S-wait short 2.20 us vs long 0.70 us。

**没能定到的部分（如实标注）**：f = 8.12 us/任务 中，≈2.4 us 是向量侧不可约的 init/epilogue 工作量，剩余 ≈5.7 us 是
依赖气泡；二者的**精确**分界需要一个 `n_slots=3` 的实测（它本身就是唯一的候选）。计划中的第三方校验点
`probe-mid`（S=1024/wl=512，avg L=2.25）**编译阻塞**：`bishengir-compile` 进入重试循环 >900 s（观测到 msprof 树
16:58 龄 + 单个 `bishengir-compile` 子进程 1:42 龄），超时 kill 后仍留 4 个孤儿进程（与 VP-A10 同源），已手工清理并
确认设备空闲。⇒ 模型目前是「2 点拟合 + 1 个独立形状校验（70b-short，−2.2 %）」。

### 20.5 候选：**无（diagnosis_only）**，current best 不变

| 候选 | 主要优化点 | 编译 | L0 | Task Duration | 裁决 |
|---|---|---|---|---|---|
| — | 本轮不派生候选（见下 3 条理由） | — | — | 8b-short 317.33 / 8b-long 963.26 / 70b-short 310.40（= current best，R18b sha `6acdc11c…`） | **diagnosis_only** |

不派生候选的理由（不是"没想法"，是"证据/预算不支持本轮盲目实现"）：

1. **唯一能攻 f 的杠杆是结构级的**：`n_slots 2→3`（cube 提前 2 个任务产出 S），flag 预算够（short `3·L_max=6`、
   long `3·L_max=15 ≤ 16`，`FLAG_TASKDONE` 在 `use_pipe2` 下本来就未使用），但它要**复制一整个 prologue trace 块**
   （≈90 行）+ 改 steady-state 的 `adv_task=t+2`。收益上界 = 向量侧可及的 ≈5.7 us/任务 ÷ 12.4 us/任务 ≈ 24 %，
   但每步 CV ping-pong（两引擎各自 ~2 us/step 互等）**不会**被跨任务深度消掉，现实预期 −5~12 %（short）、
   long 上 0~5 %（long 的每步 cube 链 ≈5 us × 3.75 = 18.8 us/任务 ≈ 实测，已贴近该结构族地板）。
2. **同类"体量翻倍"改动在本机有确定性前科**：R5 的 `nqb=2`（doubled prim_func body）在 `bishengir-compile` 上
   **不收敛**（600 s 无输出 / 2400 s 超时）；本轮 `probe-mid` 又在**未改 kernel** 的新 shape 上复现了同一失败类
   （编译重试 >900 s）。在本轮剩余 wallclock 内做「结构改动 + L0/L1 + 3 workload × 3-run 交错 A/B」风险过高，
   且一旦编译不收敛会挤掉本轮唯一的交付（诊断）。
3. **短档"更小 bm/bn"在模型上被否决**：`bm=128` 不可行（G=2 下 `l0_c=[256,256]` f32 = 256 KB > L0C 128 KB）；
   `bn 256→128` 使 band-step 数从 768→1152（+50 %），而每个 band-step 还要再付 ~2 us 的双引擎 ping-pong
   ⇒ 模型 512·8.12 + 1152·1.6 ≈ 300 us，仅 −5 % 且步数翻倍抵消，不值得。

### 20.6 本轮工具链/口径发现（Tier-0 级，待 evolver 蒸馏）

| id | vp_type | 内容 | 证据 |
|---|---|---|---|
| **VP-A14 更正（推翻）** | D | 原文「CANN 9.0.0 的 `--aic-metrics=Default` 只产出 `OpBasicInfo`，无 `aic_*/aiv_*`」**错误**：`Default` 会落盘 `PipeUtilization/ArithmeticUtilization/Memory/MemoryL0/MemoryUB/ResourceConflictRatio`，缺列的是**解析器**（只 walk `OpBasicInfo`）。今后「msprof 无 pipe 列」的判定必须先 `ls` 该次 raw 目录 | `profiles/round19/base_r19_8b-long_float16/**/PipeUtilization_*.csv`（R19 自产）+ `logs/round20_diag/bench_py_round20.diff` |
| VP-A15 | D | `PipeUtilization` 的**聚合列** `aic_scalar_wait_time` 恒 0，真实 flag 等待只在 `aic_scalar_wait_id0..9_time`；`aiv_scalar_wait_time` 是「scalar 等向量」≈`aiv_vec_time`，不可当 idle | 20.1 段两个口径陷阱 + 20.3 表 |
| VP-A16 | R | `subprocess.run(timeout=)` 只杀直接子进程；`msprof op → msopprof → python launch → bishengir-compile` 四层树会留孤儿并**占住设备**，下一轮 profiling 会串行化/超时。所有 msprof 调用必须 kill 进程组（`start_new_session` + `killpg`）或超时后 `pkill -f <tag>` 再 `npu-smi` 确认 | 本轮 `probe_mid_probe-mid` 超时后残留 4 个进程（`ps -eo pid,etime,cmd`） |
| VP-A17 | C | 案例：GQA-SW Stage 4 第 20 轮——测量盲区实为解析 bug；short 档低 Ratio 由 `f=8.12 us/任务` 的每任务固定项 ÷ 平均 band 1.5 解释（每 band-step 真实工作 short/long 相同） | 本节全部 |

### 20.7 建议的下一步（给 conductor / 下一轮）

1. **值得做 `n_slots=3`，但只值 1 轮、且必须 3/3 同向才采纳**：目标项 = 每任务固定项 f（≈5.7 us 可及部分）。
   实现要点：prologue 连续产出 T0/T1 的整条 S band（trace 复制，勿用 `T.serial`，VP-A9）；steady state 改
   `adv_task=t+2`、`adv_slot=(t+2)%3`；slot 窗口 `[s*L_max, (s+1)*L_max)`，`n_slots = 3 if 3*L_max <= 15 else 2`；
   `FLAG_TASKDONE` 在 `use_pipe2` 下保持不写。WAR 闭包同 depth-2 论证（写 slot(T+2)=slot(T-1) 之前，p2(T-1) 全部
   完成 ⇒ vector 已消费完 S(T-1)）。**先验编译**：只编 8b-short 一个 shape，`bishengir-compile` 若重试 >6 次即放弃
   （body 体量是本机最可靠的失败预测因子）。
2. **short 档最值得试的 1 个点**：就是上面的 `n_slots=3`（它同时改善 short 与 long 的任务头气泡，且不改算法/布局）。
   次优是 R5 已 blocked 的 `nqb=2`（短档模型收益最大：tasks 512→256、iters 768→448），但必须先解决编译不收敛，
   否则重现 R5 的 600 s 挂死。
3. **不要再试**：`bn` 变窄 / `bm` 变宽（模型否决 + L0C 容量）、自动多缓冲与 `T.Pipelined`（R1/R4 实测惰性）、
   关闭自动同步注入（R19 设备异常）、cross_interval 批量同步（R19 单调变差）。

### 20.8 本轮工件与校验

```text
perf_opt/bench.py                                      口径修复（+214/−48）
perf_opt/logs/round20_diag/bench_py_pre_round20.py.bak 改前备份（sha 1860926b…）
perf_opt/logs/round20_diag/bench_py_round20.diff       完整 diff
perf_opt/logs/round20_diag/run_r20.py                  本轮采集驱动
perf_opt/logs/round20_diag/base_r20*.json|.log         3 个 workload 的 pipe 原始结果
perf_opt/logs/round20_diag/model_fit.json              f/u 拟合与每任务/每步分解
perf_opt/logs/round20_diag/cfg_probe_mid_probe-mid_float16.json  probe 配置（未产出 RESULT —— 编译阻塞）
perf_opt/profiles/round20_diag/base_r20_{8b-short,8b-long,70b-short}_float16/  3 个 raw msprof op
perf_opt/profiles/round20_diag/probe_mid_probe-mid_float16/                    编译失败现场（无 CSV）
```

未改动：`_gqa_sw_fwd_kernel.py`（sha `6acdc11c…` 不变）、`perf_records.jsonl`、`workload_inventory.json`、
`perf_feedback.md`（未产出）、wrapper、benchmark、manifest、tests、`DESIGN.md`、`.stage_state.json`。
