# GQA 三算子 40% Ratio 指标边界说明报告

> 版本：v1（2026-11-08）
> 范围：`GroupedQueryAttentionSlidingWindowFwdOp` / `GroupedQueryAttentionSlidingWindowVarlenFwdOp` / `GroupedQueryAttentionBwdOp`
> 设备/工具链：Ascend 910B3 ×1（20 AIC / 40 AIV）、CANN 9.0.0、msprof 25.5.0、tilelang 0.1.2
> 数据性质：**SW / Varlen 的 Ratio 为官方 report 实测；Bwd 的 R21 Ratio 为按时延比推算（未重测官方 Ratio），报告中已显式标注。**
> 证据索引：见 §9；所有数字可在对应 `perf_opt/` 工件与 `reports/tileops/` 中逐条对账。

---

## 0. 执行摘要

1. **40% 在本算法族不可达**：40% × `computility 299.36 TOps/s` = **119.74 TOps/s** 持续算力，等于同机纯 GEMM 实测峰值（270.1 TFLOPS）的 **44.3%**；而 FlashAttention 范式的混合算子（5×GEMM + Softmax + 带状掩码 + 多路归约 + GM workspace 往返）当前 Bwd 仅 **29.3 TOps/s = 纯 GEMM 峰值的 10.9%**。
2. **当前 per-operator**：
   - SW 定长前向：**18.52%**（官方 8 档实测）
   - Varlen 变长前向：**19.87%**（官方 8 档实测）
   - Bwd 反向：**9.85%**（R21 推算；几何 9.79%）
   - **三算子批平均：16.08%，距 40% 2.49×**
3. **分算子差距**：SW **2.16×**、Varlen **2.01×**、Bwd **4.06×**；Bwd 是最远的一项，不能用它代表整批。
4. **已达成**：Bwd R20→R21 实测 **−25.22%**（相对 Stage-3 约 24×，推算）；SW **1.46×**；Varlen **2.06×**；三者精度门全绿、官方 report 均 passed。
5. **剩余有界杠杆**：Bwd `pds` BN=128（估 ~15%）、SW `n_slots=3`（估 5–12% short）。**即使全部拿到，Bwd ≈11.3%，距 40% 仍 3.53×**；乐观到 13% 仍差 3.08×。
6. **建议**：以“同族最佳实现效率”或“per-operator 实测上界”替代统一 40%；若 40% 必须保留，应明确限定为纯 GEMM 类算子，或明确它指的是 `max(GM, compute)` 的整体 roofline 而非 GM 字节项。

---

## 1. 结论先行

| 算子 | 当前 Ratio | 数据性质 | 距 40% | 主要限制 |
|---|---:|---|---:|---|
| SW 定长前向 | **18.52%** | 官方 report 实测（8 档均值） | 2.16× | 任务级固定成本、UB 192KB、flag 16、编译器稳定性 |
| Varlen 变长前向 | **19.87%** | 官方 report 实测（8 档均值） | 2.01× | 编译器（第 4 块活 L1 / 成对展开体）、UB、flag |
| Bwd 反向 | **9.85%**（几何 9.79%） | R21 按 R20 Ratio × 时延比推算 | 4.06× | 每迭代发射 / Vector 侧 `pds` / workspace 往返 / L0C&UB |
| **批平均** | **16.08%** | 2 实测 + 1 推算 | **2.49×** | — |

> 口径声明：SW/Varlen 的 Ratio 来自各自官方 `run.json` 的 8 个 tune 用例；Bwd 的 R20 Ratio 7.32% 是官方实测，R21 尚未重测官方 Ratio，9.79% 是按“counted flops 不变、时延按 R21 final”推算，**不是实测**。

---

## 2. 指标定义审计（最优先）

“40% 的 **GM Read + Write** Ratio” 至少有两种互不兼容的解释，差一个数量级：

| 解释 | Bwd 当前值（R21 推算） | 距 40% |
|---|---:|---:|
| 整体 roofline Ratio = `max(GM项, compute项)` | **9.79%（几何）/ 9.85%（算术）** | 4.06–4.09× |
| **仅 `GM Read + Write` 字节项** | **1.23%–4.06%**（8b-short 4.06%、8b-long 1.42%、70b-short 3.63%、70b-long 1.23%） | **9.8–32.5×** |

原因（Bwd opt_log §0.1.1 实测）：
- `AI = 1182.2 ops/byte`，`bw = 1.75 TB/s` ⇒ `AI×bw = 2069 TOps/s` **≫** `computility = 299.36 TOps/s`；
- 因此 roofline 的**绑定项是 compute，不是 GM**；`GM` 项天然很低，且**优化 compute 不会抬高 GM 项**。

**必须向指标方澄清的 5 个问题**：
1. 40% 是 **per-operator** 还是 **三算子平均**？
2. 分母用 **msprof 实采 counted flops** 还是 **manifest 理论 flops**？（后者把被 causal/窗口 mask 掉的块也算入，会人为压低 Ratio）
3. 是否包含 pre/post 辅助 kernel？（Bwd 的 main 占 99.0–99.7%）
4. `computility` 是 **40 cube 单元口径 299.36 TOps/s**，还是 20 AIC 的设备峰值？
5. 指的是 `max(GM, compute)` 的**整体 roofline**，还是**仅 GM Read+Write 项**？

> 若第 5 问的答案是“仅 GM 项”，则 40% 与该算子的物理位置不匹配：Bwd 是 compute-bound，GM 项 1.2–4.1%，40% 在该口径下差 10–33×，且不是靠优化能抬起来的。

---

## 3. Roofline 证据

| 项 | 数值 | 来源 |
|---|---:|---|
| computility（40 cube 单元） | 299.36 TOps/s | Bwd opt_log §0.1.1 |
| 40% 对应持续算力 | **119.74 TOps/s** | 40% × 299.36 |
| 同机纯 GEMM 实测峰值 | 270.1 TFLOPS = **90.2%** of computility | `torch.matmul(4096³)` 同机探针 |
| 纯 GEMM 的架构上限 | **90.2%** | 同上 |
| **40% 占纯 GEMM 峰值的比例** | **44.3%** | 119.74 / 270.1 |
| Bwd 当前持续算力 | 29.3–29.5 TOps/s | 9.79–9.85% × 299.36 |
| Bwd 占纯 GEMM 峰值 | **10.8–10.9%** | 29.5 / 270.1 |
| Bwd 距纯 GEMM 峰值 | **9.2×** | 270.1 / 29.3 |

> 注意措辞：**44.3% 是“目标占纯 GEMM 峰值的比例”，不是“架构上限”**。架构上限是 90.2%；44.3% 是一个对混合算子非常高的持续利用率目标。

**为什么混合算子达不到**：Bwd 的 5 个 GEMM 被 ~12 条 v-op（exp2、mask、cast、reduce）、workspace 往返（S/dW + P/dS = 12 B/score-elem）和跨引擎握手切开；R20 实测 `aiv_scalar` 占 V 核墙钟 **70%**、cube 仅 **5.5%**。R21 的 BN=128 把 `mul` 的每迭代固定成本摊薄（−42.9%/−48.2%），但 `pds`（纯 Vector）仍占 35–40%，它不是 GEMM、也不吃 cube 算力。**这是“非 GEMM 工作量 + 每迭代固定成本”问题，不是“再调几个 tile”能补齐的。**

---

## 4. 分算子瓶颈证据

### 4.1 Bwd（`_gqa_bwd_kernel`）

- R20 3-pass（Cube-only / Vector-only / Mix）已把 `dq` 区域原子与 GM 通道挤占消掉：`aiv_mte3 220.3→4.8 µs/block`、`aiv_mte2 141.1→18.3`、`aic_scalar_wait_id2 409.1→23.6`；官方 Ratio 1.46%→**7.32%**（2.90×）。
- R20 剩余：`aiv_scalar` = 106.1 / `aiv_time` 150.3 = **70%**，cube 占比 **5.5%**；每迭代固定成本（发射 + 2× AIV 冗余）是主项。
- R21 BN=128（capacity-partitioned）：`sdw`/`mul` 用 128、`pds` 保持 64 行子块；8 档 **−21.4%~−29.1%**，几何 **−25.22%**。逐 pass：`mul` −42.9%/−48.2%、`sdw` −10.3%/−17.5%、`pds` +0.9%/+2.2%（分区预期）。
- R21 后 `pds` 仍占总时长 **35–40%**，是下一个结构杠杆；L0C 在 `mul` 已 **100%**（32768/32768 fp32），`pds` 的窄口是 **UB 228.5 KB vs 192 KB**。

### 4.2 SW 定长前向（`_gqa_sw_fwd_kernel`）

- R19 的“msprof 工具链盲区”**已被推翻**：raw 目录里本来就有 `PipeUtilization_*.csv`；是 `bench.py` 只 walk `OpBasicInfo*.csv` 的解析 bug。
- short 档低 Ratio 的定量模型：`墙钟 = f·N_task + u·N_step`，**f = 8.117 µs/任务**、**u = 2.852 µs/band-step**；short 的 L≈1.5、long 的 L≈3.75 ⇒ short 的 1.65× 慢**全部**来自 `f÷L`。每 band-step 真实工作在 short/long 相同（cube 1.204 vs 1.186、mte1 0.925 vs 0.908、fixpipe 0.924 vs 0.954，≤2%）。
- 进一步的 tile/slot 改动被封：UB 192 KB（`bm=80` 需 204.8 KB）、flag 16（`n_slots·L_max ≤ 15`）、`T.serial(1)` / 第 4 块活 L1 触发 `bishengir` SIGSEGV。

### 4.3 Varlen 变长前向（`_gqa_sw_fwd_varlen_kernel`）

- R13 G=2 head-group 已把 `GM→L1 126,064→77,280 KB（−38.7%）`、`aic_mte2 0.667→0.433`；8 档 −49%~−54%（相对 Stage-3），官方 8 档 Ratio 14.05–25.15%。
- 手写片内双缓冲（理论 1.52×）被封：新增第 4 块活 L1 的 GM→L1 `T.copy` 触发 `bishengir-compile` SIGSEGV；成对展开循环体不收敛；唯一可编译的 M1c 部分重叠 **回退 −1.4%**（L1 端口互拖）。
- 进一步 tile 被封：UB 192 KB、`2·L_max ≤ 15`、`vselect` 编译不终止、`T.vsub(f16 tile, f32 scalar)` 编译拒绝。

---

## 5. 已达成的优化（硬证据）

| 算子 | 主要优化 | 相对 Stage-3 | 精度门 | 官方 report |
|---|---|---|---:|---|
| Bwd | int32→fp32 `vcmp`（3.47×）→ `dq` rank-2 区域原子（2.90×）→ 3-pass 相位拆分（1.73×）→ R21 BN=128（−25.22%） | **约 24×**（R20 18.1× × 1.337，推算；Stage-3 用 launch=10，口径需注明） | L0 8/8 + L1 12/12 + L2 12/12 + Boundary 8/8（=40），两次 `--level all` EXIT=0 | `20261106_203252_067780` passed（R20；R21 官方 Ratio 未重测） |
| SW | depth-2 跨任务流水（R3）+ G=2 头组（R18b） | **1.46×**（8 档 −19%~−39%） | `--level all` **32/32** | `20261106_105557_262003` passed（22/22 correctness + 10/10 benchmark） |
| Varlen | pipe2 + 列 OOB 守卫（−36.2%）+ S 载体 f32 + G=2 头组（R13） | **2.06×**（8 档 −49%~−54%） | `--level all` **51/51** | `20261108_013426_279753` passed（26/26 correctness + 10/10 benchmark） |

> 口径提示：SW 的 `--level all` 是 32/32（不是 51/51）；Varlen 的 51/51 是内嵌测试，官方 report 是 26/26 + 10/10；Bwd 的 8/8 是 L0，完整分层是 40。三者不可混写。

---

## 6. 剩余有界杠杆（坦诚列出）

| 杠杆 | 预估收益 | 前置条件 | 拿到后的 Ratio | 距 40% |
|---|---:|---|---:|---:|
| Bwd `pds` BN=128 | ~15%（结构级，待探针） | 省 `neg_buf` 兼作 mask 限值（−32 KB）+ `vexp2` scratch 复用（−32 KB），越过 36.5 KB 缺口 | **≈11.3%**（9.85 × 1.15） | **3.53×** |
| SW `n_slots=3` | 5–12% short / 0–5% long | `L_max≤5` 时 `3×5=15` 恰好塞进 16 id；先只编 8b-short，`bishengir` 重试 >6 次即放弃 | short 档 ~14.8–15.7% | ~2.5–2.7× |
| Varlen 手写双缓冲 | 理论 1.52× | 需先解编译器（第 4 块 L1 / 成对体） | 不可估 | — |

**结论**：即使把 `pds` BN=128 拿满，Bwd 约 **11.3%**，距 40% 仍 **3.53×**；即使乐观到 13%，仍差 **3.08×**。SW `n_slots=3` 拿满也只到 ~15% 量级，仍差 2.5× 以上。**两个有界杠杆合计也不改变“40% 不可达”的结论。**

---

## 7. 建议的目标重定义

1. **按算子设实测上界**：SW ≥18%、Varlen ≥19%、Bwd ≥11%（`pds` 后），而不是统一 40%。
2. **以“同族最佳实现效率”为基准**：Bwd R21 的 9.79% 已是旧 8.2% 参照实现的 **1.19×**；用“相对最佳实现的效率”替代绝对 Ratio。
3. **若 40% 必须保留**：明确限定为**纯 GEMM 类算子**，或明确它是 `max(GM, compute)` 的整体 roofline、且分母用实际 counted flops、不含 pre/post、`computility` 用 299.36 TOps/s。
4. **分阶段目标**：先把 Bwd 做到 ~11.3%（`pds` BN=128），再评估是否值得投入编译器能力攻坚（第 4 块 L1 / 成对体），而不是把 40% 当成单轮可达目标。

---

## 8. 风险披露（必须在正式报告中保留）

1. **Bwd R21 Ratio 是推算，不是实测**：9.79%（几何）/ 9.85%（算术）由 R20 官方 Ratio × 时延比得到；官方 Ratio 需重测。
2. **Bwd 6/8 档为单 run**（与 R20 final 同口径）；仅 8b-short/8b-long fp16 做了 3 次重复（极差 0.11%）。
3. **8.2% 是参照实现，不是理论上界**：它是 910B2 / 24 核上的 3-pass 参照；R21 在 910B3 / 20 核上已更快（short 945 vs 1016.8；long 5411 vs 7518.6），所以“8.2% 上界”的说法已失效。
4. **跨平台经验判断**：如写“FlashAttention-2/3 在 GPU 上也难到 40%”，必须标注“本仓未复现，属行业经验判断”，不得作为实测结论。
5. **口径不一致**：Bwd opt_log 有一处“37 PASS”计数笔误（分层之和 40）；Stage-3 基线用 launch=10，与后续 20 次中位有系统性差异。
6. **“已榨干所有优化”不可写**：至少还有 `pds` BN=128 与 `n_slots=3` 两个已知未做项；应写“已知高价值项已做尽，剩余两个有界项”。

---

## 9. 证据索引

| 内容 | 路径 |
|---|---|
| Bwd R21 opt_log（§21） | `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_bwd/grouped_query_attention_bwd_kernel/perf_opt/opt_log.md` |
| Bwd perf_records（8 final / 8 merged / 5 candidate / 16 baseline） | 同上目录 `perf_opt/perf_records.jsonl` |
| Bwd R20 Ratio 7.32% | 同上 `opt_log.md` §10.8 |
| Bwd R21 final 时延 | 同上 `perf_records.jsonl`（`final-r21-bn128-hpds`） |
| Bwd `[DESIGN_LIMIT]` 旧反馈 | 同上 `perf_opt/perf_feedback.md`（8.2% 参照，需更新） |
| SW R18b/§20 诊断 | `.../grouped_query_attention_sliding_window_kernel/perf_opt/opt_log.md` |
| SW 官方 report | `examples/TileOPs/reports/tileops/20261106_105557_262003_GroupedQueryAttentionSlidingWindowFwdOp/run.json` |
| Varlen R13/§13 | `.../grouped_query_attention_sliding_window_varlen_kernel/perf_opt/opt_log.md` |
| Varlen 官方 report | `examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/run.json` |
| 三算子批评审材料 | `GQA_THREE_OPS_OPTIMIZATION_REVIEW.md`（§11 执行更新） |

---

## 10. 可直接对评审说的一段话

> 40% 对应 119.7 TOps/s 持续算力，是同一颗 910B3 上纯 GEMM 实测峰值（270.1 TFLOPS）的 44.3%；而 FlashAttention 反向的 5 个 GEMM 被 Softmax、掩码、workspace 往返和跨引擎握手切开。我们已把 Bwd 从 1.46% 推到 7.32%（R20），再推 25.22%（R21），推算 9.85%，是旧 8.2% 参照实现的 1.19 倍；SW/Varlen 已实测 18.52%/19.87%。剩余两个有界杠杆（`pds` BN=128 ~15%、`n_slots=3` 5–12%）拿满后，Bwd 约 11.3%，距 40% 仍 3.53×。因此 40% 不是“再调几个 tile”能补齐的；建议按算子实测上界或同族最佳实现效率重设目标，并把 40% 明确限定为纯 GEMM 类算子。
