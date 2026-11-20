# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `_sparse_mla_kernel_DESIGN.md` §1.6.0 R4（设计期定量 roofline）的**流量项假设**——「gather 到的 KV 字节数 × 1.75 TB/s = gather 时间」，以及 §3.3 的 gather 实现选型「row-contiguous `T.serial(block_N)` + 逐行 `T.copy` + 单次 clamp」。实测该形态并不按带宽流式搬运：每行（1152 B）固定成本 **≈1.34 µs**（1000+ 行不变），整核 Task Duration 的 **94.5%** 消耗于此。该瓶颈**不是 tiling/参数可解**：`block_N ∈ {16,32,64}`、`num_stages`、行循环形态（`T.serial` / `T.Pipelined(ns=4)` / `T.Parallel` / 8 行手工展开 / `T.unroll`）、gather 切片形态（4D 切片 / 2D 显式切片 / `base+size` / KV 降秩为 2D / 目的域 `alloc_L1`）共 12 个保持语义的分支全部落在 3% 噪声内或回退（`opt_log.md` §3 Round 1）。
- 结构性加速估计: **≥8×（保守下界）**，实测锚点 **18.1×**。依据=实测外推（同口径 `msprof op Task Duration`）：把逐行动态索引 gather 换成连续块 copy（数学无效的降级探针，仅用于归因）后，同一 smoke shape 的 kernel 时延 **9022.3 µs → 499.2 µs（−94.5%，18.1×）**，`aiv_vec_ratio` 0.000617→0.010891、`aic_scalar_wait_id1_time` 343.3 µs→15.0 µs。按同一每行成本外推到 tune workload 的可落地结构（§假设 中的 dense-scan + bitmap 变体，连续块读 ≈77 ns/行）：mainstream 566.5 ms→≈6–10 ms（≈60–90×）、long-kv 35.6 ms→≈4–5 ms（≈8×，该 shape 稠密扫描的字节数是稀疏的 16×，为最保守一档）。外推部分（非实测）已在下文逐条标注。
- 与设计假设**直接矛盾**（第二门槛的另一形态）: DESIGN §1.6.0 R4 对 mainstream 的估算为 **4.0–9.4 ms**，实测 baseline **566.5 ms**，偏差 **60×–141×**；失准项为「gather 流量→时间」与「发射/延迟不计」两条前提。

## 参照锚定

- 参照实现:
  1. `examples/deepseek_v32/sparse_mla_fwd_exp.py` L250-290：稀疏 MLA 的 **两相位**结构——先把选中行 gather 进 **GM workspace**（`T.copy(ub_kv_sparse, workspace_kv[offset:offset+block_I, 0:full_dim])`，连续写），计算相位再从 workspace 以**连续块** `T.copy` 读入，并用 `sync_block_set/wait` 做相位同步。等价思想：把「逐行随机读」压缩成「逐行随机读一次 + 多次连续块读」。
  2. `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py` L88-...：同一 gather→片上/工作区→块读的两段式。
  3. `.agents/skills/tilelang-op-optimize/references/pattern-library/attention.md` PL-1.37（`gemm-prefix-columns-developer`，**status: verified 但 toolchain 戳为 0.1.2+00aeb862，对本轮 0.1.2+5a90b99 为 stale**）：该条目只保证 PV 前缀列语义；本轮按其行为在本工具链**复验通过**（`_sparse_mla_kernel.py --level all` L0 5/5、L1 5/5，PV 前缀列 GEMM 未变），**不作为本次天花板判断的依据**。
- 不可移植论证:
  1. **两相位需要 GM workspace + 跨相位同步**：本 kernel 冻结为「Developer MixCV **非持久**一维 grid（`num_logical = B·S·(H/BH)`，mainstream 4096 个逻辑核 ≫ 20 物理核）」；workspace 相位要求同一 `(b,s)` 的 gather 与其 4 个 head-block 计算落在**同一核的串行相位**内。Developer 模式下「核内 persistent 循环 + `T.gemm`」被 `TRAP-DEVMODE-PERSIST-GEMM`（`capability-gaps.md` CG-2026-0010，本任务已独立二次识别）阻断，因此不能在现有 grid 语义内插入 gather 相位。**依据：实测**（迁移期与本次 `--level all` 的 L2-5 路径均复现该限制；DESIGN §5.5 / §9.4 记录）。
  2. **参照实现是 prefill/可变长 sparse-MLA-fwd**，其 `block_H_half`、`kernel_id` 切分与 workspace 槽位布局绑定「单核处理多 head-block」的 grid 假设，与本 op 的 decode grid（每核一个 head-block、一维波次调度）不同构；直接搬运会把 KV 重读（H/BH=4）替换成 workspace 的读写往返，收益需重新实测。**该条为设计推断，标注为未实证**。
  3. **dense-scan + bitmap 掩码**（本反馈建议的替代结构）：用 `indices` 先在 UB 内建 `seq_len_kv` 位图，再对因果窗口做**连续** `T.copy` 块读并逐列套掩码。它是精确等价的（掩码集合 == `indices` 选中集合 ∩ causal），但要新增位图构造（对 topk 个索引做 scatter/位运算）与窗口上界管理，属**算法与数据流层面的重设计**，超出「行连续 gather」冻结契约，本 Stage 4 不得实施。**等价性为推理（未实测）**。
  4. 未找到任何「在冻结结构内、不改数据流即可消除逐行 gather」的同族先例——仓库内所有稀疏 attention 先例（上列 1/2）都使用 workspace 或 Expert 双 Scope；依据：`grep` 全仓 `T.copy(.*\[.*idx` 的稀疏 gather 用法清点（`examples/indexer/indexer_bwd.py`、`examples/deepseek_v32/*`、`examples/Wechat-YATT/deepseek_v4/*`）均为逐行 copy，且均配有额外相位/工作区。

## 假设

DESIGN 原文口径（`_sparse_mla_kernel_DESIGN.md`）：

- §1.6.0 R4 表（候选 2，主选 BH=32）：`Q 151M + gather 2.42G × (H/BH=4) = 9.66G + idx 8.4M + O 134M ≈ 9.95 GB`，`roofline = 9.95 GB / 1.75 TB/s`，**估算 4.0–9.4 ms**（mainstream）。
- §3.3 / §6 数据流：`GM[kv] --T.copy(串行行 gather, clamp 索引)--> L1[KV_shared (block_N,576)]`，即假设「逐行 `T.copy` 的代价 ∝ 字节数 ÷ 带宽」；设计期只论证了该形态**规避** `TRAP-C10`（跨步 gather 缺失），未对「逐行动态索引 copy 的固定成本」做实测/估算。
- §1.6.3 实验裁决：主选 BH=32（KV 重读 ×4）vs 备选 BH=64（×2），裁决常数依赖未实证的 L0C/UB 容量与转置代价。

## 实测

口径：`msprof op` → `main_mix_aic` 的 `Task Duration(us)`（唯一口径，无 NPU event / 端到端时间）。

| 项 | shape / workload | 实测 | raw profile |
|---|---|---:|---|
| baseline（主选结构） | mainstream | **566 545.37 µs**（8 launches） | `perf_opt/profiles/base_mainstream/` |
| baseline | longer-kv-lower-topk | 141 754.06 µs（5） | `perf_opt/profiles/baseline_longer-kv/` |
| baseline | long-kv-low-topk | 35 598.48 µs（5） | `perf_opt/profiles/baseline_long-kv/` |
| **降级探针**：64 次逐行 gather → 1 次 64 行连续 copy（数学无效，仅归因） | smoke（同代码路径） | **9 022.3 µs → 499.2 µs（18.1×）** | `perf_opt/profiles/{probe,probe_contig}/` |
| 逐行成本（由 3 个 workload 分别反算） | mainstream / longer-kv / long-kv | **1.34 / 1.34 / 1.36 µs/行** | 同上 |
| gather 占比（`probe` pipes） | smoke | `aic_cube_time` 1.64 µs、`aiv_vec_time` 5.56 µs vs `aiv_mte2_time` 172.4 µs、`aiv_scalar_time` 166.2 µs（block 墙钟 361.9 µs） | `perf_opt/profiles/probe/` |
| 保持语义的 12 个分支 | smoke + long-kv | 8 959.8 – 10 598.6 µs（smoke）；`c_bn32` 37 974.0 µs（long-kv，+6.7%）、`c_kvl1` 35 713.6 µs（+0.3%） | `perf_opt/profiles/probe_g1,g5,g6,probe_par,probe_l1,probe_flat,probe_bs,base_bn32,base_bn16,cand_*/` |
| BH=64（本工具链复核） | smoke | compile fail：`ub overflow, requires 2115584 bits while 1572864 bits available`（264 448 B > 196 608 B）；容量算术 L0C 144 KB > 128 KB | `perf_opt/logs/l0/final_all.log`（L2-5 WARN）、`perf_opt/probes/_probe_bh64f16.py` |
| 结构分支（2 路 buffer 拆 gather） | smoke | compile fail：`'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!` | `perf_opt/probes/_probe_split2.py` |

`final` 与 `baseline` 差异 ≤0.2%（噪声），故本反馈不是「调优不到位」，而是「结构族内已无参数解」。

## 影响面

- 直接影响：任何**以逐行/逐元素动态索引从 GM gather 到片上**的 npuir Developer 算子——稀疏 MLA decode（本算子及 decode 变体）、paged KV attention（`mha_decode_paged` 系列的 block-table gather）、MoE 路由 gather、稀疏 indexer/bwd（`examples/indexer/*`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 同形态）。凡 roofline 以「gather 字节数 ÷ 带宽」建模的设计，估算会低 1–2 个数量级。
- 对 DESIGN 的影响：§1.6.0 R4 的 roofline 需要增加「逐行 gather 固定成本」项（本次给出可用的常数：≈1.34 µs/行 @910B3 + tilelang 0.1.2+5a90b99；连续块 ≈77 ns/行），并在 §3.3 gather 选型处把「两相位 workspace / dense-scan + bitmap」列为候选而非唯一形态；§1.6.3 的 BH 裁决需补一条结论：BH 受 L0C 144 KB > 128 KB 与 UB 264 448 B > 196 608 B 双向封顶，H=128 下可选取值仅 {32}，KV 重读 ×4 为冻结结构必然代价，且**不是**主导项。
- 预期影响：设计修订（算法/数据流层）→ 需重走 Stage 1（§1.6 候选与 roofline）+ Stage 3 实现，工作量约等于一次中等改动；Stage 4 侧无可保留的增量。
- 不影响：数值语义（`indices` topk 选取、`stride_kv`、`q_start_index_s`、causal 裁剪、`sm_scale`、`dim_tail=64`）、L0/L1 精度基线、wrapper/接口契约。

## 反馈结论

本算子 94.5% 的 kernel 时延由「逐行动态索引 gather」的固定成本（≈1.34 µs/行）而非 FLOPs/带宽决定，冻结结构（行连续 gather + Developer MixCV 非持久一维 grid）内已无 tiling/参数解；实测与 DESIGN §1.6.0 R4 的 4.0–9.4 ms 估算相差 60–141×，且同口径降级探针显示改成连续块读可得 18.1×，建议按设计修订路径重开 gather 形态（两相位 workspace 或 dense-scan + bitmap 掩码）并回填 roofline 常数。

建议路由: 设计修订
