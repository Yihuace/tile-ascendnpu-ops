# 任务算子登记表（Task Operator Registry）

> **用途**：本文件是「我负责的算子集合」的唯一出处。任何「还有哪些没达标 / 现在什么状态 / 该做什么」的询问，一律以本表为准，不再临时翻报告。
>
> **维护规则**：
> 1. 新增/移出算子必须改本表；状态列随每次任务收尾更新。
> 2. `当前值` = 最新一次**有效**官方报告的 tune 案例均值（排除 `smoke` 与 `### torch` 行，按 `## <OpName>` 分节解析）。
> 3. 达标线：**60%**。判定口径见下。
> 4. `依据报告` 必须是 `reports/tileops/<run_id>/` 下真实存在的目录。

## 验收口径（2026-11-02 起）

| 口径 | 定义 | 用途 |
|---|---|---|
| **B（主口径）** | `Ratio = ideal_declared / duration`，`ideal = max(bytes/BW_peak, flops/peak)`，flops/bytes 取算子自带 `eval_roofline()` | **达标判定** |
| A（诊断） | msprof `_parse_roofline`（`GM/L2` + `GM Read + Write`），分子为**实际计数 op** | 并列诊断，不作判定 |
| 达成率（辅助） | `(ideal + F) / duration`，`F` = 该 kernel 实测的不可压缩固定开销 | 判断「是否已贴天花板」 |

> 为什么 B 作主口径：A 的分子随实现变化，会**系统性低估**「每元素 op 少、时延几十 µs」的算子（Dropout 在 A 下数学上限仅 59.6%，减 op 反而降分）。

## 算子清单（24 个）

| # | 算子 | family | 当前值 | 判定 | 可达上界 | 阻塞点 / 说明 | 依据报告 |
|---:|---|---|---:|---|---:|---|---|
| 1 | LerpTensorFwdOp | elementwise | **74.08%** | ✅ 达标 | — | — | `20261025_230253_045981` |
| 2 | ClampFwdOp | elementwise | **69.11%** | ✅ 达标 | — | — | `20261016_115833_200802` |
| 3 | HardtanhFwdOp | elementwise | **69.02%** | ✅ 达标 | — | 3 轮 tuned，2.4–2.8 TB/s 已贴地板 | `20261015_235308_236667` |
| 4 | ErfFwdOp | elementwise | **68.90%** | ✅ 达标 | — | 本轮 66.40→68.90（时延 −32.7%） | `20261102_015413_041505` |
| 5 | EluFwdOp | elementwise | **67.84%** | ✅ 达标 | — | — | `20261015_235308_236667` |
| 6 | LeakyReluFwdOp | elementwise | **67.20%** | ✅ 达标 | — | — | `20261015_235308_236667` |
| 7 | Expm1FwdOp | elementwise | **62.96%** | ✅ 达标 | — | — | `20261015_053857_371455` |
| 8 | BitwiseNotFwdOp | elementwise | **77.12%** | ✅ 达标 | — | — | `20261015_053857_371455` |
| 9 | DropoutOp | regularization | **62.00%**（B）/ 59.26%（A） | ✅ 达标（B） | — | 本轮 58.27%→62.00%；9 op，真实时延 −8.5% | `20261102_154948_981690` |
| 10 | AlibiFwdOp | elementwise | **56.29%** | ⚠️ 未达标 | — | **用户确认：他机已优化，本机不再跟进** | `20261015_205040_770919` |
| 11 | ArgmaxFwdOp | reduction | **36.71%**（官方，3/3 tune 案例） | ⚠️ 未达标（但已 1.42×） | 41–47% | **两个 bug 已修**：根因同源——`@torch.library.custom_op` 路径下 `bishengir-compile` 失败 → 重试把 `block_m` 减半 → **静默算错**（14/42 用例）并在 (256,4096) 抛 `vector core exception`。已加 `TILEOPS_BYPASS_CUSTOM_OP`（默认开，直调 jit 工厂）→ **42/42 correctness 通过**，官方 25.83%→**36.71%**（20.42/38.36/11.08 µs）。⚠️ 残留：`lm-head` 两档被工具链 `custom op 'syn_instr_mode' is unknown` 阻塞（既有问题，非本次引入） | `20261103_090008_426947` |
| 12 | EngramDecodeOp | sequence_modeling | **14.64%**（官方单 kernel）/ **9.95%**（op 级 A+B） | ⚠️ 未达标（但已 4.2×） | ~3.8%（旧结构）/ 9.95%（多核 op 级） | **已集成多核两阶段版**（`_engram_decode_mcore.py` + wrapper 开关 `TILEOPS_ENGRAM_MCORE`，默认开）：官方 3.46%→**14.64%**（5.98/13.44/9.08 µs），op 级（stats+apply）≈9.95%。⚠️ 官方值只捕获到一个 `main`（stats），apply 未计入；两版本 host-wall 均 ~770 µs（host 主导，设备时间才是差异源） | `20261102_234330_954868` |
| 13 | EngramGateConvBwdOp | sequence_modeling | **0.038%**（真实 op 级，旁路后）｜0.030%（旁路前）｜报告 0.53%（仅 k1） | ⚠️ 未达标（且被报告高估 18×） | 26%（估算） | 实为 **k1/k2/k3 三 kernel**；op 级实测 **1368/1416/1934 µs**，而报告只测到 k1（50.8/59.7/225.8 µs）→ 慢 8.6–26.9×。**逐 kernel 实测**（M=2,S=64,d=512）：k1 60.0 / k2 139.9 / k3 51.9 µs，**三者合计仅 252 µs**，而 op 级 1416 µs ⇒ **≈1164 µs（82%）不在任何 kernel 内**，**分层归因已完成**：`@torch.library.custom_op` 包装 **≈570 µs/次**、张量分配 **247 µs**（加缓存无收益已回退）、3 次 TileLang 逐次 launch Python 开销 ~200 µs、真设备 kernel 仅 **350–410 µs**（k1 60 / k2 140 / k3 52）⇒ **~85–90% 是框架开销**。已实施 **custom-op 旁路**（`TILEOPS_BYPASS_CUSTOM_OP`，默认走 `self.kernel` 即 `_run`）：op 级 1416→**1135** / 1368→**1031** / 1934→**1659** µs（−14~25%），op 级正确性 5/5 通过、官方报告 passed；真实 Ratio 0.030%→**0.038%**。⚠️ 该旁路默认生效会失去 torch.compile / fake-tensor 支持（置 `TILEOPS_BYPASS_CUSTOM_OP=0` 可恢复）。**三 kernel 合并已评估为不可行**：k1/k2 grid=M·num_tiles（多块），k3 grid=KS+2 且对 `ws` 做**跨块全局归约**（`T.reduce_sum(ws_t, dim=0)`）⇒ 合并需要跨核同步/栅栏，而该能力不存在（CG-2026-0018） | `20261102_191936_396302`（复测）/ `20261017_043414_656448`（旧） |
| 14 | FP8LightningIndexerOp | attention_indexing | **16.64%** | ⚠️ 未达标 | ~83%（fp16 epilogue 后） | 本轮 9.0→16.64（1.85×）；40% 需 4.4×，两轮结构尝试回退 | `20261101_120609_257701` |
| 15 | GQA-SlidingWindowFwdOp | attention | **18.52%** | ⚠️ 未达标 | 100%（理论） | 6 轮 + **R18 G=2 head-group batching 已转正（sha `6acdc11c`）**：H/Hkv=4，2 个 query head 共享 K/V；8 用例官方 Ratio 14.06/14.27/22.74/21.28/14.18/14.80/22.64/24.16 → 均值 **18.52%**（旧 16.83%，~1.10×）；L0/L1/L2/Boundary **32/32**、官方 correctness 22/22 + benchmark 10/10。编译器 workaround：`T.serial(1)` → `range` 展开、编译 150s 超时+重试、官方 report 逐 case 独立进程。40% 仍不可达（同 Varlen：单管道 1.55× + CG-2026-0029/前端缺口）。**R19 Tier-0（AUTO_SYNC off / cross_interval）= no_gain**：A 关 `NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC` → 首个 L0 case 即 `L0C read/write conflict`（自动注入是 PIPE_C→PIPE_FIX 核内 WAR 的兜底，非冗余）；B `cross_interval=2/4` → 8b-long **−1.6% / −12.3%**（细粒度 flag 正是 CV 重叠来源，批量反而让消费者多等 1–3 step）。**910B3 可表达的方法已穷尽**；cannbot-skills 的 950 FA 最佳实践（L0C 256KB/UB 248KB/CV 直通/4×K-V/4×L0C/14 级）与 `npu-arch` 的 DAV_2201（L0C 128KB/UB 192KB/无直通）共同证明 **40% 是 950 级目标** | `20261106_105557_262003` |
| 16 | GQA-SlidingWindowVarlenFwdOp | attention | **19.87%**（20261108 新报告；旧 19.92%） | ⚠️ 未达标 | 100%（理论） | 9 轮 tuned + **R13 G=2 head-group batching 已转正（sha `6befe384`）**：H/Hkv=4，2 个 query head 共享一次 K/V → mte2 0.65→0.43；8 用例官方 benchmark 14.32/14.20/25.02/24.35/15.78/15.53/25.45/24.77 → 均值 **19.92%**（旧 17.53%，1.10–1.15×）；L0/L1/L2/Boundary **51/51 PASS**。UB 分配器实测**按生命周期复用**（R13 修正 R12 的“静态和”）：当前 kernel 真实占用 196,608B（含编译器 staging 38.7KB、零余量），G=2 经 `ub_f16_D/f32_D → N` 别名省 24.6KB 后 189.7KB 装下。R10–R12 保留：M1 手写双缓冲封于 4th live L1、nqb=2 V 共享 UB 可行但 band-outer 封、双 kernel 3.93× 不可行。**40% 仍不可达**：单管道下界 1.55× < 1.83×。**R14 G=4 快筛 = 无效（1.043×，−4.2%，<5% 门槛）**：UB/L0C/精度均可解（acc_o f16 → plan 192,128B 编译通过；L0C 拆两个 M=2bm gemm），但 G=2 后四 pipe 和 1.176、mte2 0.427 已非单 pipe 受限；G=4 的 gemm/fixpipe issue 翻倍 + 严格串行 copy→gemm + P 按 head 往返吃掉收益。**G 轴饱和，G=2 是终点**。**R15 算法级（rolling/persistent K/V 复用）= 不可落地**：理想 1.40×（mte2 0.433→0.095），但理想形态需 5 tile/环 = **736KB > 512KB L1**；可行 N=2/3 需 **4–5 块活 L1** → CG-2026-0029 仍在（第 4 块活 L1 的 GM→L1 copy **3/3 SIGSEGV**，同缓冲 `T.clear` 29s 通过）；即使编译器修好，可行形态也只有 **1.08–1.15×**。算法级的其他候选（band-packed/细尾 5–10%、算子融合 图级、save-P/LSE 需改接口、FP8 需改精度契约）都不改变这个结论。**R16 IR workaround（避开第 4 块活 L1）= 不可落地**：单大 L1 staging 编译+L0 通过（证明 SIGSEGV 只卡「可喂 L1 分配数」，不卡总字节），但 rolling 需要的第二块 K/V **无法被 cube 消费**——`T.gemm` 忽略行偏移（sliced operand max_abs=55）、buffer alias SSA 报错、L1→L1 `cbuf→cbuf` unsupported；B2 GM workspace 不减少 GM→L1 字节（mte2-bound，无 ≥5%）；B3 Q/P 共享与 depth-2 hoist 冲突。阻塞从 CG-2026-0029 扩展到 3 个前端/编译器缺口。**R17 重设计（bm=128/bn=128+G=2）= 无效（0.920×，+8.7%）**：UB（user 140.5KB/plan ~186–190K）、flag（单槽+TASKDONE，L_max+1=12）、编译、L0 全过，但 band 字节 0.45× 只换来 **mte2 −21%**（copy 次数不变、单块 64→32KB），单槽任务边界 **+490 µs**、acc_o f16 多 2 vcast → 省 155 付 290。冻结文件单槽回退分支的 advance 会生产错任务的 S（死代码，已在原型修正）。**GQA-SW-Varlen 到达工具链上限，19.92% 不变**。官方 report correctness 因 G=2 每 shape 重编译，40min 只跑到 8/26（全 PASS）；性能数据来自官方 benchmark harness（10/10 PASS），全量正确性来自 embedded `--level all` 51/51 | `20261108_013426_279753` |
| 17 | RopeNeoxOp | position_encoding | **27.35%** | ⛔ 封顶 | **39.8%** | 3.52 µs 固定地板（launch 1.75 + 每核骨架 1.78）vs ideal 0.29–1.2 µs（实测主导）。**Round-4 受控实验（4k/8k：prefetch / fuse-cross-dtype-cast / vcast-slice）全部 no_gain → r3 逐字节保留**；4k 的 aiv_time 3.0 vs 最长 pipe 1.5 表明 pipe 已重叠饱和（prefetch 只是把等待在 pipe 间搬运），波次 8k 119/120、4k 79/80、2k 40/40 已整数最优；下一步只剩 FMA（6→4 op，能力缺口 CG-2026-0028）或 UB >192 KB | `20261101_035109_804812` |
| 18 | RopeLongRopeOp | position_encoding | **36.34%** | ⛔ 封顶 | **49.3%** | 本轮采纳 r3 1d 共享 kernel（8k 6.30→**6.08** µs，+3.6%；2d/2k 为噪声平区）；根因同 Neox：3.52 µs 固定地板 vs ideal 0.29–1.2 µs | `20261104_094733_368837` |
| 19 | RopeLlama31Op | position_encoding | **35.91%** | ⛔ 封顶 | **49.3%** | 本轮采纳 r3 1d 共享 kernel（8k 6.28→6.22 µs），但 2k 档 3.98→4.00 µs 抵消 → 均值落噪声带（两次采样 35.49/35.91；旧 36.00）；本目录 5-run 交错 A/B 判 8k +3.3% | `20261104_110558_327166` |
| 20 | EngramGateConvFwdOp | sequence_modeling | **5.57%** | ⛔ 封顶 | **15.6%** | 3 档均为微秒级 shape，被发射地板锁死 | `20261017_005814_531237` |
| 21 | GroupedQueryAttentionBwdOp | attention | **7.32%**（R20 实测）/ **≈9.85%**（R21 推算） | ⚠️ 未达标（但已超 8.2% 参照） | **已超 8.2% 参照** | 2 轮 tuned（2.90×）+ **R21 BN=128 已采纳（8 档 −21%~−29%，几何 −25.22%）** + **R20 3-pass 引擎相位拆分已转正（sha `26edd4c1`）**：pass1 Cube-only（S/dW）/ pass2 Vector-only（P/dS）/ pass3 Mix（dv/dk/dq），8 例 Ratio 6.75/6.73/7.88/7.84/6.90/6.87/7.86/7.85 → 几何均值 **7.32%**（旧 4.23%，1.73×）；相对同仓 910B2 3-pass 参照 8.2% 已到 89%；`--level all` 37 PASS、correctness 5/5。`n_kv==1` 编译器 SIGSEGV → 融合回退。40% 仍不可达（需参照的 4.9×） | `20261106_203252_067780`（R20；R21 官方 Ratio 未重测） |
| 22 | DeepSeekSparseAttentionDecodeWithKVCacheFwdOp | attention | **0.35%** | ⛔ 封顶 | **~7.5%** | 逐行 gather 占 94.5%；零-gather 地板实测 7.5%，三条减条数路径全被封 | `20261018_104248_500229` |
| 23 | MultiHeadAttentionDecodePagedWithKVCacheFwdOp | attention | **2.39%** | ⛔ 封顶 | **~26%** | 本轮 0.31→2.39（7.6×）；3 档 ideal < 0.3 µs，60% 需亚微秒 | `20261101_184433_143505` |
| 24 | FFTC2COp | spectral | **0.04%** | ⛔ 封顶 | **21.5%**（计数口径） | n≥2048 需 464 KB > 192 KB UB；向量化后 op 膨胀反而更慢 | `20261014_220803_114070` |

## 当前未达标清单（15 个 = 24 − 9 达标；按「有无下一步动作」分组）

### A. 有明确下一步（3 个）

| 算子 | 当前 | 下一步 | 预期 |
|---|---:|---|---:|
| **ArgmaxFwdOp** | 25.83% | 修 narrow-N padding 契约（3 行，op 层按单维/多维拆路径）+ 让 perf_opt 可上线 | **36.71%** |
| **EngramDecodeOp** | 3.46% | 集成多核两阶段版（workspace + 2 kernel，需改 wrapper） | **9.95%** |
| **EngramGateConvBwdOp** | 0.53% | 按 op 级（k1+k2+k3）重算 Ratio | 待测 |

### B. 已充分调优、被结构/编译器硬限封住（5 个）

`GQA-SW-Fwd 16.83%`、`GQA-SW-Varlen 17.53%`、`GQA-Bwd 4.25%`、`FP8LightningIndexer 16.64%`、`DSA-Decode 0.35%`
→ 均已多轮 Stage 4（−19~−30% / 2.9× / 1.85× 等），剩余候选被 UB 容量、flag 预算、`bishengir-compile` 稳定性、gather 形态封死。

### C. 物理封顶（6 个，60% 在数学上不可能）

`RopeNeox 27.35%（上限 39.8%）`、`RopeLongRope 36.34%（49.3%）`、`RopeLlama31 35.91%（49.3%）`、`FFTC2COp 0.04%（21.5%）`、`EngramGateConvFwd 5.57%（15.6%）`、`MHA-Decode-Paged 2.39%（~26%）`
→ 根因：`Ratio 上界 = ideal/(ideal+F)`，这些算子的 `ideal` 只有 0.03–2.4 µs，而 `F`（launch + 每核骨架）≈ 3.5 µs。

### D. 用户已确认不跟进（1 个）

`AlibiFwdOp 56.29%` —— 他机已优化。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-08 | **GQA 三件套最新**：Bwd R21 BN=128 已采纳（8 档 −21%~−29%，几何 −25.22%；推算 Ratio ≈9.85%，官方尚未重测；已超旧 8.2% 参照）。Varlen 官方 report `20261108_013426_279753` 补跑通过（correctness 26/26 + benchmark 10/10，8 档均值 19.87%）。SW 18.52% 不变；R20 诊断确认 R19 的“msprof 盲区”是 `bench.py` 只解析 `OpBasicInfo` 的 bug，short 档模型 `f=8.117 µs/任务、u=2.852 µs/step`，本轮无候选。 |
| 2026-11-06 | **GQA-Bwd Round-20 3-pass 引擎相位拆分转正 = 7.32%**：按 `examples/multi_head_attention/_gqa_bwd_kernel/perf_opt/opt_log.md` §10.3 重建 pass1 Cube-only（S=QKᵀ, dW=dOVᵀ）/ pass2 Vector-only（P=exp2(S·s−lse+pen), dS=P·(dW−delta)·s）/ pass3 Mix（dv/dk/dq，GQA 核内 groups 单写者），三 kernel + GM workspace + 每迭代 barrier；8 例 Ratio 6.75/6.73/7.88/7.84/6.90/6.87/7.86/7.85 → 几何均值 **7.32%**（旧 4.23%，1.73×），官方 report `20261106_203252_067780`；`--level all` 37 PASS、correctness 5/5、gate 4 pass。`n_kv==1` 形状因编译器 SIGSEGV 保留融合回退。**GQA 三件套现为 18.52 / 19.92 / 7.32%**；40% 仍不可达（Bwd 需参照的 4.9×） |
| 2026-11-06 | **GQA-SW Round-19 Tier-0（AUTO_SYNC off / cross_interval）= no_gain**：A 关 `NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC` → 首个 L0 case 即 `L0C read/write conflict`（PIPE_C→PIPE_FIX 的核内 WAR 依赖自动注入，不是冗余 barrier）；B `cross_interval=2/4` → 8b-long −1.6% / −12.3%（细粒度 flag 正是 CV 重叠来源，批量让消费者多等 1–3 step）。**910B3 可表达的方法已穷尽**；cannbot-skills 的 950 FA 最佳实践（L0C 256KB/UB 248KB/CV 直通/4×K-V/4×L0C/14 级）与 `npu-arch` 的 DAV_2201（L0C 128KB/UB 192KB/无直通）共同证明 **40% 是 950 级目标**，910B3 的 18.52/19.92% 是硬件代际上限 |
| 2026-11-06 | **GQA-SlidingWindowFwdOp Round-18 G=2 转正 = 18.52%**：把 Varlen 已验证的 G=2 head-group batching 移植到定长版（H/Hkv=4，2 个 query head 共享 K/V），8 用例官方 Ratio 14.06/14.27/22.74/21.28/14.18/14.80/22.64/24.16 → 均值 **18.52%**（旧 16.83%，~1.10×）；L0/L1/L2/Boundary **32/32**、官方 correctness 22/22 + benchmark 10/10。编译器稳定性 workaround：`T.serial(head_group)`→Python `range` 展开（`T.serial(1)` 会确定性 SIGSEGV）、编译 150s wallclock 超时+16 次重试（防死循环）、官方 report 改逐 case 独立进程（单进程多编译会卡死）。GQA 三件套现为 **18.52 / 19.92 / 4.25%** |
| 2026-11-05 | **GQA-SW-Varlen Round-17 重设计 = 无效（0.920×，+8.7% 回退）**：bm=128/bn=128+G=2 的 UB/flag/编译/L0 全过（user 140,544B、plan ~186–190K；单槽 L_max+1=12）；但 band 字节 0.45× 只换 mte2 **−21%**（copy 次数不变、单块 64→32KB 的 per-copy 粒度损失），单槽任务边界 **+490 µs**、acc_o f16 多 2 vcast → 省 155 µs、付 ~290 µs。**19.92% 不变；40% 的最后一根结构级稻草也试完了**。附带发现：冻结文件单槽回退是未运行死路径且 advance 生产 task_id+1 的 S（task 0 错数据），已在 r17 原型修正，baseline 未动 |
| 2026-11-05 | **GQA-SW-Varlen Round-16 IR workaround = 不可落地**：B1-a 单大 L1（`l1_kv[2*bn,dim]`）+ staging subview **编译 OK、L0 4/4 PASS**（说明 CG-2026-0029 只卡“可喂 L1 分配数”，不卡总字节）；但 B1-b 让 gemm 读第二半 → **out 4/4 错**（gemm 忽略行偏移，独立探针 max_abs=55）；buffer alias 走 `T.Buffer(elem_offset=...)` → SSA 报错；L1→L1 `cbuf→cbuf` unsupported。B2 GM workspace 不减少 GM→L1 字节（mte2-bound）；B3 Q/P 共享与 depth-2 hoist 冲突。**19.92% 不变**；rolling 从“1 个编译器 bug”变成“3 个前端/编译器缺口”（gemm offset / buffer alias / cbuf→cbuf） |
| 2026-11-05 | **GQA-SW-Varlen Round-15 算法级 rolling K/V 复用 = 不可落地**：理想形态（5 tile/环、重叠 4）纸面 **1.40×**（mte2 0.433→0.095），但 L1 放不下（**736KB > 512KB**；字面整窗行环也 544KB）；可行 N=2/3 需 **4–5 块活 L1** → CG-2026-0029 仍在：第 4 块活 L1 的首条 GM→L1 `T.copy` **3/3 SIGSEGV**，同缓冲 `T.clear` 对照 29s 编译通过（单变量判别）。即使编译器修好，可行形态也只有 **1.08–1.15×**。**19.92% 保持不变**；算法级不再是绕过编译器的路 |
| 2026-11-05 | **GQA-SW-Varlen Round-14 G=4 快筛 = 无效（1.043×）**：8b-long fp16 1-run 1769.96→1696.30 µs（−4.2%，<5% 门槛，未进 3-run）。UB/L0C/精度都可解：`acc_o` f16 → plan **192,128B** 编译通过；L0C `[4bm,256]`=256KB → 拆两个 M=2bm gemm；L0 4/4 与 baseline 逐位一致。但 G=2 后四 pipe 和 1.176、mte2 0.427 已非单 pipe 受限，再砍 mte2 ≈0.10 最多换 ≈0.09 wall；G=4 的 gemm/fixpipe issue 翻倍 + 串行 copy→gemm + P 按 head 往返把收益吃光。**结论：G 轴饱和，G=2 是终点；19.92% 保持不变** |
| 2026-11-05 | **GQA-SW-Varlen Round-13 G=2 head-group batching 转正 = 19.92%**：H/Hkv=4，2 个 query head 共享一次 K/V → mte2 0.65→0.43，8 用例官方 benchmark 14.32/14.20/25.02/24.35/15.78/15.53/25.45/24.77 → 均值 **19.92%**（旧 17.53%，1.10–1.15×）；embedded `--level all` **51/51 PASS**。关键前提：实测 UB 分配器**按生命周期复用**（修正 R12“静态和”的过度泛化），当前真实占用 196,608B（含 38.7KB 编译器 staging、零余量）；G=2 用 `ub_f16_D/f32_D → N` 别名省 24.6KB，189.7KB 装下。G=4 被 UB 224.7KB / L0C 256KB 卡住（解锁：acc_o f16 或 ub_diff/pen 别名）。官方 report 的 correctness 因每 shape 重编译，40min 只完成 8/26（全 PASS）；性能数据来自官方 benchmark harness（10/10 PASS） |
| 2026-11-05 | **GQA-SW-Varlen Round-12 三路可行性 = 40% 当前不可达**：① V 共享 UB 可行（171.5KB/192KB，余 20.5KB），但 Cube 端 band-outer pass-2 需要第 4 块活 L1 → 撞 CG-2026-0029；② dim=64 L0A fault 谓词修正为「动态 band 最后一轮满宽」，`bn_eff=128` 可规避（L0 4/4）；③ 双 kernel 拆分 **3.93× baseline**（P 物化 914.9MB）不可行。单管道实测下界 1291 µs（~34%）< 40% 所需 1097 µs；建议停止 Stage 4，保留 2003 µs（对 Stage-3 1.87×）。新增 CG-2026-0030（L0A 越界） |
| 2026-11-05 | **GQA-SW-Varlen Round-11 nqb=2 编译探针 = CONVERGED，但不可晋升**：3 块活 L1（Q 2bm 32KB + P 32KB + K/V 64KB）+ 单 body + 运行时 `slot` 选择 qb0/qb1，dim=128 编译 8.2s（基线 26–42s），无第 4 块 L1 的 SIGSEGV → 修正 R10 “编译器全封”的过强结论（封的是特定结构，不是双缓冲本身）。但探针仅共享 K、V 仍逐 qb 载入、未带 depth-2，1-run 8b-long **2282 µs（+13.9%）**；且 dim=64 且 `kv_len%256==0` 触发 `mte load3d out of range`（L0A 越界，L0 门禁过不了）。正收益需 V 共享 + Vector 按 pair 拆两 pass（在线 softmax 状态 UB 翻倍）→ 下一道门是 UB 预算，不是编译器 |
| 2026-11-04 | **GQA-SW-Varlen Round-10 M1 = no_gain（编译器封堵）**：手写 L1 双缓冲在本 kernel + 本工具链不可表达——(a) 第 4 块「活」L1 的首条 GM→L1 `T.copy` 0/38 SIGSEGV（同缓冲 `T.clear` 合法）；(b) 绕开后 Cube 内层成对展开 >15min 不收敛（基线 26–42s）。唯一可编译的零新增缓冲变体（V 预装提前）8b/70b long 同向 +1.4%。实测四管道和/墙钟 1.52×、mte2 独占 65%，理想化上界 1.52×；登记 **CG-2026-0029** + `TRAP-l1-4th-live-buffer-copy-segv`。结论：long 档 40% 的剩余路径（M1 重叠 / nqb=2）都先要过编译器可行性，不是纯 tiling 问题；下一步应先做 15min 级编译探针 |
| 2026-11-04 | **RopeNeox Round-4 受控实验（4k/8k）= no_gain**：`r4_prefetch`（零额外 UB 的下一 tile 预取）5-run A/B 4k +2.75%（更慢）/ 8k −0.64%（tie）、2k tie；`r4_fuseup`（跨 dtype copy 融合 cast）8k −13.7%；`r4_vcastslice` 编译期失败（v-op 目标必须整 buffer）。根因：pipe 已重叠饱和（prefetch 把 `scalar_vector_stall` 1.21→0.01，却新增 `scalar_mte2_stall` 0.82，aiv_time 反升）；r3 逐字节保留，官方值仍 27.35%。FMA 缺失登记 **CG-2026-0028**（`docs/Tilelang.language/数学操作/` 全目录无 vfma/vmla） |
| 2026-11-04 | **RoPE 系 r3 共享 kernel 采纳同步**：LongRope / Llama31 的 1d `perf_opt` 从 r2_contig（`796840bc`）同步为规范位 r3（`c6098298`；唯一结构变化 = 8k tile 69、4 波→3 波），1d 44/44、2d 46/46 PASS；官方报告 LongRope 36.02%→**36.34%**（8k 6.30→6.08）、Llama31 36.00%→**35.91%**（8k 6.28→6.22，均值落在噪声带）；两 op 潜伏的 `S4-PERF-RECORDS-COVERAGE`（inventory 的 `kernel_id` 指向 baseline 路径）一并修正 |
| 2026-11-02 | **EngramGateConvBwd op 级口径修正**：官方报告只捕获 `k1`（50.8/59.7/225.8 µs），op 级实测 1368/1416/1934 µs（8.6–26.9×）⇒ 真实 Ratio **0.030%**（不是 0.53%）；且 ≈1.4 ms 为与 shape 无关的固定开销，位于 k2/k3。暴露评测框架对多 kernel 算子的系统性低估（batch_norm_bwd、engram mcore 同型） |
| 2026-11-03 | **Argmax 两个 bug 修复 + 调优版上线**：根因 = custom-op 路径编译失败→重试减半 block_m→静默算错；加 `TILEOPS_BYPASS_CUSTOM_OP`（默认开）后 correctness 42/42，官方 25.83%→**36.71%**（1.42×）；lm-head 两档仍被 `syn_instr_mode` 工具链问题阻塞 |
| 2026-11-02 | **EngramDecode 多核两阶段集成完成**：官方 3.46%→**14.64%**（b1 5.98µs/5.21%、b4 13.44/22.22%、b8 9.08/16.51%），correctness 5/5；op 级（stats+apply）≈9.95%。回退开关 `TILEOPS_ENGRAM_MCORE=0` |
| 2026-11-02 | 修 6 处 `from tvm import tir` → `from tilelang.tvm import tir`（解锁整个 reduction 族官方报告）；定位 Argmax 14/42 失败的根因 = narrow-N padding 契约不一致（三向实验证明：op 路径错 1567/2048、kernel 直调 0 错、host 补 `-inf` 到 256 宽后 0 错） |
| 2026-11-02 | 建档：24 个算子；Erf 66.40→68.90、Dropout 58.27→62.00（口径 B）、FP8 9.0→16.64、MHA-Decode-Paged 0.31→2.39、GQA-Bwd 3.8→4.25；验收主口径切换为声明式 roofline |
