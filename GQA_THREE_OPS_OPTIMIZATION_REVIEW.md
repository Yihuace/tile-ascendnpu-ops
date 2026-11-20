# GQA 三算子优化全历程与剩余空间（专家评审材料）

> 生成时间：2026-11-07
> 生成者：`tilelang-op-conductor`（编排层，只做汇总与事实整理，不修改任何算子工件）
> 评审对象：
> - `GroupedQueryAttentionSlidingWindowFwdOp`（kernel `_gqa_sw_fwd_kernel`）
> - `GroupedQueryAttentionSlidingWindowVarlenFwdOp`（kernel `_gqa_sw_fwd_varlen_kernel`）
> - `GroupedQueryAttentionBwdOp`（kernel `_gqa_bwd_kernel`）
>
> 本文全部数字均可在对应 `perf_opt/` 工件中逐条对账（见 §9 证据索引）。**未实测的推断均已显式标注**，不得当作结论引用。

---

## 0. 阅读指引

本文按「先语义、再重设计、再逐轮实验、最后剩余空间」组织，供专家判断“还有没有优化空间”。建议阅读顺序：

1. 想快速判断结论 → 读 §1.4 总览表 + §6 剩余空间矩阵 + §7 下一步实验建议。
2. 想审优化过程是否扎实 → 读 §2.5 / §3.5 / §4.5 的逐轮实验表（每轮都写清了：假设 → 改动 → 实测 → 裁决）。
3. 想审“为什么停在这里” → 读各算子的「已证伪/受阻方向」小节 + §5.2 共性陷阱。
4. 想核数据 → 读 §9 证据索引，直接打开 `opt_log.md` / `perf_records.jsonl` / `profiles/`。

> 术语：`Stage-3` = 迁移/开发阶段的交付 kernel（调优前基线）；`current best` = Stage-4 当前最佳；`final` = 已转正并写入 `perf_opt/{func}.py` 的版本。三者在这三个算子中：`current best == final`。

---

## 1. 公共背景

### 1.1 硬件与工具链

| 项 | 值 | 备注 |
|---|---|---|
| 设备 | Ascend 910B3 ×1 | `cube_core_num=20`、`vector_core_num=40`，额定 1800 MHz，HBM 64 GB |
| 片上容量 | UB 192 KB / 核；L1 512 KB；L0C 128 KB（≈32768 fp32 元素） | 三个算子的 tiling 均由这三条硬上限反解 |
| CANN | 9.0.0 | msprof 25.5.0 |
| tilelang | 0.1.2（各会话 commit 不同：`cebcec9` / `9cfae94c` / `9738284c`） | 版本戳差异见各 opt_log 头部；跨会话数字差异 ≤1% |
| target | `npuir` | Expert 双 Scope（Cube/Vector）为主 |

### 1.2 测量口径（三个算子统一）

- **唯一时延口径**：`msprof op` 的 `Task Duration(us)`（kernel-only），`--launch-count=20 --warm-up=5`，取 **20 次 launch 的 median**。
- **噪声阈值**：3%（小 kernel 单 run 抖动实测 0.5%–3%）。低于 3% 的差异一律判 `no_gain`，不进 A/B。
- **不采集**：NPU event / runner 端到端时间 / 框架小算子污染（varlen 首版就因 host 侧 `GreaterEqual` 污染被判 `partial`，后改自建 `bench.py` 直调 kernel）。
- **bwd 额外口径**：官方 Roofline `Ratio`，定义
  `ratio = counted_flops / (duration × computility)`，`computility = 299.36 TOps/s`（40 cube 单元口径）。
  同机 `torch.matmul(4096³)` 实测 fp16 峰值 **270.1 TFLOPS** ⇒ 理论 Ratio 上限 90.2%。

### 1.3 三个算子的关系

它们属于同一个 attention 家族，但**不是同一份代码的复制**：

```
              同一数学家族：GQA attention
  ┌──────────────────────┬──────────────────────────┬──────────────────────────┐
  │ 定长前向             │ varlen 前向              │ 反向                     │
  │ SW Fwd               │ SW Varlen Fwd            │ Bwd                      │
  │ BSHD                 │ packed NHD + offset      │ BSHD                     │
  │ 两相位 S/P 物化      │ 沿用定长版两相位         │ 单 program 融合 → 3-pass │
  │ 带状滑动窗口         │ 带状 + ragged + OOB      │ 重算 P → dq/dk/dv        │
  └──────────────────────┴──────────────────────────┴──────────────────────────┘
```

- **定长前向**是家族里的“母本”，其 DESIGN 直接复用了同族 MHA prefill（`_gqa_prefill_fwd_kernel.py`）的两相位骨架。
- **varlen 前向**以定长版为母本，新增 ragged 任务映射、运行期长度/offset、常开尾带、packed 寻址。
- **反向**是独立算法（重算式 flash-bwd），结构上与前向不同：前向是 Cube→Vector 物化，反向最初是单 program 融合 → 后来演化成 3-pass 单引擎拆分。

**这一点很重要**：三者的“剩余空间”不能混为一谈——前向的上限是 UB/flag 预算，反向的上限是 Ratio 与参考实现上界。

### 1.4 总览：从 Stage-3 到 final

| 算子 | Stage-3 基线（8b-short fp16） | final（8b-short fp16） | 相对 Stage-3 | 相对上一版 current best | 最终精度 | 最新官方 report |
|---|---:|---:|---:|---:|---|---|
| SW Fwd | 485.13 µs | **317.84 µs** | **−34.5%（1.53×）**；8 档几何均值 ≈1.46× | R18b −9.8%→1.11× | 32/32 PASS + 官方 22/22 | `20261106_105557_262003` **passed**（benchmark 10/10） |
| SW Varlen Fwd | 613.54 µs | **312.60 µs** | **−49.1%（1.96×）**；8 档几何均值 ≈2.06× | R13 −12.4%（1.141× on 8b-long） | 51/51 PASS | 最新 `20261105_132836` **不完整**；最后完整 `20261101_134009` passed（早于 R13） |
| Bwd | 20407.45 µs（launch=10） | **1203.62 µs** | **−94.1%（≈16.95×）**；8 档几何均值 ≈18.1× | R20 −39.7%（1.66× on 8b-short） | `--level all` EXIT=0 | `20261106_203252_067780` **passed**（5/5 正确性，benchmark 8/8） |

> 口径提示（务必注意）：
> - SW Fwd 的“Stage-3”是迁移基线（sha `d442506b…`），不是 GPU 原始实现；GPU→NPU 的迁移收益不计入上表。
> - Varlen 的 `perf_records.jsonl` 当前 `phase=baseline` 指向 **r9_final_sf32**（Stage-3 之后又调过 9 轮），所以 opt_log 的 final 表显示 1.13×；本文额外给出“相对真正 Stage-3”的 2.06×，两者都对，只是参照系不同。
> - Bwd 的 Stage-3 数字用 `launch-count=10` 测量（首轮会话），与后续 20 次中位口径有系统性差异；但差距是 17 倍量级，口径差异不影响结论。

---

## 2. 算子 A：GroupedQueryAttentionSlidingWindowFwdOp

### 2.1 原始语义（做什么）

**数学语义**：BSHD 定长 GQA + 滑动窗口 flash 前向。

```
o = softmax(scale · q @ k^T + M) @ v           # BSHD 布局，online softmax
M = 掩蔽矩阵（不满足下述条件处为 −inf）

valid(q_pos, k_pos) ⟺ (¬is_causal ∨ k_pos ≤ q_pos)          # 因果上界
                     ∧ (wl < 0      ∨ k_pos ≥ q_pos − wl)    # 左窗下界
                     ∧ (wr < 0      ∨ k_pos ≤ q_pos + wr)    # 右窗上界
scale = dim^(−1/2)
is_causal 默认 true；window_size_left/right 默认 −1（= 无限制）
```

**I/O 契约**：

| 张量 | Shape | dtype | 说明 |
|---|---|---|---|
| q | `(B, S, H, D)` | fp16 / bf16 | BSHD 连续 |
| k / v | `(B, S, H_kv, D)` | same | `H % H_kv == 0`（GQA） |
| output | `(B, S, H, D)` | same | Op wrapper 只返回 output |
| lse | `(B, H, S)` | **fp32** | kernel 产出但 wrapper 丢弃；不进入 Op 精度门 |

**边界语义要点**：`−T.infinity` 掩码 + `scores_max` 有限下限 clamp（`−1e38`）防 NaN；`loop_count` 允许 0（整块出窗）；`lse = log2(logsum) + scores_max·scale`。

**源角点**：源码在 `is_causal=True ∧ wr≥0` 时**只施加 causal|left、忽略 right**；manifest 域无该组合，故不承诺（DESIGN §0.1/§9.3 已记）。

### 2.2 GPU 源算法的优化手段（为什么快）

逐项识别了 15 项（DESIGN §0.4 OP1–OP15），处置结论：

| 手段 | 移植结论 |
|---|---|
| flash online softmax（OP1） | **保留**（纯算法层） |
| **滑动窗口带状裁剪（OP2）** | **保留 —— 本算子算法核心** |
| SMEM tiling / coalesced / Tensor Core（OP3–5） | 等价替换为 L1 / UB / Cube |
| `T.Pipelined`（OP6） | **重新设计**为两相位（源 cp.async → NPU 四通道天然重叠） |
| fragment 累加 + 融合 epilogue（OP7） | **重新设计**（L0C 不能被 v 算子操作，必须 GM ws 往返） |
| warp 归约（OP8） | 等价替换为 `T.reduce` + `vmax` |
| exp2 + LOG2E 折叠（OP11） | 等价替换为 e 域 `T.vexp` |
| mask 预填 + initC 融合（OP12） | **重新设计**为向量算术惩罚掩码 |
| warp policy / 3D grid / FAST_MATH（OP9/10/13） | 舍弃或重设计 |

### 2.3 NPU 重设计（关键决策，决定后续所有优化空间）

**总体结构**：一维 persistent Kernel（20 核）× 双 Scope（Cube/Vector）× 两相位独立大循环 × 滑动窗口带状 KV 范围。

```
每个逻辑任务 = (q-block, query-head, batch)
Cube pass-1  : 串行算整条带上的全部 S 块（QK^T，Q-hoist 一次）→ per-core GM workspace
Vector pass-1: 逐块消费 S（vcast → 双侧算术惩罚掩码 → prescale → online max/α/ℓ → P 量化）→ ws_p
Cube pass-2  : 逐块消费 P 算 PV 部分和 → ws_o
Vector pass-2: 逐块延迟 rescale 累加 acc_o
epilogue     : 归一化 + transpose-free lse 写出
```

**五个关键重设计**（也是后续优化的边界来源）：

| 重设计 | 内容 | 为什么这么做 |
|---|---|---|
| **E1 一维 persistent** | `T.Kernel(20, is_npu=True)`，任务号 `cid = task·20 + kernel_id`，clamp 防 ghost task | NPU 无 3D grid |
| **E2 两相位 + GM ws** | S/P/O 三族 ws；per-band-index flag；`TASKDONE` 任务边界 | 源单遍链被两相位以 2.31× 淘汰（MHA 母本实测） |
| **E-band 带状裁剪** | `k_start/k_end` 逐字移植；`bn_eff` 守卫使 `L_max+1 ≤ 16`（flag id 0–15） | 滑动窗口省 FLOPs/访存 |
| **E-mask 算术惩罚** | `pen = clamp(diff−K_up,0,1) ∨ clamp(K_lo−diff,0,1)`；`S += pen·(−1e38)` | 规避 `vcmp` 在 int16/int32 上被标量化（PL-1.11）；探针 7/7 bit-exact |
| **E-bnd 两段式带** | 全有效中段零掩码，仅左右边界块施加惩罚 | 中段块免掩码 |

**UB 账本（设计默认 bm=64 / bn_eff=256 / dim=128 / L_max≤5）**：user alloc ≈154 KB + 编译器 staging 38.7 KB ≈ 192 KB，**零余量**。这就是后续 bm=80 / G=4 全部被 UB 挡住的根源。

### 2.4 baseline 诊断（Round 0）

| workload | Task Duration | aic_cube | aic_mte2 | aiv_vec | aiv_wait |
|---|---:|---:|---:|---:|---:|
| 8b-short fp16 | 485.13 µs | 0.100 | 0.249 | 0.326 | 128.6 µs |
| 8b-short bf16 | 490.24 µs | 0.099 | 0.255 | 0.305 | 124.7 µs |
| 8b-long fp16 | 1303.88 µs | 0.186 | 0.447 | 0.471 | 589.8 µs（45%） |
| 8b-long bf16 | 1299.17 µs | 0.186 | 0.481 | 0.447 | 566.1 µs |
| 70b-short fp16 | 483.47 µs | 0.100 | 0.251 | 0.325 | 128.7 µs |
| 70b-short bf16 | 491.19 µs | 0.099 | 0.255 | 0.304 | 124.7 µs |
| 70b-long fp16 | 1338.74 µs | 0.180 | 0.432 | 0.486 | 618.4 µs |
| 70b-long bf16 | 1336.36 µs | 0.181 | 0.466 | 0.460 | 590.6 µs |

**诊断结论（决定优化排序）**：

1. 两引擎同时 >35% 空转；`aic_cube_ratio` 仅 0.19。
2. **等待分布在任务边界**：Cube flag 等待按 id 分解为 `id0=303us / id1..4=143/123/103/103 / TASKDONE=306us` ⇒ **flag0 + TASKDONE 合计占 Cube 等待 56%**。
3. 不是带宽瓶颈：`GM_to_L1` 带宽占用 13.5%（short）/21.5%（long），L0C→GM 4.8%/7.9%。

⇒ **主因是“任务级流水填充/排空”，不是 FLOPs 或带宽。**

### 2.5 优化历程（逐轮：假设 → 改动 → 实测 → 裁决）

> 数据来源：`perf_opt/opt_log.md` §Iteration Log + `perf_records.jsonl`（37 行）+ `profiles/`。

#### Round 1 — 自动多缓冲（`r1_op1_amb`）

- **假设**：baseline Cube 三管道接近串行（mte2 0.447 + cube 0.186 + fix 0.182），打开 `NPUIR_ENABLE_AUTO_MULTI_BUFFER=True` 可让 BishengIR 双缓冲 K/V。
- **改动**：单行 pass_config。
- **实测**：8b-long fp16 1300.93 vs base 1303.88（**+0.2%**）；`PipeUtilization` 逐项一致（mte2 0.4468→0.4468）。
- **裁决**：`config_no_gain`。**结论：本工具链 Expert 双 Scope + 显式 `T.rs` + 手写 `sync_block_set/wait` 结构下，自动多缓冲是惰性的。**

#### Round 2 — 静态 extent + fp16 `ws_s` 升 f32

- **假设 A**：动态 extent（`T.min/ceildiv`）阻碍 DMA 描述符优化。
- **实测 A**：8b-long 1312.16（**−0.6%**，mte2 0.447→0.440 无变化）⇒ 证伪。每核 ~100 GB/s 是 MTE 发射率，不是描述符开销。
- **假设 B**：参考实现把 fp16 的 `ws_s` 升 f32 省一次 vcast（母本 −6.7%）。
- **实测 B**：short −1.1% / long +1.1%；但 `L0C→GM` 翻倍（18432→30720 KB）⇒ 净中性。
- **裁决**：两者均 `config_no_gain`，不采纳。

#### Round 3 — depth-2 跨任务软件流水（`r3_op1_pipe2`）✅ **采纳**

- **假设**：Cube 等待 56% 在 flag0 + TASKDONE ⇒ 任务级填充/排空。
- **改动（单一主要改动）**：
  1. `ws_*` 增加任务槽维 `n_slots=2`（flag 预算 `2·L_max ≤ 15`，本域 L_max≤5 → 10 个 id）；
  2. flag id = `l + slot·L_max`；
  3. Cube 每轮顺序改为 **[hoist Q(T+1) + pass-1(T+1)] → [pass-2(T)]**；
  4. **删除 `TASKDONE` 全栅栏**（跨任务 WAR 由 slot 交替 + id 窗口闭合）；
  5. 新增专用 `l1_q`；
  6. `2·L_max > 15` 自动回落单槽。
- **实测（8 档全测）**：

| workload | base | r3 | delta | aic_cube | aic_mte2 | aiv_vec |
|---|---:|---:|---:|---:|---:|---:|
| 8b-long fp16 | 1303.88 | 1060.31 | **+18.7%** | 0.228 | 0.570 | 0.603 |
| 8b-long bf16 | 1299.17 | 1040.37 | +19.9% | 0.232 | 0.632 | 0.578 |
| 70b-long fp16 | 1338.74 | 1058.79 | +20.9% | 0.229 | 0.572 | 0.604 |
| 70b-long bf16 | 1336.36 | 1043.69 | +21.9% | 0.232 | 0.633 | 0.577 |
| 8b-short fp16 | 485.13 | 350.92 | **+27.7%** | 0.138 | 0.366 | 0.436 |
| 8b-short bf16 | 490.24 | 343.30 | +30.0% | 0.141 | 0.383 | 0.422 |
| 70b-short fp16 | 483.47 | 353.41 | +26.9% | 0.137 | 0.364 | 0.445 |
| 70b-short bf16 | 491.19 | 346.14 | +29.5% | 0.140 | 0.382 | 0.431 |

- **机制**：cube/mte2/aiv_vec 三个有效忙碌率同时上升，`aiv_wait` 绝对量基本不变（590→640 µs，但分母缩小）⇒ 提升来自“等待被真实工作填满”，不是减少算法工作。short 收益更大，因为 `loop_count≤2`、任务级填充占比更高。
- **裁决**：**采纳，成为 current best**。

#### Round 4 — 块内 `T.Pipelined(num_stages=2)`（`r4_op1_pl_ns2`）

- **假设**：r3 后 Cube 三管道和 ≈1.03 ≈ 100% 时长，`T.Pipelined` 可把 K/V 载入与 gemm/fixpipe 重叠，理论上限 wall≈mte2（−40%）。
- **实测**：8b-long 1060.81 vs 1060.31（**−0.0%**，逐项 ratio 相同）。
- **裁决**：`config_no_gain`。**与 R1 合并结论：本结构下 `auto_multi_buffer` 与 `T.Pipelined` 两个自动重叠旋钮都是惰性的，片内重叠只能手写。**

#### Round 5 — `nqb=2` 共享 K/V（**blocked**）

- **假设**：一个 K/V tile 只服务 64 行 q；把相邻 2 个 q-block 编进同一 task（band 相同），Cube GM 读可 −40%，flag 握手减半。
- **实测**：L0-1/L0-2 PASS；L0-3（8b-short）**600s 无输出（进程 CPU 仅 18s user/16s sys ⇒ 阻塞等待）**，`--level L0` 2400s 超时。
- **裁决**：**blocked —— doubled prim_func 体（~2× 语句数）在本容器 `bishengir-compile` 上不收敛**。不写 records，不采纳。

#### Round 18 — G=2 head-group batching 移植（**未采纳，被编译器阻断**）

- **背景**：varlen 版 R13 已验证 G=2；定长版 `heads/heads_kv = 4`，两个 query head 共享同一 KV head。
- **改动**：任务轴加 `head_group=2`（条件 `groups≥2 ∧ 偶数 ∧ heads 偶数 ∧ dim≥128 ∧ 非 bf16&8b-long`）；ws/ L1/L0C M 维 ×2；fused gemm `M = bm + tmc`；Vector `acc_o=[half, G*dim]`；UB 别名 −24,576 B。
- **实测（8b-long fp16）**：1069.92 → **965.32（1.108×）**；`GM→L1 63248→38912 KB（−38.5%）`；`aic_mte2 0.571→0.367`。A/B：8b-long 1.111×、70b-long 1.107×、8b-short 1.113×。
- **阻断**：
  - `--level all` 的 Boundary `B-s520-w256`（D=64/S=520/wl=256）编译 **rc −11，40 次重试失败**；
  - 官方 report 多 case 同一进程在第 5 例 SIGSEGV 挂起（54 min 未完成）。
- **裁决**：未采纳，`perf_opt/_gqa_sw_fwd_kernel.py` 回滚 baseline sha `33f85319…`。

#### Round 18b — 编译器稳定性修复 + 转正 ✅ **采纳**

- **阻断 1 的真实根因**：`B-s520-w256` 因 `dim=64<128` 本就是 G=1 路径，但 R18 把 G=1 fallback 写成了 `for gh in T.serial(head_group)`（即 `T.serial(1)`）——这个 IR 形态让 `bishengir-compile` 对该 trace **确定性 SIGSEGV**。
- **修复 1**：5 处 `T.serial(head_group)` → **Python `range(head_group)`（trace-time 展开）**。G=1 退化为直线体（Stage-3 验证过的形态），G=2 展开两次。修复后 `B-s520-w256` **23.8 s PASS**。
- **阻断 1b（新失败模式）**：`bishengir-compile` 100% CPU **死循环**（两个孤儿进程卡 80 min / 53 min），而 stock `subprocess.run` 无 timeout。
- **修复 1b**：在 `jit_npu` 模块的 `subprocess.run` 上加 wrapper，仅对 `bishengir-compile` 补 `timeout=150s`；重试判据扩为「SIGSEGV **或** 超时」，`_COMPILE_RETRIES=16`。冷 cache 实测 86.5 s 通过。
- **阻断 2（官方 report）**：单进程下 correctness 第 14 例（wl=0）死锁 88 min；benchmark 第 5 例 `msprof` 600s 超时。**逐 case 独立进程则全部通过**。
- **方案**：保留官方 collector/analyzer/write_reports，只把 pytest 粒度改为「每 test id 一个进程」，再合并 XML/JSON。**无任何数据合成**。
- **结果**：`--level all` **32/32 PASS**（L0 4/4、L1 8/8、L2 12/12、Boundary 8/8）；官方 report `20261106_105557_262003` **passed**，correctness 22/22、benchmark 10/10；8 档 Ratio 均值 **18.52%**。
- **性能零损失**：962.66 → 962.62 µs（差 0.004%）。
- **裁决**：采纳，`perf_opt/_gqa_sw_fwd_kernel.py` sha `6acdc11c…`。

#### Round 19 — Tier-0 探针：关闭自动同步 / 批量同步（当前 best 不变）

- **候选 A**：`NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC=True`
  - 编译成功（51.0 s），但 **L0 第一个 case 就 FAIL**：`error code 507015` / `L0C read/write conflict`。
  - **结论：“关了会错”**——本 kernel 的 `T.sync_block_set/wait` 只覆盖 Cube↔Vector 跨引擎；同核内 `PIPE_C→PIPE_FIX` / `MTE2→V` / `V→MTE3` 依赖编译器自动注入。自动注入是**功能必需**，不是冗余 barrier。
- **候选 B**：cross_interval 批量跨核同步（ci=2 / ci=4）
  - ci=2：982.26 / 983.28 µs（**−1.6% / −1.9%，更慢**）
  - ci=4：1085.72 / 1090.96 µs（**−12.3% / −13.0%，明确回退**）
  - **机制**：Cube 已把整条 S band 提前产好，真正 CV 重叠来自逐 band step 的细粒度握手；合并 2–4 步等于让消费者多等 1–3 个 step，重叠粒度变粗。**“跨核同步次数”不是瓶颈，细粒度 flag 才是重叠来源。**
- **裁决**：保留 baseline；不写 records、不产出 `perf_feedback.md`。

### 2.6 最终方案与数据

**最终候选**：`r18b_g2_headgroup` = `perf_opt/_gqa_sw_fwd_kernel.py`，sha256 `6acdc11cd5a78b8da77ea65ef6fdfbca44a545757b912054c4279af75f752d71`。

**两条有效优化（均结构性）**：
1. **depth-2 跨任务软件流水**（R3）—— 等待被真实工作填满；
2. **G=2 head-group batching**（R18/R18b）—— K/V 的 GM→L1 按 head 对摊薄（−38.5%）。

**逐 workload（同一最终候选，msprof op median of 20）**：

| workload | baseline(µs) | final(µs) | 提升 | 加速 |
|---|---:|---:|---:|---:|
| 8b-short fp16 | 485.13 | 317.84 | −34.5% | 1.53× |
| 8b-short bf16 | 490.24 | 308.76 | −37.0% | 1.59× |
| 8b-long fp16 | 1303.88 | 962.62 | −26.2% | 1.35× |
| 8b-long bf16 | 1299.17 | 1051.60 | −19.1% | 1.24× |
| 70b-short fp16 | 483.47 | 311.72 | −35.5% | 1.55× |
| 70b-short bf16 | 491.19 | 300.20 | −38.9% | 1.64× |
| 70b-long fp16 | 1338.74 | 965.50 | −27.9% | 1.39× |
| 70b-long bf16 | 1336.36 | 918.28 | −31.3% | 1.46× |

**8 档几何均值加速 ≈1.46×**。官方 8 用例 Ratio 均值 **18.52%**（short 14.06%–14.80%，long 21.28%–24.16%）。

### 2.7 已证伪 / 受阻方向（专家请勿重复踩坑）

| 方向 | 结论 | 证据 |
|---|---|---|
| 自动多缓冲 `NPUIR_ENABLE_AUTO_MULTI_BUFFER` | **惰性（零代码差异）** | R1，ratio 逐项相同 |
| `T.Pipelined(num_stages=2)` | **惰性（零代码差异）** | R4 |
| 动态 extent 静态化 | 证伪（不是 MTE2 慢因） | R2 |
| fp16 `ws_s` 升 f32 | 中性（L0C→GM 翻倍） | R2 |
| `nqb=2` K/V 摊薄 | **blocked：doubled body 使 bishengir 不收敛** | R5（600s/2400s 超时） |
| 关闭自动同步注入 | **功能不成立（L0C WAR）** | R19-A |
| cross_interval 批量同步 | **no_gain / 回退**（粒度变粗） | R19-B |
| G=4 头组 | **UB 容量硬限**（需 2 个额外 acc_o） | R5 末/§1.6 |
| `T.serial(1)` 作 G=1 fallback | **确定性 SIGSEGV** | R18/R18b |
| 官方 report 单进程跑多 case | **死锁/超时** | R18b.4 |

### 2.8 剩余优化空间分析（供专家判断）

**先给结论**：该算子当前处于“**结构与参数均已到达本工具链可达上限**”的状态。DESIGN 层天花板**未触发**（未满足双门槛），剩余理论余量估算 ≈1.15×（片内重叠）× 1.15×（K/V 摊薄）= 1.32×，但两条都被硬约束封住。

**未被采纳但仍有理论收益的方向**：

| 方向 | 理论收益 | 封堵点 | 是否值得专家再评估 |
|---|---|---|---|
| `n_slots=3` 更深跨任务流水 | 进一步压缩任务级排空；可能 3–5% | **flag id 预算 16**：`n_slots·L_max ≤ 15`，L_max≤5 ⇒ n_slots 最多 3 但有风险；需重设计 flag 复用 | 中（需先做 flag 预算证明） |
| 手写片内双缓冲（L1 双缓冲 + 动态偏移 gemm 操作数） | 理论 1.15–1.5×（r3 后 Cube 三管道和 ≈1.03× 墙钟） | 自动旋钮惰性；手写需第 4 块活 L1 + 成对展开循环体，**varlen 姊妹已实测该形态编译器不收敛/SIGSEGV** | **高**（但属编译器能力攻坚，不是算法） |
| G=4 头组 | K/V 再摊薄 ~1.15× | UB 192 KB（需 G 个 `[half,dim]` acc_o） | 低（UB 硬限） |
| 关闭自动同步 + 手写全套核内 pipe 同步 | 理论减少冗余 barrier | 需补齐 PIPE_FIX→PIPE_C、MTE2→V、V→MTE3、V→MTE2 全套；成本高、收益未知 | 中（高风险探索） |
| short workload 的固定开销压缩 | short Ratio 14% vs long 22.7%，有明显固定开销 | 每任务 Q-hoist / epilogue / 任务解码；需 profile 定位 | **高**（最可能有实际收益的“低垂果实”） |
| per-shape 分派（smoke bm=32） | 只影响契约测试，不影响 tune 集合 | manifest smoke 已单独处理 | 低 |

**给专家的关键问题**：
1. `n_slots=3` 是否可通过“flag id 分组复用”突破 16 个 id 预算？当前 id = `l + slot·L_max`，是否存在更省的编码？
2. 定长版没有 varlen 的 `ub_diff/ub_pen` 双缓冲压力，是否可像 varlen 一样做 D→N 别名后腾出 G=2 空间？（R18 已用别名，但仍受限）
3. short 档的固定开销具体分布在哪个 pipe？现有 profile 只给了 `OpBasicInfo`（R19 发现 `--aic-metrics=Default` 在本机不落 pipe ratio 列），**需要换 msprof 版本或换 metrics 才能定位**——这是当前最大的“测量盲区”。

---

## 3. 算子 B：GroupedQueryAttentionSlidingWindowVarlenFwdOp

### 3.1 原始语义（做什么）

**数学语义**：packed NHD 变长 GQA + 滑动窗口 flash 前向（FA3 bottom-right 因果对齐）。

```
o = softmax(scale · q @ k^T + M) @ v           # packed NHD，online softmax

per-sample bz: q ∈ [cu_q[bz], cu_q[bz+1])，k/v ∈ [cu_k[bz], cu_k[bz+1])
q_len = cu_q[bz+1]−cu_q[bz]；kv_len = cu_k[bz+1]−cu_k[bz]；offset = kv_len − q_len
q_pos = bx·bm + i（样本内），k_pos = k·bn + j（样本内）

valid(q_pos,k_pos) ⟺ (¬is_causal ∨ k_pos ≤ q_pos + offset)      # 因果上界（FA3 右下对齐）
                   ∧ (wl < 0      ∨ k_pos ≥ q_pos + offset − wl)  # 左窗
                   ∧ (wr < 0      ∨ k_pos ≤ q_pos + offset + wr)  # 右窗
                   ∧ (q_pos < q_len) ∧ (k_pos < kv_len)           # q_oob / k_oob
```

**与定长版的 5 个语义差异**：
1. packed NHD 布局（batch 折叠进 token 轴）；
2. per-sample 运行期长度（`cu_seqlens_q/k` int32）；
3. `offset = kv_len − q_len`（FA3 右下对齐；offset=0 标准 prefill，offset>0 KV-cache）；
4. lse 布局 `(H, total_q)`；
5. ghost q-block（grid 按 `max_seqlen_q` 展开，短样本空转）。

**I/O**：q `(total_q,H,D)`、k/v `(total_k,H_kv,D)`、`cu_seqlens` `(B+1,) int32`、`max_seqlen_q` int32 标量、output `(total_q,H,D)`、lse `(H,total_q) fp32`。

### 3.2 GPU 源算法的优化手段

与定长版同源（OP1–OP15），额外三项：

| 手段 | 处置 |
|---|---|
| OP2b ghost 块空转 | 保留（NPU 承载为 `real_m ≤ 0 ⇒ loop_count=0 + 行截断`) |
| OP16 per-sample 运行期长度 | 保留（承载方式重设计：核内 GM 标量读） |
| lse `(H,total_q)` 布局 | 重新设计（transpose-free 视图） |

### 3.3 NPU 重设计（以定长版为母本）

整体结构**逐语句沿用定长版**（两相位、ws 三族、scales 延迟回放、双 AIV 行半分、Q-hoist、e 域 softmax、size-form 尾钳位）。新增/改写 8 项：

| 编号 | 内容 | 关键证据 |
|---|---|---|
| **V1 ragged 任务映射 + 运行期长度** | 核内 `q_start/kv_start/q_len/kv_len/offset/real_m` | 探针 **P-V1 maxdiff=0.0**（直接 GM 标量读）；备选“copy cu→UB”**BLOCKED** |
| **V2 动态掩码 + offset 平移 + per-row 列 OOB** | `poob = clamp(diff − ((tn_real−1) − i), 0, 1)` | 探针 **P-V2 7/7 bit-exact**；关键陷阱：列 OOB 必须是 **per-row 阈值** `diff ≥ tn_real − i`，写成 `diff ≥ tn_real` 会误掩蔽 |
| **V3 运行期带状范围** | `kv_len/offset` 为运行期值 | 同界同序移植 |
| **V4 常开尾带** | 零初始化 + 全宽掩码 + `zbuf` | 防 `P带(0)×V残留(NaN)=NaN` |
| **V5 运行期 flag 预算** | `L_max_bound = ceildiv(bm+eff_left+eff_right, bn_eff)+2`，`assert L_max_bound+1 ≤ 16` | 替代定长版“编译期扫描 bx” |
| **V6 packed 寻址** | slice-form `T.copy`（禁用 `load_nd2nz`） | CG-2026-0004 |
| **V7 lse `(H,total_q)`** | `[H,total_q,1]` 视图，零 transpose | VP-D6（transpose 毒化 2.6×） |
| **V8 接口** | 保留工厂签名 + 闭包 + lse reshape | wrapper 零改动 |

**UB 账本（G=1 设计默认）**：user `T.alloc_ub` 133,376 B + 编译器 staging 38,656 B = 196,608 B（=192 KB，**零余量**）。这是 varlen 比定长版更难腾挪的根本原因（多两块 `[half,bn]` f32 `ub_diff/ub_pen`）。

### 3.4 baseline 诊断（Stage-3，Round 0）

| workload | Task Duration |
|---|---:|
| 8b-short fp16 / bf16 | 613.54 / 608.97 µs |
| 8b-long fp16 / bf16 | 3759.49 / 3755.83 µs |
| 70b-short fp16 / bf16 | 1167.93 / 1168.24 µs |
| 70b-long fp16 / bf16 | 7514.88 / 7422.79 µs |

**诊断**（70b-long fp16）：`aiv_scalar_wait_time = 4140.8 µs = 55% 壁钟`、`aic_scalar_ratio 0.612`、`aiv_vec_ratio 0.567`。两相位有 pass-1 零等待，但**每个 task 内部** Vector pass-1 必须等 Cube 把整条 S 带写完（单槽 + TASKDONE 屏障）⇒ 瓶颈 = **跨引擎握手串行**，不是算力。

### 3.5 优化历程（逐轮）

#### Round 1 — depth-2 跨任务流水（`pipe2`）✅

- **改动**：Cube 预产生 S(T+1) 到另一槽，Vector 消费 T；flag id = `l + slot·L_max`；独立 `l1_q`；`2·L_max>15` 回落单槽。
- **实测**：8b-long fp16 3759.49→**3237.05（−13.9%）**；70b-long fp16 6434.69（−14.4%）；70b-short 902.45（−22.7%）。
- **合并后 8 档**：−13.9% ~ −24.7%。采纳。

#### Round 2 — 带状掩码链瘦身 + vselect 裁决 ✅（本算子最大单轮收益）

- **现象**：Vector pass-1 的 per-row 列 OOB 惩罚链是 **always-on 的 6 个 `[32,256]` op**（`vsub/vmul/vmax/vmin/vmul/vadd`），而全部 tune workload 的 `kv_len` 都能被 256 整除 ⇒ `tn_real == bn_eff` 恒成立 ⇒ 该链**可证明是 no-op**。
- **改动**：加运行期守卫 `if tn_real < bn_eff:`。
- **实测**：8b-long fp16 3237.05→**2065.79（−36.2%）**；合并后 8 档 −41.6% ~ −46.3%。
- **反向教训（最有价值）**：op2 把 6 个“仅尾块需要”的 op 换成 5 个“每块都跑”的 op，**完全无收益（+0.1%）** ⇒ **op 代价 ∝ 每块无条件执行的 tile 级 op 数**，与“哪些块需要掩码”无关。由此标定 **≈0.61 µs/op·per-AIV**（`[32,256]` f32）。
- **vselect 备选**：编译 8 min 56 s 未终止 ⇒ blocked。
- 采纳。合并后 `aic_scalar_ratio` 升至 **0.928**，Cube 标量管成为新的近饱和管道。

#### Round 3 — 块几何扫描 + S 载体 dtype

| 候选 | config | 实测 | 结论 |
|---|---|---:|---|
| bm=32/bn=512 | 32,512 | 3699.39（**+79%**） | 弃选 |
| bm=80/bn=256 | 80,256 | **UB 溢出 204.8KB > 192KB，bishengir SIGSEGV** | 弃选 |
| **S 载体 f32** | 64,256 | **2003.35（−3.1%）** AB 3/3 同向 | **采纳** |

- **geometric insight**：`(64,256)` 同时是 **UB 上限**（bm=80 溢出）与 **flag 预算上限**（bn=128 时 `L_max=11 ⇒ 2·L_max=22>15`，pipe2 失效）的**唯一交点**。

#### Round 4 — O 载体 / α 瘦身 / bn=128

| 候选 | 实测 | 结论 |
|---|---:|---|
| O 载体 f32 | 2000.09（+0.08%） | no_gain（噪声内） |
| α 递推瘦身 | 2001.41（+0.01%） | no_gain（**证明 tiny op 不计费**） |
| bn=128 | 2528.45（**+26%**） | 弃选（flag 预算回落单槽 + 块数翻倍） |

- 本轮 3 个分支均未超 3% 噪声阈值 ⇒ 参数/op 数方向进入 plateau。

#### Round 5 — f16 softmax 链探针（**blocked**）

- 合法形态穷举：直接 `T.vsub(f16 tile, f32 标量, f16 tile)` 编译失败（`element types should be the same`）；`T.Cast("float16", 阈值)` 合法但**破坏契约域**（f16 整数精确上限 2048，KV-cache 域 `offset` 可 >2048 ⇒ 掩码边界错列 = 精度 bug）。
- **裁决**：blocked（family_no_gain）。

#### Round 10 — M1 手写双缓冲（**单点证伪，最有价值的负面数据**）

- **现象**：Cube 四管道和 `mte2 1291 + mte1 609 + cube 483 + fixpipe 625 = 3008.9 µs = 1.536× 墙钟`（已有 50% 重叠）；mte2 单管道占 65% 墙钟；单缓冲使 V(l+1) 必须等 gemm(l) 释放 `l1_b`。理论上限 1.52×，采纳门槛只取 1.10×。
- **探针阶梯**（同 session、同工具链）：

| 探针 | 改动 | 活 L1 | 结果 |
|---|---|---|---|
| base | — | 3 块 / 112 KB | 26–42 s 通过，0 SIGSEGV |
| A | +`l1_a2` +`l1_b2` + copy | 5 / 208 KB | 420 s，3 SIGSEGV |
| A1 | +`l1_b2` + copy | 4 / 176 KB | 400 s，5 SIGSEGV |
| A2 | +`l1_a2`（从不使用） | 4 / 144 KB | **26 s 通过**（死分配被消除） |
| A3 | +`l1_pad`(4K) + copy | 4 / 116 KB | 72 次 SIGSEGV |
| C1 | +`l1_b2` + **`T.clear`** | 4 / 176 KB | **26 s 通过** |
| OL | C1 + **一条** `T.copy(v…, l1_b2)` | 4 / 176 KB | 600 s，28 SIGSEGV |
| 4L1 | 独立最小 kernel：4 块活 L1 + 2 gemm + 4 copy | 4 | **37 s 通过** |

- **结论**：
  1. **不是 L1 容量**（116 KB 失败 / 176 KB 通过）；
  2. **不是“4 块 L1”通例**（独立最小 kernel 37 s 通过）；
  3. 触发点很窄：**本 kernel 内第一条“目的地址是第 4 块活 L1 缓冲”的 GM→L1 `T.copy`**；
  4. 绕开第 4 块后，**成对展开的循环体本身**又让 compiler 不收敛（>15 min @100% CPU vs 基线 26–42 s）。
- **唯一可编译形态 M1c**：把 V 装载提到 `sync_block_wait` 之前 → 实测 **+1.4%（回退）**；机制是 L1 端口上 MTE2 写与 MTE1 操作数读互拖（PL-1.18）。
- **裁决**：不采纳，current best 保持 r9_final_sf32。

#### Round 13 — G=2 head-group batching ✅ **采纳并转正**

- **现象**（r9 final 8b-long fp16）：`mte2 1320.8 µs (0.667)`、`cube 482.6 (0.244)`、`fixpipe 632.0 (0.319)`、`scalar 1902.5 (0.960)`；`GM_to_L1 = 126,064 KB`，其中 K+V 占 ≈97.6%。
- **前提：UB 必须先瘦身**

| 项 | G=1 | G=2 未瘦身 | G=2 + D→N 别名 |
|---|---:|---:|---:|
| user `T.alloc_ub` | 133,376 B | 151,040 B | 151,040 B |
| 编译器 staging | 38,656 B | 38,656 B | 38,656 B |
| plan total | 196,608 B（0 余量） | **214,272 B（超 17,664 B）** | **189,696 B（余量 3.5%）** |

- **D→N 别名**：`ub_f16_D`/`ub_f32_D` 与 `ub_f16_N`/`ub_f32_N` 生命周期不重叠（D 只在 pass-2+epilogue，N 只在 pass-1）⇒ 声明为同内存别名，−24,576 B，op 数不变，实测中性（−0.1%）。
- **G=2 改动**：`head_group=2`（仅 `groups≥2 ∧ 偶数 ∧ heads 偶数`）；fused gemm `M = bm + tmc`；Vector 运行期 `gh` 循环。
- **实测**：

| 候选 | workload | Task Duration | vs r9 | aic_mte2 | GM→L1 | L0 |
|---|---|---:|---:|---:|---:|---|
| r9_final_sf32 | 8b-long fp16 | 2017.22 | — | 0.667 | 126,064 KB | pass |
| **r13_g2** | 8b-long fp16 | **1767.48** | **1.141×** | **0.433** | **77,280 KB（−38.7%）** | pass |
| r13_g2 | 70b-long fp16 | 3487.04 | 1.145× | — | — | pass |
| r13_g2 | 8b-short fp16 | 312.60 | 1.102× | — | — | pass |
| r13_g2 | 8b-long bf16 | 1798.08 | 1.133× | — | — | pass |

- **机制**：`mte2_time 1320.8→757.3 µs（−42.7%）`；`cube +5.5%`（fused gemm 使 L0B 操作数装载减半）、`fixpipe −9%`、`scalar −11%`。增益归因 = **GM→L1 带宽**，不是算术量变化。
- **转正**：`--level all` **51/51 PASS**（L0 4/4 + contract、L1 26/26、L2 12/12、Boundary 9/9）。
- **裁决**：采纳，`perf_opt/_gqa_sw_fwd_varlen_kernel.py` sha `6befe384…`。

#### Round 11/12（未产生 winner）

- `nqb=2`、dualkernel、bn=128 复现均无有效分支 ⇒ 结论见 `logs/round12/ROUND12_CONCLUSION.md`。

#### Round 5 后的 T-1 参照检查（防 plateau 作弊）

- varlen 同 shape 单位工作量 `final 2002.73 / (定长版 1060.31 × 2) = 0.945` ⇒ **varlen 版按单位工作量比定长版快 5.5%**，结构形态一致，未发现可迁移结构差异。

### 3.6 最终方案与数据

**最终候选**：`r13_g2_headgroup` = `perf_opt/_gqa_sw_fwd_varlen_kernel.py`，sha256 `6befe38405368d894f2b619b62e7ad5f777422364ff3c930fe8131d73a3c0c79`。

**采纳的 3 项**：T1 depth-2 跨任务流水（含删 TASKDONE + 独立 `l1_q`）；T2 列 OOB 链运行期守卫；T3 fp16 路径 f32 S 载体。**外加** R13 的 G=2 head-group + D→N 别名。

**相对上一版 r9（opt_log 口径）**：8 档 −9.3% ~ −12.6%，几何均值 ≈1.13×。

**相对 Stage-3（本文补充口径）**：

| workload | Stage-3(µs) | final(µs) | 提升 | 加速 |
|---|---:|---:|---:|---:|
| 8b-short fp16 | 613.54 | 312.60 | −49.1% | 1.96× |
| 8b-short bf16 | 608.97 | 311.14 | −48.9% | 1.96× |
| 8b-long fp16 | 3759.49 | 1767.48 | −53.0% | 2.13× |
| 8b-long bf16 | 3755.83 | 1798.08 | −52.1% | 2.09× |
| 70b-short fp16 | 1167.93 | 572.08 | −51.0% | 2.04× |
| 70b-short bf16 | 1168.24 | 575.64 | −50.7% | 2.03× |
| 70b-long fp16 | 7514.88 | 3487.04 | −53.6% | 2.16× |
| 70b-long bf16 | 7422.79 | 3533.76 | −52.4% | 2.10× |

**8 档几何均值 ≈2.06×**（相对 Stage-3）。

**精度**：`--level all` **51/51 PASS**；smoke `max|dout|` fp16 1.953e-03（atol 5e-3）、bf16 1.562e-02（tier2 2-ulp 门内）。

### 3.7 已证伪 / 受阻方向

| 方向 | 结论 | 证据 |
|---|---|---|
| 列 OOB 无守卫（每块都跑） | 6 个 always-on op = 36% 墙钟 | R2 |
| 折叠式掩码（op 数不变） | +0.1%（无收益） | R2 op2 |
| `vselect` 备选 | **编译不终止**（8m56s） | R2 op3 / §6 |
| bm=32/bn=512 | +79% | R3 |
| bm=80/bn=256 | **UB 溢出 204.8KB** | R3 |
| bn=128 | +26%（flag 预算回落） | R4 |
| O 载体 f32 / α 瘦身 | 噪声内（tiny op 不计费） | R4 |
| f16 softmax 链 | **编译器 dtype 约束 / 契约域冲突** | R5 |
| 手写双缓冲（M1a/b/d） | **编译器不收敛/SIGSEGV** | R10 |
| M1c 部分重叠 | **−1.4% 回退**（L1 端口互拖） | R10 |
| `nqb=2` | 与 “doubled body” 同墙 | R10 末 |

### 3.8 剩余优化空间分析

**结论**：varlen 当前 `aic_scalar_ratio 0.947–0.962`（Cube 侧标量近饱和），`mte2` 已从 0.667 降到 0.433，`GM→L1` 已 −38.7%。剩余理论空间集中在“**片内重叠**”（四管道和/墙钟 1.507–1.54，理论上限 1.52×），但**被编译器硬失败封死**。

| 方向 | 理论收益 | 封堵点 | 是否值得专家再评估 |
|---|---|---|---|
| 手写片内双缓冲 | **理论 1.5×**（mte2 65% 墙钟） | 第 4 块活 L1 的 GM→L1 写 + 成对展开体 ⇒ compiler SIGSEGV/不收敛 | **最高价值，但属编译器能力攻坚** |
| G=4 头组 | K/V 再摊薄 ~1.1–1.15× | UB（需 G 个 `acc_o`） | 低 |
| bm=80 | 行数摊薄 | UB 204.8KB > 192KB；需再瘦身 ~12KB | 中（若找到另一处别名） |
| 降低掩码链的 `ub_diff/ub_pen` | 腾出 bm/G 空间 | 两个参数是“精确整数 diff + 算术惩罚”恒等性论证的载体，无法删除（折叠只减 op 不减缓冲） | 中（需新掩码机制） |
| short 档固定开销 | short 相对 long 无明显劣势（都 2×），但绝对时间小 | 每任务 Q-hoist/epilogue | 低 |

**给专家的关键问题**：
1. 是否可以把“第 4 块活 L1 的 GM→L1 写”换成“L0C→L1”或“UB→L1”绕开编译器缺陷？探针未覆盖。
2. 成对展开体导致不收敛，是否可用“运行时单循环 + parity 索引”代替 trace-time 展开（但 M1a 已是 parity 方案，仍失败）？
3. 是否存在不需要第 4 块缓冲的重叠形态（例如把 P 与 V 的装载顺序交叉，复用一个缓冲）？

---

## 4. 算子 C：GroupedQueryAttentionBwdOp

### 4.1 原始语义（做什么）

**数学语义**：FlashAttention 风格 GQA 反向主 kernel。给定 `q/k/v/do` 与前序 preprocess 产出的 `lse`、`delta`，重算注意力概率并求三个梯度。

源码逐行读出（**以源码为准，不以 spec 的 math_formula 推测**）：

```
scale    = D^(-1/2) * log2(e)
sm_scale = D^(-1/2)

qkt[i,j] = sum_d k[b, kv0+i, h_kv, d] * q[b, q0+j, h, d]      # = K Q^T
P[i,j]   = exp2( qkt[i,j]*scale - lse[b, h, q0+j] )
P[i,j]  <- 0   若 (kv0+i) > (q0+j)                            # causal
dp[i,j]  = sum_d v[b, kv0+i, h_kv, d] * do[b, q0+j, h, d]     # = V dO^T
ds[i,j]  = P[i,j] * ( dp[i,j] - delta[b, h, q0+j] ) * sm_scale

dv = P^T dO,  dk = ds^T Q,  dq = ds K
```

其中 `i` 为 KV 位置，`j` 为 query 位置。**GQA**：`groups = H / H_kv`，`kv_head = q_head // groups`。

**I/O**：q `(B,S,H,D)`、k/v `(B,S,H_kv,D)`、do `(B,S,H,D)`、lse `(B,H,S) fp32`、delta `(B,H,S) fp32`、dq `(B,S,H,D) fp32 in-place`、dk/dv `(B,S,H_kv,D) fp32 in-place`。

**累加语义**：`dk/dv` 在核内寄存器跨 q-block 累加 + 跨 group head 原子累加；`dq` 核内跨 q-block 累加 + 跨 kv block 原子累加。全部 fp32 ⇒ 原子顺序不确定。

### 4.2 GPU 源算法的优化手段（17 项，逐项处置）

| # | 手段 | 处置 |
|---|---|---|
| 1 | fragment/SMEM 分级 tiling | 部分移植（UB/L1/L0C 重算） |
| 2 | 三维 block grid | **重新设计**为一维 |
| 3 | `T.Pipelined` | 移植（但实测惰性） |
| 4 | `T.gemm` Tensor Core ×5 | 等价移植 |
| 5 | `GemmWarpPolicy.FullRow` | 舍弃（无 warp） |
| 6 | `T.atomic_add` | 能力保留、**结构重设计** |
| 7 | `make_dq_layout` | 舍弃（GPU 专属） |
| 8 | `make_swizzled_layout` | 舍弃（GPU 专属） |
| 9 | lse/delta 落 shared + 列广播 | 等价移植 |
| 10 | `sm_scale` 折进 ds（两 gemm 共享） | 保留 |
| 11 | exp2 + log2 域 lse | 保留 |
| 12 | causal `loop_st` 剪枝 | 保留 |
| 13 | P/ds 降 fp16 喂 Cube | 保留 |
| 14–16 | nvcc flags / fast-math / threads | 舍弃 |
| 17 | `lru_cache` | 保留 |

### 4.3 NPU 重设计（决定后续空间的关键）

**第一版（Stage-3）**：单 program 融合，1-D 非持久 `T.Kernel(nblk)`，每 program = 一个 KV block，`groups` 个 q head 在核内展开，program 内顺序执行 5×gemm → L0C→UB → ~12 v-op → 2 cast → 3×gemm。

**四项重设计**：

| 编号 | 内容 | 语义保持 |
|---|---|---|
| **R1 并行结构** | grid 第 0 维由 **q_head 改为 kv_head**，`groups` 核内静态展开 ⇒ `dk/dv` 跨 head 归约从原子变核内寄存器累加（**零原子、单写者**）；K/V 每核只读一次 | 求和集合不变；`dk/dv` 每元素唯一写 ⇒ **累加顺序更确定**（精度更优） |
| **R2 tile 形状** | `BM=64`、`BN=64(D≤64)/32(D=128)` | L0C 24576 ≤ 32768 元素（75%）；UB ≈104.25 KB |
| **R3 归约结构** | `dq` 用 **rank-2 跨步区域原子**（`dq` 声明 `(B·S,H·D)` 视图 + 闭包）；改用行段整宽原子 | 对外签名不变 |
| **R4 布局注解** | 不使用 `annotate_layout` | 只影响排列，不影响数学 |

**算法调研（R1 候选表）**：
- **#1 选定**：基线 + GQA head 合并（grid 取 kv_head 维）。
- **#2 备选**：两遍拆分（dk/dv 一遍、dq 一遍），零原子但 FLOPs **1.4×**。
- **#3 否决**：q 维网格 ⇒ dk/dv 原子量比 #1 差 `2·BM/BN = 4×`。
- **#4/#5 否决**：零收益或精度变差。
- **#6 否决**：store-attention 需 +6.9 GB（S=2048）。
- **#7 采纳**：causal 剪枝（FLOPs 减半）。
- **#8 采纳**：掩码融合进 exp2（`NEG=-3.0e38`，精确等价）。

**等价性机器验证**：`verify_equiv.py` 10/10 **EQUIV_PASS**（fp16/bf16 × 5 角点，A_vs_B 与 C_vs_B 相对差**恒为 0**）。

**设计期 roofline 估算**（代表 shape `[2,2048,32,128]`，fp16）：

| 项 | 值 | 口径 |
|---|---:|---|
| 流量下界 | 134.2 MB / 2.49 TB/s ≈ **53.9 µs** | `CONST-elu-910B3-bw-floor` |
| 计算项 | 171.8 GFLOP / 112 TFLOP/s ≈ **1.53 ms** | `CONST-cube-gemm-l0c-traffic-floor`（保守，非严格下界） |
| 原子项 | 2.15 GB / 2.49 TB/s ≈ **0.86 ms**（若与 Cube 不重叠则叠加） | — |
| **估算下界** | **≈1.5 ms → 算力受限**（流量项有 28× 裕度） | — |

### 4.4 baseline 诊断

**Stage-3 baseline**（首轮会话，launch=10）：

| workload | Task Duration |
|---|---:|
| 8b-short fp16 / bf16 | 20407.45 / 20447.13 µs |
| 8b-long fp16 / bf16 | 140535.47 / 140647.91 µs |
| 70b-short fp16 / bf16 | 21775.21 / 21760.60 µs |
| 70b-long fp16 / bf16 | 141741.64 / 141789.75 µs |

**诊断**（8b-short bf16）：`aiv_scalar_time / aiv_time = 1192 / 1496 = 80%`，cube 空闲。⇒ **瓶颈是标量流水（scalar issue），不是流量或容量**。8b-short 仅 1.05 TFLOP/s（峰值 ~300 TFLOP/s 量级）。

### 4.5 优化历程（逐轮，**五个真实突破点**）

> 数据来源：`perf_opt/opt_log.md`（第 2 轮会话）+ `history/opt_log_prev.md`（第 1 轮会话）+ `perf_records.jsonl`（27 行）+ `perf_feedback.md`。

#### 第 1 轮会话 Round 1 — 三个分支全部失败

| 分支 | 改动 | 结果 |
|---|---|---|
| `r1_pipe2` | `T.Pipelined(num_stages=2)` 双缓冲 | +0.06%（no_gain） |
| `r1_maskguard` | mask 只在跨对角 tile 做 | **L0 FAIL**（dq 9.92/dk 8.14/dv 2.66，全超限）——条件分支内写 `qkt_ub` 后 auto-sync 未覆盖 ⇒ 静默数值错误 |
| `r1_atomtile` | `T.atomic_add(dq[bz,q0,h,0], dq_ub,[BN,dim])` 整块原子 | **L0 FAIL**（max_diff 1.43e+1）——跨步目标被当连续区域 ⇒ **静默写错** |

#### 第 1 轮会话 Round 2 — int32 `vcmp` → fp32 `vcmp` ✅ **第一个突破**

- **现象**：per-tile 标量成本 = mask（int32 `vcmp`，~50%）+ dq 原子（~50%）。编译器告警只在 int32 操作数时出现。
- **改动**：`rowmat/colmat/dif/lim_mat` 改 `accum_dtype`，`T.vbrc(T.float32(...))`。
- **实测**：20407.45 → **5882.69（−71.2%，3.47×）**；`aiv_scalar_time 1192→130 µs`，`aiv_time 1496→432 µs`。
- **裁决**：采纳。**从此该算子进入“非标量瓶颈”阶段。**

#### 第 1 轮会话 Round 3 — 诊断探针 + 结构尝试

- `r3_fp32vcmp_pipe2`：−0.3%（no_gain）。
- `r3_probe_noatomic`（**故意数值错误，仅定位**）：把 dq 原子换成普通 `T.copy` → 3876.38 µs ⇒ **原子成本 ≈2007 µs = 34%**；探针下 `aiv_mte3 220→7 µs`。**该探针不入 records，只作证据。**
- 结构尝试未实施（BM=128 L0C 超限；BN=64 被静态 UB 加和否决——后来证明该否决是错的，见下）。

#### 第 2 轮会话 Round 1 — BN 32→64 ✅ **第二个突破（推翻旧否决）**

- **假设**：q-tile 加宽减半迭代单元数 ⇒ 减半每迭代固定开销。
- **关键纠正**：旧会话以“静态 UB 加和 240 KB > 192 KB”否决 BN=64；**实测分配器按生命周期复用，编译通过（workspace 155,648 B/block）**。
- **实测**：8b-short 5865.56 → **5786.06（−1.4%）**；8b-long −4.3%。
- **附带诊断**：`mte3` 217.9、`mte2` 139.6 **几乎不动** ⇒ 证实这两项只与**行数**成正比，与每迭代指令数无关；BN 不是它们的杠杆。

#### 第 2 轮会话 Round 2 — pipe2 / atom4h

- `r2_bn64_pipe2`：+0.03%（no_gain）。
- `r2_atom4h`：把 `groups` 个 head 的 dq 合并成一条 2KB 连续区域原子 → **3 种写法全部编译失败**（`Unsupported copy from cc to cbuf` / `shape should be the same`）。blocked。

#### 第 2 轮会话 Round 3 — dq rank-2 跨步区域原子 ✅ **第三个突破（单点 −65%）**

- **机制**：`dq` 是 BSHD 固定 head 的跨步行区（行内 D 连续、行间 stride `H·D`）。源方案逐行 1-D 原子单价 **224 ns/行**；同区域用 rank-2 区域原子只需 **32 ns/行（7×）**。
- **改动**：`dq` 形参声明为 `(B·S, H·D)` 视图（BSHD 连续 fp32 的精确视图）；原子回写变为每 q block 一条 `[BN, dim]` 区域原子；工厂内薄闭包做 `dq.view(rows, cols)`，对外 9 个张量签名不变。
- **实测**：

| pipe（每 block 均值） | baseline | r1_bn64 | **r3_dqregion** |
|---|---:|---:|---:|
| `aiv_time` | 431.3 | 425.0 | **150.3** |
| `aiv_mte3`（dq 原子） | 220.3 | 217.9 | **4.8（45×）** |
| `aiv_mte2`（q/dO 装载） | 141.1 | 139.6 | **18.3（7.6×）** |
| `aiv_scalar` | 129.0 | 101.4 | **106.1（新瓶颈，70%）** |
| `aic_scalar_wait_id2` | 409.1 | 380.2 | **23.6** |

- **端到端**：5865.56 → **1995.76（−65.5%，2.90×）**；8b-long 39864.67 → **14018.17（−64.8%）**。
- **mte2 附带下降 7.6× 的机制**：原子流量消失后 GM 通道不再被挤占（`aiv_scalar_mte2_stall 333→~6 µs/block`）⇒ 此前是“GM 通道被原子流量挤占”，不是装载带宽不够。
- **精度**：`--level all` L0 8/8、L1 12/12、L2 9 + Boundary 全绿；causal/非 causal、D=64/128、groups 1/2/4/8、n_kv=1/4 全部通过。
- **裁决**：采纳。官方 Ratio 1.4610% → **4.2375%（2.90×）**。

#### 第 2 轮会话 Round 20 — 引擎相位拆分 3-pass ✅ **第四个突破（结构级，1.73×）**

- **授权**：结构级重建（多 kernel 串联 / GM workspace）。
- **现象**（base = r3_dqregion，8b-short fp16）：`aiv_time 150.3 µs`，其中 `aiv_scalar 106.1（70%）`；`aic_cube_time 8.6（5.5%）`；每迭代一次 hand-off。
- **结构**：

```
pass 1 `_gqa_bwd_sdw`  Cube only    S = Q K^T, dW = dO V^T          -> fp32 ws
pass 2 `_gqa_bwd_pds`  Vector only  P = exp2(S*scale - lse + pen),
                                    dS = P*(dW - delta)*sm_scale    -> dtype ws
pass 3 `_gqa_bwd_mul`  Mix          dv += P^T dO, dk += dS^T Q,
                                    dq = dS K (+ rank-2 区域原子)
```

- **GQA 适配**（参照是 MHA groups=1）：三 pass 的 grid owner 统一为 `(b, kv_head, kv_block)`，`groups` 个 q head 用编译期 Python `range(groups)` 核内展开 ⇒ `dk/dv` 单写者零原子、K/V 每 pass 只装载一次、workspace slot 仍按 q head 划分。
- **workspace 尺寸**：最大 manifest case（8b-long）`2×1.07 GB + 2×0.54 GB = 3.22 GB`（HBM 64 GB，不沿 KV chunk）。
- **实测**：

| workload | base r3 | 3-pass | delta | 加速 |
|---|---:|---:|---:|---:|
| 8b-short fp16 | 1994.85 | **1203.62** | −39.7% | 1.66× |
| 8b-short bf16 | 1991.91 | 1214.44 | −39.0% | 1.64× |
| 8b-long fp16 | 14019.55 | **7593.34** | −45.8% | 1.85× |
| 8b-long bf16 | 14018.17 | 7649.12 | −45.4% | 1.83× |
| 70b-short fp16 | 1991.86 | 1183.52 | −40.6% | 1.68× |
| 70b-short bf16 | 1994.54 | 1189.58 | −40.4% | 1.68× |
| 70b-long fp16 | 13992.83 | 7613.16 | −45.6% | 1.84× |
| 70b-long bf16 | 13983.87 | 7633.85 | −45.4% | 1.83× |

- **官方 Ratio**：4.2256% → **7.3172%（1.73×）**，达到 6.5–8.2% 目标区间。
- **实现中修掉的两个真实缺陷（如实登记）**：
  1. **P 被 dS 覆盖**：首版把 `vcast(acc, p16)` 放在 dS 算术之后，IR 里两次 vcast 都读 dS，`ws_p == ws_ds` 逐位相同（端到端 dq/dk 仍“看起来对”）。修正为「exp2 → 写 P → 再算 dS」后 P 误差 3.9e-4。
  2. **lse/delta 广播轴**：score tile 是 `[q,kv]`，lse/delta 必须是 `(BN,1)` 列向量；沿用 `[1,BN]` 会在错误轴广播（P 最大 3.2 vs 期望 0.73）。
- **受限项**：`n_kv==1` 走拆分时 `bishengir-compile` SIGSEGV，保留 Stage-3 融合 kernel；BM=128 L0C 超限。
- **精度**：`--level all` EXIT=0 / `All check passed!`。
- **裁决**：采纳，`perf_opt/_gqa_bwd_kernel.py` sha `26edd4c1…`。

### 4.6 最终方案与数据

**最终候选**：`final-r20-3pass` = `perf_opt/_gqa_bwd_kernel.py`，sha `26edd4c1d70e2db1dc77e7d1cfb0f4f78556bb73e5cef6e1b4ab906c098cc627`。

**相对 Stage-3（本文补充口径）**：

| workload | Stage-3(µs) | final(µs) | 提升 | 加速 |
|---|---:|---:|---:|---:|
| 8b-short fp16 | 20407.45 | 1203.62 | −94.1% | 16.95× |
| 8b-short bf16 | 20447.13 | 1214.44 | −94.1% | 16.84× |
| 8b-long fp16 | 140535.47 | 7593.34 | −94.6% | 18.51× |
| 8b-long bf16 | 140647.91 | 7649.12 | −94.6% | 18.39× |
| 70b-short fp16 | 21775.21 | 1183.52 | −94.6% | 18.40× |
| 70b-short bf16 | 21760.60 | 1189.58 | −94.5% | 18.29× |
| 70b-long fp16 | 141741.64 | 7613.16 | −94.6% | 18.62× |
| 70b-long bf16 | 141789.75 | 7633.85 | −94.6% | 18.57× |

**8 档几何均值 ≈18.1×**。

**官方 Ratio 演进（同一目标口径）**：

| 阶段 | Ratio（8 档几何均值） | 相对原始 |
|---|---:|---:|
| Stage-3（`r2_fp32vcmp`） | 1.4610% | 1.00× |
| R3 dq 区域原子 | 4.2375% | 2.90× |
| R20 3-pass | **7.3172%** | **5.01×** |

**与相邻指标的关系**：最终 Ratio 7.32% = 实测峰值 270.1 TFLOPS 的 **2.71%**；用户目标 40% = 持续 119.7 TFLOPS（峰值的 44%），而本仓 attention 族最好水平为 70 TFLOPS（Ratio 23.6%）。**同算子 3-pass 参照在 910B2 上给出 Ratio ≈8.2% 的上界。**

### 4.7 已证伪 / 受阻方向

| 方向 | 结论 | 证据 |
|---|---|---|
| 自动多缓冲 / `T.Pipelined` | 三次实测 <3%（惰性） | S1 R1/R3、S2 R2 |
| mask 条件分支包裹 | **静默数值错误**（auto-sync 未覆盖） | S1 R1 `r1_maskguard` |
| 4-D rank-reduced 整块原子 | **静默写错**（跨步当连续） | S1 R1 `r1_atomtile` |
| `[BN,groups,dim]` 合并原子 | **3 种写法编译失败** | S2 R2 `r2_atom4h` |
| BM=128 | **L0C 超限**（dk_f/dv_f 各 64KB） | S1 R3 评估 |
| 静态 UB 加和作为预算 | **口径错误**（分配器按生命周期复用） | S2 R1 纠正 |
| `n_kv==1` 走 3-pass | **compiler SIGSEGV** | R20 §10.6 |
| 去掉 dq 区域原子（n_kv==1） | 同样 SIGSEGV（崩溃不在原子） | R20 §10.6 |
| KV 轴 chunk | 910B3 64GB 下无必要 | R20 §10.6 |
| store-attention（免重算） | +6.9 GB，流量爆炸 | §1.6.0 R3 |

### 4.8 剩余优化空间分析（**最需要专家判断的算子**）

**结论**：3-pass 已经把 `perf_feedback.md` 指出的“唯一剩余设计层路径”执行完，达到 Ratio 7.32%。同算子参照上界 ≈8.2%，**剩余约 1.12×**。40% 目标在该算法族内**不可达**。

| 方向 | 理论收益 | 封堵点 | 是否值得专家再评估 |
|---|---|---|---|
| **加宽 3-pass 的 tile（BM/BN 128–256）** | 行数 ∝1/BM；3-pass 后每 pass 的片上压力小了，可能装得下更宽 tile | 需逐 pass 重算 L0C/UB；`mul` pass 仍需 dq 原子；**这是最可能拿到剩余 1.12× 的方向** | **高** |
| 减少 workspace 往返 | 当前 S/dW + P/dS 往返 12 B/score-elem | 数学必需（跨引擎）；除非改变 pass 划分 | 中 |
| 合并 pass2+pass3 | 省一次 GM 往返 | pass2 是纯 Vector、pass3 是 Mix，合并会重新引入串行 | 低 |
| `n_kv==1` 拆分 | 覆盖边界形状 | compiler SIGSEGV | 低（边界） |
| 提升 cube 连续忙 | 3-pass 后 sdw cube 1.3%、mul cube 0.9%，仍远未连续忙 | 每 pass 的指令/带宽成本；需要更深流水 | 中 |
| 重新审视 Ratio 口径 | 40% 目标是否等价于“本算子可达”？ | 40% 需 1.7× 本仓 attention 族最好水平 | **需业务侧澄清** |

**给专家的关键问题**：
1. 3-pass 的每个 pass 能否用 `BM/BN=128/256` 重跑一次片上预算？若 `mul` pass 的 dq 原子仍存在，加宽是否会触发同一个 SIGSEGV？
2. 参照上界 8.2%（910B2 / 24 核）是否可比？本机 20 核归一后是否还有 ~1.2× 空间？
3. 40% 目标是否应换算为“相对同族上界的比例”？当前 7.32/8.2 = **89%**，已接近该族可达上界。
4. `perf_feedback.md` 中“建议路由：附录补记”——但随着 R20 采纳 3-pass，该反馈的“结构性估计 5.0×”已被部分兑现（1.73× 在 Stage-4 内），是否需要更新反馈结论？**（本文只做事实整理，不修改该文件。）**

---

## 5. 横向对比与共性经验

### 5.1 三个算子真正有效的优化（可复用）

| 手段 | 适用算子 | 收益 | 机制 |
|---|---|---|---|
| **depth-2 跨任务流水 + 删 TASKDONE + 独立 `l1_q`** | 前向（定长 + varlen） | 定长 −18.7%~−30%；varlen −13.9%~−24.7% | 等待被真实工作填满（flag0 + TASKDONE 占 56%） |
| **运行期守卫剔除可证明 no-op 的链** | varlen 前向 | **−36.2%** | op 代价 ∝ 每块无条件执行的 tile op 数（0.61 µs/op） |
| **G=2 head-group batching + UB 别名** | 前向（定长 + varlen） | 1.11× / 1.14× | K/V 的 GM→L1 按 head 对摊薄（−38.5%） |
| **int32 → fp32 `vcmp`** | 反向 | **3.47×** | 消除 `vcmp` 标量化（80% 标量 → 30%） |
| **dq rank-2 跨步区域原子** | 反向 | **2.90×** | 原子单价 224→32 ns/行（7×）；连带解除 GM 通道挤占 |
| **引擎相位拆分 3-pass + GM ws** | 反向 | **1.73×** | 消除单 program 内 cube/vector 串行与每迭代 hand-off |
| **trace-time `range` 替代 `T.serial(1)`** | 前向 | 让 blocked 变可编译 | 规避 compiler 特定 SIGSEGV |

### 5.2 共性陷阱（三个算子都踩过或验证过）

1. **自动多缓冲 / `T.Pipelined` 在本工具链的 Expert 双 Scope 结构里是惰性的**（前向 R1/R4、反向三次）。不要把它当默认收益。
2. **`vcmp` 在 int16/int32 上被标量化**；必须用 fp32 操作数或改用算术惩罚式。
3. **算术惩罚掩码的 NaN 免疫有前提**：`S + pen·(−1e38)` 清不掉 NaN；必须源头零初始化 stale 列（TRAP-npuir-stale-ub-tail-columns-nan）。
4. **跨步目标不能当连续区域做原子**（`r1_atomtile` 静默写错）。
5. **静态 UB 加和 ≠ UB 预算**（分配器按生命周期复用）。这一条纠正了两个“被误判为不可行”的形态（BN=64、G=2 的 D→N 别名）。
6. **条件分支内写片上缓冲可能漏 auto-sync**（`r1_maskguard` 静默数值错误）。
7. **`bishengir-compile` 是三态**：SIGSEGV（rc −11）/ 无限循环（100% CPU）/ 成功（≈30%）。重试必须带 wallclock 上限。
8. **doubled prim_func 体（>1.7× 语句数）会让 compiler 不收敛**（前向 `nqb=2`、varlen `nqb=2`）。
9. **第 4 块活 L1 缓冲的 GM→L1 写会触发 SIGSEGV**（varlen M1 探针定位）；成对展开的循环体也会不收敛。
10. **transpose epilogue 会毒化整个 kernel（2.6×）** ⇒ lse 用 transpose-free 视图。

### 5.3 工具链缺陷清单（已在本仓实测确认）

| ID | 缺陷 | 触发条件 | 规避 |
|---|---|---|---|
| CG/T-1 | `bishengir-compile` SIGSEGV | `T.serial(1)`、第 4 块活 L1 的 GM→L1 写、`n_kv==1` 3-pass、doubled body | 改 `range()`、改 3 块活 L1、保留融合回退 |
| T-2 | `bishengir-compile` 无限循环 | 多 trace 同进程、部分形态 | `subprocess.run(timeout=150s)` + 重试 |
| T-3 | 官方 report 单进程死锁/超时 | correctness wl=0 例、benchmark 8b-long | 逐 case 独立进程 |
| T-4 | `msprof --aic-metrics=Default` 不落 pipe ratio | CANN 9.0.0 本机 | 换 metrics 或换 msprof 版本（**当前测量盲区**） |
| T-5 | `T.vsub(f16 tile, f32 标量)` 编译失败 | 混合 dtype | 保持 f32 或显式 cast（但注意契约域） |
| T-6 | `vselect` 形态编译不终止 | bool 掩码 + vselect | 改用算术惩罚式 |

### 5.4 性能上界锚点（供专家判断“还有多少空间”）

| 算子 | 当前 | 已知上界/参照 | 剩余比例 | 依据 |
|---|---:|---:|---:|---|
| SW Fwd（short） | Ratio 14.06–14.80% | — | 未知 | 官方 report 8 档 |
| SW Fwd（long） | Ratio 21.28–24.16% | — | 未知 | 官方 report |
| SW Fwd（综合） | 1.46× vs Stage-3 | 理论剩余 ~1.32×（片内重叠 × K/V 摊薄） | 剩余理论上限 ~1.32×，实际被 UB/flag/compiler 封住 | DESIGN §11 + R19 |
| SW Varlen | 2.06× vs Stage-3 | 理论剩余 1.52×（四管道和/墙钟） | 剩余理论上限 ~1.52× | R10 探针阶梯 |
| Bwd | Ratio 7.32% | 同算子参照 **≈8.2%** | 剩余 ~1.12×（7.32→8.2） | `perf_feedback.md` + 910B2 参照 |
| Bwd（用户目标） | 7.32% | 40% | **5.5× 缺口，算法族内不可达** | 需 1.7× 本仓 attention 族最好水平 |

---

## 6. 剩余空间总表（按算子 × 方向）

> 评级：★★★ = 高价值高可行；★★ = 有价值但需突破；★ = 低价值/高风险；“改动层级”指需要动 Stage 1/3/5 还是 Stage 4 内。

| 算子 | 方向 | 预估收益 | 封堵点 | 改动层级 | 评级 |
|---|---|---|---|---|---|
| SW Fwd | short 档固定开销定位 | 未知（short Ratio 14% vs long 22.7%） | **测量盲区**（T-4） | Stage 4 | ★★★ |
| SW Fwd | `n_slots=3` | 3–5% | flag id 预算 16 | Stage 4 | ★★ |
| SW Fwd | 手写片内双缓冲 | 1.15–1.5× | 编译器（4th L1 / 成对体） | Stage 4 + 编译器 | ★★ |
| SW Fwd | G=4 头组 | ~1.15× | UB 192KB | Stage 4 | ★ |
| SW Fwd | 关闭 auto-sync + 手写核内同步 | 未知 | 需全套 pipe 同步 | Stage 4 | ★★ |
| Varlen | 手写片内双缓冲 | **理论 1.52×** | 编译器硬失败 | Stage 4 + 编译器 | ★★★（价值）/ ★（可行） |
| Varlen | bm=80（再瘦身 12KB） | ~5–10% | UB 204.8KB | Stage 4 | ★★ |
| Varlen | G=4 头组 | ~1.1× | UB | Stage 4 | ★ |
| Varlen | 新掩码机制去掉 `ub_diff/ub_pen` | 腾空间 | 需保持 bit-exact | Stage 1/4 | ★★ |
| Varlen | **补跑官方 report** | 非性能 | — | Stage 5 | ★★★（流程债） |
| Bwd | **加宽 3-pass tile** | **剩余 ~1.12×** | 逐 pass 重算 L0C/UB | Stage 4（可能需 DESIGN 复核） | ★★★ |
| Bwd | 减少 workspace 往返 | 未知 | 数学必需 | Stage 1 | ★ |
| Bwd | `n_kv==1` 拆分 | 边界覆盖 | compiler SIGSEGV | Stage 4 | ★ |
| Bwd | 提升 cube 连续忙 | 未知 | 每 pass 指令/带宽 | Stage 1/4 | ★★ |
| Bwd | 40% 目标重定义 | — | 算法族上限 | 业务侧 | ★★★（需澄清） |

---

## 7. 建议的下一步实验清单（按 ROI 排序）

> 以下均为**建议**，尚未执行；执行前需按 conductor 状态机重新 `start_stage(4)` 并设定预算。

1. **【Varlen】补跑 R13 转正后的官方 report**（流程债，非性能）。
   - 当前 `reports/tileops/20261105_132836_…` 只有 `artifacts/` + `pytest/`，无 `run.json`/`report.md`。
   - 用 SW Fwd 已验证的 `report_isolated.py`（逐 case 独立进程）方案重跑，确认 51/51 精度 + 8 档 benchmark。
   - 成本：~30–60 min。风险：低。

2. **【Bwd】3-pass 加宽 tile 的单点探针**（最可能拿到剩余 1.12×）。
   - 分别对 `_gqa_bwd_sdw` / `_gqa_bwd_pds` / `_gqa_bwd_mul` 估算 `BM/BN=128`（或 128/64）的 L0C/UB 预算。
   - 先做**编译可行性探针**（不要求性能），确认不触发 `n_kv==1` 同款 SIGSEGV。
   - 成本：~30 min。风险：中（compiler）。

3. **【SW Fwd】解除测量盲区**（定位 short 档固定开销）。
   - 尝试 `--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory`（bwd 用的口径）或更换 msprof 版本，拿回 `aic_*/aiv_*` ratio 列。
   - 只有拿到 pipe 分解，才能判断 short 档的 14% Ratio 是“发射开销”还是“带宽”。
   - 成本：~20 min。风险：低。

4. **【SW Fwd / Varlen】`n_slots=3` 的 flag 编码可行性探针**。
   - 当前 id = `l + slot·L_max`，`2·L_max ≤ 15`。n_slots=3 需要 `3·L_max ≤ 15` ⇒ L_max ≤ 5 恰好可行（本域 L_max≤5）。
   - 但需确认 `n_slots=3` 的 ws 容量（`ws_*[kernel, 3, ...]`）与 flag 复用是否引入 WAR。先做容量/正确性探针。
   - 成本：~40 min。风险：中。

5. **【Varlen】“无第 4 块 L1”的重叠形态探针**。
   - 目标：不新增缓冲，只重排装载顺序（例如 P/V 交错），看能否拿到部分重叠。
   - 前提：先解决 varlen 的 `ub_diff/ub_pen` 空间（或接受 bm=64）。
   - 成本：~1–2 h。风险：高（compiler）。

6. **【跨算子】把 §5.2 的 10 条陷阱沉淀进 pattern-library**（流程收益）。
   - 特别是“静态 UB 加和 ≠ 预算”“第 4 块 L1 触发 SIGSEGV”“`T.serial(1)` 确定性 SIGSEGV”“bishengir 三态”。
   - 这些是本次三个任务最可复用的经验，能直接减少后续同类任务的试错。

---

## 8. 未验证项与如实披露

1. **SW Fwd 的 short 档固定开销**：现有 profile 的 `aic_*/aiv_*` 列为空（T-4），**未定位**。§2.8 的“最有可能是低垂果实”属推断。
2. **Varlen 的官方 report 缺失**：最后一次完整 report（`20261101_134009`）早于 R13 转正；R13 的性能/精度证据来自 opt_log 的 `--level all` + perf_records，**未经官方 report 复核**。
3. **Bwd 的 Stage-3 基线口径**：`launch-count=10` vs 后续 20；虽差距 17 倍量级，但严格意义上不可直接比。
4. **Bwd 的 8.2% 上界**：来自 910B2 / 24 核 / 不同 tilelang commit 的同算子参照；本机 20 核归一后是否可比，**未实测**。
5. **各算子“理论剩余加速”**（SW 1.32×、Varlen 1.52×）均为**基于管道和/墙钟比值的上界估计**，不是可达收益。
6. **`perf_feedback.md` 的时效性**：其“建议路由：附录补记”与“结构性估计 5.0×”是在 R20 之前写的；R20 采纳 3-pass 后已部分兑现。本文**未修改**该文件，需由流程决定是否更新。
7. **G=4 / bm=80 的收益估算**：基于 K/V 摊薄比例与行数比例外推，**未实测**。
8. **`verify_equiv.py` 的覆盖**：bwd 10/10 EQUIV_PASS 覆盖随机 + 5 角点 × 2 dtype；前向的等价性论证依赖探针（P-V1/P-V2）与 DESIGN 的恒等元证明，不是端到端机器验证。

---

## 9. 证据索引（可直接对账）

### 9.1 算子目录

```
examples/TileOPs/tileops/kernels/attention/
├── grouped_query_attention_sliding_window/
│   └── grouped_query_attention_sliding_window_kernel/
│       ├── DESIGN.md / REVIEW.md / RETROSPECTIVE.md
│       ├── perf_opt/_gqa_sw_fwd_kernel.py          (final, sha 6acdc11c…)
│       ├── perf_opt/opt_log.md                     (§Phase0/Iteration Log/R18/R18b/R19/Final)
│       ├── perf_opt/perf_records.jsonl             (37 行)
│       ├── perf_opt/workload_inventory.json        (10 项)
│       ├── perf_opt/profiles/round18/, round18b/, round19/
│       └── perf_opt/logs/round18/, round18b/, round19/
├── grouped_query_attention_sliding_window_varlen/
│   └── grouped_query_attention_sliding_window_varlen_kernel/
│       ├── DESIGN.md / REVIEW.md / RETROSPECTIVE.md
│       ├── perf_opt/_gqa_sw_fwd_varlen_kernel.py   (final, sha 6befe384…)
│       ├── perf_opt/opt_log.md                     (§5 Rounds 1-13, §7, §8, §10)
│       ├── perf_opt/perf_records.jsonl             (16 行)
│       ├── perf_opt/workload_inventory.json
│       ├── perf_opt/profiles/round10/, round12/, round13/
│       └── perf_opt/logs/round1..round13/
└── grouped_query_attention_bwd/
    └── grouped_query_attention_bwd_kernel/
        ├── DESIGN.md / REVIEW.md / RETROSPECTIVE.md
        ├── perf_opt/_gqa_bwd_kernel.py             (final, sha 26edd4c1…)
        ├── perf_opt/opt_log.md                     (§0.1, §3, §4, §6, §10 R20)
        ├── perf_opt/perf_feedback.md               ([DESIGN_LIMIT], 建议路由=附录补记)
        ├── perf_opt/perf_records.jsonl             (27 行)
        ├── perf_opt/history/opt_log_prev.md        (第 1 轮会话完整历史)
        ├── perf_opt/history/perf_records_prev.jsonl
        └── perf_opt/profiles/{baseline,round20,final}/
```

### 9.2 官方 report

| 算子 | 最新 run_id | 状态 | 备注 |
|---|---|---|---|
| SW Fwd | `20261106_105557_262003_GroupedQueryAttentionSlidingWindowFwdOp` | **passed** | correctness 22/22、benchmark 10/10；逐 case 独立进程 |
| SW Varlen Fwd | `20261105_132836_473563_…` | **不完整**（仅 artifacts+pytest） | 最后完整：`20261101_134009_138243_…`（passed，早于 R13） |
| Bwd | `20261106_203252_067780_GroupedQueryAttentionBwdOp` | **passed** | correctness 5/5、benchmark 8/8 |

### 9.3 关键文件与行号（便于专家抽查）

| 内容 | 文件 | 位置 |
|---|---|---|
| SW Fwd 语义与源算子解读 | `.../grouped_query_attention_sliding_window_kernel/DESIGN.md` | §0.1–0.4 |
| SW Fwd NPU 重设计 | 同上 | §0.6 |
| SW Fwd 逐轮实验 | `.../perf_opt/opt_log.md` | §Iteration Log、§Round 18、§Round 18b、§Round 19 |
| SW Fwd 最终数据 | 同上 | §Final Summary / Final Performance Test Data |
| Varlen 语义与 varlen 差异 | `.../grouped_query_attention_sliding_window_varlen_kernel/DESIGN.md` | §0.1、§0.5–0.6 |
| Varlen 逐轮实验 | `.../perf_opt/opt_log.md` | §5（Round 1–13）、§10 |
| Varlen 最终数据 | 同上 | §8 Final Summary |
| Bwd 语义与源算子解读 | `.../grouped_query_attention_bwd_kernel/DESIGN.md` | §0.1–0.4 |
| Bwd NPU 重设计 | 同上 | §0.6 R1–R4 |
| Bwd 算法调研 + 等价验证 | 同上 | §1.6.0、§1.6.1 |
| Bwd 逐轮实验（第 2 轮会话） | `.../perf_opt/opt_log.md` | §0.1、§3、§4、§6、§10 |
| Bwd 第 1 轮会话 | `.../perf_opt/history/opt_log_prev.md` | §2–§6 |
| Bwd 设计层反馈 | `.../perf_opt/perf_feedback.md` | 全文 |

---

## 10. 一句话总结（给专家）

- **定长前向**：已经用“depth-2 跨任务流水 + G=2 头组”把结构化收益吃到 1.46×，剩余受 UB/flag/编译器三重硬限，最大的未解是 **short 档固定开销的测量盲区**。
- **varlen 前向**：已经是三者中“单位工作量最接近理论”的（相对 Stage-3 2.06×），剩余理论空间在片内重叠（~1.52×），但被 **编译器对第 4 块 L1 / 成对展开体的硬失败** 封死——这是编译器能力问题，不是算法问题。
- **反向**：单点收益最猛（int32→fp32 vcmp 3.47×、dq 区域原子 2.90×、3-pass 1.73×），当前 Ratio 7.32% 已达同族参照上界（≈8.2%）的 **89%**；剩余约 1.12× 的可行方向是 **3-pass 加宽 tile**。
- **三者共同结论**：自动多缓冲/`T.Pipelined` 在本工具链惰性；静态 UB 加和不是预算；`vcmp` 必须用 fp32；跨步目标不能当连续区域；`bishengir-compile` 有三态失败。这五条是本次最有复用价值的经验。

---

## 11. 2026-11-08 执行更新（本轮实际落地结果）

> 本节是文档 §1–§10 之后的增量，不改变前文的历史数据。三算子均按 `scenario=optimize` 重开 Stage 4，`phase=DONE`，`gate 4` 通过。

### 11.1 Bwd：Round 21 BN=128 已采纳（最大增量）

- **新 best**：`perf_opt/_gqa_bwd_kernel.py` sha256 `0ff4a4f3c9d2abe260892d6a72053e3fb2a6932b4de6bb7174bbf6234542c20f`（R20 备份：`history/_gqa_bwd_kernel_r20_26edd4c1.py`）。
- **结构**：capacity-partitioned widening —— `sdw`/`mul` 用 `BN=128`，纯 Vector 的 `pds` 保持 64 行子块并在 slot 内 `n_sub=block_n//64` 展开；BM 仍 64。
- **8 档 final vs R20**：−21.4% ~ −29.1%，几何均值 **−25.22%**；8/8 无回退。
- **精度**：候选与转正文件各一次 `--level all`，均 EXIT=0；L0 8/8、L1 12/12、L2 12/12、Boundary 8/8（**注：opt_log 有一处“37 PASS”的计数笔误，分层之和为 40，以 EXIT=0 与分层结果为准，已在本节披露**）。
- **未验证**：官方口径 Ratio / 官方 report 未重测（R21 预计高于 R20 的 7.32%，但未测）；6/8 档为单 run（与 R20 final 同口径）。
- **修正前文结论**：§4.8 的“可行轴是 BN 不是 BM”被实测证实；§5.4 的 Bwd 上界 ≈8.2% 仍成立（7.32% 是 R20，R21 待测）。

### 11.2 SW Fwd：测量盲区是解析 bug，short 档模型定量

- **“盲区”根因**：R19 的 raw 目录里本就有完整 `PipeUtilization_*.csv`；`bench.py` 只 walk `OpBasicInfo*.csv`。改用 `--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline` 后同机可解析（与 Bwd 一致）。
- **short 档模型**：`墙钟 = f·N_task + u·N_step`，`f≈8.117 µs/任务`、`u≈2.852 µs/band-step`；short 的 L≈1.5、long 的 L≈3.75 ⇒ short 的 1.65× 慢**全部**来自 f÷L。每 band-step 真实工作在 short/long 相同（≤2%）。
- **本轮无候选采纳**，best 仍 `6acdc11c…`；`perf_records.jsonl` 未改。
- **下一步**：`n_slots=3`（L_max≤5 时 `3×5=15` 恰好可容，零余量；风险是 prim_func 体量），只值 1 轮；先只编 8b-short 一个 shape。

### 11.3 Varlen：官方 report 闭环

- **report run**：`examples/TileOPs/reports/tileops/20261108_013426_279753_GroupedQueryAttentionSlidingWindowVarlenFwdOp/`，`status=passed`，correctness **26/26**、benchmark **10/10**。
- **8 档 Ratio**：8b-short 14.05/14.33、8b-long 24.73/24.58、70b-short 15.50/15.54、70b-long 25.15/25.05，与 R13 一致。
- **工具链修复**：`subprocess.run(timeout=)` 只杀直接子进程，会留下 `bishengir-compile` 孤儿（本轮 case 8 卡 26 min 并污染后续）；已改为 `Popen(start_new_session=True)` + `os.killpg` + 逐 case 重试 + `--resume` 断点续跑。脚本：`perf_opt/logs/round13_report/report_isolated_varlen.py`。
- **修正前文结论**：§3.6 的“官方 report 缺失”已闭环；§3.8 的“补跑 report”已完成。

### 11.4 自进化

- `EVOLVE_COMPLETED`：11 个候选（D:3 / P:4 / R:2 / C:2）；Tier 0 更新 2 个 CASE + INDEX 2 行；Tier 1 入队 7 条；Tier 2 新增 1 条（采纳路径预算）。
- 最高价值两条：**L0C 100% 等号可用**（更正 capacity 先验）、**官方 report 逐 case 隔离形态**（一 id 一进程 + killpg + retry + resume）。
- 待审 Tier 2 合计 **105 条**，2026-09-07 批将于 2026-11-21 触及 75 天提醒线——需安排一次审批简报。

### 11.5 修正后的优先级

| 优先级 | 动作 | 状态 |
|---|---|---|
| P0 Bwd BN 加宽 | **已完成并采纳**（−25.22% 几何均值） | ✅ |
| P1 SW 测量修复 | **已完成**（解析 bug + short 模型；无候选） | ✅ |
| P2 Varlen report | **已完成**（26/26 + 10/10 passed） | ✅ |
| P3 Bwd `pds` BN=128 | 下一个结构杠杆（pds 占总时长 35–40%，缺口 36.5 KB；省 `neg_buf` 兼作 mask 限值 + `vexp2` scratch 复用） | 待启动 |
| P4 SW `n_slots=3` | 只值 1 轮；先只编 8b-short，bishengir 重试 >6 次即放弃 | 待启动 |

