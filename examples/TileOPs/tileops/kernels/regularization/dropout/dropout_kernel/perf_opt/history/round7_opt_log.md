# Stage 4 调优日志 — DropoutOp / `_make_dropout_kernel`（R6：Ratio 58.84% → 58.27%，时延 −20.7%~−21.4%）

- 任务: `TileOPs-_make_dropout_kernel-20261101T145102Z`（scenario=optimize, mode=full）
- 基准 kernel（只读）: `examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel.py`
- 调优产物: `perf_opt/_make_dropout_kernel.py`（sha256 `c3f71a95854ca1711d7ab8f34908f5e4a8a54d19dc406b0566ff2a9731962572`）
- 硬件/工具链: Ascend910B3（20 AI core × 2 = 40 Vector core，UB 192 KB）/ CANN 9.0.0 /
  `tilelang 0.1.2+d58388995fc943c5d79035e2a312116ea6d086cd` / torch_npu 2.7.1 / 2026-11-01
  （**版本戳变更**：上一轮为 `0.1.2+90fb7ae37f…`；按 E-5，本轮所有性能结论均在新工具链上**重新实测**，
  基线也重测，未复用上一轮任何时延数字。）
- 性能口径（唯一）: `msprof op` → `OpBasicInfo.csv::Task Duration(us)`，`--kernel-name=main --launch-count=20 --warm-up=5`，串行采集
- 上一轮产物（48 条记录 / 5 轮 / perf_feedback）原样归档在 `perf_opt/history/`

## 0. Workload Inventory（本轮不变，来源同 history）

| workload_id | shape | dtype | N | kind | 本轮处理 |
|---|---|---|---|---|---|
| `smoke-dropout-1m-float32` / `-float16` / `-bfloat16` | [1024,1024] | fp32/fp16/bf16 | 1,048,576 | smoke | 仅精度回归（L0/L1 覆盖），不采集性能 |
| `tokens-1k-hidden-4k-fp16-float16` | [1024,4096] | fp16 | 4,194,304 | **tune** | baseline + 调优 + final |
| `tokens-1k-hidden-10k-bf16-bfloat16` | [1024,10240] | bf16 | 10,485,760 | **tune** | baseline + 调优 + final |
| `tokens-1k-hidden-4k-fp32-float32` | [1024,4096] | fp32 | 4,194,304 | **tune** | baseline + 调优 + final |

单一 `kernel_id`（全 6 个案例调用同一工厂 `_make_dropout_kernel(N,dtype,p,seed)`），`target_kernel_name = main`。
完整分类见 `workload_inventory.json`；逐测量记录见 `perf_records.jsonl`（本轮 9 行：baseline 3 / candidate 3 / final 3）。

## 1. Baseline（round 0，当前 8192 配置重测）

| kernel_id | workload_id | TaskDur(us) | aiv_vec_ratio | aiv_mte2_ratio | aiv_mte3_ratio | L0 | 有效 R+W |
|---|---|---:|---:|---:|---:|---|---:|
| `…perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main` | fp16 4k | 27.898 | 0.9605 | 0.1825 | 0.0867 | pass | 0.601 TB/s |
| 〃 | bf16 10k | 73.929 | 0.9671 | 0.1702 | 0.0820 | pass | 0.567 TB/s |
| 〃 | fp32 4k | 28.735 | 0.9602 | 0.2243 | 0.1371 | pass | 1.177 TB/s |

与官方报告 `20261101_223520_519180_DropoutOp`（28.02 / 73.82 / 28.44 µs）一致（≤0.5% 内，fp32 略高 1.0%）。
**现象**：`aiv_vec_ratio 0.96`、`mte2_ratio 0.17–0.22`、`mte3_ratio 0.08–0.14` ⇒ 仍为纯 Vector 发射受限；
有效 R+W 0.57–1.18 TB/s vs 同机逐元素天花板 2.49–2.67 TB/s ⇒ 唯一有效杠杆是**再减向量 op 数**。

## 2. Iteration Log — Round 6（本轮唯一一轮）

### 2.0 本轮先做的三项量化（对应调度要求 ②③）

**(a) 波数账（推翻「尾波 ≈6% 浪费」的估计）**：`n_num = ceil(N/bs)`，bs=8192 时
- W1/W3（N=4,194,304）：**512 tile / 40 核 = 12.8 波** → 关键路径 13 tile，效率 `512/520 = 98.46%`，
  尾波损失 **1.54%**（不是 6%：最后一波 32/40 核忙，浪费仅 8 个 tile / 520）。
- W2（N=10,485,760）：**1280 tile = 32 波整除，效率 100%**。
- 可行域内没有更好的 bs：使 `n_num` 更整的 bs 需要 `N/bs = 40k`（bs≈8066/7936），既破坏 32B 对齐也超出
  UB 余量（8192 已是 cap，10240 `ub overflow, requires 1802240 bits`）。⇒ **波数不是杠杆（≤1.5%）**，
  且 W2 已 100%，无法解释三档共同的 −20% 需求。

**(b) PRNG 链的 op/µs 账（本轮基线 13 op 的 fp16 分解）**：索引 2（`fill`+`vadd`）+ hash 4
（`vmul/vshl/vxor/vmul`）+ 掩码 4（`vabs`+`vsub`+`vclamp`+`vcast`）+ 掩码转 storage 1（`vcast`）+ 数据 2
（`vmul`×2）。**int32 域的 op 占 10/13**，而 910B3 AIV 上 int32 是 64 elem/instr、fp16 是 128 elem/instr
⇒ int32 段约占向量时间 87%。因此「减 1 个 int32 op」≈ 8–9% 时延，是唯一有量级的杠杆。

**(c) hash 链再瘦身的统计检验（主机 int32 模型，`probes/hash_variants_r6.py`）**：

| 变体 | op | rate(4Mi,3σ) | block-χ²(dof 4095) | lag1–4 自相关 | 判定 |
|---|---:|---|---|---|---|
| V0 现 5-op `(cA ^ cA<<13)·B`，阈值 `|h|` | 4+3 | ok | 4110 | ≈0 | 参照 |
| **V1 同 hash，阈值改符号位** | 4+1 | **0.500130 ok** | **4129** | ≈0 | **采纳**（周期独立 0.4995–0.5003、seed 分离 0.4997） |
| V2 `(cA) ^ (cA<<13)`，阈值符号位 | 3+1 | ok | **87**（远低于 dof） | **−0.389** | 否决：掩码反相关 |
| V3 `c ^ (c<<13)` 再 `·B`，阈值符号位 | 3+1 | ok | 2578 | **+0.723** | 否决：强正自相关（格点） |

⇒ PRNG 段已到底：**5-op hash 不可再减**；可减的只有掩码段的比较/规约 op。

**(d) 掩码段的关键洞察（本轮采纳的优化点）**：p=0.5 时阈值恰为 `|h| >= 2^30`，即「符号位清零」；
而 `P(h >= 0) = 0.5` **精确成立**（hash 输出在 int32 上均匀）。于是 p=0.5 可退化为
`m = clamp(h, 0, 1)` 一个 op，取代 `vabs` + `vsub` + `vclamp` 三个 op（-2 int32 op，-8 B/elem UB）。
对 p≠0.5 无整数溢出安全的等价形式（`h - T` 会回绕），故按**编译期常量 `p == 0.5`** 选择路径（同
`native_domain` 形态，不是 shape 分派、不在 wrapper）。

另有 1 op 的候选被工具链否决（记录在案）：`i32→i16 → vshl 14 → vbitcast(fp16) = {0,2.0}` 融合
「掩码 cast + scale 乘」的方案（`_r6_bitcast.py`）编译失败：`hivm.hir.vmul op requires the same element
type for all operands`（`vbitcast` 不改变 buffer 绑定的 dtype，后端拒绝 f16×i16 的 vmul）⇒ 弃选。

### 2.1 候选 vs current best（B2 结构化对比表）

| candidate vs current best | workload | bs | TaskDur(us) | vec_ratio | mte2_ratio | mte3_ratio | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| current best = R5 winner（8192） | fp16 4k | 8192 | 27.898 | 0.9605 | 0.1825 | 0.0867 | pass | — |
| **`r6_signed`**（p=0.5 符号位阈值，-2 op） | fp16 4k | 8192 | **22.116** | 0.9528 | 0.2504 | — | pass | **winner（−20.7%）** |
| current best = R5 winner | bf16 10k | 8192 | 73.929 | 0.9671 | 0.1702 | 0.0820 | pass | — |
| **`r6_signed`** | bf16 10k | 8192 | **58.118** | 0.9737 | 0.2114 | — | pass | **winner（−21.4%）** |
| current best = R5 winner | fp32 4k | 8192 | 28.735 | 0.9602 | 0.2243 | 0.1371 | pass | — |
| **`r6_signed`** | fp32 4k | 8192 | **22.716** | 0.9547 | 0.3629 | — | pass | **winner（−20.9%）** |
| `r6_bitcast`（+vbitcast 融合） | — | — | — | — | — | — | — | **blocked（编译失败，见 2.0(d)）** |
| hash 3-op 变体 V2/V3 | — | — | — | — | — | — | 主机模型统计不合格 | **blocked（质量不符）** |

- 单点性：`r6_signed` 相对 current best 只改「p=0.5 时的掩码阈值形态」，其余（hash、seedfold、UB diet、
  fp16 存储域、8192 tile）逐字不变。
- **无需合并轮**：本轮唯一有效分支即 current best 的直接单点升级，三 workload 全部 −20.7%~−21.4%，
  无回退 ⇒ 直接作为 `perf_opt/_make_dropout_kernel.py`。
- 精度：`_r6_signed.py --level L0` 22 例 PASS（`logs/r6/L0_r6_signed.log`）；最终产物
  `--level all` → L0 22 / L1 12 / L2 14/14 / Boundary 9/9 全 PASS（`logs/r6/L0_L1_final_all.log`）。
  p≠0.5 路径（L0-2 p=0.1/0.3、L1-1 p=0.3/0.7）保持原 `|h|` 阈值，rate 实测 0.0999/0.2999/0.6998。
- smoke 精度：3 个 smoke 案例（1M × fp32/fp16/bf16）在 `--level all` 中由 L0-1 覆盖（rate 判据用**常量非零
  输入 ones**，遵守 `TRAP-dropout-rate-metric-random-input`），全部通过；smoke 未采集性能、未参与 winner。

### 2.2 op 数账（本轮）

| 阶段 | fp16 | bf16 | fp32 | 组成 |
|---|---:|---:|---:|---|
| R5 winner | 13 | 14 | 12 | 索引 2 + hash 4 + 掩码 4 + 掩码cast 1 + 数据 2 (+bf16 往返 2) |
| **R6 final** | **11** | **12** | **10** | 掩码 4 → **1**（`vclamp(h,0,1)`；p≠0.5 仍为 4） |

有效 R+W：W1 0.601→**0.759** TB/s、W2 0.567→**0.722** TB/s、W3 1.177→**1.477** TB/s。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp16-float16 | 27.898 | 22.116 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-10k-bf16-bfloat16 | 73.929 | 58.118 | final |
| examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/perf_opt/_make_dropout_kernel.py::_make_dropout_kernel::main | tokens-1k-hidden-4k-fp32-float32 | 28.735 | 22.716 | final |

- 数值逐项来自 `perf_records.jsonl` 中 `phase=baseline` / `phase=final` 的 3+3 行（同一最终候选
  `artifact_sha256=c3f71a95…`，raw profile 在 `profiles/final/<workload_id>/{baseline,final}/OPPROF_*`）。
- **Ratio（口径订正）**：官方报告 `20261101_235107_195598_DropoutOp`（`--kernel-name main`）实测，
  roofline `GM/L2` + `GM Read + Write`；本轮 final 三档与均值为：

| workload | baseline_us | final_us | baseline Ratio（13-op，旧 ideal） | final Ratio（11-op，官方实测） | 相对 60% 目标 |
|---|---:|---:|---:|---:|---|
| fp16 4k | 27.898 | 22.116 | 56.26% | **54.47%** | 未达（−5.53 点） |
| bf16 10k | 73.929 | 58.118 | 62.01% | **63.02%** | 达标 |
| fp32 4k | 28.735 | 22.716 | 57.79% | **57.31%** | 未达（−2.69 点） |
| **均值** | — | — | **58.69%**（官方 58.84%） | **58.27%** | 未达（−1.73 点） |

**口径说明（重要）**：上一版 opt_log 的 final Ratio 70.97 / 78.88 / 73.10（均值 74.32%）是**旧 ideal** 算出的，
不可再用。本轮把向量 op 从 13 降到 11，`msprof` 实际计数的 op 数同步下降，**roofline 的 ideal 也随之下调**：
- 旧（13 op，8192）：fp16 28.02 µs / 56.02% ⇒ ideal ≈ 15.70 µs；
- 新（11 op）：fp16 21.96 µs / 54.47% ⇒ ideal ≈ **11.96 µs**。

因此本轮真实时延下降 **20.7%–21.4%**，但 Ratio 基本持平（**58.84% → 58.27%**，−0.57 点）。
这是该指标的口径特性（分子 ideal 与分母 duration 同源于同一 kernel 的 op 计数，减 op 会同时压低 ideal），
**不是性能回退**。

**与更早 baseline 的对照**：对比未上线 8192 tile 时的原始 baseline **41.21% → 58.27%**，仍是真实提升（+17.06 点）。
另外，**11-op 版比 13-op 版真实快 21%，但 Ratio 略低 0.57 点；不采用 13-op 版来刷 Ratio**——Ratio 的绝对高低
不能跨 op 数版本直接比较，优化决策始终以 `msprof op` 的 Task Duration(us) 为准。

## 4. Skill Retrospective

| vp_type | 结论 | 证据三件套 |
|---|---|---|
| **D** | **910B3 int32 域 op 是 dropout 类标量-PRNG 算子的主成本**：13 op 中 10 个是 int32（64 elem/instr）≈ 向量时间 87%；把掩码段 3→1 个 int32 op 换来 **−20.7~−21.4%**（三 workload 同向），远非线性于 op 数（11/13=0.85）——链缩短也缩短关键路径 | evidence: 本文 §2.1/§2.2 + `perf_records.jsonl` r6 行；repro: `probes/hash_variants_r6.py`（统计侧）+ `_r6_signed.py` vs `_x_final.py`（代码 diff）；toolchain: tilelang 0.1.2+d5838899 / CANN 9.0.0 / 910B3 / 2026-11-01 |
| **P** | 优化点：**p 恰为 0.5 时用「hash 符号位」代替「|h| ≥ T」阈值**（`vclamp(h,0,1)` 取代 `vabs`+`vsub`+`vclamp`）。见解：hash 在 int32 上均匀 ⇒ `P(h≥0)=0.5` 精确，符号位就是一个无偏阈值；p≠0.5 因 `h−T` 会在 int32 端回绕，必须保留 `|h|` 路径（编译期 `p==0.5` 二选一）。语义无损（掩码仍是 `(seed,idx)` 纯函数、rate 精确） | evidence: §2.1 三 workload −20.7/−21.4/−20.9%，L0/L1 全过；repro: `_r6_signed.py`（before=`_x_final.py` 的 4-op 掩码段）；toolchain: 同上 |
| **D** | 反向：**3-op counter hash 在 910B3/dropout 场景不可用**——去掉末次乘法后 bit31 呈强相关（lag1 自相关 −0.389 / +0.723，block-χ² 87 / 2578 vs dof 4095），统计质量不达标 ⇒ 4-op hash（mul,shl,xor,mul）是该族的实测下界 | evidence: `probes/hash_variants_r6.py` 输出；repro: 同脚本；toolchain: numpy int32 模型 + 同上 |
| **R** | 流程：**tile 类配置改动的风险应先用「波数账」量化再决定**——本轮量化出尾波损失仅 1.54%（W1/W3）、W2 为 0%，从而把预算从 bs 扫描转向 op 数，避免了上一轮 §4.4 残留风险①的盲区；同时记录「上一轮 `PL-1.38` 未采纳」的正确性：不是「无法量化」，而是**量化后收益 ≤1.5% < 噪声阈值** | evidence: 本文 §2.0(a)；repro: 无（算术）；toolchain: 无 |
| **D** | 陷阱（新工具链 `d5838899` 复验）：`T.vbitcast` 不改变 buffer 的 dtype 绑定，`vmul(fp16_vec, int16_vec)` 被后端拒绝（`requires the same element type`）⇒ 「掩码 cast + scale 乘融合为 bitcast」路径不可用 | evidence: `_r6_bitcast.py` 编译 stderr（`logs/r6/probe_bitcast` 同源）+ §2.0(d)；repro: `_r6_bitcast.py`；toolchain: 同上 |

流程问题：本轮改动是单点、量级大（−21%），因此**未触发合并轮**（skill 允许：单点即 current best 的
升级时无需额外合并）。未采纳/未量化项已在本轮全部量化完毕（波数、PRNG op 账、bitcast 融合）。

## 5. 残留与未采纳（如实报告）

- 未采纳：`r6_bitcast`（编译约束）、3-op hash（统计不合格）、bs≠8192（波数收益 ≤1.5% 且 UB cap 已到）。
- 未量化（不在 tune 集的 smoke 小 shape）：8192 tile 在 1M 元素（128 tile / 40 核 = 3.2 波，效率 80%）下
  的理论尾波损失约 20%；但 smoke 按契约不采集性能。与上一轮不同的是：本轮把该风险从「未量化」变为
  **`n_num/40` 的算术界定**（1M：128 tile → 4 波，效率 80%）；若未来需要，可用编译期 tile 守卫
  （`PL-1.38`）按 `N/bs/40` 的整除性选 bs，代价是引入按 shape 的常量分支。
- 有效 R+W 0.72–1.48 TB/s 仍低于同机逐元素天花板 2.49–2.67 TB/s（差距 1.7–3.5x），且差距全部来自软件
  PRNG（11 op 中 7 个是 hash+掩码生成）⇒ `perf_feedback.md` 的 `[DESIGN_LIMIT]` 结论继续成立，
  已按其 append-only 纪律追加「修正附录（R6）」。
