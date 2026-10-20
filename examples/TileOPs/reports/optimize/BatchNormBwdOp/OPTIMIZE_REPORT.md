# 优化报告：BatchNormBwdOp（Stage-4 optimize）

> **口径（2026-10-20 修订）**：本文件是**过程记录**——第 3 节的逐轮测量沿用**当时工具链**，**不参与交付口径**。交付数据一律取 `perf_data/canonical.json`（CANN 9.0.0，每算子恰好一条记录），其中同时给出**迁移态（优化前）**与**交付态（当前交付）**两列；本文数字与之不一致时以 SSOT 为准。同 case 的前后对比见 `examples/TileOPs/reports/迁移与优化总报告.md` 的汇总表。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `BatchNormBwdOp`；**多 kernel（wide / split / general 三 kernel）**、单 func each、**纯 Vector（Vector-only）**；语义 = batch norm 反向（grad_input / grad_weight / grad_bias） |
| kernel（基准，只读） | `tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`（`_batch_norm_bwd_wide_kernel.py` / `_batch_norm_bwd_split_kernel.py` / `_batch_norm_bwd_kernel.py`） |
| 调优产物 | `.../batch_norm_bwd_kernel/perf_opt/BatchNormBwdOp.py`（三工厂合并）+ `perf_opt/_batch_norm_bwd_{wide,split,kernel}.py`（均 subid 分片） |
| 口径 | `msprof op Task Duration`（**唯一时延口径**）+ msprof roofline `GM Read+Write` ratio（仅作缺口归因，不用于跨 subid 前后对比）；数据来自 `perf_opt/perf_records.jsonl`（18 行） |
| **工具链版本戳** | Ascend910B3（20 AIC / 40 AIV，`get_aicore_num()=20`）、CANN 9.0.0、tilelang 0.1.2 dev-root / 2026-10-18；bishengir 与 hivmc 版本原文未记录 |
| 达标线 | **算力受限 ≥60%**；依据：三 kernel 均 Vector-only（单 `T.Scope("Vector")`、无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/L1/L0C，msprof `ArithmeticUtilization` 实测 `aic_cube_ratio=0`），roofline computility 归属 `Vec_FP32(65%) + Vec_MISC(35%)`，**不满足 Cube 主导混合类（Mix ≥40%）**，故按 60% 而非 40% 判 |
| benchmark | `bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`（6 tune + 3 smoke；原文记「共 8 案例」） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**唯一有效优化点是 subid 分片（PL-1.13-aiv-dup-subid-split）——修正 `T.Kernel(n)+T.Scope("Vector")` 在每 block 双 AIV 上重复执行的产能浪费，全域 Task Duration −33.1%~−45.9%；但 SOL 仍 14.0~25.5%，60% 算力线未达（结构性天花板），建议按净收益采纳 wrapper。**
- **达标数（双分母）**：
  - 全 shape 达标数：**0 / 8**（8 = 6 tune + 3 smoke 共 8 案例；smoke 不采性能，按精度回归计 3/3 通过）。
  - 可调 shape 达标数：**0 / 6**（6 个 tune workload 的算力受限 60% SOL 均未达标）。
  - 豁免：**3**（`smoke-fc` 三 dtype，依据：smoke 不采性能、仅做精度回归）；无微型 shape 性能豁免。
- **硬底线（Task Duration 变短）：6 / 6 达标**（**−33.1% ~ −45.9%**）。
- 一句话补充：SOL 未达 60% 属结构性天花板（见 §7），但 Task Duration 硬底线全 workload 显著下降 → **建议采纳 wrapper 切换**（有净收益）。
- **是否建议采纳 wrapper**：**是**——有净收益（−33~−46%），精度无回归（L0 三 kernel 全 PASS + smoke 全 PASS），签名 drop-in。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 达标判据归类 | `aic_cube_ratio = 0`；roofline computility = `Vec_FP32(65%) + Vec_MISC(35%)` | Vector-only，**按算力受限 60% 判**（Mix ≥40% 口径不适用） |
| AIV 是否重复执行（关键） | `Block Dim / Mix Block Dim = 40 / 80`（Mix = 2×Block Dim）；`vector0` 与 `vector1` 子块 `aiv_vec_ratio` **完全相同（0.230873）**、`aiv_vec_fops` 相同（229376） | **两个 AIV 子块指标完全相同而非各半 ⇒ 每 block 双 AIV 重复执行**，产能 2× 浪费（命中 PL-1.13 判据） |
| 向量利用率 | `aiv_vec_ratio = 0.231`；`aiv_time ≈ 10.5µs`（占 21µs 的 50%） | 约 50% 时间 AIV idle（双 wave 调度 + 重复执行） |
| 其它 pipe | `aiv_scalar_ratio ≈ 0.19`、`aiv_mte2_ratio ≈ 0.08`、`aiv_mte3_ratio ≈ 0.05` | 搬运/标量均非首要矛盾，先修「重复执行」 |
| 基线时延 | Task Duration 21.08µs（resnet-stage1-float16）；roofline GM `GM Read+Write` ratio = 0.183（18.0%） | 向量引擎占空比不足是主因 |
| 设计假设是否失配 | DESIGN §5.5 以 `num_kernels = min(C, 40)`（"AI cores ×2"）试图用满 40 向量核，实际 = 40 block × 双 AIV 冗余 | **是**：设计期的「AI cores ×2」假设带来 2× 产能浪费而非 2× 吞吐（后续已由 subid 分片修正） |
| 固定开销/噪声 | 原文未记录显式噪声阈值；final 与 round-1 候选的差异（如 13.260→13.460µs）为复测波动 | 判定以 −33~−46% 的幅度为准，远超噪声 |

### 2.2 方法（做了什么、为什么这样做）

三个 kernel 统一一个优化点：**subid 分片（PL-1.13-aiv-dup-subid-split）**。

1. **分核数修正：`num_kernels = min(C, 40)` → `min(C, aicore_num)`（aicore_num = `get_aicore_num()` = 20）**
   - 改了什么：把 block 数从「AI cores ×2 = 40」改为实际 aicore 数 20，并引入 `total_slots = num_kernels * 2`。
   - 为什么有效：原 40 block 在每 block 双 AIV 上**各执行一遍**同一份通道工作（Mix Block Dim = 80），等于 2× 冗余；改为 20 block 后用满 40 个 AIV 的**不同** subid 分摊 40 个 slot，冗余消失。
   - 正确性依据：通道集合与计算式不变，只是「哪个 AIV 处理哪个通道」的映射变化；L0 全 PASS。

2. **通道/任务索引改为 subid 感知：`c = i * total_slots + cid * 2 + subid`**
   - 改了什么：`cid`（block id）与 `subid`（`T.Kernel(...) as (cid, subid)` 的第二元）共同决定通道偏移，`subid ∈ {0,1}` 因此两个 AIV 处理**不同**通道。
   - 为什么有效：这是消除重复执行的直接手段——同一 block 的两个 AIV 从「同一通道各算一遍」变为「相邻两个通道各算一份」，产能翻倍而非重复。
   - 正确性依据：`cid*2 + subid` 在 `cid ∈ [0,20)`、`subid ∈ {0,1}` 上恰好覆盖 `[0,40)`，与原 40 个通道组一一对应（双射，无漏无重）；L0 全 PASS。

3. **split kernel 三段同步改**：stats / finalize / apply 三段一起改（stats 任务索引 `t = i*(KS*2)+cid*2+subid`；finalize/apply 通道索引同 wide）。
   - 为什么必须同步：split 的三段共享同一套分片假设，只改一段会造成 stats 与 apply 的通道归属错位。
   - 正确性依据：三段索引一致（同一双射），L0 全 PASS。

4. **签名不变（drop-in）**：三 kernel 工厂签名、`out_idx`、wrapper 调用约定均未改。
   - 依据：无 wrapper 改动需求，conductor 可直接翻转双 import 切换块；`perf_opt/BatchNormBwdOp.py` 的三工厂函数体与已测量独立文件逐字一致，编译产物不变。

**被否决/未启动的方向（保留，原因见 §7）**：消除 fp32 中转（正确性硬约束）、消除 vbrc 广播（算法固有）、降低 scalar 开销（无结构性抓手）、归约走 Cube（极瘦 GEMM，无实测先例）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | baseline（Stage-3 三 kernel） | — | 6 个 tune workload baseline = 21.080 / 20.400 / 25.351 / 42.241 / 43.181 / 5365.027 µs | 参照 |
| R1 | wide `subid-split` | 分核 40→20 + subid 通道分片 | resnet-stage1-float16 21.080→**13.260µs（−37.1%）**，L0 PASS | ✅ 采纳 |
| R1 | split `subid-split` | 三段（stats/finalize/apply）均 subid 分片 | large-spatial 5365.027→**2905.108µs（−45.9%）**，L0 PASS | ✅ 采纳 |
| R1 | 候选 vs current best 全量对比 | — | wide 各 workload −28.2%~−44.6%、split −45.9%（opt_log B2 表） | ✅ winner 合并 |
| 后续 | 消 fp32 中转 | fp16/bf16 直接算 | **不可行**：bf16 无 Vector 算术、fp16 须对齐 fp32 golden（正确性硬约束） | ❌ 否决 |
| 后续 | 消 vbrc 广播 | 隐式列广播替代 | **不可行**：逐通道标量广播是算法固有；PL-1.15 已证隐式列广播 +12~16% 税 | ❌ 否决 |
| 后续 | 降 scalar 开销 | 地址计算/reduce 优化 | 无独立结构性抓手（与「一 AIV 一通道」结构绑定），无可验证收益路径 | ⚠️ 未启动 |
| 后续 | 归约走 Cube（ones-vector GEMM） | Cube 替代向量归约 | 形态为 M=1/K=L/N=1 的极瘦 GEMM，Cube 无优势且需 GM→L1→L0C 往返，无实测先例支撑 >2× | ❌ 否决（未实证假设） |
| final | merged `BatchNormBwdOp.py` | 三工厂合并（wide/split/general） | 6 个 tune workload final 见 §3 | ✅ final |

**收束依据（plateau）**：主要结构候选（subid 分片）已充分验证并合并；其余方向为正确性硬约束或无结构性抓手，符合「连续无 >3% 提升的结构候选 + 主要方向已验证」的收束条件。

## 3. 优化结果（msprof 对比表）

| workload | shape (N,C,S) | dtype | base µs | final µs | Δ | 是否达标（60%） |
|---|---|---|---:|---:|---:|---|
| resnet-stage1-float16 | 8,64,1024 | fp16 | 21.080 | 13.460 | −36.2% | ❌ |
| resnet-stage1-bfloat16 | 8,64,1024 | bf16 | 20.400 | 13.640 | −33.1% | ❌ |
| resnet-stage2-float16 | 4,128,1024 | fp16 | 25.351 | 15.810 | −37.6% | ❌ |
| resnet-stage3-float16 | 4,256,784 | fp16 | 42.241 | 23.530 | −44.3% | ❌ |
| resnet-stage3-float32 | 4,256,784 | fp32 | 43.181 | 24.340 | −43.6% | ❌ |
| large-spatial-float16 | 4,128,262144 | fp16 | 5365.027 | 2903.328 | −45.9% | ❌ |

smoke：`smoke-fc` [32,64,1] 三 dtype 仅精度回归（wide kernel），不采性能。

**SOL 口径与「ratio 不可比」说明（本节要点）**：
- 原文只记区间、未逐 workload 记 ratio，故上表**不列 SOL 列**以免编造：**wide 14.0~15.5%、split 25.5%**（60% 未达标），全文区间 **14.0~25.5%**。
- **subid 分片后 ratio 下降是正确现象**：msprof roofline 的 `performance = counted_flops / duration`，重复执行期 counted_flops 被双计（两 AIV 各算一遍），分片后仅计一次 → **ratio 与 AI 一并下降而 Task Duration 同时下降**（opt_log 记 baseline ratio 17.995~28.164 / AI 3.96~9.36 → 候选 ratio 12.715~25.394；AI 同步下降，如 resnet-stage1 由 7.0→3.5）。
- 因此 **ratio 不可比，以 Task Duration 为准**；该 metric 在重复执行下被高估，不用于跨 subid 前后对比，也不是本算子的裁决口径。
- 逐 workload base SOL% / final SOL% 原文报告未记录（上表 SOL 列为空即此原因）。

## 4. 精度

- **L0**：三 kernel 全 PASS（wide 14 items、split 13 items、general 15 items）。
- **smoke**：`smoke-fc` [32,64,1] 三 dtype（float32 / float16 / bfloat16）全 PASS（max_abs_err ≤ 1.9e-6）。
- 无精度回归。
- 关键边界：fp16/bf16 经 fp32 中转（bf16 无 Vector 算术、fp16 对齐 fp32 golden），subid 分片只改通道归属映射、不改计算式，故逐元素结果与原实现一致。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 起止时间 | 原文未记录 |
| 总墙钟 | **≈ 55 min**（含 baseline 采数、1 轮候选、final 复测、L0/smoke 精度回归），在 180min 止损内 |
| 实验分支 | 1 轮 2 分支（wide subid-split + split subid-split），每分支单优化点 |
| 采集参数 | `msprof op` 全局串行，每 workload launch-count=20 / warm-up=5 |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/BatchNormBwdOp.py`（三工厂合并，sha256 `42b6cd08…`；函数体与已测量独立文件逐字一致） |
| 分 kernel 产物 | `perf_opt/_batch_norm_bwd_wide_kernel.py`（sha256 `3a6e5fb5…`）、`perf_opt/_batch_norm_bwd_split_kernel.py`（sha256 `cffb859b…`）、`perf_opt/_batch_norm_bwd_kernel.py`（no_tunable_workload，sha256 `ffb95322…`） |
| 调优记录 | `perf_opt/opt_log.md`（含 `Final Performance Test Data` 表 + 逐轮对比表 + Skill Retrospective） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（18 行，append-only，每 (kernel, workload) 恰一条 baseline/candidate/final） |
| 分类清单 | `perf_opt/workload_inventory.json`（分类完整，tuning_status=winner_merged / precision_pass=true） |
| 采集工件 | `perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`（过程日志） |
| 能力缺口 | 原文未记录（未登记 CG 条目） |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（建议采纳，由 conductor 在回归通过后翻转 baseline/perf_opt 双 import 切换块） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | PL-1.13 适用范围扩展到 Vector-only 逐通道归约 persistent（opt_log 提案） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项与依据**：`smoke-fc` 三 dtype（smoke 不采性能、仅精度回归，不计入达标分母）；无微型 shape 性能豁免（非 decode 类算子）。
- **SOL 60% 未达标的缺口归因（roofline + pipe 证据）**：
  1. **fp32 中转（正确性硬约束）**：fp16/bf16 → fp32 计算（bf16 无 Vector 算术、fp16 对齐 fp32 golden），ALU 吞吐减半。
  2. **Vec_MISC 35%**：vbrc 广播（逐通道 5 次）+ vcast，占向量工作 35%，封顶 ~65% SOL。
  3. **标量 + 内存**：wide `aiv_scalar 23~33%` + `aiv_mte2 12~18%`；split `aiv_mte2/mte3 31%/10%`（5-pass 流量）→ 向量引擎占空比 37~57%。
- **贴线风险**：无 <3pp 贴线项（SOL 区间 14.0~25.5% 距 60% 线甚远，属结构性缺口）。
- **ratio 不可比的再次声明**：subid 分片后 ratio（及 AI）下降与 Task Duration 下降同时发生，是 roofline 去双计的正确结果；**跨 subid 前后对比只用 Task Duration**，本报告不据 ratio 判定回退。
- **`[DESIGN_LIMIT]`**：**不触发**——无被设计冻结而可 >2× 加速的备选结构（fp32 中转 / vbrc / 内存为正确性或算法固有；「归约走 Cube」为 M=1/K=L/N=1 极瘦 GEMM，无实测先例，标未实证假设）；subid 分片这一结构性修复已在 Stage 4 内完成。未产出 `perf_feedback.md`。
- **缺口**：fp32 中转与 vbrc 为正确性/算法固有，无法在算子实现层消除；SOL 到 60% 的路径需改变数值契约（不可接受）。

## 8. 建议

1. **建议采纳 wrapper**：有净收益（−33~−46%），精度无回归，签名 drop-in；建议 conductor 在回归通过后翻转 wrapper 的 baseline/perf_opt 双 import 切换块。
2. 后续动作：工具链升级时复核两点——（a）`Mix Block Dim == 2×Block Dim` 与两 AIV 子块指标是否仍相同（subid 分片前提），（b）`fp32 中转` 是否仍为必需（若 bf16 获得 Vector 算术，Vec_FP32/Vec_MISC 结构会变，需重评 60% 判据下的天花板）。
3. **未解决项与原因**：60% SOL 未达，根因为 fp32 中转 + vbrc 广播 + 5-pass 流量的结构性组合，非 tiling/参数可解；「归约走 Cube」未经实测，不建议启动。
