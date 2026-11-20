# 优化报告：BatchNormBwdOp（Stage-4 optimize）

> **口径**：本文件是**过程记录**，交付数据一律取 `perf_data/canonical.json`（CANN 9.0.0）。
> 2026-10-20 完成 **R3 结构深挖**：§2–§3 同时保留 R1/R2（subid 分片）与 R3（三项结构改动）两轮记录；
> §3 的 R3 表为当前交付态来源，R2 表仅作历史对照。同 case 前后对比见
> `examples/TileOPs/reports/迁移与优化总报告.md`。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `BatchNormBwdOp`；**多 kernel（wide / split / general 三 kernel）**、单 func each、**纯 Vector（Vector-only）**；语义 = batch norm 反向（grad_input / grad_weight / grad_bias） |
| kernel（基准，只读） | `tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`（`_batch_norm_bwd_wide_kernel.py` / `_batch_norm_bwd_split_kernel.py` / `_batch_norm_bwd_kernel.py`） |
| 调优产物 | `.../batch_norm_bwd_kernel/perf_opt/BatchNormBwdOp.py`（三工厂合并）+ `perf_opt/_batch_norm_bwd_{wide,split,kernel}.py`；分支文件 `_*_r3_{w1,w2,s1,s2,s3,s4}.py` |
| 口径 | `msprof op Task Duration`（**唯一时延口径**）+ msprof roofline `GM Read+Write` ratio（仅作缺口归因，不用于跨 subid 前后对比）；数据来自 `perf_opt/perf_records.jsonl`（68 行，final = `r3_final_w2s3`） |
| **工具链版本戳** | Ascend910B3（20 AIC / 40 AIV，`get_aicore_num()=20`）、CANN 9.0.0、tilelang 0.1.2 dev-root（`9309edb`）；R3 实测 2026-10-20 |
| 达标线 | **算力受限 ≥60%**；依据：三 kernel 均 Vector-only（单 `T.Scope("Vector")`、无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/L1/L0C，msprof `ArithmeticUtilization` 实测 `aic_cube_ratio=0`），roofline computility 归属 `Vec_FP32 + Vec_MISC`，**不满足 Cube 主导混合类（Mix ≥40%）**，故按 60% 判 |
| benchmark | `bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`（6 tune + 3 smoke） |
| 过程日志 | `perf_opt/opt_log.md`（§4 R3 深挖 + §8 R3 Retrospective）、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`REPORT_R2_DATA.md` |
| 端到端验收 | `reports/tileops/20261020_185252_421670_BatchNormBwdOp/`（status=passed，正确性 12/12） |
| 前置版本 | v0 / v1 已归档 `history_version/` |

## 1. 结论

- **一句话**：R1/R2 用 subid 分片修掉双 AIV 重复执行（−33~−46%）；R3 再用**三项单点结构改动**（wide 的 `x_hat` 代数消去 + 常数折叠、wide/split 的 per-input loads-first、split apply 的 epilogue 折叠）把全部 6 个 tune workload 在**同会话配对**下再降 **12.3% ~ 19.8%**，相对迁移态累计 **−38.6% ~ −56.6%**；60% SOL 仍未达（结构性天花板，见 §7）。
- **本轮目标达成**：目标是 current best 再降 10–20%，实测 −12.3% / −16.7% / −15.0% / −12.4% / −13.5% / −19.8%（6/6 净收益，全部超过 3% 噪声阈值）。
- **达标数（双分母）**：
  - 可调 shape 达标数：**0 / 6**（算力受限 60% 未达）。
  - 全 shape：6 tune 全部未达标；另 3 个 `smoke-fc` 用例（fp32/fp16/bf16）仅做精度回归（3/3 PASS），不采性能、不计入达标分母。
  - 豁免：**3**（smoke）；无微型 shape 性能豁免（本算子无 ≤5µs 用例）。
- **硬底线（Task Duration 变短）：6 / 6 达标**（相对迁移态 −38.6~−56.6%；相对 R2 best −12.3~−19.8%）。
- **wrapper**：R1 后已切 `perf_opt` 并在本轮保持激活；R3 只在 `perf_opt/` 内更新，**无需再次翻转**。
- **是否建议保留采纳**：**是**——有净收益、精度零回归、签名 drop-in。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

#### R1/R2：双 AIV 重复执行（已修复）

| 观测 | 数值 | 结论 |
|---|---|---|
| 达标判据归类 | `aic_cube_ratio = 0`；roofline computility = `Vec_FP32 + Vec_MISC` | Vector-only，**按算力受限 60% 判** |
| AIV 是否重复执行（关键） | `Block Dim / Mix Block Dim = 40 / 80`；`vector0` 与 `vector1` 子块 `aiv_vec_ratio` **完全相同（0.230873）**、`aiv_vec_fops` 相同（229376） | 两个 AIV 子块指标完全相同而非各半 ⇒ **每 block 双 AIV 重复执行**，产能 2× 浪费（PL-1.13） |
| 向量利用率 | `aiv_vec_ratio = 0.231`；`aiv_time ≈ 10.5µs`（占 21µs 的 50%） | 约 50% 时间 AIV idle（双 wave + 重复执行） |
| 其它 pipe | `aiv_scalar_ratio ≈ 0.19`、`aiv_mte2_ratio ≈ 0.08`、`aiv_mte3_ratio ≈ 0.05` | 先修「重复执行」 |
| 基线时延 | 21.08µs（resnet-stage1-float16）；GM ratio 18.0% | 向量引擎占空比不足是主因 |
| 设计假设失配 | DESIGN §5.5 的 `num_kernels = min(C, 40)`（"AI cores ×2"）实际 = 40 block × 双 AIV 冗余 | **是**，由 subid 分片修正 |

#### R3：串行链与逐元素 op 数（本轮起点）

**wide（resnet-stage1-fp16，R2 best，13.27µs）**

| 指标 | 值 | 归因 |
|---|---:|---|
| `aiv_time` / Task | 9.98 / 13.27µs（75%） | 余下 3.3µs = 发射/尾波 |
| `aiv_vec_ratio` / `vec_time` | 0.373 / 4.85µs | 向量计算非瓶颈 |
| `aiv_scalar_ratio` / `scalar_wait` | 0.23–0.31 / 6.0µs | 逐通道 5 次 1 元素 copy + strided 寻址 |
| `mte2` / `mte3` | 1.9 / 0.8µs | 非带宽瓶颈 |
| 每元素向量 op | **≈17**（2 vcast + 5 vbrc + 7 arith + 2 reduce + 1 vcast） | 有代数消去空间 |
| Σpipe vs aiv_time | Σpipe ≈ aiv_time | 单 staging 缓冲把 MTE2→VCAST 串成一条链，无跨 pipe 重叠（PL-1.10 形态） |

**split（large-spatial-fp16，R2 best，2906.75µs）**

| stage | Task µs | vec_time | mte2 | mte3 | aiv_time | 重叠 |
|---|---:|---:|---:|---:|---:|---|
| stats | 1285.1 | 740.8 | 394.1 | 1.8 | 1274.0 | 22% |
| finalize | 10.0 | 0.19 | 1.5 | 3.4 | 5.6 | — |
| apply | 1611.6 | 898.4 | 496.4 | 161.3 | 1601.6 | **零重叠** |

- 有效带宽：stats 读 1.36 TB/s、apply 读 1.08 TB/s / 写 1.66 TB/s，均未饱和混合流地板（≈1.7 TB/s）⇒ **apply 是串行问题，不是带宽问题**。
- 每元素向量 op：stats 9、apply 12（2 vcast + 5 vbrc + 4 arith + 1 vcast）。

**被实测证伪的假设**：调度时假设「`N=8,S=1024` → bS=928 → 2 tile → 5 遍流量」。实际
`_plan_bS(8,1024,22,transit) = min(1024, align_down(184320//(8·22), 8)) = 1024`，resnet-stage1 早已走
single-scan（3 遍流量），该方向无实验价值、未做。

**测量噪声**：同一二进制两次独立运行在 wide 小 shape 上可差 ~7%（12.12 vs 13.02µs，各自 min–max ≤5%），
与目标增益同阶 ⇒ R3 最终比值一律取**同 NPU 会话内 A/B 交替**结果（`run_r3_ab.py`）。

### 2.2 方法（做了什么、为什么这样做）

#### R1/R2：subid 分片（PL-1.13-aiv-dup-subid-split）

1. **分核数修正：`num_kernels = min(C, 40)` → `min(C, aicore_num)`（=20）**，引入 `total_slots = num_kernels * 2`。
   - 为什么有效：原 40 block 在每 block 双 AIV 上各执行一遍同一份通道工作（Mix Block Dim = 80），等于 2× 冗余；改 20 block 后用满 40 个 AIV 的不同 subid 分摊 40 个 slot。
   - 正确性：通道集合与计算式不变，仅映射变化；L0 全 PASS。
2. **subid 感知索引：`c = i * total_slots + cid * 2 + subid`**。
   - 为什么有效：同一 block 的两个 AIV 从「同一通道各算一遍」变为「相邻两个通道各算一份」。
   - 正确性：`cid*2 + subid` 在 `cid∈[0,20)`、`subid∈{0,1}` 上恰好覆盖 `[0,40)`，双射无漏无重。
3. **split 三段同步改**（stats / finalize / apply），避免通道归属错位。

#### R3：三项单点结构改动

1. **wide：`x_hat` 代数消去 + 常数折叠（17 → 13 op/element）**
   - 改了什么：不再逐元素算 `x_hat=(x-mean)·rstd`，改为
     `gw = rstd·Σ(g·x) − (mean·rstd)·Σg`，
     `grad_x = A·g + B·x + C`（A/B/C 为逐通道常数）。
   - 为什么有效：省掉 2 次逐元素 vbrc（`mean`、`rstd`）+ 1 次 sub/mul 链，并把 epilogue 折成仿射式；
     `vec_time` 4.85→3.77µs、`aiv_time` 9.98→8.07µs（stage1）。
   - 正确性：代数恒等，长归约最不利用例（S=1048576 fp16）max_abs_err 3.906e-03，L0/L1 全 PASS。
2. **wide / split：per-input loads-first staging（PL-1.10 第三/四证）**
   - 改了什么：把 grad_out 与 x 的 DMA 前置、分开 staging，再进入计算；`brc` 缓冲复用为 `prod`（UB 中性）。
   - 为什么有效：消除「MTE2 → VCAST → 计算」单链依赖，让搬运与向量计算重叠；split 的 stats `mte2` 394→271µs、apply `mte2` 496→314µs。
3. **split apply：epilogue 折叠 + apply 独立 tile 宽**
   - 改了什么：apply 端直接由 `gb/gw/weight/rstd` 合成 `scale/shift/xc`，并给 apply 单独的 tile 宽。
   - 为什么有效：省去 finalize 的系数平面写 + apply 的读，减少 C 级 GM 往返与依赖链。
   - 结果：apply 1611.6→1233µs、stats 1285.1→1091µs。

**被否决/未启动的方向**：
- `T.Pipelined(2)` 用于 apply tile 循环（S4）：与 S3 tie（2528.35 vs 2528.01µs）→ 判 `config_no_gain`，弃。
- 消除 fp32 中转（bf16 无 Vector 算术、fp16 需对齐 fp32 golden → 正确性硬约束）。
- 消除 vbrc 广播（算法固有；PL-1.15 已证隐式列广播 +12~16% 税）。
- 归约走 Cube（M=1/K=L/N=1 极瘦 GEMM，无实测先例，未实证假设）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | baseline（Stage-3 三 kernel） | — | 6 tune 21.080 / 20.400 / 25.351 / 42.241 / 43.181 / 5365.027 µs | 参照 |
| R1 | wide `subid-split` | 分核 40→20 + subid 通道分片 | resnet-stage1 21.080→13.260µs（−37.1%），L0 PASS | ✅ 采纳 |
| R1 | split `subid-split` | 三段同步 subid 分片 | large-spatial 5365.027→2905.108µs（−45.9%），L0 PASS | ✅ 采纳 |
| R2 | merged `BatchNormBwdOp.py` | 三工厂合并 | final 13.460 / 13.640 / 15.810 / 23.530 / 24.340 / 2903.328µs | ✅ R2 交付 |
| **R3-W1** | `r3_w1_gwfold` | wide：`x_hat` 消去 + 仿射折叠 | 12.010 / 13.900 / 15.430 / 22.490 / 22.530µs | ✅ 4/5 净收益（bf16 打平） |
| **R3-W2** | `r3_w2_loadsfirst` | W1 + per-input staging + `brc`→`prod` | 12.120 / 12.400 / 14.610 / 19.170 / 20.280µs | ✅ winner（wide） |
| **R3-S1** | `r3_s1_loadsfirst` | split apply loads-first | 2715.840µs（−6.6%） | ✅ |
| **R3-S2** | `r3_s2_applyfold` | S1 + apply epilogue 折叠 + 独立 tile 宽 | 2528.010µs（−6.9%） | ✅ |
| **R3-S3** | `r3_s3_statsloadsfirst` | S2 + stats loads-first | **2333.080µs（−7.7%）** | ✅ winner（split） |
| R3-S4 | `r3_s4_applypipe2` | S2 + `T.Pipelined(2)` | 2528.350µs（tie） | ❌ 弃 |
| **R3-final** | `r3_final_w2s3` | W2 + S3 合并进 `perf_opt/BatchNormBwdOp.py` | 见 §3 | ✅ final（gate pass） |

**收束依据（success）**：目标「再降 10–20%」在 6/6 workload 达成；S4 tie 后无更高价值候选，停止。

## 3. 优化结果

### 3.1 R3 final（当前交付态，同会话 A/B 配对）

| workload | shape (N,C,S) | dtype | R2 best µs | R3 final µs | vs R2 | vs 迁移态 |
|---|---|---:|---:|---:|---:|---:|
| resnet-stage1-float16 | 8,64,1024 | fp16 | 14.360 | **12.600** | −12.3% | −38.6% |
| resnet-stage1-bfloat16 | 8,64,1024 | bf16 | 13.780 | **11.480** | −16.7% | −45.7% |
| resnet-stage2-float16 | 4,128,1024 | fp16 | 16.840 | **14.320** | −15.0% | −43.7% |
| resnet-stage3-float16 | 4,256,784 | fp16 | 23.630 | **20.690** | −12.4% | −49.9% |
| resnet-stage3-float32 | 4,256,784 | fp32 | 24.130 | **20.880** | −13.5% | −49.9% |
| large-spatial-float16 | 4,128,262144 | fp16 | 2911.340 | **2333.520** | −19.8% | −56.6% |

- 同会话配对原始值见 `REPORT_R2_DATA.md` §3；绝对时延取 final 侧记录。
- R3 final 对应 `ratio_pct`（perf_records、完整 pipeline）：11.4 / 13.5 / 10.4 / 11.8 / 16.2 / 33.3；
- 端到端 `tileops-report`（交付口径）逐 case：12.22 / 12.14 / 14.26 / 20.44 / 19.78 / 1091.52µs，ratio **10.67–28.71%**。
  **注意**：large-spatial 的端到端报告只捕获 `stats` kernel（1091.52µs），完整 stats+finalize+apply 为 2333.52µs；该行 ratio 与其它行不可直接横比。

### 3.2 R2 历史数据（subid 分片交付，仅作对照）

| workload | shape (N,C,S) | dtype | base µs | R2 final µs | Δ |
|---|---|---:|---:|---:|---:|
| resnet-stage1-float16 | 8,64,1024 | fp16 | 21.080 | 13.460 | −36.2% |
| resnet-stage1-bfloat16 | 8,64,1024 | bf16 | 20.400 | 13.640 | −33.1% |
| resnet-stage2-float16 | 4,128,1024 | fp16 | 25.351 | 15.810 | −37.6% |
| resnet-stage3-float16 | 4,256,784 | fp16 | 42.241 | 23.530 | −44.3% |
| resnet-stage3-float32 | 4,256,784 | fp32 | 43.181 | 24.340 | −43.6% |
| large-spatial-float16 | 4,128,262144 | fp16 | 5365.027 | 2903.328 | −45.9% |

**SOL 口径与「ratio 不可比」说明**：
- `performance = counted_flops / duration`。subid 分片前 counted_flops 被双计（两 AIV 各算一遍），分片后仅计一次
  → **ratio 与 AI 一并下降，而 Task Duration 同时下降**（如 resnet-stage1 的 AI 7.0→3.5）。
- 因此 **ratio 不可比，跨轮裁决只认 Task Duration**；本报告的收益判定全部基于 Task Duration。
- 60% 线未达属结构性缺口，证据见 §7。

## 4. 精度

### 4.1 R3 最终回归（conductor 亲自执行，2026-10-20）

- `python perf_opt/BatchNormBwdOp.py --level all`：
  - **L0 14/14 PASS**；**L1 14 grid + 4 契约 PASS**；**L2 4/4 PASS**；Boundary 4 项中 `extreme-weight` 为 WARN（不阻塞），其余 PASS。
  - Summary max_abs_err：L0 bf16/fp16 3.906e-03、L0 fp32 3.052e-05；L1 bf16 1.562e-02、fp16 3.906e-03、fp32 2.747e-04。
- smoke `smoke-fc` [32,64,1] fp32 / fp16 / bf16：PASS，max_abs_err 1.431e-06 / 1.907e-06 / 1.907e-06。
- 长归约（wide S=1048576 fp16，折叠路径最不利）：PASS，max_abs_err 3.906e-03。
- **端到端** `tileops-report`：正确性 **12/12（100%）**，avg max_abs_err 2.39e-03，status=**passed**。

### 4.2 R1/R2 历史

- L0：三 kernel 全 PASS（wide 14、split 13、general 15）；
- smoke 三 dtype 全 PASS（max_abs_err ≤1.9e-6）。

## 5. 时间与成本

| 项 | R1/R2 | R3 |
|---|---|---|
| 起止 | 2026-10-17 ~ 10-18 | 2026-10-20 17:23 ~ 18:42 |
| 总墙钟 | ≈55 min | **≈80 min**（基线复测 + 6 分支 + 同会话 A/B + final）；conductor 验收（回归 + 端到端 report + 数据面）另 ≈14 min；stage 总计 **94 min** |
| 实验分支 | 2 | 6（W1/W2/S1/S2/S3/S4） |
| 采集参数 | `msprof op` 串行，每 workload launch-count=20 / warm-up=5 | 同左；final 采用同会话 A/B 交替 |
| 预算 | 180 min 内 | 180 min 内（未超时） |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/BatchNormBwdOp.py`（三工厂合并，sha256 `37099e26…`） |
| 分 kernel 产物 | `perf_opt/_batch_norm_bwd_wide_kernel.py`（`b0b1a1ab…`）、`perf_opt/_batch_norm_bwd_split_kernel.py`（`e284801e…`）、`perf_opt/_batch_norm_bwd_kernel.py`（`ffb95322…`，general 无 benchmark 命中） |
| 调优记录 | `perf_opt/opt_log.md`（§4 R3 深挖 + §8 Retrospective，含每轮候选 vs best 对比表 + Final Summary） |
| 结构化记录 | `perf_opt/perf_records.jsonl`（68 行，append-only；final 6 条 = `r3_final_w2s3`，artifact 指向 `BatchNormBwdOp.py`） |
| 分类清单 | `perf_opt/workload_inventory.json`（tuning_status=winner_merged / precision_pass=true） |
| 采集工件 | `perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`（含 `logs/r3/` 备份与披露） |
| 数据交接 | `perf_opt/REPORT_R2_DATA.md`（R3 诊断/方法/逐 workload 数据） |
| gate 4 | **pass**（failures [] / warnings []） |
| 端到端 report | `examples/TileOPs/reports/tileops/20261020_185252_421670_BatchNormBwdOp/`（passed） |
| wrapper | 已激活 `perf_opt`（R1 后翻转，本轮无需再改） |

> **记录例外披露**：gate 4 要求每个 tune workload 恰好一条 `phase=final`。R3 final 采用同会话 A/B 结果，
> 故把 12 条旧 `phase=final` 行（R2 final ×6、R3 repeat-1 ×6）改标为 `phase=merged`；**数值、候选 ID、
> artifact 路径/哈希、时间戳均未改动**，原文件备份 `perf_opt/logs/r3/perf_records_before_r3_ab.jsonl`，
> 详见 `perf_opt/logs/r3/README.md`。

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：`smoke-fc` 三 dtype（smoke 不采性能、仅精度回归），不计入达标分母；无微型 shape 豁免。
- **60% SOL 未达的缺口归因**：
  1. **fp32 中转（正确性硬约束）**：fp16/bf16 → fp32 计算（bf16 无 Vector 算术、fp16 对齐 fp32 golden），ALU 吞吐减半。
  2. **Vec_MISC**：逐通道 vbrc + vcast 仍占向量工作的一部分（R3 已把 wide 的逐元素 vbrc 从 5 次降到 3 次）。
  3. **流式 Pass 数**：wide 单 channel 超 UB 时 3 遍；split 结构上必须 stats 读 2 + apply 读 2 + 写 1（5 遍）。
- **R3 后仍存的结构性限制**：
  1. **apply 的 pipe 重叠仍为零**（`aiv 1227 ≈ vec 724 + mte2 314 + mte3 179`）；`T.Pipelined(2)` 无效（tie）。
     把 3 个 vbrc 提到 tile 循环外需 3×(2,3400) fp32 = 81.6KB 常驻，超 UB 余量（nominal 136KB / 192KB，
     含 multi-buffer 膨胀后无空间）→ 需更小 tile 换 hoist，未实证（潜在 ~−10~15% apply）。
  2. **wide scalar 段**（ratio 0.23–0.31，逐通道 5 次 1 元素 copy + strided 寻址）：批量取参需「UB 标量读入」
     形态，`T.vmul`/`T.vadd` 文档 §2.3 限制「标量操作数须在 `T.Scope` 外定义」→ 当前 API 无合法形态（未实证假设）。
  3. **wide 发射/尾波 3.3µs（26%）**：小 kernel 固定开销（PL-1.22 截距 ~1.6µs + 第二波尾）；减小需改任务粒度
     （跨 AIV 归约 → 5-pass 反噬），本 shape 不划算。
  4. **wave 量化**：C=64/128/256 → 2/4/7 波，AIV 利用率 80%/80%/91%；进一步均衡需跨 AIV 归约。
- **ratio 不可比的再次声明**：subid 分片后 ratio（及 AI）下降与 Task Duration 下降同时发生，是 roofline 去双计的正确结果；
  跨 subid 前后对比只用 Task Duration。
- **`[DESIGN_LIMIT]`**：**不触发**——上述缺口均无「被设计冻结、可 >2x」的备选结构（Cube 归约仍为无实测先例的未实证假设），
  不满足 `perf-feedback.md` §1 双门槛，未产出 `perf_feedback.md`。

## 8. 建议

1. **保留采纳**：R3 final 已在 `perf_opt/` 激活，有净收益（vs R2 −12.3~−19.8%，vs 迁移态 −38.6~−56.6%），精度零回归，签名 drop-in。
2. **不再投入的方向**：fp32 中转 / vbrc 消除 / Cube 归约 / `T.Pipelined(2)`（均已举证无效或碰正确性硬约束）。
3. **如要继续**：唯一可测的剩余方向是「apply 的 vbrc hoist + 更小 tile」权衡（潜在 ~−10~15% apply），需先做 UB 账与
   L0 验证；否则建议按当前版本收束。
4. **工具链升级时复核**：（a）bf16 是否获得 Vector 算术（若获得，fp32 中转可去掉，需重评 60% 天花板）；
   （b）`Mix Block Dim == 2×Block Dim` 与两 AIV 子块指标是否仍相同（subid 分片前提）。
