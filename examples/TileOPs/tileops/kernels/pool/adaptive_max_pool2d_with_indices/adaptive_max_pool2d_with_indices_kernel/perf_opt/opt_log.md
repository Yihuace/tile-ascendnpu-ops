# Stage 4 调优日志 — `_adaptive_max_pool2d_with_indices_kernel`（AdaptiveMaxPool2dIndicesFwdOp）

- project: `adaptive_max_pool2d_with_indices`（TileOPs 集成算子，算子在
  `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/`）
- scenario / mode: `optimize` / `full`
- base（Stage 3，只读）: `../_adaptive_max_pool2d_with_indices_kernel.py`（L0/L1 精度通过，Stage 5 集成 passed）
- 最终产物: `perf_opt/_adaptive_max_pool2d_with_indices_kernel.py`（sha256
  `8156c03c7c27c5ef46bb6b56dcdab943bb87ed852175484df2d0474afe4f74b0`）
- 唯一时延口径: **`msprof op` Task Duration(us)**（median / 20 launch，`--launch-count=20 --warm-up=5`）
- 工具链戳（唯一口径，本轮实测）: `tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23 (67a0441)`
  - CANN 9.0.0 + Ascend910B3（`npu-smi 26.1.1`，`get_aicore_num()`=20 → 40 Vector 核）
  - torch 2.9.0+cpu / torch_npu 2.9.0.post8
- 预算: `budget.max_stage4_experiments = 15`（本轮 formal 实验分支 = **9**，见 §4）；`max_rounds`=10
- 采集纪律: 所有 `msprof op` 调用串行 `flock /tmp/tilelang-msprof.lock`
- **版本戳纪律**：本工具链（`67a0441`）与同族前作（avg pool `417dae9`、max pool `489385e`、argmin `417dae9`）
  **均不同**，故所有引用前作结论均标注「待重验/本轮重验」，本轮关键结论一律以本机实测为准。

---

## 1. Benchmark workload 清单（Phase 1）

展开入口 = `benchmarks/ops/bench_pool.py::test_adaptive_max_pool2d_indices_bench`（manifest-driven，
op `AdaptiveMaxPool2dIndicesFwdOp`，`workloads_to_params(..., include_extra=True)`；`include_extra` 只把
`output_size` 并入 `op_params`，**不新增案例**）。**实际展开 = 4 例**，逐项写入 `workload_inventory.json`。

| # | workload_id | shape | output_size | dtype | kind | 处理 |
|---|---|---|---|---|---|---|
| 1 | `smoke-1x8x4x4-2x2-float16` | [1,8,4,4] | (2,2) | float16 | **smoke** | 仅精度回归（label 含 `smoke` 词） |
| 2 | `global-1x1-float16` | [8,2048,7,7] | (1,1) | float16 | tune | **优先** |
| 3 | `spp-6x6-float16` | [2,128,56,56] | (6,6) | float16 | tune | 第二 |
| 4 | `nondiv-7x7-bfloat16` | [2,64,55,57] | (7,7) | bfloat16 | tune | 第三 |

skip：无。

**调优顺序理由（依自采 msprof baseline 实测负担，非 DESIGN/report 值）**

| 顺序 | workload | 自采 baseline Task(us) | BlockDim | 主要负担来源 |
|---|---|---:|---|---|
| 1 | `global-1x1` | **144.393** | 40 | mte2 62.6 + mte3 56.3（DMA 扇出：读 7.8x / 写 21x） |
| 2 | `spp-6x6` | **85.572** | 24（16 核空） | vec 45.5 + scalar 31.0（每窗口元素 op 数 + 核未占满） |
| 3 | `nondiv-7x7` | **81.412** | 14（26 核空） | vec 43.2 + scalar 30.8（同上，且核占用更差） |

> Stage 5 report 的 baseline（151.60 / 85.36 / 81.60 µs）是 wrapper 口径，仅作量级交叉验证；
> 本表为自采 `msprof op` 口径（唯一时延口径）。

---

## 2. 步骤 0：copy-floor 标定（强制前置）

标定 kernel：`calib_bench.py`（纯拷贝、同栅格 / 同 DMA 粒度）、驱动 `calib_run.py`；
写侧隔离探针：`probe_write_run.py`（写形态 A/B）、`probe_write2.py` / `probe_write3.py`（生产者隔离）。
结构化数据落 `perf_opt/copy_floor_calibration.jsonl`（**10 行**，**不写入 `perf_records.jsonl`**）。

### 2.1 字节记账（本算子真实 IO，双输出）

```
bytes = in_bytes + out_bytes + idx_bytes = N_in·2 + N_out·2 + N_out·8      # int64 = 8 B/输出元素
global-1x1   : 1 605 632 + 32 768 + 131 072 = 1 769 472 B
spp-6x6      : 1 605 632 + 18 432 +  73 728 = 1 697 792 B
nondiv-7x7   :   802 560 + 12 544 +  50 176 =   865 280 B
```

> 与 DESIGN §1.6.0 R3 表 `roofline.bytes` 逐字一致（含 int64 项）。**索引写占 IO 大头**
> （global 的 131 KB / 163 KB 输出字节 = 80%），故标定写侧必须体现双输出。

### 2.2 标定表（本机实测，`copy_floor_calibration.jsonl`）

| 标定/探针 variant | 栅格 / DMA 粒度 | CH | 轮数 | Task(us) | aiv(us) | vec(us) | mte2(us) | mte3(us) | 70% 线 = 地板/0.7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `global_ch32`（dual，**假地板**） | 整平面 ch×49 fp16，1 DMA/单元 | 32 | 13 | 110.662 | 92.61 | 3.49 | 29.40 | 79.81 | — |
| `global_ch64`（dual，**假地板**） | 同上 | 64 | 7 | 109.292 | 89.13 | 3.11 | 32.45 | 71.34 | — |
| `global_ch96`（dual，**假地板**） | 同上 | 96 | 5 | 95.102 | 78.02 | 4.61 | 28.99 | 59.21 | — |
| **`global_ch64_read_floor_elem`** | 整平面读 + 1 元素 fp16 落盘保活 | 64 | 7 | **6.860** | 4.92 | 0.01 | 2.11 | 0.00 | **9.800** |
| **`global_ch64_dual_contig_store`** | 整平面读 + fp16 值 + int64 索引，**连续生产者** | 64 | 7 | **4.730** | 2.80 | 0.17 | 2.17 | 0.92 | **6.757** |
| `global_ch64_dual_transpose_poison`（**假地板**） | 整平面读 + 双输出，**跨步 gather / transposed 生产者** | 64 | 7 | 105.742 | 88.34 | 3.11 | 32.00 | 71.47 | — |
| `general_ch64`（dual） | spp 栅格逐 bin 行 ch×56，1 波 | 64 | 1 | 10.980 | 9.29 | 0.52 | 6.71 | 1.72 | 15.686 |
| **`general_ch48`（dual，1 波 36 核）** | 同上 | 48 | 1 | **8.620** | 6.87 | 0.60 | 4.66 | 1.39 | **12.314** |
| `general_ch32`（dual，2 波） | 同上 | 32 | 2 | 11.660 | 6.21 | 0.28 | 3.81 | 1.49 | 16.657 |
| `general_ch16`（dual，3 波） | 同上 | 16 | 3 | 11.820 | 8.22 | 0.30 | 5.46 | 2.26 | 16.886 |

### 2.3 地板结论

1. **global 类：dual 输出标定 95–111 µs 是「假地板」，不可作锚**（`global_ch64_dual_transpose_poison` 105.7 µs
   与 step-0 dual 形式同值；与之对照的 **连续生产者** dual 形态 `p7` 仅 **4.73 µs**）。
   证据链（写侧隔离，同一读侧、逐项只加一个变量）：
   | 探针 | 写入内容 | Task(us) | mte3(us) | UB_to_GM(KB/blk) |
   |---|---:|---:|---:|---:|
   | `p1` fp16 值，vbrc/​vcast 生产者 | 值 | 4.73 | 0.82 | 0.8 |
   | `p7` p1 + int64 索引（vbrc→vcast32→vcast64 生产者） | 值 + 索引 | 4.73 | 0.92 | — |
   | `value`（跨步 gather 生产者） | 值 | 94.29 | 45.55 | 48.0 |
   | `index`（跨步 gather 生产者） | 索引 | 44.16 | 25.43 | 48.0 |
   | `dual`（step-0，跨步 gather 生产者） | 值 + 索引 | 109.29 | 71.34 | 96.0 |
   ⇒ **代价来自写侧生产者形态**（跨步 UB gather / transposed 生产者被降级为 UB→GM→UB 往返），
   **不是**字节、不是 int64 本身、也不是 `(real_c,1,1)` 目标 region 形态（p1/p7 用同一 region 形态）。
   这正是任务书警告的「假地板」在 global 类上的重现形态（同族 max 的 `(M,1)` 列切片假地板同族）。
2. **使用的地板锚（合法）**：
   - global 类读地板（1 元素保活）= **6.86 µs** → **70% 线 9.80 µs**（与任务书继承值 9.79 µs 一致，口径验证通过）；
   - global 类双输出连续生产者地板 = **4.73 µs** → 更严的 70% 线 6.76 µs（仅作参考，不作达标判据）。**主判据用 9.80**。
   - general 类（spp 栅格）dual 地板 = **8.62 µs**（CH=48，1 波 36 核）→ **70% 线 12.31 µs**。
3. **轮数自变量扫描（PL-1.6 纪律）**：global 同流量、13/7/5 轮 → 110.7 / 109.3 / 95.1 µs（但被假地板污染，
   轮数效应不可从该列读出）；general 同流量 1/1/2/3 轮（CH 64/48/32/16）→ 10.98 / 8.62 / 11.66 / 11.82 µs。
   **结论：general 类地板由「波数」主导（1 波最优，2/3 波分别 +35%/+37%），与每核轮数一致方向。**
4. **标定 vs 实测偏差说明（如实披露）**：
   - `global_ch96_dual` 标定用 `--ch 96` 的 dual 形式（105.7→95.1 µs）与最终 kernel 的 `CH_GLOBAL=104`
     不是同一形态（标定是 dual-假地板，最终 kernel 是 flat-plane 记录扫描），**两者不可直接相减**；
     最终 global 的 91.92 µs 落在「假地板 95.1–110.7」附近是**巧合**——它的读侧已是干净整平面读
     （`GM_to_UB = 52 KB/block ≈ 真实 42.9+写`），残余代价是**平面转置的 UB→GM→UB 往返**
     （`UB_to_GM = 104 KB/block` vs 真实写 4.5 KB）。
   - `nondiv-7x7`（[2,64,55,57] bf16）**未单独标定**（本轮标定只覆盖 global-1x1 与 spp-6x6 两条栅格）；
     其 general 类地板沿用 spp 栅格值（12.31 µs 线）作**同类近似锚**，已在 §7 标注为口径缺口。

---

## 3. 现象分析与根因（Phase 2 前置调查）

### 3.1 baseline 根因（msprof，median/20）

| workload | Task | BlockDim | aiv | vec | mte2 | mte3 | scalar | GM_to_UB(KB/blk) | UB_to_GM(KB/blk) | 真实 IO(KB/blk) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global-1x1 | 144.393 | 40 | 117.58 | 25.92 | 62.60 | 56.26 | 20.44 | 336.0 | 96.0 | 读 42.9 / 写 4.5 |
| spp-6x6 | 85.572 | 24 | 83.40 | 45.53 | 4.59 | 1.43 | 30.96 | 80.0 | 18.5 | 读 70.0 / 写 3.75 |
| nondiv-7x7 | 81.412 | 14 | 79.71 | 43.17 | 4.34 | 1.04 | 30.77 | 72.0 | 19.75 | 读 64.0 / 写 3.38 |

**根因（两条独立机制，本轮实测）**

- **G-A（global-1x1）**：`out_H==out_W==1` 在 baseline 走一般网格路径（C4 整平面路径在 Stage 3 因
  `TRAP-npuir-arange-pattern-with-whole-plane-reduce` 被停用）。该路径**逐行**读（`x[nb,cslice,ih0+kh,0:w_in]`，
  `w_in=7` 非向量宽整数倍）+ 每行一次 `(ch,7)` 转置 + 转置/回转型 epilogue 写，令 **读扇出 7.8x
  （42.9 → 336 KB/blk）**、**写扇出 21x（4.5 → 96 KB/blk）**，`mte2+mte3 = 118.9 µs` 直接吃掉整个 aiv。
- **G-B（spp / nondiv）**：窗口扫描的**每元素向量 op 数 = 8 条**（7 条有效 + 1 条死代码），
  vec 43–46 µs + scalar 23–31 µs 主导；且 `BlockDim` 24/14 < 40 → 逻辑单元数不足，大量核空闲。
  编译期即有告警 `hivm.hir.vcmp will execute by scalar instruction with low efficiency`（每元素 vcmp 的标量化）。

### 3.2 设计估算 vs 实测偏差（D-2 回填）

| workload | DESIGN §1.6.0 估算下界 | baseline 自采 Task | 偏差 | 失准项归因 |
|---|---:|---:|---:|---|
| `global-1x1` | ≈1.4 µs | 144.393 µs | **103x** | ① 流量项把 HBM 混合地板当可达时延下界，**漏掉整平面读的 DMA 扇出（7.8x）与写扇出（21x）**；② 漏掉转置/写侧 lowering 的 UB→GM→UB 往返；③ 本轮实测**纯拷贝地板即 4.73–6.86 µs**（已 3.4–4.9x 于估算下界） |
| `spp-6x6` | ≈1.4 µs | 85.572 µs | **61x** | ① 发射项按 ~0.5 µs/op 外推被设计文档自标「未文档化假设」，实测每元素 7–8 条窄向量 op 且 vcmp 标量化；② 实测 vec 忙时 45.5 µs 已 32x 于估算下界 |
| `nondiv-7x7` | ≈0.7 µs | 81.412 µs | **116x** | 同上；且 BlockDim=14（26/40 核空闲） |

结论：DESIGN 的估算行对**纯 Vector、双输出窗口选择、窄向量 op**域外推失准（固定开销、转置 lowering 往返、
发射标量化均未标定）。该偏差行是 §6 `[DESIGN_LIMIT]` 判定的量化证据（结论=**不触发**）。

### 3.3 关键结构性发现（本轮实测，构成 §4 候选来源）

| 发现 | 证据 | 影响 |
|---|---|---|
| **F1** 写侧代价由**生产者形态**决定，非字节/非 int64 | 写侧隔离探针 p1/p7（4.73 µs，UB_to_GM 0.8 KB）vs value/index/dual（94/44/109 µs，UB_to_GM 48/48/96 KB） | global 与 general 的 epilogue 都必须让 store 源为**连续 op 链**（vbrc/vcast/reduce 输出），**禁止**跨步 UB gather 或 1 侧转置作为 store 源 |
| **F2** 整平面 `(C,plane)` 转置在本工具链被降级为 UB→GM→UB 往返 | `r1_flat`：`UB_to_GM = 96 KB/blk`（= 上游 profile 的 2× `GM_to_UB`） | global 路径无法通过转置获得 `(plane,C)` 连续扫描行；残余代价 ~60 µs |
| **F3** `TRAP-npuir-arange-pattern-with-whole-plane-reduce` 是**生产者特有**，不是链特有 | `probe_global_reduce.py --check`：用**运行期构建**的 k 表（`T.serial` + typed `T.vbrc` + `T.copy`，非 arange）驱动
 `vcmp(eq)→vselect→reduce_min/max` 整平面 argmax，**索引逐元素精确**（16384/16384，含 NaN / 并列 / ±inf） | C4 整平面 argmax 的**正确性**障碍可绕过（arange 换成运行期表） |
| **F4** C4 整平面 argmax 在本工具链被 **UB 容量**封顶 | `probe_global_reduce`：ch=8 编译通过且精确；ch=16 需 1 618 432 bit > 1 572 864 bit（UB overflow）。标定得归约链单价 ≈ **237 B/平面元素**（含 ×2 multibuffer），远高于同族 max 单输出 C4 的 32 B/元素 | ch=8 → `num_local = ceil(2048/40) = 52` 轮 → 实测 **262.8 µs**（远差于 baseline）；C4 整平面 argmax 对双输出**不可采纳** |
| **F5** `(ch,plane)` 二维 `reduce(dim=1)` 标量化 + 往返 | `probe_global_2d.py` ch=64：Task **237.0 µs**、scalar **98.1 µs**、`UB_to_GM=96 KB/blk` | 归约必须走 `(ch,1,plane)` + `dim=2` 的「扁平空间轴」形态（PL-1.23 同族），不能走二维行归约 |
| **F6** `hivm.hir.vsel` 不接受**从更大 UB buffer 切出的 bool 条件** | `T.vselect(eq_mask[iw0+kw:iw0+kw+1, 0:ch_], ...)`（`eq_mask` 为 `(w_in,ch_)` bool）→ 编译期 `vsel op operands' shape are inconsistent`；同一 slice 作为 `vcmp` 操作数合法 | general 路径的「bool 切片直接 vselect」op 削减（8→6）**不可行**，只能保留 `vcmp` 产出的专用 `(1,ch_)` bool 缓冲（→ 最终只削掉死代码，8→7） |
| **F7** `(ch,1)` 列的跨步切片（stride=plane）作为向量操作数 → **core dump** | `_round1_gscan.py`：`T.vcmp(flat32[0:ch_,k:k+1], flat32[0:ch_,k:k+1], ...)` 运行期 core dump（acl 507034 族） | 「不做转置、直接按列扫描」的路线被排除 |

---

## 4. Iteration Log

**实验分支计数 = 9**（预算 15）：R1×3（flat / gen / gscan-blocked）、R2×1（merged）、
R2 参数×4（cg96/104/112/128）、R3×1（final）。另计标定/探针（非实验分支）。

### Round 1（base = Stage 3 baseline）

#### Diagnostic Context

- kernel_id: `…/_adaptive_max_pool2d_with_indices_kernel.py::_adaptive_max_pool2d_with_indices_kernel::main`
- current best: `../_adaptive_max_pool2d_with_indices_kernel.py`（baseline：global 144.393 / spp 85.572 / nondiv 81.412）
- Profile Facts：见 §3.1；结构性发现见 §3.3 F1/F2/F6

#### Current Phenomena

- P1（global-1x1）：读扇出 7.8x + 写扇出 21x，`mte2+mte3 = 118.9 µs`。
- P2（spp）：`BlockDim=24`（16 核空闲）；每元素 8 条向量 op，其中 1 条为死代码（`h_flag` 乘积未被任何消费者读取）。
- P3（nondiv）：`BlockDim=14`（26 核空闲）；同 G-B。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| G1 | P1 | global 路径改**扁平整平面**：`x` 声明 `(N,C,1,H*W)`，一次连续 DMA；再转置一次为 `(plane,C)` 供连续 `(1,C)` 行扫描 | §3.1 G-A；同族 max C4 的「扁平空间轴」在 Value 侧已验（**待重验→本轮重测**） | `is_global` 分支 + `_choose_ch_global` | Task / GM_to_UB / UB_to_GM / idx_exact | **improved（采纳）** |
| G2 | P1 | global 扫描期**只用 dtype 宽度的平面缓冲**（acc 为 fp16、NaN 掩码由 `x==x` 自比较产生），消去 fp32 平面缓冲 | §3.3 F1/F2 | `_round2_flat16` 形态 | Task / UB_to_GM | **improved（采纳，作为 G1 收尾）** |
| G3 | P1 | 跨步 UB gather / transposed 生产者 → 写侧连续化 | §3.3 F1（p1/p7 vs value/index/dual） | 由 G1/G2 统一实现 | UB_to_GM | **improved（蕴含于 G1/G2）** |
| G4 | P2+P3 | 删除死代码 `h_flag` 乘积（8→7 op/元素） | `h_flag` 无任何消费者（终判用 `nan_idx>=0`） | 删 buffer + 每元素 `T.vmul` | Task / vec | **improved（采纳）** |
| G5 | P2+P3 | general 路径**单波最小 CH 分派** | 标定 §2.3：general 地板由波数主导（1 波 8.62 < 2 波 11.66 < 3 波 11.82） | 新 `_pick_ch_general` | Task / BlockDim | **improved（采纳）** |
| G6 | P2+P3 | bool 切片直接 `vselect`（8→6 op） | 期望削 2 op | `vselect(eq_mask[slice], neg1, fvec, nanc)` | 编译 | **blocked（§3.3 F6：`vsel` shape 不一致）** |
| G7 | P1 | C4 整平面 argmax（运行期 pattern 取代 arange）+ `reduce_*` | §3.3 F3（正确性可绕过） | `probe_global_reduce` | Task / UB | **blocked（§3.3 F4：UB 封顶 ch=8 → 52 轮 → 262.8 µs）** |
| G8 | P1 | 二维 `(ch,plane)` 行归约（免转置） | 期望免去转置往返 | `probe_global_2d` | Task / scalar | **blocked（§3.3 F5：237.0 µs，scalar 98.1 µs）** |
| G9 | P1 | 「不做转置、按列扫描」（`(ch,1)` 跨步列切片的向量 op） | 期望免去转置往返 | `_round1_gscan` | 运行 | **blocked（§3.3 F7：core dump）** |
| G10 | P1 | global 块大小扫描（`CH_GLOBAL` 64→96/104/112/128） | 平坦区复测纪律 | 常量扫描 | Task | **improved（104 最优，见 R2）** |

#### Experiment Branches（R1）

| branch | base | opt_id | major_change | L0 | Task(us) | status |
|---|---|---|---|---|---|---|
| `_round1_flat.py` | baseline | G1 | global 走扁平整平面 + 转置 + 记录扫描 | pass | global **110.932** | improved（→ R2 收尾） |
| `_round1_gen.py` | baseline | G4+G5 | general 删死代码 + 单波最小 CH | pass | spp **79.742** / nondiv **72.641** | improved（winner） |
| `_round1_gscan.py` | baseline | G9 | 免转置按列扫描 | — | core dump | **blocked** |
| `probe_global_reduce.py`（探针） | baseline | G7 | 运行期 pattern + 整平面 reduce argmax | 精确 | ch=8: **262.795** | **blocked（UB 封顶）** |
| `probe_global_2d.py`（探针） | baseline | G8 | 二维行归约 | 精确 | ch=64: **236.985** | **blocked** |

#### 候选 vs current best 对比表（B2 结构化回流，数据源 `perf_records.jsonl` + raw profile）

| kernel_id | workload_id | candidate | Task Duration(us) | vs current best | aiv(us) | vec(us) | mte2(us) | mte3(us) | scalar(us) | GM_to_UB(KB/blk) | UB_to_GM(KB/blk) | BlockDim | L0 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| …::main | global-1x1-float16 | **current best = baseline** | 144.393 | 0.0% | 117.58 | 25.92 | 62.60 | 56.26 | 20.44 | 336.0 | 96.0 | 40 | pass | 待优化 |
| …::main | global-1x1-float16 | **round1_flat** | **110.932** | **-23.2%** | 92.88 | 23.33 | 32.16 | 66.33 | 17.30 | 48.0 | 96.0 | 40 | pass | **winner** |
| …::main | spp-6x6-float16 | **current best = baseline** | 85.572 | 0.0% | 83.40 | 45.53 | 4.59 | 1.43 | 30.96 | 80.0 | 18.5 | 24 | pass | 待优化 |
| …::main | spp-6x6-float16 | **round1_gen** | **79.742** | **-6.8%** | 77.89 | 42.59 | 3.90 | 1.23 | 26.53 | 60.0 | 13.9 | 36 | pass | **winner** |
| …::main | nondiv-7x7-bfloat16 | **current best = baseline** | 81.412 | 0.0% | 79.71 | 43.17 | 4.34 | 1.04 | 30.77 | 72.0 | 19.75 | 14 | pass | 待优化 |
| …::main | nondiv-7x7-bfloat16 | **round1_gen** | **72.641** | **-10.8%** | 70.93 | 40.03 | 2.31 | 0.81 | 23.00 | 36.0 | 9.875 | 28 | pass | **winner** |

#### Iteration Winner（R1）

- winner（global-1x1）: G1 → **110.93 µs（-23.2%）**（读扇出消除：GM_to_UB 336→48 KB/blk）
- winner（spp）: G4+G5 → **79.74 µs（-6.8%）**（BlockDim 24→36）
- winner（nondiv）: G4+G5 → **72.64 µs（-10.8%）**（BlockDim 14→28）
- new current best: `_round1_flat`(global) + `_round1_gen`(general) → 待合并（§R2）

### Round 2（base = round 1 winners；合并 + 扫参）

#### Diagnostic Context

- current best: global 110.93（flat）/ spp 79.74 / nondiv 72.64
- 剩余现象：global 的 `UB_to_GM = 96 KB/blk`（真实写 4.5 KB）——**平面转置被降级为 UB→GM→UB 往返**（F2）；
  且 `GM_to_UB = 48 KB/blk` 但 `mte2 = 32.2 µs`（同粒度纯拷贝仅 2.1 µs），转置同时拉长 mte2。

#### Current Phenomena

- P4：global 的两条 mte 通道均被转置往返拉长（mte2 32.2 / mte3 66.3），vec 仅 23.3。
- P5：global 的 7 轮（CH=64）→ 若减少轮数（增大 CH）可摊薄固定开销（标定 §2.3 第 3 条同向）。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| G2' | P4 | global 扫描期只用 dtype 宽度平面缓冲（消去 `row_f32`/`eq32`） | §3.3 F1 | `_round2_flat16` | Task / UB_to_GM | improved（小幅：110.9→108.9） |
| G10 | P5 | `CH_GLOBAL` 扫描 96/104/112/128 | 平坦区复测纪律 | 常量扫描 | Task | **improved（104 最优）** |

#### Experiment Branches（R2）

| branch | base | opt_id | major_change | L0 | Task(us) | status |
|---|---|---|---|---|---|---|
| `_round2_merged.py` | R1 winners | G2' | 合并（global flat16 + general 单波 CH），`CH_GLOBAL=64` | pass | global **108.942** / spp **79.902** / nondiv **72.461** | 合并非回退 ✓ |
| `_round2_cg96.py` | merged | G10 | `CH_GLOBAL=96` | pass | global **94.402** | improved |
| `_round2_cg104.py` | merged | G10 | `CH_GLOBAL=104` | pass | global **90.732** | **winner** |
| `_round2_cg112.py` | merged | G10 | `CH_GLOBAL=112` | pass | global 93.002 | rollback |
| `_round2_cg128.py` | merged | G10 | `CH_GLOBAL=128` | pass | global 97.482 | rollback |

#### 候选 vs current best 对比表（B2 结构化回流）

| kernel_id | workload_id | candidate | Task Duration(us) | vs current best | aiv(us) | vec(us) | mte2(us) | mte3(us) | scalar(us) | GM_to_UB(KB/blk) | UB_to_GM(KB/blk) | BlockDim | L0 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| …::main | global-1x1-float16 | **current best = round1_flat** | 110.932 | 0.0% | 92.88 | 23.33 | 32.16 | 66.33 | 17.30 | 48.0 | 96.0 | 40 | pass | 保持 |
| …::main | global-1x1-float16 | **round2_merged (CH=64)** | 108.942 | -1.8% | 89.24 | 22.10 | 31.45 | 67.56 | 17.02 | 48.0 | 96.0 | 40 | pass | 合并基准 |
| …::main | global-1x1-float16 | **round2_cg104** | **90.732** | **-18.2%** | 78.61 | 16.85 | 27.38 | 58.84 | 13.61 | 52.0 | 104.0 | 40 | pass | **winner** |
| …::main | global-1x1-float16 | cg96 | 94.402 | -14.9% | 75.60 | 16.11 | 29.21 | 57.14 | 13.41 | 48.0 | 96.0 | 40 | pass | rollback |
| …::main | global-1x1-float16 | cg112 | 93.002 | -16.2% | 76.47 | 16.67 | 28.86 | 52.59 | 13.84 | 56.0 | 104.0 | 40 | pass | rollback |
| …::main | global-1x1-float16 | cg128 | 97.482 | -12.1% | 74.06 | 12.97 | 31.66 | 48.39 | 11.64 | 48.0 | 96.0 | 40 | pass | rollback |
| …::main | spp-6x6-float16 | **current best = round1_gen** | 79.742 | 0.0% | 77.89 | 42.59 | 3.90 | 1.23 | 26.53 | 60.0 | 13.9 | 36 | pass | 保持 |
| …::main | spp-6x6-float16 | **round2_merged** | 79.902 | +0.2% | 77.94 | 42.58 | 3.87 | 1.37 | 26.23 | 60.0 | 13.875 | 36 | pass | **winner（噪声内保持）** |
| …::main | nondiv-7x7-bfloat16 | **current best = round1_gen** | 72.641 | 0.0% | 70.93 | 40.03 | 2.31 | 0.81 | 23.00 | 36.0 | 9.875 | 28 | pass | 保持 |
| …::main | nondiv-7x7-bfloat16 | **round2_merged** | 72.461 | -0.2% | 70.83 | 40.03 | 2.30 | 0.76 | 22.87 | 36.0 | 9.875 | 28 | pass | **winner（噪声内保持）** |

smoke 精度（R1/R2 合并版）：`smoke-1x8x4x4-2x2-float16` L0 pass（max_diff 0.0，idx_exact）——只列精度，不填时延。

#### Iteration Winner（R2）

- winner（global-1x1）: G2'+G10 → `CH_GLOBAL=104` → **90.73 µs**（相对 baseline -37.2%）
- winner（spp / nondiv）: R1 的 general 改动，合并不回退（-0.0% / -0.2%，噪声内）
- new current best: `_round2_merged` + `CH_GLOBAL=104` → 冻结为 `perf_opt/_adaptive_max_pool2d_with_indices_kernel.py`

#### 中止判定：**plateau**

- R2 在 global 上取得 18.2% 有效提升后，`CH_GLOBAL` 邻域（96/112/128）全部落在 ±3–7% 且无一优于 104；
  general 两条 workload 在 R2 无任何 valid 分支超过 3% 噪声阈值。
- **参照结构 diff 检查（T-1 morph ladder，plateau 前强制）**：检索 `pattern-library/cases.md` 参考实现集
  - `examples/TileOPs` 同族（avg / max / max_pool1d_with_indices ×2 / argmin）+ `argmax` 族。逐项结构 diff：
  | 维度 | 参照（同族最优） | 本轮 current best | 差异点是否已实验覆盖 |
  |---|---|---|---|
  | grid 组织 | 一维 persistent（PL-1.21） | 同 | — |
  | 数据流分相位 | 单相位 | 同 | — |
  | 片上驻留 | 全 UB | 同 | — |
  | 值路径布局 | 扁平空间轴宽归约（PL-1.23，max C4） | global 用扁平整平面 + 记录扫描（**双输出下 reduce 路线被 UB 封顶，见 F4**）；general 保留 C 轴向量轴 | ✅ 已覆盖（G7/G8/G9 三个变异全部实测 blocked） |
  | 索引构造 | argmax 族「掩码候选 + 归约」（PL：1D 索引型）/ argmin 的「值+索引打包」 | general 用「破纪录集合取 max」三路累加器（N3/N2）；global 同式在扁平空间轴上 | ✅ 已覆盖（G6 削 op 被 F6 挡；`vselect` 打包候选见 §6） |
  | **写侧生产者** | 参照未显式区分生产者形态 | **本轮新增纪律：store 源必须是连续 op 链**（F1） | ✅ 已覆盖（G1/G2 落实；p1/p7 定量） |
  ⇒ **差异点即全部候选结构变异，均已实验覆盖；未发现未覆盖的可移植结构差异** → 允许进入终止。
- 实验计数：R1 = 3 formal 分支 + 3 探针（blocked 记录）、R2 = 1 merged + 4 扫参、R3 = 1 final ⇒ **formal 分支 9/15**，未超预算。

---

## 5. Autotune Log

未使用 TileLang autotune API：全部候选都是**结构性**变更（布局 / 写侧生产者形态 / 扫描结构 / 分核判据），
参数面只有 `CH_GLOBAL`（64/96/104/112/128）与 `CH_GENERAL` 派生块（spp→48、nondiv→32、smoke→8），
已用等价手动粗搜 + 邻域检查覆盖：

- search space：`CH_GLOBAL ∈ {64,96,104,112,128}`；`CH_GENERAL = 64`（cap，实际块由 `_pick_ch_general` 派生）
- best config：`CH_GLOBAL = 104`、`CH_GENERAL = 64`（派生 spp→48 / nondiv→32）
- 邻域检查：winner 104 两侧 96 / 112 及更远的 64 / 128 均已测（曲线 64→108.9、96→94.4、**104→90.7**、112→93.0、128→97.5）
- winner 的 `msprof op` 独立复测：`phase=final` round 3 → **global 91.922 / spp 79.862 / nondiv 72.601**
  （与 `phase=merged`/candidate 差 ≤1.3%，噪声内一致）
- 编译失败/运行失败配置数：3（`CH_GLOBAL=128` 的 dual-标定形态 UB overflow；`probe_global_reduce` ch≥16 UB overflow；
  `_round1_gscan` 运行期 core dump）
- 未使用 `ab_test.py`：winner 的增益（global -37.2%、nondiv -10.8%）均远超 5% 噪声阈值；
  spp 的 R2 差异（+0.2%）已按「噪声内保持 R1 winner」处理，未引入新结构。

---

## 6. 设计层天花板判定（`[DESIGN_LIMIT]`，结论：**不产出 `perf_feedback.md`**）

对照 `_shared/standards/perf-feedback.md` §1 的双条件：

| 条件 | 判定 | 依据 |
|---|---|---|
| ① 天花板由**算法/设计层**决定（非 tiling/参数可解），且归因到 DESIGN 具体假设 | **不满足** | global 残余缺口（91.9 µs vs 读地板 6.86 / 连续双输出地板 4.73 µs）已定量归因到**工具链 lowering**：① `(C,plane)→(plane,C)` 平面转置被降级为 UB→GM→UB 往返（`UB_to_GM = 104 KB/blk` vs 真实写 4.5 KB，占 ~60 µs，§3.3 F2）；② 双输出整平面 argmax 的归约链 UB 单价 ≈237 B/平面元素（封顶 ch=8 → 52 轮，§3.3 F4）——两者都不是算法/设计选择，DESIGN 选定的算法（单遍在线状态机 + C 轴向量 + 核内转置）恰恰是**本轮实现并采纳**的形态；general 残余缺口归因到「每元素 7 条窄向量 op + vcmp 标量化 + 核占用」，同样是 lowering/发射层 |
| ② 结构性加速估计 > 2x **或**实测与 DESIGN 假设直接矛盾 | **部分成立** | 与 DESIGN §1.6.0「估算下界 ≈1.4 µs」直接矛盾（实测纯拷贝地板即 4.73–6.86 µs，且本 Stage 已取得 global 1.57x / nondiv 1.12x 的结构性改进）；但该矛盾已在 Stage 4 内被消化（扁平整平面 + 写侧生产者纪律 + 单波分派），不构成「不可达」天花板 |

→ **双条件未同时满足 ⇒ 禁止产出 `perf_feedback.md`**（按 §1 末段）。
**先例一致性**：同族 max pool（`489385e`，本日）在同类「工具链 lowering 缺口」下亦**未产出**
`perf_feedback.md`——任务书明确该先例为正确做法。设计估算被实测推翻的部分以**附录级事实**
记录在本日志 §3.2 与 §9 的 pattern-library 回写中；若 conductor 认为 DESIGN §1.6.0 的 roofline 口径
需要正式修订，请按 `perf-feedback.md` §3 路由表决策（本 agent 不回退 Stage 1）。

### 6.1 达标的诚实说明（目标未达成）

- `global-1x1`：**91.92 µs**，校准线 **9.80 µs 未达成**（差 9.4x）。最终 kernel 的读侧已是干净整平面读
  （`GM_to_UB = 52 KB/blk ≈ 真实`），残余代价集中在**平面转置的 UB→GM→UB 往返**（§3.3 F2）——
  与写侧 floor 探针 p7（4.73 µs，连续生产者）相差 19x，全部落在工具链 lowering。
- `spp-6x6` / `nondiv-7x7`：best-effort 未达 general 校准线 12.31 µs（79.86 / 72.60 µs，差 6.5x / 5.9x）；
  其 vec（42.6 / 40.0 µs）由每元素 7 条窄向量 op 构成，而 `T.vcmp` 被编译器标注为「scalar instruction with low efficiency」。
- **报告 `Ratio`（compute-roofline 口径）如实记录、不作达标判据**：本轮**未重采官方 report**，
  下表 Ratio 由 Stage 5 report 的 baseline Ratio 按「同形同式、Ratio ∝ 1/时延」比例换算得到，**已标注为派生值**：

| workload | report baseline Ratio(%) | 换算 Ratio(%) = 旧 × (t_base/t_new) | 说明 |
|---|---:|---:|---|
| global-1x1 | 2.54 | **4.19** | 派生值（2.54 × 151.603/91.922） |
| spp-6x6 | 4.57 | **4.88** | 派生值（4.57 × 85.3617/79.862） |
| nondiv-7x7 | 2.56 | **2.88** | 派生值（2.56 × 81.6016/72.601） |

> 池化族 compute-roofline 上限 ≈9–15%（同族前作结论，**待重验**）；派生 Ratio 仍远低于该带，与
> 「max 是选择类算子、roofline 口径不 ∝ 1/时延」的既有判断一致。

---

## 7. Final Summary

### Final Performance Test Data

`phase=final`，round 3，同一最终候选 `_adaptive_max_pool2d_with_indices_kernel`
（`artifact_sha256 = 8156c03c7c27c5ef46bb6b56dcdab943bb87ed852175484df2d0474afe4f74b0`，与最终文件一致）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py::_adaptive_max_pool2d_with_indices_kernel::main | global-1x1-float16 | 144.393 | 91.922 | _adaptive_max_pool2d_with_indices_kernel |
| tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py::_adaptive_max_pool2d_with_indices_kernel::main | spp-6x6-float16 | 85.572 | 79.862 | _adaptive_max_pool2d_with_indices_kernel |
| tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py::_adaptive_max_pool2d_with_indices_kernel::main | nondiv-7x7-bfloat16 | 81.412 | 72.601 | _adaptive_max_pool2d_with_indices_kernel |

提升：`global-1x1-float16` **-36.3%（1.57x）**；`spp-6x6-float16` **-6.7%**；`nondiv-7x7-bfloat16` **-10.8%**。

- 最终文件：`perf_opt/_adaptive_max_pool2d_with_indices_kernel.py` —— 与全部 `phase=final` 记录的
  `artifact_path`（`_adaptive_max_pool2d_with_indices_kernel.py`，按 gate 4 口径相对 `perf_records.jsonl`
  所在目录解析）/`artifact_sha256` 一致；3 条 final 记录均在同一最终文件上重新采集
  （`profiles/final/round3/adaptive_max_pool2d_with_indices_kernel_b019eac6/*/`）。

### 校准线对账（§2 标定地板 ÷ 0.7）

| workload | 标定锚 | 标定地板(us) | 70% 校准线(us) | final(us) | 达标? | 距线 |
|---|---|---:|---:|---:|---|---:|
| global-1x1 | global 读地板（1 元素保活，CH=64） | 6.860 | **9.800** | 91.922 | ✗ | 9.4x |
| global-1x1（更严参考） | global 双输出连续生产者地板（p7） | 4.730 | 6.757 | 91.922 | ✗ | 13.6x |
| spp-6x6 | general dual 地板（spp 栅格，CH=48，1 波） | 8.620 | **12.314** | 79.862 | ✗ | 6.5x |
| nondiv-7x7 | 同上（**同类近似锚**：nondiv 栅格未单独标定） | 8.620 | 12.314 | 72.601 | ✗ | 5.9x |

### 逐 workload 归因（msprof 证据）

| workload | baseline Task(us) | final Task(us) | 归因（profile 证据） |
|---|---:|---:|---|
| global-1x1 | 144.393（aiv 117.58 / vec 25.92 / mte2 62.60 / mte3 56.26 / GM_to_UB 336 KB / UB_to_GM 96 KB） | 91.922（aiv 79.24 / vec 16.85 / mte2 28.81 / mte3 59.95 / GM_to_UB 52 KB / UB_to_GM 104 KB / BD 40） | 扁平整平面连续读消除**读扇出**（GM_to_UB 336→52 KB/blk，-85%）与逐行转置；vec 25.92→16.85（-35%）；`CH_GLOBAL` 64→104 使轮数 7→5。残余 `mte3 = 59.95 µs`（UB_to_GM 104 KB/blk，真实写 4.5 KB）是**平面转置的 UB→GM→UB 往返**（§3.3 F2），属工具链 lowering |
| spp-6x6 | 85.572（aiv 83.40 / vec 45.53 / mte2 4.59 / mte3 1.43 / scalar 30.96 / BD 24） | 79.862（aiv 77.83 / vec 42.58 / mte2 3.89 / mte3 1.31 / scalar 26.22 / BD 36） | 死代码 `h_flag` 乘积移除（每元素 8→7 op）+ 单波最小 CH 分派（64→48）：逻辑单元 24→36、占用核数 24→36、scalar 30.96→26.22 |
| nondiv-7x7 | 81.412（aiv 79.71 / vec 43.17 / mte2 4.34 / mte3 1.04 / scalar 30.77 / BD 14） | 72.601（aiv 70.87 / vec 40.03 / mte2 2.28 / mte3 0.83 / scalar 22.93 / BD 28） | 同上（CH 64→32）：逻辑单元 14→28、占用核数 14→28、mte2 4.34→2.28、scalar 30.77→22.93 |
| smoke-1x8x4x4-2x2 | （不采性能） | — | 仅精度：`val_exact=True`、`idx_exact=True`（int64 精确 oracle） |

### smoke / skip

| workload_id | kind | 最终处理 | 精度 |
|---|---|---|---|
| `smoke-1x8x4x4-2x2-float16` | smoke | 每次合并后及最终仅做精度回归，无性能记录 | **pass**（`precision_pass: true`；`bench.py --check` 双输出 exact：`max_diff=0.000e+00`、`idx_exact=True`、`idx_dtype=torch.int64`） |
| — | skipped | 无 | — |

### 最终产物自检

- `python perf_opt/_adaptive_max_pool2d_with_indices_kernel.py --level all`：
  **L0 ALL PASS**（7 shapes + 3D + 2 semantic gates + guards + order）/
  **L1 ALL PASS** / L2 全 PASS / Boundary 全 PASS；
  `Summary: worst value max_diff per dtype: bfloat16=0.000e+00, float16=0.000e+00; index mismatches 0/104487`
  （**逐位精确**，atol=rtol=0；索引 `torch.equal` 精确）。
- 三条结构路径（恒等 / 整平面 global / 一般网格）分派**保持在 factory 层**；三条路径在 L0/L1/L2 中均有独立触发用例并通过：
  恒等 = L0-6/L1-identity×2；整平面 global = L0-5/L1-fp16-global-1x1-small-plane；一般网格 = L0-1..L0-4/L1 多数。
  另：`is_global` 额外以 `dtype == "float16"` 门控（bf16 无 `T.vmax` 形态 → 回退一般路径，`L1/bf16/global-1x1` 覆盖）。
- 导出（`__all__`）：`ASCEND_MODE` / `_adaptive_max_pool2d_with_indices_kernel` / `adaptive_bin` /
  `max_adaptive_bin_extent` / `golden_adaptive_max_pool2d_with_indices` / `CH_IDENTITY` / `CH_GLOBAL` / `CH_GENERAL`；
  factory 签名与 baseline 完全一致
  （`(n,c_in,h_in,w_in,out_h,out_w,dtype="float16")` → `build(block_m=None)` → `main(x)->(out,indices)`，`out_idx=[1,2]`）。
- 包标记 `perf_opt/__init__.py`（docstring-only）已建；`perf_opt/` 顶层只放最终 kernel、实验分支、日志、records、
  inventory、accounts、`profiles/`、`logs/` 与 runner/helper 脚本。

### 已知残留（如实披露）

1. **global 路径的平面转置往返未消除**（§3.3 F2）：`UB_to_GM = 104 KB/block` vs 真实写 4.5 KB。
   三条替代路线（G7 整平面 reduce / G8 二维行归约 / G9 免转置列扫描）均实测 blocked（F4/F5/F7）。
   该残留对**正确性零影响**（L0/L1 逐位精确），仅压住性能上限。
2. **`r2_flat16` 的 fp16 累加器依赖「无 bf16 `T.vmax`」这一工具链事实**：已用 `dtype == "float16"` 显式门控
   - 一般路径回退（非静默降级）。
3. **nondiv-7x7 的栅格未单独标定**，其 general 校准线借用 spp 栅格值（§2.3 第 4 条）——已在表内标注。
4. **Ratio 为派生值**（未重采官方 report，§6.1）——已在表内标注口径。
5. 死代码 `h_flag` 的移除使 `_general_bytes`（UB 预算保守估计）**缩小**，守卫仍偏保守（不会导致 UB overflow），
   但估计式与真实 `alloc` 清单的偏差已在本轮更新（见最终文件 `_general_bytes` 注释）。
6. **最终产物的模块 docstring 沿用 Stage 3 版本文本**（其中「C4 整平面路径停用」等描述相对本 Stage 4 实现已过时）：
   `phase=final` 记录写入后对 docstring 的文档性修改会使 `artifact_sha256` 失效（gate 4 对每个 tune workload 只认
   一条 `phase=final`，append-only 又不允许删旧行），故**已回退该文档性修改**以保持 final 记录有效。
   Stage 4 的全部改动、实测结果与负向结论完整记录在**本日志 §3.3 / §4 / §7 / §9**（不依赖 kernel docstring）。

### 总体结论与中止原因

- `global-1x1` 取得 **1.57x** 结构性命中（一般网格逐行扫描 → 扁平整平面连续读 + dtype 宽度扫描 + `CH_GLOBAL=104`）；
  general 以「死代码移除 + 单波最小 CH 分派」取得 spp -6.7% / nondiv -10.8%。
- **中止原因：`plateau`**（R2 在 global 上取得 18.2% 后，`CH_GLOBAL` 邻域与全部剩余结构候选均 ≤3% 或 blocked；
  已完成参照结构 diff 检查）。
- **性能目标未达成，如实报告**（§6.1）；剩余缺口属**工具链 lowering / 发射标量化**层，**不触发** `[DESIGN_LIMIT]`。

---

## 8. Skill Retrospective

### 8.1 skill 流程问题

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| R1 | **copy-floor 方法学在「双输出 / 大索引写」算子上会产出「假地板」，而 skill 未把「写侧生产者形态」列为标定变量**：step-0 的 dual 标定给出 95–111 µs（远超 baseline 的 143.9 µs 的 70% 线 156 µs，会得出「baseline 已达标」的荒谬结论）。真因是**写侧生产者形态**（跨步 gather / transposed）触发 UB→GM→UB 往返，而非字节数。skill Phase 1 的 copy-floor 纪律只要求「同栅格 / 同 DMA 粒度 / 1 元素保活」，未要求对**写侧**做「生产者形态」A/B。 | §2.2 标定表 + 写侧隔离表（p1/p7 4.73 µs vs value/index/dual 94/44/109 µs，UB_to_GM 0.8 vs 48/48/96 KB） | R 类：`PL-1.6-copy-floor` 增「**写侧生产者形态**」维度（连续 op 链 vs 跨步 gather / transposed），并把「标定地板 < baseline 时先怀疑假地板」写成纪律 |
| R2 | **阶段 0 标定脚本对单个 case 的编译失败不鲁棒**：`calib_run.py` 首轮在 ch=128 UB overflow 时直接 `statistics.median([])` 抛异常，**已测的 ch32/ch64 结果全部丢失**（只写内存、末尾统一落盘）。 | §2 标定首轮崩溃日志（`StatisticsError: no median for empty data`） | R 类：标定/实验脚本必须「**逐 case 追加落盘 + 失败 case 记 error 行**」，禁止末尾批量写 |
| R3 | **`vcmp`/`vsel` 的「切片操作数」支持不一致**（`vcmp` 接受切片、`vsel` 的 cond 不接受切片），skill 的证伪协议未把「同一 API 的操作数形态子集」区分开，容易把「某形态编译失败」误记为「该 API 不可用」。 | §3.3 F6（`vsel` 报 `operands' shape are inconsistent`；同 slice 作 `vcmp` 操作数合法），导致 G6 的 8→6 op 削减被误判为 blocked | R 类：`traps-compiler.md` 增条目；skill 的证伪协议补「**操作数形态子集**」这一维（cond/切片/rank） |
| R4 | **「整平面归约 + 索引 pattern」陷阱的适用域需要按「pattern 生产者」细分**：Stage 3 的 `TRAP-npuir-arange-pattern-with-whole-plane-reduce` 结论把整类路径判死，导致 DESIGN 直接把 C4 索引路径停用。本轮实测证明**运行期构建的 pattern 不被腐蚀**（F3），但该路线又被**UB 容量**封顶（F4）。⇒ 陷阱条目需要同时记录「可绕过的部分」与「绕不过的下一道墙」。 | §3.3 F3/F4；`probe_global_reduce.py --check` 逐位精确（16384/16384） | R 类 + D 类：更新该 trap 条目为「生产者特有 + 下一道墙 = 归约链 UB 单价 237 B/元素」 |

### 8.2 价值点（vp_type + 证据三件套）

| id | vp_type | 价值点 | evidence | repro | toolchain_stamp | 回写位置 |
|---|---|---|---|---|---|---|
| VP-1 | **D** | **`UB→GM→UB` 往返的触发条件是「写侧生产者形态」而非字节数**：同一读侧、同一 `(real_c,1,1)` 目标 region 形态下，store 源为连续 op 链（`vbrc`/`vcast`）→ Task 4.73 µs / `UB_to_GM` 0.8 KB/blk；store 源为跨步 UB gather（`flat[:,:,0]`）→ 值-only 94.29 µs / 索引-only 44.16 µs（`UB_to_GM` 48 KB/blk 各）→ dual 109.29 µs（96 KB/blk）。**int64 本身不是问题**（p7 索引+值 = 4.73 µs） | §3.3 F1；`write_probe*.jsonl`；probe_write2/3 原始 profile | `repro/TRAP-store-producer-contiguity.py`（`repro-missing`） | tilelang 0.1.2+67a0441 + CANN 9.0.0 + 910B3 | `pattern-library/traps-compiler.md`（`TRAP-transpose-ub-gm-roundtrip` update：追加「dual 输出 + 生产者形态」判据）+ `constants.md` |
| VP-2 | **D** | **`arange` 陷阱是「生产者特有」**：`T.arange→reshape→vbrc→reshape` 的平面下标 pattern 在整平面 `vcmp(eq)→vselect→reduce_min/max` 链内被腐蚀；换成**运行期构建**的 k 表（`T.serial` + typed `T.vbrc` 标量 + `T.copy` 到 `(plane,)` → `reshape` → `vbrc` 广播）后，同一链的 first-hit / last-NaN 索引**逐元素精确**（16384/16384，含 NaN / 并列 / ±inf / 全 -inf） | §3.3 F3；`probe_global_reduce.py --ch 8 --check` | `repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py`（update：追加「运行期表 = 合法绕法」控制段） | 同上 | `pattern-library/traps-compiler.md`（该条目 update，status 由 applicable-whole-class 收窄为 producer-specific） |
| VP-3 | **D** | **双输出整平面 argmax 的归约链 UB 单价 ≈237 B/平面元素**（含 ×2 multibuffer），把 C4 路线封顶在 `ch=8`：ch=16 需 1 618 432 bit > 1 572 864 bit（UB 196 608 B）。ch=8 → `num_local=52` → 实测 **262.8 µs**（远差于 baseline 144.4）。对照同族 max 单输出 C4 的 32 B/元素（`_UB_REDUCE_B_PER_ELEM`）——**索引侧的 pattern/掩码/候选平面缓冲使单价升到 ~7x**。**另**：`(ch,plane)` 二维 `reduce(dim=1)` 被标量化（scalar 98.1 µs，Task 237.0 µs）——归约轴必须走「扁平空间轴」`(ch,1,plane)+dim=2` | §3.3 F4/F5；`probe_global_reduce` / `probe_global_2d` 原始 profile | `repro/CONST-dual-output-whole-plane-reduce-ub.py`（`repro-missing`） | 同上 | `pattern-library/constants.md`（新条目）+ `traps-compiler.md` |
| VP-4 | **P** | **写侧「连续生产者」纪律（慢→快的判据）**：凡 epilogue store 的源 buffer，其生产者必须是**连续 op 链**（`T.vbrc` / `T.vcast` / `reduce_*` 输出 / 全缓冲 `T.copy`），**禁止**跨步 UB 切片 gather 与「一侧为 1」的 `T.transpose` 输出。落实手法：扫描累加器布局与 store 源同形（3-D `(C,1,1)` 或 2-D `(C,1)`），必要时把转置推迟到最小的 `(1,C)→(C,1)` 上（本轮 global 仍受平面转置所累，见 VP-3） | §3.3 F1/F2；global 144.4→91.9（1.57x） | `repro/PATT-store-producer-contiguity.py`（`repro-missing`） | 同上 | `pattern-library/layout.md`（新 P 条目） |
| VP-5 | **P** | **双输出池化的一般网格：单波最小 CH + 削死代码**：`units = N·out_H·ceil(C/CH)`；取满足 `units ≤ 物理核数` 的最小 `CH`（spp 64→48：BD 24→36、-6.8%；nondiv 64→32：BD 14→28、-10.8%）。**警告**：同族「bool 切片直送 `vselect`」的进一步 op 削减在本工具链**不可行**（`vsel` 拒绝切片 cond），故只有死代码（`h_flag`）可削，8→7 op 而非 8→6 | §3.3 F6；R1 gen 分支；perf_records round1 | `repro/PATT-min-single-wave-ch-dual.py`（`repro-missing`） | 同上 | `pattern-library/layout.md` PL-1.4（update：双输出数据点 + 削 op 的形态边界） |
| VP-6 | **C** | 案例索引：`CASE-pool-adaptivemaxpool2dindices-migration` 的 Stage 4 续档（扁平整平面读 + dtype 宽度扫描 + 写侧生产者纪律 + 单波 CH；C4 整平面 argmax 的正确性可绕过但被 UB 封顶；平面转置往返是本工具链的硬上限） | 本文件全文 + `perf_records.jsonl` | — | 同上 | `pattern-library/cases.md`（更新该 case 的 Stage 4 段） |
| BP_001 | **R** | 建议 `PL-1.6-copy-floor` 增加「**写侧生产者形态**」维度，并写入纪律：「**当标定地板显著低于 baseline 时，先怀疑标定假地板**」（本轮 dual 标定 95–111 µs 意味着 baseline 144.4 已「达标」，与真因矛盾）。 | §8.1 R1 | — | — | proposal（不自动改 skill） |
| BP_002 | **R** | 建议 skill 增加纪律：**标定 / 实验脚本必须逐 case 追加落盘**（单 case 编译失败不得丢弃已完成 case 的测量）。 | §8.1 R2 | — | — | proposal |
| BP_003 | **R** | 建议 skill 的证伪协议补充「**操作数形态子集**」维度：报告「某 API 不可用」前必须穷举（whole-buffer / 切片 / bool cond / rank）子形态，否则会把形态限制误记为 API 缺失。 | §8.1 R3、§3.3 F6 | — | — | proposal |

### 8.3 经验纪律（本轮新增，供后续复用）

1. **写侧先验**：拿到一个「值 + 索引」双输出算子，先做写侧隔离探针（1 变量：store 源的生产者形态），
   再谈优化。**双输出的代价通常不在字节，而在 store 源布局**。
2. **转置定位**：任何由 `T.transpose` 或跨步 UB 切片 feed 的 GM store，先用 `UB_to_GM(KB)` 与真实写字节对账
   （比值 >2x 即判往返），再决定是否重构。
3. **整平面归约容量的三重门**：① arange pattern 腐蚀（可用运行期表绕过）→ ② 归约链 UB 单价（索引侧 ~237 B/元素，
   封顶 ch）→ ③ 轮数（ch 小则轮数爆炸）。三道门都过不了就回退一般网格路径。

---

## 9. 回写 pattern-library（本任务内，ED-C）

| 条目 | 文件 | repro | 状态 |
|---|---|---|---|
| `TRAP-transpose-ub-gm-roundtrip`（update：追加「dual 输出 + 写侧生产者形态」判据与三维对照表，标注本轮工具链 `67a0441`） | `pattern-library/traps-compiler.md` | `repro/TRAP-store-producer-contiguity.py`（`repro-missing`） | 已回写 |
| `TRAP-npuir-arange-pattern-with-whole-plane-reduce`（update：**生产者特有**——运行期构建的 k 表可绕过；并补「下一道墙 = 归约链 UB 单价」） | `pattern-library/traps-compiler.md` | 复用既有 repro + 控制段说明 | 已回写 |
| `TRAP-hivm-vsel-cond-slice`（**新条目**：`hivm.hir.vsel` 的 cond 不接受从更大 UB buffer 切出的 bool 区域；同 slice 作 `vcmp` 操作数合法） | `pattern-library/traps-compiler.md` | `repro/TRAP-hivm-vsel-cond-slice.py`（`repro-missing`） | 已回写 |
| `CONST-dual-output-whole-plane-reduce-ub`（**新条目**：双输出整平面 argmax 归约链 UB 单价 ≈237 B/平面元素；二维行归约标量化数据点） | `pattern-library/constants.md` | `repro/CONST-dual-output-whole-plane-reduce-ub.py`（`repro-missing`） | 已回写 |
| `PL-1.6-copy-floor`（update：写侧生产者形态维度 + 假地板识别纪律） | `pattern-library/elementwise.md` | 复用本轮 `write_probe*.jsonl` | 已回写 |
| `CASE-pool-adaptivemaxpool2dindices-migration`（Stage 4 段更新） | `pattern-library/cases.md` | — | 已回写 |
| `PL-1.4-tiling-heuristic`（update：双输出单波最小 CH 数据点 + 削 op 的形态边界） | `pattern-library/layout.md` | — | 已回写 |

> 终态 `tilelang-skill-evolver` 请跳过上述已回写条目（`repro-missing` 的 3 个最小代码标注 `repro-missing`）。

---

## 10. Compiler Capability Gap（登记到 `.agents/evolution/capability-gaps.md`）

1. **平面转置的 UB→GM→UB 降级**（frontend/bishengir lowering）：`(C,plane)→(plane,C)` 且 `plane` 非向量宽整数倍时，
   转置被降级为 UB→GM→UB 往返；`plane=49`（7×7）必中。影响：所有「空间非对齐平面 + 需要 C 最内轴」的双输出池化。
   量化代价：global-1x1 上 ~60 µs（91.9 中的 60；纯拷贝地板 4.73–6.86）。
2. **双输出整平面 argmax 的归约 UB 单价**（bishengir reduce lowering）：索引侧 pattern/掩码/候选平面缓冲把单价
   从单输出 C4 的 32 B/元素抬到 ≈237 B/元素，封顶 `ch=8`。
3. **`vcmp` 标量化告警**：低效标量化（`hivm.hir.vcmp will execute by scalar instruction`）在每元素 vcmp 上
   直接抬升 scalar time（spp 26.2 µs / nondiv 22.9 µs）。
4. **`hivm.hir.vsel` 的操作数形态限制**（cond 不支持 UB 切片）。

---

## 附：产物清单（本目录）

| 文件 | 说明 |
|---|---|
| `_adaptive_max_pool2d_with_indices_kernel.py` | **最终产物**（最优版本；kernel + 内嵌 golden + L0/L1/L2/Boundary + `main --level`） |
| `__init__.py` | 包标记（docstring-only），供 wrapper 的 perf_opt 分支 import |
| `workload_inventory.json` | workload 清单（4 例：1 smoke + 3 tune） |
| `perf_records.jsonl` | 结构化性能记录（**append-only**，16 行：3 baseline + 3 R1 + 3 merged + 4 扫参 + 3 final；仅 kind=tune） |
| `copy_floor_calibration.jsonl` | 步骤 0 标定 + 写侧探针（10 行，**不混入 perf_records**） |
| `opt_log.md` | 本文件 |
| `bench.py` / `run_experiments.py` / `analyze_profile.py` | 单 workload 采集 / 批量实验 runner / profile 分析 |
| `calib_bench.py` / `calib_run.py` / `probe_write_run.py` / `probe_write2.py` / `probe_write3.py` / `probe_global_reduce.py` / `probe_global_2d.py` | 标定与写侧/归约路线探针 |
| `_adaptive_max_pool2d_with_indices_kernel_round1_flat.py` / `_round1_gen.py` / `_round1_gscan.py` / `_round2_flat16.py` / `_round2_merged.py` / `_round2_cg{96,104,112,128}.py` | 实验分支 |
| `profiles/` | raw `msprof op` 数据（`baseline/ candidate/ merged/ final/ calibration/ write_probe*/ probe/`） |
| `logs/` | 各阶段 stdout/stderr（`final/`、`round1*`、`round2*`、`calib_*`、`probe_*`、`wp*`） |
| `perf_feedback.md` | **不产出**（§6：双条件未同时满足） |
