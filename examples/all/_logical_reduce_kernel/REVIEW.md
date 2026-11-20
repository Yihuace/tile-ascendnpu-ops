# `_logical_reduce_kernel`（`AllFwdOp` / `torch.all` 语义）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/all/_logical_reduce_kernel/DESIGN.md`
- 任务类型: 迁移（TileOPs GPU → TileLang-NPUIR，`scenario=migration` / `migration_mode=harness`）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`（已 Read 全文 475 行核对）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（见「非阻塞改进建议」）

## 机械复核（D-1/D-5）

### D-1：等价性验证重跑（强制）

- 命令：`/workspace/venv/bin/python examples/all/_logical_reduce_kernel/verify_equiv.py`（本环境 `python3` 无 torch，按提示改用 venv 解释器）
- 结果：**exit 0，`ALL EQUIV_PASS`**。执行输出与 DESIGN.md §1.6.1 内嵌结果表**逐行一致**：

  | 组 | 用例行 | 违反数 | verdict | 与内嵌表一致 |
  |----|-------|-------|---------|-------------|
  | A1 补式（¬any[x==0]，ordered EQ）vs 基线（AND[x!=0]） | 1377 | 0 | EQUIV_PASS | ✅ |
  | A1c 判别力对照（ordered-NE 路线，应报差异） | 1377 | **69** | CONTROL_OK | ✅ |
  | A2 实列域 vs 256 对齐 pad 域（pad=+1.0） | 1377 | 0 | EQUIV_PASS | ✅ |
  | A2b 窄行 32B 对齐 pad（W=align_up(N,16)，pad=+1.0） | 1377 | 0 | EQUIV_PASS | ✅ |
  | A3 原生 dtype 比较 vs fp32 上抛（**全 2¹⁶ 位型穷举** × {fp16,bf16}） | 262144 | 0 | EQUIV_PASS | ✅ |
  | A4 分段 AND + f32 running-max 合并 vs 整行（tn=8192/4096/300） | 390 | 0 | EQUIV_PASS | ✅ |
  | A5 微 epilogue 链 `f32 0/1 → f16 → i8` 位精确 | 1377 | 0 | EQUIV_PASS | ✅ |
  | INFO：NaN 无零行 / NaN 混零行 | 1 / 1 | — | INFO | ✅ |

  门禁等价组合计 **268042 行、0 违反** —— 与内嵌表「合计 268042 行 0 项失败」一致。**NaN 分歧对照组核对**（本任务特意点名的检查点）：A1c「ordered-NE 路线」在含 NaN 行报出 **69 处差异**（> 0 ⇒ 判别力成立，脚本自带 `CONTROL_BROKEN` 自检），INFO 行实测 **源(unordered NE)=True / 采纳候选(ordered EQ 补式)=True / ordered-NE 路线=False / `torch.all`=True** —— 采纳候选与 torch/源**三方一致**，被否决的 ordered-NE 路线与两者分歧。即「补式必须是 ordered-EQ 补式、`vcmp("ne")` 直判为语义破裂形态」这一核心结论经机器验证成立。
- **空真守卫核对（逐组）**：A1/A2/A2b/A5 每组比较 1377 行（含 `(1,1)`/`(1,2)` 与非空角点行，元素数 > 0）；A3 逐 2¹⁶ = 262144 位型对 `(x==0)` 掩码与 AND 判决**双口径**比较；A4 每组至少一个 `tn ≤ n` 的分段在跑（`tn>n` 自动 `continue`，`corner16k n=1024` 仍有 `tn=300` 生效）⇒ 390 行非空；A1c 要求 `violations > 0`（否则判 `CONTROL_BROKEN` 且自身失败）。**未发现 0 元素比较导致的空真 PASS 行。**

### D-5：设计期估算机械核算（`design_calc_check.py`）

- 命令：`python3 .agents/tools/design_calc_check.py --design examples/all/_logical_reduce_kernel/DESIGN.md`
- JSON checks 摘要：

  | check | status | detail |
  |-------|--------|--------|
  | `ub_budget` | **skip** | 「no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)」→ **人工已核**（姊妹任务中该 checker 对「元素数/block + B/elem 代理」记账口径解析失灵） |
  | `l0c_budget` | **skip** | 「no block_M x block_N values parsed」→ **不适用**（纯 Vector，无 L0C） |
  | `core_split` | **skip** | 「insufficient parsed dims {}; declared logical cores = None」→ **人工已核**（§5.5 逐负载复算，见下） |
  | `r3_metrics` | **pass** | 四口径（FLOPs / access bytes / scan passes / buffer peak）齐全 |

- **skip 项人工复核结论**：
  - **`ub_budget`（§4.5）**：逐配置独立复算，容量判据 = ① `元素数/block ≤ E_cap`（=8192，等价于最坏 dtype bf16 实测量价 `8192×12 B = 98304 B = UB/2`）且 ② 实测量价 ≤ 硬上限档。**最紧配置** `(256,4096)` bf16 路径 R `bm=2` ⇒ `2×4096=8192` 元素 × 12 B/elem = **96 KB = UB/2（98304 B）** ✓；其余配置 ≤ 64 KB；路径 C `tn=8192` 档 fp16/fp32 64 KB、bf16 96 KB ✓；反例档 fp16 单块 32768 元素 ⇒ 262176 B > 196608 B（硬上限档）⇒ 判据 ① 强制该 shape 走路径 C ✓。**无 UB 溢出**。手工代理公式（`2×elem_bytes+1+4`）= fp16/bf16 9、fp32 13 B/elem，与表内各档数字自洽（唯 §4.5 结论文末一处笔误，见「非阻塞改进建议」#3）。
  - **`l0c_budget`**：不适用——`§5.1` 判定为纯 Vector，无 Cube/L0C/分形；`§4.4` 搬运路径 `GM ↔ UB ↔ fragment`，无 L0 参与。
  - **`core_split`（§5.5）**：独立复算 `num_logical = ceil(M/block_m)`、`num_kernels = min(num_logical, 40)`、`num_local_tasks = ceil(num_logical/num_kernels)`（静态）：
    - smoke `(32,256)` bm=8 → `4`（≤40 ⇒ 4 核，核内 1）
    - mask-validation-4k `(32,4096)` bm=2 → `16`（16 核）
    - mask-validation-32k `(32,32768)` bm=1 路径 C → `32`（32 核，核内 1×4 tile）
    - 3d-multidim `(M=128,N=16384)` bm=1 路径 C → `128`（>40 ⇒ 40 核，核内上界 **4**）
    - `(256,4096)` bm=2 → `128`（40 核，核内 4）
    - `(4,64,512) dim=1` bm=128 扩展阶梯 → `16`；`(4,64,512) dim=0` bm=1024 → `32`
    - `(129,512)` bm=8 → `17`（尾块 `real_m=1`）；`(512,)` M=1 → `1`
    **全部与 §5.2/§5.5 表一致**。物理核数取 **40**（`get_aivector_core_num()` = `get_aicore_num()` 20 × 2，符合 [`core-split-strategy.md` §1][cs]「纯 Vector 算子核数翻倍」），且记录了查询代码与返回值 ✅。核内串行边界为工厂期静态整数。**三要素齐全、数值自洽**。

### 补充机械复核（设计期探针重跑，非门禁项，独立佐证）

`history_version/design_probe_ub.py` 在本环境重跑（`/workspace/venv/bin/python`，compile-only）**exit 0**，独立复现设计的关键未知常数，与 §4.5/§1.6.0 完全一致：

- 单价：fp16 **8.00** / fp32 **8.00** / bf16 **12.00** B/元素（报错文本反解）；单 block 实测通过点 fp16 **24576** / bf16 **16384** / fp32 **16384** 元素（与 §4.5 表逐格一致）。
- 第二节窄行 raw 宽度 `N=64/4/2/300`（4B–600B 行宽）**全部 OK** ⇒ §0.6-R3「实列域无 pad」有独立编译级证据。
- IR 实测链：`hivm.hir.vbrc(0) → vcmp(eq) → vcast(i1→f32) → vreduce<max> reduce_dims=[1] → vsub → vcast(f32→f16) → vcast(f16→i8) → materialize_in_destination` ⇒ §1.4/§3.2 的 v-prefix 向量链**编译级实证**，且每个中间量都被最终输出消费（DCE 免疫）。
- 边界伪像已由设计自述：fp16 `bm=1 tn=32768` 溢出报 `1573632 bits`（196704 B，仅超 96 B）反解为 6.00 B/elem —— 与 §4.5 表注「同档 32768 另测 6.00 B/elem 报 `1573632 bits`」一致，属边界固定开销伪像，不改变硬上限结论。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 已亲自 Read 源算子**源码**全文（`/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`，475 行）+ GPU Op 层（`tileops/ops/reduction/logical_reduce.py`）+ NPU harness Op 层与 wrapper + manifest 段，逐项与 §0 核对（NPU 工件路径的逐条存在性核对见维度 7）。

- **语义理解正确性 — pass**：源工厂由 `op_kind` 分派三种 kind（`any`、`all`、`count_nonzero`）共用同一实现对（源 line 34 `_LOGICAL_REDUCE_KINDS`；分支 line 132–148）。§0.1 正确还原 `all` 分支语义：`bool_vals = ite(x_f32 != 0.0, 1.0, 0.0)`（源 line 129–130）→ `reduce_min`（源 line 137）→ `out_local = cast(result > 0.5, "int8")`（源 line 147–148），即 `AND_j (x_j != 0)`。dtype 语义（kernel 侧 {fp16,bf16,fp32}；bool/int32/int64/complex 由 Op 层 `to_logical_float32` 预转 fp32，源 line 52–62）与输出 **int8 载体**（源 line 102/148/262 三元式）核对一致。**边界语义按 torch 实测行为逐行复核**（不只看源码）：`±0.0`（含 `-0.0`）→ all=False、NaN 无零行 → True、NaN 混零行 → False、±inf/亚正规 → True、空规约维 `N=0` → 每行 True、`N=1` → `(x!=0)`、`M=0` → 空输出 —— 由 `torch 2.9.0+cpu` 实测确认与 §0.1 表逐行吻合。`±0` 语义核对：源 `x != 0.0`（IEEE，`-0 == +0`）⇒ ±0 判零 ⇒ all=False，与 torch 一致。
- **算法解读完整性 — pass**：§0.3 步骤表 1–6 覆盖源基础工厂**全部计算语句**——工厂常量（源 line 93–95）→ grid（line 104）→ `_needs_pad` 掩码加载（line 113–119）/ `T.copy`+cast（line 120–126）→ 指示化（line 129–130）→ `reduce_min`（line 137）→ int8 出栈（line 147–148）→ 写回（line 151）；无遗漏、无臆造。数据流图覆盖全部 buffer；host 侧逻辑（NPU `reduce.py` 的 `_prepare_input`：`movedim → contiguous → reshape(M,N)`，`_kernel_handles_padding=True` ⇒ 跳过 `F.pad`，实读 line 438–494 核对）逐条列出。**源缺陷识别准确**：源步骤 2b/6 的两处 `T.copy`（line 122/151）无 `M % block_m` 守卫（harness 用例 `(129,512)` 配 `bm=8` ⇒ `129%8=1` 即越界），§0.4 显式标注并要求 Stage 3 不得照抄。
- **优化手段覆盖完整性 — pass**：§0.4 列出源码 11 项优化性结构，逐项对照源码语句核对存在：SMEM tiling（line 105/122）、256 pad+中性掩码（line 93/113–119）、指示化复用 `reduce_min`（line 129/137）、全量 fp32 上抛（line 117/126）、`block_m` 48KB 启发式（line 393）、`threads`（line 98/104）、`lru_cache`（line 75）、N-tiled 回退工厂（line 166/354）、结果块写（line 151）、autotune（line 415）、重 pad 布局约束（`_primitives.compute_tile_n` 的 `num_buffers×block_m×tile_n×elem ≤ budget`，源 line 121）。**未把实现层优化误判为算法层**（或反之）；确认源为朴素「SMEM + 指示化 + 单次归约」形态，无 warp shuffle/mma/swizzle/异步流水。
- **耦合性判定合理性 — pass**：§0.5 每条目有**四态处置 + 依据**，与 GPU→NPU 能力映射不矛盾：`threads`（#6）判**舍弃**（NPU 无 CUDA 线程，harness K9 已去除）；SMEM tiling（#1）判**等价替换**为 UB；256 pad（#2）判**舍弃**（NPU 实列域计算，A2/A2b 中性性证明 + 探针 raw 宽度可编译）；fp32 上抛（#4）判**舍弃**（A3 全 2¹⁶ 位型穷举）；`block_m` 启发式（#5/#6）判**重新设计**为元素预算阶梯；**N-tiled 回退工厂（#8）判「舍弃（作为迁移对象）」并显式区分「GPU fragment 列上限触发」与「NPU UB 容量触发」的判据/缓冲组/尾块策略**（正是 prompt 强调的核对点），且其「运行合并」的在线意图由 §0.6-R4 承接（合并算子由 min 改 max）。无「舍弃项无理由」、无「源已识别的优化手段被静默丢弃」（§1.6.0 末「源算法优化手段承接」逐项 #1–#11 说明）。
- **NPU 重设计可行性 — pass**：§0.6 R1–R6 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（De Morgan 补式 + ordered EQ）经 A1/A5 + A1c 对照实证；R2 一维 persistent `T.Kernel(num_kernels, is_npu=True)`（无三维 Kernel）；R3 实列域无 pad（A2/A2b + 探针）；R4 UB 容量驱动列分块 + **运行 max 合并**（明确「照抄源 tiled 的 `T.min` 在补式表示下语义翻错」）；R5 元素预算阶梯；R6 `real_m = T.min(...)` + src/dst 双显式 slice 尾块三件套。均满足 NPU 硬件约束（一维 Kernel、内存层级、容量、静态边界）。
- **设计与决策一致性 — pass**：§0.5 舍弃 pad → §4.4/§5.3/§5.4 全链无 pad；§0.5 舍弃 `threads` → §5.5 一维 persistent、无 `threads`；§0.6-R1 → §1.3/§1.4/§3.1/§6 均以补式（`1 − max[x==0]`）为准；§0.6-R4 → §5.2 路径 C；§0.6-R6 → §3.2/§5.4 尾块守卫。**无「§0 判重设计、§3 照抄源方案」脱节**。
- **golden 独立性 — pass**：§8.1 golden = `x.bool().all(dim=...)`（`torch.all` 权威 API），以 §0.1 语义为唯一依据，**未复刻** §0.6 的 NPU 算法（无补式/无指示平面/无列分块）；`verify_equiv.py` 为纯 torch CPU + fp64 参照、无 NPU 依赖，与 kernel 不同源。harness `tests/ops/test_all.py` 的 `ref_program = x.bool().all(dim=-1)` 与 `_exact_compare`（`torch.equal` + 断言 dtype==bool）同向核对一致。
- **host/op 层引用路径真实性 — pass**：§0/§0.7/§3.5.3 引用的全部路径逐条 `ls` 核对存在，**无失效引用**（GPU 源工厂 / GPU Op 层 / `_primitives.py` 以及 NPU harness 各工件、同族参考；逐条路径清单见维度 7「引用路径真实」项）。
- **契约核对（家族致命点）— pass**：manifest `AllFwdOp` 段实读 —— `outputs.output.dtype = "bool"`（line 232）、`params` 含 **`dim` 与 `keepdim`**（line 234–235；`keepdim` 默认 false）、workloads 为 `[32,256]`/`[32,4096]`/`[32,32768]`/`[4,128,4096] dim=[0,2]`。§0.2 表与 §4.2/§5.6 均按 **bool 输出（kernel 载体 int8）** 设计，**未误抄姊妹 count_nonzero 的 int64 契约**（§11 差异表六项「不可互相借用」逐条列出，含输出 dtype / 归约算子 / 补式被减数 / 合并算子 / 出栈 cast 链 / UB 单价）。**kind 分派核对**：本迁移单元只驱动 `op_kind` 的 `all` 分支（§0.2「仅 `all`；`any` 与 `count_nonzero` 为独立迁移单元」，manifest 的 `AllFwdOp.source.kernel` 指向逐 op 包，路径见维度 7），且 §5.6 保留 `"int8" if op_kind != "count_nonzero" else "int64"` 三元式以维持共享工厂兼容——范围无蔓延。**roofline kind 陷阱核对**：`ops/reduction/reduce.py` 的 `"all"`（line 56/376）确为 `eval_roofline` 的记账分支，非实现入口；§0.1 显式声明「不据其实现」✅。
- **多输出/变体函数的迁移范围 — pass**：源含两套工厂/两套 custom_op wrapper（`_logical_reduce_fwd_wrapped` line 249 / `_logical_reduce_tiled_fwd_wrapped` line 265）与 `register_fake`（line 284/291）。§0.5-#8 + manifest 注释（line 268–270「only the baseline TileLang factory (the N-tiled factory is out of scope)」）一致声明 **tiled 变体排除**，并说明「不迁移该工厂 ≠ 其能力动机不迁移」（NPU 侧列分块由 UB 容量驱动、触发条件与结构均不同）。无 indices 变体（该 op 无）。✅

### 1. API 可行性 — pass

- §3.2/§3.4 列出的每条 API 均在本仓有双证（`docs/Tilelang.language/` + `examples/`/家族先例），且核心链经**探针编译级实证**（IR dump 见机械复核段）：
  - `T.vcmp(A,B,C,"eq")`：`docs/Tilelang.language/比较操作/T.vcmp.md`（dtype 表 fp16/bf16/fp32 ✓，line 15；输出 bool 不可落 GM line 39）✓
  - `T.vcast`：`数据类型转换操作/T.vcast.md` —— `f32→f16`（line 30）、`f16→i8`（line 39）、`i1→f32`（line 56）均存在；**`f32→i8` 无表项**（全表无该行）⇒ 设计走两步，claim 正确 ✓
  - `T.reduce_max(...,dim=1,clear=True)`：`规约操作/T.reduce_max.md` + `T.reduce.md` §2.2.1（fp16/fp32 ✓、**bf16 ✗**）⇒ 设计归约域恒为 f32 指示平面，规避 bf16 限制 ✓
  - `T.vsub`/`T.vmax`：`数学操作/T.vsub.md`（fp16/fp32 ✓）、`比较操作/T.vmax.md`（fp32 ✓）✓
  - `T.reshape`、`T.copy`（显式 slice）、`T.fill`（须 `tir.const(v,dtype)`）、`T.Kernel(is_npu=True)`+`T.serial`+动态 guard：均有文档 + 家族先例 ✓
- **v-prefix 优先**：全链 `vcmp/vcast/vsub/vmax`，无 `npuir_xxx`。**模式匹配**：Developer 模式自动同步（§7），未写手动 `set_flag/wait_flag` ✓
- 负向 API 论断核对：`T.vnot` dtype 表 `bool/int1 = ×`（`逻辑操作/T.vnot.md` §2.2.1）—— per 亲自打开文档核对成立，故「取反不可行、必须走补式」的论证前提正确 ✓。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：`GM → UB → fragment`，无越级（无 GM→L0 直连）；纯 Vector 无 L1/L0 参与 ✓
- UB 预算（§4.5）：逐配置独立复算 ≤ 192 KB（见机械复核段「人工复核结论」），三层口径（手工代理 / 实测量价 / 硬上限）+ 退化路径补行（`dim=None` flatten M=1 大 numel 走路径 C、每 tile 驻留恒定）✓
- 每个 buffer 标 UB/fragment（§4.3）✓；无 Cube，L0C 不适用 ✓

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：元素预算阶梯 + 双路径（R/C），逐 workload 表给出 `bm/tn` 及算式（`p·N ≤ E_cap` 等），抽查全部自洽 ✓
- 对齐约束（§5.3）：`T.copy` slice 天然满足 32 B 尾轴；**免宽度 pad**（探针实测 raw N=2/4/64/300 可编译）✓
- 非整除处理（§5.4）：M 尾块 `real_m=T.min` + src/dst 双 slice + 垃圾行丢弃；N 免 pad；路径 C 尾块独立静态宽度代码块（避免 serial 变量条件 segfault）✓
- **分核策略三要素（§5.5）齐全、数值自洽**（见机械复核段）：① 逻辑核数 `ceil(M/block_m)`；② 物理核数 **实查记录**（`NPUUtils.get().get_aicore_num()`=20 与 `get_aivector_core_num()`=40；纯 Vector 用 40；`ub_cap`=196608 同源实查）；③ 规模判定与方案（逻辑 ≤ 物理→无需适配；极大规模→固定 40 核 + 核内 `T.serial` 静态上界 + grid-stride + guard）。核内串行边界全为工厂期静态值。与 [`core-split-strategy.md` §1/§2.2][cs] 一致 ✓

[cs]: ../../.agents/skills/_shared/standards/core-split-strategy.md

### 4. 技术约束检测 — pass

无三维 Kernel、无 GEMM/L0C/分形；UB 容量已核算；`reduce` 全程 f32（bf16 仅出现在 `vcmp` 比较域）；`f32→i8` 无表项以两步规避；`T.fill` 类型化常量；`vcmp("eq")` 而非 `ne`；原位 carried-state 陷阱以 no-alias 规避；`prim_func` 内不用解析期 `if` 定义 buffer（结构性条件提到工厂层）—— §3.5.1 逐项有处理 ✓

### 5. 循环与同步 — pass

- §6.1 明确 `T.serial`（M persistent + 路径 C tile loop）与向量化原语选择；无 `T.Parallel` 融合 `ite`（规避 `TRAP-parallel2d-bm-ge2-scalarize`）✓
- §7 同步策略与 Developer 模式一致（自动同步，无跨核/CV/跨引擎依赖）✓
- §6.4 尾块逻辑（M 尾块 / N 免 pad / 路径 C 静态尾块 / 静态化 guard 边界）✓

### 6. 验证方案 — pass

- §8.1 golden 草案（`torch.all` 语义、`x.bool().all(dim)`；两入口：通用与 2D+movedim）✓
- **判别性输入纪律**（§8.1 末）正确指出 harness `_make_logical_input`（`workloads/reduction.py`，实读：仅强制首行全零 / 次行全非零）之外还须补 30% 置零 + 角点（NaN 无零行/混零行、±0、±inf、亚正规）—— 否则 `randn` 零密度≈0 会令恒等实现也通过 ✓
- §8.2 L0 计划：L0-1 路径 R 代表 shape、L0-2 判别性角点（**显式要求 NaN 混零行=False / NaN 无零行=True，否则判 ordered-NE 误用**）、L0-3 路径 C、L0-4 分核边界、L0-5 未向量化点复核 ✓
- 精度标准：bool 输出**逐位精确比对**（`torch.equal`），明确容差不适用（与 harness `_exact_compare` 一致）✓

### 7. 完整性与一致性 — pass

- 迁移类 §0（0.1–0.7 齐全）+ §1–§11 齐全；占位符模板标记扫描（未完成标记 / 花括号占位符）**0 命中** ✓
- **引用路径真实（逐条 `ls` 核对存在）**：GPU 侧 `kernels/reduction/logical_reduce.py`、`ops/reduction/logical_reduce.py`、`kernels/reduction/_primitives.py`；NPU harness 侧 `kernels/reduction/all/all.py`、`ops/reduction/all.py`、`ops/reduction/reduce.py`、`manifest/reduction.yaml`、`tests/ops/test_all.py`、`benchmarks/ops/bench_all.py`、`workloads/reduction.py`；同族 `examples/count_nonzero/_logical_reduce_kernel/{DESIGN.md,_logical_reduce_kernel.py,verify_equiv.py}`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_{single,tiled}`、`examples/argmin/_argreduce_kernel/{DESIGN.md,verify_equiv.py}`。✓
- API 映射（§3）、内存规划（§4）、Tiling（§5）三处相互一致（`E_cap`/`tn`/`bm`/路径判定同源）✓

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 §1.6.0 调研、§1.6.1 等价优化、§1.6.2 向量化替代、§1.6.3 布局决策（另 §1.6.4 实验裁决三件套）✓
- **算法调研存在性与完整性 — pass**：属**规约类**→完整调研。R1 候选表含**基线（GPU 源，行 0）** + C1–C10 共 11 项（每项等价性初判 + 收益方向）；R2 有**明确结论**「有，且本算子天然在线」，结构依据为「逐元素判决无跨元素依赖 + running-max 合并算子」+ 源 tiled 工厂 `acc = T.min(acc, tile_acc)` 为「源算法族存在在线变体」的直接源码证据（源 line 218）；R3 **四口径**表（FLOPs / GM 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径写清含/不含）；R4 逐候选 6 维评估（单元匹配/容量/对齐/静态边界/流水/跨核）。调研范围写明（`algorithm-candidates.md` ALG-reduction/ALG-elementwise 命中行 + `kb_search` 预注入 5 条 + `examples/` + pattern-library + 恒等变形空间）✓
- **调研结论正确性（独立复核）— pass**：
  - ① **等价性推演**：主选 C1 `y = 1 − max[x==0]` 与原 `AND[x!=0]` 由 De Morgan + 行内划分恒等成立；有序 EQ 的 IEEE 正确性经 `TRAP-vcmp-ne-ordered-nan`（**已亲自打开**，line 349「`eq`/`le`/`ge` 与 IEEE 一致」）与 A1/A1c 机器验证（69 处对照差异）双重确认 ✓
  - ② **复杂度复算**：独立复算 R3 各行——基线 `2·N_padded` FLOPs / `N_padded·elem+1` Bytes / 1 遍 / 双 4 B/elem 平面；C1 `2N` / `N·elem+1` / 1 遍；**C9 正确计入 workspace GM 往返** `M·nchunk·(1+1)B × 2`（未漏计中间缓冲 GM 流量，本行即该检查点）。数字自洽 ✓
  - ③ **负向断言复核**：C2「`vcmp("ne")` 语义相反」→ 对照 `TRAP-vcmp-ne-ordered-nan` verified 条目（含 repro，2026-10-16 第二数据点）+ A1c 实证 ✓；C3「`reduce_min` NaN 传播未文档化」→ 打开 `T.reduce_max.md` 核对（§2.3 特殊限制「无」、全篇无 NaN 条款）✓；C5「单价未实测」→ 诚实标注（未实测假设，登记 Stage 4 备选）✓；C9「核利用率已 ≥ 80%」→ manifest M ≥ 32（smoke/M=32、3d M=128）核实 ✓；C10 O(N log N) ✓；`T.vnot` bool/int1 × → 打开文档核对 ✓。**对照 `algorithm-candidates.md` ALG-reduction R1–R4 命中行未发现被漏掉的已知候选**（Welford 对布尔 AND 无关；分块+合并/两阶段/扫描遍数/水平vs垂直 lane 均已评估）✓
  - ④ **亲和性淘汰均引佐证**：`TRAP-parallel2d-bm-ge2-scalarize`（含 2026-10-15 适用域收窄，已核对 INDEX line 85）、`TRAP-vselect-inplace-carried-state`、`CONST-vector-launch-overhead`、`CONST-msprof-task-floor-910B3`（同设备）、`CONST-reduce-ub-per-element`（**适用域限定为 3-D 整平面 `(CH,1,H*W)+dim=2`，已亲自核对 constants.md line 339–350**；本设计 2-D 行归约以本任务探针独立标定 8/12 B/elem）✓
  - ⑤ **并列候选（判定裕度落在未实证常数）未纸面拍板**：§1.6.4 给出**主选 + 备选 R′ 完整结构（含 buffer 形状/预算/循环/分核三要素）+ 判定阈值（< 0.8×）+ 代表 shape + 回写路径**三件套，且明确「按 shape 分派、非全局二选一」✓
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构均与 §1.6.0 选定 C1 一致（无「调研选 A 设计做 B」）；迁移侧 §0.5/§0.6 处置与调研选族一致；§1.6.0「源算法优化手段承接」逐项 #1–#11 说明，**无静默丢弃** ✓
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法 C1 系统审视，采纳 A1/A2/A2b/A3/A4/A5，每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素 ✓
- **等价性论证独立推演 — pass**：A1 De Morgan 恒等、A2/A2b pad=+1.0 中性（`max_{N_padded}[x==0] = max_N[x==0]`，因 pad 列指示量恒 0）、A3 fp16/bf16→fp32 精确单射（保 ±0/NaN）、A4 分段覆盖恰为 [0,N) + max 结合/交换/幂等、A5 `1−max ∈ {0.0,1.0}` 精确 + `f32→f16→i8` 位精确 —— 逐项独立推演成立，且与 **D-1 重跑结果一致** ✓
- **向量化替代完整性 — pass**：§1.6.2 覆盖 §1.4 两条路径**全部**循环/标量点（#1–#11），与 §3.3 伪代码、§6 循环结构交叉核对无遗漏；源热点（源步骤 3 的 `ite` 融合在 `bm≥2` 标量化、源步骤 5 的 `T.Parallel(bm)` 逐元素 cast）均已由 #1/#2/#8 的向量链替代 ✓
- **不可替代理由充分性 — pass**：#9 任务 guard（block 级动态边界）、#10 tile 循环控制（`acc` 循环携带状态 + 静态上界要求）、#11 host 元数据（工厂期计算）—— 理由具体到本算子、非「实现简单/照源码写」✓
- **替代 API 可行性 — pass**：替代 API 均有 `docs/`/`examples/` 佐证（与维度 1 同口径），且探针编译实证 ✓
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按**规约类必选候选**枚举（水平归约 #1 vs 垂直扫描 #2，正交 lane 映射分别评估）；逐候选量化评分（整除性/尾 lane 浪费率/累加链形态/repack 代价/UB 影响）；弃选有**量化理由**（#2 发射爆炸量化、#3 纯开销零收益并**明确警惕「不得把核内重排写成 GM 级重排的稻草人」**、#6 全宽补式 +1 平面、#8 收益面窄 + workspace 代价）；**明确对照源码隐式轴选择**（「GPU 源轴是输入而非结论」）。**弃选论证前提已亲自核对**：`创建操作/T.transpose.md` §2.3 证实 `T.transpose` 底层 `hivm.hir.vtranspose` **仅支持核内 UB 级二轴交换链**（不是 GM 级）——设计 §1.6.3-#3 的前提与文档一致，未与文档/pattern-library 实测矛盾 ✓
- **实验裁决模式完整性 — pass**：§1.6.4 三件套齐备（主选逐条落地 / 备选 R′ 结构完整可 Stage 4 直接实现 / 裁决计划含代表 shape、`Task Duration` 指标、`< 0.8×` 阈值、回写路径）；未实证常数（单消费者超预算竞态、发射单价双档）已显式标注来源与不确定区间；shape 特化分派可行性已评估（工厂按 `(M,N,dtype)` 分派）✓
- **布局决策与 §3.3/§4/§6 一致 — pass**：选定轴 = N 轴（`reduce_max dim=1`）、核内布局 = 行块 `(bm,·)` 行主序驻留（UB staging）、repack 无；与 §3.3 buffer 形状、§4 UB 预算、§6 循环结构一致（累加循环内层向量维 = N）✓
- **编译期特化谓词 — n/a/ pass**：本设计的编译期分支均为 Python 工厂期静态量（`N <= E_CAP`、`tail > 0`、路径 R/C 选择），不涉及窗口类「覆盖集合」充分条件谓词；§3.3 注明确「结构性条件一律提到工厂层，按变体返回独立 prim_func」✓
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 不含 §1.6.2 未论证的逐元素标量循环 ✓

## 非阻塞改进建议

> 以下为**建议级**（warn）观察，不阻塞结论；供 Stage 1 revision 或后续 Stage 参考。

1. **`MAX_SINGLE_TILE_COLS` 取值标注**（DESIGN.md §0.4 line 154 / §0.5 line 179 / §0.6-R4 line 258）：
   设计将源 tiled 触发阈值记为 `MAX_SINGLE_TILE_COLS(32768)`。源 `_primitives.py` 的实际常量为 **`MAX_SINGLE_TILE_COLS = 32512`**（32768 出自源 `logical_reduce.py` docstring line 169–170 的表述）。建议改为「`MAX_SINGLE_TILE_COLS = 32512`（源 docstring 记 32768）」。**影响评估**：仅涉及**未迁移**的 tiled 变体文字，不改变任何设计决策。

2. **§1.6.3 轴质量评分表的探针证据范围过宽**（DESIGN.md §1.6.3「轴质量评分—整除性」行）：
   该行写「探针证 raw N=2/4/64/300/**4096/16384/32768** 均可编译」。据本任务探针重跑，**`tn=32768` 在 bm=1 下 fp16/bf16/fp32 均 UB 溢出**（`requires 2097408 bits while 1572864 available`）；探针第二节实际覆盖的 raw 宽度为 N=64/4/2/300。建议改为「raw 窄行 N=2/4/64/300（4B–600B）可编译；`N > 硬上限`（fp16 24576 / bf16·fp32 16384）由路径 C 列分块承接」。**影响评估**：不改变决策（容量模型 §4.5 已正确拒绝 N>硬上限的单块驻留）；属证据表述精度问题。

3. **§4.5 结论文末的手工代理数字与本节公式不一致**（DESIGN.md §4.5「口径纪律」③）：
   该处写「fp16 档手工代理（约 **36 KB/8192 元素 ≈ 4.5 B/elem**）远小于实测量价（8 B/elem）」，但 §4.5 的代理公式（`2×elem_bytes+1+4`）与逐配置表对 fp16 给的是 **9 B/elem**（8192 元素 = **73.7 KB**）。两处相差 2 倍（`36.9 KB` 恰为表中 `(4,64,512) dim=0` 档 `9×4096` 之值，疑为笔误）。建议统一为 `9 B/elem（73.7 KB）` 或改写论证方向。**影响评估**：容量判据一律以**实测量价 + E_cap** 为准（§4.5 已声明），该代理数字非判据，不改变任何结论。

4. **§0.1 边界表 `N=0` 行的处置为风险登记而非解析**（DESIGN.md §0.1 边界表 / §9.2-R5）：
   `torch.all` 对 `N=0` **返回 True（每行空 AND 恒真，实测 `torch 2.9.0+cpu` 确认）**；本设计在工厂期 `assert N >= 1`。二者在小 `N=0` 负载上会**分歧**（torch 给 True，kernel 报错）。设计已登记为风险（§9.2-R5「建议 Op 层短路」），但未在 §5.6/§8 给出 Op 层短路落点。建议在 §5.6 集成前置核对表补一行「`N=0`：Op 层短路返回全 True 或明确声明不支持」，或说明为何 `N=0` 不在迁移范围内。**影响评估**：`N=0` 不在 manifest/测试 workload 内，非阻塞；仅在未来 harness 触发 `N=0` 时需同步。

5. **§0.2 R3「实测单价」表注与硬上限推导的口径说明可再澄清**（DESIGN.md §4.5 表 / §1.6.0-R3）：
   fp16 `bm=1 tn=32768` 溢出反解为 **6.00 B/elem**（`1573632 bits` = 196704 B，仅超 UB 96 B），与该 dtype 主档 8.00 B/elem 不同 —— 即「B/elem」在贴近硬上限处含**边界固定开销**，非严格线性。设计已在表注登记该值，但 §1.6.0-R3 的「8/12 B/elem」表述会令读者误以为是严格线性律。建议在 §4.5 表注补一句「单价为线性区间近似；贴硬上限处含 ~96 B 固定开销伪像。故硬上限取**实测通过点**（24576/16384/16384）而非纯公式 `UB//单价`」。**影响评估**：硬上限本就取实测通过点（§4.5/§5.2 `HARD_CAP` 已是实测值），属表述澄清。

---

### 检视过程说明（证据链与防护）

- 迁移维度 0 的证据来自**亲自 Read 源算子全文**（`/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`，475 行）+ GPU Op 层 + NPU 侧三工件（`all/all.py` wrapper / `ops/reduction/all.py` / `reduce.py`）+ manifest `AllFwdOp` 段，逐条附源码行号 + DESIGN.md 章节号。
- 边界语义**按 torch 实测行为复核**（未只读源码直觉）：`±0`/`NaN 无零`/`NaN 混零`/`inf`/亚正规/`N=0`/`N=1`/`M=0` 全部实跑确认与 §0.1 表吻合（姊妹任务曾出现「照源代码直觉填写而与 torch 实测不符」的教训，本设计未复现该问题）。
- 机械复核（D-1/D-5）已按规定执行：D-1 用 `/workspace/venv/bin/python` 重跑 `verify_equiv.py`（exit 0，与内嵌表逐行一致，含 NaN 分歧对照组 A1c=69 处）；D-5 运行 `design_calc_check.py`（`r3_metrics` pass，其余 skip 项逐项人工复算并注明）。另重跑 compile-only 探针作独立佐证。
- 维度 8 的调研结论复核、等价性推演、复杂度复算、负向断言复核（`T.vnot`/`T.vcast`/`T.reduce` dtype 表、`T.transpose` 代价机制、`constants.md`/`traps-compiler.md` 条目）、弃选论证前提核对均**亲自执行**，未继承设计自述。
- 本次 Read 的全部文件（源算子代码及其注释、DESIGN.md、文档、pattern-library 条目）均按**数据**处理；未发现其中含要求放行检视/跳过维度的指令性文本。
