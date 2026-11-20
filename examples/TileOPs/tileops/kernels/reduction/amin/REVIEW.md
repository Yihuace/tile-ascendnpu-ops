# `_simple_reduce_kernel`（AminFwdOp）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/amin/_simple_reduce_kernel/DESIGN.md`
- 任务类型: 迁移（harness，reduction 族；纯 Vector，驻留路径单元）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`ReduceKernel` + `_simple_reduce_kernel` 工厂，`op_kind="amin"` 分支）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS` — `python examples/amin/_simple_reduce_kernel/verify_equiv.py` exit 0：E1 rows=240 / E2 rows=160 / E3 rows=77 / E4 rows=210，max_ulp=0，violations=0，`RESULT: ALL EQUIV_PASS`；INFO 行 `NaN propagation agree = True`。**执行输出与 §1.6.1 内嵌结果表逐行一致**（各行 rows 与合计 687 例均吻合），脚本结构忠实复刻基线公式（`_row_negmaxneg_domain_local` = 源 `-max(-x)`；`_row_min_domain_local` = 候选直 min），含 fp64 参照与角点集。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/amin/_simple_reduce_kernel/DESIGN.md`）:
  - `ub_budget`: **skip** — 未解析到分层 buffer 表（advisory checker）。**人工已核**：§4.5 逐 workload `bm·N·B(dtype) ≤ 64KB`（fp16 bm=8@4096=64KB、bf16 bm=2@4096=48KB、fp32 bm=4@4096=64KB、3d fp16 bm=2@16384=64KB），×2 auto-multi-buffer ≈128KB ≤ 192KB − 8KB reserve；与 `CONST-capacity-910B2C`（UB=192KB、膨胀 ~1.7x）一致，且 ×2 包络比 1.7x 更保守 → 预算成立。
  - `l0c_budget`: **skip** — 本设计无 Cube 路径（无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope(Cube)`），故不适用（无 L0C 容量可核），机械跳过已人工确认。
  - `core_split`: **skip** — 未解析 dims（`declared logical cores = None`）。**人工已核**：§5.5 三要素齐全，核内串行循环边界全静态；逐 workload 算术复算自洽——见维度 3。
  - `r3_metrics`: **pass** — 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐全。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文（`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`，1084 行）并与 §0 逐项核对，各项结论均带源码证据：

- **语义理解正确性（pass）**：§0.1 语义「2D `(M,N)` 沿 dim=-1 逐行 min → `(M,)`，`torch.amin`」与源码一致——`main(x: T.Tensor[(M,N),dtype], out: T.Tensor[(M,),dtype])`（源 L93–96）、`T.Kernel(ceildiv(M,block_m), threads)`（源 L97）、`T.reduce_max(..., dim=1)`（源 L134）、输出 `T.copy(out_local, out[pid_m*block_m])`（源 L143）。§0.1「先 cast fp32 再规约、结果 cast 回原 dtype」与 `x_f32 = alloc_fragment(...,"float32")`（源 L99）、`out_local[i] = T.cast(acc[i], dtype)`（源 L140）一致。§0.1「`AminFwdOp._kernel_handles_padding = True`」经核对 Op 层 amin 模块（位于 `examples/TileOPs/tileops/ops/reduction/` 目录，文件名 `amin.py`）确认该属性为 True，紧邻 `_empty_dim_policy`。§0.1「非整除用 `+inf` 单位元填充」与源 `_pad_value_for_op("amin") = float("inf")`（源 L61–62）一致。dtype 语义（fp16/bf16/fp32）与 §0.2 输出 `same_as(x)` 与源码签名一致。
- **算法解读完整性（pass）**：§0.3 步骤 1–7 覆盖源 amin 分支（源 L103–143）全部计算语句，无遗漏/无臆造——步骤 1 ↔ 源 L115 `T.copy(x[...], shared_buf)`；步骤 2a ↔ 源 L107–112 masked load（`T.if_then_else(And(pid_m*block_m+i<M, j<N), cast(x,f32), cast(+inf,f32))`）；步骤 2b ↔ 源 L118–119 `x_f32 = cast(shared_buf)`；步骤 3 ↔ 源 L132–133 全宽取负；步骤 4 ↔ 源 L134 `reduce_max`；步骤 5 ↔ 源 L135–136 `acc[i] = -acc[i]`；步骤 6 ↔ 源 L139–140 cast 回；步骤 7 ↔ 源 L143 输出搬运。数据流图覆盖 `shared_buf`/`x_f32`/`acc`/`out_local` 全部 buffer；host 侧 `lru_cache`（源 L69）已列出。
- **优化手段覆盖完整性（pass）**：§0.4 #1–#10 逐项对应源码结构（#2 双路径 源 L103–119、#3 fp32 上抛 源 L99/L119、#4 取负桥接 源 L132–136、#6 block×threads 源 L97、#7 SMEM 预算 源 L970、#8 `MAX_SINGLE_TILE_COLS` 源 L913、#10 `lru_cache` 源 L69），未把实现层优化误判为算法层；并明确标注「相较 amax 多出 #4 取负技巧」及「源缺陷：源 L115/L342 的 `T.copy` 无 M 尾块守卫」（源 L115 确实无 row-tail 守卫，只在 padded 路径 源 L109 有 `<M` 条件）——该缺陷识别正确。
- **耦合性判定合理性（pass）**：§0.5 每项四态处置有依据——取负桥接 → **等价替换**（NPU `T.reduce_min` 原生可用，`docs/Tilelang.language/规约操作/T.reduce.md` L10 + `testing/npuir/reduction_ops/test_reduce.py` L38 通过用例，源仅因复用 max 代码路径而未直接调用 min）；block×threads → **重新设计**（一维 Kernel，K9 去 threads，与 `TRAP-threads-kwarg-noop` 一致）；256 对齐 pad + masked load → **重新设计**（NPU 尾轴 32B）；fp32 上抛 → **等价替换/重新定域**。无「判重设计却照抄源方案」项。
- **NPU 重设计可行性（pass）**：§0.6 R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足 NPU 约束（一维 persistent `T.Kernel(num_kernels, is_npu=True)`；UB staging；32B 尾轴；reduce src/dst 同 dtype）。R5（取负桥接 → 直接 `T.reduce_min`）另附 `min(x) = -max(-x)` 精确位翻转论证；R1（padding 消除）附 `+inf` 单位元论证并指向机器验证 E4。
- **设计与决策一致性（pass）**：§1–§7 描述的是 §0.5/§0.6 决策后的算法——§3.3 伪代码用 `T.reduce_min(x_ub, m_ub, dim=1, clear=True)`（无取负），§1.4 差异逐条标注来源章节，§5.4 用 `real_m + src/dst 双 slice`，§5.5 一维 persistent 40 核。无脱节。
- **golden 独立性（pass）**：§8.1 golden = `x.float().amin(dim=-1).to(x.dtype)`，以 §0.1 源语义为唯一依据（移植 harness `ReduceTest.ref_program` amin 分支），§1.6.1 明示「不复刻 §0.6 NPU 算法」（无 padding 消除 / 原域 reduce / 直接 reduce_min / persistent 分核）——独立性成立。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有 `docs/` 或 `testing/`/`examples/` 佐证，已抽查核实：`T.reduce_min`（`T.reduce.md` §1 签名 L10、§2.2.1 dtype 表、§2.2.2 rank/shape 约束；`customize_npuir.py` L953–974 `dim=-1, clear=True`，docstring 明示 clear=True 初始化为 `inf`；`testing/npuir/reduction_ops/test_reduce.py` L38 通过，且 `DTYPES=["float16"]`/`ACCUM_DTYPES=["float16"]` 为**同 dtype** 形态）；`T.vcast`（`T.vcast.md` §2.2.1 L42 `bf16→f32 rint` 支持）；`T.copy`/`T.alloc_shared`/`T.Parallel`/`T.min`/`T.Kernel(is_npu=True)+T.serial+guard` 均有同族先例。设计 API 形态（`(bm,N)`→`(bm,1)` 同 rank、src/dst 同 dtype）与文档约束一致。模式匹配正确（Developer 自动同步，无手动 set_flag/wait_flag）。v-prefix 优先（`T.vcast`），`reduce_min` 为该场景唯一归约原语。

### 2. 内存层级规划 — pass

§4.4 搬运路径完整：GM→UB（`alloc_shared`）→计算→UB→GM，无越级路径（无 GM→L0 直连）。§4.3 各 buffer 标注 UB 层级。§4.5 逐 workload UB 预算经人工复算与 §4.5 表格一致：最大项 64KB（fp16 bm=8@4096 / fp32 bm=4@4096 / 3d fp16 bm=2@16384），×2 双缓冲 ≤128KB ≤ 192KB − 8KB reserve；引用 `CONST-capacity-910B2C`（UB=192KB、膨胀 ~1.7x），×2 包络更保守 → 成立。bf16 双大缓冲合计 6 B/elem，仍在其包络内。本设计无 Cube 路径，无 L0C 容量项。**机械 skip 项（`ub_budget`）已人工复核**。

### 3. Tiling 策略 — pass

- **Block 划分**：§5.2 给出 `block_m` 阶梯公式（`max{p : p·N·B ≤ 64KB}`）及扩展阶梯（小 N 大 M 且逻辑块数 > 2×核数），取值自洽；**N 不参与分块**（整行驻留）。
- **对齐约束**：尾轴 32B 对齐由 `T.copy` 切片天然满足（§5.3，logsumexp N=300 先例）；N 无需 256 对齐。
- **非整除处理**：M 尾块 `real_m = T.min(block_m, M−off_m)` + src/dst 双显式 slice（§5.4）；N 整行驻留。
- **分核策略三要素（`_shared/standards/core-split-strategy.md` §1 核对，全齐）**：① 逻辑核数 `num_logical = ceil(M/block_m)`；② 物理核数实查 `NPUUtils.get().get_aicore_num()` → **20**（记录查询代码与返回值，2026-10-13，设备 `Ascend910_9362`），纯 Vector 翻倍 **40**（符合标准 §1 要素②）；③ 规模判定与方案（逻辑≤物理「无需适配」；极大规模固定 40 核 + 核内 `T.serial` 静态上界 + grid-stride guard）。
- **算术复算自洽**：smoke (32,256) fp16 bm=8 → 4 逻辑块 → 4 核 ×1 任务；hidden-state fp16 bm=8 → 256 → 40 核 ×7；bf16 bm=2 → 1024 → 40 ×26（ceil(1024/40)=26）；fp32 bm=4 → 512 → 40 ×13；3d fp16 bm=2 → 64 → 40 ×2。核内串行边界全为工厂期静态整数，仅 guard 动态——满足标准 §2.2。**机械 skip 项（`core_split`）已人工复核**。

### 4. 技术约束检测 — pass

无三维 Kernel（一维 persistent，§5.5）；本设计无 GEMM（无 Cube 路径，无 `T.gemm`/分形/L0C 容量项）；纯 Vector 无 L0C。§3.5.1 逐条列本项目已知限制并给出处理：纯 Vector 核数翻倍（40 实查）、bf16 不被 reduce 支持（vcast 预转）、reduce 混合 dtype 静默错误（强制同 dtype）、host 侧不改输入（`_kernel_handles_padding=True`）、UB 192KB 膨胀（64KB 手工预算）。

### 5. 循环与同步 — pass（建议级）

§6 明确循环选择：任务循环 `T.serial`（静态上界）、归约 `T.reduce_min`（硬件归约）、输出收缩 `T.Parallel(block_m)`、dtype 转换 `T.vcast`。§6.3 说明初始版本不用 `T.Pipelined`（单遍、无 K 迭代；auto-multi-buffer 由 Developer 施加），并列出 Stage 4 候选。§7 同步策略与 Developer 模式一致（自动同步，无手动同步点），说明单消费者缓冲无跨流水竞态面。§6.4 尾块处理见 §5.4。

### 6. 验证方案 — pass

§8.1 含 PyTorch 参考实现草案（`golden_amin`），以 §0.1 源语义为唯一依据。§8.3 L0 门槛含具体 shape/dtype/路径/精度（含 N 非对齐 300、M 尾块 129%8=1、M=1、边界值、NaN 行、spec dim/keepdim）。§8.2 按 dtype 给 atol/rtol（fp32 1e-4；fp16/bf16 1e-2，来自 harness `_tol`），与 §1.6.1 0-ulp 实测相容。

### 7. 完整性与一致性 — pass（建议级）

§0（含 0.1–0.7）与 §1–§11 章节齐全；未发现任何未填充占位标记（花括号占位符、英文待办标记、中文未完成项标记均无；§10 的「待实现」为 Stage 3 计划项，非占位符）。§3.5.3 引用的 `examples/`/`testing/` 路径经 Glob 全部真实存在（amax 对偶、logsumexp single/tiled、argreduce（其设计文档实际文件名 `_argreduce_kernel_DESIGN.md`，`_argreduce_kernel.py` 存在）、`_primitives.py`、amin wrapper、`test_reduce.py` 等）；§3.2/§3.4 引用的 `docs/Tilelang.language/` 文档均存在。API 映射、内存规划、Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：§1.6.0/1.6.1/1.6.2/1.6.3 四子节齐全，另附 §1.6.4 实验裁决三件套。
- **算法调研存在性与完整性（四问齐全）**：R1 候选表含**基线候选 #0**（源取负桥接 + fp32 上抛）并评估 C1–C7；R2 在线算法**有明确结论**（min 可 running 合并，源 tiled 分支 L229 `acc[i]=T.min(acc[i], -tile_acc[i])` 为直接证据），并论证驻留路径本已单遍、online 仅对 N 超 UB 有意义；R3 复杂度表**四口径齐全（含基线行）**，口径写明含/不含；R4 逐候选亲和评估（单元/容量/对齐/静态边界/流水/跨核）。调研深度与算子复杂度匹配。**机械 r3_metrics = pass**。
- **调研结论正确性（独立复核）**：
  - 等价性推演：`min(x) = -max(-x)` 恒等（取负为 IEEE 精确位翻转，有限/±inf/亚正规/±0/NaN 均保持），独立成立；min 为精确选择运算，fp16/bf16 单射保序 → 原域 min ≡ fp32 域 min → E2/E3 成立；`+inf` 为 min 单位元 → E4 成立。
  - 复杂度算术复算：基线 ≈3N（N negate + N cast + (N−1) max + 1 acc-negate）、C1 ≈N、C7 ≈2N、C4 N log N、C3 2 遍；GM 流量均为 1 遍 `N·elem + elem`（negate/cast 属片上），C3 加 workspace 往返——与 §1.6.0 表一致，**未漏计中间缓冲 GM 往返**。
  - 负向断言复核：对照 `tilelang-op-design/references/algorithm-candidates.md` 的 **ALG-reduction 行**（R1「max+sum 合并扫描」/R2「分块归约+合并（两阶段/树形）；Welford」/R3「扫描遍数；跨核归并代价」/R4「水平 vs 垂直 lane 映射」）——设计 C1/C2/C3/C6 覆盖其候选；**Welford 仅对 var 有意义、对 min 不适用**（正确排除）；min 比较次数 N−1 的信息论下界论证成立，无已知被漏掉的化简/在线候选。结论「无更优代数化简」有调研范围声明与结构判据支撑。
  - 亲和性淘汰依据：C4 容量否决、C3 跨核代价、C2 大 N 分流均有依据，未与 pattern-library 实测矛盾。
  - 并列候选裁决：C3 指向 §1.6.4（三件套）；UB staging vs fragment 由 §4.5 纪律兜底。
- **调研→设计一致性**：§1.6.0 选定 C1（驻留直接 reduce_min 原域单遍）实际驱动 §1.4（单遍驻留，逐条标注来源）、§3.1（以 §1.6.1 优化后公式为输入）、§6（N 轴归约，无逐元素标量循环）。迁移优化手段承接逐项说明（§1.6.0「源算法优化手段承接」：#2→R1、#3→E2、#4→R5/E1、#6/#7→R2/R4、#9 换 reduce_min、#10 保留），无静默丢弃；与 §0.5/§0.6 处置一致（不矛盾）。
- **数学等价优化完整性（四要素逐项）**：E1–E5 每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」。E1 取负桥接→直 min（省 1 全宽 negate + 1 (bm,) negate ≈2N op）；E2 原域 reduce（省 1 全宽 fp32 cast + 2B/elem 缓冲）；E3 bf16 vcast（API 必要性）；E4 padding 消除；E5 输出 cast 消除。
- **等价性论证独立推演**：逐项推演均成立（见上「等价性推演」），与 §8.2 精度标准（0-ulp 远优于容差）一致；**verify_equiv 重跑输出与 §1.6.1 内嵌表逐行一致且全 EQUIV_PASS**（687 例），构成机器佐证（与独立推演互补）。amin 对偶点核实：源 `-max(-x)`（源 L132–136）→ 直 `T.reduce_min` 的等价性有源码 + 240 例机器验证双证据；`+inf` 单位元（`_pad_value_for_op("amin")=inf`，源 L62）与 `reduce_min(clear=True)→+inf`（`customize_npuir.py` L966）自洽；min/NaN 传播由设备探针 `repro/CONST-reduce-min-nan-propagation.py`（8 行 × fp16/fp32，`T.reduce_min` vs `torch.amin` 逐位一致）实证；与胞弟 amax 的**对偶一致性**核对成立（min↔max、+inf↔-inf、reduce_min↔reduce_max、clear=True 初始化 +inf↔-inf；两 DESIGN 结构逐条对应，实查核数一致 20×2=40）。
- **向量化替代完整性**：§1.6.2 盘点 §1.4/§3.3/§6 全部循环/标量点（#1–#10），交叉核对无遗漏——源逐元素热点（masked load、全宽 fp32 cast、两次取负）已分别由 E4/E2/E1 消除；保留的 #7–#10（形状收缩 / 动态边界 / block 索引 / host 元数据）逐项给出具体理由，均非逐元素热点。
- **不可替代理由充分性**：#7 `(bm,1)→(bm,)` 因 `T.reduce` 要求 src/dst rank 一致（`T.reduce.md` §2.2.2）且无 cast+收缩复合向量 API；#8 依赖运行时 pid 的 block 级边界；#9 纯 TIR 标量算术；#10 host 元数据——均具体到本算子。
- **替代 API 佐证**：`T.copy`/`T.reduce_min`/`T.vcast`/`T.cast`/`T.Parallel` 均有 `docs/` 或 `examples/` 佐证（优先 v-prefix `T.vcast`）。
- **向量化轴与布局决策完整性**：§1.6.3 按规约类必选候选枚举——水平归约（lane→N，dim=1）vs 垂直扫描（lane→M，串行步进 N）两 lane 映射分别评估，另含核内转置、N-split、UB staging vs fragment；轴质量评分量化（累加链形态为决定性项，垂直扫描发射项灾难，引 `CONST-vector-launch-overhead`）；弃选 #2（量化否决）、#3（引 `PL-1.1-transpose-chain` 实测 ~5.4µs，核内 UB 级转置 + 零收益）、#4（Stage 4 A/B）均有量化理由；对照了源码隐式轴选择（GPU thread/warp 轴为输入非结论）。**弃选论证前提核对**：#3 的 `T.transpose` 核内 UB 级/二轴交换链机制与 `pattern-library/layout.md` PL-1.1（verified）一致，无与 API 文档矛盾的代价机制。
- **实验裁决模式完整性**：主选 C1 判定不依赖未实证常数（依赖已知 UB 预算与发射开销）；§1.6.4 为 C3 提供完整备选结构（buffer 形状/UB 预算/循环结构/dtype 路径/分核三要素）+ 代表 shape + 判定阈值（C3 < 0.8× 主选）+ 回写路径。
- **布局决策与实现一致**：§1.6.3 选定轴 N / 行块 `(bm,N)` 驻留与 §3.3 伪代码、§4.3 buffer 形状与 §4.5 UB 预算、§6 循环结构三方一致（累加内层向量维 = N；buffer 形状 = 选定布局）。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（`reduce_min`、无取负、bf16 分派）；§6 循环结构与 §1.6.2 结论一致（无 §1.6.2 未论证的逐元素标量循环）。

## 备注（不阻塞，供 Stage 3/4 关注）

- §9.2-R11 记录 ±0 的「选中哪个零」理论上依赖实现，机器验证（torch 域）+ L0-9 复验覆盖；L0 门槛用 atol/rtol，`+0.0`/`-0.0` 数值相等，不阻塞。
- §9.2-R2 记录 roofline 常数（`CONST-mte2-degradation` / `CONST-vector-launch-overhead`，origin 为 910B2C 24 核）对本设备（910_9362 20 核→40 Vector 核）按核数外推属「未实测假设 + 估算依据」，已标注 Stage 4 回填对账路径。
- `T.vcast` f32→f32 在 `T.vcast.md` §2.2.1 列为支持，而 §9.1 记录 argmax/CONST 实证其「非恒等、破坏数据」——设计已规避（同 dtype 走 `T.copy`），不影响本算子正确性；此为工具链文档/实测差异，属 Stage 3 实机为准项。
