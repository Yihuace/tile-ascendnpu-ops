# _adaptive_max_pool2d_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md
- 任务类型: 迁移（TileOPs 同构仓迁移，harness 子模式）
- 源算子: /workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py（helper: .../pool/common.py）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（warn，均不影响放行）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `EQUIV_PASS` —— `/workspace/venv/bin/python verify_equiv.py`（脚本随算子目录留存）重跑输出
  `[EQUIV] ALL PASS — 132 cases, 0 failures`，与 §1.6.1 内嵌汇总表（22 分组 / 132 用例）逐条一致，无 `EQUIV_FAIL`。
  用例覆盖 C4（6 shape × 6 kinds × 2 dtype = 72）与 C4b（5 × 6 × 2 = 60），含 `nan` / `inf`（±inf 同张量）/ `subnormal` / `large` / `zeros`。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design …/DESIGN.md`）:
  - `ub_budget`: **skip** —— 解析器未从 §4.5 表格识别出 level 桶（表格首列非 `**UB**` 样式），无法机械求和；**人工已核**：按 §4.5 逐 buffer 复算 spp CH=64 = 35 264 B ≈ 34.4 KB（设计写 ≈34.9 KB，取整差异），resnet C4 ≈ 38.5 KB，均 < 192 KB；默认档 ×1.7 ≈ 59 KB < 192 KB。
  - `l0c_budget`: **skip** —— 纯 Vector 设计无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope("Cube")`，L0C 不适用（人工确认 §5.1/§3.5.1）。
  - `core_split`: **skip** —— 解析器未匹配到 `M=/bm=` 形态（本算子分块维为 `CH × o_h`，非 GEMM `(M,N)`）；**人工已核**：`num_logical` 逐 shape 复算一致（global-1x1 = 8×32 = 256；spp = 2×6×2 = 24；nondiv = 2×7×1 = 14；恒等 `(1,8,9,11)` = N×num_cblk = 1），`num_kernels = min(num_logical, 40)`、`num_local = ceil(num_logical/num_kernels)` 为静态 Python int（§5.5）。
  - `r3_metrics`: **pass** —— 四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）均存在（§1.6.0 R3 表）。
- 说明: D-5 的 3 项 skip 均为"文档形态超出解析器正则"所致，已按 skill 要求转人工复核并给出结论（无 fail）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子 `adaptive_max_pool2d.py`（全文 253 行）与 `common.py`（全文 192 行），并对照 GPU host 层 `tileops/ops/pool.py` 逐项核对 §0.1–§0.7。

- **语义理解正确性 — pass**。
  - 数学语义：§0.1 的 adaptive-bin 划分公式 `[floor(o·in/out), ceil((o+1)·in/out))` 与源码 `common.py:100-108` `adaptive_bin` 逐字一致；`common.py:111-118` `max_adaptive_bin_extent` 作为静态上界亦被 §0.3 步骤 1 正确引用。
  - 规约语义：§0.1「逐 `(n,c,o_h,o_w)` 独立取最大、无跨通道/batch 归约、`kh` 外层 `kw` 内层串行比较、初值 `-inf`」与源码 `adaptive_max_pool2d.py:49-63` 完全一致（`max_val = T.cast(float("-inf"), accum_dtype)` at L51；`T.max(max_val, val)` at L63）。
  - **NaN/±inf 语义（关键）— 核对通过，未发明语义**：§0.1 所述「`has_nan = has_nan | T.isnan(val)` → `result = T.if_then_else(has_nan, NaN, max_val)`（窗口内任一 NaN 胜出）」与源码 `adaptive_max_pool2d.py:50/52/62/65-69` 逐语句对应；设计进一步**独立实测复核**（本检视已用 `/workspace/venv/bin/python` + torch 独立复现）：`F.adaptive_max_pool2d([1,nan;3,4],(1,1))=nan`、`([inf,-inf;1,2],(1,1))=inf`（**±inf 同窗 → +inf，非 NaN**）、全 NaN → nan。设计对"NaN 胜出不能由求和类探测（`Σx!=Σx`）替代、必须用 `v!=v`"的推论经本检视独立推演成立。§0.6 R2 的向量化 NaN 追踪（`h_acc` 累积 + `vcmp(h_acc,h_acc,"ne")` 收尾）经独立推演：`{±inf}` 无 NaN → `cond` 恒假 → `h_acc` 保持 0 → 输出 `acc=vmax 链=+inf` ✓；含 NaN → `h_acc` 变 NaN 且不可逆 → 输出 NaN ✓；与源码/PyTorch 语义**逐位一致**。
  - dtype 语义：§0.1「输入仅 fp16/bf16（op 层 dtype gate 拒绝 fp32），kernel 内 `accum_dtype="float"`、末尾单次舍回」与源码 `adaptive_max_pool2d.py:23/58/70` 及 `common.py:153-156`（`dtype not in {fp16,bf16} → raise`）一致。§0.1「max 下 fp32 为比较域、`dtype→fp32→dtype` 无损、out 逐位等于窗口内某输入元素」经推演成立（f16/bf16→f32 无损上转 + max 选择原元素 + `rint` 回舍无损）。
  - 边界语义：bin 永不为空（`common.py:100-108`）、非整除重叠一行/列、3D CHW `unsqueeze(0)`/`squeeze(0)`（host `pool.py:1041-1044`）均与源码一致。
- **算法解读完整性 — pass**。§0.3 的 7 步分解覆盖源码全部计算语句：步骤 1 = L24-26（`total_output`/`max_kh`/`max_kw`）、步骤 2 = L35（`T.Kernel`）、步骤 3 = L37-44（lane 解码 + L38 守卫）、步骤 4 = L46-47（bin）、步骤 5 = L49-52（初值）、步骤 6 = L55-63（bin 循环 + NaN 追踪 + 取最大）、步骤 7 = L65-70（判决 + 舍回写出）。**无遗漏、无臆造**。数据流图覆盖全部 buffer（register `max_val`/`has_nan`，无 SMEM/L1 中转），且未迁移的 `_adaptive_max_pool2d_with_indices_kernel` 已显式声明排除（与 manifest `return_indices=False` 一致）。host 侧逻辑（`_normalize_output_size`/`_resolve_out_dims`/lru_cache/`_kernel_cache`/`compile_flags`）已列出（`pool.py:918/1012/959`、`adaptive_max_pool2d.py:13/28`）。
- **优化手段覆盖完整性 — pass**。§0.4 列出 10 项并在 §0.6 逐项承接（#1 静态界循环+守卫、#2 fp32 比较域+单次舍回、#3 整数 bin 边界、#4 NaN 追踪、#5 一维扁平映射、#6 `T.Parallel` 线程 lane、#7 寄存器标量比较链、#8 逐元素 GM gather、#9 lru_cache、#10 trace 期常数求值）。源码中确无 SMEM/mma/swizzle/cp.async/三维 Kernel，故"舍弃项：无"成立（§0.5 结论）。识别粒度合理——未把纯数值语义（#2/#3/#4）误判为硬件相关，亦未漏报 `T.Parallel` 与逐元素 gather 这两项真正的 GPU 耦合点。
- **耦合性判定合理性 — pass（含 1 项 warn 级备注）**。§0.5 每条有四态处置 + 依据（migration-analysis §5.3 / `docs/` / pattern-library 条目）：语义层全部「保留」；`T.Parallel` 线程 lane（#6）、寄存器标量比较链（#7）、逐元素 GM gather（#8）判「重新设计」并转入 §0.6 R1/R2/R3；一维扁平映射（#5）判「等价替换」。与 §5.3 映射表不矛盾。
  - **warn 级备注**：#7 在 §0.5 归入「重新设计」，而 §5.3 映射表中「register 累加 → UB 中间缓冲」标注为「等价替换」。设计将其判为重新设计的理由是"归约单元由标量比较链改为 C 轴向量比较器 + 向量 NaN 追踪"，属结构性重建而不仅是存储层级替换，判定可辩护。建议（非阻塞）在 §0.5 #7 行补一句"存储层级层面为等价替换（register→UB），归约结构层面为重新设计"，以免与映射表条目表面冲突。
- **NPU 重设计可行性 — pass**。§0.6 R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素：R1（一维 `T.Kernel` + `(n,o_h,c_block)` 单元 + C 轴向量并行）满足一维 Kernel 约束；R2（UB 向量 `vmax` + `vcmp/vselect` NaN 追踪）满足 Vector 单元；R3（UB tile 搬运 + 核内二轴转置）满足内存层级；R4（C4/C4b 退化特化）。语义保持论证覆盖"取值集合不变 + NaN/±inf 逐条核对 + dtype 舍入 + 边界"；重设计**未改变累加顺序所致的精度语义**（max 为选择，改并行分组不影响取值集合，设计已论证）。
- **设计与决策一致性 — pass**。§1.4/§3.1/§3.3/§4/§5/§6 描述的确为 §0.6 重设计后的 NPU 算法：§3.3 伪代码用一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.alloc_ub` + C 轴 `vmax`，无源方案残留（无 `T.Parallel`、无三维 Kernel、无逐元素 gather）；§3.5.2 列出的参考实现差异表与 §0.6 决策逐条对应；`threads` 参数按 K9 移除（§10.5）。
- **golden 独立性 — pass**。§8.1 golden 直接调用 `F.adaptive_max_pool2d(x…to(float32), size).to(x.dtype)`，以 §0.1 源语义为唯一依据，**不含**转置 / C 向量化 / C4-C4b 特化等 NPU 算法特征；与 `examples/TileOPs/tests/ops/test_pool.py:110-115/130` 的 `ref_program` 同源（torch CPU 参考），验证独立。
- **warn 级备注（路径精度）**：§0 头部与 §0.3 引用 GPU host 层 `ops/pool/adaptive_max_pool2d.py` **在源仓不存在**（实际实现于共享 `…/tileops/ops/pool.py`，`pool.py:1117` `AdaptiveMaxPool2dFwdOp`）。§0 头部已补注"源仓亦含于 `tileops/ops/pool.py` 共享池化模块"，故语义解读无碍；建议（非阻塞）把 §0.3 的 host 侧路径改指 `tileops/ops/pool.py`，避免后续读者误读。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均在 `docs/Tilelang.language/` 找到佐证，本检视逐条打开核对：
- `T.vmax`（`比较操作/T.vmax.md` §2.2.1：fp16 √ / fp32 √ / **bf16 ×**）——设计称"不含 bf16"正确；本设计仅对 fp32 buffer 调用 ✓。
- `T.vcmp`（`比较操作/T.vcmp.md` §2.2.1：fp16/fp32/bf16 √；§2.3 "ne" 模式合法、bool 输出不可落 GM）✓。
- `T.vselect`（`条件操作/T.vselect.md` §2.2.1：fp16/fp32/bf16 √）✓。
- `T.vcast`（`数据类型转换操作/T.vcast.md` §2.2.1：f16/bf16→f32 `rint` √、f32→f16/bf16 √、**i1→f32 rint √**）✓。
- `T.reduce_max`（`规约操作/T.reduce_max.md` §2.2.1：fp32 √ / bf16 ×；`clear=True` 默认）✓。
- `T.transpose`（`创建操作/T.transpose.md` §2.3：**仅支持二轴交换**；fp16/fp32/bf16 √）✓。
- `T.vsub`（`数学操作/T.vsub.md` §2.2.1：fp32 √ / bf16 ×；§2.3 支持原地、UB 上）✓（`nanbuf` 用 fp32）。
- `T.vbrc`（`shape操作/T.vbrc.md` §2.2.2：scalar→任意 shape √）✓。
- `T.clear`（`内存操作/T.clear.md` §2.2.1：fp32 √；§2.2.2：1–5 维）✓。
- `T.infinity`（`创建操作/T.infinity.md` §2.2.1：fp16/fp32 √、返回标量）✓。
- `T.alloc_ub`（`内存操作/T.alloc_ub.md` §2.2.2：1D–2D 充分验证、**3D–5D 有待充分验证**）——设计已在 §3.4/§9.1 显式标注 3D 风险并给出同族实证 + Stage 3 回退（2D `(CH,H*W)`）。
- `T.Kernel`（`编译器提示操作/T.Kernel.md` §2.3：NPU ≤2 维、1 维合法；官方示例以 `as (cid, _)` 解包 1D grid）✓。

**模式匹配**：Developer 模式，全部使用自动同步 + v-prefix 向量 API（`T.vmax/T.vcmp/T.vselect/T.vcast/T.vbrc`），无手动 `set_flag/wait_flag` ✓。`T.alloc_ub` 在 Developer 模式下使用与同族前作 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py:60/232/301`（Developer 注释 + `T.alloc_ub`）一致，属可用的兼容写法。**API 无臆造，v-prefix 优先 ✓**。

### 2. 内存层级规划 — pass

- §4.4 搬运路径完整：`GM[x] → UB[row_src] → UB[row_dst/row_f32] → UB[acc/h_acc] → UB[out32/acc16/acc_t] → GM[out]`；C4 路径 `GM→UB[flat]→…→GM`；C4b `GM→UB[stage]→GM`。**无 GM→L0 越级路径**（纯 Vector，无 Cube/L1/L0C）。
- 每个 buffer 均标注层级（UB）＋ shape＋dtype（§4.3/§4.5）✓。
- UB 预算：本检视逐 buffer 独立复算，spp CH=64 = 34.4 KB、resnet C4 ≈38.5 KB，均 < 192 KB；×1.7（auto-multi-buffer）≈ 59 / 67 KB < 192 KB ✓（§4.5 两档口径明确）。CH=64 时 `row_src(64×56 fp16)` 7 168 B、`row_f32(56×64 fp32)` 14 336 B 等均与设计表一致（取整差 <1 KB，可解释）。
- L0C：不适用（无 Cube）。
- **warn 级备注**：设计未显式讨论 bool（i1）buffer 的尾轴对齐——`T.vbrc` 文档 §2.3 指出 i1 的 VBrcOp 要求 dst 尾轴 16 对齐；C4 的 `mb/mf32 (CH,H_in,W_in)=(64,7,7)` 尾轴 7 非 16 倍数，但 `mb` 由 `vcmp` 产生、由 `vcast`/`reduce_max` 消费（非 VBrcOp 的 dst），文档未对该路径提出对齐要求。建议（非阻塞）Stage 3 若遇编译失败时优先核查此项。

### 3. Tiling 策略 — pass

- Block 划分：§5.2 给出 `CH = min(C, 64)`、`num_cblk = ceil(C/CH)` 与理由（C 恒为 8 的整数倍、UB 充裕、与分核配合）；代表 shape 分块表齐全。
- 对齐：向量轴 C 恒为 8 倍数（fp16 向量宽）✓；`W_in`（55/56/57）仅作 row 长度不作向量轴，不要求整除 ✓。
- 非整除：任务尾块 `if uid < NUM_LOGICAL`、通道尾块 `real_c = min(CH, C-cb*CH)`、bin 非整除由整数 `adaptive_bin` 精确处理（§5.3/§5.4）✓。
- **分核策略三要素齐全（对照 core-split-strategy.md §1，检视要点 §2.2）**：
  - ① 逻辑核数计算：按分支粒度给出（一般 `N×out_H×num_cblk`；C4/C4b 整平面 `N×num_cblk`），并**明确论证 C4b 必须用整平面粒度**（否则 `n=uid//num_cblk` 在 N=1 时越界，§5.2/§5.5①）——该自变量粒度定义唯一、与 §3.3 解码/§6.2 自洽（本检视复算 `(1,8,9,11)` 恒等：通用式 9 vs 整平面 1，采后者正确）。
  - ② 物理核数依据：`NPUUtils.get().get_aicore_num()` **实查 = 20**（Ascend910B3，记录查询代码与返回值，§5.5②）；纯 Vector 翻倍 `20×2 = 40`；明确**不采用** 910B2C 的 24。该实查值与 pattern-library `CONST-aicore-910B2C`（2026-10-13 跨型号数据点 910B3=20）互证 ✓。
  - ③ 规模判定与分核方案：逐 workload 判定（smoke/spp/nondiv/恒等 → 逻辑核数 ≤ 物理核数 →「无需适配」；global-1x1 = 256 > 40 → persistent，`num_kernels = min(NUM_LOGICAL, vector_cores) = 40`、`num_local = ceil(256/40) = 7`**静态 Python int**）。
- **核数与 block 取值自洽**：本检视独立复算 global-1x1 `8×1×32 = 256`、spp `2×6×2 = 24`、nondiv `2×7×1 = 14`，与设计一致；`num_kernels ≤ NUM_LOGICAL` 保证不产生空核（§5.5）。
- **核内串行边界为静态值** ✓（`num_local` 由编译期常量导出，满足 core-split-strategy §1 要素③硬要求）。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel`，任务维度在核内解码）✓。
- GEMM 非整除 / L0C 溢出：不涉及（无 Cube，§5.1 已论证 Cube 路径数学不可表达）✓。
- 动态 `T.serial` 边界：涉及并已处理（静态上界 `MAX_BIN_H` + `if kh < bh` 守卫，§3.5.1/§5.4）✓。
- host 侧不得改输入内容：仅 `unsqueeze/squeeze/contiguous` 视图操作 ✓（与 `pool.py:1041-1057` 一致）。

### 5. 循环与同步 — pass

- §6.1 循环结构明确：逻辑单元 = `T.Kernel`（block 级并行）；累加器初始化 = `T.vbrc/T.clear`（整块，无循环）；核内多任务 = `T.serial(num_local)`（静态界）；bin H 方向 = `T.serial(MAX_BIN_H)` + 守卫；bin W 方向与 `o_w` = Python 编译期展开；元素级（C 轴）= 向量 op。**§6 不出现逐元素标量循环**，与 §1.6.2 结论一致 ✓。
- §7 同步策略与 Developer 模式匹配（编译器自动同步；无手动 set_flag/wait_flag；无跨核/跨引擎依赖）✓。`T.Pipelined` 未用并给出理由（§6.3，且列为 Stage 4 候选）✓。
- §6.4 尾块处理：任务尾块/通道尾块/bin 尾块（H 守卫、W 精确 `range(bin_w)`）逐项说明 ✓。
- §3.3 伪代码与 §6.2 循环伪代码一致。

### 6. 验证方案 — pass

- §8.1 含 PyTorch golden 草案（`F.adaptive_max_pool2d`，独立于 NPU 算法）✓。
- §8.2 精度标准按 dtype 给出：fp16/bf16 均 `atol=rtol=0`，并给出依据（max 是选择不累加；`test_pool.py:130` "Max is exact"）——与源仓测试基准 `atol=0, rtol=0` 一致 ✓。
- §8.3 L0 门槛测试计划含 10 个具体规则 shape（覆盖 manifest smoke/global/spp/nondiv、C4b 恒等、部分 None、通道尾块 C=96、3D CHW、NaN/±inf 注入、K9 guard），dtype 明确，NaN 用例标注 `equal_nan=True` ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：§0–§10 共 11 章，迁移类 §0.1–0.7 小节齐全 ✓。
- 无占位符：全文未出现模板占位符标记或未完成标记（grep 复核 none；§10.2 的 ⬜ 为交付状态标记，非占位符）✓。
- 同类引用真实：§3.5.3 引用的 7 条 `examples/` 路径 + `pattern-library/layout.md` 全部 `os.path.exists` 通过（本检视逐条 `ls` 验证）✓。唯一不精确处见维度 0 的 host 层路径备注（warn，非阻塞）。
- 内部一致：§1.6.3 布局决策已同步到 §3.3/§4/§6（C 为向量轴、buffer 形状、累加内层维）；§3.1 以 §1.6.1 优化后公式为输入；§5.2 的 `NUM_LOGICAL` 分支规则与 §3.3/§5.5①/§6.2 唯一自洽 ✓。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四子节，均非空。
- **算法调研存在性与完整性 — pass**：§1.6.0 四问齐全——
  - R1 候选表**含基线**（#0 源算法；#0' 基线向量化），评估了 `algorithm-candidates.md` `ALG-pooling` 命中行全部候选（常数权重卷积 / sum-pool×常数）＋恒等变形空间（可分离 2D、前缀最大值差分、退化特化、归约融合）。
  - R2 在线算法：**明确结论「有」**并给结构判据（可分解为 running max + running NaN 标记，源 #6/#7 已是 running 比较，作为直接证据）——合法且可复核。
  - R3 复杂度对比覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表，口径写明（含/不含项）。
  - R4 逐候选硬件亲和评估（计算单元/容量/对齐/静态边界/流水/跨核）。
  - 调研结论含选定算法（#0'）与关键依据；「无更低复杂度算法族」写明调研范围（候选库命中行 + 恒等变形 + pattern-library + 本仓/同族前作），互联网检索未用并说明理由。
- **调研结论正确性（独立复核）— pass**：
  - **负向断言逐条复核**：① "常数权重卷积/GEMM 不等价"——mathematical fact（max 非线性，不存在 0/1 权重矩阵使 max = 线性加权和），成立；② "前缀最大值差分不可行因 max 无逆元"——成立（设计的反例 `x=[5,1,3]`，区间 max=3 而 `P[b]-P[a]=0` 正确）；③ "可分离 2D 运算量更大"——**独立复算成立**：spp（H=W=56,out=6，scan=60）每平面直接法 `scan_H·scan_W = 3600`，可分离法 `H_in·scan_W + scan_H·out_W = 56×60 + 60×6 = 3360+360 = 3720 > 3600`，设计数字正确。
  - **复杂度算术独立复算**：spp 基线 FLOPs `N·C·scan_H·scan_W = 2·128·60·60 = 921 600` ✓；访存 `(N·C·H·W + N·C·out_H·out_W)·2 = (802 816+9 216)·2 = 1 624 064 ≈ 1.624 M` ✓。且**与 manifest `pool.yaml` 的 `roofline.flops / bytes` 公式逐字一致**（`flops = N*C*scan_H*scan_W`，`bytes = (…)*elem_bytes`）；`scan_H = H_in + Σ_{j=1..out_H-1}[(j·H_in)%out_H≠0] = 56+4 = 60` 与 manifest 公式复算一致 ✓。中间缓冲口径明确"不含 UB 内搬运、中间缓冲不落 GM"，无漏计 GM 往返。D-2 roofline 三项（global-1x1 1.30 µs / spp 1.29 µs / nondiv 0.65 µs）独立复算流量项一致。
  - **亲和性淘汰有佐证**：候选 1 的 `reduce_max` bf16 缺支持（已核 `docs/…/T.reduce_max.md` §2.2.1）、候选 0 的标量占比反例引 PL-1.2、候选 3 host permute 106–147µs 引 PL-1.3（已核 layout.md 原文）——均与本地实测/文档一致。
  - 无漏掉已知候选（窗口/归约类未遗漏相关在线/化简变体；max 的代数化简空间确实为空，属数学事实）。
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以 §1.6.0 选定算法 #0' 为对象；§0.5/§0.6 处置与选族一致（R2 = #0' 归约向量化 + 向量 NaN 追踪、R3 = C 轴布局、R4 = 候选 #4）；源算法 §0.4 的 10 项优化手段逐项有承接（**无静默丢弃**）。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法做系统审视，采纳 C4（全局池化结构特化）与 C4b（恒等结构特化），各含四要素「原式 → 优化后公式 → 等价性论证 → 收益估算」；并写明"无倒数乘/无 CSE/无高代价算子降代换"的结论与依据（max 无归一化、每元素只参与一次比较）。
- **等价性论证独立推演 — pass**：C4（`out_H=out_W=1` ⇒ `bin(0,0)=[0,H)×[0,W)`=整平面）与 C4b（`out_H=H_in` ⇒ 单位 bin，max=元素本身）**逐位等价**，独立推演成立；且 **`verify_equiv.py` 重跑 132 用例全 `EQUIV_PASS`**（与 §1.6.1 内嵌表一致，含 NaN/±inf/subnormal 组），机器验证有效。
- **向量化替代覆盖完整性 — pass**：§1.6.2 盘点 14 个循环/标量点（#0–#13），与 §3.3 伪代码、§6.1 循环结构**交叉核对一致**；可向量化点全部给出向量替代（+ `docs/` 佐证），保留 4 处（#9 tile 级顺序依赖+动态边界、#11 block 级任务映射、#12 host/trace 元数据、#13 host 视图），理由具体到位，无"以实现简单为由"。
- **替代 API 可行性 — pass**：替代方案 API 均有 `docs/Tilelang.language/` 佐证（与维度 1 同口径），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 候选矩阵枚举了窗口/池化类**两类必选候选**（候选 1 = I/O 原生 NCHW + 最内连续 W 轴；候选 2 = 核内 NHWC + 高整除性 C 轴）＋ host permute、Cube 两条附加候选；逐候选有量化评分（整除性/尾 lane 浪费率/累加链形态/repack 代价/UB 影响）；弃选方案有量化理由（候选 1 段长 ~12% 尾 lane 浪费 + `reduce_max` 无 bf16；候选 3 host permute 106–147µs 净亏；候选 4 数学不可行 + 分形下限），**无"未考虑"或"I/O 是 NCHW 所以内部也 NCHW"式论证**；对照了源 GPU 隐式轴选择（作为输入而非结论）。**弃选论证前提已亲自核对 API 文档**：`T.transpose` 确为核内 UB 级二轴交换（`创建操作/T.transpose.md` §2.3），未误写成 GM 级 transpose；PL-1.1/1.3/1.4 实测条目已核 layout.md 原文。
- **实验裁决模式完整性 — pass**：§1.6.3 D-3 给出主选（CH=64，落在 PL-1.4 已实测 CH16/32/64 最优档）+ 备选（C4 + CH=8，结构完整可直接实现）+ 裁决计划（代表 shape、指标 msprof op Task Duration、未知常数反解方法、判定阈值 `备选 < 主选×(1−0.10)`、回写路径），并评估 shape 分派可行性。窄向量 op 每-op 代价已明标「未文档化假设」并给出方向证据（PL-1.4 实测趋势 + `CONST-vector-launch-overhead`），未把未实证量当作确定判定。
- **布局决策与实现一致 — pass**：§1.6.3 选定布局（核内 C 最内 + 二轴转置链）与 §3.3 伪代码 buffer 形状/内层向量维、§4.5 UB 预算、§6 累加内层维三方一致。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入（含 C4/C4b 行）；§6 无 §1.6.2 未论证的标量循环。
- **warn 级备注**：R3 表「扫描遍数」列对基线/候选 2 均记 "1"，而候选 2 的 §1.6.1/R4 文字写"两遍串行"、基线含"重叠行 ~7% 重复读"。该列口径为"输入 GM 读取遍数"（可分离法的第二遍在片上，故 GM 仍 1 遍），逻辑自洽，但建议补一句口径说明以免与"两遍串行"字面观感冲突（非阻塞）。

---

## 附：非阻塞改进建议（不影响放行）

> 以下为维度说明中标注的 4 项 warn，供 Stage 1 可选修订参考；均未构成阻塞级问题，故检视结论为通过。

1. §0.5 #7 / 维度 0：把「寄存器标量比较链」的处置拆为「存储层级 = 等价替换（register→UB）、归约结构 = 重新设计」，与 migration-analysis §5.3 映射表条目显式对齐。
2. §0.3 / 维度 0、7：host 侧路径 `ops/pool/adaptive_max_pool2d.py` 在源仓不存在，改指 `tileops/ops/pool.py`（`AdaptiveMaxPool2dFwdOp`）。
3. §4.3/§4.5 / 维度 2：补一句 bool（i1）buffer 尾轴对齐说明（`T.vbrc` 文档 §2.3 对 i1 dst 的 16 对齐要求；本设计 bool 由 `vcmp` 产生，路径不同但建议显式注明）。
4. §1.6.0 R3 / 维度 8：为「扫描遍数」列补口径说明（GM 读取遍数；可分离法的第二遍在片上），避免与"两遍串行"字面冲突。
