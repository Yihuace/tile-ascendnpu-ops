# `_simple_reduce_kernel`（AmaxFwdOp 驻留路径）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/amax/_simple_reduce_kernel/DESIGN.md
- 任务类型: 迁移（harness）
- 源算子（kernel）: `/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`ReduceKernel` + `_simple_reduce_kernel` 工厂）
- 源算子（Op 层）: `/tmp/opencode/TileOPs-ref/tileops/ops/reduction/reduce.py`（`_SimpleReduceOp`/`AmaxFwdOp`）
- 源算子（primitives）: `/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/_primitives.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。`python examples/amax/_simple_reduce_kernel/verify_equiv.py`（exit 0）输出
  `E1 rows=160 max_ulp=0 viol=0 EQUIV_PASS` / `E2 rows=77 max_ulp=0 viol=0 EQUIV_PASS` / `E3 rows=210 max_ulp=0 viol=0 EQUIV_PASS` /
  `INFO NaN propagation agree: True` / `RESULT: ALL EQUIV_PASS`。
  合计 **447 例**（160+77+210），与 §1.6.1 内嵌结果表逐项一致（含 E1=160 / E2=77 / E3=210 / 合计 447）。脚本内容为实质比对（候选原域串行 max 与基线 fp32 上抛逐位比对 + 角点集，含全 −inf/全 +inf/±inf 夹逼/重复极值/亚正规/混合符号零/dtype 极值/全零/全 NaN/NaN 夹逼/单调列），非橡皮图章。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/amax/_simple_reduce_kernel/DESIGN.md`（exit 0）JSON checks 摘要：
  - `r3_metrics`: **pass** — all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)。
  - `l0c_budget`: **skip** — no affirmative Cube-path API（无 T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)）→ 人工复核结论：**n/a**（纯 Vector，无 L0C，正确）。
  - `ub_budget`: **skip** — no per-level buffer tables parsed → 人工复核结论：**已核**。§4.5 手工预算逐 workload 复算：fp16 bm=8@N=4096 = 8·4096·2 = 64KB ≤65536；fp32 bm=4@4096 = 64KB；bf16 bm=2@4096 = 2·4096·6 = 48KB；3d (128,16384) bm=2 fp16 = 64KB；smoke (32,256) bm=8 = 4KB。均 ≤64KB 手工预算，×2（auto-multi-buffer）≤128KB ≤192KB。B/dtype 表（fp16 2 / fp32 4 / bf16 2+4=6）与 §3.2/§5.2 三方一致。
  - `core_split`: **skip** — insufficient parsed dims → 人工复核结论：**已核**。`num_logical=ceil(M/bm)`，`num_kernels=min(num_logical,40)`，`num_local_tasks=ceil(num_logical/num_kernels)`；hidden-state fp16（2048/8=256）→ 40 核 × 7 任务，算术正确。
- **物理核数独立实查**：`python -c "from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()"` → **20**（本机 `Ascend910_9362`），与 §5.5 记录一致；纯 Vector 翻倍 40 成立。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read `reduce.py`（kernels，1084 行）、`reduce.py`（ops，763 行）、`_primitives.py`（445 行）全文，与 §0.1–§0.7 逐项核对。

- **语义理解正确性（§0.1）**：源码 `_SIMPLE_REDUCE_MAP["amax"]="reduce_max"`（L46）、amax 分支 `T.reduce_max(x_f32, acc, dim=1)`（L129）——设计表述「逐行最大值、非最大绝对值」与源码一致。dtype 语义核对：源码 `x_f32 = cast(x, "float32")`（L110/L119）→ reduce → `T.cast(acc, dtype)`（L140）；§0.1「源实现先 cast fp32 再 reduce，结果 cast 回」与源码逐字一致。边界语义：`_pad_value_for_op("amax") = -inf`（L63–L64）与 §0.1「−inf 单位元填充」一致。host 侧：`_SimpleReduceOp._kernel_handles_padding = True`（ops/reduce.py L524，AmaxFwdOp L548 继承）→ §0.1「Op 层不施加 host 侧 F.pad」成立；`_prepare_input` 的 validate→movedim/flatten→reshape(M,N)（ops/reduce.py L425–474）与 §0.1 host 侧语义一致。
- **算法解读完整性（§0.3）**：步骤表 1–5 + 2a/2b 覆盖源码每条计算语句——步骤 1=`T.copy(x[pid_m*block_m,0], shared_buf)`（源 L115）；步骤 2a=masked `T.Parallel` + `T.if_then_else(And(row<M,col<N), cast(x,f32), cast(-inf,f32))`（源 L107–L112）；步骤 2b=`cast(shared_buf,f32)`（源 L118–L119）；步骤 3=`T.reduce_max(x_f32, acc, dim=1)`（源 L129）；步骤 4=`out_local[i]=T.cast(acc[i],dtype)`（源 L139–L140）；步骤 5=`T.copy(out_local, out[pid_m*block_m])`（源 L143）。无遗漏、无臆造。数据流图（GM→SMEM→FRAG→GM）与源码 buffer 集合（shared_buf/x_f32/acc/out_local）一致。host 侧 `functools.lru_cache`（源 L69）已列出。**源缺陷识别**：源 L115/L143 的 `T.copy` 无 `M % block_m ≠ 0` 尾块守卫，§0.3 明确标注为源缺陷并要求 Stage 3 修正——正确。
- **优化手段覆盖完整性（§0.4）**：覆盖源码全部优化性结构 9 项（SMEM tiling、双路径边界、fp32 上抛、单位元 pad、block×thread 两级并行、48KB SMEM 启发式、256 对齐+32512 分档、reduce 原语、lru_cache），与源码结构一一对应。核对 `SHARED_MEMORY_BUDGET_BYTES=48*1024`（_primitives L43）、`MAX_SINGLE_TILE_COLS=32512`（L39）、`DEFAULT_ALIGNMENT=256`（L30）——数值全部正确。未把实现层误判为算法层（如 `_needs_pad` 归入「优化」而非「语义」）。
- **耦合性判定合理性（§0.5）**：四态处置逐条有依据。核对：`T.reduce_max` 的 NPU 等价能力经 `T.reduce_max.md` §2.2.1（fp16/fp32 √、bf16 ×）与 `testing/npuir/reduction_ops/test_reduce.py`（shared→shared 同 dtype 通过）佐证——「等价替换」判定成立；`T.if_then_else` 引用循环变量条件不可移植的判定有 logsumexp DESIGN §3.5.2 依据——「重新设计」判定合理；block×thread/threads 依 K9 去除有 `TRAP-threads-kwarg-noop`（traps-compiler L124，src/ir.cc NPU 分支无 threadIdx 绑定）佐证。无「warp shuffle 判保留」「三维 Kernel」类矛盾，无「舍弃无理由」项。
- **NPU 重设计可行性（§0.6）**：R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（边界）满足 NPU 约束（32B 尾轴、raw N、`real_m` 双 slice，与 logsumexp single L74/L79–L81 及 argmax `_argreduce_kernel.py` L237–L238 直接先例同形）；R2（并行）满足一维 Kernel + 实查核数（我独立复核实查=20→40）；R3（dtype 域）语义保持论证「max 是精确选择运算」经机器验证；R4（分档）由 UB 预算替换 GPU 256 对齐/32512 上限，语义保持为「纯配置选择」。**重设计不改变累加顺序语义**（per-row 独立，max 交换幂等），无未论证的精度影响。
- **设计与决策一致性**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法——§1.4 五项与 §0.6-R1/R2/R3/R4 逐条互标；§3.3 伪代码为 raw-N + 原域 reduce_max + 一维 persistent；§6 循环结构无「§0 判重设计、§3 照抄源方案」脱节。`threads` 在 §3.3/§4.7 中确已去除（K9）。
- **golden 独立性（§8.1）**：golden = `x.float().amax(dim=-1).to(x.dtype)`，与源仓 `examples/TileOPs/tests/ops/test_reduce.py` `ReduceTest.ref_program`（我核对源码 L80–L81 逐字一致）相同；以 §0.1 源语义为唯一依据，未复刻 §0.6 的 NPU 数据流（无 padding 消除/原域 reduce/persistent 结构），验证独立性成立。

### 1. API 可行性 — pass（1 项建议级注记）

逐条 API 均有 `docs/Tilelang.language/` 或 `testing/`/`examples/` 佐证（我逐个打开核对）：
- `T.copy`（双显式 slice / UB→GM 截断）：`docs/Tilelang.language/内存操作/T.copy.md` 存在；logsumexp single L79–L81/L113–L114 同形先例。
- `T.reduce_max(src,dst,dim=1,clear=True)`：`T.reduce_max.md` §2.4 示例即 `T.reduce_max(b, s, dim=1, clear=True)`（src (M,N)→dst (M,1)）；dtype 表 fp16/fp32 √、bf16 × 与设计一致。
- `T.vcast(src,dst,round_mode)`：`T.vcast.md` §2.1 签名 + §2.2.1 `bf16→f32 rint` 行（L42）逐字核对成立。
- `T.alloc_shared`/`T.Kernel(is_npu=True)`/`T.serial`/`T.Parallel`/`T.cast`/`T.min`：均有文档或 argmax `_argreduce_kernel.py` L217–L254 直接先例（同一 persistent + grid-stride + `real_m` + 双 slice 结构）。
- v-prefix 优先：`T.vcast` 为 v-prefix；无 npuir_xxx 遗留。
- 模式匹配：Designer/Developer 自动同步，无手动 set_flag/wait_flag，与 §2.1 Developer 结论一致。

**建议级注记（不影响实现）**：§3.5.2 表述「`clear=True` 默认清零」不精确——`tilelang/language/reduce.py` L54 明确 `reduce_max` 的 `clear=True` 是「output buffer will first be initialized to **-inf**」（max 的单位元），非清零。因结论（「无需手动 `T.fill(acc,-inf)`」）正确、且 §3.3 实写 `clear=True`（文档合法形态），实现不受影响。建议改写为「clear=True 会初始化为 −inf（max 单位元）」。

### 2. 内存层级规划 — pass

- 搬运路径完整：§4.4 明确纯 Vector `GM → UB (alloc_shared) → 计算`，无越级路径（无 GM→L0 直连），无 L1/L0/L0C。
- UB 预算：§4.5 逐 workload 验证，手工复算全部成立（见「机械复核」ub_budget 人工结论）；大缓冲数 ≤2（`x_ub`，bf16 加 `x_work`），微缓冲 `m_ub`/`out_ub` 量级 ≤7·bm 字节。
- L0C 容量：n/a（无 Cube）。Buffer 层级：§4.3 每个 buffer 标注 UB（`alloc_shared`）。
- 口径声明诚实：R3 标注「UB 膨胀系数对本结构未独立实测」，依据 = 同族 argmax 探针 64KB PASS / 128KB FAIL 双向标定 + 结构更简；Stage 3 首编译以 `ub overflow` 报文核对。成立且非阻塞。

### 3. Tiling 策略 — pass

- Block 划分明确：§5.2 给出 `block_m` 阶梯公式（基础 {1,2,4,8} + 扩展 {16..2048}）及理由；逐 workload 值可复算（fp16 hidden-state bm=8、bf16 bm=2、fp32 bm=4、3d bm=2、smoke bm=8 均与预算一致）。
- 对齐约束：尾轴 32B 由 `T.copy` 切片天然满足（logsumexp N=300 先例），N 无 256 对齐——正确反映 NPU 与 GPU 对齐约束差异。
- 非整除处理：M 尾块 `real_m = T.min(block_m, M−off_m)` + src/dst 双显式 slice + 垃圾行不写出（§5.4），符合 `TRAP-T-copy-region-semantics` 推荐形态；N 无尾块（整行驻留）。
- **分核策略三要素齐备**：① 逻辑核数 `num_logical=ceil(M/block_m)`；② 物理核数依据**实查**——`NPUUtils.get().get_aicore_num()` 记录查询代码与返回值 20 + 设备名 `Ascend910_9362`（我独立复核实查=20）；③ 规模判定与分核方案表格（`num_kernels=min(num_logical,40)` + 核内静态 `T.serial` + guard）。核内串行边界全部为工厂期静态值，仅 guard 动态——符合 core-split-strategy.md 对「核内串行任务数不依赖动态 shape」的要求。
- **路由可达性与契约（重点核对）**：驻留适用边界 `∃p≥1: p·N·B ≤ 64KB` ⟹ fp16 N≤32768 / fp32 N≤16384 / bf16 N≤10922；manifest long-seq (64,32768) bf16 落界外，可达路由至兄弟单元 `_simple_reduce_kernel_tiled`（§5.2/§9.3）。契约明确：本单元只承载驻留、tiled 由兄弟单元承接，且 §9.2-R9 显式声明 wrapper 层 `_needs_tiling`/`default_config` 的 NPU 化属 wrapper/兄弟单元职责（现状仍为 GPU 语义）。本单元范围内契约完整、可达性成立；R9 为跨单元集成待办，非本设计缺陷（建议级注记见下）。

**建议级注记**：§5.2 伪代码 `route_to_sibling_tiled_kernel()` 为概念占位（非实际函数）。本设计范围明确不含 tiled 内核，故不阻塞；建议 Stage 3/wrapper 集成时以真实分派谓词落地（R9 已标注）。

### 4. 技术约束检测 — pass

§3.5.1 逐条检查：三维 Kernel（No，用一维 persistent）；纯 Vector 核数翻倍（Yes，40 实查）；bf16 不被 reduce 支持（Yes，`T.reduce.md`/`T.reduce_max.md` §2.2.1 bf16 ×，走 vcast→fp32）；reduce 混合 dtype 静默错误（Yes，同 dtype 约束，bf16 用独立 fp32 缓冲）；GEMM 非整除（No）；L0C（No）；host 不改输入（Yes，`_kernel_handles_padding=True`）。约束与 §3.3/§4/§5 处理方案一致，无遗漏。

### 5. 循环与同步 — pass

- 循环结构：§6.1 明确 `T.serial`（任务级）+ 硬件 `T.reduce_max`（N 方向）+ `T.Parallel(block_m)`（形状收缩微循环）+ `T.vcast`（dtype 转换）；明确不使用 `T.Pipelined` 并给出理由（单遍、无 K 维迭代、auto-multi-buffer 自动重叠）。
- 同步与模式匹配：§7 Developer 自动同步，单 kernel 无手动同步点、无核间协作——与 §2.1 Developer 结论一致。
- 尾块处理：§6.4 指向 §5.4，M 尾块逻辑完整。

### 6. 验证方案 — pass

- Golden 函数：§8.1 `golden_amax` = `x.float().amax(dim=-1).to(x.dtype)`，与源仓 `ReduceTest.ref_program`（`examples/TileOPs/tests/ops/test_reduce.py` L80–L81）逐字一致。
- L0 测试计划：§8.3 给 10 组具体 shape/dtype/路径/容忍度，覆盖 manifest 全 workload + N 非对齐（300）+ M 尾块（129）+ M=1 + 边界值语义 + dim/keepdim（spec 接口）。
- 精度标准：§8.2 fp32 {atol 1e-4, rtol 1e-4}、fp16/bf16 {1e-2, 1e-2}——与源仓 `_tol`（`test_reduce.py` L88–L91）一致。

### 7. 完整性与一致性 — pass

- 章节齐全：11 章节齐全，迁移类 §0 且 0.1–0.7 小节齐全。
- 无占位符：全文检索模板占位符标记（占位花括号 / 待办标记 / 待补全标记等）均无命中。
- 同类引用真实：§3.5.3 全部引用路径经 Glob 核对存在（logsumexp single/tiled、argmax `_argreduce_kernel.py`+DESIGN、`testing/npuir/reduction_ops/test_reduce*.py`、`_primitives.py`、amax wrapper/op/manifest）。
- 内部一致：§3.2 dtype 矩阵 B、§4.5 B 表、§5.2 B 字典三处 B 值一致（2/4/6）；§1.4 五项与 §0.6-R1..R4 互标一致；§5.5 workload 表与 §1.6.0 roofline 表核数/tasks 一致。唯一措辞瑕疵见维度 1 注记（`clear=True`）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与布局决策 + 1.6.4 实验裁决（超出最低要求的完整度）。
- **调研存在性与完整性（§1.6.0）**：R1 候选表**含基线（#0 GPU 源单遍）**且列 C1–C6；R2 在线算法给出**明确结论**（「有 running-max，但基线本已单遍，仅 tiled 有意义」）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径注明「per row」含/不含）；R4 硬件亲和逐候选六维评估。调研深度与规约类算子匹配。
- **调研结论正确性（独立复核）**：
  - ① 等价性：C1（原域 max）与基线（fp32 上抛）的等价性独立推演成立——max 为精确选择运算，fp16→fp32 精确单射且保序保等，两域选出同一元素，逐位相等；bf16↔fp32 同理。
  - ② 复杂度算术独立复算：基线 = N·elem + elem（1 遍，cast 在片上不计 GM）；C1 同流量、缓冲更小（fp16 2 vs 基线 4B/elem）；C3 = +workspace `M·nchunk·(elem+elem)` 往返 + 2 遍（partial 写 + merge 读），workspace 未被漏计；C4 = N log N。数字与设计一致。
  - ③ 负向断言复核：对照 `algorithm-candidates.md` `ALG-reduction` 行（L62–L68：R1「max+sum 合并扫描/分块」、R2「分块归约+合并（两阶段/树形）；Welford」、R3 扫描遍数、R4 水平 vs 垂直 lane）——设计的 **C2 分块 running-max**（online）与 **C3 N-split partial+merge** 已覆盖参考表的「分块归约+合并」；参考表的 **Welford** 属 var/std 族，对 max 不适用（非漏检）；**C5/C6**（变换域/树形）由结构判据排除（max 无保序降代换、硬件原语已含树形归约），非「未检索先否定」。**无被漏掉的已知候选**。参考表 `known_impl` 指向 logsumexp（同族），设计已引用。
  - ④ 亲和性淘汰有佐证：容量/带宽淘汰引用 `constants.md`（`CONST-capacity-910B2C` UB 192KB/膨胀 1.7x、`CONST-reduce-dims0-skinny` dims=1 快路径、`CONST-vector-launch-overhead` 0.5µs/op、`CONST-mte2-degradation` 33.9/30.2 GB/s·core）——我逐条打开 constants.md 核对数值成立；设备/核数差异（910B2C 24 核 vs 实测 20 核）已诚实标注为外推假设。
  - ⑤ 互联网来源：未触发（本地源覆盖），无需来源记录。
  - **R3 并列候选裁决**：C3 vs 主选判定裕度依赖小 M 大 N 场景，未纸面拍板——已指向 §1.6.3-#4 / §1.6.4 实验裁决（备选完整结构 + 判定阈值 + 回写路径）。
- **调研→设计一致性**：§1.6.0 选定 C1 → §1.4 算法描述为「原域单遍 reduce_max」、§3.1 公式拆解以优化后三式为输入、§6 循环结构 dim=1 restore——三处与选定算法同口径。迁移侧：§0.4 优化手段意图逐项承接（#2→R1/E3、#3→E1、#5/#6→R2/R4、#8 保留、#9 保留），无静默丢弃。
- **数学等价优化完整性（§1.6.1）**：E1–E4 各含四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」。
- **等价性论证独立推演**：E1 逐位等价成立（max 无算术无舍入）；E2 bf16 精确单射往返成立；E3 `max(S ∪ {−inf}) = max(S)` 对所有 S 成立（含全 −inf / 含 +inf / **含 NaN**——NaN 传播下两侧均 NaN，与 `torch.amax` 一致）；E4 fp16/fp32 输出 cast 为恒等成立。**verify_equiv.py 重跑全部 EQUIV_PASS、447 例 0 违反**，与 §1.6.1 内嵌表一致。
- **向量化替代完整性（§1.6.2）**：盘点 8 个计算点（#1 masked load / #2 fp32 cast / #3 行极值 / #4 输出 cast / #5 形状收缩 / #6 `real_m`+guard / #7 索引算术 / #8 host 元数据），与 §3.3 伪代码、§6 循环表交叉核对——**无遗漏**；#1–#4 已向量替代（API 有佐证，v-prefix `vcast` 优先），#5–#8 保留标量/循环且理由具体到本算子（形状收缩无复合 API、依赖运行时 pid 的 block 边界、block 级索引、host 元数据），均非逐元素热点。§6 中不存在 §1.6.2 未论证的逐元素标量循环。
- **向量化轴与布局决策（§1.6.3）**：规约类必选候选「水平归约 vs 垂直扫描（lane 映射）」**分别枚举**——#1 水平（dim=1 快路径）、#2 垂直（串行化，发射爆炸量化否决）；+ #3 核内转置、#4 N-split、#5 归约源选择。逐候选量化评分（整除性/尾 lane 浪费/累加链/repack/UB）。弃选有量化理由：#2「N 次 (bm,1) 微操作 × ~0.5µs 发射/行」、#3「主选轴 N 即 I/O 原生连续轴、零 repack 收益，另 +2 缓冲」。
- **弃选前提核对**：#3 引 `T.transpose` 为 UB 级二轴交换链——与 `pattern-library/layout.md` PL-1.1-transpose-chain（核内 UB 级、仅支持二轴交换、~5.4µs 实测 verified）一致，无「把核内重排写成 GM 级 transpose」类前提错误；代价「即使 5.4µs 亦无收益可抵」非纯开销论证缺口。
- **实验裁决模式（§1.6.4）**：C3 备选三件套完整——① 分派条件（工厂期静态谓词，评估 per-shape 分派可行性）；② 备选结构（两段 kernel，含 buffer 形状/UB 口径/核内循环/分核三要素，达 Stage 4 可实现深度）；③ 裁决计划（代表 shape、指标、**判定阈值 0.8×**、回写路径）。判定裕度未落纸面单选项。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 向量化结论一致（无被判向量化的点残留逐元素标量循环）；§1.6.3 选定布局（行块 (bm,N) 行主序 + N 轴归约）与 §3.3 伪代码、§4.3 buffer 形状、§6 循环三方一致。

> 本设计无阻塞级问题。3 项建议级注记已内联于维度 1 / 维度 3 及维度 8 相关处（`clear=True` 措辞、路由占位函数、硬件依赖假设 R1–R4/R9），供 Stage 3 实现参考，不影响进入 Stage 3。
