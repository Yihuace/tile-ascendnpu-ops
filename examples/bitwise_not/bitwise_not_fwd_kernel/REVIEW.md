# bitwise_not_fwd_kernel 设计文档检视报告（Stage 2 — 迁移任务）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md
- 任务类型: 迁移（harness，GPU→Ascend NPU）
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/bitwise.py`（`BitwiseNotFwdKernel`，L87-106；`BITWISE_DTYPES` 在 `_dtype.py` L12-19）+ `_base.py`（`UnaryKernel` L158-256 / `_restore_output_dtype` L100-101）+ `_builders.py`（`make_unary_regcopy` L85-121 / `make_unary_direct` L63-81）+ `_policy.py`（`choose_unary_strategy` L165-174）+ `_op_body.py`（`store_unary_bool_as_int8` L71-77）+ `ops/elementwise/bitwise.py`（`BitwiseNotFwdOp` L42-45）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1（§3.3 伪代码 `T.cast(1,"uint8")` 与 §3.2「Python 标量 int」口径的轻微不一致；非阻塞，见维度 1）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。`python3 examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py` 退出码 0，输出与 §1.6.1 内嵌结果表**逐项一致**：采纳项 1「`xor(x, all-ones)` → `vnot(x)`」uint8/int8/int16/int32/int64 各 4101 元素 **mismatches = 0 / EQUIV_PASS**；采纳项 2「`(x==0)` → `(x^1)`（bool 字节存储）」bool 4098 元素 **mismatches = 0 / EQUIV_PASS**；总体结论「全部 EQUIV_PASS — 可采纳」。**无 EQUIV_FAIL**，脚本缺失/执行失败均不成立。
- **design_calc_check**（脚本实际入参为 `--design`；任务书所写 `--dir` 非该脚本参数，改按工具用法执行）：`python3 .agents/tools/design_calc_check.py --design examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md` 输出 JSON `checks` 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）——**机械跳过，人工已核**：§4.5 逐 dtype 峰值 = 2 × block_size × elem_bytes（x_ub + y_ub）——1B（bs=8192）16KB、2B 32KB、4B（bs=8192）64KB、8B（bs=4096）64KB；最大 **64KB < 192KB** ✓；按 §4.5 上偏系数 ×1.12 ≈ 72KB、按 `TRAP-UB-multibuffer-inflation` ×1.7 ≈ 109KB，均 < 192KB。int64 取 bs=4096 使 8B 峰值与 int32 齐平，无溢出。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API in this design"）——**机械跳过，人工已核**：纯 Vector 算子，无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope(Cube)`，L0C 不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 40"）——**机械跳过，人工已核**：`declared logical cores = 40` 为解析伪影（40 是 §5.5 的**物理向量核数**，非逻辑核数）。分核算术独立复算自洽：逻辑核数 = `ceil(N/block_size)`（smoke-1m int32 1048576→**128**、16m int32 →**2048**、16m int64（bs=4096）→**4096**、256m int32 →**32768**）；物理核数实查 `NPUUtils.get().get_aicore_num()` = **20**（本检视独立复跑得 20，与 §5.5 记录一致），纯 Vector ×2 = **40**；`num_kernels = min(num_logical, 40)`，`num_local_tasks = ceil(num_logical/40)`（128→4、2048→52、4096→103、32768→820），与 §5.5 表逐行一致，边界全静态。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 全部源文件后逐项核对，逐条附源码证据。**

- **语义理解正确性**：§0.1 的 dtype 分支与源码 `BitwiseNotFwdKernel.op_func`（`bitwise.py` **L100-106**）逐字一致——`if x.dtype == "bool": return x == T.cast(0, "bool")`（L102-103，逻辑 NOT）；`if x.dtype == "uint8": return T.bitwise_xor(x, T.cast(255, "uint8"))`（L104-105，全 1 字节 0xFF）；`return T.bitwise_xor(x, T.cast(-1, x.dtype))`（L106，有符号全 1）✓。§1.3/§11.3 的 uint8→`⊕0xFF`、有符号整型→`⊕0xFF…FF`（= -1）与源码常数逐一对应 ✓。dtype 集合 `{bool, uint8, int8, int16, int32, int64}`（§0.1/§11.3）与 `_dtype.py` **L12-19** `BITWISE_DTYPES` 完全一致、**无 float** ✓。边界语义「无 NaN/±inf、整型 bit-exact」与源码纯按位运算一致 ✓。host 侧语义「`flat(x)` 展平视图 + `_restore_output_dtype(...).reshape(x.shape)` 回塑」与源码 `UnaryKernel.forward`（`_base.py` **L253-256**：`result = self._compiled_fn(flat(x)); return self._restore_output_dtype(result).reshape(x.shape)`）及 `_restore_output_dtype`（**L100-101**：`view(torch.bool) if self._bool_via_int8`）一致 ✓。
- **算法解读完整性**：§0.3 六步数据流（host 展平 → bool 字节视图 → 分块装载 → dtype-branched `op_func` → 分块写回 → host 回塑）覆盖源码全部计算语句，无遗漏/无臆造。§0.3 对策略路径的判定经源码核对成立：`UnaryKernel.DEFAULT_STRATEGY = "register_copy"`（`_base.py` **L174**），`choose_unary_strategy`（`_policy.py` **L165-174**）对 `input_dtype == torch.bool` 返回 **"direct"** → bool 走 `make_unary_direct`（`_builders.py` **L63-81**，1 元素/线程）并在 `_get_effective_op_func`（`_base.py` **L207-212**）经 `store_unary_bool_as_int8` 落 int8 ✓；非 bool 走 `make_unary_regcopy`（`_builders.py` **L85-121**），其 `contiguous = _narrows(dtype, out_dtype) and N % block_size == 0`（**L96**），而 `_narrows`（**L35-37**）= `DataType(out_dtype).bits < DataType(dtype).bits`；非 bool 路径 `out_dtype == dtype` ⇒ `_narrows=False` ⇒ **不触发** contiguous 分支，走 fragment 全块（`T.copy` 装载 → `T.Parallel` 逐元素 → `T.copy` 写回，**L112-117**）——与 §0.3 判定一致 ✓。bool 结果 `store_unary_bool_as_int8`（`_op_body.py` **L71-77**）源码为 `T.if_then_else(op_func(x), T.cast(1,"int8"), T.cast(0,"int8"))`，与 §0.3「`x==0` 返回 bool，落 int8 字节」一致 ✓。host 侧（Op 层 `UnaryOp.forward` `ops/elementwise/_base.py` **L78-81** `input.contiguous()` 后交 kernel）无遗漏 ✓。
- **优化手段覆盖完整性**：§0.4 六项（register_copy / coalesced 向量化 / `threads×npt` + `T.Parallel` / dtype 分支标量 `op_func` / XOR-with-all-ones 规避 CUDA int4 缺陷 / bool 字节存储 `BOOL_STORAGE_DTYPE=int8`）逐项对应源码（`_base.py` L174、`_builders.py` L112-117、`_policy.py` L12/L60-73、`bitwise.py` L100-106、`_dtype.py` **L9** `BOOL_STORAGE_DTYPE="int8"`），硬件依赖标注正确，未把实现层优化误判为算法层 ✓。§0.4 结语「源实现无 swizzle / 无 mma / 无 warp shuffle / 无 online」与源码逐项核对成立 ✓。
- **耦合性判定合理性**：§0.5 每条目四态处置（保留 / 等价替换 / 重新设计 / 舍弃）齐全：语义层、逐元素单遍、dtype 分支「保留」（纯算法层）；bool 字节存储、XOR-with-all-ones、fragment、coalesced「等价替换」（NPU 有 UB / MTE / 原生 `vnot` 佐证）；`threads×npt` 两级并行「重新设计」（CUDA 线程模型 → 一维 Kernel）；swizzle/mma/shuffle「舍弃（不适用）」并明写理由（源本未使用）✓。处置与 `ascend-constraints`「仅支持一维 Kernel / 内存层级」及 `decision-tree` 不矛盾 ✓。bool i1 不可 GM 存取的判据引用 `docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3，**经亲自打开文档核对成立** ✓。
- **NPU 重设计可行性**：§0.6 R1（`threads×npt` → 一维 `T.Kernel(num_kernels, is_npu=True)` + `block_size` 折叠 + Vector 整块 v-prefix）含「源方案 → NPU 新算法 → 语义保持论证」三要素；语义保持论证以「每元素独立、无跨元素依赖、无规约，整型 bit-exact」为据，并说明尾块「仅 copy 有效元素、垃圾尾部不写回」与源 fragment 尾块等价——满足一维 Kernel / 内存层级 / 32B 对齐 / 静态边界约束 ✓。设计明确「只有 R1 一项重设计」，其余为保留/等价替换，与 §0.5 一致 ✓。
- **设计与决策一致性**：§11.1/§11.2 逐项核对 §0 → §1–§10，且 §0.4 各优化手段意图承接有说明（#1/#2 register_copy/coalesced → UB `T.copy`；#3 `threads×npt` → Vector 整块；#4 dtype 分支标量 → §1.6.2 `vnot`/`vxor`；#5 XOR-with-all-ones → §1.6.1 项 1 升级为原生 `vnot`；#6 bool 字节存储 → uint8 字节视图），**无静默丢弃、无「§0 判重设计、§3 照抄源方案」脱节** ✓。
- **golden 独立性**：§8.1 golden 以 §0.1 源语义为唯一依据（`torch.bitwise_not(x.detach().cpu())`），**不复刻** §0.6 的 NPU 算法（不含 `vnot` / `vxor` / 字节存储 / persistent 结构），且 §0.7 声明与源仓 test `ref_program = torch.bitwise_not(x)`、`atol=rtol=0` 对齐，验证独立性成立 ✓。

### 1. API 可行性 — pass

- **API 存在性（v-prefix 优先）**：§3.2 列出的每条 API 均有 `docs/Tilelang.language/` 佐证——`T.vnot`（`逻辑操作/T.vnot.md`，`T.vnot(src, dst)`，`tilelang::vnotOP → hivm::VNotOp`）、`T.vxor`（`逻辑操作/T.vxor.md`，`T.vxor(src0, src1, dst)`，`tilelang::vxor → hivm::VXorOp`）、`T.copy`（`内存操作/T.copy.md` + 双显式 slice 形态）、`T.Kernel(一维, is_npu=True)`、`T.alloc_ub`、`T.ceildiv`/`T.min` 标量。§3.5.3 引用的 `examples/hardswish/...`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`、`examples/TileOPs/.../bitwise_not/_bitwise_not_fwd_kernels.py` **经 Glob 核对真实存在** ✓。全部为 v-prefix 优先（无 `npuir_xxx` 兼容式调用）✓。
- **API 与模式匹配**：§2 选定 Developer 模式，§7 自动同步（无手动 `set_flag/wait_flag`），与 Developer 模式约束一致 ✓；§2.3 明确无 Expert 手动同步需求 ✓。
- **专项核对（任务要求：整数计算路径的 docs 佐证）**：
  1. **「`T.vnot` 支持 int32/int64」**——**doc-backed**：`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1 DataType 表 int32 = √、int64 = √（且 uint8/int8/int16/uint16/uint32/uint64 均 √，bool/int1 = ×）。设计 §1.6.0/§1.6.1 项 1/§3.4 对该表的引用**与文档逐字一致** ✓。
  2. **「bool 走 uint8 字节域 `vxor`」**——**doc-backed**：`docs/Tilelang.language/逻辑操作/T.vxor.md` §2.2.1 uint8 = √、bool = ×，§2.3「bool(i1) 无法 load/store GM，仅可作片上中间结果」，§2.1「src1 或 Python 标量 int：被提升为与 C 相同 dtype 的常量并广播」；`T.vnot.md` §2.3 同款 bool 限制。设计 §0.5/§3.3/§9.1/§11.4 的 bool 字节存储判据**与文档一致，非未文档化假设** ✓。
- **建议级说明（非阻塞）**：§3.3 伪代码写 `one_u8 = T.cast(1, "uint8")` 后 `T.vxor(x_ub, one_u8, y_ub)`，而 §3.2 API 表将 src1 表述为「Python 标量 int」。`T.cast` 产出的是 TIR 表达式而非 Python int，与文档 §2.1 明示的「Python 标量 int 广播」形态存在轻微口径不一致；Stage 3 直接传字面量 `T.vxor(x_ub, 1, y_ub)` 即规避。属建议级澄清（不影响可编译性与语义），不阻塞。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 为纯 Vector 两级 `GM[x] --T.copy--> UB[x_ub] --vnot/vxor--> UB[y_ub] --T.copy--> GM[y]`，无 GM→L0 越级路径 ✓；§4.5/§4.3 每个 buffer 标注存储层级（`x_ub`/`y_ub` = UB）✓。
- **UB 预算**：§4.5 峰值 2 × block_size × elem_bytes，逐 dtype 最大 **64KB**（int32 bs=8192、int64 bs=4096），< 目标平台 UB **192KB**（`constants.md` `CONST-capacity-910B2C`：UB=192KB/AIV）；×1.12 ≈ 72KB、×1.7 ≈ 109KB 均安全 ✓（机械复核 `ub_budget` skip 项已人工核，见上）。
- **L0C 容量**：无 Cube，不涉及 ✓。**核查混合字节口径（VP-2026-0009 类）**：§4.5 各档位分母口径统一（2×block_size×elem_bytes，Bytes），无 B/KB 混算 ✓。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_size = 4096 if storage_dtype == "int64" else 8192`，理由（同族实测最优、32B 对齐、int64 约束 UB 峰值、宽块摊薄 `CONST-vector-launch-overhead` 发射开销）给出 ✓。
- **对齐约束**：§5.3 逐 dtype 32B 对齐 + 整除向量宽（1B×16、2B×8、4B×4、8B×2）成立 ✓。
- **非整除处理**：§5.4/§6.4 `tail_size = T.min(block_size, remaining)` + `block_id < num_logical` guard，仅 copy 有效元素 ✓。
- **分核策略三要素（物理核数适配，核对标准 `core-split-strategy.md` §1）**：① 逻辑核数计算 = `ceil(N/block_size)` 且 §5.5 给出四 workload 表 ✓；② 物理核数依据 = `NPUUtils.get().get_aicore_num()` **实查**，§5.5 记录查询代码与返回值 **20**，纯 Vector 翻倍 = 40，并如实标注与 `CONST-aicore-910B2C`（24）的差异（以实查为准）——**本检视独立复跑 `get_aicore_num()` 亦返回 20**，非文档假设/经验值 ✓；③ 规模判定与分核方案 = 小规模（逻辑核数 ≤ 40）结论「无需适配」、中等/极大规模固定启动内核数 = 40（物理向量核数 1×）、核内 `T.serial(num_local_tasks)` 串行，`num_local_tasks` 为编译期常量（N/block_size/40 均静态）✓。核数与 block 取值自洽（独立复算 128→4 / 2048→52 / 4096→103 / 32768→820 全对）✓。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（一维 `T.Kernel`）✓；**GEMM 非整除**：不适用（无 GEMM）✓；**L0C 溢出**：不适用（无 Cube）✓。
- §3.5.1 逐条检查「GPU 专用 API」（`T.get_thread_binding`/`T.vectorized`/`T.unroll`/`T.bitwise_xor` GPU 形态/bool i1 GM 存取）并给出改写方案（NPU v-prefix + UB + 字节存储），与源码实际使用的 CUDA 语义一致 ✓。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确核级并行 `T.Kernel` + tile 级 `T.serial`（persistent）+ 元素级 v-prefix 向量化三分，无逐元素标量循环 ✓。
- **同步与模式匹配**：§7 自动同步（Developer），与 §2 模式一致 ✓。
- **尾块处理**：§6.4 明确 `tail_size` + guard，垃圾 lane 不写回 ✓。
- 与 §1.6.2 向量化结论一致（§6 不出现 §1.6.2 未论证的标量热循环）✓。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 含 PyTorch 参考实现草案（`torch.bitwise_not`），以 §0.1 语义为唯一依据、独立于 NPU 算法 ✓。
- **L0 测试计划**：§8.3 `N=1048576`，六 dtype 各一例 + 拒绝 float 负向用例，规则 shape ✓。
- **精度标准**：§8.2 六整型/bool 全 `atol = rtol = 0.0`（bit-exact，与源仓 `_get_tolerances` 精确比较、max_abs_err=0 一致），并注明 §1.6.1 机器验证 mismatches=0；整型无浮点容差，标准按算子类别（整型按位）正确 ✓。

### 7. 完整性与一致性 — pass

- **章节齐全**：含 §0（0.1–0.7 齐全）及模板 §1–§11（概述/编程模式/API 映射/数据规格/内存/Tiling/循环/同步/验证/风险/交付/迁移一致性核对）✓。
- **无占位符**：全文无 `{placeholder}`/`TODO`/`待补充`（§10.2 的「待实现（Stage 3）」为交付清单状态标注，非占位符）✓。
- **同类引用真实**：§3.5.3 所列 `examples/` 路径经 Glob 全部存在 ✓。
- **内部一致**：API 映射（§3.2 用 `vnot`/`vxor`）、内存规划（§4 UB）、Tiling（§5 block_size/分核）三处相互一致；§11.1/§11.2 明确无 §0/§3 脱节 ✓。

### 8. 算法优化分析 — pass

**（§1.6.0 调研独立复核 + §1.6.1 等价性独立推演 + §1.6.2/§1.6.3 覆盖交叉核对，逐项附证据）**

- **§1.6 章节存在性**：§1.6 含 1.6.0 算法调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四子节齐全 ✓。
- **算法调研存在性与完整性（四问）**：R1 候选表**含基线候选**（源公式 dtype-branched XOR/`==`）+ 候选 1/2/3/4 逐项初判 ✓；R2 **明确结论「无在线算法」**并给结构依据（逐元素无跨元素依赖、无 running 统计量、源算法已单遍）✓；R3 **四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）齐备且基线在表，口径写明 ✓；R4 逐候选六维亲和评估 + 结论 ✓；「无更优替代」写明调研范围（algorithm-candidates ALG-elementwise 命中行、恒等变形自查、pattern-library 同族先例、源码 §0.3/§0.4）✓；复杂度级别（单步逐元素）匹配轻量调研深度 ✓。
- **调研结论正确性独立复核**：
  - **等价性独立推演**：① `x ⊕ 全1 = ~x`（固定位宽整型按位取反定义），bit-exact；② 字节存储 bool（字节 ∈ {0,1}）`(x==0) ⇔ (x⊕1)`（0→1、1→0）——两条均由本人独立推演成立；`verify_equiv.py` 重跑 **逐 dtype mismatches=0** 与 §1.6.1 内嵌表一致 ✓。
  - **复杂度独立复算**：FLOPs = `N`（1 op/elem）；访存 = `2N·elem_bytes`（读 N + 写 N）；扫描遍数 = **1**；中间缓冲峰值 = `O(block_size)`（UB 驻留，**无中间 GM 往返**，无漏计）；可并行度 = 完全并行（逐元素独立，无跨核归并）——与 §1.6.0 R3 表逐格一致 ✓。
  - **负向断言复核**：对照 `algorithm-candidates.md` ALG-elementwise 命中行（R2「逐元素映射天然单遍」、R3「融合消除中间 GM 往返」）——本算子无跨元素依赖、无中间 GM 往返，故「无在线变体」结构成立、与参考表**不矛盾**；bitwise-NOT 不可能存在 online 变体（无 running 统计量），**无被漏掉的已知在线/化简候选** ✓。参考表 R1 的「公共子表达式消除/除法乘倒数/cast 链合并」对单步逐元素算子均不适用，§1.6.1 结语已对应说明 ✓。
  - **亲和性淘汰依据核对**：R4 无候选因容量/带宽被淘汰，故未触发 `constants.md` 容量条目淘汰；D-2 roofline 引用的 `CONST-capacity-910B2C`(UB 192KB)、`CONST-vector-launch-overhead`(~0.5µs/op)、`CONST-mte2-degradation`(1M 33.9→16M 30.2→64M 20.9→256M 18.4 GB/s/core，48 核/fp16) **经核对真实存在**；设计将 48 核/fp16 档位外推到 40 核/整型**已显式标注「未实测外推」**并给出近似口径，符合负向/代价类论断证据规则 ✓。
  - **并列候选裁决**：候选 1/2 与基线四口径全同（带宽受限），但 R4 依「一元 vs 二元带常量」「单 op vs 比较+掩码」明确裁决，非纸面并列拍板，无未实证常数依赖，不触发实验裁决三件套 ✓。
- **调研→设计一致性**：§1.6.0 选定算法（整型 `vnot`、bool 字节域 `vxor`）实际驱动 §1.4 算法描述、§3.1 公式拆解、§6 循环结构；迁移任务 §0.5/§0.6 处置与调研选族一致（§0.6 明确 XOR-with-all-ones 升级为原生 `vnot`），§0.4 优化手段意图承接逐项有说明、**无静默丢弃** ✓。
- **数学等价优化完整性**：§1.6.1 对选定算法做系统审视，采纳 2 项各含四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」（项 1：二元带常量→一元；项 2：比较+掩码≥2 op→单 `vxor`）；结语给出「无其他可采纳优化项」的依据（无 CSE、无超越函数、无 cast 链、无中间 GM 往返）✓。
- **等价性论证正确性（独立推演）**：项 1/2 恒等式本人独立推演成立、bit-exact、整数域无舍入；与 §8.2 精度标准（atol=rtol=0）一致 ✓。
- **向量化替代完整性与交叉核对**：§1.6.2 覆盖全部计算点——逐元素计算（整型 `vnot`、bool `vxor`）、数据搬入/搬出（`T.copy`）已全部向量化（替代 API 均有 `docs/Tilelang.language/逻辑操作/`、`内存操作/` 佐证）；保留 2 处标量（block 起始索引 = block 级任务映射；尾块 `T.min` = 动态边界），理由具体且非逐元素热点。与 §3.3 伪代码、§6.1 循环结构**交叉核对无遗漏**（§6 无逐元素标量循环）✓。
- **替代 API 可行性**：`T.vnot`（int32/int64/uint8 √）、`T.vxor`（uint8 √、标量广播）与维度 1 同口径、**经文档核对** ✓（优先 v-prefix）。
- **向量化轴与布局决策（§1.6.3）**：单轴豁免成立（输入展平为 1-D `(N,)`，仅一个非 batch 轴，无广播/窗口/归约/重排候选）；**弃选论证前提核对**——核内无需 repack（1-D 连续，bool `view(uint8)` 为元数据视图非数据重排），本设计不涉及 `T.transpose` 类 GM 级/UB 级重排代价论断，故无与 `T.transpose.md` 矛盾的代价机制；唯一弃选候选 3（`vsub`）以「与候选 2 等价且无更优点数、语义不如 `vxor` 贴切」为由淘汰，属等价性裁决而非「API 不支持/代价高」类负向论断，无需 docs 佐证 ✓；布局决策与 §3.3（1-D buffer）、§4（UB 预算按 1-D block_size 计算）、§6（向量维 = 展平轴）**三方一致** ✓。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（无未论证标量循环）✓。

## Review Result
- stage: 2
- project: bitwise_not
- operator: bitwise_not_fwd_kernel
- output: examples/bitwise_not/bitwise_not_fwd_kernel/REVIEW.md
- is_migration: true
- source_op_path: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/bitwise.py（另读 `_base.py` / `_builders.py` / `_policy.py` / `_op_body.py` / `_dtype.py` / `ops/elementwise/bitwise.py`）
- conclusion: 通过
- validation: pass
- validation_details:
  - 结论行存在: pass
  - 结论一致: pass
  - 维度完整（迁移 9）: pass
  - 维度 0 有源码证据: pass
  - 维度 8 有推演证据: pass
  - 建议完整: n/a（通过）
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义 pass / 算法 pass / 优化 pass / 耦合性 pass / 重设计 pass / 一致性 pass / golden独立性 pass
- algo_opt_findings: 调研完整性 pass / 调研结论正确性 pass / 调研→设计一致性 pass / 优化完整性 pass / 等价论证 pass / 向量化覆盖 pass / 不可替代理由 pass / 替代API佐证 pass / 与§3§6一致性 pass
- suggestion_count: 0
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md、.agents/skills/tilelang-op-design/references/{algorithm-research.md,algorithm-candidates.md}、.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md、docs/Tilelang.language/逻辑操作/{T.vnot.md,T.vxor.md}
- summary: 迁移 9 维度全部 pass；dim0 语义/算法/优化/耦合/重设计逐条源码核对一致，dim8 等价性与复杂度独立复核、负向断言对照参考表成立、vnot/vxor 整数路径 doc-backed，机械复核（verify_equiv 全 EQUIV_PASS + design_calc_check）通过 → 结论通过。
- issues: none
