# bitwise_not_fwd_kernel 算子设计文档

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`
- 具体算子：`BitwiseNotFwdKernel`（L2273–L2289，`UnaryKernel` 子类）；
- 基类与模板：`UnaryKernel`（L680–L887，`DEFAULT_STRATEGY="direct"`，`_build_kernel` / `default_config` / `forward`）；
- 策略工厂：`_make_unary_direct`（L304–L320，1 element per thread）；
- 提取产物（lossy 脚手架）：`examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py`（仅内联 general-int 分支，bool/uint8 分支丢失）；集成包装：同目录 `bitwise_not.py`（K11 折叠 `block_size=2048`，`ascend_mode="Developer"`）。

### 0.1 源算子语义（做什么）

**数学语义**：`y = ~x`（逐元素按位取反 / 按位补码），`ref_api = torch.bitwise_not`。按输入 dtype 分三支（源码 `BitwiseNotFwdKernel.op_func`，L2283–L2289）：

| dtype 分支 | 源公式（`op_func`） | 数学含义 |
|-----------|--------------------|---------|
| `bool` | `x == T.cast(0, "bool")` | 逻辑非（True↔False） |
| `uint8` | `T.bitwise_xor(x, T.cast(255, "uint8"))` | 字节按位取反（`255 - x`） |
| `int8/int16/int32/int64` | `T.bitwise_xor(x, T.cast(-1, x.dtype))` | 二进制补码按位取反（`~x = -x-1`，全 1 掩码 `^ -1`） |

**dtype 语义**：输入输出同 dtype（`OUTPUT_DTYPE=None` → `output_dtype = dtype`）；**无中间精度提升**——纯位运算，逐元素整数域精确运算，无舍入（与 float 类算子的 fp32 中转链完全不同）。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**：整数域无 nan/inf；按位补码是双射（`~~x == x`），各级 min/max 均良定义（`~INT_MIN = INT_MAX`，`~UINT8_MAX = 0`）。空维度/非整除 shape 由 host 侧 flatten 成 1-D `[N]` 后进入 kernel。

**host 侧语义**：`UnaryKernel.__init__` 把任意 shape 的输入展平为 `N_total = numel`（`N_total` 由 Op 层传入），无 host 预处理/后处理（`forward` 仅 `self._compiled_fn(x)`）。`is_fp8` 分支、`_clamp_to_dtype_range`、`_wrap_fp8_accumulation` 对本算子不适用（非 fp8）。

**语义保持基线**：§8.1 golden 以本节语义（等价于 `torch.bitwise_not`）为唯一依据实现，**不得复刻 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始 shape 被 host 侧展平；manifest 有 `[1048576]` / `[4096,4096]` / `[8192,8192]` / `[16384,16384]`） | `bool \| uint8 \| int8 \| int16 \| int32 \| int64` | 1-D 展平输入向量，`N = product(input.shape)` |
| `y` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素按位取反输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`）。
**layout**：1-D 连续（contiguous flatten），无转置。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`direct` 策略，`_make_unary_direct`，L304–L320；源码每个计算语句均归入下表）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `idx = bx * threads_arg + i`（线程级索引） | 循环变量 `bx`,`i` | `idx` | 元素定位 |
| 2 | `y[idx] = op_func(x[idx])`（逐元素按位取反） | `x[idx]` | `y[idx]` | `y = ~x`（三分支见 §0.1） |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(thread 标量 load x[idx])--> register --op_func(逐元素 XOR/comparison)--> register --(store y[idx])--> GM[y]
（1 element per thread；无 SMEM、无 fragment 双缓冲、无分块 T.copy）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, threads), threads=256)` —— 一维 grid（`bx`）+ 256 线程/block 两级并行；每线程处理 1 个元素（`T.Parallel(threads)`，`for i in T.Parallel(threads_arg)`）。`threads` 默认 256，`block_size = 256`（direct 策略不乘 npt）。

**host 侧逻辑**：`N_total`、`dtype` 由 Op 层传入；`forward` 直接调用编译后 kernel；`bool` 输入在 `UnaryKernel.__init__` 被强制 downgrade 到 `direct` 策略（TileLang CUDA 无法 lowering `boolx<N>` 向量化 load，L742–L751）。无 padding、无 reshape 计算（flatten 由 Op 层完成）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `direct` 策略（1 element per thread） | 最简 codegen，无寄存器分块/无 SMEM | 每线程独立处理一个元素，`idx=bx*threads+i` | GPU 大线程数（256/block）+ 大 grid 的块级并行 | **硬件强相关**（GPU 线程模型） |
| 2 | 全内存带宽直读直写（Global→Register→Global） | 带宽受限算子，最小化访存层级 | 无中间 buffer，一次读一次写 | memory coalescing（连续 `idx` 访问） | 实现层（NPU 需换为 GM→UB→GM，见 §0.5） |
| 3 | 按 dtype 分派的位运算（`xor(-1)` / `xor(255)` / `x==0`） | 用通用位运算替代 `T.bitwise_not` | 源码注释：`T.bitwise_not` 在向量化 `int4` CUDA 类型上失败（L2276–L2277），故统一用 XOR 补数 | CUDA codegen 限制（非硬件算力） | 实现层（NPU 有原生 `T.vnot`，见 §0.5） |
| 4 | `bool` 强制标量路径（`direct`） | 规避 `boolx<N>` 向量化 load 的 CUDA lowering 缺口 | 强制 `strategy="direct"` | CUDA codegen 限制 | 实现层（NPU 对 bool 另有约束，见 §0.5） |
| 5 | 无 autotune/无 strategy 搜索（bitwise 固定 `direct`） | 带宽受限 + 极简逐元素，搜索 ROI 低 | `DEFAULT_STRATEGY="direct"` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |

> 识别不出来 ≠ 不存在。上述 5 项为源码中可识别的全部优化/规避手段；后续迁移必须逐项处置（§0.5），不得静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = ~x`（三分支） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| 计算算法（步骤 1–2 逐元素映射） | 算法 | 无（纯数学顺序） | — | **保留** | 逐元素位运算，纯数学、无跨元素依赖，NPU Vector 直接支持 |
| 优化 #1 `direct`（1 elem/thread）+ 256 线程/block 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel，无 `threads=`） | **重新设计** | → §0.6 R1：一维 `T.Kernel(core_num, is_npu=True)` + `T.serial` 核内网格跨步 |
| 优化 #2 直读直写 Global→Register→Global | 优化 | GPU register/memory coalescing | 有（GM→UB→GM） | **等价替换** | `T.copy` GM→UB + v-op + `T.copy` UB→GM（migration-analysis.md §5.3 shared memory→UB） |
| 优化 #3 `xor(x, 掩码)` 替代 `bitwise_not` | 优化 | CUDA codegen 限制 | 有（原生 `T.vnot`） | **重新设计（算子降代换）** | → §0.6 R2：NPU 原生 `T.vnot(src, dst)` 单指令等价替换「掩码 + XOR」两指令；见 §1.6.1 |
| 优化 #4 `bool` 强制标量路径 | 优化 | `boolx<N>` CUDA lowering 缺口 | 无直接等价（NPU bool i1 无法 load/store GM） | **重新设计** | → §0.6 R3：以 bool 的 1 字节存储语义处理（int8 视图 + `T.fill(1)` + `T.vxor`）；见 §0.1 bool 分支 |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| host 侧 flatten（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D；host 不修改张量内容（仅视图/元数据） |

> 依据：migration-analysis.md §5.3 能力映射表（shared memory→UB、block×thread 两级并行→重新设计、一维 Kernel）；NPU bool/vnot 约束见 `docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3 与 §1.6.0/§0.6 的设计期探针实证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, 256), threads=256)` + `T.Parallel(256)`，每线程 1 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 40）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、向量 op、`T.copy` UB→GM。**模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、`examples/elementwise/example_elementwise_exp2.py` L28–L43（`T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(总任务,BLOCK_SIZE))` + `if block_id < 总任务`）。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。

**重设计项 R2：整数/uint8 路径降代换（掩码 XOR → 原生 `T.vnot`）**

- **源方案**：`xor(x, all_ones_mask)`（int 用 `-1`、uint8 用 `255`），性能意图 = 用通用 XOR 得到按位补码，规避 CUDA 向量化 `int4` 上的 `bitwise_not` 失败。
- **NPU 新算法**：`T.vnot(x_ub, y_ub)`（`tilelang/language/__init__.py:158 npuir_not as vnot`；`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1 支持 int8/int16/int32/int64/uint8）——单条向量指令完成按位取反，**无需materialize 掩码常量**。
- **语义保持论证**：两类公式均为「按位取反」的精确实现——补码恒等式 `~x ≡ x ^ (-1)`（有符号，全 1 掩码），`~x ≡ x ^ 255`（uint8，8 位全 1）；位运算逐位独立、无舍入，两式对输入域内**每一个取值**逐位相同（见 §1.6.1 verify_equiv.py 机器验证）。边界：`~INT_MIN = INT_MAX`、`~255 = 0` 等极值由位运算定义直接给出，无溢出。

**重设计项 R3：bool 路径重设计（`x==0` → 1 字节存储语义的 `T.vxor(x,1)`）**

- **源方案**：`x == T.cast(0, "bool")`（CUDA 标量比较，产出 bool）；性能意图 = bool 逻辑非，规避 `boolx<N>` 向量化 load 缺口。
- **NPU 新算法**：NPU 侧 bool（i1）**无法 load/store GM**（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3 / `T.vxor.md` §2.3），故以 bool 的物理存储（1 字节，取值 {0,1}）语义处理：kernel 内部按 `int8` 存储视图计算，`T.fill(ones_ub, T.cast(1,"int8"))` + `T.vxor(x_ub, ones_ub, y_ub)`；即 `y = x ^ 1`。**设计期探针实证**（见 §9 风险点与 RETROSPECTIVE）：bool-as-int8 存储路径 `fill(1)+vxor` 实测 bit-exact 正确；直接对 `bool` dtype 用 v-op 会触发 device `vector core timeout`。
- **语义保持论证**：对布尔存储 `x ∈ {0,1}`，`x == 0` 与 `x ^ 1` 在 {0,1} 上逐值同构（0↔1）；`torch.bool` 的规范存储即 {0,1}（workload 由 `torch.rand>0.5` 生成，见 `examples/TileOPs/tileops/workloads/elementwise.py`），故字节级结果与 `x==0` 完全一致。边界：无 nan/inf；`~False=True`→`0^1=1`、`~True=False`→`1^1=0` 均正确。

> 承接说明（源优化意图不丢弃）：§0.4 优化 #1 的「大规模并行吃带宽」意图 → R1 网格跨步并行承接；#2 的「最小访存层级」→ 等价替换为 GM→UB→GM（仅 UB 一级）；#3 的「规避补码指令缺失」意图 → R2 源意图在 NPU 上**不再需要**（NPU 原生有 `T.vnot`，且比 XOR 更省一条掩码搬运），属算法替代 §6.2 模式；#4 的「bool 可用」意图 → R3 以存储语义承接；#5 保留。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`BitwiseNotFwdKernel`，L2273–L2289）。
- 源仓参考/测试基准：`examples/TileOPs/tests/ops/test_bitwise.py`（`ref_program = torch.bitwise_not(x)`，`exact_compare`）；workload：`examples/TileOPs/tileops/workloads/elementwise.py` 的 `BitwiseNotWorkload`。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `BitwiseNotFwdOp` 条目。
- golden 以 §0.1 语义为依据（`torch.bitwise_not`），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`bitwise_not_fwd_kernel`（项目 `bitwise_not`，目录 `examples/bitwise_not/bitwise_not_fwd_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素做按位取反（bitwise NOT / 按位补码），语义等价于 `torch.bitwise_not`；按输入 dtype 分三支（bool 逻辑非 / uint8 字节取反 / 有符号整型补码取反），输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖。

### 1.3 数学公式

$$
y_i = \lnot x_i \quad\text{(bitwise NOT)}, \quad i = 0,1,\ldots,N-1
$$

按 dtype 展开（§0.1）：

| dtype | 公式 |
|-------|------|
| bool | $y_i = \lnot x_i$（逻辑非，等价 $x_i = 0$） |
| uint8 | $y_i = x_i \oplus 255$ |
| int8 / int16 / int32 / int64 | $y_i = x_i \oplus (-1) \;(\equiv \lnot x_i)$ |

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法存在差异：**

- **整数路径（int8/int16/int32/int64）与 uint8 路径**：单步位反——`y = vnot(x)`（源用「掩码 + XOR」两步，NPU 用原生单指令 `T.vnot`，§0.6 R2 / §1.6.1 优化项）。
- **bool 路径**：以 1 字节存储语义做 `y = x ^ 1`（`T.fill(1)` + `T.vxor`），等价源 `x == 0`（§0.6 R3）。
- **并行结构**：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步（源为 thread 级并行，§0.6 R1）。

除上述三点外，算法与 §1.6.0 调研选定的算法族（逐元素位反，无更优替代）一致：一趟扫描、无中间缓冲、无跨元素依赖。

### 1.5 数据流图

```
整数/uint8 路径：
GM[x] --T.copy--> UB[x_ub] --T.vnot--> UB[y_ub] --T.copy--> GM[y]

bool 路径（1 字节存储视图）：
GM[x(bool,1B)] --T.copy--> UB[x_i8] ┐
                                     ├--T.vxor--> UB[y_i8] --T.copy--> GM[y(bool,1B)]
                UB[ones=fill(1)]  ───┘
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再做循环/标量向量化替代分析，最后做向量化轴与布局决策。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 计算步骤仅 2 步、无规约/窗口/矩阵），按 `_shared/standards/algorithm-research.md` §3 执行**轻量调研**（四问各一行结论 + 依据），但四问逐问回答。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：源算法「掩码 XOR」直译 | `y = x ⊕ mask`（int:`-1`、uint8:`255`）；bool:`x==0` | — | — | ✅（基线） |
| 1 | **原生位反指令**（算子降代换） | `y = vnot(x)`（int/uint8） | 数学恒等：补码 `~x ≡ x⊕(-1)`、uint8 `~x ≡ x⊕255`（逐位恒等，见 §1.6.1 机器验证） | 少算：省去掩码常量的 materialize 与一条向量指令（2 op → 1 op）；少搬：无掩码 buffer | ✅ |
| 2 | bool 存储语义改写 | `y = x ⊕ 1`（1 字节存储，bool） | 容差内/域内等价：对 `x∈{0,1}`，`x⊕1 ≡ (x==0)`（见 §1.6.1 机器验证） | 使 bool 分支在 NPU 可落地（源 `x==0` 产出 i1 无法 load/store GM） | ✅ |
| 3 | 代数变形（De Morgan / 交换律 / 结合律等） | 单目逐元素位反，无可结合结构 | 无适用变形 | 无 | ❌（单目映射无可重排的代数结构） |

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖，不存在可被「单遍/流式」消去的多遍基线）；结构判据：本算子无 running 统计量（无归约/无 max/无 sum），在线变体定义不适用。依据：源算法（§0.3）即单遍直写；`examples/elementwise/vec_add_1d.py` 等同类逐元素实现全部单遍。

**R3 复杂度对比**（四口径；本算子带宽受限，访存/遍数是主导项）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（源「掩码 XOR」直译） | N 次 XOR（按 1 bit 记 1 op；口径：不含掩码常量 materialize） | 2·N·elem_bytes（x 读 1 + y 写 1；**不含**掩码常量的较小额外流量，若掩码落 UB 则为 0 GM 往返） | 1 遍 | O(block_size)（UB 单块驻留，另需 1 个掩码 buffer） | 可分块给多核，无跨核归并代价 |
| 候选 1（`vnot`，采纳） | N 次位反（同为 N op，向量单位同一指令） | 2·N·elem_bytes（不变；**无掩码 buffer**，UB 占用 −1 buffer） | 1 遍 | O(block_size)（UB 单块驻留，少 1 buffer） | 同上 |
| 候选 2（`vxor(.,1)`，采纳，bool 专用） | N 次 XOR | 2·N·elem_bytes | 1 遍 | O(block_size)（bool 路径需 3 个 1B buffer = 6KB @bs2048） | 同上 |

> 口径说明：FLOPs 按源码「每元素 1 次位运算」计；访存 = 输入读 + 输出写，**不含**掩码常量（候选 1 相对基线少一处掩码 materialize）；扫描遍数 = 同一数据的 GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。三候选访存同为理论下界 `2N·elem_bytes`，差异在**指令数/UB buffer 数**（候选 1 少一条指令与一个 buffer）。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（掩码 XOR） | 逐元素位运算 → Vector；但需额外掩码常量（`T.cast(-1,dtype)` 作为 src1）——`hivm.hir.vxor` 实测**不接受标量 src1**（`Check failed: binaryOp.Src1().IsTensor()`），必须 materialize 掩码张量，多一条 `fill`/`brc` 指令 | UB：x+y+mask 3 buffer | uint8/int8 x4 对齐；尾 lane 由 v-op 整 buffer 处理 | N 编译期常量 ✓ | 单步无多阶段 | ⚠️ 可实现但多 1 指令 + 1 buffer（本仓 `hivm.hir.vxor` 标量 src1 限制实证） |
| 候选 1（`vnot`） | 逐元素位反 → Vector 原生 `VNotOp`（`docs/.../T.vnot.md` §3） | UB：x+y 2 buffer ✓ | 同上；`T.vnot` 支持 int8/16/32/64/uint8（同文档 §2.2.1） | N 编译期常量 ✓ | 单步 | ✅ **采纳**（单指令、少 buffer、原生支持） |
| 候选 2（bool `vxor(.,1)`） | 逐元素 XOR → Vector（`docs/.../T.vxor.md` §3 `hivm.hir.VXorOp`） | UB：x+ones+y 3×1B buffer = 6KB ✓ | 1 字节元素，UB 32B 对齐 ✓ | ✓ | 单步 | ✅ **采纳**（bool 唯一可行路径；探针实证 bit-exact） |

> 负向淘汰依据：基线候选「掩码 XOR」的问题不在算力而在**指令/IR 约束**——`hivm.hir.vxor` 的 src1 必须为 tensor（实测编译错误 `The second operand of "hivm.hir.vxor" cannot be a scalar`），故走掩码路径必须额外 materialize 掩码张量；候选 1 直接消除该开销。bool 候选 2 的负向前提（bool i1 不能 load/store GM）见 `docs/Tilelang.language/逻辑操作/T.vnot.md` §2.3、`T.vxor.md` §2.3、`T.vbrc.md` §2.3，且经设计期探针实证（对 bool dtype 用 v-op 触发 device `vector core timeout`）。

**调研结论**：**选定候选 1（`T.vnot`）+ 候选 2（bool 存储语义 `vxor(.,1)`）；无更优替代。** 关键依据：① R3 三候选访存同为下界，候选 1 以更少指令/UB buffer 胜出；② R4 中基线「掩码 XOR」受 `hivm.hir.vxor` 标量 src1 限制，候选 1 用原生 `VNotOp` 单指令实现；③ 逐元素位运算无在线/多遍/矩阵结构，无更优算法族。**调研范围**：① `tilelang-op-design/references/algorithm-candidates.md` 对「逐元素按位取反」无命中条目（单目位运算）；② `examples/elementwise/` 全系（`vec_add_1d/2d/2d_dynamic_shape/auto_brc`、`example_elementwise_exp2/log2/add`）均为单遍逐元素直译，无化简/在线变体；③ pattern-library 命中 `elementwise.md`（`PL-1.6-copy-floor` 搬运效率维度、`PL-1.17-subfp32-fp32-transit` 为 fp16/bf16 中转、本算子整数域不适用）——**注：`PL-1.6-copy-floor` / `PL-1.10` / `PL-1.14` / `PL-1.17` 四条已 stale，本设计不作为决策依据**（仅作背景）；④ 结构分析：单目逐元素位运算无规约/窗口/矩阵/代数重排空间；⑤ 源算法（§0.3/#0.4）是候选基线之一，其「规避补码指令」意图在 NPU 上由原生 `T.vnot` 直接满足（§0.6 承接说明）。本结论是 §1.4 与 §1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 `CONST-mte2-degradation`（constants.md，版本戳 tilelang 0.1.2+15ad002b3d / CANN 8.5.0 / 2026-09-21）：fp16 4-set 轮换 1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；**该常数绑定 Ascend910B2C、且 `PL-1.6-copy-floor` 同源条目已 stale——带宽数字标注「待重验」，仅作量级估算**。本设备 Ascend910_9362 实查 Vector 核 40。
- 发射项 = 向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`）÷ 核数并行度（每核摊到的 tile 数 × 0.5µs）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 `block_size = 2048`、物理 Vector 核 = 40 估算（按每核算流量）：

| workload | N | bytes | 流量项（每核 ÷ 带宽） | 发射项（n_num/40 × 0.5µs） | **估算下界 max** |
|----------|---|-------|----------------------|----------------------------|------------------|
| smoke int32 `[1048576]` | 1,048,576 | 8.0 MB | 6.2 µs | 6.4 µs | **≈ 6.4 µs** |
| smoke int64 `[1048576]` | 1,048,576 | 16.8 MB | 12.4 µs | 6.4 µs | **≈ 12.4 µs** |
| 16m int32 `[4096,4096]` | 16,777,216 | 128 MB | ≈ 111 µs | ≈ 102 µs | **≈ 111 µs** |
| 64m int32 `[8192,8192]` | 67,108,864 | 512 MB | ≈ 642 µs | ≈ 410 µs | **≈ 642 µs** |
| 256m int32 `[16384,16384]` | 268,435,456 | 2.0 GB | ≈ 2918 µs | ≈ 1638 µs | **≈ 2918 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m int32` 估算下界 ≈ **6.4 µs**（流量 6.2 + 发射 6.4 + 容量 0，引用 `CONST-mte2-degradation` / `CONST-vector-launch-overhead`，带宽数字待重验）；大 N 档由流量项主导（256m ≈ 2918 µs 下界）。小 N 档流量与发射项同量级 → block_size 是发射项的关键杠杆（发射项 ∝ 1/block_size，见 §5.2）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法。采纳项完成「原式 → 优化后 → 等价性论证 → 收益量化」四要素，并附机器验证。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 整数路径算子降代换（掩码 XOR → 原生位反） | `y = x ⊕ (-1)`（需 materialize 全 1 掩码张量） | `y = vnot(x)` | 补码恒等式 `~x ≡ x ⊕ (-1)`：位级恒等（每一位独立取反），对 int8/16/32/64 全取值域逐位相同；无舍入（整数位运算） | 向量指令 2→1（省 XOR + 掩码 materialize）；UB buffer −1（无掩码张量）；GM 流量不变 |
| 2 | uint8 路径算子降代换 | `y = x ⊕ 255` | `y = vnot(x)` | `~x ≡ x ⊕ 255`（8 位全 1 掩码），uint8 全取值域逐位相同 | 同 #1（uint8 场景） |
| 3 | bool 路径域内等价改写（存储语义） | `y = (x == 0)`（产出 i1，NPU 不可 load/store GM） | `y_byte = x_byte ⊕ 1` | 对布尔存储 `x∈{0,1}`：`x⊕1 ≡ (x==0)`（0↔1 同构）；`torch.bool` 规范存储即 {0,1} | 使 bool 分支可在 NPU 落地；指令数与 #1 同（fill 1 条 + vxor 1 条，fill 为常量物化） |

**优化结论**：采纳 3 项；**优化后公式**（供 §3.1 使用）：
- `int8/int16/int32/int64/uint8`：`y = T.vnot(x)`
- `bool`（存储视图 int8）：`y = T.vxor(x, 1)`

三项均为**位级精确等价**（无 fp16/bf16 舍入问题，整数/布尔域无中间精度语义），已通过 `verify_equiv.py` 机器验证（结果表见下）。

**等价性机器验证（D-1）**：脚本 `examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py`（CPU，任意精度 Python int 精确参照域，随机张量 + 角点值：各 dtype min/max/0/±1/符号位边界；判定基准 = 原式（基线）在同域的输出）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，ALL 6 ITEMS EQUIV_PASS）：

| 采纳项 | 候选式 | 基线式 | samples | 最大 ulp 差 | 违反率 | 结论 |
|--------|--------|--------|---------|------------|--------|------|
| #1 int8 | `~x` | `x ^ -1` | 1,048,586 | 0 | 0.000000 | **EQUIV_PASS** |
| #1 int16 | `~x` | `x ^ -1` | 1,048,586 | 0 | 0.000000 | **EQUIV_PASS** |
| #1 int32 | `~x` | `x ^ -1` | 1,048,586 | 0 | 0.000000 | **EQUIV_PASS** |
| #1 int64 | `~x` | `x ^ -1` | 1,048,586 | 0 | 0.000000 | **EQUIV_PASS** |
| #2 uint8 | `~x`（`= ~x & 0xFF`） | `x ^ 255` | 1,048,583 | 0 | 0.000000 | **EQUIV_PASS** |
| #3 bool | `x_byte ^ 1` | `(x == 0)` | 1,048,580 | 0 | 0.000000 | **EQUIV_PASS** |

> 说明：全部为位级恒等，ulp 差恒为 0、违反率 0；int64 的 `x ^ -1` 在 64 位域即 `x ^ 0xFFFF...FF`（脚本以任意精度 int 计算，避开 int64 无符号溢出）。#3 在 {0,1} 布尔域逐值对照。无失败项（全部 EQUIV_PASS）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素按位取反（int/uint8） | 源 `T.Parallel` 内标量 `xor` | `T.vnot(x_ub, y_ub)`（佐证：`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.4 + 设计期探针实测 exact） | ✅ | — |
| 2 | 逐元素 bool 逻辑非 | 源标量 `x == 0` | `T.fill(ones_ub, 1)` + `T.vxor(x_ub, ones_ub, y_ub)`（佐证：`docs/.../T.vxor.md` §2.4、`T.fill` 见 `tilelang/language/customize_npuir.py:702 npuir_fill` + 探针实测 exact） | ✅ | — |
| 3 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环，每核迭代数 `ceildiv(n_num, 40)`（如 1M@bs2048 = 13 次），非逐元素热点；`nvnot`/`vxor` 作用于整 buffer，元素级并行由向量指令完成。模式先例 `vec_add_2d.py` L32 |
| 4 | block 起始索引 / 任务映射 | `task = i*core_num + cid`、`off = task*block_size` | 无 | ❌ | block 级索引/任务映射计算，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 5 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape - off)` | 无（本设计对 manifest 整除 shape 用静态整块，见 §5.4） | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 整除 block_size，设计保留 `shape: T.int32` 守卫以支持任意 N（纯静态化对通用 N 不成立） |

**向量化结论**：逐元素计算点（#1/#2）已全部向量化（`T.vnot` / `T.fill`+`T.vxor`，均有 docs + 探针双佐证）；保留 3 处标量/循环（#3 核内调度循环、#4 block 索引、#5 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：manifest 对齐 → int32 元素 4B，向量宽 ×4 → `block_size=2048` 每 tile 2048/4=512 向量槽，尾 lane 浪费 0%（2048 整除 4）；int64/int16/int8/uint8 均按 32B 尾轴对齐（2048×elem_bytes 为 32B 整数倍）。无累加链（无串行累加维）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
- **迁移注**：GPU 源码的并行轴选择（thread `i` → 元素 `idx`）是**输入而非结论**——NPU Vector 轴与之不对应，§0.6 R1 已独立评估为「核内 grid-stride + 向量指令整 buffer」。
- **决策落地**：主选轴 = 元素轴，同步落入 §3.3 伪代码与 §6 循环结构（`T.vnot` / `T.vxor` 直接作用于 UB 一维 buffer；无逐元素标量循环）。

**布局决策结论**：选定方案（唯一候选轴）：核内布局 = 一维连续 `[block_size]`（I/O 布局即核内布局，无 repack），向量化轴 = 唯一 1D 元素轴，repack 路径 = 无。弃选候选：无（单轴算子，无第二轴可枚举，按 skill Phase 2 第 3 项豁免）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素位运算），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 位运算 API | 原生 v-prefix `T.vnot` / `T.vxor` | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步位运算 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`（Developer 模式 + `T.alloc_ub` + v-prefix API）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB 分配（v-prefix API 要求 buffer 在 UB） |
| 计算方式 | v-prefix API（`T.vnot` / `T.fill`+`T.vxor`），非 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（`y = T.vnot(x)`；bool 路径 `y = T.vxor(x, 1)`）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2 | `y_tile = ~x_tile`（int/uint8） / `x_tile ⊕ 1`（bool） | 向量位反 / bool 存储语义逻辑非 |
| 3 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+n], x_ub[0:n])` | Developer |
| 位反（int/uint8） | `y = ~x` | `T.vnot` | `T.vnot(x_ub, y_ub)`（同 shape、同 dtype UB tensor） | Developer |
| 常量掩码（bool） | `ones = 1` | `T.fill` | `T.fill(ones_ub, T.cast(1, "int8"))`（`npuir_fill`，`customize_npuir.py:702`） | Developer |
| 异或（bool） | `y = x ⊕ 1` | `T.vxor` | `T.vxor(x_ub, ones_ub, y_ub)`（src1 必须为 tensor，非标量） | Developer |
| 搬出 | `y[...] = y_tile` | `T.copy` | `T.copy(y_ub[0:n], y[off:off+n])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |

### 3.3 计算伪代码

> 遵循迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；`T.Tensor` 的 dtype 作为第二位置参数（非 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 类 false alarm。

```python
@tilelang.jit(out_idx=[1], target="npuir")
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    # dtype 分派（编译期 Python 分支，无运行时开销）
    is_bool = (dtype == "bool")
    k_dtype = "int8" if is_bool else dtype      # bool 走 1 字节存储视图
    out_dtype = output_dtype or k_dtype
    n_num = T.ceildiv(N, block_size)            # 逻辑 tile 数（N 编译期常量 → 静态值）
    core_num = 40                               # 物理 Vector 核数（NPUUtils 实查 20 aicore ×2）

    @T.prim_func
    def main(
        x: T.Tensor((N,), k_dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), k_dtype)
            y_ub = T.alloc_ub((block_size,), k_dtype)
            ones_ub = T.alloc_ub((block_size,), k_dtype)   # 仅 bool 分支使用

            for i in T.serial(T.ceildiv(n_num, core_num)):   # 静态循环上界
                task = i * core_num + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off:off + tail], x_ub[0:tail])
                    if is_bool:                              # 编译期分支
                        T.fill(ones_ub, T.cast(1, k_dtype))
                        T.vxor(x_ub, ones_ub, y_ub)
                    else:
                        T.vnot(x_ub, y_ub)
                    T.copy(y_ub[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：`if is_bool:` 为 Python 层编译期条件（JIT 参数编译期确定），不产生运行时分支。`T.copy` 的 slice `[0:tail]` 形态参考 `examples/elementwise/vec_add_1d.py` L34/L38；`T.serial` + `if task < n_num` 网格跨步参考 `examples/elementwise/vec_add_2d.py` L32–L42。v-op 作用于整 buffer（`x_ub`/`y_ub` 全 `block_size`），尾部为未初始化数据但不影响仅搬出的 `[0:tail]` 有效部分（沿用 vec_add_1d 语义）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.vnot(src, dst)` | `docs/Tilelang.language/逻辑操作/T.vnot.md`（§2.2.1 支持 int8/16/32/64/uint8；§2.4 示例）；`tilelang/language/__init__.py:158 npuir_not as vnot` | ✅ **设计期探针实测**：int8/int16/int32/int64 `vnot` 与 `torch.bitwise_not` **bit-exact**；uint8 计算值正确（见 §9 风险点 R-2 的 out_idx dtype 注意项） |
| `T.vxor(src0, src1, dst)` | `docs/Tilelang.language/逻辑操作/T.vxor.md`（§2.4 示例）；`__init__.py:89 npuir_xor as vxor` | ✅ 实测：src1 必须为 tensor（**不可为标量**，报 `hivm.hir.vxor ... cannot be a scalar`）；int8 tensor operand 路径 bit-exact |
| `T.fill(buffer, value)` | `tilelang/language/customize_npuir.py:702 npuir_fill`；`__init__.py:119 npuir_fill as fill` | ✅ 实测：`value` 须为 `PrimExpr`（`T.cast(1, "int8")` 可用；Python int 报 `value must be a tir.PrimExpr`） |
| `T.copy(src, dst)` | `examples/elementwise/vec_add_1d.py` L34/L38（slice 形态） | ✅ 已核对 |
| `T.alloc_ub((shape,), dtype)` | `examples/elementwise/vec_add_1d.py` L28–L30 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `examples/elementwise/vec_add_2d.py` L28–L42、`example_elementwise_exp2.py` L28–L43 | ✅ 已核对 |

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用 `T.bitwise_xor` + `T.Parallel` + `threads=`） | 换为 `T.vnot`/`T.vxor`/`T.fill` + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, 2048)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| bool i1 不能 load/store GM | Yes | bool 走 1 字节存储视图（§0.6 R3），kernel 内按 `int8` 处理 |

### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,threads), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(256)` 线程级 1 elem/thread | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| 位反 API | `T.bitwise_xor(x, mask)`（2 op，规避 CUDA int4） | `T.vnot(x)`（1 op，原生） | §0.6 R2 / §1.6.1 #1–#2 |
| bool 处理 | `x == T.cast(0,"bool")`（标量比较） | bool 1 字节存储视图 `T.vxor(x,1)` | §0.6 R3 / §1.6.1 #3 |
| Buffer 层级 | `T.Parallel` 直读直写（register） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 不改变 x/y 顺序与名称，新增 shape 用于尾块守卫 |
| strategy/autotune | `direct` + `threads` | 无 strategy 维度、无 autotune | K8/K9 移除（bitwise 固定单实现） |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 一维 `T.Kernel(n_num, is_npu=True)`、`T.alloc_ub`、`T.copy` slice、v-prefix 一元/二元 API、`shape` 动态尾块 |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(总任务, BLOCK_SIZE))` + `if block_id < 总任务` 网格跨步（本设计分核先例） |
| `examples/elementwise/example_elementwise_exp2.py` | 相似 | 二元 v-op（`T.vexp2`）+ `Tmp` buffer + `T.serial` persistent 形态 |
| `docs/Tilelang.language/逻辑操作/T.vnot.md` | 高度相似（API） | `T.vnot(src,dst)` 用法与 dtype 支持表 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/DESIGN.md` | 相似（迁移形态） | 同族 elementwise 迁移 DESIGN 结构（v-prefix + UB + 尾块） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)`（原始 shape 展平） | `bool \| uint8 \| int8 \| int16 \| int32 \| int64` | 1-D 展平输入向量，`N = product(shape)` 编译期常量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | `same_as(x)` | 输出，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | `k_dtype`（bool→int8 存储） | UB | 输入 tile 缓冲 |
| `y_ub` | `(block_size,)` | `k_dtype` | UB | 输出 tile 缓冲 |
| `ones_ub` | `(block_size,)` | `int8` | UB | 仅 bool 分支的全 1 掩码（`T.fill`） |

### 4.4 内存搬运路径

```
整数/uint8 路径（纯 Vector，无 L1/L0）：
GM[x] --T.copy--> UB[x_ub] --T.vnot--> UB[y_ub] --T.copy--> GM[y]

bool 路径（1 字节存储视图）：
GM[x(bool, 1B elem)] --T.copy--> UB[x_ub(int8)]
UB[ones_ub] --T.fill(1)-- (常量物化)
UB[x_ub] + UB[ones_ub] --T.vxor--> UB[y_ub] --T.copy--> GM[y(bool, 1B elem)]
```

### 4.5 UB 内存预算

`block_size = 2048`：

| dtype | x_ub | y_ub | ones_ub | 合计 | ×1.7（auto-multibuffer 上偏系数，`CONST-capacity-910B2C`） |
|-------|------|------|---------|------|-------------------------------------------------------------|
| int64 | 16 KB | 16 KB | — | 32 KB | 54 KB < 192 KB ✓ |
| int32 | 8 KB | 8 KB | — | 16 KB | 27 KB ✓ |
| int16 | 4 KB | 4 KB | — | 8 KB | 14 KB ✓ |
| int8/uint8 | 2 KB | 2 KB | — | 4 KB | 7 KB ✓ |
| bool（int8 存储） | 2 KB | 2 KB | 2 KB | 6 KB | 10 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C；本设备 Ascend910_9362 同规格）。最大占用（int64）54 KB，余量充足 → block_size 有放大空间（§5.2 讨论发射项权衡）。膨胀系数注：`CONST-capacity-910B2C` 同时记录「显式 alloc 结构实测 ~12% 余量即足」（2026-09-16 反例数据点），本设计 buffer 数少且逐 buffer 显式分配，实际上偏更小。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N` 的实际长度 | `shape: T.int32`（prim_func 参数） | 1 ~ manifest 最大 268,435,456 |

> `N` 同时作为编译期 `T.Tensor((N,), k_dtype)` 的 shape（按特定 N 特化编译）与运行时 `shape` 参数（尾块守卫）。manifest 全部 workload 的 N 均整除 `block_size=2048`，但保留 `tail` 守卫以支持任意 N。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],        # y 是 prim_func 第 2 个参数（index=1）
    target="npuir",
)
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素位运算（`T.vnot` / `T.vxor`），无 matmul、无归约、无 Cube 计算，判定为纯 Vector；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = 2048   # K11: 源 threads(256) × npt(8) 折叠为单一参数
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size=2048 选择理由**：
1. 与源一致（K11 折叠 `threads × num_per_thread = 256 × 8 = 2048`），迁移可追溯；
2. UB 对齐：2048×elem_bytes 是 32B 的整数倍（int8 2048B、int64 16384B）✓；
3. UB 容量：最坏 int64 两 buffer 32 KB（×1.7 ≈ 54 KB）< 192 KB ✓；
4. 发射项权衡：发射项 ∝ 1/block_size（`设计期估算`: 1M int32 发射 6.4µs ≈ 流量 6.2µs）。**更大 block_size（如 4096）可摊薄发射**（1M 发射降至 3.2µs，总下界趋近流量 6.2µs），但 int64 下 4096×8×2=64KB（×1.7≈109KB）仍在预算内——**列为 Stage 4 调优首选杠杆**（分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：`block_size=2048`，int32 元素 4B → 每 tile 2048/4=512 向量槽，尾 lane 浪费 0%；int64/2B/1B 元素同样满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 54 KB（int64，含 1.7x）< 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest 全部 workload 的 N（1048576 / 16,777,216 / 67,108,864 / 268,435,456）均整除 `block_size=2048`（尾块不存在）；测试 shape 1,048,576 / 4,000,000 中 **4,000,000 不整除 2048**（余 1216）→ 存在尾块。

**策略**：Kernel 内动态尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 tile 用 `tail = T.min(block_size, shape - off)` 裁剪 `T.copy` 的搬入/搬出量；v-op 仍作用于整 buffer（尾部未初始化数据不影响仅搬出的 `[0:tail]`）。参考 `examples/elementwise/vec_add_1d.py`。任务网格用 `T.ceildiv(N, block_size)` 保证覆盖最后一个不完整 tile。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子为 1-D 输出网格 `[N]`，仅 N 轴（无 M 轴，M 项取 1）：

```
num_logical_kernels = ceil(N / block_size) = ceildiv(N, 2048)
```

| workload | N | 逻辑核数 `ceildiv(N,2048)` |
|----------|---|---------------------------|
| smoke-1m | 1,048,576 | 512 |
| 16m | 16,777,216 | 8,192 |
| 64m | 67,108,864 | 32,768 |
| 256m | 268,435,456 | 131,072 |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C；本设备 Ascend910_9362 实查 20 核（与 `CONST-aicore-910_9362` 一致），**以实查为准**。

**③ 规模判定与分核方案**：**极大规模**——逻辑核数（512 ~ 131,072）远超物理核数 40，且 N 为 2 的幂、物理核数 40 = 2³×5 含非 2 因子，数学上**无法**通过调整 block_size（2 的幂）使逻辑核数对齐 40 的整数倍（5 ∤ 2ⁿ）。按标准 §1 要素③「极大规模」处置：

- **固定启动内核数 = 物理核数 40**；
- 核内 `T.serial` 网格跨步处理多个逻辑 tile：
  ```python
  for i in T.serial(T.ceildiv(n_num, core_num)):   # 循环上界静态（n_num、core_num 均编译期常量）
      task = i * core_num + cid                    # 网格跨步（grid-stride）
      if task < n_num:                             # 守卫
          ...
  ```
- **循环边界静态** ✓（`n_num`、`core_num` 均编译期常量，`T.ceildiv(n_num, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：以 1M@bs2048 为例，`tasks_per_core = ceildiv(512,40) = 13`；网格跨步下前 `512 - 40×12 = 32` 个核执行 13 次、后 8 个核执行 12 次 → 临界路径 = 13/12.8 ≈ **+1.6%**（相对理想均匀 12.8 波次）。相较「直接 `T.Kernel(n_num=512)` 由运行时串行调度 12.8 波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。大 N 档（16m/64m/256m）逻辑核数更大，残留不均占比进一步下降（<0.05%）。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、`example_elementwise_exp2.py` L28–L43。

> 备选（供 Stage 4 参考，不在本设计实现）：直接 `T.Kernel(n_num)` 由运行时调度（无核内串行）——保留为对称性对照，若实测核内串行反而引入同步开销可回退；本设计按标准要求取核内串行 persistent 形态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(core_num, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, core_num))` | 每核串行处理 `ceildiv(n_num,40)` 个 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化 | `T.vnot` / `T.fill`+`T.vxor` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

### 6.2 循环伪代码

```python
with T.Kernel(core_num, is_npu=True) as (cid, _):
    x_ub = T.alloc_ub((block_size,), k_dtype)
    y_ub = T.alloc_ub((block_size,), k_dtype)
    ones_ub = T.alloc_ub((block_size,), k_dtype)   # 仅 bool
    for i in T.serial(T.ceildiv(n_num, core_num)):  # 静态上界
        task = i * core_num + cid
        if task < n_num:
            off = task * block_size
            tail = T.min(block_size, shape - off)
            T.copy(x[off:off + tail], x_ub[0:tail])
            # bool 路径 / 整数路径（编译期分支）
            T.copy(y_ub[0:tail], y[off:off + tail])
```

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → v-op → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理（`PL-1.6-copy-floor` 已 stale，其「向量 pass 隐藏于 MTE2 窗口」结论仅作背景，不作为决策依据；实际重叠效果交 Stage 4 实测）。若需进一步优化（persistent double buffer），可参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态，设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4 表），仅测试 4,000,000 触发尾块。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [T.fill] → vnot/vxor → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| v-op 之间 / `T.fill` 后 | 自动 | 线性数据依赖，插 V→V barrier |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| tile 迭代之间（`T.serial`） | 自动 | 同 buffer 复用由 auto-multi-buffer 管理跨迭代依赖 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即 Developer
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`torch.bitwise_not`），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch

def golden_bitwise_not(x: torch.Tensor) -> torch.Tensor:
    """BitwiseNot 参考实现，等价于 torch.bitwise_not.

    支持 bool / uint8 / int8 / int16 / int32 / int64；逐元素按位取反，输出同 dtype。
    整数/布尔域位运算精确，无中间精度提升、无容差（exact_compare）。
    """
    return torch.bitwise_not(x)   # bool→逻辑非；uint8→255-x；signed→~x
```

### 8.2 精度标准

| dtype | 判定 | 说明 |
|-------|------|------|
| bool / uint8 / int8 / int16 / int32 / int64 | **exact（逐位相等）** | 整数/布尔位运算无舍入误差；对照源测试 `exact_compare`（`torch.equal`）。**不使用 atol/rtol**——位运算必须 bit-exact。 |

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 int32 smoke | `randint(-2^31, 2^31-1, [1048576], int32)`（NPU） | `torch.equal(y.cpu(), torch.bitwise_not(x.cpu()))` 为 True | 门槛通过 |
| L0-2 int64 smoke | `randint(full int64, [1048576], int64)`（NPU） | 同上 exact | 门槛通过 |
| L0-3 uint8 全范围 | `randint(0,256,[1048576], uint8)`（NPU） | exact（**须确保输出张量 dtype=uint8**，见 §9 风险点 R-2） | 门槛通过 |
| L0-4 bool | `rand([1048576]) > 0.5`（NPU） | exact | 门槛通过 |
| L0-5 非整除尾块 | `randint(...,[4000000], int32)`（NPU） | exact（覆盖尾块 4000000 % 2048 ≠ 0） | 门槛通过 |

- **执行方式**：`compile(func, target="npuir")` 后单次调用，`torch.equal` 逐位比对（golden = `torch.bitwise_not`）。
- **门槛标准**：L0-1..L0-5 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py` 的 `BitwiseNotWorkload`（bool 用 `rand>0.5`、uint8 全范围、signed 小范围）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 bool i1 不能 load/store GM**（`docs/.../T.vnot.md` §2.3） | 直接对 bool dtype 构造 UB/GM buffer 用 v-op 会失败（探针实测：device `vector core timeout`） | bool 分支走 1 字节存储视图（kernel 内 `int8`），wrapper 侧对 bool 张量做 `.view(int8)` / 结果 `.view(bool)`（host 仅视图、不改内容/stride，符合 `ascend-constraints.md` §4） |
| **R-2 uint8 依赖 out_idx 自动分配时输出 dtype 错**（实测） | `@tilelang.jit(out_idx=[1])` 的输出 dtype 映射表 `jit_npu.py:_tvm_dtype_to_torch` **缺 uint8**→ 自动分配为 float32；kernel 计算值本身正确（探针实测前 N 字节 = `torch.bitwise_not`） | uint8 场景由 Op 层/包装器**显式分配 uint8 输出张量**并传入（或修复映射表）；已登记能力缺口 `CG-2026-0014`（见 §10。此风险不影响 manifest benchmarked workload：仅 int32/int64） |
| **R-3 vxor src1 必须为 tensor**（实测） | `T.vxor(x, 1, y)` 编译失败（`hivm.hir.vxor ... cannot be a scalar`） | bool 分支先 `T.fill(ones_ub, T.cast(1,"int8"))` 再 `T.vxor(x_ub, ones_ub, y_ub)` |
| **R-4 T.fill value 须为 PrimExpr**（实测） | `T.fill(buf, 1)` 报 `value must be a tir.PrimExpr` | 用 `T.cast(1, "int8")` |
| **R-5 分核残留不均** | 40 物理核含非 2 因子 5，2^n 逻辑核无法整除对齐 | §5.5 网格跨步核内串行，残留临界路径 +1.6%（1M）/ <0.05%（大 N），已显式记录 |
| **R-6 v-op 整 buffer vs 动态 slice** | 探针中 `T.vnot(x_ub[0:tail], ...)`（动态 slice）报错 | 设计用**整 buffer** v-op（`T.vnot(x_ub, y_ub)`），仅 `T.copy` 用动态 slice（§3.3 伪代码已按此写） |
| **R-7 常数表设备/版本** | `CONST-mte2-degradation` 绑定 910B2C 且同源 `PL-1.6-copy-floor` 已 stale；本设备为 910_9362 | 带宽数字标注「待重验」，仅作量级估算；Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账 |

> **设计期 API 探针**（证据留存，非裁决探针）：`examples/bitwise_not/bitwise_not_fwd_kernel/history_version/` 下 `design_probe_vnot_int.py`（int8/16/32/64 `T.vnot` == `torch.bitwise_not` bit-exact）、`design_probe_bool_storage.py`（bool 存储视图 `fill(1)+vxor` bit-exact；对 bool dtype 直接 v-op 触发 device timeout）、`design_probe_uint8_outidx.py`（uint8 out_idx 输出 dtype 映射为 float32 复现）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大（如 int64 下 >8192） | 编译失败 | 维持 block_size ≤ 4096（int64），§4.5 |
| 尾块越界 | N 非 block_size 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| bool 直接建 buffer | `T.alloc_ub((bs,), "bool")` + v-op | device timeout | 改用 int8 存储视图（R-1） |
| uint8 输出 dtype 错 | 依赖 out_idx 自动分配 | 结果被解释为 float32 | Op 层显式分配 uint8 输出（R-2） |
| vxor 传标量 | `T.vxor(x, T.cast(1,dtype), y)` | 编译失败 | 先 fill 掩码 buffer（R-3） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃 | 通用化时 host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>2.6e8）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **bool 输入** | 存储视图（R-1），字节级 `x^1` |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/bitwise_not/bitwise_not_fwd_kernel/
├── DESIGN.md           # 本设计文档
├── verify_equiv.py     # §1.6.1 等价性机器验证脚本（D-1）
├── RETROSPECTIVE.md    # Stage 1 复盘（自进化钩子）
└── bitwise_not_fwd_kernel.py   # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | 3 采纳项等价性机器验证（全部 EQUIV_PASS） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘 |
| `bitwise_not_fwd_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |
| `test_bitwise_not.py` | ⬜ 待实现 | 分层测试（Stage 3，放 `testing/npuir/` 或仓库对应位置） |

### 10.3 命名规范

- 项目目录名: `bitwise_not`
- 算子目录名: `bitwise_not_fwd_kernel`
- 实现文件: `bitwise_not_fwd_kernel.py`（保留源仓 `@tilelang.jit` 函数名 `bitwise_not_fwd_kernel`）
- 测试文件: `test_bitwise_not.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性验证（verify_equiv.py）
3. ⬜ Golden 函数（`golden_bitwise_not`，验证基准）
4. ⬜ 算子实现（`bitwise_not_fwd_kernel.py`：kernel + wrapper + L0 测试）
5. ⬜ L0 门槛测试（L0-1..L0-5：int32/int64/uint8/bool/非整除尾块）

### 10.5 迁移执行规则遵循确认

| 规则 | 遵循情况 |
|------|----------|
| `@tilelang.jit` 装饰函数命名为 `bitwise_not_fwd_kernel` | ✅ 保留源工厂函数名 |
| Kernel 参数语义保持（`threads`/`npt` 折叠为 `block_size`） | ✅ K11：256 × 8 = 2048 |
| `@T.prim_func` 参数名 `x, y` 及顺序不变 | ✅ `main(x, y, shape)`，新增 `shape` 用于尾块 |
| 覆盖全部三个 dtype 分支 | ✅ bool（存储语义 vxor）/ uint8（vnot）/ signed int（vnot）——含脚手架丢失的 bool/uint8 分支 |
| K1–K4 | ✅ `target="npuir"`、`is_npu=True`、移除 `threads=` |
| K8/K9 | ✅ 移除 autotune / `threads` |
| 能力缺口登记 | ✅ `CG-2026-0014`（uint8 out_idx 输出 dtype 映射缺失）写入 `.agents/evolution/capability-gaps.md` |
