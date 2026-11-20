# bitwise_xor_fwd_kernel 算子设计文档

> 迁移任务（`scenario=migration`，`migration_mode=harness`，GPU 仓为 TileOPs 同构工程）。源算子 = GPU 仓 TileOPs 的 `bitwise_xor_fwd_kernel`（`_make_binary_explicit` 工厂 + `BinaryKernel` 二进制模板的 `op_func(a,b)=a^b` 特化），迁往 TileLang-NPUIR（`target="npuir"`，华为昇腾 NPU），编程模式固定 `Developer`。
>
> 本文 §0 记录源算子三问解读 → 算法调研（Phase R，结论见 §1.6.0，作为 M1 判定输入）→ 硬件耦合性判定 → NPU 算法重设计的全过程；§1–§11 基于**重设计后的 NPU 算法**展开。
>
> 关键能力结论由设计期探针实测（D-3，脚本见 `history_version/design_probe_vxor*.py`，均为最小张量、总墙钟 <10 min）：`T.vxor` 覆盖 int8/16/32/64 同形运算；**不接受任何自动广播**（`[M,N]^[1,N]`/`[M,N]^[M,1]`/`[2,3,4]^[1,3,1]`/rank-1 `(N,)^(1,)` 全部 BiShengHIR verify 失败，同工具链同形状 `T.vor` 全部通过）；**不接受标量操作数**；**uint8 GM 张量不受支持**（输出静默变 float32）。工具链 `tilelang 0.1.2+417dae9908` / CANN 8.5.0。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素按位异或 `y = a ^ b`，等价于 `torch.bitwise_xor(a, b)`。支持 NumPy/PyTorch 广播：`y.shape == broadcast_shapes(a.shape, b.shape)`。逐位运算（bit-level），无算术、无超越函数。
- **I/O 契约**：两输入一输出；两输入同 dtype；输出 dtype = 输入 dtype（`output_dtype` 缺省回退 `dtype`，manifest `same_as(input)`）；输出 shape = 两输入 broadcast 结果。算子层把两输入 `contiguous()` 后展平为 1-D `(a_numel,)` / `(b_numel,)`，输出展平为 `(N_total,)`，kernel 只看一维张量，广播 lowering（`coalesce_broadcast_dims`）在 kernel 构造期完成（`examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/bitwise_xor.py` L194）。
- **规约语义**：**无规约、无跨元素依赖**——每个输出元素只依赖同位置的 `a`/`b` 元素（经广播索引），`y[i]` 与 `y[j]` 完全独立。因此**并行组织/分核/累加分组的变化不影响任何数值结果**（无浮点累加顺序问题；整数位运算 bit-exact）。
- **dtype 语义**：整数/布尔族 `bool | uint8 | int8 | int16 | int32 | int64`（`_BITWISE_DTYPES`，`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L140–L147）。整数路径直接在原生 dtype 域做位运算（无中间提升）。**bool 特殊路径**：GPU 因 CUDA codegen 无法 lower `boolx<N>` 向量 load/store，Op 层把 `torch.bool` 张量 `view(torch.uint8)` 后走 `*BoolStorageFwdKernel`（`ops/elementwise/bitwise_xor.py` L98–L133），即 **bool 以 1 字节无符号存储承载**。
- **边界语义**：逐位映射，无 nan/inf 传播问题（整数/布尔无 nan/inf）；空维度/0-size 输入由 Op 层 `prod()==1` 兜底或上层拦截（`torch.broadcast_shapes` 把 0-dim 归一为 `(1,)`）；非整除 shape 由广播规则处理（`out_shape` 可为任意广播形状）。
- **host 侧语义**：`contiguous().view(-1)` 展平、`coalesce_broadcast_dims` 索引表示选择、`reshape(out_shape)` 回形状——属 Op 层契约（实现手段，非语义），由 NPU 侧 wrapper（`examples/TileOPs/tileops/ops/elementwise/bitwise_xor.py` 的 `BitwiseXorFwdOp` + `examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/bitwise_xor.py` 的 `BitwiseXorFwdKernel`，已含 `_as_kernel_input`/`_restore_public_dtype` int8 桥接）承接，不属于本 kernel 的重设计范围。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.bitwise_xor` 广播语义直算），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a`（input） | `(a_numel,)` | bool / uint8 / int8 / int16 / int32 / int64 | Op 层已展平的 1-D 连续张量 |
| `b`（other） | `(b_numel,)` | same_as(a) | 同上 |
| `y`（output） | `(N_total,)` | same_as(a) | `out_idx=[2]`；**输出为一维展平视图**（源码 `main` 的 `y: T.Tensor((N_total,), out_dtype)`，Op 层再 `reshape(out_shape)`），无转置/布局变换。广播时 `a_numel` / `b_numel` / `N_total` 三者可不同 |

**输出 shape 依据**：源文件 `_bitwise_xor_fwd_kernels.py` L55 / L78（`y: T.Tensor((N_total,), out_dtype)`）；NPU wrapper `BitwiseXorFwdKernel.forward` 返回后由 `BitwiseXorFwdOp._eager_forward` `reshape(self.out_shape)` 还原广播形状（`ops/elementwise/bitwise_xor.py` L120–L133）；manifest `shape_rules: output.shape == broadcast_shapes(input.shape, other.shape)`。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/_bitwise_xor_fwd_kernels.py`（Stage 0 提取件，Pattern A：`_make_binary_explicit` 内联 `op_func=a^b`）；GPU 上下文：`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L556–L609（`_make_binary_explicit`）、L890–L1128（`BinaryKernel`）、L1905–L1920（`BitwiseXorFwdKernel` / `BitwiseXorBoolStorageFwdKernel`）。

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host） | `coalesce_broadcast_dims(a_shape, b_shape)` → `out_shape, coalesced_shape, a_strides, b_strides`（stride=0 表示广播维）；`_is_contiguous_same_shape` 选快路径 | a_shape/b_shape | 索引表示（编译期常量） | 广播 lowering（实现手段） |
| 2 | `block_size = threads * num_per_thread`；grid = `T.ceildiv(N_total, block_size)` | threads/npt | CUDA 任务划分 | 任务划分 |
| 3a（同形连续快路径） | `for i,j in T.Parallel(threads,npt): idx=(bx*threads+i)*npt+j; y[idx] = a[idx] ^ b[idx]` | GM a/b | GM y | `y = a ^ b`（同形） |
| 3b（广播路径） | `flat_idx=(bx*threads+i)*npt+j; (a_off,b_off)=_compute_broadcast_offsets(flat_idx, ndim, divisors, a_strides, b_strides); y[flat_idx]=a[a_off]^b[b_off]` | GM a/b | GM y | `y = a ^ b`（广播索引） |
| 4（host） | Op 层 `reshape(out_shape)` | `(N_total,)` 输出 | broadcast shape | 后处理（Op 层契约） |

`_compute_broadcast_offsets`（源 L160–L176）是**编译期 unrolled 的 mixed-radix divmod 链**：沿 coalesced 各维依次 `coord = remaining // divisors[d]; remaining %= divisors[d]; a_off += coord*a_strides[d]`，最后 `a_off += remaining*a_strides[-1]`。**这是逐元素的标量整数 divmod**（每输出元素 O(ndim) 次除法/取模）。

**数据流与内存访问模式**（GPU 视角）：

```
GM[a] ──(逐元素 strided/标量读)──> register ┐
GM[b] ──(逐元素 strided/标量读)──> register ┼─ 标量算术 a ^ b ──> register y ──> GM[y]
（同形快路径：T.copy 连续段 → register；广播路径：逐元素标量索引 gather 读 GM）
```

**循环与并行结构**：CUDA grid（`ceildiv(N_total, block_size)` 个 block）× block 内 `T.Parallel(threads, num_per_thread)` 二维线程并行；`threads=256`，`num_per_thread=8`（整数 dtype，`_strategy_npt`，L181–L195），`block_size=2048`；尾块由 TileLang `LegalizeSafeMemoryAccess` 隐式保护。

**host 侧逻辑**：`@tilelang.jit(out_idx=[2])` + `functools.lru_cache(maxsize=32)` 按 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, …)` 特化编译；`out_dtype = output_dtype or dtype`；bool 由 Op 层 `view(uint8)` 后走 `*BoolStorageFwdKernel`。

### 0.4 优化手段解读（为什么快）

> 对照 migration-analysis.md §4.1 清单逐项核对；源码显式优化手段如下（无 SMEM tiling / warp shuffle / 异步流水 / swizzle，对照清单无遗漏）。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel`：`T.Parallel(threads, num_per_thread)` | 满并行 + 每线程多元素 ILP | 每线程连续处理 `npt` 个元素（`idx=(bx*threads+i)*npt+j`） | CUDA block/thread 线程模型 | 硬件强相关 |
| 2 | 连续段 coalesced 读写 | 访存带宽 | 同形路径按 `block_size` 连续段访问，尾轴连续 | CUDA memory coalescing | 硬件强相关 |
| 3 | `coalesce_broadcast_dims`（维度合并） | 减少 divmod 次数 | 合并同广播行为的相邻维，最小化 effective dims | 无（纯索引表示） | 可移植 |
| 4 | 编译期 unrolled divmod 索引（`_compute_broadcast_offsets`） | 免运行时 shape 查询、广播通用化 | 常量除法在 build 期展开；但**每元素仍 1 次 divmod 链** | 无（通用），但代价随 ndim 线性增长 | 可移植（代价高，见 §1.6.0 R3） |
| 5 | 同形连续快路径（`_is_contiguous_same_shape`）跳过广播机制 | 消广播开销 | 分支在 build 期选择 kernel body | 无（编译期分派） | 可移植 |
| 6 | `register_copy` 策略（fragment load→compute→store，binary 仅同形） | 数据驻留寄存器 + 128bit 向量访存 | `T.alloc_fragment` + `T.copy` | GPU 寄存器堆 + fragment 机制 | 硬件强相关 |
| 7 | 工厂闭包 + `lru_cache(maxsize=32)` | 摊薄 JIT 编译开销 | 每参数组合只编译一次 | 无（host 层） | 可移植 |
| 8 | bool 走 uint8 存储（`view(torch.uint8)`） | 绕过 CUDA 无法 lower `boolx<N>` | 1 字节无符号存储承载 bool 位模式 | CUDA codegen 限制（非硬件） | 需 NPU 侧替换（uint8 在 npuir 不可用，见 §0.5） |

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `y = a ^ b` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；整数位运算无累加顺序问题 |
| I/O 契约（1-D 进 1-D 出、same_as、广播 out_shape） | 语义 | 无 | — | **保留** | 输出 shape 一维由 §0.2 源码解读确认；广播形状还原在 Op 层 |
| dtype 契约（bool/uint8/int8/16/32/64） | 语义 | 无 | — | **保留（bool 存储表示需重设计）** | `_BITWISE_DTYPES` 沿用；bool 经 1 字节存储承载的**表示**因 npuir 不支持 uint8 改为 int8（→ §0.6 R3），dtype 契约本身不变 |
| host 展平/coalesce/回形状 | 语义（Op 契约） | 无 | — | **保留** | NPU wrapper 已承接（含 `_as_kernel_input`/`_restore_public_dtype`）；host 侧仅元数据/视图操作，符合 ascend-constraints.md §4 |
| `coalesce_broadcast_dims` 维度合并 | 算法 | 无 | — | **保留** | 索引表示选择，与硬件无关；wrapper 已承接（`bitwise_xor.py` L194） |
| 同形连续快路径（build 期分支） | 算法 | 无 | — | **等价替换** | 统一进 §0.6 R1 的行/内层方案（k=1,R=1 退化为同形路径），意图（免广播机制）承接 |
| 一维 grid `ceildiv(N_total, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1；`examples/elementwise/vec_add_1d.py`）；核数按 §5.5 分核适配（→ R2） |
| `T.Parallel(threads, npt)` 两级线程并行 | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 R2（一维 Kernel + Vector 向量化；npt 的多元素意图由 128bit 向量 lane 承接） |
| `threads=` / `num_per_thread` kwarg | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已把 `threads*npt` 折叠为单一 `block_size`，无性能意图丢失（并行组织由 R2 承接） |
| 连续段 coalesced 搬运 | 优化 | CUDA coalescing | 有（GM↔UB 连续段 T.copy） | **等价替换** | `T.copy` 连续段 GM↔UB（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款） |
| `register_copy`（fragment 驻留） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment`→`T.alloc_ub`（migration-analysis.md §5.3「register 累加→UB 中间缓冲」）；v-prefix 要求操作数在 UB |
| **逐元素 divmod 广播索引** | 优化 | 无（通用） | 无等价（NPU 无逐元素整数 divmod 快路径；且源形态为 GM 标量 gather） | **重新设计** | → §0.6 R1（外层行标量 base + innermost 维向量化）——**本算子最关键的迁移决策** |
| **`T.Parallel` 逐线程标量 `^` 直译** | 优化 | CUDA 线程模型 | ❌ 无广播/标量能力（NU 实测） | **重新设计** | → §0.6 R1+R2：**`T.vxor` 仅支持同形**（CG-2026-0020，探针 B/C/Q1），标量操作数不支持（探针 E/Q2）——**本算子区别于 bitwise_and 的关键点**：任何广播维必须由 `T.copy`/`T.vbrc` 提前物化为同形 operand，不能依赖向量指令自动广播 |
| 工厂闭包 + lru_cache 特化 | 优化（host） | 无 | — | **保留** | wrapper `forward` 已按 `block_size` 惰性构建 + 缓存（`bitwise_xor.py` L238–L241） |
| bool 走 uint8 存储 | 优化 | CUDA codegen 限制 | uint8 在 npuir **不可用**（CG-2026-0018） | **重新设计** | → §0.6 R3（改 int8 承载同一位模式） |
| GPU grid 自由超发 | 优化 | CUDA 调度器 | 无（超发被串行调度 + 核启动开销） | **重新设计** | → §0.6 R2（persistent 分核，core-split-strategy.md §1） |

**判定统计**：保留 7 项、等价替换 3 项、重新设计 4 项（R1 广播索引 / R2 并行与分核 / R3 bool 存储 dtype / R4 同形约束下的广播物化——R4 与 R1 同源，见 §0.6）、舍弃 1 项（`threads=`/`npt` kwarg，理由见表）。

### 0.6 NPU 算法重设计

**重设计项 R1: 广播索引路径（逐元素 divmod/gather → 外层行标量 base + innermost 维向量化 + 同形 operand 物化）**

- **源方案**：广播路径对**每个输出元素**执行一次编译期 unrolled mixed-radix divmod 链（`_compute_broadcast_offsets`），得到 `a_off`/`b_off` 后以标量 gather 读 GM 的 `a[a_off]`、`b[b_off]`；同形路径才用连续段。性能意图：以「编译期常量除法展开」把广播通用化的代价压到单元素 O(ndim) 整数 divmod（GPU 多核并发消化）。
- **NPU 新算法**：把 coalesced 输出视为 **`R = prod(coalesced_shape[:-1])` 个「行」× 行长 `L = coalesced_shape[-1]`** 的二级结构（`k=1` 时 `R=1, L=N_total`，同一模板覆盖同形与广播两路）。对每个「行」（即一个 task）：
  1. **行 base 标量**：`outer_a_base = Σ_{i<k-1} coord_i(row)·a_strides[i]`（`coord_i` 由 row 对 `coalesced_shape[:-1]` 的 mixed-radix 分解得，编译期 unrolled，**仅每 task 一次**，非每元素）；`b` 同理。末维：`a_base = outer_a_base + (inner_off if a_strides[-1]==1 else 0)`（末维 stride ∈ {0,1}）。
  2. **行内 operand 物化（R4：因 `T.vxor` 仅同形）**：末维 `L` 按 `block_size` 分块（`chunks = ceildiv(L, block_size)`；manifest 广播行 `L≤4096`，推荐 `block_size=4096` 使 `chunks=1`）。每块**必须**把两个 operand 都物化为 `(block_size,)` 同形 UB：
     - `a_strides[-1]==1`（a 末维连续）→ 连续段 `T.copy(a[a_base:a_base+tail] → a_ub[0:tail])`；否则（a 末维 stride 0，a 在行内为标量）→ `T.copy(a[a_base:a_base+1] → a1_ub)` + `T.vbrc(a1_ub → a_ub)`；
     - `b` 同理（`T.vbrc` 因 `T.vxor` 不接受标量/size-1 操作数，CG-2026-0019/0020；探针 E/Q2 实测标量失败、探针 D/直连探针实测 vbrc→vxor int8/16/64 通过）；
     - 两 operand 均物化为 `(block_size,)` 后 `T.vxor(a_ub, b_ub, y_ub)`（**同形**，满足 CG-2026-0020 约束）→ 连续段 `T.copy(y_ub[0:tail] → y[flat0:flat0+tail])`。
   - **末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`**（证明：输出末维 size>1 时至少一个操作数在该维 size>1，其连续 stride=1），故 `(0,0)` 组合不可能（`L==1` 全标量退化为两路 `T.vbrc`，`block_size>1` 时合法）。
- **语义保持论证**：**数学恒等**——新映射对每个 flat 索引给出的 `(a_off, b_off)` 与源 divmod 链**逐点相同**（行分解 `row=flat//L`、`inner=flat%L` 恰为末维与其余维的 mixed-radix 分解，`a_off = Σ_{i<k-1}coord_i·a_strides[i] + coord_{k-1}·a_strides[-1]`），**已由 `verify_equiv.py` E2 机器验证**（11 组形状含全部 manifest + 测试广播模式，`a_off`/`b_off` 逐点 mismatch=0、越界=0，EQUIV_PASS）。位运算本身 bit-exact，无 dtype/边界差异；空维度/0-dim 由 Op 层归一，kernel 不触达。
- **意图承接**：源优化手段 3（维度合并）原样保留（wrapper 沿用 `coalesce_broadcast_dims`）；手段 4（编译期 unrolled divmod）的「免运行时 shape 查询」意图由编译期 unrolled **行** divmod 承接，代价从「每元素 O(ndim)」降为「每 task O(ndim)」（约降 `L` 倍）；手段 5（同形快路径）由 `k=1⇒R=1` 的统一退化承接。

**重设计项 R2: 并行结构与分核（CUDA block×thread 两级并行 + grid 超发 → 一维 Kernel + Vector 向量化 + persistent 分核）**

- **源方案**：`T.Kernel(ceildiv(N_total, block_size), threads=threads)` 的 CUDA grid×block 两级并行，block 内 `T.Parallel(threads, npt)` 二维线程映射，每线程标量处理 npt 个元素；grid 大小任意（最大 workload 下数千 block），超发由 CUDA 硬件调度器吞吐。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；核内 `for i in T.serial(num_local_tasks)`（**静态边界**）以 grid-stride 映射 task `= i*num_kernels + cid`，`if task < num_logical` 屏蔽越界。元素级并行交给 Vector 单元：逐元素 `a ^ b` 由 `T.vxor` 一条向量指令覆盖整块（128bit 向量 lane：int8 ×16 / int16 ×8 / int32 ×4 / int64 ×2 元素/指令）。核数按 §5.5 分核（`num_kernels = min(num_logical, 物理 Vector 核数)`）。
- **语义保持论证**：task↔行/c块一一对应、各 task 完全独立（无跨 task 依赖、无竞写），映射顺序/分核数变化不改变任何输出值；向量化只是把逐元素标量 `^` 换成等价的逐元素向量 `^`（逐位运算，lane 间独立）。无 nan/inf/累加顺序问题。
- **意图承接**：手段 1（`T.Parallel(threads,npt)` 的满并行 + ILP）→ Vector 128bit lane + `block_size` tile 粒度；手段 2（coalesced）→ `T.copy` 连续段；手段 6（fragment 驻留）→ `T.alloc_ub` 原地缓冲；`threads=`/npt 折叠为 `block_size`（wrapper K11），无性能意图丢失。

**重设计项 R3: bool 存储 dtype（uint8 → int8）**

- **源方案**：`torch.bool` 张量由 Op 层 `view(torch.uint8)`，以 1 字节无符号存储跑 `*BoolStorageFwdKernel`（因 CUDA 无法 lower `boolx<N>`）。
- **NPU 新算法**：改用 **int8** 承载同一位模式（`bool → view(torch.int8)`），kernel 以 `int8` 声明张量与 UB。依据：探针 `design_probe_vxor.py` Probe A-uint8/G-uint8 实测 npuir 对 uint8 GM 张量输出**被静默映射为 float32、数值全错**（与 bitwise_and 的 `design_probe_uint8_isolate.py` 一致）→ 登记 CG-2026-0018；`T.vxor` 的 `int8` dtype 探针 A-int8 实测可用。wrapper 已实现 `_kernel_dtype_str`/`_as_kernel_input`/`_restore_public_dtype` 桥接（`bitwise_xor.py` L95–L139）。
- **语义保持论证**：**位级恒等**——按位异或逐比特运算，同一字节的 int8 与 uint8 位模式相同，`a^b` 结果字节相同；`verify_equiv.py` E1 以 1M 随机字节 + 全 256 角点字节对照 int8/uint8 的 XOR/OR/NOT **violations=0，EQUIV_PASS**。bool 语义（非零即真）不受影响。**注**：这要求 NPU 侧 bool 存储路径的 view 目标由 uint8 改为 int8（wrapper 已承接；Stage 5 集成项，见 §9 风险 R3）。
- **意图承接**：手段 8 的「1 字节存储承载 bool」意图完全保留，仅替换无符号→有符号表示（位模式不变）。

**重设计项 R4: 同形约束下的广播物化（`T.vxor` 无自动广播 → 显式 `T.vbrc`/连续段同形化）**

- **源方案**：GPU 无此约束——`op_func(a[a_off], b[b_off])` 在标量域逐元素读取，广播由索引 strides 隐式表达，向量指令 `boolx<N>`/整型不存在操作数同形要求。
- **NPU 新算法**：探针 Q1 实测 `T.vxor` **在同工具链同形状下与 `T.vor` 能力不等价**——`xor-r2-col`/`xor-r2-row`/`x 3d` 全部 BiShengHIR `'hivm.hir.vxor' op operands' shape are inconsistent`，而 `T.vor` 的 `[4,32]|[1,32]`、`[4,32]|[4,1]` 全部通过（CG-2026-0020：hivm.hir.vxor 缺 broadcast attribute）。因此**任何广播维必须在向量指令之前显式物化**：末维 stride=0 的 operand 经 `T.vbrc` 扩到 `(block_size,)`，末维 stride=1 的 operand 经连续段 `T.copy` 到 `(block_size,)`；`T.vxor` 只在两个 `(block_size,)` 同形 buffer 上调用（§0.6 R1 第 2 步）。
- **语义保持论证**：vbrc 把标量/单元素按广播语义复制成整向量，与 PyTorch 广播逐点一致；物化后再做同形 `T.vxor` 的每 lane 结果 = `a[off_a] ^ b[off_b]`，与源逐元素标量式逐点相同。`verify_equiv.py` E2 覆盖索引侧，E1 覆盖位级侧；vbrc 的正确性由探针 D（int32 (1,)→(N,)）与设计期直连探针（int8/int16/int64 (1,)→(N,) 后 vxor 全部 bit-correct）实测。
- **意图承接**：源「索引 strides 隐式广播」意图由「编译期行 base 分解 + 显式 vbrc/连续段物化」承接，广播语义不变，代价是每 task 至多 1~2 条 vbrc（Vector 发射 ~0.5µs，见 §1.6.0 发射项）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 `T.Parallel` 满并行 + ILP | Vector 向量化（`T.vxor` 128bit lane）+ 单一 `block_size` tile（R2） |
| 2 连续段 coalesced | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 `coalesce_broadcast_dims` 维度合并 | 原样保留（wrapper 承接，驱动 §0.6 R1 的 outer/inner 分解） |
| 4 编译期 unrolled divmod | R1 行级 unrolled divmod（每 task O(ndim)，较源每元素降 `L` 倍）+ innermost 维向量化 |
| 5 同形连续快路径 | R1 的 `k=1 ⇒ R=1` 统一退化 |
| 6 `register_copy` fragment 驻留 | `T.alloc_ub` 原地缓冲链 |
| 7 工厂 + lru_cache 特化 | wrapper `forward` 惰性构建 + 缓存（原样保留） |
| 8 bool 走 1 字节存储 | R3 int8 承载（位级恒等，uint8 不可用） |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/_bitwise_xor_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 原始上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_binary_explicit` @ L556、`BinaryKernel` @ L890、`BitwiseXorFwdKernel` @ L1905）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/bitwise.py`（`BitwiseXorFwdOp` @ L71）。
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/bitwise_xor.py`（`BitwiseXorFwdKernel`：工厂签名保留 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel)`，返回 callable(`block_size`)；`BitwiseXorBoolStorageFwdKernel` 窄化 dtype）；Op 层 `examples/TileOPs/tileops/ops/elementwise/bitwise_xor.py`。
- **规格来源**：manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `BitwiseXorFwdOp`（4 workloads + roofline `flops=N / bytes=(A+B+N)·elem_bytes`）；测试 `examples/TileOPs/tests/ops/test_bitwise_xor.py`（`exact_compare`，含 3 组广播 + bool 快路径 + float 拒绝）；基准 `examples/TileOPs/benchmarks/ops/bench_bitwise_xor.py`。
- golden 以 §0.1 语义为依据（`torch.bitwise_xor` 广播直算），见 §8.1。


## 1. 概述

### 1.1 算子名称

`bitwise_xor_fwd_kernel`（Op 层名 `BitwiseXorFwdOp` / 算子族 `bitwise_xor`）

### 1.2 功能描述

逐元素按位异或前向核：对两个已展平为一维的同 dtype 张量计算 `y = a ^ b`，支持 NumPy/PyTorch 广播（输出 shape = `broadcast_shapes(a_shape, b_shape)`）。纯逐元素、无规约、无跨元素依赖；整数/布尔族 dtype。

### 1.3 数学公式

$$
y[i] = a[\mathrm{off}_a(i)] \ \hat{}\ b[\mathrm{off}_b(i)], \qquad i = 0, 1, \ldots, N_{total}-1
$$

其中 `off_a(i)`/`off_b(i)` 为广播映射（源算法用逐元素 divmod 链，见 §0.3；NPU 用行/内层分解，见 §0.6 R1）；`^` 为逐位异或。非广播（同形）时 `off_a(i)=off_b(i)=i`。

参考 API：`torch.bitwise_xor(a, b)`。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的结构差异逐条注明来源）**

单遍逐元素行向量算法（§1.6.0 调研选定基线公式，无更优算法族替代）：

1. **任务分解（编译期）**：把 coalesced 输出视为 `R = prod(coalesced_shape[:-1])` 行 × 行长 `L = coalesced_shape[-1]`；`num_logical = R × ceildiv(L, block_size)`。`k=1`（同形 / 全标量广播）退化为 `R=1, L=N_total`。
2. **核间**：`T.Kernel(num_kernels, is_npu=True)`（persistent，`num_kernels = min(num_logical, 物理 Vector 核数)`，§5.5）。
3. **核内（每 task = 一行的一块）**：标量计算行 base（编译期 unrolled divmod，仅每 task 一次）→ 按末维 stride 选择「连续段 `T.copy`」或「单元素 `T.copy` + `T.vbrc`」**把两个 operand 都物化为 `(block_size,)` 同形 buffer** → `T.vxor(a_ub, b_ub, y_ub)` → 连续段 `T.copy` 写回（尾块 `tail` 截断）。
4. **存储**：bool 张量以 int8 承载（§0.6 R3）。

与源算法的**结构差异**：① 逐元素 divmod + 标量 gather（广播路径）→ **行级标量 base + innermost 维向量化 + 显式同形物化**（§0.6 R1/R4，根因是 NPU 逐元素整数 divmod/gather 无快路径，且 **`T.vxor` 连向量级自动广播都不支持**——必须显式 `T.vbrc`/连续段把 operand 变同形）；② CUDA grid×thread 两级并行 + `T.Parallel(threads,npt)` → 一维 Kernel + v-prefix 向量 + persistent 分核（§0.6 R2）；③ bool 存储 uint8 → int8（§0.6 R3，uint8 在 npuir 不可用）。**同形路径**与源骨架（连续段载入 → 逐元素位运算 → 连续段写回、片上驻留）一致，进化为 `k=1,R=1` 的统一形态。

**与同族 bitwise_and 的算法差异（迁移视角）**：bitwise_and 的 `T.vand` 支持 rank≥2 自动广播，故其 §1.6.3 有「rank-2 auto-broadcast」候选与对应备选；`T.vxor` **无任何自动广播**（CG-2026-0020），故本算子的广播路径**只有**「显式物化 + 同形 vxor」一条主轴，候选矩阵相应收窄（见 §1.6.3）。

### 1.5 数据流图

```
同形 / 标量广播（k=1, R=1；L 按 block_size 分块）：
  GM[a] ──T.copy(连续段, 尾块截断)──> UB[a_ub] ─┐
  GM[b] ──T.copy(连续段, 尾块截断)──> UB[b_ub] ─┼─ T.vxor(同形) ──> UB[y_ub] ──T.copy──> GM[y]

更一般广播（k>1；每 task = 一行的一块）：
  (row,chunk) 标量分解 ──> a_base/b_base（编译期 unrolled divmod，仅每 task 一次）
  a 末维 stride=1: GM[a] ──T.copy(a[a_base:a_base+tail])──> UB[a_ub]
  a 末维 stride=0: GM[a] ──T.copy(a[a_base:a_base+1])──> UB[a1_ub] ──T.vbrc──> UB[a_ub]
  b 同上 ────────────────────────────────────────────────────────────> UB[b_ub]
  UB[a_ub] ^ UB[b_ub] ──T.vxor(同形, CG-2026-0020)──> UB[y_ub] ──T.copy──> GM[y[row*L+inner_off …]]
全路径无 L1/L0 参与（纯 Vector）；无中间 GM 往返（每 task 一次进出）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。bitwise_xor 为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用**轻量调研**深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法 / 输入公式） | `y = a ^ b`（逐位异或，1 op/元素）+ 广播索引 | — | — | ✅（基线） |
| 1 | 算术恒等分解 | `a ^ b = a + b − 2·(a & b)` | 数学恒等（位运算恒等式，整数域成立） | **负收益**：1 条位运算 → 2 加 + 1 与 + 1 左移/乘（4 op/元素），引入中间量与溢出/符号域风险 | ❌ op 数 ×4 |
| 2 | 布尔逻辑分解（按位展开） | `a ^ b = (a \| b) & ~(a & b)` 或 `(~a & b) \| (a & ~b)` | 数学恒等 | **负收益**：3–4 条位运算/元素，无收益 | ❌ op 数 ×3–4 |
| 3 | 逐位重排 / 位平面转置 | 位平面分离后按平面异或 | 数学恒等 | **负收益**：引入重排（repack）代价，位数× 访存 | ❌ 需 repack，访存↑ |

**R2 在线算法**：**无在线变体需求（天然单遍）**。结构依据：`y_i` 仅依赖 `(a_{off_a(i)}, b_{off_b(i)})`，**无 running 统计量**（和/最大值/计数）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射无跨元素依赖，单遍即天然在线」）；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径（扫描遍数=1、中间缓冲 O(block_size)、UB 驻留）在候选间无差异。

**R3 复杂度对比**（带宽受限算子——算术强度 1 op/N ÷ (A+B+N)·E 字节，整数 dtype 下 < 0.1 op/B，**访存为主导项**；口径：A/B/N 为展平元素数，E=elem_bytes，`A=B=N` 时同形）：

| 算法候选 | FLOPs（口径：逐位 op 数） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|--------------------------|---------------|---------|-------------|---------------------|
| **基线（源算法）** | N（1 位运算/元素）+ **N·O(ndim) 整数 divmod**（逐元素广播索引，非位运算但同等计） | (A+B+N)·E **但广播路径为逐元素标量 gather（非连续段，访存效率低）** | 1（每元素每操作数读一次写一次） | O(block_size) tile（寄存器/UB） | N/block_size 个独立 task，零跨核代价；但逐元素 divmod 为标量热点 |
| **NPU 重设计（§0.6 R1+R4）** | N（1 位运算/元素，向量化）+ **R·O(ndim) divmod**（行级，较基线降 ~L 倍）+ 每行 ≤1 条 vbrc 物化 | (A+B+N)·E（连续段；`s=0` 行内标量广播仅读 1 元素/行 + vbrc） | 1 | O(block_size)（UB：3~5 buffer，§4.5） | num_logical 个独立 task，零跨核同步；persistent 摊薄启动 |
| 候选 1（算术恒等） | 4N + R·O(ndim) | (A+B+N)·E + 中间量驻留 | 1 | O(block_size) | 同上 |
| 候选 3（位平面重排） | N + repack | (A+B+N)·E + 位平面重排流量 | 2（重排 + 计算） | O(block_size·bits) | repack 代价 |

基线（位运算本体）已是 1 op/元素的下界；NPU 重设计在**位运算本体**上无更优公式，其收益全在**广播索引路径**（把逐元素 O(ndim) divmod/gather 降为行级 O(ndim) + innermost 维连续向量化 + 显式同形物化）。行级重读口径：`sb=1 且 b 在全部外层维广播`（如 row-broadcast `[4096,4096]^[1,4096]`）时每 task 重读一行 b，未优化口径下 b 读流量 ≈ N·E（b 仅 16KB，预期 L2 驻留，HBM 侧不放大）；该不确定项按 §1.6.3 备选 + 实验裁决处理。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 可融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源逐元素 divmod + gather） | ❌ 逐元素整数 divmod + 标量 GM gather → 落错单元（标量密集，非 Vector） | OK | ✗ 标量 gather 无向量 lane | 尾块依赖动态 shape | 单阶段 | ❌ 淘汰：NPU 无逐元素 divmod 快路径，标量 gather 严重失分 |
| **NPU 重设计（行 base + innermost 向量化 + 显式同形物化）** | ✅ 逐位异或 → Vector `T.vxor`（int8/16/32/64 同形实测 √，探针 A；标量广播由 `T.vbrc` 物化，探针 D/直连实测 √） | ✅ 3–5 buffer × block_size ≤ 192KB（§4.5，含膨胀系数后仍达标） | ✅ block_size×E = 32B 整数倍；向量宽度整除（int8×16/16×8/32×4/64×2） | ✅ N/L/block_size 全编译期特化；tail 仅 1 次/task | 单阶段无需流水；纯 Vector 无 CV 融合 | ✅ **选定** |
| 候选 1（算术恒等） | ✅ 仍落 Vector（但 op 数 ×4） | OK | OK | OK | OK | ❌ R3 无访存收益、FLOPs ×4 |
| 候选 3（位平面重排） | 需 repack（无 `T.vxor` 位平面原语） | UB 预算 ×bits | 位平面重排非对齐友好 | 依赖重排 | 两遍 | ❌ 需 repack + 额外扫描 |

**调研结论**：选定 **基线位运算本体（`y = a ^ b`，1 op/元素）的 NPU 行/内层重设计算法**（§0.6 R1：外层行标量 base + innermost 维 `T.vxor` 向量化 + `T.vbrc`/连续段显式同形物化）。关键依据：R3 表中位运算本体已是 op 数下界（1 op/元素），候选 1/2/3 均无访存收益且 op 数 ×3–4；R4 中仅 NPU 重设计对 Vector/UB 容量/对齐/静态边界全部亲和（基线逐元素 divmod/gather 在 NPU 落错单元被淘汰）。与基线的**结构差异**：广播索引从「逐元素 O(ndim) divmod + 标量 gather」变为「行级 O(ndim) divmod + innermost 维连续向量化 + 显式 `T.vbrc` 同形物化（因 `T.vxor` 无自动广播，CG-2026-0020）」。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表命中行 `ALG-elementwise`（R1：公共子表达式消除——`a`/`b` 各用一次，无公共子式；除法乘倒数——无除法；cast 链合并——无 cast；R2 逐元素天然单遍——与本算子一致）；② 恒等变形自查——位运算的本体最短形态为 1 条 `xor`，De Morgan / 算术恒等分解均 op 数更多（R1 候选 1/2）；③ 本仓 `examples/elementwise/`（vec_add_1d/2d、exp2/log2、auto_brc）与 `examples/TileOPs/.../elementwise/{mish,lerp_tensor}` 同类实现——均直排 v-prefix 单遍结构，无位运算公式变形先例；④ pattern-library `elementwise.md`/`cases.md`——elementwise 条目均为访存/布局调度类（copy-floor、loads-first、MTE3 strided、scatter-col），无位运算公式级条目；⑤ 源算子结构分析（§0.3，广播路径逐元素 divmod 是唯一可优化点，已在 R1 承接）；⑥ 位运算位平面算法为外部已知思路，检索确认无低代价硬件原语（`T.vxor` 仅逐位，无位平面 API；`docs/Tilelang.language/逻辑操作/` 目录实查仅 vand/vor/vxor/vnot/vshl/vshr）；⑦ `kb_search "vxor broadcast"` 命中 CG-2026-0020（vxor 缺 broadcast attribute）与 TRAP-vbrc-same-shape-empty-broadcast，确认「显式物化」是唯一路径。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

**设计期 roofline 估算（D-2）**（估算下界 = max(流量项, 发射项, 容量项)；常数引用 `pattern-library/constants.md`；本机实测 `NPUUtils.get().get_aicore_num()=20` ⇒ 40 Vector 核，与 `CONST-aicore-910B2C` 的 910B2C=24/48 不同源，流量项按 40/48 比例折算并标注）：

| 代表 workload | 流量项 | 发射项 | 容量项 | 估算下界 |
|---------------|--------|--------|--------|----------|
| smoke-1k `[1024]` int32（N=1K，block_size=4096） | (N+N+N)·4B = 12KB ÷ 1.05 TB/s ≈ **0.01µs** | 逻辑核 1 / 40 = 1 × vxor × 0.5µs ≈ **0.5µs** | 豁免（轻量调研；tile 级缓冲无跨引擎往返） | **≈0.5µs** |
| hidden-state-prefill `[2048,4096]` int32（N=8.39M，block_size=4096） | (N+N+N)·4B = 100.7MB ÷ 1.05 TB/s ≈ **95.9µs** | 逻辑核 2048 / 40 ≈ 52 核内 vxor × 0.5µs ≈ **25.8µs** | 豁免 | **≈96µs** |
| cnn-feat-broadcast `[16,256,56,56]^[256,1,1]` int32（block_size=4096，chunks=1） | (a 51.4MB + out 51.4MB + b 行内标量 256·4B) ÷ 1.05 TB/s ≈ **97.9µs** | 4096 task / 40 ≈ 103 × (vbrc 1 + vxor 1) × 0.5µs ≈ **103µs** | 豁免 | **≈103µs** |
| row-broadcast `[4096,4096]^[1,4096]` int32（block_size=4096，chunks=1） | (a 67.1MB + out 67.1MB + b 行 16KB) ÷ 1.05 TB/s ≈ **127.8µs**（b 行重读未计）；含 b 行重读上限口径 ≈ (67.1+64+67.1)MB ÷ 1.05 ≈ **188µs** | 4096 task / 40 ≈ 103 × vxor × 0.5µs ≈ **51.5µs** | 豁免 | **≈128µs**（HBM 口径）/ b 重读上限 ≈188µs |

> 引用常数：`CONST-mte2-degradation`（MTE2 有效带宽随流量退化，256M 档 3:1 R/W 混合全流量地板 ~1.26 TB/s @48 核，toolchain tilelang 0.1.2+ed787bb/910B2C/CANN 8.5.0，**kb_stale_check 标记 stale，引用前需重验**）；`CONST-vector-launch-overhead`（~0.5µs/op，toolchain 0.1.2+3a214cde/910B2C）；`CONST-copy-floor-method`（≥16M 档计算链被 auto multi-buffer 完全隐藏于 MTE2 窗口，佐证发射项在 compute-light 算子中与流量项同量级或更低）。流量项按本机 40 核 / 常数 48 核折算（1.26×40/48≈1.05 TB/s），**标注为跨核数外推的未实测假设**；发射项 `vbrc` 开销由 CG-2026-0019/0020 引入（源无此 op）。row-broadcast 的 b 行重读（+64MB）为未实证项，交 §1.6.3 备选 + 实验裁决。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的基线位运算算法（位运算本体不可再化简）。逐类别核查：

| # | 优化类别 | 核查结论 | 依据 |
|---|----------|----------|------|
| 1 | 恒等变形降代价（除法转乘倒数 / sqrt→rsqrt / 换底） | **无适用项**：公式无除法、无开方、无超越函数 | §1.3；`docs/Tilelang.language/数学操作/`（T.rsqrt/T.vexp 等均无关） |
| 2 | 稳定化变形（减 max / logsumexp 改写） | **无需求**：整数/布尔无溢出放大路径，无指数/求和结构 | §0.1 边界语义 |
| 3 | 公共子表达式消除 | **无公共子式**：`a`/`b` 各引用一次，无重复子表达式 | §1.3 |
| 4 | 算子降代换 | **无高代价算子**：仅 1 条逐位异或，已是 op 数下界；算术恒等 / 布尔逻辑分解 op 数 ×3–4（§1.6.0 R1） | §1.6.0 R1 候选表 |
| 5 | 访存量削减 | **本体已达下界**；广播路径的访存由 §0.6 R1/R4 行/内层分解削减（连续段替换标量 gather）——属索引路径重设计，非公式变形 | §0.6 R1；§1.6.0 R3 |
| 6 | 归约结构优化 | **无归约** | §0.1 |

**优化结论：无公式级优化空间**。依据：位运算本体 `a ^ b` 是 1 op/元素的最短形态；无公共子表达式、无高代价算子、无除法/超越函数、无归约；两种等价位运算恒等式（算术分解 / 布尔逻辑分解）op 数 ×3–4 且引入中间量，无对应收益（§1.6.0 R1 候选 1/2）。

> 例外说明：§0.6 R3 的 **int8 替换 uint8** 与 §0.6 R1/R4 的**广播索引重映射 + 显式同形物化**是**语义保持的等价替换**（存储表示 / 索引表示 / 操作数形态），不是公式级优化，故不属本节采纳项；但二者均为「必须机器验证才算完成」的等价性论证，已由 `verify_equiv.py` 覆盖（下）。

**等价性机器验证（D-1）** — `examples/bitwise_xor/bitwise_xor_fwd_kernel/verify_equiv.py`（torch CPU，bit-exact / 逐点对照），结果表（本次执行于工具链 tilelang 0.1.2+417dae9908 / CANN 8.5.0）：

| 项 | 对照对象 | 用例 | 违反数 | 结论 |
|----|----------|------|--------|------|
| E1-XOR | int8 存储 vs uint8 存储（§0.6 R3） | 1M 随机字节 + 全 256 角点字节 | 0 | `EQUIV_PASS` |
| E1-OR | 同上（同族位运算交叉核验） | 同上 | 0 | `EQUIV_PASS` |
| E1-NOT | 同上（同族位运算交叉核验） | 同上 | 0 | `EQUIV_PASS` |
| E2-smoke-1k | 新行/内层映射 vs 源 divmod 映射（§0.6 R1） | `[1024]^[1024]` | a=0,b=0,oob=0 | `EQUIV_PASS` |
| E2-hidden-state-prefill | 同上 | `[2048,4096]^[2048,4096]`（8.39M 索引） | 0,0,0 | `EQUIV_PASS` |
| E2-cnn-feat-broadcast | 同上 | `[16,256,56,56]^[256,1,1]`（12.85M 索引） | 0,0,0 | `EQUIV_PASS` |
| E2-row-broadcast | 同上 | `[4096,4096]^[1,4096]`（16.78M 索引） | 0,0,0 | `EQUIV_PASS` |
| E2-bias-add / row-broadcast-3d / scalar-broadcast / bias-3d | 同上 | 测试广播模式 | 0,0,0 | `EQUIV_PASS` |
| E2-a-broadcast-outer / -2d / -mid-broadcast | 同上 | a 侧广播（角色互换） | 0,0,0 | `EQUIV_PASS` |

**OVERALL: `EQUIV_PASS`**（脚本随算子目录留存，Stage 2 维度 8 可重跑）。§3.1 公式拆解以「优化后公式 = 原式」（本体不变、索引路径按 §0.6 R1/R4）为唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `a ^ b` | 源 `T.Parallel(threads,npt)` 标量循环逐元素位异或 | `T.vxor(a_ub, b_ub, y_ub)`（同形） | ✅ | —（佐证：`docs/Tilelang.language/逻辑操作/T.vxor.md` §2.4；探针 A/Q1-r2-same 官方风格用例；int8/16/32/64 实测） |
| 2 | 行内标量广播（末维 stride=0 操作数） | 源 stride-0 索引隐式广播 | `T.vbrc(a1_ub/b1_ub → a_ub/b_ub)` + `T.vxor`（**必需**：vxor 无自动广播，CG-2026-0020） | ✅ | —（佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；探针 D 与设计期直连探针 int8/16/64 实测） |
| 3 | 行 base divmod（`outer_a_base`/`outer_b_base`） | 源 `_compute_broadcast_offsets` 逐元素 divmod | **无**（无逐元素等价向量 API；行级标量） | ❌ | **block 级索引/任务映射计算**：每 task 仅 O(ndim) 次标量整数 divmod（ndim≤4），非逐元素热点（较源降 ~L 倍，L 达 10³–10⁶）；`T.arange`+divmod 向量化会引入逐元素标量物化（`CONST-arange-scalar-materialize` ~1ns/elem）反而更慢 |
| 4 | 尾块长度 `tail = T.min(block_size, L − inner_off)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：非整除时逐 task 不同边界；已采用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`）；padding 方案违反 host 禁改输入约束（ascend-constraints.md §4） |
| 5 | 核内 task 调度循环 `for i in T.serial(num_local_tasks)` | 循环 | 无（persistent 结构需循环推进 task） | ❌ | **tile 级调度循环**（§5.5 persistent 分核标准结构，core-split-strategy.md §1 要素③；`T.vcast.md` §2.4 示例同款）；边界为编译期静态值，循环体一次一个完整 task（内部全向量化） |
| 6 | host 侧 `num_logical/num_kernels/chunks` 计算、block_size 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合 decision-tree §1） |

**向量化结论**：2 类逐元素计算点（#1 逐位异或、#2 标量广播物化）**已全部向量化**，替代 API 均有 docs + testing 双佐证并经探针实测；保留 4 处标量/循环（#3 行 base 索引 / #4 动态边界 / #5 tile 调度循环 / #6 host 元数据），逐项理由见上表——§6 循环结构与本结论一致（kernel 内无任何逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（Op 层已把两输入展平为一维连续、输出回 broadcast shape），但**核内布局与 lane 映射是自由变量**。本节枚举候选并量化取舍。迁移任务：源 GPU 的 `threads×npt` 轴是**硬件组织的输入而非结论**（thread/warp 轴与 NPU Vector 轴不对应），须独立评估。**本算子特有约束**：`T.vxor` 无自动广播（CG-2026-0020），故「依赖向量指令自动广播」的候选在本算子直接不可实现。

**候选矩阵**（弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 展平一维（I/O 原生）+ innermost 维向量化 + 显式同形物化 | 展平 N 轴（`k=1`）/ coalesced 末维（`k>1`） | 无 | 基线：连续段 `T.copy`，访存 MTE2 最优；对齐 `block_size×E` 32B 整除；UB 3–5 buffer ≤ 192KB（§4.5）；广播维用 vbrc 物化 | ✅ **采纳**（主选） |
| 2 | 核内 N 维 auto-broadcast tile（`T.vxor` 直接吃 `[T,d1,…,dk-1]` vs `[1,d1,…,1]`，rank 一致） | 末维/交织维 | 无 | ❌ **不可实现**：探针 B-col/B-row/C-3d/Q1-r2-col/Q1-r2-row 实测 `T.vxor` 在两个操作数 rank≥2 形状不同时 BiShengHIR `'hivm.hir.vxor' op operands' shape are inconsistent`（CG-2026-0020：hivm.hir.vxor 缺 broadcast attribute；同工具链 `T.vor` 同形状通过，证明是 vxor 特性而非通用限制）；文档相符：`T.vxor.md` §2.1「src0, src1(张量时) 和 dst 应具有相同的形状」、§2.2.2 无「支持自动广播」条目 | ❌（vxor 不支持广播） |
| 3 | host 侧预广播物化 `b` 到输出 shape | 展平 N | host `expand→contiguous` | ❌ **负收益**：违反 host 禁改输入/新增物化的约束（ascend-constraints.md §4）；额外 GM 流量 `N·E`（b 物化）+ 内存；无收益 | ❌ |
| 4 | **2-D 块化 tile** `[R_blk, L]`：a_ub[R_blk,L] 连续载入；b 末维 stride=1 时 b_row[(1,L)] 载入一次 + `T.vbrc → b_ub[R_blk,L]`（或 b 全广播时从 1 元素 vbrc）；`T.vxor([R_blk,L],[R_blk,L])` **同形** | 末维 L | 无（a 连续） | 备选：**减少 b 行重读**（row-broadcast 场景从每行 1 次 copy 降为每 tile 1 次）+ 减少 task/vbrc 发射次数（R/R_blk vs R）；代价 UB `3·R_blk·L·E`（int32 R_blk=2,L=4096 → 96KB，×1.7 膨胀 ≈163KB <192KB ✓；int64 R_blk=2 → 192KB 手工 → ×1.7 超限，须 R_blk=1 退化） | ⚠️ **备选**（仅 `b` 外层维全广播场景；见裁决计划） |

**轴质量评分（主选 #1）**：整除性——末维 `L` 与平铺 `block_size` 均为向量宽度整数倍（int8×16/int16×8/int32×4/int64×2）；尾 lane 浪费 = `(ceil(L/vw)·vw − L)/L`，manifest 广播末维 `L∈{3136, 4096}`（cnn-feat 3136 / row-broadcast 4096）对 int32×4 均整除（0% 浪费），smoke/hidden-state `L=N_total` 由 `block_size=4096` 整除；累加链——无累加（逐元素位运算，无跨步串行累加维，不阻碍向量化）；repack 代价——无（直接用 I/O 原生连续段）；UB 容量——3–5 buffer × block_size ≤ 192KB（§4.5）。

**源码轴对照（迁移任务要求）**：源 `T.Parallel(threads, npt)` 的隐式轴向是「连续 N 轴 + 每线程 npt 连续元素」，与主选 #1 的「连续段 + innermost 维向量化」一致；源的**广播路径**则退化为逐元素标量索引（无向量轴），NPU 重设计的核心即把它恢复为 innermost 维向量轴 + 显式同形物化（§0.6 R1/R4）。**注意**：源 GPU 的广播索引「隐式 strides」不依赖向量指令广播能力，而 NPU `T.vxor` 无广播，故 NPU 必须在软件层先把 operand 变同形——这是 GPU→NPU 轴映射之外的一层额外重设计。

**实验裁决模式（2-D 块化 tile 备选 #4）**：主选 #1 在「`b` 在全部外层维广播」场景（row-broadcast `[4096,4096]^[1,4096]`、以及 b 全广播的 cnn-feat）每 task 重读一行 b（16–64KB）；其代价是否可被 L2 驻留吸收属**未实证常数**（`CONST-mte2-degradation` 已给 sets=1 L2 口径 43.1 GB/s/core，但此处为 sets>1 的行重读，且 row-broadcast 的 b 行重读上限口径估算 ≈188µs vs 主选 HBM 口径 ≈128µs，裕度不可忽略）。故：
- **主选**：方案 #1（per-row，进 Stage 3）。
- **备选**：方案 #4（2-D `[R_blk,L]` tile，仅「b 外层维全广播」shape；row-broadcast 取 R_blk=2、L=4096，int32 UB ≈96KB；int64 按 R_blk=1 退化为方案 #1——容量否决）。
- **裁决计划**：Stage 4 在 row-broadcast 代表 shape（`[4096,4096]^[1,4096]` int32）与 cnn-feat（`[16,256,56,56]^[256,1,1]` int32）A/B 实测方案 #1 vs #4；指标 = msprof op Task Duration + mte2_active_bw + mte2_instructions；判定阈值 = 备选 latency < 主选 × 0.97 才翻转；按 shape 分派（`b` 外层维全广播走 #4，其余走 #1）；**shape 特化工厂（lru_cache）评估**：wrapper 按 `(N_total, coalesced_shape, strides…)` 特化，天然支持按 shape 分派主选/备选，无全局二选一需求——结果回写 §1.6.3。

**布局决策结论**：选定方案 #1——核内布局 = `(block_size,)` UB buffer（a/b/y 各一，标量广播路径 a1/b1 各一），向量化轴 = **coalesced 末维**（`k=1` 时即展平 N 轴），repack = 无；广播索引经 §0.6 R1/R4 行/内层分解 + 显式同形物化。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、`T.vxor` 操作数即整 buffer）与 §6 循环结构；§4.5 UB 预算按此布局计算。方案 #4 作为 `b` 全外层广播场景备选 + 裁决计划随设计冻结进 Stage 4。


---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定；且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素位异或 + 标量广播物化），无 matmul、无规约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + 1 向量 op + 一次搬出，广播路径外加 1 vbrc） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 task 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | v-prefix（`vxor`/`vbrc`）+ `T.copy` + `T.alloc_ub` 均为 Developer 可用形态 | `T.vxor.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态；探针均为 Developer 模式实测通过 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + `alloc_ub` + v-prefix，Developer） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2；非 persistent 混合算子，无 CG-2026-0010 的 Developer/persistent+gemm 限制，本算子纯 Vector persistent）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix 要求操作数在 UB，`T.vxor.md` §2.1 / `T.vadd.md` §2.3 第 1 条） |
| 计算方式 | v-prefix 向量 API（`T.vxor` + `T.vbrc`），无 `T.Parallel` 逐元素标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式（本体不变）：`y = a ^ b`；索引映射按 §0.6 R1（行 base + innermost 维向量化）；operand 形态按 §0.6 R4（显式同形物化，因 `T.vxor` 无广播）；存储按 §0.6 R3（bool→int8）。拆解与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(row,chunk) = decompose(task)`；`a_base`, `b_base` | 行/块标量分解（编译期 unrolled divmod，仅每 task 一次） |
| 2 | `a_row ← a[a_base : a_base+tail]`（末维 stride=1）或 `a_row ← vbrc(a[a_base])`（末维 stride=0） | a operand 物化为 (block_size,)（连续段或标量广播） |
| 3 | `b_row ← b[b_base : b_base+tail]` 或 `vbrc(b[b_base])` | b operand 物化为 (block_size,)（同上） |
| 4 | `y_row = a_row ^ b_row` | 逐位异或（向量，两 operand **同形**） |
| 5 | `y[flat0 : flat0+tail] ← y_row` | 结果写回（连续段，尾块截断） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| task 分派 | `task = i*num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local_tasks)` | num_kernels/num_local_tasks 编译期常量（§5.5） | Developer |
| 行 base | `Σ coord_i·stride_i` | Python 生成的 `T.floordiv`/`T.floormod` 展开链 | 编译期常量 divisors/strides | Developer |
| operand 搬入 | `x[a_base:a_base+tail] ↔ x_ub[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（runtime 边界 OK；`T.copy.md`；`vec_add_1d.py` L34–L38） | Developer |
| 尾块长度 | `tail = min(block_size, L − inner_off)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md` | Developer |
| 逐位异或 | `y = a ^ b` | `T.vxor(a_ub, b_ub, y_ub)`（`npuir_xor`） | **两 operand 与 dst 严格同形同 dtype**（`T.vxor.md` §2.1 约束；CG-2026-0020）；dtype int8/16/32/64（探针 A） | Developer |
| 标量广播物化 | `a_ub ← vbrc(a1_ub)` | `T.vbrc(a1_ub, a_ub)` | src (1,)→dst (block_size,)（`T.vbrc.md` §2.2.2；探针 D + 设计期直连探针 int8/16/64 实测） | Developer |
| 整除向上取整 | `ceildiv(L, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板 §3.3 规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字，避免 false alarm）。工厂签名保留源契约（wrapper 已冻结调用形态），下半段为**示意**（真实实现由 JIT 参数编译期确定，`if sa==1` / `if sb==1` 为 Python 层静态分派，不产生运行时分支）。

```python
def bitwise_xor_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides,
                           a_numel, b_numel, output_dtype=None, block_size=4096):
    out_dtype = output_dtype or dtype
    # ---- 编译期常量 ----
    vector_cores = NPUUtils.get().get_aicore_num() * 2      # 纯 Vector 翻倍；实查见 §5.5
    k = len(coalesced_shape)
    L = coalesced_shape[-1]                                 # innermost 维
    outer = list(coalesced_shape[:-1])                      # 外层维
    R = math.prod(outer) if k > 1 else 1
    sa, sb = a_strides[-1], b_strides[-1]                   # 末维 stride ∈ {0,1}
    chunks = -(-L // block_size)                            # ceildiv
    num_logical = R * chunks
    num_kernels = min(num_logical, vector_cores)            # persistent 分核（§5.5）
    num_local = -(-num_logical // num_kernels)              # 静态边界

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                a1_ub = T.alloc_ub((1,), dtype)     # 标量广播源（sa==0 时用）
                b1_ub = T.alloc_ub((1,), dtype)
                for i in T.serial(num_local):                 # 核内串行，静态边界
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # --- 行 base：Python 生成 mixed-radix divmod（每 task 一次）---
                        # rem = row; a_base = Σ (rem//div)*a_strides[j] ; rem %= div
                        a_base = <outer divmod a_strides>(row) + (inner_off if sa == 1 else 0)
                        b_base = <outer divmod b_strides>(row) + (inner_off if sb == 1 else 0)
                        tail = T.min(block_size, L - inner_off)
                        # --- 物化 a 为 (block_size,)（R4：vxor 要求同形）---
                        if sa == 1:
                            T.copy(a[a_base : a_base + tail], a_ub[0:tail])
                        else:                                  # a 行内标量广播
                            T.copy(a[a_base : a_base + 1], a1_ub[0:1])
                            T.vbrc(a1_ub, a_ub)
                        # --- 物化 b 为 (block_size,) ---
                        if sb == 1:
                            T.copy(b[b_base : b_base + tail], b_ub[0:tail])
                        else:
                            T.copy(b[b_base : b_base + 1], b1_ub[0:1])
                            T.vbrc(b1_ub, b_ub)
                        # --- 同形逐位异或 & 写回 ---
                        T.vxor(a_ub, b_ub, y_ub)
                        flat0 = row * L + inner_off
                        T.copy(y_ub[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel
```

> 实现注记：① `<outer divmod …>` 为 Python 层按 `outer`/strides 生成的 `T.floordiv`/`T.floormod` 展开链（因子全为编译期常量）；`k==1` 时无外层维，`a_base=inner_off`、`b_base=inner_off`。② `sa==0` 且 `sb==0` 仅在全标量（`L==1`）出现，此时两路都走 `vbrc`（`L==1`/`block_size>1` 不触发 `TRAP-vbrc-same-shape-empty-broadcast`）。③ **`T.vxor` 两操作数必须同形同 dtype**（CG-2026-0020）——这是与 bitwise_and 伪代码的关键差异：bitwise_and 可让 `T.vand` 直接吃广播形状，本算子必须先 `vbrc`/连续段把两 operand 都物化为 `(block_size,)`。④ v-prefix 整 buffer 操作在尾块（`tail<block_size`）时对 `[tail:]` 段垃圾一并计算，但仅 `[0:tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例同款）。⑤ bool 张量由 Op 层 `view(torch.int8)` 后传入（§0.6 R3，wrapper `_as_kernel_input` 已承接）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.vxor`（`npuir_xor`） | `docs/Tilelang.language/逻辑操作/T.vxor.md` §2.1/§2.2/§2.4 | ✅ **探针实测**：int8/int16/int32/int64 **同形** bit-correct（探针 A）；**rank≥2 广播 ✗（CG-2026-0020，探针 B/C + Q1 对照 vor）、rank-1 `(N,)^(1,)` ✗、标量操作数 ✗（探针 E/Q2）、uint8 ✗（CG-2026-0018，探针 A-uint8/G）** |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2/§2.3 | ✅ **探针实测**：(1,)→(N,) 标量广播 bit-correct（int32 探针 D；int8/int16/int64 设计期直连探针）；同形 src/dst 会触发空广播维（CG-2026-0017/TRAP-vbrc-same-shape-empty-broadcast，规避） |
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vxor.md` §2.4 示例 | ✅（examples 先例） |
| `T.alloc_ub(shape, dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`vec_add_1d.py` L28–L30 | ✅ |
| `T.copy`（gm↔ub、切片、runtime 边界） | `docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` L34–L38 | ✅ |
| `T.min` / `T.ceildiv` | `docs/Tilelang.language/比较操作/T.min.md`；`docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.serial` + `task = i·K + cid` + `if` | `T.vcast.md` §2.4 示例（persistent 分发模式） | ✅ |
| `@tilelang.jit(out_idx=[2], target="npuir")` | 源 `out_idx=[2]` 契约 + `example_elementwise_add.py` L13 | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 grid；NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg / `T.Parallel(threads,npt)`） | Yes | `is_npu=True` + v-prefix 取代（§0.5 舍弃项 / §0.6 R2） |
| GEMM 非整除 | No（无 GEMM；非 GEMM 的行/尾块由 §6.4 处理） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 1–8192 vs 物理 Vector 核 40） | §5.5 persistent 分核 |
| Host 侧输入操作约束 | Yes（Op 层展平输入；b 广播不做 host 物化） | 展平在 Op 层（既有契约）；kernel 内 host 侧零数据操作，仅元数据 → 符合 ascend-constraints.md §4 |
| uint8 张量不可用 | Yes（bool 存储路径原走 uint8） | §0.6 R3：改 int8 承载（CG-2026-0018） |
| **`T.vxor` 不接受任何自动广播** | Yes（广播路径核心约束） | §0.6 R4：显式 `T.vbrc`/连续段把两 operand 物化为同形（CG-2026-0020） |
| **`T.vxor` 不接受标量操作数** | Yes（行内标量广播） | §0.6 R1/R4：`T.vbrc` 物化广播向量（CG-2026-0019/0020） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R2 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1–#2 |
| 广播索引 | 逐元素 unrolled divmod + 标量 gather（隐式 strides） | 行级 unrolled divmod + innermost 维向量化 + **显式同形物化** | §0.6 R1/R4 |
| 广播语义承载 | 向量指令/标量索引隐式 | `T.vxor` 无广播 → `T.vbrc`/连续段显式物化 | §0.6 R4（CG-2026-0020） |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| bool 存储 | `view(uint8)` + `*BoolStorageFwdKernel` | `view(int8)`（位级等价） | §0.6 R3 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(num_logical, 40)` | §0.6 R2 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[2])` | `@tilelang.jit(out_idx=[2], target="npuir")` | K1 适配 |
| 工厂参数 | `(…, threads=256, num_per_thread=8)` | `(…)`；callable(`block_size`) | wrapper K11（接口契约冻结） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail = T.min(...)` 尾块模式、runtime 边界 `T.copy`、v-prefix 用法 |
| `examples/bitwise_and/bitwise_and_fwd_kernel/DESIGN.md` | 高度相似（同族 bitwise 迁移先例，含已踩坑） | §0 迁移分析组织、persistent 分核、§1.6 调研/优化分析结构、L0 测试结构、wrapper 契约（**差异：vand 支持广播、vxor 不支持**） |
| `testing/npuir/bit_ops/test_binary_bit_op_dev.py` | 高度相似（位运算 v-prefix 官方测试） | `T.npuir_and(...)`、`T.alloc_shared`、`T.serial` 分发；位运算 dtype 覆盖 |
| `examples/elementwise/vec_add_auto_brc.py` | 中度相似（广播 v-prefix） | 广播形态 v-prefix 用法（注意 vand/vor 可自动广播，vxor 不可，仅作对照） |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/` | 高（TileOPs harness 迁移先例） | §0 迁移分析组织、persistent 分核、L0 测试结构、wrapper 契约 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `a` | `(a_numel,)` | int8/int16/int32/int64（+ bool 经 int8 存储承载） | Op 层已展平的 1-D 连续张量；广播时 `a_numel ≤ N_total` |
| `b` | `(b_numel,)` | same_as(a) | 同上 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N_total,)` | same_as(a) | `out_idx=[2]`；Op 层 `reshape(out_shape)` 还原广播形状 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a_ub` | `(block_size,)` | 输入 dtype | UB | a operand（连续段载入后 / vbrc 后，恒为 `(block_size,)`） |
| `b_ub` | `(block_size,)` | 输入 dtype | UB | b operand（同 a，供同形 `T.vxor`） |
| `y_ub` | `(block_size,)` | 输出 dtype | UB | 输出行 |
| `a1_ub` / `b1_ub`（仅末维 stride=0 时用） | `(1,)` | 输入 dtype | UB | 标量广播源（`vbrc` 输入） |

> buffer 复用：`a1_ub`/`b1_ub` 仅在需要时分配（编译期分支）；`a_ub`/`y_ub` 同 dtype 且活区间不相交时可合并（Stage 3 视分配情况）。**注意**：`T.vxor` 要求「src0, src1, dst 同形」，故 `a_ub`/`b_ub`/`y_ub` 必须同为 `(block_size,)`——不能像未限制广播的算子那样把某 operand 留成小形状。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 task）：
GM[a] ──T.copy(连续段或 1 元素)──> UB[a_ub]（(block_size,)，必要时经 vbrc 物化）─┐
GM[b] ──T.copy(连续段或 1 元素)──> UB[b_ub]（(block_size,)，必要时经 vbrc 物化）─┼─ T.vxor(同形) ──> UB[y_ub] ──T.copy(连续段)──> GM[y]
（末维 stride=0 的 operand 先经 T.vbrc 物化；全程 UB 内，无 L1/L0 参与，无中间 GM 往返）

行 base 标量：仅每 task 一次 divmod（不在 GM 数据路径上）
```

### 4.5 UB 内存预算

block_size = 4096（推荐，见 §5.2）：

| Buffer | int8 @4096 | int16 @4096 | int32 @4096 | int64 @4096 |
|--------|-----------|-------------|-------------|-------------|
| a_ub / b_ub / y_ub | 3 × 4096 B = 12288 | 3 × 8192 B = 24576 | 3 × 16384 B = 49152 | 3 × 32768 B = 98304 |
| a1_ub / b1_ub | 2 × 1 B ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| **手工总计** | ~12 KB | ~24 KB | ~48 KB | ~96 KB |
| **×1.7 auto-multi-buffer 膨胀（上界）** | ~20 KB | ~41 KB | ~82 KB | ~163 KB |
| 目标平台 UB 容量 | 192 KB | 192 KB | 192 KB | 192 KB |
| 余量 | ✓ | ✓ | ✓ | ✓（163 < 192） |

> 膨胀系数引用 `CONST-capacity-910B2C`（auto-multi-buffer ~1.7x；该条目另有「显式 alloc 结构 ~1.12x / 反例 0.981x」的结构依赖记录，本设计取保守 1.7x 上界）。block_size 上限（3 buffer 预算 + 1.7x）：int64 ≤ 4915、int32 ≤ 9830、int8 ≤ 39320；block_size=4096 对 int64 余量最紧仍达标。block_size 由 wrapper config 传入（默认 `_DEFAULT_BLOCK_SIZE=2048`（整数路径 K11）与 `4096`（bool 存储路径），本设计推荐 4096，理由见 §5.2；Stage 4 可扫）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 无 TIR 动态符号——工厂闭包按 `(N_total, dtype, coalesced_shape, strides, a_numel, b_numel)` **编译期特化**（源同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例触发一次 JIT 特化；kernel 边界全静态 |

> `L`、`R`、`chunks`、`num_logical`、`num_kernels`、`num_local` 全为 Python int（编译期常量）；kernel 内唯一运行时标量为 `tail`（1 次/task）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
# out_idx=[2]：main(a, b, y) 的 y（index 2）为输出，承接源算子契约（_bitwise_xor_fwd_kernels.py L47/L70）
# target="npuir"：K1 适配（GPU 版无 target；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素位异或（1 条 `T.vxor`）+ 标量广播物化（`T.vbrc`），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（decision-tree §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_size = 4096   # 推荐；选择理由：
# ① innermost 维单块覆盖：manifest 广播行 L ∈ {3136, 4096, 8192?}，L≤4096 者 chunks=1
#    （cnn-feat L=3136、row-broadcast L=4096 均单块）——消除行内二次分块，
#    使 vbrc/vxor 发射次数减半（§1.6.0 估算：cnn-feat 发射项 205µs→103µs）
# ② 尾轴 32B 对齐：4096×{1,2,4,8}B = 4K/8K/16K/32K，均 32 整除 ✓
# ③ 向量宽度整除：int8×16/int16×8/int32×4/int64×2 均整除 4096 → 尾 lane 浪费 0%
# ④ UB 预算：3 buffer ≤ 96KB（int64），×1.7 膨胀 ≈163KB < 192KB ✓（§4.5）
# ⑤ 同形路径（L=N_total 极大）由 chunks 分块，block_size 即 tile 粒度
# 注：wrapper 整数路径默认 block_size=2048（K11 折叠 256×8）；4096 为设计推荐值，Stage 4 扫。

L = coalesced_shape[-1]
R = prod(coalesced_shape[:-1]) if k > 1 else 1
chunks = ceildiv(L, block_size)
num_logical = R * chunks          # 逻辑 task 数（一维划分）
```

一维划分（无 block_M/block_N 之分）——本算子为逐元素/广播，GEMM 类分形/非整除策略不适用。**分类倾向**：`k==1`（同形 / 全标量广播）退化为展平 N 轴分块；`k>1`（含广播）走行/内层分块。

### 5.3 约束分析

- **对齐约束**: `block_size×E` = 4K/8K/16K/32K（32B 的整数倍）✓；末维向量维整除（尾 lane 0%）✓；尾块 `tail < block_size` 段长度非 32B 倍数时的拷贝行为依赖 `vec_add_1d.py` 先例（风险 §9.1 R8，L0-6 覆盖）
- **UB 容量**: 见 §4.5，三 dtype 路径 ×1.7 膨胀后 ≤163KB ≤ 192KB ✓
- **L0/L1 容量**: 不适用（纯 Vector，不使用 L0/L1）
- **分形限制**: 不适用（无 Cube）
- **同形约束**: `T.vxor` 三 buffer 恒为 `(block_size,)`（CG-2026-0020），§3.3 与 §4.3 一致

### 5.4 注意事项

- **非整除处理**：尾块 `tail = T.min(block_size, L − inner_off)` 截断（§6.4）；manifest 各 workload 的 `L` 对 block_size=4096 要么整除（L=4096 / L=N_total 由 chunks 整除）要么单块尾段（L=3136），尾块路径服务任意广播形状
- **`chunks>1` 的行**：`L>4096`（如 hidden-state L=8.39M → chunks=2048）→ 同行多 chunk 共享同一 `a_base` 外层项、`inner_off` 不同（Python/常量算）
- **广播物化**：末维 stride=0 的 operand 每 task 走 1 次 vbrc（§0.6 R4）；仅在 `b` 外层维全广播的 shape 上存在 b 行重读（§1.6.3 备选 #4）

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素表（依据 docs/开发指南.md §3.3）。

- **物理核数依据（实查）**: 本机 Ascend 环境实查（2026-10-13）：

  ```python
  from tilelang.utils.npu_utils import NPUUtils
  n = NPUUtils.get().get_aicore_num()   # → 20
  vector_cores = n * 2                  # → 40（纯 Vector 算子核数翻倍规则）
  ```

  实查返回值 **20**（AI Core）⇒ 纯 Vector 核 **40**。注：`CONST-aicore-910B2C` 记 910B2C=24（⇒48），与本机实查 20 不同源——按标准要求以**本机实查值 20/40** 为准（跨设备部署时工厂闭包每次 JIT 特化重新实查，无硬编码）。

- **逻辑核数**: `num_logical = R × ceildiv(L, block_size)`，按 manifest workload（block_size=4096）：

  | workload | a_shape | b_shape | k | R | L | chunks | num_logical |
  |----------|---------|---------|---|----|---|--------|-------------|
  | smoke-1k | [1024] | [1024] | 1 | 1 | 1024 | 1 | **1** |
  | hidden-state-prefill | [2048,4096] | [2048,4096] | 1 | 1 | 8 388 608 | 2048 | **2048** |
  | cnn-feat-broadcast | [16,256,56,56] | [256,1,1] | 3 | 4096 | 3136 | 1 | **4096** |
  | row-broadcast | [4096,4096] | [1,4096] | 2 | 4096 | 4096 | 1 | **4096** |

- **规模判定**: smoke-1k `num_logical=1 ≤ 40` → **无需适配**（单核单 task，附依据：逻辑核数已 ≤ 物理 Vector 核数）；其余三个 workload `num_logical ∈ {2048, 4096} ≫ 40`，且无法通过增大 block_size 缩减到 ≤40（block_size 受 UB 上限约束，§4.5；且 L/tile 粒度下限决定最小逻辑核数）→ **极大规模**。
- **分核方案**（极大规模 persistent，core-split-strategy.md §1 要素③；小 N 场景同一模板自动退化）:
  - 固定启动内核数 `num_kernels = min(num_logical, 40)`（编译期常量；smoke-1k 自动取 1）
  - 核内串行：`for i in T.serial(num_local)`，`num_local = ceildiv(num_logical, num_kernels)` —— **静态边界**（N/L/block_size/40 均编译期常量）
  - 任务映射：`task = i × num_kernels + cid`（grid-stride），`if task < num_logical` 屏蔽越界
  - 各 workload 分布：hidden-state → 40 核 × ≤52 task/核；cnn-feat / row-broadcast → 40 × ≤103——核启动次数固定 40 次（对比超发 2048–4096 次串行调度），负载不均 ≤1 task/核（≤2%）
  - **失败信号**（供 Stage 3/4 对照）：① `num_local` 依赖运行时值 → 违反标准；② msprof 显示核串行排队/启动开销占比异常 → 检查是否误用逻辑核数作 grid；③ 负载不均（末核 2 倍）→ 核对 `min()` 与 `ceildiv` 上界


---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block/task 级调度循环，无任何逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| task 级（核间） | persistent 并行 | `T.Kernel(num_kernels, is_npu=True)` | §5.5 分核：≤40 核常驻 |
| task 级（核内多任务） | 串行 | `T.serial(num_local)`（静态边界） | persistent 标准结构；摊薄核启动开销 |
| 行内块级 | 已并入 task 空间（`chunks`） | — | `chunk` 由 `task % chunks` 得到，无嵌套循环；`chunks=1`（block_size≥L）时无块维度 |
| 元素级 | 向量化 | `T.vxor`（整 buffer 一条向量指令，**同形**） | §1.6.2 #1；无 `T.Parallel` 逐元素循环（与源差异：§0.6 R2） |
| 标量广播 | 向量化 | `T.vbrc`（1 元素 → block_size） | §1.6.2 #2（末维 stride=0 时，vxor 无广播故必需） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 核间：persistent ≤40 核
    a_ub, b_ub, y_ub, a1_ub, b1_ub = T.alloc_ub(...)        # §3.3 完整版（三主 buffer 恒同形）
    for i in T.serial(num_local):                            # 核内：静态边界串行
        task = i * num_kernels + cid
        if task < num_logical:
            row = task // chunks; inner_off = (task % chunks) * block_size
            a_base, b_base = <行 base 标量>; tail = T.min(block_size, L - inner_off)
            # 物化（连续段 / vbrc 标量广播）→ T.vxor(同形) → 写回，全向量化
            ...
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：单阶段 load→compute→store 算子（每 task 一次进出），无 K 维迭代累加结构可供双缓冲；逐元素位运算计算量极小、访存主导，`auto multi-buffer` 可将向量 op 隐藏于 MTE2 窗口内（`CONST-copy-floor-method`：≥16M 档计算链被完全隐藏）。多 task 双缓冲属参数级调优，交 Stage 4 按 msprof 评估。

### 6.4 尾块处理

`L % block_size ≠ 0` 时（任意广播形状；block_size=4096 下 manifest 仅 L=3136 触发单块尾段）：

```python
inner_off = (task % chunks) * block_size
tail = T.min(block_size, L - inner_off)      # 本块有效长度
T.copy(a[a_base : a_base + tail], a_ub[0:tail])   # 零起点 dst 切片（规避 UB 非零起点对齐）
...（v-prefix 整 buffer 操作；UB 中 [tail:] 陈旧段不落 GM）
T.copy(y_ub[0:tail], y[flat0 : flat0 + tail])
```

模式来源：`examples/elementwise/vec_add_1d.py` L31–L38（官方已验证形态）。不采用 host 侧 padding（违反 host 禁改输入约束，ascend-constraints.md §4）；不采用 Kernel 动态 block（N/L 已编译期特化，`tail` 标量一次/task 开销可忽略）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

无手动同步点。依据：① 每 task 计算完全独立、无核间数据依赖（无 workspace、无跨核归约）→ 不需要 `T.sync_block_set/wait`；② task 内 `T.copy → T.vbrc/T.vxor → T.copy` 的顺序依赖由 Developer 模式自动同步保证（`T.vxor.md`/`T.vbrc.md` 均为无显式同步示例）；③ persistent 核内 `T.serial` 顺序执行天然串行；④ `if task < num_logical` 无跨 task 数据交换。

### 7.3 pass_configs 配置

无特殊 pass 配置（默认管线）。环境变量：`TILELANG_ASCEND_MODE=Developer`（§2.3）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现 `torch.bitwise_xor`），**不复刻** §0.6 的 NPU 算法（不复用行/内层索引、vbrc、persistent 结构、同形物化）——保证验证独立性。

```python
import torch

def golden_bitwise_xor(a, b):
    """Bitwise-XOR 参考实现（源算子语义，广播直算）。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_bitwise_xor.py：
    ref_program = torch.bitwise_xor(a, b)；整数/布尔按 exact_compare）。输入为
    原始 broadcast 形状（非展平），输出 = broadcast_shapes(a.shape, b.shape)。
    """
    return torch.bitwise_xor(a, b)
```

### 8.2 精度标准

逐位运算在整数/布尔域**精确**（无舍入），容差 = **bit-exact**：

| dtype | 比对方式 | 说明 |
|-------|----------|------|
| int8 / int16 / int32 / int64 | exact（`torch.equal` / `exact_compare`） | 逐位异或精确；NPU 与 golden 应 bit 级一致 |
| bool（经 int8 存储） | exact | `out.view(torch.bool)` 后与 `torch.bitwise_xor`（bool）精确比对 |
| 输出 shape | `out.shape == broadcast_shapes(a,b)` | Op 层还原，测试断言 |

来源：`examples/TileOPs/tests/ops/test_bitwise_xor.py` 的 `exact_compare`；manifest `same_as(input)`。

### 8.3 L0 门槛测试计划

> L0 为 Stage 3 开发精度门槛（golden 对比必须全过）；完整 L1/L2/Boundary 分层套件由 `tilelang-op-develop` 展开。

| # | 用例 | 输入 | 通过标准 |
|---|------|------|----------|
| L0-1 | smoke 同形全 dtype | `[1024]×[1024]`：int8/int16/int32/int64，`randint` 全范围（含负值/极值） | vs golden `torch.bitwise_xor`，exact |
| L0-2 | 大规模同形（各 dtype） | `[2048,4096]×[2048,4096]` int32/int64（hidden-state 代表） | vs golden，exact |
| L0-3 | **广播三模式** | `[16,256,56,56]×[256,1,1]`（cnn-feat 通道广播）、`[4096,4096]×[1,4096]`（row-broadcast）、`[2,64,128]×[1,1,128]`（bias）int32 | vs golden，exact + shape 断言 |
| L0-4 | **a 侧广播** | `[1,4096]×[4096,4096]`、`[2,1,3]×[2,3,3]`（角色互换，覆盖 `sa=0`） | vs golden，exact |
| L0-5 | 标量广播 / 全标量 | `[64,128]×[1,1]`、`[1]×[1]`（`L=1` 全标量退化，避 `TRAP-vbrc-same-shape`） | vs golden，exact |
| L0-6 | 尾块非整除 | `L` 非 `block_size` 整数倍：`[1000]×[1000]`、`[3136]×[3136]`、`[4097]×[4097]`（int32），单块尾段/多块 | vs golden，exact（覆盖 §6.4 / 风险 §9.1 R8） |
| L0-7 | bool 存储路径 | `[2048,4096]×[2048,4096]`、`[2,512,768]×[1,1,768]` bool → int8 存储 | vs `torch.bitwise_xor`(bool)，exact + `out.dtype==bool` |
| L0-8 | persistent 边界 | `L/N` 使 `num_logical ∈ {39, 40, 41}`（核数切换边界，block_size 使能） | vs golden，exact + 无越界 |
| L0-9 | 二补码极值 | int8/int16/int32/int64 的 `INT_MIN`/`INT_MAX`/`-1` 组合 | vs golden，exact（验证符号位不参与位异或语义） |
| L0-10 | **vxor 同形约束回归** | `[M,N]×[1,N]`、`[M,N]×[M,1]` int32（源 GPU 可直接，NPU 经显式物化） | vs golden，exact（验证 §0.6 R4 物化正确，不依赖 vxor 广播） |

**L0 测试要点**：输入经 `.npu()` 放置；`randint` 覆盖负值与二补码极值（位运算对符号敏感度检验）；广播用例额外断言 `out.shape == broadcast_shapes(a.shape, b.shape)`；bool 用例断言输出 dtype 与值。L0-10 专门覆盖本算子相对 bitwise_and 的关键差异（vxor 无广播）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束/风险 | 影响 | 缓解方案 |
|---|-----------|------|----------|
| R1 | **uint8 GM 张量在 npuir 不可用**（CG-2026-0018，探针 A-uint8/G 实测） | bool 存储路径原走 uint8 不可行 | §0.6 R3：改 int8 承载（位级恒等，E1 机器验证）；wrapper `_as_kernel_input`/`_restore_public_dtype` 已承接 |
| R2 | **`T.vxor` 不接受任何自动广播**（CG-2026-0020，探针 B/C/Q1 实测；同工具链 `T.vor` 可广播） | 广播路径不能用单条 vxor，必须显式物化 | §0.6 R4：`T.vbrc`/连续段先把两 operand 变同形；L0-10 专门回归。**这是本算子相对 bitwise_and 的最关键差异** |
| R3 | **`T.vxor` 不接受标量/size-1 操作数**（CG-2026-0019/0020，探针 E/Q2 实测） | 行内标量广播不能用单条 vxor | 用 `T.vbrc` 物化广播向量（探针 D + 直连探针验证）；代价 = 每 task 多 1 vbrc + 1 buffer |
| R4 | **`T.vxor` 文档声明 src1 可为标量但实测失败**（`T.vxor.md` §2.1 vs 探针 E/Q2；且 `T.vor` 标量同样失败） | 文档-行为不符，易误导实现 | 一律走 `T.vbrc` 物化，不依赖标量操作数；已并入 CG-2026-0019/0020 证据链 |
| R5 | **bool 存储路径 wrapper 适配**：`BitwiseXorBoolStorageFwdKernel` 现 `SUPPORTED_DTYPES=(uint8,)`、Op 层 `view(torch.uint8)` | 不改为 int8 则 bool 用例失败 | wrapper 已实现 int8 桥接（`_kernel_dtype_str`/`_as_kernel_input`/`_restore_public_dtype`）；Stage 3/5 核对 bool 用例（见 §0.6 R3 / verify_equiv E1） |
| R6 | **行内标量广播的 `vbrc` 发射开销**（§1.6.0 估算：cnn-feat 发射项约 103µs，与流量项相当） | 广播 workload 可能发射受限 | block_size 取 ≥L（单块/行）使 vbrc 次数最小；Stage 4 评估「2-D `[R_blk,L]` tile（备选 #4）」减少 task/vbrc 次数 |
| R7 | **row-broadcast 的 b 行重读**（主选每 task 重读一行 b，16KB×4096≈64MB） | 未优化时 b 读流量计入上限口径 ≈188µs（vs HBM 口径 128µs） | b 仅 16KB，预期 L2 驻留；§1.6.3 已列备选方案 #4 + Stage 4 实验裁决 |
| R8 | **尾块非 32B 对齐拷贝** | T.copy 报错或数据错乱 | 采用 `vec_add_1d.py` 零起点 dst 切片先例；L0-6 强制覆盖 |
| R9 | **常数为 stale / 跨核数外推**：`CONST-mte2-degradation` 被 kb_stale_check 标 stale；本机 20 核 ≠ 常数 48 核 | 设计期估算下界可能失准 | 估算行已标注；Stage 4 baseline 后按 hardware-cost-model §3 回填「设计估算 vs 实测」偏差行 |
| R10 | **极小 L 的效率退化**（一般形状）：若某广播形状 innermost 维很小且外层 R 很大（如 `[65536,1]×[65536,8]`：L=8, R=65536） | 每 task 向量过短、task 数过大 | manifest/测试均不触发；作为已知局限记录（不阻塞）；如出现可在 Op 层合并 broadcast 维度后重排 |
| R11 | **int64 的 2-D 备选容量约束**（备选 #4 R_blk=2 时 UB ×1.7 膨胀超 192KB） | int64 row-broadcast 无法用 2-D tile | 备选 #4 仅 int32（R_blk=2）；int64 按 R_blk=1 退化为方案 #1（容量否决，§1.6.3 已记） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 用 uint8 声明张量/UB | bool 存储路径 | 输出被静默映射 float32、数值全错（CG-2026-0018） | 一律用 int8（§0.6 R3） |
| `T.vxor(a_ub, b_ub, c_ub)` 两操作数形状不同 | 广播路径（`[M,N]×[1,N]` 等） | BiShengHIR `'hivm.hir.vxor' op operands' shape are inconsistent`（CG-2026-0020） | 先 `T.vbrc`/连续段把两 operand 物化为同形（§0.6 R4） |
| `T.vxor(a_ub, scalar, c_ub)` | 行内标量广播 | BV/MLIR verify fail（CG-2026-0019/0020） | 先 `T.vbrc` 物化广播向量 |
| 误以为 vxor 像 vand/vor 一样能自动广播 | 照抄 bitwise_and 设计 | 编译失败 | 本算子必须显式物化（§0.6 R4） |
| `T.vbrc(x, y)` src/dst 同形 | `L==1` 全标量 | `empty broadcast dims array` verify fail（CG-2026-0017/TRAP-vbrc-same-shape-empty-broadcast） | 保证 dst 为 `block_size` (>1)；`L==1` 时 src (1,)→dst (block_size,) 合法 |
| `num_kernels` 误用逻辑核数 | 直接照抄逻辑核数作 grid | 数千核串行调度、启动开销剧增 | §5.5 方案为准（`min(num_logical, 40)`） |
| `T.copy` 静默跨 dtype 转换 | copy 的 src/dst dtype 不一致 | 精度路径失控 | 所有 `T.copy` 严格同 dtype；bool→int8 在 Op 层/wrapper view 完成 |
| 行 base divmod 写成逐元素 | 把 `_compute_broadcast_offsets` 直接搬进 kernel 元素循环 | 标量热点，性能崩溃 | 仅在 task 级（每 task 一次）做，见 §0.6 R1 |
| block_size 超 UB 上限 | config 传入过大 | UB 溢出编译失败 | §4.5 上界（int64 ≤ 4915 / int32 ≤ 9830 / int8 ≤ 39320，1.7x 膨胀口径） |

### 9.3 特殊场景处理

- **同形（无广播）**：`k==1, R==1, sa==sb==1` → 展平连续段路径（与源同形快路径一致）
- **标量广播**（`[64,128]×[1,1]`）：coalesced `[(8192,1,0)]`，`k=1, sa=1, sb=0` → b 单元素 + `vbrc`
- **a 侧广播**（`[1,4096]×[4096,4096]`）：`sa=0` 分支（symmetry）
- **全标量**（0-dim×0-dim）：coalesced 兜底 `(1,0,0)`，`L=1`，两路 vbrc（dst=block_size>1）
- **非整除 N/L**：`tail` 动态裁剪（§6.4）；manifest 全整除不触发（L 对 4096 整除或单块）
- **N=0 / 空维度**：Op 层 `prod()==1` 兜底或上层拦截，kernel 不触达（与源一致）
- **混合 dtype 输入**：Op 层 `forward` 已拒绝（dtype 一致性校验，`test_bitwise_xor.py`）
- **fp8/浮点 dtype**：manifest 不含（`_BITWISE_DTYPES` 仅整数/布尔）；Op 层拒绝浮点（`test_bitwise_xor_rejects_float_dtype`）

---

## 10. 交付清单

### 10.1 目录结构

```
examples/bitwise_xor/bitwise_xor_fwd_kernel/
├── DESIGN.md                         # 本设计文档
├── verify_equiv.py                   # 等价性机器验证（E1 存储等价 + E2 索引映射等价）
├── history_version/
│   ├── design_probe_vxor.py              # D-3 设计期探针（vxor dtype/broadcast/vbrc/uint8/scalar）
│   └── design_probe_vxor_followup.py     # 探针跟进（vxor vs vor 广播对照 + 修复后的 scalar/f16-transit）
└── RETROSPECTIVE.md                  # Stage 1 复盘（skill Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | E1/E2 全 `EQUIV_PASS`（OVERALL EQUIV_PASS） |
| `bitwise_xor_fwd_kernel.py`（kernel 实现） | ⬜ 待实现（Stage 3） | 工厂 `bitwise_xor_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, output_dtype=None, block_size=4096)` → callable(block_size) → `@T.prim_func main(a, b, y)`；广播路径按 §3.3；集成目标 wrapper `examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/bitwise_xor.py` |
| golden | 随实现文件交付（§8.1） | `torch.bitwise_xor`（广播直算） |
| L0 测试 | 随实现文件交付（§8.3） | 10 个代表性 case |

### 10.3 命名规范

- 项目目录：`bitwise_xor`；算子目录：`bitwise_xor_fwd_kernel`（源算子函数名，harness 提取件同名）
- 实现文件名：`bitwise_xor_fwd_kernel.py`；测试入口内嵌（`if __name__ == "__main__"`，vec_add_1d.py 先例）
- `@tilelang.jit` 装饰函数名保留 `bitwise_xor_fwd_kernel`（迁移执行规则）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ golden（§8.1）
3. ⬜ kernel 实现（§3.3：同形 `k=1` 路径 → 广播 `k>1` 四子情形（含显式同形物化）→ bool int8 路径）
4. ⬜ L0 门槛测试全绿（§8.3）
5. ⬜ harness 集成（wrapper 契约对接 + bool 存储 view 目标 uint8→int8 适配，§9.1 R5）

---

## 11. 设计核心与 GPU→NPU 关键重设计（下游报告提炼用）

> 本节把全文设计核心收敛为 4 条可逐算子引用的要点。

1. **多核分块（persistent 分核）**：把展平后的 coalesced 输出切成 `R 行 × ceildiv(L, block_size)` 个逻辑 task，用一维 `T.Kernel(min(num_logical, 物理核数), is_npu=True)` + 核内 `T.serial`（静态边界，grid-stride 映射）常驻 ≤40 核，替换源 GPU 的 CUDA grid 超发 × `T.Parallel(threads, npt)` 两级线程并行；核数由 `NPUUtils.get().get_aicore_num()*2` 实查（本机 20⇒40）。
2. **向量化（v-prefix 取代逐元素标量）**：逐元素 `a ^ b` 由 `T.vxor` 一条向量指令（128bit lane：int8×16/16×8/32×4/64×2）覆盖整块，行内标量广播由 `T.vbrc` 物化；GM↔UB 走连续段 `T.copy`，尾块 `tail=T.min(...)` 截断。源 `T.Parallel` 逐线程标量 `^` 与 `register_copy` fragment 分别由 v-prefix 与 `T.alloc_ub` 承接。
3. **广播路径映射重设计（索引 + 显式同形物化）**：源广播路径对每元素做 O(ndim) unrolled divmod + 标量 GM gather → 改为「外层行标量 base（每 task 仅 O(ndim) 次 divmod，降 ~L 倍）+ innermost 维连续向量化」，且因 **`T.vxor` 无任何自动广播**（CG-2026-0020，探针 B/C/Q1 实测；同工具链 `T.vor` 可广播）**必须用 `T.vbrc`/连续段把两个 operand 显式物化为 `(block_size,)` 同形**后再 `T.vxor`——这是本算子相对 bitwise_and 的最关键重设计。
4. **数据类型承载（1 字节无符号桥接）**：npuir 不支持 uint8 GM 张量（CG-2026-0018，输出静默变 float32），bool/uint8 的 1 字节存储统一改由 **int8** 承载（位级恒等，`view(torch.int8)` 进、`view(public_dtype)` 出），wrapper 以 `_as_kernel_input`/`_restore_public_dtype` 桥接冻结 Op 的同 dtype 契约。

