# floor_divide_fwd_kernel 算子设计文档

> 迁移任务（`scenario=migration`，`migration_mode=harness`，GPU 仓为 TileOPs 同构工程）。源算子 = GPU 仓 TileOPs 的 `floor_divide_fwd_kernel`（`_make_binary_explicit` 工厂 + `BinaryKernel` 二进制模板的 `FloorDivideFwdKernel` 特化，`op_func(a,b) = T.Cast(a.dtype, T.floor(a_f32 / b_f32))`），迁往 TileLang-NPUIR（`target="npuir"`，华为昇腾 NPU），编程模式固定 `Developer`。
>
> 本文 §0 记录源算子三问解读 → 算法调研（Phase R，结论见 §1.6.0，作为 M1 判定输入）→ 硬件耦合性判定 → NPU 算法重设计的全过程；§1–§11 基于**重设计后的 NPU 算法**展开。
>
> 关键能力结论由设计期探针实测（D-3，脚本见 `history_version/design_probe_floordiv.py` 与 `design_probe_floordiv_bcast.py`，均为最小张量、每探针墙钟 <10 min）：
> - **`T.vfloordiv`（`npuir_floordiv`，`C=floor(A/B)`）与 `T.vfloor`（`npuir_floor`）编译成功但运行期 Vector 核挂死**（`T.vfloordiv` 同形 fp32/fp16/bf16、切片形、`alloc_shared` 形全部挂起；`T.vfloor` 同形 fp32 挂起）——**不可用**。反例：同结构 `T.vdiv` 正常运行。工具链 `tilelang 0.1.2+417dae9908ad` / CANN 8.5.0 / Ascend910。
> - **可用链**：`cast→fp32` + `T.vdiv` + `T.vcast(round_mode="floor")` + `cast→out`。实测**同形 fp32 / fp16 / bf16 bit-exact**（vs `torch.floor(a.float()/b.float()).to(dtype)`），且 `T.vcast(f32→f32, round_mode="floor")` 单独可用（即“向量 floor”）。
> - **`T.vdiv` 支持自动广播**（与 `T.vxor`/`T.vcmp` **不同**，CG-2026-0020/0021 族）：`[M,N]÷[1,N]`（列广播）、`[M,N]÷[M,1]`（行广播）、`[M,N]÷scalar`（标量）实测**全部 bit-exact**——广播路径**无需**像 eq/bitwise_xor 那样强依赖 `T.vbrc`。
> - 本机 `NPUUtils.get().get_aicore_num() = 20` ⇒ 纯 Vector 核 **40**。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素下取整除法 `y = floor(a / b)`，等价于 `torch.floor_divide(a, b)` / `torch.div(a, b, rounding_mode="floor")`（manifest `ref_api: "torch.floor_divide"`）。支持 NumPy/PyTorch 广播：`y.shape == broadcast_shapes(a.shape, b.shape)`。**先除法后下取整，商为整值浮点**（如 `floor(7.0/2.0)=3.0`）。
- **I/O 契约**：两输入一输出；两输入同 dtype；**输出 dtype = 输入 dtype（`same_as(input)`）**，非比较族那样恒为 bool；输出 shape = 两输入 broadcast 结果。算子层把两输入 `contiguous()` 后展平为 1-D `(a_numel,)` / `(b_numel,)`，输出展平为 `(N_total,)`，kernel 只看一维张量；广播 lowering（`coalesce_broadcast_dims`）在 kernel 构造期完成（`examples/TileOPs/tileops/kernels/elementwise/floor_divide/floor_divide.py` L128）。
- **规约语义**：**无规约、无跨元素依赖**——每个输出元素只依赖同位置的 `a`/`b` 元素（经广播索引），`y[i]` 与 `y[j]` 完全独立。因此**并行组织/分核/累加分组的变化不影响任何数值结果**（除法/floor 均为逐元素）。
- **dtype 语义**：**float-only**，`_FLOAT_DTYPES = float16 | bfloat16 | float32`（`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L154–L158，`FloorDivideFwdKernel.SUPPORTED_DTYPES = _FLOAT_DTYPES`，同文件 L1544；manifest signature `input: {dtype: "float16 | bfloat16 | float32"}`）。**关键精度语义**：源码注释（同文件 L1538–L1541）明确——除法与 floor **在 fp32 域计算**，原因是 (1) CUDA 的 `cutlass::half_t` 无 `hfloor`；(2) **fp16 除法会先把商舍入到 fp16 再 floor**（如 `2.999…` 在 fp16 舍入为 `3.0`，得 `floor=3` 而非 `2`）。即 `T.cast(a,fp32) / T.cast(b,fp32)` → `T.floor` → `T.Cast(a.dtype, ·)`。整数/布尔 dtype **被拒绝**（`test_floor_divide_rejects_integer_dtype`）。
- **边界语义**：无 NaN 特殊语义（除法 + floor 均按 IEEE 浮点传播：`floor(±inf)=±inf`、`floor(NaN)=NaN`）；除零 → `±inf` → floor 仍 `±inf`。**输入域**：测试/基准均用正数（`test` 用 `rand+0.1`；bench `positive=True`，两操作数 ∈ [0.1, 1.1)）以避开除零与符号边界。空维度/0-size 输入由 Op 层 `prod()==1` 兜底或上层拦截（`torch.broadcast_shapes` 把 0-dim 归一为 `(1,)`）；非整除 shape 由广播规则处理。
- **host 侧语义**：`contiguous().view(-1)` 展平、`coalesce_broadcast_dims` 索引表示选择、`reshape(out_shape)` 回形状——属 Op 层契约（实现手段，非语义），由 NPU 侧 wrapper（`examples/TileOPs/tileops/kernels/elementwise/floor_divide/floor_divide.py` 的 `FloorDivideFwdKernel`；Op 层 `examples/TileOPs/tileops/ops/elementwise/floor_divide.py` 的 `FloorDivideFwdOp`）承接，不属于本 kernel 的重设计范围。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.floor(a.float()/b.float()).to(dtype)` 广播直算），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a`（input） | `(a_numel,)` | float16 / bfloat16 / float32 | Op 层已展平的 1-D 连续张量 |
| `b`（other） | `(b_numel,)` | same_as(a) | 同上 |
| `y`（output） | `(N_total,)` | **same_as(a)** | `out_idx=[2]`；**输出为一维展平视图**（源码 `main` 的 `y: T.Tensor((N_total,), out_dtype)`），Op 层再 `reshape(out_shape)`，无转置/布局变换。广播时 `a_numel` / `b_numel` / `N_total` 三者可不同 |

**输出 shape/dtype 依据**：源文件 `_floor_divide_fwd_kernels.py` L55–L58 / L80–L83（`y: T.Tensor((N_total,), out_dtype)`，`out_dtype = output_dtype or dtype`）；manifest `shape_rules: output.shape == broadcast_shapes(input.shape, other.shape)`、`outputs.output.dtype: "same_as(input)"`；测试 `test_floor_divide.py` 断言 `out.shape`。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/floor_divide/_floor_divide_fwd_kernels.py`（Stage 0 提取件，Pattern A：`_make_binary_explicit` 内联 `op_func` 为 `a_f32 = T.cast(a, "float32"); b_f32 = T.cast(b, "float32"); y = T.Cast(a.dtype, T.floor(a_f32 / b_f32))`）；GPU 上下文：`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L556–L609（`_make_binary_explicit`）、L890–L1128（`BinaryKernel`）、L1535–L1550（`FloorDivideFwdKernel`）。

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host） | `coalesce_broadcast_dims(a_shape, b_shape)` → `out_shape, coalesced_shape, a_strides, b_strides`（stride=0 表示广播维）；`_is_contiguous_same_shape` 选快路径 | a_shape/b_shape | 索引表示（编译期常量） | 广播 lowering（实现手段） |
| 2 | `block_size = threads * num_per_thread`；grid = `T.ceildiv(N_total, block_size)` | threads/npt | CUDA 任务划分 | 任务划分 |
| 3a（同形连续快路径） | `for i,j in T.Parallel(threads,npt): idx=(bx*threads+i)*npt+j; a_f32=T.cast(a[idx],"float32"); b_f32=T.cast(b[idx],"float32"); y[idx]=T.Cast(a.dtype, T.floor(a_f32/b_f32))` | GM a/b | GM y | `y = floor(a/b)`（同形，fp32 域） |
| 3b（广播路径） | `flat_idx=…; (a_off,b_off)=_compute_broadcast_offsets(...); a_f32=T.cast(a[a_off],"float32"); b_f32=T.cast(b[b_off],"float32"); y[flat_idx]=T.Cast(a.dtype, T.floor(a_f32/b_f32))` | GM a/b | GM y | `y = floor(a/b)`（广播索引，fp32 域） |
| 4（host） | Op 层 `reshape(out_shape)` | `(N_total,)` 输出 | broadcast shape | 后处理（Op 层契约） |

`_compute_broadcast_offsets`（源 L167–L183）是**编译期 unrolled 的 mixed-radix divmod 链**：沿 coalesced 各维依次 `coord = remaining // divisors[d]; remaining %= divisors[d]; a_off += coord*a_strides[d]`，最后 `a_off += remaining*a_strides[-1]`。**这是逐元素的标量整数 divmod**（每输出元素 O(ndim) 次除法/取模）。

**数据流与内存访问模式**（GPU 视角）：

```
GM[a] ──(逐元素 strided/标量读)──> register ──cast→f32──┐
GM[b] ──(逐元素 strided/标量读)──> register ──cast→f32──┼─ f32 除 ─ floor ─ cast→dtype ──> register y ──> GM[y]
（同形快路径：连续段访问；广播路径：逐元素标量索引 gather 读 GM）
```

**循环与并行结构**：CUDA grid（`ceildiv(N_total, block_size)` 个 block）× block 内 `T.Parallel(threads, num_per_thread)` 二维线程并行；`threads=256`，`num_per_thread=4`（`_strategy_npt("explicit_parallel", <float dtype>)` 对 fp32 与「explicit_parallel + fp16/bf16」均返回 4，源 L181–L195），`block_size=1024`；尾块由 TileLang `LegalizeSafeMemoryAccess` 隐式保护。

**host 侧逻辑**：`@tilelang.jit(out_idx=[2])` + `functools.lru_cache(maxsize=32)` 按 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, …)` 特化编译；`out_dtype = output_dtype or dtype`。NPU wrapper 冻结工厂签名为返回 `callable(block_size)`（K11），`default_config` 给 `{"block_size": 256×4 = 1024}`（`floor_divide.py` L77/L161）。

### 0.4 优化手段解读（为什么快）

> 对照 migration-analysis.md §4.1 清单逐项核对；源码显式优化手段如下（无 SMEM tiling / mma / warp shuffle / 异步流水 / swizzle，对照清单无遗漏）。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel`：`T.Parallel(threads, num_per_thread)` | 满并行 + 每线程多元素 ILP | 每线程连续处理 `npt` 个元素（`idx=(bx*threads+i)*npt+j`） | CUDA block/thread 线程模型 | 硬件强相关 |
| 2 | 连续段 coalesced 读写 | 访存带宽 | 同形路径按 `block_size` 连续段访问，尾轴连续 | CUDA memory coalescing | 硬件强相关 |
| 3 | `coalesce_broadcast_dims`（维度合并） | 减少 divmod 次数 | 合并同广播行为的相邻维，最小化 effective dims | 无（纯索引表示） | 可移植 |
| 4 | 编译期 unrolled divmod 索引（`_compute_broadcast_offsets`） | 免运行时 shape 查询、广播通用化 | 常量除法在 build 期展开；但**每元素仍 1 次 divmod 链** | 无（通用），但代价随 ndim 线性增长 | 可移植（代价高，见 §1.6.0 R3） |
| 5 | 同形连续快路径（`_is_contiguous_same_shape`）跳过广播机制 | 消广播开销 | 分支在 build 期选择 kernel body | 无（编译期分派） | 可移植 |
| 6 | `register_copy` 策略（fragment load→compute→store，binary 仅同形） | 数据驻留寄存器 + 128bit 向量访存 | `T.alloc_fragment` + `T.copy` | GPU 寄存器堆 + fragment 机制 | 硬件强相关 |
| 7 | 工厂闭包 + `lru_cache(maxsize=32)` | 摊薄 JIT 编译开销 | 每参数组合只编译一次 | 无（host 层） | 可移植 |
| 8 | **fp32 域计算 + 单次结果 cast**（`T.cast→f32; div; T.floor; T.Cast→dtype`） | **精度正确性**（避 fp16 商舍入导致的 floor 偏差） | 把 fp16/bf16 提升到 fp32 做除法与 floor，结果只舍入一次 | 无（纯数值语义） | **可移植且必须保留** |

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `y = floor(a/b)`（fp32 域） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；逐元素无累加顺序问题；fp32 域精度语义**必须保留**（§0.4 手段 8） |
| I/O 契约（1-D 进 1-D 出、same_as、广播 out_shape） | 语义 | 无 | — | **保留** | 输出 shape/dtype 由 §0.2 源码解读确认；广播形状还原在 Op 层 |
| dtype 契约（float16/bf16/fp32，整数拒绝） | 语义 | 无 | — | **保留** | `_FLOAT_DTYPES` 沿用；整数在 Op 层 `SUPPORTED_DTYPES` 拒绝（对照 manifest/signature） |
| host 展平/coalesce/回形状 | 语义（Op 契约） | 无 | — | **保留** | NPU wrapper 已承接；host 侧仅元数据/视图操作，符合 ascend-constraints.md §4 |
| `coalesce_broadcast_dims` 维度合并 | 算法 | 无 | — | **保留** | 索引表示选择，与硬件无关；wrapper 已承接（`floor_divide.py` L128） |
| 同形连续快路径（build 期分支） | 算法 | 无 | — | **等价替换** | 统一进 §0.6 R1 的行/内层方案（k=1,R=1 退化为同形路径），意图（免广播机制）承接 |
| 一维 grid `ceildiv(N_total, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1；`examples/elementwise/vec_add_1d.py`）；核数按 §5.5 分核适配（→ R2） |
| `T.Parallel(threads, npt)` 两级线程并行 | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 R2（一维 Kernel + Vector 向量化；npt 的多元素意图由 128bit 向量 lane 承接） |
| `threads=` / `num_per_thread` kwarg | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已把 `threads*npt` 折叠为单一 `block_size`，无性能意图丢失（并行组织由 R2 承接） |
| 连续段 coalesced 搬运 | 优化 | CUDA coalescing | 有（GM↔UB 连续段 T.copy） | **等价替换** | `T.copy` 连续段 GM↔UB（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款） |
| `register_copy`（fragment 驻留） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment`→`T.alloc_ub`（migration-analysis.md §5.3「register 累加→UB 中间缓冲」）；v-prefix 要求操作数在 UB |
| **逐元素 divmod 广播索引** | 优化 | 无（通用） | 无等价（NPU 无逐元素整数 divmod 快路径；且源形态为 GM 标量 gather） | **重新设计** | → §0.6 R1（外层行标量 base + innermost 维向量化） |
| **fp32 域除法+floor 本体** | 优化 | 无（纯数值） | 有（`T.vdiv` + `T.vcast(round_mode="floor")`） | **等价替换 + 重设计** | → §0.6 R4（原语层：一体化 `T.vfloordiv`/`T.vfloor` **运行期挂死**（探针），改用 `vdiv`+`vcast(floor)` 两段链；fp32 域中转保留） |
| 工厂闭包 + lru_cache 特化 | 优化（host） | 无 | — | **保留** | wrapper `forward` 已按 `block_size` 惰性构建 + 缓存（`floor_divide.py` L170–L172） |
| GPU grid 自由超发 | 优化 | CUDA 调度器 | 无（超发被串行调度 + 核启动开销） | **重新设计** | → §0.6 R2（persistent 分核，core-split-strategy.md §1） |

**判定统计**：保留 7 项、等价替换 3 项、重新设计 3 项（R1 广播索引 / R2 并行与分核 / R4 除法+floor 原语链）、舍弃 1 项（`threads=`/`npt` kwarg，理由见表）。

### 0.6 NPU 算法重设计

**重设计项 R1: 广播索引路径（逐元素 divmod/gather → 外层行标量 base + innermost 维向量化）**

- **源方案**：广播路径对**每个输出元素**执行一次编译期 unrolled mixed-radix divmod 链（`_compute_broadcast_offsets`），得到 `a_off`/`b_off` 后以标量 gather 读 GM 的 `a[a_off]`、`b[b_off]`；同形路径才用连续段。性能意图：以「编译期常量除法展开」把广播通用化的代价压到单元素 O(ndim) 整数 divmod（GPU 多核并发消化）。
- **NPU 新算法**：把 coalesced 输出视为 **`R = prod(coalesced_shape[:-1])` 个「行」× 行长 `L = coalesced_shape[-1]`** 的二级结构（`k=1` 时 `R=1, L=N_total`，同一模板覆盖同形与广播两路）。对每个「行」（即一个 task）：
  1. **行 base 标量**：`outer_a_base = Σ_{i<k-1} coord_i(row)·a_strides[i]`（`coord_i` 由 row 对 `coalesced_shape[:-1]` 的 mixed-radix 分解得，编译期 unrolled，**仅每 task 一次**，非每元素）；`b` 同理。末维：`a_base = outer_a_base + (inner_off if a_strides[-1]==1 else 0)`（末维 stride ∈ {0,1}）。
  2. **行内 operand 物化**：末维 `L` 按 `block_size` 分块（`chunks = ceildiv(L, block_size)`）。每块把两个 operand 载入 `(block_size,)` UB（fp32 域，见 R4）：
     - `a_strides[-1]==1`（a 末维连续）→ 连续段 `T.copy(a[a_base:a_base+tail] → a_ub[0:tail])`；
     - 否则（a 末维 stride 0，a 在行内为标量）→ `T.copy(a[a_base:a_base+1] → a1_ub)` + `T.vbrc(a1_ub → a_ub)`（**注意**：`T.vdiv` 支持自动广播，故此 vbrc 可优化掉，见 §1.6.3；主选保留以复用同族已验证脚手架）。
     - **末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`**（证明：输出末维 size>1 时至少一个操作数在该维 size>1，其连续 stride=1），故 `(0,0)` 组合不可能（`L==1` 全标量退化为两路 `T.vbrc`，`block_size>1` 时合法）。
- **语义保持论证**：**数学恒等**——新映射对每个 flat 索引给出的 `(a_off, b_off)` 与源 divmod 链**逐点相同**（行分解 `row=flat//L`、`inner=flat%L` 恰为末维与其余维的 mixed-radix 分解，`a_off = Σ_{i<k-1}coord_i·a_strides[i] + coord_{k-1}·a_strides[-1]`），**已由 `verify_equiv.py` E1 机器验证**（9 组形状含全部 manifest + 测试广播模式，`a_off`/`b_off` 逐点 mismatch=0、越界=0，EQUIV_PASS）。除法/floor 逐元素、无跨元素依赖；广播物化（vbrc/连续段）与 PyTorch 广播逐点一致。空维度/0-dim 由 Op 层归一，kernel 不触达。
- **意图承接**：源优化手段 3（维度合并）原样保留（wrapper 沿用 `coalesce_broadcast_dims`）；手段 4（编译期 unrolled divmod）的「免运行时 shape 查询」意图由编译期 unrolled **行** divmod 承接，代价从「每元素 O(ndim)」降为「每 task O(ndim)」（约降 `L` 倍）；手段 5（同形快路径）由 `k=1⇒R=1` 的统一退化承接。

**重设计项 R2: 并行结构与分核（CUDA block×thread 两级并行 + grid 超发 → 一维 Kernel + Vector 向量化 + persistent 分核）**

- **源方案**：`T.Kernel(ceildiv(N_total, block_size), threads=threads)` 的 CUDA grid×block 两级并行，block 内 `T.Parallel(threads, npt)` 二维线程映射，每线程标量处理 npt 个元素；grid 大小任意（最大 workload 下数千 block），超发由 CUDA 硬件调度器吞吐。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；核内 `for i in T.serial(num_local_tasks)`（**静态边界**）以 grid-stride 映射 task `= i*num_kernels + cid`，`if task < num_logical` 屏蔽越界。元素级并行交给 Vector 单元：逐元素除法由 `T.vdiv` 一条向量指令覆盖整块（128bit 向量 lane：fp32 ×4 / fp16 ×8），floor 由 `T.vcast(round_mode="floor")` 一条向量指令覆盖整块。核数按 §5.5 分核（`num_kernels = min(num_logical, 物理 Vector 核数)`）。
- **语义保持论证**：task↔行/c块一一对应、各 task 完全独立（无跨 task 依赖、无竞写），映射顺序/分核数变化不改变任何输出值；向量化只是把逐元素标量运算换成等价的逐元素向量运算（lane 间独立）。无 NaN/累加顺序问题（除法/floor 逐元素，与源一致）。
- **意图承接**：手段 1（`T.Parallel(threads,npt)` 的满并行 + ILP）→ Vector 128bit lane + `block_size` tile 粒度；手段 2（coalesced）→ `T.copy` 连续段；手段 6（fragment 驻留）→ `T.alloc_ub` 原地缓冲；`threads=`/npt 折叠为 `block_size`（wrapper K11），无性能意图丢失。

**重设计项 R4: 除法+floor 原语链（一体化 floor-div 原语不可用 → `vdiv` + `vcast(round_mode="floor")` 两段链，fp32 域）**

- **源方案**：GPU 在**原生 float 域**做 `op_func(a,b) = T.Cast(a.dtype, T.floor(T.cast(a,"float32") / T.cast(b,"float32")))`——fp32 提升、fp32 除法、fp32 `T.floor`、结果 cast 回原生 dtype。CUDA 上这是标量运算，无原语形态约束。
- **NPU 新算法（探针实测驱动）**：
  - **一体化原语不可用**：npuir 语言层提供 `T.vfloordiv`（`npuir_floordiv`，`C=floor(A/B)`，`customize_npuir.py` L425–L436）与 `T.vfloor`（`npuir_floor`，同文件 L412–L422），**语义完美匹配本算子**；但探针 `design_probe_floordiv.py` 实测：`T.vfloordiv` 同形 fp32/fp16/bf16、显式切片形、`alloc_shared` 形**全部编译成功但运行期 Vector 核挂死**（进程超时）；`T.vfloor` 同形 fp32 同样挂死。同结构 `T.vdiv` 正常运行（对照）⇒ **该原语在本工具链不可用**（登记能力缺口，见 §9 R1）。
  - **可得链**：`cast(in→fp32, rint)` → `T.vdiv(a32, b32, q)`（fp32 除法）→ `T.vcast(q, yf, round_mode="floor")`（fp32 向量 floor，探针 `iso_f32 flooronly` 实测 `T.vcast(f32→f32, "floor")` 精确）→ `cast(yf→out, rint)`。实测**同形 fp32/fp16/bf16 bit-exact**（探针 `pure`/`E1`/`E2`，vs `torch.floor(a.float()/b.float()).to(dtype)`，maxdiff=0）。
  - **必须 fp32 域**：fp16/bf16 先在 fp32 域算（探针 E2 已证 bit-exact）；若在原生半精度域直接除+floor（无中转），与 golden 产生 `2.999…` 类偏差——`verify_equiv.py` E3 负向对照实测 fp16 66 处、bf16 430 处偏差（maxdiff 1/4），**证明 fp32 中转是正确性必需**，非可选优化。
  - **fp32 输入免第二次 cast**：`T.vcast(f32→f32, rint)` 恒等 cast 会产生非法结果（探针 `C1` 得 NaN），故 fp32 路径**直接** `T.vdiv` + `T.vcast(q, yf, "floor")`，不做恒等 cast（§3.3 的 `if dtype=="float32"` 分支）。
- **语义保持论证**：**逐位恒等**——`vdiv` = IEEE fp32 除法（与源 `a_f32/b_f32` 同）；`vcast(round_mode="floor")` = `T.floor`（向量 floor，探针精确）；半精度→fp32 提升为**无损**（`rint`，fp16/bf16 值在 fp32 中精确）；结果 `vcast(→out, rint)` 与源 `T.Cast(a.dtype, ·)`（floor 后为整值，舍入无关）逐位一致。整体链**已由 `verify_equiv.py` E2 机器验证**（fp16/bf16/fp32 随机 + 角点，含 ±inf/NaN/±0/次正规/整数边界，`EQUIV_PASS`，violations=0）。边界语义：`floor(±inf)=±inf`、`floor(NaN)=NaN` 由 vcast floor 的 IEEE 语义承接（L0-10 角点覆盖）。
- **意图承接**：源手段 8（fp32 域计算 + 单次结果 cast）**完整保留**——新链同样是「半精度 → fp32 运算 → 单次 cast 回」；仅把「标量 `T.floor`」换成「向量 `T.vcast(floor)`」，把「标量除法」换成「向量 `T.vdiv`」。源的一体化标量 `floor(a/b)` 意图由 `vdiv`+`vcast(floor)` 两条向量指令承接（原语缺口所致，见 §9 R1）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 `T.Parallel` 满并行 + ILP | Vector 向量化（`T.vdiv`/`T.vcast` 128bit lane）+ 单一 `block_size` tile（R2） |
| 2 连续段 coalesced | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 `coalesce_broadcast_dims` 维度合并 | 原样保留（wrapper 承接，驱动 §0.6 R1 的 outer/inner 分解） |
| 4 编译期 unrolled divmod | R1 行级 unrolled divmod（每 task O(ndim)，较源每元素降 `L` 倍）+ innermost 维向量化 |
| 5 同形连续快路径 | R1 的 `k=1 ⇒ R=1` 统一退化 |
| 6 `register_copy` fragment 驻留 | `T.alloc_ub` 原地缓冲链 |
| 7 工厂 + lru_cache 特化 | wrapper `forward` 惰性构建 + 缓存（原样保留） |
| 8 fp32 域除法+floor + 单次结果 cast | R4：fp32 中转保留（E3 证明必需）；标量 `floor` → 向量 `T.vcast(floor)`，标量除 → `T.vdiv`（一体化 `vfloordiv` 运行期挂死） |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/floor_divide/_floor_divide_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 原始上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_binary_explicit` @ L556、`BinaryKernel` @ L890、`FloorDivideFwdKernel` @ L1535）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/arithmetic.py`（`FloorDivideFwdOp` @ L127）。
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/floor_divide/floor_divide.py`（`FloorDivideFwdKernel`：工厂签名保留 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel)`，返回 callable(`block_size`)）；Op 层 `examples/TileOPs/tileops/ops/elementwise/floor_divide.py` 的 `FloorDivideFwdOp`。
- **规格来源**：manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `FloorDivideFwdOp`（4 workloads + roofline `flops=2·N / bytes=(A+B+N)·elem_bytes`）；测试 `examples/TileOPs/tests/ops/test_floor_divide.py`（`ref_program = torch.floor(a.float()/b.float()).to(a.dtype)`，含 3 组广播 + float-only 拒绝 + roofline）；基准 `examples/TileOPs/benchmarks/ops/bench_floor_divide.py`（`positive=True`）。
- golden 以 §0.1 语义为依据（`torch.floor(a.float()/b.float()).to(dtype)` 广播直算），见 §8.1。
---

## 1. 概述

### 1.1 算子名称

`floor_divide_fwd_kernel`（Op 层名 `FloorDivideFwdOp` / 算子族 `floor_divide`）

### 1.2 功能描述

逐元素下取整除法前向核：对两个已展平为一维的同 dtype 浮点张量计算 `y = floor(a / b)`，支持 NumPy/PyTorch 广播（输出 shape = `broadcast_shapes(a_shape, b_shape)`）。纯逐元素、无规约、无跨元素依赖；**float-only**（float16 / bfloat16 / float32），输出 dtype = 输入 dtype。除法与 floor 在 fp32 域执行（§0.1 精度语义）。

### 1.3 数学公式

$$
y[i] = \left\lfloor \frac{a[\mathrm{off}_a(i)]}{b[\mathrm{off}_b(i)]} \right\rfloor, \qquad i = 0, 1, \ldots, N_{total}-1
$$

其中 `off_a(i)`/`off_b(i)` 为广播映射（源算法用逐元素 divmod 链，见 §0.3；NPU 用行/内层分解，见 §0.6 R1）；除法与 floor 在 fp32 域执行后 cast 回输出 dtype（§0.6 R4）。非广播（同形）时 `off_a(i)=off_b(i)=i`。

参考 API：`torch.floor_divide(a, b)` / `torch.div(a, b, rounding_mode="floor")`。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的结构差异逐条注明来源）**

单遍逐元素行向量算法（§1.6.0 调研选定基线公式，无更优算法族替代）：

1. **任务分解（编译期）**：把 coalesced 输出视为 `R = prod(coalesced_shape[:-1])` 行 × 行长 `L = coalesced_shape[-1]`；`num_logical = R × ceildiv(L, block_size)`。`k=1`（同形 / 全标量广播）退化为 `R=1, L=N_total`。
2. **核间**：`T.Kernel(num_kernels, is_npu=True)`（persistent，`num_kernels = min(num_logical, 物理 Vector 核数)`，§5.5）。
3. **核内（每 task = 一行的一块）**：标量计算行 base（编译期 unrolled divmod，仅每 task 一次）→ 按末维 stride 选择「连续段 `T.copy`」或「单元素 `T.copy` + `T.vbrc`」把两个 operand 载入 `(block_size,)` UB → **半精度→fp32 提升**（fp16/bf16；fp32 免）→ `T.vdiv(a32, b32, q)` → `T.vcast(q, yf, round_mode="floor")` → **fp32→输出 dtype cast**（fp16/bf16；fp32 免）→ 连续段 `T.copy` 写回（尾块 `tail` 截断）。
4. 全程无 L1/L0 参与（纯 Vector）；无中间 GM 往返（每 task 一次进出）。

与源算法的**结构差异**：① 逐元素 divmod + 标量 gather（广播路径）→ **行级标量 base + innermost 维向量化**（§0.6 R1，根因是 NPU 逐元素整数 divmod/gather 无快路径）；② CUDA grid×thread 两级并行 + `T.Parallel(threads,npt)` → 一维 Kernel + v-prefix 向量 + persistent 分核（§0.6 R2）；③ **除法+floor 本体**由「原生 float 域标量 `a/b` 与 `T.floor`」改为「fp32 域向量 `T.vdiv` + `T.vcast(round_mode="floor")` 两段链」（§0.6 R4，因 npuir 一体化 `T.vfloordiv`/`T.vfloor` 运行期挂死）。**fp32 中转精度语义**与源骨架（连续段载入 → 逐元素算 → 连续段写回、片上驻留）一致，进化为 `k=1,R=1` 的统一形态。

**与同族 elementwise 算子的算法差异（迁移视角）**：① 输出 dtype = 输入 dtype（same_as，如位运算族），**无** eq 的 bool→1 字节承载问题；② 计算本体是**除法 + floor**（两条向量指令），**非**单条位运算/比较；③ **`T.vdiv` 支持自动广播**（与 `T.vxor`/`T.vcmp` 关键不同，CG-2026-0020/0021）——广播路径的 operand 物化为**可选优化**而非强制（eq/bitwise_xor 必须显式物化，§1.6.3）；④ **float-only**，整数 dtype 在 Op 层拒绝（位运算族覆盖整数）。

### 1.5 数据流图

```
同形 / 标量广播（k=1, R=1；L 按 block_size 分块）：
  GM[a] ──T.copy(连续段, 尾块截断)──> UB[a_in] ──[fp16/bf16: vcast→f32]──> UB[a32] ─┐
  GM[b] ──T.copy(连续段, 尾块截断)──> UB[b_in] ──[fp16/bf16: vcast→f32]──> UB[b32] ─┼─ T.vdiv(f32) ──> UB[q:f32]
                                                                                     │   ──vcast(f32→f32,floor)──> UB[yf:f32]
                                                                                     │   ──[fp16/bf16: vcast→out]──> UB[y_out] ──T.copy──> GM[y]
更一般广播（k>1；每 task = 一行的一块）：
  (row,chunk) 标量分解 ──> a_base/b_base（编译期 unrolled divmod，仅每 task 一次）
  a 末维 stride=1: GM[a] ──T.copy(a[a_base:a_base+tail])──> UB[a_in]
  a 末维 stride=0: GM[a] ──T.copy(a[a_base:a_base+1])──> UB[a1] ──T.vbrc──> UB[a_in]
  b 同上 ─────────────────────────────────────────────────────────────────────> UB[b_in]
  UB[a_in] ─vcast→f32─> UB[a32] ; UB[b_in] ─vcast→f32─> UB[b32]   （fp32 输入免 cast）
  UB[a32] ÷ UB[b32] ─vdiv─> UB[q] ─vcast(floor)─> UB[yf] ─vcast→out─> UB[y_out] ─T.copy─> GM[y[row*L+inner_off …]]
全路径无 L1/L0 参与（纯 Vector）；无中间 GM 往返（每 task 一次进出）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。`floor_divide` 为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用**轻量调研**深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法 / 输入公式） | `y = floor(a/b)`；fp32 域标量 除 + `T.floor` + cast；广播用逐元素 divmod | — | — | ✅（基线） |
| 1 | **一体化向量原语 `T.vfloordiv`** | `C = floor(A/B)`（npuir 单指令） | 语义完美匹配，但**运行期挂死**（探针：同形 fp32/fp16/bf16、切片形、shared 形全挂）⇒ 本工具链**不可用** | 1 向量指令/task（vs 基线 2），且免自写 floor | ❌（不可用，§0.6 R4/§9 R1） |
| 2 | 一元向量 `T.vfloor` + `T.vdiv` | `yf = floor(vdiv(a,b))` | `T.vfloor` **运行期挂死**（探针）⇒ 同样不可用 | 同 1 | ❌（不可用） |
| 3 | 除法转乘倒数 | `a/b → a·(1/b)`（`1/b` 一次倒数 + N 次乘） | **非等价**：`1/b` 的舍入使 `a·(1/b)` 在整数边界附近偏移，改变 `floor` 结果（带宽受限算子上也无收益） | 负收益（错误） | ❌（非等价） |
| 4 | 原生半精度域直接算（无 fp32 中转） | `floor(a/b)` 在 fp16/bf16 域 | **非等价**：fp16 商先舍入再 floor（`2.999…→3.0`）。E3 负向对照：fp16 66 处、bf16 430 处偏差 | 少 2~4 条 cast/task | ❌（非等价，§0.6 R4/E3） |
| 5 | 整数化 floor | `f32→i32(floor)→f32` | 等价仅限 `|q| < 2^31`（超出即错），且 +2 条 cast | 负收益（+2 op、范围受限） | ❌（范围受限） |
| 6 | 合并 floor 与输出 cast（单条 `vcast(q_f32→out, floor)`） | 一次 cast 同时 floor+舍入到 out | **非等价**：改变舍入顺序——大商（超过 out dtype 整值精度）时与「先 floor 再 cast」不同，且与 golden 不符 | 少 1 条 cast/task | ❌（非等价；§1.6.1） |

**R2 在线算法**：**无在线变体需求（天然单遍）**。结构依据：`y_i` 仅依赖 `(a_{off_a(i)}, b_{off_b(i)})`，**无 running 统计量**（和/最大值/计数）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射无跨元素依赖，单遍即天然在线」）；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径（扫描遍数=1、中间缓冲 O(block_size)、UB 驻留）在候选间无差异。

**R3 复杂度对比**（带宽受限算子——算术强度 2 op/elem ÷ ((A+B+N)·E) 字节，fp32 下 < 0.02 op/B，**访存为主导项**；口径：A/B/N 为展平元素数，E=elem_bytes，`A=B=N` 时同形；候选 3/4/5/6 因非等价**不参与**）：

| 算法候选 | FLOPs（口径：逐元素 op 数） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------------------|---------------|---------|-------------|---------------------|
| **基线（源算法）** | N（fp32 除 + floor，2 op/元素）+ **N·O(ndim) 整数 divmod**（逐元素广播索引；广播路径标量 gather） | (A+B+N)·E，但广播路径为逐元素标量 gather（访存效率低） | 1（每元素每操作数读一次写一次） | O(block_size) tile（寄存器） | N/block_size 个独立 task，零跨核代价；逐元素 divmod 为标量热点 |
| **NPU 重设计（§0.6 R1+R4）** | N·(2~5 op/元素)（fp32 路径 2：vdiv+floor；fp16/bf16 路径 5：2 升 + vdiv + floor + 1 降）+ **R·O(ndim) divmod**（行级，较基线降 ~L 倍）+ 每行 ≤2 条 vbrc 物化 | (A+B+N)·E（连续段） | 1 | O(block_size)（UB：4~7 buffer，§4.5） | num_logical 个独立 task，零跨核同步；persistent 摊薄启动 |
| 候选 1/2（一体化 vfloordiv/vfloor） | N（1 op/元素） | (A+B+N)·E | 1 | O(block_size) | 同上，但**运行期挂死 ⇒ op 数仅为纸面值** |
| 候选 4（原生半精度） | N（2 op/元素）+ R·O(ndim) | (A+B+N)·E | 1 | O(block_size) | 同上，但**非等价**（E3） |

基线（除法+floor 本体）在 **fp32 域**已是可用链的 op 数下界（2 op/元素）；NPU 重设计的收益在**广播索引路径**（把逐元素 O(ndim) divmod/gather 降为行级 O(ndim) + innermost 维连续向量化）；半精度路径多 3 条 cast 是**精度正确性必需**（E3），非冗余。行级重读口径：`sb=1 且 b 在全部外层维广播`（如 row-broadcast `[4096,4096]÷[1,4096]`）时每 task 重读一行 b，未优化口径下 b 读流量 ≈ N·E（b 仅 16KB，预期 L2 驻留，HBM 侧不放大）；该不确定项按 §1.6.3 备选 + 实验裁决处理。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` 佐证或探针实测）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 可融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源逐元素 divmod + 标量除/floor + 标量 gather） | ❌ 逐元素整数 divmod + 标量 GM gather → 落错单元（标量密集，非 Vector） | OK | ✗ 标量 gather 无向量 lane | 尾块依赖动态 shape | 单阶段 | ❌ 淘汰：NPU 无逐元素 divmod 快路径，标量 gather 严重失分 |
| **NPU 重设计（行 base + innermost 向量化 + `vdiv`/`vcast(floor)` + vbrc 物化）** | ✅ 除法→Vector `T.vdiv`（fp32 同形/广播实测 √，探针 A/C/F）；floor→`T.vcast(round_mode="floor")`（实测 √，探针 `flooronly`/`pure`/`E1`/`E2`） | ✅ 4~7 buffer × block_size ≤ 192KB（§4.5，含 1.7x 膨胀） | ✅ block_size×E = 32B 整数倍；向量宽度整除（fp32×4/fp16×8）；尾 lane 浪费 0%（L∈{1024,3136,4096,8388608}） | ✅ N/L/block_size 全编译期特化；tail 仅 1 次/task | 单阶段无需流水；纯 Vector 无 CV 融合 | ✅ **选定** |
| 候选 1（`T.vfloordiv`） | ❌ 前端存在（`customize_npuir.py` L425）但**运行期 Vector 核挂死**（探针） | OK | OK | OK | — | ❌ 淘汰：原语不可用（编译成但挂死；§9 R1） |
| 候选 2（`T.vfloor`） | ❌ 同样运行期挂死（探针） | OK | OK | OK | — | ❌ 淘汰：原语不可用 |
| 候选 3（倒数乘） | ✅ 落 Vector（`T.vrec`/`vmul`）但**非等价** | OK | OK | OK | OK | ❌ 语义不成立 |
| 候选 4（原生半精度） | ✅ 落 Vector 但**非等价**（E3：fp16 66 / bf16 430 偏差） | OK | OK | OK | OK | ❌ 语义不成立 |
| 候选 5/6（整数化 / 合并 cast） | ✅ 落 Vector 但范围受限 / 舍入顺序变 | OK | OK | OK | OK | ❌ 非等价或范围受限 |

**调研结论**：选定 **基线语义（`y = floor(a/b)`，fp32 域）的 NPU 行/内层重设计算法**（§0.6 R1：外层行标量 base + innermost 维向量化 + operand 物化；§0.6 R4：`T.vdiv` + `T.vcast(round_mode="floor")` 两段向量链 + 半精度 fp32 中转）。关键依据：R3 表中 **fp32 域可用链已是最短可运行形态**（一体化 `T.vfloordiv`/`T.vfloor` 纸面 1 op，但运行期挂死，探针实测）；候选 3/4/5/6 均非等价或范围受限（E3 机器验证候选 4 的偏差）；R4 中仅 NPU 重设计对 Vector/UB 容量/对齐/静态边界全部亲和（基线逐元素 divmod/gather 在 NPU 落错单元被淘汰）。与基线的**结构差异**：① 广播索引从「逐元素 O(ndim) divmod + 标量 gather」变为「行级 O(ndim) divmod + innermost 维连续向量化」；② 除法+floor 本体从「原生域标量 `a/b` 与 `T.floor`」变为「fp32 域向量 `T.vdiv` + `T.vcast(round_mode="floor")`」（一体化原语运行期挂死）。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表命中行 `ALG-elementwise`（R1：公共子表达式消除——`a`/`b` 各用一次，无公共子式；除法转乘倒数——**已评估但非等价**，见候选 3；cast 链合并——已评估但舍入顺序改变，候选 6；R2 逐元素天然单遍——与本算子一致）；② 恒等变形自查——`floor(a/b)` 的本体最短形态是 1 除法 + 1 floor；除法转乘倒数、整数化、合并 cast 均非等价（候选 3/5/6）；③ 本仓 `examples/elementwise/`（vec_add_1d/2d、exp2/log2、auto_brc）与 `examples/TileOPs/tileops/kernels/elementwise/mish`、`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor`、`examples/TileOPs/tileops/kernels/elementwise/bitwise_or` 迁移——均直排 v-prefix 单遍结构，无除法/floor 公式变形先例；④ pattern-library `elementwise.md`/`cases.md`——elementwise 条目均为访存/布局调度类（copy-floor、loads-first、MTE3 strided），无除法/floor 公式级条目；⑤ 源算子结构分析（§0.3，广播路径逐元素 divmod 是唯一可优化点，已在 R1 承接）；⑥ npuir 语言层实查（`tilelang/language/customize_npuir.py`）发现 `npuir_floordiv`/`npuir_floor` 前端原语，**探针实测运行期挂死**并登记能力缺口（§9 R1）；⑦ `docs/Tilelang.language/` 全目录检索——`数学操作/` 无独立 floor 文档，floor 能力仅在 `数据类型转换操作/T.vcast.md` §2.1 以 `round_mode="floor"` 形式给出（本设计采用）。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

**设计期 roofline 估算（D-2）**（估算下界 = max(流量项, 发射项, 容量项)；常数引用 `pattern-library/constants.md`；本机实测 `NPUUtils.get().get_aicore_num()=20` ⇒ 40 Vector 核，与 `CONST-mte2-degradation` 的 910B2C=48 核不同源，流量项按 40/48 比例折算并标注；block_size=4096，fp32 路径 op 数=2，加法见下）：

| 代表 workload | 流量项 | 发射项 | 容量项 | 估算下界 |
|---------------|--------|--------|--------|----------|
| smoke-1k `[1024]` fp32（N=1K） | (N+N+N)·4B = 12.3KB ÷ 1.05 TB/s ≈ **0.01µs** | 逻辑核 1 / 40 = 1 task × 2 op × 0.5µs ≈ **1.0µs** | 豁免（轻量调研；tile 级缓冲无跨引擎往返） | **≈1.0µs** |
| hidden-state-prefill `[2048,4096]` fp32（N=8.39M，chunks=2048） | (A+N+N)·4B = 100.7MB ÷ 1.05 TB/s ≈ **95.9µs** | 逻辑核 2048 / 40 ≈ 52 × 2 op × 0.5µs ≈ **52µs** | 豁免 | **≈96µs**（流量主导） |
| cnn-feat-broadcast `[16,256,56,56]÷[256,1,1]` fp32（N=12.85M，chunks=1，R=4096，sb=0→+1 vbrc） | (A 51.4MB + out 51.4MB + b 行内标量 256·4B) ÷ 1.05 TB/s ≈ **97.9µs** | 4096 task / 40 ≈ 103 × (vbrc 1 + vdiv 1 + vcast 1) × 0.5µs ≈ **154.5µs** | 豁免 | **≈155µs**（发射主导） |
| row-broadcast `[4096,4096]÷[1,4096]` fp32（N=16.78M，chunks=1，R=4096，sa=sb=1） | (A 67.1MB + out 67.1MB + b 行 16KB) ÷ 1.05 TB/s ≈ **127.8µs**（b 行重读未计）；含 b 行重读上限口径 ≈ (67.1+64+67.1)MB ÷ 1.05 ≈ **188µs** | 4096 task / 40 ≈ 103 × 2 op × 0.5µs ≈ **103µs** | 豁免 | **≈128µs**（HBM 口径）/ b 重读上限 ≈188µs |

> 引用常数：`CONST-mte2-degradation`（MTE2 有效带宽随流量退化，256M 档 3:1 R/W 混合全流量地板 ~1.26 TB/s @48 核，toolchain tilelang 0.1.2+ed787bb/910B2C/CANN 8.5.0，**kb_stale_check 标记 stale，引用前需重验**）；`CONST-vector-launch-overhead`（~0.5µs/op，toolchain 0.1.2+3a214cde/910B2C）；`CONST-copy-floor-method`（≥16M 档计算链被 auto multi-buffer 完全隐藏于 MTE2 窗口，佐证发射项在 compute-light 算子中可与流量项同量级）。流量项按本机 40 核 / 常数 48 核折算（1.26×40/48≈1.05 TB/s），**标注为跨核数外推的未实测假设**。**fp16/bf16 路径**：流量减半但发射项升为每 task 5–6 op（2 升 + vdiv + floor + 1 降 + 可能的 vbrc），广播类 workload 发射项进一步主导（cnn-feat fp16 ≈ 103×6×0.5≈309µs vs 流量 49µs）→ 直接驱动 §1.6.3 的 2-D tile 备选与 Stage 4 实验裁决。row-broadcast 的 b 行重读（未计上限 +64MB）为未实证项，交 §1.6.3 备选 + 实验裁决。容量项按轻量调研豁免（纯 Vector tile 级缓冲）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的基线算法（fp32 域 `floor(a/b)`）。逐类别核查：

| # | 优化类别 | 候选 | 四要素核查 | 结论 |
|---|----------|------|-----------|------|
| 1 | 恒等变形降代价 | 除法转乘倒数 `a·(1/b)` | 原式 `floor(a/b)` → 优化后 `floor(a·rec(b))` → **等价性论证**：**不成立**——`rec(b)` 先舍入，`a·rec(b)` 与 `a/b` 在整数边界附近可差 ±1，`floor` 结果改变 → 收益量化：N 次除法 → 1 次倒数 + N 次乘（省 ~N 除法）但**结果错误** | ❌ **否决**（非等价；负收益） |
| 2 | 恒等变形降代价 | 合并 floor 与输出 cast（`vcast(q→out, floor)`） | 原式 `cast_to_out(floor_f32(q))` → 优化后 `vcast(q, out, floor)` → **等价性论证**：**不成立**——`floor` 在 fp32 域产生整值，与「floor 后舍入到 out dtype」不同（大商超 out 整值精度时舍入顺序改变）→ 收益量化：少 1 条 cast/task 但**结果与 golden 不符** | ❌ **否决**（非等价；§1.6.0 候选 6） |
| 3 | 稳定化变形（减 max / logsumexp） | — | **无需求**：无指数/求和/溢出放大结构；除法/floor 逐元素 | ❌ 不适用 |
| 4 | 公共子表达式消除 | — | **无公共子式**：`a`/`b` 各引用一次，无重复子表达式 | ❌ 不适用 |
| 5 | 算子降代换 | 一体化 `T.vfloordiv`（1 op 取代 2） | 原式 `floor(a/b)` = vdiv+vcast(floor) → 优化后 `T.vfloordiv(a,b,y)` → **等价性论证**：语义等价（`npuir_floordiv` 定义为 `C=floor(A/B)`，与源同）→ 收益量化：2 op → 1 op/task（发射项减半，广播类 workload 显著） | ❌ **否决（不可用）**：探针实测 `T.vfloordiv` 编译成功但**运行期 Vector 核挂死**（同形/切片/shared 全挂），`T.vfloor` 同样挂死；能力缺口见 §9 R1 |
| 6 | 访存量削减 | — | **本体已达下界**：每元素读 a、b 各一次、写 y 一次；fp32 中转在 UB 内完成（无额外 GM 往返）；广播路径访存由 §0.6 R1 行/内层分解削减（连续段替换标量 gather）——属索引路径重设计，非公式变形 | ❌ 无公式级收益 |
| 7 | 归约结构优化 | — | **无归约** | ❌ 不适用 |

**优化结论：无公式级优化空间（在可用原语约束下）**。依据：`floor(a/b)` 的 fp32 域本体已是 2 条可用向量指令（`T.vdiv` + `T.vcast(floor)`）的最短形态；三种降代/化简候选——除法转乘倒数（候选 1）、合并 cast（候选 2）**非等价**，一体化 `T.vfloordiv`（候选 5）**语义等价但运行期挂死不可用**；无公共子表达式、无高代价超越函数、无归约。

> 例外说明：§0.6 R4 的 **fp32 中转链**（半精度→fp32→除→floor→cast 回）是**语义保持的等价替换**（换用可运行原语），不是公式级优化（它不减少计算量，反而在半精度路径多加 cast，但为正确性必需——E3 证明免中转非等价）；§0.6 R1 的**广播索引重映射**同为语义保持的等价替换（索引表示）。二者均为「必须机器验证才算完成」的等价性论证，已由 `verify_equiv.py` 覆盖（下）。

**等价性机器验证（D-1）** — `examples/floor_divide/floor_divide_fwd_kernel/verify_equiv.py`（torch CPU，bit-exact / 逐点对照），结果表（本次执行于工具链 tilelang 0.1.2+417dae9908ad / CANN 8.5.0 / aicore 20）：

| 项 | 对照对象 | 用例 | 结果 | 结论 |
|----|----------|------|------|------|
| E1-smoke-1k | 新行/内层映射 vs 源 divmod 映射（§0.6 R1） | `[1024]÷[1024]` | a_mm=0,b_mm=0,oob=0 | `EQUIV_PASS` |
| E1-hidden-state | 同上 | `[2048,4096]÷[2048,4096]`（8.39M 索引） | 0,0,0,0 | `EQUIV_PASS` |
| E1-cnn-feat | 同上 | `[16,256,56,56]÷[256,1,1]`（12.85M 索引） | 0,0,0,0 | `EQUIV_PASS` |
| E1-row-broadcast | 同上 | `[4096,4096]÷[1,4096]`（16.78M 索引） | 0,0,0,0 | `EQUIV_PASS` |
| E1-bias/row/scalar/a-side/全标量 | 同上 | 测试广播模式（5 组，含角色互换） | 0,0,0,0 | `EQUIV_PASS` |
| E2-fp16/fp32/bf16 | fp32 中转链（`vdiv` + `vcast(floor)`）vs golden `floor(a.float()/b.float()).to(dtype)`（§0.6 R4） | 每 dtype 66,377 例（随机正数 + 角点：±inf/NaN/±0/次正规/整数边界） | maxdiff=0、violations=0 | `EQUIV_PASS` |
| E3-fp16/bf16（负向对照） | 免 fp32 中转（原生半精度域）vs golden | 同上 | fp16：66 违反、maxdiff=1；bf16：430 违反、maxdiff=4 | 非等价（**证明 fp32 中转必需**，非采纳项） |

**OVERALL: `EQUIV_PASS`**（脚本随算子目录留存，Stage 2 维度 8 可重跑）。§3.1 公式拆解以「优化后公式 = 原式（`floor(a/b)`，fp32 域）+ §0.6 R4 可运行原语链 + §0.6 R1 索引表示」为唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素除法 `a/b` | 源 `T.Parallel(threads,npt)` 标量除 | `T.vdiv(a32, b32, q)`（同形，fp32） | ✅ | —（佐证：`docs/Tilelang.language/数学操作/T.vdiv.md` §2.1/§2.4；探针 `pure`/`vdivonly` 实测 fp32 可用） |
| 2 | 逐元素 floor | 源标量 `T.floor` | `T.vcast(q, yf, round_mode="floor")`（同形，fp32） | ✅ | —（佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.1/§2.2.1 `f32→f32 floor`；探针 `flooronly`/`pure` 实测精确） |
| 3 | 半精度→fp32 提升（fp16/bf16） | 源标量 `T.cast(a,"float32")` | `T.vcast(a_in, a32, round_mode="rint")` | ✅ | —（佐证：vcast §2.2.1 `f16→f32 rint`/`bf16→f32 rint`；探针 `E1`/`E2` 实测 bit-exact） |
| 4 | fp32→输出 cast（fp16/bf16） | 源标量 `T.Cast(a.dtype, ·)` | `T.vcast(yf, y_out, round_mode="rint")` | ✅ | —（佐证：vcast §2.2.1 `f32→f16`/`f32→bf16` 支持 rint；探针 `E1`/`E2` 实测 bit-exact） |
| 5 | 行内标量广播（末维 stride=0 操作数） | 源 stride-0 索引隐式广播 | `T.vbrc(a1/b1 → a_in/b_in)` | ✅ | —（佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；`examples/elementwise/vec_add_auto_brc.py` 先例；`bitwise_or` 同款先例） |
| 6 | 行 base divmod（`outer_a_base`/`outer_b_base`） | 源 `_compute_broadcast_offsets` 逐元素 divmod | **无**（无逐元素等价向量 API；行级标量） | ❌ | **block 级索引/任务映射计算**：每 task 仅 O(ndim) 次标量整数 divmod（ndim≤4），非逐元素热点（较源降 ~L 倍，L 达 10³–10⁶）；`T.arange`+divmod 向量化会引入逐元素标量物化（`CONST-arange-scalar-materialize` ~1ns/elem）反而更慢 |
| 7 | 尾块长度 `tail = T.min(block_size, L − inner_off)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：非整除时逐 task 不同边界；已采用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`）；padding 方案违反 host 禁改输入约束（ascend-constraints.md §4） |
| 8 | 核内 task 调度循环 `for i in T.serial(num_local_tasks)` | 循环 | 无（persistent 结构需循环推进 task） | ❌ | **tile 级调度循环**（§5.5 persistent 分核标准结构，core-split-strategy.md §1 要素③；`T.vcast.md` §2.4 示例同款 persistent 分发模式）；边界为编译期静态值，循环体一次一个完整 task（内部全向量化） |
| 9 | host 侧 `num_logical/num_kernels/chunks` 计算、block_size 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合 decision-tree §1） |

**向量化结论**：5 类逐元素计算点（#1 除法、#2 floor、#3 半精度提升、#4 输出 cast、#5 标量广播物化）**已全部向量化**，替代 API 均有 docs + 探针实测双佐证；保留 4 处标量/循环（#6 行 base 索引 / #7 动态边界 / #8 tile 调度循环 / #9 host 元数据），逐项理由见上表——§6 循环结构与本结论一致（kernel 内无任何逐元素标量循环）。**注**：与位运算族相比，本算子每 task 向量 op 数更多（fp32 2 条、fp16/bf16 5 条），是 §1.6.3 发射项裁决的动因。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（Op 层已把两输入展平为一维连续、输出回 broadcast shape），但**核内布局与 lane 映射是自由变量**。本节枚举候选并量化取舍。迁移任务：源 GPU 的 `threads×npt` 轴是**硬件组织的输入而非结论**（thread/warp 轴与 NPU Vector 轴不对应），须独立评估。**本算子有利条件**：`T.vdiv` **支持自动广播**（探针 F1/F2/F3，与 `T.vxor`/`T.vcmp` 不同）——广播路径的 operand 物化为可选优化而非强制。

**候选矩阵**（弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 展平一维（I/O 原生）+ innermost 维向量化 + operand 物化 | 展平 N 轴（`k=1`）/ coalesced 末维（`k>1`） | 无 | 基线：连续段 `T.copy`，访存 MTE2 最优；对齐 `block_size×E` 32B 整除；UB 4~7 buffer ≤ 192KB（§4.5）；广播维用 vbrc 物化；每 task 2（fp32）/5（fp16/bf16）条向量 op | ✅ **采纳**（主选） |
| 2 | **2-D 块化 tile** `[R_blk, L]` + **`T.vdiv` 自动广播**：a 连续载入 `[R_blk, L]`；b 末维 stride=1 时 b 行 `[(1,L)]` 载入一次 → `T.vdiv(a_tile[:, :], b_row[:, :], q[:, :])`（**广播**，免 vbrc）；`T.vcast(floor)` 同 2-D | 末维 L（+ 行轴 R_blk） | 无（a 连续） | 备选：**发射项按 R_blk 摊薄**（本算子广播类 workload 发射主导，§1.6.0）+ b 行重读 ÷ R_blk + **免 vbrc**（复用 §1.6.0 探针 F1 的列广播能力）；代价 UB `R_blk·L·(字节/元素)`（fp32 R_blk=2,L=4096 → 4 buffer×2×4096×4 = 128KB ×1.7 超限，须 R_blk=2 时减 buffer 或 R_blk=1 退化；见裁决计划） | ⚠️ **备选**（发射受限 / b 行重读场景；见裁决计划） |
| 3 | host 侧预广播物化 `b` 到输出 shape | 展平 N | host `expand→contiguous` | ❌ **负收益**：违反 host 禁改输入/新增物化的约束（ascend-constraints.md §4）；额外 GM 流量 `N·E`（b 物化）+ 内存；无收益 | ❌ |
| 4 | 逐元素标量 `T.Parallel` 直译 | 无向量轴 | 无 | ❌ **负收益**：NPU 无逐元素标量快路径（落 Scala 单元、无 128bit lane）；§1.6.2 #1–#4 已判定向量化 | ❌ |
| 5 | 一体化 `T.vfloordiv` 单指令布局 | 末维 | 无 | ❌ **不可实现**：探针实测 `T.vfloordiv`/`T.vfloor` 编译成功但运行期挂死（§9 R1） | ❌（原语不可用） |

**轴质量评分（主选 #1）**：整除性——末维 `L` 与平铺 `block_size` 均为向量宽度整数倍（fp32×4 / fp16×8）；尾 lane 浪费 = `(ceil(L/vw)·vw − L)/L`，manifest 广播末维 `L∈{3136, 4096}`（cnn-feat 3136 / row-broadcast 4096）对 fp32×4 均整除（0% 浪费；3136=4×784），smoke/hidden-state `L=N_total` 由 `block_size=4096` 整除；累加链——无累加（逐元素除法/floor，无跨步串行累加维，不阻碍向量化）；repack 代价——无（直接用 I/O 原生连续段）；UB 容量——4~7 buffer × block_size ≤ 192KB（§4.5）。

**源码轴对照（迁移任务要求）**：源 `T.Parallel(threads, npt)` 的隐式轴向是「连续 N 轴 + 每线程 npt 连续元素」，与主选 #1 的「连续段 + innermost 维向量化」一致；源的**广播路径**则退化为逐元素标量索引（无向量轴），NPU 重设计的核心即把它恢复为 innermost 维向量轴（§0.6 R1）。**注意**：源 GPU 的广播索引「隐式 strides」在向量指令层无广播要求（CUDA 标量运算），而 NPU `T.vdiv` **有**广播能力（探针），故 NPU 在广播路径上有「显式物化（vbrc）」与「直接复用 vdiv 广播」两条路——主选保留 vbrc 以复用同族已验证脚手架，备选 #2 利用 vdiv 广播免 vbrc。

**实验裁决模式（2-D 块化 tile 备选 #2）**：主选 #1 在广播类 shape（cnn-feat / row-broadcast）上每 task 仅处理一行、发射项主导（§1.6.0 估算 cnn-feat 155µs、row-broadcast 128µs；fp16/bf16 更高）且 row-broadcast 每 task 重读一行 b（16KB）。备选 #2 以 `[R_blk, L]` tile 把发射项与 b 行重读按 R_blk 摊薄、并借 `T.vdiv` 自动广播免去 vbrc。其收益依赖**未实证常数**（二维 vdiv 广播的每 op 开销、向量发射摊销律、b 行重读的 L2 吸收率；`CONST-vector-launch-overhead` 只给一维 op 口径），判定裕度不可忽略，故：
- **主选**：方案 #1（per-row，(block_size,) 1-D tile，进 Stage 3）。
- **备选**：方案 #2（2-D `[R_blk, L]` tile，仅 `sb==1`（b 行连续）的广播 shape；建议 fp32/fp16/bf16 先取 R_blk=2、L=4096，并按 UB 预算（§4.5）缩减 transit buffer（fp32 R_blk=2 建议只保留 a32/b32/q/yf 中活区间互斥者；若 ×1.7 膨胀超 192KB 则 R_blk=1 退化为方案 #1——容量否决）。备选结构：buffer 形状 `(R_blk, L)`、a 连续段载入、b 行 `(1,L)` 载入后由 `T.vdiv([R_blk,L],[1,L])` 广播、`T.vcast(floor)` 同 2-D、尾块按 `tail` 截断；分核三要素按 2-D task 粒度重算 `num_logical = ceildiv(R, R_blk) × ceildiv(L, block_size)`）。**`sb==0`（b 行内标量）场景仍走方案 #1**（b 无连续行可载，2-D 广播需 (R_blk,1) 列广播，收益不明）。
- **裁决计划**：Stage 4 在 row-broadcast（`[4096,4096]÷[1,4096]` fp32）与 cnn-feat（`[16,256,56,56]÷[256,1,1]` fp32）代表 shape A/B 实测方案 #1 vs #2；指标 = msprof op Task Duration + vec_ratio + mte2_active_bw；判定阈值 = 备选 latency < 主选 × 0.97 才翻转；按 shape 分派（`sb==1` 广播走 #2，其余走 #1）；**shape 特化工厂（lru_cache）评估**：wrapper 按 `(N_total, coalesced_shape, strides…)` 特化，天然支持按 shape 分派主选/备选，无全局二选一需求——结果回写 §1.6.3。

**布局决策结论**：选定方案 #1——核内布局 = `(block_size,)` UB buffer（fp32：a32/b32/q/yf；fp16/bf16 另加 a_in/b_in/y_out 与标量源 a1/b1），向量化轴 = **coalesced 末维**（`k=1` 时即展平 N 轴），repack = 无；广播索引经 §0.6 R1 行/内层分解 + operand 物化。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、`T.vdiv`/`T.vcast` 操作数即整 buffer）与 §6 循环结构；§4.5 UB 预算按此布局计算。方案 #2 作为 `sb==1` 广播 shape 备选 + 裁决计划随设计冻结进 Stage 4。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定；且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素除法 `T.vdiv` + floor `T.vcast` + 半精度 fp32 中转 cast + 标量广播物化 `T.vbrc`），无 matmul、无规约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + 2~5 向量 op + 一次搬出，广播路径外加 ≤2 vbrc） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 task 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | v-prefix（`vdiv`/`vcast`/`vbrc`）+ `T.copy` + `T.alloc_ub` 均为 Developer 可用形态 | `T.vdiv.md` §2.4 / `T.vcast.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态；探针均为 Developer 模式实测通过 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + `alloc_ub` + v-prefix，Developer）；`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/`（浮点 v-prefix 链，Developer） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2；非 persistent 混合算子，无 Developer/persistent+gemm 限制，本算子纯 Vector persistent）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix 要求操作数在 UB，`T.vdiv.md` §2.3 / `T.vcast.md` §2.3） |
| 计算方式 | v-prefix 向量 API（`T.vdiv` + `T.vcast` + `T.vbrc`），无 `T.Parallel` 逐元素标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式（本体不变）：`y = floor(a/b)`（fp32 域）；索引映射按 §0.6 R1（行 base + innermost 维向量化）；除法/floor 原语按 §0.6 R4（`T.vdiv` + `T.vcast(round_mode="floor")`，fp32 域，半精度需中转）。拆解与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(row,chunk) = decompose(task)`；`a_base`, `b_base` | 行/块标量分解（编译期 unrolled divmod，仅每 task 一次） |
| 2 | `a_in ← a[a_base : a_base+tail]`（末维 stride=1）或 `a_in ← vbrc(a[a_base])`（末维 stride=0） | a operand 物化为 (block_size,)（连续段或标量广播） |
| 3 | `b_in ←` 同步骤 2 | b operand 物化为 (block_size,)（同上） |
| 4 | `a32 = f32(a_in)`；`b32 = f32(b_in)`（仅半精度；fp32 免） | fp16/bf16 → fp32 无损提升（`rint`） |
| 5 | `q = a32 / b32` | 逐元素 fp32 向量除法（`T.vdiv`） |
| 6 | `yf = floor(q)` | 逐元素 fp32 向量 floor（`T.vcast(round_mode="floor")`） |
| 7 | `y_out = out_dtype(yf)`（仅半精度；fp32 免） | fp32 → 输出 dtype（`rint`；floor 后为整值，精确） |
| 8 | `y[flat0 : flat0+tail] ← y_out[0:tail]` | 结果写回（连续段，尾块截断） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| task 分派 | `task = i*num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local_tasks)` | num_kernels/num_local_tasks 编译期常量（§5.5） | Developer |
| 行 base | `Σ coord_i·stride_i` | Python 生成的 `T.floordiv`/`T.floormod` 展开链 | 编译期常量 divisors/strides | Developer |
| operand 搬入（连续段） | `x[a_base:a_base+tail] ↔ x_in[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（runtime 边界 OK；`T.copy.md`；`vec_add_1d.py` L34–L38） | Developer |
| 标量广播物化 | `x_in ← vbrc(x1)` | `T.vbrc(x1, x_in)` | src (1,)→dst (block_size,)（`T.vbrc.md` §2.2.2；`vec_add_auto_brc.py` 先例） | Developer |
| 半精度→fp32 提升 | `f32(x_in)` | `T.vcast(x_in, x32, round_mode="rint")` | fp16/bf16 输入（无损，`T.vcast.md` §2.2.1 `f16→f32 rint`/`bf16→f32 rint`） | Developer |
| 尾块长度 | `tail = min(block_size, L − inner_off)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md` | Developer |
| 逐元素除法 | `q = a32 / b32` | `T.vdiv(a32, b32, q)`（`npuir_div`） | 同形或自动广播；dtype fp16/fp32（本设计用 fp32）（`T.vdiv.md` §2.1/§2.2） | Developer |
| 逐元素 floor | `yf = floor(q)` | `T.vcast(q, yf, round_mode="floor")` | f32→f32（`T.vcast.md` §2.2.1）；探针实测精确 | Developer |
| fp32→输出 cast | `out_dtype(yf)` | `T.vcast(yf, y_out, round_mode="rint")` | f32→f16 / f32→bf16（`T.vcast.md` §2.2.1） | Developer |
| 整除向上取整 | `ceildiv(L, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板 §3.3 规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字，避免 false alarm）。工厂签名保留源契约（wrapper 已冻结调用形态），下半段为**示意**（真实实现由 JIT 参数编译期确定，`if sa==1` / `if half` 为 Python 层静态分派，不产生运行时分支）。block_size 相关分解**必须在 `@tilelang.jit` 函数体内**用被调用 block_size 计算（bitwise_and Stage 3 TRAP-jit-blocksize-decomposition；`T.ceildiv` 在 trace 期折叠）。

```python
def floor_divide_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides,
                            a_numel, b_numel, output_dtype=None, block_size=4096):
    # dtype ∈ {"float16", "bfloat16", "float32"}（整数在 Op 层拒绝）；输出同 dtype
    out_dtype = output_dtype or dtype
    half = dtype in ("float16", "bfloat16")          # 半精度需 fp32 中转（§0.6 R4）

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        vector_cores = NPUUtils.get().get_aicore_num() * 2   # 纯 Vector 翻倍；实查见 §5.5
        k = len(coalesced_shape)
        L = coalesced_shape[-1]                              # innermost 维
        outer = list(coalesced_shape[:-1])                   # 外层维
        R = math.prod(outer) if outer else 1
        sa, sb = a_strides[-1], b_strides[-1]                # 末维 stride ∈ {0,1}
        chunks = T.ceildiv(L, block_size)                    # trace 期折叠
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)         # persistent 分核（§5.5）
        num_local = T.ceildiv(num_logical, num_kernels)      # 静态边界

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                a32 = T.alloc_ub((block_size,), "float32")   # fp32 运算域
                b32 = T.alloc_ub((block_size,), "float32")
                q = T.alloc_ub((block_size,), "float32")     # 商
                yf = T.alloc_ub((block_size,), "float32")    # floor 结果（fp32）
                # 半精度中转缓冲（仅 half 时分配/使用；DCE 掉）
                a_in = T.alloc_ub((block_size,), dtype)
                b_in = T.alloc_ub((block_size,), dtype)
                y_out = T.alloc_ub((block_size,), out_dtype)
                # 标量广播源（仅末维 stride=0 时使用）
                a1 = T.alloc_ub((1,), dtype)
                b1 = T.alloc_ub((1,), dtype)
                a1f = T.alloc_ub((1,), "float32")
                b1f = T.alloc_ub((1,), "float32")
                for i in T.serial(num_local):                    # 核内串行，静态边界
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        a_base = _row_base(row, outer, a_strides[:-1]) + (inner_off if sa == 1 else 0)
                        b_base = _row_base(row, outer, b_strides[:-1]) + (inner_off if sb == 1 else 0)
                        tail = T.min(block_size, L - inner_off)
                        # --- 物化 a 为 (block_size,) fp32 ---
                        if sa == 1:
                            if half:
                                T.copy(a[a_base:a_base+tail], a_in[0:tail])
                                T.vcast(a_in, a32, round_mode="rint")
                            else:
                                T.copy(a[a_base:a_base+tail], a32[0:tail])
                        else:                                    # a 行内标量广播
                            T.copy(a[a_base:a_base+1], a1[0:1])
                            if half:
                                T.vcast(a1, a1f, round_mode="rint")
                                T.vbrc(a1f, a32)
                            else:
                                T.vbrc(a1, a32)
                        # --- 物化 b 为 (block_size,) fp32 ---
                        if sb == 1:
                            if half:
                                T.copy(b[b_base:b_base+tail], b_in[0:tail])
                                T.vcast(b_in, b32, round_mode="rint")
                            else:
                                T.copy(b[b_base:b_base+tail], b32[0:tail])
                        else:
                            T.copy(b[b_base:b_base+1], b1[0:1])
                            if half:
                                T.vcast(b1, b1f, round_mode="rint")
                                T.vbrc(b1f, b32)
                            else:
                                T.vbrc(b1, b32)
                        # --- 除法 + floor + 落盘 ---
                        T.vdiv(a32, b32, q)
                        T.vcast(q, yf, round_mode="floor")
                        flat0 = row * L + inner_off
                        if half:
                            T.vcast(yf, y_out, round_mode="rint")
                            T.copy(y_out[0:tail], y[flat0 : flat0 + tail])
                        else:
                            T.copy(yf[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel
```

> 实现注记：① `_row_base` 为 Python 层按 `outer`/strides 生成的 `T.floordiv`/`T.floormod` 展开链（因子全为编译期常量）；`k==1` 时无外层维，`a_base=inner_off`、`b_base=inner_off`。② `sa==0` 且 `sb==0` 仅在全标量（`L==1`）出现，此时两路都走 `vbrc`。③ **fp32 输入免恒等 cast**（`T.vcast(f32→f32, rint)` 会产生非法结果，探针 C1 得 NaN）——`if half` 分支保证 fp32 直接 `T.copy`→`vdiv`→`vcast(floor)`。④ v-prefix 整 buffer 操作在尾块（`tail<block_size`）时对 `[tail:]` 段垃圾一并计算，但仅 `[0:tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例同款）。⑤ `T.vbrc` 输入为 (1,)，输出为 (block_size,)（dst 必须 >1，避 `TRAP-vbrc-same-shape-empty-broadcast`）。⑥ block_size 分解在 jit 函数体内计算（见上）。⑦ **切片写 offset**：`T.copy(a[a_base:a_base+tail], a_in[0:tail])` 的 src 可带运行时 offset，dst 从 0 起始（`vec_add_1d.py` L34–L38 已验证）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.vdiv`（`npuir_div`） | `docs/Tilelang.language/数学操作/T.vdiv.md` §2.1/§2.2/§2.4；`tilelang/language/customize_npuir.py` L158–L169 | ✅ **探针实测**：fp32 同形 bit-correct（除零角点见 §9）；**自动广播** `[M,N]÷[1,N]`/`[M,N]÷[M,1]`/`[M,N]÷scalar` 全部 bit-correct（探针 F1/F2/F3）；fp16/fp32 支持（文档 §2.2.1），bf16 **不支持**（故 bf16 经 fp32 中转） |
| `T.vcast`（`npuir_cast`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.1/§2.2.1/§2.4 | ✅ **探针实测**：`f32→f32 floor`（向量 floor）精确（`flooronly`/`pure`）；`f16→f32 rint`、`bf16→f32 rint`、`f32→f16`、`f32→bf16` 中转 bit-exact（E1/E2）；**f32→f32 rint 恒等 cast 产生 NaN（探针 C1）→ fp32 路径禁用心跳 cast** |
| `T.vbrc`（`npuir_brc`） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2/§2.3；`examples/elementwise/vec_add_auto_brc.py` 先例 | ✅ 文档支持 fp16/fp32/bf16；(1,)→(block_size,) 标量广播为本族已验证形态（`bitwise_or` 同款）；**本设计主选沿用**（§1.6.3 明确 vdiv 广播备选可免 vbrc，但未单独探针 (1,)→(block_size,) 的 vdiv 形态，保守保留 vbrc） |
| **`T.vfloordiv`（`npuir_floordiv`）** | `tilelang/language/customize_npuir.py` L425–L436（**无 docs 条目**） | ❌ **探针实测不可用**：编译成功但**运行期 Vector 核挂死**（同形 fp32/fp16/bf16、切片形、`alloc_shared` 形全挂）→ §9 R1 能力缺口 |
| **`T.vfloor`（`npuir_floor`）** | `tilelang/language/customize_npuir.py` L412–L422（**无 docs 条目**） | ❌ **探针实测不可用**：编译成功但运行期挂死（同形 fp32） |
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vdiv.md` §2.4 | ✅（examples 先例） |
| `T.alloc_ub(shape, dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`vec_add_1d.py` L28–L30 | ✅ |
| `T.copy`（gm↔ub、切片、runtime 边界） | `docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` L34–L38 | ✅ |
| `T.min` / `T.ceildiv` | `docs/Tilelang.language/比较操作/T.min.md`；`docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.serial` + `task = i·K + cid` + `if` | `T.vcast.md` §2.4 示例（persistent 分发模式）；`bitwise_or_fwd_kernel.py` L343–L345 | ✅ |
| `@tilelang.jit(out_idx=[2], target="npuir")` | 源 `out_idx=[2]` 契约 + `example_elementwise_add.py` L13 | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 grid；NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg / `T.Parallel(threads,npt)`） | Yes | `is_npu=True` + v-prefix 取代（§0.5 舍弃项 / §0.6 R2） |
| GEMM 非整除 | No（无 GEMM；非 GEMM 的行/尾块由 §6.4 处理） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 1–4096 vs 物理 Vector 核 40） | §5.5 persistent 分核 |
| Host 侧输入操作约束 | Yes（Op 层展平输入；b 广播不做 host 物化） | 展平在 Op 层（既有契约）；kernel 内 host 侧零数据操作，仅元数据 → 符合 ascend-constraints.md §4 |
| **一体化 `T.vfloordiv`/`T.vfloor` 运行期挂死** | Yes（原语本应匹配语义） | §0.6 R4：改用 `T.vdiv` + `T.vcast(round_mode="floor")`；§9 R1 登记能力缺口 |
| **`T.vcast(f32→f32, rint)` 恒等 cast 产生 NaN** | Yes（fp32 路径若误加恒等 cast） | §3.3 注记 ③：fp32 路径不做恒等 cast（探针 C1） |
| 整数/布尔 dtype | Yes（源拒绝） | Op 层 `SUPPORTED_DTYPES` 拒绝（对照 manifest/signature） |
| 除零 / 非正除数的角点 | 部分（测试/基准均为正数） | §9 R6：记录为边界风险，L0 覆盖角点；不做特殊处理（与源一致：`±inf` 传播） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R2 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1–#4 |
| 除法本体 | 标量 `a_f32 / b_f32` | `T.vdiv`（fp32 向量） | §0.6 R4 |
| floor 本体 | 标量 `T.floor` | `T.vcast(round_mode="floor")`（fp32 向量） | §0.6 R4（一体化 `T.vfloordiv` 挂死） |
| 半精度提升/cast | 标量 `T.cast` / `T.Cast` | `T.vcast(round_mode="rint")` | §0.6 R4 |
| 广播索引 | 逐元素 unrolled divmod + 标量 gather（隐式 strides） | 行级 unrolled divmod + innermost 维向量化 | §0.6 R1 |
| 广播语义承载 | 标量域（向量指令无广播约束） | `T.vdiv` 有自动广播；主选仍用 `T.vbrc` 物化（备选 #2 利用广播） | §0.6 R1 / §1.6.3 |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(num_logical, 40)` | §0.6 R2 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[2])` | `@tilelang.jit(out_idx=[2], target="npuir")` | K1 适配 |
| 工厂参数 | `(…, threads=256, num_per_thread=8)` | `(…)`；callable(`block_size`) | wrapper K11（接口契约冻结） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/bitwise_or/bitwise_or_kernel/bitwise_or_fwd_kernel.py` | 高度相似（同族 elementwise 广播迁移先例，含完整 NPU 实现 + L0/L1 测试） | §0 迁移分析组织、persistent 分核、行/内层广播 + `vbrc` 物化、`_row_base` helper、block_size 分解在 jit 体内、UB 预算口径（**差异：本算子 float-only、同一化输出、除法+floor 两条向量 op、`T.vdiv` 支持自动广播**） |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高度相似（浮点 v-prefix 链 + fp32 中转先例） | **半精度 fp32 中转精度论证**（bf16/fp16 无原生向量算术或舍入路径差异 → `vcast(rint)→fp32→链→vcast(rint)`）、UB 预算按中转 buffer 计、`os.environ["TILELANG_ASCEND_MODE"]="Developer"` |
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail = T.min(...)` 尾块模式、runtime 边界 `T.copy`、v-prefix 用法 |
| `examples/elementwise/vec_add_auto_brc.py` | 中度相似（v-prefix 自动广播用法） | `T.vadd(A[:, :], B[0, :], C[:, :])` 的 rank-2 自动广播形态；印证 `T.vdiv` 广播候选（§1.6.3 #2） |
| `examples/bitwise_xor/bitwise_xor_fwd_kernel/DESIGN.md` + `examples/eq/eq_fwd_kernel/DESIGN.md` | 高度相似（同族迁移设计模板 + 已踩坑） | §0 三问解读/耦合性四态/§1.6 调研优化结构、persistent 分核、同形物化、L0 测试结构；**差异：本算子为浮点除法+floor，无 bool/uint8 承载问题，且 `T.vdiv` 有自动广播（与 vxor/vcmp 相反）** |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `a` | `(a_numel,)` | float16 / bfloat16 / float32 | Op 层已展平的 1-D 连续张量；广播时 `a_numel ≤ N_total` |
| `b` | `(b_numel,)` | same_as(a) | 同上 |

> **dtype 覆盖（对照 manifest signature）**：manifest `FloorDivideFwdOp` inputs 声明 `float16 | bfloat16 | float32`——**全部**在 §0.6 R4 中给出路径（fp16/bf16 经 fp32 中转；fp32 直算），无遗漏；整数/布尔 dtype 在 Op 层拒绝（对照 `test_floor_divide_rejects_integer_dtype`）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N_total,)` | **same_as(a)**（float16/bf16/fp32） | `out_idx=[2]`；Op 层 `reshape(out_shape)` 还原广播形状（无 dtype view，与 eq 不同） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a32` / `b32` | `(block_size,)` | float32 | UB | fp32 运算域 operand（连续段直载或经半精度提升/vbrc） |
| `q` | `(block_size,)` | float32 | UB | `T.vdiv` 的商 |
| `yf` | `(block_size,)` | float32 | UB | floor 结果（fp32） |
| `a_in` / `b_in` | `(block_size,)` | 输入 dtype | UB | **仅 half 路径**：连续段载入的半精度 operand |
| `y_out` | `(block_size,)` | 输出 dtype | UB | **仅 half 路径**：fp32→输出 cast 结果 |
| `a1` / `b1` / `a1f` / `b1f`（仅末维 stride=0 时用） | `(1,)` | 输入 dtype / float32 | UB | 标量广播源（`vbrc` 输入） |

> buffer 复用：`a_in`/`b_in`/`y_out` 仅 half 路径分配（Python 编译期分支，否则 DCE）；`q`/`yf` 活区间相邻，Stage 3 视分配情况可合并（注意：避免 f32→f32 恒等 cast 的 NaN 问题，`q`→`yf` 为 floor 变换非恒等，允许分开）。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 task，half 路径；fp32 免中转 buffer）：
  GM[a] ──T.copy(连续段或 1 元素)──> UB[a_in/a1] ──[half: vcast(rint)→f32]──> UB[a32] ─┐
  GM[b] ──T.copy(连续段或 1 元素)──> UB[b_in/b1] ──[half: vcast(rint)→f32]──> UB[b32] ─┼─ T.vdiv ──> UB[q:f32]
                                                                                     │  ──vcast(floor)──> UB[yf:f32]
                                                                                     │  ──[half: vcast(rint)→out]──> UB[y_out] ──T.copy(连续段)──> GM[y]
（末维 stride=0 的 operand 先经 T.vbrc 物化；全程 UB 内，无 L1/L0 参与，无中间 GM 往返）
行 base 标量：仅每 task 一次 divmod（不在 GM 数据路径上）
```

### 4.5 UB 内存预算

block_size = 4096（推荐，见 §5.2）。按各 dtype 路径的活跃 buffer 数量与字节/元素（`×1.7` auto-multi-buffer 膨胀上界，`CONST-capacity-910B2C`）：

| dtype 路径 | 活跃 buffer（字节/元素） | 手工合计 @4096 | ×1.7 膨胀 | 目标 UB | 余量 |
|-----------|------------------------|----------------|-----------|---------|------|
| float32 | a32(4)+b32(4)+q(4)+yf(4) = **16** | 64 KB | ~109 KB | 192 KB | ✓ |
| float16 / bfloat16 | a_in(2)+b_in(2)+a32(4)+b32(4)+q(4)+yf(4)+y_out(2) = **22** | 88 KB | ~150 KB | 192 KB | ✓ |

> 膨胀系数引用 `CONST-capacity-910B2C`（auto-multi-buffer ~1.7x；该条目另有「显式 alloc 结构 ~1.12x / 反例 0.981x」的结构依赖记录，本设计取保守 1.7x 上界）。block_size 上限（`floor(192KB / (bytes_per_elem × 1.7))`，忽略 `(1,)` 标量 buffer）：fp32 ≤ 7228、fp16/bf16 ≤ 5257——推荐值 4096 对全部 dtype 达标。block_size 由 wrapper config 传入（默认 1024）；本设计推荐 4096（理由见 §5.2；Stage 4 可扫，受此上限约束）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 无 TIR 动态符号——工厂闭包按 `(N_total, dtype, coalesced_shape, strides, a_numel, b_numel)` **编译期特化**（源同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例触发一次 JIT 特化；kernel 边界全静态 |

> `L`、`R`、`chunks`、`num_logical`、`num_kernels`、`num_local` 全为 Python int / trace 期常量；kernel 内唯一运行时标量为 `tail`（1 次/task）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
# out_idx=[2]：main(a, b, y) 的 y（index 2）为输出，承接源算子契约（_floor_divide_fwd_kernels.py L50/L75）
# target="npuir"：K1 适配（GPU 版无 target；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素除法（1 条 `T.vdiv`）+ floor（1 条 `T.vcast`）+ 半精度 fp32 中转（最多 3 条 `T.vcast`）+ 标量广播物化（`T.vbrc`），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（decision-tree §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_size = 4096   # 推荐；选择理由：
# ① innermost 维单块覆盖：manifest 广播行 L ∈ {3136, 4096}（cnn-feat 3136 / row-broadcast 4096）
#    对 block_size=4096 均为 chunks=1（L≤4096）——消除行内二次分块，减少发射次数
# ② 尾轴 32B 对齐：4096×{2,4}B = 8K/16K，均 32 整除 ✓（fp16/bf16 2B、fp32 4B）
# ③ 向量宽度整除：fp32×4 / fp16×8 均整除 4096 → 尾 lane 浪费 0%
# ④ UB 预算：全 dtype 路径 ≤150KB（fp16/bf16）< 192KB ✓（§4.5）
# ⑤ 同形路径（L=N_total 极大）由 chunks 分块，block_size 即 tile 粒度
# ⑥ 与同族迁移一致（4096），便于复用与对比；更大 block（8192）可摊薄发射项
#    （§1.6.0 广播类发射主导），但受 §4.5 上限约束（fp16/bf16 ≤5257 → 8192 超限，
#    fp32 ≤7228），交 Stage 4 按 dtype 扫
L = coalesced_shape[-1]
R = prod(coalesced_shape[:-1]) if k > 1 else 1
chunks = ceildiv(L, block_size)
num_logical = R * chunks          # 逻辑 task 数（一维划分）
```

一维划分（无 block_M/block_N 之分）——本算子为逐元素/广播，GEMM 类分形/非整除策略不适用。**分类倾向**：`k==1`（同形 / 全标量广播）退化为展平 N 轴分块；`k>1`（含广播）走行/内层分块。**2-D tile 备选（§1.6.3 #2）** 另按 `num_logical = ceildiv(R, R_blk) × ceildiv(L, block_size)` 划分（随裁决计划进 Stage 4，不改变主选 §3.3/§6）。

### 5.3 约束分析

- **对齐约束**: `block_size×E` = 8K/16K（32B 的整数倍）✓；末维向量维整除（尾 lane 0%）✓；尾块 `tail < block_size` 段长度非 32B 倍数时的拷贝行为依赖 `vec_add_1d.py` 先例（风险 §9 R7，L0-6 覆盖）
- **UB 容量**: 见 §4.5，全 dtype 路径 ×1.7 膨胀后 ≤150KB ≤ 192KB ✓
- **L0/L1 容量**: 不适用（纯 Vector，不使用 L0/L1）
- **分形限制**: 不适用（无 Cube）
- **原语约束**: 一体化 `T.vfloordiv`/`T.vfloor` 运行期挂死（不可用）；`T.vcast(f32→f32, rint)` 恒等 cast 非法（fp32 路径免）
- **fp32 域约束**: 半精度除法/floor 必须在 fp32 域（E3 证明免中转非等价）

### 5.4 注意事项

- **非整除处理**：尾块 `tail = T.min(block_size, L − inner_off)` 截断（§6.4）；manifest 各 workload 的 `L` 对 block_size=4096 要么整除（L=4096 / L=N_total 由 chunks 整除）要么单块尾段（L=3136），尾块路径服务任意广播形状
- **`chunks>1` 的行**：`L>4096`（如 hidden-state L=8.39M → chunks=2048）→ 同行多 chunk 共享同一 `a_base` 外层项、`inner_off` 不同（Python/常量算）
- **广播物化**：末维 stride=0 的 operand 每 task 走 1 次 vbrc（§0.6 R1）；`T.vdiv` 本身支持广播（备选 #2 可免，§1.6.3）
- **发射项**：fp32 每 task 2 条向量 op、fp16/bf16 每 task 5 条（+可能的 vbrc），广播类 workload 发射项主导（§1.6.0）→ §1.6.3 备选 #2 + Stage 4 裁决

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
| 元素级（除法） | 向量化 | `T.vdiv`（整 buffer 一条向量指令） | §1.6.2 #1；无 `T.Parallel` 逐元素循环（与源差异：§0.6 R2） |
| 元素级（floor） | 向量化 | `T.vcast(round_mode="floor")`（整 buffer） | §1.6.2 #2 |
| 元素级（半精度提升/cast） | 向量化 | `T.vcast`（整 buffer，最多 3 条） | §1.6.2 #3/#4 |
| 标量广播 | 向量化 | `T.vbrc`（1 元素 → block_size） | §1.6.2 #5（末维 stride=0 时） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 核间：persistent ≤40 核
    a32, b32, q, yf, a_in, b_in, y_out, a1, b1, a1f, b1f = T.alloc_ub(...)  # §3.3 完整版
    for i in T.serial(num_local):                            # 核内：静态边界串行
        task = i * num_kernels + cid
        if task < num_logical:
            row = task // chunks; inner_off = (task % chunks) * block_size
            a_base, b_base = <行 base 标量>; tail = T.min(block_size, L - inner_off)
            # 物化（连续段 / vbrc 标量广播；half 经 vcast→f32）→ vdiv → vcast(floor) → （half: vcast→out）→ 写回
            ...
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：单阶段 load→compute→store 算子（每 task 一次进出），无 K 维迭代累加结构可供双缓冲；逐元素除法/floor 计算量极小、访存主导，`auto multi-buffer` 可将向量 op 隐藏于 MTE2 窗口内（`CONST-copy-floor-method`：≥16M 档计算链被完全隐藏）。但**本算子每 task 向量 op 数（fp32 2 条、fp16/bf16 5 条）高于位运算族（1）**，发射项在广播类 workload 可能主导（§1.6.0）——多 task 双缓冲与 2-D tile（§1.6.3 #2）属参数/结构级调优，交 Stage 4 按 msprof 评估。

### 6.4 尾块处理

`L % block_size ≠ 0` 时（任意广播形状；block_size=4096 下 manifest 仅 L=3136 触发单块尾段）：

```python
inner_off = (task % chunks) * block_size
tail = T.min(block_size, L - inner_off)      # 本块有效长度
T.copy(a[a_base : a_base + tail], a_in[0:tail])    # 零起点 dst 切片（规避 UB 非零起点对齐）
...（v-prefix 整 buffer 操作；UB 中 [tail:] 陈旧段不落 GM）
T.copy(y_out[0:tail], y[flat0 : flat0 + tail])
```

模式来源：`examples/elementwise/vec_add_1d.py` L31–L38（官方已验证形态）。不采用 host 侧 padding（违反 host 禁改输入约束，ascend-constraints.md §4）；不采用 Kernel 动态 block（N/L 已编译期特化，`tail` 标量一次/task 开销可忽略）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

无手动同步点。依据：① 每 task 计算完全独立、无核间数据依赖（无 workspace、无跨核归约）→ 不需要 `T.sync_block_set/wait`；② task 内 `T.copy → T.vcast/T.vbrc/T.vdiv/T.vcast → T.copy` 的顺序依赖由 Developer 模式自动同步保证（`T.vdiv.md`/`T.vcast.md`/`T.vbrc.md` 均为无显式同步示例）；③ persistent 核内 `T.serial` 顺序执行天然串行；④ `if task < num_logical` 无跨 task 数据交换。**CV 融合设计不适用**（纯 Vector，无 Cube 参与，无跨引擎流水）。

### 7.3 pass_configs 配置

无特殊 pass 配置（默认管线）。环境变量：`TILELANG_ASCEND_MODE=Developer`（§2.3）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现 `test_floor_divide.py` 的 `ref_program`），**不复刻** §0.6 的 NPU 算法（不复用行/内层索引、vdiv/vcast 链、persistent 结构、vbrc 物化）——保证验证独立性。

```python
import torch

def golden_floor_divide(a, b):
    """Floor-divide reference (source-operator semantics, direct broadcast).

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_floor_divide.py:
    ref_program = torch.floor(a.float() / b.float()).to(a.dtype)）；等价于
    torch.floor_divide / torch.div(rounding_mode="floor")。输入为原始 broadcast
    形状（非展平），输出 = broadcast_shapes(a.shape, b.shape)，dtype same_as(a)。
    fp32 域除+floor 后 cast 回原生 dtype——即源算子的精度语义（§0.1）。
    """
    return torch.floor(a.detach().cpu().float() / b.detach().cpu().float()).to(a.dtype)
```

### 8.2 精度标准

除法与 floor 在 fp32 域执行、结果 cast 回原生 dtype——本设计链在探针与 `verify_equiv.py` E2 中与 golden **逐位一致**：

| dtype | 比对方式 | 说明 |
|-------|----------|------|
| float32 | `torch.equal`（exact）为主；下限 `atol=1e-5, rtol=1e-5` | 源测试 fp32 用 `atol=1e-5`（`test_floor_divide_edge_case`）；本设计链实测 exact |
| float16 / bfloat16 | `atol=1.0, rtol=0.0`（源测试契约） | 源测试对 non-fp32 用 `atol=1.0`（"fp16 floor rounding tolerance"）；本设计 fp32 中转链实测与 golden **bit-exact**（E2），实际远优于容差 |
| 输出 shape | `out.shape == broadcast_shapes(a,b)` | Op 层还原，测试断言 |
| 输出 dtype | `out.dtype == a.dtype` | same_as 契约 |

来源：`examples/TileOPs/tests/ops/test_floor_divide.py`（`FloorDivideTest.ref_program`、`atol = 1.0 if dtype != torch.float32 else 1e-5`、`test_floor_divide_broadcast` 用 `assert_close(atol=1.0)`）；manifest `outputs.output.dtype: "same_as(input)"`、`shape_rules`。

### 8.3 L0 门槛测试计划

> L0 为 Stage 3 开发精度门槛（golden 对比必须全过）；完整 L1/L2/Boundary 分层套件由 `tilelang-op-develop` 展开。

| # | 用例 | 输入 | 通过标准 |
|---|------|------|----------|
| L0-1 | smoke 同形全浮点 dtype | `[4096]÷[4096]`：fp16/bf16/fp32，正随机数 | vs golden，atol 按 §8.2（实测 exact） |
| L0-2 | 大规模同形（各 dtype） | `[2048,4096]÷[2048,4096]` fp32/fp16（hidden-state 代表，chunks=2048>1） | vs golden，§8.2 |
| L0-3 | **广播三模式** | `[16,256,56,56]÷[256,1,1]`（cnn-feat 通道广播，sb=0）、`[4096,4096]÷[1,4096]`（row-broadcast，sb=1）、`[2,64,128]÷[1,1,128]`（bias）fp32/fp16 | vs golden，§8.2 + shape 断言 |
| L0-4 | **a 侧广播** | `[1,4096]÷[4096,4096]`、`[2,64,128]÷[2,64,1]`（角色互换，覆盖 `sa=0`） | vs golden，§8.2 |
| L0-5 | 标量广播 / 全标量 | `[64,128]÷[1,1]`、`[1]÷[1]`（`L=1` 全标量退化，避 `TRAP-vbrc-same-shape`） | vs golden，§8.2 |
| L0-6 | 尾块非整除 | `L` 非 `block_size` 整数倍：`[1000]÷[1000]`、`[3136]÷[3136]`、`[4097]÷[4097]`（fp32） | vs golden，§8.2（覆盖 §6.4 / 风险 §9 R7） |
| L0-7 | **整数 dtype 拒绝** | int8/int16/int32/int64/uint8/bool | Op 层 `ValueError`（`does not support dtype`），与源一致 |
| L0-8 | **fp32 免恒等 cast 回归** | `[4096]÷[4096]` fp32（验证 §3.3 注记③：fp32 路径不插入 f32→f32 rint cast） | vs golden，exact（探针 C1 表明误加恒等 cast 得 NaN） |
| L0-9 | persistent 边界 | `N/L` 使 `num_logical ∈ {39, 40, 41}`（核数切换边界，block_size 使能） | vs golden，§8.2 + 无越界 |
| L0-10 | **浮点角点语义** | fp32/fp16：NaN÷x、±Inf÷x、x÷±Inf、±0.0、次正规数、整数边界商（如 6.0/2.0、2.9999/1.0） | vs golden（IEEE：`floor(±inf)=±inf`、`floor(NaN)=NaN`），§8.2（风险 §9 R6） |
| L0-11 | **vdiv 广播能力回归** | `[M,N]÷[1,N]`、`[M,N]÷[M,1]`、`[M,N]÷[1,1]`（验证 §0.6 R1 物化正确；同时记录 vdiv 广播可用性，供 §1.6.3 备选 #2） | vs golden，§8.2 |
| L0-12 | 除数边界 | 小正除数（`b∈[0.1,1.1)` 同 bench `positive=True`）、`b=1.0`（y=a 的 floor） | vs golden，§8.2 |

**L0 测试要点**：输入经 `.npu()` 放置；广播用例额外断言 `out.shape == broadcast_shapes(a.shape, b.shape)`；L0-8 覆盖 fp32 恒等 cast 陷阱；L0-10 覆盖浮点角点（设计期探针用正随机数，角点由 L0 闭合，风险 §9 R6）；L0-11 覆盖 `T.vdiv` 广播能力（本算子相对 vxor/vcmp 的有利差异）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束/风险 | 影响 | 缓解方案 |
|---|-----------|------|----------|
| R1 | **`T.vfloordiv`/`T.vfloor` 运行期挂死**（编译成功但 Vector 核挂起；探针实测；无 docs 条目） | 语义最匹配的一体化原语不可用 | §0.6 R4：改用 `T.vdiv` + `T.vcast(round_mode="floor")`；登记能力缺口（capability-gaps，缺文档证据 + 探针复现）；Stage 4 若工具链升级可复测 |
| R2 | **`T.vcast(f32→f32, rint)` 恒等 cast 产生 NaN**（探针 C1） | fp32 路径若插入恒等 cast 得 NaN | §3.3 注记③：fp32 路径直接 `T.copy`→`vdiv`→`vcast(floor)`，不做提升/回落 cast；L0-8 回归 |
| R3 | **半精度除法/floor 必须在 fp32 域** | 免中转则与 golden 不符（E3：fp16 66 / bf16 430 处偏差） | §0.6 R4：fp16/bf16 经 fp32 中转（`vcast(rint)` 无损提升 + 单次回落）；`verify_equiv.py` E3 机器验证 |
| R4 | **发射项主导**（§1.6.0 估算 cnn-feat 155µs / row-broadcast 128µs；fp16/bf16 更高，每 task 5 op） | 广播类 workload 可能发射受限 | §1.6.3 备选 #2（2-D `[R_blk,L]` tile + vdiv 广播免 vbrc）+ Stage 4 实验裁决；block_size 加大（受 §4.5 上限：fp16/bf16 ≤5257、fp32 ≤7228） |
| R5 | **row-broadcast 的 b 行重读**（主选每 task 重读一行 b，16KB×4096≈64MB） | 未优化时 b 读流量计入上限口径 ≈188µs（vs HBM 口径 128µs） | b 仅 16KB，预期 L2 驻留；§1.6.3 已列备选方案 #2 + Stage 4 实验裁决 |
| R6 | **除零 / 非正除数角点**：除零 → `±inf`（floor 仍 ±inf）；负除数 → floor 向 −inf | 与源算子行为一致（源同款），但测试/基准仅用正数 | 设计期探针用正随机数；L0-10 覆盖 NaN/±inf/±0 角点；不做特殊处理（保持源语义） |
| R7 | **尾块非 32B 对齐拷贝** | T.copy 报错或数据错乱 | 采用 `vec_add_1d.py` 零起点 dst 切片先例；L0-6 强制覆盖 |
| R8 | **block_size 分解一致性**：工厂 `callable(block_size)` 形态下，`chunks/num_logical/num_kernels/num_local` 若在工厂构建期用默认 block_size 计算、而 kernel 体用被调用值，则非默认 block_size 静默半错 | 非默认 block_size 下部分元素未写，形似精度 bug | 全部分解放在 `@tilelang.jit` 函数体内、用该函数 block_size 计算（§3.3 注记⑥；bitwise_and TRAP-jit-blocksize-decomposition） |
| R9 | **常数为 stale / 跨核数外推**：`CONST-mte2-degradation` 被 kb_stale_check 标 stale；本机 20 核 ≠ 常数 48 核 | 设计期流量项估算可能失准 | 估算行已标注；Stage 4 baseline 后按 hardware-cost-model §3 回填「设计估算 vs 实测」偏差行 |
| R10 | **bf16 无原生 vdiv 支持**（文档 §2.2.1 bf16 ×） | bf16 不能直接 vdiv | §0.6 R4：bf16 统一经 fp32 中转（本就必需，R3）——无额外代价 |
| R11 | **极小 L 效率退化**（一般形状）：若某广播形状 innermost 维很小且外层 R 很大（如 `[65536,1]÷[65536,8]`：L=8, R=65536） | 每 task 向量过短、task 数过大 | manifest/测试均不触发；作为已知局限记录（不阻塞）；如出现可在 Op 层合并 broadcast 维度后重排 |
| R12 | **`T.vbrc` 的 (1,)→(block_size,) 形态未单独探针**（本设计主选沿用，与 `bitwise_or` 同款） | 若该形态异常则标量广播路径失败 | L0-5 强制覆盖标量广播；备选 #2（vdiv 直接广播）可绕开 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 直接用 `T.vfloordiv`/`T.vfloor` | 以为存在正好匹配的原语 | 编译成功但运行期 Vector 核挂死 | 改用 `T.vdiv` + `T.vcast(round_mode="floor")`（§0.6 R4） |
| fp32 路径插入 `T.vcast(x, y, "rint")` 恒等 cast | 未按 dtype 分支，统一走中转链 | 输出 NaN（探针 C1） | `if half:` 分支；fp32 免恒等 cast（§3.3 注记③） |
| 半精度域直接除+floor（无 fp32 中转） | 图省 op | 与 golden 不符（E3 偏差） | fp16/bf16 经 fp32 中转（§0.6 R4） |
| 除法转乘倒数 `a*(1/b)` | 照搬通用优化 | floor 结果在整数边界错 ±1 | 禁用（§1.6.1 候选 1 非等价） |
| 合并 floor 与输出 cast 单条 `vcast(q→out, floor)` | 图省一条 cast | 大商舍入顺序改变，与 golden 不符 | 分开 floor（f32）+ cast（§1.6.1 候选 2） |
| 用 bf16 直接 `T.vdiv` | 未查 dtype 矩阵 | vdiv 不支持 bf16 | 经 fp32 中转（R10） |
| 把行 base divmod 写成逐元素 | 把 `_compute_broadcast_offsets` 直接搬进 kernel 元素循环 | 标量热点，性能崩溃 | 仅在 task 级（每 task 一次）做，见 §0.6 R1 |
| `num_kernels` 误用逻辑核数 | 直接照抄逻辑核数作 grid | 数千核串行调度、启动开销剧增 | §5.5 方案为准（`min(num_logical, 40)`） |
| block_size 相关分解写在工厂构建期 | wrapper `callable(block_size)` 契约 | 非默认 block_size 静默半错 | 分解放 jit 函数体内（R8） |
| block_size 超 UB 上限 | config 传入过大 | UB 溢出编译失败 | §4.5 上界（fp32 ≤7228 / fp16/bf16 ≤5257，1.7x 膨胀口径） |
| 用整数/布尔 dtype 声明张量 | 误以为支持 | Op 层拒绝；kernel 声明无效 | Op 层 `SUPPORTED_DTYPES` 拒绝（L0-7） |

### 9.3 特殊场景处理

- **同形（无广播）**：`k==1, R==1, sa==sb==1` → 展平连续段路径（与源同形快路径一致）
- **标量广播**（`[64,128]÷[1,1]`）：coalesced `[(8192,1,0)]`，`k=1, sa=1, sb=0` → b 单元素 + `vbrc`
- **a 侧广播**（`[1,4096]÷[4096,4096]`）：`sa=0` 分支（symmetry）
- **全标量**（0-dim×0-dim）：coalesced 兜底 `(1,0,0)`，`L=1`，两路 vbrc（dst=block_size>1）
- **非整除 N/L**：`tail` 动态裁剪（§6.4）；manifest 全整除不触发（L 对 4096 整除或单块）
- **N=0 / 空维度**：Op 层 `prod()==1` 兜底或上层拦截，kernel 不触达（与源一致）
- **混合 dtype 输入**：Op 层 `forward` 已拒绝（dtype 一致性校验，`test_floor_divide_rejects_runtime_dtype_mismatch`）
- **整数/bool/fp8/complex dtype**：manifest 不含；Op 层 `SUPPORTED_DTYPES` 拒绝（`test_floor_divide_rejects_integer_dtype`）

---

## 10. 交付清单

### 10.1 目录结构

```
examples/floor_divide/floor_divide_fwd_kernel/
├── DESIGN.md                         # 本设计文档
├── verify_equiv.py                   # 等价性机器验证（E1 索引映射等价 + E2 fp32 中转链等价 + E3 免中转非等价对照）
├── history_version/
│   ├── design_probe_floordiv.py          # D-3 探针 1（vfloordiv/vfloor 运行期 + vdiv/vcast 链 dtype/同形）
│   └── design_probe_floordiv_bcast.py    # D-3 探针 2（bf16 链 + vdiv 广播 col/row/scalar）
└── RETROSPECTIVE.md                  # Stage 1 复盘（skill Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | E1/E2 全 `EQUIV_PASS`（OVERALL `EQUIV_PASS`）；E3 负向对照证明 fp32 中转必需 |
| `history_version/design_probe_floordiv.py` | ✅ 已完成并执行 | D-3 探针 1（vfloordiv/vfloor 挂死；vdiv+vcast(floor) fp32/fp16/bf16 bit-exact） |
| `history_version/design_probe_floordiv_bcast.py` | ✅ 已完成并执行 | D-3 探针 2（bf16 链 + vdiv 广播 col/row/scalar 全部 bit-exact） |
| `floor_divide_fwd_kernel.py`（kernel 实现） | ⬜ 待实现（Stage 3） | 工厂 `floor_divide_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, output_dtype=None, block_size=4096)` → callable(block_size) → `@T.prim_func main(a, b, y)`；原语链按 §0.6 R4，广播路径按 §3.3；集成目标 wrapper `examples/TileOPs/tileops/kernels/elementwise/floor_divide/floor_divide.py` |
| golden | 随实现文件交付（§8.1） | `torch.floor(a.float()/b.float()).to(dtype)`（广播直算） |
| L0 测试 | 随实现文件交付（§8.3） | 12 个代表性 case |

### 10.3 命名规范

- 项目目录：`floor_divide`；算子目录：`floor_divide_fwd_kernel`（源算子函数名，harness 提取件同名）
- 实现文件名：`floor_divide_fwd_kernel.py`；测试入口内嵌（`if __name__ == "__main__"`，`vec_add_1d.py` 先例）
- `@tilelang.jit` 装饰函数名保留 `floor_divide_fwd_kernel`（迁移执行规则）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ golden（§8.1）
3. ⬜ kernel 实现（§3.3：同形 `k=1` 路径 → 广播 `k>1` 四子情形（含 vbrc 物化）→ fp32/半精度 dtype 分支）
4. ⬜ L0 门槛测试全绿（§8.3）
5. ⬜ harness 集成（wrapper `forward` 按 `block_size` 惰性构建 + 缓存，§0.7）

---

## 11. 设计核心与 GPU→NPU 关键重设计（下游报告提炼用）

> 本节把全文设计核心收敛为 4 条可逐算子引用的要点。

1. **无可用一体化 floor-div 原语 → 两段式向量链，且必须在 fp32 域**：npuir 语言层虽有语义完美匹配的 `T.vfloordiv`（`C=floor(A/B)`）与 `T.vfloor`，但**探针实测二者编译成功却运行期 Vector 核挂死**（同形/切片/shared 全挂）；改用 `T.vdiv`（fp32 向量除）+ `T.vcast(round_mode="floor")`（fp32 向量 floor）两条指令。**除法/floor 必须在 fp32 域**（源语义：fp16 商先舍入再 floor 会错；`verify_equiv.py` E3 实测免中转 fp16 66 / bf16 430 处偏差）——fp16/bf16 经 `vcast(rint)` 无损提升 + 单次回落，fp32 免中转且**禁止** `f32→f32 rint` 恒等 cast（探针 C1 得 NaN）。
2. **多核分块（persistent 分核）**：把展平后的 coalesced 输出切成 `R 行 × ceildiv(L, block_size)` 个逻辑 task，用一维 `T.Kernel(min(num_logical, 物理核数), is_npu=True)` + 核内 `T.serial`（静态边界，grid-stride 映射）常驻 ≤40 核，替换源 GPU 的 CUDA grid 超发 × `T.Parallel(threads, npt)` 两级线程并行；核数由 `NPUUtils.get().get_aicore_num()*2` 实查（本机 20⇒40）。
3. **广播索引重设计（行级 base + innermost 维向量化）**：源的广播路径对每个输出元素做 O(ndim) unrolled divmod + 标量 gather；改为「外层行标量 base（每 task 仅 O(ndim) 次 divmod，降 ~L 倍）+ innermost 维连续向量化」。**本算子的有利差异**：`T.vdiv` **支持自动广播**（探针 F1/F2/F3，与 `T.vxor`/`T.vcmp` 相反，CG-2026-0020/0021），故广播 operand 的 `T.vbrc` 物化只是**可选**（主选保留以复用同族脚手架；§1.6.3 备选 #2 用 `[R_blk,L]` tile + vdiv 广播免 vbrc，交 Stage 4 裁决）。
4. **dtype 契约（float-only、same_as 输出）**：源算子为 float-only（`_FLOAT_DTYPES`），输入/输出同 dtype，**无** eq 的 bool→1 字节承载问题、**无** uint8 不可用问题；整数/布尔 dtype 在 Op 层 `SUPPORTED_DTYPES` 拒绝（对照 manifest/signature 与 `test_floor_divide_rejects_integer_dtype`）。
