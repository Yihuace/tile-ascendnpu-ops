# le_fwd_kernel 算子设计文档

> 迁移任务（`scenario=migration`，`migration_mode=harness`，GPU 仓为 TileOPs 同构工程）。源算子 = GPU 仓 TileOPs 的 `le_fwd_kernel`（`_make_binary_explicit` 工厂 + `BinaryKernel` 二进制模板的 `op_func(a,b) = a <= b` 特化，`OUTPUT_DTYPE = torch.bool`），迁往 TileLang-NPUIR（`target="npuir"`，华为昇腾 NPU），编程模式固定 `Developer`。
>
> 本文 §0 记录源算子三问解读 → 算法调研（Phase R，结论见 §1.6.0，作为 M1 判定输入）→ 硬件耦合性判定 → NPU 算法重设计的全过程；§1–§11 基于**重设计后的 NPU 算法**展开。
>
> **本算子相对同族 `ge` / `gt` 的关键差异**（由 §0.6 / §1.6.1 论证、`verify_equiv.py` 机器验证）：
> 1. **`<=` 是弱序（包含相等），与 `ge` 同侧、与 `gt` 反侧**：相等操作数（含 `+0.0`/`-0.0`）恒为 **True**（`verify_equiv.py` E-ZERO 实测 4/4 角点 True、`a<=a` 全 True），与 `gt` 的「相等恒 False」相反。其**符号敏感性**与 ge/gt 相同（都依赖按 dtype 的有序比较）→ 复用 ge 的 uint8 无符号域修正套路 `u = s + 256·(s<0)`（E5 机器验证，256×256 全枚举 0 违反；E-NAIVE 证明朴素 int8 承载在 1M 随机字节中 523878 处不一致）。
> 2. **`<=` 的补集恒等式 `a <= b ≡ ¬(a > b)` 在浮点下不成立**（NaN 时 `¬(NaN>x)=True` 而 `NaN<=x=False`；E-NAN 实测 4/6 角点不一致）→ §1.6.1 R1 候选 1 否决；**NaN 等价回退式是析取 `(a < b) ∨ (a == b)`**（而非严格序 gt 的合取 `(a>=b)∧(a!=b)`）——此为**弱序**特有，E-NAN 实测 0 违反。
> 3. **int16 比较域改升 fp32**（eq 设计遗留缺口，ge 已闭合）：`T.vcmp` int16 在所有操作数形态下被标量化（pattern-library PL-1.11 实测 [32,256] 单链 236µs）；int16 在 fp32 中**精确可表示**（E4 全 65536 值 0 违反）→ le 沿用 `int16 → fp32` 升域。
>
> 关键能力结论沿用同工具链同设备的设计期探针实测（D-3，脚本见 `examples/eq/eq_fwd_kernel/history_version/design_probe_eq_cmp.py` 与 `design_probe_eq_broadcast.py`，均为最小张量、总墙钟 <10 min；本算子不新增探针，能力结论同源）：
> - **`T.vcmp` 支持 fp16/bf16/fp32/int16/int32/int64**（同形比较，输出片内 bool），比较模式含 `"le"`（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1/§2.2.1）；**不支持 int8/uint8**（同文档 §2.2.1 矩阵）。
> - **`T.vcmp` 不接受任何自动广播**：`[4,1024]` vs `[1,1024]` / `[4,1024]` vs `[4,1]` 均 BiShengHIR `'hivm.hir.vcmp' op operands' shape are inconsistent`（与 `T.vxor` 同型缺口）→ 广播维必须显式物化（CG-2026-0021；`T.vcmp.md` §2.2.2「输入 A 和 B 必须具有相同的 shape」）。
> - **bool(i1) GM 输出不可用**：`T.copy(bool_ub → bool_GM)` **编译成功**但运行期 Vector 核执行超时（`AclrtSynchronizeStreamWithTimeout` 507034）——`T.vcmp.md` §2.3 记「编译失败」，实测为运行期挂死（CG-2026-0022）→ 1 字节输出必须由 int8 承载。
> - **bool→int8(0/1) 转换链可用**：`vcast(bool→f16)` + `vcast(f16→i8, rint)` 全 dtype 实测 bit-correct（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 `i1→f16`/`f16→i8`）；`T.vselect(cond, 1.0f16, 0.0f16)` 亦可用。
> - **uint8 GM 张量不受支持**（CG-2026-0018）；**int8 vbrc dst 运行期挂死**（同族 ge/gt 设计期探针记录，本算子沿用其绕法：1 字节标量广播先升 f16 再 `vbrc`）。
> 工具链 `tilelang 0.1.2+417dae9908ad` / CANN 8.5.0 / 本机 NPU（`NPUUtils.get().get_aicore_num()` 实查 = 20）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素小于等于比较 `y = (a <= b)`，等价于 `torch.le(a, b)`。支持 NumPy/PyTorch 广播：`y.shape == broadcast_shapes(a.shape, b.shape)`。输出为 **bool**（`OUTPUT_DTYPE = torch.bool`）。浮点路径遵循 IEEE 754 比较语义：**任一操作数为 NaN 时结果恒为 False**；**相等操作数（含 `+0.0` 与 `-0.0`，以及 `±Inf` 与自身）恒为 True**（**弱序包含相等**，与 gt 的严格序相反）；`-Inf <= 任意有限值` 为 True，`+Inf <= 有限值` 为 False。整数/布尔路径为原生 dtype 的弱序（bool 为 0/1 序）。**注意**：与 `==` 不同，`<=` 的排序语义**依赖于 dtype 的符号性**——`a <= b` 在 uint8 与 int8 重解释下的结果不同（见 §0.6 R5）。
- **I/O 契约**：两输入一输出；两输入同 dtype；**输出 dtype = bool**（与位运算族「same_as(input)」不同，是本算子的关键差异之一）；输出 shape = 两输入 broadcast 结果。算子层把两输入 `contiguous()` 后展平为 1-D `(a_numel,)` / `(b_numel,)`，输出展平为 `(N_total,)`，kernel 只看一维张量；广播 lowering（`coalesce_broadcast_dims`）在 kernel 构造期完成（`examples/TileOPs/tileops/kernels/elementwise/le/le.py` L171）。
- **规约语义**：**无规约、无跨元素依赖**——每个输出元素只依赖同位置的 `a`/`b` 元素（经广播索引），`y[i]` 与 `y[j]` 完全独立。因此**并行组织/分核/累加分组的变化不影响任何数值结果**（无浮点累加顺序问题；比较为逐元素判定）。
- **dtype 语义**：输入族 `_BINARY_FULL_DTYPES = bool | uint8 | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32`（`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L166–L173；manifest `LeFwdOp` signature 同）；输出恒为 `bool`。**bool 特殊路径**：GPU 因 CUDA codegen 无法 lower `boolx<N>`，Op 层把 `torch.bool` 张量 `view(torch.uint8)` 后走 `LeBoolStorageFwdKernel`（`op_func = bitwise_or(bitwise_xor(a, 1), b)`，对 0/1 值域等价于 `~a | b = ¬(a ∧ ¬b) = ¬(a > b) = a <= b`，即弱序），即 **bool 以 1 字节存储承载**。浮点比较不做中间提升；整数比较在原生（含无符号）整型域。
- **边界语义**：浮点 NaN（任何含 NaN 的比较 → False）/±Inf/±0 按 IEEE 比较；**弱序的相等边界**（`+0.0 <= -0.0`、`-0.0 <= +0.0`、`a <= a` 均为 True）必须与金色实现一致；整型二补码/无符号值序；空维度/0-size 输入由 Op 层 `prod()==1` 兜底或上层拦截（`torch.broadcast_shapes` 把 0-dim 归一为 `(1,)`）；非整除 shape 由广播规则处理（`out_shape` 可为任意广播形状）。
- **host 侧语义**：`contiguous().view(-1)` 展平、`coalesce_broadcast_dims` 索引表示选择、`reshape(out_shape)` 回形状、bool 存储 `view(uint8)`——属 Op 层契约（实现手段，非语义），由 NPU 侧 wrapper（`examples/TileOPs/tileops/kernels/elementwise/le/le.py` 的 `LeFwdKernel` + `LeBoolStorageFwdKernel`；Op 层 `examples/TileOPs/tileops/ops/elementwise/le.py` 的 `LeFwdOp`）承接，不属于本 kernel 的重设计范围。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.le` 广播直算），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a`（input） | `(a_numel,)` | bool / uint8 / int8 / int16 / int32 / int64 / float16 / bfloat16 / float32 | Op 层已展平的 1-D 连续张量 |
| `b`（other） | `(b_numel,)` | same_as(a) | 同上 |
| `y`（output） | `(N_total,)` | **bool** | `out_idx=[2]`；**输出为一维展平视图**（源码 `main` 的 `y: T.Tensor((N_total,), out_dtype)`，`output_dtype="bool"`），Op 层再 `reshape(out_shape)`，无转置/布局变换。广播时 `a_numel` / `b_numel` / `N_total` 三者可不同 |

**输出 shape/dtype 依据**：源文件 `examples/TileOPs/tileops/kernels/elementwise/le/_le_fwd_kernels.py` L56 / L79（`y: T.Tensor((N_total,), out_dtype)`，`out_dtype = output_dtype or dtype` 且 Op 层传 `"bool"`）；NPU wrapper `LeFwdKernel` 声明 `OUTPUT_DTYPE = torch.bool`（`le.py` L153）、`output_dtype_str = Kernel.dtype_to_str(torch.bool)`（`le.py` L181），`forward` 返回后由 `LeFwdOp` 的 op 层 `reshape(self.out_shape)` 还原广播形状；manifest `shape_rules: output.shape == broadcast_shapes(input.shape, other.shape)`、`outputs.output.dtype: "bool"`、`source_output_shape: "(N_total,)"`。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/le/_le_fwd_kernels.py`（Stage 0 提取件，Pattern A：`_make_binary_explicit` 内联 `op_func(a,b) = (a <= b)`，并修正提取脚本丢失的第二个操作数索引）；GPU 上下文：`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L556–L609（`_make_binary_explicit`）、L890–L1128（`BinaryKernel`）、L1799–L1816（`LeFwdKernel` / `LeBoolStorageFwdKernel`）。

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host） | `coalesce_broadcast_dims(a_shape, b_shape)` → `out_shape, coalesced_shape, a_strides, b_strides`（stride=0 表示广播维）；`_is_contiguous_same_shape` 选快路径 | a_shape/b_shape | 索引表示（编译期常量） | 广播 lowering（实现手段） |
| 2 | `block_size = threads * num_per_thread`；grid = `T.ceildiv(N_total, block_size)` | threads/npt | CUDA 任务划分 | 任务划分 |
| 3a（同形连续快路径） | `for i,j in T.Parallel(threads,npt): idx=(bx*threads+i)*npt+j; y[idx] = a[idx] <= b[idx]` | GM a/b | GM y(bool) | `y = (a <= b)`（同形） |
| 3b（广播路径） | `flat_idx=(bx*threads+i)*npt+j; (a_off,b_off)=_compute_broadcast_offsets(flat_idx, ndim, divisors, a_strides, b_strides); y[flat_idx]=a[a_off] <= b[b_off]` | GM a/b | GM y(bool) | `y = (a <= b)`（广播索引） |
| 4（host） | Op 层 `reshape(out_shape)` | `(N_total,)` 输出 | broadcast shape | 后处理（Op 层契约） |
| 5（bool 输入，host+kernel） | Op 层 `view(torch.uint8)` → `LeBoolStorageFwdKernel`：`y = bitwise_or(bitwise_xor(a, 1), b)`（`~a | b`，0/1 值域上等价于 `a <= b`） | GM a/b（uint8 承载） | GM y(uint8 承载 bool) | `y = (a <= b)`（bool 存储路径） |

`_compute_broadcast_offsets`（源 `_le_fwd_kernels.py` L161–L177）是**编译期 unrolled 的 mixed-radix divmod 链**：沿 coalesced 各维依次 `coord = remaining // divisors[d]; remaining %= divisors[d]; a_off += coord*a_strides[d]`，最后 `a_off += remaining*a_strides[-1]`。**这是逐元素的标量整数 divmod**（每输出元素 O(ndim) 次除法/取模）。

**数据流与内存访问模式**（GPU 视角）：

```
GM[a] ──(逐元素 strided/标量读)──> register ┐
GM[b] ──(逐元素 strided/标量读)──> register ┼─ 标量比较 a <= b ──> register y(bool) ──> GM[y]
（同形快路径：T.copy 连续段 → register；广播路径：逐元素标量索引 gather 读 GM）
```

**循环与并行结构**：CUDA grid（`ceildiv(N_total, block_size)` 个 block）× block 内 `T.Parallel(threads, num_per_thread)` 二维线程并行；`threads=256`，`num_per_thread=4`（fp16/bf16/fp32，`_strategy_npt("explicit_parallel", …)` L181–L195）或 `8`（整数族），`block_size=1024`（浮点）/`2048`（整数）；尾块由 TileLang `LegalizeSafeMemoryAccess` 隐式保护。

**host 侧逻辑**：`@tilelang.jit(out_idx=[2])` + `functools.lru_cache(maxsize=32)` 按 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, …)` 特化编译；`out_dtype = output_dtype or dtype`（Op 层传 `"bool"`）；bool 输入由 Op 层 `view(uint8)` 后走 `LeBoolStorageFwdKernel`。NPU wrapper 冻结工厂签名为返回 `callable(block_size)`（K11），`default_config` 给 `{"block_size": 256×npt}`（浮点 1024 / 整数 2048 / bool 存储 4096）。

### 0.4 优化手段解读（为什么快）

> 对照 migration-analysis.md §4.1 清单逐项核对；源码显式优化手段如下（无 SMEM tiling / warp shuffle / 异步流水 / swizzle，对照清单无遗漏）。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel`：`T.Parallel(threads, num_per_thread)` | 满并行 + 每线程多元素 ILP | 每线程连续处理 `npt` 个元素（`idx=(bx*threads+i)*npt+j`） | CUDA block/thread 线程模型 | 硬件强相关 |
| 2 | 连续段 coalesced 读写 | 访存带宽 | 同形路径按 `block_size` 连续段访问，尾轴连续 | CUDA memory coalescing | 硬件强相关 |
| 3 | `coalesce_broadcast_dims`（维度合并） | 减少 divmod 次数 | 合并同广播行为的相邻维，最小化 effective dims | 无（纯索引表示） | 可移植 |
| 4 | 编译期 unrolled divmod 索引（`_compute_broadcast_offsets`） | 免运行时 shape 查询、广播通用化 | 常量除法在 build 期展开；但**每元素仍 1 次 divmod 链** | 无（通用），但代价随 ndim 线性增长 | 可移植（代价高，见 §1.6.0 R3） |
| 5 | 同形连续快路径（`_is_contiguous_same_shape`）跳过广播机制 | 消广播开销 | 分支在 build 期选择 kernel body | 无（编译期分派） | 可移植 |
| 6 | `register_copy` 策略（fragment load→compute→store，binary 仅同形且非 bool 输出） | 数据驻留寄存器 + 128bit 向量访存 | `T.alloc_fragment` + `T.copy` | GPU 寄存器堆 + fragment 机制 | 硬件强相关 |
| 7 | 工厂闭包 + `lru_cache(maxsize=32)` | 摊薄 JIT 编译开销 | 每参数组合只编译一次 | 无（host 层） | 可移植 |
| 8 | bool 走 uint8 存储（`view(torch.uint8)` + `LeBoolStorageFwdKernel` 的 `bitwise_or(a, bitwise_xor(b,1))`） | 绕过 CUDA 无法 lower `boolx<N>` | 1 字节无符号存储承载 bool 位模式；对 0/1 值域该式等价于 `a <= b`（弱序） | CUDA codegen 限制（非硬件） | 需 NPU 侧替换（uint8 在 npuir 不可用；且**排序语义对符号敏感**，见 §0.5/§0.6 R5） |
| 9 | 比较本体为单条 `<=`（1 op/元素） | 计算量下界 | 原生 dtype 域一次弱序比较（含整数无符号域） | 无 | 可移植（NPU 换 `T.vcmp("le")`，见 §0.6 R4） |

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `y = (a <= b)` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；逐元素判定无累加顺序问题；NaN/±0/相等边界 IEEE 语义由 `T.vcmp("le")` 承接（L0 角点覆盖，§9.1 R6） |
| I/O 契约（1-D 进 1-D 出、广播 out_shape） | 语义 | 无 | — | **保留** | 输出 shape 一维由 §0.2 源码解读确认；广播形状还原在 Op 层 |
| dtype 契约（输入族 bool/uint8/int8/16/32/64/fp16/bf16/fp32） | 语义 | 无 | — | **保留（无符号 uint8 表示需重设计）** | 输入 dtype 契约沿用；**uint8 因 npuir 不支持 uint8 GM 张量（CG-2026-0018）改由 int8 承载，但 `<=` 排序对符号敏感 → 必须做无符号域修正**（→ §0.6 R5，`verify_equiv.py` E-NAIVE/E5）；bool 因 CUDA/NPU 限制本就走 1 字节存储承载（→ §0.6 R3） |
| **输出 dtype 契约（bool）** | 语义 | 无 | **不可直接落盘**（bool/i1 无法 store GM，见 CG-2026-0022） | **重新设计** | 输出以 **int8 0/1** 承载，Op 层 `view(torch.bool)` 还原 bool（→ §0.6 R3）——**本算子区别于位运算族的关键点** |
| host 展平/coalesce/回形状 | 语义（Op 契约） | 无 | — | **保留** | NPU wrapper 已承接；host 侧仅元数据/视图操作，符合 ascend-constraints.md §4 |
| `coalesce_broadcast_dims` 维度合并 | 算法 | 无 | — | **保留** | 索引表示选择，与硬件无关；wrapper 已承接（`le.py` L171） |
| 同形连续快路径（build 期分支） | 算法 | 无 | — | **等价替换** | 统一进 §0.6 R1 的行/内层方案（k=1,R=1 退化为同形路径），意图（免广播机制）承接 |
| 一维 grid `ceildiv(N_total, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1；`examples/elementwise/vec_add_1d.py`）；核数按 §5.5 分核适配（→ R2） |
| `T.Parallel(threads, npt)` 两级线程并行 | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 R2（一维 Kernel + Vector 向量化；npt 的多元素意图由 128bit 向量 lane 承接） |
| `threads=` / `num_per_thread` kwarg | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已把 `threads*npt` 折叠为单一 `block_size`，无性能意图丢失（并行组织由 R2 承接） |
| 连续段 coalesced 搬运 | 优化 | CUDA coalescing | 有（GM↔UB 连续段 T.copy） | **等价替换** | `T.copy` 连续段 GM↔UB（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款） |
| `register_copy`（fragment 驻留） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment`→`T.alloc_ub`（migration-analysis.md §5.3「register 累加→UB 中间缓冲」）；v-prefix 要求操作数在 UB |
| **逐元素 divmod 广播索引** | 优化 | 无（通用） | 无等价（NPU 无逐元素整数 divmod 快路径；且源形态为 GM 标量 gather） | **重新设计** | → §0.6 R1（外层行标量 base + innermost 维向量化 + 显式同形物化） |
| **`T.Parallel` 逐元素标量 `<=` 直译** | 优化 | CUDA 线程模型 | ❌ 无广播能力（探针实测） | **重新设计** | → §0.6 R1/R4：`T.vcmp` **仅接受同形操作数**（CG-2026-0021），且不支持 int8/uint8——比较域需按 dtype 映射，广播维需显式物化 |
| **标量逐元素比较本体 `a <= b`** | 优化 | 无 | 有（`T.vcmp`，模式 `"le"`，输出片内 bool） | **等价替换 + 重设计** | → §0.6 R4（`T.vcmp("le")`）+ R5（无符号域修正）+ R3（bool→int8 落盘） |
| 工厂闭包 + lru_cache 特化 | 优化（host） | 无 | — | **保留** | wrapper `forward` 已按 `block_size` 惰性构建 + 缓存（`le.py` L216–L220） |
| bool 走 uint8 存储 | 优化 | CUDA codegen 限制 | uint8 在 npuir **不可用**（CG-2026-0018）；bool GM store **运行期挂死**（CG-2026-0022） | **重新设计** | → §0.6 R3（输入侧改 int8 承载；输出侧 bool→int8 0/1） |
| **整数无符号序（uint8 原生无符号比较）** | 优化 | 无 | NPU `T.vcmp` 无 unsigned/uint8 支持，且**排序对符号敏感** | **重新设计** | → §0.6 R5（无符号域修正 `u = s + 256·(s<0)`，`verify_equiv.py` E5 机器验证） |
| GPU grid 自由超发 | 优化 | CUDA 调度器 | 无（超发被串行调度 + 核启动开销） | **重新设计** | → §0.6 R2（persistent 分核，core-split-strategy.md §1） |

**判定统计**：保留 7 项、等价替换 3 项、重新设计 5 项（R1 广播索引 / R2 并行与分核 / R3 1 字节承载 dtype / R4 比较域与向量比较映射 / R5 uint8 无符号域修正）、舍弃 1 项（`threads=`/`npt` kwarg，理由见表）。

> **与 ge / gt 的判定差异**：处置集与 ge/gt **完全相同**（5 项重设计一一对应，因三者同为「bool 输出的有序比较」），差异仅在 **R4 的比较模式（`"le"` vs `"ge"`/`"gt"`）与相等边界/NaN 回退式**：`le` 是**弱序**，相等恒 True（`+0 <= -0` 为 True，与 gt 相反、与 ge 相同），NaN 回退式为**析取** `(a<b)∨(a==b)`（与 ge 同、与 gt 的合取相反）。**与 eq 的判定差异**：eq 的 `==` 是**符号无关**的，故「uint8→int8 承载」被列为单向等价替换；le 的 `<=` 是**符号相关**的，同一处置在本算子**升级为重新设计（R5）**——朴素 int8 承载在字节 ≥128 时全错（E-NAIVE 523878/1M），与 ge/gt 同源。

### 0.6 NPU 算法重设计

**重设计项 R1: 广播索引路径（逐元素 divmod/gather → 外层行标量 base + innermost 维向量化 + 显式同形物化）**

- **源方案**：广播路径对**每个输出元素**执行一次编译期 unrolled mixed-radix divmod 链（`_compute_broadcast_offsets`），得到 `a_off`/`b_off` 后以标量 gather 读 GM 的 `a[a_off]`、`b[b_off]`；同形路径才用连续段。性能意图：以「编译期常量除法展开」把广播通用化的代价压到单元素 O(ndim) 整数 divmod（GPU 多核并发消化）。
- **NPU 新算法**：把 coalesced 输出视为 **`R = prod(coalesced_shape[:-1])` 个「行」× 行长 `L = coalesced_shape[-1]`** 的二级结构（`k=1` 时 `R=1, L=N_total`，同一模板覆盖同形与广播两路）。对每个「行」（即一个 task）：
  1. **行 base 标量**：`outer_a_base = Σ_{i<k-1} coord_i(row)·a_strides[i]`（`coord_i` 由 row 对 `coalesced_shape[:-1]` 的 mixed-radix 分解得，编译期 unrolled，**仅每 task 一次**，非每元素）；`b` 同理。末维：`a_base = outer_a_base + (inner_off if a_strides[-1]==1 else 0)`（末维 stride ∈ {0,1}）。
  2. **行内 operand 物化（R4：因 `T.vcmp` 仅同形，CG-2026-0021）**：末维 `L` 按 `block_size` 分块（`chunks = ceildiv(L, block_size)`；manifest 广播行 `L≤4096`，推荐 `block_size=4096` 使 `chunks=1`）。每块**必须**把两个 operand 都物化为 `(block_size,)` 同形 UB（比较域见 R4/R5）：
     - `a_strides[-1]==1`（a 末维连续）→ 连续段 `T.copy(a[a_base:a_base+tail] → a_ub[0:tail])`；否则（a 末维 stride 0，a 在行内为标量）→ `T.copy(a[a_base:a_base+1] → a1)` + `T.vbrc(a1 → a_c)`；
     - `b` 同理；
     - 两 operand 均物化同形后 `T.vcmp(a_c, b_c, cond, "le")`（**同形**，满足 CG-2026-0021 约束）→ bool 经 R3 落盘。
     - **末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`**（证明：输出末维 size>1 时至少一个操作数在该维 size>1，其连续 stride=1），故 `(0,0)` 组合不可能（`L==1` 全标量退化为两路 `T.vbrc`，`block_size>1` 时合法）。
  3. **stride-0 物化的 dtype 选择（R4 附带）**：int8/uint8/bool 输入先 `vcast` 升 fp16（int8 路径先做 R5 无符号修正）再 `T.vbrc`——既满足 `T.vcmp` 的 int8 不支持、又规避 int8 dst `T.vbrc` 运行期挂死（同族 ge/gt 设计期探针记录；int8 在 fp16 中精确表示）。
- **语义保持论证**：**数学恒等**——新映射对每个 flat 索引给出的 `(a_off, b_off)` 与源 divmod 链**逐点相同**（行分解 `row=flat//L`、`inner=flat%L` 恰为末维与其余维的 mixed-radix 分解，`a_off = Σ_{i<k-1}coord_i·a_strides[i] + coord_{k-1}·a_strides[-1]`），**已由 `verify_equiv.py` E2 机器验证**（10 组形状含全部 manifest + 测试广播模式，`a_off`/`b_off` 逐点 mismatch=0、越界=0，EQUIV_PASS）。比较本身逐元素、无跨元素依赖；广播物化（vbrc/连续段）与 PyTorch 广播逐点一致。空维度/0-dim 由 Op 层归一，kernel 不触达。
- **意图承接**：源优化手段 3（维度合并）原样保留（wrapper 沿用 `coalesce_broadcast_dims`）；手段 4（编译期 unrolled divmod）的「免运行时 shape 查询」意图由编译期 unrolled **行** divmod 承接，代价从「每元素 O(ndim)」降为「每 task O(ndim)」（约降 `L` 倍）；手段 5（同形快路径）由 `k=1⇒R=1` 的统一退化承接。

**重设计项 R2: 并行结构与分核（CUDA block×thread 两级并行 + grid 超发 → 一维 Kernel + Vector 向量化 + persistent 分核）**

- **源方案**：`T.Kernel(ceildiv(N_total, block_size), threads=threads)` 的 CUDA grid×block 两级并行，block 内 `T.Parallel(threads, npt)` 二维线程映射，每线程标量处理 npt 个元素；grid 大小任意（最大 workload 下数千 block），超发由 CUDA 硬件调度器吞吐。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；核内 `for i in T.serial(num_local_tasks)`（**静态边界**）以 grid-stride 映射 task `= i*num_kernels + cid`，`if task < num_logical` 屏蔽越界。元素级并行交给 Vector 单元：逐元素 `a <= b` 由 `T.vcmp("le")` 一条向量指令覆盖整块（128bit 向量 lane：fp16 ×8 / fp32 ×4 等）；bool→int8 由 `T.vcast` 链覆盖整块。核数按 §5.5 分核（`num_kernels = min(num_logical, 物理 Vector 核数)`）。
- **语义保持论证**：task↔行/c块一一对应、各 task 完全独立（无跨 task 依赖、无竞写），映射顺序/分核数变化不改变任何输出值；向量化只是把逐元素标量 `<=` 换成等价的逐元素向量比较（lane 间独立）。无累加顺序问题（NaN/±0 语义由 vcmp 保证，L0 角点覆盖）。
- **意图承接**：手段 1（`T.Parallel(threads,npt)` 的满并行 + ILP）→ Vector 128bit lane + `block_size` tile 粒度；手段 2（coalesced）→ `T.copy` 连续段；手段 6（fragment 驻留）→ `T.alloc_ub` 原地缓冲；`threads=`/npt 折叠为 `block_size`（wrapper K11），无性能意图丢失。

**重设计项 R3: 1 字节承载 dtype（输出 bool→int8；输入 uint8/bool→int8）**

- **源方案**：输出直接声明为 `bool`（CUDA 上 1 字节可落盘）；bool 输入由 Op 层 `view(torch.uint8)`，以 1 字节无符号存储跑 `LeBoolStorageFwdKernel`（因 CUDA 无法 lower `boolx<N>`）。
- **NPU 新算法**：
  - **输出**：kernel 输出张量声明为 **int8**，值为规范 0/1；bool 由 Op 层 `result.view(torch.bool)`（非零即真）还原。依据：`T.copy(bool_ub → bool_GM)` **编译成功但运行期 Vector 核超时 507034**（CG-2026-0022；`T.vcmp.md` §2.3 记「编译失败」与实测不符）→ bool GM 输出不可用。bool→int8 链实测 bit-correct：`T.vcast(cond_bool, y_f16)`（`T.vcast.md` §2.2.1 `i1→f16`）+ `T.vcast(y_f16, y_i8, round_mode="rint")`（`f16→i8` 可用）。
  - **输入 uint8/bool**：由 Op 层 `view(torch.int8)`，kernel 以 int8 声明张量与 UB。依据 CG-2026-0018（uint8 GM 张量不可用）。**注意**：此承载对 bool（值域 {0,1}）**直接等价**（`verify_equiv.py` E1），但对**一般 uint8 必须叠加 R5 的无符号域修正**。
  - **le 特有边界（弱序）**：bool 存储路径上源用 `bitwise_or(bitwise_xor(a,1), b)`= `~a | b`（与 gt 的 `a & ~b` 不同），在 0/1 值域上恰为 `a <= b`；NPU 统一走 int8 承载 + `T.vcmp("le")`，两者逐点一致（E1）。
- **语义保持论证**：bool 输出只有 {0,1}，int8 承载无编码损失，`view(torch.bool)` 按「非零即真」还原与 `torch.le` 的 bool 输出逐元素相等（E1）；bool 存储输入（0/1 字节）在 int8/uint8 重解释下 `<=` 结果恒等（E1 violations=0）。**一般 uint8 输入的等价性由 R5 提供**（E5）。
- **意图承接**：手段 8 的「1 字节存储承载 bool」意图完全保留，仅把「无符号/布尔输出」统一替换为「int8 0/1 承载」，并对一般 uint8 叠加 R5。

**重设计项 R4: 比较域映射与向量比较（原生标量 `<=` → dtype 映射 + 同形 `T.vcmp("le")`）**

- **源方案**：GPU 在**原生 dtype 域**直接做标量 `<=`（浮点走 IEEE 严格序比较，整数走原生有/无符号排序，bool 存储走 `~a | b`），向量指令无操作数同形要求，广播由索引 strides 隐式表达。
- **NPU 新算法（dtype 映射表）**：由输入 dtype 决定「比较域 dtype」与「cmp 前置转换」：

  | 输入 dtype | 承载 dtype（kernel I/O） | 比较域 dtype | 前置转换 | 说明 |
  |-----------|------------------------|-------------|---------|------|
  | float16 / bfloat16 / float32 | 同 | 同 | 无 | `T.vcmp("le")` 原生支持 |
  | int32 / int64 | 同 | 同 | 无 | 原生域（fp32 对 int32>2²⁴ 不精确，fp64 不可用）；vcmp int32/int64 标量化风险**未文档化**，列 §9.1 R7 待确认 |
  | **int16** | 同 | **float32** | `vcast(i16→f32, rint)` | vcmp int16 **被标量化**（PL-1.11，[32,256] 单链实测 236µs）；int16 在 fp32 中精确（E4）。沿用 ge 的升域改进 |
  | int8 | int8 | **float16** | `vcast(i8→f16, rint)` | vcmp 不支持 int8；int8 有符号序在 fp16 精确（E3） |
  | **uint8（一般值）** | int8（Op 层 view） | **float16** | `vcast(i8→f16, rint)` **+ R5 无符号域修正** | vcmp 不支持 uint8，且 `<=` 排序对符号敏感 → 必须先修正为真无符号值（R5，E5） |
  | bool（存储 0/1） | int8（Op 层 view） | **float16** | `vcast(i8→f16, rint)` | 值域 {0,1}，符号无关；int8 承载直接等价（E1） |

  随后：`T.vcmp(a_c, b_c, cond, "le")` → bool 片内 → R3 的 bool→int8 链 → 连续段写回。
  浮点比较语义（**含 NaN 恒 False**、**相等恒 True（弱序）**、±0 相等）由 `T.vcmp("le")` 承接，L0 角点覆盖（§9.1 R6）。
- **语义保持论证**：比较域映射**保序**——int8、uint8（经 R5 修正）、int16（→fp32 精确）在各自升域后与原域序同构（E3/E4/E5 机器验证）；因此 `vcmp(升域(a), 升域(b), "le")` 与原 `a <= b` 逐点相同。原生 fp16/bf16/fp32/int32/int64 无转换损失（IEEE/整型序）。
- **意图承接**：源手段 9（比较本体 1 op/元素）由 `T.vcmp("le")` 一条向量指令承接；源「原生域比较」意图在 vcmp 支持的 dtype 上原样保留，仅在 vcmp 不支持的 1 字节 dtype 上以精确升域承接（不得静默降精度），在符号敏感的 uint8 上以 R5 精确修正承接。

**重设计项 R5: uint8 无符号域修正（`<=` 排序对符号敏感；le 沿用 ge 的已验证套路）**

- **源方案**：GPU 在 `uint8` 原生域做**无符号**弱序比较（`a <= b` 按 0…255 的值序）；CPU/HW 直接支持 unsigned byte 比较，无需任何显式处理。
- **NPU 新算法**：uint8 张量因 npuir 不支持而必须由 int8 承载（CG-2026-0018），但 `<=` 的排序**不是符号无关的**——同一字节在 uint8 解释（0…255）与 int8 解释（−128…127）下大小关系不同（`verify_equiv.py` E-NAIVE 实测 1M 随机字节 **523878 处不一致**，即约 50%）。因此在 `T.vcmp` 前必须把承载值修正为**真无符号值**：
  - `s = vcast(a_i8 → f16, rint)`（有符号、精确）；
  - `m = vcmp(s, 0.0f16, "lt")`（片内 bool，`s<0`）；
  - `sp = vadd(s, 256.0f16)`（f16 标量加，`T.vadd` 文档合法）；
  - `u = vselect(m, sp, s)`（`s<0 → s+256`，否则 `s`）→ `u ∈ [0,255]`，即真无符号值。
  修正后与 b 侧同样修正，再做同形 `T.vcmp("le")`。全部算子（`vcast`/`vcmp`/`vadd`/`vselect`）均为文档化 f16 向量 op。
  表达式的闭式等价：`u = s + 256·(s<0)`（`s = a_i8` 的有符号值）。
- **语义保持论证**：**值域恒等**——对全部 256 个字节，`u` 恰为 `uint8` 的真值；`u_a <= u_b ⟺ a_u8 <= b_u8`（`verify_equiv.py` E5：256×256 全枚举 + 1M 随机注入相等位，violations=0，EQUIV_PASS）。fp16 对 `u ∈ [0,255]` 精确，故比较无舍入。**弱序特有**：修正后相等对（`u_a == u_b`）的 `<=` 恒为 True（E5 注入相等位覆盖），与源一致。空/边界由 Op 层归一。
- **意图承接**：源手段 9 的「原生无符号比较」意图以「int8 承载 + 显式无符号修正」承接，语义不变；代价 = 每 uint8 操作数 +3 条 f16 向量 op（§1.6.0 R3 计入）。
- **为何不能省（相对 eq 的关键差异）**：`==` 是符号无关的（`a_u8 == b_u8 ⟺ int8(a) == int8(b)`），故 eq 的 int8 承载可直接用；`<=` 是符号相关的，省略修正在字节 ≥128 时产生确定性错误，且测试 `test_le_integer_dtype` 的 uint8 数据仅取 [0,16]（`examples/TileOPs/tests/ops/test_le.py` L148–L149），**朴素实现会「通过测试但语义错误」**——必须显式修正并在 L0 用 ≥128 的 uint8 数据覆盖（§8.3 L0-13）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 `T.Parallel` 满并行 + ILP | Vector 向量化（`T.vcmp("le")` + `T.vcast` + uint8 修正链 128bit lane）+ 单一 `block_size` tile（R2） |
| 2 连续段 coalesced | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 `coalesce_broadcast_dims` 维度合并 | 原样保留（wrapper 承接，驱动 §0.6 R1 的 outer/inner 分解） |
| 4 编译期 unrolled divmod | R1 行级 unrolled divmod（每 task O(ndim)，较源每元素降 `L` 倍）+ innermost 维向量化 |
| 5 同形连续快路径 | R1 的 `k=1 ⇒ R=1` 统一退化 |
| 6 `register_copy` fragment 驻留 | `T.alloc_ub` 原地缓冲链 |
| 7 工厂 + lru_cache 特化 | wrapper `forward` 惰性构建 + 缓存（原样保留） |
| 8 bool 走 1 字节存储 | R3：输入 bool/uint8 → int8 承载；输出 bool → int8 0/1（uint8 不可用、bool GM 挂死），一般 uint8 叠加 R5；bool 的 `~a \| b` 语义由同形 `vcmp("le")` 等价承接 |
| 9 单条 `<=`（1 op/元素） | R4：`T.vcmp("le")` 一条向量比较 + dtype 映射（1 字节精确升 f16 域、int16 升 f32 域）；R5：uint8 无符号域修正 |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/le/_le_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 原始上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_binary_explicit` @ L556、`BinaryKernel` @ L890、`LeFwdKernel` @ L1799、`LeBoolStorageFwdKernel` @ L1811）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/comparison.py`（`LeFwdOp` @ L72）。
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/le/le.py`（`LeFwdKernel`：工厂签名保留 `(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel)`，返回 callable(`block_size`)；输出承载 dtype 由 `"bool"` 改 `"int8"`；`LeBoolStorageFwdKernel` 输入承载由 uint8 窄化为 int8）；Op 层 `examples/TileOPs/tileops/ops/elementwise/le.py` 的 `LeFwdOp`。
- **规格来源**：manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `LeFwdOp`（L437，4 workloads + roofline `flops=N / bytes=(A+B)·elem_bytes + N`）；测试 `examples/TileOPs/tests/ops/test_le.py`（`_bool_compare` 精确比对，含 3 组广播 + bool/uint8/int 族 + 已知相等边界 + 不支持 dtype 拒绝 + roofline）；基准 `examples/TileOPs/benchmarks/ops/bench_le.py`。
- **同族已踩坑参考**：`examples/ge/ge_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md`（最近同类；vcmp 无广播 / bool GM 挂死 / int16 标量化 / uint8 无符号修正四教训）、`examples/gt/gt_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md`（严格序 vs 弱序的 NaN 回退式差异）、`examples/eq/eq_fwd_kernel/DESIGN.md`（比较类唯一先例）、`examples/bitwise_and/bitwise_and_fwd_kernel/RETROSPECTIVE.md`（vbrc int8 挂死 / 工厂 block_size 分解陷阱）。
- golden 以 §0.1 语义为依据（`torch.le` 广播直算），见 §8.1。

---

## 1. 概述

### 1.1 算子名称

`le_fwd_kernel`（Op 层名 `LeFwdOp` / 算子族 `le`）

### 1.2 功能描述

逐元素小于等于比较前向核：对两个已展平为一维的同 dtype 张量计算 `y = (a <= b)`，支持 NumPy/PyTorch 广播（输出 shape = `broadcast_shapes(a_shape, b_shape)`），输出为 **1 字节 bool 语义**（NPU 侧以 int8 0/1 承载）。纯逐元素、无规约、无跨元素依赖；输入浮点/整数/布尔族。**浮点路径遵循 IEEE：任一操作数为 NaN 时结果为 False；相等操作数（含 `+0.0`/`-0.0`）结果为 True**（弱序）。

### 1.3 数学公式

$$
y[i] = \big(a[\mathrm{off}_a(i)] \le b[\mathrm{off}_b(i)]\big) \in \{0,1\}, \qquad i = 0, 1, \ldots, N_{total}-1
$$

其中 `off_a(i)`/`off_b(i)` 为广播映射（源算法用逐元素 divmod 链，见 §0.3；NPU 用行/内层分解，见 §0.6 R1）；`<=` 在按 dtype 映射的比较域中执行（见 §0.6 R4），uint8 在做无符号域修正后比较（见 §0.6 R5）。非广播（同形）时 `off_a(i)=off_b(i)=i`。

参考 API：`torch.le(a, b)`。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的结构差异逐条注明来源）**

单遍逐元素行向量算法（§1.6.0 调研选定基线公式，无更优算法族替代）：

1. **任务分解（编译期）**：把 coalesced 输出视为 `R = prod(coalesced_shape[:-1])` 行 × 行长 `L = coalesced_shape[-1]`；`num_logical = R × ceildiv(L, block_size)`。`k=1`（同形 / 全标量广播）退化为 `R=1, L=N_total`。
2. **核间**：`T.Kernel(num_kernels, is_npu=True)`（persistent，`num_kernels = min(num_logical, 物理 Vector 核数)`，§5.5）。
3. **核内（每 task = 一行的一块）**：标量计算行 base（编译期 unrolled divmod，仅每 task 一次）→ 按输入 dtype 映射比较域（§0.6 R4；**int16→fp32、int8/uint8/bool→fp16**）→ **uint8 先做无符号域修正**（§0.6 R5）→ 按末维 stride 选择「连续段 `T.copy`」或「单元素 `T.copy` + `T.vbrc`」**把两个 operand 都物化为 `(block_size,)` 同形 buffer** → `T.vcmp(a_c, b_c, cond, "le")`（**同形**，CG-2026-0021）→ **bool → int8(0/1) 转换**（`vcast(bool→f16)` + `vcast(f16→i8, rint)`，§0.6 R3）→ 连续段 `T.copy` 写回（尾块 `tail` 截断）。
4. **存储**：bool 输出与 uint8/bool 输入以 int8 承载（§0.6 R3）；一般 uint8 叠加无符号域修正（§0.6 R5）。

与源算法的**结构差异**：① 逐元素 divmod + 标量 gather（广播路径）→ **行级标量 base + innermost 维向量化 + 显式同形物化**（§0.6 R1/R4，根因是 NPU 逐元素整数 divmod/gather 无快路径，且 **`T.vcmp` 不接受任何自动广播**——必须显式 `T.vbrc`/连续段把 operand 变同形）；② CUDA grid×thread 两级并行 + `T.Parallel(threads,npt)` → 一维 Kernel + v-prefix 向量 + persistent 分核（§0.6 R2）；③ **比较本体**由标量 `<=` 改为向量 `T.vcmp("le")` + dtype 映射比较域（§0.6 R4），**输出 bool 由 int8 0/1 承载**（§0.6 R3，bool GM 输出运行期挂死）；④ **uint8 无符号域修正**（§0.6 R5）——源在原生无符号域直接比较，NPU 因 int8 承载必须显式修正。**同形路径**与源骨架（连续段载入 → 逐元素比较 → 连续段写回、片上驻留）一致，进化为 `k=1,R=1` 的统一形态。

**与同族 eq / ge / gt / bitwise 算子的算法差异（迁移视角）**：位运算族输出 dtype = 输入 dtype（same_as），`T.vxor`/`T.vand` 在原生域逐位运算；`eq` 输出 bool 但 `==` **符号无关**；`ge`/`gt` 输出 bool 且为**有序比较**（符号相关，与本算子同）。**本算子 le 与 ge 的差异**：比较模式 `"le"` vs `"ge"`——两者皆为**弱序（含相等）**，相等边界一致（`+0 <= -0`=True、`+0 >= -0`=True），NaN 回退式同为**析取**；**与 gt 的差异**：`le` 弱序（相等 True）vs `gt` 严格序（相等 False），NaN 回退式为析取（le/ge）vs 合取（gt）。`T.vcmp` 与 `T.vxor` 同为「不接受自动广播」的族内例外（CG-2026-0021 / CG-2026-0020）。

### 1.5 数据流图

```
同形 / 标量广播（k=1, R=1；L 按 block_size 分块）：
  GM[a] ──T.copy(连续段, 尾块截断)──> UB[a_ub] ──[int8: vcast→f16(+ uint8 修正)]──> UB[a_c] ─┐
  GM[b] ──T.copy(连续段, 尾块截断)──> UB[b_ub] ──[int8: vcast→f16(+ uint8 修正)]──> UB[b_c] ─┼─ T.vcmp("le") ──> UB[cond:bool]
                                                                                              │   ──vcast(bool→f16)──> UB[yf]
                                                                                              │   ──vcast(f16→i8,rint)──> UB[y8] ──T.copy──> GM[y(int8)]
uint8 无符号域修正（§0.6 R5，仅 uint8 输入，每 operand）：
  UB[a_ub(i8)] ──vcast→f16──> s ──┬─ vcmp(s,0,"lt") → m ─┐
                                   └─ vadd(s,256.0) → sp  ┼─ vselect(m, sp, s) ──> UB[a_c]（真无符号值 0..255）
更一般广播（k>1；每 task = 一行的一块）：
  (row,chunk) 标量分解 ──> a_base/b_base（编译期 unrolled divmod，仅每 task 一次）
  a 末维 stride=1: GM[a] ──T.copy(a[a_base:a_base+tail])──> UB[a_c]（或经 a_ub + 升域/修正）
  a 末维 stride=0: GM[a] ──T.copy(a[a_base:a_base+1])──> UB[a1] ──[升域/修正]──> T.vbrc──> UB[a_c]
  b 同上 ─────────────────────────────────────────────────────────────────────────────> UB[b_c]
  UB[a_c] <= UB[b_c] ──T.vcmp("le", 同形, CG-2026-0021)──> UB[cond] ──vcast 链──> UB[y8] ──T.copy──> GM[y[row*L+inner_off …]]
全路径无 L1/L0 参与（纯 Vector）；无中间 GM 往返（每 task 一次进出）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。`le` 为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用**轻量调研**深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法 / 输入公式） | `y = (a <= b)`（1 次弱序比较/元素，原生有/无符号 dtype 域）+ 广播索引 | — | — | ✅（基线） |
| 1 | 补集分解 | `a <= b ≡ ¬(a > b)` | 整数域恒等；**浮点不成立**：NaN 时 `¬(NaN>x)=True` 而 `NaN<=x=False`（`verify_equiv.py` E-NAN 4/6 角点不一致） | **负收益 + 浮点错误**：1 比较 → 1 比较 + 1 取反（2 op/元素） | ❌ 浮点不等价、op 数 ×2（§1.6.1 详证） |
| 2 | 析取分解 | `a <= b ≡ (a < b) ∨ (a == b)` | **IEEE 恒等**（含 NaN：两子式均 False → 析取 False；相等时 `a==b` True → 折取 True；E-NAN 0 违反） | **负收益**：1 比较 → 2 比较 + 1 逻辑（3 op/元素） | ❌ op 数 ×3（仅作 NaN/语义回退，§9.1 R6） |
| 3 | 操作数交换 | `a <= b ≡ b >= a` | 完全等价（含 NaN） | **无收益**：同 1 op/元素，仅交换操作数 | ❌ 无收益（仅记录；`ge` 另为独立算子） |
| 4 | 算术分解 | `a <= b ≡ (a − b) <= 0` | 整型需宽域防溢出；浮点 `a-b` 引入舍入（大数相减失真，相等附近可能因舍入翻转）→ **非严格等价** | **负收益**：2 op/元素 + 浮点失真风险 | ❌ op 数 ×2、浮点不等价 |
| 5 | 位分解 | 有符号位模式比较（符号位 + 差分） | 整数可构造但 op 数 ≥2；**浮点（NaN/±0/相等）不成立** | **负收益** | ❌ op 数 ×2+、浮点不成立 |
| 6 | min/max 分解 | `a <= b ≡ (max(a,b) == b)` | 需 NaN 传播语义对齐；仍 2 比较 + 1 逻辑 | **负收益** | ❌ op 数 ≥2 |
| 7 | **比较域重映射（升域）** | vcmp 不支持的 1 字节 dtype：`(i8/i16 a <= b) ≡ (f16/f32 域 a <= b)`；**uint8 先修正为真无符号值** | **精确等价**（int8/uint8 在 fp16、int16 在 fp32 中可精确表示且序保持；E3/E4/E5 机器验证） | **可达性 + 性能收益**：解锁 `T.vcmp`（不支持 int8/uint8），int16→fp32 规避 PL-1.11 标量化；不改变比较语义 | ✅（采纳为 R4 + R5） |

> **le 相对 ge / gt 的 R1 差异**：候选 1（补集）在 ge 为 `¬(a<b)`、在 gt 为 `¬(a<=b)`、在 le 为 `¬(a>b)`——**三者均在浮点 NaN 下不成立**（同一否决结论，但反例/形态不同）；候选 2 在 ge/le（**弱序**）为**析取**（ge: `(a>b)∨(a==b)`；le: `(a<b)∨(a==b)`）、在 gt（**严格序**）为**合取** `(a>=b)∧(a!=b)`——弱序包含相等故回退式取析取，E-NAN 均 0 违反。**le 与 ge 同为「1 op/元素的弱序基线 + 升域可达性最优」，无更优化简**。

**R2 在线算法**：**无在线变体需求（天然单遍）**。结构依据：`y_i` 仅依赖 `(a_{off_a(i)}, b_{off_b(i)})`，**无 running 统计量**（和/最大值/计数/在线 softmax 类状态）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射无跨元素依赖，单遍即天然在线」）；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径（扫描遍数=1、中间缓冲 O(block_size)、UB 驻留）在候选间无差异。

**R3 复杂度对比**（带宽受限算子——算术强度 1 比较/elem ÷ ((A+B)·E + N) 字节，fp32 下 < 0.04 op/B，**访存为主导项**；口径：A/B/N 为展平元素数，E=输入 elem_bytes，输出恒 1 字节，`A=B=N` 时同形）：

| 算法候选 | FLOPs（口径：比较/转换 op 数） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------------------|---------------|---------|-------------|---------------------|
| **基线（源算法）** | N（1 比较/元素）+ **N·O(ndim) 整数 divmod**（逐元素广播索引，非比较但同等计） | (A+B)·E + N·1 **但广播路径为逐元素标量 gather（非连续段，访存效率低）** | 1（每元素每操作数读一次写一次） | O(block_size) tile（寄存器/UB） | N/block_size 个独立 task，零跨核代价；逐元素 divmod 为标量热点 |
| **NPU 重设计（§0.6 R1+R3+R4）** | N（1 向量比较/元素）+ 2N（bool→int8 的 2 条 vcast）+ [int8/bool: 2·A_c + 2·B_c 升域 vcast] + [int16: A_c + B_c 升域 vcast] + **R·O(ndim) divmod**（行级，较基线降 ~L 倍）+ 每行 ≤2 条 vbrc 物化 | (A+B)·E + N·1（连续段；`s=0` 行内标量广播仅读 1 元素/行 + vbrc） | 1 | O(block_size)（UB：3–6 buffer，§4.5） | num_logical 个独立 task，零跨核同步；persistent 摊薄启动 |
| **NPU 重设计（含 §0.6 R5 uint8 修正）** | 上式 **+ 每 uint8 operand 3 条 f16 op**（vcmp lt + vadd + vselect；vcast 已计入升域）→ 同形 uint8 路径 ≈ 5N + 2N + R·O(ndim) | 同（全 UB 内，无额外 GM 流量） | 1 | O(block_size)（uint8 路径 +2 f16 buffer，§4.5） | 同 |
| 候选 1（补集分解） | 2N + R·O(ndim) | 同基线；**浮点语义错误** | 1 | O(block_size) | 同上 |
| 候选 2（析取分解） | 3N + R·O(ndim) | 同基线 | 1 | O(block_size) | 同上 |
| **候选 7-2D（[R_blk,L] tile，§1.6.3 备选 #4）** | 同 NPU 重设计（每 tile 内 vcmp/vcast/修正一次），但 **task 数 ÷ R_blk** → 发射项 ÷ ~R_blk | 同 NPU 重设计（b 行重读 ÷ R_blk） | 1 | O(R_blk·block_size)（UB `R_blk×L×12·1.7`） | task 数 ÷ R_blk，零跨核代价 |

基线（比较本体）已是 1 op/元素的下界；NPU 重设计在**比较本体**上无更优公式（候选 1/2/4/5/6 均 op 数 ≥2，或对浮点不成立），其收益全在**广播索引路径**（把逐元素 O(ndim) divmod/gather 降为行级 O(ndim) + innermost 维连续向量化 + 显式同形物化）、**发射项**（2-D tile 摊薄，§1.6.3）与**可达性**（升域解锁 vcmp + int16 规避标量化）。行级重读口径：`sb=1 且 b 在全部外层维广播`（如 row-broadcast `[4096,4096]^[1,4096]`）时每 task 重读一行 b，未优化口径下 b 读流量 ≈ N·E（b 仅 16KB，预期 L2 驻留，HBM 侧不放大）；该不确定项按 §1.6.3 备选 + 实验裁决处理。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` / pattern-library 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 可融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源逐元素 divmod + 标量比较 + 标量 gather） | ❌ 逐元素整数 divmod + 标量 GM gather → 落错单元（标量密集，非 Vector） | OK | ✗ 标量 gather 无向量 lane | 尾块依赖动态 shape | 单阶段 | ❌ 淘汰：NPU 无逐元素 divmod 快路径，标量 gather 严重失分 |
| **NPU 重设计（行 base + innermost 向量化 + `T.vcmp("le")` + 同形物化 + bool→int8）** | ✅ 比较 → Vector `T.vcmp("le")`（fp16/bf16/fp32/int16/32/64 同形 √）；bool→int8 → `T.vcast` 链（全 dtype √）；标量广播由 `T.vbrc` 物化 | ✅ 3–6 buffer × block_size ≤ 192KB（§4.5，含 1.7x 膨胀后仍达标） | ✅ block_size×E = 32B 整数倍；向量宽度整除（fp16×8/fp32×4/int32×4）；尾 lane 浪费 0%（L∈{3136,4096,8192…}）；1 字节升域无对齐影响 | ✅ N/L/block_size 全编译期特化；tail 仅 1 次/task | 单阶段无需流水；纯 Vector 无 CV 融合 | ✅ **选定** |
| 候选 1（补集分解） | ✅ 仍落 Vector（但浮点语义错误 + op 数 ×2） | OK | OK | OK | OK | ❌ E-NAN 实测浮点不等价（NaN） |
| 候选 2（析取分解） | ✅ IEEE 等价（NaN 一致）但 op 数 ×3 | OK | OK | OK | OK | ❌ R3 无访存收益、FLOPs ×3（保留为 NaN 回退） |
| **int16 原生 vcmp（R4 备选）** | ⚠️ **vcmp int16 被标量化**（PL-1.11：所有操作数形态一律标量指令，[32,256] 单链 236µs） | OK | OK | OK | OK | ❌ **淘汰**：性能陷阱；改升 fp32（精确、非标量化） |
| 候选 7-2D（2-D tile） | ✅ 同形 vcmp/vcast（rank-2 tile） | ⚠️ UB `R_blk×L×12×1.7`（fp32 R_blk=2,L=4096 ≈163KB <192KB ✓；int64 R_blk=2 超限） | ✅ 二维 tile 连续（a 连续、b 经 vbrc） | ✅ 静态 | 单阶段 | ⚠️ **备选**（发射项受限场景，见 §1.6.3 裁决计划） |

**调研结论**：选定 **基线比较语义（`y = (a <= b)`，1 比较/元素）的 NPU 行/内层重设计算法**（§0.6 R1：外层行标量 base + innermost 维 `T.vcmp("le")` 向量化 + `T.vbrc`/连续段显式同形物化），叠加 **§0.6 R4 比较域映射**（vcmp 不支持的 int8/uint8/bool 精确升 fp16、int16 精确升 fp32 并规避 PL-1.11 标量化）、**§0.6 R5 uint8 无符号域修正**（`<=` 排序符号相关的必需处理）与 **§0.6 R3 的 bool→int8 落盘链**。关键依据：R3 表中比较本体已是 op 数下界（1 op/元素），候选 1/2/4/5/6 均无访存收益且 op 数 ≥2（候选 1 对浮点不成立—E-NAN）；R4 中仅 NPU 重设计对 Vector/UB 容量/对齐/静态边界全部亲和（基线逐元素 divmod/gather 与标量比较在 NPU 落错单元被淘汰；int16 原生 vcmp 因标量化被淘汰）。与基线的**结构差异**：① 广播索引从「逐元素 O(ndim) divmod + 标量 gather」变为「行级 O(ndim) divmod + innermost 维连续向量化 + 显式 `T.vbrc` 同形物化（因 `T.vcmp` 无自动广播，CG-2026-0021）」；② 比较本体从「原生域标量 `<=`」变为「dtype 映射比较域 + 同形 `T.vcmp("le")`」；③ 输出从「bool 直落 GM」变为「bool → int8 0/1 承载」（bool GM store 运行期挂死，CG-2026-0022）；④ uint8 从「原生无符号比较」变为「int8 承载 + 显式无符号域修正」（R5）。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表命中行 `ALG-elementwise`（R1：公共子表达式消除——`a`/`b` 各用一次，无公共子式；除法乘倒数——无除法；cast 链合并——比较本体无 cast，仅 1 字节/int16 升域 + uint8 修正链（E5 证明修正不可省）；R2 逐元素天然单遍——与本算子一致）；② 恒等变形自查——比较的本体最短形态为 1 条 `<=`，补集/析取/算术/位/幅值分解均 op 数 ≥2 或对浮点不成立（R1 候选 1/2/4/5/6；候选 1 的浮点反例已机器记录，见 §1.6.1 E-NAN）；③ 本仓 `examples/elementwise/vec_add_1d.py`、`examples/elementwise/example_elementwise_add.py`、同族 `examples/TileOPs/tileops/kernels/elementwise/mish`、`lerp_tensor`、同族 `eq`/`ge`/`gt`/`bitwise_*` 迁移——均直排 v-prefix 单遍结构，无比较类公式变形先例；④ pattern-library `elementwise.md`/`cases.md`——elementwise 条目均为访存/布局调度类（copy-floor、loads-first、MTE3 strided、scatter-col），无比较/谓词公式级条目；⑤ 源算子结构分析（§0.3，广播路径逐元素 divmod 是唯一可优化点，已在 R1 承接）；⑥ `kb_search "vcmp bool output comparison"` 命中 CG-2026-0022（bool GM store 运行期挂死）、CG-2026-0021（vcmp 无广播）、PL-1.11（vcmp int16 标量化），确认「显式物化 + int8 承载 + 无符号修正 + int16 升 fp32」是可行路径；⑦ 浮点 NaN 语义的 `<=` 特殊性由 E-NAN 机器记录（补集不等价、析取等价）。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

**设计期 roofline 估算（D-2）**（估算下界 = max(流量项, 发射项, 容量项)；常数引用 `pattern-library/constants.md`；本机实测 `NPUUtils.get().get_aicore_num()=20` ⇒ 40 Vector 核，与 `CONST-aicore-910B2C` 的 910B2C=24/48 不同源，流量项按 40/48 比例折算并标注；block_size=4096；代表 dtype fp32）：

| 代表 workload | 流量项 | 发射项 | 容量项 | 估算下界 |
|---------------|--------|--------|--------|----------|
| smoke-1k `[1024]` fp32（N=1K，block_size=4096，chunks=1） | (N+N)·4 + N·1 = 9.2KB ÷ 1.05 TB/s ≈ **0.01µs** | 逻辑核 1 / 40 = 1 × (vcmp 1 + vcast 2) × 0.5µs ≈ **1.5µs** | 豁免（轻量调研；tile 级缓冲无跨引擎往返） | **≈1.5µs** |
| hidden-state-prefill `[2048,4096]` fp32（N=8.39M，block_size=4096，chunks=2048） | (N+N)·4 + N·1 = 75.5MB ÷ 1.05 TB/s ≈ **71.9µs** | 逻辑核 2048 / 40 ≈ 52 核内 task × 3 op × 0.5µs ≈ **78µs** | 豁免 | **≈78µs**（发射项主导） |
| cnn-feat-broadcast `[16,256,56,56]^[256,1,1]` fp32（N=12.85M，chunks=1，R=4096，末维 stride-0） | (A 51.4MB + out 12.8MB + b 行内标量 256·4B) ÷ 1.05 TB/s ≈ **61.2µs** | 4096 task / 40 ≈ 103 × (vbrc 1 + vcmp 1 + vcast 2) × 0.5µs ≈ **206µs** | 豁免 | **≈206µs**（发射项主导） |
| row-broadcast `[4096,4096]^[1,4096]` fp32（N=16.78M，chunks=1，R=4096） | (A 67.1MB + out 16.8MB + b 行 16KB) ÷ 1.05 TB/s ≈ **79.9µs**（b 行重读未计）；含 b 行重读上限口径 ≈ (67.1+16.8+67.1)MB ÷ 1.05 ≈ **143.8µs** | 4096 task / 40 ≈ 103 × 3 op × 0.5µs ≈ **155µs** | 豁免 | **≈155µs**（发射项主导）/ b 重读上限 ≈144µs |

> 引用常数：`CONST-mte2-degradation`（MTE2 有效带宽随流量退化，256M 档 3:1 R/W 混合全流量地板 ~1.26 TB/s @48 核，toolchain tilelang 0.1.2+ed787bb/910B2C/CANN 8.5.0，**kb_stale_check 标记 stale，引用前需重验**）；`CONST-vector-launch-overhead`（~0.5µs/op，toolchain 0.1.2+3a214cde/910B2C）；`CONST-copy-floor-method`（≥16M 档计算链可被 auto multi-buffer 隐藏于 MTE2 窗口，佐证发射项可与流量项同量级）。流量项按本机 40 核 / 常数 48 核折算（1.26×40/48≈1.05 TB/s），**标注为跨核数外推的未实测假设**。**发射项为本估算的主导项**（le 每 task 有 3–4 条向量 op，uint8 路径 +3 条/operand；显著高于同族位运算的 1 条）——这直接驱动 §1.6.3 的 2-D tile 备选与裁决计划。row-broadcast 的 b 行重读（未计上限 +67MB）为未实证项，交 §1.6.3 备选 + 实验裁决。容量项按轻量调研豁免（纯 Vector tile 级缓冲）。**uint8 路径说明**：上表为 fp32 代表口径；uint8 同形路径每 task ≈ 11 条向量 op（含 2×3 修正 + 2 升域 + vcmp + 2 落盘），发射项更高，但 manifest workload 仅声明 fp16/bf16/fp32，uint8 为测试覆盖 dtype（§8.3 L0-13）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的基线比较算法（比较本体不可再化简）。逐类别核查：

| # | 优化类别 | 核查结论 | 依据 |
|---|----------|----------|------|
| 1 | 恒等变形降代价（除法转乘倒数 / sqrt→rsqrt / 换底） | **无适用项**：公式为单次比较，无除法、无开方、无超越函数 | §1.3；`docs/Tilelang.language/数学操作/`（T.rsqrt/T.vexp 等均无关） |
| 2 | 稳定化变形（减 max / logsumexp 改写） | **无需求**：无指数/求和/溢出放大结构；浮点比较的 NaN/±0 由 `T.vcmp("le")` 承接 | §0.1 边界语义 |
| 3 | 公共子表达式消除 | **无公共子式**：`a`/`b` 各引用一次，无重复子表达式 | §1.3 |
| 4 | 算子降代换 | **无可降的高代价算子**：仅 1 条比较，已是 op 数下界；补集/析取/算术/位/min-max 分解均 op 数 ≥2 或对浮点不成立（§1.6.0 R1）| §1.6.0 R1 候选表；E-NAN |
| 5 | 访存量削减 | **本体已达下界**；广播路径的访存由 §0.6 R1 行/内层分解削减（连续段替换标量 gather）——属索引路径重设计，非公式变形 | §0.6 R1；§1.6.0 R3 |
| 6 | 归约结构优化 | **无归约** | §0.1 |

**优化结论：无公式级优化空间**。依据：比较本体 `a <= b` 是 1 op/元素的最短形态；无公共子表达式、无高代价算子、无除法/超越函数、无归约。**本算子特有的候选否决**（与 ge 同侧）：
- **候选 1（补集 `¬(a>b)`）否决**——**浮点不成立**：NaN 时 `¬(NaN>x)=True` 与 `NaN<=x=False` 矛盾（`verify_equiv.py` E-NAN 实测 6 个浮点角点中 4 处不一致）。这是 le（关系排序）区别于 eq（相等）的关键：`==` 的 `¬(a!=b)` 对 NaN 恰好一致，`<=` 的补集则不然。
- **候选 2（析取 `(a<b)∨(a==b)`）** 虽 IEEE 精确等价（E-NAN 0 违反），但 op 数 ×3（2 比较 + 1 逻辑）→ 无收益，仅保留为 §9.1 R6 的 NaN 回退。
- **候选 4/5/6（算术/位/min-max）** op 数 ≥2 且算术分解对浮点非严格等价 → 否决。

> 例外说明：§0.6 的 **R3（bool→int8 承载 + 输入 uint8/bool→int8）**、**R1（广播索引重映射 + 显式同形物化）**、**R4（1 字节升 fp16 / int16 升 fp32）**与 **R5（uint8 无符号域修正）** 是**语义保持的等价替换**（存储表示 / 索引表示 / 操作数形态 / 比较域），不是公式级优化，故不属本节采纳项；但均为「必须机器验证才算完成」的等价性论证，已由 `verify_equiv.py` 覆盖（下）。

**等价性机器验证（D-1）** — `examples/le/le_fwd_kernel/verify_equiv.py`（torch/numpy CPU，bit-exact / 逐点对照），结果表（本次执行于工具链 tilelang 0.1.2+417dae9908ad / CANN 8.5.0 / 本机 aicore 20）：

| 项 | 对照对象 | 用例 | 违反数 | 结论 |
|----|----------|------|--------|------|
| E1-le bool int8 vs uint8 | bool 存储（值域 {0,1}）int8 承载下 `<=` 与 uint8 一致（§0.6 R3） | 1M 随机 0/1 字节 + 25% 注入相等位 | 0 | `EQUIV_PASS` |
| E1-bool view(int8 0/1) | int8 0/1 承载 `view(bool)` 与 `torch.le`(bool) 一致（§0.6 R3 输出侧） | 同上 | 0 | `EQUIV_PASS` |
| E2-smoke-1k / hidden-state-prefill | 新行/内层映射 vs 源 divmod 映射（§0.6 R1） | `[1024]^[1024]` / `[2048,4096]^[2048,4096]`（8.39M 索引） | a=0,b=0,oob=0 | `EQUIV_PASS` |
| E2-cnn-feat-broadcast / row-broadcast | 同上 | `[16,256,56,56]^[256,1,1]`（12.85M）/ `[4096,4096]^[1,4096]`（16.78M） | 0,0,0 | `EQUIV_PASS` |
| E2-bias-add / row-broadcast-3d / scalar-broadcast | 同上 | 测试广播模式 | 0,0,0 | `EQUIV_PASS` |
| E2-a-broadcast-outer / -2d / -mid-broadcast | 同上 | a 侧广播（角色互换） | 0,0,0 | `EQUIV_PASS` |
| E3-int8-exact-in-fp16 / -uint8- | fp16 精确表示 int8/uint8 全部 256 值（§0.6 R4） | 全 256 值 cast 往返 | 0 | `EQUIV_PASS` |
| E3-int8-le-preserved / -uint8- | 升 fp16 后 `<=` 与原生 `<=` 一致（§0.6 R4） | 64K 随机对（含 16K 强制相等） | 0 | `EQUIV_PASS` |
| **E4-int16-exact-in-fp32** | int16 全部 65536 值在 fp32 精确（§0.6 R4 改进项） | 全 65536 值 cast 往返 | 0 | `EQUIV_PASS` |
| **E4-int16-le-preserved** | int16→fp32 升域后 `<=` 与原生一致（含 ±32767/−32768 角点） | 131072 随机对 + 4 角点 | 0 | `EQUIV_PASS` |
| **E5-unsigned-correct 256x256 exhaustive** | uint8 无符号域修正 `u = s + 256·(s<0)` 的 `<=` vs 真无符号 `<=`（§0.6 R5） | 全部 256×256 字节对 | 0 | `EQUIV_PASS` |
| **E5-unsigned-correct random+equal** | 同上 | 1M 随机字节 + 25% 注入相等位 | 0 | `EQUIV_PASS` |

**OVERALL: `EQUIV_PASS`**（脚本随算子目录留存，Stage 2 维度 8 可重跑）。§3.1 公式拆解以「优化后公式 = 原式」（比较本体不变、索引路径按 §0.6 R1、比较域按 R4、uint8 修正按 R5、承载按 R3）为唯一输入。

**否决证据（不进 OVERALL，供 §1.6.0 R1 候选裁决复核）**：

| 证据项 | 对照 | 用例 | 计数 | 说明 |
|--------|------|------|------|------|
| E-NAIVE | 朴素 int8 承载 `<=` vs 真无符号 `<=` | 1M 随机字节 | **523878 处不一致** | 证明 uint8 无符号域修正（R5）不可省 |
| E-NAN complement | `¬(a>b)` vs `le` | 6 浮点角点（含 NaN/±0/±Inf） | **4 处不一致** | 证明 R1 候选 1 浮点不等价 |
| E-NAN disjunction | `(a<b)∨(a==b)` vs `le` | 同上 | 0 | 证明 NaN 回退式（§9.1 R6）可用 |
| E-ZERO（弱序角点） | `+0<=-0` / `-0<=+0` / `a<=a` | 4 有符号零角点 + 64K 自比较 | **4/4 True、64K/64K True** | 证明 le 的**弱序**相等边界（与 gt 严格序相反：gt 此处全 False） |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素比较 `a <= b` | 源 `T.Parallel(threads,npt)` 标量比较循环 | `T.vcmp(a_c, b_c, cond, "le")`（同形） | ✅ | —（佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1/§2.4 示例即含比较模式族；探针同源实测） |
| 2 | bool → 1 字节 int8 落盘 | 源直接写 bool（CUDA 1 字节可落） | `T.vcast(cond→f16)` + `T.vcast(f16→i8, rint)` | ✅ | —（佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 `i1→f16`/`f16→i8`；同源探针实测 bit-correct） |
| 3 | int8/uint8/bool 比较域升 fp16 | 源原生整型标量比较 | `T.vcast(i8→f16, rint)` | ✅ | —（佐证：vcast §2.2.1 `i8→f16`/`ui8→f16`；E3 机器验证精确） |
| 4 | **int16 比较域升 fp32** | 源原生 int16 标量比较 | `T.vcast(i16→f32, rint)` | ✅ | —（佐证：vcast §2.2.1 `i16→f32`；E4 机器验证精确；规避 PL-1.11 int16 标量化） |
| 5 | **uint8 无符号域修正** | 源原生无符号比较（硬件直接支持） | `T.vcmp(s,0,"lt")` + `T.vadd(s,256.0)` + `T.vselect(m,sp,s)` | ✅ | —（佐证：vcast/vcmp/vadd/vselect 均文档化 f16 向量 op；E5 机器验证；替代不了即数值错误，E-NAIVE） |
| 6 | 行内标量广播（末维 stride=0 操作数） | 源 stride-0 索引隐式广播 | `T.vbrc(a1/b1 → a_c/b_c)`（**必需**：vcmp 无自动广播，CG-2026-0021） | ✅ | —（佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；(1,)→(block_size,) 合法；1 字节先升 f16 再 vbrc 避 int8 vbrc 挂死） |
| 7 | 行 base divmod（`outer_a_base`/`outer_b_base`） | 源 `_compute_broadcast_offsets` 逐元素 divmod | **无**（无逐元素等价向量 API；行级标量） | ❌ | **block 级索引/任务映射计算**：每 task 仅 O(ndim) 次标量整数 divmod（ndim≤4），非逐元素热点（较源降 ~L 倍，L 达 10³–10⁶）；`T.arange`+divmod 向量化会引入逐元素标量物化（`CONST-arange-scalar-materialize` ~1ns/elem）反而更慢 |
| 8 | 尾块长度 `tail = T.min(block_size, L − inner_off)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：非整除时逐 task 不同边界；已采用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`）；padding 方案违反 host 禁改输入约束（ascend-constraints.md §4） |
| 9 | 核内 task 调度循环 `for i in T.serial(num_local_tasks)` | 循环 | 无（persistent 结构需循环推进 task） | ❌ | **tile 级调度循环**（§5.5 persistent 分核标准结构，core-split-strategy.md §1 要素③；`T.vcast.md` §2.4 示例同款）；边界为编译期静态值，循环体一次一个完整 task（内部全向量化） |
| 10 | host 侧 `num_logical/num_kernels/chunks` 计算、block_size 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合 decision-tree §1） |

**向量化结论**：6 类逐元素计算点（#1 比较、#2 bool→int8 落盘、#3 1 字节升域、#4 int16 升域、#5 uint8 无符号修正、#6 标量广播物化）**已全部向量化**，替代 API 均有 docs + 机器验证双佐证；保留 4 处标量/循环（#7 行 base 索引 / #8 动态边界 / #9 tile 调度循环 / #10 host 元数据），逐项理由见上表——§6 循环结构与本结论一致（kernel 内无任何逐元素标量循环）。**注**：与位运算族相比，本算子每 task 的向量 op 数更多（vcmp + 2~4 条 vcast + uint8 时 +3 修正 op），是 §1.6.3 发射项裁决的动因。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（Op 层已把两输入展平为一维连续、输出回 broadcast shape），但**核内布局与 lane 映射是自由变量**。本节枚举候选并量化取舍。迁移任务：源 GPU 的 `threads×npt` 轴是**硬件组织的输入而非结论**（thread/warp 轴与 NPU Vector 轴不对应），须独立评估。**本算子特有约束**：`T.vcmp` 无自动广播（CG-2026-0021）→「依赖向量指令自动广播」的候选直接不可实现；输出为 1 字节且 bool GM 不可写 → 承载 dtype 是布局的组成部分；uint8 需无符号域修正 → 修正链的缓冲位置也是布局的组成部分。

**候选矩阵**（弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 展平一维（I/O 原生）+ innermost 维向量化 + 显式同形物化 + bool→int8 | 展平 N 轴（`k=1`）/ coalesced 末维（`k>1`） | 无 | 基线：连续段 `T.copy`，访存 MTE2 最优；对齐 `block_size×E` 32B 整除；UB 3–6 buffer ≤ 192KB（§4.5）；广播维用 vbrc 物化；每 task 3–4 条向量 op | ✅ **采纳**（主选） |
| 2 | 核内 N 维 auto-broadcast tile（`T.vcmp` 直接吃 `[T,d1,…,dk-1]` vs `[1,d1,…,1]`，rank 一致） | 末维/交织维 | 无 | ❌ **不可实现**：`T.vcmp` 在两操作数形状不同（`[4,1024]|[1,1024]` / `[4,1024]|[4,1]`）时 BiShengHIR `'hivm.hir.vcmp' op operands' shape are inconsistent`（CG-2026-0021）；文档相符：`T.vcmp.md` §2.2.2「输入 A 和 B 必须具有相同的 shape」 | ❌（vcmp 不支持广播） |
| 3 | host 侧预广播物化 `b` 到输出 shape | 展平 N | host `expand→contiguous` | ❌ **负收益**：违反 host 禁改输入/新增物化的约束（ascend-constraints.md §4）；额外 GM 流量 `N·E`（b 物化）+ 内存；无收益 | ❌ |
| 4 | **2-D 块化 tile** `[R_blk, L]`：a_ub/tile 连续载入；b 末维 stride=1 时 b 行[(1,L)] 载入 + `T.vbrc → b_c[R_blk,L]`（或 b 全广播时从 (R_blk,1) vbrc）；`T.vcmp([R_blk,L],[R_blk,L])` **同形**；bool→int8 同 2-D | 末维 L（+ 行轴 R_blk） | 无（a 连续） | 备选：**发射项按 R_blk 摊薄**（le 发射项主导，§1.6.0）——row-broadcast/cnn-feat 从每行 3–4 op 降为每 R_blk 行一次；b 行重读 ÷ R_blk；代价 UB `R_blk×L×(E+E+1+2+1)×1.7`（fp32 R_blk=2,L=4096 → 96KB×1.7 ≈163KB <192KB ✓；int64 R_blk=2 → 超限，须 R_blk=1 退化） | ⚠️ **备选**（`b` 全外层广播 / 发射受限场景；见裁决计划） |
| 5 | 逐元素标量 `T.Parallel` 比较（源直译） | 无向量轴 | 无 | ❌ **负收益**：NPU 无逐元素标量快路径（落 Scala 单元、无 128bit lane）；§1.6.2 #1 已判定向量化 | ❌ |
| 6 | 位运算近似（`<=` → 位模式比较 / `vxor` 链） | int 专属 | 无 | ❌ 浮点不成立（NaN/±0）；int 亦 op 数 ×2+；见 §1.6.0 R1 候选 5 | ❌ |
| 7 | uint8 修正放在 GM 往返（先修正写回 GM 再比较） | 展平 N | 额外 GM 往返 | ❌ **负收益**：违反 host 禁改输入 + 引入 `N·E` 额外 GM 往返；修正链全程 UB 内（主选 #1）| ❌ |

**轴质量评分（主选 #1）**：整除性——末维 `L` 与平铺 `block_size` 均为向量宽度整数倍（fp16×8/fp32×4/int32×4）；尾 lane 浪费 = `(ceil(L/vw)·vw − L)/L`，manifest 广播末维 `L∈{3136, 4096}`（cnn-feat 3136 / row-broadcast 4096）对 fp32×4 均整除（0% 浪费），smoke/hidden-state `L=N_total` 由 `block_size=4096` 整除；累加链——无累加（逐元素比较，无跨步串行累加维，不阻碍向量化）；repack 代价——无（直接用 I/O 原生连续段）；UB 容量——3–6 buffer × block_size ≤ 192KB（§4.5）。

**源码轴对照（迁移任务要求）**：源 `T.Parallel(threads, npt)` 的隐式轴向是「连续 N 轴 + 每线程 npt 连续元素」，与主选 #1 的「连续段 + innermost 维向量化」一致；源的**广播路径**则退化为逐元素标量索引（无向量轴），NPU 重设计的核心即把它恢复为 innermost 维向量轴 + 显式同形物化（§0.6 R1/R4）。**注意**：源 GPU 的比较是标量 `<=`，其广播索引「隐式 strides」不依赖向量指令广播能力，而 NPU `T.vcmp` 无广播，故 NPU 必须在软件层先把 operand 变同形——这是 GPU→NPU 轴映射之外的一层额外重设计；uint8 修正链（R5）同样在 UB 内按 innermost 维向量化。

**实验裁决模式（2-D 块化 tile 备选 #4）**：主选 #1 在广播类 shape（cnn-feat / row-broadcast / bias）上每 task 仅处理一行、发射项主导（§1.6.0 估算 cnn-feat 206µs、row-broadcast 155µs vs 流量项 61–80µs）；备选 #4 以 `[R_blk, L]` tile 把发射项与 b 行重读按 R_blk 摊薄。其收益依赖**未实证常数**（向量发射摊销律、二维 vbrc/vcast 的每 op 开销、b 行重读的 L2 吸收率），判定裕度不可忽略，故：
- **主选**：方案 #1（per-row，(block_size,) 1-D tile，进 Stage 3）。
- **备选**：方案 #4（2-D `[R_blk, L]` tile，仅广播类 shape；推荐 fp32/fp16/bf16 取 R_blk=2、L=4096；int64 按 R_blk=1 退化为方案 #1——容量否决；备选结构：buffer 形状 `(R_blk, L)`、a 连续段载入、b 经 `T.vbrc((1,L)→(R_blk,L))` 或 `(R_blk,1)→(R_blk,L)` 物化、`T.vcmp`/`T.vcast`/uint8 修正链在 rank-2 tile 上执行、bool→int8 同 2-D、尾块按 `tail` 截断；分核三要素按 2-D task 粒度重算 `num_logical = ceildiv(R, R_blk) × ceildiv(L, block_size)`）。
- **裁决计划**：Stage 4 在 row-broadcast（`[4096,4096]^[1,4096]` fp32）与 cnn-feat（`[16,256,56,56]^[256,1,1]` fp32）代表 shape A/B 实测方案 #1 vs #4；指标 = msprof op Task Duration + vec_ratio + mte2_active_bw；判定阈值 = 备选 latency < 主选 × 0.97 才翻转；按 shape 分派（广播类走 #4，同形走 #1）；**shape 特化工厂（lru_cache）评估**：wrapper 按 `(N_total, coalesced_shape, strides…)` 特化，天然支持按 shape 分派主选/备选，无全局二选一需求——结果回写 §1.6.3。

**布局决策结论**：选定方案 #1——核内布局 = `(block_size,)` UB buffer（输入承载 a_ub/b_ub、比较域 a_c/b_c、bool cond、f16 yf、int8 y8；标量广播源 a1/b1/a1c/b1c；uint8 修正中间量 sp/m；常量 0 显式物化 zero buffer），向量化轴 = **coalesced 末维**（`k=1` 时即展平 N 轴），repack = 无；广播索引经 §0.6 R1/R4 行/内层分解 + 显式同形物化；输出经 §0.6 R3 由 int8 承载；uint8 经 §0.6 R5 修正。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、`T.vcmp`/`T.vcast`/修正链操作数即整 buffer）与 §6 循环结构；§4.5 UB 预算按此布局计算。方案 #4 作为广播类 shape 备选 + 裁决计划随设计冻结进 Stage 4。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定；且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素比较 `T.vcmp("le")` + bool→int8 `T.vcast` 链 + 1 字节升域 + uint8 无符号修正 + 标量广播物化 `T.vbrc`），无 matmul、无规约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + vcmp + 2~5 条 cast/修正 + 一次搬出，广播路径外加 ≤2 vbrc） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 task 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | v-prefix（`vcmp`/`vcast`/`vbrc`/`vadd`/`vselect`）+ `T.copy` + `T.alloc_ub` 均为 Developer 可用形态 | `T.vcmp.md` §2.4 / `T.vcast.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态；同源探针均为 Developer 模式实测通过 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + `alloc_ub` + v-prefix，Developer） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2；非 persistent 混合算子，无 CG-2026-0010 的 Developer/persistent+gemm 限制，本算子纯 Vector persistent）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix 要求操作数在 UB，`T.vcmp.md` §2.1 / `T.vcast.md` §2.3） |
| 计算方式 | v-prefix 向量 API（`T.vcmp` + `T.vcast` + `T.vbrc` + uint8 修正链 `T.vcmp`/`T.vadd`/`T.vselect`），无 `T.Parallel` 逐元素标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式（比较本体不变）：`y = (a <= b)`；索引映射按 §0.6 R1（行 base + innermost 维向量化）；比较域按 §0.6 R4（dtype 映射，1 字节升 f16、int16 升 f32）；uint8 无符号修正按 §0.6 R5；operand 形态按 §0.6 R1（显式同形物化，因 vcmp 无广播）；存储按 §0.6 R3（输出 bool→int8；输入 uint8/bool→int8）。拆解与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(row,chunk) = decompose(task)`；`a_base`, `b_base` | 行/块标量分解（编译期 unrolled divmod，仅每 task 一次） |
| 2 | `a_c ← cast(a[a_base : a_base+tail])`（末维 stride=1）或 `a_c ← vbrc(cast(a[a_base]))`（末维 stride=0） | a operand 物化为 (block_size,)（连续段或标量广播；1 字节经 fp16 升域、int16 经 fp32 升域） |
| 3 | `a_c ← ufix(a_c)`（仅 uint8） | 无符号域修正 `u = s + 256·(s<0)`（R5） |
| 4 | `b_c ←` 同步骤 2–3 | b operand 物化为 (block_size,)（同上） |
| 5 | `cond = (a_c <= b_c)` | 逐元素比较（向量，两 operand **同形**，输出片内 bool） |
| 6 | `y8 = cast(cast(cond → f16) → i8)` | bool → int8 0/1（片内；bool 不可落 GM） |
| 7 | `y[flat0 : flat0+tail] ← y8[0:tail]` | 结果写回（连续段，尾块截断） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| task 分派 | `task = i*num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local_tasks)` | num_kernels/num_local_tasks 编译期常量（§5.5） | Developer |
| 行 base | `Σ coord_i·stride_i` | Python 生成的 `T.floordiv`/`T.floormod` 展开链 | 编译期常量 divisors/strides | Developer |
| operand 搬入（连续段） | `x[a_base:a_base+tail] ↔ x_c[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（runtime 边界 OK；`T.copy.md`；`vec_add_1d.py` L34–L38） | Developer |
| 1 字节升域 | `f16(int8 x)` | `T.vcast(x_ub, x_c, round_mode="rint")` | int8/uint8/bool 输入（`T.vcast.md` §2.2.1 `i8→f16`/`ui8→f16`） | Developer |
| int16 升域 | `f32(int16 x)` | `T.vcast(x_ub, x_c, round_mode="rint")` | int16 输入（`T.vcast.md` §2.2.1 `i16→f32`；规避 PL-1.11 标量化） | Developer |
| **uint8 无符号修正** | `u = s + 256·(s<0)` | `T.vcmp(s, zero_f16, m, "lt")` + `T.vadd(s, 256.0, sp)` + `T.vselect(m, sp, s, u)` | 仅 uint8 输入（`T.vcmp.md`/`T.vadd.md`/`T.vselect.md`；E5 机器验证；`zero_f16` 须显式物化为 (block_size,) 同形 buffer——vcmp 无标量操作数形式，§9.1 R15） | Developer |
| 标量广播物化 | `x_c ← vbrc(x1)` | `T.vbrc(x1, x_c)` | src (1,)→dst (block_size,)（`T.vbrc.md` §2.2.2；1 字节路径先升 f16 再 vbrc） | Developer |
| 尾块长度 | `tail = min(block_size, L − inner_off)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md` | Developer |
| 逐元素小于等于比较 | `cond = (a_c <= b_c)` | `T.vcmp(a_c, b_c, cond_ub, "le")`（`npuir_cmp`） | **两 operand 严格同形**（`T.vcmp.md` §2.2.2；CG-2026-0021）；输出 cond_ub 为片内 bool；dtype fp16/bf16/fp32/int16/32/64 | Developer |
| bool→int8（0/1） | `y8 = int8(cond)` | `T.vcast(cond_ub, y_f16)` + `T.vcast(y_f16, y_i8, round_mode="rint")` | `T.vcast.md` §2.2.1 `i1→f16` + `f16→i8` | Developer |
| 整除向上取整 | `ceildiv(L, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板 §3.3 规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字，避免 false alarm）。工厂签名保留源契约（wrapper 已冻结调用形态），下半段为**示意**（真实实现由 JIT 参数编译期确定，`if sa==1` / `if upcast` / `if need_ufix` 为 Python 层静态分派，不产生运行时分支）。block_size 相关分解**必须在 `@tilelang.jit` 函数体内**用被调用 block_size 计算（bitwise_and Stage 3 TRAP-jit-blocksize-decomposition；`T.ceildiv` 在 trace 期折叠）。

```python
def le_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides,
                  a_numel, b_numel, output_dtype=None, block_size=4096):
    # dtype = 承载 dtype（bool/uint8 由 Op 层 view 为 int8；int8/uint8/bool 走 f16 域，int16 走 f32 域）
    out_dtype = "int8"                      # bool 输出以 int8 0/1 承载（§0.6 R3）
    need_ufix = (dtype == "int8" and is_unsigned_operand)   # uint8 承载标记（Op 层传入）
    if dtype == "int16":
        cmp_dtype = "float32"
    elif dtype == "int8":
        cmp_dtype = "float16"               # int8/uint8/bool
    else:
        cmp_dtype = dtype                   # fp16/bf16/fp32/int32/int64 原生域
    upcast = (cmp_dtype != dtype)

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        vector_cores = NPUUtils.get().get_aicore_num() * 2   # 纯 Vector 翻倍；实查见 §5.5
        k = len(coalesced_shape)
        L = coalesced_shape[-1]                              # innermost 维
        outer = list(coalesced_shape[:-1])                   # 外层维
        R = math.prod(outer) if k > 1 else 1
        sa, sb = a_strides[-1], b_strides[-1]                # 末维 stride ∈ {0,1}
        chunks = T.ceildiv(L, block_size)                    # trace 期折叠
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)         # persistent 分核（§5.5）
        num_local = T.ceildiv(num_logical, num_kernels)      # 静态边界

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),                  # 承载 dtype
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),              # int8（Op 层 view(bool)）
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_size,), dtype)      # 仅 upcast 时用（DCE 掉）
                b_ub = T.alloc_ub((block_size,), dtype)
                a_c = T.alloc_ub((block_size,), cmp_dtype)   # 比较域 operand（同形）
                b_c = T.alloc_ub((block_size,), cmp_dtype)
                a1 = T.alloc_ub((1,), dtype)                 # 标量广播源
                b1 = T.alloc_ub((1,), dtype)
                a1c = T.alloc_ub((1,), cmp_dtype)
                b1c = T.alloc_ub((1,), cmp_dtype)
                zero = T.alloc_ub((block_size,), cmp_dtype)  # 常量 0（vcmp 无标量操作数，§9.1 R15）
                m = T.alloc_ub((block_size,), "bool")        # uint8 修正掩码（R5，仅 uint8）
                sp = T.alloc_ub((block_size,), cmp_dtype)    # s + 256（R5，仅 uint8）
                cond = T.alloc_ub((block_size,), "bool")     # 片内 bool
                yf = T.alloc_ub((block_size,), "float16")
                y8 = T.alloc_ub((block_size,), "int8")
                for i in T.serial(num_local):                # 核内串行，静态边界
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        a_base = <outer divmod a_strides>(row) + (inner_off if sa == 1 else 0)
                        b_base = <outer divmod b_strides>(row) + (inner_off if sb == 1 else 0)
                        tail = T.min(block_size, L - inner_off)
                        # --- 物化 a 为 (block_size,) 比较域（R1：vcmp 要求同形）---
                        if sa == 1:
                            if upcast:
                                T.copy(a[a_base:a_base+tail], a_ub[0:tail])
                                T.vcast(a_ub, a_c, round_mode="rint")
                            else:
                                T.copy(a[a_base:a_base+tail], a_c[0:tail])
                        else:                                # a 行内标量广播
                            T.copy(a[a_base:a_base+1], a1[0:1])
                            if upcast:
                                T.vcast(a1, a1c, round_mode="rint")
                                T.vbrc(a1c, a_c)             # f16/f32 域 vbrc（避 int8 vbrc 挂死）
                            else:
                                T.vbrc(a1, a_c)
                        # --- uint8 无符号域修正（R5，仅 uint8；在 f16 域）---
                        if need_ufix:
                            T.vcmp(a_c, zero, m, "lt")       # zero 显式 (block_size,) 同形
                            T.vadd(a_c, 256.0, sp)
                            T.vselect(m, sp, a_c, a_c)       # u = s + 256·(s<0)
                        # --- 物化 b 为 (block_size,) 比较域 ---
                        if sb == 1:
                            if upcast:
                                T.copy(b[b_base:b_base+tail], b_ub[0:tail])
                                T.vcast(b_ub, b_c, round_mode="rint")
                            else:
                                T.copy(b[b_base:b_base+tail], b_c[0:tail])
                        else:
                            T.copy(b[b_base:b_base+1], b1[0:1])
                            if upcast:
                                T.vcast(b1, b1c, round_mode="rint")
                                T.vbrc(b1c, b_c)
                            else:
                                T.vbrc(b1, b_c)
                        if need_ufix:
                            T.vcmp(b_c, zero, m, "lt")
                            T.vadd(b_c, 256.0, sp)
                            T.vselect(m, sp, b_c, b_c)
                        # --- 同形向量比较 + bool→int8 落盘 & 写回 ---
                        T.vcmp(a_c, b_c, cond, "le")
                        T.vcast(cond, yf)                    # bool -> f16 (1.0/0.0)
                        T.vcast(yf, y8, round_mode="rint")   # f16 -> int8 (1/0)
                        flat0 = row * L + inner_off
                        T.copy(y8[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel
```

> 实现注记：① `<outer divmod …>` 为 Python 层按 `outer`/strides 生成的 `T.floordiv`/`T.floormod` 展开链（因子全为编译期常量）；`k==1` 时无外层维，`a_base=inner_off`、`b_base=inner_off`。② `sa==0` 且 `sb==0` 仅在全标量（`L==1`）出现，此时两路都走 `vbrc`（`L==1`/`block_size>1` 不触发 `TRAP-vbrc-same-shape-empty-broadcast`）。③ **`T.vcmp` 两操作数必须同形**（CG-2026-0021）——广播维必须先经 `vbrc`/连续段物化；**常量比较操作数（uint8 修正的 0）必须先物化为同形 `zero` buffer**（vcmp 无标量操作数形式，§9.1 R15）。④ v-prefix 整 buffer 操作在尾块（`tail<block_size`）时对 `[tail:]` 段垃圾一并计算，但仅 `[0:tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例同款）。⑤ bool/uint8 输入由 Op 层 `view(torch.int8)` 后传入；输出 `y` 为 int8 0/1，Op 层 `view(torch.bool)`（§0.6 R3）。⑥ 1 字节输入在 stride-0 广播时**先升 f16 再 vbrc**（规避 int8 vbrc dst 运行期挂死，同时满足 vcmp 的 int8 不支持）。⑦ **uint8 修正链在 f16 域、物化之后、vcmp 之前**（R5）；`need_ufix` 为 Python 编译期布尔（仅 uint8 承载置真），非运行时分支；修正用原地 `vselect(m, sp, a_c, a_c)`（`T.vsub.md` §2.3 允许 dst 与 src 别名，但列广播性能税见 §9.1 R8，Stage 3 可改独立 dst）。⑧ block_size 分解在 jit 函数体内计算（见上）。⑨ `is_unsigned_operand` 由 Op 层传入（uint8 承载标记）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.vcmp`（`npuir_cmp`） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.1/§2.2.2/§2.4 | ✅ **同源探针实测**：fp16/bf16/fp32/int16/int32/int64 **同形**比较 bit-correct；模式含 `"le"`（§2.1）；**形状不一致 ✗（CG-2026-0021）、int8/uint8 ✗（文档 §2.2.1 矩阵）、输出 bool 不可落 GM（CG-2026-0022）、int16 标量化（PL-1.11）** |
| `T.vcast`（`npuir_cast`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.3/§2.4 | ✅ `i1→f16`、`f16→i8(rint)`、`i8→f16(rint)`、`ui8→f16(rint)`、`i16→f32(rint)` 均在文档 dtype 矩阵；同源探针 + `verify_equiv.py` E3/E4 bit-correct |
| `T.vbrc`（`npuir_brc`） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2/§2.3 | ✅ (1,)→(block_size,) 标量广播；**int8 dst 运行期挂死（同族 ge/gt 设计期探针记录），故 int8 路径先升 f16 再 vbrc** |
| `T.vadd`（`npuir_add`） | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1/§2.3 | ✅ f16 标量 `+256.0`（scalar src1 文档合法；int8 ×，故修正链在 f16 域） |
| `T.vselect` | `docs/Tilelang.language/条件操作/T.vselect.md` §2.1/§2.4 | ✅ 条件选择（uint8 无符号修正链；`T.vcast.md` §2.4 亦以 vselect 配合 vcmp） |
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vcmp.md` §2.4 示例 | ✅（examples 先例） |
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
| 物理核数限制（分核三要素） | Yes（逻辑核 1–4096 vs 物理 Vector 核 40） | §5.5 persistent 分核 |
| Host 侧输入操作约束 | Yes（Op 层展平输入；b 广播/uint8 修正均不做 host 物化） | 展平在 Op 层（既有契约）；uint8 修正在 kernel 内（§0.6 R5）；kernel 内 host 侧零数据操作，仅元数据 → 符合 ascend-constraints.md §4 |
| uint8 张量不可用 | Yes（uint8 输入路径） | §0.6 R3 改 int8 承载 + §0.6 R5 无符号域修正（CG-2026-0018） |
| **输出 bool 不可落 GM** | Yes（输出契约即 bool） | §0.6 R3：int8 0/1 承载 + Op 层 `view(bool)`（CG-2026-0022：编译成功但运行期挂死） |
| **`T.vcmp` 不接受任何自动广播** | Yes（广播路径核心约束） | §0.6 R1：显式 `T.vbrc`/连续段把两 operand 物化为同形（CG-2026-0021） |
| **`T.vcmp` 不支持 int8/uint8** | Yes（int8/uint8/bool 输入） | §0.6 R4：精确升 fp16（E3 机器验证）；uint8 叠加 §0.6 R5 |
| **`T.vcmp` int16 标量化** | Yes（int16 输入） | §0.6 R4：精确升 fp32（E4 机器验证；PL-1.11） |
| **`<=` 排序符号相关** | Yes（uint8 输入） | §0.6 R5：无符号域修正（E5 机器验证；E-NAIVE 证明不可省） |
| **`T.vcmp` 无标量操作数形式** | Yes（uint8 修正与常量 0 比较） | 常量 0 显式物化为同形 `zero` buffer（§9.1 R15；`T.vcmp.md` §2.2.2） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R2 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1–#6 |
| 比较本体 | 原生域标量 `<=`（输出 bool） | `T.vcmp("le")` + dtype 映射比较域 | §0.6 R4 |
| **无符号 uint8 序** | 原生无符号比较（硬件直接支持） | int8 承载 + 显式无符号域修正 `u=s+256·(s<0)` | §0.6 R5 |
| int16 比较 | 原生 int16 标量比较 | `vcast(i16→f32)` 后 `T.vcmp`（避标量化） | §0.6 R4（PL-1.11） |
| 广播索引 | 逐元素 unrolled divmod + 标量 gather（隐式 strides） | 行级 unrolled divmod + innermost 维向量化 + **显式同形物化** | §0.6 R1 |
| 广播语义承载 | 向量/标量索引隐式 | `T.vcmp` 无广播 → `T.vbrc`/连续段显式物化 | §0.6 R1（CG-2026-0021） |
| 输出承载 | bool 直落 GM（1 字节） | int8 0/1 承载 + Op 层 `view(bool)` | §0.6 R3（CG-2026-0022） |
| bool 输入承载 | `view(uint8)` + `LeBoolStorageFwdKernel` | `view(int8)`（值域 {0,1}，位级等价，E1） | §0.6 R3（CG-2026-0018） |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(num_logical, 40)` | §0.6 R2 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[2])` | `@tilelang.jit(out_idx=[2], target="npuir")` | K1 适配 |
| 工厂参数 | `(…, threads=256, num_per_thread=8)` | `(…)`；callable(`block_size`) | wrapper K11（接口契约冻结） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/ge/ge_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md` | 高度相似（**同族弱序比较**，输出 bool、`T.vcmp` 无广播、同一 §0.6/§1.6 结构） | §0 迁移分析组织、persistent 分核、§1.6 调研/优化分析结构、L0 测试结构、wrapper 契约、bool→int8 承载与显式同形物化、uint8 无符号域修正（**差异：le 比较模式 `"le"` 与补集/NaN 回退式方向相反**） |
| `examples/gt/gt_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md` | 高度相似（**有序比较严格序对照**） | 严格序 vs 弱序的 NaN 回退式差异（合取 vs 析取）、对/错相等边界（gt 的 L0-9/L0-10 对照点）、verify_equiv E-blocks 须按关系重派生 |
| `examples/eq/eq_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md` | 高度相似（同族比较类唯一先例） | §0 迁移分析组织、persistent 分核、bool/i1 承载、`T.vcmp` 无广播、int16 标量化三教训 |
| `examples/bitwise_xor/bitwise_xor_fwd_kernel/DESIGN.md` | 高度相似（同族 elementwise 广播迁移先例，含已踩坑） | §0 迁移分析组织、persistent 分核、vbrc int8 挂死绕法、行 base helper 形态 |
| `examples/bitwise_and/bitwise_and_fwd_kernel/DESIGN.md` + `RETROSPECTIVE.md` | 高度相似（迁移先例 + vbrc int8 挂死/uint8 不可用教训） | R3 int8 承载、vbrc int8 挂死绕法（升 f16）、工厂 block_size 分解陷阱 |
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail = T.min(...)` 尾块模式、runtime 边界 `T.copy`、v-prefix 用法 |
| `examples/flash_qla/tilelang_get_warmup_chunks.py` | 中度相似（vcmp/vselect 生成 0/1 语义） | `T.vcmp(...)` + `T.vselect(cond, one, zero, out)` 生成布尔掩码（印证 vcmp→1 字节输出的可行形态） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype（kernel 承载） | 说明 |
|--------|-------|---------------------|------|
| `a` | `(a_numel,)` | 承载 dtype：float16/bfloat16/float32/int16/int32/int64/int8；bool/uint8 经 Op 层 `view(int8)` | Op 层已展平的 1-D 连续张量；广播时 `a_numel ≤ N_total` |
| `b` | `(b_numel,)` | same_as(a) | 同上 |

> **dtype 覆盖（对照 manifest signature）**：manifest `LeFwdOp` inputs 声明 `bool | uint8 | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32`——**全部**在 §0.6 R3/R4/R5 中给出等价承载/比较域/修正路径（bool→int8 承载；uint8→int8 承载 + 无符号修正；int8/uint8/bool→升 fp16；int16→升 fp32），无遗漏（对照 bitwise_and RETROSPECTIVE 的「能力缺口 dtype 覆盖盲区」教训）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N_total,)` | **int8（0/1）** | `out_idx=[2]`；Op 层先 `reshape(out_shape)` 再 `view(torch.bool)` 还原 bool（§0.6 R3） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a_ub` / `b_ub` | `(block_size,)` | 承载 dtype | UB | 仅 upcast（int8/int16）路径：连续段载入的原始 1/2 字节 operand |
| `a_c` / `b_c` | `(block_size,)` | 比较域 dtype | UB | 比较 operand（连续段直载或经 upcast/vbrc，恒为 `(block_size,)` 同形） |
| `zero` | `(block_size,)` | 比较域 dtype（f16） | UB | uint8 修正掩码的常量 0 操作数（vcmp 无标量操作数，须同形物化；仅 uint8） |
| `m` | `(block_size,)` | bool | UB | uint8 无符号修正掩码（`s<0`，§0.6 R5；仅 uint8） |
| `sp` | `(block_size,)` | 比较域 dtype（f16） | UB | uint8 修正中间量 `s + 256`（§0.6 R5；仅 uint8） |
| `cond` | `(block_size,)` | bool | UB | `T.vcmp("le")` 的片内 bool 结果（**不可落 GM**，CG-2026-0022） |
| `yf` | `(block_size,)` | float16 | UB | bool→int8 中转（`i1→f16`） |
| `y8` | `(block_size,)` | int8 | UB | 输出行（0/1） |
| `a1` / `b1` / `a1c` / `b1c`（仅末维 stride=0 时用） | `(1,)` | 承载 / 比较域 dtype | UB | 标量广播源（`vbrc` 输入；1 字节路径先升 f16 再 vbrc） |

> buffer 复用：`a_ub`/`b_ub` 仅 upcast 路径分配（Python 编译期分支，否则 DCE）；`zero`/`m`/`sp` 仅 uint8 路径分配（DCE）；`a_c`/`yf`/`y8` 活区间互不相交时可合并（Stage 3 视分配情况）。**注意**：`T.vcmp` 要求「A、B 与 Cond 同形」，故 `a_c`/`b_c`/`cond` 必须同为 `(block_size,)`——不能把某 operand 留成小形状（与位运算族 vand/vor 的自动广播不同）。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 task）：
  GM[a] ──T.copy(连续段或 1 元素)──> UB[a_ub/a1] ──[int8/int16: vcast 升域]──> UB[a_c] ─┐
  GM[b] ──T.copy(连续段或 1 元素)──> UB[b_ub/b1] ──[int8/int16: vcast 升域]──> UB[b_c] ─┼─ T.vcmp(同形,"le") ──> UB[cond:bool]
                                                                                        │  ──vcast(bool→f16)──> UB[yf]
                                                                                        │  ──vcast(f16→i8)──> UB[y8] ──T.copy(连续段)──> GM[y(int8)]
（末维 stride=0 的 operand 先经 T.vbrc 物化；uint8 在 f16 域叠加无符号修正 zero/m/sp → vselect；
 全程 UB 内，无 L1/L0 参与，无中间 GM 往返）
行 base 标量：仅每 task 一次 divmod（不在 GM 数据路径上）
```

### 4.5 UB 内存预算

block_size = 4096（推荐，见 §5.2）。按各 dtype 路径的实测/上界 buffer 数量与字节/元素（`×1.7` auto-multi-buffer 膨胀上界，`CONST-capacity-910B2C`）：

| dtype 路径 | 活跃 buffer（字节/元素） | 手工合计 @4096 | ×1.7 膨胀 | 目标 UB | 余量 |
|-----------|------------------------|----------------|-----------|---------|------|
| fp16 / bf16 | a_c(2)+b_c(2)+cond(1)+yf(2)+y8(1) = **8** | 32 KB | ~55 KB | 192 KB | ✓ |
| fp32 / int32 | 同构 = 4+4+1+2+1 = **12** | 48 KB | ~82 KB | 192 KB | ✓ |
| int64 | 8+8+1+2+1 = **20** | 80 KB | ~136 KB | 192 KB | ✓ |
| int16（升 f32） | a_ub(2)+b_ub(2)+a_c(4)+b_c(4)+cond(1)+yf(2)+y8(1) = **16** | 64 KB | ~109 KB | 192 KB | ✓ |
| int8 / bool | a_ub(1)+b_ub(1)+a_c(2)+b_c(2)+cond(1)+yf(2)+y8(1) = **10** | 40 KB | ~68 KB | 192 KB | ✓ |
| **uint8** | int8 路径 + zero(2)+m(1)+sp(2) = **15** | 60 KB | ~102 KB | 192 KB | ✓ |

> 膨胀系数引用 `CONST-capacity-910B2C`（auto-multi-buffer ~1.7x；该条目另有「显式 alloc 结构 ~1.12x / 反例 0.981x」的结构依赖记录，本设计取保守 1.7x 上界）。block_size 上限（`floor(192KB / (bytes_per_elem × 1.7))`，忽略 `(1,)` 标量 buffer）：fp16/bf16 ≤ 14456、fp32/int32 ≤ 9637、int64 ≤ 5782、int16 ≤ 7228、int8/bool ≤ 11565、uint8 ≤ 7529——推荐值 4096 对全部 dtype 达标（int64 最紧：80KB×1.7≈136KB < 192KB）；int64 因上限 5782 不能取 8192，如需更大 block 须按 §1.6.3 备选随 dtype 分档。block_size 由 wrapper config 传入（`LeFwdKernel._default_block_size`：浮点 1024 / 整数 2048 / bool 存储 4096），本设计推荐 4096（理由见 §5.2；Stage 4 可扫）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 无 TIR 动态符号——工厂闭包按 `(N_total, dtype, coalesced_shape, strides, a_numel, b_numel)` **编译期特化**（源同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例触发一次 JIT 特化；kernel 边界全静态 |

> `L`、`R`、`chunks`、`num_logical`、`num_kernels`、`num_local` 全为 Python int / trace 期常量；kernel 内唯一运行时标量为 `tail`（1 次/task）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
# out_idx=[2]：main(a, b, y) 的 y（index 2）为输出，承接源算子契约（_le_fwd_kernels.py L48/L71）
# target="npuir"：K1 适配（GPU 版无 target；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素比较（1 条 `T.vcmp("le")`）+ bool→int8（2 条 `T.vcast`）+ upcast（1~2 条 `T.vcast`）+ uint8 无符号修正（`T.vcmp`/`T.vadd`/`T.vselect`）+ 标量广播物化（`T.vbrc`），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（decision-tree §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_size = 4096   # 推荐；选择理由：
# ① innermost 维单块覆盖：manifest 广播行 L ∈ {3136, 4096}（cnn-feat 3136 / row-broadcast 4096）
#    对 block_size=4096 均为 chunks=1（L≤4096）——消除行内二次分块，
#    使 vbrc/vcmp/vcast 发射次数减半
# ② 尾轴 32B 对齐：4096×{1,2,4,8}B = 4K/8K/16K/32K，均 32 整除 ✓
# ③ 向量宽度整除：fp16×8/fp32×4/int32×4 均整除 4096 → 尾 lane 浪费 0%
# ④ UB 预算：全部 dtype 路径 ≤136KB（int64）< 192KB ✓（§4.5）
# ⑤ 同形路径（L=N_total 极大）由 chunks 分块，block_size 即 tile 粒度
# ⑥ 与同族 eq/ge/gt/bitwise 迁移一致（4096），便于复用与对比；更大 block（8192）可摊薄
#    发射项（§1.6.0 发射主导），交 Stage 4 按 dtype 扫（受 §4.5 上限约束）
L = coalesced_shape[-1]
R = prod(coalesced_shape[:-1]) if k > 1 else 1
chunks = ceildiv(L, block_size)
num_logical = R * chunks          # 逻辑 task 数（一维划分）
```

一维划分（无 block_M/block_N 之分）——本算子为逐元素/广播，GEMM 类分形/非整除策略不适用。**分类倾向**：`k==1`（同形 / 全标量广播）退化为展平 N 轴分块；`k>1`（含广播）走行/内层分块。**2-D tile 备选（§1.6.3 #4）** 另按 `num_logical = ceildiv(R, R_blk) × ceildiv(L, block_size)` 划分（随裁决计划进 Stage 4，不改变主选 §3.3/§6）。

### 5.3 约束分析

- **对齐约束**: `block_size×E` = 4K/8K/16K/32K（32B 的整数倍）✓；末维向量维整除（尾 lane 0%）✓；尾块 `tail < block_size` 段长度非 32B 倍数时的拷贝行为依赖 `vec_add_1d.py` 先例（风险 §9.1 R9，L0-6 覆盖）
- **UB 容量**: 见 §4.5，全 dtype 路径 ×1.7 膨胀后 ≤136KB ≤ 192KB ✓
- **L0/L1 容量**: 不适用（纯 Vector，不使用 L0/L1）
- **分形限制**: 不适用（无 Cube）
- **同形约束**: `T.vcmp` 要求 A/B 同形（CG-2026-0021），`a_c`/`b_c`/`cond` 恒为 `(block_size,)`，§3.3 与 §4.3 一致；常量 0 操作数须显式物化 `zero`
- **承载 dtype 约束**: 输出 int8（0/1）；输入 bool/uint8→int8；1 字节路径在 stride-0 广播时先升 f16 再 vbrc（避 int8 vbrc 挂死）；uint8 在 f16 域叠加无符号修正
- **比较域约束**: int16 升 fp32（避 PL-1.11 标量化）；int32/int64 保持原生域（fp32 不精确），其 vcmp 标量化风险 §9.1 R7 待确认

### 5.4 注意事项

- **非整除处理**：尾块 `tail = T.min(block_size, L − inner_off)` 截断（§6.4）；manifest 各 workload 的 `L` 对 block_size=4096 要么整除（L=4096 / L=N_total 由 chunks 整除）要么单块尾段（L=3136），尾块路径服务任意广播形状
- **`chunks>1` 的行**：`L>4096`（如 hidden-state L=8.39M → chunks=2048）→ 同行多 chunk 共享同一 `a_base` 外层项、`inner_off` 不同（Python/常量算）
- **广播物化**：末维 stride=0 的 operand 每 task 走 1 次 vbrc（§0.6 R1）；1 字节路径先升 f16 再 vbrc
- **uint8 修正**：仅在 uint8 承载时激活（编译期布尔），每 operand +3 条 f16 op（§0.6 R5）
- **发射项**：le 每 task 3–4 条向量 op（vs 位运算族 1 条；uint8 路径约 11 条），广播类 workload 发射项主导（§1.6.0）→ §1.6.3 备选 #4 + Stage 4 裁决

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
| 元素级（比较） | 向量化 | `T.vcmp("le")`（整 buffer 一条向量指令，**同形**） | §1.6.2 #1；无 `T.Parallel` 逐元素循环（与源差异：§0.6 R2） |
| 元素级（bool→int8） | 向量化 | `T.vcast` ×2（整 buffer） | §1.6.2 #2 |
| 元素级（升域） | 向量化 | `T.vcast`（整 buffer） | §1.6.2 #3/#4（int8/uint8/bool→f16；int16→f32） |
| 元素级（uint8 无符号修正） | 向量化 | `T.vcmp` + `T.vadd` + `T.vselect`（整 buffer） | §1.6.2 #5（仅 uint8） |
| 标量广播 | 向量化 | `T.vbrc`（1 元素 → block_size） | §1.6.2 #6（末维 stride=0 时，vcmp 无广播故必需） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 核间：persistent ≤40 核
    a_ub, b_ub, a_c, b_c, a1, b1, a1c, b1c, zero, m, sp, cond, yf, y8 = T.alloc_ub(...)  # §3.3 完整版
    for i in T.serial(num_local):                            # 核内：静态边界串行
        task = i * num_kernels + cid
        if task < num_logical:
            row = task // chunks; inner_off = (task % chunks) * block_size
            a_base, b_base = <行 base 标量>; tail = T.min(block_size, L - inner_off)
            # 物化（连续段 / vbrc 标量广播；升域）→ [uint8 修正] → T.vcmp("le") → vcast 链 → 写回
            ...
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：单阶段 load→compute→store 算子（每 task 一次进出），无 K 维迭代累加结构可供双缓冲；逐元素比较计算量极小、访存主导，`auto multi-buffer` 可将向量 op 隐藏于 MTE2 窗口内（`CONST-copy-floor-method`：≥16M 档计算链被完全隐藏）。但**本算子每 task 向量 op 数（3–4；uint8 约 11）高于位运算族（1）**，发射项在广播类 workload 可能主导（§1.6.0）——多 task 双缓冲与 2-D tile（§1.6.3 #4）属参数/结构级调优，交 Stage 4 按 msprof 评估。

### 6.4 尾块处理

`L % block_size ≠ 0` 时（任意广播形状；block_size=4096 下 manifest 仅 L=3136 触发单块尾段）：

```python
inner_off = (task % chunks) * block_size
tail = T.min(block_size, L - inner_off)      # 本块有效长度
T.copy(a[a_base : a_base + tail], a_c[0:tail])     # 零起点 dst 切片（规避 UB 非零起点对齐）
...（v-prefix 整 buffer 操作；UB 中 [tail:] 陈旧段不落 GM）
T.copy(y8[0:tail], y[flat0 : flat0 + tail])
```

模式来源：`examples/elementwise/vec_add_1d.py` L31–L38（官方已验证形态）。不采用 host 侧 padding（违反 host 禁改输入约束，ascend-constraints.md §4）；不采用 Kernel 动态 block（N/L 已编译期特化，`tail` 标量一次/task 开销可忽略）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

无手动同步点。依据：① 每 task 计算完全独立、无核间数据依赖（无 workspace、无跨核归约）→ 不需要 `T.sync_block_set/wait`；② task 内 `T.copy → T.vcast/T.vbrc/T.vcmp/T.vselect → T.copy` 的顺序依赖由 Developer 模式自动同步保证（`T.vcmp.md`/`T.vcast.md`/`T.vbrc.md`/`T.vselect.md` 均为无显式同步示例）；③ persistent 核内 `T.serial` 顺序执行天然串行；④ `if task < num_logical` 无跨 task 数据交换。**CV 融合设计不适用**（纯 Vector，无 Cube 参与，无跨引擎流水）。

### 7.3 pass_configs 配置

无特殊 pass 配置（默认管线）。环境变量：`TILELANG_ASCEND_MODE=Developer`（§2.3）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现 `torch.le`），**不复刻** §0.6 的 NPU 算法（不复用行/内层索引、vcmp、vcast 链、uint8 修正、persistent 结构、同形物化）——保证验证独立性。

```python
import torch

def golden_le(a, b):
    """Less-equal reference (source-operator semantics, direct broadcast).

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_le.py：
    ref_program = torch.le(a, b)；bool 按精确比对 _bool_compare）。输入为原始
    broadcast 形状（非展平），输出 = broadcast_shapes(a.shape, b.shape)，dtype bool。
    浮点按 IEEE 语义（任一操作数 NaN → False；+0.0 <= -0.0 与 -0.0 <= +0.0 均 True
    ——弱序包含相等），与 torch.le 一致；整数按原生（含无符号）值序。
    """
    return torch.le(a, b)
```

### 8.2 精度标准

比较结果在 bool 域**精确**（无舍入），容差 = **exact**：

| dtype | 比对方式 | 说明 |
|-------|----------|------|
| float16 / bfloat16 / float32 | exact（`torch.equal` / `_bool_compare`） | 比较结果为 bool；NPU 与 golden 应逐元素一致（含 NaN/±0 角点） |
| int32 / int64 | exact | 原生域 vcmp；二补码极值覆盖 |
| int16 | exact | 升 fp32 后 vcmp（精确，E4）；±32768/32767 角点 |
| int8 | exact | 升 fp16 后 vcmp（精确，E3） |
| **uint8** | exact | int8 承载 + 无符号域修正后升 fp16 vcmp（精确，E5）；**必须含字节 ≥128 的用例** |
| bool（经 int8 存储） | exact | `out.view(torch.bool)` 后与 `torch.le`（bool）精确比对（E1） |
| 输出 shape | `out.shape == broadcast_shapes(a,b)` | Op 层还原，测试断言 |

来源：`examples/TileOPs/tests/ops/test_le.py` 的 `_bool_compare`（断言 `output.dtype == torch.bool` + `torch.equal`）；manifest `outputs.output.dtype: "bool"`、`shape_rules`。

### 8.3 L0 门槛测试计划

> L0 为 Stage 3 开发精度门槛（golden 对比必须全过）；完整 L1/L2/Boundary 分层套件由 `tilelang-op-develop` 展开。

| # | 用例 | 输入 | 通过标准 |
|---|------|------|----------|
| L0-1 | smoke 同形全浮点 dtype | `[1024]×[1024]`：fp16/bf16/fp32，`randn` | vs golden `torch.le`，exact |
| L0-2 | 大规模同形（各 dtype） | `[2048,4096]×[2048,4096]` fp32/fp16（hidden-state 代表，chunks>1） | vs golden，exact |
| L0-3 | **广播三模式** | `[16,256,56,56]×[256,1,1]`（cnn-feat 通道广播）、`[4096,4096]×[1,4096]`（row-broadcast）、`[2,64,128]×[1,1,128]`（bias）fp32/fp16 | vs golden，exact + shape 断言 |
| L0-4 | **a 侧广播** | `[1,4096]×[4096,4096]`、`[2,1,3]×[2,3,3]`（角色互换，覆盖 `sa=0`） | vs golden，exact |
| L0-5 | 标量广播 / 全标量 | `[64,128]×[1,1]`、`[1]×[1]`（`L=1` 全标量退化，避 `TRAP-vbrc-same-shape-empty-broadcast`） | vs golden，exact |
| L0-6 | 尾块非整除 | `L` 非 `block_size` 整数倍：`[1000]×[1000]`、`[3136]×[3136]`、`[4097]×[4097]`（fp32） | vs golden，exact（覆盖 §6.4 / 风险 §9.1 R9） |
| L0-7 | bool 存储路径 | `[4096]×[4096]`、`[2,512,768]×[1,1,768]` bool → int8 存储 | vs `torch.le`(bool)，exact + `out.dtype==bool` |
| L0-8 | **整数 dtype 全集** | `[4096]×[4096]`：int8/int16/int32/int64（含 int16→f32 升域） | vs golden，exact |
| L0-9 | 已知相等边界（弱序） | 部分位置强制相等（`test_le_edge_case` 同款：`b[::2]` 置异、其余相等；相等位 `<=` 必须为 **True**） | vs golden，exact |
| L0-10 | **浮点角点语义** | fp16/fp32：NaN 对 NaN、NaN 对有限值、有限值对 NaN、+Inf 对 +Inf、+0.0 对 −0.0、次正规数 | vs `torch.le`（**任一 NaN → False**；±0 相等 → **True**），exact（风险 §9.1 R6；若 vcmp("le") 为无序/补集语义则 fail → 触发设计回退） |
| L0-11 | persistent 边界 | `N/L` 使 `num_logical ∈ {39, 40, 41}`（核数切换边界，block_size 使能） | vs golden，exact + 无越界 |
| L0-12 | **vcmp 同形约束回归** | `[M,N]×[1,N]`、`[M,N]×[M,1]`（源 GPU 可直接，NPU 经显式物化） | vs golden，exact（验证 §0.6 R1 物化正确，不依赖 vcmp 广播） |
| L0-13 | **uint8 无符号序回归（le 特有）** | `[4096]×[4096]` uint8，取值覆盖 [0,255]（**必含 ≥128**，如 `randint(0,256)`；并含 0/128/255 角点） | vs golden，exact（验证 §0.6 R5 修正；朴素 int8 承载此用例必 fail） |
| L0-14 | uint8 广播 + 无符号修正组合 | `[2,64,128]×[1,1,128]` uint8（值域 [0,255]） | vs golden，exact（覆盖 R5 修正与 R1 vbrc 物化的组合） |

**L0 测试要点**：输入经 `.npu()` 放置；广播用例额外断言 `out.shape == broadcast_shapes(a.shape, b.shape)`；bool 用例断言输出 dtype 与值；L0-10 专门覆盖浮点 IEEE 比较语义（设计期探针用 `randn`，未覆盖 NaN/±0，是设计层未闭合项）；L0-12 覆盖 vcmp 无广播；**L0-13/L0-14 是本算子相对 eq 的关键新增**——uint8 无符号序（`test_le_integer_dtype` 的 uint8 数据仅 [0,16]，无法暴露符号错误，必须显式补 ≥128 用例）；**L0-9 的相等位对 le 必须为 True**（弱序），是相对 gt 的关键对照点。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束/风险 | 影响 | 缓解方案 |
|---|-----------|------|----------|
| R1 | **bool GM 输出不可用**：`T.copy(bool_ub → bool_GM)` 编译成功但运行期 Vector 核超时 507034（CG-2026-0022） | 输出契约 bool 无法直接落盘 | §0.6 R3：int8 0/1 承载 + Op 层 `view(torch.bool)`；wrapper 输出 dtype `bool`→`int8`（Stage 3/5 集成项，§9.1 R11） |
| R2 | **`T.vcmp` 不接受自动广播**（CG-2026-0021） | 广播路径不能用单条 vcmp，必须显式物化 | §0.6 R1：`T.vbrc`/连续段先把两 operand 变同形；L0-12 回归 |
| R3 | **`T.vcmp` 不支持 int8/uint8**（文档 §2.2.1） | 1 字节 dtype 无法直接比较 | §0.6 R4：精确升 fp16（E3 机器验证） |
| R4 | **`T.vbrc` int8 dst 运行期挂死**（同族 ge/gt 设计期探针记录） | int8 标量广播路径挂死 | 先升 f16 再 `vbrc`（f16 vbrc 正常），int8 全部值在 f16 精确；§0.6 R1 步骤 3 |
| R5 | **uint8 GM 张量在 npuir 不可用**（CG-2026-0018） | uint8 输入路径不可直接声明 | §0.6 R3：Op 层 `view(int8)`；**并叠加 §0.6 R5 无符号域修正**（`<=` 排序符号相关，E-NAIVE 证明朴素承载全错） |
| R6 | **浮点比较语义未设计期闭合**：`T.vcmp("le")` 对 NaN/±0 是否严格遵循 IEEE（NaN → False、±0 相等 → True）未在探针覆盖（探针/机器验证均基于有限值） | 若 vcmp 为无序/位比较/补集语义，浮点角点结果与 `torch.le` 不符 | L0-10 强制覆盖角点；若不符则触发设计回退：改用析取式 `(a<b)∨(a==b)`（IEEE 等价，E-NAN 0 违反，但 op 数 ×3，性能回退），列入 Stage 2/3 复核项 |
| R7 | **int32/int64 的 vcmp 标量化风险未文档化**：PL-1.11 仅实证 int16 标量化；int32/int64 无实测记录 | 若 int32/int64 也标量化，性能可能崩溃 | 标注「未文档化假设」；L0-8 覆盖正确性，性能交 Stage 4 msprof 判（若标量化可评估 int32→fp64 不可用 / int64 拆分路径，作为已知局限） |
| R8 | **uint8 修正链的别名 dtype 性能税**：`vselect(m, sp, a_c, a_c)` 原地写回（PL-1.15 记录 aliasing/列广播可编译可算对但有 +12~20% 性能税） | uint8 同形/广播路径可能多付向量管线税 | §3.3 注记 ⑦：Stage 3/4 可改为独立 dst buffer（`u`）消除 alias；uint8 非 manifest 主用 dtype，优先正确性 |
| R9 | **尾块非 32B 对齐拷贝** | T.copy 报错或数据错乱 | 采用 `vec_add_1d.py` 零起点 dst 切片先例；L0-6 强制覆盖 |
| R10 | **int64 的 2-D 备选容量约束**（备选 #4 R_blk=2 时 UB ×1.7 膨胀超 192KB） | int64 广播无法用 2-D tile | 备选 #4 仅 fp32/fp16/bf16（R_blk=2）；int64 按 R_blk=1 退化为方案 #1（容量否决，§1.6.3 已记） |
| R11 | **wrapper 承载 dtype 适配**：`LeFwdKernel` 现 `OUTPUT_DTYPE=torch.bool`/`output_dtype_str="bool"`；`LeBoolStorageFwdKernel` 现 `SUPPORTED_DTYPES=(torch.uint8,)`、Op 层 `view(torch.uint8)` | 不改为 int8 承载则 bool 用例运行期挂死 / uint8 用例错值 | wrapper 输出 dtype 改 `"int8"`、bool/uint8 输入 `view(int8)`、输出边界 `view(bool)`；uint8 需向 kernel 传「无符号承载」标记以激活 §0.6 R5；Stage 3/5 核对（§0.6 R3/R5，E1/E5） |
| R12 | **block_size 分解一致性**：工厂 `callable(block_size)` 形态下，`chunks/num_logical/num_kernels/num_local` 若在工厂构建期用默认 block_size 计算、而 kernel 体用被调用值，则非默认 block_size 静默半错（无报错） | 非默认 block_size 下约 50% 元素未写，形似精度 bug | 全部分解放在 `@tilelang.jit` 函数体内、用该函数 block_size 计算（§3.3 注记 ⑧；bitwise_and TRAP-jit-blocksize-decomposition） |
| R13 | **常数为 stale / 跨核数外推**：`CONST-mte2-degradation` 被 kb_stale_check 标 stale；本机 20 核 ≠ 常数 48 核 | 设计期流量项估算可能失准 | 估算行已标注；Stage 4 baseline 后按 hardware-cost-model §3 回填「设计估算 vs 实测」偏差行 |
| R14 | **极小 L 效率退化**（一般形状）：若某广播形状 innermost 维很小且外层 R 很大（如 `[65536,1]×[65536,8]`：L=8, R=65536） | 每 task 向量过短、task 数过大 | manifest/测试均不触发；作为已知局限记录（不阻塞）；如出现可在 Op 层合并 broadcast 维度后重排 |
| R15 | **`T.vcmp` 无标量操作数形式**：uint8 修正掩码的常量 0 若直接写成标量，会触发 `'hivm.hir.vcmp' operands' shape are inconsistent`（同族 ge Stage 2 检视发现的符号闭包缺口） | 编译失败（trace 期） | §3.3/§4.3 显式物化 `zero = T.alloc_ub((block_size,), cmp_dtype)`；Stage 3 符号闭包核对 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 直接用 bool 声明输出张量 | 输出契约 bool 直译 | 编译成功但运行期 Vector 核超时（507034，CG-2026-0022） | 输出用 int8（0/1），Op 层 `view(bool)`（§0.6 R3） |
| **uint8 直接以 int8 承载比较、不做无符号修正** | uint8 输入（字节 ≥128） | 排序结果反转/错误（E-NAIVE 约 50% 不一致）；测试数据仅 [0,16] 时「通过但错误」 | §0.6 R5：`u = s + 256·(s<0)` 修正后再 vcmp；L0-13 强制 ≥128 用例 |
| **误用补集式 `¬(a>b)` 实现 le** | 浮点含 NaN | NaN <= x 被算成 True（与 IEEE 不符） | 用 `T.vcmp("le")` 直接语义或用析取式 `(a<b)∨(a==b)`（E-NAN 0 违反） |
| **误把 le 当严格序（相等返回 False）** | 相等操作数（含 ±0） | 相等位结果反了 | le 为弱序，相等恒 True（E-ZERO）；回退式用析取而非合取（与 gt 相反） |
| 把 int16 也留原生 vcmp | int16 输入 | 全操作数形态被标量化（PL-1.11，[32,256] 单链 236µs） | 升 fp32（精确，E4） |
| 把 int16 升 fp16 比较 | int16 输入 | int16 > 2048 时 fp16 不精确 → 假相等/序错 | int16→**fp32**（E4）；int8/uint8/bool 才升 fp16 |
| `T.vcmp(a_c, b_c, cond)` 两操作数形状不同 | 广播路径（`[M,N]×[1,N]` 等） | BiShengHIR `'hivm.hir.vcmp' op operands' shape are inconsistent`（CG-2026-0021） | 先 `T.vbrc`/连续段把两 operand 物化为同形（§0.6 R1） |
| `T.vcmp(x, 0.0, ...)` 常量标量操作数 | uint8 修正掩码 `s<0` | vcmp 无标量操作数形式 → 编译失败 | 显式物化 `zero` (block_size,) 同形 buffer（§9.1 R15） |
| 用 `T.vcmp` 直接比较 int8/uint8 | 1 字节 dtype | 编译失败（vcmp 不支持） | 先 `vcast` 升 fp16（精确，§0.6 R4） |
| 用 uint8 声明张量/UB | uint8 输入路径 | 输出被静默映射 float32、数值错（CG-2026-0018） | Op 层 `view(int8)`（§0.6 R3） |
| 对 int8 直接 `T.vbrc` | int8 标量广播 | 运行期 Vector 核挂死 | 先升 f16 再 vbrc（§0.6 R1 步骤 3） |
| 误以为 vcmp 像 vand/vor 一样能自动广播 | 照抄位运算设计 | 编译失败 | 本算子必须显式物化（§0.6 R1） |
| `T.vbrc(x, y)` src/dst 同形 | `L==1` 全标量 | `empty broadcast dims array` verify fail（TRAP-vbrc-same-shape-empty-broadcast） | 保证 dst 为 `block_size` (>1)；`L==1` 时 src (1,)→dst (block_size,) 合法 |
| `num_kernels` 误用逻辑核数 | 直接照抄逻辑核数作 grid | 数千核串行调度、启动开销剧增 | §5.5 方案为准（`min(num_logical, 40)`） |
| block_size 相关分解写在工厂构建期 | wrapper `callable(block_size)` 契约 | 非默认 block_size 静默半错 | 分解放 jit 函数体内（R12） |
| 行 base divmod 写成逐元素 | 把 `_compute_broadcast_offsets` 直接搬进 kernel 元素循环 | 标量热点，性能崩溃 | 仅在 task 级（每 task 一次）做，见 §0.6 R1 |
| block_size 超 UB 上限 | config 传入过大 | UB 溢出编译失败 | §4.5 上界（fp16≤14456 / fp32≤9637 / int64≤5782 / int16≤7228 / int8≤11565 / uint8≤7529，1.7x 膨胀口径） |

### 9.3 特殊场景处理

- **同形（无广播）**：`k==1, R==1, sa==sb==1` → 展平连续段路径（与源同形快路径一致）
- **标量广播**（`[64,128]×[1,1]`）：coalesced `[(8192,1,0)]`，`k=1, sa=1, sb=0` → b 单元素 + `vbrc`
- **a 侧广播**（`[1,4096]×[4096,4096]`）：`sa=0` 分支（symmetry）
- **全标量**（0-dim×0-dim）：coalesced 兜底 `(1,0,0)`，`L=1`，两路 vbrc（dst=block_size>1）
- **非整除 N/L**：`tail` 动态裁剪（§6.4）；manifest 全整除不触发（L 对 4096 整除或单块）
- **int8/uint8/bool 输入**：Op 层 `view(int8)` → 内核升 fp16 比较（§0.6 R3/R4）；**uint8 额外走无符号修正链**（§0.6 R5）
- **int16 输入**：内核升 fp32 比较（§0.6 R4）
- **N=0 / 空维度**：Op 层 `prod()==1` 兜底或上层拦截，kernel 不触达（与源一致）
- **混合 dtype 输入**：Op 层 `forward` 已拒绝（dtype 一致性校验）
- **fp8/complex dtype**：manifest 不含；Op 层 `SUPPORTED_DTYPES` 拒绝（`test_le_rejects_unsupported_dtype` 覆盖 complex64/fp8_e4m3fn/fp8_e5m2）

---

## 10. 交付清单

### 10.1 目录结构

```
examples/le/le_fwd_kernel/
├── DESIGN.md                         # 本设计文档
├── verify_equiv.py                   # 等价性机器验证（E1 承载 + E2 索引映射 + E3 1字节升域 + E4 int16→fp32 + E5 uint8 无符号修正 + 否决证据）
└── RETROSPECTIVE.md                  # Stage 1 复盘（skill Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | E1–E5 全 `EQUIV_PASS`（OVERALL EQUIV_PASS）；另含 E-NAIVE / E-NAN / E-ZERO 否决与弱序边界证据（不进 OVERALL） |
| `le_fwd_kernel.py`（kernel 实现） | ⬜ 待实现（Stage 3） | 工厂 `le_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides, a_numel, b_numel, output_dtype=None, block_size=4096)` → callable(block_size) → `@T.prim_func main(a, b, y)`；比较域映射按 §0.6 R4，uint8 修正按 §0.6 R5，承载按 §0.6 R3，广播路径按 §3.3；集成目标 wrapper `examples/TileOPs/tileops/kernels/elementwise/le/le.py` |
| golden | 随实现文件交付（§8.1） | `torch.le`（广播直算） |
| L0 测试 | 随实现文件交付（§8.3） | 14 个代表性 case |

### 10.3 命名规范

- 项目目录：`le`；算子目录：`le_fwd_kernel`（源算子函数名，harness 提取件同名）
- 实现文件名：`le_fwd_kernel.py`；测试入口内嵌（`if __name__ == "__main__"`，vec_add_1d.py 先例）
- `@tilelang.jit` 装饰函数名保留 `le_fwd_kernel`（迁移执行规则）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ golden（§8.1）
3. ⬜ kernel 实现（§3.3：同形 `k=1` 路径 → 广播 `k>1` 四子情形（含显式同形物化）→ int8/int16 升域路径 → uint8 无符号修正路径）
4. ⬜ L0 门槛测试全绿（§8.3，重点 L0-10 浮点角点 / L0-13 uint8 无符号序）
5. ⬜ harness 集成（wrapper 输出 dtype bool→int8、bool/uint8 输入 view(int8)、uint8 无符号标记、输出 view(bool)，§9.1 R11）

---

## 11. 设计核心与 GPU→NPU 关键重设计（下游报告提炼用）

> 本节把全文设计核心收敛为 4 条可逐算子引用的要点。

1. **多核分块（persistent 分核）**：把展平后的 coalesced 输出切成 `R 行 × ceildiv(L, block_size)` 个逻辑 task，用一维 `T.Kernel(min(num_logical, 物理核数), is_npu=True)` + 核内 `T.serial`（静态边界，grid-stride 映射）常驻 ≤40 核，替换源 GPU 的 CUDA grid 超发 × `T.Parallel(threads, npt)` 两级线程并行；核数由 `NPUUtils.get().get_aicore_num()*2` 实查（本机 20⇒40）。
2. **向量化比较与会话（v-prefix 取代逐元素标量）**：逐元素 `a <= b` 由 `T.vcmp("le")` 一条向量指令（128bit lane：fp16×8/fp32×4）覆盖整块，bool 结果经 `T.vcast(bool→f16)` + `T.vcast(f16→i8, rint)` 落为 int8 0/1；行内标量广播由 `T.vbrc` 物化；GM↔UB 走连续段 `T.copy`，尾块 `tail=T.min(...)` 截断。源 `T.Parallel` 逐线程标量比较与 `register_copy` fragment 分别由 v-prefix 与 `T.alloc_ub` 承接。
3. **输出 bool 的 1 字节承载 + 比较域映射**：源 GPU 直接写 bool 输出；NPU 上 **bool(i1) GM 输出编译成功但运行期 Vector 核超时**（CG-2026-0022）→ 输出改由 **int8 0/1 承载**（Op 层 `view(torch.bool)`），输入 bool/uint8 亦由 int8 承载（CG-2026-0018）。比较域：`T.vcmp` **既无自动广播（CG-2026-0021）又不支持 int8/uint8**，且 **int16 会被标量化（PL-1.11）** → **必须用 `T.vbrc`/连续段把两个 operand 显式物化为同形比较域缓冲**，1 字节经**精确升 fp16**（E3）、int16 经**精确升 fp32**（E4）后同形 `T.vcmp("le")` 执行；常量比较操作数（uint8 修正的 0）亦须显式物化。
4. **uint8 无符号域修正 + 弱序 NaN 回退（le 相对 eq 的关键新增，与 ge 同侧）**：`<=` 的排序**对符号敏感**（`==` 不然），uint8 以 int8 承载时朴素重解释在字节 ≥128 时结果全错（E-NAIVE 约 50% 不一致）→ 比较前必须显式修正 `u = s + 256·(s<0)`（`vcast` + `vcmp(lt)` + `vadd(+256)` + `vselect`，E5 机器验证 256×256 全枚举 0 违反）；同时 `<=` 的补集恒等式 `¬(a>b)` 对浮点 NaN 不成立（E-NAN），NaN 语义以 `T.vcmp("le")` 直取、**弱序**回退式为析取 `(a<b)∨(a==b)`（与 gt 严格序的合取相反），相等边界恒 True（E-ZERO）。
