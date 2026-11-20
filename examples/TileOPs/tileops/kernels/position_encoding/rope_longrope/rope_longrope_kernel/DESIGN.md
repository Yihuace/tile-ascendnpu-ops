# _make_rope_neox_1d 算子设计文档（迁移）

> **项目**：`rope_neox` ｜ **算子目录**：`examples/rope_neox/_make_rope_neox_1d/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_rope_neox_1d`（GPT-NeoX 风格 RoPE，layout=`1d`，逐元素 `y = x*cos + rotate_half(x)*sin`）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策**：① **计算核重设计 = 半段旋转（half-split）**：把源码「逐元素 flat 索引 + `col % half` + `if_then_else` 配对 gather」改为「按行块把 `head_dim` 拆为两半 a=`x[:, :half]`、b=`x[:, half:]`，在 UB 上做 `y1 = a*c − b*s`、`y2 = b*c + a*s`」——**数学恒等、零 gather、cos/sin 单次装载复用于两半（cos/sin 流量减半）**。② **计算域 = fp32 中转**（对齐契约 golden 的 fp32 opmath + 单次舍回；bf16 亦受影响）。③ **并行/内存重设计**：GPU `block×thread + 256 threads + fragment + T.Parallel` → 一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 + 2D UB tile (block_rows, half) 整 buffer 向量链。
>
> **源优化意图承接**：源 #1 `T.Parallel(threads,npt)` 线程级逐元素 → §0.6 R1 一维核 + 整 buffer 向量链；源 #2 flat `//`,`%` 索引 → §0.6 R1 2D 静态 tile 寻址；源 #3 `if_then_else` 配对 gather（免物化 rotated 拷贝）→ §0.6 R2 半段结构（免 gather，且 cos/sin 复用，流量减半）；源 #4 `out_idx=[3]` → 保留；源 #5 `lru_cache` 编译期特化 → 保留；源 #6 每元素读 cos/sin 两次的隐式复制 → §0.6 R3 显式单次装载 + 复用（**优化意图反向承接：源靠 L2 吸收的 2× 读在 UB 显式消除为 1×**）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/rope.py`

- 具体算子：工厂 `_make_rope_neox_1d(seq_len, head_dim, dtype, threads=256, num_per_thread=8)`（L47–L89，`@functools.lru_cache(maxsize=32)` + `@tilelang.jit(out_idx=[3])`），内部 `@T.prim_func main(x, cos_table, sin_table, y)`：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_rope_neox_1d(seq_len, head_dim, dtype, threads=256, num_per_thread=8):
      half = head_dim // 2
      N_total = seq_len * head_dim
      block_size = threads * num_per_thread

      @tilelang.jit(out_idx=[3])
      def kernel(threads_arg, npt_arg):
          @T.prim_func
          def main(x: T.Tensor((seq_len, head_dim), dtype),
                   cos_table: T.Tensor((seq_len, half), dtype),
                   sin_table: T.Tensor((seq_len, half), dtype),
                   y: T.Tensor((seq_len, head_dim), dtype)):
              with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                  for i, j in T.Parallel(threads_arg, npt_arg):
                      flat_idx = (bx * threads_arg + i) * npt_arg + j
                      row = flat_idx // head_dim
                      col = flat_idx % head_dim
                      freq_idx = col % half                     # Neox: col<half → cos[col], col≥half → cos[col-half]
                      c = cos_table[row, freq_idx]
                      s = sin_table[row, freq_idx]
                      val = x[row, col]
                      rotated = T.if_then_else(col < half,
                                               -x[row, col + half],
                                               x[row, col - half])
                      y[row, col] = val * c + rotated * s
          return main
      return kernel
  ```

- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py`（Pattern A，`_make_rope_neox_1d` / `_make_rope_neox_2d`，curated 到两个 neox 工厂）。
- 集成 wrapper（**Stage 5 集成目标，本迁移必须对齐其接口**）：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py`（`RopeNeoxKernel`）。K11 折叠：`_DEFAULT_BLOCK_SIZE = {"float32": 1024, "float16": 2048, "bfloat16": 2048}`；工厂签名 `_make_rope_neox_1d(seq_len, head_dim, dtype)`，返回可调用 `kernel(block_size)`，`prim_func` 签名 `main(x, cos_table, sin_table, y)`（`out_idx=[3]`），调用 `self._compiled_fn(x, cos, sin)`。

### 0.1 源算子语义（做什么）

**数学语义**（NeoX / GPT-NeoX 风格 RoPE，`layout="1d"`；参考 GPT-NeoX / HuggingFace `RotaryEmbedding`）：

$$
y[row, col] = x[row, col]\cdot c[row, col\bmod half] + \operatorname{rot}(x)[row, col]\cdot s[row, col\bmod half],\qquad half = head\_dim/2
$$

$$
\operatorname{rot}(x)[row,col]=
\begin{cases}
-x[row,\ col+half], & col < half\\[2pt]
+x[row,\ col-half], & col \ge half
\end{cases}
$$
（`c`=`cos_table`，`s`=`sin_table`，shape 均为 `(seq_len, half)`。）

**等价闭式（半段 split，本设计计算核的语义依据）**：把每行的 head 维在 `half` 处拆成两半 `a = x[:, :half]`、`b = x[:, half:]`，则对 `j ∈ [0, half)`：

$$
y[row,\ j] = a_j\,c_j - b_j\,s_j,\qquad y[row,\ half+j] = b_j\,c_j + a_j\,s_j .
$$

推导：`col=j<half` → `freq_idx=j`、`val=a_j`、`rot=−b_j`；`col=half+j` → `freq_idx=(half+j)%half=j`、`val=b_j`、`rot=+a_j`。与 §0 公式逐项一致，**数学恒等**。

**契约语义 = 源仓测试基准**（`tests/ops/test_rope.py::ref_rope_neox`）：

```python
cos_full = torch.cat([cos, cos], -1); sin_full = torch.cat([sin, sin], -1)
rot = torch.cat([-x[..., half:], x[..., :half]], -1)
y = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)
```
即 **fp32 opmath（`x.float()`/`cos.float()`）+ 末端一次舍回原 dtype**。容差（`_get_tolerances`）：fp32 `1e-5` / fp16 `1e-3` / bf16 `1.6e-2`（atol=rtol）。

**计算域语义（迁移必须保持的关键点，对照同族先例 PL-1.27）**：

| 域 | 表示 | 与契约 golden 的一致性 |
|----|------|------------------------|
| **契约 golden**（`ref_rope_neox`） | `x.float()*cos_full.float() + rot.float()*sin_full.float()` → `.to(dtype)` | 基线：fp32 域两次乘 + 一次加，末端单次舍回 |
| 源 GPU kernel | `val*c + rotated*s`，`c,s,val,rotated` 均**输入 dtype**（无显式 fp32 提升） | **与契约存在域差**：源码在输入 dtype 域算；golden 在 fp32 域算后单次舍回——**异域（需等价替换为 fp32 中转）** |

→ 与 hardtanh（源本身即 fp32）**不同**：本算子源在**输入 dtype 域**计算，而契约 golden 要求 **fp32 opmath**。§1.6.1 #2 采纳 **fp32 中转计算域**（`vcast` 升 fp32 → 向量链 → `vcast` 舍回），由 `verify_equiv.py` 诊断证明输入 dtype 域对 fp16/bf16 超出容差（fp16 `max_abs=7.81e-3 > 1e-3`）。

**dtype 语义**：`x`/`cos_table`/`sin_table`/`y` 同 dtype（fp16/bf16/fp32）。cos/sin 由 **Op 层（host）** 以 fp32 计算后 `.to(dtype)`（`tileops/ops/rope.py::_base_freqs`），故进入 kernel 时**已是目标 dtype**；本设计 `vcast` 升 fp32 后计算。中间计算精度 = fp32，输出单次舍回目标 dtype。

**规约语义**：**无规约**——逐元素旋转，`(row, col)` 与 `(row, col±half)` 构成两两配对的 2×2 旋转，元素间无累加、无归约轴，**不存在累加顺序问题**。

**边界语义**：无分支特例；`±inf`/`NaN` 经 IEEE mul/add 传播（`inf*0=NaN` 与 torch 一致，见 §8/§9）；`x=0` → `y = rot*s`；非整除 `seq_len` 由行块尾处理（§5.4）。manifest `shape_rules: x.shape[-1] % 2 == 0`。

**host 侧语义**：1d layout 下 `forward` 直接调用编译后 kernel（无 flatten / 无 pre/post 处理）；cos/sin 在 Op 层按 `(seq_len, half)` 生成并缓存（`_freq_cache`）。layout=2d 的 flatten 属另一工厂 `_make_rope_neox_2d`，**本迁移不涉及**。

**语义保持基线**：§8.1 golden 以本节语义（`ref_rope_neox`，fp32 opmath + 单次舍回）为唯一依据，**不得复刻 §0.6 的 NPU 半段向量链结构**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(seq_len, head_dim)` | `float16 \| bfloat16 \| float32` | 输入（manifest 1d workloads：`[256,64]` / `[2048,64]` / `[4096,128]`） |
| `cos_table` | `(seq_len, head_dim // 2)` | 同 `x` | 余弦频率表（Op 层生成） |
| `sin_table` | `(seq_len, head_dim // 2)` | 同 `x` | 正弦频率表 |
| `y`（输出） | `(seq_len, head_dim)` | `same_as(x)` | 旋转结果 |

**输出 shape（迁移必需）**：`same_as_input` = `(seq_len, head_dim)`（源 `out_idx=[3]` 自动分配；`source_output_shape = (seq_len, head_dim)`）。
**layout**：行主连续（row-major contiguous），**无转置**。
**动态轴**：无运行时动态轴——`seq_len` / `head_dim` 经 `lru_cache` 编译期特化（wrapper 亦按 `(seq_len, head_dim, dtype)` 固定）；尾块守卫用编译期常量 `seq_len`。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源，`T.Parallel(threads_arg, npt_arg)` 内逐元素）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `flat_idx = (bx*threads + i)*npt + j` | block 号 `bx`、线程 `i`、元素 `j` | 全局 flat 下标 | 元素定位 |
| 2 | `row = flat_idx // head_dim`；`col = flat_idx % head_dim` | `flat_idx` | `row`,`col` | 二维坐标还原 |
| 3 | `freq_idx = col % half`；`c=cos_table[row,freq_idx]`；`s=sin_table[row,freq_idx]` | `cos_table`,`sin_table` | 标量 `c`,`s` | 频率表取值（含两半共用同一 freq 轴） |
| 4 | `val = x[row,col]`；`rotated = col<half ? -x[row,col+half] : x[row,col-half]` | `x` | 标量 `val`,`rotated` | `rotate_half` 配对 |
| 5 | `y[row,col] = val*c + rotated*s` | `val`,`c`,`rotated`,`s` | `y` | 2×2 旋转 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x(seq_len,head_dim)] --(每线程 npt 个连续元素, coalesced load)--> registers
GM[cos_table/sin_table] --(逐元素按 freq_idx 取)--> registers
   --(T.Parallel(threads,npt) 逐线程标量: col%half; if_then_else 配对; val*c + rotated*s)--> registers
   --(coalesced store)--> GM[y]
（无显式 SMEM；无 fragment 缓冲——纯寄存器逐元素；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N_total, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 个**连续**元素（`T.Parallel(threads_arg, npt_arg)`）。`block_size = threads×npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048（wrapper K11 折叠至 `_DEFAULT_BLOCK_SIZE`）。

**host 侧逻辑**：`N_total = seq_len*head_dim`、`half`、`block_size` 由工厂 Python 侧计算（编译期）；`out_idx=[3]` 由 jit 自动分配输出；flatten（仅 2d）与 cos/sin 生成在 Op 层（§0.1）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `T.Parallel(threads, npt)` block×thread 两级并行 + 每线程 `npt` 连续元素 | 吃满访存带宽（合并访存） | 256 线程/block × grid(`ceildiv(N_total, bs)`) | GPU thread/warp 执行模型 + memory coalescing | **硬件强相关**（GPU 线程模型） |
| 2 | flat 一维寻址 `flat_idx // head_dim`、`% head_dim`（编译期常量除/模） | 避免二维寻址开销；grid 按 flat 均匀切分 | 常量除/模由编译器优化为移位/乘 | GPU 标量 ALU | 实现层（NPU 用 2D 静态 tile 替代） |
| 3 | 逐元素 `if_then_else` 配对（`rotate_half` 免物化） | 免去构造 rotated 张量（省一次全量读写） | 逐元素条件选择配对的另一元素 | GPU 标量 select（一次性、无额外访存） | **算法/实现混合**（NPU 无廉价逐元素 gather → 需重设计） |
| 4 | `@tilelang.jit(out_idx=[3])` | 自动分配输出张量 | jit 参数约定 | 无硬件依赖 | 可移植（NPU 保留） |
| 5 | `functools.lru_cache(maxsize=32)` | 按 `(seq_len, head_dim, dtype)` 编译期特化 | 常量固化（seq_len/head_dim/half/block_size） | 无硬件依赖 | 可移植（NPU 保留） |
| 6 | 隐式 cos/sin 复制（`freq_idx = col % half`：`col` 与 `col+half` 读同一 freq） | 免去显式 `cat([cos,cos])` 物化 | 索引映射使两半共用同一 freq 行；PTX 层每元素各读一次（L2 吸收重复） | GPU L2 缓存 | **实现层**（NPU 上 L2 不可依赖 → UB 显式单次装载复用） |
| 7 | `threads=256`，`npt=4`(fp32)/`8`(fp16·bf16) | 每线程访问 128-bit 对齐 | 启发式（`default_config`） | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`，§5） |

> 识别不出来 ≠ 不存在。上述 7 项为源码中可识别的全部优化/规避手段；后续逐项处置（§0.5）。**未识别优化会被静默丢弃**，故此处列全。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（`y=x*c+rot*s`；I/O 契约；fp32 opmath 契约；无规约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| 算法结构（`head_dim` 拆半 + 两两 2×2 旋转） | 算法 | 无（纯数学恒等式） | — | **保留（结构）** | §0.1 闭式；仅实现结构重设计 |
| 优化 #1 `T.Parallel(threads,npt)` 线程级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU `T.Kernel` 一维 block；`threads=` 无效，`TRAP-threads-kwarg-noop`） | **重新设计** | §0.6 R1：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #2 flat `//`,`%` 一维寻址 | 优化 | GPU 标量 ALU + 均匀 flat grid | 有（2D tile 静态寻址） | **等价替换** | §0.6 R1：2D tile `(block_rows, half)`，行块号 = 任务号，`half` 为内层向量轴 |
| 优化 #3 逐元素 `if_then_else` 配对 gather | 算法/优化 | GPU 廉价标量 select + 隐式 L2 | **无廉价逐元素 gather**（跨步 gather 向量指令缺失，`TRAP-C10-strided-gather`） | **重新设计** | §0.6 R2：半段（half-split）结构——UB 上 `y1=a*c−b*s`、`y2=b*c+a*s`，`a,b,c,s` 均为连续 UB 子块，**零 gather** |
| 优化 #6 隐式 cos/sin 复制（L2 吸收 2× 读） | 优化 | GPU L2 缓存 | 有（UB 显式复用） | **等价替换（并优化）** | §0.6 R3：cos/sin 按行块**单次装载**，两半复用同一 `(block_rows, half)` tile；cos/sin 访存 2×→1× |
| 优化 #4 `out_idx=[3]` | 优化 | 无 | 有 | **保留** | NPU 保留 `out_idx=[3]`（对齐 wrapper 契约） |
| 优化 #5 `lru_cache` 编译期特化 | 优化 | 无 | 有 | **保留** | 保留 `(seq_len, head_dim, dtype)` 特化（+`block_size`） |
| 优化 #7 `threads`/`npt` 对齐 | 优化 | memory coalescing | 有（32B 尾轴对齐 / 128-bit 向量宽度） | **等价替换** | K11：折叠为 `block_size`（§5.2），32B 对齐按 NPU 复算（§5.3） |
| 计算域 = 输入 dtype 域（源无显式提升） | 语义/优化 | 无 | 有（`vcast` 支持 f16/bf16↔f32，`rint`） | **等价替换** | §0.6 R2 / §1.6.1 #2：改为 **fp32 中转**（契约 golden 为 fp32 opmath；输入 dtype 域对 fp16 超容差） |
| layout=2d 工厂 `_make_rope_neox_2d` / THD / 其它变体 | 算法 | GPU | — | **舍弃（范围外）** | 本迁移只port `RopeNeoxOp.layout="1d"`（manifest 单算子契约；`_make_rope_neox_2d` 为另一函数，不在本 op_name 内） |

> **依据**：#1/#2 由 `TRAP-threads-kwarg-noop`（`src/ir.cc` NPU 分支不消费 `threads=`）与 `docs/开发指南.md §3.3`（一维 Kernel + 物理核数）佐证；#3 由 `TRAP-C10-strided-gather`（跨步 gather 向量指令缺失）佐证；#6 由 `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2（UB tile 整块装载）+ §1.6.3 复用布局佐证；计算域由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32`/`f32→f16/bf16` 均支持 `rint`）佐证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行/内存结构重设计（thread 级逐元素 → 一维核 + 核内网格跨步 + 2D UB tile）**

- **源方案**：`T.Kernel(ceildiv(N_total, block_size), threads=256)` + `T.Parallel(256, npt)`，flat 索引 `//`,`%` 逐元素寻址；性能意图 = 用大规模线程/grid + 合并访存吃满带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(ceildiv(n_num, core_num))` 网格跨步（`task = i*core_num + cid`，`if task < n_num` 守卫）；每 tile 处理 `block_rows` 行，行块号 `task` → `r0 = task*block_rows`；2D UB tile `(block_rows, half)`；数据流 `GM→UB`（`T.copy` 整块）→ 整 buffer 向量链 → `UB→GM`。
- **语义保持论证**：任务映射只重排「哪个核处理哪些行区间」，各行旋转完全独立（无跨行依赖、无累加顺序变化），逐行数学语义恒等；`row` 的扇出仅在本 tile 内，`half` 维完整落在 tile 内（**每行两半都取自同一 tile**，见 R2），不跨 tile。边界：`T.serial` 上界 `T.ceildiv(n_num, core_num)` 为编译期静态值（`n_num`,`core_num` 均常量），守卫 `task < n_num` 覆盖尾块，行块尾以 `tr = T.min(block_rows, seq_len - r0)` 动态裁剪 `T.copy`（§5.4），不越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py`（一维核 + `T.serial` 网格跨步 + 2D tile + `T.copy(A[bx,by], A_VEC)`）；`examples/bitwise_not/bitwise_not_fwd_kernel/`（Developer 一维 persistent，`PL-1.22-persistent-1d-grid-stride`）。

**重设计项 R2：计算核重设计（逐元素 select/gather → 半段 2D 向量链）**

- **源方案**：逐元素 `col<half ? -x[row,col+half] : x[row,col-half]`（`if_then_else`）+ `val*c + rotated*s`；GPU 上 select 是一步标量、配对访问由 L2 吸收。
- **NPU 新算法（整 UB buffer 向量链，半段结构）**：对每行块，令 `a_ub = x[r0:r0+br, 0:half]`、`b_ub = x[r0:r0+br, half:head_dim]`、`c_ub/s_ub = cos/sin[r0:r0+br, 0:half]`（均 `(br,half)` 2D UB tile）；fp32 中转后：
  - `y1f = vsub(vmul(af, cf, ·), vmul(bf, sf, ·), ·)` —— 对应 `y[:, 0:half] = a*c − b*s`
  - `y2f = vadd(vmul(bf, cf, ·), vmul(af, sf, ·), ·)` —— 对应 `y[:, half:] = b*c + a*s`
  - 共 **6 个向量 op**（4 `vmul` + 1 `vsub` + 1 `vadd`）作用于 `(br,half)`；sub-fp32 另 +4 `vcast`（升）+2 `vcast`（舍回）。
- **语义保持论证**：
  1. **数学恒等**（§0.1 闭式推导）：`y1_j = a_j c_j − b_j s_j` 与源 `col=j` 的 `val·c + rotated·s = a_j c_j + (−b_j) s_j` 逐项相等；`y2_j = b_j c_j + a_j s_j` 与源 `col=half+j` 相等。**零近似**。
  2. **舍入路径与 golden 一致**：golden 为 `x.float()*cos_full.float() + rot.float()*sin_full.float()` → `.to(dtype)`，即 fp32 域按「先两乘、后加」结合、末端一次舍回。NPU 链 `vmul(a,c)`、`vmul(b,s)`、`vsub` 与 golden 的 `(a*c) + ((−b)*s)` 一致（IEEE 下 `(−b)*s ≡ −(b*s)` 精确，`t1 − t2 ≡ t1 + (−t2)` 精确）→ **逐位一致**（`verify_equiv.py` #1/#2 三 dtype `max_abs=0.0`、`ulp=0`）。
  3. **边界语义**：`±inf`/`NaN` 经 4 mul + 1 add 传播，与 golden 相同运算集合；`x=0` → `y1 = −b*s`、`y2 = a*s`，恒等；非整除 `seq_len` 由 `tr` 行尾裁剪。
- **源优化意图承接**：源 #3「免物化 rotated 拷贝」→ **承接**：半段结构无需物化 `rot`，直接在 UB 上对 a/b 用两个符号项表达；源 #2「免二维寻址」→ 2D 静态 tile（行块号 + 列内 0/half 静态偏移）承接；源计算域「输入 dtype」→ **有意替换**为 fp32 中转（契约对齐，§1.6.1 #2）。

**重设计项 R3：cos/sin 访存 CSE（隐式 2× 读 → 显式 1× 读 + 复用）**

- **源方案**：每元素读一次 `cos_table[row, col%half]`；因 `col∈[0,half)` 与 `[half,head_dim)` 映射到同一 `freq_idx`，PTX 层每行 cos/sin 各读 `head_dim` 次（= `2·half` 次），重复读由 GPU L2 吸收。
- **NPU 新算法**：cos/sin 按行块**单次装载**为 `(br, half)` tile（`c_ub/s_ub`），**两半计算复用同一 tile**（`y1` 与 `y2` 都用 `cf/sf`）。
- **语义保持论证**：复用不改变数值——`y1`/`y2` 使用的 `c_j/s_j` 与源 `col=j` 和 `col=half+j` 各自的 `cos_table[row,j]` 完全同值（源两半本就映射到同一 `freq_idx`），数学恒等；仅消除重复访存。收益：cos/sin 的 GM 流量由 `2·seq_len·half·2` 降为 `seq_len·half·2`（**cos/sin 部分 −50%**，全算子访存 −~25%，与 manifest roofline `bytes=(2·x_elems + cos_sin_elems)·elem_bytes` 的理论最小一致——源实现名义上受益于 L2，NPU 在 UB 显式兑现）。
- **承接说明（源优化意图不丢弃）**：§0.4 #6「隐式复制靠 L2」→ R3 显式 UB 复用承接；#1/#2 → R1；#3 → R2；#4/#5 保留；#7 → §5。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/rope.py`（`_make_rope_neox_1d` L47–L89；`RopeNeoxKernel` L392–L399，`ROTATION_STYLE="neox"`）。
- 源仓测试基准：`/home/developer/TileOPs/tests/ops/test_rope.py`（`ref_rope_neox` L62–L75 = 契约 golden；`RopeBasicFixture` L299–L308；`_get_tolerances` L287–L293；`test_rope_neox_1d` L325–L333）。Op 层：`/home/developer/TileOPs/tileops/ops/rope.py`（`_base_freqs` L99–L118；`RopeNeoxOp` L491–L522）。
- manifest：`examples/TileOPs/tileops/manifest/position_encoding.yaml`（`RopeNeoxOp`；1d workloads `[256,64]` fp16/bf16/fp32、`[2048,64]` fp16、`[4096,128]` bf16；`roofline.bytes=(2*outer*seq_len*head_dim + seq_len*(head_dim//2)*2)*elem_bytes`，`flops=4*outer*seq_len*head_dim`）。
- 迁移元数据：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/.migration_meta.json`。
- 同族迁移先例（只读参考）：`examples/hardtanh/_make_hardtanh_kernel/`、`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/elu/_make_elu_kernel/`、`examples/dropout/_make_dropout_kernel/`、`examples/bitwise_not/bitwise_not_fwd_kernel/`。

---

## 1. 概述

### 1.1 算子名称

`_make_rope_neox_1d`（项目 `rope_neox`，目录 `examples/rope_neox/_make_rope_neox_1d/`）。

### 1.2 功能描述

对 `(seq_len, head_dim)` 张量逐行应用 GPT-NeoX 风格旋转位置编码（RoPE），输入预计算的 cos/sin 频率表 `(seq_len, head_dim//2)`，输出同 shape/dtype。语义等价于 `ref_rope_neox`（§0.1）。支持 `float16 | bfloat16 | float32`。**单步纯 Vector 逐元素/局部配对算子**——无规约、无矩阵运算、无跨行依赖、无超越函数（cos/sin 由 Op 层预计算为输入表）。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 计算核由「逐元素 `if_then_else` 配对 gather + 标量 `val*c+rot*s`」改为「**半段（half-split）2D 向量链**：`y1=a*c−b*s`、`y2=b*c+a*s`（6 个向量 op，零 gather）」；② cos/sin 由「逐元素按 `col%half` 取（隐式 2× 读）」改为「按行块单次装载 + 两半复用（显式 1× 读）」；③ 并行/内存结构由 GPU `block×thread + fragment` 改为「一维核 + 核内网格跨步 + 2D UB tile」；④ 计算域由「输入 dtype 域」改为 **fp32 中转 + 单次舍回**（契约对齐）。数学语义与 I/O 契约与源完全一致。

### 1.3 数学公式

逐元素定义（同 §0.1）：

$$
y[row,col] = x[row,col]\cdot c[row,\ col\bmod half] + \operatorname{rot}(x)[row,col]\cdot s[row,\ col\bmod half]
$$
$$
\operatorname{rot}(x)[row,col] = \begin{cases}-x[row,\ col+half], & col<half\\ +x[row,\ col-half], & col\ge half\end{cases}
$$

NPU 计算核（§1.6.1 优化后，半段 2D 向量形态；sub-fp32 在 fp32 计算域构造、末端单次舍回）：

$$
y[row,\ j]=a_j c_j - b_j s_j,\qquad y[row,\ half+j]=b_j c_j + a_j s_j,\qquad j\in[0,half)
$$
$$
a_j=x[row,j],\ b_j=x[row,half+j],\ c_j=cos[row,j],\ s_j=sin[row,j].
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在数学上恒等，在并行/内存实现与计算域上有差异：**

- **计算核心**（§0.6 R2 / §1.6.1 #1）：半段 2D 向量链——`y1 = vsub(vmul(a,c), vmul(b,s))`、`y2 = vadd(vmul(b,c), vmul(a,s))`（4 `vmul` + 1 `vsub` + 1 `vadd` = 6 op，作用于 `(block_rows, half)` 的整 UB buffer）。源为逐元素标量 `if_then_else` + `val*c+rot*s`；NPU 改为**无 gather、无分支**的整 buffer 向量链。
- **计算域**（§0.6 R2 / §1.6.1 #2）：sub-fp32（fp16/bf16）先 `T.vcast(rint)` 升 fp32 → 向量链在 fp32 域 → `T.vcast(rint)` 舍回；fp32 直算。匹配契约 golden 的 fp32 opmath（源为输入 dtype 域，**有意替换**）。
- **访存 CSE**（§0.6 R3 / §1.6.1 #3）：cos/sin 按行块单次装载并复用于两半，cos/sin GM 流量 −50%。
- **并行/内存结构**（§0.6 R1）：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 + 2D tile `(block_rows, half)`。

除上述外，算法与 §1.6.0 调研选定的算法族（**半段旋转 + fp32 中转 + cos/sin 复用**）一致：一趟扫描、无中间 GM 往返、无跨行依赖、无规约。

### 1.5 数据流图

```
fp16/bf16 路径（fp32 中转；产物别名 = 死亡输入 buffer）：
GM[x(seq_len,head_dim)] --T.copy--> a_ub(br,half)=x[r0:r0+br, 0:half]
                       --T.copy--> b_ub(br,half)=x[r0:r0+br, half:head_dim]
GM[cos_table] --T.copy--> c_ub(br,half)        GM[sin_table] --T.copy--> s_ub(br,half)
   --T.vcast(rint)--> af/bf/cf/sf (fp32)
   --[y1f=vmul(af,cf); y1f=vsub(y1f, vmul(bf,sf))  |  y2f=vmul(bf,cf); y2f=vadd(y2f, vmul(af,sf))]--> y1f/y2f (fp32)
   --T.vcast(rint)--> a_ub←y1(dtype), b_ub←y2(dtype)   [a_ub/b_ub 死亡后复用为输出]
   --T.copy--> GM[y[r0:r0+br, 0:half]] 与 GM[y[r0:r0+br, half:head_dim]]

fp32 路径（无中转；af=a_ub, bf=b_ub, cf=c_ub, sf=s_ub）：
GM[x] --T.copy--> a_ub/b_ub ; GM[cos/sin] --T.copy--> c_ub/s_ub
   --[同 6-op 向量链]--> y1f/y2f --T.copy--> GM[y]（两半分别写）
```

向量链内部（全部 UB 级、2D `(br,half)`、原地复用）：

```
y1f = vmul(af, cf, y1f)          # a*c
t   = vmul(bf, sf, t)            # b*s
y1f = vsub(y1f, t, y1f)          # y1 = a*c - b*s
y2f = vmul(bf, cf, y2f)          # b*c
t   = vmul(af, sf, t)            # a*s
y2f = vadd(y2f, t, y2f)          # y2 = b*c + a*s
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素 + 局部配对（2×2 旋转）类**（无规约/窗口/矩阵/超越函数），按 `_shared/standards/algorithm-research.md` §3 属轻量调研范畴；但因存在**多种结构重排候选**与**计算域/访存复用选择**，R1 给出候选表、R3 给出四口径对比、R4 逐候选评估（达到完整调研深度）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向（op / 元素；访存） | 纳入 R3 对比 |
|---|------|------------|-----------|--------------------|-------------|
| 0 | **基线：源算法**（flat 逐元素 + `col%half` + `if_then_else` 配对） | `y[row,col]=val*c+rot*s`；`rot=col<half ? −x[row,col+half] : x[row,col−half]` | 迭代定义式（fp32 域为契约） | 3 op/输出元素 + 逐元素索引/select；cos/sin **2× 读**（L2 吸收） | ✅（基线） |
| 1 | **半段旋转（half-split，采纳）** | `a=x[:, :half]`、`b=x[:, half:]`；`y1=a*c−b*s`、`y2=b*c+a*s` | **数学恒等**（§0.1 闭式推导，逐项相等） | 3 op/输出元素（4 mul + 2 add per 2 半）；**零 gather**；cos/sin **1× 读**（复用） | ✅（**采纳**） |
| 2 | 全宽 + 显式 `rot` 物化（`rot_ub=cat(−b,a)`） | `y = x2*c2 + rot*s2`（`c2=cat(c,c)`、`x2=(br,head_dim)`） | 数学恒等 | **+2 UB 拷贝 + 1 negate**（rot 物化）且 cos/sin 需 `cat` 复制；op 更多 | ✅（评估，不采纳，§1.6.1 E3） |
| 3 | 3 乘复数（Gauss 复数乘） | `m1=c(a+b), m2=a(s−c), m3=b(c+s)`；`y1=m1−m3, y2=m1+m2` | 数学恒等 | 3 mul + 5 add = **8 op/2 元素 = 4 op/元素**（>3）；NPU 标量 ALU 上 mul/add 同价 | ✅（评估，不采纳，§1.6.1 E2） |
| 4 | 三角函数恒等式合并（`a*c+b*s = R·sin(θ+φ)`） | 需 `sin/cos/atan2` 合并 | 数学等价但引入超越函数 | op 数与精度均劣化；cos/sin 表已是输入（无需再合成角度） | ❌（**不适用**：角度已在表中） |
| 5 | 多项式/低秩逼近（旋转矩阵近似） | 近似 `(c,s)` | **非等价**（近似引入误差） | 无收益（c,s 为精确输入，无需逼近） | ❌（**数学不适用**） |

**R2 在线算法**：**无在线化收益（结论：无）**——逐元素映射 + 行内局部配对（`(row,j)` 与 `(row,half+j)` 两两独立）本身天然单遍，不存在可被「单遍/流式」消去的多遍基线。结构判据：本算子**无 running 统计量**（无 max/sum/均值/平方和归约，无跨元素依赖），在线变体（running 量增量更新）的定义不适用。依据：源算法即单遍直写（§0.3 步骤 1–5，无循环携带依赖）；`algorithm-candidates.md` 的 `ALG-elementwise` R2 记「逐元素映射天然单遍」。

**R3 复杂度对比**（四口径；本算子**带宽/发射受限**，访存/遍数是主导项；以 manifest 1d 最大 workload `[4096,128]` bf16 为例，`x_elems=524288`、`cos_sin_elems=524288`）：

| 算法候选 | FLOPs（v-op 数/输出元素，fp32 域） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源） | 3（2 mul + 1 add）+ 逐元素索引/select | `(2·x_elems+cos_sin_elems)·eb`（cos/sin 逻辑 2× 读，靠 L2 吸收） | 1 遍 | O(tile)（寄存器级） | 高（每元素独立），无跨核归并 |
| #1 半段（**采纳**，fp32） | 3（4 mul+2 add per 2 输出） | `(2·x_elems+cos_sin_elems)·eb`（**cos/sin 1× 读**，理论最小） | 1 遍 | O(tile)（UB，~7 buffer，§4.5） | 高（每行块独立），无跨核归并 |
| #1 半段（sub-fp32，+fp32 中转） | 3 + 6 cast/2 元素 = **6** | 同上（cast 无额外 GM 流量） | 1 遍 | O(tile)（+4 dtype/+7 fp32 buffer） | 同上 |
| #2 全宽 + rot 物化 | ≥3 + (rot 拷贝/negate 摊薄) | 同上 + rot_ub UB 往返（UB 级，非 GM） | 1 遍 | O(tile)（+1 (br,head_dim) buffer） | 同上 |
| #3 3 乘复数 | 4（3 mul + 5 add per 2 输出） | 同上 | 1 遍 | O(tile) | 同上 |

> 口径说明：FLOPs 以「每输出元素的 v-op 数」计（无超越函数；mul/add 同价，`CONST-vector-op-cost-910_9362`）；访存 = 输入读 + 输出写 + cos/sin 读，**全部候选同为理论下界 `(2·x_elems+cos_sin_elems)·elem_bytes`**（无中间 GM 往返，`x` 与 `y` 各 1 读 1 写）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **gather 的有无（指令/延迟，非带宽）**、**cos/sin 是否复用（访存）** 与 **op 数**。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 半段（采纳）** | Vector（`vmul/vadd/vsub` fp16·fp32 ✓、bf16 ✗ → fp32 中转；`vcast` f16↔f32/bf16↔f32 `rint` ✓）；**无 gather** | 小（~7 fp32 + ~4 dtype buffer；d128 fp16 = 36 KB，§4.5）✓ | 尾轴 `half` 连续；fp16 `half=64`=128B、`half=32`=64B；fp32 `half=32`=128B，均 32B 整倍 ✓ | `seq_len`/`head_dim` 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| 基线（源 gather） | Vector 但**需逐元素索引 + select**；NPU 上即变为跨步 gather（`TRAP-C10-strided-gather`：**跨步 gather 向量指令缺失**，会退化为标量/低效） | 同 | ✓ | ✓ | 同 | ❌ **淘汰**（gather 无向量指令；标量化后 ×元素数） |
| #2 全宽 + rot 物化 | Vector；无 gather 但需 UB↔UB 拷贝/negate | 略大（+1 (br,head_dim) buffer） | ✓ | ✓ | 同 | ⚠️ 等价但 op/UB 更多，取 #1 |
| #3 3 乘复数 | Vector（4 op/元素 vs 3） | 略大（多 1 temp） | ✓ | ✓ | 同 | ❌ op 更多（**4 vs 3**），mul/add 同价无收益 |
| #4 三角函数合并 / #5 多项式逼近 | — | — | — | — | — | ❌ 数学不适用（角度已在表中）/ 非等价 |

**v-prefix 元算 dtype 矩阵（决定 sub-fp32 必经 fp32 中转）**：`docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 的 dtype 矩阵均为 **fp16 ✓ / fp32 ✓ / bf16 ✗**；`T.vcast.md` §2.2.1 支持 `f16→f32`、`bf16→f32`、`f32→f16/bf16`（均 `rint`）。→ sub-fp32 一律经 fp32 中转（**文档为准**，AGENTS.md「API docs are source of truth」；另 note：`testing/npuir/bf16_support_ops/test_vv_bf16.py` 存在 bf16 `T.vmul/T.vadd` 用例，与文档矩阵相左——**文档/示例分歧视作待确认**，见 §9 R-3；**本算子不使用该路径**，故不阻塞）。

**调研结论**：**选定「半段旋转（half-split）+ fp32 中转计算域 + cos/sin 单次装载复用」**（候选 #1 + 域替换 + 访存 CSE），无更优可行替代。

关键依据：① R3 显示全部候选访存同为理论下界 `(2·x_elems+cos_sin_elems)·eb`，差异在 **gather 的有无**、**cos/sin 是否复用**与 **op 数**：候选 #1 是唯一同时满足「零 gather + cos/sin 1× 读 + 3 op/元素（与基线同）」的候选；候选 #2 op/UB 更多、#3 op 更多（4 vs 3）、#4/#5 数学不适用。② **计算域**：契约 golden 为 fp32 opmath，源为输入 dtype 域——**异域**，必须等价替换为 fp32 中转；`verify_equiv.py` 诊断证明输入 dtype 域对 fp16 超容差（`max_abs=7.81e-3 > 1e-3`）、bf16 亦然（`6.25e-2 > 1.6e-2`）。③ 选定算法的全部 op（`vmul/vadd/vsub/vcast/copy`）均有本仓 docs 佐证与 `examples/` 实跑先例（§3.4）。

**调研范围**（「无更优替代」依据）：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」——本算子的 CSE = cos/sin 复用即对应项；R2 记「逐元素映射天然单遍」；R3 记「融合消除中间 GM 往返」；R4 记「单遍 Vector 流；dtype 路径」）——库中无 RoPE 专门条目，故以恒等变形自查补足候选 #1–#5；② 本仓 `examples/` 全系（`elementwise/*`、`leaky_relu`、`elu`、`hardtanh`、`erf`、`expm1`、`bitwise_not`、`dropout`、`norm/*`、`reduction/*`）无 RoPE 前例，最接近者为逐元素旋转族（恒等变形 + 标量特化）与 `argmax` 的「全宽移动窗口 + 尾块」形态；③ pattern-library（`kb_search "rope rotation elementwise"` / 消费预注入条目）：`ALG-elementwise`（候选库）、`PL-1.22-persistent-1d-grid-stride`（一维网格跨步模板）、`PL-1.23-vop-chain-inplace-alias`（长 v-op 链 UB 原地复用）、`PL-1.17-subfp32-fp32-transit`（sub-fp32 fp32 中转）、`CONST-ascend910-9362-profile`（设备带宽画像）、`CONST-vector-op-cost-910_9362`（单 op 代价表）、`TRAP-C10-strided-gather`（gather 缺失）、`TRAP-vbrc-empty-broadcast`（宽 1 轴广播被拒——本算子 `half≥32` 不触发，仅登记）；**上述条目 `kb_stale_check` 均标记 `[stale]`（当前 toolchain `0.1.2+a23572536…` 与其版本戳不同）**，故其**性能数字只作候选排序参考**，API 存在性以 `docs/Tilelang.language/` 为准（见 §9 R-12）；④ **外部已知算法族**（模型知识 + PyTorch/HF/CUDA 生态）：RoPE 的两种实现范式——「`cat`/`reshape` 半段旋转」与「逐元素交错配对」；复数乘视角（含 3 乘 Gauss 变体）；三角函数角度合并——均经 §1.6.1 逐项论证；无超越函数逼近空间。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = (2·x_elems + cos_sin_elems)·elem_bytes`（manifest roofline 同口径，`outer=1`）。引用 **`CONST-ascend910-9362-profile`**（`constants.md`，`status: verified`，版本戳 Ascend910_9362 / CANN 9.0.0 / 2026-10-13，**`kb_stale_check` 标 stale——数字作背景，非决策依据**）：单 die 核→GM ≈ **1.06 TB/s**、HBM ≈1.64 TB/s；纯逐元素大 shape 实测饱和 **1.06–1.10 TB/s（≈核→GM）**。取 **1.06 TB/s** 作流量分母。
- 发射项 = `(ceildiv(n_num, core_num)) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C **且 stale**，本设备 910_9362 未重验，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：fp32 `k=6`、sub-fp32 `k=12`（+4 `vcast` 升 +2 舍回）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 dtype 对应 `block_size`（fp32=1024、fp16/bf16=2048，K11 折叠；`block_rows=block_size/head_dim`）、物理 Vector 核 = 40 估算：

| workload | dtype | seq×dim | block_rows | half | n_num | bytes | 流量项 @1.06TB/s | 发射项 (k, stale) | **估算下界 max** |
|----------|---|------|-----------|------|-------|-------|-----------------|-------------------|------------------|
| smoke | fp32 | 256×64 | 16 | 32 | 16 | 192 KB | ≈ 0.19 µs | 1×6×0.5 = 3.0 µs | **≈ 3.0 µs** |
| smoke | fp16 | 256×64 | 32 | 32 | 8 | 96 KB | ≈ 0.09 µs | 1×12×0.5 = 6.0 µs | **≈ 6.0 µs** |
| smoke | bf16 | 256×64 | 32 | 32 | 8 | 96 KB | ≈ 0.09 µs | 6.0 µs | **≈ 6.0 µs** |
| neox-1d-2k-d64 | fp16 | 2048×64 | 32 | 32 | 64 | 768 KB | ≈ 0.74 µs | 2×12×0.5 = 12.0 µs | **≈ 12.0 µs** |
| neox-1d-4k-d128 | bf16 | 4096×128 | 16 | 64 | 256 | 3.0 MB | ≈ 2.97 µs | 7×12×0.5 = 42.0 µs | **≈ 42.0 µs** |

**设计期估算（D-2 roofline）**：代表性 `neox-1d-4k-d128 bf16` 估算下界（发射项口径）≈ **42.0 µs**（流量 2.97 + 发射 42.0 + 容量 0）；`neox-1d-2k-d64 fp16` ≈ **12.0 µs**。**估算不确定性声明（未文档化假设 + 估算依据）**：① `CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C **且 `kb_stale_check` 标记 stale**，本设备 910_9362 未重验，故发射项可能**显著上偏**；② **吞吐口径交叉核对（更可信，引用 verified 的 `CONST-vector-op-cost-910_9362`：普通 v-op ≈ 9.15µs/16m-elem ≈ 1.83e12 elem-op/s 聚合）**：`[4096,128]` 的 lane-elem-ops = `n_num × k × (block_rows×half)` = `256×12×1024` = 3.146M → **≈ 1.72 µs**（< 流量项 2.97 µs）；`[2048,64]` = `64×12×1024` = 786K → ≈ 0.43 µs（< 0.74 µs）。即**按单 op 吞吐口径，本算子由流量项主导，发射项（0.5µs/op 口径）系统性上偏**——与 `PL-1.6-copy-floor`「≥16M 计算链被 MTE2 窗口完全隐藏」一致（该条 stale 仅作背景）；③ `block_size` 是摊薄发射的第一杠杆（发射 ∝ 1/`block_rows`，§5.2/§1.6.3 备选）；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（半段旋转 + fp32 中转 + cos/sin 复用）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **半段旋转结构重排**（采纳） | 逐元素 `y[row,col]=val*c+rot*s`；`freq_idx=col%half`；`rot=col<half? −x[row,col+half] : x[row,col−half]`（逐元素索引 + select） | `a=x[:, :half]`、`b=x[:, half:]`；`y1=a*c−b*s`、`y2=b*c+a*s` | **数学恒等**（§0.1 闭式推导：`col=j` → `a_j c_j+(−b_j)s_j`；`col=half+j` → `b_j c_j+a_j s_j`）；舍入路径与 golden 一致（`(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)` IEEE 精确）→ `verify_equiv.py` #1 三 dtype × {randn,corners} `max_abs=0.0`、`ulp=0` | **消除逐元素 gather**（NPU 上会退化为标量/低效，`TRAP-C10`）；op 数不变（3/元素）；**UB 层零中间 GM 往返** |
| 2 | **fp32 中转计算域**（采纳 = 异域等价替换） | 输入 dtype 域：`val*c+rot*s`（fp16/bf16 原生域） | `vcast↑f32(rint) → 6-op 链（fp32 域）→ vcast↓dtype(rint)` | 与契约 golden（fp32 opmath + 单次舍回）**逐位一致**（`verify_equiv.py` #2 三 dtype `max_abs=0.0`、`ulp=0`）；**诊断证明输入 dtype 域超容差**：fp16 `max_abs=7.81e-3 > atol 1e-3`、bf16 `6.25e-2 > 1.6e-2` → 非等价，**必须**中转 | 正确性：对齐 golden；bf16 另受 `vmul/vadd/vsub` 无 bf16（docs §2.2.1）API 强制；代价 +4/+2 `vcast`（sub-fp32） |
| 3 | **cos/sin 单次装载 CSE**（采纳） | 逐元素按 `freq_idx=col%half` 取 `c/s`（每行读 `head_dim` 次；两半各一次） | cos/sin 按行块单次装载为 `(br,half)` tile，`y1`/`y2` 复用同一 `cf/sf` | 数值不变（两半在源中本就映射同一 `freq_idx`，取同值）→ `verify_equiv.py` #3 三 dtype `max_abs=0.0` | **cos/sin GM 流量 2×→1×**（cos/sin 部分 −50%；全算子访存 −~25%，达 manifest 理论最小） |
| E1 | 输入 dtype 原生域（**不采纳**） | — | 直接用 fp16/bf16 `vmul/vadd/vsub`（省 6 `vcast`） | **非等价**：`verify_equiv.py` 诊断 fp16 `max_abs=7.81e-3`（viol 1e-4）、bf16 `6.25e-2`（超容差）；bf16 另 API 缺失 | op 少 6 但**精度失败** → 拒绝；理由记录于此，非 Stage 4 备选 |
| E2 | 3 乘复数（Gauss）（**不采纳**） | — | `m1=c(a+b), m2=a(s−c), m3=b(c+s)`；`y1=m1−m3, y2=m1+m2` | 数学恒等 | op 更多（3 mul + 5 add = 8/2 元素 = 4/元素 vs 3），NPU mul/add 同价无收益 |
| E3 | 全宽 + `rot` 物化（**不采纳**） | — | `rot_ub=cat(−b,a)`；`y=x2*c2+rot*s2` | 数学恒等 | +UB↔UB 拷贝/negate + `c2=cat(c,c)` 复制，op/UB 更多 |

**优化结论**：采纳 **3 项**（#1 半段结构、#2 fp32 中转域、#3 cos/sin 复用）。**公式层面已无进一步计算量可优化空间**——每输出元素已是最少 **3 个算术 op**（2 mul + 1 add/sub，半段结构对 2×2 旋转达到最小乘法数；候选 #3 的 3 乘方案用更多加法换 1 次乘法，在 mul/add 同价的 NPU 上更差）；访存固定在理论下界 `(2·x_elems+cos_sin_elems)·eb`（1 读 1 写 + cos/sin 1 读、无中间 GM 往返）；唯一额外 op 是 sub-fp32 的 6 次 `vcast`（bf16 为 API 强制；fp16 为 golden 对齐强制，E1 记录其不可省）。**优化后公式**（供 §3.1 使用）：

- fp32：`y1 = vsub(vmul(a,c), vmul(b,s))`；`y2 = vadd(vmul(b,c), vmul(a,s))`（6 op）
- fp16/bf16：`af=cast32(a), bf=cast32(b), cf=cast32(c), sf=cast32(s)` → 同 6 op（fp32 域）→ `y1=cast16(y1f)`、`y2=cast16(y2f)`（+6 op）

**等价性机器验证（D-1）**：脚本 `examples/rope_neox/_make_rope_neox_1d/verify_equiv.py`（CPU；候选式 = NPU 半段向量链逐 op 复现，fp32 域计算后按目标 dtype 单次舍回；基线式 = 源 `ref_rope_neox` 公式（fp32 opmath + 单次舍回）；另 fp64 参照与输入 dtype 域诊断）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**TOTAL 30 items，无 EQUIV 未通过项（0 failures），OVERALL: ALL ITEMS EQUIV_PASS**）：

| 采纳项 / 对照 | 候选式 | 基线式 | samples / 元素数 | 最大 abs 差 | 最大 ulp | 违反比 | 结论 |
|---------------|--------|--------|---------|------------|--------|------|------|
| #1 half-split vs BASELINE [fp32/randn] 256×64 | 半段 fp32 链 | 源公式 fp32 | 16,384 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** |
| #1 half-split vs BASELINE [fp32/randn] 2048×64 | 半段 fp32 链 | 源公式 fp32 | 131,072 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** |
| #1 half-split vs BASELINE [fp32/randn] 4096×128 | 半段 fp32 链 | 源公式 fp32 | 524,288 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** |
| #1 half-split vs BASELINE [fp32/corners] 256×64 / 2048×64 / 4096×128 | 半段 fp32 链 | 源公式 fp32 | 同左（含 ±inf/NaN/±0/次正规/±65504） | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #1 half-split vs BASELINE [fp16/randn] 256×64 / 2048×64 / 4096×128 | 半段 fp32 链→fp16 | 源公式 fp32→fp16 | 16K/131K/524K | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #1 half-split vs BASELINE [fp16/corners] ×3 | 同上 | 同上 | 含角点 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #1 half-split vs BASELINE [bf16/randn] 256×64 / 2048×64 / 4096×128 | 半段 fp32 链→bf16 | 源公式 fp32→bf16 | 16K/131K/524K | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #1 half-split vs BASELINE [bf16/corners] ×3 | 同上 | 同上 | 含角点 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #2 fp32-transit vs GOLDEN [fp32] 256×64 / 2048×64 / 4096×128 | 半段 fp32 链 | `ref_rope_neox` fp32 | 16K/131K/524K | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #2 fp32-transit vs GOLDEN [fp16] ×3 | 半段 fp32 链→fp16 | `ref_rope_neox` fp32→fp16 | 同左 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #2 fp32-transit vs GOLDEN [bf16] ×3 | 半段 fp32 链→bf16 | `ref_rope_neox` fp32→bf16 | 同左 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |
| #3 cos/sin 复用 vs 逐元素重载 [fp32/fp16/bf16] | 复用 tile | 逐元素重取 | 512×64 | 0.000e+00 | 0 | 0.0000 | **EQUIV_PASS** ×3 |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2；`equal_nan`），违反率 0 即 PASS；覆盖随机张量 + 角点（±0/±inf/NaN/次正规 1e-45/±65504/±88.8/±1e-8）。**fp64 参照**：`candidate vs baseline` fp64 `max_abs=0.0000e+00`、`ulp=0`（结构重排零近似）。

**非门禁诊断（机器输出）——被拒候选 E1 的等价性证据**（说明输入 dtype 域**不可**采用，故 fp32 中转是正确性要求而非性能偏好）：

| 被拒候选 | dtype | max_abs | 违反比 | 结论 |
|----------|-------|---------|--------|------|
| 输入 dtype 原生域（跳过 fp32 中转） | fp32 | 0.000e+00 | 0.0000 | 等价（fp32 无域差） |
| 输入 dtype 原生域（跳过 fp32 中转） | fp16 | 7.8125e-03 | 0.0001 | **超 atol 1e-3 → 非等价**（viol 非 0） |
| 输入 dtype 原生域（跳过 fp32 中转） | bf16 | 6.2500e-02 | 0.0000 | **超 atol 1.6e-2 → 非等价** |

→ E1 弃选理由 = **正确性**（golden fp32 opmath）+ bf16 API 缺失，与 hardtanh（投影类，原生域逐位等价、弃选理由是 ROI）**相反**（对照 `PL-1.27-same-family-precedent-precondition`：同族先例须绑定各自契约前提）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `y=val*c+rot*s` 标量映射 | 源 `T.Parallel(threads,npt)` 内标量（每线程 npt 元素） | 半段 2D 整 buffer 向量链 `vmul/vadd/vsub`（佐证 `docs/.../数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.2 + §2.4；`examples/elementwise/vec_add_2d.py`） | ✅ | — |
| 2 | `rot = col<half ? −x[row,col+half] : x[row,col−half]` 逐元素配对/select | 源逐元素 `if_then_else`（gather） | **消除**：半段结构将配对变为同一 tile 内两组连续子块 `a_ub`/`b_ub`（无 select、无 gather） | ✅ | — |
| 3 | `freq_idx = col % half` + `cos_table[row,freq_idx]` 逐元素取表 | 源逐元素 `%` + 索引 | **消除**：cos/sin 装载为 `(br,half)` tile，`y1`/`y2` 复用（CSE） | ✅ | — |
| 4 | `row = flat_idx // head_dim`、`col = flat_idx % head_dim` | 源逐元素 flat↔2D 坐标换算 | **消除**：2D tile 静态寻址（`r0=task*block_rows`，列维 0/half 静态偏移） | ✅ | — |
| 5 | fp16/bf16 → fp32 提升 / 舍回 | 源无（输入域计算） | `T.vcast(src, dst, round_mode="rint")`（升/降；佐证 `docs/.../数据类型转换操作/T.vcast.md` §2.2.1/§2.4） | ✅ | — |
| 6 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | **tile/block 级任务调度循环**：每核迭代数 `ceildiv(n_num,40)`（≤7），非逐元素热点；元素级并行由向量指令整 buffer 完成。先例 `vec_add_2d.py`、`PL-1.22-persistent-1d-grid-stride` |
| 7 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`r0=task*block_rows` | 无 | ❌ | **block 级索引/任务映射**：每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 8 | 行尾边界（非整除 `seq_len`） | `tr = T.min(block_rows, seq_len − r0)` | 无 | ❌ | **依赖（编译期）shape 的边界处理**：`seq_len` 为编译期常量但 `r0` 依赖运行时 `cid`，尾块按 `T.min` 动态裁剪 `T.copy`；manifest workload 的 `seq_len` 均整除 `block_rows`，保留尾守卫以支持任意 `seq_len`（纯静态化对通用 shape 不成立） |

**向量化结论**：逐元素计算点（#1–#5）**已全部向量化**（`vmul/vadd/vsub/vcast`，均有 docs 佐证 + `examples/elementwise/` 实跑先例）；保留 3 处标量/循环（#6 核内调度循环、#7 block 索引、#8 行尾边界），理由类别：**tile 级调度循环 + block 索引 + 动态边界**，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

本算子有 2 个非 batch 维（`seq_len`、`head_dim`），须枚举候选布局 × 向量化轴。候选矩阵（逐候选量化）：

| 候选 | 布局 / 向量化轴 | 整除性 / 尾 lane 浪费 | 累加链形态 | repack 代价 | UB 影响 | 结论 |
|------|----------------|----------------------|-----------|------------|---------|------|
| **L1（主选）** | I/O 原生行主布局；**向量轴 = `half` 内层连续轴**（`a_ub/b_ub/c_ub/s_ub` 各 `(block_rows, half)`）；行块 = 并行/循环轴 | `half` 元素：fp16 `half=64`=128B、`half=32`=64B；fp32 `half=32`=128B；均 32B 整倍；向量宽 fp16×8 / fp32×4 整除 `half∈{32,64}`；**尾 lane 浪费 0%** | 纯逐元素，**无累加链**（无串行累加维） | **无**（I/O 布局 = 核内布局；cos/sin 原生 `(seq,half)` 直接对应） | ~7 fp32 + ~4 dtype buffer；d128 fp16 = 36 KB（§4.5）✓ | ✅ **主选** |
| L2 | flat 一维布局（源），向量轴 = flat 元素轴 + 逐元素 `col%half`/gather | flat 连续 ✓，但配对元素需跨步 gather | 无累加链，但 gather | 需 gather（**跨步 gather 向量指令缺失**：`TRAP-C10-strided-gather`） | 需按元素临时量 | ❌ **弃选**：gather 无向量指令，退化为标量/低效（依据：`TRAP-C10` 限制条款；NPU 无等价廉价 gather，`npub::gather` 为显式 `T.npuir_gather`（`tilelang/language/customize_npuir.py::npuir_gather`）索引搬移，非逐元素 ALU，代价不优） |
| L3 | I/O 原生；**整行装载** `x_ub(block_rows, head_dim)` 后 UB 内拆半（`x_ub[:, 0:half]` / `x_ub[:, half:]` 作 v-op 操作数） | 行 128B（d64 fp16）/256B，段更宽；**但 v-op 需接受静态 2D 子切片操作数**（`TRAP-expert-v-operands`：v-op 切片**仅接受静态边界**，runtime Var 被拒——`half` 为静态常量，理论上合法） | 无累加链 | 无 | +1 `(br,head_dim)` 装载 buffer，或需额外 UB↔UB 拷贝拆半 | ⚠️ **备选**（实验裁决）：单次宽行装载（MTE2 段 128B/256B vs L1 的 64B/128B）**可能**更优，但依赖 v-op 静态 2D 子切片可用性与段效率两个**未实证常数** → 见下「实验裁决」 |
| L4 | 核内转置（`seq_len` 作内层向量轴） | `seq_len` 非连续（行距 `head_dim`），跨步系数阻碍向量化；cos/sin 按行索引 → 转置后广播更复杂 | 无累加链 | 需 `T.transpose`（UB 级重排/repack） | 增 1~2 buffer | ❌ **弃选**：转置不减少任何访存（GM 流量不变），却引入 repack 开销 + cos/sin 按行广播复杂化；无收益（依据：`ALG-transpose` R3「GM 流量不变，代价在向量管线开销」） |

**逐候选量化评分**：L1 在整除性（尾 lane 0%）、repack（无）、UB（36 KB）、gather（无）四维全部达标；L2 因 gather 移除（`TRAP-C10`）；L3 保留为备选（依赖未实证段效率）；L4 无收益移除。**迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → flat 元素）是**输入而非结论**——GPU 的 flat 元素轴与 NPU Vector 的连续 `half` 轴不对应（GPU 靠 L2 + 逐元素 select 掩盖 gather），§0.6 R1/R2 已独立评估为「行块 + 半段 + `half` 向量轴」。

**布局决策结论**：主选 **L1**——核内布局 = I/O 原生行主布局（无 repack），向量化轴 = `half` 内层连续轴，行块为并行/循环轴；repack 路径 = 无。**决策落地**：主选轴与布局同步落入 §3.3 伪代码（`a_ub/b_ub/c_ub/s_ub` 均为 `(block_rows, half)`）与 §6 循环结构（向量链各 op 作用于 `(br,half)` 整 buffer，无逐元素标量循环）。

**实验裁决（判定裕度依赖未实证常数，强制三件套）**：主选 L1（拆半装载，`half` 段 64–128B）与备选 L3（整行装载，段 128–256B）的判定裕度落在两个未实证常数上——① MTE2 小段（64B）相对宽段（128B/256B）的效率损失（`CONST-mte2-degradation` 记 ≈128B 段、~2.9–3.2ns/段，**stale**，且未提供 64B 段实测）；② v-op 静态 2D 子切片操作数在 NPU 的可用性与代价（`TRAP-expert-v-operands` 仅证 runtime Var 被拒，静态边界理论合法但无本算子形态实测）。

- **主选方案（默认实现，进 Stage 3）**：L1 拆半装载（§3.3）：`T.copy(x[r0:r0+tr, 0:half], a_ub)` + `T.copy(x[r0:r0+tr, half:head_dim], b_ub)`，`c_ub/s_ub` 同理；6-op 向量链。理由 = 零 gather、无切片依赖、UB 预算最小、尾 lane 0%。
- **备选方案（结构完整，达 Stage 4 可直接实现深度）**：L3 整行装载——`x_ub (block_rows, head_dim)` 单次 `T.copy(x[r0:r0+tr, :], x_ub[0:tr, :])`；向量链操作数用**静态 2D 子切片** `x_ub[:, 0:half]`、`x_ub[:, half:head_dim]`（`half` 为编译期常量）；输出写回用两段 `T.copy`（`y_ub1`/`y_ub2` 各 `(br,half)`）或一个 `y_ub (br,head_dim)`。buffer 形状：`x_ub (br,head_dim)` + `c_ub/s_ub (br,half)` + fp32 `af/bf??`——注意 L3 下 `af/bf` 需为 `(br,half)` 视图或额外拆半 buffer（若子切片不可作 vcast dst，则退回「整行 vcast + 切片」并增 1~2 buffer）。UB 预算：`x_ub` fp16 d128 = 16×128×2 = 4 KB + 其余 ~36 KB ≈ 40 KB（×1.7 ≈ 68 KB，可行）。分核三要素同 §5.5（`block_rows` 不变）。dtype 路径同 L1。
- **实验裁决计划**：代表 shape = manifest 全部 1d workload（`[256,64]`fp16/bf16/fp32、`[2048,64]`fp16、`[4096,128]`bf16）；测量指标 = msprof op Task Duration（median-of-20，warm-up 5）+ `aic_mte2_*` 忙比（判读段效率）；**未知常数反解** = 由 L1 vs L3 的 MTE2 时间差与段数差反解 64B vs 128B 段的单位代价、并验证静态 2D 子切片 v-op 是否编译通过；**判定阈值** = 备选 latency < 主选 × (1−0.05)（即 ≥5% 增益才翻转；<5% 视为 tie，保留主选）。裁决结果回写 §1.6.3 并（若翻转）修订 §3.3/§4/§6。**shape 分派评估**：工厂为 `functools.lru_cache` 形状特化，故若按 shape 分片（如 `head_dim=128` 用 L3、`head_dim=64` 用 L1）更优，可在 Stage 4 按 manifest 分派主选/备选，达成总体最优而非全局二选一——裁决计划须评估该分派可行性。
- **设计期探针（D-3，可选前置，本设计未执行）**：所依赖常数（64B vs 128B 段效率、静态 2D 子切片 v-op 可用性）可在设计期以 ≤2 个轻量探针实测（最小 kernel），脚本落 `history_version/design_probe_*.py`；本设计按预算保守选择「不执行设计期探针，交 Stage 4 实验裁决」（常数不确定性不影响主选结构正确性）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素/局部配对 2×2 旋转向量链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约（逐元素 + 行内配对，无累加序） | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`T.vmul/vadd/vsub/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致，且 wrapper `RopeNeoxKernel.ascend_mode = "Developer"` 标注一致；参考 `examples/elementwise/vec_add_2d.py`（Developer + `T.alloc_ub` + v-prefix + 2D tile + `T.serial` 网格跨步）、`examples/bitwise_not/bitwise_not_fwd_kernel/`（Developer 一维 persistent）与 `testing/npuir/bf16_support_ops/test_vv_bf16.py`（Developer 2D `vmul/vadd` 先例）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_rows, half), dtype)` 2D 显式 UB 分配（v-prefix API 要求 buffer 在 UB） |
| 计算方式 | v-prefix API 向量链（整 buffer），非 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（半段 2D 向量链 + sub-fp32 fp32 中转 + cos/sin 复用）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `a_tile = x[r0:r0+tr, 0:half]`；`b_tile = x[r0:r0+tr, half:head_dim]` | 搬入行块的两半到 UB |
| 2 | `c_tile = cos[r0:r0+tr, 0:half]`；`s_tile = sin[r0:r0+tr, 0:half]` | 搬入 cos/sin tile（两半共用） |
| 3 | `af/bf/cf/sf = cast↑f32(·)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 4 | `y1f = a*c − b*s`；`y2f = b*c + a*s` | 半段 2×2 旋转（fp32 域，6 op） |
| 5 | `y1 = cast↓dtype(y1f)`；`y2 = cast↓dtype(y2f)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 6 | `y[r0:r0+tr, 0:half] = y1`；`y[r0:r0+tr, half:head_dim] = y2` | 两半写回 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 x 前半 | `a = x[:, 0:half]` | `T.copy` | `T.copy(x[r0:r0+tr, 0:half], a_ub[0:tr, 0:half])` | Developer |
| 搬入 x 后半 | `b = x[:, half:]` | `T.copy` | `T.copy(x[r0:r0+tr, half:head_dim], b_ub[0:tr, 0:half])` | Developer |
| 搬入 cos/sin | `c = cos`；`s = sin` | `T.copy` | `T.copy(cos_table[r0:r0+tr, 0:half], c_ub[0:tr, 0:half])` | Developer |
| cast 升 fp32（fp16/bf16） | `af = cast32(a)` | `T.vcast` | `T.vcast(a_ub, af, round_mode="rint")` | Developer |
| 乘 `a*c` / `b*s` / `b*c` / `a*s` | `vmul` | `T.vmul` | `T.vmul(A, C, y1f)` / `T.vmul(B, S, t)` / `T.vmul(B, C, y2f)` / `T.vmul(A, S, t)` | Developer |
| `y1 = a*c − b*s` | `vsub` | `T.vsub` | `T.vsub(y1f, t, y1f)`（`t=b*s`；原地落回 `y1f`） | Developer |
| `y2 = b*c + a*s` | `vadd` | `T.vadd` | `T.vadd(y2f, t, y2f)`（`t=a*s`；原地落回 `y2f`） | Developer |
| cast 舍回（fp16/bf16） | `y1 = cast16(y1f)` | `T.vcast` | `T.vcast(y1f, a_ub, round_mode="rint")`（`a_ub` 死亡后复用） | Developer |
| 搬出两半 | `y[:, 0:half]=y1`；`y[:, half:]=y2` | `T.copy` | `T.copy(a_ub[0:tr, 0:half], y[r0:r0+tr, 0:half])` / `T.copy(b_ub[0:tr, 0:half], y[r0:r0+tr, half:head_dim])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(40, is_npu=True)` + `for i in T.serial(ceildiv(n_num, 40))` | Developer |
| 行尾守卫 | `tr = min(br, seq_len−r0)` | `T.copy`（切片 extents 借用） | `tr = T.min(block_rows, seq_len - r0)` + `[0:tr, 0:half]` | Developer |
| UB 分配 | — | `T.alloc_ub` | `T.alloc_ub((block_rows, half), dtype/f32)` | Developer |

> **注（`T.vadd`/`T.vsub`/`T.vmul` 的 rank 与原地一致）**：三 op doc 均要求「参与运算的 tensor 必须 rank 相同，推荐 2D buffer」——本算子全部操作数为 `(block_rows, half)` 2D buffer，`dst` 支持与 src 原地（`T.vadd.md`/`T.vsub.md`/`T.vmul.md` §2.3 第 3 条）。`T.vmul` 仅 `src2` 可标量、`T.vadd`/`T.vsub` 仅 `src1` 可标量——本算子**无标量操作数**，规避该不对称性。

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数**数据类型直接作为第二个位置参数**（不用 `dtype=` 关键字）；`prim_func` 参数名/顺序与 wrapper 契约一致（`x, cos_table, sin_table, y`，`out_idx=[3]`，**无额外运行时 shape 参数**——`seq_len` 为编译期常量，wrapper 约定 `fn(x, cos, sin)`）；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（`PL-1.22`/`TRAP-expert-v-operands`）；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）；计算域别名用 **Python 三元表达式**（非运行时 `if`，避免 `TRAP-tvm-parser-rules` 的作用域问题）。

```python
import tilelang
import tilelang.language as T


def _make_rope_neox_1d(seq_len, head_dim, dtype, block_size=2048):
    """NPU (target=npuir) RoPE NeoX, layout=1d.

    x:(seq_len, head_dim)  cos/sin:(seq_len, half)  y:(seq_len, head_dim)
    y[:, :half] = a*c - b*s ;  y[:, half:] = b*c + a*s   (a=x[:,:half], b=x[:,half:])
    wrapper K11: block_size = GPU threads*num_per_thread (fp32 1024 / fp16,bf16 2048).
    """
    half = head_dim // 2
    f32 = "float32"
    need_cast = (dtype != f32)                       # fp16/bf16 -> fp32 中转（编译期常量）
    assert block_size % head_dim == 0, "block_size must be a multiple of head_dim"
    block_rows = max(1, min(block_size // head_dim, seq_len))   # 行/tile（编译期静态）
    core_num = 40                                    # 实查物理 Vector 核数（20 aicore x 2, §5.5 ②）
    n_num = (seq_len + block_rows - 1) // block_rows            # 逻辑 tile 数（编译期静态）
    iters = (n_num + core_num - 1) // core_num                  # 核内串行上界（编译期静态）

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(blk):                                 # blk == block_size（wrapper 调用约定）
        @T.prim_func
        def main(
            x: T.Tensor((seq_len, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((seq_len, head_dim), dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, _):
                # --- buffer 顶层无条件分配（TRAP-conditional-alloc）---
                a_ub = T.alloc_ub((block_rows, half), dtype)   # x 前半（死亡后复用为 y1 输出）
                b_ub = T.alloc_ub((block_rows, half), dtype)   # x 后半（死亡后复用为 y2 输出）
                c_ub = T.alloc_ub((block_rows, half), dtype)   # cos
                s_ub = T.alloc_ub((block_rows, half), dtype)   # sin
                af = T.alloc_ub((block_rows, half), f32)
                bf = T.alloc_ub((block_rows, half), f32)
                cf = T.alloc_ub((block_rows, half), f32)
                sf = T.alloc_ub((block_rows, half), f32)
                y1f = T.alloc_ub((block_rows, half), f32)
                y2f = T.alloc_ub((block_rows, half), f32)
                t = T.alloc_ub((block_rows, half), f32)

                # 编译期计算域别名（Python 三元；fp32 -> 输入 buffer 即计算域）
                A = af if need_cast else a_ub
                B = bf if need_cast else b_ub
                C = cf if need_cast else c_ub
                S = sf if need_cast else s_ub

                for i in T.serial(iters):                # 循环上界静态
                    task = i * core_num + cid            # 网格跨步
                    if task < n_num:                     # 尾块守卫
                        r0 = task * block_rows
                        tr = T.min(block_rows, seq_len - r0)

                        T.copy(x[r0:r0 + tr, 0:half], a_ub[0:tr, 0:half])
                        T.copy(x[r0:r0 + tr, half:head_dim], b_ub[0:tr, 0:half])
                        T.copy(cos_table[r0:r0 + tr, 0:half], c_ub[0:tr, 0:half])
                        T.copy(sin_table[r0:r0 + tr, 0:half], s_ub[0:tr, 0:half])

                        if need_cast:                    # Python 编译期条件（trace 期折叠）
                            T.vcast(a_ub, af, round_mode="rint")
                            T.vcast(b_ub, bf, round_mode="rint")
                            T.vcast(c_ub, cf, round_mode="rint")
                            T.vcast(s_ub, sf, round_mode="rint")

                        # --- 半段 2×2 旋转（fp32 域，整 (br,half) buffer）---
                        T.vmul(A, C, y1f)                # a*c
                        T.vmul(B, S, t)                  # b*s
                        T.vsub(y1f, t, y1f)              # y1 = a*c - b*s
                        T.vmul(B, C, y2f)                # b*c
                        T.vmul(A, S, t)                  # a*s
                        T.vadd(y2f, t, y2f)              # y2 = b*c + a*s

                        if need_cast:
                            T.vcast(y1f, a_ub, round_mode="rint")   # a_ub 已死，复用为 y1
                            T.vcast(y2f, b_ub, round_mode="rint")   # b_ub 已死，复用为 y2
                            T.copy(a_ub[0:tr, 0:half], y[r0:r0 + tr, 0:half])
                            T.copy(b_ub[0:tr, 0:half], y[r0:r0 + tr, half:head_dim])
                        else:
                            T.copy(y1f[0:tr, 0:half], y[r0:r0 + tr, 0:half])
                            T.copy(y2f[0:tr, 0:half], y[r0:r0 + tr, half:head_dim])

        return main
    return kernel
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件（`dtype` 编译期确定），trace 期折叠，不产生运行时分支；buffer 全部在 `with T.Kernel` 顶层**无条件分配**（`TRAP-conditional-alloc`）。② **计算域别名 `A/B/C/S`** 用 Python 三元表达式求值（外层是表达式而非 `if` 语句，规避 `TRAP-tvm-parser-rules` 的 `if` 作用域困绑）；fp32 路径 `A=a_ub` 等（无中转），fp16/bf16 路径 `A=af` 等。③ v-op 全部作用于**整 buffer**；`T.copy` 用 `[0:tr, 0:half]` 动态行切片（`TRAP-expert-v-operands`：v-op 切片仅静态边界，故行切片只出现在 `T.copy`）。④ **buffer 复用**：`a_ub`/`b_ub` 在 `vcast` 升 fp32 后死亡，链尾复用为 `y1`/`y2` 输出（`PL-1.23-vop-chain-inplace-alias` 的「读前写后」纪律——写发生在 `vcast(a_ub→af)` 之后）。⑤ `T.vsub(y1f, t, y1f)` / `T.vadd(y2f, t, y2f)` 为原地（`dst==src0`，doc §2.3 允许）。**op 计数**：fp32 = 6；fp16/bf16 = 12（+4 升 +2 舍回）。
>
> **接口适配说明（wrapper 契约遵守）**：集成 wrapper `rope_neox.py` 约定 NPU 工厂签名 `_make_rope_neox_1d(seq_len, head_dim, dtype)`（K3/K11 去 `threads`/`num_per_thread`），返回可调用 `kernel(block_size)`，`prim_func` 为 `main(x, cos_table, sin_table, y)`（`out_idx=[3]`），`forward` 调 `self._compiled_fn(x, cos, sin)`。本设计**完全对齐**：工厂 3 位置参数 + 返回 `block_size` 单参可调用；`main` 保留 x/cos/sin/y 的名与顺序；**不新增运行时 `shape` 参数**（`seq_len` 编译期常量，尾块用编译期 `seq_len` 表达）。`block_size` 语义 = K11 折叠的「每 tile x 元素数」（`block_rows = block_size // head_dim`）。

### 3.4 API 可行性确认

| API | 来源确认（本地） | 状态 |
|-----|-----------------|------|
| `T.vmul(src1, src2, dst)`（逐元素乘，2D，dst 可原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.1/§2.2.2/§2.3/§2.4；`testing/npuir/bf16_support_ops/test_vv_bf16.py`（Developer 2D `vmul` 实跑） | ✅ 真实可用（fp16 ✓ / fp32 ✓ / bf16 ✗ 按 doc） |
| `T.vadd(src0, src1, dst)` / `T.vsub(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vadd.md`·`T.vsub.md` §2.1/§2.2.2/§2.3/§2.4 | ✅ 真实可用（fp16 ✓ / fp32 ✓ / bf16 ✗ 按 doc） |
| `T.vcast(src, dst, round_mode="rint")` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32`/`f32→f16/bf16` 支持 `rint`）/§2.4 | ✅ 真实可用 |
| `T.copy`（2D slice，动态行尾块） | `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1/2（`A[bx:bx+tile_M, by:by+tile_N] → A_ub[0:tile_M,0:tile_N]`）；`examples/elementwise/vec_add_2d.py` | ✅ 已核对 |
| `T.alloc_ub`（2D） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.4（`T.alloc_ub((M,N), dtype)`）；`examples/elementwise/vec_add_2d.py` | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `examples/elementwise/vec_add_2d.py`（`T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial` + `if block_id < ...`）；`PL-1.22-persistent-1d-grid-stride` | ✅ 已核对 |
| `T.min`（行尾块 extents） | `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1（`tile_M = T.min(block_M, M - bx)`） | ✅ 已核对 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(40, is_npu=True)` |
| GPU 专用 API | Yes（源 `threads=`、`T.Parallel(threads,npt)`、flat 逐元素 select/gather） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；行尾非整除由 `tr = T.min(...)` + `T.copy` 动态切片处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数（8–256）与物理核 40 的适配见 §5.5（网格跨步 persistent） |
| **bf16 v-prefix 数学 API 缺失（按 doc）** | Yes | `vmul/vadd/vsub` 不含 bf16（doc §2.2.1）→ sub-fp32 走 fp32 中转（§0.6 R2）；**文档/示例分歧**见 §9 R-3（本算子不依赖原生 bf16 路径） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| v-op 动态 slice 不可用 | Yes | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tr, 0:half]`（§3.3） |
| `T.vbrc` 宽 1 广播被拒 | No | 本算子 `half ≥ 32`，无宽 1 轴广播（登记 `TRAP-vbrc-empty-broadcast`，不触发） |
| v-op 无 fused multiply-add | Yes | 无 `vmadd`/`vaxpy`（`tilelang/language/customize_npuir.py` 无该 op）→ 用 4 `vmul` + 2 add/sub 显式链（§1.6.1，候选 #3 的 3 乘方案不做融合替代） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[3])` | `@tilelang.jit(out_idx=[3], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N_total, block_size), threads=256)` | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 整 buffer 向量指令 | K11：`threads×npt` 折叠为 `block_size` |
| **计算核** | 逐元素 `if_then_else` 配对 + `val*c+rot*s`（gather） | 半段 2D 向量链 `y1=a*c−b*s`、`y2=b*c+a*s`（6 op，零 gather） | §0.6 R2 + §1.6.1 #1 |
| **计算域** | 输入 dtype 域 | **fp32 中转**（`vcast` 升 → 链 → `vcast` 舍回） | §0.6 R2 + §1.6.1 #2 |
| **cos/sin 访存** | 逐元素按 `col%half` 取（逻辑 2× 读，L2 吸收） | 行块单次装载 + 两半复用（1× 读） | §0.6 R3 + §1.6.1 #3 |
| Buffer 层级 | 纯寄存器（无显式 SMEM/fragment） | `T.alloc_ub`（UB）2D tile | §0.5 等价替换 |
| prim_func 参数 | `main(x, cos_table, sin_table, y)` | **同**（`out_idx=[3]`，无额外 shape） | 对齐 wrapper 契约 |
| strategy/autotune | `threads`/`num_per_thread` + `lru_cache` | 无 strategy 维度；`block_size` 单参 + `lru_cache` | K8/K9/K11 |
| layout=2d / non_neox / llama31 / yarn / longrope | 同文件其它工厂/类 | 无 | **不迁移**（本迁移只 port `RopeNeoxOp.layout="1d"`） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_2d.py` | 高度相似（一维核 + 2D tile + 网格跨步） | `T.Kernel(40, is_npu=True)` + `T.serial` + `T.copy(A[bx,by], A_VEC)` + 2D `T.vadd` |
| `examples/bitwise_not/bitwise_not_fwd_kernel/` | 高度相似（Developer 一维 persistent 迁移） | 一维 persistent 迁移形态、核内串行网格跨步、`shape` 守卫 |
| `examples/hardtanh/_make_hardtanh_kernel/` | 高度相似（同族迁移，fp32 中转 + 向量链） | 迁移文档结构、fp32 中转链、`verify_equiv.py` 范式、条件常量折叠 |
| `examples/leaky_relu/_make_leaky_relu_kernel/` | 相似（同族参数化逐元素迁移） | 迁移文档结构、`verify_equiv.py` 范式 |
| `examples/dropout/_make_dropout_kernel/` | 相似（同族迁移，v-op dtype 矩阵核对） | fp32 中转链 + bf16 API 缺失处置 + v-op 操作数形态 |
| `testing/npuir/bf16_support_ops/test_vv_bf16.py` | 参考（API 先例） | Developer 2D `T.vmul`/`T.vadd` 用法（`T.alloc_shared` 2D） |
| `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x,cos,sin,y)` 契约 + `ascend_mode="Developer"` |
| `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md`·`数据类型转换操作/T.vcast.md`·`内存操作/T.copy.md` | API | 向量链各 op 规格、dtype 矩阵与用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(seq_len, head_dim)`（编译期常量） | `float16 \| bfloat16 \| float32` | 行主连续；manifest 1d：`[256,64]`/`[2048,64]`/`[4096,128]` |
| `cos_table` | `(seq_len, head_dim // 2)` | 同 `x` | Op 层生成（fp32 计算后 `.to(dtype)`） |
| `sin_table` | `(seq_len, head_dim // 2)` | 同 `x` | 同上 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(seq_len, head_dim)` | `same_as(x)` | `out_idx=[3]` 自动分配，shape/dtype 与输入一致，无转置 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a_ub` | `(block_rows, half)` | `dtype` | UB | `x` 前半；**死亡后复用为 `y1` 输出** |
| `b_ub` | `(block_rows, half)` | `dtype` | UB | `x` 后半；**死亡后复用为 `y2` 输出** |
| `c_ub` | `(block_rows, half)` | `dtype` | UB | cos tile |
| `s_ub` | `(block_rows, half)` | `dtype` | UB | sin tile |
| `af`/`bf`/`cf`/`sf` | `(block_rows, half)` ×4 | `float32` | UB | fp32 计算域（fp16/bf16 升入；fp32 路径未用，仍顶层分配） |
| `y1f`/`y2f` | `(block_rows, half)` ×2 | `float32` | UB | 两半的 fp32 结果 |
| `t` | `(block_rows, half)` | `float32` | UB | 临时积（`b*s` / `a*s`） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；**`a_ub`/`b_ub` 死亡后复用为输出**（`PL-1.23` 读前写后纪律）；fp32 路径 `af/bf/cf/sf` 分配后未用（UB 冗余 ≈8 KB，可接受，避免条件 alloc）。

### 4.4 内存搬运路径

```
fp16/bf16 路径（fp32 中转）：
GM[x] --T.copy--> UB[a_ub(前半)/b_ub(后半)] --T.vcast(rint)--> UB[af/bf(f32)]
GM[cos/sin] --T.copy--> UB[c_ub/s_ub] --T.vcast(rint)--> UB[cf/sf(f32)]
   --[y1f=vmul(af,cf); t=vmul(bf,sf); y1f=vsub(y1f,t); y2f=vmul(bf,cf); t=vmul(af,sf); y2f=vadd(y2f,t)]--> UB[y1f/y2f(f32)]
   --T.vcast(rint)--> UB[a_ub/b_ub (复用为输出, dtype)] --T.copy--> GM[y(前半/后半)]

fp32 路径（无中转；A=a_ub, B=b_ub, C=c_ub, S=s_ub）：
GM[x] --T.copy--> a_ub/b_ub ; GM[cos/sin] --T.copy--> c_ub/s_ub
   --[同 6-op 向量链]--> UB[y1f/y2f] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

`block_size` = K11 折叠值（fp32=1024、fp16/bf16=2048）；`block_rows = block_size // head_dim`；每 tile 每 buffer 元素数 `E = block_rows × half`。

**UB 逐 buffer 预算表（最大情形：fp16/bf16, head_dim=128 → block_rows=16, half=64, E=1024）**：

| buffer | dtype | 元素数 | 单 buffer 大小 |
|--------|-------|--------|---------------|
| a_ub | fp16/bf16 | 1024 | 2 KB |
| b_ub | fp16/bf16 | 1024 | 2 KB |
| c_ub | fp16/bf16 | 1024 | 2 KB |
| s_ub | fp16/bf16 | 1024 | 2 KB |
| af | fp32 | 1024 | 4 KB |
| bf | fp32 | 1024 | 4 KB |
| cf | fp32 | 1024 | 4 KB |
| sf | fp32 | 1024 | 4 KB |
| y1f | fp32 | 1024 | 4 KB |
| y2f | fp32 | 1024 | 4 KB |
| t | fp32 | 1024 | 4 KB |
| 合计 | — | 11264 | 36 KB |

> **各 dtype/head_dim 情形**（`E = block_rows × half`）：fp16/bf16 d64（block_rows=32, half=32, E=1024）与 d128 同为 36 KB；fp32 d64（block_rows=16, half=32, E=512）与 fp32 d128（block_rows=8, half=64, E=512）同为 22 KB（4 个 dtype buffer 即 fp32 + 7 个 fp32（含未用 af~sf）× 512 × 4B = 8 KB + 28 KB）。
>
> **UB 硬上限 = 192 KB/AIV**（`CONST-capacity-910B2C`，绑定 Ascend910B2C **且 `kb_stale_check` 标 stale**；本设备 Ascend910_9362 同量级容量）。最大路径（fp16/bf16）= 36 KB，**×1.7 最坏膨胀 ≈ 61 KB < 192 KB ✓**。**膨胀系数注（CG-2026-0008）**：显式 alloc 结构实测 ~12% 余量即足（正负两向偏差均无文档，故按最坏上偏 1.7× 保守核对）。**block_size 放大空间（容量口径）**：sub-fp32 d64 `bs=4096` → block_rows=64 → E=2048 → 72 KB（×1.7 ≈ 122 KB，可行）；`bs=8192` → 144 KB（×1.7 ≈ 245 KB，**超限**）。故 `bs` 放大是 Stage 4 发射项调优杠杆（§5.2/§1.6.3 备选），默认取 K11 折叠值。

### 4.6 动态轴定义

**无运行时动态轴**：`seq_len`/`head_dim` 均为编译期常量（`lru_cache` 特化；wrapper 亦按 `(seq_len, head_dim, dtype)` 固定）。行尾块由编译期常量 `seq_len` 与运行时 `r0` 组合表达 `tr = T.min(block_rows, seq_len - r0)`，**不引入独立运行时 shape 参数**（对齐 wrapper 契约 `fn(x, cos, sin)`）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[3],        # y 是 prim_func 第 4 个参数（index=3）
    target="npuir",
)
def kernel(blk):        # blk == block_size（K11 折叠值）
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素/局部配对向量链（`vmul/vadd/vsub/vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
block_rows = min(block_size // head_dim, seq_len)   # 行/tile（编译期静态）
n_num = ceildiv(seq_len, block_rows)                # 逻辑 tile 数（编译期静态）
```

| dtype | head_dim | block_size | block_rows | half | 每 tile 行数 | tile x 元素数 |
|-------|----------|-----------|-----------|------|-------------|--------------|
| fp16/bf16 | 64 | 2048 | 32 | 32 | 32 | 2048 |
| fp16/bf16 | 128 | 2048 | 16 | 64 | 16 | 2048 |
| fp32 | 64 | 1024 | 16 | 32 | 16 | 1024 |
| fp32 | 128 | 1024 | 8 | 64 | 8 | 1024 |

**block_size 选择理由**：
1. 与源 K11 折叠值一致（`threads(256)×npt`：fp32 1024 / fp16·bf16 2048），迁移可追溯；v-prefix 整 buffer 形态下默认值无 128-bit 对齐差异；
2. UB 对齐：`(block_rows, half)` 2D buffer 行 = `half` 元素（fp16 `half=64` 128B / `half=32` 64B；fp32 `half=32` 128B），均为 32B 整倍 ✓；
3. UB 容量：最大路径（fp16/bf16）≈ 36 KB，×1.7 ≈ 61 KB < 192 KB ✓；
4. **发射项权衡**：本算子为 6–12 op 短链，§1.6.0 估算显示发射项名义主导（发射 ∝ 1/`block_rows`），但**吞吐口径交叉核对**（`CONST-vector-op-cost-910_9362`）显示实际由流量项主导、发射项被隐藏；`bs` 放大受 UB 膨胀系数不确定性约束。默认取 K11 折叠值；**`bs=4096` 列为 Stage 4 首选杠杆**（须以编译探针校准 UB 后采纳，属分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：2D tile 行 = `half` 元素（fp16 `half=64` 128B、`half=32` 64B；fp32 `half=32` 128B），整 buffer 连续，行距无跨步系数（核内布局紧凑）；满足 32B 尾轴对齐 ✓；向量宽 fp16×8 / fp32×4 整除 `half∈{32,64}`，尾 lane 浪费 0% ✓。
- **UB 容量**：最大 36 KB（fp16/bf16），×1.7 最坏膨胀 ≈ 61 KB < 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。
- **rank 一致**：全部 v-op 操作数为 2D `(block_rows, half)`（doc 推荐统一 2D）✓。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest 1d workload 的 `seq_len ∈ {256, 2048, 4096}`——`256/32=8`、`2048/32=64`、`4096/16=256` **均整除** `block_rows`，无尾块；但源仓测试 `RopeEdgeFixture`（`seq_len=1`，2d）提示通用 `seq_len` 可能很小/非整除，故**保留尾块守卫**。

**策略**：Kernel 内动态行尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 tile 用 `tr = T.min(block_rows, seq_len - r0)`（`r0 = task*block_rows`）裁剪 `T.copy` 的一维行 extent（`[0:tr, 0:half]`）；向量 op 仍作用于**整 buffer**（尾部行的陈旧数据不影响仅写出的 `[0:tr, 0:half]`；`PL-1.22` 尾块零起点切片纪律）。`block_rows = min(block_size//head_dim, seq_len)` 额外保证 `block_rows ≤ seq_len`（规避 `TRAP-ub-blockm-gt-M-overflow` 的 `block_m > M` 假性 UB overflow）。任务网格用 `ceildiv(seq_len, block_rows)` 保证覆盖最后一个不完整 tile。参考 `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 1。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子为 2D 输出网格，核切分落在 `seq_len` 轴（`head_dim` 维完整落在 tile 内，不切）：

```
num_logical_kernels = ceil(seq_len / block_rows)   # (M 轴 = seq_len, N 轴已整体入 tile)
```

| workload | dtype | seq_len × head_dim | block_rows | 逻辑核数（n_num） |
|----------|-------|--------------------|-----------|------------------|
| neox-1d-256-d64-smoke | fp32 | 256×64 | 16 | 16 |
| neox-1d-256-d64-smoke | fp16/bf16 | 256×64 | 32 | 8 |
| neox-1d-2k-d64-fp16 | fp16 | 2048×64 | 32 | 64 |
| neox-1d-4k-d128-bf16 | bf16 | 4096×128 | 16 | 256 |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-14 本会话实查；与 `CONST-aicore-910_9362` 一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**。

**③ 规模判定与分核方案**：分档处置（物理核数 40）：

- **smoke `[256,64]`（fp16/bf16 n_num=8；fp32 n_num=16）**：**逻辑核数 ≤ 物理核数** → 结论「无需适配」（8/40 = 20%、16/40 = 40% 核占用；小 shape 的核启动开销可忽略）。仍统一走网格跨步形态（`iters=1`），不特化分支。
- **`[2048,64]` fp16（n_num=64）**：中等规模（64 = 1.6×40 > 40）→ **极大规模/无法对齐处置**：固定启动内核数 = 物理核数 40，核内 `T.serial` 网格跨步分摊（见下）。注：`seq_len=2ⁿ`、`block_rows=2^k` → `n_num=2^{n-k}` 恒为 2 的幂，物理核数 40 = 2³×5 含非 2 因子 5，**数学上无法**通过调整 `block_rows`（2 的幂）使 `n_num` 对齐 40 的整数倍（5 ∤ 2ᵐ）→ 属「无法对齐兜底」，取最接近的 40 倍数并显式记录残留负载不均（见下）。
- **`[4096,128]` bf16（n_num=256）**：极大规模（256 = 6.4×40）→ 同上。

**统一分核方案（网格跨步 persistent，`core_num=40`）**：

```python
for i in T.serial(T.ceildiv(n_num, core_num)):   # 循环上界静态（n_num、core_num 均编译期常量）
    task = i * core_num + cid                    # 网格跨步（grid-stride）
    if task < n_num:                             # 守卫
        r0 = task * block_rows
        tr = T.min(block_rows, seq_len - r0)
        ...
```

- **循环边界静态** ✓（`n_num`、`core_num` 均编译期常量，`T.ceildiv(n_num, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：

| workload | n_num | iters = ceil(n_num/40) | 分配（40 核） | 临界路径惩罚 |
|----------|-------|------------------------|--------------|-------------|
| smoke fp32 | 16 | 1 | 16 核 1 次 / 24 核 0 次 | 核占用 40%（无核内不均，空转） |
| smoke fp16/bf16 | 8 | 1 | 8 核 1 次 / 32 核 0 次 | 核占用 20%（空转） |
| `[2048,64]` fp16 | 64 | 2 | 64 = 40×1 + 24 → 24 核 2 次 / 16 核 1 次 | 2/1.6 = **+25.0%** |
| `[4096,128]` bf16 | 256 | 7 | 256 = 40×6 + 16 → 16 核 7 次 / 24 核 6 次 | 7/6.4 = **+9.4%** |

- **相较直接 `T.Kernel(n_num)`**：直接超发会由运行时串行调度多波次、每次核启动开销重复；核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正（`core-split-strategy.md` §1 要素③）。
- **模式先例**：`examples/elementwise/vec_add_2d.py`（`T.Kernel(BLOCK_SIZE=20, is_npu=True)` + `T.serial` + guard）；`PL-1.22-persistent-1d-grid-stride`（**stale**，模板形态作背景，核数取值以实查为准）。

> 备选（供 Stage 4 参考，不在本设计实现）：因工厂 `lru_cache` 形状特化，可按 shape 分派不同 `block_rows`（如 smoke 用小 `block_rows` 提升核占用、大 shape 用大 `block_rows` 摊薄发射），以整组 workload 最优；按 `core-split-strategy.md` §2.4 属分核参数调优。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素/配对计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(40, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, 40))` | 每核串行处理 `ceildiv(n_num,40)` 个行块，摊薄核启动开销（§5.5） |
| 元素级 | 向量化（2D `(br,half)` 整 buffer） | `T.vmul`/`T.vadd`/`T.vsub` + `T.vcast` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：本设计以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`vec_add_2d.py`/`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(40, is_npu=True)` + `for i in T.serial(iters)` + `if task < n_num`，循环内为 6-op（fp32）或 12-op（sub-fp32）半段向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in ×4 → [vcast↑ ×4] → 6-op 链 → [vcast↓ ×2] → copy-out ×2），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。设计阶段保持简洁；若 Stage 4 需进一步优化（persistent double buffer / 更大 `block_size`），参照同族 `T.serial` 多 tile 形态。

### 6.4 尾块处理

见 §5.4。`seq_len % block_rows != 0` 时最后一个 tile 用 `tr = T.min(block_rows, seq_len - r0)` 动态裁剪 `T.copy` 的行 extent；manifest 1d workload 均整除（无尾块），尾守卫面向通用 `seq_len` 保留。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB: a,b,c,s) → [vcast↑ ×4] → vmul×2 → vsub → vmul×2 → vadd → [vcast↓ ×2] → T.copy(UB→GM: y 两半)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 6-op 链内部（同 buffer 原地复用） | 自动 | 线性数据依赖（`t` 复用、`y1f/y2f` 原地），插 V→V barrier |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| tile 迭代之间（`T.serial`） | 自动 | 同 buffer 复用由 auto-multi-buffer 管理跨迭代依赖 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作，无 CV 融合）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即 Developer
```

### 7.4 CV 融合设计

**n/a**：本算子是纯 Vector 逐元素/配对算子，无 Cube 段（无 `T.gemm`）、无跨引擎协作、无核间 workspace，不涉及 CV 融合。据 AGENTS.md「Developer-mode MixCV trigger rule」——kernel 内既无 Cube 侧 `T.gemm` 亦无跨核流水，**非 MixCV**，不触发 mixcv skill。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓参考实现 `tests/ops/test_rope.py::ref_rope_neox`，fp32 opmath + 单次舍回），**不得复刻 §0.6 的 NPU 半段向量链结构**——保证验证独立性。

```python
import torch


def golden_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE 参考实现（= 源仓 tests/ops/test_rope.py::ref_rope_neox，契约 golden）。

    语义（test 基准，保持 fp32 opmath + 末端单次舍回）：
        cos_full = cat([cos, cos], -1); sin_full = cat([sin, sin], -1)
        rot = cat([-x[..., half:], x[..., :half]], -1)
        y = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)
    独立于 NPU 结构（直接按定义式实现，不复用 §3.3 的半段 op 序列）。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return (x.float() * cos_full.float() + rot.float() * sin_full.float()).to(x.dtype)
```

> 备注：golden **独立于** NPU 向量链（按 §0.1 定义式直接实现，不复用半段结构），满足验证独立性。cos/sin 由 Op 层生成（`tileops/ops/rope.py::_base_freqs`，fp32 计算后 `.to(dtype)`），与 kernel 输入同 dtype。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`tests/ops/test_rope.py::_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对**：`verify_equiv.py` 实测半段 fp32 链 vs 契约 golden **三 dtype × {randn, corners} `max_abs=0.0`（逐位一致，ulp=0）**，远优于门禁；输入 dtype 域诊断（被拒 E1）fp16 `max_abs=7.81e-3 > 1e-3`、bf16 `6.25e-2 > 1.6e-2`（佐证 fp32 中转必要性）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含 1d manifest 全 workload、各 dtype、角点值、非整除/极小 `seq_len` 尾块**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn） | `x=randn([256,64], fp32)`，cos/sin `(256,32)` | 与 `golden_rope_neox` 比对 ≤1e-5 | 门槛通过 |
| L0-2 fp16 smoke（randn） | `x=randn([256,64], fp16)`，cos/sin `(256,32)` | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn） | `x=randn([256,64], bf16)`，cos/sin `(256,32)` | 比对 ≤1.6e-2 | 门槛通过 |
| L0-4 fp16 `[2048,64]` | `x=randn([2048,64], fp16)` | 比对 ≤1e-3 | 门槛通过 |
| L0-5 bf16 `[4096,128]` | `x=randn([4096,128], bf16)` | 比对 ≤1.6e-2 | 门槛通过 |
| L0-6 角点值（±inf/NaN/±0/极值） | `x` 与 cos/sin 填充 `[0,-0,inf,-inf,nan,±1e-45,±65504,±88.8,±1e-8]`（各 dtype） | 与 golden 比对（`equal_nan=True`；±inf 经 IEEE mul/add 传播） | 门槛通过 |
| L0-7 非整除尾块 | `x=randn([2050,64], fp16)`（2050 % 32 = 2，尾块 2 行） | 比对 ≤1e-3 | 门槛通过 |
| L0-8 极小 seq_len（尾块上界） | `x=randn([1,64], fp16)`（`block_rows=min(32,1)=1`） | 比对 ≤1e-3 | 门槛通过 |
| L0-9 通用 `seq_len` 非 2 幂 | `x=randn([1000,128], bf16)`（`block_rows=16`，1000=62×16+8 尾块 8 行） | 比对 ≤1.6e-2 | 门槛通过 |
| L0-10 cos/sin 复用等价 | L0-2 输出 vs「逐元素重取 cos/sin」的手工参考 | 逐位一致 | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.testing.assert_close(y.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，30/30 `EQUIV_PASS`，§1.6.1）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 sub-fp32 v-prefix 数学 API 缺失 + 计算域异域** | 输入 dtype 域对 fp16/bf16 超容差；bf16 按 doc 无 `vmul/vadd/vsub` | 全 sub-fp32 统一 fp32 中转（§0.6 R2 / §1.6.1 #2，API 强制 + 契约对齐） |
| **R-2 半段结构 = 结构重排（非直译）** | 若误按源逐元素 gather 直译，NPU 上退化为标量/低效 | §0.6 R2 半段结构（零 gather）；`verify_equiv.py` #1 证逐位等价 |
| **R-3 bf16 v-op 文档/示例分歧** | `docs/.../T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 记 bf16 ✗；但 `testing/npuir/bf16_support_ops/test_vv_bf16.py` 用 bf16 `T.vmul/T.vadd` | 本算子**不依赖**原生 bf16 路径（统一 fp32 中转），故不阻塞；分歧列入待确认，Stage 3 设备探针核对；**文档为准**（AGENTS.md） |
| **R-4 cos/sin 表 dtype 与 golden 一致性** | Op 层生成 cos/sin 为 fp32 后 `.to(dtype)`；kernel 再 `vcast` 升 fp32 | 与 golden 路径一致（golden 亦 `cos.float()`）；`verify_equiv.py` #2 逐位一致 |
| **R-5 发射项（0.5µs/op）名义主导** | §1.6.0 估算示发射项名义主导；但吞吐口径（`CONST-vector-op-cost-910_9362`）显示流量主导 | 标注「未文档化假设 + 估算依据」（常数 stale、可能上偏）；`block_size` 为杠杆（§5.2）；Stage 4 baseline 后对账 |
| **R-6 UB 容量 / `bs` 放大** | `bs=8192` sub-fp32 超限（×1.7 ≈ 245 KB） | 默认 K11 折叠值（最大 36 KB×1.7 ≈ 61 KB）；`bs` 放大须编译探针校准（§4.5） |
| **R-7 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3）；`af~sf` 在 fp32 路径仍分配 |
| **R-8 v-op 整 buffer vs 动态 slice** | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tr, 0:half]`（§3.3） |
| **R-9 `vcast` round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32`/`bf16→f32` **仅 `rint`** | 升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-10 `block_rows > seq_len` 假性 UB overflow**（`TRAP-ub-blockm-gt-M-overflow`） | 极小 `seq_len`（如 1）时 `block_rows` 过大 | `block_rows = min(block_size//head_dim, seq_len)`（编译期）；+`tr` 行尾守卫（§5.4） |
| **R-11 分核残留不均** | 物理核 40 = 2³×5（含 5），`seq_len=2ⁿ`、`block_rows=2^k` → `n_num=2ᵐ` 无法整除 40 | §5.5 网格跨步核内串行，残留 +9.4%~+25.0%（大 shape）；smoke 核占用 20–40%（空转，非不均），均显式记录 |
| **R-12 常数表设备/版本（stale）** | 引用的 `CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-mte2-degradation`/`PL-1.6`/`PL-1.22` 等被 `kb_stale_check` 标 `[stale]`（绑 910B2C 或旧 toolchain）；本设备 910_9362 | 性能数字标注「待重验/背景」，不作决策依据；流量项优先用 `CONST-ascend910-9362-profile`（verified，但亦 stale）；Stage 4 baseline 后对账回填 |
| **R-13 静态 2D 子切片 v-op（备选 L3）** | `TRAP-expert-v-operands` 仅证 runtime Var 被拒，静态边界理论合法但无本算子形态实测 | 主选 L1 拆半装载**不用**切片作 v-op 操作数；L3 交 Stage 4 实验裁决（§1.6.3） |
| **R-14 尾块陈旧数据** | 尾块时整 buffer 含陈旧行（可能 NaN/inf）参与 v-op | 陈旧行结果不写出（仅 `[0:tr, 0:half]` 拷出）；`PL-1.22` 尾块零起点切片纪律 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接传入数学 op | 未做 vcast 上转 | 按 doc 编译失败（R-3 分歧待核） | sub-fp32 先 `vcast` 到 fp32（R-1） |
| vcast 用非 rint 模式升级 | `f16→f32` 用 `round`/`floor` | 编译失败（仅 `rint`） | 升降统一 `round_mode="rint"`（R-9） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-7） |
| v-op 传动态 slice | `T.vmul(a_ub[0:tr], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-8） |
| `block_rows > seq_len` | 极小 seq_len | 假性 UB overflow（`TRAP-ub-blockm-gt-M-overflow`） | `block_rows=min(...,seq_len)`（R-10） |
| UB 溢出 | `block_size` 过大 | 编译失败 | 维持 K11 折叠值（默认）；§4.5 预算 + 编译探针（R-6） |
| 尾块越界 | `seq_len` 非 `block_rows` 整数倍且无 `tr` 裁剪 | 越界读/写 GM | `tr = T.min(block_rows, seq_len-r0)`（§5.4） |
| 行尾切片 v-op | 对 `[0:tr]` 行切片调 v-op | 编译/语义错误 | 行切片只在 `T.copy`（R-8） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 `seq_len`** | Kernel 内 `tr` 行尾动态裁剪（§5.4），无需 host padding |
| **极小 `seq_len`（< block_rows）** | `block_rows = min(..., seq_len)`（编译期），`n_num=1`，仅 `task==0` 的核工作 |
| **极大 `seq_len`** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf` / `NaN` 输入** | 经 4 mul + 1 add/sub IEEE 传播，与 golden 运算集合一致（`inf*0=NaN` 一致）；`equal_nan=True` |
| **`x=0` / `-0.0`** | `y1 = −b*s`、`y2 = a*s`，恒等；`±0` 符号经乘法保留 |
| **`head_dim` 奇数** | manifest `shape_rules: x.shape[-1] % 2 == 0` 排除；工厂 `assert` 前置 |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`block_size` 非 `head_dim` 整数倍** | 工厂 `assert block_size % head_dim == 0`（K11 折叠值均满足） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/rope_neox/_make_rope_neox_1d/
├── DESIGN.md            # 本文件（Stage 1 产出）
├── verify_equiv.py      # D-1 等价性机器验证（30/30 EQUIV_PASS）
├── RETROSPECTIVE.md     # Stage 1 复盘（skill Phase 8）
└── _make_rope_neox_1d.py  # NPU kernel（下游 Stage 3 产出，尚未创建）
```

### 10.2 文件清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/rope_neox/_make_rope_neox_1d/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/rope_neox/_make_rope_neox_1d/verify_equiv.py` | D-1：30/30 `EQUIV_PASS`（§1.6.1） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_rope_neox_1d.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_rope_neox`） | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py` | `block_size`（K11）+ `main(x,cos,sin,y)` 契约 + `ascend_mode="Developer"` |
| 抽取脚手架（已存在，只读参照） | `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py` | Pattern A 源直译参照 |
| 源算子 / 测试（只读参照，源真相） | `/home/developer/TileOPs/tileops/kernels/rope.py`、`/home/developer/TileOPs/tests/ops/test_rope.py` | 语义与 golden 依据 |
| manifest（只读参照） | `examples/TileOPs/tileops/manifest/position_encoding.yaml` | `RopeNeoxOp` 1d workloads + roofline |

### 10.3 命名规范

- 工厂：`_make_rope_neox_1d(seq_len, head_dim, dtype, block_size=2048)`（对齐 wrapper）。
- kernel 内 buffer：`a_ub/b_ub/c_ub/s_ub`（输入两半 + cos/sin）、`af/bf/cf/sf/y1f/y2f/t`（fp32 计算域）。

### 10.4 实现顺序

1. 实现 kernel（§3.3）：2D UB tile 半段向量链 + 一维核网格跨步 + `tr` 行尾守卫；
2. 写 `golden_rope_neox`（§8.1）；
3. 跑 L0-1..L0-10（§8.3）；
4. `verify_equiv.py` 复核（§1.6.1）。

**实现要点速览（Stage 3 输入）**：
1. **计算核 = 半段 2D 向量链**：`y1 = vsub(vmul(a,c), vmul(b,s))`、`y2 = vadd(vmul(b,c), vmul(a,s))`（6 op，作用于 `(block_rows, half)` 整 buffer，**零 gather**）；
2. **计算域 = fp32 中转**（sub-fp32 先 `vcast(rint)` 升 fp32，链尾 `vcast(rint)` 舍回；fp32 直算）；**bf16 必走 fp32**；
3. **cos/sin 单次装载 + 两半复用**（CSE，cos/sin 流量 1×）；
4. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tr` 行尾守卫；`block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048），`block_rows = block_size // head_dim`；
5. buffer 顶层无条件分配；v-op 作用于整 buffer（行切片只在 `T.copy`）；`a_ub/b_ub` 死亡后复用为输出；
6. `main(x, cos_table, sin_table, y)`（`out_idx=[3]`，**无额外 shape 参数**）。

---

## 性能目标（Stage 4 / optimize 场景；由 conductor 追加）

- 性能目标类型：`best_effort`（用户未指定目标类型 / 数值 / baseline）
- **共享 kernel 采纳**：本 op 的 `_make_rope_neox_{1d,2d}` 与 `rope_neox` **逐字节相同**；调优在规范位 `rope_neox/rope_neox_kernel/perf_opt/` 一次性完成（覆盖 `RopeNeoxOp` / `RopeLongRopeOp` / `RopeLlama31Op` 三个 bench 入口的并集 workload），本目录 `perf_opt/` 为其**逐字节采纳副本**（含 `perf_records.jsonl` + `perf_records_2d.jsonl` + `opt_log.md`，标注 `shared_measurement`）。
- 回归入口：`python {kernel_dir}/perf_opt/_make_rope_neox_{1d,2d}.py --level all`；采纳后从 `examples/TileOPs/` 运行 `python -m tileops.reporting.cli run --op RopeLongRopeOp --prof-mode msprof`
- 关注 shape：未指定（并集 workload 全部参与调优）



