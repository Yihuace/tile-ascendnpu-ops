# _make_rope_neox_2d 算子设计文档（迁移）

> **项目**：`rope_neox` ｜ **算子目录**：`examples/rope_neox/_make_rope_neox_2d/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness，同任务第 2 个函数）。把 GPU TileLang 工厂函数 `_make_rope_neox_2d`（GPT-NeoX 风格 RoPE，layout=`2d`，`x` 形状 `(batch, seq_len, num_heads, head_dim)`，cos/sin `(seq_len, head_dim//2)` 在 batch/head 上广播）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策**：① **计算核重设计 = 半段旋转（half-split）**（承接同任务 1d 函数）：把源码「逐元素 flat 索引 + 三级 div/mod 坐标还原 + `col % half` + `if_then_else` 配对 gather」改为「**3D 广播 tile `(block_tokens, num_heads, half)` + 半段向量链** `y1=a*c−b*s`、`y2=b*c+a*s`（6 向量 op，零 gather、零逐元素坐标）。② **2d 专有 = 头轴广播重设计**：cos/sin 装载为 `(block_tokens, 1, half_pad)`，靠 NPU v-op 的**同 rank size-1 轴自动广播**沿 `num_heads` 轴展开（`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py` 实证形态），源头/batch 广播退化为「tile 内共享 + 每 tile 单次装载」。③ **坐标重设计**：flat 三级 `%`,`//` → `(tokens, num_heads, head_dim)`（`tokens=batch*seq_len`）行主视图 + per-tile `(b, seq_block)` 映射，cos/sin 行索引 = `seq_start + i`（无逐元素 div/mod）。④ **计算域 = fp32 中转 + 单次舍回**（契约 golden 为 fp32 opmath，见 §0.1）。
>
> **源优化意图承接**（§0.4 逐项 → §0.5/§0.6）：源 #1 `T.Parallel(threads,npt)` → R1 一维核网格跨步；源 #2 flat 三级 `%`,`//` 坐标 → R1 静态 3D tile 视图；源 #3 逐元素 `if_then_else` 配对 gather → R2 半段结构（零 gather）；源 #4 cos/sin 逐元素取表 + batch/head 广播 → R3 单次装载 + size-1 轴广播（**优化意图反向承接**：源靠 L2 吸收的每元素重读在 UB 显式降为每 tile 一次）；源 #5 `out_idx=[3]`/源 #6 `lru_cache` → 保留；源 #7 `threads*npt` 对齐 → 折叠为 `block_size`。
>
> **设计期实证（D-3 探针 + D-1 机器验证）**：① `history_version/design_probe_npu_2d.py` 在设备上跑通「flat `(N_total,)` 输入 + 3D 声明 kernel」契约、3D size-1 轴广播链、`half_pad`、行尾守卫，5 个 shape（含 manifest `[2,2048,32,128]`）**逐位一致 `max_abs=0.0`**（fp16/bf16/fp32）；② `verify_equiv.py` **48/48 `EQUIV_PASS`**（§1.6.1）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/rope.py`

- 具体算子：工厂 `_make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype, threads=256, num_per_thread=8)`（**L92–L138**，`@functools.lru_cache(maxsize=32)` + `@tilelang.jit(out_idx=[3])`），内部 `@T.prim_func main(x, cos_table, sin_table, y)`：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype, threads=256, num_per_thread=8):
      half = head_dim // 2
      N_total = batch * seq_len * num_heads * head_dim
      block_size = threads * num_per_thread
      stride_b = seq_len * num_heads * head_dim
      stride_s = num_heads * head_dim

      @tilelang.jit(out_idx=[3])
      def kernel(threads_arg, npt_arg):
          @T.prim_func
          def main(x: T.Tensor((N_total,), dtype),
                   cos_table: T.Tensor((seq_len, half), dtype),
                   sin_table: T.Tensor((seq_len, half), dtype),
                   y: T.Tensor((N_total,), dtype)):
              with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                  for i, j in T.Parallel(threads_arg, npt_arg):
                      flat_idx = (bx * threads_arg + i) * npt_arg + j
                      rem = flat_idx % stride_b            # -> batch 内偏移
                      s_idx = rem // stride_s              # -> seq 位置
                      rem2 = rem % stride_s
                      col = rem2 % head_dim                # -> head_dim 内列
                      freq_idx = col % half
                      c = cos_table[s_idx, freq_idx]        # cos/sin 按 seq 取，batch/head 广播
                      s = sin_table[s_idx, freq_idx]
                      val = x[flat_idx]
                      paired_col = T.if_then_else(col < half, col + half, col - half)
                      head_base = flat_idx - col
                      paired_idx = head_base + paired_col   # rotate_half 配对（同 head 内）
                      paired_val = x[paired_idx]
                      rotated = T.if_then_else(col < half, -paired_val, paired_val)
                      y[flat_idx] = val * c + rotated * s
          return main
      return kernel
  ```

- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py`（Pattern A，`_make_rope_neox_2d` L65–L110，逐字复制源）。
- 集成 wrapper（**Stage 5 集成目标，本迁移必须对齐其接口**）：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py`（`RopeNeoxKernel`）。
  - `_build_kernel`（L171–L177）：`layout != "1d"` → `_make_rope_neox_2d(self.batch, self.seq_len, self.num_heads, self.head_dim, dtype_str)`（K3/K11：工厂去 `threads`/`num_per_thread`，**保留 `batch/seq_len/num_heads/head_dim` 4 个位置参数 + `dtype`**）。
  - `default_config`（L86–L90/L179–L186）：`block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}`（K11 折叠 `threads(256) × num_per_thread`）。
  - `forward`（L188–L216）：`layout=="2d"` 时 `x_flat = x.contiguous().reshape(-1)`，`result = self._compiled_fn(x_flat, cos, sin)`，`return result.reshape(x.shape)` —— **kernel 入参 `x` 为 flat `(N_total,)`**，cos/sin 为 `(seq_len, half)`。
  - `ascend_mode = "Developer"`（L128）。

### 0.1 源算子语义（做什么）

**数学语义**（NeoX / GPT-NeoX 风格 RoPE，`layout="2d"`；参考 GPT-NeoX / HuggingFace `RotaryEmbedding`）：`x ∈ (batch, seq_len, num_heads, head_dim)`，`cos_table/sin_table ∈ (seq_len, half)`（`half = head_dim // 2`），**在 batch 与 head 两轴上广播**：

$$
y[b,s,h,col] = x[b,s,h,col]\cdot c[s,\ col\bmod half] + \operatorname{rot}(x)[b,s,h,col]\cdot s[s,\ col\bmod half]
$$

$$
\operatorname{rot}(x)[b,s,h,col]=
\begin{cases}
-x[b,s,h,\ col+half], & col < half\\[2pt]
+x[b,s,h,\ col-half], & col \ge half
\end{cases}
$$

**等价闭式（半段 split，本设计计算核的语义依据）**：对每个 head 向量在 `half` 处拆成两半 `a = x[..., :half]`、`b = x[..., half:]`，则对 `j ∈ [0, half)`：

$$
y[b,s,h,\ j] = a_j\,c[s,j] - b_j\,s[s,j],\qquad y[b,s,h,\ half+j] = b_j\,c[s,j] + a_j\,s[s,j] .
$$

推导：`col=j<half` → `freq_idx=j`、`val=a_j`、`rot=−b_j`（`paired_col=col+half`）；`col=half+j` → `freq_idx=(half+j)%half=j`、`val=b_j`、`rot=+a_j`（`paired_col=col-half`）。与 §0 公式逐项一致，**数学恒等**；关键点：配对在**同一 head 内**（`head_base = flat_idx - col`），**不跨 head / 不跨 seq / 不跨 batch**，故 head 轴可整体并行、逐 head 独立。

**契约语义 = 源仓测试基准**（`tests/ops/test_rope.py::ref_rope_neox`，4D 分支）：

```python
cos_full = torch.cat([cos, cos], -1); sin_full = torch.cat([sin, sin], -1)
cos_full = cos_full.unsqueeze(0).unsqueeze(2)     # (1,S,1,d) —— batch/head 广播
sin_full = sin_full.unsqueeze(0).unsqueeze(2)
rot = torch.cat([-x[..., half:], x[..., :half]], -1)   # rotate_half
y = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)
```

即 **fp32 opmath（`x.float()`/`cos.float()`）+ 末端一次舍回原 dtype**。容差（`_get_tolerances`）：fp32 `1e-5` / fp16 `1e-3` / bf16 `1.6e-2`（atol=rtol）。

**计算域语义（迁移必须保持的关键点）**：

| 域 | 表示 | 与契约 golden 的一致性 |
|----|------|------------------------|
| **契约 golden**（`ref_rope_neox` 4D） | `x.float()*cos_full.float() + rot.float()*sin_full.float()` → `.to(dtype)` | 基线：fp32 域两乘 + 一加，末端单次舍回 |
| 源 GPU kernel | `val*c + rotated*s`，`c,s,val,rotated` 均**输入 dtype**（无显式 fp32 提升） | **与契约存在域差**：源码在输入 dtype 域算；golden 在 fp32 域算后单次舍回——**异域（需等价替换为 fp32 中转）** |

→ §1.6.1 #2 采纳 **fp32 中转计算域**（`vcast` 升 fp32 → 向量链 → `vcast` 舍回），`verify_equiv.py` 诊断证明输入 dtype 域非逐位等价（fp16 `max_abs=3.9e-3`、atol 违反比 ~1.2%），而 fp32 中转与 golden **逐位一致（`max_abs=0.0`）**。

**dtype 语义**：`x`/`cos_table`/`sin_table`/`y` 同 dtype（fp16/bf16/fp32）。cos/sin 由 **Op 层（host）** 以 fp32 计算后 `.to(dtype)`（`tileops/ops/position_encoding/rope_neox.py::_base_freqs`），进入 kernel 时**已是目标 dtype**；本设计 `vcast` 升 fp32 后计算，输出单次舍回目标 dtype。

**规约语义**：**无规约**——逐元素旋转，`(...,j)` 与 `(...,half+j)` 构成两两配对的 2×2 旋转，元素间无累加、无归约轴，**不存在累加顺序问题**。

**边界语义**：无分支特例；`±inf`/`NaN` 经 IEEE mul/add 传播（`inf*0=NaN` 与 golden 一致，§1.6.1 机器验证含角点）；`x=0` → 两半分别为 `−b*s` / `a*s`；`seq_len` 非整除 tile 由行尾处理（§5.4）。manifest `shape_rules`: `layout != '2d' or x.ndim == 4`、`x.shape[-1] % 2 == 0`。

**host 侧语义**：2d layout 下 `forward` 做 `x.contiguous().reshape(-1)` 把 4D 展平为 flat `(N_total,)` 交 kernel，再把输出 `reshape` 回 `(batch,seq_len,num_heads,head_dim)`——**输入张量真实内容不变，仅元数据视图（合规）**；cos/sin 由 Op 层按 `(seq_len, half)` 生成并缓存（`_freq_cache`）。

**语义保持基线**：§8.1 golden 以本节语义（`ref_rope_neox` 4D，fp32 opmath + 单次舍回）为唯一依据，**不得复刻 §0.6 的 NPU 半段 3D 广播链结构**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（Op 层 4D；kernel 入参 = flat） | `(batch, seq_len, num_heads, head_dim)` → flat `(N_total,)` | `float16 \| bfloat16 \| float32` | manifest 2d workload：`[2, 2048, 32, 128]` fp16 |
| `cos_table` | `(seq_len, head_dim // 2)` | 同 `x` | 余弦频率表（Op 层生成）；在 batch/head 上广播 |
| `sin_table` | `(seq_len, head_dim // 2)` | 同 `x` | 正弦频率表 |
| `y`（输出） | `(batch, seq_len, num_heads, head_dim)`（kernel `out_idx=[3]` 按 flat `(N_total,)` 分配） | `same_as(x)` | 旋转结果；wrapper `reshape` 回 4D |

**输出 shape（迁移必需）**：`source_output_shape = (batch, seq_len, num_heads, head_dim)` = `same_as(input)`。**layout**：行主连续（row-major contiguous），**无转置**。**动态轴**：无运行时动态轴——`batch/seq_len/num_heads/head_dim` 经 `lru_cache` 编译期特化（wrapper 逐 shape 构造）；尾块守卫用编译期常量。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源，`T.Parallel(threads_arg, npt_arg)` 内逐元素）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `flat_idx = (bx*threads + i)*npt + j` | block 号 `bx`、线程 `i`、元素 `j` | 全局 flat 下标 | 元素定位 |
| 2 | `rem = flat_idx % stride_b`；`s_idx = rem // stride_s`；`rem2 = rem % stride_s`；`col = rem2 % head_dim` | `flat_idx`、编译期 `stride_b/stride_s` | `s_idx`,`col`（`h = (rem2//head_dim)`） | **2d 三级坐标还原（batch/seq/head/col）** |
| 3 | `freq_idx = col % half`；`c=cos_table[s_idx,freq_idx]`；`s=sin_table[s_idx,freq_idx]` | `cos_table`,`sin_table` | 标量 `c`,`s` | 频率表取值（**按 seq 取，batch/head 广播**） |
| 4 | `val = x[flat_idx]`；`paired_col = col<half ? col+half : col-half`；`head_base = flat_idx-col`；`paired_val = x[head_base+paired_col]`；`rotated = col<half ? -paired_val : paired_val` | `x` | 标量 `val`,`rotated` | `rotate_half` 配对（**同 head 内**） |
| 5 | `y[flat_idx] = val*c + rotated*s` | `val`,`c`,`rotated`,`s` | `y` | 2×2 旋转 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x(N_total,) flat] --(每线程 npt 个连续元素, coalesced load)--> registers
GM[cos_table/sin_table] --(逐元素按 (s_idx, freq_idx) 取, batch/head 广播值)--> registers
   --(T.Parallel(threads,npt) 逐线程标量: 三级 %//坐标; col%half; if_then_else 配对; val*c+rot*s)--> registers
   --(coalesced store)--> GM[y(N_total,) flat]
（无显式 SMEM；无 fragment 缓冲——纯寄存器逐元素；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N_total, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 个**连续** flat 元素。`block_size = threads×npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048（wrapper K11 折叠）。`N_total = batch*seq_len*num_heads*head_dim`。

**host 侧逻辑**：`N_total`/`half`/`stride_b`/`stride_s`/`block_size` 由工厂 Python 侧计算（编译期）；`out_idx=[3]` 由 jit 自动分配输出；flatten（`x.contiguous().reshape(-1)`）与 cos/sin 生成在 Op 层（§0.1）。**无其它 host 侧计算。**

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `T.Parallel(threads, npt)` block×thread 两级并行 + 每线程 `npt` 连续元素 | 吃满访存带宽（合并访存） | 256 线程/block × grid(`ceildiv(N_total, bs)`) | GPU thread/warp 执行模型 + memory coalescing | **硬件强相关**（GPU 线程模型） |
| 2 | flat 三级寻址 `% stride_b`、`// stride_s`、`% head_dim`（编译期常量除/模） | flat grid 均匀切分；免二维/三维寻址开销 | 常量除/模由编译器优化为移位/乘 | GPU 标量 ALU | **实现层**（NPU 用静态 3D tile 视图替代） |
| 3 | 逐元素 `if_then_else` 配对（`rotate_half` 免物化） | 免去构造 rotated 张量（省一次全量读写） | 逐元素条件选择配对的另一元素（同 head 内 ±half） | GPU 标量 select（一次性、无额外访存） | **算法/实现混合**（NPU 无廉价逐元素 gather → 需重设计） |
| 4 | cos/sin 逐元素取表 + **batch/head 隐式广播**（`s_idx` 只依赖 flat 的 seq 分量；同一 `(s_idx,freq_idx)` 被 batch×head 个元素共享） | 免去物化 `(B,S,H,half)` 全宽 cos/sin 表 | 索引映射使 batch/head 复用同一 `cos_table[s_idx,·]` 行；PTX 层每元素各读一次（L2 吸收重复） | GPU L2 缓存 | **实现层**（NPU 上 L2 不可依赖 → UB 显式单次装载 + size-1 轴广播） |
| 5 | `@tilelang.jit(out_idx=[3])` | 自动分配输出张量 | jit 参数约定 | 无硬件依赖 | 可移植（NPU 保留） |
| 6 | `functools.lru_cache(maxsize=32)` | 按 `(batch, seq_len, num_heads, head_dim, dtype)` 编译期特化 | 常量固化（`stride_b/stride_s/N_total/half/block_size`） | 无硬件依赖 | 可移植（NPU 保留） |
| 7 | `threads=256`，`npt=4`(fp32)/`8`(fp16·bf16) | 每线程访问 128-bit 对齐 | 启发式（`default_config`） | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`，§5） |

> 识别不出来 ≠ 不存在。上述 7 项为源码中可识别的全部优化/规避手段；后续逐项处置（§0.5）。**未识别优化会被静默丢弃**，故此处列全。与 1d 函数相比**新增 #2 的三级坐标与 #4 的 batch/head 广播**——二者是本函数区别于 1d 的迁移关键点。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（`y=x*c+rot*s`；I/O 契约；batch/head 广播；fp32 opmath 契约；无规约；配对不跨 head/seq/batch） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| 算法结构（`head_dim` 拆半 + 每 head 内两两 2×2 旋转） | 算法 | 无（纯数学恒等式，且配对不跨 head） | — | **保留（结构）** | §0.1 闭式；仅实现结构重设计 |
| 优化 #1 `T.Parallel(threads,npt)` 线程级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU `T.Kernel` 一维 block；`threads=` 无效，`TRAP-threads-kwarg-noop`） | **重新设计** | §0.6 R1：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #2 flat 三级 `%`,`//` 坐标还原 | 优化 | GPU 标量 ALU + 均匀 flat grid | 有（3D 静态 tile 视图 + per-tile 映射） | **重新设计** | §0.6 R1：`x` 声明为 `(tokens, num_heads, head_dim)` 行主视图，任务映射到 `(b, seq_block)`；cos/sin 行索引 = `seq_start + i`，**消除逐元素 div/mod** |
| 优化 #3 逐元素 `if_then_else` 配对 gather | 算法/优化 | GPU 廉价标量 select + 隐式 L2 | **无廉价逐元素 gather**（跨步 gather 向量指令缺失，`TRAP-C10-strided-gather`） | **重新设计** | §0.6 R2：半段（half-split）结构——UB 上 `y1=a*c−b*s`、`y2=b*c+a*s`，`a,b` 为同 tile 内连续子块，**零 gather** |
| 优化 #4 cos/sin 逐元素取表 + batch/head 广播（L2 吸收重复） | 优化 | GPU L2 缓存 | 有（UB tile + **同 rank size-1 轴自动广播**） | **等价替换（并优化）** | §0.6 R3：cos/sin 按 tile 装载 `(block_tokens, 1, half_pad)`，沿 `num_heads` size-1 轴广播；**单次装载、全 head 复用**（源每元素重读 → 每 tile 一次） |
| 优化 #5 `out_idx=[3]` | 优化 | 无 | 有 | **保留** | NPU 保留 `out_idx=[3]`（对齐 wrapper 契约） |
| 优化 #6 `lru_cache` 编译期特化 | 优化 | 无 | 有 | **保留** | 保留 `(batch, seq_len, num_heads, head_dim, dtype)` 特化（+`block_size`） |
| 优化 #7 `threads`/`npt` 对齐 | 优化 | memory coalescing | 有（32B 尾轴对齐 / 128-bit 向量宽度） | **等价替换** | K11：折叠为 `block_size`（§5.2）；32B 对齐按 NPU 复算（§5.3） |
| 计算域 = 输入 dtype 域（源无显式提升） | 语义/优化 | 无 | 有（`vcast` 支持 f16/bf16↔f32，`rint`） | **等价替换** | §0.6 R4 / §1.6.1 #2：改为 **fp32 中转**（契约 golden 为 fp32 opmath；输入 dtype 域非逐位等价） |
| layout=1d 函数 `_make_rope_neox_1d` / THD / non_neox / llama31 / yarn / longrope 变体 | 算法 | GPU | — | **舍弃（范围外）** | 本迁移只 port `_make_rope_neox_2d`（同任务第 1 个函数 `_make_rope_neox_1d` 已独立完成；其余变体不在 op_name 内） |

> **依据**：#1/#2 由 `TRAP-threads-kwarg-noop`（`src/ir.cc` NPU 分支不消费 `threads=`）与 `docs/开发指南.md §3.3`（一维 Kernel + 物理核数）佐证；#3 由 `TRAP-C10-strided-gather`（跨步 gather 向量指令缺失）佐证；**#4 由 `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.2（自动广播；同 rank；行/列广播合法）+ `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（3D `(bt,bh,hp)` 与 `(bt,1,hp)` 的 size-1 头轴广播实证）佐证**；计算域由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32`/`f32→f16/bf16` 均支持 `rint`）佐证。**注**：`PL-1.15-broadcast-perf-tax`（verified）记「未文档化的列广播/alias 形态有 +12~20% 性能税」——本设计所用 size-1 中间轴广播属同 class，**数值可编译可算对但性能税待实证**，见 §1.6.3 实验裁决与 §9 R-4。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：坐标与并行结构重设计（flat 三级 div/mod + thread 级 → 3D 视图 + 一维核网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N_total, block_size), threads=256)` + `T.Parallel(256, npt)`，对每个 flat 元素做三级 `%`,`//` 还原 `(s_idx, h, col)`；性能意图 = 大规模 flat grid + 合并访存吃满带宽。
- **NPU 新算法**：把 wrapper 传入的 flat `(N_total,)` **在 kernel 内声明为 3D 行主视图** `x: T.Tensor((tokens, num_heads, head_dim), dtype)`（`tokens = batch*seq_len`；行主布局与 flat 逐字节一致）；任务映射到 `(b, seq_block)`：`work_blocks = batch * ceil(seq_len/block_tokens)`，第 `w` 个 tile 取 `b = w // seq_blocks`、`sb = w % seq_blocks`、`seq_start = sb*block_tokens`、`rows = min(block_tokens, seq_len - seq_start)`、`token_start = b*seq_len + seq_start`。一维 `T.Kernel(40, is_npu=True)` + 核内 `for i in T.serial(ceildiv(work_blocks, 40))` 网格跨步（`task = i*40 + cid`，`if task < work_blocks` 守卫）。cos/sin 行索引 = `seq_start + (tile 内行号)`，**无逐元素 div/mod**。
- **语义保持论证**：任务映射只重排「哪个核处理哪些 (b, seq-block)」，各 head 向量旋转完全独立（配对不跨 head，§0.1），数值恒等；`cos_table[seq_start+i, :]` 与源 `cos_table[s_idx, :]`（`s_idx = t % seq_len`，`t = token_start+i`）**同值**（`s_idx = (b*seq_len + seq_start + i) % seq_len = seq_start + i`，因每 tile 不跨 batch）。`verify_equiv.py` #0 机器验证 flat 三级算术与 `(b,s,h,col)` 查表**逐位一致**（3 shape）。边界：`T.serial` 上界为编译期静态值；守卫覆盖尾块；行尾以 `rows = T.min(block_tokens, seq_len - seq_start)` 动态裁剪 `T.copy`（§5.4），不越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py`（一维核 + `T.serial` 网格跨步 + 2D tile + `T.copy`）；`PL-1.22-persistent-1d-grid-stride`（纯 Vector 一维 persistent 模板，两任务实证）；本体 1d 函数 `examples/rope_neox/_make_rope_neox_1d/`。

**重设计项 R2：计算核重设计（逐元素 select/gather → 半段 3D 向量链）**

- **源方案**：逐元素 `col<half ? -x[head_base+col+half] : x[head_base+col-half]`（`if_then_else`）+ `val*c + rotated*s`；GPU 上 select 是一步标量、配对访问由 L2 吸收。
- **NPU 新算法（整 UB buffer 向量链，半段结构 + head 轴广播）**：对每 tile，令
  - `xlo_ub = x[token_start:token_start+rows, 0:H, 0:half]`、`xhi_ub = x[token_start:token_start+rows, 0:H, half:d]`（均 `(block_tokens, H, half_pad)` 3D UB tile）；
  - `c_ub/s_ub = cos/sin[seq_start:seq_start+rows, 0:half]`（`(block_tokens, 1, half_pad)`，沿 H 轴 size-1 广播）；
  - fp32 中转后：`y1 = vsub(vmul(a,c), vmul(b,s))`、`y2 = vadd(vmul(b,c), vmul(a,s))`（4 `vmul` + 1 `vsub` + 1 `vadd`，作用于 `(block_tokens, H, half_pad)`；sub-fp32 另 +4 `vcast` 升 +2 舍回）。
- **语义保持论证**：
  1. **数学恒等**（§0.1 闭式推导）：`y1_j = a_j c_j − b_j s_j` 与源 `col=j` 的 `val·c + rotated·s = a_j c_j + (−b_j) s_j` 逐项相等；`y2_j = b_j c_j + a_j s_j` 与源 `col=half+j` 相等。**零近似**（`verify_equiv.py` #1 三 dtype × {randn, corners} `max_abs=0.0`）。
  2. **舍入路径与 golden 一致**：golden 为 fp32 域两乘后加、末端一次舍回；NPU 链 `vmul/vsub/vadd` 在 fp32 域（IEEE 下 `(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)` 精确）→ **逐位一致**（`verify_equiv.py` #2 三 dtype `max_abs=0.0`）。
  3. **边界语义**：`±inf`/`NaN` 经 4 mul + 2 add/sub 传播，与 golden 相同运算集合；角点用例（±inf/NaN/±0/次正规/±65504）机器验证 `max_abs=0.0`；非整除 `seq_len` 由 `rows` 行尾裁剪。
- **源优化意图承接**：源 #3「免物化 rotated 拷贝」→ 承接（半段结构无需物化 rot）；源 #2「flat 均匀切分」→ 承接于 `(b, seq_block)` 任务映射；源计算域「输入 dtype」→ **有意替换**为 fp32 中转（契约对齐，§1.6.1 #2）。

**重设计项 R3：cos/sin 广播与访存 CSE（逐元素取表 + L2 吸收 → UB 单次装载 + size-1 轴广播）**

- **源方案**：每元素（batch×head×col）读一次 `cos_table[s_idx, col%half]`；同一 `(s_idx,·)` 行被 `batch×num_heads` 个元素共享，重复读由 GPU L2 吸收。等价于数学上的 `(B,S,H,half)` 广播张量，源**不物化**。
- **NPU 新算法**：cos/sin 按 tile **单次装载**为 `(block_tokens, 1, half_pad)` 的 3D UB tile（`c_ub/s_ub`），在向量链中靠 **v-op 同 rank size-1 轴自动广播**（`(bt,H,hp) op (bt,1,hp)`）沿 head 轴展开；一次装载服务该 tile 的**全部 `num_heads` 个 head**。
- **语义保持论证**：广播不改变数值——`y1/y2` 中每个 head 使用的 `c_j/s_j` 与源该 head 各列的 `cos_table[s_idx,j]` 完全同值（源本就 batch/head 广播同一行）；数学恒等。收益：cos/sin 的 GM 流量由「每元素一次」降为「每序列位置每 batch **每 tile 一次**」（tile 内 `num_heads` 复用）；以 manifest 为例，理论最小 cos/sin 流量 = `seq_len*half*2`（跨 batch 共享），实际 = `batch×`（每 batch 各 tile 各装一次）≈ 2× 理论最小，但相对 x 流量（`2*x_elems`，即 `2*batch*seq*H*d`）占比 <1%（`cos_sin/(2*x_elems) = half/(2*H*d) = 64/8192 ≈ 0.78%`）。
- **承接说明（源优化意图不丢弃）**：§0.4 #4「隐式广播靠 L2」→ R3 显式 UB 单载 + 硬件广播承接；#1/#2 → R1；#3 → R2；#5/#6 保留；#7 → §5。**与 1d 的差别**：1d 无广播轴，cos/sin tile 是 `(block_rows, half)`；2d 多出 head 广播轴，tile 升为 3D 且 cos/sin 用 size-1 头轴。

**重设计项 R4：计算域（输入 dtype 域 → fp32 中转 + 单次舍回）**

- **源方案**：`val*c + rotated*s` 在输入 dtype 域算（fp16/bf16 无显式提升）。
- **NPU 新算法**：fp16/bf16 路径先 `T.vcast(rint)` 升 fp32 → 6-op 链在 fp32 域 → 链尾 `T.vcast(rint)` 舍回原 dtype（每半一次，共两次）；fp32 路径直算（无中转）。
- **语义保持论证**：契约 golden 是 fp32 opmath + 单次舍回；fp32 中转链与 golden **逐位一致**（`verify_equiv.py` #2 三 dtype `max_abs=0.0`）；被拒的输入 dtype 域变体非逐位等价（fp16 `max_abs=3.9e-3`，atol 违反比 ~1.2%；bf16 亦非零）。**承接 1d 教训**（`RETROSPECTIVE` Transferable Lessons）：契约 golden 与源 kernel **异域**，fp32 中转为正确性要求而非性能偏好。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/rope.py`（`_make_rope_neox_2d` **L92–L138**；同类 `RopeNeoxKernel` L392–L399，`ROTATION_STYLE="neox"`）。
- 源仓测试基准：`/home/developer/TileOPs/tests/ops/test_rope.py`（`ref_rope_neox` L69–L82 = 契约 golden，4D 分支 `cos_full.unsqueeze(0).unsqueeze(2)`；`RopeBasicFixture` L118–L128；`RopeEdgeFixture` L131–L139；`_get_tolerances` L106–L112；`test_rope_neox_2d` L154–L160、`test_rope_neox_edge` L163–L170）。Op 层：`/home/developer/TileOPs/tileops/ops/rope.py`（`_base_freqs`）。NPU 版测试/Op：`examples/TileOPs/tests/ops/test_rope.py`、`examples/TileOPs/tileops/ops/position_encoding/rope_neox.py`。
- manifest：`examples/TileOPs/tileops/manifest/position_encoding.yaml`（`RopeNeoxOp`；**2d workload `x_shape=[2, 2048, 32, 128]`、`layout="2d"`、`dtypes=[float16]`、label `neox-2d-b2-s2k-h32-d128-fp16`**；`roofline`: `outer=batch*num_heads=64`、`x_elems=outer*seq_len*head_dim`、`cos_sin_elems=seq_len*(head_dim//2)*2`、`flops=4*x_elems`、`bytes=(2*x_elems+cos_sin_elems)*elem_bytes`）。
- 迁移元数据：`examples/TileOPs/tileops/kernels/position_encoding/rope_neox/.migration_meta.json`（`extracted_functions` 含 `_make_rope_neox_2d`）。
- 同族迁移先例（只读参考）：`examples/rope_neox/_make_rope_neox_1d/`（同任务 1d 函数，半段结构 + fp32 中转 + cos/sin 复用 + 一维 persistent）、`examples/posembedding/rope_llama31/`（**2d 头广播 3D tile 先例**）、`examples/hardtanh/_make_hardtanh_kernel/`、`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/dropout/_make_dropout_kernel/`、`examples/elu/_make_elu_kernel/`。

---
## 1. 概述

### 1.1 算子名称

`_make_rope_neox_2d`（项目 `rope_neox`，目录 `examples/rope_neox/_make_rope_neox_2d/`）。

### 1.2 功能描述

对 `(batch, seq_len, num_heads, head_dim)` 张量应用 GPT-NeoX 风格旋转位置编码（RoPE）；输入预计算的 cos/sin 频率表 `(seq_len, head_dim//2)`，**在 batch 与 head 两轴上广播**；输出同 shape/dtype。语义等价于 `ref_rope_neox`（§0.1）。支持 `float16 | bfloat16 | float32`。**单步纯 Vector 逐元素/局部配对算子**——无规约、无矩阵运算、无跨 head 依赖、无超越函数。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 坐标由「flat 三级 `%`,`//` 逐元素还原 `(s_idx,h,col)`」改为「`(tokens, num_heads, head_dim)` 静态 3D 视图 + per-tile `(b, seq_block)` 映射（cos/sin 行索引 = `seq_start+i`）」；② 计算核由「逐元素 `if_then_else` 配对 gather + 标量 `val*c+rot*s`」改为「**3D 半段（half-split）向量链**：`y1=a*c−b*s`、`y2=b*c+a*s`（6 向量 op，零 gather）」；③ batch/head 广播由「逐元素取表 + L2 吸收重复读」改为「cos/sin 按 tile 单次装载 `(bt,1,hpad)` + **同 rank size-1 头轴自动广播**」；④ 并行/内存结构由 GPU `block×thread + fragment` 改为「一维核 + 核内网格跨步 + 3D UB tile」；⑤ 计算域由「输入 dtype 域」改为 **fp32 中转 + 单次舍回**（契约对齐）。数学语义与 I/O 契约与源完全一致。

### 1.3 数学公式

逐元素定义（同 §0.1；`c`=`cos_table`，`s`=`sin_table`，shape 均 `(seq_len, half)`，在 batch/head 上广播）：

$$
y[b,s,h,col] = x[b,s,h,col]\cdot c[s,\ col\bmod half] + \operatorname{rot}(x)[b,s,h,col]\cdot s[s,\ col\bmod half]
$$
$$
\operatorname{rot}(x)[b,s,h,col] = \begin{cases}-x[b,s,h,\ col+half], & col<half\\ +x[b,s,h,\ col-half], & col\ge half\end{cases}
$$

NPU 计算核（§1.6.1 优化后，半段 3D 向量形态；sub-fp32 在 fp32 计算域构造、末端单次舍回）：

$$
y[b,s,h,\ j]=a_j c_j - b_j s_j,\qquad y[b,s,h,\ half+j]=b_j c_j + a_j s_j,\qquad j\in[0,half)
$$
$$
a_j=x[b,s,h,j],\ b_j=x[b,s,h,half+j],\ c_j=cos[s,j],\ s_j=sin[s,j].
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在数学上恒等，在坐标生成、并行/内存实现与计算域上有差异：**

- **坐标生成**（§0.6 R1）：`x` 在 kernel 内声明为 `(tokens, num_heads, head_dim)`（`tokens=batch*seq_len`）行主视图；逻辑 tile = `(batch, seq-block)`，每 tile 处理 `block_tokens` 个 seq 位置 × 全部 `num_heads` 个 head；cos/sin 行索引 = `seq_start + 行号`（**无逐元素 div/mod**）。源为逐元素 flat 三级 `%`,`//`。
- **计算核心**（§0.6 R2 / §1.6.1 #1）：3D 半段向量链——`y1 = vsub(vmul(a,c), vmul(b,s))`、`y2 = vadd(vmul(b,c), vmul(a,s))`（4 `vmul` + 1 `vsub` + 1 `vadd` = 6 op，作用于 `(block_tokens, num_heads, half_pad)` 的整 UB buffer）。源为逐元素标量 `if_then_else` + `val*c+rot*s`。
- **batch/head 广播**（§0.6 R3 / §1.6.1 #3）：cos/sin 装载为 `(block_tokens, 1, half_pad)`，靠 v-op 同 rank size-1 轴自动广播沿 head 轴展开，单次装载服务全部 head（源为逐元素取表 + L2 吸收）。
- **计算域**（§0.6 R4 / §1.6.1 #2）：sub-fp32 先 `T.vcast(rint)` 升 fp32 → 向量链在 fp32 域 → `T.vcast(rint)` 舍回；fp32 直算。匹配契约 golden 的 fp32 opmath。
- **并行/内存结构**（§0.6 R1）：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 + 3D tile `(block_tokens, num_heads, half_pad)`。

除上述外，算法与 §1.6.0 调研选定的算法族（**半段旋转 + fp32 中转 + cos/sin 单载广播复用**）一致：一趟扫描、无中间 GM 往返、无跨 head 依赖、无规约。与同任务 1d 函数的差异仅在：① tile 多一个 head 轴（3D vs 2D）；② cos/sin 多一个广播轴（size-1 头轴）；③ 逻辑 tile 映射到 `(batch, seq-block)` 而非 `(seq-block)`。

### 1.5 数据流图

```
fp16/bf16 路径（fp32 中转；产物别名 = 死亡输入 buffer）：
GM[x(tokens,H,d)] --T.copy--> xlo_ub(bt,H,hp)=x[token_start:+rows, 0:H, 0:half]
                  --T.copy--> xhi_ub(bt,H,hp)=x[token_start:+rows, 0:H, half:d]
GM[cos_table] --T.copy--> c_ub(bt,1,hp)        GM[sin_table] --T.copy--> s_ub(bt,1,hp)
   --T.vcast(rint)--> af/bf/cf/sf (fp32；cf/sf 沿 H 轴 size-1 广播)
   --[y1=vmul(af,cf); t=vmul(bf,sf); y1=vsub(y1,t)  |  y2=vmul(bf,cf) 原地; t=vmul(af,sf); y2=vadd(y2,t)]--> y1/y2 (fp32)
   --T.vcast(rint)--> xlo_ub←y1(dtype), xhi_ub←y2(dtype)   [xlo_ub/xhi_ub 死亡后复用为输出]
   --T.copy--> GM[y[token_start:+rows, 0:H, 0:half]] 与 GM[y[token_start:+rows, 0:H, half:d]]

fp32 路径（无中转；A=xlo_ub, B=xhi_ub, C=c_ub, S=s_ub，y2 原地落回 B）：
GM[x] --T.copy--> xlo_ub/xhi_ub ; GM[cos/sin] --T.copy--> c_ub/s_ub
   --[同 6-op 向量链]--> y1/y2 --T.copy--> GM[y]（两半分别写）
```

向量链内部（全部 UB 级、3D `(bt,H,hp)`、`vsub`/`vadd` 原地复用）：

```
olo = vmul(A, C, olo)     # a*c
tmp = vmul(Bv, S, tmp)    # b*s
olo = vsub(olo, tmp, olo) # y1 = a*c - b*s
tmp = vmul(A, S, tmp)     # a*s
Bv  = vmul(Bv, C, Bv)     # b*c   （原地，Bv 最后一次使用后即输出载体）
Bv  = vadd(Bv, tmp, Bv)   # y2 = b*c + a*s
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素 + 局部配对（2×2 旋转）类**（无规约/窗口/矩阵/超越函数），且**带一个广播轴（head）与一个坐标生成层（flat→3D）**。按 `_shared/standards/algorithm-research.md` §3 属轻量调研范畴；但因存在多种结构重排候选、坐标生成候选与计算域/访存复用选择，R1 给出候选表、R3 给出四口径对比、R4 逐候选评估（达到完整调研深度）。**2d 相对 1d 新增的调研对象 = 坐标生成方式 + head 广播实现方式。**

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向（op / 元素；访存 / 指令） | 纳入 R3 对比 |
|---|------|------------|-----------|--------------------|-------------|
| 0 | **基线：源算法**（flat 逐元素 + 三级 `%`,`//` 坐标 + `col%half` + `if_then_else` 配对 + 逐元素取表） | `y[flat]=val*c+rot*s`；`rot=col<half ? −x[paired] : x[paired]`；`s_idx/col` 逐元素还原 | 迭代定义式（fp32 域为契约） | 3 op/输出元素 + **逐元素三级 div/mod + 索引/select**；cos/sin 每元素一读（L2 吸收） | ✅（基线） |
| 1 | **半段旋转 + 3D 视图（采纳）** | `a=x[...,:half]`、`b=x[...,half:]`；`y1=a*c−b*s`、`y2=b*c+a*s`；坐标 = 静态 3D tile + per-tile `(b,seq_block)` 映射 | **数学恒等**（§0.1 闭式，逐项相等）；坐标映射逐位一致（`verify_equiv` #0） | 3 op/输出元素；**零 gather、零逐元素 div/mod**；cos/sin 1×/tile | ✅（**采纳**） |
| 2 | 半段结构 + **整行装载**（tile `(bt,H,d)` 单次装载，UB 内静态切片 `[:,:,0:half]`/`[:,:,half:]`） | 同 #1，仅装载粒度更宽 | 数学恒等 | op 数同 #1；**MTE2 段更宽**（d 连续 vs half）；**拷贝次数减半**（2 vs 4） | ✅（评估，备选，§1.6.3） |
| 3 | 全宽 + 显式 `rot` 物化（`rot=cat(−b,a)` 后 `y=x2*c2+rot*s2`） | `c2=cat(c,c)`、`x2=(bt,H,d)` | 数学恒等 | **+UB↔UB 拷贝 + negate**，op/UB 更多 | ✅（评估，不采纳，§1.6.1 E3） |
| 4 | 3 乘复数（Gauss 复数乘） | `m1=c(a+b), m2=a(s−c), m3=b(c+s)`；`y1=m1−m3, y2=m1+m2` | 数学恒等 | 3 mul + 5 add = **4 op/元素**（>3）；NPU mul/add 同价 | ✅（评估，不采纳，§1.6.1 E2） |
| 5 | flatten 2D `(M,half)` 行 + 逐行 cos 取表（`s = row // H`） | 2D tile 行 = head 向量 | 数学恒等 | 需**逐行 cos gather / 复制**（每 `H` 行同一 cos 行）→ NPU 无向量 gather | ❌（**不适用**：gather，§1.6.3 L3） |
| 6 | 三角函数恒等式合并 / 多项式逼近 | 合并角度或近似 `(c,s)` | #6 数学等价但引入超越函数；多项式**非等价** | 角度已在表中（无需合成）；逼近无收益 | ❌（**不适用**：数学/精度） |

**R2 在线算法**：**无在线化收益（结论：无）**——逐元素映射 + head 内局部配对（`(j)` 与 `(half+j)` 两两独立、配对不跨 head/seq/batch，§0.1）本身天然单遍；且 batch/head 广播是**共享**而非**归约**，不存在可被「单遍/流式」消去的多遍基线。结构判据：本算子**无 running 统计量**（无 max/sum/均值/平方和归约，无跨元素依赖）。依据：源算法即单遍直写（§0.3 步骤 1–5，无循环携带依赖）；`algorithm-candidates.md` 的 `ALG-elementwise` R2 记「逐元素映射天然单遍」。

**R3 复杂度对比**（四口径；本算子**带宽/发射受限**，访存/遍数是主导项；以 manifest 2d workload `[2,2048,32,128]` fp16 为例，`x_elems=16,777,216`、`cos_sin_elems=262,144`、`bytes=67.63 MB`、`bt=1`、`work_blocks=4096`）：

| 算法候选 | FLOPs（v-op 数/输出元素，fp32 域） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源） | 3（2 mul + 1 add）+ 逐元素 div/mod + select | `(2·x_elems+cos_sin_elems)·eb`（cos/sin 逻辑每元素一读，靠 L2 吸收） | 1 遍 | O(tile)（寄存器级） | 高（每元素独立），无跨核归并；**但逐元素 div/mod+select 在 NPU 上退化为标量** |
| #1 半段 3D 广播（**采纳**，fp32） | 3（4 mul+2 add per 2 输出） | `(2·x_elems+cos_sin_elems)·eb`（cos/sin 每 tile 一读，理论下界） | 1 遍 | O(tile)（UB，bt=1 ≈ 41 KB，§4.5） | 高（每 (b,seq-block) 独立），无跨核归并 |
| #1 半段（sub-fp32，+fp32 中转） | 3 + 6 cast/2 元素 = **6** | 同上（cast 无额外 GM 流量） | 1 遍 | O(tile)（+4 dtype/+6 fp32 buffer） | 同上 |
| #2 整行装载 | 同 #1 | 同上 | 1 遍 | O(tile)（+1 (bt,H,d) buffer） | 同上；**MTE2 段 256B vs 128B**（待实证） |
| #3 全宽 + rot 物化 | ≥3 + (rot 拷贝/negate 摊薄) | 同上 + rot UB 往返（UB 级，非 GM） | 1 遍 | O(tile)（+1 (bt,H,d) buffer） | 同上 |
| #4 3 乘复数 | 4（3 mul + 5 add per 2 输出） | 同上 | 1 遍 | O(tile) | 同上 |
| #5 flatten 2D + 逐行 cos gather | 3 + 逐行复制 | 同上 | 1 遍 | O(tile) | 低（gather 标量化） |

> 口径说明：FLOPs 以「每输出元素的 v-op 数」计（无超越函数；mul/add 同价，`CONST-vector-op-cost-910_9362`）；访存 = 输入读 + 输出写 + cos/sin 读，**全部候选同为理论下界 `(2·x_elems+cos_sin_elems)·elem_bytes`**（无中间 GM 往返，`x` 与 `y` 各 1 读 1 写）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **gather / 逐元素 div-mod 的有无（指令/延迟，非带宽）**、**cos/sin 是否复用（访存）**、**MTE2 段宽（#2）** 与 **op 数**。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 半段 3D 广播（采纳）** | Vector（`vmul/vadd/vsub` fp16·fp32 ✓、bf16 ✗ → fp32 中转；`vcast` f16↔f32/bf16↔f32 `rint` ✓）；**同 rank size-1 头轴广播** ✓（rope_llama31 实证） | 小（bt=1 ≈ 41 KB；§4.5）✓ | 尾轴 `half_pad`（fp16 `half=64`=128B；d=16 时 hpad=16=32B）32B 整倍 ✓ | `batch/seq_len/H/d` 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| 基线（源 gather + 逐元素 div/mod） | Vector 但**需逐元素索引 + select + 三级 div/mod**；NPU 上即变为跨步 gather（`TRAP-C10-strided-gather`：**跨步 gather 向量指令缺失**）→ 退化为标量/低效 | 同 | ✓ | ✓ | 同 | ❌ **淘汰**（gather 无向量指令；标量化后 ×元素数，VP-2026-0175 实测源 flat div/mod 路径 39 ms 级） |
| #2 整行装载 | Vector；无 gather | 略大（+1 (bt,H,d) buffer） | ✓ | ✓ | 同 | ⚠️ 等价、且段更宽/拷贝更少，但依赖**未实证的段效率**与静态 3D 子切片 → §1.6.3 备选 |
| #3 全宽 + rot 物化 | Vector；无 gather 但需 UB↔UB 拷贝/negate | 略大 | ✓ | ✓ | 同 | ❌ 等价但 op/UB 更多，取 #1/#2 |
| #4 3 乘复数 | Vector（4 op/元素 vs 3） | 略大（多 1 temp） | ✓ | ✓ | 同 | ❌ op 更多，mul/add 同价无收益 |
| #5 flatten 2D + 逐行 gather | — | — | — | — | — | ❌ gather 无向量指令（同基线） |
| #6 角度合并 / 多项式逼近 | — | — | — | — | — | ❌ 数学不适用 / 非等价 |

**v-prefix 元算 dtype 矩阵（决定 sub-fp32 必经 fp32 中转）**：`docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 的 dtype 矩阵均为 **fp16 ✓ / fp32 ✓ / bf16 ✗**；`T.vcast.md` §2.2.1 支持 `f16→f32`、`bf16→f32`、`f32→f16/bf16`（均 `rint`）。→ sub-fp32 一律经 fp32 中转（**文档为准**，AGENTS.md「API docs are source of truth」；另注：同任务 1d 的 Stage 3 设备探针实证 bf16 原生 v-op 在 Ascend910_9362 可用，与文档矩阵相左——**文档/设备分歧登记于 §9 R-3**，本设计按 §1.6.1 #2 统一 fp32 中转，不依赖原生 bf16 路径）。

**调研结论**：**选定「半段旋转（half-split）+ 3D 视图坐标映射 + 同 rank size-1 头轴广播 cos/sin 单载复用 + fp32 中转计算域」**（候选 #1；#2 作为段效率备选），无更优可行替代。

关键依据：① R3 显示全部候选访存同为理论下界 `(2·x_elems+cos_sin_elems)·eb`，差异在 **gather/逐元素 div-mod 的有无**、**cos/sin 是否复用**、**MTE2 段宽**与 **op 数**：候选 #1 是唯一同时满足「零 gather + 零逐元素 div/mod + cos/sin 1×/tile + 3 op/元素」的候选；候选 #2 与 #1 同 op 数、段更宽（未实证），列备选；#3 op/UB 更多、#4 op 更多（4 vs 3）、#5 gather、#6 数学不适用。② **2d 坐标**：源三级 `%`,`//` 在 NPU 上退化为逐元素标量（VP-2026-0175：S=2048/H=32/fp16 实测 39.03 ms）；3D 静态视图 + per-tile 映射消除之，且 `verify_equiv.py` #0 证明坐标逐位一致。③ **计算域**：契约 golden 为 fp32 opmath，源为输入 dtype 域——**异域**，必须等价替换为 fp32 中转。④ 选定算法的全部 op（`vmul/vadd/vsub/vcast/copy`）均有本仓 docs 佐证与 `examples/` 实跑先例（§3.4）。

**调研范围**（「无更优替代」依据）：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」——本算子的 CSE = cos/sin 单载复用 + 头轴广播；R2 记「逐元素映射天然单遍」；R3 记「融合消除中间 GM 往返」；R4 记「单遍 Vector 流；dtype 路径」）——库中无 RoPE 专门条目，故以恒等变形自查补足候选 #1–#6；② 本仓 `examples/` 全系（`elementwise/*`、`leaky_relu`、`elu`、`hardtanh`、`erf`、`expm1`、`bitwise_not`、`dropout`、`norm/*`、`reduction/*`）**唯一 2d RoPE 前例 = `examples/posembedding/rope_llama31/`（3D 头广播 tile 形态，本设计采纳其广播机制）**；同任务 1d 函数 `examples/rope_neox/_make_rope_neox_1d/`（半段 + fp32 中转 + cos/sin 复用）；③ pattern-library（`kb_search "rope rotation head broadcast 2d layout half-split"` + 消费预注入条目）：`PL-1.22-persistent-1d-grid-stride`（一维网格跨步模板）、`PL-1.23-vop-chain-inplace-alias`（长 v-op 链 UB 原地复用）、`PL-1.17-subfp32-fp32-transit`（sub-fp32 fp32 中转）、`PL-1.15-broadcast-perf-tax`（**未文档化广播形态 +12~20% 性能税** → §1.6.3 裁决输入）、`CONST-vector-op-cost-910_9362`（单 op 代价表）、`CONST-ascend910-9362-profile`（设备带宽画像）、`TRAP-C10-strided-gather`（gather 缺失）、`TRAP-expert-v-operands`（v-op 静态边界）、`TRAP-vbrc-empty-broadcast`（宽 1 轴归约广播被拒——本算子用 size-1 头轴 v-op 广播而非 vbrc，仅登记）、`VP-2026-0175`（源 flat div/mod 逐元素路径实证极慢）；**上述条目 `kb_stale_check` 多数标记 `[stale]`（当前 toolchain `0.1.2+a23572536…` 与其版本戳不同）**，故其**性能数字只作候选排序参考**，API 存在性以 `docs/Tilelang.language/` 为准（见 §9 R-12）；④ **外部已知算法族**（模型知识 + PyTorch/HF/CUDA 生态）：RoPE 的两种实现范式——「`cat`/`reshape` 半段旋转」与「逐元素交错配对」；复数乘视角（含 3 乘 Gauss 变体）；三角函数角度合并——均经 §1.6.1 逐项论证；无超越函数逼近空间。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = (2·x_elems + cos_sin_elems)·elem_bytes`（manifest roofline 同口径，`outer=batch*num_heads`）。引用 **`CONST-ascend910-9362-profile`**（`constants.md`，`status: verified`，版本戳 Ascend910_9362 / CANN 9.0.0 / 2026-10-13，**`kb_stale_check` 标 stale——数字作背景，非决策依据**）：单 die 核→GM ≈ **1.06 TB/s**、HBM ≈1.64 TB/s；纯逐元素大 shape 实测饱和 **1.06–1.10 TB/s（≈核→GM）**。取 **1.06 TB/s** 作流量分母。
- 发射项 = `(ceildiv(work_blocks, core_num)) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C **且 stale**，本设备未重验，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：fp32 `k=6`、sub-fp32 `k=12`（+4 `vcast` 升 +2 舍回）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 dtype 对应 `block_size`（fp32=1024、fp16/bf16=2048，K11 折叠；`block_tokens = max(1, block_size // (num_heads*head_dim))`）、物理 Vector 核 = 40 估算：

| workload | dtype | (b,s,H,d) | bt | work_blocks | bytes | 流量项 @1.06TB/s | 发射项 (k, stale) | **估算下界 max** |
|----------|---|------|----|-------|-------|-----------------|-------------------|------------------|
| **manifest 2d** | fp16 | (2,2048,32,128) | 1 | 4096 | 67.63 MB | **63.8 µs** | 103×12×0.5 = 618 µs | **≈ 63.8 µs**（流量主导） |
| basic | fp16 | (2,128,8,64) | 4 | 64 | 0.54 MB | 0.51 µs | 2×12×0.5 = 12 µs | **≈ 12 µs**（发射主导） |
| edge | fp32 | (1,1,1,16) | 64 | 1 | 0.19 KB | ≈ 0 µs | 1×6×0.5 = 3 µs | **≈ 3 µs** |

**设计期估算（D-2 roofline）**：代表性 **manifest 2d `[2,2048,32,128]` fp16** 估算下界 ≈ **63.8 µs**（流量 63.8 + 发射(stale 口径) 618 + 容量 0 → max 由流量项主导；发射项按 0.5µs/op 口径名义过剩）。**估算不确定性声明（未文档化假设 + 估算依据）**：① `CONST-vector-launch-overhead` 的 0.5µs/op 绑 910B2C **且 stale**，本设备未重验，故发射项可能**显著上偏**；② **吞吐口径交叉核对（更可信，引用 verified 的 `CONST-vector-op-cost-910_9362`：普通 v-op ≈ 9.15µs/16m-elem ≈ 1.83e12 lane-elem-op/s 聚合）**：manifest 的 lane-elem-ops = `work_blocks × k × (bt×H×hpad)` = `4096×12×2048` = 100.7M → **≈ 55.0 µs**（< 流量项 63.8 µs）；`[2,128,8,64]` = `64×12×1024` = 786K → ≈ 0.43 µs（< 流量 0.51）。即**按单 op 吞吐口径，manifest 由流量项主导，发射项（0.5µs/op 口径）系统性上偏**——与 `PL-1.6-copy-floor`「≥16M 计算链被 MTE2 窗口完全隐藏」一致（该条 stale 仅作背景）；③ `bt` 是摊薄发射的第一杠杆（发射 ∝ 1/`bt`，§5.2/§1.6.3 备选）；④ **设计期 D-3 探针实测**：`history_version/design_probe_npu_2d.py` 在设备上对 5 shape **数值逐位一致（`max_abs=0.0`）**，但探针 harness 的 launch/输出分配 host 开销约 310–330 µs，**掩盖了 kernel 本体时延**（tiny 与 manifest shape 实测同量级），故 kernel 时延对账交 **Stage 4 baseline（msprof op）**；⑤ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（半段旋转 + 3D 坐标映射 + head 广播 cos/sin 复用 + fp32 中转）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **半段旋转结构重排 + 3D 坐标去 div/mod**（采纳） | 逐元素 `y[flat]=val*c+rot*s`；`s_idx/col` 由三级 `%`,`//` 还原；`rot=col<half? −x[head_base+col+half] : x[head_base+col−half]` | `x`→`(tokens,H,d)` 视图；tile `(bt,H,hp)`；`a=x[...,:half]`、`b=x[...,half:]`；`y1=a*c−b*s`、`y2=b*c+a*s`；cos/sin 行 = `seq_start+i` | **数学恒等**（§0.1 闭式推导：`col=j` → `a_j c_j+(−b_j)s_j`；`col=half+j` → `b_j c_j+a_j s_j`）；坐标映射逐位一致（`verify_equiv` #0 三 shape `max_abs=0.0`）；舍入路径与 golden 一致（IEEE `(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)` 精确）→ #1 三 dtype × {randn, corners} `max_abs=0.0` | **消除逐元素 gather + 三级 div/mod**（NPU 上会退化为标量，VP-2026-0175 实测 39 ms 级）；op 数不变（3/元素）；UB 层零中间 GM 往返 |
| 2 | **fp32 中转计算域**（采纳 = 异域等价替换） | 输入 dtype 域：`val*c+rot*s`（fp16/bf16 原生域） | `vcast↑f32(rint) → 6-op 链（fp32 域）→ vcast↓dtype(rint)` | 与契约 golden（fp32 opmath + 单次舍回）**逐位一致**（#2 三 dtype `max_abs=0.0`）；**诊断证明输入 dtype 域非逐位等价**：fp16 `max_abs=3.9e-3`（atol 违反比 ~1.2%）、bf16 亦非零 → **必须**中转 | 正确性：对齐 golden；bf16 另受 `vmul/vadd/vsub` 无 bf16（docs §2.2.1）API 强制；代价 +4/+2 `vcast`（sub-fp32） |
| 3 | **cos/sin 单次装载 + head 轴广播复用（CSE）**（采纳） | 逐元素按 `(s_idx, col%half)` 取 `c/s`（每元素一读，batch×head 重复，L2 吸收） | cos/sin 按 tile 装载 `(bt,1,hp)`，靠 size-1 头轴广播服务全部 H 个 head；`y1`/`y2` 复用同一 `cf/sf` | 数值不变（源本就 batch/head 广播同一行，取同值）→ `verify_equiv` #1/#2 已含广播形态，`max_abs=0.0` | **cos/sin GM 流量 由「每元素一读」→「每 tile 一读」**；以 manifest 为例 cos/sin 逻辑读 ≈ `batch×` 理论最小（占比 <1% of x 流量），消除 L2 依赖 |
| E1 | 输入 dtype 原生域（**不采纳**） | — | 直接用 fp16/bf16 `vmul/vadd/vsub`（省 6 `vcast`） | **非等价**：诊断 fp16 `max_abs=3.9e-3`（atol 违反比 ~1.2%），非逐位；bf16 另 API 缺失 | op 少 6 但**精度不达标** → 拒绝；理由记录于此，非 Stage 4 备选 |
| E2 | 3 乘复数（Gauss）（**不采纳**） | — | `m1=c(a+b), m2=a(s−c), m3=b(c+s)`；`y1=m1−m3, y2=m1+m2` | 数学恒等 | op 更多（3 mul + 5 add = 8/2 元素 = 4/元素 vs 3），NPU mul/add 同价无收益 |
| E3 | 全宽 + `rot` 物化（**不采纳**） | — | `rot_ub=cat(−b,a)`；`y=x2*c2+rot*s2` | 数学恒等 | +UB↔UB 拷贝/negate + `c2=cat(c,c)` 复制，op/UB 更多 |

**优化结论**：采纳 **3 项**（#1 半段结构 + 坐标去 div/mod、#2 fp32 中转域、#3 cos/sin 单载广播复用）。**公式层面已无进一步计算量可优化空间**——每输出元素已是最少 **3 个算术 op**（2 mul + 1 add/sub；半段结构对 2×2 旋转达到最小乘法数；E2 的 3 乘方案用更多加法换 1 次乘法，在 mul/add 同价的 NPU 上更差）；访存固定在理论下界 `(2·x_elems+cos_sin_elems)·eb`（1 读 1 写 + cos/sin 每 tile 一读、无中间 GM 往返）；唯一额外 op 是 sub-fp32 的 6 次 `vcast`（bf16 为 API 强制；fp16 为 golden 对齐强制，E1 记录其不可省）。**优化后公式**（供 §3.1 使用）：

- fp32：`y1 = vsub(vmul(a,c), vmul(b,s))`；`y2 = vadd(vmul(b,c), vmul(a,s))`（6 op，`vsub` 原地、`y2` 原地落回 `b`）
- fp16/bf16：`af=cast32(a), bf=cast32(b), cf=cast32(c), sf=cast32(s)` → 同 6 op（fp32 域，`cf/sf` 沿 head 轴广播）→ `y1=cast16(·)`、`y2=cast16(·)`（+6 op）

**等价性机器验证（D-1）**：脚本 `examples/rope_neox/_make_rope_neox_2d/verify_equiv.py`（CPU；候选式 = NPU 半段 3D 广播链逐 op 复现，fp32 域计算后按目标 dtype 单次舍回；基线式 = 源 `ref_rope_neox` 2d 公式（per-element 全宽 cos/sin + 物化 rotate_half，fp32 opmath + 单次舍回）——**与候选式不同实现，无自比较**；另独立 fp64 参照与输入 dtype 域诊断）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`）：

| 采纳项 / 对照 | 候选式 | 基线式 | shape (b,s,H,d) / dtype | 元素数 | 最大 abs 差 | 违反比 | 结论 |
|---------------|--------|--------|----------|--------|------------|------|------|
| #0 2d 坐标映射（flat 三级 `%`,`//` vs `(b,s,h,col)` 查表） | flat 算术 | 3D 视图查表 | (2,2048,32,128) / (2,128,8,64) / (1,256,4,128) | 16.7M / 131K / 131K | 0.000e+00 | 0.0000 | **EQUIV_PASS** ×3 |
| #1 半段 vs 源公式（randn） | 半段 3D 链 | 源公式（全宽） | 三 shape × fp32/fp16/bf16 | 16.7M / 131K / 131K | 0.000e+00 | 0.0000 | **EQUIV_PASS** ×9 |
| #1/#2 fp64 结构参照 | 半段（fp64） | 源公式（fp64） | 三 shape × 三 dtype | — | 0.000e+00 | 0.0000 | **EQUIV_PASS** ×9 |
| #1/#2 角点（±inf/NaN/±0/次正规/±65504） | 半段链 | 源公式 | 三 shape × 三 dtype | 含角点 | 0.000e+00 | 0.0000 | **EQUIV_PASS** ×9 |
| #2 fp32 中转 vs GOLDEN（randn） | 半段链→dtype | `ref_rope_neox` fp32→dtype | 三 shape × 三 dtype | 同上 | 0.000e+00 | 0.0000 | **EQUIV_PASS** ×9 |
| #3 cos/sin 单载广播复用 | 共享 `(bt,1,hp)` tile | 源逐元素取表（隐含广播） | 含于 #1/#2 对照 | — | 0.000e+00 | 0.0000 | **EQUIV_PASS**（并入 #1/#2） |
| 非门禁诊断：被拒 E1（跳过 fp32 中转） | 输入 dtype 域链 | GOLDEN | 三 shape / fp16 | 同上 | **3.906e-03** | atol viol **~1.2%** | 非逐位等价（佐证 fp32 中转必要性） |
| 非门禁诊断：被拒 E1（跳过 fp32 中转） | 输入 dtype 域链 | GOLDEN | 三 shape / bf16 | 同上 | **3.125e-02** | atol viol 0（rtol 掩盖） | 非逐位等价（+ API 缺失） |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2；`equal_nan`），违反率 0 即 PASS；覆盖随机张量 + 角点（±0/±inf/NaN/次正规 1e-45/±65504/±88.8/1e-8）。**脚本总分：TOTAL 48 checks，48 `EQUIV_PASS`，0 not-pass；OVERALL: ALL ITEMS EQUIV_PASS**。**fp64 参照**：`candidate vs baseline` fp64 `max_abs=0.0000e+00`（结构重排零近似）。**角点结论**：半段 3D 广播链与源公式在 ±inf/NaN 上传播一致（`max_abs=0.0`），无 `inf-vs-finite` 分歧。

> **被拒候选 E1 的等价性证据**（说明输入 dtype 域**不可**采用，故 fp32 中转是正确性要求而非性能偏好）：fp16 `max_abs=3.906e-3`（atol 违反比 ~1.2%）、bf16 `max_abs=3.125e-2`——**非逐位等价**；对照 fp32 中转链 `max_abs=0.0`。与 hardtanh（源本身即 fp32、原生域逐位等价、弃选理由是 ROI）**相反**（对照 `PL-1.27-same-family-precedent-precondition`：同族先例须绑定各自契约前提）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `y=val*c+rot*s` 标量映射 | 源 `T.Parallel(threads,npt)` 内标量 | 半段 3D 整 buffer 向量链 `vmul/vadd/vsub`（佐证 `docs/.../数学算法/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.2/§2.4；`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`；`examples/elementwise/vec_add_2d.py`） | ✅ | — |
| 2 | `rot = col<half ? −x[paired] : x[paired]` 逐元素配对/select | 源逐元素 `if_then_else`（gather） | **消除**：半段结构把配对变为同 tile 内两组连续子块 `xlo_ub`/`xhi_ub`（无 select、无 gather） | ✅ | — |
| 3 | `rem%stride_b`、`rem//stride_s`、`rem2%head_dim` 三级坐标 + `s_idx` 逐元素 | 源逐元素 div/mod | **消除**：3D 静态视图 + per-tile `(b, seq_block)` 映射；cos 行 = `seq_start+i`（佐证 `verify_equiv` #0 逐位一致） | ✅ | — |
| 4 | `freq_idx = col % half` + `cos_table[s_idx,freq_idx]` 逐元素取表 + head/batch 广播 | 源逐元素 `%` + 索引（隐含广播） | **消除**：cos/sin 装载 `(bt,1,hp)` tile，size-1 头轴自动广播（佐证 `T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.2 自动广播 + rope_llama31 3D 形态） | ✅ | — |
| 5 | fp16/bf16 → fp32 提升 / 舍回 | 源无（输入域计算） | `T.vcast(src, dst, round_mode="rint")`（升/降；佐证 `docs/.../数据类型转换操作/T.vcast.md` §2.2.1/§2.4） | ✅ | — |
| 6 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(work_blocks, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | **tile/block 级任务调度循环**：每核迭代数 `ceildiv(work_blocks,40)`（manifest 103，可放大 bt 摊薄），非逐元素热点；元素级并行由向量指令整 buffer 完成。先例 `vec_add_2d.py`、`PL-1.22-persistent-1d-grid-stride` |
| 7 | tile 起始索引 / 任务映射 | `w=i*core_num+cid`；`b=w//seq_blocks`、`sb=w%seq_blocks`、`seq_start=sb*bt`、`token_start=b*seq_len+seq_start` | 无 | ❌ | **block 级索引/任务映射**：每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 8 | 行尾边界（非整除 `seq_len`） | `rows = T.min(bt, seq_len − seq_start)` | 无 | ❌ | **依赖（编译期常量）shape 的边界处理**：`seq_len` 为编译期常量但 `seq_start` 依赖运行时 `w`，尾块按 `T.min` 动态裁剪 `T.copy`；manifest workload 的 `seq_len=2048` 整除 `bt=1`，保留尾守卫以支持任意 `seq_len`（纯静态化对通用 shape 不成立） |

**向量化结论**：逐元素计算点（#1–#5）**已全部向量化**（`vmul/vadd/vsub/vcast`，均有 docs 佐证 + `examples/` 实跑先例）；保留 3 处标量/循环（#6 核内调度循环、#7 block 索引、#8 行尾边界），理由类别：**tile 级调度循环 + block 索引 + 动态边界**，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

本算子有 3 个非 batch 维（`seq_len`、`num_heads`、`head_dim`），须枚举候选布局 × 向量化轴。候选矩阵（逐候选量化）：

| 候选 | 布局 / 向量化轴 | 整除性 / 尾 lane 浪费 | 广播 / 累加链形态 | repack 代价 | UB 影响 | 结论 |
|------|----------------|----------------------|-----------|------------|---------|------|
| **L1（主选）** | I/O 原生 `(tokens,H,d)` 行主视图；**向量轴 = `half`（head 内层连续轴）**；head 轴为 tile 内轴（cos/sin size-1 广播）；seq 轴为 tile/循环轴 | `half_pad` 对齐 32B：fp16 d128 `half=64`=128B、d64 `half=32`=64B、d16 `half=8`→`hpad=16`=32B；向量宽 fp16×8 / fp32×4 整除 `half_pad`；**尾 lane 浪费 0%** | **head 轴广播**（`(bt,H,hp) op (bt,1,hp)`，同 rank size-1）；纯逐元素无累加链 | **无**（I/O 布局 = 核内布局；cos/sin 原生 `(seq,half)` 直对） | bt=1 ≈ 41 KB（§4.5）✓ | ✅ **主选** |
| L2（备选） | I/O 原生；**整行装载** tile `(bt,H,d)` 单次装载，向量链操作数用**静态 UB 子切片** `x_io[:,:,0:half]`/`[:,:,half:]` | 行 `d`（256B）更宽；同上对齐 | head 广播同 L1（`(bt,1,half)` 切片） | 无（切片为静态边界，`half` 编译期常量） | +1 `(bt,H,d)` buffer（bf16 d128 bt1 ≈ 8 KB） | ⚠️ **备选**（实验裁决）：**拷贝次数 4→2（x/y 各一次整行）**、MTE2 段 128B→256B，但依赖**未实证的段效率**与**静态 3D 子切片作 v-op 操作数**两个常数 → 见下「实验裁决」 |
| L3 | **flatten 2D `(M,half)`**（M=tokens×H）行 + 逐行 cos 取表（`s = row//H`） | flat 连续但 cos 行需**逐行复制/gather** | 无原生广播；需构造 `(M,half)` cos（每 H 行重复同一行） | 需 gather/复制 | 需按行临时量 | ❌ **弃选**：gather 无向量指令（`TRAP-C10-strided-gather`）；`npub::gather`（`T.npuir_gather` 索引搬移）非逐元素 ALU，代价不优 |
| L4 | 核内转置（`seq` 作内层向量轴） | `seq` 非连续（行距 `H*d`），跨步系数阻碍向量化 | cos/sin 按行索引 → 转置后更复杂 | 需 `T.transpose`（UB 级 repack） | 增 1~2 buffer | ❌ **弃选**：转置不减少任何访存（GM 流量不变），却引入 repack 开销；无收益（依据 `ALG-transpose` R3） |
| L5 | 按 head 分块（`block_heads < H`，tile `(bt,bh,hp)`） | 同 L1 | head 广播退化为 `(bt,1,hp)`→`(bt,bh,hp)`，但 cos 复用次数降为 `H/bh` | 无 | 同元素数下 UB 相同 | ❌ **弃选**：cos/sin 复用恶化（装载次数 ×`H/bh`），tile 数 ×`H/bh`，无收益（仅当 `H*d > block_size` 无法整 head 入 tile 时才是退路，本例可整 head 入 tile） |

**逐候选量化评分**：L1 在整除性（尾 lane 0%）、repack（无）、UB（41 KB）、广播（硬件自动）四维全部达标；L2 与 L1 同 op 数、段更宽/拷贝更少，保留为备选；L3 因 gather 移除（`TRAP-C10`）；L4 无收益移除；L5 cos 复用恶化移除。**迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → flat 元素，逐元素三级坐标）是**输入而非结论**——GPU 的 flat 元素轴与 NPU Vector 的连续 `half` 轴不对应（GPU 靠 L2 + 逐元素 select 掩盖 gather 与坐标开销），§0.6 R1/R2/R3 已独立评估为「3D 视图 + 行/头 tile + `half` 向量轴 + size-1 头轴广播」。

**布局决策结论**：主选 **L1**——核内布局 = I/O 原生 `(tokens,H,d)` 行主视图（无 repack），向量化轴 = `half` 内层连续轴，head 轴为 tile 内广播轴，seq 轴为 tile/循环轴。**决策落地**：主选布局同步落入 §3.3 伪代码（`xlo_ub/xhi_ub` 为 `(bt,H,hp)`、`c_ub/s_ub` 为 `(bt,1,hp)`）与 §6 循环结构（向量链各 op 作用于整 3D buffer，无逐元素标量循环）。

**实验裁决（判定裕度依赖未实证常数，强制三件套）**：主选 L1（拆半装载，`half` 段 64–128B）与备选 L2（整行装载，`d` 段 128–256B）的判定裕度落在三个**未实证常数**上——① MTE2 小段（128B）相对宽段（256B）的效率损失（`CONST-mte2-degradation` 记 ≈128B 段、~2.9–3.2ns/段，**stale**，且未提供 128B vs 256B 同口径对比；9362 适用性注记要求重验）；② **`PL-1.15-broadcast-perf-tax`（verified）**：未文档化的 size-1 中间轴广播形态 `[bt,H,hp] op [bt,1,hp]` 有 **+12~20% 性能税**（同 class 的列广播/alias 形态实测）——L1 主选依赖该形态，L2 亦依赖（切片后为同 shape 或 size-1），但 L2 可改用 `T.vbrc` 展开 + 同 shape 运算规避（增 1 `(bt,H,hp)` fp32 cos/sin buffer 与 2 个 vbrc op）；③ v-op 静态 3D 子切片操作数（L2 依赖）在 NPU 的可用性与代价（`TRAP-expert-v-operands` 仅证 runtime Var 被拒，静态边界理论合法但无本算子形态实测）。

- **主选方案（默认实现，进 Stage 3）**：L1 拆半装载（§3.3）：`T.copy(x[ts:ts+rows, 0:H, 0:half], xlo_ub)` + `T.copy(x[ts:ts+rows, 0:H, half:d], xhi_ub)`，`c_ub/s_ub` 为 `(bt,1,hp)`；6-op 向量链（size-1 头轴广播）。理由 = 零 gather/零逐元素 div-mod、无切片依赖、UB 预算最小、尾 lane 0%；**且已由 D-3 探针在设备上逐位验证（`max_abs=0.0`）**。
- **备选方案（结构完整，达 Stage 4 可直接实现深度）**：L2 整行装载——`x_io (bt,H,d)` 单次 `T.copy(x[ts:ts+rows, 0:H, 0:d], x_io[0:rows, 0:H, 0:d])`；向量链操作数用**静态 3D 子切片** `x_io[:,:,0:half]`、`x_io[:,:,half:d]`（`half` 编译期常量）；cos/sin 仍 `(bt,1,hp)`（其 `hp` 段可用 `T.vbrc` 展开到 `(bt,H,hp)` 规避广播税）。buffer 形状：`x_io (bt,H,d)`（dtype）+ 5 fp32 `(bt,H,hp)`（af/bf/olo/tmp + ct/st 为 `(bt,1,hp)`）；输出写回用整行 `T.copy`（两半拼 UB 或直接整行 `y`）。UB 预算：bf16 d128 bt1 ≈ 41 KB + 8 KB = 49 KB（×1.7 ≈ 83 KB，可行）；dtype 路径同 L1。分核三要素同 §5.5（`work_blocks` 不变）。
- **实验裁决计划**：代表 shape = manifest 2d `[2,2048,32,128]` fp16 + basic `[2,128,8,64]` fp16；测量指标 = `msprof op` Task Duration（median-of-20，warm-up 5）+ `aic_mte2_*` 忙比（判读段效率）；**未知常数反解** = 由 L1 vs L2 的 MTE2 时间差与段数差反解 128B vs 256B 段的单位代价、由 L1 直广播 vs L2-vbrc 的差值反解**广播税**、并验证静态 3D 子切片 v-op 是否编译通过；**判定阈值** = 备选 latency < 主选 × (1−0.05)（即 ≥5% 增益才翻转；<5% 视为 tie，保留主选）。裁决结果回写 §1.6.3 并（若翻转）修订 §3.3/§4/§6。**shape 分派评估**：工厂为 `functools.lru_cache` 形状特化，故若按 shape 分片（如大 `H*d` 用 L1、小 `H*d` 用 L2）更优，可在 Stage 4 按 manifest 分派主选/备选，达成总体最优而非全局二选一——裁决计划须评估该分派可行性。
- **设计期探针（D-3，已执行）**：`history_version/design_probe_npu_2d.py` 已实测 **L1 主选**的数值正确性（5 shape × fp16/bf16/fp32 逐位一致 `max_abs=0.0`）与契约/API 事实（flat 输入 + 3D 声明、size-1 头轴广播、`half_pad`、行尾守卫）；**未**实测段效率与广播税两个常数（交 Stage 4 实验裁决）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素/局部配对 2×2 旋转 + head 轴广播），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约（逐元素 + 头内配对，无累加序） | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`T.vmul/vadd/vsub/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |
| 广播需求 | cos/sin size-1 头轴广播 = v-op 文档化的自动广播能力 | Developer v-op 直接支持 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致，且 wrapper `RopeNeoxKernel.ascend_mode = "Developer"` 标注一致；参考 `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（Developer + 3D UB tile + 头轴广播 + `T.serial` 网格跨步）、`examples/elementwise/vec_add_2d.py`（Developer + `T.alloc_ub` + v-prefix + 2D tile + `T.serial`）、同任务 `examples/rope_neox/_make_rope_neox_1d/`（Developer 一维 persistent + 半段链 + fp32 中转）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_tokens, num_heads, half_pad), dtype/f32)` 3D 显式 UB 分配（v-prefix API 要求 buffer 在 UB） |
| 计算方式 | v-prefix API 向量链（整 3D buffer，size-1 头轴广播），非 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（半段 3D 向量链 + head 轴广播 + sub-fp32 fp32 中转 + cos/sin 复用）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `xlo=tile(x[ts:ts+rows, 0:H, 0:half])`；`xhi=tile(x[ts:ts+rows, 0:H, half:d])` | 搬入 tile 对应 seq 块的两半到 UB（3D `(bt,H,hp)`） |
| 2 | `c=tile(cos[seq_start:seq_start+rows, 0:half])`；`s=tile(sin[...])` | 搬入 cos/sin tile（`(bt,1,hp)`，沿 head 轴广播） |
| 3 | `af/bf/cf/sf = cast↑f32(·)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 4 | `y1 = a*c − b*s`；`y2 = b*c + a*s` | 半段 2×2 旋转（fp32 域，6 op，头轴广播） |
| 5 | `y1 = cast↓dtype(y1)`；`y2 = cast↓dtype(y2)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 6 | `y[ts:ts+rows, 0:H, 0:half] = y1`；`y[ts:ts+rows, 0:H, half:d] = y2` | 两半写回 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 x 前半 | `a = x[..., 0:half]` | `T.copy` | `T.copy(x[ts:ts+rows, 0:H, 0:half], xlo_ub[0:rows, 0:H, 0:half])` | Developer |
| 搬入 x 后半 | `b = x[..., half:]` | `T.copy` | `T.copy(x[ts:ts+rows, 0:H, half:d], xhi_ub[0:rows, 0:H, 0:half])` | Developer |
| 搬入 cos/sin | `c/s = cos/sin[seq_start:..]` | `T.copy` | `T.copy(cos_table[seq_start:seq_start+rows, 0:half], c_ub[0:rows, 0, 0:half])` | Developer |
| cast 升 fp32（fp16/bf16） | `af = cast32(a)` | `T.vcast` | `T.vcast(xlo_ub, af, round_mode="rint")` | Developer |
| 乘 `a*c` / `b*s` / `a*s` | `vmul` | `T.vmul` | `T.vmul(A, C, olo)` / `T.vmul(Bv, S, tmp)` / `T.vmul(A, S, tmp)` | Developer |
| `y1 = a*c − b*s` | `vsub` | `T.vsub` | `T.vsub(olo, tmp, olo)`（`tmp=b*s`；原地落回 `olo`） | Developer |
| `y2` 载体 `b*c` + `+a*s` | `vmul`/`vadd` | `T.vmul`/`T.vadd` | `T.vmul(Bv, C, Bv)`（原地）；`T.vadd(Bv, tmp, Bv)` | Developer |
| cast 舍回（fp16/bf16） | `y1 = cast16(y1)` | `T.vcast` | `T.vcast(olo, xlo_ub, round_mode="rint")`（`xlo_ub` 死亡后复用） | Developer |
| 搬出两半 | `y[..., 0:half]=y1`；`y[..., half:]=y2` | `T.copy` | `T.copy(xlo_ub[0:rows, 0:H, 0:half], y[ts:ts+rows, 0:H, 0:half])` / `T.copy(xhi_ub[0:rows, 0:H, 0:half], y[ts:ts+rows, 0:H, half:d])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(40, is_npu=True)` + `for i in T.serial(ceildiv(work_blocks, 40))` | Developer |
| 行尾守卫 | `rows = min(bt, seq_len−seq_start)` | `T.copy`（切片 extents 借用） | `rows = T.min(block_tokens, seq_len - seq_start)` + `[0:rows, ...]` | Developer |
| UB 分配 | — | `T.alloc_ub` | `T.alloc_ub((block_tokens, num_heads, half_pad), dtype/f32)` | Developer |

> **注（广播形态）**：`T.vmul`/`T.vadd`/`T.vsub` §2.2.2 支持**自动广播**且**要求同 rank**；本设计 cos/sin 为 3D `(bt,1,hp)`、data 为 3D `(bt,H,hp)`——**同 rank、size-1 中间轴广播**，已由 `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（`T.vmul(x_lo (bt,bh,hp), cos_tile (bt,1,hp), out_lo)`）+ 本设计 D-3 设备探针双重佐证。`T.vmul` 仅 `src2` 可标量、`T.vadd`/`T.vsub` 仅 `src1` 可标量——本算子**无标量操作数**，规避该不对称性。`PL-1.15-broadcast-perf-tax` 提示该形态有性能税，见 §1.6.3 裁决。

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数**数据类型直接作为第二个位置参数**（不用 `dtype=` 关键字）；`prim_func` 参数名/顺序与 wrapper 契约一致（`x, cos_table, sin_table, y`，`out_idx=[3]`，**无额外运行时 shape 参数**——shape 为编译期常量，wrapper 约定 `fn(x_flat, cos, sin)`）；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（`PL-1.22`/`TRAP-expert-v-operands`）；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）；计算域别名用 **Python 三元表达式**（非运行时 `if`）。**K11 铁律**：所有 tiling 派生（`block_tokens`/`seq_blocks`/`work_blocks`/`iters`）必须写在**被传入实参 `blk` 的内层 `kernel(blk)`**（承接 1d 的 Stage 3 教训）。

```python
import functools
import tilelang
import tilelang.language as T

_CORE_NUM = 40                     # 实查物理 Vector 核数（20 AI Core × 2，§5.5 ②）
_COMPUTE = "float32"
_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


@functools.lru_cache(maxsize=32)
def _make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype, block_size=2048):
    """NPU (target=npuir) RoPE NeoX, layout=2d.

    入参 x 为 wrapper 展平的 flat (N_total,)，在 kernel 内声明为 (tokens,H,d) 行主视图。
    y[:, :half] = a*c - b*s ;  y[:, half:] = b*c + a*s   (a=x[...,:half], b=x[...,half:])
    cos/sin (seq_len, half) 沿 head 轴广播（size-1 头轴）。
    wrapper K11: block_size = GPU threads*num_per_thread (fp32 1024 / fp16,bf16 2048)。
    """
    half = head_dim // 2
    tokens = batch * seq_len
    need_cast = (dtype != _COMPUTE)                          # 编译期常量
    block_elems = 32 // _ELEM_BYTES[dtype]                   # fp16/bf16=16, fp32=8（32B 对齐）
    half_pad = (half + block_elems - 1) // block_elems * block_elems

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(blk):                                         # blk == wrapper 的 block_size
        # K11：tiling 全部按传入实参 blk 派生（不得在工厂体外层用默认值）
        block_tokens = max(1, int(blk) // (num_heads * head_dim))
        seq_blocks = (seq_len + block_tokens - 1) // block_tokens
        work_blocks = batch * seq_blocks
        iters = (work_blocks + _CORE_NUM - 1) // _CORE_NUM

        @T.prim_func
        def main(
            x: T.Tensor((tokens, num_heads, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((tokens, num_heads, head_dim), dtype),
        ):
            with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
                # --- buffer 顶层无条件分配（TRAP-conditional-alloc）---
                xlo_ub = T.alloc_ub((block_tokens, num_heads, half_pad), dtype)  # x 前半（死->复用为 y1）
                xhi_ub = T.alloc_ub((block_tokens, num_heads, half_pad), dtype)  # x 后半（死->复用为 y2）
                c_ub = T.alloc_ub((block_tokens, 1, half_pad), dtype)            # cos（头轴广播）
                s_ub = T.alloc_ub((block_tokens, 1, half_pad), dtype)            # sin（头轴广播）
                af = T.alloc_ub((block_tokens, num_heads, half_pad), _COMPUTE)
                bf = T.alloc_ub((block_tokens, num_heads, half_pad), _COMPUTE)
                cf = T.alloc_ub((block_tokens, 1, half_pad), _COMPUTE)
                sf = T.alloc_ub((block_tokens, 1, half_pad), _COMPUTE)
                olo = T.alloc_ub((block_tokens, num_heads, half_pad), _COMPUTE)  # y1
                tmp = T.alloc_ub((block_tokens, num_heads, half_pad), _COMPUTE)  # 临时积

                # 编译期计算域别名（Python 三元；fp32 -> 输入 buffer 即计算域）
                A = af if need_cast else xlo_ub
                Bv = bf if need_cast else xhi_ub
                C = cf if need_cast else c_ub
                S = sf if need_cast else s_ub

                for i in T.serial(iters):                    # 循环上界静态
                    w = i * _CORE_NUM + cid                  # 网格跨步（tile = (b, seq_block)）
                    if w < work_blocks:                      # 尾块守卫
                        b = w // seq_blocks
                        sb = w % seq_blocks
                        seq_start = sb * block_tokens
                        rows = T.min(block_tokens, seq_len - seq_start)
                        ts = b * seq_len + seq_start

                        # GM -> UB（仅 T.copy 用动态切片）
                        T.copy(x[ts:ts + rows, 0:num_heads, 0:half], xlo_ub[:rows, 0:num_heads, 0:half])
                        T.copy(x[ts:ts + rows, 0:num_heads, half:head_dim], xhi_ub[:rows, 0:num_heads, 0:half])
                        # cos/sin 单次装载，沿 head 轴广播复用（R3/CSE）
                        T.copy(cos_table[seq_start:seq_start + rows, 0:half], c_ub[:rows, 0, 0:half])
                        T.copy(sin_table[seq_start:seq_start + rows, 0:half], s_ub[:rows, 0, 0:half])

                        if need_cast:                        # Python 编译期条件（trace 期折叠）
                            T.vcast(xlo_ub, af, round_mode="rint")
                            T.vcast(xhi_ub, bf, round_mode="rint")
                            T.vcast(c_ub, cf, round_mode="rint")
                            T.vcast(s_ub, sf, round_mode="rint")

                        # --- 半段 2×2 旋转（fp32 域，整 3D buffer，head 轴广播）---
                        T.vmul(A, C, olo)                # a*c
                        T.vmul(Bv, S, tmp)               # b*s
                        T.vsub(olo, tmp, olo)            # y1 = a*c - b*s
                        T.vmul(A, S, tmp)                # a*s
                        T.vmul(Bv, C, Bv)                # b*c（原地；Bv 最后使用后即输出载体）
                        T.vadd(Bv, tmp, Bv)             # y2 = b*c + a*s

                        if need_cast:
                            # 单次舍回（R-9）；xlo_ub/xhi_ub 已死，复用为输出
                            T.vcast(olo, xlo_ub, round_mode="rint")
                            T.vcast(Bv, xhi_ub, round_mode="rint")
                            T.copy(xlo_ub[:rows, 0:num_heads, 0:half], y[ts:ts + rows, 0:num_heads, 0:half])
                            T.copy(xhi_ub[:rows, 0:num_heads, 0:half], y[ts:ts + rows, 0:num_heads, half:head_dim])
                        else:
                            T.copy(olo[:rows, 0:num_heads, 0:half], y[ts:ts + rows, 0:num_heads, 0:half])
                            T.copy(Bv[:rows, 0:num_heads, 0:half], y[ts:ts + rows, 0:num_heads, half:head_dim])

        return main
    return kernel
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件，trace 期折叠，不产生运行时分支；buffer 全部在顶层**无条件分配**（`TRAP-conditional-alloc`）。② **计算域别名 `A/Bv/C/S`** 用 Python 三元表达式求值（规避 `TRAP-tvm-parser-rules` 的 `if` 作用域问题）。③ v-op 全部作用于**整 buffer**；`T.copy` 用 `[0:rows, 0:H, 0:half]` 动态行切片（`TRAP-expert-v-operands`：v-op 切片仅静态边界）。④ **buffer 复用**：`xlo_ub`/`xhi_ub` 在升 fp32 后死亡，链尾复用为输出（`PL-1.23` 读前写后纪律）；`Bv` 原地承载 y2（`dst==src0`，doc §2.3 允许；**非** `dst==src2` alias，规避 `PL-1.15` 的 alias 税）。⑤ `T.vsub(olo, tmp, olo)` 原地（`dst==src0`）。**op 计数**：fp32 = 6；fp16/bf16 = 12（+4 升 +2 舍回）。
>
> **接口适配说明（wrapper 契约遵守）**：集成 wrapper `rope_neox.py` 约定 NPU 工厂签名 `_make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype)`（K3/K11 去 `threads`/`num_per_thread`，保留 4 个 shape 位置参数），返回可调用 `kernel(block_size)`，`prim_func` 为 `main(x, cos_table, sin_table, y)`（`out_idx=[3]`），`forward` 2d 分支调 `self._compiled_fn(x_flat, cos, sin)`（`x_flat = x.contiguous().reshape(-1)`）。本设计**完全对齐**：工厂 5 位置参数 + 返回 `block_size` 单参可调用；`main` 保留 x/cos/sin/y 的名与顺序；**不新增运行时 `shape` 参数**（shape 编译期常量）。**声明视图说明**：`x`/`y` 声明为 `(tokens, H, d)` 3D，但 wrapper 传入的是 flat `(N_total,)`——NPU launcher 为**指针式**（`JitKernel_NPU.__call__` 仅校验实参个数、按 `data_ptr()` 传指针，不校验 shape），且 `(tokens,H,d)` 行主布局与 flat `(N_total,)` **逐字节一致**，故寻址等价（**D-3 探针 5 shape 设备实证通过**，§9 R-1 记录该假设）。
>
> **尾块 stale 说明**：`rows < block_tokens` 时，`xlo_ub/xhi_ub/c_ub/s_ub` 的第 `[rows:bt)` 行含 UB 陈旧数据参与 v-op，但其结果不写出（仅 `[0:rows]` 拷回），`half_pad` 的 `[half:hp)` 列同理（仅 `[0:half]` 写出）→ 尾块零污染（`PL-1.22` 尾块零起点切片纪律；D-3 探针 `rows` 尾块 shape 实证通过）。

### 3.4 API 可行性确认

| API | 来源确认（本地） | 状态 |
|-----|-----------------|------|
| `T.vmul(src1, src2, dst)`（逐元素乘，**自动广播**，dst 可原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.1/§2.2.2/§2.3/§2.4；`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（3D `(bt,H,hp)×(bt,1,hp)` 广播实跑） | ✅ 真实可用（fp16 ✓ / fp32 ✓ / bf16 ✗ 按 doc） |
| `T.vadd(src0, src1, dst)` / `T.vsub(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vadd.md`·`T.vsub.md` §2.1/§2.2.2/§2.3/§2.4 | ✅ 真实可用（fp16 ✓ / fp32 ✓ / bf16 ✗ 按 doc） |
| `T.vcast(src, dst, round_mode="rint")` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32`/`f32→f16/bf16` 支持 `rint`）/§2.4 | ✅ 真实可用 |
| `T.copy`（3D/2D slice，动态行尾块） | `docs/Tilelang.language/内存操作/T.copy.md`；`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（`x[token_start:.., head_start:.., 0:half] → x_lo_io[.., :, 0:half]`） | ✅ 已核对 |
| `T.alloc_ub`（3D） | `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（`T.alloc_ub((block_tokens, block_heads, half_pad), dtype)`）；`docs/.../T.vmul.md` §2.4 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `examples/elementwise/vec_add_2d.py`；`PL-1.22-persistent-1d-grid-stride` | ✅ 已核对 |
| `T.min`（行尾块 extents） | `docs/Tilelang.language/内存操作/T.copy.md`；`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（`T.min(block_tokens, seq_len - seq_start)`） | ✅ 已核对 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(40, is_npu=True)`（**3D 是 UB tile 的维度，非 Kernel 维度**） |
| GPU 专用 API | Yes（源 `threads=`、`T.Parallel(threads,npt)`、flat 逐元素 select/gather/div-mod） | 换为 v-prefix 3D 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；行尾非整除由 `rows = T.min(...)` + `T.copy` 动态切片处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数（1–4096）与物理核 40 的适配见 §5.5（网格跨步 persistent） |
| **bf16 v-prefix 数学 API 缺失（按 doc）** | Yes | `vmul/vadd/vsub` 不含 bf16（doc §2.2.1）→ sub-fp32 走 fp32 中转（§0.6 R4）；**文档/设备分歧**见 §9 R-3 |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| v-op 动态 slice 不可用 | Yes | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:rows, 0:H, 0:half]`（§3.3） |
| `T.vbrc` 宽 1 广播被拒 | No | 本算子用 **v-op 自动广播**（非 vbrc）；且 `half_pad ≥ 8`，无宽 1 归约轴（登记 `TRAP-vbrc-empty-broadcast`，不触发） |
| v-op 无 fused multiply-add | Yes | 无 `vmadd`/`vaxpy` → 用 4 `vmul` + 2 add/sub 显式链（§1.6.1，候选 #4 的 3 乘方案不做融合替代） |
| 未文档化广播形态性能税 | Yes | size-1 头轴广播（`PL-1.15`）；数值可用、性能税待实证 → §1.6.3 实验裁决 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[3])` | `@tilelang.jit(out_idx=[3], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N_total, block_size), threads=256)` | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 3D 整 buffer 向量指令 | K11：`threads×npt` 折叠为 `block_size` |
| **坐标生成** | flat 三级 `%`,`//` 逐元素还原 `(s_idx,h,col)` | `(tokens,H,d)` 静态视图 + per-tile `(b,seq_block)` 映射 | §0.6 R1 + §1.6.1 #1 |
| **计算核** | 逐元素 `if_then_else` 配对 + `val*c+rot*s`（gather） | 半段 3D 向量链 `y1=a*c−b*s`、`y2=b*c+a*s`（6 op，零 gather） | §0.6 R2 + §1.6.1 #1 |
| **batch/head 广播** | 逐元素取表 + L2 吸收重复读 | cos/sin `(bt,1,hp)` 单载 + size-1 头轴自动广播 | §0.6 R3 + §1.6.1 #3 |
| **计算域** | 输入 dtype 域 | **fp32 中转**（`vcast` 升 → 链 → `vcast` 舍回） | §0.6 R4 + §1.6.1 #2 |
| Buffer 层级 | 纯寄存器（无显式 SMEM/fragment） | `T.alloc_ub`（UB）3D tile | §0.5 等价替换 |
| prim_func 参数 | `main(x(N_total,), cos, sin, y(N_total,))` | `main(x(tokens,H,d), cos, sin, y(tokens,H,d))`（flat 输入 + 3D 声明视图） | 对齐 wrapper 契约（指针式 launcher） |
| strategy/autotune | `threads`/`num_per_thread` + `lru_cache` | 无 strategy 维度；`block_size` 单参 + `lru_cache` | K8/K9/K11 |
| layout=1d / THD / non_neox / llama31 / yarn / longrope | 同文件其它工厂/类 | 无 | **不迁移**（本迁移只 port `_make_rope_neox_2d`；1d 已独立完成） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py` | **高度相似（2d 头广播 3D tile 先例）** | 3D UB tile `(bt,bh,hp)` + cos/sin `(bt,1,hp)` 头轴广播 + `T.serial` 网格跨步 + `half_pad` 32B 对齐 + 行尾 `T.min` |
| `examples/posembedding/rope_llama31/kernels/table_low_precision.py` | 相似（同族 2d 变体） | 3D tile + 逐 head 串行 + fp32 中转 |
| `examples/posembedding/rope_llama31/selector.py` | 参考（UB 预算与 tile 选择） | `_ub_bytes` 预算模型、`half_pad` 对齐、核数 48（本设备实查 40） |
| `examples/rope_neox/_make_rope_neox_1d/_make_rope_neox_1d.py` | **高度相似（同任务 1d 函数）** | 半段向量链、fp32 中转 `vcast(rint)`、cos/sin 复用、一维 persistent 网格跨步、`block_rows` 由 `blk` 派生、尾块动态切片 |
| `examples/elementwise/vec_add_2d.py` | 高度相似（一维核 + tile + 网格跨步） | `T.Kernel(40, is_npu=True)` + `T.serial` + 2D `T.copy` + 2D v-op |
| `examples/bitwise_not/bitwise_not_fwd_kernel/` | 相似（Developer 一维 persistent 迁移） | 一维 persistent 迁移形态、核内串行网格跨步、守卫 |
| `examples/hardtanh/_make_hardtanh_kernel/` | 相似（同族迁移，fp32 中转 + `verify_equiv.py` 范式） | 迁移文档结构、fp32 中转链、条件常量折叠 |
| `examples/dropout/_make_dropout_kernel/` | 相似（同族迁移，v-op dtype 矩阵核对） | fp32 中转链 + bf16 API 缺失处置 + v-op 操作数形态 |
| `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x,cos,sin,y)` 契约 + 2d flat 入参 + `ascend_mode="Developer"` |
| `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md`·`数据类型转换操作/T.vcast.md`·`内存操作/T.copy.md`·`shape操作/T.vbrc.md` | API | 向量链各 op 规格、dtype 矩阵、广播形态与用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x`（Op 层 4D；kernel 声明 3D 视图） | `(batch, seq_len, num_heads, head_dim)` → `(tokens, num_heads, head_dim)`（`tokens=batch*seq_len`，编译期常量） | `float16 \| bfloat16 \| float32` | 行主连续；manifest 2d：`[2,2048,32,128]` fp16 |
| `cos_table` | `(seq_len, head_dim // 2)` | 同 `x` | Op 层生成（fp32 计算后 `.to(dtype)`）；在 batch/head 广播 |
| `sin_table` | `(seq_len, head_dim // 2)` | 同 `x` | 同上 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(tokens, num_heads, head_dim)`（`out_idx=[3]` 自动分配，wrapper `reshape` 回 4D） | `same_as(x)` | shape/dtype 与输入一致，无转置 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `xlo_ub` | `(block_tokens, num_heads, half_pad)` | `dtype` | UB | `x` 前半（每 head 的 `[0:half)`）；**死亡后复用为 `y1` 输出** |
| `xhi_ub` | `(block_tokens, num_heads, half_pad)` | `dtype` | UB | `x` 后半；**死亡后复用为 `y2` 输出** |
| `c_ub` | `(block_tokens, 1, half_pad)` | `dtype` | UB | cos tile（size-1 头轴广播） |
| `s_ub` | `(block_tokens, 1, half_pad)` | `dtype` | UB | sin tile（size-1 头轴广播） |
| `af`/`bf` | `(block_tokens, num_heads, half_pad)` ×2 | `float32` | UB | fp32 计算域 a/b（fp16/bf16 升入；fp32 路径未用，仍顶层分配） |
| `cf`/`sf` | `(block_tokens, 1, half_pad)` ×2 | `float32` | UB | fp32 计算域 cos/sin（广播） |
| `olo` | `(block_tokens, num_heads, half_pad)` | `float32` | UB | `y1` 结果 |
| `tmp` | `(block_tokens, num_heads, half_pad)` | `float32` | UB | 临时积（`b*s` / `a*s`） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；`xlo_ub`/`xhi_ub` 死亡后复用为输出（`PL-1.23` 读前写后纪律）；`bf` 原地承载 `y2`（`dst==src0`）；fp32 路径 `af/bf/cf/sf` 分配后未用（UB 冗余 ≈ 17 KB（d128）或 ≈ 4.6 KB（d64），可接受，避免条件 alloc）。

### 4.4 内存搬运路径

```
fp16/bf16 路径（fp32 中转）：
GM[x] --T.copy--> UB[xlo_ub/xhi_ub (3D, 两半)] --T.vcast(rint)--> UB[af/bf (f32)]
GM[cos/sin] --T.copy--> UB[c_ub/s_ub (bt,1,hp)] --T.vcast(rint)--> UB[cf/sf (f32, 头轴广播)]
   --[olo=vmul(A,C); tmp=vmul(Bv,S); olo=vsub(olo,tmp); tmp=vmul(A,S); Bv=vmul(Bv,C); Bv=vadd(Bv,tmp)]--> UB[olo,Bv (f32)]
   --T.vcast(rint)--> UB[xlo_ub/xhi_ub (复用为输出, dtype)] --T.copy--> GM[y (两半)]

fp32 路径（无中转；A=xlo_ub, Bv=xhi_ub, C=c_ub, S=s_ub）：
GM[x] --T.copy--> xlo_ub/xhi_ub ; GM[cos/sin] --T.copy--> c_ub/s_ub
   --[同 6-op 向量链，Bv 原地]--> olo/xhi_ub --T.copy--> GM[y]
```

### 4.5 UB 内存预算

`block_size` = K11 折叠值（fp32=1024、fp16/bf16=2048）；`block_tokens = max(1, block_size // (num_heads*head_dim))`；每 tile 每 buffer 元素数 `E = block_tokens × num_heads × half_pad`（fp32 大 buffer）/ `Et = block_tokens × half_pad`（cos/sin）。

**UB 逐 buffer 预算表（manifest 最大情形：fp16/bf16, H=32, d=128 → block_tokens=1, half_pad=64, E=2048）**：

| buffer | dtype | 元素数 | 单 buffer 大小 |
|--------|-------|--------|---------------|
| xlo_ub | fp16/bf16 | 2048 | 4 KB |
| xhi_ub | fp16/bf16 | 2048 | 4 KB |
| c_ub | fp16/bf16 | 64 | 128 B |
| s_ub | fp16/bf16 | 64 | 128 B |
| af | fp32 | 2048 | 8 KB |
| bf | fp32 | 2048 | 8 KB |
| cf | fp32 | 64 | 256 B |
| sf | fp32 | 64 | 256 B |
| olo | fp32 | 2048 | 8 KB |
| tmp | fp32 | 2048 | 8 KB |
| **合计** | — | 12288 | **40.75 KB** |

> **各 dtype/shape 情形**（`E` 随 shape 变化）：fp16/bf16 d128 H32（bt=1, E=2048）**40.75 KB**；fp32 d128 H32（bt=1, E=2048，af/bf/cf/sf 未用）≈ **48 KB**；fp16 d64 H8（bt=4, E=1024）≈ **21.5 KB**；fp32 d16 H1（bt=64, E=8）≈ **20 KB**。**最大路径 = fp32 d128 ≈ 48 KB**。
>
> **UB 硬上限 = 192 KB/AIV**（`CONST-capacity-910B2C`，容量值与 9362 一致，**`kb_stale_check` 标 stale**；本设备同量级）。最大路径 ≈ 48 KB，**×1.7 最坏膨胀 ≈ 82 KB < 192 KB ✓**。**膨胀系数注（CG-2026-0008）**：逐 buffer 显式 alloc 结构实测 ~12% 余量即足（正负两向偏差均无文档，故按最坏上偏 1.7× 保守核对）。**block_tokens 放大空间（容量口径）**：manifest sub-fp32 取 bt=2 → E=4096 → ≈ 81.5 KB（×1.7 ≈ 139 KB，可行）；bt=3 → ≈122 KB（×1.7 ≈ 208 KB，**超限**）。故 **bt 放大是 Stage 4 发射项调优杠杆**（§5.2/§1.6.3 备选），默认取 K11 派生值（manifest bt=1）。

### 4.6 动态轴定义

**无运行时动态轴**：`batch`/`seq_len`/`num_heads`/`head_dim` 均为编译期常量（`lru_cache` 特化；wrapper 逐 shape 构造）。行尾块由编译期常量 `seq_len` 与运行时 `seq_start` 组合表达 `rows = T.min(block_tokens, seq_len - seq_start)`，**不引入独立运行时 shape 参数**（对齐 wrapper 契约 `fn(x_flat, cos, sin)`）。

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

**判定依据**: 算子仅含逐元素/局部配对向量链（`vmul/vadd/vsub/vcast`）+ size-1 头轴自动广播，无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
block_tokens = max(1, block_size // (num_heads * head_dim))   # 每 tile 的 seq 位置数（编译期静态）
seq_blocks   = ceildiv(seq_len, block_tokens)                 # 每 batch 的 seq tile 数
work_blocks  = batch * seq_blocks                             # 逻辑 tile 总数
```

| dtype | H × d | block_size | block_tokens | seq_blocks | work_blocks |
|-------|-------|-----------|--------------|-----------|-------------|
| fp16/bf16 | 32 × 128（manifest） | 2048 | **1** | 2048 | 4096 |
| fp16 | 8 × 64 | 2048 | 4 | 32 | 64 |
| bf16 | 4 × 64 | 2048 | 8 | 17 | 34 |
| fp32 | 32 × 128 | 1024 | 1 | 2048 | 4096 |
| fp32 | 1 × 16（edge） | 1024 | 64 | 1 | 1 |

**block_size / block_tokens 选择理由**：
1. `block_size` 与源 K11 折叠值一致（`threads(256)×npt`：fp32 1024 / fp16·bf16 2048），迁移可追溯；`block_tokens = block_size // (H*d)` 是「每 tile 目标 x 元素数」的自然 2d 推广，`max(1, ·)` 兜底 `H*d > block_size` 的情形（manifest `H*d=4096 > 2048` → bt=1，即**一个 seq 位置 × 全部 head** 入 tile，等价于「抬高到最小可行整 head tile」）。
2. 对齐：`(block_tokens, H, half_pad)` 行 = `half_pad` 元素（fp16 `half=64` 128B、`half=32` 64B、`half=8`→`hpad=16` 32B），32B 整倍 ✓。
3. UB 容量：最大路径（fp32 d128）≈ 48 KB，×1.7 ≈ 82 KB < 192 KB ✓。
4. **发射项权衡**：本算子为 6–12 op 短链，§1.6.0 估算显示发射项名义主导（发射 ∝ 1/`block_tokens`），但**吞吐口径交叉核对**（`CONST-vector-op-cost-910_9362`）显示 manifest 由流量项主导；`bt` 放大受 UB 膨胀系数不确定性约束。默认取 K11 派生值；**`bt=2`（manifest，需显式覆盖 block_size 或按其语义抬高）列为 Stage 4 首选杠杆**（须以编译探针校准 UB 后采纳，属分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：3D tile 尾轴 = `half_pad` 元素（fp16 `half=64`=128B、`half=32`=64B、`half=8`→`hpad=16`=32B；fp32 `half=64`=256B、`half=8`→`hpad=8`=32B），整 buffer 连续，行距无跨步系数（核内布局紧凑）；满足 32B 尾轴对齐 ✓；向量宽 fp16×8 / fp32×4 整除 `half_pad`，尾 lane 浪费 0% ✓。
- **UB 容量**：最大 48 KB（fp32 d128），×1.7 最坏膨胀 ≈ 82 KB < 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。
- **rank 一致**：全部 v-op 操作数为 3D（data `(bt,H,hp)` / cos-sin `(bt,1,hp)`），同 rank + size-1 广播 ✓。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest 2d workload `seq_len=2048` 整除 `block_tokens=1`（无尾块）；但源仓测试 `RopeEdgeFixture`（`seq_len=1`）与任意 `seq_len` 提示需通用支持，故**保留尾块守卫**。

**策略**：Kernel 内动态行尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 seq tile 用 `rows = T.min(block_tokens, seq_len - seq_start)`（`seq_start = sb*block_tokens`）裁剪 `T.copy` 的一维行 extent（`[0:rows, 0:H, 0:half]`）；向量 op 仍作用于**整 buffer**（尾部行的陈旧数据不影响仅写出的 `[0:rows]`；`PL-1.22` 尾块零起点切片纪律）。`block_tokens` 由 `max(1, ...)` 保证 ≥1（规避 `TRAP-ub-blockm-gt-M-overflow` 的假性 UB overflow：tile 轴不依赖 M）。任务网格用 `ceildiv(seq_len, block_tokens)` 保证覆盖最后一个不完整 tile；D-3 探针已验证 `(2,130,4,64)` 尾块 shape 逐位一致。参考 `docs/Tilelang.language/内存操作/T.copy.md` 与 `examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子核切分落在「逻辑 tile = (batch, seq_block)」网格上（head 维与 head_dim 均完整落在 tile 内，不切）：

```
num_logical_kernels = work_blocks = batch * ceil(seq_len / block_tokens)
```

| workload | dtype | (b,s,H,d) | block_tokens | work_blocks（逻辑核数） |
|----------|-------|-----------|--------------|------------------------|
| **manifest 2d** | fp16 | (2,2048,32,128) | 1 | **4096** |
| basic | fp16 | (2,128,8,64) | 4 | 64 |
| bf16 | bf16 | (2,130,4,64) | 8 | 34 |
| edge | fp32 | (1,1,1,16) | 64 | 1 |

**② 物理核数依据（实查）**：目标设备 arch 字符串 `Ascend910B3`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
NPUUtils.get().get_arch()              # 实际返回 "Ascend910B3"
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-14 本会话实查；与 `CONST-aicore-910_9362` 的 20 AIC/40 AIV 拓扑一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**；arch 字符串 `Ascend910B3` 与 constants 表 `910_9362` 的绑定差异见 §9 R-13。

**③ 规模判定与分核方案**：分档处置（物理核数 40）：

- **edge `[1,1,1,16]` fp32（work_blocks=1）**：**逻辑核数 ≤ 物理核数** → 结论「**无需适配**」（1/40 核，其余空转，小 shape 核启动开销可忽略）。仍统一走网格跨步形态（`iters=1`），不特化分支。
- **bf16 `(2,130,4,64)`（work=34）/ basic `(2,128,8,64)`（work=64）**：**逻辑核数 ≤ / 略超物理核数**（34 < 40；64 = 1.6×40）→ 取网格跨步（`iters=1`/`2`）。64 时残留：64 = 40×1 + 24 → 24 核 2 次 / 16 核 1 次 → 临界路径惩罚 2/1.6 = **+25.0%**。
- **manifest `[2,2048,32,128]`（work=4096）/ fp32 d128（work=4096）**：**极大规模**（4096 = 102.4×40）→ **固定启动内核数 = 物理核数 40**，核内 `T.serial` 网格跨步分摊。注：`seq_len=2ⁿ`、`batch=2ᵏ` → `work_blocks` 恒为 2 的幂，物理核数 40 = 2³×5 含非 2 因子 5，**数学上无法**通过对齐 `block_tokens`（2 的幂）使 `work_blocks` 成为 40 的整数倍（5 ∤ 2ᵐ）→ 属「无法对齐兜底」，取最接近 40 倍数并显式记录残留负载不均（见下）。

**统一分核方案（网格跨步 persistent，`core_num=40`）**：

```python
for i in T.serial(T.ceildiv(work_blocks, core_num)):  # 循环上界静态（work_blocks、core_num 均编译期常量）
    w = i * core_num + cid                           # 网格跨步（grid-stride）
    if w < work_blocks:                              # 守卫
        b = w // seq_blocks ; sb = w % seq_blocks
        seq_start = sb * block_tokens
        rows = T.min(block_tokens, seq_len - seq_start)
        ...
```

- **循环边界静态** ✓（`work_blocks`、`core_num` 均编译期常量，`T.ceildiv(work_blocks, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：

| workload | work_blocks | iters = ceil(work/40) | 分配（40 核） | 临界路径惩罚 |
|----------|-------------|-----------------------|--------------|-------------|
| manifest fp16 / fp32 d128 | 4096 | 103 | 4096 = 40×102 + 16 → 16 核 103 次 / 24 核 102 次 | 103/102.4 = **+0.6%** |
| basic fp16 | 64 | 2 | 24 核 2 次 / 16 核 1 次 | 2/1.6 = **+25.0%** |
| bf16 (2,130,4,64) | 34 | 1 | 34 核 1 次 / 6 核 0 次 | 核占用 85%（空转，非不均） |
| edge fp32 | 1 | 1 | 1 核 1 次 / 39 核 0 次 | 核占用 2.5%（空转） |

- **相较直接 `T.Kernel(work_blocks)`**：直接超发会由运行时串行调度多波次、每次核启动开销重复；核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正（`core-split-strategy.md` §1 要素③）。**manifest 的残留不均仅 +0.6%**（远优于 1d 函数的 +9.4%），因 `block_tokens=1` 使 `work_blocks=batch*seq_len=4096`（比 1d 的 256 更接近 40 的倍数）。
- **模式先例**：`examples/elementwise/vec_add_2d.py`（`T.Kernel(BLOCK_SIZE=20, is_npu=True)` + `T.serial` + guard）；`examples/posembedding/rope_llama31/kernels/blocked_low_precision.py`（`T.Kernel(programs, is_npu=True)` + `T.serial(iterations)` + `if work < work_blocks`）；`PL-1.22-persistent-1d-grid-stride`（**stale**，模板形态作背景，核数取值以实查为准）。

> 备选（供 Stage 4 参考，不在本设计实现）：因工厂 `lru_cache` 形状特化，可按 shape 分派不同 `block_tokens`（如小 shape 用大 `block_tokens` 提升核占用、大 shape 用大 `block_tokens` 摊薄发射），以整组 workload 最优；按 `core-split-strategy.md` §2.4 属分核参数调优。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素/配对/坐标计算已按 §1.6.2 全部向量化或消除——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(40, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(work_blocks, 40))` | 每核串行处理 `ceildiv(work_blocks,40)` 个逻辑 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化（3D `(bt,H,hp)` 整 buffer + size-1 头轴广播） | `T.vmul`/`T.vadd`/`T.vsub` + `T.vcast` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：本设计以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`vec_add_2d.py`/`rope_llama31`/`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(40, is_npu=True)` + `for i in T.serial(iters)` + `if w < work_blocks`，循环内为 6-op（fp32）或 12-op（sub-fp32）半段 3D 向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in ×4 → [vcast↑ ×4] → 6-op 链 → [vcast↓ ×2] → copy-out ×2），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。设计阶段保持简洁；若 Stage 4 需进一步优化（persistent double buffer / 更大 `block_tokens`），参照同族 `T.serial` 多 tile 形态（`rope_llama31` 的 `iterations` 循环）。

### 6.4 尾块处理

见 §5.4。`seq_len % block_tokens != 0` 时最后一个 tile 用 `rows = T.min(block_tokens, seq_len - seq_start)` 动态裁剪 `T.copy` 的行 extent；manifest 2d workload（`seq_len=2048`, `block_tokens=1`）均整除（无尾块），尾守卫面向通用 `seq_len` 保留（D-3 探针 `(2,130,4,64)` 已验证）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB: xlo,xhi,c,s) → [vcast↑ ×4] → vmul×2 → vsub → vmul×2 → vadd → [vcast↓ ×2] → T.copy(UB→GM: y 两半)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 6-op 链内部（同 buffer 原地复用） | 自动 | 线性数据依赖（`tmp` 复用、`olo`/`Bv` 原地），插 V→V barrier |
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

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓参考实现 `tests/ops/test_rope.py::ref_rope_neox` 的 4D 分支，fp32 opmath + 单次舍回），**不得复刻 §0.6 的 NPU 半段 3D 广播链结构**——保证验证独立性。

```python
import torch


def golden_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE 参考实现（= 源仓 tests/ops/test_rope.py::ref_rope_neox，契约 golden）。

    x: (batch, seq_len, num_heads, head_dim);  cos/sin: (seq_len, head_dim//2).
    语义（test 基准，保持 fp32 opmath + 末端单次舍回）：
        cos_full = cat([cos, cos], -1).unsqueeze(0).unsqueeze(2)
        sin_full = cat([sin, sin], -1).unsqueeze(0).unsqueeze(2)
        rot = cat([-x[..., half:], x[..., :half]], -1)
        y = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)
    独立于 NPU 结构（直接按定义式实现，不复用 §3.3 的半段 op 序列）。
    """
    half = x.shape[-1] // 2
    xf = x.float()
    rot = torch.cat([-xf[..., half:], xf[..., :half]], dim=-1)
    cos_full = torch.cat([cos.float(), cos.float()], dim=-1).unsqueeze(0).unsqueeze(2)
    sin_full = torch.cat([sin.float(), sin.float()], dim=-1).unsqueeze(0).unsqueeze(2)
    return (xf * cos_full + rot * sin_full).to(x.dtype)
```

> 备注：golden **独立于** NPU 向量链（按 §0.1 定义式直接实现，不复用半段结构/头轴广播），满足验证独立性。cos/sin 由 Op 层生成（`tileops/ops/position_encoding/rope_neox.py::_base_freqs`，fp32 计算后 `.to(dtype)`），与 kernel 输入同 dtype。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`tests/ops/test_rope.py::_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对**：`verify_equiv.py` 实测半段 3D 广播链 vs 契约 golden **三 dtype × 三 shape × {randn, corners} `max_abs=0.0`（逐位一致）**，远优于门禁；输入 dtype 域诊断（被拒 E1）fp16 `max_abs=3.9e-3`（atol 违反比 ~1.2%）、bf16 `3.1e-2`（佐证 fp32 中转必要性）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含 manifest 2d workload、源仓 fixture 各 dtype、角点值、非整除/极小 seq_len 尾块、head/batch 广播正确性**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp16 manifest 2d（randn） | `x=randn([2,2048,32,128], fp16)`，cos/sin `(2048,64)` | 与 `golden_rope_neox` 比对 ≤1e-3 | 门槛通过 |
| L0-2 fp32 basic（randn） | `x=randn([2,128,8,64], fp32)`，cos/sin `(128,32)` | 比对 ≤1e-5 | 门槛通过 |
| L0-3 fp16 basic（randn） | `x=randn([2,128,8,64], fp16)` | 比对 ≤1e-3 | 门槛通过 |
| L0-4 bf16 basic（randn） | `x=randn([2,128,8,64], bf16)` | 比对 ≤1.6e-2 | 门槛通过 |
| L0-5 fp16 `[1,256,4,128]` | `x=randn([1,256,4,128], fp16)` | 比对 ≤1e-3 | 门槛通过 |
| L0-6 极小 seq_len（edge fixture） | `x=randn([1,1,1,16], fp32)`（`half=8`→`half_pad=16`，`block_tokens=64`） | 比对 ≤1e-5 | 门槛通过 |
| L0-7 角点值（±inf/NaN/±0/极值） | `x` 与 cos/sin 填 `[0,-0,inf,-inf,nan,±1e-45,±65504,±88.8,±1e-8]`（各 dtype） | 与 golden 比对（`equal_nan=True`；±inf 经 IEEE mul/add 传播） | 门槛通过 |
| L0-8 非整除 seq_len 尾块 | `x=randn([2,130,4,64], bf16)`（`block_tokens=8`，130=16×8+2 尾块 2 行） | 比对 ≤1.6e-2 | 门槛通过 |
| L0-9 head/batch 广播正确性 | `x=randn([2,128,8,64], fp16)`；用**同一 seq 位置不同 head/batch 的 cos/sin 一致性**核对（golden 4D 广播 vs kernel） | 逐位一致 | 门槛通过 |
| L0-10 cos/sin 单载复用等价 | L0-3 输出 vs「逐元素重取 cos/sin」的手工参考 | 逐位一致 | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用（wrapper 路径：`x.contiguous().reshape(-1)` 传入，输出 `reshape` 回 4D），`torch.testing.assert_close(y.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，48/48 `EQUIV_PASS`，§1.6.1）；`history_version/design_probe_npu_2d.py`（D-3 设备探针，5 shape 逐位一致 `max_abs=0.0`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 flat 入参 + 3D 声明视图假设** | wrapper 传 flat `(N_total,)`，kernel 声明 `(tokens,H,d)`——依赖 launcher 指针式（不校验 shape）与行主逐字节一致 | NPU launcher（`JitKernel_NPU.__call__`）仅校验实参个数、按 `data_ptr()` 传递，**不校验输入 shape**；`(tokens,H,d)` 行主布局与 flat 逐字节一致。**D-3 探针 5 shape 设备实证通过**（§3.3）；Stage 3 首个 L0 即复验 |
| **R-2 半段结构 = 结构重排（非直译）** | 若误按源逐元素 gather/三级 div-mod 直译，NPU 上退化为标量/低效（VP-2026-0175 实测 39 ms 级） | §0.6 R2 半段结构（零 gather）+ R1 3D 视图（零逐元素 div/mod）；`verify_equiv.py` #0/#1 证逐位等价 |
| **R-3 bf16 v-op 文档/设备分歧** | `docs/.../T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 记 bf16 ✗；同任务 1d 的 Stage 3 探针实证 Ascend910_9362 原生 bf16 v-op 可用（`repro/TRAP-bf16-vop-doc-matrix-divergence.py`） | 本算子**不依赖**原生 bf16 路径（统一 fp32 中转），故不阻塞；分歧登记，**文档为准**（AGENTS.md） |
| **R-4 未文档化广播形态性能税**（`PL-1.15-broadcast-perf-tax`） | size-1 头轴广播 `(bt,H,hp) op (bt,1,hp)` 属未文档化形态（同 class 列广播实测 +12~20% 性能税）——数值可用但可能非最优 | 该形态有 `rope_llama31` 先例 + D-3 探针数值实证；**性能税待 Stage 4 实验裁决**（§1.6.3：L2 备选可改 `T.vbrc` 展开 + 同 shape 运算规避） |
| **R-5 发射项（0.5µs/op）名义主导** | §1.6.0 估算示发射项名义主导（618 µs）；但吞吐口径（`CONST-vector-op-cost-910_9362`）显示流量主导（55 µs） | 标注「未文档化假设 + 估算依据」（常数 stale、可能上偏）；`block_tokens` 为杠杆（§5.2）；探针实测 host-launch 掩盖 kernel 时延 → Stage 4 baseline（msprof）对账 |
| **R-6 UB 容量 / `bt` 放大** | `bt=3` sub-fp32 超限（×1.7 ≈ 208 KB） | 默认 K11 派生值（最大 48 KB×1.7 ≈ 82 KB）；`bt` 放大须编译探针校准（§4.5） |
| **R-7 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3）；`af~sf` 在 fp32 路径仍分配 |
| **R-8 v-op 整 buffer vs 动态 slice** | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:rows, 0:H, 0:half]`（§3.3）；`TRAP-expert-v-operands` 仅静态边界 |
| **R-9 `vcast` round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32`/`bf16→f32` **仅 `rint`** | 升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-10 `block_tokens` 兜底** | `H*d > block_size` 时 `block_size//(H*d)=0` → 空 tile | `block_tokens = max(1, ...)`（编译期，§5.2）；tile 轴不依赖 M，规避 `TRAP-ub-blockm-gt-M-overflow` |
| **R-11 分核残留不均** | 物理核 40 = 2³×5（含 5），`work_blocks=2ᵐ` 无法整除 40 | §5.5 网格跨步核内串行；manifest 残留仅 **+0.6%**；basic `+25.0%`、小 shape 空转，均显式记录 |
| **R-12 常数表设备/版本（stale）** | 引用的 `CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-mte2-degradation`/`PL-1.6`/`PL-1.22` 等被 `kb_stale_check` 标 `[stale]`；本设备 arch 实查 `Ascend910B3` | 性能数字标注「待重验/背景」，不作决策依据；流量项优先用 `CONST-ascend910-9362-profile`（verified，但亦 stale）；**`CONST-vector-op-cost-910_9362`（verified）为吞吐交叉核对主口径**；Stage 4 baseline 后对账回填 |
| **R-13 arch 字符串 vs constants 绑定** | 实查 `get_arch()` 返回 `Ascend910B3`（20 AIC/40 AIV），而 constants 表条目绑 `910B2C`（24 核）/`910_9362`（20/40） | **核数以实查 20/40 为准**（与 9362 拓扑一致）；带宽/发射常数按「同量级背景 + stale 待重验」使用；登记待校准（`CONST-ascend910-9362-profile` 的 repro 亦为 `repro-missing`） |
| **R-14 尾块/`half_pad` 陈旧数据** | 尾行（`[rows:bt)`）与 pad 列（`[half:hp)`）含陈旧 UB 数据参与 v-op | 陈旧结果不写出（仅 `[0:rows, 0:H, 0:half]` 拷出）；`PL-1.22` 尾块零起点切片纪律；D-3 探针 `(2,130,4,64)` 与 `(1,1,1,16)` 实证 |
| **R-15 探针 host-launch 掩盖 kernel 时延** | D-3 探针实测 tiny 与 manifest shape 同量级（310–330 µs），无法分离 kernel 时延 | 探针结论只用于**数值正确性**与契约/API 事实；**时延对账交 Stage 4 msprof op baseline**（§1.6.0 D-2 声明） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接传入数学 op | 未做 vcast 上转 | 按 doc 编译失败（R-3 分歧待核） | sub-fp32 先 `vcast` 到 fp32（R-3/R-9） |
| vcast 用非 rint 模式升级 | `f16→f32` 用 `round`/`floor` | 编译失败（仅 `rint`） | 升降统一 `round_mode="rint"`（R-9） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-7） |
| v-op 传动态 slice | `T.vmul(xlo_ub[0:rows], ...)` | 编译/运行错误 | v-op 用整 buffer（R-8） |
| 广播轴 rank 不一致 | cos/sin 用 2D `(bt,hp)` 与 data 3D `(bt,H,hp)` 运算 | rank 断言失败 | cos/sin 用 3D `(bt,1,hp)`（同 rank）（§3.2） |
| `block_tokens` 未兜底 | `H*d > block_size` | `block_tokens=0` → 空 tile/除零 | `max(1, ...)`（R-10） |
| UB 溢出 | `block_tokens` 过大 | 编译失败 | 维持 K11 派生值；§4.5 预算 + 编译探针（R-6） |
| 尾块越界 | `seq_len` 非 `block_tokens` 整数倍且无 `rows` 裁剪 | 越界读/写 GM | `rows = T.min(block_tokens, seq_len-seq_start)`（§5.4） |
| 行尾切片 v-op | 对 `[0:rows]` 行切片调 v-op | 编译/语义错误 | 行切片只在 `T.copy`（R-8） |
| tiling 派生写在工厂外层 | 用工厂默认 `block_size` 派生 `block_tokens` | wrapper 的 dtype 相关 `block_size`（fp32=1024）被静默忽略 | tiling 全写在 `kernel(blk)` 内层（§3.3 K11 铁律） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 `seq_len`** | Kernel 内 `rows` 行尾动态裁剪（§5.4），无需 host padding |
| **极小 `seq_len`（< block_tokens）** | `block_tokens = max(1, ...)`（编译期），`rows=seq_len`，仅 `task` 覆盖的核工作 |
| **极大 `seq_len`/batch** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`H*d > block_size`（manifest）** | `block_tokens` 兜底为 1（整 head 入 tile）（R-10/§5.2） |
| **`half` 非 32B 整倍（d<32）** | `half_pad = ceil(half/block_elems)*block_elems`（32B 对齐），pad 列结果丢弃（R-14）；D-3 探针 d=8/16 实证 |
| **`±inf` / `NaN` 输入** | 经 4 mul + 2 add/sub IEEE 传播，与 golden 运算集合一致（`inf*0=NaN` 一致）；`equal_nan=True`（角点机器验证 `max_abs=0.0`） |
| **`x=0` / `-0.0`** | `y1 = −b*s`、`y2 = a*s`，恒等；`±0` 符号经乘法保留 |
| **`head_dim` 奇数** | manifest `shape_rules: x.shape[-1] % 2 == 0` 排除；wrapper `head_dim % 2 != 0` 抛错 |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`x` 非连续** | wrapper `forward` 先 `x.contiguous()`（Op 层已 `_validate_and_prepare` 保证）——host 侧元数据视图，合规 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/rope_neox/_make_rope_neox_2d/
├── DESIGN.md            # 本文件（Stage 1 产出）
├── verify_equiv.py      # D-1 等价性机器验证（48/48 EQUIV_PASS）
├── RETROSPECTIVE.md     # Stage 1 复盘（skill Phase 8）
├── history_version/
│   └── design_probe_npu_2d.py   # D-3 设备探针（5 shape 逐位一致）
└── _make_rope_neox_2d.py        # NPU kernel（下游 Stage 3 产出，尚未创建）
```

### 10.2 文件清单

| 交付物 | 路径 | 状态 | 说明 |
|--------|------|------|------|
| 设计文档 | `examples/rope_neox/_make_rope_neox_2d/DESIGN.md` | 已完成 | 本文件 |
| 等价性机器验证 | `examples/rope_neox/_make_rope_neox_2d/verify_equiv.py` | 已完成 | D-1：48/48 `EQUIV_PASS`（§1.6.1） |
| 设计期设备探针 | `examples/rope_neox/_make_rope_neox_2d/history_version/design_probe_npu_2d.py` | 已完成 | D-3：5 shape × 三 dtype 逐位一致 `max_abs=0.0` |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_rope_neox_2d.py` | 待实现 | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_rope_neox`） | 待实现 | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py` | 只读 | `block_size`（K11）+ `main(x,cos,sin,y)` 契约 + 2d flat 入参 + `ascend_mode="Developer"` |
| 抽取脚手架（已存在，只读参照） | `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py` | 只读 | Pattern A 源直译参照（`_make_rope_neox_2d` L65–L110） |
| 源算子 / 测试（只读参照，源真相） | `/home/developer/TileOPs/tileops/kernels/rope.py`、`/home/developer/TileOPs/tests/ops/test_rope.py` | 只读 | 语义与 golden 依据 |
| manifest（只读参照） | `examples/TileOPs/tileops/manifest/position_encoding.yaml` | 只读 | `RopeNeoxOp` 2d workload + roofline |

### 10.3 命名规范

- 工厂：`_make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype, block_size=2048)`（对齐 wrapper）。
- kernel 内 buffer：`xlo_ub/xhi_ub/c_ub/s_ub`（输入两半 + cos/sin）、`af/bf/cf/sf/olo/tmp`（fp32 计算域）。

### 10.4 实现顺序

1. 实现 kernel（§3.3）：3D UB tile 半段向量链（头轴广播）+ 一维核网格跨步 + `rows` 行尾守卫；
2. 写 `golden_rope_neox`（§8.1，4D 契约）；
3. 跑 L0-1..L0-10（§8.3）；
4. `verify_equiv.py` 复核（§1.6.1）。

**实现要点速览（Stage 3 输入）**：
1. **坐标 = 3D 静态视图 + per-tile `(b, seq_block)` 映射**：`x`/`y` 声明 `(tokens,H,d)`（接收 flat 入参）；`b=w//seq_blocks`、`sb=w%seq_blocks`、`seq_start=sb*block_tokens`；**无逐元素 div/mod**；
2. **计算核 = 半段 3D 向量链**：`y1 = vsub(vmul(a,c), vmul(b,s))`、`y2 = vadd(vmul(b,c), vmul(a,s))`（6 op，作用于 `(bt,H,hp)` 整 buffer，**零 gather**）；
3. **头轴广播**：cos/sin 为 `(bt,1,hp)`，靠 v-op 同 rank size-1 轴自动广播（`rope_llama31` 先例 + D-3 实证）；**单载复用全 head**；
4. **计算域 = fp32 中转**（sub-fp32 先 `vcast(rint)` 升 fp32，链尾 `vcast(rint)` 舍回；fp32 直算）；**bf16 必走 fp32**；
5. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `rows` 行尾守卫；`block_tokens = max(1, block_size//(H*d))`（**在 `kernel(blk)` 内层派生**，K11 铁律）；
6. `half_pad` 32B 对齐（`half` 非整倍时 pad，pad 列结果丢弃）；
7. buffer 顶层无条件分配；v-op 作用于整 buffer（行切片只在 `T.copy`）；`xlo_ub/xhi_ub` 死亡后复用为输出、`Bv` 原地承载 y2；
8. `main(x, cos_table, sin_table, y)`（`out_idx=[3]`，**无额外 shape 参数**）。
