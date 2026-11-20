# _gemv_kernel 算子设计文档

> 迁移任务（harness 模式，project=`gemm`，op=`_gemv_kernel`，编程模式 Developer）。
> 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py` 中的 `_gemv_kernel`（GPU SM90 GEMV fast path）。
> 本设计只覆盖 `_gemv_kernel`；`_gemm_kernel`（稠密 GEMM）另见 `examples/gemm/_gemm_kernel/DESIGN.md`。
> 本章结论驱动 §1–§11 全部设计决策。

## 设计核心（GPU→NPU 关键重设计，2–4 条）

1. **算法路线：Cube 分形路径 → Vector 归约路径**。GEMV 是访存受限算子（算术强度 ≈1 FLOP/Byte），源算法用
   thread 级 warp-shuffle 树归约。NPU 侧**不用 Cube**：GEMV 的 GEMM 形态必有一个维度为 1，Cube 分形（16 对齐）
   使 MAC 阵列 15/16 空转、`mmadL1` 尺寸 `[BM,BK,1]` 效率极低。**设计期实测**（probe #1，Developer 模式，
   Ascend910B2C，7168×7168 fp16）：Vector 归约 **0.188 ms** vs Cube `(M,K)×(K,1)` **1.484 ms**（7.9×）
   vs Cube `(1,K)×(K,M)+b_transpose` **0.304 ms**（1.6×）——Vector 归约全面胜出。
2. **并行结构：二维线程网格 + warp shuffle → 一维 block 网格 + 块内 `T.Pipelined` 软流水**。源
   `T.Kernel(ceildiv(n,block_n), threads=(reduce_threads,block_n))`（每行 32 线程 + `tvm_thread_allreduce`）
   → NPU `T.Kernel(ceildiv(M, block_M), is_npu=True)` 行块分核，块内沿 K 用 `T.Pipelined` 重叠 MTE2 与 Vector。
3. **归约：thread-level allreduce → 向量水平归约**。`T.reduce_sum(dst, dim=-1, clear=False)` 沿 K（W 的最内
   连续轴 = 归约快路径）做向量归约；x 用文档化形态 `T.vbrc((1,BK)→(BM,BK))` 广播、fp32 域 `T.vmul` 乘加。
4. **精度与边界：fp32 累加 + 单次舍入 + 双侧 K 尾清零**。源 `accum_dtype="float"` → 全程 fp32 乘/加，输出
   `T.copy` 单次 VCast 舍入；M 尾块动态 extent（`T.min`）；K 尾**x 与 W 双侧清零**（杜绝 `0×NaN`），免 host padding。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

- **数学语义**：GEMV（矩阵 × 向量），核心式为
  `y[r] = Σ_{k=0}^{K-1} W[r,k] · x[k]`，`W:[R,K]`（行主序，K 为最内连续轴），`x:[K]`，`y:[R]`。
  **命名映射（务必对齐）**：源 kernel 工厂签名为 `_gemv_kernel(n, k, dtype)`，其张量为
  `a:(k,)`（向量）、`b:(n,k)`（矩阵）、`c:(n,)`（输出）；即源的 `n` = 本设计的输出行数 `R`（= GEMM 逻辑 `M`，
  因 GEMV 是 `N=1` 特例）。迁移字段 `source_output_shape=(m,)` 对应 `(R,)`。本设计统一使用 `R`（输出行数）
  与 `K`（收缩维），并在 §8 提供等价命名说明。
- **GEMM 层调用契约**：`GemmOp._get_kernel` 在两处派发到 GEMV——
  `lhs_row`（`m==1 and not trans_a and trans_b`：`A=[1,K]` 行向量、`B=[N,K]`，输出 `(1,N)`）与
  `rhs_col`（`n==1 and not trans_a and not trans_b`：`A=[M,K]`、`B=[K,1]`，输出 `(M,1)`）。两者归约同一核心
  `y[r]=Σ_k W[r,k]·x[k]`：`lhs_row` 的矩阵是 `B`（`R=N`）、向量是 `a`；`rhs_col` 的矩阵是 `A`（`R=M`）、
  向量是 `b`。`GemmOp.forward` 对 `lhs_row` 调 `kernel(a.reshape(-1), b).reshape(1,n)`，对 `rhs_col` 调
  `kernel(b.reshape(-1), a).reshape(m,1)`。
- **I/O 契约**：`W:(R,K)` ND 行主序连续；`x:(K,)` 一维；`y:(R,)` 一维；dtype ∈ fp16/bf16（`a`/`b` 同 dtype），
  输出 dtype 与输入相同。输出 shape `(R,)` 即迁移必需字段 `source_output_shape=(m,)`；无转置输出。
- **规约语义**：沿 K 维求和。源码按 `block_k` 分块，**每个线程**对其 `tile_k` 个元素做顺序 FMA 累加，再用
  `tvm_thread_allreduce`（`comm_reducer(sum)`，跨 `threadIdx.x`）做树形归约得到行和。累加分组 =
  「线程内 tile_k 顺序 + 32 线程树」，与「一次扁平求和」不同，但同为 fp32 域，误差在 fp16/bf16 容差内。
- **dtype 语义**：输入 fp16/bf16，**累加必须 fp32**（源 `accum_dtype="float"`；源码显式
  `a_local[_k].astype(float) * b_shared[...].astype(float)`，即 **fp32 乘法 + fp32 累加**）；输出在 fp32 域
  单次舍入到输入 dtype（`c[...] = c_reduced[0]`，`c` 为 dtype）。NPU 侧同样要求 fp32 域（§1.6.1 E3）。
- **边界语义**：
  - nan/inf 经乘加自然传播，无掩码。
  - 非整除：`GemvBoundaryFixture` 覆盖 `R∈{3000,3001,1024}`、`K∈{1024,3000}`（`block_n=8`/`block_k=256` 的
    非整除形状）；源无 host padding，边界由 `T.ceildiv` + 循环守卫隐式处理。
  - 空维度不出现（R≥1、K≥1 用例）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（`y = W·x`，fp32 累加），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a`（源命名；向量 `x`） | `(K,)` | float16 / bfloat16 | 一维激活向量；`GemmOp.forward` 传入 `a.reshape(-1)` |
| `b`（源命名；矩阵 `W`） | `(R,K)` | same_as(a) | 行主序，K 最内连续（`lhs_row` 下即 GEMM 的 `B=[N,K]`，`R=N`） |
| `c`（输出；`y`） | `(R,)` | same_as(a) | 一维输出，ND 连续，无 transpose |

清单依据：源 `_gemv_kernel(n,k,dtype)` 签名（`/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py` 行 585–656）；
NPU 规格 `examples/TileOPs/tileops/manifest/gemm.yaml` 的 `GemmOp`（`signature`/`shape_rules`，`trans_a=false`、
`trans_b=true` 默认）；dispatch 契约 `/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py` 行 100–154。

### 0.3 实现算法解读（怎么算）

源码 `_gemv_kernel(n, k, dtype)` 为工厂函数，返回 `_gemv_func`；调用
`_gemv_func(block_n=8, reduce_threads=32, num_stages=2)` 得到编译后的 `_gemv_main`。内部推导
`tile_k = 128 // (itemsize*8)`（fp16 时 = 8，即 128-bit 事务宽度）、`block_k = reduce_threads * tile_k = 256`。

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | Grid 映射：`T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n))` → `bn`=行块；`tk=threadIdx.x`（war 内快变化）、`tn=threadIdx.y`（行） | n,block_n,reduce_threads | 每块行区间 `bn*block_n + tn` | 输出行分块 |
| 2 | 分配 `c_accum(1,)` fp32、`b_shared(block_n, block_k)`、`a_local(tile_k,)`；`T.clear(c_accum)` | — | 片上缓冲 + 累加器 | 累加器初始化 |
| 3 | K 流水：`for bk in T.Pipelined(ceildiv(k, block_k), num_stages)` | — | — | K 维分块 |
| 4 | `T.copy(b[bn*block_n, bk*block_k], b_shared, disable_tma=True)`（cp.async 装入下一 tile） | GM `b` | `b_shared` | 数据搬入 |
| 5 | `for _k in T.vectorized(tile_k): a_local[_k] = a[bk*block_k + tk*tile_k + _k]` | GM `a` | `a_local`(register) | 向量搬入（a 很小） |
| 6 | `for _k in T.serial(tile_k): c_accum[0] += a_local[_k].astype(float) * b_shared[tn, tk*tile_k+_k].astype(float)` | `a_local`,`b_shared` | 更新 `c_accum` | `Σ_k`（本线程的部分和） |
| 7 | `tvm_thread_allreduce`（`comm_reducer(sum)`；`reduce_scope`）= warp shuffle 树归约跨 `tk`(32 线程) | `c_accum[0]` | `c_reduced[0]` | 行和（跨线程合并部分和） |
| 8 | `c[bn*block_n + tn] = c_reduced[0]`（fp32→dtype 隐式单次舍入） | `c_reduced` | GM `c` | 输出写回 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[b] --cp.async(disable_tma=True)--> SMEM b_shared(block_n, block_k) --\  (2 级流水环，T.Pipelined)
                                                                          >-- FMA(fp32) --> register c_accum
GM[a] --T.vectorized(128-bit)-------> register a_local(tile_k) ----------/                     │
                                                                                               ▼
                                        warp shuffle tree reduce (tvm_thread_allreduce, reduce_threads=32)
                                                                                               │
                                                                                               ▼
                                              fp32 -> dtype 单次舍入 --> GM[c]（c[bn*block_n+tn]）
   行映射：tn=threadIdx.y 一行/线程；tk=threadIdx.x 为该行的 32 路归约 lane；coalesced：同 warp 的 tk
   连续访问 b_shared[tn, tk*tile_k..+8] 的 stride-1 列 → 128-bit 合并装载
```

**循环与并行结构**：

- 两级并行：grid（`ceildiv(n, block_n)` 行块，`block_n=8`）+ 块内 `(reduce_threads=32, block_n=8)` = 256 线程二维网格。
- 块内 K 迭代：`for bk in T.Pipelined(ceildiv(k, block_k=256), num_stages=2)`，双缓冲隐藏 HBM 延迟。
- 归约：每线程持有 `tile_k=8` 个元素的顺序 FMA 部分和，32 线程经 warp shuffle 树归约。
- `threads=(reduce_threads, block_n)` 是把「行」与「归约 lane」编码进 CUDA 线程块的二维组织。

**host 侧逻辑**：

- `GemvKernel.__init__` 调 `_gemv_kernel(n, k, dtype_str)` 建 kernel；`default_config` 依 `get_sm_version()`
  选 `{block_n:8, reduce_threads:32, num_stages:2}`（SM90 档）或 `{block_n:32, reduce_threads:32, num_stages:1}`；
  `autotune_configs` 扫 `block_n∈{1,2,4,8,16}`、`num_stages∈{1,2,3}`。
- `GemvKernel.forward(a, b)`：`a = a.flatten().contiguous()`（**host 张量视图/连续性归一，非数值改写**），
  然后 `self.kernel(block_n, reduce_threads, num_stages)(a, b)` 直接调用编译产物。
- `GemmOp.forward` 依 `_active` 模式 reshape 输入/输出（`a.reshape(-1)`、`.reshape(1,n)` / `.reshape(m,1)`）。
- 无 im2col / crop / transpose 等数值预处理或后处理。
- `_gemv_wrapped_kernel`（`torch.library.custom_op`，GPU 侧 `top::` 命名空间）仅用于 `torch.compile` 兼容，
  不在 eager 路径。

**编译配置**：`@tilelang.jit(out_idx=[-1], compile_flags=["-O3","-DENABLE_BF16"])`；
`functools.lru_cache(maxsize=32)` 缓存工厂。无 `pass_configs`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | Coalesced 128-bit 访问（`threads=(reduce_threads, block_n)`，tk 快变化） | 提高 B 装载带宽 | 同 warp 相邻线程访问 stride-1 连续列 | memory coalescing / 128-bit 事务 | 硬件强相关（NPU 搬运自动向量化，组织方式不同） |
| 2 | SMEM staging（`b_shared`） | 装载与计算解耦、支持流水 | B tile 经 cp.async 驻留 SMEM | shared memory | 硬件强相关（NPU 有 UB 等价） |
| 3 | 多级流水（`T.Pipelined`，`num_stages=2`） | 隐藏 HBM 延迟 | 双缓冲交替预取 | cp.async 异步引擎 | 模式可移植（NPU `T.Pipelined`） |
| 4 | 寄存器累加（`c_accum` 在 `alloc_local`） | 低延迟部分和 | 每线程 tile_k 个 FMA | 寄存器堆 | 等价替换（NPU 用 UB fragment） |
| 5 | warp shuffle 树归约（`tvm_thread_allreduce`） | 跨 32 线程合并部分和 | `reduce_scope` + `comm_reducer(sum)` | warp（32 线程）+ shuffle | **硬件强相关（NPU 无 warp）→ 重新设计** |
| 6 | 向量化装载 `a`（`T.vectorized(tile_k)`） | 128-bit 读 x | 连续 tensorize | 向量化访存 | 模式可移植（NPU 向量装载） |
| 7 | `disable_tma=True`（用 cp.async 而非 TMA） | 规避 TileLang 无法为手工索引 `b_shared` 推断 mbarrier | CUDA TMA vs cp.async | Hopper TMA / cp.async | 硬件强相关（NPU 无 TMA） |
| 8 | 二维 grid/thread 任务映射（行 × 归约 lane） | 两级并行划分 | `threadIdx.x/y` 编码 | block×thread 两级 | 硬件强相关（NPU 一维 Kernel）→ 重新设计 |
| 9 | `tile_k` 由 128-bit 事务宽度推导 | 保证 16B 对齐访问 | `128 // (itemsize*8)` | 128-bit 对齐 | 模式可移植（NPU 尾轴 32B / 向量宽度） |
| 10 | 工厂 `lru_cache` + 每 shape 编译 + `get_sm_version` 启发式 | 降 host 开销、shape 特化 | 缓存编译产物 | — | 部分可移植（`get_sm_version` 无 NPU 等价） |
| 11 | host `a.flatten().contiguous()` | 归一输入视图 | 视图/连续性操作 | — | 可移植（只改 stride/shape，不改数值） |

> 识别不出 ≠ 不存在：以上 11 项为源码中可识别的优化手段，逐项在 §0.5 判定处置，避免静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据映射表：tilelang-op-design skill `references/migration-analysis.md` §5.3；NPU 能力佐证：
`docs/开发指南.md` §2/§3、`docs/Tilelang.language/规约操作/T.reduce_sum.md`、`docs/Tilelang.language/内存操作/T.copy.md`、
`docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/shape操作/T.vbrc.md`、`examples/gemv/example_gemv.py`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义（`y=W·x`、fp32 累加、单次舍入、nan/inf 传播） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| Coalesced 128-bit 线程布局（#1） | 优化 | warp 线程级合并访存 | 有（NPU 搬运自动向量化） | **等价替换** | 退化为行块 `(block_M, block_K)` 连续装载（`T.copy`，MTE2 自动按 32B/向量宽度组织）；组织方式不同但带宽意图不变 |
| SMEM staging（#2） | 优化 | shared memory | 有（UB 192KB） | **等价替换** | Developer 模式 `T.alloc_shared`/`T.alloc_fragment` → UB；`docs/Tilelang.language/内存操作/T.alloc_shared.md` |
| 多级流水（#3） | 优化 | cp.async 异步引擎 | 有（MTE2） | **等价替换** | `T.Pipelined(k_iters, num_stages=S)`，stage 按 UB 容量重算；`docs/开发指南.md` §3.2 |
| 寄存器累加（#4） | 优化 | 大寄存器堆 | 有（UB fragment） | **等价替换** | `T.alloc_fragment((block_M,1),"float32")` 承载 fp32 行和 |
| warp shuffle 树归约（#5） | 优化 | warp（32 线程）+ shuffle | **无 warp 概念** | **重新设计** | → §0.6 R1：`T.reduce_sum(..., dim=-1, clear=False)` 向量水平归约 |
| 向量化装载 `a`（#6） | 优化 | 128-bit 向量访问 | 有 | **等价替换** | `T.copy`/`T.vbrc` 承载 x 的装载与广播 |
| `disable_tma=True`（#7） | 优化 | Hopper TMA | **无** | **舍弃** | NPU 无 TMA/cp.async 选择项，MTE2 由框架管理 |
| 二维 grid×thread 映射（#8） | 优化 | block×thread 两级并行 | **无**（`T.Kernel` 仅一维；`threads=` 在 npuir 无效） | **重新设计** | → §0.6 R2：一维 `T.Kernel(ceildiv(M,block_M))` + 行块分核 |
| `tile_k∝128-bit`（#9） | 优化 | 128-bit 事务 | 有（32B 尾轴 / 向量宽度） | **等价替换** | `block_K` 取向量宽度整数倍（fp16 ×8 / fp32 ×4），本设计取 256 |
| 工厂缓存 / shape 特化（#10） | 优化（host） | 无 | 有 | **保留** | 保留 `lru_cache` + 每 shape 编译；`get_sm_version` 分支去除（NPU 无 SM 版本） |
| host `flatten/contiguous`（#11） | 优化（host） | 无 | 有 | **保留** | 仅 stride/shape 视图操作，不改数值（`references/ascend-constraints.md` §4 允许） |
| `autotune_configs` / `tune`（K8） | 优化（host） | 无 | 有 | **舍弃** | NPU 侧 wrapper 已按 K8 移除 autotune，改为启发式 config 选择 |
| `_Trace`/`str2dtype` 替身（Stage 0 抽取件 helper） | helper | GPU 专有 | 无 | **保留（仅作参考）** | `_gemv_kernel` 内用到 `str2dtype` 推导 `tile_k`；NPU 重写用编译期常数替代，不依赖该 helper |
| 物理核数 / `get_sm_version()`（op 层）/ `supported_archs=[90]` | 优化（host） | CUDA SM | 无 | **舍弃** | wrapper `kernels/gemm/gemm/gemm.py` 内已改 `supported_archs=None` |

> 判定统计：**保留 4 项**（语义、#10、#11、helper 参考）、**等价替换 6 项**（#1/#2/#3/#4/#6/#9）、
> **重新设计 2 项**（#5 归约、#8 并行结构；另见 §0.6 R3 算法路线、R4 非整除）、**舍弃 4 项**（#7、autotune、SM 相关 + `_Trace`）。
> 所有舍弃项均写明理由，未静默丢弃。

### 0.6 NPU 算法重设计

重设计对照 Phase R 调研结论（§1.6.0）：调研选定 **Vector 归约路径**（R4 实测淘汰 Cube n=1 形态），故 §0.6 的
重设计承接该选择；源算法已识别的优化手段（§0.4）承接情况见各重设计项与 §1.6.0 末尾。

**重设计项 R1：warp shuffle 树归约 → `T.reduce_sum` 向量水平归约**

- **源方案**：每线程对 `tile_k=8` 个元素顺序 FMA 得部分和，再用 `tvm_thread_allreduce`（`reduce_scope` +
  `comm_reducer(sum)`）跨 `reduce_threads=32` 做 warp shuffle 树归约。性能意图 = 在 warp 内以低延迟通信合并
  部分和，避免跨 SMEM 往返。
- **NPU 新算法**：用 `T.reduce_sum(Wf (block_M, block_K), acc (block_M, 1), dim=-1, clear=False)` 对 **K 维（W 的
  最内连续轴）** 做一次向量水平归约，跨 k-chunk 用 `clear=False` 在 `acc` 上累加（`acc` 循环前 `T.clear`）。
  NPU 的并行层级是「block × 向量 lane」，函数级归约 API 即 `hivm::VReduceOp`（`T.reduce_sum.md` §3）。
- **语义保持论证**：归约结果 = 同一组数的和，仅**加法分组**由「线程内顺序 + 32 线程树」变为「chunk 内向量树 +
  chunk 间顺序」，同为 fp32 域求和；分组差异对 fp16/bf16 输出的影响经 §1.6.1 E2 机器验证（容差内等价）。边界
  （nan/inf 传播）由 `Σ` 的 IEEE 语义保持不变。

**重设计项 R2：二维 grid×thread 映射 → 一维 block 网格 + 块内 K 软流水**

- **源方案**：`T.Kernel(ceildiv(n,block_n), threads=(reduce_threads, block_n))`，`tn=threadIdx.y` 一行/线程，
  `tk=threadIdx.x` 归约 lane；每块 8 行 × 32 归约线程。
- **NPU 新算法**：`T.Kernel(num_launched, is_npu=True) as (cid, _)` 一维（`T.Kernel` 在 npuir 只接受一维 block 数，
  `threads=` 无效——`TRAP-threads-kwarg-noop`）；每个 block 处理 `block_M` 个输出行，块内行维交给向量 lane，
  块内 K 方向 `for kk in T.Pipelined(ceildiv(K, block_K), num_stages=S)` 软流水。分核三要素见 §5.5。
- **语义保持论证**：行映射由「threadIdx.y + threadIdx.x 归约」改为「block 处理行块、向量 lane 处理行/元素」，
  输出元素 `y[r]` 仍由唯一的 (block, 行内 lane) 负责，无重复/遗漏；K 仍顺序分块累加。并行度变化不改变数值
  （分组差异同 R1）。

**重设计项 R3：Cube 分形路径 → Vector 归约路径（算法路线，实测裁决）**

- **源方案**：源 GEMV 本身**已是 Vector/线程归约**（非 Cube），但其数据流（SMEM staging + 线程网格）不适用
  NPU。NPU 上的候选路线有二：**(a) Cube 分形路径**（`T.gemm`，把 GEMV 视作 `N=1` 的 GEMM；
  `examples/gemv/example_gemv.py` 的 `naive_gemv_high_perf` 即此形态）；**(b) Vector 归约路径**（`T.reduce_sum`）。
- **NPU 新算法（选定 b）**：Vector 归约路径（R1 + §1.6.3）。裁决依据为**设计期实测**（Developer 模式，
  Ascend910B2C，fp16，probe 脚本见 §1.6.3）：
  - `7168×7168`：Vector **0.188 ms** < Cube `(1,K)×(K,M)+b_transpose` 0.304 ms < Cube `(M,K)×(K,1)` 1.484 ms；
  - `3000×1024`：Vector 0.098 ms < Cube-T 0.118 ms < Cube 0.217 ms；
  - `1024×1024`：Vector 0.088 ms < Cube-T（未测该 shape）< Cube 0.156 ms。
  结构性原因：GEMV 的 GEMM 形态**必有一个维度为 1**，Cube 分形要求 16 对齐 ⇒ MAC 阵列 15/16 空转，且
  `mmadL1` 尺寸 `[BM,BK,1]` 的指令效率极低（`example_gemv.py` 也印证 `naive_gemv_high_perf` 的 Cube 形态非最优）。
  GEMV 访存受限（算术强度 ≈1），Cube 的 MAC 密度优势无从发挥。
- **语义保持论证**：两路线均计算 `Σ_k W[r,k]x[k]`，只是承载单元不同（Cube MAC vs Vector 归约）；数值等价性
  同 R1（fp32 域分组差异，§1.6.1 E2 机器验证）。边界（R/K 非整除、nan/inf）见 R4。

**重设计项 R4：非整除（R 尾块 / K 尾块）的核内处理**

- **源方案**：`T.ceildiv` + 循环守卫；无显式零填充；host 无 padding。
- **NPU 新算法**：全部核内、host 禁 padding（`references/ascend-constraints.md` §4）：
  - `rm = T.min(block_M, R - r0)`（行尾块）、`rk = T.min(block_K, K - k0)`（K 尾块）；
  - 装载用动态 extent 切片 `T.copy(W[r0:r0+rm, k0:k0+rk], Wf[0:rm, 0:rk])`（`T.copy.md` §2.2.2 规则 2）；
  - **K 尾双侧清零**：K 尾块（`rk < block_K`）先 `T.clear(Wf)`、`T.clear(xr)` 再写有效区——否则 W 尾垃圾与
    x 尾垃圾相乘可能产生 `0×NaN`/`0×inf` 污染（W、x **都**清，杜绝 IEEE 陷阱）；清零后尾项贡献 0，`Σ` 不变；
  - 输出写回用动态 extent 切片 `T.copy(ou[0:rm], Y[r0:r0+rm])`。
- **语义保持论证**：新增项 `W[r,k]·x[k] = 0·0 = 0`，对精确 K 求和数学恒等（§1.6.1 E1 机器验证）；R 尾块只影响
  被守卫不写回的行。R/K 非整除行为与源算子一致（有效区数值相同）。

**重设计项 R5：host 侧 `get_sm_version` / autotune / `trace` 去除**

- **源方案**：`default_config` 依 `get_sm_version()` 分支；`autotune_configs` 扫描；`_gemv_wrapped_kernel`
  供 `torch.compile`。
- **NPU 新算法**：去除 SM 分支与 autotune（wrapper 已按 K8 适配为启发式 config）；保留 `lru_cache` + 每 shape
  编译；`torch.compile` 兼容 wrapper 若保留则改名 `npub::`（K7）。
- **语义保持论证**：以上均为 host 侧调度/兼容逻辑，不参与数值计算，去除不影响 I/O 与数值。

### 0.7 标杆实现

- 源算子路径（golden 依据）：`/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py` 的 `_gemv_kernel`（行 585–656）、
  `GemvKernel`（行 680–738）；源 op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py` 的 `GemmOp`（dispatch 行 100–154）。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmTest.ref_program`
  （`torch.matmul(a.T if trans_a, b.T if trans_b)`）+ `GemvBoundaryFixture`（行 90–103、338–354）；
  workload 生成 `examples/TileOPs/tileops/workloads/gemm.py` 的 `GemmWorkload.gen_inputs`。
- golden 以 §0.1 语义为依据，直接移植 `ref_program`（`torch.matmul` + 可选转置），**不复刻** §0.6 的 NPU 算法。
- NPU 侧同类参考（结构，非 baseline）：`examples/gemv/example_gemv.py`（官方 GEMV 示例，含 Cube 与标量变体；
  本设计期实测对比的对象）、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`
  （`T.reduce_sum(dim=1)` + `T.vbrc`/`T.vsub` + 动态 extent 尾块的行归约范式）。
- 源仓 Stage 0 抽取件（参考，非实现基线）：`examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py`。

## 1. 概述

### 1.1 算子名称

`_gemv_kernel`（项目 `gemm`；NPU 实现文件为本算子目录下的 `_gemv_kernel.py`，由 Stage 3 产出，当前尚未生成）。

### 1.2 功能描述

访存受限的 GEMV `y = W·x`（GEMV；`W:(R,K)` 行主序、`x:(K,)`、`y:(R,)`），fp16/bf16 输入、fp32 累加、
输出 dtype 同输入；支持非整除 R/K（kernel 内处理，无 host padding）。迁移后为 NPU 侧 Developer 模式 **Vector
归约**实现（行块分核 + `T.Pipelined` K 流水 + `T.vbrc`/`T.vmul`/`T.reduce_sum`），**不使用 Cube**（§0.6 R3 实测裁决）。

### 1.3 数学公式

$$
y[r] = \sum_{k=0}^{K-1} W[r,k]\cdot x[k], \qquad r=0,\dots,R-1
$$

其中 `K` 为收缩维（W 的最内连续轴），累加在 fp32 域（与源 `accum_dtype="float"` 一致），输出 `y:[R]`
在 fp32 域**单次舍入**到输入 dtype。**命名说明**：源 kernel 用 `n` 表示输出行数、调用为 `_gemv_kernel(n,k)`；
本设计用 `R`（= 迁移字段 `m` = GEMM 逻辑 `M`，因 GEMV 为 `N=1` 特例），语义完全一致（§0.1）。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（与 §1.6.0 调研选定算法一致）：

1. **输出行分块**：把 `y[R]` 划分为 `ceil(R/block_M)` 个行块，一维 grid 每 slot 处理一个行块（§0.6 R2）。
2. **K 维分块流水**：每个行块沿 K 以 `block_K` 分块，`T.Pipelined` 流水（MTE2 装载 W/x tile 到 UB ↔ Vector
   计算）；`acc` 在循环前 `T.clear`，块内用 `T.reduce_sum(clear=False)` 累加（§0.6 R1）。
3. **乘加归约**：每 chunk `T.vbrc(xr(1,BK) → xb(BM,BK))` 广播 x、`T.vmul(Wf, xb, Wf)` fp32 乘、`T.reduce_sum`
   沿 K 归约，结果累加进 `acc`。
4. **尾块与写回**:动态 extent 装载（`rm`/`rk`）+ K 尾双侧清零 + `T.copy(ou, Y)` fp32→dtype 单次舍入（§0.6 R4）。

与源算法的结构差异（来源）：

- 源用 32 线程 warp shuffle 树归约（§0.4 #5）；NPU 改为 `T.reduce_sum` 向量水平归约（§0.6 R1）。
- 源用二维 grid×thread 映射（`threadIdx.x/y`）；NPU 改一维 block 网格 + 块内 K 流水（§0.6 R2）。
- 源候选的 Cube `N=1` 形态（`examples/gemv/example_gemv.py`）经设计期实测淘汰（§0.6 R3，7.9×/1.6× 差距）。
- 源 `disable_tma`/SM 分支/autotune 在 NPU 舍弃（§0.5）。

### 1.5 数据流图

```
GM[W] --T.copy(MTE2)--> UB Wf(block_M, block_K) fp32 --┐
GM[x] --T.copy(MTE2)--> UB xr(1, block_K) fp32 --vbrc--> xb(block_M, block_K) fp32
                                                        │
                            T.vmul(Wf, xb, Wf) fp32 <---┘
                            T.reduce_sum(Wf, acc, dim=-1, clear=False) fp32 (K 流水：T.Pipelined num_stages=S)
                                                        │
                            T.copy(UB→GM, 单次 VCast fp32→dtype, 动态 extent 守卫) --> GM[y]
宿主边界：host 仅做 (R,K) 推断、kernel 选择与 shape/stride 元数据（含可选 flatten 视图），不触碰张量内容、不 padding。
```

### 1.6 算法调研与优化分析

> 分析对象 = §1.6.0 调研选定算法（Vector 归约路径），经 §0.6 重设计落地的 NPU 算法。本节结论是 §3.1 与 §6 的输入。

#### 1.6.0 算法调研（Algorithm Research）

调研深度：**完整调研**（矩阵 × 规约类算子，方法论见 `references/algorithm-research.md` §2；检索入口
`python3 .agents/tools/kb_search.py "gemv matvec reduce_sum split-k"`，命中 `ALG-gemv`）。检索源：
`references/algorithm-candidates.md` 的 `ALG-gemv`（R1 分形路径 vs Vector 规约 / R2 split-K / R4 K 分块与 L0 容量）、
`pattern-library/INDEX.md`（CONST-vector-launch-overhead、CONST-mte2-degradation、CONST-reduce-dims0-skinny、
CONST-capacity-910B2C、TRAP-reduce-mixed-dtype-silent-corrupt〔queue VP〕）、`examples/gemv/example_gemv.py`、
`examples/TileOPs/tileops/kernels/reduction/`、源算子 §0.3/§0.4。**设计期实测 probe**（Developer 模式，Ascend910B2C）见 R4 与 §1.6.3。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：直接 GEMV（逐行点积，源算法同族） | `y[r]=Σ_k W[r,k]x[k]`，K 分块累加 | — | — | ✅（基线） |
| 1 | Cube 分形路径（GEMV 视作 N=1 GEMM） | `y = T.gemm(W_tile, x_tile)`，x 补成 1 列 / b_transpose 转置形态 | 数学恒等（同一 matvec） | 无代数节省；MAC 阵列 15/16 空转（**实测更慢**，R4 淘汰） | ✅（R3/R4 对比） |
| 2 | split-K（K 维并行分解） | K 拆 P 段独立部分和，二阶段归约 | 数学恒等（加法结合律，fp32 分组变化） | FLOPs 不变；访存 + `R·P·4` 部分和往返；可并行度↑ | ✅（R3 对比） |
| 3 | Strassen/FFT 类代数化简 | 矩阵乘法 Strassen / 卷积 FFT | **不适用**：GEMV 是 `N=1`，Strassen 需 ≥2 个向量（2×2 分块）才成立；FFT 仅对卷积/长相关有效 | 无 | ❌（结构性不适用） |
| 4 | 降低精度（fp16 累加） | 在 fp16 域累加 | **非等价**（源明确 fp32 累加；且 fp16 长 K 累加相对误差 ~√K·ε ≈ 0.1 ≫ 容差） | 无实质收益 | ❌（语义违反，§0.1 dtype 语义） |
| 5 | x 的广播因子 | 逐 k 标量 `x[k]` vs 向量 `vbrc` 广播 | 数学恒等（同一乘积） | 向量化后 op 数大幅下降（实现层，§1.6.2） | ❌（非代数化简，归 §1.6.2） |

> R1 结论：**GEMV 的密集矩阵-向量乘无代数化简空间**——FLOPs 下界即 `R·K` 次乘加（需读取全部 `R·K` 个 W 元素并
> 逐一参与乘加），Strassen 类仅在 `N≥2` 时成立，Winograd/FFT 类仅对卷积有效。可选结构只有「承载单元」（Cube vs
> Vector）与「K 是否拆分」（split-K）两类，分别在 R4 与 R3 处理。

**R2 在线算法**：**有，且基线即天然单遍**。结构依据：GEMV 的归约是**简单和**（running statistic = 部分和，
无 max/重缩放需求），按 chunk 顺序做 running accumulator 即 O(tile) 中间状态的单遍（online）归约——
`T.reduce_sum(..., clear=False)` 在 `acc` 上跨 chunk 累加正是该形态。收益口径：扫描遍数 = **1**（W 只读一遍、
不物化中间结果）；中间缓冲 = `O(block_M·block_K)`（UB 驻留，非 O(R·K)）；UB 驻留可行性 = 是（§4.5 预算）。
split-K（R1 候选 2）是**并行分解**而非 online 变体：它保持单遍但把 K 分给多核，代价是 `R·P·4` 字节的部分和
GM 往返与二阶段归约（§1.6.3 备选 B）。

**R3 复杂度对比**（四口径；基线在表；口径含/不含已注明）：

| 算法候选 | FLOPs | 访存量 Bytes | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|-------------|---------|-------------|---------------------|
| 基线（Vector 归约，ideal traffic） | `2·R·K`（乘加=2） | 理想 `(R·K + K + R)·elem_bytes`（每元素读一次 + 输出写）；**不含** UB 内部搬运与 tiling 重读（由 L2 吸收） | 1（K 单遍 running 累加） | `O(block_M·block_K)` fp32 ×2（Wf + xb），UB 驻留 | 高（行块分核）；无跨核归并 |
| Cube 分形（N=1） | `2·(16·R·K)`（分形 pad 16×，`[block_M,block_K,1]`） | 与基线同阶（W 读一遍） | 1 | L0C `block_M·16·4` + L1 tile ring | 高；无跨核归并 |
| split-K（P 段） | `2·R·K`（不变） | 理想 + `R·P·4`（部分和写读）+ 二阶段读回 | 1（K 分段） | `R·P·4`（跨核 partial 落 GM） | 更高（P×网格）；需二阶段归约 |
| 候选 5（向量广播，实现层） | `2·R·K` | `(R·K+K+R)·2`（不落中间 GM） | 1 | 同上 | 同基线 |

> 口径说明：FLOPs 按乘加=2；访存为 GM 流量（不把 UB 内部搬运计入）；理想口径假设张量经 L2 复用，tiling 重读不落
> HBM（设计期近似）。**带宽受限 + 发射受限**：设计期估算表明本算子**向量发射项主导**（§下 D-2 估算），而非纯
> 带宽受限——这一点直接决定了 Cube/Vector 的裁决（Cube 向量发射少但 mmadL1 效率低，实测更慢）。

**R4 硬件亲和性评估**（逐候选对照检查清单；**含设计期实测**，负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **Vector 归约（基线）** | ✅ 归约 + 逐元素 → Vector（`VReduceOp`/`VMulOp`/`VBrcOp`） | ✅ UB 192KB（§4.5 预算 ~129KB） | ✅ `block_K=256`（fp32 ×4 / fp16 ×8 整数倍） | ✅ R/K 编译期常量，尾块 `T.min` 动态 extent | ✅ `T.Pipelined` 双缓冲 | **✅ 选定（实测最快）** |
| Cube `(M,K)×(K,1)` | ❌ MAC 阵列 N=1 → 15/16 空转；`mmadL1 [BM,BK,1]` 指令效率极低 | ✅（L0C 8KB/L1） | ✅ 分形对齐需 pad | ✅ | ✅ | ❌ 淘汰（**实测 7168² 1.484 ms vs 0.188 ms，7.9×**） |
| Cube `(1,K)×(K,M)` b_transpose | ❌ MAC 阵列 M=1 → 15/16 空转（转置只换了空转维） | ✅ | ✅ | ✅ | ✅ | ❌ 淘汰（**实测 7168² 0.304 ms vs 0.188 ms，1.6×**；已是 Cube 最优形态） |
| split-K | ✅ 用 Vector | ✅ + 部分和缓冲 | ✅ | ✅ | 需二阶段归约 / atomic（`CONST-store-fixpipe-gm-only` 说明跨引擎强制 GM 往返） | ❌ 本 manifest R≥1024 ⇒ 行分核并行度已足（≥16 块），split-K 净亏；仅 R 极小（≪block_M）且 K 极大时的**备选**（§1.6.3） |
| 候选 5（向量广播） | ✅ 随基线 | ✅ | ✅ | ✅ | ✅ | **✅ 采纳（并入基线，§1.6.2）** |

设计期实测（probe #1，Developer 模式 / Ascend910B2C / fp16，脚本
`examples/gemm/_gemv_kernel/history_version/design_probe_gemv.py`、`..._cubeT.py`、`..._sweep.py`）：

| shape (R,K) | 流量 MB | Vector 归约 (ms) | Cube `(M,K)×(K,1)` (ms) | Cube-T `(1,K)×(K,M)` (ms) |
|-------------|--------|------------------|--------------------------|----------------------------|
| 1024×1024 | 2.1 | **0.088** | 0.156 | 未测 |
| 3000×1024 | 6.2 | **0.098** | 0.217 | 0.118 |
| 7168×7168 | 102.8 | **0.188** | 1.484 | 0.304 |

**调研结论**：选定 **Vector 归约路径（基线 `Σ_k` 单遍 running 累加）+ x 的 `vbrc` 向量广播**（R4 实测最优；
`examples/gemv/example_gemv.py` 的 Cube `N=1` 形态实测落后 7.9×/1.6×）。关键依据：① GEMV 算术强度 ≈1，
访存受限 + 向量发射受限，Cube 的 MAC 密度优势无从发挥；② GEMV 的 GEMM 形态必有 1 维为 1，Cube 分形 16 对齐
使阵列空转、`mmadL1 [BM,BK,1]` 效率极低（实测证实）；③ Vector 路径把归约交给 `VReduceOp`、把 K 保持为最内连续
轴（`CONST-reduce-dims0-skinny` 的**快路径 dims=1**）；④ split-K 在本 manifest（R≥1024）净亏。与基线的结构差异：
承载单元从源线程归约改为 NPU 向量水平归约（无代数结构变化）。**调研范围**：`algorithm-candidates.md` 的 `ALG-gemv`
（R1/R2/R4 条目全部评估）、`ALG-reduction`（水平 vs 垂直 lane）、`pattern-library/INDEX.md` + constants.md
（CONST-vector-launch-overhead / CONST-mte2-degradation / CONST-reduce-dims0-skinny / CONST-capacity-910B2C）、
`examples/gemv/example_gemv.py`、`examples/TileOPs/tileops/kernels/reduction/`、源算子 §0.3/§0.4、模型知识
（Strassen/FFT/split-K）+ 设计期实测。源算法已识别的 11 项优化手段（§0.4）承接情况：#1/#2/#3/#4/#6/#9 经
§0.5 等价替换承接、#5/#8 经 §0.6 R1/R2 重新设计承接、#7/#10(SM 部分) 舍弃、#11 保留；无静默丢弃。

**设计期估算（D-2）**：代表 workload `7168×7168`（fp16；manifest/源测试中的大 GEMV 档，另见 `18432×7168`）。
- **流量项**：理想 GM 流量 `(R·K+K+R)·2 = 102.8 MB`；按 `CONST-mte2-degradation`（2026-09-07 build / CANN 8.5.0）
  混合读写地板 ~1.26 TB/s ⇒ ≈ **82 µs**；按 `CONST-fabric-bw`（1.73 TB/s，乐观上界）⇒ ≈ 59 µs。
- **发射项**：Vector op 数 ≈ 3/chunk（vbrc + vmul + reduce_sum）× `ceil(K/block_K)=28` × `ceil(R/block_M)=112`
  = 9408 op，分 40 核 ⇒ 235 op/核 × `CONST-vector-launch-overhead` 0.5 µs/op ≈ **118 µs**。
- **容量项**：UB ~129KB < 192KB（无 GM spill）⇒ ≈ 0。
- `估算下界: ≈ max(82, 118, 0) ≈ 118 µs（发射项主导，引用 CONST-vector-launch-overhead / CONST-mte2-degradation）`；
  **实测 188 µs**（同 shape，probe #1）——量级吻合，证实本算子**向量发射受限**（不是纯带宽受限），故 §1.6.2/§1.6.3
  的杠杆是「减少向量 op 发射次数」（更宽 tile）与「减少搬运段数」，非单纯堆带宽。常数版本戳：
  tilelang 0.1.2 dev-root build 2026-10-13 / CANN 8.5.0 / Ascend910B2C。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（Vector 归约路径）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | K 尾双侧零填充 | 对精确 K 求和 `Σ_{k<K} W[r,k]x[k]` | 对补齐到 `block_K` 倍数的 K 求和 `Σ_{k<K'}`，`K'=ceil(K/block_K)·block_K`，且 `k≥K` 时 `W=x=0`（**双侧清零**） | 数学恒等：新增项 `0·0=0`（W、x **都**清零，杜绝 `0×NaN`/`0×inf`）；仅改变 fp32 求和的加零操作 | 使 `block_K ∤ K` 时免 host padding（host padding 被 `references/ascend-constraints.md` §4 禁止）；clear 只作用于尾块，代价可忽略 |
| E2 | K 维分块累加（chunked reduction，容差内等价） | 单链顺序和 `Σ_{k=0}^{K-1}`（一次扁平 fp32 累加） | `Σ_{c=0}^{C-1}(Σ_{k∈chunk_c})`：每 chunk 内向量树归约、chunk 间顺序累加（`clear=False`） | 加法结合律；fp32 分组变化 ⇒ 容差内等价（分组差异对 fp16/bf16 输出影响经机器验证，见下表 E2 行；chunk 内部分和量级更小，实测比扁平链**更接近** fp64 参照） | 使中间缓冲从 `O(R·K)` 降到 `O(block_M·block_K)`（UB 驻留，单遍无重读）；W 只读一遍 |
| E3 | 输出 fp32 单次舍入（**非回归约束**，非增量优化） | 伪基线：把 fp32 累加结果额外物化回 GM，再单独 cast 到 out dtype（**非**源算子行为） | `y = round_to_dtype(Σ W·x)`，在 UB→GM store 内**单次** RNE 舍入 | 数学恒等：两式均对同一 fp32 累加器做一次 RNE 舍入，比特级一致；前提是**经过 fp32 opmath 域**（`TRAP-fp16-opmath-golden`）。**重新定位**：源算子 `c[...] = c_reduced[0]` 本就单次融合 cast，故 E3 **非相对源算子的增量优化**，而是「不得引入 fp32 中间 GM 往返」的非回归约束 | 收益口径（相对伪基线）：省 `2·R·4` B；**相对源算子 → 0 增量收益**（冻结该行为） |

> **无其他代数化简空间**：GEMV 无除法/超越函数（无「除法转倒数 / rsqrt / exp2 缩放」类）、无公共子表达式
> （`W[r,k]·x[k]` 各出现一次）、无可降代换算子；核心乘加已是 `R·K` 次 MAC 的信息论下界（R1 候选 3/4 已判不适用）。
> 因此本项采纳 3 条（E1 精确、E2 容差内、E3 非回归）。

**机器验证（D-1）**：`examples/gemm/_gemv_kernel/verify_equiv.py`（torch/numpy CPU，fp64 参照 + fp32 opmath 域；随机
张量 + 角点值 ±inf/NaN/−0/subnormal + 边界 shape `(1000,1024)/(257,3000)/(17,7168)/(1,3000)/(5,33)`；两 dtype）。
E1/E3 为 bitwise 门（两条对照为**不同计算路径**：精确 K matmul vs 双侧零填 matmul；融合单次舍入 vs 显式 fp32 缓冲
物化（字节往返）+ 分离 cast），E2 为容差门（相对 fp64 参照的 §8.2 容差违反率）。执行结果
（`python3 examples/gemm/_gemv_kernel/verify_equiv.py`）：

| 采纳项 | dtype | 最大偏差 | 违反率 | 结论 |
|--------|-------|---------|--------|------|
| E1-ktail-zerofill (bitwise) | torch.float16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |
| E1-ktail-zerofill (bitwise) | torch.bfloat16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |
| E2-chunked-reduction (tol) | torch.float16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |
| E2-chunked-reduction (tol) | torch.bfloat16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |
| E3-fused-vs-materialised (bitwise) | torch.float16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |
| E3-fused-vs-materialised (bitwise) | torch.bfloat16 | 0.0 ulp | 0.000000 | `EQUIV_PASS` |

> **E2 口径说明**：`max_dev` 为「分块累加 vs 扁平累加」在输出 dtype 上的最大 ULP 差（本批 shape 实测为 0，即两分组
> 在输出精度下一致）；违反率为相对 fp64 参照超出 §8.2 容差的比例（0）。E2 的门是**容差门**（`EQUIV_PASS` 判据 =
> 违反率 0），而非 bitwise 门——扁平与分块分组的 fp32 差异理论上非零，本批 shape 恰好一致，故另报 ULP 供核对。
> **E1 单侧清零反例探针**（信息性，非门禁行）：只清 W 尾、x 尾置 NaN 时，`(W_pad @ x_pad)` 对精确 K 结果产生 4/4 行
> 失配——**证实双侧清零（E1）的必要性**（否则 `0×NaN=NaN` 污染）。

**优化结论**：采纳 3 项（E1 K 尾双侧清零 / E2 K 维分块累加 / E3 输出单次舍入），全部 `EQUIV_PASS`。**优化后公式**
（供 §3.1 使用）：

`y[r] = round_to_dtype( Σ_{c} Σ_{k∈chunk_c, k<K} W[r,k]·x[k] )`，其中尾部 chunk 的 `k≥K` 项取 `W=x=0`（E1），
chunk 内向量树归约、chunk 间累加（E2），`round_to_dtype` 在 store 内单次 RNE（E3）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | K 维归约累加 | 源 `for _k in T.serial(tile_k): c_accum[0] += ...`（线程内标量 FMA 循环） | `T.reduce_sum(Wf, acc, dim=-1, clear=False)`（向量水平归约，`T.reduce_sum.md`） | ✅ | — |
| 2 | x 的逐元素广播 | 源 `a_local[_k] = a[...]` + 每线程点积 | `T.vbrc(xr(1,BK) → xb(BM,BK))`（`T.vbrc.md` §2.2.2 合法形态 `(1,N)→(M,N)`） | ✅ | — |
| 3 | 逐元素乘 | 源标量 `a_local[_k].astype(float) * b_shared[...]` | `T.vmul(Wf, xb, Wf)`（fp32 同形乘，`T.vmul.md`） | ✅ | — |
| 4 | x / W tile 装载 | 源 `T.vectorized` 装载 + `T.copy` 到 SMEM | `T.copy`（GM→UB，MTE2 向量化搬运，自动 dtype cast） | ✅ | — |
| 5 | 输出写回 + cast | 源 `c[bn*block_n + tn] = c_reduced[0]` 逐元素 | `T.copy(ou[0:rm], Y[r0:r0+rm])`（MTE/VPipe 向量化 + 单次 VCast 舍入） | ✅ | — |
| 6 | K 尾零填充 | 源无（无 padding） | `T.clear(Wf)` / `T.clear(xr)`（向量 memset，仅尾块触发） | ✅ | — |
| 7 | `cid/r0/k0`、`rm/rk` 边界 | 标量乘加 / `T.min` | 无 | ❌ | **block 级索引 / 任务映射 + 边界元数据**：每 block 仅数个标量，非逐元素热点；无对应逐元素向量等价 API（索引不构成张量） |
| 8 | K 流水循环本身（`T.Pipelined`） | 迭代 | 无 | ❌ | **tile 级顺序依赖**：`acc` 跨 K-chunk 累加（循环携带依赖），已用多缓冲流水隐藏延迟；无法用单次归约替代（那需先把全 K 物化进 UB，破坏 tiling 且超容量） |
| 9 | chunk 内 `T.reduce_sum` 的树形归约步 | 库函数内部 | 无（已在向量 API 内） | ❌ | **API 内部实现**：归约为 `VReduceOp` 的向量语义，非用户可见的逐元素标量循环 |

**向量化结论**：全部**逐元素 / 归约**计算已向量化（装载 `T.copy`、广播 `T.vbrc`、乘 `T.vmul`、归约 `T.reduce_sum`、
清零 `T.clear`、写回 `T.copy`+VCast）；保留 3 处标量 / 循环，理由类别：block 索引 / 任务映射（#7）、tile 级顺序依赖
（#8）、API 内部实现（#9）——均非逐元素热点。§6 循环结构不含任何逐元素标量循环，与本节一致。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> GEMV 是**规约类**算子（softmax/layernorm 同族口径）：必须同时枚举
> ①「水平归约（lane 映射到归约轴 K，用向量 reduce）vs 垂直扫描（lane 映射到独立行 M，串行步进 K）」，
> ②「I/O 原生布局 vs 核内重排」，③「承载单元（Cube vs Vector）」三类候选。I/O layout（W `(R,K)` 行主序、x `(K,)`、
> y `(R,)`）是契约；核内布局 / lane 映射 / 承载单元是设计变量。

**轴质量评分（主选轴 = K，水平归约）**：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | K 轴 = 归约轴，`block_K=256` 为 fp32 向量宽（×4）/ fp16（×8）整数倍 | ✅ 0 尾 lane 浪费（K=1024/3000/7168 的尾块由动态 extent 处理） |
| 尾 lane 浪费率 | `(ceil(K/vw)·vw − K)/K` | 0（`block_K=256` 整除向量宽；K 非 256 倍数时由尾块动态 extent 处理，不产生 lane 浪费） |
| 累加链形态 | K 是**最内连续轴**（stride=1），归约走 `VReduceOp` 快路径（`CONST-reduce-dims0-skinny`：dims=1 行向归约为快路径；dims=0 长条归约灾难性慢 ~21ns/elem） | ✅ 快路径 |
| repack 代价 | 主选无 repack（I/O 原生行主序直接用） | ✅ 0 |
| UB 容量影响 | 主选需 Wf + xb 双 fp32 tile 同时驻留（§4.5） | ✅ ~129KB @ (BM=64,BK=256) < 192KB |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局 / 轴方案 | 向量化轴 + lane 映射 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|--------------|---------------------|------------|----------------|---------|
| 1 | **I/O 原生 (R,K) 行主序 + 水平归约** | K（归约轴）；lane → K；行块内 `reduce_sum(dim=-1)` | 无 | 基线；无 repack；快路径归约；实测 7168² **0.188 ms** | **✅ 主选** |
| 2 | 核内转置到 (BK, BM) + 垂直扫描（dim=0 归约） | M（独立行 lane），串行步进 K | 核内 `T.transpose` 二轴交换链（PL-1.1，~5.4µs/链） | dim=0 长条归约走**慢路径**（`CONST-reduce-dims0-skinny`：(400,4) 33.4µs）；且需额外转置缓冲；无收益 | ❌（慢路径 + 转置开销，`CONST-reduce-dims0-skinny` 佐证） |
| 3 | host permute W 到 (K,R) | — | host permute（PL-1.3 实测 106–147µs） | repack 代价 > 算子本体收益；且 host 禁改输入内容（`ascend-constraints.md` §4） | ❌ |
| 4 | Cube NZ 布局（`load_nd2nz`） | Cube 分形 | 框架自动 ND2NZ | 实测落后 1.6×（Cube-T 0.304 ms）～7.9×（Cube 1.484 ms）@7168² | ❌（§1.6.0 R4 实测淘汰） |
| 5 | x 广播轴方案：`vbrc` 展开 vs vmul 行/列广播 | 同 #1（K） | 无 | `T.vmul` 文档仅支持同形/行广播 `[M,N]×[M,1]`/标量；x 是 `(1,BK)→(BM,BK)` 的**首轴广播**，须 `T.vbrc`（`T.vbrc.md` §2.2.2 合法）；PL-1.15 实测非典型广播形态有 +12~16% 性能税，故走文档化 vbrc 展开 | ✅（`vbrc` + 同形 `vmul`） |
| 6 | x 常驻 vs 逐 chunk 装载 | 同 #1 | 无 | x 流量仅 `K·2` 字节（占 W 的 `1/block_M`），逐 chunk 装载与 W 同步进流水，收益 <3%；x 常驻需额外 UB | ✅（逐 chunk 装载） |

**逐候选评分（主选 #1 vs Cube 备选 #4）**：主选 #1 无 repack、归约走快路径、UB ~129KB；#4 需 ND2NZ + L1/L0C 路径、
且实测落后。判定裕度：**不落在未实证常数区间**（#4 的落后幅度 1.6×/7.9× 为直接实测，非估算），故**无需实验裁决模式**——
但为审慎，仍保留 Cube-T 形态为**备选（Stage 4 A/B）**：结构 = `T.gemm(x_s(1,BK), W_s(BM,BK), y_l(1,BM), b_transpose=True)`
（probe 脚本 `examples/gemm/_gemv_kernel/history_version/design_probe_gemv_cubeT.py` 已验证可实现、数值正确），
若 Stage 4 在某 shape 上实测 Cube-T < 主选 × 0.95 才翻转（当前数据不支持）。**split-K 备选**：仅当 `R ≪ block_M`
（并行度不足）且 K 极大时启用，本 manifest R≥1024 不触发。

**布局决策结论**：主选 **#1（I/O 原生行主序 + K 轴水平归约）**：核内布局 = `Wf(BM,BK)` fp32 + `xb(BM,BK)` fp32，
向量化轴 = **K**（归约轴，水平归约 `T.reduce_sum(dim=-1)`），lane 映射 = lane→K，repack = 无。该结论同步落入
§3.3 伪代码（累加循环内层向量维 = K）、§4 内存规划（Wf/xb 形状与预算）、§6 循环结构。弃选方案量化理由见候选矩阵。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: Developer**。

### 2.2 选型理由

- 算子为**纯 Vector**（`T.copy` / `T.vbrc` / `T.vmul` / `T.reduce_sum` / `T.clear`），无手写 NZ/L1/L0C 需求；Developer
  模式提供 `T.alloc_shared`/`T.alloc_fragment`（自动映射 UB）+ `T.copy`（自动同步/自动 VCast）+ v-prefix 向量 API 的
  完整能力，覆盖 §0.6 R1–R4 与 §1.6.3 主选的全部需求。
- 同类最近范式 `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`
  即 Developer 模式（`os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")` + `T.reduce_sum` + 动态 extent 尾块），
  是可直接借鉴的行归约模板（AGENTS.md「Operator Implementation Baseline」：优先改动既有案例）。
- 源 GPU 的 warp shuffle / cp.async disable_tma / SM 分支在 NPU 无对应能力（§0.5），Developer 的自动同步/自动布局
  正好替代手工部分。
- `op_requirements.programming_mode = developer`（用户显式指定）。

> **设计期实测注意（重要）**：probe 发现 TileLang NPUIR 的默认 `TILELANG_ASCEND_MODE` **不是** Developer；未显式设置时
> `T.copy` 跨 dtype 报 `T.copy does not support element type casting`、`alloc_shared` 落 `cbuf` 导致 `hivm.hir.store
> only support ub to gm`。**本设计所有 kernel 必须显式设定 Developer 模式**（源码内 `os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")`
> 或编译环境变量），否则 §0.6 R3 的实测结论与 §5/§6 结构均不成立。实测证据：
> `examples/gemm/_gemv_kernel/history_version/design_probe_gemv_cases.py`（Developer 模式下 4/4 结构 OK）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（tile / 输出中转 → UB）、`T.alloc_fragment`（Wf/xb/acc → UB）；不显式用 `alloc_L1/alloc_L0C` |
| 计算方式 | `T.vbrc`（广播）+ `T.vmul`（fp32 乘）+ `T.reduce_sum(clear=False)`（K 归约）+ `T.clear`（尾块/累加器） |
| 数据搬运 | `T.copy`（GM↔UB，自动 dtype cast、自动同步、动态 extent 守卫） |
| 同步 | 自动（`T.Pipelined` 由编译器插入多缓冲同步），无手动 barrier |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（K 尾双侧清零 + K 分块累加 + 单次舍入）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `r0 = cid·block_M`，`rm = min(block_M, R - r0)` | 行块起点与有效行数 |
| 2 | `k0 = kk·block_K`，`rk = min(block_K, K - k0)` | K 分块起点与有效长度 |
| 3 | `Wf[0:rm,0:rk] ← W[r0:r0+rm, k0:k0+rk]` | 装载 W tile（动态 extent，fp16→fp32 cast） |
| 4 | `xr[0:rk] ← x[k0:k0+rk]`；`xb = broadcast(xr)` | 装载 x chunk 并广播到行块 |
| 5 | `Wf ← Wf ⊙ xb` | fp32 逐元素乘（K 为内层向量维） |
| 6 | `acc[r] += Σ_{k<rk} Wf[r,k]` | 沿 K 向量水平归约并累加（chunk 间 `clear=False`） |
| 7 | `ou[r] = round_dtype(acc[r])`；`Y[r0:r0+rm] ← ou[0:rm]` | 单次舍入 + 动态 extent 写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Grid + `r0/rm` | `T.Kernel(num_launched, is_npu=True)`；`rm = T.min(block_M, R - r0)` | 一维 grid，`num_launched = ceildiv(R, block_M)`（§5.5） | Developer |
| 2 | 分配 UB | `T.alloc_fragment((block_M, block_K), "float32")`（Wf）；`T.alloc_fragment((block_M, 1), "float32")`（acc）；`T.alloc_shared((1, block_K), "float32")`（xr）；`T.alloc_shared((block_M, block_K), "float32")`（xb）；`T.alloc_shared((block_M,), dtype)`（ou） | — | Developer |
| 3 | K 循环流水 | `for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=S)` | `S` 按 UB 容量（§4.5） | Developer |
| 4 | 装载 W tile（含 cast） | `T.copy(W[r0:r0+rm, k0:k0+rk], Wf[0:rm, 0:rk])` | Developer 跨 dtype 自动 VCast（`T.copy.md` §2.3 第 4 条） | Developer |
| 5 | 装载 x chunk | `T.copy(x[k0:k0+rk], xr[0, 0:rk])` | 动态 extent | Developer |
| 6 | K 尾双侧清零 | `T.clear(Wf); T.clear(xr)`（仅 `rk < block_K` 时） | 向量 memset；双侧清杜绝 `0×NaN`（E1） | Developer |
| 7 | x 广播 | `T.vbrc(xr, xb)` | src `(1,BK)` → dst `(BM,BK)`（`T.vbrc.md` §2.2.2 合法形态） | Developer |
| 8 | 逐元素乘 | `T.vmul(Wf, xb, Wf)` | 同形 fp32 乘，原地（`T.vmul.md` §2.3 第 3 条） | Developer |
| 9 | K 归约累加 | `T.reduce_sum(Wf, acc, dim=-1, clear=False)` | `acc` 循环前 `T.clear`；src/dst **同 dtype fp32**（规避混合 dtype 静默错误） | Developer |
| 10 | 写回 + cast | `T.copy(ou[0:rm], Y[r0:r0+rm])` | 单次 VCast fp32→dtype（E3），动态 extent 守卫 | Developer |

API 依据：`docs/Tilelang.language/规约操作/T.reduce_sum.md`、`docs/Tilelang.language/shape操作/T.vbrc.md`、
`docs/Tilelang.language/数学操作/T.vmul.md`、`docs/Tilelang.language/内存操作/T.copy.md`（§2.2.2/§2.3）、
`docs/Tilelang.language/内存操作/T.clear.md`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`。

### 3.3 计算伪代码

```python
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")  # 必须（见 §2.2 实测注意）

import tilelang
import tilelang.language as T

# 工厂按 (R, K, dtype) 特化；config = (block_M, block_K, num_stages)
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemv_kernel(R, K, dtype="float16", block_M=64, block_K=256, num_stages=2):
    accum_dtype = "float32"
    num_launched = T.ceildiv(R, block_M)   # 网格数不变量：= 逻辑块数（非 persistent，禁止 min 截断）

    @T.prim_func
    def main(
        W: T.Tensor((R, K), dtype),
        x: T.Tensor((K,), dtype),
        y: T.Tensor((R,), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            r0 = cid * block_M
            rm = T.min(block_M, R - r0)

            Wf = T.alloc_fragment((block_M, block_K), accum_dtype)   # W tile（fp32）
            xr = T.alloc_shared((1, block_K), accum_dtype)           # x chunk（fp32）
            xb = T.alloc_shared((block_M, block_K), accum_dtype)     # x 广播
            acc = T.alloc_fragment((block_M, 1), accum_dtype)        # 行和累加器
            ou = T.alloc_shared((block_M,), dtype)                   # 输出中转（dtype）

            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                k0 = kk * block_K
                rk = T.min(block_K, K - k0)
                # K 尾双侧清零（仅尾块触发；先清后写）
                if rk < block_K:
                    T.clear(Wf)
                    T.clear(xr)
                # 装载（动态 extent；Developer 跨 dtype 自动 VCast）
                T.copy(W[r0:r0 + rm, k0:k0 + rk], Wf[0:rm, 0:rk])
                T.copy(x[k0:k0 + rk], xr[0, 0:rk])
                # 广播 x -> (BM, BK)，fp32 乘，沿 K 向量归约累加
                T.vbrc(xr, xb)
                T.vmul(Wf, xb, Wf)
                T.reduce_sum(Wf, acc, dim=-1, clear=False)
            # 单次舍入 + 动态 extent 写回
            for i in T.Parallel(block_M):
                ou[i] = T.cast(acc[i, 0], dtype)
            T.copy(ou[0:rm], y[r0:r0 + rm])

    return main
```

> 说明：`cf.` 源 GPU 的 `_gemv_main(a, b, c)`（`a:(k,)`/`b:(n,k)`/`c:(n,)`）——本设计按 §0.1 命名映射
> 写为 `main(W:(R,K), x:(K,), y:(R,))`，参数顺序与源不同但语义一致（源 `forward` 即 `kernel(a, b)`，此处
> `main(W, x)`），Stage 3 的 wrapper 需按此对齐（§4.7）。
> `if rk < block_K:` 是**编译期可判定形态**（`rk` 是否小于 `block_K` 取决于 K 是否整除 block_K；`kk` 只在最后一个
> 迭代取值）——Stage 3 应将其提升为**工厂层 Python 分支**（`if K % block_K != 0:` 生成带尾块处理的变体，否则生成
> 无尾块的干净变体），避免把运行时谓词留在 `T.Pipelined` 循环体内（`CG-2026-0001` Developer codegen 缺口类，
> gemm 迁移复盘教训）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.Kernel(..., is_npu=True)` | `examples/gemv/example_gemv.py`、`docs/开发指南.md` §2 | 官方示例 + 设计期实测 |
| `T.alloc_shared` / `T.alloc_fragment` | logsumexp single kernel、`docs/Tilelang.language/内存操作/T.alloc_shared.md` | 示例 + 设计期实测 |
| `T.reduce_sum(buf, out, dim=-1, clear=False)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md`（§2.3 说明 clear=False 须先初始化 dst） | 文档 + 设计期实测（probe #2 `reduce`/`vbrc` case OK） |
| `T.vbrc((1,BK)→(BM,BK))` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 合法示例、§2.4 示例 2 | 文档 + 设计期实测 |
| `T.vmul`（同形 fp32，原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2/§2.3 | 文档 |
| `T.copy`（GM↔UB，跨 dtype cast，动态 extent） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2、§2.3 第 4 条 | 文档 + 设计期实测 |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | 文档 |
| `T.Pipelined(..., num_stages=S)` | `docs/开发指南.md` §3.2、`examples/gemv/example_gemv.py` | 示例 + 设计期实测 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；实查返回 20（§5.5） | 实查已验证（本机） |
| `T.gemm`（Cube 备选） | `docs/Tilelang.language/线性代数操作/T.gemm.md`、`examples/gemv/example_gemv.py` | 示例已验证（备选；主选不用） |
| `T.tma_copy`/`cp.async`/`disable_tma`/`threads=`/`tvm_thread_allreduce`/`get_sm_version` | 源 GPU 代码（§0.4） | **不使用**（GPU-only，§0.5 舍弃） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（源为二维 grid×thread；本设计用一维） | 一维 `T.Kernel(num_launched, is_npu=True)`（§0.6 R2） |
| GPU 专用 API 不可用 | 是（warp shuffle / cp.async disable_tma / thread 网格 / `get_sm_version`） | 逐项四态处置（§0.5），NPU 等价 API 替换或舍弃 |
| GEMM 要求 M,N 为 block 整数倍 | 本设计**不用 GEMM**（纯 Vector）；但 R/K 非整除仍需处理 | kernel 内动态 extent 切片 + K 尾双侧清零 + 写回守卫（§0.6 R4）；**不做 host padding** |
| L0C 容量上限 128KB | 否（无 Cube） | n/a（备选 Cube-T 形态 L0C = `1×block_M×4`，远小于 128KB） |
| 物理核数限制 | 是 | §5.5 分核三要素（实查 20，纯 Vector 翻倍 = 40） |
| host 侧禁止改动输入 / padding | 是 | 全部逻辑在 kernel 内；host 只做 shape/stride 元数据（§0.3、`ascend-constraints.md` §4） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU SM90 `_gemv_kernel`） | 本项目（Ascend NPU） | 转换方案 |
|--------|-------------------------------------|---------------------|----------|
| Kernel 维度 | 二维 grid×thread `T.Kernel(ceildiv(n,block_n), threads=(32,block_n))` | 一维 `T.Kernel(ceildiv(R,block_M))` + 行块 | §0.6 R2 |
| 归约 | 线程内 `tile_k` 顺序 FMA + 32 线程 warp shuffle `tvm_thread_allreduce` | `T.reduce_sum(dim=-1, clear=False)` 向量水平归约 | §0.6 R1 |
| 广播 / 乘 | 每线程标量 `a_local[_k]·b_shared[...]` | `T.vbrc` 广播 + `T.vmul` fp32 | §1.6.2 #2/#3 |
| 流水 / 同步 | `T.Pipelined(num_stages=2)` + cp.async（`disable_tma=True`） | `T.Pipelined(num_stages=S)`，自动同步（无 disable_tma） | §0.5 #3/#7 |
| 内存层级 | SMEM `b_shared` + register `a_local` | UB `Wf/xb/xr/acc/ou` | §0.5 #2/#4 |
| 非整除 | `T.ceildiv` + 循环守卫（无 padding） | 动态 extent + K 尾双侧 `T.clear` | §0.6 R4 |
| dtype/累加 | fp16/bf16 → fp32 乘/加 → 输出单次舍入 | 同（`T.vmul` fp32 + `T.reduce_sum` fp32 + `T.copy` VCast） | 保留 |
| host | `get_sm_version` 分支 + `autotune_configs` | 无 SM 分支、无 autotune（wrapper K8 已适配） | §0.5 / §0.6 R5 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 高度相似 | Developer 模式行归约：`T.reduce_sum(..., dim=1)` + `T.vbrc`/`T.vsub` + 动态 extent 尾块 + fp32 累加 + 单次 cast；分档测试（L0/L1/L2/Boundary）组织 |
| `examples/gemv/example_gemv.py` | 高 | 官方 GEMV 示例（Cube `N=1` + 标量变体）；本设计期 A/B 的对照形态；`T.copy(A[...], A_shared[:,0])` 列向量装载写法 |
| `examples/gemm/matmul_dynamic_shape.py` | 中 | 动态 extent 尾块（`T.min`）与一维 grid 分解写法 |
| `examples/TileOPs/tileops/kernels/gemm/gemm/gemm.py` | 高（wrapper） | NPU 侧 `GemvKernel` wrapper（K5/K7/K8/K9 适配）；本 kernel 的上层契约 |
| `examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py` | 高（源） | GPU `_gemv_kernel` 抽取件（参考语义，非实现基线） |
| `examples/TileOPs/tests/ops/test_gemm.py` | 高 | golden `ref_program` + `GemvBoundaryFixture` + 容差（fp16 1e-3 / bf16 1.6e-2；GEMV 档 1e-2） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `W`（源 `b`） | `(R,K)` | float16 / bfloat16 | 行主序，K 最内连续 |
| `x`（源 `a`） | `(K,)` | same_as(W) | 一维激活向量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y`（源 `c`） | `(R,)` | same_as(W) | 一维，ND 连续；`source_output_shape=(m,)` 对应 `(R,)` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `Wf` | `(block_M, block_K)` | float32 | UB（fragment） | W tile（fp32；含 K 尾零填充区） |
| `xr` | `(1, block_K)` | float32 | UB（shared） | x chunk（fp32；含 K 尾零填充区） |
| `xb` | `(block_M, block_K)` | float32 | UB（shared） | x 的广播副本 `(1,BK)→(BM,BK)` |
| `acc` | `(block_M, 1)` | float32 | UB（fragment） | 行和累加器（chunk 间 `clear=False` 累加） |
| `ou` | `(block_M,)` | dtype | UB（shared） | 输出中转（单次 VCast 后 UB→GM） |

### 4.4 内存搬运路径

```
GM[W] --T.copy(MTE2, fp16→fp32 VCast)--> UB Wf(block_M, block_K) fp32 ---┐
GM[x] --T.copy(MTE2, fp16→fp32 VCast)--> UB xr(1, block_K) fp32 --vbrc--> xb(block_M, block_K) fp32
                                                                        │
                              T.vmul(Wf, xb, Wf) fp32  <---------------┘
                              T.reduce_sum(Wf, acc, dim=-1, clear=False) fp32   (K 流水 num_stages=S)
                                                                        │
                       T.Parallel cast → ou(block_M) dtype → T.copy(UB→GM, 动态 extent 守卫) --> GM[y]
宿主边界：host 仅做 (R,K) 推断、kernel 选择与 shape/stride 元数据（含 flatten 视图），不触碰张量内容、不 padding。
```

### 4.5 UB 内存预算

> 目标平台 UB = 192KB（`CONST-capacity-910B2C`，Ascend910B2C）。按 `block_M=64, block_K=256`（`@T.Pipelined num_stages=2`）评估。

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `Wf` | (64, 256) | fp32 | 65536 |
| `xr` | (1, 256) | fp32 | 1024 |
| `xb` | (64, 256) | fp32 | 65536 |
| `acc` | (64, 1) | fp32 | 256 |
| `ou` | (64,) | fp16 | 128 |
| **总计（手工核算）** | | | **132480 / 196608 (192KB)** ≈ 67% |

说明：按 `CONST-capacity-910B2C`（2026-09-15 重校）auto-multi-buffer 膨胀系数 ~1.10–1.12（显式 alloc 结构下 ≈1.0），
132480×1.12 ≈ 148KB < 192KB ✅；`TRAP-UB-multibuffer-inflation` 记录的最坏 ~1.7× 会到 225KB（超限）——**Stage 3 首验证
必须实测编译占用，若报 UB 溢出则降 `block_K=128`**（`block_M=32, block_K=256` 配置实测亦通过，UB ≈ 66KB，见 probe #1b
`sweep`）。设计期实测已确认 `(block_M=64, block_K=256, num_stages=2)` 与 `(32,256,2)`/`(32,512,2)` 在本机编译通过。

### 4.6 动态轴定义

首次设计按 manifest 的 `R`/`K` 编译期特化（每 shape 一个 kernel，`lru_cache` 缓存），**无运行时动态轴**
（源 `_gemv_kernel` 亦是每 `(n,k)` 特化）。若需动态 shape（`T.symbolic`）参考 `examples/gemm/matmul_dynamic_shape.py`，
本任务 manifest 为静态 shape，不引入。

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | — |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemv_kernel(R, K, dtype="float16", block_M=64, block_K=256, num_stages=2):
    ...
```

（去除源 GPU 的 `compile_flags=["-O3","-DENABLE_BF16"]` 与 `get_sm_version` 分支——npuir 目标不需要；若 Stage 3 遇到
codegen 问题再按 `tilelang-error-fixer` 加最小 `pass_configs`。**必须显式 Developer 模式**，见 §2.2。）

**wrapper 契约对齐（Stage 3 注意）**：NPU wrapper 的 `GemvKernel.default_config` 当前为 GPU 遗留键
`{"block_n": 8, "reduce_threads": 32, "num_stages": 2}`，而本 NPU kernel 的 tiling 参数为
`{"block_M", "block_K", "num_stages"}`。Stage 3 需按 K8/K9 的口径把 wrapper 的 config 键重映射为 NPU 键
（`block_n`→`block_M`、`reduce_threads`→`block_K`，或直接改成新键），并保持 `forward` 调用编译产物的方式不变；
`reduce_threads` 在 NPU 无 thread 语义（`TRAP-threads-kwarg-noop`），不再作为真实 tiling 参数。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（仅 `T.copy` + `T.vbrc` + `T.vmul` + `T.reduce_sum` + `T.clear`，无 `T.gemm`）。

**判定依据**: 算子仅含逐元素乘、广播与沿 K 的归约，无矩阵乘；承载单元为 U(V) 向量单元（`VReduceOp`/`VMulOp`/`VBrcOp`）。
设计期实测确认 Cube 分形路径更慢（§0.6 R3），故排除 Cube。

### 5.2 Block 划分

```python
block_M = 64    # 输出行块（行分核粒度）；UB 内 Wf+xb 双 fp32 tile 预算；probe #1b 实测 0.188ms(7168²) 最优档
block_K = 256   # K 分块（归约内层向量维）；fp32 向量宽 ×4 / fp16 ×8 整数倍；K 尾块由动态 extent 处理
num_stages = 2  # T.Pipelined 双缓冲（UB 预算内，见 §4.5）
num_launched = ceildiv(R, block_M)   # 非 persistent：网格数 == 逻辑块数
```

> 启发式：`(block_M=32, block_K=512)` 与 `(block_M=64, block_K=256)` 实测并列最优（0.189 / 0.188 ms @7168²）；
> `block_M=32, block_K=256`（UB ≈66KB）为低 UB 回退档；shape 特化选择由 `default_config` 之外的启发式按 R/K 决定
> （Stage 4 扫描）。`block_K` 应为向量宽度整数倍（fp32 ×4、fp16 ×8），`block_M` 无分形下限（非 Cube）。

### 5.3 约束分析

- **对齐约束**: `block_K=256` 为 fp32（×4）与 fp16/bf16（×8）向量宽整数倍 ⇒ 无尾 lane 浪费；K 非 256 倍数时尾块
  用动态 extent（不产生 lane 浪费）。
- **UB 容量**: 手工 ~132.5KB（×1.12 ≈ 148KB）< 192KB ✅（§4.5；低 UB 回退档 `(32,256)` ≈66KB）。
- **L0 容量**: 无 Cube 计算，不适用。
- **归约轴**: K 为最内连续轴 ⇒ `T.reduce_sum(dim=-1)` 走快路径（`CONST-reduce-dims0-skinny`）。
- **`T.reduce_sum` dtype**: src/dst 必须**同 dtype**（本设计均 fp32）；**禁止** fp16 src → fp32 dst 的混合 dtype
  （`TRAP-reduce-mixed-dtype-silent-corrupt`：静默产出哨兵级垃圾值），故 W/x 先经 `T.copy` cast 到 fp32 再 reduce。

### 5.4 注意事项

- **非整除 R**：`rm = T.min(block_M, R - r0)`，动态 extent 装载/写回；用 `T.ceildiv` 避免 `R < block_M` 时零 block 启动。
- **非整除 K**：`rk = T.min(block_K, K - k0)`，尾块先 `T.clear(Wf)/T.clear(xr)` 再写有效区（**双侧**，§0.6 R4/E1）。
- **尾块的 `T.Parallel` cast**：`ou[i] = T.cast(acc[i,0], dtype)` 对 `block_M` 全部元素执行（含 `i ≥ rm` 的垃圾行），
  但写回只覆盖 `ou[0:rm]`，故无越界、无错误输出（垃圾行不写回）。
- **K 流水循环体内的编译期条件**：`if rk < block_K` 应在工厂层提升为 Python 分支（§3.3 说明，`CG-2026-0001`）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1；依据 `docs/开发指南.md` §3.3。

- **物理核数**: `NPUUtils.get().get_aicore_num()` **实查值 = 20**；本算子为**纯 Vector**，按标准「纯 Vector 算子核数
  翻倍」⇒ **物理核数 = 40**。查询代码与实际执行：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  print(NPUUtils.get().get_aicore_num())   # 本机实查输出: 20  ⇒ 纯 Vector 物理核数 = 40
  ```
  （设备：Ascend910B2C，2026-10-13 实查。注：`CONST-aicore-910B2C` 记录 910B2C 为 24 核，与实查 20 不一致——
  以实查为准，见迁移复盘。）
- **逻辑核数**: `num_logical = ceil(R / block_M)`（一维输出行网格）。按 `block_M=64` 对源测试/边界档计算：
  `R=1024 → 16`；`R=3000 → 47`；`R=3001 → 47`；`R=7168 → 112`；`R=18432 → 288`。
- **规模判定**: 三档并存（物理核数 40）：
  1. **小规模**（`num_logical ≤ 40`，如 R=1024→16）→ 逻辑核数 ≤ 物理核数，**无需适配**；启动 16 个核。
  2. **中等规模**（`40 < num_logical ≲ 120`，如 R=3000/3001→47、R=7168→112）→ 通过调整 `block_M` 使内核总数
     接近 40 的整数倍（按实查物理核数取 1×/2×/3× = 40/80/120）：如 R=3000 取 `block_M=80` → 38 ≈ 40；
     R=7168 取 `block_M=96` → 75 ≈ 80。取值在 Stage 3/4 实测确定，原则是内核总数落在 40 的整数倍附近，消除
     「41 个内核→某核双倍任务」式负载不均。
  3. **极大规模**（`num_logical ≫ 40`，如 R=18432→288）→ 无法通过 block 缩减到物理核数整数倍（`block_M` 过大会
     超 UB）。标准方案是**固定启动核数 = 物理核数、核内 `T.serial` 串行处理多逻辑块**（循环边界静态：
     `num_local_tasks = T.ceildiv(num_logical - kid, 40)`，`num_logical`/`40` 均编译期常量）。
- **分核方案（主选，可实施）**:
  - **网格数不变量**：**非 persistent 一维网格，`num_launched = num_logical`（恒等，禁止 `min(...)` 截断）**——
    否则 `num_logical > 40` 时只启动 40 个核、其余输出行**静默漏写**（本 manifest 的 R=3000~18432 全部 > 40，
    Stage 3 首个精度用例即失败）。本设计核内**无**多任务 persistent 循环，故必须网格数 == 逻辑块数。
  - `num_logical ≤ 40`：一核一行块，无负载不均、无超发。
  - `num_logical > 40`：启动 `num_logical` 个核（非 persistent），超出物理核数的部分由运行时串行调度；结合
    要素③的中等规模档微调 `block_M` 使 `num_logical` 贴近 40 的整数倍。
  - **persistent 备选（Stage 4 A/B）**：若实测超发调度开销显著（R=18432 档），可切「固定 40 核 + 核内 `T.serial`
    静态边界」persistent 形态。**注意** `TRAP-DEVMODE-PERSIST-GEMM` 只覆盖 Developer + persistent + `T.gemm`
    （本算子**无 gemm**，理论不受该 trap 约束），但该组合对本算子**未实证**——启用前须 smoke 验证，并准备回退。
  - 备选 **split-K**：仅当 `R ≪ block_M`（行并行度不足）且 K 极大时启用（本 manifest R≥1024，不触发）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化（`T.copy`/`T.vbrc`/`T.vmul`/`T.reduce_sum`/`T.clear`）；下表循环仅指 block 级 /
> tile 级调度循环，**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 输出行（R） | block 级并行 | `T.Kernel(num_launched, is_npu=True)` | 每个 slot 处理一个行块（§0.6 R2） |
| K 方向 | tile 级流水迭代 | `for kk in T.Pipelined(ceildiv(K,block_K), num_stages=2)` | K 分块累加，MTE2 装载与 Vector 计算重叠（§0.6 R1） |
| tile 内元素 | 向量化（Vector 单元） | `T.copy`（装载/写回）、`T.vbrc`（广播）、`T.vmul`（乘）、`T.reduce_sum`（归约）、`T.clear`（零填） | §1.6.2；无标量逐元素循环 |
| 输出 cast | 向量化（`T.Parallel` 即向量 lane） | `for i in T.Parallel(block_M): ou[i] = T.cast(acc[i,0], dtype)` | §1.6.2 #5（`T.Parallel` 在 npuir 映射为向量 lane） |

### 6.2 循环伪代码

```python
# num_launched = num_logical（§5.5 网格数不变量；非 persistent，禁止 min 截断）
with T.Kernel(num_launched, is_npu=True) as (cid, _):
    r0 = cid * block_M
    rm = T.min(block_M, R - r0)
    Wf = T.alloc_fragment((block_M, block_K), "float32")
    xr = T.alloc_shared((1, block_K), "float32")
    xb = T.alloc_shared((block_M, block_K), "float32")
    acc = T.alloc_fragment((block_M, 1), "float32")
    ou = T.alloc_shared((block_M,), dtype)
    T.clear(acc)
    for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
        k0 = kk * block_K
        rk = T.min(block_K, K - k0)
        if rk < block_K:            # 工厂层提升为 Python 分支（§3.3）
            T.clear(Wf)
            T.clear(xr)
        T.copy(W[r0:r0 + rm, k0:k0 + rk], Wf[0:rm, 0:rk])
        T.copy(x[k0:k0 + rk], xr[0, 0:rk])
        T.vbrc(xr, xb)
        T.vmul(Wf, xb, Wf)
        T.reduce_sum(Wf, acc, dim=-1, clear=False)
    for i in T.Parallel(block_M):
        ou[i] = T.cast(acc[i, 0], dtype)
    T.copy(ou[0:rm], y[r0:r0 + rm])
```

### 6.3 流水线优化

使用 `T.Pipelined(k_iters, num_stages=2)`：编译器自动生成 MTE2 装载与 Vector 计算的**多缓冲流水**，把 W/x tile
预取与 `vbrc`/`vmul`/`reduce_sum` 重叠。`num_stages=2` 依据 §4.5 UB 预算（132.5KB 手工核算，×1.12 ≈ 148KB < 192KB）。
`k_iters = ceil(K/block_K)`：K=1024→4、K=3000→12、K=7168→28，均 ≥4，流水可填充。**注意**：probe #1b 实测
`num_stages=3` 与 `2` 无差异（0.211 vs 0.210 ms @7168²，`block_M=32,block_K=256`）——本算子**发射受限**（§1.6.0 D-2），
流水深度非主要杠杆，`num_stages=2` 足够。

### 6.4 尾块处理

- **R 尾块**：动态 extent 切片（`rm`）装载与写回（`T.copy.md` §2.2.2 规则 2）；写回只覆盖 `y[r0:r0+rm]`，不写坏相邻
  GM（避 `TRAP-T-copy-region-semantics` 的越界写）。
- **K 尾块**：`rk < block_K` 时先 `T.clear(Wf)`、`T.clear(xr)` 的尾区再装载，保证 `k≥K` 项贡献 0（§0.6 R4，E1）。
  本 manifest 的 `gcd(K)=1024`，`block_K=256` 对 K=1024/7168/16384 整除（K=3000 不整除，触发尾块处理）。
- **对齐填充**：`block_K=256` 为向量宽整数倍，规避尾 lane 浪费；UB tile 静态 shape（Developer 要求，`T.copy.md` §2.2.2 规则 4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式: 自动同步**（Developer 模式）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| K 流水 MTE2↔Vector | 由 `T.Pipelined` 编译器自动插入多缓冲同步 | Developer 模式自动同步；替代源 GPU 的手工流水（§0.5 #3） |
| `Wf`/`xr` 装载 → `vbrc`/`vmul`/`reduce_sum` | 由编译器自动插入依赖同步 | `T.copy` GM→UB 与向量算子间由框架管依赖 |
| `acc` 跨 chunk 累加（`clear=False`） | 由编译器自动维护循环携带依赖 | 同一 fragment 的 read-modify-write |
| UB→GM 写回 | 由 `T.copy` 自动插入依赖同步 | 保证 cast 完成后才写出 |
| 跨核 | 无需 | 每个行块独立，无跨核数据交换（无 split-K） |

无手动同步点（无 `T.barrier_all`/`T.sync_block_set/wait`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # Developer 模式默认；源 GPU 的 compile_flags / disable_tma 不适用 npuir
```

若 Stage 3 出现 codegen 问题，按 `tilelang-error-fixer` 最小化加入，不在设计期预设。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓 `GemmTest.ref_program` 的 GEMV 路径），
> **不得复刻** §0.6 的 NPU 算法——保证验证独立性。

```python
import torch

def golden_gemv(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """基于 PyTorch 的参考实现（源算子语义）：y = W @ x。

    w: (R, K)  矩阵（源 kernel 的 b，或 rhs_col 的 a）
    x: (K,)    向量（源 kernel 的 a，或 rhs_col 的 b）
    return: (R,)，dtype == w.dtype

    与源 GemmOp 语义一致：GEMV 是 GEMM 的 m==1（lhs_row）或 n==1（rhs_col）特例，
    源 ref_program 为 torch.matmul（此处即 matmul(w, x)，无需转置）。
    """
    acc = torch.matmul(w.float(), x.float())      # fp32 累加（源 accum_dtype="float"）
    return acc.to(w.dtype)                         # 单次舍入到输入 dtype
```

（等价于 `examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmTest.ref_program` 在 GEMV dispatch 下的展开；
输入生成见 `examples/TileOPs/tileops/workloads/gemm.py` 的 `GemmWorkload.gen_inputs`。**注意**
`TRAP-fp16-opmath-golden`：golden 以 fp32 中转链对齐 torch CPU opmath 域。）

### 8.2 精度标准

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| float16（GEMV 档） | 1e-2 | 1e-2 | `examples/TileOPs/tests/ops/test_gemm.py` 的 `test_gemv_boundary_*` |
| bfloat16（GEMV 档） | 1.6e-2 | 1.6e-2 | 同上（`GemvBoundaryFixture` bf16 档；与 `test_gemm` bf16 容差一致） |
| float16（通用 GEMM 档） | 1e-3 | 1e-3 | `test_gemm`（非本任务） |
| 通用参照 | 1e-2 | 1e-2 | `docs/开发指南.md` §3 示例 |

> 累加域为 fp32、输出单次舍入（§1.6.1 E3），误差来自 fp32→fp16/bf16 舍入与 K 分块加法分组，落在上述容差内。
> 设计期 probe 实测（`7168×7168` fp16，golden 用 `matmul(w.float(), x.float())`）绝对差 6.25e-2，对应相对误差
> ~7e-4 ≪ rtol=1e-2，符合预期。

### 8.3 L0 门槛测试计划（交 `tilelang-op-develop` 执行）

| 档位 | shape (R,K,dtype) | 目的 |
|------|-------------------|------|
| smoke | 1024,1024,float16 | 最小基线（编译通过 + index 等价性） |
| smoke | 1024,1024,bfloat16 | bf16 路径（fp32 中转 + 单次舍入） |
| GEMV 边界 | 3000,1024,float16 | `lhs_row` 非对齐 R（`GemvBoundaryFixture`） |
| GEMV 边界 | 3000,1024,bfloat16 | `lhs_row` 非对齐 R（bf16） |
| GEMV 边界 | 1024,3000,float16 | 非对齐 K（触发 K 尾块双侧清零） |
| GEMV 边界 | 3001,1024,float16 | `rhs_col` 非对齐 R（奇数 R 尾块） |
| 大 R | 7168,7168,float16 | 大 shape（发射/带宽主导） |
| 大 R | 18432,7168,bfloat16 | 极大规模分核（`num_logical=288 ≫ 40`） |
| R 极小 | 1,1024,float16 | `R < block_M` 单行（`rm=1`，单块启动） |
| K 极小 | 1024,16,float16 | `K < block_K` 单 chunk（`rk=16`，尾块清零路径） |

- 每档：`torch.testing.assert_close(y, golden_gemv(w, x), atol/rtol 见 §8.2)`。
- 完整分层套件（L1/L2/Boundary）交 Stage 3 的 `tilelang-op-develop`，此处不枚举。

### 8.4 CV 融合设计

**无需用户手写 CV 融合**：本算子为**纯 Vector**（无 `T.gemm`），全部计算（装载/广播/乘/归约/cast/写回）在 Vector
单元内完成，无 Cube↔Vector 跨引擎协同需求，无需 `Scope("Cube")/Scope("Vector")` 双 Scope 与 `sync_block_set/wait`。
输出 dtype 转换由 `T.copy` 的自动 VCast（`hivm::VCastOp`，RINT）承载，用户不感知。若未来引入 Cube 备选形态
（§1.6.3 备选 #4）才涉及 Cube→Vector 协同，届时按 `tilelang-mixcv-skill` 评估。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **Developer 模式必须显式设置**（设计期实测发现）：TileLang NPUIR 默认 `TILELANG_ASCEND_MODE` ≠ Developer；未显式
   设置时 `T.copy` 跨 dtype 报 `T.copy does not support element type casting`、`alloc_shared` 落 `cbuf` 导致
   `hivm.hir.store only support ub to gm`。**本算子为 Developer 选型，kernel 源码/编译环境必须显式 Developer**。
   证据：`examples/gemm/_gemv_kernel/history_version/design_probe_gemv_cases.py`（Developer 下 4/4 OK；默认模式下
   `cube_noidx`/`reduce`/`vbrc` 均失败）。
2. **UB 容量**：`(block_M=64, block_K=256)` 手工 ~132.5KB；按 `TRAP-UB-multibuffer-inflation` 的 ~1.7× 最坏会到
   ~225KB 超限。设计期实测本机编译通过，但 Stage 3 首验证须核对实际占用；溢出则降 `block_K=128` 或用低 UB 档
   `(block_M=32, block_K=256)`。
3. **R/K 非整除**：R 尾块动态 extent；K 尾块**双侧清零**（W、x 都清，否则 `0×NaN` 污染）；**禁止 host padding**
   （`ascend-constraints.md` §4）；`T.ceildiv` 避免零 block。
4. **`T.reduce_sum` dtype**：src/dst 必须**同 dtype**；**禁止** fp16 src → fp32 dst（`TRAP-reduce-mixed-dtype-silent-corrupt`：
   静默产出哨兵级垃圾值）。本设计先 `T.copy` cast 到 fp32 再 reduce（同 dtype fp32）。
5. **`clear=False` 语义**：`T.reduce_sum(clear=False)` 在 dst 现有值上累加，**调用前必须初始化 dst**
   （`T.reduce_sum.md` §2.3；对未初始化 buffer 使用不报错但静默数值错误）——本设计循环前 `T.clear(acc)`。
6. **fp32 累加**：源明确 fp32（`accum_dtype="float"`）；`T.vmul`/`T.reduce_sum` 均在 fp32 域，杜绝 fp16 长 K 累加误差。
7. **物理核数**：实查 20（本机），纯 Vector 翻倍 = 40；若换设备须重查（`NPUUtils.get().get_aicore_num()`），分核方案随核数变化。
8. **`T.Pipelined` 体内运行时谓词**：`if rk < block_K` 须提升为工厂层 Python 分支（`CG-2026-0001` Developer codegen 缺口），§3.3 已注。
9. **wrapper config 键不一致**：NPU wrapper `GemvKernel.default_config` 仍是 GPU 遗留键（`block_n`/`reduce_threads`），
   与 NPU tiling 键（`block_M`/`block_K`/`num_stages`）不一致，Stage 3 需重映射（§4.7）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | `block_M/block_K` 过大或 auto-multi-buffer 膨胀 | 编译失败（`ub overflow`） | 降 `block_K`/`block_M`（`CONST-capacity-910B2C`） |
| 默认模式非 Developer | 未设 `TILELANG_ASCEND_MODE` | `T.copy` 跨 dtype 报错 / `hfivm.hir.store` 只支持 ub→gm | 显式 Developer（§9.1.1） |
| K 尾污染 | `block_K ∤ K` 且未清零 K 尾 | 输出被垃圾乘积污染 | `T.clear(Wf)` + `T.clear(xr)`（**双侧**，E1） |
| 0×NaN | K 尾只清一侧且另一侧为 NaN | NaN 污染全部输出行 | W、x 双侧清零 |
| 归约混合 dtype | fp16 src → fp32 dst 的 `T.reduce_sum` | 静默哨兵级垃圾值 | 先 `T.copy` cast 到 fp32，再同 dtype reduce |
| `clear=False` 未初始化 | 直接 `reduce_sum(clear=False)` | 静默数值错误 | 循环前 `T.clear(acc)` |
| 漏写输出行 | 网格数用 `min(num_logical, num_physical)` 截断 | 逻辑块数 > 40 时静默漏算 | `num_launched = num_logical`（§5.5） |
| `threads=` 无效 | 照抄源 `threads=(...)` | 无效果（`TRAP-threads-kwarg-noop`） | 一维 `T.Kernel` + 向量 lane |

### 9.3 特殊场景处理

- **R 极小（R=1）**：`num_logical=1`，`rm=1`；单块启动，`block_M=64` 中 63 行为垃圾但不写回。可选启发式降 `block_M`
  到 16（Stage 3/4）。
- **K 极小（K<block_K）**：单 chunk，`rk=K<block_K` 触发尾块清零路径；`k_iters=1`，流水退化为单次。
- **K 非整除（K=3000）**：尾块 `rk=3000 mod 256=184`，双侧清零后贡献 0（E1 机器验证）。
- **R 极大规模（R=18432）**：`num_logical=288 ≫ 40`，非 persistent 超发调度；persistent 备选见 §5.5（未实证）。
- **bf16 输入**：`T.copy` 自动 cast 到 fp32（`T.vmul`/`T.reduce_sum` 无 bf16 支持，见 `T.vmul.md`/`T.reduce_sum.md`
  的 dtype 表），全部计算在 fp32 域，输出单次 cast 回 bf16。
- **`lhs_row` vs `rhs_col`**：本 kernel 只实现核心 `y=W·x`；`lhs_row`/`rhs_col` 的输入/输出 reshape 由 `GemmOp.forward`
  负责（源仓 ops 层，不在本 kernel 内）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gemm/_gemv_kernel/
├── DESIGN.md               # 本设计文档
├── RETROSPECTIVE.md        # Stage 1 复盘
├── verify_equiv.py         # D-1 等价性机器验证脚本（§1.6.1）
├── history_version/        # revision 备份 + 设计期探针（D-3）
│   ├── design_probe_gemv.py         # 算法族 A/B 实测（Vector vs Cube）
│   ├── design_probe_gemv_cubeT.py   # Cube b_transpose 形态实测
│   ├── design_probe_gemv_sweep.py   # Vector config 扫描
│   ├── design_probe_gemv_apis.py    # API/结构可行性
│   └── design_probe_gemv_cases.py   # 逐结构单进程复现
└── _gemv_kernel.py         # 算子实现（Stage 3 产出，当前尚未生成）
```

> 项目名 `gemm`（由 conductor 传入）；迁移 harness 模式逐函数落 `examples/{op_slug}/{func}/`。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 E1/E2/E3 等价性验证（6/6 `EQUIV_PASS`） |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘（自进化钩子） |
| `history_version/design_probe_gemv*.py` | 已完成 | 设计期探针（D-3，实测数据源） |
| `_gemv_kernel.py` | 待实现（Stage 3） | NPU Developer 模式 Vector GEMV kernel + golden |

### 10.3 命名规范

- 项目目录: `gemm`；算子/函数目录: `_gemv_kernel`；实现文件: `_gemv_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）+ 等价性验证（`verify_equiv.py`）+ 设计期探针（算法族 A/B 实测）
2. ⬜ Golden 函数（`golden_gemv`，以 §0.1 语义）
3. ⬜ 算子实现（`_gemv_kernel.py`，显式 Developer 模式）+ §8.3 L0 门槛精度比对
4. ⬜ 上层 wrapper（`GemvKernel`）config 键重映射（§4.7）并替换 GPU 工厂调用
