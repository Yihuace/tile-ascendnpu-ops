# _bmm_kernel 算子设计文档

> **迁移任务**（harness，developer 模式）。项目：`bmm`；算子目录：`examples/bmm/_bmm_kernel/`。
> 源算子（GPU）：`/tmp/opencode/TileOPs-ref/tileops/kernels/bmm.py` 的 `_bmm_kernel`（工厂）+ `BmmKernel`，
> 以及 op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/bmm.py` 的 `BmmFwdOp`。
> 本仓 Stage 0 抽取件：`examples/TileOPs/tileops/kernels/bmm/bmm/_bmm_fwd_kernels.py`。
> NPU 规格权威：`examples/TileOPs/tileops/manifest/bmm.yaml`（`BmmFwdOp` 段）。
> 工具链版本戳（本设计全部实测/探针结论绑定此版本）：tilelang `0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b`
> / CANN `9.0.0`（`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0`）/ Ascend910B2C（`get_aicore_num()` 实查 = 20）。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`BmmFwdOp` / `_bmm_kernel` 是 **批量稠密矩阵乘**（`torch.bmm` 语义），严格 3D-3D，**无广播**：

$$
C[b] = A[b] \cdot B[b],\quad b = 0,\dots,B-1
$$

- 逐 batch 相互独立，无跨 batch 归约/耦合。
- **规约语义**：`C[b,i,j] = Σ_{k=0}^{K-1} A[b,i,k]·B[b,k,j]`；累加顺序为沿 K 维分块（`block_k`）顺序累加，累加器为 **fp32**（`accum_dtype="float"`），无跨 block 归并（每个输出 tile 由一个 block 独立算完）。
- **dtype 语义**：输入 fp16 / bf16，累加 fp32；输出 dtype 与输入相同（`same_as(a)`）。输出由 `out_idx=[-1]` 由框架分配。
- **边界语义**：
  - batch 维必须相等、K 维必须相等（否则 host 侧 `_infer_bmnk` 抛 `ValueError`）；
  - `K % 16 == 0` 为 manifest `shape_rules` 与 `BmmKernel.__init__` 的硬前置（GPU 侧为 WGMMA K-step 对齐）；
  - **M/N 允许非整除 block**（源码 docstring："the epilogue guards the M/N tails so m/n need not be multiples of the block sizes"，NPU 测试 fixture `full-fp16-b4-mn-nonaligned` = (4,200,300,128) 固化此语义）；
  - NaN/±inf 按 IEEE 矩阵乘传播（无 clamp/稳定化改写）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（`torch.bmm`）。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `a` | 输入 | `(B, M, K)` | float16 / bfloat16 | 批量左矩阵；K 连续（row-major） |
| `b` | 输入 | `(B, K, N)` | same_as(a) | 批量右矩阵；N 连续（NT/NN 中的 **NN** 布局，`transpose_B=False`） |
| `c` | 输出 | `(B, M, N)` | same_as(a) | 批量输出（迁移必需输出 shape `(B, M, N)`） |

### 0.3 实现算法解读（怎么算）

源文件 `examples/TileOPs/tileops/kernels/bmm/bmm/_bmm_fwd_kernels.py` L15–L111（=`_bmm_kernel`）。
工厂 `_bmm_kernel(batch,m,n,k,dtype)` → `@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["-O3","-DENABLE_BF16"])`
→ `_bmm_func(block_m=128, block_n=128, block_k=64, num_stages=3, threads=128)` → `_bmm_main` prim_func。

**计算步骤分解**（源码每个计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.clear(c_local)` | — | `c_local`(fragment, fp32, block_m×block_n) | 累加器清零 |
| 2 | 3D grid 启动 `T.Kernel(ceildiv(n,bn), ceildiv(m,bm), batch)` as `(bx,by,bz)` | — | 每 block 一个 (M,N,batch) tile | 输出 tile 划分 |
| 3 | `T.use_swizzle(10, enable=True)` | — | L2 rasterization 顺序 | 与数学语义无关（仅访问顺序） |
| 4 | `for ki in T.Pipelined(ceildiv(k,block_k), num_stages)`: `T.copy(a[bz, m0:m0+bm, ks:ks+bk], a_smem)`、`T.copy(b[bz, ks:ks+bk, n0:n0+bn], b_smem)` | global A/B tile | `a_smem`/`b_smem`(shared, fp16/bf16) | 操作数装载 |
| 5 | `T.gemm(a_smem, b_smem, c_local, policy=FullRow)` | `a_smem`,`b_smem` | `c_local`(累加) | `C += A_tile·B_tile`（K 分块累加项） |
| 6 | `T.copy(c_local, c_smem)` | `c_local` fp32 | `c_smem`(shared, dtype) | epilogue SMEM 中转 + fp32→fp16 cast |
| 7 | `for i,j in T.Parallel(bm,bn): if m0+i<m and n0+j<n: c[bz,m0+i,n0+j]=c_smem[i,j]` | `c_smem` | global `c` | 尾块预测写回 |

**数据流与内存访问模式**（源硬件视角）：

```
GM a[bz] --T.copy--> SMEM[a_smem] --+
GM b[bz] --T.copy--> SMEM[b_smem] --+--> wgmma(fragment) --> c_local(fp32, register)
                                                              |
                                              T.copy --> c_smem(SMEM)
                                                              |
                          T.Parallel + 尾块谓词 --> GM c[bz]
```

**循环与并行结构**：grid = `(ceildiv(N,bn), ceildiv(M,bm), batch)`（3 维，共 `ceil(N/bn)*ceil(M/bm)*B` 个 CTA）；每 CTA 128 线程（1 warpgroup）；K 维 `T.Pipelined(num_stages=3)` 流水；`policy=GemmWarpPolicy.FullRow` 将 (M,N) fragment 按行分给 warp。

**host 侧逻辑**：`BmmFwdOp.forward` 校验 rank/batch/K 相等与 `K%16`、推导 `(B,M,N,K)`、按 `(batch,m,n,k,dtype)` lazy 建/缓存 kernel；`_bmm_kernel` 带 `functools.lru_cache` shape 特化；`BmmKernel` 维护 `default_config`/`autotune_configs`。GPU 全部 host 逻辑（含 `torch.library.custom_op` 包装）在 NPU 侧由 `examples/TileOPs/tileops/ops/bmm/bmm.py` + `.../kernels/bmm/bmm/bmm.py` 承接（Stage 0 已完成 op 类移植，K5/K7/K8/K9 适配）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`a_smem`/`b_smem`） | 减少 GM 重复访问 | A/B tile 驻留共享内存，供 K 循环与 warp 复用 | shared memory | 硬件强相关（GPU SMEM → NPU L1） |
| 2 | `T.Pipelined` K 循环（`num_stages=3`） | 搬运与计算重叠 | DMA 双缓冲流水 | 异步拷贝引擎 | 可移植（NPU `T.Pipelined`） |
| 3 | WGMMA + fp32 累加器（`policy=FullRow`） | 高吞吐张量乘 | Hopper 异步 warpgroup MMA | Tensor Core / WGMMA | 硬件强相关（NPU 等价 = Cube MMA） |
| 4 | L2 rasterization（`T.use_swizzle(10)`） | 提高 A/B 行在 L2 的复用 | 重排 (bx,by) 遍历顺序成面板 | GPU L2 缓存几何 | 硬件强相关且 NPU 无对应 API |
| 5 | SMEM swizzle（`make_swizzled_layout`） | 消 SMEM bank conflict | 重排共享内存布局 | GPU SMEM bank 结构 | 硬件强相关（NPU 由 NZ 分形通路隐式处理） |
| 6 | epilogue SMEM 中转（`c_smem`） | 谓词写仍保持合并访问 | fp32 fragment→SMEM→谓词化 GMEM 写 | SMEM + coalescing | 硬件强相关（NPU：L0C→GM 直写 + 编译器自动 VCast 量化） |
| 7 | TMA + warp specialization | 生产者/消费者分离 | 1 producer WG 发 TMA，consumer 跑 WGMMA | Hopper TMA / warpgroup | 硬件强相关；源码已 `disable_warp_specialized` |
| 8 | 线程数 `threads=128` | 映射 CUDA 线程/ warpgroup | CUDA blockDim | CUDA thread model | 硬件强相关（NPU 无效果） |
| 9 | M/N 尾块谓词保护 | 支持非整除 shape | 逐元素 if 写 | predicated store | 可移植（需换形态） |
| 10 | `out_idx=[-1]` + `lru_cache` shape 特化 | host 零手工分配 / 复用编译产物 | 框架分配输出 + 工厂缓存 | 无（host 层） | 可移植 |

> 识别不出来 ≠ 不存在。上表覆盖源码全部显式优化；未列出的（如编译器自动软件流水）由后端承担。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 0.1 批量矩阵乘语义 | 语义 | 无 | — | **保留** | 语义层无条件保留（golden 依据） |
| SMEM tiling（#1） | 优化 | shared memory | 有 | **等价替换** | `T.alloc_shared` 在 Cube 路径映射 L1（`docs/Tilelang.language/内存操作/T.alloc_shared.md` §1："Ascend中对应ub/L1上的内存"；L1 = 512KB，`CONST-capacity-910B2C`） |
| K 循环流水（#2） | 优化 | 异步拷贝引擎 | 有 | **等价替换** | `T.Pipelined(..., num_stages=...)`（`docs/开发指南.md` §3.2；`examples/gemm/example_gemm.py` L27 为 NPU 先例） |
| WGMMA + fp32 累加（#3） | 优化 | Hopper Tensor Core | 有 | **等价替换** | `T.gemm` → `hivm.hir.mmadL1`（`docs/Tilelang.language/线性代数操作/T.gemm.md` §3）；fp16 输入要求 dst/累加 fp32（同文档 §2.3） |
| L2 rasterization（#4） | 优化 | GPU L2 缓存几何 | 无对应 API | **舍弃** | NPU `T.Kernel` 无 swizzle 参数（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.1 参数表仅 blocks/threads/is_cpu/prelude/is_npu/pipeline）；L2 复用改由硬件调度器与 tile 遍历自然获得，无手写 rasterization 原语（负向断言依据为已核对的文档参数表） |
| SMEM swizzle（#5） | 优化 | SMEM bank 结构 | 由 NZ 分形隐式处理 | **舍弃** | `tilelang-cube-skill` references/nz-format.md："Developer mode kernels should keep ND layout... do not force NZ conversion"；ND→NZ 由 `T.copy`/`load_nd2nz` 后端完成 |
| epilogue SMEM 中转（#6） | 优化 | SMEM 中转 + coalescing | 有（换形态） | **重新设计** | → §0.6 R2：Developer `T.copy(L0C→GM)` **保持 ND**，fp32→fp16/bf16 由编译器自动插入 `hivm::VCastOp`(RoundMode=RINT) 完成（`docs/Tilelang.language/内存操作/T.copy.md` 支持方向表第 15 行「Developer `L0C→GM` 经 VCast 通用路径」+ §2.3 第 4 条）；消源方案显式 SMEM staging 缓冲 |
| TMA + warp specialization（#7） | 优化 | Hopper TMA / warpgroup | 无 | **舍弃** | 源码本已 `tl.disable_warp_specialized=True`；NPU 无 TMA/warpgroup 概念，`T.copy` 由 MTE2/MTE1 承担（`docs/Tilelang.language/内存操作/T.copy.md` §3） |
| `threads`（#8） | 优化 | CUDA thread model | 无效果 | **舍弃** | `TRAP-threads-kwarg-noop`（T.Kernel(threads=) 在 npuir 无效果，pattern-library traps-compiler.md，工具链版本戳见条目） |
| M/N 尾块谓词（#9） | 优化 | predicated store | 有（须换形态） | **重新设计** | → §0.6 R2：谓词 fragment→GM 直写在 npuir 崩溃（见 §0.6 证据），改 slice 形态动态 extent `T.copy` |
| 3D grid，`bz=batch` | 结构 | CUDA gridDim.z | **无**（NPU ≤ 2D block） | **重新设计** | → §0.6 R1 |
| host `lru_cache`/`out_idx`（#10） | 优化 | 无 | 有 | **保留** | `examples/TileOPs/tileops/kernels/bmm/bmm/bmm.py`（Stage 0 已移植；K8 去 autotune、K9 去 threads） |

> 每项处置均附依据（文档路径 + 条款 / pattern-library 条目 / examples 佐证）。舍弃项均写明理由（NPU 无对应 API 或语义无意义）。

### 0.6 NPU 算法重设计

#### 重设计项 R1：3D grid（batch 维）→ 一维折叠网格

- **源方案**：`T.Kernel(ceildiv(n,bn), ceildiv(m,bm), batch) as (bx,by,bz)`，把 batch 放到 `blockIdx.z`，一次 launch 跑完全部 batch。
- **NPU 新算法**：将 `(bz,by,bx)` 折叠进一维 `cid`：
  ```
  grid = B * ceil(M/block_M) * ceil(N/block_N)
  with T.Kernel(grid, is_npu=True) as (cid, _):
      bz = cid // (nt_m*nt_n); rem = cid % (nt_m*nt_n)
      by = rem // nt_n;        bx = rem % nt_n
  ```
  承接"单次 launch 跑完所有 batch"的性能意图；核内无 batch 串行依赖。
- **语义保持论证**：grid 折叠只是把同一组独立任务重新编号（双射 `cid ↔ (bz,by,bx)`），任务集合、每任务的计算与写回位置完全不变；数学上仍为 `C[b]=A[b]·B[b]`。边界语义（M/N 非整除）由每任务内 tile 边界处理，与 grid 维度无关。
- **依据**：探针实测 3D grid 被拒绝（见下"探针结论"），`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3"NPU 模式最多支持 2 个 block 维度"。

#### 重设计项 R2：epilogue 谓词 fragment→GM 直写 → 尾块安全的 slice 形态 T.copy（L0C→GM 直写，编译器自动 VCast 量化）

- **源方案**：`T.copy(c_local, c_smem)` 再 `for i,j in T.Parallel: if m0+i<m and n0+j<n: c[...]=c_smem[i,j]`——先把 fp32 fragment 搬进 SMEM，再做带谓词的 GM 写，保证谓词下仍合并。GPU 上实测比直接 fragment→GMEM 快 1–19%（源码注释）。
- **NPU 新算法**：去掉中间 SMEM 缓冲，直接
  ```
  tile_m = T.min(block_M, M - m0); tile_n = T.min(block_N, N - n0)
  T.copy(c_local[0:tile_m, 0:tile_n], c[bz, m0:m0+tile_m, n0:n0+tile_n])
  ```
  `T.copy(L0C → GM)` **保持 ND 布局**；fp32→fp16/bf16 量化由编译器自动插入 `hivm::VCastOp`（RoundMode=RINT，四舍五入到偶数）完成（`docs/Tilelang.language/内存操作/T.copy.md` 支持方向表第 15 行「Developer `L0C→GM` 经 VCast 通用路径」+ §2.3 第 4 条「Developer 跨 dtype 由编译器自动插入 `hivm::VCastOp`」）。**注**：`T.store_fixpipe`/`enable_nz2nd`/`load_nd2nz` 属 **Expert-only**，Developer 路径不使用。整块 tile 走无谓词快路径；尾块用切片动态 extent 精确裁剪，绝不写出源 tile 边界。
- **语义保持论证**：输出张量内容不变——同一 fp32 累加器做一次 fp32→目标 dtype 的舍入转换（转换次数与源一致，仅去掉中转缓冲），M/N 尾块写回区域由 `tile_m/tile_n` 精确限定（等同于源谓词条件）。IEEE 舍入路径不变：源 GPU 的 `T.copy(c_local, c_smem)`（fp32→fp16/bf16）与 NPU Developer 的 `T.copy(c_l, c)`（编译器插入 `hivm::VCastOp`，RINT）均对同一 fp32 累加器做一次 round-to-nearest-even 舍入，转换次数与舍入方向一致。边界核对：非整除 M/N 由 `T.min` 驱动、整除时 `tile_m=block_M, tile_n=block_N` 退化为整块拷贝；无 NaN/inf 特判（与源一致）。
- **依据（含实测证据）**：见下"探针结论"——谓词直写崩溃、slice 形态 `T.copy` 正确；CG-2026-0001(d)（`.agents/evolution/capability-gaps.md`）记录同形态"T.Parallel 体内条件读 fragment → 运行期 fixpipe 路径损坏 → aicore 崩溃（'The MPU address access is invalid'）"。

#### 重设计项 R3：累加器初始化 T.clear → T.gemm(initC=(k==0))

- **源方案**：`T.clear(c_local)` 后 `T.gemm(..., policy=FullRow)`（无条件累加），即 `c=0; for k: c += A_k·B_k`。
- **NPU 新算法**：删除 `T.clear`，改用 `T.gemm(a_s, b_s, c_l, initC=(k == 0))`（`k==0` 时 `dst=src1@src2`，否则累加，`T.gemm.md` §2.1）。等价形式 `c = A_0·B_0; for k>0: c += A_k·B_k`。
- **语义保持论证**：IEEE-754 下 `0 + x == x` 精确成立，故融合初始化与"清零后累加"逐位相同；K 分块累加顺序不变（仍按 ki 递增）。§1.6.1 OPT-1 机器验证（fp64 参照 + fp32 累加路径，含 inf/NaN/subnormal/边界 shape）max ulp 差 = 0。
- **依据**：`docs/开发指南.md` 迁移三处改动第 3 条（"用 initC 参数替代显式清零"）；`examples/gemm/example_gemm.py` L30。

#### 重设计项 R4：GPU 专属流水/布局手段的整体替换

- **源方案**：`T.use_swizzle`（L2 rasterization）+ `make_swizzled_layout`（SMEM bank）+ `threads` + warp-specialization/TMA + `compile_flags=["-O3","-DENABLE_BF16"]`。
- **NPU 新算法**：全部移除；保留并映射**唯一有意义的两个**——SMEM tiling → L1（Developer `T.alloc_shared`）与 K 流水（`T.Pipelined(num_stages)`）。**Developer 模式全程保持 ND**，Cube 所需的布局适配由后端在 `T.copy`/`T.gemm` lowering 中隐式完成（显式 ND↔NZ 属 Expert 路径）。
- **语义保持论证**：以上手段均不改变数学结果（仅访问顺序/布局/线程映射）。
- **依据**：`docs/开发指南.md` 迁移三处改动；`tilelang-cube-skill` nz-format.md；`TRAP-threads-kwarg-noop`；`T.Kernel.md` 参数表（无 swizzle）。

**迁移决策落地**：§1–§10 的设计**基于上述重设计后的 NPU 算法**（一维折叠网格 + L1 tiling + K 流水 + L0C→GM 直写（编译器自动 VCast 量化）），而非 GPU 源方案的直译。

#### 探针结论（设计期实测，D-3；脚本见 §10.3）

工具链：tilelang `0.1.2+417dae99` / CANN `9.0.0` / Ascend910B2C。预算：2 个探针脚本、每脚本 <10min。

| 探针 | 假设 | 结果 |
|------|------|------|
| A | NPU `T.Kernel` 不接受 3D grid | **REJECTED**（`AssertionError`：NPU kernel only supports up to 2D blocks）——与 `T.Kernel.md` §2.3 一致 |
| B | 折叠 batch 的一维 grid + Developer `T.copy`(3D 切片→2D shared) + `T.gemm(initC=(k==0))` 可用 | **PASS**（aligned fp16 b4-128，数值对齐 `torch.bmm`） |
| C | 尾块：谓词 fragment→GM 直写 | **运行时 aicore 异常**（`The MPU address access is invalid`，fixp_error）——与 CG-2026-0001(d) 同形态；**否决** |
| D | 尾块：slice 形态动态 extent `T.copy(c_l[0:tm,0:tn], c[bz, ...])` | **PASS**（non-aligned fp16 b4-200×300-128） |
| E | bf16 路径（对齐） | **PASS**（b4-128 bf16） |

探针脚本：
`examples/bmm/_bmm_kernel/history_version/design_probe_bmm.py`（探针 A + B/C 初判）、
`examples/bmm/_bmm_kernel/history_version/design_probe_bmm_v2.py`（探针 B/D/E 定稿形态）。

### 0.7 标杆实现

- **源算子路径**：`examples/TileOPs/tileops/kernels/bmm/bmm/_bmm_fwd_kernels.py`（GPU 抽取件）+ `/tmp/opencode/TileOPs-ref/tileops/kernels/bmm.py`（原始 GPU 仓库）。
- **参考实现（golden 语义）**：`torch.bmm`（`examples/TileOPs/tests/ops/test_bmm.py` L33 `ref_program`）。golden **直接复用 `torch.bmm`**，不复刻 NPU 算法（保证验证独立性）。
- **基准/工作负载**：`examples/TileOPs/tileops/manifest/bmm.yaml`（`BmmFwdOp` 段，4 条 workload），`examples/TileOPs/benchmarks/ops/bench_bmm.py`（manifest-driven，baseline = `torch.bmm`）。

---

## 1. 概述

### 1.1 算子名称

`_bmm_kernel`（`BmmFwdOp` / `torch.bmm` 语义）。

### 1.2 功能描述

批量稠密矩阵乘：对每个 batch 独立计算 `C[b] = A[b] @ B[b]`，输入 `(B,M,K)` / `(B,K,N)`，输出 `(B,M,N)`，fp16/bf16 输入 fp32 累加、输出同输入 dtype。

> **设计核心（GPU→NPU 关键重设计，2–4 条）**：
> 1. **3D grid（batch 维）→ 一维折叠网格**：NPU `T.Kernel` 最多 2D block，将 `(bz,by,bx)` 折叠进 `cid` 手动 decode（§0.6 R1）。
> 2. **epilogue：谓词 fragment→GM 直写 → 尾块安全的切片 `T.copy`**：改用 L0C→GM 直写（保持 ND；fp32→fp16/bf16 量化由编译器自动 `hivm::VCastOp`(RINT) 完成）+ 动态 extent 裁剪（§0.6 R2）。
> 3. **累加器初始化融合**：去 `T.clear` 改 `T.gemm(initC=(k==0))`，省一次 `block_M×block_N` fp32 清零写（§0.6 R3 / §1.6.1 OPT-1）。
> 4. **GPU 专属手段整体替换**：去 L2 rasterization（`T.use_swizzle`）/ SMEM swizzle / `threads` / TMA/warp-specialization / nvcc flags；保留并映射 SMEM tiling→L1、K 流水→`T.Pipelined`（§0.6 R4）。

### 1.3 数学公式

$$
C[b,i,j] = \sum_{k=0}^{K-1} A[b,i,k]\cdot B[b,k,j],\qquad b\in[0,B),\ i\in[0,M),\ j\in[0,N)
$$

### 1.4 算法描述

NPU 侧算法（= §0.6 重设计落地，与 GPU 源方案的差异均标注来源）：

1. **输出 tile 划分（R1）**：一维折叠网格 `grid = B·⌈M/block_M⌉·⌈N/block_N⌉`，核内 decode 出 `(bz, by, bx)`；每核负责一个 `(batch, M-tile, N-tile)` 的完整 GEMM。
2. **K 维分块累加（R3/R4）**：`for k in T.Pipelined(⌈K/block_K⌉, num_stages)`：`T.copy` 把 `A[bz, m0:m0+bm, ks:ks+bk]` 与 `B[bz, ks:ks+bk, n0:n0+bn]` 装入 L1（Developer `T.alloc_shared`，全程保持 ND，后端隐式完成 Cube 所需布局适配）；`T.gemm(a_s, b_s, c_l, initC=(k==0))` 累加到 fp32 fragment（Cube `mmadL1`）。首轮由 `initC` 初始化，**无独立清零**。
3. **epilogue（R2）**：`T.copy(c_l[0:tm, 0:tn], c[bz, m0:m0+tm, n0:n0+tn])`，L0C fp32 → GM 直写（保持 ND；fp32→fp16/bf16 量化由编译器自动 `hivm::VCastOp`(RINT) 完成，无显式 NZ 转换），尾块由 `tm/tn = T.min(block, dim-start)` 精确裁剪。

**与源算法差异**：① batch 由 `blockIdx.z` 改为一维折叠（R1）；② epilogue 由"SMEM 中转 + 谓词写"改为"L0C→GM 直写（编译器自动 VCast 量化）+ 切片裁剪"（R2）；③ 去 `T.clear` 改 `initC`（R3）；④ 去 `T.use_swizzle`/`make_swizzled_layout`/`threads`/TMA（R4）。数值语义不变（§0.6 各语义保持论证 + §1.6.1 机器验证）。

### 1.5 数据流图

```
GM a[bz] --T.copy(ND, GM→L1)--> L1[a_s(block_M×block_K)] --+
GM b[bz] --T.copy(ND, GM→L1)--> L1[b_s(block_K×block_N)] --+--> T.gemm(initC) --> L0C[c_l(fp32)]
                                                             (K 维 T.Pipelined 累加)
L0C[c_l] --T.copy(ND，编译器自动 VCast(RINT) fp32→fp16/bf16)--> GM c[bz, M-tile, N-tile]
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）

**调研深度**：`_bmm_kernel` 属**矩阵类**（Cube GEMM），按 `algorithm-research.md` §2 执行**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。检索入口：`python3 .agents/tools/kb_search.py "batched gemm bmm"` / `"cube gemm batch 分核"`（命中 `ALG-gemv`、`TRAP-DEVMODE-PERSIST-GEMM`、`CONST-*` 等条目）。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 对比 |
|---|------|------------|-----------|---------|-------------|
| 0 | **基线：分块 K 累加 GEMM**（源算法） | `C[b]=Σ_k A_k B_k`，L1 tile + K 流水 | —（基线） | — | ✅（基线） |
| 1 | 累加器初始化融合（源已隐含在 GPU 的 `T.clear`+`T.gemm`，NPU 可改 `initC`） | `c=A_0B_0; for k>0: c+=A_kB_k` | 数学恒等（IEEE `0+x==x`） | 少算：省一次 `block_M×block_N` fp32 清零写 | ✅（§1.6.1 OPT-1） |
| 2 | epilogue 直写（消源方案显式 SMEM staging） | `L0C→GM` 直写（Developer，编译器自动 VCast(RINT) 量化） | 数学恒等（仅去中转缓冲，转换次数不变） | 少搬：省源方案显式 `c_smem` staging 缓冲 | ✅（§1.6.1 OPT-2） |
| 3 | **Strassen / Winograd 类快速矩阵乘** | 以加法换乘法，`O(n^2.807)` | 数学等价（有额外加法/数值误差） | 少算 FLOPs（大矩阵） | ❌ 否决（见下） |
| 4 | **split-K 分解** | 沿 K 切分，多核部分和 + 跨核归并 | 数学等价（累加分组变化） | 提高并行度（小 M·N 大 K 时） | ✅（R3 对比，见下） |
| 5 | **NT 布局（B 转置为 `[B,N,K]`）** | `transpose_B=True`，B 的 K 连续 | 数学等价 | 少搬：K 连续装载 | ❌ 否决：I/O 契约为 `[B,K,N]`（N 连续），转置需 host/核内额外重排 |
| 6 | **batch 维折叠 vs 独立 kernel** | 一维折叠 grid（R1） | 数学等价（任务重编号） | 单次 launch | ✅（§0.6 R1，结构层） |

**Strassen 否决依据**：① NPU Cube 的 `T.gemm`/`mmadL1` 只表达标准 MAC 矩阵乘，Strassen 需分解为 7 个 `T.gemm` + 若干 Vector 加减（`T.gemm` 无 Winograd/Strassen 原语，`docs/Tilelang.language/线性代数操作/T.gemm.md` §1/§2 仅标准矩阵乘）；② 增加中间缓冲与跨核/片上流量；③ 数值稳定性下降（结果不逐位等价）。在 manifest 最大 shape（K=4096）下收益不足以抵偿结构代价——**结构性否决**。

**NT 布局否决依据**：`torch.bmm` 契约 `b: (B,K,N)`（N 连续，`examples/TileOPs/tileops/workloads/bmm.py` L30）；改 NT 需 host `transpose+contiguous`（额外 DtoD 拷贝，GPU 版 fp8 路径正是为此付出代价，`ops/bmm.py` L339-354 的告警）或核内重排，属"I/O 契约不可改"的负向淘汰。

**R2 在线算法**：**有（但收益为负，见 R3/R4）**。结构判据：GEMM 输出对 K 维是可结合的 running 归约（`running += A_k·B_k`），故原则上存在"分块部分和 + 归并"的在线/流式变体 = **split-K**（候选 #4）。**基线候选 #0 本身已是单遍在线形态**（每个输出 tile 只扫一遍完整 K，累加器 O(tile) 驻留 L0C，无 O(K) 中间缓冲）——即基线已满足 R2 的收益口径（单遍扫描、O(tile) 中间状态），无需再引入跨核归并型在线变体。**收益口径**：扫描遍数 1（基线已最优）；中间缓冲 O(block_M×block_N)（L0C 64KB）；UB/L1 驻留无压力。

**R3 复杂度对比**（四口径；FLOPs 与"不可消除 I/O"按 manifest 最大 workload `b4-4k`，B=4,M=N=K=4096 计）：

| 算法候选 | FLOPs | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|-------------------|---------|-------------|---------------------|
| **#0 基线分块 GEMM** | `2·B·M·N·K = 5.5e11`（乘加=2；不可再降，Strassen 见 #3） | `B·(M·K+N·K+M·N)·2 = 4.03e8`（= 输入读 + 输出写；**不可消除**） | 1（每输入元素单遍 GM 读） | `block_M·block_N·4 = 64KB`（L0C fp32） | 可并行度 = `B·⌈M/bm⌉·⌈N/bn⌉ = 4096`；无跨核归并 |
| #1 初始化融合 | 同上 | 同上 | 1 | 同上 | 同上（省一次片上写） |
| #2 epilogue 直写 | 同上 | 同上 | 1 | 同上（少一个 `block_M·block_N·2` SMEM 缓冲） | 同上 |
| #3 Strassen | `~0.6×(基线)`（1–2 层递归） | **增大**：额外中间矩阵往返 + 7 路分解 | ≥2（子问题中间结果） | `O(N²)` 级中间矩阵 | 跨核同步/归并代价大；`T.gemm` 无原语 |
| #4 split-K | 同上 + 归并加 `S·B·M·N` | 增大：部分和写回 + 读回（`2·S·B·M·N·4`） | ≥2（部分和往返） | 部分和缓冲 `S·B·M·N·4` | 需跨核归并（两阶段 kernel/原子加） |

口径说明：FLOPs 只计乘加（不含索引/地址）；访存量只计不可消除的 GM 输入读+输出写（不含中间往返，中间往返单列）；#3/#4 的"增大"项按结构给量级。

**R4 硬件亲和性评估**（逐 R3 存活候选；负向淘汰附 `docs/` / `testing/` / `examples/` 佐证或"未文档化假设"标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#0 基线** | ✅ MAC 密集落 Cube（`T.gemm`→`mmadL1`） | ✅ L0C 64KB < 128KB；L1 64KB(×2 stage) < 512KB | ✅ block_K=64 ≥ 16 分形；K%A: 是 16 倍数（manifest）；尾轴 32B 对齐 | ✅ shape 经 `lru_cache` 静态特化 | ✅ `T.Pipelined` K 流水；无 CV 跨引擎往返 | **✅ 选定** |
| #1 初始化融合 | ✅ 同基线 | ✅ 少一次写 | ✅ | ✅ | ✅ | ✅ 采纳（§1.6.1） |
| #2 epilogue 直写 | ✅ Cube→Vector 片上通路（L0C→GM，编译器自动 VCast(RINT) f32→f16/bf16） | ✅ 少显式 staging 缓冲 | ✅（VCast 支持 f32→f16/bf16） | ✅（`T.min` 动态 extent 由探针 D 证实可用） | ✅ 单 kernel 内后端自动 CV-split（无用户级调度） | ✅ 采纳（§1.6.1） |
| #3 Strassen | ❌ 需 Vector 加减链 + 多次 `T.gemm`；Cube 无原语 | ❌ O(N²) 中间矩阵超 L0C/UB | — | ❌ 递归边界动态 | ❌ 多轮跨引擎 | ❌ 否决 |
| #4 split-K | ✅ Cube 可算部分和 | ⚠️ 部分和缓冲落 GM | ✅ | ⚠️ | ❌ 需跨核归并（两阶段/原子加）；manifest 各 shape M·N 均 ≥ 128² 已有充足并行度（256–4096 逻辑核 ≫ 20 物理核）→ **split-K 无收益** | ❌ 否决（并行度已足够，归并代价净亏） |

**调研结论**：**选定基线算法族（分块 K 累加 GEMM）+ 两项结构等价优化（#1 初始化融合、#2 epilogue 直写）**。关键依据：GEMM 的 FLOPs 与不可消除 GM I/O 已达量级下界（`2·B·M·N·K` / `B·(MK+NK+MN)·2`），无更优等价公式（Strassen 结构性否决、NT 因 I/O 契约否决）；基线已满足 R2 单遍在线口径（无需 split-K，manifest shape 并行度 256–4096 逻辑核 ≫ 20 物理核）；基线计算单元匹配 Cube 且容量/对齐/静态边界/流水全通过。与基线的结构差异：仅 §0.6 的四项 NPU 重设计（一维折叠网格、L0C→GM 直写（编译器自动 VCast 量化）、`initC`、去 GPU 专属手段）——**源算法 #0 未被替换**，源算法已识别的 10 项优化手段（§0.4）逐项处置见 §0.5（保留 2 / 等价替换 3 / 重新设计 3 / 舍弃 4，意图承接：L1 tiling 承接 SMEM tiling、`T.Pipelined` 承接 K 流水、Cube MMA 承接 WGMMA、fixpipe 直写（`T.store_fixpipe`）承接 SMEM 中转的合并写意图——但仅 Expert 路径适用，本设计走 Developer（编译器自动 VCast 量化）故不使用，L2/swizzle/TMA/threads 无 NPU 对应能力故舍弃）。

**调研范围（"无更优替代"的检索边界）**：查过 `references/algorithm-candidates.md`（`ALG-gemv` / `ALG-conv` / `ALG-triangular` 等条目，无 bmm 专属化简行）；`pattern-library`（`constants.md` 容量/带宽常数、`attention.md` PL-1.8 bf16 Cube 直连、`traps-*` 陷阱、`cases.md` 案例索引）；本仓 `examples/gemm/*`、`examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py`、`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/*`；结构分析（双线性型的等价变形空间：结合律重排 = split-K、快速矩阵乘 = Strassen/Winograd）。未使用互联网检索（本地源已覆盖，无新候选）。

**设计期估算（D-2）**：估算下界 = max(流量项, 发射项, 容量项)，按 dispatch 代表 workload 计算（引用 `tilelang-op-optimize` skill pattern-library/constants.md 条目 ID + 版本戳；常数绑定 Ascend910B2C，工具链变更后待重验）：

| workload (B,M,N,K) | GM 流量项〔`CONST-fabric-bw` ≥1.73 TB/s；`CONST-mte2-degradation` 3:1 混合地板 1.26 TB/s〕 | Cube 操作数流地板〔`CONST-L1-port-bw` 148–154 GB/s/核 × 20 核〕 | 发射项〔`CONST-vector-launch-overhead`〕 | 估算下界 |
|---|---|---|---|---|
| smoke b4-128-128-128 | 0.39MB → 0.2µs | 0.26MB → 0.09µs | ~0（核内无循环 vector op；epilogue 每 tile 一次 VCast） | ~0.2µs（不计 launch） |
| b8-1024³ | 50MB → 29µs | 268MB → **89µs** | ~0 | **~89µs** |
| b16-512³ | 25MB → 15µs | 67MB → 22µs | ~0 | **~22µs** |
| b4-4096³ | 403MB → 233µs | 8.59GB → **2.86ms** | ~0 | **~2.86ms** |

`估算下界: 2860 us（流量 233us〔CONST-fabric-bw〕+ Cube 操作数流地板 2860us〔CONST-L1-port-bw〕+ 发射 ~0us）`（b4-4k）。口径：Cube 操作数流 = `B·M·N·K·elem_bytes·(1/bm+1/bn)`（bm=bn=128 时该式最小——固定 `bm·bn ≤ L0C/4 = 32768` 下 `1/bm+1/bn` 在 bm=bn 处取最小）。**Cube MAC 算力项未实测、未列入**（标注「未文档化假设」：本仓 pattern-library 无 Cube 峰值 FLOPs 常数，估算依据为 910B2C 量级公开值，留待 Stage 4 对账）；若实测算力项 > 2.86ms，则 b4-4k 转为 compute-bound。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（基线分块 K 累加 GEMM）。采纳项：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | 累加器初始化融合 | `c = 0; for k∈[0,K/bk): c ← c + A_k·B_k` | `c = A_0·B_0; for k∈[1,K/bk): c ← c + A_k·B_k`（`T.gemm(initC=(k==0))`） | **数学恒等**：IEEE-754 `0 + x == x` 精确成立（含 ±inf/NaN/subnormal）；K 分块累加顺序不变。机器验证见下（OPT-1，max ulp 差 0） | 省一次 `block_M×block_N` fp32 清零写：128×128×4 = **64KB** 片上写 / tile；全 b4-4k = 4096 tile × 64KB = **256MB** 片上写 |
| 2 | epilogue 直写（消源方案显式 SMEM staging） | `c_l(fp32 fragment) → c_smem(显式 staging，含 cast) → 谓词读回 → GM`（两跳 + 命名中转缓冲） | `c_l(L0C) → GM`（Developer 单条 `T.copy`，保持 ND；`f32→f16/bf16` 由编译器自动 `hivm::VCastOp`(RINT) 完成，无显式 staging 缓冲） | **数学恒等**：仅去掉显式中转缓冲，输出张量为同一 fp32 累加器做同一 round-to-nearest-even 舍入（转换次数不变；源 GPU `T.copy(c_local, c_smem)` 亦为 RN）；机器验证见下（OPT-2，max ulp 差 0） | 省源方案显式 `c_smem` staging 缓冲（`block_M×block_N×elem_bytes` = 128×128×2 = **32KB**/tile）的分配与写回路径；全 b4-4k = 4096 tile。**注**：Developer 的 VCast 舍入经编译器内部临时张量完成（片上隐式流量由后端管理、不落显式缓冲），故收益按「消显式 staging 缓冲」计，**不含「无 UB 往返」的绝对化主张**；精确片上流量待 Stage 4 实测 |

**优化结论**：采纳 2 项（OPT-1 初始化融合、OPT-2 epilogue 直写）；**优化后公式 = `c = A_0·B_0; for k>0: c += A_k·B_k`，输出经 Developer `T.copy`(L0C→GM，编译器自动 VCast 量化) 直写**（供 §3.1 公式拆解使用）。无更多优化空间的依据：GEMM 为双线性型，各输出元素间无公共子表达式；无超越函数/除法；乘法次数 `2·B·M·N·K` 已达稠密矩阵乘下界（Strassen 等快速算法 §1.6.0 R1/R3 结构性否决）。

**等价性机器验证（D-1）**：脚本 `examples/bmm/_bmm_kernel/verify_equiv.py`（designer 亲自执行；numpy CPU，fp64 参照 + fp32 累加路径；随机张量 + 角点值 ±inf/NaN/subnormal/rounding-halfway ties/边界 shape），结果内嵌：

| 采纳项 | max ulp 差 | 违反率 | 结论 |
|--------|-----------|--------|------|
| OPT-1 初始化融合（`initC=(k==0)` vs 清零后累加） | 0 | 0.0000 | `EQUIV_PASS` |
| OPT-2 epilogue 直写（一跳 cast vs SMEM 中转后 cast） | 0 | 0.0000 | `EQUIV_PASS` |

（执行输出：`OPT-1 initC fusion  max_ulp 0  viol_rate 0.0000  EQUIV_PASS`；`OPT-2 epilogue direct  max_ulp 0  viol_rate 0.0000  EQUIV_PASS`；exit code 0。用例含 random / ±inf / NaN / subnormal / rounding-halfway ties / 边界 shape；本次修订 v1 重跑通过——OPT-2 机制已更正为 Developer 编译器自动 `hivm::VCastOp`(RINT)，与源 GPU 的显式 staging 均为同一 round-to-nearest-even 舍入。）

#### 1.6.2 向量化替代分析（循环 / 标量消除）

本算子主体为 **Cube 算子**（全部乘加在 Cube `T.gemm`（`mmadL1`）内完成；epilogue 另含编译器自动插入的向量化跨 dtype 转换），核内**没有逐元素标量计算热点**。盘点实现方案中的全部循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|-----------|-------------|---------|-------------|
| 1 | grid 索引 decode（`bz/by/bx/m0/n0/tile_m/tile_n`） | 每核 6 条标量整除/取模/min | 无（无逐元素等价 API） | ❌ | **block 级索引 / 任务映射计算**：每核仅约 6 次标量运算，与 `block_M·block_N·K` 级 MAC 相比可忽略（非逐元素热点）；无向量 API 能表达"每核一次"的标量映射 |
| 2 | K 维累加循环 | `T.Pipelined(⌈K/bk⌉)` | — | ❌（非 Vector 循环） | **tile 级矩阵累加循环**：由 Cube MMA 承载（`T.gemm`），属 `mmadL1` 语义，不是可向量化的逐元素循环；`T.Pipelined` 已把其转化为搬运/计算重叠 |
| 3 | 乘加本身 `Σ A·B` | `T.gemm`（Cube MAC） | 无（Cube 已是专用加速单元） | ❌ | **计算单元匹配决策**：MAC 密集段应落 Cube（§1.6.0 R4），用 Vector 逐元素乘加替代反而降级（把 Cube 任务搬到 Vector = 反向优化） |
| 4 | epilogue 写回 | `T.copy(c_l[0:tm,0:tn], c[...])`（L0C→GM，含编译器自动 VCast 量化） | — | ❌ | **纯搬运 + 后端自动转换**：L0C→GM 为专用数据通路，跨 dtype 量化由后端自动插入向量化 VCast；不经用户逐元素计算，无向量替代必要 |
| 5 | 输出的 fp32→fp16/bf16 舍入 | 编译器在 `T.copy(L0C→GM)` lowering 中自动插入的 `hivm::VCastOp`(RINT)（向量化，`T.copy.md` §2.3 第 4 条） | 手工改写为显式 `T.vcast` | ❌ | **已是后端自动向量化**：Developer `T.copy(L0C→GM)` 的跨 dtype 转换由编译器自动插入向量化 `hivm::VCastOp`（RoundMode=RINT），本就不是「可优化的逐元素标量循环」；手工 `T.vcast` 需自行把 L0C 结果落 UB 再转，等价或更繁、无增益 |

**向量化结论**：实现方案中**无逐元素标量计算点**（0 处可向量化、0 处本应向量化而未向量化）；保留 5 处非向量形态，理由类别：**block 索引 / 任务映射（#1）、Cube 专用单元（#2/#3）、DMA 专用通路（#4）、后端自动向量化 VCast（#5）**。epilogue 的 fp32→fp16/bf16 舍入由编译器自动插入的向量化 `hivm::VCastOp` 承担（非用户标量循环）。逐项见上表，无"实现简单"式理由。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子为 **Cube 类**：向量化轴决策退化为 **Cube 数据布局决策**（ND/NZ 分形、转置选择、epilogue 通路）；按 `algorithm-research.md` / `design-template.md`，Cube 类须记录决策与依据。

**轴质量评分维度**（逐候选）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 / 分形对齐 | Cube 分形需 `block ≥ 16` 且尾轴 32B 对齐（fp16 尾轴 16 倍数） | block_K=64、block_N=128、K%16==0 ✓；M/N 尾部由动态 extent 处理 |
| 累加链形态 | K 维为串行累加维（MMA 内建），M/N 为并行维 | K 维分块累加由 `mmadL1` 承载，无向量 lane 浪费 |
| repack 代价 | 布局转换是否需显式重排 | Developer 全程保持 ND、后端隐式布局适配 → 零显式 repack；显式 ND↔NZ（`T.load_nd2nz`/`T.store_fixpipe`）仅 Expert 模式 |
| 片上容量影响 | 布局对 L0C/L1 驻留预算的影响 | L0C 64KB + L1 64KB(×2 stage) < 128KB / 512KB |

**候选矩阵**（全枚举；弃选给量化理由）：

| # | 布局方案 | 计算布局 / 通路 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------------|------------|--------------|------|
| 1 | **Developer ND（选定）**：`T.alloc_shared`（L1）+ `T.copy` + `T.gemm` | Cube（全程保持 ND）；后端隐式完成 Cube 所需布局适配；epilogue `T.copy`(L0C→GM，编译器自动 VCast 量化) | 无显式 repack | 基线；结构最简，探针 B/D/E 实测正确 | **✅ 主选** |
| 2 | Expert NZ：`T.alloc_L1`/`T.alloc_L0C` + `load_nd2nz` + `store_fixpipe`（`examples/gemm/matmul.py`） | Cube；显式控制 NZ/量化 | 无（显式接口即 repack） | 控制更细（可定制 channel_split/pre_relu），但对纯 GEMM 无额外收益；代码复杂度高 | ❌（本算子无量化/per-relu 需求；Developer 已足够；留作 Stage 4 若需细粒度流水调优的备选——见裁决计划） |
| 3 | NT 布局（B 转 `[B,N,K]`，`transpose_B=True`） | Cube | host `transpose(-2,-1).contiguous()`（DtoD 拷贝） | 少搬：B 的 K 连续；但 I/O 契约 `(B,K,N)` 不可改，host 转置引入额外 DtoD | ❌（I/O 契约否决；GPU fp8 路径的告警 `ops/bmm.py` L339-354 即此代价） |
| 4 | host permute 进出（[B,M,K]↔[B,K,M]） | — | 两次 host permute | repack 代价 > 算子本体流量 | ❌（PL-1.3-host-permute：host permute 路线通常净亏，实测 106–147µs） |

**布局决策结论**：选定 **#1 Developer ND 路径**——L1 承载 A/B tile（`T.alloc_shared`，全程保持 ND、后端隐式布局适配）、L0C 承载 fp32 累加器（`T.alloc_fragment`）、epilogue 走 Developer `T.copy`（保持 ND；`f32→f16/bf16` 由编译器自动 `hivm::VCastOp`(RINT) 量化）；**不做显式 repack**（无 NT/host permute）。该结论同步落入 §3.3 伪代码与 §4 内存规划、§6 循环结构（K 维为 Cube 累加循环、M/N 为 block 级并行维）。

**实验裁决计划（备选 #2）**：主选（Developer ND）与备选（Expert NZ）的判定裕度依赖**未实证常数**——显式 L1/L0C NZ 通路相对 Developer 隐式路径在流水深度/量化通路上的差异无本仓实测数字（pattern-library constants.md 无此条目）→ 按实验裁决模式产出三件套：
- **主选**：#1 Developer ND（进 Stage 3 实现）；
- **备选**：#2 Expert 双 Scope + `T.alloc_L1`/`T.alloc_L0C` + `load_nd2nz`/`store_fixpipe`（结构完整，可直接进 Stage 4：L1 双槽 A/B、L0C 单槽、K 循环静态边界；载体见 `examples/gemm/matmul.py`）；
- **裁决计划**：Stage 4 对代表 shape（b8-1k / b4-4k）实测主选 vs 备选 latency；判定阈值：备选 latency < 主选 × 0.95 才翻转；结果按 `core-split-strategy.md` §2.4 / `perf-feedback.md` 回写。shape 特化工厂（`lru_cache`）下可评估按 shape 分派主选/备选。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（与 `op_requirements.programming_mode: developer` 一致）。

### 2.2 选型理由

- 算子主体为 **Cube GEMM**：Developer 模式的 `T.alloc_shared`（Cube 路径映射 L1）+ `T.copy`（保持 ND，后端隐式完成 Cube 所需布局适配）+ `T.gemm` 已完整表达计算，无需 Expert 的显式 `T.alloc_L1`/`T.alloc_L0C`/`load_nd2nz`/`store_fixpipe`。**说明**：epilogue 的 fp32→fp16/bf16 舍入由编译器在 `T.copy(L0C→GM)` lowering 中自动插入向量化 `hivm::VCastOp`（Vector 单元，RoundMode=RINT），属后端自动 CV-split 管理，**非用户编写的 Vector 后处理**。
- `examples/gemm/example_gemm.py`（Developer，矩阵乘基准先例）与本算子结构同构；`docs/开发指南.md` 的 GPU→NPU 迁移即以 Developer 模式 matmul 为范本。
- `tilelang-cube-skill` 明确："Developer mode: Keep ND tensors and use T.copy without explicit NZ conversion"——本算子无量化/per-relu/channel-split 定制需求，Developer 路径足够。
- 备选 Expert 路径已按 §1.6.3 裁决计划登记（若 Stage 4 显示显式 NZ 通路更优则切换）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared((block_M, block_K))` / `((block_K, block_N))` → L1（Cube 路径）；`T.alloc_fragment((block_M, block_N), "float32")` → L0C 累加器 |
| 计算方式 | `T.gemm(a_s, b_s, c_l, initC=(k==0))` → `hivm.hir.mmadL1` |
| 数据搬运 | `T.copy`（Developer 全程保持 ND；GM→shared 后端隐式布局适配；L0C→GM 由编译器自动 VCast 量化） |
| 同步 | 编译器自动同步（Developer 模式，见 §7） |
| 分核 | 一维折叠逻辑网格（见 §5.5）；**不得**在 Developer 模式用 persistent（核内 `T.serial`）+ `T.gemm` 混排（`TRAP-DEVMODE-PERSIST-GEMM`） |

> **CV 融合**：本算子**无用户编写的 Vector 后处理、无用户级跨引擎数据流**，故用户级 CV 融合编排不适用；但 epilogue 的 fp32→fp16/bf16 舍入由后端在 `T.copy(L0C→GM)` 中自动插入向量化 `hivm::VCastOp`（Vector 单元），属**编译器自动 CV-split 管理的隐式跨引擎协同**——**不要**据此手工引入 Vector scope 或手写跨引擎同步。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**：`c = A_0·B_0; for k>0: c += A_k·B_k`，输出经 Developer `T.copy`(L0C→GM，编译器自动 VCast 量化) 直写。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(bz,by,bx) ← decode(cid)` | 一维折叠网格 → 三坐标（R1） |
| 2 | `m0=by·bm, n0=bx·bn, tile_m=min(bm, M-m0), tile_n=min(bn, N-n0)` | tile 起点与尾块尺寸 |
| 3 | `a_s = A[bz, m0:m0+bm, ks:ks+bk]`, `b_s = B[bz, ks:ks+bk, n0:n0+bn]` | 操作数装载（GM→L1；Developer 保持 ND，后端隐式布局适配） |
| 4 | `c ← c_l + a_s·b_s`（首轮 `c_l ← a_s·b_s`） | K 分块累加（Cube MMA，`initC=(k==0)`） |
| 5 | `c_l[0:tile_m,0:tile_n] → c[bz, m0:m0+tile_m, n0:n0+tile_n]` | epilogue（L0C→GM；编译器自动 VCast(RINT) 量化） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 网格启动 / 折叠 decode | `T.Kernel(grid, is_npu=True)` | `grid = B*nt_m*nt_n`；返回 `(cid, _)` | Developer |
| 2 | tile 边界 | `T.min(dim, block)` | 动态 extent | Developer |
| 3 | 操作数装载 | `T.copy(src_slice, a_s)` | `T.copy(a[bz, m0:m0+bm, ks:ks+bk], a_s)`；Developer 保持 ND，后端隐式布局适配（GM→L1 的**显式** ND2NZ 为 Expert 路径，见 `T.copy.md` 支持方向表第 14 行） | Developer |
| 4 | 矩阵乘累加 | `T.gemm(a_s, b_s, c_l, initC=(k==0))` | `a_s`(bm×bk)、`b_s`(bk×bn)、`c_l`(bm×bn, fp32)；→ `hivm.hir.mmadL1` | Developer |
| 5 | epilogue | `T.copy(c_l[0:tm,0:tn], c[bz, m0:m0+tm, n0:n0+tn])` | L0C→GM 保持 ND；`f32→f16/bf16` 由编译器自动 `hivm::VCastOp`(RINT) 量化（`T.copy.md` 支持方向表第 15 行 + §2.3 第 4 条） | Developer |
| 流水 | K 循环重叠 | `T.Pipelined(⌈K/bk⌉, num_stages=2)` | 双缓冲 | Developer |
| 累加器 | `c_l` 分配 | `T.alloc_fragment((bm,bn), "float32")` | fp16 输入必须 fp32 累加（`T.gemm.md` §2.3） | Developer |
| 输出分配 | `out_idx=[-1]` | `@tilelang.jit(out_idx=[-1], target="npuir")` | 框架分配 `(B,M,N)` | host |

### 3.3 计算伪代码

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1], target="npuir")
def _bmm_kernel(batch, m, n, k, block_M=128, block_N=128, block_K=None,
                num_stages=2, dtype="float16"):
    accum_dtype = "float32"
    # block_K: 64 若 k%64==0，否则 32（仍满足 k%block_K==0 且 ≥16 分形）；
    # smoke k=128 -> 32 保证 k_iters=4（流水深度，docs/开发指南.md §3.2）
    block_K = block_K if block_K is not None else (64 if k % 64 == 0 else 32)
    nt_m = (m + block_M - 1) // block_M
    nt_n = (n + block_N - 1) // block_N
    grid = batch * nt_m * nt_n

    @T.prim_func
    def _bmm_main(
        a: T.Tensor((batch, m, k), dtype),
        b: T.Tensor((batch, k, n), dtype),
        c: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(grid, is_npu=True) as (cid, _):          # R1: 一维折叠网格
            bz = cid // (nt_m * nt_n)
            rem = cid % (nt_m * nt_n)
            by = rem // nt_n
            bx = rem % nt_n
            m0 = by * block_M
            n0 = bx * block_N
            tile_m = T.min(block_M, m - m0)                    # 尾块尺寸（整除时=block_M）
            tile_n = T.min(block_N, n - n0)

            a_s = T.alloc_shared((block_M, block_K), dtype)     # → L1
            b_s = T.alloc_shared((block_K, block_N), dtype)     # → L1
            c_l = T.alloc_fragment((block_M, block_N), accum_dtype)  # → L0C

            for ks in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                T.copy(a[bz, m0:m0 + block_M, ks * block_K:ks * block_K + block_K], a_s)
                T.copy(b[bz, ks * block_K:ks * block_K + block_K, n0:n0 + block_N], b_s)
                T.gemm(a_s, b_s, c_l, initC=(ks == 0))         # R3: 首轮初始化，无 T.clear

            T.copy(c_l[0:tile_m, 0:tile_n],                    # R2: L0C→GM 直写（编译器自动 VCast 量化）+ 尾块切片裁剪
                   c[bz, m0:m0 + tile_m, n0:n0 + tile_n])

    return _bmm_main
```

> 说明：`T.Tensor((...), dtype)` 直接把 dtype 作为第二个位置参数传入（避免 `dtype=` 关键字产生的 `false alarm`）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(..., is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.4 示例1/3 | 探针实测（A/B/D/E） |
| `T.copy`（3D 切片→2D shared） | `docs/Tilelang.language/内存操作/T.copy.md` §2.4 示例 | 探针 B/D/E PASS |
| `T.gemm(..., initC=)` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.4；`examples/gemm/example_gemm.py` L30 | 探针 B/D/E PASS |
| `T.alloc_shared` / `T.alloc_fragment` | `docs/Tilelang.language/内存操作/T.alloc_shared.md`；`examples/gemm/example_gemm.py` L23-25 | 探针实测 |
| `T.Pipelined(num_stages=)` | `docs/开发指南.md` §3.2；`examples/gemm/example_gemm.py` L27 | 探针 B/D/E PASS |
| `T.min`（动态 extent） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 第 2 条（尾块动态 extents） | 探针 D PASS |
| `hivm::VCastOp`（Developer 跨 dtype 自动量化，非用户 API） | `docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条 + 支持方向表第 15 行（Developer `L0C→GM` 经 VCast 通用路径） | 探针 B/D/E PASS（fp16/bf16 输出数值正确） |
| `out_idx=[-1]` | `examples/gemm/example_gemm.py` L8 | 探针实测 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel（`T.Kernel` NPU ≤ 2D block） | **Yes** | R1：一维折叠网格 decode `(bz,by,bx)`（探针 A 证实 3D 被拒；`T.Kernel.md` §2.3） |
| GEMM 非整除（M/N 不能被 block 整除） | **Yes** | R2：epilogue 用切片动态 extent `T.copy`（探针 D 证实）；整除时退化为整块 |
| `K % 16 == 0`（Cube 分形 / 源前置） | 涉及（host 校验） | 保留源 host 前置校验；block_K ∈ {64,32} 恒为 16 倍数 |
| L0C 溢出（128KB） | **Yes** | `block_M·block_N·4 ≤ 128KB` ⇒ 128×128=64KB ✓（§5.3） |
| UB/L1 溢出（UB 192KB / L1 512KB） | 涉及（L1） | L1 预算 `(bm·bk+bk·bn)·2·num_stages = 64KB`（×1.7 膨胀 ≈ 110KB < 512KB）✓ |
| Developer + persistent + gemm 混排崩溃（`TRAP-DEVMODE-PERSIST-GEMM`） | **Yes**（约束分核） | 分核方案**不采用**核内 `T.serial` persistent（见 §5.5），用全量逻辑核 launch |
| `threads=` 在 npuir 无效果（`TRAP-threads-kwarg-noop`） | 涉及 | 不传 `threads` |
| 谓词 fragment→GM 直写触发运行期 aicore 异常（CG-2026-0001(d)，`The MPU address access is invalid`） | **Yes** | R2 改用切片 `T.copy`（探针 C 证伪谓词直写、D 证实切片） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异）

| 差异项 | 参考实现（GPU） | 本项目（Ascend/npuir） | 转换方案 |
|--------|----------------|----------------------|----------|
| Kernel 维度 | 3D `T.Kernel(cn, cm, batch)` | 最多 2D block | 一维折叠 grid + decode（§0.6 R1） |
| 网格映射 | batch→`blockIdx.z` | 折叠进 `cid` | §3.3 伪代码 |
| GEMM | `T.gemm` (WGMMA, FullRow) | `T.gemm`→`mmadL1` | 同 API，去 `policy`/`threads` |
| 累加器清零 | `T.clear(c_local)` | `T.gemm(initC=(k==0))` | §0.6 R3 |
| epilogue | SMEM 中转 + 谓词写 | L0C→GM 直写（编译器自动 VCast 量化）+ 切片裁剪 | §0.6 R2 |
| L2/SMEM 布局 | `T.use_swizzle` + `make_swizzled_layout` | 无对应 API / Developer 保持 ND（后端隐式布局适配） | §0.5 舍弃 |
| WARP specialization / TMA | producer/consumer WG + TMA | 无 | §0.5 舍弃 |
| 编译配置 | `threads=128`, `compile_flags=["-O3","-DENABLE_BF16"]`, `tl.disable_warp_specialized` | 不适用 | 移除（npuir 无这些参数） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/gemm/example_gemm.py` | **高度相似** | Developer Cube GEMM 标准形态（`T.alloc_shared` + `T.Pipelined` + `T.gemm(initC=(k==0))` + `T.copy` epilogue）；一维 grid decode |
| `examples/gemm/matmul.py` | 中（Expert 对照） | Expert NZ 备选路径（`T.alloc_L1`/`T.alloc_L0C`/`load_nd2nz`/`store_fixpipe`）；§1.6.3 备选 #2 的载体 |
| `examples/gemm/matmul_dynamic_shape.py` | 中 | 动态/尾块 GEMM 的 `T.min` 边界处理形态 |
| `testing/npuir/linalg_ops/test_2d_grid_gemm_dyn_dev.py` | 中 | Developer 2D grid GEMM（`is_npu=True`，`as (bx,by,_)`）+ 动态 shape 测试形态（官方测试） |
| `examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py` | 中（GPU 抽取件） | 一维 `T.Kernel(_total_ctas)` + `pid` decode 的分核/分组遍历先例（tile 调度） |
| `examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py` | 中（GPU 抽取件） | GPU GEMM 的 SMEM tiling / K 流水 / epilogue 结构对照 |
| `examples/bmm/_bmm_kernel/history_version/design_probe_bmm_v2.py` | **直接先例** | 本算子定稿形态（一维折叠 + slice epilogue）的实测可运行脚本 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `a` | `(B, M, K)` | float16 / bfloat16 | 左矩阵，K 连续；`K % 16 == 0` |
| `b` | `(B, K, N)` | same_as(a) | 右矩阵，N 连续（NN） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `c` | `(B, M, N)` | same_as(a) | 输出（`out_idx=[-1]` 框架分配） |

### 4.3 中间缓冲区

| Buffer | Shape | dtype | 存储层级 | 用途 |
|--------|-------|-------|----------|------|
| `a_s` | `(block_M, block_K)` | float16/bf16 | L1 | A tile（Developer 保持 ND，后端隐式布局适配） |
| `b_s` | `(block_K, block_N)` | float16/bf16 | L1 | B tile |
| `c_l` | `(block_M, block_N)` | float32 | L0C（fragment） | fp32 累加器 |
| （无双缓冲计数） | | | | `T.Pipelined` 自动多缓冲 —— 预算见 §4.5 |

### 4.4 内存搬运路径

```
GM[a] --T.copy(ND, GM→L1)--> L1[a_s] --MMA--> L0C[c_l] --T.copy(ND，编译器自动 VCast(RINT) f32→f16/bf16)--> GM[c]
GM[b] --T.copy(ND, GM→L1)--> L1[b_s] --/
（K 维 T.Pipelined(num_stages=2) 双缓冲；每 K-tile 一轮装载 + 一次 MMA 累加；Developer 全程保持 ND，不引入显式 NZ）
```

### 4.5 L1 / L0C 内存预算（block_M=block_N=128, block_K=64, fp16/bf16）

| Buffer | Shape | dtype | 大小 (Bytes) | 说明 |
|--------|-------|-------|-------------|------|
| `a_s` | (128, 64) | fp16 | 16384 | ×num_stages(2) = 32768 |
| `b_s` | (64, 128) | fp16 | 16384 | ×num_stages(2) = 32768 |
| **L1 小计** | | | **65536** | ×1.7（auto-multi-buffer 膨胀，`CONST-capacity-910B2C`）≈ **111KB** < 512KB ✓ |
| `c_l` | (128, 128) | fp32 | **65536** | L0C 上限 128KB ✓（50% 余量） |

> `c_l` 在 L0C 而非 UB；装载/累加为纯 Cube。**注**：epilogue 的 fp32→fp16/bf16 舍入由编译器自动插入的向量化 `hivm::VCastOp`（Vector 单元）承担，其内部临时张量由后端管理，**不占本设计显式 UB 预算**。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 全部 shape 经工厂 `lru_cache` 编译期特化（与源算子一致） | — |

> 若后续需支持动态 shape，可参照 `examples/gemm/matmul_dynamic_shape.py` 用 `T.symbolic` + `T.min` 动态 extent；本设计按 manifest 与源算子保持 shape 特化。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def _bmm_kernel(batch, m, n, k, dtype="float16"): ...
```

> 与源差异：无 `pass_configs={"tl.disable_warp_specialized": True}`（CUDA-only）、无 `compile_flags=["-O3","-DENABLE_BF16"]`（nvcc flags）、无 `threads`。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**Cube 主体**（GEMM 全在 Cube `T.gemm`；epilogue 的 fp32→fp16/bf16 舍入由编译器自动 `hivm::VCastOp`（Vector 单元）承担，属后端自动 CV-split，非用户 Vector 后处理）。
**判定依据**：算子仅含 `T.gemm`（矩阵乘累加），无用户编写的 element-wise/reduction/超越函数；epilogue 为 L0C→GM 搬运（含编译器自动量化）。

### 5.2 Block 划分

```python
block_M = 128   # L0C 预算上限约束下取大（1/bm+1/bn 最小化 Cube 操作数流，§1.6.0 估算口径）
block_N = 128   # 同上；128×128×4B = 64KB ≤ L0C 128KB
block_K = 64 if K % 64 == 0 else 32   # K%block_K==0 且 ≥16 分形；K=128 smoke → 32（k_iters=4）
num_stages = 2  # K 流水双缓冲；L1 64KB×1.7 < 512KB
```

**block_M/block_N 取值依据**：`block_M·block_N ≤ L0C/elem_bytes(f32) = 32768`；固定该乘积时 Cube 操作数流 `∝ (1/block_M + 1/block_N)` 在 `block_M = block_N` 处取最小（§1.6.0 估算），故取 128×128（= 16384，留 50% L0C 余量）。
**block_K 取值依据（shape 相关选取规则，与 §3.3 伪代码一致）**：`block_K = 64 if K % 64 == 0 else 32`——恒满足 `K % block_K == 0`（K 维无尾块）且 `block_K ≥ 16`（Cube 分形对齐）；manifest `K%16==0`。K≥256 取 64（L1 每 stage 32KB，`k_iters = K/64 ∈ {4,16,64}`）；K=128 的 smoke 取 32（`k_iters=4`，满足 `docs/开发指南.md` §3.2 建议的流水迭代 ≥4–6，避免 `K/64=2` 的过浅流水）。

### 5.3 约束分析

- **对齐约束**：`block_K ∈ {64,32}`（A 尾轴 K）→ fp16 64×2=128B / 32×2=64B，均 32B 倍数 ✓；`block_N=128`（B 尾轴 N）→ 256B ✓；分形 `block_K ≥ 16` ✓。
- **L0C 容量**：`128×128×4 = 64KB ≤ 128KB` ✓。
- **L1 容量**：`(128·64 + 64·128)·2·2 = 64KB`，×1.7 膨胀 ≈ 111KB `≤ 512KB` ✓。
- **尾轴对齐**：K 与 N 由 manifest 保证；M/N 非整除由 §0.6 R2 处理。

### 5.4 注意事项（非整除 / padding 策略）

- **K 维整除**：manifest 保证 `K % 16 == 0`；取 `block_K` 使 `K % block_K == 0`（256/1024/4096 用 64；128 用 32）→ **K 维无尾块，无需 padding**。
- **M/N 维非整除**：**不做 host padding**（避免额外 DtoD 拷贝与 crop），改由 kernel 内**动态 extent 切片写回**：`tile_m = T.min(block_M, M-m0)`、`tile_n = T.min(block_N, N-n0)`，`T.copy(c_l[0:tile_m,0:tile_n], c[bz, m0:m0+tile_m, n0:n0+tile_n])`（探针 D 实测通过，测试用例 b4-200×300-128 对齐 `torch.bmm`）。
- **装载侧尾块（统一为整块装载）**：A/B 装载统一采用**整块装载** `T.copy(a[bz, m0:m0+block_M, ks:ks+block_K], a_s)`（与 §3.3/§6.2 伪代码一致）；末 M/N tile 中 `m0+block_M > M` 的行/列会被读出 tile 范围。**该形态已由探针 D 实测**：非整除 shape（b4-200×300-128，bm=bn=64，末 tile 越界读 56 行）PASS——越界读到的尾行/尾列**不参与最终写回**（被 epilogue 的 `tile_m/tile_n = T.min(...)` 切片裁剪），未触发故障。**不做 host padding**（避免额外 DtoD 拷贝与 crop）。故 §3.3/§5.4/§6.2/§9.1 统一为「整块装载 + epilogue 切片裁剪」**单一形态**。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**：`num_logical_kernels = B × ⌈M/block_M⌉ × ⌈N/block_N⌉`（本算子输出网格含 batch 维，R1 折叠后即为一维 grid 长度）。按 manifest workload（block_M=block_N=128）：

  | workload | 逻辑核数 `B·⌈M/128⌉·⌈N/128⌉` | 物理核数 | 判定 |
  |---|---|---|---|
  | smoke b4-128-128-128 | 4×1×1 = **4** | 20 | 逻辑核数 ≤ 物理核数 |
  | b8-1024³ | 8×8×8 = **512** | 20 | 极大规模（≫ 物理核数） |
  | b16-512³ | 16×4×4 = **256** | 20 | 极大规模 |
  | b4-4096³ | 4×32×32 = **4096** | 20 | 极大规模 |

- **② 物理核数依据**：**实查**，非假设。查询代码与实际返回值：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()   # -> 20
  ```
  本工具链/设备实测返回 **20**（Ascend910B2C 物理 AI Core 数；Cube 算子直接使用该返回值，纯 Vector 算子才翻倍）。**注**：pattern-library `CONST-aicore-910B2C` 记录的 24 为另一环境/工具链的实测值，本设计以**本设备实查值 20** 为准（该常数标注"工具链变更后自动待重验"）。
- **③ 规模判定与分核方案**：
  - **smoke（逻辑核数 4 ≤ 20）**：**无需适配**——直接以 4 个逻辑核 launch，物理核数富余，无负载不均。
  - **b8-1024³ / b16-512³ / b4-4096³（极大规模）**：**主选 = 全量逻辑核 launch（over-subscription）**，每个逻辑核承担一个完整 `(batch, M-tile, N-tile)` GEMM 任务（内含 ⌈K/block_K⌉ 轮 Cube MMA）。
    - **为何不通过放大 block 缩减核数**：L0C 硬上限 `block_M·block_N·4 ≤ 128KB` ⇒ `block_M·block_N ≤ 32768`，128×128 已是最优容量点附近（§5.2），无法把 4096 个核缩到 20 的整数倍（也缩不到 20 量级）。
    - **为何不用核内串行 `T.serial`（标准 §1 要素③"极大规模"方案）**：**否决**——`TRAP-DEVMODE-PERSIST-GEMM`（`pattern-library/traps-runtime.md`，已实测验证）明确"Developer 模式 + persistent 分核 + gemm 混排 = 运行时崩溃（'unaligned UUB addresses'）"，本算子正是 Developer + gemm；核内串行 + 每个逻辑块单独 `T.gemm` 即属该崩溃形态。故**本设计不使用核内串行**（核内串行循环边界静态值不适用——未采用）。
    - **负载均衡**：物理核按 wave 调度逻辑核，每核承担的 tile 数差 ≤1；各 workload 波数 = `⌈logical/20⌉ ∈ {26, 13, 205}`，末波空闲比例 ≤ `(20 - logical%20)/20 ≤ 19/20` 仅作用于**最后一个末波**，整体效率 ≥ 98%（如 512 → 26 波中末波 12/20 busy，整体 512/(26×20) = 98.5%）。单个 GEMM tile 的 Cube 工作量（⌈K/64⌉ 轮 × 128×128×64 MAC）远大于一次核启动开销，故核启动开销被摊薄（定量见 §1.6.0 设计期估算：b4-4k 主导项为 L1 端口操作数流 2.86ms，非 launch）。
    - **备选（Stage 4 若实测 launch 开销显著）**：切换 **Expert 双 Scope 结构**（`PL-1.16-expert-dualscope-bypass`）实现核内串行 persistent——该结构已由两任务实证可绕开 Developer 的 persistent+gemm 崩溃；载体见 §3.5.3 `examples/gemm/matmul.py`。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> §1.6.2 已确认无逐元素标量循环；本表仅含 block 级 / tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| tile 级并行（batch×M×N） | block 级并行 | `T.Kernel(grid, is_npu=True)` | 每个逻辑核处理一个输出 tile（折叠网格 R1） |
| K 方向 | tile 级迭代（流水） | `T.Pipelined(⌈K/block_K⌉, num_stages=2)` | K 分块累加，双缓冲重叠搬运与 MMA |
| M/N 方向（tile 内） | 隐式（Cube MMA） | `T.gemm` | MMA 内建并行，无显式循环 |
| epilogue | 隐式（L0C→GM + 编译器自动 VCast） | `T.copy` | L0C→GM 直写，无用户逐元素循环 |

### 6.2 循环伪代码

```python
with T.Kernel(grid, is_npu=True) as (cid, _):          # block 级并行
    <decode bz/by/bx；分配 a_s/b_s/c_l>
    for ks in T.Pipelined(T.ceildiv(k, block_K), num_stages=2):   # K 维流水
        T.copy(a[...], a_s); T.copy(b[...], b_s)
        T.gemm(a_s, b_s, c_l, initC=(ks == 0))
    T.copy(c_l[0:tile_m, 0:tile_n], c[bz, m0:m0+tile_m, n0:n0+tile_n])
```

### 6.3 流水线优化

使用 `T.Pipelined(num_stages=2)`：L1 双缓冲 A/B tile，搬运（MTE2 GM→L1）与 Cube MMA 重叠。`num_stages=2` 是容量与深度的折中点（L1 每 stage 32KB）；`k_iters = K / block_K`（`block_K = 64 if K%64==0 else 32`：K≥256 → ∈ {4,16,64}，K=128 smoke → 4），满足 `docs/开发指南.md` §3.2 建议的流水迭代 ≥4–6。`num_stages>2` 的探索交 Stage 4。

### 6.4 尾块处理

见 §5.4：K 维无尾块（`block_K` 由 `K` 选取保证整除）；M/N 尾块的**装载**统一为整块装载、**写回**由 `T.min` 动态 extent 切片 `T.copy` 精确裁剪（探针 D 实测通过），不做 host padding。**单一形态与 §3.3/§6.2 伪代码一致**。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式）。

### 7.2 同步点说明

Developer 模式下编译器自动插入 GM↔L1↔L0C 之间所需的同步（`tilelang-cube-skill` api-cube.md："Don't use explicit T.Scope for cube sections in developer mode kernels"）；本设计**不手写** `T.sync_block_set/wait`、`pipe_barrier`、`barrier_all` 等。手工同步仅在 Expert 备选路径（§1.6.3 #2）中按双 Scope 结构引入。

### 7.3 pass_configs 配置

不设 CUDA 专属 pass_configs（源 `tl.disable_warp_specialized` 为 CUDA-only，npuir 无意义，移除）。`fast_math` 对纯 GEMM 无影响，不启用（保持与 golden 一致的 IEEE 舍入）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 以 **§0.1 源算子语义**为唯一依据，**直接复用 `torch.bmm`**（源测试 `examples/TileOPs/tests/ops/test_bmm.py` L33 `ref_program`），不复刻 §0.6 的 NPU 算法——保证验证独立性。

```python
import torch

def golden_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BmmFwdOp 语义 = torch.bmm（严格 3D-3D，无广播）。"""
    return torch.bmm(a, b)      # a:(B,M,K), b:(B,K,N) -> (B,M,N)
```

### 8.2 精度标准

沿用 NPU 测试 fixture 的门限（`examples/TileOPs/tests/ops/test_bmm.py` L103-105）：

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

### 8.3 L0 门槛测试计划

Stage 3 首过的 L0 门槛：(1) **smoke** `(B,M,N,K,dtype) = (4,128,128,128,fp16)`——最小稳健 shape，验证编译 + 数值（`golden_bmm` 比对，fp16 门限）；(2) **bf16 smoke** `(4,128,128,128,bf16)`；(3) **非整除边界** `(4,200,300,128,fp16)`——验证 §5.4 尾块策略；(4) 三条 host 契约负例（batch 不等 / K 不等 / rank≠3 抛 `ValueError`，对应 `test_bmm.py` L109-134）。完整分层套件（L1 full / L2 / Boundary）交 `tilelang-op-develop`，不在此枚举。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **Developer + persistent + gemm 崩溃**（`TRAP-DEVMODE-PERSIST-GEMM`）：分核方案已规避（§5.5），不得在 Stage 3/4 擅自引入核内 `T.serial` persistent。
2. **谓词 fragment→GM 直写触发运行期 aicore 异常**（`The MPU address access is invalid`，CG-2026-0001(d)，探针 C 复现）：epilogue 必须用切片 `T.copy`，不得回退谓词逐元素写。
3. **NPU ≤ 2D grid**（探针 A + `T.Kernel.md` §2.3）：不得改回 3D grid（batch 必须折叠）。
4. **装载侧尾块越界读**（§5.4）：装载统一整块，末 M/N tile 会越界读尾行/尾列；探针 D（b4-200×300-128，末 tile 越界 56 行）实测可容忍——不触发故障、不影响输出（被 epilogue 切片裁剪）。**残余风险**：极端大 block / 更长越界跨度的行为未逐 shape 覆盖，Stage 3 需在 L0 用例（§8.3 第 3 条 非整除边界）复测确认。
5. **Cube MAC 算力未实测**：`CONST` 表无 Cube 峰值 FLOPs 常数，b4-4k 是否 compute-bound 未定（§1.6.0 估算注记），Stage 4 对账。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | `block_M·block_N·4 > 128KB` | 编译/运行失败（`cc overflow`） | 保持 128×128（64KB）或降 block_N |
| L1 溢出 | `num_stages` 过大 / block 过大 | 编译失败（`cbuf overflow`） | 保持 `(128·64+64·128)·2·stages·1.7 < 512KB` |
| aicore 崩溃 "MPU address…" | epilogue 用谓词 fragment→GM 直写 | 运行时崩溃 | 用切片 `T.copy` 写回（§0.6 R2） |
| fp16 输入配 fp16 累加 | `alloc_fragment` dtype 设为 fp16 | 运行时卡死（`T.gemm.md` §2.3） | 累加器固定 fp32 |
| 3D grid | 直接用源 `T.Kernel(cn,cm,batch)` | `AssertionError`（探针 A） | 一维折叠 |
| persistent + gemm | 核内串行 + `T.gemm`（Developer） | 运行时崩溃 | 用全量逻辑核 launch（§5.5） |

### 9.3 特殊场景处理

- **非整除 M/N**：§5.4 整块装载 + 动态 extent 切片写回（无 host padding）。
- **K 为 16 倍数但非 64 倍数**：按 `block_K = 64 if K%64==0 else 32` 选取（如 K=128 → 32），保证 K 整除、无 K 尾块。
- **极小 shape（逻辑核数 < 物理核数）**：smoke 用例 4 核直接 launch，无需适配。
- **dtype**：fp16/bf16 共用同一 kernel（`dtype` 编译期特化）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/bmm/_bmm_kernel/
├── _bmm_kernel.py          # 算子实现（Stage 3 产出）
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，全 EQUIV_PASS）
├── RETROSPECTIVE.md        # Stage 1 复盘（自进化钩子）
└── history_version/
    ├── design_probe_bmm.py     # 设计期探针 1（3D grid 拒绝 + 初判）
    └── design_probe_bmm_v2.py  # 设计期探针 2（折叠网格 + slice epilogue 定稿形态）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成（EQUIV_PASS×2） | §1.6.1 等价性验证 |
| `history_version/design_probe_bmm.py` | 已完成 | 探针 1 |
| `history_version/design_probe_bmm_v2.py` | 已完成 | 探针 2 |
| `_bmm_kernel.py` | 待实现（Stage 3） | 算子实现 + golden |
| `test__bmm_kernel.py` | 待实现（Stage 3，可选） | 测试文件 |

### 10.3 命名与实现顺序

- 项目目录 `examples/bmm/`，算子目录 `_bmm_kernel`，实现文件 `_bmm_kernel.py`。
- 实现顺序：✅ 设计文档（DESIGN.md）→ ✅ 等价性验证（verify_equiv.py）→ ⬜ Golden（`torch.bmm`）→ ⬜ 算子实现（`_bmm_kernel.py`）→ ⬜ L0 门槛测试（§8.3）。

---

## 11. 修订说明（revision v1 相对 design_v0）

> 触发：Stage 2 检视不通过（`REVIEW.md` 问题 1，阻塞级）。本节说明本次相对 `history_version/design_v0.md` 的关键调整与为何不会再犯同一错误。

### 11.1 阻塞级修订：Developer epilogue 机制前提纠正

- **被推翻的判断**：v0 把 Developer 模式 `T.copy(fragment/L0C→GM)` 的 fp32→fp16/bf16 量化写成 **fixpipe / `enable_nz2nd`** 通路，并据此断言「本算子不含 Vector、无跨引擎数据流」、把 OPT-2 收益量化为「fixpipe 单跳、无 UB 往返」。
- **正确依据（本次亲自核对）**：`docs/Tilelang.language/内存操作/T.copy.md` 支持方向表第 15 行——Developer 的 `L0C→GM` 为「**经 VCast 通用路径**」、Expert 才是「自动转换为 Fixpipe 搬运」；同文 §2.3 第 6 条明确 fixpipe 等价「**仅 Expert 模式**」；§2.3 第 4 条明确 Developer 跨 dtype 由编译器自动插入 `hivm::VCastOp`(RoundMode=RINT)；`tilelang-cube-skill`（SKILL.md / references/api-cube.md / references/nz-format.md）三处均标 `store_fixpipe`/NZ 通路为 **Expert-only**、Developer 保持 ND。
- **本次修改的章节**：统一改写 §0.4#6、§0.5#6、§0.6 R2、§1.2#2、§1.4、§1.5、§1.6.0（R1#2 / R4#2 / 调研结论）、§1.6.1 OPT-2、§1.6.2#4/#5、§1.6.3#1、§2.2/§2.3、§3.1/§3.2、§3.4/§3.5、§4.4/§4.5、§5.1、§6.1——把 Developer epilogue 明确为「`T.copy`(L0C→GM) 保持 ND，fp32→fp16/bf16 由编译器自动 `hivm::VCastOp`(RINT) 量化」；删除 Developer 语境下的 `enable_nz2nd`/`channel_split`/`Fixpipe` 术语与 `T.copy.md` §2.3 第 6 条引用（Expert 备选 #2 仍保留 `load_nd2nz`/`store_fixpipe` 的 Expert 描述）。
- **重估 OPT-2**：收益由「fixpipe 单跳、无 UB 往返、256MB 片上往返」改为「去除源 GPU 显式 `c_smem` staging 缓冲（32KB/tile）的分配与写回路径」；删去「无 UB 往返」绝对化主张，精确片上流量标注待 Stage 4 实测（VCast 内部临时张量由后端管理）。
- **修正「无 Vector」表述**：§2.2/§2.3/§5.1 由「纯 Cube / 不含 Vector 后处理 / 无跨引擎数据流」改为「Cube 主体；epilogue 舍入由编译器自动插入向量化 `hivm::VCastOp`（Vector 单元），属后端自动 CV-split 的隐式跨引擎协同；无**用户编写**的 Vector 后处理与用户级跨引擎数据流」。
- **实现形态不变**：`T.copy(c_l[0:tm,0:tn], c[...])` 在 Developer 下合法（探针 D 实测通过），故 §3.3 计算伪代码本身不改；本次修订只纠正机制描述 / 引用 / 收益口径与「无 Vector」表述。
- **是否切 Expert**：**维持 Developer**（与 §2.1/§5.5 一致）；若 Stage 4 实测显式 fixpipe/NZ 通路更优，按 §1.6.3 备选 #2 切换 Expert 双 Scope（`T.Scope("Cube")` + `T.alloc_L1`/`T.alloc_L0C` + `load_nd2nz`/`store_fixpipe`，载体 `examples/gemm/matmul.py`）。

### 11.2 建议级修订 1：统一尾块装载形态

- v0 三处不一致：§3.3/§6.2 伪代码用**整块装载**、§5.4 把「动态 extent 装载」列为**主选**、§9.1#4 称「越界读未实测」。
- 本次**择一落定**：统一为「**整块装载 + epilogue 切片裁剪**」——该形态正是 §3.3/§6.2 伪代码与**探针 D 实际行使**的形态（b4-200×300-128，bm=bn=64，末 tile 越界读 56 行，PASS）。§5.4 改写为整块装载（附探针 D 证据）、§6.4/§9.1#4 同步更新（越界读经探针 D 实测可容忍、不影响输出；残余风险按 shape 复测）。

### 11.3 建议级修订 2：参数化 block_K

- §3.3 工厂签名与 §5.2 明确 `block_K = 64 if K % 64 == 0 else 32`（恒满足 `K % block_K == 0` 与 `≥16` 分形）；§5.3/§6.3/§9.3 同步（K=128 smoke → 32，`k_iters=4`，避免 `K/64=2` 的过浅流水）。

### 11.4 为何不会再犯同一错误

- **机制归属必查模式限定词**：凡引用 `T.copy` / 存储路径条款，先核对支持方向表的 **Developer 列 vs Expert 列**与条款正文的模式限定词（本次错因即引用 §2.3 第 6 条这条 **Expert-only** 条款支撑 Developer 设计）；涉及 `store_fixpipe`/`load_nd2nz`/`enable_nz2nd`/`channel_split`/NZ 字样，一律先判编程模式。
- **「无 Vector / 无跨引擎」断言必对照跨 dtype 路径**：Cube 算子只要 epilogue 含跨 dtype 量化，Developer 后端即引入 Vector VCast，不得再宣称纯 Cube。
- **「主选形态」必与自身伪代码 + 探针实测对齐**：不从被否定/未实测的形态反推主选，避免把已验证形态标为未实测。
- **不改动的既有结论**（本次静默保留）：网格折叠 R1、分核策略 §5.5（三要素 + 物理核实查 20）、`initC` 融合 R3、L0C/L1 预算、NZ/布局候选矩阵与 Expert 备选裁决计划、golden 独立性、`verify_equiv.py` 两项 EQUIV_PASS。
