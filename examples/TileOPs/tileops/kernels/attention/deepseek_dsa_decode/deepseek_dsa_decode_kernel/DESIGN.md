# `_sparse_mla_kernel` 算子设计文档（DeepSeek 稀疏 MLA Decode, top-k 稀疏在线 softmax, NPU 迁移）

> 迁移任务（scenario=migration, harness 子模式）。项目 `deepseek_dsa_decode`，算子目录 `examples/deepseek_dsa_decode/_sparse_mla_kernel/`。
> 源算子：`DeepSeekSparseAttentionDecodeWithKVCacheFwdOp` 的函数 `_sparse_mla_kernel`
> （`/home/developer/TileOPs/tileops/kernels/attention/deepseek_dsa_decode.py`，L17 起；Hopper **warp-specialized** WGMMA 实现）。
> 目标：`target="npuir"`（华为昇腾 NPU），编程模式 **Developer**（用户指定）。
> 工具链戳：`tilelang 0.1.2+00aeb862c8cad637350900a64d4f9e0b840427c7`（`VERSION=0.1.2`，git `00aeb86`）/
> 设备 **Ascend910B3**（`NPUUtils.get().get_arch()` 实查）/ `npu-smi 25.5.0` / CANN 版本未在本机核实（沿用仓内 9.0.0 口径，标注「未实测」）。

> ### 修订说明（revision 1，相对上一版 `history_version/design_v0.md`）
> 本次为 Stage 2 检视不通过（1 阻塞 + 4 建议）后的设计修订。**相对上一版的关键调整**：
> 1. **阻塞项（§0.3/§0.4/§0.5/§0.6/§1.4/§1.6.0/§3.5.2）**——更正源算子 head 复制维读数：
>    源码 L78 `head_kv = heads//kv_group = 128`、L130-137 `if head_kv>64: replicate_h = head_kv//64` ⇒ **本任务 `replicate_h=2`、`h_per_block=64`、
>    源 grid x `= S·replicate_h = 2048`**（v0 误把 `head_kv` 当作 `H_kv`，断言 `replicate_h=1`/「全头单 block」）。**补列被遗漏的源优化 O17
>    （head 复制拆分，register-pressure 驱动）**并纳入 §0.5 四态处置（重新设计 → R4）与 §0.6 R4 承接。逐处重算派生数字：源寄存器态
>    `acc_o=2×[64,256]f32=128KB`（+`acc_s[64,64]f32`16KB）、源 KV 遍历 **2 遍 / 5.12 GB**、§1.4 差异陈述、§3.5.2 差异表。**结论方向不变**
>    （源累加在 L0C=128KB 不可行 ⇒ 须 head 分块 ⇒ 主选 BH=32；roofline 主选估计 [4.0,9.4] ms 不受影响）。
> 2. **建议 2**——更正 `T.vexp` 的 bf16 论断：两 API 文档 dtype 矩阵（`T.vexp.md`/`T.vexp2.md` §2.2.1）**均标 bf16 ✗**；本设计 softmax 的 exp
>    在 **fp32 域**执行，故不触发该缺口。引用 `TRAP-bf16-vop-doc-matrix-divergence`，并规定 bf16 缓冲直调超越函数须加设备 micro-probe。
> 3. **建议 3**——修正 §4.5 UB 小计（统一为逐行求和 **173,568 B ≈ 169.5 KB**，消除「≈67KB / 171KB」不自洽）；显式取 `~1.7x` 与 `×1.10–1.12`
>    两档（物化态在 `~1.7x` 下 **288KB > 192KB 不可编译** ⇒ **削减方案改为必需**，削减态 ≈41.5KB，×1.7 ≈71KB ✓）；§1.6.3 备选 BH=64 补
>    「膨胀系数 > 1.26x 则不可行」判定阈值分支。
> 4. **建议 4**——§1.6.3 补「softmax 归约 lane 映射」候选枚举（水平归约 vs 垂直扫描）及量化弃选理由。
> 5. **建议 5**——更正 `verify_equiv.py` docstring：OPT-2 = **重结合 + `m·s` CSE 提升**（非「scale 折入 Q」）；`Q~=Q·s` 归入被否决段（OPT-X）。
> **为何不会再犯同一错误**：①§0.3 现**逐行引用源码行号并显式代入本任务参数**推导 `head_kv=128>64⇒replicate_h=2`，O17 已入 §0.4 清单 /
> §0.5 处置 / §0.6 R4 承接，堵住「遗漏 → 静默丢弃」通路；②bf16/UB 论断均附具体 KB 条目与文档 §2.2.1 行；③UB 小计逐行求和、膨胀两档全列。
> **首次设计的全部有效结论保留**：Developer MixCV、WGMMA→`T.gemm` 重设计、`T.Pipelined`、一维非持久 grid、indices 单次 clamp + 列掩码 + 行连续 `T.copy`、
> OPT-1/2/3、§5 分核三要素（`num_logical=4096` / 物理 20 实查 / 非持久波次）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 三问框架（读源码全文 → 语义 / 算法 / 优化手段）→ 算法调研（Phase R，见 §1.6.0）→ 硬件耦合性判定（§0.5）→ NPU 重设计（§0.6）。
> 方法论：`tilelang-op-design` skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。本章结论驱动 §1–§11。

### 0.1 源算子语义（做什么）

**数学语义**：对每个 query token `(b, s)`、每个头 `h`，按该行的 top-k 稀疏索引 `indices[b, s, g, :]`
从 KV cache 中取 `topk` 个 KV 条目，做因果裁剪的 softmax 注意力：

```
o[b,s,h,:] = Σ over 有效 idx_j of softmax_j( (q[b,s,h,:] · kv[b,idx_j,g,:]) / sqrt(D + dim_tail) ) · kv[b,idx_j,g,:][:D]

D = 512（no-PE 主体）, dim_tail = 64（共享 RoPE 尾）; 拼接维 Dqk = D + dim_tail = 576
有效判据: idx_j >= 0 且 idx_j <= max_kv_i,  max_kv_i = (q_i + 1 - stride_kv) // stride_kv,  q_i = q_start_index_s + s_i
```

**规约语义**：
- 规约轴 = top-k 维（每个 (b,s,g,h) 行在 `topk` 个候选上做 softmax + 加权和）。
- **累加顺序**：源实现按 `topk` 分块（`block_i`）做 **online softmax**（分块 running max/sum + 增量重缩放），
  累加顺序 = 块序 × 块内序，且**随 block_i 改变**。NPU 重设计保持 online 语义，但块大小 `block_N` 与块内归约实现不同
  → 累加分组变化（fp16/bf16 舍入差异，§0.6/§8 评估，tolerance 内）。
- **dtype 语义**：输入/输出 `float16 | bfloat16`；score 的 gemm 累加 **fp32**（`accum_dtype="float"`）；softmax 全程 fp32；
  `P` 在送入第二个 gemm 前 **cast 回输入 dtype**（源 `T.copy(acc_s, s_shared)`，`s_shared` 为输入 dtype）；
  最终 `o = acc_o / sumexp` 后再 cast 回输入 dtype 写出。
- **边界语义**：
  - 掩码位置（`idx > max_kv_i`）：score 置 `-T.infinity`，`exp2` 后精确为 0，不贡献。
  - `-T.infinity` 参与 `reduce_max(clear=False)`；若整块全掩码则行 max 保持 -inf，源用 `m_i` 初始 `-2**30` 规避 `-inf - -inf`（L225）。
  - `stride_kv > 1` 时源注释明确「前 `kv_stride-1` 个 token 的输出为 nan（未被使用）」；**本任务 manifest 全部 `stride_kv=1`**，该退化不触发。
  - `q_start_index_s > 0` → `cp0 = (q_start_index_s == 0) = False` → grid 的 s 维 = `seq_len`（不含 cp0 偏移），`max_kv_i = q_start_index_s + s_i`（stride_kv=1）。
  - **去重语义差异（必须在 golden 中显式处理）**：源 kernel 对 `topk` 个条目**逐一**累加（重复索引会重复计数）；
    而 TileOPs 参考实现用 `scatter` 构造 mask（**重复索引被去重**）。**manifest 与测试数据用 `randperm` 生成互异索引**
    （`supply_prog` L583），两者等价。golden（§8.1）按参考实现语义（去重）实现，测试数据保证互异。
  - 全掩码行（该行无任何有效索引）：参考实现 softmax(all -inf) → nan；源 kernel `exp2(nan)` → nan。属退化输入，
    manifest 不产生（`supply_prog` 保证每行 ≥1 有效）。列入 §9 风险。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（**优先移植 TileOPs 参考实现**），不复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `q` | `(B, S, H, D+Dt)` = `(1, 1024, 128, 576)` | float16 / bfloat16 | 查询；BSHD；尾部 `Dt=64` 为共享 RoPE key |
| `kv` | `(B, S_kv, H_kv, D+Dt)` = `(1, 2048, 1, 576)` | same_as(q) | MLA 压缩 KV cache；`H_kv = kv_group = 1` |
| `indices` | `(B, S, H_kv, topk)` = `(1, 1024, 1, 2048)` | int32 | 每 (b,s) 行的 top-k 稀疏 KV 位置 |
| `o`（输出） | `(B, S, H, D)` = `(1, 1024, 128, 512)` | same_as(q) | **输出 shape `(B,S,H,D)`**（迁移必需字段；`dim_tail` 不进输出） |

构造参数（host 侧）：`dim=512`、`tail_dim=64`、`topk=2048`、`kv_stride=1`、`q_start_index_s=1024`、
`kv_group=1`、`sm_scale=None`（→ `(1/sqrt(576))*LOG2E`）、`is_causal=True`、`cp0=False`、`dtype=float16`。
manifest 另有 `smoke (1,128,128,576)/(1,256,1,576)/topk=256/q_start=128`、`(1,512,128,576)/(1,4096,1,576)/topk=1024`、
`(1,256,128,576)/(1,8192,1,576)/topk=512`。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（入口签名 → Kernel → 原语 → 内存分配 → 循环调度 → host 逻辑 → 编译配置）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | Q → shared 拷贝（3 份：左半 D/2、右半 D/2、tail） | `q[b_i, s_i, h0:h1, :]` | `q_shared_{l,r}`、`q_tail_shared` | `q` 加载 |
| 2 | `sumexp←0`、`m_i←-2**30`、`acc_o_l←0`（初值） | — | fragment | online softmax 初值 |
| 3 | producer 取 `indices`，判 `<= max_kv_i`，async 搬运 `kv[b_i, idx, g_i, :]` → 双 buffer shared | `kv`、`indices` | `kv_shared_{0,1}_{l,r}`、`k_tail_shared_{0,1}` | top-k 稀疏取 KV（gather） |
| 4 | `acc_s = 0/-inf`（按 `is_kv_valid` 逐元素填充） | `is_kv_valid` | `acc_s` | 因果掩码 |
| 5 | `wgmma_gemm(q_shared_l, kv_shared_l, acc_s, transpose_B)` ×2 + tail ×1（3 次 WGMMA 累加） | Q/KV shared | `acc_s` | `q·kvᵀ`（576 维，切成 l/r/tail 三段） |
| 6 | `m_i_prev←m_i`；`reduce_max(acc_s, m_i, clear=False)` | `acc_s` | `m_i` | running max |
| 7 | `alpha = exp2((m_i_prev − m_i)·sm_scale)` | `m_i` | `alpha_local` | online 重缩放因子 |
| 8 | `acc_s = exp2(acc_s·sm_scale − m_i·sm_scale)` | `acc_s` | `acc_s`（=P） | softmax 分子（减 max 稳定化，2 底） |
| 9 | `reduce_sum(acc_s, sumexp_i)`；`sumexp = sumexp·alpha + sumexp_i` | `acc_s` | `sumexp` | softmax 分母（online） |
| 10 | `acc_o_l *= alpha`（rescale）；`T.gemm(s_shared, kv_shared_l, acc_o_l)` | `acc_s→s_shared`、KV | `acc_o_l` | `P·V`（左半 D/2） |
| 11 | 对称的 Buffer 1 分支（步骤 3–10） | — | — | 双缓冲流水（`n_i` 按 2 分块） |
| 12 | 收尾：`sum_exp_shared←sumexp`；`acc_o_l /= sumexp`；`sumexp←log2(sumexp)+m_i·sm_scale`；写 `output` | — | `o[b_i,s_i,h0:h1,0:D/2]` | 归一化 + LSE（供反向）|
| 13 | 消费者 2（`tx∈[128,256)`）：`acc_o_r` 的 rescale + `T.gemm(s_shared, kv_shared_r, acc_o_r)` + 归一化 + 写右半 | — | `o[..., D/2:D]` | `P·V`（右半 D/2） |

**数据流与内存访问模式（源硬件视角）**：

```
GM[q] --T.copy--> SMEM{q_shared_l,r, q_tail_shared}
GM[kv] --cp.async(producer,128线程)--> SMEM{kv_shared_{0,1}_{l,r}, k_tail_shared_{0,1}}   (双缓冲)
SMEM[q],SMEM[kv] --wgmma_gemm--> register fragment[acc_s fp32] --exp2--> SMEM[s_shared fp16]
   --T.gemm--> register fragment[acc_o_{l,r} fp32] --/=sumexp--> SMEM[o_shared] --T.copy--> GM[o]
GM[indices] --标量读--> is_kv_valid(shared) / 地址算术
```

- buffer 复用：`o_shared_l = q_shared_l`、`o_shared_r = q_shared_r`（L178-179，Q 缓冲死后复用作输出中转）。
- `is_kv_valid` 为 shared bool 数组，供消费者读。

**循环与并行结构**：`T.Kernel((S - stride_kv + 1 if cp0 else S) * replicate_h, B, kv_group, threads=384)` —— **三维 grid**；
grid x 维 = `S·replicate_h`（cp0=False）；
384 线程按 `T.get_thread_binding()` 分成 **三段 warp-specialized 角色**：
- `tx < 128`：消费者 1（左半 D/2：Q 加载 + QK + online softmax + PV 左半 + 归一化 + 写左半）；
- `128 ≤ tx < 256`：消费者 2（右半 D/2：rescale + PV 右半 + 写右半）；
- `tx ≥ 256`：生产者（读 indices、判有效、cp.async 搬运 KV 双缓冲）。
内层 `for i_i in T.serial(ceildiv(n_i, 2))`（`n_i = topk/block_i`，Buffer 0/1 交替），双缓冲 + 命名屏障握手。

**head 维（关键：源按 head 复制拆分，非全头单 block）**：源码 L78 `head_kv = heads // kv_group`；L111 `padded_h = max(next_pow2(head_kv), 16)`；
L130-137 `if head_kv > 64: replicate_h = head_kv // 64`（否则 1），`h_per_block = padded_h if replicate_h == 1 else 64`；
L204-210 `s_i = bx // replicate_h`、`h0 = g_i*padded_h + (bx % replicate_h)*64`。
**本任务 `heads=128`（manifest q_shape `[1,1024,128,576]`）、`kv_group=H_kv=1` ⇒ `head_kv = 128//1 = 128 > 64`**
⇒ **`replicate_h = 128//64 = 2`、`h_per_block = 64`、`padded_h = 128`**；源 grid x 维 = `S·replicate_h = 1024×2 = 2048`（cp0=False），
grid = `(2048, 1, 1)`。每 block 处理 **64 个头**（`h0 = (bx % 2)·64`），同一 query 行 `s_i` 由 `replicate_h=2` 个 block（各 64 头）共同覆盖
——即源把 128 个头**拆成 2 个 64 头块**，这是 **register-pressure 驱动的 head 复制分解**（源优化手段，见 §0.4 O17）。

**host 侧逻辑**：
- `SparseMlaKernel.supported_archs = [90]`（Hopper 门禁）；`DeepSeekSparseAttentionDecodeWithKVCacheFwdOp.default_kernel_map`
  在 `is_hopper()` 为假时**直接 raise**（`deepseek_dsa.py` L110-114）。
- 工厂 `_sparse_mla_kernel(...)` 带 `@functools.lru_cache(maxsize=32)`；`@tilelang.jit(out_idx=[-1], compile_flags=[...])`。
- `_sparse_mla_fwd_func(block_i, threads)` → `@T.prim_func`；autotune 搜 `block_i ∈ {64,128} × threads ∈ {384,512}`。
- 无 host 侧数据预处理/后处理（输入直接进 kernel）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **warp-specialized 生产者/消费者**（384 线程三段） | 搬运与计算重叠 | `T.get_thread_binding()` 分支 + 角色划分（tx<128 / 128–256 / ≥256） | CUDA warp/thread 模型 | **硬件强相关**（NPU 无 thread 级角色分工） |
| O2 | **WGMMA**（`T.wgmma_gemm` / `T.wait_wgmma`） | SM90 tensor core 矩阵算力 | warpgroup MMA | Hopper tensor core（SM90） | **硬件强相关**（NPU 有 Cube 等价） |
| O3 | **双缓冲 shared memory**（`kv_shared_0/1`，Buffer 0/1 交替） | 搬运/计算流水不 stall | 两套 KV 缓冲交替 + 屏障相位 `(i_i&1)` | —— | **模式可移植** |
| O4 | **mbarrier 命名屏障**（`T.alloc_barrier(arrive_count=…)` / `barrier_wait` / `barrier_arrive` / `cp_async_barrier_noinc`） | 生产者-消费者精确握手 | 每个 buffer 一 `ready`/`free` 屏障 + 计数到达 | SM90 mbarrier + cp.async 计数 | **硬件强相关**（NPU 无命名 mbarrier） |
| O5 | **cp.async 异步搬运**（`T.attr("default","async_scope",1)` + `T.vectorized(8)`） | 搬运与计算重叠、绕寄存器 | 异步拷贝引擎 + 向量化 8 元素 | SM90 cp.async | **硬件强相关**（NPU 有 MTE 等价） |
| O6 | **寄存器累加器 + 寄存器配额**（`acc_o_{l,r}`、`acc_s` fragment；`T.set_max_nreg(240/168/80, …)`） | 数据驻留寄存器、按角色配寄存器 | Hopper 大寄存器堆 + WGMMA fragment 布局 | 大寄存器堆 / `setmaxnreg` | **硬件强相关** |
| O7 | **Q/KV 的 l/r/tail 三段切分**（`d//2` ×2 + `d_tail`） | 降 WGMMA K 维寄存器压力（3 次小 WGMMA 代替 1 次 576 维） | 把 576 维 dot 拆成 256+256+64 | WGMMA 寄存器/布局约束 | **硬件强相关**（WGMMA 特有动机） |
| O8 | **online softmax（flash 结构）** | 消 O(topk) 中间缓冲、单遍扫描 | 分块 running max/sum + 增量重缩放 | **无（纯算法层）** | **可移植** |
| O9 | **mask 跳过装载**（`if is_kv_valid: … cp.async`） | 省无效 KV 的搬运 | 无效索引不发起搬运，score 置 -inf | —— | **模式可移植**（NPU 可保留，或换 clamp+掩码） |
| O10 | **exp2 + LOG2E 折入 scale**（`sm_scale = (1/sqrt(576))*LOG2E`，`exp2`） | 用 2 底指数（`exp2` 硬件更快） | 换底 `exp(x)=exp2(x·log2e)` | —— | **算法层**（可移植；NPU 有 `T.vexp2`/`T.vexp`） |
| O11 | **head 维 padded 到 next_pow2** | 对齐（**与 O17 的拆分不同：padding 只对非 2 幂 head_kv 生效，此处 `head_kv=128` 恒等**） | `padded_h = max(next_pow2(head_kv),16)` | —— | **可移植** |
| O12 | **-inf 掩码 + `-2**30` init** | 因果掩码 | score=-inf → exp=0；m init 有限大负数避 inf-inf | —— | **算法层**（可移植；NPU 需注意 `-inf` 参与 reduce 的 NaN 风险） |
| O13 | **融合 rescale epilogue** | 归一化不落中间 | `acc_o /= sumexp` 直接写 shared→GM | —— | **可移植**（Vector epilogue） |
| O14 | **lru_cache + autotune(block_i, threads)** | 编译缓存 + 参数搜索 | host 层 | GPU 特有参数（threads） | **部分**（缓存可移植；`threads` 无 NPU 语义） |
| O15 | **三维 grid**（bx=seq×rep, by=batch, bz=kv_group） | 任务映射 | 3D blockIdx | CUDA grid | **硬件强相关**（NPU 一维 Kernel） |
| O16 | **CUDA 编译 flags**（`--use_fast_math -O3 --ptxas-options=… -U__CUDA_NO_HALF_*`） | nvcc/ptxas 调优 | NVCC/pxtas | **硬件强相关**（CUDA 专有） |
| O17 | **head 复制拆分**（`replicate_h = head_kv // 64`，`head_kv>64` 时） | **控制 Hopper 寄存器压力**：使单 block 的寄存器态（`acc_o`/`acc_s` fragment）可容纳 | 把 head 维拆成 `replicate_h` 个 64 头 block，grid x 维 ×`replicate_h`；`s_i = bx//replicate_h`、`h0=(bx%replicate_h)·64` | Hopper 大但有限的寄存器堆（单 block 需同时驻 `acc_o`/`acc_s`） | **硬件强相关**（NPU 侧动机等价 = L0C/UB 容量；→ R4，`block_heads=32`） |

> 识别不出来 ≠ 不存在。O1–O17 全部列出；未列出者会在迁移中被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 §1.6.0 算法调研结论）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| **语义：数学映射 / I/O 契约 / 规约 / dtype / 边界** | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；§0.1 |
| O8 online softmax（数学算法） | 算法 | 无 | — | **保留** | 纯算法层；§1.6.0 R2 确认且**必需**（topk=2048 的 score 矩阵 128×2048×4=1MB ≫ UB） |
| O10 exp2 + LOG2E 折入 | 算法 | 无 | 有（`T.vexp2`/`T.vexp`） | **等价替换** | 采纳 §1.6.1 OPT-1（自然 exp 域，`T.vexp` 无 tmp buffer；**两 API 文档 dtype 矩阵均不含 bf16**，见 §9 R-13） |
| O11 head padded next_pow2 | 算法 | 无 | — | **保留** | `H=128`（`head_kv=128`）本身即 2 的幂，padding 恒等（**padding 与 O17 的拆分是两件事**） |
| O12 `-inf` 掩码 + `-2**30` init | 算法 | 无 | 有 | **等价替换（+稳定化）** | 有限大数哨兵 `-1e38`（§1.6.1 OPT-3，`PL-1.11`）；规避 `-inf` 参与 reduce 的 NaN 路径 |
| O9 mask 跳过装载 | 优化 | 无 | 有（可保留） | **等价替换** | 改为**索引 clamp + 列掩码**（避免流水体内运行时分支；`TRAP-scalar-min-chain-offset` 的「块对齐 + 单次 clamp」形态） |
| O13 融合 rescale epilogue | 算法 | 无 | — | **保留** | Vector epilogue（`T.vdiv`/`T.vcast`） |
| O2 WGMMA（`T.wgmma_gemm`/`T.wait_wgmma`） | 优化 | Hopper SM90 tensor core | 有（Cube `T.gemm`） | **重新设计** | §0.6 **R1**：`T.gemm`（Developer）+ L1 操作数；分形 M/N≥16、K≥32（`docs/.../T.gemm.md`；`ascend-constraints.md §3`） |
| O1 warp-specialized 384 线程 | 优化 | CUDA warp/thread | 无 | **舍弃**（NPU 单程序模型） | §0.6 **R2**：单程序 Developer MixCV，Cube/Vector 由编译器自动切分（CV-split）；`TRAP-threads-kwarg-noop` |
| O4 mbarrier 命名屏障 + cp_async_barrier | 优化 | SM90 mbarrier / cp.async 计数 | 无 | **舍弃** | §0.6 **R3**：改用 `T.Pipelined(num_stages=2)` 的自动多缓冲 + 编译期同步（Developer 自动同步） |
| O3 双缓冲 | 优化 | — | 有（`T.Pipelined`） | **等价替换** | `T.Pipelined(num_stages=2)`；stage 数按 L1/UB 容量重算（§4.5） |
| O5 cp.async 异步搬运 | 优化 | SM90 cp.async | 有（MTE2 异步 + Pipelined） | **等价替换** | `T.copy` + `T.Pipelined`；向量化由编译器（32B 对齐） |
| O6 寄存器累加器 + `set_max_nreg` | 优化 | Hopper 大寄存器堆 / `setmaxnreg` | 部分（L0C/UB fragment） | **重新设计** | §0.6 **R4**：`T.alloc_fragment`（L0C 128KB 硬上限）→ 按 L0C 预算定 `block_heads` |
| O7 Q/KV 的 l/r/tail 三段切分 | 优化 | WGMMA 寄存器/布局 | 无该动机 | **舍弃** | §0.6 **R5**：单次 576 维 `T.gemm`（NPU 无 WGMMA K 维寄存器压力）；PV 用 `size=[BH,block_N,D]` 前缀列 |
| O14 lru_cache | 优化 | — | 有 | **保留** | 工厂 `@tilelang.jit` + Python 层缓存 |
| O14 `threads` 参数 / autotune 参数 | 优化 | CUDA 线程模型 | 无 | **舍弃** | `T.Kernel(..., is_npu=True)` 无 `threads`（`TRAP-threads-kwarg-noop`）；autotune 去掉 `threads` 维 |
| O15 三维 grid | 优化 | CUDA blockIdx | 无（一维 Kernel） | **重新设计** | §0.6 **R6**：一维 `T.Kernel(num_logical, is_npu=True)` + `cid` 反解 |
| O16 CUDA 编译 flags | 实现 | NVCC/ptxas | 无 | **舍弃** | npuir 无 `TL_ENABLE_FAST_MATH` 等 CUDA flags（`mha_decode_paged` 先例：照搬源 compile_flags 即 AttributeError） |
| O17 head 复制拆分（`replicate_h=2`，64 头/block） | 优化 | Hopper 寄存器堆（控制 register pressure） | 有（L0C/UB 容量等价动机） | **重新设计** | §0.6 **R4**：NPU 侧按 L0C=128KB **重新定 `block_heads=32`**（表达同一「head 拆分降片上压力」意图，块宽由容量决定）；`H % block_heads == 0` 断言 |
| （源外）Hopper `is_hopper()` 硬门禁 | host | SM90 | 无 | **舍弃** | 脚手架已完成（K5：`supported_archs=None`、去掉 `is_hopper`） |

> 每项处置依据已注明。**无「不可判定」项**。无静默丢弃：O1–O17 逐项有处置。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法；「等价替换 / 保留」项在实现中承接源意图。

**重设计项 R1：WGMMA → Cube `T.gemm`（L1 操作数 + L0C 累加）**

- **源方案**：`T.wgmma_gemm(q_shared_l, kv_shared_l, acc_s, transpose_B=True)` ×3 段 + `T.gemm(s_shared, kv_shared_l, acc_o_l)`，累加器在寄存器 fragment。
- **NPU 新算法**：`T.gemm(Q_shared, KV_shared, acc_s, initC=False, b_transpose=True)`（QK，K 维 = D+Dt = 576，一次完成）；
  `T.gemm(P_shared, KV_shared, acc_o, initC=False, size=[block_heads, block_N, D])`（PV，只取 KV 行的前 `D` 列）。
  操作数 Q/KV 驻 **L1**，输出累加在 **L0C/fragment**。分形约束 M/N ≥ 16、K ≥ 32（`docs/Tilelang.language/线性代数操作/T.gemm.md`；`ascend-constraints.md`）。
  `size=[BH, block_N, D]` 前缀列用法**有本仓先例**：`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` 的 `CubePVMatmul`
  （`T.gemm(l1_p, l1_kv_sparse, l0_c, initC=True, size=[heads, block, dim])`）。
- **语义保持论证**：QK 与 PV 的数学与源逐字一致（同一 `q·kvᵀ`、同一 `P·V`）；仅把 WGMMA 换成 Cube T.gemm。
  源把 576 维拆 l/r/tail 三次累加，NPU 一次 576 维累加 → **累加顺序变化**，但在 fp32 L0C 累加下差异 ≪ fp16 容差
  （记入 §9 风险；`reduce` 的块内归约序变化同源，见 §0.1 规约语义）。**边界**：`-inf/-1e38` 掩码经 `initC=False` 累加到预填空值，
  mask 位精确保持（`-1e38 + finite = -1e38`，§1.6.1 OPT-3 机器验证）。

**重设计项 R2：warp-specialized 生产者/消费者 → 单程序 Developer MixCV**

- **源方案**：384 线程按 `T.get_thread_binding()` 分三段角色（2 消费者 + 1 生产者），各角色独立循环 + 屏障握手。
- **NPU 新算法**：单个 `T.Kernel(num_logical, is_npu=True)` 程序体；Cube 侧 `T.gemm` 与 Vector 侧 v-prefix 链
  由 Developer 模式的**自动 CV-split** 切分（`tilelang-mixcv-skill` Developer MixCV 形态；先例
  `examples/deepseek_v32/sparse_mla_fwd.py`、`examples/indexer/indexer_fwd.py`、`examples/deepseek_v4/example_sparse_mla_fwd_kernel.py`）。
- **语义保持论证**：角色划分只是**并行组织**，不改变数学；NPU 单程序把三段计算合并为同一序列（Cube 段 + Vector 段），
  编译器按数据依赖安排引擎。**无数值差异**。

**重设计项 R3：mbarrier + cp.async 握手 → `T.Pipelined` 自动多缓冲**

- **源方案**：6 个命名屏障（`bar_k_{0,1}_ready/free`、`bar_s_scale_and_s_ready/free`）+ `cp_async_barrier_noinc`，
  Buffer 0/1 相位 `(i_i&1)` 手动握手。
- **NPU 新算法**：KV 主循环 `for k in T.Pipelined(topk // block_N, num_stages=2)`；缓冲与跨引擎同步由编译器
  自动插入（Developer 自动同步）。文件证据：`examples/deepseek_v32/sparse_mla_fwd.py` 的 `for i_i in T.Pipelined(NI, num_stages=2)`
  （同族 sparse MLA，含 online softmax，已在本仓上库）。
- **语义保持论证**：多缓冲只是调度（不改数学）；`num_stages=2` 对应源的双缓冲。**边界**：stage 数受 L1/UB 容量约束（§4.5）。

**重设计项 R4：寄存器累加器 → L0C fragment + 按 L0C 预算定 `block_heads`**（承接源 O17 head 复制拆分）

- **源方案**：`acc_o_l/acc_o_r` fragment `[h_per_block, d//2]`、`acc_s` fragment `[h_per_block, i_block]`（Hopper 大寄存器堆）。
  **本任务 `h_per_block = 64`（`replicate_h=2`，§0.3 / O17）**：源每 block 寄存器态 = `acc_o_l + acc_o_r` = `2×[64,256]f32 = 128KB`，
  再加 `acc_s [64, i_block]f32`（`block_i=64` 时 16KB）= **≈144KB**（源以 head 复制把 128 头拆成 2×64 头块，
  使单 block 寄存器态落到 Hopper 寄存器堆可容纳范围）。
- **NPU 新算法**：`acc_o` = fragment `[block_heads, D]` fp32 + `acc_s` = fragment `[block_heads, block_N]` fp32。
  **L0C 硬上限 128KB** ⇒ `(block_heads·(D + block_N))·4 ≤ 131072` ⇒ `block_heads·(512+64) ≤ 32768` ⇒ **`block_heads ≤ 56**。
  取 **`block_heads = 32`**（H=128 整除，4 个 head block）：`acc_o` 64KB + `acc_s` 8KB = 72KB ≤ 128KB ✓。
  单 kernel 处理一个 (b, s, head_block) —— grid 的 head 复制因 `block_heads < H` 出现，**KV 需被 `H/block_heads = 4` 次重读**
  （成为 roofline 主导项，§1.6.0 R4；备选 `block_heads=64` 见 §1.6.3）。
- **语义保持论证**：按 head 分块是**输出维分解**（各 head 相互独立，无跨 head 归约）→ 精确等价；`block_heads` 只影响
  L0C 驻留与 KV 重读，不改数值。**边界**：`H % block_heads == 0` 断言（128%32=0 ✓）。

**重设计项 R5：Q/KV l/r/tail 三段 WGMMA → 单次 576 维 T.gemm**

- **源方案**：`d//2` 左半、`d//2` 右半、`d_tail` 尾 三段分别 WGMMA 累加到同一 `acc_s`。
- **NPU 新算法**：`acc_s = Q_shared @ KV_sharedᵀ`（K = 576 一次）。理由：源切段的动机是 WGMMA 的 K 维寄存器/布局约束
  （L216-218 注释明确为「share the same 128-thread bounds」），NPU Cube 无该约束。PV 用 `size=[BH, block_N, D]` 取前 512 列
  —— 恰好对应源 `acc_o_l`（V 的左半）之外的语义：**输出只取 V 的前 D 列**（`dim_tail` 不进输出，§0.2）。
- **语义保持论证**：`q·kvᵀ` 拆 3 段之和 = 一次 576 维点积（加法结合，fp32 累加差异在容差内）；
  PV 中 `V = kv[:, :D]` 是语义定义（输出 dim = D），非近似。**边界**：`size` 前缀列须为连续前缀（KV 行内连续）✓。

**重设计项 R6：三维 grid → 一维 grid + `cid` 反解**

- **源方案**：`T.Kernel(S·replicate_h, B, kv_group, threads=384)`（本任务 `S·replicate_h = 1024×2 = 2048`，grid = `(2048, 1, 1)`）。
- **NPU 新算法**：`T.Kernel(num_logical, is_npu=True) as (cid, _)`，`num_logical = B · S · (H // block_heads)`；
  反解 `hb = cid % (H//block_heads); t = cid // (H//block_heads); s = t % S; b = t // S`。
  **非持久**（每核一任务；理由见 §5.5：`TRAP-DEVMODE-PERSIST-GEMM`——Developer + persistent + gemm 混排运行时崩溃）。
- **语义保持论证**：任务→(b,s,hb) 映射是一一满射。**与源的对应关系**：源 grid 为 `S·replicate_h`（本任务 2048），
  其 (s, head-replica) 由 `(bx // replicate_h, bx % replicate_h)` 联合确定；NPU 的 `num_logical = B·S·(H/BH)` 是**独立的 head 再分块**
  （BH=32 ⇒ 每 s 4 个 head block，共 `1·1024·4 = 4096` 任务），与源的 `replicate_h=2`（每 s 2 个 64 头 block）是
  **同一「head 拆分」思想、不同块宽**（源 64 头受寄存器堆约束，NPU 32 头受 L0C=128KB 约束；R4）。
  维度乘积 `B·S = 1024` 与源的 `S·B = 1024` 一致。**边界**：`num_logical` 编译期常量，反解用整除/取模（block 级标量，非热点）。

**重设计项 R7：`threads`/autotune threads 维 → 移除**

- **源方案**：`_sparse_mla_fwd_func(block_i, threads)`；`default_config={"block_i":64,"threads":384}`。
- **NPU 新算法**：`_sparse_mla_fwd_func(block_heads, block_N, num_stages)`（脚手架 L108-110 已按无 `threads` 调用）；
  `T.Kernel(..., is_npu=True)` 不带 `threads`（`TRAP-threads-kwarg-noop`：npuir 不消费 `block_size`）。
- **语义保持论证**：`threads` 无 NPU 语义，移除不改变计算（脚手架 K9 已完成接口改造）。

**重设计项 R8：mask 装载跳过 → 索引 clamp + 列掩码**

- **源方案**：`if is_kv_valid[bi]: with async_scope: kv_shared[...] = kv[b_i, indices_local[0], ...]`（无效索引不搬运）。
- **NPU 新算法**：gather 循环**无条件**搬运，索引先 clamp 到 `[0, S_kv−1]`（`idx_c = min(max(raw,0), S_kv−1)`，避免越界读）；
  有效性 `raw ≤ max_kv_i` 转成列掩码，写入 `acc_s` 的预填值（无效列 → `-1e38` 哨兵）。
  依据：`TRAP-scalar-min-chain-offset` 的「**块对齐扫描 + 单次 `T.min(·)` clamp**」是推荐绕法（多轮标量 min/max 累加链作偏移会 mis-compile；
  单次 clamp 安全）。同时避免在 `T.Pipelined` 循环体内引入运行时分支（`TRAP-tvm-parser-rules` / 运行时 if 打断调度）。
- **语义保持论证**：无效列的 KV 值被 clamp 成某有效行（有限值），但该列 score 被哨兵覆盖 → `P=0` → 对 `acc_o` 零贡献
  （`exp(-1e38)=0` 精确，§1.6.1 OPT-3 机器验证）。源的不搬运与 clamp+掩码**输出等价**。**边界**：`raw < 0`（源未定义）被 clamp 到行 0 并掩码为无效。

**源算法优化意图承接（§0.4 → §0.6/§1.6.0）**：O2（tensor 算力）→ R1；O1（并行组织）→ R2；O3/O4/O5（流水/握手）→ R3；
O6（寄存器累加）→ R4；**O17（head 复制拆分降 register pressure）→ R4**（同一「head 拆分」意图，NPU 侧块宽由 L0C 容量重定为 32 头）；
O7（K 维切段）→ R5；O9（省无效搬运）→ R8；O10/O12（exp 域/掩码）→ §1.6.1 OPT-1/OPT-3；
O8/O11/O13/O14（online/padding/epilogue/缓存）保留；O15/O16/`threads`/`is_hopper` 舍弃（无 NPU 语义）。
**无静默丢弃**。

### 0.7 标杆实现

- **源算子路径**：`/home/developer/TileOPs/tileops/kernels/attention/deepseek_dsa_decode.py`（函数 `_sparse_mla_kernel`，L17 起）。
- **Op 类**：`/home/developer/TileOPs/tileops/ops/attention/deepseek_dsa.py`（`is_hopper()` SM90 硬门禁 → 迁移已去掉）。
- **源仓参考实现 / 测试基准**：`examples/TileOPs/tests/ops/test_deepseek_dsa_decode.py::DsaDecodeTest.ref_program`
  （masked sparse softmax：`scatter` 索引 mask + `compressed_causal_mask` + `einsum` 两段）——§8.1 golden **直接移植**（独立于 §0.6 NPU 算法）。
- **抽取副本**：`examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/_deep_seek_sparse_attention_decode_with_k_v_cache_fwd_kernels.py`。
- **NPU 集成目标（接口契约）**：`examples/TileOPs/tileops/kernels/attention/deepseek_dsa_decode/deepseek_dsa_decode.py`
  （`SparseMlaKernel`，`ascend_mode="Developer"`，`default_config={"block_i":64}`）。
- **本仓同族先例（强烈参照）**：`examples/deepseek_v32/sparse_mla_fwd.py`（Developer MixCV + online softmax + `T.Pipelined` + `T.gemm`）；
  `examples/deepseek_v4/example_sparse_mla_fwd_kernel.py`（串行行 gather + `T.copy` + v-prefix）；`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py`
  （`size=[…]` 前缀列 PV gemm + 稀疏 gather 的 workspace 形态）。

## 1. 概述

### 1.1 算子名称

`_sparse_mla_kernel`（DeepSeek 稀疏 MLA **decode**：KV cache + 每行 top-k 稀疏索引 + 在线 softmax）。

### 1.2 功能描述

对每个 query token `(b, s)` 的每个头 `h`，按该行的 top-k 索引从 KV cache 中稀疏取 `topk` 个 KV 条目，
在有效条目上做（因果裁剪的）softmax 注意力，输出 `D` 维结果；KV 为 MLA 压缩表示（`H_kv=1`），
query/key 由 `D=512` 的 no-PE 主体 + `dim_tail=64` 的共享 RoPE 尾拼接而成。

### 1.3 数学公式

$$
e_j = \frac{q_{b,s,h} \cdot \mathrm{kv}_{b,\mathrm{idx}_j,g}}{\sqrt{D + \mathrm{dim\_tail}}},\qquad
\mathcal{V}_{b,s} = \{\, j : \mathrm{idx}_j \ge 0 \ \wedge\ \mathrm{idx}_j \le \mathrm{max\_kv}_i \,\},\qquad
\mathrm{max\_kv}_i = \left\lfloor \frac{q\_start\_index\_s + s + 1 - \mathrm{stride\_kv}}{\mathrm{stride\_kv}} \right\rfloor
$$

$$
o_{b,s,h,:} = \frac{\sum_{j \in \mathcal{V}_{b,s}} \exp\left(e_j - \max_{j' \in \mathcal{V}_{b,s}} e_{j'}\right)\, \mathrm{kv}_{b,\mathrm{idx}_j,g,\,:D}}
{\sum_{j \in \mathcal{V}_{b,s}} \exp\left(e_j - \max_{j' \in \mathcal{V}_{b,s}} e_{j'}\right)}
$$

源实现以 **2 底** 表达（`exp2`）并把 `log2(e)` 折入 scale：`e_j^{src} = (q·kv)·\mathrm{sm\_scale}`，
`\mathrm{sm\_scale} = (1/\sqrt{D+Dt})\cdot\log_2 e`，故 `exp2(e_j^{src}) = exp(e_j)`（§1.6.1 OPT-1）。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6；与源算法的差异逐条标注来源）：

1. **任务分解**（R6）：一维非持久 grid `num_logical = B · S · (H / block_heads)`；`cid` 反解 `(b, s, hb)`。
   **差异**：源用三维 grid（`(S·replicate_h, B, kv_group)`，本任务 `S·replicate_h = 1024×2 = 2048`），**每 block 64 头**（`replicate_h=2`；§0.3 / O17）；NPU 因 L0C=128KB（R4）必须按 head **再分块**（`block_heads=32`，每 s 4 个 head block）。
2. **在线（flash）softmax over topk**（§1.6.0 选定算法，源 O8 保留）：把 `topk` 维按 `block_N` 分块，
   `for k in T.Pipelined(topk // block_N, num_stages=2)`（R3；替代源的 Buffer0/1 + mbarrier 手动双缓冲）。
3. **每块计算**：
   - gather（R8）：`for bi in T.serial(block_N)`，`idx_c = clamp(indices[…], 0, S_kv−1)`，
     `T.copy(KV[b, idx_c, 0, :], KV_shared[bi, :])`（**行连续**搬运；R1 前例形态）。
   - 掩码：`pen[bi] = (−1e38 if indices>max_kv_i else 0)`，写进 `acc_s` 预填值（§1.6.1 OPT-3）。
   - QK（R1/R5）：`T.gemm(Q_shared, KV_shared, acc_s, initC=False, b_transpose=True)`（K = `D+Dt = 576` 一次）。
   - 在线 softmax（§1.6.1 OPT-1/OPT-2）：`m_cur = reduce_max(acc_s, dim=1)`；`m_new = vmax(m_prev, m_cur)`；
     `alpha = vexp(m_prev·s − m_new·s)`（s = `1/sqrt(D+Dt)`）；`P = vexp(acc_s·s − m_new·s)`；`l = l·alpha + reduce_sum(P)`。
   - PV（R1/R5）：`P_f16 = vcast(P)`；`T.gemm(P_shared, KV_shared, acc_o, initC=False, size=[block_heads, block_N, D])`。
   - rescale：`acc_o *= alpha`（广播 `[BH,1]→[BH,D]`）。
4. **收尾**：`o = acc_o / l`（`T.vdiv`）；`vcast` 回输入 dtype；`T.copy` 写 `output[b, s, h0:h1, :]`。
   **与源差异**：源额外产出 LSE（`log2(sumexp) + m·sm_scale`）供反向；**本算子只有前向，不做 LSE**（源 L307 的 LSE 计算在迁移中不搬运——
   非输出契约的一部分；记为「舍弃」，见 §0.5 语义层：源把 LSE 写进 `sumexp` 局部变量而非输出，故不属输出契约）。

### 1.5 数据流图

```
GM[q] --T.copy--> L1[Q_shared (BH,576) f16] -------------------------------------+
GM[indices] --标量读--> pen[bi] (掩码)                                            |  T.gemm
GM[kv] --T.copy(串行行 gather, clamp 索引)--> L1[KV_shared (block_N,576) f16] ---+--> L0C[acc_s (BH,block_N) f32]
                                                                                  |      |
                                                              vexp / vmax / reduce |      v
                                                                        UB[m,l,alpha]    L1[P_shared (BH,block_N) f16]
                                                                                          |
                                            L0C[acc_o (BH,512) f32] <--T.gemm(size=[BH,block_N,D])--+
                                                          |
                                            vdiv(l) + vcast --> L1[o_shared] --T.copy--> GM[o]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先做算法调研四问（§1.6.0，输入公式/源算法只是基线候选之一）→ 公式级等价优化（§1.6.1，含 D-1 机器验证）
> → 循环/标量向量化替代（§1.6.2）→ 向量化轴与布局决策（§1.6.3）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = 同一数学语义（稀疏 MLA decode：top-k 选择 + 因果裁剪 softmax 注意力）的**算法族**。
> 本算子为「矩阵 + 多步 + 融合」类 → **完整调研**。参考表命中行：
> `algorithm-candidates.md` 的 `ALG-attention`（flash 结构 R2 / MixCV R4）、`ALG-softmax`（减 max、exp2、online）、
> `ALG-gemv`（K 分块累加）、`ALG-reduction`（分块归约 + 合并）。

**R1 等价化简公式候选**（基线候选必在表内；正式论证在 §1.6.1）：

| # | 候选（算法族） | 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|---------------|------|-----------|---------|--------|
| 0 | **基线：源 online-softmax flash over topk**（576 维 concat QK + exp2 + PV，分块 running max/sum；每 head-block 单遍、`replicate_h=2` ⇒ 整体 2 遍） | 单遍分块 online | — | — | ✅（基线） |
| 1 | **两遍：materialize 全 `S=[H,topk]` 再 softmax** | 先全算 score 落 GM，再 max/sum，再 PV | 容差内等价 | **反向**：多一趟 GM 往返 + O(topk) 缓冲 | ✅（R3 淘汰） |
| 2 | **head 分块 + 在线**（本设计主选；`block_heads < H`） | 输出维分解 × online | **精确恒等**（head 相互独立） | 适配 L0C 128KB；代价 = KV 重读 ×`H/block_heads` | ✅（选定） |
| 3 | **workspace 物化 gather**（先把该行 topk 行 gather 成紧凑 buffer 再复用，`tilelang_sparse_mla_fwd.py` 形态） | gather→GM ws→复用 | 容差内等价 | **反向**：多写+读 ws；KV 重读仍未消除 | ✅（R3 淘汰） |
| 4 | **自然 exp 域**（`exp2(x·log2e)` → `exp(x)`） | 公式变形 | 数学恒等 | 去 `T.vexp2` 的额外 tmp buffer | ✅（§1.6.1 OPT-1） |
| 5 | **重缩放因子重结合 + `m·s` 提升**（CSE） | 公式变形 | 数学恒等（分配律） | 每块省 1 个 `[BH,block_N]` vmul/sub | ✅（§1.6.1 OPT-2） |
| 6 | **有限大数哨兵**（`-inf` → `-1e38`） | 公式变形（稳定化） | 容差内等价（`exp(-1e38)=0` 精确） | 防 `-inf` 参与 reduce 的 NaN；`PL-1.11` 算术惩罚掩码族 | ✅（§1.6.1 OPT-3） |
| 7 | **scale 折入 Q**（`Q̃=Q·s`；`(QKᵀ)·s = Q̃Kᵀ`） | 公式变形 | 数学恒等 | 去每块 `score·s` vmul | ✅（§1.6.1 **机器验证否决**：bf16 违反率 4.55%） |
| 8 | **归一化改乘倒数**（`x/l → x·(1/l)`） | 公式变形 | 数学恒等/1 ulp | **不采纳**：向量域 `[BH,D]÷[BH,1]` 已是 **1 个 `T.vdiv`**；`vrec+vmul` = 2 op 更贵（`CONST-vector-op-cost-910_9362`：每 op 固定代价） | ❌（§1.6.1 记录） |
| 9 | **索引去重**（对齐参考实现的 `scatter` 语义） | 语义改写 | **非等价**（源 kernel 对重复索引重复计数） | — | ❌（manifest 索引互异，§0.1） |

**R2 在线算法**：**有，且为基线所采用，并且是本问题的必需结构**（结构依据 + 源证据 + 容量判据）。
① 源 `_sparse_mla_kernel` 已用 online softmax（分块 running max/sum + 重缩放，L246-255）——**源算法已用 online 是「存在在线算法」的直接证据**
（algorithm-research.md §3 R2③）；② 结构判据：softmax 的分母 `Σ exp` 与分子 `Σ exp·V` 可分解为 running 统计量（max / sum / 加权和），
故存在在线变体；③ **容量判据（本问题必需）**：mainstream `S=[H,topk]=[128,2048]` fp32 = **1.0 MB**，远超 UB 192KB / L0C 128KB
⇒ 不可能 materialize 全 score（候选 1 在容量上直接不可行，即使不论访存）。收益口径：**单 head-block 内** KV 扫描 **1 遍**、
中间缓冲 O(`block_N`) 而非 O(topk)（head 分块后的**整体**遍历遍数 = `H/block_heads`，见 R3 表）。
候选 1（两遍）与候选 3（workspace 物化）在 R3 被淘汰。

**R3 复杂度对比**（四口径；口径说明：FLOPs 只计乘加=2/次，不含 softmax 超越函数；访存按 **GM/L2 实际字节**，
含 gather 的重复读——**注意**：manifest `roofline.bytes = (q+kv+o)·2 + index_bytes ≈ 297 MB` 假设 **KV 只读一遍（缓存大小）**，
是**理想下界**，稀疏 gather 实际按每行 topk 行读取（缓存行被读 `B·S` 次），不可达；下表用**实际 gather 字节**；
扫描遍数 = 同一 KV 数据的重复读取次数；中间缓冲峰值 = 片上最大常驻 + GM 工作区往返量。
**dispatch 代表 workload = mainstream `(B=1,S=1024,H=128,D=512,Dt=64,topk=2048,S_kv=2048)`**）：

| 算法候选 | FLOPs | 访存量 (Bytes) | KV 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|------------|-------------|---------------------|
| **0 基线（源，2×64 头 online；`replicate_h=2`）** | `2·B·S·H·topk·(2D+Dt)` = **5.84e11** | Q 151M + **gather 2.42G × `replicate_h`(=2) = 4.83G** + idx 8.4M + O 134M ≈ **5.12 GB** | **2** | `S[H,topk]` 若物化 1MB（不可行）→ 分块 `[h_per_block=64,block_N]f32` + 源寄存器态 `acc_o=2×[64,256]f32=128KB`（+`acc_s[64,64]f32`=16KB ≈ 144KB） | 源 GPU：`B·S·replicate_h=2048` 任务（每 s 2×64 头 block） |
| **1 两遍 materialize** | 5.84e11 | ≈ 2.71G + **+ S 写 1.07G + 读 1.07G**（[128,2048]f32×1024 行） | 1 + S 2 = **3** | O(topk) GM 缓冲 1.07 GB | 同 0；多一趟 |
| **2 head 分块 + online（`block_heads=32`，主选）** | 5.84e11 | Q 151M + **gather 2.42G × (H/BH=4) = 9.66G** + idx 8.4M + O 134M ≈ **9.95 GB** | **4** | `acc_s[32,block_N]f32` 8KB + `acc_o[32,512]f32` 64KB（L0C 72KB ≤ 128KB） | `B·S·(H/BH)=4096` 任务 |
| **2′ head 分块 + online（`block_heads=64`，备选）** | 5.84e11 | Q 151M + **gather 2.42G × 2 = 4.83G** + idx 8.4M + O 134M ≈ **5.12 GB** | **2** | `acc_o[64,512]f32` = 128KB（须驻 **UB**，L0C 放不下 + `acc_s`） | `B·S·(H/BH)=2048` 任务 |
| **3 workspace 物化 gather** | 5.84e11 | gather 2.42G + **ws 写 2.42G + head-block 读 ws 4×2.42G** ≈ **14.5 GB** | ≥5 | GM ws 2.42 GB | 同 2 但访存↑ |

> **选定子族**：候选 **2/2′**（head 分块 + online）。候选 1 容量不可行（1MB score/行）+ 扫描遍数↑（3）；
> 候选 3 访存量最大（gather 后仍重读）。候选 0（源）以 `replicate_h=2`（每 s 2×64 头 block）**本身已把 KV 读 2 遍**（4.83G，2 pass），
> 但其**每 block 寄存器态 = `2×[64,256]f32 = 128KB` + `acc_s[64,64]f32` 16KB ≈ 144KB**——在 NPU **L0C = 128KB** 下不可行
> （`CONST-capacity-910B2C`，且须与 `acc_s` 共享）。故 KV 重读放大（主选 ×4）是 **NPU 容量约束 + 更小块宽 的直接后果**，
> 而非算法选择失当；`block_heads` 就是「KV 重读倍数」与「L0C/UB 容量」的权衡旋钮（§1.6.3 主选/备选 + 实验裁决）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附依据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/融合 | 结论 |
|---------|-------------|---------|-----------|---------|-----------|------|
| 0 基线（源，2×64 头 online） | ✅ Cube MAC + Vector v-op | ❌ 每 block `acc_o = 2×[64,256]f32` = 128KB（+`acc_s[64,64]` 16KB） > L0C 128KB（须与 `acc_s` 共享） | ✅ | ✅ | ✅ | ❌ **容量淘汰**（`CONST-capacity-910B2C` L0C=128KB / UB=192KB） |
| 1 两遍 materialize | ✅ | ❌ O(topk)=1MB/行 score 缓冲（GM） | ✅ | ✅ | ⚠️ 多一趟 | ❌ 淘汰（容量 + 访存↑ + 遍数↑） |
| **2 head 分块 BH=32（主选）** | ✅ Cube（QK 576 维 / PV 512 维）+ Vector（v-exp/reduce/vmul/vdiv） | ✅ L0C 72KB ≤128KB；L1 Q 37KB + KV 147KB（双 stage）+ P 4KB ≤512KB；UB 小 | ✅ `block_N=64`、`D=512`、`D+Dt=576` 均 32B 对齐且为 64/128 倍数；分形 M/N≥16、K≥32 ✓ | ✅ `topk % block_N == 0`（256/512/1024/2048 全整除）；KV 循环界静态 | ✅ `T.Pipelined`；Developer 自动 CV 切分 | **✅ 选定（主选）** |
| 2′ head 分块 BH=64（备选） | ✅ 同 2 | ⚠️ `acc_o[64,512]f32`=128KB 须驻 UB（+`acc_s`16KB + P 8KB ≈ 153KB，×1.10 膨胀 ≈ 168KB < 192KB，**贴限**） | ✅ | ✅ | ✅ | ✅ **备选**（KV 重读减半；§1.6.3 实验裁决） |
| 3 workspace 物化 | ✅ | ⚠️ GM ws 2.42GB | ✅ | ✅ | ⚠️ | ❌ 淘汰（访存最大） |
| 7 scale 折入 Q | — | — | — | — | — | ❌ 淘汰（**bf16 违反率 4.55%**，§1.6.1 D-1） |

**R4 定量 roofline（D-2）**：dispatch 代表 workload = mainstream `(B=1,S=1024,H=128,D=512,Dt=64,topk=2048,S_kv=2048)`（主选 BH=32）。
- **流量项** = 实际 GM/L2 字节 ÷ 有效带宽。主选流量 ≈ **9.95 GB**（Q 151M + gather 9.66G + idx 8.4M + O 134M）。
  **带宽口径（关键不确定项）**：gather 的目标值 ≈ 缓存全量 2.42GB 中的重复读，KV cache 仅 2.4MB/行集，
  大概率 **L2 驻留**，故有效带宽应取「L2/片上读带宽」类而非 HBM。本机 **Ascend910B3** 无实测该常数：
  - 按下界 **核→GM ≈1.06 TB/s**（`CONST-ascend910-9362-profile`，device=**910_9362**，2026-10-13；910B3 同 20 AIC/40 AIV 但**未实测**，标注「跨设备沿用/未实测」）→ **9.4 ms**；
  - 按 **910B3 逐元素聚合 R+W ≈2.49 TB/s**（`CONST-elu-910B3-bw-floor`，device=910B3，2026-10-15）→ **4.0 ms**。
  ⇒ 流量项 ∈ **[4.0, 9.4] ms**。
- **发射项** = 向量 op 数 × 单 op 成本。每任务每块 ≈ 12 个向量 op（掩码广播 2 + online 6 + cast 1 + rescale 1 + epilogue/32），
  块数 = `topk/block_N = 32`，任务/core = `4096/20 ≈ 205`。两种口径：
  - 元素吞吐口径（`CONST-vector-op-cost-910_9362`：普通 v-op ≈9.15 µs/16M-elem ⇒ ≈1.75e6 elem/µs）：
    每 core 元素数 = 205×32×12×(32×64) ≈ **1.6e8** ⇒ ≈ **92 µs**；
  - **`0.5 µs/op` 口径（`CONST-vector-launch-overhead`，910B2C 实测）：205×32×12 ≈ 7.9e4 op/core ⇒ ≈ 39 ms**——
    但该条目 **2026-10-15 已实证**在 9362/910B3 类设备上**高估 5.4–9.6×**（且 op 间重叠），故实际发射项远小于该值。
  ⇒ 发射项 ∈ **[0.05, 2] ms**（口径不确定，标「未实测」）。
- **容量项** = 容量决定的 tile 循环数 × 跨引擎往返代价。本设计**无 `store_fixpipe` GM 往返**（QK/PV 累加在 L0C，
  Developer 自动 CV 切分，中间不落 GM）；L0C（72KB）与 L1/UB 在限内（`CONST-capacity-910B2C`）⇒ **≈ 0**。
- **估算下界**：`估算下界: ≈ 4.0 – 9.4 ms（流量 4.0–9.4ms + 发射 0.05–2ms + 容量 ~0；引用 CONST-ascend910-9362-profile / CONST-elu-910B3-bw-floor / CONST-vector-op-cost-910_9362 / CONST-vector-launch-overhead / CONST-capacity-910B2C）`。
  **不确定性**：① 910B3 的核→GM/L2 带宽无实测（本表两条常数一条属 9362、一条属 910B3 逐元素聚合，口径不同）；
  ② `CONST-vector-launch-overhead` 的 0.5µs/op 已实证跨设备高估 ⇒ 发射项可能被 39 ms 口径严重高估。
  ⇒ **流量项主导**，估算下界区间 [4.0, 9.4] ms；**KV 重读倍数（`H/block_heads`）是最大杠杆**（BH=64 备选可把流量项降到 ≈2.1–4.8 ms）。
  **注**：候选 0（源）因 `replicate_h=2` 亦为 2 遍（5.12 GB，≈2.1–4.8 ms 同档）——**主选估计（[4.0, 9.4] ms）不受本次读数更正影响**
  （FLOPs/主选流量项均不依赖源 head 块宽）。

**调研结论**：选定 **算法族 = head 分块 + online（flash）softmax over topk 的单遍分块结构**（候选 2，备选 2′）。
以 **Developer MixCV 单 kernel** 落地：Cube `T.gemm`（QK 576 维 + PV 512 维）+ Vector v-prefix 在线 softmax 链 + `T.Pipelined(num_stages=2)`。
- **关键依据**：R3 中候选 1 容量/扫描遍数更差（1MB/行不可驻留、3 遍）、候选 3 访存最差（14.5 GB）；候选 0（源，2×64 头 block，KV 读 2 遍）在 **L0C=128KB 上不可行**
  （每 block `acc_o 2×[64,256]f32 = 128KB + acc_s 16KB`）⇒
  必须 head 分块，其 KV 重读倍数由 `block_heads` 决定（§1.6.3）；R4 中主选/备选全部通过对齐、静态界、流水检查。
- **与基线（源算法）的结构差异**：算法族相同（online flash）；差异全在**实现层**（一维 grid / head 分块 / 单次 576 维 gemm /
  `T.Pipelined` 代替 mbarrier / clamp+掩码代替跳过装载 / 自然 exp 域）+ 数学等价优化（§1.6.1）。
- **「无更优替代」的调研范围**：查过 `algorithm-candidates.md` 的 `ALG-attention`/`ALG-softmax`/`ALG-gemv`/`ALG-reduction` 命中行；
  `attention.md` 的 `PL-1.7-expert-persistent-boundary`/`PL-1.9-twophase`/`PL-1.11-causal-mask-scalartrap`/`PL-1.16-expert-dualscope-bypass`；
  `traps-compiler.md` 的 `TRAP-C10-strided-gather`/`TRAP-threads-kwarg-noop`；`traps-runtime.md` 的 `TRAP-DEVMODE-PERSIST-GEMM`；
  本仓 5 个稀疏 MLA/indexer 先例（`examples/deepseek_v32/sparse_mla_fwd{,_dynamic_shape,_exp}.py`、
  `examples/deepseek_v4/example_sparse_mla_fwd_kernel.py`、`examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py`、
  `examples/indexer/indexer_fwd.py`）；并做 R3/R4 结构分析——**无遍数或访存更低的候选**
  （KV 必须各 query 行 gathered 一次；score 无法驻留片上故 online 必需；head 重读是 L0C 容量的必然代价）。
- **源算法优化意图承接**：§0.4 O1–O17 全部由选定算法 + §0.6 承接，无静默丢弃（见 §0.6 末段）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定算法。**等价性论证 = 数学恒等 + D-1 机器验证**（`verify_equiv.py`，结果表见下）。
> baseline 式 = 源 online 链（`exp2(score·s2 − m·s2)`，raw 域，`-inf` 掩码）；候选式 = 采纳优化后的 NPU 公式。
> 本节**无采纳项被否决**；被否决的候选（scale 折入 Q、归一化乘倒数）单列于表后，机器验证脚本同源产出。

| # | 优化项 | 原式（源） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|-----------|-----------|-----------|---------|
| OPT-1 | **自然 exp 域** | `exp2(e·s2)`，`s2 = log2e/√(D+Dt)` | `exp(e·s)`，`s = 1/√(D+Dt)` | 数学恒等（换底）：`exp2(x·log2e)=exp(x)`。**D-1 fp16 max_abs 4.883e-04 / bf16 1.953e-03，违反率均 0** | 去 `log2e` 因子；改用 `T.vexp`（`docs/Tilelang.language/数学操作/T.vexp.md`）——`T.vexp2` 需额外 tmp buffer；**两 API 的文档 dtype 矩阵均不含 bf16**（`T.vexp.md`/`T.vexp2.md` §2.2.1 均标 bf16 ✗），但**本设计 softmax 全程在 fp32 域做 exp**（`acc_s` 累加、`tmp` 均为 fp32），故不触发该缺口（「文档矩阵 vs 设备实测可能分歧」，见 §9 R-13 / `TRAP-bf16-vop-doc-matrix-divergence`） |
| OPT-2 | **重缩放因子重结合 + `m·s` 提升（CSE）** | `alpha=exp2((m_prev−m_cur)·s2)`；`P=exp2(score·s2 − m_cur·s2)`（`m_cur·s2` 随元素重复计算） | `ms = m_new·s`（每行算 **1** 次）；`alpha=exp(m_prev·s − ms)`；`P=exp(acc_s·s − ms)` | 数学恒等（分配律 + 换底，与 OPT-1 合成）。**D-1 fp16 max_abs 9.766e-04（max_ulp 3.04）/ bf16 9.766e-04（max_ulp 27.6），违反率均 0**。**非空臂**：fp32 因子级对照 `exp2(m_p·s2−m_c·s2)` vs 重结合 `exp(m_p·s−m_c·s)` 为真实不同表达式 | 每 KV 块省 **1 个 `[BH,block_N]` vmul**（`m_cur·s` 由逐元素降为逐行）；`topk/block_N=32` 块/任务 ⇒ 每任务省 32×`[32,64]` = 65536 次乘 |
| OPT-3 | **有限大数哨兵** | `-T.infinity`（掩码列）+ `m_i` init `-2**30` | 掩码列 → `-1e38`（fp32）；`-1e38 + finite = -1e38`，`exp(-1e38)=0` **精确** | 容差内等价且数值更稳：`exp(-1e38 − m) = 0` 与 `exp(-inf − m)` **逐位一致**（D-1 `[OPT-3 sentinel] PASS`）；规避 `-inf` 参与 `reduce_max`/rescale 的 `-inf−(−inf)=NaN` 路径（`PL-1.11` 算术惩罚掩码族） | 计算量不变；**消除 NaN 风险**（源 `-2**30` init 是为同一目的的经验手段，本项给出更严格的有限哨兵） |

**D-1 等价性机器验证结果**（`examples/deepseek_dsa_decode/_sparse_mla_kernel/verify_equiv.py`，torch CPU；
fp16/bf16 mixed-precision 候选式 vs 基线式；`D,Dt ∈ {(512,64),(512,0)}` × `BH ∈ {16,32,64}` × `(topk,block_N) ∈ {(256,64),(512,64)}` ×
`n_invalid ∈ {0,7,topk}` × 2 seed，另加 4 个角点（大幅值 ×4 / 全零 q）；对照基准 = 同 dtype 舍入路径的基线式；门限 atol=rtol=1e-2）：

| dtype | 采纳项 | 最大绝对差 (max_abs) | 最大 ulp 差 | 违反率 | 结论 |
|-------|--------|---------------------|-------------|--------|------|
| fp16 | OPT-1（natural-exp） | 4.883e-04 | 531.00 | 0.000e+00 | **EQUIV_PASS** |
| fp16 | OPT-2（reassociation + CSE hoist） | 9.766e-04 | 3.04 | 0.000e+00 | **EQUIV_PASS** |
| fp16 | OPT-3（finite sentinel） | 0.000e+00 | 0.00 | 0.000e+00 | **EQUIV_PASS** |
| fp16 | **OPT-1+2+3（采纳集合）** | 9.766e-04 | 530.00 | 0.000e+00 | **EQUIV_PASS** |
| bf16 | OPT-1（natural-exp） | 1.953e-03 | 715.34 | 0.000e+00 | **EQUIV_PASS** |
| bf16 | OPT-2（reassociation + CSE hoist） | 9.766e-04 | 27.64 | 0.000e+00 | **EQUIV_PASS** |
| bf16 | OPT-3（finite sentinel） | 0.000e+00 | 0.00 | 0.000e+00 | **EQUIV_PASS** |
| bf16 | **OPT-1+2+3（采纳集合）** | 1.953e-03 | 715.34 | 0.000e+00 | **EQUIV_PASS** |

> **无未通过（FAIL）项**。`max_ulp` 偏大源于近零元素的 ULP 归一化（分母钳到 fp16 最小正规数 2⁻¹⁴），绝对差 ≪ atol=1e-2；
> 所有配置**违反率 0**。基线自身 vs fp64 truth：fp16 max_abs 9.813e-04 / bf16 7.807e-03（基线在容差内的 sanity 检查）。
> 附加证据：`[OPT-2 Q-rounding]` 隔离 OPT-2 的 CSE/重结合臂（非空）；`[OPT-R rejected]` 记录被弃项的数值。

**被否决的候选（记录 + 机器验证，不采纳）**：

| 候选 | 原式 → 候选式 | D-1 结果 | 否决理由（含证据） |
|------|-------------|----------|-------------------|
| **OPT-X：scale 折入 Q** | `(QKᵀ)·s` → `(Q·s)Kᵀ` | fp16 max_abs 8.179e-03（PASS）；**bf16 max_abs 9.766e-02、违反率 4.550e-02 → 未通过** | **否决**：`s=1/√576≈0.042` 在 bf16（8 位尾数）上对 `Q·s` 的舍入误差经 576 维点积放大，超出容差（`verify_equiv.py` 独立臂）；收益（省每块 1 vmul）不足以承担 bf16 精度损失。**bf16 是 manifest 契约 dtype**（§0.2） |
| **OPT-R：归一化乘倒数** | `acc_o / l` → `acc_o·(1/l)` | 数值 max_abs 1.192e-07（1 ulp，本可等价） | **不采纳**：`acc_o/l` 在向量域 `[BH,D]÷[BH,1]` 已是**单个 `T.vdiv` op**；改 `vrec+v... mul` = 2 op，按 `CONST-vector-op-cost-910_9362`（每 op 固定代价）**更贵**。与 `mha_decode_paged` 的同类否决一致 |
| **OPT-9：索引去重** | 源逐条累加 → 参考实现去重 | — | **非等价**：源 kernel 对重复索引**重复计数**；manifest 索引互异故实际等价，但**不把它当作优化**（语义约束优先，algorithm-research.md §4） |

**优化结论**：采纳 **3 项**（OPT-1/2/3，全部 EQUIV_PASS，fp16 与 bf16 均通过）。**优化后公式**（供 §3.1 使用）：

```
s  = 1 / sqrt(D + Dt)                     # 自然 exp 域（OPT-1）
l  = 0 ;  m_prev = -inf ;  acc_o = 0
# 每 KV 块（block_N 列），k = 0 .. topk/block_N - 1:
  sc = Q_shared @ KV_sharedᵀ                                  # Cube, fp32 累加 (K = D + Dt)
  sc = where(idx_j <= max_kv_i, sc, -1e38)                    # 列掩码（OPT-3 有限哨兵）
  m_cur = rowmax(sc)                                          # dim=1
  m_new = vmax(m_prev, m_cur)
  ms    = m_new * s                                           # 每行 1 次（OPT-2 CSE）
  alpha = exp(m_prev * s - ms)                                # OPT-2 重结合
  P     = exp(sc * s - ms)                                    # 广播 ms（OPT-2）；OPT-1 自然 exp
  l     = l * alpha + rowsum(P)
  P16   = cast_f16(P)                                         # P 量化点（保持源的 cast）
  acc_o = acc_o * alpha + (P16 @ KV_shared[:, :D])            # Cube PV（size=[BH,block_N,D]）
  m_prev = m_new
o = acc_o / l                                                 # 行广播 vdiv
o = cast_f16(o)                                               # → Output
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案**全部**循环与标量计算点；可替代的用 v-prefix / `T.reduce` 替代，不可替代的逐项给充分理由。
> API 存在性以 `docs/Tilelang.language/` 与 `examples/` 佐证。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 掩码列填充 `sc = valid ? sc : -1e38` | 逐元素 `T.if_then_else` | `T.vbrc` 广播 `pen[bi]` + `T.vadd`；或 `T.Parallel(BH,block_N)`（源先例，向量化） | ✅ | — |
| 2 | 行 max（running） | 标量累加循环 | `T.reduce_max(acc_s, m_cur, dim=1)`（`docs/.../规约操作/T.reduce_max.md`；`sparse_mla_fwd.py` 先例） | ✅ | — |
| 3 | `m_new = max(m_prev,m_cur)` | 标量逐元素 | `T.vmax(m_prev, m_cur, m_new)`（`sparse_mla_fwd.py` L129） | ✅ | — |
| 4 | `alpha = exp(m_prev·s − ms)` | 标量 | `T.vmul`/`T.vsub`/`T.vexp`（`[BH,1]`） | ✅ | — |
| 5 | `P = exp(acc_s·s − ms)` | 标量双重循环 | `T.vmul`/`T.vsub`/`T.vexp`（`[BH,block_N]`，`ms` 行广播；`example_sparse_mla_fwd_kernel.py` L125-131 先例） | ✅ | — |
| 6 | 行 sum（running） | 标量累加 | `T.reduce_sum(P, local_sum, dim=1)`（**`clear=True`**——`docs/.../规约操作/T.reduce.md §2.3` 记无 clear 的 sum 是静默错误形态） | ✅ | — |
| 7 | `l = l·alpha + sum` | 标量 | `T.vmul` + `T.vadd`（`[BH,1]`） | ✅ | — |
| 8 | `acc_o *= alpha` | 逐元素 | `T.vbrc`(alpha `[BH,1]`→`[BH,D]`) + `T.vmul`（`T.vbrc.md` 示例 2 广播） | ✅ | — |
| 9 | `P` 量化 | 逐元素 | `T.vcast(P, P16, round_mode="rint")`（`docs/.../数据类型转换操作/T.vcast.md`） | ✅ | — |
| 10 | `o = acc_o / l` | 逐元素 | `T.vbrc`(l) + `T.vdiv`（`[BH,D] ÷ [BH,1]`，单 op） | ✅ | — |
| 11 | 卷积/窗口扫描 | 不涉及 | — | n/a | — |
| 12 | **KV 行 gather**（`KV_shared[bi,:] = KV[b,idx_c,:]`） | 逐行串行 | **无向量等价 API** | ❌ | **数据依赖的行地址 + 连续行 DMA**：每行的源地址由 `indices`（运行时数据）决定，属「跨步/索引 gather」类；NPU 缺跨步 gather 向量指令（`TRAP-C10-strided-gather`，截至 2026-08-28 build 有效）。**可用形态 = 串行 `T.copy` 连续行拷贝**（`example_sparse_mla_fwd_kernel.py` L115-118 先例），单行 576×f16 = 1152 B（= 9×128 B 段），带宽友好。**非「实现简单」类理由**，而是 API 缺失 |
| 13 | `idx_c = clamp(idx,0,S_kv−1)` 与 `max_kv_i` 计算 | 标量/行 | 无 | ❌ | **block 级标量（每行 1 次）**：`block_N=64` 行/块，非逐元素热点；`TRAP-scalar-min-chain-offset` 要求**单次 `T.min` clamp**（多轮累加链作偏移会 mis-compile）⇒ 正是本形态 |
| 14 | `cid → (b,s,hb)` 反解 | 除法/取模标量 | 无 | ❌ | **block 级索引 / 任务映射**：每 core 1 次，非逐元素热点，无逐元素等价 API |
| 15 | KV 主循环 `for k in T.Pipelined(topk//block_N)` | tile 级循环 | 保留（`T.Pipelined`） | ❌（合理保留） | **tile 级顺序依赖**：online 递推（`m_prev/l/acc_o` 跨块携带）不可归约化；`T.Pipelined` 是调度原语非逐元素循环 |
| 16 | head 循环 | 不涉及（由 grid 承担） | — | n/a | — |

**向量化结论**：逐元素与行级计算（点 1–10）**已全部向量化**（替代 API 均有 `docs/` 或 `examples/` 佐证）；
保留的标量/循环点理由类别：**API 缺失**（点 12：KV gather，`TRAP-C10`）、**block 级索引/标量**（点 13/14）、
**tile 级顺序依赖**（点 15：online 递推）。**无「实现简单 / 照源码写」类保留**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（BSHD 进 / BSHD 出，**不重排 GM 布局**）。核内布局与向量化轴是**设计变量**。

**轴质量评分**（候选轴：`D+Dt=576`（QK 内积维）、`block_N=64`（KV 候选维 / softmax 归约维）、`D=512`（输出维）、`BH=32`（head 维））：

| 评分项 | `block_N=64`（softmax 向量/归约轴） | `D=512`（PV 输出轴） | `D+Dt=576`（QK 内积轴） | `BH=32`（head 轴） |
|--------|-----------------------------------|---------------------|------------------------|-------------------|
| 整除性 | 64 = 8×8 ✓（fp16 ×8） | 512 ✓ | 576 = 72×8 ✓ | 32 ✓ |
| 尾 lane 浪费率 | 0% | 0% | 0% | 0% |
| 累加链形态 | 归约轴（`reduce dim=1`）；**水平归约** | 向量维（∈ L0C/gemm N 维） | Cube K 维（内积，非向量轴） | 独立行（各行独立，天然并行） |
| repack 代价 | 无需 | 无需 | 无需 | 无需（head 连续） |
| UB/容量影响 | `acc_s[BH,64]f32`=8KB | `acc_o[BH,512]f32`=64KB（L0C） | 不驻留（被 Cube 消费） | 影响 `acc_o` 首维 |

**softmax 归约 lane 映射候选**（`ALG-softmax` / 规约类要求：与「选哪条轴」正交，须分别枚举「水平归约」与「垂直扫描」两种 lane 映射）：

| # | lane 映射 | 形态 | 整除性 / 尾 lane 浪费 | 累加链形态 | repack 代价 | 与实现一致性 | 采纳 |
|---|-----------|------|----------------------|------------|-------------|--------------|------|
| L1 | **水平归约（intra-row reduce）** | `T.reduce_max/sum(acc_s, dst, dim=1)`：每个 head 行（`BH` 行）在 `block_N=64` 个列 lane 上做行内归约 | `block_N=64 = 8×8`（fp16 向量宽 ×8），**0% 尾 lane 浪费** | 每行一条 64 元素归约链，`BH=32` 行并行；库实现为树形/多级归约 | **无**（`acc_s[BH,block_N]` 已是行主布局） | ✅ §3.3/§6 用 `T.reduce_*(dim=1)` | **✅ 主选** |
| L2 | **垂直扫描（column-scan reduce）** | 把归约改写为沿 `block_N` 的串行扫描（`for bi: m = vmax(m, acc_s[:, bi])`，逐列累加） | 无尾 lane（但每步向量长度仅 = `BH=32`） | `block_N=64` 步**串行依赖链**，每步向量长 32 ⇒ 发射 64 次；归约深度 = 64（串行） vs L1 的树形 log | **需列主序 / 逐列切片**（`acc_s` 行主布局下取列 = 跨步访问，UB 重排或 `T.transpose`） | ❌ 与 `acc_s` 行主布局不符 | ❌ **弃选** |

> **弃选量化**：L2 把「1 次库归约（树形，深度 ≈log₂64=6）」换成「64 步串行扫描 × 每步向量长 32（=BH）」，发射次数 ×64、
> 且引入串行依赖链 + 列主序 repack（`acc_s` 为行主）——**净亏**；L1 整除（64=8×8）、0% 尾 lane 浪费、无 repack，且与
> §3.3/§6 的 `T.reduce_max/sum(dim=1)` 直接一致。**故 lane 映射选定 L1（水平归约）**，与 §1.6.2 点 2/6 的向量化替代互相印证。

**候选矩阵**（枚举 + 弃选量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **I/O 原生 BSHD；KV tile `[block_N, D+Dt]`（行连续）** | `block_N`（softmax 归约/逐元素）；`D`（PV 输出） | 无 | 基线；QK 用 `b_transpose=True`；PV 用 `size=[BH,block_N,D]` 取行前缀（有先例） | **✅ 主选** |
| 2 | KV 的 K/V 拆两个缓冲（`KV_k[BN,576]` + `KV_v[BN,512]`） | `D` | 无 | gather 字节 576→1088 (+89%)，**净亏**（gather 是流量主导项） | ❌（流量 +89%） |
| 3 | KV tile 转置为 `[D+Dt, block_N]`（K 为 M 维） | `D+Dt` | 核内 `T.transpose`（UB 级 ~µs，`docs/.../创建操作/T.transpose.md`；`ALG-transpose`） | 无必要：`b_transpose=True` 已让 Cube 直接消费 `[block_N,D+Dt]`；额外转置只增向量管线开销 | ❌（无收益） |
| 4 | host permute Q/KV 进出 | — | 两次 host permute | repack ~百 µs 级（`PL-1.3-host-permute`，`ALG-conv` 注记）> 算子本体；且违反 host 只读约束 | ❌（净亏 + 越界） |
| 5 | **head 维进一步分块到 `BH=64`**（备选结构，非布局） | 同 #1 | 无 | `acc_o` 升至 128KB ⇒ 必须驻 **UB**（+`acc_s` ≈ 152KB；`×1.10–1.12` 膨胀贴限，`~1.7x` 档不可行——见下方膨胀分支/阈值 1.26x）；KV 重读倍数 4→2，流量 9.95G→5.12G | ✅ **备选**（实验裁决，见下） |

**布局决策结论**：主选 **#1 = I/O 原生 BSHD；KV tile `[block_N, D+Dt]` 行连续；向量化轴 = `block_N`（softmax）与 `D`（输出）；
无 repack**。该结论同步落入 §3.3 伪代码与 §6（KV 循环内层向量维 = `block_N`/`D`）。弃选 #2（gather 字节 +89%）、
#3（无收益）、#4（净亏 + 违反 host 约束）已量化。

**实验裁决模式（强制三件套）**：§1.6.0 roofline 显示**流量项主导且 `block_heads` 是最大杠杆**，
但主选（BH=32，KV 重读 ×4）与备选（BH=64，×2）的取舍依赖两个**未在本机实证的常数**：
① **910B3 的核→GM/L2 有效读带宽**（`CONST-ascend910-9362-profile` 属 9362；`CONST-elu-910B3-bw-floor` 是逐元素聚合口径）；
② **KV cache 的 L2 驻留程度**（决定 gather 的有效带宽档位）。故强制三件套：
- **主选方案（进 Stage 3）**：`block_heads = 32`，`acc_o`/`acc_s` 用 `T.alloc_fragment`（L0C，72KB）；
  §3/§4/§6 只承载主选。
- **备选方案（≤1 个）**：`block_heads = 64`，`acc_o` 改 `T.alloc_shared`（**UB** 128KB，`sparse_mla_fwd.py` 先例形态）
  + `acc_s` `[64,64]` UB 16KB + `P16` 8KB ≈ 152KB。**膨胀系数分支（见 §4.5）**：按 `×1.10–1.12`（auto-multi-buffer=false 宽 config，`CG-2026-0008`）
  ⇒ ≈ 168–171KB < 192KB 可实施；**若实测膨胀系数落在 `~1.7x` 档**（`CONST-capacity-910B2C` 主档）⇒ 152 × 1.7 ≈ **258KB > 192KB ⇒ 备选不可行**。
  ⇒ **判定阈值：实测膨胀系数须 ≤ `192/152 ≈ 1.26x` 备选方可实施**；为不依赖该阈值，备选亦须采用「行广播不物化」形态（§4.5 削减方案，把 `acc_o` 之外的小缓冲压到最小）。
  grid 减半；KV 重读 ×2（与源的 `replicate_h=2` 同块宽、同倍数）。
  达 Stage 4 可直接实现深度（buffer 形状 / 循环结构 / 分核三要素按 BH=64 重算：逻辑核 = 2048）。
- **裁决计划**：代表 shape = mainstream `(1,1024,128,576)/(1,2048,1,576)/topk=2048` 与 `longer-kv-lower-topk`
  `(1,512,128,576)/(1,4096,1,576)/topk=1024`；指标 = msprof op Task Duration（median≥20）+ 反解有效 gather 带宽
  （`gather_bytes / duration`，对照两条带宽常数）；**判定阈值** = 备选 latency < 主选 × 0.90 才翻转（全局或按 shape 分片）；
  结果回写 §1.6.3 并触发设计修订回写路径。**shape 特化**：工厂带 `lru_cache`（per-shape），可按 `H`/`topk` 分派 BH∈{32,64}。

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Developer**（单 kernel **MixCV** 形态：Cube 侧 `T.gemm`（QK + PV）+ Vector 侧 v-prefix 链
（`T.vexp`/`T.vmax`/`T.vsub`/`T.vmul`/`T.vadd`/`T.vcast`/`T.vdiv`/`T.vbrc`）+ `T.reduce_max/sum`；无显式 `Scope`、无手动同步）。

### 2.2 选型理由

- 任务指定编程模式 = `developer`；脚手架 `deepseek_dsa_decode.py` 的 `SparseMlaKernel.ascend_mode = "Developer"`
  且按无 `threads` 的 Developer 形态调用（K9）。
- 同一 kernel 同时含 **Cube 侧 `T.gemm`**（2 处）与 **Vector 侧 v-prefix**（≥1 处），满足 AGENTS.md「Developer-mode MixCV trigger rule」
  / `tilelang-mixcv-skill` 的 Developer MixCV 判定 → MixCV 形态，参照 `examples/deepseek_v32/sparse_mla_fwd.py`（同族 sparse MLA，Developer MixCV）。
- **不用 persistent**：`TRAP-DEVMODE-PERSIST-GEMM`（tilelang 0.1.2+1990aa9fe4 / 910B2C，2026-09-17，`status: verified`）
  实证 Developer + persistent 分核 + gemm 混排**运行时崩溃**（"unaligned UUB addresses"）；本设计采**非持久**一维 grid（§5.5）。
- 不使用 Expert 的理由：本设计的结构（单程序 MixCV + `T.Pipelined`）在 Developer 支持域内（先例充分），
  无需 `Scope("Cube")/Scope("Vector")` 的手动跨引擎握手；Expert 的额外复杂度不带来对等收益（§9 风险 R-8 记录切换条件）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（L1：Q/KV/P 操作数 + UB 小缓冲）、`T.alloc_fragment`（L0C：`acc_s`/`acc_o`） |
| 计算方式 | Cube：`T.gemm`；Vector：v-prefix + `T.reduce_*` |
| 同步 | 自动（`T.Pipelined` 提供多缓冲与跨引擎同步） |
| 核划分 | 一维 `T.Kernel(num_logical, is_npu=True)`，非持久 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（自然 exp 域 + CSE 重结合 + 有限哨兵）。每个步骤优先映射 v-prefix 向量 API。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `Q_shared ← q[b,s,h0:h0+BH,0:Dqk]` | 加载该 head-block 的 Q |
| 2 | `pen[bi] = (idx_bi ≤ max_kv_i) ? 0 : −1e38` | 列掩码（临时有效判据） |
| 3 | `KV_shared[bi,:] ← kv[b, clamp(idx_bi) ,0,0:Dqk]` | 稀疏 gather（行连续搬运） |
| 4 | `acc_s = pen ⊕ Q_shared · KV_sharedᵀ` | QK gemm（K = Dqk = 576），预填掩码 |
| 5 | `m_cur = rowmax(acc_s)`；`m_new = max(m_prev, m_cur)` | running max |
| 6 | `ms = m_new · s`；`alpha = exp(m_prev·s − ms)` | 重缩放因子（OPT-2） |
| 7 | `P = exp(acc_s·s − ms)`；`P16 = cast_f16(P)` | softmax 分子 + P 量化 |
| 8 | `l = l·alpha + rowsum(P)` | softmax 分母（online） |
| 9 | `acc_o = acc_o·alpha + P16 · KV_shared[:, :D]` | rescale + PV gemm（N = D = 512） |
| 10 | `o = acc_o / l`；写出 | 归一化 + cast |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Q 加载 | `T.copy(Q[b, s, h0:h0+BH, 0:Dqk], Q_shared)` | 切片形（`docs/.../内存操作/T.copy.md`） | Developer |
| 2 | 掩码填充 | `T.Parallel(BH, block_N)` + `T.if_then_else`（源先例 `sparse_mla_fwd.py` L119-124） | 逐元素（向量化） | Developer |
| 3 | 稀疏 gather | `for bi in T.serial(block_N): T.copy(KV[b, idx_c, 0, 0:Dqk], KV_shared[bi, 0:Dqk])` | 索引+切片；先例 `example_sparse_mla_fwd_kernel.py` L114-118 | Developer |
| 3b | 索引 clamp | `T.min` / `T.max`（单次 clamp，`TRAP-scalar-min-chain-offset` 推荐形态） | 标量 | Developer |
| 4 | QK | `T.gemm(Q_shared, KV_shared, acc_s, initC=False, b_transpose=True)` | `docs/.../线性代数操作/T.gemm.md`；`initC=False` ⇒ `dst = src1@src2 + dst` | Developer |
| 5 | 行 max | `T.reduce_max(acc_s, m_cur, dim=1, clear=True)` | `docs/.../规约操作/T.reduce_max.md` | Developer |
| 5b | 逐行 max | `T.vmax(m_prev, m_cur, m_new)` | `sparse_mla_fwd.py` L129 | Developer |
| 6 | 缩放/平移 | `T.vmul`、`T.vsub`、`T.vexp`（`[BH,1]`） | `docs/.../数学操作/{T.vmul,T.vsub,T.vexp}.md` | Developer |
| 7 | 逐元素 exp | `T.vmul(acc_s, s_buf, tmp)` + `T.vsub(tmp, ms_buf, tmp)` + `T.vexp(tmp, P)` | `[BH,block_N]`；`ms` 行广播 | Developer |
| 7b | P 量化 | `T.vcast(P, P16, round_mode="rint")` | `docs/.../数据类型转换操作/T.vcast.md` | Developer |
| 8 | 行 sum | `T.reduce_sum(P, local_sum, dim=1, clear=True)` | **`clear=True` 必填**（`docs/.../规约操作/T.reduce.md §2.3`） | Developer |
| 8b | online 累加 | `T.vmul` + `T.vadd`（`[BH,1]`） | — | Developer |
| 9a | rescale | `T.vbrc(alpha, alpha_mat)` + `T.vmul(acc_o, alpha_mat, acc_o)` | `T.vbrc.md` 示例 2（行广播） | Developer |
| 9b | PV | `T.gemm(P16, KV_shared, acc_o, initC=False, size=[BH, block_N, D])` | 前缀列用法先例 `sparse_mla_fwd_dynamic_shape.py` `CubePVMatmul` | Developer |
| 10 | 归一化 | `T.vbrc(l, l_mat)` + `T.vdiv(acc_o, l_mat, acc_o)` | 单 op（§1.6.1 OPT-R 否决乘倒数） | Developer |
| 10b | 写出 | `T.vcast(acc_o, o_ub, round_mode="rint")` + `T.copy(o_ub, Output[b, s, h0:h0+BH, 0:D])` | 双显式同 extent 切片 | Developer |

### 3.3 计算伪代码

> 按模板要求：`T.Tensor` 的 dtype 直接作第二位置参数（不用 `dtype=` 关键字），避免 `name 'in_dtype' is not defined` 假报警。

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def _sparse_mla_kernel(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk,
                       kv_stride, q_start_index_s, kv_group=1, sm_scale=None,
                       is_causal=True, cp0=False, dtype="float16",
                       block_heads=32, block_N=64, num_stages=2):
    assert is_causal and kv_group == 1 and kv_stride == 1 and heads % block_heads == 0
    assert topk % block_N == 0
    Dqk = dim + tail_dim
    D = dim
    s = 1.0 / math.sqrt(Dqk)                       # 自然 exp 域（OPT-1）；源 sm_scale = s*log2e
    n_hb = heads // block_heads
    num_logical = batch * seq_len * n_hb
    dtype_ = dtype
    accum = "float32"
    SENT = -1e38                                    # OPT-3 有限哨兵

    @T.prim_func
    def main(Q: T.Tensor((batch, seq_len, heads, Dqk), dtype_),
             KV: T.Tensor((batch, seq_len_kv, kv_group, Dqk), dtype_),
             Indices: T.Tensor((batch, seq_len, kv_group, topk), "int32"),
             Output: T.Tensor((batch, seq_len, heads, D), dtype_)):
        with T.Kernel(num_logical, is_npu=True) as (cid, _):
            hb = cid % n_hb
            t = cid // n_hb
            s_i = t % seq_len
            b_i = t // seq_len
            h0 = hb * block_heads
            max_kv_i = q_start_index_s + s_i          # stride_kv==1, cp0==False

            Q_shared  = T.alloc_shared((block_heads, Dqk), dtype_)
            KV_shared = T.alloc_shared((block_N, Dqk), dtype_)
            P_shared  = T.alloc_shared((block_heads, block_N), dtype_)
            O_ub      = T.alloc_shared((block_heads, D), dtype_)
            idx_ub    = T.alloc_shared((block_N,), "int32")
            pen       = T.alloc_shared((block_N,), accum)
            s_buf     = T.alloc_shared((block_heads, 1), accum)   # s 广播（可选）
            m_prev    = T.alloc_shared((block_heads, 1), accum)
            m_cur     = T.alloc_shared((block_heads, 1), accum)
            m_new     = T.alloc_shared((block_heads, 1), accum)
            ms        = T.alloc_shared((block_heads, 1), accum)
            alpha     = T.alloc_shared((block_heads, 1), accum)
            lsum      = T.alloc_shared((block_heads, 1), accum)
            loc_sum   = T.alloc_shared((block_heads, 1), accum)
            alpha_mat = T.alloc_shared((block_heads, D), accum)
            l_mat     = T.alloc_shared((block_heads, D), accum)
            tmp       = T.alloc_shared((block_heads, block_N), accum)

            acc_s = T.alloc_fragment((block_heads, block_N), accum)   # L0C
            acc_o = T.alloc_fragment((block_heads, D), accum)         # L0C

            T.copy(Q[b_i, s_i, h0:h0 + block_heads, 0:Dqk], Q_shared)
            vmin = -T.infinity(accum)
            T.vbrc(vmin, m_prev)
            value_zero = 0
            T.vbrc(value_zero, lsum)
            T.vbrc(value_zero, acc_o)

            for k in T.Pipelined(topk // block_N, num_stages=num_stages):
                T.copy(Indices[b_i, s_i, 0, k * block_N], idx_ub)      # [block_N] int32

                # (2) 列掩码 → 预填 acc_s
                for hi, bi in T.Parallel(block_heads, block_N):
                    acc_s[hi, bi] = T.if_then_else(idx_ub[bi] <= max_kv_i, 0.0, SENT)

                # (3) 稀疏 gather（行连续；单次 clamp 防越界）
                for bi in T.serial(block_N):
                    idx_c = T.max(T.min(idx_ub[bi], seq_len_kv - 1), 0)
                    T.copy(KV[b_i, idx_c, 0, 0:Dqk], KV_shared[bi, 0:Dqk])

                # (4) QK：sc = Q @ K^T + 预填掩码
                T.gemm(Q_shared, KV_shared, acc_s, initC=False, b_transpose=True)

                # (5) running max
                T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                T.vmax(m_prev, m_cur, m_new)

                # (6) alpha = exp(m_prev*s - m_new*s)   (OPT-2)
                T.vmul(m_new, s, ms)
                T.vmul(m_prev, s, tmp1_row)                 # [BH,1]
                T.vsub(tmp1_row, ms, tmp1_row)
                T.vexp(tmp1_row, alpha)

                # (7) P = exp(acc_s*s - ms)            (OPT-1 + OPT-2)
                T.vmul(acc_s, s, tmp)
                T.vsub(tmp, ms, tmp)                        # ms 行广播到 [BH,block_N]
                T.vexp(tmp, acc_s)
                T.vcast(acc_s, P_shared, round_mode="rint")

                # (8) l = l*alpha + rowsum(P)
                T.reduce_sum(acc_s, loc_sum, dim=1, clear=True)
                T.vmul(lsum, alpha, lsum)
                T.vadd(lsum, loc_sum, lsum)

                # (9) rescale + PV
                T.vbrc(alpha, alpha_mat)
                T.vmul(acc_o, alpha_mat, acc_o)
                T.gemm(P_shared, KV_shared, acc_o, initC=False, size=[block_heads, block_N, D])

                T.copy(m_new, m_prev)

            # (10) 归一化 + 写出
            T.vbrc(lsum, l_mat)                             # 注意：vbrc 标量→buffer；[BH,1]→[BH,D] 行广播见 §9 风险
            T.vdiv(acc_o, l_mat, acc_o)
            T.vcast(acc_o, O_ub, round_mode="rint")
            T.copy(O_ub, Output[b_i, s_i, h0:h0 + block_heads, 0:D])

    return main
```

> 伪代码为主选方案（BH=32）的**设计级**表达：`T.vbrc` 的行/标量广播细节、`s`/`ms` 的 buffer 化、`m_prev` 的行形态在
> Stage 3 需按 `docs/` 精确签名与设备探针落地（§9 风险 R-2/R-3）。`T.reduce_max/sum` 的 `clear` 语义以
> `docs/Tilelang.language/规约操作/T.reduce.md §2.3` 为准（源码签名默认 `clear=True`，文档正文有偏差）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm(src1,src2,dst,size,initC,a_transpose,b_transpose)` | `docs/Tilelang.language/线性代数操作/T.gemm.md`（Developer 签名 L8；`size`/`initC`/`b_transpose` 参数表 L21-24） | 文档 + 先例 `sparse_mla_fwd.py`（`size` 先例 `sparse_mla_fwd_dynamic_shape.py`） |
| `T.copy` 切片/索引 `size=` | `docs/Tilelang.language/内存操作/T.copy.md`（L20/30/33/37-42） | 文档 + 先例（串行行 gather） |
| `T.reduce_max/sum(dim=1, clear=)` | `docs/Tilelang.language/规约操作/{T.reduce_max,T.reduce_sum,T.reduce}.md` | 文档 + 先例（`clear=True` 强制） |
| `T.vmax/vmul/vsub/vadd/vdiv/vexp/vcast/vbrc` | `docs/Tilelang.language/数学操作/T.v*.md`、`数据类型转换操作/T.vcast.md`、`shape操作/T.vbrc.md` | 文档 + 多个先例 |
| `T.Pipelined` | `docs/` + `sparse_mla_fwd.py` L109 | 先例 |
| `T.alloc_fragment` / `T.alloc_shared` | `decision-tree.md §4` | 先例 |
| `T.Kernel(n, is_npu=True)` | 一维；`TRAP-threads-kwarg-noop` | 先例（全部 npuir 用例） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| **不支持三维 Kernel** | ✅ 涉及（源为 `T.Kernel(3D)`） | 一维 `T.Kernel(num_logical, is_npu=True)` + `cid` 反解（R6） |
| **GPU 专用 API** | ✅ 涉及（`T.wgmma_gemm`/`T.wait_wgmma`/`T.alloc_barrier`/`T.barrier_wait`/`cp_async_barrier_noinc`/`T.set_max_nreg`/`T.get_thread_binding`/`threads=`/CUDA `compile_flags`） | 全部**不使用**（§0.5 四态处置）；等价能力见 §3.2 |
| **GEMM 非整除风险** | ⚠️ 需显式处理 | **本算子无需 M/N padding**：`block_heads=32` 整除 `H=128`；`block_N=64` 整除 `topk ∈ {256,512,1024,2048}`；`D=512`、`Dqk=576` 为静态尾维。**构造期断言**（`heads % block_heads == 0 && topk % block_N == 0`），不满足则在 Python 层报 `ValueError`（源同款做法 L119-125） |
| **L0C 溢出风险** | ✅ 涉及 | `acc_s[32,64]f32` + `acc_o[32,512]f32` = 8KB + 64KB = **72KB ≤ 128KB** ✓（`CONST-capacity-910B2C`） |
| **分核策略缺失/不适配** | ✅ 涉及 | §5.5 三要素齐全（逻辑 4096 > 物理 20 → 极大规模；核内串行受 `TRAP-DEVMODE-PERSIST-GEMM` 限制不可用 → 非持久 + 硬件波次调度） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总；完整分析见 §0.5/§0.6）

| 差异项 | 参考实现（GPU / Hopper） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------------------|---------------------|----------|
| Kernel 维度 | `T.Kernel(S·replicate_h, B, kv_group, threads=384)`（三维 + 384 线程；本任务 `replicate_h=2` ⇒ `(2048,1,1)`） | 一维 `T.Kernel(num_logical, is_npu=True)` | R6：`cid` 反解；无 `threads`（`TRAP-threads-kwarg-noop`） |
| 线程模型 | warp-specialized（128 消费者 ×2 + 128 生产者） | 单程序 + 自动 CV-split | R2 |
| 矩阵乘 | `T.wgmma_gemm` + `T.wait_wgmma`（576 拆 3 段） | `T.gemm`（576 一次；PV 用 `size=` 前缀列） | R1/R5 |
| 累加器 | 寄存器 fragment `acc_o = 2×[64,256]`（`h_per_block=64`，`replicate_h=2`）≈ 128KB + `acc_s[64,64]` 16KB | L0C fragment `[block_heads,D]` + `[block_heads,block_N]`（128KB 上限） | R4：`block_heads=32` |
| 同步 | 6 个 mbarrier + cp.async 计数 | `T.Pipelined(num_stages=2)` 自动 | R3 |
| 稀疏 gather | 生产者线程 cp.async 向量化 | 串行 `T.copy` 行拷贝 + 单次 clamp | R8、§1.6.2 点 12 |
| exp | `exp2`（LOG2E 折入） | `T.vexp`（自然 exp 域，**fp32 域执行**） | §1.6.1 OPT-1；bf16 文档矩阵缺口见 §9 R-13 |
| 掩码 | `-inf` | `-1e38` 有限哨兵 | §1.6.1 OPT-3 |
| 编译 flags | nvcc/ptxas 专用 | 无（不设 `pass_configs`） | `mha_decode_paged` 先例 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/deepseek_v32/sparse_mla_fwd.py` | **极高** | Developer MixCV + online softmax + `T.Pipelined` + `T.gemm(b_transpose=True)` + 稀疏 gather + `T.vmax`/`T.vexp2`/`T.reduce_*` 链 |
| `examples/deepseek_v4/example_sparse_mla_fwd_kernel.py` | **极高** | 串行行 gather（`T.copy(KV[b,cur_idx,0,0:D], KV_shared[bi,0:D])`）+ v-prefix online softmax + 一维 grid 反解 |
| `examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` | 高 | `T.gemm(..., size=[heads, block, dim])` 前缀列 PV；稀疏 gather 的 clamp/掩码形态；`T.Scope("Cube"/"Vector")`（Expert 备选） |
| `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py` | 高 | 稀疏 gather + `T.serial(idx)` + `min` 钳位 + workspace 形态（R3 候选 3 的反例） |
| `examples/indexer/indexer_fwd.py` | 中 | `T.Pipelined` + `T.gemm` + `T.reduce_sum` + 标量边界的手写掩码（`T.serial` + `-inf`） |
| `examples/mha_decode_paged/`（迁移先例） | 中 | 迁移 DESIGN 结构、Developer MixCV、`T.vbrc` 广播、有限哨兵、非持久 grid 的论证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape（mainstream） | dtype | 说明 |
|--------|---------------------|-------|------|
| `q` | `(1, 1024, 128, 576)` | float16 / bfloat16 | BSHD；尾部 64 为共享 RoPE |
| `kv` | `(1, 2048, 1, 576)` | same | MLA 压缩 KV（`H_kv=1`） |
| `indices` | `(1, 1024, 1, 2048)` | int32 | 每行 top-k 稀疏位置 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Output` | `(1, 1024, 128, 512)` | same_as(q) | `o = (B,S,H,D)`（`jout_idx=[-1]` 由 JIT 分配） |

### 4.3 中间缓冲区（主选 BH=32、`block_N=64`）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `Q_shared` | `(32, 576)` | 输入 dtype | L1 | QK 的 A 操作数 |
| `KV_shared` | `(64, 576)` | 输入 dtype | L1 | QK 的 B（转置）/ PV 的 B（前缀列） |
| `P_shared` | `(32, 64)` | 输入 dtype | L1 | PV 的 A 操作数（量化后 P） |
| `acc_s` | `(32, 64)` | float32 | **L0C**（fragment） | QK 输出 / softmax 输入 |
| `acc_o` | `(32, 512)` | float32 | **L0C**（fragment） | PV 累加器 |
| `idx_ub` | `(64,)` | int32 | UB | 本块的 top-k 索引 |
| `pen` / `s_buf` / `m_*` / `ms` / `alpha` / `lsum` / `loc_sum` | `(32,1)` 或 `(64,)` | float32 | UB | 掩码/归约/在线状态 |
| `alpha_mat` / `l_mat` | `(32, 512)` | float32 | UB | 行广播目标（§9 R-2） |
| `tmp` | `(32, 64)` | float32 | UB | exp 中间量 |
| `O_ub` | `(32, 512)` | 输入 dtype | UB | 输出中转（cast 后） |

### 4.4 内存搬运路径

```
GM[Q]       --T.copy--> L1[Q_shared]
GM[indices] --T.copy--> UB[idx_ub] --scalar/vector--> 掩码 pen / 地址 idx_c
GM[KV]      --T.copy(串行 64 行, 每行 1152B 连续)--> L1[KV_shared]      (T.Pipelined 双 stage)
L1[Q_shared],L1[KV_shared] --T.gemm(b_transpose)--> L0C[acc_s] --Vector(exp/match)--> L1[P_shared]
L1[P_shared],L1[KV_shared] --T.gemm(size=[32,64,512])--> L0C[acc_o] (跨块累加, rescale 由 Vector 做)
L0C[acc_o] --vdiv/vcast--> UB[O_ub] --T.copy--> GM[Output]
```

- Q 只在 kernel 起始搬一次（`Q hoist`）；KV 每块搬一次（不可复用）。
- 无 `store_fixpipe` GM 往返（Developer 自动 CV 切分，L0C↔Vector 直连）。

### 4.5 UB / L1 / L0C 内存预算（主选 BH=32、block_N=64、num_stages=2）

| 层级 | Buffer | 大小 (Bytes) | 小计 | 上限（`CONST-capacity-910B2C`） |
|------|--------|-------------|------|-------------------------------|
| **L0C** | `acc_s (32×64) f32` | 8,192 | **72 KB** | 128 KB ✓ |
| | `acc_o (32×512) f32` | 65,536 | | |
| **L1** | `Q_shared (32×576) f16` | 36,864 | **≈ 184 KB** | 512 KB ✓ |
| | `KV_shared (64×576) f16` ×2 stage | 147,456 | | |
| | `P_shared (32×64) f16` | 4,096 | | |
| **UB** | `O_ub (32×512) f16` | 32,768 | **≈ 169.5 KB**（物化态；逐行求和 = 173,568 B） | 192 KB（**膨胀后见下，物化态不可行**） |
| | `alpha_mat (32×512) f32` | 65,536 | | |
| | `l_mat (32×512) f32` | 65,536 | | |
| | `tmp (32×64) f32` + 行状态（`s_buf/m_prev/m_cur/m_new/ms/alpha/lsum/loc_sum` = 8×32×4B）+ `idx_ub (64×4B)` + `pen (64×4B)` | 9,728 | | |

> **小计口径（本次修订统一）**：物化态 UB = `32,768 + 65,536 + 65,536 + 9,728 = 173,568 B ≈ 169.5 KB`
> （原稿 §4.5 的「≈67 KB」小计与其所列各行、注释「171KB」三处不自洽，现统一为逐行求和 173,568 B）。
> **UB 膨胀系数取档（`CONST-capacity-910B2C` 同时记录两档，必须显式取档，不得只取低档）**：
> - **`~1.7x`**（auto-multi-buffer 主档，`CONST-capacity-910B2C` 主旨）⇒ 169.5 × 1.7 ≈ **288 KB > 192 KB ⇒ 物化态不可编译**；
> - **`×1.10–1.12`**（auto-multi-buffer=false 宽 config，2026-09-15 第五轮 4 点标定，`CG-2026-0008` / `TRAP-UB-multibuffer-inflation`；`PL-1.11`）⇒ 169.5 × 1.12 ≈ **190 KB < 192 KB（贴限）**；
> - **反例边界 `0.981×`**（显式 alloc 结构，2026-09-16 r9）⇒ ≈ 166 KB（actual 低于手工）。
> ⇒ **结论：物化态在 `~1.7x` 档不可行，削减方案为必需（非可选）**。
> **削减方案（Stage 3 落地）**：`alpha_mat`/`l_mat` 用 `T.vbrc` 的 `[BH,1]→[BH,D]` 行广播直接进 `T.vmul`/`T.vdiv`
> （**不物化** `[BH,D]` 中间量），或改用 `T.Parallel` 带缓冲变量。削减态 = `O_ub 32,768 + tmp 8,192 + 行状态/idx/pen 1,536 ≈ 41.5 KB`，
> **×1.7 保守上限 ≈ 71 KB < 192 KB ✓**。**列入 §9 风险 R-2**（并与 `CG-2026-0008` 的 UB 膨胀系数不确定区挂钩）。

### 4.6 动态轴定义

无动态轴（`batch/seq_len/seq_len_kv/heads/topk` 全为编译期常量；`lru_cache` 按 shape 特化）。
参考：`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` 的 `T.symbolic` 形态可作后续扩展，但 manifest 全静态，本设计不做动态化。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: **混合（MixCV）** —— 两个 Cube `T.gemm`（QK、PV）+ Vector 在线 softmax。
**判定依据**：含 matmul（`q·kvᵀ` 与 `P·V`）与逐元素后处理（softmax），按 `decision-tree.md §2` 属「CV 融合算子」。

### 5.2 Block 划分

```python
block_heads = 32          # 理由: L0C 预算 (BH*(D+block_N)*4 ≤ 128KB ⇒ BH ≤ 56) 且整除 H=128
block_N     = 64          # 理由: 整除 topk ∈ {256,512,1024,2048}; L1 双 stage 147KB; acc_s 8KB
num_stages  = 2           # 理由: 对应源双缓冲; L1 余量充足
num_logical = B * S * (H // block_heads)      # mainstream = 1*1024*4 = 4096
```

### 5.3 约束分析

- **对齐约束**：`block_N=64`（尾轴 576×f16 = 1152B ≥ 32B ✓；64 为 fp16 向量宽 ×8 的 8 倍）；`D=512`、`Dqk=576` 均 32B 对齐。
- **分形约束**：QK `M=32≥16, N=64≥16, K=576≥32` ✓；PV `M=32≥16, N=512≥16, K=64≥32` ✓。
- **L0C 容量**：72KB ≤ 128KB ✓（§4.5）。
- **L1 容量**：≈184KB ≤ 512KB ✓。
- **UB 容量**：见 §4.5（物化态 ≈169.5KB 在 `~1.7x` 膨胀下超限 ⇒ **削减方案为必需**；削减态 ≈41.5KB，×1.7 ≈71KB ✓）。
- **GEMM 非整除**：**不涉及**（见 §3.5.1）。

### 5.4 注意事项（非整除处理策略）

- **不涉及 M/N 非整除**：本算子的分块维（`H`、`topk`）在 manifest 全部 workload 上整除（构造期断言保证）。
- **`indices` 的边界**：索引可能 > `max_kv_i`（掩码）或（理论上）≥ `seq_len_kv`；用**单次 clamp + 列掩码**处理（§0.6 R8），不引入运行时分支。
- **`block_N` 与 `topk` 的耦合**：`topk` 必须是 `block_N` 的倍数（源同款约束，L119-125 注释明确「否则会读入 index=0 导致错误 KV」）。
- **`topk < block_N`**：smoke `topk=256 ≥ 64` ✓，不出现不足一块的情形。

### 5.5 分核策略（物理核数适配）⭐

> 标准：`.agents/skills/_shared/standards/core-split-strategy.md §1/§2.1`（依据 `docs/开发指南.md §3.3`）。

- **① 逻辑核数计算**：`num_logical = B · S · (H / block_heads)`
  - mainstream：`1 · 1024 · (128/32) = **4096**`；smoke：`1 · 128 · 4 = 512`；
  - `longer-kv-lower-topk`：`1 · 512 · 4 = 2048`；`long-kv-low-topk`：`1 · 256 · 4 = 1024`。
  - （备选 BH=64：`B·S·(H/64)` = 2048 / 256 / 1024 / 512。）
- **② 物理核数依据**（实查，禁止假设）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()        # 实查返回 20（Ascend910B3；get_arch()='Ascend910B3'）
  NPUUtils.get().get_aivector_core_num() # 返回 40
  ```
  **实查记录（本机，2026-10-15）**：`get_aicore_num() → 20`，`get_aivector_core_num() → 40`，`get_arch() → 'Ascend910B3'`。
  本算子为 **Cube + Vector 混合** → 按标准「Cube/混合算子直接使用返回值」取 **物理核数 = 20**。
- **③ 规模判定与分核方案**：`num_logical = 4096 ≫ 20` → **极大规模**。
  - **首选方案（标准第三档：核内串行 persistent）在此不可用**：`TRAP-DEVMODE-PERSIST-GEMM`（`status: verified`，
    tilelang 0.1.2+1990aa9fe4 / 910B2C / 2026-09-17）实证 **Developer + persistent + gemm 混排运行时崩溃**
    （"unaligned UUB addresses"）；`mha_decode_paged` 迁移先例亦因此选非持久（其 §0.6 R1）。
  - **实际方案（走「无法对齐兜底」的等价处置）**：**非持久一维 grid**（`num_logical = 4096` 个逻辑内核），
    由运行时**波次串行调度**（超物理核部分串行执行）；核内无 `T.serial` 多任务循环。
  - **对齐取值分析**：`4096 = 20 × 204 + 16` ⇒ **204.8 波次**（16 核执行 205 个任务、4 核执行 204 个）——
    **残余负载不均 ≈ 1/205 ≈ 0.5%**；smoke `512 = 20×25 + 12` ⇒ 25.6 波次（残余不均 ≈ 4%）。
  - **核启动开销**：`T.Pipelined` 级持续计算分摊，且每个逻辑内核工作量可观（gather 64 行×32 块 + 2 个 gemm + 在线链），
    固化的核启动开销相对 payload 可摊销（Stage 4 若实测为瓶颈，可按 §1.6.3 的 shape 分派或增大 `block_heads` 减少核数）。
  - **与标准的偏差披露**：标准第三档要求「核内 `T.serial` 串行处理多个逻辑块任务、循环边界静态」；
    本设计因编译器硬限制（`TRAP-DEVMODE-PERSIST-GEMM`）改为**非持久 + 硬件波次**，**显式记录**残余不均（≈0.5% 主流 / ≈4% smoke）
    与核启动开销；备选 BH=64 把逻辑核数降到 2048（`2048 = 20×102+8` ⇒ 102.4 波次，残余 ≈1%），
    两方案均列 §1.6.3 实验裁决清单。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环。
> 若出现逐元素标量循环，必须能在 §1.6.2 找到对应充分理由。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务（`B·S·H/BH`） | block 级并行（非持久） | `T.Kernel(num_logical, is_npu=True)` | 每核一个 (b,s,hb) 任务；非持久（`TRAP-DEVMODE-PERSIST-GEMM`） |
| topk 维 | tile 级迭代（流水） | `T.Pipelined(topk // block_N, num_stages=2)` | online 递推跨块携带；双缓冲 |
| gather 行（block_N） | 数据依赖行循环 | `T.serial(block_N)` + `T.copy` | §1.6.2 点 12（无向量等价 API） |
| 元素级（掩码） | 向量化 | `T.Parallel(block_heads, block_N)` | 掩码预填（源先例） |
| softmax 归约 | 向量归约 | `T.reduce_max/sum(dim=1)` | §1.6.2 点 2/6 |

### 6.2 循环伪代码

```python
with T.Kernel(num_logical, is_npu=True) as (cid, _):
    hb, t = cid % n_hb, cid // n_hb
    s_i, b_i = t % seq_len, t // seq_len
    # Q hoist + 状态初始化
    T.copy(Q[b_i, s_i, h0:h0+BH, 0:Dqk], Q_shared)
    T.fill(m_prev, -inf); T.fill(lsum, 0); T.clear(acc_o)
    for k in T.Pipelined(topk // block_N, num_stages=num_stages):
        T.copy(Indices[b_i, s_i, 0, k*block_N], idx_ub)
        for hi, bi in T.Parallel(BH, block_N):
            acc_s[hi, bi] = T.if_then_else(idx_ub[bi] <= max_kv_i, 0.0, SENT)
        for bi in T.serial(block_N):
            idx_c = T.max(T.min(idx_ub[bi], seq_len_kv - 1), 0)
            T.copy(KV[b_i, idx_c, 0, 0:Dqk], KV_shared[bi, 0:Dqk])
        T.gemm(Q_shared, KV_shared, acc_s, initC=False, b_transpose=True)
        T.reduce_max(acc_s, m_cur, dim=1, clear=True)
        T.vmax(m_prev, m_cur, m_new)
        # alpha = exp(m_prev*s - m_new*s);  P = exp(acc_s*s - m_new*s)
        ...
        T.reduce_sum(acc_s, loc_sum, dim=1, clear=True)
        # l = l*alpha + loc_sum;  acc_o = acc_o*alpha
        ...
        T.gemm(P_shared, KV_shared, acc_o, initC=False, size=[BH, block_N, D])
        T.copy(m_new, m_prev)
    # 归一化 + 写出
```

### 6.3 流水线优化

- `T.Pipelined(num_stages=2)`：KV 双 stage，使 `GM→L1` 的 gather 与 Cube/Vector 计算重叠（对应源的双缓冲）。
- `num_stages` 可调（1/2/3）：受 L1 容量约束（每 stage 的 `KV_shared` 72KB；3 stage = 216KB + Q 37KB + P 4KB = 257KB ≤ 512KB ✓ 亦可），
  默认 **2**（与源一致）；Stage 4 可扫。
- Q hoist：`Q_shared` 在循环外加载一次（源亦如此）。

### 6.4 尾块处理

- **无尾块**：`block_N` 整除 `topk`，`block_heads` 整除 `H`（构造期断言）。
- **掩码尾**（`indices` 的无效/越界项）：由**列掩码 + 单次 clamp** 处理（非 shape 非整除，而是数据有效性问题），见 §0.6 R8 / §1.6.2 点 13。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（Developer 模式）。

### 7.2 同步点说明

- Developer 模式下，Cube（`T.gemm`）与 Vector（v-prefix）之间的跨引擎依赖由编译器在 CV-split 阶段自动插入同步；
  `T.Pipelined` 自动生成多缓冲的 DMA/计算同步（无需手写 `set/wait_flag`）。
- **无手动同步点**（不使用 `T.set_flag`/`T.wait_flag`/`T.sync_block_set`/`T.sync_block_wait`/`T.pipe_barrier`）。
- 若 Stage 3 遇到自动切分失败或性能不足，按 `PL-1.16-expert-dualscope-bypass` 评估切换 Expert 双 `Scope` 形态（§9 R-8）。

### 7.3 pass_configs 配置

```python
# 不设 pass_configs —— npuir 无 TL_ENABLE_FAST_MATH 等 CUDA flags；
# 照搬源 compile_flags 会触发 AttributeError（mha_decode_paged 先例）。
```

> **注意**：Expert 备选若启用，须按先例关闭 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`（`PL-1.16`），
> 但本设计主选为 Developer，不涉及。

## 8. 验证方案

### 8.1 Golden 函数

> **迁移任务**：golden 以 **§0.1 源算子语义**为唯一依据，**优先移植源仓参考实现**
> （`examples/TileOPs/tests/ops/test_deepseek_dsa_decode.py::DsaDecodeTest.ref_program`），**不得复刻 §0.6 的 NPU 算法**——
> 保证验证独立性（NPU 算法用 576 维单次 gemm / head 分块 / 自然 exp 域，golden 用 `einsum` 全头 + `masked_fill` + `softmax`，两者结构不同）。

```python
def golden_sparse_mla_decode(q, kv, indices, dim, dim_tail, stride_kv, q_start_index_s,
                             sm_scale=None):
    """移植自 TileOPs 参考实现（去重语义；BSHD；mask = scatter(indices) & compressed_causal_mask）。"""
    dev = q.device
    q = q.float(); kv = kv.float()
    indices = indices.transpose(1, 2)                       # [B, H_kv, S, topk]
    b, sq, h, dim_q = q.shape                               # dim_q = dim + dim_tail
    _, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * stride_kv - sq
    assert kv.shape[-1] == dim + dim_tail
    k = kv
    v = kv[..., :dim]
    g_index, h_index = g, h // g

    compressed_causal_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=dev
    ).view(-1, 1) >= torch.arange(
        stride_kv - 1, sk * stride_kv, stride_kv, dtype=torch.int32, device=dev
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
    mask[:, :, :stride_kv - 1, 0] = True                    # cp0 退化位（stride_kv=1 时为空）
    mask = mask.view(b, g_index, 1, sq, sk)

    q5 = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q5, k)        # [B, G, H/G, S, S_kv]
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk).view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim)
    return o.to(torch.float16)                              # 按 dtype 返回（bf16 用例改 to(torch.bfloat16)）
```

> **golden 的独立实现要素**（证明其不复刻 NPU 算法）：① 全头（无 head 分块）；② 显式**去重** mask（与源 kernel 的逐条累加不同，§0.1）；
> ③ `score.masked_fill(-inf).softmax()`（两遍式，非 online）；④ `einsum` 单次（非 topk 分块）；⑤ 无 exp2/LOG2E 常数。

### 8.2 精度标准

| dtype | atol | rtol | 来源 |
|-------|------|------|------|
| float16 | **3e-4** | **1e-5** | manifest 测试断言（`test_deepseek_dsa_decode.py` L109 `test.check(op, ..., atol=3e-4, rtol=1e-5)`） |
| bfloat16 | **3e-4** | **1e-5** | 同上（smoke 第二例） |
| （项目标准 fallback） | 1e-2 | 1e-2 | `design-template.md §8.2`（fp16）通用标准 |

> **风险提示**：测试断言的 `atol=3e-4` **严于** fp16 在 O(1) 输出上的单 ulp（≈9.8e-4）。本设计以测试断言为目标；
> 若 Stage 3 在此容差下不达标，按项目标准（1e-2）与 REVIEW 记录处理（**不得**通过改变算法语义来迎合容差）。
> §1.6.1 D-1 已验证采纳优化项自身在 **1e-2** 容差内违反率为 0（严格容差下的余量由 Stage 3 L0 实测确认）。

### 8.3 L0 门槛测试计划（blocking）

| 用例 | shape / 参数 | dtype | 覆盖点 |
|------|-------------|-------|--------|
| L0-1 | `q(1,128,128,576) / kv(1,256,1,576) / idx(1,128,1,256) / topk=256 / q_start=128` | fp16 | **smoke**：小 topk（4 块）、单 batch、`q_start>0`（cp0=False） |
| L0-2 | `q(1,64,128,576) / kv(1,512,1,576) / idx(1,64,1,256) / topk=256 / q_start=64` | bf16 | **smoke**：bf16 路径 + 掩码覆盖（**注意**：softmax 的 exp 在 **fp32 域**执行，不在 bf16 缓冲上直接调 `T.vexp`——两 API 文档 dtype 矩阵均不含 bf16，见 §9 R-13） |
| L0-3 | `q(1,1024,128,576) / kv(1,2048,1,576) / idx(1,1024,1,2048) / topk=2048` | fp16 | **full**：mainstream（32 块、mask 边界） |
| L0-4 | `q(1,512,128,576) / kv(1,4096,1,576) / idx(1,512,1,1024) / topk=1024` | fp16 | **full**：longer-kv-lower-topk |
| L0-5 | 构造 `indices` 含**部分越界/无效**（> `max_kv_i`）的行 | fp16 | **boundary**：clamp + 列掩码正确（§0.6 R8） |
| L0-6 | 单行 `topk` 全无效（全掩码退化） | fp16 | **boundary（informational，非 blocking）**：记录行为（`l=0` → 输出 0），与 golden 的 nan 差异列入 §9 R-11 |

> 完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，此处只列 L0 门槛。
> 集成测试入口：`examples/TileOPs/tests/ops/test_deepseek_dsa_decode.py`。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 依据 / 影响 | 缓解 |
|---|------|------------|------|
| R-1 | **KV 重读放大 + 带宽常数未实测**（主选 ×4）是 roofline 主导项，估算区间 [4.0, 9.4] ms 宽 | §1.6.0 R4（`CONST-ascend910-9362-profile` 属 9362、`CONST-elu-910B3-bw-floor` 口径不同）；910B3 核→GM/L2 带宽无实测 | §1.6.3 备选 BH=64 + 实验裁决计划；Stage 4 反解有效 gather 带宽并回写 constants |
| R-2 | **UB 物化态超限**：`alpha_mat`/`l_mat`（[32,512] f32）物化后 UB = **173,568 B ≈ 169.5 KB**；`~1.7x` 膨胀 ⇒ **288 KB > 192 KB 不可编译**（`×1.10–1.12` 档 ≈190 KB 贴限） | §4.5（小计口径本次修正为 173,568 B）；`TRAP-UB-multibuffer-inflation` / `CONST-capacity-910B2C`（膨胀系数结构依赖，`CG-2026-0008`） | **必须**用行广播直接进 `T.vmul`/`T.vdiv` 不物化中间量（削减态 ≈41.5 KB，×1.7 ≈71 KB）；Stage 3 编译探针（`ub overflow` 报文反推精确实测需求） |
| R-3 | `T.vbrc` / 行广播的精确形态 | `TRAP-vbrc-empty-broadcast`：`(BM,1)→(BM,1)` 被拒，`(BM,1)→(BM,tile_n)` 可用；`TRAP-vbrc-scalar-shared`：vbrc 标量→shared 不可用；`TRAP-expert-v-operands`：标量源须 let 绑定、v-op 切片仅静态边界 | Stage 3 探针确认；`alpha_mat`/`l_mat` 形态二选一（物化 vs 广播）；`m/alpha/lsum` 用 `[BH,1]` 行向量 v-op |
| R-4 | **Developer 禁 persistent** ⇒ 非持久 4096 内核，核启动开销 + 波次残余（≈0.5% 主流 / ≈4% smoke） | `TRAP-DEVMODE-PERSIST-GEMM`（verified）；`CG-2026-0010`（本次第二识别 → recurring） | 非持久 grid；备选 BH=64 把核数降到 2048；Stage 4 若核启动成瓶颈按 §1.6.3 分派 |
| R-5 | `T.vcmp` 在整型上可能标量化（低效） | `TRAP-expert-v-operands`：`vcmp` int16 全形态降级为标量指令（[32,256] 实测 236µs/链） | 本设计**不用** `T.vcmp` 做有效性判据——用 `T.if_then_else` + 算术掩码（`pen`）；Stage 3 若发现掩码链标量化，改「算术惩罚掩码」 `scr=clamp(diff,0,1)` + `sc += scr·(-1e38)`（`PL-1.11`） |
| R-6 | **累加顺序变化**：576 维一次 gemm（源 3 段）；head 分块；topk 分块（`block_N=64` vs 源 `block_i=64`） | 数学等价，但 fp16/bf16 舍入分组变化 | golden 独立验证；§1.6.1 D-1 已证优化链在容差内违反率 0 |
| R-7 | **测试容差严格**（atol=3e-4 < fp16 单 ulp@O(1)） | §8.2；可能对 fp16 输出过严 | Stage 3 L0 实测；不达标时按 §8.2 fallback（1e-2）并在 REVIEW 记录（不改语义） |
| R-8 | Developer 自动 CV-split 在该结构上失败/性能不足 | `PL-1.16-expert-dualscope-bypass` 判据（aiv_scalar>50% / 崩溃类 / 模式级不兼容） | 切 Expert 双 `Scope` + `T.sync_block_set/wait` + `pass_configs={TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False}` |
| R-9 | `T.gemm(..., size=[BH,block_N,D])` **前缀列**在 Developer 形态未实测（先例为 Expert：`sparse_mla_fwd_dynamic_shape.py`；Developer 文档签名含 `size=`） | 若前缀列语义不符 → PV 列错 | Stage 3 首编探针；fallback = 把 V 单独 gather 到 `KV_v[block_N, D]`（gather 字节 +89%，代价已量化）或仿 `example_sparse_mla_fwd_kernel.py` 拆 K/V 装载 |
| R-10 | 构造期不变量 `H % block_heads == 0`、`topk % block_N == 0`、`istride_kv==1 且 cp0==False` | manifest 全部满足；非 manifest 输入需在 Op 层报错或回退 | 工厂 Python 层断言（源同款）；Op 层 `q_start_index_s` 校验保留 |
| R-11 | 全掩码行（无有效索引）输出语义 | 源/golden 给 nan；本设计给 0（`l=0`） | manifest 不产生（`supply_prog` 保证 ≥1 有效）；L0-6 记录，不作 blocking |
| R-12 | 工具链 stale（`stale_count=236`） | 引用条目需核对 `status`/`device`/`toolchain` | 本设计引用均为 `status: verified`（`PL-1.9-twophase`/`PL-1.11`/`PL-1.16`/`TRAP-C10`/`TRAP-DEVMODE-PERSIST-GEMM`）；constants 逐条读 `device` 字段并在 §1.6.0 标注跨设备沿用 |
| R-13 | `T.vexp` 的 bf16 支持（**文档矩阵缺口**） | `docs/.../数学操作/T.vexp.md` §2.2.1 与 `T.vexp2.md` §2.2.1 **均标 bf16 ✗**；`CONST-elementwise-op-precision-910_9362`：`T.vexp` 真 exp（rel err 1.182e-7，910_9362 实测） | **本设计 softmax 的 exp 在 fp32 域执行**（`acc_s` 累加、`tmp` 均 fp32），**不触发** bf16 缺口；采用 `T.vexp`。**文档矩阵与设备实测可能分歧**（`TRAP-bf16-vop-doc-matrix-divergence`：`vmul/vadd/vsub` 文档标 bf16 ✗ 但 910_9362 实测原生可用且逐位一致）——**判据 = 逐 op 读 §2.2.1 + 设备 micro-probe，不按族类推**；Stage 3 若在任何 bf16 缓冲上直接调 v-prefix 超越函数，须加设备 micro-probe（不得仅凭「文档/示例分歧」放行） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB overflow | `alpha_mat`/`l_mat` 物化 + auto-multi-buffer | 编译失败（`ub overflow, requires …`） | 广播不物化（R-2） |
| L0C overflow | `block_heads` 过大 | 编译失败 / segfault | `block_heads·(D+block_N)·4 ≤ 128KB`（BH ≤ 56） |
| Developer persistent 崩溃 | 用 `T.serial` 多任务 persistent | 运行时 SIGILL | 非持久 grid（R-4） |
| 掩码链标量化 | 用 `T.vcmp` 做 int 比较 | 性能崩塌 | 算术掩码（R-5） |
| 越界读 KV | 索引未 clamp 且超出 `S_kv` | 设备越界 | 单次 clamp（§0.6 R8） |
| `topk` 非 `block_N` 倍数 | 非 manifest 输入 | 读入 index=0 的错误 KV（源注释 L119-125） | 构造期断言 |

### 9.3 特殊场景处理

- **`stride_kv > 1`**：源注释明确前 `stride_kv-1` 行输出 nan（未使用）；本设计 `assert stride_kv == 1`（manifest 契约），
  非 1 时由 Op 层拒绝或扩展 `max_kv_i`/cp0 逻辑（列入后续扩展）。
- **`kv_group > 1`（GQA）**：本设计假定 `kv_group=1`（MLA，`H_kv=1`）；GQA 需加 kv_group 维的 grid 分解（后续扩展）。
- **动态 shape**：manifest 全静态；如需，参照 `sparse_mla_fwd_dynamic_shape.py` 的 `T.symbolic` 形态（本设计不做）。
- **极小/极大 topk**：`topk ≥ block_N` 且整除（断言）。

### 9.4 NPU 能力缺口（capability gap）

1. **CG-2026-0010（本任务独立第二识别 → `occurrences` 1→2、`status` open→recurring，已登记 `capability-gaps.md`）**：
   **Developer 模式对「persistent 分核 + `T.gemm` + v-prefix」混排不支持**（运行时 "unaligned UUB addresses"，无编译期诊断）。
   - **证据**：`TRAP-DEVMODE-PERSIST-GEMM`（`repro/TRAP-DEVMODE-PERSIST-GEMM.py`，verified，tilelang 0.1.2+1990aa9fe4 / 910B2C）；
     本任务实态：mainstream 逻辑核 4096 ≫ 物理核 20，标准「极大规模 ⇒ 核内 `T.serial` persistent」路径不可用（§5.5）。
   - **绕法（本任务采用）**：非持久一维 grid + 硬件波次调度（残余不均 ≈0.5%）；代价 = 无法摊薄核启动开销 + 残余不均。
   - **升级**：达 recurring ⇒ 已产出补齐提案（目标层 codegen / 建议 auto CV-split 支持 persistent 多任务；对外建 issue 须用户批准）。
2. **`TRAP-C10-strided-gather`（已知陷阱，本任务不构成 blocking 缺口）**：NPU 缺跨步 gather 向量指令。
   - 本算子的 gather 是**行连续**（每行 576×f16 = 1152B）→ 用串行 `T.copy` 完全绕开（`example_sparse_mla_fwd_kernel.py` 先例），
     **无被迫换算法 / 无性能上限结论** ⇒ **不新增**能力缺口条目（仅引用陷阱的绕法）。
3. **910B3 核→GM/L2 带宽常数缺失**（**未文档化假设**，非编译器能力缺口）：`constants.md` 无 Ascend910B3 的核→GM 读数
   （`CONST-elu-910B3-bw-floor` 为逐元素聚合 R+W 口径；`CONST-ascend910-9362-profile` 属 910_9362）。
   影响 §1.6.0 roofline 精度 → 交由 Stage 4 实测回写 constants（D 类）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/deepseek_dsa_decode/_sparse_mla_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # D-1 等价性机器验证脚本（已执行，见 §1.6.1）
├── RETROSPECTIVE.md        # Stage 1 复盘（skill Phase 8）
└── _sparse_mla_kernel.py   # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行（EQUIV_PASS） | §1.6.1 D-1 机器验证 |
| `_sparse_mla_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |
| 集成（`examples/TileOPs/.../deepseek_dsa_decode.py`） | ⬜ 待接入 | 脚手架已就绪（`SparseMlaKernel`，Developer，K5/K8/K9 已完成） |

### 10.3 命名规范

- 项目目录名：`deepseek_dsa_decode`（snake_case）；算子目录名：`_sparse_mla_kernel`（沿用迁移函数名）。
- 实现文件：`_sparse_mla_kernel.py`（Stage 3 由脚手架工厂名派生）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ D-1 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，移植 TileOPs 参考实现）
3. ⬜ 算子实现（`_sparse_mla_kernel.py`，Developer MixCV 主选 BH=32）
4. ⬜ L0 门槛测试（§8.3）+ 与 golden 精度比对 → Stage 3
5. ⬜ 集成（`SparseMlaKernel.forward`）+ Stage 4 调优（§1.6.3 主选/备选 A/B = BH=32 vs BH=64）
