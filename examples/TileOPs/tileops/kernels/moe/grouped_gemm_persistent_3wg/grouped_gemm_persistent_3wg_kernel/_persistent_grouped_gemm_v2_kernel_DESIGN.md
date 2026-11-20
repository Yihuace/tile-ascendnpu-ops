# _persistent_grouped_gemm_v2_kernel 算子设计文档

> 迁移任务（harness，developer；**kernel-only** 单元）。源算子：
> `/tmp/opencode/TileOPs-ref/tileops/kernels/grouped_gemm/grouped_gemm_persistent_3wg.py`
> 的 `GroupedGemmPersistent3WGKernel`（GPU/Hopper SM90 warp-specialized 3-WG persistent grouped GEMM）
> 及其 JIT 工厂 `_persistent_grouped_gemm_v2_kernel`（+ 构建助手 `_make_pingpong_kernel` /
> `_make_cooperative_kernel`）。本项目目标：`target="npuir"`（Ascend NPU）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/grouped_gemm_persistent_3wg/`
> （`_persistent_grouped_gemm_v2_kernel` + `_make_pingpong_kernel` / `_make_cooperative_kernel`，
> `.migration_meta.json` 标记 `scope=kernel_only`、`op_carrier=none`）。
>
> **本单元定位**：纯 kernel（**无独立 manifest op 条目、无 Op 类、无 test/bench**）——它只作为 3 个 MoE
> fusion 算子的 `kernel_map` 成员（键 `moe_grouped_gemm_kernel`）被引用。端到端语义 = 分组 GEMM
> `C[e] = A[e] @ B[e]^T`（K 对齐），与已迁移的 `MoeGroupedGemmNopadFwdOp` 等价。规格（shape / dtype /
> distribution）参考 `examples/TileOPs/tileops/manifest/moe.yaml` 的 `MoeGroupedGemmNopadFwdOp.workloads`
> 与其 `tests/ops/test_moe_grouped_gemm_nopad.py` / `benchmarks/ops/bench_moe_grouped_gemm_nopad.py`。
>
> 迁移决策（§0）驱动 §1–§11 全部设计；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> 关键先例（**可行底座**，已冻结并含 Stage 3 实测）：
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`（v2）+ 其实现
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py`
> （Expert 单 Cube scope + 外部调度表 + 动态 extent `T.gemm(size=, b_transpose=True, initC=)` +
> `store_fixpipe`，实测 max_diff 9.537e-07）；调度表产出者胞弟设计
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`。
> 工具链版本戳（本设计全部结论绑定此版本）：tilelang
> `0.1.2+2238099280b481491f96f99482c86246a09c8c68` / CANN `9.0.0`
> （`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0`）/ Ascend910B2C
> （`NPUUtils.get().get_aicore_num()` 实查 = **20**，查询代码与返回值见 §5.5）。
>
> 等价性机器验证（D-1）：`examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/verify_equiv.py`
> （M1/M2 全 `EQUIV_PASS`）。设计期探针（D-3，设备实测）：
> `examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/history_version/design_probe_persistent_gemm.py`
> （4 用例全 PASS，worst max_diff 7.629e-06）+ 日志 `design_probe_persistent_gemm.log`。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 +
> 重设计）。本单元范围 = `_persistent_grouped_gemm_v2_kernel`（GEMM 主核，含内嵌 static-wave 调度器）。
> **分类口径**：本单元数据通路 **Cube 主导、无逐元素浮点向量工作**（唯一数值计算为 WGMMA NT 矩阵乘 +
> epilogue cast）；源的调度索引段（`s_cum` 前缀扫描 + 二分查找）为**纯标量整数**，在 NPU 上须外移
> （§0.6 R1）。详见 §2.1 分类口径澄清。

### 0.1 源算子语义（做什么）

**数学语义**（MoE no-pad 分组 GEMM，NT 布局）：

对每个专家 $e\in[0,E)$，设其 token 数为 $s_e$（`true_sizes[e]`）、在紧致（tight，无 pad）行布局中的起
始行偏移为 $o_e$（`true_offsets[e]`），则

$$
C[o_e + r,\; :] \;=\; A[o_e + r,\; :] \cdot B[e]^{T}, \qquad r \in [0, s_e)
$$

即 $C[i,n]=\sum_{k} A[i,k]\cdot B[e_i,n,k]$，其中 $e_i$ 是第 $i$ 行所属专家；等价地
$C = \text{concat}_e\big(A[o_e:o_e+s_e] \cdot B[e]^T\big)$，且 $\sum_e s_e = \text{numel}$
（有效输入下每行恰属于一个专家，无未覆盖行）。源 docstring 明确其**独立契约**即
``C[e] = A[e] @ B[e]^T`` for every expert，与 `MoeGroupedGemmNopadFwdOp` 一致。

- **规约语义**：规约轴 = $K$（特征维），完整求和；源在 `C_local`（fp32）上按 `block_k` 分块用
  `T.wgmma_gemm(..., transpose_B=True, clear_accum=(k==0))` 顺序累加。重设计改变分块粒度会改变累加
  分组，需评估 fp16/bf16 舍入（§0.6 语义保持论证）。
- **dtype 语义**：输入 A、B 为 fp16/bf16；中间累加 `accum_dtype="float"`（fp32，WGMMA 累加器）；
  输出 C 与输入同 dtype（fp16/bf16），在 epilogue 一次 `T.copy`（fragment→C_shared）内 cast。
- **边界语义 / K 对齐约束**：源**要求 K 对齐**（`K % block_k == 0`，否则 `forward` 抛
  `ValueError: K-aligned only`，docstring「K-aligned only. K-unaligned use GroupedGemmPersistentKernel」）
  与 N 对齐（`N % block_n == 0`）。`numel`（T×top_k）一般不被 `block_m` 整除，每专家尾块行数
  $s_e \bmod block\_m \neq 0$ 普遍存在。源对 A 的 M 尾用 **TMA OOB 零填充**（descriptor `globalDim=numel`）
  + epilogue 守卫（full tile 走 TMA store 快路、partial tile 走谓词 STG 慢路）精确裁剪。
  **no-pad 紧致布局的关键约束**：一个 M-tile 绝不跨专家（tile 按专家内切分），故尾块多余行属于
  **下一个专家的行**，若越界写入会破坏下一个专家的结果 —— 尾块必须精确裁剪。
- **host 侧语义**：`sm_count` 默认取 `torch.cuda.get_device_properties(...).multi_processor_count`
  （persistent grid 大小）；`forward` 只做 shape/dtype/device/contiguity/overlap 校验与 `C` 分配
  （`torch.empty`，无需 memset —— 每行恰被一个 tile 写）；**无 host 预处理、无 `F.pad`**（与
  no-pad 分组 GEMM 不同，3WG 用 TMA OOB 零填充而非 host pad）。工厂按 `numel/E/N/K/dtype/sm_count/block_k`
  `functools.lru_cache` 特化，内层 callable 再按 `(block_m, block_n, block_k, num_stages, threads, group_size_m)`
  编译。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源 op 测试的 `_ref_grouped_gemm_nopad`），
**不得复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| A | (numel, K) | float16 / bfloat16 | 紧致 permuted 激活，多专家行首尾相接，无 pad |
| B | (num_experts, N, K) | 同 A | 专家权重，NT 布局（对 B 转置后右乘），行连续 |
| true_sizes | (num_experts,) | int32 | 每专家真实 token 数 $s_e$ |
| true_offsets | (num_experts,) | int32 | 每专家在 A 中的紧致起始行 $o_e$ |
| **C（输出）** | **(numel, N)** | **同 A** | 源 `out_idx=[]`（`out` 由 host 分配或调用方传入）；非转置布局、行连续 |

> 迁移必需输出 shape = `(numel, N)`。源工厂签名
> `_persistent_grouped_gemm_v2_kernel(numel, num_experts, N, K, dtype, sm_count, block_k)`
> 返回 `_func(block_m, block_n, block_k, num_stages, threads, group_size_m)` → `@T.prim_func`，
> prim_func 实参 `(A, B, true_sizes, true_offsets, C)`。

### 0.3 实现算法解读（怎么算）

源为**单 kernel（内嵌 static-wave 调度器）+ 每 CTA 多 tile（persistent）**。工厂按 `block_m` 分派两套
模板：`block_m ≤ 64` → **pingpong**（2 个 math WG 各处理一个独立 tile）；`block_m ≥ 128` →
**cooperative**（2 个 math WG 把同一 tile 的 M 维劈半，shared-B）。两模板均为 **1 producer WG
（tx<128，TMA）+ 2 consumer/math WG（wgmma）**、`threads=384`、per-WG `num_stages` 深 SMEM ring、
每 slot 一对 full/empty mbarrier。

**计算步骤分解**（源码每个计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P0-a | `s_cum[0]=0; for e: s_cum[e+1]=s_cum[e]+(true_sizes[e]+block_m-1)//block_m`（producer thread0 串行前缀扫描，SMEM） | true_sizes | s_cum[E+1] | 专家→M-tile 段映射（调度，非数值语义） |
| P0-b | `s_total[0]=s_cum[E]*num_pid_n`；`T.sync_threads()` 广播 | s_cum | s_total | 有效 tile 总数 |
| P1-a | static-wave：`flat_id = 2*(sm_count*w + pid)`（pingpong）/ `sm_count*w + pid`（coop），`w ∈ [0,_max_waves)` | sm_count, pid, w | flat_id | tile 线性下标（**无 atomic**，确定性映射） |
| P1-b | `m_tile = flat_id // num_pid_n; n_tile = flat_id % num_pid_n`（coop 另加 TB swizzle `group_size_m`） | flat_id | m_tile, n_tile | tile 坐标 |
| P1-c | 死 tile 掩码 `flat_id < total`（越界 wave 跳过） | flat_id, total | 有效/无效 | 尾 wave 裁剪 |
| P1-d | 二分查找 `s_cum`（`log2_up` 步：`mid=(lo+hi)>>1; if s_cum[mid+1]<=m_tile: lo=mid+1 else hi=mid`） | m_tile, s_cum | expert_id | $e(t)=\max\{e:cum_e\le t\}$ |
| P1-e | `m_start=true_offsets[e]+(m_tile-s_cum[e])*block_m`；`n_start=n_tile*block_n` | 调度表, true_offsets | m_start, n_start | tile 在 A/C 中的行/列起点 |
| P1-f | `arows=min(block_m, true_sizes[e]-row)`；`acols=min(block_n, N-n_start)`（coop 劈半为 arows0/arows1） | true_sizes | 有效行/列 | 尾块边界 |
| P2-a | `T.barrier_wait(ab_empty[slot], phase)` → `T.tma_copy(A[...]→A_smem[slot])` + `T.tma_copy(B[e,...]→B_smem[slot])` → `T.barrier_arrive(ab_full[slot])` | GM A/B | SMEM A_smem/B_smem（ring） | 分块取数（异步） |
| P2-b | `T.barrier_wait(ab_full[slot], phase)` → `T.wgmma_gemm(A_smem[slot], B_smem[slot], C_local, transpose_B=True, clear_accum=(k==0))` → `T.wait_wgmma` → `T.warpgroup_fence_operand` → `T.barrier_arrive(ab_empty[slot])` | ring slot | C_local（fp32） | $C=\sum_k A\cdot B^T$ 的块累加 |
| P2-c | 1-deep WGMMA 软流水（coop）：`if k>0: wait_wgmma(1); barrier_arrive(ab_empty[ps])` | — | 槽位释放 | 重叠 WGMMA(k) 与下一次 barrier_wait |
| P3-a | `T.copy(C_local, C_local_cast)`（fragment fp32→dtype） | C_local | C_local_cast | 输出 dtype 回落 |
| P3-b | full tile 快路：`T.sync_threads(barrier_id=4/5, arrive_count=128)` → `T.copy(C_local_cast, C_shared)` → `T.fence_proxy_async()` → `T.sync_threads(...)` → `T.copy(C_shared, C[m_start,n_start])`（TMA store） | C_local_cast | C（GM） | 整块写回（合并访问） |
| P3-c | partial tile 慢路：`for i,j in T.Parallel(block_m,block_n): if i<arows and j<acols: C[m_start+i,n_start+j]=C_local_cast[i,j]` | C_local_cast | C（GM） | 尾块裁剪写出 |
| P4 | host：`sm_count←multi_processor_count`；校验 K/N 对齐与 device/contiguity/overlap；`C=torch.empty`（无 memset）；`lru_cache` 特化；顺序调用编译出的 kernel | — | C | host 侧装配 |

**数据流与内存访问模式**（源 GPU / Hopper 视角）：

```
GM[true_sizes] --producer thread0 串行扫描--> SMEM[s_cum(E+1)] ──T.sync_threads──> 全体 WG
                                                   │ 二分查找
                                                   ▼
                              flat_id(w,pid) ──> (m_tile,n_tile) ──> expert_id/row ──> m_start/n_start/arows/acols
GM[A] --TMA(cp.async.bulk)--> SMEM A_smem[num_stages, bm|half_m, bk] ┐
GM[B] --TMA(cp.async.bulk)--> SMEM B_smem[num_stages, bn, bk]        ├─ wgmma_gemm(transpose_B) --> register C_local (fp32)
   ring slot 由 gi%num_stages 选；producer/consumer 用 ab_full/ab_empty mbarrier 配对 ─┘        │
                                                                                                ▼
                                                            full: cast→C_shared→TMA store ┐
                                                            partial: 谓词 STG 逐元素 ──────┴──> GM[C]
```

**循环与并行结构**：grid = `T.Kernel(sm_count, threads=384)` 一维（**persistent**：每 CTA 跨多 wave 处理多
tile）；每 CTA：`for w in T.serial(_max_waves)` 外层 + `for k in T.Pipelined(_k_iters, num_stages=0)`
内层；CTA 内按 `tx` 分流为 1 producer WG + 2 math WG（warp specialization）。

**host 侧逻辑**：
1. `sm_count`：CUDA `multi_processor_count`（= persistent grid 大小）。
2. `_DEFAULT_CONFIG = {block_m:128, block_n:256, block_k:64, num_stages:3, threads:384, group_size_m:8}`；
   `autotune_configs` 按 H100 SMEM 上限（228KB）剪枝。
3. `forward`：K/N 对齐硬校验；`C` 分配（`torch.empty`，允许复用 out-param）；两处调用一个
   `@tilelang.jit(out_idx=[], pass_configs={TL_ENABLE_FAST_MATH, TL_DISABLE_THREAD_STORAGE_SYNC},
   compile_flags=["-O3","-DENABLE_BF16","-include",_ANCHOR_HELPER_PATH])` 工厂。

**编译配置**：`@tilelang.jit(out_idx=[], pass_configs={TL_ENABLE_FAST_MATH:True,
TL_DISABLE_THREAD_STORAGE_SYNC:True}, compile_flags=["-O3","-DENABLE_BF16","-include",_anchor_helper.h])`；
`A_smem/B_smem/C_shared` 用 `make_swizzled_layout`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM tiling（per-WG `num_stages` 深 ring） | 减少 GM 重复访问 | A/B tile 驻留 shared memory，被 K 循环复用 | shared memory + 容量 | 硬件强相关（NPU 有 L1 等价） |
| O2 | WGMMA tensor-core MMA（`T.wgmma_gemm`，`transpose_B=True`） | 矩阵算力 | Hopper 异步 warpgroup MMA + fp32 累加器 | tensor core（SM90 WGMMA） | 硬件强相关（NPU 有 Cube `T.gemm` 等价） |
| O3 | TMA 异步拷贝 + mbarrier ring + **手工 warp specialization**（1 producer + 2 consumer WG） | 访存/计算重叠、隐藏延迟 | producer 发 TMA，consumer 跑 WGMMA，per-WG ring + full/empty barrier | Hopper TMA + warpgroup + mbarrier | 硬件强相关（NPU 有 MTE2 异步，但**无 warp/warpgroup 概念**） |
| O4 | Swizzled SMEM 布局（`make_swizzled_layout`） | 消 SMEM bank conflict | 地址重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用，NZ 布局代替） |
| O5 | **static-wave scheduler**（`flat_id = sm_count*w + pid`，无 atomic counter） | 确定性 tile→CTA 映射、消除调度器延迟 | 编译期枚举 tile id + 运行时有效掩码 | 廉价 CTA 启动 + warp 内 thread0 串行扫描 | **算法层可移植**（确定性映射思想可保留）；**实现须外移**（§0.6 R1） |
| O6 | Threadblock swizzle（`group_size_m`，coop 模板） | B 权重 L2 复用、避免跨波驱逐 | 一个 group 的 M-tile 处理完全部 N 列 | L2 cache 几何 | 硬件强相关（NPU L2 可控性弱） |
| O7 | **双模板分派**（`block_m≤64` pingpong / `≥128` cooperative） | 按 SMEM 预算选结构 | block_m 决定 2 math WG 是否共享 B | Hopper SMEM 预算（228KB） | 硬件强相关（Hopper SMEM 预算产物；NPU L1 512KB 预算不同） |
| O8 | **split-A / shared-B**（coop 模板） | 省一份重复 B ring，容纳 `bn=256, ns≥3` | A 劈成 A_smem_top/bot 两块物理缓冲，两 WG 各读一块、共享 B | Hopper SMEM 容量 + WGMMA 零偏移约束 | 硬件强相关（Hopper SMEM 预算产物） |
| O9 | TMA store 快路 / 谓词 STG 慢路二分支 | 整块合并写、尾块精确写 | full tile 经 C_shared 走 TMA store；partial 走谓词逐元素 STG | Hopper TMA store + predicated STG | 算法层可移植（NPU 统一走动态 extent 写回） |
| O10 | 二分查找 O(log E) 定位专家 | 免每 CTA O(E) 专家扫描 | 单调 `s_cum` 上二分 | 无（纯算法） | 算法层可移植（但 NPU 须外移，§0.6 R1） |
| O11 | 寄存器重分配（`T.dec_max_nreg(24)` / `T.inc_max_nreg(240)`） | 给 consumer 更多寄存器装累加器 | `setmaxnreg` | 大寄存器堆 | 硬件强相关（NPU 无对应） |
| O12 | `T.warpgroup_fence_operand` | WGMMA 操作数可见性 | 寄存器 fence | Hopper WGMMA | 硬件强相关（NPU 无对应） |
| O13 | **persistent grid = sm_count**（每 CTA 处理多 tile） | 摊薄 CTA 启动、确定性映射 | `T.Kernel(sm_count)` + 每 CTA 多 wave | 廉价 CTA 启动 | 模式可移植（NPU 一维核需重定分核，§0.6 R4） |
| O14 | `lru_cache` 工厂 + 每 shape 编译 | 降 host 开销、shape 特化 | 缓存编译产物 | 无 | 可移植 |

> 识别不出来 ≠ 不存在。以上 O1–O14 为源码显式优化，逐项在 §0.5 判定，未被识别者会被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据映射表：`tilelang-op-design` skill `references/migration-analysis.md` §5.3；NPU 能力佐证：
`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`、
`docs/开发指南.md` §2/§3、可行底座先例 `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（按专家分组 NT 矩阵乘、fp32 累加、同 dtype 输出） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| I/O 契约（A/E,B/C 的 shape/dtype/行连续布局、K/N 对齐约束） | 语义 | 无 | — | **保留** | §0.2；K/N 对齐约束保留（源硬前置，non-aligned 走另一 kernel） |
| O1 SMEM tiling | 优化 | shared memory | 有（L1） | **等价替换** | GM→L1（Cube 操作数）分块搬运；L1 512KB / 32B 对齐（ascend-constraints.md §1） |
| O2 WGMMA tensor core | 优化 | Hopper Tensor Core | 有（Cube） | **等价替换** | `T.gemm`→`hivm.hir.mmadL1`（T.gemm.md §3）；分形 M/N≥16、K≥32；fp32 累加 |
| O3 TMA + mbarrier + warp specialization | 优化 | TMA / warpgroup / mbarrier | 有（MTE2 异步 + 框架同步），**无 warp** | **重新设计** | 单核 K 循环（`T.serial`，Stage 4 可试 `T.Pipelined`），MTE2 装载与 Cube 计算由框架/PIPE 同步重叠；**删除 producer/consumer 分流与手工 barrier**（§0.6 R2） |
| O4 swizzled layout | 优化 | SMEM bank 结构 | 无（NPU 无 SMEM bank conflict） | **舍弃** | NZ 分形布局由 `T.copy` 搬运隐式转换，手动 swizzle 无意义（migration-analysis.md §5.3） |
| O5 static-wave scheduler（内嵌） | 算法+优化 | thread0 串行扫描 + warp 并行 | 有（算法层），但 NPU Cube scope **无法维护调度状态** | **重新设计（外移为 GM 表）** | 调度表由前导 kernel 产出落 GM，本核读表（§0.6 R1）；**依据** CG-2026-0010（Developer persistent+gemm+整型索引段 lowering segfault / Expert Cube scope 内 loop-carry 崩溃）+ CG-2026-0027（标量消费向量产出不可同步） |
| O6 threadblock swizzle（`group_size_m`） | 优化 | 可编程 L2 | 部分（L2 存在但无显式编程接口） | **重新设计** | 改为「专家聚簇 + M-major」任务序，提升 B tile 时间局部性；无显式 L2 控制（§0.6 R5） |
| O7 双模板分派（pingpong/cooperative） | 优化/结构 | Hopper SMEM 预算（228KB） | 无等效约束（NPU L1 512KB，budget 不同） | **舍弃（合并为单模板）** | NPU 单一 Expert Cube 模板，`(block_m, block_n, block_k)` 由 L0C/L1 预算选（§0.6 R3） |
| O8 split-A / shared-B | 优化 | Hopper SMEM 容量 + WGMMA 零偏移 | 无等效收益 | **舍弃** | Hopper SMEM 预算产物；NPU 单模板不需要劈 A/共享 B，容量由 block 选择满足（§0.6 R3） |
| O9 TMA store / 谓词 STG 双路 | 优化 | TMA store + predicated STG | 有（动态 extent 写回） | **等价替换** | 统一 `T.store_fixpipe([tile_m,tile_n], enable_nz2nd=True)` 动态 extent 写回（合并两分支，§0.6 R6） |
| O10 二分查找 O(log E) | 算法 | 无 | 有（算法层），实现须外移 | **重新设计** | 表由前导 kernel（前缀扫描 + 逐专家区间展开）一次物化；本核 O(1) 读 `tile_expert_ids[bx]`（§0.6 R1） |
| O11 寄存器重分配（`dec/inc_max_nreg`） | 优化 | 大寄存器堆 setmaxnreg | 无 | **舍弃** | NPU 累加器落 L0C（128KB）而非寄存器堆 |
| O12 `warpgroup_fence_operand` | 优化 | Hopper WGMMA 寄存器 fence | 无 | **舍弃** | 随 O3/O11 一并去除；`T.gemm` 由框架管理依赖 |
| O13 persistent grid = sm_count | 优化/结构 | 廉价 CTA 启动 | 一维核超发被串行调度 | **重新设计** | NPU persistent grid = **AI core 数**（实查 20）；核内 `T.serial` 多 tile（§0.6 R4 / §5.5） |
| O14 `lru_cache` + shape 特化 | 优化（host） | 无 | 有 | **保留** | host 侧只做 shape/stride 元数据；每 shape 编译缓存 |
| TMA OOB 零填充（M 尾） | 优化 | Hopper TMA descriptor | 无 TMA，但 NPU 有动态 extent | **重新设计** | 统一走动态 extent 精确裁剪（§0.6 R7） |

> 每项处置依据见「NPU 对应方案 / 依据」列。舍弃项（O4 swizzle、O7 双模板、O8 split-A、O11 nreg、O12
> fence）理由已写明；**无静默丢弃**。判定输入含 Phase R 调研结论（§1.6.0）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0；选定算法族见该节）。
源为 GPU 单 kernel（CUDA/SM90，无 Developer/Expert 概念）；本单元为 **kernel-only**，
`kernel_map` 键 `moe_grouped_gemm_kernel`，端到端语义 = `C[e]=A[e]@B[e]^T`。

**重设计项 R0：编程模式——源 CUDA 单 kernel → Expert 单 Cube scope（迁移侧形态约束）**

- **背景**：源为 Hopper CUDA kernel，无 NPU 编程模式概念。用户指定 `programming_mode=developer`，
  但先例实测证明 Developer（persistent + `T.gemm` + 整型索引段）在 lowering 期 **segfault**
  且动态 extent `T.gemm` 静默错值（CG-2026-0010 第二证）。
- **NPU 新形态**：**Expert**（`T.Scope("Cube")`），核内只做 `T.alloc_L1`/`T.alloc_L0C` + `T.copy` 取数 +
  `T.gemm` + `store_fixpipe`；**单一 scope，无 Vector 段、无跨引擎同步**（规避 CG-2026-0026）。
- **语义保持论证**：编程模式不改变数值语义（同一 `T.gemm`、同一 fp32 累加、同一动态 extent 裁剪）。
- **依据**：先例正对照实测 max_diff 9.537e-07（`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`）；
  `.agents/evolution/capability-gaps.md` CG-2026-0010 / CG-2026-0026；pattern-library traps-runtime
  TRAP-DEVMODE-PERSIST-GEMM。

**重设计项 R1：tile 调度承载方式（O5/O10）——从「kernel 内嵌 static-wave 调度器」外移为「前导 kernel 产 GM 表 + 本核读表」**

- **源方案**：scheduler **内嵌**在本 kernel 内——producer WG 的 thread0 串行前缀扫描 `s_cum`（SMEM），
  全体 WG `T.sync_threads()` 广播；每 tile 用 `log2_up` 步二分查找落点；static-wave 枚举
  `flat_id = sm_count*w + pid`（无 atomic）。性能意图：**确定性 tile→CTA 映射（可复现）+ 消除每 CTA
  O(E) 专家扫描（O(log E) 二分）+ 零 CPU 同步**。
- **为何不能内嵌到 NPU Cube scope**：先例实测两条独立阻断——(i) Developer 单 Scope（persistent +
  `T.gemm` + 整型索引段）lowering 期 segfault（CG-2026-0010）；(ii) Expert Cube scope 内无法维护
  调度状态（loop-carry 崩溃 / 多地址 GM scalar store 被消除 / L1 scalar 读崩溃），且「标量消费向量
  scan 产出」在单 scope 内不同步（CG-2026-0027）；官方 `examples/mixcv/mixcv_mixkernel.py` 复现跨引擎
  同 block 交接不可编译（CG-2026-0026）。**故源的内嵌 static-wave 调度器在本工具链无可用落地路径。**
- **NPU 新算法（主选，两 kernel 数据流）**：
  1. **前导 kernel**（调度表产出者，**独立单元**，先例为 `_tile_scheduler_kernel`）读 `true_sizes[E]`，
     产出 3 张 **GM int32 表**：`tile_expert_ids[max_tiles]`（每 M-tile 专家号，死 tile = -1）、
     `tile_row_offsets[max_tiles]`（专家内行偏移，已乘 `block_m`）、`total_tiles[1]`（有效 M-tile 数）；
  2. **本核**（Expert Cube-only）按 `bx = tile_id // num_pid_n`、`by = tile_id % num_pid_n` 读
     `total_tiles[0]` 与 `tile_expert_ids[bx]`/`tile_row_offsets[bx]`（**GM 标量读**，O(1) 寻址），
     算 `m_start/n_start/tile_m/tile_n` 后执行 NT GEMM；
  3. 两 kernel 在**同一 stream 顺序 launch**：前导 kernel 先完成、本核再执行，顺序由 kernel 间自然
     顺序保证，**无需任何 in-block 同步、无 CPU 同步**（`total_tiles` 全程在设备侧）。
- **语义保持论证**：调度表内容与源**逐元素一致**（同一 `cum` 定义、同一 `searchsorted` 落点、同一
  哨兵 `(-1,0)`），故 tile→(expert,row) 映射逐位一致，GEMM 数值与源一致；变化仅在**求值时机**
  （前导 kernel 一次物化 vs 源核内重算）。边界：$s_e=0$ 的专家 `cnt=0`，不产生 tile；`total_tiles ≤ max_tiles`
  保证 `tile_expert_ids` 长度合法；死 tile 由单比较 `if bx < total_tiles[0]` 跳过。
- **确定性映射承接**：源 static-wave 的「无 atomic、确定性 tile→CTA」意图由「表内容确定性 + 核按块连续
  领取 task 区间」承接（§0.6 R5）。
- **对接契约（本核消费）**：见 §4.1；`max_tiles = numel // block_m + E` 与 `block_m` 为两 kernel 共享的
  编译期常量；表 dtype 全 int32；`tile_row_offsets` 单位是**行**（已乘 block_m），不是 tile 序号。
- **先验证据**：D-3 探针（`history_version/design_probe_persistent_gemm.py`）用 host-built 表 fixture
  实测本核该形态 4 用例全 PASS（worst max_diff 7.629e-06）；先例 Stage 3 正对照 max_diff 9.537e-07。

**重设计项 R2：warp specialization + TMA + mbarrier（O3）→ 单核 K 循环 + 框架管理 MTE2↔Cube 重叠**

- **源方案**：1 producer WG（tx<128）用 TMA 发 A/B，2 consumer WG 用 wgmma；`T.dec/inc_max_nreg` 分配
  寄存器；per-slot full/empty mbarrier 手工配对；coop 模板 1-deep WGMMA 软流水。
- **NPU 新算法**：单一种类核（Cube），`for k in T.serial(ceildiv(K, block_k))`（或 Stage 4 试
  `T.Pipelined(..., num_stages)`）组织 K 循环：`T.copy` 把 A/B tile 装入 L1，`T.gemm` 累加到 L0C，
  由框架/PIPE 同步重叠 MTE2 装载与 Cube 计算；删除 producer/consumer 分流、手工 barrier、
  `dec/inc_max_nreg`、`warpgroup_fence_operand`、1-deep WGMMA 软流水（由框架多缓冲替代）。
- **语义保持论证**：仅改变搬运/计算调度方式，不改变 K 的累加顺序（仍为 k=0… 的顺序 tile 累加）
  与 fp32 累加域，数学等价；性能意图（搬运-计算重叠）由 `num_stages`/框架承接。

**重设计项 R3：双模板（O7）+ split-A/shared-B（O8）→ 单一 Expert Cube 模板 + 容量驱动 block 选择**

- **源方案**：`block_m ≤ 64` pingpong（2 WG 各自独立 tile）/ `block_m ≥ 128` cooperative（2 WG 劈半 M、
  shared-B）；后者把 A 劈成 A_smem_top/bot 两块物理缓冲（为满足 WGMMA `A_offset[-2]==0` 约束），省一份
  重复 B ring 以容纳 `bn=256, ns≥3`。
- **NPU 新算法**：**单一模板**，`(block_m, block_n, block_k)` 由 L0C/L1 预算选定（§5.2/§5.3）：
  `block_m × block_n × 4 ≤ 128KB`（L0C）、`(block_m + block_n) × block_k × elem × num_stages ≤ 512KB`（L1）。
  取 `block_m=64, block_n=256, block_k=64`（L0C 64KB、L1 80KB@2stages），与先例一致。
- **语义保持论证**：模板分派与 split-A 只影响片上驻留组织（不改变 M/N/K 的输出映射与累加分组），
  数值逐位一致；block 选择同样只改 tiling 粒度。
- **依据**：源模板分派/劈 A 是 Hopper SMEM 预算（228KB）产物；NPU L1 512KB 预算不同，单模板即可满足
  （先例实测 max_diff 9.537e-07）。

**重设计项 R4：persistent grid（O13）——`sm_count` → NPU AI core 数（实查 20）+ 核内 `T.serial`**

- **源方案**：`T.Kernel(sm_count, threads=384)`，每 CTA 跨多 wave 处理多 tile；static-wave 枚举。
- **NPU 新算法**：`T.Kernel(nums_kernel, is_npu=True)`，`nums_kernel = min(max_tiles*num_pid_n, P)`
  （P = 实查 AI core 数，§5.5）；核内 `for t in T.serial(num_local_tasks)`（`num_local_tasks` 编译期常量）
  以块连续方式处理多逻辑 tile；死 tile 用 `if bx < total_tiles[0]` 单比较跳过（位于 `T.serial` 体内、
  **不在 `T.Pipelined` 体内**，规避 CG-2026-0001 的条件构造崩溃）。
- **`sm_count` 的 NPU 语义（明确）**：源的 `sm_count` = persistent grid 的 CTA 数（Hopper SM 数）；
  NPU 无 SM 概念，**persistent grid 映射为 AI core 数**——`nums_kernel` 由
  `NPUUtils.get().get_aicore_num()` 实查值（20）界定（Cube 算子直接用返回值，不翻倍）。
- **语义保持论证**：分核只改变「哪个核处理哪个 tile」的调度，不改变 tile 内计算（同一 `T.gemm` +
  同一 fp32 累加 + 同一裁剪写回），数值逐位一致（累加分组不变）；边界：`nums_kernel`/`num_local_tasks`
  均为编译期常量，循环边界静态。
- **D-3 探针证据**：`design_probe_persistent_gemm.py` 的 `persistent-P` 计划（**启动恰好 20 核**，
  核内 `num_local_tasks` 串行）4 用例全 PASS——含 `num_pid_n>1` 且 `num_local_tasks>1`（P-3：cores=20,
  local_tasks=3, max_diff 7.629e-06）。

**重设计项 R5：任务序（O6）——TB swizzle（`group_size_m`）→ 专家聚簇 + M-major**

- **源方案**：coop 模板对 m-tile 做 `group_size_m` 分组重排，令一个 group 的 M-tile 处理完全部 N 列，
  使 B 权重工作集落在 L2、避免普通 2D grid 的跨波驱逐（prefill 下约 +5%）。
- **NPU 新算法**：任务序采用**专家聚簇**（同专家相邻，由调度表天然给出：`tile_expert_ids` 按专家前缀段
  序排列）+ **M-major**（同 M-tile 先走完全部 N 列）：flat task_id → `bx = task_id // num_pid_n`、
  `by = task_id % num_pid_n`；核内按块连续领取 task 区间，使同一核的连续任务尽量同专家、同 M-tile。
- **语义保持论证**：纯任务排序，不改变任何 tile 的数值结果；累加分组与 dtype 路径不变。边界：排序对死
  tile 判定无影响。**依据缺口**：NPU 无显式 L2 编程接口、L2 容量/替换策略未在 docs/ 记载（标注
  「未文档化假设」：收益来自访存时间局部性，实测收益待 Stage 4 验证；见 §9 风险点 R-4）。

**重设计项 R6：epilogue 双路（O9）→ 单路动态 extent `store_fixpipe`**

- **源方案**：full tile 走「cast → C_shared → `sync_threads`(named barrier) → `fence_proxy_async` →
  TMA store」快路；partial tile 走「`T.Parallel` 谓词 STG」慢路。
- **NPU 新算法**：统一走 `with T.rs("PIPE_FIX"): T.store_fixpipe(c_l0c, C[m_start, n_start],
  size=[tile_m, tile_n], enable_nz2nd=True)`（动态 `[tile_m, tile_n]` 精确裁剪，f32→f16/bf16 由 fixpipe
  量化）；删除 C_shared staging、named barrier、`fence_proxy_async`。
- **语义保持论证**：两路的输出均为「同一 fp32 累加器一次舍入到目标 dtype」；动态 extent 只写有效
  `[tile_m, tile_n]`，**杜绝越界写坏下一专家行**（no-pad 关键正确性约束），等价源守卫。边界：整除时
  `tile_m=block_m, tile_n=block_n` 退化为整块写。
- **依据**：`docs/Tilelang.language/内存操作/T.store_fixpipe.md`；先例 `store_fixpipe` 动态 extent 实测
  max_diff 9.537e-07。

**重设计项 R7：尾块处理——TMA OOB 零填充（M 尾）+ epilogue 守卫 → 动态 extent 精确裁剪**

- **源方案**：A descriptor `globalDim=numel` 使 TMA OOB 读自动零填充；epilogue 用守卫裁剪。
- **NPU 新算法**：所有 GM→L1 取数与 L0C→GM 写回使用**动态 extent**：
  `tile_m = T.min(block_m, s_e - row)`、`tile_n = T.min(block_n, N - n_start)`、
  `tile_k = T.min(block_k, K - k*block_k)`；`T.copy` 切片 extents 用动态值；**K 方向的动态 extent 必须
  同时传入 `T.gemm(..., size=[tile_m, tile_k, tile_n])` 限定归约范围**（否则缓冲 `[tile_k, block_k)` 的
  陈旧值会被按静态 `block_k` 归约计入累加——§1.6.1 M1 的 `stale_tail_guard` 证实发散 ≈1.218e3）。
  M/N/K 三方向尾块统一由动态 extent 处理，**不再需要 TMA 零填充、谓词 STG、epilogue 守卫**。
- **语义保持论证**：越界贡献在数学上恒为 0（零填充或被裁掉的积项），故「只算有效区」与「零填充后全算
  再裁剪」数值等价，且该等价性以「归约范围 = 有效 `tile_k`」为前提（由 `size` 落实，§1.6.1 M1 机器验证
  fp64 0 ulp，4/4 用例 EQUIV_PASS；guard 确认缺 `size` 发散）。**no-pad 关键边界**：尾块多余行属于
  下一专家，动态 extent 只写 `[0,tile_m)` 行，杜绝越界写坏邻专家。dtype：取数/写回 dtype 一致，
  `store_fixpipe` 走 fp32→fp16/bf16 量化，与源一致。
- **说明**：源 K 对齐（`K % block_k == 0`），K 尾动态 extent 属**防御性**（对齐时恒 `tile_k=block_k`）；
  M/N 尾为普遍情形（numel/每专家 $s_e$ 一般不整除、smoke N=128<block_n=256）。

**重设计项 R8：无（算法层）**。矩阵乘算法族本身（NT GEMM + fp32 分块累加）在 Cube 上有等价能力，
按 §0.5 为「保留/等价替换」，无算法族替换（§1.6.0 结论）。

### 0.7 标杆实现

- 源算子路径：`/tmp/opencode/TileOPs-ref/tileops/kernels/grouped_gemm/grouped_gemm_persistent_3wg.py`
  （函数 `_persistent_grouped_gemm_v2_kernel` / `_make_pingpong_kernel` / `_make_cooperative_kernel` /
  类 `GroupedGemmPersistent3WGKernel`）。
- Stage 0 抽取件（参考，非实现基线）：
  `examples/TileOPs/tileops/kernels/moe/grouped_gemm_persistent_3wg/_grouped_gemm_persistent_3wg_kernels.py`；
  NPU-adapted 包装：同目录 `grouped_gemm_persistent_3wg.py`。
- 源规格 / 测试 / bench / manifest（无独立 op 条目，借用等价 op 的规格）：
  `examples/TileOPs/tileops/manifest/moe.yaml` 的 `MoeGroupedGemmNopadFwdOp.workloads`、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` 的 `_ref_grouped_gemm_nopad`
  （每专家 PyTorch NT matmul，fp32 累加后回落输入 dtype）、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad.py`。
- 本仓 NPU 可行底座先例（结构参照 + 实测）：
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`（v2）与实现
  `_moe_grouped_gemm_kernel.py`（Expert 单 Cube scope + 外部调度表 + 动态 extent + `store_fixpipe`）；
  调度表产出者 `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`。
- golden 以 §0.1 语义为唯一依据，优先移植上述源 op 测试参考实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_persistent_grouped_gemm_v2_kernel`（MoE 3-WG persistent 分组 GEMM，NT 布局，`target="npuir"`，
**kernel-only** —— 作为 3 个 MoE fusion 算子 `kernel_map` 成员，键 `moe_grouped_gemm_kernel`）。

### 1.2 功能描述

对紧致（无 pad）排列的 MoE 激活按专家分组做 NT 矩阵乘：第 $e$ 个专家的 $s_e$ 行激活乘以其权重矩阵的
转置，输出与输入同形状、同 dtype。端到端语义与已迁移的 `MoeGroupedGemmNopadFwdOp` 等价
（`C[e] = A[e] @ B[e]^T`，K 对齐）。迁移后为 NPU 侧 **Expert 单 Cube scope** 实现：核内**读 GM 调度表 +
动态 extent NT GEMM（`T.gemm(b_transpose=True, size=)`）+ `store_fixpipe` 写回**；tile 调度（源内嵌
static-wave 调度器）外移为**前导 kernel** 产出的 GM 表（§0.6 R1，理由 = 工具链阻断 Cube scope 内维护
调度状态 / 跨引擎同 block 交接）。

> **设计核心 / GPU→NPU 关键重设计（可提炼为 3 条）**：
> 1. **内嵌 static-wave 调度器 → 外移为 GM 调度表**（§0.6 R1）：源的 kernel 内前缀扫描 + 二分查找 +
>    无 atomic 确定性映射在 NPU Cube scope 内不可维护（CG-2026-0010 / CG-2026-0027），改为前导 kernel
>    产出 `tile_expert_ids`/`tile_row_offsets`/`total_tiles` 三张 GM int32 表、本核 O(1) 读表；
>    「确定性 tile→CTA 映射 + 零 CPU 同步」意图由表 + 同 stream 顺序 launch 承接。
> 2. **Hopper warp specialization + TMA + mbarrier + 双模板（含 split-A/shared-B）→ 单核 Cube 单模板**
>    （§0.6 R2/R3）：删除 producer/consumer WG 分流、手工 barrier、`dec/inc_max_nreg`、
>    `warpgroup_fence_operand`、pingpong/cooperative 分派与 split-A，改单核 K 循环 + 框架管理
>    MTE2↔Cube 重叠，block 由 L0C/L1 容量选。
> 3. **`sm_count` persistent grid → NPU AI core 数（实查 20）+ 核内 `T.serial` 多 tile；统一动态 extent
>    裁剪 + 单路 `store_fixpipe`**（§0.6 R4/R6/R7）：源 TMA OOB 零填充 + epilogue 守卫（full/partial
>    双路）合并为动态 extent 精确裁剪写回，M-tile 绝不越界写坏下一专家。

### 1.3 数学公式

$$
C[o_e + r,\; :] = A[o_e + r,\; :]\, B[e]^{T},\quad r\in[0,s_e),\quad e\in[0,E),\quad \sum_e s_e=\text{numel}
$$

即 $C[i,n]=\sum_{k=0}^{K-1} A[i,k]\, B[e_i,n,k]$，$e_i$ 为第 $i$ 行专家号；等价于逐专家 $C_e = A_e B_e^{T}$
的拼接。源清单 roofline（`examples/TileOPs/tileops/manifest/moe.yaml`，等价 op）：`flops = 2*numel*n*k`，
`bytes = (numel*k + E*n*k + numel*n)*elem_bytes`。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源算法结构的差异已逐条注明）。

**两 kernel（前导调度 kernel + 本 GEMM 核；源为单 kernel 内嵌调度）**：前导 kernel（独立单元，先例
`_tile_scheduler_kernel`）产出调度表（GM），本 GEMM 核（**Expert Cube-only**，单一 `T.Scope("Cube")`）
读表执行 NT 分组 GEMM：

1. **调度表消费（Cube scope，标量读 GM）**：核内对每个逻辑 tile（`tile_id = core_id*num_local_tasks + t`）
   算 `bx = tile_id // num_pid_n`、`by = tile_id % num_pid_n`；读 `total_tiles[0]`，`bx >= total_tiles[0]`
   的死 tile 跳过；否则读 `expert_id = tile_expert_ids[bx]`、`row = tile_row_offsets[bx]`（GM 标量读）；
   由 `true_offsets[expert_id]`、`true_sizes[expert_id]` 算 `m_start/tile_m`。
   **与源结构差异**：源为**内核内** static-wave 调度（thread0 串行前缀扫描 + 二分查找 + 无 atomic 枚举）
   + persistent grid=`sm_count`；本设计为**前导 kernel 产表 + 固定核数（=AI core 数）+ 核内 `T.serial`**
   （§0.6 R1/R4）。
2. **GEMM 阶段（Cube scope）**：据调度元数据算 `n_start` 与动态 extent `tile_n/tile_k`，GM→L1 取 A/B tile
   （NZ 自动转换），`T.gemm(a_l1, b_l1, c_l0c, size=[tile_m,tile_k,tile_n], b_transpose=True, initC=(k==0))`
   在 K 维累加（fp32，**归约范围由 `size` 限制为动态 `tile_k`**，见 §3.3），最后 `store_fixpipe`
   （动态 `[tile_m,tile_n]`）写回 C；与源结构差异：**单核 K 循环**替代 1 producer + 2 consumer WG 的
   warp specialization（§0.6 R2），**单一模板**替代 pingpong/cooperative 双模板 + split-A（§0.6 R3），
   **动态 extent 精确裁剪**替代源 TMA OOB 零填充 + epilogue full/partial 双路守卫（§0.6 R6/R7），
   **专家聚簇任务序**替代 TB swizzle `group_size_m`（§0.6 R5）。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[前导 kernel（独立单元，先例 _tile_scheduler_kernel，整数索引）]
  true_sizes[E] --前缀扫描/逐专家区间展开--> GM[tile_expert_ids(max_tiles) int32]
                                             GM[tile_row_offsets(max_tiles) int32]
                                             GM[total_tiles(1) int32]
        │  （同一 stream 顺序 launch：前导 kernel 完成后 GEMM 核执行，无 in-block 同步）
        ▼
[本 GEMM 核，Expert Cube-only]
  标量读 GM 表/标量 ──> (expert_id, row, m_start, n_start, tile_m, tile_n)
A[numel,K] ┐
B[E,N,K]   ┼─动态extent T.copy(GM→L1, ND2NZ)─> L1[a_l1/b_l1] ─T.gemm(b_transpose,size=[tile_m,tile_k,tile_n])─> L0C(fp32)
           │                                                                         │ store_fixpipe([tile_m,tile_n])
           └── (由 m_start/n_start/true_sizes 决定取哪一块)                             ▼
                                                                                   C[numel,N]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（MoE no-pad 分组 NT 矩阵乘 + 其 tile 调度）的**算法族**。输入/源算法
> （单 kernel 内嵌 static-wave 调度 + Hopper warp-specialization + TMA + 双模板）只是基线候选之一。
> 深度分级：矩阵/多步/融合类 → 完整调研。
> **可实现性约束（前置）**：本工具链禁止跨引擎同 block 交接（CG-2026-0026，官方
> `examples/mixcv/mixcv_mixkernel.py` 复现），且 Developer persistent + `T.gemm` + 整型索引段共存即
> lowering segfault（CG-2026-0010）、Cube scope 内无法维护调度状态（loop-carry 崩溃）、单 scope 内
> 「标量消费向量 scan 产出」不同步（CG-2026-0027）。故任何「调度内嵌进 Cube scope」的候选**不可实现**，
> 调研在**可实现族**内择优（可行性前置于经济性）。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | 单 kernel：内嵌 static-wave 调度（前缀扫描 + 二分）+ Hopper 3-WG warp-spec + TMA + 双模板（pingpong/cooperative，split-A）；超发 `sm_count` persistent grid | — | — | ✅ 基线 |
| C1 | **外部调度表 + Expert Cube-only GEMM 读表 + 动态 extent + 专家聚簇**（选定） | 前导 kernel 产 3 张 GM 表；GEMM 核 Expert Cube-only 读表；动态 extent；固定核数=AI core 数 + 核内串行；单模板 | 数学恒等（调度内容同源；表内容逐位一致） | 消 TMA OOB 零填充/谓词 STG 双路；无跨引擎同步（规避 CG-2026-0026/0010/0027）；消双模板 | ✅ **选定** |
| C2 | 内嵌调度进 Cube scope（源形态直译） | kernel 内 thread0 前缀扫描 + 二分查找 + static-wave 枚举 | 数学恒等（纯索引） | 理论少 1 launch + 少 3 表 GM 往返 | ❌ **不可实现**：CG-2026-0010（lowering segfault）/ CG-2026-0027（标量消费向量产出） |
| C3 | 动态 extent 精确裁剪（含动态 K 归约范围） | TMA OOB 零填充 + full/partial 守卫 → 只取/只写有效区 `tile_m/tile_n/tile_k`，`T.gemm(size=)` | 容差内等价（越界项恒 0；机器验证 §1.6.1 M1） | 少搬越界冗余 + 消零填充 + 消双路 epilogue | ✅ 采纳（并入 C1） |
| C4 | NT 形式（`b_transpose=True`） | 直接以 B[N,K] 行主序做 `A·B^T`，不物化 B^T | 数学恒等 | **源已用**（`transpose_B=True`）；B 是全仓最大张量，免一次转置 | ✅ 保留（并入 C1，§1.6.1 M3） |
| C5 | 无搜索映射（`expert = task_id // max_tiles_per_expert`） | 网格直接编码专家，免查找表 | 数学恒等 | 省表，但死 tile 比例暴增 | ✅ 对比（弃选） |
| C6 | 专家内 pad 到 block_m 整数倍 | 先 pad A 再算，末尾裁剪 | 数学恒等（补 0） | 消除 M 尾块特判，但 FLOPs/访存膨胀 | ✅ 对比（弃选） |
| C7 | split-K（K 分段多核累加） | K 维切分给多核，跨核累加 | 容差内等价（累加分组变化） | 增并行度，但引入跨核归并 + 访存倍增 | ✅ 对比（弃选） |
| C8 | host 侧调度（CPU 算表后下发） | 调度表在 host 生成 | 数学等价 | 需 `true_sizes` D2H + 表 H2D | ❌ 弃选：引入 CPU 同步，违反源「零 CPU 同步」意图 |

**R2 在线算法**：**有**，且是本题的天然形态。矩阵乘在 K 维的分块累加**本身就是单遍流式**（running
累加器 `c_l0c`，O(1) 状态）；分组结构（每专家独立 NT 矩阵乘）之间无跨专家依赖。结构判据：输出（C）不含
跨 tile 的全局依赖，M/N 两维可任意分块并行，K 维是唯一规约维且可单遍累加 —— 因此**不存在需要多遍物化
的候选**（不像 softmax/attention 那样需要两遍 max/sum）。源算法已用「分块 K 累加」即直接证据；源的
static-wave 调度器（前缀扫描，单遍 running sum）也属调度层的在线形态。收益口径：扫描遍数 = 1
（A 一遍、B 一遍、C 一遍，见 R3）；中间缓冲 O(1)（仅 L0C tile + 前导 kernel 的小表）。

**R3 复杂度对比**（四口径；代表 shape = deepseek-v3-prefill-gate-up：numel=32768，E=256，N=4096，K=7168，
bf16；elem=2B。"访存"含 A 读 + B 读 + C 写 + 调度表 GM 往返，不含 L1/L2 内部重复；口径见脚注）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| C0 基线（单 kernel 内嵌调度，Hopper 3-WG，TMA 零填充 + 双路 epilogue） | 1.92e12 | A 0.47G + **B 15.03G（TB swizzle 目标 L2 复用，理想 1 遍；普通 2D grid 可能 2× ≈ 30G）** + C 0.27G ≈ **15.77G** | B 1–2 遍；A 1 遍（每 tile 重读，靠 L2）；C 1 遍 | L0C 64×256×4=64KB；SMEM（A+B）×num_stages | 高（数千 CTA）；无跨核归并 |
| C1 **选定（外部调度表 + Expert Cube 读表 + 动态 extent + 专家聚簇 + 单模板）** | 1.92e12 | A 0.47G + B **15.03G（专家聚簇，目标 1 遍）** + C 0.27G + 表 ≈0.006G ≈ **15.77G** | A 1 遍 + B 1 遍 + C 1 遍 | L0C 64KB；L1 (8K+32K)×stages | 高（逻辑 tile 数）；核内串行，无跨核归并；**无跨引擎同步** |
| C2 内嵌调度（源形态直译，工具链不可实现） | 1.92e12 | 同 C1 少 0.006G（表不落 GM） | 同 C1 | 同 C1 + 索引状态 | 工具链**不可实现**（CG-2026-0010/0027）；弃选 |
| C5 无搜索映射 | 1.92e12 | 同 C1 | 同 C1 | 同 C1 + 免调度表 | 逻辑 tile 数 = E×max_tiles_per_expert×num_pid_n（**死 tile 比例暴增**），守卫浪费大 |
| C6 专家内 pad 到 block_m | 2×numel'*N*K（numel' 为 pad 后行数） | A 膨胀 ≈ +E×block_m×K×2；**B 重读次数随 pad 后 m-tile 数上升** | ≥1 遍且更多 | 同 C1 | 同 C1，但有效计算率下降 |
| C7 split-K（K 切 nsplit 段） | 1.92e12 | A 读 ×nsplit + B 读 ×nsplit + partial 写/读 ×nsplit | ≥nsplit 遍 | partial 缓冲 [numel,N]×nsplit×4B（落 GM） | 并行度 ×nsplit；**需跨核归并**，归并代价 ≥ 省下的时延 |
| C8 host 调度 | 同 C0 | true_sizes D2H + 表 H2D | true_sizes 1 遍 + 2 次总线往返 | host 内存 | **CPU 同步阻塞**（违反零同步） |

口径脚注：FLOPs 只算乘加（2/element）；访存 = GM 侧不可消除的输入读 + 输出写 + 必要中间往返（不含
L1/L2 复用命中，不含系数项）；扫描遍数指同一数据在 GM 侧被读的次数；中间缓冲峰值以片上（UB/L1/L0C）为准。
**主导项为 B 权重访存（15.03 GB，占 95%+），算子是访存受限**，故 R3 排序以 B 遍数/复用为主导，而非 FLOPs。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰按 negative-claim-evidence.md §2 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Cube ✅；但 warp-spec/producer-consumer 在 NPU 无对应 | L0C 64KB ✅ / L1 ✅ | 源 TMA 零填充处理 M；N/K 对齐 | grid=sm_count 编译期常量 ✅ | ❌ NPU 无 warpgroup/mbarrier | ❌ 源形态**不可编译**（warp-spec/TMA 无对应；内嵌调度 CG-2026-0010/0027） | ⚠️ 需重设计 |
| C1 **选定** | Cube ✅（Expert Cube-only）+ Vector（前导 kernel 独立执行）✅ | L0C 64KB ✅ / L1 80KB@2stage ✅ | 动态 extent 全方向 ✅ | 循环边界全静态（nums_kernel/num_local_tasks 编译期常量）✅ | Pipelined ✅；**无跨引擎同步**（单 Cube scope）✅ | ✅ **D-3 探针设备实测可编译 + 正确**（4 用例 max_diff ≤7.6e-06）；不触发 CG-2026-0010/0026/0027 | ✅ **选定：可行且最亲和** |
| C2 内嵌调度 | 需 Cube scope 内维护调度状态 | 同 C1 | 同 C1 | ✅ | ❌ | ❌ **不可用**：CG-2026-0010（lowering segfault）+ CG-2026-0027（标量消费向量产出） | ❌ 弃选 |
| C5 无搜索映射 | Cube ✅ | 同 C1 | 同 C1 | ✅ | ✅ | ✅ | ❌ 弃选：死 tile 暴增，守卫与空转成本高（R3） |
| C6 专家内 pad | Cube ✅ | A 缓冲/流量膨胀 | pad 后对齐 ✓ | ✅ | ✅ | ✅ | ❌ 弃选：FLOPs 与 A 访存膨胀 |
| C7 split-K | Cube ✅ | partial 缓冲超 LB 需落 GM | ✅ | ✅ | 跨核归并需 sync_block 或两阶段 kernel | ⚠️ | ❌ 弃选：算子是 B 访存受限，split-K 使 A/B 访存 ×nsplit，且引入归并代价 |
| C8 host 调度 | CPU | — | — | — | ❌ 引入 host-device 往返与 CPU 同步 | ✅（可行但违约束） | ❌ 弃选：违反零 CPU 同步 |

**调研结论**：选定 **C1 —— 前导 kernel 独立产出调度表（GM）+ 本 GEMM 核 Expert Cube-only 读表 +
动态 extent 精确裁剪 + 专家聚簇任务序 + 固定核数（=AI core 数）核内串行 + 单模板**。关键依据：
(i) **可实现性是前置约束** —— 本工具链禁止跨引擎同 block 交接（CG-2026-0026）、Developer persistent +
`T.gemm` + 整型索引段 segfault（CG-2026-0010）、Cube scope 内无法维护调度状态（loop-carry 崩溃）、单
scope 标量消费向量产出不同步（CG-2026-0027），故 C0 源形态直译与 C2 内嵌调度均不可实现，C1 是本工具链
唯一**可实现**形态（D-3 探针实测 + 先例 Stage 3 max_diff 9.537e-07）；
(ii) R3 显示算子是 **B 访存受限**（B 15.03 GB 占 95%+），C1 以「专家聚簇 + M-major」把 B 保持在单遍为
目标、A/C 各单遍，且无跨核归并；C1 相对 C2 多出的 launch + 表 GM 往返（≈6 KB）+ 跨 kernel barrier 合计
≪ B 访存的 1%，代价可忽略，换来可实现性；
(iii) R4 显示 C1 全方向满足 NPU 约束（无 host 改动、循环边界静态、L0C/L1 容量内、可流水、**无跨引擎
同步**），而 C2 因工具链不可实现、C5/C6/C7/C8 分别因死 tile 爆炸 / FLOPs 与访存膨胀 / 访存倍增+归并代价 /
CPU 同步被弃选。

**与基线结构差异**：把「内核内 static-wave 调度」替换为「前导 kernel 产表 + 读表」；把「Hopper 3-WG
warp specialization + TMA + 手动 barrier + 双模板（split-A/shared-B）」替换为「单核 Cube 单模板 + 框架管理
MTE2↔Cube 重叠」；把「`sm_count` persistent grid」替换为「固定核数 = AI core 数 + 核内串行」；把「TMA OOB
零填充 + full/partial 双路 epilogue」替换为「动态 extent 精确裁剪 + 单路 `store_fixpipe`」；把「TB swizzle
group_size_m」替换为「专家聚簇任务序」。**保留**：NT 形式（`b_transpose`）、fp32 分块累加、K/N 对齐约束、
零 CPU 同步意图、确定性 tile→(expert,row) 映射（由表内容确定性承接）。

**调研范围**：`references/algorithm-candidates.md`（无 MoE grouped-GEMM 专条；参考 ALG-gemv 的「Cube
分形 vs Vector 规约」与 split-K 条目）、pattern-library（PL-1.8 bf16 Cube 直连、PL-1.16 Expert 双 Scope
流水形态、CONST-* 硬件常数、traps-runtime TRAP-DEVMODE-PERSIST-GEMM）、源算子 §0.3/§0.4、本仓 examples
（`examples/gemm/`、`examples/mixcv/`、`examples/deepseek_v32/fp8_lighting_indexer.py`、
`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/`（Stage 3 实测底座）、
`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/`（调度表产出者））、
`.agents/evolution/capability-gaps.md`（CG-2026-0010/0026/0027）。未使用互联网检索（本地源已覆盖，
无新候选）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1→L1 分块（动态 extent 取数）；O2（WGMMA）→Cube
`T.gemm`（`b_transpose`）；O3（TMA+warp-spec+mbarrier）→**重设计**单核 `T.serial` K 循环 + 框架同步
（承接「搬运-计算重叠」意图：`num_stages`/框架多缓冲）；O4（swizzle）**舍弃**（NPU NZ 由搬运转换）；O5（static-wave
调度）→**R1 外移为 GM 表**（承接「确定性 tile→CTA 映射」意图：表内容确定性 + 块连续领取）；O6（TB swizzle）
→R5 专家聚簇 + M-major（承接「B 时间局部性」意图）；O7（双模板）→R3 **舍弃合并单模板**（Hopper SMEM
预算产物）；O8（split-A/shared-B）→R3 **舍弃**（同因）；O9（TMA-store/STG 双路）→R6 单路动态 extent
`store_fixpipe`（承接「整块合并写」意图）；O10（二分查找）→R1 表 O(1) 读（承接「消每 CTA O(E) 扫描」
意图）；O11/O12（nreg/fence）**舍弃**（NPU 无对应）；O13（persistent grid=sm_count）→R4 固定核数=AI core
数；O14（`lru_cache`）保留。**零 CPU 同步**：承接（`total_tiles` 全程设备侧）。

**设计期 roofline（D-2）**：以 §1.6.3 选定布局 + pattern-library/constants.md 实测常数估算下界：
- **流量项（GEMM 核，主导）**：GM 总流量 ≈ A 0.47 GB + B 15.03 GB + C 0.27 GB = **15.77 GB** ÷
  `CONST-mte2-degradation` 混合流量地板 **~1.26 TB/s**（工具链 tilelang 0.1.2+ed787bb / CANN 8.5.0）
  ≈ **12.5 ms**（其中 B 单遍 15.03 GB ≈ **11.9 ms** 为绝对主导）；
- **流量项（前导 kernel）**：`true_sizes` 读 1 KB + 3 张表写 ≈6.1 KB ≈ **< 0.1 µs**（可忽略）；
- **发射项**：前导 kernel 一次 launch + 约 4 个向量 op（`CONST-vector-launch-overhead` ~0.5µs/op）+ 一次
  跨 kernel barrier，合计 **< 0.1 ms**（一次性固定开销）；
- **容量项**：L0C 64KB / L1 80KB@2stage，均 < 上限（`CONST-capacity-910B2C`：L1 512KB / L0C 128KB）；
  本核无 UB；
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 12.5 ms**（B 访存主导）。
  常数未覆盖的量（NPU L2 对 B 的复用率、动态 extent 搬运开销、跨 kernel barrier 的精确墙钟）显式标注
  「未实测假设 + 估算依据」：按 C1 的专家聚簇目标 B 单遍；若 L2 复用失败则 B 可能 2 遍（≈ 24.4 ms）。
  实测由 Stage 4 校核。注：常数绑定 Ascend910B2C，工具链变更须按 kb_stale_check 重验。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1 算法（调度表由前导 kernel 产出、本 GEMM 核读表）。每项四要素：
> 原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：
> `examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/verify_equiv.py`。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| M1 | 动态 extent 精确裁剪 + 动态 K 归约范围（**本核采纳**） | 整块取数（TMA OOB 零填充）+ 谓词 STG/守卫（源 full/partial 双路） | 只取/只写有效区 `tile_m=min(block_m,s_e-row)`, `tile_n=min(block_n,N-n0)`, `tile_k=min(block_k,K-k*bk)`，并以 `T.gemm(size=[tile_m,tile_k,tile_n])` 令归约维=tile_k | 容差内等价：仅当归约范围=有效 tile_k 时越界项恒为 0（0·x=0）；机器验证（**含 K 分块 + 陈旧尾值注入**）2 dtype × 5 用例 fp64 0 ulp，10/10 EQUIV_PASS；`stale_tail_guard` 确认缺 `size` 会发散（max_abs_diff ≈1.218e3）。D-3 探针设备实测 max_diff ≤7.6e-06 | 消 TMA OOB 零填充与 full/partial 双路 epilogue；尾块只搬有效行/列；`size` 免去整块零填充；合并两支为单路（源每 tile 双分支守卫 → 单路动态 extent） |
| M2 | 累加器初始化融合（`initC=(k==0)`，**本核采纳**） | `c=0; for k: c += A_k·B_k`（源 `clear_accum=(k==0)` 本质等价，但需显式清零语义） | `c = A_0·B_0; for k>0: c += A_k·B_k`（`T.gemm(initC=(k==0))`） | 数学恒等：IEEE-754 `0 + x == x` 精确成立（含 ±inf/NaN/subnormal）；机器验证 2 dtype × 5 用例 fp64 **0 ulp**，10/10 EQUIV_PASS | 省一次 `block_m×block_n` fp32 清零写（64×256×4 = 64KB/ tile） |
| M3 | NT 形式（不物化 B^T）（**本核采纳；源已用**） | 显式转置/重排 B 后 `b_transpose=False` | 直接 `T.gemm(..., b_transpose=True)` 用 B[N,K] 行主序 | 数学恒等 $(AB^T)$；源即 `transpose_B=True` | 省一次 B 转置/重排（B 是最大张量 15 GB，任何全量重排都不可接受） |
| M4 | 无进一步公式化简 | — | — | 单次 NT 矩阵乘 + fp32 分块累加，无公共子表达式、无高代价超越函数、无除法可倒数化（无 `x/s` 形式） | 结论：矩阵乘核心无可降代价的恒等变形，M1–M3 即全部采纳项 |

**优化结论**：采纳 3 项（M1/M2/M3）。收益主体来自 M1（动态 extent 消 TMA 零填充 + full/partial 双路守卫，
合并单路）与 M2（首块初始化免一次清零写）；M3 承接源的 NT 形式（免 B 转置）。
**优化后公式**（供 §3.1 拆解）：对每逻辑 tile（读表得 $e$、$row$）、每 tile $(t_m, t_n)$、每 $k$-块：
$$
C[\,m_0:m_0+t_m,\ n_0:n_0+t_n\,] \;=\; \sum_{k} A[\,m_0:m_0+t_m,\ k b_k : k b_k + t_k\,]\,
B[e,\ n_0:n_0+t_n,\ k b_k : k b_k + t_k\,]^{T}\quad\text{(fp32 accumulation, first-block initC)}
$$
其中 $m_0=o_{e}+row$（`row` 已乘 block_m）、$n_0=t_n^{idx}\cdot block_n$，$t_m/t_n/t_k$ 为动态有效尺寸。

**等价性机器验证结果表**（`verify_equiv.py`，fp64 torch CPU 参照；脚本随算子目录留存）：

| 项 | 对照 | dtype | 最大 ulp 差 | 违反 / 用例 | 结论 |
|----|------|-------|-----------|-------------|------|
| M1 | 有效区 + 动态 K 归约范围 vs 零填充全 `block_k` 归约（每专家 NT matmul，含 K 分块 + 陈旧尾值） | float16 | 0.000e+00 | 0 / 5 | EQUIV_PASS |
| M1 | 同上 | bfloat16 | 0.000e+00 | 0 / 5 | EQUIV_PASS |
| M2 | `initC` 融合 vs 显式清零后累加 | float16 | 0.000e+00 | 0 / 5 | EQUIV_PASS |
| M2 | 同上 | bfloat16 | 0.000e+00 | 0 / 5 | EQUIV_PASS |
| （guard） | 省略 `size=`（全 `block_k` 归约，含陈旧尾值）vs 零填充基线 | float16 / bfloat16 | max_abs ≈ 1.218e+03 / 1.216e+03（发散，非采纳项） | 5 / 5 | 确认 M1 前提不可省 |

无失败项（所有采纳项均为 EQUIV_PASS）。`guard` 行模拟「动态取数但漏传 `size`」的错误实现，与基线产生
1.2e3 量级发散，**证明 M1 的等价性确实依赖 `T.gemm(size=[tile_m,tile_k,tile_n])` 限定归约范围**。
D-3 探针（设备实测）在相同形态上 worst max_diff 7.629e-06。
fp16/bf16 舍入影响评估：三项均为精确恒等（M2 为 IEEE `0+x==x`）或「删除恒零项 / 仅改搬运与归约范围」
（M1/M3），不改变任何实际参与累加的浮点运算，故不引入额外的 fp16/bf16 舍入误差。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点**本 GEMM 核**实现方案全部循环与标量计算点（对象 = §1.6.0 选定、§0.6 重设计后的 C1 算法）。
**说明**：源内核的调度索引段（thread0 前缀扫描 + 二分查找）已按 §0.6 R1 **外移到前导 kernel**
（其循环/标量点由调度 kernel 的 §1.6.2 承载）；本核不再含任何索引向量工作。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | K 维累加循环 | 源 producer/consumer WG + wgmma 累加环 | Cube `T.gemm`（已是矩阵单元，非标量） | ✅ | — |
| 2 | A/B tile 装载 | 源 TMA `tma_copy` | `T.copy` 切片（向量搬运 + ND2NZ） | ✅ | — |
| 3 | 输出写回 + cast | 源 full tile 经 C_shared+TMA store / partial tile 谓词 `T.Parallel` 守卫逐元素写 | 单路 `store_fixpipe([tile_m,tile_n])`（含 f32→f16/bf16 量化）——**整体消除逐元素守卫循环与 C_shared staging** | ✅ | — |
| 4 | M 尾零填充（TMA OOB） | 源 descriptor 零填充 | 动态 extent 取代（不填零，只取有效区） | ✅ | — |
| 5 | tile 坐标与基址 `bx/by/m_start/n_start` | 标量乘加（读调度表导出） | 无 | ❌ | block 级索引 / 任务映射计算（每 task 少量标量），无逐元素等价 API；输入来自 GM 调度表（运行时值） |
| 6 | 调度表标量读 `total_tiles[0]`、`tile_expert_ids[bx]`、`tile_row_offsets[bx]`、`true_offsets[e]`、`true_sizes[e]` | GM 标量读 | 无 | ❌ | 每 task 数次标量 GM 读（O(1) 寻址，正是调度表的设计目的）；非逐元素热点；`bx` 为运行时值，向量化无意义 |
| 7 | 动态 extent `tile_m/tile_n/tile_k` | 标量 `T.min` | 无（标量归约） | ❌ | 每 task 3 次 `T.min` 的 block 级标量，用于设定搬运/写回 extent，非逐元素热点 |
| 8 | 死 tile 守卫 `bx < total_tiles[0]` | 标量比较 | 无 | ❌ | block 级单比较守卫（每 task 一次），非逐元素热点 |

**向量化结论**：本 GEMM 核为 **Cube 主导**，逐元素计算已全部消除（第 3 项整体消除、第 4 项删除），
数据搬运与矩阵计算由 Cube 单元（`T.copy`/`T.gemm`/`store_fixpipe`）承担（第 1/2 项）。仅保留 4 处
block 级标量点（第 5/6/7/8 项：tile 坐标基址 / 调度表标量读 / 动态 extent / 死 tile 守卫），理由类别：
**block 级索引计算 + O(1) 运行时标量读（数据来自 GM 表）**；均非逐元素热点（每核 O(num_local_tasks) 次
标量，相对 B 访存（15 GB 量级）可忽略）。**无可向量化而未向量化的逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子为 Cube 类（I/O layout 为契约：A `(numel,K)`、B `(E,N,K)`、C `(numel,N)` 行主序），
> 核内布局是设计变量。调度阶段的向量化轴（专家轴 E）属前导 kernel（先例
> `_tile_scheduler_kernel` 设计），**本核无向量化轴**（Cube-only，无 Vector scope）。Cube 路径布局按
> cube-skill + 先例 epilogue 归属决策。

**轴质量评分**：本核无 Vector 逐元素/归约轴（索引已外移），故不适用逐 lane 整除性评分；调度表的专家轴
评分见调度 kernel 设计 §1.6.3。

**候选矩阵**（Cube 类：分形布局 + epilogue 归属）：

| # | 布局方案 | 向量化轴 / 关键决策 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| L1 | A/B GM→L1 走 `T.copy` 切片（Expert 模式自动 ND2NZ），`T.gemm(b_transpose=True, size=[tile_m,tile_k,tile_n])` 直接消费 B[N,K] | 无（本核无 Vector 轴） | 无（ND2NZ 在搬运内完成） | 无额外 GM 流量；NZ 由后端处理 | ✅ **主选** |
| L2 | 显式 host 侧把 B 转置物化为 [K,N]（或提前转置），再 `b_transpose=False` | — | host permute | B 是全仓最大张量（15 GB），任何全量 host 转置 = 百 µs~ms 级且改动输入/需额外存储 → 净亏；且 host 侧禁止改动输入真实内容（ascend-constraints.md §4） | ❌ 弃选 |
| L3 | 核内 `T.transpose` 显式把 B tile 转成 [K,N] 再 gemm | — | 核内转置链 | 多余向量管线开销；且 NZ 通路已支持 NT 形式（`b_transpose=True`，T.gemm.md §2.1）；另有 TRAP-transpose-epilogue-poison（活跃源 transpose 毒化 kernel 2.6×，pattern-library traps-compiler.md） | ❌ 弃选 |
| L4 | epilogue 归属 Vector（L0C→GM 走 GM 往返再由 Vector 逐元素处理） | — | L0C→GM→UB | 本算子 epilogue 无逐元素运算（仅裁剪写回），Vector 处理无收益且引入 GM 往返（CONST-store-fixpipe-gm-only：L0C 跨引擎传输强制 GM 往返） | ❌ 弃选 |
| L5 | epilogue 归属 Cube（`store_fixpipe` 动态 `[tile_m,tile_n]`） | — | 无（fixpipe 直出 GM，含 fp32→fp16/bf16 量化） | 单次搬运，尾块精确裁剪 | ✅ **主选** |
| L6 | 核内调度内嵌（源形态：Cube 内维护调度状态 + static-wave） | E 轴 `T.Parallel` / 标量 | — | — | ❌ 弃选（**工具链不可实现**，CG-2026-0010 / CG-2026-0027；§0.6 R1） |

**布局决策结论**：选定 **L1 + L5** —— 核内布局为 **NZ 分形**（A/B 经 GM→L1 搬运隐式转换，`T.gemm` NT 形式
直接消费 B[N,K]），**epilogue 归属 Cube 侧 `store_fixpipe`**（动态 extent 裁剪，无 Vector 二次处理）；
**本核为 Expert Cube-only，无 Vector scope、无跨引擎交接**（L6 因工具链阻断弃选，§0.6 R1）。
repack 路径：无（NZ 转换在搬运内完成）。该结论已同步 §3.3 伪代码与 §6 循环结构（本核循环只有核内
`T.serial` 任务循环 + Cube `T.gemm` 的 K 维循环，无 Vector 索引循环）。L2/L3/L4 的弃选量化理由见候选矩阵
（并依据 negative-claim-evidence.md §1：L2/L3 的负向论断附 ascend-constraints.md §4、T.gemm.md §2.1、
pattern-library TRAP-transpose-epilogue-poison 条款）。本算子无「主选 vs 备选判定裕度依赖未实证常数」
的轴决策（NT+NZ 是文档指定通路），故不触发实验裁决三件套。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Expert（唯一形态，单 Cube scope）**。执行形态：`T.Kernel(nums_kernel, is_npu=True)` +
`T.Scope("Cube")` + `T.alloc_L1`/`T.alloc_L0C` + `T.copy` + `T.gemm(b_transpose=True,
size=[tile_m,tile_k,tile_n])` + `T.store_fixpipe([tile_m,tile_n])`，**核内无 Vector 段、无跨引擎同步**；
调度表由前导 kernel（独立单元）产出（§0.6 R1）。用户虽指定 `programming_mode=developer`，但 Developer
形态经先例实测**不可实现**（lowering 期 segfault，CG-2026-0010），按「实测为仲裁依据」切换 Expert 并在
此披露。

**分类口径澄清**：源算子数据通路是 **Cube 主导**（唯一数值计算为 WGMMA NT 矩阵乘 + epilogue cast），
**无逐元素浮点向量工作**；其调度索引段为**纯标量整数**（thread0 前缀扫描 + 二分），NPU 侧已外移为前导
kernel。按 AGENTS.md 的 Developer-mode MixCV 触发规则（Cube `T.gemm` + **至少一个 v-prefix op**），
本算子严格意义上不满足 v-prefix 判据（未使用 `vadd/vmul/vexp/vcast/vbrc` 等），应归为 **纯 Cube（Expert）**；
故本核不涉及 Cube↔Vector 交接，不再需要 mixcv-skill 的跨 scope 同步范式。

### 2.2 选型理由

- **Developer 不可用（实测）**：(i) Developer 单 Scope（persistent `T.serial` + `T.gemm` + 整型索引段）
  在 lowering 期 segfault（无 stderr，signal 11；单独验证：去索引段正常、去 gemm 正常，共存即崩）；
  (ii) Developer 下动态 extent `T.gemm` 静默错值（max_diff 1.04e-2，Expert 同结构 9.5e-7）。
  二者共同封死 Developer 路径（CG-2026-0010 扩展触发面）。
- **Expert 形态且只保留 Cube scope（实测可用）**：把调度外移（§0.6 R1）后，本核为**单一 Cube scope**，
  无 Vector→Cube 交接，从而**完全规避** CG-2026-0026（`sync_block_*` 属性名不匹配，任何跨引擎同 block
  交接不可编译）。Stage 3 正对照（先例）验证：Expert Cube + 外部调度表 + 动态 extent + NT gemm +
  store_fixpipe 的 max_diff 9.537e-07；本算子 D-3 探针 4 用例全 PASS（worst max_diff 7.629e-06）。
- **零 CPU 同步**：`total_tiles` 与两张表全程留在设备侧（GM），两 kernel 同 stream 顺序 launch，
  无 `.item()`、无 host 回读。

### 2.3 模式影响

| 维度 | Expert（本核，唯一） |
|------|----------------------|
| 内存分配 | `T.alloc_L1`（A/B 操作数）、`T.alloc_L0C`（累加器）；**无 UB 分配**（调度已外移） |
| 取值 | `T.copy` 切片（GM→L1 ND2NZ） |
| 写回 | `T.store_fixpipe`（动态 extent，`enable_nz2nd=True`） |
| 调度输入 | GM 表 `tile_expert_ids`/`tile_row_offsets`/`total_tiles`（前导 kernel 产出）+ `true_sizes`/`true_offsets` |
| 同步 | 无跨引擎同步；仅 Cube scope 内 `T.rs("PIPE_FIX")` 管线同步（框架/PIPE flag） |
| 跨引擎交接 | **无**（单 scope；这正是规避 CG-2026-0026 的关键） |

---
## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的优化后公式。本核无 Vector 索引段（已外移）；矩阵单元用 `T.gemm`，搬运用 `T.copy`，
> 写回用 `T.store_fixpipe`。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `bx = tile_id // num_pid_n`，`by = tile_id % num_pid_n` | tile 坐标（核内 `T.serial` 任务） |
| 2 | 读 `total_tiles[0]`，`bx < total_tiles[0]` 判定存活 | 死 tile 守卫（GM 标量读，单比较） |
| 3 | `expert_id = tile_expert_ids[bx]`，`row = tile_row_offsets[bx]` | O(1) 寻址（GM 标量读） |
| 4 | `m_0 = true_offsets[expert_id] + row`，`n_0 = by·block_n` | tile 基址 |
| 5 | `t_m=min(block_m, true_sizes[e]−row)`, `t_n=min(block_n,N−n_0)`, `t_k=min(block_k,K−k·bk)` | 动态 extent（M1） |
| 6 | `C_tile = Σ_k A[m_0:m_0+t_m, kb:kb+t_k] · B[e, n_0:n_0+t_n, kb:kb+t_k]^T` | NT GEMM + fp32 累加（M3 + M2 首块 initC） |
| 7 | 写回 `C[m_0:m_0+t_m, n_0:n_0+t_n]` | 动态 extent store（M1） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | tile 坐标 | 标量整除/取模 | `bx/by` 标量 | Expert |
| 2 | 死 tile 守卫 | 标量比较（GM 标量读 `total_tiles[0]`） | `if bx < total_tiles[0]` | Expert |
| 3 | 调度表寻址 | GM 标量读（运行时下标） | `tile_expert_ids[bx]`、`tile_row_offsets[bx]` | Expert |
| 4/5 | 基址/动态 extent | 标量乘加 + `T.min` | `T.min(block_m, true_sizes[e]−row)` 等 | Expert |
| 6 | NT 矩阵乘 | `T.gemm(a_l1, b_l1, c_l0c, size=[tile_m, tile_k, tile_n], b_transpose=True, initC=(k==0))` | `size=[m,k,n]` 限定动态归约范围；`b_transpose=True` | Expert |
| 6 | K 维循环 | `T.serial(ceildiv(K, block_k))`（先例正对照形态；Stage 4 可试 `T.Pipelined(num_stages)`） | 分块累加 | Expert |
| 6/7 | 取数/写回 | `T.copy` 切片（GM→L1 ND2NZ）+ `T.store_fixpipe(c_l0c, C[m0,n0], size=[t_m,t_n], enable_nz2nd=True)` | 动态 extents | Expert |

> 说明：本核无逐元素运算，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix 算术 API（v-prefix 面向浮点
> 向量运算，见 AGENTS.md API Convention）——`T.gemm` 承担矩阵单元、`T.copy`/`T.store_fixpipe` 承担数据搬运。

### 3.3 计算伪代码

> `T.Tensor` 第二个位置参数直接传 dtype（不用 `dtype=` 关键字），避免 `name '...' is not defined` 类假警报。
> 本伪代码与先例 Stage 3 已验证正对照（`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`
> 的 `positive_control()`）同形态，并补全 `num_pid_n`/动态 A 行。

```python
# 编译期常量：numel, num_experts, N, K, block_m, block_n, block_k, num_stages,
#             max_tiles = numel // block_m + num_experts
#             num_pid_n = ceildiv(N, block_n)
#             nums_kernel = min(max_tiles * num_pid_n, P)   # P = 实查物理核数 20
#             num_local_tasks = ceildiv(max_tiles * num_pid_n, nums_kernel)  # 静态
# 调度表由前导 kernel（独立单元）产出（§0.6 R1），本核只读。

@tilelang.jit(out_idx=[2], target="npuir", pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
})
def _persistent_grouped_gemm_v2_kernel(numel, max_tiles, num_experts, N, K,
                                       dtype="float16", block_m=64, block_n=256,
                                       block_k=64, num_stages=2):
    num_pid_n = (N + block_n - 1) // block_n
    num_logical = max_tiles * num_pid_n
    nums_kernel = min(num_logical, PHYS_CORES)            # PHYS_CORES = 20
    num_local_tasks = (num_logical + nums_kernel - 1) // nums_kernel

    @T.prim_func
    def main(
        A: T.Tensor((numel, K), dtype),
        B: T.Tensor((num_experts, N, K), dtype),
        C: T.Tensor((numel, N), dtype),
        tile_expert_ids: T.Tensor((max_tiles,), "int32"),
        tile_row_offsets: T.Tensor((max_tiles,), "int32"),
        true_offsets: T.Tensor((num_experts,), "int32"),
        true_sizes: T.Tensor((num_experts,), "int32"),
        total_tiles: T.Tensor((1,), "int32"),
    ):
        # 单一 Cube scope：只读 GM 调度表 + NT GEMM + 写回；无 Vector 段、无跨引擎同步。
        with T.Kernel(nums_kernel, is_npu=True) as (core_id, _subid):
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((block_m, block_k), dtype)
                b_l1 = T.alloc_L1((block_n, block_k), dtype)
                c_l0c = T.alloc_L0C((block_m, block_n), "float32")
                # 固定核数（=AI core 数）+ 核内串行多任务（循环边界静态，§5.5）。
                for t in T.serial(num_local_tasks):
                    tile_id = core_id * num_local_tasks + t
                    bx = tile_id // num_pid_n
                    by = tile_id % num_pid_n
                    # 死 tile 守卫：单比较，位于 T.serial 体内、K 循环之外（规避 CG-2026-0001）。
                    if bx < total_tiles[0]:
                        expert_id = tile_expert_ids[bx]        # GM 标量读（运行时下标）
                        row = tile_row_offsets[bx]             # GM 标量读；单位=行（已乘 block_m）
                        n_start = by * block_n
                        m_start = true_offsets[expert_id] + row
                        # 动态 extent：M/N 尾块精确裁剪（no-pad 关键，§5.4）。
                        tile_m = T.min(block_m, true_sizes[expert_id] - row)
                        tile_n = T.min(block_n, N - n_start)
                        for k in T.serial(T.ceildiv(K, block_k)):
                            tile_k = T.min(block_k, K - k * block_k)   # K 对齐时恒 = block_k
                            T.copy(A[m_start:m_start + tile_m,
                                     k * block_k:k * block_k + tile_k],
                                   a_l1[0:tile_m, 0:tile_k])
                            T.copy(B[expert_id, n_start:n_start + tile_n,
                                     k * block_k:k * block_k + tile_k],
                                   b_l1[0:tile_n, 0:tile_k])
                            # size=[tile_m, tile_k, tile_n] 限定归约范围=有效 tile_k
                            # （正确性必需，见注 1 / §1.6.1 M1 stale guard）。
                            T.gemm(a_l1, b_l1, c_l0c,
                                   size=[tile_m, tile_k, tile_n],
                                   b_transpose=True, initC=(k == 0))
                        # 动态 extent 写回：只写有效 [tile_m, tile_n]，杜绝越界写坏下一专家行。
                        with T.rs("PIPE_FIX"):
                            T.store_fixpipe(c_l0c, C[m_start, n_start],
                                            size=[tile_m, tile_n],
                                            enable_nz2nd=True)

    return main
```

**注 1（K 尾正确性前提）**：`T.gemm(..., size=[tile_m, tile_k, tile_n])` 的 `size` 不是可选优化而是
**正确性必需**——`a_l1/b_l1` 的 `[tile_k, block_k)` 区未被 `T.copy` 写入，若无 `size` 则按缓冲静态
`block_k` 归约、把陈旧值计入累加（K 非整除用例必错）。本算子 K 对齐（源硬前置），对齐时 `tile_k ≡ block_k`，
该前提仅在防御性非对齐路径上生效；§1.6.1 M1 的机器验证已覆盖含 K 分块的形态并注入陈旧尾值，
`stale_tail_guard` 确认缺失 `size` 会发散（max_abs_diff ≈1.218e3）。
**注 2（为何单 Cube scope 可行、为何不再需要 Vector/warp-spec）**：调度表由**前导 kernel** 产出并落 GM，
本核只做「读 GM 表 + 读 GM 标量 + GEMM + 写回」，**不存在 Vector→Cube 同 block 交接、不存在 Cube 内
调度状态维护**，故不触发 CG-2026-0026/0010/0027；先例 Stage 3 正对照已验证该形态数值正确
（max_diff 9.537e-07），本算子 D-3 探针 4 用例全 PASS。
**注 3（`initC=(k==0)`）**：首块以 `initC=True` 覆盖初始化 L0C，后续块累加；等价于源 `clear_accum=(k==0)`
后逐块累加（避免在循环体内构造 `k==0` 条件，CG-2026-0001）。§1.6.1 M2 机器验证 0 ulp。
**注 4（`true_offsets`）**：源 op 的必需输入；本设计保留其作为输入（与 `true_sizes` 同源）。
**注 5（前导 kernel 不在本单元）**：调度表产出者（前缀扫描 + 逐专家区间展开）为独立单元（先例
`_tile_scheduler_kernel`），本单元只消费其产出（对接契约 §4.1）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.gemm(A,B,C,size=[m,k,n],b_transpose=True,initC=False)` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §1/§2.1（签名 `T.gemm(src1,src2,dst,size=[],initC=False,a_transpose=False,b_transpose=False)`；`size=[a,b,c]` 语义：src1 为 [a,b]/[b,a]、src2 为 [b,c]/[c,b]）；`examples/gemm/matmul_dynamic_shape.py`、`examples/mixcv/mixcv_mixkernel.py`、先例正对照脚本 | 已验证（示例与正对照在用；**无 `transpose_B` 形参**，本核统一用 `b_transpose`；D-3 探针实测） |
| `T.Kernel(n, is_npu=True)`（1D 核 + 核内 `T.serial`） | `examples/deepseek_v32/fp8_lighting_indexer.py`；先例正对照；D-3 探针 `persistent-P` 计划 | 已验证（示例与正对照在用；探针实测启动 20 核） |
| `T.copy` 切片 + 动态 extents（尾块）；GM→L1 自动 ND2NZ | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 第 2 条、§2.3 第 6 条、§2.4 示例 4 | 已验证（文档明确 + 正对照 + D-3 探针在用） |
| `T.min` 标量/张量 | `docs/Tilelang.language/比较操作/T.min.md`；`examples/mixcv/mixcv_mixkernel.py`；正对照 | 已验证 |
| `T.store_fixpipe`（动态 extent） | `docs/Tilelang.language/内存操作/T.store_fixpipe.md`；`examples/mixcv/mixcv_mixkernel.py`；正对照 | 已验证 |
| `T.alloc_L1` / `T.alloc_L0C` | `examples/mixcv/mixcv_mixkernel.py`；先例正对照；D-3 探针 | 已验证（Expert 路径） |
| `T.serial` / `T.Pipelined` | `docs/Tilelang.language/` + `examples/gemm/example_gemm.py` L27；正对照 | 已验证 |
| ~~`T.sync_block_set/wait`~~ | `docs/Tilelang.language/同步管道操作/T.sync_block_set.md` | **不可用（工具链缺口）**：codegen 出 `syn_instr_mode`、bishengir-compile 解析 `sync_instr_mode` → `error: custom op 'syn_instr_mode' is unknown`（CG-2026-0026，官方 mixcv 样例复现）。本设计**不使用**（单 Cube scope 无跨引擎交接） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 是 | 一维 `T.Kernel(nums_kernel)`；tile 坐标由核内 `bx/by` 除法/取模得到 |
| GEMM M/N 需为 block 整数倍 | 是（numel 一般不整除 block_m，N 可 < block_n） | 动态 extent 精确裁剪（§0.6 R7）；host 侧不做任何 padding |
| L0C 容量（128KB） | 是 | block_m×block_n×4B = 64×256×4 = 64KB ≤ 128KB ✓ |
| 物理核数限制 | 是 | `nums_kernel = min(逻辑核数, 20)`；极大规模核内 `T.serial`（§5.5） |
| host 侧禁止改动输入 | 是（源无 host pad；但须保持禁止） | 全部尾块处理在 kernel 内（动态 extent），host 只做元数据 |
| 三维 Kernel / GPU API | 源用 TMA/wgmma/warp-spec/mbarrier（CUDA） | 换 NPU 等价：`T.copy` 切片 + NZ + `T.gemm` + 动态 extent + `store_fixpipe`（§0.5） |
| 跨引擎同 block 交接 | 源无（单 CUDA kernel）；本设计不使用 | 单 Cube scope；依据 CG-2026-0026（§0.6 R1） |
| Cube scope 内维护调度状态 | 源内嵌 static-wave 调度器 | **外移为前导 kernel + GM 表**（§0.6 R1）；依据 CG-2026-0010/0027 |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU / Hopper） | 本项目（Ascend NPU） | 转换方案 |
|--------|----------------------|---------------------|----------|
| Kernel 维度 | 1D grid（`sm_count` persistent） | 1D grid（`nums_kernel`=AI core 数）+ 核内 `T.serial` 多 tile | §0.6 R4 |
| tile 调度 | **内嵌** static-wave 调度器（thread0 前缀扫描 + 二分） | **外移**：前导 kernel 产 GM 表，GEMM 读表 | §0.6 R1 |
| 并行组织 | 1 producer + 2 consumer WG（warp specialization） | 单核 K 循环（无 warp 分流） | §0.6 R2 |
| 模板 | 双模板（pingpong / cooperative，split-A/shared-B） | 单一 Expert Cube 模板 | §0.6 R3 |
| 取数 | TMA + mbarrier ring | `T.copy` 切片 + 框架同步 | §0.6 R2/R7 |
| 尾块 | TMA OOB 零填充 + full/partial 双路 epilogue | 动态 extent 精确裁剪 + 单路 `store_fixpipe` | §0.6 R6/R7 |
| swizzle | `make_swizzled_layout` | 删除（NZ 由搬运转换） | §0.5 O4 |
| `sm_count` | `multi_processor_count`（CUDA SM 数） | `NPUUtils.get().get_aicore_num()`（AI core 数 20） | §0.6 R4 |
| 编程模式 | CUDA 单 kernel | Expert 单 Cube scope（无 Vector；规避 CG-2026-0026） | §2 |
| B 复用 | TB swizzle `group_size_m`（L2） | 专家聚簇 + M-major 任务序 | §0.6 R5 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py` | 直接（可行底座实现） | Expert `T.Scope("Cube")` + 外部调度表 + 动态 extent + `T.gemm(size=)` + `store_fixpipe`（实测 max_diff 9.537e-07）——**本核结构的实测底座** |
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` | 直接（正对照） | Expert Cube 动态 extent GEMM 正对照 + CG-2026-0026 复现（官方样例） |
| `examples/deepseek_v32/fp8_lighting_indexer.py` | 高度相似 | 固定核数 + 核内 `T.serial(ceildiv(...))` 多任务 + 单比较守卫；`T.gemm(initC=True, b_transpose=True, size=[...])`；`store_fixpipe` 动态基址 |
| `examples/gemm/matmul_dynamic_shape.py` | 高度相似 | **M/N/K 全动态 extent 的规范写法**：`remain_M/remain_N/remain_K = T.min(...)` + `T.gemm(..., initC=(i==0), size=[remain_M, remain_K, remain_N])` + `store_fixpipe(size=[remain_M, remain_N])` |
| `examples/mixcv/mixcv_mixkernel.py` | 高（Expert 双 Scope 范式） | `T.Scope("Cube")/("Vector")` + `tile_size_M=T.min(...)` + `store_fixpipe`；**注意其 `sync_block_set/wait` 当前不可编译（CG-2026-0026），本设计不用** |
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | 高（调度表产出者） | 前导 kernel 产 3 张 GM int32 表（对接契约见 §4.1） |
| `examples/gemm/example_gemm.py` | 高（Cube 基础） | 一维 `T.Kernel` + `T.Pipelined` + `T.gemm` + `T.copy` 的标准 matmul 骨架 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (numel, K) | float16 / bfloat16 | 紧致激活，行连续；K 对齐（`K % block_k == 0`） |
| B | (num_experts, N, K) | 同 A | 专家权重（NT，对 B 转置恒等使用），行连续；N 对齐 |
| true_sizes | (num_experts,) | int32 | 每专家行数 $s_e$ |
| true_offsets | (num_experts,) | int32 | 每专家在 A 中的紧致起始行 $o_e$ |
| tile_expert_ids | (max_tiles,) | int32 | **调度表**（GM，前导 kernel 产出）：每 M-tile 专家号（死 tile = -1） |
| tile_row_offsets | (max_tiles,) | int32 | **调度表**：M-tile 在专家内的行偏移（已乘 block_m） |
| total_tiles | (1,) | int32 | **调度表**：有效 M-tile 总数（设备侧，不回落 CPU） |

> **对接契约（本核消费 / 前导 kernel 产出）**：① `max_tiles = numel // block_m + E` 与 `block_m` 为两
> kernel 共享的编译期常量；② 表 dtype 全 int32；③ `tile_row_offsets` 单位是**行**（已乘 block_m），不是
> tile 序号；④ 死 tile 哨兵 `(-1, 0)`；⑤ `total_tiles ≤ max_tiles`；⑥ 两 kernel 同 stream 顺序 launch，
> 前导 kernel 先完成，GEMM 核再执行（无 in-block 同步，无 CPU 同步）。
> 与已迁移 `MoeGroupedGemmNopadFwdOp` 一致（其 GEMM 核亦读此三类表）；表的产出算法（前缀扫描 + 逐专家
> 区间展开）由前导 kernel 承担，本单元不实现（§3.3 注 5）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (numel, N) | 同 A | 行连续，非转置布局 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_l1 | (block_m, block_k) | dtype | L1 | A tile（ND2NZ） |
| b_l1 | (block_n, block_k) | dtype | L1 | B tile（ND2NZ） |
| c_l0c | (block_m, block_n) | float32 | L0C | 累加器 |
| （调度表）| —— | —— | GM | 前导 kernel 产出；本核只读，不占本核片上预算 |

> 本核**无 UB 分配**（调度索引阶段已外移，§0.6 R1）。

### 4.4 内存搬运路径

```
[前导 kernel] true_sizes --前缀扫描/逐专家区间展开--> GM[表: tile_expert_ids / tile_row_offsets / total_tiles]
                                                 │  (同 stream 顺序 launch，无 in-block 同步)
                                                 ▼
[本核] 标量读 GM 表 + GM 标量(true_offsets/true_sizes) --> (expert,row,m_start,n_start,tile_m,tile_n)
GM[A] --T.copy 切片(动态extent, ND2NZ)--> L1[a_l1] ┐
GM[B] --T.copy 切片(动态extent, ND2NZ)--> L1[b_l1] ├─ T.gemm(b_transpose=True, size=[tile_m,tile_k,tile_n]) --> L0C[c_l0c fp32]
                                                    ┘                                     │
                              GM[C] <── store_fixpipe([tile_m,tile_n], nz2nd) ─────────────┘
```

### 4.5 片上内存预算（Expert，block_m=64/block_n=256/block_k=64/fp16）

| 层级 | Buffer | Shape | dtype | 大小 (Bytes) |
|------|--------|-------|-------|-------------|
| L1 | a_l1 | (64,64) | fp16 | 8192 |
| L1 | b_l1 | (256,64) | fp16 | 32768 |
| L1 | （×num_stages=2） | | | 81920 |
| L0C | c_l0c | (64,256) | fp32 | 65536 |
| UB | —（本核无 UB） | | | 0 |

> **L1 预算复核**：A(8KB)+B(32KB) × num_stages(2) = 80KB ≤ 512KB ✓（`CONST-capacity-910B2C`）；
> num_stages 可达 4（160KB），默认 2。
> **L0C 复核**：64×256×4 = 64KB ≤ 128KB ✓（`CONST-capacity-910B2C`）。
> `pass_configs` 关闭 auto-multi-buffer（`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`），故不适用 ×1.7 膨胀系数
> （该系数对应 auto-multi-buffer **开启**；Expert 显式 alloc 结构 ≈ 手工核算再 ±12%）。
> **UB**：本核无 UB 分配（调度已外移）。

### 4.6 动态轴定义

无动态轴：`numel / num_experts / N / K` 均为编译期常量（JIT 特化）。运行时变化的仅为 `true_sizes`
的数据值（不改变 shape），由动态 extent 与守卫处理。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir",
              pass_configs={
                  tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
                  tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
              })
def _persistent_grouped_gemm_v2_kernel(numel, max_tiles, num_experts, N, K,
                                       dtype="float16", block_m=64, block_n=256,
                                       block_k=64, num_stages=2):
    ...
```

> `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` 避免 buffer 重排/重作用域破坏显式 L1/L0C
> （CG-2026-0012 / TRAP-UB-dynsubview-dominance）；`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False` 关闭
> auto-multi-buffer（与先例正对照的 `_PASS` 一致）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Cube（Expert 单 Cube scope）+ 前导 kernel 的整数索引**。判定依据：本核唯一数值计算为
`T.gemm`（Cube 矩阵单元）+ `store_fixpipe`（epilogue 归 Cube），无逐元素浮点向量、无 Vector scope；
源内核的调度索引段（thread0 前缀扫描 + 二分）**外移为前导 kernel**（§0.6 R1），不在本核内。

### 5.2 Block 划分

```python
block_m = 64    # 分形 M 对齐（≥16）；L0C 64KB 预算；专家行数通常 ≥ 数十
block_n = 256   # 分形 N 对齐；L0C 64×256×4=64KB ≤128KB
block_k = 64    # 分形 K 对齐（≥32）；源 K 对齐（K%64==0），防御性动态 K 尾由 size= 处理
```

> 源默认 `block_m=128 / block_n=256 / block_k=64`（cooperative 模板）。NPU 侧取 `block_m=64`：
> 单 Cube scope 下 L0C 需容纳整个 `block_m×block_n` fp32（`128×256×4 = 128KB` = L0C 硬上限，无余量），
> 而源 cooperative 模板把 M 劈半使每 WG 仅 64×256；NPU 单模板取 `block_m=64` 等价于该劈半粒度且留容量
> 余量（64KB）。与先例选定值一致。

### 5.3 约束分析

- **对齐约束**：block_m=64、block_n=256、block_k=64 均 ≥ 分形下限（M/N≥16、K≥32）且 32B 对齐 ✓。
- **L0C 容量**：64×256×4 = 64KB ≤ 128KB ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：A(8KB)+B(32KB) × num_stages(2) = 80KB ≤ 512KB ✓；num_stages 可达 4（160KB）。
- **UB 容量**：本核无 UB 分配 ✓（调度已外移）。
- **K/N 对齐**：源硬前置 K%block_k==0、N%block_n==0；本设计保留对齐形态，并对非对齐做防御性动态
  extent（`tile_k`/`tile_n`）。
- **非整除**：numel / 每专家 $s_e$ 均可能不整除 block_m → **动态 extent 精确裁剪**（§5.4）。

### 5.4 注意事项（非整除处理）

- **M/行方向**：每专家独立切 tile，尾块 `tile_m = T.min(block_m, s_e − row)`（`row` 来自调度表，已乘
  block_m）；`store_fixpipe` 只写 `[0:tile_m)` 行，**杜绝越界写坏下一专家**（no-pad 关键正确性约束）。
- **N 方向**：`tile_n = T.min(block_n, N − n_start)`（源要求 N 对齐；smoke 用例 N=128 < block_n=256 时
  由动态 extent 处理）。
- **K 方向**：源 K 对齐，`tile_k = block_k`；防御性保留 `tile_k = T.min(block_k, K − k·block_k)`，
  **该动态 extent 必须同时作为 `T.gemm(..., size=[tile_m, tile_k, tile_n])` 的归约范围**，否则操作数缓冲
  `[tile_k, block_k)` 的陈旧值会被按静态 block_k 归约计入累加（§3.3 注 1、§1.6.1 M1 guard）。
- **无 host padding**（ascend-constraints.md §4）；源本身也无 host pad（用 TMA OOB 零填充）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。
> 本节针对**本 GEMM 核**；前导 kernel 的分核以其自身设计为准。

- **① 逻辑核数计算**：`num_logical_kernels = max_tiles × num_pid_n = (numel//block_m + E) × ceil(N/block_n)`。
  代表 shape（manifest `MoeGroupedGemmNopadFwdOp.workloads`，block_m=64/block_n=256）：
  smoke(64,4,128,64) → (1+4)×1 = **5**；decode gate-up(4096,256,4096,7168) → (64+256)×16 = **5120**；
  prefill gate-up(32768,256,4096,7168) → (512+256)×16 = **12288**；
  prefill down(32768,256,7168,2048) → (512+256)×28 = **21504**。
- **② 物理核数依据（实查）**：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`
  → **返回值 = 20**（Ascend910B2C；本机实测查询输出 `[probe] NPUUtils.get().get_aicore_num() = 20`，见
  D-3 探针日志 `history_version/design_probe_persistent_gemm.log`）。本算子为 Cube 算子 → **直接使用
  返回值 20**（纯 Vector 才翻倍为 40；**禁止假设值替代实查**）。
- **③ 规模判定与分核方案**：
  - **smoke（逻辑核数 5 ≤ 物理核数 20）**：属于「逻辑核数 ≤ 物理核数」档 → 结论 **无需适配**；
    `nums_kernel = min(5, 20) = 5`，`num_local_tasks = 1`，其余核不启动（覆盖全部逻辑 tile）。
  - **其余代表 shape（5120–21504 ≫ 20）**：**极大规模**，无法通过调 block 缩减到物理核数整数倍
    （block_m/block_n 增大只会让尾块浪费与 L0C 压力上升）→ **固定启动内核数 = 物理核数 20，核内
    `T.serial` 串行处理多个逻辑 tile**：`nums_kernel = min(num_logical_kernels, 20)`（编译期常量），
    `num_local_tasks = ceil(num_logical_kernels / nums_kernel)`（编译期常量，**循环边界静态**）。
    承接源的 **persistent grid = sm_count** 语义：`num_logical ≥ 20` 时 `nums_kernel = 20`，即
    **persistent grid = AI core 数**。
  - **负载均衡**：`tile_id = core_id * num_local_tasks + t` 块连续分配；均匀分布下每核任务数差 ≤1；
    偏斜分布下个别核多领少量死 tile（被守卫 `if bx < total_tiles[0]` 跳过），影响可忽略。块连续分配
    同时利于专家聚簇（§0.6 R5）。
  - **D-3 探针**：`persistent-P` 计划（启动恰好 20 核，核内串行）实测 4 用例全 PASS——含
    `num_pid_n>1` 且 `num_local_tasks>1`（P-3：cores=20, local_tasks=3, max_diff 7.629e-06）。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级 | 固定核数网格 | `T.Kernel(nums_kernel, is_npu=True)` | 分核策略（§5.5）；persistent grid = AI core 数 |
| 核内任务 | 串行多 tile | `T.serial(num_local_tasks)`（静态边界） | 极大规模，摊薄核启动开销；承接源 persistent 意图 |
| 死 tile 守卫 | 单比较跳过 | `if bx < total_tiles[0]` | 位于 `T.serial` 体内、K 循环之外（规避 CG-2026-0001） |
| K 维 | 分块累加 | `T.serial(ceildiv(K,block_k))`（Stage 4 可试 `T.Pipelined(..., num_stages)`） | Cube 分块累加；访存/计算重叠 |
| 元素级 | 无（已消除） | — | full/partial 双路 epilogue 被动态 extent 整体合并为单路（§1.6.2#3）；无 Vector 索引循环（已外移） |

### 6.2 循环伪代码

见 §3.3（核内 `T.serial(num_local_tasks)` 外层、K 循环内层）。**本核无 Vector 索引循环**（源内核内嵌的
static-wave 调度器已外移，§0.6 R1）。

### 6.3 流水线优化

- K 循环 `T.serial`：先例正对照以 `T.serial` 形态跑通（max_diff 9.537e-07），本算子 D-3 探针同形态
  4 用例全 PASS（worst 7.629e-06）；Stage 3/4 可实测 `T.Pipelined(num_stages=2)` 是否更优（若 Pipelined
  与 Expert 单 scope 组合不稳定，则回退 `T.serial`；源用 `T.Pipelined(_k_iters, num_stages=0)` 即退化为
  serial + 手工 barrier）。
- `num_stages` 由 L1 容量约束：A(8KB)+B(32KB) × num_stages ≤ 512KB → 2–4 可行，默认 2（源默认 3，
  NPU 侧按容量与实测重定，Stage 4 扫描）。
- 前导 kernel 与本 GEMM 核**顺序执行**（同 stream）；不同核之间无重叠（两 kernel 形态的固有代价，
  相对 B 访存可忽略，§1.6.0 roofline）。源为单 kernel（`sm_count` persistent），本设计拆为「前导调度
  kernel + GEMM 核」——这是 §0.6 R1 的可实现性代价。

### 6.4 尾块处理

三方向（M/N/K）统一用动态 extent（`T.min`）——见 §5.4；其中 **K 方向须把 `tile_k` 传入 `T.gemm(size=)`**
限定归约范围（否则操作数缓冲的陈旧尾值污染累加，见 §3.3 注 1、§1.6.1 M1 guard）。不使用 TMA OOB 零填充、
谓词 STG 或逐元素守卫（源的 full/partial 双路已合并为单路 `store_fixpipe`）。死 tile（`bx ≥ total_tiles`）
用单比较 `if` 跳过，**该 `if` 位于 `T.serial` 体内，不得放入 K 循环内**（CG-2026-0001：Pipelined 体内
条件构造崩溃）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**Expert 管线同步，但本核无跨引擎同步需求**。本核为**单 Cube scope**，只做「读 GM 表 +
GEMM + 写回」；不存在 Vector→Cube 或 Cube→Vector 的同 block 交接，故**不使用** `T.sync_block_set/wait`
（该原语在当前工具链不可编译，CG-2026-0026）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 前导 kernel → 本 GEMM 核 | **无显式同步**（同 stream 顺序 launch） | 前导 kernel 先完成、本核再执行；kernel 边界即隐式全局顺序，不需 in-block flag |
| L0C 写回 C 前 | `T.rs("PIPE_FIX")`（Cube scope 内） | 保证 `store_fixpipe` 在正确管线阶段发出（先例正对照形态） |
| A/B GM→L1 与 `T.gemm` 之间 | 框架/`T.rs` 管线同步（Cube scope 内） | MTE2→Cube 单引擎内数据依赖；非跨引擎 |
| ~~Vector→Cube `sync_block_set/wait`~~ | **不使用** | CG-2026-0026：`syn_instr_mode` 属性名与 `bishengir-compile` 不匹配，任何跨引擎同 block 交接不可编译 |
| ~~手工 mbarrier ring（源 O3）~~ | **不使用** | NPU 无 warpgroup/mbarrier；由框架多缓冲 + PIPE 同步替代（§0.6 R2） |

> **关键差异（相对源）**：源用 1 producer + 2 consumer WG 的手工 mbarrier ring 同步；本版把搬运与计算
> 交给单核 + 框架同步，**彻底移除跨引擎/手工 barrier 需求**。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```

> 与先例正对照的 `_PASS` 一致：关闭 buffer 重排/重作用域（CG-2026-0012 / TRAP-UB-dynsubview-dominance）
> 与 auto-multi-buffer，保证显式 L1/L0C 结构稳定（PL-1.16 Expert 双关闭硬边界）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源 op 测试
> `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` 的 `_ref_grouped_gemm_nopad`），
> **不得复刻 §0.6 的 NPU 算法**（保证验证独立性）。

```python
import torch

def golden_persistent_grouped_gemm(a, b, true_sizes, true_offsets):
    """每专家 NT matmul（fp32 累加后回落输入 dtype），紧致无 pad。

    a: (numel, K); b: (E, N, K); true_sizes: (E,) int32; true_offsets: (E,) int32
    返回 c: (numel, N)
    """
    numel, _ = a.shape
    E, N, _ = b.shape
    c = torch.zeros(numel, N, dtype=a.dtype, device=a.device)
    sizes = true_sizes.tolist()
    offsets = true_offsets.tolist()
    for e in range(E):
        s = sizes[e]
        if s == 0:
            continue
        off = offsets[e]
        a_e = a[off:off + s].to(torch.float32)
        b_e = b[e].to(torch.float32)
        c[off:off + s] = (a_e @ b_e.T).to(a.dtype)
    return c
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |

（与源 op 测试 `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2)` 一致；fp32 累加 + 输出回落输入
dtype。先例正对照 Expert Cube 动态 extent 形态实测 max_diff 9.537e-07；本算子 D-3 探针 worst max_diff
7.629e-06，均远优于该门限。）

### 8.3 L0 门槛测试计划

> **调度表来源**：生产路径由前导 kernel（独立单元，同 stream 先 launch）产出并落 GM；本 GEMM 核的 L0
> 单独测试可用 **host-built 调度表 fixture**（从 `true_sizes` 按源语义展开，等同先例
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py` 的
> `build_schedule` 形态）——该 fixture 只构造**辅助表输入**，不改动 A/B 真实内容（不违反
> ascend-constraints.md §4）。整算子路径须实测两 kernel 协同（验证对接契约 §4.1）。

| 用例 | shape（numel,E,N,K,dtype,dist） | 验证点 |
|------|--------------------------------|-------|
| L0-1 smoke（K 对齐、N < block_n） | 64,4,128,64,fp16,uniform | 基本正确性、N 尾（tile_n=128<256）、M 尾、读表寻址 |
| L0-2 smoke（偏斜 + fp16） | 64,4,128,64,fp16,skewed | 大量 1-token 专家、死 tile 守卫（`-1` 哨兵） |
| L0-3 K 对齐 + 多 N tile | 128,8,256,128,fp16,uniform | `num_pid_n>1` 动态列偏移、读表寻址 |
| L0-4 中等偏斜 | 233,5,130,70,bf16,skewed | 奇偶 shape、多尾块、多 local task |
| L0-5 代表大 shape | 4096,256,4096,7168,bf16,uniform | 分核极大规模、B 访存路径、无死锁 |
| L0-6 两 kernel 协同 | 64,4,128,64,fp16,uniform | 前导 kernel → GEMM 核对接契约（表内容/`total_tiles` 全程设备侧/无 CPU 同步） |

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 源为单 kernel 内嵌 static-wave 调度器；NPU 侧**必须外移为前导 kernel + GM 表**（Cube scope 内无法
  维护调度状态，CG-2026-0010/0027）→ 设计为两 kernel（§0.6 R1）。
- **跨引擎同 block 交接不可用**（CG-2026-0026）→ 本核单 Cube scope，无 Vector 段。
- Developer 模式对 persistent + gemm + 整型索引段结构不可用（CG-2026-0010）→ Expert 单 Cube scope。
- 源 K/N 对齐为硬前置；本设计保留对齐形态并对非对齐做防御性动态 extent。
- 一维核超发会被串行调度（core-split-strategy §1）→ 固定核数 = AI core 数 + 核内串行。
- **bishengir-compile flaky SIGSEGV**（CG-2026-0033）→ 抬升 `RLIMIT_STACK` + 有界重试（compile-time
  crash，非错值）。
- L0C 容量：`block_m=64`（单 Cube scope）而非源默认 128（128×256×4=128KB 恰为 L0C 硬上限，无余量）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **跨引擎同 block 同步不可编译**（CG-2026-0026） | Expert 双 Scope 内用 `T.sync_block_set/wait` | `error: custom op 'syn_instr_mode' is unknown`（官方 mixcv 样例复现） | **不使用**：调度外移到前导 kernel，本核单 Cube scope（§0.6 R1 / §7.2） |
| **Developer + persistent + gemm + 索引段崩溃**（CG-2026-0010） | 主选 Developer 单 Scope | lowering 期 segfault；动态 extent gemm 静默错值 1.04e-2 | 用 Expert 单 Cube scope（§2.1） |
| **Cube scope 内维护调度状态崩溃** | 内核内 thread0 前缀扫描 / 二分 / loop-carry | loop-carry / L1 scalar 读崩溃 | 调度外移为前导 kernel + GM 表（§0.6 R1） |
| **标量消费向量 scan 产出不同步**（CG-2026-0027） | 单 scope 内「向量 scan → 标量 per-tile 消费」 | 同产物多次运行结果不一致 | 本核无此数据流（全标量 GM 读 + Cube）（§0.6 R1） |
| K 尾忘传 `T.gemm(size=)`（陈旧尾值污染） | 只做 `T.copy` 子视图写而未限归约范围 | 非对齐用例静默错值 | 必须传 `size=[tile_m,tile_k,tile_n]`；§1.6.1 M1 `stale_tail_guard` 可检出（发散 ≈1.218e3） |
| K 循环体内条件构造崩溃（CG-2026-0001） | 死 tile 守卫误放入 K 循环内 | 编译 SIGSEGV | 守卫 `if` 放 `T.serial` 体内、K 循环之外 |
| **bishengir-compile flaky SIGSEGV**（CG-2026-0033） | `num_pid_n>1` + `num_local_tasks>1` 嵌套 scf | 编译期 err code -11（无 stderr） | 抬升 `RLIMIT_STACK` 至硬上限 + 有界重试（重建工厂） |
| `load_nd2nz` 跨步区域静默错值（CG-2026-0004） | B[e, n0:n0+tn, kb:kb+tk] 为跨步区域 | 静默错数 | 统一用 `T.copy` **slice 形态**（bit-exact）；Expert 路径优先 T.copy 而非 load_nd2nz |
| auto-multi-buffer 重排显式 buffer（CG-2026-0012 / TRAP-UB-dynsubview-dominance） | Expert 显式 alloc 结构 | codegen 变量找不到 / 结构破坏 | `pass_configs` 关闭 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION` 与 `NPUIR_ENABLE_AUTO_MULTI_BUFFER`（§7.3） |
| 越界写尾块 | 尾块未按动态 extent 裁剪 | 写坏下一专家行 | `store_fixpipe([tile_m,tile_n])` 严格裁剪（§5.4） |
| 调度表契约不一致 | 前导 kernel 的 `max_tiles`/`block_m`/行偏移单位与本核不符 | 寻址错位、错值 | 两 kernel 共享 `max_tiles = numel//block_m + E` 与 `block_m`；`tile_row_offsets` 单位为行（§4.1 契约） |

### 9.3 特殊场景处理

- **偏斜分布**：大量专家仅 1 行 → tile 严重不满（1/64）；动态 extent 保证只搬 1 行、只算 1 行。
- **空专家**（$s_e=0$）：`cnt=0`，不产生 tile（调度表天然跳过）。
- **极大规模**：核内串行 `num_local_tasks` 可达 ~1076（prefill down：21504/20）——循环边界静态，启动
  开销摊薄（D-3 探针 P-3 local_tasks=3 已实测该形态）。
- **极小 shape**（smoke numel=64）：逻辑核数 5 < 物理核数，仅 5 核参与，正确性优先。
- **两 kernel 协同**：前导 kernel 先 launch 完成，无 CPU 回读；`total_tiles` 全程设备侧。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（两 kernel 固定开销）**：launch + 表 GM 往返（≈6.1 KB）+ 跨 kernel barrier 的精确墙钟未实测
  （§1.6.0 roofline 标注「未实测假设 + 估算依据」）；Stage 4 实测，若 > B 访存 1% 则评估融合可行性
  （但当前工具链无融合路径，CG-2026-0026/0010）。
- **R-2（`T.Pipelined` vs `T.serial`）**：先例正对照与 D-3 探针均以 `T.serial` K 循环跑通；
  `T.Pipelined(num_stages=2)` 与 Expert 单 Cube scope 的组合稳定性待实测，不稳定则保持 `T.serial`（§6.3）。
- **R-3（动态 extent 分形对齐）**：动态 extent 的 `tile_m` 非 16 整数倍时 `T.copy`/`store_fixpipe` 的 NZ
  分形对齐行为 —— Stage 3 用 L0-1/L0-4 验证（D-3 探针已用非 16 倍数 `tile_m`（如 skew 下单 token 行）
  跑通，倾向可用）。
- **R-4（NPU L2 对 B 权重的实际复用率）**：B 是否单遍 —— 影响是否达到 §1.6.0 的 12.5 ms 下界；Stage 4 实测。
- **R-5（调度表对接契约）**：前导 kernel（独立单元）的 `max_tiles`/`block_m`/表 dtype/行偏移单位必须与
  本核一致；集成时核对（§4.1）。
- **R-6（`max_tiles` 死 tile 比例）**：`max_tiles = numel//block_m + E` 上界在偏斜分布下的死 tile 比例
  （守卫成本）——目前可忽略。
- **R-7（CG-2026-0033 flaky 编译）**：`num_pid_n>1 + num_local_tasks>1` 形态的编译不确定性；Stage 3
  须内置 `RLIMIT_STACK` 抬升 + 有界重试（§9.2），并记录实际重试次数。

### 9.5 能力缺口登记（Gap Rule）

- **CG-2026-0026**（codegen / BishengIR：`hivm.hir.sync_block_set` 属性名 `syn_instr_mode` 与随附
  `bishengir-compile` 期望的 `sync_instr_mode` 不一致）——**阻断一切跨引擎同 block 交接**；本设计据此把
  调度外移（§0.6 R1）。（已登记）
- **CG-2026-0010**（codegen / Developer：persistent + gemm + 整型索引段崩溃）——本设计据此选 Expert 单
  Cube scope。（已登记，occurrences=2）
- **CG-2026-0027**（codegen / Developer Vector scope：标量消费向量产出不同步）——本设计据此不混用
  「向量产出 → 标量消费」数据流（调度外移）。（已登记）
- **CG-2026-0033**（BishengIR：嵌套 scf（serial 任务循环 + 守卫 + 动态列偏移）flaky SIGSEGV）——
  **本任务独立识别（D-3 探针 P-2 首次编译 err code -11，重试通过）**，与先例
  `_moe_grouped_gemm_kernel` 的 `TRAP-BISHENGIR-FLAKY-SIGSEGV.py` 构成**第 2 个独立任务** → `recurring`；
  已在本任务登记至 `.agents/evolution/capability-gaps.md`（含 proposal：BishengIR 迭代式遍历替代递归）。
  对外建 issue 需用户批准。
- 本版设计的绕法（前导 kernel 产表 + Expert Cube 读表）**无需跨引擎同步**，从而在现有工具链可实现；
  上列缺口保持 open/recurring，待工具链升级后可按 kb_stale_check 重验。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/
├── DESIGN.md                                   # 本设计文档
├── verify_equiv.py                             # §1.6.1 等价性机器验证脚本（D-1）
├── history_version/
│   ├── design_probe_persistent_gemm.py         # D-3 设计期探针（设备实测）
│   └── design_probe_persistent_gemm.log        # 探针运行日志（4 用例全 PASS）
└── RETROSPECTIVE.md                            # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核；底座 = Expert 单 Cube scope + 外部调度表） |
| `verify_equiv.py` | 已完成 | 等价性验证（M1/M2 全 EQUIV_PASS；guard 发散确认 `size=` 不可省） |
| `history_version/design_probe_persistent_gemm.py` | 已完成（D-3） | Expert 单 Cube persistent grouped GEMM 底座探针（4 用例全 PASS） |
| 实际算子实现 `_persistent_grouped_gemm_v2_kernel.py` | **待产出**（Stage 3） | 按 §3.3 伪代码实现（Expert 单 Cube scope 读表形态） |
| 前导 kernel（调度表产出者） | 独立单元 | 须产 3 张 GM int32 表（对接契约 §4.1） |
| 测试 | 待实现 | 复用 `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` + §8.3 |

### 10.3 命名规范

- 项目目录：`examples/grouped_gemm_persistent_3wg/`
- 算子目录：`examples/grouped_gemm_persistent_3wg/_persistent_grouped_gemm_v2_kernel/`
- 实现文件：`_persistent_grouped_gemm_v2_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ D-3 探针
2. ⬜ 本 GEMM 核实现（`_persistent_grouped_gemm_v2_kernel.py`，Expert 单 Cube scope 读表）+ L0 精度比对
3. ⬜ 前导调度 kernel（独立单元）与两 kernel 协同测试（对接契约 §4.1）+ 完整分层套件

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/grouped_gemm/grouped_gemm_persistent_3wg.py`
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/grouped_gemm_persistent_3wg/`
  （`_grouped_gemm_persistent_3wg_kernels.py` + `grouped_gemm_persistent_3wg.py` + `.migration_meta.json`）
- 源规格 / 测试 / bench（等价 op，本单元无独立 op 条目）：
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoeGroupedGemmNopadFwdOp`）、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad.py`
- 本仓 NPU 可行底座先例：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`；
  调度表产出者 `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`
- 同类实现：`examples/deepseek_v32/fp8_lighting_indexer.py`、`examples/gemm/matmul_dynamic_shape.py`、
  `examples/mixcv/mixcv_mixkernel.py`、`examples/gemm/example_gemm.py`
- API 文档：`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`、
  `docs/Tilelang.language/内存操作/T.store_fixpipe.md`、`docs/Tilelang.language/比较操作/T.min.md`、
  `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/algorithm-research.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/INDEX.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/attention.md`（PL-1.16）
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0001/0004/0010/0012/0026/0027/0033）
