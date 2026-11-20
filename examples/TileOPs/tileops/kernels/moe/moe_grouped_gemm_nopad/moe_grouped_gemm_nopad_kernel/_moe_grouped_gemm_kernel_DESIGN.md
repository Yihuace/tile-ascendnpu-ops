# _moe_grouped_gemm_kernel 算子设计文档

> 迁移任务（harness，developer）。源算子：`tileops/kernels/moe/moe_grouped_gemm_nopad/moe_grouped_gemm_nopad.py`
> 中的 `_moe_grouped_gemm_kernel`（GPU/CUDA target）。本项目目标：`target="npuir"`（Ascend NPU）。
> 同名 wrapper（Stage 0 抽取件）位于 `examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/`。
> 迁移决策（§0）驱动 §1–§11 全部设计；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
>
> **配套函数**：同文件的 `_tile_scheduler_kernel`（调度表产出者）由胞弟单元
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/` 设计；本设计的 GEMM 核**消费**其输出表，
> 两函数的对接契约见 §0.6 R1 与 §4.1。

---

## 修订说明（revision 2，相对 design_v1）

本版为 **Stage 3 实施期返回 `[DESIGN_ERROR]`** 后的架构修订（revision_index=2）。v1 的设计底座
（「调度器并入 GEMM 核 + 零跨核同步 + Developer persistent」）经实证**不可实现**；本版把底座换成
**可实现的架构**，并**保留 v1 全部仍成立的结论**（动态 extent 精确裁剪、`T.gemm(size=)` 归约范围、
NT `b_transpose`、专家聚簇任务序、固定核数核内串行、B 访存主导下界估算）。

| # | 来源 | 问题 | 本版调整 | 为何不会再犯 |
|---|------|------|----------|--------------|
| 1 | 阻塞 | **Developer 主选**（单 Scope persistent `T.serial` + `T.gemm` + 整型索引段）在 lowering 期 **segfault**；且 Developer 下动态 extent `T.gemm` **静默错值**（1.04e-2 vs Expert 9.5e-7）。扩展 CG-2026-0010 触发面 | 删除 Developer 主选；编程模式改为 **Expert（`T.Scope("Cube")` Cube-only）**（§2.1） | 编程模式以本工具链实测可编译/可跑为准；结构落在 CG-2026-0010 崩溃域即判 Expert |
| 2 | 阻塞 | **Expert 回退**（Vector→Cube 同 block 交接）被工具链级缺陷阻断：`T.sync_block_set/wait` codegen 出 `syn_instr_mode`，`bishengir-compile` 解析 `sync_instr_mode` → `error: custom op 'syn_instr_mode' is unknown`；官方 `examples/mixcv/mixcv_mixkernel.py` 复现。已登记 CG-2026-0026 | **调度表不内联**：由前导 kernel `_tile_scheduler_kernel` 产出并落 GM，GEMM 核读表执行；**无跨引擎同 block 交接、无 in-block 同步**（§0.6 R1 / §3.3） | 任何跨引擎（AIV↔AIC）同 block 交接在当前工具链不可编译；改用 kernel 间同 stream 自然顺序 |
| 3 | 阻塞 | Expert **Cube scope 内无法维护调度状态**（loop-carry 崩溃 / 多地址 GM scalar store 被消除 / L1 scalar 读崩溃） | Cube scope 只做「**读 GM 表 + 读 GM 标量 + GEMM + 写回**」；状态全部来自 GM（表 + `true_sizes`/`true_offsets` 标量读），不做 loop-carry、不写 GM 标量（§3.3；Stage 3 正对照 max_diff **9.537e-07**） | 与已验证正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` 同形态 |
| 4 | 收益撤回 | §1.6.0 v1 的「**融合单核**」收益主张（少 3 张表 GM 往返 + 少一次 kernel launch）被工具链推翻 | §1.6.0 R1 候选表把「融合单核」标为**工具链不可实现（撤回）**，选定算法改为「**独立调度表 + Expert Cube 读表**」；收益重估（§1.6.0/§1.6.1） | 收益主张以实现性为前提；不可实现即撤回，不得纸面保留 |
| 5 | 协调 | 胞弟 `_tile_scheduler_kernel` 原设计结论「NPU 上不应独立、内联进 GEMM 核」同样被推翻 | 本版定义两函数**分工与对接契约**（§0.6 R1 / §4.1）：调度改为**独立产出调度表**的形态（胞弟须据此修订其 §0.6 R1 / §1.6.3 L2） | 契约显式化，避免两设计互相矛盾 |
| 6 | 保留 | v1 仍成立结论 | 动态 extent 精确裁剪、`T.gemm(size=[tile_m,tile_k,tile_n])` 限定归约范围、NT `b_transpose`、专家聚簇任务序、固定核数核内串行、B 访存主导下界估算 | 逐条保留（§1.6.0/§1.6.1/§4/§5） |

> **Stage 3 实证底座**：Expert `T.Scope("Cube")` + 外部提供的调度表 + 动态 extent 取数 +
> `T.gemm(size=[tile_m,tile_k,tile_n], b_transpose=True)` + `store_fixpipe([tile_m,tile_n])` 已实测
> max_diff **9.537e-07**（no-pad 尾块裁剪正确）；复现脚本
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`
> 的 `positive_control()`。唯一不可用环节是 Vector→Cube 的同 block 同步（本版已回避）。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 references/migration-analysis.md（三问框架 + 耦合性判定 + 重设计）。
> 本任务范围为 **`_moe_grouped_gemm_kernel`**（GEMM 主核）；配套的 `_tile_scheduler_kernel`（tile 调度）
> 由胞弟单元设计，其输出**作为本核的输入表**（对接契约见 §0.6 R1）。
> **分类口径**：调度 prompt 标「MoE 混合 Cube+Vector」，但本算子数据通路 **Cube 主导、无逐元素浮点向量工作**
> （详见 §2.1 分类口径澄清）。

### 0.1 源算子语义（做什么）

**数学语义**（MoE no-pad 分组 GEMM，NT 布局）：

对每个专家 $e\in[0,E)$，设其 token 数为 $s_e$（`true_sizes[e]`）、在紧致（tight，无 pad）行布局中的起
始行偏移为 $o_e$（`true_offsets[e]`），则

$$
C[o_e + r,\; :] \;=\; A[o_e + r,\; :] \cdot B[e]^{T}, \qquad r \in [0, s_e)
$$

即 $C[i,n]=\sum_{k} A[i,k]\cdot B[e_i,n,k]$，其中 $e_i$ 是第 $i$ 行所属专家。等价地：
$C = \text{concat}_e\big(A[o_e:o_e+s_e] \cdot B[e]^T\big)$，且 $\sum_e s_e = \text{numel}$
（有效输入下每行恰属于一个专家，无未覆盖行）。

- **规约语义**：规约轴 = $K$（特征维），完整求和，无 keepdim 概念；**累加顺序**：源在
  `C_local`（fp32）上按 `block_k` 分块累加（`T.gemm(..., transpose_B=True)`），块间为顺序累加。
  重设计改变分块粒度会改变累加分组，需评估 fp16/bf16 舍入（§0.6 语义保持论证）。
- **dtype 语义**：输入 A、B 为 fp16/bf16；中间累加 `accum_dtype="float"`（fp32）；输出 C 与输入同
  dtype（fp16/bf16）。即 **fp32 累加 + 输出回落到输入 dtype**（源 `out_idx=[2]`、`C_shape=(numel,N)`）。
- **边界语义**：`numel`（T×top_k）一般不被 `block_m` 整除；每专家尾块行数 $s_e \bmod block\_m \neq 0$
  普遍存在；$N$ 与 `block_n` 不一定整除（smoke 用例 $N=128<block\_n=256$）；$K$ 与 `block_k` 可能不整除
  （full 用例 $K=96$）。源对 M/行方向用 epilogue 守卫（`i < actual_rows`），对 K 尾用
  `T.if_then_else(...,0)` 零填充，对 N 尾用 `j < actual_cols` 守卫。**no-pad 紧致布局的关键约束**：
  一个 M-tile 绝不跨专家（tile 按专家内切分），故尾块多余行属于**下一个专家的行**，若越界写入会破坏
  下一个专家的结果 —— 尾块必须精确裁剪。
- **host 侧语义**：`torch.nn.functional.pad(A, (0,0,0,block_m))`（K 对齐时）为**实现手段而非语义**
  （仅为了让 TMA 拷贝落在合法内存），NPU 侧不得保留（ascend-constraints.md §4 禁止 host 侧改动输入真实内容）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（优先移植源测试的 `_ref_grouped_gemm_nopad`），
不得复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| A | (numel, K) | 同 dtype | 紧致 permuted 激活，多专家行首尾相接，无 pad |
| B | (E, N, K) | 同 dtype | 专家权重，NT 布局（对 B 转置后右乘） |
| true_sizes | (E,) | int32 | 每专家真实 token 数 $s_e$ |
| true_offsets | (E,) | int32 | 每专家在 A 中的紧致起始行 $o_e$ |
| tile_expert_ids | (max_tiles,) | int32 | 调度器输出：每个 M-tile 的专家号（死 tile = -1） |
| tile_row_offsets | (max_tiles,) | int32 | 调度器输出：M-tile 在专家内的行偏移（已乘 block_m） |
| total_tiles | (1,) | int32 | 有效 M-tile 总数（设备侧，不回落 CPU） |
| **C（输出）** | **(numel, N)** | **同 dtype** | 迁移必需字段：源 `out_idx=[2]`，`C_shape=(numel,N)`；非转置布局、行连续 |

### 0.3 实现算法解读（怎么算）

源为**两 kernel 两阶段**（host 侧 `forward` 顺序调用，无 CPU 同步）：

- **Phase 1 `_tile_scheduler_kernel`**：`true_sizes[E]` → 每专家 tile 数 `ceil(s_e/block_m)` 的
  排他前缀和 `s_cum[E+1]`（thread 0 串行扫描写入 SMEM）→ `total_tiles=s_cum[E]`；随后并行对每个
  tile_id 二分查找落点，写出 `tile_expert_ids[tile_id]`、`tile_row_offsets[tile_id]`（死 tile 写 -1）。
- **Phase 2 `_moe_grouped_gemm_kernel`**：网格 = `max_tiles × ceil(N/block_n)` 一维展平；`pid` 经
  GROUP_SIZE_M 重排映射到 `(bx, by)`；`bx >= total_tiles[0]` 的死 CTA 早退；有效 CTA 做单个 M-tile，
  经 GM 读 `tile_expert_ids[bx]` / `tile_row_offsets[bx]` / `total_tiles[0]` 定位。

**计算步骤分解**（源码每个计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P1-a | `s_cum[e+1]=s_cum[e]+ceil(s_e/block_m)`（thread0 串行前缀扫描） | true_sizes | s_cum[E+1] | 专家→tile 段映射（调度，非数值语义） |
| P1-b | 二分查找 `s_cum`：`mid=(lo+hi)>>1; lo/hi 更新` | tile_id, s_cum | tile_expert_ids, tile_row_offsets | tile→(专家, 行偏移) |
| P2-1 | `bx=pid//num_pid_n; by=pid%num_pid_n`（+GROUP_SIZE_M 重排） | pid | bx, by | tile 坐标 |
| P2-2 | `expert_id=tile_expert_ids[bx]`；`m_start=true_offsets[expert_id]+tile_row_offsets[bx]`；`n_start=by*block_n` | 调度表 | m_start, n_start | 该 tile 在 C/A 中的行/列起点 |
| P2-3 | `actual_rows=min(block_m, true_sizes[e]-row_in_expert)`；`actual_cols=min(block_n, N-n_start)` | true_sizes | 有效行/列数 | 尾块边界 |
| P2-4 | `T.copy(A[m_start:m_start+block_m, k*block_k:(k+1)*block_k] → A_shared)` 及 B 对应 tile | A, B | A_shared, B_shared | 分块取数 |
| P2-5 | `T.gemm(A_shared, B_shared, C_local, transpose_B=True)` | A/B tile | C_local(fp32) | $C=\sum_k A\cdot B^T$ 的块累加 |
| P2-6 | epilogue：`if i<actual_rows and j<actual_cols: C[m_start+i, n_start+j]=C_local[i,j]` | C_local | C(GM) | 尾块裁剪写出 |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[A] --T.copy(TMA)--> SMEM[A_shared(block_m,block_k)]  ┐
GM[B] --T.copy(TMA)--> SMEM[B_shared(block_n,block_k)]  ├─ T.gemm(transpose_B) --> register C_local(block_m,block_n,fp32)
                                                        ┘                                   │
                                                    GM[C] <── 守卫式 T.Parallel 逐元素写出 ────┘
调度表(GM) --1 次读 total_tiles + 标量读 tile_expert_ids/row_offsets--> m_start/n_start/边界
host padding: torch F.pad(A) --> GM[A_padded]（额外 block_m×K 拷贝；K 对齐路径）
```

**循环与并行结构**：Phase 2 grid = `max_tiles × num_pid_n` 个 CTA（1D，GROUP_SIZE_M 重排）；每 CTA：
`TC.Pipelined(ceildiv(K,block_k), num_stages)` 的 K 累加循环 + `T.Parallel(block_m,block_n)` epilogue。
Phase 1 grid = `ceildiv(max_tiles, threads)`，threads=256，每 thread 一个 tile。

**host 侧逻辑**：
1. K 对齐时 `A = F.pad(A, (0,0,0,block_m))`（Python 侧，改动输入张量）——**NPU 禁止**。
2. `max_tiles = numel//block_m + E`；`tile_scheduler` / `gemm` 两个 `@tilelang.jit` 工厂按 config 实例化。
3. 两次 kernel 调用顺序执行；`total_tiles` 始终留在设备侧（`.item()` 被规避）。

**编译配置**：`pass_configs={TL_ENABLE_FAST_MATH: True}`；`compile_flags=["-O3","-DENABLE_BF16"]`；
`out_idx=[2]`；`A_shared/B_shared` 用 `make_swizzled_layout`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM tiling（A_shared/B_shared 分块 `T.copy`） | 减少 GM 重复访问 | 分块驻留 shared memory | shared memory（bank/容量） | 硬件强相关（NPU 有 L1/UB 等价） |
| O2 | Tensor Core（`T.gemm`，WGMMA） | 矩阵算力 | tensor core MMA | tensor core | 硬件强相关（NPU 有 Cube 等价） |
| O3 | TMA 异步拷贝 + `T.Pipelined(num_stages)` + WarpSpecialized（producer/consumer warpgroup） | 访存/计算重叠 | 异步引擎多级流水 | TMA + async copy | 硬件强相关（NPU 有 MTE 等价） |
| O4 | Swizzled layout（`make_swizzled_layout`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用，NZ 布局代替） |
| O5 | GROUP_SIZE_M tile 重排 | B 权重 L2 复用、避免跨波驱逐 | 按 group 组织 M-tile 处理全 N 列 | L2 cache | 硬件强相关（NPU L2 可控性弱） |
| O6 | 死 CTA 早退 + 保守 `max_tiles` 上界（零 CPU 同步） | 规避 `total_tiles` 回落 CPU 的同步开销 | grid 超发 + 运行时单比较早退 | 廉价 CTA 启动 | 模式可移植（NPU 一维核需重新设计分核） |
| O7 | 专家 tile 调度器（前缀扫描 + 二分查找） | 消除每 CTA 的 O(E) 专家扫描 | 预计算 tile→(expert,row) 表（**独立 kernel 产出 GM 表**） | 无（纯算法/索引层） | 算法层可移植；**独立 kernel 形态保留**（见 §0.5/§0.6 R1） |
| O8 | 预判路径分派（K 对齐走 `T.copy`/TMA 快路；否则谓词零填充慢路） | 对齐时走无谓词快路 | 编译期常量分支 | TMA OOB 零填充 | 硬件强相关（TMA 特性） |
| O9 | 谓词零填充（`T.if_then_else(...,0)`）处理 K/N 尾块 | 非整除 shape 正确性 | 越界读补 0（乘 0 不影响和） | 无（数学恒等） | 算法层可移植 |
| O10 | epilogue 守卫（`T.Parallel` 内 `if i<actual_rows and j<actual_cols`） | 尾块精确裁剪，防越界写坏邻专家 | 逐元素条件写 | 无 | 算法层可移植（NPU 用动态 extent 等价实现） |

> 识别不出来 ≠ 不存在。以上 O1–O10 为源码显式优化，逐项在 §0.5 判定，未被识别者会被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（按专家分组 NT 矩阵乘、fp32 累加、同 dtype 输出） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| I/O 契约（A/E,B/C 的 shape/dtype/行连续布局） | 语义 | 无 | — | **保留** | §0.2 |
| O1 SMEM tiling | 优化 | shared memory | 有（L1/UB） | **等价替换** | GM→L1（Cube 操作数）分块搬运；L1 512KB/UB 192KB/32B 对齐（ascend-constraints.md §1、migration-analysis.md §5.3） |
| O2 Tensor Core mma | 优化 | tensor core | 有（Cube） | **等价替换** | `T.gemm`→`hivm.hir.mmadL1`（T.gemm.md §3）；分形 M/N≥16、K≥32（ascend-constraints.md §1） |
| O3 TMA + Pipelined + WarpSpecialized | 优化 | TMA/async copy/warpgroup | 有（MTE 异步 + `T.Pipelined`） | **等价替换** | GM→L1 经 MTE2/ND2NZ + `T.Pipelined(num_stages)` 双缓冲；stage 数按 L1 容量重算（migration-analysis.md §5.3） |
| O4 swizzled layout | 优化 | SMEM bank 结构 | 无（NPU 无 SMEM bank conflict） | **舍弃** | NZ 分形布局由 `load_nd2nz`/前端处理，手动 swizzle 无意义（migration-analysis.md §5.3） |
| O5 GROUP_SIZE_M L2 tile 重排 | 优化 | 可编程 L2 cache | 部分（L2 存在但无显式编程接口） | **重新设计** | 改为「按专家聚簇的任务顺序」提升 B 时间局部性；无显式 L2 控制（见 R2，依据 negative-claim-evidence.md §2 标「未文档化假设」） |
| O6 死 CTA 早退 + 超发 grid | 优化 | 廉价 CTA 启动 | 无一维核超发会被串行调度（超发开销大） | **重新设计** | 一维 `T.Kernel` + 固定核数 + 核内 `T.serial` 处理多逻辑 tile（core-split-strategy.md §1 要素③）；死 tile 用单比较守卫（非 Pipelined 内，规避 CG-2026-0001，§0.6 R3） |
| O7 专家 tile 调度器 | 算法+优化 | thread 级并行二分查找 | 有（算法层），独立 kernel 形态保留 | **保留（独立产出 GM 表）** | 源「独立调度 kernel 写 3 张 GM 表 + GEMM 核读表」的数据流**保留**；NPU 侧调度 kernel 由胞弟 `_tile_scheduler_kernel` 重设计（Vector 整数索引），GEMM 侧经 GM 读表（§0.6 R1）。**依据**：CG-2026-0026 —— 任何跨引擎同 block 交接在当前工具链不可编译，故不得把调度内联进 GEMM 核 |
| O8 对齐/非对齐编译期分派 | 优化 | TMA OOB 零填充 | 无 TMA，但 NPU 有动态 extent | **重新设计** | 统一走**动态 extent** 取数（`tile_m/tile_n/tile_k` 用 `T.min` 计算），不再需要零填充与谓词（§0.6 R2）；依据 T.copy.md §2.2.2（动态 extents 尾块支持） |
| O9 谓词零填充（K/N 尾） | 算法 | 无 | 有（数学恒等） | **重新设计/等价替换** | 动态 extent 只取有效区（越界贡献恒为 0，数学等价）；§1.6.1 M2 已机器验证 |
| O10 epilogue 守卫 | 算法 | 无 | 有（动态 extent store） | **等价替换** | `store_fixpipe` 用 `[tile_m, tile_n]` 动态尺寸只写有效区，等价于逐元素守卫（mixcv_mixkernel.py L57；T.copy.md §2.4 示例 4） |
| host `F.pad(A)` | 优化 | TMA 内存安全 | 无（host 改动输入被禁止） | **舍弃** | ascend-constraints.md §4：host 侧禁止改动输入真实内容/触发 padding；改由动态 extent 取数解决 |
| 核内调度折叠（v1 设计引入，非源） | 优化/结构 | 无 | **无可用的跨引擎同 block 交接原语** | **舍弃** | CG-2026-0026（`sync_block_*` 属性名不匹配，官方 mixcv 样例复现）+ CG-2026-0010（Developer 单 Scope 崩溃）；本版据此改为「调度表不内联」（§0.6 R1） |

> 每项处置依据见「NPU 对应方案 / 依据」列。舍弃项（O4 swizzle、host pad、核内调度折叠）理由已写明。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**独立调度表产出（前导 kernel `_tile_scheduler_kernel`）+ Expert Cube-only
GEMM 读表 + 动态 extent 取数 + 专家聚簇任务序**）。

**重设计项 R0：编程模式——Developer 主选 → Expert Cube-only（迁移侧形态约束）**

- **背景**：源为 GPU 单 kernel（CUDA），无 Developer/Expert 概念。v1 主选 Developer（auto CV-split
  + persistent）经 Stage 3 实证在 lowering 期 **segfault**（CG-2026-0010 扩展触发面），且 Developer 下
  动态 extent `T.gemm` 静默错值（1.04e-2）。
- **NPU 新形态**：**Expert**（`T.Scope("Cube")`），核内只做 `T.alloc_L1`/`T.alloc_L0C` + `T.copy` 取数 +
  `T.gemm` + `store_fixpipe`；**单一 scope，无 Vector 段、无跨引擎同步**。
- **语义保持论证**：编程模式不改变数值语义（同一 `T.gemm`、同一 fp32 累加、同一动态 extent 裁剪）；
  与 v1 Expert 形态数值等价（§7 正对照 max_diff 9.537e-07）。
- **依据**：`.agents/evolution/capability-gaps.md` CG-2026-0010；pattern-library traps-runtime
  TRAP-DEVMODE-PERSIST-GEMM 与 attention.md PL-1.16。

**重设计项 R1：调度承载方式（O7）——保留源「独立调度 kernel + GM 表」，撤回 v1「并入 GEMM 核」**

- **源方案**：独立 `_tile_scheduler_kernel`，thread 0 串行前缀扫描 `s_cum`，其余 thread 并行二分查找，
  输出 `tile_expert_ids/tile_row_offsets/total_tiles` 三张 GM 表；GEMM 核经 GM 读这三张表做 O(1)
  tile→(expert,row) 映射。性能意图：消除每 CTA 的 O(E) 专家扫描 + 保持 `total_tiles` 在设备侧（零 CPU 同步）。
- **v1 方案（已撤回）**：把调度器**内联**进 GEMM 核的 Vector scope，表不落 GM，核内 Vector→Cube 交接
  （Developer 主选为 auto CV-split，Expert 回退经 GM scratch + `sync_block_set/wait`）。收益主张：
  「少 3 张表 GM 读写往返 + 少一次 kernel launch」。
- **撤回论证**：Stage 3 实证两条独立阻断——
  (i) Developer 单 Scope（persistent + `T.gemm` + 整型索引段）lowering 期 segfault（CG-2026-0010）；
  (ii) Expert 回退的 Vector→Cube 同 block 交接被 **CG-2026-0026** 阻断：`T.sync_block_set/wait` codegen
  出 `syn_instr_mode`，随附 `bishengir-compile` 解析 `sync_instr_mode` → `error: custom op
  'syn_instr_mode' is unknown`，**官方 `examples/mixcv/mixcv_mixkernel.py` 复现同一报错** ⇒ 一切跨引擎
  同 block 交接均不可编译；去掉同步则出现竞态（同 kernel 多次运行 max_diff 取值集合含 9.5e-7 / 1.26e-2 / 全零）。
  故 v1 的「融合单核 + 零跨核同步」形态**无可用工具链路径**，收益主张随之撤回。
- **NPU 新算法**：**保留源的独立调度 kernel 数据流**——
  1. 前导 kernel `_tile_scheduler_kernel`（胞弟单元，Vector 整数索引）读 `true_sizes`，产出
     `tile_expert_ids[max_tiles]`（每 M-tile 专家号，死 tile = -1）、`tile_row_offsets[max_tiles]`
     （专家内行偏移，已乘 block_m）、`total_tiles[1]`，全部落 **GM**；
  2. 本 GEMM 核（Expert Cube-only）按 `bx = tile_id // num_pid_n`、`by = tile_id % num_pid_n` 读
     `total_tiles[0]` 与 `tile_expert_ids[bx]` / `tile_row_offsets[bx]`，算 `m_start/n_start/tile_m/tile_n`
     后执行 NT GEMM；
  3. 两 kernel 在**同一 stream 顺序 launch**：前导 kernel 先完成、GEMM 核再执行，二者之间的顺序由
     kernel 间（同 stream）自然顺序保证，**无需任何 in-block 同步**；`total_tiles` 全程留在设备侧
     （零 CPU 同步意图承接）。
- **语义保持论证**：调度表内容与源**逐元素一致**（同一 `cum` 定义、同一 `searchsorted` 落点、同一哨兵
  `(-1,0)`），故 tile→(expert,row) 映射逐位一致，GEMM 数值与源一致；变化仅在**求值时机**（前导 kernel
  一次物化 vs v1 核内重算）。边界：$s_e=0$ 的专家 `cnt=0`，不产生 tile；`total_tiles ≤ max_tiles`
  保证 `tile_expert_ids` 长度合法；死 tile 由单比较 `if bx < total_tiles[0]` 跳过。
- **对接契约（本核消费）**：见 §4.1；`max_tiles = numel // block_m + E` 与 `block_m` 为两函数共享的
  编译期常量；表 dtype 全 int32；`tile_row_offsets` 的单位是**行**（已乘 block_m），不是 tile 序号。
- **收益重估**：相对 v1 的内联形态，本形态多出 (i) 一次 kernel launch；(ii) 3 张表 GM 写 + 读
  （`(2·max_tiles+1)·4B`，prefill 下 max_tiles=768 → ≈6.1 KB）；(iii) 一次跨 kernel barrier。三项合计
  ≪ 1 ms，相对 B 权重访存 15.03 GB（§1.6.0 估算 ≈11.9 ms）**可忽略**；代价是换取**可实现性**。
  详见 §1.6.0 roofline 与 §1.6.1 收益重估。

**重设计项 R2：取数/写回从「谓词零填充 + 守卫」改为「动态 extent 精确裁剪」（O8/O9/O10）**

- **源方案**：K 对齐时 host `F.pad(A)` + TMA；非对齐时 `T.if_then_else` 对越界 K/N 零填充；epilogue
  用 `T.Parallel` 内 `if i<actual_rows and j<actual_cols` 守卫。
- **NPU 新算法**：所有 GM→L1 取数与 L0C→GM 写回使用**动态 extent**：
  `tile_m = T.min(block_m, s_e - row)`、`tile_n = T.min(block_n, N - n_start)`、
  `tile_k = T.min(block_k, K - k*block_k)`；`T.copy` 的切片 extents 用上述动态值（T.copy.md §2.2.2 尾块支持），
  **K 方向的动态 extent 必须同时传入 `T.gemm(..., size=[tile_m, tile_k, tile_n])` 限定归约范围**（否则
  缓冲 `[tile_k, block_k)` 的陈旧值会被按静态 `block_k` 归约计入累加——这是本重设计的正确性关键，
  见 §3.3 注与 §1.6.1 M2），`store_fixpipe` 用 `[tile_m, tile_n]`（mixcv_mixkernel.py L57 先例）。
  M/N/K 三方向尾块统一由动态 extent 处理，**不再需要 host padding、谓词零填充与逐元素守卫**。
- **语义保持论证**：越界贡献在数学上恒为 0（零填充或被裁掉的积项），故「只算有效区」与「零填充后全算再裁剪」
  数值等价——**该等价性以「归约范围 = 有效 `tile_k`」为前提**，由 `size=[tile_m,tile_k,tile_n]` 落实；对每专家
  NT 矩阵乘在 fp64 参照下逐元素对照（含 K 分块 + 陈旧尾值注入），最大相对差 1.003e-11（= fp64 matmul 重排舍入
  量级，远小于任何语义差异），4/4 用例 EQUIV_PASS（§1.6.1 M2）；`stale_tail_guard` 进一步确认若省略 `size`
  则发散（max_abs_diff ≈ 1.218e3），证明该前提不可省。**no-pad 关键边界**：尾块多余行属于下一专家，
  动态 extent 只写 $[0,tile_m)$ 行，杜绝越界写坏邻专家（源守卫的等价实现）。dtype：取数/写回 dtype 一致，
  `store_fixpipe` 走 fp32→fp16/bf16 量化（T.store_fixpipe.md），与源 fp32 累加后回落输入 dtype 一致。
  **Stage 3 正对照**：该形态已实测 max_diff 9.537e-07（`repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`）。

**重设计项 R3：任务划分与分核（O6）——从超发 grid + 死 CTA 早退改为固定核数 + 核内串行**

- **源方案**：grid = `max_tiles × num_pid_n`（可达数千 CTA），超发 + `bx>=total_tiles` 死 CTA 早退，
  换取高 warp 占用率与延迟隐藏。
- **NPU 新算法**：`T.Kernel(nums_kernel, is_npu=True)`，`nums_kernel = min(max_tiles*num_pid_n, P)`
  （P = 实查物理核数，见 §5.5）；核内 `for t in T.serial(num_local_tasks)`（`num_local_tasks` 为编译期常量）
  以块连续方式处理多逻辑 tile；死 tile 用 `if bx < total_tiles[0]` 单比较跳过（位于 `T.serial` 体内但
  **不在 `T.Pipelined` 体内**，规避 CG-2026-0001 的条件构造崩溃）。
- **语义保持论证**：分核只改变「哪个核处理哪个 tile」的调度，不改变 tile 内计算（同一 `T.gemm` +
  同一 fp32 累加 + 同一裁剪写回），故数值逐位一致（累加分组不变）。边界：`nums_kernel` 取编译期常量，
  循环边界静态；任务与核的映射为纯调度，无副作用。

**重设计项 R4：B 权重复用顺序（O5）**

- **源方案**：GROUP_SIZE_M 重排，使一个 group 的 M-tile 处理完全部 N 列，令 B 权重工作集落在 L2，
  避免普通 2D grid 的跨波驱逐。
- **NPU 新算法**：任务序采用 **专家聚簇**（同专家相邻，由调度表天然给出：tile_expert_ids 按专家前缀
  段序排列）+ **M-major**（同 M-tile 先走完全部 N 列）：flat task_id → `bx = task_id // num_pid_n`、
  `by = task_id % num_pid_n`；核内按块连续领取 task 区间，使同一核的连续任务尽量同专家、同 M-tile，
  提升 L1/L2 对 B tile 的复用。
- **语义保持论证**：纯任务排序，不改变任何 tile 的数值结果；累加分组与 dtype 路径不变。
  边界：排序对死 tile 判定的语义无影响。**依据缺口**：NPU 无显式 L2 编程接口、L2 容量/替换策略未在
  docs/ 记载（标注「未文档化假设」：收益来自访存时间局部性，实测收益待 Stage 4 验证；见 §9 风险点）。

**重设计项 R5：无（算法层）**。矩阵乘算法族本身（NT GEMM + fp32 分块累加）在 Cube 上有等价能力，
按 §0.5 为「保留/等价替换」，无算法层重设计。

### 0.7 标杆实现

- 源算子路径：`tileops/kernels/moe/moe_grouped_gemm_nopad/moe_grouped_gemm_nopad.py`
  （GPU 参考仓 `/tmp/opencode/TileOPs-ref`，函数 `_moe_grouped_gemm_kernel` / `_tile_scheduler_kernel`）。
- 源仓参考实现 / 测试基准：
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` 的 `_ref_grouped_gemm_nopad`
  （每专家 PyTorch NT matmul 循环，fp32 累加后回落输入 dtype）。
- 本仓 NPU 两 kernel 先例（结构参照，非数值基准）：
  `testing/autotune/anneal/moe_grouped_gemm_anneal.py`（`tile_scheduler_kernel_npu` 独立调度 kernel +
  `moe_gemm_kernel_npu` 读表 GEMM；注意其 `T.Kernel(max_tiles)` 超发形态**不采用**，见 §0.6 R3）。
- golden 以 §0.1 语义为唯一依据，优先移植上述源测试参考实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_moe_grouped_gemm_kernel`（MoE no-pad 分组 GEMM，NT 布局，`target="npuir"`）。

### 1.2 功能描述

对紧致（无 pad）排列的 MoE 激活按专家分组做 NT 矩阵乘：第 $e$ 个专家的 $s_e$ 行激活
乘以其权重矩阵的转置，输出与输入同形状、同 dtype。配套的 tile 调度功能由**前导 kernel
`_tile_scheduler_kernel` 独立产出**调度表并落 GM；本 GEMM 核**读表执行**（不内联调度，
见 §0.6 R1，理由 = 工具链阻断跨引擎同 block 交接 CG-2026-0026）。

### 1.3 数学公式

$$
C[o_e + r,\; :] = A[o_e + r,\; :]\, B[e]^{T},\quad r\in[0,s_e),\quad e\in[0,E),\quad \sum_e s_e=\text{numel}
$$

即 $C[i,n]=\sum_{k=0}^{K-1} A[i,k]\, B[e_i,n,k]$，$e_i$ 为第 $i$ 行专家号；等价于逐专家
$C_e = A_e B_e^{T}$ 的拼接。源清单 roofline：`flops = 2*numel*n*k`，
`bytes = (numel*k + E*n*k + numel*n)*elem_bytes`（`examples/TileOPs/tileops/manifest/moe.yaml`）。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源算法结构的差异已逐条注明）。

**两 kernel（与源同数据流）**：前导 kernel `_tile_scheduler_kernel` 产出调度表（GM），本 GEMM 核
（**Expert Cube-only**，单一 `T.Scope("Cube")`）读表执行 NT 分组 GEMM：

1. **调度表消费（Cube scope，标量读 GM）**：核内对每个逻辑 tile（`tile_id = core_id*num_local_tasks + t`）
   算 `bx = tile_id // num_pid_n`、`by = tile_id % num_pid_n`；读 `total_tiles[0]`，`bx >= total_tiles[0]`
   的死 tile 跳过；否则读 `expert_id = tile_expert_ids[bx]`、`row = tile_row_offsets[bx]`（GM 标量读）；
   由 `true_offsets[expert_id]`、`true_sizes[expert_id]` 算 `m_start/tile_m`。
   与源**结构差异**：源为超发 grid + 死 CTA 早退（每 CTA 一个 tile，数千 CTA）；本设计为**固定核数 +
   核内 `T.serial`**（§0.6 R3），调度表由**独立前导 kernel** 产出（§0.6 R1，撤回 v1 的核内内联）。
2. **GEMM 阶段（Cube scope）**：据调度元数据算 `n_start` 与动态 extent `tile_n/tile_k`，GM→L1 取 A/B tile
   （NZ 自动转换），`T.gemm(A_l1, B_l1, C_l0c, size=[tile_m,tile_k,tile_n], b_transpose=True, initC=(k==0))`
   在 K 维流水累加（fp32，**归约范围由 `size` 限制为动态 `tile_k`**，见 §3.3），最后 `store_fixpipe`
   （动态 `[tile_m,tile_n]`）写回 C；与源结构差异：**动态 extent 精确裁剪**替代源 host `F.pad` + TMA
   零填充 + epilogue 守卫（§0.6 R2），**专家聚簇任务序**替代 GROUP_SIZE_M L2 重排（§0.6 R4）。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[前导 kernel _tile_scheduler_kernel（胞弟单元，Vector 整数索引）]
  true_sizes[E] --前缀扫描/查找--> GM[tile_expert_ids(max_tiles) int32]
                                  GM[tile_row_offsets(max_tiles) int32]
                                  GM[total_tiles(1) int32]
        │  （同一 stream 顺序 launch：前导 kernel 完成后 GEMM 核执行，无 in-block 同步）
        ▼
[本 GEMM 核，Expert Cube-only]
  标量读 GM 表/标量 ──> (expert_id, row, m_start, n_start, tile_m, tile_n)
A[numel,K] ┐
B[E,N,K]   ┼─动态extent T.copy(GM→L1, ND2NZ)─> L1[a_l1/b_l1] ─T.gemm(b_transpose,size=[tile_m,tile_k,tile_n])─> L0C(fp32)
           │                                                                         │ store_fixpipe([tile_m,tile_n])
           └── (由 m_start/n_start/tile_* 决定取哪一块)                                ▼
                                                                                   C[numel,N]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（MoE no-pad 分组 NT 矩阵乘 + 其 tile 调度）的**算法族**。输入/源算法
> （两 kernel + 超发 grid + host pad）只是基线候选之一。深度分级：矩阵/多步/融合类 → 完整调研。
> **可实现性约束（前置）**：本工具链禁止跨引擎同 block 交接（CG-2026-0026），故任何「调度内联进
> GEMM 核的 Vector 索引段 + 同 block 交接给 Cube」的候选**不可实现**（CG-2026-0010 亦封死 Developer
> 单 Scope 绕法），调研在可实现族内择优。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | 基线（源算法） | 两 kernel：独立调度器（thread 并行二分）+ 超发 grid GEMM；谓词零填充 + epilogue 守卫；host `F.pad` | — | — | ✅ 基线 |
| C1 | **独立调度表 + Expert Cube-only GEMM 读表**（选定） | 前导 kernel 产 3 张 GM 表（保留源数据流）；GEMM 核 Expert Cube-only 读表；动态 extent；固定核数核内串行 | 数学恒等（调度内容同源；表内容逐位一致） | 消 host pad/零填充/守卫；无跨引擎同步（规避 CG-2026-0026） | ✅ **选定** |
| C2 | 融合单核（v1：调度内联进 GEMM 核，表不落 GM） | 核内 Vector 索引段 + 同 block Vector→Cube 交接 | 数学恒等（纯索引） | 理论少 3 表 GM 往返 + 少一次 launch | ❌ **撤回**：工具链不可实现（CG-2026-0026 / CG-2026-0010，官方样例复现） |
| C3 | 动态 extent 精确裁剪（含动态 K 归约范围） | 谓词零填充 + 守卫 → 只取/只写有效区 `tile_m/tile_n/tile_k`，`T.gemm(size=)` | 容差内等价（越界项恒 0） | 少搬越界冗余 + 消零填充算 + 消 host pad | ✅ 采纳（并入 C1） |
| C4 | NT 形式（`b_transpose=True`） | 直接以 B[N,K] 行主序做 `A·B^T`，不物化 B^T | 数学恒等 | 少一次转置/重排 | ✅ 采纳（并入 C1） |
| C5 | 无搜索映射（expert = task_id // max_tiles_per_expert） | 网格直接编码专家，免查找 | 数学恒等 | 省查找，但死 tile 比例暴增 | ✅ 对比（弃选） |
| C6 | 专家内 pad 到 block_m 整数倍 | 先 pad A 再算，末尾裁剪 | 数学恒等（补 0） | 消除 M 尾块特判，但 FLOPs/访存膨胀 | ✅ 对比（弃选） |
| C7 | split-K（K 分段多核累加） | K 维切分给多核，跨核累加 | 容差内等价（累加分组变化） | 增并行度，但引入跨核归并 + 访存倍增 | ✅ 对比（弃选） |
| C8 | host 侧调度（CPU 算表后下发） | 调度表在 host 生成 | 数学等价 | 需 `true_sizes` D2H + 表 H2D | ❌ 弃选：引入 CPU 同步，违反源「零 CPU 同步」意图；host 禁改输入（ascend-constraints §4） |

**R2 在线算法**：**有**，且是本题的天然形态。矩阵乘在 K 维的 `T.Pipelined` 分块累加**本身就是单遍
流式**（running 累加器 `C_l0c`，O(1) 状态）；调度器的排他前缀扫描 `cum[e+1]=cum[e]+cnt[e]` 也是
**单遍在线 running sum**（O(1) 中间状态，在胞弟单元内）。结构判据：输出（C）不含跨 tile 的全局依赖，
M/N 两维可任意分块并行，K 维是唯一规约维且可单遍累加 —— 因此**不存在需要多遍物化的候选**（不像
softmax/attention 那样需要两遍 max/sum）。源算法已用「分块 K 累加 + 前缀扫描」即直接证据。收益口径：
扫描遍数 = 1（A 一遍、B 一遍、C 一遍，见 R3）；中间缓冲 O(1)（仅 L0C tile + 前导 kernel 的小表 `cum`），
片上驻留可行。

**R3 复杂度对比**（四口径；代表性 shape = deepseek-v3-prefill-gate-up：numel=32768，E=256，N=4096，
K=7168，bf16；elem=2B。"访存"含 A 读 + B 读 + C 写 + 调度表 GM 往返，不含 L1/L2 内部重复；口径见脚注）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| C0 基线（两 kernel，超发 grid，谓词填充，host pad） | 1.92e12 | A 0.47G + **B 15.03G（若按 m-tile 重读为 2× ≈ 30G）** + C 0.27G + pad 0.001G + 表 ≈0.006G ≈ **15.8–30.8G** | B 1–2 遍；A 1 遍（每 tile 重读，靠 L2）；C 1 遍 | L0C 64×256×4=64KB；SMEM (8K+32K)×num_stages | 高（数千 CTA）；无跨核归并 |
| C1 **选定（独立调度表 + Expert Cube 读表 + 动态 extent + 专家聚簇）** | 1.92e12 | A 0.47G + B **15.03G（专家聚簇，目标 1 遍）** + C 0.27G + 表 ≈0.006G ≈ **15.77G** | A 1 遍 + B 1 遍 + C 1 遍 | L0C 64KB；L1 (8K+32K)×stages | 高（逻辑 tile 数）；核内串行，无跨核归并；**无跨引擎同步** |
| C2 融合单核（v1，不可实现） | 1.92e12 | 同 C1 少 0.006G（表不落 GM） | 同 C1 | 同 C1 + UB 调度表 ~4KB | 工具链**不可编译**（CG-2026-0026）；撤回 |
| C5 无搜索映射 | 1.92e12 | 同 C1 | 同 C1 | 同 C1 + 无调度表 | 逻辑 tile 数 = E×max_tiles_per_expert×num_pid_n（**死 tile 比例 ~64×**），守卫浪费大 |
| C6 专家内 pad 到 block_m | 2×numel'*N*K（numel' 为 pad 后行数，uniform 下 ≈numel+E×block_m/2） | A 膨胀 ≈ +E×block_m×K×2 = +0.23G；**B 重读次数随 pad 后 m-tile 数上升** | ≥1 遍且更多 | 同 C1 | 同 C1，但有效计算率下降 |
| C7 split-K（K 切 nsplit 段） | 1.92e12 | A 读 ×nsplit + B 读 ×nsplit + 中间 partial 写/读 ×nsplit | ≥nsplit 遍 | partial 缓冲 [numel,N]×nsplit×4B（可落 GM） | 并行度 ×nsplit；**需跨核归并**（sync_block 或两阶段），归并代价 ≥ 省下的时延 |
| C8 host 调度 | 同 C0 | true_sizes D2H + 表 H2D | true_sizes 1 遍 + 2 次总线往返 | host 内存 | **CPU 同步阻塞**（违反零同步） |

口径脚注：FLOPs 只算乘加（2/element）；访存 = GM 侧不可消除的输入读 + 输出写 + 必要中间往返（不含 L1/L2
复用命中，不含系数项）；扫描遍数指同一数据在 GM 侧被读的次数；中间缓冲峰值以片上（UB/L1/L0C）为准。
**主导项为 B 权重访存（15.03 GB，占 95%+），算子是访存受限**，故 R3 排序以 B 遍数/复用为主导，而非 FLOPs。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰按 negative-claim-evidence.md §2 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C0 基线 | Cube ✅ | L0C 64KB ✅ / L1（2 stage 80KB）✅ | host pad 解决 M；N/K 尾用谓词 | grid 上界编译期常量 ✅ | Pipelined ✅ | ⚠️ 可行但超发 grid 在 NPU 被串行调度（core-split-strategy §1）且 host pad 被禁（ascend-constraints §4）→ 需重设计 |
| C1 **选定** | Cube ✅（Expert Cube-only）+ Vector（前导 kernel 独立执行）✅ | L0C 64KB ✅ / L1 80KB@2stage ✅ | 动态 extent 全方向 ✅ | 循环边界全静态（nums_kernel/num_local_tasks 编译期常量）✅ | Pipelined ✅；**无跨引擎同步**（单 Cube scope）✅ | ✅ 最亲和且**可实现**：访存单遍、无跨核归并、无 host 改动、无 in-block 跨引擎交接 |
| C2 融合单核 | 需 Vector→Cube 同 block 交接 | 同 C1 | 同 C1 | ✅ | ❌ **无可用同步原语**（`sync_block_*` codegen/编译器属性名不匹配，CG-2026-0026） | ❌ 撤回：工具链不可实现 |
| C5 无搜索映射 | Cube ✅ | 同 C1 | 同 C1 | ✅ | ✅ | ❌ 弃选：死 tile ~64×，守卫与空转成本高（R3） |
| C6 专家内 pad | Cube ✅ | A 缓冲/流量膨胀 | pad 后对齐 ✓ | ✅ | ✅ | ❌ 弃选：FLOPs 与 A 访存膨胀，且 pad 需核内额外搬运（host pad 被禁） |
| C7 split-K | Cube ✅ | partial 缓冲超 LB 需落 GM | ✅ | ✅ | 跨核归并需 sync_block 或两阶段 kernel | ❌ 弃选：算子是 B 访存受限，split-K 使 A/B 访存 ×nsplit，收益被访存抵消且引入归并代价 |
| C8 host 调度 | CPU | — | — | — | ❌ 引入 host-device 往返与 CPU 同步 | ❌ 弃选：违反零 CPU 同步 |

**调研结论**：选定 **C1 —— 前导 kernel `_tile_scheduler_kernel` 独立产出调度表（GM）+ 本 GEMM 核
Expert Cube-only 读表 + 动态 extent 精确裁剪 + 专家聚簇任务序 + 固定核数核内串行**。关键依据：
(i) **可实现性是前置约束** —— 本工具链禁止跨引擎同 block 交接（CG-2026-0026，官方 mixcv 样例复现），
故 C2（v1 融合单核）不可实现，R1 收益主张（少 3 表往返 + 少一次 launch）**撤回**；
(ii) R3 显示算子是 **B 访存受限**（B 15.03 GB 占 95%+），C1 以「专家聚簇 + M-major」把 B 保持在单遍为
目标、A/C 各单遍，且无跨核归并；C1 相对 C2 多出的 launch + 表 GM 往返（≈6 KB）+ 跨 kernel barrier 合计
≪ B 访存的 1%，代价可忽略，换来可实现性；
(iii) R4 显示 C1 全方向满足 NPU 约束（无 host 改动、循环边界静态、L0C/L1 容量内、可流水、**无跨引擎
同步**），而 C2 因工具链不可实现、C5/C6/C7/C8 分别因死 tile 爆炸 / FLOPs 与访存膨胀 / 访存倍增+归并代价 /
CPU 同步被弃选。
**与基线结构差异**：把 host pad + 零填充 + 守卫统一替换为动态 extent；把超发 grid + 死 CTA 早退替换为
固定核数 + 核内串行；把 GROUP_SIZE_M L2 重排替换为专家聚簇任务序；**保留**源「独立调度 kernel 写 GM 表 +
GEMM 读表」的两 kernel 数据流（v1 的内联折叠被工具链推翻）。
**调研范围**：algorithm-candidates.md（无 MoE grouped-GEMM 专条；参考 ALG-gemv 的「Cube 分形 vs Vector
规约」与 split-K 条目）、pattern-library（PL-1.8 bf16 Cube 直连、CONST-* 硬件常数、traps-runtime
TRAP-DEVMODE-PERSIST-GEMM）、源算子 §0.3/§0.4、本仓 examples
（examples/gemm、examples/mixcv、examples/deepseek_v32/fp8_lighting_indexer.py、
testing/autotune/anneal/moe_grouped_gemm_anneal.py 的两 kernel NPU 先例）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1→L1 分块（R2 动态 extent）；O2→Cube `T.gemm`；
O3→MTE + `T.Pipelined`；O4（swizzle）**舍弃**（NPU NZ 由搬运转换，理由 §0.5）；O5（L2 重排）→R4 专家聚簇；
O6（早退+超发）→R3 固定核数+核内串行（承接「零 CPU 同步」意图：`total_tiles` 全程在设备侧）；
O7（调度器）→R1 **保留独立调度 kernel + GM 表**（承接「消除每 CTA 的 O(E) 扫描、以 O(log E) 定位」意图：
表由前导 kernel 一次物化，GEMM 核 O(1) 读表）；O8/O9（对齐分派/零填充）→R2 动态 extent（承接「对齐时走
无谓词快路」意图：NPU 无谓词需求）；O10（epilogue 守卫）→R2 动态 extent 写回；host pad→**舍弃**（禁止，理由 §0.5）。

**设计期 roofline（D-2）**：以 §1.6.3 选定布局 + pattern-library/constants.md 实测常数估算下界：
- **流量项（GEMM 核，主导）**：GM 总流量 ≈ A 0.47 GB + B 15.03 GB + C 0.27 GB = **15.77 GB** ÷
  CONST-mte2-degradation 混合流量地板 **~1.26 TB/s**（CONST-mte2-degradation，工具链
  tilelang 0.1.2+ed787bb / CANN 8.5.0）≈ **12.5 ms**（其中 B 单遍 15.03 GB ≈ **11.9 ms** 为绝对主导）；
- **流量项（前导 kernel）**：`true_sizes` 读 1 KB + 3 张表写 ≈6.1 KB ≈ **< 0.1 µs**（可忽略）；
- **发射项**：前导 kernel 一次 launch + 约 4 个向量 op（CONST-vector-launch-overhead ~0.5µs/op）+ 一次
  跨 kernel barrier，合计 **< 0.1 ms**（一次性固定开销）；
- **容量项**：L0C 64KB / L1 80KB@2stage，均 < 上限（CONST-capacity-910B2C：L1 512KB / L0C 128KB）；
  GEMM 核无 UB 调度表（调度已外移）；
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 12.5 ms**（B 访存主导）。
  常数未覆盖的量（NPU L2 对 B 的复用率、动态 extent 搬运开销、跨 kernel barrier 的精确墙钟）显式标注
  「未实测假设 + 估算依据」：按 C1 的专家聚簇目标 B 单遍；若 L2 复用失败则 B 可能 2 遍（≈ 24.4 ms）。
  实测由 Stage 4 校核。注：常数绑定 Ascend910B2C，工具链变更须按 kb_stale_check 重验。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1 算法（调度表由前导 kernel 产出、本 GEMM 核读表）。每项四要素：
> 原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/verify_equiv.py`。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| M1 | 整数 ceil 的移位形式（**调度表语义，前导 kernel 采纳，本核消费**） | `(s_e + block_m - 1) // block_m`（源调度器整数除，源 L95） | `(s_e + block_m - 1) >> log2(block_m)`（block_m 为 2 的幂） | 整数恒等，$s_e\ge0$、block_m=2^k 精确；机器验证 24612 例 0 违反 | 前导 kernel 每专家以 1 次移位替代 1 次整数除（实测收益≈0，属无损整理）；本核**消费其产出的表**，该等价性保证表语义与源一致 |
| M2 | 动态 extent 精确裁剪 + 动态 K 归约范围（**本核采纳**） | 谓词零填充整块 `if_then_else(j<K-k*block_k and ..., B[...], 0)` + 逐元素守卫写 | 只取/只写有效区 `tile_m=min(block_m,s_e-row)`, `tile_n=min(block_n,N-n0)`, `tile_k=min(block_k,K-k*bk)`，并以 `T.gemm(size=[tile_m,tile_k,tile_n])` 令归约维=tile_k | 容差内等价：仅当归约范围=有效 tile_k 时越界项恒为 0（0·x=0）；机器验证（**含 K 分块 + 陈旧尾值注入**）4 用例 fp64 最大相对差 1.003e-11（matmul 重排舍入），EQUIV_PASS；`stale_tail_guard` 确认缺 `size` 会发散（≈1.218e3）。Stage 3 正对照实测 max_diff 9.537e-07 | 消 K/N 零填充逐元素算（源每 tile 每 k 一次 `if_then_else`）；消 host pad 的 block_m×K 拷贝；尾块只搬有效行/列；`size` 免去整块零填充 |
| M3 | NT 形式（不物化 B^T）（**本核采纳**） | 显式转置/重排 B 后 `b_transpose=False` | 直接 `T.gemm(A,B,C,b_transpose=True)` 用 B[N,K] 行主序 | 数学恒等 $(AB^T)$；机器验证 0 违反 | 省一次 B 转置/重排（B 是最大张量，任何全量重排都不可接受） |
| M4 | 无进一步公式化简 | — | — | 单次 NT 矩阵乘 + fp32 分块累加，无公共子表达式、无高代价超越函数、无除法可倒数化（无 `x/s` 形式） | 结论：矩阵乘核心无可降代价的恒等变形，M1–M3 即全部采纳项 |

**优化结论**：采纳 3 项（M1/M2/M3）。**收益重估（撤回 v1 的 R1 主张）**：v1 曾把「调度折叠进核（表不落 GM）」
列为收益项（少 3 张表 GM 往返 + 少一次 kernel launch）；经 Stage 3 实证该形态**不可实现**（CG-2026-0026 /
CG-2026-0010）后**撤回**，重估为：本两 kernel 形态相对 v1 内联形态多出一次 launch + 表 GM 往返（≈6.1 KB）
+ 一次跨 kernel barrier，合计 **< 1 ms，相对 B 访存 15.03 GB（≈11.9 ms）可忽略**；收益主体仍来自 M2
（动态 extent 消 host pad + 零填充 + 守卫）与 M3（NT 省 B 转置）。
**优化后公式**（供 §3.1 拆解）：对每逻辑 tile（读表得 $e$、$row$）、每 tile $(t_m, t_n)$、每 $k$-块：
$$
C[\,m_0:m_0+t_m,\ n_0:n_0+t_n\,] \;=\; \sum_{k} A[\,m_0:m_0+t_m,\ k b_k : k b_k + t_k\,]\,
B[e,\ n_0:n_0+t_n,\ k b_k : k b_k + t_k\,]^{T}\quad\text{(fp32 accumulation)}
$$
其中 $m_0=o_{e}+row$（`row` 已乘 block_m）、$n_0=t_n^{idx}\cdot block_n$，$t_m/t_n/t_k$ 为动态有效尺寸。

**等价性机器验证结果表**（`verify_equiv.py`，fp64 torch CPU 参照；脚本随算子目录留存）：

| 项 | 对照 | 最大度量值 | 违反 / 用例 | 结论 |
|----|------|-----------|-------------|------|
| M1 | 移位式 vs 源码整数 `//`（源 L95） | 0.000e+00 | 0 / 24612 | EQUIV_PASS |
| M2 | 有效区 + 动态 K 归约范围 vs 零填充全 `block_k`（每专家 NT matmul，含 K 分块） | 1.003e-11（相对差，fp64 matmul 重排舍入） | 0 / 4 | EQUIV_PASS |
| M3 | NT（`b_transpose`）vs 物化 B^T | 0.000e+00 | 0 / 1 | EQUIV_PASS |
| （guard） | 省略 `size=`（全 `block_k` 归约，含陈旧尾值）vs 零填充基线 | 1.218e+03（发散，非采纳项） | — / 4 | 确认 M2 前提不可省 |

无失败项（所有采纳项均为 EQUIV_PASS）。`stale_tail_guard` 行的意义：它刻意模拟「动态取数但漏传 `size`」
的错误实现，与基线产生 1.2e3 量级发散，**证明 M2 的等价性确实依赖 `T.gemm(size=[tile_m,tile_k,tile_n])`
限定归约范围**（机器验证现覆盖含 K 分块的形态）。该形态与 Stage 3 已验证正对照
（`repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`，max_diff 9.537e-07）一致。
fp16/bf16 舍入影响评估：三项均为精确恒等或「删除恒零项」，不改变任何实际参与累加的
浮点运算，故不引入额外的 fp16/bf16 舍入误差；M2 中动态 extent 只改变搬运范围与归约范围，有效累加项集合不变。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点**本 GEMM 核**实现方案全部循环与标量计算点（对象 = §1.6.0 选定、§0.6 重设计后的 C1 算法）。
**说明**：调度阶段（前缀扫描 / 查找 / 向量化）已按 §0.6 R1 外移到前导 kernel `_tile_scheduler_kernel`，
其循环/标量点由胞弟设计的 §1.6.2 承载；本核不再含任何索引段向量工作。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | K 维累加循环 | `T.Pipelined(ceildiv(K,bk))` | Cube `T.gemm`（已是矩阵单元，非标量） | ✅ | — |
| 2 | A/B 分块取数 | 标量/逐元素 | `T.copy` 切片（向量搬运 + ND2NZ） | ✅ | — |
| 3 | epilogue 逐元素守卫 `if i<actual_rows and j<actual_cols` | `T.Parallel(block_m,block_n)` 条件写 | 动态 extent 写回（`store_fixpipe([tile_m,tile_n])`）——**整体消除该循环** | ✅ | — |
| 4 | host `F.pad(A)` | host 逐元素 pad | 删除（动态 extent 取代） | ✅ | — |
| 5 | tile 坐标与基址 `bx/by/m_start/n_start` | 标量乘加（读调度表导出） | 无 | ❌ | block 级索引 / 任务映射计算（每 task 少量标量），无逐元素等价 API；输入来自 GM 调度表（运行时值） |
| 6 | 调度表标量读 `total_tiles[0]`、`tile_expert_ids[bx]`、`tile_row_offsets[bx]`、`true_offsets[e]`、`true_sizes[e]` | GM 标量读 | 无 | ❌ | 每 task 数次标量 GM 读（O(1) 寻址，正是源调度器的设计目的）；非逐元素热点；`bx` 为运行时值，向量化无意义 |
| 7 | 动态 extent `tile_m/tile_n/tile_k` | 标量 `T.min` | 无（标量归约） | ❌ | 每 task 3 次 `T.min` 的 block 级标量，用于设定搬运/写回 extent，非逐元素热点 |
| 8 | 死 tile 守卫 `bx < total_tiles[0]` | 标量比较 | 无 | ❌ | block 级单比较守卫（每 task 一次），非逐元素热点 |

**向量化结论**：本 GEMM 核为 **Cube 主导**，逐元素计算已全部消除（第 3 项整体消除、第 4 项删除），
数据搬运与矩阵计算由 Cube 单元（`T.copy`/`T.gemm`/`store_fixpipe`）承担（第 1/2 项）。仅保留 3 处
block 级标量点（第 5/6/7/8 项：tile 坐标基址 / 调度表标量读 / 动态 extent / 死 tile 守卫），
理由类别：**block 级索引计算 + O(1) 运行时标量读（数据来自 GM 表）**；均非逐元素热点
（每核 O(num_local_tasks) 次标量，相对 B 访存（15 GB 量级）可忽略）。**无可向量化而未向量化的逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子为 Cube 类（I/O layout 为契约：A `(numel,K)`、B `(E,N,K)`、C `(numel,N)` 行主序），
> 核内布局是设计变量。调度阶段的向量化轴（专家轴 E）属前导 kernel（胞弟设计），**本核无向量化轴**
> （Cube-only，无 Vector scope）。Cube 路径布局按 cube-skill + MixCV epilogue 归属决策。

**轴质量评分**：本核无 Vector 逐元素/归约轴（索引已外移），故不适用逐 lane 整除性评分；
调度表的专家轴评分见胞弟设计 §1.6.3。

**候选矩阵**（Cube 类：分形布局 + epilogue 归属）：

| # | 布局方案 | 向量化轴 / 关键决策 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| L1 | A/B GM→L1 走 `T.copy` 切片（Expert 模式自动 ND2NZ），`T.gemm(b_transpose=True, size=[tile_m,tile_k,tile_n])` 直接消费 B[N,K] | 无（本核无 Vector 轴） | 无（ND2NZ 在搬运内完成） | 无额外 GM 流量；NZ 由后端处理 | ✅ **主选** |
| L2 | 显式 host 侧把 B 转置物化为 [K,N]（或提前转置），再 `b_transpose=False` | — | host permute | B 是全仓最大张量（15 GB），任何全量 host 转置 = 百 µs~ms 级且改动输入/需额外存储 → 净亏；且 host 侧禁止改动输入真实内容（ascend-constraints §4） | ❌ 弃选 |
| L3 | 核内 `T.transpose` 显式把 B tile 转成 [K,N] 再 gemm | — | 核内转置链 | 多余向量管线开销；且 NZ 通路已支持 NT 形式（`b_transpose=True`，T.gemm.md §2.1）；另有 TRAP-transpose-epilogue-poison（活跃源 transpose 毒化 kernel 2.6×，pattern-library traps-compiler.md） | ❌ 弃选 |
| L4 | epilogue 归属 Vector（L0C→GM 走 GM 往返再由 Vector 逐元素处理） | — | L0C→GM→UB | 本算子 epilogue 无逐元素运算（仅裁剪写回），Vector 处理无收益且引入 GM 往返（CONST-store-fixpipe-gm-only：L0C 跨引擎传输强制 GM 往返） | ❌ 弃选 |
| L5 | epilogue 归属 Cube（`store_fixpipe` 动态 `[tile_m,tile_n]`） | — | 无（fixpipe 直出 GM，含 fp32→fp16/bf16 量化） | 单次搬运，尾块精确裁剪 | ✅ **主选** |
| L6 | 核内调度内联（v1：Vector 索引段 + 同 block 交接给 Cube） | E 轴 `T.Parallel` | — | — | ❌ 弃选（**工具链不可实现**，CG-2026-0026 / CG-2026-0010；§0.6 R1） |

**布局决策结论**：选定 **L1 + L5** —— 核内布局为 **NZ 分形**（A/B 经 GM→L1 搬运隐式转换，`T.gemm` NT 形式
直接消费 B[N,K]），**epilogue 归属 Cube 侧 `store_fixpipe`**（动态 extent 裁剪，无 Vector 二次处理）；
**本核为 Expert Cube-only，无 Vector scope、无跨引擎交接**（L6 因工具链阻断弃选，§0.6 R1）。
repack 路径：无（NZ 转换在搬运内完成）。该结论已同步 §3.3 伪代码与 §6 循环结构（本核循环只有核内
`T.serial` 任务循环 + Cube `T.gemm` 的 K 维流水，无 Vector 索引循环）。L2/L3/L4 的弃选量化理由见候选矩阵
（并依据 negative-claim-evidence.md §2：L2/L3 的负向论断附 ascend-constraints §4、T.gemm.md §2.1、
pattern-library TRAP-transpose-epilogue-poison 条款）。本算子无「主选 vs 备选判定裕度依赖未实证常数」
的轴决策（NT+NZ 是文档指定通路），故不触发实验裁决三件套。
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Expert（唯一形态，单 Cube scope）**。执行形态：`T.Kernel(nums_kernel, is_npu=True)` +
`T.Scope("Cube")` + `T.alloc_L1`/`T.alloc_L0C` + `T.copy` + `T.gemm(b_transpose=True,
size=[tile_m,tile_k,tile_n])` + `T.store_fixpipe([tile_m,tile_n])`，**核内无 Vector 段、无跨引擎同步**；
调度表由前导 kernel `_tile_scheduler_kernel` 独立产出（§0.6 R1）。用户虽指定 `programming_mode=developer`，
但 Developer 形态经 Stage 3 实证**不可实现**（lowering 期 segfault，CG-2026-0010），按「实测为仲裁依据」
切换 Expert 并在此披露。

**分类口径澄清**：调度 prompt 将本算子标为「MoE 混合 Cube+Vector」，但本算子**数据通路是 Cube 主导**
（唯一数值计算为 `T.gemm` NT 矩阵乘 + `store_fixpipe` 写回），**无逐元素浮点向量工作、无 Vector scope**；
调度索引（前缀扫描/查找）由前导 kernel 独立承担，属**另一个单元**。按 AGENTS.md 的 Developer-mode MixCV
触发规则（Cube `T.gemm` + **至少一个 v-prefix op**），本算子严格意义上不满足 v-prefix 判据（未使用
`vadd/vmul/vexp/vcast/vbrc` 等），应归为 **纯 Cube（Expert）**；故本核不涉及 Cube↔Vector 交接，
不再需要 mixcv-skill 的跨 scope 同步范式。

### 2.2 选型理由

- **Developer 不可用（实测）**：(i) Developer 单 Scope（persistent `T.serial` + `T.gemm` + 整型索引段）
  在 lowering 期 segfault（无 stderr，signal 11；单独验证：去索引段正常、去 gemm 正常，共存即崩）；
  (ii) Developer 下动态 extent `T.gemm` 静默错值（max_diff 1.04e-2，Expert 同结构 9.5e-7）。
  二者共同封死 Developer 路径（CG-2026-0010 扩展触发面）。
- **Expert 形态且只保留 Cube scope（实测可用）**：把调度外移（§0.6 R1）后，本核为**单一 Cube scope**，
  无 Vector→Cube 交接，从而**完全规避** CG-2026-0026（`sync_block_*` 属性名不匹配，任何跨引擎同 block
  交接不可编译）。Stage 3 正对照验证：Expert Cube + 外部调度表 + 动态 extent + NT gemm + store_fixpipe
  的 max_diff **9.537e-07**。
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
| 5 | `t_m=min(block_m, true_sizes[e]−row)`, `t_n=min(block_n,N−n_0)`, `t_k=min(block_k,K−k·bk)` | 动态 extent（M2） |
| 6 | `C_tile = Σ_k A[m_0:m_0+t_m, kb:kb+t_k] · B[e, n_0:n_0+t_n, kb:kb+t_k]^T` | NT GEMM + fp32 累加（M3） |
| 7 | 写回 `C[m_0:m_0+t_m, n_0:n_0+t_n]` | 动态 extent store（M2） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | tile 坐标 | 标量整除/取模 | `bx/by` 标量 | Expert |
| 2 | 死 tile 守卫 | 标量比较（GM 标量读 `total_tiles[0]`） | `if bx < total_tiles[0]` | Expert |
| 3 | 调度表寻址 | GM 标量读（运行时下标） | `tile_expert_ids[bx]`、`tile_row_offsets[bx]` | Expert |
| 4/5 | 基址/动态 extent | 标量乘加 + `T.min` | `T.min(block_m, true_sizes[e]−row)` 等 | Expert |
| 6 | NT 矩阵乘 | `T.gemm(a_l1, b_l1, c_l0c, size=[tile_m, tile_k, tile_n], b_transpose=True, initC=(k==0))` | `size=[m,k,n]` 限定动态归约范围；`b_transpose=True` | Expert |
| 6 | K 维流水 | `T.Pipelined(ceildiv(K, block_k), num_stages)`（或 Expert 下 `T.serial` 与 `T.rs` 组合） | 分块累加 | Expert |
| 6/7 | 取数/写回 | `T.copy` 切片（GM→L1 ND2NZ）+ `T.store_fixpipe(c_l0c, C[m0,n0], size=[t_m,t_n], enable_nz2nd=True)` | 动态 extents | Expert |

> 说明：本核无逐元素运算，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix 算术 API（v-prefix 面向浮点
> 向量运算，见 AGENTS.md API Convention）——`T.gemm` 承担矩阵单元、`T.copy`/`T.store_fixpipe` 承担数据搬运。

### 3.3 计算伪代码

> `T.Tensor` 第二个位置参数直接传 dtype（不用 `dtype=` 关键字），避免 `name '...' is not defined` 类假警报。
> 本伪代码与 Stage 3 已验证正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` 的 `positive_control()` 同形态
> （含核内任务循环与死 tile 守卫），并补全 `num_pid_n`/动态 A 行。

```python
# 编译期常量：numel, num_experts, N, K, block_m, block_n, block_k, num_stages,
#             max_tiles = numel // block_m + num_experts
#             num_pid_n = ceildiv(N, block_n)
#             nums_kernel = min(max_tiles * num_pid_n, P)   # P = 实查物理核数
#             num_local_tasks = ceildiv(max_tiles * num_pid_n, nums_kernel)  # 静态
# 调度表由前导 kernel _tile_scheduler_kernel 产出（§0.6 R1），本核只读。

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
    with T.Kernel(nums_kernel, is_npu=True) as (core_id, subid):
        # ---- 单 Cube scope：读表 + NT GEMM + 写回（无 Vector 段、无跨引擎同步）----
        with T.Scope("Cube"):
            a_l1  = T.alloc_L1((block_m, block_k), dtype)
            b_l1  = T.alloc_L1((block_n, block_k), dtype)
            c_l0c = T.alloc_L0C((block_m, block_n), "float32")
            for t in T.serial(num_local_tasks):
                tile_id = core_id * num_local_tasks + t
                bx = tile_id // num_pid_n
                by = tile_id % num_pid_n
                # 死 tile 守卫：单比较，位于 T.serial 体内、T.Pipelined 之外（规避 CG-2026-0001）
                if bx < total_tiles[0]:
                    expert_id = tile_expert_ids[bx]        # GM 标量读（运行时下标）
                    row       = tile_row_offsets[bx]       # GM 标量读；单位=行（已乘 block_m）
                    n_start   = by * block_n
                    m_start   = true_offsets[expert_id] + row
                    tile_m    = T.min(block_m, true_sizes[expert_id] - row)
                    tile_n    = T.min(block_n, N - n_start)
                    for k in T.serial(T.ceildiv(K, block_k)):
                        tile_k = T.min(block_k, K - k * block_k)
                        T.copy(A[m_start:m_start + tile_m, k * block_k:k * block_k + tile_k],
                               a_l1[0:tile_m, 0:tile_k])
                        T.copy(B[expert_id, n_start:n_start + tile_n,
                                 k * block_k:k * block_k + tile_k],
                               b_l1[0:tile_n, 0:tile_k])
                        # size=[tile_m, tile_k, tile_n] 限定归约范围=有效 tile_k（正确性必需，
                        # 见 §3.3 注 / §1.6.1 M2 stale_tail_guard）。
                        T.gemm(a_l1, b_l1, c_l0c,
                               size=[tile_m, tile_k, tile_n],
                               b_transpose=True, initC=(k == 0))
                    with T.rs("PIPE_FIX"):
                        T.store_fixpipe(c_l0c, C[m_start, n_start],
                                        size=[tile_m, tile_n], enable_nz2nd=True)
```

**注 1（K 尾正确性前提）**：`T.gemm(..., size=[tile_m, tile_k, tile_n])` 的 `size` 不是可选优化而是
**正确性必需**——`a_l1/b_l1` 的 `[tile_k, block_k)` 区未被 `T.copy` 写入，若无 `size` 则按缓冲静态
`block_k` 归约、把陈旧值计入累加（K 非整除用例必错）。§1.6.1 M2 的机器验证已覆盖含 K 分块的形态并注入
陈旧尾值，`stale_tail_guard` 确认缺失 `size` 会发散（max_abs_diff ≈ 1.218e3）。
**注 2（为何单 Cube scope 可行、为何不再需要 Vector）**：调度表由**前导 kernel** 产出并落 GM，
本核只做「读 GM 表 + 读 GM 标量 + GEMM + 写回」，**不存在 Vector→Cube 同 block 交接**，故不触发
CG-2026-0026；Stage 3 正对照已验证该形态数值正确（max_diff 9.537e-07）。
**注 3（`initC=(k==0)`）**：首块以 `initC=True` 覆盖初始化 L0C，后续块累加；等价于源 `T.clear(C_local)`
后逐块累加（避免在 `T.Pipelined` 体内构造 `k==0` 条件，CG-2026-0001）。
**注 4（`true_offsets`）**：源 op 的必需输入；本设计保留其作为输入（与 `true_sizes` 同源）。
若前导 kernel 一并产出 `true_offsets`（行前缀和），可进一步减少输入依赖，Stage 3/集成按「减少输入依赖」择一。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.gemm(A,B,C,size=[m,k,n],b_transpose=True,initC=False)` | docs/Tilelang.language/线性代数操作/T.gemm.md §1/§2.1（签名 `T.gemm(src1,src2,dst,size=[],initC=False,a_transpose=False,b_transpose=False)`；`size=[a,b,c]` 语义：src1 为 [a,b]/[b,a]、src2 为 [b,c]/[c,b]）；本机 `inspect.signature(T.gemm)` = `(A, B, C, size=None, initC=False, a_transpose=False, b_transpose=False)`（`tilelang/language/customize_npuir.py` 的 `npuir_dot`，`tilelang/language/__init__.py` L102-103 `npuir_dot as gemm`）；examples/gemm/matmul_dynamic_shape.py L37-41、examples/mixcv/mixcv_mixkernel.py L53、Stage 3 正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` | 已验证（示例与正对照在用；**无 `transpose_B` 形参**，本核统一用 `b_transpose`） |
| `T.Kernel(n, is_npu=True)`（1D 核 + 核内 `T.serial`） | examples/deepseek_v32/fp8_lighting_indexer.py L53-56（`T.serial(ceildiv(...))` + 单比较守卫）；Stage 3 正对照 | 已验证（示例与正对照在用） |
| `T.copy` 切片 + 动态 extents（尾块）；GM→L1 自动 ND2NZ | docs/Tilelang.language/内存操作/T.copy.md §2.2.2 第 2 条、§2.3 第 6 条、§2.4 示例 4 | 已验证（文档明确 + 正对照在用） |
| `T.min` 标量/张量 | docs/Tilelang.language/比较操作/T.min.md；examples/mixcv/mixcv_mixkernel.py L44；Stage 3 正对照 | 已验证 |
| `T.store_fixpipe`（动态 extent） | docs/Tilelang.language/内存操作/T.store_fixpipe.md；examples/mixcv/mixcv_mixkernel.py L57；Stage 3 正对照 | 已验证 |
| `T.alloc_L1` / `T.alloc_L0C` | examples/mixcv/mixcv_mixkernel.py；Stage 3 正对照 | 已验证（Expert 路径） |
| `T.serial` / `T.Pipelined` | docs/Tilelang.language/ + examples/gemm/example_gemm.py L27；Stage 3 正对照 | 已验证 |
| ~~`T.sync_block_set/wait`~~ | docs/Tilelang.language/同步管道操作/T.sync_block_set.md | **不可用（工具链缺口）**：codegen 出 `syn_instr_mode`、bishengir-compile 解析 `sync_instr_mode` → `error: custom op 'syn_instr_mode' is unknown`（CG-2026-0026，官方 mixcv 样例复现）。本设计**不使用**（单 Cube scope 无跨引擎交接） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 是（源为展平 2D grid 的一维化） | 一维 `T.Kernel(nums_kernel)`；tile 坐标由核内 `bx/by` 除法/取模得到 |
| GEMM M/N 需为 block 整数倍 | 是（numel 一般不整除 block_m，N 可 < block_n） | 动态 extent 精确裁剪（§0.6 R2）；host 侧不做任何 padding |
| L0C 容量 （128KB） | 是 | block_m×block_n×4B = 64×256×4 = 64KB ≤ 128KB ✓ |
| 物理核数限制 | 是 | `nums_kernel = min(逻辑核数, 20)`；极大规模核内 `T.serial`（§5.5） |
| host 侧禁止改动输入 | 是（源有 `F.pad(A)`） | 舍弃 host pad，改动态 extent（§0.5） |
| 三维 Kernel / GPU API | 源用 `T.copy` + TMA/swizzle（CUDA） | 换 NPU 等价：`T.copy` 切片 + NZ/动态 extent（§0.5） |
| 跨引擎同 block 交接 | v1 曾用（已撤回） | 本版**不使用**（单 Cube scope）；依据 CG-2026-0026（§0.6 R1） |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| Kernel 维度 | 1D grid（超发 `max_tiles×num_pid_n`） | 1D grid（`nums_kernel` 固定）+ 核内 `T.serial` 多 tile | §0.6 R3 |
| tile 调度 | 独立第二 kernel（thread 并行二分），GEMM 读表 | **保留**：独立前导 kernel，GEMM 读表（v1 的内联已撤回） | §0.6 R1 |
| 取数 | `T.copy`(TMA) + 谓词零填充 | `T.copy` 切片 + 动态 extent | §0.6 R2 |
| host pad | `F.pad(A)` | 删除 | §0.5 |
| swizzle | `make_swizzled_layout` | 删除（NZ 由搬运转换） | §0.5 O4 |
| 尾块写回 | `T.Parallel` 逐元素守卫 | `store_fixpipe` 动态 extent | §0.6 R2 |
| 编程模式 | CUDA 单 kernel | Expert 单 Cube scope（无 Vector；规避 CG-2026-0026） | §2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` | 直接（Stage 3 正对照） | Expert `T.Scope("Cube")` + 外部调度表 + 动态 extent + `T.gemm(size=)` + `store_fixpipe`（max_diff 9.537e-07）——**本核结构的实测底座** |
| `examples/deepseek_v32/fp8_lighting_indexer.py` | 高度相似 | 固定核数 + 核内 `T.serial(ceildiv(...))` 多任务 + 单比较守卫（L53-57）；Cube+Vector 双 Scope；`T.gemm(initC=True, b_transpose=True, size=[H,K,BLOCK_SIZE_N])`（L74-81）；`store_fixpipe` 动态基址（L83） |
| `examples/gemm/matmul_dynamic_shape.py` | 高度相似 | **M/N/K 全动态 extent 的规范写法**：`remain_M/remain_N/remain_K = T.min(...)` + `T.gemm(..., initC=(i==0), b_transpose=False, size=[remain_M, remain_K, remain_N])`（L28-44）+ `store_fixpipe(size=[remain_M, remain_N])`（直接对应本算子 K 尾归约范围与尾块写回） |
| `examples/mixcv/mixcv_mixkernel.py` | 高（Expert 双 Scope 范式） | `T.Scope("Cube")/("Vector")` + `tile_size_M=T.min(block_M, ...)` 动态 extent + `store_fixpipe([tile_size_M,tile_size_N])`（L42-58）；**注意其 `sync_block_set/wait` 当前不可编译（CG-2026-0026），本设计不用** |
| `testing/autotune/anneal/moe_grouped_gemm_anneal.py` | 高（两 kernel 数据流参照） | 独立 `tile_scheduler_kernel_npu` 写 3 张 GM 表 + `moe_gemm_kernel_npu` 读表 GEMM；**其 `T.Kernel(max_tiles)` 超发形态不采用**（§0.6 R3） |
| `examples/gemm/example_gemm.py` | 高（Cube 基础） | 一维 `T.Kernel` + `T.Pipelined` + `T.gemm` + `T.copy` 的标准 matmul 骨架 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (numel, K) | float16 / bfloat16 | 紧致激活，行连续 |
| B | (num_experts, N, K) | 同 A | 专家权重（NT，对 B 转置恒等使用） |
| true_sizes | (num_experts,) | int32 | 每专家行数 $s_e$ |
| true_offsets | (num_experts,) | int32 | 每专家在 A 中的紧致起始行 $o_e$（可由前导 kernel 一并产出，见 §3.3 注 4） |
| tile_expert_ids | (max_tiles,) | int32 | **调度表**（GM，前导 kernel `_tile_scheduler_kernel` 产出）：每 M-tile 专家号（死 tile = -1） |
| tile_row_offsets | (max_tiles,) | int32 | **调度表**：M-tile 在专家内的行偏移（已乘 block_m） |
| total_tiles | (1,) | int32 | **调度表**：有效 M-tile 总数（设备侧，不回落 CPU） |

> **对接契约（本核消费 / 胞弟产出）**：① `max_tiles = numel // block_m + E` 与 `block_m` 为两函数共享的
> 编译期常量；② 表 dtype 全 int32；③ `tile_row_offsets` 单位是**行**（已乘 block_m），不是 tile 序号；
> ④ 死 tile 哨兵 `(-1, 0)`；⑤ `total_tiles ≤ max_tiles`；⑥ 两 kernel 同 stream 顺序 launch，前导 kernel
> 先完成，GEMM 核再执行（无 in-block 同步，无 CPU 同步）。
> 源另有 `tile_expert_ids/tile_row_offsets/total_tiles` 三类调度表输入 —— 本设计**保留其为输入**
> （与源同数据流），由前导 kernel 产出（§0.6 R1）。

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
[前导 kernel] true_sizes --前缀扫描/查找--> GM[表: tile_expert_ids / tile_row_offsets / total_tiles]
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

> **L1 预算复核**：A(8KB)+B(32KB) × num_stages(2) = 80KB ≤ 512KB ✓（CONST-capacity-910B2C）；
> num_stages 可达 4（160KB），默认 2。
> **L0C 复核**：64×256×4 = 64KB ≤ 128KB ✓（CONST-capacity-910B2C）。
> `pass_configs` 关闭 auto-multi-buffer（`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`），故不适用
> ×1.7 膨胀系数（该系数对应 auto-multi-buffer **开启**；Expert 显式 alloc 结构 ≈ 手工核算再 ±12%，
> 见 CONST-capacity-910B2C 第三/第六数据点）。
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
def _moe_grouped_gemm_kernel(numel, max_tiles, num_experts, N, K, dtype="float16",
                             block_m=64, block_n=256, block_k=64, num_stages=2):
    ...
```

> `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` 避免 buffer 重排/重作用域破坏显式 L1/L0C
> （CG-2026-0012 / TRAP-UB-dynsubview-dominance）；`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False` 关闭
> auto-multi-buffer（与 Stage 3 正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` 的 `_PASS` 一致）。

---
## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Cube（Expert 单 Cube scope）+ 前导 kernel 的整数 Vector 索引**。判定依据：本核唯一数值
计算为 `T.gemm`（Cube 矩阵单元）+ `store_fixpipe`（epilogue 归 Cube），无逐元素浮点向量、无 Vector scope；
调度索引段（整数前缀扫描/查找）由**前导 kernel `_tile_scheduler_kernel`** 独立承担（§0.6 R1），不在本核内。

### 5.2 Block 划分

```python
block_m = 64    # 分形 M 对齐（≥16）；L0C 64KB 预算；专家行数通常 ≥ 数十
block_n = 256   # 分形 N 对齐；L0C 64×256×4=64KB ≤128KB
block_k = 64    # 分形 K 对齐（≥32）；K=96 等非整除由动态 extent + `T.gemm(size=)` 归约范围处理
```

### 5.3 约束分析

- **对齐约束**：block_m=64、block_n=256、block_k=64 均 ≥ 分形下限（M/N≥16、K≥32）且 32B 对齐 ✓。
- **L0C 容量**：64×256×4 = 64KB ≤ 128KB ✓（CONST-capacity-910B2C）。
- **L1 容量**：A(8KB)+B(32KB) × num_stages(2) = 80KB ≤ 512KB ✓；num_stages 可达 4（160KB）。
- **UB 容量**：本核无 UB 分配 ✓（调度已外移）。
- **非整除**：numel / 每专家 $s_e$ / N / K 均可能不整除 → **动态 extent 精确裁剪**（§5.4）。

### 5.4 注意事项（非整除处理）

- **M/行方向**：每专家独立切 tile，尾块 `tile_m = T.min(block_m, s_e − row)`（`row` 来自调度表，已乘
  block_m）；`store_fixpipe` 只写 `[0:tile_m)` 行，**杜绝越界写坏下一专家**（no-pad 关键正确性约束）。
- **N 方向**：`tile_n = T.min(block_n, N − n_start)`（smoke 用例 N=128 < block_n=256）。
- **K 方向**：`tile_k = T.min(block_k, K − k·block_k)`（full 用例 K=96 → 末块 tile_k=32，K=70 → 末块 6）；
  **该动态 extent 必须同时作为 `T.gemm(..., size=[tile_m, tile_k, tile_n])` 的归约范围**，否则操作数缓冲
  `[tile_k, block_k)` 的陈旧值会被按静态 block_k 归约计入累加（正确性关键，§3.3 注 1、§1.6.1 M2 guard）。
- **无 host padding**（ascend-constraints §4）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。
> 本节针对**本 GEMM 核**；前导 kernel 的分核由胞弟设计（其核数/串行形态自有 §5.5）。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **返回值 = 20**（本机查询代码与实测返回值：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → 20，Ascend 910B2C，
  2026-10-13）。本算子为 Cube 算子 → **直接使用返回值 20**（纯 Vector 才翻倍为 40）。
- **逻辑核数**：`num_logical_kernels = max_tiles × num_pid_n = (numel//block_m + E) × ceil(N/block_n)`。
  代表 shape：smoke(64,4,128,64) → 5×1 = **5**；decode gate-up(4096,256,4096,7168) → 320×16 = **5120**；
  prefill gate-up(32768,256,4096,7168) → 768×16 = **12288**；prefill down(32768,256,7168,2048) → 768×28 = **21504**。
- **规模判定**：smoke 逻辑核数 5 ≤ 20（**无需适配**）；其余代表 shape 逻辑核数 5120–21504 **远超物理核数 20**
  → **极大规模**，无法通过调 block 缩减到物理核数整数倍（block_m/block_n 增大只会让尾块浪费与 L0C 压力上升）。
- **分核方案**：
  - 统一取 `nums_kernel = min(num_logical_kernels, 20)`（编译期常量）；
    `num_local_tasks = ceildiv(num_logical_kernels, nums_kernel)`（编译期常量，**循环边界静态**）。
  - 极大规模：固定启动内核数 = 物理核数 20，核内 `for t in T.serial(num_local_tasks)` **核内串行**处理多个
    逻辑 tile（`tile_id = core_id * num_local_tasks + t`，块连续以利专家聚簇），摊薄核启动开销。
  - smoke：`nums_kernel = 5`，`num_local_tasks = 1`，无超发、无负载不均（5 < 20 但覆盖全部逻辑 tile，
    其余 15 核不启动）。
  - 负载均衡：块连续分配在均匀分布下每核任务数差 ≤1；偏斜分布下个别核可能多领少量死 tile（被守卫跳过），
    影响可忽略。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级 | 固定核数网格 | `T.Kernel(nums_kernel, is_npu=True)` | 分核策略（§5.5） |
| 核内任务 | 串行多 tile | `T.serial(num_local_tasks)`（静态边界） | 极大规模，摊薄核启动开销 |
| 死 tile 守卫 | 单比较跳过 | `if bx < total_tiles[0]` | 位于 `T.serial` 体内、`T.Pipelined` 之外（规避 CG-2026-0001） |
| K 维 | 流水累加 | `T.serial(ceildiv(K,block_k))` 或 `T.Pipelined(..., num_stages)` | Cube 分块累加/访存重叠 |
| 元素级 | 无（已消除） | — | epilogue 守卫被动态 extent 整体消除（§1.6.2#3）；无 Vector 索引循环（已外移） |

### 6.2 循环伪代码

见 §3.3（核内 `T.serial(num_local_tasks)` 外层、K 循环内层）。**本核无 Vector 索引循环**（调度已外移，
§0.6 R1）。

### 6.3 流水线优化

- K 循环 `T.serial`/`T.Pipelined`：A/B tile 双缓冲，MTE2 搬运与 Cube 计算重叠。Stage 3 正对照以
  `T.serial` 形态跑通（max_diff 9.537e-07）；Stage 3/4 可实测 `T.Pipelined(num_stages=2)` 是否更优
  （若 Pipelined 与 Expert 单 scope 组合不稳定，则回退 `T.serial` + `T.rs` 显式流水）。
- `num_stages` 由 L1 容量约束：A(8KB)+B(32KB) × num_stages ≤ 512KB → 2–4 可行，默认 2（源默认 3，
  NPU 侧按容量与实测重定，Stage 4 扫描）。
- 前导 kernel 与本 GEMM 核**顺序执行**（同 stream）；不同核之间无重叠（这是两 kernel 形态的固有代价，
  相对 B 访存可忽略，§1.6.0 roofline）。

### 6.4 尾块处理

三方向（M/N/K）统一用动态 extent（`T.min`）——见 §5.4；其中 **K 方向须把 `tile_k` 传入 `T.gemm(size=)`**
限定归约范围（否则操作数缓冲的陈旧尾值污染累加，见 §3.3 注 1、§1.6.1 M2 guard）。不使用 host padding、
谓词零填充或逐元素守卫。死 tile（`bx ≥ total_tiles`）用单比较 `if` 跳过，**该 `if` 位于 `T.serial` 体内，
不得放入 `T.Pipelined` 体内**（CG-2026-0001：Pipelined 体内条件构造崩溃）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**Expert 手动/管线同步，但本核无跨引擎同步需求**。本核为**单 Cube scope**，只做「读 GM 表 +
GEMM + 写回」；不存在 Vector→Cube 或 Cube→Vector 的同 block 交接，故**不使用** `T.sync_block_set/wait`
（该原语在当前工具链不可编译，CG-2026-0026）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 前导 kernel → 本 GEMM 核 | **无显式同步**（同 stream 顺序 launch） | 前导 kernel 先完成、本核再执行；kernel 边界即隐式全局顺序，不需 in-block flag |
| L0C 写回 C 前 | `T.rs("PIPE_FIX")`（Cube scope 内） | 保证 `store_fixpipe` 在正确管线阶段发出（Stage 3 正对照形态） |
| A/B GM→L1 与 `T.gemm` 之间 | 框架/`T.rs` 管线同步（Cube scope 内） | MTE2→Cube 单引擎内数据依赖；非跨引擎 |
| ~~Vector→Cube `sync_block_set/wait`~~ | **不使用** | CG-2026-0026：`syn_instr_mode` 属性名与 `bishengir-compile` 不匹配，任何跨引擎同 block 交接不可编译 |

> **关键差异（相对 v1）**：v1 的 Expert 回退依赖 Vector→Cube 同 block 交接（GM scratch + `sync_block_*`），
> 被 CG-2026-0026 阻断；本版把调度外移到前导 kernel，**彻底移除跨引擎同步需求**，只保留 Cube scope 内
> 管线同步与 kernel 间自然顺序。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```

> 与 Stage 3 正对照 `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` 的 `_PASS` 一致：关闭 buffer 重排/重作用域
> （CG-2026-0012 / TRAP-UB-dynsubview-dominance）与 auto-multi-buffer，保证显式 L1/L0C 结构稳定
> （PL-1.16 Expert 双关闭硬边界）。

---
## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源测试
> `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` 的 `_ref_grouped_gemm_nopad`），
> **不得复刻 §0.6 的 NPU 算法**（保证验证独立性）。

```python
import torch

def golden_moe_grouped_gemm_nopad(a, b, true_sizes, true_offsets):
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

（与源测试 `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2)` 一致；fp32 累加 + 输出回落输入 dtype。
Stage 3 正对照 Expert Cube 动态 extent 形态实测 max_diff 9.537e-07，远优于该门限。）

### 8.3 L0 门槛测试计划

> **调度表来源**：生产路径由前导 kernel `_tile_scheduler_kernel`（同 stream 先 launch）产出并落 GM；
> 本 GEMM 核的 L0 单独测试可用 **host-built 调度表 fixture**（从 `true_sizes` 按源语义展开，
> 等同 `testing/autotune/anneal/moe_grouped_gemm_anneal.py` 的 `supply_prog` 形态）——该 fixture 只构造
> **辅助表输入**，不改动 A/B 真实内容（不违反 ascend-constraints §4）。整算子路径须实测两 kernel 协同
> （验证对接契约 §4.1）。

| 用例 | shape（numel,E,N,K,dtype,dist） | 验证点 |
|------|--------------------------------|-------|
| L0-1 smoke（K 对齐、N < block_n） | 64,4,128,64,fp16,uniform | 基本正确性、N 尾（tile_n=128<256）、M 尾、读表寻址 |
| L0-2 smoke（偏斜 + fp16） | 64,4,128,64,fp16,skewed | 大量 1-token 专家、死 tile 守卫（`-1` 哨兵） |
| L0-3 full（K 非整除） | 128,8,256,96,bf16,uniform | K 尾动态 extent、`T.gemm(size=)` 归约范围、M/N 尾 |
| L0-4 中等偏斜 | 233,5,130,70,bf16,skewed | 奇偶 shape、多尾块 |
| L0-5 代表大 shape | 4096,256,4096,7168,bf16,uniform | 分核极大规模、B 访存路径、无死锁 |
| L0-6 两 kernel 协同 | 64,4,128,64,fp16,uniform | 前导 kernel → GEMM 核对接契约（表内容/`total_tiles` 全程设备侧/无 CPU 同步） |

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 源 `host F.pad(A)` 被禁止（ascend-constraints §4）→ 已用动态 extent 取代。
- 不支持三维 Kernel → 一维网格 + 核内坐标计算。
- 一维核超发会被串行调度（core-split-strategy §1）→ 固定 20 核 + 核内串行。
- **跨引擎同 block 交接不可用**（CG-2026-0026）→ 调度外移到前导 kernel，本核单 Cube scope。
- Developer 模式对 persistent + gemm 结构不可用（CG-2026-0010）→ Expert 单 Cube scope。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **跨引擎同 block 同步不可编译**（CG-2026-0026） | Expert 双 Scope 内用 `T.sync_block_set/wait` | `error: custom op 'syn_instr_mode' is unknown`（官方 mixcv 样例复现） | **不使用**：调度外移到前导 kernel，本核单 Cube scope（§0.6 R1 / §7.2） |
| **Developer + persistent + gemm 崩溃**（CG-2026-0010） | 主选 Developer 单 Scope | lowering 期 segfault；动态 extent gemm 静默错值 1.04e-2 | 用 Expert 单 Cube scope（§2.1） |
| K 尾忘传 `T.gemm(size=)`（陈旧尾值污染） | 只做 `T.copy` 子视图写而未限归约范围 | K 非整除用例静默错值 | 必须传 `size=[tile_m,tile_k,tile_n]`；§1.6.1 M2 `stale_tail_guard` 可检出（发散 ≈1.218e3） |
| Pipelined 体内条件构造崩溃（CG-2026-0001） | 死 tile 守卫误放入 K 循环内 | 编译 SIGSEGV | 守卫 `if` 放 `T.serial` 体内、`T.Pipelined` 之外 |
| `load_nd2nz` 跨步区域静默错值（CG-2026-0004） | B[e, n0:n0+tn, kb:kb+tk] 为跨步区域 | 静默错数 | 统一用 `T.copy` **slice 形态**（bit-exact）；Expert 路径优先 T.copy 而非 load_nd2nz |
| auto-multi-buffer 重排显式 buffer（CG-2026-0012 / TRAP-UB-dynsubview-dominance） | Expert 显式 alloc 结构 | codegen 变量找不到 / 结构破坏 | `pass_configs` 关闭 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION` 与 `NPUIR_ENABLE_AUTO_MULTI_BUFFER`（§7.3） |
| 越界写尾块 | 尾块未按动态 extent 裁剪 | 写坏下一专家行 | `store_fixpipe([tile_m,tile_n])` 严格裁剪（§5.4） |
| 调度表契约不一致 | 前导 kernel 的 `max_tiles`/`block_m`/行偏移单位与本核不符 | 寻址错位、错值 | 两函数共享 `max_tiles = numel//block_m + E` 与 `block_m`；`tile_row_offsets` 单位为行（§4.1 契约） |

### 9.3 特殊场景处理

- **偏斜分布**：大量专家仅 1 行 → tile 严重不满（1/64）；动态 extent 保证只搬 1 行、只算 1 行，浪费
  仅在 Cube 分形对齐必要的最小 tile。
- **极大规模**：核内串行 `num_local_tasks` 可达 ~1076（prefill down）——循环边界静态，启动开销摊薄。
- **极小 shape**（smoke numel=64）：逻辑核数 5 < 物理核数，仅 5 核参与，正确性优先。
- **两 kernel 协同**：前导 kernel 先 launch 完成，无 CPU 回读；`total_tiles` 全程设备侧。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（两 kernel 固定开销）**：launch + 表 GM 往返（≈6.1 KB）+ 跨 kernel barrier 的精确墙钟未实测
  （§1.6.0 roofline 标注「未文档化假设 + 估算依据」）；Stage 4 实测，若 > B 访存 1% 则评估融合可行性
  （但当前工具链无融合路径，CG-2026-0026）。
- **R-2（`T.Pipelined` vs `T.serial`）**：Stage 3 正对照以 `T.serial` K 循环跑通；`T.Pipelined(num_stages=2)`
  与 Expert 单 Cube scope 的组合稳定性待实测，不稳定则回退 `T.serial` + `T.rs` 显式流水（§6.3）。
- **R-3（动态 extent 分形对齐）**：动态 extent 的 `tile_m` 非 16 整数倍时 `T.copy`/`store_fixpipe` 的 NZ
  分形对齐行为 —— Stage 3 用 L0-1/L0-4 验证（若分形对齐拒绝非 16 倍数，则退化为「分形对齐向下取整 +
  剩余行标量补算」）。
- **R-4（NPU L2 对 B 权重的实际复用率）**：B 是否单遍 —— 影响是否达到 §1.6.0 的 12.5 ms 下界；Stage 4 实测。
- **R-5（调度表对接契约）**：胞弟 `_tile_scheduler_kernel` 原设计（内联进 GEMM 核）已被工具链推翻，
  须据本契约（§0.6 R1 / §4.1）改为**独立产出调度表**形态；两设计的 `max_tiles`/`block_m`/表 dtype/行偏移
  单位必须一致，集成时核对。
- **R-6（`max_tiles` 死 tile 比例）**：`max_tiles = numel//block_m + E` 上界在偏斜分布下的死 tile 比例
  （守卫成本）——目前可忽略。

### 9.5 能力缺口登记（Gap Rule）

- **CG-2026-0026**（codegen / BishengIR：`hivm.hir.sync_block_set` 属性名 `syn_instr_mode` 与随附
  `bishengir-compile` 期望的 `sync_instr_mode` 不一致）——**阻断一切跨引擎同 block 交接**；
  本设计据此把调度外移（§0.6 R1）。已由 Stage 3 登记（occurrences ≥1）。
- **CG-2026-0010**（codegen / Developer：persistent + gemm + 索引段崩溃）——本设计据此选 Expert 单 Cube scope；
  已由 Stage 3 扩展登记（第二证）。
- 本版设计的绕法（前导 kernel 产表 + Expert Cube 读表）**无需跨引擎同步**，从而在现有工具链可实现；
  上两条缺口保持 open，待工具链升级后可按 kb_stale_check 重验并考虑融合优化。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/
├── DESIGN.md               # 本设计文档（revision 2）
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1）
├── repro/
│   └── TRAP-SYNCBLOCK-ATTR-MISMATCH.py   # Stage 3 工具链缺口复现 + 正对照（底座证据）
├── _moe_grouped_gemm_kernel.py   # 算子实现（Stage 3：Expert 单 Cube scope 读表形态）
└── RETROSPECTIVE.md        # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision 2） | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核；底座改 Expert 单 Cube scope + 外部调度表） |
| `verify_equiv.py` | 已完成 | 等价性验证（M1/M2/M3 全 EQUIV_PASS；M2 覆盖 K 分块 + 陈旧尾 guard） |
| `repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` | 已完成（Stage 3） | CG-2026-0026 复现 + Expert Cube 正对照（max_diff 9.537e-07） |
| `_moe_grouped_gemm_kernel.py` | **待改写**（Stage 3） | 现文件为 v1 的（被阻断的）核内调度双 Scope 形态，须改写为 **Expert 单 Cube scope 读表** 形态（§3.3） |
| 前导 kernel `_tile_scheduler_kernel` | 胞弟单元（Stage 1/2） | 须据本契约（§0.6 R1 / §4.1）改为**独立产出调度表**形态 |
| 测试 | 待实现 | 复用 `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py` + §8.3 |

### 10.3 命名规范

- 项目目录：`examples/moe_grouped_gemm_nopad/`
- 算子目录：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/`
- 实现文件：`_moe_grouped_gemm_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md, revision 2）+ 等价性验证（verify_equiv.py）
2. ⬜ 前导 kernel `_tile_scheduler_kernel`（独立产表形态；胞弟单元）
3. ⬜ 本 GEMM 核实现（`_moe_grouped_gemm_kernel.py`，Expert 单 Cube scope 读表）+ L0 精度比对
4. ⬜ 两 kernel 协同测试（对接契约 L0-6）+ 完整分层套件

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_nopad.py`
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py`
- 源 op / 测试 / bench / manifest：
  `examples/TileOPs/tileops/ops/moe/moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`
- 胞弟单元（调度表产出者）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`
- Stage 3 底座证据：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`
- 同类实现：`examples/deepseek_v32/fp8_lighting_indexer.py`、`examples/mixcv/mixcv_mixkernel.py`、
  `examples/gemm/matmul_dynamic_shape.py`、`examples/gemm/example_gemm.py`、
  `testing/autotune/anneal/moe_grouped_gemm_anneal.py`
- API 文档：`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`、
  `docs/Tilelang.language/内存操作/T.store_fixpipe.md`、`docs/Tilelang.language/内存操作/T.load_nd2nz.md`、
  `docs/Tilelang.language/比较操作/T.min.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/algorithm-research.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/INDEX.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0001/0004/0010/0012/0015/0026）

---
