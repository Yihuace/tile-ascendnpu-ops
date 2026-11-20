# _fused_act_kernel 算子设计文档

> 迁移任务（harness，developer）。源算子：`tileops/kernels/moe/moe_grouped_gemm_persistent_3wg_fused_act.py`
> 中的 `_fused_act_kernel`（GPU/CUDA target，SM90 3-warpgroup persistent grouped GEMM + 融合激活）。
> 本项目目标：`target="npuir"`（Ascend NPU）。Stage 0 抽取件位于
> `examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad_3wg_fused_act/`。
> 迁移决策（§0）驱动 §1–§11 全部设计；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
>
> **配套单元**：调度表产出者 `_tile_scheduler_kernel` 由胞弟单元
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/` 设计；本设计 **消费** 其输出表
> （对接契约见 §0.6 R1 与 §4.1）。本算子的 **归一化 GEMM 核** 与胞兄
> `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/` 同底座（Expert 单 Cube scope）。

---

## 设计核心（GPU→NPU 关键重设计，4 条）

1. **3WG warp-specialized persistent（1 producer + 2 math WG，TMA ring + 双 WGMMA 累加器）→
   Expert 单 Cube scope**：NPU 无 warpgroup；producer 的 TMA 多级 ring 与 consumer 的 register
   累加映射为 Cube 单元的 `T.copy` 取数（GM→L1，隐式 ND2NZ）+ **双 fp32 L0C 累加器**（gate/up）
   + 每次 K 步两条 `T.gemm`（共享 A tile）。**D-3 探针实测该形态可用**（见 §9.6）。
2. **核内调度（thread-0 前缀扫描 `s_cum` + 每 tile 二分查找）→ 外置调度表**：源 3WG 核内
   调度不可在 NPU 核内实现（Developer persistent + gemm + 整型索引段 **lowering segfault**，
   CG-2026-0010；Expert 跨引擎同 block 交接被 CG-2026-0026 阻断），改为消费前导 kernel
   `_tile_scheduler_kernel` 产出的 GM 调度表（复用非融合变体的对接契约，block_m 由本核配置）。
3. **融合 epilogue（`act(gate)*up` 在 consumer register 内完成，`[numel,2*ffn]` 中间张量不落 GM）
   → 两 kernel 分解**：Cube 侧无逐元素/超越函数单元，且 CV 同 block 交接被 CG-2026-0026 阻断，
   故激活必须出 GEMM 核 —— Cube 核用 **双 `store_fixpipe`** 写 packed `gate_up[numel,2*ffn]`
   （fp32）到 GM，Vector 核（**复用已验证 packed silu/gelu 先例**）计算 `act(gate)*up`。
   **代价**：源刻意避免的中间张量落 GM（fp32，prefill 下 0.54 GB 写 + 0.54 GB 读 ≈ 0.85 ms，
   相对 B 访存 15.03 GB ≈ 12 ms 可忽略，§1.6.0 roofline）。
4. **TMA OOB 零填 + epilogue 行掩码 / 谓词零填 → 动态 extent 精确裁剪**：NPU 用
   `tile_m/tile_n/tile_k = T.min(...)` 只取/只写有效区，并把 `tile_k` 传入
   `T.gemm(size=[tile_m,tile_k,tile_n])` 限定归约范围；host 不做任何 padding（源 `forward` 对
   A 无 pad，K 非整除/ffn 非整除直接抛错，本设计统一用动态 extent 推广，见 §0.6 R2）。

> 迁移决策四态统计（对应 §0.5 的 15 行判定表）：**保留** 4 项（计算语义层 / I-O 契约 / NT 算法结构 /
> persistent 分核）；**等价替换** 4 项（SMEM tiling→L1 / tensor core→Cube `T.gemm` / 双累加器共享 A→双 L0C /
> 对齐预判→动态 extent 推广）；**重新设计** 4 项（TMA+warp-specialization 流水→单 Cube scope / 核内调度承载
> →外置 GM 表 / 融合 epilogue 归属→拆两 kernel / 取数写回→动态 extent）；**舍弃** 3 项（swizzle /
> warp specialization / 谓词零填充）。详见 §0.5。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性判定 + 重设计）
> 与 `references/algorithm-research.md`（Phase R）。本任务范围为 **`_fused_act_kernel`**（融合激活 GEMM 主核）；
> 配套调度表由胞弟单元设计（对接契约见 §0.6 R1）。

### 0.1 源算子语义（做什么）

**数学语义**（MoE no-pad 分组 GEMM + 融合 gate/up 激活，NT 布局）：

对每个专家 $e\in[0,E)$，其 token 数 $s_e$（`true_sizes[e]`）、紧致行偏移 $o_e$（`true_offsets[e]`）：

$$
g_e = A[o_e:o_e+s_e]\cdot B[e,\,0:\text{ffn}]^{T},\qquad
u_e = A[o_e:o_e+s_e]\cdot B[e,\,\text{ffn}:2\text{ffn}]^{T},\qquad
C[o_e:o_e+s_e] = \mathrm{act}(g_e)\odot u_e
$$

即 $g[i,n]=\sum_k A[i,k]B[e_i,n,k]$、$u[i,n]=\sum_k A[i,k]B[e_i,\text{ffn}+n,k]$，
$C[i,n]=\mathrm{act}(g[i,n])\cdot u[i,n]$；`act` ∈ {silu_and_mul, gelu_and_mul}。
$\sum_e s_e=\text{numel}$（紧致，无 pad；每行恰属一个专家）。

- **规约语义**：规约轴 = $K$，完整求和。源在 `C_gate/C_up`（fp32 `accum_dtype`）上按 `block_k`
  分块累加（`T.wgmma_gemm(..., transpose_B=True, clear_accum=(k==0))`），块间顺序累加。
- **dtype 语义**：A、B 为 fp16/bf16；中间累加 fp32；输出 C 与输入同 dtype。**gate/up 停在 fp32**，
  `act(g·)` 在 fp32 域求值，`T.cast(..., dtype)` **只在最后一步**（源 L449-450）。
- **激活语义**（源 `_act_expr`，compile-time 特化）：
  - `silu_and_mul`：$g/(1+e^{-g})$（直接 exp 形式，L36）；
  - `gelu_and_mul`：$0.5\,g\,(1+\mathrm{erf}(g/\sqrt2))$（**精确 erf**，L39）。
- **界语义**：源 `forward()` **要求** $K\bmod block\_k=0$（否则抛错）且 $\text{ffn}\bmod block\_n=0$（L146-149）；
  M（`numel`）**不 pad**，靠 TMA 描述符把越界行零填 + epilogue `i < arows` 掩码（L151-153、L451-467）。
  即源仅覆盖「K 对齐 + ffn 整除」路径；M 尾块由零填 + 掩码处理。
- **host 侧语义**：`forward` 只做输入校验 + `C=torch.zeros(numel,ffn)`，**不做任何 padding / 数据重排**。
  故无 host 预处理需内移；NPU 侧同样禁止 host 改动输入（ascend-constraints §4）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（优先移植源测试
`examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad_3wg_fused_act.py` 的 `_ref_fused_act`），
**不得复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| A | (numel, K) | 同 dtype | 紧致 permuted 激活，无 pad |
| B | (E, 2·ffn, K) | 同 dtype | 专家权重，前 ffn 行为 gate、后 ffn 行为 up（NT：对 B 转置后右乘） |
| true_sizes | (E,) | int32 | 每专家真实 token 数 $s_e$ |
| true_offsets | (E,) | int32 | 每专家在 A 中的紧致起始行 $o_e$ |
| **C（输出）** | **(numel, ffn)** | **同 dtype** | 源 `out_idx=[]`、`C_shape=(numel, ffn)`；行连续 |

> `activation` 为构造期 compile-time 常量（`silu_and_mul` / `gelu_and_mul`），非运行时张量。
> 无 `reverse` / `interleave` / `permute` 张量语义：gate = B 的前 ffn 行，up = 后 ffn 行，epilogue 为
> `act(gate)*up`，**无任何顺序反转或交织**（§0.3 数据流逐语句核对；token 级 permute/unpermute 属
> 外围 MoE 流水线 `examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`，不在本函数范围）。

### 0.3 实现算法解读（怎么算）

源为**单 kernel、3-warpgroup（3WG）persistent**：`T.Kernel(sm_count, threads=384)`，按 `block_m`
派发两种模板（`_fused_act_kernel` 工厂 L908-915：`block_m<=64` → pingpong；`block_m>=128` → cooperative）。

**计算步骤分解**（源码每个计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|---------|
| P1 | thread 0 串行前缀扫描 `s_cum[e+1]=s_cum[e]+ceil(s_e/block_m)`；`s_total=s_cum[E]*num_pid_n`（L271-277 / L655-661） | true_sizes | `s_cum[E+1]`（SMEM）, `s_total` | 调度（非数值语义） |
| P2 | 每 wave 由 `flat_id = k(sm_count·w+pid)` 经 `log2_up` 步二分 `s_cum` 解出 `(expert_id,row)`；算 `m_start=offsets[e]+row`、`n_start=n_tile·block_n`、`arows=min(block_m,s_e-row)`（L283-325 / L397-419） | s_cum, true_offsets, true_sizes | m_start/n_start/arows | tile→(专家,行) 映射 + 尾块裁剪 |
| P3 | producer WG：每个 $k$-step 发 3 条 TMA（A、B_gate、B_up）到 per-WG ring SMEM slot（L332-357 / L687-726） | A, B | A_smem, B_gate_smem, B_up_smem | 分块取数 |
| P4 | consumer WG：每 $k$-step **两条 WGMMA**（`A_smem×B_gate_smem^T`→`C_gate`；`A_smem×B_up_smem^T`→`C_up`），fp32 累加（L425-440 / L761-784） | A/B tile | C_gate, C_up(fp32) | $g=A B_{gate}^T$、$u=A B_{up}^T$ 的块累加 |
| P5 | **融合 epilogue**：`C_local_cast[i,j] = cast(act(C_gate[i,j]) * C_up[i,j], dtype)`（L448-450 / L528-530、L788-790 / L870-872） | C_gate, C_up | C_local_cast | $\mathrm{act}(g)\odot u$，再回舍 dtype |
| P6 | 写回：`arows==block_m` 走 `T.copy(C_local_cast → C_shared) → C[m_start,n_start]`（TMA store）；否则逐元素守卫 `if i<arows` 写（L451-467 / L537-541） | C_local_cast | C(GM) | 尾块裁剪写出 |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[A]  --TMA--> SMEM[A_smem (num_stages ring)]        ┐
GM[B]  --TMA--> SMEM[B_gate/B_up_smem (per-WG ring)]  ├─ wgmma(transpose_B, clear_accum=(k==0))
                                                       ┘        --> register C_gate/C_up (fp32)
                                   act(gate)*up (register, T.Parallel) --> C_local_cast (dtype)
GM[C]  <── TMA store / 守卫式逐元素写 ─────────────────────────────────┘
调度：thread0 SMEM 前缀扫描 s_cum --> 每 tile 二分查找 --> (expert,row,m_start,arows)
```

**循环与并行结构**：`T.Kernel(sm_count, threads=384)` persistent；核内 `T.serial(_max_waves)` 波次循环；
每波内 producer（tx<128）为 staticwave 调度 + 交错 K 循环（WG0 装填后 WG1 装填，L327-388），
consumer WG0/WG1（各 128 线程）各处理一个逻辑 tile；K 循环 `T.Pipelined(_k_iters, num_stages=0)`
（手动 barrier）。pingpong：每 CTA 2 tile/wave；cooperative：2 math WG 拆同一 tile 的 M 维（split-A）。

**host 侧逻辑**：`forward` 校验 K 整除 block_k 与 ffn 整除 block_n，`C=torch.zeros(...)`，调 kernel
（无 padding、无数据重排、无 `.item()` 回读）。编译配置：`pass_configs={TL_ENABLE_FAST_MATH: True,
TL_DISABLE_THREAD_STORAGE_SYNC: True}`；`compile_flags=["-O3","-DENABLE_BF16","-include",<anchor_helper.h>]`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM tiling（A/B 分块 TMA 入 ring） | 减 GM 重复访问 | 分块驻留 shared memory | SMEM/bank/容量 | 硬件强相关（NPU 有 L1/L0 等价） |
| O2 | Tensor Core（双 WGMMA，共享 A） | 矩阵算力 + 省 A 访存 | 两条 wgmma/K 步 | tensor core | 硬件强相关（NPU Cube 有等价） |
| O3 | TMA + `T.Pipelined(num_stages)` + warp specialization（1 producer + 2 consumer） | 访存/计算重叠 | 异步引擎多级流水 + 专用 producer | TMA + async copy + warpgroup | 硬件强相关（NPU 有 MTE 等价，无 warpgroup） |
| O4 | swizzled layout（`make_swizzled_layout`） | 消 SMEM bank conflict | 索引重排 | SMEM bank 结构 | 硬件强相关（NPU 不适用，NZ 代替） |
| O5 | persistent 固定 CTA 数（sm_count）+ 核内波次 | 消 CTA 启动开销、静态任务映射 | 常驻循环 | 廉价 CTA | 模式可移植（NPU 用固定核数 + 核内串行） |
| O6 | 核内调度器（SMEM 前缀扫描 + 每 tile 二分） | 消 host 同步、O(logE) 定位、`total` 留设备侧 | 核内整数索引段 | warp 并行二分 | 算法层可移植（形态需重设计） |
| O7 | **融合 epilogue（`act(gate)*up` in register，`[numel,2*ffn]` 不落 GM）** | 省中间张量写出+读回 | consumer register 上逐元素 | 大寄存器堆（fragment） | **算法意图保留，但依赖 register fragment** |
| O8 | 双 B 累加器（gate+up 共享 A，一次 K 循环） | A 只读一遍 | 每 K 步 2 条 wgmma | tensor core 双发射 | 硬件强相关（NPU Cube 有等价，需双 L0C） |
| O9 | M 无 pad + TMA OOB 零填 + epilogue 行掩码 | 免 host A 拷贝 | TMA 描述符 globalDim + 掩码写 | TMA OOB 零填 | 硬件强相关（NPU 用动态 extent 等价） |
| O10 | K/N 对齐预判（`forward` 直接拒绝非整除） | 走无谓词快路 | 编译期/运行期常量校验 | 无 | 算法层可移植（NPU 用动态 extent 推广） |

> 识别不出来 ≠ 不存在。以上 O1–O10 为源码显式优化，逐项在 §0.5 判定，未被识别者会被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（按专家分组双 NT 矩阵乘 + fp32 累加 + `act(g)*u` + 同 dtype 输出） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| I/O 契约（A/B/C 的 shape/dtype/行连续布局、gate=前 ffn 行） | 语义 | 无 | — | **保留** | §0.2 |
| NT 算法结构（`A·B^T`，不物化 B^T） | 算法 | 无 | 有（`T.gemm(b_transpose=True)`） | **保留** | §1.6.1 A2（机器验证 0 违反） |
| O1 SMEM tiling | 优化 | shared memory | 有（L1） | **等价替换** | GM→L1 分块 `T.copy`（Cube 操作数）；L1 512KB/32B 对齐（ascend-constraints §1） |
| O2 Tensor Core 双 WGMMA | 优化 | tensor core | 有（Cube） | **等价替换** | `T.gemm`→Cube MMA；分形 M/N≥16、K≥32（T.gemm.md §2.1） |
| O3 TMA + Pipelined + warp specialization | 优化 | TMA/async/warpgroup | 部分（MTE 有，warpgroup 无） | **等价替换 + 重设计** | `T.copy`（GM→L1，隐式 ND2NZ）+ `T.serial` K 循环（C 侧流水由 MTE/Cube 引擎交叠）；**warpgroup 概念舍弃**，producer/consumer 角色由同一 Cube scope 的顺序发射承担（§0.6 R2） |
| O4 swizzled layout | 优化 | SMEM bank | 无（NPU 无 bank conflict） | **舍弃** | NZ 分形由搬运/前端处理，手动 swizzle 无意义（migration-analysis.md §5.3） |
| O5 persistent 固定 CTA + 核内波次 | 优化 | 廉价 CTA 启动 | 部分（一维核超发被串行调度） | **保留/等价替换** | NPU 固定核数 = 物理核数（20）+ 核内 `T.serial`（core-split-strategy §1 要素③）；承接「常驻、零启动开销」意图 |
| O6 核内调度器（前缀扫描 + 二分） | 算法+优化 | warp 并行整数索引 | 有（算法层），**核内形态无** | **重新设计** | 外移到前导 kernel `_tile_scheduler_kernel`（GM 表）；依据 CG-2026-0010（核内索引段 + gemm 崩溃）/ CG-2026-0026（跨引擎交接不可编译）（§0.6 R1） |
| O7 融合 epilogue（register 内 `act(gate)*up`，不落 GM） | 算法+优化 | fragment/register 逐元素 | **无**（Cube 无逐元素/超越单元；CV 交接被阻断） | **重新设计** | 分解为 **Cube GEMM 核（双 `store_fixpipe` 写 packed `gate_up`）+ Vector 激活核**（复用已验证 packed silu/gelu）；**融合意图（中间张量不落 GM）在当前工具链不可保留**，代价量化见 §0.6 R5 / §1.6.0（§0.6 R5，依据 CG-2026-0026） |
| O8 双累加器共享 A | 优化 | tensor core 双发射 | 有（Cube 双 L0C 累加器） | **等价替换** | 双 `T.alloc_L0C` + 每 K 步两条 `T.gemm`（共享 `a_l1`）；**D-3 探针实测可用**（§9.6，max_diff 4.7e-09） |
| O9 M 无 pad + TMA OOB 零填 + 行掩码 | 优化 | TMA OOB 零填 | 无 TMA，但有动态 extent | **重新设计** | 动态 extent 精确裁剪（`tile_m/tile_n/tile_k=T.min(...)`）+ `T.gemm(size=)` 限定归约范围（§0.6 R2；T.copy.md §2.2.2） |
| O10 K/N 对齐预判（源拒绝非整除） | 算法 | 无 | 有（动态 extent） | **等价替换/推广** | 统一动态 extent 处理三方向尾块（比源更一般，超集，不改变语义） |
| warp specialization（producer 与 math WG 分离） | 实现 | warpgroup + named barrier | 无（NPU 无 warpgroup） | **舍弃** | NPU Cube scope 内的 MTE 引擎与 Cube 引擎由硬件/框架流水交叠，无需显式 producer 线程组（§0.6 R2） |
| 谓词零填充 | 算法 | 无 | 有（数学恒等，但无必要） | **舍弃** | 动态 extent 只取有效区，无需零填（§0.6 R2） |

> 每项处置依据见「NPU 对应方案 / 依据」列。舍弃项（O4 swizzle、warp specialization、谓词零填充）理由已写明。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计 / 形态确认」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**前导调度表（GM）+ Expert 单 Cube 双累加器 GEMM 落 packed `gate_up` +
独立 Vector packed 激活核**）。

**重设计项 R0：编程模式——源单 kernel 3WG → NPU 混合（Cube 段 Expert + Vector 段 Developer，两 kernel）**

- **背景**：源为 GPU 单 kernel（CUDA SM90，无 Developer/Expert 概念），producer/math WG 全在一个 kernel 内。
- **NPU 新形态**：Cube GEMM 核 = **Expert**（`T.Scope("Cube")`，仅 `alloc_L1/alloc_L0C` + `T.copy` +
  `T.gemm` + `store_fixpipe`；**无 Vector scope、无跨引擎同 block 交接**）；Vector 激活核 = **Developer**
  （纯 Vector，v-prefix 链，复用已验证先例）。两核同 stream 顺序 launch。
- **语义保持论证**：编程模式不改变数值语义（同一 `T.gemm` NT + fp32 累加、同一 `act` 公式、同一 dtype 路径）。
- **依据**：CG-2026-0010（Developer persistent + gemm + 索引段 segfault）、CG-2026-0026
  （跨引擎同 block 交接 `syn_instr_mode` 不可编译，官方 mixcv 样例复现）。

**重设计项 R1：调度承载方式（O6）——源核内前缀扫描/二分 → 外置调度表（GM）**

- **源方案**：核内 thread-0 用 SMEM 前缀扫描建 `s_cum`，每个逻辑 tile 再 `log2_up` 步二分查 `(expert,row)`；
  性能意图 = **消除 host 同步**（`total` 留设备侧）+ **O(log E) 定位**（免每 tile O(E) 扫描）。
- **NPU 新算法**：**消费**胞弟 `_tile_scheduler_kernel` 产出的 3 张 GM 表
  （`tile_expert_ids[max_tiles]`、`tile_row_offsets[max_tiles]`、`total_tiles[1]`，全 int32）——
  本核按 `bx=tile_id//num_pid_n`、`by=tile_id%num_pid_n` 读表，O(1) 寻址得到 `m_start/n_start/tile_m/tile_n`。
  两 kernel 同 stream 顺序 launch，前导完成本核再执行（kernel 边界隐式顺序，无 in-block 同步）。
- **语义保持论证**：调度表内容与源核内二分**逐元素一致**（同一 `cum[e+1]=cum[e]+ceil(s_e/block_m)`、
  同一落点、同一死 tile 哨兵），故 tile→(专家,行) 映射逐位一致；变化仅在**求值时机**（前导 kernel 一次物化）。
  边界：$s_e=0$ 专家不产生 tile；`total_tiles ≤ max_tiles`；死 tile 由单比较 `if bx < total_tiles[0]` 跳过。
- **对接契约（本核消费）**：见 §4.1；`max_tiles = numel // block_m + E` 与 `block_m` 为两函数共享的编译期
  常量（**本核 block_m=128**，故前导 kernel 须以 block_m=128 实例化）；表 dtype 全 int32；
  `tile_row_offsets` 单位为**行**（已乘 block_m）。胞弟设计见
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（其非融合变体用 block_m=64；本算子需按 §4.1 契约取 block_m=128）。
- **依据**：CG-2026-0010 / CG-2026-0026；胞兄 Stage 3 实测该形态 max_diff 9.537e-07
  （`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`）。

**重设计项 R2：GEMM 主循环（O1/O2/O3/O8）——3WG warp-specialized → Expert 单 Cube 双累加器**

- **源方案**：producer WG 发 TMA 到 per-WG ring；两个 math WG 各持 register `C_gate/C_up` 累加器，
  每 K 步两条 `wgmma`（共享 A），`T.Pipelined(..., num_stages=0)` + 手动 barrier。
- **NPU 新算法**：`T.Scope("Cube")` 内：
  1. `a_l1 = T.alloc_L1((block_m, block_k))`、`bg_l1/bu_l1 = T.alloc_L1((block_n, block_k))`；
     `c_gate/c_up = T.alloc_L0C((block_m, block_n), "float32")`（**双 fp32 累加器**）；
  2. 对每个逻辑 tile、每个 $k$-块：`T.copy` 取 A tile（动态 extent）、B_gate tile（`B[e,n_start:...]`）、
     B_up tile（`B[e,ffn+n_start:...]`）到 L1（隐式 ND2NZ）；
  3. `T.gemm(a_l1, bg_l1, c_gate, size=[tile_m,tile_k,tile_n], b_transpose=True, initC=(k==0))`；
     `T.gemm(a_l1, bu_l1, c_up, size=[tile_m,tile_k,tile_n], b_transpose=True, initC=(k==0))`；
  4. 双 `T.rs("PIPE_FIX")` 内 `T.store_fixpipe(c_gate, GU[m_start, n_start], [tile_m,tile_n], enable_nz2nd=True)`
     与 `T.store_fixpipe(c_up, GU[m_start, ffn+n_start], [tile_m,tile_n], enable_nz2nd=True)`。
- **语义保持论证**：数值等价 —— `c_gate/c_up` 累加顺序（K 分块顺序）与源一致（均 fp32 顺序累加），
  `act` 输入仍为 fp32 gate/up（见 R5）；边界：动态 extent 只写有效 `[tile_m,tile_n]`，nor-pad 尾块不越界邻专家。
  映射变化：producer/consumer 分离 → 单一 Cube scope 的顺序发射（数值无影响）。
- **D-3 探针实测**：Expert 单 Cube scope + 双 L0C + 两条 `T.gemm`/K 步 + 双 `store_fixpipe`（写 packed
  `gate_up` 的 disjoint 列区间）在（64,128）/（128,128）/（128,64）三配置下**全部编译通过且数值正确**
  （gate_up max_diff ≤ 5.6e-09），脚本 `history_version/design_probe_dual_acc.py`（§9.6）。

**重设计项 R3：任务划分与分核（O5）——超发 grid + persistent CTA → 固定物理核数 + 核内串行**

- **源方案**：`T.Kernel(sm_count)` persistent + 核内 `T.serial(_max_waves)` 波次（sm_count = SM 数）。
- **NPU 新算法**：`T.Kernel(nums_kernel, is_npu=True)`，`nums_kernel = min(num_logical, P)`（P = 实查物理核数 20）；
  核内 `for t in T.serial(num_local_tasks)`（`num_local_tasks = ceildiv(num_logical, nums_kernel)`，**编译期常量**）
  以块连续方式处理多逻辑 tile；死 tile 单比较守卫（位于 `T.serial` 体内、K 循环外，规避 CG-2026-0001）。
- **语义保持论证**：分核只改「哪个核处理哪个 tile」的调度，不改变 tile 内计算（同一 `T.gemm` + 同一 fp32 累加
  + 同一裁剪写回），故数值逐位一致。边界：循环边界静态；任务↔核映射为纯调度。

**重设计项 R4：取数/写回（O9/O10）——零填 + 谓词守卫 → 动态 extent 精确裁剪**

- **源方案**：A 无 pad（TMA OOB 零填越界行 + epilogue `i<arows` 掩码）；源直接拒绝 K 非整除 / ffn 非整除。
- **NPU 新算法**：全方向动态 extent：`tile_m=min(block_m,s_e-row)`、`tile_n=min(block_n,ffn-n_start)`、
  `tile_k=min(block_k,K-k·block_k)`；`T.copy` 切片用上述动态值；**K 方向动态 extent 必须同时作
  `T.gemm(..., size=[tile_m,tile_k,tile_n])`**（否则缓冲 `[tile_k,block_k)` 的陈旧尾值被按静态 block_k 归约，
  正确性关键，§1.6.1 A1 的 stale-tail guard 确认缺失即发散 ≈2.8e3）；`store_fixpipe` 用 `[tile_m,tile_n]`。
  本设计**推广**源契约（源仅 K 对齐/ffn 整除），非整除输入同样正确（超集，不改变已覆盖路径语义）。
- **语义保持论证**：越界贡献恒为 0（零填或被裁的积项），「只算有效区」与「零填全算再裁」等价——
  以「归约范围 = 有效 tile_k」为前提（§1.6.1 A1，机器验证 0 违反；含 stale-tail guard）。no-pad 关键边界：
  尾块多余行属下一专家，动态 extent 只写 `[0,tile_m)` 行，杜绝越界写坏邻专家。
- **依据**：T.copy.md §2.2.2（动态 extents 尾块支持）；T.store_fixpipe.md（`size` 参数）；mixcv_mixkernel.py。

**重设计项 R5：融合激活 epilogue（O7）——源 register 融合 → Cube packed 落 GM + Vector 激活核**

- **源方案**：consumer WG 在 register `C_local_cast[i,j] = cast(act(C_gate[i,j])*C_up[i,j], dtype)`
  （`T.Parallel(block_m, block_n)`），**`[numel, 2·ffn]` 中间张量从不落 GM**（源 docstring L8-9）。
- **NPU 新算法（两 kernel 分解）**：
  1. **Cube GEMM 核**（R2）把 `c_gate`、`c_up`（fp32，L0C）经双 `store_fixpipe` 写入 GM 的
     **packed `gate_up[numel, 2·ffn]`（fp32）**：gate 到列 `[n_start, n_start+tile_n)`、up 到列
     `[ffn+n_start, ffn+n_start+tile_n)`（与源 B 的 gate||up 布局一致）；
  2. **Vector 激活核**（Developer）读 packed `gate_up`，按 `silu_and_mul` / `gelu_and_mul` 语义
     `C[m,n] = act(gate_up[m,n]) · gate_up[m,ffn+n]`，单次回舍到输出 dtype。该核**直接复用本仓已验证
     packed 激活先例**：`examples/silu_and_mul/_make_fused_gated_explicit/`（silu，5 条 v-prefix 链）、
     `examples/gelu_and_mul/_make_fused_gated_explicit/`（gelu，odd-symmetry + A&S erf）。
- **语义保持论证**：
  - `gate_up` 以 **fp32** 存放 → `act` 的输入与源一样是 fp32 累加值，**最后一次回舍**（源在 register 域
    同样最后才 `cast`），dtype 语义一致（§1.6.1 A3/A4 机器验证采用公式等价）。
  - 中间张量落 GM **不改变数学结果**（逐元素搬运），只引入不可消除的 GM 往返。
  - 边界：packed 布局的列切片 `[n_start,n_start+tile_n)` 与 `[ffn+n_start,...)` 不相交；动态 extent 裁剪一致。
- **融合意图的处置（诚实披露）**：源定义性优化（O7「`[numel,2·ffn]` 不落 GM」）在当前工具链
  **不可保留**——Cube 侧无逐元素/超越函数单元，且 CV 同 block 交接被 CG-2026-0026 阻断。
  **代价量化**：中间张量 fp32 写 + 读 = 2 × numel × 2·ffn × 4B；prefill 下 = 2 × 32768 × 4096 × 4
  = 1.07 GB ≈ **0.85 ms**（÷ 1.26 TB/s 混合地板），相对 B 访存 15.03 GB（≈11.9 ms）**约 7%**，
  相对全算子 ≈ 4%。承接：源「**省去中间张量往返**」的意图由「**双累加器共享 A（O8/A5）+ NT 不物化 B^T（A2）**」
  部分承接；被牺牲的仅是 epilogue 侧的中间张量往返。工具链修复（CG-2026-0026 closed）后可按
  `kb_stale_check.py` 重验并复原单核融合（§9.5）。
- **依据**：CG-2026-0026（跨引擎同 block 交接不可编译，官方样例复现）；CONST-store-fixpipe-gm-only
  （L0C 跨引擎传输强制 GM 往返）。

**重设计项 R6：无（算法层矩阵乘本身）**。NT 分组 GEMM + fp32 分块累加在 Cube 上有等价能力（§0.5 保留/
等价替换）；R2–R5 只改实现形态与融合归属，不换算法族。

### 0.7 标杆实现

- 源算子：`tileops/kernels/moe/moe_grouped_gemm_persistent_3wg_fused_act.py`
  （GPU 参考仓 `/tmp/opencode/TileOPs-ref`，`_fused_act_kernel` + `_make_pingpong_fused_act_kernel` /
  `_make_cooperative_fused_act_kernel` + `_act_expr`）；op 包装
  `tileops/ops/moe/routed_expert/moe_grouped_gemm_nopad_fused_act.py`。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad_3wg_fused_act/`。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad_3wg_fused_act.py`
  的 `_ref_fused_act`（每专家 PyTorch NT matmul + `F.silu`/`F.gelu`，fp32 累加后回落输入 dtype）。
- 本仓 NPU 底座先例（结构参照，非数值基准）：
  - 归一化 GEMM：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/`（Expert 单 Cube scope 读表，
    Stage 3 max_diff 9.537e-07）；调度表产出者：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/`。
  - 激活：`examples/silu_and_mul/_make_fused_gated_explicit/`、`examples/gelu_and_mul/_make_fused_gated_explicit/`。
- golden 以 §0.1 语义为唯一依据，移植上述源测试参考实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_fused_act_kernel`（MoE no-pad 分组 GEMM + 融合 gate/up 激活，NT 布局，`target="npuir"`）。

### 1.2 功能描述

对紧致（无 pad）排列的 MoE 激活按专家分组做**两个共享 A 的 NT 矩阵乘**（gate = `A·B[:, :ffn]^T`、
up = `A·B[:, ffn:]^T`），并按 `C = act(gate)·up`（`act` ∈ {silu_and_mul, gelu_and_mul}）融合写出
`[numel, ffn]`。本设计的 NPU 实现为**两 kernel 分解**：Cube 分组 GEMM 核（Expert 单 Cube scope，双累加器）
落 packed `gate_up[numel, 2·ffn]`（fp32）→ Vector 激活核（Developer，packed 链）算 `C`。配套的 tile 调度
由**前导 kernel `_tile_scheduler_kernel` 独立产出**（GM 表，本核读表执行；§0.6 R1）。

### 1.3 数学公式

$$
g_e = A[o_e{:}o_e{+}s_e]\,B[e,0{:}\text{ffn}]^{T},\quad
u_e = A[o_e{:}o_e{+}s_e]\,B[e,\text{ffn}{:}2\text{ffn}]^{T},\quad
C[o_e{:}o_e{+}s_e] = \mathrm{act}(g_e)\odot u_e
$$

即 $C[i,n]=\mathrm{act}\!\big(\sum_k A[i,k]B[e_i,n,k]\big)\cdot\sum_k A[i,k]B[e_i,\text{ffn}+n,k]$。
源清单 roofline（`examples/TileOPs/tileops/manifest/moe.yaml`）：
`flops = 4*numel*ffn*k`，`bytes = (numel*k + E*2*ffn*k + numel*ffn)*elem_bytes + 2*E*4`。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源算法结构的差异逐条注明）。

**三 kernel（前导调度 + 本算子两 kernel）**：

1. **调度表产出（前导 kernel `_tile_scheduler_kernel`，胞弟单元）**：`true_sizes` → `tile_expert_ids[max_tiles]`
   / `tile_row_offsets[max_tiles]` / `total_tiles[1]`（int32，落 GM）。
   与源**结构差异**：源在 GEMM 核内做前缀扫描 + 二分；本设计外移（§0.6 R1）。
2. **Cube 分组 GEMM 核（Expert 单 Cube scope，本算子）**：对每个逻辑 tile（`tile_id=core_id*num_local+t`，
   `bx=tile_id//num_pid_n`、`by=tile_id%num_pid_n`，死 tile 单比较跳过）：读表得 `(e,row)`，算
   `m_start=true_offsets[e]+row`、`n_start=by*block_n`、动态 extent `tile_m/tile_n/tile_k`；K 循环内
   `T.copy` 取 `a_l1 / bg_l1 / bu_l1`（GM→L1，ND2NZ），每 K 步两条 `T.gemm`（共享 `a_l1`，分别入 `c_gate/c_up`
   fp32 L0C），末了双 `store_fixpipe` 把 `c_gate/c_up` 写入 packed `gate_up[m_start, n_start]` /
   `gate_up[m_start, ffn+n_start]`（fp32 GM）。
   与源**结构差异**：3WG warp-specialized → 单 Cube scope 顺序发射（§0.6 R2）；核内调度 → 外部读表（R1）；
   动态 extent 替代 TMA OOB 零填 + 谓词（R4）；融合 epilogue 输出改为 packed `gate_up` 落 GM（R5）。
3. **Vector 激活核（Developer，本算子）**：读 packed `gate_up`，`C[m,n]=act(gate_up[m,n])·gate_up[m,ffn+n]`
   单次回舍；复用已验证 packed silu/gelu 先例。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[前导 kernel _tile_scheduler_kernel（胞弟单元，Vector 整数索引）]
  true_sizes[E] --前缀扫描/查找--> GM[tile_expert_ids(max_tiles) int32]
                                  GM[tile_row_offsets(max_tiles) int32]
                                  GM[total_tiles(1) int32]
        │  （同一 stream 顺序 launch：前导完成后 GEMM 核执行，无 in-block 同步）
        ▼
[本算子 kernel G：Expert 单 Cube scope]
  标量读 GM 表/标量 ──> (e, row, m_start, n_start, tile_m, tile_n, tile_k)
A[numel,K] ┐
B[E,2ffn,K]┼─动态 extent T.copy(GM→L1,ND2NZ)─> L1[a_l1/bg_l1/bu_l1]
           │        ├─ T.gemm(a_l1,bg_l1,c_gate,size=[tm,tk,tn],b_transpose) ─> L0C[c_gate fp32]
           │        └─ T.gemm(a_l1,bu_l1,c_up,  size=[tm,tk,tn],b_transpose) ─> L0C[c_up   fp32]
           │                          │ 双 store_fixpipe([tm,tn], nz2nd)
           ▼                          ▼
                          GM[gate_up(numel,2ffn) fp32]   (gate 列 [n_start,..)、up 列 [ffn+n_start,..))
                                  │  （同 stream 顺序 launch）
                                  ▼
[本算子 kernel V：Developer Vector 激活]
  T.copy 取段(gate 段 + up 段) → v-prefix 链(act) → T.copy 写回
                                  ▼
                          GM[C(numel,ffn) 输入 dtype]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（MoE no-pad 分组 fused gate/up GEMM——两个共享 A 的 NT 矩阵乘 + 逐元素融合激活）
> 的**算法族**。输入/源算法（单 3WG 持久化核 + 核内调度 + register 融合 epilogue）只是基线候选之一。
> 深度分级：矩阵 + 多步 + 融合类 → **完整调研**。
> **可实现性前置约束**：本工具链禁止跨引擎同 block 交接（CG-2026-0026，官方 mixcv 样例复现），且
> Developer persistent + gemm + 整型索引段 lowering segfault（CG-2026-0010）→ 任何「Cube 与 Vector
> 在同一 block 内交接（含源式 register 融合 epilogue）」的候选**不可实现**，调研在可实现族内择优。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | 基线（源） | 单 kernel 3WG persistent：核内前缀扫描 + 二分；TMA ring + 双 WGMMA；**register 融合 epilogue**（`[numel,2ffn]` 不落 GM） | — | — | ✅ 基线 |
| C1 | **外置调度表 + Expert 单 Cube 双累加器 GEMM 落 packed `gate_up` + 独立 Vector 激活核**（选定） | 前导 kernel 产 3 张 GM 表；Cube 核读表 + 双 L0C 累加器 + 双 `store_fixpipe` 写 packed `gate_up`(fp32)；Vector 核 packed 激活 | 数学恒等（映射/公式位精确；激活公式见 A3/A4） | 消核内调度/谓词；双累加器共享 A（省 A 一遍）；**但中间张量落 GM** | ✅ **选定** |
| C2 | 单 Developer kernel（auto CV-split）内融合 epilogue + 外部调度表 | 一个核内 Cube GEMM + Vector 激活自动切分 | 数学恒等 | 保留融合（中间不落 GM） | ❌ **撤回**：Developer persistent + gemm + 索引段 segfault（CG-2026-0010，第二证）；非持久超发 grid 在 NPU 被串行调度 |
| C3 | Expert 双 scope 单 kernel（Cube GEMM + Vector 激活 + `sync_block_set/wait` 交接） | 同一 block 内 CV 交接 | 数学恒等 | 保留融合 | ❌ **撤回**：CG-2026-0026（`syn_instr_mode` 不可编译，官方样例复现）；去同步则竞态（max_diff 取值集合含 9.5e-7/1.26e-2/全零） |
| C4 | C1 但中间张量用 **bf16/fp16**（而非 fp32） | `store_fixpipe` f32→bf16 量化写 packed `gate_up` | 容差内等价（多一次前置舍入） | 中间访存减半（1.07 GB → 0.54 GB） | ✅ 对比（**弃选**：精确性优先，见 §1.6.1 A3/A4 与 §1.6.3 L 候选） |
| C5 | 单累加器 + 两段 K 循环（gate 循环后 up 循环，复用同一 L0C） | 同一 L0C 顺序算 gate/up | 数学恒等 | 省一个 L0C 累加器（L0C≤32768）| ✅ 对比（**弃选**：A tile 被 K 循环重复装载 → A 访存翻倍；且两段 K 循环间 store_fixpipe→gemm 有 WAW 依赖） |
| C6 | split-K（K 切多段跨核累加） | K 维切分给多核 | 容差内等价（累加分组变化） | 增并行度 | ✅ 对比（**弃选**：算子 B 访存受限，split-K 使 A/B 访存 ×nsplit 且需跨核归并） |
| C7 | 无搜索映射（expert=task_id//max_tiles_per_expert） | 网格直接编码专家 | 数学恒等 | 省查找 | ✅ 对比（**弃选**：死 tile 比例暴增） |
| C8 | host 侧调度（CPU 算表下发） | 调度表在 host 生成 | 数学等价 | 无 | ❌ 弃选：需 `true_sizes` D2H + 表 H2D，引入 CPU 同步；host 禁改输入（ascend-constraints §4） |

**R2 在线算法**：**有**，且是本题天然形态。矩阵乘在 K 维的**分块累加本身就是单遍流式**（running fp32
累加器 `c_gate/c_up`，O(1) 状态）；调度器的排他前缀扫描 `cum[e+1]=cum[e]+cnt[e]` 是**单遍在线 running sum**
（在胞弟单元内）；激活核是**逐元素映射**（单遍，无跨元素依赖）。结构判据：输出 C 不含跨 tile 全局依赖，
M/N 两维可任意分块并行，K 是唯一规约维且可单遍累加 —— **不存在需要多遍物化的候选**（不像 softmax/attention
需要两遍 max/sum）。源算法已用「分块 K 累加 + 前缀扫描」即直接证据。收益口径：扫描遍数 = 1
（A 一遍、B 一遍、C 一遍、中间张量一遍写+一遍读，见 R3）；中间缓冲 O(1)（L0C tile + 前导 kernel 的小表 `cum`）。

**R3 复杂度对比**（四口径；代表 shape = deepseek-v3-prefill-gate-up：numel=32768，E=256，ffn=2048，K=7168，
bf16，elem=2B；block_m=128，block_n=128，block_k=64。"访存"含 A 读 + B 读 + C 写 + 中间张量 GM 往返，
口径见脚注）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| C0 基线（源，单核 3WG + register 融合） | 1.925e12 | A 0.47G + B **15.03G** + C 0.134G（中间不落 GM）≈ **15.64G** | A 1（L2 助读放大）/ B 1（block_m=128，每专家 1 M-tile）/ C 1 | L0C(register) 双 tile；SMEM ring | 高；无跨核归并 |
| C1 **选定（外置表 + 双累加器 + packed GM + Vector 激活）** | 1.925e12 | A 0.47G + B **15.03G** + 中间 **1.07G**（fp32 写 0.54+读 0.54）+ C 0.134G ≈ **16.71G** | A 1（动态 extent，L2 助读 ×16） / B 1 / 中间 1 写 + 1 读 / C 1 | L0C 双 [128,128]fp32 = 128KB（贴上界）；L1 (16+8+8)KB×stages | 高；核内串行，**无跨核归并、无跨引擎同步** |
| C2 单 Developer 融合（不可实现） | 1.925e12 | 同 C0（中间不落 GM） | 同 C0 | 同 C0 + auto CV-split | 工具链**不可编译/崩溃**（CG-2026-0010）；撤回 |
| C3 Expert 双 scope 融合（不可实现） | 1.925e12 | 同 C0 | 同 C0 | 同上 + GM workspace | **不可编译**（CG-2026-0026）；撤回 |
| C4 C1 + bf16 中间 | 1.925e12 | 同 C1 但中间 **0.54G** ≈ **16.18G** | 中间 1 写 + 1 读 | 同 C1 | 同 C1（精确性降，§1.6.1） |
| C5 单累加器两段 K | 1.925e12 | A **0.94G**（K 循环重载翻倍）+ B 15.03G + 中间 1.07G + C 0.134G ≈ **17.18G** | A 2 | L0C 单 [128,128]fp32 = 64KB | 同 C1 |
| C6 split-K（nsplit） | 1.925e12 | A×nsplit + B×nsplit + partial 写/读×nsplit | ≥ nsplit | partial 缓冲 [numel,ffn]×nsplit×4B（可落 GM） | 并行度 ×nsplit；**需跨核归并** |
| C7 无搜索映射 | 1.925e12 | 同 C1 | 同 C1 | 同 C1 + 无表 | 逻辑 tile 数 = E×max_tiles_per_expert×num_pid_n，死 tile 比例暴增 |
| C8 host 调度 | 同 C0 | + `true_sizes` D2H + 表 H2D | + 2 次总线往返 | host 内存 | **CPU 同步阻塞** |

口径脚注：FLOPs 只算乘加（4·numel·ffn·K）；访存 = GM 侧不可消除的输入读 + 输出写 + 必要中间往返（不含
L1/L2 复用命中，不含系数项；A 的「×16」指 M-tile 内 16 个 N-tile 复用 A tile、预期由 L2 吸收，未计入强制
HBM 流量）；扫描遍数指同一数据在 GM 侧被读的次数；中间缓冲峰值以片上（L1/L0C/UB）为准。
**主导项为 B 权重访存（15.03 GB，占 ~90%），算子是访存受限**，故 R3 排序以 B 遍数/复用为主导，而非 FLOPs。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰按 negative-claim-evidence.md §1 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C0 基线 | Cube ✅ + Vector(激活) ❌ 同 block | L0C(register) ✅ / SMEM ✅ | TMA OOB 零填 ✅ | 波数编译期常量 ✅ | 需 CV 融合 ❌ | ⚠️ 源形态在 NPU 不可实现（见 C2/C3） |
| C1 **选定** | Cube ✅（GEMM）+ Vector ✅（激活，独立核） | L0C 128KB（贴界）✅ / L1 64KB@2stage ✅ / UB 激活 ≤192KB ✅ | 动态 extent 全方向 ✅ | nums_kernel/num_local 编译期常量 ✅ | `T.serial`/`T.Pipelined` ✅；**无跨引擎同步** ✅ | ✅ 最亲和且**可实现**（D-3 实测） |
| C2 单 Developer 融合 | 需 auto CV-split | — | — | — | ❌ persistent+gemm+索引段 segfault（CG-2026-0010） | ❌ 撤回 |
| C3 Expert 双 scope 融合 | 需 CV 同 block 交接 | — | — | — | ❌ 无可用同步原语（CG-2026-0026） | ❌ 撤回 |
| C4 C1+bf16 中间 | 同 C1 | 同 C1 | ✅ | ✅ | ✅ | ❌ 弃选：多一次前置舍入（精确性优先，§1.6.1） |
| C5 单累加器两段 K | Cube ✅ | L0C 64KB ✅ | ✅ | ✅ | ✅ | ❌ 弃选：A 访存翻倍 + WAW 依赖（R3） |
| C6 split-K | Cube ✅ | partial 需落 GM | ✅ | ✅ | 跨核归并 | ❌ 弃选：B 访存受限，split-K 使 A/B ×nsplit |
| C7 无搜索映射 | Cube ✅ | 同 C1 | ✅ | ✅ | ✅ | ❌ 弃选：死 tile 暴增 |
| C8 host 调度 | CPU | — | — | — | ❌ CPU 同步 | ❌ 弃选 |

**调研结论**：选定 **C1 —— 前导 kernel `_tile_scheduler_kernel` 独立产出调度表（GM）+ 本算子 Cube 核
（Expert 单 Cube scope，双 fp32 L0C 累加器，NT `T.gemm`，动态 extent）落 packed `gate_up`(fp32) +
独立 Vector 激活核（Developer，packed silu/gelu 链）**。关键依据：
(i) **可实现性是前置约束** —— CV 同 block 交接不可编译（CG-2026-0026，官方样例复现）、Developer
persistent+gemm 崩溃（CG-2026-0010），故 C2/C3（保留 register 融合）不可实现，其收益主张（中间张量不落 GM）
**撤回**；
(ii) R3 显示算子是 **B 访存受限**（B 15.03 GB 占 ~90%），C1 以 block_m=128 使 B 单遍、A/C 各单遍，
并以双累加器（共享 A）避免 A 翻倍；
(iii) R4 显示 C1 全方向满足 NPU 约束（无 host 改动、循环边界静态、L0C/L1 容量内、可流水、**无跨引擎同步**），
且关键形态已由 **D-3 探针实测可用**（§9.6）；C4/C5/C6/C7/C8 分别因精确性 / A 访存翻倍 / 访存倍增+归并 /
死 tile 暴增 / CPU 同步被弃选。
**与基线结构差异**：融合 epilogue（register）→ 拆为 Cube packed 落 GM + Vector 激活核；核内调度 → 外部表；
TMA 零填/谓词 → 动态 extent；3WG warp specialization → 单 Cube scope。
**调研范围**：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（ALG-gemm〔Cube 分形 vs
Vector 规约 / split-K 分块累加〕、ALG-attention〔online 单遍与中间缓冲口径〕命中行均已评估）；
pattern-library（constants.md 硬件常数、traps-runtime.md TRAP-DEVMODE-PERSIST-GEMM、attention.md PL-1.16
Expert 双 scope 绕法与 CG-2026-0026 阻断）；源算子 §0.3/§0.4；本仓 examples
（moe_grouped_gemm_nopad 两单元、mixcv/mixcv_mixkernel.py、silu_and_mul / gelu_and_mul、
gemm/matmul_dynamic_shape.py）；capability-gaps.md（CG-2026-0010 / 0026 / 0027）。**已核查未命中更优替代**。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1→L1 分块（`T.copy`）；O2→Cube `T.gemm`（双）；
O3（TMA+Pipelined+warp specialization）→**等价替换 + 重设计**：`T.copy` + `T.serial`/`T.Pipelined`（MTE 与
Cube 引擎交叠），warp specialization 概念舍弃；O4（swizzle）**舍弃**（NZ 由搬运转换）；O5（persistent）→固定
物理核数 + 核内串行（承接常驻意图）；O6（核内调度器）→外置表（承接「零 CPU 同步 + O(log E) 定位」意图：
表一次物化，GEMM 核 O(1) 读表）；O7（融合 epilogue）→**承接意图部分实现**：双累加器共享 A + NT 不物化 B^T
承接「省访存」意图，**牺牲**「中间张量不落 GM」（工具链硬约束，§0.6 R5，代价 ≈0.85 ms）；O8（双累加器共享 A）
→双 L0C 等价替换；O9（TMA OOB 零填 + 掩码）→动态 extent；O10（对齐预判）→动态 extent 推广（超集）。

**设计期 roofline（D-2）**：以 §1.6.3 选定布局 + pattern-library/constants.md 实测常数估算下界：
- **流量项（GEMM 核，主导）**：B 15.03 GB（block_m=128，每专家 1 M-tile → 单遍）+ A 强制 HBM ≈0.47 GB
  （N-major 任务序使 A M-tile 于 L2 助读，×16 重复预期命中）+ 中间写 0.54 + 读 0.54 + C 0.134 ≈ **16.7 GB**，
  ÷ CONST-mte2-degradation 混合流量地板 **~1.26 TB/s**（工具链 tilelang 0.1.2+ed787bb / CANN 8.5.0）
  ≈ **13.3 ms**；**若 A 的 ×16 重复未被 L2 吸收**，则 A ≈7.5 GB → 总 23.8 GB ≈ **18.9 ms**；
- **流量项（Vector 激活核）**：读 0.54 GB（fp32 packed）+ 写 0.134 GB ≈ 0.67 GB ≈ **0.53 ms**；
- **流量项（前导 kernel）**：`true_sizes` 读 1 KB + 3 张表写 ≈6.1 KB ≈ **< 0.1 µs**（可忽略）；
- **发射项**：3 次 kernel launch + vector op 链（CONST-vector-launch-overhead ~0.5µs/op）合计 **< 0.5 ms**；
- **容量项**：L0C 双 [128,128]fp32 = 128KB（**贴 128KB 上界**，D-3 实测可用）/ L1 (16+8+8)KB×2 = 64KB
  ≤512KB / 激活核 UB ≤192KB（CONST-capacity-910B2C）；
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 13.3–18.9 ms**（B/A 访存主导）。
  常数未覆盖量（NPU L2 对 A/B 的实际复用率、`T.serial` K 循环 MTE-Cube 交叠效率）显式标注
  「未实测假设 + 估算依据」；实测由 Stage 4 校核。注：常数绑定 Ascend910B2C，工具链变更须按 `kb_stale_check.py` 重验。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1 算法（Cube 核读表 + 双累加器 + packed `gate_up`；Vector 激活核）。
> 每项四要素：原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：
> `examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/verify_equiv.py`。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| A1 | 动态 extent 精确裁剪 + 动态 K 归约范围（**Cube 核采纳**） | A 无 pad 靠 TMA OOB 零填越界行 + epilogue 掩码；源直接拒绝 K 非整除；谓词零填则按静态 `block_k` 归约整块 | 只取/只写有效区 `tile_m=min(block_m,s_e-row)`, `tile_n=min(block_n,ffn-n0)`, `tile_k=min(block_k,K-k·bk)`，并以 `T.gemm(size=[tile_m,tile_k,tile_n])` 令归约维=tile_k | 容差内等价：越界项恒为 0（0·x=0）当且仅当归约范围=有效 tile_k；机器验证 2 dtype、143 例 0 违反（EQUIV_PASS）；stale-tail guard 确认缺 `size=` 即发散（≈2.8e3） | 消 K/N 零填充逐元素算；尾块只搬有效行/列；`size` 免整块归约；推广到非整除输入（超集） |
| A2 | NT 形式（不物化 B^T）（**Cube 核采纳**） | 显式转置/重排 B 后 `b_transpose=False` | 直接 `T.gemm(A,B,C,b_transpose=True)` 用 B[·,K] 行主序 | 数学恒等 $(AB^T)$；机器验证 0 违反（323 例，2 dtype） | 省一次 B 转置/重排（B 是最大张量 15 GB，任何全量重排不可接受） |
| A3 | silu_and_mul 公式与结合序（**激活核采纳**） | 源 `g/(1+exp(-g))`（GPU 用 exp2 降代：`g/(1+exp2(-g·log2e))`）；输出 `silu(gate)·up` | `g/(1+exp(-g))`（direct-exp），输出按**未合并结合序** `(g/d)·v` | `exp` 与 `exp2(−g·log2e)` 是同一实函数（NPU `vexp` 与 `vexp2` 同为单指令，log2e 缩放徒增舍入）；机器验证 0 违反（4096 例 ×2 dtype，含 NaN/±Inf/±0/次正规/大值） | 消 log2e 缩放（源设计仅因 CUDA `expf` 多 op）；`(g/d)·v` 避免 `|g|,|v|` 同近 dtype max 时分子 `g·v`→inf、`inf/inf`→NaN（assoc guard 实测 fp16 发散 1.95e-3 / bf16 inf_mismatch 164） |
| A4 | gelu_and_mul 公式（**激活核采纳**） | 源 `0.5·g·(1+erf(g/√2))`（精确 erf，源用 `T.erf`） | `0.5·(g + |g|·erf(|g|/√2))`；`erf` 用 odd-symmetry + Abramowitz & Stegun 7.1.26 erfc 有理逼近 + 常量折叠 | NPU `T.verf` 为低阶 Taylor（>1 不可用，见 `examples/gelu_and_mul` DESIGN）；odd-symmetry 是精确恒等式；A&S 逼近绝对误差 ≤1.5e-7；机器验证 0 违反（4096 例 ×2 dtype） | 消符号分支（`erf(-z)=-erf(z)`）；用可用的高精度 erf 构造替代不可用的原生 erf；0.5/√2 折叠 |
| A5 | 双累加器共享 A（**Cube 核采纳**） | 两段 K 循环分别算 gate/up（A tile 重载） | 一次 K 循环内两条 `T.gemm` 共享同一 `a_l1`，分别入 `c_gate/c_up` | 数学恒等：$A B_{gate}^T$ 与 $A B_{up}^T$ 为两个独立矩阵乘，共享 A 不改变结果；机器验证 0 违反（1008 例，2 dtype） | A tile 装载减半（R3：C5 单累加器 A 0.94G vs C1 0.47G）；实现已由 D-3 探针实测（§9.6） |
| A6 | 无进一步公式化简 | — | — | 矩阵乘核心（NT + fp32 分块累加）+ 两个激活公式均无可再降代价的恒等变形；无除法可倒数化（silu 分母 `1+exp(-g)` 含超越函数，非纯常数倒数）、无公共子表达式（gate/up 相互独立） | 结论：A1–A5 即全部采纳项 |

**优化结论**：采纳 5 项（A1–A5）。**融合意图的代价重估**：源 O7（register 融合，中间张量不落 GM）在当前
工具链**不可保留**（CG-2026-0026）；本设计的 C1 额外付出中间张量 GM 往返（fp32，prefill 下 1.07 GB ≈
0.85 ms），相对 B 访存 15.03 GB（≈11.9 ms）约 7%，相对全算子 ≈4% —— 代价以「换取可实现性」为前提（§0.6 R5）。
**优化后公式**（供 §3.1 拆解）：

Cube 核（每逻辑 tile $(e,row,t_m,t_n)$，每 $k$-块）：
$$
c_{gate}[m_0{:}m_0{+}t_m,\,n_0{:}n_0{+}t_n]=\sum_k A[m_0{:}m_0{+}t_m,\,k b_k{:}k b_k{+}t_k]\,
B[e,\,n_0{:}n_0{+}t_n,\,k b_k{:}k b_k{+}t_k]^{T}\ (\text{fp32})
$$
$$
c_{up}[\,\cdot\,]=\sum_k A[\,\cdot,\,k b_k{:}k b_k{+}t_k]\,B[e,\,\text{ffn}{+}n_0{:}\text{ffn}{+}n_0{+}t_n,\,k b_k{:}k b_k{+}t_k]^{T}\ (\text{fp32})
$$
其中 $m_0=o_e+row$（`row` 已乘 block_m）、$n_0=n^{idx}_0\cdot block_n$，$t_m/t_n/t_k$ 为动态有效尺寸。

Vector 激活核（packed `gu`）：$C[m,n]=\mathrm{act}(gu[m,n])\cdot gu[m,\text{ffn}+n]$，$\mathrm{act}$ 按 A3/A4，
fp32 域求值后单次回舍。

**等价性机器验证结果表**（`verify_equiv.py`，fp64 torch CPU 参照；脚本随算子目录留存）：

| 项 | 对照 | 最大度量值 | 违反 / 用例 | 结论 |
|----|------|-----------|-------------|------|
| A1 | 动态 extent + 动态 K 归约 vs 零填全 `block_k`（K=70，含 M/N 尾） | 0.000e+00 | 0 / 143（fp16）/ 143（bf16） | EQUIV_PASS |
| A2 | NT（`b_transpose`）vs 物化 B^T | 0.000e+00 | 0 / 323（两种 dtype） | EQUIV_PASS |
| A3 | silu direct-exp `(g/d)·v` vs torch `F.silu(g)·v` | 0.000e+00 | 0 / 4096（两种 dtype） | EQUIV_PASS |
| A4 | gelu odd-symmetry + A&S erf vs torch `F.gelu(g)·v` | 0.000e+00 | 0 / 4096（两种 dtype） | EQUIV_PASS |
| A5 | 双累加器（共享 A）vs 拼接 B 单 gemm 后列切分 | 0.000e+00 | 0 / 1008（两种 dtype） | EQUIV_PASS |
| （guard 1） | 动态取数但省略 `size=`（全 `block_k` 归约，含陈旧尾值）vs 零填基线 | 2.84e+03（发散） | 143 / 143（两种 dtype） | GUARD_DIVERGES（确认 A1 前提不可省） |
| （guard 2） | 合并结合序 `(g·v)/d`（dtype 域）vs 未合并 `(g/d)·v` | fp16 1.95e-03 / bf16 inf_mismatch 164 | — | GUARD_DIVERGES（确认 A3 结合序在溢出角点必要） |

无失败项（所有**采纳项**均为 EQUIV_PASS；guard 行为前提检查，按设计发散）。guard 1 的意义：刻意模拟
「动态取数但漏传 `size`」，与基线产生 2.8e3 量级发散，**证明 A1 的等价性确实依赖
`T.gemm(size=[tile_m,tile_k,tile_n])` 限定归约范围**。guard 2 的意义：合并分子在 dtype 域溢出，证明
un-merged 结合序是必要选择。fp16/bf16 舍入影响评估：A1/A2/A5 为精确恒等或「删除恒零项」，不改变实际参与
累加的浮点运算；A3 的 exp2→exp 消除一处常量缩放引发的舍入，A4 的 A&S 逼近绝对误差 ≤1.5e-7（远小于
bf16 ulp），两者均不改变 dtype 层结论。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点**本算子两 kernel** 实现方案全部循环与标量计算点（对象 = §1.6.0 选定、§0.6 重设计后的 C1 算法）。
**说明**：调度阶段（前缀扫描 / 查找 / 向量化）已按 §0.6 R1 外移到前导 kernel `_tile_scheduler_kernel`，
其循环/标量点由胞弟设计承载；本算子不含索引段向量工作。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | K 维累加循环（Cube 核） | `T.serial(T.ceildiv(K,block_k))` | Cube `T.gemm`（矩阵单元，非标量） | ✅ | — |
| 2 | A/B 分块取数（Cube 核） | 逐元素 | `T.copy` 切片（向量搬运 + ND2NZ） | ✅ | — |
| 3 | **融合激活 epilogue `act(C_gate[i,j])·C_up[i,j]`（源 register `T.Parallel`）** | `T.Parallel(block_m,block_n)` 逐元素（源 L448-450） | 独立 Vector 激活核的 v-prefix 链（**整体改由 Vector 单元承担，非标量循环**）：`vcast→vmul(-1)→vexp→vadd(+1)→vdiv→vmul→vcast`（silu）/ A&S erfc 链（gelu） | ✅ | — |
| 4 | 写回掩码（Cube 核） | 逐元素 `if i<arows` | 动态 extent `store_fixpipe([tile_m,tile_n])` —— **整体消除该循环** | ✅ | — |
| 5 | 双 `store_fixpipe` 的目标列偏移 | 标量列偏移（gate→`n_start`，up→`ffn+n_start`） | 无 | ❌ | block 级索引（每 tile 常数偏移），非逐元素热点，无逐元素等价 API |
| 6 | tile 坐标与基址 `bx/by/m_start/n_start` | 标量乘加（读调度表导出） | 无 | ❌ | block 级索引 / 任务映射计算（每 task 少量标量），输入来自 GM 调度表（运行时值） |
| 7 | 调度表标量读 `total_tiles[0]` / `tile_expert_ids[bx]` / `tile_row_offsets[bx]` / `true_offsets[e]` / `true_sizes[e]` | GM 标量读 | 无 | ❌ | 每 task 数次标量 GM 读（O(1) 寻址，正是调度表的设计目的）；非逐元素热点 |
| 8 | 动态 extent `tile_m/tile_n/tile_k` | 标量 `T.min` | 无（标量归约） | ❌ | 每 task 3 次 `T.min` 的 block 级标量，用于设定搬运/写回/归约 extent，非逐元素热点 |
| 9 | 死 tile 守卫 `bx < total_tiles[0]` | 标量比较 | 无 | ❌ | block 级单比较守卫（每 task 一次），非逐元素热点 |
| 10 | 激活核的行/列任务映射 `mb/ck/m0/n0` | 标量整除/取模 | 无 | ❌ | 每 task 少量标量任务映射（persistent grid-stride），非逐元素热点 |

**向量化结论**：本算子 = **Cube 主导 + Vector 激活**。逐元素计算已全部交由合适的单元：源 register
逐元素 epilogue（#3）改由 **Vector 激活核的 v-prefix 链**承担（比标量链更优），写回掩码（#4）整体消除
（动态 extent）。数据搬运与矩阵计算由 Cube 单元（`T.copy`/`T.gemm`/`store_fixpipe`）承担（#1/#2）。仅保留
4 处 block 级标量点（#5–#9：目标列偏移 / tile 坐标 / 调度表标量读 / 动态 extent / 死 tile 守卫）与激活核
2 处任务映射（#10），理由类别：**block 级索引计算 + O(1) 运行时标量读（数据来自 GM 表）**；均非逐元素热点
（每核 O(num_local_tasks) 次标量，相对 B 访存（15 GB 量级）可忽略）。**无可向量化而未向量化的逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子 I/O layout 为契约：A `(numel,K)`、B `(E,2·ffn,K)`、C `(numel,ffn)` 行主序；核内布局是设计变量。
> Cube 路径布局按 cube-skill 决策；Vector 激活核的向量化轴按 elementwise 类决策。

**轴质量评分**：
- Cube 核：无逐元素向量轴（索引已外移），布局为 NZ 分形（搬运隐式转换）。
- 激活核：输入 packed `gate_up (numel, 2·ffn)`，输出 `(numel, ffn)`；唯一有意义的向量化轴 = 最内连续轴
  （第 1 维，即 ffn/2ffn 方向）。fp16/bf16 向量宽度 ×8、fp32 ×4；block_n 取 256/1024 的整数倍以整除宽度。
  行方向（M）广播无（纯逐元素，无广播轴），按行块切分。

**候选矩阵——Cube 路径（分形布局 + epilogue 归属）**：

| # | 布局方案 | 向量化轴 / 关键决策 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| L1 | A/B GM→L1 走 `T.copy` 切片（Expert 自动 ND2NZ），`T.gemm(b_transpose=True, size=[tm,tk,tn])` 直接消费 B[N,K] | 无（Cube 核无 Vector 轴） | 无（ND2NZ 在搬运内完成） | 无额外 GM 流量；NZ 由后端处理 | ✅ **主选** |
| L2 | host 侧把 B 转置物化为 [K,·] 再 `b_transpose=False` | — | host permute | B 为 15 GB，全量 host 转置 = 百 µs~ms 级且禁改输入（ascend-constraints §4）→ 净亏 | ❌ 弃选 |
| L3 | 核内 `T.transpose` 显式把 B tile 转成 [K,N] 再 gemm | — | 核内转置链 | 多余向量管线开销；NZ 通路已支持 NT（`b_transpose=True`，T.gemm.md §2.1）；另有 TRAP-transpose-epilogue-poison | ❌ 弃选 |
| L4 | epilogue 归属 Vector（L0C→GM 往返再由 Vector 逐元素处理同一个 kernel） | — | L0C→GM→UB | **不可实现**：CV 同 block 交接（CG-2026-0026）；CONST-store-fixpipe-gm-only（L0C 跨引擎传输强制 GM 往返） | ❌ 弃选 |
| L5 | epilogue 归属 Cube（双 `store_fixpipe` 写 packed `gate_up`，动态 `[tm,tn]`） | — | 无（fixpipe 直出 GM；本设计不跨 dtype 量化，保持 fp32） | 单次搬运、尾块精确裁剪、保持 fp32 精度（源 gate/up 停在 fp32） | ✅ **主选** |
| L6 | 核内调度内联（Vector 索引段 + 同 block 交接给 Cube） | — | — | — | ❌ 弃选（CG-2026-0026 / CG-2026-0010；§0.6 R1） |

**候选矩阵——激活核（elementwise 类：连续轴选择 + 分块宽度）**：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| V1 | I/O 原生 packed 布局，最内连续轴（第 1 维），packed 段门控 `[n0,n0+bn)` 与 `[ffn+n0,…)` 段直取 | 第 1 维（ffn） | 无 | 与已验证 silu/gelu 先例同形态；fp32 输入 | ✅ **主选** |
| V2 | host 把 packed 拆成两个独立 gate/up 张量再激活 | 第 1 维 | host split | 需额外存储 + 破坏 packed 契约；无收益 | ❌ 弃选 |
| V3 | 激活核输入 quantization 到 bf16（配 C4 中间 bf16） | 第 1 维 | 无 | 省中间带宽，但多一次前置舍入（§1.6.1） | ❌ 弃选（精确性优先） |

**布局决策结论**：
- **Cube 核**：选定 **L1 + L5** —— 核内 NZ 分形（A/B 经 GM→L1 隐式转换，`T.gemm` NT 直接消费 B[N,K]），
  **epilogue 归属 Cube 侧双 `store_fixpipe`**（动态 extent 裁剪，写 packed `gate_up` **fp32**，不跨 dtype 量化）。
  本核 Expert Cube-only，无 Vector scope、无跨引擎交接（L4/L6 因 CG-2026-0026 弃选）。
- **激活核**：选定 **V1** —— packed 原生布局 + 最内连续轴，复用已验证 packed silu/gelu 链。
- **中间张量 dtype 决策（L5 关键）**：选 **fp32**（而非 C4 的 bf16）——源 gate/up 停在 fp32、`act` 在 fp32 域
  求值、单次回舍；fp32 中间使激活核的输入与源一致（代价：中间访存 1.07 GB ≈ 0.85 ms，见 §1.6.0 roofline）。
  **备选（Stage 4 A/B）**：若 Stage 4 实测访存为瓶颈且精度余量充足，可切 C4（bf16 中间，`store_fixpipe`
  f32→bf16 量化）——该备选结构与主选同形，仅改 `gate_up` dtype 与激活核输入 dtype。
- repack 路径：无（NZ 转换在搬运内完成；packed 段门控为原生切片）。
- 该结论已同步 §3.3 伪代码与 §6 循环结构（Cube 核循环只有核内 `T.serial` 任务循环 + Cube `T.gemm` 的 K 维
  循环；激活核为 persistent grid-stride 逐元素链，无 Vector 索引循环）。L2/L3/L4 与 V2/V3 的弃选量化理由见
  候选矩阵（负向论断附 T.gemm.md §2.1、ascend-constraints §4、CONST-store-fixpipe-gm-only、CG-2026-0026）。
- **实验裁决模式**：主选（fp32 中间）与备选（bf16 中间 C4）的判定裕度依赖「中间访存 vs 精度」的未实证
  权衡，但两者**结构同形、仅 dtype 不同**，不触发三件套（备选设计即主选结构 + dtype 替换，已在本节写明）；
  Stage 4 按代表 shape 实测主选 vs 备选，按 §1.6.0 roofline 的 <5% 判定阈值决定是否翻转（§9.4 R-4）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式：混合（hybrid）—— Cube GEMM 核用 Expert，Vector 激活核用 Developer。**

| kernel | 模式 | 形态 |
|--------|------|------|
| kernel G（分组 GEMM + packed 落 GM） | **Expert** | `T.Kernel(nums_kernel, is_npu=True)` + 单 `T.Scope("Cube")` + `T.alloc_L1`×3 + `T.alloc_L0C`×2 + `T.copy` + `T.gemm(b_transpose=True,size=)` + 双 `T.store_fixpipe`；**无 Vector scope、无跨引擎同 block 交接** |
| kernel V（packed 激活） | **Developer** | `T.Kernel(num_kernels, is_npu=True)` + `T.serial` persistent + `T.copy` / v-prefix 链 / `T.copy`（纯 Vector，无 gemm，无手动同步） |

用户/迁移默认 `programming_mode=developer`，但**本算子的 GEMM 段经工具链实证无法用 Developer**
（persistent + gemm + 整型索引段在 lowering 期 segfault，CG-2026-0010 第二证），按「实测为仲裁依据」
把 GEMM 段切为 Expert 并在此披露；激活段无 gemm，Developer 可用（与 `examples/silu_and_mul`、
`examples/gelu_and_mul` 先例一致）。

### 2.2 选型理由

- **GEMM 段必须 Expert 单 Cube scope**：(i) Developer 单 Scope（persistent + `T.gemm` + 整型索引段）
  lowering 期 segfault（CG-2026-0010，本仓第二证）；(ii) 本核只需「读 GM 表 + `T.gemm` + `store_fixpipe`」，
  单一 Cube scope 无跨引擎交接，从而**完全规避** CG-2026-0026（`sync_block_*` 属性名不匹配，任何跨引擎同
  block 交接不可编译）；胞兄 Stage 3 正对照实测该形态 max_diff 9.537e-07。
- **激活段用 Developer**：纯 Vector、无 gemm、无持久化混合，`examples/silu_and_mul` /
  `examples/gelu_and_mul` 的 `_make_fused_gated_explicit` 已在 Developer 下验证；且 Developer 无手动同步
  负担（自动同步足够）。用 Expert 反而需手动 `T.rs` 且无收益。
- **零 CPU 同步**：`total_tiles` 与调度表全程留在设备侧（GM），三 kernel 同 stream 顺序 launch，
  无 `.item()`、无 host 回读。

### 2.3 模式影响

| 维度 | kernel G（Expert 单 Cube scope） | kernel V（Developer 纯 Vector） |
|------|----------------------------------|----------------------------------|
| 内存分配 | `T.alloc_L1`（a/bg/bu）、`T.alloc_L0C`（c_gate/c_up）；**无 UB** | `T.alloc_ub`（gate/up 段 + fp32 中转）；**无 L1/L0C** |
| 取值 | `T.copy` 切片（GM→L1 ND2NZ） | `T.copy` 段切片（GM→UB） |
| 写回 | 双 `T.store_fixpipe`（动态 extent，`enable_nz2nd=True`） | `T.copy`（UB→GM） |
| 调度输入 | GM 表 `tile_expert_ids`/`tile_row_offsets`/`total_tiles` + `true_sizes`/`true_offsets` | 无（逐元素，无调度表） |
| 同步 | 无跨引擎同步；仅 Cube scope 内 `T.rs("PIPE_FIX")` | Developer 自动同步（无显式点） |
| 跨引擎交接 | **无**（规避 CG-2026-0026） | **无**（独立 kernel，边界隐式顺序） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的优化后公式。本算子无核内 Vector 索引段（已外移）；矩阵单元用 `T.gemm`，搬运用 `T.copy`，
> GEMM 写回用 `T.store_fixpipe`，激活用 v-prefix 链。

**kernel G（Cube 分组 GEMM）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `bx = tile_id // num_pid_n`，`by = tile_id % num_pid_n` | tile 坐标（核内 `T.serial` 任务） |
| 2 | 读 `total_tiles[0]`，`bx < total_tiles[0]` 判定存活 | 死 tile 守卫（GM 标量读，单比较） |
| 3 | `e = tile_expert_ids[bx]`，`row = tile_row_offsets[bx]` | O(1) 寻址（GM 标量读） |
| 4 | `m_0 = true_offsets[e] + row`，`n_0 = by·block_n` | tile 基址 |
| 5 | `t_m=min(block_m,true_sizes[e]−row)`, `t_n=min(block_n,ffn−n_0)`, `t_k=min(block_k,K−k·bk)` | 动态 extent（A1） |
| 6 | `c_gate = Σ_k A[m0:m0+tm, kb:kb+tk]·B[e, n0:n0+tn, kb:kb+tk]^T` | NT GEMM + fp32 累加（A2） |
| 7 | `c_up = Σ_k A[m0:m0+tm, kb:kb+tk]·B[e, ffn+n0:ffn+n0+tn, kb:kb+tk]^T` | 共享 A 的第二 GEMM（A5） |
| 8 | `GU[m0:m0+tm, n0:n0+tn] ← c_gate`；`GU[m0:m0+tm, ffn+n0:ffn+n0+tn] ← c_up` | 双 store_fixpipe（packed） |

**kernel V（Vector 激活）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 9 | `C[m,n] = act(GU[m,n]) · GU[m,ffn+n]` | packed 段门控 + act 链 + 单次回舍（A3/A4） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | tile 坐标 | 标量整除/取模 | `bx/by` 标量 | Expert |
| 2 | 死 tile 守卫 | 标量比较（GM 标量读 `total_tiles[0]`） | `if bx < total_tiles[0]` | Expert |
| 3 | 调度表寻址 | GM 标量读（运行时下标） | `tile_expert_ids[bx]`、`tile_row_offsets[bx]` | Expert |
| 4/5 | 基址/动态 extent | 标量乘加 + `T.min` | `T.min(block_m, true_sizes[e]−row)` 等 | Expert |
| 6/7 | 双 NT 矩阵乘（共享 A） | `T.gemm(a_l1, bg_l1, c_gate, size=[t_m,t_k,t_n], b_transpose=True, initC=(k==0))`；同型第二条入 `c_up` | `size=[m,k,n]` 限归约范围；`b_transpose=True` | Expert |
| 6/7 | K 维循环 | `T.serial(T.ceildiv(K, block_k))`（或 Stage 4 实测 `T.Pipelined(num_stages)`） | 分块累加 | Expert |
| 6/7 | 取数 | `T.copy` 切片（GM→L1 ND2NZ，动态 extent） | 三路 a/bg/bu | Expert |
| 8 | 写回 | `T.store_fixpipe(c_gate, GU[m0,n0], size=[t_m,t_n], enable_nz2nd=True)`；`store_fixpipe(c_up, GU[m0,ffn+n0], ...)`，各包 `T.rs("PIPE_FIX")` | 动态 extents；目标列区间不相交 | Expert |
| 9 | packed 激活 | `T.copy`（段取）→ `T.vcast/vmul/vexp/vadd/vdiv/vmul/vcast`（silu）或 A&S erf 链（gelu）→ `T.copy`（写回） | block 内 v-prefix 链 | Developer |

> 说明：激活段用 v-prefix 浮点向量 API（`vexp/vadd/vmul/vdiv/vcast`，gelu 另用 `vabs/vrec/verf` 家族替代）；
> GEMM 段不涉及 v-prefix（`T.gemm`/`T.copy`/`T.store_fixpipe` 承担）。所有 v-prefix API 有
> `docs/Tilelang.language/数学操作/`、`数据类型转换操作/T.vcast.md` 与既有先例佐证。

### 3.3 计算伪代码

> `T.Tensor` 第二个位置参数直接传 dtype。GEMM 伪代码与 D-3 已验证探针
> `history_version/design_probe_dual_acc.py` 同形态。

```python
# ── kernel G：Expert 单 Cube scope（分组 GEMM + packed gate_up）──
# 编译期常量：numel, E, ffn, K, block_m=128, block_n=128, block_k=64
#   max_tiles = numel // block_m + E          # 与调度表共用的编译期常量
#   num_pid_n = ceildiv(ffn, block_n)
#   nums_kernel = min(max_tiles * num_pid_n, P)      # P = 实查物理核数 20
#   num_local_tasks = ceildiv(max_tiles * num_pid_n, nums_kernel)

@T.prim_func
def gemm_main(
    A: T.Tensor((numel, K), dtype),
    B: T.Tensor((E, 2 * ffn, K), dtype),
    GU: T.Tensor((numel, 2 * ffn), "float32"),           # out_idx
    tile_expert_ids: T.Tensor((max_tiles,), "int32"),
    tile_row_offsets: T.Tensor((max_tiles,), "int32"),
    true_offsets: T.Tensor((E,), "int32"),
    true_sizes: T.Tensor((E,), "int32"),
    total_tiles: T.Tensor((1,), "int32"),
):
    with T.Kernel(nums_kernel, is_npu=True) as (core_id, subid):
        with T.Scope("Cube"):
            a_l1  = T.alloc_L1((block_m, block_k), dtype)
            bg_l1 = T.alloc_L1((block_n, block_k), dtype)
            bu_l1 = T.alloc_L1((block_n, block_k), dtype)
            c_gate = T.alloc_L0C((block_m, block_n), "float32")
            c_up   = T.alloc_L0C((block_m, block_n), "float32")
            for t in T.serial(num_local_tasks):
                tile_id = core_id * num_local_tasks + t
                bx = tile_id // num_pid_n
                by = tile_id % num_pid_n
                # 死 tile 守卫：单比较，位于 T.serial 体内、K 循环之外（规避 CG-2026-0001）
                if bx < total_tiles[0]:
                    e = tile_expert_ids[bx]
                    row = tile_row_offsets[bx]
                    n_start = by * block_n
                    m_start = true_offsets[e] + row
                    t_m = T.min(block_m, true_sizes[e] - row)
                    t_n = T.min(block_n, ffn - n_start)
                    for k in T.serial(T.ceildiv(K, block_k)):
                        t_k = T.min(block_k, K - k * block_k)
                        T.copy(A[m_start:m_start + t_m, k*block_k:k*block_k + t_k],
                               a_l1[0:t_m, 0:t_k])
                        T.copy(B[e, n_start:n_start + t_n, k*block_k:k*block_k + t_k],
                               bg_l1[0:t_n, 0:t_k])
                        T.copy(B[e, ffn+n_start:ffn+n_start + t_n, k*block_k:k*block_k + t_k],
                               bu_l1[0:t_n, 0:t_k])
                        # size=[t_m, t_k, t_n] 限定归约范围 = 有效 t_k（正确性必需，§1.6.1 A1）
                        T.gemm(a_l1, bg_l1, c_gate,
                               size=[t_m, t_k, t_n], b_transpose=True, initC=(k == 0))
                        T.gemm(a_l1, bu_l1, c_up,
                               size=[t_m, t_k, t_n], b_transpose=True, initC=(k == 0))
                    with T.rs("PIPE_FIX"):
                        T.store_fixpipe(c_gate, GU[m_start, n_start],
                                        size=[t_m, t_n], enable_nz2nd=True)
                    with T.rs("PIPE_FIX"):
                        T.store_fixpipe(c_up, GU[m_start, ffn + n_start],
                                        size=[t_m, t_n], enable_nz2nd=True)

# ── kernel V：Developer packed 激活（复用已验证 silu/gelu 先例）──
#   block_m=2, block_n=256（UB 预算内；Stage 4 可调宽）
#   chunks = ceildiv(ffn, block_n); m_blocks = ceildiv(numel, block_m)
#   num_kernels = min(m_blocks * chunks, 2*P)        # 纯 Vector 物理核数 40
#   num_local = ceildiv(m_blocks * chunks, num_kernels)

@T.prim_func
def act_main(
    GU: T.Tensor((numel, 2 * ffn), "float32"),
    C:  T.Tensor((numel, ffn), dtype),                    # out_idx
):
    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
        g_ub = T.alloc_ub((block_m, block_n), "float32")
        v_ub = T.alloc_ub((block_m, block_n), "float32")
        t32  = T.alloc_ub((block_m, block_n), "float32")
        d32  = T.alloc_ub((block_m, block_n), "float32")
        p32  = T.alloc_ub((block_m, block_n), "float32")
        y_ub = T.alloc_ub((block_m, block_n), dtype)
        for i in T.serial(num_local):
            task = i * num_kernels + cid
            if task < m_blocks * chunks:
                mb = task // chunks; ck = task % chunks
                m0 = mb * block_m;  n0 = ck * block_n
                m_t = T.min(block_m, numel - m0)
                n_t = T.min(block_n, ffn - n0)
                # gate 段 [n0,n0+nt)、up 段 [ffn+n0, ffn+n0+nt)（packed，无交织/无反转）
                T.copy(GU[m0:m0 + m_t, n0:n0 + n_t], g_ub[0:m_t, 0:n_t])
                T.copy(GU[m0:m0 + m_t, ffn+n0:ffn+n0 + n_t], v_ub[0:m_t, 0:n_t])
                # silu 链（§1.6.1 A3）：(g/d)*v，未合并结合序
                T.vmul(g_ub, -1.0, t32)       # t = -g
                T.vexp(t32, t32)              # t = exp(-g)
                T.vadd(t32, 1.0, d32)         # d = 1 + exp(-g)
                T.vdiv(g_ub, d32, p32)        # p = silu(g)
                T.vmul(p32, v_ub, p32)        # p = silu(g) * v
                T.vcast(p32, y_ub, round_mode="rint")
                T.copy(y_ub[0:m_t, 0:n_t], C[m0:m0 + m_t, n0:n0 + n_t])
                # gelu 分支同构：odd-symmetry + A&S erfc 链（见 examples/gelu_and_mul）
```

**注 1（K 尾正确性前提）**：`T.gemm(..., size=[t_m, t_k, t_n])` 的 `size` 不是可选优化而是**正确性必需**
——`a_l1/bg_l1/bu_l1` 的 `[t_k, block_k)` 区未被 `T.copy` 写入，若无 `size` 则按缓冲静态 `block_k` 归约、
把陈旧值计入累加（K 非整除用例必错）。§1.6.1 A1 的 stale-tail guard 已确认缺失会发散（≈2.8e3）。
**注 2（为何两核而非单核）**：Cube scope 只做「读 GM 表 + GEMM + 写回」，激活出到独立 Vector 核；
单核内 CV 交接被 CG-2026-0026 阻断（§0.6 R5）。**注 3（`initC=(k==0)`）**：首块覆盖初始化 L0C，后续累加；
等价于源 `clear_accum=(k==0)`。**注 4（packed 布局）**：`GU` 的 gate 列 `[n_start,n_start+t_n)` 与 up 列
`[ffn+n_start,…)` 不相交，双 `store_fixpipe` 无写冲突；与源 B 的 gate||up 布局一致。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.gemm(A,B,C,size=[m,k,n],b_transpose=True,initC=False)` | docs/Tilelang.language/线性代数操作/T.gemm.md §1/§2.1；examples/gemm/matmul_dynamic_shape.py；examples/mixcv/mixcv_mixkernel.py；**D-3 探针三层配置实测**（§9.6） | 已验证 |
| `T.alloc_L0C` ×2（双累加器） | D-3 探针 `history_version/design_probe_dual_acc.py`（三配置编译通过 + 数值正确） | 已验证（本任务新增实证） |
| `T.store_fixpipe`（双写 disjoint 列 + 动态 extent） | docs/Tilelang.language/内存操作/T.store_fixpipe.md；D-3 探针 | 已验证 |
| `T.Kernel(n, is_npu=True)` + 核内 `T.serial` | examples/deepseek_v32/fp8_lighting_indexer.py；D-3 探针 | 已验证 |
| `T.copy` 切片 + 动态 extents（尾块）；GM→L1 自动 ND2NZ | docs/Tilelang.language/内存操作/T.copy.md §2.2.2/§2.3；D-3 探针 | 已验证 |
| `T.min` 标量 | docs/Tilelang.language/比较操作/T.min.md；D-3 探针 | 已验证 |
| `T.alloc_L1` | examples/mixcv/mixcv_mixkernel.py；D-3 探针 | 已验证（Expert 路径） |
| `T.vcast/vmul/vexp/vadd/vdiv`（激活链） | docs/Tilelang.language/数学操作/、数据类型转换操作/T.vcast.md；examples/silu_and_mul、examples/gelu_and_mul（compile+run 验证） | 已验证 |
| `T.verf` | docs/Tilelang.language/数学操作/T.verf.md（**低阶 Taylor，>1 不可用**） | **不采用**（改用 A&S erf，见 §1.6.1 A4） |
| ~~`T.sync_block_set/wait`~~ | docs/Tilelang.language/同步管道操作/T.sync_block_set.md | **不可用**（CG-2026-0026）；本设计不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 涉及（源为展平 2D grid 的一维化） | 一维 `T.Kernel`；tile 坐标由核内 `bx/by` 除法/取模得到 |
| GEMM M/N 需为 block 整数倍 | 涉及（numel/每专家 s_e 一般不被 block_m 整除；ffn 可 < block_n） | 动态 extent 精确裁剪（§0.6 R4）；host 不做任何 padding |
| L0C 容量（128KB） | 涉及 | 双 [128,128]fp32 = 131072B = 128KB（**贴界**，D-3 实测可用）；备选 block_n=64（64KB） |
| 物理核数限制 | 涉及 | Cube `nums_kernel=min(logical,20)`；Vector `min(logical,40)`；核内 `T.serial`（§5） |
| host 侧禁止改动输入 | 涉及（源无 pad，但仍不 host 改动） | host 只做 shape/metadata 视图；`true_sizes` 等原样入核 |
| 三维 Kernel / GPU API | 源用 TMA/swizzle/wgmma（CUDA） | 换 NPU 等价：`T.copy` 切片 + NZ/动态 extent + `T.gemm`（§0.5） |
| 跨引擎同 block 交接 | 源式融合 epilogue 需要 | **不使用**（拆两核）；依据 CG-2026-0026（§0.6 R5） |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| Kernel 数 | 1（3WG 融合） | **3**（前导调度 + GEMM 核 + 激活核） | §0.6 R1/R5 |
| 并行结构 | producer + 2 math WG（warp-specialized） | 单 Cube scope 顺序发射 + 固定核数核内串行 | §0.6 R2/R3 |
| tile 调度 | 核内 thread-0 前缀扫描 + 每 tile 二分 | 外部前导 kernel 产 GM 表，GEMM 核读表 | §0.6 R1 |
| 累加器 | register fragment（CUDA） | 双 L0C（Cube） | §0.6 R2 |
| 取数 | TMA（OOB 零填）+ `T.Pipelined` | `T.copy` 切片 + 动态 extent | §0.6 R4 |
| 融合 epilogue | register 内 `act(gate)*up`，中间不落 GM | Cube 落 packed `gate_up`(fp32) → Vector 核激活 | §0.6 R5 |
| swizzle | `make_swizzled_layout` | 删除（NZ 由搬运转换） | §0.5 O4 |
| 对齐契约 | `forward` 拒绝 K 非整除 / ffn 非整除 | 动态 extent 推广（超集） | §0.6 R4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/_moe_grouped_gemm_kernel.py` | 直接（归一化 GEMM 底座） | Expert 单 Cube scope + 外部调度表 + 动态 extent + `T.gemm(size=)` + `store_fixpipe`（Stage 3 max_diff 9.537e-07） |
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py` | 直接（CG-2026-0026 复现 + 正对照） | `positive_control()` 的 Expert Cube GEMM 结构 |
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | 高（调度表产出者） | 三张 GM 表契约（int32、行偏移单位、死 tile 哨兵）、两 kernel GM 交接 |
| `examples/silu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py` | 直接（激活核底座） | Developer packed `(M,2N)` silu 链 + 结合序 `(g/d)*v` + persistent Vector 分核 |
| `examples/gelu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py` | 直接（gelu 激活底座） | odd-symmetry + A&S erf 构造 + 常量折叠（`T.verf` 不可用的替代） |
| `examples/mixcv/mixcv_mixkernel.py` | 中（Expert 双 scope 范式） | 动态 extent + `store_fixpipe([tm,tn])`；**其 `sync_block_set/wait` 当前不可编译（CG-2026-0026），本设计不用** |
| `examples/gemm/matmul_dynamic_shape.py` | 高 | M/N/K 全动态 extent 的规范写法 + `T.gemm(size=)` + `store_fixpipe(size=)` |
| `testing/autotune/anneal/moe_grouped_gemm_anneal.py` | 中（两 kernel 数据流参照） | 独立 scheduler 写 3 张 GM 表 + GEMM 读表 |
| `examples/deepseek_v32/fp8_lighting_indexer.py` | 中 | 固定核数 + 核内 `T.serial(ceildiv)` 多任务 + 单比较守卫 + `T.gemm(initC=True,b_transpose=True,size=)` |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (numel, K) | float16 / bfloat16 | 紧致激活，行连续 |
| B | (E, 2·ffn, K) | 同 A | 专家权重（前 ffn 行 gate、后 ffn 行 up；NT 恒等使用） |
| true_sizes | (E,) | int32 | 每专家行数 $s_e$ |
| true_offsets | (E,) | int32 | 每专家在 A 中的紧致起始行 $o_e$ |
| tile_expert_ids | (max_tiles,) | int32 | **调度表**（GM，前导 kernel 产出）：每 M-tile 专家号（死 tile = -1） |
| tile_row_offsets | (max_tiles,) | int32 | **调度表**：M-tile 在专家内的行偏移（已乘 block_m） |
| total_tiles | (1,) | int32 | **调度表**：有效 M-tile 总数（设备侧，不回落 CPU） |

> **对接契约（本核消费 / 胞弟产出）**：① `max_tiles = numel // block_m + E` 与 `block_m`（**本算子 128**）
> 为两函数共享的编译期常量，前导 kernel 须以 block_m=128 实例化；② 表 dtype 全 int32；
> ③ `tile_row_offsets` 单位是**行**（已乘 block_m），不是 tile 序号；④ 死 tile 哨兵 `(-1, 0)`；
> ⑤ `total_tiles ≤ max_tiles`；⑥ 三 kernel 同 stream 顺序 launch（前导 → GEMM → 激活），无 in-block 同步。
> **注意**：胞弟现有非融合变体用 block_m=64；本算子因双累加器与 B 复用（§1.6.0 R3）需 block_m=128，
> 前导 kernel 必须按本契约重新实例化（公式不变，仅常量不同）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (numel, ffn) | 同 A | 行连续，非转置布局 |
| GU（中间，kernel G 输出 / kernel V 输入） | (numel, 2·ffn) | float32 | packed gate_up（gate 列 [0,ffn)、up 列 [ffn,2ffn)） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_l1 | (block_m, block_k) | dtype | L1 | A tile（ND2NZ） |
| bg_l1 / bu_l1 | (block_n, block_k) | dtype | L1 | B gate / up tile（ND2NZ） |
| c_gate / c_up | (block_m, block_n) | float32 | L0C | 双累加器（gate/up） |
| g_ub / v_ub | (block_m, block_n) | float32 | UB（激活核） | packed gate / up 段 |
| t32 / d32 / p32 | (block_m, block_n) | float32 | UB（激活核） | 激活链中转 |
| y_ub | (block_m, block_n) | dtype | UB（激活核） | 回舍结果 / 输出 staging |
| GU | (numel, 2·ffn) | float32 | GM | 中间张量（跨两 kernel；占 GM 非片上预算） |

### 4.4 内存搬运路径

```
[前导 kernel] true_sizes --前缀扫描/查找--> GM[表: tile_expert_ids / tile_row_offsets / total_tiles]
                                                 │ (同 stream 顺序 launch)
                                                 ▼
[kernel G] 标量读 GM 表 + GM 标量(true_offsets/true_sizes) --> (e,row,m_start,n_start,t_m,t_n,t_k)
GM[A]  --T.copy 切片(动态extent,ND2NZ)--> L1[a_l1]  ┐
GM[B]  --T.copy 切片(动态extent,ND2NZ)--> L1[bg_l1] ├─ T.gemm(b_transpose,size=[tm,tk,tn]) --> L0C[c_gate fp32]
GM[B]  --T.copy 切片(动态extent,ND2NZ)--> L1[bu_l1] ┘─ T.gemm(b_transpose,size=[tm,tk,tn]) --> L0C[c_up   fp32]
                                        │ 双 store_fixpipe([tm,tn], nz2nd)
                                        ▼
                                  GM[GU(numel,2ffn) fp32]
                                        │ (同 stream 顺序 launch)
                                        ▼
[kernel V] GM[GU] --T.copy 段切片--> UB[g_ub / v_ub] --v-prefix 链(act)--> UB[y_ub] --T.copy--> GM[C]
```

### 4.5 片上内存预算（block_m=128 / block_n=128 / block_k=64 / fp16）

| 层级 | Buffer | Shape | dtype | 大小 (Bytes) |
|------|--------|-------|-------|-------------|
| L1 | a_l1 | (128,64) | fp16 | 16384 |
| L1 | bg_l1 | (128,64) | fp16 | 16384 |
| L1 | bu_l1 | (128,64) | fp16 | 16384 |
| L1 | （×num_stages=2） | | | 98304 |
| L0C | c_gate | (128,128) | fp32 | 65536 |
| L0C | c_up | (128,128) | fp32 | 65536 |
| L0C | 合计 | | | **131072 = 128KB（贴界）** |
| UB（激活核） | g_ub+v_ub+t32+d32+p32（fp32）+ y_ub（bf16） | (2,256) | 混合 | ≈ (5×4+2)×512 = 11264 |

> **L1 预算复核**：A(16KB)+Bg(16KB)+Bu(16KB)=48KB × num_stages(2) = 96KB ≤ 512KB ✓（CONST-capacity-910B2C）；
> num_stages 可达更多但受 L0C/流水收益约束（Stage 4 扫描）。
> **L0C 复核**：2×128×128×4 = 131072 = 128KB（上界 = 128KB；D-3 探针在（128,128）配置实测可用，§9.6）。
> 若后续 shape 触发 L0C 溢出，回退 block_n=64（L0C 64KB，代价 A 访存略升，§1.6.0 R3）。
> **UB（激活核）**：block_m=2×block_n=256，上表 ≈11.3KB，远低于 192KB；可放大 block_n（如 1024/2048）
> 减少向量 op 数（Stage 4，依据 CONST-vector-launch-overhead）。
> `pass_configs` 关闭 auto-multi-buffer，故不适用 ×1.7 膨胀系数（该系数对应 auto-multi-buffer 开启；
> Expert 显式 alloc ≈ 手工核算 ±12%，见 CONST-capacity-910B2C）。

### 4.6 动态轴定义

无动态轴：`numel / E / ffn / K` 均为编译期常量（JIT 特化）。运行时变化的仅为 `true_sizes` 的数据值
（不改变 shape），由动态 extent 与守卫处理。

### 4.7 JIT 配置

```python
_PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}

@tilelang.jit(out_idx=[2], target="npuir", pass_configs=_PASS_CFG)
def _gemm_gate_up(numel, max_tiles, E, ffn, K, dtype="float16",
                  block_m=128, block_n=128, block_k=64):
    ...

@tilelang.jit(out_idx=[1], target="npuir")          # Developer 激活核（无 L1/L0C 显式 alloc）
def _act_packed(numel, ffn, dtype="float16"):
    ...
```

> `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` 避免 buffer 重排/重作用域破坏显式 L1/L0C
> （CG-2026-0012 / TRAP-UB-dynsubview-dominance）；`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False` 关闭
> auto-multi-buffer（与 Stage 3 正对照 / D-3 探针一致）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**Cube 主导（GEMM 核，Expert 单 Cube scope）+ 逐元素 Vector（激活核，Developer）+ 前导 kernel 的
整数 Vector 索引（胞弟单元）**。判定依据：GEMM 核唯一数值计算为 `T.gemm` NT（Cube）+ `store_fixpipe`
（epilogue 归 Cube）；激活核为纯逐元素 v-prefix 链（无规约、无跨元素依赖）；调度索引段由前导 kernel 独立承担。

### 5.2 Block 划分

```python
# GEMM 核（kernel G）
block_m = 128   # 分形 M 对齐（≥16）；使每专家 1 个 M-tile（B 单遍，§1.6.0 R3）
block_n = 128   # 分形 N 对齐；L0C 双 [128,128]fp32 = 128KB（贴界，D-3 实测可用）
block_k = 64    # 分形 K 对齐（≥32）；K 非整除由动态 extent + `T.gemm(size=)` 归约范围处理
# 激活核（kernel V）
block_m = 2     # 行 tile（沿用已验证 silu/gelu 先例；可调）
block_n = 256   # 列 tile（= 向量宽度整数倍；Stage 4 可放大到 UB 预算允许）
```

### 5.3 约束分析

- **对齐约束**：block_m=128、block_n=128、block_k=64 均 ≥ 分形下限（M/N≥16、K≥32）且 32B 对齐 ✓。
- **L0C 容量**：2×128×128×4 = 131072B = 128KB（上界 = 128KB）✓（D-3 实测可用）；备选 block_n=64 → 64KB。
- **L1 容量**：48KB × num_stages(2) = 96KB ≤ 512KB ✓。
- **UB 容量（激活核）**：≈11.3KB（block_m=2×block_n=256）≤ 192KB ✓。
- **非整除**：numel / 每专家 $s_e$ / ffn / K 均可能不整除 → **动态 extent 精确裁剪**（§5.4）。

### 5.4 注意事项（非整除处理）

- **M/行方向**：每专家独立切 tile，尾块 `t_m = T.min(block_m, s_e − row)`（`row` 来自调度表，已乘 block_m）；
  `store_fixpipe` 只写 `[0:t_m)` 行，**杜绝越界写坏下一专家**（no-pad 关键正确性约束）。
- **N 方向**：`t_n = T.min(block_n, ffn − n_start)`（smoke 用例 ffn=128 = block_n；小 ffn 时尾块）。
- **K 方向**：`t_k = T.min(block_k, K − k·block_k)`；**该动态 extent 必须同时作为
  `T.gemm(..., size=[t_m, t_k, t_n])` 的归约范围**，否则操作数缓冲 `[t_k, block_k)` 的陈旧值会被按静态
  `block_k` 归约计入累加（§3.3 注 1、§1.6.1 A1 guard）。
- **无 host padding**（ascend-constraints §4）。
- **packed 中间张量**：gate 列 `[n_start,n_start+t_n)` 与 up 列 `[ffn+n_start,…)` 不相交，双 store_fixpipe 无冲突。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。
> 本节按两个本算子 kernel 分别给出三要素；前导 kernel 分核由胞弟设计。

**① 逻辑核数计算**

- kernel G（Cube）：`num_logical_G = max_tiles × num_pid_n`，其中 `max_tiles = numel // block_m + E`、
  `num_pid_n = ceildiv(ffn, block_n)`。
- kernel V（Vector）：`num_logical_V = ceildiv(numel, block_m_v) × ceildiv(ffn, block_n_v)`。

**② 物理核数依据（实查）**

- 查询代码与实测：`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` →
  **返回 20**（Ascend910B2C，2026-10-15，本机实查；查询也见 `D-3 探针` 运行日志）。
- kernel G 为 **Cube 算子 → 直接使用返回值 20**；kernel V 为 **纯 Vector → 翻倍 40**（core-split-strategy §1）。

**③ 规模判定与分核方案**

| 代表 shape（numel,E,ffn,K） | max_tiles | num_pid_n | num_logical_G | 判定 |
|------------------------------|-----------|-----------|---------------|------|
| smoke (64,4,128,64) | 4 | 1 | 4 | ≤ 20 → **无需适配** |
| qwen-decode (4096,8,768,2048) | 40 | 6 | 240 | 极大规模 |
| qwen-prefill (32768,8,768,2048) | 264 | 6 | 1584 | 极大规模 |
| decode gate-up (4096,256,2048,7168) | 288 | 16 | 4608 | 极大规模 |
| prefill gate-up (32768,256,2048,7168) | 512 | 16 | 8192 | 极大规模 |

- **统一方案**：`nums_kernel = min(num_logical_G, 20)`（编译期常量）；
  `num_local_tasks = ceildiv(num_logical_G, nums_kernel)`（编译期常量，**循环边界静态**）。
  极大规模下固定启动 20 核、核内 `for t in T.serial(num_local_tasks)` **串行**处理多个逻辑 tile
  （`tile_id = core_id*num_local_tasks + t`，块连续以利专家聚簇），摊薄核启动开销，无法通过调 block 缩减
  核数（block_m/block_n 已受 L0C 双累加器约束，再增会溢出）。
- **smoke**：逻辑核数 4 ≤ 20，`nums_kernel=4`，无超发、无负载不均（其余 16 核不启动）。
- **kernel V**：`num_kernels = min(num_logical_V, 40)`，`num_local = ceildiv(num_logical_V, num_kernels)`（静态）；
  极大规模下固定 40 核 + 核内串行。
- **负载均衡**：块连续分配在均匀分布下每核任务数差 ≤1；偏斜分布下个别核多领少量死 tile（被守卫跳过），影响可忽略。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（G） | 固定核数网格 | `T.Kernel(nums_kernel, is_npu=True)` | 分核策略（§5.5） |
| 核内任务（G） | 串行多 tile | `T.serial(num_local_tasks)`（静态边界） | 极大规模，摊薄核启动开销 |
| 死 tile 守卫（G） | 单比较跳过 | `if bx < total_tiles[0]` | 位于 `T.serial` 体内、K 循环外（规避 CG-2026-0001） |
| K 维（G） | 累加循环 | `T.serial(ceildiv(K,block_k))`（Stage 4 可试 `T.Pipelined(num_stages)`） | 分块累加；A/B tile 双缓冲 |
| 核级（V） | 固定核数网格 | `T.Kernel(num_kernels, is_npu=True)` | 纯 Vector 分核（40） |
| 核内任务（V） | 串行 grid-stride | `T.serial(num_local)`（静态边界） | persistent 摊薄启动 |
| 元素级（V） | 无标量循环 | v-prefix 链 | 逐元素已向量化（§1.6.2） |

### 6.2 循环伪代码

见 §3.3（GEMM 核内 `T.serial(num_local_tasks)` 外层、K 循环内层；激活核内 `T.serial(num_local)` 外层、
v-prefix 链内联）。**本算子无 Vector 索引循环**（调度已外移，§0.6 R1）。

### 6.3 流水线优化

- K 循环 `T.serial`/`T.Pipelined`：A/BG/BU tile 双缓冲，MTE2 搬运与 Cube 计算重叠。D-3 探针以 `T.serial`
  形态跑通；Stage 4 可实测 `T.Pipelined(num_stages=2)` 是否更优（若与 Expert 单 Cube scope 组合不稳定，
  回退 `T.serial` + `T.rs` 显式流水）。
- `num_stages` 由 L1 容量约束：48KB × num_stages ≤ 512KB → 2 起，Stage 4 扫描。
- 三 kernel **顺序执行**（同 stream）；不同 kernel 无重叠（多 kernel 形态的固有代价，相对 B 访存可忽略，
  §1.6.0 roofline）。
- 激活核：`block_n` 增大可减少向量 op 总数（CONST-vector-launch-overhead），Stage 4 扫描。

### 6.4 尾块处理

三方向（M/N/K）统一用动态 extent（`T.min`）——见 §5.4；其中 **K 方向须把 `t_k` 传入 `T.gemm(size=)`**
限定归约范围（否则操作数缓冲的陈旧尾值污染累加，§3.3 注 1、§1.6.1 A1 guard）。不使用 host padding、
谓词零填充。死 tile（`bx ≥ total_tiles`）用单比较 `if` 跳过，**该 `if` 位于 `T.serial` 体内，不得放入
`T.Pipelined` 体内**（CG-2026-0001：Pipelined 体内条件构造崩溃）。

---

## 7. 同步策略

### 7.1 同步模式

**GEMM 核（Expert）**：管线同步（Cube scope 内 `T.rs("PIPE_FIX")`）；**无跨引擎同步**。
**激活核（Developer）**：自动同步（Developer 自动注入；无显式同步点）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 前导 kernel → GEMM 核 → 激活核 | **无显式同步**（同 stream 顺序 launch） | kernel 边界即隐式全局顺序；不需 in-block flag |
| L0C 写回 `GU` 前 | `T.rs("PIPE_FIX")`（Cube scope 内，每条 `store_fixpipe`） | 保证 fixpipe 在正确管线阶段发出（D-3 探针形态） |
| A/BG/BU GM→L1 与 `T.gemm` 之间 | 框架/管线同步（Cube scope 内） | MTE2→Cube 单引擎内数据依赖；非跨引擎 |
| 激活核 GM→UB→计算→GM | Developer 自动同步 | 单引擎内数据依赖 |
| ~~Vector↔Cube `sync_block_set/wait`~~ | **不使用** | CG-2026-0026：`syn_instr_mode` 属性名不匹配，任何跨引擎同 block 交接不可编译 |

> **关键差异（相对源）**：源在单 kernel 内用 named barrier 在 WG 间交接；本设计把激活外移为独立 kernel，
> **彻底移除跨引擎同 block 同步需求**，只保留 Cube scope 内管线同步与 kernel 间自然顺序。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```

> 与 Stage 3 正对照 / D-3 探针一致：关闭 buffer 重排/重作用域（CG-2026-0012 / TRAP-UB-dynsubview-dominance）
> 与 auto-multi-buffer，保证显式 L1/L0C 结构稳定（PL-1.16 Expert 双关闭硬边界）。激活核为 Developer
> 纯 Vector，可单独不加该配置（沿用 silu/gelu 先例）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源测试
> `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad_3wg_fused_act.py` 的 `_ref_fused_act`），
> **不得复刻 §0.6 的 NPU 算法**（保证验证独立性；不引用本设计的 packed/双累加器/v-prefix 链）。

```python
import torch
import torch.nn.functional as F


def golden_fused_act(a, b, true_sizes, true_offsets, activation="silu_and_mul"):
    """per-expert: C = act(A_e @ B[e,:ffn].T) * (A_e @ B[e,ffn:].T), fp32 -> dtype.

    a: (numel, K); b: (E, 2*ffn, K); true_sizes: (E,) int32; true_offsets: (E,) int32
    returns c: (numel, ffn), same dtype as a.
    """
    numel, _ = a.shape
    E, two_ffn, _ = b.shape
    ffn = two_ffn // 2
    if activation == "silu_and_mul":
        act = lambda g, u: F.silu(g) * u
    elif activation == "gelu_and_mul":
        act = lambda g, u: F.gelu(g, approximate="none") * u
    else:
        raise ValueError(activation)
    c = torch.zeros(numel, ffn, dtype=a.dtype, device=a.device)
    sizes = true_sizes.tolist(); offsets = true_offsets.tolist()
    for e in range(E):
        s = sizes[e]
        if s == 0:
            continue
        o = offsets[e]
        gu = a[o:o + s].to(torch.float32) @ b[e].to(torch.float32).T
        c[o:o + s] = act(gu[:, :ffn], gu[:, ffn:]).to(a.dtype)
    return c
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |

（与源测试 `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2)` 一致；fp32 累加 + 激活后单次回落输入
dtype。D-3 探针 Expert Cube 双累加器动态 extent 形态实测 max_diff 2.98e-08，远优于该门限。）

### 8.3 L0 门槛测试计划

> **调度表来源**：生产路径由前导 kernel `_tile_scheduler_kernel`（以 block_m=128 实例化）产出并落 GM；
> 本算子 L0 单独测试用 **host-built 调度表 fixture**（从 `true_sizes` 按源语义展开，等同
> `testing/autotune/anneal/moe_grouped_gemm_anneal.py` 的 `supply_prog` 形态，或 D-3 探针的
> `build_schedule`）——只构造辅助表输入，不改 A/B 真实内容（不违反 ascend-constraints §4）。

| 用例 | shape（numel,E,ffn,K,dtype,act,dist） | 验证点 |
|------|----------------------------------------|-------|
| L0-1 smoke silu bf16 | 64,4,128,64,bf16,silu,uniform | 基本正确性、ffn=block_n、M 尾（s_e=16<128）、读表寻址 |
| L0-2 smoke gelu fp16 | 64,4,128,64,fp16,gelu,uniform | gelu A&S erf 链、fp16 路径 |
| L0-3 full silu bf16 | 256,8,256,128,bf16,silu,uniform | 多专家、num_pid_n=2、双累加器 |
| L0-4 K 非整除 | 128,8,256,70,bf16,silu,uniform | K 尾动态 extent、`T.gemm(size=)` 归约范围 |
| L0-5 偏斜 + 空专家 | 233,5,130,96,bf16,silu,skewed | 死 tile 守卫（-1 哨兵）、奇偶 shape |
| L0-6 代表大 shape | 32768,256,2048,7168,bf16,silu,uniform | 分核极大规模、B 访存路径、L0C 贴界、无死锁 |
| L0-7 qwen gelu prefill | 32768,8,768,2048,bf16,gelu,uniform | 小 E、gelu、大 numel |
| L0-8 三 kernel 协同 | 64,4,128,64,fp16,silu,uniform | 前导 kernel → GEMM → 激活对接契约（表内容/`total_tiles` 全程设备侧/无 CPU 同步） |

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 源式**融合 epilogue 不可保留**（Cube 无逐元素单元 + CG-2026-0026）→ 拆两核，代价 ≈0.85 ms（§0.6 R5）。
- **跨引擎同 block 交接不可用**（CG-2026-0026）→ 调度外移 + 激活外移，两核均无 in-block 跨引擎交接。
- Developer 模式对 persistent + gemm + 索引段不可用（CG-2026-0010）→ GEMM 段用 Expert 单 Cube scope。
- 不支持三维 Kernel → 一维网格 + 核内坐标计算。
- 一维核超发会被串行调度（core-split-strategy §1）→ 固定 20/40 核 + 核内串行。
- L0C 双 [128,128]fp32 = 128KB **贴界** → 若溢出回退 block_n=64（§5.3）。
- **`bishengir-compile` flaky SIGSEGV**（CG-2026-0033）：本设计 GEMM 核（核内 serial 多任务 + 守卫 +
  动态列偏移）落该触发域 → 需 `RLIMIT_STACK` 抬升 + 有界重试（D-3 探针首次编译即遇，重试通过，§9.6）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **跨引擎同 block 同步不可编译**（CG-2026-0026） | 试图在单核内把 Cube GEMM 交 Vector 激活 | `error: custom op 'syn_instr_mode' is unknown`（官方 mixcv 样例复现） | **不使用**：激活外移独立 kernel（§0.6 R5） |
| **Developer + persistent + gemm 崩溃**（CG-2026-0010） | GEMM 段误用 Developer | lowering segfault / 动态 extent gemm 静默错值 | GEMM 段用 Expert 单 Cube scope（§2） |
| K 尾忘传 `T.gemm(size=)`（陈旧尾值污染） | 只做 `T.copy` 子视图写而未限归约范围 | K 非整除用例静默错值 | 必须传 `size=[t_m,t_k,t_n]`；§1.6.1 A1 guard 可检出（发散 ≈2.8e3） |
| Pipelined 体内条件构造崩溃（CG-2026-0001） | 死 tile 守卫误放入 K 循环内 | 编译 SIGSEGV | 守卫 `if` 放 `T.serial` 体内、`T.Pipelined` 之外 |
| **bishengir flaky SIGSEGV**（CG-2026-0033） | 核内 serial 多任务 + 守卫 + 动态列偏移（`num_pid_n>1` 且 `num_local_tasks>1`） | 编译期 err code -11（非错值） | 抬升 `RLIMIT_STACK` 至硬上限 + 有界重建重试（MOE_COMPILE_ATTEMPTS，默认 6） |
| L0C 双累加器溢出 | block_n 过大（如 128×256） | segfault | block_m×block_n ≤ 16384（双 fp32）；128×128 贴界（D-3 实测可用），必要时 block_n=64 |
| 越界写尾块 | 尾块未按动态 extent 裁剪 | 写坏下一专家行 | `store_fixpipe([t_m,t_n])` 严格裁剪（§5.4） |
| packed 列区间交叠 | gate/up 目标列算错 | 相互覆盖 | gate→`n_start`、up→`ffn+n_start`，两区间不相交（§3.3 注 4） |
| 调度表契约不一致 | 前导 kernel block_m 与本核不符 | 寻址错位、错值 | 两函数共享 `max_tiles=numel//block_m+E` 与 block_m=128；行偏移单位=行（§4.1） |
| gelu 用 `T.verf` | 直接移植源 `T.erf` | 低阶 Taylor，`|z|>1` 错值 | 改用 odd-symmetry + A&S erfc 链（§1.6.1 A4） |

### 9.3 特殊场景处理

- **偏斜分布**：大量专家仅 1 行 → tile 严重不满；动态 extent 只搬/只算 1 行，浪费仅在分形对齐最小 tile。
- **极大规模**：核内串行 `num_local_tasks` 可达 ~410（prefill，8192/20）——循环边界静态，启动开销摊薄。
- **极小 shape**（smoke numel=64）：逻辑核数 4 < 物理核数，仅 4 核参与，正确性优先。
- **ffn 小于 block_n**：`num_pid_n=1`，`t_n=min(block_n,ffn)` 裁剪（smoke ffn=128=block_n）。
- **三 kernel 协同**：前导先完成，无 CPU 回读；`total_tiles` 全程设备侧。

### 9.4 待确认项（交 Stage 2/3/4 复核）

- **R-1（融合代价实测）**：中间张量 GM 往返的精确墙钟未实测（§1.6.0 roofline 标「未实测假设 + 估算依据」）；
  Stage 4 实测，若 > B 访存 1% 则评估融合可行性（但当前工具链无融合路径，CG-2026-0026）。
- **R-2（`T.Pipelined` vs `T.serial`）**：D-3 探针以 `T.serial` K 循环跑通；`T.Pipelined(num_stages=2)` 与
  Expert 单 Cube scope 的稳定性待实测，不稳定则回退 `T.serial` + `T.rs`。
- **R-3（L0C 贴界）**：双 [128,128]fp32 = 128KB 为精确上界；代表大 shape 下若溢出则回退 block_n=64
  （D-3 已在 smoke 尺寸实测（128,128）可用；大 shape 的 L0C 占用与 tile 数无关，风险低）。
- **R-4（中间张量 dtype A/B）**：fp32（主选）vs bf16（C4 备选）——Stage 4 按代表 shape 实测访存/精度权衡，
  阈值 <5% 则维持 fp32。
- **R-5（NPU L2 对 A/B 的实际复用率）**：A 的 ×16 重复是否被 L2 吸收、B 是否真单遍 —— 影响是否达到
  §1.6.0 的 13.3 ms 下界；Stage 4 实测。**关键**：B 单遍依赖 block_m=128（每专家 1 M-tile）；若 block_m 被迫减小则 B 翻倍。
- **R-6（调度表对接契约）**：胞弟 `_tile_scheduler_kernel` 需按本契约以 **block_m=128** 实例化（其现有设计用
  64）；两设计的 `max_tiles`/`block_m`/表 dtype/行偏移单位必须一致，集成时核对。
- **R-7（分形对齐）**：动态 extent 的 `t_m` 非 16 整数倍时 `T.copy`/`store_fixpipe` 的 NZ 分形对齐行为 ——
  Stage 3 用 L0-1/L0-5 验证（s_e=16 < block_m=128 已属该情形，D-3 探针 s_e=64 亦非 128）。
- **R-8（激活核 UB 预算与 block_n 调宽）**：block_n 增大可减向量 op 数，但需按 UB 预算与膨胀系数核算；Stage 4 扫描。

### 9.5 能力缺口登记（Gap Rule）

本设计期（Stage 1 算法调研 / API 映射）独立确认两处工具链能力不足，已按 AGENTS.md Gap Rule 登记
（`.agents/evolution/capability-gaps.md`）：

- **CG-2026-0026**（codegen / BishengIR：`hivm.hir.sync_block_set` 属性名 `syn_instr_mode` 与随附
  `bishengir-compile` 期望的 `sync_instr_mode` 不一致）——**阻断一切跨引擎同 block 交接**，迫使本设计
  把「融合 epilogue」拆为两 kernel（§0.6 R5）。本任务独立识别 → `occurrences` 1→2（达 recurring 阈值），
  已附 **capability-fill proposal**：统一属性名、恢复 CV in-block 交接，估计省中间张量 GM 往返 1.07 GB
  （≈0.85 ms）并少一次 launch；受影响算子 = 所有 MixCV Expert。**待用户批准后可提 issue**。
- **CG-2026-0033**（BishengIR：核内 serial 多任务 + 守卫 + 动态列偏移的嵌套 scf 递归走栈 → flaky SIGSEGV）
  —— D-3 探针首次编译即遇（err code -11，重试通过），`occurrences` 2→3；本设计以
  `RLIMIT_STACK` 抬升 + 有界重试规避（§9.2）。
- 本设计已用 D-3 探针实测**绕开**上述缺口的可用形态（Expert 单 Cube 双累加器 + 外部调度表 + 动态 extent
  + 独立 Vector 激活，§9.6）；两缺口保持 open/recurring，待工具链升级后按 `kb_stale_check.py` 重验并考虑复原单核融合。

### 9.6 D-3 设计期探针（关键能力实测）

- **脚本**：`examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/history_version/design_probe_dual_acc.py`
  （≤10 min，1 个探针；只测未知常数/能力，不实现完整算子）。
- **探针问题**：Expert 单 Cube scope 能否保有两个 fp32 L0C 累加器、每 K 步发两条共享 A 的 `T.gemm`、
  并用两条 `store_fixpipe` 写 packed `gate_up[numel,2ffn]` 的不相交列区间？端到端（+ Vector packed 激活核）
  数值是否正确？
- **实测结果**（Ascend910B2C，`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False` + `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False`）：

| 配置（block_m,block_n） | gate_up GEMM max_diff | 端到端（GEMM+激活）max_diff | 结论 |
|--------------------------|----------------------|------------------------------|------|
| (64, 128) | 5.59e-09 | 5.82e-11 | PASS |
| (128, 128) | 4.66e-09 | 2.98e-08 | PASS（L0C 128KB 贴界） |
| (128, 64) | 3.73e-09 | 7.45e-09 | PASS |

- **结论**：**选定底座可实现**——Expert 单 Cube scope + 双 L0C 累加器 + 双 `store_fixpipe` 写 packed
  `gate_up` + 独立 Developer Vector 激活核，三配置全部编译通过且数值正确（远优于 1e-2 门限）。
  **（128,128）配置**（B 单遍最优、L0C 贴界）实测可用，作为主选；备选（128,64）已验证。
- **附带发现**：首次编译命中 `bishengir-compile` flaky SIGSEGV（err code -11），重试通过 → 登记 CG-2026-0033。
- **常数标定**：本探针未引入新的硬件常数（复用 pattern-library/constants.md 既有条目）；工具链戳
  = tilelang dev root（`/mnt/workspace/tilelang-mlir-ascend/build`）+ CANN 9.0.0 / Ascend910B2C / 2026-10-15。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/
├── DESIGN.md                              # 本设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核）
├── verify_equiv.py                        # §1.6.1 等价性机器验证（D-1，A1–A5 全 EQUIV_PASS）
├── history_version/
│   └── design_probe_dual_acc.py           # D-3 探针（双累加器 + packed 落 GM 能力实测）
└── RETROSPECTIVE.md                       # Stage 1 复盘（自进化钩子；Phase 8 追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（§0 迁移 + §1.6 + §5 分核） |
| `verify_equiv.py` | 已完成 | 等价性验证（A1–A5 EQUIV_PASS；A1 stale-tail guard + A3 assoc guard 按设计发散） |
| `history_version/design_probe_dual_acc.py` | 已完成（实测 PASS） | D-3 探针（三配置） |
| `_fused_act_kernel.py` | **待实现**（Stage 3） | kernel G（Expert Cube）+ kernel V（Developer 激活）+ golden |
| 前导 kernel `_tile_scheduler_kernel`（block_m=128 实例化） | 胞弟单元（Stage 1/2） | 须按本契约（§0.6 R1 / §4.1）提供调度表 |
| 测试 | 待实现 | 复用 `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad_3wg_fused_act.py` + §8.3 |

### 10.3 命名规范

- 项目目录：`examples/moe_grouped_gemm_nopad_3wg_fused_act/`
- 算子目录：`examples/moe_grouped_gemm_nopad_3wg_fused_act/_fused_act_kernel/`
- 实现文件：`_fused_act_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ D-3 探针
2. ⬜ 前导 kernel `_tile_scheduler_kernel`（block_m=128 实例化；胞弟单元）
3. ⬜ kernel G（Expert 单 Cube 双累加器 GEMM）+ kernel V（Developer packed 激活）+ L0 精度比对
4. ⬜ 三 kernel 协同测试（L0-8）+ 完整分层套件

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_persistent_3wg_fused_act.py`
  （`_fused_act_kernel` / `_make_pingpong_fused_act_kernel` / `_make_cooperative_fused_act_kernel` / `_act_expr`）
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tileops/ops/moe/moe_grouped_gemm_nopad_3wg_fused_act.py`、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad_3wg_fused_act.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad_3wg_fused_act.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad_3wg_fused_act/`
- 胞兄单元（归一化 GEMM 底座 + 调度表消费者）：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`
- 胞弟单元（调度表产出者）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`
- 底座证据（CG-2026-0026 复现 + Expert Cube 正对照）：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`
- 同类实现：`examples/silu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py`、
  `examples/gelu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py`、
  `examples/mixcv/mixcv_mixkernel.py`、`examples/gemm/matmul_dynamic_shape.py`、`examples/gemm/example_gemm.py`、
  `testing/autotune/anneal/moe_grouped_gemm_anneal.py`
- 等价性先例：`examples/silu_and_mul/_make_fused_gated_explicit/verify_equiv.py`、
  `examples/gelu_and_mul/_make_fused_gated_explicit/verify_equiv.py`、
  `examples/silu_and_mul/_make_fused_gated_explicit/repro/TRAP-silu-numerator-merge-overflow.py`
- API 文档：`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`、
  `docs/Tilelang.language/内存操作/T.store_fixpipe.md`、`docs/Tilelang.language/内存操作/T.alloc_L1.md`、
  `docs/Tilelang.language/内存操作/T.alloc_L0C.md`、`docs/Tilelang.language/比较操作/T.min.md`、
  `docs/Tilelang.language/数学操作/T.vexp.md`、`docs/Tilelang.language/数学操作/T.vdiv.md`、
  `docs/Tilelang.language/数学操作/T.verf.md`、`docs/Tilelang.language/数据类型转换操作/T.vcast.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/algorithm-research.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/INDEX.md`、
  `.../constants.md`、`.../traps-runtime.md`、`.../attention.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0001/0010/0012/0026/0033）
