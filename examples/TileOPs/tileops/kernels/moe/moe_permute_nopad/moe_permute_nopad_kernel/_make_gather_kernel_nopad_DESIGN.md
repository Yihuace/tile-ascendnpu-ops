# _make_gather_kernel_nopad 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py`
> 中的 `_make_gather_kernel_nopad`（GPU/CUDA target，MoE no-pad token 重排的**搬运单元 / Phase 2 gather**）。
> 本单元目标：`target="npuir"`（Ascend NPU，Ascend910B2C）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`（`_make_gather_kernel_nopad`）。
> 强参照（同族整数索引算子，含工具链陷阱规避）：
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（两 kernel GM 交接、CG-2026-0027/0028 规避、分核三要素）。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> **设计期可行性/性能实测（D-3，本机设备）**：
> `examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather.py`（小/中 shape + 双 dtype，
> 变体 A/B **逐位精确 + 6 次稳定**）、
> `examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather_perf.py`
> （代表 shape T=4096/H=7168/numel=32768：A=719.8µs、B(R=4)=696.5µs、B(R=8)=694.4µs，均 max_diff=0）。

**本单元的一句话定位**：`_make_gather_kernel_nopad` 是 MoE no-pad token 重排的**搬运单元**——按 scan 单元
（`_make_scan_kernel_nopad`）算出的 `permuted_idx[slot]`（每 tight 槽位对应的源 token 行号），把
`hidden_states[T, H]` 的对应行**收集（gather）**到专家连续区 `perm_h[numel, H]`（`numel = T*K`，无专家间 padding）。
它**不含任何数值算术**（无 +−×÷、无超越函数、无规约），是一个**纯行搬运 / 带宽受限**单元。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **GM→GM 直连拷贝改为「UB 中转 + 两段 DMA」**：源 kernel 用一条 `T.copy(hidden_states[permuted_idx[slot], 0:H], perm_h[slot, 0:H])`
   直接 GM→GM（GPU 上降为寄存器往返的两条向量 load/store）。**Developer 模式显式禁止 GM→GM**（`docs/Tilelang.language/内存操作/T.copy.md`
   §2.3-5「Unsupported memref to memref copy yet」，编译期报错）→ NPU 形态拆成 **GM→UB（动态源行）→ UB→GM（动态目的行）**
   两段 DMA。
2. **动态源行 gather 从「向量化索引 load」改为「标量 GM 读 → `T.copy` 动态 slice 起始」**：源
   `hidden_states[permuted_idx[slot], 0:H]` 是 GPU 的向量化 gather。NPU **无 GM 行 gather 向量原语**——`T.gather` 只按
   **最后一维**取数（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3「gather 当前按最后一维取数，不提供额外的
   `dim` 参数」），且跨步 gather 向量指令缺失（pattern-library `TRAP-C10-strided-gather`）→ 改为**每 slot 一次标量 GM 读
   `permuted_idx[slot]` 作为 `T.copy` 源 slice 的起始**。**D-3 探针设备实测位精确 + 6 次稳定**（变体 A）。
3. **超发 grid（`ceil(numel/8)` block × 每 thread 8 元素 128-bit）改为「固定物理核数（纯 Vector → 40）+ 核内 `T.serial`
   slot 循环」，整行 H 由单条 DMA 向量搬运**：源的 thread 级 `uint4` 向量化属 GPU thread/SIMD-lane 模型（NPU 无
   thread-per-element），NPU 用一条 DMA 承接整行（元素轴 H 连续、32B 对齐）。核内 `T.serial` 边界为**编译期常量**
   （core-split-strategy.md §1 要素③）。
4. **本单元必须与 scan 单元保持为两个独立 kernel（不得把 gather 内联进 scan 的 scope）**：scan 的 `permuted_idx` 由
   向量 / 原子 op 产出，gather 需要把它当**标量下标**消费；同一 scope 内「**向量产出 → 标量消费**」在 Developer 工具链
   **不可实现**（**CG-2026-0027**：同编译产物多次运行结果不一致、`T.pipe_barrier` 无效）。唯一可行形态是
   **`permuted_idx` 落 GM + kernel 边界隔离**，gather 侧**标量只读 GM**（同胞弟 `_tile_scheduler_kernel` §0.5 的
   `cum_gm` 标量 GM 读形态）。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = **`_make_gather_kernel_nopad`**（Phase 2 gather）。同文件的 `_make_scan_kernel_nopad` /
> `_make_scan_kernel_nopad_ep`（Phase 1 counting-sort scan）与 `MoePermuteNopadKernel.forward`（host/op 编排）由**胞弟单元**
> `_make_scan_kernel_nopad` 设计/覆盖，**不在本单元**；本单元的输入 `permuted_idx` 正是胞弟 scan 单元的产出（§0.6 对接契约）。

### 0.1 源算子语义（做什么）

**数学语义**（纯行搬运 / gather，无算术；输入输出同 dtype 的向量行）：

$$
\text{perm\_h}[s,\; :] \;=\; \text{hidden\_states}\big[\,\text{permuted\_idx}[s],\; :\,\big],
\qquad s\in[0,\,\text{numel}),\quad \text{numel}=T\cdot K
$$

- `permuted_idx[s] ∈ [0, T)`：tight 输出槽位 $s$ 对应的**源 token 行号**（由 scan 单元的 scatter 依据专家聚簇顺序写入）。
- 输出 `perm_h` 为 **tight（无 padding）**布局：恰好 `numel = T*K` 行，行 $s$ 即 tight 槽位；无专家间 gap。
- **无规约语义**：既无浮点累加也无整数累加；每个输出行独立于其它行（逐元素映射，无跨行依赖）→ **无累加顺序问题**。
- **dtype 语义**：`hidden_states` / `perm_h` 同 dtype（fp16 / bf16）；`permuted_idx` 为 int32；**无 dtype 提升或回落**，
  行搬运逐位透传（等价性可要求逐位精确）。
- **边界语义**：
  - `numel` 可能**不是** `ROWS_PER_BLOCK(=8)` 的整数倍（源测试 `partial-tail-numel10` 即 `T=5,K=2 → numel=10`）→ 源用
    `if slot < numel` 守卫最后一个 block 的尾部槽位；NPU 侧必须保留等价的尾块守卫（§5.4）。
  - `permuted_idx` 由调用方 `MoePermuteNopadKernel.forward` **零初始化**（`torch.zeros`），故 EP 模式下
    **未写入的槽位默认指向源行 0**（safe dummy 数据：`perm_h` 的 `[M_local, numel)` 段为源行 0 的副本，下游 tile scheduler
    从不访问）。本单元只需忠实执行 `perm_h[s]=hidden_states[permuted_idx[s]]`，该 dummy 语义**自动成立**（无需特判）。
  - 源 host 断言 `permuted_idx.numel() == numel`、`hidden_states.is_cuda`；NPU 侧由 op 层（胞弟/集成）保证。
- **host 侧语义**：`_make_gather_kernel_nopad` 本身**无 host 预处理**（纯 kernel 工厂）；`num_tokens`、`numel`、`hidden_size`、
  `dtype` 均为编译期参数（经 `functools.lru_cache` / op 层 `_kernel_cache` 按 shape 特化）。唯一 host 协作在调用方
  `forward`（分配 `perm_h`、顺序调用 scan 再调用 gather），不属于本单元语义。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（PyTorch 高级索引 / `torch.gather` 的**行 gather**），**不得复刻**
§0.6 的 NPU 算法（UB 中转 + 标量 GM 读动态 slice）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `hidden_states` | `(num_tokens, hidden_size)` = `(T, H)` | fp16 / bf16 | 输入激活（行 = token） |
| `permuted_idx` | `(numel,)` = `(T*K,)` | int32 | 每 tight 槽位对应的源 token 行号（scan 单元产出） |
| `perm_h` | `(numel, hidden_size)` = `(T*K, H)` | fp16 / bf16 | **输出**：tight 专家连续区（gather 结果） |

源签名：`@tilelang.jit(out_idx=[], ...)`（**输出由调用方分配并传入**，非设备侧 `out_idx` 分配）；编译配置
`compile_flags=["-O3", "-DENABLE_BF16"]`。源 `_make_gather_kernel_nopad` 的工厂参数为
`(num_tokens, numel, hidden_size, dtype)`。

### 0.3 实现算法解读（怎么算）

源 kernel 结构（源 L218–L258）：工厂先算
`VEC = 8; threads = min(1024, H // 8)` 并递减到 `H % threads == 0`（至少 1）；
`ROWS_PER_BLOCK = 8; grid = ceil(numel / 8)`。kernel 为 1D `T.Kernel(grid, threads=threads)`：

- **每 block**：`for r in T.serial(ROWS_PER_BLOCK)`（8 个槽位）；`slot = bid*8 + r`；`if slot < numel` 守卫；
  对每个 slot 执行 **一次直接 GM→GM 向量化行拷贝**：
  `T.copy(hidden_states[permuted_idx[slot], 0:H], perm_h[slot, 0:H])`（源注释：无寄存器 fragment 往返，
  两者 lowering 一致）。每 thread 处理 8 个 bf16（128-bit `uint4`）。

**计算步骤分解**（源码每条计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| G1 | `slot = bid * ROWS_PER_BLOCK + r`（r 串行 0..7） | block/循环 id | `slot` | 输出槽位下标 $s$ |
| G2 | `if slot < numel` | `slot` | 分支 | 尾块守卫（`numel % 8 != 0` 时） |
| G3 | `src = permuted_idx[slot]` | `permuted_idx` | 源行号 | $\text{permuted\_idx}[s]$ |
| G4 | 行拷贝 `hidden_states[src, 0:H] → perm_h[slot, 0:H]`（128-bit 向量化，每 thread 8 元素） | `hidden_states` | `perm_h` | $\text{perm\_h}[s,:]=\text{hidden\_states}[\text{pidx}[s],:]$ |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[hidden_states(T, H)] ──┐
                          ├─(每 block r=0..7：src=permuted_idx[slot])──> GM[perm_h(numel, H)]
GM[permuted_idx(numel)] ──┘       直接 GM→GM 向量化行拷贝（无中间 buffer）
```

**循环与并行结构**：grid = `ceil(numel/8)` 个 block（prefill 下 numel=32768 → 4096 个 block）；每 block
`T.serial(8)` 个槽位；thread 级并行（threads ≈ H/8，每 thread 8 元素 128-bit）。**未使用 `T.Parallel` / `T.Pipelined`**。

**host 侧逻辑**：无（本 kernel 无 host 预处理；`perm_h` 由调用方分配，`permuted_idx` 由前面的 scan 提供）。
**编译配置**：`@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | `ROWS_PER_BLOCK=8` 槽位 / block（而非一 block 一槽位） | 摊薄 launch / 调度开销 | 每 block 连续处理 8 个输出行 | CUDA 廉价 block 启动 + 网格调度 | 硬件相关（超发 grid 形态） |
| O2 | 直接 GM→GM 向量化行拷贝（无寄存器 fragment 往返） | 省一次寄存器中转 | TileLang 将两侧 lowering 一致 | GPU 统一地址空间（load/store 同址空间） | 硬件强相关（NPU 禁 GM→GM） |
| O3 | 每 thread 8 个 bf16（128-bit `uint4`）向量化 load/store | 用满内存事务宽度 | 线程级 128-bit 访问 | CUDA thread + 128-bit LDG/STG | 硬件强相关（NPU 无 thread-per-element） |
| O4 | `threads = H // 8` 按行宽配线程数（cap 1024，并保证整除 H） | 一行恰由一线程块铺满 | thread 数与 H 匹配 | CUDA warp/thread 模型 | 硬件强相关（NPU 由 DMA 承接整行） |
| O5 | 动态源行 gather（`permuted_idx[slot]` 作源行号） | 实现 tight 重排的按槽位寻址 | 运行期行号索引 load | 无（纯算法 / 索引语义） | **算法层可移植**（语义保留，形态见 R2） |
| O6 | `grid = ceil(numel/8)` 超发块 + `if slot < numel` 尾守卫 | 输出定长、免 host 预处理 | 网格按上界静态分配，尾块守卫 | CUDA 廉价 block 启动 | 硬件相关（NPU 超发被串行调度） |
| O7 | `hidden_states[src,0:H]` 用 2D 行切片而非逐元素索引 | 保持整行连续 DMA 语义 | 整行 H 连续 | 无（算法层已隐含整行连续性） | 算法层可移植（NPU 形态必须保留整行 DMA） |

> 识别不出来 ≠ 不存在。O1–O7 为源码显式优化（源码注释与结构直读），逐项在 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（`perm_h[s]=hidden_states[pidx[s]]`、tight 布局、int32 索引、fp16/bf16 逐位透传、尾块守卫、dummy-row-0 语义） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| O5 动态源行 gather（按槽位选源行） | 算法 | 无 | 有（算法层） | **保留** | 行选择语义保留；实现形态从「向量化索引 load」改为「标量 GM 读 → `T.copy` 动态 slice」（R2，D-3 探针实测位精确） |
| O7 整行连续搬运 | 算法 | 无 | 有 | **保留** | 保留「整行一次 DMA」语义（元素轴 H 连续、32B 对齐） |
| O2 直接 GM→GM 向量化行拷贝 | 优化 | GPU 统一地址空间 | **无**（Developer 禁 GM→GM） | **等价替换** | 拆为 **GM→UB（动态源行）→ UB→GM（动态目的行）** 两段 DMA；依据 `docs/Tilelang.language/内存操作/T.copy.md` §2.3-5（「Developer 模式不支持 GM→GM 拷贝（编译期报错）」）（R1） |
| O3 每 thread 8 bf16 128-bit 向量化 | 优化 | CUDA thread + 128-bit LDG/STG | 无（NPU 无 thread-per-element） | **重新设计** | 整行 H 由**单条 DMA** 向量搬运（NPU MTE2 自动向量化，元素轴连续）；依据 `docs/Tilelang.language/内存操作/T.copy.md` + D-3 探针（R3） |
| O4 `threads=H//8` 按行宽配线程 | 优化 | CUDA thread 模型 | 无 | **舍弃** | NPU 无 thread 概念；线程数由 DMA 粒度替代（`docs/Tilelang.language/编译器提示操作/T.Kernel.md`；`TRAP-threads-kwarg-noop`：npuir 下 `threads=` 无效）（R3） |
| O1 `ROWS_PER_BLOCK=8` / block | 优化 | CUDA 廉价 block 启动 | 无（一维核超发被串行调度） | **重新设计** | 固定物理核数（纯 Vector → 40）+ 核内 `T.serial` slot 循环（core-split-strategy.md §1 要素③）（R3） |
| O6 `grid=ceil(numel/8)` 超发块 + 尾守卫 | 优化 | CUDA 廉价 block 启动 | 无（超发被串行调度） | **重新设计** | 固定核数 + 核内串行，保留 `if slot < numel` 尾守卫（静态 `spc` 边界）（R3） |
| 「gather 内联进 scan 单元」 | 结构（非源，本单元新引入候选） | 无 | **无可用的「向量产出 → 标量消费」同步原语** | **舍弃** | **CG-2026-0027**：`permuted_idx` 由 scan 的向量/原子 op 产出，若在同 scope 内被 gather 当标量下标消费 → 同产物多运行不一致；`T.pipe_barrier` 无效。**故 scan→gather 的 kernel 边界（`permuted_idx` 落 GM + 标量只读 GM）是必要条件，不是组织偏好**（R4） |
| 「host 侧 gather」 | 结构 | 无 | 有（CPU 可行） | **舍弃** | 需 `hidden_states` 全量设备→host→设备往返，引入 CPU 同步、违反源「零 CPU 同步 / 设备侧数据流」意图，且带宽数量级劣化（PCIe vs HBM） |

> 每项处置依据见「NPU 对应方案 / 依据」列；**语义层不可妥协**。
> **判定输入含 Phase R 调研结论（§1.6.0）**：调研在「可实现族」内择优——UB 中转 + 标量 GM 读动态 slice（主选，D-3 探针位精确）
> 胜出；批量写（变体 B）作为可实现的备选；gather 内联（CG-2026-0027）与 host 侧（CPU 同步）被否决（可行性前置于经济性）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计 / 等价替换」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**单独立 Vector kernel，固定物理核数（纯 Vector → 40）+ 核内 `T.serial` slot 循环；
每 slot「标量 GM 读 `permuted_idx[slot]` → 动态源行 GM→UB → 动态目的行 UB→GM」两段 DMA；元素轴 H 由整行 DMA 向量搬运**）。

**重设计项 R1：数据通路——从「直接 GM→GM」改为「UB 中转 + 两段 DMA」**

- **源方案**：`T.copy(hidden_states[permuted_idx[slot], 0:H], perm_h[slot, 0:H])` 一条 GM→GM 拷贝（GPU 上等效
  两条向量 load/store；源码注释强调「无寄存器 fragment 往返」）。
- **NPU 新算法**：`ub_row = T.alloc_ub((1, H), dtype)`；每 slot 两段：
  `T.copy(hidden_states[src:src+1, 0:H], ub_row[0:1, 0:H])`（GM→UB，动态源行）+
  `T.copy(ub_row[0:1, 0:H], perm_h[slot:slot+1, 0:H])`（UB→GM，动态目的行）。
- **语义保持论证**：两段拼接即恒等拷贝（`A→UB→B` 当 UB 容量 ≥ 一行，逐位透传）；行宽 H 由 DMA 一次性搬运，
  无重排、无 cast（dtype 两侧一致）。D-3 探针在 bf16/fp16 × 多 shape 下 **max_diff=0（逐位精确）+ 6 次稳定**。
  边界：`numel` 非 8 倍数的尾块由 `if slot < numel` 守卫（源等价）。
- **与源优化意图的承接**（§0.4 逐项，**不得静默丢弃**）：O7（整行连续搬运）→ 保留（两段 DMA 均整行）；
  O2（免寄存器往返）→ **意图承接、形态替换**（NPU 无统一地址空间，UB 中转是唯一路径，仍保持「无逐元素标量循环」）。

**重设计项 R2：动态源行 gather——从「向量化索引 load」改为「标量 GM 读 → `T.copy` 动态 slice 起始」**

- **源方案**：`hidden_states[permuted_idx[slot], 0:H]`——运行期行号索引，GPU 上为一次向量化 gather load。
- **NPU 新算法**：`src = permuted_idx[slot]`（**标量 GM 读**，非 UB 向量产出）；以 `hidden_states[src:src+1, 0:H]`
  的**动态 slice 起始**驱动 GM→UB DMA。每 slot 1 次标量 GM 读 + 1 条行 DMA。
- **为什么不能用 `T.gather`**：`T.gather` 只按**最后一维**取数、`src` 须驻 UB
  （`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3/§2.4：`T.gather(A_VEC, B_VEC, indices, [M,N])` 沿最后一维，
  无 `dim` 参数）；且跨步 gather 向量指令缺失（pattern-library `TRAP-C10-strided-gather`，status: verified）。
  hidden_states 全量（prefill 58.7MB）不可能驻 UB。→ 行 gather 无向量原语，必须逐行动态偏移 DMA。
- **语义保持论证**：`src = permuted_idx[slot]` 就是源行号；动态 slice 请求的正是该行。D-3 探针**位精确 + 6 次稳定**，
  **闭合「标量 GM 读 → 动态 slice 起始」这一未文档化假设**（对比：胞弟 `_tile_scheduler_kernel` 的标量 GM 读只控制
  **长度/写起点**，本单元首次用它控制**读起点**，已实测）。
- **规避 CG-2026-0027 的关键**：`src` 读自 **GM**（由 scan 在**另一个 kernel** 写入），**不是本 scope 向量 op 产出的 UB 值**；
  同 scope 内无「向量产出 → 标量消费」数据流。

**重设计项 R3：任务划分与分核——从「超发 grid + thread 级向量化」改为「固定物理核数 + 核内 `T.serial` + 整行 DMA」**

- **源方案**：`T.Kernel(ceil(numel/8), threads≈H/8)` 超发 block；每 thread 8 元素 128-bit。
- **NPU 新算法**：`nk = min(numel, P)`（`P` = **纯 Vector 实查物理核数 = 40**，§5.5）；`spc = ceildiv(numel, nk)`
  （**编译期常量**，核内 `T.serial` 边界静态）；核内 `for s in T.serial(spc): slot = core_id*spc + s`；
  `if slot < numel:` 守卫。整行 H 由单条 DMA 搬运（元素轴向量化由 DMA 承接）。
- **语义保持论证**：分核只改变「哪个核处理哪些输出行」，不改变任何行的 gather 结果；纯搬运、无副作用。
  边界：`spc` 静态保证循环边界安全；末核 `if slot < numel` 守卫。
- **说明**：超发 grid 的源形态在 NPU 上会被**串行调度**（core-split-strategy.md §1），不可取；`threads=` 在 npuir 下
  无效（`TRAP-threads-kwarg-noop`）。

**重设计项 R4：承载形态——gather 必须独立成 kernel，`permuted_idx` 经 GM 交接**

- **源方案**：gather 是独立的 Phase 2 kernel（源本身就独立；host 顺序调用 scan → gather）。
- **NPU 新算法（保留 + 强化）**：**保留**源「scan → gather 两 kernel、`permuted_idx` 经 GM 交接」的数据流；
  并**强化为必要条件**：gather 的标量下标必须读 **GM**（跨 kernel），**绝不能**把 gather 内联进 scan scope 后
  标量消费该 scope 的向量产出（CG-2026-0027）。
- **语义保持论证**：读写的唯一变化是「gather 从 GM 读 `permuted_idx`」，与源 host 顺序调用语义一致（scan 先写、
  gather 后读，同 stream 顺序）。
- **对接契约（与胞弟 `_make_scan_kernel_nopad` 严格对齐）**：
  1. `permuted_idx` dtype **int32**、shape `(numel,)`；值域 `[0, num_tokens)`（EP 模式下未写槽位默认 0）；
  2. `perm_h` dtype 与 `hidden_states` 一致（fp16/bf16）；shape `(numel, H)`；
  3. **scan 先于 gather、同一 stream 顺序 launch**，无 in-block 同步、无 CPU 同步；
  4. `permuted_idx` **零初始化**（调用方负责）保证 dummy-row-0 语义（EP 模式）。

**重设计项 R5：无（算法层）**。行 gather 的**算法本身**（按槽位选源行 + 整行拷贝）在 NPU 上有等价能力，
按 §0.5 为「保留」；R1–R3 只改**实现形态**（UB 中转 / 标量 GM 读动态 slice / 固定核数核内串行），不换算法族；
R4 为承载形态确认 + 工具链必要性强化。

### 0.7 标杆实现

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py` 的 `_make_gather_kernel_nopad`（L218–L258）。
- 源 op（host 编排 / 接口）：`/tmp/opencode/TileOPs-ref/tileops/ops/moe/routed_expert/permute_nopad.py`（`MoePermuteNopadFwdOp`）。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`（`_make_gather_kernel_nopad`）。
- op 级参考实现（golden 依据）：`examples/TileOPs/tests/ops/test_moe_permute_nopad.py`
  的 `_ref_moe_permute_nopad`（纯 PyTorch counting-sort + 高级索引 gather）。
- 同类行 gather 先例（Expert 模式）：`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py`
  （L157–L167：`for i in T.serial(block_I): cur_idx = idx_frag[i]; T.copy(IndexK[cur_idx:cur_idx+1, 0:index_dim], k_shared[i:i+1, 0:index_dim])`
  ——「gather via idx loop with explicit serial reads」，2-D slice 保持 rank 一致）。
- 同族工具链陷阱规避先例：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（两 kernel GM 交接 / CG-2026-0027/0028）。
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`、`examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoePermuteNopadFwdOp`）。
- D-3 探针：`examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather.py`、
  `examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather_perf.py`。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_make_gather_kernel_nopad`（MoE no-pad token 重排的**搬运单元 / Phase 2 gather**，纯行搬运、带宽受限，`target="npuir"`）。

### 1.2 功能描述

按 scan 单元产出的 `permuted_idx[slot]`（每 tight 槽位的源 token 行号），把 `hidden_states[T, H]` 的对应行
**收集**到专家连续区 `perm_h[numel, H]`（`numel = T*K`，tight 无 padding）。**迁移决策后的 NPU 形态**（§0.6）：
**独立 Vector kernel**，固定物理核数（纯 Vector → 40）+ 核内 `T.serial` slot 循环；每 slot「标量 GM 读
`permuted_idx[slot]` → 动态源行 GM→UB → 动态目的行 UB→GM」两段 DMA；整行 H 由 DMA 向量搬运。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
\text{perm\_h}[s,:]=\text{hidden\_states}[\text{permuted\_idx}[s],:],
\qquad s\in[0,\text{numel}),\ \text{numel}=T\cdot K
$$

输入 `hidden_states (T,H) fp16/bf16`、`permuted_idx (numel,) int32`；输出 `perm_h (numel,H) fp16/bf16`。
**无算术 — FLOPs = 0**（manifest roofline `flops: "0"`，`examples/TileOPs/tileops/manifest/moe.yaml`）。
manifest op 级 roofline `bytes: "(T*H + T*K*H)*elem_bytes + metadata"` 描述的是 **op 整体**（含 scan 的 metadata），
本搬运单元的流量见 §1.6.0 R3。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源结构差异已逐条注明）：

1. **分核与任务划分**（§0.6 R3）：`nk = min(numel, P)`（`P` = 纯 Vector 实查物理核数 = 40，§5.5）；
   `spc = ceildiv(numel, nk)`（编译期常量）；核内 `for s in T.serial(spc): slot = core_id*spc + s`；`if slot < numel` 守卫。
   **与源差异**：源为**超发 grid（`ceil(numel/8)` block）+ thread 级一槽位**；v-NPU 为**固定物理核数 + 核内串行**。
2. **单槽 gather 数据通路**（§0.6 R1/R2）：`src = permuted_idx[slot]`（标量 GM 读）→
   `T.copy(hidden_states[src:src+1, 0:H], ub_row[0:1, 0:H])`（GM→UB，动态源行）→
   `T.copy(ub_row[0:1, 0:H], perm_h[slot:slot+1, 0:H])`（UB→GM，动态目的行）。
   **与源差异**：源为**直接 GM→GM 一条拷贝 + 向量化索引 load**；v-NPU 为**UB 中转两段 DMA + 标量 GM 读驱动动态 slice**。
3. **承载形态**（§0.6 R4）：本单元是**独立 kernel**，`permuted_idx` 与 `perm_h` 均经 **GM**；与 scan 单元同 stream
   顺序 launch（scan 先写 `permuted_idx`，gather 后读）。本单元不产出/不消费任何向量 op 产出的 UB 值给标量代码。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1（结论：纯搬运无优化空间）；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[scan 单元 · 独立 kernel]  ──（同 stream 顺序）──┐
  产出 GM[permuted_idx(numel) int32]             │
                                                 ▼
[本单元 · Vector · nk = min(numel, 40) cores]
  GM[permuted_idx] ──标量 GM 读 src──┐
                                     ▼
  GM[hidden_states(T,H)] ──T.copy 动态源行[src:src+1,0:H]──▶ UB[ub_row(1,H)]
                                                                    │ T.copy（动态目的行 [slot:slot+1,0:H]）
                                                                    ▼
                                            GM[perm_h(numel,H) fp16/bf16]
```

> 数据路径只有 **GM → UB → GM**（无 L1 / L0A / L0B / L0C；本单元无矩阵计算、无 Cube）。
> 每核独立处理 `spc` 个槽位；核间无通信、无归并。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（**按给定行号序列逐行 gather 重排** `perm_h[s]=hidden_states[pidx[s]]`）的**算法族**。
> 输入/源算法（GPU 超发 grid + GM→GM 向量化行拷贝 + thread 级 128-bit）只是基线候选之一。
> 深度分级：纯搬运类 → 按 §2 可轻量，但本单元含**动态索引 + 超发 grid 形态改造 + 工具链可行性**三处非平凡决策，
> 故做**完整调研**（R1 候选表 + R3 四口径 + R4 逐候选）。
> **可实现性约束（前置，来自 Phase M1 + 本仓工具链证据）**：
> (a) Developer 模式**禁止 GM→GM**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3-5，编译期报错）→ 源「一条 GM→GM 拷贝」形态
>     在本项目**不可直接落地**，必须 UB 中转；
> (b) 同一 scope 内「向量产出 → 标量消费」不可同步（**CG-2026-0027**，`permuted_idx` 若在 scan scope 内被标量当行号消费即命中）
>     → 任何「gather 内联进 scan」候选**不可实现**；
> (c) 行 gather 无向量原语：`T.gather` 仅尾维（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3）、
>     跨步 gather 向量指令缺失（`TRAP-C10-strided-gather`）→ 任何「单指令向量行 gather」候选**不可实现**。
> 调研在**可实现族**内择优（可行性前置于经济性）。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | 超发 grid（`ceil(numel/8)` block）+ thread 级 128-bit；**直接 GM→GM** 向量化行拷贝（动态源行） | — | — | ✅ 基线 |
| C1 | **UB 中转 + 标量 GM 读动态 slice**（**选定**） | 固定核数 + 核内 serial；每 slot：标量 GM 读 `src` → GM→UB（动态源行）→ UB→GM（动态目的行） | 数学恒等（`A→UB→B` 恒等拷贝，逐位透传） | 保持整行 DMA；规避 GM→GM 禁令；无逐元素标量 | ✅ **选定** |
| C2 | **批量写（R 行攒批）** | 在 C1 基础上每 R 个输出行：逐行 gather 入 `ub_batch[j:j+1]`，再一条连续 UB→GM 写 R 行 | 数学恒等（同 C1，写侧合并） | 写侧 DMA 数 ÷R（读侧不变） | ✅ 对比（备选，实测 +3%） |
| C3 | **读一次 & 反向散写（read-once scatter）** | 按**源 token**（T 行各读一次）广播写到其 K 个目标槽位（用 `fwd_idx` 反映射） | 数学恒等（输出与 C1 逐位一致） | 流量从 `2·numel·H` 降至 `(T + numel)·H`（**达 roofline**，prefill −44%） | ❌ 弃选：**接口不含 `fwd_idx`**（本单元输入只有 `permuted_idx`；需 `fwd_idx` 或核内求逆） |
| C4 | **`T.gather` 尾维 gather（转置法）** | 把 `hidden_states` 视作 `(H, T)` 尾维，用 `T.gather(A_VEC, C_VEC, permuted_idx)` 一次取全体、再转置回 | 数学恒等（转置后尾维即 token 轴） | 理论上单指令完成 gather | ❌ 弃选：`T.gather` 的 `src` 须驻 **UB** 且沿尾维；`(H,T)` 需 58.7MB（prefill）≫ 192KB；分块则引入巨大转置链与 UB 驻留（见 §1.6.3 L3） |
| C5 | **选择矩阵 GEMM** `perm_h = P·hidden_states` | `P (numel,T)` 0/1 选择矩阵，Cube 乘 | 数学恒等（P 每行恰一 1） | 与 C1 同结果 | ❌ 弃选：FLOPs `2·numel·T·H ≈ 1.9e12`（prefill），且 `P` 需物化 268MB；**纯搬运退化为巨量 MAC** |
| C6 | **host 侧 gather** | CPU 索引拷贝 | 数学等价 | — | ❌ 弃选：全量设备→host→设备往返 + CPU 同步（违反源「零 CPU 同步」意图；带宽数量级劣化） |
| C7 | **逐元素索引（`hidden_states[pidx[s], h]` 标量）** | 每元素一次标量 load/store | 数学恒等 | — | ❌ 弃选：把整行 DMA 打回逐元素标量（`H·numel` 次标量访存，失 SIMD/DMA）；源本身即「整行」 |

**R2 在线算法**：**有（且本质就是单遍流式）**。本算子是**纯逐元素/逐行映射**（输出行 $s$ 只依赖输入行
`permuted_idx[s]`），无跨行反向依赖、无 running 统计量 → **单遍即天然在线**：每读一次 `permuted_idx[s]` 就产出一行，
O(1)（实际 O(1 行)）中间状态。**结论严谨化**：不存在「两遍/多遍物化」的候选（不同于 softmax/attention 的
max→sum 两遍）；唯一可能的多遍形态是 C3（读源一次 + 写目标）——它仍是**单遍**（每源行读一次、每目标行写一次），
只是遍历方向由「按槽位」改为「按源 token」。
（对照 `references/algorithm-candidates.md` 的 `ALG-reduction` / gather-分桶族——本单元属「逐行索引映射」，无归约。）

**R3 复杂度对比**（四口径；代表 shape = kimi-k2-prefill：T=4096，K=8，H=7168，numel=32768，bf16；
`nk=40`、`spc=ceildiv(32768,40)=820`；elem=2B）：

| 算法候选 | 计算量（FLOPs/索引 op） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 |
|---------|------------------------|---------------|---------|-------------|
| C0 基线（源；GPU 超发 grid + GM→GM） | 0（无算术）+ `numel` 次行索引 | 读 `numel·H·2 = 470MB` + 写 `470MB` = **939.5MB** | 输入 1 读 + 输出 1 写 | 无（GM↔GM 直连） |
| **C1 选定（UB 中转 + 标量 GM 读）** | 0 + `numel` 次标量 GM 读（4B） | 读 470MB + 写 470MB + 标量读 `numel·4=131KB` ≈ **939.6MB** | 同 C0（1 读 1 写） | UB `ub_row` = `H·2 = 14KB` |
| C2 批量写（R=4） | 0 + 同上 | 同 C1（**流量不变**，仅 DMA 次数 ↓） | 同 C1 | UB `ub_batch` = `R·H·2 = 57KB` |
| C3 读一次散写（接口受限） | 0 + 反映射索引 | 读 `T·H·2=58.7MB` + 写 `numel·H·2=470MB` = **528.5MB**（达 roofline） | 源 1 读 + 输出 1 写 | 无（或 1 行） |
| C4 尾维 gather 转置 | 0 + 转置 | 读 470MB + 写 470MB + **转置链额外 UB/时间** | ≥2（转置往返） | `(H_tile·numel)` 巨大 → 必分块 |
| C5 选择矩阵 GEMM | **1.9e12 MAC**（Cube） | +物化 `P` 268MB | 1 | `P` 268MB + L0C |
| C7 逐元素索引 | 0 + `numel·H` 次标量访存 | 同 C1 流量但微小事务 | `numel·H` 次标量 | 无 |

口径脚注：计算量只计算术（本算子 FLOPs=0，索引 op 另列）；访存 = GM 侧实际流量（读 + 写），不含 UB 内部；
扫描遍数指输入/输出在 GM 侧各被读写的次数；中间缓冲峰值以片上（UB）为准。
**本单元是纯带宽受限搬运**（C1 实测 719.8µs @ 939.6MB = **1.31 TB/s**，已 ≥ MTE2 混合流量地板 ~1.26 TB/s，
见下 R4/roofline），故 R3 排序以**访存量**为主导；C2/C3 的价值在于削减 DMA 次数 / 流量，而非算力。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向论断按 negative-claim-evidence.md §1 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | 需 **GM→GM**（NPU 无此路径）→ ❌ | 无 UB 需求 ✅ | 行 32B 对齐 ✅ | grid 编译期 ✅ | 无 | ❌ `docs/Tilelang.language/内存操作/T.copy.md` §2.3-5「Developer 模式不支持 GM→GM」编译期报错 | ⚠️ **不可直接落地**（须 UB 中转，转 C1） |
| C1 **选定** | Vector/MTE2（整行 DMA）✅；无 Cube 需求 | UB 14KB ≪ 192KB ✅ | H·2 为 32B 倍数（H=7168→14336B）✅ | `nk`/`spc` 编译期常量 ✅ | 单 scope，段间可流水 ✅ | ✅ **D-3 探针设备实测位精确 + 6 次稳定**（双 dtype × 多 shape），**闭合「标量 GM 读 → 动态 slice」假设** | ✅ **选定** |
| C2 批量写 | 同 C1 ✅ | UB 57KB（R=4）≪ 192KB ✅ | 同 C1；`ub_batch[j:j+1]` 为非零起点 UB dst（历史 `TRAP-UB-dst-align`）→ 探针**实测未触发**（本工具链） | 同 C1 ✅ | 同 C1 ✅ | ✅ D-3 探针实测位精确 + 6 次稳定；**性能仅 +3%（696.5µs vs 719.8µs）** | ✅ **备选**（实测增益小，风险面略大） |
| C3 读一次散写 | Vector/MTE2 ✅ | 无 ✅ | ✅ | ✅ | ✅ | ✅ 可实现**但接口不含 `fwd_idx`** | ❌ 弃选（接口；若未来传入 `fwd_idx`，可省 44% 流量，见风险 R-4） |
| C4 尾维 gather 转置 | Vector（VGather）✅ | ❌ `(H,T)`=58.7MB ≫ 192KB，分块后转置链成本高 | VGather 索引须 int32、src 驻 UB | 需分块静态化 | 转置链 + 分块 → 多段 | ⚠️ API 存在（`testing/npuir/indexing_ops/test_slice_gather_dev.py` 实测可用）但不满足容量 | ❌ 弃选（容量 + 转置代价，§1.6.3 L3） |
| C5 选择矩阵 GEMM | Cube ✅ | L0C 128KB / P 268MB 超 L1 | 分形对齐 | ✅ | GEMM 可流水 | ✅ Cube 可用但 **1.9e12 MAC** | ❌ 弃选（FLOPs 爆炸，退化为巨量 MAC） |
| C6 host gather | CPU | — | — | — | ❌ 设备↔host 往返 + CPU 同步 | ✅ 可行但违源意图 | ❌ 弃选（违反零 CPU 同步 + 带宽劣化） |
| C7 逐元素索引 | Vector 标量 | 无 | — | — | ❌ 标量洪流 | ✅ 可编译但极慢 | ❌ 弃选（`numel·H` 次标量访存，失 DMA） |

**调研结论**：选定 **C1——单独立 Vector kernel，固定物理核数（纯 Vector → 40）+ 核内 `T.serial` slot 循环；
每 slot「标量 GM 读 `permuted_idx[slot]` → 动态源行 GM→UB → 动态目的行 UB→GM」两段 DMA；整行 H 由 DMA 向量搬运**；
**备选 C2（批量写 R 行）**（D-3 探针实测 +3%，结构完整可实现）。
关键依据：(i) **可实现性是前置约束**——C0（GM→GM）被 Developer 禁令阻断、C4（`T.gather` 容量）与 C5（FLOPs）与 C3（接口）
分别被容量/复杂度/接口淘汰，C1 是唯一同时满足「整行 DMA + 无 GM→GM + 无向量→标量同 scope 数据流」的形态，且经
**D-3 探针设备实测位精确 + 6 次稳定**；(ii) C1 实测 **1.31 TB/s** 已达 MTE2 混合流量地板（~1.26 TB/s，`CONST-mte2-degradation`），
即**已在带宽 roofline 上**，继续结构改造的边际空间极小（C2 仅 +3%）；(iii) C3 的 44% 流量收益受接口约束不可得（列入风险 R-4）。
**与基线结构差异一句话**：保留源「独立 Phase 2 kernel、按槽位选源行、整行连续拷贝」的算法骨架，把源「超发 grid + thread
级 128-bit + 直接 GM→GM」换成「固定物理核数核内串行 + 整行 DMA + UB 中转两段拷贝」，并把「标量消费向量产出」用
**kernel 边界（`permuted_idx` 落 GM）** 彻底隔离。
**调研范围**：`references/algorithm-candidates.md`（无 gather/permute 专条；参考 gather-分桶 / `ALG-reduction` 族条目）、
pattern-library（`layout.md`、`elementwise.md` PL-1.6-copy-floor、`traps-runtime.md`、`traps-compiler.md` TRAP-C10、`constants.md`）、
源算子 §0.3/§0.4、本仓 examples（`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` 的行 gather 先例、
`examples/cumprod/_cumulative_kernel`、`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`）、
`testing/npuir/indexing_ops/test_slice_gather_dev.py`、`.agents/evolution/capability-gaps.md`（CG-2026-0027/0028）。
互联网检索：**未执行**（本地信息源已覆盖 gather 候选空间：直接索引 / 尾维 gather / 选择矩阵 / 分桶；无需外部补充）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：
O1（`ROWS_PER_BLOCK=8`/block，摊薄启动）→ 重设计为「固定核数 + 每核 `spc=820` 槽位串行」（**意图承接**：摊薄启动）；
O2（免寄存器往返的直连拷贝）→ **形态替换**（UB 中转是 NPU 唯一路径，仍保持「无逐元素标量」）；
O3（thread 级 128-bit 向量化）→ 重设计（整行 DMA 自动向量化）；
O4（`threads=H//8`）→ 舍弃（NPU 无 thread；`TRAP-threads-kwarg-noop`）；
O5（动态源行 gather）→ **保留**（算法层语义保留，实现改为标量 GM 读驱动动态 slice）；
O6（超发 grid + 尾守卫）→ 重设计（固定核数 + 保留尾守卫）；
O7（整行连续切片）→ **保留**（两段 DMA 均整行）。

**设计期 roofline（D-2，代表 shape kimi-k2-prefill）**：以 §1.6.3 选定布局 + `pattern-library/constants.md` 实测常数估算下界：
- **流量项**：`numel·H·2·2 = 939.5MB` ÷ MTE2 混合流量地板 `~1.26 TB/s`（`CONST-mte2-degradation`，
  toolchain tilelang 0.1.2+ed787bb / CANN 8.5.0，origin lerp_tensor/ada_layer_norm/ssd_chunk_scan 任务）≈ **746 µs**。
- **发射项**：DMA 发射成本**未在 pattern-library 实测**（`CONST-vector-launch-overhead` 是**向量 op** 的 ~0.5µs/op，
  不直接适用于 MTE2 DMA）——标注「未文档化假设 + 估算依据」；由 D-3 性能探针标定（实测 719.8µs，见下）。
- **容量项**：UB `ub_row` 14KB ≪ 192KB（`CONST-capacity-910B2C`）。
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 746 µs（流量项主导）**。
- **D-3 实测标定（本机设备，2026-10-14，toolchain tilelang 0.1.2+3f0f004 / CANN 8.5.0 / Ascend910B2C）**：
  C1=**719.8µs**（1.31 TB/s）、C2(R=4)=**696.5µs**、C2(R=8)=**694.4µs**，全部 max_diff=0。
  → 估算下界与实测吻合（实测略优于估算 3.5%，因 L2 驻留/streaming 使有效带宽略高于 HBM 地板）；**发射项非主导**，
  每行 DMA 的发射开销在 14KB 传输下可忽略（否则 65536 次 DMA 会显著超出 746µs）。
- **未文档化假设项**：单行 DMA（含动态偏移）的**发射成本**与「标量 GM 读 → 动态 slice 起始」的**生成码正确性**
  ——后者已由 D-3 探针**转正为实测结论**（位精确 + 稳定）；前者不影响主选/备选裁决（带宽已主导）。

**实验裁决计划（三件套）**：

- **主选（进 Stage 3）**：C1——固定核数 + 核内 `T.serial`；每 slot 两段 DMA（0 起点 UB dst，最简形态，§3.3 主选）。
- **备选（≤1，结构完整可实现）**：C2——在 C1 上增加 R=4 的 `ub_batch((R,H))`，逐行 gather 入 `ub_batch[j:j+1, 0:H]`
  （非零起点 UB dst，须复核 `TRAP-UB-dst-align`），再一条连续 UB→GM 写 R 行。UB 预算 57KB；分核三要素按 batch 粒度重算
  （`nk=min(numel//R, 40)`、`spc` 取 R 的倍数）。D-3 探针**已实测位精确 + 6 次稳定 + 性能 +3%**（§3.4 备选伪代码）。
- **裁决计划**：
  ① **代表 shape** = smoke（T=4,K=2,H=64,numel=8）、partial-tail（T=5,K=2,H=64,numel=10）、
     kimi-k2-prefill（T=4096,K=8,H=7168,numel=32768）、qwen3-30b-prefill（T=4096,K=8,H=3072,numel=32768）；
  ② **测量指标** = 首编编译结果 + 输出与 golden **逐位精确一致**（非容差）+ 反复运行稳定 + 墙钟延迟（msprof / 计时）；
  ③ **判定阈值** = 备选 latency < 主选 × (1−ε) 才翻转，取 **ε=0.05**（实测仅 +3%，低于阈值 → **默认保持主选 C1**）；
     若某 shape 上备选显著更快（>5%）则按 shape 分派；
  ④ **回写路径** = 实测数字回写 §1.6.0 roofline / §5.3 / §9；若备选胜出按 perf-feedback.md 受控路由。
  **shape 特化可行性**：本 kernel 经工厂按 `(num_tokens, numel, hidden_size, dtype)` 特化（源即工厂 + op 层 `lru_cache`），
  **可按 shape 分派主选/备选**（如大 H 走主选、小 H 走备选），达成总体最优。
  **设计期探针（D-3，已执行）**：见上「D-3 实测标定」——主选与备选均**已验证可编译、位精确、稳定**，且已给出性能差。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1 算法（UB 中转 + 标量 GM 读动态 slice）。

**优化结论：采纳 0 项——本单元为纯行搬运（FLOPs = 0），无数学等价优化空间；未采纳项 = C3「读一次散写」**
（虽数学等价且可削减 −44% 流量，但需 `fwd_idx`，不在本单元输入接口，已在 §1.6.0 作候选弃选，登记风险 R-4）。
**等价性机器验证**：因无采纳项，按 Phase 2 第 5 步「显式『无优化空间』结论」豁免 `verify_equiv.py`；
等价性由 **D-3 探针设备数值对照**承担（`perm_h` 与 `torch.gather(hidden_states,0,pidx)` 逐位精确，max_diff=0，6 次稳定）。
依据（逐项）：

1. **无算术可化简**：语义式为纯行搬运 $perm\_h[s,:]=hidden\_states[pidx[s],:]$，**FLOPs = 0**——不存在除法/倒数、
   sqrt/rsqrt、log/exp、稳定化 max 减法等可降代价的恒等变形（对照 §1.6.2 表格中「原式→优化后」的典型类别，本算子全部为空集）。
2. **无公共子表达式**：每输出行只读一个输入行、只写一个输出行，无重复子式可复用；`permuted_idx[s]` 每槽位读一次即用。
3. **无算子降代换 / 访存削减空间**：dtype 两侧一致（**无 cast**），行内无中间结果（两段 DMA 是搬运不是计算），
   UB 驻留已是「一行」，无需再削减。
4. **唯一存在的「流量削减」候选（C3 读一次散写）** 是**数学等价的**（输出逐位一致）且能把流量从 939.5MB 降到 528.5MB，
   但它**需要 `fwd_idx` 反映射**——**不在本单元的输入接口**（本单元输入仅 `hidden_states` + `permuted_idx`），且核内求逆
   需 O(numel) 且引入「向量产出→标量消费」风险（CG-2026-0027）。故**在 §1.6.0 作为候选弃选**（接口层），
   不作为 §1.6.1 采纳项；其潜在收益（−44% 流量）登记为风险 R-4，供上游接口演进参考。
5. **代码形态级调整（slice 形态、动态源行）不是数学优化**：它们是 §0.6 R1/R2 的**可行性形态**，语义恒等性由 D-3 探针
   直接数值验证（max_diff=0），不属于公式级化简。

**优化后公式（供 §3.1 拆解）**：与 §1.3 公式**完全一致**（无公式级改动）；
实现映射见 §3.1/§3.2（两段 DMA + 标量 GM 读）。
**等价性机器验证（D-1）**：因**无采纳优化项**，按 skill Phase 2 第 5 步「显式『无优化空间』结论除外」**不强制生成
`verify_equiv.py`**；本单元改以 **D-3 探针的设备数值对照**承担等价性证据（`perm_h` 与 `torch.gather(hidden_states,0,pidx)`
**逐位精确**，bf16/fp16 × 多 shape，max_diff=0，6 次稳定——见 `history_version/design_probe_gather.py`）。
**量化影响评估**：无算术 → **不引入任何 fp16/bf16 舍入误差**；精度标准为「输出逐位精确一致」（§8.2）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案全部循环与标量计算点（对象 = §1.6.0 选定、经 §0.6 重设计落地的 C1 算法）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 行内元素搬运（H 个元素） | 源 thread 级 128-bit 标量循环 | **整行 `T.copy` DMA**（元素轴连续、DMA 自动向量化） | ✅ | — |
| 2 | 输出槽位循环（`spc` 个槽位） | 源 thread/block 并行 | 无法整段向量化（每槽位源行不同）；**改为核内 `T.serial(spc)`**，但**行内元素已全向量化**（每槽位 1 条整行 DMA，非逐元素） | ❌ | **所需向量 API 在本项目确不存在**：行 gather 无向量原语——`T.gather` 仅**尾维**（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3「不提供额外 `dim` 参数」）；跨步 gather 向量指令缺失（`TRAP-C10-strided-gather`）。且 `hidden_states` 全量（prefill 58.7MB）不可能驻 UB（192KB）。**非逐元素热点**（1 条 DMA/行） |
| 3 | 源行号读取 `src = permuted_idx[slot]` | 源向量化索引 load | 可先 `T.copy` 一段 `permuted_idx` 入 UB 再向量读——**但标量消费该 UB 值命中 CG-2026-0027**（向量产出→标量消费不可同步） | ❌ | **CG-2026-0027**：标量读由向量 op 写入 UB 的值不同步（同产物多运行不一致，`T.pipe_barrier` 无效）。**必须标量 GM 读**（跨 kernel，`permuted_idx` 由 scan 写入 GM）。**非逐元素热点**（每 7168 元素 1 次 4B 标量读） |
| 4 | 尾块守卫 `if slot < numel` | 源标量比较守卫 | 无向量等价（控制流谓词） | ❌ | **动态边界**：`numel` 可能非核数整数倍（如 numel=10）；为 tile/slot 级控制流谓词，非逐元素计算；D-3 探针实测该标量 `if` 形态可编译且正确（对照：CG-2026-0028(c) 禁的是**标量 `T.if_then_else` 表达式**，非 `if` 语句） |
| 5 | 核内任务起点 `slot = core_id*spc + s` | — | 无向量等价 | ❌ | **block 级索引计算**（每核 `spc` 次整数算术），非逐元素热点 |

**有向量形态但弃选者**：
- C4「尾维 gather 转置」：VGather 原语存在（`testing/npuir/indexing_ops/test_slice_gather_dev.py` 实测可用）但 `src` 须驻 UB
  且沿尾维——`(H, T)` 需 58.7MB（prefill）≫ 192KB，分块则需巨型转置链（`T.transpose` 仅两轴交换、UB 级；`layout.md`
  PL-1.1-transpose-chain 实测 ~5.4µs 但数据量远超 UB）。弃选理由是**容量 + 转置代价**，非「API 不存在」（§1.6.3 L3）。
- C3「读一次散写」：可向量化但需 `fwd_idx`（接口外，§1.6.0）。

**向量化结论**：**行内元素已全部向量化**（第 1 项：整行 `T.copy` DMA，无逐元素标量循环）；数据通路是
「DMA 向量搬运」，本单元**不存在逐元素标量热点**。仅保留 4 处标量点（第 2/3/4/5 项：槽位串行循环、标量 GM 行号读、
尾块守卫、核起点算术），理由类别：**无行 gather 向量原语（附文档/陷阱佐证）**、**标量消费向量产出不可同步
（CG-2026-0027）**、**动态边界控制流**、**block 级索引计算**；均**非逐元素热点**（每行 1 次，相对每行 7168 元素的
1 条 DMA 可忽略）。**无可向量化而未向量化的逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本单元 I/O layout 为**契约**（`hidden_states (T,H)` 行主序、`permuted_idx (numel,)`、`perm_h (numel,H)` 行主序）；
> 核内布局与 lane 映射是**自由变量**。本算子非 batch 维有 2 个：**槽位轴 numel**（gather 的索引轴）与
> **元素轴 H**（行内连续轴）。逐元素类 / 索引类的候选枚举：连续轴选择 × 索引轴向量化方式 × 批量粒度。

**轴质量评分**（两个候选向量化轴）：

| 评分项 | 槽位轴 numel（索引轴） | 元素轴 H（行内连续轴） |
|--------|----------------------|----------------------|
| 整除性 / 尾 lane 浪费 | 非向量轴；`spc` 静态、尾块由守卫处理 | H·2 为 32B 倍数（H=7168→14336B）；DMA 无尾 lane 问题（H=64/128/3072/7168 全对齐） |
| 累加链形态 | 无（逐槽位独立） | 无（纯搬运，无累加） |
| repack 代价 | 0（行主序直读） | 0（行内连续，无需转置/NZ） |
| UB 容量影响 | 无 | `ub_row` = H·2（14KB @H=7168） |

**候选矩阵**（gather 类：连续批量 gather 粒度 × 向量轴对齐 × 索引 vs 分段拷贝）：

| # | 方案 | 关键决策 | 代价 / 收益 | 采纳 |
|---|------|---------|------------|------|
| L1 | **槽位串行 + 元素轴整行 DMA**（主选 C1） | 向量化轴 = **元素轴 H**（整行一条 DMA）；索引轴串行（标量 GM 读控制动态源行） | DMA 数 = 2·numel（读+写各 numel）；**实测 719.8µs = 1.31 TB/s（≥ 地板）** | ✅ **主选** |
| L2 | **批量写（R 行攒批）**（备选 C2） | 写侧合并：`ub_batch((R,H))` 逐行填入，一条连续 UB→GM 写 R 行 | DMA 数 = numel + numel/R；UB R·H·2（R=4→57KB）；**实测 +3%**；含非零起点 UB dst（历史 `TRAP-UB-dst-align`，本工具链探针未触发） | ✅ **备选**（D-3 三件套） |
| L3 | **尾维 gather 转置**（C4） | 把 token 轴搬到尾维、`T.gather` 一次取全体 | `(H,T)`=58.7MB ≫ UB；分块则引入巨型转置链（`T.transpose` 二轴交换、UB 级）+ 多段；VGather 索引须 int32 且与 numel 等长 | ❌ 弃选（容量 + 转置代价；VGather 本身于 `testing/npuir/indexing_ops/test_slice_gather_dev.py` 可用——**非 API 缺失**） |
| L4 | **选择矩阵 GEMM**（C5） | 把 gather 变成 Cube 矩阵乘 | FLOPs 1.9e12 + `P` 物化 268MB；纯搬运退化为巨量 MAC | ❌ 弃选（复杂度爆炸） |
| L5 | **读一次散写**（C3） | 遍历方向反转（按源 token 写 K 目标） | 流量 939.5→528.5MB（−44%，达 roofline）；**需 `fwd_idx`（接口外）** | ❌ 弃选（接口；登记风险 R-4） |
| L6 | **host 侧 gather** | CPU 索引拷贝 | 设备↔host 往返 + CPU 同步；带宽劣化 | ❌ 弃选（违反零 CPU 同步） |

**布局决策结论**：选定 **L1（槽位串行 + 元素轴整行 DMA，即 C1）**——向量化轴 = **元素轴 H**（每行一条 32B 对齐的整行
DMA，无逐元素标量、无 repack/NZ）；索引轴（槽位）串行，由**标量 GM 读** `permuted_idx[slot]` 驱动动态源行 slice。
索引缓冲驻 **UB（`ub_row (1,H)`）**；无核内重排（repack = 0）。该结论已同步 §3.3 主选伪代码、§4.3/§4.5（`ub_row` 形状与
预算）、§6（循环结构：核内 `T.serial` 外层 + 整行 `T.copy` 内层）。L2–L6 的弃选理由均已附 `docs/` / `testing/` /
`examples/` 路径或 pattern-library 条目，符合 negative-claim-evidence.md §1。
**实验裁决三件套（L1 vs L2）**：主选/备选/裁决计划见 §1.6.0 末「实验裁决计划」——主选 = L1（0 起点 UB dst 最简形态）；
备选 = L2（R=4 批量写）；判定阈值 ε=0.05（实测仅 +3% → **默认保持主选**）；回写路径 §1.6.0 roofline / §5.3 / perf-feedback.md。
**无「仅一个候选轴」豁免**：本算子两个候选向量化轴（numel / H）与两种批量粒度（per-slot / R-batch）均已评估。

> **与源并行轴的独立评估（迁移任务要求）**：源 GPU 的并行轴选择（block=8 槽位、thread=行内 8 元素）是**输入而非结论**——
> GPU 的 thread 轴**不对应** NPU 的向量轴。本设计**独立评估**后把「行内元素」映射到 **DMA 整行**（非 lane）、把「槽位」
> 映射到**核内串行**（非 block/thread 超发），与源的 block/thread 划分不一致（§0.6 R3）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（单一模式）。本单元为**纯 Vector / MTE2 搬运 kernel**（`T.alloc_ub` + `T.copy` 动态 slice +
核内 `T.serial`），**不含 `T.gemm`、不含 Cube scope、不含跨引擎交接**，故 Developer 形态完全可用。
用户指定 `programming_mode=developer`，与本结论一致；**无需 Expert 回退**。

### 2.2 选型理由

- 本单元数据通路为 **GM→UB→GM 的行搬运**（无浮点算术、无矩阵）——按 AGENTS.md 的 Developer-mode MixCV 触发规则
  （Cube `T.gemm` + 至少一个 v-prefix op），**本单元不满足触发判据**（无 `T.gemm`、无 v-prefix 算术），归为「纯搬运 Vector」，
  走 Developer。
- **规避三条工具链阻断**：Developer persistent + `T.gemm` + 整型索引段共存即崩（CG-2026-0010）——本单元**无 `T.gemm`**，
  不触发；跨引擎同 block 交接不可编译（CG-2026-0026）——本单元**无跨引擎交接**；单 scope 内标量消费向量产出不可同步
  （CG-2026-0027）——本单元的标量只读 **GM**（跨 kernel 的 `permuted_idx`），不读本 scope 向量产出的 UB，不触发。
- Developer 用 `alloc_ub` + `T.copy`（动态 slice）+ 核内 `T.serial` 表达，实现负担低；**D-3 探针已在 Developer 下设备实测通过**。

### 2.3 模式影响

| 维度 | Developer（选定） |
|------|------------------|
| 内存分配 | `T.alloc_ub`（`ub_row (1,H)`）；GM：输入 `hidden_states`/`permuted_idx`、输出 `perm_h` |
| 搬运原语 | `T.copy`（GM→UB / UB→GM，动态 slice 起始） |
| 扫描原语 | 无（无 scan / 无归约） |
| 跨 scope 交接 | **无**（单 Vector scope；`permuted_idx` 经 **GM** + kernel 间同 stream 顺序） |
| 同步 | scope 内编译器自动同步；**不依赖 auto-sync 跨「向量生产 → 标量消费」**（该形态已弃用，CG-2026-0027）；kernel 间由 stream 顺序保证 |
| 分核 | 固定物理核数（纯 Vector → 40）+ 核内 `T.serial`（§5.5） |

---
## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的**优化后公式**（与 §1.3 完全一致，无公式级改动）。逐步骤映射到 §1.6.2 的向量化结论：
> 行内元素由整行 DMA 承接（无 v-prefix 算术算子；本单元无浮点）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| P1 | `slot = core_id·spc + s`，`s ∈ [0,spc)` | 每核 `spc` 个输出槽位（核内串行） |
| P2 | `src = permuted_idx[slot]` | 源 token 行号（标量 GM 读，驱动动态 slice） |
| P3 | `ub_row ← hidden_states[src, 0:H]` | 整行 GM→UB DMA（动态源行） |
| P4 | `perm_h[slot, 0:H] ← ub_row` | 整行 UB→GM DMA（动态目的行） |

`perm_h[s,:] = hidden_states[permuted_idx[s],:]` 的右式即 P3，左式即 P4；P1/P2 提供索引。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| P1 | 核与槽位 | `with T.Kernel(nk, is_npu=True) as (core_id, _subid):` + `for s in T.serial(spc):` | `nk=min(numel,40)`、`spc=ceildiv(numel,nk)`（均编译期常量） | Developer |
| P2 | 取源行号 | `src = permuted_idx[slot]` | 标量 **GM** 读（非 UB 向量产出；规避 CG-2026-0027） | Developer |
| P3 | 整行取源 | `T.copy(hidden_states[src : src + 1, 0:H], ub_row[0:1, 0:H])` | 动态源 slice（静态 extent 1）；UB 0 起点 dst | Developer |
| P4 | 整行写目标 | `T.copy(ub_row[0:1, 0:H], perm_h[slot : slot + 1, 0:H])` | 动态目的 slice（静态 extent 1） | Developer |
| 守卫 | 尾块 | `if slot < numel:` | 标量 `if` 语句（**非** `T.if_then_else` 表达式，避免 CG-2026-0028(c)） | Developer |

> 说明：本单元**无浮点逐元素运算**，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix API；行内向量化由 **MTE2 整行 DMA**
> 承担（§1.6.2 #1）。

### 3.3 计算伪代码

> `T.Tensor` 第二位置参数直接传 dtype。`nk`/`spc`/`num_tokens`/`numel`/`hidden_size` 均为编译期常量（工厂 + op 层特化）。
> **主选（C1）与备选（C2）伪代码均与本仓 D-3 探针同构**（`history_version/design_probe_gather.py` 已设备实测通过）。
> 探针为便利用 `out_idx=[2]`（设备侧分配 `perm_h`）；**交付形态建议 `out_idx=[]`**（`perm_h` 由调用方分配、与源一致）——
> `out_idx` 只影响输出张量的分配方，不影响已实测的拷贝形态。

```python
# 编译期常量：num_tokens, numel, hidden_size, dtype
#   P = NPUUtils.get().get_aicore_num() * 2       # 纯 Vector 物理核数（§5.5 实查；= 40）
#   nk = min(numel, P); spc = ceildiv(numel, nk)  # 核内 T.serial 边界（静态）

_PASS = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}

# ---- 主选 C1：槽位串行 + 元素轴整行 DMA ----
@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
def _make_gather_kernel_nopad(num_tokens, numel, hidden_size, dtype):
    nk = min(numel, 40)
    spc = (numel + nk - 1) // nk                      # 静态边界

    @T.prim_func
    def _gather_main(
        hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
        permuted_idx: T.Tensor([numel], "int32"),
        perm_h: T.Tensor([numel, hidden_size], dtype),
    ):
        with T.Kernel(nk, is_npu=True) as (core_id, _subid):
            ub_row = T.alloc_ub((1, hidden_size), dtype)
            for s in T.serial(spc):                   # 静态边界（core-split-strategy.md §1③）
                slot = core_id * spc + s
                if slot < numel:                      # 尾块守卫（源等价）
                    src = permuted_idx[slot]          # 标量 GM 读（非 UB 向量产出）
                    T.copy(hidden_states[src : src + 1, 0:hidden_size],
                           ub_row[0:1, 0:hidden_size])          # GM→UB（动态源行）
                    T.copy(ub_row[0:1, 0:hidden_size],
                           perm_h[slot : slot + 1, 0:hidden_size])  # UB→GM（动态目的行）

    return _gather_main


# ---- 备选 C2（R 行批量写；D-3 探针实测位精确 + 稳定，性能 +3%）----
@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
def _make_gather_kernel_nopad_batched(num_tokens, numel, hidden_size, dtype, R=4):
    nk = min(numel // R, 40)
    spc = ((numel + nk - 1) // nk + R - 1) // R * R   # 静态边界，取 R 的倍数

    @T.prim_func
    def _gather_main_b(
        hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
        permuted_idx: T.Tensor([numel], "int32"),
        perm_h: T.Tensor([numel, hidden_size], dtype),
    ):
        with T.Kernel(nk, is_npu=True) as (core_id, _subid):
            ub_batch = T.alloc_ub((R, hidden_size), dtype)
            for s in T.serial(spc // R):
                base = core_id * spc + s * R
                if base < numel:
                    for j in T.serial(R):
                        src = permuted_idx[base + j]
                        T.copy(hidden_states[src : src + 1, 0:hidden_size],
                               ub_batch[j : j + 1, 0:hidden_size])   # 非零起点 UB dst（见 §9 R-3）
                    T.copy(ub_batch[0:R, 0:hidden_size],
                           perm_h[base : base + R, 0:hidden_size])    # 一条连续写 R 行

    return _gather_main_b
```

> **注 1（规避 CG-2026-0027）**：标量 `src` 只读 **GM `permuted_idx`**（由 scan 单元在**另一 kernel** 写入），
> **不读本 scope 内向量 op 写入的 UB 值**——无「向量产出 → 标量消费」数据流。
> **注 2（规避 CG-2026-0028）**：无 `T.alloc_local`、无 int32 `>>`、无标量 `T.if_then_else` 表达式；
> 尾块用标量 `if` 语句（D-3 探针实测可用）。
> **注 3（slice 形态）**：统一用 `src:src+1` 显式 slice（而非 `base+size`）——`TRAP-load-nd2nz-strided` /
> `TRAP-T-copy-region-semantics` 均指向 **slice 形态**为跨步/动态区域的正确形态。
> **注 4（整行对齐）**：`hidden_size · itemsize` 须为 **32B 倍数**（否则整行起点非对齐）；manifest 全部 workload
> （H=64/128/3072/7168，fp16/bf16）均满足 → 见 §5.3 / §9 R-5。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.copy(src_slice, dst_slice)`（GM↔UB，动态 slice 起始） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1（切片语法）/ §2.4 示例1-2（动态 `bx`/`tile_M`）；本仓先例 `examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` L161-166（动态源行 `IndexK[cur_idx:cur_idx+1, :]`） | **已验证**（D-3 探针设备实测位精确 + 6 次稳定，bf16/fp16 × 多 shape） |
| 标量 GM 读驱动动态 slice 起始 | 胞弟 `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` §1.6.2 #6（标量 GM 读驱动动态 extent，已实测）+ 本单元 D-3 探针（首次用于驱动**读起点**） | **已验证**（D-3 探针） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | 已验证 |
| `T.Kernel(nk, is_npu=True)`（1D 核） | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；`examples/deepseek_v32/fp8_lighting_indexer.py` | 已验证 |
| `T.serial(spc)`（静态边界核内串行） | `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`（核内静态串行）；core-split-strategy.md §1③ | 已验证 |
| 标量 `if slot < numel:` 守卫 | D-3 探针（`numel=10` 尾块用例通过）；胞弟 D-3 探针 `if e < E` | 已验证 |
| `@tilelang.jit(out_idx=[], target="npuir", pass_configs=...)` | 源 L238 `@tilelang.jit(out_idx=[], ...)`；`_PASS` 同胞弟探针 | 已验证（探针同款 pass_configs） |
| `T.gather(src, dst, indices, [m,n])`（尾维） | `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3（仅尾维）；`testing/npuir/indexing_ops/test_slice_gather_dev.py` | **可用但不适用**（仅尾维 + `src` 须驻 UB；本设计不依赖，§1.6.0 C4 弃选） |
| ~~GM→GM `T.copy`~~ | `docs/Tilelang.language/内存操作/T.copy.md` §2.3-5 | **不可用**（Developer 编译期报错）；本设计用 UB 中转 |
| ~~`T.Kernel(threads=...)` 线程控制~~ | pattern-library `TRAP-threads-kwarg-noop` | **无效**（npuir 下 `threads=` 无效果）；本设计不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（1D `T.Kernel`） | 一维核 |
| GEMM M/N 需为 block 整数倍 | 否（无矩阵） | — |
| L0C 容量（128KB） | 否（无 L0C） | — |
| 物理核数限制 | 是 | `nk = min(numel, 40)`（纯 Vector 实查 ×2）；核内 `T.serial(spc)`（§5.5） |
| host 侧禁止改动输入 | 是（源 host 无预处理；`perm_h` 由调用方分配） | 本单元无 host 预处理 |
| 一维核超发被串行调度 | 是（源 `ceil(numel/8)` 超发） | 固定核数 + 核内串行（§0.6 R3、§5.5） |
| **Developer 禁 GM→GM** | 是（源为直接 GM→GM） | **UB 中转两段 DMA**（§0.6 R1）；`docs/Tilelang.language/内存操作/T.copy.md` §2.3-5 |
| **单 scope「向量产出→标量消费」不可同步（CG-2026-0027）** | 是（若内联进 scan） | **kernel 边界隔离**：`permuted_idx` 落 GM、gather 标量只读 GM（§0.6 R4） |
| int32 `>>` / `T.alloc_local`+`serial` / 标量 `T.if_then_else`（CG-2026-0028） | 否 | 全部回避（无 `>>`、无 `alloc_local`、尾块用 `if` 语句） |
| 跨引擎同 block 交接不可编译（CG-2026-0026） | 否（无跨引擎交接） | 不涉及 |
| 整行 32B 对齐 | 是 | 要求 `hidden_size·itemsize % 32 == 0`（manifest workload 全满足）；§9 R-5 |
| `T.copy` 区域语义 | 是 | src/dst **双显式 slice**；extent 两侧一致（1 行） |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| 数据通路 | 直接 GM→GM 一条拷贝 | **UB 中转两段 DMA** | §0.6 R1（Developer 禁 GM→GM） |
| 源行寻址 | 向量化索引 load `hidden_states[pidx[slot]]` | **标量 GM 读 → 动态 slice 起始** | §0.6 R2（无行 gather 向量原语） |
| 行内向量化 | thread 级 128-bit（`uint4`，8×bf16） | **整行 `T.copy` DMA** | §0.6 R3（NPU 无 thread-per-element） |
| 并行结构 | 超发 grid `ceil(numel/8)` + thread | **固定物理核数（纯 Vector → 40）+ 核内 `T.serial`** | §0.6 R3（超发被串行调度） |
| 承载形态 | 独立 Phase 2 kernel | **保留独立 kernel**（强化为 CG-2026-0027 的必要条件） | §0.6 R4 |
| 输出分配 | `out_idx=[]`（调用方分配） | 保留 `out_idx=[]`（`perm_h` 由调用方分配） | 与源一致 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` | **极高（行 gather 直接先例）** | L157-167：`for i in T.serial(block_I): cur_idx = idx_frag[i]; T.copy(IndexK[cur_idx:cur_idx+1, 0:index_dim], k_shared[i:i+1, 0:index_dim])`——「gather via idx loop with explicit serial reads」；2-D slice 保 rank 一致（R-KA-8 教训）。注：该例为 Expert 模式 + 其索引来自 fragment；本单元改用 **Developer + 标量 GM 读**（D-3 探针已证） |
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | **极高（同族工具链陷阱规避 + 分核范式）** | §0.5（CG-2026-0027/0028 规避）、§0.6 R3/R4（固定核数 + 核内 `T.serial` 静态边界 + 标量 GM 读驱动 DMA）、§5.5（纯 Vector 核数 ×2）、§3.4（`T.gather` 尾维/签名冲突登记） |
| `testing/npuir/indexing_ops/test_slice_gather_dev.py` | 高（`T.gather` 能力边界佐证） | 证明 VGather 可用但**仅尾维、`src` 须驻 UB**（本题 58.7MB 不可驻）——支撑 §1.6.0 C4 弃选 |
| `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 中（核内静态串行范式） | `T.serial(num_local_tasks)` 静态边界核内串行 |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 中（Developer 非 persistent 网格范式） | Developer 模式下单 Vector scope + `T.Kernel(n, is_npu=True)` |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `hidden_states` | `(num_tokens, hidden_size)` | fp16 / bf16 | 输入激活（行 = token） |
| `permuted_idx` | `(numel,)` = `(T*K,)` | int32 | 源行号（scan 单元产出；调用方零初始化） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `perm_h` | `(numel, hidden_size)` | fp16 / bf16 | tight 专家连续区（gather 结果）；`out_idx=[]` 由调用方分配（与源一致） |

> **对接契约（胞弟 `_make_scan_kernel_nopad` 产出 / 本单元消费，§0.6 R4）**：
> ① `permuted_idx` **int32**、shape `(numel,)`、值域 `[0, num_tokens)`；② **scan → gather 同 stream 顺序 launch**；
> ③ `permuted_idx` 零初始化保证 EP dummy-row-0 语义；④ 无 in-block 同步、无 CPU 同步。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `ub_row` | `(1, hidden_size)` | fp16 / bf16 | UB | 主选 C1：单行中转（GM→UB→GM） |
| `ub_batch` | `(R, hidden_size)` | fp16 / bf16 | UB | 备选 C2：R 行攒批（仅备选使用） |

### 4.4 内存搬运路径

```
[scan 单元] ──同 stream 顺序──▶ GM[permuted_idx(numel) int32]
                                        │ 标量 GM 读 src = permuted_idx[slot]
                                        ▼
GM[hidden_states(T,H)] ──T.copy 动态源行[src:src+1,0:H]──▶ UB[ub_row(1,H)]
                                                                     │ T.copy 动态目的行[slot:slot+1,0:H]
                                                                     ▼
                                             GM[perm_h(numel,H) fp16/bf16]
```

> **层级说明**：本单元数据路径只有 **GM → UB → GM**（**无 L1 / L0A / L0B / L0C**；无 Cube 矩阵计算）。
> 每核处理 `spc` 个（主选）或 `spc/R` 批（备选）槽位；核间无通信。

### 4.5 UB 内存预算（Developer，代表 shape H=7168，fp16/bf16 = 2B）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| **主选 C1** | | | |
| `ub_row` | (1, 7168) | bf16 | 14 336 |
| C1 小计 | | | ≈ **14.0 KB** |
| **备选 C2（R=4）** | | | |
| `ub_batch` | (4, 7168) | bf16 | 57 344 |
| C2 小计 | | | ≈ **56.0 KB** |

> 主选 `ub_row` = H·2 字节（H=7168→14KB；H=3072→6KB；H=128→256B；H=64→128B），均 ≪ UB 上限
> （`CONST-capacity-910B2C`：192KB，含 ~1.7× 膨胀仍充裕；两处 buffer 均为单版本，`pass_configs` 关闭 auto-multi-buffer）。
> 备选 `ub_batch` 57KB（含膨胀 ~97KB）仍在限内。

### 4.6 动态轴定义

**无动态 shape 轴**：`num_tokens / numel / hidden_size / nk / spc` 均为编译期常量（工厂按 shape 特化，源亦为工厂 +
`lru_cache`）。运行时变化的仅为 `permuted_idx` 的**数据值**（不改变 shape），由标量 GM 读驱动动态 slice 处理。
（`numel` 非 8 倍数的尾块是**编译期已知**的边界，由 `if slot < numel` 守卫处理，非动态轴。）

### 4.7 JIT 配置

```python
_PASS = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}
# @tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
# 源编译配置对比：compile_flags=["-O3", "-DENABLE_BF16"]（GPU）；NPU 侧无 fast-math/bf16 flag 需求，
#   dtype 由 T.Tensor 直接指定。
```

> `_PASS` 与 D-3 探针 / 胞弟探针同款：保证 UB 单版本、避免 buffer 重排干扰动态 slice 拷贝（实测该配置下位精确且稳定）。
> 按 `(num_tokens, numel, hidden_size, dtype)` 经工厂 + op 层 `lru_cache` 特化（与源一致）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector / MTE2 行搬运（gather）**（无算术、无矩阵、无 spatial tiling）。判定依据：全部计算为
「整行 GM↔UB 搬运」——落 Vector（AIV）/ MTE2 单元。本单元**无 `T.gemm`、无 Cube scope**。

### 5.2 Block 划分

本单元**无空间 block 划分**（无 M/N/K tile）。工作粒度：
- **1 个输出行 = 1 个工作项**（一次整行 DMA，元素轴 H 不分块——整行搬运）；
- 每核处理 `spc` 个槽位（核内 `T.serial`）；核间按**连续槽位区间**划分（`[core_id·spc, (core_id+1)·spc)`）。

### 5.3 约束分析

- **对齐约束**：整行起点 `src·H·itemsize` 须 32B 对齐 → 要求 `H·itemsize % 32 == 0`（fp16/bf16 即 `H % 16 == 0`）。
  manifest workload H ∈ {64, 128, 3072, 7168} 全部满足（§9 R-5）。
- **UB 容量**：主选 14KB（H=7168）≪ 192KB ✓；备选 57KB ✓。
- **非整除**：`numel` 可能非核数整数倍（如 numel=10）→ `if slot < numel` 守卫（源等价）；无 GEMM 非整除问题。
- **静态边界**：`nk`、`spc` 均为编译期常量 ✓（core-split-strategy.md §1 要素③要求）。
- **带宽**：实测 1.31 TB/s ≥ MTE2 混合流量地板 ~1.26 TB/s（§1.6.0 roofline）——已在 roofline 上。

### 5.4 注意事项（尾块/边界处理）

- **尾块**：`numel % nk != 0` 时末核的 `[slot ≥ numel]` 由 `if slot < numel` 守卫跳过（源 L250 等价）；
  `numel < nk` 时 `nk = numel`、`spc = 1`（探测覆盖 numel=8/10）。
- **`numel = 0`**：不启动有效工作（`nk=0` 非法 → 工厂须保证 `numel ≥ 1`；调用方契约 `numel = T*K ≥ 1`）。
- **EP dummy 行**：`permuted_idx` 未写槽位为 0 → `perm_h` 对应行 = 源行 0 副本（源语义，无需特判）。
- **`hidden_size` 对齐**：非 32B 倍数时不支持整行 DMA——见 §9 R-5（manifest 无此情况）。
- **kernel 顺序**：scan 写 `permuted_idx` → gather 读 `permuted_idx` 写 `perm_h`，必须**同一 stream 顺序 launch**（§7）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **返回值 = 20**。
  查询代码与实测值：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20`
  （Ascend910B2C，2026-10-14 本机实查）。
  **本单元为纯 Vector / MTE2 搬运单元** → 物理核数取实查值翻倍 **`P = get_aicore_num() * 2 = 40`**
  （标准 core-split-strategy.md §1②「纯 Vector 算子核数翻倍」）。
- **逻辑核数计算**：`num_logical_kernels = numel`（每输出行 = 1 工作项；`numel = T*K`）。
  代表值：smoke（T=4,K=2）→ 8；partial-tail（T=5,K=2）→ 10；kimi-k2/deepseek-v3-prefill（T=4096,K=8）→ **32768**；
  qwen3-30b-prefill（T=4096,K=8）→ **32768**。
- **规模判定与分核方案**：
  - **`numel ≤ P`（小规模，如 numel=8/10）**：**无需适配**——`nk = numel`、`spc = 1`（一核一槽位，其余核不启动）。
  - **`numel > P`（极大规模，prefill numel=32768）**：**无法通过分块缩减核数**（工作项就是输出行；减小块即减小并行度）→
    **固定启动核数 = 物理核数 40 + 核内 `T.serial` 串行**：`nk = min(numel, P) = 40`、`spc = ceildiv(numel, nk)`
    （**编译期常量，静态循环边界**），核内 `for s in T.serial(spc): slot = core_id*spc + s` 串行处理多个槽位，
    摊薄核启动开销。
  - 代表值：prefill `nk=40`、`spc = ceildiv(32768,40) = 820`；smoke `nk=8`、`spc=1`。
  - **负载均衡**：每核槽位数差 ≤1（连续区间划分）；每槽位工作量恒定（1 行 DMA）→ 负载天然均衡（数据分布无关）。
- **备注（D-3 实测）**：`nk=40`、`spc=820` 的 prefill 形态实测 719.8µs @ 1.31 TB/s，证明固定核数 + 核内串行已饱和带宽。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级 | 固定物理核数网格 | `T.Kernel(nk, is_npu=True)`，`nk = min(numel, 40)` | 分核策略（§5.5） |
| 核内槽位 | 串行遍历 | `T.serial(spc)`（静态边界） | 每槽位源行不同（无行 gather 向量原语），slot 摊到核内串行（§0.6 R3） |
| 行内元素 | 向量化（DMA） | `T.copy`（整行 H） | 元素轴连续，MTE2 整行搬运（§1.6.2 #1） |
| 元素级标量 | 无（已消除） | — | 无逐元素标量循环（§1.6.2） |
| T.Parallel / T.Pipelined / T.Persistent | **未使用** | — | 无规约 / 无 spatial 分块 / 非持久化 |

### 6.2 循环伪代码

见 §3.3（主选：核级 `T.Kernel(nk)` → 核内 `T.serial(spc)` → 每槽位两次整行 `T.copy`）。

### 6.3 流水线优化

- **核内槽位循环不显式 `T.Pipelined`**：相邻槽位之间**无数据依赖**（各写各的输出行），是否流水由编译器自动调度；
  `pass_configs` 关闭 auto-multi-buffer 以保证 UB 单版本（与探针一致）。
- **可优化空间已很小**：实测已达带宽地板（§1.6.0 roofline），双缓冲重叠读/写（MTE2 与 MTE3 重叠）理论上可把
  读+写从串行变为重叠，但读写共用同一 MTE2 通路与 HBM 带宽，收益受限；作为 Stage 4 调优项登记（§9 R-6）。
- **跨 kernel**：scan → gather 同 stream 顺序执行（无显式同步原语）。

### 6.4 尾块处理

- 尾块：末核 `if slot < numel` 守卫（源等价）。
- 空工作：`numel < nk` 时 `nk=numel`，无空核。
- 任意 `numel`：见 §5.4；不使用 host padding、不使用逐元素守卫。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**scope 内编译器自动同步 + kernel 间 stream 顺序**（Developer）。
- 本单元**单 Vector scope**，scope 内 MTE（`T.copy`）→ MTE 依赖（同 `ub_row` 的写后读）由编译器自动同步；
- **关键约束（CG-2026-0027）**：**不在标量上下文中读取本 scope 内向量 op 写入的 UB 值**——本设计已彻底避免
  （标量只读 **GM** `permuted_idx`，其值由 scan 写入）；
- **不使用任何手动同步原语**（不用 `T.sync_block_set/wait`、不用 `T.set_flag/wait_flag`、不依赖 `T.pipe_barrier`）。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| 槽位内 GM→UB → UB→GM（同 `ub_row`） | 编译器自动同步 | 单 scope 内写后读（纯 MTE/向量，无标量消费） |
| 跨槽位复用 `ub_row`（loop-carried） | 编译器自动同步 | 单 UB 缓冲的串行复用（读写序由 auto-sync 保证；探针 6 次稳定） |
| scan → gather（`permuted_idx` / `perm_h`） | **kernel 间同 stream 顺序**（无显式同步原语） | scan 写 `permuted_idx` 完成、gather 再读；**标量 GM 读不跨「向量生产 → 标量消费」同 scope 边界** |

> **不引入 `T.sync_block_*` 的依据**：CG-2026-0026（`syn_instr_mode` codegen 不匹配，官方样例复现）。本设计以
> 「`permuted_idx` 落 GM + kernel 间顺序」替代同 block 交接，**完全规避**。

### 7.3 pass_configs 配置

```python
pass_configs = {
    NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}
```

> 与 D-3 探针一致：保证 UB 单版本、避免 buffer 重排干扰动态 slice 拷贝（实测该配置下位精确且稳定）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现，**不得复刻** §0.6 的 NPU 算法（UB 中转 + 标量 GM 读）。
> 独立实现 = PyTorch 行 gather（高级索引 / `torch.gather`），**不经过** UB 中转 / 动态 slice / 分核串行。
> 理想对照还应与 op 级 counting-sort 参考（`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` 的
> `_ref_moe_permute_nopad`）交叉验证 `permuted_idx` 语义。

```python
import torch

def golden_gather(hidden_states, permuted_idx):
    """§0.1 语义：perm_h[s, :] = hidden_states[permuted_idx[s], :]。

    与源实现同语义的纯 PyTorch 行 gather（独立于 NPU 算法：无 UB 中转、无动态 slice、无分核）。
    dtype 透传（fp16/bf16 逐位一致）。
    """
    idx = permuted_idx.long().view(-1, 1)                     # (numel, 1)
    return torch.gather(hidden_states, 0, idx.expand(-1, hidden_states.shape[1]))
```

### 8.2 精度标准

| dtype | 标准 |
|-------|------|
| fp16 / bf16 | **逐位精确一致**（纯搬运无算术 → 无舍入；atol=0、rtol=0）。D-3 探针 max_diff=0 已证 |

> 注：本单元无浮点运算，故不存在 `TRAP-fp16-opmath-golden` 类 opmath 域分歧。

### 8.3 L0 门槛测试计划

> **主线（交付形态）= C1 单 kernel 行 gather + 与 `golden_gather` 逐位精确比对**。
> D-3 探针已预验 smoke / numel10 / H=128 / 批量 四类（bf16+fp16，6 次稳定）。

| 用例 | 参数 | 验证点 |
|------|------|-------|
| L0-1 tiny-bf16 | T=4,K=2,E=4,H=64（numel=8） | 与 golden 逐位一致；`nk=8`、`spc=1` |
| L0-2 tiny-fp16 | T=4,K=2,E=4,H=64（numel=8），fp16 | dtype 透传逐位一致 |
| L0-3 small | T=16,K=2,E=8,H=128（numel=32） | 多核分槽位正确；逐位一致 |
| L0-4 partial-tail | **T=5,K=2,E=4,H=64（numel=10）** | **尾块 `if slot < numel` 守卫**（numel 非 ROWS=8 倍数）逐位一致 |
| L0-5 重复源行 | 构造 `permuted_idx` 含重复 token 行（K 个槽位共享同 token） | 重复 gather 正确（多对一） |
| L0-6 EP dummy 行 | `permuted_idx` 尾部若干槽位为 0（零初始化模拟 EP） | `perm_h` 对应行 = 源行 0 副本（源 dummy 语义） |
| L0-7 代表 prefill | T=4096,K=8,H=7168,numel=32768 | `nk=40`、`spc=820`；无死锁；逐位一致（D-3 探针已覆盖同形） |
| L0-8 多次运行稳定 | L0-1/L0-7 重复 6 次 | 结果稳定（无 CG-2026-0027 型未同步读） |
| L0-9 对接契约 | scan 单元产出的 `permuted_idx` 直灌本单元 | 两 kernel 同 stream 顺序；gather 读到正确 `permuted_idx`；逐位一致 |

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 源**超发 grid**（`ceil(numel/8)`）在 NPU 上被串行调度（core-split-strategy.md §1）→ 固定物理核数 + 核内串行。
- **Developer 禁 GM→GM**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3-5）→ 必须 UB 中转。
- **无行 gather 向量原语**（`T.gather` 仅尾维 `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3；
  `TRAP-C10-strided-gather`）→ 逐行动态偏移 DMA。
- **单 scope「向量产出 → 标量消费」不可同步**（CG-2026-0027）→ kernel 边界隔离，标量只读 GM。
- **int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` 使 codegen SIGSEGV**（CG-2026-0028）→ 全部回避。
- 跨引擎同 block 交接不可编译（CG-2026-0026）→ 不涉及。
- `T.Kernel(threads=)` 在 npuir 无效果（`TRAP-threads-kwarg-noop`）→ 不使用。
- **带宽已饱和**：实测 1.31 TB/s ≥ 地板 ~1.26 TB/s，继续结构改造边际极小（§1.6.0）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **GM→GM 直连拷贝** | 照源写 `T.copy(hidden_states[src,:], perm_h[slot,:])` | Developer 编译期报错「Unsupported memref to memref copy yet」 | UB 中转两段 DMA（§0.6 R1） |
| **gather 内联进 scan scope** | 为省一次 GM 往返把 gather 并入 scan | 标量消费向量产出的 `permuted_idx` → 同产物多运行不一致（CG-2026-0027） | 独立 kernel + `permuted_idx` 落 GM（§0.6 R4） |
| **用 `T.gather` 做行 gather** | 期望一次指令完成 | `T.gather` 仅尾维 + `src` 须驻 UB；行 gather 用不了 | 逐行动态 slice（§0.6 R2） |
| 试图先 `T.copy` `permuted_idx` 入 UB 再向量读作行号 | 「向量化」源行号 | 标量消费 UB 向量产出（CG-2026-0027） | 标量 **GM** 读（§1.6.2 #3） |
| `base+size` 形态用于动态/跨步区域 | dst 用 `size=[...]` | `TRAP-load-nd2nz-strided` / `TRAP-T-copy-region-semantics` 平坦误读/越界 | **src/dst 双显式 slice**（§3.3 注 3） |
| 整行 slice width 与 UB 容量不匹配 | `ub_row` 形状小于 H | `T.copy` extent 不匹配 → 越界写（`TRAP-T-copy-region-semantics`②） | `ub_row` 形状 = `(1, hidden_size)`（静态） |
| 尾块无守卫 | numel 非核数整数倍（numel=10） | 越界读/写 | `if slot < numel`（源等价，探针覆盖） |
| 超发 grid（`T.Kernel(numel)`） | 照源形态 | 核启动开销剧增（串行调度） | 固定核数 40 + 核内 `T.serial`（§5.5） |
| 备选 `ub_batch[j:j+1]` 非零起点 UB dst | 批量写形态 | 历史 `TRAP-UB-dst-align`（本工具链探针未触发，但需复核） | 主选 C1 无此形态；用 C2 时 Stage 3 首编复核（R-3） |
| scan→gather 异性 stream launch | 破坏同 stream 顺序 | gather 读 `permuted_idx` 竞态 | 严格同 stream 顺序 launch |

### 9.3 特殊场景处理

- **多对一/重复源行**：`permuted_idx` 中同一 token 行出现 K 次 → 每槽位独立 gather，结果正确（L0-5）。
- **EP dummy 行**：未写槽位 = 0 → 源行 0 副本（源语义，L0-6）。
- **极小 shape**（numel=8）：`nk=8 < 40`，正确性优先。
- **偏斜/任意排列**：分核按**连续槽位区间**划分，每槽位工作量恒定 → 与数据分布无关（L0-7）。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（已闭合，正向）**：**「标量 GM 读 → `T.copy` 动态源 slice 起始」设备实测可用**——D-3 探针
  （`history_version/design_probe_gather.py`）bf16/fp16 × 多 shape **max_diff=0 + 6 次稳定**；闭合迁移任务
  指定的关键未文档化假设（「T.copy 动态行偏移」）。
- **R-2（已闭合，正向）**：Developer 模式 GM→GM 禁令的替代路径（UB 中转两段 DMA）实测位精确（同上）。
- **R-3**：备选 C2 的 `ub_batch[j:j+1, 0:H]`（**非零起点 UB dst + 32B 倍宽度**）——历史 `TRAP-UB-dst-align` 形态；
  **本工具链（2026-10-14）D-3 探针实测未触发**（位精确 + 稳定），但该陷阱条目版本绑定（截至 2026-08-28）。
  若 Stage 3 首编命中，按 §1.6.0 裁决计划保持主选 C1（本单元无 UB 非零起点形态）。
- **R-4（接口，非本单元可解）**：**C3「读一次散写」可把流量从 939.5MB 降到 528.5MB（−44%，达 roofline）**，
  但需 `fwd_idx`（本单元输入接口外）。当前 C1 实测 719.8µs 已在 roofline 上（1.31 TB/s ≥ 地板），故 C3 的现实收益需
  在 **HBM 带宽未饱和**或 **K 更大**的场景才显著。**建议**：若上游接口可增加 `fwd_idx` 输入，Stage 4 可评估 C3 变体。
- **R-5（约束）**：整行 DMA 要求 `hidden_size·itemsize % 32 == 0`（fp16/bf16 → `H % 16 == 0`）；manifest workload
  全满足。若未来出现非对齐 H，须 host 侧 pad 或退回逐元素路径（当前不支持，登记）。
- **R-6（Stage 4 调优项）**：读/写重叠（双缓冲）理论上可提升，但读写共用 MTE2 与 HBM 带宽，实测已达带宽地板，
  预期收益有限；交 Stage 4 按 §1.6.0 裁决计划实测（若备选 C2 在部分 shape 更优则按 shape 分派）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_nopad/_make_gather_kernel_nopad/
├── DESIGN.md                    # 本设计文档
├── RETROSPECTIVE.md             # Stage 1 复盘（自进化钩子）
└── history_version/
    ├── design_probe_gather.py       # D-3 探针 #1：可行性/正确性/稳定性（变体 A/B）
    └── design_probe_gather_perf.py  # D-3 探针 #2：代表 shape 吞吐 A vs B
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核） |
| `history_version/design_probe_gather.py` | 已完成（D-3） | 变体 A/B 位精确 + 6 次稳定（bf16/fp16 × 多 shape） |
| `history_version/design_probe_gather_perf.py` | 已完成（D-3） | prefill 代表 shape：A=719.8µs、B(R=4)=696.5µs、B(R=8)=694.4µs |
| `verify_equiv.py` | **不产**（无数学等价采纳项，§1.6.1） | 本单元为纯搬运无算术 → Phase 2 第 5 步「无优化空间」豁免；等价性由 D-3 探针数值对照承担 |
| 算子实现（Stage 3） | 待实现 | 单 Vector gather kernel（§3.3 主选伪代码） |
| 集成对接（Stage 5） | 待实现 | 与胞弟 `_make_scan_kernel_nopad`（产 `permuted_idx`）同 stream 顺序 launch |

### 10.3 命名规范

- 项目目录：`examples/moe_permute_nopad/`
- 算子目录：`examples/moe_permute_nopad/_make_gather_kernel_nopad/`
- 规范文件名：`_make_gather_kernel_nopad.py`（Stage 3 产文件；含 `_make_gather_kernel_nopad` 工厂）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ D-3 可行性/性能探针
2. ⬜ Golden 函数（§8.1）
3. ⬜ 主选 kernel `_make_gather_kernel_nopad`（§3.3 伪代码）+ L0 逐位比对（§8.3 主线）
4. ⬜ 与胞弟 scan 单元同 stream 顺序集成（对接契约 §0.6 R4 / §4.2）
5. ⬜（Stage 4，可选）按 §1.6.0 裁决计划实测主选 vs 备选；若上游接口增 `fwd_idx` 则评估 C3（R-4）

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py`（`_make_gather_kernel_nopad`，L218–L258）
- 源 op：`/tmp/opencode/TileOPs-ref/tileops/ops/moe/routed_expert/permute_nopad.py`（`MoePermuteNopadFwdOp`）
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`
- 同族先例 / 工具链陷阱规避：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/RETROSPECTIVE.md`
- 行 gather 先例：`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py`
- 其他同类实现：`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、
  `examples/flash_attention/flash_attn_npuir_dev.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`
- `T.gather` 能力边界佐证：`testing/npuir/indexing_ops/test_slice_gather_dev.py`、
  `testing/npuir/indexing_ops/test_slice_gather_exp.py`
- API 文档：`docs/Tilelang.language/内存操作/T.copy.md`、`docs/Tilelang.language/内存操作/T.alloc_ub.md`、
  `docs/Tilelang.language/索引与元素操作/T.gather.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`、
  `docs/Tilelang.language/编译器提示操作/T.Parallel.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/algorithm-research.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`、
  `.agents/skills/_shared/standards/hardware-cost-model.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（`CONST-mte2-degradation` /
  `CONST-capacity-910B2C` / `CONST-vector-launch-overhead`）、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`（`TRAP-C10-strided-gather`）、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md`
  （`TRAP-UB-dst-align` / `TRAP-load-nd2nz-strided` / `TRAP-T-copy-region-semantics`）、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0010 / 0026 / 0027 / 0028）
- 本单元 D-3 探针：`examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather.py`、
  `examples/moe_permute_nopad/_make_gather_kernel_nopad/history_version/design_probe_gather_perf.py`

---
