# _make_scan_kernel_nopad 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`tileops/kernels/moe/permute_nopad.py` 中的
> `_make_scan_kernel_nopad`（GPU/CUDA target，MoE no-pad token permute 的 **Phase-1 计数排序扫描单元**）。
> 本单元目标：`target="npuir"`（Ascend NPU）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`
> （含 `_make_scan_kernel_nopad`）。
> 强参照（胞兄，同族整数索引算子，已冻结并含 Stage 3 实测）：
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（revision v2，两 kernel GM 交接）。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> 等价性机器验证（D-1）：`examples/moe_permute_nopad/_make_scan_kernel_nopad/verify_equiv.py`
> （E1/E2/E2-corner/E3 全 EQUIV_PASS，见 §1.6.1 结果表）。

**本单元的一句话定位**：`_make_scan_kernel_nopad` 是 MoE no-pad token 重排的**计数排序（counting sort）扫描单元**——
从 `flat_ids[T*K]`（每 token 的 top-k 专家号）算出：① 每专家 token 数 `true_sizes[E]`；② 每专家紧致起始偏移
`true_offsets[E]` 与非填充排他前缀 `expert_first_token_offset[E+1]`（int64）；③ 正向映射 `fwd_idx[T*K]`
（flat 位置 → 紧致 slot）与 gather 索引 `permuted_idx[T*K]`（slot → token 行）。它**不含任何浮点数值计算**
（全程 int32/int64 索引），是一个纯整数索引/计数单元；产物由胞弟 gather 单元（`_make_gather_kernel_nopad`）消费。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **三 kernel GM 交接，按数据流域切分（计数+前缀 / 散写）**：**kernel A**（`_scan_count_kernel`，Vector，多核）
   只做直方图计数（GM **普通 RMW** `bh[core_id,e]=bh[core_id,e]+1`，落 GM `bh`）；**kernel B**（`_scan_prefix_kernel`，Vector，单核）做
   跨块/跨专家前缀（唯一一次 `T.cumsum`）并产 `true_sizes/true_offsets/expert_first_token_offset` 与
   `slot_base`；**kernel C**（`_scan_scatter_kernel`，Vector，多核）逐块做**有序运行 rank 散写**产
   `permuted_idx/fwd_idx`。三 kernel 同 stream 顺序 launch，**同 scope 内无「向量产出 → 标量消费」**
   （规避 CG-2026-0027），无 in-block 跨引擎交接（规避 CG-2026-0026）。
2. **直方图改用 `(block, expert)` GM **普通 RMW** 计数，每块独占一核 → 无跨核争用**：把源
   `T.atomic_add(s_counts[flat_ids[idx]],1)`（UB 运行期下标散加）改为**跨 kernel 可读的 GM scratch**
   `bh[NB, E]`，每核只写自己的块行、逐位置 `bh[core_id,e] = bh[core_id,e] + 1`（**不用原子**：块核 1:1、
   每 (block,expert) 计数私有单核，RMW 语义安全且实测可编译——`T.atomic_add` 标量载荷在本工具链前端失败，
   CG-2026-0029）；全局 `counts[e]` 由 kernel B 列求和得到。
3. **散写/fetch-add 的 NPU 等价重设计：确定性两级块并行稳定计数排序**。源用 `tl_atomic_add_offset`
   （CUDA 定制 extern，**返回旧值**的 fetch-add）拿 slot；NPU `T.atomic_add` 降级为 `hivm::StoreOp`
   （**无返回值**，见 `docs/Tilelang.language/原子操作/T.atomic_add.md` §3），fetch-add 无等价 API。
   本设计以「`off[e]` + 块内前缀 `block_base[b,e]` + 块内运行 rank」的确定性公式替代——**逐位复现源/golden 的
   稳定（flat 位置升序）计数排序顺序**（§1.6.1 E2 机器验证 0 违反），且每 slot 只写一次。
4. **分核：固定物理核数（纯 Vector → 40）+ 核内静态 `T.serial`，块与核 1:1**（`NB = min(P, numel)`），
   否决源「1 block × 1024 thread」与任何超发 grid；核内串行边界（`C = ceildiv(numel, NB)`）全为编译期常量。

### 修订说明（revision v1，相对 v0）

> 本版为 Stage 2 检视「不通过」后的修订（`revision_index=1`）。**仅修订实现层 API 形态与口径，
> 算法族（三 kernel GM 交接 + 确定性块并行稳定计数排序）、分核方案（固定物理核 40、块核 1:1、
> 核内静态 `T.serial`）、`T.cumsum` 前缀（E1）、等价优化项 E1–E5 与全部既有结论均保持不变**。

**相对 v0 的关键调整**（阻塞 2 项 + 建议 4 项）：

| # | 级别 | v0 写法 | v1 修订 | 依据 |
|---|------|---------|---------|------|
| 1 | 阻塞 | kernel A 用 `T.atomic_add(bh[core_id, flat_ids[..]], 1)`（**标量载荷**） | 改**普通 RMW** `e=flat_ids[..]; bh[core_id, e]=bh[core_id, e]+1` | 块核 1:1、每行私有 → 原子语义非必需；标量载荷前端报 `DiagnosticError: 'int' object has no attribute 'buffer'`（CG-2026-0029），普通 RMW 全 shape 实测编译通过 |
| 2 | 阻塞 | kernel B 在 `T.Parallel(E)` 体内**直写 GM 表** `true_sizes/true_offsets/efo` | 改 **UB 暂存（含 int64 `ub_efo`）+ `T.copy` 落 GM** | `T.Parallel.md` §2.3 禁止循环体内直写传入的 GM 张量；实测直写触发 `NPU IR opt failed … crash`，UB 暂存 + `T.copy` 编译通过 |
| 3 | 建议 | §1.6.0 R3 C5 行 `numel·log²numel/2 ≈ 7.4M`（算式与数值差 2×） | 统一口径为 `numel·log²numel ≈ 7.4M`（删 `/2`），脚注注明计数口径 | REVIEW 问题 3 |
| 4 | 建议 | 访存口径（缓冲尺寸 vs 访问次数）混用；GM scratch 峰值「250KB」与 §4.3/§4.5「184KB」不一致 | R3 脚注统一「按缓冲区尺寸 × 实际往返次数」；GM scratch 峰值统一为 **184KB** | REVIEW 问题 4 |
| 5 | 建议 | §0.6 R3 引入 `nk=min(NB,P)`/`bpc=ceildiv(NB,nk)` 与 §3.3/§5.3/§5.5 块核 1:1 脱节 | 删除 `nk`/`bpc`，统一块核 1:1（注明因 `NB ≤ P`，退化为 `b = core_id`） | REVIEW 问题 5 |
| 6 | 建议 | §9.4 R-1 声明未登记全局台账 | 补注已登记条目 **CG-2026-0029**（`T.atomic_add` 标量载荷）与 **CG-2026-0030**（无 fetch-add） | AGENTS.md Gap Rule；REVIEW 问题 6 |

**为何不会再犯同一错误**：

- 两处阻塞问题的共同根因是「**以文档参数表（`T.atomic_add.md` §2.1、`T.Parallel` 通用性）推定用法可用，
  未核 `examples/` 同型先例 / 未做最小 kernel 探针**」。v1 起，凡设计采用「参数表允许但官方示例未覆盖」
  的 API 形态，一律以**同型先例或最小 kernel 编译探针**为准（本轮已由 Stage 2 探针 P1–P7 定案），
  §3.4 的「已验证」标注仅对**有实测/先例佐证的具体形态**授予。
- 口径类问题（问题 3/4/5）的根因是「**同一量在多处手工记账、未设单一权威口径**」。v1 统一：
  复杂度口径脚注为唯一记账规则、GM scratch 峰值以 §4.3 缓冲形状为准、分核表述以 §5.5 块核 1:1 为准，
  其余章节引用而非重算。
- 不再依赖「胞兄同款」这一口头类比：v1 逐形态核对胞兄 `_tile_scheduler_kernel.py`（其 GM 写一律经
  `T.copy`，无 `T.Parallel` 内直写 GM），并在 §3.4 标注差异。

**未静默丢弃的既有结论**（v1 全部保留）：三 kernel GM 交接（§0.6 R1/§1.4/§1.5/§5.6）、确定性两级块并行
稳定序（§0.6 R2/§1.6.1 E2，`verify_equiv.py` 4000+11 例 0 违反）、等价优化 E1–E5（§1.6.1）、
分核 40 核 + 块核 1:1（§5.5）、`T.cumsum` 硬件前缀（§0.5 O6/E1）、C4 单核降级备选与实验裁决计划（§1.6.0）。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = **`_make_scan_kernel_nopad`**（Phase-1 扫描单元）。同文件的 `_make_scan_kernel_nopad_ep`
> （EP 变体，含 `expert_map`）与 `_make_gather_kernel_nopad`（Phase-2 gather）**不在本单元**——前者是
> 带 `expert_map` 的语义扩展（§0.1 边界），后者由胞弟单元 `_make_gather_kernel_nopad` 设计。

### 0.1 源算子语义（做什么）

**数学语义**（纯整数计数排序；无浮点）：

输入 `flat_ids[numel]`（int32，`flat_ids[idx] ∈ [0, E)`，`numel = T*K`，`idx = t*K + k`），构造
**稳定计数排序**映射：

$$
counts[e]=\#\{\,idx:flat\_ids[idx]=e\,\},\qquad
off[0]=0,\quad off[e+1]=off[e]+counts[e]
$$

$$
fwd\_idx[idx]=off[flat\_ids[idx]]+rank(idx)
$$

`fwd_idx[idx]` 表示位置 `idx` 在其专家段内的稳定序号。

$$
permuted\_idx[slot]=idx//K\quad(fwd\_idx[idx]=slot),\qquad
slot\in[off[e],\,off[e+1])
$$

其中 `rank(idx)` = 满足 `j < idx` 且 `flat_ids[j] == flat_ids[idx]` 的 `j` 个数（**稳定序**）；第二式括号条件表示当 `fwd_idx[idx] == slot` 时写入 `permuted_idx[slot]`。输出表：

- `true_sizes[e] = counts[e]`（int32）
- `true_offsets[e] = off[e]`（int32，紧致起始）
- `expert_first_token_offset[e] = off[e]`（int64，非填充排他前缀；长度 `E+1`）
- `fwd_idx[idx] = off_{flat_ids[idx]} + rank(idx)`（int32，flat 位置 → 紧致 slot）
- `permuted_idx[slot] = idx//K`（int32，slot → token 行；供 gather 读 `hidden_states`）

- **规约语义**：计数/前缀为**整数精确**运算（无累加顺序/舍入问题）。`expert_first_token_offset` 用 int64
  累加（源 L82–L84）；其余全 int32。**整数计数排序的等价性可要求逐位精确**。
- **dtype 语义**：全程 int32（`flat_ids`、`true_*`、`permuted_idx`、`fwd_idx`、`counts/off32`）+ int64
  （`expert_first_token_offset`）。无 dtype 提升/回落、无浮点。
- **边界语义**：
  - `counts[e] = 0`（空专家）→ `off[e] = off[e+1]`，该专家段为空，映射仍唯一确定。
  - `numel = 0`（全空）→ 全 `counts=0`、`off` 全 0；无 slot 被写。
  - **`fwd_idx` 的专家内顺序**：源用 `tl_atomic_add_offset`（fetch-add）取 slot，**专家内顺序取决于并发
    atomic 执行序**，属**非确定**；op 测试（`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` L105–L111）
    只校验**顺序无关**一致性 `perm_h[fwd_idx[flat]] == hidden_states[flat//K]`（以及 `offsets/sizes` 精确），
    **不校验 `fwd_idx` 的专家内具体顺序**；host 侧 golden（`_ref_moe_permute_nopad` L55–L59）用顺序 `write_ptr`
    产出**稳定序**。故本设计选择的确定性**稳定序**既满足 op 契约、又与 golden 逐位一致（§1.6.1 E2）。
  - `top_k = K`：`permuted_idx[slot] = idx // K`（编译期常量，整数除法，非负）。

**语义保持基线**：§8.1 golden 以本节为唯一依据（纯 Python/torch 顺序计数排序；移植源仓测试
`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` 的参考实现 `_ref_moe_permute_nopad`），**不得复刻** §0.6 的
NPU 算法（三 kernel 块并行直方图 + 运行 rank）。

**host 侧语义**：本 kernel **无 host 预处理/后处理**；`numel/top_k/num_experts` 均为编译期 Python 常量经
`lru_cache` 特化（源 op `_get_kernel` 按 `(T,K,E,H,dtype,device)` 缓存）。host 侧仅**分配** scratch/输出张量
（`torch.zeros` 零初始化，非改动输入内容）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `flat_ids` | `(numel,)` | int32 | 输入（= `topk_ids.flatten()`）：每位置的目标专家号 |
| `expert_first_token_offset` | `(num_experts + 1,)` | int64 | **输出**：非填充排他前缀 |
| `true_offsets` | `(num_experts,)` | int32 | **输出**：每专家紧致起始 |
| `true_sizes` | `(num_experts,)` | int32 | **输出**：每专家真实 token 数 |
| `permuted_idx` | `(numel,)` | int32 | **输出（内部交接）**：slot → token 行 |
| `fwd_idx` | `(numel,)` | int32 | **输出**：flat 位置 → 紧致 slot |
| `write_offsets` | `(num_experts,)` | int32 | **内部 scratch**（源为 fetch-add 目标） |

源 `@tilelang.jit(out_idx=[], ...)`：输出由 host 侧 `torch.empty/zeros` 分配；`_SCAN_THREADS = 1024`，
`T.Kernel(1, threads=threads)`（**单 block**）。`compile_flags=["-O3","-include", _ATOMIC_HELPER_H]`
（extern 声明 `tl_atomic_add_offset`，见 `examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_atomic_helper.h`）。

### 0.3 实现算法解读（怎么算）

源 kernel 结构（源 L59–L115）：`@tilelang.jit(...)` 工厂，`T.Kernel(1, threads=1024)` **单 block**；`tx = T.get_thread_binding()`。
SMEM：`s_counts[E]`（int32）、`s_offset[E+1]`（int64）、`slot_buf[1]`（`T.alloc_local`）。五个串行相位：

- **Step 1**：`for i in serial(ceildiv(E,threads)): if idx<E: s_counts[idx]=0` + `sync_threads`。
- **Step 2**：`for i in serial(ceildiv(numel,threads)): if idx<numel: T.atomic_add(s_counts[flat_ids[idx]],1)`
  （**逐元素散加**到 SMEM，运行期下标） + `sync_threads`。
- **Step 3**：`if tx==0: s_offset[0]=0; for e in serial(E): s_offset[e+1]=s_offset[e]+int64(s_counts[e])`
  （**thread 0 串行**排他前缀，int64） + `sync_threads`。
- **Step 4**：向量化写 `expert_first_token_offset[0:E+1]=s_offset`；`true_offsets[e]=int32(s_offset[e])`、
  `true_sizes[e]=s_counts[e]`、`write_offsets[e]=int32(s_offset[e])`（初始化散写指针）。
- **Step 5**：`for i in serial(ceildiv(numel,threads))`：`eid=flat_ids[idx]`；
  `slot_buf[0]=tl_atomic_add_offset(&write_offsets[0], eid, 1)`（**fetch-add 返回旧值 = slot**）；
  `permuted_idx[slot]=idx//top_k`；`fwd_idx[idx]=slot`。

**计算步骤分解**（源码每条计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P1 | `s_counts[idx]=0` | — | SMEM `s_counts` | 清零 |
| P2 | `T.atomic_add(s_counts[flat_ids[idx]],1)`（逐元素散加） | `flat_ids` | `s_counts` | $counts[e]$ |
| P3 | `s_offset[0]=0; s_offset[e+1]=s_offset[e]+int64(s_counts[e])`（thread 0 串行） | `s_counts` | SMEM `s_offset`(int64) | $off[0]=0,\ off[e+1]=off[e]+counts[e]$ |
| P4a | `expert_first_token_offset[idx]=s_offset[idx]` | `s_offset` | GM `efo`(int64) | $efo_e=off[e]$ |
| P4b | `true_offsets[e]=int32(s_offset[e])` | `s_offset` | GM `true_offsets` | $true\_offsets_e=off[e]$ |
| P4c | `true_sizes[e]=s_counts[e]` | `s_counts` | GM `true_sizes` | $true\_sizes_e=counts[e]$ |
| P4d | `write_offsets[e]=int32(s_offset[e])` | `s_offset` | GM `write_offsets` | 散写指针初值 $off[e]$ |
| P5a | `slot=tl_atomic_add_offset(&write_offsets[eid],1)` | `flat_ids`, `write_offsets` | `slot_buf` | $slot=off[e]+rank$ |
| P5b | `permuted_idx[slot]=idx//top_k` | `idx`, `slot` | GM `permuted_idx` | $permuted_idx[slot]=idx//K$ |
| P5c | `fwd_idx[idx]=slot` | `idx`, `slot` | GM `fwd_idx` | $fwd_idx[idx]=off[e]+rank$ |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[flat_ids(numel,)] --逐元素散加--> SMEM[s_counts(E) int32]  (atomic_add, 运行期下标)
                                          │  step3 (thread0 串行 int64)
                                          ▼
                                   SMEM[s_offset(E+1) int64]
                                          │  step4
              ┌───────────────────────────┼──────────────────────────────┐
              ▼                           ▼                              ▼
   GM[efo(E+1) int64]        GM[true_offsets/true_sizes(E) int32]   GM[write_offsets(E) int32]
                                                                          │ step5 fetch-add
                                                                          ▼
                                          GM[flat_ids] --eid--> slot = fetch_add(write_offsets[eid])
                                                                          │
                                              ┌───────────────────────────┴───────────────┐
                                              ▼                                           ▼
                              GM[permuted_idx(numel) int32]              GM[fwd_idx(numel) int32]
```

**循环与并行结构**：**单 block**（grid=1）+ 1024 线程。每线程处理 `numel/1024` 个位置；step3 仅 thread 0
工作其余空等；step1/2/4/5 为 thread 级并行 + `T.sync_threads()` 相位间隔。**未使用 `T.Parallel` /
`T.Pipelined`**——纯 thread 级 scalar/atomic 循环。

**host 侧逻辑**（源 op `forward`，属调用方，非本单元）：`flat_ids = topk_ids.flatten().contiguous()`；
分配 `permuted_idx = torch.zeros(numel)`（**零初始化**以兜底未写 slot）、其余 `torch.empty`；调用 scan 后
调用 gather。**无 host 侧数据预处理**。

**编译配置**：`@tilelang.jit(out_idx=[], compile_flags=["-O3","-include", _ATOMIC_HELPER_H])`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM 驻留 `s_counts`/`s_offset`（`T.alloc_shared`） | 免 GM 往返、读低延迟 | 块内共享计数/前缀 | shared memory | 硬件强相关（NPU 有 UB/GM 等价） |
| O2 | **单 block**（grid=1）一次性完成计数→前缀→散写 | 免跨 block 归并 / 全局同步 | 一个 block 内 `sync_threads` 相位 | 大 block + 廉价 block 内同步 | 硬件强相关（NPU 一维核 + 无块内 barrier 可用） |
| O3 | thread 级并行（1024 线程摊 numel） | 摊平计数/散写的串行度 | 每 thread 一位置 | warp/thread 模型 | 硬件强相关（NPU 是 SIMD lane，非 thread） |
| O4 | `tl_atomic_add_offset` **fetch-add** 散写 | 免每位置 rank 前缀，并发直接拿 slot | 原子取旧值 + 自增（CUDA 定制 extern） | 全局/共享原子 RMW（有返回值） | **硬件强相关且 NPU 无等价**（`T.atomic_add` 无返回值） |
| O5 | int64 排他前缀（仅 thread 0 串行） | 免溢出 | 64-bit 累加 | 无 | 算法层可移植 |
| O6 | thread 0 串行前缀（`T.serial(E)`） | 简单、E 小（≤384）成本可接受 | 单线程线性扫描 | 无 | 算法层可移植（NPU 改硬件 scan） |
| O7 | `permuted_idx`/`fwd_idx` 落 GM 供 Phase-2 gather 读 | 两 kernel 解耦（计数与搬运分离） | 独立 scan kernel + GM 交接 | GPU 多 kernel 顺序执行 | 模式可移植（NPU GM 表交接，胞兄已证） |

> 识别不出来 ≠ 不存在。O1–O7 为源码显式优化，逐项在 §0.5 判定。
> 注：Phase-2 gather 的 `ROWS_PER_BLOCK` 分块与 128-bit `uint4` 向量化（源 L218–L258）属胞弟单元，不在本表。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（计数排序映射、`true_*`、`permuted_idx`/`fwd_idx` 互逆、int32/int64） | 语义 | 无 | — | **保留** | 语义无条件保留（migration-analysis.md §5.1）；稳定序选择见 §0.1 边界语义（op 测试顺序无关，golden 为稳定序） |
| O1 SMEM `s_counts`/`s_offset` | 优化 | shared memory | 有（UB/GM） | **等价替换** | 计数表改 **GM scratch**（跨 kernel 可读，见 O4/R2），前缀中间量用 `T.alloc_ub`（§4.5） |
| O2 单 block（grid=1） | 优化/结构 | 大 block + 廉价块内同步 | 无（NPU 一维核；无块内 barrier 原语） | **重新设计** | 拆 **三 kernel GM 交接**（计数 / 前缀 / 散写）；跨核沟通只用 GM + stream 顺序（胞兄 `_tile_scheduler_kernel` 已证底座） |
| O3 thread 级并行（一 thread 一位置） | 优化 | CUDA thread/warp | 无（NPU 为 SIMD lane） | **重新设计** | 分核：块与核 1:1（`NB=min(P,numel)`），核内 `T.serial(C)` 静态边界；可向量化部分用 `T.Parallel`（§0.6 R3） |
| O4 fetch-add 散写（`tl_atomic_add_offset`） | 算法/优化 | CUDA 定制 extern，**fetch-add 返回旧值** | **无**（`T.atomic_add` 降 `hivm::StoreOp`，无返回值） | **重新设计** | §0.6 R2：**确定性两级块并行稳定计数排序**（`off[e] + block_base[b,e] + 块内 rank`），逐位复现稳定序（§1.6.1 E2）；fetch-add 缺口登记为能力缺口（§9.4 R-1） |
| O5 int64 排他前缀 | 优化 | 无 | 有 | **保留** | int32 偏移 `T.cast` 到 int64 写 `efo`（numel<2^31，逐位精确，§1.6.1 E3） |
| O6 thread 0 串行前缀（`T.serial(E)`） | 算法/优化 | 无 | 有（算法层） | **等价替换** | 改硬件 `T.cumsum`（`(1,E)` int32）+ 1-lane 移位（E1；胞兄 kernel A 同款、设备实测可用） |
| O7 GM 表交接 + 独立 scan kernel | 优化/结构 | 无（GPU 多 kernel） | 有 | **保留（主选承载形态）** | 三 kernel 同 stream 顺序 launch，产物落 GM；胞弟 gather 读 `permuted_idx`。**依据**：胞兄 `_tile_scheduler_kernel` Stage 3 实测两 kernel GM 交接位精确（`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/_tile_scheduler_kernel.py`） |
| （舍弃）UB 内运行期下标散加（源 P2 的 SMEM atomic） | 优化 | shared memory + 运行期下标 | 无可用形态 | **舍弃** | CG-2026-0028(d)：**标量读 UB 的运行期计算下标污染相邻结果**；改为 GM `bh`（R2） |

> 每项处置依据见「NPU 对应方案 / 依据」列；**语义层不可妥协**。
> **判定输入含 Phase R 调研结论（§1.6.0）**：调研在「可实现族」内择优——三 kernel GM 交接 + 确定性块并行计数排序
> （§1.6.1 E2 位精确、构造规避 CG-2026-0027/0028）胜出；源单 block fetch-add 形态因 NPU 无 fetch-add 被否决（可行性前置）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**三 kernel GM 交接——A 计数 + B 前缀 + C 确定性块并行散写**）。

**重设计项 R1：承载形态——从「单 block 全相位」改为「三 kernel GM 交接」**

- **源方案**：1 个 block（1024 线程）内 5 个相位，`sync_threads` 间隔；thread 级并行。
- **NPU 新算法**：拆三 kernel（同 stream 顺序 launch）：
  - **kernel A**（`_scan_count_kernel`，Vector，`T.Kernel(NB)`）：逐元素普通 RMW
    `e=flat_ids[idx]; bh_gm[b, e]=bh_gm[b, e]+1`（**不用原子**：块核 1:1、每行私有）产**每块直方图**
    `bh[NB, E]`（`b = idx // C`）；每块独占一核 → 无跨核争用。
  - **kernel B**（`_scan_prefix_kernel`，Vector，`T.Kernel(1)`）：`counts[e] = Σ_b bh[b,e]`（核内 `T.serial(NB)` ×
    `T.Parallel(E)` 向量加）→ **一次 `T.cumsum(counts)` (1,E)** + 1-lane 移位 → `off`；写 `true_sizes`、
    `true_offsets`、`expert_first_token_offset`（int64 cast）；并算 `slot_base[b,e] = off[e] + Σ[b'<b] bh[b',e]`（落 GM）。
  - **kernel C**（`_scan_scatter_kernel`，Vector，`T.Kernel(NB)`）：逐块核内 `T.serial(C)` 运行 rank：
    `e=flat_ids[idx]`、`r=run_gm[b,e]`、`run_gm[b,e]=r+1`、`slot=slot_base_gm[b,e]+r`、
    `fwd_idx[idx]=slot`、`permuted_idx[slot]=idx//top_k`。
- **语义保持论证**：三 kernel 的产物逐元素等于源语义（§0.1）——`counts`/`off` 为整数精确定义；`fwd_idx`/
  `permuted_idx` 的**稳定序**与源 host golden 逐位一致（§1.6.1 E2：块并行展开 == 顺序计数排序，4000+11 例 0 违反）。
  **无跨引擎交接、无同 scope「向量产出 → 标量消费」**（标量只读**跨 kernel GM**）。
- **与源算法优化意图的承接**（§0.4 逐项，**不得静默丢弃**）：
  - O1（SMEM 计数）：承接（GM `bh`，跨 kernel 可读）；O2（单 block）：重设计（R1 三 kernel）；O3（thread 并行）：
    重设计（R3 分核）；O4（fetch-add）：**意图承接、形态替换**——源意图「免每位置 O(E) rank 前缀，并发直接得 slot」；
    NPU 无 fetch-add，改为**确定性块并行**（每 slot 只写一次，O(numel) 运行 rank，位精确复制稳定序）；
    O5（int64 前缀）：承接（cast）；O6（串行前缀）：等价替换（`T.cumsum`）；O7（GM 表交接）：保留。

**重设计项 R2：直方图——从「UB 运行期下标原子散加」改为「(block, expert) GM 普通 RMW 计数」+ 确定性块并行散写**

- **源方案**：`T.atomic_add(s_counts[flat_ids[idx]],1)`（SMEM，运行期下标）；散写用 `tl_atomic_add_offset`。
- **NPU 新算法**：
  - 计数：**普通 RMW** `e=flat_ids[base+c]; bh_gm[b, e]=bh_gm[b, e]+1`，`b = core_id`（每核一行，无争用）；
    `bh[NB,E]` GM scratch 零初始化。因块核 1:1、每 (block,expert) 计数**私有单核**，无需原子语义；
    **不用 `T.atomic_add`**——其标量载荷形态在本工具链前端失败（`DiagnosticError: 'int' object has no
    attribute 'buffer'`，CG-2026-0029），且本场景无并发写者，RMW 与原子等价。GM 动态下标 RMW 已由
    Stage 2 最小 kernel 探针在代表 shape（`numel=32768/E=384/NB=40/C=820`）实测编译通过。
  - 散写：**不用原子**——每个 (block, expert) 的 `run[b,e]` 仅由拥有该块的核写（1 核 1 块），普通 RMW 即安全；
    `slot = slot_base[b,e] + r` 由**确定性公式**给出，与执行时序无关。
- **为什么必须去原子散写**：NPU `T.atomic_add` 无返回值（fetch-add 缺口，CG-2026-0030），源的
  `slot = fetch_add(...)` 无法直译；且 UB 运行期下标（源 P2/P5 形态）命中 CG-2026-0028(d)。故散写改为确定性
  rank 公式；计数侧则因块核 1:1 而仅需普通 RMW。
- **语义保持论证**：块并行展开与源/golden 的**稳定**计数排序逐位相同（§1.6.1 E2 机器验证）；空专家段为空、
  `numel=0` 无 slot 被写（corner 验证含 numel=0/1、`numel<NB`、非整除）。

**重设计项 R3：任务划分与分核——从「单 block × 1024 thread」改为「固定物理核数 + 块核 1:1 + 核内静态串行」**

- **源方案**：`T.Kernel(1, threads=1024)`，一 thread 一元素。
- **NPU 新算法**：`NB = min(P, numel)`（`P` = 纯 Vector 实查物理核数 = 40，§5.5）；`C = ceildiv(numel, NB)`
  （均编译期常量）。因 `NB ≤ P` 恒成立，**块核 1:1**（`b = core_id`），无 `nk`/`bpc` 中间层（`nk ≡ NB`、
  `bpc ≡ 1`）。kernel A/C 用 `T.Kernel(NB)`，核内 `for c in T.serial(C)`（**静态边界**），边界用
  `if base + c < numel` 守卫（普通 `if`，非 `T.if_then_else` 表达式）。kernel B 单核。
- **语义保持论证**：分核只改「哪个核写哪些块」，不改任何 slot 的 (expert, rank) 结果；块内顺序 = 位置顺序，
  块间由 `block_base` 保证有序 → 全序仍是稳定序，故结果与核数/NB 无关（§1.6.1 E2 对多个 NB 随机验证）。

**弃选：单核全局顺序散写（无块结构）**

- 结构：kernel C 单核，`for idx in T.serial(numel)` + 全局 `run[E]` 计数器。**弃选理由（性能）**：`numel` 最大
  32768、40 核仅用 1 核，scalar 运行 rank 全串行（`numel × ~6 GM 访问` 于单核）；与 §5.5「极大规模核内串行摊薄
  启动开销」精神相悖。保留为 §1.6.0 备选（主选失败时降级，见实验裁决计划）。

**弃选：内联进 gather 核（表不落 GM）**

- 结构：scan 与 gather 合并，`permuted_idx` 不落 GM。**弃选理由（可行性）**：CG-2026-0010（Developer
  persistent + `T.gemm` + 整型索引段共存即 lowering segfault）+ CG-2026-0026（跨引擎同 block 交接不可编译，
  官方 `examples/mixcv/mixcv_mixkernel.py` 复现）——gather 若含 `T.gemm`/跨引擎交接则不可实现；且本单元与
  gather 单元独立设计，保持 GM 接口即胞兄已证底座。

### 0.7 标杆实现

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py` 的 `_make_scan_kernel_nopad`（L43–L119）。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`（L57–L133）。
- 源 op / 测试 / bench / manifest / workload：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`（`_ref_moe_permute_nopad` 为 golden 依据，L31–L70）、
  `examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoePermuteNopadFwdOp`）、
  `examples/TileOPs/tileops/workloads/moe.py`。
- 胞兄设计（同族整数索引算子，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/_tile_scheduler_kernel.py`。
- 计数排序与等价 golden 参考（纯 Python 逐专家 pieces 展开）：`examples/TileOPs/tests/ops/test_moe_permute_nopad.py`
  的 `_ref_moe_permute_nopad`；胞兄 golden 亦用 `torch.repeat_interleave` + `cumsum`（`_tile_scheduler_kernel.py` L105–L144）。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的三 kernel 算法）。

---

## 1. 概述

### 1.1 算子名称

`_make_scan_kernel_nopad`（MoE no-pad token 重排的**计数排序扫描单元**，纯整数索引，`target="npuir"`）。

### 1.2 功能描述

从 `flat_ids[numel]`（int32，`numel = T*K`）计算 no-pad 计数排序的元数据：每专家 token 数 `true_sizes[E]`、
紧致起始 `true_offsets[E]`、非填充排他前缀 `expert_first_token_offset[E+1]`（int64）、正向映射
`fwd_idx[numel]`、gather 索引 `permuted_idx[numel]`。**迁移决策后的 NPU 形态**（§0.6 R1）：
**三 Vector kernel GM 交接**（A：普通 RMW 块直方图 `bh`；B：`T.cumsum` 前缀 + `slot_base`；
C：确定性块并行运行 rank 散写 `permuted_idx`/`fwd_idx`），同 stream 顺序 launch；`permuted_idx` 交胞弟
gather 单元读。全程整数、无浮点、无 CPU 同步。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
counts[e]=\#\{idx:flat\_ids[idx]=e\},\quad
off[0]=0,\ off[e+1]=off[e]+counts[e]
$$

$$
fwd\_idx[idx]=off[e]+rank(idx),\quad
rank(idx)=\#\{j<idx:flat\_ids[j]=flat\_ids[idx]\}
$$

$$
permuted\_idx[off[e]+rank(idx)]=idx//K
$$

输出 `(true_sizes, true_offsets, expert_first_token_offset, fwd_idx, permuted_idx)`。**全程 int32/int64，无浮点**。
manifest roofline（`examples/TileOPs/tileops/manifest/moe.yaml` `MoePermuteNopadFwdOp`）是 **op 级**（含 gather 的
`T*K*H` 搬运 + 元数据字节），属整个 permute op，非本扫描单元。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源结构差异已逐条注明）：

1. **kernel A — 块直方图（Vector，多核）**：`T.Kernel(NB)`（块核 1:1，`b=core_id`）；核内 `T.serial(C)`
   逐位置普通 RMW `e=flat_ids[base+c]; bh_gm[b, e]=bh_gm[b, e]+1`（**不用原子**，每行私有）；产 GM
   `bh[NB, E]`（零初始化）。
   **与源差异**：源为 SMEM 运行期下标原子散加（CG-2026-0028d 禁止），改为 GM `bh` 普通 RMW + 块核 1:1；
   `T.atomic_add` 标量载荷不可编译（CG-2026-0029），本场景无并发写者故 RMW 等价。
2. **kernel B — 跨块求和 + 前缀 + 输出（Vector，单核）**：`counts[e]=Σ_b bh[b,e]`（核内串行 `b` × `T.Parallel(E)`）；
   **唯一一次 `T.cumsum(counts)` (1,E)** + 1-lane 移位 → `off[E+1]`；`true_sizes/true_offsets/
   expert_first_token_offset`（int64 cast）经 **UB 暂存 + `T.copy`** 落 GM；算
   `slot_base[b,e]=off[e]+Σ[b'<b]bh[b',e]` 落 GM。
   **与源差异**：源 thread0 `T.serial(E)` 串行 → 硬件 scan（E1）；源逐元素循环写表 → UB 暂存 + `T.copy`。
3. **kernel C — 确定性块并行散写（Vector，多核）**：`T.Kernel(NB)`（块核 1:1）核内 `T.serial(C)` 位置：
   `e=flat_ids[idx]; r=run_gm[b,e]; run_gm[b,e]=r+1; slot=slot_base_gm[b,e]+r; fwd_idx[idx]=slot;
   permuted_idx[slot]=idx//top_k`。
   **与源差异**：源 fetch-add 原子取 slot（NPU 无等价）→ 确定性 `off[e]+block_base+rank` 公式（R2），
   逐位复现稳定序（§1.6.1 E2）。
4. **消费方式**：5 张表落 **GM**，`permuted_idx` 由胞弟 gather 单元读（`hidden_states[permuted_idx[slot]]`）；
   本单元自身不搬运浮点数据。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[kernel A · Vector · NB=min(P,numel), 块核 1:1]
flat_ids[numel] ──T.serial(C) 逐元素 普通 RMW──▶ GM[bh(NB,E) int32]（每块直方图，零初始化）
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel B · Vector · grid=1]
  GM[bh] ──串行 b × T.Parallel(E) 求和──▶ counts(E) ──T.cumsum + 1-lane 移位──▶ off(E+1)
  GM[bh] ──串行 b 运行累加 + off──▶ GM[slot_base(NB,E) int32]
  counts/off ──写──▶ GM[true_sizes(E)] / GM[true_offsets(E)] / GM[expert_first_token_offset(E+1) int64]
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel C · Vector · nk=min(NB,P)]
  flat_ids + slot_base + run(NB,E)（零初始化）──串行运行 rank──▶
        GM[fwd_idx(numel) int32]（flat 位置 → slot）
        GM[permuted_idx(numel) int32]（slot → token 行）──┐
                                                          ▼ （同 stream 顺序 launch）
                                   [胞弟 _make_gather_kernel_nopad] perm_h[slot]=hidden_states[permuted_idx[slot]]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（「`flat_ids` → 稳定计数排序的 counts/前缀/fwd_idx/permuted_idx」）的**算法族**。
> 输入/源算法（单 block SMEM 原子计数 + fetch-add 散写）只是基线候选之一。深度分级：多步索引/搜索/散写类
> → **完整调研**。
> **可实现性约束（前置，来自 Phase M1 + 胞兄 Stage 3 实测）**：
> (a) 本工具链**无 fetch-add**——`T.atomic_add` 降 `hivm::StoreOp`（`docs/Tilelang.language/原子操作/T.atomic_add.md`
>     §3），源的 `tl_atomic_add_offset`（返回旧值）无等价 API；且 UB 运行期下标散加/读污染相邻结果
>     （CG-2026-0028(d)，`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`）；
> (b) **同 scope「向量产出 → 标量消费」不可实现**（CG-2026-0027，同产物多次运行结果不一致，`T.pipe_barrier` 无效）；
> (c) 跨引擎同 block 交接不可编译（CG-2026-0026，官方样例复现）；Developer persistent + `T.gemm` + 整型索引段
>     共存即 segfault（CG-2026-0010）；
> (d) int32 `>>`/`T.alloc_local`+`T.serial`/标量 `T.if_then_else`/多行 int32 `T.cumsum`/一 kernel 两次 `T.cumsum`
>     均 SIGSEGV 或错值（CG-2026-0028）；
> (e) `T.atomic_add` **标量载荷**前端失败（`DiagnosticError: 'int' object has no attribute 'buffer'`，CG-2026-0029），
>     逐元素 `+1` 散加需改普通 RMW（写入者互斥时）或向量载荷；
> (f) `T.Parallel` 循环体内**禁直写传入的 GM 张量**（`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3），
>     GM 写一律经 UB 暂存 + `T.copy`。
> 调研在**可实现族**内择优（可行性前置于经济性）。证据：`.agents/evolution/capability-gaps.md` CG-2026-0027/0028/0029/0030。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | 单 block；SMEM 原子计数 → thread0 串行 int64 前缀 → fetch-add 散写 | — | — | ✅ 基线 |
| C1 | **三 kernel GM 交接 + 确定性块并行计数排序**（**选定**） | A：GM `bh` 普通 RMW 块直方图；B：`T.cumsum` 前缀 + `slot_base`；C：块内运行 rank 散写 | 逐位复现稳定计数排序（E2） | 去 fetch-add；分核到 40；无同 scope 标量消费向量 | ✅ **选定** |
| C2 | **硬件前缀 scan**（`T.cumsum` + 1-lane 移位） | 排他前缀 = 含前缀 cumsum 后 1-lane 移位（E1） | 整数恒等 | O(E) 串行标量 → 1 op | ✅ 采纳（并入 C1/B） |
| C3 | **确定性块并行稳定计数排序**（rank = `off+block_base+运行r`） | 块内运行 rank + 块前缀 `block_base` | 与顺序稳定计数排序逐位相同（E2） | 每 slot 只写一次；无原子 fetch-add；O(numel) 运行 rank | ✅ 采纳（并入 C1/C） |
| C4 | **单核全局顺序散写**（无块结构） | 单核 `T.serial(numel)` + 全局 `run[E]` | 数学恒等（即稳定序） | 结构最简，但只用 1/40 核 | ✅ 对比（**备选**） |
| C5 | **`T.vsort` 排序驱动**（key=`flat_ids*numel+idx`，两次 vsort 得 permuted/fwd） | 排序即计数排序；逆置换由第二次 vsort 得 | 数学恒等（非稳定，但契约允许） | 全向量；但 **UB 容量不足**（3×numel×4=384KB>192KB）且 `T.vsort` 仅 fp16/fp32（int 需 cast，>2^24 键不安全） | ✅ 对比（弃选） |
| C6 | **逐专家全轴 masked scan**（per-expert 对 numel 做指示量前缀） | 对每专家扫描全 numel 求 rank | 数学恒等 | O(E·numel)=12.6M + 需多次 cumsum（CG-2026-0028e） | ✅ 对比（弃选） |
| C7 | **`T.gather` 逆置换** | 用 gather 求 `fwd_idx` 逆 | 数学恒等 | gather 的索引本身就是待求置换（循环依赖） | ✅ 对比（弃选） |
| C8 | **host 侧计数排序** | CPU 算表后下发 | 数学等价 | 引入 CPU 同步 + host 禁改输入（ascend-constraints.md §4） | ✅ 对比（弃选） |

**R2 在线算法**：**有**。计数排序本身即**单遍在线**形态——顺序扫一遍 `flat_ids`，对每位置维护
per-expert running counter（O(E) 状态）即得 `rank`；前缀 `off` 亦为单遍 running sum。
本设计的 kernel A（直方图）与 kernel C（运行 rank）都是**单遍流式**（对 `flat_ids` 各扫一遍、无反向依赖）。
**结构判据**：计数/前缀是可分解的 running 统计量（计数/和）；散写的稳定序由「块序 × 块内位置序」全序给定，
无需多遍物化。收益口径：扫描遍数 = 2（A、C 各 1 遍 `flat_ids`）；中间缓冲 = `bh`/`slot_base`/`run`
各 `O(NB·E)`（≤ 40×384 int32 = 61KB，落 GM）。**不存在需要 GM 级多遍物化的候选**。

**R3 复杂度对比**（四口径；代表 shape = Kimi-K2 prefill：`T=4096, K=8, E=384` → `numel=32768`；
`NB=min(P,numel)=40`、`C=ceildiv(numel,NB)=820`；elem=int32=4B）：

| 算法候选 | 计算量（整数 op） | 访存量 (Bytes) | 扫描遍数（flat_ids） | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|---------------|---------------------|-------------|---------------------|
| C0 基线（单 block SMEM 原子 + fetch-add） | numel 原子加 + E 串行 + numel fetch-add ≈ **65.9K** | flat_ids 128KB + 表 7.7KB + permuted 128KB + fwd 128KB ≈ **400KB** | 2（P2、P5） | SMEM `(E+E+1)` ≈ 2.3KB | 1 block × 1024 thread；**NPU 无 fetch-add（不可实现）** |
| **C1+C2+C3 选定（三 kernel）** | A：numel RMW ≈ 32.8K；B：`NB·E` 前缀 ≈ 15.4K + 1 scan；C：numel 运行 rank ≈ 32.8K；合计 ≈ **81K** | flat_ids 128KB×2 + `bh` 61KB×3 + `slot_base` 61KB×2 + `run` 61KB×2 + fwd 128KB + permuted 128KB + 表 7.7KB ≈ **950KB** | 2（A、C） | GM scratch ≈ 184KB（`bh`+`slot_base`+`run` 各 61.4KB；UB 仅 ~13.8KB） | 40 核；**无跨核归并、无 in-block 跨引擎同步** |
| C4 单核顺序（备选） | numel 运行 rank ≈ 32.8K + numel 直方图 ≈ 32.8K | 同 C1（`run` 降为全局 `run[E]`） | 2 | GM scratch ≈ 124KB（`bh`+`slot_base` 仍由 A/B 产，`run[E]` 1.5KB） | **1 核**（40× 串行度） |
| C5 `T.vsort` | numel·log²numel ≈ **7.4M** compare-exchange | 3×numel×4×↔ ≈ **393KB/遍** | 2 | **UB 384KB > 192KB（容量否决）** | 高但容量不可行；`T.vsort` 仅 fp16/fp32 |
| C6 逐专家 masked scan | `E·numel` = **12.6M** + 前缀 | flat_ids 重读 E 次 ≈ 49MB | E=384 | UB E×C 超容 | 高但**触发 CG-2026-0028(e/f)** |
| C7 gather 逆置换 | 循环依赖 | — | — | — | 不可构造 |
| C8 host 排序 | host O(numel log numel) | flat_ids 设备→host + 表 host→设备 | 1 + 2 总线往返 | host 内存 | **CPU 同步阻塞** |

口径脚注：计算量计整数加/比较/原子（本算子无浮点）；**访存统一按「缓冲区尺寸 × 该 buffer 在 GM 侧的读/写往返次数」
计**（读或写各计一次，**不按元素访问次数**；用于跨候选同口径比较）——故 `bh` 计 3 次往返（A 写 1 + B 趟 1 读 1 +
B 趟 2 读 1）、`slot_base` 计 2 次（B 写 1 + C 读 1）、`run` 计 2 次（C 读 1 + 写 1）；不含 UB 内部重复；
扫描遍数指 `flat_ids` 在 GM 侧被读的次数；中间缓冲峰值以 GM scratch **驻留峰值**（buffer 形状之和，非累计往返）为准。
**本单元是极小工作量的索引算子**（prefill 下 ≪10^5 整数 op），故 R3 排序以**固定开销（发射 / GM 往返）**为主导。

**R4 硬件亲和性评估**（逐候选对照检查清单；「工具链可用性」列按 negative-claim-evidence.md §2 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Vector（整数）✅ 但单 block 1024 thread → SIMD lane 不匹配 ⚠️ | SMEM ✅ | ✅ | ✅ | ❌ 块内无屏障原语 | ❌ **不可实现**：NPU 无 fetch-add（`T.atomic_add` 无返回值）+ UB 运行期下标（CG-2026-0028d） | ❌ 弃选（可行性） |
| **C1+C2+C3 选定** | Vector（GM 普通 RMW / `T.Parallel` / `T.cumsum`；标量 rank 为顺序依赖）✅ | UB ~13.8KB ≪ 192KB ✅ | E=384、numel 任意（守卫）✅ | `C`/`NB` 编译期常量 ✅ | ✅ 三独立 Vector kernel，无 `T.gemm`、无跨引擎交接 | ✅ **构造上规避 CG-2026-0010/0026/0027/0028**；GM 动态下标 RMW 全 shape 实测可编译 / `T.cumsum` int32(1,E) / 动态标量 GM 读均有本仓先例 | ✅ **选定** |
| C4 单核顺序（备选） | Vector ✅（但 1/40 核） | UB ~13.8KB ✅ | ✅ | ✅ | ✅ | ✅ 结构最简（同 C1 构件） | ✅ **备选**（主选编译/性能失败时降级） |
| C5 `T.vsort` | Vector（VSortOp）✅ | ❌ 3×numel×4=384KB > 192KB | ✅ | ✅ | ✅ | ⚠️ `T.vsort` 仅 fp16/fp32（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3） | ❌ 弃选：容量否决 + 键 cast 精确性风险 |
| C6 逐专家 masked scan | Vector ✅ | ❌ E×C 超 UB | ✅ | ✅ | — | ❌ **触发 CG-2026-0028(e/f)**（多 cumsum / 多行 int32 cumsum） | ❌ 弃选 |
| C7 gather 逆置 | Vector ✅ | ✅ | ✅ | ✅ | ✅ | ✅ API 存在（`docs/Tilelang.language/索引与元素操作/T.gather.md`）但**索引循环依赖** | ❌ 弃选（不可构造） |
| C8 host 排序 | CPU | — | — | — | ❌ host-device 往返 | ✅（可行但违约束） | ❌ 弃选：违反零 CPU 同步 + host 禁改输入 |

**调研结论**：选定 **C1 + C2 + C3 —— 三 kernel GM 交接（A：GM `bh` 普通 RMW 块直方图；B：一次 `T.cumsum`
(1,E) 前缀产 `off`/输出/`slot_base`；C：确定性块并行运行 rank 散写 `permuted_idx`/`fwd_idx`）+
固定物理核数（纯 Vector → 40）+ 块核 1:1 + 核内静态 `T.serial`**。
关键依据：(i) **可实现性是前置约束**——源 fetch-add 散写无 NPU 等价（`T.atomic_add` 无返回值），C5 容量否决，
C6 触发已知 codegen 缺口，C7 循环依赖；(ii) 本单元全部计算量 ≪10^5 整数 op（R3），C1 以三 kernel 顺序 launch
消源单 block fetch-add，且**逐位复现稳定序**（§1.6.1 E2）；(iii) C4（单核）作为结构最简降级备选。
**与基线结构差异一句话**：保留源「独立 scan 单元写 GM 表 + gather 读表」数据流，把源单 block 五相位拆成
**三 kernel GM 交接**，把 UB 运行期下标原子计数改为 **GM `bh` 块直方图**，把 thread0 串行前缀换成 **`T.cumsum`**，
把 fetch-add 散写换成 **确定性块并行稳定计数排序**。
**调研范围**：`references/algorithm-candidates.md`（无「counting sort / MoE permute scan」专条，参考
scan/分桶/gather 族条目）、pattern-library（layout/elementwise/traps-*、constants.md）、源算子 §0.3/§0.4、
本仓 examples（`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、
`testing/autotune/anneal/moe_grouped_gemm_anneal.py`、`examples/indexer/indexer_bwd.py`）、
`.agents/evolution/capability-gaps.md`（CG-2026-0027/0028）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1（SMEM 计数）→ GM `bh`；O2（单 block）→ 重设计（R1）；
O3（thread 并行）→ 重设计（R3）；O4（fetch-add）→ **意图承接、形态替换**为确定性块并行稳定计数排序（R2）；
O5（int64 前缀）→ 承接（cast）；O6（thread0 串行前缀）→ C2 硬件 scan；O7（GM 表交接）→ 保留（R1）。

**设计期 roofline（D-2）**：代表 shape = Kimi-K2 prefill（`numel=32768, E=384, K=8, NB=40, C=820`）：
- **流量项**：GM 流量 ≈ 950KB（R3 C1，按缓冲区尺寸×往返口径）÷ `~1.26 TB/s`（`CONST-mte2-degradation`，混合读写地板，
  tilelang 0.1.2+ed787bb / CANN 8.5.0）≈ **0.75 µs**（可忽略）。
- **发射项**：kernel B（单核）两趟 `T.serial(NB)`：趟 1（`T.copy(bh 行)` + `T.Parallel(E)` 累加）≈ 2 op/轮，
  趟 2（`T.Parallel(E)` 写 `slot_base` + `T.copy(bh 行)` + `T.Parallel(E)` 累加）≈ 3 op/轮，另 B3 表输出
  UB 暂存 + `T.copy` ≈ 5 op 一次性，合计 ≈ `(2+3)×40 + 5 = 205 op` × `~0.5µs/op`（`CONST-vector-launch-overhead`）
  ≈ **103 µs**；kernel A/C 为标量循环（发射按指令计，40 核并行、每核 ≈ `C×~6` ≈ 5K 指令，估算 ≪ 100µs）；
  合计发射项 ≈ **105 µs**。
- **容量项**：kernel B UB ≈ 8 buffer（7×int32 `(1,E/E+1)` + 1×int64 `ub_efo (1,E+1)`）≈ **13.5KB**（含
  `CONST-capacity-910B2C` auto-multi-buffer ~1.7x 膨胀余量 ≈ 23KB 仍 ≪ 192KB）；kernel A/C 无 UB；
  GM scratch ≈ 184KB（不占 UB）。
- **估算下界 = max(流量 0.75µs, 发射 ~105µs, 容量 ~0) ≈ 105 µs（发射项主导）**，一次性、与胞弟 gather（搬运
  `2×numel×H×2B ≈ 0.94 GB ≈ 0.8 ms`）同 stream 串行叠加时占比 ≈ 13%。`NB` 为可调旋钮（减小 NB 降发射项、
  但降 kernel C 并行度，权衡交 Stage 4）。
- **未文档化假设项**：kernel B「串行 over NB × `T.Parallel(E)`」的逐轮发射成本、以及 kernel A/C 标量循环的
  单核指令吞吐**未在 pattern-library 实测**（标注「未文档化假设 + 估算依据」）；该量**不影响主选/弃选裁决**
  （可行性已由构造成分 + 本仓先例硬裁定），仅影响成本特征 → 交**实验裁决计划**（下）标定；`NB` 为可调项
  （减小 `NB` 降 B 的发射项、但降 C 的并行度，权衡交 Stage 4）。

**实验裁决计划（三件套）**：

- **主选（进 Stage 3）**：三 kernel GM 交接——A 块直方图（GM 普通 RMW）、B 单核 `T.cumsum` 前缀 +
  `slot_base`、C 块并行运行 rank 散写（本设计 §3.3）。
- **备选（≤1，结构完整可实现）**：kernel C 退化为**单核全局顺序散写**（`T.Kernel(1)`，`T.serial(numel)` +
  全局 `run_gm[E]`，`slot=off_gm[e]+r`）；kernel A/B 与全部 buffer 形状、5 表契约不变。触发条件：
  主选 kernel C 的「双运行期下标的标量 GM 读写（`flat_ids[idx]` 取值 + `run_gm[b,e]` RMW + `permuted_idx[slot]`
  散写）」在本 shape 编译失败或出现 flaky SIGSEGV / 错值。
- **裁决计划**：① **代表 shape** = smoke（`numel=8,E=384,K=8`）、prefill（`numel=32768,E=384,K=8`）；
  ② **测量指标** = 首编编译结果 + 5 张表与 golden 逐元素精确一致（int32/int64，非容差）+ 反复运行稳定性；
  ③ **判定阈值** = 主选 kernel C 编译失败/数值错 → 立即切备选；若扫描单元墙钟 > gather 有效时间 10%
  方考虑进一步优化（`NB` 调节 / kernel B 的 `slot_base` 改 fp32 多行 cumsum 单 op）；
  ④ **回写路径** = 实测数字回写 §1.6.0 roofline 与 §5.6；触发降级/升级按 perf-feedback.md 受控路由。
  **shape 特化可行性**：本单元经 `lru_cache` 按 `(numel, num_experts, top_k)` 特化，可按 shape 分派主选/备选。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1+C2+C3 算法（三 kernel）。每项四要素：原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：`examples/moe_permute_nopad/_make_scan_kernel_nopad/verify_equiv.py`（**整数精确，非容差**）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | 排他前缀 = 含前缀 + 1-lane 移位（kernel B） | 源 thread0 串行 `off[0]=0; off[e+1]=off[e]+counts[e]` | `incl=T.cumsum(counts); off[0]=0; off[1:E+1]=incl[0:E]` | 整数恒等（`off[e+1]=incl[e]`）；机器验证 4000 例 0 违反；**shift guard**：省略移位（用含前缀当排他）4000 例中 3750 例失配，证明移位是负荷项 | 把 O(E) 标量串行压成 1 条 `VCumsumOp` + 1 次向量拷贝 |
| E2 | 稳定计数排序的散写：fetch-add → 确定性块并行 rank（kernel C） | 源 `slot=fetch_add(write_offsets[eid],1)`（时序相关、非确定） | `slot = off[e] + block_base[b,e] + r`，`r` = 块内该专家已出现次数（0 起）；`block_base[b,e]=Σ[b'<b]bh[b',e]`；合并为 `slot_base[b,e]=off[e]+block_base[b,e]`（E4） | **逐位复现稳定计数排序**（块序 × 块内位置序 = 全局位置序）；机器验证 4000 例 + 11 corner（含 numel=0/1、`numel<NB`、非整除、单专家、空专家）0 违反（`permuted_idx`/`fwd_idx`/`true_*`/`efo` 全覆盖） | 去除 fetch-add 依赖（NPU 无等价）；每 slot 只写一次，运行 rank O(numel) 且按块并行；与 golden 稳定序逐位一致 |
| E3 | int32 紧致偏移 cast 到 int64（kernel B） | 源 int64 直接累加 `s_offset` | `off32`（int32 前缀）；`efo[k]=int64(off32[k])` | 整数恒等（`numel<2^31`，无溢出）；机器验证 2000 例 0 违反 | 前缀在 int32 域完成（scan 原语 int32 已实测可用），仅最终 cast 到 int64 |
| E4 | `slot_base` 预计算（kernel B→C 交接） | kernel C 每次读 `off[e]` + `block_base[b,e]`（2 次 GM 读） | kernel B 预计算 `slot_base[b,e]=off[e]+block_base[b,e]`，kernel C 只读 1 次 | 整数恒等（同值预加，结合律） | kernel C 每位置少 1 次 GM 读（`numel` 次 GM 读削减） |
| E5 | `idx // top_k`（kernel C） | 源 `idx // T.int32(top_k)` | 同（`//`，**不用 `>>`**） | 非负整数除法，恒等 | 避免 int32 `>>` 的 CG-2026-0028(a) SIGSEGV；无收益损失 |

**优化结论**：采纳 **E1/E2/E3/E4/E5**。全部为**整数逐位精确恒等**（非容差内等价），**不引入任何 fp16/bf16
舍入误差**——精度标准为「5 张表逐元素精确一致」（§8.2）。

**等价性机器验证结果表**（`verify_equiv.py`，纯整数对照，torch CPU；metric = 最大绝对差）：

| 项 | 对照 | 最大度量值 | 违反 / 用例 | 结论 |
|----|------|-----------|-------------|------|
| E1 | 含前缀 cumsum + 1-lane 移位 vs 源串行排他前缀 | 0 | 0 / 4000 | EQUIV_PASS |
| E1（guard） | 省略 1-lane 移位（含前缀当排他）vs 正确前缀 | — | 3750 / 4000 失配 | GUARD_OK（确认移位负荷不可省） |
| E2 | 块并行两级稳定计数排序 vs 顺序 counting-sort golden（`permuted_idx`/`fwd_idx`/`true_sizes`/`true_offsets`/`efo` 全字段） | 0 | 0 / 4000 | EQUIV_PASS |
| E2（corner） | numel=0/1、`numel<NB`、非整除、单专家、多专家、全同专家、E=384×numel=32768 | 0 | 0 / 11 | EQUIV_PASS |
| E3 | int32 偏移 cast int64 vs int64 原生前缀 | 0 | 0 / 2000 | EQUIV_PASS |

无失败项（所有采纳项均为 EQUIV_PASS）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案全部循环与标量计算点（对象 = §1.6.0 选定、经 §0.6 重设计落地的 C1+C2+C3 三 kernel 算法）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 块直方图累加 `bh[b,e]+=1`（kernel A） | 源 SMEM 运行期下标 atomic | 无向量 scatter-add API（`T.atomic_add` 是单基址向量载荷 store，非逐 lane 散加；`docs/Tilelang.language/原子操作/T.atomic_add.md` §2）；且其标量载荷形态前端不可用（CG-2026-0029） | ❌ | **API 缺失**：NPU 无「逐 lane 不同目的地下标」的向量散加原语；已用 **GM 普通 RMW** `bh[core_id,e]=bh[core_id,e]+1` 承载（块核 1:1、每行私有 → 无需原子；Stage 2 探针 P5 全 shape 实测可编译）。计数本身与顺序无关（可任意次序），但**无向量等价指令**——查 `docs/Tilelang.language/` 全子目录（含 `原子操作/`、`索引与元素操作/`）无 scatter API |
| 2 | 全局 counts 列求和（kernel B） | 源无此步（SMEM 天然共享） | `T.serial(NB) × T.Parallel(E)` 向量加 | ✅ | — |
| 3 | 排他前缀 `off`（kernel B） | 源 thread0 串行 `T.serial(E)` | **`T.cumsum` (1,E) + 1-lane 移位**（E1） | ✅ | — |
| 4 | 输出表写 `true_sizes/true_offsets/efo`（kernel B） | 源逐元素循环 | `T.Parallel(E)` 逐元素写 **UB 暂存** + `T.cast`（`ub_efo` int64）→ `T.copy` 落 GM | ✅ | — |
| 5 | 块前缀 `block_base`/`slot_base`（kernel B） | 无（源为 fetch-add） | `T.serial(NB) × T.Parallel(E)` 运行累加 | ✅ | — |
| 6 | 块内运行 rank `r`（kernel C） | 源 fetch-add 隐式给出 | 无 | ❌ | **循环携带顺序依赖**：`r` 是块内该专家的出现序（`run[b,e]` 读后自增），**本质串行**、无法用归约或双缓冲消除（稳定序要求「先出现者先得 slot」）；且运行下标须落 **GM**（UB 运行期下标被 CG-2026-0028(d) 禁止）。属 checklist「tile 级顺序依赖（循环携带依赖）」类别 |
| 7 | 散写 `fwd_idx[idx]=slot`、`permuted_idx[slot]=idx//K`（kernel C） | 源散写 | 无 | ❌ | **API 缺失 + 运行期下标**：无向量 scatter API；`permuted_idx[slot]` 为运行期下标 GM 写（GM 允许，Stage 2 探针 P6 实测可编译；UB 则被 CG-2026-0028(d) 禁止）。逐位置只写一次，无热点可再向量化 |
| 8 | `e = flat_ids[idx]`（kernel C） | 源逐位置读 | `T.gather`/向量读 | ❌ | 作为 §6 的循环携带量（决定 `run[b,e]`/`slot`）必须逐位置取得；单次 GM 标量读（本仓动态标量 GM 读先例：胞兄 GEMM 核读 `true_offsets[expert_id]`），非逐元素热点 |
| 9 | `idx // top_k`（kernel C） | 源同 | 可 `T.Parallel` 逐元素（若单独成段） | ✅（已并入散写循环的向量操作数） | — |

**有向量形态但弃选者**：C5「`T.vsort` 排序驱动」（全向量）——弃选理由：(1) **UB 容量否决**：排序 numel=32768
需 `src+dst_value+dst_index` 三缓冲 ≈ 384KB > 192KB（`CONST-capacity-910B2C`）；(2) `T.vsort` 仅支持 fp16/fp32
（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3），int32 key 需 cast，键 `flat_ids*numel+idx` 需 < 2^24
才精确（prefill 下 `384×32768+32768 ≈ 12.6M < 2^24 ✓`，但大 E/numel 组合不安全）；(3) 排序复杂度 O(numel·log²numel)
（R3 ≈ 7.4M vs C1 的 81K）。

**向量化结论**：计数/前缀/输出等**与顺序无关**的部分已全部向量化（第 2/3/4/5 项：`T.cumsum` + `T.Parallel`）。
仅保留 3 处标量点（第 1/6/7 项：GM 块直方图 RMW、块内运行 rank、运行期下标散写），理由类别：
**API 缺失（无向量 scatter / 逐 lane 散加原语）** 与 **循环携带顺序依赖（稳定序要求，无法归约消除）**；
均**非逐元素热点型可替代项**（第 1/7 项每元素一次 GM 写、第 6 项为 O(1) 计数更新），由 §5.5 分核摊平（40 核）。
**无可向量化而未向量化的逐元素标量循环**；C5 的全向量路线因容量/精度被否决。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本单元为纯整数索引/计数，**无浮点、无矩阵、无 spatial tiling**；I/O layout 为契约（`flat_ids(numel,)`、
> 5 张表长度 `E/E/E+1/numel/numel`）不变。核内「布局」决策 = 索引缓冲的位置与分块轴（位置轴 numel）、
> 以及专家轴 E 的向量化方式。

**轴质量评分**（向量化轴 = 专家轴 E（kernel B 前缀/求和，`T.Parallel(E)`）；分块轴 = 位置轴 numel）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | E 对 int32 向量宽度的整除 | E∈{128,256,384}：128=4×32 ✓、256 ✓、384=12×32 ✓；`T.cumsum` 对任意 E 合法（尾部由原语处理） |
| 尾 lane 浪费率 | `(ceil(E/vw)*vw-E)/E` | 0（E 为 32 的整数倍 / 原语处理尾部） |
| 累加链形态 | counts/前缀为 stride=1 逐元素；scan 为 `VCumsumOp` | 逐元素无跨步系数问题；scan 原语内部处理 |
| repack 代价 | 无核内重排 | 0（一维 int32 缓冲，无需转置/NZ） |
| UB 容量影响 | kernel B：8 buffer = `ub_acc/ub_bh/ub_incl/ub_sb/ub_ts/ub_to` `[1,E]` + `ub_off` `[1,E+1]` int32 + `ub_efo` `[1,E+1]` int64 | ≈13.5KB ≪ 192KB（kernel A/C 无 UB） |
| 分块轴 numel | `C=ceildiv(numel,NB)` 静态；块核 1:1 | 位置轴仅用于块划分（非向量化轴），块内为串行 rank（顺序依赖） |

**候选矩阵**（计数排序数据流形态 + 索引缓冲位置）：

| # | 方案 | 关键决策 | 代价 / 收益 | 采纳 |
|---|------|---------|------------|------|
| L1 | **三 kernel GM 交接**（A：GM `bh` 普通 RMW 块直方图；B：`T.cumsum` 前缀 + `slot_base`；C：块并行运行 rank 散写） | 索引缓冲 `bh`/`slot_base`/`run` 驻 **GM**；kernel 边界隔离计数/前缀/散写 | 3 次 launch + GM scratch 驻留 184KB + UB ~13.5KB；**构造规避 CG-2026-0027/0028** | ✅ **主选** |
| L2 | 单 kernel 单 scope（UB 内计数 + scan + 标量消费） | UB 内完成全部 | 少 launch | ❌ 弃选（可行性：CG-2026-0027/0028，胞兄 Stage 3 实测证伪） |
| L3 | 内联进 gather 核（表不落 GM） | 同 block Vector→(gather) 交接 | 理论零表 GM | ❌ 弃选（可行性：CG-2026-0010/0026，官方样例复现） |
| L4 | `T.vsort` 全向量排序 | 排序即计数排序 | 全向量、无标量 | ❌ 弃选（容量：384KB>192KB；dtype：仅 fp16/fp32，见 §1.6.2） |
| L5 | host 侧算表 | CPU | — | ❌ 弃选（违反零 CPU 同步 + host 禁改输入，ascend-constraints.md §4） |
| L6 | 单核全局顺序散写（备选） | 1 核 + 全局 `run[E]` | 结构最简、少 `block_base` 往返，但 40× 串行 | ✅ **备选**（实验裁决计划） |

**布局决策结论**：选定 **L1（三 kernel GM 交接）** —— 索引缓冲 `bh[NB,E]`/`slot_base[NB,E]`/`run[NB,E]` 驻 **GM**
（跨 kernel 可读、规避 UB 运行期下标），kernel B 的向量化轴为**专家轴 E**（`T.Parallel(E)` + `T.cumsum`），
分块轴为**位置轴 numel**（`NB` 块，块核 1:1）。repack 路径：无。该结论已同步 §3.3 伪代码、§3.2 API 映射、
§4 内存规划与 §6 循环结构；L2/L3/L4/L5 的弃选理由均已附 `docs/` 路径或工具链实测证据，符合
negative-claim-evidence.md §2；L6 作为可实现的备选（实验裁决计划三件套见 §1.6.0 末）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（单一模式）。本单元为**三 kernel 纯 Vector 整数索引单元**：kernel A（GM 普通 RMW
块直方图 + 核内 `T.serial`）、kernel B（`T.Parallel`/`T.cumsum`/`T.copy`，含 UB 暂存）、kernel C（核内 `T.serial` 运行 rank +
运行期下标 GM 散写），**不含 `T.gemm`、不含跨 scope 交接**，故 Developer 形态完全可用。
调度参数指定 `programming_mode=developer`，与本结论一致。**无需 Expert 回退**。

### 2.2 选型理由

- 数据通路为**整数计数/前缀/散写**（GM 普通 RMW / `T.cumsum` / `T.Parallel` / `T.copy`），无浮点、无矩阵——
  按 AGENTS.md 的 Developer-mode MixCV 触发规则（Cube `T.gemm` + 至少一个 v-prefix op），**本单元不满足触发判据**
  （无 `T.gemm`），归为「整数 Vector 计数单元」，走 Developer。
- **规避三条工具链阻断**：Developer persistent + `T.gemm` + 整型索引段共存即崩（CG-2026-0010）——本单元**无 `T.gemm`**；
  跨引擎同 block 交接不可编译（CG-2026-0026）——本单元**无跨引擎交接**；单 scope「向量产出 → 标量消费」不可同步
  （CG-2026-0027）——本单元用**三 kernel GM 交接**隔离（标量只读跨 kernel GM）。
- Developer 用 `T.alloc_ub` + `T.Parallel`/`T.cumsum`/`T.copy` + GM 普通 RMW 表达，实现负担低。

### 2.3 模式影响

| 维度 | Developer（选定） |
|------|------------------|
| 内存分配 | kernel B：`T.alloc_ub`（counts/incl/off/bh/slot_base + 表输出暂存 `ub_ts`/`ub_to`/`ub_efo`）；GM scratch：`bh`/`slot_base`/`run`；GM 输出表 |
| 扫描原语 | `T.cumsum`（kernel B，单核一次）/ `T.Parallel`（kernel B）/ GM 普通 RMW（kernel A/C） |
| 跨 scope 交接 | **无 in-block 交接**（三 kernel 各自单 Vector scope；数据经 **GM** + kernel 间同 stream 顺序） |
| 同步 | 各 scope 内编译器自动同步；**不依赖 auto-sync 跨「向量产出 → 标量消费」**（该形态已弃用，CG-2026-0027）；kernel 间由 stream 顺序保证 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的优化后公式。逐步骤映射到 §1.6.2 的向量化结论（索引段为**整数** GM RMW / `T.cumsum` / `T.Parallel`，
> 无 v-prefix 浮点算子）。A* = kernel A；B* = kernel B；C* = kernel C。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| A1 | `bh[b, flat_ids[idx]] += 1`（`b=idx//C`） | 块直方图（GM **普通 RMW**，块核 1:1）（§1.6.2 #1） |
| B1 | `counts[e] = Σ_b bh[b,e]` | 列求和（向量加）（§1.6.2 #2） |
| B2 | `incl = cumsum(counts); off[0]=0; off[1:E+1]=incl[0:E]` | 硬件 scan + 1-lane 移位（E1）（§1.6.2 #3） |
| B3 | `true_sizes[e]=counts[e]`；`true_offsets[e]=off[e]`；`efo[e]=int64(off[e])` | 3 张表输出（E3）：`T.Parallel` 写 UB 暂存 → `T.copy` 落 GM（§1.6.2 #4） |
| B4 | `slot_base[b,e] = off[e] + Σ[b'<b] bh[b',e]` | 块前缀 + 专家偏移预加（E4）（§1.6.2 #5） |
| C1 | `e=flat_ids[idx]; r=run[b,e]; run[b,e]=r+1; slot=slot_base[b,e]+r` | 块内运行 rank（E2）（§1.6.2 #6） |
| C2 | `fwd_idx[idx]=slot`；`permuted_idx[slot]=idx//top_k` | 运行期下标散写（E2/E5）（§1.6.2 #7） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| A1 | `bh[b,e]+=1` | `e=flat_ids[base+c]; bh[core_id,e]=bh[core_id,e]+1` | GM **普通 RMW**（非原子），`b=core_id`，动态下标 | Developer |
| B1 | `counts=Σ_b bh[b,:]` | `T.copy(bh[b,0:E], ub_bh[0,0:E])` + `for e in T.Parallel(E): ub_acc[0,e]=ub_acc[0,e]+ub_bh[0,e]` | `(1,E)` 向量加（核内 `T.serial(NB)` 外层） | Developer |
| B2 | 含前缀 scan | `T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)` | 2D `(1,E)` int32；**唯一一次 scan** | Developer |
| B2 | 排他移位 | `ub_off[0,0]=0`；`T.copy(ub_incl[0,0:E], ub_off[0,1:E+1])` | 1-lane 移位切片 | Developer |
| B3 | 表输出 | `for e in T.Parallel(E): ub_ts[0,e]=ub_acc[0,e]; ub_to[0,e]=ub_off[0,e]`；`for k in T.Parallel(E+1): ub_efo[0,k]=T.Cast("int64", ub_off[0,k])`；随后 `T.copy(ub_ts[0,0:E], true_sizes[0:E])`、`T.copy(ub_to[0,0:E], true_offsets[0:E])`、`T.copy(ub_efo[0,0:E+1], efo[0:E+1])` | **UB 暂存 + `T.copy` 落 GM**（T.Parallel 体内禁直写 GM，§2.3）；含 int32→int64 cast | Developer |
| B4 | 块前缀 | `for b in T.serial(NB): for e in T.Parallel(E): ub_sb[0,e]=ub_acc[0,e]; T.copy(ub_sb[0,0:E], slot_base[b,0:E]); ub_acc[0,e]=ub_acc[0,e]+ub_bh[0,e]` | 运行累加（向量加） | Developer |
| C1 | 运行 rank | `e=flat_ids[idx]; r=run[core_id,e]; run[core_id,e]=r+1; slot=slot_base[core_id,e]+r` | 标量 GM 读/RMW（循环携带） | Developer |
| C2 | 散写 | `fwd_idx[idx]=slot`；`permuted_idx[slot]=idx//top_k` | 运行期下标 GM 写 | Developer |

> 说明：本单元无浮点逐元素运算，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix 算术 API；整数向量化仅用
> `T.Parallel` / `T.cumsum` / `T.copy`（`T.Parallel` 体内只读写 UB 暂存，GM 一律经 `T.copy`）。

### 3.3 计算伪代码

> `T.Tensor` 第二位置参数直接传 dtype。**scratch `bh`/`run` 由 host 以 `torch.zeros` 零初始化后传入**
> （kernel 参数，非 `out_idx`）；5 张输出表由 host `torch.empty` 分配并传入。本单元 = 三 kernel 链式。
> `NB`/`C`/`E`/`top_k` 均为编译期常量。

```python
import functools
import tilelang
import tilelang.language as T

# 编译期常量（代表 shape: numel=32768, E=384, top_k=8）
#   P     = NPUUtils.get().get_aicore_num() * 2      # 纯 Vector 物理核数（§5.5 实查 = 40）
#   NB    = min(P, numel)                             # 位置块数 = 核数
#   C     = ceildiv(numel, NB)                        # 每块位置数（静态边界）
_PASS_CFG = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}


# ---- kernel A: 块直方图（GM 普通 RMW，块核 1:1）----------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_count_kernel(numel, num_experts, nb, cblk):
    E = num_experts

    @T.prim_func
    def _count_main(flat_ids: T.Tensor([numel], "int32"),
                    bh: T.Tensor([nb, E], "int32")):      # 零初始化 GM scratch
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            base = core_id * cblk
            for c in T.serial(cblk):                       # 静态边界
                if base + c < numel:                       # 普通 if 守卫（非 T.if_then_else）
                    e = flat_ids[base + c]                 # 标量 GM 读（运行期下标）
                    bh[core_id, e] = bh[core_id, e] + 1    # 普通 RMW（块核 1:1、行私有，无需原子）

    return _count_main


# ---- kernel B: 跨块求和 + 前缀 + 输出 + slot_base（单核）--------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_prefix_kernel(numel, num_experts, nb, cblk, top_k):
    E = num_experts

    @T.prim_func
    def _prefix_main(bh: T.Tensor([nb, E], "int32"),
                     true_offsets: T.Tensor([E], "int32"),
                     true_sizes: T.Tensor([E], "int32"),
                     efo: T.Tensor([E + 1], "int64"),
                     slot_base: T.Tensor([nb, E], "int32")):
        with T.Kernel(1, is_npu=True) as (_cid, _subid):   # 单核一次
            ub_acc  = T.alloc_ub((1, E), "int32")
            ub_bh   = T.alloc_ub((1, E), "int32")
            ub_incl = T.alloc_ub((1, E), "int32")
            ub_off  = T.alloc_ub((1, E + 1), "int32")
            ub_sb   = T.alloc_ub((1, E), "int32")
            # 表输出 UB 暂存（T.Parallel 体内禁直写 GM，须先写 UB 再 T.copy）
            ub_ts   = T.alloc_ub((1, E), "int32")           # true_sizes 暂存
            ub_to   = T.alloc_ub((1, E), "int32")           # true_offsets 暂存
            ub_efo  = T.alloc_ub((1, E + 1), "int64")       # efo 暂存（int64）

            # B1: counts = column-sum of bh（与顺序无关，可向量化）
            for e in T.Parallel(E):
                ub_acc[0, e] = 0
            for b in T.serial(nb):                          # 静态边界
                T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

            # B2: 唯一一次 scan（int32 / (1,E) 已设备实测可用）
            T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)
            ub_off[0, 0] = 0
            T.copy(ub_incl[0, 0:E], ub_off[0, 1:E + 1])     # E1: 1-lane 移位（T.Parallel 禁下标变换）

            # B3: 三张表输出（先写 UB 暂存，再 T.copy 落 GM；T.Parallel 体内禁直写 GM）
            for e in T.Parallel(E):
                ub_ts[0, e] = ub_acc[0, e]                  # counts
                ub_to[0, e] = ub_off[0, e]                  # off
            for k in T.Parallel(E + 1):
                ub_efo[0, k] = T.Cast("int64", ub_off[0, k])  # E3: int32 -> int64
            T.copy(ub_ts[0, 0:E], true_sizes[0:E])
            T.copy(ub_to[0, 0:E], true_offsets[0:E])
            T.copy(ub_efo[0, 0:E + 1], efo[0:E + 1])

            # B4: slot_base[b,e] = off[e] + sum_{b'<b} bh[b',e]（E4）
            for e in T.Parallel(E):
                ub_acc[0, e] = ub_off[0, e]
            for b in T.serial(nb):
                for e in T.Parallel(E):
                    ub_sb[0, e] = ub_acc[0, e]
                T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

    return _prefix_main


# ---- kernel C: 确定性块并行运行 rank 散写（块核 1:1）------------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_scatter_kernel(numel, num_experts, nb, cblk, top_k):
    E = num_experts

    @T.prim_func
    def _scatter_main(flat_ids: T.Tensor([numel], "int32"),
                      slot_base: T.Tensor([nb, E], "int32"),
                      run: T.Tensor([nb, E], "int32"),        # 零初始化 GM scratch
                      fwd_idx: T.Tensor([numel], "int32"),
                      permuted_idx: T.Tensor([numel], "int32")):
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            base = core_id * cblk
            for c in T.serial(cblk):                           # 静态边界
                idx = base + c
                if idx < numel:
                    e = flat_ids[idx]                          # 标量 GM 读（运行期下标）
                    r = run[core_id, e]                        # 块内运行计数（块核 1:1，无争用）
                    run[core_id, e] = r + 1
                    slot = slot_base[core_id, e] + r           # E2: 确定性 slot
                    fwd_idx[idx] = slot
                    permuted_idx[slot] = idx // top_k          # E5: //，非 >>

    return _scatter_main


# ---- 公共入口：同 stream 顺序跑 A -> B -> C --------------------------------
@functools.lru_cache(maxsize=32)
def _make_scan_kernel_nopad(numel, num_experts, top_k):
    P = 40
    nb = min(P, numel) if numel > 0 else 1
    cblk = (numel + nb - 1) // nb if numel > 0 else 1
    kA = _scan_count_kernel(numel, num_experts, nb, cblk)
    kB = _scan_prefix_kernel(numel, num_experts, nb, cblk, top_k)
    kC = _scan_scatter_kernel(numel, num_experts, nb, cblk, top_k)

    def run(flat_ids, bh, true_offsets, true_sizes, efo, slot_base, run_buf,
            fwd_idx, permuted_idx):
        kA(flat_ids, bh)
        kB(bh, true_offsets, true_sizes, efo, slot_base)
        kC(flat_ids, slot_base, run_buf, fwd_idx, permuted_idx)
        return fwd_idx, permuted_idx

    return run
```

> **注 1（规避 CG-2026-0027/0028）**：三 kernel 均**不在标量上下文读本 scope 向量 op 写入的 UB 值**——
> kernel A/C 的标量只读 **GM**（`flat_ids`/`slot_base`/`run`，由 host 或上一 kernel 提供）；无 `T.alloc_local`、
> 无 int32 `>>`（`idx//top_k`）、无标量 `T.if_then_else`（用普通 `if`）、每个 kernel **至多一次 `T.cumsum`**、
> 不用运行期下标读 UB。kernel A/C 的运行期下标读写仅针对 **GM**（Stage 2 探针 P5/P6 实测全 shape 可编译）。
> **注 2（输出承接源语义）**：`fwd_idx`/`permuted_idx` 互逆且为**稳定序**，与源 host golden 逐位一致（§1.6.1 E2）。
> **注 3（零初始化）**：`bh`/`run` 必须 host 侧 `torch.zeros` 预清零（源 `permuted_idx` 亦 `torch.zeros`）。
> **注 4（scan int32）**：`T.cumsum` int32/`(1,E)` 路径设备实测可用（胞兄 `repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`
> ASSERT D `[1,3,6,10]`）；int32 scan 以 scalar instruction lower（低效但 E≤384 单核一次可忽略）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| GM 普通 RMW（`bh[core_id,e]=bh[core_id,e]+1`，运行期下标） | Stage 2 最小 kernel 探针 P5（代表 shape `numel=32768/E=384/NB=40/C=820` 编译通过）；块核 1:1 → 无需原子 | 已验证（块核 1:1 专属；非本设计依赖 `T.atomic_add`） |
| ~~`T.atomic_add(dst, value)`（标量载荷）~~ | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.1 声明 `src` 支持 scalar，但 §2.4 示例仅向量载荷；实测标量载荷前端失败 | **不可用**（CG-2026-0029，`DiagnosticError: 'int' object has no attribute 'buffer'`）；本设计改用普通 RMW |
| `T.cumsum(src, dst, dim=1, reverse=False)`（int32, `(1,E)`） | `docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1（int32 √）；`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` L186；胞兄设备实测 | 已验证（int32/`(1,E)`；编译器 warning「scalar instruction low efficiency」） |
| `T.Parallel(N)` 整型向量化（kernel B 求和/表暂存/运行累加） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.2.1（支持 `+ - * /`、比较）；§2.3「下标不能做变换」「循环体须用 alloc 的 UB buffer」 | 已验证（索引未变换；体内只写 UB 暂存，GM 经 `T.copy`；Stage 2 探针 P1/P2 实测） |
| `T.copy` 切片（GM→UB / UB→GM，含 1-lane 移位与 int32→int64 表写出） | `docs/Tilelang.language/内存操作/T.copy.md`；`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`；胞兄 kernel A/B | 已验证（表输出 UB 暂存 + `T.copy` 落 GM 见 Stage 2 探针 P7） |
| `T.Kernel(n, is_npu=True)`（1D 核；A/C = `NB`，B = 1） | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；胞兄 kernel B | 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | 已验证 |
| `T.Cast("int64", x)`（int32→int64） | `docs/Tilelang.language/数据类型转换操作/` 下转换算子；源 L84 `T.Cast(T.int64, ...)` | 已验证（源同款；若 `T.Parallel` 内 cast 受限，回退 int64 累加路线见 §9.4 R-4） |
| ~~fetch-add（`tl_atomic_add_offset`）~~ | 源 `_atomic_helper.h`（CUDA 定制 extern）；`T.atomic_add` 降 `hivm::StoreOp`（**无返回值**） | **NPU 无等价**（能力缺口，§9.4 R-1），本设计不依赖 |
| ~~int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum`~~ | — | **不可用**（CG-2026-0028 a/b/c/e/f）；本设计全部回避 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（三 kernel 均为 1D `T.Kernel`） | 一维核 + 核内任务计算 |
| GEMM M/N 需为 block 整数倍 | 否（本单元无矩阵） | — |
| L0C 容量（128KB） | 否（本单元无 L0C） | — |
| 物理核数限制 | 是（`NB=min(P,numel)`，P=40 实查） | kernel A/C `T.Kernel(NB)`、块核 1:1、核内静态 `T.serial(C)`；kernel B 单核（§5.5） |
| host 侧禁止改动输入 | 是（需 host 分配 scratch/输出） | host 仅**分配** scratch（`torch.zeros`）与输出（`torch.empty`），不改动 `flat_ids` 真实内容；无 padding（ascend-constraints.md §4） |
| 一维核超发被串行调度 | 是（源单 block×1024 thread 的形态） | 固定核数 `NB=min(P,numel) ≤ P`，无超发（§0.6 R3） |
| `T.Parallel` 下标禁变换 / 循环体禁直写 GM | 是 | kernel B 用切片 `T.copy` 完成 1-lane 移位；求和/表暂存索引未变换；**GM 写一律经 UB 暂存 + `T.copy`**（B3，§T.Parallel.md §2.3） |
| 同 scope「向量产出 → 标量消费」不可同步（CG-2026-0027） | 是 | 三 kernel GM 交接：标量只读跨 kernel **GM**，不读本 scope 向量产出的 UB |
| 跨引擎同 block 交接不可编译（CG-2026-0026） | 否 | 本单元无跨引擎交接 |
| int32 `>>` / `alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum`（CG-2026-0028） | 是 | 全回避：`//` 替代 `>>`；无 `alloc_local`；普通 `if` 替代 `T.if_then_else`；scan 只用 `(1,E)` 单行且每 kernel 一次 |
| UB 运行期下标读/写（CG-2026-0028(d)） | 是（源 SMEM 运行期下标） | 运行期下标只用于 **GM**（`bh`/`run`/`slot_base`/`permuted_idx`），UB 全为静态索引 |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| 承载方式 | 单 block（grid=1, 1024 thread）5 相位 | **三 kernel GM 交接**（A 计数 / B 前缀+输出 / C 散写） | §0.6 R1 |
| 直方图 | SMEM `atomic_add`（运行期下标） | GM `bh[NB,E]` 普通 RMW（`T.atomic_add` 标量载荷不可用，块核 1:1 故免原子），块核 1:1 | §0.5（CG-2026-0028d/0029） |
| 表输出 | 逐元素写 GM（thread 并行） | `T.Parallel(E)` 写 UB 暂存 + `T.copy` 落 GM | §3.2 B3（`T.Parallel.md` §2.3） |
| 前缀扫描 | thread0 `T.serial(E)` int64 | kernel B 硬件 `T.cumsum` + 1-lane 移位（int32），efo cast int64 | §0.6 R1 / E1 / E3 |
| 散写 slot | `tl_atomic_add_offset` **fetch-add** | 确定性 `off[e]+block_base[b,e]+块内 rank` | §0.6 R2 / E2 |
| 分核 | 1 block × 1024 thread | `NB=min(P,numel)` 块核 1:1 + 核内 `T.serial(C)`；B 单核 | §0.6 R3 / §5.5 |
| 同步 | `T.sync_threads()` 相位 | 各 scope auto-sync + kernel 间 stream 顺序 | §7 |
| 输出分配 | `out_idx=[]`（host `torch.empty/zeros`） | 保留：host 分配 + 传入（`bh`/`run` 零初始化） | §4.1/§4.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` / `_tile_scheduler_kernel.py` | **极高（同族整数索引算子，含 Stage 3 实测）** | 两 kernel GM 交接形态；kernel A `T.cumsum`(1,E) + 1-lane 移位；单核 scan + 固定核数 kernel B + 核内 `T.serial`；CG-2026-0027/0028 规避清单 |
| `examples/TileOPs/tests/ops/test_moe_permute_nopad.py` | **极高（本单元 golden 依据）** | `_ref_moe_permute_nopad`（顺序计数排序）；L105–L111 顺序无关一致性校验（`perm_h[fwd_idx]==hidden_states[idx//K]`） |
| `examples/indexer/indexer_bwd.py` | 中（GM 运行期下标写先例，**向量载荷原子**） | `T.atomic_add(dIndexK[target_k_idx, 0], dk_temp_ub[j,0], [head_dim])`（运行期下标 GM scatter）L119 |
| `examples/deepseek_v32/sparse_mla_bwd.py` | 中（GM 运行期下标写先例，**向量载荷原子**） | `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])` L257/288 |
| `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 高（scan + 持久分核） | `T.cumsum(..., dim=1)` 用法；`T.serial(num_local_tasks)` 静态核内串行；2D 多行 fp32 cumsum |
| `testing/autotune/anneal/moe_grouped_gemm_anneal.py` | 中（调度器反例） | `tile_scheduler_kernel_npu` 用 `T.Kernel(max_tiles)` **超发核**（反例，本设计按 §5.5 修正） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `flat_ids` | `(numel,)` | int32 | 每位置专家号（`= topk_ids.flatten()`；host 侧 view，不改内容） |
| `bh` | `(NB, num_experts)` | int32 | **GM scratch（输入侧）**，host `torch.zeros` 零初始化 |
| `run` | `(NB, num_experts)` | int32 | **GM scratch（输入侧）**，host `torch.zeros` 零初始化 |
| `slot_base` | `(NB, num_experts)` | int32 | **GM scratch（kernel B 产 / C 消费）**，host `torch.empty` |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `true_offsets` | `(num_experts,)` | int32 | 每专家紧致起始 `off[e]`（kernel B 写） |
| `true_sizes` | `(num_experts,)` | int32 | 每专家 token 数 `counts[e]`（kernel B 写） |
| `expert_first_token_offset` | `(num_experts + 1,)` | int64 | 非填充排他前缀 `off`（kernel B 写，int32 cast） |
| `fwd_idx` | `(numel,)` | int32 | flat 位置 → 紧致 slot（kernel C 写） |
| `permuted_idx` | `(numel,)` | int32 | slot → token 行（kernel C 写；交胞弟 gather 单元） |

> **对接契约（本单元产出 / 胞弟 gather 消费）**：① `permuted_idx` int32；② `fwd_idx` 与 `permuted_idx` **互逆**
> 且为**稳定序**（与 op golden 一致）；③ `numel`/`top_k`/`num_experts` 为两单元共享编译期常量；
> ④ **kernel A → B → C → 胞弟 gather** 同 stream 顺序 launch（无 in-block 同步、无 CPU 同步）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 所在 kernel / 用途 |
|-----------|-------|-------|----------|--------------------|
| `ub_acc` | `(1, E)` | int32 | UB | B：counts 累加 / slot_base 运行累加 |
| `ub_bh` | `(1, E)` | int32 | UB | B：`bh` 逐块载入 |
| `ub_incl` | `(1, E)` | int32 | UB | B：含前缀 scan 输出 |
| `ub_off` | `(1, E+1)` | int32 | UB | B：排他前缀 |
| `ub_sb` | `(1, E)` | int32 | UB | B：`slot_base` 行写出 |
| `ub_ts` | `(1, E)` | int32 | UB | B：`true_sizes` 暂存（→ `T.copy` 落 GM） |
| `ub_to` | `(1, E)` | int32 | UB | B：`true_offsets` 暂存（→ `T.copy` 落 GM） |
| `ub_efo` | `(1, E+1)` | int64 | UB | B：`efo` int64 暂存（int32→int64 cast，→ `T.copy` 落 GM） |
| `bh` | `(NB, E)` | int32 | **GM** | A 写 / B 读：块直方图 |
| `slot_base` | `(NB, E)` | int32 | **GM** | B 写 / C 读：`off[e]+block_base` |
| `run` | `(NB, E)` | int32 | **GM** | C 读写：块内运行计数 |

### 4.4 内存搬运路径

```
[kernel A · Vector · NB 核（块核 1:1）]
GM[flat_ids(numel) int32] --T.serial(C) 逐元素 普通 RMW--> GM[bh(NB,E) int32]
                                                                    │ （kernel 边界，同 stream 顺序）
                                                                    ▼
[kernel B · Vector · grid=1]
  GM[bh] --T.copy--> UB[ub_bh] --T.Parallel(E) 累加--> UB[ub_acc=counts]
        --T.cumsum(dim=1)--> UB[ub_incl] --1-lane 移位--> UB[ub_off]
        --T.Parallel(E)/(E+1)--> UB[ub_ts/ub_to/ub_efo] --T.copy--> GM[true_sizes(E)] /
             GM[true_offsets(E)] / GM[efo(E+1) int64]（表输出经 UB 暂存，禁 T.Parallel 直写 GM）
  GM[bh] + UB[ub_off] --串行 b 运行累加--> GM[slot_base(NB,E) int32]
                                                                    │ （kernel 边界，同 stream 顺序）
                                                                    ▼
[kernel C · Vector · NB 核（块核 1:1）]
  GM[flat_ids] + GM[slot_base] + GM[run] --串行运行 rank--> GM[fwd_idx(numel) int32]
                                                            GM[permuted_idx(numel) int32]
                                                                   │ （同 stream 顺序 launch）
                                                                   ▼
              [胞弟 _make_gather_kernel_nopad：perm_h[slot]=hidden_states[permuted_idx[slot]]]
```

> **层级说明**：本单元为**整数 Vector 计数/索引**，数据路径只有 **GM → UB**（`bh`）与 **UB → GM**（表、`slot_base`）
> 及 **GM 标量读写**（`flat_ids`/`run`/`permuted_idx`），**无 L1/L0A/L0B/L0C**（无 Cube 矩阵计算）。

### 4.5 UB 内存预算（Developer，num_experts=384，int32 `(1,E)` + `ub_efo` int64）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| **kernel B（唯一用 UB 的 kernel）** | | | |
| `ub_acc` | (1,384) | int32 | 1536 |
| `ub_bh` | (1,384) | int32 | 1536 |
| `ub_incl` | (1,384) | int32 | 1536 |
| `ub_off` | (1,385) | int32 | 1540 |
| `ub_sb` | (1,384) | int32 | 1536 |
| `ub_ts` | (1,384) | int32 | 1536 |
| `ub_to` | (1,384) | int32 | 1536 |
| `ub_efo` | (1,385) | int64 | 3080 |
| B 小计 | | | ≈ **13836** |
| kernel A / C | — | — | **0**（仅 GM） |

> 按 `CONST-capacity-910B2C` 的 auto-multi-buffer 膨胀系数（~1.7x）复算 ≈ **23.5KB ≪ 192KB** ✓。
> `bh`/`slot_base`/`run` 位于 **GM**（各 `NB×E×4 = 40×384×4 = 61.4KB`，驻留峰值合计 ≈ **184KB**，不占 UB）。

### 4.6 动态轴定义

无动态轴：`numel / num_experts / top_k / NB / C` 均为编译期常量（JIT 特化）。运行时变化的仅为
`flat_ids` 的**数据值**（不改变 shape）。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)   # 无 out_idx（host 分配传入）
def _scan_count_kernel(numel, num_experts, nb, cblk): ...
def _scan_prefix_kernel(numel, num_experts, nb, cblk, top_k): ...
def _scan_scatter_kernel(numel, num_experts, nb, cblk, top_k): ...

# 公共入口（同 stream 链式）
def _make_scan_kernel_nopad(numel, num_experts, top_k): ...
```

> `pass_configs`（同胞兄、保证 UB 单版本、避免 buffer 重排干扰）：`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`、
> `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False`。无 `compile_flags`（纯整数，无 fast-math/bf16 需求）。
> 按 `(numel, num_experts, top_k)` 经 `functools.lru_cache` 特化。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**整数 Vector 计数/索引/散写（三 kernel）**（无浮点、无矩阵、无 spatial tiling）。判定依据：全部计算为
`counts`（直方图）、前缀 scan、块前缀、运行 rank 与散写——落 Vector（AIV）单元。kernel A/C 多核，kernel B 单核。
本单元**无 `T.gemm`、无 Cube scope**。

### 5.2 Block 划分

本单元**无空间 block 划分**（无 M/N/K tile）。工作粒度：
- kernel A/C：位置轴按 `NB = min(P, numel)` 块划分，`C = ceildiv(numel, NB)` 每块位置数；**块核 1:1**。
- kernel B：单核，核内 `T.serial(NB)` 遍历块 + `T.Parallel(E)` 向量化专家轴。

### 5.3 约束分析

- **对齐约束**：E∈{128,256,384} 为 32 的整数倍（int32 向量宽度整除，尾 lane 0%）；`T.cumsum` 对任意 E 合法。
- **UB 容量**：kernel B ≈13.8KB（膨胀后 ≈23.5KB）≪ 192KB ✓（`CONST-capacity-910B2C`）。
- **非整除**：`C = ceildiv(numel, NB)`，尾块由 `if base + c < numel` 守卫；`numel < NB` 时仅前 `numel` 块非空。
- **静态边界**：`NB`、`C` 均为编译期常量 ✓（core-split-strategy.md §1 要素③要求）。

### 5.4 注意事项（尾块/边界处理）

- **尾块**：`idx ≥ numel` 由普通 `if` 守卫跳过（kernel A 不加；kernel C 不加写）。
- **空专家**（`counts[e]=0`）：`off[e]==off[e+1]`，段为空；无位置写入该段（§1.6.1 E2 corner 覆盖）。
- **`numel=0`**：无块有位置，5 张表全 0（`bh`/`run` 零初始化；`off` 全 0）；corner 验证覆盖。
- **`numel < NB`**：`NB=numel`，`C=1`，仅在位块启动（其余核空转，边界守卫）。
- **E 任意 / top_k 任意**：`T.cumsum` 处理任意 E；`idx//top_k` 用整数除（非 `>>`）。
- **kernel 顺序**：A 写 `bh` → B 读 `bh` 写表/`slot_base` → C 读 `slot_base` 写 `fwd_idx`/`permuted_idx`，
  必须**同一 stream 顺序 launch**（§7.2）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 / §2.1（依据 docs/开发指南.md §3.3）。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **返回值 = 20**。
  查询代码与实测值：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20`
  （Ascend910B2C，2026-10-14 本机实查）。本单元为**纯 Vector 单元**（无 Cube）→ 物理核数取实查值翻倍
  **`P = get_aicore_num() * 2 = 40`**（标准 core-split-strategy.md §1②「纯 Vector 算子核数翻倍」）。
- **逻辑核数**：
  - **kernel A**：逻辑核数 = `NB = min(P, numel)`（每核一块位置）。
  - **kernel B**：逻辑核数 = **1**（一次性 E 轴 scan + 表输出，单核）。
  - **kernel C**：逻辑核数 = `NB = min(P, numel)`。
  代表 shape：smoke（`numel=8`）→ `NB=8`；prefill（`numel=32768`）→ `NB=40`。
- **规模判定**：
  - kernel A/C：逻辑核数 `NB ≤ P = 40` **恒成立**（`NB=min(P,numel)`）→ 结论「**无需适配**」；
    prefill 下 `NB=P=40` 恰好为物理核整数倍（1×），无负载不均。
  - kernel B：逻辑核数 1 ≤ 40 → 「**无需适配**」（单核一次性 scan）。
- **分核方案**：
  - kernel A/C：`T.Kernel(NB, is_npu=True)`（`NB=min(P,numel)`，编译期常量），**块核 1:1**（`b = core_id`）；
    核内 `for c in T.serial(C)` 静态边界。代表值：prefill `NB=40, C=820`；smoke `NB=8, C=1`。
  - kernel B：`T.Kernel(1, is_npu=True)`，核内 `for b in T.serial(NB)`（静态边界）× `T.Parallel(E)`。
  - 负载均衡：kernel A/C 每核位置数差 ≤1（`C` 分块）；各专家分布偏斜时，块内运行 rank 工作量 ∝ 块内位置数
    （均衡），总工作量 O(numel) 摊到 40 核。
  - **无死区/无跨核写竞态**：`bh[b,·]`/`run[b,·]`/`slot_base[b,·]` 每行仅由拥有该块的核访问；
    `permuted_idx[slot]` 各 slot 全局唯一 → 无写竞态。

### 5.6 承载方式取舍（三 kernel GM 交接 vs 单 scope vs 内联）⭐（本单元核心论证）

| 维度 | 单 scope UB（弃选） | 内联进 gather 核（弃选） | **三 kernel GM 交接（主选）** |
|------|---------------------|--------------------------|-------------------------------|
| kernel launch | 1 | 0（与 gather 同 launch） | **3 次**（A/B/C；与 gather 同 stream 串行） |
| GM 往返 | 无表；但 UB 承载全部 | 无表；各核重算 | `bh`+`slot_base`+`run` 驻留 ≈ 184KB + 5 表写（按缓冲尺寸×往返口径 ≈950KB） |
| 同步/数据流 | **同 scope 标量消费向量产出**（不可同步） | 需 in-block 跨引擎交接 | **无 in-block 同步**；kernel 边界隔离；stream 顺序保证 |
| 工具链可实现性 | ❌ 不可实现：CG-2026-0027/0028（胞兄 Stage 3 实测证伪） | ❌ 不可实现：CG-2026-0010/0026（官方样例复现） | ✅ 构造上规避全部已知阻断；构件均有实测/先例（GM 动态下标 RMW / `T.cumsum` int32(1,E) / 动态标量 GM 读 / UB 暂存 + `T.copy`） |
| 计算冗余 | 各核重扫 flat_ids | 调度与 gather 同核 | kernel A/C 各扫 flat_ids 一遍（2 遍，无冗余重扫） |
| 结论 | **弃选（可行性）** | **弃选（可行性）** | **选定** |

**论证要点**：单 scope 与内联两种形态在本工具链**均不可实现**（前者 CG-2026-0027/0028，后者 CG-2026-0010/0026）。
三 kernel GM 交接是**唯一同时满足「无跨引擎交接、无同 scope 标量消费向量、无 codegen 崩溃」的形态**，其构件
（GM 动态下标 RMW、`T.cumsum` int32 `(1,E)`、核内 `T.serial` + `T.Parallel(E)`（写 UB 暂存）、动态标量 GM 读、
UB 暂存 + `T.copy` 落 GM）
逐一有实测/先例。多出的 2 次 launch 与 184KB scratch 驻留为实现性支付的可接受成本（§1.6.0 roofline：
发射项 ≈ 105µs，相对胞弟 gather 的 ≈0.8ms 占比 ≈ 13%）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| kernel A 核级 | 固定核数网格 | `T.Kernel(NB, is_npu=True)` | 块核 1:1（§5.5） |
| kernel A 块内 | 串行 | `T.serial(C)`（静态边界） | 逐位置普通 RMW（§1.6.2 #1） |
| kernel B 核级 | 单核网格 | `T.Kernel(1, is_npu=True)` | 一次性 scan + 表输出 |
| kernel B 跨块 | 串行 | `T.serial(NB)`（静态边界） | 块求和 / 块前缀运行累加 |
| kernel B 专家轴 | 向量化 | `T.Parallel(E)` + `T.cumsum` | 求和/表暂存/运行累加（§1.6.2 #2/#3/#4/#5；GM 经 `T.copy` 落） |
| kernel C 核级 | 固定核数网格 | `T.Kernel(NB, is_npu=True)` | 块核 1:1 |
| kernel C 块内 | 串行（循环携带） | `T.serial(C)`（静态边界） | 运行 rank + 散写（§1.6.2 #6/#7；循环携带依赖） |
| 元素级 | 无（已消除/整体替代） | — | 无逐元素浮点标量循环 |

### 6.2 循环伪代码

见 §3.3（kernel A：`T.serial(C)` 普通 RMW；kernel B：`T.serial(NB)` 外层 + `T.Parallel(E)` 内层 + 单次 `T.cumsum`
+ 表输出 UB 暂存 `T.copy`；kernel C：`T.serial(C)` 运行 rank 散写）。

### 6.3 流水线优化

- kernel A/C 的块内串行为顺序/散加，无跨迭代独立任务可流水（且核内已按块并行）。
- kernel B 的跨块串行是运行累加（循环携带依赖），不可流水。
- **不引入 `T.Pipelined`**（无 K 循环/无多级搬运；且 CG-2026-0001 记录 Pipelined 体内条件构造崩溃）。
- 跨 kernel：A → B → C → 胞弟 gather **同 stream 顺序执行**（顺序由 stream 保证，无需显式同步）。

### 6.4 尾块处理

- kernel A/C 尾块：`if base + c < numel` 普通 `if` 守卫。
- 空专家：`counts[e]=0` → 该段无位置写入（无特殊分支）。
- `numel=0`：无守卫通过，全表 0。
- 不使用 host padding、谓词零填充或逐元素 select。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**kernel 内自动同步 + kernel 间 stream 顺序**（Developer）。
- kernel A/B/C 各自**单 Vector scope**，scope 内 MTE（`T.copy`）/ Vector（`T.Parallel`/`T.cumsum`）依赖由编译器自动同步；
- **关键约束（CG-2026-0027）**：**不在标量上下文中读取本 scope 内向量 op 写入的 UB 值**——本设计已彻底避免
  （kernel A/C 的标量只读 **GM**；kernel B 无标量消费向量产出）；
- **不使用任何手动同步原语**（不使用 `T.sync_block_set/wait`、`T.set_flag/wait_flag`，不依赖 `T.pipe_barrier`）。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| kernel A 内普通 RMW 之间 | 无依赖（不同 `bh` 行，块核 1:1） | 每核独占行，无跨核 / 同核冲突 |
| kernel B 内 `T.copy`/`T.Parallel`/`T.cumsum` 之间 | 编译器自动同步 | 单 Vector scope 内向量→向量依赖 |
| kernel A → kernel B（`bh`） | **kernel 间同 stream 顺序**（无显式同步原语） | A 写 `bh` 完成、B 再读 |
| kernel B → kernel C（`slot_base`） | **kernel 间同 stream 顺序** | B 写 `slot_base` 完成、C 再读 |
| kernel C → 胞弟 gather（`permuted_idx`） | **kernel 间同 stream 顺序** | C 写 `permuted_idx` 完成、gather 再读 |

> **不引入 `T.sync_block_*` 的依据**：CG-2026-0026（codegen `syn_instr_mode` 与编译器 `sync_instr_mode` 不匹配，
> 官方 `examples/mixcv/mixcv_mixkernel.py` 复现）。本设计以「scratch/表落 GM + kernel 间顺序」替代同 block 交接。

### 7.3 pass_configs 配置

```python
pass_configs = {
    NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}
```

> 与胞兄 Stage 3 一致：保证 UB 缓冲单版本、避免 buffer 重排干扰 kernel B 的切片拷贝。三 kernel 均为纯整数
> Vector kernel、无 `T.gemm`。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（纯 Python/torch **顺序计数排序**），**不得复刻** §0.6 的
> 三 kernel 算法（保证验证独立性）。移植源仓测试的参考实现 `_ref_moe_permute_nopad`，见
> `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`（L31–L70）。

```python
import torch

def golden_scan(flat_ids, num_experts, top_k):
    """§0.1 语义：flat_ids[numel] -> (true_offsets, true_sizes, efo, fwd_idx, permuted_idx)。

    顺序（稳定）计数排序 —— 与源实现同语义（顺序 write_ptr），独立于 NPU 算法。
    """
    E = int(num_experts)
    K = int(top_k)
    ids = flat_ids.detach().to(torch.int64).cpu().tolist()
    numel = len(ids)

    counts = [0] * E
    for e in ids:
        counts[e] += 1

    off = [0] * (E + 1)                        # int64 exclusive prefix
    for e in range(E):
        off[e + 1] = off[e] + counts[e]

    write_ptr = list(off[:-1])
    permuted_idx = [0] * numel
    fwd_idx = [0] * numel
    for idx, e in enumerate(ids):
        slot = write_ptr[e]
        permuted_idx[slot] = idx // K
        fwd_idx[idx] = slot
        write_ptr[e] += 1

    return (
        torch.tensor(off[:-1], dtype=torch.int32),   # true_offsets
        torch.tensor(counts, dtype=torch.int32),     # true_sizes
        torch.tensor(off, dtype=torch.int64),        # expert_first_token_offset
        torch.tensor(fwd_idx, dtype=torch.int32),    # fwd_idx
        torch.tensor(permuted_idx, dtype=torch.int32),  # permuted_idx
    )
```

> **顺序无关等价性**：op 测试（`test_moe_permute_nopad.py` L105–L111）只校验 `perm_h[fwd_idx]==hidden_states[idx//K]`
> 与 `offsets/sizes` 精确；本设计产出**稳定序**，与上 golden 逐位一致（§1.6.1 E2），故可直接逐元素精确比对。

### 8.2 精度标准

| dtype | 标准 |
|-------|------|
| int32 / int64 | **逐元素精确一致**（无浮点参与，无 atol/rtol；§1.6.1 采纳项 E1–E5 机器验证全 0 违反） |

> 注：胞弟 gather 的 `perm_h` 输出精度（bf16/fp16 精确搬运）属其单元，不在本单元。

### 8.3 L0 门槛测试计划

> **主线（交付形态）= 三 kernel 扫描单元 + 与 `golden_scan` 逐元素精确比对**（5 张表，int32/int64 逐位相等）。

| 用例 | 参数 | 验证点 |
|------|------|-------|
| L0-1 smoke | `numel=8, E=384, K=8`（manifest smoke 的 topk 部分） | 5 表逐元素相等；空专家段为空；`NB=8,C=1` |
| L0-2 均匀 | `numel=64, E=4, K=2` | 5 表逐位一致；`bh`/`run` 零初始化正确 |
| L0-3 偏斜（大量单次专家） | `numel=256, E=256, K=8`，`flat_ids[i]=i` | 每专家 1 token；`slot=fwd` 稳定序 |
| L0-4 单专家 | `numel=64, E=4, K=2`，全落专家 0 | 段唯一、无空洞 |
| L0-5 空专家混合 | `E=8`，部分 `counts=0`（含首/尾专家） | 重复 `off` 边界；段空 |
| L0-6 奇偶 shape | `E=5`（非 2 幂）、`numel` 非整除 `NB` | `T.cumsum` 尾块、`//` 正确 |
| L0-7 代表大 shape | `numel=32768, E=384, K=8`（Kimi-K2 prefill） | `NB=40, C=820`；无死锁；5 表逐位一致 |
| L0-8 多次运行稳定性 | 同 L0-1/L0-7 重复运行 6 次 | 5 表**稳定**（无 CG-2026-0027 型未同步读） |
| L0-9 对接契约 | 抽取 L0-1/L0-7 输出 | int32/int64 shape 与 §4.2 契约一致；`permuted_idx[fwd_idx[idx]]==idx//K` |
| L0-10 三 kernel 顺序 | A→B→C 同 stream | C 读到 B 写的 `slot_base`、B 读到 A 写的 `bh`（无旧值/未初始化） |

> 注：§1.6.1 的 `verify_equiv.py` 已在**设计期**对 L0-2/3/4/5/6/7 的算法等价性做 CPU 位精确预验证。

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **NPU 无 fetch-add**（`T.atomic_add` 无返回值，`docs/Tilelang.language/原子操作/T.atomic_add.md` §3；CG-2026-0030）→ 源散写形态直译不可行，
  改确定性块并行 rank（§0.6 R2）。
- **`T.atomic_add` 标量载荷前端不可用**（CG-2026-0029，`DiagnosticError`）→ kernel A 直方图改**普通 RMW**（块核 1:1，无需原子）。
- **UB 运行期下标**（CG-2026-0028(d)）→ 运行期下标仅用于 GM（Stage 2 探针 P5/P6 全 shape 实测可编译）。
- **`T.Parallel` 循环体内禁直写 GM 张量**（`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3）→ 表输出经 UB 暂存 + `T.copy`（Stage 2 探针 P1/P2/P7）。
- **同 scope「向量产出 → 标量消费」不可同步**（CG-2026-0027）→ 三 kernel GM 交接规避。
- **int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum` / 一 kernel 两次 `T.cumsum`**
  （CG-2026-0028）→ 全部回避（`//`、无 `alloc_local`、普通 `if`、`(1,E)` 单行单次 scan）。
- **跨引擎同 block 交接不可编译**（CG-2026-0026）→ 本设计无跨引擎交接。
- `T.cumsum` int32 路径以 scalar instruction lower（编译器 warning「low efficiency」）→ E≤384 单核一次，成本可忽略。
- **kernel B 发射项**（`T.serial(NB) × T.Parallel(E)`）× 两趟 ≈ `NB×5` 向量 op ≈ 105µs（§1.6.0 roofline）——
  相对胞弟 gather（≈0.8ms）占比 ≈13%；`NB` 为可调旋钮（Stage 4）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **标量消费向量产出**（CG-2026-0027） | 标量读 `T.cumsum`/`T.Parallel` 写入 UB 的值并消费 | 同编译产物多运行不一致 | 三 kernel GM 交接：标量只读跨 kernel GM |
| `T.atomic_add` 标量载荷（CG-2026-0029） | `T.atomic_add(dst, 1)`（单元素） | 前端 `DiagnosticError: 'int' object has no attribute 'buffer'` | 块核 1:1 场景改**普通 RMW**；通用场景用向量载荷 |
| `T.Parallel` 体内直写 GM 张量（`T.Parallel.md` §2.3） | `for e in T.Parallel(E): gm[e]=ub[0,e]` | `NPU IR opt failed … crash backtrace` | 先写 UB 暂存，再 `T.copy` 落 GM（含 int64 cast 经 `ub_efo`） |
| int32 `>>` / `alloc_local`+`serial` / 标量 `T.if_then_else`（CG-2026-0028） | 误用 | device_codegen SIGSEGV | `//` 替代 `>>`；免 `alloc_local`；普通 `if` |
| 运行期下标读 UB（CG-2026-0028d） | 把 `bh`/`run` 放 UB | 污染相邻结果 | scratch 全放 **GM** |
| 同 kernel 两次/多行 int32 `T.cumsum`（CG-2026-0028e/f） | kernel B 想二次 scan 或 2D 多行 | SIGSEGV / 垃圾值 | 唯一一次 `(1,E)` 单行 scan |
| kernel 边界顺序反了 | C 先于 B、或异性 stream | C 读 `slot_base` 旧值/未初始化 | 严格同 stream 顺序 A→B→C |
| `bh`/`run` 未零初始化 | host 用 `torch.empty` | 计数/rank 错 | host `torch.zeros` 预清零 |
| 块前缀累积漏 `off[e]` | `slot_base=block_base`（忘加 `off[e]`） | slot 全局错位 | `slot_base[b,e]=off[e]+Σ[b'<b]bh[b',e]`（E4） |
| 尾块越界 | `numel` 非整除 `C` | 越界读写 | `if base+c < numel` 守卫 |
| `idx//top_k` 写成 `>>` | — | SIGSEGV | 用整数除 `//`（E5） |

### 9.3 特殊场景处理

- **偏斜分布**：块内运行 rank 工作量 ∝ 块内位置数（均衡），总 O(numel)。
- **空专家 / 全空输入**：段为空 / 无位置写入（§8.3 L0-4/L0-5）。
- **极小 shape**：`NB=min(P,numel)`（smoke `NB=8`），正确性优先。
- **`numel=0`**：存在 `NB=1,C=1` 的退化块，守卫失效 → 5 表全 0。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（能力缺口，正向登记）**：NPU **无 fetch-add** 原子原语（源 `tl_atomic_add_offset` 返回旧值无等价 API，
  CG-2026-0030）与 **`T.atomic_add` 标量载荷不可用**（CG-2026-0029）——两者层：**Frontend API**。
  本设计以**普通 RMW**（kernel A，块核 1:1）+ **确定性块并行 rank**（kernel C）绕开（不阻塞）。按 AGENTS.md
  Gap Rule 已登记进 `.agents/evolution/capability-gaps.md`：**CG-2026-0030**（缺 fetch-add）与
  **CG-2026-0029**（`T.atomic_add` 标量载荷，doc §2.1 与实现不一致），`status: open`。
- **R-2（GM 运行期下标标量访问，Stage 2 已探针确认）**：含三处：
  (a) kernel A 的**普通 RMW `bh[core_id,e]=bh[core_id,e]+1`**（`e=flat_ids[idx]` 运行期下标）；~~原标量载荷
  `T.atomic_add(bh[core_id, flat_ids[idx]], 1)` 已弃用~~（CG-2026-0029）；
  (b) kernel C 的运行期下标 GM 标量 RMW（`run[core_id,e]` 读后自增）；
  (c) kernel C 的运行期下标 GM 标量散写（`permuted_idx[slot]=idx//top_k`）。
  三者均由 Stage 2 最小 kernel 探针在**代表 shape 实测编译通过**（P5/P6）；均因每 (block,expert) 计数私有于单核而
  **无需原子语义**。仍建议 Stage 3 首编反复运行验证稳定性；命中问题 → 切 §1.6.0 备选（kernel C 单核 + 全局 `run[E]`）。
- **R-3（表输出 UB 暂存，Stage 2 已探针确认）**：kernel B 的 `T.Parallel(E)` 内 `T.Cast("int64", ...)` 写
  `ub_efo`（int64 UB 暂存）再 `T.copy` 落 GM——该形态由 Stage 2 探针 P7 在代表 shape 实测编译通过；
  若 `T.copy` 的 UB→GM int32→int64 dtype 转换受限，回退为分离转换步骤（`docs/Tilelang.language/内存操作/T.copy.md` 跨 dtype）
  或 int64 域累加 `T.cumsum`（仅当 int64 scan 实测可用）。不影响语义。
- **R-4（性能，交 Stage 4）**：`NB` 旋钮（B 发射项 vs C 并行度）与 kernel B 的 `slot_base` 备选实现
  （fp32 多行 `T.cumsum` on `(E, NB)` 单 op）——按 §1.6.0 实验裁决计划标定；不影响主选/弃选裁决。
- **R-5（语义，交 Stage 2 复核）**：本设计采**稳定序**（与 op golden 一致）；若下游 unpermute 依赖**非稳定**
  atomic 序的隐式契约，则需复核（依据：op 测试仅校验顺序无关一致性，manifest 仅约束 shape/dtype）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_nopad/_make_scan_kernel_nopad/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1）
└── (Stage 3 将产出) _make_scan_kernel_nopad.py
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核/承载方式取舍 + §8 验证） |
| `verify_equiv.py` | 已完成（重跑通过） | 等价性验证（E1/E2/E2-corner/E3 全 EQUIV_PASS + E1 shift guard） |
| 算子实现（Stage 3） | 待实现 | 三 kernel 扫描单元（§3.3 伪代码）：`_scan_count_kernel` / `_scan_prefix_kernel` / `_scan_scatter_kernel` |
| 集成对接（Stage 5） | 待实现 | 与胞弟 `_make_gather_kernel_nopad`（读 `permuted_idx`）同 stream 顺序 launch |

### 10.3 命名规范

- 项目目录：`examples/moe_permute_nopad/`
- 算子目录：`examples/moe_permute_nopad/_make_scan_kernel_nopad/`
- 规范文件名：`_make_scan_kernel_nopad.py`（Stage 3 产文件；含三 kernel 工厂 + 公共入口 `_make_scan_kernel_nopad`）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）+ L0 五表逐元素比对（§8.3 主线）
3. ⬜ kernel A/B/C（§3.3 伪代码），先 smoke（`numel=8`）再代表大 shape（`numel=32768`）
4. ⬜ 与胞弟 gather 核同 stream 顺序集成（对接契约 §4.2）

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py`（`_make_scan_kernel_nopad`，L43–L119）
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`
- 源 op / 测试 / bench / manifest / workload：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`、`examples/TileOPs/tileops/workloads/moe.py`
- 胞兄（同族整数索引算子，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/_tile_scheduler_kernel.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/verify_equiv.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`
- 同类实现：`examples/indexer/indexer_bwd.py`、`examples/deepseek_v32/sparse_mla_bwd.py`、
  `examples/elementwise/atomic_add.py`、`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、
  `testing/autotune/anneal/moe_grouped_gemm_anneal.py`
- API 文档：`docs/Tilelang.language/原子操作/T.atomic_add.md`、`docs/Tilelang.language/排序操作/T.vcumsum.md`、
  `docs/Tilelang.language/排序操作/T.vsort.md`、`docs/Tilelang.language/索引与元素操作/T.gather.md`、
  `docs/Tilelang.language/内存操作/T.copy.md`、`docs/Tilelang.language/内存操作/T.alloc_ub.md`、
  `docs/Tilelang.language/编译器提示操作/T.Parallel.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`、
  `.agents/skills/_shared/standards/hardware-cost-model.md`、`docs/开发指南.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0010/0026/0027/0028/0029/0030）
