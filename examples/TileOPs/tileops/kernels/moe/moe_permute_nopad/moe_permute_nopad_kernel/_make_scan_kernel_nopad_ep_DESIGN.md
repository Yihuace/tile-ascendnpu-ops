# _make_scan_kernel_nopad_ep 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`tileops/kernels/moe/permute_nopad.py` 中的
> `_make_scan_kernel_nopad_ep`（GPU/CUDA target，MoE no-pad token permute 的 **Phase-1 计数排序扫描单元 —— EP
> 专家并行 / `expert_map` 变体**）。本单元目标：`target="npuir"`（Ascend NPU）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`
> （含 `_make_scan_kernel_nopad_ep`，L134–L227）。
> **强参照（胞兄，已冻结并含 Stage 3 实测的「非 EP 版」）**：
> `examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md`、
> `examples/moe_permute_nopad/_make_scan_kernel_nopad/_make_scan_kernel_nopad.py`、
> `examples/moe_permute_nopad/_make_scan_kernel_nopad/verify_equiv.py`。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> 等价性机器验证（D-1）：`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/verify_equiv.py`
> （E1 / E1-guard / E2-EP / E2-EP-guard / E2-EP-corner / E3 / F1 全部通过，见 §1.6.1 结果表）。
> 设计期探针（D-3）：`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_primitives.py`
> （EP 两级间接 + 守卫）、`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_prefix.py`
> （`E_local` 专家轴）。

**本单元的一句话定位**：`_make_scan_kernel_nopad_ep` 是 MoE no-pad token 重排的**EP（专家并行）计数排序扫描单元**——
在非 EP 版（`_make_scan_kernel_nopad`）基础上叠加 `expert_map[E_global]` 语义：把每个 `flat_ids` 位置的目标专家号
`g` 经**下拉重映射（pull-down remap）** `le = expert_map[g]` 映射到**本地专家域** `[0, E_local)`；`le < 0` 的
**非本地对**被排除（`fwd_idx[idx] = -1` 哨兵），本地对按稳定计数排序得到：① 每本地专家 token 数
`true_sizes[E_local]`；② 紧致起始 `true_offsets[E_local]` 与非填充排他前缀 `expert_first_token_offset[E_local+1]`
（int64）；③ 正向映射 `fwd_idx[T*K]` 与 gather 索引 `permuted_idx[T*K]`。**全程 int32/int64 索引、无浮点**；
产物由胞弟 gather 单元（`_make_gather_kernel_nopad`）消费。**本单元 = 非 EP 版 + EP 重映射/过滤语义**。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **承载形态整段承接非 EP 版「三 kernel GM 交接」**：**kernel A**（`_scan_count_ep_kernel`，Vector，多核）产
   **每块（block, 本地专家）直方图** `bh[NB, E_local]`；**kernel B**（`_scan_prefix_ep_kernel`，Vector，单核）做跨块列求和 +
   **唯一一次 `T.cumsum`(1, E_local)** 产 `off` 与前缀/输出表 `true_sizes/true_offsets/expert_first_token_offset`
   （int64 cast）+ `slot_base[NB, E_local]`；**kernel C**（`_scan_scatter_ep_kernel`，Vector，多核）逐块运行 rank 散写
   `permuted_idx/fwd_idx`。三 kernel 同 stream 顺序 launch，**同 scope 内无「向量产出 → 标量消费」**
   （规避 CG-2026-0027）、无 in-block 跨引擎交接（规避 CG-2026-0026）。**只把专家轴长度由 `E` 换成 `E_local`**。
2. **EP 语义在扫描内侧实现为「两级 GM 间接 + 守卫」（pull-down remap）**：`le = expert_map[flat_ids[idx]]`（两级标量
   GM 读链），`le >= 0` 才进入计数/散写；**计数、前缀、rank 全在本地专家域 `E_local` 完成**——不需要全局直方图再
   聚合（§1.6.0 R1 候选 C-EP1 弃选）。设计期探针实测该形态在代表 shape（`numel=32768, E_global=384, E_local=128,
   NB=40, C=820`）位精确、可编译。
3. **非本地对（`le < 0`）= -1 哨兵经 UB 暂存 + 一次动态 extent `T.copy` 落 `fwd_idx`**，**不进入 rank/slot 计算**；
   `permuted_idx` 仅由本地对写入（`[0, M_local)`），host 侧零初始化兜底，`[M_local, numel)` 保持 row-0 占位（源 op 同款）。
4. **GM 写原语承接非 EP 版 Stage 3 实测结论**（多核标量 GM store 会随机丢写 → 只能用 MTE3 `T.copy` 或**向量载荷**
   `T.atomic_add`），并在本变体新增探针实测「两级间接 + 守卫」形态下这些原语依旧可靠（§0.6 R2/R4）。

> **与非 EP 版的关系（差异清单，其余全部复用）**：① 新增输入 `expert_map[E_global]` int32；② 专家轴由 `E` → `E_local`；
> ③ kernel A/C 增加一级 `expert_map` GM 读与 `le >= 0` 普通 `if` 守卫；④ 非本地对写 -1 哨兵、不写 `permuted_idx`；
> ⑤ 表长度 `E_local`（`efo` 为 `E_local+1`）。**算法族（三 kernel GM 交接 / 确定性块并行稳定计数排序 /
> 分核 40 + 块核 1:1 / `T.cumsum` 前缀 / 工具链规避）逐条不变**。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = **`_make_scan_kernel_nopad_ep`**（Phase-1 扫描单元的 EP 变体，源 L122–L215）。
> 同文件的 `_make_scan_kernel_nopad`（非 EP 版，L43–L119）已由胞兄单元完成迁移（含 Stage 3 实测），本单元
> **整段承接其结论**并只对 EP 差异做重设计；`_make_gather_kernel_nopad`（Phase-2 gather）由胞弟单元设计，不在本单元。

### 0.1 源算子语义（做什么）

**数学语义**（纯整数计数排序；无浮点）。输入 `flat_ids[numel]`（int32，`numel = T*K`，`idx = t*K + k`）、
`expert_map[E_global]`（int32，`g ∈ [0, E_global)`，值域 `{-1} ∪ [0, E_local)`）。定义**下拉重映射**：

$$
le(idx) = expert\_map[\,flat\_ids[idx]\,],\qquad
\mathcal{L} = \{\,idx : le(idx) \ge 0\,\} \quad \text{(local set)}
$$

在**本地专家域**上构造**稳定计数排序**：

$$
counts[l]=\#\{\,idx\in\mathcal{L}: le(idx)=l\,\},\qquad
off[0]=0,\quad off[l+1]=off[l]+counts[l],\quad l\in[0,E_{local})
$$

$$
fwd\_idx[idx]=
\begin{cases}
-1, & le(idx)<0 \quad \text{(non-local sentinel)}\\
off[le]+rank(idx), & le(idx)\ge 0
\end{cases}
\qquad
permuted\_idx[\,off[le]+rank(idx)\,]=idx//K
$$

其中 `rank(idx)` = 满足 `j < idx` 且 `le(j) == le(idx)` 的 `j` 个数（**稳定序**，与源 host 序一致）。
输出表：

- `true_sizes[l] = counts[l]`（int32，长度 `E_local`）
- `true_offsets[l] = off[l]`（int32，紧致起始，长度 `E_local`）
- `expert_first_token_offset[l] = off[l]`（int64，非填充排他前缀；长度 `E_local+1`）
- `fwd_idx[idx] = off[le]+rank(idx)` 或 `-1`（int32，flat 位置 → 紧致 slot / 非本地哨兵）
- `permuted_idx[slot] = idx//K`（int32，slot → token 行；供 gather 读 `hidden_states`）

记 `M_local = off[E_local]`（本地对总数，`≤ numel`）。

- **规约语义**：计数/前缀为**整数精确**运算（无累加顺序/舍入问题）。`expert_first_token_offset` 用 int64 累加
  （源 L189）；其余全 int32。**整数计数排序的等价性可要求逐位精确**。
- **dtype 语义**：全程 int32（`flat_ids`、`expert_map`、`true_*`、`permuted_idx`、`fwd_idx`、`counts/off32`）+
  int64（`expert_first_token_offset`）。无 dtype 提升/回落、无浮点。
- **边界语义**：
  - `le(idx) < 0`（非本地）→ `fwd_idx[idx] = -1`，**不写 `permuted_idx`**；源 `permuted_idx` 由 host `torch.zeros`
    分配（源 L364），故未写 slot（`[M_local, numel)`）保持 0。
  - `counts[l] = 0`（空本地专家）→ `off[l] = off[l+1]`，该本地专家段为空，映射仍唯一确定。
  - `M_local = 0`（全非本地，如 `expert_map` 全 -1）→ 全 `counts=0`、`off` 全 0；无 slot 被写；全 `fwd_idx=-1`。
  - `numel = 0` → 全表 0；无 slot 被写。
  - **`E_local` 的取值域**：`num_experts_local = int((expert_map >= 0).sum().item())`（源 L313）。**`E_local ≥ 1`**
    是 kernel B 的可编译前提（`E_local = 0` 时 `T.alloc_ub((1, 0))` 触发 MLIR `strides must not be zero`，
    设计期探针实测，§9.4 R-1）；host 侧须保证 `E_local ≥ 1`（或走 `M_local = 0` 退化分支）。
  - **`fwd_idx` 的专家内顺序**：源 EP 同样用 `tl_atomic_add_offset`（fetch-add）取 slot，**专家内顺序取决于并发
    atomic 执行序**，属**非确定**；op 契约（`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` L105–L111）只校验
    **顺序无关**一致性 `perm_h[fwd_idx[flat]] == hidden_states[flat//K]`（以及 `offsets/sizes` 精确）。
    故本设计选择的确定性**稳定序**满足契约，且与 §8.1 EP golden 逐位一致（§1.6.1 E2-EP）。
  - `top_k = K`：`permuted_idx[slot] = idx // K`（编译期常量，整数除法，非负）。

**语义保持基线**：§8.1 golden 以本节为唯一依据（纯 Python/torch **顺序** EP 计数排序：先 `expert_map` 重映射，
本地域计数/前缀，非本地写 -1），**不得复刻** §0.6 的 NPU 算法（三 kernel 块并行直方图 + 运行 rank）。

**host 侧语义**：本单元 **无 host 预处理/后处理**；`numel / E_global / E_local / top_k` 均为编译期 Python 常量经
`functools.lru_cache` 特化（源 op `_get_kernel` 按 `(T,K,E,H,dtype,device)` 缓存；`E_local` 由 host 从
`expert_map` 计数得出）。host 侧仅**分配** scratch/输出张量（`bh`/`run` 用 `torch.zeros` 零初始化；
`permuted_idx` 用 `torch.zeros` 兜底未写 slot），不改动任何输入内容。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `flat_ids` | `(numel,)` | int32 | 输入（= `topk_ids.flatten()`）：每位置的目标**全局**专家号 |
| `expert_map` | `(num_experts_global,)` | int32 | 输入：全局专家号 → 本地专家号（`-1` = 不在本 rank），**EP 变体新增** |
| `expert_first_token_offset` | `(num_experts_local + 1,)` | int64 | **输出**：本地域非填充排他前缀 |
| `true_offsets` | `(num_experts_local,)` | int32 | **输出**：每本地专家紧致起始 |
| `true_sizes` | `(num_experts_local,)` | int32 | **输出**：每本地专家真实 token 数 |
| `permuted_idx` | `(numel,)` | int32 | **输出（内部交接）**：slot → token 行（仅 `[0, M_local)` 被写） |
| `fwd_idx` | `(numel,)` | int32 | **输出**：flat 位置 → 紧致 slot / 非本地 `-1` |
| `write_offsets` | `(num_experts_local,)` | int32 | **内部 scratch**（源为 fetch-add 目标） |

源 `@tilelang.jit(out_idx=[], ...)`：输出由 host 侧 `torch.empty/zeros` 分配；`_SCAN_THREADS = 1024`，
`T.Kernel(1, threads=threads)`（**单 block**）。`compile_flags=["-O3","-include", _ATOMIC_HELPER_H]`
（extern 声明 `tl_atomic_add_offset`）。**与非 EP 版差异**：多一个 `expert_map` 入参；表长度 `E_local`；计数/散写各多一步
`expert_map` 查表与 `>= 0` 守卫。

### 0.3 实现算法解读（怎么算）

源 EP kernel 结构（源 L135–L213）：`@tilelang.jit(...)` 工厂，`T.Kernel(1, threads=1024)` **单 block**；
`tx = T.get_thread_binding()`。SMEM：`s_counts[E_local]`（int32）、`s_offset[E_local+1]`（int64）、
`slot_buf[1]`（`T.alloc_local`）。五个串行相位：

- **Step 1**：`for i in serial(ceildiv(E_local,threads)): if idx<E_local: s_counts[idx]=0` + `sync_threads`。
- **Step 2**：`for i in serial(ceildiv(numel,threads))`：`g=flat_ids[idx]; le=expert_map[g]; if le>=0: T.atomic_add(s_counts[le],1)`
  （**两级 GM 查表 + 守卫 + 逐元素散加**到 SMEM） + `sync_threads`。
- **Step 3**：`if tx==0: s_offset[0]=0; for e in serial(E_local): s_offset[e+1]=s_offset[e]+int64(s_counts[e])`
  （**thread 0 串行**排他前缀，int64） + `sync_threads`。
- **Step 4**：向量化写 `efo[0:E_local+1]=s_offset`；`true_offsets[e]=int32(s_offset[e])`、`true_sizes[e]=s_counts[e]`、
  `write_offsets[e]=int32(s_offset[e])`。
- **Step 5**：`for i in serial(ceildiv(numel,threads))`：`g=flat_ids[idx]; le=expert_map[g];`
  `if le<0: fwd_idx[idx]=-1; else: slot_buf[0]=tl_atomic_add_offset(&write_offsets[0], le, 1);`
  `permuted_idx[slot]=idx//top_k; fwd_idx[idx]=slot`。

**计算步骤分解**（源码每条计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P1 | `s_counts[idx]=0` | — | SMEM `s_counts` | 清零 |
| P2 | `le=expert_map[flat_ids[idx]]; if le>=0: T.atomic_add(s_counts[le],1)` | `flat_ids`,`expert_map` | SMEM `s_counts` | $le(idx)$ + $counts[l]$ |
| P3 | `s_offset[0]=0; s_offset[e+1]=s_offset[e]+int64(s_counts[e])`（thread 0 串行） | `s_counts` | SMEM `s_offset`(int64) | $off[0]=0,\ off[l+1]=off[l]+counts[l]$ |
| P4a | `expert_first_token_offset[idx]=s_offset[idx]` | `s_offset` | GM `efo`(int64) | $efo_l=off[l]$ |
| P4b | `true_offsets[e]=int32(s_offset[e])` | `s_offset` | GM `true_offsets` | $true\_offsets_l=off[l]$ |
| P4c | `true_sizes[e]=s_counts[e]` | `s_counts` | GM `true_sizes` | $true\_sizes_l=counts[l]$ |
| P4d | `write_offsets[e]=int32(s_offset[e])` | `s_offset` | GM `write_offsets` | 散写指针初值 $off[l]$ |
| P5a | `le=expert_map[flat_ids[idx]]; if le<0: fwd_idx[idx]=-1` | `flat_ids`,`expert_map` | GM `fwd_idx` | 非本地哨兵 $-1$ |
| P5b | `slot=tl_atomic_add_offset(&write_offsets[le],1)`（仅 `le>=0`） | `flat_ids`,`expert_map`,`write_offsets` | `slot_buf` | $slot=off[l]+rank$ |
| P5c | `permuted_idx[slot]=idx//top_k` | `idx`, `slot` | GM `permuted_idx` | $permuted\_idx[slot]=idx//K$ |
| P5d | `fwd_idx[idx]=slot`（仅 `le>=0`） | `idx`, `slot` | GM `fwd_idx` | $fwd\_idx[idx]=off[l]+rank$ |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[flat_ids(numel,)] ─┐
                      ├─(level-1 读 g)─▶ GM[expert_map(E_g,)] ─(level-2 读 le)─┐
                      │                                                        │
                      ▼                                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ P2: if le>=0 ── 逐元素散加(atomic) ──▶ SMEM[s_counts(E_l) i32] │
        └──────────────────────────────────────────────────────────────┘
                                   │ P3 (thread0 串行 int64)
                                   ▼
                            SMEM[s_offset(E_l+1) int64]
                                   │ P4
        ┌──────────────────────────┼──────────────────────────────────┐
        ▼                          ▼                                  ▼
  GM[efo(E_l+1) i64]   GM[true_offsets/true_sizes(E_l) i32]   GM[write_offsets(E_l) i32]
                                                                      │ P5b fetch-add
                                                                      ▼
                le>=0 ? slot = fetch_add(write_offsets[le]) : (fwd=-1)
                                                                      │
                                          ┌───────────────────────────┴───────────────┐
                                          ▼                                           ▼
                          GM[permuted_idx(numel) i32]              GM[fwd_idx(numel) i32]
```

**循环与并行结构**：**单 block**（grid=1）+ 1024 线程。每线程处理 `numel/1024` 个位置；step3 仅 thread 0 工作其余空等；
step1/2/4/5 为 thread 级并行 + `T.sync_threads()` 相位间隔。**未使用 `T.Parallel` / `T.Pipelined`**——纯 thread 级
scalar/branch/atomic 循环。

**host 侧逻辑**（源 op `forward`，属调用方，非本单元）：`flat_ids = topk_ids.flatten().contiguous()`；
`num_experts_local = int((expert_map >= 0).sum().item())`；分配 `permuted_idx = torch.zeros(numel)`（**零初始化**，
兜底 `[M_local, numel)` 未写 slot）、其余 `torch.empty`；调用 scan 后调用 gather（`perm_h` 恒为 `numel` 行，
`[M_local, numel)` 为 row-0 占位）。**无 host 侧数据预处理**。

**编译配置**：`@tilelang.jit(out_idx=[], compile_flags=["-O3","-include", _ATOMIC_HELPER_H])`。
### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM 驻留 `s_counts`/`s_offset`（`T.alloc_shared`） | 免 GM 往返、读低延迟 | 块内共享计数/前缀 | shared memory | 硬件强相关（NPU 有 UB/GM 等价） |
| O2 | **单 block**（grid=1）一次性完成计数→前缀→散写 | 免跨 block 归并 / 全局同步 | 一个 block 内 `sync_threads` 相位 | 大 block + 廉价 block 内同步 | 硬件强相关（NPU 一维核 + 无块内 barrier 可用） |
| O3 | thread 级并行（1024 线程摊 numel） | 摊平计数/散写的串行度 | 每 thread 一位置 | warp/thread 模型 | 硬件强相关（NPU 是 SIMD lane，非 thread） |
| O4 | `tl_atomic_add_offset` **fetch-add** 散写 | 免每位置 rank 前缀，并发直接拿 slot | 原子取旧值 + 自增（CUDA 定制 extern） | 全局/共享原子 RMW（有返回值） | **硬件强相关且 NPU 无等价**（`T.atomic_add` 无返回值） |
| O5 | int64 排他前缀（仅 thread 0 串行） | 免溢出 | 64-bit 累加 | 无 | 算法层可移植 |
| O6 | thread 0 串行前缀（`T.serial(E_local)`） | 简单、`E_local` 小（≤384）成本可接受 | 单线程线性扫描 | 无 | 算法层可移植（NPU 改硬件 scan） |
| O7 | `permuted_idx`/`fwd_idx` 落 GM 供 Phase-2 gather 读 | 两 kernel 解耦（计数与搬运分离） | 独立 scan kernel + GM 交接 | GPU 多 kernel 顺序执行 | 模式可移植（NPU GM 表交接，胞兄已证） |
| **O-EP1** | **`expert_map` 内联查表**（共享表读 `expert_map[g]`） | 免物化「重映射后的 ids 数组」 | 原位两级索引（1 次额外 GM 读/位置） | 共享/全局小表随机读 | **模式可移植**（NPU GM 标量读；设计期探针实测两级链可编译位精确） |
| **O-EP2** | **扫描内谓词过滤**（`le<0` 跳过） | 单遍完成过滤 + 计数，免独立 compaction pass | 分支跳过非本地对 | 分支/predicate | **模式可移植但形态受限**：NPU 标量上下文禁用 `T.if_then_else`（CG-2026-0028c），须用普通 `if` |
| **O-EP3** | **`-1` 哨兵编码排除**（`fwd_idx[idx]=-1`） | 免独立 mask 数组；排除信息就地编码 | 哨兵值语义 | 无 | 算法层可移植（哨兵只写 `fwd_idx`，host 零初始化 `permuted_idx` 兜底） |
| **O-EP4** | **本地域尺寸压缩**（表/直方图/scan 长度 `E_local ≤ E_global`） | 表更小、scan 更短、scratch 更小 | `expert_map` 把值域压到 `E_local` | 无 | 算法层可移植（结构性收益，无条件保留） |

> 识别不出来 ≠ 不存在。O1–O7 与非 EP 版同源（逐项在 §0.5 判定）；**O-EP1–O-EP4 为 EP 变体新增的显式优化**，
> 逐项在 §0.5 判定。注：Phase-2 gather 的 `ROWS_PER_BLOCK` 分块与 128-bit `uint4` 向量化（源 L218–L258）属胞弟单元，不在本表。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（EP 计数排序映射、`le=expert_map[g]`、`le<0` 排除、`true_*`、`permuted_idx`/`fwd_idx` 互逆、int32/int64） | 语义 | 无 | — | **保留** | 语义无条件保留（migration-analysis.md §5.1）；稳定序选择见 §0.1 边界语义（op 测试顺序无关，golden 为稳定序） |
| O1 SMEM `s_counts`/`s_offset` | 优化 | shared memory | 有（UB/GM） | **等价替换** | 计数表改 **GM scratch** `bh[NB,E_local]`（跨 kernel 可读），前缀中间量用 `T.alloc_ub`（§4.5） |
| O2 单 block（grid=1） | 优化/结构 | 大 block + 廉价块内同步 | 无（NPU 一维核；无块内 barrier 原语） | **重新设计** | 拆 **三 kernel GM 交接**（计数 / 前缀 / 散写）；跨核沟通只用 GM + stream 顺序（胞兄 `_make_scan_kernel_nopad` 与非 EP 版已证底座） |
| O3 thread 级并行（一 thread 一位置） | 优化 | CUDA thread/warp | 无（NPU 为 SIMD lane） | **重新设计** | 分核：块与核 1:1（`NB=min(P,numel)`），核内 `T.serial(C)` 静态边界（§0.6 R3） |
| O4 fetch-add 散写（`tl_atomic_add_offset`） | 算法/优化 | CUDA 定制 extern，**fetch-add 返回旧值** | **无**（`T.atomic_add` 降 `hivm::StoreOp`，无返回值） | **重新设计** | §0.6 R2：**确定性两级块并行稳定计数排序**（`slot=slot_base[b,le]+r`），逐位复现稳定序（§1.6.1 E2-EP）；fetch-add 缺口已登记 CG-2026-0030 |
| O5 int64 排他前缀 | 优化 | 无 | 有 | **保留** | int32 偏移 `T.cast` 到 int64 写 `efo`（`M_local ≤ numel < 2^31`，逐位精确，§1.6.1 E3） |
| O6 thread 0 串行前缀（`T.serial(E_local)`） | 算法/优化 | 无 | 有（算法层） | **等价替换** | 改硬件 `T.cumsum`（`(1,E_local)` int32）+ 1-lane 移位（E1；设计期探针 P2-EP 实测 `E_local ∈ {1,3,5,96,384}` 位精确） |
| O7 GM 表交接 + 独立 scan kernel | 优化/结构 | 无（GPU 多 kernel） | 有 | **保留（主选承载形态）** | 三 kernel 同 stream 顺序 launch，产物落 GM；胞弟 gather 读 `permuted_idx`。**依据**：非 EP 版 Stage 3 集成实测通过 |
| **O-EP1** `expert_map` 内联查表（两级 GM 间接） | 优化 | 共享/全局小表随机读 | 有（GM 标量读） | **保留（主选）** | `le = expert_map[g]` 作为 kernel A/C 的标量 GM 读链；**设计期探针 P1-EP 实测**代表 shape（`numel=32768,E_g=384,E_l=128,NB=40`）编译通过 + 位精确（§9.4 R-2） |
| **O-EP2** 扫描内谓词过滤（`le<0` 跳过） | 优化 | 分支/predicate | 有（普通 `if`） | **等价替换** | 用**普通 `if le >= 0:`** 守卫（**禁用**标量 `T.if_then_else`，CG-2026-0028c；非 EP 版的边界守卫同款形态）；探针实测守卫分支正确（含全非本地用例） |
| **O-EP3** `-1` 哨兵（`fwd_idx[idx]=-1`） | 优化 | 无 | 有 | **保留** | 非本地对在 UB 暂存行写 -1，随块内 `fwd_idx` 一次动态 extent `T.copy` 落 GM；`permuted_idx` 不写（host 零初始化兜底，源 op 同款） |
| **O-EP4** 本地域尺寸压缩（`E_local` 轴） | 优化/结构 | 无 | 有 | **保留** | 直方图/前缀/表/`slot_base`/`run` 全按 `E_local` 定尺寸；scratch 与 scan 长度随 `E_local/E_global` 比例线性缩小（§1.6.0 R3） |
| （舍弃）UB 内运行期下标散加（源 P2 的 SMEM atomic） | 优化 | shared memory + 运行期下标 | 无可用形态 | **舍弃** | CG-2026-0028(d)：**标量读 UB 的运行期计算下标污染相邻结果**；改为 GM `bh`（R2） |
| （舍弃）多核标量 GM store（非 EP 版 §3.3 伪代码形态） | 实现 | — | 无 | **舍弃** | 非 EP 版 Stage 3 实测：多核 Vector kernel 的**直接标量 GM store 随机丢写**（`_make_scan_kernel_nopad.py` 模块头注，tilelang 0.1.2+2853d71 / Ascend910B2C）→ 一律改 `T.copy`（MTE3）或**向量载荷** `T.atomic_add` |

> 每项处置依据见「NPU 对应方案 / 依据」列；**语义层不可妥协**。
> **判定输入含 Phase R 调研结论（§1.6.0）**：调研在「可实现族」内择优——三 kernel GM 交接 + 确定性块并行计数排序
> + **扫描内联 pull-down remap**（§1.6.0 选定 C1-EP）胜出；源单 block fetch-add 形态因 NPU 无 fetch-add 被否决（可行性前置）；
> 「全局直方图再聚合」（C-EP1）与「物化重映射 ids」（C-EP2）因额外 scratch/额外扫描被否决（§1.6.0 R1/R3）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**三 kernel GM 交接——A 本地域守卫计数 + B 前缀 + C 确定性块并行守卫散写**）。

**重设计项 R1：承载形态——从「单 block 全相位」改为「三 kernel GM 交接」（EP 版）**

- **源方案**：1 个 block（1024 线程）内 5 个相位，`sync_threads` 间隔；thread 级并行。
- **NPU 新算法**：拆三 kernel（同 stream 顺序 launch）：
  - **kernel A**（`_scan_count_ep_kernel`，Vector，`T.Kernel(NB)`）：逐位置 `g=flat_ids[idx]; le=expert_map[g];`
    `if le >= 0: T.atomic_add(bh[core_id, le], ub_one, [1])`（**向量载荷原子**作可靠 GM store；块核 1:1、每行私有
    故原子性非必需）产**每块本地域直方图** `bh[NB, E_local]`（`b = idx // C`）。
  - **kernel B**（`_scan_prefix_ep_kernel`，Vector，`T.Kernel(1)`）：`counts[l] = Σ_b bh[b,l]`（核内 `T.serial(NB)` ×
    `T.Parallel(E_local)` 向量加）→ **一次 `T.cumsum(counts)` (1,E_local)** + 1-lane 移位 → `off`；写
    `true_sizes`、`true_offsets`、`expert_first_token_offset`（int64 cast）；并算
    `slot_base[b,l] = off[l] + Σ[b'<b] bh[b',l]`（落 GM）。
  - **kernel C**（`_scan_scatter_ep_kernel`，Vector，`T.Kernel(NB)`）：逐块核内 `T.serial(C)`：
    `g=flat_ids[idx]; le=expert_map[g];` `if le < 0: ub_fwd[0,c] = -1`（**哨兵**）
    `else: r=run[core_id,le]; T.atomic_add(run[core_id,le], ub_one, [1]); slot=slot_base[core_id,le]+r;`
    `ub_fwd[0,c]=slot; T.atomic_add(permuted_idx[slot], ub_val, [1])`；块内 `fwd_idx` 一次动态 extent `T.copy` 落 GM。
- **语义保持论证**：三 kernel 的产物逐元素等于源 EP 语义（§0.1）——`counts`/`off` 为整数精确定义；`fwd_idx`/
  `permuted_idx` 的**稳定序**与 §8.1 EP golden 逐位一致（§1.6.1 E2-EP：块并行 EP 展开 == 顺序 EP 计数排序，
  4000 随机 + 11 corner 0 违反；`-1` 哨兵集合与 `le<0` 逐位一致）。**无跨引擎交接、无同 scope「向量产出 → 标量消费」**
  （标量只读**跨 kernel GM**：`flat_ids`/`expert_map`/`slot_base`/`run`）。
- **与源算法优化意图的承接**（§0.4 逐项，**不得静默丢弃**）：O1（SMEM 计数）→ GM `bh`；O2（单 block）→ 重设计（R1）；
  O3（thread 并行）→ 重设计（R3）；O4（fetch-add）→ **意图承接、形态替换**为确定性块并行稳定计数排序（R2）；
  O5（int64 前缀）→ 承接（cast）；O6（串行前缀）→ `T.cumsum`；O7（GM 表交接）→ 保留；
  **O-EP1**（内联查表）→ 保留（两级 GM 读链，探针实测）；**O-EP2**（谓词过滤）→ 等价替换（普通 `if`）；
  **O-EP3**（-1 哨兵）→ 保留；**O-EP4**（本地域压缩）→ 保留。

**重设计项 R2：直方图/散写——从「UB 运行期下标原子散加 + fetch-add」改为「(block, 本地专家) GM 守卫计数 + 确定性块并行散写」**

- **源方案**：`if le>=0: T.atomic_add(s_counts[le],1)`（SMEM，运行期下标，两级查表）；散写用 `tl_atomic_add_offset`。
- **NPU 新算法**：
  - 计数：**向量载荷原子作可靠 GM store** `g=flat_ids[base+c]; le=expert_map[g]; if le>=0: T.atomic_add(bh[core_id,le], ub_one, [1])`，
    `b = core_id`（每核一行，无争用）。**不用标量载荷 `T.atomic_add`**（CG-2026-0029 前端报错），**也不用普通 RMW**
    （非 EP 版 Stage 3 实测：多核标量 GM store 随机丢写）；向量载荷原子在**块核 1:1、每行私有**下纯粹充当可靠 store，
    语义与 RMW 等价（无并发写者）。
  - 散写：**不用 fetch-add**（NPU 无等价，CG-2026-0030）——`slot = slot_base[b,le] + r` 由**确定性公式**给出，与执行
    时序无关；`run[core_id,le]` 的 RMW 用「标量 GM 读 `r` + 向量载荷原子自增」两步（可靠写路径）；
    `permuted_idx[slot]` 用**向量载荷原子**（host 预零；每 slot 只写一次）；`fwd_idx` 用「UB 暂存 `ub_fwd` 行 + 一次
    动态 extent `T.copy`」（含 `-1` 哨兵，一并对齐块内偏移）。
- **语义保持论证**：块并行展开与源/§8.1 golden 的**稳定**计数排序逐位相同（§1.6.1 E2-EP）；非本地对恰在
  `le<0` 分支写 -1，`permuted_idx` 不加写；空本地专家段为空；`M_local=0`/`numel=0` 无 slot 被写（corner 覆盖）。
  所有标量操作数要么是循环变量、要么是 **GM 读**（`flat_ids`/`expert_map`/`slot_base`/`run`）——**无标量消费本 scope
  向量产出**（CG-2026-0027 规避保持）。

**重设计项 R3：任务划分与分核——从「单 block × 1024 thread」改为「固定物理核数 + 块核 1:1 + 核内静态串行」**

- **源方案**：`T.Kernel(1, threads=1024)`，一 thread 一元素。
- **NPU 新算法**：`NB = min(P, numel)`（`P` = 纯 Vector 实查物理核数 = 40，§5.5）；`C = ceildiv(numel, NB)`
  （均编译期常量）。因 `NB ≤ P` 恒成立，**块核 1:1**（`b = core_id`）。kernel A/C 用 `T.Kernel(NB)`，核内
  `for c in T.serial(C)`（**静态边界**），边界用 `if base + c < numel` 守卫（普通 `if`）。kernel B 单核。
  **`E_local` 不参与分核**（只作向量化轴长度，§1.6.3）。
- **语义保持论证**：分核只改「哪个核写哪些块」，不改任何 slot 的 (本地专家, rank) 结果；块内顺序 = 位置顺序，
  块间由 `slot_base` 的 `Σ[b'<b]` 保证有序 → 全序仍是稳定序，故结果与核数/NB 无关（§1.6.1 E2-EP 对多组
  `NB ∈ {1,2,3,5,8,16,40,64}` 随机验证）。

**弃选：单核全局顺序散写（无块结构）**

- 结构：kernel C 单核，`for idx in T.serial(numel)` + 全局 `run[E_local]` 计数器。**弃选理由（性能）**：`numel` 最大
  32768、40 核仅用 1 核，scalar 运行 rank 全串行；与「极大规模核内串行摊薄启动开销」精神相悖（仅当 40 核并行形态
  失败时降级）。保留为 §1.6.0 备选（实验裁决计划）。

**弃选：内联进 gather 核（表不落 GM）**

- 结构：scan 与 gather 合并，`permuted_idx` 不落 GM。**弃选理由（可行性）**：CG-2026-0010（Developer persistent +
  `T.gemm` + 整型索引段共存即 lowering segfault）+ CG-2026-0026（跨引擎同 block 交接不可编译，官方
  `examples/mixcv/mixcv_mixkernel.py` 复现）；且本单元与 gather 单元独立设计，保持 GM 接口即胞兄已证底座。

### 0.7 标杆实现

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py` 的 `_make_scan_kernel_nopad_ep`（L122–L215）。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`（L134–L227）。
- **强参照（非 EP 版，含 Stage 3 实测与集成）**：`examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md`、
  `examples/moe_permute_nopad/_make_scan_kernel_nopad/_make_scan_kernel_nopad.py`、
  `examples/moe_permute_nopad/_make_scan_kernel_nopad/verify_equiv.py`。
- 源 op / 测试 / bench / manifest / workload：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`（`_ref_moe_permute_nopad` 为非 EP golden 依据，L31–L70）、
  `examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoePermuteNopadFwdOp`）、
  `examples/TileOPs/tileops/workloads/moe.py`。
- 胞兄设计（同族整数索引算子，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/_tile_scheduler_kernel.py`。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的三 kernel 算法）。

---

## 1. 概述

### 1.1 算子名称

`_make_scan_kernel_nopad_ep`（MoE no-pad token 重排的**EP 计数排序扫描单元**，纯整数索引，`target="npuir"`）。

### 1.2 功能描述

从 `flat_ids[numel]`（int32，`numel = T*K`）与 `expert_map[E_global]`（int32，下拉重映射）计算 no-pad EP 计数排序的
元数据：每**本地**专家 token 数 `true_sizes[E_local]`、紧致起始 `true_offsets[E_local]`、非填充排他前缀
`expert_first_token_offset[E_local+1]`（int64）、正向映射 `fwd_idx[numel]`（非本地为 `-1`）、gather 索引
`permuted_idx[numel]`（仅 `[0, M_local)` 有效）。**迁移决策后的 NPU 形态**（§0.6 R1）：
**三 Vector kernel GM 交接**（A：`le=expert_map[g]` 守卫的块直方图 `bh[NB,E_local]`；B：`T.cumsum` 前缀产
`off`/输出/`slot_base`；C：守卫的确定性块并行运行 rank 散写 `permuted_idx`/`fwd_idx`），同 stream 顺序 launch；
`permuted_idx` 交胞弟 gather 单元读。全程整数、无浮点、无 CPU 同步。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
le(idx)=expert\_map[flat\_ids[idx]],\qquad
counts[l]=\#\{idx: le(idx)=l \ge 0\},\quad
off[0]=0,\ off[l+1]=off[l]+counts[l]
$$

$$
fwd\_idx[idx]=\begin{cases}-1, & le(idx)<0\\ off[le]+rank(idx), & le(idx)\ge 0\end{cases},\quad
rank(idx)=\#\{j<idx: le(j)=le(idx)\}
$$

$$
permuted\_idx[\,off[le]+rank(idx)\,]=idx//K
$$

输出 `(true_sizes, true_offsets, expert_first_token_offset, fwd_idx, permuted_idx)`，长度分别为
`E_local / E_local / E_local+1 / numel / numel`。**全程 int32/int64，无浮点**。
manifest roofline（`examples/TileOPs/tileops/manifest/moe.yaml` `MoePermuteNopadFwdOp`）是 **op 级**（含 gather 的
`T*K*H` 搬运 + 元数据字节），属整个 permute op，非本扫描单元。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源结构差异已逐条注明）：

1. **kernel A — 本地域块直方图（Vector，多核）**：`T.Kernel(NB)`（块核 1:1，`b=core_id`）；核内 `T.serial(C)`
   逐位置 `g=flat_ids[base+c]; le=expert_map[g];` `if le>=0:` 向量载荷 `T.atomic_add(bh[core_id, le], ub_one, [1])`；
   产 GM `bh[NB, E_local]`（零初始化）。
   **与源差异**：源为 SMEM 运行期下标原子散加（CG-2026-0028d 禁止）+ 标量载荷原子（CG-2026-0029 不可用）；
   改为 GM `bh` + 向量载荷原子（可靠 store）+ 块核 1:1。
2. **kernel B — 跨块求和 + 前缀 + 输出（Vector，单核）**：`counts[l]=Σ_b bh[b,l]`（核内串行 `b` × `T.Parallel(E_local)`）；
   **唯一一次 `T.cumsum(counts)` (1,E_local)** + 1-lane 移位 → `off[E_local+1]`；`true_sizes/true_offsets/
   expert_first_token_offset`（int64 cast）经 **UB 暂存 + `T.copy`** 落 GM；算 `slot_base[b,l]=off[l]+Σ[b'<b]bh[b',l]` 落 GM。
   **与源差异**：源 thread0 `T.serial(E_local)` 串行 → 硬件 scan（E1）；源逐元素循环写表 → UB 暂存 + `T.copy`；
   轴长 `E_local` 而非 `E`。
3. **kernel C — 守卫的确定性块并行散写（Vector，多核）**：`T.Kernel(NB)`（块核 1:1）核内 `T.serial(C)` 位置：
   `g=flat_ids[idx]; le=expert_map[g];` `if le<0: ub_fwd[0,c]=-1` else
   `r=run[core_id,le]; run[core_id,le]=r+1; slot=slot_base[core_id,le]+r; ub_fwd[0,c]=slot;
   permuted_idx[slot]=idx//top_k`（向量载荷原子）；块尾一次动态 extent `T.copy(ub_fwd → fwd_idx)`。
   **与源差异**：源 fetch-add 原子取 slot（NPU 无等价）→ 确定性 `off[l]+block_base+rank` 公式（R2），
   逐位复现稳定序（§1.6.1 E2-EP）；`-1` 哨兵与本地对写在同一 UB 暂存行内（O-EP3）。
4. **消费方式**：5 张表落 **GM**，`permuted_idx` 由胞弟 gather 单元读（`hidden_states[permuted_idx[slot]]`）；
   本单元自身不搬运浮点数据。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[kernel A · Vector · NB=min(P,numel), 块核 1:1]
flat_ids[numel] --g--> expert_map[E_global] --le--> if le>=0: 向量载荷原子自增 --> GM[bh(NB,E_local) int32]
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel B · Vector · grid=1]
  GM[bh] ──串行 b × T.Parallel(E_local) 求和──▶ counts(E_local) ──T.cumsum + 1-lane 移位──▶ off(E_local+1)
  GM[bh] ──串行 b 运行累加 + off──▶ GM[slot_base(NB,E_local) int32]
  counts/off ──UB 暂存 + T.copy──▶ GM[true_sizes(E_local)] / GM[true_offsets(E_local)] / GM[efo(E_local+1) int64]
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel C · Vector · NB 核（块核 1:1）]
  flat_ids + expert_map + slot_base + run(NB,E_local)（零初始化）
        ──串行：le<0 ? -1 : slot_base[b,le]+r──▶ GM[fwd_idx(numel) int32]（经 UB 暂存 + T.copy）
                                              ──▶ GM[permuted_idx(numel) int32]（仅本地对；host 预零）──┐
                                                          ▼ （同 stream 顺序 launch）
                                   [胞弟 _make_gather_kernel_nopad] perm_h[slot]=hidden_states[permuted_idx[slot]]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（「`flat_ids` + `expert_map` → 本地域稳定计数排序的 counts/前缀/fwd_idx/permuted_idx」）的
> **算法族**。输入/源算法（单 block SMEM 原子计数 + fetch-add 散写 + 内联查表）只是基线候选之一。
> 深度分级：多步索引/搜索/散写类 → **完整调研**。
> **可实现性约束（前置，来自 Phase M1 + 非 EP 版 Stage 3 实测 + 本变体设计期探针）**：
> (a) 本工具链**无 fetch-add**——`T.atomic_add` 降 `hivm::StoreOp`（`docs/Tilelang.language/原子操作/T.atomic_add.md`
>     §3），源 `tl_atomic_add_offset` 无等价 API（CG-2026-0030）；
> (b) **多核向量 kernel 的直接标量 GM store 会随机丢写**（非 EP 版 Stage 3 实测，模块头注）→ 可靠写路径仅
>     MTE3 `T.copy` 与**向量载荷** `T.atomic_add`；
> (c) UB 运行期下标读/写污染相邻结果（CG-2026-0028(d)）；
> (d) **同 scope「向量产出 → 标量消费」不可实现**（CG-2026-0027，`T.pipe_barrier` 无效）；
> (e) 跨引擎同 block 交接不可编译（CG-2026-0026）；Developer persistent + `T.gemm` + 整型索引段共存即 segfault（CG-2026-0010）；
> (f) int32 `>>`/`T.alloc_local`+`T.serial`/标量 `T.if_then_else`/多行 int32 `T.cumsum`/一 kernel 两次 `T.cumsum`
>     均 SIGSEGV 或错值（CG-2026-0028）；`T.atomic_add` **标量载荷**前端失败（CG-2026-0029）；
> (g) `T.Parallel` 循环体内**禁直写传入的 GM 张量**（`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3），
>     GM 写一律经 UB 暂存 + `T.copy`；
> (h) **两级标量 GM 间接链**（`le = expert_map[flat_ids[idx]]`）+ `le>=0` 守卫在 kernel A/C 中**可编译且位精确**
>     （本变体设计期探针 P1-EP，代表 shape 8 例全过）；
> (i) `E_local = 0` 时 kernel B 的 `T.alloc_ub((1,0))` 触发 MLIR `strides must not be zero`（探针 E_local=0 实测）
>     → `E_local ≥ 1` 为可编译前提。
> 调研在**可实现族**内择优（可行性前置于经济性）。证据：`.agents/evolution/capability-gaps.md`
> CG-2026-0010/0026/0027/0028/0029/0030。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源 EP 算法）** | 单 block；两级查表 + SMEM 原子计数 → thread0 串行 int64 前缀 → fetch-add 散写 | — | — | 基线 |
| C1-EP | **三 kernel GM 交接 + 扫描内联 pull-down remap**（**选定**） | A：守卫的 GM `bh[NB,E_local]` 直方图；B：`T.cumsum` 前缀 + `slot_base`；C：守卫的块内运行 rank 散写 | 逐位复现本地域稳定计数排序（E2-EP）；remap 与「全局计数再聚合」等价（F1） | 去 fetch-add；分核到 40；无同 scope 标量消费向量；scratch ∝ `E_local` | **选定** |
| C-EP1 | **全局直方图 + 再聚合**（`bh_g[NB,E_global]` → `counts_g` → `counts_l[map[e]] += counts_g[e]`） | 先按全局专家计数，再经 `expert_map` 聚合到本地域 | 数学等价（F1 的右式） | 免每位置 remap 读；**但** `bh_g` 达 `NB·E_global`（384 时 61KB vs 20KB）、聚合是 `E_global` 长**标量散加**（无向量 scatter），且 B 侧 scan 轴仍须回到 `E_local` | 对比（**弃选**） |
| C-EP2 | **先物化重映射 ids**（`mapped[i]=expert_map[flat_ids[i]]`）再跑非 EP 扫描 | 4th kernel 或融合的前置 pass | 数学等价 | 多一趟 `numel` GM 写 + 读（+256KB 流量），哨兵仍需在计数侧守卫 → 纯增开销 | 对比（**弃选**） |
| C-EP3 | **两阶段 filter/compact**（先压缩本地对，再排序） | compaction 后串行排序 | 数学等价 | 多一趟扫描 + 额外索引缓冲，无收益 | 对比（**弃选**） |
| C2 | **硬件前缀 scan**（`T.cumsum` + 1-lane 移位） | 排他前缀 = 含前缀 cumsum 后 1-lane 移位（E1） | 整数恒等 | `O(E_local)` 串行标量 → 1 op | 采纳（并入 C1-EP/B） |
| C3 | **确定性块并行稳定计数排序**（`slot=off[l]+block_base+运行r`，本地域） | 块内运行 rank + 块前缀 `block_base` | 与顺序稳定计数排序逐位相同（E2-EP） | 每 slot 只写一次；无 fetch-add；`O(numel)` 运行 rank | 采纳（并入 C1-EP/C） |
| C4 | **单核全局顺序散写**（无块结构） | 单核 `T.serial(numel)` + 全局 `run[E_local]` | 数学恒等（即稳定序） | 结构最简，但只用 1/40 核 | 对比（**备选**） |
| C5 | **`T.vsort` 排序驱动**（key=`remap(flat_ids)*numel+idx`） | 排序即计数排序；需先算 key（含 remap） | 数学恒等（非稳定，契约允许） | 全向量；但 **UB 容量不足**（3×numel×4=384KB>192KB）且 `T.vsort` 仅 fp16/fp32（int 键 >2^24 不安全） | 对比（弃选） |
| C6 | **逐本地专家 masked scan** | 对每本地专家扫描全 numel 求 rank | 数学恒等 | `O(E_local·numel)` + 多 cumsum（CG-2026-0028e） | 对比（弃选） |
| C7 | **`T.gather` 逆置换** | 用 gather 求 `fwd_idx` 逆 | 数学恒等 | gather 的索引本身就是待求置换（循环依赖） | 对比（弃选） |
| C8 | **host 侧计数排序** | CPU 算表后下发 | 数学等价 | 引入 CPU 同步 + host 禁改输入（`references/ascend-constraints.md` §4） | 对比（弃选） |

**R2 在线算法**：**有**。EP 计数排序仍是**单遍在线**形态——顺序扫一遍 `flat_ids`，对每位置先做一次 `expert_map`
查表（**逐元素过滤**，不引入额外遍），维护 per-本地专家 running counter（`O(E_local)` 状态）即得 `rank`；前缀 `off`
亦为单遍 running sum。本设计的 kernel A（守卫直方图）与 kernel C（守卫运行 rank）都是**单遍流式**（对 `flat_ids`
各扫一遍、无反向依赖；过滤谓词是元素级、非多遍）。**结构判据**：计数/前缀是可分解的 running 统计量；过滤/重映射
是**逐元素映射**（可融合进任意一遍），故**不存在**「不先压缩就无法单遍」的情形。收益口径：扫描遍数 = 2（A、C 各 1 遍）；
中间缓冲 = `bh`/`slot_base`/`run` 各 `O(NB·E_local)`（≤ 40×384 int32 = 61.4KB，落 GM）。

**R3 复杂度对比**（四口径；代表 EP shape：`T=4096, K=8, E_global=384, E_local=128` → `numel=32768`；
`NB=40`、`C=820`；elem=int32=4B）：

| 算法候选 | 计算量（整数 op） | 访存量 (Bytes) | 扫描遍数（flat_ids） | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|---------------|---------------------|-------------|---------------------|
| C0 基线（单 block SMEM 原子 + fetch-add） | numel 守卫原子加 + `E_local` 串行 + numel 守卫 fetch-add ≈ **65.7K** | flat_ids 128KB + expert_map 1.5KB + 表 1.6KB + permuted 128KB + fwd 128KB ≈ **388KB** | 2（P2、P5） | SMEM `(E_local+E_local+1)` ≈ 1KB | 1 block × 1024 thread；**NPU 无 fetch-add（不可实现）** |
| **C1-EP 选定（三 kernel）** | A：numel 守卫计数 ≈ 32.8K（+remap 读）；B：`NB·E_local` 前缀 ≈ 5.1K + 1 scan；C：numel 守卫运行 rank ≈ 32.8K；合计 ≈ **70.8K** | flat_ids 128KB×2 + expert_map 1.5KB×2 + `bh` 20KB×3 + `slot_base` 20KB×2 + `run` 20KB×2 + fwd 128KB + permuted 128KB + 表 1.6KB ≈ **658KB** | 2（A、C） | GM scratch ≈ **60KB**（`bh`+`slot_base`+`run` 各 20KB；UB 内核 B ≈4.5KB、内核 C ≈3.3KB） | 40 核；**无跨核归并、无 in-block 跨引擎同步** |
| C-EP1 全局直方图 + 聚合 | 同 C1 + 聚合 `E_global`=384 标量散加 ≈ 71.2K | `bh_g` 61.4KB×3（代替 20KB×3）→ ≈ **787KB** | 2 | GM scratch ≈ **184KB** | 聚合段**标量散加**（无向量 scatter，CG 约束）→ 单核 / 低效 |
| C-EP2 物化重映射 ids | 同 C1 + numel 额外写/读 | 同 C1 + mapped 128KB×2 ≈ **914KB** | 3（多一遍 remap pass） | GM scratch ≈ 60KB + 128KB | 多一趟 GM 往返，无收益 |
| C4 单核顺序（备选） | numel 运行 rank 32.8K + numel 守卫直方图 32.8K | 同 C1（`run` 降为全局 `run[E_local]` 0.5KB） | 2 | GM scratch ≈ 41KB | **1 核**（40× 串行度） |
| C5 `T.vsort` | numel·log²numel ≈ **7.4M** compare-exchange | 3×numel×4×↔ ≈ **393KB/遍** | 2 | **UB 384KB > 192KB（容量否决）** | 高但容量不可行；`T.vsort` 仅 fp16/fp32 |
| C6 逐本地专家 masked scan | `E_local·numel` = **4.2M** + 前缀 | flat_ids 重读 `E_local` 次 ≈ 16.8MB | `E_local`=128 | UB `E_local×C` 超容 | 高但**触发 CG-2026-0028(e/f)** |
| C7 gather 逆置换 | 循环依赖 | — | — | — | 不可构造 |
| C8 host 排序 | host `O(numel log numel)` | flat_ids + expert_map 设备→host + 表 host→设备 | 1 + 2 总线往返 | host 内存 | **CPU 同步阻塞** |

口径脚注：计算量计整数加/比较/查表（本算子无浮点）；**访存统一按「缓冲区尺寸 × 该 buffer 在 GM 侧的读/写往返次数」
计**（读或写各计一次，**不按元素访问次数**）——故 `bh` 计 3 次往返（A 写 1 + B 趟 1 读 1 + B 趟 2 读 1）、
`slot_base` 计 2 次（B 写 1 + C 读 1）、`run` 计 2 次（C 读 1 + 写 1）、`expert_map` 计 2 次（A/C 各读 1）；
扫描遍数指 `flat_ids` 在 GM 侧被读的次数；中间缓冲峰值以 GM scratch **驻留峰值**（buffer 形状之和，非累计往返）为准。
**本单元是极小工作量的索引算子**（prefill 下 ≪10^5 整数 op），故 R3 排序以**固定开销（发射 / GM 往返）**为主导。

**R4 硬件亲和性评估**（逐候选对照检查清单；「工具链可用性」列按 `negative-claim-evidence.md` §2 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Vector（整数）但单 block 1024 thread → SIMD lane 不匹配 | SMEM 可行 | 可行 | 可行 | 块内无屏障原语 | **不可实现**：NPU 无 fetch-add + 多核标量 GM store 丢写 + UB 运行期下标 | 弃选（可行性） |
| **C1-EP 选定** | Vector（守卫 GM 计数 / `T.Parallel` / `T.cumsum`；标量 rank 顺序依赖） | UB 内核 B ≈4.5KB（`E_local`=128）/ ≈13.8KB（`E_local`=384）≪192KB | `E_local` 任意（探针 P2-EP 实测 1/3/5/96/384 位精确） | `C`/`NB` 编译期常量 | 三独立 Vector kernel，无 `T.gemm`、无跨引擎交接 | **构造上规避 CG-2026-0010/0026/0027/0028/0029/0030**；两级 GM 间接 + 守卫经 **P1-EP 探针代表 shape 实测**；向量载荷原子/`T.copy`/`T.cumsum` int32 `(1,E_local)` 均有实测 | **选定** |
| C-EP1 全局聚合 | Vector，但聚合段为 `E_global` 标量散加 | `bh_g` 61.4KB 占 GM | 可行 | 可行 | 聚合无向量 scatter | 聚合段**无向量散加 API**（`docs/Tilelang.language/原子操作/` 仅 `T.atomic_add` 单基址向量载荷 + `索引与元素操作/` 无 scatter）→ 需单核标量 | 弃选（经济性：scratch 3×、聚合串行） |
| C-EP2 物化重映射 | Vector | 可行 | 可行 | 可行 | — | 可行但多一趟 | 弃选（纯增流量） |
| C-EP3 两阶段 filter | Vector | 可行 | 可行 | 可行 | compaction 需 scatter | 可行但无收益 | 弃选 |
| C4 单核顺序（备选） | Vector（仅 1/40 核） | UB ≈4.5KB | 可行 | 可行 | 结构最简（同 C1-EP 构件） | 同 C1-EP 构件 | **备选**（主选编译/性能失败时降级） |
| C5 `T.vsort` | Vector（VSortOp） | 3×numel×4=384KB > 192KB | 可行 | 可行 | — | `T.vsort` 仅 fp16/fp32（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3） | 弃选：容量否决 + 键 cast 精确性风险 |
| C6 逐专家 masked scan | Vector | `E_local×C` 超 UB | 可行 | 可行 | — | **触发 CG-2026-0028(e/f)**（多 cumsum / 多行 int32 cumsum） | 弃选 |
| C7 gather 逆置 | Vector | 可行 | 可行 | 可行 | — | API 存在（`docs/Tilelang.language/索引与元素操作/T.gather.md`）但**索引循环依赖** | 弃选（不可构造） |
| C8 host 排序 | CPU | — | — | — | host-device 往返 | 可行但违约束 | 弃选：违反零 CPU 同步 + host 禁改输入 |

**调研结论**：选定 **C1-EP + C2 + C3 —— 三 kernel GM 交接（A：`le=expert_map[g]` 守卫的 GM `bh[NB,E_local]`
向量载荷原子直方图；B：一次 `T.cumsum`(1,E_local) 前缀产 `off`/输出/`slot_base`；C：守卫的确定性块并行运行 rank
散写 `permuted_idx`/`fwd_idx`，非本地写 -1 哨兵）+ 固定物理核数（纯 Vector → 40）+ 块核 1:1 + 核内静态 `T.serial`**。
关键依据：(i) **可实现性是前置约束**——源 fetch-add 散写无 NPU 等价、多核标量 GM store 丢写（非 EP Stage 3 实测）、
C5 容量否决、C6 触发已知 codegen 缺口、C7 循环依赖；(ii) **EP 重映射以「扫描内联 pull-down + 谓词过滤」实现最省**
——C-EP1 使 scratch 增至 3×（`bh_g` 61.4KB）且聚合退化为 `E_global` 标量散加，C-EP2 多一趟 256KB 流量；
(iii) 全部计算量 ≪10^5 整数 op（R3），C1-EP 以三 kernel 顺序 launch 消源单 block fetch-add，且**逐位复现稳定序**
（§1.6.1 E2-EP）；(iv) C4（单核）作为结构最简降级备选。
**与基线结构差异一句话**：保留源「独立 scan 单元写 GM 表 + gather 读表」数据流与 `expert_map` 下拉重映射语义，
把源单 block 五相位拆成 **三 kernel GM 交接**，把 UB 运行期下标原子计数改为 **GM `bh` 守卫向量载荷原子计数**，
把 thread0 串行前缀换成 **`T.cumsum`**，把 fetch-add 散写换成 **确定性块并行稳定计数排序 + `-1` 哨兵**，
把 `E` 轴换成 **`E_local`** 轴。
**调研范围**：`references/algorithm-candidates.md`（无「MoE permute / EP counting sort」专条，参考 scan/分桶/gather 族条目）、
pattern-library（`layout.md`/`constants.md`/`traps-compiler.md`/`INDEX.md`）、源算子 §0.3/§0.4、
本仓 examples（`examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md`、
`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、
`testing/autotune/anneal/moe_grouped_gemm_anneal.py`、`examples/indexer/indexer_bwd.py`）、
`.agents/evolution/capability-gaps.md`（CG-2026-0010/0026/0027/0028/0029/0030）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1→GM `bh`；O2→重设计（R1）；O3→重设计（R3）；
O4（fetch-add）→**意图承接、形态替换**为确定性块并行稳定计数排序（R2）；O5→承接（cast）；O6→C2 硬件 scan；
O7→保留（R1）；**O-EP1→保留**（两级 GM 读链）；**O-EP2→等价替换**（普通 `if` 守卫）；**O-EP3→保留**（-1 哨兵）；
**O-EP4→保留**（`E_local` 轴压缩）。

**设计期 roofline（D-2）**：代表 EP shape（`numel=32768, E_global=384, E_local=128, K=8, NB=40, C=820`）：
- **流量项**：GM 流量 ≈ 658KB（R3 C1-EP，按缓冲区尺寸×往返口径）÷ `~1.26 TB/s`（`CONST-mte2-degradation`，混合读写地板，
  tilelang 0.1.2+ed787bb / CANN 8.5.0）≈ **0.52 µs**（可忽略）。
- **发射项**：kernel B（单核）两趟 `T.serial(NB)`：趟 1（`T.copy(bh 行)` + `T.Parallel(E_local)` 累加）≈ 2 op/轮，
  趟 2（`T.Parallel(E_local)` 写 `slot_base` + `T.copy(bh 行)` + `T.Parallel(E_local)` 累加）≈ 3 op/轮，另 B3 表输出
  UB 暂存 + `T.copy` ≈ 5 op 一次性，合计 ≈ `(2+3)×40 + 5 = 205 op` × `~0.5µs/op`（`CONST-vector-launch-overhead`）
  ≈ **103 µs**；kernel A/C 为标量循环（发射按指令计，40 核并行、每核 ≈ `C×~8` ≈ 6.6K 指令，估算 ≪ 100µs，
  含 EP 多出的一级 GM 读）；合计发射项 ≈ **105 µs**。
- **容量项**：kernel B UB ≈ 8 buffer（7×int32 `(1,E_local/E_local+1)` + 1×int64 `ub_efo`）≈ **4.5KB**
  （`E_local`=128）/ **13.5KB**（`E_local`=384，上界）；kernel C UB ≈ `ub_one`+`ub_val`+`ub_fwd(1,C)` ≈ **3.3KB**；
  含 `CONST-capacity-910B2C` auto-multi-buffer 膨胀余量仍 ≪ 192KB；GM scratch ≈ 60KB（`E_local`=128，不占 UB）。
- **估算下界 = max(流量 0.52µs, 发射 ~105µs, 容量 ~0) ≈ 105 µs（发射项主导）**，一次性、与胞弟 gather（搬运
  `2×numel×H×2B ≈ 0.94 GB ≈ 0.8 ms`）同 stream 串行叠加时占比 ≈ 13%（与非 EP 版同量级）。`NB` 为可调旋钮
  （减小 NB 降发射项、但降 kernel C 并行度，权衡交 Stage 4）。
- **未文档化假设项**：kernel B「串行 over NB × `T.Parallel(E_local)`」的逐轮发射成本、以及 kernel A/C 标量循环的
  单核指令吞吐**未在 pattern-library 实测**（标注「未文档化假设 + 估算依据」）；该量**不影响主选/弃选裁决**
  （可行性已由构造成分 + 探针硬裁定），仅影响成本特征 → 交**实验裁决计划**（下）标定；`NB` 为可调项。

**实验裁决计划（三件套）**：

- **主选（进 Stage 3）**：三 kernel GM 交接——A 守卫块直方图（两级查表 + 向量载荷原子）、B 单核 `T.cumsum` 前缀 +
  `slot_base`、C 守卫块并行运行 rank 散写 + `-1` 哨兵（本设计 §3.3）。
- **备选（≤1，结构完整可实现）**：kernel C 退化为**单核全局顺序散写**（`T.Kernel(1)`，`T.serial(numel)` +
  全局 `run_gm[E_local]`，`le<0 ? -1 : off_gm[le]+r`）；kernel A/B 与全部 buffer 形状、5 表契约不变。
  触发条件：主选 kernel C 的「两级间接 + 守卫分支 + 双运行期下标标量 GM 读写」在本 shape 编译失败或出现
  flaky SIGSEGV / 错值。
- **裁决计划**：① **代表 shape** = smoke（`numel=8, E_g=384, E_l=128`）、prefill（`numel=32768, E_g=384, E_l=128`）；
  ② **测量指标** = 首编编译结果 + 5 张表与 §8.1 EP golden 逐元素精确一致（int32/int64，非容差）+ 哨兵集合精确 +
  反复运行稳定性；③ **判定阈值** = 主选 kernel C 编译失败/数值错 → 立即切备选；若扫描单元墙钟 > gather 有效时间
  10% 方考虑进一步优化（`NB` 调节）；④ **回写路径** = 实测数字回写 §1.6.0 roofline 与 §5.6；触发降级/升级按
  `perf-feedback.md` 受控路由。**shape 特化可行性**：本单元经 `lru_cache` 按 `(numel, E_global, E_local, top_k)`
  特化，可按 shape 分派主选/备选。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1-EP 算法（三 kernel，本地域 + 扫描内联 remap）。
> 每项四要素：原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/verify_equiv.py`
> （**整数精确，非容差**）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | 排他前缀 = 含前缀 + 1-lane 移位（kernel B，`E_local` 轴） | 源 thread0 串行 `off[0]=0; off[l+1]=off[l]+counts[l]` | `incl=T.cumsum(counts); off[0]=0; off[1:E_local+1]=incl[0:E_local]` | 整数恒等（`off[l+1]=incl[l]`）；机器验证 4000 例 0 违反；**shift guard**：省略移位 4000 例中 3736 例失配，证明移位是负荷项 | 把 `O(E_local)` 标量串行压成 1 条 `VCumsumOp` + 1 次向量拷贝 |
| E2-EP | 本地域稳定计数排序的散写：fetch-add → 确定性块并行 rank + `-1` 哨兵 | 源 `le<0 ? fwd=-1 : slot=fetch_add(write_offsets[le],1)`（时序相关、非确定） | `le<0 ? fwd=-1 : slot = slot_base[b,le] + r`，`r` = 块内本地专家 `le` 已出现次数（0 起）；`slot_base[b,le]=off[le]+Σ[b'<b]bh[b',le]`（E4） | **逐位复现本地域稳定计数排序**（块序 × 块内位置序 = 全局位置序），且**非本地哨兵集合与 `le<0` 逐位一致**；机器验证 4000 例 + 11 corner（含 `numel=0/1`、`numel<NB`、非整除、`E_local ∈ {1,2,3}`、全本地、全非本地、离散 `expert_map`）0 违反（`permuted_idx`/`fwd_idx`/`true_*`/`efo`/哨兵全覆盖） | 去除 fetch-add 依赖（NPU 无等价）；每 slot 只写一次，运行 rank `O(numel)` 且按块并行；与 EP golden 稳定序逐位一致 |
| E3 | int32 紧致偏移 cast 到 int64（kernel B） | 源 int64 直接累加 `s_offset` | `off32`（int32 前缀）；`efo[k]=int64(off32[k])` | 整数恒等（`M_local ≤ numel < 2^31`，无溢出）；机器验证 2000 例 0 违反 | 前缀在 int32 域完成（scan 原语 int32 已实测可用），仅最终 cast 到 int64 |
| E4 | `slot_base` 预计算（kernel B→C 交接） | kernel C 每次读 `off[le]` + `block_base[b,le]`（2 次 GM 读） | kernel B 预计算 `slot_base[b,le]=off[le]+Σ[b'<b]bh[b',le]`，kernel C 只读 1 次 | 整数恒等（同值预加，结合律） | kernel C 每位置少 1 次 GM 读（`numel` 次 GM 读削减） |
| E5 | `idx // top_k`（kernel C） | 源 `idx // T.int32(top_k)` | 同（`//`，**不用 `>>`**） | 非负整数除法，恒等 | 避免 int32 `>>` 的 CG-2026-0028(a) SIGSEGV；无收益损失 |
| **F1** | **EP pull-down remap 与「全局计数再聚合」等价（kernel A/C）** | 源内联查表 `le=expert_map[g]`，逐位置过滤后计数 | （等价右式）`counts_l[l]=Σ_{g: map[g]=l} counts_g[g]`，`counts_g[g]=#{idx: flat_ids[idx]=g}` | **逐位精确等价**（同一多重集按 `expert_map` 归类，聚合是同一划分的重排）；机器验证 2000 例 0 违反 | 支撑 R1 选型：**内联 remap 免一次全局直方图 + 免一次聚合散加**（C-EP1 对照，scratch 减 3×） |

> **`F1` 的性质说明**：F1 不是「用右式替换左式」的实现优化，而是**为 R1 选型提供等价性依据**——它证明
> 「扫描内联 pull-down remap」（选定 C1-EP）与「全局直方图再聚合」（C-EP1）在数学上产出同一 `counts_l`，
> 从而允许按**代价**而非语义在两者间择优；脚本同时给出右侧作为左式的独立参照实现。

**优化结论**：采纳 **E1 / E2-EP / E3 / E4 / E5**，并以 **F1** 佐证 R1 选型。全部为**整数逐位精确恒等**
（非容差内等价），**不引入任何 fp16/bf16 舍入误差**——精度标准为「5 张表 + 哨兵集合逐元素精确一致」（§8.2）。

**等价性机器验证结果表**（`verify_equiv.py`，纯整数对照，torch CPU；metric = 最大绝对差）：

| 项 | 对照 | 最大度量值 | 违反 / 用例 | 结论 |
|----|------|-----------|-------------|------|
| E1 | 含前缀 cumsum + 1-lane 移位 vs 源串行排他前缀（`E_local` 轴） | 0 | 0 / 4000 | EQUIV_PASS |
| E1（guard） | 省略 1-lane 移位（含前缀当排他）vs 正确前缀 | — | 3736 / 4000 失配 | GUARD_OK（确认移位负荷不可省） |
| E2-EP | 块并行本地域稳定计数排序（含 remap + 哨兵）vs 顺序 EP golden（`permuted_idx`/`fwd_idx`/`true_sizes`/`true_offsets`/`efo`/哨兵 全字段） | 0 | 0 / 4000 | EQUIV_PASS |
| E2-EP（guard） | 省略 `le>=0` 过滤（非本地按本地处理）vs 正确过滤 | — | 494 / 500 失配 | GUARD_OK（确认过滤守卫负荷不可省） |
| E2-EP（corner） | `numel=0/1`、`numel<NB`、非整除、全本地、全非本地、离散 `expert_map`、`E_g=384×numel=32768` | 0 | 0 / 11 | EQUIV_PASS |
| E3 | int32 偏移 cast int64 vs int64 原生前缀 | 0 | 0 / 2000 | EQUIV_PASS |
| F1 | 扫描内联 remap vs 全局计数再聚合（支持 R1 选型） | 0 | 0 / 2000 | EQUIV_PASS |

无失败项（所有采纳项均为 EQUIV_PASS）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案全部循环与标量计算点（对象 = §1.6.0 选定、经 §0.6 重设计落地的 C1-EP 三 kernel 算法）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 块直方图累加 `bh[b,le]+=1`（kernel A） | 源 SMEM 运行期下标 atomic | 无向量 scatter-add API（`T.atomic_add` 是单基址向量载荷 store，非逐 lane 散加；`docs/Tilelang.language/原子操作/T.atomic_add.md` §2）；标量载荷形态前端不可用（CG-2026-0029） | **否** | **API 缺失**：NPU 无「逐 lane 不同目的地下标」的向量散加原语；已用**向量载荷 `T.atomic_add(bh[core_id,le], ub_one, [1])`** 作可靠 GM store 承载（块核 1:1、每行私有 → 无需原子语义；设计期探针 P1-EP 实测代表 shape 可编译位精确）。查 `docs/Tilelang.language/` 全子目录（含 `原子操作/`、`索引与元素操作/`）无 scatter API |
| 2 | 两级重映射 `g=flat_ids[idx]; le=expert_map[g]`（kernel A/C） | 源同（共享表读） | 无 | **否** | **循环携带量（逐位置）**：`le` 决定 `bh`/`run`/`slot` 的下标，必须逐位置取得；即**逐元素映射**（`le = map[g]` 是值域压缩映射，非归约），且第二级为运行期下标标量 GM 读。属「block 级索引/任务映射计算」类别（每位置 2 次 GM 标量读，非逐元素热点）。设计期探针 P1-EP 实测两级链可编译位精确 |
| 3 | `le >= 0` 过滤守卫（kernel A/C） | 源同（分支） | 无 | **否** | **循环携带依赖 + API 约束**：守卫决定该位置是否进入计数/散写（循环携带的分支），且标量上下文**禁用** `T.if_then_else`/比较（CG-2026-0028(c) SIGSEGV）→ 只可用**普通 `if`**。探针 P1-EP 含全非本地/混合用例实测正确 |
| 4 | 全局 counts 列求和（kernel B） | 源无此步（SMEM 天然共享） | `T.serial(NB) × T.Parallel(E_local)` 向量加 | **是** | — |
| 5 | 排他前缀 `off`（kernel B） | 源 thread0 串行 `T.serial(E_local)` | **`T.cumsum` (1,E_local) + 1-lane 移位**（E1） | **是** | — |
| 6 | 输出表写 `true_sizes/true_offsets/efo`（kernel B） | 源逐元素循环 | `T.Parallel(E_local)` 写 **UB 暂存** + `T.cast`（`ub_efo` int64）→ `T.copy` 落 GM | **是** | — |
| 7 | 块前缀 `block_base`/`slot_base`（kernel B） | 无（源为 fetch-add） | `T.serial(NB) × T.Parallel(E_local)` 运行累加 | **是** | — |
| 8 | 块内运行 rank `r`（kernel C） | 源 fetch-add 隐式给出 | 无 | **否** | **循环携带顺序依赖**：`r` 是块内该本地专家的出现序（`run[b,le]` 读后自增），**本质串行**、无法用归约或双缓冲消除（稳定序要求「先出现者先得 slot」）；且 `run` 须落 **GM**（UB 运行期下标被 CG-2026-0028(d) 禁止）。属 checklist「tile 级顺序依赖（循环携带依赖）」类别 |
| 9 | 散写 `fwd_idx[idx]=slot`、`permuted_idx[slot]=idx//K`（kernel C） | 源散写 | 无 | **否** | **API 缺失 + 运行期下标**：无向量 scatter API；`permuted_idx[slot]` 为运行期下标 GM 写（GM 允许，向量载荷原子可靠，探针 P1-EP 实测）；UB 则被 CG-2026-0028(d) 禁止。`fwd_idx` 经 UB 暂存 + 动态 extent `T.copy`（块内一次），**块内偏移由循环变量 `c` 静态给定**（非运行期下标）。逐位置只写一次，无热点可再向量化 |
| 10 | `idx // top_k`（kernel C） | 源同 | 可 `T.Parallel` 逐元素（若单独成段） | **是（已并入散写循环的向量操作数）** | — |

**有向量形态但弃选者**：C5「`T.vsort` 排序驱动」（全向量）——弃选理由：(1) **UB 容量否决**：排序 numel=32768
需 `src+dst_value+dst_index` 三缓冲 ≈ 384KB > 192KB（`CONST-capacity-910B2C`）；(2) `T.vsort` 仅支持 fp16/fp32
（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3），int32 键需 cast，键 `remap(flat_ids)*numel+idx` 需 < 2^24
才精确（EP 下 `E_local×numel+numel ≈ 4.2M < 2^24` 可行，但大 `E_local/numel` 组合不安全）；(3) 排序复杂度
`O(numel·log²numel)`（R3 ≈ 7.4M vs C1-EP 的 70.8K）。

**向量化结论**：计数/前缀/输出等**与顺序无关**的部分已全部向量化（第 4/5/6/7/10 项：`T.cumsum` + `T.Parallel`）。
仅保留 4 处标量点（第 1/2/3/8/9 项：GM 直方图向量载荷原子、两级间接 remap、守卫分支、块内运行 rank、运行期下标散写），
理由类别：**API 缺失（无向量 scatter / 逐 lane 散加原语）**、**循环携带顺序依赖（稳定序要求，无法归约消除）**、
**循环携带量 + 标量 select 被 CG-2026-0028(c) 禁用**；均**非逐元素热点型可替代项**（每元素 O(1) 次 GM 访问），
由 §5.5 分核摊平（40 核）。**无可向量化而未向量化的逐元素标量循环**；C5 的全向量路线因容量/精度被否决。
**EP 未新增任何可向量化而未向量化的点**——`expert_map` 查表与过滤守卫是逐位置循环携带量，属上述既有类别。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本单元为纯整数索引/计数，**无浮点、无矩阵、无 spatial tiling**；I/O layout 为契约（`flat_ids(numel,)`、
> `expert_map(E_global,)`、5 张表长度 `E_local/E_local/E_local+1/numel/numel`）不变。核内「布局」决策 = 索引缓冲的
> 位置与分块轴（位置轴 numel）、以及**本地专家轴 `E_local`** 的向量化方式。

**轴质量评分**（向量化轴 = 本地专家轴 `E_local`（kernel B 前缀/求和，`T.Parallel(E_local)`）；分块轴 = 位置轴 numel）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | `E_local` 对 int32 向量宽度的整除 | `E_local` 为 EP 派生量，**任意值**（非 2 幂、非 32 倍）；`T.cumsum`/`T.Parallel` 对任意 `E_local` 合法——**探针 P2-EP 实测 `E_local ∈ {1,3,5,96,384}` 位精确**（含尾 lane 3/5） |
| 尾 lane 浪费率 | `(ceil(E_local/vw)*vw-E_local)/E_local` | `E_local` 非 32 倍时存在尾浪费（如 5 → 27/32 浪费），但 `E_local ≤ 384` 且**单核一次性**，绝对成本可忽略（UB ~4.5KB） |
| 累加链形态 | counts/前缀为 stride=1 逐元素；scan 为 `VCumsumOp` | 逐元素无跨步系数问题；scan 原语内部处理 |
| repack 代价 | 无核内重排 | 0（一维 int32 缓冲，无需转置/NZ） |
| UB 容量影响 | kernel B：8 buffer = `ub_acc/ub_bh/ub_incl/ub_sb/ub_ts/ub_to` `[1,E_local]` + `ub_off` `[1,E_local+1]` int32 + `ub_efo` `[1,E_local+1]` int64；kernel C：`ub_one`+`ub_val`+`ub_fwd[1,C]` | `E_local`=128 时 ≈4.5KB（B）+3.3KB（C）；上界 `E_local`=384 时 ≈13.5KB ≪ 192KB |
| 分块轴 numel | `C=ceildiv(numel,NB)` 静态；块核 1:1 | 位置轴仅用于块划分（非向量化轴），块内为串行 rank（顺序依赖） |

**候选矩阵**（EP 计数排序数据流形态 + 索引缓冲位置 + 重映射落点）：

| # | 方案 | 关键决策 | 代价 / 收益 | 采纳 |
|---|------|---------|------------|------|
| L1-EP | **三 kernel GM 交接 + 扫描内联 remap**（A：守卫向量载荷原子块直方图；B：`T.cumsum` 前缀 + `slot_base`；C：守卫块并行运行 rank + `-1` 哨兵） | 索引缓冲 `bh`/`slot_base`/`run` 驻 **GM**；`expert_map` 在 A/C 内联读；kernel 边界隔离计数/前缀/散写 | 3 次 launch + GM scratch 60KB（`E_local`=128）+ UB ≈ 8KB；**构造规避 CG-2026-0027/0028/0029**；探针实测可编译位精确 | **主选** |
| L2-EP | **全局域直方图 + 单点聚合**（`bh_g[NB,E_global]` → `counts_g` → B 内聚合到 `E_local`） | 直方图落全局域 | scratch 3×（184KB）；聚合为 `E_global` 标量散加（无向量 scatter） | 弃选（经济性；F1 证明语义等价故按代价否决） |
| L3-EP | **物化重映射 ids 数组**（第 4 kernel / 前置 pass） | `mapped` 落 GM | 多 256KB 流量 + 多一趟扫描；守卫仍需在计数侧 | 弃选（纯增开销） |
| L4 | 单 kernel 单 scope（UB 内计数 + scan + 标量消费） | UB 内完成全部 | 少 launch | 弃选（可行性：CG-2026-0027/0028） |
| L5 | 内联进 gather 核（表不落 GM） | 同 block Vector→(gather) 交接 | 理论零表 GM | 弃选（可行性：CG-2026-0010/0026） |
| L6 | `T.vsort` 全向量排序 | 排序即计数排序（需先算 remap 键） | 全向量、无标量 | 弃选（容量 384KB>192KB；dtype 仅 fp16/fp32） |
| L7 | host 侧算表 | CPU | — | 弃选（违反零 CPU 同步 + host 禁改输入，`references/ascend-constraints.md` §4） |
| L8 | 单核全局顺序散写（备选） | 1 核 + 全局 `run[E_local]` | 结构最简、少 `block_base` 往返，但 40× 串行 | **备选**（实验裁决计划） |

**布局决策结论**：选定 **L1-EP（三 kernel GM 交接 + 扫描内联 remap）** —— 索引缓冲 `bh[NB,E_local]`/
`slot_base[NB,E_local]`/`run[NB,E_local]` 驻 **GM**（跨 kernel 可读、规避 UB 运行期下标），kernel B 的向量化轴为
**本地专家轴 `E_local`**（`T.Parallel(E_local)` + `T.cumsum`），分块轴为**位置轴 numel**（`NB` 块，块核 1:1）；
`expert_map` 作为**只读小表**在 kernel A/C 内经两级 GM 读内联消费（不物化）。repack 路径：无。
该结论已同步 §3.3 伪代码、§3.2 API 映射、§4 内存规划与 §6 循环结构；L2-EP/L3-EP/L4/L5/L6/L7 的弃选理由均已附
`docs/` 路径、探针或 F1 等价性证据，符合 `negative-claim-evidence.md` §2；L8 作为可实现的备选
（实验裁决计划三件套见 §1.6.0 末）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（单一模式）。本单元为**三 kernel 纯 Vector 整数索引单元**：kernel A（两级查表 +
守卫 + 向量载荷原子块直方图 + 核内 `T.serial`）、kernel B（`T.Parallel`/`T.cumsum`/`T.copy`，含 UB 暂存）、
kernel C（核内 `T.serial` 运行 rank + 守卫 + 运行期下标 GM 散写），**不含 `T.gemm`、不含跨 scope 交接**，
故 Developer 形态完全可用。调度参数指定 `programming_mode=developer`，与本结论一致。**无需 Expert 回退**。

### 2.2 选型理由

- 数据通路为**整数查表/计数/前缀/散写**（GM 标量读 / 向量载荷原子 / `T.cumsum` / `T.Parallel` / `T.copy`），
  无浮点、无矩阵——按 `AGENTS.md` 的 Developer-mode MixCV 触发规则（Cube `T.gemm` + 至少一个 v-prefix op），
  **本单元不满足触发判据**（无 `T.gemm`），归为「整数 Vector 计数单元」，走 Developer。
- **规避三条工具链阻断**：Developer persistent + `T.gemm` + 整型索引段共存即崩（CG-2026-0010）——本单元**无 `T.gemm`**；
  跨引擎同 block 交接不可编译（CG-2026-0026）——本单元**无跨引擎交接**；单 scope「向量产出 → 标量消费」不可同步
  （CG-2026-0027）——本单元用**三 kernel GM 交接**隔离（标量只读跨 kernel GM）。
- Developer 用 `T.alloc_ub` + `T.Parallel`/`T.cumsum`/`T.copy` + 向量载荷 `T.atomic_add` 表达，实现负担低。

### 2.3 模式影响

| 维度 | Developer（选定） |
|------|------------------|
| 内存分配 | kernel A：`T.alloc_ub((1,1))` 常量载体；kernel B：`T.alloc_ub`（counts/incl/off/bh/slot_base + 表输出暂存 `ub_ts`/`ub_to`/`ub_efo`）；kernel C：`ub_one`/`ub_val`/`ub_fwd(1,C)`；GM scratch：`bh`/`slot_base`/`run`；GM 输出表 |
| 扫描原语 | `T.cumsum`（kernel B，单核一次）/ `T.Parallel`（kernel B）/ 向量载荷 `T.atomic_add` + GM 标量读（kernel A/C） |
| 跨 scope 交接 | **无 in-block 交接**（三 kernel 各自单 Vector scope；数据经 **GM** + kernel 间同 stream 顺序） |
| 同步 | 各 scope 内编译器自动同步；**不依赖 auto-sync 跨「向量产出 → 标量消费」**（该形态已弃用，CG-2026-0027）；kernel 间由 stream 顺序保证 |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的优化后公式。逐步骤映射到 §1.6.2 的向量化结论（索引段为**整数** GM 标量读 / 向量载荷原子 /
> `T.cumsum` / `T.Parallel`，无 v-prefix 浮点算子）。A* = kernel A；B* = kernel B；C* = kernel C；
> `E_l` = `E_local`。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| A1 | `g = flat_ids[idx]`；`le = expert_map[g]` | 两级 GM 标量读（运行期下标）（§1.6.2 #2） |
| A2 | `if le >= 0: bh[b, le] += 1`（`b=idx//C`） | 守卫本地域块直方图（向量载荷原子作可靠 store）（§1.6.2 #1/#3） |
| B1 | `counts[l] = Σ_b bh[b,l]` | 本地域列求和（向量加）（§1.6.2 #4） |
| B2 | `incl = cumsum(counts); off[0]=0; off[1:E_l+1]=incl[0:E_l]` | 硬件 scan + 1-lane 移位（E1）（§1.6.2 #5） |
| B3 | `true_sizes[l]=counts[l]`；`true_offsets[l]=off[l]`；`efo[l]=int64(off[l])` | 3 张表输出（E3）：`T.Parallel` 写 UB 暂存 → `T.copy` 落 GM（§1.6.2 #6） |
| B4 | `slot_base[b,l] = off[l] + Σ[b'<b] bh[b',l]` | 块前缀 + 本地专家偏移预加（E4）（§1.6.2 #7） |
| C1 | `g=flat_ids[idx]; le=expert_map[g]`；`if le<0: fwd=-1` else `r=run[b,le]; run[b,le]=r+1; slot=slot_base[b,le]+r` | 守卫 + 块内运行 rank（E2-EP）（§1.6.2 #8） |
| C2 | `fwd_idx[idx]=...`；`permuted_idx[slot]=idx//top_k` | 哨兵/运行期下标散写（E2-EP/E5）（§1.6.2 #9/#10） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| A1 | `le = expert_map[flat_ids[idx]]` | `g = flat_ids[base + c]; le = expert_map[g]` | 两级运行期下标 **GM 标量读**（探针 P1-EP 实测） | Developer |
| A2 | `if le>=0: bh[b,le]+=1` | `ub_one[0,0]=1`；`if le >= 0: T.atomic_add(bh[core_id, le], ub_one[0,0:1], [1])` | 普通 `if` 守卫；**向量载荷**原子作可靠 GM store（非标量载荷，CG-2026-0029；非裸标量 store，多核丢写） | Developer |
| B1 | `counts=Σ_b bh[b,:]` | `T.copy(bh[b,0:E_l], ub_bh[0,0:E_l])` + `for e in T.Parallel(E_l): ub_acc[0,e]=ub_acc[0,e]+ub_bh[0,e]` | `(1,E_l)` 向量加（核内 `T.serial(NB)` 外层） | Developer |
| B2 | 含前缀 scan | `T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)` | 2D `(1,E_l)` int32；**唯一一次 scan** | Developer |
| B2 | 排他移位 | `ub_off[0,0]=0`；`T.copy(ub_incl[0,0:E_l], ub_off[0,1:E_l+1])` | 1-lane 移位切片 | Developer |
| B3 | 表输出 | `for e in T.Parallel(E_l): ub_ts[0,e]=ub_acc[0,e]; ub_to[0,e]=ub_off[0,e]`；`for k in T.Parallel(E_l+1): ub_efo[0,k]=T.Cast("int64", ub_off[0,k])`；随后 `T.copy(ub_ts[0,0:E_l], true_sizes[0:E_l])`、`T.copy(ub_to[0,0:E_l], true_offsets[0:E_l])`、`T.copy(ub_efo[0,0:E_l+1], efo[0:E_l+1])` | **UB 暂存 + `T.copy` 落 GM**（`T.Parallel` 体内禁直写 GM，§2.3）；含 int32→int64 cast | Developer |
| B4 | 块前缀 | `for b in T.serial(NB): for e in T.Parallel(E_l): ub_sb[0,e]=ub_acc[0,e]; T.copy(ub_sb[0,0:E_l], slot_base[b,0:E_l]); T.copy(bh[b,0:E_l], ub_bh[0,0:E_l]); for e in T.Parallel(E_l): ub_acc[0,e]=ub_acc[0,e]+ub_bh[0,e]` | 运行累加（向量加） | Developer |
| C1 | 守卫 + 运行 rank | `g=flat_ids[idx]; le=expert_map[g]`；`if le < 0: ub_fwd[0,c]=-1` `else: r=run[core_id,le]; T.atomic_add(run[core_id,le], ub_one[0,0:1], [1]); slot=slot_base[core_id,le]+r; ub_fwd[0,c]=slot` | 普通 `if`（禁标量 `T.if_then_else`，CG-2026-0028c）；`run` 标量读 + 向量载荷原子自增 | Developer |
| C2 | 哨兵 + 散写 | `ub_fwd[0,c]=-1/slot`；`ub_val[0,0]=idx//top_k`；`T.atomic_add(permuted_idx[slot], ub_val[0,0:1], [1])`（仅本地）；块尾 `ct=T.min(C, numel-base)`；`if ct>0: T.copy(ub_fwd[0,0:ct], fwd_idx[base:base+ct])` | `fwd_idx` 经 UB 暂存 + 动态 extent `T.copy`；`permuted_idx` host 预零 + 向量载荷原子 | Developer |

> 说明：本单元无浮点逐元素运算，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix 算术 API；整数向量化仅用
> `T.Parallel` / `T.cumsum` / `T.copy`（`T.Parallel` 体内只读写 UB 暂存，GM 一律经 `T.copy`）；整数 GM 写用
> **向量载荷 `T.atomic_add`**（可靠 store，见 §3.4）。

### 3.3 计算伪代码

> `T.Tensor` 第二位置参数直接传 dtype。**scratch `bh`/`run` 与输出 `permuted_idx` 由 host 以 `torch.zeros` 零初始化后
> 传入**（kernel 参数，非 `out_idx`）；其余输出表由 host `torch.empty` 分配并传入。本单元 = 三 kernel 链式。
> `NB`/`C`/`E_g`/`E_l`/`top_k` 均为编译期常量。

```python
import functools
import tilelang
import tilelang.language as T

# 编译期常量（代表 EP shape: numel=32768, E_g=384, E_l=128, top_k=8）
#   P    = NPUUtils.get().get_aicore_num() * 2      # 纯 Vector 物理核数（§5.5 实查 = 40）
#   NB   = min(P, numel)                             # 位置块数 = 核数
#   C    = ceildiv(numel, NB)                        # 每块位置数（静态边界）
_PASS_CFG = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}


# ---- kernel A: 本地域守卫块直方图（两级查表 + 向量载荷原子）------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_count_ep_kernel(numel, e_global, e_local, nb, cblk):
    @T.prim_func
    def _count_ep_main(flat_ids: T.Tensor([numel], "int32"),
                       expert_map: T.Tensor([e_global], "int32"),
                       bh: T.Tensor([nb, e_local], "int32")):   # host 零初始化 GM scratch
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                              # 静态边界
                if base + c < numel:                              # 普通 if 守卫（非 T.if_then_else）
                    g = flat_ids[base + c]                        # level-1 GM 标量读
                    le = expert_map[g]                            # level-2 GM 标量读（pull-down remap）
                    if le >= 0:                                   # EP 过滤守卫
                        T.atomic_add(bh[core_id, le], ub_one[0, 0:1], [1])  # 向量载荷（可靠 GM store）

    return _count_ep_main


# ---- kernel B: 跨块求和 + 前缀 + 输出 + slot_base（单核，E_local 轴）---------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_prefix_ep_kernel(numel, e_global, e_local, nb, cblk, top_k):
    E = e_local

    @T.prim_func
    def _prefix_ep_main(bh: T.Tensor([nb, E], "int32"),
                        true_offsets: T.Tensor([E], "int32"),
                        true_sizes: T.Tensor([E], "int32"),
                        efo: T.Tensor([E + 1], "int64"),
                        slot_base: T.Tensor([nb, E], "int32")):
        with T.Kernel(1, is_npu=True) as (_cid, _subid):            # 单核一次
            ub_acc  = T.alloc_ub((1, E), "int32")
            ub_bh   = T.alloc_ub((1, E), "int32")
            ub_incl = T.alloc_ub((1, E), "int32")
            ub_off  = T.alloc_ub((1, E + 1), "int32")
            ub_sb   = T.alloc_ub((1, E), "int32")
            ub_ts   = T.alloc_ub((1, E), "int32")                   # true_sizes 暂存
            ub_to   = T.alloc_ub((1, E), "int32")                   # true_offsets 暂存
            ub_efo  = T.alloc_ub((1, E + 1), "int64")               # efo 暂存（int64）

            # B1: counts[l] = 本地域列求和（与顺序无关，可向量化）
            for e in T.Parallel(E):
                ub_acc[0, e] = 0
            for b in T.serial(nb):                                  # 静态边界
                T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

            # B2: 唯一一次 scan（int32 / (1,E_local) 探针 P2-EP 实测可用）
            T.cumsum(ub_acc, ub_incl, dim=1, reverse=False)
            ub_off[0, 0] = 0
            T.copy(ub_incl[0, 0:E], ub_off[0, 1:E + 1])             # E1: 1-lane 移位

            # B3: 三张表输出（先写 UB 暂存，再 T.copy 落 GM）
            for e in T.Parallel(E):
                ub_ts[0, e] = ub_acc[0, e]                          # counts
                ub_to[0, e] = ub_off[0, e]                          # off
            for k in T.Parallel(E + 1):
                ub_efo[0, k] = T.Cast("int64", ub_off[0, k])        # E3: int32 -> int64
            T.copy(ub_ts[0, 0:E], true_sizes[0:E])
            T.copy(ub_to[0, 0:E], true_offsets[0:E])
            T.copy(ub_efo[0, 0:E + 1], efo[0:E + 1])

            # B4: slot_base[b,l] = off[l] + sum_{b'<b} bh[b',l]（E4）
            for e in T.Parallel(E):
                ub_acc[0, e] = ub_off[0, e]
            for b in T.serial(nb):
                for e in T.Parallel(E):
                    ub_sb[0, e] = ub_acc[0, e]
                T.copy(ub_sb[0, 0:E], slot_base[b, 0:E])
                T.copy(bh[b, 0:E], ub_bh[0, 0:E])
                for e in T.Parallel(E):
                    ub_acc[0, e] = ub_acc[0, e] + ub_bh[0, e]

    return _prefix_ep_main


# ---- kernel C: 守卫的确定性块并行稳定 rank 散写（块核 1:1）------------------
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)
def _scan_scatter_ep_kernel(numel, e_global, e_local, nb, cblk, top_k):
    E = e_local

    @T.prim_func
    def _scatter_ep_main(flat_ids: T.Tensor([numel], "int32"),
                         expert_map: T.Tensor([e_global], "int32"),
                         slot_base: T.Tensor([nb, E], "int32"),
                         run: T.Tensor([nb, E], "int32"),           # host 零初始化 GM scratch
                         fwd_idx: T.Tensor([numel], "int32"),
                         permuted_idx: T.Tensor([numel], "int32")):  # host 零初始化（哨兵兜底）
        with T.Kernel(nb, is_npu=True) as (core_id, _subid):
            ub_one = T.alloc_ub((1, 1), "int32")
            ub_val = T.alloc_ub((1, 1), "int32")
            ub_fwd = T.alloc_ub((1, cblk), "int32")                 # 块内 fwd_idx 暂存
            ub_one[0, 0] = 1
            base = core_id * cblk
            for c in T.serial(cblk):                                # 静态边界（循环携带 rank）
                idx = base + c
                if idx < numel:
                    g = flat_ids[idx]                               # level-1 GM 标量读
                    le = expert_map[g]                              # level-2 GM 标量读
                    if le < 0:
                        ub_fwd[0, c] = -1                           # EP 哨兵（非本地）
                    else:
                        r = run[core_id, le]                        # 块内运行计数（块核 1:1，无争用）
                        T.atomic_add(run[core_id, le], ub_one[0, 0:1], [1])  # 向量载荷原子自增
                        slot = slot_base[core_id, le] + r           # E2-EP: 确定性稳定序 slot
                        ub_fwd[0, c] = slot                         # UB 暂存（循环变量下标）
                        ub_val[0, 0] = idx // top_k                 # E5: //（非 >>）
                        T.atomic_add(permuted_idx[slot], ub_val[0, 0:1], [1])  # 仅本地对写
            ct = T.min(cblk, numel - base)
            if ct > 0:
                T.copy(ub_fwd[0, 0:ct], fwd_idx[base:base + ct])    # 动态 extent 落 GM

    return _scatter_ep_main


# ---- 公共入口：同 stream 顺序跑 A -> B -> C --------------------------------
@functools.lru_cache(maxsize=32)
def _make_scan_kernel_nopad_ep(numel, e_global, e_local, top_k):
    P = 40
    nb = min(P, numel) if numel > 0 else 1
    cblk = (numel + nb - 1) // nb if numel > 0 else 1
    kA = _scan_count_ep_kernel(numel, e_global, e_local, nb, cblk)
    kB = _scan_prefix_ep_kernel(numel, e_global, e_local, nb, cblk, top_k)
    kC = _scan_scatter_ep_kernel(numel, e_global, e_local, nb, cblk, top_k)

    def run(flat_ids, expert_map, bh, true_offsets, true_sizes, efo, slot_base,
            run_buf, fwd_idx, permuted_idx):
        kA(flat_ids, expert_map, bh)
        kB(bh, true_offsets, true_sizes, efo, slot_base)
        kC(flat_ids, expert_map, slot_base, run_buf, fwd_idx, permuted_idx)
        return fwd_idx, permuted_idx

    return run
```

> **注 1（规避 CG-2026-0027/0028/0029）**：三 kernel 均**不在标量上下文读本 scope 向量 op 写入的 UB 值**——
> kernel A/C 的标量只读 **GM**（`flat_ids`/`expert_map`/`slot_base`/`run`）；无 `T.alloc_local`、无 int32 `>>`
> （`idx//top_k`）、无标量 `T.if_then_else`（用普通 `if`）、每个 kernel **至多一次 `T.cumsum`**、不用运行期下标读 UB。
> kernel A/C 的运行期下标读写仅针对 **GM**（探针 P1-EP 全 shape 实测可编译位精确）。
> **注 2（输出承接源语义）**：`fwd_idx`/`permuted_idx` 互逆且为**稳定序**，非本地对 `fwd_idx=-1`；与 §8.1 EP golden
> 逐位一致（§1.6.1 E2-EP）。
> **注 3（零初始化）**：`bh`/`run` 必须 host 侧 `torch.zeros` 预清零；`permuted_idx` 亦须 `torch.zeros`（源 op 同款，
> 兜底 `[M_local, numel)` 未写 slot）。
> **注 4（scan int32）**：`T.cumsum` int32/`(1,E_local)` 路径设备实测可用（非 EP 版设备实测 + 本变体探针 P2-EP
> `E_local ∈ {1,3,5,96,384}` 位精确）；int32 scan 以 scalar instruction lower（低效但 `E_local ≤ 384` 单核一次可忽略）。
> **注 5（`E_local ≥ 1`）**：`E_local = 0` 使 kernel B 的零长 UB 分配触发 MLIR `strides must not be zero`（探针实测）
> ——host 侧须保证 `E_local ≥ 1`（或走 `M_local = 0` 退化分支，§9.4 R-1）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| **两级标量 GM 间接**（`g=flat_ids[idx]; le=expert_map[g]`，运行期下标） | **本变体设计期探针 P1-EP**（`history_version/design_probe_ep_primitives.py`，代表 shape `numel=32768/E_g=384/E_l=128/NB=40` 等 8 例编译通过 + 位精确） | 已验证（EP 新增，探针实测） |
| 守卫分支（`if le >= 0:` / `if le < 0:`，普通 `if`） | 探针 P1-EP（含全非本地 `remote_only` / 混合 / 离散 map 用例） | 已验证（普通 `if`；**禁用**标量 `T.if_then_else`，CG-2026-0028c） |
| **向量载荷 `T.atomic_add(dst_rt_indexed, ub_src, [n])`** | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.4；非 EP 版 Stage 3 实测（`_make_scan_kernel_nopad.py` 模块头注）；**本变体探针 P1-EP 在两级间接 + 守卫形态下实测** | 已验证（多核可靠 GM 写路径；**非**标量载荷，CG-2026-0029） |
| `T.cumsum(src, dst, dim=1, reverse=False)`（int32, `(1,E_local)`） | `docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1（int32 支持）；**探针 P2-EP 实测 `E_local ∈ {1,3,5,96,384}`** | 已验证（含非 32 倍尾 lane） |
| `T.Parallel(N)` 整型向量化（kernel B 求和/表暂存/运行累加） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.2.1；§2.3「下标不能做变换」「循环体须用 alloc 的 UB buffer」；非 EP 版 Stage 2 探针 | 已验证（索引未变换；体内只写 UB 暂存，GM 经 `T.copy`） |
| `T.copy` 切片（GM→UB / UB→GM，含 1-lane 移位、动态 extent 块内 fwd 落盘、int32→int64 表写出） | `docs/Tilelang.language/内存操作/T.copy.md`；非 EP 版 Stage 3 实现（动态 extent `T.copy` 落 `fwd_idx`） | 已验证（非 EP 版 Stage 3 实测同款形态） |
| `T.Kernel(n, is_npu=True)`（1D 核；A/C = `NB`，B = 1） | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；非 EP 版 Stage 3 实现 | 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | 已验证 |
| `T.Cast("int64", x)`（int32→int64） | `docs/Tilelang.language/数据类型转换操作/`；探针 P2-EP（`ub_efo` int64 暂存 + `T.copy`）+ 非 EP 版 Stage 3 同款 | 已验证（编译器 warning：`Parallel` cast 用固定 `rint` 舍入——整数 cast 无影响） |
| ~~裸标量 GM store（`bh[core_id,e]=bh[core_id,e]+1` / `fwd_idx[idx]=slot`）~~ | 非 EP 版 Stage 3 实测：多核 Vector kernel 直接标量 GM store **随机丢写**（`_make_scan_kernel_nopad.py` 模块头注，tilelang 0.1.2+2853d71 / Ascend910B2C） | **不可靠**；本设计改用向量载荷原子 / UB 暂存 + `T.copy` |
| ~~`T.atomic_add(dst, 1)`（标量载荷）~~ | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2.1 声明 `src` 支持 scalar，§2.4 示例仅向量载荷；实测标量载荷前端失败 | **不可用**（CG-2026-0029）；本设计用向量载荷 |
| ~~fetch-add（`tl_atomic_add_offset`）~~ | 源 `_atomic_helper.h`（CUDA 定制 extern）；`T.atomic_add` 降 `hivm::StoreOp`（**无返回值**） | **NPU 无等价**（CG-2026-0030），本设计不依赖 |
| ~~int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum` / 单 kernel 两次 `T.cumsum`~~ | — | **不可用**（CG-2026-0028 a/b/c/e/f）；本设计全部回避 |
| ~~`E_local = 0` 的 kernel B（零长 UB）~~ | 本变体探针（`E_local=0` → MLIR `strides must not be zero`） | **不可用**；host 须保证 `E_local ≥ 1`（§9.4 R-1） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（三 kernel 均为 1D `T.Kernel`） | 一维核 + 核内任务计算 |
| GEMM M/N 需为 block 整数倍 | 否（本单元无矩阵） | — |
| L0C 容量（128KB） | 否（本单元无 L0C） | — |
| 物理核数限制 | 是（`NB=min(P,numel)`，P=40 实查） | kernel A/C `T.Kernel(NB)`、块核 1:1、核内静态 `T.serial(C)`；kernel B 单核（§5.5） |
| host 侧禁止改动输入 | 是（需 host 分配 scratch/输出） | host 仅**分配** scratch（`torch.zeros`）与输出（`torch.empty/zeros`），不改动 `flat_ids`/`expert_map` 真实内容；无 padding（`references/ascend-constraints.md` §4） |
| 一维核超发被串行调度 | 是（源单 block×1024 thread 的形态） | 固定核数 `NB=min(P,numel) ≤ P`，无超发（§0.6 R3） |
| `T.Parallel` 下标禁变换 / 循环体禁直写 GM | 是 | kernel B 用切片 `T.copy` 完成 1-lane 移位与落盘；求和/表暂存索引未变换；**GM 写一律经 UB 暂存 + `T.copy`**（B3）或**向量载荷原子** |
| 同 scope「向量产出 → 标量消费」不可同步（CG-2026-0027） | 是 | 三 kernel GM 交接：标量只读跨 kernel **GM**，不读本 scope 向量产出的 UB |
| 跨引擎同 block 交接不可编译（CG-2026-0026） | 否 | 本单元无跨引擎交接 |
| int32 `>>` / `alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum`（CG-2026-0028） | 是 | 全回避：`//` 替代 `>>`；无 `alloc_local`；普通 `if`（含 EP 守卫）替代 `T.if_then_else`；scan 只用 `(1,E_local)` 单行且每 kernel 一次 |
| UB 运行期下标读/写（CG-2026-0028(d)） | 是（源 SMEM 运行期下标 + EP 两级查表） | 运行期下标只用于 **GM**（`flat_ids`/`expert_map`/`bh`/`run`/`slot_base`/`permuted_idx`），UB 全为静态索引（`ub_fwd[0,c]` 的 `c` 为循环变量） |
| 多核标量 GM store 丢写（非 EP Stage 3 实测） | 是 | 所有多核 GM 写改**向量载荷 `T.atomic_add`** 或 **UB 暂存 + `T.copy`**（§3.3 全程遵守） |
| `E_local = 0` 退化（零长 UB） | 是（EP 特有） | host 保证 `E_local ≥ 1`；`M_local = 0`（全非本地）本身可正常工作（`E_local ≥ 1` 时） |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| 承载方式 | 单 block（grid=1, 1024 thread）5 相位 | **三 kernel GM 交接**（A 计数 / B 前缀+输出 / C 散写） | §0.6 R1 |
| 专家域 | 全局域 `E`（非 EP）/ 本地域 `E_local`（EP，内联查表） | 本地域 `E_local`（表 / 直方图 / scan 轴） | §0.5 O-EP4 / §0.6 |
| EP 重映射 | 共享表内联读 `expert_map[g]` | **两级 GM 标量读链**（`flat_ids` → `expert_map`），探针实测 | §0.5 O-EP1 |
| EP 过滤 | `if le>=0`（CUDA 分支） | 普通 `if` 守卫（**禁**标量 `T.if_then_else`） | §0.5 O-EP2 |
| 非本地对 | `fwd_idx=-1`（散写） | `-1` 写入 UB 暂存行 → 一次动态 extent `T.copy` 落 GM | §0.5 O-EP3 / §3.2 C2 |
| 直方图 | SMEM `atomic_add`（运行期下标） | GM `bh[NB,E_local]` 守卫计数（**向量载荷原子**作可靠 store） | §0.5（CG-2026-0028d/0029 + 多核丢写） |
| 表输出 | 逐元素写 GM（thread 并行） | `T.Parallel(E_local)` 写 UB 暂存 + `T.copy` 落 GM | §3.2 B3（`T.Parallel.md` §2.3） |
| 前缀扫描 | thread0 `T.serial(E_local)` int64 | kernel B 硬件 `T.cumsum` + 1-lane 移位（int32），efo cast int64 | E1 / E3 |
| 散写 slot | `tl_atomic_add_offset` **fetch-add** | 确定性 `off[l]+block_base[b,l]+块内 rank` | §0.6 R2 / E2-EP |
| `run` RMW | 无（由 fetch-add 隐式） | 标量 GM 读 + 向量载荷原子自增 | §0.6 R2 |
| 分核 | 1 block × 1024 thread | `NB=min(P,numel)` 块核 1:1 + 核内 `T.serial(C)`；B 单核 | §0.6 R3 / §5.5 |
| 同步 | `T.sync_threads()` 相位 | 各 scope auto-sync + kernel 间 stream 顺序 | §7 |
| 输出分配 | `out_idx=[]`（host `torch.empty/zeros`） | 保留：host 分配 + 传入（`bh`/`run`/`permuted_idx` 零初始化） | §4.1/§4.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md` / `_make_scan_kernel_nopad.py` | **极高（非 EP 版，含 Stage 3 实测与集成）** | 三 kernel GM 交接；`T.cumsum`(1,E) + 1-lane 移位；块核 1:1 + 核内 `T.serial`；向量载荷原子 / UB 暂存 + `T.copy` 可靠写路径；CG-2026-0027/0028/0029/0030 规避清单 |
| `examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_primitives.py` | — | 本变体 D-3 设计期探针：EP 两级间接 + 守卫 + 向量载荷原子的最小 kernel 实测（代表 shape 8 例全过） |
| `examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_prefix.py` | — | 本变体 D-3 设计期探针：`E_local` 专家轴（含非 32 倍尾 lane）位精确实测 |
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` / `_tile_scheduler_kernel.py` | **极高（同族整数索引算子，含 Stage 3 实测）** | 两 kernel GM 交接；单核 scan + 固定核数 kernel B + 核内 `T.serial`；CG-2026-0027 规避 |
| `examples/TileOPs/tests/ops/test_moe_permute_nopad.py` | **极高（本单元契约/golden 参照）** | 顺序无关一致性校验（L105–L111） |
| `examples/indexer/indexer_bwd.py` | 中（GM 运行期下标写先例，**向量载荷原子**） | `T.atomic_add(dIndexK[target_k_idx, 0], dk_temp_ub[j,0], [head_dim])`（运行期下标 GM scatter） |
| `examples/deepseek_v32/sparse_mla_bwd.py` | 中（GM 运行期下标写先例，**向量载荷原子**） | `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])` |
| `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 高（scan + 持久分核） | `T.cumsum(..., dim=1)` 用法；`T.serial` 静态核内串行 |
| `testing/autotune/anneal/moe_grouped_gemm_anneal.py` | 中（调度器反例） | `T.Kernel(max_tiles)` **超发核**（反例，本设计按 §5.5 修正） |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `flat_ids` | `(numel,)` | int32 | 每位置**全局**专家号（`= topk_ids.flatten()`；host 侧 view，不改内容） |
| `expert_map` | `(E_global,)` | int32 | **EP 下拉重映射表**（`-1` = 非本地）；只读 |
| `bh` | `(NB, E_local)` | int32 | **GM scratch（输入侧）**，host `torch.zeros` 零初始化 |
| `run` | `(NB, E_local)` | int32 | **GM scratch（输入侧）**，host `torch.zeros` 零初始化 |
| `slot_base` | `(NB, E_local)` | int32 | **GM scratch（kernel B 产 / C 消费）**，host `torch.empty` |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `true_offsets` | `(E_local,)` | int32 | 每本地专家紧致起始 `off[l]`（kernel B 写） |
| `true_sizes` | `(E_local,)` | int32 | 每本地专家 token 数 `counts[l]`（kernel B 写） |
| `expert_first_token_offset` | `(E_local + 1,)` | int64 | 本地域非填充排他前缀 `off`（kernel B 写，int32 cast） |
| `fwd_idx` | `(numel,)` | int32 | flat 位置 → 紧致 slot；**非本地对写 `-1`**（kernel C 经 UB 暂存 + `T.copy` 写） |
| `permuted_idx` | `(numel,)` | int32 | slot → token 行（kernel C 写，仅本地对；host 零初始化，`[M_local, numel)` 保持 0） |

> **对接契约（本单元产出 / 胞弟 gather 消费）**：① `permuted_idx` int32；② `fwd_idx` 与 `permuted_idx` 在
> **本地槽域 `[0, M_local)`** 上互逆且为**稳定序**（与 §8.1 EP golden 一致），非本地对 `fwd_idx=-1`；
> ③ `numel`/`top_k`/`E_global`/`E_local` 为两单元共享编译期常量；④ **kernel A → B → C → 胞弟 gather**
> 同 stream 顺序 launch（无 in-block 同步、无 CPU 同步）；⑤ `perm_h` 恒 `numel` 行，`[M_local, numel)` 为 row-0 占位
> （源 op 同款，tile 调度器不访问）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 所在 kernel / 用途 |
|-----------|-------|-------|----------|--------------------|
| `ub_one` | `(1, 1)` | int32 | UB | A/C：常量 1 载体（向量载荷原子源） |
| `ub_val` | `(1, 1)` | int32 | UB | C：`idx // top_k` 载体（向量载荷原子源） |
| `ub_fwd` | `(1, C)` | int32 | UB | C：块内 `fwd_idx` 暂存（含 `-1` 哨兵），块尾一次 `T.copy` 落 GM |
| `ub_acc` | `(1, E_local)` | int32 | UB | B：counts 累加 / slot_base 运行累加 |
| `ub_bh` | `(1, E_local)` | int32 | UB | B：`bh` 逐块载入 |
| `ub_incl` | `(1, E_local)` | int32 | UB | B：含前缀 scan 输出 |
| `ub_off` | `(1, E_local+1)` | int32 | UB | B：排他前缀 |
| `ub_sb` | `(1, E_local)` | int32 | UB | B：`slot_base` 行写出 |
| `ub_ts` | `(1, E_local)` | int32 | UB | B：`true_sizes` 暂存（→ `T.copy` 落 GM） |
| `ub_to` | `(1, E_local)` | int32 | UB | B：`true_offsets` 暂存（→ `T.copy` 落 GM） |
| `ub_efo` | `(1, E_local+1)` | int64 | UB | B：`efo` int64 暂存（int32→int64 cast，→ `T.copy` 落 GM） |
| `bh` | `(NB, E_local)` | int32 | **GM** | A 写 / B 读：本地域块直方图 |
| `slot_base` | `(NB, E_local)` | int32 | **GM** | B 写 / C 读：`off[l]+block_base` |
| `run` | `(NB, E_local)` | int32 | **GM** | C 读写：块内运行计数 |

### 4.4 内存搬运路径

```
[kernel A · Vector · NB 核（块核 1:1）]
 GM[flat_ids(numel) i32] --level-1--> GM[expert_map(E_g) i32] --level-2(le)-->
     if le>=0: 向量载荷 T.atomic_add --> GM[bh(NB,E_local) i32]（host 零初始化）
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel B · Vector · grid=1]
  GM[bh] --T.copy--> UB[ub_bh] --T.Parallel(E_local) 累加--> UB[ub_acc=counts]
        --T.cumsum(dim=1)--> UB[ub_incl] --1-lane 移位--> UB[ub_off]
        --T.Parallel(E_l)/(E_l+1)--> UB[ub_ts/ub_to/ub_efo] --T.copy--> GM[true_sizes(E_l)] /
             GM[true_offsets(E_l)] / GM[efo(E_l+1) i64]（表输出经 UB 暂存，禁 T.Parallel 直写 GM）
  GM[bh] + UB[ub_off] --串行 b 运行累加--> GM[slot_base(NB,E_l) i32]
                                                          │ （kernel 边界，同 stream 顺序）
                                                          ▼
[kernel C · Vector · NB 核（块核 1:1）]
  GM[flat_ids] + GM[expert_map] + GM[slot_base] + GM[run(零初始化)]
      --串行: le<0 ? ub_fwd[0,c]=-1 : (r=run[b,le]; 向量载荷原子自增; slot=slot_base[b,le]+r;
              ub_fwd[0,c]=slot; 向量载荷原子写 GM[permuted_idx[slot]])-->
      UB[ub_fwd] --动态 extent T.copy--> GM[fwd_idx(numel) i32]
                                                          │ （同 stream 顺序 launch）
                                                          ▼
   [胞弟 _make_gather_kernel_nopad：perm_h[slot]=hidden_states[permuted_idx[slot]]]
```

> **层级说明**：本单元为**整数 Vector 计数/索引**，数据路径只有 **GM → UB**（`bh`）与 **UB → GM**（表、`slot_base`、
> 块内 `fwd_idx`）及 **GM 标量读 + 向量载荷原子写**（`flat_ids`/`expert_map`/`run`/`permuted_idx`），
> **无 L1/L0A/L0B/L0C**（无 Cube 矩阵计算）。

### 4.5 UB 内存预算（Developer；代表 `E_local=128`，上界 `E_local=384`）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| **kernel B（主用 UB 的 kernel）** | | | |
| `ub_acc` | (1,128) | int32 | 512 |
| `ub_bh` | (1,128) | int32 | 512 |
| `ub_incl` | (1,128) | int32 | 512 |
| `ub_off` | (1,129) | int32 | 516 |
| `ub_sb` | (1,128) | int32 | 512 |
| `ub_ts` | (1,128) | int32 | 512 |
| `ub_to` | (1,128) | int32 | 512 |
| `ub_efo` | (1,129) | int64 | 1032 |
| B 小计（`E_local`=128） | | | ≈ **4620** |
| B 小计（`E_local`=384，上界） | | | ≈ **13836** |
| **kernel C** | | | |
| `ub_one` | (1,1) | int32 | 4 |
| `ub_val` | (1,1) | int32 | 4 |
| `ub_fwd` | (1,C) | int32 | `4·C`（prefill C=820 → 3280） |
| C 小计（prefill） | | | ≈ **3288** |
| **kernel A** | | | 4（`ub_one`） |

> 按 `CONST-capacity-910B2C` 的 auto-multi-buffer 膨胀系数（~1.7x）复算：kernel B ≈ **7.9KB**（`E_local`=128）/
> **23.5KB**（`E_local`=384）≪ **192KB** ✓；kernel C ≈ **5.6KB**（prefill）✓。
> `bh`/`slot_base`/`run` 位于 **GM**（`E_local`=128 时各 `NB×E_local×4 = 40×128×4 = 20480 B = 20KB`，
> **GM scratch 驻留峰值 ≈ 60KB**；上界 `E_local`=384 时各 61.4KB，峰值 ≈ **184KB**，与 `E_local/E_global` 成比例缩小，不占 UB）。

### 4.6 动态轴定义

无动态轴：`numel / E_global / E_local / top_k / NB / C` 均为编译期常量（JIT 特化）。运行时变化的仅为
`flat_ids`/`expert_map` 的**数据值**（不改变 shape）。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir", pass_configs=_PASS_CFG)   # 无 out_idx（host 分配传入）
def _scan_count_ep_kernel(numel, e_global, e_local, nb, cblk): ...
def _scan_prefix_ep_kernel(numel, e_global, e_local, nb, cblk, top_k): ...
def _scan_scatter_ep_kernel(numel, e_global, e_local, nb, cblk, top_k): ...

# 公共入口（同 stream 链式）
def _make_scan_kernel_nopad_ep(numel, e_global, e_local, top_k): ...
```

> `pass_configs`（同非 EP 版、保证 UB 单版本、避免 buffer 重排干扰）：`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`、
> `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False`。无 `compile_flags`（纯整数，无 fast-math/bf16 需求）。
> 按 `(numel, e_global, e_local, top_k)` 经 `functools.lru_cache` 特化。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**整数 Vector 计数/索引/散写（三 kernel，EP 变体）**（无浮点、无矩阵、无 spatial tiling）。判定依据：
全部计算为 `le=expert_map[g]` 重映射、`counts`（本地域直方图）、前缀 scan、块前缀、运行 rank 与散写——落
Vector（AIV）单元。kernel A/C 多核，kernel B 单核。本单元**无 `T.gemm`、无 Cube scope**。

### 5.2 Block 划分

本单元**无空间 block 划分**（无 M/N/K tile）。工作粒度：
- kernel A/C：位置轴按 `NB = min(P, numel)` 块划分，`C = ceildiv(numel, NB)` 每块位置数；**块核 1:1**。
- kernel B：单核，核内 `T.serial(NB)` 遍历块 + `T.Parallel(E_local)` 向量化本地专家轴。
- `E_local` **只作向量化轴长度**，不参与分核。

### 5.3 约束分析

- **对齐约束**：`E_local` 为 EP 派生量（任意值，非 32 倍），`T.cumsum`/`T.Parallel` 对任意 `E_local` 合法
  （探针 P2-EP 实测 `E_local ∈ {1,3,5,96,384}` 位精确）。
- **UB 容量**：kernel B ≈4.5KB（`E_local`=128）/ ≈13.5KB（`E_local`=384，上界），膨胀后 ≪ 192KB ✓
  （`CONST-capacity-910B2C`）。
- **非整除**：`C = ceildiv(numel, NB)`，尾块由 `if base + c < numel` 守卫；`numel < NB` 时仅前 `numel` 块非空。
- **静态边界**：`NB`、`C` 均为编译期常量 ✓（`core-split-strategy.md` §1 要素③要求）。
- **`E_local ≥ 1`**：可编译前提（`E_local = 0` 零长 UB → MLIR `strides must not be zero`，探针实测）。

### 5.4 注意事项（尾块/边界处理）

- **尾块**：`idx ≥ numel` 由普通 `if` 守卫跳过（kernel A/C）。
- **非本地对**（`le < 0`）：kernel A 不计入直方图；kernel C 写 `fwd_idx=-1`、不写 `permuted_idx`。
- **空本地专家**（`counts[l]=0`）：`off[l]==off[l+1]`，段为空；无位置写入该段（§1.6.1 E2-EP corner 覆盖）。
- **全非本地**（`M_local=0`，`expert_map` 全 -1）：`bh` 全 0、`off` 全 0、全 `fwd_idx=-1`、`permuted_idx` 全 0
  （host 零初始化）；探针 `remote_only` 用例与 verify corner 覆盖。
- **`numel=0`**：无块有位置，5 张表全 0（`bh`/`run` 零初始化；`off` 全 0）；corner 覆盖。
- **`numel < NB`**：`NB=numel`，`C=1`，仅在位块启动（其余核空转，边界守卫）。
- **kernel 顺序**：A 写 `bh` → B 读 `bh` 写表/`slot_base` → C 读 `slot_base` 写 `fwd_idx`/`permuted_idx`，
  必须**同一 stream 顺序 launch**（§7.2）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 / §2.1（依据 `docs/开发指南.md` §3.3）。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **返回值 = 20**。
  查询代码与实测值：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20`
  （Ascend910B2C，2026-10-16 本机实查）。本单元为**纯 Vector 单元**（无 Cube）→ 物理核数取实查值翻倍
  **`P = get_aicore_num() * 2 = 40`**（标准 `core-split-strategy.md` §1②「纯 Vector 算子核数翻倍」）。
- **逻辑核数计算**：
  - **kernel A**：逻辑核数 = `NB = min(P, numel)`（每核一块位置）。
  - **kernel B**：逻辑核数 = **1**（一次性 `E_local` 轴 scan + 表输出，单核）。
  - **kernel C**：逻辑核数 = `NB = min(P, numel)`。
  代表 shape：smoke（`numel=8, E_g=384, E_l=128`）→ `NB=8`；prefill（`numel=32768, E_g=384, E_l=128`）→ `NB=40, C=820`。
- **规模判定**（三选一）：
  - kernel A/C：逻辑核数 `NB ≤ P = 40` **恒成立**（`NB=min(P,numel)`）→ 结论「**无需适配**」；
    prefill 下 `NB=P=40` 恰好为物理核整数倍（1×），无负载不均。
  - kernel B：逻辑核数 1 ≤ 40 → 「**无需适配**」（单核一次性 scan）。
  - **无需核内串行 persistent 化**（逻辑核数未超物理核数；核内 `T.serial(C)` 仅用于块内逐位置处理，
    边界为编译期常量）。
- **分核方案**：
  - kernel A/C：`T.Kernel(NB, is_npu=True)`（`NB=min(P,numel)`，编译期常量），**块核 1:1**（`b = core_id`）；
    核内 `for c in T.serial(C)` 静态边界。代表值：prefill `NB=40, C=820`；smoke `NB=8, C=1`。
  - kernel B：`T.Kernel(1, is_npu=True)`，核内 `for b in T.serial(NB)`（静态边界）× `T.Parallel(E_local)`。
  - 负载均衡：kernel A/C 每核位置数差 ≤1（`C` 分块）；各本地专家分布偏斜时，块内运行 rank 工作量 ∝ 块内位置数
    （均衡），总工作量 `O(numel)` 摊到 40 核。
  - **无死区/无跨核写竞态**：`bh[b,·]`/`run[b,·]`/`slot_base[b,·]` 每行仅由拥有该块的核访问；
    `permuted_idx[slot]` 各 slot 全局唯一 → 无写竞态。

### 5.6 承载方式取舍（三 kernel GM 交接 vs 全局域聚合 vs 单 scope）⭐（本单元核心论证）

| 维度 | 单 scope UB（弃选） | 全局域直方图 + 聚合（弃选） | **三 kernel GM 交接 + 内联 remap（主选）** |
|------|---------------------|-----------------------------|--------------------------------------------|
| kernel launch | 1 | 3 | **3 次**（A/B/C；与 gather 同 stream 串行） |
| GM scratch | 无表；但 UB 承载全部 | `bh_g[NB,E_g]` 达 61KB/份（3 份 184KB） | **`bh`+`slot_base`+`run` 各 20KB（`E_l`=128）→ 60KB** |
| 重映射落点 | UB 运行期下标（不可用） | 计数用全局域 + B 内 `E_g` 标量聚合 | **A/C 内联两级 GM 读**（免物化、免聚合） |
| 同步/数据流 | **同 scope 标量消费向量产出**（不可同步） | 无 in-block 同步 | **无 in-block 同步**；kernel 边界隔离；stream 顺序保证 |
| 工具链可实现性 | **不可实现**：CG-2026-0027/0028 | 可行但聚合段无向量 scatter | ✅ 构造上规避全部已知阻断；**探针 P1-EP/P2-EP 实测**（两级间接 + `E_local` 轴） |
| 计算冗余 | 各核重扫 flat_ids | 额外 `E_g` 聚合 | kernel A/C 各扫 flat_ids 一遍（2 遍，无冗余重扫） |
| 结论 | **弃选（可行性）** | **弃选（经济性，F1 证明语义等价）** | **选定** |

**论证要点**：单 scope 形态在本工具链**不可实现**（CG-2026-0027/0028）；全局域聚合形态语义等价（F1 机器验证
2000 例 0 违反）但 scratch 增至 3×且聚合退化为 `E_g` 标量散加（无向量 scatter API）。三 kernel GM 交接 +
扫描内联 remap 是**同时满足「无跨引擎交接、无同 scope 标量消费向量、无 codegen 崩溃、scratch ∝ E_local」**的形态，
其构件（两级标量 GM 间接、守卫普通 `if`、向量载荷原子、`T.cumsum` int32 `(1,E_local)`、核内 `T.serial` +
`T.Parallel(E_local)`、UB 暂存 + `T.copy`）逐一有探针实测或非 EP 版 Stage 3 先例。多出的 2 次 launch 与 60KB
scratch 驻留为实现性支付的可接受成本（§1.6.0 roofline：发射项 ≈ 105µs，相对胞弟 gather 的 ≈0.8ms 占比 ≈ 13%）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| kernel A 核级 | 固定核数网格 | `T.Kernel(NB, is_npu=True)` | 块核 1:1（§5.5） |
| kernel A 块内 | 串行 | `T.serial(C)`（静态边界） | 逐位置两级查表 + 守卫 + 向量载荷原子（§1.6.2 #1/#2/#3） |
| kernel B 核级 | 单核网格 | `T.Kernel(1, is_npu=True)` | 一次性 scan + 表输出 |
| kernel B 跨块 | 串行 | `T.serial(NB)`（静态边界） | 块求和 / 块前缀运行累加 |
| kernel B 本地专家轴 | 向量化 | `T.Parallel(E_local)` + `T.cumsum` | 求和/表暂存/运行累加（§1.6.2 #4/#5/#6/#7；GM 经 `T.copy` 落） |
| kernel C 核级 | 固定核数网格 | `T.Kernel(NB, is_npu=True)` | 块核 1:1 |
| kernel C 块内 | 串行（循环携带） | `T.serial(C)`（静态边界） | 两级查表 + 守卫 + 运行 rank + 散写（§1.6.2 #8/#9；循环携带依赖） |
| 元素级 | 无（已消除/整体替代） | — | 无逐元素浮点标量循环 |

### 6.2 循环伪代码

见 §3.3（kernel A：`T.serial(C)` 两级查表 + 守卫 + 向量载荷原子；kernel B：`T.serial(NB)` 外层 +
`T.Parallel(E_local)` 内层 + 单次 `T.cumsum` + 表输出 UB 暂存 `T.copy`；kernel C：`T.serial(C)` 守卫运行 rank 散写 +
UB 暂存 `fwd` 动态 extent `T.copy`）。

### 6.3 流水线优化

- kernel A/C 的块内串行为顺序/查表/散加，无跨迭代独立任务可流水（且核内已按块并行）。
- kernel B 的跨块串行是运行累加（循环携带依赖），不可流水。
- **不引入 `T.Pipelined`**（无 K 循环/无多级搬运；且 CG-2026-0001 记录 Pipelined 体内条件构造崩溃）。
- 跨 kernel：A → B → C → 胞弟 gather **同 stream 顺序执行**（顺序由 stream 保证，无需显式同步）。

### 6.4 尾块处理

- kernel A/C 尾块：`if base + c < numel` 普通 `if` 守卫。
- 非本地对：kernel A 跳过计数；kernel C 写 `-1` 哨兵。
- 空本地专家：`counts[l]=0` → 该段无位置写入（无特殊分支）。
- `numel=0`：无守卫通过，全表 0。
- 不使用 host padding、谓词零填充或逐元素 select。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**kernel 内自动同步 + kernel 间 stream 顺序**（Developer）。
- kernel A/B/C 各自**单 Vector scope**，scope 内 MTE（`T.copy`）/ Vector（`T.Parallel`/`T.cumsum`）/ 原子
  （向量载荷 `T.atomic_add`）依赖由编译器自动同步；
- **关键约束（CG-2026-0027）**：**不在标量上下文中读取本 scope 内向量 op 写入的 UB 值**——本设计已彻底避免
  （kernel A/C 的标量只读 **GM**：`flat_ids`/`expert_map`/`slot_base`/`run`；kernel B 无标量消费向量产出）；
- **不使用任何手动同步原语**（不使用 `T.sync_block_set/wait`、`T.set_flag/wait_flag`，不依赖 `T.pipe_barrier`）。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| kernel A 内计数之间 | 无依赖（不同 `bh` 行，块核 1:1） | 每核独占行，无跨核 / 同核冲突 |
| kernel B 内 `T.copy`/`T.Parallel`/`T.cumsum` 之间 | 编译器自动同步 | 单 Vector scope 内向量→向量依赖 |
| kernel C 内 `run` 自增与 `slot` 计算之间 | 标量依赖（同线程顺序，GM 读/写） | 循环携带依赖，无跨核冲突（块核 1:1） |
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

> 与非 EP 版 Stage 3 一致：保证 UB 缓冲单版本、避免 buffer 重排干扰 kernel B 的切片拷贝。三 kernel 均为纯整数
> Vector kernel、无 `T.gemm`。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源 EP 算子语义**为唯一依据实现（纯 Python/torch **顺序** EP 计数排序：先 `expert_map`
> 重映射，本地域计数/前缀，非本地写 `-1`），**不得复刻** §0.6 的三 kernel 算法（保证验证独立性）。
> 源仓测试的 `_ref_moe_permute_nopad`（`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` L31–L70）**只覆盖
> 非 EP 分支**（无 `expert_map` 入参），故本变体 golden 按 §0.1 独立实现并冻结（`verify_equiv.py` 的
> `baseline_ep` 即其参照）。

```python
import torch

def golden_scan_ep(flat_ids, expert_map, e_local, top_k):
    """§0.1 语义：flat_ids[numel] + expert_map[E_g] ->
    (true_offsets, true_sizes, efo, fwd_idx, permuted_idx)。

    顺序（稳定）EP 计数排序 —— 本地域计数、非本地写 -1 哨兵，独立于 NPU 算法。
    """
    ids = flat_ids.detach().to(torch.int64).cpu().tolist()
    emap = expert_map.detach().to(torch.int64).cpu().tolist()
    K = int(top_k)
    numel = len(ids)

    counts = [0] * e_local
    for g in ids:
        le = emap[g]
        if le >= 0:
            counts[le] += 1

    off = [0] * (e_local + 1)                 # int64 exclusive prefix
    for l in range(e_local):
        off[l + 1] = off[l] + counts[l]

    write_ptr = list(off[:-1])
    permuted_idx = [0] * numel                # host 零初始化语义
    fwd_idx = [0] * numel
    for idx, g in enumerate(ids):
        le = emap[g]
        if le < 0:
            fwd_idx[idx] = -1                 # 非本地哨兵
        else:
            slot = write_ptr[le]
            permuted_idx[slot] = idx // K
            fwd_idx[idx] = slot
            write_ptr[le] += 1

    return (
        torch.tensor(off[:-1], dtype=torch.int32),   # true_offsets
        torch.tensor(counts, dtype=torch.int32),     # true_sizes
        torch.tensor(off, dtype=torch.int64),        # expert_first_token_offset
        torch.tensor(fwd_idx, dtype=torch.int32),    # fwd_idx
        torch.tensor(permuted_idx, dtype=torch.int32),  # permuted_idx
    )
```

> **顺序无关等价性**：op 契约（`examples/TileOPs/tests/ops/test_moe_permute_nopad.py` L105–L111）只校验
> `perm_h[fwd_idx]==hidden_states[idx//K]`（本地位置）与 `offsets/sizes` 精确；本设计产出**稳定序**，与上 golden
> 逐位一致（§1.6.1 E2-EP），故可直接逐元素精确比对。**非本地对**额外校验「`fwd_idx == -1` 的集合 ==
> `expert_map[flat_ids] < 0` 的集合」（哨兵契约）。

### 8.2 精度标准

| dtype | 标准 |
|-------|------|
| int32 / int64 | **逐元素精确一致**（无浮点参与，无 atol/rtol；§1.6.1 采纳项 E1/E2-EP/E3/E4/E5 机器验证全 0 违反） + **哨兵集合精确一致** |

> 注：胞弟 gather 的 `perm_h` 输出精度（bf16/fp16 精确搬运）属其单元，不在本单元。

### 8.3 L0 门槛测试计划

> **主线（交付形态）= 三 kernel EP 扫描单元 + 与 `golden_scan_ep` 逐元素精确比对**（5 张表 int32/int64 逐位相等 +
> 哨兵集合相等）。

| 用例 | 参数 | 验证点 |
|------|------|-------|
| L0-EP-1 smoke | `numel=8, E_g=8, E_l=4, K=2`（全本地） | 5 表逐元素相等；`NB=8,C=1` |
| L0-EP-2 EP smoke | `numel=8, E_g=384, E_l=128, K=8`（混合） | 重映射 + 过滤正确；`bh`/`run` 零初始化 |
| L0-EP-3 过滤生效 | `numel=64, E_g=384, E_l=128, K=2` | 非本地对 `fwd_idx=-1`；`M_local < numel`；`permuted_idx[M_local:]` 为 0 |
| L0-EP-4 全非本地 | `numel=100, E_g=64, E_l=64, K=5`，`flat_ids` 全落非本地 | 全 `fwd_idx=-1`；全 `counts=0`；`off` 全 0；`permuted_idx` 全 0 |
| L0-EP-5 离散 map | `E_g=512, E_l=64`，`expert_map` 随机置换 | 两级间接正确；稳定序 |
| L0-EP-6 尾 lane `E_l` | `E_l ∈ {1, 3, 5}`（非 32 倍） | `T.cumsum`/`T.Parallel` 尾 lane 正确（探针 P2-EP 已预验证） |
| L0-EP-7 单本地专家 | `E_l=1`，`flat_ids` 混本地/非本地 | 段唯一 + 哨兵 |
| L0-EP-8 代表大 shape | `numel=32768, E_g=384, E_l=128, K=8`（EP prefill） | `NB=40, C=820`；无死锁；5 表 + 哨兵逐位一致 |
| L0-EP-9 多次运行稳定性 | 同 L0-EP-2 / L0-EP-8 重复运行 6 次 | 5 表 + 哨兵**稳定**（无 CG-2026-0027 型未同步读） |
| L0-EP-10 对接契约 | 抽取 L0-EP-3 / L0-EP-8 输出 | int32/int64 shape 与 §4.2 一致；本地槽 `permuted_idx[fwd_idx[idx]]==idx//K` |
| L0-EP-11 三 kernel 顺序 | A→B→C 同 stream | C 读到 B 写的 `slot_base`、B 读到 A 写的 `bh`（无旧值/未初始化） |
| L0-EP-12 `numel` 尾部 | `numel=39`（< NB 上界）、`numel=100`（非整除 C=3） | 尾块守卫、非整除正确 |

> 注：§1.6.1 的 `verify_equiv.py` 已在**设计期**对 L0-EP-3/4/5/6/7/8/12 的**算法等价性**做 CPU 位精确预验证；
> §0.6 的设计期探针（`history_version/design_probe_ep_primitives.py`、`history_version/design_probe_ep_prefix.py`）
> 已在**设备上**对 L0-EP-2/3/4/5/6/7/8/12 的核心原语做可编译 + 位精确预验证。

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **NPU 无 fetch-add**（`T.atomic_add` 无返回值，`docs/Tilelang.language/原子操作/T.atomic_add.md` §3；CG-2026-0030）
  → 源散写形态直译不可行，改确定性块并行 rank（§0.6 R2）。
- **`T.atomic_add` 标量载荷前端不可用**（CG-2026-0029，`DiagnosticError`）→ 一律用**向量载荷**。
- **多核 Vector kernel 的直接标量 GM store 随机丢写**（非 EP 版 Stage 3 实测）→ GM 写一律走向量载荷原子或
  UB 暂存 + `T.copy`。
- **UB 运行期下标**（CG-2026-0028(d)）→ 运行期下标仅用于 GM（探针 P1-EP 实测）。
- **`T.Parallel` 循环体内禁直写 GM 张量**（`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3）→ 表输出
  经 UB 暂存 + `T.copy`。
- **同 scope「向量产出 → 标量消费」不可同步**（CG-2026-0027）→ 三 kernel GM 交接规避。
- **int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` / 多行 int32 `T.cumsum` / 一 kernel 两次 `T.cumsum`**
  （CG-2026-0028）→ 全部回避（`//`、无 `alloc_local`、**EP 守卫用普通 `if`**、`(1,E_local)` 单行单次 scan）。
- **跨引擎同 block 交接不可编译**（CG-2026-0026）→ 本设计无跨引擎交接。
- **`E_local = 0` 不可编译**（零长 UB → MLIR `strides must not be zero`，探针实测）→ host 保证 `E_local ≥ 1`。
- `T.cumsum` int32 路径以 scalar instruction lower（编译器 warning「low efficiency」）→ `E_local ≤ 384` 单核一次，
  成本可忽略；`T.Parallel` int32→int64 cast 使用固定 `rint` 舍入模式（编译器 warning）→ 整数 cast 无影响。
- **kernel B 发射项**（`T.serial(NB) × T.Parallel(E_local)`）× 两趟 ≈ `NB×5` 向量 op ≈ 103µs（§1.6.0 roofline）——
  相对胞弟 gather（≈0.8ms）占比 ≈13%；`NB` 为可调旋钮（Stage 4）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **漏 EP 过滤守卫**（只 remap 不判 `>=0`） | kernel A/C 写 `if le>=0` 漏掉 | 非本地对被计入本地域 → 表/槽全错（verify E2-EP-guard：494/500 例失配） | 计数与散写两处都必须 `if le>=0` 守卫 |
| **漏 `-1` 哨兵** | kernel C 非本地分支未写 `fwd_idx=-1` | 下游 gather 读到错误行 | `if le<0: ub_fwd[0,c]=-1`（哨兵与本地对同 UB 暂存行，一并 `T.copy`） |
| **标量消费向量产出**（CG-2026-0027） | 标量读 `T.cumsum`/`T.Parallel` 写入 UB 的值并消费 | 同编译产物多运行不一致 | 三 kernel GM 交接：标量只读跨 kernel GM |
| `T.atomic_add` 标量载荷（CG-2026-0029） | `T.atomic_add(dst, 1)`（单元素） | 前端 `DiagnosticError: 'int' object has no attribute 'buffer'` | 用**向量载荷** `T.atomic_add(dst, ub_one[0,0:1], [1])` |
| 多核裸标量 GM store | `bh[core_id,e]=bh[core_id,e]+1` / `fwd_idx[idx]=slot` | 随机丢写（非 EP Stage 3 实测） | 向量载荷原子 / UB 暂存 + `T.copy` |
| `T.Parallel` 体内直写 GM 张量（`T.Parallel.md` §2.3） | `for e in T.Parallel(E): gm[e]=ub[0,e]` | `NPU IR opt failed … crash backtrace` | 先写 UB 暂存，再 `T.copy` 落 GM（含 int64 cast 经 `ub_efo`） |
| int32 `>>` / `alloc_local`+`serial` / 标量 `T.if_then_else`（CG-2026-0028） | 误用 | device_codegen SIGSEGV | `//` 替代 `>>`；免 `alloc_local`；**普通 `if`** |
| 运行期下标读 UB（CG-2026-0028d） | 把 `bh`/`run` 放 UB | 污染相邻结果 | scratch 全放 **GM** |
| 同 kernel 两次/多行 int32 `T.cumsum`（CG-2026-0028e/f） | kernel B 想二次 scan 或 2D 多行 | SIGSEGV / 垃圾值 | 唯一一次 `(1,E_local)` 单行 scan |
| kernel 边界顺序反了 | C 先于 B、或异性 stream | C 读 `slot_base` 旧值/未初始化 | 严格同 stream 顺序 A→B→C |
| `bh`/`run`/`permuted_idx` 未零初始化 | host 用 `torch.empty` | 计数/rank 错、未写 slot 非 0 | host `torch.zeros` 预清零（源 op 同款） |
| 块前缀累积漏 `off[l]` | `slot_base=block_base`（忘加 `off[l]`） | slot 全局错位 | `slot_base[b,l]=off[l]+Σ[b'<b]bh[b',l]`（E4） |
| 尾块越界 | `numel` 非整除 `C` | 越界读写 | `if base+c < numel` 守卫（含 kernel C 的 `ct=T.min(C,numel-base)`） |
| `idx//top_k` 写成 `>>` | — | SIGSEGV | 用整数除 `//`（E5） |
| 传给 kernel B 的 `E_local=0` | 极端 EP（本 rank 无专家） | MLIR `strides must not be zero` | host 保证 `E_local ≥ 1`（否则走退化分支） |

### 9.3 特殊场景处理

- **偏斜分布**：块内运行 rank 工作量 ∝ 块内位置数（均衡），总 `O(numel)`。
- **高度过滤**（大量非本地对）：kernel A/C 的工作量仍为 `O(numel)`（守卫只跳过计数/散写，不改扫描长度）；
  `M_local` 小则 `permuted_idx` 有效前缀短，正常。
- **空本地专家 / 全非本地 / 全空输入**：段为空 / 全哨兵 / 无位置写入（§8.3 L0-EP-3/4）。
- **极小 shape**：`NB=min(P,numel)`（smoke `NB=8`），正确性优先。
- **`E_local` 非 32 倍**：`T.cumsum`/`T.Parallel` 尾 lane 由原语处理（探针 P2-EP 实测 1/3/5）。
- **`numel=0`**：存在 `NB=1,C=1` 的退化块，守卫失效 → 5 表全 0。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（EP 边界，正向登记）**：`E_local = 0`（本 rank 无本地专家）时 kernel B 的 `T.alloc_ub((1,0))` 触发
  MLIR 解析错误 `strides must not be zero`（设计期探针实测，tilelang 0.1.2+c641af3e / Ascend910B2C / 2026-10-16）。
  **层：定义域约束 / 前端**。处置：host 侧断言 `E_local ≥ 1`；`E_local = 0` 视为退化输入（源 op 亦无有意义输出）。
  建议按 `AGENTS.md` Gap Rule 登记为一族「零长 UB 分配不可编译」边界（occurrences=1，待 Stage 2/3 复核是否升为缺口）。
- **R-2（EP 两级 GM 间接 + 守卫，已探针确认）**：`g=flat_ids[idx]; le=expert_map[g]`（两级运行期下标 GM 标量读）+
  `if le>=0`（普通 `if`）+ 向量载荷原子，在 kernel A/C 中由设计期探针 P1-EP 在 8 例代表 shape（含 prefill
  `numel=32768/E_g=384/E_l=128/NB=40/C=820`、离散 `expert_map`、全非本地、`numel<NB`）实测**编译通过 + 5 表 +
  哨兵位精确**。仍建议 Stage 3 首编反复运行验证稳定性；命中问题 → 切 §1.6.0 备选（kernel C 单核 + 全局 `run[E_local]`）。
- **R-3（`E_local` 专家轴，已探针确认）**：kernel B 的 `T.Parallel(E_local)` / `T.cumsum((1,E_local))` / UB 暂存 +
  `T.copy`（含 int64 cast）由探针 P2-EP 在 `E_local ∈ {1,3,5,96,384}`（含非 32 倍尾 lane）实测位精确；
  若 `T.copy` 的 UB→GM int32→int64 dtype 转换在某 shape 受限，回退为分离转换步骤或 int64 累加路线。不影响语义。
- **R-4（性能，交 Stage 4）**：`NB` 旋钮（B 发射项 vs C 并行度）；kernel A/C 的标量循环含 **2 次 GM 读/位置**
  （`flat_ids` + `expert_map`）——是否值得把 `expert_map` 预载入 UB（需评估 UB 运行期下标限制 CG-2026-0028(d)，
  仅可按块静态载入）交 Stage 4 按实测标定；不影响主选/弃选裁决。
- **R-5（语义，交 Stage 2 复核）**：本设计采**稳定序**（与 §8.1 EP golden 一致）；源 EP 的专家内顺序为非确定
  （fetch-add）。若下游 unpermute 依赖**非稳定** atomic 序的隐式契约，则需复核（依据：op 测试仅校验顺序无关一致性，
  manifest 仅约束 shape/dtype；源 op 亦只保证 `perm_h[fwd_idx]` 一致性）。
- **R-6（集成接线，交 Stage 5）**：`MoePermuteNopadKernel` 当前 EP 路径绑定 GPU 工厂（`compile_flags` 形态在本
  toolchain 报 `TypeError: jit() got an unexpected keyword argument 'compile_flags'`），需由集成阶段把 EP 分支重接到
  本单元的 `_make_scan_kernel_nopad_ep`（NPU 形态：`expert_map` 传入 + `E_local` 由 host 计数得出 + 三 kernel 链式）。
  属集成职责，不影响本单元算法/接口设计。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1，E1/E2-EP/E3/F1 全 EQUIV_PASS）
├── history_version/
│   ├── design_probe_ep_primitives.py   # D-3 探针：EP 两级间接 + 守卫（kernel A/C）
│   └── design_probe_ep_prefix.py       # D-3 探针：E_local 专家轴（kernel B）
└── (Stage 3 将产出) _make_scan_kernel_nopad_ep.py
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核/承载方式取舍 + §8 验证） |
| `verify_equiv.py` | 已完成（已运行） | 等价性验证（E1/E1-guard/E2-EP/E2-EP-guard/E2-EP-corner/E3/F1） |
| `history_version/design_probe_ep_primitives.py` | 已完成（已运行） | D-3 探针 1：两级间接 + 守卫 + 向量载荷原子（8 例全过） |
| `history_version/design_probe_ep_prefix.py` | 已完成（已运行） | D-3 探针 2：`E_local` 专家轴（6 例全过） |
| 算子实现（Stage 3） | 待实现 | 三 kernel EP 扫描单元（§3.3 伪代码）：`_scan_count_ep_kernel` / `_scan_prefix_ep_kernel` / `_scan_scatter_ep_kernel` |
| 集成对接（Stage 5） | 待实现 | 与胞弟 `_make_gather_kernel_nopad`（读 `permuted_idx`）同 stream 顺序 launch；EP 路径重接线（§9.4 R-6） |

### 10.3 命名规范

- 项目目录：`examples/moe_permute_nopad/`
- 算子目录：`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/`
- 规范文件名：`_make_scan_kernel_nopad_ep.py`（Stage 3 产文件；含三 kernel 工厂 + 公共入口 `_make_scan_kernel_nopad_ep`）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ D-3 探针
2. ⬜ Golden 函数（§8.1 `golden_scan_ep`）+ L0-EP 五表 + 哨兵逐元素比对（§8.3 主线）
3. ⬜ kernel A/B/C（§3.3 伪代码），先 smoke（`numel=8, E_g=8, E_l=4`）再代表大 shape（`numel=32768, E_l=128`）
4. ⬜ 与胞弟 gather 核同 stream 顺序集成（对接契约 §4.2）+ EP 路径重接线（§9.4 R-6）

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py`（`_make_scan_kernel_nopad_ep`，L122–L215）
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_permute_nopad/_moe_permute_nopad_fwd_kernels.py`
- 非 EP 版强参照：`examples/moe_permute_nopad/_make_scan_kernel_nopad/DESIGN.md`、
  `examples/moe_permute_nopad/_make_scan_kernel_nopad/_make_scan_kernel_nopad.py`、
  `examples/moe_permute_nopad/_make_scan_kernel_nopad/verify_equiv.py`
- 源 op / 测试 / bench / manifest / workload：`examples/TileOPs/tileops/ops/moe/moe_permute_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_permute_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_permute_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`、`examples/TileOPs/tileops/workloads/moe.py`
- 胞兄（同族整数索引算子，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/_tile_scheduler_kernel.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/verify_equiv.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`
- 本变体设计期探针：`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_primitives.py`、
  `examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/history_version/design_probe_ep_prefix.py`
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
  `.agents/skills/tilelang-op-optimize/references/pattern-library/INDEX.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0010/0026/0027/0028/0029/0030）
