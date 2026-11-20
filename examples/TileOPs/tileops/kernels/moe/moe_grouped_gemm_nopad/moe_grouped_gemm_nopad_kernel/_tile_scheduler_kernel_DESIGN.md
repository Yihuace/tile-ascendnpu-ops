# _tile_scheduler_kernel 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`tileops/kernels/moe/moe_grouped_gemm_nopad/moe_grouped_gemm_nopad.py`
> 中的 `_tile_scheduler_kernel`（GPU/CUDA target，独立 tile 调度 kernel）。本单元目标：`target="npuir"`（Ascend NPU）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py`（`_tile_scheduler_kernel`）。
> 强参照（胞弟，已冻结并含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> **本版为 revision v2**（Stage 3 报 `[DESIGN_ERROR]` 后重做）：v1 的「单 Vector scope：UB 排他前缀 scan + 标量逐 tile
> 二分/单调指针消费」被实测证伪——**标量消费向量产出**在当前工具链不可实现（CG-2026-0027；连带 CG-2026-0028
> 一串 codegen SIGSEGV）。v2 改为**两 kernel GM 交接**，不混用「向量生产 → 标量消费」。差异与教训见 §12。
> 等价性机器验证（D-1）：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/verify_equiv.py`（E1/E2/E3 全 EQUIV_PASS）。
> 可行性机器验证（D-3 设计期探针，设备实测）：
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/history_version/design_probe_two_kernel.py`、
> `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/history_version/design_probe_two_kernel_large.py`
> （代表 shape E=256 prefill，6 次重复位精确稳定）。

**本单元的一句话定位**：`_tile_scheduler_kernel` 是 MoE no-pad 分组 GEMM 的**索引前置**——把 `true_sizes[E]`
（每专家 token 数）翻译成「第 t 个 M-tile 属于哪个专家、在专家内的行偏移」，供 GEMM 核做 O(1) 寻址，
消除每 CTA 的 O(E) 专家扫描。它**不含任何浮点数值计算**（全程 int32 索引），是一个纯调度单元。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **两 kernel GM 交接**（v2 核心）：**kernel A**（`_sched_scan_kernel`，Vector，grid=1）把 `true_sizes[E]`
   经一次 `T.cumsum` 变成**排他前缀 `cum[E+1]` 落 GM**；**kernel B**（`_sched_expand_kernel`，Vector，固定核数）
   读 GM `cum` 做**逐专家区间展开**，产出 3 张 GM int32 表。两 kernel 同 stream 顺序 launch，**无 in-block 同步、
   无标量消费向量产出**（规避 CG-2026-0027 / CG-2026-0028）。表由胞弟 Expert Cube GEMM 核经 GM 读表消费。
2. **前缀扫描改用硬件 scan 原语 `T.cumsum`**（源：thread 0 串行标量前缀扫描）：kernel A 单条 scan 得含前缀，
   再 1-lane 移位得排他前缀；E=256 时把 ~256 次标量加压缩为 1 条 `VCumsumOp`（int32 路径实机实测可用）。
3. **tile→(expert,row) 解析从「每 tile 独立二分」改为「逐专家区间展开」**：因为「标量消费向量 scan 结果」
   在本工具链不可实现（CG-2026-0027），v1 的标量二分/单调指针被**否决**；v2 改为 kernel B 按专家 e 读 GM
   `cum_e, cum_{e+1}`（标量 **GM** 读，而非 UB 向量产出值），把区间 `[cum_e, cum_{e+1})` 一次性向量填充为
   `(e, k·block_m)`——数学上与 `searchsorted(cum,t,right=True)-1` **逐位相同**（§1.6.1 E2 机器验证），
   且每个有效 tile 只写一次，代价 O(total_tiles) ≤ O(max_tiles)。
4. **分核：kernel A 单核（grid=1，一次性小向量 scan）；kernel B 固定物理核数（纯 Vector → 40）+ 核内
   `T.serial` 遍历专家**，`total_tiles` 全程留在设备侧 —— 承接源「零 CPU 同步」意图；死 tile 区间
   `[total_tiles, max_tiles)` 由单一核写哨兵 `(-1,0)`（与有效区间不相交，无跨核写竞态）。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = **`_tile_scheduler_kernel`**（独立调度 kernel）。同文件的 `_moe_grouped_gemm_kernel`（GEMM 主核）
> 由胞弟单元 `_moe_grouped_gemm_kernel` 设计，**不在本单元**；但两单元共享同一「专家聚簇任务序 + 前缀扫描 + 查找」
> 调度语义——本单元的产出（3 张 GM 表）正是胞弟 GEMM 核的**消费接口**（§0.6 R1 对接契约）。

### 0.1 源算子语义（做什么）

**数学语义**（纯整数索引映射；无浮点）：

输入 `true_sizes[E]`（int32，每专家真实 token 数 $s_e\ge 0$）与编译期常量 `block_m`，构造 M-tile 调度：

$$
cnt_e=\Big\lceil \tfrac{s_e}{block\_m}\Big\rceil,\qquad
cum_0=0,\quad cum_{e+1}=cum_e+cnt_e,\qquad
total\_tiles=cum_E
$$

对每个 M-tile 下标 $t\in[0,\,max\_tiles)$：

$$
e(t)=
\begin{cases}
\max\{\,e : cum_e\le t\,\}, & t < total\_tiles\\
-1, & t \ge total\_tiles
\end{cases}
\qquad
row(t)=
\begin{cases}
(t-cum_{e(t)})\cdot block\_m, & t<total\_tiles\\
0, & t\ge total\_tiles
\end{cases}
$$

输出 `tile_expert_ids[t]=e(t)`、`tile_row_offsets[t]=row(t)`、`total_tiles[0]=cum_E`。语义上等价于
`e(t) = torch.searchsorted(cum, t, right=True) - 1`（在有效区，即「$t$ 落在第 $e$ 个专家的 tile 段
$[cum_e,cum_{e+1})$ 内」）。

- **规约语义**：无浮点规约；前缀和 `cum` 为**整数精确**运算，无累加顺序/舍入问题（这是本单元与
  Cumprod/Softmax 类 scan 的本质区别——**整数 scan 的等价性可要求逐位精确**）。
- **dtype 语义**：全程 int32（`true_sizes`、三张输出表、内部 `cum` 与全部索引）。无 dtype 提升/回落。
- **边界语义**：
  - $s_e=0$ → $cnt_e=0$，该专家不产生 tile；`cum` 出现**重复值**（相邻专家同 `cum`），映射仍唯一确定
    （空专家段为空区间，不产生歧义，见 §1.6.1 E2/E3 机器验证）。
  - `max_tiles = numel // block_m + E` 是**保守上界**（源 host 侧定义，见 §0.3）；$t\in[total\_tiles,max\_tiles)$
    为**死 tile**，写哨兵 `(-1, 0)`。
  - $e(t)$ 的定位：源实现由二分查找的静态迭代数 `log2_up = max(1, ceil(log2(E+1)))` 保证收敛；NPU 侧由
    逐专家区间展开直接给出（§0.6 R3）。
- **host 侧语义**：`_tile_scheduler_kernel` 本身**无 host 预处理**（纯 kernel）；`max_tiles`、`num_experts`、
  `block_m` 均为编译期 Python 常量，经 `functools.lru_cache` 特化。唯一的 host 协作在**调用方 GEMM**（`forward`
  里 K 对齐时的 `F.pad(A)`），不属于本单元语义（见 §0.3 host 逻辑）。

**语义保持基线**：§8.1 golden 以本节整数映射为唯一依据（纯 Python / torch 整数实现），**不得复刻** §0.6 的
NPU 算法（两 kernel 向量 scan + 逐专家区间展开）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `true_sizes` | `(num_experts,)` | int32 | 每专家真实 token 数 $s_e$ |
| `tile_expert_ids` | `(max_tiles,)` | int32 | **输出**：每 M-tile 的专家号（死 tile = -1） |
| `tile_row_offsets` | `(max_tiles,)` | int32 | **输出**：M-tile 在专家内的行偏移（$=\text{tile\_idx}\cdot block\_m$） |
| `total_tiles` | `(1,)` | int32 | **输出**：有效 M-tile 总数（设备侧，不下回 CPU） |

`@tilelang.jit(out_idx=[1, 2, 3])`：三个输出由 TileLang 在设备侧分配并返回（**零 CPU 同步**的实现手段之一）。
源 `_SCHED_THREADS = 256`（每 block 256 线程）。

### 0.3 实现算法解读（怎么算）

源 kernel 结构（源 L74–L120）：`@tilelang.jit(out_idx=[1,2,3])` 工厂，`T.Kernel(T.ceildiv(max_tiles, threads), threads=threads)`
一维 grid；每 block 内**两个子相位**：

- **子相位 (a)**：`thread 0` 在 SMEM 里串行前缀扫描：`s_cum[0]=0; for e in serial(E): s_cum[e+1]=s_cum[e]+(true_sizes[e]+block_m-1)//block_m`；
  `bx==0` 的 block 把 `s_cum[E]` 写入 `total_tiles[0]`；随后 `T.sync_threads()`。
- **子相位 (b)**：每 thread 认领一个 M-tile `tile_id = bx*threads + tx`；若 `tile_id < s_cum[E]`，
  对 `s_cum` 做 `log2_up` 步并行二分查找得 `lo`，写 `tile_expert_ids[tile_id]=lo`、
  `tile_row_offsets[tile_id]=(tile_id - s_cum[lo])*block_m`；否则写哨兵 `(-1, 0)`。

**计算步骤分解**（源码每条计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| P1-a1 | `s_cum[0]=0`（thread 0） | — | `s_cum[0]`（SMEM） | $cum_0=0$ |
| P1-a2 | `s_cum[e+1]=s_cum[e]+(s_e+block_m-1)//block_m`（thread 0 串行） | `true_sizes` | `s_cum[E+1]` | $cnt_e$ 与 $cum_{e+1}$ |
| P1-a3 | `total_tiles[0]=s_cum[E]`（bx==0） | `s_cum` | `total_tiles` | $total\_tiles=cum_E$ |
| P1-a4 | `T.sync_threads()` | — | — | 子相位 (a)→(b) 交接 |
| P1-b1 | `tile_id = bx*threads + tx` | block/thread id | `tile_id` | M-tile 下标 $t$ |
| P1-b2 | `if tile_id < s_cum[E]` | `tile_id`, `s_cum[E]` | 分支 | 死 tile 判定 |
| P1-b3 | 二分：`mid=(lo+hi)>>1; if s_cum[mid+1]<=tile_id: lo=mid+1 else hi=mid`（`log2_up` 步） | `tile_id`, `s_cum` | `lo` | $e(t)=\max\{e:cum_e\le t\}$ |
| P1-b4 | `tile_expert_ids[tile_id]=lo` | `lo` | 输出表 | $e(t)$ |
| P1-b5 | `tile_row_offsets[tile_id]=(tile_id-s_cum[lo])*block_m` | `tile_id`, `s_cum[lo]` | 输出表 | $row(t)$ |
| P1-b6 | 死 tile：`tile_expert_ids[t]=-1; tile_row_offsets[t]=0` | `tile_id` | 输出表 | 哨兵语义 |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[true_sizes(E,)] --标量循环读--> SMEM[s_cum(E+1) int32]  (thread 0 串行前缀扫描)
                                        │  T.sync_threads()
                                        ▼
                       每 thread 一个 tile_id：SMEM 二分查找 ---> lo
                                        │
                                        ├─> GM[tile_expert_ids(max_tiles)]  (lo / -1)
                                        └─> GM[tile_row_offsets(max_tiles)] ((tile_id-s_cum[lo])*block_m / 0)
       bx==0: SMEM[s_cum(E)] ------------> GM[total_tiles(1)]
```

**循环与并行结构**：grid = `ceil(max_tiles/threads)` 个 block（threads=256，故约 `max_tiles/256` 个 block）；
每 block：`T.serial(E)` 串行扫描（仅 thread 0）+ `T.serial(log2_up)` 并行二分（每 thread）。
**未使用 `T.Parallel` / `T.Pipelined`**——纯 scalar/thread 级 Python-int 循环。
**源码另有一处未单列的冗余特征**：每个 block 各自重复一遍 thread-0 扫描（prefill 下 3 个 block 重扫 3 遍）
——NPU 侧由 kernel A **单核一次**向量 scan 承接（无冗余，见 §0.6 R2）。

**host 侧逻辑**：
1. 本 kernel 无 host 预处理；`max_tiles`、`num_experts`、`block_m` 为编译期常量（`lru_cache` 特化）。
2. 调用方 `forward` 里有两处 host 协作（**不属于本单元**，但解释了三张表的消费方式）：
   (i) K 对齐时 `A = F.pad(A, (0,0,0,block_m))`（改动输入真实内容——NPU 禁止，见 §0.5）；
   (ii) 顺序调用 `sched_fn(true_sizes)` → `gemm_fn(A,B,tile_expert_ids,tile_row_offsets,true_offsets,true_sizes,total_tiles_t)`。

**编译配置**：`@tilelang.jit(out_idx=[1,2,3])`（无 `pass_configs`、无 `compile_flags`——本 kernel 无 fast-math/bf16 需求）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM 驻留 `s_cum`（`T.alloc_shared`） | 免每 thread 重算前缀和 | 一次扫描写 SMEM，块内共享 | shared memory | 硬件强相关（NPU 有 UB 等价） |
| O2 | 排他前缀扫描（线性 O(E) 一遍） | 免每 tile O(E) 重数 | 一次扫描得 `cum` | 无（纯算法） | 算法层可移植 |
| O3 | 二分查找 O(log E) | 免每 tile O(E) 线性扫描 | 单调 `cum` 上二分 | 无（纯算法） | 算法层可移植 |
| O4 | thread 级并行（一 thread 一 tile） | 摊平 max_tiles 个 tile 的搜索 | 每 thread 独立二分 | warp/thread 模型 | 硬件强相关（NPU 是 SIMD lane，非 thread） |
| O5 | 保守上界 `max_tiles` + 死 tile 哨兵 | 输出定长、零 CPU 同步 | grid 按上界静态分配，死 tile 写 -1 | 廉价 block 启动 | 模式可移植（一维核超发被串行调度） |
| O6 | `out_idx=[1,2,3]` 设备侧输出分配 | `total_tiles` 全程在设备侧 | TileLang 设备侧 tensor 分配 | 无 | 模式可移植（由 op/harness 分配承接） |
| O7 | 独立调度 kernel 写 3 张 GM 表、GEMM 核读表 | 解耦调度与 GEMM，O(1) 寻址 | 两 kernel 顺序执行 + GM 表交接 | 无（GPU 多 kernel 顺序执行） | **NPU 上可移植且当前工具链唯一可用底座**（§0.5/§0.6 R1） |

> 识别不出来 ≠ 不存在。O1–O7 为源码显式优化，逐项在 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（M-tile→(expert,row) 整数映射、死 tile 哨兵、int32） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 输出定长 + `total_tiles` 设备侧（零 CPU 同步） | 语义/优化 | 无 | 有 | **保留** | `total_tiles=cum_E`（`<= max_tiles`）保证 `tile_expert_ids` 长度合法；全程设备侧 |
| O1 SMEM 驻留 `s_cum` | 优化 | shared memory | 有（UB） | **等价替换** | kernel A 改 `T.alloc_ub`（int32 索引表，UB 192KB 内；§4.5） |
| O2 排他前缀扫描 | 算法 | 无 | 有 | **保留** | 算法本身保留；实现改为硬件 scan（§0.6 R2，kernel A） |
| O3 二分查找（每 tile O(log E) 定位专家） | 算法/优化 | 无 | 有（算法层） | **等价替换** | v1 的标量二分/单调指针在本工具链**不可实现**（CG-2026-0027）；改为**逐专家区间展开**（数学等价、位精确，§0.6 R3、§1.6.1 E2）。承接源「免每 tile O(E) 扫描」意图 |
| O4 thread 级并行（一 thread 一 tile） | 优化 | CUDA thread/warp | 无（NPU 为 SIMD lane / 无 thread-per-element） | **重新设计** | kernel B 固定物理核 + 核内 `T.serial` 遍历专家；填充用 `T.Parallel` 向量化（§0.6 R4） |
| O5 超发 grid（`ceil(max_tiles/threads)` block）+ 死 tile 哨兵 | 优化 | 廉价 block 启动 | 无（一维核超发被串行调度，core-split-strategy.md §1） | **重新设计** | kernel B 固定物理核数 + 核内 `T.serial`（静态边界）；死 tile 区间 `[total,max_tiles)` 由单核写哨兵 `(-1,0)`（§0.6 R4） |
| O6 `out_idx` 设备侧输出分配 | 优化 | TileLang jit 特性 | 有 | **保留** | kernel B 有 3 张 GM 输出张量：`out_idx=[1,2,3]` 由 TileLang 设备侧分配（或 op/harness 分配），与源一致（§4.2） |
| O7 独立调度 kernel + 跨 kernel GM 表交接（调度 kernel 写、GEMM 核读 3 张表） | 优化/结构 | 无（GPU 多 kernel 顺序执行） | 有 | **保留（主选承载形态）** | 本单元为**独立调度单元**（v2 细分为 kernel A 产 `cum`、kernel B 产 3 表），产表落 GM；胞弟 Expert Cube GEMM 核读表。kernel 间与与 GEMM 均同 stream 顺序 launch，**无 in-block 同步**。**依据**：CG-2026-0026（任何跨引擎同 block 交接在当前工具链不可编译，官方 `examples/mixcv/mixcv_mixkernel.py` 复现）+ CG-2026-0010（Developer 单 Scope persistent+gemm+整型索引段 lowering segfault）；胞弟 Stage 3 实测底座 max_diff 9.537e-07（`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/repro.log`） |
| **R6（v2 新增）单 Vector scope 内「向量产出 → 标量消费」**（scan 结果驻 UB 后由标量 per-tile 二分/守卫/下标读取） | 优化/结构（v1 §3.3/§7.1 主选） | 无 | **无可用同步原语** | **舍弃** | **CG-2026-0027**：同一编译产物多次运行结果不一致（stale 读），`T.pipe_barrier` 无效；连带 **CG-2026-0028** 的 `alloc_local+serial` / `>>` / 标量 `if_then_else` / 运行期下标 UB 读 / 双 scan 等一串 codegen SIGSEGV。证据：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py` + `.agents/evolution/capability-gaps.md`。**这是 v2 改两 kernel 的直接原因** |
| 「内联折叠」（v0 引入，非源）：调度并入 GEMM 核 Vector 索引段、表不落 GM | 优化/结构 | 无 | **无可用的跨引擎同 block 交接原语** | **舍弃** | CG-2026-0026（`T.sync_block_set/wait` codegen `syn_instr_mode` 与随附编译器 `sync_instr_mode` 不匹配，官方样例复现）+ CG-2026-0010（Developer persistent + `T.gemm` + 整型索引段共存即崩）。**故 v0 的「NPU 上不应独立」结论撤回**（§0.6 R1） |

> 每项处置依据见「NPU 对应方案 / 依据」列。舍弃项（内联折叠、R6 标量消费向量 scan）理由均含**工具链可行性**证据；
> **语义层不可妥协**。
> **判定输入含 Phase R 调研结论（§1.6.0）**：调研在「可实现族」内择优——v2 两 kernel GM 交接（D-3 探针设备实测位精确）
> 胜出；v1 单 scope 标量消费向量 scan（CG-2026-0027）与内联形态（CG-2026-0010/0026）因工具链不可实现被否决
>（可行性前置于经济性）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计 / 形态确认」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**两 kernel GM 交接——kernel A 硬件前缀 scan 产 `cum` 落 GM；kernel B 逐专家
区间展开产 3 张 GM 调度表**）。

**承载方式确认 R1：保留源「独立调度单元 + GM 表」，撤回 v0「并入 GEMM 核」；v2 内部细分为两 kernel**

- **源方案**：`_tile_scheduler_kernel` 独立 kernel，写 `tile_expert_ids/tile_row_offsets/total_tiles` 三张 GM 表；
  GEMM 核经 GM 读 `tile_expert_ids[bx]/tile_row_offsets[bx]/total_tiles[0]`（每个逻辑 tile 一次标量读）。
  性能意图：**消除每 CTA 的 O(E) 专家扫描** + 保持 `total_tiles` 在设备侧（零 CPU 同步）。
- **v0 方案（已撤回/舍弃）**：把调度**内联**进 GEMM 核的 Vector scope，表不落 GM，核内 Vector→Cube 交接
  （Developer 主选 auto CV-split；Expert 回退经 GM scratch + `sync_block_set/wait`）。收益主张：
  「消 1 次 kernel launch + 3 张表 GM 往返 + 1 次跨 kernel barrier」。
- **v0 撤回论证（胞弟 Stage 3 实测，本仓可复现）**——
  (i) Developer 单 Scope（persistent + `T.gemm` + 整型索引段）lowering 期 **segfault**（CG-2026-0010 第二证，
  occurrences=2，「二者共存即崩」）；
  (ii) Expert 回退的跨引擎同 block 交接被 **CG-2026-0026** 阻断：`T.sync_block_set/wait` codegen 出
  `syn_instr_mode`，随附 `bishengir-compile` 解析 `sync_instr_mode` → `error: custom op 'syn_instr_mode' is unknown`，
  **官方 `examples/mixcv/mixcv_mixkernel.py` 复现同一报错** ⇒ 一切跨引擎同 block 交接均不可编译；去掉同步则出现
  竞态（同 kernel 多次运行 max_diff 取值集合含 9.5e-7 / 1.26e-2 / 全零）。
  证据：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/repro.log`、`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`、
  `.agents/evolution/capability-gaps.md` CG-2026-0010 / CG-2026-0026。
  故 v0 的「融合单核 + 零跨核同步」形态**无可用工具链路径**，收益主张随之撤回。
- **v2 细化（两 kernel 拆分）**：源「单 kernel 内 thread-0 扫描 + 每 thread 二分」在 NPU 上**不能一步落地**——v1
  把它译为「单 Vector scope：UB 内 `T.cumsum` scan + 标量逐 tile 二分/单调指针」，被 Stage 3 实测证伪
  （**CG-2026-0027**：标量消费向量产出结果不可同步，同编译产物多次运行结果不一致）。承载形态**仍是独立单元 + GM 表**，
  但内部拆成两个 kernel（数据经 GM `cum_gm` 交接，不跨「向量生产 → 标量消费」同 scope 边界）：
  - **kernel A**（`_sched_scan_kernel`，Vector，`T.Kernel(1, is_npu=True)`）：`true_sizes[E]`（GM）→ `T.copy` 入 UB →
    `cnt` 向量化 → **一次 `T.cumsum`** 得含前缀 → 1-lane 移位得排他前缀 `cum[E+1]` → `T.copy` 落 **GM `cum_gm[E+1]`（int32）**。
    全程向量域，无标量消费向量值。
  - **kernel B**（`_sched_expand_kernel`，Vector，固定核数）：读 GM `cum_gm`（**标量 GM 读**，与胞弟 GEMM 的
    `tile_expert_ids[bx]`/`true_offsets[expert_id]` 同型，已证可用），对本核所辖每个专家 e 把区间
    `[cum_e, cum_{e+1})` 一次性向量填充为 `(e, k·block_m)`；死 tile 区间 `[total, max_tiles)` 由单核写哨兵 `(-1,0)`。
  - **为什么必须两个 kernel**：kernel B 需要 `cum_e` 作为**标量**控制 DMA 切片长度；若 `cum` 停留在 kernel 的 UB
    （由 `T.cumsum` 向量产出），标量读它正是 CG-2026-0027 的失败形态。把 `cum` 落 **GM** 并在**另一个 kernel**
    标量读，读写由 **kernel 边界**隔离（同 stream 顺序），不混用标量/向量数据流。
- **NPU 新算法（主选，两 kernel）**：
  1. kernel A：`T.copy(true_sizes → ub_sizes[E])`；`T.Parallel(E): ub_cnt[e]=(s_e+block_m-1)//block_m`；
     `T.cumsum(ub_cnt, ub_incl, dim=1)`（**唯一一次 scan**）；`ub_cum[0]=0; ub_cum[1:E+1]=ub_incl[0:E]`；
     `T.copy(ub_cum → cum_gm)`。
  2. kernel B：`total = cum_gm[E]`（core 0 写 `total_tiles[0]`）；对专家 e（核内 `T.serial`）：
     `c_lo=cum_gm[e]`、`c_hi=cum_gm[e+1]`（标量 GM 读）；`T.Parallel(mte): ub_e[tt]=e, ub_r[tt]=tt*block_m`；
     `if c_lo<c_hi: T.copy(ub_e[0:cnt] → tile_expert_ids[c_lo:c_hi]); T.copy(ub_r[0:cnt] → tile_row_offsets[c_lo:c_hi])`
     （`cnt=c_hi-c_lo`，动态 extent DMA，同胞弟 GEMM 的动态 extent 用法）。
  3. 死 tile：`if core_id==0 and total<max_tiles:` 向量填 `(-1,0)` 到 `[total, max_tiles)`（与有效区间不相交，无写竞态）。
  表经 **GM** 交给胞弟 GEMM 核（kernel 间 + 与 GEMM 同 stream 顺序 launch）。
- **语义保持论证**：三张表**逐元素值完全不变**——kernel B 的区间展开与源 `searchsorted(cum,t,right=True)-1`
  逐位相同（§1.6.1 E2：expansion == 二分 == 线性分桶 == searchsorted，48320 例 0 违反）；空专家（`c_lo==c_hi`）
  不写；死 tile 写 `(-1,0)`。int32 逐位精确语义不变。
- **与源算法优化意图的承接**（§0.4 逐项，**不得静默丢弃**）：
  - **O2（前缀扫描）：承接**——kernel A 硬件 `T.cumsum`（含前缀 + 1-lane 移位 = 排他前缀）；
  - **O3（二分查找 O(log E) 定位 tile 归属）：意图承接、形态替换**——源意图是「免每 tile O(E) 扫描」；v1 的标量
    二分/单调指针在 NPU 上不可实现（CG-2026-0027 + CG-2026-0028b 循环携带标量），v2 改为**逐专家区间展开**，
    每个有效 tile 只写一次，代价 O(total_tiles) ≤ max_tiles（**不超过**源 O(max_tiles·logE) 的比较量），
    且与源查找数学位精确等价（§1.6.1 E2）；这是「O(log E) 查找」在向量/标量-GM 受限工具链下的等价最优落地，
    源「免 O(E) 逐 tile 扫描」意图完整承接；
  - O1（SMEM `cum`）：**等价替换**（kernel A UB）；O4/O5（thread 并行 + 超发 grid）：**重设计**（R4）；
    O6（`out_idx` 设备侧分配）：**保留**（kernel B 3 张 GM 输出表）；O7（独立调度表 + GM 交接）：**保留**；
  - **「零 CPU 同步」：承接**（`total_tiles` 全程设备侧，`cum_gm`/3 表均为设备侧）。
- **对接契约（与胞弟 `_moe_grouped_gemm_kernel` 严格对齐，详见其 §0.6 R1 / §4.1）**：
  1. `max_tiles = numel // block_m + E` 与 `block_m` 为两函数**共享的编译期常量**；
  2. 表 dtype 全 **int32**；`tile_row_offsets` 单位是**行**（已乘 `block_m`），不是 tile 序号；
  3. 死 tile 哨兵 `(-1, 0)`；`total_tiles ≤ max_tiles`；
  4. 消费方式：胞弟 GEMM 核按 `bx = tile_id // num_pid_n`、`by = tile_id % num_pid_n` 读
     `total_tiles[0]` 与 `tile_expert_ids[bx]` / `tile_row_offsets[bx]`，`bx >= total_tiles[0]` 死 CTA 早退；
  5. kernel A → kernel B → GEMM 核 **同一 stream 顺序 launch**：前者完成后者再执行，**无 in-block 同步、无 CPU 同步**。

**重设计项 R2：前缀扫描——从「thread 0 串行标量」改为「kernel A 硬件 scan `T.cumsum`」**

- **源方案**：`thread 0` 用 `T.serial(E)` 标量循环逐元素累加写 SMEM；其余 thread 空等 + `sync_threads`。
- **NPU 新算法（kernel A）**：`T.cumsum(ub_cnt, ub_incl, dim=1)`（`ub_cnt` 形如 `(1,E)` int32）一次性得含前缀和；
  再 `T.copy(ub_incl[0,0:E], ub_cum[0,1:E+1])` + `ub_cum[0,0]=0` 得排他前缀，`T.copy` 落 GM `cum_gm`。E=256 时把
  ~256 次标量加压缩为 1 条 `VCumsumOp` + 1 次向量拷贝。**单核一次**（`T.Kernel(1)`；源为每 block 冗余重扫，
  v1 为每核冗余重扫，v2 收敛为单核一次）。
- **语义保持论证**：含前缀 → 排他前缀的 1-lane 移位是**整数恒等式**（`cum[e+1]=incl[e]`）；机器验证 E1（2244 例 0 违反），
  另设 **shift guard**：若省略移位直接用含前缀和做 `cum`，映射错位（175571 个 tile 失配）——证明该移位是负荷项而非装饰。
  **实机正向结论（闭合 v1 §9.4 R-1/R-4）**：`T.cumsum` 在 **int32、`(1,E)` 首维=1** 上**设备实测正确**
  （`repro/TRAP-VECTOR-SCALAR-DATAFLOW.py` ASSERT D → `[1,3,6,10]`；D-3 探针亦通过）。
  **效率注记**：int32 scan 在本工具链以 scalar instruction lower（编译器 warning
  `hivm.hir.vcumsum will execute by scalar instruction with low efficiency`），E≤256、单核一次，成本可忽略。

**重设计项 R3：tile→(expert,row) 解析——从「每 tile 独立二分」改为「kernel B 逐专家区间展开」**

- **源方案**：每 thread 认领一个 M-tile，各自独立 `log2_up` 步二分。
- **v1 方案（已否决）**：核内单调前向指针 + 首 tile 一次二分，**标量消费 UB `cum`**（`while ub_cum[0,cur_e+1]<=tile_id`）——
  命中 CG-2026-0027（标量读向量产出 UB 不同步）+ CG-2026-0028b（循环携带标量），**不可实现**。
- **NPU 新算法（kernel B）**：区间 `[cum_e, cum_{e+1})` 内的每个 tile t 都属于专家 e，且行偏移
  `(t-cum_e)*block_m` 恰为区间内序号 k 的 `k*block_m`。故对每个专家 e：广播 `(e, k*block_m)` 到 UB
  （`T.Parallel(mte)`），再以**动态 extent DMA** 一次写 `[cum_e, cum_{e+1})`。无需任何搜索、无需循环携带标量。
  代价 O(Σ cnt_e)=O(total_tiles) ≤ max_tiles，每 tile 只写一次。
- **语义保持论证**：映射与源 `searchsorted(cum,t,right=True)-1` **逐位一致**（§1.6.1 E2：expansion == 二分 == 线性
  分桶 == searchsorted，48320 例 0 违反）。边界：空专家 `c_lo==c_hi` 跳过；死 tile 由独立分支写哨兵（E3：13 例 0 违反）。

**重设计项 R4：任务划分与分核——从「超发 grid」改为「固定物理核数 + 核内串行」**

- **源方案**：`T.Kernel(ceil(max_tiles/threads), threads=256)` 超发 block（thread 级一 tile）。
- **NPU 新算法**：**kernel A** 单核（`grid=1`，一次小向量 scan）；**kernel B** `nk_B = min(E, P)`
  （`P` = **纯 Vector 实查物理核数 = 40**，§5.5），核内 `for ee in T.serial(epc)`（`epc = ceildiv(E, nk_B)`，
  编译期常量，**循环边界静态**）遍历本核所辖专家；死 tile 区间由单核守卫写。
- **语义保持论证**：分核只改变「哪个核写哪些 tile」，不改变任何 tile 的 `(expert,row)` 结果；纯调度，无副作用。
  边界：`epc` 静态保证循环边界安全；`total` 由 `cum_gm[E]` 给出；死 tile 区间 `[total,max_tiles)` 与有效区间不相交。
- **说明**：本单元为**纯 Vector 调度单元**，物理核数基线 = `get_aicore_num() * 2 = 40`（§5.5 实查记录）；
  超发 grid 的源形态在 NPU 上会被串行调度（core-split-strategy.md §1），不可取。

**重设计项 R5：无（算法层）**。前缀扫描与 tile→专家映射的**算法本身**（线性一遍 + 单调映射）在 NPU 上有等价能力，
按 §0.5 为「保留」；R2/R3 只改**实现形态**（硬件 scan / 逐专家区间展开），且展开与源查找数学位精确等价
（§1.6.1 E2），不换算法族；R1 为承载形态确认 + 两 kernel 细分。

**弃选：v1 单 Vector scope「UB scan + 标量消费」（Stage 3 实测证伪，本版直接修订对象）**

- **结构**：单核内 UB `T.cumsum` 产 `cum`，随后**标量** per-tile 二分/单调前向指针 + 死 tile 单比较守卫
  （`if tile_id < ub_cum[0,E]`），依赖 Developer 自动同步（v1 §7.1）。
- **弃选理由（可行性）**：**CG-2026-0027**——标量读由向量 op 写入 UB 的值不同步，同一编译产物多次运行结果不一致
  （`[0,2,4,6,8]` vs `[0,0,4,6,8]`），`T.pipe_barrier("PIPE_V"/"PIPE_ALL")` 无效；连带 **CG-2026-0028** 的
  `T.alloc_local+serial` / int32 `>>` / 标量 `T.if_then_else` / 运行期下标 UB 读 / 双 `T.cumsum` 等一串
  codegen SIGSEGV。证据：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`、
  `.agents/evolution/capability-gaps.md`。v2 用「`cum` 落 GM + kernel B 标量 **GM** 读 + 区间展开」彻底规避
  （D-3 探针设备实测位精确、6 次稳定）。

**弃选：内联融合（v0 主选）**

- **结构**：调度并入 GEMM 核 Vector scope，表不落 GM，核内 Vector→Cube 交接。
- **弃选理由（可行性，非仅经济性）**：工具链**不可实现**——Developer（CG-2026-0010）/ Expert（CG-2026-0026）双阻断，
  官方样例复现。该形态**从未被任何本仓先例证明可编译**；实测可用底座是**独立调度单元产出 GM 表**
  （胞弟 Stage 3 max_diff 9.537e-07；本单元 v2 两 kernel 形态 D-3 探针位精确）。理论收益（少 1 launch + 少 3 表往返 ≈6 KB）
  在「不可实现」面前不成立（§12 教训）。

**弃选：host 侧调度**

- 需要 `true_sizes`（**运行时**路由结果）设备→host 回传 + 表 host→设备下发 → 引入 CPU 同步，**直接违反**源
  「零 CPU 同步」的核心设计意图（源 docstring L2–L4）；且 host 侧禁止改动/搬运输入真实内容
  （ascend-constraints.md §4）。弃选。

### 0.7 标杆实现

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_nopad.py` 的 `_tile_scheduler_kernel`（L55–L120）。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py`。
- 胞弟设计（消费本单元产出的调度表，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md` §0.6 R1 / §4.1。
- host 侧 golden 参考（`true_offsets` 的构造与逐 tile 展开）：`testing/autotune/anneal/moe_grouped_gemm_anneal.py`
  的 `supply_prog`（L124–L143，纯 Python pieces-per-expert 展开）。
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tileops/ops/moe/moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoeGroupedGemmNopadFwdOp`）。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_tile_scheduler_kernel`（MoE no-pad 分组 GEMM 的 tile 调度器，纯整数索引，`target="npuir"`）。

### 1.2 功能描述

把每专家 token 数 `true_sizes[E]` 翻译为「M-tile → (专家号, 专家内行偏移)」的调度元数据，并给出有效
M-tile 总数 `total_tiles`，供 no-pad 分组 GEMM 做 O(1) 寻址。**迁移决策后的 NPU 形态**（§0.6 R1）：
作为**两 kernel GM 交接的独立调度单元**（kernel A：`true_sizes` → `cum_gm`；kernel B：`cum_gm` → 3 张 GM int32 表
`tile_expert_ids` / `tile_row_offsets` / `total_tiles`），由胞弟 Expert Cube GEMM 核经 GM 读表消费；
kernel A → kernel B → GEMM 同 stream 顺序 launch，无 in-block 同步。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
cnt_e=\Big\lceil \tfrac{s_e}{block\_m}\Big\rceil,\quad
cum_0=0,\ cum_{e+1}=cum_e+cnt_e,\quad total\_tiles=cum_E
$$

$$
e(t)=\max\{e:cum_e\le t\}\ (t<total\_tiles),\quad row(t)=(t-cum_{e(t)})\cdot block\_m
$$

输入 `true_sizes`（int32）；输出 `(tile_expert_ids, tile_row_offsets, total_tiles)`（3 张 GM 表）。
**全程 int32，无浮点**。源清单 roofline（`examples/TileOPs/tileops/manifest/moe.yaml`）属于 **op 级**（含 GEMM），
非本调度单元。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源结构差异已逐条注明）：

1. **kernel A — 前缀 scan（Vector，单核）**：`true_sizes`（GM）→ UB → 向量化算每专家 tile 数
   `cnt[e]=(s_e+block_m-1)//block_m` → **一次 `T.cumsum`** + 1-lane 移位得排他前缀 `cum[E+1]` →
   `T.copy` 落 **GM `cum_gm`**。全程向量域；**无标量消费向量产出**。
2. **kernel B — 逐专家区间展开（Vector，固定核数）**：读 GM `cum_gm`（标量 GM 读）→ 对本核所辖每个专家 e
   把区间 `[cum_e, cum_{e+1})` 向量广播填充为 `(e, k·block_m)`，写 2 张 GM 表；死 tile 区间 `[total,max_tiles)`
   写哨兵；`total_tiles[0]=cum_gm[E]`（设备侧）。
   **与源结构差异**：源为**超发 grid + 每 thread 一 tile 并行二分 + 每 block 重复 thread-0 串行扫描**；
   v1 为**单 scope UB scan + 标量消费**（实测不可实现，CG-2026-0027）；v2 为**两 kernel GM 交接 + 区间展开 +
   固定物理核数（纯 Vector → 40）核内 `T.serial`**（§0.6 R1–R4）。
3. **消费方式**：3 张表落 **GM**，由胞弟 Expert Cube GEMM 核经 GM 标量读（O(1) 寻址）+ 死 CTA 早退守卫消费
   （对接契约见 §0.6 R1）。本单元自身**不消费**、不做任何 GEMM。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[kernel A · Vector · grid=1]
true_sizes[E] ──T.copy──▶ UB[ub_sizes(1,E)] ──T.Parallel(E)──▶ UB[ub_cnt] ──T.cumsum + 1-lane 移位──▶ UB[ub_cum(1,E+1)]
                                                                                        │ T.copy
                                                                                        ▼
                                                                    GM[cum_gm(E+1) int32]  ──（kernel 边界，同 stream 顺序）──┐
                                                                                                                              ▼
[kernel B · Vector · nk_B=min(E,40) cores]
  GM[cum_gm] ──标量 GM 读 c_lo/c_hi（控制 DMA 长度）──▶ UB[ub_e/ub_r 广播 (e, k·block_m)]
                                                            │  动态 extent T.copy（区间 [cum_e,cum_{e+1})）
                                                            ▼
                        GM[tile_expert_ids(max_tiles) int32]（专家号 / 死 tile=-1）
                        GM[tile_row_offsets(max_tiles) int32]（专家内行偏移，已乘 block_m）
                        GM[total_tiles(1) int32]（= cum_gm[E]，设备侧）
                                     │  （同一 stream 顺序 launch：调度单元完成后 GEMM 核执行，无 in-block 同步）
                                     ▼
                    [胞弟 _moe_grouped_gemm_kernel，Expert Cube-only] 标量读表 + 动态 extent T.gemm
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（「每专家 token 数 → 每 M-tile 的专家/行归属」的整数调度映射）的**算法族**。
> 输入/源算法（独立 kernel + thread 并行二分 + GM 表）只是基线候选之一。深度分级：多步索引/搜索类 → 完整调研。
> **可实现性约束（前置，来自 Phase M1 + Stage 3 实测）**：
> (a) 本工具链禁止跨引擎同 block 交接（CG-2026-0026，官方 `examples/mixcv/mixcv_mixkernel.py` 复现），且
>     Developer persistent + `T.gemm` + 整型索引段共存即 lowering segfault（CG-2026-0010）→ 任何「调度内联进
>     GEMM 核」的候选**不可实现**；
> (b) **单 Vector scope 内「向量产出 → 标量消费」不可实现**（CG-2026-0027：`T.cumsum`/`T.Parallel` 写入 UB 的值
>     被标量代码读取消费时不同步，同产物多次运行结果不一致；`T.pipe_barrier` 无效），且 `T.alloc_local+serial` /
>     int32 `>>` / 标量 `T.if_then_else` / 运行期下标 UB 读 / 双 `T.cumsum` 均 SIGSEGV（CG-2026-0028）。
>     证据：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`。
> 调研在**可实现族**内择优（可行性前置于经济性）。v1 的 C1+C3（单 scope 标量消费）因此在 v2 被排除。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | 独立 kernel；thread0 串行前缀扫描 → SMEM `s_cum`；每 thread 一个 M-tile 并行二分；超发 grid；写 3 张 GM 表 | — | — | ✅ 基线 |
| C1 | **两 kernel GM 交接 + 逐专家区间展开**（**选定**） | kernel A（Vector，1 核）`T.cumsum` 产排他前缀 `cum` 落 GM；kernel B（Vector，固定核数）逐专家把区间 `[cum_e,cum_{e+1})` 展开为 `(e,k·block_m)` | 数学恒等（表内容与源逐位一致，§1.6.1 E2） | 消源超发 grid/每 block 重扫；**无标量消费向量产出**（规避 CG-2026-0027） | ✅ **选定** |
| C2 | **硬件前缀 scan**（`T.cumsum` + 1-lane 移位） | 排他前缀 = 含前缀 cumsum 后 1-lane 移位（整数恒等 E1） | 数学恒等（整数精确） | O(E) 标量串行 → 1 条 VCumsumOp | ✅ 采纳（并入 C1 kernel A） |
| C3 | **逐专家区间展开**（`[cum_e,cum_{e+1})` 一次填 `(e,k·block_m)`） | 展开 == `searchsorted(cum,t,right=True)-1`（E2） | 数学恒等（同一映射，不同求值） | 每有效 tile 只写一次 O(total_tiles)；免每 tile 搜索 | ✅ 采纳（并入 C1 kernel B） |
| C4 | **v1 单 scope：UB scan + 标量二分/单调指针** | 单 Vector scope 内 UB `cum` 由标量消费（二分/前向指针/死 tile 守卫） | 数学恒等（算法等价） | 单 kernel、无表往返 | ❌ **撤回/弃选**：CG-2026-0027（标量消费向量产出不可同步）+ CG-2026-0028（一串 SIGSEGV），Stage 3 实测证伪 |
| C5 | **内联融合**（v0 主选，表不落 GM） | 调度并入 GEMM 核 Vector 索引段 + 同 block 交接 | 数学恒等 | 理论消 1 次 launch + 3 表往返 + 1 barrier | ❌ **撤回/弃选**：CG-2026-0010 / CG-2026-0026，官方样例复现 |
| C6 | 无搜索映射（`expert = tile // max_tiles_per_expert`） | 网格直接编码专家，免查找 | 数学恒等 | 省查找，但死 tile 比例暴增 | ✅ 对比（弃选） |
| C7 | 排序/直方图分桶（counting sort / radix） | 按专家分桶后前缀 | 数学恒等 | 通用度高但需额外物化，E 小（≤256）时 overkill | ✅ 对比（弃选） |
| C8 | host 侧调度 | CPU 算表后下发 | 数学等价 | 引入 CPU 同步，违反零 CPU 同步意图 | ✅ 对比（弃选） |
| C9 | LUT / 线段树 | 每专家直接查表 | 数学恒等 | 需 O(max_tiles) 表，与 C1 重叠且更重 | ✅ 对比（弃选） |

**R2 在线算法**：**有**。前缀扫描 `cum[e+1]=cum[e]+cnt[e]` 是天然的**单遍在线 running sum**（O(1) 状态）；
逐专家区间展开也是**单遍流式**（对 `true_sizes` 与输出表各扫一遍、无反向依赖）。
结构判据：输出（`e(t),row(t)`）是单调查找函数，无跨 tile 反向依赖；`true_sizes` 只需扫一遍。
**结论严谨化**：不存在需要 **GM 级多遍物化**的候选（不同于 softmax/attention 的两遍 max/sum）；唯一的多步形态是
**UB 内**的 Hillis–Steele 树形扫描回退（$\log_2 E$ 轮位移加，不落 GM；仅在 `T.cumsum` int32 不可用时启用）。
（对照 `references/algorithm-candidates.md` 的 ALG-reduction：「分块归约 + 合并（两阶段 / 树形）」即此类片上多步候选。）
收益口径：扫描遍数 = 1（`true_sizes` 一遍）；中间缓冲 O(E)（`cum` 表 int32，落 GM `cum_gm`）。

**R3 复杂度对比**（四口径；代表 shape = deepseek-v3-prefill-gate-up：numel=32768，E=256，block_m=64
→ `max_tiles=768`、`nk_B=min(E,40)=40`、`epc=ceildiv(256,40)=7`、`total_tiles=512`；elem=int32=4B。
「访存」含 `true_sizes` 读 + `cum` 中间表写读 + 3 张表 GM 写 + 消费侧标量读（GEMM 侧，L2 命中））：

| 算法候选 | 计算量（整数 op） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|---------------|---------|-------------|---------------------|
| C0 基线（thread0 串行扫描 + 每 thread 二分 + 超发 grid） | E 加 + max_tiles·log2(E) 比较 ≈ 256 + 768·8 ≈ **6400** | `true_sizes` 1KB×3 blocks = 3KB + 表写 6.1KB + 消费读 ≈98KB ≈ **107KB** | true_sizes 1 遍（×3）；表 1 写 1 读 | SMEM `cum` 1KB + GM 表 6.1KB | 高（thread 级）；**跨 kernel barrier** |
| **C1+C2+C3 选定（kernel A 单核 scan + kernel B 逐专家展开）** | kernel A：E 加（1 核）≈256；kernel B：Σ cnt_e = total_tiles ≈512 次「广播+拷贝」；合计 ≈ **768** | `true_sizes` 1KB（单核 1 次）+ `cum` 写读 1KB+1KB + 表写 6.1KB + 消费读 ≈98KB ≈ **108KB** | true_sizes 1 遍（单核）；表 1 写 1 读 | UB A ≈4KB + UB B ≈10KB | 高（kernel B 按专家分布到 40 核）；**无跨核归并、无 in-block 跨引擎同步** |
| C4 v1 单 scope 标量消费（不可实现） | E 加 + 每 tile 搜索摊销 ≈768+256 | `true_sizes`×nums_kernel(40)=40KB + 表写 6.1KB | true_sizes ×40 | UB ≈4KB | **不可实现**（CG-2026-0027/0028） |
| C5 内联融合（v0，不可实现） | 每核 E 加 + 摊销指针 | `true_sizes` 1KB×20 = 20KB | true_sizes ×20 | UB ≈4KB | 工具链**不可实现**（CG-2026-0010/0026） |
| C6 无搜索映射 | ~max_tiles 次整除/取模 + 守卫 | 同 C1 | 同 C1 | 同 C1 + 免 `cum` | 逻辑 tile = E·max_tiles_per_expert = 256·512 = **131072**，相对 C1(768) **死 tile ≈ 170×**（占比 ≈99.4%） |
| C7 分桶/排序 | O(numel·E) 比较（重） | 需额外桶缓冲物化 | ≥2 遍 | O(E·max_tiles_per_expert) 起 | 通用但物化缓冲远超本单元粒度 |
| C8 host 调度 | host 侧 O(E+max_tiles·logE) | `true_sizes` 设备→host + 表 host→设备 | 1 遍 + 2 次总线往返 | host 内存 | **CPU 同步阻塞**（违反零同步） |
| C9 LUT | O(max_tiles) 表 | 表 materialize 更大 | 1 遍 | O(max_tiles·E) | overkill |

口径脚注：计算量只计整数加/比较（本算子无浮点）；访存 = 输入读 + 输出写 + 必要中间往返（不含 UB 内部重复）；
扫描遍数指同一数据在 GM 侧被读的次数；中间缓冲峰值以片上（UB）为准。
**本单元是极小工作量的索引算子**（prefill 下全部计算量 ≪ 10^5 次整数 op），故 R3 排序以**固定开销（launch）
与 GM 往返**为主导，而非算力或带宽。**v2 相较 v1**：多 1 次 kernel launch 与 `cum_gm` 的 1KB 写+1KB 读，
但 `true_sizes` 从 v1 的 40 次重读降为 1 次（kernel A 单核），总体访存基本持平、且换来**可实现性**（v1 不可实现）。

**R4 硬件亲和性评估**（逐候选对照检查清单；「工具链可用性」列按 negative-claim-evidence.md §2 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Vector（整数）✅；但**一 thread 一 tile 的并行形态**在 NPU 是 SIMD lane，非 scalar thread → ⚠️ | SMEM ✅ | ✅ | grid 编译期常量 ✅ | ❌ 超发 grid 被串行调度 | ✅ 可编译（源形态） | ⚠️ 可行但超发 grid + 每 block 冗余重扫 → 需重设计 |
| C1+C2+C3 **选定** | Vector（`T.Parallel`/`T.cumsum`/动态 extent `T.copy`）✅ + 无 Cube 需求 | UB A≈4KB + UB B≈10KB ≪ 192KB ✅ | E=256 为向量宽整数倍 ✅ | kernel B `epc`/`nk_B` 编译期常量 ✅ | ✅ 两独立 Vector 核，无 `T.gemm`、无跨引擎交接；**无标量消费 UB** | ✅ **D-3 探针设备实测可编译 + 位精确**（小 shape 3 例 + E=256 prefill 6 次稳定，`history_version/design_probe_two_kernel*.py`）；不触发 CG-2026-0010/0026/0027/0028 | ✅ **选定：可实现、可编译、位精确** |
| C4 v1 单 scope 标量消费 | Vector ✅（但数据流混用标量/向量） | UB ≈4KB | ✅ | ✅ | — | ❌ **不可用**：CG-2026-0027（标量消费向量产出不同步）+ CG-2026-0028（一串 SIGSEGV），Stage 3 实测证伪 | ❌ **撤回/弃选** |
| C5 内联融合 | 需 Vector→Cube 同 block 交接 | UB ≈4KB | ✅ | ✅ | ❌ 需跨引擎交接 | ❌ **不可用**：CG-2026-0010 + CG-2026-0026（官方样例复现） | ❌ **撤回/弃选** |
| C6 无搜索映射 | Vector ✅ | 同 C1 | 同 C1 | ✅ | ✅ | ✅ | ❌ 弃选：死 tile ≈170×，守卫与空转成本高（R3） |
| C7 分桶/排序 | Vector ✅ | 桶缓冲超 UB 需落 GM | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：E≤256 下物化开销远超收益（overkill） |
| C8 host 调度 | CPU | — | — | — | ❌ 引入 host-device 往返与 CPU 同步 | ✅（可行但违约束） | ❌ 弃选：违反零 CPU 同步（源 docstring L2–L4） |
| C9 LUT | Vector ✅ | 表 O(max_tiles·E) 超 UB | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：与 C1 语义重叠且更重 |

**调研结论**：选定 **C1 + C2 + C3 —— 两 kernel GM 交接（kernel A 硬件前缀 `T.cumsum` 产 `cum` 落 GM；kernel B
逐专家区间展开产 3 张 GM 表）+ 固定物理核数（纯 Vector → 40）核内串行**。
关键依据：(i) **可实现性是前置约束**——v1 单 scope 标量消费被 CG-2026-0027/0028 证伪，内联候选被 CG-2026-0010/0026
双阻断；v2 两 kernel 形态经 **D-3 探针设备实测可编译 + 位精确 + 多运行稳定**（唯一同时满足全约束的形态）；
(ii) 本单元全部计算量 ≪10^5 次整数 op（R3），工作量为**固定开销主导**，C1 消源超发 grid 与每 block 冗余重扫、无跨核归并；
(iii) C6/C7/C8/C9 分别因死 tile 爆炸 / overkill / CPU 同步 / 表过重被弃选。
**与基线结构差异一句话**：保留源「独立调度单元写 GM 表 + GEMM 读表」数据流，把源 thread-0 串行扫描换成
kernel A 硬件向量 scan、把每 tile 独立二分换成 kernel B 逐专家区间展开、把超发 grid 换成固定物理核数核内串行，
并把「标量消费向量 scan」用 kernel 边界（GM `cum_gm`）隔离。
**调研范围**：`references/algorithm-candidates.md`（无整数调度/scan-scheduler 专条；参考 ALG-reduction scan 族
「分块归约 + 合并（两阶段/树形）」与「gather/分桶」条目）、pattern-library（layout/elementwise/traps-*、constants.md）、
源算子 §0.3/§0.4、本仓 examples（`examples/cumprod/_cumulative_kernel`、`testing/autotune/anneal/moe_grouped_gemm_anneal.py`、
`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md` 的 Stage 3 实测）、
`.agents/evolution/capability-gaps.md`（CG-2026-0027/0028）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：O1（SMEM `cum`）→ kernel A 的 UB `cum`；O2（排他前缀扫描）
→ C2 硬件 scan（承接「一遍扫描」意图）；O3（二分查找 O(log E)）→ C3 逐专家区间展开（承接「免每 tile O(E) 扫描」
意图，代价 O(total_tiles) ≤ max_tiles，位精确等价）；O4（thread 级并行）→ 重设计（§0.6 R4）；O5（保守上界 + 死 tile 哨兵）
→ §0.6 R4 固定核数 + 死 tile 区间单核哨兵（承接「输出定长、零 CPU 同步」意图）；O6（`out_idx` 设备侧分配）
→ 保留（kernel B 3 张 GM 输出表，`out_idx=[1,2,3]`）；O7（独立调度表 + GM 交接）→ R1 **保留**（数据流逐项保留）。

**设计期 roofline（D-2）**：以 §1.6.3 选定布局 + `pattern-library/constants.md` 实测常数估算下界：
- **流量项（本调度单元）**：`true_sizes` 读 `256·4B = 1KB`（kernel A 单核 1 次）+ `cum_gm` 写读 `1KB+1KB`
  + 3 张表写 `(2·768+1)·4 ≈ 6.1KB` ≈ 9.1KB ÷ MTE2 混合流量地板 `~1.26 TB/s`（CONST-mte2-degradation，
  工具链 tilelang 0.1.2+ed787bb / CANN 8.5.0）≈ **8 ns**（可忽略）；消费侧读（GEMM，L2 命中）另计、不属本单元 roofline。
- **发射项**：2 次 kernel launch（A、B）+ kernel A `T.copy + cnt(T.Parallel) + T.cumsum + 移位拷贝` ≈ 4 op
  × `~0.5µs/op`（CONST-vector-launch-overhead）≈ **2 µs**；kernel B 每核 `epc` 次「T.Parallel 广播 + 2 次动态
  extent T.copy」≈ `7·3 = 21 op` × `~0.5µs` ≈ **10 µs/核**（各核并行 → 墙钟 ≈ 10 µs）。
- **容量项**：UB A ≈4KB + UB B ≈10KB ≪ 192KB（CONST-capacity-910B2C）。
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 2–10 µs（发射项主导）**，一次性、与 GEMM 同 stream 串行叠加。
- **未文档化假设项**：kernel B 动态 extent `T.copy` 的单次发射成本与两次 kernel launch 固定开销**未在 pattern-library 实测**
  （标注「未文档化假设 + 估算依据」）；该量**不影响主选/弃选的裁决**（可行性已由 D-3 探针 + 工具链证据硬裁定），
  仅影响成本特征刻画 → 交**实验裁决计划**（下）。

**实验裁决计划（三件套）**：

- **主选（进 Stage 3）**：两 kernel GM 交接——kernel A 单核 `T.cumsum`（int32）产 `cum_gm`；kernel B 逐专家
  **动态 extent DMA 区间展开**（本设计 §3.3）。
- **备选（≤1，结构完整可实现）**：把 kernel B 的「多核 + 动态 extent 区间写」退化为**单核对本核专家 `T.serial` 标量写**
  （`for e in T.serial(epc): ... for k in T.serial(mte): if k < cum_{e+1}-cum_e: tile_expert_ids[cum_e+k]=e;`），
  kernel A 不变。buffer 形状（`cum_gm(E+1)`、表 `(max_tiles,)`）、三表契约、kernel A 结构均与主选一致；
  触发条件：kernel B 动态 extent `T.copy` 在本 shape 编译失败或出现编译期 flaky SIGSEGV（同胞弟 GEMM 记录）。
- **实验裁决计划**：① **代表 shape** = smoke（E=4,max_tiles=9）、prefill gate-up（E=256,max_tiles=768）；
  ② **测量指标** = 首编编译结果 + 三表与 golden 逐元素精确一致（int32，非容差）+ 反复运行稳定；
  ③ **判定阈值** = 若主选 kernel B 动态 extent 路径编译失败/数值错 → 立即切备选；若调度单元墙钟 > GEMM 有效时间 5%
  方考虑进一步优化（合并写入 / kernel A+B 融合的可行性探针）；
  ④ **回写路径** = 实测数字回写 §1.6.0 roofline 与 §5.6；触发降级/升级按 perf-feedback.md 受控路由。
  **shape 特化可行性**：本单元经 `lru_cache` 按 `(num_experts, max_tiles, block_m)` 特化，可按 shape 分派主选/备选。
  **设计期探针（D-3，已执行，见 §3.4/§9）**：`history_version/design_probe_two_kernel.py`（小 shape 3 例 + 6 次稳定）
  与 `history_version/design_probe_two_kernel_large.py`（E=256 prefill，6 次稳定）**设备实测位精确通过**——
  主选的动态 extent 区间展开路径已被证明可编译可运行（判定依赖的未实测常数已由探针标定为「可用」）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1+C2+C3 算法（两 kernel：kernel A scan + kernel B 逐专家展开）。每项四要素：
> 原式 → 优化后公式 → 等价性论证 → 收益量化。
> 等价性机器验证（D-1）脚本：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/verify_equiv.py`（**整数精确，非容差**）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | 排他前缀 = 含前缀 + 1-lane 移位（kernel A） | 源串行 `cum[0]=0; cum[e+1]=cum[e]+cnt[e]` | `incl=T.cumsum(cnt); cum[0]=0; cum[1:E+1]=incl[0:E]` | 整数恒等（`cum[e+1]=incl[e]`）；机器验证 2244 例 0 违反；**shift guard** 证明省略移位会错位（175571 tile 失配） | 把 O(E) 标量串行压成 1 条 `VCumsumOp` + 1 次向量拷贝；使扫描在 NPU 可向量化 |
| E2 | 逐专家区间展开（kernel B） | 源每 tile 独立二分 `e(t)=searchsorted(cum,t,right=True)-1` | 对专家 e 把区间 `[cum_e, cum_{e+1})` 一次填 `(e, k·block_m)` | 数学恒等（同一映射，逐位一致）；机器验证 48320 例 0 违反（expansion == 二分 == 线性分桶 == searchsorted） | 每有效 tile 只写一次，O(total_tiles) ≤ O(max_tiles)（prefill：512 vs 源 768·8=6144 比较）；**无标量消费向量产出** |
| E3 | 死 tile 哨兵 | — | `t≥total` → `(-1,0)`（kernel B 独立区间写） | 与有效映射无重叠/无空洞；机器验证 13 例 0 违反 | 定长输出、零 CPU 同步 |
| — | （撤回项）v1 整数 ceil 移位式 `>>` | 源 `(s+block_m-1)//block_m` | v1 曾主张 `>> log2(block_m)` | 数学恒等（诊断 24612 例 0 违反），**但不采纳**：int32 `>>` 使 codegen SIGSEGV（CG-2026-0028a） | **撤回**：改用源 `// block_m`（kernel A），数学等价、无收益损失 |
| — | （撤回项）v1 单调前向指针 | 源每 tile 二分 | v1 曾主张「首 tile 二分 + 单调指针」 | 数学恒等（v1 48320 例验证），**但不采纳**：标量消费 UB scan 不可同步（CG-2026-0027） | **撤回**：改用 E2 逐专家区间展开（等价、更省且可实现） |

**优化结论**：v2 采纳 **E1/E2/E3**（v1 的 M1/M3 撤回：M1 因 codegen `>>` SIGSEGV、M3 因 CG-2026-0027）。
**优化后公式**（供 §3.1 拆解）：对每专家 $e$、`cnt_e=(s_e+block_m-1)//block_m`；`incl=cumsum(cnt)`；
`cum_0=0, cum_{e+1}=incl_e`；`total=cum_E`；对每个专家 `e`：写 $t\in[cum_e,cum_{e+1})$ 为
`tile_expert_ids[t]=e`、`tile_row_offsets[t]=(t-cum_e)*block_m`。

**等价性机器验证结果表**（`verify_equiv.py`，纯整数对照，torch CPU；metric 为最大绝对差）：

| 项 | 对照 | 最大度量值 | 违反 / 用例 | 结论 |
|----|------|-----------|-------------|------|
| E1 | 含前缀 cumsum + 移位 vs 源串行排他扫描 | 0.000e+00 | 0 / 2244 | EQUIV_PASS |
| E2 | 逐专家区间展开 vs 源每 tile 二分 vs ground-truth 线性分桶 vs searchsorted | 0.000e+00 | 0 / 48320 | EQUIV_PASS |
| E3 | 死 tile 哨兵（`t≥total` → `(-1,0)`）与有效映射无重叠/无空洞 | 0.000e+00 | 0 / 13 | EQUIV_PASS |
| （guard） | 省略 E1 的 1-lane 移位（直接用含前缀和）vs 正确映射 | 175571 个 tile 失配 | — / 80 | 确认 E1 移位负荷不可省 |
| （diag，未采纳） | ceil 移位恒等（`>>` vs `//`） | 0.000e+00 | 0 / 24612 | 恒等成立，但 `>>` codegen 不可用（CG-2026-0028a），改用 `//` |

无失败项（所有采纳项均为 EQUIV_PASS）。**量化影响评估**：本单元**全程 int32 精确整数运算**，
采纳项均为**逐位精确恒等**（非容差内等价），**不引入任何 fp16/bf16 舍入误差**——因此精度标准为
「输出表逐元素精确一致」（§8.2）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案全部循环与标量计算点（对象 = §1.6.0 选定、经 §0.6 重设计落地的 C1+C2+C3 两 kernel 算法）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 每专家 tile 数 `cnt[e]`（kernel A） | 源 thread0 标量循环 | `T.Parallel(E)` 逐元素整数；`true_sizes` 经 `T.copy` GM→UB | ✅ | — |
| 2 | 排他前缀 `cum`（kernel A） | 源 thread0 串行标量扫描 | **`T.cumsum`（硬件 `VCumsumOp`）+ 1-lane 移位**（E1） | ✅ | — |
| 3 | `true_sizes → ub` 搬运（kernel A） | 标量循环读 | `T.copy`（GM→UB 向量搬运） | ✅ | — |
| 4 | 表区间填充（kernel B） | 源每 thread 标量二分 + 标量散写 | **逐专家区间展开**：`T.Parallel(mte)` 广播 `(e, k·block_m)` + 动态 extent `T.copy` 写整段区间（E2） | ✅ | —（v1 判「不可向量替代」的逐 tile 搜索已被 E2 整体消除） |
| 5 | 死 tile 哨兵写（kernel B） | 标量逐 tile 比较 + 写 | `T.Parallel(max_tiles)` 填 `(-1,0)` + 动态 extent `T.copy` 写 `[total,max_tiles)` | ✅ | — |
| 6 | 专家循环与区间下标 `e`、`c_lo/c_hi`（kernel B） | 标量循环 + 标量 GM 读 | 无（标量控制 DMA 长度） | ❌ | **标量控制流**：`e` 为 `T.serial` 循环变量，`c_lo/c_hi = cum_gm[e]` 是**标量 GM 读**（读取由**另一 kernel** 写入 GM 的值，非本 scope 向量产出），用于给出**动态 extent DMA 长度**；无逐元素等价 API，且这是规避 CG-2026-0027 的必要形态（D-3 探针实测可用） |
| 7 | `total_tiles=cum_gm[E]` 与写 `total_tiles[0]` | 标量 GM 读 + 标量 GM 写 | 无 | ❌ | 单次标量 GM 读（编译期下标 E）+ 单次标量 GM 写（core 0）；非逐元素热点，无向量收益 |

**有向量形态但弃选者**：v1 的「逐 lane `T.gather(cum, mid)` + `T.vmin/T.max` 二分查找」——弃选理由：
(1) 其余向量化搜索在 `T.Parallel` 内受**下标变换禁令**（`e+1` 非法）与**循环变量条件禁令**限制；
(2) `T.gather` 的索引语义存在文档/签名冲突（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1 标 `tensor`(int32)、
    `inspect.signature` 注解 `Union[list, tuple]`），标注「未文档化假设」；
(3) 搜索已被 E2 的区间展开**整体消除**（无需搜索），故无向量化收益。

**向量化结论**：逐元素计算已全部向量化或整体消除（第 1 项 `T.Parallel(E)`、第 2 项 `T.cumsum`、第 3 项 `T.copy`、
**第 4/5 项区间展开广播 + 动态 extent `T.copy`**）。仅保留 2 处标量点（第 6/7 项：专家循环 + 标量 GM 控制流 /
total 读与写），理由类别：**标量控制流（循环变量 + 跨 kernel GM 标量读，控制动态 DMA 长度）** 与 **单次标量读写**；
均非逐元素热点（每核 O(epc)，相对宿主 GEMM 的 B 访存（15 GB 量级）可忽略）。
**无可向量化而未向量化的逐元素标量循环**；v1 判「不可替代」的逐 tile 有序查找已由 E2 消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本单元为纯整数索引/调度，**无浮点、无矩阵、无 spatial tiling**；「布局」决策 = 索引缓冲的存续位置与
> 扫描轴（专家轴 E）的向量化方式。I/O layout 为契约（`true_sizes` `(E,)`、三张表长度 `max_tiles`）不变。

**轴质量评分**（向量化轴 = 专家轴 E；E=256 为 int32 向量宽度的整数倍）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | E 对向量宽度 | E=256 ✓（2 的幂，int32 向量化无尾 lane） |
| 尾 lane 浪费率 | `(ceil(E/vw)*vw - E)/E` | 0（E 为 2 幂）；`T.cumsum` 对任意 E 亦合法（尾部由原语处理） |
| 累加链形态 | `cnt` 为 stride=1 逐元素；scan 为 `VCumsumOp` | 逐元素无跨步系数问题；scan 为原语内部处理 |
| repack 代价 | 无核内重排 | 0（int32 一维缓冲，无需转置/NZ） |
| UB 容量影响 | kernel A：`ub_sizes/ub_cnt/ub_incl/ub_cum`=`[1,E(+1)]`；kernel B：`ub_e/ub_r`=`[1,mte]`、`ub_neg/ub_zro`=`[1,max_tiles]` | A≈4KB + B≈10KB ≪ 192KB |

**候选矩阵**（调度单元：数据流形态 + 存续位置）：

| # | 方案 | 关键决策 | 代价 / 收益 | 采纳 |
|---|------|---------|------------|------|
| L1 | **两 kernel GM 交接**（kernel A 单核 scan 产 `cum_gm`；kernel B 逐专家区间展开产 3 张 GM 表） | `cum` 中间量落 GM；kernel 边界隔离「向量生产 → 标量消费」 | 2 次 launch + `cum_gm` 往返（≈2KB）+ 表写（≈6.1KB）；**D-3 探针实测可编译 + 位精确** | ✅ **主选** |
| L2 | v1 单 Vector scope（UB scan + 标量消费） | 单 kernel、无表往返 | **工具链不可实现**（CG-2026-0027/0028，Stage 3 实测证伪） | ❌ 弃选（可行性） |
| L3 | 内联进 GEMM 核 Vector 索引段（v0 主选），表不落 GM | 同 block Vector→Cube 交接 | 理论零额外 launch、零表 GM；**工具链不可实现**（CG-2026-0010 / CG-2026-0026，官方样例复现） | ❌ 弃选（可行性；详 §0.6 R1） |
| L4 | host 侧算表 | CPU 调度 | 需 `true_sizes` 设备→host + 表 host→设备；**违反零 CPU 同步**；host 禁改输入（ascend-constraints.md §4） | ❌ 弃选 |
| L5 | 逐 lane 向量化查找（gather 轮询） | 单指令搜索 | `T.gather` 索引语义文档/签名矛盾（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1 vs `inspect.signature`，未文档化假设）+ 展开已消除搜索 | ❌ 弃选（§1.6.2） |

**布局决策结论**：选定 **L1（两 kernel GM 交接）** —— kernel A（Vector，grid=1）产排他前缀 `cum_gm[E+1]`（int32，GM）；
kernel B（Vector，`nk_B=min(E,40)` 核）逐专家区间展开产 3 张 GM int32 表（`tile_expert_ids`/`tile_row_offsets`/`total_tiles`），
索引缓冲驻 **UB（int32）**，向量化轴为**专家轴 E**（kernel A `T.Parallel(E)`+`T.cumsum`；kernel B `T.Parallel(mte)` 广播 +
动态 extent `T.copy`）。repack 路径：无。该结论已同步 §3.3 伪代码、§5.5 分核与 §6 循环结构；L2/L3/L4/L5 的弃选理由
均已附 `docs/` 路径或「未文档化假设 + 估算依据」，符合 negative-claim-evidence.md §2。

**实验裁决三件套（两 kernel 承载 + kernel B 展开路径）**：主选/备选/裁决计划见 §1.6.0 末「实验裁决计划」——
主选 = 两 kernel GM 交接 + kernel B 动态 extent 区间展开；备选 = kernel B 退化为单核 `T.serial` 标量区间写；
裁决计划含代表 shape（smoke E=4 / prefill E=256 max_tiles=768）、测量指标（首编 + 三表逐位一致 + 反复运行稳定）、
判定阈值（kernel B 动态 extent 编译失败即切备选；调度墙钟 > GEMM 有效时间 5% 才考虑进一步优化）、回写路径
（§1.6.0 roofline / §5.6 / perf-feedback.md）。本单元无「布局轴」自由度（int32 一维索引），但承载形态与
kernel B 展开路径的成本特征含未实测常数，故按三件套给出；该常数已由 **D-3 探针**标定为「可用」。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（单一模式）。本单元为**两 kernel 纯 Vector 调度单元**：kernel A（`alloc_ub` 整型索引链 +
`T.Parallel`/`T.cumsum`）与 kernel B（`T.Parallel` 广播 + 动态 extent `T.copy` + 标量 GM 控制流），
**不含 `T.gemm`、不含跨 scope 交接**，故 Developer 形态完全可用。
用户指定 `programming_mode=developer`，与本结论一致。**无需 Expert 回退**（v0 的 Expert 回退随内联形态一并撤回；
v1 的单 scope「UB scan + 标量消费」已因 CG-2026-0027 撤回，v2 改两 kernel）。

### 2.2 选型理由

- 本单元数据通路为**整数 Vector 索引/区间展开**（kernel A `T.Parallel(E)`+`T.cumsum`；kernel B `T.Parallel(mte)`+动态
  `T.copy`），无浮点、无矩阵——按 AGENTS.md 的 Developer-mode MixCV 触发规则（Cube `T.gemm` + 至少一个 v-prefix op），
  **本单元自身不满足触发判据**（无 `T.gemm`），归为「整数 Vector 调度单元」，走 Developer。
- **规避三条工具链阻断**：Developer persistent + `T.gemm` + 整型索引段共存即崩（CG-2026-0010）——本单元**无 `T.gemm`**，
  不触发；跨引擎同 block 交接不可编译（CG-2026-0026）——本单元**无跨引擎交接**；单 scope 内标量消费向量产出不可同步
  （CG-2026-0027）——本单元用**两 kernel GM 交接**隔离，不触发。
- Developer 用 `alloc_ub` + `T.Parallel`/`T.cumsum`/`T.copy` 表达，实现负担低。

### 2.3 模式影响

| 维度 | Developer（选定） |
|------|------------------|
| 内存分配 | kernel A：`T.alloc_ub`（索引表）；kernel B：`T.alloc_ub`（广播缓冲 + 哨兵缓冲）；GM 中间量 `cum_gm`；GM 输出表（`out_idx=[1,2,3]`） |
| 扫描原语 | `T.cumsum`（kernel A，单核一次）/ `T.Parallel`（kernel A/B） |
| 跨 scope 交接 | **无**（两 kernel 各自单 Vector scope；`cum` 经 **GM `cum_gm`** + kernel 间同 stream 顺序，无 in-block 交接） |
| 同步 | 各自 scope 内编译器自动同步；**不依赖 auto-sync 跨「向量生产 → 标量消费」**（该形态已弃用，CG-2026-0027）；kernel 间由 stream 顺序保证 |

> 说明：胞弟 `_moe_grouped_gemm_kernel`（消费本单元产出的 GEMM 核）为 **Expert Cube-only** 形态，
> 属其自身单元（§0.6 R1 / §2.1 于胞弟设计），不在本单元范围。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的优化后公式。逐步骤映射到 §1.6.2 的向量化结论（索引段为**整数** `T.Parallel`/`T.cumsum`，
> 无 v-prefix 浮点算子）。A* = kernel A（scan）；B* = kernel B（expand）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| A1 | `s_e`（每专家行数） | GM→UB 向量搬运 `true_sizes` |
| A2 | `cnt_e = (s_e + block_m - 1) // block_m` | 整数 ceil（源形式；**不用 `>>`**，CG-2026-0028a）；`T.Parallel(E)` |
| A3 | `incl = cumsum(cnt)`；`cum_0=0, cum_{e+1}=incl_e`；`total=cum_E` | 硬件 scan + 1-lane 移位（E1），`cum` 落 GM |
| B1 | `t ∈ [cum_e, cum_{e+1})` → `(e, (t-cum_e)·block_m)` | 逐专家区间展开（E2） |
| B2 | 写 `tile_expert_ids[t]=e / -1`、`tile_row_offsets[t]=row / 0`、`total_tiles[0]=cum_E` | 3 张 GM 表输出（死 tile 哨兵 E3） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| A1 | `true_sizes → ub` | `T.copy(true_sizes[0:E], ub_sizes[0,0:E])` | GM→UB 切片搬运 | Developer |
| A2 | `cnt_e` | `T.Parallel(E)` 内整型算式 | 循环体 `ub_cnt[0,e]=(ub_sizes[0,e]+(block_m-1))//block_m` | Developer |
| A3 | 含前缀 scan | `T.cumsum(ub_cnt, ub_incl, dim=1, reverse=False)` | 2D `(1,E)` int32；**唯一一次 scan** | Developer |
| A4 | 排他移位 + 落 GM | `ub_cum[0,0]=0`；`T.copy(ub_incl[0,0:E], ub_cum[0,1:E+1])`；`T.copy(ub_cum[0,0:E+1], cum_gm[0:E+1])` | 1-lane 移位切片 + UB→GM | Developer |
| B1 | 读区间端点 | `c_lo = cum_gm[e]`；`c_hi = cum_gm[e+1]` | 标量 **GM** 读（跨 kernel；非 UB 向量产出） | Developer |
| B2 | 广播区间值 | `for tt in T.Parallel(mte): ub_e[0,tt]=e; ub_r[0,tt]=tt*block_m` | 标量广播到 UB | Developer |
| B3 | 区间写表 | `T.copy(ub_e[0,0:c_hi-c_lo], tile_expert_ids[c_lo:c_hi])`；`T.copy(ub_r[0,0:c_hi-c_lo], tile_row_offsets[c_lo:c_hi])` | 动态 extent 切片 DMA | Developer |
| B4 | 死 tile 哨兵 | `T.Parallel(max_tiles)` 填 `(-1,0)` + `T.copy(..., tile_expert_ids[total:max_tiles])` | 动态 extent 区间写（core 0） | Developer |
| B5 | total | `if core_id==0: total_tiles[0]=cum_gm[E]` | 标量 GM 读 + 写 | Developer |

> 说明：索引段的逐元素计算为**整型**（`cnt`、前缀 scan、区间展开广播），用 `T.Parallel` + `T.cumsum` + `T.copy`
> 承载；本单元无浮点逐元素运算，故不涉及 `vadd/vmul/vexp/vcast/vbrc` 等 v-prefix 算术 API。

### 3.3 计算伪代码

> `T.Tensor` 第二位置参数直接传 dtype。本单元为**两 kernel GM 交接**：kernel A 产 `cum_gm`；kernel B 产 3 张 GM 表。
> `nk_B`/`epc`/`mte`/`max_tiles`/`E`/`block_m` 均为编译期常量。伪代码与设备实测探针
> `history_version/design_probe_two_kernel.py` 同构（探针已位精确通过）。

```python
# 编译期常量：E, max_tiles, block_m,
#   P = NPUUtils.get().get_aicore_num() * 2                # 纯 Vector 物理核数（§5.5 实查 = 40）
#   nk_B = min(E, P); epc = ceildiv(E, nk_B)               # kernel B 分核（静态边界）
#   mte = max_tiles - E + 1                                 # 每专家 tile 数上界（>= ceil(numel/block_m)）

@tilelang.jit(out_idx=[1], target="npuir")                  # kernel A 只输出 cum_gm
def _sched_scan_kernel(num_experts, max_tiles, block_m):
    @T.prim_func
    def _scan_main(true_sizes: T.Tensor([E], "int32"),
                   cum_gm: T.Tensor([E + 1], "int32")):
        with T.Kernel(1, is_npu=True) as (_cid, _sub):       # 单核一次
            ub_sizes = T.alloc_ub((1, E), "int32")
            ub_cnt   = T.alloc_ub((1, E), "int32")
            ub_incl  = T.alloc_ub((1, E), "int32")
            ub_cum   = T.alloc_ub((1, E + 1), "int32")
            T.copy(true_sizes[0:E], ub_sizes[0, 0:E])
            for e in T.Parallel(E):
                ub_cnt[0, e] = (ub_sizes[0, e] + (block_m - 1)) // block_m   # 源形 ceil，不用 >>
            T.cumsum(ub_cnt, ub_incl, dim=1, reverse=False)                  # E1: 唯一 scan（int32 (1,E)）
            ub_cum[0, 0] = 0                                                 # E1: 排他 = 含前缀 1-lane 移位
            T.copy(ub_incl[0, 0:E], ub_cum[0, 1:E + 1])
            T.copy(ub_cum[0, 0:E + 1], cum_gm[0:E + 1])                     # 落 GM（kernel 边界）
    return _scan_main

@tilelang.jit(out_idx=[1, 2, 3], target="npuir")            # kernel B 输出 3 张表
def _sched_expand_kernel(num_experts, max_tiles, block_m):
    @T.prim_func
    def _expand_main(cum_gm: T.Tensor([E + 1], "int32"),
                     tile_expert_ids: T.Tensor([max_tiles], "int32"),
                     tile_row_offsets: T.Tensor([max_tiles], "int32"),
                     total_tiles: T.Tensor([1], "int32")):
        with T.Kernel(nk_B, is_npu=True) as (core_id, _sub):
            ub_e   = T.alloc_ub((1, mte), "int32")
            ub_r   = T.alloc_ub((1, mte), "int32")
            ub_neg = T.alloc_ub((1, max_tiles), "int32")
            ub_zro = T.alloc_ub((1, max_tiles), "int32")
            if core_id == 0:
                total_tiles[0] = cum_gm[E]                                   # 设备侧 total
            # E2: 逐专家区间展开（标量 GM 读控制动态 extent DMA）
            for ee in T.serial(epc):                                         # 静态边界
                e = core_id * epc + ee
                if e < E:
                    c_lo = cum_gm[e]                                         # 标量 GM 读（跨 kernel，非 UB 产出）
                    c_hi = cum_gm[e + 1]
                    for tt in T.Parallel(mte):
                        ub_e[0, tt] = e
                        ub_r[0, tt] = tt * block_m
                    if c_lo < c_hi:
                        T.copy(ub_e[0, 0 : c_hi - c_lo], tile_expert_ids[c_lo:c_hi])
                        T.copy(ub_r[0, 0 : c_hi - c_lo], tile_row_offsets[c_lo:c_hi])
            # E3: 死 tile 区间 [total, max_tiles) 写哨兵（单核；与有效区间不相交，无写竞态）
            if core_id == 0:
                tot = cum_gm[E]
                if tot < max_tiles:
                    for i in T.Parallel(max_tiles):
                        ub_neg[0, i] = -1
                        ub_zro[0, i] = 0
                    T.copy(ub_neg[0, 0 : max_tiles - tot], tile_expert_ids[tot:max_tiles])
                    T.copy(ub_zro[0, 0 : max_tiles - tot], tile_row_offsets[tot:max_tiles])
    return _expand_main

# 公共入口（供集成/Stage 5 调用）：链式跑两 kernel，返回 3 张表
def _tile_scheduler_kernel(num_experts, max_tiles, block_m):
    scan = _sched_scan_kernel(num_experts, max_tiles, block_m)
    expand = _sched_expand_kernel(num_experts, max_tiles, block_m)
    def run(true_sizes):
        cum = scan(true_sizes)
        return expand(cum)
    return run
```

> **注 1（规避 CG-2026-0027/0028）**：kernel B 全程**不读任何由本 scope 向量 op 写入 UB 的值**——`c_lo/c_hi` 是
> **GM** 读（由 kernel A 写、kernel 边界隔离），区间写全是向量 op；无标量二分 / `alloc_local` / `>>` / 标量 `if_then_else`。
> **注 2（输出承接源语义）**：`tile_row_offsets` 已是**行偏移**（乘 `block_m`），死 tile 哨兵 `(-1, 0)`，
> `total_tiles ≤ max_tiles`——与源 L112–L116 逐元素一致，供胞弟 GEMM 核读表（对接契约 §0.6 R1）。
> **注 3（scan int32）**：`T.cumsum` 的 int32/`(1,E)` 路径**设备实测可用**（`repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`
> ASSERT D `[1,3,6,10]`、D-3 探针），闭合 v1 §9.4 R-1/R-4；int32 scan 以 scalar instruction lower（低效但 E≤256 可忽略）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.cumsum(src, dst, dim=1, reverse=False)`（int32, `(1,E)`） | `docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1（int32 √）+ `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` L186（在用）；**设备实测**（`repro/TRAP-VECTOR-SCALAR-DATAFLOW.py` ASSERT D、D-3 探针） | 已验证（int32/`(1,E)` 实机通过；编译器 warning「scalar instruction low efficiency」） |
| `T.Parallel` 1D 整型向量化（kernel A `T.Parallel(E)`；kernel B `T.Parallel(mte)` 广播） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.2.1（支持 `+ - * /`、比较、`T.if_then_else`）；§2.3「下标不能做变换」「循环体须用 alloc 的 UB buffer」 | 已验证（广播 `ub_r[0,tt]=tt*block_m` 索引未变换） |
| `T.copy` 切片（GM→UB / UB→GM / 动态 extent） | `docs/Tilelang.language/内存操作/T.copy.md`；动态 extent 参照胞弟 GEMM `T.copy(A[m_start:m_start+tile_m, ...], ...)` 与 `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 已验证（D-3 探针实测动态 extent 区间写通过） |
| `T.Kernel(n, is_npu=True)`（1D 核；kernel A grid=1，kernel B 固定核） | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；`examples/deepseek_v32/fp8_lighting_indexer.py` L53-57 | 已验证 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | 已验证 |
| `@tilelang.jit(out_idx=..., target="npuir")`（设备侧输出分配） | 源 L74 `@tilelang.jit(out_idx=[1,2,3])`；TileLang jit 设备侧 tensor 分配 | 已验证（与源一致） |
| ~~int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else`~~ | — | **不可用**（CG-2026-0028a/b/c，codegen SIGSEGV）；v2 已全部回避 |
| `T.gather(src,dst,indices,size)` | `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1 标 `indices=tensor`(int32)；`inspect.signature` 注解 `Union[list, tuple]` | **文档/签名冲突**（未文档化假设，§9 R-5）；本设计不依赖 |
| `T.vbrc`（可选广播替代） | `docs/Tilelang.language/shape操作/T.vbrc.md`；kernel B 主选用 `T.Parallel` 广播（已实测） | 已验证（备选） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（两 kernel 均为 1D `T.Kernel`） | 一维核 + 核内任务计算 |
| GEMM M/N 需为 block 整数倍 | 否（本单元无矩阵；尾块 extent 由宿主 GEMM 处理） | — |
| L0C 容量（128KB） | 否（本单元无 L0C） | — |
| 物理核数限制 | 是（kernel B 固定核数分核） | `nk_B = min(E, P) = min(E, 40)`（纯 Vector 实查 ×2）；核内 `T.serial(epc)`（§5.5）；kernel A 单核 |
| host 侧禁止改动输入 | 是（源宿主 `forward` 有 `F.pad(A)`，不属本单元但需说明） | 本单元无 host 预处理；宿主 padding 由胞弟 §0.5 舍弃，改动态 extent |
| 一维核超发被串行调度 | 是（源 `ceildiv(max_tiles,threads)` 超发） | kernel B 固定核数 + 核内串行（§0.6 R4、§5.5） |
| `T.Parallel` 下标禁变换 | 是 | kernel A 用切片 `T.copy` 完成 1-lane 移位；kernel B 广播 `ub_r[0,tt]=tt*block_m`（索引未变换） |
| 跨引擎同 block 交接不可编译 | 是（v0 内联形态的阻断点） | **本单元无跨引擎交接**；`cum` 经 GM + kernel 间同 stream 顺序（CG-2026-0026） |
| 单 scope 内标量消费向量产出不可同步（CG-2026-0027） | 是（v1 主选形态的阻断点） | **v2 两 kernel GM 交接**：标量只读 GM（`cum_gm`），不读本 scope 向量产出的 UB |
| int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else`（CG-2026-0028） | 是（v1 §3.3 使用过） | v2 全部回避：ceil 用 `//`；无 `alloc_local`；无标量 `T.if_then_else`（死 tile 用独立区间写） |

#### 3.5.2 参考实现差异说明

| 差异项 | 源实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|--------------|---------------------|----------|
| 承载方式 | 独立调度 kernel 写 3 张 GM 表 | **两 kernel GM 交接**：kernel A（Vector）产 `cum_gm`；kernel B（Vector）产 3 张 GM 表 | §0.6 R1（v0 内联已撤回；v1 单 scope 已因 CG-2026-0027 撤回） |
| 前缀扫描 | thread0 串行标量扫描（每 block 冗余重扫） | kernel A 硬件 `T.cumsum` + 1-lane 移位（单核一次） | §0.6 R2 |
| 查找形态 | 每 thread 一个 M-tile 独立二分 | kernel B 逐专家区间展开（`[cum_e,cum_{e+1})` 一次填） | §0.6 R3 |
| 分核 | `ceildiv(max_tiles,threads)` 超发 block | kernel B 固定物理核数（纯 Vector → 40）+ 核内 `T.serial(epc)`；kernel A 单核 | §0.6 R4 |
| 设备侧输出 | `out_idx=[1,2,3]` 分配 GM 张量 | 保留：kernel B `out_idx=[1,2,3]` | §0.5 O6 |
| 标量/向量数据流 | thread0 标量扫描 + thread 并行（同 kernel 内） | **kernel 边界隔离**：kernel A 纯向量；kernel B 标量仅读跨 kernel GM | §0.5 R6（CG-2026-0027） |
| `total_tiles` 回落 | 始终留设备侧（零 CPU 同步） | 承接：kernel B `core_id==0` 写 `total_tiles[0]=cum_gm[E]` | §0.6 R1 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/autotune/anneal/moe_grouped_gemm_anneal.py` | **极高（本单元直接先例）** | `tile_scheduler_kernel_npu`（L16-50）：NPU 版独立调度器——`T.Kernel(max_tiles)` + fragment `s_cum` + `T.serial(E)` 串行扫描 + 每 tile 二分 + 写 GM 表。**注意**：它用 `T.Kernel(max_tiles)` **超发 max_tiles 个核**，是本设计 §0.6 R4 要修正的反例；其 `supply_prog`（L124-143）是本单元 golden 的纯 Python 参照 |
| `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md` | **极高（消费本单元产出的胞弟，含 Stage 3 实测）** | §0.6 R1 / §4.1 定义两张 kernel 的对接契约（3 张 GM int32 表 + 同 stream 顺序 launch + 无 in-block 同步）；其 §1.6.0 记载内联形态被 CG-2026-0010/0026 双阻断的实测 |
| `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 高（scan 原语） | `T.cumsum(..., dim=1, reverse=False)` 的 NPU 用法（L186/L247）；`T.serial(num_local_tasks)` 静态核内串行（L173/L225） |
| `examples/deepseek_v32/fp8_lighting_indexer.py` | 高（分核形态） | 固定核数 + 核内 `T.serial(ceildiv(...))` + 单比较守卫（L53-57）；`T.Kernel(nums_kernel, is_npu=True)` |

---
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `true_sizes` | `(num_experts,)` | int32 | 每专家真实 token 数 $s_e$ |

> `true_offsets`（`(num_experts,)` int32，每专家在 A 中的紧致起始行）为**胞弟 GEMM 核**的输入，
> 与源数据流一致，不属本调度单元。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `tile_expert_ids` | `(max_tiles,)` | int32 | 每 M-tile 的专家号（死 tile = -1）；`out_idx=[1]`，设备侧分配 |
| `tile_row_offsets` | `(max_tiles,)` | int32 | M-tile 在专家内的行偏移（已乘 `block_m`）；`out_idx=[2]`，设备侧分配 |
| `total_tiles` | `(1,)` | int32 | 有效 M-tile 总数（设备侧，不回落 CPU）；`out_idx=[3]`，设备侧分配 |

> **对接契约（本单元产出 / 胞弟消费，详见 §0.6 R1）**：① `max_tiles = numel // block_m + E`、`block_m` 为共享
> 编译期常量；② 全 int32；③ `tile_row_offsets` 单位是**行**（已乘 `block_m`）；④ 死 tile 哨兵 `(-1,0)`；
> ⑤ `total_tiles ≤ max_tiles`；⑥ **kernel A → kernel B → 胞弟 GEMM** 同 stream 顺序 launch（无 in-block 同步、无 CPU 同步）。
> 三张表由 **kernel B** 产出（`out_idx=[1,2,3]`）；kernel A 另产内部 `cum_gm`（`out_idx=[1]`，不对外）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 所在 kernel / 用途 |
|-----------|-------|-------|----------|--------------------|
| `ub_sizes` | `(1, num_experts)` | int32 | UB | A：每专家行数 |
| `ub_cnt` | `(1, num_experts)` | int32 | UB | A：每专家 tile 数 |
| `ub_incl` | `(1, num_experts)` | int32 | UB | A：含前缀 scan 输出 |
| `ub_cum` | `(1, num_experts+1)` | int32 | UB | A：排他前缀（落 GM 前） |
| `cum_gm` | `(num_experts+1,)` | int32 | **GM** | A 输出 / B 输入：排他前缀（kernel 边界交接） |
| `ub_e` | `(1, mte)` | int32 | UB | B：广播专家号 e |
| `ub_r` | `(1, mte)` | int32 | UB | B：广播行偏移 `k·block_m` |
| `ub_neg` / `ub_zro` | `(1, max_tiles)` | int32 | UB | B：死 tile 哨兵缓冲（`-1` / `0`） |
| （3 张输出表）| `(max_tiles,)/(max_tiles,)/(1,)` | int32 | **GM** | B 输出（`out_idx=[1,2,3]`） |

> `mte = max_tiles - E + 1`（每专家 tile 数上界，编译期常量）。

### 4.4 内存搬运路径

```
[kernel A · Vector · grid=1]
GM[true_sizes(E)] --T.copy--> UB[ub_sizes] --T.Parallel(E)--> ub_cnt --T.cumsum + 1-lane 移位--> ub_cum
                                                                                          │ T.copy
                                                                                          ▼
                                                                     GM[cum_gm(E+1) int32] ──（kernel 边界）──┐
                                                                                                             ▼
[kernel B · Vector · nk_B=min(E,40)]
  GM[cum_gm] --标量 GM 读 c_lo/c_hi--> UB[ub_e/ub_r 广播 (e, k·block_m)]
                                                     │ 动态 extent T.copy（区间 [cum_e, cum_{e+1})）
                                                     ▼
                              GM[tile_expert_ids(max_tiles)] / GM[tile_row_offsets(max_tiles)] / GM[total_tiles(1)]
                                                     │ （同 stream 顺序 launch：kernel A → B → GEMM）
                                                     ▼
                        [胞弟 _moe_grouped_gemm_kernel：Expert Cube-only 读表执行]
```

> **层级说明**：本单元为**整数 Vector 调度**，数据路径只有 **GM → UB**（`true_sizes`、`cum_gm`）与 **UB → GM**
> （`cum_gm`、3 张调度表），**无 L1/L0A/L0B/L0C**（无 Cube 矩阵计算）。胞弟 GEMM 的
> `GM[A/B] → L1 → L0A/L0B → L0C → GM[C]` 搬运路径见其 §4.4。

### 4.5 UB 内存预算（Developer，num_experts=256，max_tiles=768，mte=513，int32）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| **kernel A** | | | |
| `ub_sizes` | (1,256) | int32 | 1024 |
| `ub_cnt` | (1,256) | int32 | 1024 |
| `ub_incl` | (1,256) | int32 | 1024 |
| `ub_cum` | (1,257) | int32 | 1028 |
| A 小计 | | | ≈ **4100** |
| **kernel B** | | | |
| `ub_e` | (1,513) | int32 | 2052 |
| `ub_r` | (1,513) | int32 | 2052 |
| `ub_neg` | (1,768) | int32 | 3072 |
| `ub_zro` | (1,768) | int32 | 3072 |
| B 小计 | | | ≈ **10248** |

> 两 kernel 各自独立 UB 预算（不同 kernel），分别为 ≈4.0KB / ≈10.0KB，均 ≪ UB 上限
> （CONST-capacity-910B2C，192KB）。`cum_gm` 与 3 张调度表位于 **GM**，不占 UB。

### 4.6 动态轴定义

无动态轴：`num_experts / max_tiles / block_m / nk_B / epc / mte` 均为编译期常量（JIT 特化）。
运行时变化的仅为 `true_sizes` 的**数据值**（不改变 shape），由前缀 scan 与 kernel B 区间展开/哨兵处理。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")                 # kernel A -> cum_gm
def _sched_scan_kernel(num_experts, max_tiles, block_m): ...

@tilelang.jit(out_idx=[1, 2, 3], target="npuir")           # kernel B -> 3 张表
def _sched_expand_kernel(num_experts, max_tiles, block_m): ...

def _tile_scheduler_kernel(num_experts, max_tiles, block_m):  # 公共入口，链式跑 A→B
    ...
```

> `pass_configs`（D-3 探针同款，保证 UB 单版本、避免 buffer 重排干扰动态 extent 拷贝）：
> `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`、`TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False`。
> 无 `compile_flags`（纯整数，无 fast-math/bf16 需求）。按 `(num_experts, max_tiles, block_m)` 经
> `functools.lru_cache` 特化（与源一致）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**整数 Vector 索引/调度（两 kernel）**（无浮点、无矩阵、无 spatial tiling）。判定依据：全部计算为
`cnt`（整数 ceil）、前缀 scan、区间展开与区间写——落 Vector（AIV）单元。kernel A 单核 scan；kernel B 固定核数展开。
本单元**无 `T.gemm`、无 Cube scope**。

### 5.2 Block 划分

本单元**无空间 block 划分**（无 M/N/K tile）。工作粒度：
- kernel A：1 个 E 轴向量块（`T.Parallel(E)` + `T.cumsum`），单核；
- kernel B：按专家分核（每核 `epc = ceildiv(E, nk_B)` 个专家），每专家一次「`T.Parallel(mte)` 广播 + 动态 extent 区间写」。

### 5.3 约束分析

- **对齐约束**：E=256 为 int32 向量宽度整数倍（尾 lane 0%）；`T.cumsum` 对任意 E 合法。
- **UB 容量**：kernel A ≈4.0KB、kernel B ≈10.0KB，均 ≪ 192KB ✓。
- **非整除**：`max_tiles` 保守上界 → 死 tile 区间 `[total,max_tiles)` 由单核写哨兵；`T.cumsum` 天然支持任意 E。
- **静态边界**：`nk_B`、`epc`、`mte` 均为编译期常量 ✓（core-split-strategy.md §1 要素③要求）。

### 5.4 注意事项（尾块/边界处理）

- **死 M-tile**：`max_tiles = numel//block_m + E` 上界下，$t\ge total\_tiles$ 为死 tile，由 kernel B 单核用
  动态 extent 区间写哨兵 `(-1,0)` 到 `[total, max_tiles)`（与有效区间不相交，无跨核写竞态；源 L114–L116 等价）。
- **空专家**（$s_e=0$）：`cnt=0`，`c_lo==c_hi` → kernel B 跳过该专家；映射唯一，由 E2/E3 机器验证覆盖。
- **block_m 任意**：kernel A 的 `cnt` 用 `//`（源形式），不要求 block_m 为 2 的幂。
- **E 非 2 幂 / E < 核数**：`T.cumsum` 处理尾块；`nk_B = min(E, 40)`，`epc = ceildiv(E, nk_B)`，末尾核 `if e < E` 守卫。
- **kernel 顺序**：kernel A 写 `cum_gm` → kernel B 读 `cum_gm` 写表 → GEMM 核读表，必须**同一 stream 顺序 launch**（§7）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数（实查）**：`NPUUtils.get().get_aicore_num()` → **返回值 = 20**。
  查询代码与实测值：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20`
  （Ascend910B2C，2026-10-14 本机实查）。本单元为**纯 Vector 单元** → 物理核数取实查值翻倍
  **`P = get_aicore_num() * 2 = 40`**（标准 core-split-strategy.md §1②「纯 Vector 算子核数翻倍」）。
- **逻辑核数**：
  - **kernel A**：逻辑核数 = **1**（一次性 E 轴 scan，单核）。
  - **kernel B**：逻辑核数 = **E**（每专家一个工作项；`num_logical = num_experts`）。
    代表 shape：smoke（E=4）→ 4；decode gate-up（E=256）→ 256；prefill gate-up（E=256）→ 256。
    （注：M-tile 数 `max_tiles` 不再是逻辑核数——区间展开按专家分核，不按 tile。）
- **规模判定**：
  - kernel A：逻辑核数 1 ≤ 40 → 结论「**无需适配**」（单核 scan）。
  - kernel B：smoke E=4 ≤ 40 → 「**无需适配**」；decode/prefill E=256 > 40 → **极大规模**（专家数远超物理核数）。
- **分核方案**：
  - kernel A：`T.Kernel(1)`（单核，grid=1）。
  - kernel B：`nk_B = min(E, P) = min(E, 40)`（编译期常量）；`epc = ceildiv(E, nk_B)`（编译期常量，**循环边界静态**）；
    核内 `for ee in T.serial(epc)` **核内串行**遍历本核所辖专家（`e = core_id*epc + ee`）。
    代表值：prefill/decode gate-up `nk_B=40`、`epc = ceildiv(256,40) = 7`；smoke `nk_B=4`、`epc=1`。
  - 负载均衡：每核专家数差 ≤1；各专家区间写入量 = `cnt_e`（偏斜分布下核间工作量不等，但每项极小、总工作量
    O(total_tiles)，可忽略）。
  - **死 tile 区间 `[total,max_tiles)`**：由 core 0 独立写（`if core_id==0`），与各核有效区间不相交。

### 5.6 承载方式取舍（两 kernel GM 交接 vs 内联 vs v1 单 scope）⭐（本单元核心论证）

| 维度 | 内联进 GEMM 核（v0，弃选） | v1 单 scope UB+标量消费（弃选） | **两 kernel GM 交接（主选）** |
|------|---------------------------|-------------------------------|------------------------------|
| kernel launch | 0（与 GEMM 同 launch） | 1 次 | **2 次**（A、B；与 GEMM 同 stream 串行） |
| GM 往返 | 无表；各核重算 | 3 表写 ≈6.1KB | `cum_gm` ≈2KB + 3 表写 ≈6.1KB |
| 同步/数据流 | 需 in-block Vector→Cube 交接（`sync_block_*`） | **单 scope 标量消费向量产出**（不可同步） | **无 in-block 同步**；kernel 边界隔离标量/向量；stream 顺序保证 |
| 工具链可实现性 | ❌ 不可实现：CG-2026-0010 / CG-2026-0026（官方样例复现） | ❌ 不可实现：CG-2026-0027 / CG-2026-0028（Stage 3 实测证伪，同产物多运行不一致） | ✅ **D-3 探针设备实测可编译 + 位精确 + 6 次稳定**（`history_version/design_probe_two_kernel*.py`） |
| 计算冗余 | 调度与 GEMM 同核 | 各核重扫 `true_sizes`×40 | kernel A 单核 scan；kernel B 按专家分核 |
| 结论 | **弃选（工具链不可实现）** | **弃选（工具链不可实现）** | **选定** |

**论证要点（v2）**：v0 的「内联」与 v1 的「单 scope 标量消费」在本工具链**均不可实现**（前者 CG-2026-0010/0026，
后者 CG-2026-0027/0028，Stage 3 实测证伪）。两 kernel GM 交接是**唯一同时满足「无跨引擎交接、无标量消费向量产出、
无 codegen 崩溃」且经设备实测位精确的形态**。多出的 1 次 launch 与 `cum_gm` 往返（≈2KB，L2 命中量级）是
为实现性支付的可接受成本（合计 ≪ 1 ms，相对胞弟 GEMM 的 B 权重访存 15.03 GB ≈11.9 ms 可忽略；精确值由
§1.6.0 实验裁决计划标定）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| kernel A 核级 | 单核网格 | `T.Kernel(1, is_npu=True)` | 一次性 E 轴 scan（§5.5） |
| kernel A 专家索引 | 向量化 | `T.Parallel(num_experts)` + `T.cumsum` | 整数逐元素 + 硬件 scan（§1.6.2#1/#2） |
| kernel B 核级 | 固定物理核数网格 | `T.Kernel(nk_B, is_npu=True)` | 分核策略（§5.5） |
| kernel B 核内专家 | 串行遍历专家 | `T.serial(epc)`（静态边界） | E 专家摊到核内串行（§0.6 R4） |
| kernel B 区间填充 | 向量化广播 | `T.Parallel(mte)` + 动态 extent `T.copy` | 逐专家区间展开（§1.6.2#4/#5） |
| kernel B 哨兵写 | 向量化区间写 | `T.Parallel(max_tiles)` + 动态 extent `T.copy`（core 0） | 死 tile 区间（§1.6.2#5） |
| 元素级 | 无（已消除） | — | 无逐元素标量循环（§1.6.2） |

### 6.2 循环伪代码

见 §3.3（kernel A：`T.Parallel(E)` + `T.cumsum`；kernel B：核内 `T.serial(epc)` 外层、`T.Parallel(mte)` 广播 +
动态 extent `T.copy` 区间写内层）。

### 6.3 流水线优化

- kernel A 为一次性（单核），无循环流水需求。
- kernel B 核内专家循环不流水（专家间无数据依赖，且为纯区间写；动态 extent 拷贝已高效）。
- 跨 kernel：kernel A → kernel B → 胞弟 GEMM 核**同 stream 顺序执行**（顺序由 stream 保证，无需显式同步）。

### 6.4 尾块处理

- 死 M-tile：kernel B 单核写 `(-1,0)` 到 `[total, max_tiles)`（动态 extent 区间写）。
- 空专家：`c_lo==c_hi` → 跳过。
- E 尾核：`if e < E` 守卫。
- 任意 E / block_m：见 §5.4；不使用 host padding、谓词零填充或逐元素守卫。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**kernel 内自动同步 + kernel 间 stream 顺序**（Developer）。
- kernel A / kernel B 各自**单 Vector scope**，scope 内 MTE（`T.copy`）/ Vector 依赖由编译器自动同步；
- **关键约束（CG-2026-0027）**：**不在标量上下文中读取本 scope 内向量 op 写入的 UB 值**——本设计已彻底避免该形态
  （kernel B 的标量只读 **GM** `cum_gm`，其值由 **kernel A** 写入，读写由 kernel 边界隔离）；
- **不使用任何手动同步原语**（不使用 `T.sync_block_set/wait`、不使用 `T.set_flag/wait_flag`、不依赖 `T.pipe_barrier`）。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| kernel A 内 `T.Parallel(E)`/`T.cumsum`/`T.copy` 之间 | 编译器自动同步 | 单 Vector scope 内数据依赖（纯向量，无标量消费） |
| kernel B 内 `T.Parallel(mte)` 广播 → 动态 extent `T.copy` | 编译器自动同步 | 单 Vector scope 内向量→向量依赖 |
| kernel A → kernel B（`cum_gm`） | **kernel 间同 stream 顺序**（无显式同步原语） | A 写 `cum_gm` 完成、B 再读；**标量 GM 读不跨「向量生产 → 标量消费」同 scope 边界** |
| kernel B → 胞弟 GEMM kernel（3 张表） | **kernel 间同 stream 顺序** | B 写表完成、GEMM 核再读；**无 in-block 跨引擎交接** |

> **v1 §7.1 的「auto-sync 足够」断言已证伪并撤回**：Stage 3 实测（CG-2026-0027）表明单 scope 内「向量产出 → 标量消费」
> 无可用同步原语（`T.pipe_barrier` 无效）。v2 用 kernel 边界（GM `cum_gm`）隔离该数据流，不再依赖 auto-sync 跨此边界。
> **不引入 `T.sync_block_*` 的依据**：CG-2026-0026（codegen `syn_instr_mode` 与编译器 `sync_instr_mode` 不匹配，
> 官方 `examples/mixcv/mixcv_mixkernel.py` 复现）。本设计以「中间量/表落 GM + kernel 间顺序」替代同 block 交接，**完全规避**。

### 7.3 pass_configs 配置

```python
pass_configs = {
    NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}
```

> 与 D-3 探针一致：保证 UB 缓冲单版本、避免 buffer 重排干扰 kernel B 的动态 extent 区间拷贝（实测该配置下
> 两 kernel 位精确且多运行稳定）。kernel A/B 均为纯整数 Vector kernel、无 `T.gemm`。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（纯整数映射），**不得复刻** §0.6 的 NPU 算法
> （保证验证独立性；理想实现为「线性分桶」而非「前缀和 + 二分/单调指针」）。

```python
import torch

def golden_tile_scheduler(true_sizes, block_m, max_tiles):
    """§0.1 语义：true_sizes[E] -> (tile_expert_ids, tile_row_offsets, total_tiles)。

    纯整数、与源实现同语义（线性分桶，非前缀和/二分；独立于 NPU 算法）。
    """
    E = int(true_sizes.shape[0])
    sizes = true_sizes.tolist()
    # 每专家 tile 数（整数 ceil，线性展开每 tile 的 (expert,row)）
    tile_expert_ids = []
    tile_row_offsets = []
    for e in range(E):
        s = sizes[e]
        n_tiles = (s + block_m - 1) // block_m if s > 0 else 0
        for k in range(n_tiles):
            tile_expert_ids.append(e)
            tile_row_offsets.append(k * block_m)
    total = len(tile_expert_ids)
    tile_expert_ids += [-1] * (max_tiles - total)   # 死 tile 哨兵
    tile_row_offsets += [0] * (max_tiles - total)
    return (
        torch.tensor(tile_expert_ids, dtype=torch.int32),
        torch.tensor(tile_row_offsets, dtype=torch.int32),
        total,
    )
```

### 8.2 精度标准

| dtype | 标准 |
|-------|------|
| int32 | **逐元素精确一致**（无浮点参与，无 atol/rtol —— 本单元为整数索引，等价性为逐位精确；§1.6.1 采纳项 E1/E2/E3 机器验证全 0 违反） |

> 注：胞弟 GEMM 的输出精度标准（fp16/bf16 atol=rtol=1e-2）属其单元，不在本单元。

### 8.3 L0 门槛测试计划

> **主线（交付形态）= 两 kernel 调度单元（kernel A 产 `cum_gm`，kernel B 产 3 张 GM 表）+ 与 `golden_tile_scheduler`
> 逐元素精确比对**（`total_tiles` 一并核对，int32 逐位相等）。下表全部面向该交付形态。

| 用例 | 参数 | 验证点 |
|------|------|-------|
| L0-1 smoke 均匀 | numel=64,E=4,block_m=64 → max_tiles=5，uniform sizes | `cum_gm` + 三表 + `total_tiles` 与 golden 逐元素相等；死 tile 哨兵 `(-1,0)` 正确 |
| L0-2 smoke 偏斜 | numel=64,E=4,block_m=64，大量 1-token 专家 | 空专家（`c_lo==c_hi`）映射唯一；三表逐位一致 |
| L0-3 空专家混合 | E=8，部分 `s_e=0`（含首/尾专家） | 重复 `cum` 边界、`e(t)` 唯一；哨兵区间与有效段无重叠/无空洞 |
| L0-4 奇偶 shape | E=5（非 2 幂）、任意 sizes | `T.cumsum` 尾块、非整除 E、`nk_B=min(E,40)`；三表逐位一致 |
| L0-5 代表大 shape | E=256，numel=32768，block_m=64 → max_tiles=768 | `nk_B=40`、`epc=7`；无死锁；三表逐位一致（D-3 探针已覆盖） |
| L0-6 多次运行稳定性 | 同 L0-1/L0-5 重复运行 6 次 | `cum` 与三表**稳定**（无 CG-2026-0027 型未同步读），逐次比对一致 |
| L0-7 对接契约 | 抽取 L0-1/L0-5 输出的 dtype/shape/sentinel | 与胞弟 §4.1 契约一致（int32、`tile_row_offsets` 为行偏移、`total_tiles ≤ max_tiles`、哨兵 `(-1,0)`），供 GEMM 核消费 |
| L0-8 两 kernel 顺序 | kernel A→B 同 stream 顺序 | B 读到 A 写入的 `cum_gm`（无旧值/未初始化），三表正确 |

> 注：**D-3 设计期探针已先行验证** L0-1/L0-5 两类（含 E=256 prefill，6 次位精确稳定，见 §9 与 `history_version/`）。

（L1/L2/Boundary 完整分层套件交 `tilelang-op-develop`。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 源独立 kernel 的**超发 grid**（`ceildiv(max_tiles,threads)`）在 NPU 上被串行调度（core-split-strategy.md §1）→
  kernel B 固定物理核数（纯 Vector → 40）+ 核内串行。
- **单 scope「向量产出 → 标量消费」不可同步**（CG-2026-0027）→ v2 两 kernel GM 交接规避（标量只读跨 kernel GM）。
- **int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else` 使 codegen SIGSEGV**（CG-2026-0028）→ v2 全部回避。
- `T.Parallel` 下标禁变换（`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3）→ kernel A 移位用切片 `T.copy`；
  kernel B 广播 `ub_r[0,tt]=tt*block_m`（索引未变换）。
- 跨引擎同 block 交接不可编译（CG-2026-0026）→ 本设计以「中间量/表落 GM + kernel 间顺序」规避。
- `T.cumsum` int32 路径以 scalar instruction lower（编译器 warning "low efficiency"）→ E≤256 单核一次，成本可忽略。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **单 scope 标量消费向量 scan**（CG-2026-0027） | 标量读 `T.cumsum`/`T.Parallel` 写入 UB 的值并消费（分支/算术/下标） | 同编译产物多次运行结果不一致（stale 读） | **两 kernel GM 交接**：标量只读跨 kernel GM（`cum_gm`），不读本 scope UB |
| int32 `>>` / `T.alloc_local`+`T.serial` / 标量 `T.if_then_else`（CG-2026-0028） | v1 §3.3 使用过 | device_codegen SIGSEGV | ceil 用 `//`；循环标量用循环变量/表达式；死 tile 用独立区间写而非标量 select |
| **内联进 GEMM 核不可实现**（CG-2026-0010 / CG-2026-0026） | 若把调度并入含 `T.gemm` 的核 | lowering segfault / sync op 不可编译 | 本设计：两独立 Vector kernel（§0.6 R1） |
| Pipelined 体内条件构造崩溃（CG-2026-0001） | 死 tile 守卫误放入 K 循环内（胞弟消费侧） | 编译 SIGSEGV | 守卫放 `T.serial` 体内、`T.Pipelined` 之外（胞弟侧） |
| 前缀扫描少移一位（用含前缀当排他） | 省略 1-lane 移位 | 映射整体错位（175571 tile 失配） | `cum[0]=0; cum[1:E+1]=incl[0:E]`；§1.6.1 shift guard 可检 |
| `T.Parallel` 内写下标变换 `cum[e+1]=incl[e]` | 违反 §2.3 限制 | 编译/lower 异常 | 改用切片 `T.copy(ub_incl[0,0:E], ub_cum[0,1:E+1])` |
| 动态 extent 切片长度取错（`c_hi-c_lo`） | kernel B 区间写长度错误 | 表错位/越界 | `c_lo=cum_gm[e]`、`c_hi=cum_gm[e+1]`、`cnt=c_hi-c_lo`；空专家 `c_lo==c_hi` 跳过 |
| 死 tile 区间与有效区间重叠写（跨核竞态） | 多核都写 `[total,max_tiles)` | 竞态/错值 | 死 tile 区间仅 core 0 写；有效区间 `⊆[0,total)` 与死区间不相交 |
| 独立形态超出物理核数（源超发） | 用 `T.Kernel(max_tiles)` | 核启动开销剧增 | kernel B 固定核数 + 核内 `T.serial`（§5.5） |
| kernel A→B 异性 stream launch | 破坏同 stream 顺序 | B 读 `cum_gm` 竞态 | 严格同 stream 顺序 launch |

### 9.3 特殊场景处理

- **偏斜分布**：大量 1-token 专家 → 每专家 1 tile；核间工作量不均但每项极小（总 O(total_tiles)）。
- **空专家**：`cnt=0` → `c_lo==c_hi`，kernel B 跳过；不产生 tile。
- **极小 shape**（smoke E=4）：`nk_B=4 < 40`，正确性优先，其余核不启动。
- **全空输入**（numel=0）：`cum` 全 0、`total=0`；kernel B 无有效区间，死 tile 区间 `[0,max_tiles)` 写哨兵。

### 9.4 待确认项（交 Stage 2/3 复核）

- **R-1（已闭合，正向）**：`T.cumsum` 的 **int32、`(1,E)` 首维=1** 路径**设备实测可用**
  （`repro/TRAP-VECTOR-SCALAR-DATAFLOW.py` ASSERT D `[1,3,6,10]`；D-3 探针通过）——v1 的「文档声明、实机未覆盖」
  歧义已解决。注：int32 scan 以 scalar instruction lower（编译器 warning "low efficiency"），E≤256 可忽略。
- **R-2（已实测结论）**：v0「内联进 GEMM 核」（CG-2026-0010/0026）与 v1「单 scope 标量消费」（CG-2026-0027/0028）
  在本工具链**均不可实现**（Stage 3 实测证伪）；v2 据此改为两 kernel GM 交接（§0.6 R1），保留登记以提醒后续算子。
- **R-3**：两 kernel 的 launch + kernel B 动态 extent `T.copy` 发射成本（未在 pattern-library 实测）——按
  「未文档化假设 + 估算依据」记录，由 §1.6.0 **实验裁决计划**标定（代表 shape / 反测方法 / 判定阈值 / 回写路径齐全）。
- **R-4（已闭合）**：`T.cumsum` 对 `(1,E)` 首维=1 的 2D 缓冲——设备实测通过（同上）。
- **R-5**：`T.gather` 的 `indices` 语义存在**文档/签名冲突**（`docs/Tilelang.language/索引与元素操作/T.gather.md`
  §2.1 标 `tensor`(int32)、§2.4 示例运行时填充；`inspect.signature` 注解 `Union[list, tuple]`）——标注「未文档化假设」；
  本设计**不依赖**该点（搜索已由 E2 区间展开消除）。如需可补最小探针。
- **R-6（v2 新增，交 Stage 3 首编确认）**：kernel B 的**动态 extent 区间 `T.copy`（UB→GM，长度 `c_hi-c_lo`）**在
  全 shape 下的编译稳定性——D-3 探针已覆盖 smoke 与 E=256 prefill（位精确），但存在工具链对动态 extent+DMA 的
  flaky SIGSEGV 先例（胞弟 `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-BISHENGIR-FLAKY-SIGSEGV.py`）
  → Stage 3 若命中，切 §1.6.0 备选（单核标量区间写）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/
├── DESIGN.md               # 本设计文档（revision v2）
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1）
├── RETROSPECTIVE.md        # Stage 1 复盘（自进化钩子）
├── repro/                  # Stage 3 实测复现（CG-2026-0027/0028 证据）
└── history_version/        # 历史设计备份（design_v0/v1）+ D-3 设计期探针
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision v2） | 设计文档（§0 迁移 + §1.6 算法调研/优化 + §5 分核/承载方式取舍 + §12 修订说明） |
| `verify_equiv.py` | 已完成 | 等价性验证（E1/E2/E3 全 EQUIV_PASS + shift 负荷 guard；重跑通过） |
| `history_version/design_probe_two_kernel.py` | 已完成（D-3） | 两 kernel 方案小 shape 位精确 + 6 次稳定探针（设备实测） |
| `history_version/design_probe_two_kernel_large.py` | 已完成（D-3） | E=256 prefill 代表 shape 位精确 + 6 次稳定探针（设备实测） |
| 算子实现（Stage 3） | 待实现 | 两 kernel 调度单元（§3.3 伪代码）：kernel A 产 `cum_gm`，kernel B 产 3 张 GM 表 |
| 集成对接（Stage 5） | 待实现 | 与胞弟 `_moe_grouped_gemm_kernel`（Expert Cube 读表）同 stream 顺序 launch |

### 10.3 命名规范

- 项目目录：`examples/moe_grouped_gemm_nopad/`
- 算子目录：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/`
- 规范文件名：`_tile_scheduler_kernel.py`（Stage 3 产文件；含 `_sched_scan_kernel` / `_sched_expand_kernel` 两工厂）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ D-3 可行性探针
2. ⬜ Golden 函数（§8.1）
3. ⬜ kernel A `_sched_scan_kernel` + kernel B `_sched_expand_kernel`（§3.3 伪代码）+ L0 三表逐元素比对（§8.3 主线）
4. ⬜ 与胞弟 GEMM 核同 stream 顺序集成（对接契约 §0.6 R1 / §4.2）

---

## 11. 参考

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/moe_grouped_gemm_nopad.py`（`_tile_scheduler_kernel`）
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py`
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tileops/ops/moe/moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tests/ops/test_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_grouped_gemm_nopad.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`
- 胞弟（消费本单元产出的 GEMM 核，含 Stage 3 实测）：`examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/DESIGN.md`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/verify_equiv.py`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/repro.log`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-SYNCBLOCK-ATTR-MISMATCH.py`
- 同类实现：`testing/autotune/anneal/moe_grouped_gemm_anneal.py`（`tile_scheduler_kernel_npu`）、
  `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`、
  `examples/mixcv/mixcv_mixkernel.py`
- API 文档：`docs/Tilelang.language/排序操作/T.vcumsum.md`、
  `docs/Tilelang.language/索引与元素操作/T.gather.md`、`docs/Tilelang.language/逻辑操作/T.vshr.md`、
  `docs/Tilelang.language/内存操作/T.copy.md`、`docs/Tilelang.language/内存操作/T.alloc_ub.md`、
  `docs/Tilelang.language/编译器提示操作/T.Parallel.md`、`docs/Tilelang.language/编译器提示操作/T.Kernel.md`、
  `docs/Tilelang.language/shape操作/T.vbrc.md`、`docs/Tilelang.language/比较操作/T.min.md`、
  `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`docs/Tilelang.language/同步管道操作/T.sync_block_wait.md`
- 标准：`.agents/skills/_shared/standards/core-split-strategy.md`、
  `.agents/skills/_shared/standards/algorithm-research.md`、
  `.agents/skills/_shared/standards/negative-claim-evidence.md`、
  `.agents/skills/_shared/standards/hardware-cost-model.md`
- pattern-library：`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`、
  `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md`
- 能力缺口登记：`.agents/evolution/capability-gaps.md`（CG-2026-0001/0010/0012/0026 历史；**CG-2026-0027** 单 scope
  标量消费向量产出不可同步、**CG-2026-0028** int32 `>>` / `alloc_local` / 标量 `if_then_else` 等 codegen SIGSEGV）
- 本单元 Stage 3 复现 / D-3 探针：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/history_version/design_probe_two_kernel.py`、
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/history_version/design_probe_two_kernel_large.py`、
  `examples/moe_grouped_gemm_nopad/_moe_grouped_gemm_kernel/repro/TRAP-BISHENGIR-FLAKY-SIGSEGV.py`

---

## 12. 相对上一版（design_v1）的关键调整与教训

> 本版为 Stage 3 报 `[DESIGN_ERROR]` 后的 revision v2。**仅在 v1 基础上修订；未静默丢弃仍成立结论**（见 12.2）。
> 上一版备份：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/history_version/design_v1.md`。

### 12.1 关键调整（相对 v1）

| # | `design_error_summary` 问题 | v1 | v2 | 依据 |
|---|-----------------------------|----|----|------|
| 1 | **标量消费向量 scan 不可实现**（CG-2026-0027）：v1 §3.3/§7.1 把排他前缀 scan 放 UB（`T.cumsum`），再用标量逐 tile 消费（二分/单调指针/死 tile 守卫），依赖 Developer auto-sync | 单 Vector scope，UB scan + 标量消费 | **两 kernel GM 交接**：kernel A（Vector，单核）产 `cum` 落 **GM**；kernel B（Vector，固定核数）读 **GM** 做逐专家区间展开 | CG-2026-0027 + Stage 3 实测（同产物多运行不一致，`id_diff=1`）；D-3 探针位精确 |
| 2 | **一串 codegen SIGSEGV**（CG-2026-0028）：`T.alloc_local+serial` / int32 `>>` / 标量 `T.if_then_else` / 运行期下标 UB 读 / 双 `T.cumsum` | §3.3 使用 `>>`、`alloc_local`、标量分支 | 全部回避：ceil 用 `//`；无 `alloc_local`；死 tile 用独立区间写而非标量 select；**一个 kernel 只用一次 `T.cumsum`**（kernel A） | CG-2026-0028 + `repro/TRAP-VECTOR-SCALAR-DATAFLOW.py` |
| 3 | **向量化回退也被挡**：`T.Parallel` 禁索引变换 `e+1` 与循环变量条件 | §1.6.2 判「有序查找不可向量替代」，保留标量 | **整体消除搜索**：kernel B 逐专家**区间展开**（广播 + 动态 extent `T.copy`），无需搜索也无需向量化查找 | §1.6.2（E2）、`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3 |
| 4 | §1.6.0 调研结论（选定单 scope 标量消费）被推翻 | R4 判 C1「可实现且可编译（纯 Vector 核不触发 CG-0010/0026）」 | R1/R4 **新增 C4（v1 单 scope 标量消费）为弃选**（CG-2026-0027/0028），选定 **两 kernel GM 交接**；R2/R3 口径按两 kernel 重算；roofline 按 2 次 launch 重估 | Stage 3 实测 + D-3 探针 |
| 5 | §1.6.1 采纳项含不可实现/低效项 | M1（`>>`）、M3（单调指针） | **M1 撤回**（`>>` SIGSEGV，CG-2026-0028a）；**M3 撤回**（标量消费 UB scan，CG-2026-0027）；新增 **E2 区间展开**（与 searchsorted 位精确等价）；`verify_equiv.py` 同步重写并重跑 | §1.6.1 E1–E3 结果表 |
| 6 | §5.5 分核对象随形态变化 | `nums_kernel=min(max_tiles,40)` 按 M-tile 分核 | kernel A 单核；kernel B `nk_B=min(E,40)`、`epc=ceildiv(E,nk_B)` 按**专家**分核（静态边界） | §5.5 |
| 7 | §7 同步策略断言错误 | 「auto-sync 足够、不使用任何手动同步」 | **明确不依赖 auto-sync 跨「向量生产 → 标量消费」**；kernel 内 auto-sync + kernel 间 stream 顺序；该数据流由 GM 边界隔离 | CG-2026-0027 |
| 8 | §8.3 L0 对象随形态变化 | 独立单 kernel + 三表比对 | 两 kernel + `cum_gm` + 三表比对；新增 L0-8 两 kernel 顺序；D-3 探针预验 L0-1/L0-5 | 交付形态 = 两 kernel |
| 9 | 可行性证据要求（实验裁决三件套 / 等价证据） | v1 三件套面向 scan 原语 | 三件套改为面向 **kernel B 展开路径**（主选 动态 extent / 备选 单核标量区间写）；**D-3 探针设备实测位精确**（小 shape 3 例 + E=256 prefill 6 次稳定） | `history_version/design_probe_two_kernel*.py` |

### 12.2 保留项（v1/v0 仍成立、本版未静默丢弃）

- **整数逐位精确语义**：全程 int32、等价性逐位精确（§0.1、§1.6.1、§8.2）；`cum` 与三表逐元素值不变。
- **源「零 CPU 同步」目标**：`total_tiles` 全程设备侧承接（§0.5、§0.6 R1）。
- **前缀和算法意图**：kernel A 硬件 `T.cumsum`（含前缀 + 1-lane 移位 = 排他）承接源串行前缀扫描意图（E1）。
- **查找算法意图**：源 O3「免每 tile O(E) 扫描、O(log E) 定位」意图由 E2 逐专家区间展开承接（每 tile 只写一次，
  O(total_tiles) ≤ max_tiles，位精确等价；见 §0.6 R3 承接说明）。
- **UB staging 意图**：kernel A `ub_sizes/ub_cnt/ub_incl/ub_cum` 与 kernel B 广播/哨兵缓冲（§0.5 O1、§4.5）。
- **`T.cumsum` int32 正向结论**：闭合 v1 §9.4 R-1/R-4（int32 与 `(1,E)` 首维=1 设备可用）。
- **分核原则**：固定物理核数 + 核内 `T.serial`（静态边界），否决源超发 grid（§0.6 R4）。
- **已知工具链阻断登记**：CG-2026-0010/0026（内联/跨引擎交接）、CG-2026-0027/0028（标量消费/混用构造）。

### 12.3 为何不会再犯同一错误（可迁移教训）

- **同步充分性必须前置核验**：凡设计中存在「标量/serial 代码读取向量 op 产出的 UB/L1 值（用于分支/select/
  算术/下标）」的数据流，**必须**列为待实测风险并给可行备选（全向量数据流 or 双 kernel GM 交接），**不得默认
  auto-sync**。本单元 v1 的唯一致命依赖恰是该断言（§7.1），且未被登记为风险。
- **「不可向量替代」的结论必须核工具链**：v1 §1.6.2 判「有序查找不可向量替代」是基于复杂度，**未核**该查找的
  标量形态是否可编译；正确做法是先做「标量消费向量结果」最小探针（`repro/TRAP-VECTOR-SCALAR-DATAFLOW.py`），
  通过后再实现，不通过即按 DESIGN_ERROR 上报，不逐构造试错。
- **int32 禁用 `>>`**：一律 `// 2**k`（非负整数逐位等价）。
- **一个 kernel 只用一次 `T.cumsum`/`T.reduce_sum`**，且只对 `(1, E)` 形状；需要两个归约量时用「打包 + 单次 scan」
  或拆 kernel。
- **承载形态/数据流取舍先可行性后经济性**：先 grep `.agents/evolution/capability-gaps.md` 匹配 CG 条目，再读
  消费者单元 Stage 3 状态，最后才比较 launch/GM 往返代价；可行性由工具链硬证据裁定。
- **判定依赖未实测常数时，主选之外必须给出可实现备选 + 判定阈值 + 回写路径**，并优先用 **D-3 探针**把关键
  未实测常数标定为「可用/不可用」（本单元以两个探针直接证明两 kernel 形态可编译可运行）。

---
