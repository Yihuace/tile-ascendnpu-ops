# _make_unpermute_kernel 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/unpermute.py`
> 中的 `_make_unpermute_kernel`（GPU/CUDA target，MoE token **还原**单元：专家连续区的 token 按原顺序 + 路由权重还原）。
> 本单元目标：`target="npuir"`（Ascend NPU，Ascend910B2C，实查 `get_aicore_num()=20`）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_unpermute/_moe_unpermute_fwd_kernels.py`（`_make_unpermute_kernel`）。
> 强参照（同族**搬运单元**，已完成且工具链形态验证）：
> `examples/moe_permute_nopad/_make_gather_kernel_nopad/DESIGN.md`（动态源行 gather = 标量 GM 读驱动 `T.copy` 动态 slice + 两段 DMA + 固定 40 核）。
> 工具链陷阱参照：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（CG-2026-0027/0028，勿混用「向量产出 → 标量消费」）。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> **设计期可行性/性能实测（D-3，本机设备 2026-10-14，Ascend910B2C，aicore 20）**：
> `examples/moe_unpermute/_make_unpermute_kernel/history_version/design_probe_unpermute.py`
> （smoke/tiny/topk1/large-h/ep-mask/scaling × bf16/fp16 × 两种累加变体 **20/20 编译+精度+多运行稳定全过**）、
> `examples/moe_unpermute/_make_unpermute_kernel/history_version/design_probe_unpermute_perf.py`
> （稳态延迟：kimi-k2-prefill 610µs / 866 GB/s、qwen3-30b-prefill 833µs / 272 GB/s、large-hidden-medium 96µs / 691 GB/s）。
> 等价性机器验证（D-1）：`examples/moe_unpermute/_make_unpermute_kernel/verify_equiv.py`（E1 EP-mask 跳过 vs 零权，EQUIV_PASS）。

**本单元的一句话定位**：`_make_unpermute_kernel` 是 MoE token 还原单元（permute 的**逆过程**）——按 `fwd_idx`
（`flat_idx = i*K+k → padded 槽位`）把专家连续区 `mm2_pad[padded_batch_sum, H]` 的 K 行**收集**回每个 token，
乘以路由权重 `topk_weights[i,k]` 后 **沿 K 归约**，写回原 token 顺序的 `output[T, H]`（可选乘 `routed_scaling_factor`）。
它是**逐 token 的加权 gather-reduce**（不是跨 token 的 scatter-add），**FLOPs 非零但很低**（`2·T·K·H`），
带宽受限为主。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **跨 token 累加在数学上不存在 → 无需原子、天然确定性**：输出行 `i` 只依赖 `fwd_idx[i*K .. i*K+K-1]`
   指向的 K 个输入行，**每行只被唯一一个核写一次**（同一 token 的多专家加权和是**核内** K 串行累加，
   不是跨 token/跨核的 scatter-add）。故无需 `T.atomic_add`（CG-2026-0029/0030 的缺口不触发）、无归并顺序不确定性问题；
   累加顺序 = `k=0..K-1`（与源 loop-carried `acc` 一致）。
2. **动态源行 gather：从「向量化索引 load」改为「标量 GM 读 → `T.copy` 动态 slice 起始」**：NPU 无 GM 行 gather
   向量原语（`T.gather` 只按最后一维，`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3；跨步 gather 向量指令缺失
   `TRAP-C10-strided-gather`）→ 每 k 一次标量 GM 读 `fwd_idx[flat]` 驱动 `mm2_pad[raw:raw+1, 0:H]` 的 GM→UB DMA。
   `raw` 读自 **GM**（permute 单元跨 kernel 写入），**不是本 scope 向量产出的 UB 值**（规避 CG-2026-0027）。
3. **超发 grid + thread 级向量化 → 固定物理核数（纯 Vector → 40）+ 核内 `T.serial` token 循环**：
   源 `T.Kernel(num_tokens, threads≈H/8)` 是 GPU thread/SIMD 模型；NPU 无 thread-per-element，整行 H 由一条 DMA /
   一条向量 op 承接；token 轴按 `nk=min(T,40)`、`spc=ceildiv(T,nk)`（**编译期静态边界**）在核内串行。
4. **K 归约保持 online 单累加器（fp32 UB 行）**：源 `alloc_fragment([H],"float32")` 寄存器累加器 → NPU `T.alloc_ub((1,H),"float32")`
   单累加器逐 k 在线累加；**两阶段（先物化 K×H 再归约）在 UB 容量上不可行**（K·H·4 = 229KB > 192KB @H=7168,K=8），
   故 online 是**必需形态**而非可选。EP 掩码（`fwd_idx==-1`）用 `if raw >= 0:` **语句**跳过（源用 clamp+零权，
   数学等价：加 0），规避标量 `T.if_then_else`（CG-2026-0028c）。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = **`_make_unpermute_kernel`**（token 还原 kernel）；同文件的 `MoeUnpermuteKernel`（host 编排 / 接口）
> 由 op 层承接，**不在本单元**；上游 permute 单元（`_make_scan_kernel_nopad` / `_make_gather_kernel_nopad`）
> 是 `fwd_idx` 的产出方（§0.6 对接契约）。

### 0.1 源算子语义（做什么）

**数学语义**（逐 token 加权 gather-reduce；fp32 累加、一次落盘 cast）：

$$
\text{out}[i,:]=\text{scale}\cdot\sum_{k=0}^{K-1}\Big([\,\text{fwd\_idx}[iK+k]\ge 0\,]\cdot\text{topk\_weights}[i,k]\cdot
\text{mm2\_pad}\big[\max(\text{fwd\_idx}[iK+k],0),:\big]\Big),
\qquad i\in[0,T)
$$

- `fwd_idx[iK+k] ∈ [0, padded_batch_sum)`：flat 槽位 $iK+k$ 对应的 padded 槽位（permute 单元产出）；
  **EP 模式 `fwd_idx==-1` 表示非本 rank 专家 → 贡献恒零**（源用 slot 0 安全读取 + 权重置 0，见 §0.3）。
- **规约语义**：沿 K 的**加权和归约**，**累加顺序 = k 升序**（源 `acc` 是 loop-carried fragment，`T.Pipelined` 只重叠
  load、不改变 add 顺序）；`k` 是**每个 token 私有的**归约轴 → **无跨 token / 跨核累加**。
- **dtype 语义**：`mm2_pad` / `output` = fp16 / bf16；`fwd_idx` = int32；`topk_weights` = **float32**；
  **中间累加在 float32**（源 `acc`、`src` 转 fp32 再乘加），最后一次 `Cast(dtype, ...)` 落盘；`scaling` 为 fp32 标量。
  无 dtype 提升链、无中间截断。
- **边界语义**：
  - `padded_batch_sum ≥ T*K`（padding 区间不被索引；`fwd_idx` 值域保证）；
  - `top_k == 1` 时 K 循环退化为单次；`top_k < 2` 时源禁用 pipeline（`T.serial`）；
  - `fwd_idx==-1` 的掩码语义（EP）；`output` 由调用方分配（`out_idx=[]`），可复用 `out=` 缓冲；
  - nan/inf：纯乘加，无稳定化需求（无 exp/sum 类溢出路径）。
- **host 侧语义**：`_make_unpermute_kernel` 本身无 host 预处理（纯 kernel 工厂，`num_tokens`/`top_k`/`hidden_size`/
  `padded_batch_sum`/`dtype`/`scaling` 均编译期特化）。唯一 host 协作在调用方 `MoeUnpermuteKernel.forward`：
  分配 `output`、`out=` 复用校验（device/contiguity/与 mm2_pad 不重叠）、顺序调用 kernel，不属于本单元语义。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（纯 PyTorch 逐 token gather + 加权和），**不得复刻** §0.6 的
NPU 算法（固定核数 / 核内串行 / 标量 GM 读驱动动态 slice）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `mm2_pad` | `(padded_batch_sum, H)` | fp16 / bf16 | 输入：下投影输出（padded 专家连续区） |
| `fwd_idx` | `(T*K,)` = `(numel,)` | int32 | 输入：flat 槽位 → padded 槽位（`-1` = 非本地专家） |
| `topk_weights` | `(T, K)` | float32 | 输入：路由权重 |
| `output` | `(T, H)` | fp16 / bf16 | **输出**：还原后的 token 激活（同 `mm2_pad` dtype） |

源签名：`@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])`（**输出由调用方分配并传入**，
非设备侧 `out_idx` 分配）。源工厂参数：`(num_tokens, top_k, hidden_size, padded_batch_sum, dtype, scaling=1.0)`。
manifest op 级 shape rule：`output.shape == (total_tokens, hidden_size)`
（`examples/TileOPs/tileops/manifest/moe.yaml` 的 `MoeUnpermuteFwdOp`）。

### 0.3 实现算法解读（怎么算）

源 kernel 结构（源 L41–L113）：工厂先按 `VEC=8` 算 `threads = min(256, H // VEC)` 并下取 2 的幂（至少 1）；
`numel = num_tokens * top_k`。kernel 为 **1D `T.Kernel(num_tokens, threads=threads)`（一 block 一 token）**：

- **每 block（token `i`）**：`acc[H]` fp32 fragment 清零；`for k in T.Pipelined(top_k, num_stages=2)`（`top_k<2` 退化 `T.serial`）：
  `flat = i*K+k`；`raw = fwd_idx[flat]`；`safe = (raw>=0) ? raw : 0`；
  `weight = topk_weights[i,k] * ((raw>=0) ? 1 : 0)`；
  `T.copy(mm2_pad[safe, 0:H], src)`（**直接向量化 gather 行 load 进 fragment**）；
  `for j in T.Parallel(H): acc[j] += Cast(f32, src[j]) * weight`。
- **落盘**：`out_frag[j] = Cast(dtype, acc[j] * (scaling==1.0 ? 1 : scaling))`；`T.copy(out_frag, output[i, 0:H])`。

**计算步骤分解**（源码每条计算语句均归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| U1 | `flat = i*K + k`（token `i`、k 串行 0..K-1） | block/循环 id | `flat` | flat 槽位下标 $iK+k$ |
| U2 | `raw = fwd_idx[flat]` | `fwd_idx` | `raw` | $\text{fwd\_idx}[iK+k]$ |
| U3 | `safe = if_then_else(raw>=0, raw, 0)` | `raw` | `safe` | EP 安全读取槽位 $\max(\cdot,0)$ |
| U4 | `weight = topk_weights[i,k] * if_then_else(raw>=0, 1, 0)` | `topk_weights`, `raw` | `weight` | 掩码后的路由权重 $[\cdot\ge0]\cdot w_{i,k}$ |
| U5 | `T.copy(mm2_pad[safe, 0:H], src)`（向量化行 load） | `mm2_pad` | `src[H]` fragment | $\text{mm2\_pad}[\text{safe},:]$ |
| U6 | `for j in T.Parallel(H): acc[j] += Cast(f32, src[j]) * weight` | `src`, `weight`, `acc` | `acc[H]` fp32 | 沿 K 的加权和（fp32 累加） |
| U7 | `out_frag[j] = Cast(dtype, acc[j] * scaling)`（`scaling==1.0` 时省乘） | `acc`, `scaling` | `out_frag[H]` | $\text{scale}\cdot\text{acc}$ + dtype 落盘 cast |
| U8 | `T.copy(out_frag, output[i, 0:H])` | `out_frag` | `output` | $\text{out}[i,:]$ |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[fwd_idx(T*K) int32] ──scalar/vector index load──┐
                                                   ▼
GM[mm2_pad(padded_batch_sum,H)] ──(每 token i、每 k：raw=fwd_idx[i*K+k], safe=clamp)──向量化行 load──▶ fragment[src(H)]
                                                                                                        │ Cast f32 × weight
GM[topk_weights(T,K) f32] ──weight 标量────────────┘                                                    ▼
                                                                                          fragment[acc(H) f32] ──(k 循环累加)──▶ acc
                                                                                                                                │ ×scaling, Cast(dtype)
                                                                                          fragment[out_frag(H) dtype] ──T.copy──▶ GM[output(T,H)]
```

**循环与并行结构**：grid = `num_tokens` 个 block（prefill T=4096 → 4096 block）；每 block：K 串行（2-stage pipeline
重叠 load）；**thread 级并行**（`threads ≈ H/8`，每 thread `ceil(H/threads)` 个元素，128-bit `uint4` load/store）。
**未使用**跨 token 归约 / 跨 block 通信。

**host 侧逻辑**：无 host 预处理。调用方 `forward`：`output` 分配 / `out=` 校验（见 §0.1 host 语义）。
**编译配置**：`@tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])`（无 `pass_configs`）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **一 block 一 token**（grid=T） | K 归约局部化、零跨 block 通信 | 每 token 私有若 K 累加器 | CUDA 廉价 block 启动 | 硬件相关（超发 grid 形态） |
| O2 | **thread 协作 H 维**（`threads≈H/8`，cap 256） | 一行由一线程块铺满 | thread-per-element | CUDA thread/SIMD-lane | 硬件强相关（NPU 无 thread-per-element） |
| O3 | **每 thread 8×bf16 128-bit `uint4` load/store** | 用满内存事务宽度 | 向量化访存 | CUDA 128-bit LDG/STG | 硬件强相关（NPU 由 DMA 承接整行） |
| O4 | **fp32 fragment 累加器 `acc[H]`** | 精度 + 寄存器驻留 | 寄存器累加 | 大寄存器堆 | 硬件强相关（NPU 无寄存器 fragment） |
| O5 | **K 循环 software pipeline（`num_stages=2`）** | 重叠散列 load 与累加（latency-bound） | 双 buffer 预取 | 异步拷贝 + 双缓冲 | 硬件相关（NPU 有 MTE 等价，但条件受限） |
| O6 | **EP 掩码：clamp 到 slot 0 + 权重置零** | `-1` 槽位零贡献且免分支 | `if_then_else` 选择安全槽位/权重 | 无（纯算法/索引语义） | 算法层可移植（形态需换） |
| O7 | **`scaling` 折叠到落盘前一次向量乘** | 消额外 pass | 单次 `acc*scale` | 无（纯算法层） | 可移植 |
| O8 | **`out_idx=[]` 调用方分配输出** | 复用 `out=` 缓冲（vLLM 式） | 显式传入 dst | 无 | 可移植（由 op/harness 承接） |
| O9 | **直接向量化索引行 load 进 fragment（无 SMEM 中转）** | 省一次中转 | 索引 load + 寄存器 | GPU 统一地址空间 + gather load | 硬件强相关（NPU 无 GM 行 gather 向量原语） |

> 识别不出来 ≠ 不存在。O1–O9 为源码显式优化（源码注释与结构直读），逐项在 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（逐 token 加权 gather + K 归约、fp32 累加、一次 cast、EP `-1` 掩码、`scaling`、int32 索引） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| O6 EP 掩码（clamp slot 0 + 权重置零） | 算法 | 无 | 有 | **等价替换** | 语义保留；实现改为 `if raw >= 0:` **语句**跳过该 k（= 加 0，数学等价），或 `T.vmin/T.vmax` 掩码。**禁止**标量 `T.if_then_else`（CG-2026-0028c 标量 select → SIGSEGV）。依据：D-1 `verify_equiv.py` E1（跳过 vs 零权 EQUIV_PASS）+ D-3 探针 ep-mask 用例位精确 |
| O7 `scaling` 折叠末次向量乘 | 算法 | 无 | 有 | **保留** | 末次 `T.vmul(ub_acc, scaling, ub_acc)`（标量 src2，`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2「标量广播」+ §2.3-4「仅 src2 可为标量」）；D-3 探针 scaling 用例通过 |
| O4 fp32 累加器（寄存器 fragment） | 优化 | 大寄存器堆 | 有（UB） | **等价替换** | `T.alloc_ub((1,H),"float32")` 单累加器；UB 192KB 内（§4.5） |
| O1 一 block 一 token（grid=T） | 优化 | CUDA 廉价 block 启动 | 无（一维核超发被串行调度） | **重新设计** | 固定物理核数（纯 Vector → 40）+ 核内 `T.serial` token 循环；承接「K 归约局部化、零跨 block 通信」意图（§0.6 R3） |
| O2 thread 协作 H 维 | 优化 | CUDA thread/SIMD-lane | 无 | **舍弃** | NPU 无 thread-per-element；H 由**整行 DMA / 单条向量 op** 承接（`TRAP-threads-kwarg-noop`：npuir 下 `threads=` 无效） |
| O3 每 thread 128-bit 向量化 load/store | 优化 | CUDA 128-bit LDG/STG | 无（形态） | **重新设计** | 整行 H 一条 `T.copy` DMA（MTE2 自动向量化，元素轴连续、32B 对齐）；向量算术由整行向量 op 承接（§0.6 R4） |
| O5 K 循环 software pipeline（`num_stages=2`） | 优化 | 异步拷贝 + 双缓冲 | 有（MTE），但**受条件限制** | **重新设计** | 保留「重叠 load 与累加」意图 → 核内 `T.serial(K)` 顺序累加（**不得**用 `T.Pipelined`：其体内运行时 `if` 触发 Developer codegen SIGSEGV，**CG-2026-0001(a)**；本单元每 k 有 `if raw>=0`）；重叠留待 Stage 4（当前 auto-multi-buffer 关闭，见 §7.3 风险 R-2） |
| O9 直接向量化索引行 load 进 fragment | 优化 | GPU 统一地址空间 + gather load | 无 | **重新设计** | 拆为 **标量 GM 读 `raw` → `T.copy(mm2_pad[raw:raw+1, 0:H], ub_src)`（动态源行 GM→UB）**；NPU 无 GM 行 gather 向量原语（`T.gather` 仅尾维 `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3 + `TRAP-C10-strided-gather`）；D-3 探针实测可编译 + 精度通过（§0.6 R2） |
| O8 `out_idx=[]` 调用方分配输出 | 优化 | TileLang jit 特性 | 有 | **保留** | 保持 `out_idx=[]`、`output` 由 op/harness 分配/复用（与源一致） |
| 「跨 token scatter-add + 原子」（**非源、本单元新引入候选**） | 结构 | 无 | **无可用原语且语义不需要** | **舍弃** | 本单元输出行 `i` 只依赖自身 K 个槽位、**每行唯一写者**，无需跨 token 累加；且 `T.atomic_add` 标量载荷不可用（**CG-2026-0029**）、无 fetch-add（**CG-2026-0030**）。**故「无原子」是数学结论而非实现偏好** |
| 「host 侧 gather+reduce」 | 结构 | 无 | 有（CPU） | **舍弃** | 需 `mm2_pad` 全量设备→host→设备往返 + CPU 同步，违反源「零 CPU 同步 / 设备侧数据流」意图，带宽数量级劣化 |

> 每项处置依据见「NPU 对应方案 / 依据」列；**语义层不可妥协**。
> **判定输入含 Phase R 调研结论（§1.6.0）**：调研在「可实现族」内择优——固定核数 + 核内串行 + 标量 GM 读驱动动态行
> （主选，D-3 探针位精确/稳定）胜出；两阶段物化（UB 容量否决）、跨 token scatter-add（无原语 + 不需要）、GEMM 表述
> （FLOPs 爆炸）、host 侧（CPU 同步）被否决（可行性前置于经济性）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计 / 等价替换」的条目逐项给出新算法，并对照 Phase R 调研结论（§1.6.0）。
调研选定算法族见 §1.6.0（结论：**单独立 Vector kernel，固定物理核数（纯 Vector → 40）+ 核内 `T.serial` token 循环；
每 token 内 K 串行 online 累加；每 k「标量 GM 读 `fwd_idx` → 动态源行 GM→UB → vcast fp32 → vmul 标量权重 → vadd 入
fp32 UB 行累加器」；末次 scaling + cast → UB→GM 整行写出**）。

**重设计项 R1：数据通路——从「向量化索引行 load 进 fragment + 寄存器累加」改为「UB 中转 + 标量 GM 读驱动动态 slice」**

- **源方案**：`T.copy(mm2_pad[safe, 0:H], src)` 直接以 `fwd_idx` 行号索引 load 进寄存器 fragment；`acc` 为寄存器累加器。
- **NPU 新算法**：每 k：`raw = fwd_idx[flat]`（**标量 GM 读**）→（若 `raw>=0`）
  `T.copy(mm2_pad[raw:raw+1, 0:H], ub_src[0:1, 0:H])`（GM→UB，动态源行）→ `T.vcast(ub_src, ub_f32, round_mode="rint")`
  → `T.vmul(ub_f32, w, ub_f32)`（标量 w，in-place）→ `T.vadd(ub_acc, ub_f32, ub_acc)`（in-place）。
- **语义保持论证**：动态 slice `[raw:raw+1, 0:H]` 请求的正是第 `raw` 行；`vcast` 是源 `Cast(f32, src[j])` 的向量形态；
  `vmul`+`vadd` 是源 `acc[j] + Cast(f32,src[j])*weight` 的向量形态（fp32、k 升序累加）。D-3 探针在 bf16/fp16 ×
  {tiny/topk1/large-h/ep-mask/scaling} 下**全部通过容差 + 多运行稳定**（20/20）。边界：`raw<0` 由 `if` 跳过（≡ 加 0）。
- **与源优化意图的承接**（§0.4 逐项，**不得静默丢弃**）：O9（索引行 load）→ **形态替换、意图承接**（无 NPU 行 gather
  向量原语，标量 GM 读驱动动态 slice）；O4（fp32 累加器）→ 等价替换为 UB 行累加器。

**重设计项 R2：EP 掩码——从「clamp slot 0 + 权重置零（标量 if_then_else）」改为「`if raw >= 0:` 语句跳过」**

- **源方案**：`safe = if_then_else(raw>=0, raw, 0)`；`weight = topk_weights * if_then_else(raw>=0,1,0)`；恒执行 load，靠零权抵消。
- **NPU 新算法**：`if raw >= 0:` 语句包裹该 k 的 load + 累加；`raw<0` 时整段跳过。`w = topk_weights[token,k]`（无掩码乘）。
- **语义保持论证**：`raw<0` 时源贡献 = `mm2_pad[0]*w*0 = 0`，跳过 = 不累加 = 加 0 → **逐位等价**（fp32 加 0 精确；
  源本身该 slot 的贡献就是精确 0）。**D-1 `verify_equiv.py` E1**：跳过式 vs 零权式在 fp64 参照下随机张量 + 角点
  （全 `-1` / 交替 `-1` / 无 `-1`）**0 违反 EQUIV_PASS**。**规避 CG-2026-0028c**（标量 `T.if_then_else`/比较表达式在
  `T.Parallel` 外 → SIGSEGV）：用 `if` **语句**（守卫），D-3 探针 ep-mask 用例通过。
- **承接**：O6 意图（`-1` 零贡献）完整承接；额外**少读**被掩码槽位的行（EP 下带宽收益，非语义变化）。

**重设计项 R3：任务划分与分核——从「一 block 一 token + thread 协作 H」改为「固定物理核数 + 核内 `T.serial` token + 整行向量操作」**

- **源方案**：`T.Kernel(num_tokens, threads≈H/8)`——每 token 一 block，block 内 thread 铺满 H。
- **NPU 新算法**：`nk = min(T, P)`（`P` = **纯 Vector 实查物理核数 = `get_aicore_num()*2 = 40`**，§5.5）；
  `spc = ceildiv(T, nk)`（**编译期常量**，核内 `T.serial` 边界静态）；核内 `for t in T.serial(spc): token = core_id*spc + t`；
  `if token < T:` 尾守卫。整行 H 由一条 DMA + 整行向量 op 承接（无逐元素标量）。
- **语义保持论证**：分核只改变「哪个核处理哪些 token」，不改变任何 token 的加权和结果；token 间**完全独立**（无共享
  写、无归约），核间无通信、无归并。边界：`spc` 静态保证循环安全；末核 `if token < T` 守卫。
- **说明**：超发 grid 在 NPU 被**串行调度**（core-split-strategy.md §1），不可取；`threads=` 在 npuir 下无效
  （`TRAP-threads-kwarg-noop`）。

**重设计项 R4：K 归约——保持 online 单累加器，并论证其为必需（非可选）**

- **源方案**：K 循环内 `T.Pipelined(num_stages=2)` 重叠 load；单 `acc[H]` 在线累加。
- **NPU 新算法**：`T.serial(K)`（**不用 `T.Pipelined`**，CG-2026-0001(a) 禁 runtime `if` in Pipelined），
  单 `T.alloc_ub((1,H),"float32")` 在线累加。
- **语义保持论证**：k 升序在线累加 = 源 loop-carried 累加顺序（pipeline 不改 add 序）。**online 是必需形态**：
  两阶段（先物化 K×H 再归约）需 `K·H·4 = 229KB`（fp32）/ `K·H·2 = 114KB`（bf16）+ 累加器 28KB + 输出 14KB，
  fp32 形态 **> 192KB UB**（`CONST-capacity-910B2C`）→ 容量否决（§1.6.0 R4）。源本身即 online，承接其意图。
- **代价**：无 pipeline → load 与累加不重叠（源有 2-stage overlap）。量化：见 §1.6.0 D-2 与 §9 风险 R-2（实测
  kimi-k2-prefill 866 GB/s < MTE2 地板 1.26 TB/s，重叠缺失是候选主因，留 Stage 4）。

**重设计项 R5：无（算法层）**。逐 token 加权 gather-reduce 的**算法本身**（按槽位选源行 + fp32 加权在线归约）在 NPU
上有等价能力，按 §0.5 为「保留」；R1–R4 只改**实现形态**（UB 中转 / 标量 GM 读动态 slice / 固定核数核内串行 / online 必要化），
不换算法族；R2 为 EP 掩码形态替换（数学等价），不换语义。

**重设计项 R6：承载形态——单独立 Vector kernel，`fwd_idx` 经 GM 交接**

- **源方案**：本 kernel 是独立 Phase 2'（还原）kernel；`fwd_idx` 由上游 permute 单元写入 GM。
- **NPU 新算法（保留 + 强化）**：**保留**「permute 写 `fwd_idx` 落 GM → 本单元标量只读 GM」的数据流；并**强化为
  必要条件**：`fwd_idx` 的标量下标必须读 **GM**（跨 kernel 交接），**绝不能**内联进上游 permute scope 后标量消费其
  向量产出（**CG-2026-0027**：同 scope「向量产出 → 标量消费」不可同步）。
- **语义保持论证**：唯一变化是「从 GM 读 `fwd_idx`」，与源 host 顺序调用语义一致（上游先写、本单元后读，同 stream 顺序）。
- **对接契约（与上游 permute / op 层严格对齐）**：
  1. `fwd_idx` dtype **int32**、shape `(T*K,)`；值域 `[0, padded_batch_sum)`，`-1` 表示 EP 非本地专家；
  2. `output` dtype 与 `mm2_pad` 一致（fp16/bf16）；shape `(T, H)`；
  3. `output` 与 `mm2_pad` **不得内存重叠**（源 host 校验）；本单元按契约由 op/harness 保证；
  4. 本单元为**独立 kernel**，同 stream 顺序 launch，无 in-block 同步、无 CPU 同步。

### 0.7 标杆实现

- 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/unpermute.py` 的 `_make_unpermute_kernel`（L33–L113）。
- 源 op（host 编排 / 接口）：`/tmp/opencode/TileOPs-ref/tileops/ops/moe/routed_expert/unpermute.py`（`MoeUnpermuteFwdOp`）。
- Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/moe_unpermute/_moe_unpermute_fwd_kernels.py`（`_make_unpermute_kernel`）。
- op 级参考实现（golden 依据）：`examples/TileOPs/tests/ops/test_moe_unpermute.py` 的 `_ref_moe_unpermute`
  （纯 PyTorch 逐 token gather + 加权累加，含 EP `-1` 变体）与 `examples/TileOPs/tileops/workloads/moe.py`
  （`MoeUnpermuteWorkload.gen_inputs`）。
- 同族搬运单元先例（动态源行 gather / 固定核数）：`examples/moe_permute_nopad/_make_gather_kernel_nopad/DESIGN.md`
  与其实现 `examples/moe_permute_nopad/_make_gather_kernel_nopad/_make_gather_kernel_nopad.py`。
- 同族索引/工具链陷阱先例：`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（CG-2026-0027/0028）。
- 行 gather（串行读索引 + 2-D slice 保 rank）先例（Expert 模式）：
  `examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py`（L161–L167）。
- 源 op / 测试 / bench / manifest：`examples/TileOPs/tests/ops/test_moe_unpermute.py`、
  `examples/TileOPs/benchmarks/ops/bench_moe_unpermute.py`、
  `examples/TileOPs/tileops/manifest/moe.yaml`（`MoeUnpermuteFwdOp`）。
- D-3 探针：`examples/moe_unpermute/_make_unpermute_kernel/history_version/design_probe_unpermute.py`、
  `examples/moe_unpermute/_make_unpermute_kernel/history_version/design_probe_unpermute_perf.py`。
- D-1 等价验证：`examples/moe_unpermute/_make_unpermute_kernel/verify_equiv.py`。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的 NPU 算法）。

---

## 1. 概述

### 1.1 算子名称

`_make_unpermute_kernel`（MoE token **还原**单元：专家连续区 → 原 token 顺序 + 加权 K 归约，`target="npuir"`）。

### 1.2 功能描述

按上游 permute 写入的 `fwd_idx[flat=i*K+k] → padded 槽位`，把 `mm2_pad` 的对应 K 行 gather 回 token `i`，乘
`topk_weights[i,k]` 后沿 K 做 fp32 加权和（`fwd_idx==-1` 的槽位零贡献），乘 `routed_scaling_factor` 后 cast 写回
`output[i,:]`。**迁移决策后的 NPU 形态**（§0.6）：**独立 Vector kernel**，固定物理核数（纯 Vector → 40）+ 核内
`T.serial` token 循环；每 token 内 K 串行**在线**累加（fp32 UB 行累加器）；每 k 标量 GM 读 `fwd_idx` 驱动**动态源行**
`T.copy`；整行 H 由向量 op 承接；末次 scaling + cast 后整行写回 GM。**无跨 token 累加 → 无需原子、天然确定性**。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
\text{out}[i,:]=\text{scale}\cdot\sum_{k=0}^{K-1}\Big([\,fwd\_idx[iK+k]\ge 0\,]\cdot topk\_weights[i,k]\cdot
mm2\_pad\big[\max(fwd\_idx[iK+k],0),:\big]\Big),\quad i\in[0,T)
$$

输入 `mm2_pad (padded_batch_sum, H) fp16/bf16`、`fwd_idx (T*K,) int32`、`topk_weights (T,K) fp32`；
输出 `output (T,H) fp16/bf16`。manifest roofline（`examples/TileOPs/tileops/manifest/moe.yaml`）：
`flops = 2*T*K*H`，`bytes = (T*K*H + T*H)*elem_bytes + T*K*4 + T*K*4`（**只读 T*K 行，padding 区不读**）。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6，与源结构差异已逐条注明）：

1. **分核与任务划分**（§0.6 R3）：`nk = min(T, P)`（`P` = 纯 Vector 实查物理核数 = 40，§5.5）；
   `spc = ceildiv(T, nk)`（编译期常量）；核内 `for t in T.serial(spc): token = core_id*spc + t`；`if token < T` 守卫。
   **与源差异**：源为**一 block 一 token（grid=T）+ thread 协作 H**；v-NPU 为**固定物理核数 + 核内串行 token**。
2. **单 token 的 K 归约**（§0.6 R1/R2/R4）：`T.clear(ub_acc)`；`for k in T.serial(K)`：
   `raw = fwd_idx[token*K+k]`（标量 GM 读）；`if raw >= 0:` → `w = topk_weights[token,k]`；
   `T.copy(mm2_pad[raw:raw+1, 0:H], ub_src)`（GM→UB 动态源行）→ `T.vcast(ub_src, ub_f32, round_mode="rint")`
   → `T.vmul(ub_f32, w, ub_f32)`（标量权重，in-place）→ `T.vadd(ub_acc, ub_f32, ub_acc)`（fp32 在线累加）。
   **与源差异**：源为**向量化索引行 load + 寄存器累加 + 2-stage pipeline**；v-NPU 为**标量 GM 读驱动动态 slice +
   UB 行累加 + `T.serial`（无 pipeline）**。
3. **落盘**（§0.6 R1）：`if scaling != 1.0: T.vmul(ub_acc, scaling, ub_acc)`；`T.vcast(ub_acc, ub_out, round_mode="round")`；
   `T.copy(ub_out, output[token:token+1, 0:H])`。**与源差异**：源 fragment→GM；v-NPU UB→GM 整行 DMA。
4. **承载形态**（§0.6 R6）：本单元是**独立 kernel**，`fwd_idx` 与 `output` 均经 **GM**；`fwd_idx` 标量只读 **GM**
   （跨 kernel 交接），不消费任何本 scope 向量产出的 UB 值给标量代码。

算法族选型依据见 §1.6.0；公式级优化见 §1.6.1；向量化替代见 §1.6.2；布局见 §1.6.3。

### 1.5 数据流图

```
[上游 permute 单元 · 独立 kernel] ──（同 stream 顺序）──┐
  GM[fwd_idx(T*K) int32]                               │
                                                       ▼
[本单元 · Vector · nk = min(T, 40) cores]
  每 token（核内 T.serial）:
    GM[fwd_idx] ──标量 GM 读 raw──┐
                                  ▼
    GM[mm2_pad] ──T.copy 动态源行[raw:raw+1,0:H]──▶ UB[ub_src(1,H) dtype]
                                                        │ T.vcast(→fp32)
    GM[topk_weights] ──标量 w──────┐                    ▼
                                   └──T.vmul 标量────▶ UB[ub_f32(1,H) f32]
                                                        │ T.vadd（K 串行在线累加）
                                                        ▼
                                             UB[ub_acc(1,H) f32] ──×scaling, T.vcast(→dtype)──▶ UB[ub_out(1,H) dtype]
                                                        │ T.copy（动态目的行 [token:token+1,0:H]）
                                                        ▼
                                                  GM[output(T,H) fp16/bf16]
```

> 数据路径只有 **GM → UB → GM**（无 L1 / L0A / L0B / L0C；本单元无矩阵计算、无 Cube）。
> 每核独立处理 `spc` 个 token；核间无通信、无归并。**无跨 token 累加 → 无原子**。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：同一数学语义（**逐 token 加权 gather + 沿 K 归约** `out[i]=Σ_k w_{i,k}·x[idx_{i,k}]`）的**算法族**。
> 输入/源算法（一 block 一 token + thread 协作 H + 寄存器 fp32 累加 + 2-stage pipeline）只是基线候选之一。
> 深度分级：含**索引 gather + 归约**两要素、且含**动态索引 + 形态改造 + 工具链可行性**三处非平凡决策 → **完整调研**
> （R1 候选表 + R3 四口径 + R4 逐候选）。
> **可实现性约束（前置，来自 Phase M1 + 本仓工具链证据）**：
> (a) Developer 模式**禁止 GM→GM**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3-5，编译期报错）→ 任何「一条
>     GM→GM 拷贝」形态必须 UB 中转；
> (b) 同一 scope 内「向量产出 → 标量消费」不可同步（**CG-2026-0027**）→ `fwd_idx` 必须以**标量 GM 读**消费（跨 kernel），
>     不得读本 scope 向量产出的 UB 值；
> (c) 行 gather 无向量原语：`T.gather` 仅尾维（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3）、
>     跨步 gather 向量指令缺失（`TRAP-C10-strided-gather`）→ 任何「单指令向量行 gather」候选不可实现；
> (d) Developer 模式 `T.Pipelined` 体内运行时 `if` → codegen SIGSEGV（**CG-2026-0001(a)**）→ K 循环用 `T.serial`；
> (e) `T.atomic_add` 标量载荷不可用（**CG-2026-0029**）、无 fetch-add（**CG-2026-0030**）→ scatter-add 族不可实现。
> 调研在**可实现族**内择优（可行性前置于经济性）。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | 一 block 一 token（grid=T）+ thread 协作 H + 迭代 K 行 gather + **寄存器 fp32 累加器** + 2-stage pipeline | — | — | ✅ 基线 |
| C1 | **固定核数 + 核内 token 串行（selected）** | `nk=min(T,40)`、`spc=ceildiv(T,nk)`；每 token 内 `T.serial(K)` **online** 累加（fp32 UB 行）；每 k 标量 GM 读驱动动态行 GM→UB → vcast → vmul 标量 → vadd | 数学恒等（fp32、k 升序累加，逐 token 独立） | 保留整行向量化；规避 GM→GM/线程模型/标量消费向量/原子缺失；**无跨 token 累加** | ✅ **选定** |
| C2 | **两阶段（先物化 K×H 再归约）** | 每 token 先把 K 行全部 gather 进 `ub_k(K,H)`，再一次性 reduce | 数学恒等（同 C1，归约顺序可保持 k 升序） | 减少 UB 反复读；但需 `K·H` 驻留 | ❌ 弃选：**UB 容量**——`K·H·4=229KB`(fp32) > 192KB @H=7168,K=8（`CONST-capacity-910B2C`）；bf16 形态 114KB + 累加 28KB + 输出 14KB = 156KB 过紧且无收益 |
| C3 | **源主序 scatter-add（按 padded 槽位 → token）** | 对每 padded 槽位，广播加到其归属 token（需 `fwd_idx` 逆映射） | 数学恒等（同一加权和，改变遍历方向） | 理论上源行只读一次 | ❌ 弃选：**无逆映射**（`fwd_idx` 非单射，多 token 可指向同槽位）；跨 token 写 → 需原子（**CG-2026-0029/0030** 无可用原子）且**非确定** |
| C4 | **选择矩阵 GEMM** `out = P·mm2_pad`（P 为 `(T,padded)` 0/1 稀疏） | Cube 乘 | 数学恒等（P 每行 K 个非零 = 权重） | 复用 Cube | ❌ 弃选：FLOPs `2·T·padded_batch_sum·H`（prefill `≈1.9e12`），纯加权和退化为巨量 MAC |
| C5 | **`T.gather` 尾维 gather（转置法）** | 把 `mm2_pad` 视作尾维为 token 轴，一次 `T.gather` 取全体再转置 | 数学恒等 | 理论上单指令 gather | ❌ 弃选：`T.gather` 的 `src` 须驻 **UB** 且沿尾维；`(H, padded)` 需 ≫192KB；分块引入巨型转置链（`TRAP-C10-strided-gather`） |
| C6 | **host 侧 gather+reduce** | CPU 索引拷贝 + 加权和 | 数学等价 | — | ❌ 弃选：`mm2_pad` 全量设备→host→设备往返 + CPU 同步（违反源零同步意图；带宽数量级劣化） |
| C7 | **逐元素索引（`mm2_pad[idx, h]` 标量）** | 每元素一次标量 load/store | 数学恒等 | — | ❌ 弃选：把整行 DMA 打回逐元素标量（`T*K*H` 次标量访存，失 DMA/向量）；源本身即整行 |
| C8 | **idx 去重（同槽位只读一次）** | 对重复 `fwd_idx` 值去重后复用行 | 数学恒等 | 减少重复读 | ❌ 弃选：去重需排序/哈希（额外 pass + scratch），且 workload `fwd_idx` 为**随机排列**（多数无重复），`examples/TileOPs/tileops/workloads/moe.py`；收益不成立 |

**R2 在线算法**：**有，且为必需**。K 加权和是天然的**单遍在线 running sum**（`acc += w_k·x_k`，O(1 行) 状态、无反向
依赖），源本身即用单个 loop-carried `acc` 在线累加（§0.3，直接证据）。**必需性**：唯一「非在线」候选 C2（两阶段物化）
在 UB 容量上不可行（`K·H·4 = 229KB > 192KB`，§R4）→ 本算子**只能**用 online 单累加器。**结论严谨化**：不存在需要
**GM 级多遍物化**的候选；遍历只有「按 token × 按 k」一遍。
（对照 `references/algorithm-candidates.md` 的 `ALG-reduction`（分块归约 + 合并两阶段/树形）——那是一般归约族的替代，
本算子归约轴 K≤8 极小且在线即最优，两阶段被 UB 容量否决。）

**R3 复杂度对比**（四口径；代表 shape = kimi-k2-prefill：T=4096，K=8，H=7168，`padded_batch_sum = T*K = 32768`，bf16；
`nk=40`、`spc=ceildiv(4096,40)=103`；elem=2B）：

| 算法候选 | 计算量（FLOPs/MAC） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 |
|---------|--------------------|---------------|---------|-------------|
| C0 基线（源；grid=T + thread + 寄存器） | `2TKH = 470M` | 读 `T·K·H·2 = 470MB` + 写 `T·H·2 = 58.7MB` + 元数据 `2·T·K·4 = 262KB` ≈ **528.8MB** | 输入 1 读 + 输出 1 写 | 寄存器 `acc H·4 = 28KB` + `src H·2 = 14KB` |
| **C1 选定（固定核数 + online UB 行）** | 同 C0（470M） | **同 C0（528.8MB）** + 标量读 `T·K·4 = 131KB`（`fwd_idx`，含于元数据） | 同 C0（1 读 1 写） | UB `ub_src 14KB + ub_f32 28KB + ub_acc 28KB + ub_out 14KB = 84KB` |
| C2 两阶段物化（interface/容量受限） | 同 C1 | 同 C1 | 同 C1 | `ub_k K·H·2 = 114KB`(bf16) / `228KB`(fp32) + `acc 28KB` + `out 14KB` → **fp32 >192KB** |
| C3 源主序 scatter-add（不可实现） | 同 C1 + 反映射 | 读 `T·H·2 = 58.7MB` + 写 `T·K·H·2 = 470MB`（理论更低） | 1 读 1 写 | 无；但**需原子 + 非确定**（CG-2026-0029/0030） |
| C4 选择矩阵 GEMM | `2·T·padded·H = 1.9e12 MAC`（Cube） | +物化 `P` `T·padded·2 = 268MB` | 1 | `P` 268MB + L0C |
| C6 host | host 侧 O(T*K*H) | `mm2_pad` 设备→host→设备 + `output` host→设备 | 1 + 2 次总线往返 | host 内存 |
| C7 逐元素索引 | `T·K·H` 次标量访存 | 同 C1 流量但微小事务 | `T·K·H` 次标量 | 无 |

口径脚注：计算量只计乘加（`2·T·K·H`）；访存 = GM 侧实际流量（读 + 写），不含 UB 内部；扫描遍数指输入/输出在 GM 侧
被读写次数；中间缓冲峰值以片上（UB）为准。**本单元带宽受限**（C1 实测 kimi-k2-prefill 528.8MB / 610µs = **866 GB/s**，
见下 D-2），故 R3 排序以**访存量 + UB 可行容量**为主导；C1 与 C0 流量相同，C1 的价值在**可实现性 + 无原子确定性**。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向论断按 negative-claim-evidence.md §1 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | 需 **thread-per-element + 寄存器 fragment**（NPU 无）→ ❌ | 寄存器模型 N/A | H·2 32B 对齐 ✅ | grid 编译期 ✅ | 2-stage pipeline ✅ | ⚠️ 源形态（`threads=`、fragment）在 npuir 下不可直接落地 | ⚠️ **不可直接落地**（转 C1） |
| C1 **选定** | Vector/MTE2（整行 DMA + 整行 v-op：vcast/vmul/vadd）✅；无 Cube 需求 | UB 84KB ≪ 192KB ✅ | H·2 为 32B 倍数（H∈{64,256,512,2048,3072,7168} 全部 ✅） | `nk`/`spc` 编译期常量 ✅ | 单 scope；`T.serial(K)`（无 pipeline） | ✅ **D-3 探针实测可编译 + 精度通过 + 多运行稳定**（bf16/fp16 × 8 用例 + 3 大 shape + 2 变体 = 20/20） | ✅ **选定** |
| C1-varB（T.Parallel 融合累加，**备选**） | 同 C1（1 条 `T.Parallel(H)` 融合 `acc+=f32*w`）✅ | 同 C1 ✅ | 同 C1 ✅ | 同 C1 ✅ | 同 C1 | ✅ D-3 探针实测同过（20/20），延迟与 A 基本持平 | ✅ **备选**（A/B 裁决见 D-2） |
| C2 两阶段 | Vector ✅ | ❌ `K·H·4=229KB` > 192KB（fp32）；bf16 156KB 过紧 | ✅ | ✅ | ✅ | ✅ 可实现但容量否决 | ❌ 弃选（容量） |
| C3 源主序 scatter-add | Vector ✅（但需跨 token 写） | 无 ✅ | ✅ | ✅ | ✅ | ❌ **无可用原子**（CG-2026-0029 标量载荷 + CG-2026-0030 无 fetch-add）且需逆映射 | ❌ 弃选（无原语 + 非确定） |
| C4 选择矩阵 GEMM | Cube ✅ | `P` 268MB 超 L1 | 分形对齐 | ✅ | GEMM 可流水 | ✅ Cube 可用但 **1.9e12 MAC** | ❌ 弃选（FLOPs 爆炸） |
| C5 尾维 gather 转置 | Vector（VGather）✅ | ❌ `(H,padded)` ≫ 192KB | src 须 int32、驻 UB | 需分块静态化 | 转置链 | ⚠️ API 存在但容量不满足 | ❌ 弃选（容量 + 转置代价） |
| C6 host | CPU | — | — | — | ❌ 设备↔host 往返 + CPU 同步 | ✅ 可行但违源意图 | ❌ 弃选（违反零 CPU 同步） |
| C7 逐元素索引 | Vector 标量 | 无 | — | — | ❌ 标量洪流 | ✅ 可编译但极慢 | ❌ 弃选（失 DMA） |
| C8 idx 去重 | Vector ✅ | 需 scratch | ✅ | 需排序静态化 | ✅ | ✅ | ❌ 弃选（需排序/哈希 pass，随机排列下无重复收益） |

**调研结论**：选定 **C1——单独立 Vector kernel，固定物理核数（纯 Vector → 40）+ 核内 `T.serial` token 循环；每 token
内 `T.serial(K)` online 累加（fp32 UB 行累加器）；每 k「标量 GM 读 `fwd_idx` → 动态源行 GM→UB → vcast fp32 →
vmul 标量权重 → vadd 入累加器」；末次 scaling + cast → UB→GM 整行写出**；**备选 C1-varB**（同一结构，累加用单条
`T.Parallel(H)` 融合 `acc[j]+=f32[j]*w`，D-3 探针同样 20/20 通过）。
关键依据：(i) **可实现性是前置约束**——C0（thread/fragment/`threads=`）不可直接落地、C2（UB 容量）、C3（无原子 + 不可逆映射）、
C4（FLOPs）、C5（容量）、C8（随机排列无收益）分别被容量/复杂度/接口/原语淘汰，C1 是唯一同时满足「整行向量化 + 无 GM→GM
+ 无向量→标量同 scope 数据流 + 无原子」的形态，且经 **D-3 探针设备实测编译 + 精度 + 多运行稳定**；(ii) **无跨 token
累加**是数学结论（每输出行唯一写者），故 C1 天然**确定性**，无需原子；(iii) C1 实测 **866 GB/s @ kimi-k2-prefill**（528.8MB/610µs），
低于 MTE2 混合流量地板 1.26 TB/s（`CONST-mte2-degradation`），差额归因于内核无 pipeline（auto-multi-buffer 关闭，§7.3），
留 Stage 4 深挖（C1-varB 与主选基本持平，见 D-2 裁决）。
**与基线结构差异一句话**：保留源「一 token 一行输出、K 串行加权 fp32 在线累加、整行向量化、EP 掩码零贡献」的算法骨架，
把源「一 block 一 token + thread 协作 H + 寄存器累加 + 2-stage pipeline」换成「固定物理核数核内串行 token + 整行 UB 中转
+ 标量 GM 读驱动动态 slice + 无 pipeline」，并把「标量消费向量产出」与「跨 token 原子」两种形态彻底排除。
**调研范围**：`references/algorithm-candidates.md`（`ALG-reduction`（row-reduction / online）/`ALG-gemv`；无 gather/unpermute 专条）、
pattern-library（`layout.md`、`elementwise.md`、`traps-compiler.md`（`TRAP-C10-strided-gather`）、`traps-runtime.md`、`constants.md`）、
源算子 §0.3/§0.4、本仓 examples（`examples/moe_permute_nopad/_make_gather_kernel_nopad/DESIGN.md` 的行 gather 先例、
`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` 的串行行 gather、
`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` 的工具链陷阱）、
`.agents/evolution/capability-gaps.md`（CG-2026-0001/0027/0028/0029/0030）。
互联网检索：**未执行**（本地信息源已覆盖候选空间：直接索引 / 尾维 gather / 选择矩阵 / scatter-add / 分桶去重 / 两阶段，
无需外部补充）。

**源算法优化手段意图承接**（§0.4 逐项，不得静默丢弃）：
O1（一 block 一 token）→ 重设计为「固定核数 + 每核 `spc` token 串行」（**意图承接**：K 归约局部化 + 零跨 block 通信）；
O2（thread 协作 H）→ 舍弃（NPU 无 thread；`TRAP-threads-kwarg-noop`）；
O3（thread 级 128-bit 向量化）→ 重设计（整行 DMA + 整行向量 op 自动向量化）；
O4（寄存器 fp32 累加器）→ 等价替换（UB fp32 行累加器）；
O5（2-stage pipeline）→ **意图承接、形态替换**（`T.serial` 无 pipeline；重叠留 Stage 4，CG-2026-0001(a) 禁 Pipelined 内 `if`）；
O6（EP clamp + 零权）→ 等价替换（`if raw>=0` 跳过，D-1 E1 位精确）；
O7（scaling 末次向量乘）→ 保留；O8（`out_idx=[]`）→ 保留；O9（索引行 load 进 fragment）→ **保留语义、形态替换**
（标量 GM 读 + 动态 slice，D-3 探针位精确）。

**设计期 roofline（D-2，代表 shape kimi-k2-prefill）**：以 §1.6.3 选定布局 + `pattern-library/constants.md` 实测常数估算下界：
- **流量项**：`T·K·H·2 + T·H·2 = 528.8MB` ÷ MTE2 混合流量地板 `~1.26 TB/s`（`CONST-mte2-degradation`，
  toolchain tilelang 0.1.2+ed787bb / CANN 8.5.0，origin lerp_tensor/ada_layer_norm/ssd_chunk_scan）≈ **420 µs**。
- **发射项**：每 token 约 `3K+2` 条向量 op + `K+1` 条 DMA；每核 `spc=103` token → 每核 ~2678 向量 op；向量 op
  发射成本**未在本场景实测**（`CONST-vector-launch-overhead` 的 ~0.5µs/op 是**大 tile 链**的量级，不直接适用整行 op）——
  标注「未文档化假设 + 估算依据」；由 D-3 性能探针标定（实测 610µs，见下）。
- **容量项**：UB `84KB` ≪ 192KB（`CONST-capacity-910B2C`）。
- **估算下界 = max(流量项, 发射项, 容量项) ≈ 420 µs（流量项主导）**。
- **D-3 实测标定（本机设备，2026-10-14，Ascend910B2C aicore 20）**：C1-varA=**610µs**（866 GB/s）、C1-varB=**604µs**（876 GB/s），
  bf16 精度通过；qwen3-30b-prefill（H=3072，227MB）A=**833µs**（272 GB/s）、B=**976µs**；large-hidden-medium（T=512）A=B≈**96µs**（691 GB/s）。
  → 实测**高于**估算下界 ~1.45×：核实测有效带宽**低于 MTE2 地板**（866 GB/s < 1.26 TB/s），差额归因于**内核无 pipeline
  / auto-multi-buffer 关闭**（每 token 的 K 段 DMA 与向量算术在单核内**串行**，未重叠——对标源 O5 的 2-stage pipeline 被
  CG-2026-0001(a) 阻断）。**发射项是本次实测的主嫌疑**（小 H 的 qwen3-30b 反而更慢，与「固定 op 开销摊薄在更少元素上」
  一致），但未文档化 → 交实验裁决 / Stage 4。
- **未文档化假设项**：整行向量 op（vcast/vmul/vadd）与单行 DMA（含动态偏移）的**发射成本**、以及
  「`T.serial(K)` 无 pipeline 时 DMA 与向量算术的串行代价」——均标注「未文档化假设 + 估算依据」；**不影响主选/备选裁决**
  （可行性已由 D-3 探针硬裁定），仅影响成本刻画与 Stage 4 方向。

**实验裁决计划（三件套）**：

- **主选（进 Stage 3）**：C1-varA——固定核数 + 核内 `T.serial`；每 k 显式 **v-prefix 链**（`T.vcast` → `T.vmul`(标量) →
  `T.vadd`，in-place）（§3.3 主选）。
- **备选（≤1，结构完整可实现）**：C1-varB——同结构，仅把「vcast + vmul + vadd」三条 op 换成单条
  `T.Parallel(H): ub_acc[0,j] = ub_acc[0,j] + ub_f32[0,j] * w`（融合加权累加；scalar `w` 为循环不变量）。UB 预算与分核三要素
  同主选。D-3 探针**已验证编译 + 精度 + 多运行稳定**，且延迟与主选**基本持平**（kimi 604 vs 610µs）。
- **裁决计划**：
  ① **代表 shape** = smoke（T=1,K=8,H=512）、tiny（T=4,K=2,H=64）、topk1（T=8,K=1,H=64）、ep-mask（T=16,K=4,H=256）、
     kimi-k2-prefill（T=4096,K=8,H=7168）、qwen3-30b-prefill（T=4096,K=8,H=3072）、large-hidden-medium（T=512,K=8,H=7168）；
  ② **测量指标** = 首编编译结果 + 输出与 golden 对齐（精度标准 §8.2）+ 反复运行稳定 + 稳态延迟（msprof / 计时，warmup 后多次均值）；
  ③ **判定阈值** = 备选 latency < 主选 × (1−ε) 才翻转，取 **ε=0.05**（D-3 实测差 < 2%，低于阈值 → **默认保持主选 C1-varA**）；
     若某 shape 上备选显著更快（>5%）则按 shape 分派；
  ④ **回写路径** = 实测数字回写 §1.6.0 roofline / §5.3 / §8；若备选胜出按 perf-feedback.md 受控路由。
  **shape 特化可行性**：本 kernel 经工厂按 `(T, K, H, padded_batch_sum, dtype, scaling)` 特化（源即工厂 + op 层 `lru_cache`），
  **可按 shape 分派主选/备选**（如大 H 走主选、小 H 走备选）。
  **设计期探针（D-3，已执行）**：见上「D-3 实测标定」——主选与备选均**已验证可编译、精度通过、多运行稳定**，并已给出稳态延迟。
  **Stage 4 额外候选（非本设计承诺，登记 §9 风险 R-2）**：token 批处理（一次 gather B 个 token 的 K 行入 `ub_src(B,H)`，
  用 `(B,1)` 权重行广播一次 `vmul`/`vadd`）以摊薄固定 op 开销——受 TB-对齐 与 auto-multi-buffer 约束，待 Stage 4 实测裁决。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 C1 算法（固定核数 + 核内 token 串行 + online K 累加）。

**采纳 1 项**（EP 掩码「跳过」替代「零权」）；其余候选（见下）无收益或改变舍入顺序，不采纳。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | EP 掩码：跳过 vs 零权 | `acc += mm2_pad[max(idx,0)]·w·[idx≥0]`（恒执行 load） | `if idx ≥ 0: acc += mm2_pad[idx]·w`（`idx<0` 不累加） | `[idx<0]` 时源贡献 = `mm2_pad[0]·w·0 = 0`，跳过 = 加 0；**fp32 加 0 精确**，两式逐位相同（**E1 EQUIV_PASS**） | `idx<0` 槽位**少一次行 DMA + 3 条向量 op**；EP 下节省 ∝ 非本地槽位比例 |

**等价性机器验证（D-1）**：`examples/moe_unpermute/_make_unpermute_kernel/verify_equiv.py`
（候选「跳过式」vs 基线「零权式」，torch CPU fp64 参照，5 类用例 × bf16/fp16 = 10 行）：

| 用例 | dtype | 最大 ulp 差 | 违反率 | 结论 |
|------|-------|------------|--------|------|
| random permutation (no -1) | bfloat16 / float16 | 0 / 0 | 0 / 0 | EQUIV_PASS / EQUIV_PASS |
| every 3rd slot = -1 | bfloat16 / float16 | 0 / 0 | 0 / 0 | EQUIV_PASS / EQUIV_PASS |
| all -1 | bfloat16 / float16 | 0 / 0 | 0 / 0 | EQUIV_PASS / EQUIV_PASS |
| K = 1 | bfloat16 / float16 | 0 / 0 | 0 / 0 | EQUIV_PASS / EQUIV_PASS |
| all same slot | bfloat16 / float16 | 0 / 0 | 0 / 0 | EQUIV_PASS / EQUIV_PASS |

（`EQUIV_RESULT: ALL_PASS`；脚本存在于算子目录，Stage 2 可重跑复核。）

**未采纳项**（逐项结论与依据）：
1. **`scaling` 折叠进权重**（`out = Σ (scale·w_k)·x_k` 替代 `out = scale·Σ w_k·x_k`）：**不采纳**。数学上等价（分配律），
   但 **fp32 舍入顺序不同**（先缩放各项 vs 先求和再缩放），且源与 manifest 契约是**末次缩放**（源 `acc*T.float32(scaling)`）；
   无收益（同为 `1+T*K` 次乘），且会引入与源不一致的舍入 → 保持末次缩放。
2. **除法/倒数、sqrt/rsqrt、log/exp 降代换**：**不采纳**——本算子**无除法/超越函数**（纯乘加 + 加），不存在此类恒等变形空间。
3. **公共子表达式消除**：**不采纳**——每 token 的 K 次 gather 各读不同行、无重复子式；`fwd_idx`/`topk_weights` 每处读一次即用。
4. **访存削减（去重 / 源主序）**：**不采纳**——见 §1.6.0 C8（随机排列无重复收益）/ C3（需逆映射 + 原子，不可实现）。
5. **稳定化变形（减 max）**：**不采纳**——无 exp/sum，无溢出路径，无稳定化需求。

**优化结论**：采纳 **1 项**（EP 掩码跳过，D-1 全 PASS）；优化后公式（供 §3.1 拆解）：

$$
\text{out}[i,:]=\text{scale}\cdot\sum_{k:\;fwd\_idx[iK+k]\ge0} topk\_weights[i,k]\cdot mm2\_pad\big[fwd\_idx[iK+k],:\big]
$$

（与 §1.3 公式数学等价；`idx<0` 项天然为零，表达为条件累加。）
**量化影响评估**：跳过项不引入任何额外 fp16/bf16 舍入；累加仍在 fp32、k 升序；精度标准见 §8.2。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案全部循环与标量计算点（对象 = §1.6.0 选定、经 §0.6 重设计落地的 C1 算法）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 行内逐元素乘加（H 元素）`acc[j]+=f32[j]*w` | 源 `T.Parallel(H)`（GPU thread） | 主选 `T.vcast + T.vmul(标量) + T.vadd`（整行向量 op）；备选单条 `T.Parallel(H)` 融合 | ✅ | —（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播、`T.vadd.md`、`T.vcast.md`；D-3 探针实测两形态均可编译/精度通过） |
| 2 | 行内 bf16/fp16→fp32 提升 | 源 `T.Cast("float32", src[j])`（逐元素） | **整行 `T.vcast(ub_src, ub_f32, round_mode="rint")`** | ✅ | —（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 `bf16/f16→f32 rint`） |
| 3 | 行内 fp32→bf16/fp16 落盘 cast | 源 `T.Cast(dtype, acc[j]*scale)` | **整行 `T.vcast(ub_acc, ub_out, round_mode="round")`** | ✅ | —（同 §2.2.1 `f32→bf16/f16 round`） |
| 4 | 末次 `scaling` 缩放（H 元素） | 源 `T.Parallel(H): acc[j]*scale` | **整行 `T.vmul(ub_acc, scale, ub_acc)`**（标量 src2，in-place） | ✅ | —（`T.vmul.md` §2.3-4「仅 src2 可为标量」；D-3 探针 scaling 用例通过） |
| 5 | token 轴循环（`spc` 个 token/核） | 源 block 级并行（grid=T） | 无向量等价（每 token 的 gather 行不同）；**核内 `T.serial(spc)`**，但行内元素已全向量化（每 token 每 k 一条整行 DMA / 整行 v-op，非逐元素） | ❌ | **block 级任务映射**（每核 `spc` 次整数算术），非逐元素热点；所需「批量行 gather」向量原语在本项目确不存在（`T.gather` 仅尾维 `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3 + `TRAP-C10-strided-gather`），且 `mm2_pad` 全量（prefill 470MB）不可能驻 UB |
| 6 | K 归约循环（`T.serial(K)`） | 源 `T.Pipelined(K)` | 无法向量化（每 k 源行不同、且累加器有 loop-carried 依赖）；**改核内 `T.serial(K)` 在线累加**，行内元素已向量化 | ❌ | **tile 级顺序依赖**（online 累加的 loop-carried 依赖，§0.6 R4；归约轴 K≤8 极小）；`T.Pipelined` 因 CG-2026-0001(a)（体内 runtime `if`）不可用 |
| 7 | 源行号读取 `raw = fwd_idx[flat]` | 源向量化索引 load | 可先 `T.copy` 一段 `fwd_idx` 入 UB 再向量读——**但标量消费该 UB 值命中 CG-2026-0027** | ❌ | **CG-2026-0027**：标量读由向量 op 写入 UB 的值不同步（同产物多运行不一致，`T.pipe_barrier` 无效）。**必须标量 GM 读**（跨 kernel；`fwd_idx` 由上游 permute 写 GM）。**非逐元素热点**（每 7168 元素 1 次 4B 标量读） |
| 8 | 路由权重读取 `w = topk_weights[token,k]` | 源标量/向量读 | 标量 GM 读（作为 `T.vmul` 标量 src2） | ❌（标量） | **标量操作数必须在 `T.Kernel` 内、`T.Scope` 外定义**（`T.vmul.md` §2.3-2）；`topk_weights` 每 token 每 k 读 1 次 4B 标量，非逐元素热点 |
| 9 | 尾守卫 `if token < T` / EP 守卫 `if raw >= 0` | 源标量比较守卫 | 无向量等价（控制流谓词） | ❌ | **动态边界控制流**：`T` 可能非核数整数倍、`raw` 运行期含 `-1`；为 tile 级谓词，非逐元素计算；D-3 探针实测该标量 `if` 形态可编译且正确（**对照**：CG-2026-0028c 禁的是**标量 `T.if_then_else` 表达式**，非 `if` 语句） |
| 10 | 核内任务起点 `token = core_id*spc + t` | — | 无向量等价 | ❌ | **block 级索引计算**（每核 `spc` 次整数算术），非逐元素热点 |

**向量化结论**：**行内元素已全部向量化**（第 1–4 项：整行 `vcast/vmul/vadd`，D-3 探针实测）；本单元**不存在逐元素标量
热点**。仅保留 6 处标量/循环点（第 5–10 项：token 串行、K 串行 online、标量 GM 行号读、标量权重、尾/EP 守卫、核起点算术），
理由类别：**block 级任务映射 / tile 级顺序依赖 / 无行 gather 向量原语（附文档 + 陷阱佐证）/ 标量消费向量产出不可同步
（CG-2026-0027）/ 动态边界控制流 / 标量操作数约束**；均**非逐元素热点**。**无可向量化而未向量化的逐元素标量循环。**

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本单元 I/O layout 为**契约**（`mm2_pad (padded,H)` 行主序、`fwd_idx (T*K,)`、`output (T,H)` 行主序）；
> 核内布局与 lane 映射是**自由变量**。本算子非 batch 维有 2 个：**token 轴 T**（输出行 / 归约载体）与
> **元素轴 H**（行内连续轴）；另有 **K 轴**（归约轴）与 **padded 轴**（输入行索引轴）。

**轴质量评分**（三个候选向量化轴）：

| 评分项 | token 轴 T | 元素轴 H（行内连续轴） | K 轴 |
|--------|-----------|----------------------|------|
| 整除性 / 尾 lane 浪费 | 非向量轴；`spc` 静态、尾块由 `if` 守卫 | **H·2 为 32B 倍数**（H∈{64,256,512,2048,3072,7168} 全部 ✅）；DMA 无尾 lane 问题 | 非向量轴（K≤8）；源行不同，不可合并 |
| 累加链形态 | 无（逐 token 独立） | 无（行内无累加；归约在 K 轴） | **串行累加维**（loop-carried fp32 累加器）；K 小（≤8），串行代价可忽略 |
| repack 代价 | 0 | 0（行内连续，无需转置/NZ） | 0（沿用行主序） |
| UB 容量影响 | 无 | `ub_src/ub_f32/ub_acc/ub_out = 84KB @H=7168` | 两阶段物化被容量否决（§1.6.0 C2） |

**候选矩阵**（gather+reduce 类：向量轴选择 × 索引处理方式 × 任务粒度）：

| # | 方案 | 向量化轴 / 关键决策 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|------|--------------------|------------|--------------|---------|
| L1 | **元素轴整行 + token 核内串行**（主选 C1-varA） | 向量化轴 = **元素轴 H**（整行 DMA / 整行向量 op）；token 轴串行；K 轴串行 online | 无 | DMA 数 = `(K+1)·T`；DMA 小行；**实测 kimi 610µs / 866 GB/s** | ✅ **主选** |
| L2 | **元素轴整行 + T.Parallel 融合累加**（备选 C1-varB） | 同 L1，累加用单条 `T.Parallel(H)` 融合 `acc+=f32*w` | 无 | 同 L1；**实测 kimi 604µs / 876 GB/s**（与 L1 持平） | ✅ **备选**（D-2 三件套） |
| L3 | **token 轴批量（B token × H 一 tile）** | 向量化轴 = **元素轴 H**，一次处理 B 个 token（`ub_src(B,H)`；权重用 `(B,1)` 行广播） | 无（但 gather 用非零起点 UB dst，见 `TRAP-UB-dst-align`） | 摊薄 per-op 固定开销 / B；UB ×B；受 auto-multi-buffer 与非零起点 dst 约束 | ⚠️ **Stage 4 候选**（§1.6.0 裁决计划；登记 §9 风险 R-2） |
| L4 | **K 轴向量化（把 K 行物化后一次归约）** | 向量化轴含 K | 需 `ub_k(K,H)` | 减少 UB 读；但 `K·H·4=229KB > 192KB` | ❌ 弃选（容量，§1.6.0 C2） |
| L5 | **`T.gather` 尾维（转置使 token 轴为尾维）** | 尾维 = token 轴 | 需大转置链 | `(H,padded)` ≫ 192KB | ❌ 弃选（容量 + 转置代价，§1.6.0 C5） |

**布局决策结论**：选定 **L1（主选）/ L2（备选）**：核内布局 = **行主序原生布局（无 repack）**、向量化轴 = **元素轴 H**
（整行 `T.copy` DMA + 整行 `vcast/vmul/vadd`）、索引轴（token/padded）串行（标量 GM 读控制动态源行）、K 轴串行 online 累加。
该结论同步落入 §3.3 伪代码与 §6 循环结构（累加循环的内层向量维 = 元素轴 H）。**弃选的量化理由**见候选矩阵（L4 容量、
L5 容量 + 转置）；L3 为 Stage 4 候选而非本设计承诺。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Developer**（与源 op 的 harness 迁移基线一致；迁移任务默认模式）。

### 2.2 选型理由

- 本单元是**纯 Vector / MTE2** 算子（整行 gather-DMA + 逐元素向量算术 + 行归约），**无 `T.gemm`、无 Cube、无跨引擎交接**。
  迁移决策后不含「persistent + gemm 混排」（**CG-2026-0010** 的 Developer 崩溃形态不触发），也不含跨引擎同步
  （CG-2026-0026 不触发）。
- Developer 模式对**单 scope 纯 Vector 结构**提供自动同步（§7.1），无需手写 flag；D-3 探针在 Developer 下
  **20/20 编译 + 精度 + 多运行稳定**，直接验证了本结构的可用性。
- 参照先例：同族搬运单元 `examples/moe_permute_nopad/_make_gather_kernel_nopad/_make_gather_kernel_nopad.py`
  与 `examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` 均为单 scope Vector 形态。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 显式 `T.alloc_ub`（`ub_src`/`ub_f32`/`ub_acc`/`ub_out`），无 L1/L0A/L0B/L0C |
| 计算方式 | 整行 v-prefix 向量 op（`T.vcast`/`T.vmul`/`T.vadd`）+ `T.clear`；无 `T.Parallel` 于主选（备选用单条 `T.Parallel`） |
| 同步 | Developer 自动同步（§7.1），无手动 flag |
| pass_configs | 关闭 `NPUIR_ENABLE_AUTO_MULTI_BUFFER` 与 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`（§7.3，动态源行 range copy 可靠性 + UB 预算可控） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（EP 掩码跳过）；每个步骤优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `ub_acc ← 0` | 每 token 累加器清零 |
| 2 | `flat ← token·K + k`（k 串行 0..K-1） | flat 槽位下标 |
| 3 | `raw ← fwd_idx[flat]` | 标量 GM 读（EP：`-1` 标记非本地） |
| 4 | `if raw ≥ 0`：`w ← topk_weights[token,k]`；`ub_src ← mm2_pad[raw,:]` | EP 跳过 + 动态源行整行 load |
| 5 | `ub_f32 ← Cast_f32(ub_src)` | 行内 fp16/bf16→fp32 |
| 6 | `ub_f32 ← ub_f32 · w`；`ub_acc ← ub_acc + ub_f32` | 标量权重乘 + 在线累加（fp32） |
| 7 | `ub_acc ← ub_acc · scale`（`scale≠1` 时） | 末次缩放 |
| 8 | `ub_out ← Cast_dtype(ub_acc)`；`output[token,:] ← ub_out` | 落盘 cast + 整行写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 核内 token 循环 | `for t in T.serial(spc)` + `T.Kernel(nk, is_npu=True)` | `nk=min(T,40)`、`spc=ceildiv(T,nk)`（静态） | Developer |
| 1 | `ub_acc ← 0` | `T.clear(ub_acc)` | `ub_acc` = `alloc_ub((1,H),"float32")` | Developer |
| 2 | `flat = token*K + k`；`raw = fwd_idx[flat]` | 标量 GM 读 | `T.serial(K)` 内 | Developer |
| 3 | `if raw >= 0:` 守卫（EP skip） | `if` 语句 | 不用 `T.if_then_else`（CG-2026-0028c） | Developer |
| 4 | `ub_src ← mm2_pad[raw,:]` | `T.copy(mm2_pad[raw:raw+1, 0:H], ub_src[0:1,0:H])` | 动态源行 GM→UB | Developer |
| 5 | `ub_f32 ← Cast_f32(ub_src)` | `T.vcast(ub_src, ub_f32, round_mode="rint")` | `round_mode` **必须关键字** | Developer |
| 6a | `ub_f32 ← ub_f32 · w` | `T.vmul(ub_f32, w, ub_f32)` | `w` = 标量（`topk_weights[token,k]`），in-place | Developer |
| 6b | `ub_acc ← ub_acc + ub_f32` | `T.vadd(ub_acc, ub_f32, ub_acc)` | in-place（dst==src0 合法） | Developer |
| 7 | `ub_acc ← ub_acc · scale` | `T.vmul(ub_acc, scale, ub_acc)` | `scale` = 编译期 float（`scaling≠1` 时） | Developer |
| 8a | `ub_out ← Cast_dtype(ub_acc)` | `T.vcast(ub_acc, ub_out, round_mode="round")` | f32→bf16/f16 | Developer |
| 8b | `output[token,:] ← ub_out` | `T.copy(ub_out[0:1,0:H], output[token:token+1,0:H])` | 动态目的行 UB→GM | Developer |

**备选（C1-varB）差异**：步骤 5–6b 合并为单条 `for j in T.Parallel(H): ub_acc[0,j] = ub_acc[0,j] + ub_f32[0,j] * w`
（`w` 为循环不变量标量；下标不变换，符合 `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3）。

### 3.3 计算伪代码

```python
# 基于 TileLang API 的计算流程伪代码（§1.6.0 主选 C1-varA；D-3 探针实测形态）
nk = min(T, VECTOR_CORES)          # VECTOR_CORES = 2 * aicore_num = 40（§5.5）
spc = (T + nk - 1) // nk            # 编译期静态边界

with T.Kernel(nk, is_npu=True) as (core_id, _subid):
    ub_src = T.alloc_ub((1, H), dtype)          # 动态源行缓冲（bf16/fp16）
    ub_f32 = T.alloc_ub((1, H), "float32")      # 提升 + 乘权重后的行
    ub_acc = T.alloc_ub((1, H), "float32")      # fp32 在线累加器
    ub_out = T.alloc_ub((1, H), dtype)          # 落盘行
    for t in T.serial(spc):                      # 核内 token 串行（静态边界）
        token = core_id * spc + t
        if token < T:                            # 尾守卫
            T.clear(ub_acc)                      # acc ← 0
            for k in T.serial(K):                # K 串行 online 累加
                raw = fwd_idx[token * K + k]     # 标量 GM 读（跨 kernel）
                if raw >= 0:                     # EP 跳过（§1.6.1 采纳项）
                    w = topk_weights[token, k]   # 标量权重（Kernel 内 / Scope 外）
                    T.copy(mm2_pad[raw : raw + 1, 0:H], ub_src[0:1, 0:H])  # 动态源行 GM→UB
                    T.vcast(ub_src, ub_f32, round_mode="rint")             # bf16/fp16→fp32
                    T.vmul(ub_f32, w, ub_f32)                              # × 标量权重（in-place）
                    T.vadd(ub_acc, ub_f32, ub_acc)                         # 在线累加（in-place）
            if scaling != 1.0:                   # 编译期常量分支（trace-time）
                T.vmul(ub_acc, scaling, ub_acc)  # 末次缩放
            T.vcast(ub_acc, ub_out, round_mode="round")   # fp32→dtype
            T.copy(ub_out[0:1, 0:H], output[token : token + 1, 0:H])  # 整行写回
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证 |
|-----|---------|------|
| `T.Kernel(n, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；先例 `_make_gather_kernel_nopad.py` | ✅ D-3 探针 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/`；先例同上 | ✅ |
| `T.clear(buf)` | `docs/Tilelang.language/内存操作/T.clear.md` §2.2.1（fp32 √） | ✅ D-3 探针 |
| `T.copy(GM slice→UB)` / `T.copy(UB→GM slice)`（动态 offset） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.1（GM↔UB ✅）、§2.3-5（**禁 GM→GM**） | ✅ D-3 探针（动态源行 + 动态目的行） |
| `T.vcast(src, dst, round_mode=...)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`bf16→f32 rint`、`f16→f32 rint`、`f32→bf16/f16 round`） | ✅ D-3 探针（**`round_mode` 必须关键字**，见 §9） |
| `T.vmul(tensor, scalar, tensor)` | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量广播 + §2.3-2/4 | ✅ D-3 探针 |
| `T.vadd(tensor, tensor, tensor)` | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2 + §2.3-3 | ✅ D-3 探针 |
| `T.Parallel(H)`（备选） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.2.1/§2.3 | ✅ D-3 探针 varB |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **No** | 一维 `T.Kernel(nk, is_npu=True)` |
| Developer 禁 GM→GM 拷贝 | No（规避） | 一切搬运经 UB 中转（`T.copy` GM→UB→GM 两段） |
| L0C / Cube 容量 | No | 无 `T.gemm`、无 L0C |
| `threads=` 无效 | No（未使用） | token 轴由核内 `T.serial` 承接，不用 `threads=` |
| `T.Pipelined` 体内 runtime `if`（CG-2026-0001a） | No（规避） | K 循环用 `T.serial`，不用 `T.Pipelined` |
| 标量消费向量产出（CG-2026-0027） | No（规避） | `fwd_idx` 标量读自 **GM**（跨 kernel） |
| 标量 `T.if_then_else` / 比较表达式（CG-2026-0028c） | No（规避） | EP/尾守卫用 `if` **语句**；权重无标量 select |
| 标量载荷原子 / fetch-add（CG-2026-0029/0030） | No（不需要） | 无跨 token 累加 → 无需原子 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | 1D `T.Kernel(num_tokens, threads=...)` | 1D `T.Kernel(nk, is_npu=True)` | 固定物理核数 + 核内 token 串行（§0.6 R3） |
| 并行模型 | block/thread 两级 | 核级 + 向量 lane | 整行 DMA / 整行向量 op 承接 H（§0.6 R3/R4） |
| 行 gather | 向量化索引 load 进 fragment | 标量 GM 读驱动动态 slice GM→UB | §0.6 R1（无行 gather 向量原语） |
| 累加器 | 寄存器 fragment `acc[H]` | UB fp32 行 `T.alloc_ub` | §0.6 R1（等价替换） |
| 流水 | `T.Pipelined(num_stages=2)` | `T.serial(K)`（无 pipeline） | §0.6 R4（CG-2026-0001a）；重叠留 Stage 4 |
| EP 掩码 | clamp slot 0 + 权重零 | `if raw>=0` 跳过 | §0.6 R2（等价，D-1 E1） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/moe_permute_nopad/_make_gather_kernel_nopad/_make_gather_kernel_nopad.py` | **高** | 动态源行 gather（标量 GM 读 → `T.copy` 动态 slice）、固定 40 核 + 核内 `T.serial`、`_PASS` 配置、无手动同步 |
| `examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py` | 中 | `for i in T.serial(block_I): cur_idx=...; T.copy(IndexK[cur_idx:cur_idx+1,0:D], ...)` 串行行 gather（2-D slice 保 rank） |
| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | 中（陷阱） | CG-2026-0027/0028 工具链陷阱、分核三要素写法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `mm2_pad` | `(padded_batch_sum, H)` | fp16 / bf16 | 下投影输出（padded 专家连续区） |
| `fwd_idx` | `(T*K,)` | int32 | flat 槽位 → padded 槽位（`-1` = 非本地专家） |
| `topk_weights` | `(T, K)` | float32 | 路由权重 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | `(T, H)` | fp16 / bf16（同 `mm2_pad`） | 还原后 token 激活；`out_idx=[]` 调用方分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `ub_src` | `(1, H)` | fp16 / bf16 | UB | 动态源行缓冲（每 k 一行） |
| `ub_f32` | `(1, H)` | float32 | UB | 提升 + 乘权重后的行 |
| `ub_acc` | `(1, H)` | float32 | UB | fp32 在线累加器（每 token 清零） |
| `ub_out` | `(1, H)` | fp16 / bf16 | UB | 落盘行（cast 后） |

### 4.4 内存搬运路径

```
GM[fwd_idx(T*K)] ──标量 GM 读 raw──┐
GM[mm2_pad(padded,H)] ──T.copy 动态源行 GM→UB──▶ ub_src ──vcast──▶ ub_f32 ──vmul(标量 w)──▶ ub_f32
                                                                                              │ vadd（K 串行在线）
                                                                                              ▼
GM[topk_weights(T,K)] ──标量 w───────────────────────────────────────────────────────▶ ub_acc(1,H) f32
                                                                                              │ vmul(scale), vcast
                                                                                              ▼
                                                                       ub_out(1,H) dtype ──T.copy 动态目的行 UB→GM──▶ GM[output(T,H)]
```

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) @H=7168 |
|--------|-------|-------|---------------------|
| `ub_src` | `(1, 7168)` | bf16 | 14336 |
| `ub_f32` | `(1, 7168)` | float32 | 28672 |
| `ub_acc` | `(1, 7168)` | float32 | 28672 |
| `ub_out` | `(1, 7168)` | bf16 | 14336 |
| **总计** | | | **86016 / 196608 (192KB, Ascend910B2C A2/A3)** |

> 余量 ~2.2×（含 `CONST-capacity-910B2C` 的 ×1.10–1.12 放大后仍 ~96KB ≪ 192KB）。auto-multi-buffer 已关闭（§7.3），
> 无 ×1.7 膨胀。H 更大（>16384）时预算线性增长，须复核（风险 R-3）。

### 4.6 动态轴定义

无运行时动态轴：`T`、`K`、`H`、`padded_batch_sum`、`dtype`、`scaling` 均**编译期特化**（源即工厂 + op 层 `lru_cache`）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
def _make_unpermute_kernel(num_tokens, top_k, hidden_size, padded_batch_sum, dtype, scaling=1.0):
    ...
# _PASS = {
#   tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
#   tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
# }
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: **纯 Vector / MTE2**。

**判定依据**: 仅含整行 DMA + 逐元素向量算术（vcast/vmul/vadd）+ 行内归约（K 轴串行 online），**无 matmul**，判定纯 Vector；
无 `T.gemm` → 不涉及 L1/L0A/L0B/L0C/NZ/分形。

### 5.2 Block 划分

本算子无 GEMM 式 `block_M × block_N`：**逻辑任务 = 每个输出 token 一行**（共 `T` 个），每核处理 `spc` 个 token。

```python
nk  = min(T, VECTOR_CORES)      # 固定物理核数（纯 Vector → 40）；smoke T=1 → nk=1
spc = (T + nk - 1) // nk        # 编译期静态核内 token 数
# 核内：for t in T.serial(spc): token = core_id*spc + t; if token < T: ...
```
- 元素轴 H 不分块（整行一条 DMA / 整行向量 op；H·2 为 32B 倍数）。

### 5.3 约束分析

- **对齐约束**: 行宽 `H·2` 为 32B 倍数（H∈{64,256,512,2048,3072,7168} 全部满足；H=64→128B、H=7168→14336B）；
  元素轴向量宽度：bf16/fp16 ×8 / fp32 ×4，H 为 8 的倍数时无尾 lane 浪费（全部 workload H 满足）。
- **UB 容量**: 总 buffer = 86016B @H=7168 < 192KB ✓（§4.5）。
- **L0 容量**: 无 Cube 计算，不适用。
- **非整除**: token 轴 `T` 非 `nk` 整数倍时由 `if token < T` 尾守卫处理（**不是** padding+crop；输出定长）。
  EP 的 `raw<0` 由 `if raw >= 0` 守卫（§1.6.1）。

### 5.4 注意事项

- **尾块**：`spc` 静态边界保证循环安全；末核越界 token 由 `if token < T` 跳过（与源 grid=`T` 语义等价）。
- **EP 槽位**：`raw<0` 跳过该 k 的 load/累加（等价零贡献）。
- **输出唯一写者**：每个 output 行由唯一 token 归属的核写一次 → 无写竞态、无原子。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **物理核数**（实查，禁止假设）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()   # -> 20  (Ascend910B2C, 本机 2026-10-14 实查)
  ```
  本单元为**纯 Vector**算子 → 物理核数 = `get_aicore_num() * 2 = 40`（`VECTOR_CORES`）。
- **逻辑核数**：`num_logical_kernels = T`（一 token 一行输出；prefill T=4096；smoke T=1）。
- **规模判定**：prefill `T=4096 ≫ 40` → **极大规模**（无法通过缩减 token 粒度减少核数）；smoke `T=1 ≤ 40` → **无需适配**。
- **分核方案（极大规模）**：固定启动核数 = **物理核数 40**；核内 `T.serial(spc)` 串行处理 `spc = ceildiv(4096,40) = 103`
  个 token（**循环边界静态**）；末核 `if token < T` 守卫。**不超发逻辑核**（一维核超发会被串行调度，core-split-strategy.md §1）。
  smoke/小 shape：`nk = min(T, 40)`（如 T=1 → nk=1、spc=1；T=16 → nk=16、spc=1），无需核内串行。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环。保留的标量/循环点理由见 §1.6.2。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级并行 | 固定物理核 | `T.Kernel(nk, is_npu=True)` | nk = min(T, 40)（§5.5） |
| token 轴（核内） | 核内串行 | `T.serial(spc)` | 静态边界；每核 `spc` 个 token（§0.6 R3） |
| K 归约轴 | 核内串行 | `T.serial(K)` | online loop-carried 依赖；K≤8；不用 `T.Pipelined`（CG-2026-0001a） |
| 元素轴 H | **向量化** | `T.vcast`/`T.vmul`/`T.vadd`（整行） | §1.6.3 主选向量轴 = H（整行向量 op） |

**未使用** `T.Parallel`（主选，备选 varB 用 1 条）/ `T.Pipelined` / `T.Persistent`。

### 6.2 循环伪代码

```python
with T.Kernel(nk, is_npu=True) as (core_id, _subid):
    # 分配 ub_src / ub_f32 / ub_acc / ub_out（§4.3）
    for t in T.serial(spc):                 # token 轴核内串行（静态）
        token = core_id * spc + t
        if token < T:                       # 尾守卫
            T.clear(ub_acc)
            for k in T.serial(K):           # K 归约轴串行 online
                raw = fwd_idx[token * K + k]
                if raw >= 0:                # EP 守卫
                    w = topk_weights[token, k]
                    T.copy(mm2_pad[raw:raw+1, 0:H], ub_src[0:1, 0:H])
                    T.vcast(ub_src, ub_f32, round_mode="rint")
                    T.vmul(ub_f32, w, ub_f32)
                    T.vadd(ub_acc, ub_f32, ub_acc)
            if scaling != 1.0:
                T.vmul(ub_acc, scaling, ub_acc)
            T.vcast(ub_acc, ub_out, round_mode="round")
            T.copy(ub_out[0:1, 0:H], output[token:token+1, 0:H])
```

### 6.3 流水线优化

**不使用** `T.Pipelined`：本单元 K 循环体内有运行时 `if raw >= 0`，置于 `T.Pipelined` 会触发 Developer codegen
SIGSEGV（**CG-2026-0001(a)**）；`top_k < 2` 时源本身也退化 `T.serial`。源 O5 的「重叠 load 与累加」意图当前以
`T.serial` 承接，重叠优化留 Stage 4（§9 风险 R-2，token 批处理 / 受控启用 multi-buffer）。

### 6.4 尾块处理

- token 尾块：`if token < T` 守卫（`spc` 静态）。
- EP 槽位：`if raw >= 0` 守卫。
- 无 GEMM 非整除 padding 需求（输出定长 `(T,H)`）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（Developer 单 scope，无手动 flag）。

### 7.2 同步点说明

无手动同步点。跨引擎依赖（MTE2 GM→UB 写 `ub_src` → Vector vcast 读 → 下一 k MTE2 再写 `ub_src` 的 WAR；
Vector vcast 写 `ub_out` → MTE3 UB→GM 读）由 Developer 自动同步插入。**依据**：D-3 探针在 6 次重复运行下**逐位稳定**
（未观察到 CG-2026-0009 类的 WAR 竞态），与同族 `_make_gather_kernel_nopad.py`（同为 MTE2→MTE3 同 buffer 流转、无手动同步）
一致。若 Stage 3/4 观察到不稳定，备选为显式多版本 `ub_src`（受 UB 预算约束，§9 风险 R-2）。

### 7.3 pass_configs 配置

```python
_PASS = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}
```
理由（与同族 `_make_gather_kernel_nopad.py` 一致）：动态源行 / 动态目的行的 range `T.copy` 在 auto-multi-buffer /
buffer 规划开启时可能被重写为非支配 IR（`TRAP-UB-dynsubview-dominance`，CG-2026-0012）；关闭后 UB 单版本、预算是
§4.5 静态值，且 D-3 探针在该配置下**编译 + 精度 + 多运行稳定**。代价：无自动多缓冲 → 无跨 token 重叠（§9 风险 R-2）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 以 **§0.1 源算子语义**为唯一依据实现（移植源仓 `examples/TileOPs/tests/ops/test_moe_unpermute.py` 的
> `_ref_moe_unpermute` + EP 变体），**不得复刻** §0.6 的 NPU 算法。

```python
def golden_unpermute(mm2_pad, fwd_idx, topk_weights, scaling=1.0):
    """逐 token 加权 gather-reduce（源语义；EP: fwd_idx==-1 贡献零）。"""
    T_tok, K = topk_weights.shape
    H = mm2_pad.shape[1]
    dtype = mm2_pad.dtype
    out = torch.zeros(T_tok, H, dtype=torch.float32, device=mm2_pad.device)
    for i in range(T_tok):
        for k in range(K):
            slot = int(fwd_idx[i * K + k].item())
            if slot >= 0:                         # EP: -1 贡献零
                out[i] += mm2_pad[slot].float() * topk_weights[i, k].item()
    return out.to(dtype) if scaling == 1.0 else (out * scaling).to(dtype)
```

（可用向量化等价实现加速，但须保持「fp32 累加 + 末次 scale + 单次 cast」与语义一致。）

### 8.2 精度标准

| dtype | rtol | atol | 依据 |
|-------|------|------|------|
| bfloat16 | 1.6e-2 | 1.6e-2 | `examples/TileOPs/tests/ops/test_moe_unpermute.py` L107–108 |
| float16 | 1e-3 | 1e-3 | 同上 |

（fp32 中间累加 + 单次 cast；D-1 E1 逐位等价；D-3 探针实测 bf16/fp16 全部在容差内——large-h bf16 max_diff 9.8e-4、
fp16 max_diff 1.2e-4；EP/scaling 用例位精确。）

### 8.3 L0 门槛测试计划

| 用例 | shape | dtype | 判定 |
|------|-------|-------|------|
| L0-1 smoke | T=1,K=8,H=512 | bf16/fp16 | 容差内通过 |
| L0-2 tiny | T=4,K=2,H=64 | bf16/fp16 | 容差内通过 |
| L0-3 top_k=1 | T=8,K=1,H=64 | bf16 | 通过 + 6 次稳定 |
| L0-4 EP 掩码 | T=16,K=4,H=256（含 `-1`） | bf16 | 容差内通过 |
| L0-5 scaling | T=8,K=2,H=256, scale=2.827 | bf16 | 容差内通过 |
| L0-6 代表 prefill | T=4096,K=8,H=7168 | bf16 | 容差内通过 |
| L0-7 代表 prefill | T=4096,K=8,H=3072 | bf16 | 容差内通过 |
| L0-8 多运行稳定 | T=64,K=8,H=2048 | bf16 | 6 次逐位一致 |

（分层套件 L1/L2/Boundary 由 `tilelang-op-develop` 承接，不在此枚举。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **API**：`T.vcast` 的 `round_mode` **必须按关键字传**（`T.vcast(src, dst, round_mode="rint")`）；位置第 3 参是
  `size`（`tilelang/language/customize_npuir.py` L750 `npuir_cast(src, dst, size=None, round_mode="rint")`），
  传字符串位置参会触发 codegen `StringImmNode case not supported`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`
  §1 将 `round_mode` 列为第 3 参，**与前端签名不一致**，实测踩坑）。处理：一律关键字传递。
- **工具链**：`T.Pipelined` 体内 runtime `if` 触发 SIGSEGV（CG-2026-0001a）→ 用 `T.serial`；标量消费向量产出不可同步
  （CG-2026-0027）→ `fwd_idx` 读 GM；无标量原子 / fetch-add（CG-2026-0029/0030）→ 不依赖原子（本单元确实不需要）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| `StringImmNode case not supported` | `T.vcast` 位置传 `round_mode` | 编译失败 | 用 `round_mode=` 关键字 |
| codegen SIGSEGV | `T.Pipelined` 体内 runtime `if` | 编译崩溃 | K 循环用 `T.serial` |
| 精度错 / 随机 stale | 标量读本 scope 向量产出的 UB 值 | 多运行不一致 | 标量读 **GM**（跨 kernel） |
| UB 溢出 | H 很大（>16384）或多 buffer | 编译失败 | 复核 §4.5；auto-multi-buffer 已关闭 |
| 尾块越界 | `T` 非 `nk` 整数倍 | 越界读/写 | `if token < T` 守卫 |

### 9.3 特殊场景处理

- **EP 模式**（`fwd_idx==-1`）：`if raw >= 0` 跳过（等价零贡献，D-1 E1）。
- **top_k=1**：K 循环单次（等价直接缩放复制）。
- **T < 物理核数**（smoke T=1）：`nk=min(T,40)`，无核内串行。
- **非整除 token**：尾守卫。

### 9.4 风险清单

| # | 风险 | 说明 | 缓解 |
|---|------|------|------|
| R-1 | **实测带宽低于地板** | kimi-k2-prefill 866 GB/s < MTE2 地板 1.26 TB/s（`CONST-mte2-degradation`）；qwen3-30b 272 GB/s（小 H 更差） | Stage 4：token 批处理摊薄 op 开销（§1.6.0 L3）/ 受控启用 multi-buffer / 尝试 pipeline 替代形态 |
| R-2 | **无 pipeline / 无多缓冲 → 无重叠** | `T.serial` + auto-multi-buffer 关闭 → K 段 DMA 与向量算术单核串行 | §1.6.0 裁决计划 + Stage 4；若显式多版本 UB 可行则启用 |
| R-3 | **UB 预算随 H 线性增长** | H>16384 时 4 buffer 超 192KB | 工厂按 H 特化；超限时降级（如省 `ub_out` 复用 `ub_src`）并复核 |
| R-4 | **`T.vcast` 文档-签名不一致** | 位置传参踩坑 | §9.1 已记录；关键字传参 |
| R-5 | **本次 D-3 延迟为单进程 20-iter 均值** | 非 msprof，含 host 侧波动 | Stage 3/4 以 msprof 复核；主选/备选差 <2% 不影响裁决 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_unpermute/_make_unpermute_kernel/
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # D-1 等价性机器验证（E1）
├── history_version/
│   ├── design_probe_unpermute.py        # D-3 可行性/精度/稳定探针
│   └── design_probe_unpermute_perf.py   # D-3 稳态延迟探针
├── RETROSPECTIVE.md             # Stage 1 复盘（自进化钩子）
└── _make_unpermute_kernel.py    # 算子实现（Stage 3 待产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | D-1 等价验证（ALL_PASS） |
| `history_version/design_probe_unpermute.py` | 已完成 | D-3 探针（20/20 通过） |
| `history_version/design_probe_unpermute_perf.py` | 已完成 | D-3 稳态延迟 |
| `RETROSPECTIVE.md` | 已完成 | Phase 8 复盘 |
| `_make_unpermute_kernel.py` | 待实现 | Stage 3 算子实现 + L0 测试 |

### 10.3 命名规范

- 项目目录名: `moe_unpermute`；算子目录名: `_make_unpermute_kernel`；实现文件: `_make_unpermute_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ D-1 等价验证（verify_equiv.py）/ D-3 探针
3. ⬜ 算子实现（`_make_unpermute_kernel.py`）+ 与 golden 精度比对（§8.3）

