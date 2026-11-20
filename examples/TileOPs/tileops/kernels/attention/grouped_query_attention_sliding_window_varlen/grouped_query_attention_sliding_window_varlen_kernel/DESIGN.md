# `_gqa_sw_fwd_varlen_kernel` 算子设计文档（GroupedQueryAttentionSlidingWindowVarlenFwdOp · GPU→NPU 迁移 · Expert 模式 · 两相位 S/P 物化 + **per-sample 变长（packed NHD）** 滑动窗口带状掩码）

> **迁移任务**（scenario=migration, migration_mode=harness）。源算子：`/mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_varlen_fwd.py` 的架构通用工厂 `_gqa_sw_fwd_varlen_kernel`（→ `GQASlidingWindowVarlenFwdKernel`）。同文件 Hopper 专属变体 `_gqa_sw_fwd_varlen_wgmma_pipelined_kernel` 按 add-npu-op K5「NPU 折叠架构特化」排除，本设计不迁移。
>
> **本文件是增量设计**：骨架 = 同族姊妹算子 **定长版** `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/DESIGN.md`（`GroupedQueryAttentionSlidingWindowFwdOp`，BSHD 定长 GQA + 双侧滑动窗口，**Expert 两相位**，已 22/22 通过 + 集成一次通过）。凡与 varlen 无关的结论**直接沿用并标注「沿用定长版」**；设计精力集中在 **varlen 增量轴**（§0.8 汇总「沿用清单 vs varlen 新增清单」）。
>
> 本文件依据 `tilelang-op-design` skill 的 **Phase M0（源算子三问解读）→ Phase R（算法调研四问）→ Phase M1（硬件耦合性判定与 NPU 算法重设计）→ Phase 1–8** 生成。
>
> 工具链（设计期探针实测戳）：`source /mnt/workspace/Agentkj/env_npuir.sh`；tilelang `0.1.2+76e45509964c313ceca43930cceff43c7820551c`（dev root `/mnt/workspace/Agentkj/build`）；Ascend **910B3**（`NPUUtils.get().get_aicore_num()` 实查 **20** AIC / 40 AIV）；CANN 9.0.0；2026-10-16。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（packed NHD 变长 GQA + 滑动窗口 flash 前向，FA3 bottom-right 因果对齐）：

```
o   = softmax(scale · q @ k^T + M) @ v           # packed NHD，online softmax
M   = 逐元素掩蔽（下述条件「不满足」处 −inf / 0）
per-sample bz:  q ∈ [cu_q[bz], cu_q[bz+1])，k/v ∈ [cu_k[bz], cu_k[bz+1])
q_len = cu_q[bz+1]−cu_q[bz]；kv_len = cu_k[bz+1]−cu_k[bz]；offset = kv_len − q_len
局部 q_pos = bx·bm + i（样本内），局部 k_pos = k·bn + j（样本内）
valid(q_pos, k_pos) ⟺
    (¬is_causal ∨ k_pos ≤ q_pos + offset)                 # 因果上界（FA3 右下对齐）
  ∧ (wl < 0      ∨ k_pos ≥ q_pos + offset − wl)           # 左窗下界
  ∧ (wr < 0      ∨ k_pos ≤ q_pos + offset + wr)           # 右窗上界
  ∧ (q_pos < q_len) ∧ (k_pos < kv_len)                    # q_oob / k_oob
scale = dim^{−1/2}；is_causal 默认 true；window_size_left/right 默认 −1（= 无限制）
```

> **与定长版的语义差异（varlen 核心）**：
> 1. **packed NHD 布局**——q `(total_q,H,D)`、k/v `(total_k,H_kv,D)`，batch 维被折叠进单一 token 轴；样本 bz 占据 `[cu_q[bz], cu_q[bz+1])`。
> 2. **per-sample 运行期长度**——`q_len/kv_len/offset` 由 `cu_seqlens_q/cu_seqlens_k`（**int32**）在运行期从 GM 读出；定长版是编译期常量。
> 3. **`offset = kv_len − q_len`（FA3 右下对齐）**——因果上界是 `q_pos + offset` 而非定长版的 `q_pos`；`offset = 0` 退化为标准 causal，`offset > 0` 为 KV-cache 形态。**掩码的 K_up/K_lo 阈值整体平移 +offset**。
> 4. **lse 布局 `(H, total_q)`**（head-major、token 内层），定长版是 `(B,H,S)`。
> 5. **ghost q-block**——grid 按编译期 `max_seqlen_q` 展开（`ceildiv(max_seqlen_q, bm)`），短样本的 `bx·bm ≥ q_len` 块为空转（源以 `if bx·bm+i < q_len` 写出守卫）。

> **角点说明（`is_causal ∧ wr ≥ 0`，源行为与通用式不一致）**：源 `is_causal and has_window` 分支（源码 L61–72）在 causal 下**只施加 causal|left、忽略 right**（且 `k_end` 不含 `wr`，L186–188）。manifest 与测试域均无该组合（全部 causal 用例 `wr=−1`），故目标域不受影响；本设计按其实际行为分派 trace，§9.3 记明该组合不承诺。

**I/O 契约**（源码 L140–150 与 wrapper/Op 亲自核对）：

| 项 | 内容 |
|----|------|
| q | `(total_q, H, D)`，float16 / bfloat16，packed NHD 连续 |
| k / v | `(total_k, H_kv, D)`，`same_as(q)`，`H % H_kv == 0`（GQA） |
| cu_seqlens_q / cu_seqlens_k | `(B+1,)`，**int32**（与 q/k/v 不同 dtype） |
| max_seqlen_q | `T.int32` 标量（**grid 尺寸**；`≥` 实际最大 q_len） |
| output | `(total_q, H, D)`，`same_as(q)`（`out_idx=[6,7]` 的第 6 号输出） |
| lse | `(H, total_q)`，**float32**（第 7 号输出；**Op wrapper `GroupedQueryAttentionSlidingWindowVarlenFwdOp.forward` 只返回 output，丢弃 lse**——源 Op L248 `output, _ = self.kernel.forward(...)`） |
| 中间 dtype | `accum_dtype = "float"`（fp32）——S、online 统计量在 fp32 域；`acc_s_cast` 量化回**输入 dtype** 后进第二个 GEMM |

**语义要素表**：

| 要素 | 结论 | 依据 |
|------|------|------|
| 数学语义 | 上四条件带掩码的 per-sample softmax attention；offset 把因果窗右下对齐到样本尾部 | 源码 L60–100 四掩码路径 + Op docstring L58–68 |
| I/O 契约 | 见上表；输出 shape/dtype 与 q 同；lse `(H,total_q)` fp32 | 源码 L140–150；`migration_meta.json` `source_output_shape=(total_q,H,D)` |
| 规约语义 | per-sample softmax 沿 **kv 轴**（样本内 k_pos）规约；online 累加**按 kv 块顺序**推进（`k_start..k_end`），块内 `reduce_max/sum(dim=1)` | 源码 L206–237 |
| dtype 语义 | S/统计量 fp32；**P 量化到输入 dtype**（`T.copy(acc_s, acc_s_cast)`，源码 L220）；`acc_o` 全程 fp32；lse fp32 | 源码 L158–165, L220 |
| 边界语义 | `−T.infinity` 掩码 + scores_max 有限下限 clamp（`−1e38`，`make_online_softmax_with_mask_guard` L66），防 `exp2(−inf−(−inf))=NaN`；`loop_count = max(k_end−k_start, 0)` 允许 0；lse `= log2(logsum) + scores_max·scale`；ghost 行以 `q_oob` 掩蔽 | 源码 L60–100, L183, L203, L236 |
| host 侧语义 | 无预处理/后处理（q/k/v 直接进 kernel；`contiguous()` 为契约保证；Op 侧校验 cu_seqlens 单调/起点为 0/不超张量长） | 源码 L168–179；Op L212–246 |
| **offset 语义** | `offset = kv_len − q_len`（样本内常量）；`offset=0` 标准 prefill，`offset>0` KV-cache | 源码 L177 + docstring L6–11 |

**完成标准核对**：§8.1 golden 可直接由本节语义写出（与源仓测试 `ref_program` 等价：逐样本切片 → expand KV heads → 带 offset 的四条件掩码 → softmax → @v → 拼回 packed）。

### 0.2 源算子输入输出

见 §0.1 表。**额外发现（影响 §0.6 接口）**：源算子同时产出 `output` 与 `lse`（`out_idx=[6,7]`），而 Op wrapper 丢弃 lse；NPU 侧 kernel 仍须产出 lse（保持 kernel 契约与 wrapper 兼容），但 lse 不进入 Op 精度门（Op 只校验 output）。**lse 布局为 `(H, total_q)`**（与定长版 `(B,H,S)` 不同）——见 §0.6 V7。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 `@T.prim_func _gqa_sw_fwd_varlen_main`，L140–239）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|---|------|------|------|------|----------|
| S1 | grid 划分 | `T.Kernel(ceildiv(max_seqlen_q,bm), H, B, threads=)` 三维 | max_seqlen_q | (bx,by,bz) | 任务 = (样本内 q-block, head, 样本)；**bx 为样本内局部块** |
| S1b | **per-sample 边界（运行期）** | `q_start=cu_q[bz]`、`kv_start=cu_k[bz]`、`q_len=cu_q[bz+1]−q_start`、`kv_len=cu_k[bz+1]−kv_start`、`offset=kv_len−q_len` | cu_q/cu_k（GM int32） | 5 个运行期标量 | **varlen 核心**：长度与 offset 运行期读出 |
| S2 | 状态初始化 | `clear(acc_o)`、`clear(logsum)`、`fill(scores_max, −inf)` | — | acc_o, logsum, scores_max | online 初值 |
| S3 | Q 装载 | `T.copy(q[q_start+bx·bm : q_start+(bx+1)·bm, by, :], q_shared)` | GM q | SMEM `[bm,dim]` | **packed 地址 = q_start + 块偏移** |
| S4 | KV 循环范围（运行期） | causal→`k_end=ceildiv(min(kv_len, offset+(bx+1)bm), bn)`；`wr≥0` 非 causal→`+wr`；否则 `ceildiv(kv_len,bn)`；`wl≥0`→`k_start=max(0,offset+bx·bm−wl)//bn` 否则 0；`loop_count=max(k_end−k_start,0)` | 运行期 q_len/kv_len/offset | 循环边界 | 带状裁剪（**运行期版本**） |
| S5 | mask 预填 | `T.Parallel(bm,bn)` 逐元素 `if_then_else(掩码, −inf, 0)` 写入 `acc_s`（四编译期路径含 q_oob/k_oob） | (bx,k_idx,q_len,kv_len,offset) | acc_s | M |
| S6 | QK^T | `T.gemm(q_shared, k_shared, acc_s, transpose_B=True, FullRow)`（`initC=False` 累加 S5 掩码） | q,k tile | acc_s | `q@k^T + M` |
| S7 | online max | 单调 max + clamp `≥−1e38` | acc_s | scores_max | 数值稳定化 |
| S8 | α 缩放因子 | `exp2(m_prev·scale − m·scale)` | 行向量 | scores_scale | rescale |
| S9 | P = exp2 与行和 | `acc_s = exp2(acc_s·scale − m·scale)`；`reduce_sum` | acc_s | acc_s, scores_sum | softmax 分子/分母 |
| S10 | logsum 递推 | `logsum = logsum·scores_scale + scores_sum` | — | logsum | online 递推 |
| S11 | P 量化 + rescale acc_o | `T.copy(acc_s, acc_s_cast)`；`acc_o·=scores_scale[i]` | acc_s | acc_s_cast, acc_o | 量化点 + rescale |
| S12 | PV | `T.gemm(acc_s_cast, v_shared, acc_o, FullRow)` | P, v tile | acc_o | `P@v` |
| S13 | epilogue + 边界守卫 | 逐元素 `if bx·bm+i < q_len: output[q_start+bx·bm+i, by, j] = (logsum>0 ? acc_o/logsum : 0)`；`lse[by, q_start+bx·bm+i] = log2(logsum)+scores_max·scale` | — | output, lse | 归一化 + lse + **ghost 行守卫** |

**数据流图（源 GPU 视角）**：

```
GM[cu_q/cu_k (int32)] ──标量读──> q_start/kv_start/q_len/kv_len/offset (运行期标量)
GM[q] ──T.copy(slice q_start+..)──> SMEM[q_shared (bm,dim)] ─────────────────┐
GM[k] ──T.copy(slice kv_start+..)─> SMEM[k_shared (bn,dim)] ─T.gemm(transpose_B)─> frag[acc_s (bm,bn) fp32]
                                                                   │  (+ S5 掩码预填)
    frag[acc_s] ──reduce_max─vmax─clamp─exp2(·scale−m·scale)─reduce_sum ─> rows[bm] (m, α, ℓ)
    frag[acc_s] ──T.copy(cast)──> frag[acc_s_cast (bm,bn) 输入 dtype] ─┐
GM[v] ──T.copy(slice kv_start+..)─> SMEM[v_shared (bn,dim)] ─T.gemm─> frag[acc_o (bm,dim) fp32] ◄┘(×α)
    frag[acc_o] ──÷ℓ (ghost 守卫)──> GM[output[q_start+.., by, :]]；rows ──log2+·──> GM[lse[by, q_start+..]]
```

**循环与并行结构**：三维 grid（`ceildiv(max_seqlen_q,bm)` × H × B）× `threads=128`；KV 循环 `T.Pipelined(loop_count, num_stages)`；元素并行由 `T.Parallel` 与 fragment 上的 mma 承载。

**host 侧逻辑**：`functools.lru_cache(maxsize=None)` 工厂缓存、`heads % heads_kv` 校验、`make_log2e_scale(dim)` 常量折叠、`super().__init__()`/`init_config`（config = block_m/block_n/num_stages/threads）、`total_q/total_k` **运行期**从 `q.shape[0]/k.shape[0]` 解析。

### 0.4 优化手段解读（为什么快）

**逐项识别清单**（与定长版 §0.4 同族；varlen 差异列注明）：

| # | 优化手段 | 源码特征 | 目的 | 依赖源硬件特性 | 初判 |
|---|----------|----------|------|----------------|------|
| OP1 | **flash online softmax** | S7–S10 running max/sum + α rescale | 消 O(S²) 中间量 | **无（纯算法层）** | 可移植 |
| OP2 | **滑动窗口带状裁剪（loop range）** | S4 `k_start`/`k_end` + `loop_count` | 只处理窗内 KV 块 | 无 | 可移植（**算法核心**；varlen 版含 **运行期 offset**） |
| OP2b | **ghost 块空转** | S13 `if bx·bm+i < q_len` 写出守卫 | 短样本的 padding 块不写 | 无 | 可移植（**varlen 新增**） |
| OP3 | SMEM tiling | `alloc_shared` + 分块 `T.copy` | 减 GM 重复访问 | shared memory | 硬件强相关 |
| OP4 | coalesced 访存 | 尾轴 `:dim` 连续 | 带宽 | memory coalescing | 硬件强相关 |
| OP5 | Tensor Core mma | `T.gemm`（fragment dst） | 矩阵算力 | tensor core | 硬件强相关（NPU Cube 等价） |
| OP6 | `T.Pipelined` 多级软件流水 | KV 循环 `num_stages` | 装载/计算重叠 | cp.async | 硬件强相关（NPU MTE 等价） |
| OP7 | fragment 寄存器累加 + 融合 epilogue | `alloc_fragment`，epilogue 原地 | 数据驻留、消中间回写 | 大寄存器堆 | 硬件强相关（NPU L0C/UB 等价） |
| OP8 | warp 归约 | `reduce_max/sum(dim=1)` on fragment | 低延迟行归约 | warp shuffle | 硬件强相关（NPU 行归约等价） |
| OP9 | `GemmWarpPolicy.FullRow` | gemm 参数 | warp 划分 | CUDA warp | 无 NPU 对应 |
| OP10 | three-dim grid + threads | `T.Kernel(m,H,B,threads=)` | CTA 级并行 | GPU grid/block | NPU 一维 Kernel |
| OP11 | exp2 + LOG2E 预乘折叠 | `make_log2e_scale(dim)` | 用 exp2 硬件指令 | ffma 融合 | 换底等价可移植 |
| OP12 | mask 预填 + initC 累加融合 | S5+S6 | 省独立 mask pass | tensor core C 累加 | 需重判 |
| OP13 | `TL_ENABLE_FAST_MATH` + `compile_flags=[-O3,-DENABLE_BF16]` | `@tilelang.jit` | CUDA 编译链优化 | CUDA 编译链 | NPU 无此 pass |
| OP14 | `lru_cache` 工厂缓存 | `functools.lru_cache` | 复用编译产物 | 无 | 可移植（host） |
| OP15 | bf16 直连 tensor core | `-DENABLE_BF16` | bf16 算力/带宽 | bf16 mma | NPU Cube bf16 等价 |
| OP16 | **per-sample 运行期长度（cu_seqlens 标量读）** | S1b | 免 padding、支撑变长 | 无（**纯算法层**） | 可移植（**varlen 核心**；NPU 承载见 V1/探针 P-V1） |

### 0.5 硬件耦合性分析与 NPU 适配决策（Expert 模式口径）

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？**本轮判定输入含 §1.6.0 调研结论**：源算法（单遍 per-block online-softmax 链 + 运行期变长）与调研候选（两相位 S/P 物化等）同台比较——调研选定两相位结构后（见 §1.6.0），§0.4 各优化手段的处置对照该结论逐项重判。**与定长版的关键差异**：多出 **OP2b ghost 块空转**、**OP16 运行期 per-sample 长度/offset**（源 GPU 侧为标量 GM 读）与 lse `(H,total_q)` 布局。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（公式 / 四条件掩码含 offset / P 量化点 / dtype / lse log2 域 / packed 输出 shape） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| **OP2 滑动窗口带状裁剪（loop range）** | 算法 | 无 | — | **保留** | 纯算法层——`k_start/k_end`（含**运行期** `kv_len/offset`）承载为本设计的 KV 循环范围 `[k_start,k_end)`（§0.6 V3） |
| **OP2b ghost 块空转** | 算法 | 无 | — | **保留** | 纯算法层——源式 `q_oob` 写出守卫；NPU 承载为 `real_m ≤ 0 ⇒ loop_count=0 + 行截断写出`（§0.6 V1/V7） |
| **OP16 per-sample 运行期长度（cu_seqlens 标量读）** | 算法 | 无 | — | **保留（承载方式重设计）** | 源 GPU 侧为逐线程标量 GM 读；NPU 承载 = **核内 GM 标量读 `cu[i]`**（探针 **P-V1 实测 maxdiff=0.0**，`cu` 为 int32 标量操作数；备选 copy-to-UB 路径 **P-V1 实测 BLOCKED**）——§0.6 V1/V6 |
| OP1 online softmax 递推 | 算法 | 无 | — | **保留** | 纯算法层；两相位下递推在 Vector pass-1 逐块保持，rescale 因子存 UB `scales[]` 延迟到 pass-2 回放（V5） |
| S7 scores_max 有限下限 clamp | 算法 | 无 | — | **保留** | `T.vmax(m, −1e38, m)`（源 NaN guard 逐字保留）——探针 P-V2 实测 sentinel 精确 |
| q 驻留复用 + k/v 流式分块 | 算法 | 无 | — | **保留** | 承载走 V1/V2/V3/V4 |
| OP3 SMEM tiling | 优化 | CUDA SMEM | 有（L1 512KB / UB 192KB） | **等价替换** | Cube 侧 `T.alloc_L1`（l1_a/l1_b 生命周期复用，§4.3）；§5.3 映射表「shared memory → L1/UB」 |
| OP5 Tensor Core GEMM | 优化 | NVIDIA tensor core | 有（Cube `T.gemm` + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a,l1_b,l0_c, initC=True, b_transpose=True, size=[tmc,dim,tnc])`（QK^T）/ `…size=[tmc,tnc,dim]`（PV）（T.gemm.md §2.1；定长版同构） |
| OP6 `T.Pipelined` 软件流水 | 优化 | cp.async | **意图由两相位结构承接** | **重新设计** | → V5：装载/计算重叠不再依赖 per-block 乒乓——Cube pass-1 无等待直跑（MTE2/MTE1/C/FIX 四通道天然重叠）、pass-2 逐块等 P（PL-1.9-twophase；定长版同款） |
| OP7 fragment 寄存器累加 + 融合 epilogue | 优化 | CUDA 寄存器堆 | **无直接等价**（L0C 不能被 v 算子操作；L0C 出数 dst 仅 GM——`docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1`） | **重新设计** | → V5：S 与每块 PV 部分和经 GM ws 跨引擎往返（API 强制，CG-2026-0005）；acc_o 驻留 Vector UB，pass-2 延迟 rescale 累加 |
| OP8 warp 归约 + `T.Parallel(bm)` 标量 max | 优化 | warp shuffle | 有（Vector 行归约 + vmax） | **等价替换** | `T.reduce(ub, dst, dims=[1], reduce_mode="max"/"sum", clear=True)`（T.reduce.md §2.2.2；**clear 显式 True**——§2.3 静默错误条款） + `T.vmax` |
| 行级 rescale / logsum 更新 | 优化 | warp 标量并行 | 有（v 算子行广播） | **等价替换** | `T.vmul(ub, ub_rowvec, ub)`（[M,N]×[M,1] 行广播，T.vmul.md §2.2.2）+ `scales[]` 切片存取 |
| **OP12 mask 预填 + initC 累加融合** | 算法 | tensor core C 累加 | 有（initC），**但两相位 gemm1 恒 initC=True 逐块覆写**；且 developer 实测 `T.Parallel` 谓词预填是 aiv_scalar 主瓶颈 | **重新设计** | → VM-mask/VM-bnd：掩码移至 **Vector pass-1、S 到达后 softmax 前**，**全向量算术惩罚式**（PL-1.11：vcmp 全形态标量化）——**含 offset 平移**（varlen 新增）+ **per-row 列 OOB 惩罚**（varlen 新增，探针 P-V2 7/7 bit-exact） |
| OP9 `GemmWarpPolicy.FullRow` | 优化 | CUDA warp | 无（NPU 无 warp） | **舍弃** | Cube 矩阵并行由分形承担，`b_transpose=True` 即完整语义 |
| OP10 three-dim grid + threads | 优化（结构） | GPU grid/block | **无**（`T.Kernel` 一维约束） | **重新设计** | → V1：一维 persistent `T.Kernel(20, is_npu=True) as (kernel_id, subid)` + 分核策略（§5.5）；`threads` 参数删除（K3/K9，wrapper 已无该参数） |
| OP13 FAST_MATH + compile_flags | 优化 | CUDA 编译链 | 无 | **舍弃** | npuir 分支无此 pass；改设 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` + `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`（防编译器重排手动两相位流，§7.3） |
| OP11 exp2 + LOG2E 折叠 | 优化 | CUDA ffma | 有等价（`T.vexp` e 底） | **等价替换（换底方式变更）** | → VM-e：e 域 `T.vexp` 直接实现（数学恒等）；lse 保持 log2 域（`T.vlog2`） |
| OP15 bf16 直连 tensor core | 优化（dtype 承载） | GPU bf16 mma | 有（T.gemm.md §2.2.1 bf16 √；PL-1.8） | **等价替换** | Cube bf16 直连；`P` 量化 `T.vcast(f32→bf16,"rint")`；**ws 承载 fp16: f16/f16/f16、bf16: f32/bf16/f32**（V5） |
| OP4 coalesced 访问 | 优化 | memory coalescing | 等价（尾轴连续 `T.copy`） | **等价替换** | 全部 GM↔L1/UB 搬运用 slice-form `T.copy`（**packed NHD 定头切片**——CG-2026-0004） |

**统计**（每行恰计一次，合计 = 表体 20 行）：**保留 7 项**（计算语义〔语义〕、OP2 带状裁剪〔算法·核心〕、OP2b ghost 空转〔算法·varlen 新增〕、OP16 运行期长度〔算法·varlen 核心〕、OP1 online 递推、S7 clamp、q 驻留/kv 流式分块）/ **等价替换 7 项**（OP3、OP5、OP8、行级 rescale、OP11、OP15、OP4）/ **重新设计 4 项**（OP6、OP7、OP12、OP10）/ **舍弃 2 项**（OP9、OP13）。7+7+4+2 = 20 ✓。（`threads` 参数并入 OP10 行处理〔移除〕，不单独计行。）

### 0.6 NPU 算法重设计（Expert 模式）

> **总体结构（§1.6.0 调研选定；沿用定长版两相位）**：**一维 persistent Kernel（20 核）× 双 Scope（Cube/Vector）× 两相位独立大循环 × per-sample 滑动窗口带状 KV 范围 × packed NHD 运行期寻址**——结构原型 = 定长版 `_gqa_sw_fwd_kernel`（同族已迁移，Expert 两相位）。每个**逻辑任务 = (样本内 q-block, query-head, batch 样本)**；任务内：**Cube pass-1** 串行计算整条带上的全部 S 块（QK^T，Q-hoist）物化到 per-core GM workspace；**Vector pass-1** 逐块消费 S（vcast → **offset 平移的双侧算术惩罚掩码 + per-row 列 OOB 惩罚** → prescale → online max/α/ℓ 递推 → P 量化）；**Cube pass-2** 逐块消费 P 计算 PV 部分和；**Vector pass-2** 逐块延迟 rescale 累加 acc_o；epilogue 归一化 + transpose-free lse 写出。跨引擎协同 = per-带内块下标 flag（S/P/O 三次握手复用同一 id）+ 任务边界 FLAG_TASKDONE。

**与定长版的异同 / 复用点**（⭐ 本文核心增量视图）：

| 维度 | 定长版（骨架来源） | 本算子（varlen） | 处置 |
|------|-------------------|------------------|------|
| 语义 | BSHD 定长 GQA + 双侧滑动窗口；`k_pos ≤ q_pos` | packed NHD per-sample + **offset 右下对齐** `k_pos ≤ q_pos+offset` | **语义扩展**（V2） |
| 整体结构 | 一维 persistent 20/24 核 + 双 Scope + 两相位独立大循环 + S/P/O GM ws + per-块 flag | **完全复用** | ✅ 直接复用 |
| 任务划分 | `ceildiv(S,bm)×H×B`，bx 全局 | `ceildiv(max_seqlen_q,bm)×H×B`，**bx 样本内局部** | 🔁 **V1（ragged 任务映射）** |
| 任务解码 | `cid→(bx,by,bz)` 纯编译期映射 | `cid→(bx,by,bz)` + **逐任务运行期读 cu_seqlens 得 q_start/kv_start/q_len/kv_len/offset** | 🔁 **V1/V6（本算子最大新增）** |
| 带状 KV 范围 | `k_start/k_end` on 编译期 `S` | **同式移植但 `kv_len/offset` 为运行期值** | 🔁 **V3** |
| 掩码 | 双侧 `K_up/K_lo`（`s_lo` 基准）算术惩罚式；两段式带状分界 | **`s_lo → bx·bm+offset` 平移**；**+ 运行期 per-row 列 OOB 惩罚** | 🔁 **V2/VM-mask/VM-bnd（本算子最高新颖度）** |
| flag 协议 | id = 带内块局部下标 l；预算由 `(bm+wl+wr)/bn` 守卫 | **同款**；但 `loop_count` 运行期 → **L_max 用解析上界**（非编译期扫描） | 🔁 **V5（预算守卫改解析式）** |
| 尾块 / 尾带 | 契约域；`has_tail` trace-time 门控 | **运行期 kv_len 非整除 ⇒ 尾带恒可能 ⇒ 机制常开**（零初始化 + per-row 列 OOB） | 🔁 **V4（常开化）** |
| Q/K/V 装载 | BSHD 定头 slice-form `T.copy`（CG-2026-0004） | **packed NHD 定头 slice-form `T.copy`**（地址含运行期 `q_start/kv_start`） | 🔁 **V6** |
| online softmax | e 域 vexp + scales 延迟回放 + ℓ clear=True | **完全复用** | ✅ |
| fp16/bf16 路径 | fp16: f16/f16/f16；bf16: f32/bf16/f32 | **完全复用** | ✅ |
| lse | `[B,H,S,1]` 视图 transpose-free | **`(H,total_q)` → `[H,total_q,1]` 视图** transpose-free | 🔁 **V7** |
| 物理核数 | 20（910B3，实查） | **20（同机，实查）** | ✅ |

**复用点一句话**：除 **ragged 任务映射（V1）**、**per-sample 动态掩码（V2）**、**运行期带状范围（V3）**、**常开尾带（V4）**、**运行期 flag 预算（V5）**、**packed 寻址（V6）**、**lse 布局（V7）**、**接口（V8）** 八处为 varlen 新增/改写外，**其余结构（两相位分工、ws 三族、scales 延迟回放、双 AIV 行半分、Q-hoist、e 域 softmax、fp16/bf16 双 trace、size-form 尾钳位、sync shim）逐语句沿用定长版**——该母本已 22/22 通过 + 集成一次通过，是低风险高置信的载体。

**重设计项 V1: 三维 grid + 编译期任务映射 → 一维 persistent + ragged 任务解码 + 运行期 per-sample 长度**

- **源方案**：`T.Kernel(ceildiv(max_seqlen_q,bm), H, B, threads=)` 三维 grid；每 (bx,by,bz) 一个 CTA；样本边界由 `cu_seqlens[bz]` 在 CTA 内标量读。
- **NPU 新算法**：一维 `T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid)`，**NUM_KERNELS = 20 = `NPUUtils.get().get_aicore_num()` 实查值**（§5.5）。逻辑任务数 `num_logical = ceildiv(max_seqlen_q,bm) × H × B`；任务解码 `cid = T.min(task_id·20 + kernel_id, num_logical−1) → (bx = cid // HB, by = (cid // B) % H, bz = cid % B)`，`HB = H·B`；核内 persistent 串行 `num_local = T.ceildiv(num_logical − kernel_id, 20)`。**逐任务运行期读**（per-Scope，Cube/Vector 各自读同一批值 → 一致）：
  ```
  q_start = cu_q[bz] ; kv_start = cu_k[bz]
  q_len   = cu_q[bz+1] − q_start ; kv_len = cu_k[bz+1] − kv_start ; offset = kv_len − q_len
  real_m  = T.min(bm, q_len − bx·bm)          # 本块真实行数；ghost 块 ≤ 0
  ```
- **语义保持论证**：(样本内 q-block, head, 样本) 逻辑任务完全独立——读写不相交的 `output[q_start+…]` / `lse`；q/k/v 只读共享（GQA 下多 query head 读同一 kv head 切片）。任务到核映射与核内串行顺序不改变任何数值——**数学等价**；clamp 导致的重复任务写 bit-identical 输出。**ghost 块（`real_m ≤ 0`）**：`loop_count = 0`、跳过 Q 装载、行截断写出（rm=0 不写）——与源 `q_oob` 守卫同效（这些块无对应输出 token，packed 输出只含真实 token）。
- **设备证据（探针 P-V1）**：`cu_seqlens` 的**核内 GM 标量读**路径实测 **PASS（maxdiff=0.0）**——运行期标量同时用于 (a) 动态 slice 起点 `q[q_start+bx·bm:…]`、(b) 动态 extent（`real_m ≤ bm` 钳位）、(c) 动态输出地址、(d) 动态 `T.serial` 循环界 `T.ceildiv(q_len,bm)`（含 ghost 块 real_m=0）；**备选「copy cu→UB 再标量索引」路径实测 BLOCKED**（`InternalError: cannot find variable cub`）→ 采纳直接 GM 标量读。

**重设计项 V2: per-sample 动态掩码（offset 平移 + 常开列 OOB）——本算子最高新颖度** ⭐

- **源方案**：`T.Parallel(bm,bn)` 逐元素谓词写 `acc_s[i,j]=if_then_else(掩码,−inf,0)`（四编译期路径，含 `q_oob`/`k_oob`），`k_end`/mask 用运行期 `q_len/kv_len/offset`。MHA/定长迁移实测该形态为 aiv_scalar 主瓶颈。
- **NPU 新算法**：掩码移至 **Vector pass-1、S vcast 到 f32 之后、softmax 之前**，**全向量、NaN 无关、无 vcmp**（PL-1.11 第五轮：`vcmp` int16 全形态标量化；本容器 P-V2 复现无 warning）。**算术惩罚式 + offset 平移 + per-row 列 OOB**：
  ```
  diff  = j − i                       # [half,bn] f32，arange 一次性物化（任务无关，每核构建一次）
  # 因果/右窗上界：valid ⇔ diff ≤ K_up，K_up = (bx·bm + row0 + offset) − n_lo   （offset 平移）
  pu = clamp(diff − K_up, 0, 1)
  # 左窗下界：valid ⇔ diff ≥ K_lo，K_lo = (bx·bm + row0 + offset − wl) − n_lo
  pl = clamp(K_lo − diff, 0, 1)                     # 以 -(diff−K_lo) 实现（vsub 标量恒作 src2）
  # 列 OOB：valid ⇔ j < tn_real ⇔ diff ≤ (tn_real−1) − i（**per-row 阈值**）
  poob = clamp(diff − ((tn_real−1) − i), 0, 1)
  pen  = vmax(pu, vmax(pl, poob))                   # diff 整数 → pen ∈ {0,1} 精确
  S    = S + pen·(−1e38)                            # 掩蔽位精确 −1e38 → P = +0.0
  ```
  其中 `row0` = per-AIV 行偏移（自适应半分），`n_lo = (k_start+l)·bn_eff`，`tn_real = min(bn_eff, kv_len − n_lo)`。
- **语义保持论证**：① **作用点相同**（QK 之后、softmax 之前）；② **值无关**——pen 由 (i,j,n) 决定；掩蔽位 `S'=S+(−1e38)`（f32 大数吸收有限 S）→ `P=e^{−1e38−m}`（m 经 clamp 有限）→ **逐位 +0.0** ≡ 源 `e^{−inf}=0`；③ **offset 平移**：`k_pos ≤ q_pos+offset` ⟺ `diff ≤ (bx·bm+row0+offset)−n_lo`（端点代入：`k=n_lo+j`、`q=bx·bm+row0+i`）——与源 S5 谓词代数等价；④ **列 OOB = per-row 阈值**（varlen 新增关键点）：源 `k_oob ⟺ k_pos ≥ kv_len ⟺ j ≥ tn_real`，是**绝对列**条件，在 diff 空间的等价式必须含行号 i：`j ≥ tn_real ⟺ diff ≥ tn_real − i`（**不可**写成 `diff ≥ tn_real`，否则误掩蔽——**这是探针 P-V2 首版暴露并由修复确认的落点**）；⑤ NaN 免疫——S 来自 L0C gemm 恒有限，唯一 NaN 源为尾带 stale 列，经 V4 零初始化根治。
- **设备证据（探针 P-V2，`history_version/design_probe_varlen_mask.py`）**：主选「offset 平移双侧算术惩罚链 + per-row 列 OOB」**7/7 配置 bit-exact PASS**（有效掩码逐位 + `S'=S+pen·(−1e38)` sentinel 逐位校验 + NaN-free），**零标量化 warning**，其中 offset 以**运行期 GM 标量读**进入阈值（`ku=kid·0.0+(…)` 顶层 Add 形态）、覆盖 KV-cache（offset>0）与 ragged（`tn_real<bn`）组合。

**重设计项 V3: 带状 KV 范围（运行期 `kv_len`/`offset` 版）**

- **源方案**：`k_start = max(0, offset+bx·bm−wl)//bn`（`wl≥0`，否则 0）；`k_end = ceildiv(min(kv_len, offset+(bx+1)bm), bn)`（causal）/ `+wr`（非 causal 右窗）/ `ceildiv(kv_len,bn)`（否则）；`loop_count = max(k_end−k_start, 0)`。
- **NPU 新算法**：完全承接源式的 `k_start`/`k_end`（逐字移植，per-task PrimExpr，**`kv_len`/`offset` 为运行期值**），KV 循环改为 `for l in T.serial(loop_count): n = k_start + l`。**`max(0,·)` 在所有除法之前**（`k_start` 的分子非负）；`k_end`/`k_start` 的除法分子为 `offset+…`（运行期，符号不可静态界定）——**但它们只用于循环边界（非块分类谓词），不进入 §CG-2026-0024 的 `floordiv` 陷阱形态**；**两段式带状分界（VM-bnd）明确采用无除法直接整数比较**（见 VM-bnd）。
- **语义保持论证**：① **被裁剪块 = 恒等更新**——`[0,k_start)` 块全列被 `pl` 掩蔽、`[k_end,·)` 块全列被 `pu` 掩蔽；全掩蔽块贡献 = online 递推恒等元（`α=1`、`Δℓ=0`、`Δo=0`，`verify_equiv.py` M4' 实证）；② 源本身已裁剪（`loop_count`），本设计与源**同界同序**（保序子序列）。

**重设计项 V4: 常开 ragged KV 尾带（运行期 `kv_len % bn ≠ 0`）**

- **源方案**：`k_oob` 谓词 + `T.Pipelined` 谓词装载/回写。
- **NPU 新算法**：`tn_real = min(bn_eff, kv_len − n_lo)`（运行期）；**每个样本的最后一块恒可能 ragged**（manifest 均匀域 `kv_len ∈ {128,512,2048}` 中 512/2048 整除 256、128 不整除 → **smoke 域名触发**；测试域普遍触发）→ 机制**常开**（不定长版为 trace-time 门控）：
  - Vector N 载体（`ub_f16_N`/`ub_f32_N`）**每核一次零初始化**（`T.vbrc(0, ·)`）——保证 stale 列 `[tn_real, bn_eff)` 有限（`TRAP-npuir-stale-ub-tail-columns-nan`：`-1e38` 惩罚**清不掉 NaN**，须源头零化）；
  - per-row 列 OOB 惩罚（V2 的 `poob`）在**全宽** `[half,bn_eff]` 上把 `j ≥ tn_real` 列打成 `−1e38`；
  - Cube 侧 `l1_b`(V) 每核一次 `zbuf` 零初始化（PL-1.11：`P带(0)×V残留(NaN)=NaN` 整列污染）。
- **语义保持论证**：OOB 列（`k_pos ≥ kv_len`）在源被 `k_oob` 掩蔽、在本设计被 `poob` 打成 `−1e38` → **同效**（`e^{−1e38−m}=+0.0`）；零初始化保证 stale 列有限（有 NaN 时 P/lse 全 NaN，数值灾难）——三条件（零初始化 + 全宽掩码 + zbuf）由 PL-1.11 与 `TRAP-npuir-stale-ub-tail-columns-nan` 双证。**P-V2 实测**（partial-width copy + 零初始化 + per-row 列 OOB）**7/7 sentinel 逐位 + NaN-free**。

**重设计项 V5: 两相位跨引擎结构（逐语句沿用定长版）+ 运行期 flag 预算守卫**

- **NPU 新算法（两相位，同定长版）**：每核独立单槽 GM workspace 三族（尺寸 §4.3）：`ws_s [20,bm,pad_band]`（fp16: f16 / bf16: f32）、`ws_p [20,bm,pad_band]`（输入 dtype）、`ws_o [20,bm,dim·L_max]`（fp16: f16 / bf16: f32），`pad_band = L_max·bn_eff`。Cube pass-1 全 S 无等待直跑；Cube pass-2 全 PV；Vector pass-1 掩码+softmax+`scales[]` 存档；Vector pass-2 延迟 rescale 累加；epilogue。
- **flag 预算守卫（varlen 改写）**：定长版以**编译期扫描 bx** 求 `L_max`；varlen 的 `loop_count` 依赖运行期 `kv_len/offset` → **改为解析上界** `L_max_bound = ceildiv(bm + eff_left + eff_right, bn_eff) + 2`（`eff_left = wl if wl≥0 else max_seqlen_q`（无左窗=不裁剪，上界取 max q-block 跨度）, `eff_right = wr if wr≥0 and not causal else 0`），工厂期 `assert L_max_bound + 1 ≤ 16`（`T.set_flag.md §2.1` event_id 0–15）；超限则 `bn_eff += 16` 重算。**依据**：`k_end−k_start ≤ ceildiv((bm+wl+wr)/bn_eff)+2`（带内块数上界，kv_len/offset 项相消）。
- **语义保持论证**：数据流等价——S/P/O 块内容与源 S6/S11/S12 逐块对应；`o = o·α_l + O_partial(l)` 与源 `acc_o = acc_o·α + P@v` 代数同构（仅累加分组变化，fp32 ulp 级，§8.2 容差覆盖）；块间 flag 强保序 → ℓ/o 更新次序与源一致。

**重设计项 V6: packed NHD 运行期寻址（Q/K/V 装载 + output 写出）**

- **NPU 新算法**：全部 GM↔L1/UB 搬运 = **slice-form `T.copy`**（**禁止 `T.load_nd2nz`**——CG-2026-0004：定头跨步切片被静默平铺误读；packed NHD 定头切片同形态）：
  - Q：`T.copy(q[q_start+bx·bm : q_start+bx·bm+real_m, by, 0:dim], l1_a[0:real_m, 0:dim])`（Q-hoist 每任务一次）；
  - K/V：`T.copy(k[kv_start+n_lo : kv_start+n_lo+tn_real, kv_head, 0:dim], l1_b[0:tn_real, 0:dim])`，`kv_head = by//groups`；
  - output：`T.copy(ub_f16_D[0:rm,0:dim], output[q_start+bx·bm+row0 : …+rm, by, 0:dim])`（行截断）。
- **语义保持论证**：切片在真实行列上与源谓词装载一一对应；`q_start/kv_start` 为运行期标量（P-V1 实测寻址 maxdiff=0.0）；packed 定头切片与 BSHD 定头切片同 CG-2026-0004 约束（slice-form 正确）。

**重设计项 V7: lse `(H,total_q)` transpose-free 布局 + ghost 行截断**

- **NPU 新算法**：lse 声明为 `(H, total_q, 1)` **视图**（同一连续内存）；epilogue `T.copy(ub_lse[0:rm,0:1], lse[by, t0:t0+rm, 0:1])`（`t0 = q_start+bx·bm+row0`，token 轴为内层连续轴 → 自然 2D 区域拷贝，**零 transpose**——VP-D6：活跃源 transpose epilogue 毒化整 kernel 2.6×）；出口闭包 `lse_t.reshape(H, total_q)` 还原契约形状。ghost/尾块以行截断只写真实行。
- **语义保持论证**：写地址 `(by, q_start+bx·bm+i)` 与源 S13 的 `lse[by, q_start+bx·bm+i]` 逐位一致；`[H,total_q]` 与 `[H,total_q,1]` 是同内存视图，reshape 零拷贝。

**重设计项 V8: 接口契约适配（cu_seqlens + total_q/total_k + 闭包 wrapper + lse reshape）**

- 工厂外层签名 `_gqa_sw_fwd_varlen_kernel(batch, heads, heads_kv, total_q, total_k, dim, is_causal, wl, wr, dtype='float16')`（**沿用源签名 + `lru_cache`**）；
- 内层 callable 签名 `(block_m, block_n, num_stages)`（**`threads` 移除**，K9——wrapper 已无 threads，§0.3 亲自核对）；
- 返回值 = **Python 闭包**：按 ws shape 分配、调用编译 kernel、出口 `lse_t.reshape(heads, total_q)`；`(block_m,block_n,num_stages)(q,k,v,cu_q,cu_k,max_seqlen_q) → (output, lse)` 调用形态逐字保持（wrapper custom_op `npub::gqa_sw_fwd_varlen_wrapped_kernel` 与 Kernel class 零改动）；
- prim_func 参数 `(q, k, v, cu_q, cu_k, ws_s, ws_p, ws_o, zbuf, output, lse)`，`@tilelang.jit(out_idx=[-2,-1], target="npuir", pass_configs={…})`（`max_seqlen_q` 不需进 prim_func 签名——它在 host 侧已折入 grid 尺寸 `ceildiv(max_seqlen_q,bm)`，作为编译期常量成为 trace 的一部分）。

### 0.7 标杆实现

- **源算子路径（golden 语义来源）**：`/mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_varlen_fwd.py`（`_gqa_sw_fwd_varlen_kernel` L105–241）+ `tileops/kernels/online_softmax.py`（`make_log2e_scale` / `make_online_softmax_with_mask_guard` / `make_rescale`）；本仓提取件 `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window_varlen/_grouped_query_attention_sliding_window_varlen_fwd_kernels.py`。
- **源仓测试基准（golden 直接来源）**：`examples/TileOPs/tests/ops/test_grouped_query_attention_sliding_window_varlen.py` 的 `GroupedQueryAttentionSlidingWindowVarlenFwdTest.ref_program`（逐样本切片 → expand KV heads → 带 offset 四条件掩码 → softmax → @v → 拼回 packed；`atol=1e-2, rtol=1e-2`，不检查 lse）。
- **NPU 侧同类标杆**：
  - `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/`（**首选范式 / 直接母本**：Expert 两相位 + Q-hoist + scales 延迟回放 + transpose-free lse + fp16/bf16 双 trace + sync shim；22/22 + 集成一次通过）；
  - `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（更早同型母本：GQA prefill Expert 两相位）；
  - `examples/flash_attention/flash_attn_npuir.py`（两相位 S/P 物化结构原型）。
- **pattern-library 实测证据**：`cases.md` **`CASE-gqa-sliding-window-migration`**（定长版姊妹迁移完整档案）；`attention.md` PL-1.7/1.8/1.9/1.9-hardlimits/1.9-twophase/1.11/1.12/1.13/1.16；`traps-compiler.md` VP-D6 / TRAP-bishengir-sync-instr-mode-attr / TRAP-npuir-negative-floordiv-divsi / TRAP-expert-v-operands；`traps-runtime.md` TRAP-npuir-stale-ub-tail-columns-nan；`constants.md`（容量/带宽/发射率常数）。

---

## 1. 概述

### 1.1 算子名称

`_gqa_sw_fwd_varlen_kernel`（GQA 变长滑动窗口前向，`GroupedQueryAttentionSlidingWindowVarlenFwdOp` 的 TileLang kernel；NPU 侧 Expert 模式两相位重实现，目标域 = manifest 5 workload〔smoke + Llama-3.1-8B/70B 短/长序列 × wl∈{256,1024}〕+ 契约/测试域〔KV-cache offset>0、ragged、混合长度、对称带、wl=0〕）。

### 1.2 功能描述

packed NHD 布局 q/k/v 上的 flash-attention 风格 online-softmax 前向，带 **per-sample 运行期长度**（`cu_seqlens_q/k` int32）与 **offset 右下对齐的滑动窗口带状掩码**：输出 `o = softmax(scale·q@k^T + M) @ v` 与每行 log2 域 log-sum-exp `lse`；支持 GQA 分组（`heads % heads_kv == 0`）、`offset = kv_len − q_len`（KV-cache）、四态掩码（causal / 左窗 / 右窗 / 无）、`wl/wr = −1`（无限制）、fp16/bf16。**Op wrapper 只返回 output（丢弃 lse）**，kernel 仍产出 lse 以保持契约。

### 1.3 数学公式

同 §0.1（valid 四条件含 offset / softmax / scale）。滑动窗口有效区间：`k_pos ∈ [q_pos+offset−wl, q_pos+offset]`（causal+左窗）；对称带 `[q_pos+offset−wl, q_pos+offset+wr]`（非 causal）。

### 1.4 算法描述（迁移决策后的 NPU 侧 Expert 算法）

**选定算法族**（§1.6.0 调研结论）：**两相位 S/P 物化 + per-带内块 online softmax + per-sample 运行期带状裁剪**（定长版结构族沿用；源算法的单遍 per-block 链形态在定长版迁移中被实测淘汰，但 online-softmax 数学递推完整保留）。**NPU Expert 形态**（§0.6 V1–V8 落地）：

1. **任务划分**：一维 persistent **20 核**；逻辑任务 = (样本内 q-block × head × batch 样本)；核内 `T.serial` 串行（V1）。
2. **运行期 per-sample 上下文**：逐任务从 GM 读 `cu_seqlens` 得 `q_start/kv_start/q_len/kv_len/offset`（V1/V6）。
3. **带状 KV 范围**：任务内 KV 循环仅覆盖 `[k_start, k_end)`（源 S4 公式逐字移植，运行期 `kv_len/offset`），带宽 `loop_count = k_end − k_start`（V3）。
4. **双引擎两相位分工**：Cube 段承担全部 GEMM——pass-1 串行算完带内全部 S 块（QK^T，Q-hoist，无等待直跑），pass-2 串行算完带内全部 PV 部分和；Vector 段 pass-1 逐块消费 S（vcast → **offset 平移的双侧算术惩罚掩码 + per-row 列 OOB** → prescale → online max/α/ℓ 递推 → P 量化）+ 因子存 UB `scales[]`，pass-2 延迟 rescale 累加 acc_o，epilogue 归一化 + transpose-free lse（V5，双 AIV 自适应行半分）。
5. **跨引擎数据环**：`l0c_s →(FIX)→ ws_s →(MTE2)→ UB →(v 链)→ ws_p →(MTE2)→ l1_p →(gemm2)→ l0c_o →(FIX)→ ws_o →(MTE2)→ UB 累加`；每核单槽 ws 三族 + per-带内块下标 flag（三次握手复用）+ FLAG_TASKDONE（V5）。
6. **掩码**：Vector pass-1 双侧算术惩罚式（`clamp(diff−K,0,1)` + `vmax` + `S += pen·(−1e38)`，**无 vcmp**）+ **offset 平移** + **per-row 列 OOB 惩罚**（V2）；两段式带状分界（无除法直接整数比较）使全有效中段块零掩码开销（VM-bnd）。
7. **尾块/尾带**：size-form gemm + 尾钳位切片 copy + 分形下限钳位（M 16 / K 32）+ **常开**尾带（N 载体零初始化 + 全宽 per-row 列 OOB + `l1_b` zbuf 零初始化）+ 行截断写出 + **ghost 行守卫**（V4/V7）。
8. **dtype 路径**：fp16/bf16 **同构直连**——q/k/v slice-form copy 进 L1，`T.gemm` fp16×fp16 / bf16×bf16 → fp32；P 量化 vcast 到输入 dtype；ws 承载 fp16: f16/f16/f16、bf16: f32/bf16/f32。

**与源算法的结构差异**（来源标注）：① 承载结构从单遍 per-block 链换为两相位独立大循环（§1.6.0 + V5）；② acc_o 从 fragment 驻留改为 UB 驻留 + PV 部分和 GM 物化 + scales 延迟回放（V5）；③ 掩码从 Cube gemm 预填移至 Vector 双侧算术惩罚（VM-mask，含 offset 平移 + per-row 列 OOB）；④ exp2 域改 e 域（VM-e）；⑤ 三维 grid 改一维 persistent（V1）；⑥ 运行期 per-sample 长度由 GPU 标量 GM 读承载为 NPU 核内标量读（V1）；⑦ packed NHD 运行期寻址（V6）；⑧ lse `(H,total_q)` 视图（V7）。算法层（online softmax 递推、**带状裁剪**、**运行期长度/offset**、P 量化点、lse 域、NaN guard）**逐项保留**。

### 1.5 数据流图

```
GM[cu_q,cu_k (int32)] ──核内标量读──> q_start/kv_start/q_len/kv_len/offset（运行期）
GM[q] ──slice T.copy(q_start+bx·bm+..)──> L1[l1_a [bm,block_share]]（Q-hoist）────────────┐
GM[k] ──slice T.copy(kv_start+n_lo+..)──> L1[l1_b]（pass-1 K / pass-2 V 生命周期复用）     │
   L1[l1_a]×L1[l1_b] ─T.gemm(b_transpose,initC=True,size=[tmc,dim,tnc])─> L0C[l0_c]
                                          │ ─T.copy(FIX, f32→f16)─> GM[ws_s] ─set(l)
GM[ws_s] ─MTE2─> UB[ub_f16_N] ─vcast─> UB[ub_f32_N] ─[offset 平移双侧惩罚 + per-row 列 OOB]─[prescale]
   ─reduce(max)─vmax×2─α存档 scales[]─vsub/vexp(P)─reduce(sum)─vcast─> UB[ub_f16_N]
UB[ub_f16_N] ─MTE3─> GM[ws_p] ─set(l) ─> L1[l1_a] ─┐
GM[v] ─slice T.copy(kv_start+n_lo+..)─> L1[l1_b] ──┘
   L1[l1_a]×L1[l1_b] ─T.gemm(initC=True,size=[tmc,tnc,dim])─> L0C[l0_c] ─FIX(f32→f16)→ GM[ws_o] ─set(l)
GM[ws_o] ─MTE2─> UB[ub_f16_D] ─vcast─> UB ─vmul(α 回放)+vadd─> UB[acc_o]
UB[acc_o]/ub_ell/ub_m ─vdiv/vlog2/vcast─> UB ─行截断 slice─> GM[output[...,by,:]] / GM[lse[by,..,0:1] 视图]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（§1.6.0，调研对象为同一数学语义的算法族，源算法只是基线候选之一）；再数学等价地优化公式（§1.6.1）；再把全部计算点向量化（§1.6.2）；最后决定向量化轴与核内布局（§1.6.3）。分析对象 = §1.6.0 选定、经 §0.6 V1–V8 重设计落地的 NPU Expert 两相位算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：`o = softmax(scale·q@k^T + M)@v`（+lse）的 **per-sample 变长 sliding-window GQA 前向**算法族，**目标域画像** = manifest 5 workload（packed NHD，`total_q ∈ {256,2048,8192}`，batch 2/4，H×Hkv ∈ {8×2, 32×8, 64×8}，D ∈ {64,128}，wl ∈ {−1,256,1024}，max_seqlen_q ∈ {128,512,2048}）+ 契约/测试域（offset>0、ragged、混合长度）。**调研深度**：融合类（GEMM + per-sample 动态掩码 + GEMM + 运行期带状裁剪）→ 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。**信息源**：① skill 参考表 `algorithm-candidates.md` 命中行（`ALG-attention` / `ALG-softmax`）；② pattern-library 全主题（`attention.md` PL-1.7/1.8/1.9/1.9-hardlimits/1.9-twophase/1.11/1.12/1.13/1.16；`traps-*.md`；`constants.md`；`cases.md` **`CASE-gqa-sliding-window-migration`**）；③ 本仓 `examples/` attention 族检索（`flash_attention/`、`grouped_query_attention_sliding_window/`（**定长版姊妹，直接母本**）、`TileOPs/` 侧 MHA 两相位归档）；④ 源算法本身（§0.3/§0.4）；⑤ 外部已知算法（PyTorch/flash-attention 生态模型知识——仅取算法思路；FlashInfer/FA3 varlen 布局与 bottom-right 对齐约定）。**网络检索**：未执行——定长版姊妹已完整覆盖本算子族的本地先例与实测档案，varlen 增量（运行期长度/掩码/寻址）与算法族选择正交；按 algorithm-research.md §6 第 6 层纪律如实记录。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源 flash online softmax + slot-window 裁剪（运行期 per-sample）**（单遍 per-block，exp2·LOG2E 域） | 分块 running max/sum + rescale，P 量化后进 GEMM2，fragment 驻留；`[k_start,k_end)` 运行期裁剪 | — | — | ✅（基线） |
| 1 | **单遍 per-block 跨引擎链**（基线的 NPU 承载） | 每 KV 块 C1→FLAG_S→V1→FLAG_P→C2→FLAG_O→V2 握手链 | 数学等价（同族算法不同承载） | —（定长版 MHA 实测淘汰：fa 域两相位 98.05µs vs 单遍调优终值 226.72µs，**2.31×**） | ✅（证据对照） |
| 2 | **两相位 S/P 物化**（定长版结构族，本设计选定） | Cube pass-1 全 S 物化 → Vector pass-1 softmax + scales[] 延迟回放 → Cube pass-2 全 PV 物化 → Vector pass-2 累加；per-块 flag | 数学等价（online 递推保留，α 回放序列与源逐块更新恒等） | Cube pass-1 零等待直跑；α/PV 解耦；实测 fa 域 2.31× | ✅ **（选定）** |
| 3 | e 域 vexp 换底 | `e^{x−m}` 直接计算 | 数学恒等 | 少 2 次 [half,bn] 乘法舍入 | ✅（§1.6.1 M1' 采纳，并入 #2） |
| 4 | **带状 KV 范围裁剪**（运行期 kv_len/offset 版） | `[k_start, k_end)` 块级裁剪，带外块不计算 | **恒等精确**（全掩码块 = online 递推恒等元：α=1、Δℓ=0、Δo=0） | 执行块数 8b-long 15360 vs 全 causal 18432（**−16.7%**）；**FLOPs 103.2 vs 137.5 GFLOP（−24.9%）** | ✅（§1.6.1 M4' 采纳，并入 #2；源已用） |
| 5 | **offset 平移双侧掩码算术惩罚式**（varlen 版） | `pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1))`，`K_*` 含 `+offset`；`S += pen·(−1e38)` | **恒等精确**（pen∈{0,1}；掩蔽位精确 `−1e38` → P=+0.0） | 消 vcmp/vselect 标量化（PL-1.11；P-V2 零 warning） | ✅（§1.6.1 M5' 采纳；P-V2 7/7 bit-exact） |
| 6 | **per-row 列 OOB 惩罚**（varlen 新增，常开尾带） | `poob = clamp(diff − ((tn_real−1)−i), 0, 1)`（**含行号**，绝对列条件） | **恒等精确**（`j ≥ tn_real` ⟺ `diff ≥ tn_real−i`） | 免尾带 NaN 污染（配合 N 载体零初始化） | ✅（§1.6.1 M5' 采纳；P-V2 7/7 bit-exact + NaN-free） |
| 7 | **KV-split / flash-decoding**（KV 维切分多核并行 + 跨核合并） | 每 split 独立 online 状态 + log-sum-exp 合并 | 数学等价 | 小 Sq 欠载时并行度↑ | ✅（R3/R4 评估后 defer） |
| 8 | S/P/O 跨引擎 f16 传输 | S、P、O_partial 以 f16 物化 | **容差内等价**（S 域 2^-11 → ΔP/P ~0.26%；O 部分和 f16 实测 tier-1 0 flips） | 三族 ws 字节减半 | ✅（§1.6.1 M8'/M9' 采纳） |
| 9 | scale 折入 q（任务头一次 [bm,dim] 缩放替代每块 prescale） | `(s·q)·k ≡ s·(q·k)` | **容差内等价但精度降级**（源在 fp32 域乘 scale；f16 q 折入引入 2^-11 级 S 误差） | 每 task 省 NK−1 次 vmul | ✅（精度论证后否决，§1.6.1 M11'） |
| 10 | lse 域公式 `log2(ℓ) + m·LOG2E` | `log2(ℓ·e^m)` 恒等 | 数学恒等 | 免大数指数化 | ✅（§1.6.1 M2' 采纳） |
| 11 | f16 softmax 链（[half,bn] 大 pass 全 f16 算术） | vexp/vsub/vmul/reduce f16 | 容差内等价（定长版 v9 实测 −3.5%~−14.1%） | 大 pass 字节减半 | ✅（**defer Stage 4**：f32 链 + f16 传输已达标） |
| 12 | 除法转乘倒数（o/ℓ → o·(1/ℓ)） | 行广播除一次 vs 倒数+乘 | 数学恒等 | 无收益（每任务 1 次 vdiv） | ❌（op 数相同） |
| 13 | **band 内逐列裁剪**（boundary 块内全掩蔽列跳过） | 逐列细粒度跳过 | 恒等精确 | 边界块内 ±bn 列浪费 | ❌（列粒度 < 向量宽度，无对应 API；块粒度裁剪已由 #4 承担） |
| 14 | **host 预计算块表**（每样本有效块前缀和 → host 传入扁平任务表） | 取代核内 cu_seqlens 标量读 | 数学等价 | 免核内标量读 | ❌（**契约不允许**：Op wrapper 传的是 device cu_seqlens，host 侧读取需 D2H 同步，破坏 custom_op 契约且引入同步开销；源亦用核内读） |
| 15 | **packed↔padded 转换**（host 侧 padding 成定长再走定长版） | 变长 → 定长 padding + mask | 数学等价（pad 行掩蔽） | 复用定长版 | ❌（**改契约**：输出须 packed；且 padding 浪费算力/访存，违背 varlen 初衷） |

**R2 在线算法**：**有**——直接证据：源算法本身即 online softmax（§0.4 OP1：分块增量 max/sum/logsum + rescale，单遍样本内 KV）；**选定结构 #2 同样保留 online 递推**（Vector pass-1 逐块 running max/ℓ，per-block α 因子存档回放——「在线」性质在 softmax 层完整保留；物化的是跨引擎传输的 S/P/O tile，非全量 score 矩形）。收益口径：KV 扫描 1 遍（vs 两遍 materialize-softmax 的 2 遍）；中间缓冲 O(bm × band)（per-core 单槽 ws）vs O(sq × sk)。**varlen 特化**：per-sample 带可能短到单块（`loop_count=1`，如 smoke 或 KV-cache 短 q）——此时 online 递推退化为直接 softmax（α_0 值无关），trace 层统一处理（不做退化特化——causal/window 域 loop_count 运行期变化，收益微小，M7'/f13）。结构判据复核：softmax 分母与 PV 分子均可分解为 running 统计量 → 在线变体存在且源已采用 ✓。

**R3 复杂度对比**（四口径；**以 manifest varlen `8b-long`（total_q=8192,batch=4,q_len=kv_len=2048,H=32,Hkv=8,D=128 causal fp16 wl=1024）为基准**，有效 config 统一 bm=64/bn_eff=256；FLOPs 口径 = 乘加计 2 = `4·H·total_attended·D`；访存分列「GM 必流量」与「跨引擎 fabric 流量」；扫描遍数 = q/k/v 各自 GM 读取特征；中间缓冲 = 峰值驻留)。**执行块数 E 口径** = Σ_over (head,样本) loop_count。**逐项复算**（bm=64,bn=256,wl=1024,causal,offset=0）：`k_end(bx)=ceildiv((bx+1)·64,256)=ceildiv((bx+1)/4)`、`k_start(bx)=max(0,bx·64−1024)//256`；逐 bx（0–31）loop_count = **1×4,2×4,3×4,4×4,5×4,5×12** → Σ=4+8+12+16+20+60=**120**/（head,样本）；× H·B = 32·4 = **128** → **E = 15360** 执行块。全 causal（wl=−1）：`k_start≡0`，Σk_end=4·(1+…+8)=**144** → ×128 = **18432** → 裁剪削减 **−16.7%**。**FLOPs 复算**：每样本 Σ_q_pos attended = `Σ_{p<1024}(p+1) + 1024·1025 = 524800+1049600 = 1574400`；total_attended = 4·1574400 = 6297600；FLOPs = 4·32·6297600·128 = **1.0318e11 = 103.2 GFLOP**。全 causal（wl=−1）attended/样本 = Σ_{p=0}^{2047}(p+1) = 2048·2049/2 = 2098176 → total_attended = 4·2098176 = 8392704 → FLOPs = 4·32·8392704·128 = **1.375e11 = 137.5 GFLOP**（比值 103.2/137.5 = 0.750 → **FLOPs 削减 −24.9%**，与定长版同族 −25% 一致；注意此与**块数**削减 −16.7% 是不同口径——块数削减 ≠ FLOPs 削减）。

| 算法候选 | FLOPs | 访存量 | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|--------|---------|-------------|---------------------|
| #0/#1 基线 flash online（单遍 per-block 链 + 运行期带状裁剪） | **103.2 GFLOP** | HBM 必流量 **167.8 MB**（`2·(q_elems+kv_elems)·2B`，`q_elems=8192·32·128=33.55M`、`kv_elems=8192·8·128=8.39M`）；fabric：KV tile 流式重读 = **E** × (K+V tile 各 [256,128]f16 64KB) = 15360 × 128KB = **2.01 GB**；ws 往返（写+读）= E × 160KB = **2.52 GB**；合计 ≈ **4.7 GB** | q 1 遍；k/v 每任务流式 1 遍（H·B 组重读，L2 命中） | per 块 S 32KB + P 32KB + O f32 32KB | 任务级 20 核并行（4096 任务）；跨核零同步；**每块 3 次 flag 握手 × 15360——发射流阻塞实测为结构地板**（定长版 MHA busy aic_scalar 38–47%） |
| #2 两相位 S/P 物化（本设计，含 #3–#6/#8/#10） | 同 103.2 GFLOP（GEMM 工作量与 #1 逐块相同） | HBM 同 167.8 MB；fabric：KV tile 流式重读 2.01 GB（同 #1）+ ws 往返 = E × (S 32KB+P 32KB+O **f16** 16KB)×2 = 15360 × 160KB = **2.52 GB**；合计 ≈ **4.5 GB** | 同 #1 | per-core 单槽 ws = `pad_band·bm·(f16)·2 + dim·L_max·bm·f16 ≈ 1.05MB`（L_max≈6）；总量 20 核 ≈ 21MB（L2 ✓） | 任务级 20 核并行（4096 任务）；跨核零同步；flag 次数同量级 **但 wait 与真依赖对齐**（pass-1 零等待直跑）——定长版同口径实测 2.31× |
| #4 运行期带状裁剪（并入 #2 后量测） | 103.2 GFLOP（= 全 causal 137.5 的 0.750×，**FLOPs −24.9%**） | HBM 同；fabric 随 E 线性降 | 同 | 同 | **E 15360 vs 全 causal 18432（块粒度 −16.7%）**；内段全有效块 Σmax(loop_count−2,0)=60×128=7680、边界掩码块 7680；无需额外结构 |
| #6 KV-split（split=2） | 同 + 合并开销 O(sq·D·split) 可忽略 | HBM 同；fabric + 每 split 状态回传 | 同 | + 每 split 状态 (m,ℓ,o) × 核数 | 任务数 4096 » 20 核，**无欠载场景**；**无核间原语**（T.set_flag.md §1 / T.sync_block_set.md §1 适用范围均为核内）——跨核合并须两阶段 kernel |
| #8 f16 传输（并入 #2 后） | 同 | O/S 列已计入 #2 | 同 | 同 #2 | 同 #2 |
| #14 host 块表 | 同 | 同 + D2H 同步 | 同 | 同 | **契约不允许**（D2H 同步；源亦核内读） |
| #15 packed↔padded | padding 域算力↑（padding 率） | padding 域访存↑ | 同 | 同 | 改契约（输出须 packed）——否决 |

**R3 块数逐项复算明细**（8b-long：q_len=kv_len=2048, bm=64, bn_eff=256, wl=1024, causal, offset=0）：

| bx | 0–3 | 4–7 | 8–11 | 12–15 | 16–19 | 20–31 |
|----|-----|-----|------|-------|-------|-------|
| loop_count | 1 | 2 | 3 | 4 | 5 | 5 |
| 小计（4 或 12 项） | 4 | 8 | 12 | 16 | 20 | 60 |

Σloop_count = 4+8+12+16+20+60 = **120**/（head,样本）；×（H·B = 32·4 = 128）= **15360 执行块**。全 causal（wl=−1，`k_start≡0`）Σ = 144 ×128 = **18432** → **−16.7%**。内段全有效块 = Σmax(loop_count−2,0) = 60 ×128 = **7680**（边界掩码块 7680）。8b-short（q_len=512, wl=256）：Σloop_count=12（bx0–3:1，bx4–7:2）×128 = **1536**，与全 causal 同值（0 减免）。smoke（q_len=128, wl=−1, causal）：Σ=2（bx0,1:1）×（H·B=16）= **32**。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| #2 两相位 + 运行期带状裁剪（本设计） | GEMM→Cube（分形 M≥16/K≥32 由尾钳位、K=dim 64/128 ✓）、softmax/掩码/累加→Vector ✓ | L1 96KB/512KB（l1_a [64,256]+l1_b [256,128] f16）✓；L0C [64,256] f32 = 64KB/128KB ✓；UB ~154KB/192KB per AIV（§4.5）✓ | 尾轴 bn_eff/dim 为 16 倍数（守卫公式）；`[half,1]` 行向量非搬运主路径 | `k_start/k_end/loop_count/L_*_full/tmc/tnc` 为 kernel_id/bx + **运行期 cu_seqlens 读** 派生 PrimExpr（**P-V1 实测可 lower/运行**） | Cube pass-1 零等待；pass-2 P-wait 与真依赖对齐；per-块 flag 严格交替 | 任务独立零跨核同步 | ✅ **选定**（目标域任务数 1024–8192 ≥ 20 核，饱和域；smoke 32 任务轻度欠载，§5.5） |
| #0/#1 单遍 per-block 链 | 同上 | 同上 | ✓ | ✓ | **定长版同族实测否决**：`sync_block_wait` 阻塞整条发射流（PL-1.9-hardlimits：ns=1≈ns=2、wide≈serial、深流水 +12% 三重证据）；fa 域调优终值仍 2.31× 慢 | 同上 | ❌（发射流结构地板，非 tiling 可解） |
| #14 host 块表 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ❌ **契约不允许**（device cu_seqlens → D2H 同步破坏 custom_op；源亦核内读） |
| #15 packed↔padded | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ❌ **改契约 + 浪费算力**（输出须 packed；padding 率随最短样本放大） |
| #6 KV-split | ✓ | ✓ | ✓ | ✓ | 需**跨核**数据交换：T.set_flag.md §1 / T.sync_block_set.md §1 均为核内语义——**无核间原语**（负向依据：两文档 §1 适用范围条款亲自核对） | 任务数 4096 » 20 核，**无欠载场景**——split 增并行度无需求方 | ⏸ **defer**（收益域为 decode/小 batch，非本 workload；列入 §9.2） |
| #5/#6 offset 平移 + per-row 列 OOB | Vector ✓（vsub/vmax/vmin/vmul/vadd 全 ALU） | ✓（复用 f32 N-buffer） | ✓（diff/阈值 f32 精确整数） | K_up/K_lo/tnr 为运行期 PrimExpr ✓（P-V2 实测） | ✓ | ✓ | ✅ **采纳并入 #2**（PL-1.11 + P-V2 7/7 bit-exact） |
| #9 scale 折入 q | ✓ | ✓ | ✓ | ✓ | 引入任务头跨引擎串行依赖 | ✓ | ❌ **否决**（精度语义降级，§1.6.1 M11'） |
| #11 f16 softmax 链 | ✓ | ✓（大 pass 减半） | ✓ | ✓ | ✓ | ✓ | ⏸ **defer Stage 4**（f32 链 + f16 传输已达标） |

**调研结论**：**选定算法族 = 两相位 S/P 物化 + per-带内块 online softmax + per-sample 运行期带状裁剪（候选 #2，含 #3/#4/#5/#6/#8/#10 子项采纳）**。关键依据：① R4 实测证据——定长版同族 fa 域两相位 98.05µs vs 单遍链调优终值 226.72µs（2.31×），单遍链发射流阻塞经 11 轮调优被证实为结构地板；② R3——两相位不增加 GEMM 工作量与 HBM 必流量，fabric 增量略优（O f16 减半）；运行期带状裁剪额外削减执行块数（8b-long 15360 vs 全 causal 18432，−16.7%）；③ R4 全项亲和通过（L1/L0C/UB 预算、运行期 PrimExpr 边界 **P-V1 实测**、per-块 flag、任务独立零跨核）。**与基线（源算法）的结构差异一句话**：online softmax 数学递推、运行期带状裁剪、per-sample 长度逐项保留，承载结构从「单遍 per-block 跨引擎握手链」换为「Cube/Vector 两相位独立大循环 + S/P/O GM 物化 + scales 延迟回放」。**「无更优替代」的调研范围**：skill 参考表 attention/softmax 命中行、pattern-library 全主题（attention/traps/constants/cases 含 `CASE-gqa-sliding-window-migration`）、examples/ attention 族检索（含定长版姊妹全档案）、源码 §0.3/§0.4；候选 #6/#11 以量化理由 defer（非「未考虑」），#1/#9/#13/#14/#15 以实测/精度/契约/结构理由否决。**源算法优化手段的意图承接**（§0.4 → 本设计，逐项）：OP1 online softmax→保留；**OP2 带状裁剪→保留（V3，逐字移植源式，运行期 kv_len/offset）**；**OP2b ghost 空转→保留（V1/V7，real_m≤0 ⇒ loop_count=0 + 行截断）**；OP3 SMEM tiling→L1 生命周期复用 + Q-hoist；OP6 Pipelined→两相位（重叠意图由 pass-1 零等待承接）；OP5 tensor core→Cube；OP11 exp2 折叠→e 域 vexp（M1'）；OP7 fragment 驻留→UB 驻留 + ws 物化（消 GM 往返意图让渡给 L2 命中的 ws 流量，代价计入 R3 且定长版实测净赚 2.31×）；OP8 warp 归约→Vector 行归约；OP12 mask 融合→offset 平移算术惩罚 + per-row 列 OOB（免独立 mask pass 意图由「全有效中段零掩码 + 全向量构造」承接）；OP4 coalesced→slice-form copy；OP15 bf16 直连→Cube bf16；**OP16 运行期 per-sample 长度→核内 GM 标量读（P-V1 实测）**；OP9/OP10/OP13 CUDA 专属→舍弃/重设计；OP14 lru_cache→保留。

**设计期估算（D-2）**：代表 workload = varlen `8b-long`（total_q=8192,batch=4,q_len=kv_len=2048,H=32,Hkv=8,D=128 causal wl=1024 fp16；**执行块数 E=15360**，§1.6.0 R3 逐 bx 复算）。**流量项**：GM 必流量 167.8MB ÷ 有效带宽（`CONST-mte2-degradation` 大流量档 ~18–34 GB/s/core × 20 核 ≈ 0.36–0.68 TB/s；910B2C 常数跨 SoC 未重验，标「未实测假设 + 估算依据」）= ~247–466µs；fabric ws 往返 **2.52GB** ÷ fabric（`CONST-fabric-bw` ≥1.73 TB/s，910B2C 实测，跨 SoC 未重验）= ~1.46ms（可与计算重叠）。**发射项**：Vector op 数 ≈ 15360 块 × ~17 + epilogue ≈ **261k**，per AIV（÷40 AIV）≈ 6528 op × `CONST-vector-launch-overhead` ~0.5µs/op（910B2C 实测，跨 SoC 未重验）= **~3.26ms/AIV**。**容量项**：UB ~154KB/192KB、L0C 64KB/128KB、L1 96KB/512KB，均未触发容量循环 → 0。**朴素发射口径估算下界 ≈ 3.3ms**（发射项主导，流量项并行重叠）；**定长版 MHA 同族实测锚点交叉核对**：定长版同结构 8b-long（H·B=64）实测 1279.54µs（`CASE-gqa-sliding-window-migration`，msprof）——本算子 H·B=128（2× 任务）→ 1279.54 × (128/64) × (15360/15360) ≈ **2.56ms**；乐观侧按发射/流量双口径 ≈ 2.0–3.3ms。**Stage 4 首采校准**：fabric 有效常数、910B3 上的向量发射率（本设计期未实测）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的两相位 + 运行期带状裁剪算法。R1 中初判采纳的候选在此完成四要素论证（原式 → 优化后公式 → 等价性论证 → 收益量化）。M1'–M13' 与定长版 M1–M13 同族，**varlen 增量项标 `'` 并注明 offset/运行期差异**。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| M1' | e 域换底 | `exp2(x·scale − m·scale)`，scale = s·LOG2E（源截断 1.44269504） | `vexp(x' − m')`，x' = prescale·x（fp32 域 vmul） | 数学恒等：`e^u = 2^{u·log2e}`；少 2 次 [half,bn] 乘法舍入 | 每 [rm,bn] 块省 2 个 v-op 路径长度（`verify_equiv.py` M1': 6 ulp, 违反率 0） |
| M2' | lse 尾式 | `log2(Σ e^{x−m}·e^m)`（需恢复大数指数） | `lse = vlog2(ℓ) + m·LOG2E`（全精度 1.4426950408889634） | `log2(ℓ·e^m) = log2(ℓ) + m·log2(e)` 恒等 | 免 [rm,1] 大数乘法与潜在溢出（M2': 1 ulp, 0） |
| M3' | softmax 稳定化（减 max） | `e^{x_i}/Σe^{x_j}` | `e^{x_i−m}/Σe^{x_j−m}`，m = 行 max | 指数平移恒等（分子分母同乘 e^{−m}） | 消溢出（必选，源同构；M3' 实证 f64 无 inf） |
| **M4'** | **运行期带状 KV 裁剪（滑动窗口核心）** | 全 KV 循环 `[0, ceil(kv_len/bn))` | `[k_start, k_end)`（源 S4 公式，**运行期 kv_len/offset**），带外块不计算 | **恒等精确**：全掩码块贡献 = online 递推恒等元——`m_cur=−1e38`（clamp 有限）、`α=1`、`P=+0.0`、`Δℓ=0`、`Δo=0`；逐块累加序为保序子序列 | 块数 **−16.7%**（8b-long E 15360 vs 全 causal 18432）；M4' 实证 `alpha(all −inf block)==1` |
| **M5'** | **offset 平移双侧算术惩罚掩码 + per-row 列 OOB（运行期）** | 源 `if_then_else(掩码, −inf, 0)`（Cube 预填）；定长迁移用 `vcmp`+`vselect` | `pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1), clamp(diff−((tn_real−1)−i),0,1))`，`K_up=(bx·bm+row0+offset)−n_lo`、`K_lo=(bx·bm+row0+offset−wl)−n_lo`；`S'=S+pen·(−1e38)` | **恒等精确**：diff 为整数 → pen∈{0,1}；掩蔽位 `S'=S−1e38`（f32 大数吸收）→ `P=e^{−1e38−m}`（m 经 M6' clamp 有限）= **逐位 +0.0** ≡ 源 `e^{−inf}=0`；**offset 平移**：`k≤q+offset ⟺ diff≤(bx·bm+row0+offset)−n_lo`（端点代入）；**列 OOB**：`j≥tn_real ⟺ diff≥tn_real−i`（**必含行号**——绝对列条件，P-V2 首版暴露） | 消 vcmp/vselect 标量化（P-V2 零 warning）；掩码链 ~8 个全速 ALU op；全有效中段块经 VM-bnd 零掩码（M5' 实证「masked bit-exact +0.0=True」） |
| M6' | 单调 max + clamp | `max(max_prev, block_max)` + clamp ≥ −1e38 | 同式（vmax ×2，[rm,1] 行向量） | 源 NaN guard 逐字保留：`m ≥ −1e38` 恒成立 → masked 位 `e^{−1e38−m}` 精确 0、无 `−inf−(−inf)` NaN 路径 | 防御性（M6' 实证） |
| M7' | prescale=1.0 跳过 | `vmul(S, 1.0)` | trace-time `if prescale != 1.0` 跳过 | 乘 1 恒等 | 卫生项（本算子 prescale ≠ 1 恒成立，无跳过域） |
| M8' | S f16 物化（fp16 路径） | S f32 跨引擎 | `T.copy(l0_c f32 → ws_s f16)`（rint 隐式量化） | **容差内等价**：S 值域 \|S\| ≲ 60·dim^{−1/2}，f16 相对误差 2^-11（PL-1.9 probe max_rel 4.7e-4） | S 跨引擎字节减半（M8'/M9' 违反率 0） |
| M9' | O 部分和 f16 物化（fp16 路径） | O_partial f32 跨引擎 | `T.copy(l0_c f32 → ws_o f16)` | **容差内等价**（PL-1.9；v11nt O-f16 tier-1 0 flips） | O 跨引擎字节减半 |
| M10' | P 量化点保持 | 源 `acc_s(fp32) → cast 输入 dtype → GEMM2` | `vcast(P_f32 → 输入 dtype, rint)` 后经 ws_p 进 Cube | 量化点与源一致（softmax 后、GEMM2 前、最近舍入）；与源逐位同 dtype | 语义对齐（防漂移项；M10' 实证） |
| M11' | **否决**：scale 折入 q | 每块 `vmul(S_f32, s)`（fp32 域） | 任务头 `vmul(q_f16, s)` 后进 GEMM | **容差内等价但精度降级，不采纳**：源乘 scale 在 fp32 域（相对误差 2^-24）；f16 q 折入使每 q 元素先舍入（2^-11）再进 L0C fp32 累加——S 相对误差升至 ~2^-11，威胁 §8.2 冻结容差 | （否决记录）理论省 NK−1 次 vmul/任务，精度代价不成比例 |
| M12' | Q-hoist | 每块重读 Q | Q 在任务头一次装载 l1_a，pass-1 全程复用 | 数据不变（同一 Q 块），装载位置前移 | 免 `loop_count−1` 次 [tmc,dim] GM→L1 重读 |
| M13' | ℓ 递推 clear=True | `T.reduce(...,"sum")` 无 clear | `T.reduce(...,"sum", clear=True)` | 无 clear 形态为**文档化静默错误形态**（`docs/Tilelang.language/规约操作/T.reduce.md §2.3`）；clear=True 后 `ℓ ← ℓ·α + ΣP` 与源 S10 恒等 | 正确性修复项 |
| M14' | **否决**：boundary 块内逐列裁剪 | 边界块全宽处理 | 逐列跳过 | 恒等精确但**列粒度 < 向量宽度**，无对应 API | （否决记录）块粒度裁剪已由 M4' 承担 |

**优化结论**：采纳 M1'/M2'/M3'/M4'/M5'/M6'/M7'/M8'/M9'/M10'/M12'/M13' 共 12 项；否决 M11'/M14'（精度 / 无 API）；M7' 在本算子 prescale≠1 域无触发。**验证覆盖口径**：**9 个数值型采纳项（M1'/M2'/M3'/M4'/M5'/M6'/M8'/M9'/M10'）+ 4 个 E2E 域经 `verify_equiv.py` 机器验证 11/11 EQUIV_PASS**（结果表见下）；**M7'/M12'/M13' 未入机器表，按构造论证**（M7' 本域无触发；M12' 结构性数据复用、无数值变更；M13' `clear=True` 为 T.reduce.md §2.3 文档化静默错误形态的正确性修复）。

**优化后公式**（供 §3.1 使用）：

$$
\text{prescale} = \text{dim}^{-1/2},\quad
\text{pen}_{i,j} = \max\Big(\text{clamp}(\text{diff}_{i,j}-K_{up},0,1),\ \text{clamp}(K_{lo}-\text{diff}_{i,j},0,1),\ \text{clamp}(\text{diff}_{i,j}-((tn_{real}-1)-i),0,1)\Big)
$$
$$
S''_{i,j} = \text{prescale}\cdot S_{i,j} + \text{pen}_{i,j}\cdot(-10^{38}),\qquad
\text{diff}_{i,j} = j-i,\quad K_{up} = bx\cdot bm + row_0 + \text{offset} - n_{lo},\quad K_{lo} = K_{up} - wl
$$
$$
m_{\text{cur}} = \max_j S''_{i,j},\quad m = \max(m_{\text{prev}}, m_{\text{cur}}, -10^{38}),\quad
\alpha_l = e^{m_{\text{prev}} - m},\quad P = e^{S'' - m}
$$
$$
\ell \leftarrow \ell\cdot\alpha_l + \textstyle\sum_j P_{i,j},\qquad
o \leftarrow o\cdot\alpha_l + \big(\textstyle\sum_j P_{i,j}\, v_{j,:}\big),\qquad
\text{out} = o/\ell,\quad \text{lse} = \log_2 \ell + m\cdot\text{LOG2E}
$$

（`K_up`/`K_lo`/`tn_real` 为 per-AIV、per-带内块的**运行期** PrimExpr，§0.6 V2/V3；α_l 序列在 pass-1 计算并作用于 ℓ，同时存档 `scales[]`，pass-2 以同一序列作用于 o。PV 部分和由 Cube pass-2 计算、经 ws_o 以 f16 物化回传（M9'；bf16 路径 f32）。）

**等价性机器验证（S1-EQUIV-EXEC 门禁项；`verify_equiv.py`，torch CPU，2026-10-16，退出码 0，11/11 EQUIV_PASS）**

对照口径：**基线式** = 上表「原式」列（源 `_gqa_sw_fwd_varlen_kernel` 公式忠实模拟：per-sample packed 切片 + 运行期 offset + exp2 域 + 源截断 LOG2E=1.44269504 + 源运算序 + 全 KV 循环 + 源逐元素掩码〔q_oob/k_oob〕 + P 量化到输入 dtype + fp32 累加）；**优化式** = 上表「优化后公式」列（e 域 + 全精度常数 + 运行期带状裁剪 + offset 平移双侧算术惩罚 + per-row 列 OOB + f16 S/O 物化）；**参照** = fp64 真值 + `golden_varlen`（源语义 per-sample 物化 softmax）。工作负载 = `cu_q=[0,64,160,288]`、`cu_k=[0,96,224,384]`（**offset=32/64/96 KV-cache 域**），H=4/Hkv=2/D=64，wl=16 causal，fp16 + bf16 镜像。判定阈值：恒等链类 ≤128 ulp；容差类 tier-1 违反率 =0（bf16 另测 tier-2 2-ulp 回退）。

| 项 | 比较域 (dtype) | max ULP (opt vs base) | 违反率 (opt vs 参照) | 违反率 (opt vs base) | 结论 |
|----|---------------|----------------------|--------------------|--------------------|------|
| M1' | P (f32) | 6 | 0 | 0 | EQUIV_PASS |
| M2' | lse (f32) | 1 | 0 | 0 | EQUIV_PASS |
| M3' | p (f64) | 0 | 0 | 0 | EQUIV_PASS |
| M5'/M6' | P masked/valid (f32) | 0 | 0 | 0 | EQUIV_PASS |
| M4' | α (f32) | 0 | 0 | 0 | EQUIV_PASS |
| M8'/M9' | out (f16) | 2147798666※ | 0 | 0 | EQUIV_PASS |
| M10' | P (f16) | 0 | 0 | 0 | EQUIV_PASS |
| E2E-fp16 | out (f16) | 2147798666※ | 0 | 0（vs 源基线） | EQUIV_PASS |
| E2E-fp16-golden | out (f16) | 2178593504※ | 0（tier-1） | — | EQUIV_PASS |
| E2E-bf16 | out (bf16) | 734 | 0 | 0（vs 源基线） | EQUIV_PASS |
| E2E-bf16-golden | out (bf16) | 2192168490※ | 0（tier-1 率 5.8e-4 → **tier-2 率 0**） | — | EQUIV_PASS |

※ 容差类行的 ULP 为**诊断值**：小幅度 out 元素在 f16/bf16 网格上 ulp 极小，atol=5e-3 主导——判定以违反率为准。**关键断言明细**：M5'/M6' masked 位 P 逐位 +0.0（bit-exact）、无 NaN；M4' 全 −inf 块 α=1 精确；M10' rint 量化误差 ≤0.5 ulp；E2E-bf16 的 tier-1 5.8e-4 率经 2-ulp 回退门归零——属源 `acc_s_cast` P 量化噪声在早 causal 行的放大（定长版同款 tier-3 分类口径；fp16 无此现象，tier-1 即 0）。**验证方式标注**：上表 11 行覆盖 9 个数值型采纳项 + 4 个 E2E；**M7'/M12'/M13' 不在表内**，按构造论证。M11'/M14'（已否决）不在验证范围。**设备侧正交证据**：`history_version/design_probe_varlen_mask.py`（P-V2）对 offset 平移 + per-row 列 OOB + 常开尾带掩码链 **7/7 bit-exact**（运行期 offset + 运行期 tn_real 进阈值）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 分析对象 = 两相位 + 运行期带状裁剪实现方案的全部循环与标量计算点。替代方案所用 API 均有 `docs/Tilelang.language/` 或 `examples/` 佐证（目录存在性已逐一核验：数学操作/、规约操作/、创建操作/、比较操作/、条件操作/、逻辑操作/、shape操作/、数据类型转换操作/）。

| # | 计算点 | 原实现形态（源 / 定长版） | 向量替代方案（本设计） | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------------------|------------------------|---------|------------------------------|
| 1 | **offset 平移双侧掩码 + per-row 列 OOB 构造** | 源 `T.Parallel(bm,bn)` 嵌套 if/else 谓词写 | `T.arange`(f32 colmat/rowmat/rowi，每核一次) + `T.vsub`(diff) + `clamp(diff−K)`/`clamp(K_lo−diff)`/`clamp(diff−thr)`（vsub+vmax+vmin）+ `T.vmax`(合成) + `T.vmul`(·−1e38) + `T.vadd`（V2；API：创建操作/T.arange.md、数学操作/T.vsub.md、比较操作/T.vmax.md、比较操作/T.vmin.md、数学操作/T.vmul.md、数学操作/T.vadd.md）**P-V2 设备 7/7 bit-exact** | ✅ | — |
| 2 | 全有效块分类（两段式分界，无除法） | 源无条件全块掩码 | `if n_lo > bx·bm+row0+offset−(bn−1):` / `if n_lo < bx·bm+row0+rm−1+offset−wl:` 两段 per-AIV 运行时（**直接整数比较**，避 CG-2026-0024） | ✅（控制流经 VM-bnd 消除掩码 op） | — |
| 3 | prescale 乘 | `T.Parallel` / 每元素 | `T.vmul(ub_f32_N, prescale, ub_f32_N)`（标量广播） | ✅ | — |
| 4 | 行 max | warp 归约树 / reduce_max | `T.reduce(ub_f32_N, ub_mcur, dims=[1], reduce_mode="max", clear=True)` | ✅ | — |
| 5 | 单调 max / clamp | `T.Parallel(bm)` 标量 max | `T.vmax` ×2（行向量 [rm,1]） | ✅ | — |
| 6 | α = e^{m_prev−m} | `T.Parallel(bm)` exp2 | `T.vsub` + `T.vexp`（行向量） | ✅ | — |
| 7 | P = e^{S−m} | `T.Parallel(bm,bn)` exp2 | `T.vsub`（[M,N]−[M,1] 行广播）+ `T.vexp` | ✅ | — |
| 8 | 行 sum | warp 归约 | `T.reduce(..., "sum", clear=True)`（**clear 显式 True**） | ✅ | — |
| 9 | ℓ 更新 / o rescale 累加 | `T.Parallel` 标量 | `T.vmul`/`T.vadd`（行广播 / [rm,dim]） | ✅ | — |
| 10 | P 量化 cast | `T.copy(acc_s, acc_s_cast)` | `T.vcast(ub_f32_N, ub_f16_N, round_mode="rint")` | ✅ | — |
| 11 | epilogue 归一化 | `T.Parallel` 除法 | `T.vdiv(acc_o, ub_ell, acc_o)`（行广播） | ✅ | — |
| 12 | lse 尾式 | `T.Parallel(bm)` log2 + 乘加 | `T.vlog2(ub_ell, ub_lse, ub_lse_tmp)` + `T.vmul` + `T.vadd` | ✅ | — |
| 13 | S/O 到达后的 vcast（f16→f32） | —（两相位新增搬运点） | `T.vcast(ub_f16_N, ub_f32_N, "rint")` | ✅ | — |
| 14 | 输出 / lse 写出 | 谓词标量 GM 写（源 S13，含 ghost 守卫） | 行截断切片 `T.copy`（src `[0:rm,…]` 与 dst 同形；ghost 以 `rm>0` 守卫） | ✅ | — |
| 15 | 状态初始化 + 尾带载体零初始化 | `T.clear`/`T.fill`（源 S2） | `T.vbrc(0/−inf, ub)`（shape操作/T.vbrc.md；标量 let-bound） | ✅ | — |
| 16 | **cu_seqlens 运行期标量读**（q_start/kv_start/q_len/kv_len/offset） | 源 GPU 逐线程标量 GM 读 | 无向量等价（每任务 O(1) 标量 GM 读） | ❌ | **block 级索引 / 任务映射计算**：每任务 ~5 次 GM 标量读（int32，P-V1 实测可 lower/运行），非逐元素热点（对比每任务数万向量元素）；源同形态 |
| 17 | 任务解码 / `k_start`/`k_end`/`loop_count` / 两段式分界 / `tmc`/`tnc`/`real_m`/`tn_real` 标量 | — | 无向量等价（每任务 O(1) 索引算术） | ❌ | **block 级索引 / 任务映射计算**：每任务 ~20 次整数标量运算，非逐元素热点 |
| 18 | `if n_lo > …` / `if task_id > 0` / `if l != 0` / `if rm > 0` / `if real_m > 0` 控制分支 | — | 无向量等价 | ❌ | **tile 级顺序依赖 / 块分类**：两段式分界、任务边界握手、首块 α 语义、ghost 空转——均为块序控制流非数据并行 |
| 19 | 工厂层 config 守卫 / 校验 / trace 分派 | — | 无向量等价 | ❌ | **host 侧元数据计算**（不在 kernel 内）：trace-time Python（`L_max_bound` 守卫 + trace 家族分派） |

**向量化结论**：逐元素计算已全部向量化（15 项 ✅，替代方案 API 均有 docs 佐证；掩码 1 项由源 `T.Parallel` 谓词改为**全 ALU 算术惩罚链 + per-row 列 OOB**，消除 vcmp 标量化，P-V2 设备 7/7 bit-exact）；保留 4 类标量/控制流——运行期 cu_seqlens 标量读（#16）、block 索引与运行期范围（#17）、tile 级顺序依赖/块分类控制流（#18）、host 元数据（#19），逐项理由见表。**与 §6 一致性**：§6 循环结构中不出现任何逐元素标量循环。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout（packed NHD 3D 进 / packed NHD 3D out + `(H,total_q)` lse）是契约；核内布局与 lane 映射是设计变量。本算子为 Cube/Vector 混合类（MixCV）：Cube 侧分形布局由 `T.gemm`/`b_transpose` 内部承载，Vector 侧 softmax 链、掩码链与累加链须枚举轴候选。

**Vector 侧候选矩阵**（fp16 向量宽度 ×8 = 128bit；行向量 = [rm,1] 广播语义；**沿用定长版方案 #1**）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **[rm, bn] 行主序 tile（S/P 链 + 掩码链）+ [rm, dim] 行主序 tile（O 链），lane = 列连续轴** | 列轴（l 维 / d 维，stride=1 连续） | **无需 repack**——ws_s/ws_p 的 [bm, pad_band] 行主序与 UB tile 同构，切片 copy 直达 | bn=256 fp16 → 每行 32 个 8-lane 向量组，尾 lane 浪费 0%；行广播（vmul [M,N]×[M,1]）沿行维展开为常数广播；行归约 `T.reduce(dims=[1])` 沿连续轴归约 | ✅ **主选** |
| 2 | [bn, rm] 列主序（lane = 行轴） | 行轴 | 需核内转置（`T.transpose` 二轴交换——UB 级执行但 **VP-D6 实测活跃源 transpose 毒化整 kernel 2.6×**） | 行广播退化为跨步广播；归约沿跨步轴——**transpose 毒化税 2.6× 直接否决**（traps-compiler.md VP-D6） | ❌（毒化实测 + 广播退化双重否决） |
| 3 | 单 AIV 全行 [bm, bn]（不半分） | 列轴 | 无 | UB 预算翻倍（154KB → ~300KB » 192KB ✗ 编译失败）；放弃 2×AIV 并行 | ❌（UB 容量硬上限否决；`CONST-capacity-910B2C` UB 192KB/AIV） |
| 4 | GPU 源码轴映射（warp 32 线程 × `GemmWarpPolicy.FullRow`） | — | — | — | ❌（GPU thread/warp 轴与 NPU Vector 轴不对应——migration-analysis.md §5.3；本设计 Vector 轴独立评估，见 #1） |

**掩码式选择（沿用定长版结论 + varlen 扩展）**：

| 候选 | 机制 | 代价 | 依据 | 决策 |
|------|------|------|------|------|
| **算术惩罚式 + offset 平移 + per-row 列 OOB** | 全 ALU 向量路径 | ~8 op/块，含掩码 | PL-1.11 第五轮实测（vcmp int16 全形态标量化）；本容器 **P-V2 探针实测 7/7 bit-exact 且零标量化 warning**（`design_probe_varlen_mask.py`） | ✅ **主选（常开）** |
| `vcmp`+`vand`+`vselect` | bool 掩码选择语义 | vcmp 标量化（aiv_scalar 主导） | PL-1.11 实测标量化；定长版 P2 复现 2 条 `hivm.hir.vcmp … scalar instruction` warning | ⏸ **备选**（实验裁决；加法掩码对 NaN 无效，须配合载体零初始化） |

**Cube 侧决策（沿用定长版）**：分形 NZ 布局由 `T.gemm(l1_a, l1_b, l0_c, b_transpose=True, size=[tmc,dim,tnc])` 内部承载（T.gemm.md §2.1）；**不采用** `T.load_nd2nz`（CG-2026-0004：对 packed NHD 定头跨步切片静默平铺误读——**必须 slice-form `T.copy`**）；L1 缓冲 l1_a/l1_b 生命周期复用（pass-1 Q/K → pass-2 P/V，`block_share = max(bn_eff, dim)`）；epilogue 归属：归一化/lse/cast 全在 Vector 侧（L0C 不能被 v 算子直接操作，出数必经 GM ws，V5/CG-2026-0005）。

**布局决策结论**：选定方案 #1——Vector 链核内布局 = 行主序 [rm,bn]/[rm,dim] tile、向量化轴 = 列连续轴、repack = 无；掩码 = **offset 平移算术惩罚式 + per-row 列 OOB（常开）**，备选 vselect 仅实验裁决；Cube 侧 = `T.gemm` 内部分形 + slice-form copy 装载 + L1 生命周期复用。**该结论同步落入 §3.3 伪代码与 §6 循环结构**（累加循环的内层向量维 = 列轴）；弃选方案量化理由见候选矩阵（#2 毒化税 2.6×、#3 UB 超限、#4 轴系不对应）。

**实验裁决计划（D-3，未知常数依赖）**：主选（算术惩罚式）与备选（vselect）的判定裕度依赖未在本容器实测的量——`vmin/vmax` 的向量吞吐与 `vcmp` 标量化税的相对比（PL-1.11 常数绑定 910B2C / CANN 8.5.0，本容器为 910B3 / CANN 9.0.0，跨 SoC/工具链未重验）。**主选 = 算术惩罚式**（进 Stage 3）；**备选 = vselect 形态**（结构完整：掩码链替换为 `arange int16 colmat/rowmat` + `vcmp le/ge` + `vand` + `vselect(−inf 哨兵)` + 载体零初始化，其余结构及分核不变）；**裁决计划**：代表 shape = smoke + 8b-short/8b-long × fp16，测 `msprof op Task Duration`（median of 20），判定阈值 = 备选 latency < 主选 × 0.95 才翻转，结果回写 §1.6.3。**设计期探针（P-V2）**：本容器实测**主选链 7/7 配置 bit-exact PASS 且零标量化 warning**（含运行期 offset + per-row 列 OOB + 常开尾带）——结论已写入 §0.6 V2 与本节主选理由（探针脚本 `history_version/design_probe_varlen_mask.py`）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Expert**（显式双 Scope）。**证据化结论**：本算子为 MixCV（Cube `T.gemm` + Vector softmax/掩码链），**Developer 模式（auto CV-split）在本容器上不可行**——设计期探针 **P-V1 在本结构上复现**（下述）。切 Expert 的决策**非默认假定，而是探针实测 + 先例 + 同族定长版三重证据**。

### 2.2 选型理由

1. **结构表达力（决定性）**：选定算法（§1.6.0 #2 两相位 + 运行期带状裁剪）要求 `T.Scope("Cube")`/`T.Scope("Vector")` 显式双引擎分工、两相位独立大循环的手动调度、`T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub` 显式分层、per-带内块 `T.sync_block_set/wait` 手动握手、`T.rs("PIPE_*")` 通道标注——均为 Expert 形态 API（`T.load_nd2nz.md §2.3`「适用 Expert 模式」同族口径；`TILELANG_ASCEND_MODE` 默认即 Expert——docs/developer/EnvironmentVariables.md L31）。**结构原型定长版与 MHA 母本均为 Expert 形态**。

2. **Developer 模式对照（本结构探针 P-V1 实测，`history_version/design_probe_varlen_structure.py`）**：

| 探针变体 | 结构 | 结果 |
|---------|------|------|
| A | Developer：persistent（`T.Kernel(4,is_npu=True)`+`T.serial(2)`）+ `T.gemm`，**无 v-prefix** | **PASS**（编译+运行、数值 finite） |
| B | Developer：同上 + `T.vmul(acc, fac, acc)`（**v-prefix 混排**） | **BLOCKED**（BishengIR 编译崩溃 `std::out_of_range` / `map::at`，`ERR99999`） |
| C | Expert：显式双 Scope（`alloc_L1/L0C/ub` + `PIPE_*`）同结构 | **PASS**（编译+运行） |

   **结论**：Developer auto CV-split **无法承载**「persistent 分核 + `T.gemm` + v-prefix 向量 op」的混排（变体 A 无向量 op 时正常，变体 B 加向量 op 即崩）；Expert 双 Scope 同结构正常。**本容器（tilelang `0.1.2+76e4550…` / CANN 9.0.0 / Ascend910B3）的失败点位于 BishengIR 编译期**（`map::at` 越界崩溃，无编译期诊断），与 CG-2026-0010 记录的运行时崩溃失败层不同但结论同向。**与同族定长版 P1 探针在本结构上复现一致**（定长版 toolchain `0.1.2+cf6f999…` 变体 A PASS / B BLOCKED / C PASS）——即失败结论在本工具链**复现**。CG-2026-0010 明确写：「persistent 混合算子在 Developer 模式不可实现——被迫切换 Expert（GQA/sparse_mla 先例）完成」；PL-1.16 给出判据（persistent 分核 + gemm 混排 → Expert 双 Scope bypass）。本算子正落在该缺口与判据的交集。

3. **Developer 可表达子集已被排除**：Developer 形态（fragment + 自动 cv_split + `T.Pipelined`）产出的是 per-block 串行链形态（定长版 MHA developer 基线：causal 域长 KV 比两相位慢 1.57–1.63×，PL-1.7）；两相位结构超出 Developer 表达力。

4. **同型先例**：定长版姊妹 `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/`（Expert 两相位，22/22 + 集成一次通过）与 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`、`examples/flash_attention/flash_attn_npuir.py` 均为 Expert 双 Scope——本算子与它们结构同型。

**备选（未采纳）**：**hybrid**（部分 Developer/部分 Expert）——无先例且本算子为单一 MIX kernel（不可拆分），不采用。**CG-2026-0010 能力缺口登记**：本算子复现了「Developer 对 persistent+gemm+向量混排的不支持」（`occurrences` 应 +1；缺口层 = codegen / BishengIR），§9.4 记录（供后续登记；本容器失败层为编译期）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 显式 `T.alloc_L1`（l1_a/l1_b）、`T.alloc_L0C`（l0_c 单缓冲）、`T.alloc_ub`（Vector 链全量缓冲，§4.5 预算） |
| 计算方式 | Cube：`T.gemm`（Scope("Cube") 内，initC/b_transpose/size）；Vector：v-prefix 链（vcast/vsub/vmax/vmin/vmul/vadd/vexp/reduce/vdiv/vlog2/vbrc） |
| 同步 | 手动 `T.sync_block_set/wait`（per-带内块 id + FLAG_TASKDONE，§7.2）；`T.rs("PIPE_*")` 通道标注 |
| 流水 | 手动两相位（无 `T.Pipelined`）；pass_configs 双关闭防编译器重排（§7.3） |
| 环境变量 | `TILELANG_ASCEND_MODE=Expert`（setdefault，docs/developer/EnvironmentVariables.md L31，显式防残留） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `cu_seqlens` 标量读 → `q_start/kv_start/q_len/kv_len/offset`；`real_m=min(bm, q_len−bx·bm)` | **varlen 前置（运行期）** |
| 1 | `S = q_tile @ k_tile^T`（size [tmc, dim, tnc]），`k_tile = k[kv_start+n_lo : …, kv_head, :]`，`n_lo=(k_start+l)·bn_eff` | QK^T（Cube pass-1，Q-hoist，带状块） |
| 2 | `S_f16 ← f16(S_f32)`；`Ŝ ← f32(S_f16)` | S 物化 + 到达（M8'；bf16 路径 f32 直传） |
| 3 | `pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1), clamp(diff−thr,0,1))`，`K_up=(bx·bm+row0+offset)−n_lo`，`thr=(tn_real−1)−i` | **offset 平移双侧掩码 + per-row 列 OOB**（M5'） |
| 4 | `S'' = prescale·Ŝ + pen·(−1e38)` | prescale（M1' 前置）+ 掩码（M5'） |
| 5 | `m_cur = max_j S''`；`m = max(m_prev, m_cur, −1e38)` | 行 max + 单调 + clamp（M3'/M6'） |
| 6 | `α_l = e^{m_prev−m}`；`P = e^{S''−m}` | α 与 P（M1' e 域） |
| 7 | `ℓ ← ℓ·α_l + Σ_j P`；`scales[l] ← α_l` | ℓ 递推 + α 存档（clear=True，M13'） |
| 8 | `P_dtype ← dtype(P)`；`O_partial(l) = P_dtype @ v_tile`（size [tmc, tnc, dim]） | P 量化（M10'）+ PV 部分和（Cube pass-2） |
| 9 | `o ← o·α_l + f32(O_partial_f16)` | 延迟 rescale 累加（Vector pass-2，M9'） |
| 10 | `out = o/ℓ`（写 packed 地址）；`lse = log2(ℓ) + m·LOG2E` | epilogue（M2'；transpose-free 写出 + ghost 行截断） |

### 3.2 TileLang API 映射（Expert 模式，逐段）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | cu_seqlens 标量读 | 元素级 `cu_q[bz]`（GM BufferLoad → 运行期标量 PrimExpr；**P-V1 实测**） | `q_start=cu_q[bz]` 等 | Expert |
| 1 | Q/K/V 装载 | `T.copy`（slice 形态） | `T.copy(q[q_start+bx·bm : q_start+bx·bm+real_m, by, 0:dim], l1_a[0:real_m,0:dim])`；k/v 同款（**packed NHD 定头切片必须 slice 形态**，CG-2026-0004） | Expert |
| 1 | QK^T | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tmc, dim, tnc])`（T.gemm.md §2.1） | Expert |
| 2 | S 物化 | `T.copy` | `T.copy(l0_c[0:tmc,0:tnc], ws_s[kid, 0:tmc, l·bn_eff:l·bn_eff+tnc])`（f32→f16 隐式转换，PL-1.9） | Expert |
| 3 | 掩码 | `T.arange` + `T.vsub` + `T.vmax` + `T.vmin` + `T.vmul` + `T.vadd` | `diff` f32 [half,bn]（每核一次）；分段 `if`（**直接整数比较**，避 CG-2026-0024） | Expert |
| 4 | prescale | `T.vmul` | 标量广播（数学操作/T.vmul.md §2.2.2） | Expert |
| 5 | 行 max / 单调 / clamp | `T.reduce` + `T.vmax` ×2 | `dims=[1], reduce_mode="max", clear=True` | Expert |
| 6 | α / P | `T.vsub` + `T.vexp` | 行向量 / 行广播 | Expert |
| 7 | 行 sum + ℓ 更新 + α 存档 | `T.reduce` + `T.vmul`/`T.vadd` + `T.copy` | `reduce_mode="sum", clear=True`；`scales[l·half:…,0:1]` 切片存取 | Expert |
| 8 | P 量化 + 回传 | `T.vcast` + `T.copy` | `round_mode="rint"`；PIPE_MTE3 段内 copy + set flag | Expert |
| 8 | PV 部分和 | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, size=[tmc, tnc, dim])` | Expert |
| 9 | O 累加 | `T.vcast` + `T.vmul` + `T.vadd` | `vcast(ub_f16_D, ub_f32_D, "rint")`；行广播 rescale | Expert |
| 10 | epilogue | `T.vdiv` + `T.vcast` + `T.vlog2` + `T.copy` | `vlog2(ub_ell, ub_lse, ub_lse_tmp)`；行截断切片写出（`[H,total_q,1]` 视图） | Expert |
| — | 跨引擎握手 | `T.sync_block_set` / `T.sync_block_wait` | flag id = 带内块局部下标 l + FLAG_TASKDONE（§7.2） | Expert |
| — | 状态初始化 | `T.vbrc` | let-bound 标量（D4） | Expert |

### 3.3 计算伪代码

> `T.Tensor` 的 dtype 按位置参数传递。完整结构 = §0.6 V1–V7 的 Cube/Vector 流伪代码，此处补任务解码外壳与 trace 家族说明：

```python
# 工厂层（trace-time）
# bm = 64; bn_caller（wrapper 默认 64）；num_stages 保留旋钮
# eff_left  = wl if wl >= 0 else max_seqlen_q     # 无左窗=不裁剪（上界取 max q-block 跨度）
# eff_right = wr if (wr >= 0 and not is_causal) else 0   # causal 忽略 wr（源角点）
# L_max_bound = ceildiv(bm + eff_left + eff_right, bn_eff) + 2
# bn_eff = max(bn_caller or 256, ceil16(ceildiv(bm + eff_left + eff_right, 13)))  # flag 预算守卫
# assert L_max_bound + 1 <= 16                    # 运行期 loop_count 的解析上界
# pad_band = L_max_bound * bn_eff
# trace 家族（工厂级多 @T.prim_func 分派）: mask 域 {(causal+left)/(causal)/(left+right)/(none)}
#   × fp16/bf16（ws dtype: f16/f16/f16 vs f32/bf16/f32）

@T.prim_func
def _gqa_sw_fwd_varlen_main(
    q: T.Tensor((total_q, heads, dim), dtype),
    k: T.Tensor((total_k, heads_kv, dim), dtype),
    v: T.Tensor((total_k, heads_kv, dim), dtype),
    cu_q: T.Tensor((batch + 1,), "int32"),
    cu_k: T.Tensor((batch + 1,), "int32"),
    ws_s: T.Tensor((NUM_KERNELS, bm, pad_band), ws_s_dtype),
    ws_p: T.Tensor((NUM_KERNELS, bm, pad_band), dtype),
    ws_o: T.Tensor((NUM_KERNELS, bm, dim * L_max_bound), ws_o_dtype),
    zbuf: T.Tensor((bn_eff, dim), dtype),
    output: T.Tensor((total_q, heads, dim), dtype),
    lse: T.Tensor((heads, total_q, 1), "float32"),      # [H,total_q,1] 视图（V7，零 transpose）
):
    with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
        HB = heads * batch
        num_local = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)
        # per-core once: diff = arange(col) - arange(row); rowi = arange(row)（[half,bn_eff] f32）
        with T.Scope("Cube"):
            l1_a = T.alloc_L1([bm, block_share], dtype); l1_b = T.alloc_L1([bn_eff, dim], dtype)
            l0_c = T.alloc_L0C([bm, block_share], "float32")
            for task_id in T.serial(num_local):
                cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                bz = cid % batch; by = (cid // batch) % heads; bx = cid // HB
                q_start = cu_q[bz]; kv_start = cu_k[bz]
                q_len = cu_q[bz+1] - q_start; kv_len = cu_k[bz+1] - kv_start; offset = kv_len - q_len
                real_m = T.min(bm, q_len - bx*bm); kv_head = by // groups
                tmc = T.max(16, T.min(bm, T.ceildiv(real_m, 16) * 16))       # 尾钳位（real_m<=0 -> skip）
                k_end = (T.ceildiv(T.min(kv_len, offset + (bx+1)*bm), bn_eff) if causal
                         else T.ceildiv(kv_len, bn_eff))
                k_start = (T.max(0, offset + bx*bm - wl) // bn_eff if (wl >= 0) else 0)
                loop_count = (0 if real_m <= 0 else T.max(k_end - k_start, 0))   # ghost 空转
                if task_id > 0:
                    with T.rs("PIPE_MTE2"): T.sync_block_wait(FLAG_TASKDONE)
                with T.rs("PIPE_MTE2"):
                    T.copy(q[q_start+bx*bm : q_start+bx*bm+real_m, by, 0:dim], l1_a[0:real_m,0:dim])
                for l in T.serial(loop_count):     # pass-1 (全 S，无等待直跑)
                    n_lo = (k_start + l) * bn_eff; tn_real = T.min(bn_eff, kv_len - n_lo)
                    tnc = T.max(32, T.min(bn_eff, T.ceildiv(tn_real, 16) * 16))
                    # K 装载 + gemm1 + l0_c→ws_s + set(l)
                for l in T.serial(loop_count):     # pass-2 (全 PV)
                    # wait(l) + ws_p→l1_a + V 装载 + gemm2 + l0_c→ws_o + set(l)
        with T.Scope("Vector"):
            # ub 缓冲族（§4.5）；diff/rowi ([half,bn_eff] f32) 任务循环外一次构建
            for task_id in T.serial(num_local):
                # 同款任务解码（同 cu_seqlens 读）；real_m 自适应半分；per-AIV K_up/K_lo/thr
                for l in T.serial(loop_count):     # pass-1: 掩码 → softmax 链 → P 量化 → ℓ 递推
                    ...
                for l in T.serial(loop_count):     # pass-2: 延迟 rescale 累加
                    ...
                # epilogue + 行截断写出 + set FLAG_TASKDONE
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| **核内 GM 标量读 `cu[i]`**（int32 → 运行期 PrimExpr 用于 slice 起点/extent/循环界/输出地址） | 源 L173–177 同形态；本仓 `examples/` attention 族无先例 | ✅ **P-V1 探针设备实测 PASS（maxdiff=0.0）** |
| `T.gemm`（Expert 签名 initC/b_transpose/size） | docs/Tilelang.language/线性代数操作/T.gemm.md §2.1/§2.2.1/§2.3；定长版母本 | ✅ 定长版生产实测 |
| `T.copy`（slice 形态 GM↔L1/UB、L0C→GM，跨 dtype 隐式转换；**packed NHD 定头 slice**） | debug_log（定长版）；PL-1.9（f32→f16 转换）；P-V1 | ✅ P-V1 + 定长版实测 |
| `T.vcast`（rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md；定长版母本 | ✅ |
| `T.reduce`（dims=[1]，clear=True） | docs/Tilelang.language/规约操作/T.reduce.md §2.2.2/§2.3；定长版母本 | ✅ |
| `T.arange`/`T.vsub`/`T.vmax`/`T.vmin`/`T.vmul`/`T.vadd`/`T.vexp`/`T.vdiv`/`T.vlog2`/`T.vbrc` | docs/Tilelang.language/ 对应子目录逐一存在性核验；**offset 平移掩码链 + per-row 列 OOB + 运行期阈值经本容器 P-V2 探针 7/7 bit-exact 实测** | ✅ 实测 |
| `T.sync_block_set/wait`（带内块 id 复用 + TASKDONE） | docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1；定长版母本 | ✅ 定长版生产实测（+ CG-2026-0023 sync shim） |
| `T.Kernel(20, is_npu=True) as (kernel_id, subid)` | 定长版生产（persistent）；P-V1 variant C | ✅ P-V1 实测 |
| `T.rs("PIPE_*")` 通道标注 | mixcv/deepseek 先例；定长版母本；P-V1/P-V2 | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测 5 项）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 grid） | V1：一维 persistent 20 核 + ragged 任务解码（bx/by/bz 派生） |
| L0C 容量 128KB | **Yes**（l0_c [bm, block_share] f32） | 单缓冲 [64, 256] f32 = 64KB ≤ 128KB ✓（`CONST-capacity-910B2C`） |
| L1 容量 512KB / UB 192KB | **Yes** | L1 = l1_a 32KB + l1_b 64KB = 96KB ✓；UB ≈ 154KB per AIV（§4.5 逐项）✓ |
| GEMM 非整除 / 小尺寸分形 | **Yes**（运行期 `real_m < 16`/`tn_real < 32`、`kv_len % bn ≠ 0`） | 尾钳位（tmc/tnc 分形下限 16/32）+ **常开尾带**（N 载体零初始化 + per-row 列 OOB + zbuf）+ 行截断写出 + ghost 空转（V4/V7） |
| store_fixpipe / L0C 出数 dst 仅 GM | **Yes**（S/O 跨引擎） | V5：ws 三族 GM 物化（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1；CG-2026-0005） |
| **Developer MIX 不可用** | **Yes**（persistent + gemm + v-prefix） | §2：Expert 双 Scope（**P-V1 探针复现** + CG-2026-0010 + PL-1.16） |
| **`T.sync_block_set` 编译失败（CANN 9.0.0）** | **Yes**（两相位手动握手） | **CG-2026-0023 运行期 shim**（`syn_instr_mode`→`sync_instr_mode`，逐语句沿用定长版，§7.4） |
| **`T.floordiv/ceildiv` 负分子截断（CG-2026-0024）** | **Yes**（两段式分界谓词） | VM-bnd：块分类谓词一律用**无除法直接整数比较**（`n_lo > bx·bm+row0+offset−(bn−1)` / `n_lo < bx·bm+row0+rm−1+offset−wl`） |
| **`vcmp` int16 标量化（PL-1.11）** | **Yes**（掩码） | VM-mask：算术惩罚式（无 vcmp；P-V2 零 warning） |
| **stale UB 尾列 NaN（TRAP-…-nan）** | **Yes**（运行期 ragged 恒可能） | V4：N 载体每核零初始化 + 全宽 per-row 列 OOB（P-V2 NaN-free） |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | 三维 `T.Kernel(m, H, B)` + threads | 一维 persistent 20 + ragged 任务解码 | V1 |
| **布局** | **packed NHD**（`(total,H,D)`） | **packed NHD**（同）但寄存器/SMEM 为 L1/UB | V6（slice-form copy 定头切片） |
| **per-sample 长度** | 逐线程标量 GM 读 | **核内 GM 标量读**（P-V1 实测） | V1 |
| GEMM API | `T.gemm(transpose_B, policy=FullRow)` fragment | `T.gemm(L1, L1, L0C, initC, b_transpose, size)` | §3.2；FullRow 舍弃 |
| 内存分配 | alloc_shared/fragment 自动 | alloc_L1/L0C/ub 显式分层 + GM ws | Expert 模式 |
| 流水 | `T.Pipelined`（cp.async） | 两相位手动调度 + per-带内块 flag | V5 |
| **带状 KV 范围** | `k_start`/`k_end`（运行期 kv_len/offset） | 同式移植，`for l in T.serial(loop_count)` | V3 |
| **掩码** | `T.Parallel` 谓词 `if_then_else`（运行期） | offset 平移算术惩罚 + per-row 列 OOB | V2（PL-1.11 + P-V2） |
| 尾块处理 | 谓词逐元素装载/回写 | size-form + 尾钳位切片 + 常开尾带 + 行截断 | V4/V7 |
| **lse 写出** | `T.copy` 2D `[H,total_q]` | `[H,total_q,1]` 视图 + 自然 2D 区域写 + host reshape | V7（VP-D6） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/`（DESIGN.md + `_gqa_sw_fwd_kernel.py`） | **极高（直接母本）** | Expert 两相位、per-块 flag、单 L0C 复用、尾钳位 size-form、scales 回放、自适应行半分、ws 三族、Q-hoist、transpose-free lse、sync shim、双侧算术惩罚掩码 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高 | GQA prefill Expert 两相位（更早母本） |
| `examples/flash_attention/flash_attn_npuir.py` | 中（结构原型） | 两相位大循环、双 Scope、sync_block_set/wait 最小形态 |
| `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/history_version/design_probe_devmode_mix.py` | 中（探针范式） | MIX 模式判定探针（本任务 P-V1 据此改写） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | (total_q, heads, dim) | float16 / bfloat16 | packed NHD；**total_q 编译期静态**（工厂 key） |
| k | (total_k, heads_kv, dim) | same_as(q) | GQA：`heads % heads_kv == 0` |
| v | (total_k, heads_kv, dim) | same_as(q) | |
| cu_seqlens_q | (batch+1,) | int32 | 累积 q 长度；运行期核内标量读 |
| cu_seqlens_k | (batch+1,) | int32 | 累积 kv 长度；运行期核内标量读 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| output | (total_q, heads, dim) | same_as(q) | Op 契约输出（wrapper 返回） |
| lse | (heads, total_q) | float32 | kernel 第 2 输出；prim_func 声明 `[H,total_q,1]` 视图（V7）；**Op wrapper 丢弃** |

### 4.3 中间缓冲区（Expert 显式分层；设计默认 bm=64, bn_eff=256, dim=128, L_max_bound≤6, half=32）

| 层级 | buffer | shape | dtype | 生命周期 / 说明 |
|------|--------|-------|-------|-----------------|
| L1 | l1_a | [bm, block_share] = [64, 256] | dtype | pass-1 持 Q（rows×dim）/ pass-2 持 P（rows×tnc）——生命周期复用（block_share = max(bn_eff,dim)） |
| L1 | l1_b | [bn_eff, dim] = [256, 128] | dtype | pass-1 持 K / pass-2 持 V |
| L0C | l0_c | [bm, block_share] = [64, 256] | float32 | 单缓冲，gemm1/gemm2 顺序复用（64KB/128KB） |
| GM ws | ws_s | [NUM_KERNELS=20, bm, pad_band] | fp16: f16 / bf16: f32 | Cube→Vector 的 S 块 |
| GM ws | ws_p | [20, bm, pad_band] | 输入 dtype | Vector→Cube 的 P 块 |
| GM ws | ws_o | [20, bm, dim·L_max_bound] = [20,64,768] | fp16: f16 / bf16: f32 | Cube→Vector 的 PV 部分和 |
| GM | zbuf | [bn_eff, dim] | dtype | 尾带载体零初始化（V4，PL-1.11） |
| UB | 见 §4.5 | — | — | Vector 链全量缓冲 |

`pad_band = L_max_bound · bn_eff`。ws 总量（8b-long，fp16，L_max_bound=6, pad_band=1536）：ws_s = 20·64·1536·2 = 3.93MB；ws_p = 3.93MB；ws_o = 20·64·768·2 = 1.97MB；合计 **≈ 9.8MB**（bf16 路径 ws_s/ws_o 升 f32 ≈ 16.6MB）——L2 下与 KV 工作集共存无压力。

### 4.4 内存搬运路径

```
GM[cu_q,cu_k] ─核内标量读─> q_start/kv_start/q_len/kv_len/offset
GM[q] ─slice T.copy(Q-hoist, 每任务一次)─> L1[l1_a]
GM[k] ─slice T.copy(每带内块)─> L1[l1_b] ─┐
   L1[l1_a]×L1[l1_b] ─T.gemm(b_transpose)→ L0C[l0_c] ─T.copy(FIX, f32→f16)─> GM[ws_s] ─set(l)
GM[ws_s] ─MTE2─> UB[ub_f16_N] ─vcast─> UB[ub_f32_N] ─掩码/softmax 链─> UB[ub_f16_N(P)]
UB[ub_f16_N] ─MTE3─> GM[ws_p] ─set(l) ─> L1[l1_a] ─┐
GM[v] ─slice T.copy(每带内块)─> L1[l1_b] ─┘
   L1[l1_a]×L1[l1_b] ─T.gemm(initC=True)→ L0C[l0_c] ─T.copy(FIX)→ GM[ws_o] ─set(l)
GM[ws_o] ─MTE2─> UB[ub_f16_D] ─vcast─> UB[ub_f32_D] ─vmul(α)+vadd─> UB[acc_o]
UB[acc_o]/ub_ell ─vdiv/vlog2/vcast─> UB ─行截断 slice─> GM[output[...,by,:]] / GM[lse[by,..,0:1] 视图]
```

**layout 变换点**：① GM→L1 的 packed NHD 定头切片（slice-form `T.copy`，**禁止 `T.load_nd2nz`**——CG-2026-0004）；② L0C→GM 的 f32→f16 隐式量化（PL-1.9）；③ UB→GM 行截断切片（src `[0:rm,...]` 与 dst 同形）。**无核内重排、无 transpose**。

### 4.5 UB 内存预算（设计默认 bm=64/bn_eff=256/dim=128/L_max_bound=6/half=32；per AIV）

| buffer | shape | dtype | 字节 |
|--------|-------|-------|------|
| ub_f16_N | [32, 256] | float16 | 16 KB |
| ub_f32_N | [32, 256] | float32 | 32 KB |
| ub_f16_D | [32, 128] | float16 | 8 KB |
| ub_f32_D | [32, 128] | float32 | 16 KB |
| acc_o | [32, 128] | float32 | 16 KB |
| ub_diff / ub_rowi / ub_pen / ub_thr / t1/t2/t3 | [32, 256] f32 × 5（diff/pen/thr/t1/t2） | float32 | 160 KB→**见下** |
| ub_m / ub_mprev / ub_mcur / ub_alpha / ub_ell / ub_ellcur / ub_t / ub_lse / ub_lse_tmp | [32, 1] × 9 | float32 | 1.1 KB |
| scales | [L_max_bound·half, 1] = [192,1] | float32 | 0.77 KB |

> **UB 预算收紧（varlen 新增掩码缓冲）**：定长版掩码缓冲（diff/pen 各 [32,256] f32 = 64KB）已在 154KB 预算内；varlen 版新增 per-row 列 OOB 需 `rowi`/`thr`（另 2×[32,256]f32 = 64KB）——**须原位复用**：`ub_thr` 复用 `ub_pen` 之后的临时区、`t1/t2/t3` 串行复用同一 scratch（同一 PIPE_V 段顺序执行，无交叠生命段）。按原位复用后预算 ≈ **154KB**（同定长版：ub_f16_N 16 + ub_f32_N 32 + ub_f16_D 8 + ub_f32_D 16 + acc_o 16 + ub_diff 32 + ub_pen 32 + 行向量 ~1.1 + scales 0.77 ≈ 154KB）≤ 192KB ✓（`CONST-capacity-910B2C`；余量 ~38KB；auto-multi-buffer=false 下显式 alloc 结构实测 ~0.98–1.0×，CG-2026-0008 反例）。**diet 候选见 §9**。

### 4.6 动态轴定义

**无运行时动态 shape**——batch/heads/heads_kv/total_q/total_k/dim/max_seqlen_q 均工厂期静态（`lru_cache` key 的一部分）；**per-sample `q_len/kv_len/offset` 是运行期值**（从 GM 读），但不是 shape——它们只进入 PrimExpr（slice 起点/extent/循环界/掩码阈值），不改变张量形状。`k_start/k_end/loop_count/tmc/tnc/real_m/tn_real/L_*` 为 kernel_id/bx + cu_seqlens 派生 PrimExpr（静态可 lower，**P-V1 实证**）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-2, -1],                       # output, lse 居末两位（源 out_idx=[6,7] 同构）
    target="npuir",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    },
)
```
`TILELANG_ASCEND_MODE=Expert`（setdefault）。CUDA 的 `TL_ENABLE_FAST_MATH` / `compile_flags` 不迁移（§0.5 舍弃）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube GEMM × 2 + Vector softmax/掩码/归约/累加链）——**MixCV**（同一 kernel 含 Cube 侧 `T.gemm` 与 Vector 侧多个 v-prefix 算子，AGENTS.md Developer-mode MixCV 判据同款结构，Expert 形态）。

**判定依据**: QK^T 与 PV 为 MAC 密集段（Cube），softmax 链/offset 平移掩码/累加/epilogue 为逐元素与规约段（Vector）；两相位结构以 per-带内块 flag 协同双引擎。

### 5.2 Block 划分

```python
bm = 64        # 设计默认。依据：① UB 预算（§4.5：bm=64 时 ~154KB ✓）；② 与 wrapper 默认 block_m 一致；
               #   ③ L0C [bm,256] f32 = 64KB 余量大；④ 自适应半分 half=bm//2=32。
bn_eff = 256   # 设计默认（caller 传 wrapper 默认 (64,32) 时替换）。依据：① 向量发射开销定律（PL-1.9-hardlimits）；
               #   ② flag 预算守卫（V5）；Guard: bn_eff = max(bn_caller, ceil16(ceildiv(bm+eff_left+eff_right, 13)))
L_max_bound = ceildiv(bm + eff_left + eff_right, bn_eff) + 2     # 运行期 loop_count 解析上界（varlen）
# assert L_max_bound + 1 <= 16
# GEMM1 tile：M=tmc(≤bm, ≥16)、N=tnc(≤bn_eff, ≥32)、K=dim(64/128 ≥32)——QK^T（b_transpose=True）
# GEMM2 tile：M=tmc、N=dim、K=tnc(≥32)——PV（initC=True 每块覆写）
```

**目标域 config 核验表**（设计默认 bn_eff=256/bm=64 下的有效值；benchmark 将 total_q/total_k 均分给 batch）：

| workload | total_q/total_k | q_lens/k_lens | max_q | num_q_blocks | H×Hkv | B | num_logical | L_max_bound | 尾块/尾带 |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 256/256 | [128,128] | 128 | 2 | 8×2 | 2 | 32 | 3 | 128%256≠0 → **尾带命中** |
| 8b-short | 2048/2048 | [512]×4 | 512 | 8 | 32×8 | 4 | 1024 | 3 | 512%256=0 → 无 |
| 8b-long | 8192/8192 | [2048]×4 | 2048 | 32 | 32×8 | 4 | 4096 | 6 | 2048%256=0 → 无 |
| 70b-short | 2048/2048 | [512]×4 | 512 | 8 | 64×8 | 4 | 2048 | 3 | 无 |
| 70b-long | 8192/8192 | [2048]×4 | 2048 | 32 | 64×8 | 4 | 8192 | 6 | 无 |

（**执行块数红利**：8b-long **15360 块 vs 全 causal 18432 块 = −16.7%**；内段全有效块 7680、边界掩码块 7680；**wl=256 短序列块粒度下带未越界，执行块数与全 causal 同（0 减免）**。**smoke 域名 `kv_len=128 % bn_eff=256 ≠ 0` → 常开尾带机制在 manifest 域即触发**——这是 varlen 与定长版的关键差异：定长版 manifest 全域整除、尾带机制零触发。）

### 5.3 约束分析

- **对齐约束**: `bn_eff` 与 `dim` 为 16 倍数（守卫公式 ceil16 保证；dim ∈ {64,128}）；fp16/bf16 尾轴 256·2B = 512B ≥ 32B ✓；分形 M/N ≥ 16、K ≥ 32 由尾钳位保证。
- **L1 容量**: l1_a + l1_b = 96KB ≤ 512KB ✓。
- **L0C 容量**: l0_c [64,256] f32 = 64KB ≤ 128KB ✓。
- **UB 容量**: ~154KB ≤ 192KB per AIV ✓（§4.5，掩码缓冲原位复用）。
- **flag 预算**: `L_max_bound + 1 ≤ 16` ✓（T.set_flag.md §2.1 event_id 0–15；V5 守卫公式强制）。
- **带状范围**: `loop_count ≥ 0`（ghost 块 ⇒ 0）；`k_start ≤ k_end` 由源式保证（`max(0,offset+bx·bm−wl) ≤ offset+(bx+1)bm`，因 `wl ≥ 0` 且 `bx·bm ≤ (bx+1)bm`）。

### 5.4 注意事项

- **运行期 ragged**（varlen 核心）：每样本 `kv_len % bn_eff ≠ 0` → 尾带常开（V4）；每 q-block `real_m ≤ bm`、ghost 块 `real_m ≤ 0`（V1/V7）。
- **带状边界块**：每任务至多 2 个边界块（左窗 + 因果）走掩码链；中段全有效块经 VM-bnd 零掩码——掩码开销为结构性常数。
- **bn_eff 守卫**：`wl` 越大要求 `bn_eff` 越大才能压住 `L_max_bound`（V5 公式）；manifest 全域 bn_eff=256 合规（L_max_bound ≤ 6）。
- **bm 上界**：L0C 公式 bm·max(bn_eff,dim)·4 ≤ 128KB → bm ≤ 128@bn=256；UB 公式约 (0.6·bm + 116)·1024 ≤ 192KB → bm ≤ 118——均不先绑定（bm=64 取安全值）。
- **prescale ≠ 1**：本算子 `prescale = dim^{−1/2} ≠ 1` 恒成立。
- **packed 地址**：全部 GM 访问地址 = `q_start/kv_start + 块偏移`（运行期），禁止假定 batch 维（packed 无 batch 维）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **物理核数（要素②，实查）**: **20**。查询代码与返回值记录：

```python
from tilelang.utils.npu_utils import NPUUtils
u = NPUUtils.get()
print(u.get_arch(), u.get_aicore_num(), u.get_aivector_core_num())
# 2026-10-16 18:52 实查输出：Ascend910B3 20 40
```

（混合算子是 Cube+Vector，直接使用 `get_aicore_num()` 返回值 **20**——本容器 `get_arch()` = `Ascend910B3`。Cube/Vector 混合核数按 Cube 计数 20；双 AIV 由 `subid` 分片。）

- **逻辑核数（要素①）**: `num_logical = ceildiv(max_seqlen_q, bm) × heads × batch`（bm=64 设计默认）：

| workload | max_seqlen_q | num_q_blocks | num_logical | 与 20 核关系 |
|---|---|---|---|---|
| smoke | 128 | 2 | 2×8×2 = **32** | 1.6×（**欠载**） |
| 8b-short | 512 | 8 | 8×32×4 = **1024** | 51.2×（饱和） |
| 8b-long | 2048 | 32 | 32×32×4 = **4096** | 204.8×（饱和） |
| 70b-short | 512 | 8 | 8×64×4 = **2048** | 102.4×（饱和） |
| 70b-long | 2048 | 32 | 32×64×4 = **8192** | 409.6×（饱和） |

- **规模判定（要素③）**: **极大规模**（4/5 workload 逻辑任务数 1024–8192 » 20，无法通过调整分块缩减到核数量级——bm 上调受 UB/L0C 双上限约束且减少任务数与均衡性冲突）；**smoke 32 任务为欠载**（32/20 = 1.6 任务/核，~2 波次，核心部分空闲）。**无「无法对齐兜底」情形**（逻辑核数非被钉为 2 的幂）。
- **分核方案**: **固定启动内核数 = 物理核数 20，核内 `T.serial` 串行处理多个逻辑任务**：`T.Kernel(20, is_npu=True)`，每核 `num_local = T.ceildiv(num_logical − kernel_id, 20)` 个任务、任务解码 `cid = T.min(task_id·20 + kernel_id, num_logical − 1)`（轮转——**运行期带状 loop_count 随 bx 先增后减**（wl≥0）/ 单调增（wl<0），轮转使每核 bx 均匀覆盖，负载均衡）。循环边界为 kernel_id 派生的 PrimExpr 表达式（进入循环前一次计算、循环内不变——**P-V1 实证可 lower**）。**smoke 欠载处置**：32 任务下 20 核部分忙（12 核 2 任务 / 8 核 1 任务，不均衡比 2.0×）——Stage 4 可按 shape 分派 bm=32（64 任务）或依赖两相位单任务流水性吸收，不在 Stage 1 特化。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（per 核） | persistent 串行 | `for task_id in T.serial(num_local)` | 极大规模分核（§5.5）；runtime PrimExpr 边界（P-V1 实证） |
| 带内块 pass-1 维（Cube） | 串行大循环 | `for l in T.serial(loop_count)` | 两相位全 S（无等待直跑）；`loop_count = k_end − k_start`（**运行期**，V3） |
| 带内块 pass-2 维（Cube） | 串行大循环 | `for l in T.serial(loop_count)` | 两相位全 PV（逐块等 P） |
| 带内块 pass-1/pass-2 维（Vector） | 串行大循环 ×2 | `for l in T.serial(loop_count)` | softmax 链 + 掩码 / 延迟累加 |
| 元素级 | **向量化（无循环）** | v-prefix 链 + `T.reduce` | §1.6.2 全覆盖（15 项 ✅）；无逐元素标量循环 |

### 6.2 循环伪代码

```python
with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
    # 每核一次：ub_diff = arange(col) - arange(row)；ub_rowi = arange(row)（[half,bn_eff] f32）
    with T.Scope("Cube"):
        for task_id in T.serial(num_local):
            # cid→(bx,by,bz)；cu_seqlens 标量读 → q_start/kv_start/q_len/kv_len/offset；real_m
            # k_end/k_start/loop_count（V3，运行期）；tmc/tnc 钳位；ghost → loop_count=0
            if task_id > 0:
                with T.rs("PIPE_MTE2"): T.sync_block_wait(FLAG_TASKDONE)   # 单槽跨任务 WAR
            with T.rs("PIPE_MTE2"): T.copy(q[q_start+bx*bm:…, by, 0:dim], l1_a[0:real_m,0:dim])
            for l in T.serial(loop_count):        # pass-1（§0.6 V5 Cube 流 + V3）
                ...
            for l in T.serial(loop_count):        # pass-2
                ...
    with T.Scope("Vector"):
        for task_id in T.serial(num_local):
            # 同款任务解码 + cu_seqlens 读；real_m 自适应半分；per-AIV K_up/K_lo/thr 阈值
            for l in T.serial(loop_count):        # pass-1（V2 offset 平移掩码 + softmax 链）
                ...
            for l in T.serial(loop_count):        # pass-2
                ...
            # epilogue + 行截断写出（output/lse）+ set FLAG_TASKDONE（§0.6 V5/V7）
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**（Expert 手动两相位，§0.5/§2）。流水机制 = 两相位结构自带的跨引擎重叠：① Cube pass-1 的 MTE2/MTE1/C/FIX 四通道硬件排队自然重叠（无软件等待插入）；② Cube pass-2(l) 与 Vector pass-1(l'>l) 经 per-带内块 flag 重叠推进；③ Vector pass-1 的 MTE2/V/MTE3 三通道重叠。**槽位安全性论证**：单槽 + 带内块 id 严格交替 + TASKDONE 边界排空——每个 flag id 任一时刻至多一个未决事件；ws 生命段（pass-1 写 ws_s[l] → Vector 读 → Vector 写 ws_p[l] → Cube 读 → Cube 写 ws_o[l] → Vector 读）全部由同一 id 的三次握手定序。

### 6.4 尾块处理（**常开**——varlen 运行期 ragged）

**manifest 域 smoke 即触发**（`kv_len=128 % 256 ≠ 0`）；全机制：

1. **q 尾块/ghost 块**：`real_m = min(bm, q_len − bx·bm)`（运行期）；`tmc = max(16, min(bm, ceil16(real_m)))`——gemm 计算宽 tmc（垃圾行 [real_m, tmc) 为 l1_a 残留），Vector 自适应半分只覆盖真实行（`real_m0 = (real_m+1)//2`），输出行截断写出；**`real_m ≤ 0` 的 ghost 块 ⇒ `loop_count=0` + 跳过 Q 装载 + `rm ≤ 0` 不写**（与源 `q_oob` 守卫同效）。
2. **kv 尾块无带域**（`tn_real ≥ 32 且 16 | tn_real`）：`tnc = tn_real`。
3. **kv 尾带域**（`tn_real < 32 或 16 ∤ tn_real`，**运行期判定，机制常开**）：`tnc = max(32, min(bn_eff, ceil16(tn_real)))`；Vector 对 `j ≥ tn_real` 列以 **per-row 列 OOB 算术惩罚**（`clamp(diff − ((tn_real−1)−i), 0, 1)`）强制 P=0；**同时 `ub_f16_N`/`ub_f32_N` 每核一次零初始化**（`TRAP-npuir-stale-ub-tail-columns-nan`：加法掩码无法清 NaN，须源头零化）；**`l1_b`(V) 每核一次 `zbuf` 零初始化**（PL-1.11：`P带(0)×V残留(NaN)=NaN` 整列污染，须源头零化）。**P-V2 实测**（partial-width copy + 零初始化 + per-row 列 OOB）7/7 sentinel 逐位 + NaN-free。
4. **全 OOB kv 块**：`loop_count` 由 `k_start/k_end` 保证不越界（V3）。
5. **GQA**：`kv_head = by // groups` 切片寻址。
6. **左/右窗不相等**：`wl<0 →` 左段 trace-time 剔除；`is_causal=False ∧ wr<0 →` 无上界（右段剔除）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 手动同步（Expert）——per-带内块 `T.sync_block_set/wait` + `T.rs("PIPE_*")` 通道标注；无 `T.Pipelined` / 自动同步依赖。

### 7.2 同步点说明（flag id 分配表 + 逐点理由）

**flag id 分配**（`L_max_bound ≤ 6`；**总 id 数 = L_max_bound + 1 ≤ 16 = 事件预算 0–15**——T.set_flag.md §2.1；V5 守卫公式强制 §5.2）：

| flag 族 | id | 方向 | 设置点 | 等待点 | 语义 |
|---------|---|------|--------|--------|------|
| 带内块三次握手 | l ∈ [0, L_max_bound) | Cube↔Vector（三次转向） | ① Cube pass-1 FIX `set(l)`〔S 就绪〕② Vector pass-1 MTE3 `set(l)`〔P 就绪〕③ Cube pass-2 FIX `set(l)`〔O 就绪〕 | ① Vector pass-1 MTE2 `wait(l)` ② Cube pass-2 MTE2 `wait(l)` ③ Vector pass-2 MTE2 `wait(l)` | 每 id 严格三次握手、per-id 交替（任一时刻 ≤1 未决事件） |
| FLAG_TASKDONE | L_max_bound | Vector→Cube | Vector epilogue 写出后 set（双 AIV 各 set 同一 id） | Cube 下一任务 Q-hoist 前 wait（`if task_id > 0`） | 单槽跨任务 WAR 保护 |

**varlen flag 复用安全论证**：不同任务的 `loop_count` 运行期不同，但 **Cube/Vector 两侧用同一 task 解码 + 同一 cu_seqlens 读得同一 `loop_count`/同一 l 序列**（两侧读同址同值，确定性）；前一任务对 id l 的三次握手在 TASKDONE 排空前已全部闭合。**id 复用的充分条件**：单任务内 `loop_count ≤ L_max_bound`（解析上界，V5），跨任务由 TASKDONE 排空——与定长版同论证。

**跨任务 id 复用安全**：同定长版（单槽论证）。

**2-AIV 聚合语义（R-1 风险项）**：Vector 段在每核两个 AIV（subid 0/1）上各执行一份，二者写各自半区的 P 后 set 同一 flag id——Cube 单次 wait 的聚合语义依定长版母本（22/22 + 生产）成立；若 Stage 3 探针证伪，回退 = FLAG_TASKDONE 按 subid 拆双 id + Cube 双 wait（id 总数 L_max_bound+2 ≤ 16 仍合规）。

**跨核同步**：无（任务独立、q/k/v 只读共享、output/lse 切片不相交）；本仓同步原语均为**核内**语义（T.set_flag.md §1 / T.sync_block_set.md §1 适用范围条款亲自核对）。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```
（防编译器重排手动两相位流水与显式 workspace——定长版先例；CUDA 的 FAST_MATH/compile_flags 不迁移，§0.5。）

### 7.4 sync_block_set 运行期 shim（CG-2026-0023，逐语句沿用定长版）

本容器（CANN 9.0.0）NPUIR codegen 为 `T.sync_block_set` 发射的属性名为 `syn_instr_mode`，而 `bishengir-compile` 解析器注册名为 `sync_instr_mode` → **任何含 `T.sync_block_set` 的 kernel 解析期失败**。**沿用定长版的运行期 shim**（`_install_sync_block_attr_shim()`：包装 `tilelang.jit.jit_npu.compiler_npu._npuir_to_bin_enable_npu_compile`，在调用外部 `bishengir-compile` 前把 `self.mlir_content` 的 `syn_instr_mode` 改写为 `sync_instr_mode`；不改磁盘工具链）——**Stage 5 集成须确认 shim 随 kernel 模块被导入**。另沿用定长版的**编译重试**（`SIGSEGV` / `NPU IR opt failed` 类外部工具链偶发崩溃重试；真实 lower 错误立即 re-raise）。

### 7.5 CV 融合设计（MixCV）说明

同一 kernel 内 `T.Scope("Cube")`（两相位 GEMM 大循环）与 `T.Scope("Vector")`（掩码/softmax/累加/epilogue）顺序书写、硬件上并行执行于同一 AI Core 的双引擎，经 per-带内块 flag 协同——MixCV 结构。**编程模式为 Expert（P-V1 探针 + CG-2026-0010 证据化结论，§2）**。tilelang-mixcv-skill 的核心关注点（sync_block_set/wait、Scope 分工、PIPE_FIX/MTE2/MTE3）在本设计 §0.6/§7.2 全量落地。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（**优先移植源仓测试 `ref_program`**，补 lse），**不得复刻 §0.6 的两相位 NPU 算法**——保证验证独立性。

```python
def golden_gqa_sw_fwd_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                             is_causal, window_size_left, window_size_right):
    """Variable-length sliding-window GQA forward reference (packed NHD).

    q:    [total_q, H,   D]    fp16/bf16
    k/v:  [total_k, H_kv, D]   fp16/bf16
    cu_seqlens_q/k: [B+1] int32
    Returns (output: q shape/dtype, lse: [H, total_q] fp32).

    Independence: no online softmax, no band trimming, no e-domain reorder, no
    arithmetic-penalty mask -- per-sample materialized softmax in fp32 with the
    source's exact four-condition mask (offset = seqlen_k - seqlen_q, FA3
    bottom-right), P quantized at the source acc_s_cast point.
    """
    groups = q.shape[1] // k.shape[1]
    scale = q.shape[2] ** -0.5
    outs = []
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"), dtype=torch.float32)
    for i in range(len(cu_seqlens_q) - 1):
        q0, q1 = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        k0, k1 = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        sq, sk = q1 - q0, k1 - k0
        offset = sk - sq
        qi = q[q0:q1].float().transpose(0, 1)                                   # [H,sq,D]
        ki = k[k0:k1].float().repeat_interleave(groups, dim=1).transpose(0, 1)  # [H,sk,D]
        vi = v[k0:k1].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        sc = (qi @ ki.transpose(-2, -1)) * scale                                # [H,sq,sk]
        qp = torch.arange(sq).view(sq, 1); kp = torch.arange(sk).view(1, sk)
        mask = torch.zeros(sq, sk, dtype=torch.bool)
        if is_causal: mask |= kp > qp + offset
        if window_size_left >= 0: mask |= kp < qp + offset - window_size_left
        if window_size_right >= 0: mask |= kp > qp + offset + window_size_right
        sc = sc.masked_fill(mask.unsqueeze(0), float("-inf"))
        m = sc.max(-1, keepdim=True).values.clamp(min=-1e38)
        p = torch.exp(sc - m)
        lse[:, q0:q1] = (torch.log(p.sum(-1)) + m.squeeze(-1))                  # natural-log
        p = p / p.sum(-1, keepdim=True)
        pq = p.to(q.dtype).float()                                             # P quantize point
        outs.append((pq @ vi).transpose(0, 1).to(q.dtype))                     # [sq,H,D]
    return torch.cat(outs, dim=0), lse
```

（**golden 独立性核对**：无 online softmax、无带状裁剪、无 e 域换底、无 f16 S/O 物化、无算术惩罚掩码——与 §0.6 的 NPU 算法结构正交；与源仓测试 `ref_program` 语义一致，仅补 lse。Op 契约只返回 output。）

### 8.2 精度标准与 L0 门槛测试计划

**精度门**（承接定长版谱系 D6/D8 校准的双门 + tier-3 分类；**Op 测试门为 `atol=1e-2, rtol=1e-2`**——源仓 `test_gqa_sliding_window_varlen_fwd_op` 冻结值）：

| dtype | 输出 atol/rtol（tier-1） | tier-2（2-ulp 回退） | lse atol/rtol |
|-------|--------------------------|---------------------|----------------|
| float16 | 5e-3 / 1e-5（Op 门 1e-2/1e-2 更宽） | +2·2^-10 相对余量 | 1e-3 / 1e-3 |
| bfloat16 | 同上 | +2·2^-7 相对余量 | 同上 |

tier-3 噪声分类（源 P 量化在早 causal 行的放大域：率 ≤ 0.01%、包络 2·flip_rel·v_absmax、lse 门不受扰）——`verify_equiv.py` E2E-bf16-golden 实测 tier-1 率 5.8e-4、tier-2 率 0（属该分类域，非缺陷）。

**分层测试策略**（回归入口 = `python _gqa_sw_fwd_varlen_kernel.py --level all`；L0/L1 阻塞、L2/Boundary 告警不阻塞）：

| 层 | 阻塞 | 用例 | 覆盖目标 |
|---|---|---|---|
| L0 | ✅ | L0-1 smoke fp16（packed 可变 + offset=0）；L0-2 smoke bf16（**两相位 bf16 首验**）；L0-3 **KV-cache offset>0 fp16**（本算子最高新颖度：offset 平移掩码）；L0-4 **混合长度 ragged fp16** + 契约校验 | 门禁最小集 + offset 主 trace + ragged |
| L1 | ✅ | manifest 全 sweep：smoke + 8b/70b short（wl=256）/long（wl=1024）× fp16/bf16 | 目标域全量 |
| L2 | ⚠️ | 源仓测试域：KV-cache（q<kv offset>0）× 四态掩码；对称带 wl=64,wr=64 非 causal；GQA h8/hkv8、h16/hkv1；wl=0（仅自注意力）；batch 3 混合长度 [128,256,384]；右窗 wr=64 非 causal | 四态掩码 + 运行期长度路径全覆 |
| Boundary | ⚠️ | 极短样本（q_len=1）；单 token 样本；`kv_len` 非 16 倍数（尾带）；zeros（lse 精确 = log2(band_len)）；batch 含 q_len=0 样本（cu 相等）；max_seqlen_q > 实际最大值（ghost 块） | 边界 + 病态 |

**L0 门槛计划**（Stage 3 首验顺序）：① **L0-1（fp16 smoke offset=0）→ 验证两相位主 trace + ragged 寻址（P-V1 同形）**；② **L0-3（fp16 KV-cache offset>0）→ 验证 offset 平移掩码 + 运行期 kv_len/offset（本算子最高新颖度）**；③ L0-4（混合长度 ragged）→ 验证 ghost 块 + 常开尾带；④ L0-2（bf16 smoke）→ 两相位 bf16 探针；⑤ 契约校验。任一失败按 §9.2 风险表的回退路径处置后重验。

---

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论汇总）

1. **一维 Kernel**（三维 grid 禁止）→ V1 persistent + ragged 任务解码。
2. **L0C=128KB / L1=512KB / UB=192KB per AIV**（`CONST-capacity-910B2C`）→ §4.3/§4.5 预算内设计。
3. **L0C 出数 dst 仅 GM**（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1）→ S/O 跨引擎必经 GM ws（V5，CG-2026-0005）。
4. **`store_fixpipe` f32→f16 为正确数值转换**（PL-1.9 probe 4.7e-4 = 1 ulp）→ M8'/M9' 采纳依据。
5. **活跃源 `T.transpose` epilogue 毒化 2.6×**（VP-D6）→ transpose-free lse（V7）。
6. **`load_nd2nz` 对定头跨步切片静默平铺误读**（CG-2026-0004）→ 全部 GM→L1 用 slice-form `T.copy`（packed NHD 同约束，V6）。
7. **向量算子发射开销 ~0.5µs/op**（`CONST-vector-launch-overhead`，910B2C 绑定）→ bn=256 宽块主 trace。
8. **flag event_id ∈ 0–15**（T.set_flag.md §2.1）→ V5 守卫公式 + 带内块 id 严格交替。
9. **parser 不折叠 if / 条件 alloc 非法 / vbrc 标量须 let-bound / v-op 标量 operand 须 kernel 派生 float PrimExpr / `T.reduce` 无 clear 为静默错误**（TRAP-tvm-parser-rules / TRAP-expert-v-operands / T.reduce.md §2.3）→ §3.2/§3.3 全项落实。
10. **`sync_block_wait` 阻塞整条发射流**（PL-1.9-hardlimits）→ 两相位结构本身就是该约束的规避。
11. **`vcmp` int16 全形态标量化**（PL-1.11）→ 掩码用算术惩罚式（P-V2 零 warning）。
12. **fractal band `0×NaN=NaN` 整列污染**（PL-1.11 P8）→ 常开尾带用 per-row 列 OOB + `l1_b`(V) `zbuf` 零初始化（V4）。
13. **stale UB 尾列 NaN**（TRAP-npuir-stale-ub-tail-columns-nan）→ N 载体每核零初始化 + 全宽列 OOB（V4）。
14. **Developer auto CV-split 不支持 persistent+gemm+v-prefix**（CG-2026-0010 / PL-1.16）→ Expert 双 Scope（§2，**P-V1 复现**）。
15. **`T.sync_block_set` 在本容器不可编译**（CG-2026-0023 / TRAP-bishengir-sync-instr-mode-attr）→ 运行期 shim（§7.4）。
16. **`T.floordiv/ceildiv` 负分子截断**（CG-2026-0024 / TRAP-npuir-negative-floordiv-divsi）→ 块分类谓词用**无除法直接整数比较**（VM-bnd）。
17. **v-op 标量 operand 拒绝裸 `tir.Cast` / `T.vcmp` 阈值须 let-bound 或 kernel 派生 PrimExpr**（TRAP-expert-v-operands）→ 运行期 offset/thr 走 `kid·0.0 + (…expr)` 顶层 Add 形态（**P-V2 实测**）；**`vsub` 标量恒作 src2**（buffer 作 src1，`K_lo−diff` 以 `−(diff−K_lo)` 实现）。

### 9.2 风险清单与常见错误

| # | 风险 | 等级 | 触发场景 | 缓解 / 回退 |
|---|------|------|----------|------------|
| R-1 | 2-AIV flag 聚合语义（双 AIV set 同 id、Cube 单 wait） | 中 | 跨任务槽位复用时 TASKDONE 单 wait 可能提前放行 | 定长版母本双生产；L0-1 含多任务复现；回退 = TASKDONE 按 subid 拆双 id + Cube 双 wait（L_max_bound+2 ≤ 16 ✓） |
| R-2 | 两相位 bf16 端到端（bf16 L1 gemm + f32 ws_s + bf16 ws_p）无本工具链先例 | 中 | L0-2 bf16 smoke | 保守承载（f32 ws_s/ws_o）；回退 = ws 全 f32 + 装载点 vcast |
| R-3 | **offset 平移掩码公式正确性（K_up/K_lo 符号 + offset 方向 + per-row 列 OOB 阈值）** | **高** | 任何 offset>0 或 ragged 用例 | 代数重推 + **双端点代入纪律**（`K_up=(bx·bm+row0+offset)−n_lo`，`thr=(tn_real−1)−i`）；**P-V2 探针 7/7 bit-exact 实测（含 offset=0/32/64/96）+ verify_equiv.py 11/11**；L0-3 + L2 四态掩码全覆盖；回退 = 单块全掩码（不裁剪，正确但慢） |
| R-4 | **运行期 cu_seqlens 标量读路径**（slice 起点/extent/循环界/输出地址） | **中** | 首编译/运行 | **P-V1 实测 maxdiff=0.0**；备选（copy cu→UB 标量索引）**实测 BLOCKED**——若 Stage 3 遇运行期读异常，回退 = host 侧闭包预读 cu_seqlens 成 device 标量张量按任务索引（仍需 device 读，收效存疑）；见 §9.3 |
| R-5 | UB 预算贴近 192KB（varlen 新增掩码缓冲 rowi/thr） | 中 | bn_eff 上调 / dim>128 | §4.5 口径**原位复用**（thr 复用 pen 后、t1/t2/t3 串行复用）；溢出时 UB diet（ub_pen 与 ub_f32_N 原位复用 −32KB）或 bn_eff 降 128 |
| R-6 | S/O f16 物化精度 | 低 | L1 fp16 全 sweep | PL-1.9 + v11nt 实证（tier-1 0 flips）+ §8.2 双门；超限时 ws_s/ws_o 升 f32 |
| R-7 | flag 预算超限（L_max_bound + 1 > 16） | 低 | wl 很大 + bn_eff 过小 | V5 守卫公式（工厂期强制 `assert`）；超限时 bn_eff += 16 重算 |
| R-8 | lse `[H,total_q,1]` 视图 + host reshape 契约 | 低 | 契约校验 | 定长版同款（wrapped reshape 零拷贝）；register_fake 形状一致（`(heads, total_q)`） |
| R-9 | 常开尾带机制遗漏域（varlen 与定长版的关键差异） | 中 | smoke（kv_len=128%256≠0）+ 测试域 | §6.4 数值例全覆盖；N 载体零初始化 + per-row 列 OOB + zbuf（**P-V2 7/7 NaN-free**） |
| R-10 | 目标域性能预期的不确定常数（910B3 上的向量发射率 / fabric 有效带宽未实测） | 中 | Stage 4 首采 | §1.6.0「设计期估算」按保守/乐观双口径；Stage 4 Phase 1 首采后校准；结构性下界（两相位 vs 单遍 2.31×）不受影响 |
| R-11 | 掩码式选择（算术惩罚 vs vselect）的判定裕度依赖未实证常数 | 低 | Stage 4 | §1.6.3 实验裁决计划（主选算术惩罚 + 备选 vselect + 阈值 0.95）；P-V2 已标定主选链设备 7/7 bit-exact 可运行（零 warning） |
| R-12 | smoke 欠载（32 任务 / 20 核，2.0× 不均衡） | 低 | smoke case | 核心部分忙、两相位单任务流水吸收；Stage 4 按 shape 分派 bm=32 候选 |
| R-13 | **ghost 块（real_m ≤ 0）**处理错误（误写/误算负 extent） | 中 | max_seqlen_q > 样本实际 q_len（测试域混合长度） | `loop_count=0` + `rm ≤ 0` 不写 + `tmc` 钳位 ≥16（垃圾行不入 Vector）；L0-4 + Boundary 覆盖 |

### 9.3 特殊场景处理

- **非整除分块 / ragged（常开）**：§6.4 尾块全套（tmc/tnc 钳位 + 尾带 per-row 列 OOB + N 载体零初始化 + zbuf + 行截断）。
- **ghost 块**：`real_m ≤ 0 ⇒ loop_count=0 + 不写`；packed 输出只含真实 token，无语义损失。
- **KV-cache（offset>0）**：因果窗右下对齐；`k_start` 可大（起于样本尾部附近）；band 长度仍受 V5 上界约束。
- **极短 shape / 空样本**（q_len=1 / q_len=0）：`loop_count` 可能 1 或 0；空样本（cu 相等 ⇒ q_len=0）无任务（bx 从 0 起 `real_m ≤ 0`）——正确性同构。
- **混合精度**：fp16/bf16 双 trace 同构；ws dtype 分 dtype 承载。
- **GQA**：`kv_head = by // groups`；heads_kv < heads 时 k/v 切片跨头共享只读。
- **四态掩码**：causal+left / causal only / left only（+right）/ none——trace 家族分派，各 trace 的段按需剔除。**`is_causal ∧ wr≥0` 组合不承诺**（源 kernel 在该分支忽略 right window，见 §0.1 角点；manifest/测试域无此组合）。
- **wl=0**（仅自注意力）：band = 单列，`loop_count` 可能 1，两段式分界覆盖整块（正确的保守退化）。

### 9.4 编译能力缺口登记（AGENTS.md「Compiler Capability Gap Rule」）

本任务探针 P-V1 复现 **CG-2026-0010** 同缺口（`layer: codegen / BishengIR`；capability：Developer auto CV-split 对 persistent + `T.gemm` + v-prefix 混排不支持）。**本容器失败层为编译期**（BishengIR `std::out_of_range` / `map::at`），与既有条目的运行期崩溃层不同——属该能力缺口在新工具链（tilelang `0.1.2+76e4550…` / CANN 9.0.0 / 910B3）下的复现，建议在 `.agents/evolution/capability-gaps.md` CG-2026-0010 条目下 `occurrences += 1`（证据 = `history_version/design_probe_varlen_structure.py` variant A/B/C 对照）。另：**「copy cu_seqlens→UB 再标量索引」路径不可用**（`InternalError: cannot find variable cub`）——若后续同族任务再现，可作 CG-2026-0005（前端 API 能力）或新条目的候选；本任务以核内 GM 标量读绕过（P-V1 实证可用）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_query_attention_sliding_window_varlen/_gqa_sw_fwd_varlen_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # §1.6.1 采纳项等价性机器验证（torch CPU，11/11 EQUIV_PASS）
├── _gqa_sw_fwd_varlen_kernel.py       # 算子实现 + 分层测试驱动（Stage 3 生成）
├── RETROSPECTIVE.md                   # 任务复盘（追加）
└── history_version/
    ├── design_probe_varlen_structure.py  # P-V1（模式复现 + ragged 映射，§2/§0.6 V1 证据）
    └── design_probe_varlen_mask.py       # P-V2（offset 平移掩码 + 常开尾带，§0.6 V2/V4 证据）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件） | 设计文档 |
| `verify_equiv.py` | ✅ 已完成（§1.6.1 结果表联动，11/11 EQUIV_PASS，exit 0） | 数学等价优化采纳项机器验证 |
| `history_version/design_probe_varlen_structure.py` | ✅ 已完成（P-V1：Developer BLOCKED / Expert PASS；ragged 直接 GM 标量读 PASS maxdiff=0.0） | 编程模式 + ragged 映射证据 |
| `history_version/design_probe_varlen_mask.py` | ✅ 已完成（P-V2：offset 平移双侧惩罚 + per-row 列 OOB + 常开尾带 7/7 bit-exact，零 warning） | 掩码 + 尾带证据 |
| `_gqa_sw_fwd_varlen_kernel.py` | ⬜ 待实现（Stage 3） | 两相位 kernel + golden + L0/L1/L2/Boundary 驱动（`--level all`） |
| TileOPs 侧 wrapper | 不改动 | `grouped_query_attention_sliding_window_varlen.py` 契约零变化（V8 闭包兼容） |

### 10.3 命名规范

- 项目目录: `grouped_query_attention_sliding_window_varlen`；算子目录: `_gqa_sw_fwd_varlen_kernel`；实现文件: `_gqa_sw_fwd_varlen_kernel.py`——与 conductor 指定目录一致。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 探针（P-V1/P-V2）+ `verify_equiv.py`
2. ⬜ Golden 函数 + 精度门（§8.1/§8.2，源仓 `ref_program` 移植 + lse）
3. ⬜ 工厂层（trace 家族分派 + bn_eff/L_max_bound 守卫 + ws 闭包 + sync shim）+ `_builder_2phase(mask 域 = causal+left)` 主 trace（fp16）
4. ⬜ **L0-1 首验**（fp16 smoke offset=0）→ **L0-3**（fp16 KV-cache offset>0）→ L0-4（混合长度 ragged）→ L0-2（bf16）→ 契约校验
5. ⬜ 其余 mask 域 trace + 常开尾带变体 + L2/Boundary 全套
6. ⬜ `--level all` 全绿 → Stage 2 检视入口

---

## 11. 性能预期与度量计划（varlen manifest 域 · Stage 4 度量锚点）

> 本节为 Stage 4 调优提供预期锚点与度量计划（best_effort，无硬性数值目标）；预期基于结构证据外推（定长版同族实测 + §1.6.0 设计期估算），全部待实测校准。

**结构证据锚点**（定长版同族同机同口径 msprof op Task Duration）：
- 两相位 vs 单遍链（fa 域）：**2.31×**（结构性下界，不依赖目标域常数）。
- 运行期带状裁剪执行块数红利：8b-long **15360 块 vs 全 causal 18432 块（−16.7%）**；FLOPs 103.2 vs 137.5 GFLOP（−24.9%）。
- 定长版同结构 8b-long（H·B=64）实测 **1279.54µs**（`CASE-gqa-sliding-window-migration`）——本算子 H·B=128（2× 任务）。

**预期区间**（保守 = 定长版实测锚点 × 任务数比；乐观 = §1.6.0 D-2 发射/流量双口径外推——910B3 常数未实测，R-10）：

| workload | 执行块数 E | 全 causal E | FLOPs | 保守预期（定长版锚点外推） | 乐观预期（§1.6.0 D-2） |
|---|---|---|---|---|---|
| smoke（wl=−1） | 32 | 32 | 0.034 GFLOP | ~0.1ms（欠载） | ~0.04–0.1ms |
| 8b-short（wl=256） | 1536 | 1536 | 6.47 GFLOP | ~2.5ms（短 KV 摊销不足） | ~1.0–1.6ms |
| 8b-long（wl=1024） | **15360** | 18432 | 103.2 GFLOP | **~2.56ms**（1279.54µs ×(128/64)） | **~2.0–3.3ms**（§1.6.0 D-2 双口径） |
| 70b-short（wl=256） | 3072 | 3072 | 12.93 GFLOP | ~3.5ms | ~1.5–2.5ms |
| 70b-long（wl=1024） | **30720** | 36864 | 206.4 GFLOP | **~5.1ms**（1279.54µs ×(256/64)） | ~4.0–6.5ms |

**Stage 4 度量计划**：① Phase 1 首采 5 case fp16（msprof op Task Duration，launch-count=20/warm-up=5/median，captured op = 新 kernel 的 mix_aic 名）；② 对照上表校准 910B3 的向量发射率与 fabric 有效常数；③ **实验裁决**（§1.6.3）：代表 shape 实测主选（算术惩罚式）vs 备选（vselect），判定阈值 = 备选 < 主选 × 0.95 才翻转；④ 调优候选优先序：bm ∈ {64, 80} × bn_eff ∈ {128, 256} 扫描（UB/L0C 公式约束内）→ f16 softmax 链（#11 defer 项）→ 双槽 id-offset 变体（R-7）→ shape 分派（smoke bm=32）；⑤ 回归纪律：`--level all` 全绿为每轮前置。

---

## 附：声明

- 本设计为 **增量迁移设计（first_design）**——骨架 = 同族定长版 `examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/DESIGN.md`，凡与 varlen 无关的结论**沿用并标注「沿用定长版」**；设计精力集中在 varlen 新增轴（§0.6 V1–V8）。
- 迁移决策（§0）驱动全部后续章节：§1–§7 基于 §0.6 重设计后的 NPU 算法展开，未照抄源方案（源单遍 per-block 链 → 两相位；三维 grid → 一维 persistent + ragged 解码；fragment 驻留 → UB + ws 物化；源掩码预填 → offset 平移算术惩罚 + per-row 列 OOB）。
- 源算法全部已识别优化手段（§0.4 OP1–OP16）在 §0.5 逐项处置、在 §1.6.0 逐项承接，无静默丢弃。
- 算法调研（Phase R）先于算法级优化（Phase 2）：§1.6.0 选定算法 = §1.4 算法描述 = §1.6.1–§1.6.3 分析对象；源算法仅作为基线候选之一参与 R1–R4 对比。

### 沿用定长版清单 vs varlen 新增清单（检视速查）

**沿用定长版（逐语句/逐项）**：一维 persistent 20 核分核方案（§5.5）；双 Scope 两相位结构（V5）；ws 三族 + scales[] 延迟回放 + 双 AIV 自适应行半分 + Q-hoist（V5）；e 域 vexp 链（M1'）；lse log2 域（M2'）；减 max 稳定化（M3'）；arithmetic-penalty 掩码机制本体（M5' 的 `clamp+vmax+vmul(−1e38)+vadd`）；单调 max + clamp（M6'）；f16 S/O 物化（M8'/M9'）；P 量化点（M10'）；Q-hoist（M12'）；`T.reduce clear=True`（M13'）；fp16/bf16 双 trace + ws dtype 承载；size-form gemm + 分形钳位；L1/L0C 生命周期复用；slice-form copy（CG-2026-0004）；transpose-free lse（V7 机制）；sync_block_set 运行期 shim + 编译重试（§7.4）；pass_configs 双关闭（§7.3）；精度双门 + tier-3 分类（§8.2）；编程模式 = Expert（§2，P-V1 复现）；`T.gemm` Expert 签名；`T.vcast/T.reduce/T.varlog2/vbrc` API 选型。

**varlen 新增/改写（8 轴）**：**V1** ragged 任务映射（bx 样本内局部 + 运行期 cu_seqlens 标量读 + ghost 空转）；**V2** offset 平移的双侧掩码 + **per-row 列 OOB**（绝对列条件必含行号）；**V3** 运行期带状范围（kv_len/offset）；**V4** 常开 ragged 尾带（运行期 `kv_len%bn ≠ 0`，N 载体零初始化 + 全宽列 OOB + zbuf）；**V5** 运行期 flag 预算守卫（解析上界 `L_max_bound`，非编译期扫描）；**V6** packed NHD 运行期寻址（Q/K/V 装载 + output 写出，地址含 q_start/kv_start）；**V7** lse `(H,total_q)` 布局 + ghost 行截断；**V8** 接口（cu_seqlens + total_q/total_k + 闭包 + lse reshape）。

### 设备探针实测结论汇总（可复算）

| 探针 | 脚本 | 变体 / 配置 | 结论 |
|------|------|-------------|------|
| **P-V1** | `history_version/design_probe_varlen_structure.py` | A Developer persist+gemm（无 vec） | **PASS** |
| | | B Developer persist+gemm+v-prefix | **BLOCKED**（BishengIR `std::out_of_range`/`map::at`） |
| | | C Expert 双 Scope 同结构 | **PASS** |
| | | V1 ragged：直接 GM 标量读 `cu[bz]` + 动态 slice/extent/循环界/输出地址（含 ghost 块） | **PASS（maxdiff=0.0）** |
| | | V2 ragged：copy cu→UB 再标量索引 | **BLOCKED**（`InternalError: cannot find variable cub`） |
| **P-V2** | `history_version/design_probe_varlen_mask.py` | 主选 offset 平移双侧算术惩罚 + per-row 列 OOB + 常开尾带（运行期 offset∈{0,32,64,96} + 运行期 tn_real + 部分宽度 copy + 零初始化） | **7/7 bit-exact，sentinel 逐位，NaN-free，零标量化 warning** |
| **verify_equiv** | `verify_equiv.py` | 9 数值采纳项 + 4 E2E（KV-cache offset 32/64/96，fp16/bf16） | **11/11 EQUIV_PASS，exit 0** |
