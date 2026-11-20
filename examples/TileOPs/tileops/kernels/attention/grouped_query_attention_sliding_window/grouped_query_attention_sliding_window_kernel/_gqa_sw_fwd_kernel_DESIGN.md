# `_gqa_sw_fwd_kernel` 算子设计文档（GroupedQueryAttentionSlidingWindowFwdOp · GPU→NPU 迁移 · Expert 模式 · 两相位 S/P 物化 + 滑动窗口带状掩码）

> **迁移任务**（scenario=migration, migration_mode=harness）。源算子：`/mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_fwd.py` 的架构通用工厂 `_gqa_sw_fwd_kernel`（→ `GQASlidingWindowFwdKernel`）。同文件 Hopper 专属变体 `_gqa_sw_fwd_wgmma_pipelined_kernel` 按 add-npu-op K5「NPU 折叠架构特化」排除，本设计不迁移。
>
> 本文件依据 `tilelang-op-design` skill 的 **Phase M0（源算子三问解读）→ Phase R（算法调研四问）→ Phase M1（硬件耦合性判定与 NPU 算法重设计）→ Phase 1–8** 生成。首选定范式 = 同型 NPU 先例 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（GQA/MHA prefill 前向，同样 BSHD + flash online softmax，已成功迁移为 Expert 两相位）。
>
> 工具链（设计期探针实测戳）：`source /mnt/workspace/Agentkj/env_npuir.sh`；tilelang `0.1.2+cf6f999389f4f3481da5778aa06d73478684ce2f`（dev root `/mnt/workspace/Agentkj/build`）；Ascend **910B3**（`NPUUtils.get().get_aicore_num()` 实查 **20** AIC / 40 AIV）；CANN 9.0.0；2026-10-16。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**（BSHD 定长 GQA + 滑动窗口 flash 前向）：

```
o   = softmax(scale · q @ k^T + M) @ v           # BSHD 布局，online softmax
M   = 掩蔽矩阵                                    # 下述三条件的「不满足」并集（逐元素 −inf / 0）
valid(q_pos, k_pos) ⟺ (¬is_causal ∨ k_pos ≤ q_pos)          # 因果上界
                     ∧ (wl < 0      ∨ k_pos ≥ q_pos − wl)    # 左窗下界
                     ∧ (wr < 0      ∨ k_pos ≤ q_pos + wr)    # 右窗上界
scale   = dim^{−1/2}
is_causal 默认 true；window_size_left/right 默认 −1（−1 = 无限制）
```

> **角点说明（`is_causal ∧ wr ≥ 0`，源行为与通用式不一致）**：上式「三条件并集」是语义层的**通用**掩码定义；但**源 kernel** 的 `is_causal and has_window` 分支（源码 L100–112）在 causal 下**只施加 causal|left、忽略 right**（且 `k_end` 不含 `wr`，L76–77）。即源在 `is_causal=True ∧ wr≥0` 时的实际有效区间为 `[q_pos−wl, q_pos]`（右窗不生效），与本通用式的 `[q_pos−wl, q_pos+wr]` 不同。**manifest 与测试域均无该组合**（全部 causal 用例 `wr=−1`），故目标域不受影响；本设计按其实际行为分派 trace（§3.3 trace 家族不含 causal+right 变体），并在 §9.3 记明该组合不承诺。

**I/O 契约**（源码 L43–52 与 wrapper 亲自核对）：

| 项 | 内容 |
|----|------|
| q | `(B, S, H, D)`，float16 / bfloat16，BSHD 连续 |
| k / v | `(B, S, H_kv, D)`，`same_as(q)`，`H % H_kv == 0`（GQA） |
| output | `(B, S, H, D)`，`same_as(q)`（`out_idx=[3,4]` 的第 3 号输出） |
| lse | `(B, H, S)`，**float32**（第 4 号输出；**Op wrapper `GroupedQueryAttentionSlidingWindowFwdOp.forward` 只返回 `output`，丢弃 lse**——源 `gqa.py` L1640 `output, _ = self.kernel.forward(q, k, v)`） |
| 中间 dtype | `accum_dtype = "float"`（fp32）——S、online 统计量在 fp32 域；`acc_s_cast` 量化回**输入 dtype** 后进第二个 GEMM |

**语义要素表**：

| 要素 | 结论 | 依据 |
|------|------|------|
| 数学语义 | 上三条件带掩码的 softmax attention（non-causal 时为对称带 `[q_pos−wl, q_pos+wr]`；causal 时 `wl=−1,wr=−1` 退化为下三角 causal） | 源码 L100–123 四掩码路径 + `gqa.py` Op docstring |
| I/O 契约 | 见上表；输出 shape/dtype 与 q 同 | 源码 L43–52；`migration_meta.json` `source_output_shape=(B,S,H,D)` |
| 规约语义 | softmax 沿 **kv 轴**（尾轴方向跨 S）规约；online 累加**按 kv 块顺序**推进（`k_start..k_end`），块内 `reduce_max/sum(dim=1)` | 源码 L93–161 |
| dtype 语义 | S/统计量 fp32；**P 量化到输入 dtype**（`T.copy(acc_s, acc_s_cast)`）；`acc_o` 全程 fp32；lse fp32 | 源码 L61–68, L150 |
| 边界语义 | `−T.infinity` 掩码 + scores_max 有限下限 clamp（`−1e38`），防 `exp2(−inf−(−inf))=NaN`；`loop_count = max(k_end−k_start, 0)` 允许 0（整块出窗）；`lse = log2(logsum) + scores_max·scale` | 源码 L73, L136–146, L159–160 |
| host 侧语义 | 无预处理/后处理（q/k/v 直接进 kernel；`contiguous()` 为契约保证） | 源码 L168–179 |

**完成标准核对**：§8.1 golden 可直接由本节语义写出（与源仓测试 `GroupedQueryAttentionSlidingWindowFwdTest.ref_program` 等价：expand KV heads → 三元掩码 → softmax → @v → 转回 BSHD）。

### 0.2 源算子输入输出

见 §0.1 表。**额外发现（重要，影响 §0.6 接口）**：源算子同时产出 `output` 与 `lse`（`out_idx=[3,4]`），而 Op wrapper 丢弃 lse；NPU 侧 kernel 仍须产出 lse（保持 kernel 契约与 wrapper 兼容），但 lse 不进入 Op 精度门（Op 只校验 output）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（源 `@T.prim_func _gqa_sw_fwd_main`，L46–163）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|---|------|------|------|------|----------|
| S1 | grid 划分 | `T.Kernel(ceildiv(S,bm), H, B, threads=)` 三维 | — | (bx,by,bz) | 任务 = (q-block, head, batch) |
| S2 | 状态初始化 | `clear(acc_o)`、`clear(logsum)`、`fill(scores_max, −inf)` | — | acc_o, logsum, scores_max | online 初值 |
| S3 | Q 装载 | `T.copy(q[bz, bx·bm:(bx+1)·bm, by, :], q_shared)` | GM q | SMEM `[bm,dim]` | — |
| S4 | KV 循环范围 | causal→`k_end=ceildiv(min(S,(bx+1)bm),bn)`；`wr≥0`→`+wr`；否则 `k_end=ceildiv(S,bn)`；`wl≥0`→`k_start=max(0,bx·bm−wl)//bn` 否则 0；`loop_count=max(k_end−k_start,0)` | 参数 | 循环边界 | 带状裁剪（**滑动窗口的算法核心**） |
| S5 | mask 预填 | `T.Parallel(bm,bn)` 逐元素 `if_then_else(掩码, −inf, 0)` 写入 `acc_s` | (bx,bn,bm,wl,wr) | acc_s | M |
| S6 | QK^T | `T.gemm(q_shared, k_shared, acc_s, transpose_B=True, FullRow)`（`initC=False` 累加到 S5 的 mask 预填上） | q,k tile | acc_s | `q@k^T + M` |
| S7 | online max | `scores_max_prev←scores_max`；`fill(scores_max,−inf)`；`reduce_max(acc_s, dim=1, clear=False)`；`max(·,prev)`；clamp `≥−1e38` | acc_s | scores_max | 数值稳定化 |
| S8 | α 缩放因子 | `scores_scale = exp2(m_prev·scale − m·scale)` | 行向量 | scores_scale | rescale |
| S9 | P = exp2 与行和 | `acc_s = exp2(acc_s·scale − m·scale)`；`reduce_sum(acc_s, dim=1)` | acc_s | acc_s, scores_sum | softmax 分子/分母 |
| S10 | logsum 递推 | `logsum = logsum·scores_scale + scores_sum` | — | logsum | online 递推 |
| S11 | P 量化 + rescale acc_o | `T.copy(acc_s, acc_s_cast)`（fp32→输入 dtype）；`acc_o[i,j]·=scores_scale[i]` | acc_s | acc_s_cast, acc_o | 量化点 + rescale |
| S12 | PV | `T.gemm(acc_s_cast, v_shared, acc_o, FullRow)` | P tile, v tile | acc_o | `P@v` |
| S13 | epilogue | `acc_o[i,j] /= logsum[i]`；`T.copy(acc_o, output[...])`；`logsum = log2(logsum) + scores_max·scale`；`T.copy(logsum, lse[...])` | — | output, lse | 归一化 + lse |

**数据流图（源 GPU 视角）**：

```
GM[q] ──T.copy──> SMEM[q_shared (bm,dim)] ───────────────────────────┐
GM[k] ──T.copy──> SMEM[k_shared (bn,dim)] ──T.gemm(transpose_B)──> frag[acc_s (bm,bn) fp32]
                                                                    │  (+ S5 mask 预填)
   frag[acc_s] ──reduce_max─vmax─clamp─exp2(·scale−m·scale)─reduce_sum ─> rows[bm] (m, α, ℓ)
   frag[acc_s] ──T.copy(cast)──> frag[acc_s_cast (bm,bn) 输入 dtype] ─┐
GM[v] ──T.copy──> SMEM[v_shared (bn,dim)] ──T.gemm──> frag[acc_o (bm,dim) fp32] ◄┘ (×α rescale)
   frag[acc_o] ──除以 ℓ──> GM[output]；rows ──log2+·──> GM[lse]
```

**循环与并行结构**：三维 grid（`ceildiv(S,bm)` × H × B）× `threads=128` 线程；KV 循环 `T.Pipelined(loop_count, num_stages)`；元素并行由 `T.Parallel` 与 fragment 上的 mma 承载。

**host 侧逻辑**：`functools.lru_cache(32)` 工厂缓存、`heads % heads_kv` 校验、`make_log2e_scale(dim)` 常量折叠、`super().__init__()`/`init_config`（config = block_m/block_n/num_stages/threads）。

### 0.4 优化手段解读（为什么快）

**逐项识别清单**（识别以源码特征 + 语义目的 + 依赖的源硬件特性为准）：

| # | 优化手段 | 源码特征 | 目的 | 依赖源硬件特性 | 初判 |
|---|----------|----------|------|----------------|------|
| OP1 | **flash online softmax** | S7–S10 的 running max/sum + α rescale | 消 O(S²) 中间量，O(bm·S) 缓冲 | **无（纯算法层）** | 可移植 |
| OP2 | **滑动窗口带状裁剪（loop range）** | S4 `k_start`/`k_end` + `loop_count` | 只处理窗内 KV 块，省 FLOPs/访存 | 无（纯算法层） | 可移植（**本算子算法核心**） |
| OP3 | SMEM tiling | `alloc_shared` + 分块 `T.copy` | 减 GM 重复访问 | shared memory | 硬件强相关 |
| OP4 | coalesced 访存 | 尾轴 `:dim` 连续 | 带宽 | memory coalescing | 硬件强相关 |
| OP5 | Tensor Core mma | `T.gemm`（fragment dst） | 矩阵算力 | tensor core | 硬件强相关（NPU 有 Cube 等价） |
| OP6 | `T.Pipelined` 多级软件流水 | KV 循环 `num_stages` | 装载/计算重叠 | cp.async | 硬件强相关（NPU 有 MTE 等价） |
| OP7 | fragment 寄存器累加 + 融合 epilogue | `alloc_fragment`，epilogue 原地 | 数据驻留、消中间回写 | 大寄存器堆 | 硬件强相关（NPU L0C/UB 等价但形状不同） |
| OP8 | warp 归约 | `reduce_max/sum(dim=1)` on fragment | 低延迟行归约 | warp shuffle | 硬件强相关（NPU 行归约等价） |
| OP9 | `GemmWarpPolicy.FullRow` | gemm 参数 | warp 划分 | CUDA warp | 无 NPU 对应 |
| OP10 | three-dim grid + threads | `T.Kernel(m,H,B,threads=)` | CTA 级并行 | GPU grid/block | NPU 一维 Kernel |
| OP11 | exp2 + LOG2E 预乘折叠 | `make_log2e_scale` | 用 exp2 硬件指令，少一次换底 | ffma 融合 | 换底等价可移植 |
| OP12 | mask 预填 + initC 累加融合 | S5+S6 | 省独立 mask pass | tensor core C 累加 | 需重判 |
| OP13 | `TL_ENABLE_FAST_MATH` + `compile_flags=[-O3,-DENABLE_BF16]` | `@tilelang.jit` | CUDA 编译链优化 | CUDA 编译链 | NPU 无此 pass |
| OP14 | `lru_cache(32)` 工厂缓存 | `functools.lru_cache` | 复用编译产物 | 无 | 可移植（host） |
| OP15 | bf16 直连 tensor core | `-DENABLE_BF16` | bf16 算力/带宽 | bf16 mma | NPU Cube bf16 等价 |
---

### 0.5 硬件耦合性分析与 NPU 适配决策（Expert 模式口径）

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？**本轮判定输入含 §1.6.0 调研结论**：源算法（单遍 per-block online-softmax 链）与调研候选（两相位 S/P 物化等）同台比较——调研选定两相位结构后，§0.4 各优化手段的处置对照该结论逐项重判（承接规则见 §1.6.0 调研结论段）。**与前序 MHA 迁移的关键差异**：本算子多出 **OP2 滑动窗口带状裁剪**（算法核心）与**双侧带状掩码**（左窗 + 因果），且 KV 循环范围随 bx 变化为 `[k_start, k_end)` 而非 `[0, NK)`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（公式 / 三条件掩码 / P 量化点 / dtype / lse log2 域 / 输出 shape） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| **OP2 滑动窗口带状裁剪（loop range）** | 算法 | 无 | — | **保留** | 纯算法层（§5.4 规则 2）——`k_start = max(0, s_lo−wl)//bn`（`wl≥0`）、`k_end = ceildiv(min(S,(bx+1)bm),bn)`（causal）/`ceildiv(S,bn)`（非 causal），承载为本设计的 KV 循环范围 `[k_start,k_end)`（§0.6 E-band） |
| OP1 online softmax 递推 | 算法 | 无 | — | **保留** | 纯算法层；两相位下递推在 Vector pass-1 逐块保持，rescale 因子存 UB `scales[]` 延迟到 pass-2 回放（E2） |
| S7 scores_max 有限下限 clamp | 算法 | 无 | — | **保留** | `T.vmax(m, −1e38, m)`（源 NaN guard 逐字保留，M6） |
| q 驻留复用 + k/v 流式分块 | 算法 | 无 | — | **保留** | 承载走 E1（persistent 任务解码）/E2（两相位 pass 循环）/E-bnd（带状范围）/E7（尾块钳位） |
| OP3 SMEM tiling | 优化 | CUDA SMEM | 有（L1 512KB / UB 192KB） | **等价替换** | Cube 侧 `T.alloc_L1`（l1_a/l1_b 生命周期复用，§4.3）；§5.3 映射表「shared memory → L1/UB」 |
| OP5 Tensor Core GEMM | 优化 | NVIDIA tensor core | 有（Cube `T.gemm` + L0A/L0B/L0C） | **等价替换** | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tmc, dim, tnc])`（QK^T）/ `T.gemm(l1_a,l1_b,l0_c,initC=True,size=[tmc,tnc,dim])`（PV）（T.gemm.md §2.1；flash_attn_npuir.py L102–151 同构） |
| OP6 `T.Pipelined` 软件流水 | 优化 | cp.async | **意图由两相位结构承接** | **重新设计** | → E2：装载/计算重叠不再依赖 per-block 乒乓——Cube pass-1 无等待直跑（MTE2/MTE1/C/FIX 四通道天然重叠）、pass-2 逐块等 P；`T.Pipelined` 不再使用（PL-1.9-twophase） |
| OP7 fragment 寄存器累加 + 融合 epilogue | 优化 | CUDA 寄存器堆 | **无直接等价**（L0C 不能被 v 算子操作；L0C 出数 dst 仅 GM——docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1「src 必须来自 L0C / dst 必须来自 GM」） | **重新设计** | → E2：S 与每块 PV 部分和经 GM workspace 跨引擎往返（API 强制，CG-2026-0005）；acc_o 驻留 Vector UB，pass-2 延迟 rescale 累加 |
| OP8 warp 归约 + `T.Parallel(bm)` 标量 max | 优化 | warp shuffle | 有（Vector 行归约 + vmax） | **等价替换** | `T.reduce(ub, dst, dims=[1], reduce_mode="max"/"sum", clear=True)`（docs/Tilelang.language/规约操作/T.reduce.md §2.2.2；**clear 必须显式 True**——§2.3 静默错误条款，M14）+ `T.vmax` |
| 行级 rescale / logsum 更新 | 优化 | warp 标量并行 | 有（v 算子行广播） | **等价替换** | `T.vmul(ub, ub_rowvec, ub)`（[M,N]×[M,1] 行广播，docs/Tilelang.language/数学操作/T.vmul.md §2.2.2）+ `scales[]` 切片存取 |
| **OP12 mask 预填 + initC 累加融合** | 算法 | tensor core C 累加 | 有（initC），**但两相位 gemm1 恒 initC=True 逐块覆写（无 mask 预填位）**；且 developer 实测 `T.Parallel` 谓词预填是 aiv_scalar 主瓶颈 | **重新设计** | → E-mask：掩码移至 **Vector pass-1、S 到达后 softmax 前**，**全向量算术惩罚式**（PL-1.11 实测：vcmp 全形态标量化）——双侧下界/上界各一条 `clamp(diff−K,0,1)` 链 + `vmax` 合成 + `S += pen·(−1e38)`；两段式带状分界使全有效块零掩码开销（E-bnd） |
| OP9 `GemmWarpPolicy.FullRow` | 优化 | CUDA warp | 无（NPU 无 warp） | **舍弃** | Cube 矩阵并行由分形（M/N/K 切分进 L0A/L0B/L0C）承担，`b_transpose=True` 即完整语义 |
| OP10 three-dim grid + threads | 优化（结构） | GPU grid/block | **无**（`T.Kernel` 一维约束） | **重新设计** | → E1：一维 persistent `T.Kernel(20, is_npu=True) as (kernel_id, subid)` + 分核策略（§5.5）；`threads` 参数删除（K3/K9，wrapper 已无该参数） |
| OP13 FAST_MATH + compile_flags | 优化 | CUDA 编译链 | 无 | **舍弃** | npuir 分支无此 pass；改设 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION=False` + `NPUIR_ENABLE_AUTO_MULTI_BUFFER=False`（防编译器重排手动两相位流，§7.3） |
| OP11 exp2 + LOG2E 折叠 | 优化 | CUDA ffma | 有等价（`T.vexp` e 底） | **等价替换（换底方式变更）** | → E4：e 域 `T.vexp` 直接实现（数学恒等，M1）；lse 保持 log2 域（`T.vlog2`，M2） |
| OP15 bf16 直连 tensor core | 优化（dtype 承载） | GPU bf16 mma | 有（T.gemm.md §2.2.1 bf16 √；§2.3 bf16×bf16 dst fp32 实测可用；PL-1.8 Expert 端到端实测） | **等价替换** | Cube bf16 直连（slice-form `T.copy` bf16 进 L1 + `T.gemm` bf16×bf16→fp32）；`P` 量化 `T.vcast(f32→bf16,"rint")`；**ws 承载 fp16: f16/f16/f16、bf16: f32/bf16/f32**（E5） |
| OP4 coalesced 访问 | 优化 | memory coalescing | 等价（尾轴连续 `T.copy`） | **等价替换** | 全部 GM↔L1/UB 搬运用 slice-form `T.copy`（D1 约束：BSHD 定头切片必须 slice 形态，CG-2026-0004） |

**统计**（每行恰计一次，合计 = 表体 18 行）：**保留 5 项**（计算语义〔语义〕、OP2 带状裁剪〔算法·本算子核心〕、OP1 online 递推〔算法〕、S7 clamp〔算法〕、q 驻留/kv 流式分块〔算法〕）/ **等价替换 7 项**（OP3、OP5、OP8、行级 rescale、OP11、OP15、OP4）/ **重新设计 4 项**（OP6、OP7、OP12、OP10）/ **舍弃 2 项**（OP9 FullRow、OP13 FAST_MATH+compile_flags）。5+7+4+2 = 18 ✓。（`threads` 参数并入 OP10 行处理〔移除〕，不单独计行；`disable_tma` 源算子未使用，不列。）

### 0.6 NPU 算法重设计（Expert 模式）

> **总体结构（§1.6.0 调研选定）**：**一维 persistent Kernel（20 核）× 双 Scope（Cube/Vector）× 两相位独立大循环 × 滑动窗口带状 KV 范围**——结构原型 = 同型 NPU 先例 `_gqa_prefill_fwd_kernel.py`（MHA Expert 两相位），叠加本算子特有的**带状裁剪 + 双侧掩码**。每个逻辑任务 = (q-block, query-head, batch)；任务内：**Cube pass-1** 串行计算整条带上的全部 S 块（QK^T，Q-hoist 一次装载）物化到 per-core GM workspace；**Vector pass-1** 逐块消费 S（vcast → 双侧算术惩罚掩码 → prescale → online max/α/ℓ 递推 → P 量化）回写 workspace 并把 per-block rescale 因子存 UB `scales[]`；**Cube pass-2** 逐块消费 P 计算 PV 部分和（initC=True 逐块覆写）物化；**Vector pass-2** 逐块延迟 rescale 累加 acc_o；epilogue 归一化 + transpose-free lse 写出。跨引擎协同 = per-带内块下标 flag（S/P/O 三次握手复用同一 id，严格 per-id 交替）+ 任务边界 FLAG_TASKDONE。

#### 与 MHA 先例（`_gqa_prefill_fwd_kernel.py`）的异同 / 复用点 ⭐

| 维度 | MHA 先例 | 本算子 | 处置 |
|------|----------|--------|------|
| 语义 | `softmax(scale·q@k^T + causal/oob mask)@v`，S_q==S_kv，GQA | 同左 + **双侧滑动窗口掩码** + **带状裁剪** | 语义扩展 |
| 整体结构 | 一维 persistent 20/24 核 + 双 Scope + 两相位独立大循环 + S/P/O GM ws + per-n-block flag | **完全复用** | ✅ 直接复用 |
| Cube pass-1/pass-2 | `for i in T.serial(NK)`，n_lo = i·bn，`NK = ceildiv(min((bx+1)bm+off, S), bn)` | `for l in T.serial(loop_count)`，**`n_lo = k_start·bn + l·bn`**，`loop_count = k_end − k_start` | ✅ 复用 + 带状范围 |
| flag 协议 | id = n-block 下标 i ∈ [0, NK_max)，三次握手复用 + FLAG_TASKDONE=NK_max | **id = 带内块局部下标 l ∈ [0, L_max)**（两 pass 与两 Vector pass 迭代同一 `[k_start,k_end)`） | ✅ 复用，id 语义局部化 |
| 掩码 | 单侧因果 `vcmp(diff, K_blk, "le")` + `vselect`；两段式 K_A（floordiv）分界 | **双侧**（左窗下界 + 因果上界）**算术惩罚式**（PL-1.11：避免 vcmp 标量化）；**两段式带状分界**（`L_lo_full`/`L_hi_full`）——区间中段全有效块零掩码 | 🔁 **本算子的最大重设计点**（E-mask/E-bnd） |
| online softmax | e 域 vexp + scales 延迟回放 + ℓ clear=True | **完全复用** | ✅ |
| fp16/bf16 路径 | fp16: f16/f16/f16；bf16: f32/bf16/f32 | **完全复用** | ✅ |
| 尾块 | size-form gemm + 分形钳位 + 尾带 OOB 掩码 + 行截断 | **完全复用**（manifest 域零触发） | ✅ |
| lse | transpose-free [B,H,S,1] 视图 | **完全复用** | ✅ |
| 物理核数 | 24（910B2C） | **20（910B3，实查）** | 参数替换 |

**复用点一句话**：除「带状 KV 范围 + 双侧算术惩罚掩码 + 两段式带状分界」三处为本算子新增（滑动窗口特有）外，**其余结构逐语句复用 MHA 先例**——该先例已 29/29 用例通过 + 11 轮 Stage 4 调优（fa 域 2.31× vs 单遍链），是低风险高置信的载体。

**重设计项 E1: 三维 grid + threads → 一维 persistent 双 Scope Kernel + 分核策略**

- **源方案**：`T.Kernel(ceildiv(S,bm), H, B, threads=threads)` 三维 grid；每 (q-block, head, batch) 一个 CTA。
- **NPU 新算法**：一维 `T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid)`，**NUM_KERNELS = 20 = `NPUUtils.get().get_aicore_num()` 实查值**（查询代码与记录见 §5.5）。逻辑任务数 `num_logical = ceildiv(S,bm) × H × B`；任务解码 `cid = task_id·20 + kernel_id → (bx = cid // HB, by = (cid // B) % H, bz = cid % B)`，`HB = H·B`；核内 persistent 串行 `num_local_tasks = T.ceildiv(num_logical − kernel_id, 20)`。同 MHA 先例的 ghost-task clamp：`cid = T.min(task_id·20 + kernel_id, num_logical − 1)`（D3 实证：`T.ceildiv` 下界为 1、idle 核会被 BishengIR 重规范化跑越界任务——clamp 把 ghost 任务钉到最后一个合法任务）。
- **语义保持论证**：(q-block, head, batch) 逻辑任务完全独立——读写不相交的 `output[bz, s_lo:s_lo+tail_m, by, :]` / `lse[bz, by, s_lo:s_lo+tail_m]`，q/k/v 只读共享（GQA 下多 query head 读同一 kv head 切片）。任务到核映射与核内串行顺序不改变任何数值——**数学等价**；clamp 导致的重复任务写 bit-identical 输出（同输入同指令序）。

**重设计项 E-band: 滑动窗口带状 KV 范围（本算子算法核心）**

- **源方案**：`k_start = max(0, bx·bm − wl)//bn`（`wl≥0`，否则 0）；`k_end = ceildiv(min(S,(bx+1)bm),bn)`（causal）/ `ceildiv(min(S,(bx+1)bm+wr),bn)`（`wr≥0` 且非 causal）/ `ceildiv(S,bn)`（否则）；`loop_count = max(k_end−k_start, 0)`。
- **NPU 新算法**：完全承接源式的 `k_start`/`k_end`（逐字移植，per-task PrimExpr），KV 循环改为 `for l in T.serial(loop_count): n = k_start + l`；带内块数 `loop_count` 上界 `L_max = max over tasks`。**flag id 预算守卫**：工厂期令 `bn_eff = max(bn_caller, ceil16(ceildiv(bm + eff_left + eff_right, 14)))`（`eff_left = wl if wl≥0 else S`，`eff_right = wr if wr≥0 else 0`），使 `L_max + 1 ≤ 16`（`T.set_flag.md §2.1` event_id 0–15）；trace-time `assert L_max ≤ 15`（超限则 `bn_eff += 16` 重算）。
- **语义保持论证**：① **被裁剪块 = 恒等更新**——源在 `[0, k_start)` 的块全列被 `left_mask` 掩蔽、在 `[k_end, NK)` 的块全列被 causal/right 掩蔽；全掩蔽块的贡献为：`m_cur = −1e38`（clamp 后有限）、`α = e^{m_prev − max(m_prev,−1e38)} = 1`、`P = e^{−1e38 − m} = 精确 0`、`Δℓ = 0`、`Δo = 0`——即 online 递推的**恒等元**（`verify_equiv.py` M4 实证 `alpha(all −inf block)==1`）；逐块累加序不被裁剪改变（保序子序列）。② 源本身已裁剪（`loop_count`），本设计与源**同界同序**——不是新增近似。③ 带外块的 S 甚至不计算（省 FLOPs），不产生任何被消费的中间量。

**重设计项 E2: fragment 流水 + per-block 跨引擎链 → 两相位独立大循环 + GM workspace 物化 + per-带内块 flag**（承接 MHA 先例核心，逐语句复用）

- **源方案**：`T.Pipelined` KV 流水 + fragment 上 v 链（acc_s/acc_o 全程驻留寄存器，epilogue 融合）。**实测（MHA 前序）**：`sync_block_wait` 阻塞整条发射流（busy 核 aic_scalar 38–47%），单遍链被两相位结构以 2.31× 淘汰（PL-1.9-twophase）。
- **NPU 新算法（两相位，同 MHA 先例）**：每核独立单槽 GM workspace 三族（尺寸 §4.3）：
  - `ws_s [20, bm, pad_band]`（fp16: f16 / bf16: f32）：Cube→Vector 的 QK 分数块（`T.copy(l0_c[0:tmc,0:tnc] → ws_s[kid,0:tmc,l·bn:(l+1)·bn])`，L0C f32→f16 隐式转换，PL-1.9 probe max_rel 4.7e-4 = 1 f16 ulp）；
  - `ws_p [20, bm, pad_band]` 输入 dtype：Vector→Cube 的 P 块；
  - `ws_o [20, bm, dim·L_max]`（fp16: f16 / bf16: f32）：Cube→Vector 的每块 PV 部分和。
  - `pad_band = L_max · bn_eff`。
- **Cube 流**（`T.Scope("Cube")` 内）：任务头 [task_id>0: 等 FLAG_TASKDONE] → Q-hoist → pass-1 `for l in T.serial(loop_count)`：K 装载 + `T.gemm(initC=True,b_transpose=True,size=[tmc,dim,tnc])` + `T.copy(l0_c→ws_s)` + `set(l)`；pass-2 `for l in T.serial(loop_count)`：`wait(l)` + P/V 装载 + `T.gemm(initC=True,size=[tmc,tnc,dim])` + `T.copy(l0_c→ws_o)` + `set(l)`。
- **Vector 流**（`T.Scope("Vector")` 内，per-AIV 自适应行半分）：任务头 vbrc 状态初值 → pass-1 `for l`：`wait(l)` + S 装载 + vcast + **算术惩罚掩码（E-mask）** + prescale + reduce max/单调查断 + α + reduce sum + P 量化 + `set(l)` + ℓ 递推；pass-2 `for l`：`wait(l)` + O 装载 + rescale 累加；epilogue 归一化 + lse + `set(FLAG_TASKDONE)`。
- **语义保持论证**：① 数据流等价——S/P/O 块内容与源 S6/S11/S12 逐块对应；`o = o·α_l + O_partial(l)` 与源 `acc_o = acc_o·α + P@v` 代数同构（仅累加分组变化，fp32 ulp 级，§8.2 容差覆盖）；② 归约语义——块间保序（flag 强保序），ℓ/o 更新次序与源一致；③ 恒等元——pass-1 零等待直跑，pass-2 的 P-wait 在稳态已就绪（MHA 先例实测该重叠正是 2.31× 加速来源）。

**重设计项 E-mask: mask 预填 + initC 累加 → Vector pass-1 双侧算术惩罚掩码（滑动窗口特有）** ⭐

- **源方案**：`T.Parallel(bm,bn)` 逐元素谓词写 `acc_s[i,j]=if_then_else(掩码,−inf,0)` 后 `T.gemm(initC=False)` 累加。MHA 迁移实测该形态为 aiv_scalar 主瓶颈。
- **NPU 新算法**：掩码移至 **Vector pass-1、S vcast 到 f32 之后、softmax 之前**，**全向量、NaN 无关、无 vcmp**（PL-1.11 第五轮实测：`vcmp` int16 全形态标量化；本容器探针 P2 复现同一 warning `Op 'hivm.hir.vcmp' will execute by scalar instruction with low efficieny`）。**算术惩罚式**（PL-1.11 已验证形态）：

  ```
  diff  = colmat − rowmat            # [half,bn] f32，arange 一次性物化（j−i，与任务无关，每核构建一次）
  # 因果/右窗上界：valid ⇔ diff ≤ K_up  ⇔ K_up − diff ≥ 0 → 惩罚 clamp(diff − K_up, 0, 1)
  pu = vmin(vmax(vsub(diff, K_up), 0), 1)
  # 左窗下界：valid ⇔ diff ≥ K_lo → 惩罚 clamp(K_lo − diff, 0, 1)
  pl = vmin(vmax(vsub(K_lo, diff), 0), 1)
  pen = vmax(pu, pl)                 # diff 为整数，pen ∈ {0,1} 精确
  S   = vadd(S, vmul(pen, −1e38))    # 掩蔽位精确 −1e38
  ```

  其中（per-AIV，`row0 = subid·real_m0`，`n = k_start + l`，全局块下标）：
  `K_up = (s_lo + row0) − n·bn`（causal）；`K_up = (s_lo + row0 + wr) − n·bn`（`wr≥0` 且非 causal）；`K_lo = (s_lo + row0 − wl) − n·bn`（`wl≥0`）。
- **语义保持论证**：① **作用点相同**（QK 之后、softmax 之前）；② **值无关**——pen 由 (i,j,n) 决定，与 S 值无关；掩蔽位 `S' = S + (−1e38)`：`|S| ≲ 60·dim^{−1/2}` 远小于 1e38，f32 大数吸收 → 精确 `−1e38`，`P = e^{−1e38 − m}`（m ≥ −1e38 经 M6 clamp 有限）→ **精确 +0.0**，与源 `e^{−inf} = 0` 同效（`verify_equiv.py` M5/M6 实证「masked bit-exact +0.0=True」）；③ **NaN 免疫边界**——加法掩码无法清除 NaN（PL-1.11 P8）：S 来自 L0C gemm 恒有限，掩码后 `−1e38 + 有限` 有限；唯一 NaN 源为**尾带残留**（ragged `S_kv`，fractal band）——该域按 PL-1.11 **回退 `vselect`（P 侧值替换）+ `l1_b`(V) 每核一次 `zbuf` 零初始化**（根治 gemm2 K 维 fractal band 的 `P带(0)×V残留(NaN)=NaN` 整列污染；`[bn_eff,dim]` 零张量内部参数，成本 ~0.06%），manifest 域零触发（§6.4）；④ 掩码序与源一致（mask 先于 softmax；`prescale` 在掩码后、`sm_scale=0` 病态路径与源逐位一致）。
- **探针实测佐证（P2，rev-1 修复并重跑，`history_version/design_probe_band_mask.py`）**：**主选「双侧算术惩罚链」**（`vsub/vmax/vmin/vmax` + `vmul(pen,−1e38)` + `vadd`）在设备上 **7/7 配置 bit-exact PASS**（有效掩码 + `S'=S+pen·(−1e38)` 哨兵逐位校验），**不产生任何标量化 warning**；被否决的 `vcmp/vand/vselect` 备选链亦 7/7 bit-exact 编译运行，但复现两条 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency` warning → 据此采纳算术惩罚式为 band-free trace 主选。（探针阈值已按 rev-1 修为 let-bound / kernel 派生 PrimExpr。）
**重设计项 E-bnd: causal 单侧 K_A 两段式 → 双侧带状两段式分界（全有效中段零掩码）**

- **源方案（MHA）**：单侧因果下界 `K_A = min(floordiv(s_lo+off+1, bn_eff), S//bn_eff)`——块 `i < K_A` 全有效零掩码，仅对角块走掩码链。
- **NPU 新算法（双侧）**：带状的有效区间中段 = 全有效块。块 `n = k_start + l` 全有效 ⟺ 对 tile 内全部 (i,j)：`diff_min = −(tm−1) ≥ K_lo` 且 `diff_max = bn−1 ≤ K_up`：
  - 下界满足（全行）：`n·bn ≥ s_lo + row0 + (tm−1) − wl` → **`l ≥ L_lo_full`**，`L_lo_full = ceildiv(s_lo + row0 + tm − 1 − wl, bn) − k_start`；
  - 上界满足（全行）：`n·bn ≤ s_lo + row0 − (bn−1)` → **`l < L_hi_full`**，`L_hi_full = floordiv(s_lo + row0 − (bn−1), bn) − k_start + 1`。

  分两段顺序 `if`（规避 TRAP-tvm-parser-rules 复合条件风险，且各对应一条边界的惩罚）：
  ```
  if l < L_lo_full:   # 左侧边界块：施加左窗下界惩罚（pl）
      S = vadd(S, vmul(vmin(vmax(vsub(K_lo, diff),0),1), −1e38))
  if l >= L_hi_full:  # 右侧边界块：施加因果/右窗上界惩罚（pu）
      S = vadd(S, vmul(vmin(vmax(vsub(diff, K_up),0),1), −1e38))
  ```
  `wl<0` 时 `L_lo_full = 0`（左段 trace-time 剔除）；`is_causal=False ∧ wr<0` 时无上界（右段剔除）；`wl≥0 ∧ ¬causal ∧ wr≥0` 时两段齐备（对称带）。**per-AIV 阈值**：`row0 = subid·real_m0` 进 `L_lo_full`/`L_hi_full`（两 AIV 各自判定，可能分叉——Vector 段本就分行域执行，合法）。
- **语义保持论证**：全有效块跳过惩罚 ⟺ 源掩码条件在全块恒真（充要：`diff_min ≥ K_lo ∧ diff_max ≤ K_up` 是全称量化的逐行展开）；中段块零掩码与施加惩罚（pen ≡ 0 → `+0`）**逐位一致**（f32 `x + 0.0 = x`）。`verify_equiv.py` M4 覆盖恒等元，P2 探针覆盖双侧有效掩码（算术惩罚链 + vcmp 备选链各 7/7 设备 bit-exact，`history_version/design_probe_band_mask.py`）。

**重设计项 E4: exp2 + LOG2E 折叠 → e 域 vexp 链（换底等价替换）**

- **源方案**：`scale = make_log2e_scale(dim) = dim^{−1/2}·LOG2E`（LOG2E 截断 1.44269504），`exp2(x·scale − m·scale)`。
- **NPU 新算法**：e 域直接实现（`T.vexp`）：① prescale `T.vmul(ub_f32_N, prescale, ub_f32_N)`（`prescale = dim^{−1/2}`，trace-time 常量门控跳过 `prescale==1`）；② reduce max → 单调查断 + clamp；③ `α = vexp(vsub(m_prev, m))`；④ `P = vexp(vsub(S', m))`；⑤ reduce sum；⑥ ℓ/o 更新；⑦ `lse = vlog2(ℓ) + m·LOG2E`（全精度 1.4426950408889634）。
- **语义保持论证**：**数学恒等** `exp2(u·log2e) = e^u`（M1：最大 5 ulp，违反率 0）；lse 同理（M2：最大 1 ulp）。数值路径更短（少两次 [half,bn] 乘法舍入）。

**重设计项 E5: bf16 输入路径 → Cube bf16 直连 + f32 S/O 物化**

- **NPU 新算法**：bf16 trace 下 q/k/v **直接** slice-form `T.copy` bf16 进 L1；`T.gemm` bf16×bf16→fp32；P 量化 `T.vcast(f32→bf16,"rint")`；输出 `T.vcast(f32→bf16,"rint")`。**ws 承载**：`ws_s = f32`（S 值域相对误差 bf16 2^-8→经 exp 传播威胁容差；f16 路径 2^-11 已实证）、`ws_p = bf16`（源语义）、`ws_o = f32`（部分和保守承载）。fp16 与 bf16 路径**结构逐语句同构，仅 dtype 字符串与 ws dtype 随 trace 切换**。
- **语义保持论证**：输入精度与源一致；P 量化点与源逐位同 dtype；输出 cast 同 rint；NaN 传播一致；GEMM 累加分组差异 fp32 ulp 级。**风险（R-2）**：无本工具链 bf16 端到端先例 → L0-2 前置首验；回退 = ws 族全 f32 + 装载点 vcast 链。

**重设计项 E6: 接口契约适配（workspace 参数 + 闭包 wrapper + lse 视图 + config 语义）**

- 工厂外层签名 `_gqa_sw_fwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, window_size_left, window_size_right, dtype='float16')` **完全保留**（含 `lru_cache(32)`、参数校验）；
- 内层 callable 签名 `(block_m, block_n, num_stages)`（`threads` 移除，K3/K9——wrapper 已无 threads，§0.3 亲自核对）；
- 返回值 = **Python 闭包**：按 ws shape `torch.empty(..., device=q.device)` 分配 ws 族、调用编译 kernel、出口 `lse_t.reshape(B,H,S)` 还原契约形状；`(block_m,block_n,num_stages)(q,k,v) → (output, lse)` 调用形态逐字保持（wrapper custom_op `npub::gqa_sw_fwd_wrapped_kernel` 与 Kernel class 零改动）；
- prim_func 参数 `(q, k, v, ws_s, ws_p, ws_o, output, lse)`，`@tilelang.jit(out_idx=[-2,-1], target="npuir", pass_configs={…})`；
- **lse 以 `[B,H,S,1]` 视图声明**（同一连续内存），epilogue 自然 2D 区域拷贝——**零 transpose**（VP-D6：活跃源 transpose epilogue 毒化整 kernel 2.6×）；出口 reshape 还原 `[B,H,S]`；
- **config 语义（trace-time）**：`bn_eff = max(bn_caller, ceil16(ceildiv(bm + eff_left + eff_right, 14)))`（E-band 守卫）；caller 传 wrapper 默认 `(64, 64, 1)` / `(64, 32, 1)` 时替换为设计默认 `(bm=64, bn=256, num_stages=1)` 后再套守卫（S4-5 分派纪律：显式非默认 config 一律尊重）；`num_stages` 为保留旋钮（单槽两相位下不改变行为）。

**重设计项 E7: 对齐/非对齐双路径 + 谓词标量 GM 写 → size-form 尾钳位统一路径**

- **NPU 新算法**：`tail_m_real = min(bm, S−s_lo)`；`tmc = max(16, min(bm, ceil16(tail_m_real)))`（M/N 分形下限 16）；`tn_real = min(bn_eff, S−n_lo)`；`tnc = max(32, min(bn_eff, ceil16(tn_real)))`（K 分形下限 32）；size-form `T.gemm` + 同形切片 `T.copy`；尾带 OOB 掩码（`has_band = tnc > tn_real`，manifest 域不触发）：P 侧 `vselect` 值替换置 0 **且 `l1_b`(V) 每核一次 `zbuf` 零初始化**（PL-1.11：`P带(0)×V残留(NaN)=NaN` 须源头零化，成本 ~0.06%）；行截断写出（`rm` 自适应半分）。
- **语义保持论证**：size-form gemm 与尾钳位切片在真实行列上与源谓词装载一一对应；q 尾垃圾行被自适应半分（只覆盖真实行）隔离、不进入 Vector 域、不写出；尾带的 stale V 行**经 `zbuf` 零初始化保证**贡献恰为 0（非「任意值×0=0」假设——PL-1.11 实测该假设对 NaN 位不成立）。manifest 5 workload 全部整除（零尾块零尾带，§5.2），机制仅服务契约测试域。

### 0.7 标杆实现

- **源算子路径（golden 语义来源）**：`/mnt/workspace/TileOPs/tileops/kernels/attention/gqa_sliding_window_fwd.py`（`_gqa_sw_fwd_kernel` L19–165）+ `tileops/kernels/online_softmax.py`（`make_log2e_scale`）；本仓提取件 `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/_grouped_query_attention_sliding_window_fwd_kernels.py`。
- **源仓测试基准（golden 直接来源）**：`examples/TileOPs/tests/ops/test_grouped_query_attention_sliding_window.py` 的 `GroupedQueryAttentionSlidingWindowFwdTest.ref_program`（expand KV heads → 三元掩码 `q_pos<k_pos` / `k_pos<q_pos−wl` / `k_pos>q_pos+wr` → softmax → `@v` → 转回 BSHD；`atol=1e-2, rtol=1e-2`，不检查 lse）。
- **NPU 侧同类标杆**：
  - `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（**首选范式 / 直接母本**：Expert 两相位 + Q-hoist + scales 延迟回放 + transpose-free lse + fp16/bf16 双 trace；配套 DESIGN 151KB + RETROSPECTIVE + perf_opt）；
  - `examples/flash_attention/flash_attn_npuir.py`（两相位 S/P 物化结构原型，L70–265）；
  - `examples/flash_attention/flash_attn_npuir_dev.py`（Developer 形态对照）；
  - `examples/mixcv/mixcv_mixkernel.py`、`examples/mixcv/example_mixcv.py`（`T.rs` + `sync_block_set/wait` 最小形态）；
  - `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/`（迁移范式）。
- **pattern-library 实测证据**：`attention.md` PL-1.7/1.8/1.9/1.9-hardlimits/1.9-twophase/1.11/1.12/1.13/1.16；`traps-compiler.md` VP-D6（transpose 毒化）；`traps-runtime.md` TRAP-DEVMODE-PERSIST-GEMM；`constants.md`（容量/带宽/发射率常数）。
---

## 1. 概述

### 1.1 算子名称

`_gqa_sw_fwd_kernel`（GQA 滑动窗口前向，`GroupedQueryAttentionSlidingWindowFwdOp` 的 TileLang kernel；NPU 侧 Expert 模式两相位重实现，目标域 = manifest 4 workload〔Llama-3.1-8B/70B 短/长序列 × wl∈{256,1024}〕+ smoke〔wl=−1〕）。

### 1.2 功能描述

BSHD 布局 q/k/v 上的 flash-attention 风格 online-softmax 前向，带**滑动窗口带状掩码**：输出 `o = softmax(scale·q@k^T + M) @ v` 与每行 log2 域 log-sum-exp `lse`；支持 GQA 分组（`heads % heads_kv == 0`）、四态掩码（causal / 左窗 / 右窗 / 无）、`wl/wr = −1`（无限制）、fp16/bf16。**Op wrapper 只返回 output（丢弃 lse）**，kernel 仍产出 lse 以保持契约。

### 1.3 数学公式

同 §0.1 三式（valid 条件 / softmax / scale）。滑动窗口有效区间：`k_pos ∈ [q_pos − wl, q_pos]`（causal+左窗）；对称带 `[q_pos − wl, q_pos + wr]`（非 causal）。

### 1.4 算法描述（迁移决策后的 NPU 侧 Expert 算法）

**选定算法族**（§1.6.0 调研结论）：**两相位 S/P 物化 + per-带内块 online softmax + 滑动窗口带状裁剪**（MHA 先例结构族；源算法的单遍 per-block 链形态在 MHA 迁移中被实测淘汰，但 online-softmax 数学递推完整保留）。**NPU Expert 形态**（§0.6 E1–E7 落地）：

1. **任务划分**：一维 persistent **20 核**；逻辑任务 = (q-block × head × batch)；核内 `T.serial` 串行（E1）。
2. **带状 KV 范围**：任务内 KV 循环仅覆盖 `[k_start, k_end)`（源 S4 公式逐字移植），带内块数 `loop_count = k_end − k_start`（E-band）。
3. **双引擎两相位分工**：Cube 段承担全部 GEMM——pass-1 串行算完带内全部 S 块（QK^T，Q-hoist，无等待直跑），pass-2 串行算完带内全部 PV 部分和（initC=True 逐块覆写）；Vector 段 pass-1 逐块消费 S（vcast → **双侧算术惩罚掩码** → prescale → online max/α/ℓ 递推 → P 量化）+ 因子存 UB `scales[]`，pass-2 延迟 rescale 累加 acc_o，epilogue 归一化 + transpose-free lse（E2，双 AIV 自适应行半分）。
4. **跨引擎数据环**：`l0c_s →(T.copy FIX)→ ws_s →(MTE2)→ UB →(v 链)→ ws_p →(MTE2)→ l1_p →(gemm2)→ l0c_o →(FIX)→ ws_o →(MTE2)→ UB 累加`；每核单槽 ws 三族 + per-带内块下标 flag（三次握手复用）+ FLAG_TASKDONE（E2）。
5. **掩码**：Vector pass-1 双侧算术惩罚式（`clamp(diff−K,0,1)` + `vmax` + `S += pen·(−1e38)`，**无 vcmp**——避免 PL-1.11 标量化）；两段式带状分界（`L_lo_full`/`L_hi_full`）使全有效中段块零掩码开销（E-mask/E-bnd）。
6. **尾块**：size-form gemm + 尾钳位切片 copy + 分形下限钳位（M 16 / K 32）+ 尾带 OOB 掩码 + 行截断写出（E7）。
7. **dtype 路径**：fp16/bf16 **同构直连**——q/k/v slice-form copy 进 L1，`T.gemm` fp16×fp16 / bf16×bf16 → fp32；P 量化 vcast 到输入 dtype；ws 承载 fp16: f16/f16/f16、bf16: f32/bf16/f32（E5）。

**与源算法的结构差异**（来源标注）：① 承载结构从单遍 per-block 链换为两相位独立大循环（§1.6.0 调研结论 + §0.6 E2）；② acc_o 从 fragment 驻留改为 UB 驻留 + PV 部分和 GM 物化 + scales 延迟回放（E2）；③ 掩码从 Cube gemm 预填移至 Vector 双侧算术惩罚（E-mask）；④ exp2 域改 e 域（E4）；⑤ 三维 grid 改一维 persistent（E1）；⑥ 尾块双路径改 size-form 统一路径（E7）；⑦ KV 循环范围 `[0,NK)` → `[k_start,k_end)`（E-band，源式保留）。算法层（online softmax 递推、**带状裁剪**、P 量化点、lse 域、NaN guard）**逐项保留**。

### 1.5 数据流图

```
GM[q] ──T.copy(slice, Q-hoist)──> L1[l1_a [bm, block_share]] ──────────────────────────┐
GM[k] ──T.copy(slice)──> L1[l1_b [bn_eff, dim]]（pass-1 K / pass-2 V 生命周期复用）      │
   L1[l1_a]×L1[l1_b] ──T.gemm(b_transpose,initC=True,size=[tmc,dim,tnc])──> L0C[l0_c]
                                                            │ ──T.copy(FIX, f32→f16)──> GM[ws_s]
                                                            │  set flag l
GM[ws_s] ──MTE2──> UB[ub_f16_N] ──vcast──> UB[ub_f32_N] ──[双侧算术惩罚掩码]─[prescale]
   ──reduce(max)─vmax×2─vsub/vexp(α→scales[])─vsub/vexp(P)─reduce(sum)─vcast──> UB[ub_f16_N]
UB[ub_f16_N] ──MTE3──> GM[ws_p] ──MTE2──> L1[l1_a] ─┐ set flag l
   L1[l1_a]×L1[l1_b(V)] ──T.gemm(initC=True,size=[tmc,tnc,dim])──> L0C[l0_c] ──FIX(f32→f16)──> GM[ws_o]
                                                         │ set flag l
GM[ws_o] ──MTE2──> UB[ub_f16_D] ──vcast──> UB[ub_f32_D] ──vmul(α)+vadd──> UB[acc_o]（pass-2 逐块累加）
UB[acc_o]/ub_ell/ub_m ──vdiv/vlog2/vcast──> UB ──行截断切片──> GM[output (BSHD)] / GM[lse ([B,H,S,1] 视图)]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（§1.6.0，调研对象为同一数学语义的算法族，源算法只是基线候选之一）；再数学等价地优化公式（§1.6.1）；再把全部计算点向量化（§1.6.2）；最后决定向量化轴与核内布局（§1.6.3）。分析对象 = §1.6.0 选定、经 §0.6 E1–E7 重设计落地的 NPU Expert 两相位算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：`o = softmax(scale·q@k^T + M)@v`（+lse）的 sliding-window GQA 前向算法族，**目标域画像** = manifest 4 workload（Llama-3.1-8B/70B，H×B ∈ {16,32,64,128}，任务数 1024–2048，S ∈ {512,2048}，D ∈ {64,128}，wl ∈ {256,1024}）+ smoke（wl=−1）。**调研深度**：融合类（GEMM + softmax + GEMM + 带状掩码）→ 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。**信息源**：① skill 参考表 `algorithm-candidates.md` 命中行（`ALG-attention` / `ALG-softmax` / `ALG-elementwise`）；② pattern-library 全主题文件（`attention.md` PL-1.7/1.8/1.9/1.9-hardlimits/1.9-twophase/1.11/1.12/1.13/1.16；`traps-*.md`；`constants.md`；`layout.md`）；③ **本仓 `examples/` 全目录 attention 族检索**（`flash_attention/{flash_attn_npuir.py, flash_attn_npuir_dev.py}`、`deepseek_v4/example_sparse_attn_kernel_highperf.py`、`deepseek_v32/sparse_mla_fwd_exp.py`、`mixcv/`、`TileOPs/` 侧 MHA 两相位归档——**两遍式 S/P 物化先例纳入为第一候选**）；④ 源算法本身（§0.3/§0.4）；⑤ 外部已知算法（PyTorch/flash-attention 生态模型知识——仅取算法思路）。**网络检索**：未执行——本地信息源已充分覆盖本算子族全部主流结构（flash 单遍 / 两相位物化 / developer 形态均有本地先例或实测档案）；按 algorithm-research.md §6 第 6 层纪律如实记录。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源 flash online softmax + 带状裁剪**（单遍 per-block，exp2·LOG2E 域） | 分块 running max/sum + rescale，P 量化后进 GEMM2，fragment 驻留；`[k_start,k_end)` 裁剪 | — | — | ✅（基线） |
| 1 | **E1-E7 单遍 per-block 跨引擎链**（基线的 NPU 承载） | 每 KV 块 C1→FLAG_S→V1→FLAG_P→C2→FLAG_O→V2 握手链 | 数学等价（同族算法不同承载） | —（**已被 MHA 实测淘汰**：fa 域两相位 98.05µs vs 单遍调优终值 226.72µs，2.31×） | ✅（证据对照） |
| 2 | **两相位 S/P 物化**（MHA 先例结构族，本设计选定） | Cube pass-1 全 S 物化 → Vector pass-1 softmax + scales[] 延迟回放 → Cube pass-2 全 PV 部分和物化 → Vector pass-2 累加；per-块 flag | 数学等价（online 递推保留，α 回放序列与源逐块更新恒等） | Cube pass-1 零等待直跑；α/PV 解耦；实测 fa 域 2.31× | ✅ **（选定）** |
| 3 | e 域 vexp 换底 | `e^{x−m}` 直接计算（`e^u = 2^{u·log2e}`） | 数学恒等 | 少 2 次 [half,bn] 乘法舍入 | ✅（§1.6.1 M1 采纳，并入 #2） |
| 4 | **带状 KV 范围裁剪**（滑动窗口特有） | `[k_start, k_end)` 块级裁剪，带外块不计算 | **恒等精确**（全掩码块 = online 递推恒等元：α=1、Δℓ=0、Δo=0） | **执行块数 7680 vs 全 causal 9216（8b-long：−16.7%）**；FLOPs 51.6 vs 68.8 GFLOP | ✅（§1.6.1 M4 采纳，并入 #2；源已用） |
| 5 | 双侧掩码算术惩罚式（`clamp(diff−K,0,1)` + `S+=pen·(−1e38)`） | 掩码 = 有限大数哨兵加性惩罚 | **恒等精确**（pen∈{0,1}；掩蔽位 `S'=S−1e38→−1e38`，`e^{−1e38−m}=+0.0`） | 消 vcmp/vselect 标量化（PL-1.11 实测 vcmp 全形态标量化；本容器 P2 复现 warning） | ✅（§1.6.1 M5 采纳） |
| 6 | **KV-split / flash-decoding**（KV 维切分多核并行 + 跨核合并） | 每 split 独立 online 状态，合并 `ℓ=Σℓ_k e^{m_k−m}`、`o=Σo_k e^{m_k−m}` | 数学等价（log-sum-exp 合并律） | 小 Sq 欠载时并行度↑ | ✅（R3/R4 评估后 defer） |
| 7 | 批量重基准 G（G 块共享批末 max，O 回传降频） | batch 内延迟 m 收敛，O 每 G 块回传一次 | 数学等价（m 为真 running max，P≤1 安全） | O 跨引擎流量 ÷G | ✅（R3/R4 评估后 defer——scales[] 回放已解除 α/PV 耦合） |
| 8 | S/P/O 跨引擎 f16 传输 | S、P、O_partial 以 f16 物化 | **容差内等价**（S 域 2^-11 → ΔP/P ~0.26%；O 部分和 f16 实测 tier-1 0 flips） | 三族 ws 字节减半 | ✅（§1.6.1 M8/M9 采纳，并入 #2） |
| 9 | scale 折入 q（任务头一次 [bm,dim] 缩放替代每块 prescale） | `(s·q)·k ≡ s·(q·k)` | **容差内等价但精度降级**（源在 fp32 域乘 scale；f16 q 折入引入 2^-11 级 S 误差） | 每 task 省 NK−1 次 [half,bn] vmul | ✅（精度论证后否决，§1.6.1 M11） |
| 10 | lse 域公式 `log2(ℓ) + m·LOG2E` | `log2(ℓ·e^m)` 恒等 | 数学恒等 | 免大数指数化 | ✅（§1.6.1 M2 采纳） |
| 11 | f16 softmax 链（[half,bn] 大 pass 全 f16 算术） | vexp/vsub/vmul/reduce f16 + f16 running max | 容差内等价（v9 实测 −3.5%~−14.1%；f16 ℓ 求和间距是残余 lse 误差主导项 ~1e-3） | 大 pass 字节减半 + 消 2 个 vcast | ✅（**defer Stage 4**：f32 链 + f16 传输已达标） |
| 12 | 除法转乘倒数（o/ℓ → o·(1/ℓ)） | 行广播除一次 vs 倒数+乘 | 数学恒等 | 无收益（每任务 1 次 vdiv） | ❌（op 数相同） |
| 13 | **band 内逐列裁剪**（boundary 块内全掩蔽列的跳过） | 逐列细粒度跳过 | 恒等精确 | 边界块内 ±bn 列浪费 | ❌（列粒度 < 向量宽度，无对应 API；块粒度裁剪已由 #4 承担） |
| 14 | Winograd / 变换域 attention | 不适用（非卷积结构；attention 的 GEMM K=dim 小；参考表无此族行） | — | — | ❌（结构不匹配） |

**R2 在线算法**：**有**——直接证据：源算法本身即 online softmax（§0.4 OP1：分块增量 max/sum/logsum + rescale，单遍 KV）；**选定结构 #2 同样保留 online 递推**（Vector pass-1 逐块 running max/ℓ，per-block α 因子存档回放——「在线」性质在 softmax 层完整保留；物化的是跨引擎传输的 S/P/O tile，非全量 score 矩形）。收益口径：KV 扫描 1 遍（vs 两遍 materialize-softmax 的 2 遍）；中间缓冲 O(bm × band)（per-core 单槽 ws）vs O(S_q × S_kv)。滑动窗口下带可能短到单块（`loop_count=1`）：此时 online 递推退化为直接 softmax（α_0 值无关），但 trace 层统一处理（不做退化特化——causal/window 域 loop_count 随 bx 变化，收益微小，§1.6.1 M7/f13）。结构判据复核：softmax 分母与 PV 分子均可分解为 running 统计量 → 在线变体存在且源已采用 ✓。无需进一步检索。

**R3 复杂度对比**（四口径；以 manifest `llama-3.1-8b-long` (2,2048,32,128) causal fp16 **wl=1024** 为基准，有效 config 统一 bm=64/bn_eff=256；FLOPs 口径 = 乘加计 2；访存分列「GM 必流量」与「跨引擎 fabric 流量」；扫描遍数 = q/k/v 各自 GM 读取特征；中间缓冲 = 峰值驻留）：

| 算法候选 | FLOPs | 访存量 | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|--------|---------|-------------|---------------------|
| #0/#1 基线 flash online（单遍 per-block 链 + 带状裁剪，E1-E7 承载） | **51.6 GFLOP**（2 GEMM 主导；`4·B·H·total_attended·D`，`total_attended=1573376`） | HBM 必流量 ≈ 84MB（q/k/v 读 + o/lse 写各一次）；fabric（逐项可复算）：KV tile 流式重读 = **7680** 块 × (K+V tile 各 [256,128]f16 64KB) = 7680 × 128KB = **0.94GB**；ws 往返（写+读）= **7680** 块 × (S 64KB + P 64KB + O f32 64KB) = 7680 × 192KB = **1.41GB**；合计 ≈ **2.35GB** | q 1 遍；k/v 每任务流式 1 遍（H·B 组 × 32 q-block 重读，L2 命中） | per 块 S 32KB + P 32KB + O f32 32KB | 任务级 20 核并行（2048 任务）；跨核零同步；**每块 3 次 flag 握手 × 7680 块——发射流阻塞实测为结构地板**（MHA fa 域 busy aic_scalar 38–47%） |
| #2 两相位 S/P 物化（本设计） | 同 51.6 GFLOP（GEMM 工作量与 #1 逐块相同） | HBM 必流量同 ≈ 84MB；fabric：KV tile 流式重读 **0.94GB**（同 #1）+ ws 往返 = 7680 块 × (S 64KB + P 64KB + O **f16** 32KB) = 7680 × 160KB = **1.17GB**；合计 ≈ **2.11GB**（比 #1 少 0.23GB ≈ **10%**——O f16 减半） | 同 #1 | per-core 单槽 ws = `pad_band·bm·(f16)·2 + dim·L_max·bm·f16 ≈ 1.05MB`（L_max=5）；总量 20 核 ≈ 21MB（L2 ✓） | 任务级 20 核并行（2048 任务）；跨核零同步；flag 次数同量级 **但 wait 与真依赖对齐**（pass-1 零等待直跑）——MHA 同口径实测 2.31× |
| #4 带状裁剪（并入 #2 后量测） | 51.6 GFLOP（= 全 causal 68.8 的 0.75×） | HBM 同；fabric 随执行块数线性降 | 同 | 同 | **执行块数 7680 vs 全 causal 9216（−16.7%，块粒度裁剪）**；内段全有效块（零掩码）3840、边界掩码块 3840；无需额外结构 |
| #6 KV-split（split=2） | 同 + 合并开销 O(S·D·split) 可忽略 | HBM 同；fabric + 每 split 状态回传 | 同 | + 每 split 状态 (m,ℓ,o) × 核数 | 任务数 1024–2048 » 20 核，**无欠载场景**；**无核间原语**（T.set_flag.md §1 / T.sync_block_set.md §1 适用范围均为核内）——跨核合并须两阶段 kernel |
| #7 批量重基准（G=2/4） | 同 | O 回传 ÷G（两相位下 O 已 f16，收益再 ÷2） | 同 | UB + G×S 块缓冲 | 两相位下 α/PV 已解耦，G 的主要收益被吸收；残余仅 O 流量 |
| #8 f16 传输（并入 #2 后） | 同 | O/S 列已计入 #2 | 同 | 同 #2 | 同 #2 |

**R3 块数逐项复算明细**（8b-long：S=2048, bm=64, bn_eff=256, wl=1024, causal；`k_end=ceildiv((bx+1)·64,256)`、`k_start=max(0,bx·64−1024)//256`、`loop_count=k_end−k_start`）：

| bx | 0–3 | 4–7 | 8–11 | 12–15 | 16–19 | 20–31 |
|----|-----|-----|------|-------|-------|-------|
| loop_count | 1 | 2 | 3 | 4 | 5 | 5 |
| 小计（4 或 12 项） | 4 | 8 | 12 | 16 | 20 | 60 |

Σloop_count = 4+8+12+16+20+60 = **120**/（head,batch）；×（H·B = 32·2 = 64）= **7680 执行块**。全 causal（wl=−1，`k_end` 同式、`k_start≡0`）Σ = 144 ×64 = **9216** → 削减 **−16.7%**。内段全有效块 = Σmax(loop_count−2,0) = 60 ×64 = **3840**（边界掩码块 3840）。**旧值 3584 = Σ(loop_count−2) 未钳位 = 56×64**，系「剔除每任务 2 边界块的内段块数」，非执行块数。wl=256（8b/70b-short，S=512）：Σloop_count=12 ×(H·B) = 全 causal 同值（0 减免）。

**口径说明**：FLOPs 各候选相同（数学等价族，GEMM 块集合一致）——**本算子族的决定项是 fabric 流量与发射流行为**，非 FLOPs；#1 与 #2 的 fabric 流量差 ~10%（O f16 减半，0.23GB），但 MHA 同族实测性能差 2.31×——**结构差异（wait 与依赖的对齐方式）才是主导**。**与 MHA 的全量 causal 差异**：滑动窗口的块级裁剪使执行块数由 9216 降至 7680（8b-long wl=1024，**−16.7%**，按 §0.6 E-band 源式逐 bx 复算 Σloop_count=120/（head,batch）×64=7680）；**wl=256 短序列（8b/70b-short）块粒度下带未越界，执行块数与全 causal 相同（0 减免）**——带状红利集中在 wl=1024 长序列 + FLOPs 削减（−25%），而非块数。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 跨核结构 | 结论 |
|---------|-------------|---------|------------|---------|------------|---------|------|
| #2 两相位 + 带状裁剪（本设计） | GEMM→Cube（分形 M≥16/K≥32 由 E7 钳位、K=dim 64/128 ✓）、softmax/掩码/累加→Vector ✓ | L1 96KB/512KB（l1_a [64,256]+l1_b [256,128] f16，生命周期复用）✓；L0C [64,256] f32 = 64KB/128KB 单缓冲顺序复用 ✓；UB ~154KB/192KB per AIV（§4.5）✓ | 尾轴 bn_eff/dim 为 16 倍数（bn_eff 守卫公式保证）；[half,1] 行向量非搬运主路径（v11 `[L·half,1]` scales 先例） | `k_start/k_end/loop_count/L_*_full/tmc/tnc` 为 kernel_id/task_id 派生 PrimExpr（D3 实证 + 开发指南 §3.3 模板同类） | Cube pass-1 零等待（MTE2/MTE1/C/FIX 四通道天然重叠）；pass-2 P-wait 与真依赖对齐；per-块 flag 严格交替 | 任务独立零跨核同步 | ✅ **选定**（目标域任务数 1024–2048 ≥ 20 核，饱和域；smoke 128 任务轻度欠载，§5.5） |
| #0/#1 单遍 per-block 链 | 同上 | 同上 | ✓ | ✓ | **MHA 同族实测否决**：`sync_block_wait` 阻塞整条发射流（PL-1.9-hardlimits：ns=1≈ns=2、wide≈serial、深流水 +12% 三重证据）；fa 域调优终值仍 2.31× 慢于 #2 | 同上 | ❌（发射流结构地板，非 tiling 可解） |
| #6 KV-split | 合并段 Vector ✓ | ✓ | ✓ | ✓ | 需**跨核**数据交换：T.set_flag.md §1 / T.sync_block_set.md §1 均为核内语义——**无核间原语**（负向断言依据：两文档 §1 适用范围条款亲自核对）；两阶段 kernel host launch 开销淹没小 shape 收益 | 目标域任务数 1024–2048 » 20 核，**无欠载场景**——split 增加并行度无需求方 | ⏸ **defer**（收益域为 decode/小 batch，非本 workload；列入 §9.2） |
| #7 批量重基准 | ✓ | UB +G 块缓冲 | ✓ | ✓ | 两相位下 α/PV 已解耦，G 的结构收益被吸收；残余仅 O 流量 ÷G | ✓ | ⏸ **defer**（边际收益 <5%，复杂度不成比例；Stage 4 候选） |
| #8 f16 传输（fp16 路径） | ✓ | ✓ | ✓ | ✓ | ✓（PL-1.9：fixpipe f32→f16 正确数值转换 probe 实测；v11nt O-f16 tier-1 0 flips） | ✓ | ✅ **采纳并入 #2**（bf16 路径升 f32 承载） |
| #11 f16 softmax 链 | ✓ | ✓（大 pass 减半） | ✓ | ✓ | ✓（v9 实测有效） | ✓ | ⏸ **defer Stage 4**（f32 链 + f16 传输已达标） |
| #9 scale 折入 q | ✓ | ✓ | ✓ | ✓ | 引入任务头跨引擎串行依赖 | ✓ | ❌ **否决**（精度语义降级，§1.6.1 M11） |
| #5 双侧算术惩罚掩码 | Vector ✓（`vmin/vmax/vsub/vadd/vmul` 全 ALU 路径） | ✓（复用 f32 N-buffer） | ✓（diff f32 精确整数） | K_up/K_lo 为 PrimExpr ✓ | ✓ | ✓ | ✅ **采纳并入 #2**（PL-1.11 已验证；本容器 P2 探针 bit-exact + 复现 vcmp 标量化 warning） |
| #4 带状裁剪 | — | — | — | ✓（PrimExpr） | — | — | ✅ **采纳并入 #2**（源式保留 + 恒等元论证 M4） |

**调研结论**：**选定算法族 = 两相位 S/P 物化 + per-带内块 online softmax + 滑动窗口带状裁剪（候选 #2，含 #3/#4/#5/#8/#10 子项采纳）**。关键依据：① R4 实测证据——MHA 同族 fa 域两相位 98.05µs vs 单遍链调优终值 226.72µs（2.31×），单遍链发射流阻塞（aic_scalar 38–47% flag 自旋）经 11 轮调优被证实为结构地板；② R3——两相位不增加 GEMM 工作量与 HBM 必流量，fabric 增量反而略优（O f16 减半 ws 往返 1.41→1.17GB；总 fabric 2.35→2.11GB）；带状裁剪额外削减**执行块数（8b-long 7680 vs 全 causal 9216，−16.7%）**与 FLOPs（−25% vs 全 causal）；③ R4 全项亲和通过（L1/L0C/UB 预算 §4.5、PrimExpr 边界 D3 实证、per-块 flag 同款、任务独立零跨核）。**与基线（源算法）的结构差异一句话**：online softmax 数学递推与带状裁剪逐项保留，承载结构从「单遍 per-block 跨引擎握手链」换为「Cube/Vector 两相位独立大循环 + S/P/O GM 物化 + scales 延迟回放」。**「无更优替代」的调研范围**：skill 参考表 attention/softmax/elementwise 命中行、pattern-library 全主题（attention/traps/constants/layout/cases）、**examples/ 全目录 attention 族检索（12+ 文件）**、源码 §0.3/§0.4、TileOPs 侧 MHA 三轮调优全档案；候选 #6/#7/#11 以量化理由 defer（非「未考虑」），#1/#9 以实测/精度理由否决，#13/#14 结构不匹配。**源算法优化手段的意图承接**（§0.4 → 本设计，逐项）：OP1 online softmax→保留；**OP2 带状裁剪→保留（E-band，逐字移植源式）**；OP3 SMEM tiling→L1 生命周期复用 + Q-hoist；OP6 Pipelined→两相位（意图「重叠」由 pass-1 零等待承接）；OP5 tensor core→Cube；OP11 exp2 折叠→e 域 vexp（M1）；OP7 fragment 驻留→UB 驻留 + ws 物化（意图「消 GM 往返」让渡给 L2 命中的 ws 流量，代价计入 R3 且实测净赚 2.31×）；OP8 warp 归约→Vector 行归约；OP12 mask 融合→双侧算术惩罚（意图「免独立 mask pass」由「全有效中段零掩码 + 全向量构造」承接）；OP4 coalesced→slice-form copy；OP15 bf16 直连→Cube bf16；OP9/OP10/OP13 CUDA 专属→舍弃/重设计；OP14 lru_cache→保留。

**设计期估算（D-2）**：代表 workload = `llama-3.1-8b-long` (2,2048,32,128) causal wl=1024 fp16（**执行块数 7680**，§1.6.0 R3 逐 bx 复算）。**流量项**：GM 必流量 84MB ÷ 有效带宽（`CONST-mte2-degradation` 大流量档 ~18–34 GB/s/core × 20 核 ≈ 0.4–0.7 TB/s；910B2C 常数跨 SoC 未重验，标「未实测假设 + 估算依据」）= ~120–210µs；fabric ws 往返 **1.17GB** ÷ fabric（`CONST-fabric-bw` ≥1.73 TB/s，910B2C 实测，跨 SoC 未重验）= ~680µs（可与计算重叠）。**发射项**：Vector op 数 ≈ 130560（**7680 块 × ~17 + epilogue ≈ 132.6k**），per AIV（÷40 AIV）≈ 3315 op × `CONST-vector-launch-overhead` ~0.5µs/op（910B2C 实测，跨 SoC 未重验）= **~1660µs/AIV**。**容量项**：UB 154KB/192KB、L0C 64KB/128KB、L1 96KB/512KB，均未触发容量循环（单缓冲顺序复用）→ 0。**朴素发射口径估算下界 ≈ 1.6–2.5 ms**（发射项主导，流量项并行重叠）；**MHA 同族实测锚点交叉核对**：MHA 8b-long 全 causal 同结构实测 1191.6µs（PL-1.11，msprof op）→ 带状 7680/9216 = 0.83× ≈ **0.99ms**，乐观侧 ≈ 1.0–1.7ms——**两口径收敛区间 ≈ 1.0–2.5 ms**（朴素 op×0.5µs 上界 vs MHA 实测下界）。**Stage 4 首采校准**：fabric 有效常数、910B3 上的向量发射率（本设计期未实测）。
#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的两相位 + 带状裁剪算法。R1 中初判采纳的候选在此完成四要素论证（原式 → 优化后公式 → 等价性论证 → 收益量化）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| M1 | e 域换底 | `exp2(x·scale − m·scale)`，scale = s·LOG2E（源截断 1.44269504） | `vexp(x' − m')`，x' = prescale·x（fp32 域 vmul；prescale=1 时 trace-time 跳过） | 数学恒等：`e^u = 2^{u·log2e}`；本设计少 2 次 [half,bn] 乘法舍入 | 每 [rm,bn] 块省 2 个 v-op 的数值路径长度；数值更稳（MHA 谱系 23/23 + v11nt 实证；`verify_equiv.py` M1: 5 ulp, 违反率 0） |
| M2 | lse 尾式 | `log2(Σ e^{x−m}·e^m)`（需恢复大数指数） | `lse = vlog2(ℓ) + m·LOG2E`（全精度 1.4426950408889634） | `log2(ℓ·e^m) = log2(ℓ) + m·log2(e)` 恒等 | 免 [rm,1] 大数乘法与潜在溢出；与源 S13 同式（M2: 1 ulp, 0） |
| M3 | softmax 稳定化（减 max） | `e^{x_i}/Σe^{x_j}` | `e^{x_i−m}/Σe^{x_j−m}`，m = 行 max | 指数平移恒等（分子分母同乘 e^{−m}） | 消溢出（必选，源同构；M3 实证 f64 无 inf） |
| **M4** | **带状 KV 裁剪（滑动窗口核心）** | 全 KV 循环 `[0, NK)` | `[k_start, k_end)`（源 S4 公式），带外块不计算 | **恒等精确**：全掩码块贡献 = online 递推恒等元——`m_cur=−1e38`（clamp 有限）、`α=e^{m_prev−max(m_prev,−1e38)}=1`、`P=e^{−1e38−m}=+0.0`、`Δℓ=0`、`Δo=0`；逐块累加序为保序子序列 | 块数 **−16.7%**（8b-long 执行块 7680 vs 全 causal 9216；内段全有效块 3840、边界掩码块 3840）；FLOPs 51.6 vs 68.8 GFLOP；M4 实证 `alpha(all −inf block)==1` |
| **M5** | **双侧算术惩罚掩码（NaN 无关、全向量）** | 源 `if_then_else(掩码, −inf, 0)`（Cube 预填）；MHA 迁移用 `vcmp`+`vselect` | `pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1))`；`S' = S + pen·(−1e38)` | **恒等精确**：diff 为整数 → pen∈{0,1}；掩蔽位 `S'=S−1e38`（f32 大数吸收有限 S）→ `P=e^{−1e38−m}`（m 经 M6 clamp 有限）= **逐位 +0.0** ≡ 源 `e^{−inf}=0`；值无关（与 S 值无关，NaN/±Inf 无关） | 消 vcmp/vselect 标量化（PL-1.11 实测；本容器 P2 复现 `hivm.hir.vcmp ... scalar instruction` warning）；掩码链 ~6 个全速 ALU op；全有效中段块经 E-bnd 零掩码（M5 实证「masked bit-exact +0.0=True」） |
| M6 | 单调 max + clamp | `max(max_prev, block_max)` + clamp ≥ −1e38 | 同式（vmax ×2，[rm,1] 行向量） | 源 NaN guard 逐字保留：`m ≥ −1e38` 恒成立 → masked 位 `e^{−1e38−m}` 精确 0、无 `−inf−(−inf)` NaN 路径 | 防御性（必选，源同构）；2 个行向量 op/块（M6 实证） |
| M7 | prescale=1.0 跳过 | `vmul(S, 1.0)` | trace-time `if prescale != 1.0` 跳过 | 乘 1 恒等（x·1.0 = x，bit-exact） | 卫生项（本算子 prescale = dim^{−1/2} ≠ 1 恒成立，无跳过域） |
| M8 | S f16 物化（fp16 路径） | S f32 跨引擎 | `T.copy(l0_c f32 → ws_s f16)`（rint 隐式量化） | **容差内等价**：S 原始值域 \|S\| ≲ 60·dim^{−1/2} 尺度，f16 相对误差 2^-11 → ΔP/P ~2.6e-3（叠加 P 自身量化与 fp32 累加后仍在 §8.2 容差内；PL-1.9 probe max_rel 4.7e-4 = 1 f16 ulp） | S 跨引擎字节减半（[64,256] f32→f16）；Vector 侧 MTE2 读减半（M8/M9 实证 ULP 8779、违反率 0） |
| M9 | O 部分和 f16 物化（fp16 路径） | O_partial f32 跨引擎 | `T.copy(l0_c f32 → ws_o f16)` | **容差内等价，实测采纳**：PL-1.9 实测 `store_fixpipe f32(L0C)→f16(GM)` 为正确数值转换（max_rel 4.7e-4）；v11nt O-f16 tier-1 0 flips | O 跨引擎字节减半（32→16KB/块 [64,128]） |
| M10 | P 量化点保持 | 源：`acc_s(fp32) → cast 输入 dtype → GEMM2` | `vcast(P_f32 → 输入 dtype, rint)` 后经 ws_p 进 Cube | 量化点与源一致（softmax 后、GEMM2 前、最近舍入）；与源逐位同 dtype | 语义对齐（防漂移项，无独立收益；M10 实证） |
| M11 | **否决**：scale 折入 q | 每块 `vmul(S_f32, s)`（fp32 域） | 任务头 `vmul(q_f16, s)` 后进 GEMM | **容差内等价但精度语义降级，不采纳**：源乘 scale 在 fp32 域（相对误差 2^-24）；f16 q 折入使每 q 元素先舍入（2^-11）再进 L0C fp32 累加——S 相对误差升至 ~2^-11，经 exp 传播 ΔP/P ~2^-11·\|S\|·scale，威胁 §8.2 冻结容差 | （否决记录）理论省 NK−1 次 [half,bn] vmul/任务，但精度代价不成比例 |
| M12 | Q-hoist | 参考实现 pass-1 每块重读 Q | Q 在任务头一次装载 l1_a，pass-1 全程复用 | 数据不变（同一 Q 块），装载位置前移 | 免 `loop_count−1` 次 [tmc,dim] GM→L1 重读（MHA 谱系优势项） |
| M13 | ℓ 递推 clear=True | `T.reduce(...,"sum")` 无 clear | `T.reduce(...,"sum", clear=True)` | 参考的无 clear 形态为**文档化静默错误形态**（docs/Tilelang.language/规约操作/T.reduce.md §2.3）；clear=True 后 `ℓ ← ℓ·α + ΣP` 与源 S10 恒等 | 正确性修复项 |
| M14 | **否决**：boundary 块内逐列裁剪 | 边界块全宽处理（含全掩蔽列） | 逐列跳过 | 恒等精确但**列粒度 < 向量宽度**，无对应 API（向量 op 按整行/lane 组执行） | （否决记录）块粒度裁剪已由 M4 承担，残余 ±bn 列浪费为结构性常数 |

**优化结论**：采纳 M1/M2/M3/M4/M5/M6/M7/M8/M9/M10/M12/M13 共 12 项；否决 M11/M14（精度 / 无 API）；M7 在本算子 prescale≠1 域无触发。**验证覆盖口径（精确表述）**：**9 个数值型采纳项（M1/M2/M3/M4/M5/M6/M8/M9/M10）+ 4 个 E2E 域经 `verify_equiv.py` 机器验证 11/11 EQUIV_PASS**（结果表见下）；**M7/M12/M13 未入机器表，按构造论证**（M7 `/prescale=1` 本域无触发；M12 Q-hoist 结构性数据复用、无数值变更；M13 `clear=True` 为 `T.reduce.md §2.3` 文档化静默错误形态的正确性修复）——**非全部 12 项均过机器数值验证**，措辞以本段为准。

**优化后公式**（供 §3.1 使用）：

$$
\text{prescale} = \text{dim}^{-1/2},\quad
\text{pen}_{i,j} = \max\big(\text{clamp}(\text{diff}_{i,j}-K_{up},0,1),\ \text{clamp}(K_{lo}-\text{diff}_{i,j},0,1)\big)
$$
$$
S_{i,j}'' = \text{prescale}\cdot S_{i,j} + \text{pen}_{i,j}\cdot(-10^{38}),\qquad
\text{diff}_{i,j} = j-i
$$
$$
m_{\text{cur}} = \max_j S''_{i,j},\quad m = \max(m_{\text{prev}}, m_{\text{cur}}, -10^{38}),\quad
\alpha_l = e^{m_{\text{prev}} - m},\quad P = e^{S'' - m}
$$
$$
\ell \leftarrow \ell\cdot\alpha_l + \textstyle\sum_j P_{i,j},\qquad
o \leftarrow o\cdot\alpha_l + \big(\textstyle\sum_j P_{i,j}\, v_{j,:}\big)
$$
$$
\text{out} = o/\ell,\quad \text{lse} = \log_2 \ell + m\cdot\text{LOG2E}
$$

（`K_up`/`K_lo` 为 per-AIV、per-带内块的 PrimExpr 阈值，§0.6 E-mask；α_l 序列在 pass-1 计算并作用于 ℓ，同时存档 `scales[]`，pass-2 以同一序列作用于 o。PV 部分和由 Cube pass-2 计算、经 ws_o 以 f16 物化回传（M9；bf16 路径 f32——E5）。）

**等价性机器验证（S1-EQUIV-EXEC 门禁项；`verify_equiv.py`，torch CPU，2026-10-16 **rev-1 重跑**，退出码 0，11/11 EQUIV_PASS）**

对照口径：**基线式** = 上表「原式」列（GPU 源算子公式忠实模拟：exp2 域 + 源截断 LOG2E=1.44269504 + 源运算序 + 全 KV 循环 + 源逐元素掩码 + P 量化到输入 dtype + fp32 累加）；**优化式** = 上表「优化后公式」列（e 域 + 全精度常数 + 带状裁剪 + 双侧算术惩罚掩码 + f16 S/O 物化）；**参照** = fp64（torch CPU double）真值 + `golden_sw`（源语义物化 softmax）。工作负载 = S=256, H=4, H_kv=2, D=64, wl=64, causal, fp16 + bf16 镜像 + 角点（±inf 掩码位、全掩码块）。判定阈值：恒等链类 ≤128 ulp（第一性原理最坏界）；容差类 tier-1 违反率 =0（bf16 另测 tier-2 2-ulp 回退）。

| 项 | 比较域 (dtype) | max ULP (opt vs base) | 违反率 (opt vs 参照) | 违反率 (opt vs base) | 结论 |
|----|---------------|----------------------|--------------------|--------------------|------|
| M1 | P (f32) | 5 | 0 | 0 | EQUIV_PASS |
| M2 | lse (f32) | 1 | 0 | 0 | EQUIV_PASS |
| M3 | p (f64) | 0 | 0 | 0 | EQUIV_PASS |
| M5/M6 | P masked/valid (f32) | 0 | 0 | 0 | EQUIV_PASS |
| M4 | α (f32) | 0 | 0 | 0 | EQUIV_PASS |
| M8/M9 | out (f16) | 8779※ | 0 | 0 | EQUIV_PASS |
| M10 | P (f16) | 0 | 0 | 0 | EQUIV_PASS |
| E2E-fp16 | out (f16) | 196215※ | 0 | 0（vs 源基线） | EQUIV_PASS |
| E2E-fp16-golden | out (f16) | 2169264494※ | 0（tier-1） | — | EQUIV_PASS |
| E2E-bf16 | out (bf16) | 201611※ | 0 | 0（vs 源基线） | EQUIV_PASS |
| E2E-bf16-golden | out (bf16) | 2180678995※ | 0（tier-1 率 4.3e-4 → **tier-2 率 0**） | — | EQUIV_PASS |

※ 容差类行的 ULP 为**诊断值**：小幅度 out 元素（\|out\| ~ 1e-4）在 f16 网格上 ulp 极小，atol=5e-3 主导——判定以违反率为准。**关键断言明细**：M5/M6 masked 位 P 逐位 +0.0（bit-exact）、无 NaN；M4 全 −inf 块 α=1 精确；M10 rint 量化误差 ≤0.5 ulp；E2E-bf16 的 tier-1 4.3e-4 率经 D8 校准的 2-ulp 回退门（+2·2^-7 相对余量）归零——属源 `acc_s_cast` P 量化噪声在早 causal 行的放大（MHA 谱系同款 tier-3 分类口径），非缺陷（fp16 无此现象，tier-1 即 0）。**验证方式标注**：上表 11 行覆盖 9 个数值型采纳项（M1/M2/M3/M4/M5/M6/M8/M9/M10）+ 4 个 E2E；**M7/M12/M13 不在表内**，按构造论证——M7（`prescale=1` 跳过）本域无触发、M12（Q-hoist）结构性数据复用无数值变更、M13（`clear=True`）文档化静默错误形态的正确性修复。M11/M14（已否决）不在验证范围。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 分析对象 = 两相位 + 带状裁剪实现方案的全部循环与标量计算点。替代方案所用 API 均有 `docs/Tilelang.language/` 或 `examples/` 佐证（目录存在性已逐一核验：数学操作/、规约操作/、创建操作/、比较操作/、条件操作/、逻辑操作/、shape操作/、数据类型转换操作/）。

| # | 计算点 | 原实现形态（源 / MHA） | 向量替代方案（本设计） | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------------------|------------------------|---------|------------------------------|
| 1 | **双侧带状掩码构造** | 源 `T.Parallel(bm,bn)` 嵌套 if/else 谓词写；MHA `vcmp`+`vselect` | `T.arange`(f32 colmat/rowmat，每核一次) + `T.vsub`(diff) + `clamp(diff−K_up)`/`clamp(K_lo−diff)`（`vsub`+`vmax`+`vmin`）+ `T.vmax`(合成) + `T.vmul`(·−1e38) + `T.vadd`(入 S)（E-mask；API：创建操作/T.arange.md、数学操作/T.vsub.md、比较操作/T.vmax.md、比较操作/T.vmin.md、数学操作/T.vmul.md、数学操作/T.vadd.md） | ✅ | — |
| 2 | 全有效块分类（两段式分界） | 源无条件全块掩码；MHA 单侧 K_A | `if l < L_lo_full:` / `if l >= L_hi_full:` 两段 per-AIV 运行时（PrimExpr 阈值） | ✅（控制流经 E-bnd 消除掩码 op；块级判定非逐元素） | — |
| 3 | prescale 乘 | `T.Parallel` / 每元素 | `T.vmul(ub_f32_N, prescale, ub_f32_N)`（标量广播，数学操作/T.vmul.md §2.2.2） | ✅ | — |
| 4 | 行 max | warp 归约树 / reduce_max | `T.reduce(ub_f32_N, ub_mcur, dims=[1], reduce_mode="max", clear=True)`（规约操作/T.reduce.md §2.2.2） | ✅ | — |
| 5 | 单调 max / clamp | `T.Parallel(bm)` 标量 max | `T.vmax` ×2（行向量 [rm,1]，比较操作/T.vmax.md） | ✅ | — |
| 6 | α = e^{m_prev−m} | `T.Parallel(bm)` exp2 | `T.vsub` + `T.vexp`（行向量，数学操作/T.vexp.md） | ✅ | — |
| 7 | P = e^{S−m} | `T.Parallel(bm,bn)` exp2 | `T.vsub`（[M,N]−[M,1] 行广播）+ `T.vexp` | ✅ | — |
| 8 | 行 sum | warp 归约 | `T.reduce(..., "sum", clear=True)`（**clear 显式 True**——T.reduce.md §2.3 静默错误条款，M13） | ✅ | — |
| 9 | ℓ 更新 / o rescale 累加 | `T.Parallel` 标量 | `T.vmul`/`T.vadd`（行广播 / [rm,dim]，数学操作/T.vadd.md） | ✅ | — |
| 10 | P 量化 cast | `T.copy(acc_s, acc_s_cast)` | `T.vcast(ub_f32_N, ub_f16_N, round_mode="rint")`（数据类型转换操作/T.vcast.md） | ✅ | — |
| 11 | epilogue 归一化 | `T.Parallel` 除法 | `T.vdiv(acc_o, ub_ell, acc_o)`（[M,N]÷[M,1] 行广播，数学操作/T.vdiv.md） | ✅ | — |
| 12 | lse 尾式 | `T.Parallel(bm)` log2 + 乘加 | `T.vlog2(ub_ell, ub_lse, ub_lse_tmp)`（三参，数学操作/T.vlog2.md §1）+ `T.vmul` + `T.vadd` | ✅ | — |
| 13 | S/O 到达后的 vcast（f16→f32） | —（两相位新增搬运点） | `T.vcast(ub_f16_N, ub_f32_N, "rint")`（MHA L1375–1377 同构） | ✅ | — |
| 14 | 输出 / lse 写出 | 谓词标量 GM 写（源 S13） | 行截断切片 `T.copy`（E7；D3 教训：src 切片 `[0:rm, 0:dim]` 与 dst 同形） | ✅ | — |
| 15 | 状态初始化 | `T.clear`/`T.fill`（源 S2） | `T.vbrc(0/−inf, ub)`（shape操作/T.vbrc.md；标量 let-bound，D4） | ✅ | — |
| 16 | 任务解码 / `k_start`/`k_end`/`loop_count` / `L_lo_full`/`L_hi_full` / `tmc`/`tnc` 标量 | — | 无向量等价（每任务 O(1) 索引算术） | ❌ | **block 级索引 / 任务映射计算**：每任务 ~15 次整数标量运算（cid→bx/by/bz、带状范围、分界、分形钳位），非逐元素热点（对比每任务数万向量元素）；源 loop_range 亦为标量 |
| 17 | `if l < L_lo_full` / `if l >= L_hi_full` / `if task_id > 0` / `if l != 0` 控制分支 | — | 无向量等价 | ❌ | **tile 级顺序依赖 / 块分类**：两段式带状分界（E-bnd）、任务边界握手、首块 α 语义（logsum/acc_o 为 0，α_0 值无关）——均为块序控制流非数据并行；MHA/E1-E7 生产先例 |
| 18 | 工厂层 config 守卫 / 校验 / trace 分派 | — | 无向量等价 | ❌ | **host 侧元数据计算**（不在 kernel 内）：trace-time Python（`bn_eff` 守卫 + trace 家族分派先例） |

**向量化结论**：逐元素计算已全部向量化（15 项 ✅，替代方案 API 均有 docs 佐证；掩码 1 项由源 `T.Parallel` 谓词 + MHA `vcmp/vselect` 改为**全 ALU 算术惩罚链**，消除 vcmp 标量化）；保留 3 类标量/控制流——block 索引与带状范围（#16）、tile 级顺序依赖/块分类控制流（#17）、host 元数据（#18），逐项理由见表。**与 §6 一致性**：§6 循环结构中不出现任何逐元素标量循环。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout（BSHD 4D 进 / BSHD 4D out + [B,H,S] lse）是契约；核内布局与 lane 映射是设计变量。本算子为 Cube/Vector 混合类（MixCV）：Cube 侧分形布局由 `T.gemm`/`b_transpose` 内部承载，Vector 侧的 softmax 链、掩码链与累加链须枚举轴候选。

**Vector 侧候选矩阵**（fp16 向量宽度 ×8 = 128bit；行向量 = [rm,1] 广播语义）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **[rm, bn] 行主序 tile（S/P 链 + 掩码链）+ [rm, dim] 行主序 tile（O 链），lane = 列连续轴** | 列轴（l 维 / d 维，stride=1 连续） | **无需 repack**——ws_s/ws_p 的 [bm, pad_band] 行主序与 UB tile 同构，切片 copy 直达 | bn=256 fp16 → 每行 32 个 8-lane 向量组，尾 lane 浪费 0%；行广播（vmul [M,N]×[M,1]）沿行维展开为常数广播（T.vmul.md §2.2.2）；行归约 `T.reduce(dims=[1])` 沿连续轴归约（T.reduce.md §2.2.2） | ✅ **主选** |
| 2 | [bn, rm] 列主序（lane = 行轴） | 行轴 | 需核内转置（`T.transpose` 二轴交换，创建操作/T.transpose.md——UB 级执行但 **VP-D6 实测活跃源 transpose 毒化整 kernel 2.6×**） | 行广播退化为跨步广播（stride=bn）；归约沿跨步轴——**且 transpose 毒化税 2.6× 直接否决**（pattern-library traps-compiler.md VP-D6） | ❌（毒化实测 + 广播退化双重否决） |
| 3 | 单 AIV 全行 [bm, bn]（不半分） | 列轴 | 无 | UB 预算翻倍（154KB → ~300KB » 192KB ✗ 编译失败）；放弃 2×AIV 并行 | ❌（UB 容量硬上限否决；`CONST-capacity-910B2C` UB 192KB/AIV） |
| 4 | GPU 源码轴映射（warp 32 线程 × `GemmWarpPolicy.FullRow`） | — | — | — | ❌（GPU thread/warp 轴与 NPU Vector 轴不对应——migration-analysis.md §5.3；本设计 Vector 轴独立评估，见 #1） |

**掩码式选择（本算子特有的阻塞级决策）**：

| 候选 | 机制 | 代价 | 依据 | 决策 |
|------|------|------|------|------|
| **算术惩罚式**（`clamp(diff−K,0,1)`+`S+=pen·(−1e38)`） | 全 ALU 向量路径 | ~6 op/块，含掩码 | PL-1.11 第五轮实测（vcmp int16 全形态标量化 236µs/链 @[32,256]）；本容器 **P2 探针实测主选链 7/7 bit-exact 且零标量化 warning**（`design_probe_band_mask.py`） | ✅ **主选（band-free trace）** |
| `vcmp`+`vand`+`vselect` | bool 掩码选择语义 | vcmp 标量化（aiv_scalar 主导） | docs 比较操作/T.vcmp.md 支持，但 PL-1.11 实测标量化；本容器 **P2 探针复现 2 条 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency` warning**（数值 7/7 bit-exact，但发射代价高） | ⏸ **备选**（`has_band` 尾带 ragged trace 回退——加法掩码无法清 NaN，PL-1.11 P8；配合 `l1_b` zbuf 零初始化） |

**Cube 侧决策**：分形 NZ 布局由 `T.gemm(l1_a, l1_b, l0_c, b_transpose=True, size=[tmc,dim,tnc])` 内部承载（T.gemm.md §2.1 Expert 签名）；**不采用** `T.load_nd2nz`（D1 实测：对 BSHD 跨头切片静默平铺误读——`T.load_nd2nz(q[bz,s_lo,by,0],l1,[m,dim])` 把跨维 (S,D) tile 当连续内存读，probe 复现 max diff 49；**BSHD 定头切片必须 slice-form `T.copy`**，CG-2026-0004）；L1 缓冲 l1_a/l1_b 生命周期复用（pass-1 Q/K → pass-2 P/V，`block_share = max(bn_eff, dim)`）；epilogue 归属：归一化/lse/cast 全在 Vector 侧（L0C 不能被 v 算子直接操作，出数必经 GM ws，E2/CG-2026-0005）。

**布局决策结论**：选定方案 #1——Vector 链核内布局 = 行主序 [rm,bn]/[rm,dim] tile、向量化轴 = 列连续轴、repack = 无；掩码 = 算术惩罚式（备选 vselect 仅尾带 ragged trace）；Cube 侧 = `T.gemm` 内部分形 + slice-form copy 装载 + L1 生命周期复用。**该结论同步落入 §3.3 伪代码与 §6 循环结构**（累加循环的内层向量维 = 列轴）；弃选方案量化理由见候选矩阵（#2 毒化税 2.6×、#3 UB 超限、#4 轴系不对应）。

**实验裁决计划（D-3，未知常数依赖）**：主选（算术惩罚式）与备选（vselect）的判定裕度依赖未在本容器实测的量——`vmin/vmax` 的向量吞吐与 `vcmp` 标量化税的相对比（PL-1.11 常数绑定 910B2C / CANN 8.5.0，本容器为 910B3 / CANN 9.0.0，跨 SoC/工具链未重验）。**主选 = 算术惩罚式**（进 Stage 3）；**备选 = vselect 形态**（结构完整：掩码链替换为 `arange int16 colmat/rowmat` + `vcmp le/ge` + `vand` + `vselect(−inf哨兵)`，其余结构及分核不变）；**裁决计划**：代表 shape = smoke + 8b-short/8b-long × fp16，测 `msprof op Task Duration`（median of 20），判定阈值 = 备选 latency < 主选 × 0.95 才翻转，结果回写 §1.6.3。**设计期探针（P2，rev-1 修复并重跑）**：本容器实测**主选双侧算术惩罚链 7/7 配置 bit-exact PASS 且零标量化 warning**（有效掩码 + `S'=S+pen·(−1e38)` 哨兵逐位校验）；被否决的 `vcmp/vand/vselect` 备选链亦 7/7 bit-exact，但复现 2 条 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency` warning——结论已写入 §0.6 E-mask 与本节主选理由（探针脚本 `history_version/design_probe_band_mask.py`；rev-1 修正阈值 let-bound/kernel 派生 PrimExpr 并补入主选链）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Expert**（显式双 Scope）。**证据化结论**：本算子为 MixCV（Cube `T.gemm` + Vector softmax/掩码链），**Developer 模式（auto CV-split）在本容器上不可行**——设计期探针 **P1 实测确认**（下述），与已知能力缺口 **CG-2026-0010** 及模式 **PL-1.16-expert-dualscope-bypass** 一致。切 Expert 的决策**非默认假定，而是探针实测 + 先例双证**。

### 2.2 选型理由

1. **结构表达力（决定性）**：选定算法（§1.6.0 #2 两相位 + 带状裁剪）要求 `T.Scope("Cube")`/`T.Scope("Vector")` 显式双引擎分工、两相位独立大循环的手动调度、`T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub` 显式分层、per-带内块 `T.sync_block_set/wait` 手动握手、`T.rs("PIPE_*")` 通道标注——均为 Expert 形态 API（`T.load_nd2nz.md §2.3`「适用 Expert 模式，在 Developer 模式中使用 copy 接口」同族口径；`TILELANG_ASCEND_MODE` 默认即 Expert——docs/developer/EnvironmentVariables.md L31）。**结构原型 MHA 先例与 tuned 母本均为 Expert 形态**。

2. **Developer 模式对照（本容器探针 P1 实测，`history_version/design_probe_devmode_mix.py`）**：

| 探针变体 | 结构 | 结果 |
|---------|------|------|
| A | Developer：persistent（`T.Kernel(4,is_npu=True)`+`T.serial(2)`）+ `T.gemm`，**无 v-prefix** | **PASS**（编译+运行、数值 finite） |
| B | Developer：同上 + `T.vmul(acc, fac, acc)`（**v-prefix 混排**） | **BLOCKED**（BishengIR 编译崩溃 `std::out_of_range` / `map::at`） |
| C | Expert：显式双 Scope（`alloc_L1/L0C/ub` + `PIPE_*`）同结构 | **PASS**（编译+运行） |

   **结论**：Developer auto CV-split **无法承载**「persistent 分核 + `T.gemm` + v-prefix 向量 op」的混排（变体 A 无向量 op 时正常，变体 B 加向量 op 即崩）；Expert 双 Scope 同结构正常。**本容器（tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / Ascend910B3）的失败点位于 BishengIR 编译期**（`map::at` 越界崩溃，无编译期诊断），与 CG-2026-0010 记录的运行时崩溃（`Illegal instruction … unaligned UUB addresses`，retCode=0x31）**失败层不同但结论同向**——均属「Developer MIX 不可用」。CG-2026-0010 明确写：「persistent 混合算子在 Developer 模式不可实现——被迫切换 Expert（**GQA/sparse_mla 先例**）完成」；PL-1.16 给出判据（persistent 分核 + gemm 混排运行时崩溃 → Expert 双 Scope bypass）与 repro `repro/TRAP-DEVMODE-PERSIST-GEMM.py`。本算子正落在该缺口与判据的交集。

3. **Developer 可表达子集已被排除**：Developer 形态（fragment + 自动 cv_split + `T.Pipelined`）产出的是 per-block 串行链形态（MHA E1-E7 developer 基线：causal 域长 KV 比两相位慢 1.57–1.63×，PL-1.7）；两相位结构超出 Developer 表达力（无法表达「Cube pass-1 全 S 无等待直跑 + pass-2 逐块等 P」与 per-块手动握手）。

4. **同型先例**：`examples/TileOPs/.../multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（GQA/MHA prefill，Expert 两相位，29/29 + 11 轮调优）与 `examples/flash_attention/flash_attn_npuir.py`、`examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` 均为 Expert 双 Scope——本算子与它们结构同型（BSHD + flash online softmax + MixCV）。

**备选（未采纳）**：**hybrid**（部分 Developer/部分 Expert）——无先例且本算子为单一 MIX kernel（不可拆分），不采用。**CG-2026-0010 的能力缺口登记**：本算子复现了「Developer 对 persistent+gemm+向量混排的不支持」（`occurrences` 应 +1；缺口层 = codegen / BishengIR），按 AGENTS.md「Compiler Capability Gap Rule」在 §9 记录（供后续登记；本容器失败层为编译期，与既有条目的运行期不同，属同缺口的新触发层，需在 capability-gaps.md 内注明）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 显式 `T.alloc_L1`（l1_a/l1_b）、`T.alloc_L0C`（l0_c 单缓冲）、`T.alloc_ub`（Vector 链全量缓冲，§4.5 预算） |
| 计算方式 | Cube：`T.gemm`（Scope("Cube") 内，initC/b_transpose/size）；Vector：v-prefix 链（vcast/vsub/vmax/vmin/vmul/vadd/vexp/reduce/vdiv/vlog2/vbrc/vcast） |
| 同步 | 手动 `T.sync_block_set/wait`（per-带内块 id + FLAG_TASKDONE，§7.2）；`T.rs("PIPE_MTE2"/"PIPE_MTE3"/"PIPE_FIX"/"PIPE_V")` 通道标注 |
| 流水 | 手动两相位（无 `T.Pipelined`）；pass_configs 双关闭防编译器重排（§7.3） |
| 环境变量 | `TILELANG_ASCEND_MODE=Expert`（setdefault，docs/developer/EnvironmentVariables.md L31，显式防残留） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 优化后公式；每步与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `S = q_tile @ k_tile^T`（size [tmc, dim, tnc]），`k_tile = k[·, n_lo:n_lo+tnc, kv_head, :]`，`n_lo = (k_start+l)·bn_eff` | QK^T（Cube pass-1，Q-hoist，带状块） |
| 2 | `S_f16 ← f16(S_f32)`；`Ŝ ← f32(S_f16)` | S 物化 + 到达（M8；bf16 路径 f32 直传） |
| 3 | `pen = max(clamp(diff−K_up,0,1), clamp(K_lo−diff,0,1))`（分段：仅 `l<L_lo_full` 施左窗段、`l≥L_hi_full` 施因果段） | 双侧带状掩码（M5，算术惩罚式） |
| 4 | `S'' = prescale·Ŝ + pen·(−1e38)` | prescale（M1 前置）+ 掩码（M5） |
| 5 | `m_cur = max_j S''`；`m = max(m_prev, m_cur, −1e38)` | 行 max + 单调 + clamp（M3/M6） |
| 6 | `α_l = e^{m_prev−m}`；`P = e^{S''−m}` | α 与 P（M1 e 域） |
| 7 | `ℓ ← ℓ·α_l + Σ_j P`；`scales[l] ← α_l` | ℓ 递推 + α 存档（clear=True，M13） |
| 8 | `P_dtype ← dtype(P)`；`O_partial(l) = P_dtype @ v_tile`（size [tmc, tnc, dim]） | P 量化（M10）+ PV 部分和（Cube pass-2） |
| 9 | `o ← o·α_l + f32(O_partial_f16)` | 延迟 rescale 累加（Vector pass-2，M9） |
| 10 | `out = o/ℓ`；`lse = log2(ℓ) + m·LOG2E` | epilogue（M2；transpose-free 写出） |

### 3.2 TileLang API 映射（Expert 模式，逐段）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Q/K/V 装载 | `T.copy`（slice 形态） | `T.copy(q[bz, s_lo:s_lo+tmc, by, 0:dim], l1_a[0:tmc,0:dim])`；k/v 同款（D1 约束：BSHD 定头切片必须 slice 形态，CG-2026-0004） | Expert |
| 1 | QK^T | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True, size=[tmc, dim, tnc])`（T.gemm.md §2.1） | Expert |
| 2 | S 物化 | `T.copy` | `T.copy(l0_c[0:tmc,0:tnc], ws_s[kid, 0:tmc, l·bn_eff:(l+1)·bn_eff])`（f32→f16 隐式转换，PL-1.9 probe） | Expert |
| 3 | 掩码 | `T.arange` + `T.vsub` + `T.vmax` + `T.vmin` + `T.vmul` + `T.vadd` | `diff` f32 [half,bn]（每核一次）；分段 `if l < L_lo_full` / `if l >= L_hi_full`（PrimExpr 阈值） | Expert |
| 4 | prescale | `T.vmul` | 标量广播（数学操作/T.vmul.md §2.2.2） | Expert |
| 5 | 行 max / 单调 / clamp | `T.reduce` + `T.vmax` ×2 | `dims=[1], reduce_mode="max", clear=True`（T.reduce.md §2.2.2） | Expert |
| 6 | α / P | `T.vsub` + `T.vexp` | 行向量 / 行广播 | Expert |
| 7 | 行 sum + ℓ 更新 + α 存档 | `T.reduce` + `T.vmul`/`T.vadd` + `T.copy` | `reduce_mode="sum", clear=True`（§2.3 强制）；`scales[l·half:…,0:1]` 切片存取 | Expert |
| 8 | P 量化 + 回传 | `T.vcast` + `T.copy` | `round_mode="rint"` 到输入 dtype（M10）；PIPE_MTE3 段内 copy + set flag | Expert |
| 8 | PV 部分和 | `T.gemm` | `T.gemm(l1_a, l1_b, l0_c, initC=True, size=[tmc, tnc, dim])` | Expert |
| 9 | O 累加 | `T.vcast` + `T.vmul` + `T.vadd` | `vcast(ub_f16_D, ub_f32_D, "rint")`；行广播 rescale | Expert |
| 10 | epilogue | `T.vdiv` + `T.vcast` + `T.vlog2` + `T.copy` | `vlog2(ub_ell, ub_lse, ub_lse_tmp)` 三参；行截断切片写出（[B,H,S,1] 视图） | Expert |
| — | 跨引擎握手 | `T.sync_block_set` / `T.sync_block_wait` | flag id = 带内块局部下标 l + FLAG_TASKDONE（§7.2 表） | Expert |
| — | 状态初始化 | `T.vbrc` | let-bound 标量（D4） | Expert |

### 3.3 计算伪代码

> `T.Tensor` 的 dtype 按模板 §3.3 规则以位置参数传递（false-alarm 规避）。完整结构 = §0.6 E2/E-band/E-mask 的 Cube/Vector 流伪代码，此处补任务解码外壳与 trace 家族说明：

```python
# 工厂层（trace-time）
# bm = 64; bn_caller（wrapper 默认 64/32）; num_stages 保留旋钮
# eff_left  = wl if wl >= 0 else seq_len
# eff_right = wr if wr >= 0 else 0
# bn_eff = max(bn_caller or 256, ceil16(ceildiv(bm + eff_left + eff_right, 14)))  # flag 预算守卫（E-band）
# L_max  = 工厂期按 k_start/k_end 公式遍历 bx 求最大 loop_count；assert L_max + 1 <= 16
# pad_band = L_max * bn_eff
# trace 家族（VP-P2 工厂级多 @T.prim_func 分派，Python if 选 builder）:
#   mask 域: {causal+left} / {causal only} / {left only} / {left+right} / {none}
#   × fp16 / bf16（ws dtype: f16/f16/f16 vs f32/bf16/f32）
#   × has_band（尾带 OOB 掩码变体，seq_len % bn_eff 或 tail < 32 时）

@T.prim_func
def _gqa_sw_fwd_main(
    q: T.Tensor((batch, seq_len, heads, dim), dtype),
    k: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
    v: T.Tensor((batch, seq_len, heads_kv, dim), dtype),
    ws_s: T.Tensor((NUM_KERNELS, bm, pad_band), ws_s_dtype),
    ws_p: T.Tensor((NUM_KERNELS, bm, pad_band), dtype),
    ws_o: T.Tensor((NUM_KERNELS, bm, dim * L_max), ws_o_dtype),
    output: T.Tensor((batch, seq_len, heads, dim), dtype),
    lse: T.Tensor((batch, heads, seq_len, 1), "float32"),   # [B,H,S,1] 视图（E6，零 transpose）
):
    with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
        HB = heads * batch
        num_local = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)
        # 每核一次：diff = arange(col) - arange(row)（[half, bn_eff] f32）
        # ---- Cube Scope: persistent 任务循环 → [TASKDONE wait] → Q-hoist → pass-1 → pass-2
        with T.Scope("Cube"):
            l1_a = T.alloc_L1([bm, block_share], dtype)      # block_share = max(bn_eff, dim)
            l1_b = T.alloc_L1([bn_eff, dim], dtype)
            l0_c = T.alloc_L0C([bm, block_share], "float32")  # 单缓冲，gemm1/gemm2 顺序复用
            for task_id in T.serial(num_local):
                cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                bz = cid % batch; by = (cid // batch) % heads; bx = cid // HB
                s_lo = bx * bm; kv_head = by // groups
                tail_m_real = T.min(bm, seq_len - s_lo)
                tmc = T.max(16, T.min(bm, T.ceildiv(tail_m_real, 16) * 16))
                # 带状范围（源 S4 逐字移植）
                k_end = (T.ceildiv(T.min(seq_len, (bx+1)*bm), bn_eff) if is_causal
                         else T.ceildiv(T.min(seq_len, (bx+1)*bm + eff_right_t), bn_eff))
                k_start = T.max(0, bx*bm - wl) // bn_eff if wl >= 0 else 0
                loop_count = T.max(k_end - k_start, 0)
                if task_id > 0:
                    with T.rs("PIPE_MTE2"): T.sync_block_wait(FLAG_TASKDONE)
                with T.rs("PIPE_MTE2"):
                    T.copy(q[bz, s_lo:s_lo+tail_m_real, by, 0:dim], l1_a[0:tail_m_real, 0:dim])
                for l in T.serial(loop_count):   # pass-1 (全 S，无等待直跑)
                    # n_lo = (k_start+l)*bn_eff; tn_real/tnc 钳位；K 装载 + gemm1 + ws_s 物化 + set(l)
                for l in T.serial(loop_count):   # pass-2 (全 PV)
                    # wait(l) + P/V 装载 + gemm2 + ws_o 物化 + set(l)
        # ---- Vector Scope: persistent 任务循环 → init → pass-1 → pass-2 → epilogue
        with T.Scope("Vector"):
            # ub 缓冲族（§4.5 预算）；diff [half,bn_eff] f32 任务循环外一次构建
            for task_id in T.serial(num_local):
                # 同款任务解码；real_m 自适应半分；L_lo_full/L_hi_full per-AIV 阈值
                for l in T.serial(loop_count):   # pass-1: 掩码 → softmax 链 → P 量化 → θ 递推
                    ...
                for l in T.serial(loop_count):   # pass-2: 延迟 rescale 累加
                    ...
                # epilogue + set FLAG_TASKDONE
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.gemm`（Expert 签名 initC/b_transpose/size） | docs/Tilelang.language/线性代数操作/T.gemm.md §2.1/§2.2.1/§2.3；MHA 先例 L334/L376 | ✅ MHA 生产实测 |
| `T.copy`（slice 形态 GM↔L1/UB、L0C→GM，跨 dtype 隐式转换） | debug_log D1；PL-1.9（f32→f16 转换 probe）；MHA 全套 | ✅ MHA 实测 |
| `T.vcast`（rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md；MHA L506/L599/L636/L650 | ✅ |
| `T.reduce`（dims=[1]，clear=True） | docs/Tilelang.language/规约操作/T.reduce.md §2.2.2/§2.3；MHA L575/L594 | ✅ |
| `T.arange`/`T.vsub`/`T.vmax`/`T.vmin`/`T.vmul`/`T.vadd`/`T.vexp`/`T.vdiv`/`T.vlog2`/`T.vbrc` | docs/Tilelang.language/ 对应子目录（数学操作/、创建操作/、shape操作/）逐一存在性核验；MHA 生产使用；**`T.arange`+`T.vsub` 掩码链 + `T.vmin/vmax` 算术惩罚式经本容器 P2 探针 bit-exact 实测** | ✅ 实测 |
| `T.sync_block_set/wait`（带内块 id 复用 + TASKDONE） | docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1；MHA L308/L351/L362/L386 | ✅ MHA 生产实测 |
| `T.Kernel(20, is_npu=True) as (kernel_id, subid)` | MHA 生产（persistent）；T.sync_block_set.md §2.4 示例同款 (cid, subid) | ✅ P1 探针 variant C 实测 |
| `T.rs("PIPE_*")` 通道标注 | mixcv/deepseek 先例；MHA 全套；P1 探针 variant C | ✅ |
| `T.vselect`（备选掩码形态） | docs/Tilelang.language/条件操作/T.vselect.md；MHA L554/L563；P2 探针 | ✅（备选） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（强制检测 5 项）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源为三维 grid） | E1：一维 persistent 20 核 + 任务解码（bx/by/bz 派生） |
| L0C 容量 128KB | **Yes**（l0_c [bm, block_share] f32） | 单缓冲 [64, 256] f32 = 64KB ≤ 128KB ✓（`CONST-capacity-910B2C`；bm·max(bn_eff,dim)·4 ≤ 128KB → bm ≤ 128@bn=256） |
| L1 容量 512KB / UB 192KB | **Yes** | L1 = l1_a 32KB + l1_b 64KB = 96KB ✓；UB ≈ 154KB per AIV（§4.5 逐项）✓ |
| GEMM 非整除 / 小尺寸分形 | **Yes**（契约域 tail < 16/32、S % bn ≠ 0） | E7：tmc/tnc 分形下限钳位（16/32）+ 尾带 OOB 掩码（trace-time 门控）+ 行截断写出 |
| store_fixpipe / L0C 出数 dst 仅 GM | **Yes**（S/O 跨引擎） | E2：ws 三族 GM 物化（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1；CG-2026-0005） |
| **Developer MIX 不可用** | **Yes**（persistent + gemm + v-prefix） | §2：Expert 双 Scope（P1 探针 + CG-2026-0010 + PL-1.16） |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | 三维 `T.Kernel(m, heads, batch)` + threads | 一维 persistent 20 + 任务解码 | E1 |
| GEMM API | `T.gemm(transpose_B, policy=FullRow)` fragment | `T.gemm(L1, L1, L0C, initC, b_transpose, size)` | §3.2；FullRow 舍弃 |
| 内存分配 | alloc_shared/fragment 自动 | alloc_L1/L0C/ub 显式分层 + GM ws | Expert 模式 |
| 流水 | `T.Pipelined`（cp.async） | 两相位手动调度 + per-带内块 flag | E2/E-band |
| **带状 KV 范围** | `k_start`/`k_end` on `T.Pipelined(loop_count)` | 同式移植，`for l in T.serial(loop_count)`，`n = k_start+l` | E-band |
| **掩码** | `T.Parallel` 谓词 `if_then_else` | 双侧算术惩罚式（`clamp`+`vmax`+`vadd`） | E-mask（PL-1.11） |
| 尾块处理 | 谓词逐元素装载/回写 | size-form + 尾钳位切片 + 掩码 | E7 |
| lse 写出 | `T.copy` 3D | [B,H,S,1] 视图 + 自然 2D 区域写 + host reshape | E6（VP-D6） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | **极高（直接母本）** | Expert 两相位、per-块 flag、单 L0C 复用、尾钳位 size-form、scales 回放、自适应行半分、ws 三族、Q-hoist、transpose-free lse |
| `examples/flash_attention/flash_attn_npuir.py` | **高（结构原型）** | 两相位大循环、双 Scope、sync_block_set/wait 最小形态 |
| `examples/mixcv/mixcv_mixkernel.py` | 中 | `T.rs` + `sync_block_set/wait` 最小形态 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/.../perf_opt/` | 中 | persistent 分核 + 迁移范式 + TUNED 分派表模式 |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| q | (batch, seq_len, heads, dim) | float16 / bfloat16 | BSHD；工厂期全静态 |
| k | (batch, seq_len, heads_kv, dim) | same_as(q) | GQA：`heads % heads_kv == 0` |
| v | (batch, seq_len, heads_kv, dim) | same_as(q) | |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| output | (batch, seq_len, heads, dim) | same_as(q) | Op 契约输出（wrapper 返回） |
| lse | (batch, heads, seq_len) | float32 | kernel 第 2 输出；prim_func 声明 `[B,H,S,1]` 视图（E6）；**Op wrapper 丢弃** |

### 4.3 中间缓冲区（Expert 显式分层；设计默认 bm=64, bn_eff=256, dim=128, L_max≤5, half=32 基准）

| 层级 | buffer | shape | dtype | 生命周期 / 说明 |
|------|--------|-------|-------|-----------------|
| L1 | l1_a | [bm, block_share] = [64, 256] | dtype | pass-1 持 Q（rows×dim）/ pass-2 持 P（rows×tnc）——生命周期复用（block_share = max(bn_eff,dim)） |
| L1 | l1_b | [bn_eff, dim] = [256, 128] | dtype | pass-1 持 K / pass-2 持 V |
| L0C | l0_c | [bm, block_share] = [64, 256] | float32 | 单缓冲，gemm1/gemm2 顺序复用（64KB/128KB） |
| GM ws | ws_s | [NUM_KERNELS=20, bm, pad_band] | fp16: f16 / bf16: f32 | Cube→Vector 的 S 块（单槽，per-带内块切片） |
| GM ws | ws_p | [20, bm, pad_band] | 输入 dtype | Vector→Cube 的 P 块 |
| GM ws | ws_o | [20, bm, dim·L_max] = [20,64,640] | fp16: f16 / bf16: f32 | Cube→Vector 的 PV 部分和 |
| UB | 见 §4.5 | — | — | Vector 链全量缓冲 |

`pad_band = L_max · bn_eff`。ws 总量（8b-long，fp16，L_max=5, pad_band=1280）：ws_s = 20·64·1280·2 = 3.28MB；ws_p = 3.28MB；ws_o = 20·64·640·2 = 1.64MB；合计 **≈ 8.2MB**（bf16 路径 ws_s/ws_o 升 f32 ≈ 13.8MB）——L2（192MB 口径，MHA 谱系）下与 KV 工作集共存无压力。

### 4.4 内存搬运路径

```
GM[q] ─slice T.copy(Q-hoist, 每任务一次)─> L1[l1_a]
GM[k] ─slice T.copy(每带内块)─> L1[l1_b] ─┐
   L1[l1_a]×L1[l1_b] ─T.gemm(b_transpose)→ L0C[l0_c] ─T.copy(FIX, f32→f16 隐式)→ GM[ws_s] ─set(l)
GM[ws_s] ─T.copy(MTE2)→ UB[ub_f16_N] ─vcast→ UB[ub_f32_N] ─掩码/softmax 链→ UB[ub_f16_N(P)]
UB[ub_f16_N] ─T.copy(MTE3)→ GM[ws_p] ─set(l) ─> L1[l1_a] ─┐
GM[v] ─slice T.copy(每带内块)─> L1[l1_b] ─┘
   L1[l1_a]×L1[l1_b] ─T.gemm(initC=True)→ L0C[l0_c] ─T.copy(FIX)→ GM[ws_o] ─set(l)
GM[ws_o] ─T.copy(MTE2)→ UB[ub_f16_D] ─vcast→ UB[ub_f32_D] ─vmul(α)+vadd→ UB[acc_o]
UB[acc_o]/ub_ell ─vdiv/vlog2/vcast→ UB ─slice T.copy(行截断)→ GM[output] / GM[lse ([B,H,S,1] 视图)]
```

**layout 变换点**：① GM→L1 的 BSHD 定头切片（slice-form `T.copy`，**禁止 `T.load_nd2nz`**——D1 实测跨头切片静默平铺误读，CG-2026-0004）；② L0C→GM 的 f32→f16 隐式量化（PL-1.9，正确数值转换）；③ UB→GM 行截断切片（src `[0:rm,...]` 与 dst 同形）。**无核内重排、无 transpose**（Vector 布局与 ws 同构，行主序直达）。

### 4.5 UB 内存预算（设计默认 bm=64/bn_eff=256/dim=128/L_max=5/half=32；causal+left 主 trace；per AIV）

| buffer | shape | dtype | 字节 |
|--------|-------|-------|------|
| ub_f16_N | [32, 256] | float16 | 16 KB |
| ub_f32_N | [32, 256] | float32 | 32 KB |
| ub_f16_D | [32, 128] | float16 | 8 KB |
| ub_f32_D | [32, 128] | float32 | 16 KB |
| acc_o | [32, 128] | float32 | 16 KB |
| ub_diff | [32, 256] | float32 | 32 KB |
| ub_pen | [32, 256] | float32 | 32 KB |
| ub_m / ub_mprev / ub_mcur / ub_alpha / ub_ell / ub_ellcur / ub_t / ub_zero / ub_lse / ub_lse_tmp | [32, 1] × 10 | float32 | 1.25 KB |
| scales | [L_max·half, 1] = [160,1] | float32 | 0.64 KB |
| **合计** | | | **≈ 154 KB** |

**对照 192KB/AIV**（`CONST-capacity-910B2C`）：余量 ~38KB（~20%）。按 BishengIR auto-multi-buffer 通涨系数 ×1.10–1.12（CG-2026-0008，显式 alloc 结构实测 ~0.98–1.0）→ 154×1.12 ≈ **172KB ≤ 192KB ✓**。`has_band` trace 另需 `ub_neg [32,bn_eff] f32`（+32KB → 186KB，仍 ✓，但裕量收窄至 ~3%——UB diet 候选见 §9）。`ub_pen` 可与 `ub_f32_N` 原位复用（省 32KB → 122KB），本表按保守（独立缓冲）预算。

### 4.6 动态轴定义

无运行时动态 shape——batch/heads/heads_kv/seq_len/dim 均工厂期静态（`lru_cache` key 的一部分）；`k_start/k_end/loop_count/tmc/tnc/L_lo_full/L_hi_full` 为 kernel_id/task_id/l 派生 PrimExpr（静态可 lower，D3 实证）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-2, -1],                       # output, lse 居末两位（源 out_idx=[3,4] 同构）
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

**判定依据**: QK^T 与 PV 为 MAC 密集段（Cube），softmax 链/带状掩码/累加/epilogue 为逐元素与规约段（Vector）；两相位结构以 per-带内块 flag 协同双引擎。

### 5.2 Block 划分

```python
bm = 64        # 设计默认。依据：① UB 预算（§4.5：bm=64 时 154KB ✓；bm=96 需 UB diet）；② 与 wrapper 默认 block_m 一致；
               #   ③ L0C [bm,256] f32 = 64KB 余量大（上限 bm=128）；④ 自减半分 half=bm//2=32。
bn_eff = 256   # 设计默认（caller 传 wrapper 默认 (64/32) 时替换）。依据：① 向量发射开销定律（PL-1.9-hardlimits：
               #   每 op ~0.5µs 固定成本，宽块 op 恒优于窄块 op 翻倍）；② flag 预算守卫（E-band）；
               #   Guard: bn_eff = max(bn_caller, ceil16(ceildiv(bm + eff_left + eff_right, 14)))
L_max = 工厂期按 k_start/k_end 遍历 bx 求最大 loop_count；assert L_max + 1 <= 16
# GEMM1 tile：M=tmc(≤bm, ≥16)、N=tnc(≤bn_eff, ≥32)、K=dim(64/128 ≥32)——QK^T（b_transpose=True）
# GEMM2 tile：M=tmc、N=dim、K=tnc(≥32)——PV（initC=True 每块覆写）
```

**目标域 config 核验表**（设计默认 bn_eff=256/bm=64 下的有效值）：

| workload | q_shape [B,S,H,D] | wl | num_q_blocks | loop_count 范围 | L_max | flag ids | 尾块 |
|---|---|---|---|---|---|---|---|
| smoke | [2,512,8,64] | −1 | 8 | 1..2 | 2 | 3 | 无（512%64=0） |
| 8b-short | [4,512,32,128] | 256 | 8 | 1..2 | 2 | 3 | 无 |
| 8b-long | [2,2048,32,128] | 1024 | 32 | 1..5 | 5 | 6 | 无 |
| 70b-short | [2,512,64,128] | 256 | 8 | 1..2 | 2 | 3 | 无 |
| 70b-long | [1,2048,64,128] | 1024 | 32 | 1..5 | 5 | 6 | 无 |

（manifest 5 workload 全部整除、零尾块零尾带——尾块机制仅服务契约测试域。**带状裁剪的执行块数红利**：8b-long **7680 块 vs 全 causal 9216 块 = −16.7%**（按 §0.6 E-band 源式 `loop_count=k_end−k_start` 逐 bx 复算：Σloop_count=120/（head,batch）× H·B=64 = 7680；内段全有效块 Σmax(loop_count−2,0)=60×64=3840、边界掩码块 3840）；**wl=256 短序列块粒度下带未越界，执行块数与全 causal 同（0 减免）**；首尾任务 loop_count 小、中段大——**轮转任务分配使每核 bx 均匀覆盖**，负载均衡见 §5.5。）

### 5.3 约束分析

- **对齐约束**: `bn_eff` 与 `dim` 为 16 倍数（守卫公式 ceil16 保证；dim ∈ {64,128}）；fp16/bf16 尾轴 256·2B = 512B ≥ 32B ✓；分形 M/N ≥ 16、K ≥ 32 由 E7 钳位保证。
- **L1 容量**: l1_a + l1_b = 96KB ≤ 512KB ✓（`CONST-capacity-910B2C`）。
- **L0C 容量**: l0_c [64,256] f32 = 64KB ≤ 128KB ✓。
- **UB 容量**: ~154KB ≤ 192KB per AIV ✓（§4.5）。
- **flag 预算**: `L_max + 1 ≤ 16` ✓（T.set_flag.md §2.1 event_id 0–15；E-band 守卫公式强制 + trace-time assert）。
- **带状范围**：`loop_count ≥ 0`（整块出窗时 ≤ 0 → 循环体不执行）；`k_start ≤ k_end` 由源式保证（`max(0,bx·bm−wl) ≤ (bx+1)bm`）。

### 5.4 注意事项

- **非整除 / 小尾块**（契约域）：tmc/tnc 分形下限钳位（M/N≥16、K≥32）+ `has_band` trace-time 门控（尾带 OOB 掩码 + `vselect` 回退）+ 行截断写出（E7 全套，§6.4）；manifest 域零触发。
- **带状边界块**：每任务至多 2 个边界块（左窗 + 因果/右窗）走掩码链；中段全有效块经 E-bnd 零掩码——掩码开销为结构性常数（≤2 块 × ~6 op/任务）。
- **bn_eff 守卫**：`wl` 越大要求 `bn_eff` 越大才能压住 `L_max`（E-band 公式）；manifest 全域 bn_eff=256 合规。
- **bm 上界**：L0C 公式 bm·max(bn_eff,dim)·4 ≤ 128KB → bm ≤ 128@bn=256；UB 公式约 (0.6·bm + 116)·1024 ≤ 192KB → **bm ≤ 118**——UB 与 L0C 均不先绑定（bm=64 取安全值）。
- **prescale ≠ 1**：本算子 `prescale = dim^{−1/2} ≠ 1` 恒成立（M7 在本域无触发）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（依据 docs/开发指南.md §3.3）。

- **物理核数（要素②，实查）**: **20**。查询代码与返回值记录：

```python
from tilelang.utils.npu_utils import NPUUtils
u = NPUUtils.get()
print(u.get_arch(), u.get_aicore_num(), u.get_aivector_core_num())
# 2026-10-16 09:09 实查输出：Ascend910B3 20 40
```

（混合算子是 Cube+Vector，直接使用 `get_aicore_num()` 返回值 **20**——**非 9362 的 20 是巧合、非 910B2C 的 24**；本容器 `get_device_name` = `Ascend910B3`，`npui-smi` 显示 Name `910B3`。Cube/Vector 混合核数按 Cube 计数 20；双 AIV 由 `subid` 分片。）

- **逻辑核数（要素①）**: `num_logical = ceildiv(S, bm) × heads × batch`（bm=64 设计默认）：

| workload | q_shape [B,S,H,D] | num_q_blocks | num_logical | 与 20 核关系 |
|---|---|---|---|---|
| smoke | [2,512,8,64] | 8 | 8×8×2 = **128** | 6.4×（轻度欠载边缘） |
| 8b-short | [4,512,32,128] | 8 | 8×32×4 = **1024** | 51.2×（饱和） |
| 8b-long | [2,2048,32,128] | 32 | 32×32×2 = **2048** | 102.4×（饱和） |
| 70b-short | [2,512,64,128] | 8 | 8×64×2 = **1024** | 51.2×（饱和） |
| 70b-long | [1,2048,64,128] | 32 | 32×64×1 = **2048** | 102.4×（饱和） |

- **规模判定（要素③）**: **极大规模**（4/5 workload 逻辑任务数 1024–2048 » 20，无法通过调整分块缩减到核数量级——bm 上调受 UB/L0C 双上限约束且减少任务数与均衡性冲突）；smoke 128 任务为轻度欠载（128/20 = 6.4 任务/核，核心全忙）。**无「无法对齐兜底」情形**（逻辑核数非被钉为 2 的幂）。
- **分核方案**: **固定启动内核数 = 物理核数 20，核内 `T.serial` 串行处理多个逻辑任务**：`T.Kernel(20, is_npu=True)`，每核 `num_local_tasks = T.ceildiv(num_logical − kernel_id, 20)` 个任务、任务解码 `cid = T.min(task_id·20 + kernel_id, num_logical − 1)`（轮转——**带状 loop_count 随 bx 先增后减**〔wl<0 时单调增〕，轮转使每核 bx 均匀覆盖 `[0, num_q_blocks)`，负载均衡）。循环边界为 kernel_id 派生的 PrimExpr 表达式（进入循环前一次计算、循环内不变——D3 实证可 lower；开发指南 §3.3 模板同形）。**smoke 欠载处置**：128 任务下 20 核全忙（8 核 7 任务 / 12 核 6 任务，不均衡比 1.17×）——Stage 4 可按 shape 分派 bm=32（256 任务）或依赖两相位单任务流水性吸收，不在 Stage 1 特化。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务维（per 核） | persistent 串行 | `for task_id in T.serial(num_local_tasks)` | 极大规模分核（§5.5）；runtime PrimExpr 边界（kernel_id 派生，D3 实证） |
| 带内块 pass-1 维（Cube） | 串行大循环 | `for l in T.serial(loop_count)` | 两相位全 S（无等待直跑）；`loop_count = k_end − k_start`（E-band 带状范围） |
| 带内块 pass-2 维（Cube） | 串行大循环 | `for l in T.serial(loop_count)` | 两相位全 PV（逐块等 P） |
| 带内块 pass-1/pass-2 维（Vector） | 串行大循环 ×2 | `for l in T.serial(loop_count)` | softmax 链 + 掩码 / 延迟累加 |
| 元素级 | **向量化（无循环）** | v-prefix 链 + `T.reduce` | §1.6.2 全覆盖（15 项 ✅）；无逐元素标量循环 |

### 6.2 循环伪代码

```python
with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
    # 每核一次：ub_diff = arange(col) - arange(row)（[half,bn_eff] f32，任务无关）
    with T.Scope("Cube"):
        for task_id in T.serial(num_local_tasks):
            # cid → (bx,by,bz)；k_end/k_start/loop_count/tmc/tnc 计算（E-band/E7）
            if task_id > 0:
                with T.rs("PIPE_MTE2"): T.sync_block_wait(FLAG_TASKDONE)   # 单槽跨任务 WAR
            with T.rs("PIPE_MTE2"): T.copy(q[...], l1_a[...])               # Q-hoist
            for l in T.serial(loop_count):        # pass-1（§0.6 E2 Cube 流 + E-band）
                ...
            for l in T.serial(loop_count):        # pass-2
                ...
    with T.Scope("Vector"):
        for task_id in T.serial(num_local_tasks):
            # 同款任务解码；real_m 自适应半分；L_lo_full/L_hi_full per-AIV 阈值
            for l in T.serial(loop_count):        # pass-1（掩码 E-mask + softmax 链）
                ...
            for l in T.serial(loop_count):        # pass-2
                ...
            # epilogue + set FLAG_TASKDONE（§0.6 E2 Vector 流）
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**（Expert 手动两相位，§0.5/§2）。流水机制 = 两相位结构自带的跨引擎重叠：① Cube pass-1 的 MTE2（K 装载）/MTE1（喂 L0）/C（gemm）/FIX（写 ws_s）四通道硬件排队自然重叠（无软件等待插入）；② Cube pass-2(l) 与 Vector pass-1(l'>l) 经 per-带内块 flag 重叠推进；③ Vector pass-1 的 MTE2（读 ws_s）/V（掩码+softmax 链）/MTE3（写 ws_p）三通道重叠。**槽位安全性论证**：单槽 + 带内块 id 严格交替 + TASKDONE 边界排空——每个 flag id 任一时刻至多一个未决事件，无 WAR 窗口；ws 生命段（pass-1 写 ws_s[l] → Vector 读 → Vector 写 ws_p[l] → Cube 读 → Cube 写 ws_o[l] → Vector 读）全部由同一 id 的三次握手定序。

### 6.4 尾块处理

**manifest 域零触发**（S%bm=0、S%bn_eff=0）；契约域全套机制（E7）：

1. **q 尾块**：`tail_m_real = min(bm, S−s_lo)`；`tmc = max(16, min(bm, ceil16(tail_m_real)))`——gemm 计算宽 tmc（垃圾行 [tail_m_real, tmc) 为 l1_a 残留），Vector 自适应半分只覆盖真实行（`real_m0 = (tail_m_real+1)//2`），输出行截断写出。
2. **kv 尾块无带域**（`tn_real ≥ 32 且 16 | tn_real`）：`tnc = tn_real`。
3. **kv 尾带域**（`tn_real < 32 或 16 ∤ tn_real`，`has_band` trace-time 门控）：`tnc = max(32, min(bn_eff, ceil16(tn_real)))`；Vector 对 `j ≥ tn_real` 列以 OOB 掩码（**此处回退 `vselect` 值替换形态**——加法掩码无法清 NaN，PL-1.11 P8）强制 P=0；**同时 `l1_b`(V) 每核一次 `zbuf` 零初始化**（`[bn_eff,dim]` 零张量内部参数，成本 ~0.06%——PL-1.11：`gemm2` K 维 fractal band 的 `P带(0) × V残留(NaN) = NaN` 整列污染；「V残留任意值 × 0 = 0」假设对 NaN 位**不成立**，须源头零化）。**经 zbuf 零初始化后**，gemm2 K 带 `P带(0) × V残留(0) = 0`——该结论由零初始化保证，非依赖 stale V 值。
4. **全 OOB kv 块**：`loop_count` 由 `k_start/k_end` 保证不越界（E-band）；带外块不计算。
5. **GQA**：`kv_head = by // groups` 切片寻址。
6. **左/右窗不相等**：`wl<0 → L_lo_full=0`（左段 trace-time 剔除）；`is_causal=False ∧ wr<0 →` 无上界（右段剔除）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 手动同步（Expert）——per-带内块 `T.sync_block_set/wait` + `T.rs("PIPE_*")` 通道标注；无 `T.Pipelined` / 自动同步依赖。

### 7.2 同步点说明（flag id 分配表 + 逐点理由）

**flag id 分配**（设计默认 L_max ≤ 5；**总 id 数 = L_max + 1 ≤ 16 = 事件预算 0–15**——T.set_flag.md §2.1；E-band 守卫公式强制 §5.2）：

| flag 族 | id | 方向 | 设置点 | 等待点 | 语义 |
|---------|---|------|--------|--------|------|
| 带内块三次握手 | l ∈ [0, L_max) | Cube↔Vector（三次转向） | ① Cube pass-1 FIX 段 `set(l)`〔S 就绪〕② Vector pass-1 MTE3 段 `set(l)`〔P 就绪〕③ Cube pass-2 FIX 段 `set(l)`〔O 就绪〕 | ① Vector pass-1 MTE2 段 `wait(l)` ② Cube pass-2 MTE2 段 `wait(l)` ③ Vector pass-2 MTE2 段 `wait(l)` | 每 id 严格三次握手、per-id 交替（任一时刻 ≤1 未决事件） |
| FLAG_TASKDONE | L_max | Vector→Cube | Vector epilogue 写出后 set（双 AIV 各 set 同一 id） | Cube 下一任务 Q-hoist 前 wait（`if task_id > 0`） | 单槽跨任务 WAR 保护 |

**逐点设计依据**：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| Cube pass-1 FIX 后 `set(l)` | `T.sync_block_set`（PIPE_FIX 段内） | 通知 Vector S(l) 可读（L0C→GM copy 完成） |
| Vector pass-1 MTE2 前 `wait(l)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 S 数据落 GM |
| Vector pass-1 MTE3 后 `set(l)` | `T.sync_block_set`（PIPE_MTE3 段内） | 通知 Cube P(l) 可读（含「S 已消费」传递语义） |
| Cube pass-2 MTE2 前 `wait(l)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 P 数据落 GM |
| Cube pass-2 FIX 后 `set(l)` | `T.sync_block_set`（PIPE_FIX 段内） | 通知 Vector O(l) 可读（含「P 已消费」传递语义） |
| Vector pass-2 MTE2 前 `wait(l)` | `T.sync_block_wait`（PIPE_MTE2 段内） | 等 O 部分和落 GM |
| Vector epilogue 后 `set(FLAG_TASKDONE)` | `T.sync_block_set` | 任务排空握手 |
| Cube 下一任务 Q-hoist 前 `wait(FLAG_TASKDONE)` | `T.sync_block_wait`（`if task_id > 0`） | 防新任务 pass-1 覆写未消费 ws |

**跨任务 id 复用安全**：不同任务的 `loop_count` 可不同，但 Cube/Vector 两侧用同一 task 解码得同一 `loop_count` → 同一 l 序列；前一任务对 id l 的三次握手在 TASKDONE 排空前已全部闭合（同 MHA 单槽论证）。

**2-AIV 聚合语义（R-1 风险项）**：Vector 段代码在每核两个 AIV（subid 0/1）上各执行一份，二者写各自半区的 P 后 set 同一 flag id——Cube 单次 wait 的聚合语义（「两 AIV 均完成」才放行）依 MHA 先例（29/29 + 11 轮生产）成立；若 Stage 3 探针证伪，回退 = FLAG_TASKDONE 按 subid 拆双 id + Cube 双 wait（id 总数 L_max+2 ≤ 16 仍合规）。

**跨核同步**：无（任务独立、q/k/v 只读共享、output/lse 切片不相交）；本仓同步原语均为**核内**语义（T.set_flag.md §1「同一个 Cube/Vector 核内的不同通道之间」/ T.sync_block_set.md §1「同一 block 中的其他执行单元」——适用范围条款亲自核对）。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
}
```
（防编译器重排手动两相位流水与显式 workspace——MHA/v11nt/highperf 先例；CUDA 的 FAST_MATH/compile_flags 不迁移，§0.5。）

### 7.4 CV 融合设计（MixCV）说明

同一 kernel 内 `T.Scope("Cube")`（两相位 GEMM 大循环）与 `T.Scope("Vector")`（掩码/softmax/累加/epilogue）顺序书写、硬件上并行执行于同一 AI Core 的双引擎，经 per-带内块 flag 协同——MixCV 结构（AGENTS.md 判据同款）。**编程模式为 Expert（P1 探针 + CG-2026-0010 证据化结论，§2）**——Developer auto CV-split 对本结构不可用。tilelang-mixcv-skill 的核心关注点（sync_block_set/wait、Scope 分工、PIPE_FIX/MTE2/MTE3 通道）在本设计 §0.6/§7.2 全量落地；Stage 3 实现时可加载 mixcv skill 复核同步细节。
---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（**优先移植源仓测试 `ref_program`**，补 lse），**不得复刻 §0.6 的两相位 NPU 算法**——保证验证独立性。

```python
def golden_gqa_sw_fwd(q, k, v, is_causal, window_size_left, window_size_right):
    """Sliding-window GQA forward reference (BSHD, batched materialize softmax).

    q:    [B, S, H, D]        fp16/bf16 (BSHD)
    k/v:  [B, S, H_kv, D]     fp16/bf16 (BSHD)
    Returns (output: q shape/dtype, lse: [B, H, S] fp32).

    Independence: no online softmax, no band trimming, no e-domain reordering --
    per (b, h) materialized batched softmax in fp32 with the source's exact
    three-condition mask, P quantized at the source acc_s_cast point.
    """
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    groups = H // Hkv
    scale = D ** -0.5

    k_exp = k.repeat_interleave(groups, dim=2).float()   # [B, S, H, D]
    v_exp = v.repeat_interleave(groups, dim=2).float()

    scores = torch.matmul(q.float().transpose(1, 2),
                          k_exp.transpose(1, 2).transpose(-2, -1)) * scale  # [B,H,S,S]

    q_pos = torch.arange(S, device=q.device).unsqueeze(1)   # [S,1]
    k_pos = torch.arange(S, device=q.device).unsqueeze(0)   # [1,S]
    mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
    if is_causal:
        mask = mask | (q_pos < k_pos)
    if window_size_left >= 0:
        mask = mask | (k_pos < q_pos - window_size_left)
    if window_size_right >= 0:
        mask = mask | (k_pos > q_pos + window_size_right)

    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1) / math.log(2.0)          # log2 域（D7 校准）
    probs = torch.softmax(scores, dim=-1)
    p_q = probs.to(q.dtype).float()                                # P 量化到输入 dtype（源 acc_s_cast 点）
    output = torch.matmul(p_q, v_exp.transpose(1, 2)).to(q.dtype)  # [B,H,S,D]
    return output.transpose(1, 2), lse                             # [B,S,H,D], [B,H,S]
```

（**golden 独立性核对**：本函数无 online softmax、无带状裁剪、无 e 域换底、无 f16 S/O 物化、无算术惩罚掩码——与 §0.6 的 NPU 算法结构正交；与源仓测试 `ref_program` 语义一致，仅补 lse。机器核对见 `verify_equiv.py` 的 E2E-golden 行。）

### 8.2 精度标准与 L0 门槛测试计划

**精度门**（承接 MHA 谱系 D6/D8 校准的双门 + tier-3 分类；**Op 测试门为 `atol=1e-2, rtol=1e-2`**——源仓 `test_gqa_sliding_window_fwd_op` 冻结值）：

| dtype | 输出 atol/rtol（tier-1） | tier-2（2-ulp 回退） | lse atol/rtol |
|-------|--------------------------|---------------------|----------------|
| float16 | 5e-3 / 1e-5（Op 门 1e-2/1e-2 更宽） | +2·2^-10 相对余量 | 1e-3 / 1e-3 |
| bfloat16 | 同上 | +2·2^-7 相对余量 | 同上 |

tier-3 噪声分类（源 P 量化在早 causal 行的放大域：率 ≤ 0.01%、包络 2·flip_rel·v_absmax、lse 门不受扰）——`verify_equiv.py` E2E-bf16-golden 实测 tier-1 率 4.3e-4、tier-2 率 0（属该分类域，非缺陷）。

**分层测试策略**（回归入口 = `python _gqa_sw_fwd_kernel.py --level all`；L0/L1 阻塞、L2/Boundary 告警不阻塞）：

| 层 | 阻塞 | 用例 | 覆盖目标 |
|---|---|---|---|
| L0 | ✅ | L0-1 smoke causal wl=−1 fp16（seed=0）；L0-2 smoke bf16（**R-2 两相位 bf16 探针前置首验**）；L0-3 wl=256 fp16 long；L0-4 契约校验（shape/dtype/lse 域 + golden 交叉 + SDPA 交叉〔wl=0 域〕） | 门禁最小集 + 带状主 trace |
| L1 | ✅ | manifest 全 sweep：8b/70b short（wl=256）/long（wl=1024）× fp16/bf16 = 8 例 | 目标域全量 |
| L2 | ⚠️ | 源仓测试域：wl=128 causal；非 causal wl=−1；**对称带 wl=64,wr=64 非 causal**；wl=1（S=128）；GQA h8/hkv8（无分组）、h16/hkv1（G=16）；非 2 幂 S=384/768；**右窗 wr=64 非 causal**；**wl=0（仅自注意力）**；S=2048 wl=512 | 四态掩码路径全覆盖 |
| Boundary | ⚠️ | S=520（tmc 钳位 + 尾带）；S=100（nk_total 单/双块）；zeros（lse 精确 = log2(band_len)）；wl>S（等价 wl=−1）；极长 S=4096 wl=1024 | 边界 + 病态 |

**L0 门槛计划**（Stage 3 首验顺序）：① **L0-1（fp16 smoke wl=−1）→ 验证两相位主 trace + 带状范围（wl=−1 退化为 full causal）**；② **L0-3（fp16 wl=256）→ 验证带状裁剪 + 双侧掩码（本算子最高新颖度）**；③ L0-2（bf16 smoke）→ R-2 探针；④ 契约校验。任一失败按 §9.2 风险表的回退路径处置后重验。

---

## 9. 风险点与注意事项

### 9.1 已知约束（技术约束检测结论汇总）

1. **一维 Kernel**（三维 grid 禁止）→ E1 persistent 任务解码。
2. **L0C(cc)=128KB / L1(cbuf)=512KB / UB=192KB per AIV**（三项实测上限，`CONST-capacity-910B2C`）→ §4.3/§4.5 预算内设计。
3. **L0C 出数 dst 仅 GM**（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1）→ S/O 跨引擎必经 GM ws（E2，CG-2026-0005）。
4. **`store_fixpipe` f32→f16 为正确数值转换**（PL-1.9 probe 4.7e-4 = 1 ulp）→ M8/M9 采纳依据。
5. **活跃源 `T.transpose` epilogue 毒化 2.6×**（VP-D6）→ transpose-free lse（E6）。
6. **`load_nd2nz` 对 BSHD 跨头切片静默平铺误读**（D1，CG-2026-0004）→ 全部 GM→L1 用 slice-form `T.copy`。
7. **向量算子发射开销 ~0.5µs/op**（`CONST-vector-launch-overhead`，910B2C 绑定）→ bn=256 宽块主 trace。
8. **flag event_id ∈ 0–15**（T.set_flag.md §2.1）→ bn_eff 守卫公式（E-band）+ 带内块 id 严格交替（E2）。
9. **parser 不折叠 if / 条件 alloc 非法 / vbrc 标量须 let-bound / vcmp 标量拒 tir.Cast（int16 阈值）/ vtanh 须 fp32 操作数 / `T.reduce` 无 clear 为静默错误**（D4/D5/T.reduce.md §2.3）→ §3.2/§3.3 全项落实。
10. **`sync_block_wait` 阻塞整条发射流**（PL-1.9-hardlimits）→ 两相位结构本身就是该约束的规避（pass-1 零等待）。
11. **`vcmp` int16 全形态标量化**（PL-1.11；本容器 P2 探针 ALT 链复现 2 条 `Op 'hivm.hir.vcmp' … scalar instruction` warning，主选算术惩罚链零 warning）→ 掩码用算术惩罚式（E-mask）。
12. **fractal band `0×NaN=NaN` 整列污染**（PL-1.11 P8）→ `has_band` 尾带用 `vselect` 值替换 + `l1_b`(V) `zbuf` 零初始化（E7/§6.4，manifest 域零触发）。
13. **Developer auto CV-split 不支持 persistent+gemm+v-prefix**（CG-2026-0010 / PL-1.16；本容器 P1 探针复现〔编译期 `map::at`〕）→ Expert 双 Scope（§2）。

### 9.2 风险清单与常见错误

| # | 风险 | 等级 | 触发场景 | 缓解 / 回退 |
|---|------|------|----------|------------|
| R-1 | 2-AIV flag 聚合语义（双 AIV set 同 id、Cube 单 wait） | 中 | 跨任务槽位复用时 TASKDONE 单 wait 可能提前放行 | MHA 先例双生产；L0-1 含多任务复现；回退 = TASKDONE 按 subid 拆双 id + Cube 双 wait（id 预算 L_max+2 ≤ 16 ✓） |
| R-2 | 两相位 bf16 端到端（bf16 L1 gemm + f32 ws_s + bf16 ws_p）无本工具链先例 | 中 | L0-2 bf16 smoke | E5 保守承载（f32 ws_s/ws_o）；回退 = ws 全 f32 + 装载点 vcast（结构不变） |
| **R-3** | **双侧掩码公式正确性（K_up/K_lo 符号 + L_lo_full/L_hi_full 分界方向）** | **高** | 任何 wl≥0 用例 | 代数重推 + **双端点代入纪律**（K_up/K_lo 由 `q_pos=s_lo+row0+i`、`kv_pos=n·bn+j` 推导；见 §0.6 E-mask/E-bnd）；**P2 探针双侧链各 7 配置 bit-exact 实测**（主选算术惩罚链 + vcmp 备选链）；L0-3 + L2 四态掩码全覆盖；回退 = 单块全掩码（不裁剪，正确但慢） |
| R-4 | 带状范围 PrimExpr 边界（k_start/k_end/loop_count/L_*_full/tmc/tnc）下 lower 失败 | 低 | 首编译 | D3/E3 生产实证同族形态；K_blk 双 lane 端点数值例；失败时以标量三元式（D4 合法形态）改写 |
| R-5 | UB 预算 154KB 贴近 192KB（bn_eff 上调 / dim>128 / has_band 加 ub_neg 时溢出） | 中 | bn_eff > 256 或 has_band trace | §4.5 口径公式前置校验（工厂期）；溢出时 UB diet（`ub_pen` 与 `ub_f32_N` 原位复用 −32KB；`ub_diff` f16 化）或 bn_eff 降 128 |
| R-6 | S/O f16 物化精度 | 低 | L1 fp16 全 sweep | PL-1.9 + v11nt fa-tuned 实证（tier-1 0 flips）+ §8.2 双门 + tier-3 分类兜底；超限时 ws_s/ws_o 升 f32（带宽代价，正确性优先） |
| R-7 | flag 预算超限（L_max + 1 > 16） | 低 | wl 很大 + bn_eff 过小 | E-band 守卫公式（工厂期强制）+ trace-time assert；超限时 bn_eff += 16 重算 |
| R-8 | lse [B,H,S,1] 视图 + host reshape 契约 | 低 | 契约校验 | MHA 先例生产同款（wrapped reshape 零拷贝）；register_fake 形状一致 |
| R-9 | 尾带机制（has_band trace）遗漏域 | 中 | L2/Boundary 非整除用例 | §6.4 数值例全覆盖；尾带回退 `vselect` 值替换 + **`l1_b`(V) `zbuf` 零初始化**（加法掩码无法清 NaN 且 `0×NaN=NaN`，PL-1.11 P8） |
| R-10 | 目标域性能预期的不确定常数（910B3 上的向量发射率 / fabric 有效带宽未实测） | 中 | Stage 4 首采 | §1.6.0「设计期估算」行按保守/乐观双口径给出；Stage 4 Phase 1 首采 5 case 后校准；结构性下界（两相位 vs 单遍 2.31×，MHA 同族实测）不受该常数影响 |
| R-11 | 掩码式选择（算术惩罚 vs vselect）的判定裕度依赖未实证常数 | 低 | Stage 4 | §1.6.3 实验裁决计划（主选算术惩罚 + 备选 vselect + 阈值 0.95）；P2 探针已标定主选算术惩罚链设备 7/7 bit-exact 可运行（零 warning），备选 vcmp 链亦 7/7（含标量化 warning） |
| R-12 | smoke 欠载（128 任务 / 20 核，1.17× 不均衡） | 低 | smoke case | 核心全忙、两相位单任务流水吸收；Stage 4 按 shape 分派 bm=32 候选 |

### 9.3 特殊场景处理

- **非整除分块**：§6.4 尾块全套（tmc/tnc 钳位 + 尾带掩码 + 行截断）。
- **极小 shape**（S=16/32）：`loop_count=1` 单块域——两相位退化为 pass-1/pass-2 各一块（正确性同构，性能非目标）。
- **混合精度**：fp16/bf16 双 trace 同构（E5）；ws dtype 分 dtype 承载。
- **GQA**：`kv_head = by // groups`；heads_kv < heads 时 k/v 切片跨头共享只读。
- **四态掩码**：causal+left / causal only / left only（+right）/ none——trace 家族分派（§3.3），各 trace 的 `L_lo_full`/`L_hi_full` 段按需剔除。**`is_causal ∧ wr≥0` 组合不承诺**（源 kernel 在该分支忽略 right window，见 §0.1 角点说明；manifest/测试域无此组合）。
- **wl=0**（仅自注意力）：band = 单列，`loop_count` 可能 1，`L_lo_full`/`L_hi_full` 使掩码段覆盖整块（正确的保守退化）。

### 9.4 编译能力缺口登记（AGENTS.md「Compiler Capability Gap Rule」）

本任务探针 P1 复现 **CG-2026-0010** 同缺口（`layer: codegen / BishengIR`；capability：Developer auto CV-split 对 persistent + `T.gemm` + v-prefix 混排不支持）。**本容器失败层为编译期**（BishengIR `std::out_of_range` / `map::at`），与既有条目的运行期崩溃（`Illegal instruction … unaligned UUB addresses`）**层不同**——属该能力缺口在新工具链（tilelang `0.1.2+cf6f999…` / CANN 9.0.0 / 910B3）下的新触发层，建议在 `.agents/evolution/capability-gaps.md` CG-2026-0010 条目下 `occurrences += 1` 并补「编译期触发」注记（具体缺失能力 = Developer 模式 CV-split pass 对 MIX 结构的 lowering/allocate 支持；证据 = `history_version/design_probe_devmode_mix.py` variant A/B/C 对照）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_query_attention_sliding_window/_gqa_sw_fwd_kernel/
├── DESIGN.md                     # 本设计文档
├── verify_equiv.py               # §1.6.1 采纳项等价性机器验证（torch CPU，11/11 EQUIV_PASS）
├── _gqa_sw_fwd_kernel.py         # 算子实现 + 分层测试驱动（Stage 3 生成）
├── RETROSPECTIVE.md              # 任务复盘（追加）
└── history_version/
    ├── design_probe_devmode_mix.py   # P1 探针（Developer vs Expert MIX，§2 证据）
    └── design_probe_band_mask.py     # P2 探针 rev-1（主选算术惩罚链 7/7 bit-exact + 备选 vcmp 链 warning，§1.6.3 证据）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件） | 设计文档 |
| `verify_equiv.py` | ✅ 已完成（§1.6.1 结果表联动，11/11 EQUIV_PASS） | 数学等价优化采纳项机器验证 |
| `history_version/design_probe_devmode_mix.py` | ✅ 已完成（P1：Developer BLOCKED / Expert PASS） | 编程模式选型证据 |
| `history_version/design_probe_band_mask.py` | ✅ 已完成（P2 rev-1：主选算术惩罚链 7/7 bit-exact 零 warning；备选 vcmp/vselect 链 7/7 含标量化 warning） | 掩码式选择证据 |
| `_gqa_sw_fwd_kernel.py` | ⬜ 待实现（Stage 3） | 两相位 kernel + golden + L0/L1/L2/Boundary 驱动（`--level all`） |
| TileOPs 侧 wrapper | 不改动 | `grouped_query_attention_sliding_window.py` 契约零变化（E6 闭包兼容） |

### 10.3 命名规范

- 项目目录: `grouped_query_attention_sliding_window`；算子目录: `_gqa_sw_fwd_kernel`；实现文件: `_gqa_sw_fwd_kernel.py`——与 conductor 指定目录一致。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 探针（P1/P2）+ `verify_equiv.py`
2. ⬜ Golden 函数 + 精度门（§8.1/§8.2，源仓 `ref_program` 移植 + lse）
3. ⬜ 工厂层（trace 家族分派 + bn_eff/L_max 守卫 + ws 闭包）+ `_builder_2phase(mask 域 = causal+left)` 主 trace（fp16）
4. ⬜ **L0-1 首验**（fp16 smoke wl=−1）→ **L0-3**（fp16 wl=256 带状主 trace）→ L0-2（bf16，R-2）→ 契约校验
5. ⬜ 其余 mask 域 trace（causal only / left only / right / none）+ has_band 变体 + L2/Boundary 全套
6. ⬜ `--level all` 全绿 → Stage 2 检视入口

---

## 11. 性能预期与度量计划（manifest 滑动窗口域 · Stage 4 度量锚点）

> 本节为 Stage 4 调优提供预期锚点与度量计划（best_effort，无硬性数值目标）；预期基于结构证据外推（MHA 同族实测 + §1.6.0 设计期估算），全部待实测校准。

**结构证据锚点**（MHA 同族同机同口径 msprof op Task Duration）：
- 两相位 vs 单遍链（fa 域、同 workload 族、调优终值对比）：**2.31×**（fa4096: 98.05 vs 226.72µs）；结构性下界，不依赖目标域常数。
- 带状裁剪执行块数红利：8b-long **7680 块 vs 全 causal 9216 块（−16.7%）**；FLOPs 51.6 vs 68.8 GFLOP（−25%）。
- 8b-short 带状更短（wl=256 → loop_count ≤ 2），固定开销占比上升——短 workload 预期为**弱侧**（MHA PL-1.7 记录：两相位短 KV 相对单遍链有 1.31× 回退，属该形态预期画像）。

**预期区间**（保守 = 仅套用 2.31× 结构下界；乐观 = 按发射/流量双口径外推——910B3 常数未实测，R-10）：

| workload | 带状 FLOPs | 全 causal FLOPs | 保守预期结构下界 | 乐观预期（§1.6.0 估算下界） |
|---|---|---|---|---|
| 8b-short（wl=256） | 4·4·32·98432·128 = 6.45 GFLOP | 8.59 GFLOP | ~1.6ms（×2.31 外推自 developer 基线） | ~0.3–0.7ms（**执行块数无减免**——wl=256 下带未越界；短 S 摊销不足，取下限） |
| 8b-long（wl=1024） | 51.6 GFLOP | 68.8 GFLOP | ~2.5ms | **~1.0–2.5ms**（§1.6.0 D-2 双口径收敛区间：MHA 锚点 0.99ms ↔ 朴素发射 1.66ms） |
| 70b-short（wl=256） | 6.45 GFLOP | 8.59 GFLOP | 同 8b-short | 同 8b-short |
| 70b-long（wl=1024） | 51.6 GFLOP | 68.8 GFLOP | 同 8b-long | 同 8b-long |
| smoke（wl=−1） | 0.27 GFLOP | 0.27 GFLOP | ~0.12ms | ~0.04–0.1ms（轻度欠载） |

**Stage 4 度量计划**：① Phase 1 首采 5 case fp16（msprof op Task Duration，launch-count=20/warm-up=5/median，captured op = 新 kernel 的 mix_aic 名）；② 对照上表校准 910B3 的向量发射率与 fabric 有效常数；③ **实验裁决**（§1.6.3）：代表 shape 实测主选（算术惩罚式）vs 备选（vselect），判定阈值 = 备选 < 主选 × 0.95 才翻转；④ 调优候选优先序：bm ∈ {64, 80} × bn_eff ∈ {128, 256} 扫描（UB/L0C 公式约束内）→ f16 softmax 链（#11 defer 项）→ 双槽 id-offset 变体（R-7，L_max ≤ 7 域）→ shape 分派（smoke bm=32）；⑤ 回归纪律：`--level all` 全绿为每轮前置，滑动窗口域互不回退 ≤3%。

---

## 附：声明

- 本设计为 **revision（revision_index=1，mode=revision）**——相对上一版 `history_version/design_v0.md` 的关键调整见下「修订说明」。
- 迁移决策（§0）驱动全部后续章节：§1–§7 基于 §0.6 重设计后的 NPU 算法展开，未照抄源方案（源单遍 per-block 链 → 两相位；三维 grid → 一维 persistent；fragment 驻留 → UB + ws 物化；源掩码预填 → 双侧算术惩罚）。
- 源算法全部已识别优化手段（§0.4 OP1–OP15）在 §0.5 逐项处置、在 §1.6.0 逐项承接，无静默丢弃。
- 算法调研（Phase R）先于算法级优化（Phase 2）：§1.6.0 选定算法 = §1.4 算法描述 = §1.6.1–§1.6.3 分析对象；源算法仅作为基线候选之一参与 R1–R4 对比。

### 修订说明（revision_index=1）

**相对上一版（design_v0）的关键调整**：

1. **块数 / fabric 算术订正（阻塞项 1）**——§1.6.0 R3/R4、§1.6.1 M4、§5.2/§5.4、§11 及任何引用块数处统一改为**执行块数口径**：8b-long（wl=1024）**执行块数 7680**（对全 causal 9216 为 **−16.7%**），按 §0.6 E-band 源式 `k_end=ceildiv(min(S,(bx+1)bm),bn_eff)`、`k_start=max(0,bx·bm−wl)//bn_eff`、`loop_count=k_end−k_start` **逐 bx 复算**：Σloop_count = 120/（head,batch）×（H·B=64）= **7680**（旧值 3584 = 每任务剔除 2 边界块的**内段块数** Σ(loop_count−2)=56×64，被误当执行块数并与全 causal 相减）。**内段全有效块单列 3840**（Σmax(loop_count−2,0)=60×64）；**wl=256 短序列执行块数与全 causal 相同（0 减免）**。Fabric 逐项复算（表达式 + 数值）：KV tile 流式重读 = 7680 × (K+V tile 各 [256,128]f16 64KB) = 7680 × 128KB = **0.94GB**；ws 往返（写+读）= 7680 × (S 64KB + P 64KB + O f32 64KB) = 7680 × 192KB = **1.41GB**（#1）/ O f16 32KB → 7680 × 160KB = **1.17GB**（#2）；合计 #1 ≈ **2.35GB**、#2 ≈ **2.11GB**（差 0.23GB ≈ 10%）。D-2 发射项按 7680 块重算（7680×~17 ≈ 132.6k op ÷40 AIV ≈ 3315 op × 0.5µs ≈ **1660µs**）并与 MHA 实测锚点（1191.6µs × 7680/9216 ≈ 0.99ms）交叉核对。**FLOPs 口径（51.6 vs 68.8 GFLOP，−25%）复算正确，保留。**
2. **P2 探针修复 + 重跑（阻塞项 2）**——`history_version/design_probe_band_mask.py`：阈值由裸 int 改为 **let-bound / kernel 派生 PrimExpr**（`ku = kid*0 + const`；修正 v1 trace 期 `'int' object has no attribute 'buffer'`）；**补入被采纳的主选「双侧算术惩罚链」**（`vsub/vmax/vmin/vmax` + `vmul(pen,−1e38)` + `vadd`）；新增输入张量 S（纯输出 kernel 在本设备 abort）。重跑结果：**主选算术惩罚链 7/7 配置 bit-exact PASS 且零标量化 warning**；备选 `vcmp/vand/vselect` 链 7/7 bit-exact 但复现 2 条 `Op 'hivm.hir.vcmp' … scalar instruction` warning。§0.6 E-mask / §1.6.3 / §10.2 表述据实订正。
3. **zbuf 零初始化补齐（建议项 3）**——§6.4 第 3 条 + §0.6 E7 补入 PL-1.11 的 `l1_b`(V) 每核一次 `zbuf` 零初始化（根治 `P带(0)×V残留(NaN)=NaN` 整列污染；成本 ~0.06%），把「V残留任意值×0=0」假设改为「经 zbuf 零初始化保证为 0」。
4. **§0.1 角点澄清（建议项 4）**——补 `is_causal ∧ wr≥0` 角点说明（源 kernel 该分支忽略 right window；目标域无此组合），§9.3 记明不承诺。
5. **§1.6.1 验证覆盖措辞订正（建议项 5）**——改为「9 个数值型采纳项 + 4 E2E 经 `verify_equiv.py` 机器验证 11/11；M7/M12/M13 按构造论证」，并在结果表后加「验证方式标注」。

**为何不会再犯同一错误**：

- **块数 / 复杂度算术**：所有结构红利数字改为「**表达式 + 数值**」并附**逐项复算推导**（§1.6.0 R3 行内 + §5.4），且显式区分「执行块数」与「内段全有效块」两种口径、不再混用；fabric GB 数与括号内乘法逐项对账（同一口径）。
- **证据产物可运行性**：探针交付前在本工具链**实跑**（脚本内置 `tilelang.disable_cache()` 强制复编译以**确定性捕获编译期 warning**），且脚本**覆盖被采纳的形态**（主选算术惩罚链），不再以被否决的备选链冒充主选证据；引用探针结果的正文与实跑输出逐字对齐。

**已通过、本次未改动（不得破坏）**：迁移三问解读（§0.1–§0.4）、耦合性四态判定（§0.5）、E1–E7/E-band/E-mask/E-bnd 重设计（§0.6，除 E7 补 zbuf 外）、**Expert 编程模式结论（§2，P1 探针证据）**、M1/M2/M4/M5/M6 等价论证、向量化覆盖（§1.6.2）、§5.5 分核三要素（实查 20 AIC/40 AIV）、`verify_equiv.py` 11/11、与 MHA 母本的复用一致性。

---

## 12. 性能目标（optimize 场景预检追加）

> 本节由 conductor 在 optimize 场景预检后**追加**，不覆盖以上任何既有内容。以下字段即 Stage 4 的调优契约。

| 字段 | 值 |
|------|-----|
| 场景 | `scenario=optimize`（已有算子定制优化，`stage_plan=[4, 回归]`） |
| 算子目录（= kernel 所在目录） | `examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/` |
| 基准 kernel（**永不修改**） | 同目录 `_gqa_sw_fwd_kernel.py` |
| 调优产物目录 | 同目录 `perf_opt/`（`perf_opt/_gqa_sw_fwd_kernel.py`、`perf_records.jsonl`、`opt_log.md`） |
| **性能目标类型** | **`best_effort`**（尽力调优；无硬性数值门槛） |
| 目标数值 | 无（type=best_effort） |
| Baseline | 无外部 baseline；以迁移基线（§12.1）为改进参照 |
| **关注 shape（仅影响优先级）** | **优先最大绝对耗时 `llama-3.1-70b-long-w1024`（1352.78 us bf16 / 1339.04 us fp16）与 `llama-3.1-8b-long-w1024`（1319.00 / 1279.54 us）** |
| 测试 shape | 由 optimizer 展开 `benchmarks/ops/bench_grouped_query_attention_sliding_window.py::test_gqa_sliding_window_fwd_bench` 的实际参数化用例并分类（5 workload × fp16/bf16 = 10 case）；conductor 不预先缩减集合 |
| benchmark 参数化入口 | `examples/TileOPs/benchmarks/ops/bench_grouped_query_attention_sliding_window.py::test_gqa_sliding_window_fwd_bench`（`_manifest_params(load_workloads("GroupedQueryAttentionSlidingWindowFwdOp"))`） |
| 回归入口 | `python examples/TileOPs/tileops/kernels/attention/grouped_query_attention_sliding_window/grouped_query_attention_sliding_window_kernel/perf_opt/_gqa_sw_fwd_kernel.py --level all`（L0/L1 失败阻塞，L2/Boundary 告警不阻塞） |
| 噪声阈值 | 3%（默认） |
| **最大迭代数** | **10**（默认） |
| 预算 | 无预算限制 |

### 12.1 迁移基线性能（msprof 实测，2026-10-16，report `20261016_161519_904205_GroupedQueryAttentionSlidingWindowFwdOp`）

| workload | fp16 (us) | bf16 (us) | roofline util |
|---|---:|---:|---:|
| `gqa-sw-fwd-smoke-s512-h8-kv2-d64` | 99.56 | 101.76 | 2.8% |
| `llama-3.1-8b-short-w256`（S=512 H32/kv8 d128） | 481.78 | 494.74 | ~9.1% |
| **`llama-3.1-8b-long-w1024`（S=2048）** | **1279.54** | **1319.00** | ~16.9% |
| `llama-3.1-70b-short-w256`（S=512 H64/kv8 d128） | 479.56 | 494.12 | ~9.1% |
| **`llama-3.1-70b-long-w1024`（S=2048）** | **1339.04** | **1352.78** | ~16.3% |

### 12.2 首要优化候选线索（conductor 预注入，请独立验证后采纳或证伪）

1. ⭐⭐ **【采纳硬约束 · 必须遵守】`torch.library.custom_op` 输出别名契约**：本算子经 wrapper 的 `custom_op` 暴露给 TileOPs；**输出张量不得互为别名、也不得别名输入**（torch 会 `RuntimeError: ... The output of this custom operator (1) must not also be an input ... (2) may not alias ... other returns`）。
   - **血泪教训（同族序列上一个 optimize 任务，2026-10-17）**：`EngramGateConvBwdOp` 的 Stage 4 把 3 个权重梯度输出做成**同一 workspace 的切片（互相别名）**，内嵌 `--level all`（直调工厂、不经 `custom_op`）全过，但**采纳后 TileOPs 端到端 5/5 失败**，54.6× 增益作废。
   - **你的义务**：任何「共享 workspace / 输出由闭包切片 / 复用输入 buffer 作输出」的优化，产出后**必须经 `custom_op` 路径做端到端 smoke**（而非仅直调工厂），确保输出是独立张量（必要时物化/切片后 `contiguous()`+克隆语义）。见 `VP-2026-0302` / `VP-2026-0304`。
2. ⭐ **`CG-2026-0005`（本算子性能天花板的主要构成项）**：**L0C（Cube gemm 累加器）→ Vector UB/L1 无直接搬运通路**，Cube→Vector 必须经 **GM workspace 往返**（`T.store_fixpipe` dst 仅 GM）；S/P 中间量的往返流量与算法中间量同阶。→ 可评估空间：S/P **dtype 减半（f16）**（该缺口记录的唯一绕法）、减少**每 block 往返次数**、提高 **ws 复用**、优化 **Cube/Vector 流水重叠**（等待与依赖对齐）。已知 `PL-1.9-twophase`：该族两相位结构实测 2.31× 优于单遍链。
3. ⭐ **`PL-1.9-twophase`（attention.md，verified）**：非持久 grid `ceildiv(S,bm=96)` + Cube pass-1（全 S）/pass-2（全 O_partial）独立串行大循环 + 单 L0C 复用 + Vector 侧 rescale 回放——**本算子已采用该结构族**（Expert 两相位），调优时注意「wait 与依赖对齐」而非 FLOPs。
4. **`VP-2026-0283` / `TRAP-npuir-parallel-elemcast-load-scalarize`（同族已合入）**：`T.Parallel` 内**元素级 `T.cast` 读 GM / loop-carried 累加**会被 lower 为标量 `scf.for`；改「raw dtype 整块 `T.copy` + 整 tile `T.vcast`」/「非累加乘积 tile + `T.reduce_sum(dim=0, clear=False)`」。→ 请检查本算子 **Vector 侧 S 块消费 / 掩码链 / online softmax 累加** 是否有同类标量化（同族 `engram_decode` 因此提速 191–302×，`engram_gate_conv_fwd` 的 k1/k2 因此提速 9.4–66×）。
5. **`PL-1.16-expert-dualscope-bypass`** / `PL-1.7` / `PL-1.8` / `PL-1.9-hardlimits` / `PL-1.11` / `PL-1.12` / `PL-1.13`（`attention.md`，均 verified）：Expert 双 Scope 绕法、blockwidth、bf16 cube 直连、causal 掩码标量陷阱、band 域 stale 列双判据、流水深度、AIV 分片等——**本算子的滑窗 band 域掩码与这些条目直接相关**。
6. **`TRAP-npuir-per-copy-ub-staging-sum`**：`T.copy` GM→UB 的 staging **按 copy site 求和、不做 liveness 合并**——调 tile/加 copy 时须按 Σ(copy site × tile bytes) 复核 UB。
7. ⚠️ **不得回退的纪律**：Expert 双 Scope、CG-2026-0023 运行期 sync shim、`bishengir-compile` SIGSEGV 区分性重试、算术惩罚掩码（禁带 buffer 的 `T.if_then_else`）、禁裸标量广播（`T.vbrc`+`T.vmul`）、store 处禁 direct-cast、禁 `T.atomic_add`、`T.floordiv/T.ceildiv` 负分子须除法-free（`CG-2026-0024`）——见本文 §3.3/§9。
8. **同族可参考的既有 perf_opt（只读对照，勿照搬）**：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（MHA prefill 已调优版）。
9. **stale 提示**：`kb_stale_check.py` 报 `stale_count=268`；`status: stale/overturned` 条目不得作为决策依据。
