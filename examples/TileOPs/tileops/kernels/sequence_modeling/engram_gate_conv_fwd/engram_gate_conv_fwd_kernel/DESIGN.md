# `_engram_gate_conv_fwd_kernel` 算子设计文档（迁移）

> 项目：`engram_gate_conv_fwd`；算子目录：`examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/`。
> 迁移场景：`scenario=migration`（harness 子模式，GPU TileOPs → Ascend NPU npuir）。
> 编程模式：Developer。目标设备：Ascend910B3（实查 20 AI Core / 40 Vector Core）。
> 工具链戳：tilelang `0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2`（`/mnt/workspace/Agentkj/build`，USE_NPUIR=ON）/ CANN 9.0.0 / 2026-10-15。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 算法调研 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_fwd.py`

- 本迁移对象 = 工厂 `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)`（L42–L188，`@functools.lru_cache` L41）内的 `@tilelang.jit(out_idx=[6,7,8,9,10,11], compile_flags=["-O3","-DENABLE_BF16"])` 函数 `_func(threads)`（L46–L188）→ `@T.macro _gate_fwd`（L52–L164，全部计算）→ `@T.prim_func main`（L166–L184）；含**两个** `with T.Kernel(seq_len, M, threads=threads)` 块（Pass1 L68、Pass2 L128）。
- 辅助常量：`ALIGNMENT=256`（L33）、`CONV_KERNEL_SIZE=4`（L34）、`_align_up`（L37–L38）。
- `@torch.library.custom_op("top::engram_gate_conv_fwd", mutates_args=())` 包装（L191–L209）+ `register_fake`（L212–L224）；`class EngramGateConvFwdKernel`（L227–L309，`supported_archs=[80,86,89,90]`、`default_config={"threads":128}`、`autotune_configs=[128,256,512]`）。
- host 侧 `forward`（L269–L309）：对 `H/k/v/rms_w_h/rms_w_v/conv_w` 尾维 `F.pad` 到 `d_padded=_align_up(d,256)`，调用后把 `results[0]/results[1]`（Y/vhat）crop 回 `d`。
- NPU 脚手架（Stage 0 已生成，只读参照）：提取内核 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernels.py`（Pattern B，`_align_up` 已补全）、wrapper 同目录 `engram_gate_conv_fwd.py`（K5/K7/K8/K9 已改：`supported_archs=None`、`npub::engram_gate_conv_fwd`、去 autotune/tune、去 `threads`、`ascend_mode="Developer"`）、op `examples/TileOPs/tileops/ops/sequence_modeling/engram_gate_conv_fwd.py`（`EngramGateConvFwdOp` + `eval_roofline`）。
- **本仓既有 NPU 参照（同算子，只读）**：`examples/engram/engram_fwd.py`（`target="npuir"`、`T.Kernel(seq_len*M, is_npu=True)`、`T.alloc_shared`、`T.vrsqrt`、两 kernel（pass1/pass2）+ 每 token 一 block）。该参照证实**纯 Vector（无 Cube）路线可编译**，其 `T.vrsqrt`（近似指令）与「每 token 一 block、pass2 每输出 token 4 次 GM 行读」为本设计的显式改进点（§0.6 / §1.6）。
- **同族迁移案例（只读，预注入）**：`CASE-engram-decode-migration`（`pattern-library/cases.md` L590）+ 能力缺口 `CG-2026-0022`（`.agents/evolution/capability-gaps.md` L961）——同族 `engram_decode` 迁移的完整根因链，本算子逐条对照见 §0.5/§1.6.3/§9。

### 0.1 源算子语义（做什么）

**数学语义**（源码 docstring L1–L18 + `_gate_fwd` L52–L164 逐语句核对；与 `op_requirements.math_formula` 一致，**以源码为准**）：

```
h_norm = RMSNorm(H, rms_w_h)                          # (M, T, d)
k_norm = RMSNorm(k, rms_w_h)                          # (M, T, d)，注意共用 rms_w_h
alpha  = sigmoid( Σ_d (h_norm ⊙ k_norm) / sqrt(d) )    # (M, T)
v_hat  = alpha ⊙ v                                     # (M, T, d)
vhat_n  = RMSNorm(v_hat.to(H.dtype), rms_w_v)          # (M, T, d)
conv_o = Σ_{p=0}^{3} conv_w[p] ⊙ vhat_n[ t-(3-p), : ]  # (M, T, d)  因果深度卷积
Y      = SiLU(conv_o) + v_hat                          # (M, T, d)
```

其中 `RMSNorm(x,w) = x · rsqrt(Σ_d x² / d + eps) · w`（**除以真 d，非 d_padded**；源码 L100/L101/L125 用 `sumsq/float(d)+eps`）。

**关键订正（相对 `op_requirements` 初版公式）**：`op_requirements` 写作「dilated_causal_depthwise_conv1d」，但**源码 `engram_fwd.py` 的 GateConv-fwd 是无膨胀（dilation=1）因果深度卷积**——Pass2 的 `src_t = tid - (CONV_KERNEL_SIZE - 1) + p`（L146）步长为 1，无 dilation 因子；带 dilation 的是 `engram_decode`（本次不同算子）。本设计按**无膨胀**实现（§8.1 golden 亦按无膨胀）。

**I/O 契约**（`_gate_fwd` 签名 L53–L65 + `main` L167–L179 + op manifest `EngramGateConvFwdOp`）：

| 参数 | Shape（kernel 内） | dtype | 说明 |
|------|--------------------|-------|------|
| `H` | `(M, seq_len, d_padded)` | `float16 \| bfloat16` | 隐状态（host 侧尾维 pad） |
| `k` | `(M, seq_len, d_padded)` | same | key 投影（外部 GEMM 产物） |
| `v` | `(M, seq_len, d_padded)` | same | value 投影（外部 GEMM 产物） |
| `rms_w_h` | `(d_padded,)` | same | H/k 共用 RMSNorm 权重 |
| `rms_w_v` | `(d_padded,)` | same | v_hat 的 RMSNorm 权重 |
| `conv_w` | `(4, d_padded)` | same | 深度卷积权重 |
| `Y`（out[0]） | `(M, seq_len, d_padded)` | same | 输出（作为残差） |
| `vhat`（out[1]） | `(M, seq_len, d_padded)` | same | 保存的 `alpha⊙v`（backward 策略 B） |
| `alpha`（out[2]） | `(M, seq_len)` | `float32` | 门控标量 |
| `rrms_h`（out[3]） | `(M, seq_len)` | `float32` | H 的倒数 rms |
| `rrms_k`（out[4]） | `(M, seq_len)` | `float32` | k 的倒数 rms |
| `rrms_v`（out[5]） | `(M, seq_len)` | `float32` | v_hat 的倒数 rms |

- **输出 shape（迁移必需）**：`(M, seq_len, d)×2 + (M, seq_len)×4`（`source_output_shape` 一致）；`Y/vhat` 与输入同 dtype，4 个标量输出为 fp32。host 侧把 `Y`/`vhat` 由 `d_padded` crop 回 `d`（wrapper `forward` L306–L308）。
- **layout**：行主序连续，尾轴 `d` 连续；无转置、无 in-place。
- **动态轴**：无。`M/seq_len/d/eps/dtype` 全为编译期常量（`lru_cache` 特化）。
- **规约语义（关键）**：4 处行内规约（`sumsq_h`、`sumsq_k`、`dot_hk`、`sumsq_v`，源码 L92/L97/L111/L124）——沿尾轴 `d` 的 **fp32 行求和**；累加顺序未指定（GPU 线程级 reduce），本设计改为向量水平归约（§1.6.2），fp16/bf16 下与 golden 的差异落在测试容差内（`verify_equiv` FULL 行实测 max_abs 3.05e-5，见 §1.6.1）。
- **边界语义**：`d_padded` 尾区由 host 零填充（`H/k/v/rms_w* /conv_w` 的 pad 位为 0）→ `Σ_{d_padded} x² = Σ_d x²`（源码除以 `float(d)` 正确）；conv 的因果左边界 `src_t<0` 在源码用 `T.if_then_else(src_t>=0, ..., 0)` 置 0（L148–L157），等价于左侧零填充（`F.pad(v_perm, (3,0))`，golden L50）。**无 dilation 越界问题**。
- **dtype 语义**：源全程 `accum_dtype="float"`（fp32）计算，仅在写回 `Y`（L164）、`vhat_buf`（L122）时 `T.cast(..., dtype)`；`alpha/rrms_*` 直接写 fp32。即 fp16/bf16 输入 → fp32 计算域 → 单次舍回。
- **host 侧语义**：`F.pad` 尾维 pad 到 `d_padded` + 输出 crop——元数据/视图级处理（不改输入张量真实内容），属算子契约外的实现手段，NPU 侧保留（wrapper 已实现）。
- **语义保持基线**：§8.1 golden 以本节语义为唯一依据（**优先移植源仓参考实现/测试基准** `examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py::engram_gate_conv_fwd_torch`，L35–L63），**不得复刻 §0.6 / §1.6.3 的 NPU 结构**（保证验证独立性）。

### 0.2 源算子输入输出

见 §0.1 I/O 契约表。对外契约 shape 取自 manifest `EngramGateConvFwdOp.signature`：`H/k/v:[M,seq_len,d]`、`rms_w_h/rms_w_v:[d]`、`conv_w:[4,d]`；输出 `Y/vhat:[M,seq_len,d]`、`alpha/rrms_h/rrms_k/rrms_v:[M,seq_len]`；`shape_rules` 仅 `key.shape==H.shape` 等形状一致约束（**无 `d%N==0` 约束**，但 manifest workloads 的 `d∈{256,512}` 均 256 对齐）。

**dtype 语义（迁移必须保持）**：fp16/bf16 输入 → fp32 计算域 → `Y/vhat` 单次舍回原 dtype、4 个标量输出 fp32（源 `accum_dtype="float"`）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`_gate_fwd` L52–L164 逐语句，源码每个计算语句均归入某步骤）：

| # | Pass | 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|------|------|------|------|------|-------------|
| 1 | P1 | 载入 H/k/v 行 | `T.Parallel(d_padded)`：`h_2d[0,j]=cast(H[bid,tid,j])`、同 k、`v_1d[j]=cast(v[...])`（L84–L87） | GM `H,k,v` | frag `h_2d,k_2d,v_1d` | `H,k,v` |
| 2 | P1 | sumsq_h | `hsq=h_2d²`（L90–L91）→ `T.reduce_sum(dim=1)`（L92） | `h_2d` | `sumsq_h` | `Σ H²` |
| 3 | P1 | sumsq_k | `ksq=k_2d²`（L95–L96）→ reduce（L97） | `k_2d` | `sumsq_k` | `Σ k²` |
| 4 | P1 | rrms_h/rrms_k | `T.rsqrt(sumsq/float(d)+eps)`（L100–L101）→ 写 `rrms_*_buf`（L102–L103） | `sumsq_*` | GM `rrms_h/rrms_k` | `rrms` |
| 5 | P1 | h_norm/k_norm | `h_2d[0,j] = h_2d[0,j]·rrms_h·cast(rms_w_h[j])`（L106–L108，k 同式共用 `rms_w_h`） | `h_2d,k_2d,rrms_*` | `h_2d,k_2d`(归一) | `h_norm,k_norm` |
| 6 | P1 | 门控点积 | `hk_prod=h_2d·k_2d`（L109–L110）→ reduce（L111） | `h_2d,k_2d` | `dot_hk` | `Σ(h_norm⊙k_norm)` |
| 7 | P1 | alpha | `1/(1+exp(-(dot/T.sqrt(d))))`（L114）→ 写 `alpha_buf`（L115） | `dot_hk` | GM `alpha` | `sigmoid(dot/√d)` |
| 8 | P1 | vhat + rrms_v | `vhat=alpha·v_1d`（L121）；`vhat_buf=cast(vhat,dtype)`（L122）；`vhsq=vhat²`（L123）→ reduce（L124）→ `rrms_v=rsqrt(sumsq_v/d+eps)`（L125）→ 写 `rrms_v_buf` | `alpha,v_1d` | GM `vhat,rrms_v` | `v_hat,rrms_v` |
| 9 | P2 | 载入 vhat 当前行 | `vhat_cur[j]=cast(vhat_buf[bid,tid,j])`（L136–L137） | GM `vhat` | `vhat_cur` | 残差 `v_hat` |
| 10 | P2 | 因果深度卷积 | `conv_out≡0`（L140–L141）；`T.serial(4)`：`src_t=tid-3+p`；`raw=if_then_else(src_t≥0, cast(vhat_buf[bid,src_t,j]),0)`；`src_rrms=if_then_else(src_t≥0, rrms_v_buf[bid,src_t],0)`；`normed=raw·src_rrms·cast(rms_w_v[j])`；`conv_out[j]+=cast(conv_w[p,j])·normed`（L145–L159） | GM `vhat,rrms_v`, `rms_w_v,conv_w` | `conv_out` | `conv_o` |
| 11 | P2 | SiLU + 残差 | `sig=1/(1+exp(-conv_out[j]))`；`Y[bid,tid,j]=cast(conv_out[j]·sig+vhat_cur[j],dtype)`（L162–L164） | `conv_out,vhat_cur` | GM `Y` | `Y` |

**数据流图（源硬件 = GPU 视角）**：

```
GM[H,k,v] --T.Parallel(threads 映射 d)--> REG[h_2d/k_2d(1,d)/v_1d(d)]  (每 block = 一个 (batch,token))
  REG[h_2d]² --reduce_sum(dim=1)--> sumsq_h --rsqrt--> rrms_h --(h·rrms_h·w_h)--> REG[h_norm]
  REG[k_2d]² --reduce_sum(dim=1)--> sumsq_k --rsqrt--> rrms_k --(k·rrms_k·w_h)--> REG[k_norm]
  REG[h_norm]⊙REG[k_norm] --reduce_sum--> dot_hk --(sigmoid(dot/√d))--> scalar alpha --> GM[alpha]
  alpha⊙REG[v_1d] --> vhat(REG) --cast--> GM[vhat] ;  vhat² --reduce--> sumsq_v --rsqrt--> GM[rrms_v]
---- 分离的第二 grid 启动（Pass2）----
GM[vhat] --(行 tid)--> REG[vhat_cur]
GM[conv_w], GM[vhat], GM[rrms_v], GM[rms_w_v] --(serial p=0..3, 行 tid-3+p)--> REG --FMA--> REG[conv_out]
  REG[conv_out] --(conv·sigmoid(conv)+vhat_cur)--> GM[Y]
```

- **无 SMEM**：全部中间量驻留 register/fragment（`T.alloc_fragment` ×12，L70–L78/L118–L119/L129–L130）；`block_size = threads`。
- **Pass2 通过 GM 复用**：`vhat_buf`（fp16/bf16）与 `rrms_v_buf`（fp32）在 Pass1 写出、Pass2 读回——**邻居行由运行时行索引 `src_t` 从 GM 读取**（L150/L155），这是 causal 窗口在源实现里的落地方式。

**循环与并行结构**：`T.Kernel(seq_len, M, threads=threads)` ×2 —— **2D grid `(seq_len, M)`**（`bx=tid=token`、`by=bid=batch`）+ `threads` 线程/block（`op_requirements`：grid=(T,M)，每 block 一个 (batch, token)）。`T.Parallel(d_padded)` 把 d 元素分给 threads；conv tap 维走 `T.serial(4)`，元素维走 `T.Parallel`。

**host 侧逻辑**：`F.pad` 尾维 pad 到 `d_padded`、输出 crop（L293–L308）；`lru_cache` 按 `(M,seq_len,d,eps,dtype)` 编译期特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **两阶段融合进同一 prim_func**（Pass1 门控+norm / Pass2 卷积+SiLU，共享 `out_idx`） | 减 kernel 启动数与 Python 分派 | 两个 `T.Kernel` 块、中间量经 GM 交接 | 无（结构层） | 结构可移植 → **保留（结构）/ 部分重设计（§0.6 R1）** |
| 2 | **2D fragment `(1,d_padded)` 供 `T.reduce_sum(dim=1)`** | 行内归约 | 沿 dim=1 的水平 reduce | GPU 线程级 reduce | 优化层 → **等价替换（UB reduce）** |
| 3 | **全程 fp32 累加**（`accum_dtype="float"`，每处输入 `T.cast`） | 数值精度（对齐 golden fp32 opmath） | fp16/bf16 → fp32 计算、末端单次舍回 | 半精度 ALU 下需中转 | **语义层 → 保留并强化（NPU bf16 v-op 缺失）** |
| 4 | **fragment/register 驻留**（`h_2d/k_2d/v_1d/...` ×12） | 中间量不出寄存器 | 大寄存器堆 | GPU 寄存器堆 | **硬件强相关 → 等价替换（UB）** |
| 5 | **2D grid `(seq_len, M)` + `threads` 线程级 d 并行** | token/batch 级任务并行 + 元素并行 | `T.Kernel(seq_len, M, threads)` | GPU block/thread 调度 | **硬件强相关 → 重新设计（一维 Kernel + 向量化）** |
| 6 | **256 对齐 pad（`d_padded`）** | GPU 向量对齐、免尾界守卫 | host `F.pad` + 编译期 `d_padded` | memory coalescing | 实现层 → **保留逻辑**（manifest `d∈{256,512}` 已对齐） |
| 7 | **Pass1 在 vhat 仍在寄存器时立即算 rrms_v**（L118–L125） | 省一次 vhat 重载/重算 | 复用 register 内 `vhat_val` | 寄存器驻留 | 算法层 → **保留（CSE 物化，§1.6.1 O3）** |
| 8 | **Pass2 经 GM 复用 Pass1 的 `vhat_buf`/`rrms_v_buf`，邻居行用运行时索引读** | 保存 backward 中间量（策略 B）+ causal 窗口 | GM 落盘 + 运行时行索引 | 无（算法/结构层） | **重新设计（§0.6 R1：融合单 kernel，UB 内窗口，消 GM 往返）** |
| 9 | **`out_idx=[6..11]` 自动分配 6 输出** | 免 host 手动分配 | JIT 输出约定 | 无 | 保留 |
| 10 | **`T.if_then_else(src_t≥0, 载入, 0)` 处理因果左边界**（L148–L157） | 免显式分支 / 左零填充等价 | select 语义 | — | **重新设计（§0.6 R4：算术行掩码——`T.if_then_else` 带 buffer 操作数在 npuir codegen 段错误，见 §1.6.3 探针 V2）** |
| 11 | **权重（`rms_w_h/rms_w_v/conv_w`）逐元素直接读 GM**（L107–L108/L158–L159） | 免权重预载 | 片上直读 | — | 保留（权重极小，可整块载入 UB） |
| 12 | **无膨胀因果深度卷积（kernel 4）** | 结构简单、窗口静态 | 编译期常量 tap 偏移 | — | 保留 |
| 13 | **`threads` autotune（128/256/512）+ `lru_cache` 特化** | 按 shape 选线程数 / 消运行时分派 | 编译期常量固化 | `threads` 无 NPU 对应 | `threads` → **舍弃**；`lru_cache` → **保留** |

> 识别不出来 ≠ 不存在。上述 13 项为源码可识别的全部优化/规避手段；逐项处置见 §0.5。未识别的优化会被静默丢弃，导致迁移后性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0：源算法与调研候选同台比较）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义（3×RMSNorm + sigmoid 门控 + 无膨胀因果深度卷积 + SiLU 残差；I/O、边界、host pad 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| #1 两阶段融合（含各步骤数学顺序） | 算法 | 无（纯数学顺序，无跨 step 硬件假设） | — | **保留（结构）** | NPU 单 kernel 内保留「门控/norm → 卷积」的顺序（§0.6 R1） |
| #2 2D fragment `reduce_sum` | 优化 | GPU 线程级 reduce | **有**（`T.reduce_sum` on UB，`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.2：src/dst 同 rank、dst 归约维为 1） | **等价替换** | UB `(R,d_padded) f32 → (R,1) f32` 的 `T.reduce_sum(..., dim=1)`（§1.6.2 #1） |
| #3 全程 fp32 累加 | 语义/优化 | 无 | **有**（fp32 计算 + `T.vcast`）；且 **NPU v-op 无 bf16 使其成为 API 强制** | **保留（并强化）** | → §0.6 R2：fp16/bf16 经 `vcast(rint)` 升 fp32 → fp32 向量链 → `vcast(rint)` 舍回 |
| #4 fragment/register 驻留 | 优化 | GPU 寄存器堆 | 无直接等价（NPU 无 `threads`，v-op 作用于 UB buffer） | **等价替换** | → §0.6 R3：`T.alloc_shared`（UB）整块驻留 + v-prefix 整 buffer 向量链 |
| #5 2D grid `(seq_len,M)` + threads | 优化 | GPU block/thread 调度 | **无**（npuir 仅一维 `T.Kernel`；`threads=` 无效，`TRAP-threads-kwarg-noop`） | **重新设计** | → §0.6 R3：一维 `T.Kernel(M*num_tiles, is_npu=True)`，元素并行交给向量单元 |
| #6 256 对齐 pad | 优化 | memory coalescing | 有（32B 尾轴对齐） | **保留（逻辑）** | manifest `d∈{256,512}` → `d_padded==d`，不触发；保留变量以支持通用 d（§5.4） |
| #7 Pass1 复用 register 内 vhat 算 rrms_v | 优化 | 寄存器驻留 | 有（UB 充裕） | **保留** | → §1.6.1 O3：fp32 `vhat` 物化于 UB，供 vhat_norm 与残差共用 |
| **#8 Pass2 经 GM 复用 vhat/rrms_v + 运行时行索引读邻居** | 优化 | 无（算法结构） | 有（UB 内窗口 + 运行时索引/切片） | **重新设计** | → §0.6 R1：融合单 kernel，seq 轴分块，vhat_norm 在 UB 内做因果滑窗，消 Pass2 的 4× GM 行重读 |
| #9 `out_idx` 自动输出 | 优化 | 无 | 有（`out_idx` 约定，`TRAP-outidx-call-convention`） | **保留** | 6 输出 `out_idx=[6,7,8,9,10,11]`（与源一致） |
| #10 `T.if_then_else(src_t≥0,…)` 因果边界 | 优化 | select 语义 | 有（但**带 buffer 操作数的 `T.if_then_else` 在 npuir codegen 段错误**——本任务探针 V2 实测 EXIT=139） | **重新设计** | → §0.6 R4：索引 clamp（`T.max/T.min`）+ 算术行掩码（`T.vbrc`+`T.vmul`）替代 |
| #11 权重 GM 直读 | 优化 | 无 | 有 | **保留** | 权重极小（`6·d·elem`），整块载入 UB 复用 |
| #12 无膨胀因果深度卷积（kernel 4） | 算法 | 无 | 有 | **保留** | tap 编译期展开（4 个常量偏移），UB 内 shifted 读（§0.6 R1） |
| #13 `threads` autotune | 优化 | GPU 线程模型 | 无 | **舍弃** | K3/K9：npuir 无 `threads`；`lru_cache` 保留 |
| fp8 / 其他非 manifest dtype | — | — | 无（manifest 仅 fp16/bf16） | **舍弃** | 不在契约内 |

> **依据**：#3 的 bf16 缺失由 `docs/Tilelang.language/数学操作/T.vadd.md`·`T.vmul.md`·`T.vdiv.md`·`T.sqrt.md`（`T.vsqrt`）、`数据类型转换操作/T.vcast.md` §2.2.1 的 dtype 矩阵佐证（`vsigmoid`/`vsqrt`/`vdiv`/`vmul`/`vadd` 均 **bf16 ✗**，fp16 ✓ / fp32 ✓；`vcast` 支持 `f16→f32`/`bf16→f32` 的 `rint`）；#5 由 `pattern-library/traps-compiler.md` `TRAP-threads-kwarg-noop`（status=verified）+ 脚手架 K3/K9（wrapper 已去 `threads`）+ 本仓 `examples/engram/engram_fwd.py`（`T.Kernel(seq_len*M, is_npu=True)` 一维）佐证；#10 由本任务设计期探针（`history_version/probe_c_bisect_dynrow.py` V2，2026-10-15，工具链戳见页首）实测佐证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目（R1/R3/R4）逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。**重设计直接承接 §1.6.0 调研选定算法族（纯 Vector 序列分块单 kernel + fp32 中转 + rsqrt 绕法 + UB 内因果滑窗）。**

**重设计项 R1：执行结构重设计（两 Pass + GM 邻居读 → 单 kernel 序列分块 + UB 内因果滑窗）**

- **源方案**（步骤 8–10，L117–L159）：Pass1 把 `vhat_buf`（dtype）与 `rrms_v_buf`（fp32）写回 GM；Pass2 每输出 token 经**运行时行索引**从 GM 读 `vhat_buf[bid, tid-3+p, :]` 与 `rrms_v_buf[bid, tid-3+p]`（4 个 tap 行），再 `normed = raw·src_rrms·rms_w_v`，累加。性能意图 = 用 GM 落盘保存 backward 中间量（策略 B 契约要求 `vhat` 输出），并以运行时索引实现 causal 滑窗。
- **NPU 新算法**：**单 kernel**，一维 grid `M × num_tiles`（`num_tiles = ceil(seq_len/BLK_S)`，`BLK_S` 为每 block 处理的输出 token 数，默认 8）；每 block 处理某 batch 的 `BLK_S` 个连续输出 token，并**在 block 内重算 3 个因果 halo token 的 `vhat_norm`**（halo = 卷积核长 − 1 = 3）：
  - 载入窗口 tile `(R, d_padded)`（`R = BLK_S + 3`）的 `H/k/v`：行索引 `gt = t0 - 3 + i`（`t0 = tile·BLK_S`，运行时值）经 **clamp**（`T.max(gt,0)` / `T.min(...,seq_len-1)`）后按元素读 GM（探针 V1 实证该动态行索引读合法）。
  - fp32 域完成全部 RMSNorm、门控、`vhat`、`vhat_norm`（`(R,d_padded)` 2D 向量链 + `reduce_sum(dim=1)` 行归约）。
  - **因果卷积在 UB 内完成**：`conv_t[i,j] = Σ_{p=0}^{3} cw[p,j] · vn[i+p, j]`——`vn[i+p,j]` 为 UB 内**仿射行偏移读**（探针 A 实证合法）。
  - 只对 block 自身的 `BLK_S` 行做输出写回（`Y/vhat/alpha/rrms_*` 按全局行索引 `t0+i` 写 GM；探针 V1/V3 实证动态行索引读写合法）。
  - **Pass2 的 4× GM 行重读被消除**（vhat_norm 不落 GM），GM 流量从「3× 读 + 1× 写 + 4× 重读」降至「3×(R/BLK_S)× 读 + 2× 写」（§1.6.0 R3）。
- **语义保持论证**：
  1. **数学等价**：分块后每输出 token `t` 的卷积仍为 `Σ_p cw[p]⊙vn[t-3+p]`（halo 覆盖 `t-3..t`），仅把「从 GM 读邻居行」换成「block 内重算邻居行」——计算内容一致（`vhat_norm` 是确定的逐元素函数），无累加顺序变化。归约沿 `d` 的向量水平 reduce 与源线程级 reduce 数学同式（分组/顺序微差，`verify_equiv` FULL 行实测 max_abs 3.05e-5）；
  2. **边界语义**：`gt<0`（tile 0 因果左边界）与 `gt≥seq_len`（末 tile 右侧尾块）由**索引 clamp + 算术行掩码**处理——无效 halo 行的 `vn` 被乘 0，贡献与源 `if_then_else(...,0)` 逐元素一致；无效输出行不写（`tail` 裁剪）。**无 dilation 越界**（本算子无膨胀）；
  3. **精度**：全程 fp32；`vhat` 单次 `vcast(rint)` 舍回输出 dtype；不改变源 fp32 计算域。**源优化意图承接**：源 #8「GM 复用 + 运行时索引窗口」→ UB 驻留 + UB 内仿射窗口承接（消 GM 往返）；源 #9「backward 中间量落盘」→ `vhat` 输出契约不变，UB 内 vhat 兼作 norm 与残差来源。

**重设计项 R2：计算域与倒数 rms 重设计（原生域 → fp32 中转 + rsqrt 绕法）**

- **源方案**：`T.cast(...,"float")` 升 fp32 计算、末端 `T.cast(...,dtype)` 舍回；3 次 `T.rsqrt(sumsq/float(d)+eps)`（L100/L101/L125）。
- **NPU 新算法**：
  - **fp32 中转**：fp16/bf16 输入先 `T.vcast(..., round_mode="rint")` 升 fp32 → 全部向量链 fp32 → 末端 `T.vcast(..., round_mode="rint")` 单次舍回（输出 `Y/vhat` 与标量）。**bf16 为 API 强制**（`vsigmoid/vsqrt/vdiv/vmul/vadd` 均无 bf16）；fp16 保留同路径以对齐源 fp32 语义并统一实现。
  - **rsqrt 绕法**：`rrms = rsqrt(s)` → `t := vsqrt(s)`；`rrms := t/s`（`vsqrt` + 标量除，等价 `1/sqrt(s)`），消除 `T.vrsqrt` 近似指令（`TRAP-vrsqrt-plain-precision` 实测 max rel err ~2.9e-3；本算子 3 次 RMSNorm 累积）。
- **语义保持论证**：① 计算域与源（`accum_dtype="float"`）及契约 golden（`.float()` 全链路）**同域**，非替代；② `sqrt(s)/s ≡ s^{-1/2}`（s>0）实数恒等；③ 边界 `s = Σx²/d + eps ≥ eps = 1e-6 > 0`，定义域内。**源优化意图承接**：源 #3「fp32 累加保障精度」→ 承接并**强化**（bf16 API 强制 + 绕开近似 rsqrt）。

**重设计项 R3：并行/内存结构重设计（2D grid + fragment + threads → 一维 Kernel + UB + 序列分块）**

- **源方案**：#5：`T.Kernel(seq_len, M, threads=t)`（每 block 一个 (batch,token)）+ #4 `T.alloc_fragment`，`threads ∈ {128,256,512}` autotune。
- **NPU 新算法**：一维 `T.Kernel(M*num_tiles, is_npu=True)`（每 block 处理一个 batch 的 `BLK_S` 个连续 token + 3-halo）；所有 `(R,d_padded)` 中间量放 `T.alloc_shared`（UB，fp32）；`GM→UB` 用整块 `T.copy`/`T.Parallel` 载入，向量链用 v-prefix op 作用于整 buffer；`threads`/autotune 移除（K3/K9）。
- **语义保持论证**：任务映射仅重排「哪个核处理哪些 token」，token 间除 causal 窗口外无依赖（窗口由 halo 覆盖），数学逐元素/逐行恒等；无累加顺序变化；无边界影响。**源优化意图承接**：源 #4「中间量驻留片上」→ UB 承接；源 #5「token 级并行」→ 一维 grid 的 block 级并行 + block 内 `BLK_S` 行向量化承接。

**重设计项 R4：因果边界重设计（`T.if_then_else` select → 索引 clamp + 算术行掩码）**

- **源方案**：#10（L148–L157）：`raw = T.if_then_else(src_t≥0, cast(vhat_buf[...]), 0)`、`src_rrms = T.if_then_else(src_t≥0, rrms_v_buf[...], 0)`，用 select 语义吸收因果左边界。
- **NPU 新算法**：**避免带 buffer 操作数的 `T.if_then_else`**（本任务探针 V2 实测其在 npuir Developer codegen **段错误 EXIT=139、无诊断**，见 §1.6.3 与 §9.2）。改为：
  1. **索引 clamp**：载入行索引 `gc = T.min(T.max(gt, 0), seq_len-1)`（纯标量 `T.max/T.min`，探针 V1 实证合法），保证 GM 读写不越界；
  2. **算术行掩码**：每 halo/tile 行计算标量掩码 `rmask[w] = cast(min(max(gt+1,0),1) · min(max(seq_len-gt,0),1), f32)`，经 `T.vbrc` 广播到 `(R,d_padded)` 后 `T.vmul` 把无效行的 `vn` 置 0（等价左/右零填充）。
- **语义保持论证**：掩码对 `gt∈[0,seq_len)` 取 1、否则取 0，与源 `if_then_else(gt≥0,·,0)` 及 golden 的因果零填充（`F.pad(v_perm,(3,0))`）在有效行上逐元素一致；无效行 `vn·0=0` 对卷积贡献 0（与源同）；掩码标量构造全程 fp32 标量算术，无 buffer-select。**源优化意图承接**：源 #10「select 吸收边界」→ 算术掩码承接（语义等价，且规避 npuir codegen 缺陷）。

**重设计项 R5：门控点积因子化（数学等价，设计新增，非源方案替换）**

- **源方案**（步骤 5–6，L106–L111）：先物化 `h_norm = H·rrms_h·w_h` 与 `k_norm = k·rrms_k·w_h`（各 2 个 `vmul`），再 `hk_prod = h_norm⊙k_norm`（1 个 `vmul`），再行归约。
- **NPU 新算法**：`h_norm⊙k_norm = (rrms_h·rrms_k)·(H⊙k⊙w_h²)` ⇒ 直接算 `dot = (rrms_h·rrms_k)·Σ_d(H⊙k⊙w_h²)`，省去 `h_norm/k_norm` 的两次物化（因二者仅用于该点积）。
- **语义保持论证**：见 §1.6.1 O4（实数恒等，机器验证 EQUIV_PASS）。**源优化意图承接**：源 #7「寄存器复用」的意图（减少中间量）由本项承接并强化。

> **承接说明（源优化意图不丢弃）**：#1→单 kernel 结构保留（R1）；#2→§1.6.2 #1；#3→R2（强化）；#4→R3（UB）；#5→R3（一维 Kernel，`threads` 舍弃）；#6→保留逻辑；#7→保留（§1.6.1 O3）；#8→**R1**；#9→保留；#10→**R4**；#11→保留；#12→保留；#13→舍弃 `threads` + 保留 `lru_cache`。数学语义（§0.1）逐项保持。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_fwd.py`（工厂 L42–L188、`EngramGateConvFwdKernel` L227–L309）。
- 源仓测试基准（**golden 直接移植对象**）：`examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py`（`engram_gate_conv_fwd_torch` L35–L63 + `_rmsnorm` L27–L32；用例 `EngramGateConvFwdFixture` L71–L79：`(1,32,256,fp16)` / `(1,32,256,bf16)` / `(2,64,512,fp16)` / `(2,16,256,bf16)`；容差 fp16 `atol=1e-1`、bf16 `atol=2e-1`、`rtol=1e-1`）。
- manifest：`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `EngramGateConvFwdOp`（signature/shape_rules/workloads/roofline；smoke `fwd-smoke-b1-s16-d256-fp16` + 3 个 full workload）。
- 同算子 NPU 只读参照：`examples/engram/engram_fwd.py`（两 kernel、每 token 一 block、`T.vrsqrt`、GM 邻居读——本设计的显式改进基线）。
- 同族迁移档案（只读）：`examples/engram_decode/_engram_decode_kernel/DESIGN.md`（norm+GEMV+conv 融合迁移范式）、`pattern-library/cases.md` `CASE-engram-decode-migration`（MIX 缺口 + 向量迁移双陷阱）、`capability-gaps.md` `CG-2026-0022`、`pattern-library/traps-compiler.md`（`TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce` / `TRAP-C13` / `TRAP-conditional-alloc`）。
- golden 以 §0.1 语义为依据（移植 `engram_gate_conv_fwd_torch`），**不复刻 §0.6/§1.6.3 的 NPU 结构**（保证验证独立性）。

---
## 1. 概述

### 1.1 算子名称

`_engram_gate_conv_fwd_kernel`（项目 `engram_gate_conv_fwd`，目录 `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/`）。

### 1.2 功能描述

序列级 Engram GateConv 前向融合算子（post-projection fusion）：对每个 batch 的整条序列，把 3 组隐向量 `H/k/v`（`(M, seq_len, d)`）做 **3 次 RMSNorm + sigmoid 门控 + 无膨胀因果深度卷积 + SiLU 残差**：`H/k` 归一后点积得门控标量 `alpha`，`v_hat = alpha⊙v` 再归一后做因果深度卷积，`Y = SiLU(conv_out) + v_hat`。投影（`k = E·W_K`、`v = E·W_V`）由外部 GEMM 完成。支持 `float16 | bfloat16`；输出 `Y/vhat:(M,seq_len,d)`（同输入 dtype）+ `alpha/rrms_h/rrms_k/rrms_v:(M,seq_len)`（fp32）。含 4 次行内规约 + 窗口卷积 + 逐元素链，属**多步融合类**算子，**纯 Vector（无矩阵乘）**。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 执行结构由「2 个 grid 启动（Pass1/Pass2）+ 经 GM 复用中间量 + 运行时行索引读邻居」改为「**单 kernel、序列分块（每 block `BLK_S` 个 token + 3-halo）、vhat_norm 在 UB 内做因果滑窗**」；② 计算域统一 fp32 中转（bf16 API 强制 + 源 fp32 语义）；③ 3 次 RMSNorm 的 `rsqrt` 换成 `vsqrt+vdiv`（绕近似指令）；④ 因果边界由 `T.if_then_else` 改为「索引 clamp + 算术行掩码」；⑤ 门控点积因子化（省 2 次向量物化）。**计算语义（数学公式、I/O、边界）与源一致。**

### 1.3 数学公式

$$
\begin{aligned}
h_n &= \mathrm{RMSNorm}(H, w_h),\qquad k_n = \mathrm{RMSNorm}(k, w_h) && (\text{2}\times\text{RMSNorm})\\
\alpha &= \sigma\!\Big(\tfrac{1}{\sqrt d}\sum_{c} h_n[c]\,k_n[c]\Big),\qquad \hat v = \alpha\odot v && (\text{gate})\\
v_n &= \mathrm{RMSNorm}\big(\hat v.\mathrm{to}(\mathrm{dtype}), w_v\big) && (\text{3rd RMSNorm})\\
c_o[t] &= \sum_{p=0}^{3} \mathrm{cw}_p \odot v_n[\,t-3+p\,]\qquad (v_n[\cdot]=0\ \text{when}\ \cdot<0) && (\text{causal DWConv1D, k=4})\\
Y &= \mathrm{SiLU}(c_o) + \hat v && (\text{SiLU}+\text{residual})
\end{aligned}
$$

其中 `RMSNorm(x,w) = x · (∑_c x²/d + ε)^{-1/2} ⊙ w`（**除以真 d**，非 d_padded）、`σ = sigmoid`、`⊙` 为逐通道逐元素乘、索引 `[t]` 表示序列位置。**无膨胀**（dilation=1）。

NPU 计算核（§1.6.1 优化后，fp32 计算域、末端单次 `vcast(rint)` 舍回；`rrms(s)=sqrt(s)/s`、`·(1/d)`、`·(1/√d)` 常数折叠；门控点积因子化）：

$$
\mathrm{rrms}(s)=\frac{\sqrt{s}}{s},\ \ s=\Big(\sum_c x^2\Big)\cdot\underline{\mathrm{fl}(1/d)}+\epsilon;\qquad
\alpha=\sigma\!\big(\mathrm{rrms}_h\,\mathrm{rrms}_k\sum_c H[k]\!\odot\!k[c]\!\odot\! w_h[c]^2 \,\cdot \underline{\mathrm{fl}(1/\sqrt d)}\big)
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法数学等价、在并行/内存实现上有差异：**

- **投影**：无（外部 GEMM 已完成 `k/v`）。
- **序列分块单 kernel（§0.6 R1/R3）**：一维 grid `M × num_tiles`（`num_tiles = ceil(seq_len/BLK_S)`）；每 block 载入窗口 `(R, d_padded)`（`R = BLK_S + 3`）的 `H/k/v`（因果 halo 3 行），全流程 fp32；只在 block 内 `BLK_S` 行写输出。
- **RMSNorm ×3（§0.6 R2）**：`(R,d_padded)` 行内向量归约 `T.reduce_sum(dim=1)`；`s = Σx²·fl(1/d)+eps`；`t = vsqrt(s)`；`rrms = t/s`（绕 `vrsqrt`）；`x_n = x·rrms_brc⊙w_brc`。
- **门控（§0.6 R5 / §1.6.1 O4）**：`dot = (rrms_h·rrms_k)·Σ_d(H⊙k⊙w_h²)`（省去 h_norm/k_norm 物化）；`alpha = vsigmoid(dot·fl(1/√d))`；`vhat = vhat = alpha_brc ⊙ v`。
- **因果深度卷积（§0.6 R1）**：4 个 tap 编译期展开；`conv_t[i,:] = Σ_p cw[p,:] ⊙ vn[i+p,:]`——**UB 内仿射行偏移读**（探针 A 实证）；无效 halo 行由算术行掩码置 0。
- **SiLU + 残差**：`Y = c_o ⊙ vsigmoid(c_o) + vhat`，末端 `vcast(rint)` 舍回 `dtype`。
- 除上述外与 §1.6.0 调研选定算法族一致：单 kernel、无中间 GM 往返（除输出契约）。

### 1.5 数据流图

```
GM[H,k,v] --(clamped 运行时行索引, T.Parallel)--> UB[H/k/v tile (R,d) f32]
   --[RMSNorm/门控/门控因子化: vcast↑ · vmul · reduce_sum(dim=1) · vsqrt/vdiv · vsigmoid · vbrc]--> UB[vhat_t, vn_t]
   --[因果 DWConv: vn_t[i+p] 仿射偏移 + cw_brc 逐 tap FMA]--> UB[conv_t (BLK_S,d) f32]
   --[SiLU(vsigmoid) + vhat 残差 + vcast↓]--> GM[Y, vhat, alpha, rrms_h/k/v]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代，最后回答「在哪个轴向量化」。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**多步融合类**（3 次行内规约 + 门控 + 窗口卷积 + 逐元素链），按 `references/algorithm-research.md` §2 属**完整调研**：R1 候选表、R2 在线算法、R3 四口径对比、R4 逐候选评估全部展开。**基线与候选同台**：源算法（两 Pass + token-per-block + GM 邻居读）只是基线候选之一。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3/R4 |
|---|------|------------|-----------|----------|-----------|
| 0 | **基线：源算法** | 2×`T.Kernel(seq_len,M)`；每 token 一 block；Pass2 经 GM 读邻居行；`T.rsqrt`；逐元素 `T.Parallel` | 定义式 | 7 行读 / 2 行写 per token | ✅（基线） |
| 1 | **序列分块单 kernel（采纳）** | 单 `T.Kernel(M·num_tiles)`，每 block `BLK_S` token + 3-halo，UB 内因果滑窗 | 数学等价（halo 重算 = 邻居读；§0.6 R1） | 消 Pass2 的 4× GM 行重读 | ✅（采纳） |
| 2 | **rsqrt 绕法（采纳）** | `rsqrt(s)` → `sqrt(s)/s`（`vsqrt`+标量除） | 实数恒等（s>0）；机器验证 O1 | 消灭 ~2.9e-3 近似误差（3 次 Norm 累积） | ✅（§1.6.1 O1） |
| 3 | **常数倒数折叠（采纳）** | `Σx²/d`→`Σx²·fl(1/d)`；`dot/√d`→`dot·fl(1/√d)` | 容差内等价（fp32 ≤1ulp；D-1 O2） | `vdiv`→`vmul`（op 型替换） | ✅（§1.6.1 O2） |
| 4 | **CSE 物化 vhat（采纳）** | `vhat=alpha⊙v` 物化一次，共用于 `vn` 与残差 | 逐位恒等（D-1 O3） | 少重算（源 pass2 经 GM 重读） | ✅（§1.6.1 O3） |
| 5 | **门控点积因子化（采纳）** | `Σ(h_n⊙k_n)` → `rrms_h·rrms_k·Σ(H⊙k⊙w_h²)` | 实数恒等（标量外提）；D-1 O4 | −3 向量 op（省 h_n/k_n 物化） | ✅（§1.6.1 O4） |
| 6 | **fp32 中转计算域（采纳 = 保留 + API 强制）** | fp16/bf16 →`vcast(rint)`→ fp32 链 →`vcast(rint)` | 与源/契约同域（源即 fp32）；bf16 为 API 强制 | 正确性（bf16 可编译） | ✅（§1.6.1 O5） |
| 7 | **conv → im2col/GEMM（不采纳）** | 4-tap 深度卷积展开为 GEMM | 数学等价 | — | ❌ 弃选：**depthwise（逐通道）**、无共享权重矩阵，tap=4 极小，GEMM 化不成立（同族 `CASE-engram-decode-migration` 结论） |
| 8 | **归约用 Cube `T.gemm` GEMV（不采纳）** | `sumsq`/`dot` 表达为 M=d/K=d/N=1 的 GEMM | 数学等价 | MAC 落 Cube | ❌ 弃选：本算子**无权重矩阵**（归约是逐通道自身点积，非 GEMM）；且 `CG-2026-0022` 实测「Developer auto CV-split 下 Cube `T.gemm` 与 shared 向量规约+广播无法共存」，组合 kernel 被 BiShengIR verifier 拒（详见 §1.6.3 MIX 评估） |
| 9 | **sigmoid/softmax 减 max 稳定化（不适用）** | `σ(x)=e^{x-m}/(1+e^{x-m})` | 数学等价 | — | ❌ 不适用：`1/(1+e^{-x})` 在 fp32 全域无上溢（x→−∞ → 结果 0，正确）；门控 `dot/√d` 量级 ~O(√d/√d)=O(1)，无稳定化必要 |
| 10 | **两 Pass 序列分块（备选）** | Pass1 写 `vhat_norm` 到 GM；Pass2 分块读 halo 做卷积 | 数学等价 | 结构简单、每构造有先例 | ⚠️ 备选（§1.6.3 实验裁决；GM 往返多一次） |

**R2 在线算法**：**结论 = 分段明确**（有明确结论 + 结构依据）：

- **行内规约（sumsq_h/sumsq_k/dot/sumsq_v）：无在线化收益**。4 处 `T.reduce_sum` 分别作用于 `H`、`k`、`(H⊙k)`、`vhat` 四个**不同量**，每次均为单遍 `(R,d)→(R,1)`；每个输入张量只读一次，**不存在可被「单遍/流式」消去的多遍基线**（结构依据；源 §0.3 步骤 2/3/6/8 为四个独立行归约）。
- **因果深度卷积：存在 block 级「流式/halo」变体**（本设计采纳）——沿 seq 轴为因果滑窗，每 block 维护 3 行历史（halo）即可 O(1) 状态推进；收益口径 = 扫描遍数（conv 邻居从「每输出 token 4 次 GM 读」降为「每 block 一次 halo 重算」）、中间缓冲 O(BLK_S) UB 驻留。**源算法已用「GM 缓存邻居行」的类在线形式**（`vhat_buf` + 运行时行索引），是本变体存在的直接证据（§0.4 #8）。
- 归约维度 `d` 本身不可在线化（需完整行才能出 `rrms`），但每 token 独立、单遍即可。

**R3 复杂度对比**（四口径；本算子**访存 + 发射混合受限**，权重访存与向量 op 数均为主导项；代表 workload W3 = `M2,S64,d512,fp16`）：

| 算法候选 | FLOPs（manifest 口径） | 访存量 (Bytes) | 扫描遍数（GM 读） | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------------|---------------|------------------|-------------|---------------------|
| 基线（源：2 Pass，token-per-block） | `M·S·(24d) + 20MS` ≈ 3.15 M（W3） | read ≈ `7·M·S·d·elem` = **0.92 MB** + write `2·M·S·d·elem` = 0.26 MB（W3 ≈ **1.18 MB**；vhat 被 4× 重读） | 输入 1×，`vhat` **4×** | 无 GM workspace；fragment 12 个 | grid `2·M·S`（W3=256 blocks）；无跨核 |
| **#1 序列分块单 kernel（采纳）** | 同基线 FLOPs | read ≈ `3·(R/BLK_S)·M·S·d·elem` = **0.52 MB** + 权重 + write 0.26 MB（W3 ≈ **0.80 MB**，BLK_S=8 → `R/BLK_S=1.375`） | 输入 **1.375×**，`vhat` 1×（不落 GM 复用） | `3~4 × (R,d) f32` UB ≈ 76 KB（BLK_S=8,d=512） | grid `M·num_tiles`（W3=16 blocks）；无跨核 |
| #2 rsqrt 绕法 | 每 Norm +1 向量 op | 不变 | 不变 | +2 `(R,1)` f32 | 同 |
| #3 常数倒数折叠 | −（标量 op 型替换） | 不变 | 不变 | 不变 | 同 |
| #4 CSE v_hat | −1 向量 op（省重算） | 不变 | 不变 | +1 `(R,d) f32` | 同 |
| #5 门控点积因子化 | −3 向量 op（省 h_n/k_n 物化） | 不变 | 不变 | −2 `(R,d) f32`（省两 buffer） | 同 |
| #7 conv→GEMM | — | — | — | — | ❌ 不适用（depthwise，无权重矩阵） |
| #10 两 Pass 序列分块（备选） | 同 | read `3·1× + 1× + (R/BLK_S)×` ≈ 0.68 MB + write 0.26 MB（W3 ≈ **0.94 MB**） | pass2 `vhat_norm` `(R/BLK_S)×` | 同 | 同；多 1 次 kernel 启动 + vhat_norm GM 落盘 |

> 口径说明：**FLOPs** 取 manifest `M·S·(24d)+20MS`（乘加=2，含 3 次 norm 的平方和、点积、卷积 tap）；**访存量** = GM 读写总字节（输入读 + 输出写 + conv 邻居读；分母是不可消除的 I/O）；**扫描遍数** = 各张量 GM 读取次数；**中间缓冲峰值** = UB 驻留上限（见 §4.5）。W3 manifest `bytes` = 0.664 MB（理想一次读写）；实现口径的额外流量来自 causal halo 与权重。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 序列分块单 kernel（采纳）** | Vector（逐元素 + 行归约 + 超越函数，非 MAC 密集）✓ | UB 192KB：峰值 `3~4×(R,d) f32` ≈ 76KB（W3，×1.7 ≈ 129KB）< 192KB ✓（`CONST-capacity-910B2C` 容量值**绑 910B2C/9362、本设备 910B3 未重验**） | `d∈{256,512}` 均整除向量宽（fp16 ×8/fp32 ×4）✓；`(R,d)` 尾轴 32B 对齐 ✓ | 循环边界全静态（`BLK_S/R/K=4` 编译期常量）✓ | 向量链可 auto-multi-buffer 重叠；无 Cube → 非 MixCV、无 CV 同步 | ✅ **采纳** |
| 基线（源 2 Pass token-per-block） | Vector ✓ | UB ✓ | ✓ | ✓ | 2 次 grid 启动 + GM 往返 | ⚠️ 可用但 GM 流量高（§0.6 R1 重设计） |
| #8 归约用 Cube GEMV | **Cube 落错单元**（归约非 MAC 密集、无权重矩阵） | L1/L0C 需额外占用 | — | — | **CV 混排** | ❌ 弃选（无权重矩阵 + `CG-2026-0022`） |
| #7 conv→GEMM | Cube（depthwise 无共享权重） | — | — | — | — | ❌ 不适用（depthwise） |
| #2/#3/#4/#5/#6 | Vector（`vsqrt`/`vdiv`/`vmul`/`vbrc`/`vcast` 均 fp32 ✓、bf16 ✗→fp32 域） | +小 buffer | ✓ | ✓ | — | ✅ 采纳 |
| #10 两 Pass 序列分块（备选） | Vector ✓ | UB ✓ | ✓ | ✓ | 多 1 次 kernel 启动；`vhat_norm` GM 落盘 | ⚠️ 备选（§1.6.3） |

**v-prefix 元算 dtype 矩阵（决定 fp32 中转 + 排除候选）**：`vadd`/`vmul`/`vdiv`/`vsqrt`/`vsigmoid`、`reduce_sum` **均不含 bf16**（各 op doc §2.2.1：fp16 ✓ / fp32 ✓ / bf16 ✗/×）；`vbrc` ✓（含 bf16，`shape操作/T.vbrc.md` §2.2.1）、`vcast` 支持 `f16→f32`/`bf16→f32`（`rint`）与 `f32→f16/bf16`（`数据类型转换操作/T.vcast.md` §2.2.1）；`reduce_sum` 支持 fp16/fp32 + 整型（`规约操作/T.reduce_sum.md` §2.2.1）。→ 全部向量数学一律 fp32 域，bf16 为 API 强制中转。

**调研结论**：**选定「纯 Vector 序列分块单 kernel（每 block `BLK_S` token + 3-halo，UB 内因果滑窗）+ fp32 中转计算域 + `vsqrt/vdiv` rsqrt 绕法 + 常数倒数折叠 + CSE 物化 vhat + 门控点积因子化 + 算术行掩码边界」**（候选 #1/#2/#3/#4/#5/#6 采纳）。

关键依据：① **R3/R4**：本算子纯逐元素/行归约，无 MAC 密集段，向量单元正确；访存 + 发射混合受限，`#1` 相对基线把 `vhat` 的 4× GM 重读消除（W3 流量 1.18→0.80 MB）；② **R2**：4 处 reduce 作用于 4 个不同量、无多遍基线，在线化无收益（结构依据）；conv 的 block 级 halo 滑窗为在线形式；③ **R1**：候选 #7/#8/#9 的弃选理由分别为数学不成立/无权重矩阵 + 能力缺口/无上溢问题；④ 全部采纳项（`vcast`、`vmul`、`vadd`、`vsqrt`、`vdiv`、`vsigmoid`、`vbrc`、`reduce_sum`、`copy`）均有本仓 `docs/Tilelang.language/` 佐证 + `examples/` 实跑先例（§3.4）。

**调研范围（「无更优替代」依据）**：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md`（命中 `ALG-rmsnorm`〔rsqrt 绕法 + 行归约基准形态〕、`ALG-conv`〔im2col/implicit GEMM/Winograd——本算子 depthwise 逐通道、无共享权重，均不成立〕、`ALG-reduction`〔row-reduction 迁移基准形态〕、`ALG-elementwise`〔单遍 Vector 流〕）；② 本仓 `examples/`：`engram/engram_fwd.py`（同算子 NPU 参照——两 kernel/token-per-block），`engram_decode/_engram_decode_kernel/DESIGN.md`（同族迁移范式），`TileOPs/.../kernels/reduction/`、`examples/norm/`（行归约族）；③ pattern-library（预注入 + `kb_search` 消费）：`CASE-engram-decode-migration`（status=verified）、`CG-2026-0022`（Cube+Vector MIX 缺口，见 §1.6.3）、`TRAP-vrsqrt-plain-precision`（status=verified，rsqrt 绕法）、`TRAP-npuir-vec-scalar-broadcast`（status=verified，标量广播写法）、`TRAP-npuir-fp32-vector-reduce`（status=verified，fp32 GM 输入路径）、`TRAP-C13-parser-no-python-unroll`（tap 展开）、`TRAP-conditional-alloc`（顶层分配）、`TRAP-UB-multibuffer-inflation`（UB 膨胀）、`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-ascend910-9362-profile`（D-2 常数）；④ **外部已知算法族**（模型知识）：因果深度卷积的直接法/implicit GEMM/状态缓存三类实现、RMSNorm 的 rsqrt/Newton 迭代、门控点积的因子化——逐项由 §1.6.3 枚举与本表评估；⑤ `docs/Tilelang.language/` 全目录枚举（含 `规约操作/`、`数学操作/`、`数据类型转换操作/`、`shape操作/T.vbrc.md`、`内存操作/T.copy.md`）确认采纳 API 存在性。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)）**：

- **流量项** = `GM bytes ÷ 有效带宽`。W3（`M2,S64,d512,fp16`）实现口径 GM 流量 ≈ **0.80 MB**（§1.6.3 R3 表；含 halo 1.375× 与权重）。**本设备为 `Ascend910B3`（实查 20 AI Core / 40 Vector Core），`pattern-library/constants.md` 无 910B3 条目**，取 `CONST-ascend910-9362-profile`（9362 单 die 核→GM ≈ **1.06 TB/s**，status=verified）作分母并标注**跨 SoC 未重验（未文档化假设 + 估算依据）**：流量项 ≈ `0.80MB / 1.06TB/s ≈ 0.75 µs`。
- **发射项** = `向量 op 数 × ~0.5 µs/op`（`CONST-vector-launch-overhead`，**绑 910B2C 且 `kb_stale_check` 标 stale，910B3 未重验 → 未文档化假设 + 估算依据**）。W3 每 block 向量 op 数 ≈ **46**（载入 3 + 平方/积 3 + reduce 4 + sqrt/div 6 + 门控 2 + vhat 2 + vhat_norm 4 + 行掩码 1 + conv clear/4 tap×2 = 9 + SiLU 3 + 输出 cast/copy 4 + 广播 3）：16 blocks（1 波次）→ 发射项 ≈ **23 µs**。
- **容量项** = 无（中间量全 UB 驻留，无跨引擎 GM 往返；`CONST-store-fixpipe-gm-only` 不适用）。

**设计期估算（D-2 roofline）**：代表 workload W3（`M2,S64,d512,fp16`）估算下界 ≈ **max(0.75, 23, 0) ≈ 23 µs（发射项主导）**；W1 smoke（`M1,S16,d256`，2 blocks）≈ **23 µs（发射项主导）**。**不确定性声明**：① `0.5µs/op` 与 `1.06 TB/s` 均绑 910B2C/9362、`kb_stale_check` 标 stale，910B3 未重验，发射项可能**上偏**；② 对照证据：同族短链逐元素算子在实测中常由流量项主导、发射项被 MTE2 窗口隐藏（`bitwise_not` 1M 实测 7.72µs ≈ 流量）——本算子数据量极小（≤0.8MB），发射项或系统性上偏；③ **`BLK_S` 是 Stage 4 发射/流量调优杠杆**；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（纯 Vector 序列分块 + fp32 中转 + rsqrt 绕法）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| O1 | **rsqrt 绕法** | `rrms = rsqrt(s)`（设备为近似指令） | `t = vsqrt(s)`；`rrms = t/s`（`vsqrt`+标量除） | 实数恒等 `sqrt(s)/s ≡ s^{-1/2}`（s>0；本算子 `s = Σx²/d + eps ≥ eps = 1e-6`）；D-1 O1 vs 真值 fp32 max_abs 6.1e-5、违反 0 → EQUIV_PASS；**精度优于基线设备指令**（2.9e-3 → 1.07e-7） | 每 Norm 1 op 型替换（`vrsqrt`→`vsqrt`+标量除，+1 op）；消除 3 次 Norm 的 ~2.9e-3 相对误差累积 |
| O2 | **编译期常数倒数折叠** | `Σx²/d`；`dot/√d` 逐元素除法 | `Σx²·fl(1/d)`；`dot·fl(1/√d)`（host 期 Python 常量） | 容差内等价；D-1 O2 四行：`x·fl(1/d)` 与 `x/d` **逐位一致**（d=256/512 为 2 的幂，`fl(1/d)` 精确）；`dot·fl(1/√d)` vs `dot/√d` max_abs 2.4e-7，违反 0 → EQUIV_PASS | `vdiv`→`vmul`（op 型替换）；`·(1/d)` 与 `+eps` 合成一步 |
| O3 | **CSE 物化 `vhat`** | 源 pass2 经 GM 重读 `vhat_buf` 并重算 `normed` | 单 kernel 内物化 `vhat = alpha⊙v`（fp32 `(R,d)`），共用于 `vn` 与残差 | 同一确定性 fp32 计算，物化值与重算值**逐位一致**；D-1 O3 两 dtype `max_abs=0.0` → EQUIV_PASS | 消除源 pass2 的 4× GM 行重读；代价 +1 UB tile |
| O4 | **门控点积因子化**（§0.6 R5） | `h_n=H·rrms_h·w_h`；`k_n=k·rrms_k·w_h`；`dot=Σ(h_n⊙k_n)`（5 vmul + reduce） | `dot=(rrms_h·rrms_k)·Σ(H⊙k⊙w_h²)`（2 vmul + reduce） | 实数恒等（`h_n⊙k_n = (rrms_h·rrms_k)·(H⊙k⊙w_h²)`，标量外提）；D-1 O4 三 dtype：与 fp32 原式 max_abs ≤1.6e-5、与 fp64（同输入）max_abs ≤2.4e-5，违反 0 → EQUIV_PASS | −3 向量 op/tile；省 2 个 `(R,d)` buffer |
| O5 | **fp32 中转计算域**（保留源 + API 强制） | 概念上「输入 dtype 域直译」 | `vcast(rint)↑f32 → fp32 向量链 → vcast(rint)↓dtype` | 与源（`accum_dtype="float"`）及契约 golden（`.float()` 全链路）**同域**；bf16 为 API 强制（v-op 无 bf16）；D-1 FULL 三 workload `Y` 均 EQUIV_PASS | 正确性：bf16 可编译；代价 +2 `vcast`/张量 |

**优化结论**：采纳 **5 项**（O1 rsqrt 绕法、O2 常数倒数折叠、O3 CSE vhat、O4 门控点积因子化、O5 fp32 中转）。**优化后公式**（供 §3.1 使用）：
- RMSNorm：`s = Σx²·fl(1/d) + eps`；`t = vsqrt(s)`；`rrms = t/s`；`x_n = x·rrms_brc⊙w_brc`
- 门控：`dot = (rrms_h·rrms_k)·Σ_d(H⊙k⊙w_h²)`；`alpha = vsigmoid(dot·fl(1/√d))`；`vhat = alpha_brc⊙v`（物化）
- 卷积：`c_o[i,:] = Σ_{p=0}^{3} cw[p,:]⊙vn[i+p,:]`（tap 编译期展开，行掩码清零无效 halo）
- SiLU：`Y = c_o⊙vsigmoid(c_o) + vhat`，末端 `vcast(rint)`

公式层面已无进一步计算量/访存量可优化空间：**中间量全部 UB 驻留**（无中间 GM 往返，除输出契约），**4 个 reduce 非冗余**（不同数据），**conv 为 depthwise 无权重矩阵**（#7/#8 不成立）。

**等价性机器验证（D-1）**：脚本 `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/verify_equiv.py`（CPU；候选式 = NPU fp32 流水线逐 op 复现；基线式 = fp64 精确 / 源式；判定 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`，fp32 `1e-4/1e-4`、fp16 `1e-1/1e-1`、bf16 `2e-1/1e-1`——取自 `test_engram_gate_conv_fwd.py`）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**TOTAL 22 ITEMS, 22 EQUIV_PASS, 0 失败项**）：

| 采纳项 / 对照 | 候选式 | 基线式 | dtype | samples | 最大 abs 差 | 违反率 | 违反数 | 结论 |
|---------------|--------|--------|-------|---------|------------|--------|--------|------|
| O1 rsqrt=sqrt/s | `sqrt_f32(s)/s` | `1/sqrt(s) [fp64]` | fp32 | 1,048,576 | 6.104e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 x/d → x·fl(1/d) (d=256) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 dot/√d → dot·fl(1/√d) (d=256) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 x/d → x·fl(1/d) (d=512) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 dot/√d → dot·fl(1/√d) (d=512) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 2.384e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE v_hat reuse | `vhat (materialised)` | `alpha*v (recomputed)` | fp16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE v_hat reuse | `vhat (materialised)` | `alpha*v (recomputed)` | bf16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 gate-dot factorise (d=256) | `rrms_h*rrms_k*Σ(H*k*w²)` | `Σ(h_n*k_n) [fp32]` | fp16 | 4,096 | 1.144e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 factor vs fp64 (d=256) | `fp32 factorised` | `fp64 exact (same inputs)` | bf16 | 4,096 | 1.399e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 gate-dot factorise (d=512) | `rrms_h*rrms_k*Σ(H*k*w²)` | `Σ(h_n*k_n) [fp32]` | fp16 | 4,096 | 1.526e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 factor vs fp64 (d=512) | `fp32 factorised` | `fp64 exact (same inputs)` | fp32 | 4,096 | 1.941e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 FULL Y (`M1 S32 d256 fp16`) | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 8,192 | 3.052e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 FULL Y (`M2 S64 d512 fp16`) | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 65,536 | 3.052e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 FULL Y (`M1 S128 d256 bf16`) | `fp32 NPU pipeline` | `fp64 golden` | bf16 | 32,768 | 3.052e-05 | 0.000000 | 0 | **EQUIV_PASS** |

> 判定准则 = `|cand−base| ≤ atol + rtol·|base|`，违反数 0 即 PASS；覆盖随机张量 + 角点（rsqrt 域 1e-6~3.4e38 含次正规邻域、有符号全域 ±1e6/0/±1e-8、极值 scale）。完整 22 行见脚本输出（同族 O1/O2/O3/O4 × {d=256,512} × {fp16,bf16,fp32} 亦全 PASS）。**注**：O4 的「vs fp64」基线用**同一批已舍入输入**构造（隔离 fp32 路径误差与输入 cast 效应）——首版用未舍入输入对照曾对 bf16 报伪 FAIL（输入 cast 效应），已修正。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 4 处行内归约（`sumsq_h/sumsq_k/dot/sumsq_v`） | 源 `hsq_2d` 等 2D fragment + `T.reduce_sum(dim=1)` | `T.reduce_sum` on UB（`(R,d_padded) f32 → (R,1)`，佐证 `docs/Tilelang.language/规约操作/T.reduce_sum.md`；先例 `examples/engram/engram_fwd.py` L64） | ✅ | — |
| 2 | 逐元素链（cast/平方/乘/加/`rsqrt`/`sigmoid`/掩码） | 源 `T.Parallel(d_padded)` 标量逐元素 | `T.vcast`/`T.vmul`/`T.vadd`/`T.vsqrt`/`T.vdiv`/`T.vsigmoid`（佐证各 op doc §2.4；先例 `examples/engram/engram_fwd.py`） | ✅ | — |
| 3 | **载入/写回的行索引** | 源 `H[bid,tid,j]`（token 固定 → 静态） | 本设计为 `H[bid, gt, j]`（`gt = t0-3+i`，运行时行索引）——`T.Parallel` 元素读 + clamp | ✅ | —（探针 V1 实证动态行索引读合法；探针 V3 实证 `T.copy` 运行时行索引合法） |
| 4 | **因果卷积窗口** | 源 `T.serial(4)` + 逐 tap 从 GM 读邻居行（`src_t = tid-3+p`） | UB 内仿射行偏移读 `vn[i+p, :]`（探针 A 实证）+ `cw[p,:]` `T.vbrc` 广播 + `T.vmul` | ✅ | — |
| 5 | **标量广播到 d（`vhat=α⊙v` / `x⊙rrms⊙w`）** | 源标量 `alpha_val` 逐元素乘（`T.Parallel`） | `T.vbrc`（把 `(R,1)` 行标量广播到 `(R,d_padded)`）+ `T.vmul`（佐证 `shape操作/T.vbrc.md` §2.2.2；`≥2` 向量操作数形态规避 `TRAP-npuir-vec-scalar-broadcast`） | ✅ | — |
| 6 | **门控标量 `σ(dot·INVSQRT_D)`** | `dot_hk` 标量式 | `T.vsigmoid`（作用 `(R,1)` buffer；佐证 `数学操作/T.sigmoid.md`）+ 常数乘 | ✅ | — |
| 7 | **因果边界的行掩码构造** | 源 `T.if_then_else(src_t≥0, 载入, 0)` | 索引 `T.max/T.min` 标量 clamp + 标量掩码 `rmask`（`T.Parallel(R,1)` 内标量算术） | ✅ | —（`T.if_then_else` 带 buffer 操作数在 npuir codegen 段错误——探针 V2，故改用算术掩码） |
| 8 | conv tap 展开（p=0..3） | 源 `T.serial(CONV_KERNEL_SIZE)` + 运行时 `src_t` | host 层 Python 展开（`p` 为编译期常量偏移；`conv_t[i+p]` 静态偏移） | ✅（host 展开） | —（`TRAP-C13`：prim_func 内 `for p in range()` 是运行时 Var，故在 factory 层展开） |
| 9 | 核级任务映射 | 源 `T.Kernel(seq_len, M)` + `threads` | 一维 `T.Kernel(M*num_tiles, is_npu=True)` | ❌ | block 级任务映射（每 block 少量标量）；`threads` 在 npuir 无效（`TRAP-threads-kwarg-noop`），无逐元素等价 API；元素级并行由 v-prefix 整 buffer 完成 |
| 10 | host 侧 pad/crop（尾维 pad、输出 crop） | `F.pad`/切片 | 保留在 host（元数据/视图级） | ❌ | host 侧 shape/stride 元数据处理（不在 kernel 内）；`ascend-constraints.md` §4 允许 stride/shape 视图操作、禁改输入内容——本算子 host 仅 pad/crop |

**向量化结论**：逐元素/归约/搬运/窗口计算点（#1–#8）**已全部向量化或 host 层展开**（替代 API 均有 docs 佐证 + `examples/` 实跑先例）；保留 2 处非逐元素点（#9 核级任务映射、#10 host 元数据），逐项理由如上，**无「实现简单」类理由**。与 §6 循环结构一致——kernel 内**无逐元素标量循环**；唯一串行结构为 conv 的 4-tap 编译期展开（#8）与 block 级任务映射（#9）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**候选枚举**（本算子的非 batch 维 = `seq_len`（卷积/任务轴）与 `d`（行内规约/逐元素轴）；I/O 布局 = 行主序、尾轴 `d` 连续，是契约；核内布局是设计变量）：

| 候选 | 向量化轴 / 布局 | 轴质量（整除/尾 lane 浪费） | repack 代价 | UB 预算 | 结论 |
|------|----------------|---------------------------|-----------|---------|------|
| **A（主选）** | **核内 = I/O 布局**；向量化轴 = 尾轴 `d`（连续）；`seq_len` 轴作 block 级分块（每 block `BLK_S` token + 3-halo）；行内归约沿 `d` 水平 reduce（`reduce_sum dim=1`） | `d∈{256,512}` 整除向量宽（fp16 ×8 / fp32 ×4）→ 尾 lane 浪费 0%；`(R,d)` 尾轴 32B 对齐 ✓ | 无（I/O 布局 = 核内布局） | `3~4 ×(R,d) f32` ≈ 76KB（W3, BLK_S=8）→ ×1.7 ≈ 129KB < 192KB ✓ | ✅ **主选** |
| B（备选，实验裁决） | **两 Pass 序列分块**：Pass1 写 `vhat_norm` → GM；Pass2 分块载入 `(R,d)` halo（clip 到 `[0,seq_len)`）→ 同一 UB 内卷积 | 同 A | 无（同布局） | 同 A | ⚠️ 备选（结构简单、每步有先例；多 1 次 kernel 启动 + `vhat_norm` GM 落盘约 0.13MB/次） |
| C（弃选） | 核内 `T.transpose` 把 `seq_len` 变尾轴（让卷积沿连续轴） | `seq_len∈{16..128}` 长度尚可 | `T.transpose` 二轴交换链（UB 级，`PL-1.1` 实测 ~5.4µs/链）；`(R,d)`→`(d,R)` | 转置中间 tile 翻倍 | ❌ 弃选：**无收益**——卷积沿 seq 轴是滑窗（跨步访问），即便转置后沿连续轴仍需 4 次移位读；且 repack 引入 UB 级交换链开销与容量。**非 dtype 障碍**（`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 逐格取值：fp16/fp32/bf16 均 √） |
| D（弃选） | host permute 把 `seq_len`/`d` 换序 | 同 C | host permute（`PL-1.3` 实测 106–147µs） | — | ❌ 弃选：host permute 净亏（`PL-1.3` status=verified），且改张量内容违反 host 契约 |
| E（弃选） | 垂直扫描归约（lane 映射独立行、串行步进 `d`） | — | — | — | ❌ 弃选：核内归约维 = 尾轴 `d`，水平归约（`reduce_sum` 1 op）优于垂直扫描（O(d) 次串行）；先例 `examples/engram/engram_fwd.py` 用水平归约 |
| F（弃选） | token-per-block（`BLK_S=1`，grid=`M·seq_len`） | 同 A | 无 | 峰值仅 `3×(4,d)`（更小） | ❌ 弃选：并行度最高（W3=128 blocks）但**每输出 token 需重算 4 个 halo 行**（有 4× 冗余计算）且 conv 需从 GM 或 UB 移位重读；相对 A 无收益（A 的 halo 仅 3/(BLK_S+3) 冗余） |

**弃选论证证据**：候选 C 依赖的 `T.transpose` 已枚举并按 `negative-claim-evidence.md` §2.2 在 `docs/Tilelang.language/` 子目录检索（命中 `创建操作/T.transpose.md`，§2.2.1 Ascend dtype 行**逐格实读**：fp16/fp32/bf16 均 √；§2.3 限制为仅二轴交换）——该候选**弃选不依赖 dtype 负向断言**，仅依据「滑窗跨步 + repack 开销无收益」；候选 D 引用 `PL-1.3-host-permute`（status=verified）；候选 E 为结构分析（水平 vs 垂直）；候选 F 为冗余度分析（4× vs 1.375×）。**A–F 全行负向论断已逐条复核**。

**MIX（Cube + Vector）可行性评估** ⭐（预注入要求）：

- **本算子是否属于 MIX（Cube `T.gemm` + Vector 混合）结构？——否。** 源算子 `engram_fwd.py` **无任何 `T.gemm`/matmul**（投影 `k=E·W_K`、`v=E·W_V` 在外部 GEMM 完成；本 kernel 只剩 3×RMSNorm + 门控 + 深度卷积 + SiLU）。4 处规约是**逐通道自身点积/平方和**（`Σ H²`、`Σ k²`、`Σ H⊙k`、`Σ vhat²`），**无权重矩阵**，不是 GEMM；把其表达为 Cube GEMV 需构造 d×d 算子，数学上不成立（候选 #8 弃选）。故本设计**纯 Vector**（`T.Kernel(..., is_npu=True)` + v-prefix + `reduce_sum`），**非 MixCV、无 CV 同步**，与 `examples/engram/engram_fwd.py` 同路线。
- **`CG-2026-0022` 影响评估**：该缺口 = 「Developer auto CV-split 下 Cube `T.gemm` 与 shared 向量规约 + 标量广播链**无法共存**」（BiShengIR verifier 三类拒绝：`cbuf to cc` / `load src==gm` / `expected result type with stride`），**仅在同时含 `T.gemm` 与 shared 向量规约广播时触发**。本设计**无 Cube 段 → 不触发该缺口**（本任务探针 A/V1/V3 实证纯 Vector 结构可编译运行）。**附带缺陷**（`TRAP-npuir-fp32-vector-reduce`：GM 输入张量本身为 fp32 时纯向量 shared 规约 + 标量广播被拒，fp16 经 `vcast` 升 fp32 通过）**亦不触发**——本算子输入为 fp16/bf16（manifest 契约），fp32 仅在片上 UB 中转（§0.6 R2）。**平台 `examples/engram/engram_fwd.py` 的存在即为同算子纯 Vector 路线可行的先例。**
- **若强行选 Cube 路线的降级方案**：仅当未来把外部投影 GEMM 也融合进来时才会成为 MIX；届时须先做 `CG-2026-0022` 的**最小 MIX 探针**（gemm + 归约 + 广播），失败则回退本设计的纯 Vector 路线。**当前不需要**。

**设计期探针（D-3，实测本设备/工具链，脚本落 `history_version/`）**：本设计的两个关键不确定构造已用最小探针实测（未实现完整算子，仅最小张量）：

| 探针 | 构造 | 实测结果 | 结论 |
|------|------|---------|------|
| A（`probe_a_shift.py` / `probe_c_bisect_dynrow.py` V3-V4） | UB 内**仿射行偏移读** `xt[i+p, j]`（p 编译期常量） | **编译 + 运行成功**（shifted read `max_abs` = 1 ulp fp16） | ✅ 主选 conv 形态可行 |
| B（`probe_b_dynrow.py` / V2） | `T.Parallel` 内**带 buffer 操作数的 `T.if_then_else`** + 动态行索引 | **段错误 EXIT=139，无诊断** | ❌ 禁用；改用算术行掩码（§0.6 R4） |
| V1（`probe_c_bisect_dynrow.py` V1） | `T.Parallel` 内**动态行索引**读 GM（clamp，无 `if_then_else`） | **编译 + 运行成功** | ✅ 分块 tile 载入可行 |
| V3 | `T.copy` 运行时行索引（新 1D buffer） | **编译 + 运行成功**（decode 已用同形态） | ✅ 备选载入形态可行 |

> 工具链戳同上。探针结论已同步为本设计决策依据（§0.6 R4 / §3.3 / §6）。**注**：探针 A 首版直接 `O[i,j]=T.cast(...)` 写 GM 时被 MLIR verifier 拒（`hivm.hir.vcast` result 须为 ranked tensor，实为 memref）——**store 处的 direct-cast 不可用，须先 `T.vcast` 到 dtype UB 再 `T.copy` 落 GM**（与 decode 的 `T.vcast`+`T.copy` 一致），已采纳入 §3.2。

**实验裁决三件套（判定裕度落在未实证常数内 → 强制）**：

- **裁决对象**：候选 A（单 kernel 序列分块，主选）vs 候选 B（两 Pass 序列分块，备选）。判定裕度依赖的未实证常数：910B3 的**向量发射成本**（`CONST-vector-launch-overhead` 绑 910B2C 且 stale）与**核→GM 带宽**（`CONST-ascend910-9362-profile` 绑 9362）——A 省一次 kernel 启动与 `vhat_norm` GM 落盘（约 0.13MB/次），B 结构更简、每步有先例。
- **主选方案（默认实现，进 Stage 3）**：候选 A——单 `@tilelang.jit` kernel、一维 grid `M·num_tiles`、`BLK_S=8`、`R=11`；结构见 §3.3 / §4 / §6。
- **备选方案（≤1，结构完整，达 Stage 4 可直接实现）**：候选 B（两 Pass 序列分块）——
  - **常驻 buffer（全列出，@d=512, BLK_S=8）**：Pass1（门控/norm）：`h_t/k_t/tmp_t/vhat_t (11,512) f32` = 3×22.5KB = **67.5KB** + 权重 (4×512×4 ≈ 8KB) + 归约 `(11,1) f32` ≈ 0.2KB；写 `vhat/src/rrms_*` 到 GM。Pass2（卷积）：`vn_t (11,512) f32` = 22.5KB + `conv_t (8,512) f32` = 16KB + 权重 8KB ≈ 46.5KB。两 Pass 各自独立编译、UB 峰值均 < 80KB（×1.7 < 136KB）✓。
  - 循环：Pass1 `T.copy(H[bid, t0-3+0 : t0-3+R, 0:D], h_t)`（clip 行界）→ fp32 链 → 写 `vhat`/`rrms_*`/`vhat_norm`（workspace）；Pass2 `T.copy(vhat_norm_ws[bid, t0-3 : t0-3+R, 0:D], vn_t)` → 4-tap 卷积 → 写 `Y`。**建议 `vhat_norm` workspace 由 host 以 `torch.empty` 分配传入（显式输出参数，规避 `out_idx` 多输出 + workspace 组合风险，`TRAP-zero-input-crash` 经验）**。
  - 分核三要素：grid `M·ceil(seq_len/BLK_S)`（同主选）；物理核 = **40**（纯 Vector 翻倍）；W3=16 ≤ 40（或 d=256 档 ≤ 32）→ 无需适配。
  - 该变体与主选数学等价（§1.6.1 全项复用；FULL 行覆盖）；差异仅为结构（多一次 kernel 启动 + `vhat_norm` GM 往返）。**保留意义 = 主选若在 Stage 3 遇到 UB 溢出 / auto-multi-buffer 支配性问题时的正确性兜底**。
  - **Stage 4 UB 编译探针计划（强制；B 被裁决选中或以兜底启用时执行）**：① 上靶前 `tilelang.compile(main, target="npuir")` 编译 smoke 与 W3 两档，读 `ub overflow, requires N bits` 报文取精确实测需求（`TRAP-UB-multibuffer-inflation` / `CG-2026-0008`：手工逐 buffer 预算 ≠ 可编译证明）；② 若实测需求超手工预算 ×1.12，按 `BLK_S=8→4`（`R=7`，主 tile 线性缩小）或合并 `h_t/k_t` 生命周期回收；③ 探针结果回写本节（`perf_feedback.md` 路径）。
- **裁决计划**：代表 shape = 全部 3 个 full workload + smoke；指标 = `msprof op` Task Duration（median-of-20）；未知常数反解 = 用 A/B 实测时长反推 910B3 的有效 per-op 发射成本与核→GM 带宽；**判定阈值**：备选 B latency < 主选 A × (1−0.10) 才翻转（10% 噪声余量，`BP_run_state_bimodality` ±3–5%）；裁决结果回写 §1.6.3。
- **shape 特化分派可行性**：工厂 `lru_cache` 按 `(M,seq_len,d,eps,dtype)` 特化，**支持按 shape 分派主选/备选 / 按 shape 分派 `BLK_S`**（如 d=256 走 `BLK_S=16`、d=512 走 `BLK_S=8`），裁决须评估**按 workload 分派**而非全局二选一。

**决策落地约束**：主选方案的向量化轴 = 尾轴 `d`，归约维 = `d`（水平归约）；该轴同步写入 §3.3 伪代码（各 v-op 作用于 `(R,d_padded)` buffer）与 §6 循环结构（无逐元素标量循环）。备选 B 的结构设计独立记录于本节（不与主选混排）。

**布局决策结论**：选定**主选 A**——核内布局 = I/O 行主序（**无 repack**）、向量化轴 = 尾轴 `d`、`seq_len` 轴 block 级分块（`BLK_S` token + 3-halo）、行内水平归约 `reduce_sum(dim=1)`、因果卷积 UB 内仿射滑窗。**弃选候选**：B（两 Pass，列为实验裁决备选）、C（核内 `T.transpose`，无 dtype 障碍但滑窗跨步 + repack 无收益）、D（host permute，106–147µs 净亏）、E（垂直扫描归约，劣于水平归约）、F（token-per-block，4× 冗余计算）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`ascend_mode="Developer"`；脚手架 wrapper `EngramGateConvFwdKernel.ascend_mode="Developer"` 已固定，与 `op_requirements.programming_mode=developer` 一致）。

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | **纯 Vector**：3×RMSNorm + 门控 + 深度卷积 + SiLU，无 `T.gemm`/matmul（投影在外部） | 仅需 UB；Developer `T.reduce_sum` on UB + v-prefix 足够，无需 Expert 手动 L1/L0C 控制 |
| 是否含归约 | 4 次行内 `reduce_sum`（H / k / (H⊙k) / vhat 不同数据） | Developer 的 `T.reduce_sum` on UB 足够 |
| 是否需流水线 | 向量链线性；分块 tile 间无依赖（halo 由 block 内重算覆盖） | 无需 `T.Pipelined`；auto-multi-buffer 处理片内重叠 |
| 向量链 API | `vcast/vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc/reduce_sum` | v-prefix（AGENTS.md 强制） |
| 同步需求 | 无跨核 workspace、无 Cube 段 | Developer 自动同步即可 |
| 是否 persistent | **否**（grid = `M·num_tiles` ≤ 40） | 保持非 persistent 简洁结构（规避 `TRAP-DEVMODE-PERSIST-GEMM` 类风险） |

决策树路径：`纯 Vector（无 matmul）→ 非 persistent → Developer（自动同步）→ Developer 模式`。**先例**：`examples/engram/engram_fwd.py`（同算子 Developer 纯 Vector）、`examples/norm/rms_norm.py`（Developer + `T.reduce_sum`）、`examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py`（Developer 纯 AIV 融合）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared((R, d_padded), "float32")`（UB 承载向量链与归约）；dtype 输出暂存 `T.alloc_shared((BLK_S, d_padded), dtype)` |
| 计算方式 | `T.Parallel`（载入/逐元素）+ v-prefix 整 buffer 向量链 + `T.reduce_sum(dim=1)` 行归约 |
| 同步 | Developer 自动同步 |
| 分核 | 一维 `T.Kernel(M*num_tiles, is_npu=True)`（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（fp32 中转 + rsqrt 绕法 + 常数倒数折叠 + CSE + 门控因子化）。窗口行索引 `i` 遍历 `R = BLK_S + 3` 行（全局 token `gt = t0-3+i`）；`·_brc` 表示经 `T.vbrc` 广播到 `(R,d_padded)` 的行标量。`INV_D = fl(1/d)`、`INV_SQRT_D = fl(1/√d)` 为 host 期 Python 常量。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `t0 = tile·BLK_S`；`gt = t0-3+i`；`gc = min(max(gt,0), seq_len-1)`；`rmk[i] = 1[0≤gt<seq_len]` | 窗口定位 + 索引 clamp + 行掩码标量 |
| 1 | `H_t[i,c] = f32(H[bid,gc,c])`；同 `k_t`、`v_t` | 载入窗口 tile（clamped 运行时行索引） |
| 2 | `s_h[i] = Σ_c H_t²·INV_D + eps`；`rrms_h[i] = sqrt(s_h)/s_h` | H 的 RMSNorm 统计量（O1/O2） |
| 3 | `s_k[i] = Σ_c k_t²·INV_D + eps`；`rrms_k[i] = sqrt(s_k)/s_k` | k 的 RMSNorm 统计量 |
| 4 | `dot[i] = (rrms_h[i]·rrms_k[i])·Σ_c(H_t·k_t·w_h[c]²)`；`alpha[i] = vsigmoid(dot[i]·INV_SQRT_D)` | 门控点积（O4 因子化）+ sigmoid |
| 5 | `vhat[i,c] = alpha_brc[i,c]·v_t[i,c]`（物化） | CSE（O3） |
| 6 | `s_v[i] = Σ_c vhat²·INV_D + eps`；`rrms_v[i] = sqrt(s_v)/s_v` | v_hat 的 RMSNorm 统计量 |
| 7 | `vn[i,c] = vhat·rrms_v_brc·w_v_brc·rmk_brc` | v_hat_norm（无效 halo 行清零） |
| 8 | `c_o[i,c] = Σ_{p=0}^{3} cw[p,c]·vn[i+p,c]`（i∈[0,BLK_S)） | 因果 DWConv1D（tap 编译期展开，UB 仿射滑窗） |
| 9 | `Y[bid,t0+i,c] = dtype(c_o·vsigmoid(c_o)[i,c] + vhat[i,c])`；`vhat_out = dtype(vhat)`；`alpha/rrms_*_out[bid,t0+i]` | SiLU + 残差 + 舍回；仅写 own 行 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 权重载入 | `w_h, w_v, cw[p] → f32` | `T.Parallel` 元素读 + `T.cast` | `[(0,j) in T.Parallel(d_padded)]: wh2[0,j] = T.cast(rms_w_h[j], "float32")`；`T.vmul(wh2,wh2,wh2)` 得 `w_h²` | Developer |
| 窗口定位/掩码 | `gc, rmk` | 标量 `T.max/T.min` + `T.cast` | `[(i,) in T.Parallel(R)]: rmk[i,0] = T.cast(T.min(T.max(gt+1,0),1)*T.min(T.max(seq_len-gt,0),1), "float32")` | Developer |
| 载入 tile | `H_t = f32(H[bid,gc,:])` | `T.Parallel(R, d_padded)` 元素读 + `T.cast` | `ht[i,j] = T.cast(H[bid, T.min(T.max(t0-3+i,0),seq_len-1), j], "float32")`（探针 V1） | Developer |
| 上转（buffer→buffer） | `cast32` | `T.vcast` | `T.vcast(src_dtype, dst_f32, round_mode="rint")`（`数据类型转换操作/T.vcast.md` §2.2.1：f16/bf16→f32 = rint） | Developer |
| 平方/积 | `x²`、`H·k·w_h²` | `T.vmul` | `T.vmul(ht, ht, prod)`；`T.vmul(ht, kt, prod)` 后 `T.vmul(prod, bro, prod)` | Developer |
| 行归约 | `Σ_c ·` | `T.reduce_sum` | `T.reduce_sum(prod, red, dim=1, clear=True)`（`(R,d_padded) f32 → (R,1)`；`规约操作/T.reduce_sum.md` §2.2.2） | Developer |
| 倒数 rms | `sqrt(s)/s` | `T.vsqrt` + 标量除 | `T.vsqrt(s_2d, t_2d)`；`rr[i,0] = t_2d[i,0]/s_2d[i,0]`（`数学操作/T.vsqrt.md`） | Developer |
| 广播行标量 | `a_brc` | `T.vbrc` | `T.vbrc(rr_2d, bro)`（`(R,1)→(R,d_padded)`，`shape操作/T.vbrc.md` §2.2.2；N>1 合法） | Developer |
| 逐元素乘/加 | `x·rrms⊙w` | `T.vmul` / `T.vadd` | `T.vmul(vt, bro, vn)`；`T.vmul(vn, bro, vn)`（in-place） | Developer |
| sigmoid | `σ(x)` | `T.vsigmoid` | `T.vsigmoid(red, red)`（α）；`T.vsigmoid(conv, sig)`（`数学操作/T.sigmoid.md`） | Developer |
| 卷积 tap | `Σ_p cw[p]⊙vn[i+p]` | `T.Parallel(BLK_S,d)` + `T.vbrc`+`T.vmul`+`+=` | host 展开 `for p in range(4)`；`T.vbrc(cwp_2d, cb)`；`conv[i,j] += cb[i,j]*vn[i+p,j]`（仿射偏移，探针 A） | Developer |
| 清零 | `c_o≡0` | `T.clear` | `T.clear(conv)`（`内存操作/T.clear.md`） | Developer |
| 下转 | `dtype(·)` | `T.vcast` | `T.vcast(conv, oy, round_mode="rint")`（f32→f16/bf16） | Developer |
| 写回 | `Y[bid,t0+i,:]=…` | `T.copy` | 先 `T.vcast` 到 dtype UB，再 `T.copy(oy, Y[bid, t0:t0+BLK_S, 0:d_padded])`（**禁止** store 处 direct-cast——探针 A 首版被 MLIR verifier 拒） | Developer |

### 3.3 计算伪代码

> 说明（依 design-template §3.3）：源 `T.Tensor` 签名已用**位置参数**传 dtype（`T.Tensor((M,seq_len,d_padded), dtype)`），NPU 实现保持该形式（不使用 `dtype=` 关键字）。伪代码为**主选方案（候选 A）**的完整结构；`BLK_S=8` 为默认（Stage-4 调优杠杆）。

```python
import functools
import math
import tilelang
import tilelang.language as T

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4                 # 卷积核长
HALO = CONV_KERNEL_SIZE - 1          # 因果历史 3 行
BLK_S = 8                            # 每 block 输出 token 数

@functools.lru_cache(maxsize=32)
def _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype):
    accum = "float32"
    d_padded = ((d + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    R = BLK_S + HALO                                     # 窗口行数 = BLK_S + 3
    num_tiles = (seq_len + BLK_S - 1) // BLK_S
    INV_D = 1.0 / float(d)
    INV_SQRT_D = 1.0 / math.sqrt(float(d))

    @tilelang.jit(out_idx=[6, 7, 8, 9, 10, 11], target="npuir")
    def _func():
        @T.prim_func
        def main(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            Y: T.Tensor((M, seq_len, d_padded), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            alpha_buf: T.Tensor((M, seq_len), accum),
            rrms_h_buf: T.Tensor((M, seq_len), accum),
            rrms_k_buf: T.Tensor((M, seq_len), accum),
            rrms_v_buf: T.Tensor((M, seq_len), accum),
        ):
            with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
                bid = cid % M
                tile = cid // M
                t0 = tile * BLK_S

                # ---- buffers: unconditionally allocated at Kernel top (TRAP-conditional-alloc) ----
                ht   = T.alloc_shared((R, d_padded), accum)        # H tile
                kt   = T.alloc_shared((R, d_padded), accum)        # k tile
                prod = T.alloc_shared((R, d_padded), accum)        # squares / product scratch
                vt   = T.alloc_shared((R, d_padded), accum)        # v_hat (materialised, CSE)
                vn   = T.alloc_shared((R, d_padded), accum)        # v_hat_norm
                conv = T.alloc_shared((BLK_S, d_padded), accum)
                sig  = T.alloc_shared((BLK_S, d_padded), accum)
                bro  = T.alloc_shared((R, d_padded), accum)        # row-broadcast scratch (R,d)
                cb   = T.alloc_shared((BLK_S, d_padded), accum)    # conv-weight broadcast scratch
                wh2  = T.alloc_shared((1, d_padded), accum)        # w_h^2
                wv2  = T.alloc_shared((1, d_padded), accum)        # w_v
                cwp  = T.alloc_shared((1, d_padded), accum)        # conv_w[p] row (f32)
                cwp_d= T.alloc_shared((1, d_padded), dtype)        # conv_w[p] row (dtype)
                red  = T.alloc_shared((R, 1), accum)               # reduce out / dot / alpha
                tmp1 = T.alloc_shared((R, 1), accum)               # sqrt temp
                rh   = T.alloc_shared((R, 1), accum)               # rrms_h
                rk   = T.alloc_shared((R, 1), accum)               # rrms_k
                rv   = T.alloc_shared((R, 1), accum)               # rrms_v
                rmk  = T.alloc_shared((R, 1), accum)               # causal row mask
                oy   = T.alloc_shared((BLK_S, d_padded), dtype)
                ov   = T.alloc_shared((BLK_S, d_padded), dtype)

                # ---- weights -> f32 (small) ----
                for j in T.Parallel(d_padded):
                    wh2[0, j] = T.cast(rms_w_h[j], accum)
                    wv2[0, j] = T.cast(rms_w_v[j], accum)
                T.vmul(wh2, wh2, wh2)                          # wh2 = w_h^2

                # ---- row mask (arithmetic; NO if_then_else: probe V2) ----
                for i in T.Parallel(R):
                    gt = t0 - HALO + i
                    lo = T.min(T.max(gt + 1, 0), 1)
                    hi = T.min(T.max(seq_len - gt, 0), 1)
                    rmk[i, 0] = T.cast(lo * hi, accum)

                # ---- load H / k tiles (clamped runtime row index: probe V1) ----
                for i, j in T.Parallel(R, d_padded):
                    gc = T.min(T.max(t0 - HALO + i, 0), seq_len - 1)
                    ht[i, j] = T.cast(H[bid, gc, j], accum)
                    kt[i, j] = T.cast(k[bid, gc, j], accum)

                # ---- sumsq_h -> rrms_h (vsqrt + scalar div = rsqrt bypass) ----
                T.vmul(ht, ht, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in T.Parallel(R):
                    red[i, 0] = red[i, 0] * INV_D + eps
                T.vsqrt(red, tmp1)
                for i in T.Parallel(R):
                    rh[i, 0] = tmp1[i, 0] / red[i, 0]
                for i in T.Parallel(BLK_S):
                    rrms_h_buf[bid, t0 + i] = rh[HALO + i, 0]

                # ---- sumsq_k -> rrms_k ----
                T.vmul(kt, kt, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in T.Parallel(R):
                    red[i, 0] = red[i, 0] * INV_D + eps
                T.vsqrt(red, tmp1)
                for i in T.Parallel(R):
                    rk[i, 0] = tmp1[i, 0] / red[i, 0]
                for i in T.Parallel(BLK_S):
                    rrms_k_buf[bid, t0 + i] = rk[HALO + i, 0]

                # ---- gate-dot factorise (O4): dot = rrms_h*rrms_k*sum(H*k*w_h^2) ----
                T.vmul(ht, kt, prod)
                T.vbrc(wh2, bro)                               # (1,d)->(R,d)
                T.vmul(prod, bro, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in T.Parallel(R):
                    red[i, 0] = red[i, 0] * (rh[i, 0] * rk[i, 0]) * INV_SQRT_D
                T.vsigmoid(red, red)                           # alpha
                for i in T.Parallel(BLK_S):
                    alpha_buf[bid, t0 + i] = red[HALO + i, 0]

                # ---- v_hat = alpha (.) v (materialised; O3) ----
                for i, j in T.Parallel(R, d_padded):
                    gc = T.min(T.max(t0 - HALO + i, 0), seq_len - 1)
                    vt[i, j] = T.cast(v[bid, gc, j], accum)
                T.vbrc(red, bro)                               # alpha -> (R,d)
                T.vmul(vt, bro, vt)

                # ---- sumsq_v -> rrms_v ----
                T.vmul(vt, vt, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)
                for i in T.Parallel(R):
                    red[i, 0] = red[i, 0] * INV_D + eps
                T.vsqrt(red, tmp1)
                for i in T.Parallel(R):
                    rv[i, 0] = tmp1[i, 0] / red[i, 0]
                for i in T.Parallel(BLK_S):
                    rrms_v_buf[bid, t0 + i] = rv[HALO + i, 0]

                # ---- vn = vhat * rrms_v * w_v * rmask ----
                T.vbrc(rv, bro); T.vmul(vt, bro, vn)
                T.vbrc(wv2, bro); T.vmul(vn, bro, vn)
                T.vbrc(rmk, bro); T.vmul(vn, bro, vn)

                # ---- causal DWConv1D (host-expanded taps; UB affine shift: probe A) ----
                T.clear(conv)
                for p in range(CONV_KERNEL_SIZE):              # host-level python unroll
                    T.copy(conv_w[p:p + 1, 0:d_padded], cwp_d)
                    T.vcast(cwp_d, cwp, round_mode="rint")
                    T.vbrc(cwp, cb)                            # (1,d)->(BLK_S,d)
                    for i, j in T.Parallel(BLK_S, d_padded):
                        conv[i, j] += cb[i, j] * vn[i + p, j]

                # ---- SiLU + residual -> outputs (own rows only) ----
                T.vsigmoid(conv, sig)
                for i, j in T.Parallel(BLK_S, d_padded):
                    conv[i, j] = conv[i, j] * sig[i, j] + vt[HALO + i, j]
                T.vcast(conv, oy, round_mode="rint")
                T.copy(oy, Y[bid, t0:t0 + BLK_S, 0:d_padded])
                for i, j in T.Parallel(BLK_S, d_padded):
                    ov[i, j] = T.cast(vt[HALO + i, j], dtype)
                T.copy(ov, vhat_buf[bid, t0:t0 + BLK_S, 0:d_padded])

        return main
    return _func
```

> **实现注意（Stage 3 输入，逐条与 §9 风险对应）**：
> 1. `T.if_then_else` **禁止**用于带 buffer 操作数的因果边界（探针 V2 段错误）——统一用索引 clamp + 算术行掩码。
> 2. store 处**禁止** direct-cast（探针 A 首版 MLIR verifier 拒：`hivm.hir.vcast` result 须为 ranked tensor）——先 `T.vcast` 到 dtype UB 再 `T.copy`。
> 3. 标量广播**禁止**裸 out-of-place 形态 `dst[j]=s*vec[j]`（`TRAP-npuir-vec-scalar-broadcast` 段错误，1 向量操作数）——统一 `T.vbrc` + `T.vmul`（≥2 向量操作数）；`T.vbrc` 源须为 `(R,1)`/`(1,d)` UB buffer。
> 4. conv tap `p` 必须在 **factory 层 Python 展开**（`for p in range(4)` 在 prim_func 内是运行时 Var，`TRAP-C13`）；`vn[i+p, j]` 用编译期常量偏移。
> 5. buffer 一律在 `T.Kernel` 顶层**无条件分配**（`TRAP-conditional-alloc`）。
> 6. 输出写回 `T.copy(oy, Y[bid, t0:t0+BLK_S, ...])`：manifest 各 workload `seq_len % BLK_S == 0`（16/32/64/128 均整除 8）→ 无尾块；通用非整除时取 `tail = T.min(BLK_S, seq_len - t0)` 并双侧裁剪切片（§5.4/§9.3）。
> 7. 伪代码中 `bro`/`cb`/`oy`/`ov` 等 scratch buffer 为**复用**候选（§4.5 给出保守预算，未做激进复用），Stage 3 可按实际编译报文进一步收紧。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|---------|
| `T.vcast(src,dst,round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint`；f32→f16/bf16） | 文档 + 探针 A |
| `T.vmul` / `T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md` §2.2.1（fp16/fp32；bf16 ✗） | 文档 + 探针 A |
| `T.vsqrt` | `docs/Tilelang.language/数学操作/T.vsqrt.md` §2.2.1（fp16/fp32） | 文档 + 先例 `examples/engram/engram_fwd.py` |
| `T.vdiv` | `docs/Tilelang.language/数学操作/T.vdiv.md` §2.3（dst 可 aliasing；src2 可为标量） | 文档 + 先例（decode） |
| `T.vsigmoid` | `docs/Tilelang.language/数学操作/T.sigmoid.md`（fp16/fp32；降 VAdd/VSub/VExp/VDiv） | 文档 + 先例 |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（`(R,1)→(R,N)`，N>1 合法；`(BM,1)→(BM,1)` 被拒 `TRAP-vbrc-empty-broadcast`） | 文档 |
| `T.reduce_sum(buf,out,dim=1,clear)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.2（同 rank，dst 归约维为 1；fp16/fp32） | 文档 + 先例 `examples/engram/engram_fwd.py` L64 |
| `T.copy` | `docs/Tilelang.language/内存操作/T.copy.md`；slice 形态正确（`TRAP-load-nd2nz-strided`） | 文档 + 探针 V3 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md`（UB 承载）；先例（decode / engram_fwd） | 文档 + 探针 A |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | 文档 + 先例 |
| `T.Kernel(..., is_npu=True)` | 先例 `examples/engram/engram_fwd.py` L38 | 文档 + 先例 |
| **动态行索引 `T.Parallel` 元素读** | 本任务探针 V1 | ✅ 实测（设备） |
| **UB 仿射行偏移读** | 本任务探针 A | ✅ 实测（设备） |
| **`T.if_then_else` 带 buffer 操作数** | 本任务探针 V2 | ❌ 实测段错误（禁用） |
| **store 处 direct-cast** | 本任务探针 A 首版 | ❌ 实测 MLIR verifier 拒（禁用） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（源为 2D grid `(seq_len,M)`） | 一维 `T.Kernel(M*num_tiles)`，`bid=cid%M`、`tile=cid//M`（§5.5） |
| GPU 专用 API（warp shuffle / cp.async / SMEM swizzle） | 否（源只用 `T.Parallel`/`reduce_sum`/`T.serial`） | 无需替换 |
| GEMM 非整除 / 分形限制 | 否（无 `T.gemm`） | 不涉及 |
| L0C 溢出 | 否（无 Cube） | 不涉及 |
| UB 192KB 容量 | 是 | §4.5 预算 + ×1.7 膨胀核对 + `BLK_S` 回退 |
| bf16 v-op 缺失 | 是（W4 bf16） | 全链 fp32 中转（§0.6 R2） |

#### 3.5.2 参考实现差异说明

> 迁移任务完整差异分析见 §0.5/§0.6；此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU `engram_fwd.py`） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------------------|----------------------|----------|
| Kernel 维度 | `T.Kernel(seq_len, M, threads=threads)`（2D） | 一维 `T.Kernel(..., is_npu=True)` | 展平为 `M*num_tiles`（§5.5） |
| 线程/元素并行 | `threads` + `T.Parallel` | `threads` 无效果（`TRAP-threads-kwarg-noop`）；v-prefix 整 buffer | 移除 `threads`；`T.Parallel` 保留为向量化载体 |
| 内存层级 | fragment/register（`T.alloc_fragment`） | UB（`T.alloc_shared`） | `T.alloc_shared` 承载全部中间量 |
| 倒数 rms | `T.rsqrt` | `T.vsqrt` + 标量除 | §0.6 R2 |
| 因果边界 | `T.if_then_else(src_t≥0,…)` | 索引 clamp + 算术行掩码 | §0.6 R4（探针 V2） |
| 邻居读 | pass2 经 GM 运行时行索引 | UB 内仿射滑窗 + 单 kernel | §0.6 R1（探针 A） |
| bf16 | 原生 | 无 v-op bf16 → fp32 中转 | §0.6 R2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/engram/engram_fwd.py` | **极高度**（同算子 NPU 版） | `T.Kernel(seq_len*M, is_npu=True)`、`T.alloc_shared`、`T.reduce_sum(dim=1)`、`T.vsigmoid`、conv tap + 行读 |
| `examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py` | **高度**（同族融合迁移） | Developer 纯 AIV 融合、`vsqrt`+除 的 rsqrt 绕法、buffer 顶层分配、GM 行 `T.copy` 运行时索引 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/` | 中（norm 族） | 行归约迁移基准形态、`alloc_shared` 向量规约 |
| `examples/norm/rms_norm.py` | 中 | RMSNorm 的 `reduce_sum` + v-prefix |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `H` | `(M, seq_len, d_padded)` | `float16 \| bfloat16` | 隐状态（host 侧 pad；`d_padded==d`） |
| `k` | `(M, seq_len, d_padded)` | same | key 投影（外部 GEMM） |
| `v` | `(M, seq_len, d_padded)` | same | value 投影（外部 GEMM） |
| `rms_w_h` / `rms_w_v` | `(d_padded,)` | same | RMSNorm 权重 |
| `conv_w` | `(4, d_padded)` | same | 深度卷积权重 |

### 4.2 输出张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `Y` | `(M, seq_len, d_padded)` | `same_as(H)` | 输出（host crop 到 d） |
| `vhat_buf` | `(M, seq_len, d_padded)` | `same_as(H)` | `alpha⊙v`（backward 策略 B） |
| `alpha_buf` | `(M, seq_len)` | `float32` | 门控标量 |
| `rrms_h_buf` / `rrms_k_buf` / `rrms_v_buf` | `(M, seq_len)` | `float32` | 3 个倒数 rms |

**对外契约 shape**：`Y/vhat:(M,seq_len,d)`、`alpha/rrms_h/k/v:(M,seq_len)`（与 `source_output_shape` 一致；host crop `d_padded→d`）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `ht` / `kt` | `(R, d_padded)` | f32 | UB | H / k 窗口 tile |
| `prod` | `(R, d_padded)` | f32 | UB | 平方 / 逐元素积 scratch |
| `vt` | `(R, d_padded)` | f32 | UB | `v_hat`（物化，CSE） |
| `vn` | `(R, d_padded)` | f32 | UB | `v_hat_norm` |
| `conv` | `(BLK_S, d_padded)` | f32 | UB | 卷积累加 |
| `sig` | `(BLK_S, d_padded)` | f32 | UB | `sigmoid(conv)` |
| `bro` | `(R, d_padded)` | f32 | UB | 行标量广播 scratch |
| `cb` | `(BLK_S, d_padded)` | f32 | UB | conv 权重广播 scratch |
| `wh2` / `wv2` / `cwp` | `(1, d_padded)` | f32 | UB | `w_h²` / `w_v` / `conv_w[p]` |
| `cwp_d` | `(1, d_padded)` | dtype | UB | `conv_w[p]` 原始 dtype（vcast 源） |
| `red` / `tmp1` / `rh` / `rk` / `rv` / `rmk` | `(R, 1)` | f32 | UB | 归约输出 / sqrt 临时 / rrms×3 / 行掩码 |
| `oy` / `ov` | `(BLK_S, d_padded)` | dtype | UB | `Y` / `vhat` 输出暂存（vcast 目标） |

### 4.4 内存搬运路径

```
纯 Vector 路径（GM → UB → GM）:
GM[H,k,v] --T.Parallel(clamped 运行时行索引)--> UB[ht/kt (R,d) f32]
   --[fp32 向量链: vmul/vsqrt/标量除/vsigmoid/vbrc/reduce_sum/vadd]--> UB[vt/vn (R,d) f32]
   --[因果卷积: vn[i+p] 仿射偏移 ⊙ vbrc(cw[p]) 累加]--> UB[conv (BLK_S,d) f32]
   --[vsigmoid + 残差 + vcast(rint)]--> UB[oy/ov dtype] --T.copy--> GM[Y, vhat]
   标量输出 alpha/rrms_* : UB[(R,1)] --(own 行)--> GM[(M,seq_len)] f32
```

### 4.5 UB 内存预算

> 口径：按主选候选 A、`BLK_S=8`（`R=11`）；两个代表 d 档。**未做激进 buffer 复用**，保守计。

| Buffer 组 | W4/W2 (d=256) | W3 (d=512) |
|-----------|---------------|-----------|
| `(R,d)` f32 × 8（`ht,kt,prod,vt,vn,bro` 6 + `conv,sig` 各 `(BLK_S,d)` 折算 2） | 8×11×256×4 ≈ 90 KB | 8×11×512×4 ≈ 180 KB |
| `cb` `(BLK_S,d)` f32 | 8 KB | 16 KB |
| 权重/掩码小 buffer（`wh2,wv2,cwp,cwp_d,red,tmp1,rh,rk,rv,rmk`） | ≈ 4 KB | ≈ 8 KB |
| 输出暂存 `oy,ov` `(BLK_S,d)` dtype | 8 KB | 16 KB |
| **UB 合计（raw）** | **≈ 110 KB** | **≈ 220 KB** |

> **风险与处置**：raw ≈ 220KB（W3）**已超 192KB**（未复用），但上述为**最坏并列**上界——实际生命周期存在明显重叠可复用（`prod`↔`bro`、`ht/kt` 在步骤 4 后释放给 `vt/vn`、`sig`↔`cb`、`oy`/`ov` 顺序复用；§3.3 伪代码即按此复用）。**复用后预估**：峰值 = `{ht,kt,prod}`(phase A) 与 `{vt,vn,bro}`+`conv`(phase B) 取大 → 约 **3×(R,d) + conv ≈ 76 KB**（W3）→ 按 `TRAP-UB-multibuffer-inflation` 最坏 ×1.7 ≈ **129 KB < 192 KB ✓**。**强制动作**：① Stage 3 上靶前 `tilelang.compile(main, target="npuir")` 读 `ub overflow, requires N bits` 取精确实测需求（`CG-2026-0008`：手工预算 ≠ 可编译证明）；② 溢出则 `BLK_S 8→4`（`R=7`，主 tile 线性缩小 → 复用后 ≈ 86KB raw）+ 合并 `ht/kt` 生命周期；③ `d=256` 档可放大 `BLK_S=16`。**注**：`CONST-capacity-910B2C`（UB 192KB）**绑 910B2C/9362、本设备 910B3 未重验**。

**L1 / L0C**：无 Cube 段，不涉及。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | 全参数编译期常量（`lru_cache` 按 `(M,seq_len,d,eps,dtype)` 特化） |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[6, 7, 8, 9, 10, 11],   # Y, vhat, alpha, rrms_h, rrms_k, rrms_v = prim_func 第 7..12 参数（idx 6..11）
    target="npuir",
)
def _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype):
    ...
```

> **wrapper 契约**：脚手架 wrapper `engram_gate_conv_fwd.py` 的 `EngramGateConvFwdKernel.forward` 调用 `_engram_gate_conv_fwd_kernel(M,seq_len,d,eps,dtype)(H,k,v,rms_w_h,rms_w_v,conv_w)`（无 `threads`，K9），并依赖 kernel 的 `out_idx=[6,7,8,9,10,11]` 自动分配 6 输出；`register_fake` 返回 `Y/vhat:(M,seq_len,d_padded)` 与 4 个 `(M,seq_len)` fp32。工厂签名保持 `(M,seq_len,d,eps,dtype)`（与 GPU 源一致）。**注**：`BLK_S` 若需作为可调参数，可作为工厂新增可选 kwarg（`BLK_S=8`），不影响 wrapper 调用。

---
## 5. Tiling 策略

### 5.1 计算类型

**类型**: **纯 Vector**（逐元素链 + 行内归约 + 窗口卷积；无矩阵乘）。

**判定依据**: 源算子无 `T.gemm`/matmul（投影在外部 GEMM 完成）；4 处 `T.reduce_sum` 为行内归约（H/k/(H⊙k)/vhat），非 MAC 密集段 → 落 Vector（UB）。**非 MixCV**（无 Cube 段）→ 无 CV 同步（§1.6.3 MIX 评估）。

### 5.2 Block 划分

```python
BLK_S   = 8                                    # 每 block 处理的输出 token 数（Stage-4 调优杠杆）
HALO    = 3                                    # 因果历史行 = 卷积核长 − 1
R       = BLK_S + HALO                         # 窗口行数 = 11
num_tiles = ceil(seq_len / BLK_S)              # M 轴=batch，seq_len 轴分块
grid    = M * num_tiles                        # 一维 T.Kernel
# d 不划分：(R, d_padded) 的向量链与行归约整块驻留 UB，无需 d 方向 tiling（避免跨核归约）
```

**block 选择理由**：
1. **seq_len 轴分块（非 d）**：卷积沿 `seq_len` 有因果依赖（窗口 `t-3..t`），故 `seq_len` 作分块轴（每 block 覆盖 `BLK_S` 连续 token + 3-halo）；`d` 是全算子唯一的连续/规约轴，保持整块（向量宽度整除、避免跨核 reduce）。
2. `BLK_S=8`：halo 冗余 `3/(BLK_S+3) = 27%`（流量），UB 峰值 ≈ 76KB（复用后，W3，§4.5）；`BLK_S` 为 Stage-4 调优杠杆（d=256 档可 16；UB 贴限可 4）。
3. `d` 不划分：`d_padded ≤ 512`，单 `(R,d_padded)` f32 ≤ 22.5KB，全链整块驻留 UB。
4. `R = BLK_S + HALO`：conv 输出第 `i` 行需 `vn[i..i+3]`（窗口行 `0..R-1`）。

### 5.3 约束分析

- **对齐约束**：`(R, d_padded)` f32 尾轴 `512×4=2048B`（32B 整数倍 ✓）；向量宽 fp32 ×4（512 整除）、fp16 ×8（256/512 整除）→ 尾 lane 浪费 0%。
- **UB 容量**：复用后峰值 ≈ 76 KB（W3），×1.7 ≈ 129 KB < 192 KB ✓（§4.5；含强制编译探针）。
- **L1 / L0C**：无 Cube，不涉及。
- **非整除**：manifest `d∈{256,512}` 均 256 对齐（`d_padded==d`）；`seq_len∈{16,32,64,128}` 均整除 `BLK_S=8` → **无尾块**。通用处理见 §5.4。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：
- `d`：manifest 各 workload `d∈{256,512}` 均 256 对齐 → `d_padded==d`，**无尾块**；保留 `d_padded=_align_up(d,256)` 以支持通用 d（host pad + 输出 crop，wrapper 已实现）。
- `seq_len`：manifest 各 workload 均整除 `BLK_S=8` → **无尾块**。为通用性，末 tile 若 `t0+BLK_S > seq_len`，取 `tail = T.min(BLK_S, seq_len - t0)`，输出 `T.copy(oy[0:tail,:], Y[bid, t0:t0+tail, :])` 双侧裁剪；输入载入已由索引 clamp（`gc ∈ [0,seq_len)`）保证不越界，无效 halo 行由 `rmk` 掩码清零。**推荐通用形态 = src/dst 双显式 slice**（`TRAP-T-copy-region-semantics` 规则②）。
- **不做 host padding 改内容**（`ascend-constraints.md` §4）；非整除一律 kernel 内 `tail` 裁剪 + host 元数据 pad。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子输出网格 = `M`（batch）× `num_tiles`（seq 分块），`d` 轴不划分：

```
num_logical_kernels = ceil(M/1) × ceil(seq_len / BLK_S) = M × ceil(seq_len / 8)
```

| workload | M | seq_len | d | BLK_S | num_tiles | 逻辑核数 |
|----------|---|---------|---|-------|-----------|----------|
| fwd-smoke-b1-s16-d256-fp16 | 1 | 16 | 256 | 8 | 2 | **2** |
| fwd-b1-s32-d256-fp16 | 1 | 32 | 256 | 8 | 4 | **4** |
| fwd-b2-s64-d512-fp16 | 2 | 64 | 512 | 8 | 8 | **16** |
| fwd-b1-s128-d256-bf16 | 1 | 128 | 256 | 8 | 16 | **16** |

**② 物理核数依据（实查）**：目标设备 `Ascend910B3`，实查代码与返回值（本会话 2026-10-15）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
NPUUtils.get().get_arch()              # 实际返回 "Ascend910B3"
```

→ **本算子为纯 Vector 算子，物理核数取 Vector Core = 40**（`core-split-strategy.md` §1 要素②：「纯 Vector 算子核数翻倍，即 `get_aicore_num() * 2`」）。**注**：`pattern-library/constants.md` 的 `CONST-aicore-910B2C`=24 / `CONST-aicore-910_9362`=20 均非本设备，**以实查为准**。

**③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数 → 无需适配**。

- 最大逻辑核数 = **16**（W3/W4）≤ 40；W1 smoke = 2、W2 = 4。
- **结论：无需适配**（标准 §1 要素③第一分支）。grid = `M·num_tiles` 直接下发，**无核内 `T.serial` 串行、无多波次串行调度、无残留负载不均**（核数 ≤ 物理核数，每核至多 1 个逻辑块）。
- **利用率说明（显式记录）**：16/40 = 40% 峰值占用（W1 仅 2/40）——这是本算子数据量小（`M·seq_len ≤ 128` token）的固有特征（低并行度、延迟导向）。**更高并行度候选**（token-per-block `BLK_S=1`，grid 最大 128）已在 §1.6.3 候选 F 评估：4× 冗余计算 + 需 GM/UB 移位重读，**列为 Stage 4 结构级备选**（不改变本设计分核方案）；`BLK_S` 可下调（8→4）提升并行度（W3→32）。
- **模式先例**：源算子 grid=(seq_len, M)（2D）；NPU 侧一维 `T.Kernel(seq_len*M, is_npu=True)`（`examples/engram/engram_fwd.py` L38）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级任务映射与 conv 的编译期 tap 展开；kernel 内**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务） | block 级并行 | `T.Kernel(M*num_tiles, is_npu=True)` | 每核处理一个 (batch, seq-tile) 的窗口全流程（§5.5：逻辑核 ≤ 40，无需适配） |
| 元素级（窗口行/通道） | 向量化 | `T.Parallel(R, d_padded)` / `T.Parallel(BLK_S, d_padded)` + v-prefix | 整 buffer 一次完成，无标量逐元素循环 |
| 行归约 | 向量化 | `T.reduce_sum(..., dim=1)` | `(R,d_padded)→(R,1)` 一次水平归约 |
| conv tap | **编译期展开**（factory 层 Python `for p in range(4)`） | host-level unroll | tap 偏移须为编译期常量（`TRAP-C13`）；4 条语句 |
| 行掩码/标量 | 向量化（`(R,1)` 上标量算术） | `T.Parallel(R)` | 每行 1 个标量，非逐元素热点 |

**T.Persistent 选择说明**：不使用 `T.Persistent`。grid = `M·num_tiles` ≤ 40 已 ≤ 物理核数，无多波次；且本算子无 Cube（`TRAP-DEVMODE-PERSIST-GEMM` 不适用，但保持结构简洁）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(M*num_tiles, is_npu=True)` + 线性 v-prefix 向量链 + factory 层 conv tap 展开；无 `T.Pipelined`、无核内逐元素串行循环）。

### 6.3 流水线优化

**当前设计**：不做显式流水线。原因：① 本算子纯向量链，片内重叠由 Developer 模式 **auto-multi-buffer** 处理（GM→UB 载入可与上一 tile 的向量计算重叠）；② block 间无数据依赖（halo 由 block 内重算覆盖），无需跨 tile 流水；③ 数据量小（≤0.8MB），发射项主导（§1.6.0 D-2），流水收益有限。**Stage 4 杠杆**：`BLK_S` 调整（流量/发射权衡）、载入合并（3 个 `(R,d)` 载入合并为一次 `T.copy` 大区域）、auto-multi-buffer 显式控制。

### 6.4 尾块处理

见 §5.4。manifest 全部 workload 无尾块（`d%256==0`、`seq_len%8==0`）；保留 `tail=T.min(BLK_S, seq_len-t0)` 双显式 slice + 索引 clamp + 行掩码以支持通用 shape。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）。

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链（全在 UB，无跨核/跨引擎）：

```
GM[H,k,v] --T.Parallel 载入--> UB[ht,kt]  (MTE2→V 自动等待)
   --[向量链 + reduce_sum: V→V barrier 自动]--> UB[vt,vn,conv]
   --[vcast + T.copy 落 GM]--> GM[Y,vhat,alpha,rrms_*]  (V→MTE3 自动等待)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| GM→UB 载入 → 向量链 | 自动 | 编译器插 MTE2→V 等待 |
| 向量链内同 buffer（in-place `vmul`）/ reduce | 自动 | V→V barrier |
| UB→GM 写回 | 自动 | 编译器插 V→MTE3 等待 |
| tile 间 auto-multi-buffer | 自动 | 跨 tile 载入重叠 |

无手动 `T.sync_block_set/wait`（无跨核 workspace、无 Cube 段）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs（auto-multi-buffer / 自动同步开启）
```

### 7.4 CV 融合设计

**本算子非 MixCV**（纯 Vector，无 `T.gemm`；AGENTS.md MixCV 触发规则不满足）。无 Cube/Vector 跨引擎同步、无 L0C→UB 直连、无跨核 workspace。**若未来融合外部投影 GEMM 则成为 MixCV，须先做 `CG-2026-0022` 最小 MIX 探针**（§1.6.3）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现——**直接移植源仓参考实现/测试基准** `examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py::engram_gate_conv_fwd_torch`（L35–L63），**不得复刻 §0.6/§1.6.3 的 NPU 结构**（保证验证独立性）。

```python
import torch
import torch.nn.functional as F

CONV_KERNEL_SIZE = 4


def _rmsnorm(x, w, eps=1e-6):
    """移植自 test_engram_gate_conv_fwd.py L27-L32（x:(M,T,d), w:(d,)）；返回 (normed, rrms)。"""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * w.float(), rrms.squeeze(-1)


def golden_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """Engram GateConv 前向参考实现（源语义；移植 test_engram_gate_conv_fwd.py::engram_gate_conv_fwd_torch）。

    契约：全程 fp32 计算（`.float()`），末端 `.to(H.dtype)` 单次舍回；
    与 §0.6/§1.6.3 的 NPU 结构（序列分块 / UB 滑窗 / vsqrt 绕法）无关——独立验证。
    """
    _, _, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)                      # 共用 rms_w_h

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)                    # (M, d, T)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))             # 左零填充（因果）
    conv_w_expanded = conv_w.float().T.unsqueeze(1)                 # (d, 1, 4)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (Y.to(H.dtype), v_hat.to(H.dtype), alpha.squeeze(-1).float(),
            rrms_h.float(), rrms_k.float(), rrms_v.float())
```

> 备注：golden **独立于** NPU 流水线（直接调用 PyTorch `@`/`mean`/`rsqrt`/`sigmoid`/`silu`/`conv1d`，不复用 §3.3 的 op 序列），满足验证独立性；输入生成参考 `examples/TileOPs/tileops/workloads/sequence_modeling.py`（`EngramGateConvFwdWorkload.gen_inputs`：`rms_w_h/rms_w_v` 为 ones）。
> **注（golden 与源内核的 ≤1ulp 差异，容差内）**：golden 的 `rrms_v`/`v_hat_norm` 由**已舍回 dtype** 的 `v_hat` 计算（`v_hat.to(H.dtype)`），而源内核 pass1 由 fp32 `vhat_val` 计算；golden 的残差用 **fp32 未舍回** `v_hat`（`+ v_hat.float()`），源内核 pass2 残差用**已舍回** `vhat_buf`。两处差异均 ≤1 ulp；`rtol=1e-1` 门禁下由 `verify_equiv` FULL 行（fp32 pipeline vs fp64 golden，max_abs 3.05e-5）与 §8.3 L0 覆盖。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float16 | 1e-1 | 1e-1 | 来源 `examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py` L87–L88 |
| bfloat16 | 2e-1 | 1e-1 | 来源 同上（bf16 精度低，容忍度放宽） |

**manifest roofline（供对账）**：`flops = M·seq_len·(24d) + 20·M·seq_len`；`bytes = (5·M·seq_len·d)·elem + 4·M·seq_len·4 + 6·d·elem`。

**误差预算核对**：`verify_equiv.py` 实测——O1（rsqrt 绕法）vs fp64 真值 max_abs 6.1e-5；O2（常数折叠）逐位一致（d 为 2 的幂）；O3（CSE）逐位一致；O4（门控因子化）vs fp32 原式 ≤1.6e-5；**全链路三 workload（M1S32 fp16 / M2S64 fp16 / M1S128 bf16）`Y` 与 fp64 golden 均 max_abs 3.05e-5、违反数 0**（22/22 `EQUIV_PASS`）。相对 `atol=1e-1`/`2e-1` 门禁余量充裕。Stage 3 门禁以 §8.1 golden 为准。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式覆盖：3 个 manifest workload + smoke、fp16/bf16、因果左边界（t=0/1/2）、非整除 seq_len、门控角点。**

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 smoke fp16 | `(M=1, seq_len=16, d=256)` fp16 | 与 §8.1 golden 比对 `Y/vhat/alpha/rrms_*` ≤ `atol=1e-1, rtol=1e-1` | 门槛通过 |
| L0-2 full W2 fp16 | `(1, 32, 256)` fp16 | 同上 | 门槛通过 |
| L0-3 full W3 fp16 | `(2, 64, 512)` fp16 | 同上 | 门槛通过 |
| L0-4 full W4 bf16 | `(1, 128, 256)` bf16 | 同上（`atol=2e-1`） | 门槛通过 |
| L0-5 因果左边界 | `seq_len=4`（< BLK_S，单 tile），已知 `vn` 序列 | `Y[:,0]/.1/.2]` 的卷积只含有效 tap（无效 tap 贡献 0）与 golden 一致 | 门槛通过 |
| L0-6 非整除 seq_len | `seq_len=13`（`13%8≠0`，末 tile 尾块裁剪） | `Y` 形状 `(M,13,d)` 正确、与 golden 一致（无越界写） | 门槛通过 |
| L0-7 门控角点 | 构造 `⟨h_n,k_n⟩` 大正/大负 → `alpha→1/0` | `v_hat≈v`/`≈0`、无 NaN/Inf；与 golden 一致 | 门槛通过 |
| L0-8 rrms 一致性 | 任意 workload | `rrms_h/k/v`（fp32）与 golden `≤1e-4`（标量输出更严） | 门槛通过 |
| L0-9 bf16 全链 | W4 bf16 | `vcast` 升/降无精度异常（对照 golden） | 门槛通过 |
| L0-10 d 对齐敏感性 | `d=256` 与 `d=512` 两档 | UB 预算 + 无尾块 | 门槛通过 |

- **执行方式**：`tilelang.compile(main, target="npuir")` 后经 `EngramGateConvFwdOp`/wrapper 单次调用，`torch.testing.assert_close(op_out.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，22/22 `EQUIV_PASS`，§1.6.1）；`history_version/design_probe_seq_tile_conv.py` + `probe_a_shift.py` + `probe_b_dynrow.py` + `probe_c_bisect_dynrow.py`（D-3 结构探针，§1.6.3）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 bf16 v-prefix 数学 API 缺失**（`vadd/vmul/vdiv/vsqrt/vsigmoid`、`reduce_sum` 各 op doc §2.2.1 bf16 ✗） | W4（bf16）直传编译失败 | 全链 fp32 中转（`vcast`，§0.6 R2） |
| **R-2 `T.vrsqrt` 近似指令**（`TRAP-vrsqrt-plain-precision`，2.9e-3） | 3 次 RMSNorm 误差累积 | `vsqrt`+标量除绕法（§1.6.1 O1，D-1 证 6.1e-5） |
| **R-3 `T.if_then_else` 带 buffer 操作数 → codegen 段错误**（本任务探针 V2，EXIT=139 无诊断） | 因果边界/无效行处理直接崩 | 索引 clamp（`T.max/T.min`）+ 算术行掩码（`T.vbrc`+`T.vmul`），§0.6 R4 |
| **R-4 store 处 direct-cast 被 MLIR verifier 拒**（探针 A 首版：`hivm.hir.vcast` result 须 ranked tensor） | 输出写回编译失败 | 先 `T.vcast` 到 dtype UB 再 `T.copy`（§3.2） |
| **R-5 裸标量广播段错误**（`TRAP-npuir-vec-scalar-broadcast`，1 向量操作数 `dst[j]=s*vec[j]`） | codegen 段错误 | 统一 `T.vbrc`+`T.vmul`（≥2 向量操作数）；行标量法 |
| **R-6 UB 容量**（`TRAP-UB-multibuffer-inflation` status=verified；`CG-2026-0008` 基础分配差） | 大 `BLK_S`/多 buffer 可能超 192KB | §4.5 复用后 ≈76KB（W3）；**Stage 3 强制编译探针**读精确需求；`BLK_S 8→4` 回退；重建 buffer 生命周期 |
| **R-7 动态行索引读写**（分块 tile 行界） | — | 探针 V1/V3 已实证合法（clamp 后读、slice 写） |
| **R-8 UB 仿射行偏移读**（conv 滑窗） | 若不可用则主选 conv 形态失效 | 探针 A 已实证合法；备选 = 候选 B（两 Pass，GM halo） |
| **R-9 `T.vbrc` `(R,1)→(R,1)` 被拒**（`TRAP-vbrc-empty-broadcast`） | 广播维为 1 时报错 | 本算子广播目标 N=d_padded>1（合法）；`rmk` 广播到 `(R,d_padded)` 目标轴 >1 ✓ |
| **R-10 常数表设备/版本**（`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-ascend910-9362-profile` 均绑 910B2C/9362，无 910B3 条目，`kb_stale_check` 标 stale） | D-2 估算不确定性 | 数字标注「未文档化假设 + 估算依据」；Stage 4 baseline 后对账回填 |
| **R-11 发射项主导（名义）** | §1.6.0 估算示发射项主导（23µs） | 标注 stale 常数；`BLK_S` 为杠杆；Stage 4 后对账 |
| **R-12 非整除 seq_len 尾块** | 末 tile 输出越界 | `tail=T.min(BLK_S,seq_len-t0)` 双侧裁剪 + 索引 clamp + 行掩码；manifest 不触发 |
| **R-13 fp32 GM 输入路径**（`TRAP-npuir-fp32-vector-reduce`） | fp32 输入编译失败 | **不触发**（manifest 输入 fp16/bf16，fp32 仅片上 UB 中转） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `T.if_then_else` 带 buffer 操作数 | 因果边界 select | codegen 段错误 EXIT=139 | 索引 clamp + 算术行掩码（R-3/§0.6 R4） |
| store 处 `T.cast(...)` direct | 输出写回 | MLIR verifier 拒 | `T.vcast`+`T.copy`（R-4） |
| 裸标量广播 `dst[j]=s*vec[j]` | `alpha⊙v` 写法 | codegen 段错误 | `T.vbrc`+`T.vmul`（R-5） |
| bf16 直接传入 v-op | 未 `vcast` 上转 | 编译失败 | bf16 先 `vcast(rint)` 到 fp32（R-1） |
| `vcast` 非 `rint` 升级 | f16→f32/bf16→f32 仅 `rint` | 编译失败 | 升降统一 `round_mode="rint"` |
| `T.copy` 跨 dtype | 用 copy 兼做 cast | 隐式 `VCastOp`、舍入不可控（`TRAP-C12`） | 显式 `T.vcast` |
| `for p in range()` 写在 prim_func 内 | conv tap 展开 | 运行时 Var → 动态 slice 非法（`TRAP-C13`） | factory 层展开 |
| buffer 在 `if` 内分配 | 顶层分配纪律 | `Undefined variable`（`TRAP-conditional-alloc`） | 顶层无条件分配 |
| UB 溢出 | `BLK_S`/buffer 过多 | 编译失败（`ub overflow`） | §4.5 预算 + 编译探针（R-6） |
| 行掩码方向错 | `rmk` 公式 | 因果边界/尾块错误 | L0-5/L0-6 逐位核对 |
| conv 输出行映射错 | `conv[i]=Σvn[i+p]` | 语义错 | 对照 golden（无膨胀，`src_t=t-3+p`） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **因果左边界（t<3）** | tile 0 的 halo 行 `gt<0` 由索引 clamp（读 token 0）+ 行掩码清零；与 golden 左零填充等价（L0-5） |
| **非整除 seq_len** | 末 tile `tail` 裁剪 + 索引 clamp（R-12；manifest 不触发） |
| **d 非 256 倍数（通用）** | host 尾维 pad 到 `d_padded` + 输出 crop（wrapper 既有；manifest 不触发） |
| **bf16 workload** | 全链 `vcast(rint)` 中转（R-1） |
| **门控饱和（alpha→0/1）** | `sigmoid` 全域稳定；`v_hat→0` 时 RMSNorm 的 `s=eps` 非零，无除零 |
| **`rms_w_h/rms_w_v = ones`（manifest）** | 权重广播仍走通用路径（`wh2=1`），无特化 |
| **NaN/Inf 输入/中间值** | 契约 workload 不含；`sqrt(s)`/`s` 定义域 `s≥eps>0`；Stage 3 `equal_nan=True` 口径核对 |
| **极大 seq_len/ M** | 逻辑核数 `M·ceil(seq_len/BLK_S)` 随规模增长；超 40 时触发运行时串行调度（当前 manifest 不涉及，若扩展按 §5.5 第三分支核内 `T.serial` persistent 化或 `BLK_S` 下调） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # D-1 等价性机器验证（22/22 EQUIV_PASS）
└── history_version/
    ├── design_probe_seq_tile_conv.py     # D-3 探针（原始合并版）
    ├── probe_a_shift.py                  # D-3 探针 A：UB 仿射行偏移读
    ├── probe_b_dynrow.py                 # D-3 探针 B：动态行索引 + if_then_else（失败）
    └── probe_c_bisect_dynrow.py          # D-3 探针：V1..V4 二分
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | **已完成** | 本设计文档（含 §0 迁移分析、§1.6 算法调研与优化、§5.5 分核三要素） |
| `verify_equiv.py` | **已完成** | D-1：22/22 `EQUIV_PASS`（§1.6.1） |
| `history_version/probe_*.py` | **已完成** | D-3 结构探针（§1.6.3） |
| `_engram_gate_conv_fwd_kernel.py` | 待实现（Stage 3） | 按 §3.3 伪代码 + §5 分核实现（含 §8.1 golden） |
| `RETROSPECTIVE.md` | Stage 1 复盘（追加） | 见同目录 |

> 项目名 `engram_gate_conv_fwd` 由 conductor 传入；下游 Stage 3 产出在脚手架 wrapper 目录 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_fwd/`。

### 10.3 命名规范

- 算子目录名: `_engram_gate_conv_fwd_kernel`（snake_case，与迁移函数名一致）
- 实现文件: `_engram_gate_conv_fwd_kernel.py`
- 测试/基准: 复用 `examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py` / `benchmarks/ops/bench_engram_gate_conv_fwd.py`

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）
2. ✅ 等价性机器验证（`verify_equiv.py`）+ 结构探针（`history_version/probe_*.py`）
3. ⬜ Golden 函数（§8.1，移植源仓参考实现）
4. ⬜ 算子实现（`_engram_gate_conv_fwd_kernel.py`）+ 与 Golden 精度比对 + L0 门槛

**实现要点速览（Stage 3 输入）**：
1. **结构**：一维 `T.Kernel(M*num_tiles, is_npu=True)`，每 block 处理一个 (batch, seq-tile) 的 `BLK_S=8` 输出 token + 3-halo；一维 grid，`bid=cid%M`、`tile=cid//M`。
2. **计算域**：fp16/bf16 经 `T.cast`/`T.vcast(rint)` 升 fp32，全链 fp32，末端 `vcast(rint)` 单次舍回（bf16 API 强制）。
3. **RMSNorm**：`s=Σx²·fl(1/d)+eps` → `t=vsqrt(s)` → `rrms=t/s`（绕 `vrsqrt`）→ `x·rrms_brc⊙w_brc`。
4. **门控**：`dot=(rrms_h·rrms_k)·Σ(H⊙k⊙w_h²)`（因子化）；`alpha=vsigmoid(dot·fl(1/√d))`；`vhat=alpha_brc⊙v`（物化）。
5. **卷积**：4 tap factory 层展开；`conv[i,j]+=cb[i,j]*vn[i+p,j]`（UB 仿射偏移，探针 A）。
6. **边界**：索引 clamp（`T.max/T.min`）+ 算术行掩码（禁 `T.if_then_else` 带 buffer，探针 V2）。
7. **纪律**：buffer 顶层无条件分配；`T.vbrc`+`T.vmul`（禁裸标量广播）；store 处禁 direct-cast（`vcast`+`copy`）。
8. **分核**：grid `M·num_tiles` ≤ 40（纯 Vector）→ 无需适配；`BLK_S` 为 Stage-4 杠杆。

---
