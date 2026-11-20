# `_engram_gate_conv_bwd_kernel` 算子设计文档（迁移）

> 项目：`engram_gate_conv_bwd`；算子目录：`examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/`。
> 迁移场景：`scenario=migration`（harness 子模式，GPU TileOPs → Ascend NPU npuir）。
> 编程模式：Developer。目标设备：Ascend910B3（实查 20 AI Core / 40 Vector Core）。
> 工具链戳：tilelang `0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2`（`/mnt/workspace/Agentkj/build`，USE_NPUIR=ON）/ CANN 9.0.0 / 2026-10-15。
> **修订版 v1（`revision_index=1`，2026-10-15）**：本次修订针对 Stage 2 检视（1 阻塞 + 3 建议）——**阻塞项**修正 §3.3 K2 伪代码 `drms_w_h` 别名错误（`dhn`/`dkn` 被就地覆写为 `dH`/`dk` 后仍被读取）；**建议项** ① `d=512` 默认 `BLK_S=4`（shape 分派）并补全 `bro_s` 预算、② `verify_equiv.py` 优化项编号与本文档统一（O1–O4）并增补 `drms_w_h/drms_w_v/dconv_w` 的 FULL 对照行（别名回归护栏）、③ §0.1 行号引用订正。**逐项处置与「为何不会再犯同一错误」见 §0.8**。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 算法调研 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_bwd.py`

- 本迁移对象 = 工厂 `_engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype)`（`@functools.lru_cache` L37，工厂体 L38–L347）内的 `@tilelang.jit(out_idx=[12,13,14,15,16,17,18,19], compile_flags=["-O3","-DENABLE_BF16"])` 函数 `_func(threads)`（L43–L345）→ `@T.macro _gate_conv_bwd`（L50–L314，全部计算）→ `@T.prim_func main`（L316–L345）；**单 prim_func 内含 5 个 `with T.Kernel(...)` 块**：zero-init（L73）、Pass 1（L79）、dconv_w 归约（L123）、Pass 1b（L150）、Pass 2（L215）。
- 辅助常量：`ALIGNMENT=256`（L29）、`CONV_KERNEL_SIZE=4`（L30）、`_align_up`（L33–L34）。
- `@torch.library.custom_op("top::engram_gate_conv_bwd", mutates_args=())` 包装（L350–L375）+ `register_fake`（L378–L394）；`class EngramGateConvBwdKernel`（L397–L499，`supported_archs=[80,86,89,90]`、`default_config={"threads":128}`、`autotune_configs=[128,256,512]`）。
- host 侧 `forward`（L439–L499）：对 `dY/H/k/v/vhat/rms_w_h/rms_w_v/conv_w` 尾维 `F.pad` 到 `d_padded=_align_up(d,256)`；调用后把 `results[:6]`（dH/dk/dv/drms_w_h/drms_w_v/dconv_w）crop 回 `d` 返回（`dvhat_buf`/`dvhat_out` 两个形参是 kernel 内部 workspace，不返回）。
- NPU 脚手架（Stage 0 已生成，只读参照）：提取内核 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernels.py`（Pattern B，`_align_up` 已补全）、wrapper 同目录 `engram_gate_conv_bwd.py`（K5/K7/K8/K9 已改：`supported_archs=None`、`npub::engram_gate_conv_bwd`、去 autotune/tune、去 `threads`、`ascend_mode="Developer"`）、op `examples/TileOPs/tileops/ops/sequence_modeling/engram_gate_conv_bwd.py`（`EngramGateConvBwdOp` + `eval_roofline`）。
- **本仓既有 NPU 参照（同算子，只读）**：`examples/engram/engram_bwd.py`（`target="npuir"`、三个独立 `@tl.jit` kernel pass1/pass2/pass3、每 token 一 block、`T.alloc_shared`、`T.atomic_add`、`T.copy` 运行时行索引、语句级 `if src_t >= 0:`）。该参照证实**纯 Vector 多 kernel 路线可在 npuir 编译**，但其 `T.atomic_add`（drms_w_h/drms_w_v/dconv_w 跨 token 累加）在本工具链实测**崩溃**（§1.6.3 探针 P4）、语句级 `if buffer 条件` + 每 token 一 block 结构为本设计的显式改进点（§0.6）。
- **同族迁移案例（只读，预注入）**：`CASE-engram-gate-conv-fwd-migration`（`pattern-library/cases.md` L614，同族**正向**迁移）与 `CASE-engram-decode-migration`（同 L590，同族 decode）；能力缺口 `CG-2026-0022`（`.agents/evolution/capability-gaps.md` L961，Cube+Vector MIX 缺口）。本算子逐条对照见 §0.5/§1.6.3/§9。

### 0.1 源算子语义（做什么）

**数学语义**（源码 docstring L1–L15 + `_gate_conv_bwd` L50–L314 逐语句核对；与 `op_requirements.math_formula` 一致，**以源码为准**）：

记前向（`_engram_gate_conv_bwd` 的输入即前向保存量/输入）：

```
h_norm  = RMSNorm(H, rms_w_h)                       # (M, T, d)   仅用于反向
k_norm  = RMSNorm(k, rms_w_h)                       # (M, T, d)   共用 rms_w_h
alpha   = sigmoid( Σ_d (h_norm ⊙ k_norm) / sqrt(d) ) # (M, T)      前向已存
v_hat   = alpha ⊙ v                                  # (M, T, d)   前向已存（vhat）
vhat_n  = RMSNorm(v_hat, rms_w_v)                    # (M, T, d)
conv_o[t] = Σ_{p=0}^{3} conv_w[p] ⊙ vhat_n[t-(3-p)]  # 因果深度卷积 k=4（无膨胀）
Y       = SiLU(conv_o) + v_hat
```

反向输入：`dY`（Y 的梯度）+ 前向保存量 `vhat`、`alpha`、`rrms_h/k/v` + 前向输入 `H,k,v,rms_w_h,rms_w_v,conv_w`。**反向计算**（与源码 L72–L314 逐语句一致）：

```
# Pass 1a：dconv_w（权重梯度，跨 token 归约）
vn[s]        = vhat[s] ⊙ rrms_v[s] ⊙ rms_w_v            # (M,T,d)；s<0 时为 0
conv_o[t]    = Σ_p conv_w[p] ⊙ vn[t-(3-p)]              # 重算（未保存）
d_silu[t]    = dY[t] ⊙ ( σ(conv_o[t]) + conv_o[t]·σ(conv_o[t])·(1-σ(conv_o[t])) )
dconv_w[p]   = Σ_{t: t-(3-p)≥0} d_silu[t] ⊙ vn[t-(3-p)] # (4,d)
# Pass 1b：conv 转置 + RMSNorm(vhat) 反向
d_vnorm[t]   = Σ_p conv_w[p] ⊙ d_silu[t+(3-p)]          # (M,T,d)；t+(3-p)∉[0,T) 时 0
drms_w_v     = Σ_t d_vnorm[t] ⊙ vhat[t] ⊙ rrms_v[t]      # (d,)
dot_v[t]     = Σ_c vhat[t,c]·rms_w_v[c]·d_vnorm[t,c]     # (M,T)
d(vhat)[t]   = rrms_v·rms_w_v·d_vnorm[t] − rrms_v³·vhat[t]·dot_v[t]/d + dY[t]
# Pass 2：门控反向 + RMSNorm(H)/RMSNorm(k) 反向
dv[t]        = alpha[t] ⊙ d(vhat)[t]
dalpha[t]    = Σ_c d(vhat)[t,c]·v[t,c]
ddot[t]      = dalpha[t]·alpha[t]·(1−alpha[t])/sqrt(d)
dh_norm[t]   = ddot[t] ⊙ k_norm[t] ;  dk_norm[t] = ddot[t] ⊙ h_norm[t]
dot_h[t]     = Σ_c H[t,c]·rms_w_h[c]·dh_norm[t,c]
dH[t]        = rrms_h·rms_w_h·dh_norm[t] − rrms_h³·H[t]·dot_h[t]/d
dot_k[t]     = Σ_c k[t,c]·rms_w_h[c]·dk_norm[t,c]
dk[t]        = rrms_k·rms_w_h·dk_norm[t] − rrms_k³·k[t]·dot_k[t]/d
drms_w_h     = Σ_t ( dh_norm[t]⊙H[t]⊙rrms_h + dk_norm[t]⊙k[t]⊙rrms_k )   # (d,)
```

其中 `RMSNorm(x,w) = x · rrms · w`，`rrms = (Σ_c x²/d + ε)^(-1/2)`（**除以真 d，非 d_padded**；源码 **L203 / L277 / L300** 用 `/float(d)`，L256 用 `/T.sqrt(float(d))`）。

**关键语义要点（迁移必须保持）**：
- **`vhat`/`alpha`/`rrms_*` 是前向保存量（输入）**，反向不重算它们；反向只重算 `conv_o`（用于 SiLU 导数）。
- **卷积无膨胀**（dilation=1，kernel=4）：`vn` 索引 `t-(3-p)` / `d_silu` 索引 `t+(3-p)`，无 dilation 因子（带膨胀的是同族 `engram_decode`，本算子不同）。
- **两处因果/边界零填充**：Pass1 左边界 `src_t = t-(3-p) < 0` → `vn=0`（源码 L96-L105）；Pass1b 右/左边界 `dst_t = t+(3-p) ∉ [0,T)` → `d_silu=0`（源码 L169-L173）。等价于前向卷积的左侧零填充 + 转置的右侧零填充。
- **dtype 语义**：源全程 `accum_dtype="float"`（fp32）计算，仅在写回 `dH/dk/dv`（L312-L314）时 `T.cast(..., dtype)`；`drms_w_h/drms_w_v/dconv_w` 直接 fp32。即 fp16/bf16 输入 → fp32 计算域 → 单次舍回。
- **输出 shape（迁移必需）**：`dH/dk/dv:(M,seq_len,d)`（同 `dY` dtype）+ `drms_w_h/drms_w_v:(d,)`、`dconv_w:(4,d)`（fp32）。host 侧把 `d_padded` crop 回 `d`（wrapper L491–L498）。
- **host 侧语义**：`F.pad` 尾维 pad 到 `d_padded` + 输出 crop——元数据/视图级处理（不改张量真实内容），NPU 侧保留（wrapper 已实现）。
- **语义保持基线**：§8.1 golden 以本节语义为唯一依据（**优先移植源仓测试基准** `examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py::ref_engram_gate_conv_bwd`，L29–L74），**不得复刻 §0.6 / §1.6.3 的 NPU 结构**（保证验证独立性）。

### 0.2 源算子输入输出

**I/O 契约**（`_gate_conv_bwd` 签名 L51–L71 + `main` L318–L338 + manifest `EngramGateConvBwdOp`）：

| 参数 | 角色 | Shape（kernel 内） | dtype | 说明 |
|------|------|--------------------|-------|------|
| `dY` | 输入 | `(M, seq_len, d_padded)` | `float16 \| bfloat16` | 输出梯度 |
| `H` | 输入 | `(M, seq_len, d_padded)` | same | 隐状态（前向输入） |
| `k` | 输入 | `(M, seq_len, d_padded)` | same | key 投影（外部 GEMM） |
| `v` | 输入 | `(M, seq_len, d_padded)` | same | value 投影（外部 GEMM） |
| `rms_w_h` | 输入 | `(d_padded,)` | same | H/k 共用 RMSNorm 权重 |
| `rms_w_v` | 输入 | `(d_padded,)` | same | vhat 的 RMSNorm 权重 |
| `conv_w` | 输入 | `(4, d_padded)` | same | 深度卷积权重 |
| `vhat` | 输入 | `(M, seq_len, d_padded)` | same | 前向保存的 `alpha⊙v` |
| `alpha` | 输入 | `(M, seq_len)` | `float32` | 前向保存的门控标量 |
| `rrms_h` | 输入 | `(M, seq_len)` | `float32` | H 的倒数 rms |
| `rrms_k` | 输入 | `(M, seq_len)` | `float32` | k 的倒数 rms |
| `rrms_v` | 输入 | `(M, seq_len)` | `float32` | vhat 的倒数 rms |
| `dH` (out0) | 输出 | `(M, seq_len, d_padded)` | same_as(dY) | host crop 到 d |
| `dk` (out1) | 输出 | `(M, seq_len, d_padded)` | same_as(dY) | |
| `dv` (out2) | 输出 | `(M, seq_len, d_padded)` | same_as(dY) | |
| `drms_w_h` (out3) | 输出 | `(d_padded,)` | `float32` | 跨 token 归约 |
| `drms_w_v` (out4) | 输出 | `(d_padded,)` | `float32` | 跨 token 归约 |
| `dconv_w` (out5) | 输出 | `(4, d_padded)` | `float32` | 跨 token 归约 |
| `dvhat_buf` (out6) | **内部 workspace** | `(M, seq_len, d_padded)` | `float32` | 存 `d_silu`（Pass1→1a/1b 交接） |
| `dvhat_out` (out7) | **内部 workspace** | `(M, seq_len, d_padded)` | `float32` | 存 `d(vhat)`（Pass1b→Pass2 交接） |

- **对外契约 = 12 输入 + 6 输出**（`out_idx=[12..19]` 把 workspace 也声明为 kernel 输出、由 JIT 自动分配；wrapper `forward` 只返回 `results[:6]`）。manifest `signature` 列 12 in / 6 out；`shape_rules` 为形状一致约束（`H/k/v/vhat.shape==dY.shape` 等；**无 `d%N==0` 约束**，但 workloads 的 `d∈{256,512}` 均 256 对齐）。
- **layout**：行主序连续，尾轴 `d` 连续；无转置、无 in-place。
- **动态轴**：无。`M/seq_len/d/eps/dtype` 全为编译期常量（`lru_cache` 特化）。
- **规约语义（关键）**：源码含 4 类规约——① 行内 `T.reduce_sum(dim=1)`（`dot_v`/`dot_h`/`dot_k`，源码 L197/L271/L294，沿 d）；② 跨 token 归约 `dconv_w`（独立 kernel，`T.serial(M)` × `T.serial(seq_len)`，L123–L145）；③ 跨 token 归约 `drms_w_v`/`drms_w_h`（`T.atomic_add`，L184/L282/L305）。**累加顺序未指定**（② 沿 t 串行、③ 原子累加无序）；本设计改为**分块行归约 + 确定性 combine**（§0.6 R5），逐元素结果落在测试容差内（§1.6.1 FULL 行实测 max_abs ≤6.1e-5）。
- **边界语义**：`d_padded` 尾区由 host 零填充 → `Σ_{d_padded} x² = Σ_d x²`（源码除以 `float(d)` 正确）；`drms_w_*`/`dconv_w` 的 pad 区由 host crop 丢弃。卷积左/右边界零填充见 §0.1。
- **host 侧语义**：`F.pad` 尾维 pad 到 `d_padded` + 输出 crop（L475–L498）——元数据/视图级处理，NPU 侧保留。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`_gate_conv_bwd` L50–L314 逐语句；源码每个计算语句均归入某步骤）：

| # | Kernel 块 | 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|---|-----------|------|------|------|------|----------|
| 0 | zero-init（L73） | 清零权重梯度 | `T.Parallel(d_padded)`：`drms_w_h[j]=0`、`drms_w_v[j]=0` | — | GM `drms_w_h/drms_w_v` | — |
| 1 | Pass1（L79） | 载入 dY 行 | `dy_local[j]=cast(dY[bid,tid,j])` | GM `dY` | frag | — |
| 2 | Pass1 | 重算 conv_out | `T.serial(4)`：`src_t=tid-3+p`；`raw=if_then_else(src_t≥0, cast(vhat[bid,src_t,j]),0)`；`src_rrms=if_then_else(src_t≥0, rrms_v[bid,src_t],0)`；`normed=raw·src_rrms·cast(rms_w_v[j])`；`conv_out[j]+=cast(conv_w[p,j])·normed` | GM `vhat,rrms_v,rms_w_v,conv_w` | `conv_out` | `conv_o[t]` |
| 3 | Pass1 | SiLU 导数 | `sig=1/(1+exp(-conv_out[j]))`；`d_silu[j]=dy_local[j]·(sig+conv_out[j]·sig·(1-sig))` | `conv_out,dy_local` | `d_silu` | `d_silu[t]` |
| 4 | Pass1 | 缓冲 d_silu | `dvhat_buf[bid,tid,j]=d_silu[j]` | `d_silu` | GM `dvhat_buf` | — |
| 5 | dconv（L123） | 权重梯度 dconv_w | `T.Kernel(KS)`（`p=bx`）；`T.serial(M)`×`T.serial(seq_len)`：重算 `normed`（同步骤 2）；`dconv_local[j]+=dvhat_buf[bid,tid,j]·normed`；写 `dconv_w[p,j]` | GM `dvhat_buf,vhat,rrms_v,rms_w_v` | GM `dconv_w` | `dconv_w[p]` |
| 6 | Pass1b（L150） | conv 转置 | `T.serial(4)`：`dst_t=tid+3-p`；`d_si=if_then_else((dst_t≥0)·(dst_t<seq_len), dvhat_buf[bid,dst_t,j],0)`；`d_vnorm[j]+=cast(conv_w[p,j])·d_si` | GM `dvhat_buf,conv_w` | `d_vnorm` | `d_vnorm[t]` |
| 7 | Pass1b | 载入 vhat/rrms_v | `rrms_v_val=rrms_v[bid,tid]`；`vhat_local[j]=cast(vhat[bid,tid,j])` | GM `vhat,rrms_v` | frag | — |
| 8 | Pass1b | drms_w_v 累加 | `T.atomic_add(drms_w_v[j], d_vnorm[j]·vhat_local[j]·rrms_v_val)` | `d_vnorm,vhat` | GM `drms_w_v` | `drms_w_v` |
| 9 | Pass1b | dot_v + d(vhat) | `dot_2d[0,j]=vhat_local[j]·cast(rms_w_v[j])·d_vnorm[j]`→`reduce_sum(dim=1)`；`dvhat_local[j]=rrms_v·rms_w_v·d_vnorm − rrms_v³·vhat_local·dot_sum/d`；`dvhat_local[j]+=cast(dY[bid,tid,j])` | `vhat,d_vnorm,rms_w_v,dY` | `dvhat_local` | `dot_v[t]`,`d(vhat)[t]` |
| 10 | Pass1b | 存 d(vhat) | `dvhat_out[bid,tid,j]=dvhat_local[j]` | `dvhat_local` | GM `dvhat_out` | — |
| 11 | Pass2（L215） | 载入 | `dvhat_local[j]=dvhat_out[bid,tid,j]`；`v/h/k_local`；标量 `alpha/rrms_h/rrms_k` | GM `dvhat_out,v,H,k,alpha,rrms_h,rrms_k` | frag | — |
| 12 | Pass2 | dalpha + dv | `dalpha_2d[0,j]=dvhat_local[j]·v_local[j]`；`dv_local[j]=alpha_val·dvhat_local[j]`；`reduce_sum(dim=1)` | `dvhat,v` | `dalpha_sum,dv_local` | `dalpha[t]`,`dv[t]` |
| 13 | Pass2 | ddot | `ddot_val=dalpha_sum[0]·alpha_val·(1−alpha_val)/T.sqrt(float(d))` | `dalpha_sum,alpha` | `ddot_val` | `ddot[t]` |
| 14 | Pass2 | 重算 h_norm/k_norm | `h_norm_local[j]=h_local[j]·rrms_h_val·cast(rms_w_h[j])`；k 同（共用 `rms_w_h`） | GM `H,k,rms_w_h` | `h_norm,k_norm` | `h_norm,k_norm` |
| 15 | Pass2 | dot_h + dH | `dot_h_2d[0,j]=h_local[j]·cast(rms_w_h[j])·ddot_val·k_norm_local[j]`→`reduce_sum`；`dh_local[j]=rrms_h·rms_w_h·ddot_val·k_norm − rrms_h³·h_local·dot_h_sum/d` | `H,k_norm` | `dh_local` | `dot_h[t]`,`dH[t]` |
| 16 | Pass2 | drms_w_h（H 路） | `T.atomic_add(drms_w_h[j], ddot_val·k_norm_local[j]·h_local[j]·rrms_h_val)` | `H,k_norm` | GM `drms_w_h` | `drms_w_h` |
| 17 | Pass2 | dot_k + dk | `dot_k_2d[0,j]=k_local[j]·cast(rms_w_h[j])·ddot_val·h_norm_local[j]`→`reduce_sum`；`dk_local[j]=rrms_k·rms_w_h·ddot_val·h_norm − rrms_k³·k_local·dot_k_sum/d` | `k,h_norm` | `dk_local` | `dot_k[t]`,`dk[t]` |
| 18 | Pass2 | drms_w_h（k 路） | `T.atomic_add(drms_w_h[j], ddot_val·h_norm_local[j]·k_local[j]·rrms_k_val)` | `k,h_norm` | GM `drms_w_h` | `drms_w_h` |
| 19 | Pass2 | 写输出 | `dH[bid,tid,j]=cast(dh_local[j],dtype)`；同 dk/dv | `dh/dk/dv_local` | GM `dH,dk,dv` | — |

**数据流图（源硬件 = GPU 视角）**：

```
GM[dY] --T.Parallel--> REG[dy_local (d)]
GM[vhat,rrms_v] --(serial p=0..3, 行 tid-3+p, 运行时行索引)--> REG --FMA(conv_w)--> REG[conv_out]
  REG[conv_out] --(silu')--> REG[d_silu] --cast--> GM[dvhat_buf(M,T,d) f32]
---- 第二个 grid 启动（dconv_w，grid=KS）----
GM[dvhat_buf,vhat,rrms_v,rms_w_v] --(serial M × serial T)--> REG[dconv_local (d)] --写--> GM[dconv_w(4,d)]
---- 第三个 grid 启动（Pass1b，grid=(T,M)）----
GM[dvhat_buf] --(serial p=0..3, 行 tid+3-p)--> REG[d_vnorm] ;  GM[vhat,rrms_v,rms_w_v] --> REG
   REG[d_vnorm,vhat] --atomic_add--> GM[drms_w_v(d)]
   REG-->dot_v --reduce_sum--> d(vhat) --(+dY)--> GM[dvhat_out(M,T,d) f32]
---- 第四个 grid 启动（Pass2，grid=(T,M)）----
GM[dvhat_out,v,H,k,rms_w_h,alpha,rrms_h/k] --T.Parallel--> REG
   REG --dalpha/ddot--> dH/dk/dv --cast--> GM[dH,dk,dv]
   REG --atomic_add--> GM[drms_w_h(d)]
```

- **无 SMEM**：全部中间量驻留 register/fragment（`T.alloc_fragment`，Pass1/1b/2 共约 20 个）；`block_size = threads`。
- **5 个 grid 启动在同一个 prim_func 内**；**跨块交接经 GM**：`dvhat_buf`（d_silu，fp32）在 Pass1 写出、dconv 与 Pass1b 读回；`dvhat_out`（d(vhat)，fp32）在 Pass1b 写出、Pass2 读回。**邻居行由运行时行索引 `src_t`/`dst_t` 从 GM 读取**（L98/L155/L171）——`d_silu` 窗口与 `vhat` 窗口在源实现里的落地方式。
- **跨 token 权重梯度归约**：`dconv_w` 用独立 kernel（grid=KS，串行 M×T，避开源注释所述 `atomic_add` lowering 缺陷 L120-L122）；`drms_w_h/drms_w_v` 用 `T.atomic_add` 跨 block 累加（无零初始化保证 → 由 zero-init kernel 保证）。

**循环与并行结构**：`T.Kernel(seq_len, M, threads=threads)`（2D grid `(seq_len, M)`：`bx=tid=token`、`by=bid=batch`）×3（Pass1/1b/2）+ `T.Kernel(1)`（zero-init）+ `T.Kernel(KS)`（dconv）；`T.Parallel(d_padded)` 把 d 分给 threads；conv/转置 tap 维走 `T.serial(4)`；token 归约走 `T.serial(M)/T.serial(seq_len)`。

**host 侧逻辑**：`F.pad` 尾维 pad 到 `d_padded`、输出 crop（L475–L498）；`lru_cache` 按 `(M,seq_len,d,eps,dtype)` 编译期特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **5 个阶段融合进同一 prim_func**（zero-init / Pass1 / dconv / Pass1b / Pass2，共享 `out_idx`） | 减 Python 分派、共享中间 workspace 声明 | 五个 `T.Kernel` 块、中间量经 GM 交接 | 无（结构层） | 结构可移植性由 npuir 多 `T.Kernel` 支持决定 → **重新设计（§0.6 R1）** |
| 2 | **2D fragment `(1,d_padded)` 供 `T.reduce_sum(dim=1)`** | 行内归约（dot_v/dot_h/dot_k） | 沿 dim=1 水平 reduce | GPU 线程级 reduce | 优化层 → **等价替换（UB reduce）** |
| 3 | **全程 fp32 累加**（`accum_dtype="float"` + 每处 `T.cast`） | 数值精度（对齐 golden fp32 opmath） | fp16/bf16 → fp32 计算、末端单次舍回 | 半精度 ALU 下需中转 | **语义层 → 保留并强化（NPU bf16 v-op 缺失）** |
| 4 | **fragment/register 驻留**（约 20 个 `T.alloc_fragment`） | 中间量不出寄存器 | 大寄存器堆 | GPU 寄存器堆 | **硬件强相关 → 等价替换（UB）** |
| 5 | **2D grid `(seq_len,M)` + `threads` 线程级 d 并行** | token/batch 并行 + 元素并行 | `T.Kernel(seq_len,M,threads)` | GPU block/thread 调度 | **硬件强相关 → 重新设计（一维 Kernel + 向量化）** |
| 6 | **256 对齐 pad（`d_padded`）** | GPU 向量对齐、免尾界守卫 | host `F.pad` + 编译期 `d_padded` | memory coalescing | 实现层 → **保留逻辑**（manifest `d∈{256,512}` 已对齐） |
| 7 | **`dconv_w` 独立归约 kernel 规避 `atomic_add` lowering 缺陷**（L120-L122 注释：TileLang 0.1.11 把 Pass1 内 `atomic_add(dconv_w…)` lower 成 launch-failing kernel） | 正确性规避 | 独立 `T.Kernel(KS)` + `T.serial(M)/T.serial(T)` 串行归约 | TileLang 0.1.11 lowering 缺陷 | **重新设计（§0.6 R5：确定性分块归约 + combine）** |
| 8 | **`drms_w_h/drms_w_v` 用 `T.atomic_add` 跨 block 累加** | 免跨块同步 | 原子加 | GPU atomic | **重新设计（§0.6 R5：npnir `atomic_add` 实测崩溃——探针 P4）** |
| 9 | **`out_idx=[12..19]` 自动分配 8 输出（含 2 内部 workspace）** | 免 host 手动分配 | JIT 输出约定 | 无 | 结构层 → **重新设计（多 kernel 闭包，§0.6 R1）** |
| 10 | **`T.if_then_else(src_t≥0,…)` 处理卷积左/右边界**（L96/L101/L169） | 免显式分支 / 零填充等价 | select 语义 | — | **重新设计（§0.6 R4：算术行掩码——带 buffer 操作数的 `T.if_then_else` 在 npuir codegen 段错误）** |
| 11 | **权重（`rms_w_h/rms_w_v/conv_w`）逐元素直接读 GM** | 免权重预载 | 片上直读 | — | 保留（权重极小，可整块载入 UB） |
| 12 | **无膨胀因果深度卷积（kernel 4）** | 结构简单、窗口静态 | 编译期常量 tap 偏移 | — | 保留 |
| 13 | **`threads` autotune（128/256/512）+ `lru_cache` 特化** | 按 shape 选线程数 / 消运行时分派 | 编译期常量固化 | `threads` 无 NPU 对应 | `threads` → **舍弃**；`lru_cache` → **保留** |
| 14 | **`d_silu` 落 GM 缓冲（`dvhat_buf`）供 dconv 与 Pass1b 复用** | 免重算 SiLU 导数 | GM 落盘 + 运行时行索引读邻居 | 无（算法/结构层） | **重新设计（§0.6 R1：分块 halo 重算，降 GM 往返）** |
| 15 | **`conv_o` 激活重算**（前向未保存，Pass1 重算以取 silu'） | 省前向保存量 | 重算 vs 保存权衡 | 无（算法层） | **保留（语义必需：conv_o 未保存）** |
| 16 | **`d(vhat)` 落 GM 缓冲（`dvhat_out`）供 Pass2 复用** | 跨阶段交接 | GM 落盘 | 无 | **重新设计（§0.6 R1：Pass1b+Pass2 融合，消 GM 往返）** |

> 识别不出来 ≠ 不存在。上述 16 项为源码可识别的全部优化/规避手段；逐项处置见 §0.5。未识别的优化会被静默丢弃，导致迁移后性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0：源算法与调研候选同台比较）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义（GateConv 反向；I/O、两处边界零填充、host pad 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| #1 五阶段融合进一 prim_func（含各步骤数学顺序与跨阶段 GM 交接） | 算法/结构 | 依赖「一个 prim_func 含多 `T.Kernel`」 | **无**（npuir JIT 不支持单 prim_func 多 `T.Kernel`——探针 P1 实测 grid 求值失败） | **重新设计** | → §0.6 R1：Python 工厂闭包按序调度 **3 个独立 `@tilelang.jit` kernel**（与 NPU 参照 `examples/engram/engram_bwd.py` 同族） |
| #2 2D fragment `reduce_sum(dim=1)` | 优化 | GPU 线程级 reduce | **有**（`T.reduce_sum` on UB；`docs/Tilelang.language/规约操作/T.reduce_sum.md`） | **等价替换** | UB `(R,d) f32 → (R,1)` 的 `T.reduce_sum(..., dim=1)`（§1.6.2 #1）；token 向 `dim=0` 归约实测可用（探针 P2） |
| #3 全程 fp32 累加 | 语义/优化 | 无 | **有**（fp32 计算 + `T.vcast`）；且 **NPU v-op 无 bf16 使其成为 API 强制** | **保留（并强化）** | → §0.6 R2：fp16/bf16 经 `vcast(rint)` 升 fp32 → fp32 向量链 → `vcast(rint)` 舍回 |
| #4 fragment/register 驻留 | 优化 | GPU 寄存器堆 | 无直接等价（NPU 无 `threads`，v-op 作用于 UB buffer） | **等价替换** | → §0.6 R3：`T.alloc_shared`（UB）整块驻留 + v-prefix 整 buffer 向量链 |
| #5 2D grid `(seq_len,M)` + threads | 优化 | GPU block/thread 调度 | **无**（npuir 仅一维 `T.Kernel`；`threads=` 无效，`TRAP-threads-kwarg-noop`） | **重新设计** | → §0.6 R3：一维 `T.Kernel(M*num_tiles, is_npu=True)`，元素并行交给向量单元 |
| #6 256 对齐 pad | 优化 | memory coalescing | 有（32B 尾轴对齐） | **保留（逻辑）** | manifest `d∈{256,512}` → `d_padded==d`，不触发；保留变量以支持通用 d（§5.4） |
| #7 dconv_w 独立归约 kernel | 优化/规避 | TileLang 0.1.11 lowering 缺陷 | 有（确定性分块归约 + combine；`atomic_add` 崩溃见 #8） | **重新设计** | → §0.6 R5：K1 每 block 内 token 向 `reduce_sum(dim=0)` 得 `(1,d)` 部分和 → K3 combine（探针 P2/P5 实测正确） |
| #8 `T.atomic_add` 跨 block 累加 | 优化 | GPU atomic RMW | **无**（npnir Developer `T.atomic_add` 实测 SIGABRT，探针 P4） | **重新设计** | → §0.6 R5：确定性「分块部分和 + combine」替代原子累加（**注意：测试要求多次运行确定性**，原子累加不满足） |
| #9 `out_idx` 自动分配 8 输出 | 优化 | 无 | 部分（多 kernel 闭包需手工分配） | **重新设计** | → §0.6 R1/R6：工厂闭包内 `torch.empty` 分配 6 个契约输出 + 内部 workspace；wrapper `register_fake` 契约数由 8 对齐到 6（§3.4） |
| #10 `T.if_then_else(src_t≥0,…)` 边界 | 优化 | select 语义 | 有（但**带 buffer 操作数的 `T.if_then_else` 在 npuir codegen 段错误**——`TRAP-npuir-ifthenelse-parallel-segv`） | **重新设计** | → §0.6 R4：索引 clamp（`T.max/T.min`）+ 算术行掩码（`T.vbrc`+`T.vmul`），掩码标量用**编译期 unroll**（探针 P6a vs P6b） |
| #11 权重 GM 直读 | 优化 | 无 | 有 | **保留** | 权重极小（`6·d·elem`），整块载入 UB 复用 |
| #12 无膨胀深度卷积（kernel 4） | 算法 | 无 | 有 | **保留** | tap 编译期展开（4 个常量偏移），UB 内 shifted 读（§0.6 R1；探针 P6a） |
| #13 `threads` autotune | 优化 | GPU 线程模型 | 无 | **舍弃** | K3/K9：npuir 无 `threads`；`lru_cache` 保留 |
| #14 `d_silu` GM 缓冲 | 优化 | 无（算法结构） | 有（分块 halo 重算） | **重整（部分保留）** | → §0.6 R1：`d_silu`（`dvhat_buf`）仍需跨 K1→K2 交接（K2 conv 转置需邻居），保留为 workspace 但**从 2 次 GM 往返降为 1 写 1 读** |
| #15 `conv_o` 激活重算 | 算法 | 无 | 有 | **保留** | 语义必需（前向未保存 `conv_o`；silu' 需之）；在 K1 内重算 |
| #16 `d(vhat)` GM 缓冲（`dvhat_out`） | 优化 | 无（算法结构） | 有（Pass1b+Pass2 同 token 融合） | **重新设计** | → §0.6 R1：K2 内 Pass1b 与 Pass2 融合——`d(vhat)` 在同一 block 内一次物化即用，**消除 `dvhat_out` GM 往返** |
| fp8 / 其他非 manifest dtype | — | — | 无（manifest 仅 fp16/bf16） | **舍弃** | 不在契约内 |

> **依据**：#3 的 bf16 缺失由 `docs/Tilelang.language/数学操作/T.vadd.md`·`T.vmul.md`·`T.vdiv.md`·`T.vsqrt.md`、`数据类型转换操作/T.vcast.md` 的 dtype 矩阵佐证（数学 v-op 均 **bf16 ✗**，fp16 ✓ / fp32 ✓；`vcast` 支持 `f16→f32`/`bf16→f32` 与反向的 `rint`）；#5 由 `pattern-library/traps-compiler.md` `TRAP-threads-kwarg-noop`（status=verified）+ 脚手架 K3/K9（wrapper 已去 `threads`）+ 本仓 `examples/engram/engram_bwd.py`（一维 `T.Kernel(seq_len*M, is_npu=True)`）佐证；#1/#8/#10 由本任务设计期探针（`history_version/design_probe_bwd_structure.py`，2026-10-15，工具链戳见页首）实测佐证（P1 多块 grid 求值失败 / P4 `atomic_add` SIGABRT / P6b 并行运行时 minmax 掩码错误）；#2/#7 由探针 P2/P5 实测佐证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目（R1/R3/R4/R5/R6）逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。**重设计直接承接 §1.6.0 调研选定算法族（纯 Vector 分块序列 kernel + fp32 中转 + 确定性分块归约 + UB 内因果滑窗 + 算术行掩码边界）。**

**重设计项 R1：执行结构重设计（单 prim_func 五 `T.Kernel` + GM 交接 → 工厂闭包调度 3 个独立 kernel）**

- **源方案**（步骤 1–19）：单 prim_func 内 5 个 `T.Kernel` 块；Pass1 把 `d_silu` 写 `dvhat_buf`，dconv 与 Pass1b 读回；Pass1b 把 `d(vhat)` 写 `dvhat_out`，Pass2 读回；5 次 grid 启动共享一个 launch 单元。
- **NPU 新算法**：**npnir 单 prim_func 不支持多 `T.Kernel`**（探针 P1 实测 grid 求值失败），改为工厂返回一个 **Python 闭包**，按序启动 **3 个独立 `@tilelang.jit(target="npuir")` kernel**（同族形态 `examples/engram/engram_bwd.py`）：
  - **K1（Pass 1 + dconv 部分和）**：grid `M×num_tiles`，每 block 处理 `BLK_S` 个输出 token；在 UB 内**以 3 行因果 halo（blk-内重算）** 载入 `vhat/rrms_v` 窗口，物化 `vn`（O3 CSE），重算 `conv_o`、`d_silu`；把 `d_silu` 写 `dvhat_buf`；**token 向 `reduce_sum(dim=0)`** 得 `dconv_w` 的 (KS,d) 每-block 部分和 → `ws_conv[·, cid, ·]`。
  - **K2（Pass 1b + Pass 2 + drms 部分和）**：grid `M×num_tiles`；载入 `d_silu` 的**3 行前向 halo**（K1 产物，clamp + 右边界算术掩码）做 **conv 转置**得 `d_vnorm`；随后**同 block 内**完成 RMSNorm(vhat) 反向（得 `d(vhat)`，含 `dY` 残差）与门控反向 + RMSNorm(H)/RMSNorm(k) 反向（得 dH/dk/dv），**消除源 Pass1b→Pass2 的 `dvhat_out` GM 往返**；token 向 `reduce_sum(dim=0)` 得 `drms_w_h/rms_w_v` 每-block 部分和 → `ws_h/ws_v[cid]`。
  - **K3（combine）**：grid 1；把 `ws_h/ws_v/ws_conv` 沿 block 轴归约成 `drms_w_h/drms_w_v/dconv_w`（确定性线性求和）。
- **语义保持论证**：
  1. **数学等价**：三 kernel 的划分与源五阶段**逐步对应**（K1=Pass1；K1 的 dconv 部分和 + K3=源 dconv kernel；K2=Pass1b+Pass2；K3=源 atomic 累加）；每 token 的 `d_silu`/`d(vhat)`/`dH/dk/dv` 计算内容与源逐式一致。block 划分只重排「哪个核处理哪些 token」，token 间除因果窗口（halo 覆盖）外无依赖。
  2. **边界语义**：K1 的 `vn` 左边界（`src_t<0`）与 K2 的 `d_silu` 右/左边界（`dst_t∉[0,T)`）由**索引 clamp + 算术行掩码**（§0.6 R4）吸收，无效行乘 0 → 与源 `if_then_else(...,0)` 逐元素一致。
  3. **精度**：全程 fp32，与源 `accum_dtype="float"` 同域；`ws_*` 部分和为 fp32 确定性累加（K3 沿 block 轴串行求和），与源 atomic 累加数学同式（顺序微差，§1.6.1 FULL 行 ≤6.1e-5）。**测试要求多次运行 bit 确定**（`test_engram_gate_conv_bwd.py` L97-L101 双跑比对）→ 本确定性 reduce 满足，源 `atomic_add` 形态不满足。
  4. **源优化意图承接**：#1「阶段融合于一 launch 单元」→ 多 kernel 闭包（npuir 结构约束，意图=减少 host 分派，承接）；#7「dconv 独立归约」→ K1 部分和 + K3（承接，且并行化）；#8「atomic 累加」→ 确定性 combine（承接意图，改确定性实现）；#9「out_idx 自动分配」→ 闭包手工分配（承接）；#14「d_silu GM 缓冲」→ 保留 1 写 1 读（承接）；#16「d(vhat) GM 缓冲」→ Pass1b+Pass2 融合消除（承接意图=减少往返）。

**重设计项 R2：计算域重设计（原生域 → fp32 中转）**

- **源方案**：`T.cast(...,"float")` 升 fp32 计算、末端 `T.cast(...,dtype)` 舍回（`accum_dtype="float"`）。
- **NPU 新算法**：fp16/bf16 输入先 `T.vcast(..., round_mode="rint")` 升 fp32 → 全部向量链 fp32 → 末端 `T.vcast(..., round_mode="rint")` 单次舍回（`dH/dk/dv`）。**bf16 为 API 强制**（数学 v-op 无 bf16）；fp16 保留同路径以对齐源 fp32 语义并统一实现。
- **语义保持论证**：① 与源计算域及契约 golden（`.float()` 全链路）**同域**，非替代；② 舍回点与源一致（仅 `dH/dk/dv`），`drms_w_*`/`dconv_w` 保持 fp32。**源优化意图承接**：源 #3「fp32 累加保障精度」→ 承接并**强化**（bf16 API 强制）。

**重设计项 R3：并行/内存结构重设计（2D grid + fragment + threads → 一维 Kernel + UB + 序列分块）**

- **源方案**：#5 `T.Kernel(seq_len, M, threads=t)`（每 block 一个 (batch,token)）+ #4 `T.alloc_fragment`，`threads ∈ {128,256,512}` autotune。
- **NPU 新算法**：一维 `T.Kernel(M*num_tiles, is_npu=True)`（每 block 一个 batch 的 `BLK_S` 个连续 token + halo）；所有中间量放 `T.alloc_shared`（UB，fp32）；`GM→UB` 用整块 `T.Parallel` 载入（clamp 运行时行索引，探针实测合法），向量链用 v-prefix op 作用于整 buffer；`threads`/autotune 移除（K3/K9）。
- **语义保持论证**：任务映射仅重排「哪个核处理哪些 token」，token 间除因果窗口外无依赖（窗口由 halo 覆盖），数学逐元素/逐行恒等；无累加顺序变化。**源优化意图承接**：源 #4「中间量驻留片上」→ UB 承接；源 #5「token 级并行」→ 一维 grid 的 block 级并行 + block 内 `BLK_S` 行向量化承接。

**重设计项 R4：边界重设计（`T.if_then_else` select → 索引 clamp + 算术行掩码）**

- **源方案**：#10：`raw = T.if_then_else(src_t≥0, cast(vhat[..]), 0)`、`d_si = T.if_then_else((dst_t≥0)·(dst_t<T), dvhat_buf[..], 0)`，用 select 语义吸收卷积左/右边界。
- **NPU 新算法**：**避免带 buffer 操作数的 `T.if_then_else`**（`TRAP-npuir-ifthenelse-parallel-segv`：npuir codegen 段错误 EXIT=139、无诊断）。改为：
  1. **索引 clamp**：载入行索引 `gc = T.min(T.max(gt,0), seq_len-1)`（纯标量 `T.max/T.min`），保证 GM 读不越界（探针 P2/P6 实测合法）；
  2. **算术行掩码**：每 halo 行计算标量掩码 `rmk[r] = cast(min(max(gt+1,0),1)·min(max(seq_len-gt,0),1), f32)`，经 `T.vbrc` 广播到 `(R,d)` 后 `T.vmul` 把无效行的 `vn`/`d_silu` 置 0（等价左/右零填充）。**掩码标量必须用编译期 unroll 的标量语句**（探针 P6a vs P6b：`T.Parallel` 内的运行时 `T.min/T.max` 标量 store 静默错值，见 `TRAP-npuir-parallel-runtime-minmax`）。
- **语义保持论证**：掩码对 `gt∈[0,seq_len)` 取 1、否则取 0，与源 `if_then_else` 在有效行上逐元素一致；无效行贡献 0（与源同）。**源优化意图承接**：源 #10「select 吸收边界」→ 算术掩码承接（语义等价，且规避 npuir codegen 缺陷）。

**重设计项 R5：跨 token 权重梯度归约重设计（`atomic_add` → 确定性分块部分和 + combine）**

- **源方案**：#7 独立 `T.Kernel(KS)` 串行 `M×seq_len` 得 `dconv_w`；#8 `T.atomic_add(drms_w_h/drms_w_v, per-token)` 在 (seq_len,M) grid 上累加。
- **NPU 新算法**：**npnir `T.atomic_add` 实测崩溃**（探针 P4 SIGABRT），且**测试要求多次运行结果 bit 一致**（原子累加不满足）。改为确定性两段式：
  1. **K1/K2 每 block 部分和**：对 `dconv_w`（K1）与 `drms_w_h/drms_w_v`（K2），先算每 token 贡献向量 `(BLK_S,d)`，再 **token 向 `T.reduce_sum(dim=0)`**（或用 `T.vadd` 逐 token 累加）得 `(1,d)` 部分和 → `T.copy` 写 `ws_conv[·,cid,·]` / `ws_h[cid]` / `ws_v[cid]`（探针 P2/P5 实测正确）；
  2. **K3 combine**：grid 1，把 `ws_*` 沿 block 轴（`NB ≤ 32`）`reduce_sum(dim=0)` → 写 `drms_w_h/drms_w_v/dconv_w`（unroll fill + 单次 `T.copy`，规避 `TRAP-npuir-parallel-scalar-store-race`）。
- **语义保持论证**：线性求和可结合，分块 + combine 与整段求和数学相等（确定性顺序）；`dconv_w[p,c] = Σ_{t: t-(3-p)≥0} d_silu[t,c]·vn[t-(3-p),c]` 按 block 分段求和再合并 = 全域求和。**源优化意图承接**：源 #7/#8「跨 token 权重梯度归约」→ 确定性两段式承接（意图不变，实现改为确定性且并行）。

**重设计项 R6：权重梯度与向量链的标量因子化（数学等价，设计新增，非源方案替换）**

- **源方案**（步骤 13–18）：逐处物化 `ddd_val`/`rrms` 与向量乘（如 `dh = rrms_h·w_h·ddot·k_norm − rrms_h³·h·dot_h_sum/d`）。
- **NPU 新算法**（承接 §1.6.1 O2/O4）：常量倒数折叠（`/d → ·fl(1/d)`、`/√d → ·fl(1/√d)`）；标量因子外提（`ddot`、`rrms_h`、`rrms_v`）；CSE 物化（`vn`、`h_norm`、`k_norm`、`d_vnorm`、`d(vhat)`）。
- **语义保持论证**：见 §1.6.1 O2/O3/O4（实数恒等/容差内等价，机器验证 32/32 EQUIV_PASS）。**源优化意图承接**：源 #15「激活重算」被保留，本项在其之上减少向量 op 数与中间 buffer。

> **承接说明（源优化意图不丢弃）**：#1→R1；#2→§1.6.2 #1；#3→R2；#4→R3；#5→R3（`threads` 舍弃）；#6→保留逻辑；#7→R5；#8→R5；#9→R1/R6；#10→R4；#11→保留；#12→保留；#13→舍弃 `threads` + 保留 `lru_cache`；#14→R1（保留 1 写 1 读）；#15→保留；#16→R1（融合消除）；数学语义（§0.1）逐项保持。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_bwd.py`（工厂 L38–L347、`EngramGateConvBwdKernel` L397–L499）。
- 源仓测试基准（**golden 直接移植对象**）：`examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py`（`ref_engram_gate_conv_bwd` L29–L74：autograd 反向 + `F.conv1d` 左零填充；用例 `EngramGateConvBwdFixture` L83–L93：`(1,32,256,fp16)` / `(1,32,256,bf16)` / `(2,64,512,fp16)` / `(2,16,256,bf16)` / `(2,512,256,fp16)`；容差 fp16 `atol=2e-1`、bf16 `atol=3e-1`、`rtol=2e-1`；**并含双跑确定性断言** L97–L101）。
- manifest：`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `EngramGateConvBwdOp`（signature/shape_rules/workloads/roofline；smoke `bwd-smoke-b1-s16-d256-fp16` + 3 个 full workload）。
- 同算子 NPU 只读参照：`examples/engram/engram_bwd.py`（3 个独立 kernel、每 token 一 block、`T.atomic_add`、`T.copy` 运行时行索引、语句级 `if`——本设计的显式改进基线）。
- 同族迁移档案（只读）：`examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md`（同族**正向**迁移范式：序列分块单 kernel + UB 内因果滑窗 + 设计期探针四约束 + 22/22 EQUIV_PASS）、`examples/engram_decode/_engram_decode_kernel/DESIGN.md`、`pattern-library/cases.md` `CASE-engram-gate-conv-fwd-migration` / `CASE-engram-decode-migration`、`capability-gaps.md` `CG-2026-0022`、`pattern-library/traps-compiler.md`（`TRAP-npuir-parallel-runtime-minmax` / `TRAP-npuir-vcast-shape-mismatch` / `TRAP-npuir-parallel-scalar-store-race` / `TRAP-npuir-ifthenelse-parallel-segv` / `TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce`）。
- **与前向案例的异同（显式对照）**：
  - **同**：纯 Vector（无 `T.gemm`→非 MixCV）；一维 grid `M·num_tiles` + 序列分块 + 3 行因果 halo；fp32 中转（bf16 API 强制）；索引 clamp + 算术行掩码边界；buffer 顶层无条件分配；`T.vbrc`+`T.vmul` 标量广播；GM 写回走 `T.copy`；末端 `vcast(rint)` + `T.copy`。
  - **异（本算子新增难点）**：① **反向**——`vhat/alpha/rrms_*` 是输入（不重算），但需重算 `conv_o` 取 silu'（#15）；② **conv 转置**（Pass1b，`d_silu` 的*前向* halo，右边界掩码）——前向是因果滑窗（左边界）；③ **跨 token 权重梯度归约**（`drms_w_h/drms_w_v/dconv_w`）——前向无归约，本算子须新增「部分和 + combine」（`atomic_add` 不可用）；④ **6 输出 + 混合 dtype**（3 个 `dY` dtype + 3 个 fp32）；⑤ **多 kernel**：前向单 kernel，本算子因多 `T.Kernel` 不支持而降为 3 kernel 闭包；⑥ **确定性要求**（测试双跑比对）。
  - **复用点**：前向的设备探针结论（动态行索引读合法、UB 仿射行偏移读合法、`T.if_then_else` 带 buffer 段错误、`vcast` 要求等 shape、`(R,1)` own-rows 写回需 unroll fill + `T.copy`）全部直接复用；本任务补充反向专属探针（多 T.Kernel / 部分和 / `atomic_add` / combine / 转置掩码，见 §1.6.3）。
- golden 以 §0.1 语义为依据（移植 `ref_engram_gate_conv_bwd`），**不复刻 §0.6/§1.6.3 的 NPU 结构**（保证验证独立性）。

### 0.8 修订记录（v1，`revision_index=1`）——相对上一版（v0）的关键调整

> 触发来源：Stage 2 检视**不通过**（1 阻塞 + 3 建议）。本节逐项说明处置，并给出「为何不会再犯同一错误」的机制性保障。**已通过部分（迁移三问解读 §0.1–§0.4、耦合性判定 §0.5 的 保留 6/等价替换 2/重设计 6/舍弃 2、R1–R6 重设计 §0.6、探针 P1–P6b 结论、MIX 非触发判定、§5.5 分核三要素、§1.6.0 调研四问与 D-2 估算、向量化替代 8/3 结论、golden 独立性）全部保留，未改动。**

**（A）阻塞项 1——§3.3 K2 伪代码 `drms_w_h` 别名错误（已修）**

- **问题**：v0 伪代码在 `dhn = ddot·kn` / `dkn = ddot·hn` 物化后，**就地覆写** `dhn→dH`（L855）、`dkn→dk`（L870），而 `drms_w_h` 部分和（v0 L874–879）在覆写**之后**才读取 `dhn`/`dkn`，实际算成 `Σ_t(dH·H·rrms_h + dk·k·rrms_k)`，与源（`engram_bwd.py` L282–285 / L305–308）及 §3.1 步骤 18 / §1.6.1 O2 / §0.1 要求的 `Σ_t(ddot·k_norm·H·rrms_h + ddot·h_norm·k·rrms_k)` 不符。
- **关键调整**：把 `drms_w_h` 部分和**前移**到 `dhn`/`dkn` 物化之后、覆写之前（§3.3 K2 修正后伪代码 + §3.1 步骤重排为 15→16(drms_w_h)→17(dH)→18(dk)）。选「前移」而非「另设独立 buffer」是为**不新增 UB buffer**（`d=512` UB 预算已紧，见建议 1）。
- **为何不会再犯**：① 该错误类别已在 `pattern-library/elementwise.md` **PL-1.23-vop-chain-inplace-alias**（status=verified，三/四任务实证）登记为「读取源数据的前置 op 必须发生在覆盖该 buffer 之前」的前提铁律；本设计 K2 的全链已按该纪律逐 buffer 审计（`hn`/`kn`/`dvn`/`vhat_l`/`h_l`/`k_l`/`v_l` 均不被覆写，仅 `dhn`/`dkn` 就地覆写且消费点已前移），结论写入 §3.3 实现注意 #13。② **机器可拦**：`verify_equiv.py` 新增 `drms_w_h`/`drms_w_v`/`dconv_w` 的 FULL 对照行（v0 的 FULL 仅覆盖 `dH/dk/dv`，故该错漏网）；实测把该错注回脚本即报 `EQUIV_FAIL`（违反率 100%，max_abs 0.32–0.39），护栏有效性已证（§1.6.1 结果表 + 本节末）。

**（B）建议 1——`d=512` 默认 `BLK_S` 超 UB（已修）**

- **问题**：v0 全局默认 `BLK_S=8`；`d=512`（manifest W3 `bwd-b2-s64-d512-fp16`）K2 复用后 ≈255KB > 192KB。
- **关键调整**：改为 **shape 分派** `BLK_S = 8 if d ≤ 256 else 4`（§5.2/§3.3）；`d=512` 时 `R = BLK_S+HALO = 7`，K2 复用后 ≈108KB < 192KB，K1 ≈83KB。§4.3/§4.5 补全 `bro_s (BLK_S,d)` 显式预算；§5.5 分核表 W3 逻辑核数随之 16→32（仍 ≤ 40）。该取值与同族正向迁移实测一致（`examples/engram_gate_conv_fwd/.../RETROSPECTIVE.md` Stage 3：`d=512` 用 `BLK_S=4`、`d=256` 用 `BLK_S=8`，UB 未溢出）。
- **为何不会再犯**：UB 预算表改为「按 `d` 档的 `BLK_S`」逐档核算（不再单点 8），并保留 §4.5 强制编译探针（`ub overflow, requires N bits`）+ `BLK_S` 回退链。

**（C）建议 2——优化项编号错位（已修）**

- **问题**：`verify_equiv.py` 内部用 O2–O5，本文档 §1.6.1 用 O1–O4，编号错位。
- **关键调整**：以**文档编号 O1–O4 为准**统一 `verify_equiv.py` 输出标签（O1 常量倒数折叠 / O2 标量因子外提 / O3 CSE 物化 / O4 fp32 中转），并在 §1.6.1 表中把 D-1 交叉引用改为 O1/O2a/O2b/O3；脚本 docstring 同步。

**（D）建议 3——§0.1 行号引用订正（已修）**

- `/float(d)` 的实际行号为 **L203 / L277 / L300**（v0 误写 L265），L256 为 `/T.sqrt(float(d))`；§0.1 已订正。

**（E）reduction FULL 行的意义与别名回归护栏反证（v1 新增说明；结果表正文见 §1.6.1）**

- `drms_w_h/drms_w_v/dconv_w` 为 fp32 契约输出，§1.6.1 的这 9 行 FULL 对照以 **fp32 域**（不 round 到 `dt`）与 fp64 golden 比对、门禁 `1e-4/1e-4`（区别于 `dH/dk/dv` 的 dtype 域门禁）。
- **回归护栏反证（证明护栏有效）**：把 v0 的「覆写后读取」别名错序注回 `verify_equiv.py` 后重跑，该 3 组输出的共 9 行 reduction FULL **全部失败**（违反率 100%，max_abs 0.32–0.39）——即 §0.8（A）别名错误可被机器拦截；不注回时 9 行全部通过（§1.6.1 结果表）。

**修订后回归**：`verify_equiv.py` 重跑 **TOTAL 32 ITEMS / 32 EQUIV_PASS / 0 EQUIV_FAIL**（原 23 行 + 新增 9 行 reduction FULL）；别名回归反证（把 v0 错序注回脚本）报 **3 EQUIV_FAIL**。

---
## 1. 概述

### 1.1 算子名称

`_engram_gate_conv_bwd_kernel`（项目 `engram_gate_conv_bwd`，目录 `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/`）。

### 1.2 功能描述

序列级 Engram GateConv **反向**融合算子（post-projection fusion grads）：给定输出梯度 `dY` 与前向保存量（`vhat/alpha/rrms_h/rrms_k/rrms_v`）及前向输入（`H/k/v/rms_w_h/rms_w_v/conv_w`），计算 **dH/dk/dv（(M,seq_len,d)，同 dY dtype）+ drms_w_h/drms_w_v（(d,)，fp32）+ dconv_w（(4,d)，fp32）**，其中 `drms_w_h/drms_w_v/dconv_w` 是**跨 M×seq_len 的权重梯度归约**。含 3 次 RMSNorm 反向、sigmoid 门控反向、因果深度卷积（k=4，无膨胀）的**转置**及 SiLU 反向、跨 token 权重梯度归约，属**多步融合 + 归约类**算子，**纯 Vector（无矩阵乘；投影反向由外部 GEMM 完成）**。支持 `float16 | bfloat16`（`alpha/rrms_*` 为 fp32）。**共 12 输入 + 6 输出**。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 执行结构由「单 prim_func 五 `T.Kernel` + 经 GM 复用 `d_silu`/`d(vhat)` + `atomic_add` 权重梯度」改为「**工厂闭包调度 3 个独立 kernel**（K1 Pass1+dconv 部分和 / K2 Pass1b+Pass2+drms 部分和 / K3 确定性 combine），Pass1b 与 Pass2 融合消除 `d(vhat)` GM 往返，权重梯度改确定性分块部分和 + combine」；② 计算域统一 fp32 中转（bf16 API 强制 + 源 fp32 语义）；③ 卷积左/右边界由 `T.if_then_else` 改为「索引 clamp + 算术行掩码」；④ 常量倒数折叠 + 标量因子化 + CSE（§1.6.1）。**计算语义（数学公式、I/O、边界、dtype）与源一致。**

### 1.3 数学公式

**反向解析式**（与 §0.1 一致；`vn[s] = vhat[s]·rrms_v[s]·w_v`，`σ` 为 sigmoid，`⊙` 逐元素）：

$$
\begin{aligned}
c_o[t] &= \sum_{p=0}^{3} \mathrm{cw}_p \odot vn[\,t-3+p\,],\qquad d_s[t] = dY[t]\odot\big(\sigma(c_o)+c_o\odot\sigma(c_o)\odot(1-\sigma(c_o))\big)\\
\mathrm{dconv}_{w}[p] &= \sum_{t:\,t-3+p\ge 0} d_s[t]\odot vn[\,t-3+p\,]\\
dvn[t] &= \sum_{p=0}^{3} \mathrm{cw}_p \odot d_s[\,t+3-p\,],\qquad \mathrm{drms}_{w,v} = \sum_{t} dvn[t]\odot \hat v[t]\odot rrms_v[t]\\
d\hat v[t] &= rrms_v\, w_v\odot dvn[t] - rrms_v^{3}\,\hat v[t]\,\tfrac{1}{d}\!\sum_c \hat v[t,c]\,w_v[c]\,dvn[t,c] \;+\; dY[t]\\
dv[t] &= \alpha[t]\odot d\hat v[t],\qquad d\alpha[t]=\sum_c d\hat v[t,c]\,v[t,c],\qquad dd[t]=d\alpha[t]\,\alpha[t]\,(1-\alpha[t])\,/\,\sqrt d\\
dH[t] &= rrms_h\, w_h\odot\big(dd[t]\odot k_n[t]\big) - rrms_h^{3}\,H[t]\,\tfrac{1}{d}\!\sum_c H[t,c]\,w_h[c]\,\big(dd[t]\odot k_n[t,c]\big)\\
dk[t] &= rrms_k\, w_h\odot\big(dd[t]\odot h_n[t]\big) - rrms_k^{3}\,k[t]\,\tfrac{1}{d}\!\sum_c k[t,c]\,w_h[c]\,\big(dd[t]\odot h_n[t,c]\big)\\
\mathrm{drms}_{w,h} &= \sum_t\Big( dd[t]\odot k_n[t]\odot H[t]\odot rrms_h + dd[t]\odot h_n[t]\odot k[t]\odot rrms_k \Big)
\end{aligned}
$$

其中 `h_n = RMSNorm(H,w_h)`、`k_n = RMSNorm(k,w_h)`、`RMSNorm(x,w)=x·rrms·w`（**除以真 d**，非 `d_padded`）。

**NPU 计算核（§1.6.1 优化后，fp32 计算域、末端单次 `vcast(rint)` 舍回）**：`·fl(1/d)`、`·fl(1/√d)` 常量折叠；`ddot`/`rrms_h`/`rrms_v` 标量外提；`vn/h_norm/k_norm/d(vhat)` CSE 物化；权重梯度 token 向 `reduce_sum(dim=0)` + combine。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法数学等价、在并行/内存实现上有差异：**

- **投影反向**：无（外部 GEMM 已完成 `dk/dv → dE/dW_K/dW_V`）。
- **K1（Pass 1 + dconv 部分和，§0.6 R1）**：一维 grid `M×num_tiles`；每 block 载入 `vhat/rrms_v` 的 `(BLK_S+3, d)` 窗口（3 行因果 halo，clamp + 左边界算术掩码）→ 物化 `vn`（O3）→ UB 内仿射滑窗得 `conv_o` → `d_silu`（`vsigmoid`）；写 `d_silu` 到 `dvhat_buf`；token 向 `reduce_sum(dim=0)` 得 `dconv_w` 的每-block `(4,d)` 部分和。
- **K2（Pass 1b + Pass 2 + drms 部分和，§0.6 R1）**：一维 grid `M×num_tiles`；载入 `d_silu` 的 `(BLK_S+3, d)` 窗口（3 行**前向** halo，clamp + **右**边界算术掩码）→ conv 转置得 `d_vnorm`；随后同 block 完成 RMSNorm(vhat) 反向（得 `d(vhat)`，含 `dY` 残差）与门控反向 + RMSNorm(H)/RMSNorm(k) 反向（得 `dH/dk/dv`）；token 向 `reduce_sum(dim=0)` 得 `drms_w_h/drms_w_v` 每-block 部分和。
- **K3（combine，§0.6 R5）**：grid 1；`ws_h/ws_v/ws_conv` 沿 block 轴 `reduce_sum(dim=0)` → `drms_w_h/drms_w_v/dconv_w`。
- **RMSNorm 反向（§1.6.1 O2/O4）**：`dot = Σ_d x·w·dx`；`dx_out = rrms·(w·dx − rrms²·x·dot·fl(1/d))`。
- 除上述外与 §1.6.0 调研选定算法族一致：纯 Vector、确定性归约、无中间 GM 往返（除 `d_silu` 一次交接契约）。

### 1.5 数据流图

```
GM[w_h,w_v,cw] --T.Parallel--> UB[kernel weight tiles (f32)]
K1: GM[vhat,rrms_v] --(clamped 运行时行索引, T.Parallel)--> UB[vhat_w (BLK+3,d) f32]
      --[vn=vhat·rrms_v·w_v (vbrc+vmul); UB 仿射滑窗; vsigmoid]--> UB[ds (BLK,d) f32]
      --T.copy--> GM[dvhat_buf] ;  --reduce_sum(dim=0)--> WS[ws_conv (4,NB,d) f32]
K2: GM[dvhat_buf] --(前向 halo, clamp+右掩码)--> UB[ds_w (BLK+3,d)]
      --[conv 转置: ds_w[i+3-p]⊙cw[p] FMA]--> UB[d_vnorm]
      --[RMSNorm(vhat) bwd + vsigmoid/gate bwd + RMSNorm(H)/RMSNorm(k) bwd]--> UB[dH/dk/dv]
      --vcast(rint)+T.copy--> GM[dH,dk,dv] ;  --reduce_sum(dim=0)--> WS[ws_h,ws_v]
K3: WS[ws_h,ws_v,ws_conv] --reduce_sum(dim=0)--> GM[drms_w_h,drms_w_v,dconv_w]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代，最后回答「在哪个轴向量化」。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**多步融合 + 跨 token 归约类**（3 次 RMSNorm 反向 + 门控反向 + 卷积转置 + 权重梯度归约 + 逐元素链），按 `references/algorithm-research.md` §2 属**完整调研**：R1 候选表、R2 在线算法、R3 四口径对比、R4 逐候选评估全部展开。**基线与候选同台**：源算法（单 prim_func 五 kernel + GM 交接 + atomic 累加）只是基线候选之一。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3/R4 |
|---|------|------------|-----------|----------|-----------|
| 0 | **基线：源算法** | 单 prim_func 五 `T.Kernel`；Pass1 写 `d_silu`→GM；dconv 独立串行归约；Pass1b 写 `d(vhat)`→GM；Pass2；`atomic_add` 权重梯度 | 定义式 | 5 grid 启动 / 2 次 GM 往返 / 原子累加 | ✅（基线） |
| 1 | **3-kernel 确定性分块（采纳）** | K1(Pass1+dconv 部分和) / K2(Pass1b+Pass2 融合 + drms 部分和) / K3(combine) | 数学等价（分块重排 + 线性求和结合律；§0.6 R1/R5） | 消 `d(vhat)` GM 往返；atomic→确定性；并行归约 | ✅（采纳） |
| 2 | **`atomic_add` 累加权重梯度（不采纳）** | 直接对 `drms_w_*`/`dconv_w` 原子累加 | 数学等价（无序求和） | — | ❌ 弃选：npnir Developer **`T.atomic_add` 实测 SIGABRT**（探针 P4）；且**测试要求双跑 bit 一致**，原子序不定不满足 |
| 3 | **常量倒数折叠（采纳）** | `/d → ·fl(1/d)`；`/√d → ·fl(1/√d)`（host 常量） | 容差内等价（D-1 O1，d 为 2 的幂时逐位一致） | `vdiv`→`vmul`（op 型替换） | ✅（§1.6.1 O1） |
| 4 | **标量因子化 + CSE（采纳）** | `ddot`/`rrms` 外提；`vn/h_norm/k_norm/d_vnorm/d(vhat)` 物化一次 | 实数恒等 / 逐位恒等（D-1 O2a/O2b/O3） | −3~6 向量 op/tile、−2 buffer | ✅（§1.6.1 O2/O3） |
| 5 | **fp32 中转计算域（采纳 = 保留 + API 强制）** | fp16/bf16 →`vcast(rint)`→ fp32 链 →`vcast(rint)` | 与源/契约同域（源即 fp32）；bf16 为 API 强制 | 正确性（bf16 可编译） | ✅（§1.6.1 O4） |
| 6 | **全融合 2-kernel（不采纳，备选）** | 单 kernel 融合 Pass1+1a+1b+2（窗 `BLK_S+6`）+ combine | 数学等价 | 消 `d_silu` GM 往返；少 1 launch | ⚠️ 备选（UB 压力大，§1.6.3 实验裁决） |
| 7 | **conv 转置 → GEMM/im2col（不采纳）** | 4-tap 深度卷积转置展开为 GEMM | 数学等价 | — | ❌ 弃选：**depthwise（逐通道）**，无共享权重矩阵，tap=4 极小，GEMM 化不成立（同族 `CASE-engram-gate-conv-fwd-migration` 结论，`tilelang-op-design/references/algorithm-candidates.md` ALG-conv） |
| 8 | **权重梯度 / 规约用 Cube `T.gemm`（不采纳）** | `dconv_w[p,c]=Σ_t ds[t,c]vn[t-3+p,c]` 表为 GEMM/GEMV | 数学等价 | MAC 落 Cube | ❌ 弃选：**逐通道相关（无跨 c 共享权重）**，非 GEMM；且 `CG-2026-0022` 实测「Developer auto CV-split 下 Cube `T.gemm` 与 shared 向量规约+广播无法共存」（详见 §1.6.3 MIX 评估） |
| 9 | **sigmoid 减 max 稳定化（不适用）** | `σ(x)=e^{x-m}/(1+e^{x-m})` | 数学等价 | — | ❌ 不适用：`1/(1+e^{-x})` 在 fp32 全域无上溢；`conv_o`/`ddot·k_norm` 量级 O(1)，无稳定化必要 |
| 10 | **两段式归约（block 内 partial + combine）（采纳）** | `reduce_sum(dim=0)` 每 block 部分和 → combine | 线性求和结合律（确定性） | 并行归约 + 确定性 | ✅（§0.6 R5） |
| 11 | **token-per-block（`BLK_S=1`，不采纳）** | 每 token 一 block | 数学等价 | — | ❌ 弃选：conv/转置需每 token 重算 3 行 halo（4× 冗余）且邻居读需 GM 往返（源基线形态） |

**R2 在线算法**：**结论 = 分段明确**（有明确结论 + 结构依据）：

- **3 处行内规约（`dot_v`/`dot_h`/`dot_k`）：无在线化收益**。分别作用于 `d_vnorm⊙vhat`、`H⊙dh_norm`、`k⊙dk_norm` 三个**不同量**，每次均为单遍 `(R,d)→(R,1)`；每个输入张量只读一次，**不存在可被「单遍/流式」消去的多遍基线**（结构依据；源 §0.3 步骤 9/15/17 为三个独立行归约）。
- **卷积/转置（`conv_o`、`d_vnorm`）：存在 block 级「流式/halo」变体**（本设计采纳）——沿 seq 轴为因果/反因果滑窗，每 block 维护 3 行历史即可 O(1) 状态推进；收益口径 = 扫描遍数（`d_silu`/`vhat` 邻居从「每 token 重算」降为「每 block 一次 halo 重算」）。**源算法已用「GM 缓存 + 运行时行索引」的类在线形式**（`dvhat_buf` + `dst_t`，`dvhat_out` + `dvhat_buf`），是本变体存在的直接证据（§0.4 #14/#16）。
- **跨 token 权重梯度归约（`drms_w_h/drms_w_v/dconv_w`）：不可在线化**。它们是**全域（M×seq_len）求和**（输出维度 d 与 token 无关），单遍扫描即需完整 token 集，无「running statistic」可提前消费；结构依据：源 §0.3 步骤 5/8/16/18 为全域归约，`dconv_w` 甚至需 `T.serial(M)×T.serial(T)` 完整遍历。→ 用**确定性两段式（partial + combine）** 而非在线。
- 归约维度 `d` 本身不可在线化（需完整行），但每 token 独立、单遍即可。

**R3 复杂度对比**（四口径；本算子**发射 + 访存混合受限**，且**多 pass 权重梯度归约**带来额外 GM 部分和流量；代表 workload W3 = `M2,S64,d512,fp16`）：

| 算法候选 | FLOPs（manifest 口径） | 访存量 (Bytes) | 扫描遍数（GM 读） | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------------|---------------|------------------|-------------|---------------------|
| 基线（源：5 grid，token-per-block，atomic） | `M·S·(24d)+20MS`×2.5 ≈ **3.94 M**（W3） | read ≈ `(1+4+…)·M·S·d·elem` + `d_silu`/`d(vhat)` fp32 各 1×；write `3·M·S·d·elem` + 2 workspace fp32；**W3 ≈ 2.9 MB**（`d_silu`/`d(vhat)` fp32 往返 + 卷积邻居重算） | 输入 ~1×；`vhat` 邻居 4×（重算）、`d_silu` 2×、`d(vhat)` 2× | fragment 约 20 个；无 GM workspace 归约部分 | grid `3·M·S`（W3=384 blocks）；无跨核 |
| **#1 3-kernel 确定性分块（采纳）** | 同基线 FLOPs | read ≈ `1.375·M·S·d·2`(vhat) + `1.375·M·S·d·4`(d_silu) + `4·M·S·d·2`(vhat/H/k/v) + ws；write `M·S·d·4`(d_silu) + `3·M·S·d·2`(dH/dk/dv) + ws；**W3 ≈ 2.7 MB** | 输入 3/（vhat 1.375×，d_silu 1.375×，其余 1×） | K1 ≈ 3×(R,d) f32+scratch；K2 ≈ 4×(R,d) f32+scratch | grid `M·num_tiles`（W3=16）+ K3 grid1；无跨核 |
| #6 全融合 2-kernel（备选） | 同基线 | read ≈ `3.375·M·S·d·2`+`4·M·S·d·2`；write `3·M·S·d·2`+ws；**W3 ≈ 2.3 MB**（消 `d_silu` 往返） | 输入 1.375×，无中间 GM 往返 | `(BLK+6,d)` 窗，峰值最高（UB 压力） | grid `M·num_tiles`（W3=16）+ combine；无跨核 |
| #2 atomic 累加 | 同 | 同基线但无 ws 部分和 | 同 | 同 | ❌ 崩溃 + 不确定性 |
| #10 两段式归约 | −（无额外 FLOPs） | +`ws` 部分和写读（W3 ≈ +0.4 MB） | + block 部分和写读 1× | + `ws_conv (4,NB,d)` f32 = 128KB（GM，非 UB） | ✅ 并行 + 确定性 |
| #8 归约用 Cube GEMV | — | — | — | — | ❌ 不适用（逐通道相关，无权重矩阵） |
| #7 conv→GEMM | — | — | — | — | ❌ 不适用（depthwise） |

> 口径说明：**FLOPs** 取 manifest `(M·S·(24d)+20MS)×2.5`（含 3 次 norm 反向平方和、门控点积、卷积 tap、silu'）；**访存量** = GM 读写总字节（输入读 + 输出写 + `d_silu` 交接 + 权重梯度部分和；分母是不可消除的 I/O）；**扫描遍数** = 各张量 GM 读取次数；**中间缓冲峰值** = UB 驻留上限（见 §4.5）。W3 manifest `bytes` = 1.34 MB（理想一次读写）。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 3-kernel 确定性分块（采纳）** | Vector（逐元素 + 行/列归约 + 超越函数，非 MAC 密集）✓ | UB 192KB：K1/K2 峰值 3~4×(R,d) f32 ≈ 76KB（W3，×1.7 ≈ 129KB）< 192KB ✓（`CONST-capacity-910B2C` 容量值**绑 910B2C/9362、本设备 910B3 未重验**） | `d∈{256,512}` 整除向量宽（fp16 ×8/fp32 ×4）✓；`(R,d)` 尾轴 32B 对齐 ✓ | 循环边界全静态（`BLK_S/R/K=4` 编译期常量）✓ | 向量链可 auto-multi-buffer 重叠；无 Cube → 非 MixCV、无 CV 同步 | ✅ **采纳** |
| 基线（源 5 grid，token-per-block，atomic） | Vector ✓ | UB ✓ | ✓ | ✓ | 5 次 grid 启动 + 2 次 GM 往返 + 原子序不定 | ⚠️ 可用但 GM 流量高、**atomic 崩溃**、不确定性（§0.6 R1/R5 重设计） |
| #6 全融合 2-kernel（备选） | Vector ✓ | UB 峰值最高（`BLK+6` 窗）⚠️ | ✓ | ✓ | 2 次启动；无 `d_silu` 往返 | ⚠️ 备选（§1.6.3 实验裁决） |
| #8 归约用 Cube GEMV | **Cube 落错单元**（逐通道相关、无权重矩阵） | L1/L0C 需额外占用 | — | — | **CV 混排** | ❌ 弃选（无权重矩阵 + `CG-2026-0022`） |
| #7 conv→GEMM | Cube（depthwise 无共享权重） | — | — | — | — | ❌ 不适用（depthwise） |
| #2 atomic 累加 | Vector ✓ | — | — | — | — | ❌ 弃选（SIGABRT + 不确定性） |
| #3/#4/#5/#10 | Vector（`vmul`/`vadd`/`vdiv`/`vsigmoid`/`vsqrt`/`vbrc`/`vcast`/`reduce_sum` 均 fp32 ✓、bf16 ✗→fp32 域） | +小 buffer / GM ws | ✓ | ✓ | — | ✅ 采纳 |

**v-prefix 元算 dtype 矩阵（决定 fp32 中转 + 排除候选）**：`vadd`/`vmul`/`vdiv`/`vsqrt`/`vsigmoid`、`reduce_sum` **均不含 bf16**（各 op doc `§2.2.1`：fp16 ✓ / fp32 ✓ / bf16 ✗/×；`reduce_sum` 还支持整型，`规约操作/T.reduce_sum.md`）；`vbrc` ✓（含 bf16，`shape操作/T.vbrc.md`）；`vcast` 支持 `f16→f32`/`bf16→f32`（`rint`）与 `f32→f16/bf16`（`数据类型转换操作/T.vcast.md`）。→ 全部向量数学一律 fp32 域，bf16 为 API 强制中转。**本算子不用 `rsqrt`**（`rrms_*` 是输入，非需重算的倒数）——故前向案例的「`vrsqrt` 近似」问题在本算子**不出现**（亦不引入）。

**调研结论**：**选定「纯 Vector 3-kernel 确定性分块（K1 Pass1+dconv 部分和 / K2 Pass1b+Pass2 融合+drms 部分和 / K3 combine）+ fp32 中转计算域 + 常量倒数折叠 + 标量因子化 + CSE 物化 + 索引 clamp/算术行掩码边界」**（候选 #1/#3/#4/#5/#10 采纳；#6 列为实验裁决备选）。

关键依据：① **R3/R4**：本算子纯逐元素/行/列归约，无 MAC 密集段，向量单元正确；发射 + 访存混合受限，`#1` 相对基线消 `d(vhat)` GM 往返并把权重梯度归约从 atomic（崩溃/不定）改为确定性并行两段式；② **R2**：3 处行规约为不同量、无多遍基线；卷积/转置的 block 级 halo 滑窗为在线形式；权重梯度归约为全域求和、不可在线化（结构依据）；③ **R1**：候选 #7/#8/#9 的弃选理由分别为数学不成立（depthwise）/无权重矩阵 + 能力缺口/无上溢问题；④ 全部采纳项（`vcast`、`vmul`、`vadd`、`vdiv`、`vsigmoid`、`vbrc`、`reduce_sum`、`copy`、`clear`）均有本仓 `docs/Tilelang.language/` 佐证 + `examples/` 实跑先例（§3.4）。

**调研范围（「无更优替代」依据）**：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md`（命中 `ALG-rmsnorm`〔行归约基准形态〕、`ALG-conv`〔im2col/implicit GEMM/Winograd——本算子 depthwise 逐通道、无共享权重，均不成立〕、`ALG-reduction`〔row-reduction 迁移基准形态〕、`ALG-elementwise`〔单遍 Vector 流〕）；② 本仓 `examples/`：`engram/engram_bwd.py`（同算子 NPU 参照——三 kernel/atomic/语句级 if）、`engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md`（同族正向迁移范式）、`engram_decode/_engram_decode_kernel/DESIGN.md`（同族 decode）；③ pattern-library（预注入 + `kb_search` 消费）：`CASE-engram-gate-conv-fwd-migration`（status=verified）、`CASE-engram-decode-migration`（status=verified）、`CG-2026-0022`（Cube+Vector MIX 缺口）、`TRAP-npuir-parallel-runtime-minmax` / `TRAP-npuir-parallel-scalar-store-race` / `TRAP-npuir-vcast-shape-mismatch` / `TRAP-npuir-ifthenelse-parallel-segv` / `TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce` / `TRAP-threads-kwarg-noop` / `TRAP-vbrc-empty-broadcast` / `TRAP-UB-multibuffer-inflation`（均 status=verified）、`CONST-vector-launch-overhead` / `CONST-capacity-910B2C` / `CONST-ascend910-9362-profile`（D-2 常数）；④ **外部已知算法族**（模型知识）：因果深度卷积/转置的直接法/implicit GEMM/状态缓存三类实现、RMSNorm 反向的 rsqrt/公式化、门控反演的因子化、跨 token 权重梯度归约的原子 vs 两段式——逐项由 §1.6.3 枚举与本表评估；⑤ `docs/Tilelang.language/` 全目录枚举（含 `规约操作/`、`数学操作/`、`数据类型转换操作/`、`shape操作/T.vbrc.md`、`内存操作/T.copy.md`、`原子操作/T.atomic_add.md`）确认采纳 API 存在性与弃选 API（`T.atomic_add`）的实测不可用。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)）**：

- **流量项** = `GM bytes ÷ 有效带宽`。W3（`M2,S64,d512,fp16`）实现口径 GM 流量 ≈ **2.7 MB**（§1.6.3 R3 表；含 halo 1.375×、`d_silu` fp32 交接、权重梯度部分和）。**本设备为 `Ascend910B3`（实查 20 AI Core / 40 Vector Core），`pattern-library/constants.md` 无 910B3 条目**，取 `CONST-ascend910-9362-profile`（9362 单 die 核→GM ≈ **1.06 TB/s**，status=verified）作分母并标注**跨 SoC 未重验（未文档化假设 + 估算依据）**：流量项 ≈ `2.7MB / 1.06TB/s ≈ 2.5 µs`。
- **发射项** = `向量 op 数 × ~0.5 µs/op`（`CONST-vector-launch-overhead`，**绑 910B2C 且 `kb_stale_check` 标 stale，910B3 未重验 → 未文档化假设 + 估算依据**）。W3 每 block 向量 op 数 ≈ **103**（K1 ≈33：载入 1 + 掩码 2 + `vn` 2 + conv clear/4-tap 9 + `vsigmoid` 1 + `d_silu` 4 + store 1 + dconv 4×(vmul+reduce+copy) 12 + 写 1；K2 ≈55：载入 1 + 掩码 2 + 转置 8 + RMSNorm 反向/gate 反向/drms 约 40 + 写 4；K3 ≈15）：16 blocks（1 波次）→ 发射项 ≈ **51 µs**。
- **容量项** = 无（中间量全 UB 驻留，无跨引擎 GM 往返；`CONST-store-fixpipe-gm-only` 不适用）。

**设计期估算（D-2 roofline）**：代表 workload W3（`M2,S64,d512,fp16`）估算下界 ≈ **max(2.5, 51, 0) ≈ 51 µs（发射项主导）**；W1 smoke（`M1,S16,d256`，2 blocks）≈ 同量级发射项主导。**不确定性声明**：① `0.5µs/op` 与 `1.06 TB/s` 均绑 910B2C/9362、`kb_stale_check` 标 stale，910B3 未重验，发射项可能**上偏**；② 对照证据：同族短链逐元素算子在实测中常由流量项主导、发射项被 MTE2 窗口隐藏（`bitwise_not` 1M 实测 7.72µs ≈ 流量）——本算子数据量小（≤2.7MB）且 op 链长（反向），发射项或主导；③ **`BLK_S` 与「K1/K2 融合度（3-kernel vs 2-kernel）」是 Stage 4 发射/流量调优杠杆**；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（纯 Vector 3-kernel 确定性分块 + fp32 中转）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| O1 | **常量倒数折叠** | `ddot = dα·α·(1−α)/T.sqrt(d)`；`…·dot/float(d)`（3 处 RMSNorm 反向） | `ddot = dα·α·(1−α)·fl(1/√d)`；`…·dot·fl(1/d)`（host 期 Python 常量） | 容差内等价；D-1 O1：`x·fl(1/d)` 与 `x/d` **逐位一致**（d=256/512 为 2 的幂，`fl(1/d)` 精确）；`dot·fl(1/√d)` vs `dot/√d` max_abs 2.4e-7、违反 0 → EQUIV_PASS | `vdiv`→`vmul`（op 型替换，消除 4 处逐元素除法） |
| O2 | **标量因子外提（`ddot`/`rrms_h`/`rrms_v`）** | `dot_h = Σ_c H·w_h·(ddot⊙k_n)`；`dh = rr_h·w_h·(ddot⊙k_n) − rr_h³·H·dot_h/d` | `dot_h = ddot·Σ_c(H·w_h·k_n)`；`dh = rr_h·(w_h·(ddot⊙k_n) − rr_h²·H·dot_h·fl(1/d))` | 实数恒等（标量对向量/求和线性）；D-1 O2a/O2b：max_abs ≤1.9e-6、违反 0 → EQUIV_PASS | −2~3 向量 op/Norm；`dot_h`/`dot_k` 的 pre-reduce 乘减少 |
| O3 | **CSE 物化（`vn`/`h_norm`/`k_norm`/`d_vnorm`/`d(vhat)`）** | 源各处重算 `normed`/`h_norm`/`k_norm`；`d(vhat)` 经 GM 重读 | 每个确定性中间量物化一次，多消费点复用 | 同一确定性 fp32 计算，物化值与重算值**逐位一致**；D-1 O3 max_abs=0.0 → EQUIV_PASS | 消除源 `normed` 重算（dconv 路）与 `d(vhat)` GM 重读；代价 +UB tile |
| O4 | **fp32 中转计算域**（保留源 + API 强制） | 概念上「输入 dtype 域直译」 | `vcast(rint)↑f32 → fp32 向量链 → vcast(rint)↓dtype` | 与源（`accum_dtype="float"`）及契约 golden（`.float()`）**同域**；bf16 为 API 强制（v-op 无 bf16）；D-1 FULL 三 workload 六输出（`dH/dk/dv/drms_w_h/drms_w_v/dconv_w`）均 EQUIV_PASS | 正确性：bf16 可编译；代价 +2 `vcast`/张量 |

**优化结论**：采纳 **4 项**（O1 常量倒数折叠、O2 标量因子外提、O3 CSE 物化、O4 fp32 中转）。**优化后公式**（供 §3.1 使用）：
- RMSNorm 反向：`dot = Σ_d x·w·dx`；`dx_out = rrms·(w·dx − rrms²·x·dot·fl(1/d))`
- 门控反向：`ddot = dα·α·(1−α)·fl(1/√d)`；`dα = Σ_d d(vhat)⊙v`；`dv = α_brc⊙d(vhat)`
- H/k 反向：`dot_h = ddot·Σ_d(H·w_h·k_n)`；`dH = rr_h·(w_h·(ddot⊙k_n) − rr_h²·H·dot_h·fl(1/d))`；`dk` 对称
- 权重梯度：`drms_w_h = Σ_t(ddot⊙k_n⊙H⊙rr_h + ddot⊙h_n⊙k⊙rr_k)`；`drms_w_v = Σ_t d_vnorm⊙vhat⊙rr_v`

公式层面已无进一步计算量/访存量可优化空间：**中间量全 UB 驻留**（除 `d_silu` 交接契约），**6 处规约非冗余**（不同数据/不同轴），**卷积为 depthwise 无权重矩阵**（#7/#8 不成立），**本算子无 `rsqrt` 重算**（`rrms_*` 为输入）。

**等价性机器验证（D-1）**：脚本 `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/verify_equiv.py`（CPU；候选式 = NPU fp32 流水线逐 op 复现；基线式 = fp64 精确 / 源式；判定 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`，fp32 `1e-4/1e-4`、fp16 `2e-1/2e-1`、bf16 `3e-1/2e-1`——取自 `test_engram_gate_conv_bwd.py` L95–L96）。优化项编号与本文档 §1.6.1 表格**统一为 O1–O4**（v0 脚本内部 O2–O5 的编号错位已消除）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**TOTAL 32 ITEMS, 32 EQUIV_PASS, 0 失败项**）：

| 采纳项 / 对照 | 候选式 | 基线式 | dtype | samples | 最大 abs 差 | 违反率 | 违反数 | 结论 |
|---------------|--------|--------|-------|---------|------------|--------|--------|------|
| O1 x/d → x·fl(1/d) (d=256) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O1 dot/√d → dot·fl(1/√d) (d=256) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O1 x/d → x·fl(1/d) (d=512) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O1 dot/√d → dot·fl(1/√d) (d=512) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 2.384e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE 物化 normed | `normed (materialised)` | `normed (recomputed)` | fp16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE 物化 normed | `normed (materialised)` | `normed (recomputed)` | bf16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2a factor ddot out of dot_h (d=256) | `ddot*Σ(h*w*kn)` | `Σ(h*w*(ddot*kn))` | fp16 | 4,096 | 1.431e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O2b factor rrms_h in dH (d=256) | `rr*(w*dhn − rr²*h*dot/d)` | `rr*w*dhn − rr³*h*dot/d` | fp16 | 1,048,576 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O2a factor ddot out of dot_h (d=256) | `ddot*Σ(h*w*kn)` | `Σ(h*w*(ddot*kn))` | bf16 | 4,096 | 1.431e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O2b factor rrms_h in dH (d=256) | `rr*(w*dhn − rr²*h*dot/d)` | `rr*w*dhn − rr³*h*dot/d` | bf16 | 1,048,576 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O2a factor ddot out of dot_h (d=512) | `ddot*Σ(h*w*kn)` | `Σ(h*w*(ddot*kn))` | fp16 | 4,096 | 1.907e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O2b factor rrms_h in dH (d=512) | `rr*(w*dhn − rr²*h*dot/d)` | `rr*w*dhn − rr³*h*dot/d` | fp16 | 2,097,152 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O2a factor ddot out of dot_h (d=512) | `ddot*Σ(h*w*kn)` | `Σ(h*w*(ddot*kn))` | bf16 | 4,096 | 1.669e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O2b factor rrms_h in dH (d=512) | `rr*(w*dhn − rr²*h*dot/d)` | `rr*w*dhn − rr³*h*dot/d` | bf16 | 2,097,152 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dH (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 8,192 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dk (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 8,192 | 3.815e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dv (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 8,192 | 1.526e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_h** (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 256 | 2.978e-09 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_v** (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 256 | 1.647e-09 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **dconv_w** (`M1 S32 d256 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 1,024 | 6.080e-08 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dH (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 65,536 | 1.907e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dk (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 65,536 | 1.526e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dv (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp16 | 65,536 | 1.526e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_h** (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 512 | 1.089e-08 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_v** (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 512 | 3.954e-09 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **dconv_w** (`M2 S64 d512 fp16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 2,048 | 1.634e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dH (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | bf16 | 32,768 | 9.537e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dk (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | bf16 | 32,768 | 7.629e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL dv (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | bf16 | 32,768 | 6.104e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_h** (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 256 | 7.645e-09 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **drms_w_v** (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 256 | 2.771e-09 | 0.000000 | 0 | **EQUIV_PASS** |
| O4 FULL **dconv_w** (`M1 S128 d256 bf16`) | `fp32 NPU bwd pipeline` | `fp64 golden` | fp32 | 1,024 | 1.231e-07 | 0.000000 | 0 | **EQUIV_PASS** |

> **reduction FULL 行意义（v1 新增）**：`drms_w_h/drms_w_v/dconv_w` 为 fp32 契约输出，故以 **fp32 域**（不 round 到 `dt`）与 fp64 golden 对照、门禁 `1e-4/1e-4`；这 9 行是 §0.8（A）别名错误的**回归护栏**，护栏有效性的反证（注回 v0 错序后 9 行全部失败）见 §0.8 修订记录（E）。

> 判定准则 = `|cand−base| ≤ atol + rtol·|base|`，违反数 0 即 PASS；覆盖随机张量 + 角点（`d/vhat` 有符号全域、卷积左/右边界、`dot` 大正/大负）。完整 32 行见脚本输出。FULL 行的 fp64 golden 与候选式用**同一批已舍入 dtype 输入**构造（隔离 fp32 路径误差与输入 cast 效应，避免 `TRAP-golden-input-cast-mismatch` 伪误差）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 3 处行内归约（`dot_v`/`dot_h`/`dot_k`） | 源 2D fragment + `T.reduce_sum(dim=1)` | `T.reduce_sum` on UB（`(R,d) f32 → (R,1)`，佐证 `docs/Tilelang.language/规约操作/T.reduce_sum.md`；先例 `examples/engram/engram_bwd.py` L169） | ✅ | — |
| 2 | 3 处跨 token 归约（`drms_w_h/drms_w_v/dconv_w`） | 源 `atomic_add` / 独立串行归约 | UB 内先 `T.vmul` 得 `(BLK_S,d)` 贡献，再 `T.reduce_sum(dim=0)` → `(1,d)` 部分和（探针 P2 实测 maxdiff 0.0）+ K3 combine（探针 P5 实测 4.8e-7） | ✅ | — |
| 3 | 逐元素链（cast/平方/乘/加/sigmoid/掩码） | 源 `T.Parallel(d_padded)` 标量逐元素 | `T.vcast`/`T.vmul`/`T.vadd`/`T.vsub`/`T.vsigmoid`（佐证各 op doc；先例 `examples/engram/engram_bwd.py`） | ✅ | — |
| 4 | **载入/写回的行索引** | 源 `src_t`/`dst_t` 运行时行索引（GM 读） | 本设计为 `T.Parallel` 元素读 + clamp（`T.min(T.max(gt,0),S-1)`）；写回 `T.copy` 区域 slice | ✅ | —（探针 P2/P6 实证动态行索引读写合法；前向探针 V1/V3 同证） |
| 5 | **卷积/转置窗口** | 源 `T.serial(4)` + 逐 tap 从 GM 读邻居行 | UB 内仿射行偏移读 `vn[i+p,:]`（探针 P6a）/ `ds_w[i+3-p,:]` + `cw[p,:]` `T.vbrc` 广播 + `T.vmul` | ✅ | —（探针 P6a 实测 maxdiff 0.0） |
| 6 | **标量广播到 d（`α⊙v` / `x⊙rrms⊙w` / `ddot⊙k_n`）** | 源标量 `alpha_val` 逐元素乘 | `T.vbrc`（`(R,1)→(R,d)`）+ `T.vmul`（佐证 `shape操作/T.vbrc.md`；`≥2` 向量操作数 / in-place 形态规避 `TRAP-npuir-vec-scalar-broadcast`） | ✅ | —（探针 P6a 的 `T.vbrc`+`T.vmul` 掩码即该形态） |
| 7 | **卷积/转置边界的行掩码构造** | 源 `T.if_then_else(src_t≥0, 载入, 0)` | 索引 `T.max/T.min` 标量 clamp + **编译期 unroll** 标量掩码 `rmk`（探针 P6a） | ✅ | —（`T.Parallel` 内运行时 `T.min/T.max` 标量 store 静默错值——探针 P6b：`TRAP-npuir-parallel-runtime-minmax`；带 buffer 操作数的 `T.if_then_else` 段错误——`TRAP-npuir-ifthenelse-parallel-segv`） |
| 8 | conv/转置 tap 展开（p=0..3） | 源 `T.serial(4)` + 运行时 `src_t` | host 层 Python 展开（`p` 为编译期常量偏移；`vn[i+p]`/`ds_w[i+3-p]` 静态偏移） | ✅（host 展开） | —（`TRAP-C13`：prim_func 内 `for p in range()` 是运行时 Var，故在 factory 层展开） |
| 9 | 核级任务映射 | 源 `T.Kernel(seq_len, M)` + `threads` | 一维 `T.Kernel(M*num_tiles, is_npu=True)` | ❌ | block 级任务映射（每 block 少量标量）；`threads` 在 npuir 无效（`TRAP-threads-kwarg-noop`），无逐元素等价 API；元素级并行由 v-prefix 整 buffer 完成 |
| 10 | 跨 block combine | 源 `atomic_add`（分散累加） | K3 grid=1 内 `T.copy` 载入 `(NB,d)` + `reduce_sum(dim=0)` | ❌（列为独立 kernel，非带内循环） | 跨 block 求和需全部 block 完成——**不能**在元素级/带内循环消除；`atomic_add` 实测崩溃（探针 P4），故独立确定性 combine kernel（`T.reduce_sum(dim=0)`，探针 P5） |
| 11 | host 侧 pad/crop（尾维 pad、输出 crop） | `F.pad`/切片 | 保留在 host（元数据/视图级） | ❌ | host 侧 shape/stride 元数据处理（不在 kernel 内）；`ascend-constraints.md` §4 允许 stride/shape 视图操作、禁改输入内容——本算子 host 仅 pad/crop |

**向量化结论**：逐元素/归约/搬运/窗口计算点（#1–#8）**已全部向量化或 host 层展开**（替代 API 均有 docs 佐证 + `examples/` 实跑先例）；保留 3 处非逐元素点（#9 核级任务映射、#10 跨 block combine、#11 host 元数据），逐项理由如上，**无「实现简单」类理由**。与 §6 循环结构一致——K1/K2 内**无逐元素标量循环**；唯一串行结构为 conv/转置的 4-tap 编译期展开（#8）、block 级任务映射（#9）与 K3 的 block 轴归约（#10）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**候选枚举**（本算子的非 batch 维 = `seq_len`（卷积/转置/任务轴）与 `d`（行内规约/逐元素轴）；I/O 布局 = 行主序、尾轴 `d` 连续，是契约；核内布局是设计变量）：

| 候选 | 向量化轴 / 布局 | 轴质量（整除/尾 lane 浪费） | repack 代价 | UB 预算 | 结论 |
|------|----------------|---------------------------|-----------|---------|------|
| **A（主选）** | **核内 = I/O 布局**；向量化轴 = 尾轴 `d`（连续）；`seq_len` 轴 block 级分块（每 block `BLK_S` token + 3-halo）；行内归约沿 `d` 水平 `reduce_sum(dim=1)`、token 归约沿 token 轴 `reduce_sum(dim=0)` | `d∈{256,512}` 整除向量宽（fp16 ×8 / fp32 ×4）→ 尾 lane 浪费 0%；`(R,d)` 尾轴 32B 对齐 ✓ | 无（I/O 布局 = 核内布局） | K1/K2 各 3~4×(R,d) f32；d=256（`BLK_S=8`）≈76KB→×1.7≈129KB；d=512（`BLK_S=4`，§4.5）≈64KB→×1.7≈109KB，**均 < 192KB ✓** | ✅ **主选** |
| B（备选，实验裁决） | **全融合 2-kernel**：单 kernel 融合 Pass1/1a/1b/2，`vhat` 窗口 `BLK_S+6`（前后 halo），无 `d_silu` GM 往返 + K3 combine | 同 A | 无（同布局） | `(BLK_S+6,d)` 窗，峰值最高（≈1.5×A） | ⚠️ 备选（结构完整，达 Stage 4 可直接实现；见下裁决三件套） |
| C（弃选） | 核内 `T.transpose` 把 `seq_len` 变尾轴（让卷积沿连续轴） | `seq_len∈{16..128}` 长度尚可 | `T.transpose` 二轴交换链（UB 级，`layout.md` PL-1.1 实测 ~5.4µs/链）；`(R,d)`→`(d,R)` | 转置中间 tile 翻倍 | ❌ 弃选：**无收益**——卷积/转置沿 seq 轴是滑窗（跨步访问），即便转置后沿连续轴仍需 4 次移位读；且 repack 引入 UB 级交换链开销与容量。**非 dtype 障碍**（`docs/Tilelang.language/创建操作/T.transpose.md` 逐格取值：fp16/fp32/bf16 均 √） |
| D（弃选） | host permute 把 `seq_len`/`d` 换序 | 同 C | host permute（`layout.md` PL-1.3 实测 106–147µs） | — | ❌ 弃选：host permute 净亏（`PL-1.3` status=verified），且改张量内容违反 host 契约 |
| E（弃选） | 垂直扫描归约（lane 映射独立行、串行步进 `d`） | — | — | — | ❌ 弃选：核内归约维 = 尾轴 `d`，水平归约（`reduce_sum` 1 op）优于垂直扫描（O(d) 次串行）；先例 `examples/engram/engram_bwd.py` 用水平归约 |
| F（弃选） | token-per-block（`BLK_S=1`，grid=`M·seq_len`） | 同 A | 无 | 峰值最小 | ❌ 弃选：并行度最高（W3=128 blocks）但**每 token 需重算 4 个 halo 行**（4× 冗余）且邻居读需 GM 往返（=源基线）；相对 A 无收益 |
| G（弃选） | conv/转置 → im2col/GEMM（Cube 路径） | — | — | L1/L0C 占用 | ❌ 不适用：**depthwise（逐通道）**、无共享权重矩阵（§1.6.0 #7） |
| H（弃选） | 权重梯度/规约用 Cube GEMV | — | — | L1/L0C 占用 | ❌ 弃选：逐通道相关非 GEMM + `CG-2026-0022`（见 MIX 评估） |

**弃选论证证据**：候选 C 依赖的 `T.transpose` 已枚举并按 `negative-claim-evidence.md` §2.2 在 `docs/Tilelang.language/` 子目录检索（命中 `创建操作/T.transpose.md`；§2.2.1 Ascend dtype 行 fp16/fp32/bf16 均 √；限制为仅二轴交换）——该候选**弃选不依赖 dtype 负向断言**，仅依据「滑窗跨步 + repack 开销无收益」；候选 D 引用 `layout.md` PL-1.3（status=verified）；候选 E 为结构分析（水平 vs 垂直）；候选 F 为冗余度分析（4× vs 1.375×）；候选 G 为数学不成立（depthwise）；候选 H 为无权重矩阵 + 能力缺口（`CG-2026-0022`）。**A–H 全行负向论断已逐条复核**。

**MIX（Cube + Vector）可行性评估** ⭐（预注入要求）：

- **本算子是否属于 MIX（Cube `T.gemm` + Vector 混合）结构？——否。** 源算子 `engram_bwd.py` **无任何 `T.gemm`/matmul**（投影反向 `dE=dk@W_K^T+dv@W_V^T`、`dW_K=E^T@dk` 在外部 GEMM 完成；本 kernel 只剩 3 次 RMSNorm 反向 + 门控反向 + 深度卷积转置 + SiLU 反向）。三处规约是**逐通道自身点积/平方和/相关**（`Σ_c d(vhat)⊙v`、`Σ_c H⊙dh_norm`、`Σ_c k⊙dk_norm`，以及权重梯度 `Σ_t …`），**无跨通道共享权重矩阵**，不是 GEMM。故本设计**纯 Vector**（`T.Kernel(..., is_npu=True)` + v-prefix + `reduce_sum`），**非 MixCV、无 CV 同步**，与 `examples/engram/engram_bwd.py` / `examples/engram_gate_conv_fwd`（同族）同路线。
- **`CG-2026-0022` 影响评估**：该缺口 = 「Developer auto CV-split 下 Cube `T.gemm` 与 shared 向量规约 + 标量广播链**无法共存**」（BiShengIR verifier 三类拒绝：`cbuf to cc` / `load src==gm` / `expected result type with stride`），**仅在同时含 `T.gemm` 与 shared 向量规约广播时触发**。本设计**无 Cube 段 → 不触发该缺口**（探针 P1–P6 实证纯 Vector 多 kernel 结构可编译运行）。**附带缺陷**（`TRAP-npuir-fp32-vector-reduce`：GM 输入张量本身为 fp32 时纯向量 shared 规约 + 标量广播被拒，fp16 经 `vcast` 升 fp32 通过）**亦不触发**——本算子输入为 fp16/bf16（manifest 契约），fp32 仅在片上 UB 中转（§0.6 R2）。
- **若强行选 Cube 路线的降级方案**：仅当未来把外部投影 GEMM 反向也融合进来时才会成为 MIX；届时须先做 `CG-2026-0022` 的**最小 MIX 探针**（gemm + 归约 + 广播），失败则回退本设计的纯 Vector 路线。**当前不需要**。

**设计期探针（D-3，实测本设备/工具链，脚本落 `history_version/`）**：本设计的反向专属不确定构造已用最小探针实测（未实现完整算子，仅最小张量；脚本 `history_version/design_probe_bwd_structure.py`，每变体独立子进程）：

| 探针 | 构造 | 实测结果 | 结论 |
|------|------|---------|------|
| P1 | 单 prim_func 内两个 `with T.Kernel` | **失败**（`Failed to evaluate grid expression 'None'`） | ❌ 多 `T.Kernel` 不支持 → 3-kernel 闭包（§0.6 R1） |
| P2 | 每 block token 向 `reduce_sum(dim=0)` → `(NB,d)` workspace | **运行成功，maxdiff 0.0**（对照参考） | ✅ 部分和归约可行（§0.6 R5） |
| P3 | `T.Parallel` 内 `stg[0,j] += ...`（`T.serial` 累加） | **运行成功但值错且不定**（WSsum 多次运行不同，如 −30.87 / 57.19，期望 −54.59） | ❌ 串行 `+=` 累加不可用 → 用 `reduce_sum(dim=0)`/`T.vadd` |
| P4 | `T.atomic_add`（Per-block grid → 全局 `(KS,d)`） | **SIGABRT（exit −6）** | ❌ `atomic_add` 崩溃 → 确定性两段式（§0.6 R5） |
| P5 | 独立 grid=1 combine kernel（`(NB,d)` 列归约 → `(d,)` GM） | **运行成功，maxdiff 4.8e-7** | ✅ combine 形态可行（§0.6 R5） |
| P6a | conv 转置仿射偏移 `ds[i+3-p]` + **unroll 掩码**（`vbrc`+`vmul`） | **运行成功，maxdiff 0.0** | ✅ 主选转置形态可行（§0.6 R1/R4） |
| P6b | conv 转置 + **`T.Parallel` 内运行时 `T.min/T.max` 掩码** | **运行成功但值错**（maxdiff 9.26） | ❌ 并行运行时 minmax 标量掩码错值 → 用 unroll（`TRAP-npuir-parallel-runtime-minmax`） |

> 工具链戳同上。探针结论已同步为本设计决策依据（§0.6 R1/R4/R5 / §3.3 / §6）。**复用前向探针结论**：动态行索引读合法（前向 V1）、UB 仿射行偏移读合法（前向 A）、store 处 direct-cast 被 MLIR verifier 拒（前向 A 首版）、`(R,1)` own-rows 写回需 unroll fill + 单次 `T.copy`（`TRAP-npuir-parallel-scalar-store-race`）、`T.vcast` 要求 src/dst 等 shape（`TRAP-npuir-vcast-shape-mismatch`）——本设计直接采纳。

**实验裁决三件套（判定裕度落在未实证常数内 → 强制）**：

- **裁决对象**：候选 A（3-kernel 确定性分块，主选）vs 候选 B（全融合 2-kernel，备选）。判定裕度依赖的未实证常数：910B3 的**向量发射成本**（`CONST-vector-launch-overhead` 绑 910B2C 且 stale）与**核→GM 带宽**（`CONST-ascend910-9362-profile` 绑 9362）——B 省一次 kernel 启动与 `d_silu` GM 往返（W3 约 0.62MB），但 UB 峰值更高。
- **主选方案（默认实现，进 Stage 3）**：候选 A——3 个独立 kernel（K1/K2/K3），一维 grid `M·num_tiles`、`BLK_S=8`(d≤256)/`BLK_S=4`(d=512)（`R=11`/`R=7`）；结构见 §3.3 / §4 / §6。
- **备选方案（≤1，结构完整，达 Stage 4 可直接实现）**：候选 B（全融合 2-kernel）——
  - **常驻 buffer（全列出，@d=512, BLK_S=8, 窗 `W=BLK_S+6=14`）**：`vhat_w(14,512) f32`+`vn_w(14,512)`+`dsw(14,512)`（=3×28KB）+`dy_w(11,512)`（22KB）+`vhat_own/H/k/v(4×8×512)`（16KB×4）+`h_n/k_n(2×16)`+`d_vnorm(16)`+`dvhat(16)`+`conv(16)`+权重/掩码小 buffer ≈ 8KB → 复用后峰值 ≈ 6×(14,512) ≈ **168KB raw**（×1.7 ≈ 286KB ⚠️ 超 192KB）→ 需 `BLK_S=4`（`W=10`，≈ 120KB raw ×1.7 ≈ 204KB ⚠️ 仍紧）或更激进复用。**该预算不确定性正是裁决所依赖的未实证常数。**
  - 循环：单 kernel grid `M·num_tiles`，`T.copy`/`T.Parallel` 载入 `vhat` 窗 `[t0-3, t0+BLK_S+2]` + clamp + 双侧算术掩码 → 全链 fp32 → 写 `dH/dk/dv` + `ws_*`；combine 为第二 kernel。**建议 `ws_*`/workspace 由 host `torch.empty` 显式传入（规避多输出 + workspace 组合风险，`TRAP-zero-input-crash` 经验）**。
  - 分核三要素：grid `M·ceil(seq_len/BLK_S)`（同主选）；物理核 = **40**（纯 Vector 翻倍）；W3=16 ≤ 40 → 无需适配。
  - 该变体与主选数学等价（§1.6.1 全项复用；FULL 行覆盖）；差异仅为结构（少 1 次启动 + 无 `d_silu` 往返，代价 UB 峰值）。**保留意义 = 主选若在 Stage 3 遇到 UB 溢出 / auto-multi-buffer 支配性问题时的结构备选；若 UB 实测可行且少 1 launch 有收益则翻转。**
  - **Stage 4 UB 编译探针计划（强制；B 被裁决选中或以兜底启用时执行）**：① 上靶前 `tilelang.compile(main, target="npuir")` 编译 smoke 与 W3 两档，读 `ub overflow, requires N bits` 报文取精确实测需求（`TRAP-UB-multibuffer-inflation` / `CG-2026-0008`：手工逐 buffer 预算 ≠ 可编译证明）；② 若实测需求超手工预算 ×1.12，按 `BLK_S=8→4`（窗线性缩小）或合并 `vhat_w/vn_w` 生命周期回收；③ 探针结果回写本节（`perf_feedback.md` 路径）。
- **裁决计划**：代表 shape = 全部 3 个 full workload + smoke；指标 = `msprof op` Task Duration（median-of-20）；未知常数反解 = 用 A/B 实测时长反推 910B3 的有效 per-op 发射成本与核→GM 带宽；**判定阈值**：备选 B latency < 主选 A × (1−0.10) 才翻转（10% 噪声余量，`BP_run_state_bimodality` ±3–5%）；裁决结果回写 §1.6.3。**主选 A 已在探针中证明可编译（P1–P6）；备选 B 的 UB 可行性未实证，须 Stage 4 编译探针。**
- **shape 特化分派可行性**：工厂 `lru_cache` 按 `(M,seq_len,d,eps,dtype)` 特化，**支持按 shape 分派主选/备选 / 按 shape 分派 `BLK_S`**（本设计已按 `d` 分派：`d≤256 → BLK_S=8`、`d=512 → BLK_S=4`，§5.2），裁决须评估**按 workload 分派**而非全局二选一。

**决策落地约束**：主选方案的向量化轴 = 尾轴 `d`，行归约维 = `d`（水平 `reduce_sum(dim=1)`），token 归约维 = token 轴（`reduce_sum(dim=0)`）；该轴同步写入 §3.3 伪代码（各 v-op 作用于 `(R,d_padded)` buffer）与 §6 循环结构（无逐元素标量循环）。备选 B 的结构设计独立记录于本节（不与主选混排）。

**布局决策结论**：选定**主选 A**——核内布局 = I/O 行主序（**无 repack**）、向量化轴 = 尾轴 `d`、`seq_len` 轴 block 级分块（`BLK_S` token + 3-halo）、行内水平归约 + token 轴 `reduce_sum(dim=0)`、卷积/转置 UB 内仿射滑窗、权重梯度确定性分块部分和 + combine。**弃选候选**：B（全融合 2-kernel，列为实验裁决备选）、C（核内 `T.transpose`，无 dtype 障碍但滑窗跨步 + repack 无收益）、D（host permute，106–147µs 净亏）、E（垂直扫描归约，劣于水平归约）、F（token-per-block，4× 冗余计算）、G（conv→GEMM，depthwise 不成立）、H（Cube GEMV，无权重矩阵 + `CG-2026-0022`）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`ascend_mode="Developer"`；脚手架 wrapper `EngramGateConvBwdKernel.ascend_mode="Developer"` 已固定，与 `op_requirements.programming_mode=developer` 一致）。

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | **纯 Vector**：3×RMSNorm 反向 + 门控反向 + 深度卷积转置 + SiLU 反向，无 `T.gemm`/matmul（投影反向在外部） | 仅需 UB；Developer `T.reduce_sum` on UB + v-prefix 足够，无需 Expert 手动 L1/L0C 控制 |
| 是否含归约 | 6 处规约（`dot_v/h/k` 行向 + `drms_w_h/v`、`dconv_w` 跨 token） | Developer 的 `T.reduce_sum`（dim=1 / dim=0）足够 |
| 是否需流水线 | 向量链线性；分块 tile 间无依赖（halo 由 block 内重算覆盖） | 无需 `T.Pipelined`；auto-multi-buffer 处理片内重叠 |
| 向量链 API | `vcast/vmul/vadd/vsub/vdiv/vsigmoid/vbrc/reduce_sum` | v-prefix（AGENTS.md 强制） |
| 同步需求 | 无跨核 workspace、无 Cube 段（多 kernel 间为顺序启动） | Developer 自动同步即可 |
| 是否 persistent | **否**（grid = `M·num_tiles` ≤ 40） | 保持非 persistent 简洁结构（规避 `TRAP-DEVMODE-PERSIST-GEMM` 类风险） |

决策树路径：`纯 Vector（无 matmul）→ 非 persistent → Developer（自动同步）→ Developer 模式`。**先例**：`examples/engram/engram_bwd.py`（同算子 Developer 纯 Vector 多 kernel）、`examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel`（同族正向 Developer 纯 Vector）、`examples/norm/rms_norm.py`（Developer + `T.reduce_sum`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared((R, d_padded), "float32")`（UB 承载向量链与归约）；dtype 输出暂存 `T.alloc_shared((BLK_S, d_padded), dtype)`；跨 block 部分和用 GM workspace（`ws_h/ws_v/ws_conv`） |
| 计算方式 | `T.Parallel`（载入/逐元素）+ v-prefix 整 buffer 向量链 + `T.reduce_sum(dim=1)` 行归约 + `T.reduce_sum(dim=0)` token 归约 |
| 同步 | Developer 自动同步（kernel 内）；多 kernel 间顺序启动（host 侧） |
| 分核 | 一维 `T.Kernel(M*num_tiles, is_npu=True)`（K1/K2），K3 `T.Kernel(1, is_npu=True)`（§5.5） |

---
## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（fp32 中转 + 常量倒数折叠 + 标量因子外提 + CSE）。窗口行索引 `r` 遍历 `R = BLK_S + 3` 行；`·_brc` 表示经 `T.vbrc` 广播到 `(R,d_padded)` 的行标量。`INV_D = fl(1/d)`、`INV_SQRT_D = fl(1/√d)` 为 host 期 Python 常量。`dt` = 输入 dtype（fp16/bf16）。

| 步骤 | Kernel | 数学表达 | 说明 |
|------|--------|----------|------|
| 0 | — | `t0 = tile·BLK_S`；`num_tiles = ceil(seq_len/BLK_S)`；`NB = M·num_tiles` | 分块定位（编译期） |
| 1 | K1 | `gt1 = t0-3+r`；`gc1 = min(max(gt1,0), seq_len-1)`；`rmk1[r] = 1[0≤gt1<seq_len]`（unroll） | vhat 窗口定位 + clamp + 左边界掩码 |
| 2 | K1 | `vn[r,c] = f32(vhat[bid,gc1,c])·rrms_v[bid,gc1]·w_v[c]·rmk1[r]` | `vn` 物化（窗口） |
| 3 | K1 | `c_o[i,c] = Σ_{p=0}^{3} cw[p,c]·vn[i+p,c]`（i∈[0,BLK_S)） | 因果 DWConv（tap 编译期展开） |
| 4 | K1 | `ds[i,c] = f32(dY[bid,t0+i,c])·(σ(c_o)+c_o·σ(c_o)·(1-σ(c_o)))` | SiLU 反向（`vsigmoid`） |
| 5 | K1 | `dconv_ws[p,c] += Σ_i ds[i,c]·vn[i+p,c]` | dconv 部分和（`reduce_sum(dim=0)`） |
| 6 | K1 | `dvhat_buf[bid,t0+i,c] = ds[i,c]` | `d_silu` 落盘（K1→K2 交接） |
| 7 | K2 | `gt2 = t0+r`；`gc2 = min(max(gt2,0), seq_len-1)`；`rmk2[r] = 1[0≤gt2<seq_len]`（unroll） | `d_silu` 窗口定位 + 右边界掩码 |
| 8 | K2 | `dsw[r,c] = dvhat_buf[bid,gc2,c]·rmk2[r]` | 载入 + 掩码 |
| 9 | K2 | `dvn[i,c] = Σ_{p=0}^{3} cw[p,c]·dsw[i+3-p,c]` | conv 转置（tap 编译期展开） |
| 10 | K2 | `drms_w_v_ws[c] += Σ_i dvn[i,c]·f32(vhat[bid,t0+i,c])·rrms_v[bid,t0+i]` | drms_w_v 部分和 |
| 11 | K2 | `dotv[i] = Σ_c f32(vhat)·w_v·dvn`；`dvhat[i,c] = rrms_v·(w_v·dvn − rrms_v²·f32(vhat)·dotv·INV_D) + f32(dY)` | RMSNorm(vhat) 反向 + dY 残差 |
| 12 | K2 | `dv[i,c] = α·dvhat[i,c]` | `dv`（广播 α） |
| 13 | K2 | `dα[i] = Σ_c dvhat[i,c]·f32(v)`；`dd[i] = dα·α·(1−α)·INV_SQRT_D` | 门控反向 |
| 14 | K2 | `hn[i,c] = f32(H)·rrms_h·w_h`；`kn[i,c] = f32(k)·rrms_k·w_h` | h_norm/k_norm 物化（CSE） |
| 15 | K2 | `dhn[i,c] = dd[i]·kn[i,c]`；`dkn[i,c] = dd[i]·hn[i,c]`（O3 物化） | d(h_norm)/d(k_norm) |
| 16 | K2 | **（必须在步骤 17/18 就地覆写 `dhn`/`dkn` 之前执行——PL-1.23 读取先于覆写）** `drms_w_h_ws[c] += Σ_i (dhn[i,c]·f32(H)·rrms_h + dkn[i,c]·f32(k)·rrms_k)` | drms_w_h 部分和（源 L282–285 / L305–308） |
| 17 | K2 | `doth[i] = dd[i]·Σ_c f32(H)·w_h·kn`；`dH[i,c] = rrms_h·(w_h·dhn − rrms_h²·f32(H)·doth[i]·INV_D)`（**就地覆写 `dhn → dH`**） | RMSNorm(H) 反向（因子化） |
| 18 | K2 | `dotk[i] = dd[i]·Σ_c f32(k)·w_h·hn`；`dk[i,c] = rrms_k·(w_h·dkn − rrms_k²·f32(k)·dotk[i]·INV_D)`（**就地覆写 `dkn → dk`**） | RMSNorm(k) 反向 |
| 19 | K2 | `dH/dk/dv[bid,t0+i,c] = dt(rint)` | 单次 `vcast(rint)` + `T.copy` 写回 own 行 |
| 20 | K3 | `drms_w_h[c] = Σ_{blk} ws_h[blk,c]`；`drms_w_v[c]` 同；`dconv_w[p,c] = Σ_{blk} ws_conv[p·NB+blk,c]` | 确定性 combine（`reduce_sum(dim=0)`） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 权重载入 | `w_v, w_h, cw[p] → f32` | `T.Parallel` 元素读 + `T.cast` | `[(0,j) in T.Parallel(d_padded)]: wv[0,j] = T.cast(rms_w_v[j], "float32")`；`[(p,j) in T.Parallel(KS,d_padded)]: cw[p,j] = T.cast(conv_w[p,j], "float32")` | Developer |
| 窗口定位/掩码 | `gc, rmk` | 标量 `T.max/T.min`（**unroll 语句**） | `for r in range(R): gt = t0-3+r; rmk[r,0] = T.cast(T.min(T.max(gt+1,0),1)*T.min(T.max(seq_len-gt,0),1), "float32")` | Developer |
| 载入 vhat 窗 | `vn[·]` | `T.Parallel(R,d_padded)` 元素读 + `T.cast` | `vn[r,j] = T.cast(vhat[bid, T.min(T.max(t0-3+r,0),seq_len-1), j], "float32")`（探针 P2/P6） | Developer |
| 标量广播 | `rrms_v_brc` / `w_v_brc` | `T.vbrc` | `(R,1)→(R,d_padded)`、`(1,d_padded)→(R,d_padded)`（`shape操作/T.vbrc.md`） | Developer |
| 逐元素乘/加 | `vn = vhat·rrms_v·w_v·rmk` | `T.vmul` | `T.vmul(vn, rv_brc, vn)`；`T.vmul(vn, wv_brc, vn)`；`T.vmul(vn, rmk_brc, vn)`（in-place，≥2 操作数） | Developer |
| 卷积/转置 | `Σ_p cw[p]⊙vn[i+p]` | `T.Parallel(BLK_S,d_padded)` + `+=` | host 展开 `for p in range(4)`；`conv[i,j] += cw[p,j]*vn[i+p,j]`（仿射偏移，探针 P6a） | Developer |
| 清零 | `c_o≡0` | `T.clear` | `T.clear(conv)`（`内存操作/T.clear.md`） | Developer |
| sigmoid | `σ(x)` | `T.vsigmoid` | `T.vsigmoid(conv, sig)`（`数学操作/T.sigmoid.md`） | Developer |
| 行归约 | `Σ_c ·` | `T.reduce_sum(dim=1)` | `T.reduce_sum(prod, red, dim=1, clear=True)`（`(R,d_padded)→(R,1)`，`规约操作/T.reduce_sum.md`） | Developer |
| token 归约 | `Σ_t ·` | `T.reduce_sum(dim=0)` | `T.reduce_sum(prod, red0, dim=0, clear=True)`（`(BLK_S,d_padded)→(1,d_padded)`；探针 P2 实测） | Developer |
| 上转（buffer→buffer） | `cast32` | `T.vcast` | `T.vcast(src_dt, dst_f32, round_mode="rint")`（`数据类型转换操作/T.vcast.md`；src/dst **等 shape**） | Developer |
| 下转 | `dt(·)` | `T.vcast` | `T.vcast(f32buf, dtbuf, round_mode="rint")` | Developer |
| 写回 | `dH[bid,t0:t0+BLK_S,:]=…` | `T.copy` | 先 `T.vcast` 到 dtype UB，再 `T.copy(ostg, dH[bid, t0:t0+BLK_S, 0:d_padded])`（**禁止** store 处 direct-cast） | Developer |
| 部分和写回 | `ws_h[cid,:]=…` | `T.copy` | `T.copy(red0, ws_h[cid:cid+1, 0:d_padded])`（探针 P2/P5） | Developer |
| combine 标量落盘 | `drms_w_h[j]=Σ_blk` | unroll fill + `T.copy` | `for j in range(d_padded): stg[j] = red[0,j]`；`T.copy(stg, drms_w_h)`（规避 `TRAP-npuir-parallel-scalar-store-race`；探针 P5） | Developer |
| 任务映射 | grid | `T.Kernel(M*num_tiles, is_npu=True)` | `bid = cid % M`、`tile = cid // M` | Developer |

### 3.3 计算伪代码

> 说明（依 design-template §3.3）：源 `T.Tensor` 签名已用**位置参数**传 dtype（`T.Tensor((M,seq_len,d_padded), dtype)`），NPU 实现保持该形式（不使用 `dtype=` 关键字）。伪代码为**主选方案（候选 A，3-kernel）**的完整结构；`BLK_S = 8 if d ≤ 256 else 4`（shape 分派，§5.2）。**工厂返回 Python 闭包调度 3 个独立 kernel**（§0.6 R1）。

```python
import functools
import math
import torch
import tilelang
import tilelang.language as T

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4
HALO = CONV_KERNEL_SIZE - 1          # 3
BLK_S = 8 if d <= 256 else 4         # 每 block 输出 token 数（shape 分派：d=512 降档保 UB，§5.2）

def _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype):
    accum = "float32"
    d_padded = ((d + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    R = BLK_S + HALO                                  # 11（d≤256）/ 7（d=512），窗口行数
    num_tiles = (seq_len + BLK_S - 1) // BLK_S
    NB = M * num_tiles
    INV_D = 1.0 / float(d)
    INV_SQRT_D = 1.0 / math.sqrt(float(d))

    # ---------------- K1: Pass 1 + dconv 部分和 ----------------
    @tilelang.jit(target="npuir")
    def _k1():
        @T.prim_func
        def k1(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            rrms_v: T.Tensor((M, seq_len), accum),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum),
            ws_conv: T.Tensor((CONV_KERNEL_SIZE * NB, d_padded), accum),
        ):
            with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
                bid = cid % M
                t0 = (cid // M) * BLK_S
                vnw  = T.alloc_shared((R, d_padded), accum)      # vhat*rrms_v*w_v 窗口
                dy   = T.alloc_shared((BLK_S, d_padded), accum)
                conv = T.alloc_shared((BLK_S, d_padded), accum)
                sig  = T.alloc_shared((BLK_S, d_padded), accum)
                ds   = T.alloc_shared((BLK_S, d_padded), accum)
                prod = T.alloc_shared((BLK_S, d_padded), accum)
                bro  = T.alloc_shared((R, d_padded), accum)
                rmk  = T.alloc_shared((R, 1), accum)
                rv   = T.alloc_shared((R, 1), accum)
                wv   = T.alloc_shared((1, d_padded), accum)
                cw   = T.alloc_shared((CONV_KERNEL_SIZE, d_padded), accum)
                red0 = T.alloc_shared((1, d_padded), accum)
                ostg = T.alloc_shared((BLK_S, d_padded), dtype)

                for j in T.Parallel(d_padded):
                    wv[0, j] = T.cast(rms_w_v[j], accum)
                for p, j in T.Parallel(CONV_KERNEL_SIZE, d_padded):
                    cw[p, j] = T.cast(conv_w[p, j], accum)

                # --- row mask + rrms_v window (unrolled scalar; probe P6a/P6b) ---
                for r in range(R):
                    gt = t0 - HALO + r
                    gc = T.min(T.max(gt, 0), seq_len - 1)
                    rmk[r, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                       * T.min(T.max(seq_len - gt, 0), 1), accum)
                    rv[r, 0] = rrms_v[bid, gc]

                # --- load vhat window + materialise vn (O3) ---
                for r, j in T.Parallel(R, d_padded):
                    gc = T.min(T.max(t0 - HALO + r, 0), seq_len - 1)
                    vnw[r, j] = T.cast(vhat[bid, gc, j], accum)
                T.vbrc(rv, bro);  T.vmul(vnw, bro, vnw)          # * rrms_v
                T.vbrc(wv, bro);  T.vmul(vnw, bro, vnw)          # * w_v
                T.vbrc(rmk, bro); T.vmul(vnw, bro, vnw)          # * row mask

                # --- causal DWConv (host-expanded taps; UB affine shift) ---
                T.clear(conv)
                for p in range(CONV_KERNEL_SIZE):
                    for i, j in T.Parallel(BLK_S, d_padded):
                        conv[i, j] += cw[p, j] * vnw[i + p, j]

                # --- SiLU bwd ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    dy[i, j] = T.cast(dY[bid, t0 + i, j], accum)
                T.vsigmoid(conv, sig)
                for i, j in T.Parallel(BLK_S, d_padded):
                    ds[i, j] = dy[i, j] * (sig[i, j] + conv[i, j] * sig[i, j] * (1.0 - sig[i, j]))

                # --- store d_silu (K1->K2 handoff) ---
                T.copy(ds, dvhat_buf[bid, t0:t0 + BLK_S, 0:d_padded])

                # --- dconv_w partial sum (token reduce dim=0; probe P2) ---
                for p in range(CONV_KERNEL_SIZE):
                    for i, j in T.Parallel(BLK_S, d_padded):
                        prod[i, j] = ds[i, j] * vnw[i + p, j]
                    T.reduce_sum(prod, red0, dim=0, clear=True)
                    T.copy(red0, ws_conv[p * NB + cid: p * NB + cid + 1, 0:d_padded])
        return k1

    # ---------------- K2: Pass 1b + Pass 2 + drms 部分和 ----------------
    @tilelang.jit(target="npuir")
    def _k2():
        @T.prim_func
        def k2(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            alpha: T.Tensor((M, seq_len), accum),
            rrms_h: T.Tensor((M, seq_len), accum),
            rrms_k: T.Tensor((M, seq_len), accum),
            rrms_v: T.Tensor((M, seq_len), accum),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum),
            dH: T.Tensor((M, seq_len, d_padded), dtype),
            dk: T.Tensor((M, seq_len, d_padded), dtype),
            dv: T.Tensor((M, seq_len, d_padded), dtype),
            ws_h: T.Tensor((NB, d_padded), accum),
            ws_v: T.Tensor((NB, d_padded), accum),
        ):
            with T.Kernel(M * num_tiles, is_npu=True) as (cid, _):
                bid = cid % M
                t0 = (cid // M) * BLK_S
                dsw  = T.alloc_shared((R, d_padded), accum)
                dvn  = T.alloc_shared((BLK_S, d_padded), accum)
                vhat_l = T.alloc_shared((BLK_S, d_padded), accum)
                h_l  = T.alloc_shared((BLK_S, d_padded), accum)
                k_l  = T.alloc_shared((BLK_S, d_padded), accum)
                v_l  = T.alloc_shared((BLK_S, d_padded), accum)
                hn   = T.alloc_shared((BLK_S, d_padded), accum)
                kn   = T.alloc_shared((BLK_S, d_padded), accum)
                dhn  = T.alloc_shared((BLK_S, d_padded), accum)
                dkn  = T.alloc_shared((BLK_S, d_padded), accum)
                dvh  = T.alloc_shared((BLK_S, d_padded), accum)
                prod = T.alloc_shared((BLK_S, d_padded), accum)
                bro  = T.alloc_shared((R, d_padded), accum)       # dsw mask broadcast
                bro_s = T.alloc_shared((BLK_S, d_padded), accum)   # own-row broadcast
                rmk  = T.alloc_shared((R, 1), accum)
                cw   = T.alloc_shared((CONV_KERNEL_SIZE, d_padded), accum)
                wv   = T.alloc_shared((1, d_padded), accum)
                wh   = T.alloc_shared((1, d_padded), accum)
                al   = T.alloc_shared((BLK_S, 1), accum)   # alpha
                rh   = T.alloc_shared((BLK_S, 1), accum)
                rk   = T.alloc_shared((BLK_S, 1), accum)
                rv   = T.alloc_shared((BLK_S, 1), accum)
                ddot_buf = T.alloc_shared((BLK_S, 1), accum)
                red  = T.alloc_shared((BLK_S, 1), accum)   # row reduce out
                red0 = T.alloc_shared((1, d_padded), accum)  # token reduce out
                ostg = T.alloc_shared((BLK_S, d_padded), dtype)

                for j in T.Parallel(d_padded):
                    wv[0, j] = T.cast(rms_w_v[j], accum)
                    wh[0, j] = T.cast(rms_w_h[j], accum)
                for p, j in T.Parallel(CONV_KERNEL_SIZE, d_padded):
                    cw[p, j] = T.cast(conv_w[p, j], accum)

                # --- d_silu window: right-edge mask (unrolled scalar) ---
                for r in range(R):
                    gt = t0 + r
                    gc = T.min(T.max(gt, 0), seq_len - 1)
                    rmk[r, 0] = T.cast(T.min(T.max(gt + 1, 0), 1)
                                       * T.min(T.max(seq_len - gt, 0), 1), accum)
                for r, j in T.Parallel(R, d_padded):
                    gc = T.min(T.max(t0 + r, 0), seq_len - 1)
                    dsw[r, j] = dvhat_buf[bid, gc, j]
                T.vbrc(rmk, bro); T.vmul(dsw, bro, dsw)

                # --- conv transpose (host-expanded taps) ---
                T.clear(dvn)
                for p in range(CONV_KERNEL_SIZE):
                    for i, j in T.Parallel(BLK_S, d_padded):
                        dvn[i, j] += cw[p, j] * dsw[i + (CONV_KERNEL_SIZE - 1) - p, j]

                # --- load own rows + scalars (unrolled scalar loads) ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    vhat_l[i, j] = T.cast(vhat[bid, t0 + i, j], accum)
                    h_l[i, j] = T.cast(H[bid, t0 + i, j], accum)
                    k_l[i, j] = T.cast(k[bid, t0 + i, j], accum)
                    v_l[i, j] = T.cast(v[bid, t0 + i, j], accum)
                for i in range(BLK_S):
                    al[i, 0] = alpha[bid, t0 + i]
                    rh[i, 0] = rrms_h[bid, t0 + i]
                    rk[i, 0] = rrms_k[bid, t0 + i]
                    rv[i, 0] = rrms_v[bid, t0 + i]

                # --- drms_w_v partial ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = dvn[i, j] * vhat_l[i, j]
                T.vbrc(rv, bro_s); T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red0, dim=0, clear=True)
                T.copy(red0, ws_v[cid:cid + 1, 0:d_padded])

                # --- RMSNorm(vhat) bwd -> d(vhat) (+ dY residual) ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = vhat_l[i, j] * dvn[i, j]
                T.vbrc(wv, bro_s); T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)          # dot_v (BLK_S,1)
                # dvhat = rrms_v*(w_v*dvn - rrms_v^2*vhat*dot_v*INV_D) + dY
                T.vbrc(wv, bro_s); T.vmul(dvn, bro_s, dvh)   # dvh = w_v*dvn
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = (rv[i, 0] * rv[i, 0]) * vhat_l[i, j] * red[i, 0] * INV_D
                for i, j in T.Parallel(BLK_S, d_padded):
                    dvh[i, j] = rv[i, 0] * (dvh[i, j] - prod[i, j]) + T.cast(dY[bid, t0 + i, j], accum)

                # --- dv = alpha * dvhat ; dalpha ; ddot ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = al[i, 0] * dvh[i, j]               # dv
                T.vcast(prod, ostg, round_mode="rint")
                T.copy(ostg, dv[bid, t0:t0 + BLK_S, 0:d_padded])
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = dvh[i, j] * v_l[i, j]
                T.reduce_sum(prod, red, dim=1, clear=True)              # dalpha (BLK_S,1)
                for i in range(BLK_S):                                  # ddot -> ddot_buf
                    ddot_buf[i, 0] = red[i, 0] * al[i, 0] * (1.0 - al[i, 0]) * INV_SQRT_D

                # --- h_norm / k_norm (CSE, O3) ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    hn[i, j] = h_l[i, j] * rh[i, 0]
                    kn[i, j] = k_l[i, j] * rk[i, 0]
                T.vbrc(wh, bro_s)
                T.vmul(hn, bro_s, hn); T.vmul(kn, bro_s, kn)   # * w_h

                # --- dhn = ddot*kn ; dkn = ddot*hn (O3 materialise) ---
                T.vbrc(ddot_buf, bro_s)
                T.vmul(kn, bro_s, dhn); T.vmul(hn, bro_s, dkn)

                # --- drms_w_h partial: Σ_t (dhn*H*rrms_h + dkn*k*rrms_k) ---
                #  MUST precede the in-place overwrite of dhn/dkn by dH/dk below
                #  (PL-1.23 read-before-overwrite discipline; source L282-285/L305-308).
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = (dhn[i, j] * h_l[i, j] * rh[i, 0]
                                  + dkn[i, j] * k_l[i, j] * rk[i, 0])
                T.reduce_sum(prod, red0, dim=0, clear=True)
                T.copy(red0, ws_h[cid:cid + 1, 0:d_padded])

                # --- dH = rrms_h*(w_h*dhn - rrms_h^2*H*doth*INV_D)  [dhn 就地覆写为 dH] ---
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = h_l[i, j] * kn[i, j]               # H * kn
                T.vbrc(wh, bro_s); T.vmul(prod, bro_s, prod)   # * w_h
                T.reduce_sum(prod, red, dim=1, clear=True)          # sum(H*w_h*kn)
                for i in range(BLK_S):
                    red[i, 0] = red[i, 0] * ddot_buf[i, 0]          # d0th = ddot * sum
                T.vbrc(wh, bro_s); T.vmul(dhn, bro_s, dhn)     # w_h*dhn
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = (rh[i, 0] * rh[i, 0]) * h_l[i, j] * red[i, 0] * INV_D
                for i, j in T.Parallel(BLK_S, d_padded):
                    dhn[i, j] = rh[i, 0] * (dhn[i, j] - prod[i, j])
                T.vcast(dhn, ostg, round_mode="rint")
                T.copy(ostg, dH[bid, t0:t0 + BLK_S, 0:d_padded])

                # --- dk（对称）---
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = k_l[i, j] * hn[i, j]               # k * hn
                T.vbrc(wh, bro_s); T.vmul(prod, bro_s, prod)
                T.reduce_sum(prod, red, dim=1, clear=True)          # sum(k*w_h*hn)
                for i in range(BLK_S):
                    red[i, 0] = red[i, 0] * ddot_buf[i, 0]
                T.vbrc(wh, bro_s); T.vmul(dkn, bro_s, dkn)     # w_h*dkn
                for i, j in T.Parallel(BLK_S, d_padded):
                    prod[i, j] = (rk[i, 0] * rk[i, 0]) * k_l[i, j] * red[i, 0] * INV_D
                for i, j in T.Parallel(BLK_S, d_padded):
                    dkn[i, j] = rk[i, 0] * (dkn[i, j] - prod[i, j])
                T.vcast(dkn, ostg, round_mode="rint")
                T.copy(ostg, dk[bid, t0:t0 + BLK_S, 0:d_padded])
        return k2

    # ---------------- K3: deterministic combine ----------------
    @tilelang.jit(target="npuir")
    def _k3():
        @T.prim_func
        def k3(
            ws_h: T.Tensor((NB, d_padded), accum),
            ws_v: T.Tensor((NB, d_padded), accum),
            ws_conv: T.Tensor((CONV_KERNEL_SIZE * NB, d_padded), accum),
            drms_w_h: T.Tensor((d_padded,), accum),
            drms_w_v: T.Tensor((d_padded,), accum),
            dconv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), accum),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                ws_t = T.alloc_shared((NB, d_padded), accum)
                red0 = T.alloc_shared((1, d_padded), accum)
                stg  = T.alloc_shared((d_padded,), accum)
                # drms_w_h
                T.copy(ws_h, ws_t)
                T.reduce_sum(ws_t, red0, dim=0, clear=True)
                for j in range(d_padded):
                    stg[j] = red0[0, j]
                T.copy(stg, drms_w_h)
                # drms_w_v
                T.copy(ws_v, ws_t)
                T.reduce_sum(ws_t, red0, dim=0, clear=True)
                for j in range(d_padded):
                    stg[j] = red0[0, j]
                T.copy(stg, drms_w_v)
                # dconv_w（逐 tap）
                for p in range(CONV_KERNEL_SIZE):
                    T.copy(ws_conv[p * NB:(p + 1) * NB, 0:d_padded], ws_t)
                    T.reduce_sum(ws_t, red0, dim=0, clear=True)
                    for j in range(d_padded):
                        stg[j] = red0[0, j]
                    T.copy(stg, dconv_w[p:p + 1, 0:d_padded])
        return k3

    # ---------------- factory closure: allocate + dispatch ----------------
    def _run(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha,
             rrms_h, rrms_k, rrms_v):
        dev = dY.device
        dt = torch.float16 if "float16" in dtype else torch.bfloat16

        def new(shape, t):
            return torch.empty(shape, dtype=t, device=dev)

        dH = new((M, seq_len, d_padded), dt)
        dk = new((M, seq_len, d_padded), dt)
        dv = new((M, seq_len, d_padded), dt)
        drms_w_h = new((d_padded,), torch.float32)
        drms_w_v = new((d_padded,), torch.float32)
        dconv_w = new((CONV_KERNEL_SIZE, d_padded), torch.float32)
        dvhat_buf = new((M, seq_len, d_padded), torch.float32)
        ws_h = new((NB, d_padded), torch.float32)
        ws_v = new((NB, d_padded), torch.float32)
        ws_conv = new((CONV_KERNEL_SIZE * NB, d_padded), torch.float32)

        _k1()(dY, vhat, rrms_v, rms_w_v, conv_w, dvhat_buf, ws_conv)
        _k2()(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha,
              rrms_h, rrms_k, rrms_v, dvhat_buf, dH, dk, dv, ws_h, ws_v)
        _k3()(ws_h, ws_v, ws_conv, drms_w_h, drms_w_v, dconv_w)
        return [dH, dk, dv, drms_w_h, drms_w_v, dconv_w]

    return _run
```

> **实现注意（Stage 3 输入，逐条与 §10 风险对应）**：
> 1. **多 `T.Kernel` 不支持**（探针 P1）——一律 Python 闭包调度独立 kernel；每个 kernel 单独 `@tilelang.jit(target="npuir")`。
> 2. **`T.atomic_add` 禁用**（探针 P4 SIGABRT）——权重梯度走 K1/K2 部分和 + K3 combine。
> 3. **`T.Parallel` 内运行时 `T.min/T.max` 标量 store 禁用**（探针 P6b 错值；`TRAP-npuir-parallel-runtime-minmax`）——行掩码与 `rrms_v` 窗口一律用**编译期 unroll 标量语句**。
> 4. **带 buffer 操作数的 `T.if_then_else` 禁用**（`TRAP-npuir-ifthenelse-parallel-segv`）——边界统一用索引 clamp + 算术行掩码。
> 5. **标量广播用 `T.vbrc`+`T.vmul`（≥2 向量操作数 / in-place）**（`TRAP-npuir-vec-scalar-broadcast`）；`T.vbrc` 源须为 `(R,1)`/`(1,d)` UB buffer，目标轴 >1（`TRAP-vbrc-empty-broadcast`）。**广播 scratch 专 buffer 专用**（`bro` (R,d) 供窗口链、`bro_s` (BLK_S,d) 供 own-rows 链），**不使用 UB buffer 的静态切片作 v-op dst**（dst 静态切片可能被静默忽略，`TRAP-ub-partial-region-semantics`）。
> 6. **`T.vcast` 要求 src/dst 等 shape**（`TRAP-npuir-vcast-shape-mismatch`）——下转前先 `T.copy` 到同形 dtype buffer；store 处禁 direct-cast（前向 A 首版被拒）。
> 7. **GM 写回一律走 `T.copy`**；`(1,d)→(d,)` 标量行输出用 unroll fill + 单次 `T.copy`（`TRAP-npuir-parallel-scalar-store-race`）。
> 8. **conv/转置 tap `p` 必须 factory 层展开**（`TRAP-C13`）。
> 9. **buffer 一律在 `T.Kernel` 顶层无条件分配**（`TRAP-conditional-alloc`）。
> 10. **`ddot` 存于独立 `ddot_buf (BLK_S,1)`**（伪代码已显式物化）：`ddot` 在步骤 13 由 `dalpha` 算出即存 `ddot_buf`，步骤 15（`dhn/dkn`）与 17/18（`doth/dotk` 的 `ddot` 因子）直接广播复用（避免重算；O2 因子化的物化点）。
> 11. `ws_conv` 采用扁平 `(KS·NB, d_padded)` 以保持 `T.copy` 双向 slice 为 2-D（区域语义最稳）。
> 12. **输出契约 = 6 输出**（`dH/dk/dv/drms_w_h/drms_w_v/dconv_w`）；脚手架 `register_fake`（现 8）须对齐到 6（§3.4）。
> 13. **读取先于覆写（别名纪律，`PL-1.23-vop-chain-inplace-alias`）**：K2 内 `dhn`/`dkn` **就地覆写**为 `dH`/`dk`（步骤 17/18）——任何读取 `dhn`/`dkn` **原值**的计算（步骤 16 `drms_w_h` 部分和）**必须先于覆写**。v0 曾把 `drms_w_h` 置于覆写之后（读成 `Σ(dH·H·rrms_h + dk·k·rrms_k)`），v1 已前移修正（§0.8 A）。K2 其余 buffer（`hn/kn/dvn/vhat_l/h_l/k_l/v_l`）不被覆写；单 op 原地形态（`T.vmul(vnw,bro,vnw)` / `T.vmul(dsw,bro,dsw)` 等）本身满足「读取先于覆写」。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|---------|
| `T.vcast(src,dst,round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（f16/bf16→f32 `rint`；f32→f16/bf16；**src/dst 等 shape**） | 文档 + 探针 P2/P5 |
| `T.vmul` / `T.vadd` / `T.vsub` | `docs/Tilelang.language/数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md`（fp16/fp32；bf16 ✗） | 文档 + 探针 P2/P6a + 先例 `examples/engram/engram_bwd.py` |
| `T.vsigmoid` | `docs/Tilelang.language/数学操作/T.sigmoid.md`（fp16/fp32） | 文档 + 先例 `examples/engram/engram_bwd.py` L70 |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md`（`(R,1)→(R,N)`，N>1；`(BM,1)→(BM,1)` 被拒） | 文档 + 探针 P6a |
| `T.reduce_sum(buf,out,dim,clear)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md`（同 rank，dst 归约维为 1；`dim` 在范围内；fp16/fp32） | 文档 + 探针 P2（dim=0）/P5 + 先例 `examples/engram/engram_bwd.py` L169（dim=1） |
| `T.copy` | `docs/Tilelang.language/内存操作/T.copy.md`；slice 形态正确（`TRAP-T-copy-region-semantics`：src/dst 双显式 slice） | 文档 + 探针 P2/P5 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md`（UB 承载）；先例（decode / engram_bwd） | 文档 + 探针 |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | 文档 + 先例 |
| `T.Kernel(..., is_npu=True)` | 先例 `examples/engram/engram_bwd.py` L41 | 文档 + 先例 |
| **多 `T.Kernel` in 单 prim_func** | 本任务探针 P1 | ❌ 实测失败（grid 求值 None）→ 多 kernel 闭包 |
| **`T.atomic_add`** | 本任务探针 P4（`docs/Tilelang.language/原子操作/T.atomic_add.md` 存在但设备实测不可用） | ❌ 实测 SIGABRT |
| **token 向 `T.reduce_sum(dim=0)`** | 本任务探针 P2 | ✅ 实测 maxdiff 0.0 |
| **grid=1 combine** | 本任务探针 P5 | ✅ 实测 maxdiff 4.8e-7 |
| **conv 转置仿射 shift + unroll 掩码** | 本任务探针 P6a | ✅ 实测 maxdiff 0.0 |
| **`T.Parallel` 内运行时 minmax 掩码** | 本任务探针 P6b | ❌ 实测错值（禁用） |

> **output 契约对齐（K7）**：源 `out_idx=[12..19]`（8 输出，含 2 内部 workspace）。本设计工厂闭包返回 **6 个契约输出**（workspace 内部 `torch.empty` 分配、不返回）。Stage 3 须把脚手架 wrapper 的 `register_fake`（现返回 8 张量，`examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd.py` L113–L129）对齐到 **6**，并保持 NPU 单级调用约定 `factory(params)(tensors)`（本设计工厂返回闭包，`type(...)` **非** `JitKernel_NPU`——`TRAP-outidx-call-convention` 的零成本判定在此**不适用**，须显式核对闭包签名与 wrapper 调用点一致）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **是**（源为 2D grid `(seq_len,M)`） | 一维 `T.Kernel(M*num_tiles)`，`bid=cid%M`、`tile=cid//M`（§5.5） |
| GPU 专用 API（warp shuffle / cp.async / SMEM swizzle） | 否（源只用 `T.Parallel`/`reduce_sum`/`T.serial`/`atomic_add`/`if_then_else`） | `atomic_add`/带 buffer 的 `if_then_else` 已在 npuir 实测不可用 → 重设计（§0.6 R4/R5） |
| 单 prim_func 多 `T.Kernel` | **是**（源 5 块） | 探针 P1 实测不支持 → 多 kernel 闭包（§0.6 R1） |
| GEMM 非整除 / 分形限制 | 否（无 `T.gemm`） | 不涉及 |
| L0C 溢出 | 否（无 Cube） | 不涉及 |
| UB 192KB 容量 | 是 | §4.5 预算 + ×1.7 膨胀核对 + `BLK_S` 回退 |
| bf16 v-op 缺失 | 是（bf16 workload） | 全链 fp32 中转（§0.6 R2） |
| fp32 GM 输入路径（`TRAP-npuir-fp32-vector-reduce`） | 否 | manifest 输入 fp16/bf16；fp32 仅片上 UB 中转 |

#### 3.5.2 参考实现差异说明

> 迁移任务完整差异分析见 §0.5/§0.6；此处只汇总影响 API 选型的关键技术差异。

| 差异项 | 参考实现（GPU `engram_bwd.py`） | 本项目（Ascend npuir） | 转换方案 |
|--------|--------------------------------|----------------------|----------|
| Kernel 维度 | `T.Kernel(seq_len, M, threads=threads)`（2D）×3 + `T.Kernel(1)` + `T.Kernel(4)` | 一维 `T.Kernel(..., is_npu=True)` | 展平为 `M*num_tiles`（§5.5） |
| 阶段组织 | 单 prim_func 五 `T.Kernel` | 多 `T.Kernel` 不支持 | 3-kernel 闭包（§0.6 R1） |
| 线程/元素并行 | `threads` + `T.Parallel` | `threads` 无效果；v-prefix 整 buffer | 移除 `threads`；保留 `T.Parallel` 作向量化载体 |
| 内存层级 | fragment/register | UB（`T.alloc_shared`） | `T.alloc_shared` 承载全部中间量 |
| 权重梯度归约 | `T.atomic_add` / 独立串行 kernel | `atomic_add` 崩溃 | 确定性部分和 + combine（§0.6 R5） |
| 边界 | `T.if_then_else(src_t≥0,…)` | 索引 clamp + 算术行掩码 | §0.6 R4（探针 P6a/P6b） |
| 跨阶段交接 | `d_silu`/`d(vhat)` 经 GM 两次往返 | `d_silu` 一次（Pass1b+Pass2 融合） | §0.6 R1 |
| bf16 | 原生 | 无 v-op bf16 → fp32 中转 | §0.6 R2 |
| 确定性 | atomic 序不定 | 测试要求双跑 bit 一致 | 确定性 combine（§0.6 R5） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/engram/engram_bwd.py` | **极高度**（同算子 NPU 版） | 3 个独立 kernel、`T.Kernel(seq_len*M, is_npu=True)`、`T.alloc_shared`、`T.reduce_sum(dim=1)`、`T.vsigmoid`、conv/转置 tap + 行读、`T.copy` 运行时行索引 |
| `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py` | **高度**（同族正向迁移） | Developer 纯 Vector 序列分块、UB 内因果滑窗、算术行掩码、`vbrc`+`vmul`、`vcast`+`T.copy` 写回、`(R,1)` own-rows unroll fill + `T.copy` |
| `examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py` | 中（同族融合迁移） | Developer 纯 AIV 融合、buffer 顶层分配、GM 行 `T.copy` 运行时索引 |
| `examples/norm/rms_norm.py` | 中 | RMSNorm 的 `reduce_sum` + v-prefix |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/` | 中（norm 族） | 行归约迁移基准形态、`alloc_shared` 向量规约 |

---
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `dY` / `H` / `k` / `v` / `vhat` | `(M, seq_len, d_padded)` | `float16 \| bfloat16` | host 侧 pad；`d_padded==d` |
| `rms_w_h` / `rms_w_v` | `(d_padded,)` | same | RMSNorm 权重 |
| `conv_w` | `(4, d_padded)` | same | 深度卷积权重 |
| `alpha` / `rrms_h` / `rrms_k` / `rrms_v` | `(M, seq_len)` | `float32` | 前向保存标量 |

### 4.2 输出张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `dH` / `dk` / `dv` | `(M, seq_len, d_padded)` | `same_as(dY)` | host crop 到 d |
| `drms_w_h` / `drms_w_v` | `(d_padded,)` | `float32` | 跨 token 归约；host crop 到 d |
| `dconv_w` | `(4, d_padded)` | `float32` | 跨 token 归约；host crop 到 d |

**对外契约 shape**：`dH/dk/dv:(M,seq_len,d)`、`drms_w_h/drms_w_v:(d,)`、`dconv_w:(4,d)`（与 `source_output_shape` 一致；host crop `d_padded→d`）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 层级 | 用途 | Kernel |
|-----------|-------|-------|------|------|--------|
| `dvhat_buf` | `(M, seq_len, d_padded)` | f32 | **GM workspace** | `d_silu`（K1→K2 交接） | K1 写 / K2 读 |
| `ws_conv` | `(4·NB, d_padded)` | f32 | GM workspace | `dconv_w` 每-block 部分和 | K1 写 / K3 读 |
| `ws_h` / `ws_v` | `(NB, d_padded)` | f32 | GM workspace | `drms_w_h`/`drms_w_v` 每-block 部分和 | K2 写 / K3 读 |
| `vnw` / `dsw` | `(R, d_padded)` | f32 | UB | `vn` / `d_silu` 窗口 | K1 / K2 |
| `dvn` / `dvh` / `hn` / `kn` / `dhn` / `dkn` | `(BLK_S, d_padded)` | f32 | UB | 转置 / d(vhat) / h_norm / k_norm / d(h_norm) / d(k_norm) | K2 |
| `h_l` / `k_l` / `v_l` / `vhat_l` | `(BLK_S, d_padded)` | f32 | UB | own rows | K2 |
| `conv` / `sig` / `ds` / `dy` / `prod` | `(BLK_S, d_padded)` | f32 | UB | 卷积累加 / sigmoid / d_silu / dY / scratch | K1 / K2 |
| `bro` | `(R, d_padded)` | f32 | UB | 窗口链行标量广播 scratch（K1 `vnw` 链 / K2 `dsw` 掩码链） | K1 / K2 |
| `bro_s` | `(BLK_S, d_padded)` | f32 | UB | own-rows 链标量广播 scratch（`w_v/w_h/ddot/α` 广播） | K1 / K2 |
| `cw` / `wv` / `wh` | `(4,d_padded)` / `(1,d_padded)` | f32 | UB | 权重 tile | K1 / K2 |
| `rmk` / `rv` / `rh` / `rk` / `al` / `red` / `ddot_buf` | `(R,1)` / `(BLK_S,1)` | f32 | UB | 掩码 / 标量列 / 行归约输出 / ddot | K1 / K2 |
| `red0` | `(1, d_padded)` | f32 | UB | token 归约输出 | K1 / K2 / K3 |
| `ostg` | `(BLK_S, d_padded)` | dtype | UB | `dH/dk/dv` 输出暂存（vcast 目标） | K2 |
| `ws_t` / `stg` | `(NB,d_padded)` / `(d_padded,)` | f32 | UB | combine 载入 / 标量落盘 | K3 |

### 4.4 内存搬运路径

```
K1（纯 Vector，GM → UB → GM）:
GM[dY,vhat,rrms_v] --T.Parallel(clamped 运行时行索引)--> UB[vnw(R,d) f32]
   --[vn 物化: vbrc(v_rrms/w_v/rmk) + vmul]--> UB[vnw]
   --[因果 DWConv: vnw[i+p] 仿射偏移 ⊙ cw[p] 累加]--> UB[conv (BLK_S,d)]
   --[vsigmoid + silu' ]--> UB[ds] --T.copy--> GM[dvhat_buf]
                            --[reduce_sum(dim=0)]--> GM[ws_conv]
K2（纯 Vector，GM → UB → GM）:
GM[dvhat_buf] --(前向 halo, clamp+右掩码)--> UB[dsw(R,d)]
   --[conv 转置: dsw[i+3-p]⊙cw[p] FMA]--> UB[dvn]
   --[RMSNorm(vhat) bwd + gate bwd + RMSNorm(H)/RMSNorm(k) bwd]--> UB[dH/dk/dv f32]
   --[vcast(rint) + T.copy]--> GM[dH,dk,dv] ;  --[reduce_sum(dim=0)]--> GM[ws_h,ws_v]
K3（grid=1）:
GM[ws_h,ws_v,ws_conv] --T.copy--> UB[(NB,d) f32] --[reduce_sum(dim=0) + unroll fill]--> GM[drms_w_h,drms_w_v,dconv_w]
```

### 4.5 UB 内存预算

> 口径：按主选候选 A、**shape 分派 `BLK_S`**（`d ≤ 256 → BLK_S=8, R=11`；`d=512 → BLK_S=4, R=7`，见 §5.2）。**未做激进 buffer 复用**，保守计。

| Buffer 组 | d=256（BLK_S=8, R=11） | d=512（BLK_S=4, R=7） |
|-----------|------------------------|------------------------|
| **K1** `(R,d)` f32 ×2（`vnw`+`bro`） + `(BLK_S,d)` f32 ×5（`dy,conv,sig,ds,prod`） | 2×11×256×4 + 5×8×256×4 ≈ 22.5+40 KB ≈ **63 KB** | 2×7×512×4 + 5×4×512×4 ≈ 28.7+40 KB ≈ **69 KB** |
| K1 权重/标量/暂存（`cw(4,d)`+`wv/rmk/rv/red0` 小 + `ostg(BLK_S,d)dt`） | ≈ 4+1+4 KB ≈ **9 KB** | ≈ 8+2+4 KB ≈ **14 KB** |
| **K1 合计（raw）** | ≈ **72 KB** | ≈ **83 KB** |
| **K2** `(R,d)` f32 ×2（`dsw`+`bro`） + `(BLK_S,d)` f32 ×11（`dvn,vhat_l,h_l,k_l,v_l,hn,kn,dhn,dkn,dvh,prod`） + `bro_s(BLK_S,d)` f32 ×1 | 22.5 + 11×8×256×4(=90) + 8 KB ≈ **120 KB** | 28.7 + 11×4×512×4(=90) + 8 KB ≈ **127 KB** |
| K2 权重/标量/暂存（`cw`+`wv/wh`+`al/rh/rk/rv/red/ddot_buf`+`ostg`） | ≈ 4+2+4+4 KB ≈ **14 KB** | ≈ 8+4+4+4 KB ≈ **20 KB** |
| **K2 合计（raw）** | ≈ **134 KB** | ≈ **147 KB** |
| **K3** `ws_t(NB,d)` f32 + `stg` | NB=16 → 16 KB | NB=32 → 64 KB |

> **风险与处置**：K2 raw 上界（并列、未复用）≈134KB（d=256）/ ≈147KB（d=512）；实际生命周期重叠可复用（`prod`↔`dvh`、`hn/kn` 在步骤 15 后释放给 `dhn/dkn`、`h_l/k_l/v_l` 在最后使用后释放、`dsw` 在步骤 9 后释放、`ostg` 顺序复用）。**复用后预估** ≈ `6×(BLK_S,d)+1×(R,d)`：d=256 ≈ 60KB raw、d=512 ≈ 64KB raw → ×1.7 ≈ **103KB / 108KB < 192KB ✓**。**强制动作（保留）**：① Stage 3 上靶前 `tilelang.compile(k1/k2, target="npuir")` 读 `ub overflow, requires N bits` 取精确实测需求（`CG-2026-0008`：手工预算 ≠ 可编译证明；`TRAP-UB-multibuffer-inflation`）；② 若实测超手工预算 ×1.12，按 `BLK_S` 再降（8→4 / 4→2）+ 合并 `h_l/k_l/v_l` 生命周期。**注**：`CONST-capacity-910B2C`（UB 192KB）**绑 910B2C/9362、本设备 910B3 未重验**。K1/K3 各档在限内（K3 NB=32 时 `ws_t` 64KB < 192KB）。

**L1 / L0C**：无 Cube 段，不涉及。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | 全参数编译期常量（`lru_cache` 按 `(M,seq_len,d,eps,dtype)` 特化） |

### 4.7 JIT 配置

```python
# 3 个独立 kernel，各自 target="npuir"；输出由工厂闭包 torch.empty 分配（不用 out_idx）
@tilelang.jit(target="npuir")
def _k1(): ...
@tilelang.jit(target="npuir")
def _k2(): ...
@tilelang.jit(target="npuir")
def _k3(): ...
# 工厂 `_engram_gate_conv_bwd_kernel(M,seq_len,d,eps,dtype)` 返回 `_run` 闭包（12 张量入参 → 6 张量返回）
```

> **wrapper 契约**：脚手架 wrapper `engram_gate_conv_bwd.py` 的 `EngramGateConvBwdKernel.forward` 调用 `_engram_gate_conv_bwd_kernel(M,seq_len,d,eps,dtype_str)(dY,H,k,...,rrms_v)`（无 `threads`，K9），拿到 6 个契约输出并 crop；**须把 `register_fake` 由 8 张量对齐到 6**（§3.4）。工厂签名保持 `(M,seq_len,d,eps,dtype)`（与 GPU 源一致）。**注**：`BLK_S` 默认按 `d` 分派（`8`/`4`，§5.2），如需调优可将其作为工厂新增可选 kwarg（不影响 wrapper 调用）。**workspace（`dvhat_buf`/`ws_*`）由闭包内 `torch.empty` 分配**；若 Stage 3 遇自动输出分配（`out_idx`）与 workspace 组合风险，可改为显式输出参数传入（`TRAP-zero-input-crash` 经验）。

---
## 5. Tiling 策略

### 5.1 计算类型

**类型**: **纯 Vector**（逐元素链 + 行/列归约 + 卷积转置 + 跨 token 权重梯度归约；无矩阵乘）。

**判定依据**: 源算子无 `T.gemm`/matmul（投影反向在外部 GEMM 完成）；6 处规约为行向/ token 向归约（`dot_v/h/k`、`drms_w_h/v`、`dconv_w`），非 MAC 密集段 → 落 Vector（UB）。**非 MixCV**（无 Cube 段）→ 无 CV 同步（§1.6.3 MIX 评估）。

### 5.2 Block 划分

```python
BLK_S   = 8 if d <= 256 else 4                 # 每 block 输出 token 数（shape 分派：d=512 降档保 UB，§4.5）
HALO    = 3                                    # 因果历史行 = 卷积核长 − 1
R       = BLK_S + HALO                         # 窗口行数：d≤256 → 11；d=512 → 7
num_tiles = ceil(seq_len / BLK_S)              # M 轴=batch，seq_len 轴分块
grid    = M * num_tiles                        # 一维 T.Kernel（K1/K2）
# d 不划分：(R, d_padded)/(BLK_S,d_padded) 的向量链与归约整块驻留 UB，无需 d 方向 tiling（避免跨核归约）
# K3 combine: grid = 1（列归约 NB ≤ 32 个 block 的部分和）
```

**block 选择理由**：
1. **seq_len 轴分块（非 d）**：卷积（左窗口 `t-3..t`）与转置（右窗口 `t..t+3`）沿 `seq_len` 有依赖，故 `seq_len` 作分块轴（每 block 覆盖 `BLK_S` 连续 token + 3-halo）；`d` 是全算子唯一的连续/规约轴，保持整块。
2. **`BLK_S` 按 d 档分派**：`d ≤ 256 → BLK_S=8`（`R=11`，halo 冗余 `3/11 ≈ 27%`，K2 复用后 ≈103KB）；`d=512 → BLK_S=4`（`R=7`，halo 冗余 `3/7 ≈ 43%`，但 K2 复用后 ≈108KB < 192KB，规避 v0 全局 `BLK_S=8` 在 `d=512` 的 ≈255KB 溢出）。halo 冗余随 `BLK_S` 减小而增大，故 `d=512` 仅在 UB 约束下取 4；`BLK_S` 仍为 Stage-4 调优杠杆。该取值与同族正向迁移实测一致（`engram_gate_conv_fwd` Stage 3：d=512 用 4、d=256 用 8）。
3. `d` 不划分：`d_padded ≤ 512`，全链整块驻留 UB。
4. `R = BLK_S + HALO`：K1 的 `conv[i]` 需 `vn[i..i+3]`、K2 的 `dvn[i]` 需 `dsw[i..i+3]`（窗口行 `0..R-1`）。
5. **权重梯度 cross-token 归约**：block 内 `reduce_sum(dim=0)` 部分和（不切 d，避免部分 d 归约）+ K3 combine（`NB ≤ 32`）。

### 5.3 约束分析

- **对齐约束**：`(R, d_padded)` f32 尾轴 `512×4=2048B`（32B 整数倍 ✓）；向量宽 fp32 ×4（512 整除）、fp16 ×8（256/512 整除）→ 尾 lane 浪费 0%。
- **UB 容量**：K1 复用后 ≈ 72KB（d=256）/ ≈ 83KB（d=512）；K2 复用后 ≈ 103KB（d=256, `BLK_S=8`）/ ≈ 108KB（d=512, `BLK_S=4`）——**两档均 < 192KB**（仍保留 Stage 3 编译探针，见 §4.5）。
- **L1 / L0C**：无 Cube，不涉及。
- **非整除**：manifest `d∈{256,512}` 均 256 对齐（`d_padded==d`）；`seq_len∈{16,32,64,128}` 均整除 `BLK_S`（8 或 4）→ **无尾块**。通用处理见 §5.4。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：
- `d`：manifest 各 workload `d∈{256,512}` 均 256 对齐 → `d_padded==d`，**无尾块**；保留 `d_padded=_align_up(d,256)` 以支持通用 d（host pad + 输出 crop，wrapper 已实现）。
- `seq_len`：manifest 各 workload 均整除 `BLK_S`（8 或 4）→ **无尾块**。为通用性，末 tile 若 `t0+BLK_S > seq_len`，取 `tail = T.min(BLK_S, seq_len - t0)`，输出 `T.copy(ostg[0:tail,:], dH[bid, t0:t0+tail, :])` 双侧裁剪；输入载入已由索引 clamp（`gc ∈ [0,seq_len)`）保证不越界，无效 halo 行由 `rmk` 掩码清零。**推荐通用形态 = src/dst 双显式 slice**（`TRAP-T-copy-region-semantics` 规则②）。
- **不做 host padding 改内容**（`ascend-constraints.md` §4）；非整除一律 kernel 内 `tail` 裁剪 + host 元数据 pad。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：K1/K2 输出网格 = `M`（batch）× `num_tiles`（seq 分块），`d` 轴不划分；K3 = 1。

```
num_logical_kernels(K1/K2) = ceil(M/1) × ceil(seq_len / BLK_S) = M × ceil(seq_len / BLK_S)   # BLK_S=8 (d≤256) 或 4 (d=512)
num_logical_kernels(K3)    = 1
```

| workload | M | seq_len | d | BLK_S | num_tiles | K1/K2 逻辑核数 | K3 |
|----------|---|---------|---|-------|-----------|----------------|-----|
| bwd-smoke-b1-s16-d256-fp16 | 1 | 16 | 256 | 8 | 2 | **2** | 1 |
| bwd-b1-s32-d256-fp16 | 1 | 32 | 256 | 8 | 4 | **4** | 1 |
| bwd-b2-s64-d512-fp16 | 2 | 64 | 512 | **4** | 16 | **32** | 1 |
| bwd-b1-s128-d256-bf16 | 1 | 128 | 256 | 8 | 16 | **16** | 1 |

**② 物理核数依据（实查）**：目标设备 `Ascend910B3`，实查代码与返回值（本会话 2026-10-15）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
NPUUtils.get().get_arch()              # 实际返回 "Ascend910B3"
```

→ **本算子为纯 Vector 算子，物理核数取 Vector Core = 40**（`core-split-strategy.md` §1 要素②：「纯 Vector 算子核数翻倍，即 `get_aicore_num() * 2`」）。**注**：`pattern-library/constants.md` 的 `CONST-aicore-910B2C`=24 / `CONST-aicore-910_9362`=20 均非本设备，**以实查为准**。

**③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数 → 无需适配**。

- 最大逻辑核数 = **32**（W3，`BLK_S=4`）≤ 40；W4 = 16、W2 = 4、W1 smoke = 2；K3 = 1。
- **结论：无需适配**（标准 §1 要素③第一分支）。grid = `M·num_tiles`（K1/K2）+ 1（K3）直接下发，**无核内 `T.serial` 串行、无多波次串行调度、无残留负载不均**（核数 ≤ 物理核数，每核至多 1 个逻辑块）。
- **利用率说明（显式记录）**：32/40 = 80% 峰值占用（W4 16/40 = 40%、W1 仅 2/40）——这是本算子数据量小（`M·seq_len ≤ 256` token）的固有特征（低并行度、延迟导向）。**更高并行度候选**（token-per-block `BLK_S=1`，grid 最大 128）已在 §1.6.3 候选 F 评估：4× 冗余计算 + 需 GM/UB 移位重读，**列为 Stage 4 结构级备选**（不改变本设计分核方案）；`d=512` 档取 `BLK_S=4` 已使 W3 逻辑核数由 16 升至 32。
- **模式先例**：源算子 grid=(seq_len, M)（2D）；NPU 侧一维 `T.Kernel(seq_len*M, is_npu=True)`（`examples/engram/engram_bwd.py` L41）。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级任务映射、conv/转置的编译期 tap 展开与 K3 的 block 轴归约；K1/K2 内**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务） | block 级并行 | `T.Kernel(M*num_tiles, is_npu=True)`（K1/K2）；`T.Kernel(1, is_npu=True)`（K3） | 每核处理一个 (batch, seq-tile) 的窗口全流程（§5.5：逻辑核 ≤ 40，无需适配） |
| 元素级（窗口行/通道） | 向量化 | `T.Parallel(R, d_padded)` / `T.Parallel(BLK_S, d_padded)` + v-prefix | 整 buffer 一次完成，无标量逐元素循环 |
| 行归约（`dot_v/h/k`） | 向量化 | `T.reduce_sum(..., dim=1)` | `(BLK_S,d_padded)→(BLK_S,1)` 一次水平归约 |
| token 归约（`drms/dconv`） | 向量化 | `T.reduce_sum(..., dim=0)` | `(BLK_S,d_padded)→(1,d_padded)` 一次列归约（探针 P2） |
| conv/转置 tap | **编译期展开**（factory 层 Python `for p in range(4)`） | host-level unroll | tap 偏移须为编译期常量（`TRAP-C13`）；4 条语句 |
| 行掩码/标量列 | **编译期展开**（`for r in range(R)`） | host-level unroll | `T.Parallel` 内运行时 minmax 标量 store 错值（探针 P6b） |
| K3 block 轴归约 | 向量化 | `T.reduce_sum(..., dim=0)` + unroll fill | `(NB,d_padded)→(1,d_padded)`；`NB ≤ 32`（探针 P5） |

**T.Persistent 选择说明**：不使用 `T.Persistent`。grid = `M·num_tiles` ≤ 40 已 ≤ 物理核数，无多波次；且本算子无 Cube（`TRAP-DEVMODE-PERSIST-GEMM` 不适用，但保持结构简洁）。

### 6.2 循环伪代码

见 §3.3（K1/K2 `with T.Kernel(M*num_tiles, is_npu=True)` + 线性 v-prefix 向量链 + factory 层 tap/掩码展开；K3 `with T.Kernel(1, is_npu=True)`；无 `T.Pipelined`、无核内逐元素串行循环）。

### 6.3 流水线优化

**当前设计**：不做显式流水线。原因：① 本算子纯向量链，片内重叠由 Developer 模式 **auto-multi-buffer** 处理（GM→UB 载入可与上一 tile 的向量计算重叠）；② block 间无数据依赖（halo 由 block 内重算覆盖），无需跨 tile 流水；③ 数据量小（≤2.7MB），发射项主导（§1.6.0 D-2），流水收益有限。**Stage 4 杠杆**：`BLK_S` 调整（流量/发射权衡）、K1/K2 融合度（3-kernel vs 2-kernel，§1.6.3 裁决）、载入合并、auto-multi-buffer 显式控制。

### 6.4 尾块处理

见 §5.4。manifest 全部 workload 无尾块（`d%256==0`、`seq_len%8==0`）；保留 `tail=T.min(BLK_S, seq_len-t0)` 双显式 slice + 索引 clamp + 行掩码以支持通用 shape。

---
## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式，各 kernel 内）；**kernel 之间为 host 侧顺序启动**（K1 → K2 → K3，同一 stream 顺序执行，数据依赖由 stream 序保证）。

### 7.2 同步点说明

Developer 模式下编译器自动插入 kernel 内同步。核内数据依赖链（全在 UB，无跨核/跨引擎）：

```
GM[dY,vhat,rrms_v] --T.Parallel 载入--> UB[vnw]  (MTE2→V 自动等待)
   --[向量链 + reduce_sum: V→V barrier 自动]--> UB[conv,ds]
   --[T.copy 落 GM]--> GM[dvhat_buf,ws_conv]  (V→MTE3 自动等待)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| GM→UB 载入 → 向量链 | 自动 | 编译器插 MTE2→V 等待 |
| 向量链内同 buffer（in-place `vmul`）/ reduce | 自动 | V→V barrier |
| UB→GM 写回 | 自动 | 编译器插 V→MTE3 等待 |
| tile 间 auto-multi-buffer | 自动 | 跨 tile 载入重叠 |
| **K1→K2→K3 跨 kernel** | **host 侧顺序启动（同 stream）** | 数据依赖（`dvhat_buf`/`ws_*`）由 kernel 启动序保证；无手动 flag |

无手动 `T.sync_block_set/wait`（无跨核 workspace、无 Cube 段、无多 `T.Kernel` in 单 prim_func）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs（auto-multi-buffer / 自动同步开启）
```

### 7.4 CV 融合设计

**本算子非 MixCV**（纯 Vector，无 `T.gemm`；AGENTS.md MixCV 触发规则不满足）。无 Cube/Vector 跨引擎同步、无 L0C→UB 直连、无跨核 workspace；3 个 kernel 为纯顺序 host 启动。**若未来融合外部投影 GEMM 反向则成为 MixCV，须先做 `CG-2026-0022` 最小 MIX 探针**（§1.6.3）。

---
## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现——**直接移植源仓测试基准** `examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py::ref_engram_gate_conv_bwd`（L29–L74，autograd 反向 + `F.conv1d` 左零填充），**不得复刻 §0.6/§1.6.3 的 NPU 结构**（保证验证独立性）。

```python
import torch
import torch.nn.functional as F

CONV_KERNEL_SIZE = 4


def golden_engram_gate_conv_bwd(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                                vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6):
    """Engram GateConv 反向参考实现（源语义；移植 test_engram_gate_conv_bwd.py::ref_engram_gate_conv_bwd）。

    契约：全程 fp32 计算（float().detach()），末端 dH/dk/dv `.to(H.dtype)` 单次舍回，
    drms_w_h/drms_w_v/dconv_w 保持 fp32；与 §0.6/§1.6.3 的 NPU 结构无关——独立验证。
    注：本 golden 由 autograd 反向实现（dY 经 Y_ag.backward），与源测试基准一致；
    `vhat/alpha/rrms_*` 为输入（前向保存量），不使用其复算。
    """
    M, T, d = H.shape
    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    def _rmsnorm(x, w):
        return x * (x ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w

    h_norm = _rmsnorm(H_ag, w_h_ag)
    k_norm = _rmsnorm(k_ag, w_h_ag)
    alpha_ag = torch.sigmoid((h_norm * k_norm).sum(-1, keepdim=True) / (d ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm = _rmsnorm(v_hat_ag, w_v_ag)
    v_perm = v_hat_norm.permute(0, 2, 1)                       # (M, d, T)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))        # 左零填充（因果）
    conv_out = F.conv1d(v_padded, cw_ag.T.unsqueeze(1), groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag
    Y_ag.backward(dY.float())
    return (H_ag.grad.to(H.dtype), k_ag.grad.to(H.dtype), v_ag.grad.to(H.dtype),
            w_h_ag.grad, w_v_ag.grad, cw_ag.grad)
```

> 备注：golden **独立于** NPU 流水线（autograd 反向 + PyTorch `mean`/`rsqrt`/`sigmoid`/`silu`/`conv1d`，不复用 §3.3 的 op 序列），满足验证独立性。**注（golden 与源内核的 ≤1ulp 差异，容差内）**：golden 由 autograd 从 fp32 `v_hat_ag`（未舍回）反向，源内核 Pass1b 重算 `vn` 由**输入 dtype 的 `vhat`**（经 cast）得到；两处差异均 ≤1 ulp；`rtol=2e-1` 门禁下由 `verify_equiv` FULL 行（fp32 pipeline vs fp64 golden，max_abs ≤6.1e-5）与 §8.3 L0 覆盖。
> **输入生成**参考 `examples/TileOPs/tileops/workloads/sequence_modeling.py`（`EngramGateConvBwdWorkload.gen_inputs`：`H=randn`、`k/v/dY=randn×0.1`、`rms_w_*=ones`、`conv_w=randn×0.02`、`alpha/rrms_*` 由参考前向导出）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float16 | 2e-1 | 2e-1 | 来源 `examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py` L95–L96 |
| bfloat16 | 3e-1 | 2e-1 | 来源 同上（bf16 精度低，容忍度放宽） |

**manifest roofline（供对账）**：`flops = (M·seq_len·(24d) + 20·M·seq_len)·2.5`；`bytes = (5·M·seq_len·d)·elem + 6·d·elem + 4·M·seq_len·4 + (3·M·seq_len·d)·elem + 10·d·4 + M·seq_len·d·4`。

**误差预算核对**：`verify_equiv.py` 实测——O1（常数折叠）逐位一致（d 为 2 的幂）/ `dot·fl(1/√d)` ≤2.4e-7；O3（CSE）逐位一致；O2（标量因子化）≤1.9e-6；**全链路三 workload（M1S32 fp16 / M2S64 fp16 / M1S128 bf16）`dH/dk/dv` 与 fp64 golden 均 max_abs ≤6.1e-5、`drms_w_h/drms_w_v/dconv_w`（fp32 域）max_abs ≤1.7e-7，违反数全 0**（32/32 `EQUIV_PASS`）。相对 `atol=2e-1`/`3e-1` 门禁余量充裕。**注意 `drms_w_h/drms_w_v/dconv_w` 是跨 token 归约输出，其 fp32 误差随 token 数累积**——L0-8 用 `≤1e-3`（归约输出更严于元素输出但仍有裕度）核对，Stage 3 门禁以 §8.1 golden 为准。Stage 3 门禁以 §8.1 golden 为准。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式覆盖：3 个 manifest workload + smoke、fp16/bf16、卷积左边界（t=0/1/2）、转置右边界（t=T-1..T-3）、非整除 seq_len、门控角点、确定性双跑、权重梯度归约。**

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 smoke fp16 | `(M=1, seq_len=16, d=256)` fp16 | 与 §8.1 golden 比对 6 输出 ≤ `atol=2e-1, rtol=2e-1` | 门槛通过 |
| L0-2 full W2 fp16 | `(1, 32, 256)` fp16 | 同上 | 门槛通过 |
| L0-3 full W3 fp16 | `(2, 64, 512)` fp16 | 同上 | 门槛通过 |
| L0-4 full W4 bf16 | `(1, 128, 256)` bf16 | 同上（`atol=3e-1`） | 门槛通过 |
| L0-5 卷积左边界 | `seq_len=4`（< BLK_S，单 tile），已知 `vn` 序列 | `dconv_w`/`ds` 的卷积只含有效 tap（`t-3+p<0` 贡献 0）与 golden 一致 | 门槛通过 |
| L0-6 转置右边界 | `seq_len=13`（`13%8≠0`，末 tile 尾块 + 右边界） | `dvn` 的 `t+3-p≥T` 贡献 0、输出形状 `(M,13,d)` 正确、无越界写 | 门槛通过 |
| L0-7 门控角点 | 构造 `⟨h_n,k_n⟩` 大正/大负 → `alpha→1/0` | `dv/dH/dk` 无 NaN/Inf；与 golden 一致 | 门槛通过 |
| L0-8 权重梯度归约 | 任意 full workload | `drms_w_h/drms_w_v/dconv_w`（fp32）与 golden `≤1e-3`（归约输出） | 门槛通过 |
| L0-9 **确定性双跑** | W3 fp16 | `op(*inputs)` 两次调用 6 输出 `torch.equal`（对齐 `test_engram_gate_conv_bwd.py` L97–L101） | 门槛通过 |
| L0-10 bf16 全链 | W4 bf16 | `vcast` 升/降无精度异常（对照 golden） | 门槛通过 |
| L0-11 d 对齐敏感性 | `d=256`（`BLK_S=8`）与 `d=512`（`BLK_S=4`）两档 | UB 预算 + 无尾块 + K1/K2 编译不溢出 | 门槛通过 |
| L0-12 部分和/combine 一致性 | W3 `NB=16` | `ws_*` 部分和之和 == K3 输出（逐位或 ≤1e-6） | 门槛通过 |

- **执行方式**：`tilelang.compile(k1/k2/k3, target="npuir")` 后经 `EngramGateConvBwdOp`/wrapper 单次调用，`torch.testing.assert_close(op_out[i].cpu(), golden[i].cpu(), atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-12 全过即进入 Stage 3；任一失败不得继续。**特别地 L0-11 为 K2 UB 编译门禁**（§4.5 强制动作）。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，32/32 `EQUIV_PASS`，§1.6.1）；`history_version/design_probe_bwd_structure.py`（D-3 结构探针 P1–P6，§1.6.3）。

---
## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 单 prim_func 多 `T.Kernel` 不支持**（探针 P1，grid 求值 None） | 无法照源 5 块结构 | 3-kernel Python 闭包（§0.6 R1） |
| **R-2 `T.atomic_add` 崩溃**（探针 P4，SIGABRT） | 权重梯度无法原子累加 | 确定性部分和 + combine（§0.6 R5） |
| **R-3 `T.Parallel` 内运行时 `T.min/T.max` 标量 store 错值**（探针 P6b；`TRAP-npuir-parallel-runtime-minmax`） | 行掩码/窗口错值 | 编译期 unroll 标量语句（§0.6 R4） |
| **R-4 带 buffer 操作数的 `T.if_then_else` 段错误**（`TRAP-npuir-ifthenelse-parallel-segv`） | 边界处理直接崩 | 索引 clamp + 算术行掩码（§0.6 R4） |
| **R-5 裸标量广播段错误**（`TRAP-npuir-vec-scalar-broadcast`，1 向量操作数 out-of-place） | codegen 段错误 | 统一 `T.vbrc`+`T.vmul`（≥2 向量操作数 / in-place） |
| **R-6 `T.vcast` 要求 src/dst 等 shape**（`TRAP-npuir-vcast-shape-mismatch`）/ store 处 direct-cast 被拒 | 输出写回编译失败 | 先 `T.copy` 到同形 buffer 再 `T.vcast`；先 `T.vcast` 到 dtype UB 再 `T.copy` |
| **R-7 `(R,1)` 归约 own-rows 标量输出不确定**（`TRAP-npuir-parallel-scalar-store-race`） | 权重梯度写出不确定 | unroll fill + 单次 `T.copy`；GM 写回一律 `T.copy` |
| **R-8 UB 容量**（`TRAP-UB-multibuffer-inflation` status=verified；`CG-2026-0008` 基础分配差） | K2 大 `BLK_S` 可能超 192KB | §4.5 按 d 档：K2 复用后 ≈103KB（d=256, `BLK_S=8`）/ ≈108KB（d=512, `BLK_S=4`）——**两档均 < 192KB**；仍保留 Stage 3 强制编译探针读精确需求 + `BLK_S` 再降回退 |
| **R-9 bf16 v-prefix 数学 API 缺失**（`vadd/vmul/vdiv/vsqrt/vsigmoid`、`reduce_sum` 各 op doc bf16 ✗） | bf16 workload 直传编译失败 | 全链 fp32 中转（`vcast`，§0.6 R2） |
| **R-10 串行 `+=` 累加错值**（探针 P3） | 部分和实现错 | 用 `T.reduce_sum(dim=0)`/`T.vadd`（探针 P2） |
| **R-11 `T.vbrc` `(R,1)→(R,1)` 被拒**（`TRAP-vbrc-empty-broadcast`） | 广播维为 1 时报错 | 本算子广播目标 N=d_padded>1（合法）；`rmk` 广播到 `(R,d)` ✓ |
| **R-12 非整除 seq_len 尾块** | 末 tile 输出越界 | `tail` 双侧裁剪 + 索引 clamp + 行掩码；manifest 不触发 |
| **R-13 输出契约数（6 vs 脚手架 register_fake 8）** | 集成期 fake/real 不一致 | §3.4：Stage 3 对齐 `register_fake` 到 6；工厂返回闭包（非 `JitKernel_NPU`，`TRAP-outidx-call-convention` 不适用） |
| **R-14 fp32 GM 输入路径**（`TRAP-npuir-fp32-vector-reduce`） | fp32 输入编译失败 | **不触发**（manifest 输入 fp16/bf16，fp32 仅片上 UB 中转） |
| **R-15 常数表设备/版本**（`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-ascend910-9362-profile` 均绑 910B2C/9362，无 910B3 条目，`kb_stale_check` 标 stale） | D-2 估算不确定性 | 数字标注「未文档化假设 + 估算依据」；Stage 4 baseline 后对账回填 |
| **R-16 发射项主导（名义）** | §1.6.0 估算示发射项主导（51µs） | 标注 stale 常数；`BLK_S`/融合度为杠杆；Stage 4 后对账 |
| **R-17 UB buffer 就地覆写别名**（`PL-1.23-vop-chain-inplace-alias` status=verified；v1 修订触发） | 读取被就地覆写的 buffer 原值 → 静默语义错（v0 的 `drms_w_h`；独立复算相对误差 ~34%） | **读取先于覆写**：消费被覆写 buffer 原值的计算必须前移到覆写之前（§3.3 注 13）；`verify_equiv.py` 的 reduction FULL 行（`drms_w_h/drms_w_v/dconv_w`）为回归护栏 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 单 prim_func 写多 `T.Kernel` | 阶段组织 | grid 求值失败 | 3-kernel 闭包（R-1） |
| `T.atomic_add` | 权重梯度归约 | SIGABRT | 部分和 + combine（R-2） |
| `T.Parallel` 内运行时 `T.min/T.max` | 行掩码 | 静默错值 | 编译期 unroll（R-3） |
| `T.if_then_else` 带 buffer 操作数 | 卷积边界 | codegen EXIT=139 | 索引 clamp + 算术掩码（R-4） |
| 裸标量广播 `dst[j]=s*vec[j]`（1 向量操作数 out-of-place） | `α⊙v` 写法 | codegen 段错误 | `T.vbrc`+`T.vmul`（R-5） |
| store 处 `T.cast(...)` direct / 跨 shape `vcast` | 输出写回 | MLIR verifier 拒 | `T.copy` 同形 + `T.vcast`（R-6） |
| 标量行输出直接元素级 store | `drms_w_*` 落盘 | 不确定 | unroll fill + `T.copy`（R-7） |
| 串行 `+=` 累加 | 部分和 | 错值 | `reduce_sum(dim=0)`（R-10） |
| bf16 直接传入 v-op | 未 `vcast` 上转 | 编译失败 | bf16 先 `vcast(rint)` 到 fp32（R-9） |
| `vcast` 非 `rint` 升级 | f16→f32/bf16→f32 仅 `rint` | 编译失败 | 升降统一 `round_mode="rint"` |
| `T.copy` 跨 dtype | 用 copy 兼做 cast | 隐式 `VCastOp`、舍入不可控（`TRAP-C12`） | 显式 `T.vcast` |
| `for p in range()` 写在 prim_func 内 | tap/掩码展开 | 运行时 Var → 动态 slice 非法（`TRAP-C13`） | factory 层展开 |
| buffer 在 `if` 内分配 | 顶层分配纪律 | `Undefined variable`（`TRAP-conditional-alloc`） | 顶层无条件分配 |
| UB 溢出 | K2 `BLK_S`/buffer 过多 | 编译失败（`ub overflow`） | §4.5 预算 + 编译探针（R-8） |
| 转置输出行映射错 | `dvn[i]=Σcw[p]*dsw[i+3-p]` | 语义错 | 对照 golden（无膨胀，`dst_t=t+3-p`） |
| 掩码方向错（左/右） | K1 左掩码 / K2 右掩码 | 边界错误 | L0-5/L0-6 逐位核对 |
| **读取被就地覆写的 buffer 原值**（别名） | `drms_w_h` 读 `dhn/dkn`（已被 `dH/dk` 覆写） | 静默语义错（`Σ(dH·H·rrh+dk·k·rrk)`） | 读取先于覆写（§3.3 注 13，R-17）；reduction FULL 行护栏 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **卷积左边界（t<3）** | K1 tile 0 的 halo 行 `gt1<0` 由索引 clamp（读 token 0）+ 行掩码清零；与 golden 左零填充等价（L0-5） |
| **转置右边界（t+3-p≥T）** | K2 末 tile 的 halo 行 `gt2≥T` 由索引 clamp（读 token T-1）+ 行掩码清零；与 golden 右零填充等价（L0-6） |
| **非整除 seq_len** | 末 tile `tail` 裁剪 + 索引 clamp（R-12；manifest 不触发） |
| **d 非 256 倍数（通用）** | host 尾维 pad 到 `d_padded` + 输出 crop（wrapper 既有；manifest 不触发） |
| **bf16 workload** | 全链 `vcast(rint)` 中转（R-9） |
| **门控饱和（alpha→0/1）** | `sigmoid` 全域稳定；`d_vnorm`/`d(vhat)` 无除零 |
| **`rms_w_h/rms_w_v = ones`（manifest）** | 权重仍走通用路径（`wh=1`），无特化 |
| **NaN/Inf 输入/中间值** | 契约 workload 不含；`rrms_*` 为输入正值；Stage 3 `equal_nan=True` 口径核对 |
| **多次运行确定性** | 权重梯度为确定性两段式归约（非 atomic）；L0-9 双跑 `torch.equal` |
| **极大 seq_len/ M** | 逻辑核数 `M·ceil(seq_len/BLK_S)` 随规模增长；超 40 时触发运行时串行调度（当前 manifest 不涉及，若扩展按 §5.5 第三分支核内 `T.serial` persistent 化或 `BLK_S` 下调） |

---
## 10. 交付清单

### 10.1 目录结构

```
examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # D-1 等价性机器验证（32/32 EQUIV_PASS）
└── history_version/
    └── design_probe_bwd_structure.py   # D-3 结构探针（P1..P6，反向专属）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | **已完成** | 本设计文档（含 §0 迁移分析、§1.6 算法调研与优化、§5.5 分核三要素） |
| `verify_equiv.py` | **已完成** | D-1：32/32 `EQUIV_PASS`（§1.6.1） |
| `history_version/design_probe_bwd_structure.py` | **已完成** | D-3 结构探针（P1–P6；§1.6.3） |
| `_engram_gate_conv_bwd_kernel.py` | 待实现（Stage 3） | 按 §3.3 伪代码 + §5 分核实现（含 §8.1 golden） |
| `RETROSPECTIVE.md` | Stage 1 复盘（追加） | 见同目录 |

> 项目名 `engram_gate_conv_bwd` 由 conductor 传入；下游 Stage 3 产出在脚手架 wrapper 目录 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/`。

### 10.3 命名规范

- 算子目录名: `_engram_gate_conv_bwd_kernel`（snake_case，与迁移函数名一致）
- 实现文件: `_engram_gate_conv_bwd_kernel.py`
- 测试/基准: 复用 `examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py` / `benchmarks/ops/bench_engram_gate_conv_bwd.py`

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）
2. ✅ 等价性机器验证（`verify_equiv.py`）+ 结构探针（`history_version/design_probe_bwd_structure.py`）
3. ⬜ Golden 函数（§8.1，移植源仓参考实现）
4. ⬜ 算子实现（`_engram_gate_conv_bwd_kernel.py`）+ 与 Golden 精度比对 + L0 门槛（**先过 K2 UB 编译探针 L0-11**）

**实现要点速览（Stage 3 输入）**：
1. **结构**：工厂返回 Python 闭包调度 3 个独立 `@tilelang.jit kernel`；K1/K2 一维 grid `M*num_tiles`（`BLK_S = 8 if d≤256 else 4`）、K3 grid 1。
2. **K1**：`vhat` 窗口 `(R,d)`（clamp+左掩码 unroll）→ `vn` 物化 → 因果 DWConv（tap 编译期展开）→ `d_silu`（`vsigmoid`）→ 写 `dvhat_buf`；`dconv_w` 部分和（`reduce_sum(dim=0)`）。
3. **K2**：`d_silu` 窗口（前向 halo，clamp+右掩码 unroll）→ conv 转置 → RMSNorm(vhat) 反向 + 门控反向 + RMSNorm(H)/RMSNorm(k) 反向（`dot_v/h/k` 行向 `reduce_sum(dim=1)`）；`drms_w_h/v` 部分和（`reduce_sum(dim=0)`）；`dH/dk/dv` `vcast(rint)`+`T.copy`。
4. **K3**：`ws_*` 列归约 `reduce_sum(dim=0)` + unroll fill + `T.copy` → 3 个权重梯度输出。
5. **计算域**：fp16/bf16 经 `T.cast`/`T.vcast(rint)` 升 fp32，全链 fp32，末端 `vcast(rint)` 单次舍回（bf16 API 强制）。
6. **边界**：索引 clamp（`T.max/T.min`）+ **unroll** 算术行掩码（禁 `T.if_then_else` 带 buffer、禁 `T.Parallel` 内运行时 minmax）。
7. **纪律**：buffer 顶层无条件分配；`T.vbrc`+`T.vmul`（禁裸标量广播）；store 处禁 direct-cast；GM 写回走 `T.copy`；`atomic_add` 禁用。
8. **分核**：grid `M·num_tiles` ≤ 40（纯 Vector，最大 32 ≤ 40）→ 无需适配；`BLK_S=8`(d≤256)/`4`(d=512) 为 Stage-4 杠杆；K1/K2 UB 需编译探针（§4.5）。
9. **契约**：工厂返回 6 输出；`register_fake` 对齐到 6；调用约定 `factory(params)(tensors)`。

---
