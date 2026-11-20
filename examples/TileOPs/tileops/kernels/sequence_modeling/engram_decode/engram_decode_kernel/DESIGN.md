# _engram_decode_kernel 算子设计文档（迁移）

> **项目**：`engram_decode` ｜ **算子目录**：`examples/engram_decode/_engram_decode_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_engram_decode_kernel`（`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_decode.py` L59–L236）重设计为 NPU（`target="npuir"`）实现。算子 = 单 token decode 全融合流水线（GEMV 投影 → 3× RMSNorm → sigmoid 门控 → 膨胀因果深度卷积 + 状态更新 → SiLU + 残差）。
>
> **本版核心决策**：① **算法调研（§1.6.0）结论**：源 GEMV 的「d_mem 串行向量 FMA 链」被替换为 **Cube `T.gemm`（K 分块 + fp32 L0C 累加）**——GEMV 占 manifest FLOPs 的 99.4%、占 GM 流量的 88%，是 MAC 密集段，应落 Cube 单元（R4），且串行链的 op 数为 O(d_mem)（W3 达 2048 ops）；② **计算域 = fp32 中转**：NPU 侧 `vadd/vmul/vexp/vsigmoid/vsqrt/vdiv/vmin/vrsqrt` 的 dtype 矩阵**均不含 bf16**（各 op doc §2.2.1），bf16/fp16 一律经 `T.vcast(rint)` 升 fp32 计算、末端单次舍回（W4 为 bf16 workload，**API 强制**）；③ **RMSNorm 的 rsqrt 用 `vsqrt+vdiv` 绕法**：设备 `T.vrsqrt` 为近似指令（`TRAP-vrsqrt-plain-precision`，status=verified，~2.9e-3 rel），`sqrt(s)/s ≡ 1/sqrt(s)` 两 op 全精度（实测 1.07e-7），3 次 RMSNorm 累积放大风险被消除；④ **conv_state 左移**由源 `T.serial(max_conv_len-1)` × `T.Parallel(d_padded)` 改为**一次二维区域 `T.copy`**（GM→UB→GM）；⑤ 4 次 `T.reduce_sum`、`v_hat = alpha*v` 的 CSE 保留（源在步骤 11 重算，本设计物化复用）；⑥ 膨胀索引 `max_conv_len-(w-1-p)*δ` 为编译期常量，按 tap 显式展开读取（避免 `TRAP-C13` 的运行时 Var 索引致动态 slice）。
>
> **源优化意图承接（§0.4 → §0.6/§1.6，逐项不丢弃）**：源 #1 单 kernel 全融合 → 保留（NPU 单 kernel）；源 #2 fragment 驻留 → 等价替换（UB buffer + gemm fragment/L0C）；源 #3 grid=(B,) 每 batch 一 block → 保留（一维 `T.Kernel(B)`）；源 #4 fp32 累加 → **保留并强化**（bf16/fp16 强制 fp32 中转）；源 #5 `threads` autotune + `lru_cache` 特化 → `threads` 舍弃（K3/K9，npuir 无效）、`lru_cache` 编译期特化保留；源 #6 256 对齐 pad → 保留逻辑（manifest `d%256==0` → 实际不触发）；源 #7 2D fragment 供 `reduce_sum` → 等价替换（UB 上 `T.reduce_sum`）；源 #8 重算 `alpha*v` → **重新设计**（CSE 物化，§1.6.1 O3）；源 #9 `T.Parallel` 逐元素 → 等价替换（v-prefix 整 buffer 向量 op）；源 #10 **GEMV 串行 FMA 链 → 重新设计**（Cube T.gemm，§0.6 R1）；源 #11 状态左移串行循环 → 等价替换（二维区域 `T.copy`，§1.6.2）；源 #12 编译期膨胀索引 → 保留；源 #13 左填充零状态（conv 无边界检查）→ 保留。

---

## 本版相对上一版（v0）的关键调整 ⭐（revision_index=1）

> **触发源**：Stage 2 检视 `REVIEW.md` 结论「不通过」（2 阻塞 + 3 建议，集中在 §1.6.3 布局/候选决策与计数口径）。本版**仅修订检视指出的问题**；**主选算法（Cube `T.gemm` GEMV + fp32 中转 + `vsqrt/vdiv` rsqrt 绕法 + CSE 物化 + 二维区域搬运）与 §0 迁移决策（源算子三问解读 / 耦合性判定 / NPU 重设计）均不变**。

| # | 检视问题（严重度） | v0 表述 | v1 修订（本版） |
|---|--------------------|---------|-----------------|
| 1 | §1.6.3 候选 C 弃选前提与 API 文档矛盾（**阻塞**） | 「`T.transpose` … dtype 矩阵 **bf16 ×**」「bf16 不可用」 | **删除 dtype 负向断言**；逐格重核 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 Ascend dtype 行（int8/int16/int32/int64/fp16/fp32/**bf16 = √**；uint8/uint16/uint32/bool = ×），§2.3 仅「二轴交换」限制；弃选理由改为「dtype 支持 bf16，但主选 A 已由 load 侧 `a_transpose` 零 repack 处理转置，候选 C 无收益」。§1.6.3 候选表 C 行 + 弃选论证证据段 + 布局决策结论**三处同步修正** |
| 2 | §1.6.3 备选 B 的 UB 预算漏计 `e_brc`/`w` tile（**阻塞**） | 只计一个 `(BK=32,d_padded) f32`（64KB）+ 向量 buffer ≈20KB → ×1.7 ≈143KB | **列全常驻 buffer 重算**（@d=512, **BK 32→8**，`e_brc` 经 `T.vmul(w_f,e_brc,e_brc)` 原地复用为乘积）：`w_h` 8KB + `w_f` 16KB + `e_brc` 16KB + `acc_k/acc_v` 4KB + 共享向量链 59KB = **103KB raw → ×1.7 ≈ 175KB < 192KB ✓**；补 **Stage 4 UB 编译探针计划**（引用 VP-2026-0059「预算表 ≠ 可编译证明」）。候选表 B 行 / R4 表 B 行 / §1.6.1 E1 行同步 |
| 3 | §4.5 f32 buffer 计数与 §3.3/§4.3 不一致（建议） | 「~11 × `(d_padded,)` × 4B」→ W3 ≈48KB | 统一为 **17** 个 f32 buffer（§3.3/§4.3 同源列名）→ W3 **≈59KB**、×1.7 **≈101KB < 192KB ✓**（结论不变）；§4.5 表、§5.3、§9 R-8 同步 |
| 4 | §0.4 #2 `T.alloc_fragment` 计数（建议） | 「`alloc_fragment` **×11**，L97–L112」 | 更正为 **×14**（6 向量 + 4 个 2D + 4 标量）；§1.6.0 R3 基线行同口径更正为「fragment 14 个（…）」 |
| 5 | §3.3 conv tap p=0 初始化与载入次序不自洽（建议） | 先 `T.vmul(cwf, win_f, c_out)` 初始化，`cwf/win_f` 载入写为「略」 | step 11 补齐 p=0 的 `T.copy(conv_state[…]→win)` + `T.vcast(win,win_f,"rint")` + `T.copy(conv_w[0]→cw_dtype)` + `T.vcast(cw_dtype,cwf,"rint")`，**再**写初始化；与 §3.2 API 映射次序一致（并补齐循环内 `cwf` 逐 tap 载入） |

**为何不会再犯同一错误**：

- **问题 1**（「引章节号未逐格取值」，ALG-transpose 已记载更正形态 / VP-2026-0168）：今后引用任何 API dtype 矩阵时**逐格读行取值**并注明文档章节路径（本版已对 `T.transpose.md` §2.2.1 逐格取值、对 §2.3 限制单列）；弃选候选的负向论断一律**先枚举 API 名字再全目录检索**（`negative-claim-evidence.md` §1/§2），负向断言与所引文档不符即视为决策依据无效。
- **问题 2**（VP-2026-0059「手工逐 buffer 预算 ≠ 可编译证明」）：主选/备选 UB 预算今后**列全所有常驻 buffer**（含中间 tile、广播 tile、累加器、以及共享向量链的全量 17 个 f32 buffer），不以单个代表性 buffer 代替；凡预算贴限（×1.7 后 > 80% 上限）即附**编译探针计划**（上靶前 `tilelang.compile` 读 `ub overflow` 报文取精确实测需求）。
- **问题 3 / 4**（计数口径漂移）：预算表/对比表的 buffer 计数与正文声明（§3.3/§4.3）**同源**——本版 §4.5 显式标注计数为「与 §3.3/§4.3 一致的 17 个」，§0.4/§1.6.0 的 fragment 计数同步为 14。
- **问题 5**（伪代码压缩致次序不自洽）：伪代码的「载入 → 使用」次序与 §3.2 API 映射表逐条对齐，压缩省略处标注为「压缩表达」而非留「略」。

**本版未改动（Stage 2 已通过项，不得破坏）**：§0.1–§0.7 迁移三问/耦合性/重设计、§1.6.0 调研四问与选定算法、§1.6.1 采纳项（O1–O5）、§1.6.2 向量化结论、§5.5 分核三要素、§8 golden 独立性。`verify_equiv.py` 的候选/基线式**未随本版变更**（本版未触及任何采纳项的数学式），已重跑确认 **28/28 EQUIV_PASS**（§1.6.1 结果表不变）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 算法调研 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_decode.py`

- 本迁移对象 = 工厂 `_engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype)`（L59–L236）内的 `@tilelang.jit(out_idx=[8,9])` 函数 `_func(threads)` → `@T.prim_func main(args)` → `@T.macro _decode_fused`（L82–L213，全部计算）。
- 辅助：`ALIGNMENT=256`（L52）+ `_align_up`（L55–L56）；`@torch.library.custom_op("top::engram_decode")` 包装（L239–L262）+ `register_fake`（L265–L275）；`class EngramDecodeKernel`（L278–L393，`supported_archs=[80,86,89,90]`、`default_config={"threads":128}`、`autotune_configs=[128,256,512]`）。
- host 侧 `forward`（L342–L393）：`conv_state` 左 pad 到 `max_conv_len`（`F.pad(..., state_pad, 0)`）、全部张量尾维 pad 到 `d_padded=_align_up(d,256)`、调用后 crop 回 `d`。
- NPU 脚手架（Stage 0 已生成，只读参照）：提取内核 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_kernels.py`（Pattern B）、wrapper 同目录 `engram_decode.py`（K5/K7/K8/K9 已改：`supported_archs=None`、`npub::engram_decode`、去 autotune/tune、去 `threads`、`ascend_mode="Developer"`）、op `examples/TileOPs/tileops/ops/sequence_modeling/engram_decode.py`（`EngramDecodeOp` + `eval_roofline`）。
- **本仓既有 NPU 参照（同算子）**：`examples/engram/engram_decode.py`（`target="npuir"`、`is_npu=True`、`T.alloc_shared`、`T.vrsqrt`、串行 GEMV）——是**只读参照**，其 `T.vrsqrt` 与串行 GEMV 均为本设计显式改进点（§0.5/§1.6）。

### 0.1 源算子语义（做什么）

**数学语义**（源码 docstring L5–L14 + `@T.macro` L82–L213 逐语句核对，与 `op_requirements.math_formula` 一致）：

```
k   = e @ W_K                                  # GEMV: (d_mem,) -> (d,)
v   = e @ W_V                                  # GEMV: (d_mem,) -> (d,)
h_n = RMSNorm(h, w_h)                          # w_h = rms_w_h
k_n = RMSNorm(k, w_h)
alpha = sigmoid(dot(h_n, k_n) / sqrt(d))       # 标量门控
v_hat = alpha * v
v_n = RMSNorm(v_hat, w_v)                      # w_v = rms_w_v   （即 v_hat_norm）
conv_out = Σ_{p=0}^{w-1} conv_w[p] ⊙ window[p]
           window[p] = state[max_conv_len-(w-1-p)*δ]   (p<w-1，δ=dilation)
           window[w-1] = v_n                            (current)
new_state = shift_left(state) ⊕ v_n            # 左移 1、末行 append
y = SiLU(conv_out) + v_hat                     # y = conv_out·sigmoid(conv_out) + alpha·v
```

其中 `RMSNorm(x,w) = x · rsqrt(mean(x²)+eps) · w`，`mean` 按**真 d**（非 d_padded）归一（源码 L135/L143/L161 用 `sumsq/float(d)+eps`）。

**I/O 契约**（源码 `_decode_fused` 签名 L82–L94 + wrapper L342–L393）：

| 参数 | Shape（kernel 内） | dtype | 说明 |
|------|--------------------|-------|------|
| `e_t` | `(B, d_mem)` | `float16 \| bfloat16` | 当前 token 的 N-gram 聚合嵌入 |
| `h_t` | `(B, d_padded)` | same | 当前 token 的隐状态（host 侧尾维 pad） |
| `conv_state` | `(B, max_conv_len, d_padded)` | same | 历史 `v_hat_norm` 缓存（左零填充，host 侧先左 pad 到 `max_conv_len`） |
| `W_K, W_V` | `(d_mem, d_padded)` | same | 投影权重 |
| `rms_w_h` | `(d_padded,)` | same | h/k 的 RMSNorm 权重 |
| `rms_w_v` | `(d_padded,)` | same | v_hat 的 RMSNorm 权重 |
| `conv_w` | `(w, d_padded)` | same | 深度卷积权重（w=conv_kernel_size 为模型参数） |
| `y_t`（out[0]） | `(B, d_padded)` | same | 输出（作为残差加到 h_t） |
| `new_conv_state`（out[1]） | `(B, max_conv_len, d_padded)` | same | 更新后的状态 |

- **输出 shape（迁移必需）**：`y_t:(B,d)` 与 `new_conv_state:(B, max_conv_len, d)`（host 侧 crop 掉 `d→d_padded` 的 pad，wrapper L390–L392；与 `op_requirements.source_output_shape` 一致）。
- **layout**：全部行主序连续，尾轴 `d` 连续；无转置、无 in-place。
- **动态轴**：无（`B/d_mem/d/max_conv_len/w/δ/eps/dtype` 全为编译期常量，`lru_cache` 按参数特化）。`forward` 里 `conv_state` 的运行时长度 `L ≤ max_conv_len` 由 host 左 pad 吸收，kernel 只认 `max_conv_len`。

**规约语义**（关键）：4 处 `T.reduce_sum`（L134/142/150/160）——`sumsq_h`、`sumsq_k`、`dot_hk`、`sumsq_v`，均为**行内沿 d 轴的 fp32 求和**（`2D fragment (1,d_padded)` → `(1,)`）。累加顺序 = 未指定（GPU 线程级 reduce），fp16/bf16 下与 golden 的差异落在测试容差（`atol=5e-2` fp16 / `1e-1` bf16，`rtol=5e-2`）内。**边界语义**：`d_padded` 区域由 host 零填充（h/conv_state/rms_w/conv_w 的 pad 位为 0），故 `Σ_{d_padded} x² = Σ_{d} x²`（源除以 `float(d)` 正确）；conv 的越界膨胀位读取落在左零填充区，贡献 0（源码注释 L171–L177）；`max_conv_len ≥ δ(w-1)` 由 `EngramDecodeKernel.__init__`（L321–L326）强制。

**dtype 语义**：源全程 `accum_dtype="float"`（fp32）累加/计算，仅在写回 `y_t`（L210）与 `new_conv_state` 末行（L203）时 `T.cast(..., dtype)`。即 fp16/bf16 输入 → fp32 计算域 → 单次舍回。

**host 侧语义**：`F.pad` 左填充 `conv_state` 到 `max_conv_len`、八个张量尾维 pad 到 `d_padded`、结果 crop——均为**元数据/视图级**处理（不改输入张量真实内容），属算子契约外的实现手段，NPU 侧保留（wrapper 已实现）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（**优先移植源仓参考实现** `examples/TileOPs/tests/ops/test_engram_decode.py::engram_decode_step_torch`，L38–L86），**不得复刻 §0.6 的 NPU 结构决策**（保证验证独立性）。

### 0.2 源算子输入输出

见 §0.1 I/O 契约表（kernel 内 shape 为 `d_padded`/`max_conv_len` 版本；对外契约 shape 为 `d` 版本，见 `engram_decode.py` op manifest `EngramDecodeOp.signature`：`e_t:[B,d_mem]`、`h_t:[B,d]`、`conv_state:[B,max_conv_len,d]`、`W_K/W_V:[d_mem,d]`、`rms_w_h/rms_w_v:[d]`、`conv_w:[w,d]`；输出 `y_t:[B,d]`、`new_conv_state:[B,max_conv_len,d]`；`shape_rules: d%256==0`）。

**dtype 语义（迁移必须保持）**：fp16/bf16 输入 → fp32 累加计算 → 单次舍回原 dtype（源 `accum_dtype="float"`）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解表**（`@T.macro _decode_fused` 逐语句，L95–L213；源码每个计算语句均归入某步骤）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|------|------|------|------|-------------|
| 1 | 载入 e | `T.Parallel(d_mem)`：`e_local[i]=cast(e_t[bid,i])` | GM `e_t` | frag `e_local` | `e` |
| 2 | **GEMV（K/V）** | 清 0（L119–121）→ `T.serial(d_mem)` × `T.Parallel(d_padded)`：`k_local[j]+=e_local[i]·cast(W_K[i,j])`；`v_local[j]+=e_local[i]·cast(W_V[i,j])` | `e_local`、`W_K/W_V` | frag `k_local`/`v_local` | `k=e@W_K`, `v=e@W_V` |
| 3 | 载入 h | `T.Parallel(d_padded)`：`h_local[j]=cast(h_t[bid,j])` | GM `h_t` | frag `h_local` | `h` |
| 4 | **RMSNorm(h)** | `hsq_2d[0,j]=h²`（L133）→ `T.reduce_sum(dim=1)` → `rrms_h=T.rsqrt(sumsq_h/d+eps)`（L135）→ `h_local[j]=h_local[j]·rrms_h·cast(rms_w_h[j])` | `h_local`、`rms_w_h` | `h_local`(归一后) | `h_n` |
| 5 | **RMSNorm(k)** | 同式：`ksq_2d=k²` → reduce → `rrms_k=vrsqrt` → `k_local=k_local·rrms_k·rms_w_h` | `k_local`、`rms_w_h` | `k_local`(归一后) | `k_n` |
| 6 | **标量门控** | `hk_2d[0,j]=h_local[j]·k_local[j]` → reduce → `alpha=1/(1+exp(-(dot_hk/T.sqrt(d))))`（L151） | `h_local`,`k_local` | 标量 `alpha` | `alpha` |
| 7 | v_hat | `T.Parallel(d_padded)`：`vhat_local[j]=alpha·v_local[j]`（L155） | `alpha`,`v_local` | `vhat_local` | `v_hat` |
| 8 | **RMSNorm(v_hat)** | `vsq_2d=vhat²` → reduce → `rrms_v=vrsqrt` → `vhat_local=vhat·rrms_v·rms_w_v`（**原地覆写 → v_hat 丢失**） | `vhat_local`、`rms_w_v` | `vhat_local`(=v_n) | `v_n` |
| 9 | **膨胀因果卷积** | `conv_out[j]=0`（L180）→ `T.serial(w-1)` × `T.Parallel(d_padded)`：`conv_out[j]+=cast(conv_w[p,j])·cast(conv_state[bid, max_conv_len-(w-1-p)·δ, j])`（L183–L191）→ 当前 tap：`conv_out[j]+=conv_w[w-1,j]·vhat_local[j]`（L193–194） | `conv_w`,`conv_state`,`vhat_local` | `conv_out` | `conv_out` |
| 10 | **状态左移 + append** | `T.serial(max_conv_len-1)` × `T.Parallel(d_padded)`：`new_conv_state[bid,s,j]=conv_state[bid,s+1,j]`（L199–201）→ `new_conv_state[bid,max_conv_len-1,j]=cast(vhat_local[j])`（L202–203） | `conv_state`,`vhat_local` | GM `new_conv_state` | `new_state` |
| 11 | **SiLU + 残差** | `T.Parallel(d_padded)`：`sig=1/(1+exp(-conv_out[j]))`；`y_t[bid,j]=cast(conv_out[j]·sig + alpha·v_local[j])`（**重算 alpha·v**，L208–213） | `conv_out`,`alpha`,`v_local` | GM `y_t` | `y` |

**数据流图（源硬件 = GPU 视角）**：

```
GM[e_t] --T.Parallel(threads 映射 d_mem)--> REG[e_local] ─┐
GM[W_K] --(每 i 行, register load)--> REG ────────────────┼─(serial i=0..d_mem-1, FMA over d_padded)--> REG[k_local]
GM[W_V] --...--> REG ─────────────────────────────────────┘                                                   REG[v_local]
GM[h_t] --T.Parallel--> REG[h_local] --square--> REG[hsq_2d] --T.reduce_sum(dim=1)--> REG[sumsq_h] --T.rsqrt--> rrms_h
                                     --(h·rrms_h·rms_w_h)--> REG[h_local(归一)]
GM[rms_w_h] --T.Parallel 广播读--> REG
             REG[k_local] --square--> REG[ksq_2d] --reduce--> sumsq_k --vrsqrt--> rrms_k --(k·rrms_k·rms_w_h)--> REG[k_local(n)]
             REG[h_local(n)] ⊙ REG[k_local(n)] --> hk_2d --reduce--> dot_hk --(1/(1+exp(-x/√d)))--> scalar alpha
             alpha ⊙ REG[v_local] --> REG[vhat_local] --square--> vsq_2d --reduce--> sumsq_v --vrsqrt--> rrms_v --(·rms_w_v)--> REG[vhat_local(n)]
GM[conv_state] --(每 tap 行 w-1 次, 膨胀位置)--> REG --FMA conv_w[p]--> REG[conv_out]  (+ current tap ⊙ vhat_n)
GM[conv_w] --...--> REG
GM[conv_state] --(serial s, T.Parallel d_padded, GM→REG→GM)--> GM[new_conv_state]   ; 末行 ← cast(vhat_n)
REG[conv_out] --(conv_out·sigmoid(conv_out) + alpha·v)--> GM[y_t]
```

- register/fragment 双缓冲由编译器管理；无显式 SMEM；`block_size = threads`（每 block 一个 batch，threads ∈ {128,256,512}）。

**循环与并行结构**：`T.Kernel(batch, threads=threads)` —— **一维 grid（`batch`）+ `threads` 线程/block** 两级并行（`op_requirements`：grid=(B,) 每 batch 一 block）。线程内映射：`T.Parallel(d_padded)` 把 d 元素分给 threads；GEMV/卷积/状态左移的归约或 tap 维度走 `T.serial`，元素维度走 `T.Parallel`。

**host 侧逻辑**：见 §0.1（左 pad 状态、尾维 pad 到 `d_padded`、结果 crop）；`lru_cache` 按 `(batch,d_mem,d,max_conv_len,w,δ,eps,dtype)` 编译期特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **单 kernel 全融合**（GEMV+3×Norm+gate+conv+state+SiLU 一个 `@T.prim_func`） | 消 kernel 启动与中间 GM 往返 | 全部中间量驻留寄存器 | 无（算法/结构层） | **算法层 → 可移植（保留）** |
| 2 | **fragment/register 驻留**（`T.alloc_fragment` ×14，L97–L112） | 中间量不出寄存器 | 大寄存器堆 + 编译器分配 | GPU 寄存器堆 | **硬件强相关 → 等价替换（UB）** |
| 3 | **grid=(B,) 每 batch 一 block** | batch 级任务并行 | `T.Kernel(batch)` | GPU block 调度 | 模式可移植 → **保留** |
| 4 | **全程 fp32 累加**（`accum_dtype="float"`，`T.cast(...,accum)` 于每处输入） | 数值精度（匹配 golden fp32 opmath） | fp16/bf16 → fp32 计算、末端单次舍回 | 半精度 ALU 下需中转 | **语义层 → 保留并强化（NPU bf16 API 缺失）** |
| 5 | **`threads` autotune（128/256/512）+ `lru_cache` 编译期特化** | 按 shape 选线程数 / 消运行时参数分派 | 编译期常量固化 | `threads` 无 NPU 对应 | `threads` → **舍弃**（K3/K9）；`lru_cache` → **保留** |
| 6 | **256 对齐 pad（`d_padded`）** | GPU 向量对齐、免尾界守卫 | host 侧 `F.pad` + 编译期 `d_padded` | memory coalescing | 实现层 → **保留逻辑**（manifest `d%256==0` 实际不触发） |
| 7 | **2D fragment `(1,d_padded)` 供 `T.reduce_sum(dim=1)`** | 行内归约 | `reduce_sum` 沿 dim=1 | GPU 线程级 reduce | 优化层 → **等价替换（UB reduce）** |
| 8 | **步骤 11 重算 `alpha·v`**（不保留 `vhat_local`） | 省 1 个 `(d_padded,)` 寄存器缓冲 | 复算换寄存器 | 寄存器稀缺 | **重新设计**（CSE 物化，§1.6.1 O3） |
| 9 | **`T.Parallel(d_padded)` 逐元素向量化** | 元素并行 | 线程级 elementwise | GPU SIMT | 优化层 → **等价替换（v-prefix 整 buffer）** |
| 10 | **GEMV = `T.serial(d_mem)` × `T.Parallel(d_padded)` 寄存器累加** | 低代价 GEMV（无 SMEM） | 长串行累加链 | 寄存器 file + 高发射率 | **硬件强相关 → 重新设计（Cube）** |
| 11 | **状态左移 `T.serial(max_conv_len-1)` × `T.Parallel(d_padded)`** | 逐行移位 + append | 行级串行、元素并行 | — | 优化层 → **等价替换（二维区域 copy）** |
| 12 | **膨胀索引编译期常量**（`max_conv_len-(w-1-p)·δ`，L188） | 免运行时索引算术 | 全部参数编译期 | — | 可移植 → **保留** |
| 13 | **左零填充状态（conv 免边界检查）** | 消越界分支 | pad 区贡献 0 | — | 可移植 → **保留** |

> 识别不出来 ≠ 不存在。上述 13 项为源码中可识别的全部优化/规避手段；逐项处置见 §0.5，未识别的优化会被静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0；源算法与调研候选同台比较）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义（GEMV+3×RMSNorm+gate+膨胀因果 conv+state+SiLU；I/O、边界、host pad 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| #1 单 kernel 全融合（含各步骤的数学顺序） | 算法 | 无（纯数学顺序，无跨 step 硬件假设） | — | **保留（结构）** | NPU 单 kernel；融合顺序不变 |
| #2 fragment/register 驻留 | 优化 | GPU 寄存器堆 | 无直接等价（NPU 无 `threads`，v-op 作用于 UB buffer；Cube 段用 fragment/L0C） | **等价替换** | → §0.6 R3：`GM→UB` 整块 `T.copy` + v-prefix 整 buffer 向量链；Cube 段 `T.alloc_fragment`/L0C |
| #3 grid=(B,) 每 batch 一 block | 优化 | GPU block 调度 | **有**（一维 `T.Kernel(B, is_npu=True)`） | **保留** | 一维 Kernel，语义与任务映射一致；§5.5 |
| #4 全程 fp32 累加 | 语义/优化 | 无 | **有**（fp32 计算 + `T.vcast`）；且 **NPU bf16 v-op 缺失使其成为 API 强制** | **保留（并强化）** | → §0.6 R2：fp16/bf16 经 `vcast(rint)` 升 fp32 → fp32 向量链 → `vcast(rint)` 舍回 |
| #5 `threads` autotune | 优化 | GPU 线程模型（`threads=` 在 npuir 无效，`TRAP-threads-kwarg-noop`） | 无 | **舍弃** | K3/K9：npuir 无 `threads`；`block_size` 折叠由向量宽度承载（§5.2） |
| #5b `lru_cache` 编译期特化 | 优化 | 无 | 有 | **保留** | 工厂保持全参数编译期常量（含 `BK`/`grid`） |
| #6 256 对齐 pad | 优化 | memory coalescing | 有（32B 尾轴对齐） | **保留（逻辑）** | manifest `d%256==0` → `d_padded==d`，不触发；保留变量以支持通用 d（§5.3） |
| #7 2D fragment `reduce_sum` | 优化 | GPU 线程级 reduce | **有**（`T.reduce_sum` on UB，`docs/.../规约操作/T.reduce_sum.md`） | **等价替换** | UB `(1,d_padded)` → `(1,1)` fp32 `T.reduce_sum(..., dim=1)`（§1.6.2 #1） |
| #8 重算 `alpha·v` | 优化 | 寄存器稀缺 | 有（UB 充裕） | **重新设计** | → §0.6 R4 / §1.6.1 O3：物化 `vhat`（fp32 `(d_padded,)`，≤2KB）复用 |
| #9 `T.Parallel` 逐元素 | 优化 | GPU SIMT 线程 | **有**（v-prefix 整 buffer） | **等价替换** | `vcast/vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc`（§1.6.2） |
| **#10 GEMV 串行向量 FMA 链** | 优化 | 寄存器 file + 高发射率；每投影 op 数 = **d_mem**（串行依赖链） | **有**（Cube `T.gemm`，`hivm.hir.mmadL1`） | **重新设计** | → §0.6 R1：Cube `T.gemm`（K 分块 `T.Pipelined` + fp32 累加）；GEMV 占 FLOPs 99.4%/流量 88% 属 MAC 密集段（§1.6.0 R4） |
| #11 状态左移串行循环 | 优化 | 无（仅为逐行搬运） | **有**（二维区域 `T.copy`） | **等价替换** | RS1：`T.copy(conv_state[bid,1:L,:], ub)` + `T.copy(ub, new_conv_state[bid,0:L-1,:])`（§1.6.2 #3） |
| #12 编译期膨胀索引 | 优化 | 无 | 有 | **保留** | tap 显式展开（w 编译期常量）；规避 `TRAP-C13` 运行时 Var 索引 |
| #13 左零填充状态 | 算法 | 无 | 有 | **保留** | 同语义；conv 无边界分支 |
| fp8 / 其他非 manifest dtype | — | — | 无（manifest 仅 fp16/bf16） | **舍弃** | 不在契约内 |

> **依据**：#4 的 bf16 缺失由 `docs/Tilelang.language/数学操作/T.vadd.md`·`T.vmul.md`·`T.vdiv.md`·`T.vexp.md`、`比较操作/T.vmin.md`、`数据类型转换操作/T.vcast.md` §2.2.1 的 dtype 矩阵佐证（vadd/vmul/vdiv/vexp/vsigmoid/vsqrt/vrsqrt/vmin 均 **bf16 ✗**；`vbrc` ✓、`vcast` f16↔f32/bf16↔f32 ✓、`T.gemm` bf16 ✓）；#10 的 Cube 可用性由 `docs/.../线性代数操作/T.gemm.md`（`hivm.hir.mmadL1`，fp16/bf16/fp32 输入、fp32 累加）+ `examples/gemv/example_gemv.py`（Developer `is_npu=True` + `T.gemm` GEMV 实跑）+ `examples/flash_attention/flash_attn_npuir_dev.py`（Developer + `T.gemm` + `T.vmul/T.vexp` 混排实跑先例）佐证；#5 由 `pattern-library/traps-compiler.md` `TRAP-threads-kwarg-noop`（status=verified）+ 脚手架 K3/K9（wrapper 已去 `threads`）佐证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目（R1/R4）以及判定为「等价替换但需结构级重写」的条目（R2/R3/R5；其中 R2 主体为「保留」，因 bf16 API 强制与 rsqrt 精度而附加结构变更）逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。**重设计直接承接 §1.6.0 调研选定算法族（Cube GEMV + fp32 中转 + rsqrt 绕法）。**

**重设计项 R1：GEMV 重设计（d_mem 串行向量 FMA 链 → Cube `T.gemm` K 分块）**

- **源方案**（步骤 2，L119–125）：`T.serial(d_mem)` 外层 + `T.Parallel(d_padded)` 内层，逐 i 做 `k_local[j]+=e_local[i]·W_K[i,j]`——每投影 **d_mem 次向量 FMA**，且对同一 `k_local` 形成**长度 d_mem 的循环携带依赖链**（W3：d_mem=1024 → 每投影 1024 op，两投影 2048 op）。性能意图 = 用寄存器累加避免 SMEM 往返。
- **NPU 新算法**：把两个投影实现为 **Cube `T.gemm`**（`hivm.hir.mmadL1`），K 轴（d_mem）分块 `BK=128` + `T.Pipelined` 双缓冲，累加器 fp32 fragment（L0C）：
  - 形式（沿用 `examples/gemv/example_gemv.py::naive_gemv_high_perf` 已验证的「M=输出维 / N=1」合同 + `a_transpose=True` 适配本算子的 `(K,N)` 权重布局）：
    `T.gemm(W_K_shared (BK, d_padded), e_shared (BK, 1), C_k (d_padded, 1), a_transpose=True, initC=(k==0))`，
    即 `C_k[j,0] = Σ_i W_K[i,j]·e[i]`（`a_transpose` → 有效 A 为 `(d_padded, BK)`，M=d_padded≥16、K=BK、N=1）；`W_V` 同式。
  - 备选形式（若 `a_transpose` 有编译/效率问题）：`A = e_pad (16, d_mem)`（M=16，行 0 = e、余行 0）+ `B = W_K (d_mem, d_padded)`，`C=(16,d_padded)` 取行 0（M 补零 16×，计算非瓶颈；见 §1.6.3 实验裁决）。
  - 结果 `C_k/C_v` 由 `T.copy` 落 UB（fp32），供后续向量链使用。
- **语义保持论证**：
  1. **数学等价**：`C[j,0]=Σ_i W[i,j]·e[i]` 与源逐 i 累加同式，仅**累加分组/顺序**变化（串行链 → K 分块树形）。源为 fp32 累加，Cube 亦 fp32 累加（`T.gemm.md` §2.3：fp16/bf16 输入 dst 必须 fp32）；`verify_equiv.py` O4 各行（B4/B8/B1 × fp16/bf16/fp32）与 fp64 精确值最大 abs 差 ≤ 3.1e-5（fp16 舍回后），违反率 0 → EQUIV_PASS。
  2. **边界语义**：`d_padded` 尾区 W 的 pad 列为 0、e 无 pad → `C[j]=0 (j≥d)`，与源一致；`e` 长度 d_mem 精确，K 分块末块按 `BK | d_mem`（d_mem ∈ {256,512,1024} 均整除 128）无尾块；若未来 d_mem 非整除，K 分块用 `tail=T.min(BK, d_mem-k0)` 裁剪。
  3. **精度**：Cube 累加器 fp32，与源 `accum_dtype="float"` 同域；全程无 fp16 累加。**源优化意图承接**：源 #10「寄存器累加避免 SMEM 往返」→ Cube 的 fp32 **L0C 累加器**承接（片上驻留、零中间 GM 往返），且把 O(d_mem) 的串行依赖链替换为硬件 K 归约（op 数从 O(d_mem) 降至 O(d_mem/BK)）。

**重设计项 R2：计算域与精度重设计（fp16/bf16 原生域 → fp32 中转 + rsqrt 绕法）**

- **源方案**：输入 `T.cast(..., "float")` 升 fp32 计算、末端 `T.cast(..., dtype)` 舍回（步骤 1/3/4/5/8/10/11）；RMSNorm 用 `T.rsqrt`。
- **NPU 新算法**：
  - **fp32 中转**：fp16/bf16 输入先 `T.vcast(..., round_mode="rint")` 升 fp32 → 全部向量链在 fp32 域 → 末端 `T.vcast(..., round_mode="rint")` 单次舍回（输出仅 `y_t` 与 `new_conv_state` 末行）。**bf16 为 API 强制**（vadd/vmul/vexp/vsigmoid/vsqrt/vdiv/vmin 均无 bf16）；fp16 保留同路径以对齐源 fp32 语义并统一实现。
  - **rsqrt 绕法**：`rrms = rsqrt(s)` → `r := sqrt(s)`；`rrms := r/s`（`vsqrt` + `vdiv`，等价 `1/sqrt(s)`），消除 `T.vrsqrt` 近似指令（`TRAP-vrsqrt-plain-precision` 实测 max rel err ~2.9e-3；本算子 3 次 RMSNorm 累积，风险叠加）。
- **语义保持论证**：① 计算域与源（fp32）及契约 golden（`k.float()@W.float()` + `.float()` RMSNorm，测试 `_rmsnorm_decode` L32–L35）**同域**，非替代；② `sqrt(s)/s ≡ s^{-1/2}`（s>0）实数恒等，fp32 下与真值最大 abs 差 6.1e-5、`(·)^3` 连乘场景违反率 0（`verify_equiv.py` O1 两行 EQUIV_PASS）；③ 边界：`s = sumsq/d+eps ≥ eps=1e-6 > 0`，rsqrt 定义域内（O1 用小角点 1e-6/1e-4 域覆盖）。**源优化意图承接**：源 #4「fp32 累加保障精度」→ 本项承接并**强化**（bf16 API 强制 + 绕开近似 rsqrt）。

**重设计项 R3：并行/内存结构重设计（fragment + `threads` → 一维 Kernel + UB + L1/L0C）**

- **源方案**：`T.Kernel(batch, threads=t)` + `T.alloc_fragment`，`threads ∈ {128,256,512}` autotune。
- **NPU 新算法**：一维 `T.Kernel(B, is_npu=True)`（每 batch 一 block，与源 grid 语义一致）；所有 (d_padded,) 中间量放 `T.alloc_ub`（fp32），Cube 段用 `T.alloc_shared`（→L1）+ `T.alloc_fragment`（→L0C）；`GM→UB` 用整块 `T.copy`，向量链用 v-prefix op 作用于整 buffer。`threads`/autotune 移除（K3/K9）。
- **语义保持论证**：任务映射仅重排「哪个核处理哪个 batch」，各 batch 完全独立（无跨 batch 依赖），数学逐元素恒等；无累加顺序变化；无边界影响。**源优化意图承接**：源 #2「中间量驻留片上」→ UB/L0C 承接；源 #1「单 kernel 融合」→ 保留。

**重设计项 R4：CSE 重设计（重算 `alpha·v` → 物化复用）**

- **源方案**（步骤 8/11）：`vhat_local` 被 RMSNorm 原地覆写为 `v_n`，步骤 11 用 `alpha·v_local` **重算** `v_hat`。
- **NPU 新算法**：额外分配 `vhat_ub (d_padded,)` fp32（≤2KB）物化 `v_hat = alpha·v`，供步骤 11 直接复用；`v_n` 另置 `vn_ub`。
- **语义保持论证**：`alpha·v` 为确定性 fp32 计算，物化值与重算值**逐位一致**（`verify_equiv.py` O3 三 dtype max_abs=0.0 EQUIV_PASS）；无边界/顺序影响。**源优化意图承接**：源 #8「省 1 个寄存器缓冲」在本设计中 UB 充裕（≤2KB / 192KB），CSE 收益（少 1 vmul）大于成本，故改判为「重新设计」。

**重设计项 R5：状态左移重设计（逐行串行循环 → 二维区域 `T.copy`）**

- **源方案**（步骤 10）：`T.serial(max_conv_len-1)` × `T.Parallel(d_padded)` 逐行搬移。
- **NPU 新算法**：`T.copy(conv_state[bid, 1:max_conv_len, :], st_ub)` + `T.copy(st_ub, new_conv_state[bid, 0:max_conv_len-1, :])`（两块二维区域 copy，经 UB 中转；GM→GM 非直连）；末行 `T.copy(vn_cast_ub, new_conv_state[bid, max_conv_len-1, :])`。
- **语义保持论证**：`new_state[s]=old_state[s+1]`（s<L-1）与 `new_state[L-1]=v_n` 逐元素恒等；slice 形态 `T.copy`（`TRAP-load-nd2nz-strided` 推荐的合法跨步绕法）；UB 驻留 `(max_conv_len-1)·d_padded·elem`（W3：19×512×2=19.5KB fp16）✓。**源优化意图承接**：源 #11 → op 数从 O(max_conv_len) 降至 3 次 copy。

> **承接说明（源优化意图不丢弃）**：#1→保留；#2→R3；#3→保留；#4→R2（强化）；#5→舍弃 `threads`（依据 `TRAP-threads-kwarg-noop`）+ 保留 `lru_cache`；#6→保留逻辑；#7→§1.6.2 #1；#8→R4；#9→§1.6.2 #2；#10→**R1**；#11→R5；#12→保留（tap 显式展开）；#13→保留。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/mnt/workspace/TileOPs/tileops/kernels/engram/engram_decode.py`（工厂 L59–L236、`EngramDecodeKernel` L278–L393）。
- 源仓测试基准（**golden 直接移植对象**）：`examples/TileOPs/tests/ops/test_engram_decode.py`（`engram_decode_step_torch` L38–L86 = PyTorch 参考；`_rmsnorm_decode` L32–L35；用例 `EngramDecodeFixture` L98–L107：`(1,512,256,12,4,3,fp16)` / `(1,512,256,12,4,3,bf16)` / `(4,1024,512,20,4,5,fp16)` / `(8,512,256,18,4,3,bf16)`；容差 fp16 `atol=5e-2` / bf16 `atol=1e-1`，`rtol=5e-2`；多步一致性 `test_engram_decode_multi_step` L122–L164）。
- manifest：`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `EngramDecodeOp`（signature/shape_rules/workloads/roofline 见 §8.2；smoke `decode-smoke-b1-dmem256-d256-fp16` + 3 个 full workload）。
- 同算子 NPU 只读参照：`examples/engram/engram_decode.py`（NPU 版串行 GEMV + `T.vrsqrt`，本设计显式改进）。
- 结构先例（只读）：`examples/gemv/example_gemv.py`、`examples/flash_attention/flash_attn_npuir_dev.py`（Developer + `T.gemm` + v-prefix 混排）、`examples/norm/rms_norm.py`、`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md`（norm 族迁移档案，vrsqrt 绕法）。
- golden 以 §0.1 语义为依据（移植 `engram_decode_step_torch`），**不复刻 §0.6 的 NPU 结构**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

`_engram_decode_kernel`（项目 `engram_decode`，目录 `examples/engram_decode/_engram_decode_kernel/`）。

### 1.2 功能描述

单 token（decode, seq_len=1）Engram 全融合算子：对每个 batch，把 N-gram 嵌入投影为 k/v（GEMV），对 h/k 做 RMSNorm 后经 sigmoid 门控得到 `v_hat`，对 `v_hat` 再做 RMSNorm，做膨胀因果深度卷积（读历史 `conv_state`）并更新状态，最后 `y = SiLU(conv_out) + v_hat`。支持 `float16 | bfloat16`；输出 `y_t:(B,d)` 与 `new_conv_state:(B,max_conv_len,d)`。含矩阵乘（GEMV 投影）+ 3 次行内规约 + 逐元素链，属**多步融合类**算子。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① GEMV 由「d_mem 串行向量 FMA 链」改为「Cube `T.gemm`（K 分块 + fp32 L0C 累加）」；② 计算域统一 fp32 中转（bf16 API 强制 + 源 fp32 语义）；③ RMSNorm 的 `vrsqrt` 换成 `vsqrt+vdiv`（绕近似指令）；④ 状态左移由串行循环改为二维区域 `T.copy`；⑤ `v_hat` 物化复用。**计算语义（数学公式、I/O、边界）与源一致。**

### 1.3 数学公式

$$
\begin{aligned}
k &= e\,W_K,\qquad v = e\,W_V && (\text{GEMV})\\
h_n &= \frac{h}{\sqrt{\operatorname{mean}(h^2)+\epsilon}}\odot w_h,\qquad
k_n = \frac{k}{\sqrt{\operatorname{mean}(k^2)+\epsilon}}\odot w_h && (\text{RMSNorm})\\
\alpha &= \sigma\!\Big(\tfrac{\langle h_n, k_n\rangle}{\sqrt{d}}\Big),\qquad
\hat v = \alpha\, v,\qquad v_n = \frac{\hat v}{\sqrt{\operatorname{mean}(\hat v^2)+\epsilon}}\odot w_v && (\text{gate}+\text{RMSNorm})\\
c &= \sum_{p=0}^{w-1} \mathrm{cw}_p \odot \mathrm{win}_p,\quad
\mathrm{win}_p = \mathrm{state}[\,L-(w{-}1{-}p)\delta,\ :\,]\ (p<w{-}1),\ \ \mathrm{win}_{w-1}=v_n && (\text{dilated causal conv})\\
y &= \mathrm{SiLU}(c) + \hat v,\qquad
\mathrm{new\_state} = \mathrm{shift}_1(\mathrm{state})\ \Vert\ v_n && (\text{SiLU+residual, state})
\end{aligned}
$$

其中 `mean(·)=Σ·/d`（**真 d**，非 d_padded）、`σ= sigmoid`、`L=max_conv_len`、`δ=dilation`、`w=conv_kernel_size`、`⊙` 为逐通道（逐元素）乘。

NPU 计算核（§1.6.1 优化后，fp32 计算域、末端单次舍回；形如 `rrms(s)=sqrt(s)/s`、`·(1/d)`、`·(1/√d)` 的常数折叠）：

$$
k=eW_K,\ v=eW_V\ (\text{Cube});\quad
\mathrm{rrms}(s)=\frac{\sqrt{s}}{s},\ s=\operatorname{mean}(x^2)+\epsilon;\quad
\alpha=\sigma\!\big(\langle h_n,k_n\rangle\cdot \underline{\mathrm{fl}(1/\sqrt d)}\big)
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在数学上等价、在并行/内存实现与累加分组上有差异：**

- **投影（§0.6 R1 / §1.6.0 选定）**：Cube `T.gemm`，M=d_padded、K=d_mem（BK 分块）、N=1；fp32 L0C 累加。
- **RMSNorm（§0.6 R2）**：fp32 行内规约 `T.reduce_sum`；`rrms(s)=sqrt(s)/s`（vsqrt+vdiv 绕法）；`s=sumsq·(1/d)+eps`（常数倒数折叠）。
- **门控**：`alpha=sigmoid(dot·(1/√d))`；点积后 `v_hat=alpha·v` 物化（§0.6 R4）。
- **膨胀因果卷积**：w 个 tap（w 编译期常量，显式展开），每 tap 一次向量 FMA；历史 tap 从 `conv_state` 的编译期固定行读取（左零填充区自动贡献 0）。
- **状态更新（§0.6 R5）**：二维区域 `T.copy` 左移 + 末行 append。
- **SiLU + 残差**：`y = c·sigmoid(c) + v_hat`，末端 `vcast(rint)` 舍回输出 dtype。
- 除上述外，与 §1.6.0 调研选定算法族一致：单 kernel 全融合、无中间 GM 往返（除状态读写的契约要求）、每 batch 一趟。

### 1.5 数据流图

```
Cube 段（每 batch 一 block）:
GM[e_t](B,d_mem) --T.copy--> L1[e_sh(BK,1)]                       GM[W_K/W_V](d_mem,d_padded) --T.copy--> L1[W(BK,d_padded)]
        --T.gemm(a_transpose, initC=(k==0))--> L0C[C_k/C_v (d_padded,1) f32] --T.copy--> UB[k_ub,v_ub (d_padded,) f32]

Vector 段（fp32 域；dtype=fp16/bf16 时先 vcast(rint) 升 fp32）:
GM[h_t]  --T.copy+vcast--> UB[h_f32]  --vmul(h·h)--> UB[h2] --reduce_sum--> S_h --(·1/d +eps)--> UB[s_h] --vsqrt--> r --vdiv(r/s)--> UB[rrms_h]
         UB[h_f32] ⊙ rrms_h ⊙ w_h  --(vbrc+vmul+vmul)--> UB[h_n]
         UB[k_ub] --vmul(k·k)--> UB[k2] --reduce_sum--> S_k --...--> UB[rrms_k] ; UB[k_n]=k ⊙ rrms_k ⊙ w_h
         UB[h_n] ⊙ UB[k_n] --reduce_sum--> S.hk --(·fl(1/√d), vsigmoid)--> α(1,1) --vbrc+vmul(v_ub)--> UB[vhat]
         UB[vhat] --vmul(vhat·vhat)--> UB[v2] --reduce_sum--> S_v --(·1/d +eps, vsqrt, vdiv)--> UB[rrms_v] ; UB[vn]=vhat ⊙ rrms_v ⊙ w_v
GM[conv_state](B,L,d) 第 p 行(p<w-1) --T.copy+vcast--> UB[win_p] ; UB[conv] += UB[cw_p] ⊙ UB[win_p]   (w-1 次)
         UB[conv] += UB[cw_{w-1}] ⊙ UB[vn]
GM[conv_state] --T.copy(区域 [1:L,:])--> UB[st] --T.copy--> GM[new_conv_state[0:L-1,:]] ; UB[vn] --vcast--> GM[new_conv_state[L-1,:]]
         UB[conv] --vsigmoid--> UB[sig] ; UB[y]=UB[conv] ⊙ sig + UB[vhat] --vcast(rint)--> GM[y_t]
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**多步融合类**（矩阵乘 + 3 次行内规约 + 窗口卷积 + 状态递推 + 逐元素链），按算法调研 §2 属**完整调研**：R1 候选表、R2 在线算法、R3 四口径对比、R4 逐候选评估全部展开。**基线与候选同台**：源算法（GEMV 串行向量链 + 逐元素 `T.Parallel`）只是基线候选之一。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3/R4 |
|---|------|------------|-----------|----------|-----------|
| 0 | **基线：源算法** | GEMV = `T.serial(d_mem)`×`T.Parallel(d_padded)` 寄存器链；3× RMSNorm 用 `T.rsqrt`；逐元素 `T.Parallel`；状态左移 `T.serial(L-1)` | 定义式 | 每投影 op 数 = d_mem（串行链） | ✅（基线） |
| 1 | **Cube `T.gemm` GEMV（采纳）** | `k=eW`, `v=eW` 作 M=d_padded / K=d_mem / N=1 的 GEMM，K 分块 + fp32 L0C | 数学等价（仅累加分组变化，§0.6 R1） | op 数 O(d_mem)→O(d_mem/BK)；MAC 落 Cube | ✅（采纳） |
| 2 | **rsqrt 绕法（采纳）** | `rsqrt(s)` → `sqrt(s)/s`（`vsqrt`+`vdiv`） | 实数恒等（s>0）；机器验证 O1 | 消灭 ~2.9e-3 近似误差（3 次 Norm 累积） | ✅（§1.6.1 O1） |
| 3 | **常数倒数折叠（采纳）** | `x/d`→`x·fl(1/d)`；`dot/√d`→`dot·fl(1/√d)`；`x·(1/d)+eps` 合序 | 容差内等价（fp32 ≤1 ulp；D-1 O2） | div → mul（标量 op 型替换） | ✅（§1.6.1 O2） |
| 4 | **CSE 物化 `v_hat`（采纳）** | 物化 `alpha·v` 复用（源步骤 11 重算） | 逐位恒等（D-1 O3） | 少 1 次 `(d,)` vmul | ✅（§1.6.1 O3） |
| 5 | **fp32 中转计算域（采纳 = 保留 + API 强制）** | fp16/bf16 → `vcast(rint)` → fp32 链 → `vcast(rint)` | 与源/契约同域（源即 fp32）；bf16 为 API 强制 | 正确性（bf16 可编译） | ✅（§1.6.1 O4） |
| 6 | **归约合并**（hsq 与 ksq 合入一个 `(2,d)` reduce） | 一次 `reduce_sum` 得两个平方和 | 数学等价 | 4 reduce → 3 | ✅（评估，**不采纳**：省 1 op，却引入 (2,d) 布局与拆分代价） |
| 7 | **状态循环缓冲（不落物化）** | 用环形写指针替代左移 | 数学等价（状态语义不变） | 消 O(L·d) 拷贝 | ❌ 弃选：**输出契约要求物化** `new_conv_state`（manifest 输出 + roofline 计入 `2·B·L·d` 字节） |
| 8 | **conv 转 GEMM/im2col** | 4 tap 的深度卷积展开为 GEMM | 数学等价 | — | ❌ 弃选：这是 **depthwise（逐通道）** 卷积，非通道混合；tap 数 w=4 极小，GEMM 化不成立（无共享权重矩阵） |
| 9 | **sigmoid 稳定化减最大值** | `σ(x)=e^{x-m}/(1+e^{x-m})` | 数学等价 | — | ❌ 不适用：`1/(1+e^{-x})` 在 fp32 全域无上溢（x→−∞ 时 e^{-x}→+∞ → 结果→0，正确）；无需稳定化 |

**R2 在线算法**：**无在线化收益（有明确结论 + 结构依据）**。

- 结构判据：本算子的 4 处 `T.reduce_sum` 分别作用于 **h、k、v_hat 三个不同向量**（不是同一批数据的多遍扫描），每次均为单遍 `(1,d)→(1,1)`；整算子对每个输入张量只读一次（除 `conv_state` 按契约的移位读+写）；**不存在可被"单遍/流式"消去的多遍基线**。
- 跨 step 的**状态递推本身即在线形式**（O(L·d) 状态、每步更新），已是 O(1) 中间状态；源算法已用该形式（`conv_state` 移位缓存），故"在线变体"不适用。
- 依据：源 §0.3 步骤 4/5/6/8 为四个独立行内归约；§0.4 #1 单 kernel 融合已消除跨 kernel 多遍；`examples/TileOPs/tileops/kernels/reduction/` 与 `examples/norm/` 全系为单遍行归约。

**R3 复杂度对比**（四口径；本算子**访存/发射混合受限**，权重访存与向量 op 数均为主导项）：

| 算法候选 | FLOPs（W3: B4,d_mem1024,d512） | 访存量 (Bytes, W3) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------------------------------|-------------------|---------|-------------|---------------------|
| 基线（源：串行 GEMV + `T.Parallel`） | 8.44 M（GEMV 8.39 M）+ **GEMV op 数 2·d_mem=2048** | `2.28 MB`（manifest 口径，W 只读一次；实现上每 batch 重读 W → `B·2MB`） | 1 | fragment 14 个（`(d_padded,)`×5 + `e_local (d_mem,)`×1 + 2D `(1,d)`×4 + 标量 `(1,)`×4，寄存器） | batch 级 `grid=B`（B≤8）；线程级 d 并行 |
| #1 Cube GEMV（**采纳**） | 同基线 FLOPs | 同上 | 1 | L1 `(BK,d_padded)` fp16 = 128KB + L0C `(d_padded,1) f32`=2KB + UB `(d_padded,)`×~10 f32 | 同；Cube 段跨引擎（CV） |
| #2 rsqrt 绕法 | 每个 Norm +1 向量 op（`vdiv`）；`vsqrt` 替代 `vrsqrt` | 不变 | 1 | +1 `(d,)` f32 | 同 |
| #3 常数倒数折叠 | −0（标量 op 型替换） | 不变 | 1 | 不变 | 同 |
| #4 CSE v_hat | −1 向量 op（省重算 `(d,)` vmul） | 不变 | 1 | +1 `(d,)` f32（≤2KB） | 同 |
| #6 归约合并 | −1 reduce op | 不变 | 1 | +(2,d) buffer | 同 |
| #7 环形状态缓冲 | 消 O(L·d) 拷贝 | **改动契约**（无物化输出） | 1 | 不变 | — |
| #8 conv→GEMM | — | — | — | — | ❌ 不适用（depthwise） |

> 口径说明：**FLOPs** 取 manifest roofline `4B·d_mem·d + B(16d+2wd)+20B`（乘加=2）；**访存量** = manifest `bytes`（输入读 + 输出写 + 状态读写 + 权重，**权重按读一次计**），实现中 `grid=B` 会每 batch 重读权重（W3 实际 ≈`B·2MB=8MB`），此偏差在 §1.6.0 D-2 估算中显式计入；**扫描遍数** = 各张量 GM 读取次数 = 1（`conv_state` 读 1 写 1 属契约）；**中间缓冲峰值** 计 UB/L1/L0C 驻留上限（见 §4.5）。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 Cube GEMV（采纳）** | **Cube**（MAC 密集段，`hivm.hir.mmadL1`；GEMV = FLOPs 99.4%）✓ | L1 512KB（`(BK,d_padded)` fp16 128KB，`T.Pipelined` stage=2 → 256KB ✓）；L0C 128KB（`(d_padded,1)` f32=2KB ✓，`CONST-capacity-910B2C` 容量值，**status=verified 但绑 910B2C/9362，本设备 910B3 未重验**） | K 分块 128 整除 d_mem∈{256,512,1024} ✓；M=d_padded≥16 ✓（fp16 分形） | 全参编译期 ✓ | `T.Pipelined` K 分块双缓冲 ✓；CV 融合（Developer auto-split，先例 `flash_attn_npuir_dev.py`）✓ | ✅ **采纳** |
| 基线（源串行 GEMV） | Vector（**MAC 密集段落错单元**）：op 数 2·d_mem=2048（W3）→ 发射下界 ~1.0 ms（`CONST-vector-launch-overhead` 0.5µs/op）vs Cube 路径 ~µs 级 → **重度失分** | UB ✓ | ✓ | ✓ | 串行依赖链阻碍流水 | ❌ 弃选（§0.6 R1） |
| 向量分块 GEMV（备选 B，见 §1.6.3） | Vector（`vbrc`+`vmul`+`reduce`） | 需 `(BK,d_padded)` f32 tile ×2（`w_f`+`e_brc`，BK=8 → 各 16KB @d=512）✓，常驻合计 ≈103KB raw → ×1.7 ≈175KB < 192KB ✓（紧，附编译探针） | ✓ | ✓ | ✓ | ⚠️ 备选（发射项 3·(d_mem/BK)·2 = 768 op（BK=8），见实验裁决） |
| #2 rsqrt 绕法 | Vector（`vsqrt`+`vdiv` 均 fp32 ✓，bf16 ✗ → fp32 域） | +1 `(d,)` f32 | ✓ | ✓ | — | ✅ 采纳 |
| #5 fp32 中转 | Vector（`vcast` f16↔f32/bf16↔f32 ✓） | +fp32 buffer | ✓ | ✓ | — | ✅ 采纳（bf16 API 强制） |
| #6 归约合并 | Vector（`reduce_sum` dim=1 ✓） | +(2,d) f32 | ✓ | ✓ | — | ⚠️ 评估后不采纳（边际） |
| #7 环形状态缓冲 | — | — | — | — | — | ❌ 弃选（契约要求物化） |
| #8 conv→GEMM | — | — | — | — | — | ❌ 不适用（depthwise） |

**v-prefix 元算 dtype 矩阵（决定 fp32 中转 + 排除/保留候选）**：`vadd`/`vmul`/`vdiv`/`vexp`/`vsigmoid`/`vsqrt`/`vrsqrt`/`vmin` **均不含 bf16**（各 op doc §2.2.1：fp16 ✓ / fp32 ✓ / bf16 ✗）；`vbrc` ✓（含 bf16）、`vcast` 支持 `f16→f32`/`bf16→f32`（`rint`）与 `f32→f16/bf16`、`T.gemm` 支持 fp16/bf16/fp32 输入（fp32 累加）。→ 全部向量数学一律 fp32 域，bf16 为 API 强制中转；Cube 段可直接吃 fp16/bf16。

**调研结论**：**选定「Cube `T.gemm` GEMV（K 分块 fp32 L0C）+ fp32 中转计算域 + `vsqrt/vdiv` rsqrt 绕法 + 常数倒数折叠 + CSE 物化 v_hat + 二维区域状态搬运」**（候选 #1/#2/#3/#4/#5 采纳），无可行的更优替代。

关键依据：① **R3/R4**：GEMV 占 FLOPs 99.4%、GM 流量 88% 且为 MAC 密集长依赖串行链（W3: 2·d_mem=2048 op），是唯一需要结构性重设计的段落——Cube 单元是 NPU 唯一的 MAC 密集硬件能力（`hivm.hir.mmadL1`），基线落错单元属重度失分；② **R2**：四个 reduce 作用于三个不同向量、无多遍基线，在线化无收益（结构依据）；③ **R1**：候选 #6/#7/#8/#9 的弃选理由分别为边际收益/契约冲突/数学不适用/无上溢问题；④ 全部采纳项（`T.gemm`、`vcast`、`vsqrt`、`vdiv`、`vmul`、`vbrc`、`vsigmoid`、`reduce_sum`、`copy`）均有本仓 `docs/Tilelang.language/` 佐证 + `examples/` 实跑先例（§3.4）。

**调研范围（「无更优替代」依据）**：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md`（`kb_search "gemv rmsnorm dilated conv decode fusion"` 命中 `CASE-norm-adalayern-migration`、`PL-1.10-loads-first-decoupling`、`TRAP-vrsqrt-plain-precision`）——候选库含 GEMV/RMSNorm 族条目；② 本仓 `examples/`：`gemv/example_gemv.py`（Cube GEMV 四形态）、`engram/engram_decode.py`（同算子 NPU 串行 GEMV 基线）、`norm/rms_norm.py`（3 形态 RMSNorm）、`flash_attention/flash_attn_npuir_dev.py`（Developer+gemm+v-op 混排）、`TileOPs/.../reduction/logsumexp/`（规约族）；③ pattern-library（预注入 + `kb_search` 消费）：`TRAP-vrsqrt-plain-precision`（status=verified，rsqrt 绕法直接依据）、`CASE-norm-adalayern-migration`（status=verified，norm 族迁移 + verify_equiv 范式）、`TRAP-C12-copy-dtype-cast`（status=verified，`T.copy` 隐式跨 dtype 转换——本设计**显式** `vcast`，不用 copy 做 cast）、`TRAP-verf-maclaurin-truncation`（status=verified，"文档语义≠设备实现"——本算子不用 `verf`，但提示 `vsigmoid`/`vexp` 需设备级精度探针）、`BP_redundant_work_or_roundtrip`（status=open，多 reduce/多 pass 冗余——本算子 4 reduce 作用于不同数据、非冗余，已逐项核对）、`CONST-vector-op-cost-910_9362`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-mte2-degradation`（D-2 常数）；④ **外部已知算法族**（模型知识）：GEMV 的 Cube/向量分块/split-K 三类实现、RMSNorm 的 rsqrt/Newton 迭代、因果深度卷积的直接法/环形缓冲——逐项由 §1.6.3 枚举与本表评估；⑤ `docs/Tilelang.language/` 全目录枚举（含 `线性代数操作/T.gemm.md`、`规约操作/`、`数学操作/`、`数据类型转换操作/`、`shape操作/T.vbrc.md`）确认采纳 API 存在性。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)）**：

- **流量项** = `GM bytes ÷ 有效带宽`。manifest bytes（W3）= 2.28 MB；**实现偏差**：`grid=B` 每 batch 重读 `W_K/W_V`（`2·d_mem·d·elem`），W3 实际 GM 流量 ≈ `B·2MB + 状态/小张量 ≈ 8.3MB`。**本设备为 `Ascend910B3`（实查 20 AICore / 40 Vector Core），`pattern-library/constants.md` 无 910B3 条目**，取 `CONST-ascend910-9362-profile`（9362 单 die 核→GM ≈ **1.06 TB/s**，status=verified）作分母并标注**跨 SoC 未重验（未文档化假设 + 估算依据）**：流量项 ≈ `8.3MB / 1.06TB/s ≈ 7.8 µs`。
- **发射项** = `向量 op 数 × ~0.5 µs/op`（`CONST-vector-launch-overhead`，**绑 910B2C 且 `kb_stale_check` 标 stale，910B3 未重验 → 未文档化假设 + 估算依据**）。W3 每 block 向量 op 数 ≈ **50**（1 载入 cast + h/k 各 7 + gate 5 + vhat 4 + conv 8 + state 2 + SiLU 3 + 输出 cast 1 + 其余 copy/标量）：发射项 ≈ **25 µs**。
- **容量项 / Cube 操作数流地板** = W 入 L1（每 block 2MB fp16）÷ L1 端口 `≈148 GB/s/核`（`CONST-L1-port-bw`，status=verified but 910B2C/9362 → `**跨设备未重验**`）：≈ **13.5 µs**（各 block 并行 → wall ≈13.5µs）。容量项（跨引擎 GM 往返）本算子无（中间量全 UB 驻留；`CONST-store-fixpipe-gm-only` 不适用）。

**设计期估算（D-2 roofline）**：代表 workload W3（`B4,d_mem1024,d512`）估算下界 ≈ **max(7.8, 25.0, 13.5) ≈ 25 µs（发射项主导）**；W1（`B1,d_mem256,d256`）≈ max(0.3, ~12, ~3) ≈ **12 µs（发射项主导）**。**不确定性声明**：① 0.5µs/op 与 148GB/s 均绑 910B2C/9362、`kb_stale_check` 标 stale，910B3 未重验，发射项/容量项可能**上偏**；② 对照证据（实测）：同族短链逐元素算子在实测中常由流量项主导、发射项被 MTE2 窗口隐藏（`bitwise_not` 1M 实测 7.72µs ≈ 流量 7.5µs）——本算子发射项或系统性上偏；③ **K 分块大小 BK 与 `a_transpose` 形式是 Stage 4 发射/操作数流调优杠杆**；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（Cube GEMV + fp32 中转 + rsqrt 绕法）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| O1 | **rsqrt 绕法** | `rrms = rsqrt(s)`（设备为近似指令） | `t=sqrt(s)`；`rrms = t/s`（`vsqrt`+`vdiv`） | 实数恒等 `sqrt(s)/s ≡ s^{-1/2}`（s>0；本算子 `s=mean(x²)+eps ≥ eps=1e-6`）；D-1 O1：vs 真值 fp32 最大 abs 差 6.1e-5、`(·)^3` 连乘 max_abs 192（相对量级，违反数 0）→ EQUIV_PASS；**精度优于基线设备指令**（2.9e-3 → 1.07e-7） | 每 Norm 1 op 型替换（`vrsqrt`→`vsqrt`+`vdiv`，+1 op）；消除 3 次 Norm 的 ~2.9e-3 相对误差累积 |
| O2 | **编译期常数倒数折叠** | `x/d`；`dot/sqrt(d)` 逐元素除法 | `x·fl(1/d)`；`dot·fl(1/√d)`（host 期 Python 常量） | 容差内等价（PL-1.5「乘编译期常数倒数」先例）；D-1 O2 四行：`x*fl(1/d)` 与 `x/d` **逐位一致**（d=256/512 为 2 的幂，`fl(1/d)` 精确）；`dot·fl(1/√d)` vs `dot/√d` max_abs 3.9e-3（相对容差内，违反 0）→ EQUIV_PASS | `vdiv` → `vmul`（标量/向量 op 型替换）；`·(1/d)` 与 `+eps` 合成一步；省 1 `vdiv`/Norm |
| O3 | **CSE 物化 `v_hat`** | 步骤 8 原地覆写 `vhat` → 步骤 11 重算 `alpha·v` | 物化 `vhat = alpha·v`（fp32 `(d,)`）供步骤 8 与 11 共用 | 同一确定性 fp32 计算，物化值与重算值**逐位一致**；D-1 O3 三 dtype `max_abs=0.0` → EQUIV_PASS | −1 次 `(d,)` `vmul`；代价 +1 UB buffer（≤2KB / 192KB） |
| O4 | **fp32 中转计算域**（保留源 + API 强制） | 概念上「输入 dtype 域直译」 | `vcast(rint)↑f32 → fp32 向量链 → vcast(rint)↓dtype` | 与源（`accum_dtype="float"`）及契约 golden（`.float()` 全链路）**同域**；bf16 为 API 强制（vadd/vmul/vsigmoid/vsqrt/vmin 无 bf16）；D-1 FULL 三 workload `y`/`st` 均 EQUIV_PASS | 正确性：bf16 可编译；代价 +2 `vcast`/张量 |
| O5 | **GEMV 累加结构重排**（Cube K 分块） | `T.serial(d_mem)` 串行链 fp32 累加 | K 分块 `T.gemm` + fp32 L0C 树形累加 | 数学等价（同 `Σ_i W[i,j]e[i]`），仅分组变化；D-1 O4（B4/B8/B1 × fp16/bf16/fp32）与 fp64 最大 abs 差 ≤ 3.1e-5，违反 0 → EQUIV_PASS | op 数 O(d_mem)→O(d_mem/BK)（W3: 2048→16）；MAC 落 Cube |
| E1 | 向量分块 GEMV（`vbrc`+`vmul`+`reduce`）（**不采纳，备选 B**） | — | 每 BK 块 3 向量 op | 数学等价（分块外积 + 归约）；D-1 O4 已证分块累加等价 | 纯 Vector，无 CV 混排；op 数 3·(d_mem/BK)·2 = **768**（BK=8，受 UB：两张 `(BK,d_padded)` f32 tile，见 §1.6.3）> Cube 的 16；列为**实验裁决备选**（§1.6.3） |
| E2 | 归约合并（hsq+ksq 合一）（**不采纳**） | 2 次 `reduce_sum` | 1 次 `(2,d)` reduce | 数学等价 | 省 1 reduce op，但引入 `(2,d)` 布局与拆分；边际收益 < 实现复杂/UB 成本 |
| E3 | 除法稳定化 / sigmoid 减 max（**不采纳**） | — | — | — | `1/(1+e^{-x})` 在 fp32 全域无上溢（x→−∞ → 结果 0，正确）；无稳定化必要 |

**优化结论**：采纳 **5 项**（O1 rsqrt 绕法、O2 常数倒数折叠、O3 CSE v_hat、O4 fp32 中转、O5 GEMV 结构重排）。**优化后公式**（供 §3.1 使用）：
- 投影：`C_k/C_v = T.gemm(W， e， a_transpose=True, initC=(k==0))`（K 分块 `T.Pipelined`）
- RMSNorm：`s = sumsq·N_inv + eps`；`t = vsqrt(s)`；`rrms = t/s`；`x_n = x·rrms·w`
- 门控：`α = vsigmoid(dot · INVSQRT_D)`；`vhat = α·v`（物化）
- SiLU：`y = c·vsigmoid(c) + vhat`，末端 `vcast(rint)`

公式层面已无进一步计算量/访存量可优化空间：**权重访存为理论下界的一部分**（每 batch 必读 W；唯 split-K/N 跨核可摊分，属结构级改动 → 交 §1.6.3 备选/Stage 4），**中间量全部 UB 驻留**（无中间 GM 往返），**4 个 reduce 非冗余**（不同数据，`BP_redundant_work_or_roundtrip` 逐项核对）。

**等价性机器验证（D-1）**：脚本 `examples/engram_decode/_engram_decode_kernel/verify_equiv.py`（CPU；候选式 = NPU fp32 流水线逐 op 复现；基线式 = fp64 精确数学 / 源式；判定 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`，fp32 `1e-5/1e-5`、fp16 `5e-2/5e-2`、bf16 `1e-1/5e-2`）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**TOTAL 28 ITEMS, 28 EQUIV_PASS, 0 失败项**）：

| 采纳项 / 对照 | 候选式 | 基线式 | dtype | samples | 最大 abs 差 | 违反率 | 违反数 | 结论 |
|---------------|--------|--------|-------|---------|------------|--------|--------|------|
| O1 rsqrt=sqrt/s | `sqrt_f32(s)/s` | `1/sqrt(s) [fp64]` | fp32 | 1,048,576 | 6.104e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O1 (sqrt/s)^3（3-norm 连乘） | `(sqrt_f32(s)/s)^3` | `s^-1.5 [fp64]` | fp32 | 1,048,576 | 1.920e+02 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 x/d → x·fl(1/d) (d=256) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 dot/√d → dot·fl(1/√d) (d=256) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 x/d → x·fl(1/d) (d=512) | `fp32 x*fl(1/d)` | `fp32 x/d` | fp32 | 1,048,576 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O2 dot/√d → dot·fl(1/√d) (d=512) | `fp32 dot*fl(1/√d)` | `fp32 dot/√d` | fp32 | 1,048,576 | 3.906e-03 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE v_hat reuse | `vhat (materialized)` | `alpha*v (recomputed)` | fp32 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE v_hat reuse | `vhat (materialized)` | `alpha*v (recomputed)` | fp16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O3 CSE v_hat reuse | `vhat (materialized)` | `alpha*v (recomputed)` | bf16 | 262,144 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B4,dm1024,d512,bk128) | `fp32 tiled-K matmul` | `fp64 exact` | fp16 | 2,048 | 9.537e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B4,dm1024,d512,bk128) | `fp32 tiled-K matmul` | `fp64 exact` | bf16 | 2,048 | 1.907e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV fp32 (B4,dm1024,d512) | `fp32 tiled-K matmul` | `fp64 exact` | fp32 | 2,048 | 4.501e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B8,dm512,d256,bk128) | `fp32 tiled-K matmul` | `fp64 exact` | fp16 | 2,048 | 3.052e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B8,dm512,d256,bk128) | `fp32 tiled-K matmul` | `fp64 exact` | bf16 | 2,048 | 1.907e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV fp32 (B8,dm512,d256) | `fp32 tiled-K matmul` | `fp64 exact` | fp32 | 2,048 | 4.761e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B1,dm512,d256,bk64) | `fp32 tiled-K matmul` | `fp64 exact` | fp16 | 256 | 1.192e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV tiled-K (B1,dm512,d256,bk64) | `fp32 tiled-K matmul` | `fp64 exact` | bf16 | 256 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 GEMV fp32 (B1,dm512,d256) | `fp32 tiled-K matmul` | `fp64 exact` | fp32 | 256 | 1.993e-07 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL y | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 256 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL st | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 3,072 | 2.441e-04 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL y | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 2,048 | 1.526e-05 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL st | `fp32 NPU pipeline` | `fp64 golden` | fp16 | 40,960 | 7.629e-06 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL y | `fp32 NPU pipeline` | `fp64 golden` | bf16 | 2,048 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O-全链路 FULL st | `fp32 NPU pipeline` | `fp64 golden` | bf16 | 36,864 | 0.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 sumsq fp32-reduce (d=256) | `fp32 sum of squares` | `fp64 sum (→dtype)` | fp16 | 4,096 | 5.000e-01 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 sumsq fp32-reduce (d=256) | `fp32 sum of squares` | `fp64 sum (→dtype)` | bf16 | 4,096 | 3.419e+09 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 sumsq fp32-reduce (d=512) | `fp32 sum of squares` | `fp64 sum (→dtype)` | fp16 | 4,096 | 1.000e+00 | 0.000000 | 0 | **EQUIV_PASS** |
| O5 sumsq fp32-reduce (d=512) | `fp32 sum of squares` | `fp64 sum (→dtype)` | bf16 | 4,096 | 3.419e+09 | 0.000000 | 0 | **EQUIV_PASS** |

> 判定准则 = `|cand−base| ≤ atol + rtol·|base|`，违反数 0 即 PASS；覆盖随机张量 + 角点（rsqrt 域 1e-6~3.4e38 含次正规邻域、有符号全域 ±1e6/±6e4/0/±1e-8、极值 scale）。**注**：`O5 sumsq`/`O1 ^3` 的 max_abs 因 base 量级极大（如 bf16 平方和 ~1e11）而绝对数大，判定以相对容差（`rtol`）为准，违反数为 0。**被拒候选 E1（向量分块 GEMV）的等价性**由 O4 行覆盖（分块累加等价），其弃选理由为发射项（op 数 192 > 16）而非正确性，列为 §1.6.3 备选。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 4 处行内归约（sumsq_h / sumsq_k / dot_hk / sumsq_v） | 源 `hsq_2d` 等 2D fragment + `T.reduce_sum(dim=1)` | `T.reduce_sum` on UB（`(1,d_padded) f32 → (1,1)`，佐证 `docs/.../规约操作/T.reduce_sum.md`；先例 `examples/norm/rms_norm.py` L43） | ✅ | — |
| 2 | 逐元素链（cast/平方/乘/加/`rsqrt`/`sigmoid`） | 源 `T.Parallel(d_padded)` 标量逐元素 | `T.vcast`/`T.vmul`/`T.vadd`/`T.vsqrt`/`T.vdiv`/`T.vsigmoid`（佐证各 op doc §2.4；先例 `examples/engram/engram_decode.py`） | ✅ | — |
| 3 | **状态左移** | 源 `T.serial(max_conv_len-1)` × `T.Parallel(d_padded)` 逐行搬移 | 二维区域 `T.copy`（`T.copy(conv_state[bid,1:L,:]→ub)` + `T.copy(ub→new_conv_state[bid,0:L-1,:])`；佐证 `TRAP-load-nd2nz-strided` 的 slice 形态） | ✅ | — |
| 4 | **GEMV** | 源 `T.serial(d_mem)` × `T.Parallel(d_padded)` FMA 链 | Cube `T.gemm`（K 分块；佐证 `docs/.../线性代数操作/T.gemm.md` + `examples/gemv/example_gemv.py`） | ✅ | — |
| 5 | 标量 `alpha` 广播到 d（`vhat=α·v` / `y=…+vhat`） | 源标量 `alpha` 逐元素乘 | `T.vbrc(α(1,1) → (d_padded,))` + `T.vmul`（佐证 `shape操作/T.vbrc.md` §2.2.2 `(1,1)→(M,N)` 合法；`PL-1.15` 提示列广播有性能税故走 vbrc 展开） | ✅ | — |
| 6 | 标量门控 `σ(dot·INVSQRT_D)` | `dot_hk` 标量式 | `T.vmul`/标量乘 + `T.vsigmoid`（作用 `(1,1)` buffer；佐证 `数学操作/T.sigmoid.md`） | ✅ | — |
| 7 | Cube K 分块循环 | — | `T.Pipelined(ceildiv(d_mem,BK), num_stages=2)` | ❌ | 计算流水循环（非逐元素）；K 分块双缓冲重叠 MTE2/L1，非逐元素热点；边界 `BK | d_mem` 静态 |
| 8 | conv tap 展开（w−1 历史 tap） | 源 `T.serial(w-1)` + 膨胀索引 | 显式展开（w 编译期常量）；**不能转 `T.Parallel`** | ❌ | **编译期展开循环**：tap 行索引 `L-(w-1-p)δ` 为编译期常量，须 Python/`@T.macro` 展开（`TRAP-C13`：`for p in range()` 被解析为运行时 Var → 动态 slice 非法）；w≤4，展开 ≤3 条语句，非逐元素热点 |
| 9 | 核级任务映射 | `T.Kernel(batch)` + `threads` | 一维 `T.Kernel(B, is_npu=True)` | ❌ | block 级任务映射（每 block ≤ 少量标量）；`threads` 在 npuir 无效（`TRAP-threads-kwarg-noop`），无逐元素等价 API；元素级并行由 v-prefix 整 buffer 完成 |
| 10 | host 侧 pad/crop（左 pad 状态、尾维 pad、结果 crop） | `F.pad`/切片 | 保留在 host（元数据/视图级） | ❌ | host 侧 shape/stride 元数据处理（不在 kernel 内）；`ascend-constraints.md` §4 允许 stride/shape 视图操作，禁改输入内容——本算子 host 仅 pad/crop |

**向量化结论**：逐元素/归约/搬运/矩阵计算点（#1–#6）**已全部向量化或交由 Cube**（替代 API 均有 docs 佐证 + `examples/` 实跑先例）；保留 4 处非逐元素循环/标量（#7 计算流水、#8 编译期 tap 展开、#9 核级任务映射、#10 host 元数据），逐项理由如上，无「实现简单」类理由。与 §6 循环结构一致——kernel 内**无逐元素标量循环**；唯一串行结构为 Cube 的 K 分块流水（#7）与编译期 tap 展开（#8）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**候选枚举**（本算子的非 batch 维度 = `d_mem`（投影归约轴）与 `d`（输出/规约轴）；I/O 布局 = 行主序、尾轴 `d` 连续，是契约；核内布局是设计变量）：

| 候选 | 向量化轴 / 布局 | 轴质量（整除/尾 lane 浪费） | repack 代价 | UB 预算 | 结论 |
|------|----------------|---------------------------|-----------|---------|------|
| **A（主选）** | **Cube 段**：M=d_padded / K=d_mem 分块 / N=1；**Vector 段**：向量化轴 = 尾轴 `d`（连续），归约沿 `d` 水平归约（`reduce_sum dim=1`） | d ∈{256,512} 整除向量宽（fp16 ×8 / fp32 ×4）✓；d_padded=d（`d%256==0`）→ 尾 lane 浪费 0% | 无（I/O 布局 = 核内布局） | L1 128KB（BK=128, fp16）+ L0C 2KB + UB ~10×(d,)f32 ≈ 20KB | ✅ **主选** |
| B（备选，实验裁决） | **纯 Vector GEMV**：向量化轴 = 尾轴 `d`（输出维），归约轴 `d_mem` 走「广播+乘+`reduce_sum(dim=0, clear=False)` 累加」，**BK=8**（v1 重算 UB 后下调，见右列） | d ✓；`BK\|d_mem` ✓ | 无 | **重算全常驻 buffer（@d=512, BK=8）**：`w_h (8,512) dtype` 8KB + `w_f (8,512) f32` 16KB + `e_brc (8,512) f32`（经 `T.vmul(w_f,e_brc,e_brc)` **原地复用为乘积**）16KB + `acc_k/acc_v` 4KB + 共享向量链 59KB ≈ **103KB raw → ×1.7 ≈ 175KB < 192KB ✓**（余量 ~17KB，紧；BK=4 回落 ≈141KB） | ⚠️ 备选（§实验裁决；op 数 3·(d_mem/BK)·2 = **768**（W3, BK=8）> Cube 16；正确性兜底非性能竞争者） |
| C（弃选） | 核内 `T.transpose` 转置 W → 使 `d_mem` 成尾轴（水平归约） | — | `T.transpose` 二轴交换链（UB 级，`PL-1.1` 实测 ~5.4µs/链）；W 规模 1024×512 → 多 tile 链 | 转置中间 tile 翻倍 | ❌ 弃选：`T.transpose` **dtype 支持 bf16**（`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 Ascend dtype 行**逐格取值**：int8/int16/int32/int64/fp16/fp32/bf16 = √，uint8/uint16/uint32/bool = ×；§2.3 限制为「仅二轴交换」）——**故 dtype 不构成弃选理由**；真正弃选理由：该候选引入 UB 级二轴交换链 repack（W 规模 1024×512 → 多 tile 链，`PL-1.1` 实测 ~5.4µs/链），而主选 A 的 Cube 已用 `a_transpose` 标志由 load 侧处理转置（零额外 UB、零 repack），候选 C **相对主选无收益** |
| D（弃选） | host permute 转置 W | — | `PL-1.3-host-permute` 实测 106–147µs | — | ❌ 弃选：host permute 净亏（`PL-1.3` status=verified），且改张量内容违反 host 契约 |
| E（弃选） | 垂直扫描归约（lane 映射独立行、串行步进归约维） | — | — | — | ❌ 弃选：本算子归约维 = 尾轴 `d`，水平归约（`reduce_sum` 1 op）优于垂直扫描（O(d) 次串行），且先例 `rms_norm.py` 用水平归约 |

**弃选论证证据**：候选 C 依赖的 `T.transpose` 已枚举并按 `negative-claim-evidence.md` §2.2 在各 `docs/Tilelang.language/` 子目录检索（命中 `创建操作/T.transpose.md`，§2.2.1 Ascend dtype 行**逐格实读**：int8/int16/int32/int64/fp16/fp32/**bf16 = √**、uint8/uint16/uint32/bool = ×；§2.3 限制为仅二轴交换）——**该候选弃选不依赖任何 dtype 负向断言**，仅依据「主选 A 已由 load 侧 `a_transpose` 零 repack 处理转置 → 候选 C 无收益」；候选 D 引用 `PL-1.3-host-permute`（status=verified，106–147µs）；候选 E 为结构分析（水平 vs 垂直，无外部常数依赖）。**A/B/C/D/E 全行负向论断已逐条复核，无「引章节号未逐格取值」形态残留**（ALG-transpose 2026-10-14 更正条目精神，见 VP-2026-0168）。

**实验裁决三件套（判定裕度落在未实证常数内 → 强制）**：

- **裁决对象**：Cube GEMV（主选 A）vs 向量分块 GEMV（备选 B）。判定裕度依赖的未实证常数：910B3 的 L1 端口带宽/向量发射成本/小向量 op 吞吐（`CONST-L1-port-bw`、`CONST-vector-launch-overhead` 均绑 910B2C/9362，`kb_stale_check` 标 stale）。
- **主选方案（默认实现，进 Stage 3）**：候选 A（Cube `T.gemm`，`a_transpose=True`，K 分块 BK=128，`T.Pipelined(…, num_stages=2)`）；结构见 §3.3 / §4 / §6。
- **备选方案（≤1，结构完整，达 Stage 4 可直接实现）**：候选 B（纯 Vector 分块 GEMV；**v1 重算 UB 预算后 BK 由 32 下调为 8，并改为「原地乘积复用」结构**）——
  - **常驻 buffer（全列出，@d=512, BK=8）**：`w_h (BK,d_padded) dtype` = 8×512×2 = **8KB**；`w_f (BK,d_padded) f32`（w 升 f32）= 8×512×4 = **16KB**；`e_brc (BK,d_padded) f32`（e 块广播，经 `T.vmul(w_f, e_brc, e_brc)` **原地复用为乘积 tile**，省掉独立 `prod_ub`）= **16KB**；`acc_k/acc_v (1,d_padded) f32` = 2×2KB = **4KB**；**共享向量链**（17×(d_padded,) f32 = 34KB + ~3×(d_padded,) dtype = 3KB + `st_ub` 19.5KB + 归约 fragment ≈2KB）≈ **59KB**。**raw 合计 ≈ 103KB**，按最坏 ×1.7（`TRAP-UB-multibuffer-inflation`，status=verified）≈ **175KB < 192KB ✓**（余量 ~17KB，**紧**）。若编译探针示溢出，`BK` 回落 4（两张主 tile 减半）→ raw ≈83KB、×1.7 ≈141KB。
  - 循环：`for k0 in T.serial(d_mem // BK)`：`T.copy(W_K[k0*BK:(k0+1)*BK, :], w_h)` → `T.vcast(w_h, w_f, "rint")` → `T.vbrc(e_ub[k0*BK:(k0+1)*BK] as (BK,1), e_brc)` → `T.vmul(w_f, e_brc, e_brc)`（原地乘积）→ `T.reduce_sum(e_brc, acc_k, dim=0, clear=False)`；W_V 同式（落 `acc_v`）；共 `3×(d_mem/BK)×2` 向量 op = **768**（W3, BK=8）；`acc_k/acc_v` 即后续链的 `k_f/v_f`。
  - 分核三要素：grid=B（同主选）；物理核 = 20 AIC（混合算子用返回值）；纯 Vector 变体按标准核数翻倍为 40；B≤8 ≤ 20（或 40）→ 无需适配；
  - 该变体无 Cube 段 → **非 MixCV**（无 CV 同步），代价为 op 数上升（768 vs Cube ~16；W3 发射下界 ~0.38ms vs Cube 路径 µs 级，**性能远劣**）——保留意义为 `a_transpose`/Cube 路径编译失败时的**正确性兜底**，而非性能竞争者。
  - **Stage 4 UB 编译探针计划（强制；B 被裁决选中或以兜底启用时执行）**：① 上靶前先 `tilelang.compile(main, target="npuir")` 编译 smoke 与 W3 两档，读 `ub overflow, requires N bits` 报文取**精确实测需求**（VP-2026-0059：手工逐 buffer 预算 ≠ 可编译证明）；② 若实测需求 > 手工预算 ×1.12（`CONST-capacity-910B2C` 基础分配差；注 `TRAP-UB-multibuffer-inflation` 反例边界记录「显式 alloc 结构实测 0.981×manual」，系数结构依赖），按 `BK=8→4` 或合并 `w_h`/`w_f` 生命周期回收；③ 探针结果回写本节（`perf_feedback.md` 路径）。
- **裁决计划**：代表 shape = 全部 3 个 full workload + smoke；指标 = `msprof op` Task Duration（median-of-20）；未知常数反解 = 用 B 的实测 op 数与 A 的实测时长反推 910B3 的有效 per-op 成本与 L1 端口带宽；**判定阈值**：备选 B latency < 主选 A × (1−0.10) 才翻转（10% 噪声余量，`BP_run_state_bimodality` ±3–5%）；裁决结果回写 §1.6.3（`perf_feedback.md` 路径）。**`a_transpose` 形式 vs M=16 补零形式**亦在同一裁决中 A/B（A1=`a_transpose`，A2=`M=16 补零`），阈值同。
- **shape 特化分派可行性**：工厂 `lru_cache` 按 `(B,d_mem,d,L,w,δ,eps,dtype)` 特化，**支持按 shape 分派主选/备选**（如大 d_mem 走 A、小 shape 走 B），裁决须评估**按 workload 分派**而非全局二选一。

**决策落地约束**：主选方案的向量化轴 = 尾轴 `d`，累加/归约维 = `d`（水平归约）；该轴同步写入 §3.3 伪代码（各 v-op 作用于 `(d_padded,)` buffer）与 §6 循环结构（无逐元素标量循环）。备选 B 的结构设计独立记录于本节（不与主选混排）。

**布局决策结论**：选定**主选 A** —— Cube 段（M=d_padded / K=d_mem 分块 BK=128 / N=1，`a_transpose=True`）+ Vector 段（向量化轴 = 尾轴 `d`，水平归约 `reduce_sum(dim=1)`，核内布局 = I/O 行主序布局，**无 repack**）；向量化轴唯一候选为尾轴 `d`（连续、32B 对齐、整除向量宽），无第二轴可枚举。**弃选候选**：B（纯 Vector 分块 GEMV，列为实验裁决备选）、C（核内 `T.transpose`，dtype 支持 bf16、无 dtype 障碍，但 UB 级 repack 相对主选 A 的 load 侧 `a_transpose` 零收益）、D（host permute，106–147µs 净亏）、E（垂直扫描归约，劣于水平归约）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`ascend_mode="Developer"`，脚手架 wrapper 已固定 `EngramDecodeKernel.ascend_mode = "Developer"`，与 `op_requirements.programming_mode=developer` 一致）。

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | **混合**：GEMV（Cube `T.gemm`，占 FLOPs 99.4%）+ 逐元素/规约向量链（Vector） | 需 Cube（L1/L0C）+ Vector（UB）；Developer 支持 auto CV-split |
| 是否含归约 | 4 次行内 `reduce_sum`（h/k/vhat 不同数据） | Developer 的 `T.reduce_sum` on UB 足够；无需 Expert 手动 buffer 控制 |
| 是否需流水线 | Cube K 分块 `T.Pipelined` 双缓冲；Vector 段线性 | Developer 支持 `T.Pipelined` |
| 向量链 API | `vcast/vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc` | v-prefix |
| 同步需求 | Developer 自动同步（CV auto-split）；无跨核 workspace | Developer 自动同步即可 |
| 是否 persistent | **否**（grid=B≤8） | 规避 `TRAP-DEVMODE-PERSIST-GEMM`（Developer+persistent+gemm 崩溃）；非 persistent Developer+gemm+v-op 有先例 |

决策树路径：`混合（Cube+Vector）→ 非 persistent → Developer auto CV-split → Developer 模式`。**先例**：`examples/flash_attention/flash_attn_npuir_dev.py`（Developer + `T.gemm` + `T.vmul/T.vexp` 混排实跑）、`examples/gemv/example_gemv.py`（Developer + `T.gemm` GEMV）、`examples/norm/rms_norm.py`（Developer + `T.reduce_sum`/`T.vrsqrt`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | Vector 段 `T.alloc_ub((d_padded,), f32/dtype)`；Cube 段 `T.alloc_shared((BK, d_padded), dtype)`（→L1）+ `T.alloc_fragment((d_padded,1), f32)`（→L0C，供 `T.gemm`） |
| 计算方式 | Cube `T.gemm` + v-prefix 整 buffer 向量链 + `T.reduce_sum` |
| 同步 | Developer 自动同步（含 CV auto-split 的 Cube↔Vector 等待） |
| 分核 | 一维 `T.Kernel(B, is_npu=True)`（每 batch 一 block，§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（Cube GEMV + fp32 中转 + rsqrt 绕法 + 常数倒数折叠 + CSE）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `e_sh ← e_t[bid,:]`；`w_sh ← W_K/W_V`（K 分块） | 搬入 Cube 操作数 |
| 1 | `C_k[j,0]=Σ_i W_K[i,j]e[i]`；`C_v[j,0]=Σ_i W_V[i,j]e[i]` | Cube `T.gemm`（K 分块 `T.Pipelined`，fp32 累加） |
| 2 | `k_ub ← C_k[:,0]`（f32）；`v_ub ← C_v[:,0]`（f32） | L0C→UB |
| 3 | `h_f = cast32(h_t[bid,:])` | fp32 中转（fp16/bf16） |
| 4 | `s_h = Σ h_f² · fl(1/d) + eps` | 平方 + 归约 + 常数倒数折叠 |
| 5 | `t_h = sqrt(s_h)`；`rrms_h = t_h / s_h` | rsqrt 绕法 |
| 6 | `h_n = h_f · rrms_h · rms_w_h` | 归一 + 权重（`rrms_h` 经 `vbrc` 广播） |
| 7 | `s_k = Σ k_ub² · fl(1/d) + eps`；`rrms_k = sqrt(s_k)/s_k`；`k_n = k_ub · rrms_k · rms_w_h` | 同 4–6 |
| 8 | `dot = Σ (h_n · k_n)`；`α = sigmoid(dot · fl(1/√d))` | 点积 + 标量门控 |
| 9 | `vhat = α · v_ub`（物化） | CSE |
| 10 | `s_v = Σ vhat² · fl(1/d) + eps`；`rrms_v = sqrt(s_v)/s_v`；`vn = vhat · rrms_v · rms_w_v` | 第 3 个 RMSNorm |
| 11 | `c = Σ_{p=0}^{w-2} cw_p·state[L-(w-1-p)δ,:] (+ cw_{w-1}·vn)` | 膨胀因果卷积（tap 编译期展开） |
| 12 | `ns[0:L-1,:] = state[1:L,:]`；`ns[L-1,:] = cast(vn)` | 状态左移 + append（二维区域 copy） |
| 13 | `y = c · sigmoid(c) + vhat`；`y_t[bid,:] = cast(y)` | SiLU + 残差 + 舍回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入操作数 | `e_sh, w_sh` | `T.copy` | `T.copy(e_t[bid, k0*BK:(k0+1)*BK], e_sh[:,0])`（→ `(BK,1)`）；`T.copy(W_K[k0*BK:(k0+1)*BK, :], w_sh)` | Developer |
| 投影 | `C=eW` | `T.gemm` | `T.gemm(w_sh, e_sh, c_frag, a_transpose=True, initC=(k0==0))`（`docs/.../T.gemm.md` §2.4；先例 `example_gemv.py` L65） | Developer |
| K 分块流水 | — | `T.Pipelined` | `for k0 in T.Pipelined(T.ceildiv(d_mem,BK), num_stages=2)` | Developer |
| L0C→UB | `k_ub` | `T.copy` | `T.copy(c_frag, k_ub2d)` → 取 `[:,0]`；或用 `T.alloc_fragment((d_padded,1))` + `T.copy(c_frag[:,0], k_ub)` | Developer |
| cast 升 fp32 | `h_f` | `T.vcast` | `T.vcast(h_ub, h_f, round_mode="rint")`（bf16→f32 / f16→f32 仅 `rint`） | Developer |
| 平方 | `x²` | `T.vmul` | `T.vmul(x_f, x_f, x2)` | Developer |
| 行归约 | `Σx²` | `T.reduce_sum` | `T.reduce_sum(x2_2d, s_2d, dim=1)`（`(1,d_padded)→(1,1)` f32） | Developer |
| 常数倒数 + eps | `s=S·fl(1/d)+eps` | `T.vmul` / `T.vadd` | `T.vmul(s_2d, N_INV, s_2d)`；`T.vadd(s_2d, EPS, s_2d)`（`N_INV`/`EPS` 为 fp32 标量；标量 src 合法，`T.vmul.md`/`T.vadd.md` §2.1） | Developer |
| sqrt 绕法 | `t=√s` | `T.vsqrt` | `T.vsqrt(s_2d, t_2d)` | Developer |
| 倒数 | `rrms=t/s` | `T.vdiv` | `T.vdiv(t_2d, s_2d, rrms_2d)` | Developer |
| 广播标量 | `rrms→(d,)` | `T.vbrc` | `T.vbrc(rrms_2d, rrms_ub)`（`(1,1)→(d_padded,)`，`T.vbrc.md` §2.2.2） | Developer |
| 归一 | `x·rrms·w` | `T.vmul` | `T.vmul(x_f, rrms_ub, x_f)`；`T.vmul(x_f, w_f, x_f)` | Developer |
| 点积 | `⟨h_n,k_n⟩` | `T.vmul` + `T.reduce_sum` | `T.vmul(h_n,k_n,hk2)`；`T.reduce_sum(hk2_2d, dot_2d, dim=1)` | Developer |
| 门控 | `σ(dot·fl(1/√d))` | `T.vmul` + `T.vsigmoid` | `T.vmul(dot_2d, INVSQRT_D, dot_2d)`；`T.vsigmoid(dot_2d, alpha_2d)`（`T.sigmoid.md` §2.4） | Developer |
| v_hat（CSE） | `α·v` | `T.vbrc` + `T.vmul` | `T.vbrc(alpha_2d, alpha_ub)`；`T.vmul(alpha_ub, v_ub, vhat)` | Developer |
| conv tap（历史 p） | `c += cw_p·state[idx_p]` | `T.copy`+`T.vcast`+`T.vmul`+`T.vadd` | `T.copy(conv_state[bid, L-(w-1-p)*δ, :], win_ub)`；`T.vcast(win_ub, win_f, "rint")`；`T.vmul(cw_f, win_f, tmp)`；`T.vadd(c, tmp, c)`（p 编译期展开，≤3 次） | Developer |
| conv 当前 tap | `c += cw_{w-1}·vn` | `T.vmul` + `T.vadd` | `T.vmul(cw_last_f, vn, tmp)`；`T.vadd(c, tmp, c)` | Developer |
| 状态左移 | `ns[0:L-1]=state[1:L]` | `T.copy` | `T.copy(conv_state[bid, 1:max_conv_len, :], st_ub)`；`T.copy(st_ub, new_conv_state[bid, 0:max_conv_len-1, :])` | Developer |
| 状态 append | `ns[L-1]=vn` | `T.vcast` + `T.copy` | `T.vcast(vn, vn_out, "rint")`；`T.copy(vn_out, new_conv_state[bid, max_conv_len-1, :])` | Developer |
| SiLU | `c·σ(c)` | `T.vsigmoid` + `T.vmul` | `T.vsigmoid(c, sig)`；`T.vmul(c, sig, y)` | Developer |
| 残差 | `y += vhat` | `T.vadd` | `T.vadd(y, vhat, y)` | Developer |
| 舍回 | `cast(y)` | `T.vcast` | `T.vcast(y, y_out, round_mode="rint")`；`T.copy(y_out, y_t[bid,:])` | Developer |

> **注（Cube → Vector 数据流）**：`T.gemm` 输出为 `T.alloc_fragment((d_padded,1), "float32")`；经 `T.copy` 落 UB 供向量链使用。Developer auto CV-split 在 Cube(Vector) 间自动插同步（先例 `flash_attn_npuir_dev.py`）。
> **注（`a_transpose`）**：`T.gemm(w_sh (BK,d_padded), e_sh (BK,1), c_frag, a_transpose=True)` 的有效 A = `(d_padded, BK)`、B = `(BK,1)` → `C[j,0]=Σ_i w[i,j]e[i]`，与源 GEMV 同式。`a_transpose` 是关键假设，列为 §9 风险 R-3 + §1.6.3 实验裁决 A1/A2。
> **注（标量缓冲）**：`N_INV`/`INVSQRT_D`/`EPS` 为 fp32 标量（host 期 Python 常量）；`T.vmul`/`T.vadd` 的 `src2` 支持 scalar（§2.1）。
> **注（不做 cast 的 copy）**：`TRAP-C12-copy-dtype-cast`（status=verified）——`T.copy` 跨 dtype 会隐式 `VCastOp` 且舍入不可控；本设计**显式 `T.vcast`**，`T.copy` 只做同 dtype 搬运。

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数 dtype 直接作为第二位置参数；**v-prefix op 只作用于整 UB buffer**（`PL-1.22`/`PL-1.23`）；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）；conv tap 编译期展开（避免 `TRAP-C13` 动态索引）。

```python
import math
import tilelang
import tilelang.language as T

# Engram decode: 单 token 全融合（迁移 §0.6 重设计后）
@tilelang.jit(out_idx=[8, 9], target="npuir")
def _engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size,
                          dilation, eps, dtype, BK=128):
    f32 = "float32"
    d_padded = ((d + 255) // 256) * 256          # manifest d%256==0 → d_padded==d
    w = conv_kernel_size
    need_cast = (dtype != f32)                    # fp16/bf16 → fp32 中转
    n_inv = 1.0 / float(d)                        # 常数倒数折叠 (O2)
    inv_sqrt_d = 1.0 / math.sqrt(float(d))        # 常数倒数折叠 (O2)
    eps_f = float(eps)

    @T.prim_func
    def main(
        e_t: T.Tensor((batch, d_mem), dtype),
        h_t: T.Tensor((batch, d_padded), dtype),
        conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        W_K: T.Tensor((d_mem, d_padded), dtype),
        W_V: T.Tensor((d_mem, d_padded), dtype),
        rms_w_h: T.Tensor((d_padded,), dtype),
        rms_w_v: T.Tensor((d_padded,), dtype),
        conv_w: T.Tensor((w, d_padded), dtype),
        y_t: T.Tensor((batch, d_padded), dtype),
        new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (bid, _):
            # ---- Cube 段 buffer（顶层无条件分配）----
            e_sh  = T.alloc_shared((BK, 1), dtype)
            w_sh  = T.alloc_shared((BK, d_padded), dtype)
            c_frag = T.alloc_fragment((d_padded, 1), f32)   # → L0C
            k_f   = T.alloc_ub((d_padded,), f32)
            v_f   = T.alloc_ub((d_padded,), f32)
            # ---- Vector 段 buffer（f32 计算域 + dtype 暂存）----
            h_ub  = T.alloc_ub((d_padded,), dtype)
            h_f   = T.alloc_ub((d_padded,), f32)
            k2    = T.alloc_ub((d_padded,), f32)
            v2    = T.alloc_ub((d_padded,), f32)
            hk    = T.alloc_ub((d_padded,), f32)
            h_n   = T.alloc_ub((d_padded,), f32)
            k_n   = T.alloc_ub((d_padded,), f32)
            vhat  = T.alloc_ub((d_padded,), f32)     # CSE 物化 (O3)
            vn    = T.alloc_ub((d_padded,), f32)
            c_out = T.alloc_ub((d_padded,), f32)
            sig   = T.alloc_ub((d_padded,), f32)
            tmp   = T.alloc_ub((d_padded,), f32)
            win   = T.alloc_ub((d_padded,), dtype)
            win_f = T.alloc_ub((d_padded,), f32)
            cwf   = T.alloc_ub((d_padded,), f32)
            wh_f  = T.alloc_ub((d_padded,), f32)
            rrms  = T.alloc_ub((d_padded,), f32)
            st_ub = T.alloc_ub((max_conv_len - 1, d_padded), dtype)
            out_d = T.alloc_ub((d_padded,), dtype)
            s_2d  = T.alloc_fragment((1, 1), f32)    # 归约标量
            t_2d  = T.alloc_fragment((1, 1), f32)
            x2_2d = T.alloc_fragment((1, d_padded), f32)
            red_2d = T.alloc_fragment((1, 1), f32)
            al_2d = T.alloc_fragment((1, 1), f32)

            # ---- 权重权重 weight 载入（rms_w_h/v 一次）[同 dtype copy，无隐式 cast] ----
            T.copy(rms_w_h, wh_f_as_dtype)     # 见下：实际经 vcast 升 f32（略）
            # ---- Step 1-2: Cube GEMV ----
            for k0 in T.Pipelined(T.ceildiv(d_mem, BK), num_stages=2):
                T.copy(e_t[bid, k0 * BK:(k0 + 1) * BK], e_sh[:, 0])
                T.copy(W_K[k0 * BK:(k0 + 1) * BK, :], w_sh)
                T.gemm(w_sh, e_sh, c_frag, a_transpose=True, initC=(k0 == 0))
            T.copy(c_frag[:, 0], k_f)
            for k0 in T.Pipelined(T.ceildiv(d_mem, BK), num_stages=2):
                T.copy(e_t[bid, k0 * BK:(k0 + 1) * BK], e_sh[:, 0])
                T.copy(W_V[k0 * BK:(k0 + 1) * BK, :], w_sh)
                T.gemm(w_sh, e_sh, c_frag, a_transpose=True, initC=(k0 == 0))
            T.copy(c_frag[:, 0], v_f)

            # ---- Step 4-6: RMSNorm(h) ----
            T.copy(h_t[bid, :], h_ub)
            if need_cast:
                T.vcast(h_ub, h_f, round_mode="rint")
            else:
                T.copy(h_ub, h_f)
            T.vmul(h_f, h_f, x2_2d[0:1, :])            # 需 (1,d) 视图；实现见注
            T.reduce_sum(x2_2d, s_2d, dim=1)
            T.vmul(s_2d, n_inv, s_2d)
            T.vadd(s_2d, eps_f, s_2d)
            T.vsqrt(s_2d, t_2d)
            T.vdiv(t_2d, s_2d, red_2d)                 # rrms = sqrt(s)/s  (O1)
            T.vbrc(red_2d, rrms)
            T.vmul(h_f, rrms, h_n)
            T.vmul(h_n, wh_f, h_n)                     # wh_f = cast32(rms_w_h)

            # ---- Step 7: RMSNorm(k)（同式，作用于 k_f）----
            ...  # 与 Step 4-6 同结构，结果落 k_n

            # ---- Step 8: gate ----
            T.vmul(h_n, k_n, hk)
            T.vmul(hk, hk, x2_2d[0:1, :])
            T.reduce_sum(x2_2d, red_2d, dim=1)
            T.vmul(red_2d, inv_sqrt_d, al_2d)
            T.vsigmoid(al_2d, al_2d)

            # ---- Step 9: v_hat = alpha * v (CSE) ----
            T.vbrc(al_2d, rrms)
            T.vmul(rrms, v_f, vhat)

            # ---- Step 10: RMSNorm(v_hat) ----
            ...  # 同 Step 4-6，结果落 vn

            # ---- Step 11: dilated causal conv（tap 编译期展开）----
            # p=0 历史 tap：先载入 state 行与 conv_w[0] 并升 f32，再初始化 c_out = cw_0 ⊙ win_0
            T.copy(conv_state[bid, max_conv_len - (w - 1) * dilation, :], win)
            T.vcast(win, win_f, "rint")
            T.copy(conv_w[0, :], cw_dtype)
            T.vcast(cw_dtype, cwf, "rint")
            T.vmul(cwf, win_f, c_out)                  # p=0 初始化（载入之后）
            for p in range(1, w - 1):                  # Python 层展开（w 编译期常量）
                idx = max_conv_len - (w - 1 - p) * dilation
                T.copy(conv_state[bid, idx, :], win)
                T.vcast(win, win_f, "rint")
                T.copy(conv_w[p, :], cw_dtype)
                T.vcast(cw_dtype, cwf, "rint")
                T.vmul(cwf, win_f, tmp)
                T.vadd(c_out, tmp, c_out)
            # 当前 tap (p=w-1)：用 vn（不读 state 行）
            T.copy(conv_w[w - 1, :], cw_dtype)
            T.vcast(cw_dtype, cwf, "rint")
            T.vmul(cwf, vn, tmp)
            T.vadd(c_out, tmp, c_out)

            # ---- Step 12: 状态左移 + append（二维区域 copy）----
            T.copy(conv_state[bid, 1:max_conv_len, :], st_ub)
            T.copy(st_ub, new_conv_state[bid, 0:max_conv_len - 1, :])
            T.vcast(vn, out_d, "rint")
            T.copy(out_d, new_conv_state[bid, max_conv_len - 1, :])

            # ---- Step 13: SiLU + 残差 + 舍回 ----
            T.vsigmoid(c_out, sig)
            T.vmul(c_out, sig, c_out)
            T.vadd(c_out, vhat, c_out)
            T.vcast(c_out, out_d, "rint")
            T.copy(out_d, y_t[bid, :])

    return main
```

> **伪代码说明**：① **`for p in range(1, w-1)` 是 host 层 Python 展开**（工厂函数体内、`@T.prim_func` trace 之外——若写在 `prim_func` 内会成为运行时 Var，见 `TRAP-C13`）；实现时建议用 `@T.macro` 链逐级展开以保证 tap 索引为编译期常量。② Cube 段两次 `T.Pipelined`（K/V 各一），共享 `e_sh`/`w_sh`/`c_frag`（`initC=(k0==0)` 重置累加）。③ 所有 v-op 作用于**整 buffer**；`[...]` 仅出现在 `T.copy` 的 slice。④ `x2_2d` 与 `(d_padded,)` buffer 的视图/别名在实现时用独立的 `(1,d_padded)` buffer（伪代码为压缩表达），Stage 3 以实际 buffer 形状为准。⑤ 权重 `rms_w_h/rms_w_v/conv_w` 先 `vcast` 升 fp32（`wh_f`/`cwf`）后参与 fp32 链。⑥ `out_d` 复用为输出暂存（fp16/bf16）。

### 3.4 API 可行性确认

| API | 来源确认（本地） | 状态 |
|-----|-----------------|------|
| `T.gemm(src1, src2, dst, initC, a_transpose, b_transpose)` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.3/§2.4（`hivm.hir.mmadL1`；fp16/bf16/fp32 输入、fp32 累加；`src1` 可为 `[b,a]` 转置形）；`examples/gemv/example_gemv.py` L65（Developer `is_npu=True` + `T.gemm` GEMV 实跑）；`examples/flash_attention/flash_attn_npuir_dev.py` L59/81（Developer + `T.gemm` + v-op） | ✅ 存在（`a_transpose=True` 为待设备验证点，§9 R-3） |
| `T.vcast(src, dst, round_mode)` | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32`/`bf16→f32` 仅 `rint`；`f32→f16/bf16` 全模式） | ✅ |
| `T.vmul` / `T.vadd(src1, src2, dst)`（src2 可 scalar） | `docs/.../数学操作/T.vmul.md`·`T.vadd.md` §2.1（fp16/fp32 ✓，bf16 ✗）；`examples/engram/engram_decode.py`、`examples/norm/rms_norm.py` 实际调用 | ✅ |
| `T.vsqrt` / `T.vdiv(src, dst)` | `docs/.../数学操作/T.vsqrt.md`·`T.vdiv.md` §2.2.1（fp16/fp32 ✓，bf16 ✗） | ✅ |
| `T.vsigmoid(src, dst)` | `docs/.../数学操作/T.sigmoid.md` §2.4（`T.vsigmoid` 向量算子；降为 VAdd/VSub/VExp/VDiv） | ✅ |
| `T.vbrc(src, dst)` | `docs/.../shape操作/T.vbrc.md` §2.2.2（`(1,1)→(M,N)` 合法） | ✅ |
| `T.reduce_sum(buffer, out, dim, clear)` | `docs/.../规约操作/T.reduce_sum.md` §2.1/§2.4（`(M,N)→(M,1)`；f32 ✓）；`examples/norm/rms_norm.py` L43 | ✅ |
| `T.copy`（slice 区域） | `docs/.../内存操作/T.copy.md`；`TRAP-load-nd2nz-strided`（slice 形态为合法跨步装载）；`examples/norm/rms_norm.py` | ✅ |
| `T.alloc_ub` / `T.alloc_shared` / `T.alloc_fragment` | `docs/.../内存操作/T.alloc_ub.md`·`T.alloc_shared.md`；`examples/gemv/example_gemv.py`（`alloc_shared`+`T.gemm`） | ✅ |
| `T.Pipelined(n, num_stages)` | `docs/.../T.gemm.md` §2.4 示例；`examples/gemv` 变体 | ✅ |
| `T.Kernel(n, is_npu=True)` | `examples/gemv/example_gemv.py`、`examples/flash_attention/flash_attn_npuir_dev.py` | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(batch, is_npu=True)` |
| GPU 专用 API（`threads=`、`T.Parallel` 线程映射、`T.alloc_fragment` 寄存器模型） | Yes | `threads` 移除（K9）；元素级 `T.Parallel` → v-prefix 整 buffer（§1.6.2）；fragment 仅作 Cube 累加器（§0.6 R3） |
| GEMM 非整除 | No（K 分块整除；见 §5.4） | `BK=128 | d_mem`（d_mem∈{256,512,1024}）；保留 `tail` 守卫以支持通用 d_mem |
| L0C 溢出 | Yes（Cube 输出） | `(d_padded,1)` f32 = 2KB ≪ 128KB ✓（`CONST-capacity-910B2C`） |
| L1 容量 | Yes | `(BK,d_padded)` fp16 = 128KB × 2 stage = 256KB < 512KB ✓（§4.5） |
| 物理核数限制 | Yes（grid=B≤8） | 逻辑核数 B ≤ 物理核 20 → 无需适配（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes（W4 为 bf16） | bf16 全链 fp32 中转（`vcast`；Cube 段可直接吃 bf16）（§0.6 R2） |
| 条件 `T.alloc` 非法（`TRAP-conditional-alloc`） | Yes | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| `T.copy` 隐式跨 dtype cast（`TRAP-C12`） | Yes | `T.copy` 仅同 dtype；cast 走显式 `T.vcast`（§3.2 注） |
| v-op 动态 slice 不可用 | Yes | v-op 作用于整 buffer；slice 仅在 `T.copy`（§3.3） |
| `for p in range()` 在 prim_func 内成运行时 Var（`TRAP-C13`） | Yes（conv tap 展开） | tap 在 host 层 Python 展开 / `@T.macro` 链（§3.3 注） |
| `T.vrsqrt` 近似指令（`TRAP-vrsqrt-plain-precision`） | Yes（3 次 RMSNorm） | `vsqrt+vdiv` 绕法（§1.6.1 O1） |
| Developer+persistent+gemm 崩溃（`TRAP-DEVMODE-PERSIST-GEMM`） | No | 本 kernel **非 persistent**（grid=B≤8），不触发；先例 `flash_attn_npuir_dev.py` |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[8,9], compile_flags=[...])` | `@tilelang.jit(out_idx=[8,9], target="npuir")` | K1 加 `target="npuir"`，去 compile_flags |
| Kernel 声明 | `T.Kernel(batch, threads=t)` | `T.Kernel(batch, is_npu=True)` | K2/K3 去 `threads=`、加 `is_npu=True` |
| 并行模型 | `T.Parallel(d_padded)` 线程级 + `threads` autotune | v-prefix 整 buffer + 一维核 | K9/K11 |
| **GEMV** | `T.serial(d_mem)`×`T.Parallel(d_padded)` 寄存器 FMA 链 | Cube `T.gemm`（K 分块 `T.Pipelined`） | §0.6 R1 |
| **计算域** | fp32 累加（`accum_dtype="float"`） | **同 fp32**（bf16 为 API 强制） | §0.6 R2 |
| RMSNorm 倒数 | `T.rsqrt` | `T.vsqrt`+`T.vdiv`（绕法） | §1.6.1 O1 |
| 状态左移 | `T.serial(max_conv_len-1)`×`T.Parallel` | 二维区域 `T.copy` | §0.6 R5 |
| Buffer 层级 | `T.alloc_fragment` | `T.alloc_ub`（Vector）+ `T.alloc_shared`/fragment（Cube） | §0.6 R3 |
| `threads`/autotune | `default_config={"threads":128}`、`autotune_configs` | 无（K8/K9） | 移除 |
| 256 对齐 pad | host `F.pad` + 编译期 `d_padded` | 保留逻辑（`d%256==0` 不触发） | §0.5 #6 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/engram/engram_decode.py` | **最高（同算子）** | NPU `target="npuir"`/`is_npu=True`/`T.alloc_shared` 版全流水线；`T.vsigmoid`/`T.vsqrt`/`T.vdiv` 用法；其串行 GEMV 与 `T.vrsqrt` 是本设计的改进点 |
| `examples/gemv/example_gemv.py` | 高（Cube GEMV） | `T.gemm(weights, vec, out, initC)` 的 M=输出维/N=1 合同；`is_npu=True` Developer |
| `examples/flash_attention/flash_attn_npuir_dev.py` | 高（Developer+gemm+v-op） | Developer 下 `T.gemm` 与 `T.vmul/T.vexp` 混排、auto CV-split |
| `examples/norm/rms_norm.py` | 高（RMSNorm） | `T.alloc_shared`/`T.alloc_fragment`/`T.reduce_sum(dim=1)`/`T.vrsqrt` 三形态 |
| `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md` | 高（norm 族迁移） | vrsqrt 陷阱与绕法、verify_equiv 范式、迁移文档结构 |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/...` | 中 | Developer/Expert 混排与 CV 同步先例 |
| `docs/Tilelang.language/线性代数操作/T.gemm.md`·`数学操作/`·`规约操作/`·`数据类型转换操作/`·`shape操作/T.vbrc.md` | API | 各 API 规格 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `e_t` | `(B, d_mem)` | `float16 \| bfloat16` | N-gram 聚合嵌入 |
| `h_t` | `(B, d_padded)` | same | 隐状态（host 侧 pad；`d_padded==d`） |
| `conv_state` | `(B, max_conv_len, d_padded)` | same | 历史 `v_n` 缓存（左零填充） |
| `W_K` / `W_V` | `(d_mem, d_padded)` | same | 投影权重 |
| `rms_w_h` / `rms_w_v` | `(d_padded,)` | same | RMSNorm 权重 |
| `conv_w` | `(w, d_padded)` | same | 深度卷积权重 |

### 4.2 输出张量

| 参数名 | Shape（kernel） | dtype | 说明 |
|--------|-----------------|-------|------|
| `y_t` | `(B, d_padded)` | `same_as(e_t)` | 残差输出（host crop 到 d） |
| `new_conv_state` | `(B, max_conv_len, d_padded)` | `same_as(e_t)` | 更新后状态（host crop 到 d） |

**对外契约 shape**：`y_t:(B,d)`、`new_conv_state:(B,max_conv_len,d)`（与 `op_requirements.source_output_shape` 一致）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `e_sh` | `(BK, 1)` | dtype | L1 | Cube 操作数（e 的 K 分块列向量） |
| `w_sh` | `(BK, d_padded)` | dtype | L1 | Cube 操作数（W 的 K 分块行） |
| `c_frag` | `(d_padded, 1)` | f32 | L0C | Cube fp32 累加器 |
| `k_f` / `v_f` | `(d_padded,)` | f32 | UB | 投影结果 |
| `h_ub` | `(d_padded,)` | dtype | UB | h 载入暂存 |
| `h_f` / `h_n` / `k_n` | `(d_padded,)` | f32 | UB | 归一前/后向量 |
| `k2` / `v2` / `hk` | `(d_padded,)` | f32 | UB | 平方/点积中间量 |
| `vhat` / `vn` | `(d_padded,)` | f32 | UB | v_hat（CSE 物化）/ v_n |
| `c_out` / `sig` / `tmp` | `(d_padded,)` | f32 | UB | 卷积累加 / sigmoid / 临时 |
| `win` / `win_f` | `(d_padded,)` | dtype / f32 | UB | conv 窗口 tap（原始/升 f32） |
| `cwf` / `wh_f` | `(d_padded,)` | f32 | UB | conv_w / rms_w 升 f32 |
| `rrms` | `(d_padded,)` | f32 | UB | 广播后的 rrms/alpha 标量 |
| `st_ub` | `(max_conv_len-1, d_padded)` | dtype | UB | 状态左移中转 |
| `out_d` | `(d_padded,)` | dtype | UB | 输出舍回暂存 |
| `x2_2d` / `red_2d` / `s_2d` / `t_2d` / `al_2d` | `(1, d_padded)` / `(1,1)` | f32 | UB/fragment | 归约输入 / 标量结果 |

### 4.4 内存搬运路径

```
Cube 路径（GM → L1 → L0A/L0B → L0C）:
GM[W_K/W_V] --T.copy--> L1[w_sh (BK,d_padded)] --load_nd2nz(框架)--> L0A/L0B --T.gemm--> L0C[c_frag f32] --T.copy--> UB[k_f/v_f]
GM[e_t]     --T.copy--> L1[e_sh (BK,1)]  ──────────────────────────────────────────┘

Vector 路径（GM → UB → GM）:
GM[h_t/conv_state/conv_w/rms_w] --T.copy--> UB[win/h_ub/...] --vcast(rint)--> UB[f32]
   --[fp32 向量链: vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc/reduce_sum]--> UB[f32]
   --vcast(rint)--> UB[dtype] --T.copy--> GM[y_t / new_conv_state]
```

### 4.5 UB 内存预算

| 类别 | W1 (d=256) | W3 (d=512) | ×1.7（`TRAP-UB-multibuffer-inflation`） |
|------|-----------|-----------|------------------------------------------|
| Vector f32 buffer（**17** × `(d_padded,)` × 4B；与 §3.3/§4.3 声明一致：`k_f/v_f`、`h_f/h_n/k_n`、`k2/v2/hk`、`vhat/vn`、`c_out/sig/tmp`、`win_f`、`cwf/wh_f`、`rrms`） | 17×256×4 = 17 KB | 17×512×4 = 34 KB | — |
| Vector dtype buffer（~4 × `(d_padded,)` × 2B） | 2 KB | 4 KB | — |
| `st_ub` `(L-1, d_padded)` dtype | 8×256×2 = 4 KB | 19×512×2 = 19.5 KB | — |
| 归约 fragment（`(1,d_padded)` f32 + 4×(1,1)） | ~1 KB | ~2 KB | — |
| **UB 合计** | **≈ 24 KB** | **≈ 59 KB** | **≈ 101 KB < 192 KB ✓** |

**L1**：`w_sh (BK,d_padded) fp16` × `num_stages=2` = 128KB×2 = 256KB（W3）；`e_sh` 256B → **≈ 256KB < 512KB ✓**（`CONST-capacity-910B2C`）。
**L0C**：`c_frag (d_padded,1) f32` = 2KB（W3）≪ 128KB ✓。
> **膨胀系数注（`CG-2026-0008`）**：显式 alloc 结构实测 ~12% 余量即足（正负两向偏差均无文档），按最坏上偏 1.7× 保守核对。`BK` 放大（→256）会线性放大 L1 占用（256KB×2=512KB 触顶），故默认 BK=128。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | 全参数编译期常量（`lru_cache` 特化）；`conv_state` 运行时长度 `L` 由 host 左 pad 吸收 |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[8, 9],          # y_t, new_conv_state 为 prim_func 第 9/10 个参数（idx 8/9）
    target="npuir",
)
def _engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size,
                          dilation, eps, dtype, BK=128):
    ...
```

> **wrapper 契约**：脚手架 wrapper `engram_decode.py` 的 `EngramDecodeKernel.forward` 调用 `_engram_decode_kernel(...)(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)`（无 `threads`，K9），并依赖 kernel 的 `out_idx=[8,9]` 自动分配两输出；`register_fake` 返回 `(B,d_padded)`/`(B,max_conv_len,d_padded)`。工厂签名保持 `(batch,d_mem,d,max_conv_len,conv_kernel_size,dilation,eps,dtype)`（新增可选 `BK`）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube `T.gemm` GEMV + Vector 归约/逐元素链）。

**判定依据**: 含矩阵乘（`T.gemm`，占 FLOPs 99.4%）→ Cube（L1/L0A/L0B/L0C）；含 3 次行内 RMSNorm 归约 + 逐元素链 + 深度卷积 + 状态搬运 → Vector（UB）。按 AGENTS.md「Developer-mode MixCV trigger rule」（Cube 侧 `T.gemm` + Vector 侧 v-prefix op）本算子为 **MixCV**（Developer auto CV-split）。

### 5.2 Block 划分

```python
grid = batch                       # 每 batch 一 block（一维 T.Kernel）
BK   = 128                         # Cube K 分块（d_mem 方向）
# d 不划分：d_padded ∈ {256,512}，(d_padded,) 向量与 (d_padded,1) 累加器均单块驻留
```

**block/BK 选择理由**：
1. grid = batch 与源一致（`op_requirements`：每 batch 一 block），且 batch 间完全独立（无跨 batch 依赖）；
2. `BK=128`：L1 占用 `(128, d_padded) fp16 × 2 stage` = 256KB（W3）< 512KB；`BK | d_mem`（256/512/1024 均整除）；分形 K≥32 ✓；
3. `d` 不划分：`d_padded ≤ 512`，单个 `(d_padded,)` f32 buffer ≤ 2KB，全部向量链可整块驻留 UB，无需 d 方向 tiling（避免跨核归约）；
4. **分核参数（BK/grid）为 Stage 4 调优杠杆**（见 §1.6.3 实验裁决）。

### 5.3 约束分析

- **对齐约束**：`(d_padded,)` f32 buffer = 512×4=2048B（32B 整数倍 ✓）；`(BK,d_padded)` fp16 = 128×512×2=128KB；向量宽 fp32 ×4（512 整除）、fp16 ×8（512 整除）→ 尾 lane 浪费 0%。
- **UB 容量**：≈59 KB（W3），×1.7 ≈ 101 KB < 192 KB ✓（§4.5）。
- **L1 容量**：≈256KB（W3，BK=128×2 stage）< 512KB ✓。
- **L0C 容量**：2KB ≪ 128KB ✓。
- **分形限制**：M=d_padded≥16 ✓、N=1（`example_gemv.py` 已验证 N=1 合法）、K=BK=128≥32 ✓。
- **非整除**：`d%256==0`（manifest）→ 无尾块；`BK|d_mem` → 无 K 尾块。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：
- `d`：manifest `shape_rules: d%256==0` → `d_padded==d`，所有 workload（d=256/512）均 32B 对齐，**无尾块**；保留 `d_padded=_align_up(d,256)` 逻辑以支持通用 d（若未来 d 非 256 倍数，host pad + 输出 crop，wrapper 已实现）。
- `d_mem`：W1/W2 d_mem=256/512、W3=1024、W4=512，均整除 `BK=128` → **无 K 尾块**；为通用性，K 分块以 `tail = T.min(BK, d_mem-k0·BK)` 裁剪 `T.copy`（仅 `T.copy` 用动态 slice，`T.gemm` 用静态 `BK`；若尾块非 BK，退化为 `T.ceildiv` 静态上界 + 尾块补 0 或走 §1.6.3 备选 B）。
- `max_conv_len`：编译期常量，无尾块；`conv_state` 运行时较短长度由 host 左 pad 吸收。
- **不做 host padding 改内容**（`ascend-constraints.md` §4）；非整除一律 kernel 内 `tail` 裁剪或 host 元数据 pad（wrapper 既有）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子输出网格为 batch 轴（每 block 处理一个 batch 的全流水线，d 轴不划分）：

```
num_logical_kernels = ceildiv(B, 1) = B          # M 轴=batch，N 轴=1（d 不划分），故 = B
```

| workload | B | block | 逻辑核数 |
|----------|---|-------|----------|
| decode-smoke-b1-dmem256-d256-fp16 | 1 | 1 batch/block | 1 |
| decode-b1-dmem512-d256-fp16 | 1 | 1 batch/block | 1 |
| decode-b4-dmem1024-d512-fp16 | 4 | 1 batch/block | 4 |
| decode-b8-dmem512-d256-bf16 | 8 | 1 batch/block | 8 |

**② 物理核数依据（实查）**：目标设备 `Ascend910B3`，实查代码与返回值：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
NPUUtils.get().get_arch()              # 实际返回 "Ascend910B3"
```

→ **本算子为混合（Cube+Vector）算子，物理核数取 AI Core 返回值 = 20**（`core-split-strategy.md` §1 要素②：「Cube/混合算子直接使用返回值」；纯 Vector 才翻倍为 40）。实查于 2026-10-14 本会话（npu-smi 25.5.0 显示 `910B3`，NPU 2）。**注**：`pattern-library/constants.md` 的 `CONST-aicore-910B2C`=24 / `CONST-aicore-910_9362`=20 均非本设备，**以实查为准**。

**③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数** —— `B ∈ {1,4,8} ≤ 20`（最大 workload B=8），且 B 由模型/请求决定、不可通过分块调整。

- **结论：无需适配**（标准 §1 要素③第一分支）。grid = B 直接下发，无核内 `T.serial` 串行、无多波次串行调度、无残留负载不均。
- **利用率说明（显式记录）**：B≤8 时仅 8/20 核工作（W1 仅 1/20）——这是 decode 算子 batch 维天然小的固有特征（低并行度、延迟导向）。**更高并行度的结构候选**（split-N/split-K 跨核 + 跨核归约）已在 §1.6.3 评估：split-N 需跨核 RMSNorm 归约（`sync_block_set/wait` + workspace），结构复杂度显著上升，且 B 小、单核工作量小，**列为 Stage 4 结构级备选**（不改变本设计的分核方案）。
- **模式先例**：源算子 grid=(B,)（`EngramDecodeKernel`，一维）；NPU 侧 `T.Kernel(batch, is_npu=True)`（`examples/engram/engram_decode.py` L39）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化或交 Cube——本表循环仅指 Cube K 分块、编译期 tap 展开与核级任务映射；kernel 内**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务） | block 级并行 | `T.Kernel(batch, is_npu=True)` | B 个物理核各处理一个 batch 的全流水线（§5.5：B≤20，无需适配） |
| Cube K 分块 | 流水迭代 | `T.Pipelined(ceildiv(d_mem,BK), num_stages=2)` | K 分块双缓冲重叠 MTE2/L1（K/V 各一次）；边界静态 |
| conv tap | **编译期展开**（host 层 Python / `@T.macro` 链） | `for p in range(1, w-1)`（host 层） | `w` 编译期常量，tap 行索引 `L-(w-1-p)δ` 须为编译期常量（`TRAP-C13`）；≤3 条语句 |
| 元素级 | 向量化 | `T.gemm` + `vcast/vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc/reduce_sum` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |
| 状态搬运 | 向量化 | `T.copy`（二维区域） | 一次搬移 `(L-1, d_padded)`，替代源 `T.serial(L-1)` |

**T.Persistent 选择说明**：不使用 `T.Persistent`。grid=B≤20 已固定 ≤ 物理核数，无多波次；且 persistent+gemm 在 Developer 模式崩溃（`TRAP-DEVMODE-PERSIST-GEMM`），本设计显式规避。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(batch, is_npu=True)` + 两次 `for k0 in T.Pipelined(...)` + host 层 tap 展开；循环体内为 Cube `T.gemm` 或 v-prefix 整 buffer 向量链）。

### 6.3 流水线优化

**当前设计**：Cube 段用 `T.Pipelined(..., num_stages=2)` 对 K 分块双缓冲（`w_sh`/`e_sh` 交替），重叠 GM→L1（MTE2）与 L1→L0 计算（MTE1/Cube）。Vector 段为线性数据依赖链，跨张量的重叠由 Developer 模式 auto-multi-buffer 处理。**设计阶段保持简洁**；若 Stage 4 需进一步优化（提高 `num_stages`、`BK` 放大、CV 段间流水），按 §1.6.3 裁决计划与调优流程评估（`num_stages` 受 L1 512KB 约束，BK=128 时 ≤4 stage）。

### 6.4 尾块处理

见 §5.4。manifest 全部 workload 无尾块（`d%256==0`、`BK|d_mem`、`max_conv_len` 编译期常量）；保留 K 尾块 `tail=T.min(BK, d_mem-k0·BK)` 与 d 非 256 倍数时的 host pad/crop（wrapper 既有）以支持通用 shape。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）。

### 7.2 同步点说明

Developer 模式下编译器自动插入同步（含 Cube↔Vector 的 auto-split 等待）。核内数据依赖链：

```
[Cube] T.copy(GM→L1) → T.gemm(L1×L0 → L0C) → T.copy(L0C→UB)
[Vector] T.copy(GM→UB) → vcast↑ → [v-prefix 链 + reduce_sum] → vcast↓ → T.copy(UB→GM)
[Cube→Vector] L0C→UB 后向量链读 k_f/v_f  ← auto CV-split 插同步
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| GM→L1 (`T.copy`) | 自动 | 编译器插 MTE2→MTE1 等待 |
| `T.gemm` L1→L0C | 自动 | MTE1→Cube |
| L0C→UB (`T.copy`) → 向量链 | 自动（CV auto-split） | Cube→Vector 跨引擎等待 |
| 向量链内同 buffer 原地依赖 | 自动 | V→V barrier |
| UB→GM (`T.copy`) | 自动 | 编译器插 V→MTE3 等待 |
| `T.Pipelined` K 迭代间 | 自动 | 双缓冲跨 stage 依赖 |

无手动 `T.sync_block_set/wait`（无跨核/跨 block workspace 协作）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs（auto CV-split 开启）
```

### 7.4 CV 融合设计

**本算子为 MixCV**（Cube `T.gemm` + Vector v-prefix op，触发 AGENTS.md MixCV 规则）。融合形态：**同一 kernel 内 Cube 段（投影）与 Vector 段（RMSNorm/门控/卷积/SiLU）串接**，经 L0C→UB 直连传递（`CONST-store-fixpipe-gm-only`：store_fixpipe 仅 L0C→GM，故用 `T.copy` 走 L0C→UB）——**无中间结果落 GM**。同步由 Developer auto CV-split 自动插入（先例 `examples/flash_attention/flash_attn_npuir_dev.py`：Developer + `T.gemm` + `T.vmul/T.vexp` 混排、无显式 sync）。**非 persistent**，规避 `TRAP-DEVMODE-PERSIST-GEMM`。无跨核 workspace、无 `sync_block_set/wait`。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现——**直接移植源仓参考实现** `examples/TileOPs/tests/ops/test_engram_decode.py::engram_decode_step_torch`（L38–L86），**不得复刻 §0.6 的 NPU 结构**（保证验证独立性）。

```python
import math
import torch
import torch.nn.functional as F


def _rmsnorm_decode(x, w, eps=1e-6):
    # 移植自 test_engram_decode.py L32-L35（x: (B,1,d), w: (d,)）
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def golden_engram_decode(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                         max_conv_len, dilation, eps=1e-6):
    """Engram decode 单步参考实现（源语义；移植 test_engram_decode.py::engram_decode_step_torch）。

    契约：全部 fp32 计算（`.float()`），末端 `.to(h_t.dtype)` 单次舍回。
    与 §0.6 的 NPU 结构（Cube GEMV / vsqrt+vdiv / 二维区域 copy）无关——独立验证。
    """
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()                      # (B, d)
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1); k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    if max_conv_len > L:
        new_conv_state = torch.cat([conv_state, v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)
    else:
        new_conv_state = torch.cat([conv_state[:, 1:, :],
                                    v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state
```

> 备注：golden **独立于** NPU 流水线（直接调用 PyTorch `@`/`mean`/`rsqrt`/`sigmoid`/`silu`，不复用 §3.3 的 op 序列），满足验证独立性；输入生成参考 `examples/TileOPs/tileops/workloads/sequence_modeling.py`（`EngramDecodeWorkload`）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float16 | 5e-2 | 5e-2 | 来源 `examples/TileOPs/tests/ops/test_engram_decode.py` L117–L118 |
| bfloat16 | 1e-1 | 5e-2 | 来源 同上（bf16 精度低，容忍度放宽） |

**manifest roofline（供对账）**：`flops = 4·B·d_mem·d + B(16d + 2wd) + 20B`；`bytes = (B·d_mem + B·d + 2·B·L·d + 2·d_mem·d + 2d + w·d + B·d)·elem`。

**误差预算核对**：`verify_equiv.py` 实测——O1（rsqrt 绕法）vs fp64 真值最大 abs 差 6.1e-5（fp32 容差内）；O2/O3 逐位一致；O5（Cube 分块累加）fp16 舍回后 ≤3.1e-5；**全链路三 workload（B1/B4 fp16、B8 bf16）`y`/`st` 与 fp64 golden 均违反数 0**（28/28 `EQUIV_PASS`）。相对 `atol=5e-2`/`1e-1` 门禁余量充裕。Stage 3 门禁以 §8.1 golden 为准。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式覆盖：3 个 manifest workload + smoke、fp16/bf16、最小/增长 `conv_state`、多步递推、门控角点、膨胀位置、状态一致性。**

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 smoke fp16 | `(B=1, d_mem=256, d=256, L=9, w=4, δ=3)` fp16 | 与 §8.1 golden 比对 `y` ≤5e-2、`st` ≤5e-2 | 门槛通过 |
| L0-2 full W2 fp16 | `(1, 512, 256, 12, 4, 3)` fp16 | 同上 | 门槛通过 |
| L0-3 full W3 fp16 | `(4, 1024, 512, 20, 4, 5)` fp16 | 同上 | 门槛通过 |
| L0-4 full W4 bf16 | `(8, 512, 256, 18, 4, 3)` bf16 | 同上（`y` ≤1e-1、`st` ≤1e-1） | 门槛通过 |
| L0-5 多步递推 fp16 | `B=2,d_mem=256,d=256,L=9,δ=3`，`conv_state` 从 `L=0` 空态起递推 `L+8` 步 | 每步 `y max_err < 0.1`、有效状态 `max_err < 0.05`（同 `test_engram_decode_multi_step`） | 门槛通过 |
| L0-6 空态起步 | `conv_state:(B,0,d)`（host 左 pad 到 `max_conv_len`） | `new_conv_state` 前 `L-1` 行全 0、末行 `= v_n`；`y` 与 golden 一致 | 门槛通过 |
| L0-7 门控角点 | 构造 `⟨h_n,k_n⟩` 为大正/大负 → `alpha→1/0` | `v_hat` 分别 ≈`v`/≈`0`，无 NaN/Inf；`y` 与 golden 一致 | 门槛通过 |
| L0-8 膨胀位置 | `δ=3,w=4`（tap 行 `L-9/L-6/L-3/L-0`）+ `δ=5,w=4`（W3，tap 行 `L-15/L-10/L-5/L-0`） | tap 读取行正确（与 golden 逐 tap 对照） | 门槛通过 |
| L0-9 bf16 全链 | W4 bf16 | `vcast` 升/降无精度异常（对照 golden，`equal_nan` 口径见 §9） | 门槛通过 |
| L0-10 状态一致性 | 单步后 `new_conv_state[:, :, :d]` 与 `conv_state[:, 1:, :]` 逐位一致（位移部分） | 位移语义正确 | 门槛通过 |

- **执行方式**：`tilelang.compile(main, target="npuir")` 后经 `EngramDecodeOp`/wrapper 单次调用，`torch.testing.assert_close(op_out.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，28/28 `EQUIV_PASS`，§1.6.1）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 bf16 v-prefix 数学 API 缺失**（vadd/vmul/vdiv/vexp/vsigmoid/vsqrt/vrsqrt/vmin 各 op doc §2.2.1 bf16 ✗） | W4（bf16）直传编译失败 | 全链 fp32 中转（`vcast`，§0.6 R2）；Cube 段可直接吃 bf16 |
| **R-2 `T.vrsqrt` 近似指令**（`TRAP-vrsqrt-plain-precision`，2.9e-3） | 3 次 RMSNorm 误差累积 | `vsqrt+vdiv` 绕法（§1.6.1 O1，D-1 证 6.1e-5） |
| **R-3 `T.gemm` 的 `a_transpose=True` 未在 examples/gemv 直接实证** | 若编译/效率问题，主选结构受阻 | 备选 A2 = M=16 补零形式（§1.6.3 裁决）；备选 B = 纯 Vector GEMV（结构完整，§1.6.3） |
| **R-4 Cube↔Vector（MixCV）在 Developer 模式的 auto-split** | 若同步/兼容问题 → 编译或结果错误 | 先例 `flash_attn_npuir_dev.py`（Developer+gemm+v-op）；**非 persistent**（规避 `TRAP-DEVMODE-PERSIST-GEMM`）；列 Stage 3 首要验证点 |
| **R-5 物理核数利用率低（B≤8 vs 20 核）** | W1/W2（B=1）仅 1 核工作 | decode 算子固有（batch 维小）；split-N/split-K 跨核结构列 Stage 4 备选（§5.5） |
| **R-6 `grid=B` 每 batch 重读权重（B× GM 流量）** | W3 实际流量 ≈8.3MB vs 理想 2.28MB | R3/R4 候选评估已计入（§1.6.0 D-2）；split-N 备选可摊分 |
| **R-7 发射项主导（名义）** | §1.6.0 估算示发射项主导（25µs） | 标注「未文档化假设 + 估算依据」（0.5µs/op 绑 910B2C 且 stale，910B3 未重验）；BK/grid 为杠杆；Stage 4 baseline 后对账 |
| **R-8 UB 膨胀系数不确定**（`TRAP-UB-multibuffer-inflation`，status=verified；VP-2026-0059「预算表 ≠ 可编译证明」） | 大 BK/多 buffer 可能超 192KB | 按 ×1.7 保守核对（≈101KB 安全，含 17 个 f32 buffer 全量计）；`BK` 放大须编译探针校准；备选 B 附 Stage 4 UB 编译探针计划（§1.6.3） |
| **R-9 L1 容量** | `BK` 放大 × `num_stages` 线性增长 | BK=128×2 stage = 256KB < 512KB；BK=256×2=512KB 触顶，故默认 128 |
| **R-10 条件 `T.alloc` / 运行时 Var 索引**（`TRAP-conditional-alloc`/`TRAP-C13`） | 编译期 `if` 内 alloc、`for p in range()` 在 prim_func 内 → trace/编译失败 | buffer 顶层无条件分配；tap 在 host 层展开（§3.3） |
| **R-11 `T.copy` 隐式跨 dtype cast**（`TRAP-C12`） | 舍入不可控 | `T.copy` 仅同 dtype；cast 走显式 `T.vcast`（§3.2） |
| **R-12 设备语义未验证**（`TRAP-verf-maclaurin-truncation` 提示"文档语义≠设备实现"；`vsigmoid` 降为 VAdd/VSub/VExp/VDiv） | `vsigmoid`/`vexp` 的设备精度可能与文档不符 | 设计期 `verify_equiv.py` 已证 fp64 域等价；Stage 3 加设备微探针（`vsigmoid` vs `torch.sigmoid`，±10 全域）核对 |
| **R-13 常数表设备/版本**（`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-L1-port-bw` 绑 910B2C/9362 且 `kb_stale_check` 标 stale；无 910B3 条目） | D-2 估算不确定性 | 数字标注「待重验」；Stage 4 baseline 后对账回填 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接传入 v-op | 未做 `vcast` 上转 | 编译失败 | bf16 先 `vcast(rint)` 到 fp32（R-1） |
| `vcast` 用非 `rint` 升级 | `f16→f32`/`bf16→f32` 仅 `rint` | 编译失败 | 升降统一 `round_mode="rint"` |
| `T.copy` 跨 dtype | 用 copy 兼做 cast | 隐式 `VCastOp`、舍入不可控 | 显式 `T.vcast`（R-11） |
| `for p in range()` 写在 prim_func 内 | conv tap 展开 | 运行时 Var → 动态 slice 非法 | host 层展开 / `@T.macro` 链（R-10） |
| `T.gemm` 的输出设为 fp16 | fp16×fp16 场景 | 运行时卡死（`T.gemm.md` §2.3） | `c_frag` 用 fp32 |
| `initC` 未在 k0==0 重置 | K 分块累加 | 结果累加错误 | `initC=(k0==0)` |
| L1 溢出 | `BK`/`num_stages` 过大 | 编译失败（`cbuf overflow`） | BK=128、stage=2（R-9） |
| UB 溢出 | buffer 过多/过大 | 编译失败（`ub overflow`） | §4.5 预算 + 编译探针（R-8） |
| 膨胀 tap 索引越界 | `max_conv_len < δ(w-1)` | 越界读 | `EngramDecodeKernel.__init__` 已强制（§0.1）；左零填充区贡献 0 |
| 状态位移方向错 | `new_state[s]=old_state[s+1]` 写成 `s-1` | 语义错 | §8.3 L0-10 逐位核对 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **`conv_state` 空态（L=0）** | host 左 pad 到 `max_conv_len`（全 0）→ kernel 只认 `max_conv_len`；前 `L-1` 行读出全 0 |
| **`conv_state` 增长期（L < max_conv_len）** | 同上（host pad）；`new_state` 由 kernel 输出 `max_conv_len` 后 host 无需 crop（除 d pad） |
| **`d` 非 256 倍数（通用）** | host 尾维 pad 到 `d_padded` + 输出 crop（wrapper 既有）；manifest 不触发 |
| **bf16 workload** | 全链 `vcast(rint)` 中转；Cube 段直吃 bf16 |
| **门控饱和（`alpha→0/1`）** | `sigmoid` 全域稳定（§1.6.1 E3）；`v_hat→0` 时 `RMSNorm` 的 `s=eps` 非零，无除零 |
| **`NaN`/`Inf` 输入/中间值** | 契约 workload 不含；`sigmoid`/`rsqrt` 定义域保护（`s≥eps>0`）；Stage 3 `equal_nan=True` 口径核对（R-12） |
| **极小/极大 batch** | B≤物理核数，grid=B 无需特判；极大 B（>20）触发运行时串行调度（当前 manifest 不涉及，若扩展需 §5.5 第三分支） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/engram_decode/_engram_decode_kernel/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/engram_decode/_engram_decode_kernel/verify_equiv.py` | D-1：28/28 `EQUIV_PASS`（§1.6.1） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_engram_decode_kernel.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_engram_decode`） | 依 §8.1（移植源仓参考实现） |
| NPU wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode.py` | `out_idx=[8,9]` 契约 + `ascend_mode="Developer"`（K5/K7/K8/K9） |
| 抽取脚手架（已存在，只读参照） | `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_kernels.py` | Pattern B 源直译参照 |
| op 集成（已存在，只读参照） | `examples/TileOPs/tileops/ops/sequence_modeling/engram_decode.py` | `EngramDecodeOp` + `eval_roofline` |
| 规格（已存在，只读参照） | `examples/TileOPs/tileops/manifest/sequence_modeling.yaml` | `EngramDecodeOp` workloads/roofline |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_engram_decode.py` | `engram_decode_step_torch`（golden 源）+ fixture |
| 性能用例（已存在，只读参照） | `examples/TileOPs/benchmarks/ops/bench_engram_decode.py` | manifest-driven bench |

**实现要点速览（Stage 3 输入）**：
1. **结构**：一维 `T.Kernel(batch, is_npu=True)`，每 batch 一 block；Cube `T.gemm`（a_transpose=True，K 分块 `T.Pipelined` BK=128/stage=2，fp32 `c_frag`）+ Vector fp32 链。
2. **计算域**：fp16/bf16 经 `T.vcast(rint)` 升 fp32，全链 fp32，末端 `vcast(rint)` 单次舍回（bf16 API 强制）。
3. **RMSNorm**：`s=sumsq·fl(1/d)+eps` → `t=vsqrt(s)` → `rrms=t/s`（绕开 `vrsqrt`）→ `x·rrms·w`。
4. **门控**：`alpha=vsigmoid(dot·fl(1/√d))`；`vhat=alpha·v` 物化（CSE）。
5. **conv**：tap（≤3 历史 + 1 当前）host 层展开；历史行索引 `L-(w-1-p)δ` 编译期常量；左零填充区贡献 0。
6. **状态**：二维区域 `T.copy` 左移 + 末行 `vcast` append。
7. **纪律**：buffer 顶层无条件分配；v-op 作用于整 buffer；`T.copy` 不跨 dtype；`for p in range()` 放 host 层；`initC=(k0==0)`。
8. **bf16**：必须 fp32 中转（v-op 无 bf16）。
