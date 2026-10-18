# prefill_blocksolve_A_bthd 算子设计文档

> GDN prefill 流水线**函数 2 / 6**。所属算子：`GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`）。
> 源算子（source of truth）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`
> 的 `_prefill_blocksolve_A_bthd_tl`（L158 起）与包装 `prefill_blocksolve_A_bthd`（L654 起）；
> 仓内快照 `examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/_gdn_prefill_kernels.py`（L174 / L670）。
> 工具链版本戳：**CANN 9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （`/home/developer/tilelang-mlir-ascend`）/ **Ascend910B3**（`npu-smi 25.5.0`，Chip 6）/ 2026-10-17。
> 编程模式：**Expert**（`.bashrc` 默认 `TILELANG_ASCEND_MODE=Developer`，必须显式覆盖，见 §2）。
> 算子级迁移设计：`examples/gdn_prefill/DESIGN.md`（rev.2）；工具链实测：`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12。
>
> **本文件是 revision 1（design v1）**，取代 `history_version/design_v0.md`。相对 v0 的关键调整见 §12：
> 4 个阻塞级问题（广播方向、`N` 的负号、输出落盘 API 语义、复杂度表算术）全部修正，并新增
> **4 个设计期探针**把其中 3 项从"纸面论证"升级为**真机实测**（§3.4 / §11）：
> `design_probe_store_layout.py`（A 落盘路径）、`design_probe_gate_sign.py`（门/符号全链路）、
> `design_probe_vcast_shapes.py` + `design_probe_mask.py`（API 形态微探针）。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

给定 packed 序列的 key `k`、chunk 内累积的 log 门 `g`、步长 `beta`，**逐 chunk（C = 64 行）**构造
delta rule 的**系统矩阵**并输出其逆：

```
S_sys[i,j] = beta_i · exp(g_i - g_j) · (k_i · k_j)        i > j   （严格下三角）
S_sys[i,j] = 0                                            i <= j
A          = (I + S_sys)^{-1}
```

- `A` 是 **64×64 单位下三角矩阵**（对角为 1，上三角为 0），按行写回 `A[0, base+i, hid, :]`。
- 逐 **chunk 重置**：`g` 已经是 kernel 1 输出的 *chunk 内*累积门，本函数不再跨 chunk 累积。
- 每个 `chunk` 内 `i` 的参考原点就是该 chunk 第一行（`g_i` 相对 chunk 起点），故 `exp(g_i-g_j) <= 1`（门单调非增时）。

**符号归属（本版单独复核，是本次修订的核心）**：源码对角块写 `a_s[diag] = -G·s`
（L480–L503），行串行递推为 `x[r] = e_r + Σ_{m<r} a[r,m]·x[m]`（L519–L540），
即 `X = (I - a_dd)^{-1} = (I + S_sys)^{-1}`；非对角块 `a_s[1] = +G10·s10`（L500）正是 `S_sys` 的 (1,0) 块，
回代时再取负（L543–L550）。**数值裁决**（`verify_equiv.py` 末尾，fp64 结构通路）：
源分块算法与 `inv(I+S_sys)` 差 **2.8e-17**、与 `inv(I-S_sys)` 差 **4.4e-01** ⇒ 符号唯一，
`S_sys` 的符号归属由代码事实确定，不是书写约定。v0 在 `N` 上丢掉的负号即此。

**dtype 语义**（源码逐行核对）：

| 量 | 源实现 dtype | 依据 |
|---|---|---|
| `k` / `beta` / `A` | 输入 dtype（fp16 / bf16） | `A = torch.empty(..., dtype=k.dtype)`（L688） |
| `g` | **fp32** | 张量签名 `g: T.Tensor([1, total_tokens, head], "float32")`（L216） |
| Gram / 矩阵乘 | fp32 累加 | `accum_dtype = "float32"`（L199） |
| 门比值与 `a_s`/`i_s`（中间逆块） | **落到 `solve_dtype = dtype`（fp16/bf16）** | `solve_dtype = dtype`（L200）、`i_s = T.alloc_shared(..., solve_dtype)`（L242） |
| `beta` 参与运算前 | **先舍入到 dtype** | `beta_s[t] = T.cast(step, dtype)`（L295） |

**规约语义**：Gram 的规约轴是 `dim_k`，源实现按 `block_k = 64` 分 `dim_k/block_k` 段串行累加
（`for kt in T.Serial(dim_k // block_k)`，L310）→ fp32 累加器，段内 64 深、段间顺序累加。

**边界语义**（关键，逐行核对 L288–L298、L317–L324、L621–L650）：

- 序列最后一个 chunk 可以不满 64 行（如 `S = 3000` 的最后一块只有 56 行）。
- 越界行（`base + t >= end`）的处理是**双管齐下**：`g` 取 `g[min(base+t, end-1)]`（重复最后一行），
  `beta` 置 **0**；`k` 的行索引同样 clamp 到 `end-1`（L325–L345 的 `else` 分支）。
  于是越界行在 `S_sys` 中整行为 0 ⇒ 其逆的行就是单位行 `e_i`；**输出时再按 `base + i < end` 逐行掩码，越界行不写**。
- 该 clamp 是**语义正确性的一部分**：若越界行填入未初始化数据，其列会污染合法行的逆（见 §3.5.2）。

**`step_size` 语义**（`prefill_common.py:11`）：本函数 in-scope 取 `beta_sigmoid=False`、`allow_neg_eigval=False`
⇒ `step_size(raw) = raw`（恒等，无 sigmoid）；源码仍把它 `cast` 到 dtype。`l2norm=False` ⇒ `k_rnorm` 不被写
（host 只分配 1 token 占位，L689–L691）。`use_gate=True` ⇒ 读 `g`。

### 0.2 源算子输入输出

| 名称 | 形状 | dtype | in-scope | 说明 |
|---|---|---|---|---|
| `k` | `[1, total_tokens, key_head, dim_k]` | fp16 / bf16 | ✅ `dim_k ∈ {64,128}` | packed key（上游已归一化，本内核不再归一化） |
| `g` | `[1, total_tokens, head]` | **fp32** | ✅ | chunk 内累积 log 门（kernel 1 输出） |
| `beta` | `[1, total_tokens, head]` | 输入 dtype | ✅ | delta rule 步长 |
| `cu_seqlens` | `[num_sequences+1]` int32 | int32 | ⚠️ 等长时**不读**（`uniform_seq_len > 0` 走算术分支，L257–L260） | 序列 offsets |
| `A`（输出） | `[1, total_tokens, head, 64]` | 输入 dtype | ✅ | `(I+S_sys)^{-1}` 的非零下三角部分 |
| `k_rnorm`（输出） | `[1, total_tokens or 1, key_head]` fp32 | fp32 | ❌ 仅 `l2norm=True` 写 | 本批占位（1 token） |

`head % key_head == 0`（L191 硬校验），`group = head // key_head`；第 `hid` 个 recurrent head 读第
`khid = hid // group` 个 key head（L270）。`chunk_size != 64 or dim_k not in (64,128)` 直接 `ValueError`（L189–L192）。

### 0.3 实现算法解读（怎么算）

源 kernel 的并行结构：`T.Kernel(num_chunks, head)`（L222，二维 grid）→ 一个 block 负责
**一个 chunk × 一个 head**，`threads=32`；block 内把整个 64×64 问题留在 shared memory + fragment 里。

**计算步骤分解**（源码每个计算语句的归属）：

| 步 | 计算 | 源码行 | 输入 | 输出 |
|---|---|---|---|---|
| P0 | 解出 `base / end / seq`（等长走算术，否则 `GroupTiling.decode`） | L256–L271 | `cid`, `cu_seqlens` | `base`, `end`, `khid` |
| P1 | 读门与步长（越界行 clamp + β=0） | L288–L298 | `g`, `beta` | `g_s[64]`, `beta_s[64]` |
| P2 | 10 个 16×16 Gram 块：`G_ij = k_i k_jᵀ`（`dim_k/64` 段累加） | L299–L358 | `k` 四个 16×dim_k 片 | `G00,G10,G11,G20,G21,G22,G30,G31,G32,G33`（fragment fp32） |
| P3 | （仅 `l2norm`）用 Gram 对角线做行归一化并缩放全部块 | L360–L386 | — | 本批跳过 |
| P4 | 4 组门比值 `gate{0..3}_s[64] = exp2((g_t - g_{16b})·log2e)` | L388–L427 | `g_s` | 4×`gate`（**dtype** 存储） |
| P5 | 组装 10 个下三角块 `a_s[10,16,16]`：对角块 `-G_bb·β_i·gate_b[i]/gate_b[j]`（严格下三角），非对角块 `+G_ij·β_i·gate_i[i]/gate_j[j]` | L429–L505 | `G*`, `gate*`, `beta` | `a_s`（fp16/bf16） |
| P6 | **4 个对角 16×16 块的前代求逆**：`X = I + a X` 逐行（`r` 串行 16 步，`m < r` 内积，lane 承担 2 列，奇偶项分链） | L507–L540 | `a_s[diag]` | `i_s[4,16,16]` |
| P7 | **块前代**（下三角块回代，共 **16** 次 16³ gemm + 中间 `work_s` 搬运）：`A_10 = -X_1 S_10 X_0`；`A_20 = -X_2(S_20 X_0 + S_21 A_10)`；`A_21 = -X_2 S_21 X_1`；`A_30 = -X_3(S_30X_0+S_31A_10+S_32A_20)`；`A_31 = -X_3(S_31X_1+S_32A_21)`；`A_32 = -X_3 S_32 X_2` | L542–L620 | `a_s`, `i_s` | 覆写 `a_s[1,3,4,6,7,8]` 为逆的非对角块 |
| P8 | 按行写回 `A`：行块 0..3 = `[i_s0 / (A_10,i_s1) / (A_30,A_31,i_s2) / (A_60,A_71,A_82,i_s3)]`，上三角显式写 0；`base+i < end` 才写 | L621–L650 | `i_s`, `a_s` | `A` |
| P9 | （仅 `l2norm`）把 `rnorm` 交给后续 kernel | L379–L385 | — | 本批跳过 |

**gemm 计数（本版更正 v0 的 15/25）**：块回代 `T.gemm` 出现在 L543、548、555、559、564、569、574、579、
583、587、592、597、601、606、611、616 共 **16 次**；加 10 次 Gram（`dim_k=128` 时按 `block_k=64` 分 2 段
⇒ 20 次）⇒ **26（dim_k=64）/ 36（dim_k=128）** 个 gemm/任务。

**数据流图（源码硬件视角）**：

```
GM(k,g,beta) --cp.async/标量--> SMEM(k0..k3 16x64, g_s, beta_s)
   --> fragment(G_ij fp32, mma)  --共享内存-->  a_s[10,16,16] (fp16)
   --> i_s[4,16,16] (fp16, 寄存器行串行求解)  --> 16x gemm(16^3) 经 work_s 回代
   --> 按行 store --> GM(A)
```

所有中间量都在 SMEM/fragment 内，**没有任何跨 block 通信、没有中间量落回 GM**。

**host 侧逻辑**（L654–L716）：分配 `A`（`torch.empty`，形状 `[1,total_tokens,head,chunk_size]`）与 `k_rnorm`
（`l2norm` 为假时 1 token），按形状/开关查 `lru_cache` 编译好的 kernel，调用后返回 `(A, k_rnorm)`。
host **无任何数据变换**；grid 由 `GroupTiling.tile_upper_bound` 决定上界，非等长分支下**越界 block 被
`T.min(cid, last)` clamp 到最后一个 chunk 幂等重算并重复写同样的字节**（L263–L266 原文注释
"a block past their count repeats the last one and writes its own bytes"）——不是早退。

### 0.4 优化手段解读（为什么快）

| # | 手段 | 源码位置 | 目的 | 机制 | 依赖的源硬件 |
|---|---|---|---|---|---|
| O1 | 每 chunk 一个 block、整块问题驻留 shared | L222–L255 | 把 64×64 问题钉在片上，零中间 GM 往返 | SMEM + fragment 复用 | CUDA shared memory / 大寄存器堆 |
| O2 | 只算 10 个下三角 Gram 块（而非 16 个） | L349–L358 | 省 37.5% Gram 乘加 | 利用对称性 | Tensor Core |
| O3 | `async_copy` + `ptx_wait_group` 预取 4 个 key 片 | L317–L324 | 掩盖 GM 延迟 | cp.async | 异步拷贝引擎 |
| O4 | Gram 按 `block_k=64` 分段累加到 fragment | L310, L349 | 支持 `dim_k=128` 且累加器常驻 | register 累加 | register file |
| O5 | 门比值用"块内参考 + 两次 exp + 一次除法" | L388–L505 | **门表复用**：4×64 次 exp 替代逐元素 4096 次 exp，并把除法摊成"门表相除"（每次 1 次 `/`） | 分块参考 `r=16b` + `exp2` 换底 | 无（算法层） |
| O6 | 对角块用**行串行前代**而非求逆公式 | L507–L540 | O(16³/6) 而非 O(16³)，且天然下三角 | 顺序依赖链 | 32 lane × 2 列的线程映射 |
| O7 | 块前代：用 4 个小逆 + **16** 次 16³ gemm 拼出 64×64 逆 | L542–L620 | 把 64³ 的逆降到 ~0.4 MFLOP（比 log-doubling 少 8× 乘加） | 分块消元 | Tensor Core（小 mma） |
| O8 | 奇偶分链（`e/o` 两个累加器）缩短行内依赖 | L529–L538 | 降低 16 步链的指令延迟 | ILP | 指令级并行 |
| O9 | 越界行 clamp + β=0 + 输出掩码，使"尾部 chunk"与整 chunk 同一程序 | L288–L324, L621 | 消分支、消 padding kernel | 数据无关掩码 | 无（算法层） |
| O10 | `T.annotate_layout(swizzled)` | L266–L285 | 消 SMEM bank conflict | swizzle | SMEM bank 结构 |
| O11 | 等长走算术解 chunk 边界（不读 offsets） | L256–L260, L186–L188 | 免一次 GM 读与依赖 | 常量折叠 | 无 |
| O12 | 网格取 offsets 上界，越界 block **clamp 到末 chunk 幂等重算**（非早退） | L192–L196, L263–L266 | launch 范围与数据无关 | 重复计算 + 重复写同样字节 | 无 |

> **本版修正（REVIEW v0 问题 11）**：① O12 不是"立即退出"——源码是 `T.min(cid, last)` 的幂等重算；
> 本设计改"固定核数 + `if t<T` 守卫"是**等价替换**（少算重复任务），不是承接"早退"语义。
> ② O5 的真实收益是"门表复用 + 除法复用"，**不是**数值稳定化——`exp(g_i-g_j)` 全差式在数值上并不更差
> （本设计 A1 实测 worst rel 3.0e-8，且少一次 fp16 门表舍入）。

### 0.5 硬件耦合性分析与 NPU 适配决策

| # | 源方案 | 硬件前提 | 处置 | 依据 |
|---|---|---|---|---|
| O1 | 整块问题驻留 SMEM/fragment，中间量零 GM 往返 | CUDA shared + 大寄存器堆；L0C 无"存回片上"通路 | **重新设计** | NPU 的 Cube 结果只能 `fixpipe` 落 **GM**（`docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1，`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` CONST-store-fixpipe-gm-only），片内 L0C→L1 直连不存在 ⇒ 依赖链必须经 GM 往返，必须让**依赖级数**最小 |
| O2 | 只算下三角 Gram | 无 | **保留** | 数学对称性；NPU 侧等价为"整块 Gram + 掩码"，见 §0.6 |
| O3 | `async_copy` 预取 | cp.async | **等价替换** | `T.copy`/`T.load_nd2nz` 走 MTE2；两 slot 交替 + Cube 领先一任务的 Gram 预取即覆盖同一意图（§6） |
| O4 | Gram 分段累加（fragment） | register file | **等价替换** | `T.gemm(..., initC=False)` 跨段累加 L0C（实测 err=0，`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1）；本设计进一步合并为单次 `size=[64,dim_k,64]` |
| O5 | 块内参考 + 除法 | 无 | **等价替换（并简化）** | `exp(g_i-g_j)` 与 `exp(g_i-r)/exp(g_j-r)` 数学恒等；NPU 上可直接算 64×64 差值 + 一次 `vexp`，省去 4 组参考与 2016 次除法（§1.6.1 A1/A2） |
| O6 | 对角块**行串行**前代（16 步 × 内积） | 32 lane 线程映射 + 寄存器 | **重新设计** | NPU Vector 无"每 lane 一列 + 动态下标读 UB"的高效形态；16 步标量链在 Vector 上 ≈ 120 次窄 op（按 `CONST-vector-launch-overhead` 的 **~0.5 µs/op** 计 ≈ 60 µs/任务），且 `T.get_thread_binding` 线程模型在 npuir 不适用（`docs/Tilelang.language/` 16 个子目录均无该 API）⇒ 改用 **log-doubling**（§0.6 R1） |
| O7 | 4 小逆 + **16** 次 16³ gemm 分块消元 | Tensor Core 小 mma 便宜 | **重新设计** | NPU 上每个 gemm 都要 L1/L0 装载 + L0C 落盘，**op 数**而非 MAC 数主导（§1.6.0 R3/R4）：16 次 16³ + 10 次 Gram = **26** 个 gemm（`dim_k=128` 36 个）；改为 **13** 个 64³ gemm（含 Gram 与 `gemm(I,I)` 播种） |
| O8 | 奇偶分链 ILP | 指令级并行 | **舍弃** | 随 O6 一并消失（无行串行链）；理由：该手段只服务于行串行求逆的延迟 |
| O9 | 越界行 clamp + β=0 + 输出掩码 | 无 | **保留（语义层）+ 局部等价替换** | clamp/β=0/掩码的语义必须保留；NPU 上"逐行 clamp 装载"代价高，改为「动态 extent 装载 + 0/1 常量掩码乘法」（§3.5.2），数学等价 |
| O10 | swizzle 消 bank conflict | SMEM bank | **舍弃** | NPU 无 SMEM bank conflict；布局由 `load_nd2nz` 的 NZ 分形承担（`docs/Tilelang.language/内存操作/T.load_nd2nz.md`） |
| O11 | 等长算术解边界 | 无 | **保留** | 纯 host/常量算术（in-scope 等长） |
| O12 | 网格上界 + clamp 重算 | 无 | **等价替换** | 改为"固定物理核数 + 核内 `T.serial` 任务循环 + `task < T` 守卫"（§5），越界任务直接不启动（比源"幂等重算"更省） |
| — | `T.Kernel(num_chunks, head)` 二维 grid | 三维/二维 block 索引 | **等价替换** | npuir 只接受一维 block 数（`.agents/skills/tilelang-op-design/references/ascend-constraints.md`）⇒ `(cid,hid)` 展平为一维任务号 `t` |
| — | `l2norm` 分支（P3/P9） | 无 | **舍弃（本批）** | in-scope `l2norm=False`（算子级 §1），保留 host 侧接口与 `k_rnorm` 占位返回 |

**统计**：保留 3 项（O2/O9/O11）+ 等价替换 4 项（O3/O4/O5/O12 + 二维 grid）+ 重新设计 3 项（O1/O6/O7）+ 舍弃 3 项（O8/O10/l2norm 分支）。

### 0.6 NPU 算法重设计

重设计的输入是 §1.6.0 的调研结论（选定 **R1 log-doubling 求逆**），不是源方案的直译。

**R1（承接 O1/O6/O7）：64×64 单位下三角求逆改为 log-doubling 乘积形式**

```
N ≜ -S_sys        （严格下三角，故 N^64 = 0）
A = (I - N)^{-1} = Σ_{j=0}^{63} N^j = Π_{i=0}^{5} (I + N^{2^i})          （= (I + S_sys)^{-1}）
  T_0 = I;  P_j = N^{2^j}
  T_{j+1} = T_j + P_j T_j        （gemm(P_j, T_j, L0C, initC=False)，L0C 内已有 T_j）
  P_{j+1} = P_j P_j              （j = 0..4；P_5 = N^32 用后即弃，不需要 N^64）
```

- **恒等式书写（本版更正 v0 的代数笔误）**：`N ≜ -S_sys` 时 `A = Σ N^j = (I - N)^{-1} = (I + S_sys)^{-1}`。
  v0 写的 `A = (I+N)^{-1} = Σ N^j` 在 `N=-S_sys` 下实际等于 `(I - S_sys)^{-1}`，与源码相反。
- 语义保持：`N` 严格下三角 ⇒ `N^64 = 0` ⇒ 级数在 64 项处截断；`{1,2,4,8,16,32}` 的子集和恰好覆盖
  `0..63`，乘积展开逐项等于 `Σ N^j` ⇒ **数学恒等**（机器验证 §1.6.1 A4：rel ≤ 4.0e-6；A6 逐句展开：rel ≤ 4.6e-4）。
- **负号的唯一落点**：Vector 取到 `beta`（fp16）后 `vcast` 到 fp32，立刻 `vmul(beta, -1.0, -beta)`
  （**1 个 64-wide op**），此后 `N = mask01 · (-β_i) · exp(g_i-g_j) · G`。这样 `N` 的定义与源码
  `a_s[diag] = -G·s` 逐字对应，且不引入任何 64×64 级额外 op（详见 §3.1 S3 / §3.3）。
- 边界语义：越界行 `β=0 ⇒ N 的该行为 0 ⇒ A 的该行为 e_i`，与源一致；该行在输出上被掩码（§3.5.2）。
- 代价：MAC 8.7×（0.40 M → 3.47 M），但**依赖级 gemm 26 → 13**（含 Gram）、无行串行链、无 16×16 小 gemm；
  Cube 上 64³ 是 64 个分形 op（利用率满），16³ 只有 1 个分形 op（发射开销主导）。
- 源 O7 的"少算乘加"意图**被显式放弃并给出量化依据**（§1.6.0 R3/R4）：在 NPU 上 MAC 便宜、op 贵；
  同理 O6 的"O(16³/6) 而非 O(16³)"意图被"无行串行链"替换，O8 随之不再需要。

**R2（承接 O1）：Cube 只做 Gram 与链式 gemm，逐元素/掩码放 Vector**

Cube 一个 scope 跑全部任务的 Gram 与链；Vector 一个 scope 跑全部门的差值/exp/掩码/组装
（单 kernel 只允许一个 Cube scope + 一个 Vector scope，`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12.2）。
两引擎各自跑任务循环，用 2 slot 双向 flag 握手做软件流水（§7）。

**R3（承接 O9）：尾部 chunk 的越界行**

源方案 = 逐行 clamp（k/g 行索引 `min(base+t, end-1)`）+ β=0 + 输出逐行掩码。NPU 方案：

1. **装载**：`T.copy` 用**动态 extent**（`valid = min(64, end-base)`）只搬合法行
   （仓内先例：`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（L237–L254）
   的 `x[off_m : off_m + real_m]`），越界行留在 L1/UB 的是上一个任务的旧值（有限数）。
2. **掩码**：`N = mask01 ⊙ (-β_i)·exp(min(g_i-g_j,0))·G`，`mask01` 是 **64×64 的 0/1 fp32 常量**（host 提供，
   每 block 装载一次）。`vmin(d, 0)` 先把丢弃区指数限幅 ⇒ `e ≤ 1` 恒有限 ⇒ `0 × 有限 = 0` **精确为 0**，
   不存在 `0 × inf = NaN`；因此不需要 bool 掩码（bool 路径在本工具链上**实测不可用**，见 §3.4 / §12）。
3. **输出**：host 把 `A_pad` 分配到 **chunk 对齐行数** `ceil(total_tokens/64)*64`，kernel 无条件写满 64 行，
   wrapper 返回 `A_pad[:total_tokens]` 的**连续视图**（纯元数据操作）。合法行数值与源逐位一致。

**语义保持论证**：`i<valid` 且 `j<i` 时 `i,j` 都合法（因为 `j<i<valid`），故被保留元素只依赖合法
k/g/β；被丢弃元素恒为 0。又 `N` 的行 `i` 只经 `N^k` 的行 `i` 传播（矩阵乘行独立），越界行的垃圾值
**不可能污染合法行**。故合法行输出与 §0.1 定义逐元素相同（fp 舍入内）。

**与算子级 §6（Neumann 求逆）的关系——本版新增裁决**

算子级 `examples/gdn_prefill/DESIGN.md` §6 把 "C2：Neumann 求逆" 写成流水线的求逆方案，其形态是
`W = (I+S_sys)^{-1} rhs`（**矩阵×单向量**，`rhs` 是 kernel 5/6 内部的状态增量）；本函数的 contract 是
**materialize 完整 64×64 `A`**（`A[1,S,head,64]`，下游 `correct_initial_states` / `fused_gdr_h` /
`fused_gdr_fwd` 三处各自消费整块 A），两者 **contract 不同**：本函数无法用"直接求解替代求逆"
（`ALG-triangular` 的 R1 线索）规避 materialize。因此本函数选 log-doubling 全矩阵求逆**不构成对算子级 §6
的否定**，而是"全矩阵 + 少 op"场景下的最优（量化否证见 §1.6.0 R3 的 C3 行：Neumann 全矩阵形态
≥28 µs/任务 ≈ 3× C1）。本文件不改算子级 §6；建议算子级下一版把 kernel 5/6 的 Neumann 改为
"直接 `gemm(A, rhs)`"（1 个 gemm 替代 63 级链），该建议由 conductor 转到算子级 revision。

### 0.7 标杆实现

- 源实现：`prefill_prepare.py:158`（CUDA，SMEM + fragment + 行串行前代）。
- NPU 侧形态参考（Expert MixCV + 双循环握手）：`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`；
  回喂形态：`examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py`；
  Cube 侧调度先例：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`。
- 本函数的设计期探针（D-3，实测结论见 §1.6.3 / §3.4 / §11）：
  - `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_doubling_cube.py`（v0 起，2 级链 + per-op 时延）；
  - `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py`（v0 起，V1–V3；**V4 分支本轮判废**，见 §11）；
  - `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_store_layout.py`（**本版新增**：A 落盘路径正确性 + strided/contig 时延）；
  - `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_gate_sign.py`（**本版新增**：门/符号全链路真机对拍 + 2 级链 fp16/fp32 + A/B 同 buffer）；
  - `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_vcast_shapes.py`、`design_probe_mask.py`（**本版新增**：vcast/vbrc 形态与掩码构造微探针）。

---
## 1. 概述

### 1.1 算子名称

`prefill_blocksolve_A_bthd`（GDN prefill 流水线函数 2/6；NPU 内核名 `prefill_blocksolve_A_bthd_kernel`）。

### 1.2 功能描述

逐 `(chunk, head)` 构造 delta rule 的系统矩阵 `I + S_sys` 并输出其逆 `A`（64×64 单位下三角），
供后续 `correct_initial_states` / `fused_gdr_h` / `fused_gdr_fwd` 三个 kernel 消费。
in-scope：`chunk_size=64`、`dim_k ∈ {64,128}`、`use_gate=True`、`l2norm=False`、`beta_sigmoid=False`、
`allow_neg_eigval=False`、等长序列（`uniform_seq_len > 0`，`cu_seqlens` 不读）、**支持序列最后一个不满 64 行的 chunk**。

### 1.3 数学公式

对第 `c` 个 chunk（行区间 `[base, end)`，`valid = min(64, end-base)`）、第 `hid` 个 recurrent head、
`khid = hid // group`（`group = head // key_head`）：

```
S_sys[i,j] = beta_i · exp(g_i - g_j) · (k_i · k_j)       0 <= j < i < valid      （严格下三角）
S_sys[i,j] = 0                                           其他（含 i < valid <= j）

N[i,j]   = -S_sys[i,j]  = -beta_i · exp(g_i - g_j) · (k_i · k_j)   0 <= j < i < valid
N[i,j]   = 0                                                       其他

A        = (I - N)^{-1} = (I + S_sys)^{-1}
         = Σ_{j=0}^{63} N^j = Π_{i=0}^{5} (I + N^{2^i})              （N^64 = 0）
A[i,i]   = 1,  A[i,j] = 0 (i < j)
A[0, base+i, hid, :] = A[i, :]                           仅 i < valid 被读出（其余落在 padding 区）
```

其中 `g_i` 是 chunk 内累积门（fp32），`beta_i` 是步长（in-scope 无 sigmoid），`k_i` 是该 head 的
`dim_k` 维 key 行。**符号来源**（与源码逐字对应）：`a_s[diag] = -G·β_i·gate[i]/gate[j]`（L480–L503），
行串行 `x = e + a x`（L519–L540）⇒ `X = (I - a)^{-1}`；而 `s = β_i·gate[i]/gate[j] = β_i·exp(g_i-g_j)`（L430–L434）
⇒ `a_dd = -S_sys`、`X = (I + S_sys)^{-1}`。即 `N` 就是源码的 `a_dd`（对角块），`A` 的下三角非对角块
就是源码回代后的 `a_s[1,3,4,6,7,8]`（L543–L618）。数值裁决见 §0.1（2.8e-17 vs 4.4e-1）。

### 1.4 算法描述

迁移重设计后的 NPU 算法（与 §1.6.0 选定算法一致，见 §0.6）：

1. **任务映射**：逻辑任务 `t = (chunk, head)`，共 `T = nch_total × head`；固定启动 `K` 个 block（§5），
   每个 block 用静态上界的 `T.serial` 处理 `t = bid + it·K`（`it < ceil(T/K)`）。
2. **Cube（一个 scope）**：对任务 `t` 取 `k` 的 64×dim_k 片（动态 extent `valid` 行），
   一次 `T.gemm` 算整块 Gram `G = k kᵀ`（fp32 累加，`b_transpose=True` = 直接复用
   `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.4 形态），落 GM（`gs_G[slot]`）。
   随后按 §1.6.1 A4 的 **6 级 log-doubling（5 次平方）** 求逆：`gemm(I,I)` 播种 L0C（1 次）+
   6 次 `T` 链 gemm + 5 次 `P²` gemm = **12 个链 gemm**，加 Gram 共 **13 个 gemm**（fp32 操作数、
   GM scratch 回喂）。最后 `fixpipe` 把 L0C 量化落到**连续的 64×64 中转块** `y_blk[slot]`（f32→dtype）。
3. **Vector（一个 scope）**：读 `gs_G[slot]` + `g`/`beta` 片段，按 §1.6.1 A1/A4 组出 `N`：
   两次 `vbrc` 广播（列= `g_i`、行= `g_j`）→ `vsub(g_i, g_j)` → `vmin(d, 0)` → `vexp` →
   `vcast(β)` + **一次取负** → `vmul`(β) → `vmul`(G) → `vmul`(mask01)，写 `ws_S[slot]`。
4. **落盘（Vector，一任务滞后）**：把上一任务的 `y_blk` 读进 UB，再以**一次 strided region 写**
   落到 `A_pad[base:base+64, hid, :]`（64 行 × 64 连续元素，行距 `head·64`）。
5. **两引擎流水**：各自跑同一任务列表，2 slot 双向 flag 握手（§7），Cube 侧领先一个任务算 Gram
   并领先一个任务落 `N`，使 Vector 的逐元素相位被 Cube 的链式相位覆盖。

### 1.5 数据流图

```
GM(k)   --MTE2(T.copy slice, 动态 extent valid 行)--> L1(k 片)  --T.gemm--> L0C(G)
        --fixpipe--> GM(gs_G[slot]) fp32
GM(g,beta) --MTE2--> UB --vcast/vmul(-1)/vbrc/vsub/vmin/vexp/vmul/vmul(mask01)--> UB(N)
        --MTE3--> GM(ws_S[slot]) fp32
GM(ws_S[slot]) --MTE2--> L1(P_0)                                   (Cube)
GM(ws_T)  --MTE2--> L1(T_j) --T.gemm(P_j,T_j,L0C,initC=False)--> L0C(T_{j+1}) --fixpipe--> GM(ws_T)
L0C(T_{j+1}) --(gemm P_j,P_j 同 L1 buffer)--> L0C(P_{j+1}) --fixpipe--> GM(ws_P) --MTE2--> L1
... 6 级 ...
L0C(T_6) --fixpipe(f32->dtype)--> GM(y_blk[slot])  contiguous 64x64
GM(y_blk[slot]) --MTE2--> UB(fp16/bf16) --MTE3(strided region)--> GM(A_pad[base:base+64, hid, :])
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

调研对象：**同一数学语义**（给定 64×64 单位下三角 `I + S_sys`，输出其逆）的算法族。基线候选 = 源算法。
调研深度：完整调研（矩阵类）。检索面：`references/algorithm-candidates.md`（线性代数 / 三角求解族）、
`pattern-library`（cases/layout/constants/traps + `python3 .agents/tools/kb_search.py` 命中条目）、
仓内 `examples/`（`multi_head_attention` 的 MixCV 形态、`ssd_chunk_scan` 的分块状态族）、源算子与
FLA 系列公开实现（模型知识层）。

**R1 等价化简公式候选**（含基线；正式论证在 §1.6.1）

| 候选 | 公式/结构 | 等价性初判 | 收益方向 | 采纳 |
|---|---|---|---|---|
| **B0 源算法（基线）** | 4 个 16×16 对角块行串行前代（`X=I+aX`）+ **16** 次 16³ gemm 块前代 | — （语义基准） | MAC 最少（0.40 M/task） | 基线 |
| R1-a | `exp(g_i-g_j)` 单次指数（替代"块内参考 + 两次 exp + 一次除法"） | 数学恒等；源式另含 fp16 门表舍入 | 每元素 2 exp+1 div → 1 exp；省 4 组参考表 | **采纳**（§1.6.1 A1） |
| R1-b | 除法 → 乘倒数（保留"参考因子"分解时的退路） | 恒等（`1/x` 一次） | 2016 次除法 → 2 次 64-wide 倒数 + 2 次 vmul | 采纳为**备选形态**（§1.6.1 A2；主选路径无除法） |
| R1-c | `β` 保持 fp32（源 `cast` 到 dtype） | 容差内等价（差异 = 1 次 fp16 舍入） | 少一次舍入 | **采纳**（§1.6.1 A3） |
| R1-d | 把 `−1` 折进 **β 的标量乘**（而非在 64×64 tile 上取负），`I` 由 `gemm(I,I)` 播种 | 恒等（标量可交换） | 1 个 **64-wide** op 替代 2 个 64×64 op（≈ −0.95 µs/任务） | **采纳**（§1.6.1 A4，落点见 §3.1 S3） |
| R1-e | 把行/列门因子折进 key（`N = -(D K)(D' K)ᵀ`），Gram 直出带符号 `N` | 恒等（`rank-1` 缩放可结合） | 省 64×64 的两次 `vmul` | 弃：需把缩放后的 key 写回 GM（2×16 KB 写 + 16 KB 读）并仍要 64×64 掩码；净增 32 KB/任务流量与 2 个 op（按 0.25 µs/op、72 GB/s 折算 +0.9 µs），无收益 |
| R1-f | 只算下三角 Gram（源 O2） | 恒等（对称） | 省 37.5% Gram MAC | 弃（NPU 形态）：Cube 的一次 64×64×dim_k gemm 已是最少 op 数；裁三角需拆成 10 次 16×16 gemm = 回到 B0 的 op 数 |
| R1-g | **直接求解替代显式求逆**（`ALG-triangular` 命中行"只需解不需逆时直接求解"） | 对"只算 `A·rhs`"的调用方等价 | 免 materialize，省整块逆 | 弃：本函数 contract 就是输出 64×64 逆（§0.2 `A[1,S,head,64]`、下游三 kernel 各自消费整块 A）⇒ 该线索适用于 kernel 5/6 的 `(I+S_sys)^{-1}rhs`，不适用于本函数（§0.6 裁决段） |
| R1-h | **对角/三角降阶**（`ALG-triangular` 命中行"对角/三角特例降阶"） | 恒等（分块消元） | MAC 降 8× | 对应候选 **C2**（16 块 + 对角 16³ doubling）；弃选理由 = op 数（58 gemm vs 13），量化见 R3 |

**R2 在线算法：结论「对本函数无在线变体」**，结构依据三条：

1. 输出**就是** 64×64 逆矩阵本身（不是某个向量/标量统计量），任何流式变体最终仍要 materialize 这 64×64 个元素；
2. 跨 chunk 的流式结构已经存在——但它在 kernel 5/6 里（把 `A` 折进状态递推），本函数的 contract 是"逐 chunk 独立"，
   所以"在线化"只能是把递推搬进本函数 → 那是**改变函数 contract**，不是等价实现；
3. token 级在线形式（每次 rank-1 更新 `(I+β k kᵀ)` 再求逆）在本 chunk 内需要 64 次顺序更新（含 64 次 gemm + 逐元素），
   op 数是 C1 的 6 倍以上，且需要 64 次跨引擎往返 → 严格劣于任何候选（量化见 R3 的 C3 行）。
   *（该结论为结构判据 + 源算子自身"逐 chunk 分块"证据，非"未检索"式断言。）*

**R3 算法复杂度四口径对比**（口径：**每任务 = 1 个 chunk × 1 个 head，`C=64`、`dim_k=128`（fp16 key）**；
`FLOPs = 2×MAC`，只计乘加、不含掩码/指数折算；`访存 Bytes` = 输入读 + 输出写 + 中间 GM 往返（64×64 fp32 块 = 16 KB）；
`扫描遍数` = 该任务对 64×64 中间块的 **GM store+load 往返对数**（依赖级的落盘次数）；
`中间缓冲峰值` = 单 block 片上峰值（UB/L1/L0C）；`op 总数` 按 §1.6.0 末尾的计价口径逐项相加）

| 候选 | FLOPs/任务 | 访存 Bytes/任务 | 扫描遍数 | 中间缓冲峰值 | gemm/任务 | op 总数/任务 | 可并行度 |
|---|---|---|---|---|---|---|---|
| **B0 源算法直译** | 0.80 M | 41 KB | 16 | ~16 KB | **26**（`dim_k=128` 时 **36**） | 26 gemm + ~120 窄 vop + 16 块搬运 ≈ 162 | T |
| **C1 log-doubling（选定）** | 6.94 M | 425 KB | 13 | UB 122 KB + L1 64 KB + L0C 48 KB | **13** | **55**（37 Cube + 18 Vector） | T |
| C2 分层（16 块 + 对角 doubling） | 1.05 M | 104 KB | 16 | 2×16 KB | **58** | ~90 | T |
| C3 Neumann 级数（算子级 §6 方案） | 33.7 M | 2.0 MB | 63 | 2×16 KB | **63** | ~130 | T |
| C4 Newton 迭代 `X←X(2I-MX)`（6 次） | 6.94 M | 0.60 MB | 12 | 3×16 KB | **12** | 36 + 6 次强制 CV 往返 | T（但相位串行） |
| C5 纯 Vector 分块前代（B0 搬到 UB） | 0.80 M | 30 KB | 1 | ~100 KB UB | 0 | ~200 窄 vop | T（Vector 核 ×2） |
| C6 纯 Vector 全 64×64 行串行 | 0.80 M | 30 KB | 1 | ~64 KB UB | 0 | 2016 窄 vop | T |

**逐候选定量（估算口径见本节末尾的常数表）**

- **B0**：`26 gemm × 0.25 = 6.5 µs` + `120 窄向量 op × 0.5 = 60 µs`（`CONST-vector-launch-overhead` 实测 ~0.5 µs/op）
  ⇒ **≈ 66.5 µs/任务**；且 16 级 GM 往返把 Cube 与 Vector 串成一条长链。
- **C1**：Cube 37 op × 0.25 = **9.25 µs**；Vector 11×0.5 + 7×0.25 = **7.25 µs**（两引擎流水 ⇒ max）；
  流量 425 KB ÷ 72 GB/s = 5.9 µs ⇒ **9.25 µs/任务**，**比 B0 快 7.2×**。
- **C2**：58 gemm × 0.25 ≈ **14.5 µs/任务**（MAC 只有 C1 的 15%，但 op 数是 4.5×）⇒ 淘汰。
- **C3（算子级 §6 的 Neumann 方案，全矩阵形态）**：63 次依赖 gemm ≈ 15.75 µs + 2.0 MB 往返 ÷ 72 GB/s ≈ 28 µs
  ⇒ **≥ 28 µs/任务 ≈ 3× C1**；仅当"逆只作用于单个 rhs 向量"时 Neumann 才划算（那是 kernel 5/6 的形态，
  见 §0.6 裁决段）。
- **C4（Newton）**：op 数与 C1 相当（12 gemm + 24 拷贝 + 6 次 64×64 vop），但 **6 个迭代步每一步都要求
  Cube→Vector→Cube 交接**，2 slot 任务级流水被任务内交接打断 ⇒ 每任务时延 ≈ Σ(1 gemm + 1 vop + 2 flag)
  ≈ 12 µs 以上且不可跨任务重叠 ⇒ 淘汰（结构判据，不依赖未实测常数）。
- **C5/C6 全部落在 Vector**：单 op 最窄（16~64 lane），按 0.5 µs/op 折算 200 op ≈ 100 µs、2016 op ≈ 1.0 ms/任务，
  比 C1 慢 1~2 个数量级，且把 Cube 完全闲置（与算子级"MixCV 主干"结构不符）。

**R4 硬件亲和性逐候选评估**（检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合 / 跨核结构）

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/跨核 | 判定 |
|---|---|---|---|---|---|---|
| B0 | ✗ 对角求逆是 16 步**标量链**，落到 Vector 即 ~120 窄 op；落到 Cube 无法表达（gemm 不做前代） | ✓ | 16 分形对齐 ✓ | ✓ | 16 级 CV 往返 | 淘汰 |
| **C1** | ✓ 全部 MAC 在 Cube（64³ 满利用率）；逐元素仅 11 个宽 op 在 Vector | ✓ UB 122 KB / L1 64 KB / L0C 48 KB | ✓ 64 是分形 16 的整数倍；fp32 尾轴 64×4 B = 256 B | ✓ 6 级在 Python 层静态展开 | ✓ 2 slot 双向握手（`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py` 已验证） | **选定** |
| C2 | ✓ | ✓ | ✓ | △ 需 4 块对角打包 + 索引，代码分支多 | ✓ | 次优（op 数 4.5×） |
| C3 | ✓ | ✓ | ✓ | ✓ | ✗ 63 级 GM 往返 | 淘汰 |
| C4 | ✓ | ✓ | ✓ | ✓ | ✗ 任务内 6 次强制 CV 交接 | 淘汰 |
| C5/C6 | ✗ 矩阵乘落 Vector（每 lane 一元素，无 MAC 阵列） | ✓ | △ 16 lane 宽度浪费 | ✓ | ✓ 无跨引擎 | 淘汰 |

**定量 roofline（D-2，按 `.agents/skills/_shared/standards/hardware-cost-model.md` §2 口径，C1 主选）**

常数来源（版本戳见 §11）：`CONST-vector-launch-overhead`（**向量 op 发射 ~0.5 µs/op**，910B2C，2026-09-09）、
`CONST-fabric-bw`（per-core fabric ≈ 72 GB/s）、`CONST-capacity-910B2C`（UB 192 KB / L1 512 KB / L0C 128 KB）、
`CONST-store-fixpipe-gm-only`（L0C 只能落 GM）；**Cube 侧 0.25 µs/op** 是本任务
`design_probe_doubling_cube.py` PROBE-TIMING 的实测（6 op/级 = 1.52 µs；**新增常数，未蒸馏**）。
**口径声明（本版更正 v0 的误引）**：v0 把 0.25 µs 挂在 `CONST-vector-launch-overhead` 名下，而该条目正文是
~0.5 µs/op；本版把两者分开：Cube 侧 op（gemm/fixpipe/MTE2）= 0.25 µs，Vector 侧计算 op = 0.5 µs。
**16³ gemm 与 64³ gemm 同价（均按 0.25 µs/op）**——这是**未实测假设**（发射开销主导；本版未新测 16³，
B0/C2 的 op 计价因此可能偏贵，但方向性结论不受影响：B0 仅 120 个窄向量 op 按 0.5 µs/op 已 60 µs）。

| 项 | 估算 | 值 |
|---|---|---|
| 流量项 | 425 KB/任务 ÷ 72 GB/s | **5.9 µs/任务** |
| Cube 发射项 | 37 op × 0.25 µs | **9.25 µs/任务** |
| Vector 发射项 | 11 op × 0.5 + 7 拷贝 × 0.25 | **7.25 µs/任务** |
| 容量项 | UB 122 KB/192 KB、L1 64 KB/512 KB、L0C 48 KB/128 KB | 不约束 |
| **估算下界 = max(流量, Cube 发射, Vector 发射)** | | **9.25 µs/任务**（Cube 主导，Vector 余量 1.28×） |
| 合计（`S=3000,H=16` → 752 任务 / 20 核 = 38 任务/核） | | ≈ **0.35 ms** |
| 合计（`S=4096,H=16` → 1024 任务 / 20 核 = 52 任务/核） | | ≈ **0.48 ms** |
| 合计（`S=12288,H=16` → 3072 → 154 任务/核） | | ≈ **1.42 ms** |
| 合计（`S=32768,H=16` → **8192** → 410 任务/核） | | ≈ **3.79 ms** |

> 本版更正：v0 的 `S=32768` 写成 6144 任务 / 3.1 ms（漏算 512×16 = 8192），并与 §5.1、§4.5 自相矛盾；
> 现统一为 8192 任务 / 410 任务每核 / 3.79 ms。op 总数由 v0 的 40 修正为 **55**（v0 漏计 `gemm(I,I)` 播种、
> 落盘链路、取负与掩码 op，并多算了 5 次 `l1q` 重复装载）。

**调研结论**：选定 **C1 = log-doubling 乘积形式**（`A = Π_{i=0..5}(I+N^{2^i})`，`N = -S_sys`，13 个 64³ gemm 含 Gram，
Cube 承担全部 MAC，Vector 只做 11 个宽 op 的掩码/组装）。关键依据：① R1-a/A3 的恒等化简使逐元素相位从
"8 次 64-wide exp + 2016 次除法 + 4 组 fp16 门表"降到"2 次 vbrc + 1 次 vsub + 1 次 vmin + 1 次 vexp + 3 次 vmul"；
② R3 显示 NPU 上**依赖级 op 数**主导（B0 ≈ 66.5 µs vs C1 9.25 µs/任务），为此接受 8.7× MAC；
③ R4 显示 C1 是唯一"全 MAC 在 Cube、无行串行链、无逐级 CV 往返"的候选；④ 精度上 C1 反而最优
（§1.6.1 A4：fp32 全链路 rel ≤ 4.0e-6，A6 逐句展开 rel ≤ 4.6e-4，且源 B0 的 fp16 中间块 rel ≈ 1e-4、
`|A| > 65504` 会溢出）。**B0 的"少算乘加"（O7）与"行串行省 6× 计算"（O6）优化意图被显式替换**
（理由：NPU 上 MAC 便宜、op 贵），其"整块驻留片上"（O1）意图由"GM 往返级数最小化（16 → 13，且 13 级中有
10 级完全在 Cube 内、只与 Vector 交接 2 次）"承接；"越界行 clamp+掩码"（O9）意图由 §0.6 R3 承接。

#### 1.6.1 数学等价优化（公式级）

| 项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|---|---|---|---|
| **A1** 门比值单次指数 | 源式（L388–L427、L429–L505）：按 16 行块参考 `r=16b`，`gate_b[t]=exp2((g_t-g_r)·log2e)`（**存 dtype**），再算 `s = β_i·gate_b[i]/gate_b[j]` | `e[i,j] = exp(g_i - g_j)`（一次 64×64 外差 + 一次 `vexp`） | 数学恒等：`exp(a-r)/exp(b-r) = exp(a-b)`；机器验证 **A1**：`i>j` 区 worst rel 3.0e-8、max ulp 0.5、viol 0（且本式没有源的 fp16 门表舍入，更接近 fp64 语义基准） | 每元素 2 exp + 1 div → 1 exp；实现从"4 组 64-wide exp 表 + 2016 次除法"变成"2 次 `vbrc` + 1 次 `vsub` + 1 次 `vexp`"（-3 op，-1 个 `vdiv` 相位）；同时去掉 fp16 门表中转（精度 +） |
| **A2** 除法→乘倒数（备选形态） | `exp(g_i-r)/exp(g_j-r)` | `exp(g_i-r)·exp(r-g_j)` | 恒等（`1/exp(x)=exp(-x)`；一次倒数 vs 2016 次除法）；机器验证 **A2**：worst rel 1.2e-07、max ulp 1.3、viol 0 | 主选路径（A1）已无除法；该形态作为"若 64×64 `vexp` 代价超预期"的退路，op 数 3 vs 4、省 2016 次除法 |
| **A3** β 保持 fp32 | `beta_s[t] = cast(step, dtype)`（L295），后续按 fp16 值参与乘 | β 先 `vcast` 到 fp32，其后全程 fp32（`vbrc` 到 64×64 后直接乘） | 容差内等价：差异恰为 1 次 fp16 舍入（≤1 ulp）；机器验证 **A3**：worst rel 3.7e-4、viol 0 | 少一次舍入（精度 +），成本 = 1 个 64-wide `vcast` |
| **A4** 求逆用 log-doubling（含 R1-d 的符号折进 β） | `A = Σ_{j<64} N^j`（源用分块前代，等价于级数截断） | `A = (I-N)^{-1} = Π_{i=0}^{5}(I + N^{2^i})`，`N = -S_sys`；`T_{j+1}=T_j + P_jT_j`（`initC=False` 累加）、`P_{j+1}=P_j²`；`-1` 由 **β 的一次标量取负**落实（`vmul(β, -1)`），`I` 由 `gemm(I,I)` 初始化 | 数学恒等：`N` 严格下三角 ⇒ `N^64=0`；`{1,2,4,8,16,32}` 的子集和唯一覆盖 `0..63` ⇒ 乘积展开 = 级数；机器验证 **A4**：worst rel 4.0e-6、max ulp 1.8、viol 0（含 `β×20`、`k×4`、尾 chunk、零 β 行等角点） | 依赖级 gemm：26（B0）/ 63（Neumann）→ **13（含 Gram）**；无 16 步行串行链；fp32 全链路（源 fp16 中间块 rel≈1e-4，`|A|>65504` 时溢出） |
| **A5** 单次量化直出 | 中间逆块存 dtype（`solve_dtype=dtype`，L200/L242），逐块 cast | 链全程 fp32，`fixpipe` 落 `y_blk` 时做 f32→dtype 量化一次（后续同 dtype 搬到 A，不再舍入） | 容差内等价：仅保留最终一次舍入；机器验证 **A5**：worst rel 9.3e-5、max ulp 1.0、viol 0 | 少 10 次 fp16 中间存储/读回（≈ -160 KB GM 流量/任务）、去掉 fp16 溢出面 |
| **A6** 伪代码逐句展开（**本版新增，D-1 的"符号类错误"护栏**） | §3.3 的字面 Vector 序列：`vbrc(g_col)→g_i`、`vbrc(g_row)→g_j`、`vsub(g_i,g_j)`、`vmin(d,0)`、`vexp`、`vcast(β)+vmul(-1)`、`vmul`、`vmul(G)`、`vmul(mask01)`，紧接 6 级 doubling | 同上（**不是**新优化项，而是把 A4 的输入 `N` 也纳入机检） | 与 `inv(I+S_sys)` 的 fp64 解对比；机器验证 **A6**：worst rel 4.6e-4、viol 0 | 使"广播方向写反 / 负号丢失"这类**只在可执行层出现**的缺陷进入 D-1 机检；反例对照（同判据）显示 v0 的方向缺陷 rel 1.3e-1/viol 3.7e-1、符号缺陷 rel 4.4e-1/viol 4.4e-1 ⇒ **A6 有牙齿** |

**机器验证结果表**（`examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py` 实跑 2026-10-17，torch CPU，fp64 参照；
判据 `|cand-ref| <= 5e-3·|ref| + 2·eps_fp16·max|ref|`，viol = 违反率）：

| case | A1 exp(a-b) rel/viol | A2 mul rel/viol | A3 beta-fp32 rel/viol | A4 doubling rel/viol | A5 fp16-cast rel/viol | A6 literal-chain rel/viol |
|---|---|---|---|---|---|---|
| SC-typical | 3.0e-08 / 0.0e+00 | 1.2e-07 / 0.0e+00 | 2.2e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 4.8e-05 / 0.0e+00 |
| SC-strong-decay | 3.0e-08 / 0.0e+00 | 6.0e-08 / 0.0e+00 | 1.4e-04 / 0.0e+00 | 7.1e-09 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 2.2e-05 / 0.0e+00 |
| SC-no-decay | 3.0e-08 / 0.0e+00 | 6.0e-08 / 0.0e+00 | 1.8e-04 / 0.0e+00 | 2.4e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 4.3e-05 / 0.0e+00 |
| SC-beta-x20 | 3.0e-08 / 0.0e+00 | 1.2e-07 / 0.0e+00 | 3.7e-04 / 0.0e+00 | 4.0e-06 / 0.0e+00 | nan / 0.0e+00 | 3.1e-04 / 0.0e+00 |
| SC-k-x4 | 3.0e-08 / 0.0e+00 | 1.2e-07 / 0.0e+00 | 2.7e-04 / 0.0e+00 | 1.4e-06 / 0.0e+00 | 9.3e-05 / 0.0e+00 | 4.6e-04 / 0.0e+00 |
| SC-tail-56 | 3.0e-08 / 0.0e+00 | 1.2e-07 / 0.0e+00 | 2.6e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 5.7e-05 / 0.0e+00 |
| SC-zero-beta-rows | 3.0e-08 / 0.0e+00 | 1.2e-07 / 0.0e+00 | 2.7e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 5.3e-05 / 0.0e+00 |
| SC-near-overflow | 2.9e-08 / 0.0e+00 | 6.0e-08 / 0.0e+00 | 2.6e-04 / 0.0e+00 | 5.0e-09 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 2.4e-05 / 0.0e+00 |

| 项 | worst rel | worst ulp | worst viol | 结论 |
|---|---|---|---|---|
| A1 | 3.0e-08 | 0.5 | 0.0e+00 | **EQUIV_PASS** |
| A2 | 1.2e-07 | 1.3 | 0.0e+00 | **EQUIV_PASS** |
| A3 | 3.7e-04 | 3957.5 | 0.0e+00 | **EQUIV_PASS** |
| A4 | 4.0e-06 | 1.8 | 0.0e+00 | **EQUIV_PASS** |
| A5 | 9.3e-05 | 1.0 | 0.0e+00 | **EQUIV_PASS** |
| A6 | 4.6e-04 | 310.8 | 0.0e+00 | **EQUIV_PASS** |

> `SC-beta-x20` 的 A5 出现 `nan`：该 case `|A| ≈ 3e4`，fp16 输出上溢（`> 65504` 的中间逆块）——这是**源算法**的
> 溢出边界（源把逆块存 fp16），本设计 fp32 链路不受影响（A4/A6 该 case 正常）；列在表内仅为记录该边界。
> `A3` / `A6` 的 worst ulp 偏大是因为 ULP 以 fp32 尺度度量 fp16 量化量（3957 fp32-ulp ≈ 1 fp16-ulp），viol=0 才是判据。
> **符号裁决（脚本末尾，fp64 结构通路）**：源分块算法与 `inv(I+S_sys)` 差 **2.776e-17**、与 `inv(I-S_sys)` 差
> **4.361e-01** ⇒ `N = -S_sys` **CONFIRMED**。**反例对照**（A6 判据）：方向写反 ⇒ viol 3.7e-01；符号写反 ⇒ viol 4.4e-01。

**优化结论**：采纳 A1/A3/A4/A5（A2 作为退路形态一并验证通过），并把 A6 作为"A4 输入构造"的等价性护栏：
六项均 `EQUIV_PASS`、viol 全 0。其中 A4 是算法级恒等（`Σ N^j` → 乘积形式），A1/A3/A5 是"更少 op / 更少舍入"的
公式级优化；A1 使逐元素相位从 **~2016 次除法 + 8 次窄 exp** 降到 **2 次 vbrc + 1 次 vsub + 1 次 vmin + 1 次 vexp**（4096 元素宽 op）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案（§3.3 伪代码）中**全部**循环与标量计算点：

| # | 循环 / 标量点 | 位置 | 处置 | 依据 |
|---|---|---|---|---|
| V1 | 任务→`(chunk, head, base, valid, slot)` 映射（`t//head`、`t%head`、`it%2`） | 两 scope 的任务循环头 | **保留标量** | block 级索引 / 元数据（每任务 ~6 条标量，非逐元素热点）；等价向量化无意义 |
| V2 | `N` 逐元素构造（源 L429–L505 的 `for i,j in T.Parallel(16,16)` 共 6 次 + 4 个对角块） | Vector | **已向量化**（11 个宽 op，全链路见 §3.3）：`vcast(β)` → `vmul(β,-1)` → `vbrc(g_col)→g_i` → `vbrc(g_row)→g_j` → `vsub(g_i,g_j)` → `vmin(d,0)` → `vexp` → `vbrc(-β)` → `vmul` → `vmul(G)` → `vmul(mask01)` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（(M,1)/(1,N)→(M,N) 广播合法，rank 必须相同）、`docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1（fp32 ✓）、`docs/Tilelang.language/比较操作/T.vmin.md` §2.2.2（同形 + 标量 ✓）；**广播方向与符号已真机对拍**（`design_probe_gate_sign.py`） |
| V3 | 源的门参考表 + 每元素除法（L388–L429、L429–L505 的 2016 次 `/`） | Vector | **已消除**（A1：`exp(g_i-g_j)`） | §1.6.1 A1（EQUIV_PASS）；`vdiv` 相位整体消失 |
| V4 | 4 个 16×16 对角块行串行前代（源 L512–L540：`for r in T.unroll(16)` × 内层 `for m in T.unroll(r)`） | — | **已消除**（算法级） | 被 A4 的 6 级 gemm 链替代；无行串行链、无 `T.alloc_var` 标量累加 |
| V5 | **16** 次块前代 gemm + `work_s` 搬运（源 L542–L620） | — | **已消除**（算法级） | 同一 64×64 逆由 12 个链 gemm（+1 个播种）给出（`T.gemm` 原生，无需逐块搬 UB） |
| V6 | 输出逐行块写回 + 逐行 `if base+i < end`（源 L622–L651） | Cube + Vector | **已消除逐行分支，但落盘改为 3 步**（本版更正 v0 的"一次 `store_fixpipe` 落 64 行"）：① Cube `fixpipe` → `y_blk[slot]`（连续 64×64，量化一次）；② Vector `T.copy(y_blk → UB)`；③ Vector `T.copy(UB → A_pad[base:base+64, hid:hid+1, 0:64])`（一次 strided region 写）。边界仍由"chunk 对齐 padding + 返回视图"承担 | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1（src 2D、dst 2-5D、rank 不一致时 size 对齐最后维 ⇒ 直写 A 的 4D 切片会覆盖 `(head,64)` 平面，v0 的错误）；`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2（静态 1 维折叠、rank 可不同）；真机实测 `design_probe_store_layout.py`：目标行区逐元素正确、相邻 (token,head) 槽零污染 |
| V7 | 越界行 clamp（源 L288–L298、L317–L324 的逐行 `min(base+t, end-1)`） | Vector/Cube | **已消除** | 改为动态 extent 装载（`valid` 行）+ 0/1 常量掩码乘法；`T.copy` 动态 extent 有仓内先例（`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py` 的 L237–L254） |
| V8 | 6 级 doubling 的 `j` 循环 | Cube | **保留为 Python 静态展开**（不生成 TIR 循环） | 每级读写不同 slot/buffer 且末级无平方，展开后边界全静态（`.agents/skills/_shared/standards/core-split-strategy.md` §1 要素③"循环边界静态"）；不用 `T.serial` 也可避免"循环携带 Cube 依赖"这一历史风险面（`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §10） |
| V9 | 掩码 `(i>j)` 构造 | Vector | **本版变更：`T.arange`→`vcmp` 路径实测不可用（静默产出全 0 掩码），改为 host 常量 0/1 tile + `vmul`（每 block 装载一次）** | 微探针 `history_version/design_probe_mask.py`：`arange(1-D)→reshape→vbrc→vcmp→vselect` 三形态（单/双 arange）均得全 0 掩码；`vmin`+标量运算与"GM 常量 tile"路径精确（0 误差）。`dim_k` 无关，16 KB 常量 tile 每 block 一次（摊到 ~400 任务 ≈ 0.04 KB/任务） |
| V10 | Gram 的 `dim_k` 规约 | Cube | **不适用向量化**：一次 `T.gemm(size=[64,dim_k,64], b_transpose=True)` 交给 Cube 分形 | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.4；`dim_k ∈ {64,128}` 均满足 K 维分形 16 对齐 |
| V11 | `β` 的 fp16/bf16 → fp32 提升（源 L295 直接 `cast` 到 dtype 后参与乘） | Vector | **1 个 64-wide `vcast`（`round_mode="rint"` 关键字形式）** | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16→f32 仅 `rint`）；Expert 模式 `T.copy` 禁止跨 dtype（`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 2 条），必须显式 cast；**签名陷阱**：第 3 个位置参数是 `size` 不是 `round_mode`（`design_probe_vcast_shapes.py` 实测：位置写法直接编译失败） |

**向量化结论**：逐元素相位（V2/V3/V7/V11）全部落在 64-wide 或 64×64 的 v-prefix 宽 op 上（共 **11 个宽 op/任务**），
源的三处**标量/逐行点**（V4 的 16 步行串行、V6 的逐行写回、V7 的逐行 clamp）由算法重设计与落盘重构整体消除；
仅 V1（block 级索引）与 V9 的常量 tile 装载保留在"block 级索引 / 常数物化"类别，不构成逐元素热点。

#### 1.6.3 向量化轴与数据布局决策

**I/O 契约（不可变）**：`k[1,S,kh,dim_k]`、`g/beta[1,S,head]`、`A[1,S,head,64]`（尾轴 64 连续）。
**核内布局与落盘路径是设计变量**，按 MixCV/Cube 类清单枚举（含 epilogue 归属侧）：

| 候选 | 形态 | 整除性/对齐 | repack / 落盘代价 | 预算 | 结论 |
|---|---|---|---|---|---|
| **L0 原生长宽 + 行主序 64×64 + 连续中转落盘（选定）** | Gram/链一律 64×64 fp32 行主序（尾轴 64 = 256 B）；Cube 侧 L1 操作数 64×64 fp32；A 经 `y_blk`（连续 64×64、量化一次）→ UB → **一次 strided region 写**进 `A_pad[base:base+64, hid, :]` | 64 = 分形 16 整数倍；fp16 行段 128 B = 1 个 DMA 段；行距 `head·64·2 B`（`head=16` 时 2048 B，128 B 对齐） | 零重排；+16 KB GM 流量/任务（y 写 8 + 读 8）；**实测**：连续 8 KB 写 **0.084 µs**、strided 写 **0.194 µs**（2.29×，`head=4`/512 B 行距）⇒ 落盘相位 ≈ +0.3 µs/任务 | UB 122 KB；L1 64 KB；L0C 48 KB | **采纳** |
| L1 `k` 转置后按 `dim_k` 作向量轴 | 把 64×dim_k 转成 dim_k×64 | dim_k 64/128 ✓ | 需核内 `T.transpose` 链（每次 64×128 = 8 K 元素的 VTransposeOp） | +32 KB UB | 弃：Gram 走 Cube 分形，`dim_k` 不需要成为向量轴；转置纯增代价（`docs/Tilelang.language/创建操作/T.transpose.md` §2.3：仅双轴交换、bf16 不支持） |
| L2 链矩阵用 fp16 scratch | 中间 T/P 存 fp16（操作数也随之 fp16） | ✓ | 零 repack，但每级 1 次量化 | UB/L1 减半、GM 流量 −50%（425 → 265 KB，流量项 5.9 → 3.7 µs） | 弃（主选）：精度面；**本版实测**（`design_probe_gate_sign.py` PROBE-CHAIN，2 级链）：fp32 操作数 rel **5.5e-9** vs fp16 操作数 rel **2.8e-5**，6 级链的 fp16 误差会再累积 ⇒ 只作 Stage 4 备选（见 §9 R1 的阈值与 §11 对 v0 "V4 err=1.875" 的判废说明） |
| L3 两个 head 打包成 128×128 块对角链 | 摊薄每 head 的 op 数 | ✓ | 需 128×128 块对角打包/掩码 | L0C 2×64 KB = 128 KB = 上限 | 弃（主选）：L0C 顶格，无余量给 G/T/P 三缓冲；列为 Stage 4 结构优化候选 |
| L4 `A` 先落 `[head, S_pad, 64]`（head-major）再由 host `permute+.contiguous()` | 让每个任务的 64×64 tile 在 GM 上连续（一次 `store_fixpipe` 直出） | ✓ | **+1 遍 A 级拷贝**：`S=32768,H=16` 时 64 MB 读 + 64 MB 写 ≈ 128 MB ÷ 1.73 TB/s ≈ **74 µs**（占 3.79 ms 的 2%），峰值 +64 MB；且返回张量必须先 `.contiguous()` 才满足下游"连续 `[1,T,head,64]`"契约 | — | 弃（备选）：**v0 的弃选理由错误**——"`A` 的 `[head,64]` 切片本就连续"只对**单个 token**成立；固定 `hid` 时 64 个 token 行的行距是 `head·64` 元素，**不是**连续块。本版按真实代价（+74 µs、+64 MB）把它降为 L0 落盘的回退路径（§9 R14） |
| L5 `A` 逐行 `store_fixpipe`/逐行 `T.copy` | 64 次 128 B 段写 | ✓ | 64 次窄 op（每行 1 段，段级延迟完全不可摊销） | — | 弃：按实测 `contig 0.084 µs / strided-region 0.194 µs` 与 PL-1.14 的段级律外推，逐行版本 ≈ 64×(0.084+launch) ≥ 5.4 µs/任务（region 写只 0.194 µs）⇒ 仅作 L0 的回退（§9 R14） |

**落盘路径的真机证据（本版新增，`history_version/design_probe_store_layout.py`）**：

```
Cube : gemm(I, x) -> L0C(fp32) --fixpipe--> y16[1,64,64] (fp16 连续)
Vector: y16 -> UB --T.copy--> A3[0:64, hid:hid+1, 0:64]   （A3 形如 [S_pad, head, 64]）
  PROBE-STORE-STRIDED 目标行区 == x.half()      : max_abs_err = 2.419e-04 (= x 的 fp16 舍入；fixpipe 量化本身 err = 0)
  PROBE-STORE-STRIDED 相邻 (token, head) 槽洁净  : max|other| = 0.000e+00
  PROBE-STORE-TIMING  contig 0.084 µs/8 KB, strided 0.194 µs/8 KB (2.29×, head=4/512 B 行距)
```

**布局决策结论**：采纳 **L0**——Cube 侧一律 64×64 fp32 行主序（`k` 片经 `T.copy` slice 进 L1、`b_transpose=True`），
Vector 侧 64×64 fp32 行主序（向量轴 = 尾轴 64 lane，256 B 对齐）；`A` 的落盘走"**连续中转 + 一次 strided region 写**"
（`head` 轴以 size-1 参与 region，靠 `T.copy` 的静态 1 维折叠变成"64 行 × 64 连续元素、行距 head·64"），
**不做任何核内整块重排**。fp16 链（L2）与双 head 打包（L3）留在 Stage 4 作为流量/时延备选，
head-major + host 重排（L4）与逐行写（L5）作为 L0 落盘的回退路径，裁决阈值见 §9。

---
## 2. 编程模式选型

### 2.1 模式结论

**Expert**（同一 kernel 内一个 `T.Scope("Cube")` + 一个 `T.Scope("Vector")`，手动 L1/L0C/UB 分配与手动同步）。

### 2.2 选型理由

1. 本函数必须同时用 Cube（Gram + 12 个链 gemm）与 Vector（门/β/掩码 11 个宽 op + 落盘 3 步搬运），是 MixCV；
2. CANN 9.0 实测：Developer 模式下"Cube+Vector 同 kernel"会走错流水（`hivm.hir.store` 约束、
   `multiple entry functions detected`），Expert 下 `Conv3dBiasFwdOp` 17/17 通过
   （`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12.0/§12.1）；
3. 单 kernel 只能有 **一个** Cube scope + **一个** Vector scope（`EXPERT_FINDINGS.md` §12.2），
   因此两引擎必须各自跑"整段任务循环"并用 flag 握手（§7），这只有 Expert 能表达。

### 2.3 模式强制方式（必须写死，不得 `setdefault`）

- **进程级**：模块导入时 `os.environ["TILELANG_ASCEND_MODE"] = "Expert"`（**赋值**，非 `setdefault`）——
  `.bashrc` 导出的是 `Developer`，`setdefault` 静默失效曾导致 11 个探针得出 3 条假限制
  （`EXPERT_FINDINGS.md` §12.0）。本版 4 个探针脚本同样在 import tilelang 前强制赋值。
- **调用级**：kernel 类声明 `ascend_mode = "Expert"`；`Kernel.__call__` 会用
  `scoped_ascend_mode(self.ascend_mode)` 包住首次调用（三个 env 读取点都在 scope 内），
  避免同进程混跑 Developer 内核时模式串味（`examples/TileOPs/tileops/utils/ascend_mode.py`，
  CG-2026-0010 的既定做法）。
- **测试级**：`gdn_prefill.py --level all` 入口同样先赋值 env，再 import tilelang。

### 2.4 模式影响

- 所有 buffer 必须显式指定层级：`T.alloc_L1`（Cube 操作数）、`T.alloc_L0C`（fp32 累加器）、
  `T.alloc_ub`（Vector）。
- 所有 L0C→GM 必须包在 `with T.rs("PIPE_FIX")` 内（否则 lower 成 `hivm.hir.store` 被拒，
  `examples/gdn_prefill/probe/experiments/07_c1_cube_cann9.py` 头部注释）。
- 跨引擎交接必须显式 `sync_block_set/wait`（`EXPERT_FINDINGS.md` §2 第 3 条）。

---

## 3. API 映射设计

### 3.1 公式拆解

| 步 | 公式（§1.6.1 优化后） | 说明 |
|---|---|---|
| S1 | `t = bid + it·K`；`chunk = t // head`、`hid = t % head`、`khid = hid // group`、`base = chunk·64`、`valid = min(64, end-base)` | 任务映射（V1，标量） |
| S2 | `G = k_片 · k_片ᵀ`（64×64，K 维 = `dim_k`） | 一次 Cube gemm |
| S3 | **`d[i,j] = g_i - g_j`**（`vbrc` 列 → `g_i`、`vbrc` 行 → `g_j`，**被减数是列广播**）；`d ← min(d, 0)`；`e = exp(d)`；**`bn = -β_i`（`vcast` 后一次标量乘 −1）**；`N = mask01 ⊙ bn_i·e·G` | Vector；`N` 的负号**唯一落点**在 `bn`；含 §0.6 R3 的掩码语义 |
| S4 | `T ← I`；`P ← N` | `T` 由 `gemm(I,I)` 给出（L0C 只能被 gemm 初始化；见 §3.2） |
| S5 | `T ← T + P·T`（`initC=False` 累加进 L0C） | 6 次（lev 0..5），lev 0 的 B 操作数直接用单位阵 |
| S6 | `P ← P·P` | 5 次（P_1..P_5；不需要 `N^64`）；**同一 L1 buffer 同时作 A/B（已实测）** |
| S7 | `A` 落盘：`l0t --fixpipe--> y_blk[slot]`（f32→dtype，连续 64×64）→ `y_blk → UB` → `T.copy(UB, A_pad[base:base+64, hid:hid+1, 0:64])` | 3 步（§1.6.3 实测路径）；`A_pad` 为 `[S_pad, head, 64]` |
| S8 | `A = A_pad[:total_tokens].unsqueeze(0)` | host 纯视图，返回 `[1,T,head,64]` 连续 |

> **S3 的方向自查**（REVIEW v0 问题 1 的根因）：`T.vbrc` 的语义是"src 的 1 维沿该轴复制到 dst 的全长"，
> 故 `vbrc((64,1))[i,j] = src[i,0] = g_i`（**列向量 → 每行同值 = `g_i`**），
> `vbrc((1,64))[i,j] = src[0,j] = g_j`（**行向量 → 每列同值 = `g_j`**）。
> 因此 `vsub(g_i_mat, g_j_mat)` 得 `g_i - g_j` ✓；v0 把 `g_row`（= `g_j`）放在被减数位，实际算出 `g_j - g_i`。
> 另据 `VP-2026-0087`（`.agents/evolution/queue.md`）：`vmul/vsub/vadd` 的文档化 Shape 只有四形、
> **不含 `[M,N] * [1,N]`**，且 `M==N` 时方向写反**不报 shape 错、静默算错**——所以本设计**两个操作数都先
> `vbrc` 到 64×64 全形**再相减，不依赖任何隐式行/列广播。该序列已在
> `history_version/design_probe_gate_sign.py` 上真机对拍（见 §3.4）。

### 3.2 TileLang API 映射

| 步 | API | 参数要点 | 佐证 |
|---|---|---|---|
| S2 装载 | `T.copy(k[0, base:base+valid, khid, 0:dim_k], l1k[0:valid, 0:dim_k])` | 动态 extent（`valid`）；**必须用 `T.copy` 的 slice 形态，不得用 `T.load_nd2nz`**（固定 head 的跨步 tile 在 load_nd2nz 下会静默平坦误读） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 1/4；`.agents/evolution/capability-gaps.md` CG-2026-0004 + `TRAP-load-nd2nz-strided`；先例 `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（L237–L254） |
| S2 gemm | `T.gemm(l1k, l1k, l0g, size=[64, dim_k, 64], b_transpose=True, initC=True)` | fp16/bf16 输入、fp32 累加；`dim_k ∈ {64,128}`；同一 L1 同时作 A/B（`b_transpose=True` 已有交付内核先例） | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.3（fp16×fp16 必须 fp32 累加）、§2.4；`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py` L43 |
| S2 落盘 | `with T.rs("PIPE_FIX"): T.copy(l0g, gs_G[bid, slot])` | f32→f32；dst 是 4D buffer 的 `[slot]` 平面（index access + src extents） | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1/§2.4 |
| S4 初始化 | `T.gemm(l1e, l1e, l0t, size=[64,64,64], initC=True)`（`l1e` = 64×64 fp32 单位阵，host 常量） | 给 L0C 一个初值（GM→L0C 无直连通路） | `docs/Tilelang.language/内存操作/T.copy.md` §1 方向表（无 GM→L0C）；探针 `design_probe_doubling_cube.py`（T1 = I+N 实测 err 2.9e-8） |
| S5/S6 | `T.gemm(l1p, l1t, l0t, size=[64,64,64], initC=False)` / `T.gemm(l1p, l1p, l0p, size=[64,64,64], initC=True)` | `initC=False` = 累加进当前 L0C；`P²` 用**同一个 `l1p` 作 A 和 B**（本版实测 rel 5.5e-9） | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1；`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1；本设计 `design_probe_gate_sign.py` PROBE-CHAIN |
| S3 广播 | `T.vbrc(ub_gc(64,1), ub_gi(64,64))`、`T.vbrc(ub_gr(1,64), ub_gj(64,64))`、`T.vbrc(ub_bn(64,1), ub_bm(64,64))` | rank 必须相同（src 的 1 维沿该轴复制）；**不得**把 `(1,N)` 直接喂给 `vsub` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；`VP-2026-0087`；本版微探针 M3/M4（`(64,1)`/`(1,64)` 广播 err = 0） |
| S3 差值/限幅/指数 | `T.vsub(ub_gi, ub_gj, ub_gi)`、`T.vmin(ub_gi, T.float32(0.0), ub_gi)`、`T.vexp(ub_gi, ub_e)` | 全部 fp32、同 shape；`vmin` 限幅把丢弃区指数压在 `e ≤ 1`（不出 inf） | `docs/Tilelang.language/数学操作/T.vsub.md`、`T.vmin.md` §2.2.2（同形 + 标量）、`T.vexp.md` §2.2.1 |
| S3 β 提升与取负 | `T.vcast(ub_b16, ub_b32, round_mode="rint")`；`T.vmul(ub_b32, T.float32(-1.0), ub_bn)` | **`round_mode` 必须是关键字**（第 3 个位置参数是 `size`）；标量乘是 `vmul` 文档化四形之一 | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；`tilelang/language/customize_npuir.py` 的 `npuir_cast(src, dst, size=None, round_mode="rint")`；本版微探针 M1/M5 + `design_probe_gate_sign.py` |
| S3 乘与掩码 | `T.vmul(ub_bm, ub_e, ub_t1)`、`T.vmul(ub_t1, ub_G, ub_t2)`、`T.vmul(ub_t2, ub_mask01, ub_n)` | `ub_mask01` = host 常量 0/1 严格下三角（每 block 装载一次）；`0 × 有限 = 0` 精确 | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2；微探针 MB（常量 tile 精确） |
| S3 回写 | `with T.rs("PIPE_MTE3"): T.copy(ub_n, ws_S[bid, slot]); T.sync_block_set(F_S + slot)` | flag id = 常量基址 + `slot`（TIR 表达式） | `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`；动态 id 先例 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` L351 `T.sync_block_set(i)` |
| S7 落盘 | `with T.rs("PIPE_FIX"): T.copy(l0t, y_blk[bid, slot])`；`T.copy(y_blk[bid,slot], ub_y)`；`with T.rs("PIPE_MTE3"): T.copy(ub_y, A_pad[base:base+64, hid:hid+1, 0:64])` | f32→dtype 量化只在 fixpipe；后续同 dtype 搬运不再舍入；strided region 靠静态 1 维折叠成"64 行 × 64 连续元素" | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1（四种合法 dtype 配对含 f32→f16/bf16）；`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 1；实测 `design_probe_store_layout.py` |
| 同步 | `T.sync_block_set(F + slot)` / `T.sync_block_wait(F + slot)` + `with T.rs("PIPE_FIX"/"PIPE_MTE2"/"PIPE_MTE3")` | 6 个 flag id（§7） | `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md`；形态 `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py` |

### 3.3 计算伪代码

```python
# ---- host（wrapper）----
head, group = H, H // key_head
S_pad  = ceil(total_tokens / 64) * 64
A_pad  = torch.empty(S_pad, head, 64, dtype=k.dtype, device=k.device)      # 3D: [token, head, 64]
eye    = torch.eye(64, dtype=torch.float32, device=k.device)               # 常量输入
mask01 = torch.tril(torch.ones(64, 64, dtype=torch.float32, device=k.device), -1)  # 严格下三角 0/1
gs_G   = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)   # Gram 交接 2 slot
ws_S   = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)   # N     交接 2 slot
ws_T   = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)   # T_j   轮转 2 slot
ws_P   = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)   # P_j   轮转 2 slot
y_blk  = torch.empty(K, 2, 64, 64, dtype=k.dtype,    device=k.device)      # A 的量化中转（连续）
kernel(k, g, beta, eye, mask01, gs_G, ws_S, ws_T, ws_P, y_blk, A_pad)
A = A_pad[:total_tokens].unsqueeze(0)          # [1,T,head,64]，连续视图（纯元数据）
return A, k_rnorm                              # k_rnorm 占位（l2norm=False）

# ---- kernel ----
F_G, F_S, F_A = 0, 2, 4        # flag id 基址：f_g = F_G+slot, f_s = F_S+slot, f_a = F_A+slot（共 6 个）
TASKS = nch_total * head        # 编译期常量
NPER  = ceil(TASKS / K)         # 编译期常量
seq_end = total_tokens          # in-scope 等长：单序列时即 end（num_sequences>1 见 §9 R11）

def gram(tt, gslot):                      # 每任务一次 Gram + 落盘 + set
    b = (tt // head) * 64
    vv = T.min(64, seq_end - b)           # 等长单序列：seq_end = S；多序列见 §9 R11
    kk = (tt % head) // group
    T.copy(k[0, b:b + vv, kk, 0:dim_k], l1k[0:vv, 0:dim_k])
    T.gemm(l1k, l1k, l0g, size=[64, dim_k, 64], b_transpose=True, initC=True)
    with T.rs("PIPE_FIX"):
        T.copy(l0g, gs_G[bid, gslot])
        T.sync_block_set(F_G + gslot)

def scatter(tt, pslot, bid):                # Vector 侧：连续中转 → strided region 写
    pb = (tt // head) * 64
    ph = tt % head
    with T.rs("PIPE_MTE2"):
        T.sync_block_wait(F_A + pslot)
        T.copy(y_blk[bid, pslot], UB_Y)
    with T.rs("PIPE_MTE3"):
        T.copy(UB_Y, A_pad[pb:pb + 64, ph:ph + 1, 0:64])        # 64 行 × 64 连续元素，行距 head·64

with T.Kernel(K, is_npu=True) as (bid, _):
    # ============= Cube：Gram + 6 级 log-doubling（5 次平方）=============
    with T.Scope("Cube"):
        l1k = T.alloc_L1([64, dim_k], dtype)      # k 片
        l1e = T.alloc_L1([64, 64], "float32")     # 单位阵常量（T_0 / lev0 的 B 操作数）
        l1p = T.alloc_L1([64, 64], "float32")     # P_j（同时作 P² 的 A 与 B）
        l1t = T.alloc_L1([64, 64], "float32")     # T_j（gemm B 操作数）
        l0g = T.alloc_L0C([64, 64], "float32")
        l0t = T.alloc_L0C([64, 64], "float32")    # T 累加器
        l0p = T.alloc_L0C([64, 64], "float32")
        T.copy(eye[0:64, 0:64], l1e)
        gram(bid, 0)                              # 前导：任务 t0，slot=0
        for it in T.serial(NPER):
            t = bid + it * K
            slot = it % 2
            if t < TASKS:
                if it + 1 < NPER and t + K < TASKS:
                    gram(t + K, 1 - slot)         # 提前一拍算下一任务的 Gram
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(F_S + slot)
                    T.copy(ws_S[bid, slot], l1p)                      # P_0 = N
                T.gemm(l1e, l1e, l0t, size=[64, 64, 64], initC=True)  # T_0 = I（L0C 的播种）
                for lev in range(6):                                  # Python 静态展开，非 TIR 循环
                    if lev == 0:
                        T.gemm(l1p, l1e, l0t, size=[64, 64, 64], initC=False)   # T_1 = I + N
                    else:
                        T.copy(ws_T[bid, lev % 2], l1t)                          # T_lev
                        T.gemm(l1p, l1t, l0t, size=[64, 64, 64], initC=False)   # T += P·T
                    if lev < 5:
                        T.gemm(l1p, l1p, l0p, size=[64, 64, 64], initC=True)    # P ← P²（A/B 同 buffer）
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, (lev + 1) % 2])              # T_{lev+1}
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, (lev + 1) % 2])              # P_{lev+1}
                        T.copy(ws_P[bid, (lev + 1) % 2], l1p)
                with T.rs("PIPE_FIX"):
                    T.copy(l0t, y_blk[bid, slot])                  # f32->dtype 量化，连续 64×64
                    T.sync_block_set(F_A + slot)

    # ================= Vector：g/β/掩码 → N；并把上一任务的 A 落盘 ================
    with T.Scope("Vector"):
        MK   = T.alloc_ub([64, 64], "float32")    # 静止常量 0/1（每 block 一次）
        UB_G = T.alloc_ub([64, 64], "float32")    # G
        UB_A = T.alloc_ub([64, 64], "float32")    # g_i → d → (复用)
        UB_B = T.alloc_ub([64, 64], "float32")    # g_j
        UB_E = T.alloc_ub([64, 64], "float32")    # e
        UB_F = T.alloc_ub([64, 64], "float32")    # bm → t1
        UB_N = T.alloc_ub([64, 64], "float32")    # t2 → N
        UB_Y = T.alloc_ub([64, 64], dtype)        # A tile 中转
        ub_gc  = T.alloc_ub([64, 1], "float32")   # g_i 源（(64,1) 落位）
        ub_gr  = T.alloc_ub([1, 64], "float32")   # g_j 源（(1,64) 落位）
        ub_b16 = T.alloc_ub([64, 1], dtype)       # β（fp16/bf16 原样）
        ub_b32 = T.alloc_ub([64, 1], "float32")
        ub_bn  = T.alloc_ub([64, 1], "float32")   # -β
        T.copy(mask01[0:64, 0:64], MK)

        for it in T.serial(NPER):
            t = bid + it * K
            slot = it % 2
            if t < TASKS:
                b = (t // head) * 64
                vv = T.min(64, seq_end - b)
                hd = t % head
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(F_G + slot)
                    T.copy(gs_G[bid, slot], UB_G)
                    T.copy(g[0, b:b + vv, hd:hd + 1],    ub_gc[0:vv, 0:1])
                    T.copy(g[0, b:b + vv, hd:hd + 1],    ub_gr[0:1, 0:vv])   # 同区、不同落位 → g_j
                    T.copy(beta[0, b:b + vv, hd:hd + 1], ub_b16[0:vv, 0:1])
                T.vcast(ub_b16, ub_b32, round_mode="rint")
                T.vmul(ub_b32, T.float32(-1.0), ub_bn)          # ← N 的负号在此落实（1 个 64-wide op）
                T.vbrc(ub_gc, UB_A)                             # UB_A[i,j] = g_i
                T.vbrc(ub_gr, UB_B)                             # UB_B[i,j] = g_j
                T.vsub(UB_A, UB_B, UB_A)                        # d = g_i - g_j
                T.vmin(UB_A, T.float32(0.0), UB_A)              # 丢弃区限幅（e ≤ 1，无 inf）
                T.vexp(UB_A, UB_E)                              # e = exp(d)
                T.vbrc(ub_bn, UB_F)                             # UB_F[i,j] = -β_i
                T.vmul(UB_F, UB_E, UB_F)                        # -β_i · e
                T.vmul(UB_F, UB_G, UB_N)                        # · G
                T.vmul(UB_N, MK, UB_N)                          # · mask01 = N
                with T.rs("PIPE_MTE3"):
                    T.copy(UB_N, ws_S[bid, slot])
                    T.sync_block_set(F_S + slot)
                if it >= 1:                                     # 一任务滞后落盘（保持 2 slot 流水）
                    scatter(t - K, (it - 1) % 2, bid)
        if bid + (NPER - 1) * K < TASKS:                        # 尾部：最后一个任务
            scatter(bid + (NPER - 1) * K, (NPER - 1) % 2, bid)

```

> **可执行性自查（REVIEW v0 问题 7 的四处，逐条落点）**：① flag 不再用 Python list 下标
> （`f_s[slot]`），改为 `F_S + slot` 的常量基址 + TIR 表达式（先例
> `_gqa_prefill_fwd_kernel.py:351`）；若动态 id 在某次编译上被拒，回退方案 = 把任务循环按 2 展开成
> 编译期常量 id（`for it2 in T.serial(NPER//2)` + 两个 unroll 体，id 变成常量）。② 前导 `gram(bid, 0)` 显式
> 给 `gslot=0`。③ **无 `T.transpose`**（`g_j` 由第二次装载的 `(1,64)` 落位得到），兜底方案才是
> `T.transpose(ub_gc, ub_gr, permutation=[1, 0])`（`docs/Tilelang.language/创建操作/T.transpose.md` §2.1，fp32 ✓）。
> ④ `β` 的跨 dtype 有显式 `T.vcast(..., round_mode="rint")`，且 `round_mode` 用关键字（位置写法实测编译失败）。

### 3.4 API 可行性确认

| 断言 | 佐证 |
|---|---|
| Expert 下 fp32（HF32）gemm 64×64×64 可用且精度 ≈ fp32 | 本设计探针 `design_probe_doubling_cube.py`：`T1=I+N` err 2.9e-8、`T2=(I+N²)(I+N)` err 2.7e-7；`design_probe_accum_form.py` V1–V3 全通过（err 2.9e-8）；**本版** `design_probe_gate_sign.py` PROBE-CHAIN fp32 2 级链 rel **5.5e-9** |
| `initC=False` 累加进"已落过盘"的 L0C 可用 | `design_probe_accum_form.py` V3（`gemm→store→gemm(initC=False)`，err 2.9e-8）；`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1（err 0） |
| **同一 L1 buffer 可同时作 gemm 的 A 与 B（`b_transpose=False`）** | 本版 `design_probe_gate_sign.py` PROBE-CHAIN：`T.gemm(l1n, l1n, l0p, ...)` ⇒ fp32 rel 5.5e-9 / fp16 rel 2.8e-5（v0 为此复制了 `l1q`，理由不成立，本版删除，省 5 次 16 KB 装载 ≈ 80 KB/任务） |
| L0C→GM(fp32 scratch)→L1 回喂可用 | `examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py`（fp16 scratch，err 9.7e-3）+ 本设计 fp32 scratch（err 2.7e-7） |
| 回喂**不得**经 `out_idx` 张量 | `EXPERT_FINDINGS.md` §12.1（K2：err 9.7~32）；本设计探针第一版读回 `out_idx` 张量直接触发 bishengir SIGSEGV ⇒ 设计一律用独立 scratch 输入张量（`gs_G`/`ws_S`/`ws_T`/`ws_P`/`y_blk`） |
| **A 落盘：连续中转 + strided region 写正确、无越界** | 本版 `design_probe_store_layout.py`：目标行区逐元素正确（与 `x.half()` 差 2.4e-4 = fp16 自身舍入；fixpipe 量化 err = 0），相邻 `(token, head)` 槽 `max|other| = 0` |
| **strided 落盘的时延** | 同上 PROBE-TIMING：连续 8 KB 写 0.084 µs、strided 8 KB 写 0.194 µs（2.29×，`head=4`/512 B 行距；`head=16`/2048 B 未实测，按同段数外推） |
| **Vector 门/符号全链路真机对拍** | 本版 `design_probe_gate_sign.py`：`e` 与 `exp(g_i-g_j)` 在 `i>j` 区 max_abs_err 1.4e-6（值域 O(1)）、与 `exp(g_j-g_i)` 差 1.07e14；`N` 与 `-β_i·e·G·mask01` 差 1.1e-7（`|ref|max` 0.126）、与 `+β` 差 2.5e-1 ⇒ 方向与符号均为实测通过 |
| `vcast` / `vbrc` 的形态与签名 | 本版微探针 `design_probe_vcast_shapes.py`：`vcast(src,dst,round_mode="rint")`（**关键字**）在 `(64,1)` 与 `(64,64)` 均 err = 0；`vbrc((64,1)→(64,64))` / `vbrc((1,64)→(64,64))` err = 0 |
| **`(i>j)` 掩码不得用 `arange+vcmp`** | 本版微探针 `design_probe_mask.py`：`arange(1-D)→reshape→vbrc→vcmp→vselect` 三种写法（单 arange / 双 arange）均得**全 0 掩码**（无报错、静默错）；改为 host 常量 0/1 tile 后 err = 0 |
| 每级 (gemm+store+load) 的 op 代价 | `design_probe_doubling_cube.py` PROBE-TIMING：6 op/级 = **1.52 µs** ⇒ 0.25 µs/op（64×64 fp32，Cube 侧） |
| 双循环 + 双向多轮握手 | `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`（err 9.8e-4 vs fp16 通路参考） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

- **三维 Kernel**：不使用（`T.Kernel(K)` 一维）；`(chunk, head)` 展平为 `t`。
- **`threads=` 参数**：npuir 下无效（`pattern-library/traps-compiler.md` TRAP-threads-kwarg-noop），不使用。
- **L0C 容量**：`l0g + l0t + l0p = 3 × 16 KB = 48 KB` < 128 KB ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：`l1k(16 KB, dim_k=128) + 3 × 16 KB = 64 KB` < 512 KB ✓（v0 的 `l1q` 已删）。
- **UB 容量**：122 KB 手工 → ×1.10~1.12 膨胀 ≈ 134–137 KB < 192 KB ✓（§4.5）。
- **GEMM 非整除**：`64`（M/N）与 `dim_k ∈ {64,128}`（K）全部是分形 16 的整数倍，**无尾块**；
  真正的"非整除"只在**行数**（序列尾 chunk），由 §0.6 R3 的装载 extent + 掩码 + padding 输出处理。
- **Cube 环路/回喂**：不使用 `out_idx` 回喂（§3.4）；每级只有一次"store 后复用作 gemm dst"，
  已被探针 V2（store-then-reuse）与本设计时序核（`T.serial` 内复用 L0C）证实可用。
- **单 kernel 一段 Cube + 一段 Vector**：满足（§2）。
- **`T.gemm` dst 不支持 L0C region 写**（`CG-2026-0011`）：本设计所有 gemm 的 dst 都是**整个** L0C buffer，
  不用 region；三个 L0C（`l0g/l0t/l0p`）互不复用同一块区域。

#### 3.5.2 参考实现差异说明

1. **越界行处理**：源用逐行 clamp（重复末行），本设计用"动态 extent 装载 + 0/1 常量掩码 + padding 输出"。
   语义等价性见 §0.6 R3 的三段论证（行独立传播 + 丢弃元素恒 0）。差异：`A` 的 padding 行会被写入
   单位行（源不写），wrapper 返回 `A_pad[:total_tokens]` 视图后对调用方不可见。
2. **精度路径**：源把 `a_s`/`i_s`（中间逆块）与 `gate*_s` 存 fp16；本设计全程 fp32，仅在最终落盘
   `fixpipe` 时量化一次 ⇒ 合法行数值更接近 fp64 语义基准（§1.6.1 A3/A5），不改变数学定义。
3. **`dim_k=128` 的 Gram 累加**：源按 2 段 64 深顺序累加；本设计一次 `T.gemm(K=128)`，fp32 累加器内部
   累加顺序由 Cube 分形决定 ⇒ 数值差异在 fp32 舍入级（L0 用例覆盖两种 `dim_k`）。
4. **落盘路径**：源是"逐行 store + 逐行掩码"，本设计是"连续 64×64 量化中转 + Vector 侧一次 strided
   region 写"。数学上写入的字节完全相同（只是搬运顺序不同）；代价是 +16 KB GM 流量/任务（§1.6.3 实测 ≈ +0.3 µs）。

#### 3.5.3 本项目同类实现参考

- Expert MixCV + 双循环握手：`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`。
- Cube 侧 Gram+fixpipe+Vector 后处理：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`。
- 动态 extent 装载 + 尾块掩码（Vector）：`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（L237–L254）。
- 同流水线已交付 kernel（命名/测试分层/`--level` 入口）：`examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py`。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | 形状 | dtype | 对齐说明 |
|---|---|---|---|
| `k` | `[1, total_tokens, key_head, dim_k]` | fp16 / bf16 | 尾轴 `dim_k ∈ {64,128}` ⇒ 128/256 B，≥ 32 B ✓ |
| `g` | `[1, total_tokens, head]` | fp32 | 按 `hid` 取 64 行 ⇒ 跨步 `head`（两种 UB 落位：`(64,1)` 与 `(1,64)`，见 §3.3） |
| `beta` | `[1, total_tokens, head]` | fp16 / bf16 | 同上，按 `(64,1)` 落位后 `vcast` 到 fp32 |
| `eye` | `[64, 64]` | **fp32** | host 常量，给 `T_0 = I` 播种用（`T.gemm(l1e,l1e,...)`） |
| `mask01` | `[64, 64]` | **fp32** | host 常量，严格下三角 0/1（替代 bool 掩码，§1.6.2 V9） |
| 工作区 `gs_G` / `ws_S` / `ws_T` / `ws_P` | `[K, 2, 64, 64]` | fp32 | 每 block 私有；`K ≤ 20` ⇒ 各 640 KB（合计 2.56 MB） |
| 工作区 `y_blk` | `[K, 2, 64, 64]` | 输入 dtype | A 的量化中转；`K ≤ 20` ⇒ 1.28 MB（fp16） |
| `cu_seqlens` | `[num_sequences+1]` | int32 | 接口保留，等长分支**不读** |

### 4.2 输出张量

| 名称 | 形状 | dtype | 说明 |
|---|---|---|---|
| `A`（返回视图） | `[1, total_tokens, head, 64]` | 输入 dtype | `A_pad[:total_tokens].unsqueeze(0)`，`A_pad` 行数 = `ceil(total_tokens/64)*64`；**连续**（`is_contiguous()` 为真） |
| `k_rnorm` | `[1, 1, key_head]` | fp32 | `l2norm=False` 占位（源码同款，L689） |

> 内核签名里 `A_pad` 取 **3D `[S_pad, head, 64]`**：落盘 region 因此是 `[64, 1, 64]`（只需折叠 1 个 size-1 轴），
> 比 4D `[1,S,head,64]` 少一层折叠；wrapper 用 `unsqueeze(0)` 给出 4D 连续视图，I/O 契约不变。

### 4.3 中间缓冲区

| 层级 | buffer | 形状 / dtype | 说明 |
|---|---|---|---|
| L1（Cube） | `l1k` | `[64, dim_k]` 输入 dtype | k 片（`T.copy` slice 装载，NZ 转换由后端完成） |
| L1 | `l1e` | `[64,64]` fp32 | 单位阵常量（每 block 装一次） |
| L1 | `l1p` | `[64,64]` fp32 | `P_j`；**同一 buffer 兼作 `P²` 的 A 与 B**（实测可行） |
| L1 | `l1t` | `[64,64]` fp32 | `T_j`（gemm B 操作数） |
| L0C | `l0g` / `l0t` / `l0p` | `[64,64]` fp32 | Gram / T 累加器 / P² |
| UB（Vector） | `MK` | `[64,64]` fp32 | 常量 0/1 掩码（每 block 一次） |
| UB | `UB_G` | `[64,64]` fp32 | Gram |
| UB | `UB_A` | `[64,64]` fp32 | `g_i` → `d` → 复用 |
| UB | `UB_B` | `[64,64]` fp32 | `g_j` |
| UB | `UB_E` | `[64,64]` fp32 | `e` |
| UB | `UB_F` | `[64,64]` fp32 | `bm` → `t1` |
| UB | `UB_N` | `[64,64]` fp32 | `t2` → `N` |
| UB | `UB_Y` | `[64,64]` 输入 dtype | A tile 中转（8 KB） |
| UB | `ub_gc` / `ub_gr` | `[64,1]` / `[1,64]` fp32 | `g` 的两种落位（各 256 B） |
| UB | `ub_b16` / `ub_b32` / `ub_bn` | `[64,1]` 输入 dtype / fp32 / fp32 | β 原值 / fp32 / `-β`（各 ≤ 256 B） |
| GM | `gs_G[K,2]` / `ws_S[K,2]` / `ws_T[K,2]` / `ws_P[K,2]` | 64×64 fp32 | Cube↔Vector 交接与链轮转 |
| GM | `y_blk[K,2]` | 64×64 输入 dtype | A 的量化中转（连续块） |

**UB alias 表（本版新增，落实 REVIEW v0 问题 12；§3.3 用的就是这张表）**：

| 复用的 buffer | 生命周期（同一任务内） | 依据 |
|---|---|---|
| `UB_A` | `g_i`（vbrc）→ `d`（vsub 原地）→ 后续不再需要（`e` 进 `UB_E`） | 三步都只读上一版自身的元素，原地安全 |
| `UB_B` | `g_j`（vbrc）→ vsub 之后空闲（本版不再需要零 tile，因为它已被 mask01 取代） | — |
| `UB_F` | `bm`（vbrc）→ `t1`（vmul 原地）→ 之后 `t2` 进 `UB_N` | 逐元素 |
| `UB_E` | `e`（vexp 输出）→ 只被 `vmul(UB_F, UB_E)` 读一次 | — |
| `MK` | 全 block 生命周期（常量），不与任何 tile 复用 | 复用会破坏后续任务 |
| `ub_gc` / `ub_gr` / `ub_b16` / `ub_b32` / `ub_bn` | 每任务重写；`ub_bn` 只在 vbrc 前有效 | 64 元素级，不参与 alias |

> 显式 alias 后的 UB 峰值 = **7×16 KB（`MK/UB_G/UB_A/UB_B/UB_E/UB_F/UB_N`）+ 8 KB（`UB_Y`）+ ~1.5 KB（小 buffer）≈ 122 KB**；
> 不做 alias 的最坏形态是 10×16 KB + 8 KB ≈ 168 KB（×1.12 ≈ 188 KB，逼近上限）——**Stage 3 必须按上表复用 buffer**，
> 不得一 buffer 一名（`§9 R13`）。

### 4.4 内存搬运路径

`GM(k) → L1(MTE2, T.copy slice + 隐式 NZ) → L0A/L0B → L0C(fp32) → GM(gs_G) → UB(MTE2) → Vector 组装 → GM(ws_S)
→ L1(MTE2) → L0C(链, 6 级, 每级经 GM ws_T/ws_P 往返) → fixpipe(f32→dtype) → GM(y_blk) → UB(MTE2) → GM(A_pad, strided region)`。
无越级路径（无 GM→L0 直连，无 L1↔L1 搬运）。

### 4.5 UB / L1 / L0C / GM 预算

| 层级 | 峰值 | 上限 | 余量 |
|---|---|---|---|
| UB（Vector） | `MK` + `UB_G` + `UB_A` + `UB_B` + `UB_E` + `UB_F` + `UB_N` = 7×16 KB；`UB_Y` 8 KB；`ub_gc/ub_gr/ub_b16/ub_b32/ub_bn` ≈ 1.5 KB ⇒ **手工 alias ≈ 122 KB**；不 alias 的最坏形态 10×16 KB + 8 KB ≈ 168 KB | 192 KB | alias 形态 ×1.12 ≈ **137 KB**（余量 55 KB）；不 alias 形态 ×1.12 ≈ 188 KB（余量 4 KB，仅兜底）；两种形态均 < 192 KB |
| L1（Cube） | `16 KB（l1k, dim_k=128）+ 3×16 KB = 64 KB` | 512 KB | 12.5% |
| L0C | `3×16 KB = 48 KB` | 128 KB | 62% |
| GM 工作区 | `4 × 20 × 2 × 16 KB = 2.56 MB`（`gs_G`/`ws_S`/`ws_T`/`ws_P`）+ `y_blk` 1.28 MB + `A_pad`（`S=32768,H=16`：512×64×16×64×2 B ≈ 64 MB）+ `mask01`/`eye` 32 KB | HBM 64 GB | — |

---

## 5. 核切分与任务映射（Tiling 策略）

### 5.1 ① 逻辑核数计算

任务空间 = `(chunk, head)` 笛卡尔积：

```
chunks_per_seq = ceil(S / 64)                     # in-scope 等长序列
nch_total      = num_sequences × chunks_per_seq    # packed 行布局下的 chunk 总数
T (逻辑任务数)  = nch_total × head
```

| 形状（in-scope） | `chunks_per_seq` | `nch_total` | `head` | **T（逻辑任务数）** |
|---|---|---|---|---|
| L0：`S=64, head=1` | 1 | 1 | 1 | **1** |
| `S=3000, head=16` | 47 | 47 | 16 | **752** |
| `S=4096, head=16` | 64 | 64 | 16 | **1024** |
| `S=12288, head=16` | 192 | 192 | 16 | **3072** |
| `S=32768, head=16` | 512 | 512 | 16 | **8192** |

每个任务的工作量**恒定**（一个 64×64 逆，`dim_k` 固定），故逻辑任务数 = 任务数，无权重不均。
（与 §1.6.0 R3 的 `8192 任务 / 410 任务每核 / 3.79 ms`、§4.5 的 `512×64×16×64` 三处同源。）

### 5.2 ② 物理核数依据

实查（2026-10-17，本机 Ascend910B3，npu-smi 25.5.0 / Chip 6）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()      # -> 20
```

记录：**返回 20**（AI Core 数）。本内核是 **MixCV**（Cube + Vector 同 kernel），瓶颈引擎是 Cube
（55 op/任务中 37 个在 Cube 侧：13 gemm + 24 搬运），因此按 **Cube/混合算子口径直接使用返回值 20**
（不乘 2）；`get_aicore_num() * 2 = 40` 只适用于纯 Vector 算子
（`.agents/skills/_shared/standards/core-split-strategy.md` §1 要素②）。
（既有常数条目 `CONST-aicore-910B2C` 的 24 是 910B2C 档位；本机实查 20，**以实查为准**。）

### 5.3 ③ 规模判定与分核方案

**规模判定**：所有 in-scope 形状 `T ≥ 752 ≫ 20`（`S=3000` 已是最小真实形状），属**极大规模**；
把 `T` 摊到 20 个核上每个核仍有 38~410 个任务，无法通过调 block 尺寸把逻辑任务数降到物理核数量级
（block 尺寸由语义固定：chunk = 64、逆矩阵 64×64，不可再切分——切分 64×64 会破坏 gemm 链的块结构）。

**分核方案（三选一 → 极大规模：固定启动核数 = 物理核数 + 核内 `T.serial` 串行）**：

```
K     = min(20, T)                       # 启动 block 数（编译期常量）
NPER  = ceil(T / K)                      # 每 block 任务循环上界（编译期常量，静态）
任务映射：t = bid + it * K ，it ∈ [0, NPER)，bid ∈ [0, K)   # 跨步分配，负载差 ≤ 1 个任务
守卫：if t < T （尾部空任务直接跳过）
```

- **循环边界静态**：`K` 与 `NPER` 全部由 host 侧静态参数（`S/head/num_sequences`）算出，是编译期常量，
  满足"核内串行循环边界必须为静态值"。
- **对齐物理核整数倍**：`K = 20` 恰好等于物理核数 ⇒ 无双核轮转、无欠载；跨步映射使每个核拿到的任务
  数只差 ≤1（`T=1024` → 12 个核 52 个、8 个核 51 个）。
- **不需要跨核同步**：任务之间完全独立（每个 chunk 独立求逆），无归并、无 workspace 竞争
  （`gs_G/ws_S/ws_T/ws_P/y_blk` 均按 `bid` 私有）。
- **`A_pad` 写冲突**：任务边界互不重叠（每任务写自己的 64 行 × 自己的 head 列）⇒ 无原子/加锁需求。

### 5.4 Block 划分与 Tile Shape（非整除处理）

| 项 | 取值 | 非整除策略 |
|---|---|---|
| 任务粒度 | 1 chunk × 1 head（64 行） | 行数非整除（尾 chunk）：动态 extent 装载 + 0/1 掩码 + `A_pad` padding（§0.6 R3） |
| Gram tile | M=64, N=64, K=`dim_k` | `dim_k ∈ {64,128}` 均为 16 的整数倍 ⇒ 无 K 尾块 |
| 链 tile | 64×64×64 | 64 = 16×4 分形对齐 ⇒ 无尾块 |
| Vector tile | 64×64（尾轴 64 = 256 B） | 无尾 lane 浪费（fp32 向量宽 8 lane，64 = 8×8） |
| 落盘 tile | 64 行 × 64 连续元素（行距 `head·64`） | 由 `A_pad` 保证 64 行总在界内（尾 chunk 的 padding 行写单位行、返回视图裁掉） |

---

## 6. 循环与调度结构

| 结构 | 选择 | 理由 |
|---|---|---|
| 任务循环 | **`T.serial(NPER)`**（`NPER` 静态），任务号 `t = bid + it*K` | 极大规模分核（§5.3）；`T.serial` 能携带跨任务的 GM 依赖（每任务独立，但 buffer 复用） |
| 链级循环 | **Python 层静态展开 `for lev in range(6)`**（不生成 TIR 循环） | 每级读写不同 slot、末级无平方；展开后无动态边界，也避开"循环携带 Cube 依赖"的历史不确定面（`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §10） |
| Gram 的 K 规约 | 单次 `T.gemm(size=[64,dim_k,64])` | `dim_k ≤ 128`，一次覆盖；不引入 `T.serial` K 段循环（源 O4 的意图由 Cube 内部累加承接） |
| 落盘的位置 | Vector 任务循环内**一任务滞后**（`if it >= 1: scatter(t-K, ...)`）+ 循环后尾部一次 | 若在同任务末尾落盘，Cube 需等 Vector 完成才能进下一任务 ⇒ 2 slot 流水退化为串行（§7 论证）；滞后一拍后两引擎可各自背靠背执行 |
| `T.Parallel` | **不使用** | Vector 侧全部用 tile 级 v-prefix op（`vbrc/vsub/vmin/vexp/vmul/vcast`），由编译器映射到向量 lane；不需要手工 lane 并行 |
| `T.Pipelined` | **不使用（本版）** | 唯一可流水的是 Cube 的"装载→gemm"，而 `l1k` 单缓冲 + 每任务一次装载；CV 级流水已由 2 slot 握手提供。列入 Stage 4 备选（`num_stages=2` 双缓冲 `l1k`，预期 −0.3 µs/任务） |
| `T.Persistent` | **不使用** | 与本设计"固定 K=20 + 核内 `T.serial`"等价，无需额外原语；`T.serial` 已摊薄启动开销（20 次 block 启动/launch） |
| 跨 chunk 累积 | **无**（每 chunk 独立清零） | 与源一致；`g` 已是 kernel 1 的 chunk 内累积值 |

---

## 7. 同步策略

Expert 模式：**全部手动同步**。仅 **6 个 flag id**（≤15 预算，`CONST-flag-id-budget`）：

| flag（id） | 方向 | set 侧 | wait 侧 | 语义 |
|---|---|---|---|---|
| `f_g[0]`(0) / `f_g[1]`(1) | Cube → Vector | `with T.rs("PIPE_FIX")`：`l0g → gs_G[slot]` 后 | `with T.rs("PIPE_MTE2")`：读 `gs_G[slot]` 前 | Gram(slot) 就绪 |
| `f_s[0]`(2) / `f_s[1]`(3) | Vector → Cube | `with T.rs("PIPE_MTE3")`：`UB_N → ws_S[slot]` 后 | `with T.rs("PIPE_MTE2")`：读 `ws_S[slot]` 前 | `N`(slot) 就绪 |
| `f_a[0]`(4) / `f_a[1]`(5) | Cube → Vector | `with T.rs("PIPE_FIX")`：`l0t → y_blk[slot]` 后 | `with T.rs("PIPE_MTE2")`：读 `y_blk[slot]` 前（落盘时） | A tile(slot) 就绪 |

flag id 写作 `F + slot`（`F ∈ {0,2,4}`，`slot = it % 2` 是 TIR 表达式）；动态 id 的可用性由
`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`
L351 的 `T.sync_block_set(i)` 佐证，回退方案见 §3.3 的自查段。

**无死锁 / 无竞争论证**（Cube 领先一拍取 Gram；Vector 落后一拍落盘）：

1. Cube 前导阶段完成 `gram(t0, 0)` 并 `set(f_g[0])`；Vector 任务 0 的 `wait(f_g[0])` 必然被满足。
2. 迭代 `it` 的 Cube 顺序：`gram(t+K, 1-slot)`（写 `gs_G[(it+1)%2]`）→ `wait(f_s[slot])` → 链 → `l0t → y_blk[slot]` → `set(f_a[slot])`。
   Vector 迭代 `it`：`wait(f_g[slot])` → 组装 → `ws_S[slot]` → `set(f_s[slot])` → （`it≥1`）`wait(f_a[(it-1)%2])` → 落盘上一任务 → `set` 无（见 4）。
3. **依赖链无环**：`Cube(it)` 依赖 `Vector(it)`（`f_s`），`Vector(it)` 只依赖 `Cube(it-1)`（`f_a` 的上一 slot）与
   `Cube` 的 Gram（在 `it-1` 就已预取）⇒ 依赖图是 `Cube(it-1) → Vector(it) → Cube(it)` 的**两级流水**，
   稳态吞吐 = max(Cube, Vector) = 9.25 µs/任务（§1.6.0 roofline），不会退化为相加。
4. **`gs_G` 的 WAR**：Cube 在迭代 `it` 重写 `gs_G[(it+1)%2]`，该 slot 上一次被 Vector 读是在任务 `t-K`；
   Cube 在**上一迭代**已 `wait(f_s[(it-1)%2])`，即 Vector 已完成任务 `t-K` ⇒ 不覆盖未消费数据。
5. **`ws_S` 的 WAR**：Vector 在任务 `t+K` 重写 `ws_S[slot]`，该 slot 上一次被 Cube 读是在迭代 `it` 的链（本迭代先完成）⇒ 安全。
6. **`y_blk` 的 WAR（本版新增，关键）**：Cube 在迭代 `it` 写 `y_blk[slot]`，Vector 在迭代 `it+1`（`it≥0` 的滞后落盘）
   读同一 slot；Cube 下一次写该 slot 在迭代 `it+2`，而它在 `it+2` 开头会 `wait(f_s[slot])`——该 flag 由
   Vector 的迭代 `it+2` 在**其迭代 `it+1` 的整体程序序（含落盘读）之后**才 set ⇒ 读先于下一次写。
7. flag 计数严格配对：每 slot 每任务恰好一次 `set`/一次 `wait`（`f_a` 的 wait 分别在循环内的滞后路径与尾部
   epilogue 各出现一次，合计等于该 slot 的任务数；`it=0` 无滞后落盘，由 `if it >= 1` 与 `NPER` 尾部守卫配对）。
   形态同 `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`（含多轮复用已验证）。

**其他同步点**：

- 所有 L0C→GM 必须 `with T.rs("PIPE_FIX")`（否则 lower 成 `hivm.hir.store` 被拒）。
- Cube 侧 `T.copy(GM→L1)` 与随后的 `T.gemm` 之间由 MTE2→M 的隐式依赖保证（同引擎程序序 +
  框架注入 sync，`--disable-hivm-auto-inject-sync=false` 默认开启）。
- Vector 侧同 scope 内的 UB 读写靠 `PIPE_V` 程序序；跨引擎只经 GM + flag。
- **禁止**：把中间积回喂到 `out_idx` 张量（`EXPERT_FINDINGS.md` §12.1 K2 数值错；本设计探针复现为
  编译器 SIGSEGV）——本设计所有回喂走独立 scratch 输入张量。

---
## 8. 验证方案

### 8.1 Golden 函数（独立于 NPU 算法）

依据 §0.1 语义（**不是**本设计的 doubling 算法），纯 torch CPU，fp64：

```python
def golden_blocksolve_A(k, g, beta, chunk=64, uniform_seq_len=0, num_sequences=1):
    """k:[1,T,kh,dim_k] g:[1,T,H] beta:[1,T,H] -> A:[1,T,H,chunk] (fp64)

    逐 chunk 构造 S_sys = beta_i·exp(g_i-g_j)·(k_i·k_j) (i>j)，再做 fp64 直接求解；
    末尾 chunk 不足 chunk 行时：g/k 行取 min(row, end-1)（源实现的 clamp 语义），beta=0。
    """
    T_, kh, dk = k.shape[1], k.shape[2], k.shape[3]
    H = g.shape[2]
    group = H // kh
    A = torch.zeros(1, T_, H, chunk, dtype=torch.float64)
    for seq in range(num_sequences):
        s0 = seq * uniform_seq_len
        end = s0 + uniform_seq_len
        for c in range((uniform_seq_len + chunk - 1) // chunk):
            base = s0 + c * chunk
            valid = min(chunk, end - base)
            rows = torch.clamp(torch.arange(base, base + chunk), max=end - 1)
            for hid in range(H):
                kk = k[0, rows, hid // group, :].double()          # [64, dk]
                gg = g[0, rows, hid].double()                       # [64]
                bb = beta[0, rows, hid].double().clone()
                bb[valid:] = 0.0                                    # 越界行 beta=0
                G = kk @ kk.T
                mask = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), -1)
                S = torch.where(mask, bb[:, None] * torch.exp(gg[:, None] - gg[None, :]) * G,
                                torch.zeros((), dtype=torch.float64))
                A[0, base:base + valid, hid, :] = torch.linalg.solve(
                    torch.eye(chunk, dtype=torch.float64) + S[:valid, :valid],
                    torch.eye(valid, dtype=torch.float64))
    return A
```

- **交叉校验（结构层，fp64 全通路）**：用"分块前代"实现（16×16 对角块 + 16 次块回代，即 §0.3 P6/P7 的 fp64 版，
  见 `verify_equiv.py` 的 `blocked_source(..., store_dtype=torch.float64)`）独立复算一次，与
  `torch.linalg.solve` 必须一致到 **1e-10**。**实测 2.776e-17**（v0 的 `blocked_source` 无条件降到 fp32，
  使该通道实为 fp32、实测 2.3e-8 与门槛矛盾；本版已修：`store_dtype=torch.float64` 时全程保留 fp64）。
- **符号裁决（结构层）**：同一次运行打印 `|source - inv(I+S)|` 与 `|source - inv(I-S)|`，实测
  **2.8e-17 / 4.4e-1** ⇒ `N = -S_sys` 由机器判定，不依赖书写约定。
- **量化层（信息性，不设门槛）**：`blocked_source(..., store_dtype=torch.float16)` 模拟源的 fp16 中间块
  通路，用于"本实现不劣于源通路"的对照（实测 `SC-typical` rel 7.0e-5；`SC-beta-x20` 出现 nan
  ——即源的 fp16 溢出边界，与 §1.6.1 的 A5 注一致）。
- **对照源实现**：`prefill_blocksolve_A_bthd`（GPU）在 fp16 通路下的输出作为"源通路参考"，仅用于
  "本实现不劣于源通路"的判定（源的 fp16 中间块误差 ~1e-4，见 §1.6.1 附注）。

### 8.2 精度标准

| dtype | 判据 | 依据 |
|---|---|---|
| fp16 | `max_abs_err ≤ 5e-3` 且 `max_rel_err ≤ 5e-3`（对 `|A| ≤ 4` 的元素；更大元素按相对判据） | §1.6.1 A4 rel ≤ 4.0e-6、A6（含 β 的 fp16 量化）rel ≤ 4.6e-4、A5 rel ≤ 9.3e-5 + fp16 eps 4.9e-4，留 ~5× 余量 |
| bf16 | `max_abs_err ≤ 4e-2` 且 `max_rel_err ≤ 4e-2` | bf16 eps 7.8e-3（8 bit 尾数）× 6 级链余量 |
| fp32 scratch（内部） | 不设门禁，仅 L0 打印 | 内部量 |
| 门/掩码单点（L0-h） | `N` 与 `-β_i·e·G·mask01` 的 `max_abs_err ≤ 1e-5`（`dim_k=128`、`|G|≤1`） | 本版真机实测 1.1e-7（`|ref|max` 0.126），留 2 个数量级余量 |

### 8.3 L0 / L1 / L2 测试计划

| 层 | 用例 | 目的 / 判据 |
|---|---|---|
| **L0-a** | `S=64, head=1, key_head=1, dim_k=64, fp16` | 单任务全矩阵：对角块=1、上三角=0、下三角逐元素比对；判据 fp16 门槛；**同时断言写出的 `A` 行区间外（若 `A_pad` 行 > 64）无改动** |
| **L0-b** | 同 L0-a，`dim_k=128` | Gram 的 K 维 128 档（一次 gemm 的累加路径） |
| **L0-c** | 同 L0-a，`bf16` | `fixpipe` f32→bf16 + 全链路 bf16 输出路径 |
| **L0-d** | `S=100, head=1, key_head=1, dim_k=64`（2 chunk，尾 chunk 36 行） | 动态 extent 装载 + 掩码；padding 行不越界、`A_pad[:100]` 视图正确、`A.is_contiguous()` 为真 |
| **L0-e** | `S=64, head=2, key_head=1, dim_k=64`（`group=2`） | `khid = hid // group` 映射 + **落盘的行距（`head·64`）正确性**：两个 head 的结果互不覆盖（本版专门覆盖 REVIEW 问题 3 的场景） |
| **L0-f** | `S=64, beta` 含 0 行 + 大 gate 跨度（`g` 跨度 ≈ -40） | 数值边界：0 行 → 单位行；极端 gate 不产生 NaN（`vmin` 限幅 + 0/1 掩码生效） |
| **L0-g** | `S=128, head=16, key_head=16, dim_k=128, fp16`，只跑 1 个 chunk 的 16 个任务 | **strided 落盘在最坏行距（2048 B）下的正确性**：16 个 head 的 64×64 结果全部逐元素正确、`A_pad` 其余行/列零改动（本版探针只测到 `head=4`，此处补 `head=16`） |
| **L0-h** | 同 L0-a，把中间 `N` 与 `ub_gc/ub_gr/ub_bn` 也 dump 出来 | **门/符号单点校验**（§1.6.1 A6 的设备侧对应）：`d == g_i-g_j`、`bn == -β`、`N == -β·e·G·mask01` |
| **L1-a** | `S=1024, head=4, key_head=2, dim_k=128, fp16`（T=64 > 20） | 多任务 / 多核跨步映射 + 6 个 flag 多轮复用（含 `f_a` 的滞后路径） |
| **L1-b** | `S=3000, head=16, key_head=1, dim_k=128, fp16`（T=752，尾 chunk 56 行） | in-scope 真实尾 chunk + 分核（38 任务/核）+ 真实时延（§9 R1 的裁决输入） |
| **L2** | `S ∈ {4096,12288,32768}` × `key_head ∈ {1,16}` × `dim_k ∈ {64,128}` × `{fp16,bf16}` | 交付验证（12 组合），与 golden + 源通路双对照 |
| Boundary | `S=56`（单 chunk 且不满）、`S=64`（整除单块）、`S=65`（2 chunk，尾 1 行） | chunk 边界三态：`valid < 64` / `== 64` / `== 1` |

**L0 门槛**：L0-a~L0-h 全绿才允许提交 L1；L1-a/L1-b 全绿才允许 L2。任一失败先查 §9 的对应风险条目。

### 8.4 通过标准

L0 + L1 + L2 + Boundary 全绿 ⇒ Stage 3 出口 `[PRECISION_PASS]`；L2 任一 dtype/shape 失败 ⇒
回到 §1.6.1 复核（A4 链级数、A5 量化、A6 门构造）或 §9 的 R2/R3/R14 回退路径。

---

## 9. 风险点与注意事项

| # | 风险 | 影响 | 处置 / 裁决阈值 |
|---|---|---|---|
| R1 | 性能：`0.48 ms`（S=4096）/ `3.79 ms`（S=32768）是**估算**（Cube 37 op × 0.25 µs = 9.25 µs/任务，Vector 7.25 µs） | 时延不达标 | Stage 3 先测 L1-b 真实时延；若 > **1.2×** 估算 → 评估 §1.6.3 备选 **L2（fp16 链 scratch + fp16 操作数，GM 流量 425 → 265 KB，流量项 5.9 → 3.7 µs）**；若 > **1.5×** → 评估 **L3（双 head 打包，op/head −50%，受 L0C 128 KB 限制）**。注意本设计 Cube 是瓶颈（9.25 vs Vector 7.25），所以 L2 只有在 Cube 的**流量等待**占比高时才有效；若瓶颈在 Cube 发射项，则应先做"任务内减少 L1 往返"（把 `ws_T/ws_P` 的 5+5 次往返合并成 2 slot 直连）而非 L2 |
| R2 | `initC=False` 累加只实测到 **2 级链**（本探针）与 3 步链（`10` K1）；6 级链 + 每级"store 后复用 L0C"未验证 | 数值错/编译崩 | L0-a 先跑单任务 6 级；失败回退：每级改为"显式 `gemm(·,·,l0,initC=True)` + 下一次 `T.vadd` 合并"（+5 op/任务，≈ +1.25 µs） |
| R3 | fp32（HF32）gemm 的**长链**数值行为（探针只到 2 级） | rel 超 5e-3 | L0 直接对照 fp64 golden；CPU 侧 6 级 fp32 仿真 rel ≤ 4.0e-6（§1.6.1 A4），余量充足 |
| R4 | **动态 extent 装载**（`valid` 行）：`T.copy(GM→L1)` 与 `g`/`beta` 的列切片装载在 `valid<64` 时都带动态 extent，仓内先例只有 GM→UB（`_argreduce_kernel.py` L237–L254）；本版探针只覆盖了 `valid=64` 的静态形态 | 编译失败/读越界 | L0-d（`S=100`，尾 chunk 36 行）第一个覆盖；失败回退"整 64 行静态装载 + `t = min(t, T-1)` 式 clamp"（源 O12 同形态）或"尾 chunk 逐行拷贝" |
| R5 | Cube/Cube-Vector scope 内 `if t < T` 动态守卫 | lower 失败或分支错 | 回退：把越界任务映射到 `t = T-1`（幂等重算 + 重复写同一行，结果正确；与源的 O12 同形态） |
| R6 | 6 个 flag id 的多轮复用语义（计数配对）+ `f_a` 的一任务滞后 | 死锁 / 读旧数据 | 形态已被 `14` 验证（2 轮）+ 本设计 §7 第 6/7 条论证；L1-a（64 任务/20 核 ⇒ 4 轮）覆盖；失败回退到每任务轮换 flag（id = `4*slot + dir`，仍 ≤ 12） |
| **R7** | **门单调性依赖**：`vmin(d, 0)` 只在"chunk 内 `g` 单调非增"时是恒等（否则会截断下半三角的合法正值） | 数值错（若上游换门定义） | GDN 的 `g` 是 chunk 内累积 log 门（`cumsum(log_sigmoid)` ≤ 0、单调非增，由 kernel 1 保证）⇒ 本设计**声明该依赖**并由 L0-f（`g` 跨度 −40 的单调序列）+ L2 全 shape 覆盖。若上游改为非单调门：必须去掉 `vmin` 并换回 bool 掩码（当前工具链的 `arange+vcmp` 路径实测不可用，见 §1.6.2 V9 ⇒ 需先补编译器能力，登记在 §9 的"未文档化假设"清单） |
| R8 | `A_pad` + 视图返回：`A` 的底层存储比 `total_tokens` 大（≤ 63 行） | 下游若 `clone`/`contiguous` 会产生拷贝；若断言 storage 尺寸会失败 | wrapper 文档中注明；L0-d 校验 `A.shape`、`A.is_contiguous()`、数值（本版 `A_pad` 为 3D + `unsqueeze(0)`，视图**连续**，优于 v0 的 4D 切片） |
| R9 | L0C 复用（同 buffer 跨任务/跨级 store 后复用） | 编译期约束（历史上 8.5.2 曾报错） | 9.0 已实测可用（`EXPERT_FINDINGS.md` §12.1、本探针时序核 `T.serial` 内复用）；L0 覆盖 |
| R10 | GM 工作区按 `K=20` 固定分配（2.56 MB + 1.28 MB），若 Stage 4 改 `K` 需同步改 host | shape 不匹配 | host 侧由同一常量派生（`K = min(20, T)`），不硬编码两份 |
| R11 | `num_sequences > 1`（packed 等长）虽不在 in-scope，但任务映射已按 `seq = t // (ch_per_seq*head)` 写 | 若启用需补测 | 列为 out-of-scope（算子级 §1），代码保留通用路径，L2 不覆盖 |
| R12 | 设计期常数（Cube 0.25 µs/op、Vector 0.5 µs/op、72 GB/s/core、strided 落盘 0.194 µs@head=4）绑 910B3 + CANN 9.0.0 | 版本升级后失效 | §11 打版本戳；`kb_stale_check.py` 会在工具链变更后标记待重验 |
| R13 | UB alias 不生效（Stage 3 一 buffer 一名）⇒ 168 KB × 1.12 ≈ 188 KB | 逼近/超出 UB 上限 | §4.3 已给**显式 alias 表**（`UB_A/UB_B/UB_F/UB_E` 的复用链），Stage 3 必须照表落 buffer；L0-a 编译一次即可暴露（UB overflow 报错文本含实际字节数） |
| **R14** | **strided 落盘在 `head=16`（2048 B 行距）未实测**：探针只覆盖 `head=4`（512 B） | 落盘变慢或（最坏）写错 | L0-g 专门覆盖 `head=16`；若时延超预期（> 1 µs/任务）回退 L4（head-major + host `permute().contiguous()`，+74 µs 总量）；若写错，回退 L5（逐行写，代价见 §1.6.3） |
| **R15** | **`T.arange`+`T.vcmp` 掩码路径在本工具链静默失效**（全 0 掩码、无报错） | 若 Stage 3 擅自"优化"回 arange 掩码 ⇒ 结果全错且无编译错误 | 已从设计删除该路径（§1.6.2 V9），改用 host 常量 `mask01`；L0-h 会在数值上拦截（`N` 的严格下三角之外出现非零即失败）。该陷阱已登记到本任务复盘，建议蒸馏进 pattern-library |
| **R16** | `T.vcast` 的第 3 个**位置**参数是 `size` 不是 `round_mode`（文档签名与实际实现不一致） | 位置写法编译失败（`load->indices.size() == ndim` 误报） | 一律用 `round_mode="rint"` 关键字（§3.2/§3.3）；已登记复盘 |

**技术约束检测结论**：三维 Kernel（不涉及）/ L0C 容量（48 KB < 128 KB ✓）/ GEMM 非整除（分形全对齐，
仅行数非整除 → 已给 padding+mask 方案）/ 三维 grid 与 `threads=`（不使用）/ 单 Cube+单 Vector scope（满足）/
`T.gemm` 不支持 L0C region dst（不使用 region）/ GM→L1 跨步装载用 `T.copy` slice 而非 `load_nd2nz`（满足）。

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|---|---|---|
| 设计文档 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md` | 本文（revision 1） |
| 等价性验证脚本 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py` | D-1，6 项 `EQUIV_PASS`（A1–A6）+ 符号裁决 + 反例对照 |
| 设计期探针 1 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_doubling_cube.py` | 2 级 doubling 数值 + per-op 时延（v0 起，仍有效） |
| 设计期探针 2 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py` | `initC=False` / store-then-reuse 形态；**V4 分支本轮判废**（§11） |
| 设计期探针 3（新增） | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_store_layout.py` | A 落盘路径正确性 + strided/contig 时延 |
| 设计期探针 4（新增） | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_gate_sign.py` | 门/符号全链路真机对拍；2 级链 fp16/fp32；A/B 同 L1 buffer |
| 设计期微探针（新增） | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_vcast_shapes.py`、`design_probe_mask.py` | `vcast/vbrc` 形态与签名；掩码构造路径裁决 |
| 探针环境脚本 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/_probe_env.sh` | CANN 9.0.0 + tilelang dev root + `TILELANG_ASCEND_MODE=Expert` 的统一入口 |
| 内核（Stage 3 产出，本阶段不写） | 算子目录下的 `prefill_blocksolve_A_bthd.py` | 由 Stage 3 落盘（本阶段只交付设计工件） |
| 复盘 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/RETROSPECTIVE.md` | Stage 1（revision 1）复盘章节 |

---

## 11. 版本戳与参考

**工具链**：CANN **9.0.0**（`bishengir-compile` 1.1.0 / `hivmc` 0.2.0）/ tilelang **0.1.2** dev root
（`/home/developer/tilelang-mlir-ascend/build`）/ **Ascend910B3**（`npu-smi 25.5.0`，Chip 6，
`ASCEND_VISIBLE_DEVICES=6`）/ 2026-10-17。`NPUUtils.get().get_aicore_num() == 20`（实查）。

**模式**：`TILELANG_ASCEND_MODE=Expert`（显式赋值）；`Kernel.ascend_mode = "Expert"` +
`scoped_ascend_mode`（`examples/TileOPs/tileops/utils/ascend_mode.py`，CG-2026-0010）。

**实测常数与方法（本任务，待蒸馏入 pattern-library）**：

| 常数 / 事实 | 值 | 来源（探针 + 日期） |
|---|---|---|
| 64×64 fp32 的 gemm/store/load 平均 op 代价（Cube 侧） | **0.25 µs/op**（6 op 一级 = 1.52 µs，`reps=2` vs `202` 差分） | `design_probe_doubling_cube.py` PROBE-TIMING（2026-10-17） |
| Expert 下 fp32(HF32) gemm 64³ 精度 | err 2.9e-8（`|I+N| ≈ 1`）；2 级链 err 2.7e-7 | 同上 PROBE-CORRECT |
| `initC=False` 累加进"已 store 过"的 L0C | 正确（err 2.9e-8） | `design_probe_accum_form.py` V3 |
| 回喂经 `out_idx` 张量 | 编译器 SIGSEGV（9.0 上比"数值错"更硬） | 本组探针第一版（已修） |
| **同一 L1 buffer 作 gemm 的 A 与 B（`b_transpose=False`）** | fp32 rel **5.5e-9**、fp16 rel **2.8e-5**（2 级链） | `design_probe_gate_sign.py` PROBE-CHAIN（2026-10-17） |
| **Vector 门/符号全链路（真机）** | `e` vs `exp(g_i-g_j)`：1.4e-6（`i>j`，O(1) 值域）；vs 反方向 `exp(g_j-g_i)`：1.07e14。`N` vs `-β·e·G·mask01`：1.1e-7（`|ref|max` 0.126）；vs `+β`：2.5e-1 | `design_probe_gate_sign.py` PROBE-GATE |
| **A 落盘（连续中转 + strided region 写）正确性** | 目标行区 err 2.4e-4（= fp16 自身舍入；fixpipe 量化 err 0）、相邻 `(token,head)` 槽 `max|other| = 0` | `design_probe_store_layout.py`（2026-10-17） |
| **strided vs 连续 8 KB 落盘时延** | 0.194 vs 0.084 µs（**2.29×**，`head=4`/512 B 行距） | 同上 PROBE-STORE-TIMING |
| **`T.vcast` 签名陷阱** | 位置第 3 参是 `size`；必须 `round_mode="rint"` 关键字。`(64,1)`/`(64,64)` fp16→fp32 err 0 | `design_probe_vcast_shapes.py` M1/M2/M5 |
| **`T.vbrc` 形状** | `(64,1)→(64,64)` 与 `(1,64)→(64,64)` err 0；`(M,1)` 语义 = 每行同值（`g_i`） | 同上 M3/M4 |
| **`arange+reshape+vbrc+vcmp` 掩码路径** | **静默全 0**（单/双 arange 皆然，无编译错）；host 常量 0/1 tile 精确 | `design_probe_mask.py` MA/MC/MD/MB |
| **v0 `design_probe_accum_form.py` 的 V4 分支判废** | 该分支在 if-chain 中**没有实现**（`V1/V2/V3` 之外无 `elif mode == "V4"`），实际执行"什么都不做"，`err=1.875` 是未初始化 `out_idx` 张量的垃圾值 ⇒ **不能用它证明 fp16 scratch 不可用**；fp16 路径的真实误差见上表 2 级链 2.8e-5、K5 的 3 级 9.7e-3 | 代码复核（`design_probe_accum_form.py` L35–L55）+ `examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py` K5 |

**参考**（按 Docs Auto Routing 优先级）：

1. `docs/Tilelang.language/线性代数操作/T.gemm.md`（fp32 HF32 路径 §2.2.1、`initC` 语义 §2.1、累加限制 §2.3）
2. `docs/Tilelang.language/内存操作/T.alloc_L1.md`、`T.alloc_L0C.md`、`T.store_fixpipe.md`（f32→f16/bf16 合法配对、rank 对齐规则）、`T.copy.md`（方向表、extent 推断规则、dtype 约束）、`T.load_nd2nz.md`
3. `docs/Tilelang.language/shape操作/T.vbrc.md`、`docs/Tilelang.language/数学操作/T.vexp.md`、`T.vsub.md`、`T.vmul.md`、`T.vmin.md`、
   `docs/Tilelang.language/数据类型转换操作/T.vcast.md`、`docs/Tilelang.language/创建操作/T.transpose.md`、`T.arange.md`
4. `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md`
5. `docs/开发指南.md` §3.3（物理核数与分核策略）、`.agents/skills/_shared/standards/core-split-strategy.md`、
   `.agents/skills/_shared/standards/hardware-cost-model.md`
6. `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（CONST-vector-launch-overhead / CONST-fabric-bw /
   CONST-capacity-910B2C / CONST-store-fixpipe-gm-only / CONST-aicore-910B2C / CONST-flag-id-budget）、
   `elementwise.md` PL-1.14（MTE3 段级延迟）、`traps-runtime.md` TRAP-load-nd2nz-strided、`layout.md`
7. `.agents/evolution/capability-gaps.md` CG-2026-0004 / CG-2026-0011、`.agents/evolution/queue.md` VP-2026-0087
8. `examples/gdn_prefill/DESIGN.md`（算子级 rev.2）、`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12、
   `examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py`、`13_l0c_fp16_scratch_cann9.py`、`14_two_loop_handshake.py`
9. `examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py`（同流水线 1/6 的命名与测试分层）
10. 源算子：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`（L158–L712）

---

## 12. 相对上一版（v0）的关键调整

### 12.1 4 个阻塞级问题的修法（逐条对照 REVIEW.md）

| REVIEW 问题 | v0 的写法 | 本版修法 | 同源证据 |
|---|---|---|---|
| **1. §3.3 广播方向反了**（实际算 `g_j-g_i`） | `vbrc((1,64))→g_row`、`vbrc((64,1))→g_col`、`vsub(g_row, g_col, d)` | 命名与语义对齐：`ub_gc(64,1)` → **`g_i` 矩阵**、`ub_gr(1,64)` → **`g_j` 矩阵**，`vsub(g_i_mat, g_j_mat)`；§1.6.2 V2 的算子清单、§3.1 S3、§3.3 三处统一；**并把该序列加进 D-1（A6）**，使方向错误今后由机检拦截（反例对照 viol 3.7e-1） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；`VP-2026-0087`；真机 `design_probe_gate_sign.py`（e 与 `exp(g_i-g_j)` 差 1.4e-6、与反方向差 1.07e14） |
| **2. `N` 的负号全链路丢失**（§0.6/§1.3 恒等式笔误） | `A=(I+N)^{-1}=ΣN^j`；§3.1/§3.3/§1.4/§1.6.2 无 `−1` | ① 恒等式改为 `N ≜ -S_sys`、`A = (I-N)^{-1} = Σ N^j = (I+S_sys)^{-1} = Π(I+N^{2^i})`（§0.6/§1.3）；② **落点唯一化**：`vcast(β)` 后 `vmul(β,-1)`（1 个 64-wide op，§3.1 S3 / §3.3 / §1.6.2 V2 / §3.2 四处同源）；③ `verify_equiv.py` 的 `doubling()` 用 `N=-S`，并新增 **符号裁决**（`|source-inv(I+S)|=2.8e-17` vs `|source-inv(I-S)|=4.4e-1`）与 A6 反例（viol 4.4e-1） | 源码 L480–L503/L500/L519–L540；`verify_equiv.py` 末尾 |
| **3. 输出落盘 API 语义错**（`size=[64,64]` 对 4D dst ⇒ `(head,64)` 平面） | `store_fixpipe(l0t, A_pad[0,base,hid,0], size=[64,64])`；§1.6.3 L4 称"`[head,64]` 切片本就连续" | 改为**三步落盘**：`fixpipe → y_blk`（连续 64×64，量化一次）→ `T.copy` 到 UB → **一次 strided region 写** `A_pad[base:base+64, hid:hid+1, 0:64]`；`A_pad` 改 3D（少一层 size-1 折叠）；§1.6.3 的 L4 论证改正（单 token 连续 ≠ 64 token 块连续，行距 `head·64`），并把 L4/L5 明确降为回退；roofline 重算（+16 KB 流量、+0.3 µs/任务）；L0-g/L0-e 覆盖最坏行距 | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1；`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2；真机 `design_probe_store_layout.py`（相邻槽零污染、时序 0.194 vs 0.084 µs） |
| **4. 复杂度表算术/依据不可追溯** | `S=32768` 6144 任务；B0 15/25 gemm；C1 11 链 gemm；16³ 0.7 µs；误引 `CONST-vector-launch-overhead` 为 0.25 µs；`r3_metrics` fail | ① `S=32768` → **8192 任务 / 410 每核 / 3.79 ms**；② B0 → **16/26**（`dim_k=128` 36，附源码行号）；③ C1 → **12 链 gemm（含 `gemm(I,I)` 播种）+ Gram = 13**，op 总数 **55**；④ 16³ 与 64³ **统一按 0.25 µs/op** 并显式标注"未实测假设"；⑤ 常数分离：Cube 0.25（本任务新增）/ Vector 0.5（既有 `CONST-vector-launch-overhead`）；⑥ R3 表列名改为 **`FLOPs` / `访存 Bytes` / `扫描遍数` / `中间缓冲峰值`**，`design_calc_check.py --design ...` 的 `r3_metrics` 现为 **pass** | §1.6.0 R3/roofline；`design_calc_check.py` 输出 |

### 12.2 建议级问题的处置（9 项）

| REVIEW 问题 | 处置 |
|---|---|
| 5. fp16 退路与 V4（err=1.875）冲突 | **V4 判废**：该分支在探针里根本没实现（if-chain 缺失 ⇒ 空执行 ⇒ 垃圾输出），已在本版实测重做（2 级链 fp16 rel 2.8e-5 vs fp32 5.5e-9）；§1.6.3 L2 与 §9 R1 的退路改为"fp16 链 scratch + fp16 操作数"，并给出**必须先过 L0 6 级链**的前置条件 |
| 6. §8.1 fp64 通道与 1e-10 矛盾 | `blocked_source` 在 `store_dtype=torch.float64` 时保留 fp64（实测 2.8e-17）；§8.1 拆成"结构层（1e-10 门槛）"与"量化层（fp16，信息性）"两个口径 |
| 7. §3.3 四处不可执行写法 | `f_s[slot]` → `F_S + slot`（+回退方案）；前导 `gram(bid, 0)`；`T.transpose` 直接删除（`g_j` 用第二次装载的 `(1,64)` 落位）；`β` 补 `vcast(round_mode="rint")`（**关键字**，实测位置写法编译失败） |
| 8. `g_ub/b_ub` 一维与 `vbrc` rank 冲突 | §4.1/§4.3 统一写 `ub_gc(64,1)` / `ub_gr(1,64)` / `ub_b16(64,1)` / `ub_b32(64,1)` / `ub_bn(64,1)`，并列入 §4.5 的小 buffer 明细 |
| 9. "gemm A/B 不能同 buffer"与事实不符 | 删除该断言，删掉 `l1q`（省 5 次 16 KB 装载 ≈ 80 KB/任务、5 个 op）；`P²` 直接用 `T.gemm(l1p, l1p, l0p)`，**本版实测 rel 5.5e-9** |
| 10. R1 未落笔参考表命中行 | R1 表新增 **R1-g（直接求解替代求逆，弃：contract 不同）** 与 **R1-h（对角/三角降阶，对应 C2，弃：op 数 58 vs 13）**，并各给弃选依据 |
| 11. O5/O12 动机误判 | O5 改为"门表复用（4096→4×64 次 exp）+ 除法复用"，并注明数值上全差式不更差；O12 改为"clamp 到末 chunk 幂等重算（非早退）"，并说明本设计"`if t<T` 守卫"是等价替换 |
| 12. UB alias 未列表化 | §4.3 新增 **alias 表**（`UB_A`/`UB_B`/`UB_F`/`UB_E` 的复用链）+ 两档预算（122 KB / 168 KB） |
| 13. 算子级 §6（Neumann）未裁决 | §0.6 新增裁决段：contract 不同（materialize 完整 A vs `(I+S)^{-1}rhs`）、量化对比（C3 ≥28 µs vs C1 9.25 µs）、不改算子级文档、建议 kernel 5/6 改 `gemm(A,rhs)` |

### 12.3 为什么不会再犯同一类错误（流程性改动）

1. **符号/方向类缺陷进入 D-1 机检**：新增 A6"伪代码逐句展开"，从 `g`/`β` 一路算到 `A`，与 fp64 解对比；
   并给出两条反例对照（方向 viol 3.7e-1、符号 viol 4.4e-1）证明该判据有牙齿。v0 的 4 个阻塞项里有 2 个
   属于"只在可执行层出现、散文论证看不见"的缺陷，此后再不会只靠散文。
2. **"API 语义"类断言一律用真机或前端源码定案**：落盘 region 语义由 `store_fixpipe` / `copy` 的 rank 规则 +
   `design_probe_store_layout.py` 双证；`vcast` 位置参数、`arange` 掩码这类"文档与实现不一致"的点，
   本版全部先用微探针试形（`design_probe_vcast_shapes.py` / `design_probe_mask.py`）。
3. **计数类数字一律带推导式与源码行号**：gemm 数（16/26/13）、任务数（8192）、op 数（55）都在文中给出
   相加过程，并用 `design_calc_check.py` 与 §5.1/§4.5 交叉对齐。
4. **探针的"空分支"必须被显式检查**：v0 的 V4 分支从未执行却给出"结论数字"，本版复盘把
   "每个 probe 分支都要打印实际执行的路径标签/断言"列为纪律（见 RETROSPECTIVE）。

### 12.4 已知的、本版未解决的事项

- `head=16`（2048 B 行距）的 strided 落盘时延未实测（R14，L0-g 覆盖）；`dim_k=128` 的 6 级 fp32 长链
  数值未在真机覆盖（R2/R3，L0-a/b 覆盖）。
- 工具链的 `T.arange`+`T.vcmp` 掩码路径静默失效属于**编译器能力缺口**；本设计已绕开（host 常量 tile），
  但若后续算子需要"按运行期值构造的 bool 掩码"，需按 `AGENTS.md` 的 Compiler Capability Gap Rule 登记。
- 算子级 §6 的 Neumann 方案与本函数的裁决只写在本文件 §0.6；算子级文档本身未修改（按本次任务边界）。

---
