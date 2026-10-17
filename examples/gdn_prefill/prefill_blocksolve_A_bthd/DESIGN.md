# prefill_blocksolve_A_bthd 算子设计文档

> GDN prefill 流水线**函数 2 / 6**。所属算子：`GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`）。
> 源算子（source of truth）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`
> 的 `_prefill_blocksolve_A_bthd_tl`（L158 起）与包装 `prefill_blocksolve_A_bthd`（L654 起）；
> 仓内快照 `examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/_gdn_prefill_kernels.py`（L174 / L670）。
> 工具链版本戳：**CANN 9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> （`/home/developer/tilelang-mlir-ascend`）/ **Ascend910B3**（`npu-smi 25.5.0`，Chip 6）/ 2026-10-17。
> 编程模式：**Expert**（`.bashrc` 默认 `TILELANG_ASCEND_MODE=Developer`，必须显式覆盖，见 §2）。
> 算子级迁移设计：`examples/gdn_prefill/DESIGN.md`（rev.2）；工具链实测：`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12。

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
| P7 | **块前代**（下三角块回代，共 15 次 16³ gemm + 中间 `work_s` 搬运）：`A_10 = -X_1 S_10 X_0`；`A_20 = -X_2(S_20 X_0 + S_21 A_10)`；`A_21 = -X_2 S_21 X_1`；`A_30 = -X_3(S_30X_0+S_31A_10+S_32A_20)`；`A_31 = -X_3(S_31X_1+S_32A_21)`；`A_32 = -X_3 S_32 X_2` | L542–L620 | `a_s`, `i_s` | 覆写 `a_s[1,3,4,6,7,8]` 为逆的非对角块 |
| P8 | 按行写回 `A`：行块 0..3 = `[i_s0 / (A_10,i_s1) / (A_30,A_31,i_s2) / (A_60,A_71,A_82,i_s3)]`，上三角显式写 0；`base+i < end` 才写 | L621–L650 | `i_s`, `a_s` | `A` |
| P9 | （仅 `l2norm`）把 `rnorm` 交给后续 kernel | L379–L385 | — | 本批跳过 |

**数据流图（源码硬件视角）**：

```
GM(k,g,beta) --cp.async/标量--> SMEM(k0..k3 16x64, g_s, beta_s)
   --> fragment(G_ij fp32, mma)  --共享内存-->  a_s[10,16,16] (fp16)
   --> i_s[4,16,16] (fp16, 寄存器行串行求解)  --> 15x gemm(16^3) 经 work_s 回代
   --> 按行 store --> GM(A)
```

所有中间量都在 SMEM/fragment 内，**没有任何跨 block 通信、没有中间量落回 GM**。

**host 侧逻辑**（L654–L716）：分配 `A`（`torch.empty`，形状 `[1,total_tokens,head,chunk_size]`）与 `k_rnorm`
（`l2norm` 为假时 1 token），按形状/开关查 `lru_cache` 编译好的 kernel，调用后返回 `(A, k_rnorm)`。
host **无任何数据变换**（grid 由 `GroupTiling.tile_upper_bound` 决定上界，多余 block 立即退出）。

### 0.4 优化手段解读（为什么快）

| # | 手段 | 源码位置 | 目的 | 机制 | 依赖的源硬件 |
|---|---|---|---|---|---|
| O1 | 每 chunk 一个 block、整块问题驻留 shared | L222–L255 | 把 64×64 问题钉在片上，零中间 GM 往返 | SMEM + fragment 复用 | CUDA shared memory / 大寄存器堆 |
| O2 | 只算 10 个下三角 Gram 块（而非 16 个） | L349–L358 | 省 37.5% Gram 乘加 | 利用对称性 | Tensor Core |
| O3 | `async_copy` + `ptx_wait_group` 预取 4 个 key 片 | L317–L324 | 掩盖 GM 延迟 | cp.async | 异步拷贝引擎 |
| O4 | Gram 按 `block_k=64` 分段累加到 fragment | L310, L349 | 支持 `dim_k=128` 且累加器常驻 | register 累加 | register file |
| O5 | 门比值用"块内参考 + 两次 exp + 一次除法" | L388–L505 | 避免 `exp(g_i-g_j)` 的指数范围/精度问题（两侧都 ≤ 1 / ≥ 1 各一侧被约束） | 数值稳定化 | 无（算法层） |
| O6 | 对角块用**行串行前代**而非求逆公式 | L507–L540 | O(16³/6) 而非 O(16³)，且天然下三角 | 顺序依赖链 | 32 lane × 2 列的线程映射 |
| O7 | 块前代：用 4 个小逆 + 15 次 16³ gemm 拼出 64×64 逆 | L542–L620 | 把 64³ 的逆降到 ~0.4 MFLOP（比 log-doubling 少 8× 乘加） | 分块消元 | Tensor Core（小 mma） |
| O8 | 奇偶分链（`e/o` 两个累加器）缩短行内依赖 | L529–L538 | 降低 16 步链的指令延迟 | ILP | 指令级并行 |
| O9 | 越界行 clamp + β=0 + 输出掩码，使"尾部 chunk"与整 chunk 同一程序 | L288–L324, L621 | 消分支、消 padding kernel | 数据无关掩码 | 无（算法层） |
| O10 | `T.annotate_layout(swizzled)` | L266–L285 | 消 SMEM bank conflict | swizzle | SMEM bank 结构 |
| O11 | 等长走算术解 chunk 边界（不读 offsets） | L256–L260, L186–L188 | 免一次 GM 读与依赖 | 常量折叠 | 无 |
| O12 | 网格取 offsets 上界、多余 block 立即退出 | L192–L196 | launch 范围与数据无关 | 早退 | 无 |

### 0.5 硬件耦合性分析与 NPU 适配决策

| # | 源方案 | 硬件前提 | 处置 | 依据 |
|---|---|---|---|---|
| O1 | 整块问题驻留 SMEM/fragment，中间量零 GM 往返 | CUDA shared + 大寄存器堆；L0C 无"存回片上"通路 | **重新设计** | NPU 的 Cube 结果只能 `fixpipe` 落 **GM**（`docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1，`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` CONST-store-fixpipe-gm-only），片内 L0C→L1 直连不存在 ⇒ 依赖链必须经 GM 往返，必须让**依赖级数**最小 |
| O2 | 只算下三角 Gram | 无 | **保留** | 数学对称性；NPU 侧等价为"整块 Gram + 掩码"，见 §0.6 |
| O3 | `async_copy` 预取 | cp.async | **等价替换** | `T.copy`/`T.load_nd2nz` 走 MTE2；两 slot 交替即覆盖同一意图（§6） |
| O4 | Gram 分段累加（fragment） | register file | **等价替换** | `T.gemm(..., initC=False)` 跨段累加 L0C（实测 err=0，`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1）；本设计进一步合并为单次 `size=[64,dim_k,64]` |
| O5 | 块内参考 + 除法 | 无 | **等价替换（并简化）** | `exp(g_i-g_j)` 与 `exp(g_i-r)/exp(g_j-r)` 数学恒等；NPU 上可直接算 64×64 差值外积 + 一次 `vexp`，省去 4 组参考与除法（§1.6.1 A1/A2） |
| O6 | 对角块**行串行**前代（16 步 × 内积） | 32 lane 线程映射 + 寄存器 | **重新设计** | NPU Vector 无"每 lane 一列 + 动态下标读 UB"的高效形态；16 步标量链在 Vector 上 ≈ 120 次窄 op（按 §1.6.0 实测 0.25 µs/op ≈ 30 µs/任务），且 `T.get_thread_binding` 线程模型在 npuir 不适用（`docs/Tilelang.language/` 无数线程绑定 API）⇒ 改用 **log-doubling**（§0.6 R1） |
| O7 | 4 小逆 + 15 次 16³ gemm 分块消元 | Tensor Core 小 mma 便宜 | **重新设计** | NPU 上每个 gemm 都要 L1/L0 装载 + L0C 落盘，**op 数**而非 MAC 数主导（§1.6.0 R3/R4）：15 次 16³ + 10 次 Gram = 25 个 gemm；改为 11 个 64³ gemm |
| O8 | 奇偶分链 ILP | 指令级并行 | **舍弃** | 随 O6 一并消失（无行串行链）；理由：该手段只服务于行串行求逆的延迟 |
| O9 | 越界行 clamp + β=0 + 输出掩码 | 无 | **保留（语义层）+ 局部等价替换** | clamp/β=0/掩码的语义必须保留；NPU 上"逐行 clamp 装载"代价高，改为「动态 extent 装载 + bool 掩码 select」（§3.5.2），数学等价 |
| O10 | swizzle 消 bank conflict | SMEM bank | **舍弃** | NPU 无 SMEM bank conflict；布局由 `load_nd2nz` 的 NZ 分形承担（`docs/Tilelang.language/内存操作/T.load_nd2nz.md`） |
| O11 | 等长算术解边界 | 无 | **保留** | 纯 host/常量算术（in-scope 等长） |
| O12 | 网格上界 + 早退 | 无 | **等价替换** | 改为"固定物理核数 + 核内 `T.serial` 任务循环 + `task < T` 守卫"（§5） |
| — | `T.Kernel(num_chunks, head)` 二维 grid | 三维/二维 block 索引 | **等价替换** | npuir 只接受一维 block 数（`.agents/skills/tilelang-op-design/references/ascend-constraints.md`）⇒ `cid → (task)` 一维展开 |
| — | `l2norm` 分支（P3/P9） | 无 | **舍弃（本批）** | in-scope `l2norm=False`（算子级 §1），保留 host 侧接口与 `k_rnorm` 占位返回 |

**统计**：保留 3 项（O2/O9/O11）+ 等价替换 4 项（O3/O4/O5/O12 + 二维 grid）+ 重新设计 3 项（O1/O6/O7）+ 舍弃 3 项（O8/O10/l2norm 分支）。

### 0.6 NPU 算法重设计

重设计的输入是 §1.6.0 的调研结论（选定 **R1 log-doubling 求逆**），不是源方案的直译。

**R1（承接 O1/O6/O7）：64×64 单位下三角求逆改为 log-doubling 乘积形式**

```
N = -S_sys           （严格下三角，故 N^64 = 0）
A = (I + N)^{-1} = Σ_{j=0}^{63} N^j = Π_{i=0}^{5} (I + N^{2^i})
  T_0 = I;  P_j = N^{2^j}
  T_{j+1} = T_j + P_j T_j        （gemm(P_j, T_j, L0C, initC=False)，L0C 内已有 T_j）
  P_{j+1} = P_j P_j              （j = 0..4；P_5 = N^32 用后即弃，不需要 N^64）
```

- 语义保持：`N` 严格下三角 ⇒ `N^64 = 0` ⇒ 级数在 64 项处截断；`{1,2,4,8,16,32}` 的子集和恰好覆盖
  `0..63`，乘积展开逐项等于 `Σ N^j` ⇒ **数学恒等**（机器验证见 §1.6.1 A4：rel ≤ 4.0e-6）。
- 边界语义：越界行 `β=0 ⇒ N 的该行为 0 ⇒ A 的该行为 e_i`，与源一致；该行在输出上被掩码（§3.5.2）。
- 代价：MAC 8×（0.39 M → 3.15 M），但**依赖级 25 gemm → 11 gemm**、无行串行链、无 16×16 小 gemm；
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
   （仓内先例：`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（L238）
   的 `x[off_m : off_m + real_m]`），越界行留在 L1/UB 的是上一个任务的旧值（有限数）。
2. **掩码**：`N = vselect(cond_strict_lower, β_i·exp(g_i-g_j)·G_ij, 0)`。用 **select 而非乘 0**：
   越界行/列的 `inf`/`NaN` 被选择丢弃而不参与算术（`0 × NaN = NaN` 的经典坑）。
3. **输出**：host 把 `A` 分配到 **chunk 对齐行数** `ceil(total_tokens/64)*64`，kernel 无条件写满 64 行，
   wrapper 返回 `A[:, :total_tokens]` 的**视图**（纯元数据操作）。合法行数值与源逐位一致。

**语义保持论证**：`i<valid` 且 `j<i` 时 `i,j` 都合法（因为 `j<i<valid`），故被保留元素只依赖合法
k/g/β；被丢弃元素恒为 0。又 `N` 的行 `i` 只经 `N^k` 的行 `i` 传播（矩阵乘行独立），越界行的垃圾值
**不可能污染合法行**。故合法行输出与 §0.1 定义逐元素相同（fp 舍入内）。

### 0.7 标杆实现

- 源实现：`prefill_prepare.py:158`（CUDA，SMEM + fragment + 行串行前代）。
- NPU 侧形态参考（Expert MixCV + 双循环握手）：`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`；
  回喂形态：`examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py`；
  Cube 侧调度先例：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`。
- 本函数两个设计期探针（D-3，实测结论见 §1.6.0/§9）：
  `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_doubling_cube.py`、
  `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py`。

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
N[i,j]   = - beta_i · exp(g_i - g_j) · (k_i · k_j)       0 <= j < i < valid      （严格下三角）
N[i,j]   = 0                                             其他（含 i < valid <= j）
A        = (I + N)^{-1} = Σ_{j=0}^{63} N^j = Π_{i=0}^{5} (I + N^{2^i})           （N^64 = 0）
A[i,i]   = 1,  A[i,j] = 0 (i < j)
A[0, base+i, hid, :] = A[i, :]                           仅 i < valid 被读出（其余落在 padding 区）
```

其中 `g_i` 是 chunk 内累积门（fp32），`beta_i` 是步长（in-scope 无 sigmoid），`k_i` 是该 head 的
`dim_k` 维 key 行。与 §0.1 的 `S_sys` 关系：`N = -S_sys`（源实现同样以 `a = -S_sys` 的对角块形式
写作 `X = I + aX`，见 `prefill_prepare.py:480`、`:528`）。

### 1.4 算法描述

迁移重设计后的 NPU 算法（与 §1.6.0 选定算法一致，见 §0.6）：

1. **任务映射**：逻辑任务 `t = (chunk, head)`，共 `T = nch_total × head`；固定启动 `K` 个 block（§5），
   每个 block 用静态上界的 `T.serial` 处理 `t = bid + it·K`（`it < ceil(T/K)`）。
2. **Cube（一个 scope）**：对任务 `t` 取 `k` 的 64×dim_k 片（动态 extent `valid` 行），
   一次 `T.gemm` 算整块 Gram `G = k kᵀ`（fp32 累加，`b_transpose=True` = 直接复用 `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.4 形态），
   落 GM（`gs_G[slot]`）。随后按 §1.6.1 A4 的 **6 级 log-doubling（5 次平方）** 求逆（11 个 gemm，fp32 操作数、
   GM scratch 回喂），最后 `store_fixpipe` 直出 `A`（f32→dtype 量化）。
3. **Vector（一个 scope）**：读 `gs_G[slot]` + `g`/`beta` 片段，按 §1.6.1 A1/A4 组出 `N`：
   行/列广播求差 → `vexp` → 乘 β/乘 G → `vselect` 严格下三角掩码，写 `ws_S[slot]`。
4. **两引擎流水**：各自跑同一任务列表，2 slot 双向 flag 握手（§7），Cube 侧领先一个任务算 Gram，
   使 Vector 的逐元素相位被 Cube 的链式相位覆盖。

### 1.5 数据流图

```
GM(k)   --MTE2(动态 extent valid 行)--> L1(k 片)  --T.gemm--> L0C(G) --fixpipe--> GM(gs_G[slot]) fp32
GM(g,beta) --MTE2--> UB --vbrc/vsub/vexp--> UB(e) --vmul(β,G)--> UB --> vselect(cond) --> UB(N)
        --MTE3--> GM(ws_S[slot]) fp32
GM(ws_S[slot]) --MTE2--> L1(P_0)                                   (Cube)
GM(ws_T)       --MTE2--> L1(T_j) --T.gemm(P_j,T_j,L0C,initC=False)--> L0C(T_{j+1}) --fixpipe--> GM(ws_T)
L0C(T_{j+1}) --(gemm P_j,P_j)--> L0C(P_{j+1}) --fixpipe--> GM(ws_P) --MTE2--> L1
... 6 级 ...
L0C(T_6) --store_fixpipe(f32->dtype, enable_nz2nd)--> GM(A[0, base, hid, 0], size=[64,64])
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

调研对象：**同一数学语义**（给定 64×64 单位下三角 `I + S_sys`，输出其逆）的算法族。基线候选 = 源算法。
调研深度：完整调研（矩阵类）。检索面：`references/algorithm-candidates.md`（线性代数 / 三角求解族）、
`pattern-library`（cases/layout/constants/traps）、仓内 `examples/`（`ssd_chunk_scan` 的分块状态族、
`multi_head_attention` 的 MixCV 形态）、源算子与 FLA 系列公开实现（模型知识层，§6 第 5 层）。

**R1 等价化简公式候选**（含基线；正式论证在 §1.6.1）

| 候选 | 公式/结构 | 等价性初判 | 收益方向 | 采纳 |
|---|---|---|---|---|
| **B0 源算法（基线）** | 4 个 16×16 对角块行串行前代（`X=I+aX`）+ 15 次 16³ gemm 块前代 | — （语义基准） | MAC 最少（0.39 M/task） | 基线 |
| R1-a | `exp(g_i-g_j)` 单次指数（替代"块内参考 + 两次 exp + 一次除法"） | 数学恒等；源式另含 fp16 门表舍入 | 每元素 2 exp+1 div → 1 exp；省 4 组参考表 | **采纳**（§1.6.1 A1） |
| R1-b | 除法 → 乘倒数（保留"参考因子"分解时的退路） | 恒等（`1/x` 一次） | 2016 次除法 → 2 次 64-wide 倒数 + 2 次 vmul | 采纳为**备选形态**（§1.6.1 A2；主选路径无除法） |
| R1-c | `β` 保持 fp32（源 `cast` 到 dtype） | 容差内等价（差异 = 1 次 fp16 舍入） | 少一次舍入 | **采纳**（§1.6.1 A3） |
| R1-d | 把 `-1` 折进掩码常量、`I` 由 gemm 初始化提供 | 恒等（代数重排） | 省 2 个 64×64 逐元素 op | **采纳**（§1.6.1 A4 的一部分） |
| R1-e | 把行/列门因子折进 key（`N = -(D K)(D' K)ᵀ`），Gram 直出带符号 `N` | 恒等（`rank-1` 缩放可结合） | 省 64×64 的两次 `vmul` | 弃：需把缩放后的 key 写回 GM（2×16 KB 写 + 16 KB 读）并仍要 64×64 掩码；净增 32 KB/任务流量与 2 个 op（按 0.25 µs/op、72 GB/s 折算 +0.9 µs），无收益 |
| R1-f | 只算下三角 Gram（源 O2） | 恒等（对称） | 省 37.5% Gram MAC | 弃（NPU 形态）：Cube 的一次 64×64×dim_k gemm 已是最少 op 数；裁三角需拆成 10 次 16×16 gemm = 回到 B0 的 op 数 |

**R2 在线算法：结论「对本函数无在线变体」**，结构依据三条：

1. 输出**就是** 64×64 逆矩阵本身（不是某个向量/标量统计量），任何流式变体最终仍要materialize这 64×64 个元素；
2. 跨 chunk 的流式结构已经存在——但它在 kernel 5/6 里（把 `A` 折进状态递推），本函数的 contract 是"逐 chunk 独立"，
   所以"在线化"只能是把递推搬进本函数 → 那是**改变函数 contract**，不是等价实现；
3. token 级在线形式（每次 rank-1 更新 `(I+β k kᵀ)` 再求逆）在本 chunk 内需要 64 次顺序更新（含 64 次 gemm + 逐元素），
   op 数是 C1 的 6 倍以上，且需要 64 次跨引擎往返 → 严格劣于任何候选（量化见 R3 的 C3 行）。
   *（该结论为结构判据 + 源算子自身"逐 chunk 分块"证据，非"未检索"式断言。）*

**R3 算法复杂度四口径对比**（口径：**每任务 = 1 个 chunk × 1 个 head，`C=64`、`dim_k=128`**；
MAC 只计乘加、不含掩码/指数折算；GM 流量 = 输入读 + 输出写 + 中间 GM 往返；"依赖级" = 必须串行等待的
gemm/矩阵级 op 数，是 NPU 上的实际主项；中间缓冲 = 单 block 峰值）

| 候选 | MAC/任务 | GM 流量/任务 | GM 往返遍数 | 依赖级(op) | 中间缓冲峰值 | 可并行度 |
|---|---|---|---|---|---|---|
| **B0 源算法直译** | 0.39 M | ~0.05 MB（16×16 块小，但 19 级往返） | 19 | **25 gemm + 120 窄向量 op** | ~10 KB | T = nch·head |
| **C1 log-doubling（选定）** | 3.15 M | ~0.50 MB | 10 | **11 gemm + 29 拷贝（40 op）** | 2×16 KB + 2 slot | T |
| C2 分层（16 块 + 对角 doubling） | 2.12 M | ~0.40 MB | 12 | 13 gemm（+ 4 块打包/索引代码） | 同上 | T |
| C3 Neumann 级数（算子级 §6 方案） | 16.8 M | ~2.7 MB | 62 | **62 gemm** | 2×16 KB | T |
| C4 Newton 迭代 `X←X(2I-MX)`（6 次） | 3.67 M | ~0.60 MB | 12 gemm + 6 次 CV 往返 | 12 gemm + 12 逐元素 | 3×16 KB | T |
| C5 纯 Vector 分块前代（B0 搬到 UB） | 0.39 M | ~0.03 MB | 1（Gram 也在 UB） | ~200 窄向量 op | ~100 KB UB | T（Vector 核 ×2） |
| C6 纯 Vector 全 64×64 行串行 | 0.39 M | ~0.03 MB | 1 | **2016 窄向量 op** | ~64 KB UB | T |

- **C1 的 MAC 是 B0 的 8.0×**，但 Cube 上一个 64³ gemm = 64 个 16×16×16 分形 op（利用率满），
  一次 16³ gemm 只有 1 个分形 op（发射/装载开销主导）；实测 64×64 fp32 的一次 gemm+store+load 平均
  **0.25 µs/op**（§9 探针），故 B0 的 25 gemm + 120 窄 op ≈ 25×0.7 + 120×0.25 ≈ 47 µs/任务，
  C1 ≈ 40×0.25 ≈ 10 µs/任务。**op 数才是主导项，MAC 不是。**
- **C3（算子级 §6 的 Neumann 方案）在本函数被否**：62 次依赖 gemm ≈ 15.5 µs gemm 时间 + 62 次
  GM 往返（2.7 MB/任务 ≈ 37 µs @72 GB/s/core）→ 至少 5× 于 C1；仅当"逆只作用于单个 rhs 向量"时
  Neumann 才划算（那是 kernel 5/6 的形态，不是本函数）。
- **C4 需要每级一次 Vector 参与**（`2I - MX` 的逐元素），即 6 次跨引擎往返 + 每次往返的 flag 延迟；
  且 Newton 的收敛依赖初值，必须凑满 6 级才精确 ⇒ 无优势。
- **C5/C6 全部落在 Vector**：单 op 最窄（16~64 lane），按实测 0.25 µs/op 折算 200 op ≈ 50 µs/任务，
  比 C1 慢 4~5×，且把 Cube 完全闲置（与算子级"MixCV 主干"结构不符）。

**R4 硬件亲和性逐候选评估**（检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合 / 跨核结构）

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/跨核 | 判定 |
|---|---|---|---|---|---|---|
| B0 | ✗ 对角求逆是 16 步**标量链**，落到 Vector 即 ~120 窄 op；落到 Cube 无法表达（gemm 不做前代） | ✓ | 16 分形对齐 ✓ | ✓ | 19 级 CV 往返 | 淘汰 |
| **C1** | ✓ 全部 MAC 在 Cube（64³ 满利用率）；逐元素仅 12 个 64-wide/64×64 op 在 Vector | ✓ UB 104 KB / L1 64 KB / L0C 48 KB | ✓ 64 是分形 16 的整数倍；fp32 尾轴 64×4 B = 256 B | ✓ 6 级在 Python 层静态展开 | ✓ 2 slot 双向握手（`examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py` 已验证） | **选定** |
| C2 | ✓ | ✓ | ✓ | △ 需 4 块对角打包 + 索引，代码分支多 | ✓ | 次优（同 MAC 量级但 op 更多） |
| C3 | ✓ | ✓ | ✓ | ✓ | ✗ 62 级 GM 往返 | 淘汰 |
| C4 | ✓ | ✓ | ✓ | ✓ | ✗ 6 次 CV 交接 | 淘汰 |
| C5/C6 | ✗ 矩阵乘落 Vector（每 lane 一元素，无 MAC 阵列） | ✓ | △ 16 lane 宽度浪费 | ✓ | ✓ 无跨引擎 | 淘汰 |

**定量 roofline（D-2，按 `.agents/skills/_shared/standards/hardware-cost-model.md` §2 口径，C1 主选）**

常数来源（版本戳见 §11）：`CONST-vector-launch-overhead`（向量/矩阵 op 发射 ~0.25 µs/op，64×64 fp32 实测）、
`CONST-fabric-bw`（per-core fabric ≈ 72 GB/s）、`CONST-store-fixpipe-gm-only`（L0C 只能落 GM）、
`CONST-capacity-910B2C`（UB 192 KB / L1 512 KB / L0C 128 KB）。

| 项 | 估算 | 值 |
|---|---|---|
| 流量项 | 0.50 MB/任务 ÷ 72 GB/s | **6.9 µs/任务** |
| 发射项 | 40 op × 0.25 µs | **10.0 µs/任务** |
| 容量项 | UB 104 KB/192 KB、L1 64 KB/512 KB、L0C 48 KB/128 KB | 不约束 |
| **估算下界 = max(流量, 发射, 容量)** | | **10.0 µs/任务** |
| 合计（`S=4096,H=16` → 1024 任务 / 20 核 = 52 任务/核） | | ≈ **0.52 ms** |
| 合计（`S=32768,H=16` → 6144 任务 / 20 核 = 308 任务/核） | | ≈ **3.1 ms** |

**调研结论**：选定 **C1 = log-doubling 乘积形式**（`A = Π_{i=0..5}(I+N^{2^i})`，11 个 64³ gemm，Cube 承担全部
MAC，Vector 只做 12 个宽 op 的掩码/组装）。关键依据：① R1-a/A3 的恒等化简使逐元素相位从"8 次 64-wide exp
+ 2016 次除法 + 4 组 fp16 门表"降到"1 次 vsub + 1 次 vexp + 2 次 vmul + 1 次 vselect"；② R3 显示 NPU 上
**依赖级 op 数**主导（B0 47 µs vs C1 11 µs/任务），为此接受 8× MAC；③ R4 显示 C1 是唯一"全 MAC 在 Cube、
无行串行链、无逐级 CV 往返"的候选；④ 精度上 C1 反而最优（§1.6.1 A4：fp32 全链路 rel ≤ 4.0e-6，
源 B0 的 fp16 中间块约 1e-4 且 `|A| > 65504` 会溢出）。**B0 的"少算乘加"（O7）与"行串行省 6× 计算"（O6）
优化意图被显式替换**（理由：NPU 上 MAC 便宜、op 贵），其"整块驻留片上"（O1）意图由"GM 往返级数最小化
（19 → 10）"承接；"越界行 clamp+掩码"（O9）意图由 §0.6 R3 承接。

#### 1.6.1 数学等价优化（公式级）

| 项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|---|---|---|---|
| **A1** 门比值单次指数 | 源式（L388–L427、L429–L505）：按 16 行块参考 `r=16b`，`gate_b[t]=exp2((g_t-g_r)·log2e)`（**存 dtype**），再算 `s = β_i·gate_b[i]/gate_b[j]` | `e[i,j] = exp(g_i - g_j)`（一次 64×64 外差 + 一次 `vexp`） | 数学恒等：`exp(a-r)/exp(b-r) = exp(a-b)`；机器验证 **A1**：worst rel 4.3e-8、max ulp 0.5、viol 0（且本式没有源的 fp16 门表舍入，更接近 fp64 语义基准） | 每元素 2 exp + 1 div → 1 exp；实现从"4 组 64-wide exp 表 + 4096 次除法"变成"2 次 `vbrc` + 1 次 `vsub` + 1 次 `vexp`"（-2 op，-1 个 `vdiv` 相位 ≈ -0.3 µs/任务）；同时去掉 fp16 门表中转（精度 +） |
| **A2** 除法→乘倒数（备选形态） | `exp(g_i-r)/exp(g_j-r)` | `exp(g_i-r)·exp(r-g_j)` | 恒等（`1/exp(x)=exp(-x)`；一次倒数 vs 4096 次除法）；机器验证 **A2**：worst rel 9.0e-8、max ulp 1.3、viol 0 | 主选路径（A1）已无除法；该形态作为"若 64×64 `vexp` 代价超预期"的退路，an op 数 3 vs 4、省 4096 次除法 |
| **A3** β 保持 fp32 | `beta_s[t] = cast(step, dtype)`（L295），后续按 fp16 值参与乘 | β 全程 fp32（`vbrc` 到 64×64 后直接乘） | 容差内等价：差异恰为 1 次 fp16 舍入（≤1 ulp）；机器验证 **A3**：worst rel 3.7e-4、viol 0 | 少一次舍入（精度 +），零额外成本（β 只在 Vector 用一次） |
| **A4** 求逆用 log-doubling | `A = Σ_{j<64} N^j`（源用分块前代，等价于 63 项级数截断） | `A = Π_{i=0}^{5}(I + N^{2^i})`；`T_{j+1}=T_j + P_jT_j`（`initC=False` 累加）、`P_{j+1}=P_j²`；`-1` 折进掩码常量、`I` 由 `gemm(I,I)` 初始化（含 R1-d） | 数学恒等：`N` 严格下三角 ⇒ `N^64=0`；`{1,2,4,8,16,32}` 的子集和唯一覆盖 `0..63` ⇒ 乘积展开 = 级数。机器验证 **A4**：worst rel 4.0e-6、max ulp 1.8、viol 0（含 `β×20`、`k×4`、尾 chunk、零 β 行等角点） | 依赖级：25 gemm（B0 直译）/ 62 gemm（Neumann）→ **11 gemm**；无 16 步行串行链；fp32 全链路（源 fp16 中间块 rel≈1e-4，`|A|>65504` 时溢出） |
| **A5** 单次量化直出 | 中间逆块存 dtype（`solve_dtype=dtype`，L200/L242），逐块 cast | 链全程 fp32，最后一次 `store_fixpipe` 做 f32→dtype 量化 | 容差内等价：仅保留最终一次舍入；机器验证 **A5**：worst rel 9.3e-5、max ulp 1.0、viol 0 | 少 10 次 fp16 中间存储/读回（≈ -160 KB GM 流量/任务）、去掉 fp16 溢出面 |

**机器验证结果表**（`examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py` 实跑，torch CPU，fp64 参照；
判据 `|cand-ref| <= 5e-3·|ref| + 2·eps_fp16·max|ref|`，viol = 违反率）：

| case | A1 exp(a-b) rel/viol | A2 mul rel/viol | A3 beta-fp32 rel/viol | A4 doubling rel/viol | A5 fp16-cast rel/viol |
|---|---|---|---|---|---|
| SC-typical | 3.2e-08 / 0.0e+00 | 6.4e-08 / 0.0e+00 | 2.2e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 |
| SC-strong-decay | 2.5e-08 / 0.0e+00 | 6.8e-08 / 0.0e+00 | 1.4e-04 / 0.0e+00 | 7.1e-09 / 0.0e+00 | 0.0e+00 / 0.0e+00 |
| SC-no-decay | 4.3e-08 / 0.0e+00 | 8.7e-08 / 0.0e+00 | 1.8e-04 / 0.0e+00 | 2.4e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 |
| SC-beta-x20 | 4.2e-08 / 0.0e+00 | 8.6e-08 / 0.0e+00 | 3.7e-04 / 0.0e+00 | 4.0e-06 / 0.0e+00 | nan / 0.0e+00 |
| SC-k-x4 | 3.7e-08 / 0.0e+00 | 7.5e-08 / 0.0e+00 | 2.7e-04 / 0.0e+00 | 1.4e-06 / 0.0e+00 | 9.3e-05 / 0.0e+00 |
| SC-tail-56 | 3.9e-08 / 0.0e+00 | 7.9e-08 / 0.0e+00 | 2.6e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 |
| SC-zero-beta-rows | 4.3e-08 / 0.0e+00 | 9.0e-08 / 0.0e+00 | 2.7e-04 / 0.0e+00 | 1.3e-08 / 0.0e+00 | 0.0e+00 / 0.0e+00 |
| SC-near-overflow | 3.4e-08 / 0.0e+00 | 7.6e-08 / 0.0e+00 | 2.6e-04 / 0.0e+00 | 5.0e-09 / 0.0e+00 | 0.0e+00 / 0.0e+00 |

| 项 | worst rel | worst ulp | worst viol | 结论 |
|---|---|---|---|---|
| A1 | 4.3e-08 | 0.5 | 0.0e+00 | **EQUIV_PASS** |
| A2 | 9.0e-08 | 1.3 | 0.0e+00 | **EQUIV_PASS** |
| A3 | 3.7e-04 | 3957.5 | 0.0e+00 | **EQUIV_PASS** |
| A4 | 4.0e-06 | 1.8 | 0.0e+00 | **EQUIV_PASS** |
| A5 | 9.3e-05 | 1.0 | 0.0e+00 | **EQUIV_PASS** |

> `SC-beta-x20` 的 A5 出现 `nan`：该 case `|A| ≈ 3e4`，fp16 输出上溢（`> 65504` 的中间逆块）——这是**源算法**的
> 溢出边界（源把逆块存 fp16），本设计 fp32 链路不受影响（A4 该 case rel 4.0e-6 正常）；列在表内仅为记录该边界。
> `A3` 的 worst ulp 大是因为 ULP 以 fp32 尺度度量 fp16 舍入量（3957 fp32-ulp ≈ 1 fp16-ulp），viol=0 才是判据。

**优化结论**：采纳 A1/A3/A4/A5（A2 作为退路形态一并验证通过）；五项均 `EQUIV_PASS`，viol 全 0。
其中 A4 是算法级恒等（`Σ N^j` → 乘积形式），A1/A3/A5 是"更少 op / 更少舍入"的公式级优化；
A1 使逐元素相位从 **~2016 次除法 + 8 次窄 exp** 降到 **1 次 vsub + 1 次 vexp**（4096 元素宽 op）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案（§3.3 伪代码）中**全部**循环与标量计算点：

| # | 循环 / 标量点 | 位置 | 处置 | 依据 |
|---|---|---|---|---|
| V1 | 任务→`(chunk, head, base, end, valid, slot)` 映射（`t//head`、`t%head`、`it%2`） | 两 scope 的任务循环头 | **保留标量** | block 级索引 / 元数据（每任务 ~6 条标量，非逐元素热点）；等价向量化无意义 |
| V2 | `S_sys` 逐元素构造（源 L429–L505 的 `for i,j in T.Parallel(16,16)` 共 6 次 + 4 个对角块） | Vector | **已向量化**：`vbrc`(g→64×64 行/列) + `vsub` + `vexp` + `vmul`(β 广播) + `vmul`(G) + `vselect`(cond) | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（(M,1,K)→(M,N,K) 广播合法）、`docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1（fp32 ✓）、`docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 |
| V3 | 源的门参考表 + 每元素除法（L388–L429、L429–L505 的 2016 次 `/`） | Vector | **已消除**（A1：`exp(g_i-g_j)`） | §1.6.1 A1（EQUIV_PASS）；`vdiv` 相位整体消失 |
| V4 | 4 个 16×16 对角块行串行前代（源 L512–L540：`for r in T.unroll(16)` × 内层 `for m in T.unroll(r)`） | — | **已消除**（算法级） | 被 A4 的 6 级 gemm 链替代；无行串行链、无 `T.alloc_var` 标量累加 |
| V5 | 15 次块前代 gemm + `work_s` 搬运（源 L542–L620） | — | **已消除**（算法级） | 同一 64×64 逆由 11 个 64³ gemm 给出（`T.gemm` 原生，无需逐块搬 UB） |
| V6 | 输出逐行块写回 + 逐行 `if base+i < end`（源 L622–L651） | Cube | **已向量化/消除** | `store_fixpipe(size=[64,64], enable_nz2nd=True)` 一次落 64 行；边界改由"host 侧 chunk 对齐 padding + 返回切片"承担（§0.6 R3），kernel 内无逐行分支 |
| V7 | 越界行 clamp（源 L288–L298、L317–L324 的逐行 `min(base+t, end-1)`） | Vector/Cube | **已消除** | 改为动态 extent 装载（`valid` 行）+ bool 掩码 select；`T.copy` 动态 extent 有仓内先例（`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py` 的 L238） |
| V8 | 6 级 doubling 的 `j` 循环 | Cube | **保留为 Python 静态展开**（不生成 TIR 循环） | 每级读写不同 slot/buffer 且末级无平方，展开后边界全静态（`.agents/skills/_shared/standards/core-split-strategy.md` §1 要素③"循环边界静态"）；不用 `T.serial` 也可避免"循环携带 Cube 依赖"这一历史风险面（`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §10） |
| V9 | 掩码 `(i>j)` 构造 | Vector | **一次/block，非逐任务**：`T.arange`(1,64)×2 → `vbrc` → `vcmp(...,"gt")` → bool tile（4 KB） | `docs/Tilelang.language/创建操作/T.arange.md` §2.3.1（二维 arange 逐元素 scalar 写，~1 ns/elem ⇒ 必须用 1-D + `vbrc`）；摊到 ~52 任务/block ≈ 0.08 µs/任务 |
| V10 | Gram 的 `dim_k` 规约 | Cube | **不适用向量化**：一次 `T.gemm(size=[64,dim_k,64], b_transpose=True)` 交给 Cube 分形 | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.4；`dim_k ∈ {64,128}` 均满足 K 维分形 16 对齐 |

**向量化结论**：逐元素相位（V2/V3/V7）全部落在 64-wide 或 64×64 的 v-prefix 宽 op 上（共 **12 个 op/任务**），
源的两处**标量循环**（V4 的 16 步行串行、V6 的逐行写回）由算法重设计与 `store_fixpipe` 整体消除；
仅 V1（block 级索引）与 V9（每 block 一次的常量掩码）保留标量/索引构造，属"block 级索引 / 常数物化"类别，
不构成逐元素热点。

#### 1.6.3 向量化轴与数据布局决策

**I/O 契约（不可变）**：`k[1,S,kh,dim_k]`、`g/beta[1,S,head]`、`A[1,S,head,64]`（尾轴 64 连续）。
**核内布局是设计变量**，按 MixCV/Cube 类清单枚举：

| 候选 | 形态 | 整除性/对齐 | repack 代价 | UB/UB 预算 | 结论 |
|---|---|---|---|---|---|
| **L0 原生长宽 + 行主序 64×64（选定）** | Gram/链一律 64×64 fp32 行主序（尾轴 64 = 256 B）；Cube 侧 L1 操作数 64×64 fp32、`A` 直出 `enable_nz2nd` | 64 = 分形 16 的整数倍；尾轴 256 B = 2×128 B 段 | **零**（`A` 的 `[token, head, 64]` 与 64×64 行主序同构；`k` 片按 `load_nd2nz` 自然转 NZ） | UB 104 KB；L1 64 KB；L0C 48 KB | **采纳** |
| L1 `k` 转置后按 `dim_k` 作向量轴 | 把 64×dim_k 转成 dim_k×64 | dim_k 64/128 ✓ | 需核内 `T.transpose` 链（每次 64×128 = 8 K 元素的 VTransposeOp） | +32 KB UB | 弃：Gram 走 Cube 分形，`dim_k` 不需要成为向量轴；转置纯增代价（`docs/Tilelang.language/创建操作/T.transpose.md`） |
| L2 链矩阵用 fp16 scratch | 中间 T/P 存 fp16 | ✓ | 零 repack，但每级 1 次量化 | UB/L1 减半、GM 流量 -50% | 弃（主选）：精度面（§1.6.1 A4/A5 要求 fp32 链，fp16 会把 rel 抬到 ~1e-3 并引入 `|P|` 溢出）；**保留为 Stage 4 备选**（若流量成为瓶颈，按 §9 风险条的裁决阈值评估） |
| L3 两个 head 打包成 128×128 块对角链 | 摊薄每 head 的 op 数 | ✓ | 需 128×128 块对角打包/掩码 | L0C 2×64 KB = 128 KB = 上限 | 弃（主选）：L0C 顶格，无余量给 G/T/P 三缓冲；列为 Stage 4 结构优化候选 |
| L4 `A` 先写 `[64,64]` 暂存再按 `[token,head]` 重排 | 便于批量写 | ✓ | 需 GM 级重排 | — | 弃：多一趟 GM 往返，且 `A` 的 `[head,64]` 切片本就连续 |

**布局决策结论**：采纳 **L0**——Cube 侧一律 64×64 fp32 行主序（`k` 片经 `load_nd2nz` 进 L1、`b_transpose=True`），
Vector 侧 64×64 fp32 行主序、向量轴 = 尾轴（64 lane，256 B 对齐），`A` 直接以 `[64,64]` 切片
`store_fixpipe(enable_nz2nd=True)` 落 `A[0, base, hid, 0]`；**不做任何核内重排**。fp16 链（L2）与双 head
打包（L3）留在 Stage 4 作为流量/时延备选，其裁决阈值见 §9。

---

## 2. 编程模式选型

### 2.1 模式结论

**Expert**（同一 kernel 内一个 `T.Scope("Cube")` + 一个 `T.Scope("Vector")`，手动 L1/L0C/UB 分配与手动同步）。

### 2.2 选型理由

1. 本函数必须同时用 Cube（Gram + 11 个链 gemm）与 Vector（门/β/掩码 12 个宽 op），是 MixCV；
2. CANN 9.0 实测：Developer 模式下"Cube+Vector 同 kernel"会走错流水（`hivm.hir.store` 约束、
   `multiple entry functions detected`），Expert 下 `Conv3dBiasFwdOp` 17/17 通过
   （`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12.0/§12.1）；
3. 单 kernel 只能有 **一个** Cube scope + **一个** Vector scope（`EXPERT_FINDINGS.md` §12.2），
   因此两引擎必须各自跑"整段任务循环"并用 flag 握手（§7），这只有 Expert 能表达。

### 2.3 模式强制方式（必须写死，不得 `setdefault`）

- **进程级**：模块导入时 `os.environ["TILELANG_ASCEND_MODE"] = "Expert"`（**赋值**，非 `setdefault`）——
  `.bashrc` 导出的是 `Developer`，`setdefault` 静默失效曾导致 11 个探针得出 3 条假限制
  （`EXPERT_FINDINGS.md` §12.0）。
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
| S1 | `t = bid + it·K`；`base = (t//head)·S + (t%head 对应的 chunk)·64`、`valid = min(64, end-base)` | 任务映射（V1，标量） |
| S2 | `G = k_片 · k_片ᵀ`（64×64，K 维 = `dim_k`） | 一次 Cube gemm |
| S3 | `d[i,j] = g_i - g_j`；`e = exp(d)`；`N = vselect(i>j, β_i·e·G, 0)` | Vector；含 §0.6 R3 的掩码语义 |
| S4 | `T ← I`；`P ← N` | `T` 由 `gemm(I,I)` 给出（L0C 只能被 gemm 初始化） |
| S5 | `T ← T + P·T`（`initC=False` 累加进 L0C） | 6 次（lev 0..5） |
| S6 | `P ← P·P` | 5 次（P_1..P_5；不需要 `N^64`） |
| S7 | `A[0,base,hid,:] = T`（f32→dtype 量化） | `store_fixpipe` |

### 3.2 TileLang API 映射

| 步 | API | 参数要点 | 佐证 |
|---|---|---|---|
| S2 装载 | `T.copy(k[0, base:base+valid, khid, 0:dim_k], l1k)` | 动态 extent（`valid`） | `docs/Tilelang.language/内存操作/T.copy.md`；先例 `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（L238） |
| S2 gemm | `T.gemm(l1k, l1k, l0g, size=[64, dim_k, 64], b_transpose=True, initC=True)` | fp16/bf16 输入、fp32 累加；`dim_k ∈ {64,128}` | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1/§2.3（fp16×fp16 必须 fp32 累加） |
| S2 落盘 | `with T.rs("PIPE_FIX"): T.copy(l0g, gs_G[slot])` | f32→f32 | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1（无 dtype 变换） |
| S5/S6 | `T.gemm(l1p, l1t, l0t, size=[64,64,64], initC=False)` / `T.gemm(l1p, l1c, l0p, size=[64,64,64], initC=True)` | `initC=False` = 累加进当前 L0C（**已实测 err=0**） | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1；`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1；本设计探针 `design_probe_accum_form.py` V3 |
| S4 初始化 | `T.gemm(l1e, l1e, l0t, initC=True)`（`l1e` = 64×64 fp32 单位阵，host 常量） | 给 L0C 一个初值 | 探针 `design_probe_doubling_cube.py`（T1 = I+N 实测 err 2.9e-8） |
| S3 广播 | `T.vbrc(g_ub_1x64, g_row)` / `T.vbrc(g_ub_64x1, g_col)` | (1,N)→(M,N)、(M,1)→(M,N) | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 |
| S3 差值/指数 | `T.vsub(g_row, g_col, d)`、`T.vexp(d, e)` | fp32 | `docs/Tilelang.language/数学操作/T.vsub.md`、`T.vexp.md` §2.2.1 |
| S3 乘 | `T.vmul(beta_bcast, e, t1)`、`T.vmul(t1, G_ub, t2)` | fp32，同 shape | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 |
| S3 掩码 | `T.vselect(cond, t2, zero_ub, n_ub)`；`cond` 由 `T.arange`+`T.vbrc`+`T.vcmp(...,"gt")` 每 block 造一次 | bool 仅片上 | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1；`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3（bool 不可落 GM）；`docs/Tilelang.language/创建操作/T.arange.md` §2.3.1 |
| S3 回写 | `with T.rs("PIPE_MTE3"): T.copy(n_ub, ws_S[slot]); T.sync_block_set(f_s[slot])` | f32→f32 | `docs/Tilelang.language/同步管道操作/T.sync_block_set.md` |
| S7 输出 | `T.store_fixpipe(l0t, A[0, base, hid, 0], size=[64,64], enable_nz2nd=True)` | f32→fp16/bf16 量化合法 | `docs/Tilelang.language/内存操作/T.store_fixpipe.md` §2.1（四种合法配对含 `f32→f16`/`f32→bf16`） |
| 同步 | `T.sync_block_set(id)` / `T.sync_block_wait(id)` + `with T.rs("PIPE_FIX"/"PIPE_MTE2"/"PIPE_MTE3")` | 4 个 flag id（§7） | `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md`；形态 `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py` |

### 3.3 计算伪代码

```python
# ---- host（wrapper）----
A_pad = torch.empty(1, ceil(total_tokens/64)*64, head, 64, dtype=k.dtype, device=k.device)
eye   = torch.eye(64, dtype=torch.float32, device=k.device)[None]        # 常量输入
gs_G  = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)  # Gram 交接 2 slot
ws_S  = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)  # N   交接 2 slot
ws_T  = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)  # T_j 轮转 2 slot
ws_P  = torch.empty(K, 2, 64, 64, dtype=torch.float32, device=k.device)  # P_j 轮转 2 slot
kernel(k, g, beta, eye, gs_G, ws_S, ws_T, ws_P, A_pad)
A = A_pad[:, :total_tokens]                                              # 纯视图
return A, k_rnorm                                                        # k_rnorm 占位（l2norm=False）

# ---- kernel ----
with T.Kernel(K, is_npu=True) as (bid, _):
    # ============= Cube：Gram + 6 级 log-doubling（5 次平方）=============
    with T.Scope("Cube"):
        l1k = T.alloc_L1([64, dim_k], dtype)      # k 片
        l1e = T.alloc_L1([64, 64], "float32")     # 单位阵常量（T_0 / 首级 B 操作数）
        l1p = T.alloc_L1([64, 64], "float32")     # P_j（gemm A 操作数）
        l1q = T.alloc_L1([64, 64], "float32")     # P_j 第二份（gemm A/B 不能同 buffer）
        l1t = T.alloc_L1([64, 64], "float32")     # T_j（gemm B 操作数）
        l0g = T.alloc_L0C([64, 64], "float32")
        l0t = T.alloc_L0C([64, 64], "float32")    # T 累加器（跨级 store 后复用）
        l0p = T.alloc_L0C([64, 64], "float32")
        T.copy(eye[0], l1e)
        # 前导：只算首个任务的 Gram（其余在循环内提前一拍）
        gram(bid)                                                 # 见下 macro
        for it in T.serial(NPER):                                 # NPER = ceil(T/K)，静态
            t = bid + it * K
            slot = it % 2
            if t < T:
                # 1) 预取下一个任务的 Gram（与 Vector 处理 t 重叠）
                if it + 1 < NPER and t + K < T:
                    gram(t + K)                                   # 写 gs_G[(it+1)%2]，set f_g
                # 2) 等 Vector 组好 N(t)，取 P_0 = N
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(f_s[slot])
                    T.copy(ws_S[bid, slot], l1p)
                    T.copy(ws_S[bid, slot], l1q)
                # 3) T_0 = I（L0C 只能被 gemm 初始化）；首级的 B 操作数直接用 l1e
                T.gemm(l1e, l1e, l0t, size=[64,64,64], initC=True)
                for lev in range(6):                              # Python 静态展开，非 TIR 循环
                    if lev == 0:
                        T.gemm(l1p, l1e, l0t, size=[64,64,64], initC=False)   # T_1 = N + I
                    else:
                        T.copy(ws_T[bid, lev % 2], l1t)                       # T_lev
                        T.gemm(l1p, l1t, l0t, size=[64,64,64], initC=False)   # T += P@T
                    if lev < 5:
                        T.gemm(l1p, l1q, l0p, size=[64,64,64], initC=True)    # P²
                        with T.rs("PIPE_FIX"):
                            T.copy(l0t, ws_T[bid, (lev + 1) % 2])             # T_{lev+1}
                        with T.rs("PIPE_FIX"):
                            T.copy(l0p, ws_P[bid, (lev + 1) % 2])             # P_{lev+1}
                        T.copy(ws_P[bid, (lev + 1) % 2], l1p)
                        T.copy(ws_P[bid, (lev + 1) % 2], l1q)
                with T.rs("PIPE_FIX"):
                    T.store_fixpipe(l0t, A_pad[0, base, hid, 0], size=[64,64], enable_nz2nd=True)

    # ================= Vector：g/β → N =================================
    with T.Scope("Vector"):
        build_cond_once()          # arange(1,64)×2 → vbrc → vcmp("gt") → bool[64,64]
        for it in T.serial(NPER):
            t = bid + it * K
            slot = it % 2
            if t < T:
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(f_g[slot])
                    T.copy(gs_G[bid, slot], G_ub)                        # 64×64 f32
                    T.copy(g[0, base:base+valid, hid:hid+1], g_col_ub[0:valid, 0:1])    # (64,1) 列切片
                    T.copy(g[0, base:base+valid, hid:hid+1], g_t_ub[0:valid, 0:1])      # 再转置成 g_row
                    T.copy(beta[0, base:base+valid, hid:hid+1], b_col_ub[0:valid, 0:1])
                T.transpose(g_t_ub, g_row_ub)                              # (64,1) -> (1,64)
                T.vbrc(g_row_ub, g_row); T.vbrc(g_col_ub, g_col)
                T.vsub(g_row, g_col, d_ub); T.vexp(d_ub, e_ub)
                T.vbrc(b_col_ub, b_bcast); T.vmul(b_bcast, e_ub, t1_ub)
                T.vmul(t1_ub, G_ub, t2_ub); T.vselect(cond, t2_ub, zero_ub, n_ub)
                with T.rs("PIPE_MTE3"):
                    T.copy(n_ub, ws_S[bid, slot]); T.sync_block_set(f_s[slot])
```

`gram(t)` macro：`T.copy(k[0, base:base+valid, khid, 0:dim_k], l1k)` →
`T.gemm(l1k, l1k, l0g, size=[64,dim_k,64], b_transpose=True, initC=True)` →
`with T.rs("PIPE_FIX"): T.copy(l0g, gs_G[bid, slot]); T.sync_block_set(f_g[slot])`。

### 3.4 API 可行性确认

| 断言 | 佐证 |
|---|---|
| Expert 下 fp32（HF32）gemm 64×64×64 可用且精度 ≈ fp32 | 本设计探针 `design_probe_doubling_cube.py`：`T1=I+N` err 2.9e-8、`T2=(I+N²)(I+N)` err 2.7e-7；`design_probe_accum_form.py` V1–V3 全通过（err 2.9e-8） |
| `initC=False` 累加进"已落过盘"的 L0C 可用 | 探针 V3（`gemm→store→gemm(initC=False)`，err 2.9e-8）；`examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py` K1（err 0） |
| L0C→GM(fp32 scratch)→L1 回喂可用 | `examples/gdn_prefill/probe/experiments/13_l0c_fp16_scratch_cann9.py`（fp16 scratch，err 9.7e-3）+ 本探针 fp32 scratch（err 2.7e-7） |
| 回喂**不得**经 `out_idx` 张量 | `EXPERT_FINDINGS.md` §12.1（K2：err 9.7~32）；本探针第一版读回 `out_idx` 张量直接触发 bishengir SIGSEGV ⇒ 设计一律用独立 scratch 输入张量（`gs_G`/`ws_S`/`ws_T`/`ws_P`） |
| 每级 (gemm+store+load) 的 op 代价 | 探针 `PROBE-TIMING`：6 op/级 = **1.52 µs** ⇒ 0.25 µs/op（64×64 fp32） |
| 双循环 + 双向多轮握手 | `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`（err 9.8e-4 vs fp16 通路参考） |
| `T.vselect` / `T.vcmp` / `T.vbrc` 的 dtype 与广播形态 | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2、`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3、`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

- **三维 Kernel**：不使用（`T.Kernel(K)` 一维）；`(chunk, head)` 展平为 `t`。
- **`threads=` 参数**：npuir 下无效（`pattern-library/traps-compiler.md` TRAP-threads-kwarg-noop），不使用。
- **L0C 容量**：`l0g + l0t + l0p = 3 × 16 KB = 48 KB` < 128 KB ✓（`CONST-capacity-910B2C`）。
- **L1 容量**：`l1k(16 KB, dim_k=128) + 4 × 16 KB = 80 KB` < 512 KB ✓。
- **UB 容量**：见 §4.5（≈104 KB 手工 → ×1.10~1.12 膨胀 ≈ 117 KB < 192 KB）✓。
- **GEMM 非整除**：`64`（M/N）与 `dim_k ∈ {64,128}`（K）全部是分形 16 的整数倍，**无尾块**；
  真正的"非整除"只在**行数**（序列尾 chunk），由 §0.6 R3 的装载 extent + 掩码 + padding 输出处理。
- **Cube 环路/回喂**：不使用 `out_idx` 回喂（§3.4）；每级只有一次"store 后复用作 gemm dst"，
  已被探针 V2（store-then-reuse）与本探针时序核（`T.serial` 内复用 L0C）证实可用。
- **单 kernel 一段 Cube + 一段 Vector**：满足（§2）。

#### 3.5.2 参考实现差异说明

1. **越界行处理**：源用逐行 clamp（重复末行），本设计用"动态 extent 装载 + `vselect` 掩码 + padding 输出"。
   语义等价性见 §0.6 R3 的三段论证（行独立传播 + 丢弃元素恒 0）。差异：`A` 的 padding 行会被写入
   单位行（源不写），wrapper 返回 `A_pad[:, :total_tokens]` 视图后对调用方不可见。
2. **精度路径**：源把 `a_s`/`i_s`（中间逆块）与 `gate*_s` 存 fp16；本设计全程 fp32，仅在最终 `store_fixpipe`
   量化一次 ⇒ 合法行数值更接近 fp64 语义基准（§1.6.1 A3/A5），不改变数学定义。
3. **`dim_k=128` 的 Gram 累加**：源按 2 段 64 深顺序累加；本设计一次 `T.gemm(K=128)`，fp32 累加器内部
   累加顺序由 Cube 分形决定 ⇒ 数值差异在 fp32 舍入级（L0 用例覆盖两种 `dim_k`）。

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
| `g` | `[1, total_tokens, head]` | fp32 | 按 `hid` 取 64 行 ⇒ 跨步 `head`（Vector 侧动态 extent 切片） |
| `beta` | `[1, total_tokens, head]` | fp16 / bf16 | 同上（用前 `vcast`/直接乘 fp32） |
| `eye` | `[1, 64, 64]` | fp32 | host 常量，给 `T_0 = I` 用 |
| 工作区 `gs_G` / `ws_S` / `ws_T` / `ws_P` | `[K, 2, 64, 64]` | fp32 | 每 block 私有；`K ≤ 20` ⇒ 各 640 KB（合计 2.56 MB） |
| `cu_seqlens` | `[num_sequences+1]` | int32 | 接口保留，等长分支**不读** |

### 4.2 输出张量

| 名称 | 形状 | dtype | 说明 |
|---|---|---|---|
| `A`（返回视图） | `[1, total_tokens, head, 64]` | 输入 dtype | `A_pad[:, :total_tokens]`，`A_pad` 行数 = `ceil(total_tokens/64)*64` |
| `k_rnorm` | `[1, 1, key_head]` | fp32 | `l2norm=False` 占位（源码同款，L689） |

### 4.3 中间缓冲区

Cube：`l1k / l1e / l1p / l1q / l1t`（L1）、`l0g / l0t / l0p`（L0C）；Vector：`g_ub/b_ub`（64）、
`g_row/g_col/d/e/t1/t2/n_ub/G_ub`（64×64 fp32，允许 alias 至 6 块）、`cond`（bool 64×64）、`zero_ub`。
GM：`gs_G[K,2]`（Gram 交接）、`ws_S[K,2]`（N 交接）、`ws_T[K,2]` / `ws_P[K,2]`（T/P 轮转）。

### 4.4 内存搬运路径

`GM(k) → L1(MTE2, nd2nz) → L0A/L0B → L0C(fp32) → GM(gs_G) → UB(MTE2) → Vector 组装 → GM(ws_S)
→ L1(MTE2) → L0C(链, 6 级, 每级经 GM ws_T/ws_P 往返) → store_fixpipe(f32→dtype) → GM(A)`。

### 4.5 UB / L1 / L0C / GM 预算

| 层级 | 峰值 | 上限 | 余量 |
|---|---|---|---|
| UB（Vector） | 矩阵块：`G_ub`、`g_row`、`g_col`、`d`、`e`、`t1`、`t2`、`n_ub`、`zero_ub` 共 9×16 KB；`g_t`/`g_col`/`b_col` 各 ≤ 256 B；(64,1)/(1,64) 复用小 buffer；`cond` bool 4 KB。**手工 alias 后 ≈ 5×16 KB + 4 KB ≈ 84 KB；不 alias 的最坏形态 144 KB + 4 KB ≈ 148 KB**（×1.10~1.12 ⇒ 163 KB） | 192 KB | 两种形态均 < 192 KB（§9 R13：alias 由显式 buffer 复用保证，不依赖 auto-multi-buffer） |
| L1（Cube） | `16 KB（l1k, dim_k=128）+ 4×16 KB = 80 KB` | 512 KB | 84% |
| L0C | `3×16 KB = 48 KB` | 128 KB | 62% |
| GM 工作区 | `4 × 20 × 2 × 16 KB = 2.56 MB`（`gs_G`/`ws_S`/`ws_T`/`ws_P`）+ `A_pad`（`S=32768,H=16`：512×64×16×64×2 B ≈ 64 MB） | HBM 64 GB | — |

---

## 5. 核切分与任务映射（Tiling 策略）

### 5.1 ① 逻辑核数计算

任务空间 = `(chunk, head)` 笛卡尔积：

```
chunks_per_seq = ceil(S / 64)                     # in-scope 等长序列
nch_total      = num_sequences × chunks_per_seq    # packed 行布局下的 chunk 总数
T (逻辑任务数)  = nch_total × head
```

| 形状（in-scope） | `chunks_per_seq` | `nch_total` | `head` | **T（逻辑核数）** |
|---|---|---|---|---|
| L0：`S=64, head=1` | 1 | 1 | 1 | **1** |
| `S=3000, head=16` | 47 | 47 | 16 | **752** |
| `S=4096, head=16` | 64 | 64 | 16 | **1024** |
| `S=12288, head=16` | 192 | 192 | 16 | **3072** |
| `S=32768, head=16` | 512 | 512 | 16 | **8192** |

每个任务的工作量**恒定**（一个 64×64 逆，`dim_k` 固定），故逻辑核数 = 任务数，无权重不均。

### 5.2 ② 物理核数依据

实查（2026-10-17，本机 Ascend910B3，npu-smi 25.5.0 / Chip 6）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()      # -> 20
```

记录：**返回 20**（AI Core 数）。本内核是 **MixCV**（Cube + Vector 同 kernel），瓶颈引擎是 Cube
（40 op/任务中 11 个 gemm + 29 次 L1/GM 搬运都在 Cube 侧，Vector 侧仅 12 op），因此按
**Cube/混合算子口径直接使用返回值 20**（不乘 2）；`get_aicore_num() * 2 = 40` 只适用于纯 Vector 算子
（`.agents/skills/_shared/standards/core-split-strategy.md` §1 要素②）。
（既有常数条目 `CONST-aicore-910B2C` 的 24 是 910B2C 档位；本机实查 20，**以实查为准**。）

### 5.3 ③ 规模判定与分核方案

**规模判定**：所有 in-scope 形状 `T ≥ 752 ≫ 20`（`S=3000` 已是最小真实形状），属**极大规模**；
把 `T` 摊到 20 个核上每个核仍有 38~410 个任务，无法通过调 block 尺寸把逻辑核数降到物理核数量级
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
  （`gs_G/ws_S/ws_T/ws_P` 均按 `bid` 私有）。
- **`A_pad` 写冲突**：任务边界互不重叠（每任务写自己的 64 行）⇒ 无原子/加锁需求。

### 5.4 Block 划分与 Tile Shape（非整除处理）

| 项 | 取值 | 非整除策略 |
|---|---|---|
| 任务粒度 | 1 chunk × 1 head（64 行） | 行数非整除（尾 chunk）：动态 extent 装载 + `vselect` 掩码 + `A_pad` padding（§0.6 R3） |
| Gram tile | M=64, N=64, K=`dim_k` | `dim_k ∈ {64,128}` 均为 16 的整数倍 ⇒ 无 K 尾块 |
| 链 tile | 64×64×64 | 64 = 16×4 分形对齐 ⇒ 无尾块 |
| Vector tile | 64×64（尾轴 64 = 256 B） | 无尾 lane 浪费（fp32 向量宽 8 lane，64 = 8×8） |
| 输出 tile | 64×64 → `A[0, base, hid, 0]` | 由 `A_pad` 保证 64 行落盘总在界内 |

---

## 6. 循环与调度结构

| 结构 | 选择 | 理由 |
|---|---|---|
| 任务循环 | **`T.serial(NPER)`**（`NPER` 静态），任务号 `t = bid + it*K` | 极大规模分核（§5.3）；`T.serial` 能携带跨任务的 GM 依赖（每任务独立，但 buffer 复用） |
| 链级循环 | **Python 层静态展开 `for lev in range(6)`**（不生成 TIR 循环） | 每级读写不同 slot、末级无平方；展开后无动态边界，也避开"循环携带 Cube 依赖"的历史不确定面（`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §10） |
| Gram 的 K 规约 | 单次 `T.gemm(size=[64,dim_k,64])` | `dim_k ≤ 128`，一次覆盖；不引入 `T.serial` K 段循环（源 O4 的意图由 Cube 内部累加承接） |
| `T.Parallel` | **不使用** | Vector 侧全部用 tile 级 v-prefix op（`vbrc/vsub/vexp/vmul/vselect`），由编译器映射到向量 lane；不需要手工 lane 并行 |
| `T.Pipelined` | **不使用（本版）** | 唯一可流水的是 Cube 的"装载→gemm"，而 `l1k` 单缓冲 + 每任务一次装载；CV 级流水已由 2 slot 握手提供。列入 Stage 4 备选（`num_stages=2` 双缓冲 `l1k`，预期 -0.3 µs/任务） |
| `T.Persistent` | **不使用** | 与本设计"固定 K=20 + 核内 `T.serial`"等价，无需额外原语；`T.serial` 已摊薄启动开销（20 次 block 启动/launch） |
| 跨 chunk 累积 | **无**（每 chunk 独立清零） | 与源一致；`g` 已是 kernel 1 的 chunk 内累积值 |

---

## 7. 同步策略

Expert 模式：**全部手动同步**。仅 4 个 flag id（≤15 预算，`CONST-flag-id-budget`）：

| flag | 方向 | set 侧 | wait 侧 | 语义 |
|---|---|---|---|---|
| `f_g[0]` / `f_g[1]` | Cube → Vector | `with T.rs("PIPE_FIX")`：`store l0g → gs_G[slot]` 后 | `with T.rs("PIPE_MTE2")`：读 `gs_G[slot]` 前 | Gram(slot) 就绪 |
| `f_s[0]` / `f_s[1]` | Vector → Cube | `with T.rs("PIPE_MTE3")`：`store N → ws_S[slot]` 后 | `with T.rs("PIPE_MTE2")`：读 `ws_S[slot]` 前 | `N`(slot) 就绪 |

（`slot = it % 2`；`f_g`、`f_s` 各占 2 个 id ⇒ 共 4 个。）

**无死锁 / 无竞争论证**（Cube 循环带 1 任务前瞻）：

1. Cube 前导阶段完成 `gram(t0)` 并 `set(f_g[0])`；Vector 任务 0 的 `wait(f_g[0])` 必然被满足。
2. 迭代 `it` 的 Cube 顺序是：`gram(t+K)`（写 `gs_G[(it+1)%2]`）→ `wait(f_s[it%2])` → 链 → 写 `A`。
   Vector 迭代 `it`：`wait(f_g[it%2])` → 组装 → `set(f_s[it%2])`。
3. Cube 在迭代 `it` 里重写 `gs_G[(it+1)%2]`，而该 slot 上一次被 Vector 读是在任务 `t-K`；
   Cube 在**上一迭代**已经 `wait(f_s[(it-1)%2])`，即 Vector 已完成任务 `t-K` ⇒ 不覆盖未消费数据。
4. Vector 在任务 `t+K` 重写 `ws_S[it%2]`，该 slot 上一次被 Cube 读是在迭代 `it` 的链（本迭代先完成）
   ⇒ 不发生"写覆盖正在读"。
5. flag id 按 slot 固定、每 slot 每任务恰好一次 `set`/一次 `wait` ⇒ 计数严格配对（形态同
   `examples/gdn_prefill/probe/experiments/14_two_loop_handshake.py`，含多轮复用已验证）。

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

- **交叉校验**（同一脚本内，防止 golden 自身写错）：用"分块前代"实现（16×16 对角块 + 块回代，
  即 §0.3 P6/P7 的 fp64 版本，见 `examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py`
  的 `blocked_source(..., store_dtype=torch.float64)`，实测与 `torch.linalg.solve` 差 ≤ 2.3e-8）
  独立复算一次，两者必须一致到 1e-10。
- **对照源实现**：`prefill_blocksolve_A_bthd`（GPU）在 fp16 通路下的输出作为"源通路参考"，
  仅用于"本实现不劣于源通路"的判定（源的 fp16 中间块误差 ~1e-4，见 §1.6.1 附注）。

### 8.2 精度标准

| dtype | 判据 | 依据 |
|---|---|---|
| fp16 | `max_abs_err ≤ 5e-3` 且 `max_rel_err ≤ 5e-3`（对 `|A| ≤ 4` 的元素；更大元素按相对判据） | §1.6.1 A4 rel ≤ 4.0e-6、A5 rel ≤ 9.3e-5 + fp16 eps 4.9e-4，留 ~5× 余量 |
| bf16 | `max_abs_err ≤ 4e-2` 且 `max_rel_err ≤ 4e-2` | bf16 eps 7.8e-3（8 bit 尾数）× 6 级链余量 |
| fp32 scratch（内部） | 不设门禁，仅 L0 打印 | 内部量 |

### 8.3 L0 / L1 / L2 测试计划

| 层 | 用例 | 目的 / 判据 |
|---|---|---|
| **L0-a** | `S=64, head=1, key_head=1, dim_k=64, fp16` | 单任务全矩阵：对角块=1、上三角=0、下三角逐元素比对；判据 fp16 门槛 |
| **L0-b** | 同 L0-a，`dim_k=128` | Gram 的 K 维 128 档（一次 gemm 的累加路径） |
| **L0-c** | 同 L0-a，`bf16` | `store_fixpipe` f32→bf16 路径 |
| **L0-d** | `S=100, head=1, key_head=1, dim_k=64`（2 chunk，尾 chunk 36 行） | 动态 extent 装载 + 掩码；padding 行不越界、`A_pad[:, :100]` 视图正确 |
| **L0-e** | `S=64, head=2, key_head=1, dim_k=64`（`group=2`） | `khid = hid // group` 映射（两个 head 读同一 key head） |
| **L0-f** | `S=64, beta` 含 0 行 + 大 gate 跨度（`g` 跨度 ≈ -40） | 数值边界：0 行 → 单位行；极端 gate 不产生 NaN（掩码 select 生效） |
| **L1-a** | `S=1024, head=4, key_head=2, dim_k=128, fp16`（T=64 > 20） | 多任务 / 多核跨步映射 + flag 多轮复用 |
| **L1-b** | `S=3000, head=16, key_head=1, dim_k=128, fp16`（T=752，尾 chunk 56 行） | in-scope 真实尾 chunk + 分核（38 任务/核） |
| **L2** | `S ∈ {4096,12288,32768}` × `key_head ∈ {1,16}` × `dim_k ∈ {64,128}` × `{fp16,bf16}` | 交付验证（12 组合），与 golden + 源通路双对照 |
| Boundary | `S=56`（单 chunk 且不满）、`S=64`（整除单块）、`S=65`（2 chunk，尾 1 行） | chunk 边界三态：`valid < 64` / `== 64` / `== 1` |

**L0 门槛**：L0-a~L0-f 全绿才允许提交 L1；L1-a/L1-b 全绿才允许 L2。任一失败先查 §9 的对应风险条目。

### 8.4 通过标准

L0 + L1 + L2 + Boundary 全绿 ⇒ Stage 3 出口 `[PRECISION_PASS]`；L2 任一 dtype/shape 失败 ⇒
回到 §1.6.1 复核（A4 链级数、A5 量化）或 §9 的 R2/R3 回退路径。

---

## 9. 风险点与注意事项

| # | 风险 | 影响 | 处置 / 裁决阈值 |
|---|---|---|---|
| R1 | 性能：`0.52 ms`（S=4096）是**估算**（40 op × 0.25 µs），实际链级 op 数可能更高 | 时延不达标 | Stage 3 先测 L1-b 真实时延；若 > 1.2× 估算 → 评估 §1.6.3 备选 **L2（fp16 scratch，GM 流量 -50%）**；若 > 1.5× → 评估 **L3（双 head 打包，op/head -50%，受 L0C 128 KB 限制）** |
| R2 | `initC=False` 累加只实测到 **2 级链**（本探针）与 3 步链（`10` K1）；6 级链 + 每级"store 后复用 L0C"未验证 | 数值错/编译崩 | L0-a 先跑单任务 6 级；失败回退：每级改为"显式 `gemm(·,·,l0,initC=True)` + 下一次 `T.vadd` 合并"（+5 op/任务，≈ +1.2 µs） |
| R3 | fp32（HF32）gemm 的**长链**数值行为（探针只到 2 级） | rel 超 5e-3 | L0 直接对照 fp64 golden；CPU 侧 6 级 fp32 仿真 rel ≤ 4e-6（§1.6.1 A4），余量充足 |
| R4 | `T.copy(GM→L1)` 的**动态 extent**（`valid` 行）无仓内先例（先例是 GM→UB） | 编译失败/读越界 | L0-d 覆盖；失败回退"整 64 行静态装载 + `t = min(t, T-1)` 式 clamp"或"尾 chunk 逐行拷贝" |
| R5 | Cube scope 内 `if t < T` 动态守卫 | lower 失败或分支错 | 回退：把越界任务映射到 `t = T-1`（幂等重算 + 重复写同一行，结果正确） |
| R6 | 4 个 flag id 的多轮复用语义（计数配对） | 死锁 / 读旧数据 | 形态已被 `14` 验证（2 轮）；L1-a（64 任务/20 核 ⇒ 4 轮）覆盖；失败回退到每任务轮换 flag（id = `2*slot + dir`，仍 ≤ 8） |
| R7 | 极端 gate 跨度：`exp(g_i-g_j)` 在 `i<j` 丢弃区可上溢 fp32（上限 e^88） | 产生 inf 后被 `vselect` 丢弃，不影响结果 | 记录为「未文档化假设」：GDN gate 单调非增且 chunk 内跨度 < 88（典型 ≤ 6.4）。若上游出现更大跨度，需在 kernel 1 侧加保护 |
| R8 | `A_pad` + 视图返回：`A` 的底层存储比 `total_tokens` 大（≤ 63 行） | 下游若 `clone`/`contiguous` 会产生拷贝；若断言 storage 尺寸会失败 | wrapper 文档中注明；L0-d 校验 `A.shape`、`A.is_contiguous()`、数值 |
| R9 | L0C 复用（同 buffer 跨任务/跨级 store 后复用） | 编译期约束（历史上 8.5.2 曾报错） | 9.0 已实测可用（`EXPERT_FINDINGS.md` §12.1、本探针时序核 `T.serial` 内复用）；L0 覆盖 |
| R10 | GM 工作区按 `K=20` 固定分配（1.92 MB），若 Stage 4 改 `K` 需同步改 host | shape 不匹配 | host 侧由同一常量派生（`K = min(20, T)`），不硬编码两份 |
| R11 | `num_sequences > 1`（packed 等长）虽不在 in-scope，但任务映射已按 `seq = t // (ch_per_seq*head)` 写 | 若启用需补测 | 列为 out-of-scope（算子级 §1），代码保留通用路径，L2 不覆盖 |
| R12 | 设计期常数（0.25 µs/op、72 GB/s/core）绑 910B3 + CANN 9.0.0 | 版本升级后失效 | §11 打版本戳；`kb_stale_check.py` 会在工具链变更后标记待重验 |
| R13 | `g`/`beta` 的**列切片**（`g[..., hid:hid+1]`，内层 extent=1、跨步 `head`）在 MTE2 上未实测 | 装载失败/取到错列 | L0-e（`head=2, key_head=1`）专测该路径；失败回退：整块搬 `[valid, head]`（≤ 64×16×4 B = 4 KB）后 `T.transpose` + 行切片（+1 op/任务）。同理 UB 的 alias 若不生效，最坏形态 148 KB × 1.12 ≈ 163 KB 仍 < 192 KB（§4.5） |

**技术约束检测结论**：三维 Kernel（不涉及）/ L0C 容量（48 KB < 128 KB ✓）/ GEMM 非整除（分形全对齐，
仅行数非整除 → 已给 padding+mask 方案）/ 三维 grid 与 `threads=`（不使用）/ 单 Cube+单 Vector scope（满足）。

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|---|---|---|
| 设计文档 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md` | 本文 |
| 等价性验证脚本 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py` | D-1，5 项 `EQUIV_PASS` |
| 设计期探针 1 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_doubling_cube.py` | 2 级 doubling 数值 + per-op 时延 |
| 设计期探针 2 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_probe_accum_form.py` | `initC=False` / store-then-reuse / fp32 gemm 形态二分 |
| 内核（Stage 3 产出，本阶段不写） | 算子目录下的 `prefill_blocksolve_A_bthd.py` | 由 Stage 3 落盘（本阶段只交付设计工件） |
| 复盘 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/RETROSPECTIVE.md` | Stage 1 复盘（Phase 8） |

---

## 11. 版本戳与参考

**工具链**：CANN **9.0.0**（`bishengir-compile` 1.1.0 / `hivmc` 0.2.0）/ tilelang **0.1.2** dev root
（`/home/developer/tilelang-mlir-ascend/build`）/ **Ascend910B3**（`npu-smi 25.5.0`，Chip 6，
`ASCEND_VISIBLE_DEVICES=6`）/ 2026-10-17。`NPUUtils.get().get_aicore_num() == 20`（实查）。

**模式**：`TILELANG_ASCEND_MODE=Expert`（显式赋值）；`Kernel.ascend_mode = "Expert"` +
`scoped_ascend_mode`（`examples/TileOPs/tileops/utils/ascend_mode.py`，CG-2026-0010）。

**实测常数（本任务新增，待蒸馏入 pattern-library）**：

| 常数 | 值 | 来源 |
|---|---|---|
| 64×64 fp32 的 gemm/store/load 平均 op 代价 | **0.25 µs/op**（6 op 一级 = 1.52 µs，`reps=2` vs `202` 差分） | `design_probe_doubling_cube.py` PROBE-TIMING |
| Expert 下 fp32(HF32) gemm 64³ 精度 | err 2.9e-8（|I+N| ≈ 1）；2 级链 err 2.7e-7 | 同上 PROBE-CORRECT |
| `initC=False` 累加进"已 store 过"的 L0C | 正确（err 2.9e-8） | `design_probe_accum_form.py` V3 |
| 回喂经 `out_idx` 张量 | 编译器 SIGSEGV（9.0 上比"数值错"更硬） | 本探针第一版（已修） |
| fp16 scratch + fp32 L0C 混合 store/回读 | 数值错（err 1.875） | `design_probe_accum_form.py` V4 |

**参考**（按 Docs Auto Routing 优先级）：

1. `docs/Tilelang.language/线性代数操作/T.gemm.md`（fp32 HF32 路径 §2.2.1、`initC` 语义 §2.1、累加限制 §2.3）
2. `docs/Tilelang.language/内存操作/T.alloc_L1.md`、`T.alloc_L0C.md`、`T.store_fixpipe.md`（f32→f16/bf16 合法配对）、`T.copy.md`
3. `docs/Tilelang.language/shape操作/T.vbrc.md`、`docs/Tilelang.language/数学操作/T.vexp.md`、`T.vsub.md`、`T.vmul.md`、
   `docs/Tilelang.language/条件操作/T.vselect.md`、`docs/Tilelang.language/比较操作/T.vcmp.md`、`docs/Tilelang.language/创建操作/T.arange.md`
4. `docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md`
5. `docs/开发指南.md` §3.3（物理核数与分核策略）、`.agents/skills/_shared/standards/core-split-strategy.md`
6. `examples/gdn_prefill/DESIGN.md`（算子级 rev.2）、`examples/gdn_prefill/probe/EXPERT_FINDINGS.md` §12
7. `examples/gdn_prefill/probe/experiments/10_feedback_and_acc_cann9.py`、`13_l0c_fp16_scratch_cann9.py`、`14_two_loop_handshake.py`
8. `examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/gdn_prefill.py`（同流水线 1/6 的命名与测试分层）、
   `examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill/DESIGN.md`（1/6 设计）
9. 源算子：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`（L158–L712）
