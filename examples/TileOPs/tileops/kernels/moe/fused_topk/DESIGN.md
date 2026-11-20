# _fused_topk_kernel 算子设计文档

> 迁移任务（harness，**Developer**）。源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`
> 的 `FusedTopKKernel` 工厂 `_fused_topk_kernel`（GPU/CUDA target）；op 层
> `/tmp/opencode/TileOPs-ref/tileops/ops/moe/fused_topk.py` 的 `FusedTopKOp`。本单元目标：`target="npuir"`（Ascend NPU）。
> Stage 0 抽取件：`examples/TileOPs/tileops/kernels/moe/fused_topk/_fused_top_k_kernels.py`；NPU 侧已移植脚手架
> `examples/TileOPs/tileops/kernels/moe/fused_topk/fused_topk.py` + `examples/TileOPs/tileops/ops/moe/fused_topk.py`。
> 迁移决策（§0）驱动 §1–§10；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
> 等价性机器验证（D-1）：`examples/fused_topk/_fused_topk_kernel/verify_equiv.py`（136 例全 EQUIV_PASS，
> max_rel_fp32=1.075e-06、违反率 0）。
> 可行性机器验证（D-3 设计期探针，设备实测 2026-10-14，Ascend910_9362）：
> `examples/fused_topk/_fused_topk_kernel/history_version/design_probe_vsort_topk.py`（P1：`T.vsort` 多 block
> 降序 top-k 位精确）、`examples/fused_topk/_fused_topk_kernel/history_version/design_probe_fused_scoring.py`
> （P2：softmax/sigmoid × renorm 端到端 vs torch golden，误差 ≤1.8e-7）。

## 修订说明（revision v1，相对 design_v0）

> mode=revision，revision_index=1；上一版备份 `examples/fused_topk/_fused_topk_kernel/history_version/design_v0.md`；
> 触发来源：Stage 2 检视「不通过」（`examples/fused_topk/_fused_topk_kernel/REVIEW.md`，2 阻塞 + 3 建议）。
> 本次仅修订仍成立结论之外的缺陷项，**未静默丢弃**任何 v0 有效结论。

**相对上一版的关键调整**：

- **阻塞-1（R3 口径名）**：§1.6.0-R3 复杂度表首列由「计算量（逐行）」改为 **「FLOPs（计算量/比较次数，逐行）」**，
  并在口径脚注 ① 显式命名「FLOPs（浮点/比较运算计数）」；使 `design_calc_check.py` 的 `r3_metrics` 判 `pass`。
  仅改口径名，四口径实质（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）与基线 C0 参与对比均不变。
- **阻塞-2（规约类必选候选缺失）**：§1.6.3 新增「**规约类 lane 映射正交决策：水平归约 vs 垂直扫描**」表 +
  候选矩阵行 **#5（垂直扫描，lane→T 行）**，给出量化评分（`T.vsort` 尾轴兼容性 / 整除性与尾 lane 浪费 /
  累加链形态 / repack / UB / 发射经济性）与**结构否决**理由；并同步 §3.3 伪代码说明与 §6.1 循环结构说明
  （水平归约 `T.reduce_*(dim=1)`，无垂直扫描逐行串行循环）。
- **建议-3（softmax padding 机制）**：§0.1 更正为——softmax 路径填充位经 `exp(-inf−m)=0` 成为行内最小值、
  sigmoid 路径填充位保持 `-inf` 且 `σ(-inf)=0`（原「sigmoid(-inf)≈0」系误植；结论「不入选 / 免 padding 等价」不变）。
- **建议-4（转录笔误）**：§1.6.0 信息源引用 `ALG-topk` 命中行时把 `R3：O(N·k) vs O(N log N)` 由误写的 `R5` 更正为 `R3`。
- **建议-5（tie-break 依据）**：§0.1 把「`torch.topk` 同取最小下标」改标注为「**未文档化假设**」（PyTorch 未承诺
  ties 时取最小下标），保留「tie-break 差异在参考测试口径内可接受」的结论（与 §8.1 校验口径、§9 R-2 一致）。

**为何不会再犯同一错误**：

- **口径名**：后续任何 §1.6.0-R3 表首列一律以 canonical `FLOPs`（含「浮点/比较运算计数」括注）命名，
  杜绝机械 `r3_metrics` 假阴性。
- **规约类候选**：凡算子含行/列归约子步骤（含由库原语 `T.reduce_*` 承担），§1.6.3 必列「水平归约 vs 垂直扫描」
  正交候选行（本算子已被 `T.vsort` 尾轴整行驻留约束锁定为水平归约）。
- **转录**：引用参考表命中行时逐字对齐，R 编号不得跨行错位。
- **事实断言**：非文档保证的行为（如 `torch.topk` ties 下标归属）一律标注「未文档化假设 / 待确认」并列入 §9 风险点。

**保持不变（未静默丢弃的 v0 既有结论）**：E1 单调保序选择（原始 logits 上选 top-k）、E2 softmax+renorm 全行分母抵消、
`T.vsort` 降序取前 K 方案、D-3 实机探针证据（P1/P2）、D-1 等价性机器验证证据（136 例 ALL EQUIV_PASS）、
分核三要素（40 核 persistent + 核内静态 `T.serial`）、R5 renorm 融合与全行分母抵消。

**本单元的一句话定位**：`_fused_topk_kernel` 是 MoE 的 **top-k 路由单元**——对每 token 的门控 logits
先做打分（softmax 或 sigmoid），再选出得分最高的 `top_k` 个专家并（可选）把选中权重归一化到和为 1；
输出 `topk_weights [T,K]`（fp32）与 `topk_ids [T,K]`（int32）。它是一个**逐行选择 + 逐行打分**的纯 Vector 算子。

**设计核心 / GPU→NPU 关键重设计（可提炼为 4 条）**：

1. **选择改用硬件排序原语 `T.vsort`**：源 GPU 用「one warp per token + K 轮 masked argmax + `shfl_xor` 跨 lane 归约」
   选 top-k。NPU 无 warp shuffle、无 thread 级并行，改为**每行块 `(BM,E)` 一次性 `T.vsort`（降序）**取前 K
   （值 + int32 原始下标）——1 条 `hivm::VSortOp` 替代 K 轮 argmax 归约，且直接产出下标。
2. **选择在原始 logits 上做（单调等价）**：`softmax`/`sigmoid` 都是严格单调增函数，故 top-k 集合与在原始
   logits 上选 top-k **逐位一致**（§1.6.1-E1 机器验证）——于是**只对选中的 K 个元素打分**，省去全行打分物化
   （softmax+renorm 时全行分母完全抵消、连全行 exp/sum 都不必算）。
3. **打分 / 归一化全向量化**：`T.reduce_max`（行 max 稳定化）+ `T.vsub`/`T.vexp`/`T.vsigmoid`/`T.reduce_sum`/
   `T.vdiv`（行广播），renormalize 融合进同一 kernel 的 K 元素切片上（无第二次 host 侧 `torch.topk`/除法）。
4. **分核：一维 persistent，固定物理核数 + 核内串行**：`num_kernels = min(num_logical, 40)`（纯 Vector 物理核
   = 实查 aicore 20 × 2 = 40），核内 `T.serial(num_local_tasks)` grid-stride（静态界）+ 动态 `if task < num_logical`
   守卫；`real_m = T.min(BM, T − off)` 行尾块 + 双 slice `T.copy` 处理非整除 T。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = `FusedTopKKernel` 的 `_fused_topk_kernel` 工厂（含 `with_correction_bias=True/False` 两分支）。
> §0.5/§0.6 的判定输入含 Phase R 调研结论（§1.6.0）。

### 0.1 源算子语义（做什么）

**数学语义**：对二维门控 logits `x ∈ (T, E)` 逐行先打分再选 top-k：

$$
s_{te}=\begin{cases}
\dfrac{e^{\,x_{te}-m_t}}{\sum_{e'}e^{\,x_{te'}-m_t}},\quad m_t=\max_{e'}x_{te'}, & \text{scoring\_func="softmax"}\\[2mm]
\dfrac{1}{1+e^{-x_{te}}}, & \text{scoring\_func="sigmoid"}
\end{cases}
$$

选出每行得分最大的 K 个专家下标 $\text{ids}_t = \operatorname{topK}_e s_{te}$（按原始专家号），
输出权重 $w_{t,k} = s_{t,\text{ids}_{t,k}}$；若 `renormalize=True` 再令 $w_{t,k} \leftarrow w_{t,k}/\sum_{k'}w_{t,k'}$。

- **选择语义 / tie-break**：源在等值时取**最小专家号**（lane-local argmax 用严格 `>`、跨 lane 归约在
  `other_val == l_best_val` 时取 `other_idx < l_best_idx`）；`torch.topk` 在 ties 时的下标归属**未文档化保证**
  （PyTorch 未承诺 tie 时取最小下标——本项标注为「未文档化假设」，与 §9 R-2 一致，非事实断言）。NPU 侧 `T.vsort`
  在等值时的下标归属可能与二者不同（文档 `docs/Tilelang.language/排序操作/T.vsort.md` §2.3-4 明示
  「排序稳定性差异，验证按『值匹配 + 下标指向的值匹配』双重校验」）——**参考测试 `test_moe_fused_topk.py` 的 `_check`
  正是这样校验的**（只比较权重、校验无重复 id 且每个选中得分 ≥ 第 K 大参考得分），故 tie-break 差异在容差内可接受。
- **规约语义**：softmax 分母是对行内全部 E 个元素的求和，源由 warp `shfl_xor` 树形归约（累加顺序 = 二进制树）。
  NPU 侧由 `T.reduce_sum` 承接，**累加分组不同** → fp32 ulp 量级差异（§1.6.1 机器验证违反率 0）。
- **dtype 语义**：源全程 **fp32 计算域**——op 层 `gating_output.to(torch.float32)` 上送（任意浮点输入先升 fp32），
  kernel 内 logits/打分/权重全 fp32；输出 `topk_weights` fp32、`topk_ids` int32。softmax 减行 max 稳定化
  （sigmoid 无 max）。
- **边界语义**：
  - **E 非 32 整除**：源每 lane 持 `ELEMS_PER_THREAD = ceil(E/32)` 个专家，`expert_idx >= E` 的填充位初始化为
    `-inf`。**softmax 路径**：填充位经 `exp(-inf−m)=0` 成为行内最小值（源 L255），不入选；**sigmoid 路径**：
    填充位保持 `-inf`，`σ(-inf)=0`（源 L268 注释），选择时不入选。
    NPU 侧**不 padding**（排序只作用于真实 E 列，等价）。
  - **T 尾块**：源 `if token_id < num_tokens` 守卫，越界 warp 不工作。NPU 侧由 `real_m = min(BM, T-off)` + slice 处理。
  - **K=1 / K=E**：源 K 轮 argmax 天然覆盖；NPU 侧 vsort 取前 K 天然覆盖。
  - **全相等值**：源按最小下标选 K 个不同专家；NPU 侧排序给出 K 个不同下标（顺序可不同，值相同）。
  - **±inf/NaN**：源对填充位用 `-inf`；真实 logits 若含 `-inf`，softmax exp(-inf-max)=0、argmax 不选它。NPU 侧
    vsort 将 `-inf` 排到末尾（降序），行为一致；NaN 不属本算子输入域（列为 §9 风险）。
- **correction_bias 变体语义**（`with_correction_bias=True`，仅 `scoring_func="sigmoid"`）：选择依据为
  `my_biased = sigmoid(logit) + correction_bias[e]`，而**输出权重仍是未加偏置的 sigmoid 得分**。
  ——本 harness 的 `FusedTopKOp` 签名 inputs 只含 `gating_output`、workload 不生成 `correction_bias`，
  参考测试也**不做 correction_bias 数值用例**（仅测构造/设备校验）。故本单元首版设计**标准变体**，
  correction_bias 变体列为 §9 风险（并给出可行落地路径：排序 biased 值 → `T.gather` 取回 unbiased sigmoid）。
- **host 侧语义**：op 层做 validate + `_get_kernel(T,E,K)`（`lru_cache` 按 shape 特化）+ `logits.to(fp32)`；
  源 kernel `forward` 在 `renormalize=True` 时于 **host 侧 PyTorch** 做 `w / w.sum(-1,keepdim=True)`（一次额外算子）。
  `movedim`/`reshape` 等不涉及本算子。**renormalize 与 fp32 上送均视为实现手段**（可 kernel 内化）。

**语义保持基线**：§8 的 golden 以本节语义为唯一依据（`examples/TileOPs/tests/ops/test_moe_fused_topk.py` 的 `fused_topk_torch`
即 `torch.softmax/sigmoid → torch.topk → 可选 renormalize`），**不得复刻** §0.6 的 NPU 算法（vsort + 原始 logits 选择）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `gating_output` | `(T, E)` | float16 / bfloat16 / float32 | 门控 logits（2D，行主序连续）|
| `correction_bias` | `(E,)` | float32 | 可选（`with_correction_bias=True` 时必需）|
| `topk_weights` | `(T, K)` | float32 | **输出**：选中专家的权重（可选归一化）|
| `topk_ids` | `(T, K)` | int32 | **输出**：选中专家的原始下标 |

工厂契约：`_fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func, with_correction_bias)`
（`functools.lru_cache(maxsize=64)` 特化）→ 内层 `@tilelang.jit(out_idx=[])` 的 `_func(TOKENS_PER_BLOCK)`
→ `main(gating_output, [correction_bias], topk_weights, topk_ids)`。源 `TOKENS_PER_BLOCK=16`
（`T.Kernel(ceil(T/16), threads=16*32=512)`，每 warp 一 token）。

### 0.3 实现算法解读（怎么算）

**源 kernel 结构**（源 L70–L313）：`@tilelang.jit(out_idx=[])` 工厂，`T.Kernel(ceil(T/TPB), threads=TPB*32)`
一维 grid；`tx = get_thread_binding()`，`warp_id = tx//32`，`lane_id = tx%32`，`token_id = block_id*TPB + warp_id`
——**每个 warp 独立处理一个 token**。关键中间量：`ELEMS_PER_THREAD = ceil(E/32)`（每 lane 持有的专家数）。

**计算步骤分解**（源码每条计算语句均归入下表；标准变体为例，correction_bias 变体差异见步骤 2'/3'）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | 载入：`for j in serial(ELEMS): expert_idx=j*32+lane_id; my_scores[j]= x[tid,expert_idx] if expert_idx<E else -inf` | x | `my_scores[ELEMS]`（寄存器） | 读入 logits（+ 非整除填充 `-inf`）|
| 2a | softmax 行 max：lane 局部 `l_max=max_j my_scores[j]`；`l_max = max(l_max, shfl_xor(l_max, mask))`（5 轮） | my_scores | `l_max`（warp 全体） | $m_t=\max_e x_{te}$ |
| 2b | softmax 行 exp+sum：`val=exp(my_scores[j]-l_max); my_scores[j]=val; l_sum+=val`；`l_sum += shfl_xor(l_sum,mask)`（5 轮）；`inv_row_sum=1/l_sum` | my_scores | `my_scores`（=softmax 分子）、`inv_row_sum` | $s_{te}=e^{x_{te}-m_t}/\sum$ |
| 2c | sigmoid：`my_scores[j]=1/(1+exp(-my_scores[j]))`（逐元素，无归约） | my_scores | `my_scores`（=sigmoid 得分） | $s_{te}=\sigma(x_{te})$ |
| 3a | K 轮 argmax（每轮 k）：lane 局部 `l_best_val=-inf; for j: if my_scores[j]>l_best_val: 更新 val/idx` | my_scores | `l_best_val/l_best_idx`（lane 局部） | 行内第 k 大得分 |
| 3b | warp 归约：5 轮 `other_val=shfl_xor(l_best_val,mask); other_idx=shfl_xor(l_best_idx,mask);` 更新 `l_best_idx`（`other_val>l_best_val` 或等值取小下标）与 `l_best_val=max(...)` | lane 局部 best | warp 全体 best | 行内 argmax（tie-break 最小下标）|
| 3c | `if lane_id==0: topk_weights[tid,k]=l_best_val*inv_row_sum; topk_ids[tid,k]=l_best_idx` | best | 2 张输出表 | $w_{t,k}=s_{t,\text{ids}_{t,k}}$ |
| 3d | 掩码：`for j: if j*32+lane_id==l_best_idx: my_scores[j]=-inf`（所有 lane 各自掩码，无需同步） | l_best_idx | my_scores | 第 k+1 轮排除已选专家 |
| 4 | host 侧（op 层/kernel forward）：`if renormalize: topk_weights /= topk_weights.sum(-1,keepdim=True)` | 输出 | 归一化输出 | renormalize |

**correction_bias 变体差异**（源 L96–L207）：步骤 2c 额外算 `my_biased[j]=sigmoid+bias`；步骤 3 的
lane-local argmax/warp 归约**以 `my_biased` 比较**，同时把未加偏置的 `my_scores` 值随
`l_best_orig` 经 `shfl_xor` 一起搬运，`lane 0` 写出的是 `l_best_orig`（unbiased sigmoid）；掩码同时清
`my_scores` 与 `my_biased`。

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[gating_output(T,E)] --每 lane 标量循环读 ELEMS 个--> REG[my_scores(ELEMS) fp32]  (越界填 -inf)
                        │
   softmax: shfl_xor 树形 max → l_max；REG in-place exp(x-max) + shfl_xor 树形 sum → inv_row_sum
   sigmoid: REG in-place 逐元素 σ(x)
                        ▼
   K 轮: REG lane-local argmax → shfl_xor (val,idx) 树形归约 → warp best → lane0 写 GM
                                   └─ REG 掩码已选为 -inf（所有 lane 各自）
                        ▼
   GM[topk_weights(T,K) fp32] (×inv_row_sum)、GM[topk_ids(T,K) int32]
   （renormalize 在 host 侧对 topk_weights 再做一次逐行除法）
```

**循环与并行结构**：grid = `ceil(T/TPB)` 个 block（TPB=16）＝ `ceil(T/16)` 个 block；每 block 512 线程
＝ 16 warp；**warp 内 32 lane 并行**，lane 内 `T.serial(ELEMS)` 顺序扫描；K 轮 `T.serial(top_k)`。
无 `T.Parallel` / `T.Pipelined`。
**冗余特征**：每个 warp 独立、block 内 16 warp 无共享，block 数仅由 T 决定。

**host 侧逻辑**：
1. op 层 `FusedTopKOp.forward`：validate（shape/dtype/top_k≤E/correction_bias 规则）→ `_get_kernel(T,E,K,dev)`
   （`lru_cache` 按 shape 特化）→ `kernel(gating_output, correction_bias)`。
2. `FusedTopKKernel.forward`：`logits_f32 = gating_output.to(torch.float32)`（**host 侧 dtype 上送**）→
   分配两张输出 → `fn(logits_f32, [bias_f32], topk_weights, topk_ids)` → `if renormalize: w /= w.sum(...)`。

**编译配置**：`@tilelang.jit(out_idx=[])`（无 `pass_configs`、无 `compile_flags`）。`supported_archs=[80,86,89,90]`
（H100/A100/…），NPU 侧脚手架已改 `None`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **one warp per token（thread 级任务划分）** | 每 token 独立、无跨 warp 通信 | 每 warp 32 lane 认领一个 token | CUDA warp / thread 模型 | 硬件强相关（NPU 无 warp/thread-per-token）→ 重设计 |
| O2 | **寄存器驻留全部专家**（`T.alloc_local`） | 免 SMEM、免重复访存 | 每 lane 持 `ceil(E/32)` 个 logits | 大寄存器堆 | 硬件强相关（NPU 用 UB 驻留）→ 等价替换 |
| O3 | **warp `shfl_xor` 树形归约**（softmax max/sum、argmax） | **零 `__syncthreads`**（源文档明示 0 barrier，vs 旧设计 22 sync） | lane 间 5 轮 xor-shuffle 全归约 | warp shuffle 指令 | 硬件强相关（NPU 无 shuffle）→ 重设计 |
| O4 | **K 轮 masked argmax 复用寄存器** | 免排序、O(K·E) 比较 | 每轮选最大后把该专家掩码为 `-inf`，寄存器原址复用 | 寄存器 + 掩码 | 算法层可移植，但**索引恢复**依赖 shuffle → 重设计 |
| O5 | **fp32 计算域 + softmax 减行 max 稳定化** | 数值稳定、与 torch 一致 | 全链 fp32；`exp(x-m)` | 无（纯数值） | 可移植（保留） |
| O6 | **非整除 E 的 `-inf` 填充** | 32 lane 对齐，填充位不入选 | 越界专家置 `-inf` | lane 宽度 32 | 可移植（NPU 免 padding，见 O9） |
| O7 | **`lru_cache` 工厂特化** | 消重复编译 | `(T,E,K,scoring_func,bias)` 键 | 无 | 可移植（保留） |
| O8 | **输出为独立 GM 张量（`out_idx`/显式分配）** | 输出直达 GM，无中间物化 | 两输出由 op/kernel 分配 | 无 | 可移植（保留） |
| O9 | **renormalize 在 host 侧 PyTorch** | 实现简单 | kernel 出未归一权重后再逐行除 | 无 | 可移植但引入额外算子 → NPU 融合进 kernel（重设计） |

> 识别完整性：以上覆盖 `_fused_topk_kernel` 两分支的全部优化性结构（步骤表已把每条计算语句归入步骤 1–4）。
> 无 SMEM tiling、无 mma/Cube、无异步流水、无 swizzle、无 persistent（源为超发 grid）。**识别不出来 ≠ 不存在**，
> O1–O9 逐项在 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

> 判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？判定输入含 Phase R 调研结论（§1.6.0：
> 选定「原始 logits 上 `T.vsort` 降序取前 K + 仅对选中 K 打分」，源 warp K 轮 argmax 被替换）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（softmax/sigmoid 打分、top-k 选择、可选归一化、fp32 域、tie-break 容差内） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 以此为基线（§8） |
| O1 one warp per token（thread 级任务划分） | 优化 | CUDA warp/thread | **无**（本项目一维 Kernel，无 thread/warp） | **重新设计** | §0.6-R1：一维 persistent，token 行块 `(BM,E)` 分核 + 核内 `T.serial` grid-stride |
| O2 寄存器驻留专家 | 优化 | 大寄存器堆 / `alloc_local` | 有（UB 驻留，但 `alloc_local + T.serial` 在本工具链 SIGSEGV，CG-2026-0028b） | **等价替换** | §0.6-R2：`T.alloc_shared`（UB）驻留 `(BM,E)` fp32；不用 `alloc_local` |
| O3 warp `shfl_xor` 树形归约 | 优化 | warp shuffle | **无**（NPU 无 shuffle；映射表 `migration-analysis.md` §5.3 判「重新设计」） | **重新设计** | §0.6-R2：行 max/sum 改 `T.reduce_max`/`T.reduce_sum`；softmax 分母聚合顺序改变（§1.6.1-E4 机器验证） |
| O4 K 轮 masked argmax + 掩码复用 | 算法/优化 | 无（算法可移植），但**跨 lane (val,idx) 归约**依赖 shuffle | 算法等价，实现不可用 | **重新设计** | §0.6-R3：改 `T.vsort` 降序取前 K（一条 `hivm::VSortOp` 直接产出值 + int32 下标）；承接 O4「免排序」意图的**性能面**改为「一次硬件排序替代 K 轮归约」 |
| O5 fp32 计算域 + softmax 减 max | 算法/优化 | 无 | 有（Vector fp32） | **保留** | 全链 fp32；`T.vsort` 仅 fp16/fp32（bf16 须升 fp32，§1.6.1-E5）；减 max 由 `T.reduce_max` + `T.vsub` 承接 |
| O6 非整除 E 的 `-inf` 填充 | 优化 | lane 宽度 32 | **无此需求**（排序作用于真实 E 列） | **等价替换** | §0.6-R4：直接对原始 E 列 `T.vsort`，无 padding、无 `-inf` 哨兵（等价，§1.6.1-E3 机器验证含非整除 E） |
| O7 `lru_cache` 工厂特化 | 优化 | 无 | — | **保留** | 工厂层与硬件无关（脚手架 `_get_kernel` 已保留） |
| O8 输出独立 GM 张量 | 优化 | 无 | — | **保留** | kernel 写 `topk_weights`/`topk_ids` 两张 GM 表（`out_idx=[1,2]` 或显式分配） |
| O9 host 侧 renormalize（额外 PyTorch 算子） | 优化 | 无 | 有（GPU 也可融合，源未做） | **重新设计** | §0.6-R5：renormalize 融合进 kernel 的 K 元素切片（`T.reduce_sum` + `T.vdiv`），消除第二次 host 算子与 GM 往返 |

> **四态处置统计**：保留 4 项（语义 / O5 fp32+减 max / O7 lru_cache / O8 输出表）｜等价替换 2 项（O2 寄存器→UB、
> O6 填充消除）｜重新设计 4 项（O1 并行结构〔R1〕、O3 shuffle 归约〔R2〕、O4 K 轮 argmax〔R3〕、O9 renorm 融合〔R5〕）｜
> 舍弃 0 项。**无「舍弃后性能意图丢失」项**：O1/O3/O4 的低延迟意图由 R1/R2/R3 承接，O9 的归一化语义由 R5 融合承接。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（对照 Phase R 调研结论 §1.6.0：
调研在「可实现且亲和」的算法族内择优——选定 **C1：原始 logits 上 `T.vsort` 降序取前 K + 仅对选中 K 打分**；
源 warp K 轮 argmax 因 NPU 无 shuffle 且索引恢复无干净原语被替换；源优化手段 O1–O9 的意图承接见下）。

**重设计项 R1：并行结构（源：one warp per token + 超发 grid）→ 一维 persistent 行块分核**

- **源方案**：`T.Kernel(ceil(T/16), threads=512)`；每 warp 认领一个 token，lane 持 `ceil(E/32)` 个专家。
- **NPU 新算法**：一维 persistent——`T.Kernel(num_kernels, is_npu=True) as (cid,_)`，
  `num_kernels = min(num_logical, 40)`（纯 Vector 物理核数，§5.5），`num_local_tasks = ceil(num_logical/40)`（静态），
  核内 `for s in T.serial(num_local_tasks): task = s*num_kernels + cid; if task < num_logical:` 处理一个
  `(BM,E)` 行块（`real_m = T.min(BM, T - task*BM)`，双 slice `T.copy`）。
- **语义保持论证**：token 行之间**完全独立**（逐行打分 + 逐行选择，无跨行依赖），行块划分只改变调度、
  不改变任何 token 的结果；尾块 `real_m` 只裁掉不存在的行。**脚本先例**：`examples/cumprod/_cumulative_kernel`
  的 persistent + 静态 `T.serial` + 动态 guard 结构（已实现并验证）。

**重设计项 R2：warp shuffle 归约（softmax max/sum）→ Vector 行归约**

- **源方案**：lane 局部归约 + 5 轮 `shfl_xor` 树形全归约得 `l_max`/`l_sum`。
- **NPU 新算法**：`T.reduce_max(xf, mr, dim=1, clear=True)`（行 max）与 `T.reduce_sum(d, sr, dim=1, clear=True)`
  （行 exp 和）；行广播减/除用 `T.vsub`/`T.vdiv`（`[BM,E]±[BM,1]` 行广播，文档 `docs/Tilelang.language/数学操作/T.vsub.md`
  §2.2.2、`T.vdiv.md` §2.2.2）。softmax 稳定化与源一致（减行 max）。
- **语义保持论证**：归约结果数学等价，**累加分组不同**（树形 shuffle vs Vector 归约）→ fp32 ulp 量级差异；
  §1.6.1 机器验证（136 例，含 large_magnitude×50、pos_neg_mix×200）违反率 0、max_rel_fp32=1.075e-06。边界：
  max/sum 对全行（含 `-inf` 项）正确；`-inf` 项 `exp(-inf-m)=0` 不污染和。

**重设计项 R3：K 轮 masked argmax → `T.vsort` 降序取前 K**

- **源方案**：K 次「lane-local argmax + 5 轮 (val,idx) `shfl_xor` 归约」，每轮把选中专家掩码为 `-inf`。
- **NPU 新算法**：对本行 `(BM,E)` 原始 logits 一次 `T.vsort(xf, sv, si, descending=True, sort_axis=-1)`
  （`hivm::VSortOp`，`docs/Tilelang.language/排序操作/T.vsort.md`），`sv`=降序值、`si`=对应 int32 原始下标；
  取前 K 列 `sv[:,0:K]`/`si[:,0:K]`。**一条原语替代 K 轮归约**，且直接产出专家下标（源需跨 lane 搬运下标）。
- **语义保持论证**：**排序与选择数学等价**——选出的是行内得分（=原始 logits 的单调变换）最大的 K 个；因
  选择改在原始 logits 上做（§1.6.1-E1），需论证 top-k 集合不变：`softmax`（`exp` 严格单调）与 `sigmoid`
  均为严格单调增函数，故集合**逐位一致**（机器验证选择集失配 0/136；含 ties 用例）。tie-break 差异在
  §0.1 已论证为容差内可接受（参考测试口径）。
  **意图承接**：源 O4「免排序、O(K·E) 比较」的**性能意图**由「一次硬件排序原语」承接——在本工具链上
  K 轮 argmax 需要 argmax 索引恢复（`T.reduce_max` 无下标输出，见 §1.6.0-R4），代价与不可靠性更高。

**重设计项 R4：非整除 E 的 `-inf` 填充消除**

- **源方案**：每 lane 持 `ceil(E/32)` 个专家，越界位填 `-inf`。
- **NPU 新算法**：排序与打分只作用于**真实 E 列**（`(BM,E)` 满行驻留 UB），无需 padding、无需 `-inf` 哨兵。
- **语义保持论证**：padding 的 `-inf` 在源中不改变任何输出（`exp(-inf-m)=0`、argmax 不选 `-inf`），
  故消除 padding 等价（§1.6.1-E3 机器验证含 E=8/33/100 等非整除用例）。

**重设计项 R5：renormalize 内移 kernel（+ 全行分母可取消）**

- **源方案**：kernel 出未归一权重，host 侧 `w / w.sum(-1,keepdim=True)`（第二算子）。
- **NPU 新算法**：融合进同一 kernel——取 `w_k =`（softmax：`exp(x_k-m)`；sigmoid：`σ(x_k)`）后
  `T.reduce_sum(w, sr, dim=1)` + `T.vdiv(w, sr, w)`（行广播）得归一权重。
  且 **softmax+renorm 时全行 softmax 分母 $\sum_{all}e^{x-m}$ 完全抵消**（分子分母同除）→ 不必计算全行 exp/sum。
- **语义保持论证**：`s_k/Σ_K s_k = (e^{x_k-m}/S)/(Σ_K e^{x_j-m}/S)`，数学恒等（S = 全行分母）；
  机器验证 renorm 用例违反率 0。承接 O9 的归一化语义并消除第二次 host 算子与 GM 往返。

**语义保持 / 边界核对汇总**：数学等价项（E1 选择、E3 padding 消除、E5 renorm）与容差内等价项
（E2 归约分组、E4 sigmoid→exp 链）均在 §1.6.1 给出四要素论证 + `verify_equiv.py` 结果；边界（T 尾块、
K=1/K=E、全相等、非整除 E、±inf）在 §0.1 与 §1.6.1 逐项核对。

### 0.7 标杆实现

- **源算子**：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`（`_fused_topk_kernel` / `FusedTopKKernel`，
  §0.1–§0.4 解读对象）；op 层 `/tmp/opencode/TileOPs-ref/tileops/ops/moe/fused_topk.py`（`FusedTopKOp`）。
- **NPU 侧 harness 工件**：Stage 0 抽取件 `examples/TileOPs/tileops/kernels/moe/fused_topk/_fused_top_k_kernels.py`；
  已移植脚手架 `examples/TileOPs/tileops/kernels/moe/fused_topk/fused_topk.py`（`FusedTopKKernel`，K5/K8 完成）、
  `examples/TileOPs/tileops/ops/moe/fused_topk.py`（`FusedTopKOp`，O1–O6 完成）。
- **规格 / 测试 / bench**：`examples/TileOPs/tileops/manifest/moe.yaml`（`FusedTopKOp` 段）、
  `examples/TileOPs/tests/ops/test_moe_fused_topk.py`（`fused_topk_torch` 参考实现 + `_check` 校验口径）、
  `examples/TileOPs/benchmarks/ops/bench_moe_fused_topk.py`、`examples/TileOPs/tileops/workloads/moe.py`（`FusedTopKWorkload`）。
- **同族已验证参考**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`
  （Developer 行归约 + `T.reduce_max`/`T.vsub` 广播/`T.vexp` + slice `T.copy` 的实证形态）；
  `examples/cumprod/_cumulative_kernel/DESIGN.md`（persistent 分核 `/ T.cumsum` 原语 / UB 预算 / D-3 探针方法论）；
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（同族整数/索引类；工具链陷阱规避清单）；
  `testing/npuir/sort_ops/test_vsort_dev.py` + `testing/npuir/sort_ops/test_vsort_exp.py`（`T.vsort` Developer/Expert 通过用例）。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的 NPU 算法）。

---

## 1. 概述

### 1.1 算子名称

`_fused_topk_kernel`（承载 `FusedTopKOp` 的 NPU MoE top-k 路由 kernel；`target="npuir"`，Developer 模式）。

### 1.2 功能描述

对 `(T,E)` 门控 logits 逐行打分（softmax 或 sigmoid）并选取得分最高的 K 个专家，输出权重 `topk_weights [T,K]`
（fp32）与下标 `topk_ids [T,K]`（int32）；`renormalize=True` 时把选中权重归一化到和为 1。
**迁移决策后的 NPU 形态**（§0.6）：一维 persistent Vector kernel，按 token 行块 `(BM,E)` 分核驻留 UB，
用 **`T.vsort` 降序**在**原始 logits** 上选前 K，仅对选中 K 个元素做打分/归一化（全向量），
非整除 T 由行尾块 slice 处理、非整除 E 无需 padding。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
s_{te}=\text{softmax}_e(x_{t\bullet})\ \mathrm{or}\ \sigma(x_{te}),\qquad
\text{ids}_t=\operatorname{topK}_{e}s_{te},\qquad w_{t,k}=s_{t,\text{ids}_{t,k}},\qquad
w_{t,k}\leftarrow w_{t,k}/\textstyle\sum_{k'}w_{t,k'}\ (\text{renormalize})
$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解；依赖 softmax/sigmoid 严格单调）：

$$
\text{ids}_t=\operatorname{topK}_e x_{te}\ (\text{=== topK}_e\,s_{te}),\qquad
w_{t,k}=
\begin{cases}
\dfrac{e^{\,x_{t,\text{ids}_{t,k}}-m_t}}{\sum_{e'}e^{\,x_{te'}-m_t}}\quad(\text{renorm}: \div \sum_{k'}e^{\,x_{t,\text{ids}_{t,k'}}-m_t}), & \text{softmax}\\[2mm]
\sigma(x_{t,\text{ids}_{t,k}})\quad(\text{renorm}: \div \textstyle\sum_{k'}\sigma(x_{t,\text{ids}_{t,k'}})), & \text{sigmoid}
\end{cases}
$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

纯 Vector 的 **persistent 行块 + 原始 logits 硬件排序 + K 切片打分** 算法（行间独立；与源结构差异逐条注明）：

1. **载入**：`T.copy(gating[off:off+real, 0:E], xf[0:real, 0:E])`——GM 行块 → UB，**跨 dtype 自动升 fp32**
   （bf16 必需：`T.vsort` 不支持 bf16，见 `docs/Tilelang.language/排序操作/T.vsort.md` §2.2.1；探针 P1 已证
   bf16→fp32 自动 `T.copy` 可用）。取代源 O1 的 per-lane 标量载入〔§0.6-R1/R2〕。
2. **选择**：`T.vsort(xf, sv, si, descending=True, sort_axis=-1)`——对**原始 logits** 降序排序得值 `sv` 与
   int32 原始专家下标 `si`；取前 K：`sv[:,0:K]`/`si[:,0:K]`〔§0.6-R3〕。取代源 K 轮 warp argmax。
3. **打分（仅选中 K 个）**〔§0.6-R5〕：
   - **softmax**：`T.reduce_max(xf, mr, dim=1)` 得行 max `m`；`T.vsub(sv_k, mr, vk)`（`[BM,K]-[BM,1]` 行广播）；
     `T.vexp(vk, vk)` 得分子 `e^{x_k-m}`；若 `renormalize` 用 `T.reduce_sum(vk,sr)` + `T.vdiv(vk,sr,vk)`（全行分母抵消）；
     否则 `T.vsub(xf,mr,xf)`+`T.vexp(xf,xf)`+`T.reduce_sum(xf,sr)` 得全行分母，再 `T.vdiv(vk,sr,vk)`。
   - **sigmoid**：`T.vsigmoid(vk, vk)`（=σ(x_k)）；`renormalize` 时 `T.reduce_sum(vk,sr)` + `T.vdiv(vk,sr,vk)`。
4. **写出**：`T.copy(vk, topk_weights[off:off+real, 0:K])`、`T.copy(ik, topk_ids[off:off+real, 0:K])`。
5. **调度**：一维 persistent，`num_kernels = min(num_logical, 40)`，核内 `T.serial(num_local_tasks)` grid-stride
   + `if task < num_logical` 守卫〔§0.6-R1，§5.5〕。

**与源算法差异**：① one-warp-per-token → 行块 persistent 分核（R1）；② warp shuffle 归约 → Vector `T.reduce_max`/
`T.reduce_sum`（R2）；③ K 轮 masked argmax → `T.vsort` 降序取前 K（R3）；④ 非整除 E padding → 无 padding（R4）；
⑤ host 侧 renorm → kernel 内融合、softmax+renorm 全行分母抵消（R5）。**保留**：fp32 计算域 + softmax 减 max
稳定化（O5）、`lru_cache`（O7）、输出独立 GM 表（O8）。

### 1.5 数据流图

```
[单 kernel · Vector · persistent num_kernels = min(⌈T/BM⌉, 40)]
GM[gating_output(T,E) fp16/bf16/fp32] --T.copy(行块 slice, 自动升 fp32)--> UB[xf (BM,E) fp32]
    --T.vsort(descending)--> UB[sv (BM,E) fp32] + UB[si (BM,E) int32]     （原始 logits 降序 + 原始下标）
    --取前 K（列 slice）--> UB[vk (BM,K) fp32=原始 logits]、UB[ik (BM,K) int32]

  ├─ softmax:  T.reduce_max(xf)-->UB[mr (BM,1)]
  │            T.vsub(vk,mr,vk) --T.vexp--> UB[vk = exp(x_k-m)]
  │            renorm?  T.reduce_sum(vk,sr) --T.vdiv(vk,sr)--> UB[vk]
  │            else:    T.vsub(xf,mr,xf) --T.vexp--> T.reduce_sum(xf,sr) --T.vdiv(vk,sr)--> UB[vk]
  └─ sigmoid:  T.vsigmoid(vk,vk)  （renorm 时 T.reduce_sum + T.vdiv）

    --T.copy--> GM[topk_weights(T,K) fp32]
    --T.copy--> GM[topk_ids(T,K) int32]     （= UB[ik]，原始专家下标）
无 L1/L0 参与、无 Cube、无 workspace、无核间同步；gating 读一遍、两输出各写一遍。
```

### 1.6 算法调研与优化分析

> **设计第一优先级**：先**调研**——对同一数学语义（逐行打分 + top-k 选择）的算法族回答调研四问（§1.6.0；
> 迁移任务在源算子解读〔§0〕后、耦合性判定与重设计〔§0.5/§0.6〕前完成，调研结论已作为 M1 判定输入）；
> 再在保证数学等价的前提下优化公式（§1.6.1，**含机器验证**）；再把保留下来的计算全部交给向量单元（§1.6.2）；
> 最后回答「在哪个轴上向量化」（§1.6.3，阻塞级）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**逐行 top-k 选择 + 融合打分**这一数学语义的算法族。源算法（warp K 轮 masked argmax）只是基线候选之一。
> 调研深度：**完整调研**（多步 + 规约 + 选择类 → R1 候选表 + R3 四口径 + R4 逐候选评估）。
> 信息源：skill 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**命中行 `ALG-topk`**：
> 「R1：部分选择 vs 全排序；分块 top-k + 归并。R3：O(N·k) vs O(N log N)」）；`kb_search "topk sort 排序 top-k 路由 MoE"`
> （命中 `ALG-topk` / `CG-2026-0030`〔无 fetch-add〕/ `CONST-*`）；本仓 `examples/`（`sort_ops/test_vsort_dev.py`、
> logsumexp、cumprod）；`docs/Tilelang.language/排序操作/`（仅 `T.vsort.md` + `T.vcumsum.md`）、
> `规约操作/T.reduce_max.md`、`比较操作/T.vcmp.md`、`索引与元素操作/T.gather.md`；源算子 §0.3/§0.4；
> **设计期探针**（D-3，实机 NPU：`T.vsort` 多 block top-k 与端到端打分，见下）。
> **互联网检索**：`ALG-topk` 命中行 + 经典算法知识已覆盖该成熟族（部分选择 vs 全排序、分块 top-k + 归并），未触发联网。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | warp per token；lane 持 `ceil(E/32)` 专家；K 轮 masked argmax + `shfl_xor` (val,idx) 归约 | — | — | ✅ 基线 |
| C1 | **原始 logits `T.vsort` 降序取前 K + 仅对选中 K 打分**（**选定**） | `ids = topK(x)`；对 K 个元素算 `exp(x-m)/Σ` 或 `σ(x)`（renorm 反之） | **数学恒等**（softmax/sigmoid 严格单调 → top-k 集合逐位一致，§1.6.1-E1）；归约分组差异属 fp32 ulp | 1 条 `VSortOp` 替代 K 轮归约；仅对 K 个元素打分；softmax+renorm 全行分母抵消 | ✅ **选定** |
| C2 | 全行打分后再 `T.vsort`（先算 `s` 再排序） | `s=f(x)`；`T.vsort(s)` 取前 K | 数学恒等（与源同为「先打分后选」） | 多一次全行打分物化/重排；无收益 | ✅ 对比（弃选，并入 C1 保守形态） |
| C3 | **NPU K 轮 masked argmax**（`T.reduce_max` + 索引恢复） | 每轮 `reduce_max` 得值，`T.vcmp(eq)` 找下标，掩码后重复 K 轮 | 数学恒等 | 免排序；但 `T.reduce_max` **无下标输出**（`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.1 仅 out 值），索引恢复需 `vcmp`+`select`+扫描 → 命中标量/向量混用陷阱（CG-2026-0027/0028） | ✅ 对比（弃选，见 R4） |
| C4 | `T.vtopk` 硬件 top-k 原语 | 直接选 top-k | — | — | ❌ **不存在**（查证：`ls docs/Tilelang.language/排序操作/` 仅 `T.vsort.md`+`T.vcumsum.md`；全 `docs/` grep `vtopk` 无命中；`kb_search topk` 仅返回 `ALG-topk` 候选与 `CG-2026-0030`，无 API） |
| C5 | 升序排序后取末 K / 反转 | `T.vsort(ascending)` + 反转 | 数学恒等 | `descending=True` 直接可用，升序需额外翻转（+repack）| ✅ 对比（弃选） |
| C6 | 分块 top-k + 归并（`ALG-topk` R2 候选） | 行分块各取局部 top-k，再归并 | 数学恒等 | E≤384 单行即驻 UB，分块归并徒增缓冲与多轮发射 | ✅ 对比（弃选） |
| C7 | 阈值 / 分桶选择 | 直方图/阈值筛选 | 容差内（非精确） | 需额外物化、精度不可控 | ✅ 对比（弃选） |
| C8 | host 侧 `torch.topk` | CPU/框架选 | 等价 | 引入 device-host 往返，违反 host 约束（`ascend-constraints.md` §4） | ✅ 对比（弃选） |

> **代数化简空间的结构性结论**：top-k 是**选择**（非代数运算），无「更少算式的恒等变形」；可优化空间集中在
> ① 选择在哪一域做（原始 logits vs 打分后）与 ② 用什么原语做（排序 vs K 轮 argmax）。C1 同时优化二者。

**R2 在线算法**：**有，但本场景不采纳**。流式 top-k（大小 K 的堆，`O(E log K)` 单遍）与 online softmax
（running max/sum）都是公认在线算法；但：① 本源对每行 logits 已**整行驻留 UB**（E≤384 ≪ UB），单遍读取，
在线化不减少 GM 遍数（源与 C1 都是「gating 读一遍」）；② 堆结构在 NPU Vector 上无可表达原语；
③ softmax 非 renorm 需要整行分母，无法用「仅 top-K 状态」增量给出（除非精确重算）。故 R2 结论 = **在线变体存在
但无收益**（结构依据：整行 UB 驻留 + 遍数不变 + 堆不可向量化）。

**R3 复杂度对比**（四口径；代表 shape = deepseek-v3（T=4096, E=256, K=8, bf16）+ kimi-k2（T=4096, E=384, K=8）：
「访存」含 gating 读 + 两张输出写，不含 on-chip；「发射」为 NPU 真实成本主项，按 `CONST-vector-launch-overhead`
~0.5µs/op 口径，`op_tile` 计数见下）：

| 算法候选 | FLOPs（计算量/比较次数，逐行） | 访存量 Bytes（GM） | 扫描遍数（GM） | 中间缓冲峰值 | 向量 op 数 / 核（40 核） |
|---------|---------------|-------------------|---------------|-------------|------------------------|
| C0 基线（warp K 轮 argmax + shuffle） | K·E 比较 + K·(5 轮 shuffle ×3) | `T·E·elem + 2·T·K·4` = 2.36MB(E=256) / 3.41MB(E=384) | 1（gating 读一遍）| 寄存器 `ELEMS` + 2 输出 | **NPU 不可直译**（无 warp）；若译 K 轮 reduce+索引恢复 ≈ K·c op/tile |
| **C1 选定（`T.vsort` 降序 + K 切片打分）** | `O(E log²E)` 排序比较 + `O(E)` 分母 + `O(K)` 打分/归一 | 同 C0 | 1 | UB `(BM,E)·12B` + 小缓冲（§4.5） | **op_tile≈13–14**（载入+vsort+2 切片拷贝+reduce_max+sub+exp+reduce+sub+exp+reduce+div+2 写出）≈ 每核 `⌈num_logical/40⌉·14` ≈ 34–51 op → **17–26µs 发射下界**；vsort 单条成本未标定（见 roofline 未实证项） |
| C2 先打分后排序 | 同 C1 + 全行打分链 | 同 C0 | 1 | 同 C1 | ≈ C1 + 1–3 op/tile（多一次全行打分物化）→ 弃选 |
| C3 NPU K 轮 reduce_max + 索引恢复 | K·E 比较 + K 轮 reduce | 同 C0 | 1 | 同 C1 | K 轮 ×（reduce_max + vcmp + select/扫描）≈ K·(3–5) op → 24–40 op/tile，**且索引恢复不可靠**（CG-2026-0027/0028）→ 弃选 |
| C6 分块 top-k + 归并 | 局部 `O(bn log²bn)` + 归并 | 同 C0 + 局部缓冲 | 1 | 多块局部值/下标缓冲 | 多轮发射 > C1，E 小时 overkill → 弃选 |
| C8 host 侧 topk | `O(T·E log E)`（CPU/框架） | + device→host→device 往返 | 1 + 2 次总线 | host 内存 | **CPU 同步阻塞**，违反 host 约束 → 弃选 |

> 口径脚注：① **FLOPs（浮点/比较运算计数）**只计比较/超越/算术，`O(E log²E)` 为排序次数级估算（VSortOp 内部实现未公开，
> 按经典排序下界标注）；② 访存量 = 必要 I/O（不可消除），三候选相同；③ 扫描遍数指同一数据在 GM 被读次数；
> ④ **发射项**是本算子 NPU 侧主导（§1.6.0 roofline），故 op_tile 计数是候选排序的关键口径。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰按 `negative-claim-evidence.md` §1 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Vector 可做归约，但**one-warp-per-token + shuffle 在 NPU 无对应** → ⚠️ | 寄存器/UB ✅ | lane 32 对齐需 padding | grid 编译期常量 ✅ | ❌ shuffle 无替代即改归约形态 | 源形态不可编译（CUDA 专用 `T.Kernel(threads=)`/`shfl_xor`） | ⚠️ 需重设计（→C1） |
| **C1 选定** | ✅ 纯 Vector：`T.vsort`（`hivm::VSortOp`）+ `reduce_max/reduce_sum` + `vsub/vexp/vsigmoid/vdiv`（fp32 ✓） | UB `(BM,E)·12B` ≪192KB ✅（§4.5） | **尾轴 E = 排序轴**（`T.vsort` 仅尾轴，`docs/Tilelang.language/排序操作/T.vsort.md` §2.3-1）✅ 天然无需 repack；E 任意宽度 ✅ | `T`/`E`/`K` 工厂期常量；核内循环静态界（仅 guard 动态）✅ | ✅ 单引擎 Stream、无 Cube、无 CV 交接；Developer auto-sync | ✅ **探针 P1/P2 设备实测**：多 block 降序 top-k 位精确、fp32/fp16/bf16→fp32 自动 cast、端到端打分 ≤1.8e-7 | ✅ **选定** |
| C2 先打分后排序 | ✅ Vector | ✅ | ✅ | ✅ | ✅ | ✅ 可编译 | ❌ 弃选：多一次全行打分物化，无收益（C1 的子集） |
| C3 K 轮 reduce_max + 索引恢复 | ✅ Vector 但**索引恢复需标量/向量混用** | ✅ | ✅ | ✅ | ⚠️ K 轮携带依赖 | ❌ **不可靠**：`T.reduce_max` 无下标输出；`vcmp`→bool 不可写 GM、需 `vselect`/扫描；标量消费向量产出命中 **CG-2026-0027**（不同步）+ **CG-2026-0028**（`>>`/`alloc_local`/标量 `if_then_else` SIGSEGV） | ❌ 弃选（可行性 + 复杂度劣于 C1） |
| C4 `T.vtopk` | — | — | — | — | — | ❌ **无此 API**（`docs/Tilelang.language/排序操作/` 仅 vsort/vcumsum；全 docs grep 无命中） | ❌ 弃选（不存在） |
| C5 升序+反转 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：`descending=True` 直接可用，升序需额外翻转 |
| C6 分块 top-k + 归并 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：E≤384 单行驻 UB，分块归并徒增缓冲/发射 |
| C7 阈值/分桶 | ✅ | ⚠️ 需物化 | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：精度不可控 + 物化开销 |
| C8 host 侧 | CPU | — | — | — | ❌ device-host 往返 | ✅（可行但违约束） | ❌ 弃选：违反 host 约束（`ascend-constraints.md` §4） |

**调研结论**：选定 **C1 —— 原始 logits 上 `T.vsort` 降序取前 K + 仅对选中 K 打分（+ renorm 融合 / 全行分母抵消）**。
关键依据：(i) **可行性**——C4 无该原语、C3 索引恢复命中 CG-2026-0027/0028、C0 源形态 CUDA 专用；
C1 经 **D-3 探针设备实测可编译、位精确、端到端 ≤1.8e-7**（唯一同时满足全约束的形态）；
(ii) **发射经济性**——D-2 roofline 显示本算子发射/排序主导；C1 用 1 条排序原语替代 C0/C3 的 K 轮归约（`op_tile` 最低）；
(iii) **布局零代价**——`T.vsort` 仅尾轴排序，而 I/O 的 E 轴天然是尾轴，**无需任何 repack**（§1.6.3）。
**与基线结构差异一句话**：保留源「逐行打分 + 逐行选 top-k」的数据流，把 「warp K 轮 masked argmax + shuffle»
换成「**一次 `T.vsort` 降序 + 原始 logits 选择 + K 切片打分**」，把 warp/thread 并行换成行块 persistent 分核、
把 host 侧 renormalize 换成 kernel 内融合。
**调研范围**（「无更优替代」的查证范围）：`algorithm-candidates.md` 的 `ALG-topk` 命中行（C6 分块归并已评估）；
`kb_search "topk sort"` 命中条目；`docs/Tilelang.language/排序操作/` 与 `规约操作/`、`比较操作/`、`索引与元素操作/`
全子目录枚举（确认无 vtopk，`T.gather` 存在备 correction_bias 用）；本仓 examples（sort/logsumexp/cumprod）；
源算子 §0.3/§0.4；**实机探针 P1/P2**；恒等变形空间自查（C1–C8 全枚举，结构判据排除）。**源算法优化手段承接**
（§0.4）：O1→R1、O2→UB、O3→R2、O4→R3（意图承接）、O5 保留、O6→R4、O7/O8 保留、O9→R5。

**设计期 roofline（D-2；协议：`_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：

器件口径声明：引用常数 `CONST-vector-launch-overhead` / `CONST-mte2-degradation` / `CONST-capacity-910B2C` 的
origin 工具链为 **Ascend910B2C（24 AICore）**；本设备实查为 **Ascend910_9362（20 AICore → 纯 Vector 40 核）、
toolchain tilelang 0.1.2+302ad3ce**（2026-10-14）。下述估算按「per-core 发射率不变 × 40 核」外推——属
**未实测假设 + 估算依据**（设备/核数差异已标注；Stage 4 回填实测对账）。

| workload（manifest） | dtype | BM | num_logical | 流量项 | 发射项（op_tile≈14，0.5µs/op） | **估算下界 = max** |
|---|---|---|---|---|---|---|
| smoke (32,128) K=8 | fp16/bf16 | 32 | 1 | 10.2KB ÷ (40·~30GB/s) ≈ 0.01µs | 1/40·14·0.5 = **0.18µs** | ≈0.2µs（发射主导）|
| qwen3-235b-prefill (4096,128) | bf16 | 64 | 64 | 1.31MB ÷ (40·~20GB/s) ≈ 1.6µs | 64/40·14·0.5 = **11µs** | **≈11µs（发射主导）** |
| deepseek-v3-prefill (4096,256) | bf16 | 42 | 98 | 2.36MB ≈ 3.0µs | 98/40·14·0.5 = **17µs** | **≈17µs（发射主导）** |
| kimi-k2-prefill (4096,384) | bf16 | 28 | 147 | 3.41MB ≈ 4.3µs | 147/40·14·0.5 = **26µs** | **≈26µs（发射主导）** |

> 估算下界 = max(流量项, 发射项)（容量项 n/a——纯 Vector 单引擎、无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only`
> 不适用）。**结论：本设计为发射 / 排序主导（非访存主导）**。**未文档化假设项**：`T.vsort`（VSortOp）单条成本
> **未在 pattern-library 实测**（0.5µs/op 是通用向量发射常数，排序原语实际成本可能显著更高）；该量**不影响主选/弃选
> 裁决**（可行性已由 D-3 探针 + 工具链证据硬裁定），仅影响成本特征刻画 → 交 **Stage 4 实测**（§9 R-4）。
> 发射项为串行上界，Developer auto-multi-buffer 下实际可低于该值。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法 C1。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）；
> **等价性机器验证（D-1）**：`examples/fused_topk/_fused_topk_kernel/verify_equiv.py`（torch CPU，fp64 参照 +
> fp32 仿真），结果表内嵌于下；**实机探针（D-3）**：`history_version/design_probe_vsort_topk.py`（P1）与
> `history_version/design_probe_fused_scoring.py`（P2），2026-10-14 实机 NPU 执行。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | **选择改在原始 logits 上做**（承接 §0.6-R3） | 先算全行得分 `s=f(x)`，再 `topK(s)` | `topK(x)`，再仅对选中 K 个元素算 `f(x_k)` | **数学恒等**：`softmax` 分子 `exp` 与 `sigmoid` 均为**严格单调增**函数，逐行 `s=f(x)` 保序（`x_i>x_j ⟺ s_i>s_j`；等值↔等值）→ top-k 下标集合**逐位一致**。机器验证选择集失配 0/136（含 ties） | 省去「先全行打分再排序」的一次全行打分物化（对 sigmoid：全行 σ(E 元素) → 仅 K 个）；排序源数据仍为 `(BM,E)`（vsort 成本不变）|
| E2 | **softmax+renorm 全行分母抵消**（承接 §0.6-R5） | `s_k=e^{x_k-m}/S`（`S=Σ_all e^{x-m}`）再 renorm `s_k/Σ_K s_j` | renorm 直接 `w_k=e^{x_k-m}/Σ_K e^{x_j-m}` | **数学恒等**：`(e^{x_k-m}/S)/(Σ_K e^{x_j-m}/S) = e^{x_k-m}/Σ_K e^{x_j-m}`（S>0 约去）。机器验证 renorm 用例违反率 0 | 消去全行 `reduce_sum`（1 op/tile）+ 全行除法；仅需 K 元素 `reduce_sum` + `vdiv` |
| E3 | **非整除 E 无 padding**（承接 §0.6-R4） | 越界专家填 `-inf` 后再选 | 排序/打分只作用于真实 E 列（`(BM,E)` 满行） | **数学恒等**：源 pad 的 `-inf` 在源中不改变输出（`exp(-inf-m)=0`、argmax 不选 `-inf`），消除 padding 不改变任何结果。机器验证含 E=8/33/100 非整除用例违反率 0 | 消除 masked load 分支与 `-inf` 哨兵；E 任意宽度直承 |
| E4 | **softmax 减行 max 稳定化**（保留源 O5） | 源同款 | `e^{x-m}`，`m=Σ reduce_max(x)` | 保留源数值路径（非新优化）；`T.reduce_max` 承接源 warp max 归约，**归约分组不同** → fp32 ulp（机器验证 max_rel_fp32=1.075e-06，违反率 0） | 防 `exp` 上溢；分组差异在容差内 |
| E5 | **全 dtype 统一 fp32 计算域 + bf16 升域**（承接 §0.4-O5） | 源：host 升 fp32 → kernel fp32 | 同源计算域；kernel 直接收原生 dtype，`T.copy` 自动升 fp32 | 保留源数值路径；且是 **API 必要性**——`T.vsort` 的 `src`/`dst_value` 仅 fp16/fp32、**bf16 ×**（`docs/Tilelang.language/排序操作/T.vsort.md` §2.2.1），bf16 必须升 fp32。舍入点与源一致（fp32 中间 + 输出 fp32） | 消除 host 侧 `gating_output.to(torch.float32)`（一次额外算子 + GM 往返）；bf16 路径由 `T.copy` 自动 `VCast`（探针 P1 已证）|
| E6 | **renormalize 融合进 kernel 的 K 切片**（承接 §0.6-R5） | kernel 出未归一权重 + host 侧 `w/w.sum` | kernel 内 `T.reduce_sum(w)` + `T.vdiv(w, sr)` | **数学恒等**：同一除法，仅执行位置不同。机器验证 renorm 用例违反率 0 | 消除第二次 host 算子与一次输出 GM 往返；只在 K 元素切片上归约（非全行） |

**机器验证结果表**（`verify_equiv.py`，2026-10-14 执行，torch CPU，exit 0；基线 = 源语义
`softmax/sigmoid → topk(sorted) → 可选 renorm` 的 fp64 复刻；候选 = E1–E6 的组合式 fp64 计算 +
fp32 域仿真；判定口径：harness 容差 rtol=atol=1e-3，`max_rel` 为候选/基线相对误差最大值）：

| 验证项 | dtype | 用例数 | max_rel | 违反率 | 结论 |
|--------|-------|-------|---------|--------|------|
| E1 选择等价（原始 logits＝得分选择，选择集逐位一致） | fp64 | 136 | 0（选择集失配 0） | 0 | **EQUIV_PASS** |
| E2/E3/E4/E5/E6 权重等价（fp64 域，含 non-pow2 E / ties / ±极值） | fp64 | 136 | 5.6e-17 | 0 | **EQUIV_PASS** |
| 组合 fp32 域仿真（NPU 计算域 vs fp64 基线） | float32/float16/bfloat16 | 136 | **1.075e-06** | **0** | **EQUIV_PASS** |
| **合计** | — | **136** | — | **0** | **ALL EQUIV_PASS** |

用例覆盖：随机网格 `E∈{8,33,100,128,256,384} × K∈{1,2,4,8} × {softmax,sigmoid} × renorm{False,True}`（96 例）+
角点 `large_magnitude×50 / all_equal_ties / many_ties / pos_neg_mix×200 / single_expert / K=E / near_tie 1e-6 /
large_batch(256,384)`（32 例）+ dtype 舍入路径 fp16/bf16（8 例）。**无 `EQUIV_FAIL` 项**，E1–E6 全部采纳。

**实机 NPU 探针结果**（D-3，2026-10-14，Ascend910_9362，toolchain tilelang 0.1.2+302ad3ce）：

| 探针项 | 输入 | 结果 | 结论 |
|--------|------|------|------|
| P1-A `T.vsort` 多 block 降序 top-k | (T=32,E=128,K=8) fp32 | value_max_err **0**、gather_err **0**、set_ok True | ✅ 多 block（grid>1）可用、降序、值+int32 下标正确 |
| P1-B 同上 E=384 | (T=64,E=384,K=8) fp32 | 全部 **0** | ✅ Kimi K2 专家数可用 |
| P1-C bf16→fp32 自动 cast 供 vsort | (T=32,E=128) bf16 / fp16 | 全部 **0** | ✅ `T.copy` 跨 dtype 自动升 fp32 可直接喂 `T.vsort` |
| P2-A softmax no-renorm 端到端 | (32,128) fp32 | w_err **1.79e-07**（vs torch golden）| ✅ 减 max + 全行分母路径正确 |
| P2-B softmax + renorm | (32,128) fp32 | w_err **1.19e-07** | ✅ 全行分母抵消（E2）成立 |
| P2-C sigmoid + renorm | (32,256) fp32 | w_err **1.49e-08** | ✅ |
| P2-D sigmoid no-renorm | (32,384) fp32 | w_err **5.96e-08** | ✅ |

**优化结论**：采纳 E1–E6 共 6 项（E1/E2/E3/E6 为收益项；E4/E5 为保留源数值路径 + API 必要性并附论证）；
优化后公式 = §1.3 的两式（供 §3.1 拆解）。**无优化空间项**：top-k 选择本身无代数化简（选择非代数运算，
§1.6.0 R1 结构性结论）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/`、`testing/`
> 或实机探针佐证（列内标注）。

| # | 计算点 | 原实现形态（源/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | top-k 选择 | 源 K 轮 masked argmax + `shfl_xor` | `T.vsort(descending=True)` 取前 K（`docs/Tilelang.language/排序操作/T.vsort.md`；探针 P1） | ✅（消除） | — |
| 2 | 行 max（softmax 稳定化） | 源 lane 局部 max + shuffle 树归约 | `T.reduce_max(xf, mr, dim=1)`（`规约操作/T.reduce_max.md`） | ✅ | — |
| 3 | 减行 max（行广播） | per-lane 标量减 | `T.vsub(vk, mr, vk)`（`数学操作/T.vsub.md` §2.2.2 行广播 `[M,N]-[M,1]`）| ✅ | — |
| 4 | exp | per-lane `T.exp` | `T.vexp(vk, vk)`（`数学操作/T.vexp.md`，fp32 ✓） | ✅ | — |
| 5 | 行 sum（softmax 分母 / renorm 分母） | 源 shuffle 树归约 | `T.reduce_sum(src, sr, dim=1)`（`规约操作/T.reduce_sum.md`）| ✅ | — |
| 6 | sigmoid | per-lane `1/(1+exp(-x))` | `T.vsigmoid(vk, vk)`（`数学操作/T.sigmoid.md`，fp32 ✓；探针 P2-C/D） | ✅ | — |
| 7 | 归一化除法（行广播） | 标量除 | `T.vdiv(vk, sr, vk)`（`数学操作/T.vdiv.md` §2.2.2 行广播）| ✅ | — |
| 8 | GM↔UB 载入/写出（含 dtype 升/降） | per-lane 标量 load/store | `T.copy` slice（跨 dtype 自动 `VCast`；`内存操作/T.copy.md`；探针 P1-C）| ✅ | — |
| 9 | 输出行块提取（`sv[:,0:K]`/`si[:,0:K]` → `vk`/`ik`） | 逐元素 | `T.copy` 列 slice（`内存操作/T.copy.md` §2.2.2；探针 P1/P2 实证）| ✅ | — |
| 10 | persistent 任务循环 `for s in T.serial(num_local_tasks)` | block 级串行 | 无 | ❌（保留） | **block 级任务映射 + grid-stride 调度**：核内任务序（`task=s·num_kernels+cid`）是分核结构本身（`core-split-strategy.md` §1 要素③ 极大规模方案），非逐元素热点；无逐元素等价 API |
| 11 | 任务/块索引算术（`task`、`off`、`real_m=T.min(...)`） | 标量整除/取余/min | 无（纯 TIR 标量算术） | ❌（保留） | **block 级索引/任务映射与动态尾块边界**：每 tile O(1) 条，非逐元素热点；进 slice extents |
| 12 | host 侧 `lru_cache` 工厂特化 / `parse_config` | Python 工厂期标量 | 无 | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(T,E,K,scoring_func)` 一次）|

**向量化结论**：逐元素 / 规约 / 选择计算点（#1–#9）**全部向量化**（替代 API 均有 `docs/` 条目或实机探针佐证：
`T.vsort`/`T.reduce_max`/`T.reduce_sum`/`T.vsub`/`T.vexp`/`T.vsigmoid`/`T.vdiv`/`T.copy`）；保留 3 类标量/循环
（#10–#12），理由类别：分核与任务调度串行（#10）、block 索引 + 动态尾块边界（#11）、host 元数据（#12）——
**均非逐元素热点**（每 tile/任务 1 次发射量级），与 §6 循环结构一致（§6 无任何逐元素标量循环）。
源实现中唯一的逐元素热点（K 轮 per-lane argmax 与 shuffle 归约）已被 #1/#5 消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`gating_output (T,E)` 行主序进、`topk_weights (T,K)`/`topk_ids (T,K)` 行主序出）；
> **核内布局与 lane 映射是设计变量**。本算子为**选择 + 规约类**，有 2 个非 batch 维度（T 为任务维、E 为选择/规约维）。

**轴质量评分**（候选轴 = 选择/归约轴 E；T 为任务分核轴）：

| 评分项 | 方案 A：行块 `(BM,E)` 驻留，lane→E（**排序轴 = 尾轴 E**） | 方案 B：转置 `(E,BM)` 后排序 | 方案 C：每行 BM=1 |
|--------|------|------|------|
| 整除性 | E 任意宽度（`T.vsort` 尾轴全排序）| 需核内转置链 | E 任意 |
| **排序/选择原语匹配** | ✅ `T.vsort` **仅支持尾轴**（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3-1），E 恰为尾轴 → 直接可用 | ❌ `T.vsort` 不支持非尾轴排序（编译期报错）→ 必须转置 | ✅ 尾轴 |
| repack 代价 | **无**（I/O 原生行主序，E 连续）| 需 `T.transpose` 二轴交换链（UB 级开销 + 额外缓冲）| 无 |
| 发射经济性 | ✅ `(BM,E)` 宽块，`CONST-vector-launch-overhead`「宽块恒优于窄块」；BM 越大 op 数越少 | 多转置 op | ❌ BM=1 → `num_logical=T` → op 数最多（发射灾难）|
| UB 容量影响 | `(BM,E)·12B + 小缓冲`（§4.5）| 多一份转置缓冲 | `(1,E)·12B`，UB 浪费（无法摊薄）|
| 采纳 | ✅ **主选** | ❌ 弃选 | ❌ 弃选 |

**规约类 lane 映射正交决策：水平归约 vs 垂直扫描**（正交于「选哪条轴」的必选候选；本算子含行归约子步骤
softmax 行 max/sum、renorm 分母 `Σ_K`，§1.6.2-#2/#5）：

| 评分项 | H：水平归约（lane→归约轴 E，`T.reduce_*(dim=1)` 向量 reduce）**主选** | V：垂直扫描（lane→独立行 T，串行步进归约维 E）|
|--------|------|------|
| lane 映射语义 | 归约轴 E 落到向量 lane 宽度，`T.reduce_max/sum(dim=1)` 在 E 上做向量归约 | 行轴 T 落到向量 lane，每 lane 独立串行累加 E 个元素 |
| **与 `T.vsort` 兼容性** | ✅ 同一 E 尾轴：排序后 `(BM,E)` 整行驻留，`reduce_*` 直接作用在同一 buffer → **零 repack** | ❌ 需令行轴成为 lane 连续轴（对 `(BM,E)` 做转置、尾轴换为 T），与 `T.vsort` 强制尾轴 E 排序冲突；且 I/O 原生 E 为尾轴 → 需核内转置链 |
| 整除性 / 尾 lane 浪费 | ✅ E 任意宽度（reduce 原语内部处理尾 lane）| ❌ 行轴映射到向量宽度（fp32 ×4/×8）→ BM 须向 lane 宽度对齐，尾 lane 浪费 |
| 累加链形态 | ✅ 多路/树形并行归约（`T.reduce_*`），无长串行依赖 | ❌ 串行累加链长度 = E（256/384 步），ILP 受限 |
| repack / UB 影响 | ✅ 无额外缓冲（复用 `xf` 原地）| ❌ 需 `(E,BM)` 转置缓冲（≈`BM·E·4` 额外 UB）|
| 发射经济性 | ✅ 1 条 reduce 原语/行块 | ❌ 每行 E 次串行步进的循环/发射开销 |
| 采纳 | ✅ **主选** | ❌ **结构否决**（与 `T.vsort` 尾轴整行驻留约束冲突）|

> **两遍 vs online 与 lane 映射的交互**：§1.6.0-R2 已判定在线变体无收益（整行 UB 驻留、GM 遍数不变、堆不可向量化）；
> 故水平归约沿用「排序后两遍（max 一遍 / sum 一遍）」形态，lane 映射不改变 GM 扫描遍数（均 1 遍）。
> 垂直扫描不会减少任何遍数，反而引入 E 长串行链与转置，被结构否决——**水平归约被 `T.vsort` 的尾轴整行驻留约束锁定为主选**。

**候选矩阵**（含源码 GPU 轴选择对照——源码 warp→token、lane→专家子集（步长 32），是输入非结论）：

| # | 布局方案 | 向量化轴 / 关键结构 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| 1 | **行块 `(BM,E)` UB 驻留 + 尾轴排序**（选定） | E（尾轴）；`T.vsort(sort_axis=-1)` + `reduce_*(dim=1)` | 无（I/O 原生）| 单 kernel、零 repack、承载全部优化 | ✅ **主选** |
| 2 | 核内转置 `(E,BM)` 后用 `vtopk`/非尾轴排序 | — | `T.transpose` 链 | ❌ 前提被否：`T.vsort` 不支持非尾轴（文档明示编译期报错），且无 `T.vtopk` | ❌（文档否决） |
| 3 | 每行一块（BM=1） | E | 无 | `num_logical=T=4096` → op 数 ~6× 于 BM=28，发射灾难 | ❌（量化否决） |
| 4 | 列切片先行（只搬 top 候选列到小缓冲排序）| E | 需额外候选筛选 | 需先筛选（=选择本身），循环依赖 | ❌（逻辑不成立） |
| 5 | 垂直扫描 lane 映射（lane→T 行、串行步进 E）| 行轴 T | 需核内转置（尾轴换 T）| ❌ 与 `T.vsort` 尾轴 E 排序冲突 + E 长串行累加链 → 发射/依赖双劣势 | ❌（结构否决） |

**布局决策结论**：选定 **#1**：核内布局 = 行块 `(BM,E)` 行主序驻留 UB、**向量化/排序轴 = 尾轴 E**、
**规约 lane 映射 = 水平归约（lane→E，`T.reduce_*(dim=1)`）**、repack 无、分核 = persistent 行块（§5.5）。
该结论同步落入 §3.3 伪代码与 §6 循环结构（排序/逐元素/归约的向量维 = E 轴；buffer 形状 = `(BM,E)` + `(BM,1)` + `(BM,K)`；
归约以 `T.reduce_*(dim=1)` 水平形态呈现，非逐行串行扫描）。
**弃选理由**：#2 因 `T.vsort` 仅尾轴 + 无 `T.vtopk`（文档/检索否决）；#3 量化发射灾难；#4 逻辑循环依赖；
#5 垂直扫描因与 `T.vsort` 尾轴整行驻留约束冲突 + E 长串行累加链被**结构否决**（量化评分见上「水平归约 vs 垂直扫描」表）。
**主选与弃选判定裕度不依赖未实证常数**（#2 被文档能力硬否决、#3 是 ~6× 量级差、#5 被 `T.vsort` 尾轴约束硬否决 → 不触发实验裁决模式）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（harness 迁移 prompt 指定 developer）。

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.vsort` + `reduce_max/reduce_sum` + 逐元素链），无 matmul、无 Cube | 不需 L1/L0；`T.alloc_shared`→UB 即可 |
| 选择原语 | `T.vsort`（Developer/Expert 均可用，`testing/npuir/sort_ops/test_vsort_dev.py` 为 **Developer** 通过用例；**探针 P1 实证**）| Developer 可用 |
| 内存层级 | 仅 GM ↔ UB，无跨引擎、无 workspace | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作 | Developer 自动同步即可 |
| 家族先例 | logsumexp（行归约）、cumprod（persistent 分核）均为 Developer 且 L0 通过 | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging + 排序/打分缓冲；编译器映射 UB）|
| 计算方式 | `T.vsort`（选择）+ `T.reduce_max`/`T.reduce_sum`（行归约）+ `T.vsub`/`T.vexp`/`T.vsigmoid`/`T.vdiv`（逐元素/行广播）；无逐元素标量循环（§1.6.2）|
| 同步 | Developer 自动同步（无 `sync_block_set/wait`）|
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `xf ← gating[off:off+real, 0:E]` | GM 行块 → UB（跨 dtype 自动升 fp32）|
| 2 | `(sv, si) ← sort_desc(xf)` | 原始 logits 降序排序（值 + 原始下标）|
| 3 | `x_K ← sv[:,0:K]`；`ids ← si[:,0:K]` | 前 K（原始 logits 值 + 专家下标）|
| 4 | `m ← max_e xf` | 行 max（softmax 稳定化；sigmoid 跳过）|
| 5 | `w ← exp(x_K − m)` / `σ(x_K)` | 仅对选中 K 个打分 |
| 6 | `den ← Σ_all exp(xf−m)`（softmax 非 renorm）/ `Σ_K w`（renorm）| 分母（renorm 时 softmax 全行分母抵消）|
| 7 | `w ← w / den` | 归一化（行广播；非 renorm softmax 与 renorm 均需）|
| 8 | `topk_weights[off:off+real, :] ← w`；`topk_ids[off:off+real, :] ← ids` | UB → GM 写出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| 1 载入 | GM→UB 行块（升 fp32） | `T.copy(gating[off:off+real, 0:E], xf[0:real, 0:E])` | 双显式 slice；跨 dtype 自动 `VCast` | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.1/§2.2.2；**探针 P1-C** |
| 2 选择 | 降序排序 | `T.vsort(xf, sv, si, descending=True, sort_axis=-1)` | `si` int32；仅尾轴 | Developer | `docs/Tilelang.language/排序操作/T.vsort.md`；`testing/npuir/sort_ops/test_vsort_dev.py`；**探针 P1** |
| 3 取前 K | 列 slice 拷贝 | `T.copy(sv[0:real,0:K], vk[0:real,0:K])`；`T.copy(si[0:real,0:K], ik[0:real,0:K])` | src/dst 同 slice | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2；**探针 P1/P2** |
| 4 行 max | `max_e` | `T.reduce_max(xf, mr, dim=1, clear=True)` | `dst` rank=src rank，归约维=1（`(BM,E)→(BM,1)`）| Developer | `docs/Tilelang.language/规约操作/T.reduce_max.md`；logsumexp single |
| 5a 减 max | 行广播减 | `T.vsub(vk, mr, vk)` | `[BM,K]-[BM,1]` 行广播；原地 | Developer | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2 |
| 5b exp | `e^{·}` | `T.vexp(vk, vk)` | fp32 ✓；原地 | Developer | `docs/Tilelang.language/数学操作/T.vexp.md` |
| 5c sigmoid | `σ` | `T.vsigmoid(vk, vk)` | fp32 ✓ | Developer | `docs/Tilelang.language/数学操作/T.sigmoid.md`；**探针 P2** |
| 6a 全行分母 | `Σ_all exp(xf−m)` | `T.vsub(xf, mr, xf)` → `T.vexp(xf, xf)` → `T.reduce_sum(xf, sr, dim=1, clear=True)` | 原地复用 `xf` | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md`；logsumexp single |
| 6b K 切片分母 | `Σ_K w` | `T.reduce_sum(vk, sr, dim=1, clear=True)` | `(BM,K)→(BM,1)` | Developer | 同上；**探针 P2** |
| 7 归一化 | 行广播除 | `T.vdiv(vk, sr, vk)` | `[BM,K]/[BM,1]` 行广播；原地 | Developer | `docs/Tilelang.language/数学操作/T.vdiv.md` §2.2.2；**探针 P2** |
| 8 写出 | UB→GM | `T.copy(vk[0:real,0:K], topk_weights[off:off+real,0:K])`；`T.copy(ik[0:real,0:K], topk_ids[off:off+real,0:K])` | 双 slice | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 |
| 分核 | persistent | `T.Kernel(num_kernels, is_npu=True) as (cid,_)` + `for s in T.serial(num_local_tasks)` + `if task < num_logical` | 一维 + 静态界 + 动态 guard | Developer | `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`；`docs/开发指南.md` §3.3 |

> **注（correction_bias 变体前瞻）**：`with_correction_bias=True` 时选择依据为 `σ(x)+bias`、输出为 `σ(x)`。
> 可行落地路径（首版不做，§9 R-1）：排序 biased 值 → `T.vsort` 得下标 → 用 `T.gather(xf, unbiased, si_K, [BM,K])`
> （`docs/Tilelang.language/索引与元素操作/T.gather.md`，需 Stage 3 验证 (BM,K) 索引形态）再 `T.vsigmoid`。

### 3.3 计算伪代码

工厂期全静态量：`BM`、`num_logical = ceil(T/BM)`、`num_kernels = min(num_logical, 40)`、
`num_local_tasks = ceil(num_logical/num_kernels)`（§5.5）。dtype 分派：fp32 原域；fp16/bf16 由 `T.copy` 自动升 fp32。
**规约 lane 映射**（§1.6.3 主选）：行 max/行 sum/renorm 分母均以**水平归约** `T.reduce_*(dim=1)` 向量原语实现
（lane 映射到归约轴 E），**非垂直扫描**（无逐行串行步进 E 的循环）。

```python
def _build(T, E, K, scoring_func, renormalize, BM):
    num_logical = ceildiv(T, BM)
    num_kernels = min(num_logical, 40)                 # 纯 Vector 物理核数（实查 aicore 20 × 2）
    num_local_tasks = ceildiv(num_logical, num_kernels)  # 静态上界

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func():
        @T.prim_func
        def main(gating: T.Tensor((T, E), in_dtype),          # fp16 / bf16 / fp32
                 topk_weights: T.Tensor((T, K), "float32"),
                 topk_ids: T.Tensor((T, K), "int32")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xf = T.alloc_shared((BM, E), "float32")   # 升 fp32 行块
                sv = T.alloc_shared((BM, E), "float32")   # 降序值
                si = T.alloc_shared((BM, E), "int32")     # 降序原始下标
                mr = T.alloc_shared((BM, 1), "float32")   # 行 max
                sr = T.alloc_shared((BM, 1), "float32")   # 分母
                vk = T.alloc_shared((BM, K), "float32")   # 选中原始 logits → 权重
                ik = T.alloc_shared((BM, K), "int32")     # 选中专家下标
                for s in T.serial(num_local_tasks):
                    task = s * num_kernels + cid
                    if task < num_logical:
                        off = task * BM
                        real = T.min(BM, T - off)
                        # 1. 载入（自动升 fp32）
                        T.copy(gating[off:off + real, 0:E], xf[0:real, 0:E])
                        # 2. 选择（原始 logits 降序）
                        T.vsort(xf, sv, si, descending=True, sort_axis=-1)
                        T.copy(sv[0:real, 0:K], vk[0:real, 0:K])
                        T.copy(si[0:real, 0:K], ik[0:real, 0:K])
                        # 3-7. 打分 / 归一化（仅选中 K）
                        if scoring_func == "softmax":
                            T.reduce_max(xf, mr, dim=1, clear=True)
                            T.vsub(vk, mr, vk)
                            T.vexp(vk, vk)
                            if renormalize:
                                T.reduce_sum(vk, sr, dim=1, clear=True)   # 全行分母抵消
                            else:
                                T.vsub(xf, mr, xf)
                                T.vexp(xf, xf)
                                T.reduce_sum(xf, sr, dim=1, clear=True)
                            T.vdiv(vk, sr, vk)
                        else:  # sigmoid
                            T.vsigmoid(vk, vk)
                            if renormalize:
                                T.reduce_sum(vk, sr, dim=1, clear=True)
                                T.vdiv(vk, sr, vk)
                        # 8. 写出
                        T.copy(vk[0:real, 0:K], topk_weights[off:off + real, 0:K])
                        T.copy(ik[0:real, 0:K], topk_ids[off:off + real, 0:K])
        return main
    return _func
```

### 3.4 API 可行性确认

| API | 存在性佐证 | 探针 |
|-----|-----------|------|
| `T.vsort`（降序、多 block、fp32/int32 下标）| `docs/Tilelang.language/排序操作/T.vsort.md`；`testing/npuir/sort_ops/test_vsort_dev.py` | **P1 实证** |
| `T.copy`（跨 dtype 自动 cast、列 slice、双 slice 尾块）| `docs/Tilelang.language/内存操作/T.copy.md`；logsumexp single | **P1-C/P2 实证** |
| `T.reduce_max` / `T.reduce_sum`（fp32，`(BM,E)→(BM,1)`）| `docs/Tilelang.language/规约操作/T.reduce_max.md`、`T.reduce_sum.md`；logsumexp single | **P2 实证** |
| `T.vsub` / `T.vdiv`（行广播 `[BM,N]±[BM,1]`，原地）| `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2、`T.vdiv.md` §2.2.2 | **P2 实证** |
| `T.vexp` / `T.vsigmoid`（fp32，原地）| `docs/Tilelang.language/数学操作/T.vexp.md`、`T.sigmoid.md` | **P2 实证** |

### 3.5 技术约束确认

**强制检测结果**：
- **三维 Kernel**：`T.Kernel(num_kernels, is_npu=True)` **一维**——源为 `T.Kernel(grid, threads=...)` 一维 grid，无三维；✅ 无冲突。
- **GPU 专用 API**：源用 `T.get_thread_binding`/`T.shfl_xor`/`T.alloc_local`/`T.Kernel(threads=)`——**全部 CUDA 专用**，
  本设计已全部替换为 Vector 原语（§0.5/§0.6）；✅ 无残留。
- **GEMM 非整除 / L0C 溢出**：本算子**无 GEMM、无 Cube、无 L0/L0C**；✅ n/a。
- **非整除 T**：`real = T.min(BM, T − off)` + 双 slice 处理（§5.4）；✅ 有策略。
- **分核策略**：`num_logical` vs 物理核 40（实查）三要素见 §5.5；✅ 齐备。
- **工具链陷阱规避**（承接 `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel` 先例）：① **不用 `T.alloc_local`**
  （CG-2026-0028b：`alloc_local + T.serial` SIGSEGV）；② **不用标量 `T.if_then_else`/`>>`**（CG-2026-0028a/c）；
  ③ **不标量消费向量产出**（CG-2026-0027）；④ **不在 `T.Parallel` 体内直写 GM**（CG-2026-0030）；⑤ **不用
  `T.atomic_add` 标量载荷**（CG-2026-0029）。本设计全部数据流为向量→向量（UB）+ `T.copy`→GM，guard 仅
  `if task < num_logical` 这一 block 级分段条件（cumprod 先例同款））。

**本项目同类实现参考**（详见 §10 交付清单）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | Shape | dtype | Layout | 说明 |
|------|-------|-------|--------|------|
| `gating_output` | `(T, E)` | float16 / bfloat16 / float32 | 行主序连续 | 门控 logits；T/E 为编译期常量（工厂按 shape 特化）|

### 4.2 输出张量

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `topk_weights` | `(T, K)` | float32 | 选中权重（renormalize 时已归一）|
| `topk_ids` | `(T, K)` | int32 | 选中专家原始下标 |

输出为独立 GM 张量（`out_idx=[1,2]` 由 TileLang 设备侧分配，或由 op 层分配后传入）。

### 4.3 中间缓冲区（UB）

| 名称 | Shape | dtype | 字节 | 用途 |
|------|-------|-------|------|------|
| `xf` | `(BM, E)` | float32 | `BM·E·4` | 升 fp32 行块（排序源；非 renorm softmax 时原地复用为全行 exp）|
| `sv` | `(BM, E)` | float32 | `BM·E·4` | 降序值（`T.vsort` 输出）|
| `si` | `(BM, E)` | int32 | `BM·E·4` | 降序原始下标（`T.vsort` 输出）|
| `mr` | `(BM, 1)` | float32 | `BM·4` | 行 max |
| `sr` | `(BM, 1)` | float32 | `BM·4` | 分母 |
| `vk` | `(BM, K)` | float32 | `BM·K·4` | 选中原始 logits → 权重 |
| `ik` | `(BM, K)` | int32 | `BM·K·4` | 选中专家下标 |

**大缓冲** = `xf`+`sv`+`si` = `BM·E·12` 字节/行块。

### 4.4 内存搬运路径

```
GM[gating_output(T,E)] --T.copy(slice, 自动 VCast→fp32)--> UB[xf (BM,E) fp32]
UB[xf] --T.vsort--> UB[sv fp32] + UB[si int32]
UB[sv/si] --T.copy 列 slice--> UB[vk/ik (BM,K)]
UB[vk] --reduce/sub/exp(or sigmoid)/sum/div--> UB[vk (归一权重)]
UB[vk] --T.copy--> GM[topk_weights(T,K) fp32]
UB[ik] --T.copy--> GM[topk_ids(T,K) int32]
无 L1、无 L0A/L0B/L0C、无 workspace、无核间搬运。
```

### 4.5 UB 内存预算

参照 `CONST-capacity-910B2C`（UB = 192KB/AIV；显式 alloc 结构实测余量 ~12%，即有效预算 ≈ 168KB 可用）。
本设计的 UB 占用（大缓冲 `BM·E·12` + 小缓冲 `BM·(16+8K)` 量级）按 **UB_BUDGET = 128KB** 上限设计（保守留余量，
应对 `T.vsort` 内部可能的 scratch）：

$$
BM = \min\Big(BM_{max}=64,\ \Big\lfloor \tfrac{UB\_BUDGET}{E\cdot 12}\Big\rfloor\Big),\qquad
BM = \min(BM,\ T)\ \text{(avoid empty blocks)}
$$

| workload | E | `⌊128KB/(E·12)⌋` | BM | 大缓冲占用 | 小缓冲（`mr/sr/vk/ik`）| 合计 ≈ |
|---|---|---|---|---|---|---|
| smoke (32,128) | 128 | 85 → 64 | 32 | 48KB | <1KB | ~49KB |
| qwen3 (4096,128) | 128 | 85 → 64 | 64 | 96KB | <2KB | ~98KB |
| deepseek (4096,256) | 256 | 42 | 42 | 126KB | <2KB | ~128KB |
| kimi (4096,384) | 384 | 28 | 28 | 126KB | <2KB | ~128KB |

> 若 Stage 3 实测 `T.vsort` 需要额外 UB scratch 致溢出，则回退缩小 `BM`（§9 R-4 的调优杠杆）。

### 4.6 动态轴定义

`T`/`E`/`K` 均为**工厂期编译常量**（op 层 `_get_kernel` 按 shape 特化，`lru_cache` 缓存）；运行期无动态 shape
（仅尾块 `real = min(BM, T-off)` 与任务 guard 为运行期值，但循环边界静态）。dtype 为编译期分派（fp16/bf16/fp32）。

### 4.7 JIT 配置

`@tilelang.jit(out_idx=[1,2], target="npuir")`；Developer 模式（`TILELANG_ASCEND_MODE` 由 harness 设定）；
无 `pass_configs`、无 `compile_flags`。工厂层 `functools.lru_cache` 按 `(T,E,K,scoring_func,renormalize,dtype)` 特化。

---

## 5. Tiling 策略

### 5.1 计算类型

**纯 Vector**（无 Cube/matmul）：选择（`T.vsort`）+ 行归约（`T.reduce_max/reduce_sum`）+ 逐元素/行广播
（`T.vsub/vexp/vsigmoid/vdiv`）。仅需 UB，无 L1/L0。

### 5.2 Block 划分与 Tile Shape

| 参数 | 取值 | 依据 |
|------|------|------|
| Tile shape | `(BM, E)` 行块（整 E 驻留 UB）| E 为排序/规约尾轴，必须整行驻留（`T.vsort` 全排序 + 行归约）|
| `BM` | `min(64, ⌊UB_BUDGET/(E·12)⌋, T)` | UB 预算（§4.5）；`BM` 越大 op 数越少（发射摊薄）|
| `K` | 编译期常量（源 `top_k`）| manifest（1/8）|
| 输出 tile | `(BM, K)` | 前 K 列 |

### 5.3 约束分析

- **尾轴对齐**：`T.copy` 要求尾轴 32B 对齐——E 行主序连续天然满足（fp32 UB）；`(BM,K)` 切片 K=1 时 4B，
  T.copy 尾块按元素处理（logsumexp P1 用非 256 对齐 N 已证可用）。
- **`T.vsort` 限制**：仅尾轴排序（E 满足）、fp16/fp32（fp32 满足）、`si` int32（满足）、同 shape src/dst（满足）。
- **UB 容量**：§4.5 控制在 128KB 预算内。
- **无分形 / L0C 约束**（无 Cube）。

### 5.4 非整除处理策略

| 维度 | 情况 | 策略 |
|------|------|------|
| **T（行）** | `T % BM ≠ 0` | 尾块 `real = T.min(BM, T − off)`，`T.copy` **src/dst 双显式 slice** `gating[off:off+real]`↔`xf[0:real]`，越界行不载入/不写出；与源 `if token_id < num_tokens` 守卫等价。**不 padding**（host 禁改输入真实内容）|
| **E（专家）** | 任意 | 排序/打分作用于真实 E 列，**无需 padding**（§1.6.1-E3）；E 不需 32/16 整除 |
| **K** | `K ≤ E`（op 层已校验）| `sv[:,0:K]` 直接切片 |

### 5.5 分核策略（三要素）

**① 逻辑核数计算**：`num_logical = ceil(T / BM)`。
- (4096,128) BM=64 → **64**；(4096,256) BM=42 → **98**；(4096,384) BM=28 → **147**；(32,128) BM=32 → **1**。

**② 物理核数依据**：`NPUUtils.get().get_aicore_num()` **实查 = 20**（代码
`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`，2026-10-14 执行，设备
**Ascend910_9362**）；纯 Vector 算子核数翻倍 → **物理核数 = 20 × 2 = 40**（依据 `_shared/standards/core-split-strategy.md` §1 要素②）。

**③ 规模判定与分核方案**：
- **smoke (32,128)**：`num_logical = 1 ≤ 40` → **无需适配**（结论 + 依据：单核即可）。
- **full workload (T=4096, E=128/256/384)**：`num_logical = 64/98/147 > 40`；且 `BM` 已受 UB 预算上界约束
  （§4.5，E=384 时 BM≤28，无法继续增大以把 `num_logical` 压到 ≤40）→ **极大规模** →
  **固定启动内核数 `num_kernels = min(num_logical, 40) = 40`**，核内 `T.serial(num_local_tasks)` grid-stride
  （`num_local_tasks = ceil(num_logical/40)` = 2/3/4，**静态值**）+ 动态 `if task < num_logical` 守卫。
  （依据 `core-split-strategy.md` §1 要素③「极大规模」方案；先例 `examples/cumprod/_cumulative_kernel` 同款。）

> 注：`T=4096 = 2^12` 不含因子 5，无法令 `num_logical` 恰为 40 的整数倍（无「中等规模对齐」解），故取固定核数 +
> 核内串行方案。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环 | 类型 | 边界 | 说明 |
|------|------|------|------|
| 核内任务循环 `for s in T.serial(num_local_tasks)` | `T.serial` | **静态**（`num_local_tasks` 编译期常量）| grid-stride 任务映射（`task = s·num_kernels + cid`）|
| 任务守卫 `if task < num_logical` | 条件分支（block 级）| 运行期值 | 仅 block 级分段，非逐元素；尾任务跳过 |
| 选择/打分/归约 | **无显式循环** | — | 全部由 `T.vsort`/`T.reduce_*`/逐元素 v-op 一条原语在 `(BM,E)`/`(BM,K)` 向量上完成 |

**无 `T.Parallel`**（无需显式元素级并行——原语内部已向量化）；**无 `T.Pipelined`**（单行块内无多 stage 依赖，
Developer auto-multi-buffer 自动处理 GM→UB 重叠）；**无 `T.Persistent`**（本设计用显式 persistent 网格 +
`T.serial` 任务循环，与 cumprod 一致）。**无任何逐元素标量循环**（§1.6.2 结论）；行归约 lane 映射 = **水平归约**
（lane→E，`T.reduce_*(dim=1)`），**无垂直扫描逐行串行循环**（§1.6.3 主选，与排序轴尾轴 E 同轴）。

### 6.2 调度伪代码

见 §3.3。核间无依赖、无流水线；每核顺序处理其 grid-stride 任务集；核内单行块「载入 → 排序 → 打分 → 写出」
按 Developer 自动同步顺序执行。

---

## 7. 同步策略

### 7.1 Developer 自动同步（主选）

**Developer 模式自动同步**：单 kernel、单引擎（Vector）、核内顺序数据流（UB 读写 + `T.copy` DMA），
无跨引擎（无 Cube）、无核间通信、无 workspace，故**不显式使用任何同步原语**（无 `sync_block_set/wait`、
无 `pipe_barrier`）。Developer 编译器自动在多 stage（载入 DMA → 排序/打分计算 → 写出 DMA）间插入同步与
auto-multi-buffer。

> **工具链注意**：本设计**不依赖**「向量产出 → 标量消费」的同步（CG-2026-0027 该形态不可靠）——所有中间值
> 都在 UB 以**向量**形式被后续向量 op 消费（`T.vsort`/`T.reduce_*`/v-op 链），标量只用于 block 级
> `task`/`off`/`real` 索引算术（不消费向量产出）。故自动同步足够。

---

### 7.2 CV 融合设计

**n/a**——本算子为**纯 Vector**，不含 Cube/`T.gemm`，无 CV 融合点。选择（排序）+ 打分（逐元素/归约）
全部在 Vector 引擎完成；两输出经 `T.copy` 直达 GM，无跨引擎交接需求。

---

## 8. 验证方案

### 8.1 Golden 函数草案（PyTorch 参考实现）

**以 §0.1 语义为唯一依据**，直接移植参考测试 `examples/TileOPs/tests/ops/test_moe_fused_topk.py` 的 `fused_topk_torch`
（**不复刻** §0.6 的 NPU 算法）：

```python
def golden_fused_topk(gating_output, top_k, scoring_func="softmax", renormalize=False):
    """Source semantics: score full row -> topk (sorted) -> optional renorm."""
    logits = gating_output.to(torch.float32)          # 计算域 fp32（与源一致）
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(scoring_func)
    w, ids = torch.topk(scores, top_k, dim=-1, sorted=False)   # 源用 sorted=False；索引集合即可
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), ids.int()
```

**校验口径**（与参考测试一致，容忍 tie-break 差异）：① `topk_weights` 排序后 `assert_close(rtol=1e-3, atol=1e-3)`；
② `topk_ids` 无重复且每个选中得分 ≥ 第 K 大参考得分 −1e-4；③ 两输出 shape/dtype 正确
（weights fp32、ids int32、`(T,K)`）。

### 8.2 L0 门槛测试计划（必须通过）

| 用例 | shape | dtype | scoring / renorm | 观测点 |
|------|-------|-------|------------------|--------|
| L0-1 smoke softmax | (32,128) K=8 | fp32 | softmax / False | 基本路径（`docs` smoke）|
| L0-2 smoke softmax fp16 | (32,128) K=8 | fp16 | softmax / False | 升 fp32 + 降回 |
| L0-3 smoke softmax bf16 | (32,128) K=8 | bf16 | softmax / False | bf16 升 fp32（vsort 必需）|
| L0-4 sigmoid+renorm bf16 | (32,256) K=8 | bf16 | sigmoid / True | renorm 融合 |
| L0-5 非整除 E=100 | (32,100) K=4 | bf16 | softmax / False | 无 padding 路径 |
| L0-6 非整除 E=33 | (32,33) K=2 | bf16 | sigmoid / False | 无 padding + K=2 |
| L0-7 尾块 T=4096·不整除 | (4096,256) K=8 | bf16 | sigmoid / False | T 尾块 + persistent 分核（deepseek-v3）|
| L0-8 kimi-k2 | (4096,384) K=8 | bf16 | sigmoid / True | 最大 E + renorm |
| L0-9 qwen3 | (4096,128) K=8 | bf16 | softmax / False | softmax 全行分母 |
| L0-10 K=1 | (512,64) K=1 | bf16 | softmax / False | 最小 K |

> 完整分层套件（L1 功能覆盖 / L2 边界 / Boundary 极值）交由 `tilelang-op-develop`；本设计列 L0 门槛即可。
> **Boundary 观测（建议）**：全相等 ties、large_magnitude×50、T 极小（4/5）、E=8。

### 8.3 精度标准

按 harness 容差（`test_moe_fused_topk.py`）：`rtol=atol=1e-3`（权重排序后比较）。fp32 计算域 + 输出 fp32；
bf16 经 fp32 升域（D-3 探针 P1-C/P2 实测 ≤1.8e-7，远优于 1e-3）。

---

## 9. 风险点与注意事项

| 编号 | 风险 | 影响 | 缓解 / 依据 |
|------|------|------|------------|
| R-1 | **correction_bias 变体未实现** | 该变体（Kimi K2/DeepSeekV3 变体）无可运行 kernel | harness workload 不生成 `correction_bias`、参考测试无数值用例 → 首版不做；落地路径见 §3.2 注（`T.vsort` biased + `T.gather` unbiased）；若 Stage 2/3 要求，按路径补 |
| R-2 | **`T.vsort` tie-break 与 torch/源不同** | 等值专家下标可能不同 | 参考测试口径容忍（比较权重 + 无重复 + 得分阈值）；§1.6.1 机器验证含 ties 违反率 0 |
| R-3 | **`T.vsort` 大 BM/大 E 的稳定性未实测** | 探针仅至 BM=16、E=384 | 探针 P1-B 已测 E=384（Kimi）；Stage 3 L0-8 实测；若失败缩 BM |
| R-4 | **`T.vsort`（VSortOp）单条成本未标定** | roofline 发射项可能低估（本算子发射/排序主导）| 属**运行期吞吐画像**，交 Stage 4 实测（§1.6.0 roofline 未实证项）；不影响主选裁决 |
| R-5 | **`T.vsort` 可能需额外 UB scratch 致溢出** | 编译期 UB overflow | §4.5 按 128KB 保守预算；溢出则缩 `BM`（Stage 4 杠杆）|
| R-6 | **fp32 计算域分组差异** | 归约分组 vs 源 shuffle 树 | §1.6.1 机器验证 max_rel_fp32=1.075e-06、违反率 0，≪1e-3 |
| R-7 | **NaN 输入的 vsort 行为未定** | NaN logits 语义未文档化 | 非 manifest 输入域；记入本风险，L2 观测 |
| R-8 | **工具链陷阱复发** | 编译 SIGSEGV / 错值 | 已按 §3.5 规避（不用 `alloc_local`/标量 `if_then_else`/`>>`/标量消费向量产出/`Parallel` 直写 GM）；guard 仅 block 级 |
| R-9 | **非对齐 K 尾列 `T.copy`（K=1，4B）** | 尾轴 <32B | K 固定小；Stage 3 L0-10 验证；必要时 K 切片 + 元素级写 |

---

## 10. 交付清单

| 项 | 路径 | 状态 |
|----|------|------|
| 设计文档 | `examples/fused_topk/_fused_topk_kernel/DESIGN.md` | 本文件（revision v1）|
| 设计文档重跑校验（修订后）| `.agents/tools/design_calc_check.py --design examples/fused_topk/_fused_topk_kernel/DESIGN.md` | **`r3_metrics=pass`**；`ub_budget`/`l0c_budget`/`core_split=skip`（advisory，已人工复核）|
| 等价性机器验证（D-1）| `examples/fused_topk/_fused_topk_kernel/verify_equiv.py` | 已执行（136 例 ALL EQUIV_PASS）|
| 设计期探针 P1（D-3）| `examples/fused_topk/_fused_topk_kernel/history_version/design_probe_vsort_topk.py` | 已执行（设备实测位精确）|
| 设计期探针 P2（D-3）| `examples/fused_topk/_fused_topk_kernel/history_version/design_probe_fused_scoring.py` | 已执行（端到端 ≤1.8e-7）|
| 源算子 | `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`、`.../ops/moe/fused_topk.py` | 解读对象 |
| Stage 0 抽取件 | `examples/TileOPs/tileops/kernels/moe/fused_topk/_fused_top_k_kernels.py` | 已存在 |
| NPU 脚手架 | `examples/TileOPs/tileops/kernels/moe/fused_topk/fused_topk.py`、`examples/TileOPs/tileops/ops/moe/fused_topk.py` | 已存在（Stage 3 重写 kernel 体）|
| 规格 | `examples/TileOPs/tileops/manifest/moe.yaml`（`FusedTopKOp`）| 已存在 |
| 参考实现 / 测试 | `examples/TileOPs/tests/ops/test_moe_fused_topk.py` | 已存在 |
| bench / workload | `examples/TileOPs/benchmarks/ops/bench_moe_fused_topk.py`、`examples/TileOPs/tileops/workloads/moe.py` | 已存在 |
| 同族实现（选择/排序）| `testing/npuir/sort_ops/test_vsort_dev.py`、`testing/npuir/sort_ops/test_vsort_exp.py`、`docs/Tilelang.language/排序操作/T.vsort.md` | 已存在 |
| 同族实现（行归约）| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 已存在 |
| 同族实现（persistent 分核）| `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、`examples/cumprod/_cumulative_kernel/DESIGN.md` | 已存在 |
| 同族实现（索引类/陷阱规避）| `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | 已存在 |
| 原子/规约/广播 API | `docs/Tilelang.language/规约操作/T.reduce_max.md`、`T.reduce_sum.md`、`docs/Tilelang.language/数学操作/T.vsub.md`、`T.vdiv.md`、`docs/Tilelang.language/内存操作/T.copy.md` | 已存在 |
