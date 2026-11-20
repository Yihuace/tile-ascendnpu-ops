# _fused_topk_kernel_cb 算子设计文档（correction-bias 变体）

> 迁移任务（plain，harness，**Developer**）。源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`
> 的 `_fused_topk_kernel` 工厂 **`with_correction_bias=True` 分支**（GPU/CUDA target，源 L96–L207）。
> 本单元目标：`target="npuir"`（Ascend NPU），落盘 `examples/fused_topk_cb/_fused_topk_kernel_cb/`。
> 迁移决策（§0）驱动 §1–§11；算法调研（§1.6.0）先于算法优化（§1.6.1–§1.6.3）。
>
> **等价性机器验证（D-1）**：`examples/fused_topk_cb/_fused_topk_kernel_cb/verify_equiv.py`
> （65 例 + 200 例近 tie 敏感性，**全 EQUIV_PASS**，max_rel_fp32=2.377e-07，id 失配 0）。
> **可行性机器验证（D-3 设计期探针，设备实测 2026-10-16，Ascend910_9362，chip 1）**：
> `history_version/design_probe_cb_path.py`、`history_version/design_probe_ub_sweep.py`、
> `history_version/design_probe_isolation.py`——完整 cb 路径（vsigmoid → vbrc+vadd bias 广播 →
> vsort 降序 → copy si → **full-width `T.gather`** → 切片 → reduce_sum/vdiv）在 E∈{32,64,128,256,384}
> 上编译通过、`topk_ids` 与 torch golden **逐元素一致**、权重误差 ≤2.98e-08。
>
> **本单元的一句话定位**：`_fused_topk_kernel_cb` 是 MoE 的 **sigmoid + per-expert correction bias 路由单元**
> （DeepSeek-V3 / Kimi-K2 风格）——对每 token 的门控 logits 先算 `σ(x)`，用 **`σ(x) + correction_bias[e]` 作
> 选择键**选 top-k 专家，但**输出权重是未加偏置的 `σ(x)`**；`renormalize=True` 时把选中权重归一化到和为 1。
> 输出 `topk_weights [T,K]`（fp32）与 `topk_ids [T,K]`（int32）。**纯 Vector 的选择 + 索引类算子**。

---

## 0. 源算子解读与迁移分析

> 方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问框架 + 耦合性四态判定 + 重设计）。
> 本单元范围 = `_fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func, with_correction_bias=True)`
> 的 **correction-bias 分支**（源 L96–L207）；同族标准变体（`with_correction_bias=False`）已有独立单元
> `examples/fused_topk/_fused_topk_kernel/`（已 `[PRECISION_PASS]`），本设计复用其已验证的 API / 分核 / 循环骨架，
> 但**算法选择路径不同**（见 §0.1「与标准变体的逐项差异」）。
> §0.5/§0.6 的判定输入含 Phase R 调研结论（§1.6.0）。

### 0.1 源算子语义（做什么）

**数学语义**（`scoring_func="sigmoid"`，`with_correction_bias=True`；源 L96–L207）：
对二维门控 logits `x ∈ (T, E)` 与 per-expert 偏置 `b ∈ (E,)` 逐行：

$$
s_{te}=\sigma(x_{te})=\frac{1}{1+e^{-x_{te}}},\qquad
\tilde{s}_{te}=s_{te}+b_e\ \ (\text{selection key}),\qquad
\text{ids}_t=\operatorname{topK}_e \tilde{s}_{te}\ (\text{descending}),\qquad
w_{t,k}=s_{t,\text{ids}_{t,k}}\ \ (\text{unbiased}),
$$

$$
\text{renormalize}=True:\quad w_{t,k}\leftarrow w_{t,k}\Big/\sum_{k'}w_{t,k'}.
$$

- **关键语义（本变体与标准变体的分水岭）**：**选择依据是 `σ(x)+b`，输出权重是 `σ(x)`**——偏置**只**影响
  选择、**不**进入输出权重（源 L114–115 / L143–147 / L197–200 的 `l_best_orig` 就是为此存在）。
- **选择语义 / tie-break**：源 lane-local argmax 用严格 `>`（先出现者胜，等价于 experts 顺序中最小子号），
  跨 lane 归约在 `other_val == l_best_val` 时取 `other_idx < l_best_idx`（源 L173–194）——即**等值时取最小专家号**。
  这是源实现的确定性 tie-break，但 `torch.topk` / `torch.sort` 对 ties 的下标归属**未文档化保证**，
  NPU `T.vsort` 亦然（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3-4「排序稳定性差异」）。tie-break 方案与
  验证方法见 §8.2 与 §9 R-1。
- **规约语义**：本变体**无行内归约**（sigmoid 逐元素；选择是排序/选择而非归约；renorm 是 K 元素求和）。
  这是与标准变体（softmax 需行 max/sum 归约）的又一差异。
- **dtype 语义**：源全程 **fp32 计算域**——op 层 `gating_output.to(torch.float32)` 上送（任意浮点输入先升 fp32），
  `correction_bias` 亦 fp32；kernel 内 logits/σ/biased/权重全 fp32；输出 `topk_weights` fp32、`topk_ids` int32。
- **边界语义**：
  - **E 非 32 整除**：源每 lane 持 `ELEMS_PER_THREAD = ceil(E/32)` 个专家，`expert_idx >= E` 的填充位
    `my_scores`/`my_biased` 均置 `-inf`（源 L137–139），选择时不入选。NPU 侧**不 padding**（排序只作用于真实 E 列）。
  - **T 尾块**：源 `if token_id < num_tokens` 守卫（源 L119）。NPU 侧由 `real = min(BM, T-off)` + 双 slice 处理。
  - **K=1 / K=E**：源 K 轮 argmax 天然覆盖；NPU 侧 vsort 取前 K 天然覆盖。
  - **全相等值 / 近 tie**：源按最小子号选取；NPU 侧 vsort 顺序未保证（§8.2 / §9 R-1）。
  - **±inf/NaN**：源对填充位用 `-inf`；真实 logits 含 `-inf` ⇒ `σ(-inf)=0`，`biased=-inf+b=...`（b 有限时仍可能
    参与选择——这是本变体与标准变体的**真实语义差异**，见 §9 R-2）；NaN 不属本算子输入域（列 §9 风险）。
  - **correction_bias 取值**：源文档未约束取值范围；`b` 有限即可（`biased` 仍可比较；`±inf` 的 b 见 §9 R-2）。
- **softmax + correction_bias 是非法组合**：源 host 侧（`FusedTopKKernel.__init__`，源 L368–372）抛
  `ValueError("with_correction_bias=True requires scoring_func='sigmoid'")`；NPU 侧脚手架同样在
  `FusedTopKOp.__init__`（`examples/TileOPs/tileops/ops/moe/fused_topk.py` L84–85）抛 `ValueError`，
  且 `FusedMoeFwdCbFwdOp` 把 `scoring_func` 默认钉为 `sigmoid`。**校验责任归 host（op 层构造期）**，
  kernel 侧不重复校验（编译期 `scoring_func` 为常量，本变体恒为 sigmoid）。
- **host 侧语义**：op 层做 validate（2D/dtype/top_k≤E/correction_bias shape=`(E,)` 且 dtype=fp32 且同 device）
  + `_get_kernel`（按 shape 特化缓存）+ `kernel(gating, correction_bias)`；`FusedTopKKernel.forward` 做
  `gating_output.to(torch.float32)` / `correction_bias.to(torch.float32)` 上送；源 `renormalize=True` 时
  在 **host 侧 PyTorch** 做 `w / w.sum(-1, keepdim=True)`（一次额外算子）。**renormalize 与 fp32 上送均视为实现手段**
  （可 kernel 内化）。

**与标准变体的逐项差异（迁移解读重点）**：

| 维度 | 标准变体（`with_correction_bias=False`） | **本变体（`with_correction_bias=True`）** |
|------|------------------------------------------|------------------------------------------|
| 选择键 | `σ(x)` 或 `softmax(x)`（x 的**严格单调**函数） | **`σ(x) + b[e]`**（**非** x 的单调函数：b 逐专家不同） |
| 可选「在原始 logits 上选 top-k」优化 | ✅ 成立（单调保序，sibling §1.6.1-E1） | ❌ **不成立**——必须显式计算 `σ(x)+b` 全行后排序 |
| 打分物化范围 | 只需对选中 K 个元素打分（sibling E1） | **必须对全行 E 个元素算 σ 并加 b**（选择键需全行） |
| 输出权重来源 | 选中 K 个元素的 `σ`/`softmax` | **选中 K 个位置的 `σ(x)`（未加偏置）** |
| 归约语义 | softmax 需行 max/sum 归约 | 无行归约 |
| 选择→输出的衔接 | 值即选择键（排序输出直接可用） | **排序输出是 biased；输出需 unbiased ⇒ 需按排序得到的下标取回 `σ(x)`** |
| softmax 合法性 | 合法 | **非法组合**（host 层须报错） |
| renormalize | 可选（softmax+renorm 全行分母抵消） | 可选，但**无抵消**（σ 无公共全行分母） |

**语义保持基线**：§8 的 golden 以本节语义为唯一依据；参考实现 = 源语义的 torch 复刻
`biased=σ(x)+b → topk(sorted) → w=σ(x)[ids] → 可选 renorm`，**不得复刻** §0.6 的 NPU 算法（vsort + gather）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `gating_output` | `(T, E)` | float16 / bfloat16 / float32 | 门控 logits（2D，行主序连续）|
| `correction_bias` | `(E,)` | float32 | per-expert 选择偏置（`with_correction_bias=True` 时必需）|
| `topk_weights` | `(T, K)` | float32 | **输出**：选中专家的**未加偏置** σ 权重（renorm 时已归一）|
| `topk_ids` | `(T, K)` | int32 | **输出**：选中专家的原始下标（按 biased 降序）|

工厂契约：`_fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func, with_correction_bias)`
（`functools.lru_cache(maxsize=64)` 特化）→ 内层 `@tilelang.jit` 的 `_func(TOKENS_PER_BLOCK)` →
`main(gating_output, correction_bias, topk_weights, topk_ids)`。源 `TOKENS_PER_BLOCK=16`
（`T.Kernel(ceil(T/16), threads=16*32=512)`，每 warp 一 token）。NPU 侧保留工厂键，追加
`renormalize` 与 `in_dtype` 两个 NPU 侧扩展参数（与 sibling 一致）。

### 0.3 实现算法解读（怎么算）

**源 kernel 结构**（源 L96–L207）：`@tilelang.jit(out_idx=[])` 工厂；`T.Kernel(ceil(T/TPB), threads=TPB*32)` 一维 grid；
`tx=get_thread_binding()`，`warp_id=tx//32`，`lane_id=tx%32`，`token_id=block_id*TPB+warp_id`——**每 warp 独立处理一个 token**；
`ELEMS_PER_THREAD=ceil(E/32)`。**双缓冲**：`my_scores[ELEMS]`（未加偏置 σ，供输出）与
`my_biased[ELEMS]`（σ+b，供选择）；K 轮 argmax 在 `my_biased` 上比较，同时把 `my_scores` 值随
`l_best_orig` 经 `shfl_xor` 一起搬运（源 L163–195），使 lane 0 握有正确的 unbiased 权重。

**计算步骤分解**（源码每条计算语句均归入下表）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|------|------|------|------|-------------|
| 1 | 载入：`for j: expert_idx=j*32+lane_id; my_scores[j]=x[tid,expert_idx] if expert_idx<E else -inf`（源 L122–127）| x | `my_scores[ELEMS]`（寄存器）| 读 logits（+ 非整除 `-inf` 填充）|
| 2 | σ + bias：`sig=1/(1+exp(-my_scores[j])); my_scores[j]=sig; my_biased[j]=sig+correction_bias[expert_idx]`；越界位两数组均 `-inf`（源 L130–139）| my_scores, bias | `my_scores`（=σ）、`my_biased`（=σ+b）| $s=\sigma(x)$；$\tilde{s}=s+b_e$ |
| 3a | K 轮 lane-local argmax：`l_best_val=-inf; for j: if my_biased[j]>l_best_val: 更新 val/orig/idx`（源 L152–161）| my_biased, my_scores | `l_best_val/orig/idx` | 行内第 k 大 biased |
| 3b | warp 归约：5 轮 `other_{val,orig,idx}=shfl_xor(...,mask)`；`l_best_orig` 随 `l_best_val/l_best_idx` 同步更新（等值取小 `other_idx`）（源 L167–195）| lane 局部 best | warp 全体 (val, orig, idx) | 行内 argmax（tie-break 最小子号）|
| 3c | `if lane_id==0: topk_weights[tid,k]=l_best_orig; topk_ids[tid,k]=l_best_idx`（源 L198–200）| best | 两张输出表 | $w_{t,k}=s_{t,\text{ids}_{t,k}}$ |
| 3d | 掩码：`for j: if j*32+lane_id==l_best_idx: my_scores[j]=-inf; my_biased[j]=-inf`（源 L203–206）| l_best_idx | 两寄存器数组 | 下一轮排除已选专家 |
| 4 | host 侧：`if renormalize: topk_weights /= topk_weights.sum(-1,keepdim=True)`（源 L439–440）| 输出 | 归一化输出 | renormalize |

**数据流图**（源 GPU 视角，覆盖全部 buffer）：

```
GM[gating_output(T,E)] --每 lane 标量循环读 ELEMS 个--> REG[my_scores(ELEMS) fp32]  (越界填 -inf)
GM[correction_bias(E)] --按 expert_idx 标量读--> REG 临时
   REG[my_scores] --σ 逐元素--> REG[my_scores]=σ(x)（供输出）
                              └--> REG[my_biased]=σ(x)+b（供选择）
   K 轮: REG[my_biased] lane-local argmax → shfl_xor (val,orig,idx) 树形归约
          → warp best → lane0 写 GM[topk_weights/topk_ids]
          └─ REG[my_scores] 与 REG[my_biased] 同时掩码已选为 -inf
   （renormalize 在 host 侧对 topk_weights 再做一次逐行除法）
```

**循环与并行结构**：grid = `ceil(T/16)` block；每 block 512 线程 = 16 warp；**warp 内 32 lane 并行**，
lane 内 `T.serial(ELEMS)` 顺序扫描；K 轮 `T.serial(top_k)`。无 `T.Parallel` / `T.Pipelined`。
**冗余特征**：每 warp 独立、block 内 16 warp 无共享。

**host 侧逻辑**：
1. op 层 `FusedTopKOp.forward`：validate（2D/dtype/top_k≤E；Cb 时 `correction_bias` 必需、shape `(E,)`、fp32、同 device）
   → `_get_kernel(T,E,K,dev)`（按 shape 特化缓存）→ `kernel(gating_output, correction_bias)`；构造期校验
   `with_correction_bias ⇒ scoring_func=="sigmoid"`（非法组合抛 `ValueError`）。
2. `FusedTopKKernel.forward`：`logits_f32=gating_output.to(torch.float32)`、`bias_f32=correction_bias.to(torch.float32)`
   → 分配两张输出 → `fn(logits_f32, bias_f32, topk_weights, topk_ids)` →
   `if renormalize: topk_weights /= topk_weights.sum(-1,keepdim=True)`。
3. `FusedMoe.forward`（集成 orchestrator，本单元不改）调 `self._fused_topk(gating_output, correction_bias)`，
   Cb 变体随后乘 `routed_scaling_factor`。

**编译配置**：`@tilelang.jit(out_idx=[])`（无 `pass_configs`、无 `compile_flags`）。`supported_archs=[80,86,89,90]`
（H100/A100/…），NPU 侧脚手架已改 `None`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **one warp per token（thread 级任务划分）** | 每 token 独立、无跨 warp 通信 | 每 warp 32 lane 认领一 token | CUDA warp / thread 模型 | 硬件强相关（NPU 无 warp/thread-per-token）→ 重设计 |
| O2 | **寄存器驻留全部专家**（`T.alloc_local`） | 免 SMEM、免重复访存 | 每 lane 持 `ceil(E/32)` 个 logits/σ | 大寄存器堆 | 硬件强相关（NPU 用 UB 驻留）→ 等价替换 |
| O3 | **warp `shfl_xor` 树形归约 + 双数组同步搬运** | **零 `__syncthreads`**；让 lane0 同时握有 biased 与 unbiased | 5 轮 xor-shuffle 全归约，`l_best_orig` 随 `l_best_val/l_best_idx` 一起搬 | warp shuffle 指令 | 硬件强相关（NPU 无 shuffle）→ 重设计 |
| O4 | **双寄存器数组 `my_scores`/`my_biased`** | 选择用 biased、输出用 unbiased（**本变体核心语义**）| 两份 (ELEMS) 寄存器，同时掩码 | 寄存器容量 | **算法层可移植、且必须保留**；NPU 侧以「排序 biased → 用下标取回 σ」承接 → 重设计 |
| O5 | **K 轮 masked argmax 复用寄存器** | 免排序、`O(K·E)` 比较 | 每轮选最大后把该专家掩码为 `-inf`，原位复用 | 寄存器 + 掩码 | 算法层可移植，但**索引恢复**依赖 shuffle → 重设计 |
| O6 | **fp32 计算域** | 与 torch 一致、数值稳定 | 全链 fp32；host 上升 fp32 | 无 | 可移植（保留）|
| O7 | **非整除 E 的 `-inf` 填充** | 32 lane 对齐，填充位不入选 | 越界专家两数组均置 `-inf` | lane 宽度 32 | 可移植（NPU 免 padding）|
| O8 | **`lru_cache` 工厂特化** | 消重复编译 | `(T,E,K,scoring_func,bias)` 键 | 无 | 可移植（保留）|
| O9 | **输出为独立 GM 张量** | 输出直达 GM | 两输出由 op/kernel 分配 | 无 | 可移植（保留）|
| O10 | **renormalize 在 host 侧 PyTorch** | 实现简单 | kernel 出未归一权重后逐行除 | 无 | 可移植但引入额外算子 → NPU 融合进 kernel（重设计）|

> 识别完整性：以上覆盖 correction-bias 分支的全部优化性结构（步骤表已把每条计算语句归入步骤 1–4）。
> 无 SMEM tiling、无 mma/Cube、无异步流水、无 swizzle、无 persistent（源为超发 grid）、无量化/dequant。
> **识别不出来 ≠ 不存在**：O1–O10 逐项在 §0.5 判定。
> **量化/反量化路径（用户告诫 #4）**：本算子**不涉及** dequant——源对任意浮点输入统一 `to(torch.float32)`，
> 无 scale/zero-point、无低比特路径。故无 dequant 代价需量化（n/a，依据：源 L426/L434 与
> `FusedTopKOp.forward` 仅做 dtype 上送）。

### 0.5 硬件耦合性分析与 NPU 适配决策

> 判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？判定输入含 Phase R 调研结论（§1.6.0：
> 选定「**在 `biased=σ(x)+b` 上 `T.vsort` 降序取前 K** 得下标 → **用 full-width `T.gather` 按该置换取回
> 已算好的 `σ(x)`** → K 切片 renorm」，源 warp 双数组 K 轮 argmax 被替换）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义层（`biased` 选择 / 未加偏置 `σ` 输出 / K 切片 renorm / fp32 域 / tie-break 容差内）| 语义 | 无 | — | **保留** | 语义层无条件保留；golden 以此为基线（§8）|
| O1 one warp per token（thread 级任务划分）| 优化 | CUDA warp/thread | **无**（本项目一维 Kernel，无 thread/warp）| **重新设计** | §0.6-R1：一维 persistent，token 行块 `(BM,E)` 分核 + 核内 `T.serial` grid-stride |
| O2 寄存器驻留专家 | 优化 | 大寄存器堆 / `alloc_local` | 有（UB 驻留，但 `alloc_local + T.serial` 在本工具链 SIGSEGV，CG-2026-0028b）| **等价替换** | §0.6-R2：`T.alloc_shared`（UB）驻留 `(BM,E)` fp32；不用 `alloc_local` |
| O3 warp `shfl_xor` 树形归约 | 优化 | warp shuffle | **无**（映射表 `migration-analysis.md` §5.3 判「重新设计」）| **重新设计** | §0.6-R3：选择改 `T.vsort` 降序；置换由 `T.gather` 承接（无归约语义需保留——本变体无行归约）|
| O4 双寄存器数组（**选择用 biased / 输出用 unbiased**）| 优化/语义 | 寄存器 + shuffle 搬运 | **有**（两数组可用两个 UB buffer；置换回取用 `T.gather`）| **重新设计（意图必保）** | §0.6-R4：源「`l_best_orig` 随选择键搬运」的意图 = 「**选择键与输出值解耦**」；NPU 侧用 `T.gather(σ, si)` 一次取回全部 K 个 unbiased 值（比逐轮搬运更直接）。**O4 的语义不得丢弃**（这是本变体的定义）|
| O5 K 轮 masked argmax + 掩码复用 | 算法/优化 | 无（算法可移植），但**跨 lane (val,idx) 归约**依赖 shuffle | 算法等价，实现不可用 | **重新设计** | §0.6-R3：改 `T.vsort` 降序取前 K（一条 `hivm::VSortOp` 直接产出值 + int32 下标）；承接 O5「免排序」意图的**性能面**改为「一次硬件排序替代 K 轮归约」|
| O6 fp32 计算域 | 算法/优化 | 无 | 有（Vector fp32）| **保留** | 全链 fp32；`T.vsort` 仅 fp16/fp32（bf16 须升 fp32，§1.6.1-E4）；σ 由 `T.vsigmoid` 承接 |
| O7 非整除 E 的 `-inf` 填充 | 优化 | lane 宽度 32 | **无此需求**（排序/打分作用于真实 E 列）| **等价替换** | §0.6-R5：直接对原始 E 列 `T.vsort`/`T.gather`，无 padding、无 `-inf` 哨兵（等价，§1.6.1-E3 机器验证含非整除 E）|
| O8 `lru_cache` 工厂特化 | 优化 | 无 | — | **保留** | 工厂层与硬件无关（脚手架 `_get_kernel` 已保留）|
| O9 输出独立 GM 张量 | 优化 | 无 | — | **保留** | kernel 写 `topk_weights`/`topk_ids` 两张 GM 表（`out_idx=[2,3]` 或显式分配）|
| O10 host 侧 renormalize | 优化 | 无 | 有（GPU 也可融合，源未做）| **重新设计** | §0.6-R6：renorm 融合进 kernel 的 K 元素切片（`T.reduce_sum` + `T.vdiv`），消除第二次 host 算子与 GM 往返 |

> **四态处置统计**：保留 5 项（语义 / O6 fp32 / O8 lru_cache / O9 输出表 / 语义层校验责任）｜
> 等价替换 2 项（O2 寄存器→UB、O7 填充消除）｜重新设计 4 项（O1〔R1〕、O3〔R3〕、O4〔R4〕、O5〔R3〕、O10〔R6〕
> — 其中 O3/O5 同由 R3 承接）｜**舍弃 0 项**。
> **无「舍弃后性能意图丢失」项**：O1/O3/O5 的低延迟意图由 R1/R3 承接，**O4 的双数组语义由 R4 显式承接**，
> O10 的归一化语义由 R6 融合承接。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（对照 Phase R 调研结论 §1.6.0：调研在「可实现且亲和」
的算法族内择优——选定 **C1：`biased=σ(x)+b` 上 `T.vsort` 降序取前 K 得下标 → full-width `T.gather` 取回已算好的
`σ(x)` → K 切片打分/renorm**；源 warp 双数组 K 轮 argmax 因 NPU 无 shuffle 且索引恢复无干净原语被替换）。

**重设计项 R1：并行结构（源：one warp per token + 超发 grid）→ 一维 persistent 行块分核**

- **源方案**：`T.Kernel(ceil(T/16), threads=512)`；每 warp 认领一个 token，lane 持 `ceil(E/32)` 个专家。
- **NPU 新算法**：一维 persistent——`T.Kernel(num_kernels, is_npu=True) as (cid,_)`，
  `num_kernels = min(num_logical, 40)`（纯 Vector 物理核数，§5.5），`num_local_tasks = ceil(num_logical/40)`（静态），
  核内 `for s in T.serial(num_local_tasks): task = s*num_kernels + cid; if task < num_logical:` 处理一个
  `(BM,E)` 行块（`real = T.min(BM, T - task*BM)`，双 slice `T.copy`）。
- **语义保持论证**：token 行之间**完全独立**（逐行 σ + 逐行选择，无跨行依赖），行块划分只改变调度、
  不改变任何 token 的结果；尾块 `real` 只裁掉不存在的行。**脚本先例**：`examples/cumprod/_cumulative_kernel`
  的 persistent + 静态 `T.serial` + 动态 guard 结构（已实现并验证）。

**重设计项 R2：寄存器驻留 → UB 行块驻留**

- **源方案**：`T.alloc_local` 每 lane 持 `ceil(E/32)` 个元素。
- **NPU 新算法**：`T.alloc_shared((BM,E), "float32")` 驻留整行块（排序/取值需整 E 行在 UB）。
- **语义保持论证**：驻留介质改变不改变数值；`T.vsort`/`T.gather` 要求整行驻留。**不用 `T.alloc_local`**
  （CG-2026-0028b：`alloc_local + T.serial` SIGSEGV）。

**重设计项 R3：K 轮 masked argmax（+shuffle 归约）→ `T.vsort` 降序取前 K**

- **源方案**：K 次「lane-local argmax over `my_biased` + 5 轮 (val,orig,idx) `shfl_xor` 归约」，每轮把选中专家掩码为 `-inf`。
- **NPU 新算法**：对 `biased` 行块一次 `T.vsort(biased, sv, si, descending=True, sort_axis=-1)`
  （`hivm::VSortOp`，`docs/Tilelang.language/排序操作/T.vsort.md`），`sv`=降序 biased 值、`si`=对应 int32 原始下标；
  取前 K 列 `si[:,0:K]` 即 `topk_ids`。**一条原语替代 K 轮归约**。
- **语义保持论证**：**选择等价**——选出的是 `biased` 最大的 K 个专家（与源一致；**本变体不能用 siblings 的
  「原始 logits 上选」优化**，因 biased 非 x 单调）。tie-break 差异在 §0.1/§8.2/§9 R-1 论证为容差内可接受。
  **意图承接**：源 O5「免排序、O(K·E) 比较」的性能意图由「一次硬件排序原语」承接——本工具链上 K 轮 argmax 需
  argmax 索引恢复（`T.reduce_max` 无下标输出），代价与不可靠性更高（CG-2026-0027/0028）。

**重设计项 R4：双数组「选择键 vs 输出值」解耦（本变体核心语义的 NPU 承接）**

- **源方案**：两份寄存器数组 `my_scores`（=σ，输出）与 `my_biased`（=σ+b，选择）；K 轮 argmax 比较 `my_biased`，
  同时把未加偏置的 `my_scores` 值随 `l_best_orig` 经 `shfl_xor` 一起搬运，lane 0 写出 `l_best_orig`。
- **NPU 新算法**：σ **只算一次**并整行驻留（`xf`）；`biased = σ + b`（`T.vbrc` 广播偏置 + `T.vadd`）整行驻留（`biased`）；
  `T.vsort(biased, sv, si, descending=True)` 得选择置换 `si`；**`T.gather(xf, sv, si)`** 用 `si`（runtime int32 下标）
  一次取回**被选择置换重排的 σ**，切片 `sv[:,0:K]` 即 `topk_weights`（未加偏置）。
- **语义保持论证**：源逐轮把 unbiased 值沿选择链搬运 ≡ NPU 用选择置换整体重排 σ（**逐元素同一映射**
  `w_k = σ(x)[si_k]`）；等价性由 §1.6.1-E1 机器验证（`gather-reorder σ` vs `gather-recompute σ` 差 0）
  与 §1.6.1-D-3 实机探针（ids 逐元素一致、权重误差 ≤2.98e-08）双证。
- **可行性依据**：`T.gather` 支持**运行时 int32 张量下标**（`testing/npuir/indexing_ops/test_slice_gather_dev.py`
  用 GM 载入的 int32 `index_VEC` 做 gather；`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1/§2.4），
  且**接受 `T.vsort` 产生的下标张量**（D-3 探针 P1 实机通过）。**full-width（`src`/`indices`/`dst` 同形 `(BM,E)`）**
  是探针验证的唯一形态（`docs` 示例与 `testing/` 用例均要求三者同形）。

**重设计项 R5：非整除 E 的 `-inf` 填充消除**

- **源方案**：每 lane 持 `ceil(E/32)` 个专家，越界位（两数组）填 `-inf`。
- **NPU 新算法**：σ / 广播偏置 / 排序 / gather 只作用于**真实 E 列**（`(BM,E)` 满行驻留 UB），无需 padding。
- **语义保持论证**：源填 `-inf` 的目的是让填充位不入选；NPU 侧根本不存在填充列，故输出等价
  （§1.6.1-E3 机器验证含 E=4/33/100 等用例）。

**重设计项 R6：renormalize 内移 kernel**

- **源方案**：kernel 出未归一权重，host 侧 `w / w.sum(-1,keepdim=True)`（第二算子）。
- **NPU 新算法**：融合进同一 kernel——取 `vk=σ(x)[si_K]` 后 `T.reduce_sum(vk, sr, dim=1)` + `T.vdiv(vk, sr, vk)`
  （`[BM,K]/[BM,1]` 行广播）得归一权重。
- **语义保持论证**：同一除法，仅执行位置不同（数学恒等）；§1.6.1-E2 机器验证 0 误差。
  **注意**：与 siblings 的 softmax+renorm 不同，**本变体无「全行分母抵消」**（σ 无公共全行分母），
  故 renorm 恒需 K 元素 `reduce_sum`。

**语义保持 / 边界核对汇总**：数学等价项（E1 gather-reorder σ、E2 renorm 融合、E3 padding 消除）均在
§1.6.1 给出四要素论证 + `verify_equiv.py` 结果；边界（T 尾块、K=1/K=E、全相等、非整除 E、±inf、近 tie）
在 §0.1 与 §1.6.1 / §9 逐项核对。

### 0.7 标杆实现

- **源算子**：`/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`（`_fused_topk_kernel` 的 Cb 分支，
  §0.1–§0.4 解读对象）；op 层 `/tmp/opencode/TileOPs-ref/tileops/ops/moe/fused_topk.py`（`FusedTopKOp`）。
- **NPU 侧集成目标（本 Stage 不改）**：`examples/TileOPs/tileops/kernels/moe/fused_topk/fused_topk.py`
  （`FusedTopKKernel`；当前 `with_correction_bias=True` 处为脚手架，`_fused_topk_kernel` 由 Stage 0 抽取件导入）、
  `examples/TileOPs/tileops/ops/moe/fused_topk.py`（`FusedTopKOp`，已完整 validate）、
  `examples/TileOPs/tileops/ops/moe/fused_moe.py`（`FusedMoe.forward` 调 `self._fused_topk(gating_output, correction_bias)`）。
  **集成由 conductor 后续做**；本单元只产出设计与（Stage 3）实现。
- **规格 / 测试 / bench**：`examples/TileOPs/tileops/manifest/moe.yaml`（`FusedTopKOp` 段 + `FusedMoeFwdCbFwdOp` 段）、
  `examples/TileOPs/tests/ops/test_moe_fused_moe_cb.py`（Kimi K2 Cb 端到端，以 `FusedTopKOp` 作路由参考）、
  `examples/TileOPs/benchmarks/ops/bench_moe_fused_moe_cb.py`。
- **同族已验证参考（复用 API / 循环 / 分核骨架）**：
  `examples/fused_topk/_fused_topk_kernel/_fused_topk_kernel.py`（标准变体，`[PRECISION_PASS]`）与
  其 `DESIGN.md`（§9 R-1 给出本变体落地路径 = 排序 biased → `T.gather` 取回 unbiased σ）；
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`
  （Developer 行归约 + `T.reduce_*` + slice `T.copy` 实证形态）；
  `examples/cumprod/_cumulative_kernel/DESIGN.md`（persistent 分核 / UB 预算 / D-3 探针方法论）；
  `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`（索引类算子 + 工具链陷阱规避清单）；
  `testing/npuir/indexing_ops/test_slice_gather_dev.py` + `test_slice_gather_exp.py`（`T.gather` Developer/Expert 通过用例）；
  `testing/npuir/sort_ops/test_vsort_dev.py`（`T.vsort` Developer 通过用例）。
- golden 以 §0.1 语义为唯一依据实现（**不复刻** §0.6 的 NPU 算法）。

---
## 1. 概述

### 1.1 算子名称

`_fused_topk_kernel_cb`（承载 `FusedTopKOp(with_correction_bias=True)` 的 NPU MoE top-k 路由 kernel；
`target="npuir"`，Developer 模式）。集成身份：`FusedMoeFwdCbFwdOp`（Kimi-K2 风格）的路由子算子。

### 1.2 功能描述

对 `(T,E)` 门控 logits 逐行算 `σ(x)`，以 **`σ(x)+correction_bias[e]`** 为选择键选取最高的 K 个专家，
输出**未加偏置的 `σ(x)`** 作为权重（`topk_weights [T,K]` fp32）与专家下标（`topk_ids [T,K]` int32）；
`renormalize=True` 时把选中 K 个权重归一化到和为 1。**迁移决策后的 NPU 形态**（§0.6）：
一维 persistent Vector kernel，按 token 行块 `(BM,E)` 分核驻留 UB——`T.vsigmoid` 全行算 σ（只算一次），
`T.vbrc`+`T.vadd` 广播偏置得 `biased`，`T.vsort` 在 `biased` 上**降序**取前 K 得下标，
`T.gather` 用该下标置换取回 σ，K 切片 `reduce_sum`/`vdiv` 融合 renorm；非整除 T 由行尾块 slice 处理、
非整除 E 无需 padding。

### 1.3 数学公式

（同 §0.1，NPU 侧形式不变）

$$
s_{te}=\sigma(x_{te}),\quad \tilde{s}_{te}=s_{te}+b_e,\quad
\text{ids}_t=\operatorname{topK}^{\downarrow}_e \tilde{s}_{te},\quad
w_{t,k}=s_{t,\text{ids}_{t,k}},\quad
w_{t,k}\leftarrow w_{t,k}\Big/\textstyle\sum_{k'}w_{t,k'}\ (\text{renormalize})
$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解；**本变体不得用「原始 logits 上选 top-k」**，因 `biased` 非 x 单调）：

$$
s=\sigma(x)\ \text{(full-row, once)},\quad \tilde{s}=s+b,\quad
\text{ids}=\pi^{\downarrow}_{K}(\tilde{s}),\quad w_k=s_{\text{ids}_k}\ (=\operatorname{gather}(s,\text{ids})),
\quad w\leftarrow w/\textstyle\sum_{k'}w_{k'}
$$

其中 $\pi^{\downarrow}_{K}$ = 降序排序置换取前 K（`T.vsort(descending=True)` 输出的前 K 列）。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

纯 Vector 的 **persistent 行块 + 选择键硬件排序 + 置换回取 σ** 算法（行间独立；与源结构差异逐条注明）：

1. **载入**：`T.copy(gating[off:off+real, 0:E], xf[0:real, 0:E])`——GM 行块 → UB，**跨 dtype 自动升 fp32**
   （bf16 必需：`T.vsort` 不支持 bf16，见 `docs/Tilelang.language/排序操作/T.vsort.md` §2.2.1；实机探针已证
   bf16→fp32 自动 `T.copy` 可用）。取代源 O1 的 per-lane 标量载入〔§0.6-R1/R2〕。
2. **打分（全行，仅一次）**：`T.vsigmoid(xf, xf)`——`xf` = 未加偏置 σ(x) 整行〔§0.6-R4〕。
3. **偏置广播 + 选择键**：`T.copy(bias[0:E], bb[0:1,0:E])`（`bb` 形状 `(1,E)`）；
   `T.vbrc(bb, biased)`（`(1,E)→(BM,E)` 广播，`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2）；
   `T.vadd(xf, biased, biased)`——`biased = σ(x) + b`〔§0.6-R4〕。
4. **选择**：`T.vsort(biased, sv, si, descending=True, sort_axis=-1)`——对 **biased** 降序排序得值 `sv` 与
   int32 原始专家下标 `si`；`T.copy(si[0:real,0:K], ik[0:real,0:K])` 即 `topk_ids`〔§0.6-R3〕。
5. **置换回取 σ（本变体核心）**：`T.gather(xf, sv, si)`（**full-width**：`src`/`indices`/`dst` 同形 `(BM,E)`）——
   `sv[k] = xf[si[k]] = σ(x)[si[k]]`；`T.copy(sv[0:real,0:K], vk[0:real,0:K])` 即未加偏置权重〔§0.6-R4〕。
6. **renormalize（可选，融合）**：`T.reduce_sum(vk, sr, dim=1, clear=True)` + `T.vdiv(vk, sr, vk)`
   （`[BM,K]/[BM,1]` 行广播）〔§0.6-R6〕。
7. **写出**：`T.copy(vk[0:real,0:K], topk_weights[off:off+real,0:K])`、
   `T.copy(ik[0:real,0:K], topk_ids[off:off+real,0:K])`。
8. **调度**：一维 persistent，`num_kernels = min(num_logical, 40)`，核内 `T.serial(num_local_tasks)` grid-stride
   + `if task < num_logical` 守卫〔§0.6-R1，§5.5〕。

**与源算法差异**：① one-warp-per-token → 行块 persistent 分核（R1）；② 寄存器驻留 → UB 行块（R2）；
③ K 轮双数组 masked argmax + shuffle → `T.vsort` 降序 + `T.gather` 置换回取（R3/R4）；④ 非整除 E padding →
无 padding（R5）；⑤ host 侧 renorm → kernel 内融合（R6）。**保留**：fp32 计算域（O6）、`lru_cache`（O8）、
输出独立 GM 表（O9）。

### 1.5 数据流图

```
[单 kernel · Vector · persistent num_kernels = min(⌈T/BM⌉, 40)]
GM[gating_output(T,E) fp16/bf16/fp32] --T.copy(行块 slice, 自动升 fp32)--> UB[xf (BM,E) fp32]
    --T.vsigmoid--> UB[xf = σ(x)]                              （未加偏置，供输出）
GM[correction_bias(E) fp32] --T.copy--> UB[bb (1,E) fp32]
    --T.vbrc--> UB[biased (BM,E)] --T.vadd(xf, biased)--> UB[biased = σ(x)+b]   （选择键）
    --T.vsort(descending)--> UB[sv (BM,E) fp32=降序 biased] + UB[si (BM,E) int32=原始下标]
    --T.copy 列 slice (si[:,0:K])--> UB[ik (BM,K) int32]        （= topk_ids）
    --T.gather(xf, sv, si)--> UB[sv = σ(x) 按 si 置换重排]      （full-width）
    --T.copy 列 slice (sv[:,0:K])--> UB[vk (BM,K) fp32]         （= 未加偏置权重）
    --(renorm) T.reduce_sum(vk,sr) --T.vdiv(vk,sr,vk)--> UB[vk]
    --T.copy--> GM[topk_weights(T,K) fp32]
    --T.copy--> GM[topk_ids(T,K) int32]     （= UB[ik]）
无 L1/L0 参与、无 Cube、无 workspace、无核间同步；gating 读一遍、bias 读一遍、两输出各写一遍。
```

### 1.6 算法调研与优化分析

> **设计第一优先级**：先**调研**——对同一数学语义（`biased=σ(x)+b` 逐行选 top-k + 输出未加偏置 σ）的算法族
> 回答调研四问（§1.6.0；迁移任务在源算子解读〔§0〕后、耦合性判定与重设计〔§0.5/§0.6〕前完成，调研结论已作为
> M1 判定输入）；再在保证数学等价的前提下优化公式（§1.6.1，**含机器验证**）；再把保留下来的计算全部交给向量单元
> （§1.6.2）；最后回答「在哪个轴上向量化 / 如何布局」（§1.6.3，阻塞级）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。
>
> **Ratio 口径说明（用户告诫 #1）**：本仓报告的 Ratio 分母是 **matrix peak**（Cube 算力峰值）。本算子**零矩阵
> （Cube）工作量**——主体是逐元素 σ + 广播加 + 排序/选择 + 索引置换，全部落在 Vector 单元，故 **Ratio 天花板受分母
> 口径限制（结构性低值，非缺陷）**。评估本算子应用 **Mem/BW 利用率**作辅助口径（§1.6.0 roofline 的流量项与实测
> MTE2 busy 占比），不用 Ratio 作为调优目标。这一点写死，避免 Stage 4 误判「Ratio 低 = 未优化」。

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**「per-expert 偏置修正 + σ 打分 + top-k 选择 + 未加偏置权重输出」**这一数学语义的算法族。
> 源算法（warp 双数组 K 轮 masked argmax + shuffle 搬运）只是基线候选之一。
> 调研深度：**完整调研**（多步 + 选择 + 索引置换类 → R1 候选表 + R3 四口径 + R4 逐候选评估）。
> 信息源：skill 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（**命中行 `ALG-topk`**：
> 「R1：部分选择 vs 全排序；分块 top-k + 归并。R2：分块 + 归并。R3：O(N·k) vs O(N log N)。R4：归并链的核内结构
> 与缓冲。」）；`kb_search "fused_topk correction bias sigmoid topk 选择 MoE 路由"`（命中 `ALG-topk`、
> `VP-2026-0051/0137`〔queue pending，仅作线索〕、`CG-2026-0033`/`CG-2026-0021`〔open，仅作线索〕）；
> 本仓 `examples/`（sibling `_fused_topk_kernel` 标准变体、`sort_ops/test_vsort_dev.py`、`logsumexp`、`cumprod`、
> `indexing_ops/test_slice_gather_*`）；`docs/Tilelang.language/排序操作/`（仅 `T.vsort.md`+`T.vcumsum.md`）、
> `索引与元素操作/T.gather.md`、`数学操作/T.sigmoid.md`、`shape操作/T.vbrc.md`、`规约操作/T.reduce_sum.md`；
> 源算子 §0.3/§0.4；**设计期探针**（D-3，实机 NPU：完整 cb 路径 + UB 容量包线，见下）。
> **互联网检索**：`ALG-topk` 命中行 + 经典算法知识已覆盖该成熟族（部分选择 vs 全排序、分块 top-k + 归并、
> 稳定排序 tie-break），**未触发联网**（本地源充分）。

**R1 等价化简公式候选**（基线候选在表内；正式论证见 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| C0 | **基线（源算法）** | warp per token；双寄存器数组 `my_scores`(σ)/`my_biased`(σ+b)；K 轮 masked argmax + `shfl_xor` 三值归约 | — | — | ✅ 基线 |
| C1 | **`T.vsort(biased)` 降序取前 K + `T.gather` 置换回取 σ**（**选定**）| `biased=σ(x)+b`；`T.vsort(biased, sv, si, desc)`；`ids=si[:,0:K]`；`w=T.gather(σ, si)[:,0:K]` | **数学恒等**（选择集与源一致；输出权重 = σ 在选择置换下的重排，源以 `l_best_orig` 逐轮搬运实现同一映射，§1.6.1-E1）| 1 条 `VSortOp` 替代 K 轮 argmax+shuffle；σ 只算一次；1 条 `VGatherOp` 取回全部 K 个 unbiased 值 | ✅ **选定** |
| C2 | **`T.vsort(biased)` + gather 偏置再相减**（`w = sv_K - b[si_K]`）| 与 C1 同排序；输出改为 `biased_sorted_k − b[si_k]` | 数学恒等（`σ = (σ+b) − b`）| **需再 gather 一次**（取 `b[si_k]`，且 b 需先广播/展开为行结构）+ 一次 `vsub` → 比 C1 多 2 op/tile | ✅ 对比（弃选）|
| C3 | **两路独立排序**（sort σ 供输出 + sort biased 供 ids）| 两次 `T.vsort` | ❌ **不等价**——两置换不同，权重与 id **对不上**（σ 的 top-k ≠ biased 的 top-k）| — | ✅ 对比（**正确性否决**）|
| C4 | **gather 原始 logits 再逐 K 重算 σ** | `ids` 同 C1；`w = σ(gather(x, si)[:,0:K])` | 数学恒等（逐点函数，`σ(x)[ids]=σ(x[ids])`，§1.6.1-E1 与 C1 差 0）| 要在 K 上多做一次 `T.vsigmoid`（1 op/tile），且需保留原始 logits 整行（同 C1 内存）→ 劣于 C1 | ✅ 对比（弃选，作 C1 的等价替代）|
| C5 | **在原始 logits 上选 top-k**（sibling 标准变体的 E1）| `ids = topK(x)` | ❌ **不等价**——`σ(x)+b` **非 x 的单调函数**（b 逐专家不同），top-k 集合与 `topK(x)` 不同 | — | ✅ 对比（**正确性否决**，本变体不可用）|
| C6 | `T.vtopk` 硬件 top-k 原语 | 直接选 top-k | — | — | ❌ **不存在**（查证：`ls docs/Tilelang.language/排序操作/` 仅 `T.vsort.md`+`T.vcumsum.md`；全 `docs/` grep `vtopk` 无命中）|
| C7 | 升序排序后取末 K / 反转 | `T.vsort(ascending)` + 反转 | 数学恒等 | `descending=True` 直接可用，升序需额外翻转（+repack）| ✅ 对比（弃选）|
| C8 | 分块 top-k + 归并（`ALG-topk` R2 候选）| 行分块各取局部 top-k 再归并 | 数学恒等 | E≤384 单行即驻 UB，分块归并徒增缓冲与多轮发射 | ✅ 对比（弃选）|
| C9 | 阈值 / 分桶选择 | 直方图/阈值筛选 | 容差内（非精确）| 需额外物化、精度不可控（会破坏 id 逐元素一致）| ✅ 对比（弃选）|
| C10 | host 侧 `torch.topk` | CPU/框架选 | 等价 | 引入 device-host 往返，违反 host 约束（`ascend-constraints.md` §4）| ✅ 对比（弃选）|

> **代数化简空间的结构性结论**：top-k 是**选择**（非代数运算），且本变体多一层**索引置换回取**（选择键 ≠ 输出值），
> 无「更少算式的恒等变形」。可优化空间集中在 ① 选择键在哪个域构造（必须显式 `σ(x)+b`，C5 被正确性否决）、
> ② 用什么原语做选择（排序 vs K 轮 argmax，C1 vs C0）、③ 如何把 unbiased σ 回填到选择置换上（C1 gather-reorder
> vs C2 相减 vs C4 重算）。**C1 同时优化三者，且是 D-3 探针唯一实机验证的形态。**

**R2 在线算法**：**有，但本场景不采纳**。流式 top-k（大小 K 的堆，`O(E log K)` 单遍）与 online σ 都是公认在线算法；
但：① 每行 logits 已**整行驻留 UB**（E≤384 ≪ UB），单遍读取，在线化不减少 GM 遍数（源与 C1 都是「gating 读一遍」）；
② 堆结构在 NPU Vector 上无可表达原语（无 `T.vheap`/堆 API，`docs/Tilelang.language/` 全子目录枚举无命中）；
③ 本变体选择键 `σ(x)+b` 需**全行**才能排序（选择是全局 arg-top-k，不能只用「running K 个状态」增量给出精确 top-k
  除非带 K 堆）。故 R2 结论 = **在线变体存在但无收益**（结构依据：整行 UB 驻留 + 遍数不变 + 堆不可向量化）。

**R3 复杂度对比**（四口径；代表 shape = manifest `FusedMoeFwdCbFwdOp` 的 `kimi-k2-prefill`（T=4096, E=256, K=8,
bf16, sigmoid, renorm=True）；「访存」含 gating 读 + bias 读 + 两张输出写，不含 on-chip；「发射」为 NPU 真实成本主项，
按 `CONST-vector-launch-overhead` ~0.5µs/op 口径，`op_tile` 计数见下）：

| 算法候选 | **FLOPs**（浮点/比较运算计数，逐行） | 访存量 Bytes（GM） | 扫描遍数（GM） | 中间缓冲峰值 | 向量 op 数 / 核（40 核） |
|---------|---------------|-------------------|---------------|-------------|------------------------|
| C0 基线（warp 双数组 K 轮 argmax + shuffle）| `E` 次 σ + `K·E` 比较 + `K·(5 轮 shuffle ×3)` | 2.36MB | 1（gating 读一遍）| 寄存器 `2·ELEMS` + 2 输出 | **NPU 不可直译**（无 warp）；若译 K 轮 reduce+索引恢复 ≈ `K·c` op/tile |
| **C1 选定（`T.vsort(biased)` + `T.gather` σ）** | `O(E log²E)` 排序比较 + `E` 次 σ + `E` 次 vadd + `O(E)` gather + `O(K)` renorm | 同 C0 | 1 | UB `(BM,E)·16B` + 小缓冲（§4.5）| **op_tile≈13**（载入+vsigmoid+copy-bias+vbrc+vadd+vsort+copy si+gather+copy sv+reduce+vdiv+2 写出）≈ 每核 `⌈171/40⌉·13 ≈ 65 op → ~28µs` 发射下界（见 roofline 未实证项）|
| C2 sort + gather 偏置相减 | 同 C1 + `O(E)` gather(b) + `O(K)` vsub | 同 C0 | 1 | 同 C1 | ≈ C1 + 2 op/tile → 弃选 |
| C3 两路独立排序 | `2·O(E log²E)` | 同 C0 | 1 | 多一份排序值缓冲 | ❌ **正确性否决**（置换不匹配）|
| C4 gather logits + 逐 K 重算 σ | 同 C1 + `O(K)` σ | 同 C0 | 1 | 同 C1 | ≈ C1 + 1 op/tile（K 上 σ）→ 弃选（等价替代）|
| C8 分块 top-k + 归并 | 局部 `O(bn log²bn)` + 归并 | 同 C0 + 局部缓冲 | 1 | 多块局部值/下标缓冲 | 多轮发射 > C1，E 小时 overkill → 弃选 |
| C10 host 侧 topk | `O(T·E log E)`（CPU/框架）| + device→host→device 往返 | 1 + 2 次总线 | host 内存 | **CPU 同步阻塞**，违反 host 约束 → 弃选 |

> 口径脚注：① **FLOPs（浮点/比较运算计数）** 只计比较/超越/算术，`O(E log²E)` 为排序次数级估算（VSortOp 内部
> 实现未公开，按经典排序下界标注）；② 访存量 = 必要 I/O（不可消除），各候选相同；③ 扫描遍数指同一数据在 GM 被读次数；
> ④ **发射项是本算子 NPU 侧主导（§1.6.0 roofline）**，故 `op_tile` 计数是候选排序的关键口径；
> ⑤ C0 的 `K·(5 轮 shuffle ×3)` 为 warp 归约量，NPU 无对应，仅作基线复杂度参照。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰按 `negative-claim-evidence.md` §1 举证）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 工具链可用性 | 结论 |
|------|-------------|---------|------------|---------|------------|-------------|------|
| C0 基线 | Vector 可做归约，但**one-warp-per-token + shuffle 在 NPU 无对应** → ⚠️ | 寄存器/UB ✅ | lane 32 对齐需 padding | grid 编译期常量 ✅ | ❌ shuffle 无替代即改形态 | 源形态不可编译（CUDA 专用 `T.Kernel(threads=)`/`shfl_xor`）| ⚠️ 需重设计（→C1）|
| **C1 选定** | ✅ 纯 Vector：`T.vsigmoid`/`T.vbrc`/`T.vadd`/`T.vsort`/`T.gather`/`T.reduce_sum`/`T.vdiv`（fp32 ✓）| UB `(BM,E)·16B` ≪192KB ✅（§4.5）| **尾轴 E = 排序轴**（`T.vsort` 仅尾轴）✅；gather 沿末维 ✅；E 任意宽度 ✅ | `T`/`E`/`K` 工厂期常量；核内循环静态界（仅 guard 动态）✅ | ✅ 单引擎 Stream、无 Cube、无 CV 交接；Developer auto-sync | ✅ **D-3 实机探针**：E∈{32,64,128,256,384} 编译通过、ids 逐元素一致、权重误差 ≤2.98e-08 | ✅ **选定** |
| C2 sort + 相减 | ✅ Vector | ✅ | ✅ | ✅ | ✅ | ✅（但需 b 的行展开）| ❌ 弃选：多 1 gather + 1 vsub，无收益 |
| C3 两路排序 | ✅ Vector | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：**正确性否决**（置换不匹配，§1.6.1）|
| C4 gather logits + 重算 | ✅ Vector | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：多 1 op/tile，等价替代（C1 子集）|
| C5 原始 logits 选择 | ✅ Vector | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：**正确性否决**（biased 非单调，本变体不适用）|
| C6 `T.vtopk` | — | — | — | — | — | ❌ **无此 API**（`docs/Tilelang.language/排序操作/` 仅 vsort/vcumsum；全 docs grep 无命中）| ❌ 弃选（不存在）|
| C7 升序+反转 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：`descending=True` 直接可用 |
| C8 分块 top-k + 归并 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：E≤384 单行驻 UB，徒增缓冲/发射 |
| C9 阈值/分桶 | ✅ | ⚠️ 需物化 | ✅ | ✅ | ✅ | ✅ | ❌ 弃选：精度不可控 + 破坏 id 逐元素一致 |
| C10 host 侧 | CPU | — | — | — | ❌ device-host 往返 | ✅（可行但违约束）| ❌ 弃选：违反 host 约束（`ascend-constraints.md` §4）|

**调研结论**：选定 **C1 —— `biased=σ(x)+b` 上 `T.vsort` 降序取前 K 得下标 → full-width `T.gather` 按该置换取回
已算好的 σ → K 切片 renorm**。
关键依据：(i) **正确性**——C5（sibling 的单调优化）与 C3（两路排序）被正确性否决；本变体选择键非 x 单调，
必须显式计算全行 `biased` 后排序；(ii) **可行性**——C6 无该原语、C0 源形态 CUDA 专用；C1 经 **D-3 探针设备实测**
（`T.vsort` + `T.gather` 组合编译通过、ids 逐元素一致、权重误差 ≤2.98e-08）；(iii) **发射经济性**——D-2 roofline
显示本算子发射/排序主导；C1 用 1 条排序原语替代 C0 的 K 轮归约，`op_tile` 最低（13）；
(iv) **布局零代价**——`T.vsort` 仅尾轴排序、`T.gather` 仅末维 gather，而 I/O 的 E 轴天然是尾轴，**无需任何 repack**
（§1.6.3）。
**与基线结构差异一句话**：保留源「逐行 σ + 逐行选 top-k + 输出未加偏置 σ」的数据流，把
「warp 双数组 K 轮 masked argmax + shuffle 搬运」换成「**一次 `T.vsort(biased)` 降序 + 一次 `T.gather` 置换回取 σ**」，
把 warp/thread 并行换成行块 persistent 分核，把 host 侧 renormalize 换成 kernel 内融合。
**调研范围**（「无更优替代」的查证范围）：`algorithm-candidates.md` 的 `ALG-topk` 命中行（C8 分块归并已评估）；
`kb_search` 命中条目（`ALG-topk`；`VP-2026-0051/0137` 为 queue pending，仅线索不作依据；`CG-2026-0033`/`CG-2026-0021`
为 open，仅线索）；`docs/Tilelang.language/排序操作/`、`索引与元素操作/`、`规约操作/`、`shape操作/`、`数学操作/`
全子目录枚举（确认无 `vtopk`，`T.gather`/`T.vbrc`/`T.vsigmoid` 存在）；本仓 examples（sibling topk / sort / logsumexp /
cumprod / gather 测试）；源算子 §0.3/§0.4；**实机探针**；恒等变形空间自查（C1–C10 全枚举，C3/C5 由正确性排除，
C2/C4/C7/C8/C9/C10 由收益/代价排除）。**源算法优化手段承接**（§0.4）：O1→R1、O2→UB、O3→R3、O4→R4（**意图显式承接**）、
O5→R3、O6 保留、O7→R5、O8/O9 保留、O10→R6。

**设计期 roofline（D-2；协议：`_shared/standards/hardware-cost-model.md` §2；常数带版本戳与陈旧性标注）**：

> **常数陈旧性声明（`kb_stale_check.py`，2026-10-16 实跑）**：本条引用的 `CONST-vector-launch-overhead` /
> `CONST-mte2-degradation` / `CONST-capacity-910B2C` 在 pattern-library 中 front-matter `status: verified`，但
> **`kb_stale_check.py` 报告 `stale_count=133`——三者均被标 `[stale]`**（绑定旧工具链戳 `tilelang 0.1.2+3a214cde /
> +ed787bb + CANN 8.5.0`，当前戳为 `tilelang 0.1.2+cd100d3dda + CANN ?`）。**故本设计不将上述常数作为任何
> 选定/弃选裁决的依据**（裁决依据为正确性论证 + D-3 实机探针 + 结构判据）；仅将按下式算出的**估算下界**作为
> **「未实测假设 + 估算依据」**的性能画像，**待重验**、Stage 4 实测对账。
> **设备口径**：常数 origin 设备为 **Ascend910B2C（24 AICore）**，本设备实查为 **Ascend910_9362（20 AICore →
> 纯 Vector 40 核）**（`NPUUtils.get().get_aicore_num()` = 20，2026-10-16 实查，chip 1）；下述估算按「per-core 发射率
> 不变 × 40 核」外推——属**未实测假设**。

| workload（manifest） | dtype | BM | num_logical | 流量项 | 发射项（op_tile≈13，0.5µs/op） | **估算下界 = max** |
|---|---|---|---|---|---|---|
| `fused-moe-cb-smoke` (16,4) K=2 | fp16/bf16 | 16 | 1 | 0.4KB ÷ (40·~30GB/s) ≈ 0.00µs | 13·0.5 = **6.5µs**（单逻辑核）| ≈6.5µs（发射主导）|
| `kimi-k2-decode` (512,256) K=8 | bf16 | 24 | 22 | 294KB ≈ 0.4µs | 22/40·13·0.5 = **3.6µs** | **≈3.6µs（发射主导）** |
| `kimi-k2-medium` (2048,256) K=8 | bf16 | 24 | 86 | 1.18MB ≈ 1.5µs | 86/40·13·0.5 = **14µs** | **≈14µs（发射主导）** |
| `kimi-k2-prefill` (4096,256) K=8 | bf16 | 24 | 171 | 2.36MB ≈ 3.0µs | 171/40·13·0.5 = **28µs** | **≈28µs（发射主导）** |

> 估算下界 = max(流量项, 发射项)（容量项 **n/a**——纯 Vector 单引擎、无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only`
> 不适用）。**结论：本设计为发射 / 选择主导（非访存主导）**，故 §1.6 开头的 Mem/BW 利用率辅助口径应配合
> `vec_ratio`/`mte2_ratio` 观测，而非 Ratio。
> **未文档化假设项**：`T.vsort`（VSortOp）+ `T.gather`（VGatherOp）单条成本**未在 pattern-library 实测**
> （0.5µs/op 是通用向量发射常数，排序/索引原语实际成本可能显著更高）；该量**不影响主选/弃选裁决**（可行性已由
> D-3 探针 + 工具链证据裁定），仅影响成本特征刻画 → 交 **Stage 4 实测**（§9 R-6）。
> **用户告诫 #2 落实**：本条仅为**设计期 roofline 估算下界**（待测假设）；「向量链被 MTE2 吸收」类论断**不得**作为
> 性能目标写入设计——本设计**不设**该目标，一切以 Stage 4 msprof 实测为准（§9 R-6）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法 C1。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）；
> **等价性机器验证（D-1）**：`examples/fused_topk_cb/_fused_topk_kernel_cb/verify_equiv.py`
> （torch CPU，fp64 参照 + fp32 仿真），结果表内嵌于下；**实机探针（D-3）**：
> `history_version/design_probe_cb_path.py`（完整路径）、`history_version/design_probe_ub_sweep.py`（容量包线）、
> `history_version/design_probe_isolation.py`（`vsort+gather` / `vbrc+vadd` 隔离），2026-10-16 实机 NPU 执行。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | **σ 只算一次 + `T.gather` 置换回取**（承接 §0.6-R4；替代「gather logits 再逐 K 重算 σ」C4）| `w_k = σ(x[ids_k])`（在 K 上重算）| `w_k = σ(x)[ids_k]`（全行 σ 一次，按选择置换重排）| **数学恒等**：σ 是**逐点函数**，`σ(x)[ids] == σ(x[ids])`（逐元素同一映射）。机器验证 `gather-reorder` vs `gather-recompute` 差 **0/65**（fp64 全等）| 省 K 上的 `T.vsigmoid`（1 op/tile）+ 省一次「保留原始 logits 供重算」的语义约束；且 1 条 `VGatherOp` 一次取回整个选择置换下的 σ |
| E2 | **renormalize 融合进 kernel 的 K 切片**（承接 §0.6-R6）| kernel 出未归一权重 + host 侧 `w/w.sum` | kernel 内 `T.reduce_sum(w)` + `T.vdiv(w, sr)` | **数学恒等**：同一除法，仅执行位置不同。机器验证 renorm 用例违反率 **0**、max_rel **0** | 消除第二次 host 算子与一次输出 GM 往返；只在 K 元素切片上归约（非全行）。**注：本变体无 softmax 式全行分母抵消**（σ 无公共全行分母），故 renorm 恒需 `reduce_sum`——如实记录，不套用 sibling 结论 |
| E3 | **非整除 E 无 padding**（承接 §0.6-R5）| 越界专家填 `-inf` 后再选 | 排序/σ/gather 只作用于真实 E 列（`(BM,E)` 满行）| **数学恒等**：源 pad 的 `-inf` 在源中不改变输出（填充位不入选）；消除 padding 不改变任何结果。机器验证含 E=4/33/100 非整除用例违反率 **0** | 消除 masked load 分支与 `-inf` 哨兵；E 任意宽度直承 |
| E4 | **全 dtype 统一 fp32 计算域 + bf16 升域**（承接 §0.4-O6）| 源：host 升 fp32 → kernel fp32 | 同源计算域；kernel 直接收原生 dtype，`T.copy` 自动升 fp32 | 保留源数值路径；且是 **API 必要性**——`T.vsort` 的 `src`/`dst_value` 仅 fp16/fp32、**bf16 ×**（`docs/Tilelang.language/排序操作/T.vsort.md` §2.2.1），bf16 必须升 fp32。舍入点与源一致（fp32 中间 + 输出 fp32）| 消除 host 侧 `gating_output.to(torch.float32)` / `correction_bias.to(torch.float32)`（两次额外算子 + GM 往返）；bf16 路径由 `T.copy` 自动 `VCast` |

**机器验证结果表**（`verify_equiv.py`，2026-10-16 执行，torch CPU，exit 0；基线 = 源 cb 语义
`biased=σ(x)+b → topk(sorted=True) → w=σ(x)[ids] → 可选 renorm` 的 fp64 复刻；候选 = C1 的组合式
fp64 / fp32 计算；判定口径：harness 容差 rtol=atol=1e-3，`max_rel` 为候选/基线相对误差最大值）：

| 验证项 | dtype | 用例数 | max_rel | 违反率 | 结论 |
|--------|-------|-------|---------|--------|------|
| E1 `gather-reorder σ` vs `gather-recompute σ`（逐点函数等价）| fp64 | 65 | 0（选择集/权重全等）| 0 | **EQUIV_PASS** |
| C1 候选 vs 基线权重（fp64，含 ties / 非 pow2 E / ±大值 / K=E / dtype 舍入）| fp64 | 65 | 0 | 0 | **EQUIV_PASS** |
| E2 renorm 融合 vs host 除法（fp32）| fp32 | 35（renorm=True 子集）| 0 | 0 | **EQUIV_PASS** |
| 组合 fp32 域仿真（NPU 计算域 vs fp64 基线）：权重 | float32/float16/bfloat16 | 65 | **2.377e-07** | **0** | **EQUIV_PASS** |
| 组合 fp32 域仿真：`topk_ids` | 同上 | 65 | id 失配 **0/65** | 0 | **EQUIV_PASS** |
| **合计** | — | **65** | — | **0** | **ALL EQUIV_PASS** |

用例覆盖：随机网格 `E∈{4,8,33,100,128,256,384} × K∈{1,2,4,8} × renorm{False,True}`（46 例）+
角点 `all_equal_ties / few_groups_ties / large_magnitude_x50 / large_bias_x5 / pos_neg_mix_x200 / single_expert /
K_equals_E / large_batch(256,384)`（14 例）+ dtype 舍入路径 fp16/bf16（2 例）+ 追加网格（3 例）。
**近 tie 敏感性**：对 σ 施加 **1-ULP** 扰动（`nextafter(σ, +inf)`，模拟 NPU `VSigmoidOp` 与 torch σ 的
最坏 ULP 分歧）后 `topk_ids` 失配 **0/200**；实测 K/(K+1) 名 biased 值最小间隔 2.277e-04（中位 4.906e-03），
远大于 ULP 噪声（~1e-07）——**证明随机连续 logits 下 id 极稳健**（tie 是唯一例外，§9 R-1）。
**无 `EQUIV_FAIL` 项**，E1–E4 全部采纳。

**实机 NPU 探针结果**（D-3，2026-10-16，Ascend910_9362，chip 1，Developer 模式）：

| 探针项 | 输入 | 结果 | 结论 |
|--------|------|------|------|
| P0-A `T.gather` + `T.vsort` 隔离（full-width）| (8,32)/(16,64) fp32 | 编译通过 | ✅ vsort→gather 组合在本工具链可用 |
| P0-B `T.vbrc`+`T.vadd` 偏置广播隔离 | (8,32)/(16,64) fp32 | 编译通过 | ✅ `(1,E)→(BM,E)` 广播 + 行加可用 |
| P2-A 完整 cb 路径 vs torch golden | (32,128,K=8) fp32 | ids **逐元素一致**、w_max **2.98e-08** | ✅ σ→vbrc/vadd→vsort→copy si→gather→renorm 全链正确 |
| P2-B 同上 | (16,256,K=8) | ids 一致、w_max 2.98e-08 | ✅ |
| P2-C 同上（Kimi 专家数）| (16,384,K=8) | ids 一致、w_max 1.49e-08 | ✅ E=384 可用 |
| P2-D 同上（非整除 E）| (16,32,K=4) / (32,256,K=8) / (32,64,K=8) | ids 一致、w_max ≤2.98e-08 | ✅ |
| P3 UB 容量包线（同结构扫描）| (8..32, 32..384) | 全部编译 + 数值正确；**65KB（32,128）曾一次报 `ub overflow`（同一 TIR 复跑通过）** | ⚠️ 容量近阈值 + **编译间歇性 UB 判定不稳**（§9 R-5）|

**优化结论**：采纳 E1–E4 共 4 项（E1/E2/E3 为收益项；E4 为保留源数值路径 + API 必要性并附论证）；
优化后公式 = §1.3 的第二式（供 §3.1 拆解）。**无优化空间项**：top-k 选择本身无代数化简（选择非代数运算，
§1.6.0 R1 结构性结论）；**本变体特有的「偏置只影响选择、不影响输出」是语义要求而非可优化项**——
它带来的额外成本（全行 `σ` + biased 构造 + 置换回取 gather）已由 E1 最小化（σ 只算一次）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/`、`testing/`
> 或实机探针佐证（列内标注）。

| # | 计算点 | 原实现形态（源/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | top-k 选择 | 源 K 轮 masked argmax + `shfl_xor` | `T.vsort(descending=True)` 取前 K（`docs/Tilelang.language/排序操作/T.vsort.md`；探针 P0-A/P2）| ✅（消除）| — |
| 2 | σ(x) 全行打分 | per-lane `1/(1+exp(-x))` | `T.vsigmoid(xf, xf)`（`docs/Tilelang.language/数学操作/T.sigmoid.md`，fp32 ✓；探针 P2）| ✅ | — |
| 3 | 偏置广播 `(1,E)→(BM,E)` | per-lane 读 `bias[expert_idx]` | `T.vbrc(bb, biased)`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2；探针 P0-B）| ✅ | — |
| 4 | 选择键 `σ+b` | per-lane 加 | `T.vadd(xf, biased, biased)`（`docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2；探针 P2）| ✅ | — |
| 5 | **置换回取 σ**（本变体核心）| 源 `l_best_orig` 随 `shfl_xor` 逐轮搬运 | `T.gather(xf, sv, si)`（full-width，runtime int32 下标；`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1/§2.4；`testing/npuir/indexing_ops/test_slice_gather_dev.py`；探针 P0-A/P2）| ✅ | — |
| 6 | 选择下标提取 `si[:,0:K]` | 逐元素 | `T.copy` 列 slice（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2；探针 P2）| ✅ | — |
| 7 | renorm 分母 `Σ_K w` | host 侧 `w.sum(-1)` | `T.reduce_sum(vk, sr, dim=1)`（`docs/Tilelang.language/规约操作/T.reduce_sum.md`；探针 P2）| ✅ | — |
| 8 | renorm 除法（行广播）| 标量除 | `T.vdiv(vk, sr, vk)`（`docs/Tilelang.language/数学操作/T.vdiv.md` §2.2.2 行广播）| ✅ | — |
| 9 | GM↔UB 载入/写出（含 dtype 升/降） | per-lane 标量 load/store | `T.copy` slice（跨 dtype 自动 `VCast`；`docs/Tilelang.language/内存操作/T.copy.md`；探针 P2）| ✅ | — |
| 10 | 输出行块提取（`sv[:,0:K]`/`si[:,0:K]`） | 逐元素 | `T.copy` 列 slice（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2；探针 P2）| ✅ | — |
| 11 | persistent 任务循环 `for s in T.serial(num_local_tasks)` | block 级串行 | 无 | ❌（保留）| **block 级任务映射 + grid-stride 调度**：核内任务序（`task=s·num_kernels+cid`）是分核结构本身（`core-split-strategy.md` §1 要素③ 极大规模方案），非逐元素热点；无逐元素等价 API |
| 12 | 任务/块索引算术（`task`、`off`、`real=T.min(...)`） | 标量整除/取余/min | 无（纯 TIR 标量算术）| ❌（保留）| **block 级索引/任务映射与动态尾块边界**：每 tile O(1) 条，非逐元素热点；进 slice extents |
| 13 | host 侧 `lru_cache` 工厂特化 / validate / dtype 上送 | Python 工厂期标量 | 无 | ❌（保留）| **host 元数据计算**（不在 kernel 内；每 `(T,E,K,scoring_func,bias)` 一次）|

**向量化结论**：逐元素 / 选择 / 索引置换计算点（#1–#10）**全部向量化**（替代 API 均有 `docs/` 条目、`testing/`
用例或实机探针佐证：`T.vsort`/`T.vsigmoid`/`T.vbrc`/`T.vadd`/`T.gather`/`T.reduce_sum`/`T.vdiv`/`T.copy`）；
保留 3 类标量/循环（#11–#13），理由类别：分核与任务调度串行（#11）、block 索引 + 动态尾块边界（#12）、
host 元数据（#13）——**均非逐元素热点**（每 tile/任务 1 次发射量级），与 §6 循环结构一致（§6 无任何逐元素标量循环）。
源实现中唯一的逐元素热点（K 轮 per-lane argmax + shuffle 归约 + 逐轮 `l_best_orig` 搬运）已被 #1/#5 消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`gating_output (T,E)` 行主序进、`topk_weights (T,K)`/`topk_ids (T,K)` 行主序出、
> `correction_bias (E,)` 行主序进）；**核内布局与 lane 映射是设计变量**。本算子为**选择 + 索引置换类**，
> 有 2 个非 batch 维度（T 为任务维、E 为选择/置换维）。

**轴质量评分**（候选轴 = 选择/置换轴 E；T 为任务分核轴）：

| 评分项 | 方案 A：行块 `(BM,E)` 驻留，lane→E（**排序轴 = 尾轴 E**）| 方案 B：转置 `(E,BM)` 后排序 | 方案 C：每行 BM=1 |
|--------|------|------|------|
| 整除性 | E 任意宽度（`T.vsort` 尾轴全排序）| 需核内转置链 | E 任意 |
| **排序原语匹配** | ✅ `T.vsort` **仅支持尾轴**（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3-1），E 恰为尾轴 → 直接可用 | ❌ `T.vsort` 不支持非尾轴（编译期报错）→ 必须转置 | ✅ 尾轴 |
| **gather 匹配** | ✅ `T.gather` **仅沿末维**（`docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3），E 恰为末维且 `indices=si` 同形 | ❌ 末维变为 BM ≠ 原专家维，语义错位 | ✅ 末维 |
| repack 代价 | **无**（I/O 原生行主序，E 连续）| 需 `T.transpose` 二轴交换链（UB 级开销 + 额外缓冲）| 无 |
| 发射经济性 | ✅ `(BM,E)` 宽块，`CONST-vector-launch-overhead`「宽块恒优于窄块」；BM 越大 op 数越少 | 多转置 op | ❌ BM=1 → `num_logical=T` → op 数最多（发射灾难）|
| UB 容量影响 | `(BM,E)·16B + 小缓冲`（§4.5）| 多一份转置缓冲 | `(1,E)·16B`，UB 浪费（无法摊薄）|
| 采纳 | ✅ **主选** | ❌ 弃选 | ❌ 弃选 |

**规约/索引 lane 映射正交决策：水平形态 vs 垂直扫描**（正交于「选哪条轴」的必选候选；本算子含 K 元素
`reduce_sum` renorm 分母，§1.6.2-#7）：

| 评分项 | H：水平形态（lane→E/K，`T.reduce_sum(dim=1)` 向量归约）**主选** | V：垂直扫描（lane→独立行 T，串行步进）|
|--------|------|------|
| lane 映射语义 | 归约轴 K 落到向量 lane 宽度，`T.reduce_sum(vk, dim=1)` 在 K 上做向量归约 | 行轴 T 落到向量 lane，每 lane 独立串行累加 |
| **与 `T.vsort`/`T.gather` 兼容性** | ✅ 同一 E 末轴：排序/置换后 `(BM,E)`/`(BM,K)` 整行驻留，`reduce_sum` 直接作用在同一 buffer → **零 repack** | ❌ 需令行轴成为 lane 连续轴（核内转置、尾轴换 T），与 `T.vsort` 强制尾轴 E 排序冲突 |
| 整除性 / 尾 lane 浪费 | ✅ K 任意宽度（`reduce_*` 内部处理尾 lane）| ❌ 行轴映射到向量宽度（fp32 ×4）→ BM 须向 lane 宽度对齐，尾 lane 浪费 |
| 累加链形态 | ✅ 树形/多路并行归约，无长串行依赖（K≤8，链极短）| ❌ 串行链长 = K |
| 发射经济性 | ✅ 1 条 reduce 原语/行块 | ❌ 每行 K 次串行步进的循环/发射开销 |
| 采纳 | ✅ **主选** | ❌ **结构否决**（与 `T.vsort`/`T.gather` 末轴整行驻留约束冲突）|

> **两遍 vs online 与 lane 映射的交互**：§1.6.0-R2 已判定在线变体无收益（整行 UB 驻留、GM 遍数不变、堆不可向量化）；
> 故水平形态沿用「σ 一遍 / biased 一遍 / 排序一遍 / gather 一遍」的离线形态，lane 映射不改变 GM 扫描遍数（均 1 遍）。

**候选矩阵**（含源码 GPU 轴选择对照——源码 warp→token、lane→专家子集（步长 32），是输入非结论）：

| # | 布局方案 | 向量化轴 / 关键结构 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| 1 | **行块 `(BM,E)` UB 驻留 + 末轴排序 + 末维 gather**（选定）| E（末轴）；`T.vsort(sort_axis=-1)` + `T.gather`（末维）+ `reduce_sum(dim=1)` | 无（I/O 原生）| 单 kernel、零 repack、承载全部优化 | ✅ **主选** |
| 2 | 核内转置 `(E,BM)` 后用非尾轴排序 | — | `T.transpose` 链 | ❌ 前提被否：`T.vsort` 不支持非尾轴（文档明示编译期报错），且 gather 语义错位 | ❌（文档否决）|
| 3 | 每行一块（BM=1）| E | 无 | `num_logical=T=4096` → op 数 ~6× 于 BM=24，发射灾难 | ❌（量化否决）|
| 4 | 全宽 gather 改为 K 宽 gather（`indices=(BM,K)`、`dst=(BM,K)`）| E | 无 | `docs` 示例与 `testing/` 用例均要求 `src`/`indices`/`dst` **同形**（`test_slice_gather_*` 三切片同形）；K 宽形态**未验证**；全宽形态已由 D-3 探针 P0-A/P2 实机验证且复用 `sv` 缓冲（零额外 UB）| ❌（证据不足；全宽已实证且更省）|
| 5 | 垂直扫描 lane 映射（lane→T 行、串行步进 E/K）| 行轴 T | 需核内转置 | ❌ 与 `T.vsort`/`T.gather` 末轴约束冲突 + 长串行链 → 结构否决 | ❌（结构否决）|

**布局决策结论**：选定 **#1**：核内布局 = 行块 `(BM,E)` 行主序驻留 UB、**向量化/排序/置换轴 = 尾轴 E**、
**规约 lane 映射 = 水平形态（`T.reduce_sum(dim=1)`）**、repack 无、分核 = persistent 行块（§5.5）。
该结论同步落入 §3.3 伪代码与 §6 循环结构（排序/gather/逐元素/规约的向量维 = E 轴；buffer 形状 = `(BM,E)` ×3 +
`(BM,E)` int32 + `(1,E)` + `(BM,K)` ×2 + `(BM,1)`）。
**弃选理由**：#2 因 `T.vsort` 仅尾轴 + `T.gather` 仅末维（文档否决）；#3 量化发射灾难；#4 因「K 宽 gather 形态
证据不足」且全宽已实机验证并复用缓冲；#5 垂直扫描被末轴整行驻留约束**结构否决**。
**主选与弃选判定裕度不依赖未实证常数**（#2 被文档能力硬否决、#3 是 ~6× 量级差、#5 被末轴约束硬否决、
#4 以实机证据 + UB 账取胜 → **不触发实验裁决模式**）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（harness 迁移 prompt 指定 developer；与 sibling 标准变体一致）。

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.vsigmoid` + `T.vbrc`/`T.vadd` + `T.vsort` + `T.gather` + `T.reduce_sum`/`T.vdiv`），无 matmul、无 Cube | 不需 L1/L0；`T.alloc_shared`→UB 即可 |
| 选择/索引原语 | `T.vsort`（`testing/npuir/sort_ops/test_vsort_dev.py` 为 **Developer** 通过用例）、`T.gather`（`testing/npuir/indexing_ops/test_slice_gather_dev.py` 为 **Developer** 通过用例 + D-3 探针实证）| Developer 可用 |
| 内存层级 | 仅 GM ↔ UB，无跨引擎、无 workspace | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作 | Developer 自动同步即可 |
| 家族先例 | sibling `_fused_topk_kernel`、logsumexp（行归约）、cumprod（persistent 分核）均为 Developer 且通过 | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging + σ/biased 行块 + 排序/置换缓冲；编译器映射 UB）|
| 计算方式 | `T.vsigmoid`（σ）+ `T.vbrc`/`T.vadd`（偏置广播 + 选择键）+ `T.vsort`（选择）+ `T.gather`（置换回取）+ `T.reduce_sum`/`T.vdiv`（renorm）；无逐元素标量循环（§1.6.2）|
| 同步 | Developer 自动同步（无 `sync_block_set/wait`）|
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

### 2.4 AIV 拆分 / subid 切分说明（用户告诫 #3）

本设计的分核方案**不依赖** AIV 拆分 / subid 切分（核内一维 persistent + `T.serial` grid-stride 在 Developer 模式
已可实现，D-3 探针与 sibling/cumprod 先例均在 Developer 通过）。**若 Stage 4 需要**「单核内两 AIV 子块并行」
（`PL-1.13-aiv-dup-subid-split` 形态）以进一步摊薄发射，则**必须切到 Expert 模式**（Developer 模式做不到）；
届时降级方案 = 保持本设计的 Developer 一维 persistent（核数 40、核内串行），以增大 `BM`（减少 `op_tile` 数）
作为 Developer 侧的性能杠杆，而非引入 subid。该判断为**可选优化**，不影响本设计可行性。

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `xf ← gating[off:off+real, 0:E]` | GM 行块 → UB（跨 dtype 自动升 fp32）|
| 2 | `xf ← σ(xf)` | 全行 σ，**只算一次**（供选择键构造与输出置换回取）|
| 3 | `biased ← b`（`(1,E)`→`(BM,E)` 广播）| 偏置广播 |
| 4 | `biased ← xf + biased` | 选择键 `biased = σ(x) + b` |
| 5 | `(sv, si) ← sort_desc(biased)` | biased 降序排序（值 + 原始 int32 下标）|
| 6 | `ids ← si[:,0:K]` | 前 K 下标 = `topk_ids` |
| 7 | `sv ← gather(xf, si)`；`w ← sv[:,0:K]` | 用选择置换取回 σ = 未加偏置权重 |
| 8 | `w ← w / Σ_K w`（renormalize 时）| K 切片行广播除 |
| 9 | `topk_weights[off:off+real,:] ← w`；`topk_ids[off:off+real,:] ← ids` | UB → GM 写出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| 1 载入 | GM→UB 行块（升 fp32）| `T.copy(gating[off:off+real, 0:E], xf[0:real, 0:E])` | 双显式 slice；跨 dtype 自动 `VCast` | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.1/§2.2.2；sibling 探针 P1-C |
| 2 σ | `σ(x)` | `T.vsigmoid(xf, xf)` | fp32 ✓；原地 | Developer | `docs/Tilelang.language/数学操作/T.sigmoid.md`；**D-3 P2** |
| 3 偏置广播 | `(1,E)→(BM,E)` | `T.copy(bias[0:E], bb[0:1,0:E])` + `T.vbrc(bb, biased)` | rank 同（2D），首维 1→BM 广播 | Developer | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（`(1,N,K)→(M,N,K)` 合法）；**D-3 P0-B/P2** |
| 4 选择键 | `σ+b` | `T.vadd(xf, biased, biased)` | 同形逐元素；原地 | Developer | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2/§2.3-3；**D-3 P2** |
| 5 选择 | 降序排序 | `T.vsort(biased, sv, si, descending=True, sort_axis=-1)` | `si` int32；仅尾轴；`src`/`dst` 同形 | Developer | `docs/Tilelang.language/排序操作/T.vsort.md`；`testing/npuir/sort_ops/test_vsort_dev.py`；**D-3 P0-A/P2** |
| 6 取下标 | 列 slice | `T.copy(si[0:real,0:K], ik[0:real,0:K])` | src/dst 同 slice | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2；**D-3 P2** |
| 7a 置换回取 | `σ(x)[si]` | `T.gather(xf, sv, si)` | **full-width**：`src`/`indices`/`dst` 同形 `(BM,E)`；indices int32（runtime）| Developer | `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1/§2.3/§2.4；`testing/npuir/indexing_ops/test_slice_gather_dev.py`；**D-3 P0-A/P2** |
| 7b 取前 K | 列 slice | `T.copy(sv[0:real,0:K], vk[0:real,0:K])` | 复用 `sv` 缓冲 | Developer | 同上；**D-3 P2** |
| 8 renorm | `Σ_K w` / 行广播除 | `T.reduce_sum(vk, sr, dim=1, clear=True)` + `T.vdiv(vk, sr, vk)` | `(BM,K)→(BM,1)`；`[BM,K]/[BM,1]` | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md`、`数学操作/T.vdiv.md` §2.2.2；**D-3 P2** |
| 9 写出 | UB→GM | `T.copy(vk[0:real,0:K], topk_weights[off:off+real,0:K])`；`T.copy(ik[0:real,0:K], topk_ids[off:off+real,0:K])` | 双 slice | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 |
| 分核 | persistent | `T.Kernel(num_kernels, is_npu=True) as (cid,_)` + `for s in T.serial(num_local_tasks)` + `if task < num_logical` | 一维 + 静态界 + 动态 guard | Developer | `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`；`docs/开发指南.md` §3.3 |

### 3.3 计算伪代码

工厂期全静态量：`BM`、`num_logical = ceil(T/BM)`、`num_kernels = min(num_logical, 40)`、
`num_local_tasks = ceil(num_logical/num_kernels)`（§5.5）。dtype 分派：fp32 原域；fp16/bf16 由 `T.copy` 自动升 fp32。
**规约 lane 映射**（§1.6.3 主选）：renorm 分母以**水平形态** `T.reduce_sum(dim=1)` 向量原语实现
（lane 映射到 K 轴），**非垂直扫描**；排序/置换轴 = 尾轴 E（与 I/O 原生布局同轴）。

```python
def _build(T, E, K, renormalize, in_dtype, BM):
    num_logical = ceildiv(T, BM)
    num_kernels = min(num_logical, 40)                 # 纯 Vector 物理核数（实查 aicore 20 × 2）
    num_local_tasks = ceildiv(num_logical, num_kernels)  # 静态上界

    @tilelang.jit(out_idx=[2, 3], target="npuir")
    def _func():
        @T.prim_func
        def main(gating: T.Tensor((T, E), in_dtype),          # fp16 / bf16 / fp32
                 correction_bias: T.Tensor((E,), "float32"),
                 topk_weights: T.Tensor((T, K), "float32"),
                 topk_ids: T.Tensor((T, K), "int32")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xf = T.alloc_shared((BM, E), "float32")   # σ(x)（升 fp32 行块 → 原地 σ）
                biased = T.alloc_shared((BM, E), "float32")  # σ(x) + b（选择键）
                sv = T.alloc_shared((BM, E), "float32")   # 降序 biased 值 → 复用为置换后的 σ
                si = T.alloc_shared((BM, E), "int32")     # 降序原始下标
                bb = T.alloc_shared((1, E), "float32")    # 偏置行
                vk = T.alloc_shared((BM, K), "float32")   # 选中权重
                ik = T.alloc_shared((BM, K), "int32")     # 选中下标
                sr = T.alloc_shared((BM, 1), "float32")   # renorm 分母
                for s in T.serial(num_local_tasks):
                    task = s * num_kernels + cid
                    if task < num_logical:
                        off = task * BM
                        real = T.min(BM, T - off)
                        # 1. 载入（自动升 fp32）
                        T.copy(gating[off:off + real, 0:E], xf[0:real, 0:E])
                        # 2. 全行 σ（只算一次）
                        T.vsigmoid(xf, xf)
                        # 3. 偏置广播 (1,E) -> (BM,E)
                        T.copy(correction_bias[0:E], bb[0:1, 0:E])
                        T.vbrc(bb, biased)
                        # 4. 选择键 = σ + b
                        T.vadd(xf, biased, biased)
                        # 5. biased 降序排序（值 + 原始 int32 下标）
                        T.vsort(biased, sv, si, descending=True, sort_axis=-1)
                        # 6. top-k 下标
                        T.copy(si[0:real, 0:top_k], ik[0:real, 0:top_k])
                        # 7. 用选择置换取回 σ（full-width gather，复用 sv 缓冲）
                        T.gather(xf, sv, si)
                        T.copy(sv[0:real, 0:top_k], vk[0:real, 0:top_k])
                        # 8. renorm（K 切片）
                        if renormalize:
                            T.reduce_sum(vk, sr, dim=1, clear=True)
                            T.vdiv(vk, sr, vk)
                        # 9. 写出
                        T.copy(vk[0:real, 0:top_k], topk_weights[off:off + real, 0:top_k])
                        T.copy(ik[0:real, 0:top_k], topk_ids[off:off + real, 0:top_k])
        return main
    return _func
```

> **注**：`real` 只用于尾块 slice extents；`T.vsort`/`T.gather` 作用于 `[0:real, 0:E]` 的满行（`real == BM` 时无差别）。
> Stage 3 实现时对尾块可对 `xf`/`biased`/`sv`/`si` 均以 `[0:real, 0:E]` slice 传入 `T.vsort`/`T.gather`（探针已验证
> full-width 形态；`real < BM` 的 slice 需 Stage 3 用最小尾块用例实测，列入 §9 R-4）。

### 3.4 API 可行性确认

| API | 存在性佐证 | 探针 |
|-----|-----------|------|
| `T.vsort`（降序、fp32、int32 下标、尾轴）| `docs/Tilelang.language/排序操作/T.vsort.md`；`testing/npuir/sort_ops/test_vsort_dev.py` | **D-3 P0-A/P2 实证** |
| `T.gather`（**末维**、runtime int32 下标、full-width）| `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.1/§2.4；`testing/npuir/indexing_ops/test_slice_gather_dev.py` + `_exp.py` | **D-3 P0-A/P2 实证** |
| `T.vbrc`（`(1,E)→(BM,E)` 广播）| `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | **D-3 P0-B 实证** |
| `T.vadd`（同形逐元素、原地）| `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2/§2.3-3 | **D-3 P2 实证** |
| `T.vsigmoid`（fp32，原地）| `docs/Tilelang.language/数学操作/T.sigmoid.md` | **D-3 P2 实证** |
| `T.reduce_sum` / `T.vdiv`（`(BM,K)→(BM,1)` / 行广播）| `docs/Tilelang.language/规约操作/T.reduce_sum.md`、`数学操作/T.vdiv.md` §2.2.2 | **D-3 P2 实证** |
| `T.copy`（跨 dtype 自动 cast、列 slice、双 slice 尾块）| `docs/Tilelang.language/内存操作/T.copy.md`；sibling 探针 | **D-3 P2 实证** |

### 3.5 技术约束确认

**强制检测结果**：
- **三维 Kernel**：`T.Kernel(num_kernels, is_npu=True)` **一维**——源为 `T.Kernel(grid, threads=...)` 一维 grid，无三维；✅ 无冲突。
- **GPU 专用 API**：源用 `T.get_thread_binding`/`T.shfl_xor`/`T.alloc_local`/`T.Kernel(threads=)`——**全部 CUDA 专用**，
  本设计已全部替换为 Vector 原语（§0.5/§0.6）；✅ 无残留。
- **GEMM 非整除 / L0C 溢出**：本算子**无 GEMM、无 Cube、无 L0/L0C**；✅ n/a。
- **非整除 T**：`real = T.min(BM, T − off)` + 双 slice 处理（§5.4）；✅ 有策略。
- **分核策略**：`num_logical` vs 物理核 40（实查）三要素见 §5.5；✅ 齐备。
- **UB 容量**：`(BM,E)·16B` 行块 + 小缓冲，按 `UB_BUDGET = 96KB` 预算（§4.5）；✅ 有预算（D-3 容量扫描见 §9 R-5）。
- **工具链陷阱规避**（承接 sibling 与 `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel` 先例）：
  ① **不用 `T.alloc_local`**（CG-2026-0028b：`alloc_local + T.serial` SIGSEGV）；② **不用标量 `T.if_then_else`/`>>`**
  （CG-2026-0028a/c）；③ **不标量消费向量产出**（CG-2026-0027）；④ **不在 `T.Parallel` 体内直写 GM**（CG-2026-0030）；
  ⑤ **不用 `T.atomic_add` 标量载荷**（CG-2026-0029）。本设计全部数据流为向量→向量（UB）+ `T.copy`→GM，
  guard 仅 `if task < num_logical` 这一 block 级分段条件（cumprod 先例同款）。
- **`T.vcmp` 无标量操作数**（CG-2026-0021）：本设计**不使用** `T.vcmp`（选择由 `T.vsort` 完成，无逐元素比较链）；✅ n/a。

**本项目同类实现参考**（详见 §10 交付清单）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | Shape | dtype | Layout | 说明 |
|------|-------|-------|--------|------|
| `gating_output` | `(T, E)` | float16 / bfloat16 / float32 | 行主序连续 | 门控 logits；T/E 为编译期常量（工厂按 shape 特化）|
| `correction_bias` | `(E,)` | float32 | 行主序连续 | per-expert 选择偏置（fp32，op 层校验）|

### 4.2 输出张量

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `topk_weights` | `(T, K)` | float32 | 选中专家的**未加偏置** σ 权重（renorm 时已归一）|
| `topk_ids` | `(T, K)` | int32 | 选中专家原始下标（按 biased 降序）|

输出为独立 GM 张量（`out_idx=[2,3]` 由 TileLang 设备侧分配，或由 op 层分配后传入）。

### 4.3 中间缓冲区（UB）

| 名称 | Shape | dtype | 字节 | 用途 |
|------|-------|-------|------|------|
| `xf` | `(BM, E)` | float32 | `BM·E·4` | 升 fp32 行块 → 原地 `T.vsigmoid` = σ(x)（供选择键与置换回取）|
| `biased` | `(BM, E)` | float32 | `BM·E·4` | 选择键 `σ(x)+b` |
| `sv` | `(BM, E)` | float32 | `BM·E·4` | `T.vsort` 降序值 → **复用**为 `T.gather` 的置换后 σ 输出 |
| `si` | `(BM, E)` | int32 | `BM·E·4` | `T.vsort` 降序原始下标（兼作 `T.gather` 的 indices）|
| `bb` | `(1, E)` | float32 | `E·4` | 偏置行（`T.vbrc` 源）|
| `vk` | `(BM, K)` | float32 | `BM·K·4` | 选中权重 |
| `ik` | `(BM, K)` | int32 | `BM·K·4` | 选中下标 |
| `sr` | `(BM, 1)` | float32 | `BM·4` | renorm 分母 |

**大缓冲** = `xf`+`biased`+`sv`+`si` = `BM·E·16` 字节/行块（比 sibling 标准变体的 12B/elem 多 `biased` 一份，
因本变体需**同时**保有选择键与未加偏置 σ——这是语义必需，非冗余）。

### 4.4 内存搬运路径

```
GM[gating_output(T,E)] --T.copy(slice, 自动 VCast→fp32)--> UB[xf (BM,E) fp32]
UB[xf] --T.vsigmoid--> UB[xf = σ(x)]
GM[correction_bias(E)] --T.copy--> UB[bb (1,E)]
UB[bb] --T.vbrc--> UB[biased (BM,E)] --T.vadd(xf)--> UB[biased = σ+b]
UB[biased] --T.vsort--> UB[sv fp32] + UB[si int32]
UB[si] --T.copy 列 slice--> UB[ik (BM,K)]
UB[xf]&UB[si] --T.gather--> UB[sv = σ 按 si 重排] --T.copy 列 slice--> UB[vk (BM,K)]
UB[vk] --reduce_sum/vdiv--> UB[vk (归一权重)]
UB[vk] --T.copy--> GM[topk_weights(T,K) fp32]
UB[ik] --T.copy--> GM[topk_ids(T,K) int32]
无 L1、无 L0A/L0B/L0C、无 workspace、无核间搬运。
```

### 4.5 UB 内存预算

参照 `CONST-capacity-910B2C`（UB = 192KB/AIV；**该常数 `[stale]`，见 §1.6.0 陈旧性声明**；显式 alloc 结构实测
余量 ~12%，即有效预算 ≈ 168KB 可用）。本设计的 UB 占用（大缓冲 `BM·E·16` + 小缓冲 `BM·(8+8K)` 量级）按
**`UB_BUDGET = 96KB`** 上限设计（保守——比 192KB 大幅留余量，应对 `T.vsort`/`T.gather`/`T.vbrc` 内部 scratch +
auto-multi-buffer 膨胀；D-3 容量扫描显示 (32,256)=131KB 仍可编译，故 96KB 有安全裕度）：

$$
BM = \min\Big(BM_{max}=64,\ \Big\lfloor \tfrac{UB\_BUDGET}{E\cdot 16}\Big\rfloor\Big),\qquad
BM = \min(BM,\ T)\ \text{(avoid empty blocks)}
$$

| workload | E | `⌊96KB/(E·16)⌋` | BM | 大缓冲占用 | 小缓冲（`bb/vk/ik/sr`）| 合计 ≈ |
|---|---|---|---|---|---|---|
| `fused-moe-cb-smoke` (16,4) | 4 | 1536 → 64 | 16 | 1KB | <1KB | ~2KB |
| `kimi-k2-decode` (512,256) | 256 | 24 | 24 | 96KB | <1KB | ~97KB |
| `kimi-k2-medium` (2048,256) | 256 | 24 | 24 | 96KB | <1KB | ~97KB |
| `kimi-k2-prefill` (4096,256) | 256 | 24 | 24 | 96KB | <1KB | ~97KB |
| （参考）E=128 | 128 | 48 | 48 | 96KB | <1KB | ~97KB |
| （参考）E=384 | 384 | 16 | 16 | 96KB | <1KB | ~97KB |

> 若 Stage 3 实测 `T.vsort`/`T.gather` 需要额外 UB scratch 致溢出（§9 R-5 已观测一次间歇性 UB 判定不稳），
> 则回退缩小 `BM`（§9 R-5 的调优杠杆）；`BM=32`（131KB）在 D-3 扫描中可编译，可作上限试探。

### 4.6 动态轴定义

`T`/`E`/`K` 均为**工厂期编译常量**（op 层 `_get_kernel` 按 shape 特化，缓存）；运行期无动态 shape
（仅尾块 `real = min(BM, T-off)` 与任务 guard 为运行期值，但循环边界静态）。dtype 为编译期分派（fp16/bf16/fp32）。

### 4.7 JIT 配置

`@tilelang.jit(out_idx=[2,3], target="npuir")`；Developer 模式（`TILELANG_ASCEND_MODE` 由 harness 设定）；
无 `pass_configs`、无 `compile_flags`。工厂层按 `(T,E,K,scoring_func,renormalize,dtype)` 特化缓存。

---

## 5. Tiling 策略

### 5.1 计算类型

**纯 Vector**（无 Cube/matmul）：σ（`T.vsigmoid`）+ 偏置广播加（`T.vbrc`/`T.vadd`）+ 选择（`T.vsort`）+
置换回取（`T.gather`）+ K 切片归约/除（`T.reduce_sum`/`T.vdiv`）。仅需 UB，无 L1/L0。

### 5.2 Block 划分与 Tile Shape

| 参数 | 取值 | 依据 |
|------|------|------|
| Tile shape | `(BM, E)` 行块（整 E 驻留 UB）| E 为排序/置换末轴，必须整行驻留（`T.vsort` 全排序 + `T.gather` 全宽）|
| `BM` | `min(64, ⌊UB_BUDGET/(E·16)⌋, T)` | UB 预算（§4.5）；`BM` 越大 op 数越少（发射摊薄）|
| `K` | 编译期常量（源 `top_k`）| manifest（2/8）|
| 输出 tile | `(BM, K)` | 前 K 列 |

### 5.3 约束分析

- **尾轴对齐**：`T.copy` 要求尾轴 32B 对齐——E 行主序连续天然满足（fp32 UB）；`(BM,K)` 切片 K=2/8 时 8B/32B，
  K=2 非 32B，`T.copy` 尾块按元素处理（sibling 的 K=1（4B）路径已通过，D-3 P2-D E=32/K=4 亦通过）。
- **`T.vsort` 限制**：仅尾轴排序（E 满足）、fp16/fp32（fp32 满足）、`si` int32（满足）、同 shape src/dst（满足）。
- **`T.gather` 限制**：仅**末维** gather（E 满足）、indices int32（满足）、`src`/`indices`/`dst` **同形**
  （full-width `(BM,E)` 满足）、fp16/fp32/bf16 src（fp32 满足）。
- **UB 容量**：§4.5 控制在 96KB 预算内。
- **无分形 / L0C 约束**（无 Cube）。

### 5.4 非整除处理策略

| 维度 | 情况 | 策略 |
|------|------|------|
| **T（行）** | `T % BM ≠ 0` | 尾块 `real = T.min(BM, T − off)`，`T.copy` **src/dst 双显式 slice** `gating[off:off+real]`↔`xf[0:real]`，越界行不载入/不写出；与源 `if token_id < num_tokens` 守卫等价。**不 padding**（host 禁改输入真实内容）|
| **E（专家）** | 任意 | σ/偏置广播/排序/gather 作用于真实 E 列，**无需 padding**（§1.6.1-E3）；E 不需 32/16 整除 |
| **K** | `K ≤ E`（op 层已校验）| `sv[:,0:K]` 直接切片 |
| **correction_bias** | `(E,)` | op 层校验 shape/dtype/同 device；kernel 内 `T.copy(bias[0:E], bb[0:1,0:E])` |

### 5.5 分核策略（三要素）

**① 逻辑核数计算**：`num_logical = ceil(T / BM)`。
- smoke (16,4) BM=16 → **1**；`kimi-k2-decode` (512,256) BM=24 → **22**；`kimi-k2-medium` (2048,256) BM=24 → **86**；
  `kimi-k2-prefill` (4096,256) BM=24 → **171**。

**② 物理核数依据**：`NPUUtils.get().get_aicore_num()` **实查 = 20**（代码
`from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()`，**2026-10-16 执行**，
设备 **Ascend910_9362**，chip 1）；纯 Vector 算子核数翻倍 → **物理核数 = 20 × 2 = 40**
（依据 `_shared/standards/core-split-strategy.md` §1 要素②）。

**③ 规模判定与分核方案**：
- **smoke (16,4)**：`num_logical = 1 ≤ 40` → **无需适配**（结论 + 依据：单核即可，启动 `min(num_logical,40)=1` 核）。
- **`kimi-k2-decode` (512,256)**：`num_logical = 22 ≤ 40` → **无需适配**（单核一轮即覆盖，启动 22 核）。
- **`kimi-k2-medium` (2048,256)**：`num_logical = 86 > 40`；且 `BM` 已受 UB 预算上界约束（§4.5，E=256 时 BM≤24，
  无法继续增大以把 `num_logical` 压到 ≤40）→ **极大规模** → **固定启动内核数 `num_kernels = min(num_logical, 40) = 40`**，
  核内 `T.serial(num_local_tasks)` grid-stride（`num_local_tasks = ceil(86/40) = 3`，**静态值**）+
  动态 `if task < num_logical` 守卫。
- **`kimi-k2-prefill` (4096,256)**：`num_logical = 171 > 40` → 同上，`num_kernels = 40`，
  `num_local_tasks = ceil(171/40) = 5`（**静态值**）+ guard。
  （依据 `core-split-strategy.md` §1 要素③「极大规模」方案；先例 `examples/cumprod/_cumulative_kernel` 同款。）

> 注：T=2048/4096 含因子 2 的幂，`num_logical = 86/171` 不含因子 5，无法令 `num_logical` 恰为 40 的整数倍
> （无「中等规模对齐」解），故取固定核数 + 核内串行方案。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环 | 类型 | 边界 | 说明 |
|------|------|------|------|
| 核内任务循环 `for s in T.serial(num_local_tasks)` | `T.serial` | **静态**（`num_local_tasks` 编译期常量）| grid-stride 任务映射（`task = s·num_kernels + cid`）|
| 任务守卫 `if task < num_logical` | 条件分支（block 级）| 运行期值 | 仅 block 级分段，非逐元素；尾任务跳过 |
| σ / 广播加 / 选择 / 置换 / 归约 | **无显式循环** | — | 全部由 `T.vsigmoid`/`T.vbrc`/`T.vadd`/`T.vsort`/`T.gather`/`T.reduce_sum`/`T.vdiv` 一条原语在 `(BM,E)`/`(BM,K)` 向量上完成 |

**无 `T.Parallel`**（无需显式元素级并行——原语内部已向量化）；**无 `T.Pipelined`**（单行块内无多 stage 依赖，
Developer auto-multi-buffer 自动处理 GM→UB 重叠）；**无 `T.Persistent`**（本设计用显式 persistent 网格 +
`T.serial` 任务循环，与 cumprod 一致）。**无任何逐元素标量循环**（§1.6.2 结论）；renorm 归约 lane 映射 = **水平形态**
（`T.reduce_sum(dim=1)`），**无垂直扫描逐行串行循环**（§1.6.3 主选，与排序/置换轴末轴 E 同轴）。

### 6.2 调度伪代码

见 §3.3。核间无依赖、无流水线；每核顺序处理其 grid-stride 任务集；核内单行块「载入 → σ → 选择键 → 排序 →
置换回取 → renorm → 写出」按 Developer 自动同步顺序执行。

---

## 7. 同步策略

### 7.1 Developer 自动同步（主选）

**Developer 模式自动同步**：单 kernel、单引擎（Vector）、核内顺序数据流（UB 读写 + `T.copy` DMA），
无跨引擎（无 Cube）、无核间通信、无 workspace，故**不显式使用任何同步原语**（无 `sync_block_set/wait`、
无 `pipe_barrier`）。Developer 编译器自动在多 stage（载入 DMA → σ/选择/置换计算 → 写出 DMA）间插入同步与
auto-multi-buffer。

> **工具链注意**：本设计**不依赖**「向量产出 → 标量消费」的同步（CG-2026-0027 该形态不可靠）——所有中间值
> 都在 UB 以**向量**形式被后续向量 op 消费（`T.vsigmoid`/`T.vbrc`/`T.vadd`/`T.vsort`/`T.gather`/`T.reduce_sum`/v-op 链），
> 标量只用于 block 级 `task`/`off`/`real` 索引算术（不消费向量产出）。故自动同步足够。

### 7.2 CV 融合设计

**n/a**——本算子为**纯 Vector**，不含 Cube/`T.gemm`，无 CV 融合点。σ + 选择键 + 选择（排序）+ 置换回取（gather）
+ renorm 全部在 Vector 引擎完成；两输出经 `T.copy` 直达 GM，无跨引擎交接需求。

---

## 8. 验证方案

### 8.1 Golden 函数草案（PyTorch 参考实现）

**以 §0.1 语义为唯一依据**（**不复刻** §0.6 的 NPU 算法：无 vsort、无 gather-reorder）：

```python
def golden_fused_topk_cb(gating_output, correction_bias, top_k,
                         scoring_func="sigmoid", renormalize=False):
    """Source cb semantics: biased key selection -> unbiased sigmoid weights."""
    assert scoring_func == "sigmoid"          # softmax + correction_bias 非法（host 校验）
    logits = gating_output.to(torch.float32)                    # fp32 计算域（与源一致）
    scores = torch.sigmoid(logits)                              # 未加偏置 σ
    biased = scores + correction_bias.to(torch.float32).unsqueeze(0)   # 选择键
    # 降序（与 NPU vsort descending、输出 id 排序一致）
    _, ids = torch.topk(biased, top_k, dim=-1, sorted=True)
    w = torch.gather(scores, 1, ids)                            # 输出未加偏置 σ
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), ids.int()
```

**校验口径**（三级）：
1. `topk_weights` 排序后 `assert_close(rtol=1e-3, atol=1e-3)`（harness 容差）；
2. `topk_ids`：**非 tie 输入下逐元素完全一致**（§8.3 给出保证条件与验证法）；tie 输入下退化为
   「无重复 + 每个选中专家在 `biased` 下的值 ≥ 第 K 大 `biased` 值 − slack」（浮点 slack 取 1e-4，
   与 `test_moe_fused_topk.py::_check` 同口径）；
3. 两输出 shape/dtype 正确（weights fp32、ids int32、`(T,K)`）。

### 8.2 tie-break 方案与验证方法（用户重点告诫）

- **确定性方案**：
  - **选择键唯一确定**：`ids = argtopK_↓(σ(x)+b)`。当 `biased` 中第 K 与第 K+1 名**严格不等**时，id **唯一定义**，
    NPU 与 torch golden **逐元素一致**（D-3 探针 + `verify_equiv.py` 实证）。
  - **等值（tie）时**：`T.vsort` 的下标归属**未文档化**（`docs/Tilelang.language/排序操作/T.vsort.md` §2.3-4
    「排序稳定性差异，验证按『值匹配 + 下标指向的值匹配』双重校验」）；源 GPU 用「最小专家号」、`torch.topk` ties
    归属亦未保证。**本设计不承诺** tie 时与 torch 逐元素一致，理由是**该差异在值语义上不可观测**：
    选中的 K 个 id 各自对应的 `biased` 值相同，输出权重同为 `σ` 值（ties 在 σ 域亦相等），故：
    - **可直接测的强断言**：`weights` 排序后 `assert_close`（ties 下权重集合完全相同）、`ids` 无重复、
      每个选中 id 的 `biased` 值 ≥ 第 K 大 `biased` 值（slack 内）。
    - **实现侧努力**：若 Stage 3 实测发现 `T.vsort` ties 取的是**最大**专家号而目标需要**最小**专家号
      （源语义），可在**不影响排序正确性**的前提下尝试把 `biased` 的 tie 打破（例如对相同 `biased` 值不应加扰动——
      只能在确定「ties 是否真的会出现在目标域」后决定；目标域为 `torch.randn` 连续输入，ties 概率 0）。
- **验证方法（三级，Stage 3 必做）**：
  1. **主验证**：T/K/E 与 manifest 一致、`gating_output = torch.randn(...)`（连续），1000+ seed 网格下
     `ids` **逐元素 `torch.equal`**；`weights` 排序后 `assert_close`。此为主契约。
  2. **近 tie 验证**：构造 `biased` 中 K/(K+1) 名间隔 ~1e-5/1e-6/1e-7 的用例，记录 ids 是否翻转；
     对照 `verify_equiv.py` 的 1-ULP 敏感性结论（0/200 翻转，最小间隔 2.277e-04）。
  3. **显式 tie 用例**（全零 logits、少量离散值）：用**集合 + 值有效性**口径断言（不逐元素比 id），
     并把 `T.vsort` 的 tie 归属实测记录下来（§9 R-1）。
- **依据**：源 tie-break = 最小专家号（源 L158 严格 `>` + L178–179/L189–190 `other_idx < l_best_idx`）；
  `torch.topk` ties 未文档化；`T.vsort` ties 未文档化（文档 §2.3-4）。**这三者不一致是已知事实**，
  本设计以「值语义 + 主契约（连续输入逐元素）」覆盖，等值场景按容差口径。

### 8.3 L0 门槛测试计划（必须通过）

| 用例 | shape | dtype | scoring / renorm | 观测点 |
|------|-------|-------|------------------|--------|
| L0-1 cb smoke | (16,4) K=2 | bf16 | sigmoid / True | manifest smoke 路径（`fused-moe-cb-smoke`）|
| L0-2 cb smoke fp16 | (16,4) K=2 | fp16 | sigmoid / True | fp16 升 fp32 路径 |
| L0-3 cb smoke no-renorm | (32,8) K=2 | bf16 | sigmoid / False | renorm 关闭路径（矩阵全覆盖）|
| L0-4 kimi-k2-smoke | (64,384) K=8 | bf16 | sigmoid / True | E=384（`test_moe_fused_moe_cb.py` kimi-k2-smoke）+ renorm |
| L0-5 kimi-k2-decode | (512,256) K=8 | bf16 | sigmoid / True | 无核内串行（num_logical=22≤40）|
| L0-6 kimi-k2-medium | (2048,256) K=8 | bf16 | sigmoid / True | 核内串行 `num_local_tasks=3` |
| L0-7 kimi-k2-prefill | (4096,256) K=8 | bf16 | sigmoid / True | 核内串行 `num_local_tasks=5` + T 尾块 |
| L0-8 非整除 E=33 | (32,33) K=4 | bf16 | sigmoid / True | 无 padding 路径 |
| L0-9 非整除 E=100 | (32,100) K=8 | fp32 | sigmoid / False | 无 padding + fp32 原域 |
| L0-10 K=1 | (512,256) K=1 | bf16 | sigmoid / True | 最小 K |
| L0-11 大偏置 | (32,256) K=8 | fp32 | sigmoid / True | `bias *= 5`（选择键被偏置主导）|
| L0-12 负偏置 | (32,256) K=8 | fp32 | sigmoid / True | `bias` 含较大负值 |

> 完整分层套件（L1 功能覆盖 / L2 边界 / Boundary 极值）交由 `tilelang-op-develop`；本设计列 L0 门槛即可。
> **Boundary 观测（建议）**：全零 logits（ties）、`bias` 全相等、T 极小（4/5）、E=1、近 tie（1e-6）。
> **`scoring_func="softmax" + with_correction_bias=True` 必须报错**（host 层负向用例）：L0 之外在 op 层单测覆盖。

### 8.4 精度标准

按 harness 容差（`test_moe_fused_moe_cb.py`）：端到端 `rtol=atol=1e-2`（复合算子）；
路由子算子权重（本单元）用 `rtol=atol=1e-3`（对照 sibling `test_moe_fused_topk.py`）。
fp32 计算域 + 输出 fp32；bf16 经 fp32 升域（D-3 探针实测权重误差 ≤2.98e-08，`verify_equiv.py` fp32 仿真
max_rel 2.377e-07，远优于 1e-3）。**`topk_ids` 精度目标 = 非 tie 输入下逐元素完全一致**（§8.2）。

---

## 9. 风险点与注意事项

| 编号 | 风险 | 影响 | 缓解 / 依据 |
|------|------|------|------------|
| R-1 | **`T.vsort` tie-break 与 torch golden 不同**（用户重点告诫）| ties 时 `topk_ids` 元素顺序/归属可能不同 | **方案确定**：选择键唯一确定 → 非 tie 下逐元素一致（D-3 + `verify_equiv.py` 实证）；tie 下按**值语义 + 集合 + slack** 口径（§8.2）。目标域 `torch.randn` 连续输入 ties 概率 0；显式 tie 用例记录 `T.vsort` 实测归属 |
| R-2 | **`correction_bias` 含 `±inf` 的语义** | `σ(x)+b` 可能得 `±inf`/`NaN`（`-inf + b` 或 `inf + b`），排序行为未文档化 | 源文档未约束 b 范围（源仅要求 fp32）；本设计视 `±inf` 的 b 为**域外输入**（与 NaN logits 同类），记入风险，L2 观测；op 层不额外校验 |
| R-3 | **softmax + correction_bias 非法组合的校验归属** | 若 host 未拦，则 `scoring_func="softmax"` 会静默走 sigmoid 语义 | **归 host（op 层构造期）**：`FusedTopKOp.__init__` 已抛 `ValueError`；kernel 侧不重复（保证 kernel 语义单一）|
| R-4 | **尾块 slice 的 `T.vsort`/`T.gather`（`real < BM`）未实机验证** | 尾块可能编译/取值异常 | D-3 探针用满行（`real==BM`）验证；Stage 3 必须加最小尾块用例（如 T=25, BM=24）实测；回退策略：尾块可 padding 行再裁（不改变输出） |
| R-5 | **UB 容量近阈值 + 编译间歇性 UB 判定不稳** | 偶发 `ub overflow` 致编译失败（非错值）| **实测异常**：同一 TIR 在 (32,128)（大缓冲 65KB）曾一次报 `requires 3194880 bits while 1572864 bits available`，**复跑通过**（D-3 P3）。缓解：① `UB_BUDGET=96KB` 保守（(32,256)=131KB 仍可编译，安全裕度 ≥35%）；② 编译失败可 `resource.setrlimit(RLIMIT_STACK,...)` + 有界重试（CG-2026-0033 同款 workaround）；③ 缩 `BM` 降 UB。**未文档化假设**：UB 判定不稳的根因未定（疑 auto-multi-buffer 决定非确定，参照 CG-2026-0002/CG-2026-0033 记法）|
| R-6 | **`T.vsort`/`T.gather` 单条成本未标定** | roofline 发射项可能显著低估（本算子发射/选择主导）| 属**运行期吞吐画像**，交 Stage 4 msprof 实测（§1.6.0 roofline 未实证项）；**不作为性能目标写入设计**（用户告诫 #2）；不影响主选裁决。**Ratio 口径**：本算子向量受限、Ratio 天花板受分母口径限制（§1.6 开头），评估用 Mem/BW 利用率 + `vec_ratio`/`mte2_ratio` |
| R-7 | **近 tie / fp32 ULP 导致 id 翻转** | 极小间隔下 NPU σ 实现与 torch 差 1 ULP 可能翻选择 | `verify_equiv.py` 1-ULP 敏感性 **0/200 翻转**，实测最小 K/(K+1) 间隔 2.277e-04 ≫ ULP(~1e-07)；若 Stage 3 出现翻转，用 `_ID_SCORE_SLACK` 口径判定（值等价）|
| R-8 | **NaN 输入的 vsort/gather 行为未定** | NaN logits/bias 语义未文档化 | 非 manifest 输入域；记入本风险，L2 观测 |
| R-9 | **工具链陷阱复发** | 编译 SIGSEGV / 错值 | 已按 §3.5 规避（不用 `alloc_local`/标量 `if_then_else`/`>>`/标量消费向量产出/`Parallel` 直写 GM/`T.vcmp`）；guard 仅 block 级 |
| R-10 | **非对齐 K 尾列 `T.copy`（K=2，8B）** | 尾轴 <32B | K 固定小；Stage 3 L0-10（K=1）与 L0-1（K=2）验证；D-3 P2-D K=4 已通过；必要时 K 切片 + 元素级写 |
| R-11 | **`T.gather` indices 语义的文档/签名冲突** | `docs` §2.1 标 int32 tensor（runtime）但 `inspect.signature` 注解 `Union[list, tuple]` | **本设计已实机消解**：D-3 探针用 `T.vsort` 产出的 runtime int32 张量作 indices，编译并数值正确；`testing/npuir/indexing_ops/test_slice_gather_dev.py` 亦用 GM 载入的 int32 张量。仍保留为风险（工具链升级后需重验）|

---

## 10. 交付清单

| 项 | 路径 | 状态 |
|----|------|------|
| 设计文档 | `examples/fused_topk_cb/_fused_topk_kernel_cb/DESIGN.md` | 本文件 |
| 等价性机器验证（D-1）| `examples/fused_topk_cb/_fused_topk_kernel_cb/verify_equiv.py` | 已执行（65 例 ALL EQUIV_PASS）|
| 设计期探针 P2（D-3，完整路径）| `examples/fused_topk_cb/_fused_topk_kernel_cb/history_version/design_probe_cb_path.py` | 已执行（ids 逐元素一致，w ≤2.98e-08）|
| 设计期探针 P3（D-3，UB 容量包线）| `examples/fused_topk_cb/_fused_topk_kernel_cb/history_version/design_probe_ub_sweep.py` | 已执行（E≤384 全通过 + 一次 UB 异常）|
| 设计期探针 P0（D-3，原语隔离）| `examples/fused_topk_cb/_fused_topk_kernel_cb/history_version/design_probe_isolation.py` | 已执行（`vsort+gather` / `vbrc+vadd` 均通过）|
| 源算子 | `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`（Cb 分支）、`.../ops/moe/fused_topk.py` | 解读对象 |
| NPU 集成目标（conductor 后续集成本单元）| `examples/TileOPs/tileops/kernels/moe/fused_topk/fused_topk.py`、`examples/TileOPs/tileops/ops/moe/fused_topk.py`、`examples/TileOPs/tileops/ops/moe/fused_moe.py` | 已存在（本 Stage 不改）|
| 规格 | `examples/TileOPs/tileops/manifest/moe.yaml`（`FusedTopKOp` + `FusedMoeFwdCbFwdOp`）| 已存在 |
| 参考实现 / 测试 | `examples/TileOPs/tests/ops/test_moe_fused_moe_cb.py` | 已存在 |
| bench | `examples/TileOPs/benchmarks/ops/bench_moe_fused_moe_cb.py` | 已存在 |
| 同族实现（topk 标准变体）| `examples/fused_topk/_fused_topk_kernel/_fused_topk_kernel.py`、`.../DESIGN.md` | 已存在（`[PRECISION_PASS]`）|
| 同族实现（gather）| `testing/npuir/indexing_ops/test_slice_gather_dev.py`、`test_slice_gather_exp.py`、`docs/Tilelang.language/索引与元素操作/T.gather.md` | 已存在 |
| 同族实现（vsort/行归约/persistent）| `testing/npuir/sort_ops/test_vsort_dev.py`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py`、`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` | 已存在 |
| 陷阱规避参考 | `examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md` | 已存在 |

---

## 11. 未决项 / 需 conductor 或用户决策

1. **R-5 的 UB 判定不稳**：建议 conductor 允许 Stage 3 以「缩小 `BM`」或「有界重试」处置；若 Stage 4 复现
   且确认根因，考虑登记 capability-gap（**本 Stage 未触发能力缺口**——算法可编译可运行，D-3 已实证）。
2. **R-1 tie-break 与用户「逐元素一致」要求的边界**：本设计给出「连续输入逐元素一致 + ties 值语义口径」的确定方案；
   若 conductor/用户要求 **ties 下也逐元素一致**，需 Stage 3 探针刻画 `T.vsort` tie 归属后再定（可能需承认值语义口径）。
3. **renormalize 边界**：Cb 路径调用点恒 `renormalize=True`（manifest `FusedMoeFwdCbFwdOp` 全 workload）；
   本设计仍实现 `renormalize=False/True` 两分支，但**仅在 True 路径有 manifest 覆盖**（False 由 L0-3 内部覆盖）。

---

