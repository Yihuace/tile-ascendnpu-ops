# _batch_norm_bwd_split_kernel 算子设计文档

> **revision v1**（修订上一版 `history_version/design_v0.md`；相对 v0 的关键调整见 §0.0，修订日志见 §11）。
> 迁移任务（harness 模式）：BatchNormBwdOp 的拆分核（一个通道被切成 `splits` 段跨多个 block 归约：分块部分和 → 合并（finalize）→ 逐元素映射 grad_x）。
> 源算子（GPU 原始源码）：`TileOPs-main/src/tileops/kernels/norm/batch_norm.py` 的 `_batch_norm_bwd_split_kernel`（L1795）+ host 类 `BatchNormBwdSplitKernel`（L2044）。
> Stage 0 提取件（本函数迁移的精确源）：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（内含 `_batch_norm_bwd_split_kernel`，返回 `(stats, finalize, apply)` 三个 jit 工厂三元组）。
> NPU wrapper：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的 `BatchNormBwdSplitKernel`；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`；manifest `examples/TileOPs/tileops/manifest/norm.yaml`（`BatchNormBwdOp`）；test `examples/TileOPs/tests/ops/test_batch_norm_bwd.py`；bench `examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`。
> 可行性探针（Stage 0，真机 PASS，见 `examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`）：验证了 strided 逐通道 plane gather、双累加器两步 `T.reduce`、`T.vbrc` 标量广播、fp16/bf16 fp32 中转链、one-core-per-channel 与超大 S 的核内 S-tiling。
> 本函数补充设计探针（D-3，真机 PASS，2026-10-14）：`examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py` —— 验证了「`c = t // splits` / `sp = t % splits` 的 floordiv/mod 通道-段索引可用于 `T.copy` 的中间轴 strided 切片」「分块部分和写 GM workspace `[C, splits]`（标量 slice 写）」「finalize 二阶段合并（无 atomics）」「逐通道 apply」三段结构，真机 `max_abs_err grad_x=9.5e-07`（fp32，S=250 非整除尾段）。
> 工具链版本戳：tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3（`NPUUtils.get().get_aicore_num()` 实查 = 20，纯 Vector ×2 = 40）+ torch 2.7.1+cpu + torch_npu 2.7.1.post4。
> 同族参考（已完成）：宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（one core per channel 的 held/tiled 归约）与本任务其 `RETROSPECTIVE.md`（可迁移教训）。

## 0. 源算子解读与迁移分析

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `.agents/skills/tilelang-op-design/references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.0 相对上一版的关键调整（v0 → v1）

> 本版为 v0 的修订版（上一版：`history_version/design_v0.md`）。触发：Stage 2 `REVIEW.md`「结论: 不通过」——**3 个阻塞项**（apply 的 S-子块循环上界复用 stats 的 `n_tile`，使 `splits>1` 时 apply 只覆盖前 `ceil(chunk/bS)·bS` 列；段 extent `real=T.min(chunk,S-s0)` 在 `splits>ceil(S/chunk)` 时为 0/负；UB 预算只对小 N 成立、缺 N 维 tiling）+ **3 个建议项**。本版保留 v0 全部通过项（三段结构、GM workspace + finalize 无 atomics、S 轴仿射段、两步 `T.reduce`、fp16/bf16 fp32 中转链、分布式 epilogue O1/O2、D-1 `verify_equiv` `EQUIV_PASS` 21 案例、§1.6 调研与向量化轴决策、`apply` 的 `out_idx=[-1]` 单输出返回序），仅作下述定点修正。

**阻塞项 1（apply 子块数复用 stats）根因**：v0 工厂层只定义一个 `n_tile = ceil(chunk/bS)`（`chunk=ceil(S/splits)`），stats 与 apply 共用。但 stats 的段长是 `chunk`，apply 须覆盖整张平面（长度 `S`）；`splits>1` 时 `ceil(S/bS) ≠ ceil(chunk/bS)`，apply 只跑到 `ceil(chunk/bS)·bS` 列。独立复算 large-spatial：`S=262144, splits=4, bS=2272 → chunk=65536`，共享 `n_tile=29`，apply 仅覆盖 `29·2272=65,888` 列（≈S/4），`grad_x[:, c, 65888:]` 永不写出；且与 §4.5/§9.3 正文的 `n_tile=116` 自相矛盾。

**v1 修正（apply 用独立的 `n_tile_S_apply`）**：工厂层拆为两个常量——`n_tile_S_stats = ceil(max_seg/bS)`（stats 用，`max_seg` 为最长段的 S 长度）与 `n_tile_S_apply = ceil(S/bS)`（apply 用，覆盖全 S）；§3.2/§3.3/§5.2/§6.1 同步。apply 的 `real = T.min(bS, S-s0)` 不变。新增 L0-13 显式比对整张 `grad_x` 门住。

**阻塞项 2（段 extent 空段/负值）根因**：v0 段定义 `s0=sp·chunk`、`real=T.min(chunk, S-s0)`、`chunk=ceil(S/splits)`。`splits = min(_split_seed, S)` 不保证 `splits ≤ ceil(S/chunk)`；当 `splits > ceil(S/chunk)` 时存在任务 `s0 ≥ S`，`real ≤ 0`（反例 `S=5, splits=4 → chunk=2 → sp=3: real=min(2,-1)=-1`）。负 extent 传入 `T.copy`/`T.reduce` 为非法切片；v0 §5.4「cap 到 S 保证 `real ≥ 1`」的论断为假。

**v1 修正（余额分配的均衡切段，保证 `real ≥ 1`）**：段边界改为 `dist = S//splits`、`rem = S%splits`，`s0 = sp·dist + T.min(sp, rem)`、`s1 = (sp+1)·dist + T.min(sp+1, rem)`、`real = s1 - s0`。因 `splits ≤ S ⇒ dist ≥ 1`，每段 `real ∈ {dist, dist+1} ≥ 1`，恰 `splits` 个非空段且恰好覆盖 `[0, S)`（`s1(last)=S`）。段内子块用 `r2 = T.min(bS, s1-ss)` 并以 `if r2 > 0:` 守卫（尾子块可能空），杜绝负/零 extent。§5.4/§9.1 的错误论断删除并改写；新增 L0-11 门住。

**阻塞项 3（大 N UB 溢出）根因**：v0 `bS = align_down(163_840//(N·BPE), vw)` 只在 `N·BPE ≤ 163_840` 时为正；`N` 更大时（如 split 服务区内 `(8192,128,8)` fp16，`L=65536 ≥ _SPLIT_MIN_L`）商为 0、`bS` 被 `max(vw,·)` 抬回 8，`(N,bS)` 的 `g32/x32/prod/brc` 四路 fp32 必然超 UB。

**v1 修正（二维 `(bN,bS)` tiling）**：引入 N 维分块 `bN`：`N==1` 时 `bN=1`；`N>1` 时预留至少 2 行（使 `dims=[0]` 的 N 归约合法）后按 `bN = min(N, max(2, (UB_SAFE-SMALL)//(bS·BPE + ROW_OVERHEAD)))` 取。核内新增 `for nn in T.serial(n_tile_N)` 外层（`n0=nn·bN`、`rN=T.min(bN,N-n0)`）；行和累加器 `red/red2` 改 `(bN,1)`，N-块内跨 S-子块 `clear=False` 累加、N-块间以 `dims=[0], clear=False, size=[rN,1]` 累加进 `(1,1)` 的 `gb/gw`。§4.5/§5.2/§6/§9.1 同步；新增 L0-12 门住。

**建议项修正**：

- **建议 4（合并序表述）**：§0.5 行 #3 的「finalize 固定 split 序**串行合并**」改为「finalize 用 `T.reduce(dims=[1])` **硬件定序树**合并（确定性、无 atomics）；与源 `T.serial(k in splits)` 串行合并的差异仅分组顺序，fp32 舍入差在 §8.2 容差内」；§0.6 R3 同步。
- **建议 5（S=1 退化分支）**：`s_gt1=False` 分支由 `T.copy(g32, red)` 改为显式切片 `T.copy(g32[0:rN,0:1], red[0:rN,0:1])`（`S=1` 时 `bS = min(…, S) = 1`，形状按切片匹配）；`prod→red2` 同改；§5.4/§9.3 同步。
- **建议 6（L0 回归覆盖）**：§8.3 新增 L0-11（均衡段/空段回归 `(13108,128,5)` fp16）、L0-12（大 N UB 回归 `(8192,128,8)` fp16）、L0-13（apply 全 S 覆盖回归：`splits>1` large-spatial 显式比对整张 `grad_x`），分别门住问题 1/2/3。

**为何不会再犯同一错误（防复发机制）**：

- **子块上界单一来源约束（写入 §3.2 / §3.3 / §5.2 / §5.4 / §6.1）**：`stats` 与 `apply` 的 S-子块上界**必须分别**由 `n_tile_S_stats = ceil(max_seg/bS)` 与 `n_tile_S_apply = ceil(S/bS)` 定义，禁止共用任何单一 `n_tile`；凡出现「apply 尾列 `< S`」即视为缺陷，由 L0-13 显式断言整张 `grad_x` 门住。
- **段非空约束（写入 §3.3 / §5.2 / §5.4 / §9.1）**：段切分**必须**用余额分配（`dist/rem`）保证恰 `splits` 个非空段、`real ≥ 1`；禁止 `s0 = sp·chunk` + `min(chunk, S-s0)` 这类可能产生零/负 extent 的定义；子块用 `if r2 > 0` 守卫。任何「cap 到 S 即非空」的论断不得出现，除非附 `splits ≤ ceil(S/chunk)` 的证明。
- **UB 预算二维口径（写入 §4.5 / §5.2）**：预算公式**必须**同时含 `bN` 与 `bS`（`bN·bS·BPE + bN·ROW_OVERHEAD + SMALL ≤ UB_SAFE`）；禁止只按固定 `N` 的 `(N,bS)` 口径估算。大 N 服务区 shape 由 L0-12 门住。
- **口径单一来源约束（写入 §3.3 / §4.5 / §5.2）**：§4.5 表值必须由 §3.3/§5.2 公式可互推（同 BPE、同对齐、同 `bN`）；避免「表一套、公式一套」。

### 0.1 源算子语义（做什么）

**数学语义**（BatchNorm 训练反向，从训练前向保存的逐通道统计量反推梯度）。输入 `grad_out, x : [N, C, S]`、`weight, mean, rstd : [C]`（fp32），记 `L = N*S`、`x_hat[c,l] = (x[c,l] - mean[c]) * rstd[c]`，逐通道 `c`：

$$
\text{grad\_bias}[c] = \sum_{l<L} \text{grad\_out}[c,l],\qquad
\text{grad\_weight}[c] = \sum_{l<L} \text{grad\_out}[c,l]\cdot \hat{x}[c,l]
$$

$$
\text{grad\_x}[c,l] = \frac{w[c]\cdot \text{rstd}[c]}{L}\Big(L\cdot\text{grad\_out}[c,l] - \text{grad\_bias}[c] - \hat{x}[c,l]\cdot\text{grad\_weight}[c]\Big)
$$

**I/O 契约**：`grad_out, x` 形状 `[N, C, S]`、dtype ∈ {float32, float16, bfloat16}（manifest `DType[float32|float16|bfloat16]`）；`weight, mean, rstd` 形状 `[C]` fp32；输出 `grad_x : [N, C, S]` 同输入 dtype、`grad_weight : [C]` / `grad_bias : [C]` fp32。host 侧把任意 `(N, C, *spatial)` contiguous view 成 `(N, C, S)`（S=spatial 维乘积，无转置、非 in-place）；**输出 shape 从源码/op 回填确认**：host `forward` 末尾 `grad_x.view(x.shape)`（`examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`）。

**规约语义**：逐通道 `c` 沿展平轴 `N·S`（原生 layout 的 `{0, 2}` 轴）归约两个独立标量——`grad_bias = Σ grad_out`、`grad_weight = Σ grad_out·x_hat`。**通道轴 `C` 是中间轴**：逐通道平面 `x[:, c, :]` 在 GM 中是 strided 区域（行跨步 `C*S`，行内 stride=1），归约轴集合不是尾轴。**本变体的核心特征 = 跨 block 拆分累加**：一个通道被切成 `splits` 段，每段一个 block 产出部分和 `partial_sum[c, s]` / `partial_sum_xhat[c, s]`，再由一个 finalize block 合并成通道梯度；**累加顺序固定**（源 stats 用固定归约树，finalize 按 split 序串行求和——源码 L1834/L1899 注释明言「the channel gradients must not depend on a merge order」）。**源全程无 atomics**（`partial_sum[C, splits]` workspace + 独立 finalize kernel，非原子累加）。

**dtype 语义**：累加全程 fp32（`accum_dtype = "float32"`）；`grad_out/x` cast 到 fp32 参与计算；`grad_x` 单次舍回输入 dtype；`grad_weight/grad_bias/scale/shift/centered_coef` 输出 fp32。即「fp32 opmath + 单次舍回」。

**边界语义**：分段边界 `end = T.min(start + chunk, L)`（L1831），块内逐向量判 `head + num_per_thread <= end`、否则逐元素判 `l < end`（L1846/L1859）；`apply` 的平铺映射边界 `base + num_per_thread <= total` / `base + i < total`。NaN/Inf 自然传播（无特殊处理）；`N≥1, C≥1, S≥1`。**空张量 `L=0`**：GPU op 在 `numel()==0` 时早退，但本迁移使用的 **NPU op** `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py` 的 `forward` **无** `numel()==0` 守卫——该拦截须由 NPU op/wrapper 承接（集成期补偿，见 §9.1/§9.3）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`），不复刻 NPU 算法。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `grad_out` | 输入 | `(N, C, S)` | float16 / bfloat16 / float32 | 上游梯度；S=spatial 维乘积 |
| `x` | 输入 | `(N, C, S)` | same_as(grad_out) | 原始输入 |
| `weight` | 输入 | `(C,)` | float32 | 仿射缩放 gamma |
| `mean` | 输入 | `(C,)` | float32 | 前向保存的逐通道 batch mean |
| `rstd` | 输入 | `(C,)` | float32 | 前向保存的逐通道 reciprocal std |
| `grad_x` | 输出 | `(N, C, S)` | same_as(grad_out) | `apply` 末参数（源 `out_idx=[-1]`） |
| `grad_weight` | 输出 | `(C,)` | float32 | 通道梯度 1 |
| `grad_bias` | 输出 | `(C,)` | float32 | 通道梯度 2 |

**三段接口（源 `@T.prim_func` 参数序，逐字保持）**：

- `stats(grad_out, x, mean, rstd, partial_sum[C,splits], partial_sum_xhat[C,splits])`——GM workspace 两片部分和，由 host 预分配传入（无 `out_idx`）。
- `finalize(partial_sum, partial_sum_xhat, weight, rstd, grad_weight[C], grad_bias[C], scale_out[C], shift_out[C], centered_coef_out[C])`——合并 + 逐通道系数预折叠（无 `out_idx`）。
- `apply(grad_out, x, mean, scale, shift, centered_coef) -> grad_x`——源 `out_idx=[-1]`（仅 `grad_x` 自动分配）。

> **NPU wrapper 调用约定**（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的 `BatchNormBwdSplitKernel.forward`）：`stats, finalize, apply_ = self.stages`；依次 `stats()(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)`、`finalize()(partial_sum, partial_sum_xhat, weight, rstd, grad_weight, grad_bias, scale, shift, centered_coef)`、`grad_x = apply_()(grad_out, x, mean, scale, shift, centered_coef)`。即**三个工厂均以无参形式调用**（GPU 的 `splits/threads/num_per_thread/blocks` 调参参数全部在工厂闭包内烘焙，见 §3.3 注），workspace `partial_sum/partial_sum_xhat` 形状 `[C, splits]` 与 `scale/shift/centered_coef` 由 host 预分配。**三段必须使用同一个 `splits`**（否则 workspace 形状错位）。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂模式）：`_batch_norm_bwd_split_kernel(N, C, S, dtype)` → 返回 `(stats, finalize, apply)` 三个 `@tilelang.jit` 工厂三元组：`_stats_func(splits, threads, num_per_thread)`、`_finalize_func(splits, threads)`、`_apply_func(blocks, threads, num_per_thread)`。host `BatchNormBwdSplitKernel`（L2044）用 `_split_seed(call)` 选 `splits`，`forward` 预分配 workspace 与 `scale/shift/centered_coef` 后依次调用三段。

**计算步骤分解**（源码每个计算语句均已归入）：

| 步骤 | 计算（源码行） | 输入 | 输出 | 对应语义公式部分 |
|------|----------------|------|------|-----------------|
| S-P0 | `bc = bx // splits`；`start = (bx % splits) * chunk`；`end = min(start+chunk, L)`（L1828-1831） | grid id | 标量索引 | 分块切分 |
| S-P1a | 向量装载 `held_g[i] = grad_out[head//S, bc, head%S+i]`（L1847-1848） | GM grad_out | register `held_g` | 输入搬运 |
| S-P1b | 向量装载 `held_x[i] = x[...]`（L1849-1850） | GM x | register `held_x` | 输入搬运 |
| S-P1c | 累加 `g=cast(held_g); x_hat=(cast(held_x)-mean)*rstd; sums[j]+=g; sums_xhat[j]+=g*x_hat`（L1851-1855） | held_g, held_x | `sums/sums_xhat[1,threads]` fp32 | grad_bias / grad_weight 局部贡献 |
| S-P1d | 尾向量谓词分支 `if l < end` 逐元素（L1856-1865） | 同上 | 同上 | 非整除边界 |
| S-R1 | 线程内归约 `T.reduce_sum(sums, sum_result, dim=1)`（L1868-1869） | sums | `sum_result[1]` | 段内归约 |
| S-W1 | `if thread_binding==0: partial_sum[bc, bx%splits]=sum_result[0]`（L1870-1872） | sum_result | GM `partial_sum[C,splits]` | 写部分和 |
| F-P0 | finalize：`total += partial_sum[bc,k]`，`total_xhat += partial_sum_xhat[bc,k]`（L1899-1901） | workspace | `total/total_xhat` | **跨段合并** |
| F-W1 | `grad_bias[bc]=total; grad_weight[bc]=total_xhat`（L1902-1903） | total | GM 通道梯度 | 写 grad_bias/grad_weight |
| F-E0 | 系数预折叠 `scale=w*rstd; per=scale/L; shift=-per*total; centered_coef=per*total_xhat*rstd`（L1905-1910） | total, weight, rstd | GM `scale/shift/centered_coef[C]` | epilogue 系数 |
| A-P0 | apply：`base = bx*span + tx*num_per_thread`（L1939） | grid id | 标量索引 | 平铺映射 |
| A-P1 | `g[i]=grad_out[base+i]; v[i]=x[base+i]`（L1941-1944） | GM | register | 输入搬运 |
| A-P1b | `ch=(base//S)%C`（每向量持单通道）或 `ch=((base+i)//S)%C`（L1945-1946/L1956） | base | 标量 ch | 通道索引 |
| A-E1 | `o[i]=cast(scale[ch]*g+shift[ch]-centered_coef[ch]*(v-mean[ch]), dtype)`（L1947-1962） | g,v,系数 | register `o` | grad_x 公式 |
| A-W1 | 向量写 `grad_x[base+i]=o[i]`（L1963-1964） | o | GM grad_x | 输出搬运 |

**数据流与内存访问模式**（源硬件视角）：

```
stats (grid = C*splits blocks, threads 线程/块):
  GM[grad_out], GM[x] --per-thread vectorized strided (head//S, bc, head%S)--> REG[held_g, held_x]
    --(逐元素 cast + x_hat + 两个 += 到 sums[1,threads])--> REG[sums, sums_xhat]
    --T.reduce_sum(dim=1)--> REG[sum_result] --thread0--> GM[partial_sum[C,splits], partial_sum_xhat[C,splits]]
finalize (grid = 1 block, threads 线程):
  GM[partial_sum/...] --串行 k in splits--> REG[total, total_xhat] --tx--> GM[grad_bias, grad_weight, scale, shift, centered_coef]
apply (grid = blocks, threads 线程; 平铺映射):
  GM[grad_out], GM[x] --T.vectorized flat (base)--> REG[g, v] --(按 ch=(base//S)%C 取逐通道系数)--> REG[o] --T.vectorized--> GM[grad_x]
```

**循环与并行结构**：三段各自一个 grid。
- `stats`：`T.Kernel(C * splits, threads=threads)`——逻辑 block 数 = `C*splits`；每 block 一个 (通道, 段) 对；block 内 `T.Parallel(1, threads)` + `T.serial(step)` 协作段内扁平 `L` 区间；段内 `T.reduce_sum(dim=1)` 线程归约。
- `finalize`：`T.Kernel(1, threads=threads)`——单 block；`T.serial(ceildiv(C, threads))` 每线程一通道；`T.serial(splits)` 串行合并。
- `apply`：`T.Kernel(blocks, threads=threads)`——逻辑 block 数 = `ceil(N*C*S / (threads*num_per_thread))`；平铺 flat 映射，向量持单通道时 `ch=(base//S)%C` 否则逐元素 `ch=((base+i)//S)%C`。

**host 侧逻辑**：`BatchNormBwdSplitKernel.__init__` 计算 `splits = _split_seed(n, c, spatial)`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`，`_SPLIT_TARGET_BLOCKS=512`、`_SPLIT_MAX_C=1024`、`_SPLIT_MIN_L=65536`）；`forward` 做 `contiguous()`、预分配 `grad_weight/grad_bias/partial_sum/partial_sum_xhat/scale/shift/centered_coef`、依次调用三段。op 层 `BatchNormBwdOp.forward` 把任意 `(N,C,*spatial)` view 成 `(N,C,S)`、`contiguous()`，返回 `grad_x.view(x.shape)`。**无数据内容预处理**（纯 shape 元数据视图）。`@functools.lru_cache(maxsize=32)` 工厂按 `(N,C,S,dtype)` 特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 跨 block 通道拆分（C*splits blocks） | 小 C 大 S 时提升并行度、摊薄单块长度 | 通道切 `splits` 段各一 block；部分和落 workspace | CTA 调度（大 grid） | 硬件弱相关（并行组织可重设计） |
| 2 | 部分和 workspace + finalize 合并 | 跨 block 归约 | `partial_sum[C,splits]` + 单 block 串行求和 | 无（纯数据流/算法层） | 可移植 |
| 3 | 固定归约顺序（reduce tree + 串行合并） | `grad_weight/grad_bias` 逐次运行确定 | 段内固定树 + finalize 固定 split 序 | 无（纯算法层） | 可移植（意图） |
| 4 | two-accumulator 融合归约（sums / sums_xhat） | 单遍同时得两个和 | 同一遍两个 `+=`（L1854-1855） | 无（纯算法层融合） | 可移植 |
| 5 | 段内线程协作 + `T.Parallel(1,threads)` | 块内并行度 | 线程各自扫段内扁平区间 | warp/CTA 线程 | 硬件强相关 |
| 6 | 每线程寄存器累加（sums 驻留） | 消中间往返 | `alloc_fragment([1,threads])` | 寄存器堆 | 硬件强相关 |
| 7 | 平铺 apply（flat map） | 全张量一次映射，通道索引按式算 | `base` 平铺 + `ch=(base//S)%C` | 无（数据并行） | 模式可移植 |
| 8 | epilogue 系数预折叠（scale/per/shift/centered_coef） | 逐元素只做 2 mul + 1 add/sub + 1 sub | finalize 逐通道算一次 | 无（纯算法层） | 可移植 |
| 9 | fp32 opmath + 单次舍回 | 精度（fp32 累加，输出 dtype） | `T.cast(...,"float32")` 累加 + 赋值截断 | 寄存器 | 硬件强相关 |
| 10 | 整除快路径 / 逐向量谓词（vector_holds_one_item + `if head+npt<=end`） | 段内向量边界一趟判 | 编译期分支 + 逐元素守卫 | 无（模式可移植） | 模式可移植 |
| 11 | 向量化访存（`T.vectorized`） | 访存带宽 | 每线程 `num_per_thread` 连续元素一次读写 | coalesced | 模式可移植 |
| 12 | lru_cache 工厂 + per-shape JIT 特化 | 摊编译开销 | `functools.lru_cache(maxsize=32)` | 无（host 层） | 可移植 |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：`.agents/skills/tilelang-op-design/references/migration-analysis.md` §5 三层模型 + §5.3 映射表；`examples/` 佐证；`docs/` 条目；pattern-library 实测条目；探针 `examples/batch_norm_bwd/probe/PROBE.md` + 本函数设计探针 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py`（真机 PASS）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（逐通道两标量归约 + grad_x 公式 + fp32 opmath + 单次舍回） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| 跨 block 通道拆分（#1） | 算法/优化 | CTA 调度 | 有（一维 grid + 物理核数） | **重新设计** | → §0.6 R1：一维 persistent `T.Kernel(min(C*splits, 40))` + grid-stride（`migration-analysis.md` §5.3「block×thread 两级并行 → 一维 block」） |
| 部分和 workspace + finalize 合并（#2） | 算法 | 无（纯数据流） | 有（GM workspace + 二阶段 kernel） | **保留/等价替换** | → §0.6 R3：GM `partial_sum[C,splits]` workspace + 独立 finalize kernel；`migration-analysis.md` §6.2「跨 block：workspace + 核间同步，或拆两阶段 kernel」——本设计取**两阶段 kernel、无 atomics**（保 merge 顺序确定性） |
| 固定归约顺序（#3） | 算法 | 无 | — | **保留** | 段内 `T.reduce` 每核定序 + finalize 用 `T.reduce(dims=[1])` **硬件定序树**合并（确定性、无 atomics；与源 `T.serial(k in splits)` 串行合并仅分组顺序不同，fp32 舍入差在 §8.2 容差内）（探针 PROBE §6） |
| two-accumulator 融合归约（#4） | 算法 | 无 | — | **保留** | 同遍两个 `T.reduce`（探针真机验证） |
| epilogue 系数预折叠（#8） | 算法 | 无 | — | **保留** | finalize 逐通道算一次（§1.6.1 O2 承接） |
| 段内线程协作 + `T.Parallel(1,threads)`（#5） | 优化 | warp/CTA 线程 | 无直接等价 | **重新设计** | → §0.6 R2：段内改为「lane 向量化逐元素累加 `(bN,bS)` 分块 + 两步 `T.reduce`」 |
| 每线程寄存器累加（#6） | 优化 | 寄存器堆 | 有（UB 192KB） | **等价替换** | `T.alloc_ub`（Expert，`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；累加器 `(N,1)` fp32 驻留 UB |
| 平铺 apply + `ch=(base//S)%C`（#7） | 优化 | 数据并行 | 有（逐通道 plane） | **重新设计** | → §0.6 R4：逐通道平面 epilogue（`T.copy` strided plane + `T.vbrc` 广播），避免逐元素标量通道索引 |
| 逐向量谓词 `if head+npt<=end` / `l<end`（#10） | 优化 | 无 | 有更优形态 | **重新设计** | → §0.6 R5：`T.min` + src/dst 双显式 slice（I/O）+ reduce `size` 截断（归约） |
| 向量化访存 `T.vectorized`（#11） | 优化 | coalesced | 有（slice copy + 向量宽度） | **等价替换** | `T.copy` 双显式 slice（GM↔UB，`docs/Tilelang.language/内存操作/T.copy.md`）；向量宽 fp16/bf16 ×8、fp32 ×4 |
| fp32 opmath + 单次舍回（#9） | 优化 | 寄存器 cast | 有（UB fp32 + vcast） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；bf16 无 Vector 算术 → 中转必需） |
| `T.Kernel(C*splits, threads=)` + `threads`/`num_per_thread` | 优化 | CUDA 网格/线程 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果；并行交给 Vector lane（§1.6.2/§1.6.3） |
| `apply` 的 flat view（`T.Tensor([total], dtype, data)` 别名） | 优化 | 平铺地址 | 无必要 | **舍弃** | NPU 走逐通道 plane epilogue（§0.6 R4）；不再有 `T.reshape`/int32 溢出问题 |
| `T.vectorized`/`T.get_thread_binding`/`T.Parallel(1,threads)` | 优化 | CUDA 线程 | 无 | **舍弃** | 后端不适用；归约改 `T.reduce`（§0.6 R2） |
| lru_cache 工厂 + JIT 特化（#12） | host | 无 | — | **保留** | host 层硬件无关 |
| 三 kernel 选核策略（wide/split/general；源 `call.sm_count`） | host | CUDA `sm_count` | 有（物理核数） | **等价替换** | wrapper 已移植 `_AI_CORE_COUNT=40` 与 `_split_seed`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`） |

### 0.6 NPU 算法重设计

**重设计项 R1: 拆分并行结构（C*splits blocks 超发 → 一维 persistent 分核）**

- **源方案**：`T.Kernel(C * splits, threads=threads)`——逻辑 block 数 = `C*splits`，依赖 CUDA 运行时调度到 SM；块内 threads 协作段内扁平区间。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(C*splits, 40)`（40 = 20 AICore × 2，纯 Vector 翻倍，§5.5 实查）；核内 `for i in T.serial(num_local_tasks)` grid-stride 映射任务 `t = i*num_kernels + cid`，`if t < C*splits` 守卫；任务到 (通道, 段) 的映射 `c = t // splits`、`sp = t % splits`（floordiv/mod 索引，本函数设计探针真机验证可用于 `T.copy` 中间轴切片）；段内先 `T.serial(n_tile_N)` 遍历 N-子块（v1 新增，§0.0 阻塞项 3），再 `T.serial(n_tile_S_stats)` 遍历 S-子块；段 extent 用余额分配 `real = s1 - s0 ≥ 1`（v1，§0.0 阻塞项 2）。`num_local_tasks`/`n_tile_N`/`n_tile_S_stats` 为 host 编译期常量（静态边界）。意图承接：消除超发逻辑内核的串行调度开销（`.agents/skills/_shared/standards/core-split-strategy.md` §1 极大规模分支）。参照同族 `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` persistent 分核。
- **语义保持论证**：`C*splits` 逻辑任务只是源 grid 的重新映射；`(c, sp)` 与源 `bc=bx//splits`、`bx%splits` 一一对应，不改变任何任务的计算语义。

**重设计项 R2: 段内归约（per-thread 累加 + 线程树 → lane 向量化累加 + 两步 `T.reduce`）**

- **源方案**：每线程持 `num_per_thread` 元素向量累加到 `sums[1,threads]`，再 `T.reduce_sum(dim=1)` 线程归约；`thread 0` 写出。
- **NPU 新算法**：段区间 `[s0, s1)`（余额分配 `s0 = sp·dist + T.min(sp, rem)`、`s1 = (sp+1)·dist + T.min(sp+1, rem)`、`real = s1-s0 ≥ 1`）内，外层 `T.serial(n_tile_N)` 遍历 N-子块（`n0 = nn·bN`、`rN = T.min(bN, N-n0)`；v1 N-tiling，§0.0 阻塞项 3），内层 `T.serial(n_tile_S_stats)` 遍历 S-子块（`ss = s0 + tt·bS`、`r2 = T.min(bS, s1-ss)`，`if r2 > 0:` 守卫）：`T.copy` 双 slice 载入 `(rN,r2)` + `T.vcast` fp32 中转 + `T.vbrc` 广播 + `T.vsub/T.vmul` 算 `x̂`、`prod = g·x̂`；两路 `T.reduce(..., dims=[1], clear=False, size=[rN, r2])` 将行和累加进 `red/red2 (bN,1)`（跨 S-子块累加）；每个 N-块末 `T.reduce(red, gb, dims=[0], clear=False, size=[rN, 1])` 把该 N-块行和累加进 `(1,1)` 的 `gb/gw`（跨 N-块累加；`N=1` 时工厂层跳过退化层改 `T.copy`）；段末 `T.copy(gb, partial_sum[c, sp:sp+1])` 标量 slice 写 workspace。
- **语义保持论证**：归约是加法结合律下的同一数学和；NPU 用固定树序（`hivm::VReduceOp`，`docs/Tilelang.language/规约操作/T.reduce.md` §3），每核内定序、N-块/S-子块按固定 `serial` 序分组累加、finalize 用 `T.reduce(dims=[1])` 定序树 → 通道梯度确定（探针 PROBE §6）。累加分组与 GPU 不同 → fp16/bf16 舍入差异在 op 容差内（§8.2，探针最坏 max_abs_err 4.883e-04；本函数设计探针 fp32 grad_x 9.5e-07）。N-tiling 只是把行归约按 `n0` 分组后再按全序累加，结合律下等价。

**重设计项 R3: 跨段合并（finalize kernel + workspace，无 atomics）**⭐

- **源方案**：独立 `_finalize_func` kernel（`T.Kernel(1)`），每线程一通道，`T.serial(splits)` 串行累加 `partial_sum[bc, k]`/`partial_sum_xhat[bc, k]`，写通道梯度与系数。
- **NPU 新算法**：**采用「部分和 workspace `partial_sum[C,splits]` / `partial_sum_xhat[C,splits]`（GM）+ 第二阶段 finalize kernel」的合并方式；全程不使用 atomics**。finalize 为 `T.Kernel(min(C,40), is_npu=True)` persistent：每任务载入 `(1, splits)` 行 `pw/pwx`（`T.copy` slice），`T.reduce(pw, tot, dims=[1])` 合并（`splits=1` 时工厂层跳过退化层、直接 copy），写 `grad_bias/grad_weight`；再 `T.vmul` 预折叠 `scale = w·rstd`、`per = scale·L_inv`、`shift = −per·gb`、`centered_coef = per·gw·rstd` 写三片系数。
- **为何不用 atomics**：源注释要求 `grad_weight/grad_bias` **不依赖合并顺序**（L1834/L1899），`T.atomic_add`（`docs/Tilelang.language/原子操作/T.atomic_add.md`，fp32 支持）的累加顺序不确定会破坏该确定性；workspace + 固定序 finalize 既确定又无原子开销。源亦全程无 atomics。
- **语义保持论证**：部分和求和的结合律重排 = 展平 `L` 求和；finalize 用 `T.reduce(pw, tot, dims=[1])` 的**硬件定序树**合并（确定，无 atomics）——与源 `T.serial(k in splits)` 串行合并仅分组顺序不同，仍逐次运行确定。跨段合并的 fp32 舍入差在容差内（§8.2）。

**重设计项 R4: apply（flat map + 逐元素通道索引 → 逐通道 plane epilogue）**

- **源方案**：`T.Kernel(blocks)` 平铺 flat 映射，`base = bx*span + tx*num_per_thread`，按 `ch=(base//S)%C`（或逐元素）取逐通道系数。
- **NPU 新算法**：`T.Kernel(min(C,40), is_npu=True)` persistent 逐通道：外层 `T.serial(n_tile_N)` 遍历 N-子块、内层 `T.serial(n_tile_S_apply)`（`= ceil(S/bS)`，**独立于 stats**，覆盖全 S —— v1，§0.0 阻塞项 1）遍历 S-子块；载入 `grad_out[n0:n0+rN, c, s0:s0+real]` / `x[...]` plane，`T.vbrc` 广播 `mean/scale/shift/centered_coef`，逐元素 `grad_x = scale·g + shift − centered_coef·(x − mean)`，`T.vcast(rint)` 舍回 + `T.copy` 双 slice 写回。
- **语义保持论证**：逐通道 plane epilogue 与「flat map 按 `ch` 取系数」逐元素恒等（同一公式）；消除了逐元素标量通道索引（非仿射）带来的向量化障碍。

**重设计项 R5: 边界处理（逐向量谓词 → `T.min` + 双显式 slice + reduce `size` 截断）**

- **源方案**：`end=min(start+chunk,L)` + 逐向量 `if head+npt<=end` + 退出逐元素 `if l<end`。
- **NPU 新算法**：统一单路径——段 extent 用**余额分配** `s0 = sp·dist + T.min(sp, rem)`、`s1 = (sp+1)·dist + T.min(sp+1, rem)`、`real = s1-s0 ≥ 1`（v1，§0.0 阻塞项 2）；N-子块 extent `rN = T.min(bN, N-n0)`；S-子块 extent `r2 = T.min(bS, s1-ss)` 并以 `if r2 > 0:` 守卫（尾子块可能空，v1）；apply 侧 `real = T.min(bS, S-s0)`。装载/写出用 src/dst 双显式 slice；**归约同样须截断**：`T.reduce(dims=[1])` 传 `size=[rN, r2]`、`T.reduce(dims=[0])` 传 `size=[rN, 1]`，只对有效范围求和（否则尾子块/尾 N-块垃圾计入部分和）。`n_tile_N`/`n_tile_S_stats`/`n_tile_S_apply` 核内 `T.serial` 边界静态。
- **语义保持论证**：越界元素不参与装载/写出，归约以 `size` 限定有效 extent，与源谓词掩码等价；见同族 `.agents/skills/tilelang-op-optimize/references/pattern-library/layout.md` 尾块归约形态与 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` §0.6 R3。

**重设计项 R6: dtype 路径（寄存器隐式 cast → 显式 `T.vcast(rint)` fp32 中转链）**

- **源方案**：`T.cast(..., "float32")` 上 cast + 赋值隐式截断回 dtype。
- **NPU 新算法**：fp16/bf16 输入：`T.vcast(stage, x32, round_mode="rint")` 升 fp32 → fp32 v-prefix 链 → `T.vcast(..., round_mode="rint")` 单次舍回；fp32 输入无 cast。**bf16 无 Vector 算术**（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 bf16 列为 ×）→ fp32 中转必需。
- **语义保持论证**：fp16/bf16→fp32 精确无损；fp32→目标 dtype 单次 RNE 舍回 = 源赋值截断语义；跨 dtype `T.copy` 的隐藏 VCast 不做 cast 融合（`docs/Tilelang.language/内存操作/T.copy.md` §3）。

**舍弃项的意图承接汇总**（不得静默丢弃，`migration-analysis.md` §6.4）：
- register 驻留（#6）意图 = 消中间往返 → UB 累加器 `(N,1)` 驻留承接；
- 线程协作/树归约（#5）意图 = 块内并行度 + 低延迟归约 → Vector lane 向量化 + `T.reduce` 硬件树承接；
- flat map（#7）意图 = 全张量并行映射 → 逐通道 plane 映射（等价并行度）；
- 谓词掩码（#10）意图 = 越界安全 → `T.min` + 双显式 slice + reduce `size` 承接。

### 0.7 标杆实现

- 源算子：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（Stage 0 提取件，迁移精确源，含 `_batch_norm_bwd_split_kernel`）；GPU 原始 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_batch_norm_bwd_split_kernel` L1795 + host `BatchNormBwdSplitKernel` L2044）。
- 参考实现 / 测试基准：`examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`（独立纯 PyTorch，fp32 中间量 + 舍回输入 dtype）——golden 以 §0.1 语义为依据移植此实现，不复刻 NPU 算法。
- manifest：`examples/TileOPs/tileops/manifest/norm.yaml` `BatchNormBwdOp` 条目（5 输入 3 输出；workloads smoke-fc/resnet-stage1/2/3/large-spatial；roofline `flops = 9*N*C*S`，`bytes = (3*N*C*S + 5*C)*elem_bytes`）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`（manifest 驱动）。
- 探针：`examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`（真机 PASS）；本函数设计探针 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py`（真机 PASS，三段结构 validate）。

---
## 1. 概述

### 1.1 算子名称

`_batch_norm_bwd_split_kernel`（BatchNormBwdOp 的拆分核变体；NPU 侧 = 分块部分和 + finalize 合并 + 逐通道 apply 三段）

### 1.2 功能描述

对 `(N, C, S)` 输入，逐通道沿展平轴 `N·S` 归约两个标量 `grad_bias = Σ grad_out` 与 `grad_weight = Σ grad_out·(x−mean)·rstd`，并据此逐元素写出 `grad_x`。本变体把每个通道切成 `splits` 段跨多个 block 做分块部分和，再由 finalize 阶段合并为通道梯度与 epilogue 系数，最后 apply 阶段逐通道映射 `grad_x`（NPU 重设计后算法与源差异见 §0.6/§1.6.0：一维 persistent 分核、lane 向量化 + `T.reduce` 替代线程树、GM workspace + finalize 二阶段合并（无 atomics）、逐通道 plane apply、fp32 中转链）。

### 1.3 数学公式

$$
\hat{x}[c,l] = (x[c,l] - \text{mean}[c])\cdot \text{rstd}[c],\qquad
\text{gb}[c] = \sum_{l<L}\text{grad\_out}[c,l],\qquad
\text{gw}[c] = \sum_{l<L}\text{grad\_out}[c,l]\cdot\hat{x}[c,l]
$$

$$
\text{grad\_x}[c,l] = \frac{w[c]\cdot\text{rstd}[c]}{L}\Big(L\cdot\text{grad\_out}[c,l] - \text{gb}[c] - \hat{x}[c,l]\cdot\text{gw}[c]\Big),\qquad L = N\cdot S
$$

其中两个和按 `splits` 段分块累加后合并（固定顺序）；`grad_x` 在 fp32 域按上式算得后单次舍回输入 dtype；`grad_weight/grad_bias` 为 fp32。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（中心化 `x̂` + 双标量归约 + 跨段部分和 + finalize 合并 + 逐通道 apply）。三段：

1. **stats（分块部分和）**：逻辑任务 `C*splits`，一维 persistent 分核（`min(C*splits,40)`）grid-stride；任务 `(c, sp)` 覆盖段 `[s0, s1)`（**余额分配** `s0 = sp·dist + T.min(sp, rem)`、`s1 = (sp+1)·dist + T.min(sp+1, rem)`、`real = s1-s0 ≥ 1`，`dist=S//splits`、`rem=S%splits`）。段内**外层** `T.serial(n_tile_N)` 遍历 N-子块（`n0=nn·bN`、`rN=T.min(bN,N-n0)`，v1 N-tiling）、**内层** `T.serial(n_tile_S_stats)` 遍历 S-子块（`ss=s0+tt·bS`、`r2=T.min(bS,s1-ss)`，`if r2>0`）：`T.copy` 双 slice 载入 `(rN,r2)` 的 `grad_out/x`，fp16/bf16 经 `T.vcast(rint)` 升 fp32；`x̂ = (x−mean)·rstd`（`T.vbrc` 广播）；`prod = g·x̂`；两路 `T.reduce(dims=[1], clear=False, size=[rN,r2])` 跨子块累加行和到 `red/red2 (bN,1)`；每 N-块末 `T.reduce(dims=[0], clear=False, size=[rN,1])`（`N>1`）把该 N-块行和累加进 `gb/gw (1,1)`；段末 `T.copy` 标量 slice 写 `partial_sum[c, sp]` / `partial_sum_xhat[c, sp]`。
2. **finalize（合并 + 系数预折叠）**：一维 persistent 分核（`min(C,40)`）；逐通道载入 `(1,splits)` 行，`T.reduce(dims=[1])`（`splits>1`）合并为 `gb/gw`；写 `grad_bias/grad_weight`；预折叠 `scale = w·rstd`、`per = scale·L_inv`、`shift = −per·gb`、`centered_coef = per·gw·rstd` 写三片系数。
3. **apply（逐通道 grad_x）**：一维 persistent 分核（`min(C,40)`）；逐通道**外层** `T.serial(n_tile_N)`、**内层** `T.serial(n_tile_S_apply)`（`=ceil(S/bS)`，独立于 stats，**覆盖全 S**）载入 `grad_out/x` plane 子块（`n0=nn·bN`、`rN=T.min(bN,N-n0)`、`s0=tt·bS`、`real=T.min(bS,S-s0)`），`T.vbrc` 广播 `mean/scale/shift/centered_coef`，`grad_x = scale·g + shift − centered_coef·(x − mean)`，`T.vcast(rint)` 舍回，`T.copy` 写回。

与源算法差异及来源：并行组织从「C*splits 超发 block + threads」改为「一维 persistent 分核 + 核内 serial」（§0.6 R1）；段内归约从「per-thread 向量累加 + `T.reduce_sum(dim=1)` 线程树」改为「lane 向量化逐元素累加 + 两步 `T.reduce`」（§0.6 R2）；跨段合并从「独立 finalize（同为独立 kernel）」改为「GM workspace + 二阶段 finalize（**无 atomics**，保确定性）」（§0.6 R3）；apply 从「flat map + 逐元素 `ch`」改为「逐通道 plane epilogue」（§0.6 R4）；边界从逐向量谓词改为「`T.min` + 双 slice + reduce `size`」（§0.6 R5）；dtype 从隐式 cast 改为显式 `T.vcast(rint)` 中转链（§0.6 R6）。**保留**源算法意图：跨 block 拆分、部分和 workspace + 合并、two-accumulator 单遍融合、固定归约顺序、epilogue 标量预折叠、fp32 opmath 单次舍回。

### 1.5 数据流图

```
stats 阶段（grid = min(C*splits, 40)，核内 grid-stride；段 [s0,s1) 余额分配）:
  for nn in serial(n_tile_N):                          # N-子块（v1）
   for tt in serial(n_tile_S_stats):                   # S-子块
    GM[grad_out] --T.copy(双slice,rN×r2)--> UB[stage(dtype)] --vcast(rint)--> UB[g32 (bN,bS) fp32]
    GM[x]        --T.copy(双slice,rN×r2)--> UB[stage(dtype)] --vcast(rint)--> UB[x32 (bN,bS) fp32]
    UB[x32] --vbrc(m)+vsub+vbrc(rstd)+vmul--> UB[x32 = x̂]·UB[g32] --vmul--> UB[prod]
    UB[g32]/UB[prod] --T.reduce(dims=[1],clear=False,size=[rN,r2])--> UB[red/red2 (bN,1)]
   UB[red]/UB[red2] --T.reduce(dims=[0],clear=False,size=[rN,1])--> UB[gb/gw (1,1)]
  UB[gb/gw] --T.copy(标量slice)--> GM[partial_sum[C,splits], partial_sum_xhat[C,splits]]
finalize 阶段（grid = min(C,40)）:
  GM[partial_sum/...] --T.copy(1×splits)--> UB[pw/pwx] --T.reduce(dims=[1])--> UB[gb/gw] --T.copy--> GM[grad_bias, grad_weight]
  UB[w1]·UB[r1] --vmul--> UB[scale]; --(*L_inv)--> UB[per]; --(·gb, ·−1)--> UB[shift]; --(·gw·rstd)--> UB[centered_coef] --T.copy--> GM[scale/shift/centered_coef]
apply 阶段（grid = min(C,40)；内层遍历全 S，独立于 stats）:
  for nn in serial(n_tile_N):
   for tt in serial(n_tile_S_apply):                   # = ceil(S/bS)，覆盖全 S（v1 修复）
    GM[grad_out]/GM[x] --T.copy(双slice,rN×real)--> UB[stage] --vcast(rint)--> UB[g32/x32 (bN,bS)]
    UB[g32],UB[x32] + UB[mean/scale/shift/centered_coef (1,1)] --vbrc 广播 + vmul/vadd/vsub--> UB[gx32]
    UB[gx32] --vcast(rint)--> UB[stage(dtype)] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环；三段间经 GM workspace partial_sum/partial_sum_xhat 与系数三片交接，无中间张量往返出 GM）
```

### 1.6 算法调研与优化分析 ⭐

> 调研对象：同一数学语义（逐通道 BN 反向，从 forward 保存的 mean/rstd 反推三梯度）的算法族。源算法（跨 block 拆分 + finalize 合并）只是基线候选之一。调研深度：**完整调研**（规约/统计类）。信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md` ALG-reduction 族命中行 + 本仓 examples（`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` 同族迁移、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` 尾块模式）+ pattern-library（`constants.md` CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C / CONST-capacity-910B2C）+ 源码 §0.3/§0.4 + 探针真机结论 + 结构判据。无互联网检索（本地源全覆盖四问所需候选）。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | 基线（源 `_batch_norm_bwd_split_kernel`） | 中心化 `x̂` + 双 `+=` 累加 + 跨段部分和 `[C,splits]` + finalize 串行合并 + flat apply | — | — | ✅（基线） |
| 1 | 分布式 epilogue + 倒数乘 | `gx = scale·g + shift − centered_coef·(x−mean)`，`scale=w·rstd`、`per=scale·L⁻¹`、`shift=−per·gb`、`centered_coef=per·gw·rstd` | 数学恒等（分配律；L 编译期常量取 `fl(1/L)`） | 归约遍省 `x̂` 复算、epilogue 每元素省 1 mul | ✅（主选，§1.6.1 O1/O2） |
| 2 | raw-moment `grad_weight` | `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)` | 代数恒等，但大均值小方差 fp32 **灾难性抵消**（D-1 判别段 A） | 归约遍省每元素 1 sub + 1 mul | ❌（数值否决，§1.6.1 判别段 A 机器证据） |
| 3 | one-core-per-channel（宽核形态） | 每通道一核、核内 S-tile 两扫描（`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`） | 数学恒等 | 消跨段合并（少 workspace 往返） | ❌（小 C 时并行度不足，见 R4；宽核函数已覆盖此形态） |
| 4 | 原子累加单部分和 | `T.atomic_add` 累加到 `partial_sum[C]` | 数学恒等 | 省 finalize 一趟 | ❌（累加顺序不确定，破坏源「merge 顺序无关」要求，见 R4） |
| 5 | online / Welford 变体 | 逐 tile 增量统计 | — | — | ❌（本反向的两和是输入无关函数的可加一阶矩，归约天然单遍；无 variance/max 可在线更新——见 R2） |

**R2 在线算法**：**无新增收益（等价于基线）。** 结构依据：BN 反向的 `mean/rstd` 是**前向输入**（非本算子计算），本算子只算两个加性一阶矩 `Σ grad_out` 与 `Σ grad_out·x̂`——任何分块/流式归约就是把这两和的求和项分组，本身就是「单遍 streaming reduce」，不存在需要重缩放（rescale）的 running 统计量（对比 online softmax 的 running max/sum）。源 stats kernel（L1842-1865）的分块累加即为该族既有 streaming 结构。因此「在线化」在本算子上退化为 R1#0/#1 的「分段部分和 + 合并」结构选择，不构成独立算法族。

**R3 复杂度对比**（每通道、`L = N·S`、`eb` = dtype 字节；FLOPs 量纲 = 向量操作数（含 cast 计数为 1 op/元素），无超越函数；GM Bytes 口径 = `grad_out` 读 + `x` 读 + `grad_x` 写 + workspace 往返 + `weight/mean/rstd` 与通道梯度小项 5C；扫描遍数 = 同一输入张量的 GM 重复读取次数）：

| 算法候选 | FLOPs（每通道） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 跨段拆分（=候选 0） | ≈9L（x̂ 2L + 双累加 3L + epilogue 4L；与 manifest roofline `flops=9·N·C·S` 一致） | 5·L·C·eb（stats 读 g+x，apply 再读 g+x；+workspace `2·C·splits`·4B） | grad_out 2、x 2 | `(bN,bS)`·BPE + `bN`(red/red2) + `[C,splits]` | **C·splits 并行**；跨核代价 = finalize 一趟（+workspace 往返） |
| 分布式 epilogue（主选，=候选 1） | ≈9L | 同基线 | 同基线 | 同基线 | 同基线 |
| raw-moment（候选 2） | ≈8L（省归约遍 x̂ 的 2L） | 同基线 | 同基线 | 同基线 | 同；**抵消风险** |
| one-core-per-channel（候选 3） | ≈9L | 5·L·C·eb（或 held 档 3·L·C·eb，当 `N·S` 入 UB） | grad_out 1–2、x 1–2 | `(N,bS)`（tiled）或整平面（held） | **C 并行**；无跨核；`C<40` 时并行度不足 |
| 原子累加（候选 4） | ≈9L | 4·L·C·eb + workspace 累加 | 2 | `[C]` | 需 atomics；**顺序不确定** |

代表 shape（large-spatial，即 split 服务负载 `N=4, C=128, S=262144` fp16，L=4194304，splits=4）：流量 5·4194304·128·2 ≈ **5.37GB**；workspace `2·128·4·4 = 4KB`（可忽略）。带宽受限算子上访存/遍数是主导项（`constants.md` CONST-mte2-degradation）。

**R4 硬件亲和性评估**（检查清单逐项；负向淘汰附已核对的 docs 条款或 examples 佐证，或「未文档化假设 + 估算依据」）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线/主选 跨段拆分（0/1） | Vector：`T.reduce` + v-prefix 逐元素（`docs/Tilelang.language/规约操作/T.reduce.md` / `T.vmul.md`），无 MAC 段 ✓ | `(bN,bS)` 子块 ≈ `bN·bS·BPE + bN·8` B ≤ 160KB（§4.5，v1 二维 tiling）+ `[C,splits]` workspace（4KB）✓ | S 尾轴连续 stride=1；fp16/bf16 ×8、fp32 ×4；尾子块/尾 N-块 `T.min` 截尾 ✓ | C/S/splits/bS/bN/n_tile_N/n_tile_S_* 编译期常量 ✓ | 段内顺序依赖链；装载/计算 overlap 交 auto multibuffer ✓ | ✅ **主选** |
| raw-moment（2） | Vector ✓ | 同上 | 同上 | ✓ | ✓ | ❌ 数值否决（R1#2 判别段 A） |
| one-core-per-channel（3） | Vector ✓ | 同上 | 同上 | ✓ | ✓ | ❌（`C<40` 时逻辑核数 < 物理核 → 并行度不足；split 服务负载正是「C 不够填满核」区；且宽核已覆盖） |
| 原子累加（4） | Vector ✓ | `[C]` | ✓ | ✓ | 需 `T.atomic_add` | ❌（累加顺序不确定，破坏源确定性和「无 atomics」契约；`docs/Tilelang.language/原子操作/T.atomic_add.md` 可用但无必要） |
| online/Welford（5） | 合并链对 Vector 失分 | 同 | 同 | 同 | 依赖链长 | ❌（R2：无对应 running 统计量） |

**调研结论**：选定**「中心化 `x̂` + 双标量 `T.reduce` 归约 + 跨段部分和 `[C,splits]` workspace + finalize 合并（无 atomics）+ 逐通道 plane apply」算法族**（= 基线内核数学 + 候选 1 的分布式 epilogue），采纳 §1.6.1 的 O1/O2。关键依据：R3 表中跨段拆分的 GM 流量与 one-core-per-channel 同量级，但对「C 不足以填满物理核」的 split 服务负载提供 `C·splits` 并行度（R4 #3 否选依据）；raw-moment 因大均值小方差抵消被机器证据否决（§1.6.1 判别段 A）；原子累加破坏顺序确定性（R4 #4）。与基线的结构差异一句话：分段切分从「扁平 `L` 的 block 区间 + threads」改为「S 轴仿射子段 + 一维 persistent 分核」、归约从线程树改 lane 向量化 + `T.reduce`、合并保持 workspace + 二阶段但**显式声明无 atomics**、apply 从 flat map 改逐通道 plane、dtype 改显式 fp32 中转。源算法优化手段意图承接见 §0.6 末尾汇总。调研范围：`algorithm-candidates.md` ALG-reduction 族命中行、`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md`、`constants.md` CONST-* 四条、`examples/batch_norm_bwd/probe/PROBE.md`、源码 §0.3/§0.4、本函数设计探针真机结论、结构判据（R2）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；常数引用 `constants.md` CONST-mte2-degradation 3:1 R/W 地板 ~1.26 TB/s、CONST-vector-launch-overhead ~0.5µs/op）**：

- **large-spatial（split 主负载）** `4×128×262144` fp16（流量 5.37GB）：流量项 ≈ 5.37GB / 1.26TB/s ≈ **4.26ms**（发射项：40 核 × 13 任务 × ~29 S-子块 × ~12 op × 0.5µs ≈ 0.9ms，被 MTE2 窗口隐藏，取 max = 流量项）。workspace 往返 4KB（可忽略）+ finalize（`min(C,40)` 核 × 4 任务 × splits 4 × 2 op，≈ µs 级）。
- **test 中档** `4×64×4096` fp16（L=16384，splits=min(16384,8)=8，dist=512）：流量 5·16384·64·2 ≈ 10.5MB → 流量项 ≈ **8.3µs**；发射主导量级 **数十µs**。
- 容量项：0（纯 Vector，无跨引擎往返）。**⚠️ stale 标注**：CONST-mte2-degradation（tilelang 0.1.2+ed787bb，2026-09-07）被 kb_stale_check 标记 stale（当前 417dae99，2026-10-10）；数字仅用于候选排序与 Stage 4 对账，结构性结论（split 流量 = one-core-per-channel 同量级、workspace 可忽略）不依赖其精确值。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（跨段部分和 + finalize 合并 + 逐通道 apply）。逐项四要素：

| # | 优化项 | 原式（源/数学） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 除法转乘倒数（L 为编译期常量） | `per = w·rstd / L` | `per = (w·rstd)·fl(1/L)`（`L_inv` 工厂期 Python 常量） | L 编译期常量；`fl(1/L)` 相对误差 ≤ 2⁻²⁴（L=2^k 时精确）；D-1 机器验证 21 案例（3 dtype × 7 shape）全过：候选式 vs 基线式 **max ulp = 0、违反率 0** | 逐通道 1 次 vdiv → 1 次 vmul（(1,1) buffer，op 型替换，零风险） |
| O2 | distributed epilogue（折 L 与 coef 入逐通道标量） | `grad_x = coef·(L·grad_out − gb − x̂·gw)` | `grad_x = scale·grad_out + shift − centered_coef·(x−mean)`，`scale = w·rstd`、`shift = −per·gb`、`centered_coef = per·gw·rstd` | 分配律恒等：`coef·L·grad_out = (w·rstd)·grad_out = scale·grad_out`；`coef·gb = per·gb = −shift`；`coef·gw·x̂ = per·gw·rstd·(x−mean) = centered_coef·(x−mean)`。D-1：**max ulp = 0、违反率 0**（三 dtype 全过） | 逐元素由 3 mul（`L·g`、`x̂·gw`、末乘 `coef`）降为 2 mul（`scale·g`、`centered_coef·(x−mean)`），每元素省 1 向量 mul（≈L op/通道） |
| O3 | `x̂` 公共子表达式消除（CSE，段内） | 归约遍与乘积各算一次 `(x−mean)·rstd` | 段内 `x̂` 算一次后即用于 `prod = g·x̂`（同一子块内复用） | 同值复用（同一 buffer 恒等读取），无浮点语义变化 | 段内每元素省 1 vsub + 1 vmul；**apply 阶段无法跨 kernel 复用**，须重算（已在 R3 流量中计） |

**D-1 等价性机器验证结果表**（脚本 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/verify_equiv.py` 已执行，2026-10-14，torch CPU、fp64 参照 + 角点值；判定准则：候选式 vs 基线式在 dtype 容差内且违反率 = 0）：

| 验证项 | 案例数 | fp16/bf16 舍回后 max ulp | fp32 max ulp | 违反率 | 结论 |
|--------|--------|---------|---------|--------|------|
| O1+O2 联合（候选式 vs 基线式） | 21（3 dtype × 7 shape：(32,1)/(8,1024)/(4,1024)/(4,784)/(4,2048)/(1,1)/(2,3)） | **0（逐位一致）** | **0** | **0**（三 dtype 全 0） | **EQUIV_PASS** |
| 判别段 A：raw-moment `gw`（拒绝证据，非采纳项） | 7 shape × 3 dtype | — | fp32 max 违反率 **0.1553**（N=4,S=2048 大均值小方差） | — | 机器判别：候选式违反率 0.1553 → **放弃采纳（FAIL）**（支撑 R1#2 否决；fp16/bf16 宽容差掩蔽，正是中心化式跨 dtype 保留的理由） |

形态覆盖：randn 梯度、大均值小方差 `x`（≈3e3 ± 0.5，raw-moment 抵消压力源）、常量 `x`（零方差通道）、大动态范围梯度、次正规点位、±inf、NaN、极值 `weight`、退化 shape N=1/2/S=1。

**优化结论**：采纳 O1/O2/O3。优化后公式（§3.1 唯一输入，`r = fl(1/L)`）：

$$
\text{scale} = w\cdot\text{rstd},\quad \text{per} = \text{scale}\cdot r,\quad
\text{shift} = -\,\text{per}\cdot\text{gb},\quad \text{center} = \text{per}\cdot\text{gw}\cdot\text{rstd}
$$
$$
\text{grad\_x} = \text{scale}\cdot\text{grad\_out} + \text{shift} - \text{center}\cdot(x-\text{mean})
$$

（`shift` 取负号与源 finalize（`shift_out = -per_elem*total`，L1909）逐字一致；apply 用 `+shift`。）

**否决项及依据**（负向断言举证）：
- raw-moment `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)`：D-1 判别段 A 机器证据（fp32 违反率 0.1553），非纸面论证；
- fp16/bf16 原生域计算链（不经 fp32 中转）：`docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` §2.2.1 bf16 列为 ×（无 Vector 算术）——fp32 中转为**必需**；
- 跨 dtype `T.copy` 作 cast 融合：`docs/Tilelang.language/内存操作/T.copy.md` §3 跨 dtype copy 不减少 vector pass 且舍入不可控——统一走显式 `T.vcast(rint)`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（GPU 源） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|---------------------|--------------------------|---------|--------------|
| 1 | 段内平面 gather | `T.vectorized` 循环 `grad_out[head//S, bc, head%S+i]` | `T.copy` 双显式 slice（strided 区域，`docs/Tilelang.language/内存操作/T.copy.md`） | ✅ | — |
| 2 | 上 cast fp32 | `T.cast(..., "float32")` 逐元素 | `T.vcast(stage, x32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 3 | `x̂ = (x − mean)·rstd` | 逐元素 `(cast(held_x)-mean)*rstd` | `T.vbrc(m1, brc)` + `T.vsub` + `T.vbrc(r1, brc)` + `T.vmul`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | ✅ | — |
| 4 | 双 `+=` 累加 | 逐元素 `sums[j]+=g; sums_xhat[j]+=g*x_hat` | `T.reduce(g32, red, dims=[1], clear=False, size=[N,r2])` + `T.reduce(prod, red2, ...)`（`docs/Tilelang.language/规约操作/T.reduce.md`） | ✅ | — |
| 5 | 段内线程树归约 | `T.reduce_sum(sums, dim=1)` | 同上（`dims=[1]` 行和累加）+ 段末 `T.reduce(dims=[0])`（`docs/Tilelang.language/规约操作/T.reduce_sum.md`） | ✅ | — |
| 6 | 跨段合并（finalize） | `T.serial(splits)` 标量 `total+=` | `T.copy` 载 `(1,splits)` + `T.reduce(pw, tot, dims=[1])`（探针真机验证） | ✅ | — |
| 7 | epilogue 系数 | 逐通道标量 `scale/per/shift/centered_coef` | (1,1) 上 `T.vmul` 标量 op（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量 ✓） | ✅ | — |
| 8 | `grad_x` epilogue | 逐元素复合标量式 | `T.vbrc` 广播 + `T.vmul`/`T.vadd`/`T.vsub`（O2 分布式） | ✅ | — |
| 9 | 下 cast + 写出 | 赋值隐式截断 + `T.vectorized` 写 | `T.vcast(gx32 → stage, rint)` + `T.copy` 双 slice | ✅ | — |
| 10 | 段内 N/S-子块循环 | 源无核内子块循环（段=block） | `for nn in T.serial(n_tile_N)` + `for tt in T.serial(n_tile_S_stats\|n_tile_S_apply)`（静态边界） | ❌ 保留 | **tile 级调度循环**：`n_tile_N`/`n_tile_S_*` 为 host 编译期常量；承载 UB 有界（超大 N/段），非逐元素计算 |
| 11 | 任务 grid-stride 循环 | 一 block 一 (c,段)（无核内循环） | `for i in T.serial(num_local_tasks)` + `if t < C*splits` 守卫 | ❌ 保留 | **block 级任务映射计算**：每 task 数个标量索引（t/c/sp），非逐元素热点；是 persistent 分核（§0.6 R1）的载体 |
| 12 | 通道-段索引 | grid id 直算 | `c = t // splits`、`sp = t % splits`（floordiv/mod 标量索引，探针真机验证） | ❌ 保留 | **block 级索引计算**：每任务 2 次整数 div/mod，非逐元素热点；无逐元素等价 API |
| 13 | 段/子块有效长度 | 逐向量/逐元素谓词 `if head+npt<=end` | 余额分配 `s0=sp·dist+T.min(sp,rem)`、`s1=(sp+1)·dist+T.min(sp+1,rem)`、`real=s1-s0`；`rN=T.min(bN,N-n0)`、`r2=T.min(bS,s1-ss)`（标量 `T.min`） | ❌ 保留 | **动态边界元数据**：余额分段 / 尾 N-块 / 尾子块 extent；per-段/per-子块 1 次标量 min + slice 界，`if r2>0` 守卫空子块 |
| 14 | 逐通道标量装载 | 线程读 `weight[bc]/mean[bc]/rstd[bc]` | `T.copy(weight[c:c+1], w1)` 标量 slice | ✅ | — |

**向量化结论**：逐元素计算（gather / cast / x̂ / 乘积 / 累加归约 / epilogue / 舍回）与通道归约、跨段合并**已全部向量化/归约化**（替代 API 均有 docs+探针佐证，上表）。保留 4 类标量/循环——段内 S-子块调度（#10）、任务 grid-stride（#11）、通道-段 floordiv/mod 索引（#12）、尾段 `T.min`（#13），逐项理由如上，无「实现简单」类理由。§6 循环结构与本表一致（kernel 内无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）**；另评估核内重排（转置）与拆分轴。I/O 原生 `(N,C,S)` 的逐通道平面是 `(N,S)`（归约轴 = `N·S`，非尾轴）；本设计的段切分轴与向量化轴需分别决策。

**轴质量评分**（主选向量化轴 = S；昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 逐通道平面行内 stride=1（尾轴 S）；fp16/bf16 向量宽 ×8：S=262144（32768×8 ✓）；fp32 ×4：S=784（196×4 ✓） |
| 尾 lane 浪费率 | S=262144/1024/784 均为向量宽整数倍（0%）；S=250（设计探针）→ ceil(250/8)·8−250=6，浪费 2.4%（仅探针） |
| 累加链形态 | 归约维 = `dims=[1]`（S 子块内，连续 stride=1，硬件树归约）+ `dims=[0]`（N，小）；无跨步系数 |
| 段切分轴 | 沿 **S 轴**切成 `splits` 个仿射子段 `[0:N, s0:s1)`（**余额分配** `dist/rem`，`real ≥ 1`，恰覆盖 `[0,S)`）——每段仍是规则 `(N,·)` 区域，`T.copy` 直载；`splits` 上限 cap 到 `S`（`splits ≤ S ⇒ dist ≥ 1`） |
| repack 代价 | **无**（I/O 原生 `(N,C,S)` 行内连续 = 核内平面行；逐通道/逐段 gather 是 strided GM 区域，`T.copy` 直接支持） |
| UB 容量影响 | 段内 `(bN,bS)` 子块 ≈ `bN·bS·BPE + bN·8` B ≤ 160KB（§4.5，v1 二维 tiling）；workspace `[C,splits]` = 4KB |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | 段切分轴 | repack 路径 | 预估收益/代价 | 采纳？ |
|---|---------|---------|---------|------------|--------------|--------|
| 1 | I/O 原生 `(N,C,S)` + 逐通道平面 `(N,S)` + **沿 S 轴仿射子段** | S（尾轴连续；`reduce dims=[1]` 水平归约 + `dims=[0]`） | S（`c=t//splits`、`sp=t%splits`，段 = `[0:N, s0:s0+real)`） | 无（GM strided slice 直载） | C·splits 并行；流量 5·L·C·eb（与 one-core-per-channel 同量级） | ✅ **主选**（探针真机验证） |
| 2 | 沿扁平 `L` 切段（GPU 源式） | S | L（flat `[start, end)`，元素索引 `l//S`/`l%S`） | 需逐元素 flat 地址 | 归约轴不变；但 flat 段不保证 plane 对齐（跨批行），NPU `T.copy` 需仿射区域、flat 索引 `l//S`/`l%S` 非仿射（`docs/Tilelang.language/内存操作/T.copy.md` 未列非仿射索引）；逐元素通道索引破坏向量化 | ❌（非仿射 gather，无 API 支撑） |
| 3 | 沿 N 轴切段 `[n0:n1, c, 0:S)` | S | N | 无 | 当 `N≥splits` 时等价可行（large-spatial N=4=splits 恰为每段一批）；当 `N<splits` 并行度受限（须退化为 S 轴） | ❌（通用性差；S 轴切分对 `N<splits` 亦成立，统一取 #1） |
| 4 | 转置布局 `(N,C,S)→(N,S,C)` 使通道尾轴 | S/C | — | host permute（百 µs 级）或核内 `T.transpose` 链 | 归约轴不变（仍是 `N·S`）；repack 纯开销；op 契约输出 `(N,C,S)` 需再转回 | ❌ |
| 5 | 垂直扫描（lane→独立行，串行步进 `N·S`） | 行 | — | 无 | 归约维访问跨步 → 无向量宽度；跨步累加退化为标量 | ❌ |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：GPU 的 stats 把 `threads` 映射到段内扁平 `L` 的连续段（`head = start + (step*threads+j)*num_per_thread`，源码 L1844），lane 沿「连续访存轴」；apply 用平铺 `base`（L1939）。本设计独立评估后**结论一致（连续访存轴）、实现不同**：NPU 选 S 为向量轴（行内 stride=1，v-op 广播形态为行内 `[M,N]↔[M,1]`，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2），段切分改沿 S 轴仿射子段（`#1`）而非 flat（`#2`）；GPU 的线程树 `T.reduce_sum(dim=1)` 与 NPU `T.reduce` 归约轴不一一对应，故归约结构按 NPU 硬件重设计（§0.6 R2）。

**实验裁决模式评估**：主选 #1 vs 备选（one-core-per-channel #3 形态）的判定裕度 = 并行度（split 提供 `C·splits` 逻辑核，one-core-per-channel 提供 `C`）——当 `C ≥ 40` 时两者并行度均足够、split 多一趟 finalize/workspace（小）；当 `C < 40` 时 split 明显优。该裕度由**编译期已知的 C**决定（非未实证常数），故无需三件套实验裁决：工厂/选核按 `C` 属主（wrapper `select_bwd_kernel` 已把该函数路由到 split 负载）。唯一未实证量是 **auto-multibuffer 膨胀系数**（决定 `bS` 取值）——这是**参数级**而非结构级决策，设计期取保守预算（§4.5），`bS` 留 Stage 4；编译期 UB 溢出报错为天然 guard。**不启动额外设计期探针（D-3）**：结构决策不依赖该常数，且本函数设计探针已真机验证三段结构。

**布局决策结论**：选定 #1：核内布局 = I/O 原生 `(N,C,S)`、逐通道平面 `(N,S)` 物化到 UB、向量化轴 = S（尾轴）、段切分轴 = S（`c=t//splits`、`sp=t%splits` 仿射子段，cap `splits≤S`）、repack = 无。该结论同步落入 §3.3 伪代码与 §6 循环结构（归约/广播的内层向量维 = S；buffer 形状 = `(bN,bS)`，N 维分块见 §4.5/§5.2）；弃选方案的量化理由见候选矩阵。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（Vector 路径）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.reduce` + v-prefix 逐元素 + `T.vbrc` 标量广播），无 matmul | 无 Cube/L0/L1 参与；单 `T.Scope("Vector")` |
| 归约 | `T.reduce` dims=[1]/[0] | `.agents/skills/tilelang-op-design/references/decision-tree.md`：`T.reduce` 在 Developer 与 Expert 两模式都可使用——模式选择取决于是否需显式控制 UB 层级与同步，而非用了哪个 API |
| 内存层级 | 仅 GM↔UB，需显式 UB 放置 | `T.alloc_ub`（`docs/Tilelang.language/内存操作/T.alloc_ub.md`：`T.alloc_ub [Expert mode]`） |
| 三段结构 | 三个 kernel 顺序调用、workspace 经 GM 交接（非核内跨核同步） | 无核间 flag；三段各自 Vector scope 内顺序执行、自动流水 |
| 先例 | `examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py`（纯 Vector、`T.Scope("Vector")` + `T.alloc_ub`、persistent，Expert 已交付，含 `T.reduce`/`T.vmul`/`T.vcast`）；`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（同族 Expert/Vector） | Expert/Vector 形态已验证可行 |
| 探针 | `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` 真机 PASS；本函数设计探针 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/history_version/design_probe_split_phase.py` 真机 PASS（Expert `T.Scope("Vector")` + `T.alloc_ub` 下三段结构全编译运行） | 核心构造已真机验证 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub`（显式 UB）——v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1） |
| Scope | 每段单 `with T.Scope("Vector"):`（无 Cube scope，非 MixCV） |
| 计算方式 | v-prefix API（vcast/vmul/vsub/vadd/vbrc）+ `T.reduce` + `T.min` 标量边界 |
| 同步 | Vector scope 内默认顺序/自动流水；三段间由运行时流序保证（§7） |
| Kernel 启动 | 三段各 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |
| 一致性 | 实现期设 `ASCEND_MODE = "Expert"`；Stage 0 wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 现为 `ascend_mode = "Developer"`，须在集成期同步改为 `"Expert"`（记入 §9.1 风险） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 优化后公式（fp32 计算域；`r = fl(1/L)`；`scale = w·rstd`、`per = scale·r`、`shift = −per·gb`、`center = per·gw·rstd`）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `g32 = fp32(grad_out)`；`x32 = fp32(x)` | 上 cast（fp32 输入直拷） |
| 2 | `xhat = (x32 − mean)·rstd`（段内） | 逐通道标量广播 |
| 3a | 段部分和 `p_sum[c,sp] = Σ_{seg} g32`；`p_sumx[c,sp] = Σ_{seg} g32·xhat` | 段内双归约（stats） |
| 3b | `gb = Σ_{k<splits} p_sum[c,k]`；`gw = Σ_k p_sumx[c,k]` | 跨段合并（finalize） |
| 4 | `scale = w·rstd`；`per = scale·r`；`shift = −per·gb`；`center = per·gw·rstd` | epilogue 系数（O1，finalize） |
| 5 | `grad_x = scale·g32 + shift − center·(x32 − mean)` | 分布式 epilogue（O2，apply） |
| 6 | `grad_x_dtype = dtype_round(grad_x)` | 单次舍回 |
| 7 | `grad_bias = gb`；`grad_weight = gw` | 通道梯度输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务映射（stats） | `t = i·K + cid`；`c = t//splits`；`sp = t%splits` | `T.serial(num_local_tasks_s)` + `if t < C*splits` | 静态边界/整数索引（探针验证） | Expert |
| 段 extent（余额分配） | `s0=sp·dist+T.min(sp,rem)`；`s1=(sp+1)·dist+T.min(sp+1,rem)`；`real=s1-s0≥1` | `dist/rem`（工厂常量）+ 两次标量 `T.min` | 标量（`docs/Tilelang.language/比较操作/T.min.md`）；保证非空（v1） | Expert |
| N-子块 extent | `rN = min(bN, N−n0)` | `T.min(bN, N - n0)` | 标量；`n0=nn·bN`（v1 N-tiling） | Expert |
| 平面 gather | GM strided 区域→UB | `T.copy(grad_out[n0:n0+rN, c, ss:ss+r2], stage[0:rN, 0:r2])` | src/dst 双显式 slice，dst 零起点 | Expert |
| 上 cast | fp16/bf16→fp32 | `T.vcast(stage, g32, round_mode="rint")` | f16→f32 / bf16→f32 rint ✓ | Expert |
| 标量广播 | `[1,1]→[bN,bS]` | `T.vbrc(m1, brc)` | src (1,1) → dst (bN,bS) ✓（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | Expert |
| `x̂` 中心化 | `(x−mean)·rstd` | `T.vsub(xh32, brc, xh32)` + `T.vmul(xh32, brc, xh32)` | 同形 (bN,bS) 就地 | Expert |
| 段内归约（S） | `Σ` over 子块列 | `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[rN, r2])` | 跨子块累加；`size` 截断有效列（`docs/Tilelang.language/规约操作/T.reduce.md` §2.1） | Expert |
| N-块归约（N） | `Σ` over N-块行 | `T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=False, size=[rN, 1])` | (bN,1)→(1,1)，`clear=False` 跨 N-块累加（N>1；否则工厂 `T.copy`）（v1） | Expert |
| 部分和写 | 段部分和→GM workspace | `T.copy(gb, partial_sum[c:c+1, sp:sp+1])` | 2D 1×1 标量 slice（探针验证） | Expert |
| 跨段合并（finalize） | `Σ_k p_sum[c,k]` | `T.copy(partial_sum[c:c+1, 0:splits], pw)` + `T.reduce(pw, tot, dims=[1], reduce_mode="sum", clear=True)` | (1,splits)→(1,1)（splits>1；否则工厂 copy） | Expert |
| 系数预折叠 | `scale/per/shift/center` | `T.vmul` + `T.vmul(sc, L_inv, per)` + `T.vmul(per, tot, shift)` + `T.vmul(shift, -1.0, shift)` + `T.vmul(center, r1, center)` | (1,1) 标量 op ✓（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2） | Expert |
| 通道梯度写 | `gb/gw → GM` | `T.copy(tot, grad_bias[c:c+1])` | 标量 slice | Expert |
| apply N/S-子块上界 | `n_tile_N=ceil(N/bN)`；`n_tile_S_apply=ceil(S/bS)`（**独立于 stats**） | 工厂常量（v1 修复，§0.0 阻塞项 1） | S-子块 `T.serial(n_tile_S_apply)` 覆盖全 S | Expert |
| apply 平面 gather | GM→UB | `T.copy(grad_out[n0:n0+rN, c, s0:s0+real], stage[0:rN, 0:real])` | 双显式 slice | Expert |
| apply epilogue | `scale·g + shift − center·(x−mean)` | `T.vbrc` + `T.vmul` + `T.vadd` + `T.vsub` | 同形 (bN,bS) | Expert |
| 舍回 + 写出 | fp32→dtype→GM | `T.vcast(gx32, stage, round_mode="rint")` + `T.copy(stage[0:rN,0:real], grad_x[n0:n0+rN, c, s0:s0+real])` | f32→f16 / f32→bf16 rint ✓ | Expert |

### 3.3 计算伪代码

> 按模板 §3.3 注记：`T.Tensor` 的 dtype 以第二个位置参数传入，不用 `dtype=` 关键字。**Expert 模式**：每个 `with T.Kernel(...)` 内套 `with T.Scope("Vector"):`，UB 缓冲用 `T.alloc_ub`（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）。工厂接口 `(N, C, S, dtype)` 与源一致；**三个返回工厂均无参调用**（源 `splits/threads/num_per_thread/blocks` 全部在工厂闭包内烘焙——NPU wrapper `BatchNormBwdSplitKernel.forward` 以 `stats()(...)`/`finalize()(...)`/`apply_()(...)` 调用，见 §0.2 注）。`splits` 取 wrapper 同源 `_split_seed` 语义并 cap 到 S；`if N>1`/`if splits>1` 等退化分支与 `use_transit` 分支在**工厂层**解析（trace 体之外，规避 parser 作用域陷阱）。三段共用同一 `splits`。

```python
def _batch_norm_bwd_split_kernel(N, C, S, dtype):
    L = N * S
    L_inv = 1.0 / float(L)                                  # O1: 编译期常量
    use_transit = dtype in ("float16", "bfloat16")          # bf16 无 Vector 算术 -> 必需中转
    elem_bytes = 2 if use_transit else 4
    BPE = elem_bytes + 16                                   # stage(elem_bytes) + g32/x32/prod/brc 各 4
    UB_SAFE_BYTES = 163_840                                 # 160 KiB（同族宽核 held 档实测 144KB 通过）
    vw = 8 if use_transit else 4                            # 向量宽
    vector_cores = NPUUtils.get().get_aicore_num() * 2      # 实查 20 -> 40（纯 Vector 翻倍）
    splits = max(1, min(L, -(-512 // C)))                   # = wrapper _split_seed 语义
    splits = min(splits, S)                                 # cap 到 S（保证 dist>=1；v1）

    # ---- 余额分配（均衡段，每段 real>=1；v1 §0.0 阻塞项 2）----
    dist = S // splits
    rem = S % splits
    max_seg = dist + (1 if rem > 0 else 0)                  # 最长段的 S 长度（工厂常量）

    # ---- 二维 (bN,bS) tiling（v1 §0.0 阻塞项 3）----
    SMALL = 64                                              # (1,1)/(1,splits) 小项余量
    ROW_OVERHEAD = 8                                        # red + red2 = 2*(bN,1) fp32
    if N == 1:
        bN = 1
        bS = max(vw, ((UB_SAFE_BYTES - SMALL) // BPE // vw) * vw)
    else:
        # 预留至少 2 行（使 dims=[0] 的 N 归约合法），再按剩余容量取 bN
        bS = max(vw, (((UB_SAFE_BYTES - SMALL) // 2) // BPE // vw) * vw)
        bS = min(bS, S)
        bN = min(N, max(2, (UB_SAFE_BYTES - SMALL) // (bS * BPE + ROW_OVERHEAD)))
    bS = min(bS, S)                                         # 小 S 全覆盖
    n_tile_N = -(-N // bN)                                  # N-子块数（静态）
    n_tile_S_apply = -(-S // bS)                            # apply 覆盖全 S（v1 修复，§0.0 阻塞项 1）
    n_tile_S_stats = -(-max_seg // bS)                      # stats 段上界（v1 修复，§0.0 阻塞项 1）

    s_gt1 = S > 1
    n_gt1 = N > 1
    split_gt1 = splits > 1

    n_tasks = C * splits
    KS = min(n_tasks, vector_cores);  nlS = -(-n_tasks // KS)   # stats persistent
    KC = min(C, vector_cores);        nlC = -(-C // KC)         # finalize/apply persistent

    @tilelang.jit(target="npuir")
    def _stats():
        @T.prim_func
        def stats(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            partial_sum: T.Tensor((C, splits), "float32"),
            partial_sum_xhat: T.Tensor((C, splits), "float32"),
        ):
            with T.Kernel(KS, is_npu=True) as (cid, _):
              with T.Scope("Vector"):
                stage = T.alloc_ub((bN, bS), dtype)          # g / x transit
                g32 = T.alloc_ub((bN, bS), "float32")
                xh32 = T.alloc_ub((bN, bS), "float32")       # x32 -> x_hat
                prod = T.alloc_ub((bN, bS), "float32")
                brc = T.alloc_ub((bN, bS), "float32")
                red = T.alloc_ub((bN, 1), "float32")         # 行和累加器（跨 S-子块）
                red2 = T.alloc_ub((bN, 1), "float32")
                gb = T.alloc_ub((1, 1), "float32")           # 段部分和（跨 N-块累加）
                gw = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "float32")
                r1 = T.alloc_ub((1, 1), "float32")

                for i in T.serial(nlS):
                    t = i * KS + cid
                    if t < n_tasks:
                        c = t // splits
                        sp = t % splits
                        # 余额分配段 [s0, s1)：real = s1 - s0 >= 1（v1 §0.0 阻塞项 2）
                        s0 = sp * dist + T.min(sp, rem)
                        s1 = (sp + 1) * dist + T.min(sp + 1, rem)
                        T.copy(mean[c:c + 1], m1)
                        T.copy(rstd[c:c + 1], r1)
                        T.clear(gb)
                        T.clear(gw)
                        for nn in T.serial(n_tile_N):                # N-子块（v1）
                            n0 = nn * bN
                            rN = T.min(bN, N - n0)                   # >= 1
                            T.clear(red)
                            T.clear(red2)
                            for tt in T.serial(n_tile_S_stats):
                                ss = s0 + tt * bS
                                r2 = T.min(bS, s1 - ss)              # s1 = s0 + real
                                if r2 > 0:                           # 尾子块可能空（v1 守卫）
                                    T.copy(grad_out[n0:n0 + rN, c, ss:ss + r2],
                                           stage[0:rN, 0:r2])
                                    if use_transit:
                                        T.vcast(stage, g32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:r2], g32[0:rN, 0:r2])
                                    T.copy(x[n0:n0 + rN, c, ss:ss + r2],
                                           stage[0:rN, 0:r2])
                                    if use_transit:
                                        T.vcast(stage, xh32, round_mode="rint")
                                    else:
                                        T.copy(stage[0:rN, 0:r2], xh32[0:rN, 0:r2])
                                    T.vbrc(m1, brc); T.vsub(xh32, brc, xh32)   # x - mean
                                    T.vbrc(r1, brc); T.vmul(xh32, brc, xh32)   # x_hat
                                    T.vmul(g32, xh32, prod)                    # g * x_hat
                                    # size=[rN,r2] 截断有效范围；clear=False 跨 S-子块累加
                                    if s_gt1:
                                        T.reduce(g32, red, dims=[1], reduce_mode="sum",
                                                 clear=False, size=[rN, r2])
                                        T.reduce(prod, red2, dims=[1], reduce_mode="sum",
                                                 clear=False, size=[rN, r2])
                                    else:
                                        T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                        T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                            # N-块行和累加进 (1,1) gb/gw（跨 N-块；size 截尾 N-块）
                            if n_gt1:
                                T.reduce(red, gb, dims=[0], reduce_mode="sum",
                                         clear=False, size=[rN, 1])
                                T.reduce(red2, gw, dims=[0], reduce_mode="sum",
                                         clear=False, size=[rN, 1])
                            else:
                                T.copy(red[0:1, 0:1], gb); T.copy(red2[0:1, 0:1], gw)
                        T.copy(gb, partial_sum[c:c + 1, sp:sp + 1])
                        T.copy(gw, partial_sum_xhat[c:c + 1, sp:sp + 1])
        return stats

    @tilelang.jit(target="npuir")
    def _finalize():
        @T.prim_func
        def finalize(
            partial_sum: T.Tensor((C, splits), "float32"),
            partial_sum_xhat: T.Tensor((C, splits), "float32"),
            weight: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),
            grad_bias: T.Tensor((C,), "float32"),
            scale_out: T.Tensor((C,), "float32"),
            shift_out: T.Tensor((C,), "float32"),
            centered_coef_out: T.Tensor((C,), "float32"),
        ):
            with T.Kernel(KC, is_npu=True) as (cid, _):
              with T.Scope("Vector"):
                pw = T.alloc_ub((1, splits), "float32")
                pwx = T.alloc_ub((1, splits), "float32")
                tot = T.alloc_ub((1, 1), "float32")
                totx = T.alloc_ub((1, 1), "float32")
                w1 = T.alloc_ub((1, 1), "float32")
                r1 = T.alloc_ub((1, 1), "float32")
                sc = T.alloc_ub((1, 1), "float32")
                per = T.alloc_ub((1, 1), "float32")
                sh = T.alloc_ub((1, 1), "float32")
                ce = T.alloc_ub((1, 1), "float32")
                for i in T.serial(nlC):
                    c = i * KC + cid
                    if c < C:
                        T.copy(partial_sum[c:c + 1, 0:splits], pw)
                        T.copy(partial_sum_xhat[c:c + 1, 0:splits], pwx)
                        if split_gt1:
                            T.reduce(pw, tot, dims=[1], reduce_mode="sum", clear=True)
                            T.reduce(pwx, totx, dims=[1], reduce_mode="sum", clear=True)
                        else:
                            T.copy(pw, tot); T.copy(pwx, totx)
                        T.copy(tot, grad_bias[c:c + 1])
                        T.copy(totx, grad_weight[c:c + 1])
                        T.copy(weight[c:c + 1], w1)
                        T.copy(rstd[c:c + 1], r1)
                        T.vmul(w1, r1, sc)                 # scale = w * rstd
                        T.vmul(sc, L_inv, per)             # per = scale * fl(1/L)  (O1)
                        T.vmul(per, tot, sh)               # + per * gb
                        T.vmul(sh, -1.0, sh)               # shift = -per * gb (源 finalize 同号)
                        T.vmul(per, totx, ce)              # per * gw
                        T.vmul(ce, r1, ce)                 # centered_coef = per * gw * rstd
                        T.copy(sc, scale_out[c:c + 1])
                        T.copy(sh, shift_out[c:c + 1])
                        T.copy(ce, centered_coef_out[c:c + 1])
        return finalize

    @tilelang.jit(out_idx=[-1], target="npuir")
    def _apply():
        @T.prim_func
        def apply(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            mean: T.Tensor((C,), "float32"),
            scale: T.Tensor((C,), "float32"),
            shift: T.Tensor((C,), "float32"),
            center: T.Tensor((C,), "float32"),
            grad_x: T.Tensor((N, C, S), dtype),
        ):
            with T.Kernel(KC, is_npu=True) as (cid, _):
              with T.Scope("Vector"):
                stage = T.alloc_ub((bN, bS), dtype)
                g32 = T.alloc_ub((bN, bS), "float32")
                x32 = T.alloc_ub((bN, bS), "float32")
                brc = T.alloc_ub((bN, bS), "float32")
                m1 = T.alloc_ub((1, 1), "float32")
                s1 = T.alloc_ub((1, 1), "float32")
                h1 = T.alloc_ub((1, 1), "float32")
                c1 = T.alloc_ub((1, 1), "float32")
                for i in T.serial(nlC):
                    c = i * KC + cid
                    if c < C:
                        T.copy(mean[c:c + 1], m1)
                        T.copy(scale[c:c + 1], s1)
                        T.copy(shift[c:c + 1], h1)
                        T.copy(center[c:c + 1], c1)
                        for nn in T.serial(n_tile_N):              # N-子块（v1）
                            n0 = nn * bN
                            rN = T.min(bN, N - n0)                 # >= 1
                            for tt in T.serial(n_tile_S_apply):    # = ceil(S/bS)，覆盖全 S（v1 修复）
                                s0 = tt * bS
                                real = T.min(bS, S - s0)           # >= 1
                                T.copy(grad_out[n0:n0 + rN, c, s0:s0 + real],
                                       stage[0:rN, 0:real])
                                if use_transit:
                                    T.vcast(stage, g32, round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:real], g32[0:rN, 0:real])
                                T.copy(x[n0:n0 + rN, c, s0:s0 + real],
                                       stage[0:rN, 0:real])
                                if use_transit:
                                    T.vcast(stage, x32, round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:real], x32[0:rN, 0:real])
                                T.vbrc(m1, brc); T.vsub(x32, brc, x32)     # x - mean
                                T.vbrc(s1, brc); T.vmul(g32, brc, g32)     # scale * g
                                T.vbrc(h1, brc); T.vadd(g32, brc, g32)     # + shift
                                T.vbrc(c1, brc); T.vmul(x32, brc, brc)     # center * (x - mean)
                                T.vsub(g32, brc, g32)                      # - center * (x - mean)
                                if use_transit:
                                    T.vcast(g32, stage, round_mode="rint")
                                else:
                                    T.copy(g32[0:rN, 0:real], stage[0:rN, 0:real])
                                T.copy(stage[0:rN, 0:real],
                                       grad_x[n0:n0 + rN, c, s0:s0 + real])
        return apply

    return _stats, _finalize, _apply
```

> **v1 关键修正说明**：①**apply 独立子块上界**——stats 用 `n_tile_S_stats = ceil(max_seg/bS)`、apply 用 `n_tile_S_apply = ceil(S/bS)`，二者**分别定义、禁止共用**（v0 共用 `n_tile=ceil(chunk/bS)` 使 `splits>1` 时 apply 只覆盖 `ceil(chunk/bS)·bS` 列，§0.0 阻塞项 1）；②**余额分配段**——`s0 = sp·dist + T.min(sp, rem)`、`s1 = (sp+1)·dist + T.min(sp+1, rem)`、`real = s1-s0 ≥ 1`（v0 的 `s0=sp·chunk`+`min(chunk,S-s0)` 可产生负 extent，§0.0 阻塞项 2）；③**二维 (bN,bS) tiling**——新增 `for nn in T.serial(n_tile_N)` 与 `rN`，`red/red2` 为 `(bN,1)`、N-块间以 `dims=[0]` 累加进 `gb/gw`（v0 只对固定 N 取 bS，大 N 超 UB，§0.0 阻塞项 3）。
>
> **尾段/尾 N-块/尾子块说明**：段 extent `real = s1-s0`（余额分配，恒 ≥ 1）、N-块 extent `rN = T.min(bN, N-n0)`、S-子块 extent `r2 = T.min(bS, s1-ss)`——**装载/写出**用双显式 slice 只取有效 extent，**归约**用 `size=[rN, r2]`（dims=[1]）与 `size=[rN, 1]`（dims=[0]）把范围截断（`docs/Tilelang.language/规约操作/T.reduce.md` §2.1：`size` 控制 reduce 实际参与计算的数据范围）——slice 管 I/O、`size` 管归约，二者成对，缺一即把未初始化/残留列计入部分和（同族宽核 v0 缺陷根因）。尾子块以 `if r2 > 0:` 守卫（可能为空，v1）。
>
> **`dims=[0]` 归约形态**：`T.reduce(red, gb, dims=[0], clear=False, size=[rN,1])` 与既有的 `dims=[1], clear=False, size=[rN,r2]` 同口径（`clear=False` 累加到 dst、`size` 限范围），故 N-块间无需额外累加指令。**实现期（Stage 3）须真机核对**该 `dims=[0]` 组合；若 `clear=False` 与 `size` 在 `dims=[0]` 上不支持，**回退方案**：`T.reduce(red, part, dims=[0], clear=True, size=[rN,1])`（`part (1,1)`）+ `T.vadd(gb, part, gb)`（标量 v-add 累加），语义等价。
>
> **workspace 契约**：三段共用同一 `splits`；`partial_sum/partial_sum_xhat` 由 host 预分配为 `[C, splits]`，`scale/shift/centered_coef` 为 `[C]`。**接口保持**：`stats/finalize/apply` 参数名/顺序与源逐字一致；`apply` 保持源 `out_idx=[-1]`（返回 `grad_x`，单输出，wrapper `grad_x = apply_()(...)` 解包）。**合并方式声明**：采用 **GM 部分和 workspace `[C,splits]` + 第二阶段 finalize kernel**，**不使用 atomics**（原因见 §0.6 R3）。**三段间顺序**由运行时同一流的顺序下发保证（见 §7）。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` | ✅ 已核对 | 1D；`(cid, _)` 解包 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | ✅ 已核对 | Expert 显式 UB；1D~2D ✓ |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` | ✅ 已核对 | GM↔UB ✓；strided 区域（源 slice 非单位步长）✓；中间轴标量索引 ✓（探针） |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已核对 | f16→f32 / bf16→f32 / f32→f16 / f32→bf16 rint ✓；shape 须一致 |
| `T.reduce`（dims/clear/size） | `docs/Tilelang.language/规约操作/T.reduce.md` | ✅ 已核对（`dims=[0]` 组合留 Stage 3 真机核对，回退见 §3.3 注） | fp16/fp32 ✓ bf16 ×；`(bN,bS)→(bN,1)`、`(bN,1)→(1,1)`、`(1,splits)→(1,1)`；`clear=False` 跨子块/跨 N-块累加；`size=[rN,r2]`/`size=[rN,1]` 截断（§2.1） |
| `T.reduce_sum` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 已核对 | 等价可用 |
| `T.vmul`/`T.vsub`/`T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播/标量 ✓；就地 ✓ |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` | ✅ 已核对 | src (1,1)→(bN,bS) 合法；bf16 支持 |
| `T.min` | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 标量 min，尾段/子块 extent |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | ✅ 已核对 | 累加器初始化 |
| `T.serial` + `if` 守卫 + floordiv/mod 索引 | `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §3.3（persistent grid-stride 同款）；本函数设计探针（floordiv/mod 切片） | ✅ 已核对 + 真机 | persistent 任务映射 |
| `T.atomic_add`（可用但**未采用**） | `docs/Tilelang.language/原子操作/T.atomic_add.md` | ✅ 已核对 | fp32 支持；本设计因确定性要求弃用（§0.6 R3） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 三段各一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API | Yes | `T.vectorized`/`T.get_thread_binding`/`T.Parallel(1,threads)`/`threads=`/flat `T.Tensor` 别名 不迁移（§0.5 舍弃清单） |
| GEMM 分形/非整除 | No | 无 GEMM；S 非整除经 `T.min`+slice |
| L0C 容量 | No | 纯 Vector，无 L0 |
| 物理核数适配 | Yes | §5.5 三要素（实查 20→40） |
| bf16 无 Vector 算术 | Yes | §0.6 R6 fp32 中转链（必需） |
| 跨 block 归约 | Yes | GM workspace `[C,splits]` + finalize 二阶段（**无 atomics**，§0.6 R3） |
| 中间轴归约（非尾轴） | Yes | 逐通道 `(N,S)` 平面 strided gather + 两步 `T.reduce`（探针验证） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5/§0.6）

| 差异项 | GPU 源 | 本项目（Ascend） | 转换方案 |
|--------|--------|-----------------|----------|
| threads 参数 | `T.Kernel(C*splits, threads=)` | npuir 无效 | 移除；一维 persistent `T.Kernel(min(C*splits,40))` |
| 段切分轴 | 扁平 `L` 区间（`start=(bx%splits)*chunk`） | S 轴仿射子段（余额分配） | `c=t//splits`、`sp=t%splits`、`s0=sp·dist+T.min(sp,rem)`、`s1=(sp+1)·dist+T.min(sp+1,rem)`、`real=s1-s0≥1` |
| register/fragment 计算域 | `T.alloc_fragment`/`T.alloc_local` | v-op 要求 UB 操作数 | 全部 `T.alloc_ub` |
| 线程树归约 | `T.reduce_sum(sums, dim=1)` | 无线程树 | lane 向量化累加 + 两步 `T.reduce`（§0.6 R2） |
| finalize 合并 | `T.Kernel(1,threads)` 串行 `+=` | 一维 persistent + `T.reduce` | `T.copy` 载行 + `T.reduce(dims=[1])`；**无 atomics**（§0.6 R3） |
| apply 平铺映射 | flat `base` + `ch=(base//S)%C` | 逐通道 plane | 逐通道 `T.copy` 平面 epilogue（§0.6 R4） |
| 逐元素/逐向量谓词 | `if head+npt<=end` / `l<end` | `T.min`+双显式 slice | §0.6 R5 |
| 隐式 dtype 截断 | `T.cast` 赋值 | 显式 `T.vcast(rint)` | §0.6 R6 |
| `apply` out_idx | `out_idx=[-1]` | 保持 `out_idx=[-1]` | 返回 `grad_x`；host 预分配其余 buffer |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` | 极高（同族迁移，已交付） | strided 逐通道 plane gather、两步 `T.reduce`、`T.vbrc`、fp32 中转、尾块 `T.min`+双 slice+reduce `size`、分核实查、UB 预算 |
| `examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` | 高（**Expert/Vector** 纯 Vector 迁移先例，已交付） | `ascend_mode="Expert"` + 单 `T.Scope("Vector")` + `T.alloc_ub`、persistent 分核、`T.reduce`/`T.vmul`/`T.vcast` |
| `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` | 极高（同族行归约迁移，persistent 分核先例） | persistent 分核、尾块 `T.min`+双 slice、fp32 计算域、bf16 fp32 中转、分核实查 |
| `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` | 极高（本算子 NPU wrapper） | `BatchNormBwdSplitKernel.forward` 三段调用约定、`_split_seed`、三段共用 `splits`、`ascend_mode` 集成期须由 `"Developer"` 改 `"Expert"` |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` | 高（同族 row-reduction 设计） | 尾块 `T.min`+双显式 slice、`(bm,1)` reduce 输出形态 |
| `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` | 极高（本算子可行性探针，真机 PASS） | strided plane gather、两步 `T.reduce`、`T.vbrc` 广播、held/tiled 结构 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py` | 高（persistent 分核已调优） | `min(logical, cores)` 分核 + grid-stride serial、UB 预算 guard |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `grad_out` | `(N, C, S)` | float16 / bfloat16 / float32 | 上游梯度；N=C 轴外、S=spatial 乘积 |
| `x` | `(N, C, S)` | same_as(grad_out) | 原始输入 |
| `weight` | `(C,)` | float32 | 仿射缩放 |
| `mean` | `(C,)` | float32 | 前向保存 mean |
| `rstd` | `(C,)` | float32 | 前向保存 rstd |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `grad_x` | `(N, C, S)` | same_as(grad_out) | `apply` 的 `out_idx=[-1]`（末参数）自动分配 |
| `grad_weight` | `(C,)` | float32 | finalize 写出（host 预分配传入） |
| `grad_bias` | `(C,)` | float32 | finalize 写出（host 预分配传入） |

> 三段中间工件（非 op 输出）：`partial_sum` / `partial_sum_xhat` 形状 `(C, splits)` fp32；`scale` / `shift` / `centered_coef` 形状 `(C,)` fp32。均由 wrapper 预分配。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 段 | 用途（角色复用时序） |
|-----------|-------|-------|----------|----|---------------------|
| `stage` | `(bN, bS)` | dtype | UB | stats/apply | g 中转 → x 中转 → gx 舍回中转 |
| `g32` | `(bN, bS)` | float32 | UB | stats/apply | grad_out 平面 fp32 |
| `xh32`/`x32` | `(bN, bS)` | float32 | UB | stats/apply | x32 → x̂（stats）；x−mean（apply） |
| `prod` | `(bN, bS)` | float32 | UB | stats | `g·x̂` 乘积 / 广播 scratch |
| `brc` | `(bN, bS)` | float32 | UB | stats/apply | `T.vbrc` 标量广播 scratch |
| `red`/`red2` | `(bN, 1)` | float32 | UB | stats | dims=[1] 行和累加器（跨 S-子块，跨 N-块 `gb` 再累加） |
| `gb`/`gw` | `(1, 1)` | float32 | UB | stats/finalize | 段部分和 / 通道和 |
| `m1`/`r1`/`w1` | `(1, 1)` | float32 | UB | stats/finalize/apply | 逐通道参数标量 |
| `pw`/`pwx` | `(1, splits)` | float32 | UB | finalize | 载入 workspace 行（合并输入） |
| `tot`/`totx` | `(1, 1)` | float32 | UB | finalize | 合并结果 |
| `sc`/`sh`/`ce` | `(1, 1)` | float32 | UB | finalize | scale / shift / centered_coef 预折叠 |
| `s1`/`h1`/`c1` | `(1, 1)` | float32 | UB | apply | scale / shift / center 标量 |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0/L1 参与）：
stats（每任务：N-块 nn 外层、S-子块 tt 内层）:
  GM[grad_out] --T.copy(双slice,rN×r2)--> UB[stage] --vcast(rint)--> UB[g32 (bN,bS)]
  GM[x]        --T.copy(双slice,rN×r2)--> UB[stage] --vcast(rint)--> UB[xh32 (bN,bS)]
  UB[xh32] --vbrc(m1)+vsub+vbrc(r1)+vmul--> UB[xh32 = x̂]·UB[g32] --vmul--> UB[prod]
  UB[g32]/UB[prod] --T.reduce(dims=[1],clear=False,size=[rN,r2])--> UB[red/red2 (bN,1)]
  UB[red/red2] --T.reduce(dims=[0],clear=False,size=[rN,1])--> UB[gb/gw (1,1)] --T.copy(标量slice)--> GM[partial_sum/partial_sum_xhat]
finalize:
  GM[partial_sum/...] --T.copy(1×splits)--> UB[pw/pwx] --T.reduce(dims=[1])--> UB[tot/totx] --T.copy--> GM[grad_bias, grad_weight]
  UB[w1]·UB[r1] --vmul--> UB[sc]; --vmul(L_inv)--> per; --vmul(tot)--> UB[sh]; --vsub(0,·)--> -per·gb; --(·totx·rstd)--> UB[ce] --T.copy--> GM[scale/shift/centered_coef]
apply（每日通道：N-块 nn 外层、S-子块 tt 内层，覆盖全 S）:
  GM[grad_out]/GM[x] --T.copy(双slice,rN×real)--> UB[stage] --vcast(rint)--> UB[g32/x32 (bN,bS)]
  UB[g32],UB[x32] + UB[m1/s1/h1/c1] --vbrc 广播 + vmul/vadd/vsub--> UB[g32 = grad_x fp32]
  UB[g32] --vcast(rint)--> UB[stage(dtype)] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环；三段间经 GM workspace partial_sum/partial_sum_xhat 与系数三片交接）
```

关键约束：v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1）；GM 不可直达 fragment。

### 4.5 UB 内存预算

预算目标：总驻留 ≤ UB 容量 192KB，取 `UB_SAFE_BYTES = 163_840`（160 KiB）——依据：同族宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` 的 held 档（`(N=8,C=64,S=1024)` fp16 分配 ≈144KB）与探针 `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` 真机编译运行 PASS；160KB 留 32KB 余量。BPE 口径：`BPE = itemsize + 16`——`stage` 中转缓冲对所有 dtype 均分配，故 fp16/bf16 = `2 + 16 = 18`、fp32 = `4 + 16 = 20` B/元素（16 = `g32/x32/prod/brc` 各 4）。**v1 二维驻留** = `bN·bS·BPE`（主缓冲）+ `bN·ROW_OVERHEAD`（`ROW_OVERHEAD=8`，`red`/`red2` 各 `(bN,1)` fp32）+ `SMALL=64`（`(1,1)`/`(1,splits)` 小项）。`bN`/`bS` 的选取保证 `bN·bS·BPE + bN·8 + 64 ≤ 163_840`。

**bN/bS 公式（§3.3 工厂同源）**：

- `bS = min(S, align_down((UB_SAFE_BYTES - SMALL)//BPE, vw))`（`N=1`）；`bS = min(S, align_down((UB_SAFE_BYTES - SMALL)//(2·BPE), vw))`（`N>1`，`//2` 预留至少 2 行）。
- `bN = min(N, max(2, (UB_SAFE_BYTES - SMALL)//(bS·BPE + ROW_OVERHEAD)))`（`N>1`）；`bN = 1`（`N=1`）。`max(2,·)` 因 `bS·BPE ≤ (UB_SAFE-SMALL)//2 ⇒ 2·bS·BPE + SMALL ≤ UB_SAFE`，故 `bN=2` 恒安全。
- `n_tile_N = ceil(N/bN)`；`n_tile_S_stats = ceil(max_seg/bS)`（`max_seg = dist + (rem>0)`）；`n_tile_S_apply = ceil(S/bS)`。

| 段 | workload (N,C,S) | dtype | bN | bS | 驻留 (B) | n_tile_N | n_tile_S_stats | n_tile_S_apply |
|----|------------------|-------|----|----|----------|----------|----------------|----------------|
| stats | large-spatial (4,128,262144) | fp16 | 2 | 4544 | 2·4544·18+2·8 = 163,600 | 2 | 15 | — |
| stats | test (4,64,4096) | fp16 | 2 | 4096 | 2·4096·18+16 = 147,472 | 2 | 1 | — |
| stats | 大 N (8192,128,8) | fp16 | 1077 | 8 | 1077·8·18+1077·8 = 163,704 | 8 | 1 | — |
| stats | fp32 (4,256,65536) | fp32 | 2 | 4092 | 2·4092·20+16 = 163,696 | 2 | 16 | — |
| stats | N=1 (1,64,131072) | fp16 | 1 | 9096 | 9096·18+8+64 = 163,800 | 1 | 15 | — |
| finalize | 任意 | fp32 | — | — | 2·(1·splits)·4 ≈ 4·splits | — | — | — |
| apply | large-spatial (4,128,262144) | fp16 | 2 | 4544 | （复用 stats 主缓冲口径，实际更小） | 2 | — | 58 |

> **bS/bN 推导（fp16、BPE=18、SMALL=64、ROW_OVERHEAD=8）**：
> - large-spatial（N=4>1）：`bS = align_down((163_840-64)//(2·18), 8) = align_down(4549, 8) = 4544`；`bN = min(4, 163_776//(4544·18+8)=2) = 2`；驻留 `2·4544·18 + 16 = 163,600 ≤ 163,840` ✓。`dist=262144//4=65536, rem=0 ⇒ max_seg=65536`，`n_tile_S_stats = ceil(65536/4544) = 15`，`n_tile_S_apply = ceil(262144/4544) = 58`（apply 覆盖全 S，v1 修复）。
> - 大 N（N=8192,S=8）：`splits=min(65536, ceil(512/128)=4)=4`，`dist=2, rem=0`；`bS = min(4544, S=8) = 8`；`bN = min(8192, 163_776//(8·18+8)=1077) = 1077`；驻留 `1077·8·18 + 1077·8 = 163,704 ≤ 163,840` ✓（v0 在此 shape 因 `bS=8` 而 `(N,bS)` 超 UB，v1 由 N-tiling 化解）。`n_tile_N = ceil(8192/1077) = 8`。
> - fp32（BPE=20）：`bS = align_down((163_776)//(2·20), 4) = align_down(4094, 4) = 4092`；`bN = min(4, 163_776//(4092·20+8)=2) = 2`；驻留 `2·4092·20 + 16 = 163,696` ✓。
> - N=1：`bN=1`，`bS = align_down(163_776//18, 8) = 9096`；驻留 `9096·18 + 8 + 64 = 163,800` ✓。
> - **口径单一来源**：表值即 §3.3 工厂公式值（同 BPE、同对齐、同 `bN`），可互推。workspace `partial_sum/partial_sum_xhat` 各 `C·splits·4`（GM，非 UB；large-spatial `128·4·4 = 2048 B`）。

默认公式同 §3.3；`bN≥2`（`N>1`）保证 `dims=[0]` 的 N 归约合法，`bN·bS·BPE + bN·8 + 64 ≤ UB_SAFE` 保证容量。**注**：auto-multibuffer 可能膨胀（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd` 同族实测参考 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md` 的 TRAP-UB-multibuffer-inflation：transit ≈22 B/元素、fp32 ≈25.8 B/元素）；本设计取 §3.3 公式值 + Stage 4 实测，超预算即降 `bS`（并相应增大 `bN` 覆盖）。

### 4.6 动态轴定义

无运行时动态轴。N、C、S、splits、dist、rem、max_seg、bN、bS、n_tile_N、n_tile_S_stats、n_tile_S_apply 均为工厂期常量（per-build JIT 特化，源 lru_cache 工厂同款语义）；段/子块 extent `real`/`rN`/`r2` 是 kernel 内 `T.min` 标量。op 层把任意 `(N, C, *spatial)` view 成 `(N, C, S)`——host 元数据视图操作。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir")            # stats（无 out_idx：workspace 由 host 预分配传入）
@tilelang.jit(target="npuir")            # finalize（无 out_idx）
@tilelang.jit(out_idx=[-1], target="npuir")  # apply：仅 grad_x 自动分配 -> 返回 grad_x
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：仅 `T.reduce` + v-prefix 逐元素运算 + `T.vbrc` 广播，无 matmul。

### 5.2 Block 划分

```python
splits = min(max(1, min(N * S, ceil(512 / C))), S)   # 段数（= wrapper _split_seed，cap 到 S）
dist, rem = S // splits, S % splits                  # 余额分配：每段 real = dist 或 dist+1 ≥ 1
max_seg = dist + (1 if rem > 0 else 0)               # 最长段 S 长度
# 二维 (bN, bS) tiling（§4.5 公式）：
if N == 1:
    bN = 1
    bS = min(S, align_down((UB_SAFE_BYTES - SMALL) // BPE, vw))
else:
    bS = min(S, align_down((UB_SAFE_BYTES - SMALL) // (2 * BPE), vw))
    bN = min(N, max(2, (UB_SAFE_BYTES - SMALL) // (bS * BPE + ROW_OVERHEAD)))
n_tile_N = ceil(N / bN)                              # N-子块数（静态）
n_tile_S_stats = ceil(max_seg / bS)                  # stats 段内 S-子块上界（v1）
n_tile_S_apply = ceil(S / bS)                        # apply S-子块上界，覆盖全 S（v1）
# stats：逻辑任务 = C*splits
num_kernels_s = min(C * splits, 40)                  # 实启动内核数
num_local_tasks_s = ceil(C * splits / num_kernels_s) # 核内串行任务数（静态）
# finalize / apply：逻辑任务 = C
num_kernels_c = min(C, 40)
num_local_tasks_c = ceil(C / num_kernels_c)          # 静态
```

**选择理由**：归约轴 `N·S` 的维度上——S（尾轴）作为**向量化/归约内维**（连续 stride=1），N 作为外层 reduce 维；通道沿 S 轴切成 `splits` 个仿射子段（**余额分配**，每段非空）以提升并行度（`C·splits` 逻辑核）；`(bN,bS)` 二维子块保证 UB 有界（超大 N 与超大 S 两侧都覆盖，v1，§0.0 阻塞项 3）；`n_tile_S_stats` 与 `n_tile_S_apply` 分别定义（v1，§0.0 阻塞项 1）。

### 5.3 约束分析

- **对齐约束**：UB 32B 对齐由编译器管理（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；逐通道平面行内 stride=1；S 尾 lane 浪费率见 §1.6.3（manifest shapes 均 0%）。
- **UB 容量**：§4.5 表，段内 `(bN,bS)` 驻留 ≤ `bN·bS·BPE + bN·8 + 64 ≤ 163,840B`✓（大 N 由 N-tiling 覆盖）；workspace 落 GM。
- **L0/L1/分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **尾段/尾 N-块/尾子块（v1）**：段 extent 用**余额分配** `real = s1-s0`（恒 ≥ 1，`s1=(sp+1)·dist+T.min(sp+1,rem)`）；N-块 extent `rN = T.min(bN, N-n0)`；S-子块 extent `r2 = T.min(bS, s1-ss)`（`if r2 > 0` 守卫，尾子块可能空）；装载/写出双显式 slice 截断 **且** reduce 传 `size=[rN, r2]`（dims=[1]）与 `size=[rN, 1]`（dims=[0]）限定有效范围——I/O 与归约两侧都须截断，缺一即把垃圾计入部分和。
- **N=1（单 batch）**：`bN=1`，`(1, bS)` 平面；`dims=[0]` reduce 在 size-1 上退化（`(1,1)→(1,1)`），工厂层 `if N > 1:` 跳过（等价地 copy `red→gb`）。
- **S=1（v1 修复建议 5）**：`bS = min(…, S) = 1`，平面 `(bN,1)`；`dims=[1]` reduce 退化 → 工厂层 `if S > 1:` 跳过第一层 reduce，改 `T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])`（`prod→red2` 同改；切片保证形状匹配）。注：split kernel 一般不服务 S=1（`_holder` 判 `thread`），此处仅作鲁棒性兜底。
- **splits=1（退化）**：`T.reduce(pw, tot, dims=[1])` 在 size-1 上退化 → 工厂层 `if splits > 1:` 跳过，直接 `T.copy(pw, tot)`。
- **`splits` 上限 cap 到 S + 余额分配**：`_split_seed` 可能 > S（小 S 大 N），cap 到 S ⇒ `dist = S//splits ≥ 1` ⇒ 每段 `real ∈ {dist, dist+1} ≥ 1`，恰 `splits` 个非空段且恰好覆盖 `[0, S)`（**v1 修正**：v0 的 `min(chunk, S-s0)` 在 `splits > ceil(S/chunk)` 时可产生 0/负 extent，§0.0 阻塞项 2）。极端时 `splits=1`（整通道一段）。
- **三段共用 `splits`**：工厂内的 `splits` 必须与 wrapper 预分配 workspace 的 `splits` 一致，否则 workspace 形状错位（§0.2 注）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数**：**20**（AI Core，Ascend910B3）。实查记录（2026-10-14，本设计执行）：

```python
from tilelang.utils.npu_utils import NPUUtils
n = NPUUtils.get().get_aicore_num()   # -> 20
vector_cores = n * 2                  # 纯 Vector 算子翻倍 -> 40
```

  与 `constants.md` CONST-aicore-910B2C 的 2026-10-12 注（本机 `get_aicore_num()` 返回 20，设备档 910B3）一致；禁止按 24 假设。
- **逻辑核数**（`num_logical_kernels = ceil(M/block_M) × ceil(N/block_N)` 口径，本算子三段分别计算）：

| 段 | workload | shape (N,C,S) | dtype | splits | 逻辑核数 | num_kernels=min(logical,40) | num_local_tasks |
|----|----------|---------------|-------|--------|---------|-----------------------------|-----------------|
| stats | large-spatial | 4×128×262144 | fp16 | 4 | C·splits = 512 | 40 | 13 |
| finalize | large-spatial | 4×128×262144 | fp16 | 4 | C = 128 | 40 | 4 |
| apply | large-spatial | 4×128×262144 | fp16 | 4 | C = 128 | 40 | 4 |

> **选核口径**：本核仅服务 `_holder` 判为 `split` 的 shape（wrapper `select_bwd_kernel`：`spatial ≥ _SPLIT_MIN_L(65536)` 且 `C < _SPLIT_MAX_C(1024)`）。manifest 中 **large-spatial `[4,128,262144]`** 路由到 split；smoke-fc/resnet-stage1/2/3 因 `_block_launch` 可行被路由到宽核（`bwd_wide`），不在本核服务范围（见 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` §5.5）。其余 shape 路由 general `bwd_kernel`。

- **规模判定与分核方案**：manifest split 负载 `C ∈ {128}`（`C < 1024`）——三段逻辑核数（stats 512、finalize/apply 128）均 > 40 → **极大规模分支**——固定启动 `num_kernels = 40`，核内 `T.serial(num_local_tasks)` grid-stride（stats `t = i·40 + cid`；finalize/apply `c = i·40 + cid`），`if t < C*splits`/`if c < C` 守卫；**循环边界静态**（`num_local_tasks` 为 host 层编译期常量）。每个 stats/apply 任务内部再 `T.serial(n_tile_N)` × `T.serial(n_tile_S_stats|n_tile_S_apply)` 遍历二维子块（v1 N-tiling；均静态边界）。负载不均：stats 512 = 40·12 + 32（前 32 核各多 1 任务，≤ 2.5%）；finalize/apply 128 = 40·3 + 8。该形态消除逻辑核超发的串行调度开销（`core-split-strategy.md` §1），并获 grid-stride 聚集窗口的 MTE2 访问效益（`constants.md` CONST-mte2-degradation 实测 +3.2%）。**跨段合并完全经 GM workspace + 二阶段 kernel（无 atomics、无核间 flag）**；通道归约在段内单核闭环（每段部分和独立写 `partial_sum[c, sp]`，无写冲突）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block/tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| stats 任务维 | 核级并行 + 核内串行 | `T.Kernel(min(C*splits,40))` + `T.serial(num_local_tasks_s)` | persistent 分核（§5.5）；任务 = (通道, 段) |
| stats 段内 N-子块维 | 核内串行 | `T.serial(n_tile_N)` | UB 有界（§0.6 R2；v1 N-tiling）；静态 |
| stats 段内 S-子块维 | 核内串行 | `T.serial(n_tile_S_stats)` + `if r2 > 0` | UB 有界（§0.6 R2）；`n_tile_S_stats` 静态 |
| finalize 通道维 | 核级并行 + 核内串行 | `T.Kernel(min(C,40))` + `T.serial(num_local_tasks_c)` | persistent 分核 |
| finalize 段合并维 | 归约化 | `T.reduce(pw, tot, dims=[1])`（splits>1；硬件定序树） | 跨段合并（§0.6 R3） |
| apply 通道维 | 核级并行 + 核内串行 | `T.Kernel(min(C,40))` + `T.serial(num_local_tasks_c)` | persistent 分核 |
| apply N-子块维 | 核内串行 | `T.serial(n_tile_N)` | UB 有界（§0.6 R4；v1） |
| apply S-子块维 | 核内串行 | `T.serial(n_tile_S_apply)` | UB 有界（§0.6 R4）；**覆盖全 S（v1）** |
| 元素维（S） | 向量化（无循环） | `T.copy`/`T.vcast`/v-op/`T.reduce` 全 buffer 操作 | §1.6.2 #1-#9 |
| 尾段/尾 N-块/尾子块 | 标量 min + slice（I/O）+ size（归约） | `T.min` + 双显式 slice + `T.reduce(size=[rN, r2])`/`(size=[rN, 1])` | §1.6.2 #13；reduce 须同步限 extent |

### 6.2 循环伪代码

见 §3.3（stats：`with T.Kernel(KS, is_npu=True) as (cid, _):` → `with T.Scope("Vector"):` → 缓冲分配（`T.alloc_ub`，serial 外，跨任务复用）→ `for i in T.serial(nlS):` → `if t < n_tasks:` → 余额分段 `s0/s1` → `for nn in T.serial(n_tile_N):`（v1）→ `for tt in T.serial(n_tile_S_stats):`（`if r2>0`）累加归约 → N-块 `dims=[0]` 累加 → 写 workspace；finalize：`for i in T.serial(nlC):` → `if c < C:` → 载行 `T.reduce(dims=[1])` 合并 + 系数预折叠；apply：`for i in T.serial(nlC):` → `if c < C:` → `for nn in T.serial(n_tile_N):` → `for tt in T.serial(n_tile_S_apply):`（覆盖全 S）epilogue 写 grad_x）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内为顺序依赖链（装载→cast→x̂→累加→reduce→写），无 K 维迭代可重叠；任务间/子块间装载/计算 overlap 交由编译器 auto multi-buffer。Stage 4 选项：显式双缓冲（预取下一 S-子块）或 `T.Pipelined` 包裹 serial 循环——若 baseline 偏离 §1.6.0 估算下界 >2x 再评估。

### 6.4 尾块处理

**stats 尾段/尾 N-块/尾子块（v1）**：段用余额分配（`real = s1-s0`，恒 ≥ 1）；`bS` 不整除段长时 `r2 < bS`、`bN` 不整除 N 时末 N-块 `rN < bN`——装载/写出双显式 slice 截断 **且** 两处 `T.reduce` 传 `size=[rN, r2]`（dims=[1]）与 `size=[rN, 1]`（dims=[0]，仅对有效行/列求和；否则垃圾污染部分和）；空尾子块由 `if r2 > 0` 守卫。**apply 尾子块**：同样双 slice + `T.min` 截断写出（apply 无归约，无需 `size`）。**通道尾任务**：`C % num_kernels ≠ 0` / `C*splits % num_kernels ≠ 0` 时 `if` 守卫跳过越界任务。**N=1 / S=1 / splits=1**：工厂层分支跳过退化的 reduce 层（§5.4）；`N>1` 保证 `bN ≥ 2` 使 `dims=[0]` 合法。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：Expert（显式 UB + 每段单 `T.Scope("Vector")`）；无核间/跨引擎协作标志。

### 7.2 同步点说明

**核内**：Expert 模式下 UB 分配与数据放置显式，但**单个 Vector scope 内的顺序依赖链仍由编译器按 MTE2→V→MTE3 拓扑自动插 pipe_barrier**（`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` 同款 Expert/Vector 形态；本函数设计探针三段顺序书写、未手写 sync 即编译运行）。无需 `T.set_flag/T.wait_flag`。

**三段间（部分和交接）**：`stats` 写 GM workspace `partial_sum/partial_sum_xhat` → `finalize` 读 → 写 `scale/shift/centered_coef` → `apply` 读。**NPU 单 grid 无全局 barrier**，故跨段交接不靠核内同步，而靠**同一流上顺序下发的三次 kernel 调用**（运行时序：上一 kernel 完成才启动下一 kernel）——这正是「GM 部分和 workspace + 二阶段 kernel」相对「单 kernel persistent 两相位 + atomics」的选择（§0.6 R3）：**既无 atomics，也无 `T.sync_block_set/wait` 核间 flag**。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 后 | 编译器自动 | MTE2→V 拓扑自动 pipe_barrier |
| v-op / reduce / vbrc / vcast 之间 | 编译器自动（顺序执行） | 同 Vector 引擎顺序依赖 |
| `T.copy`(UB→GM) 前 | 编译器自动 | V→MTE3 拓扑自动 pipe_barrier |
| stats→finalize（workspace 读） | 运行时流序（kernel 顺序） | 单 grid 无全局 barrier；顺序下发保证 stats 完成 |
| finalize→apply（系数读） | 运行时流序（kernel 顺序） | 同上 |
| 段内任务/子块间 | 编译器自动 | 同核串行，无并发 |
| 跨核（同段内不同通道/段） | 无需 | 每 (通道,段) 独立写自己那格 workspace，无写冲突 |

**显式同步备选**：若 Stage 3 引入显式双缓冲（预取下一 S-子块）后实测出现 WAR/RAW 冒险，按 `.agents/skills/tilelang-op-design/references/decision-tree.md` 用 `T.set_flag`/`T.wait_flag` 管理；本设计默认不引入（见 §6.3）。

### 7.3 pass_configs 配置

无特殊 pass_configs。Expert Vector scope 默认编译流水线（`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` 同款）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`），**不复刻 §0.6 的 NPU 算法**（无 vcast / fp32 中转 / persistent / 拆分部分和 / finalize 合并 / 两步 reduce / 分布式 epilogue）——保证验证独立性。**golden 直接调用 `torch.*` 模块函数**（`torch.sum`/`torch.rsqrt` 等），不用纯张量方法链（满足 Stage 3 `S3-GOLDEN-TORCH` 的「须有 `torch.` 调用」AST 校验）。

```python
import torch

def golden_batch_norm_bwd(grad_out, x, weight, mean, rstd):
    """独立纯 PyTorch CPU BN 反向（fp32 中间量 + 舍回输入 dtype）。

    输入: grad_out/x (N, C, S) 同形同 dtype；weight/mean/rstd (C,) fp32
    输出: (grad_x, grad_weight, grad_bias) —— grad_x 同输入 dtype；通道梯度 fp32
    语义（§0.1）: x_hat = (x - mean) * rstd；grad_bias = sum_{N,S}(grad_out)；
                  grad_weight = sum_{N,S}(grad_out * x_hat)；
                  grad_x = (w*rstd/L) * (L*grad_out - grad_bias - x_hat*grad_weight)
    """
    N, C, S = grad_out.shape
    L = N * S
    shape = (1, C, 1)
    x_hat = (x.float() - mean.view(shape)) * rstd.view(shape)
    grad_bias = torch.sum(grad_out.float(), dim=(0, 2))            # (C,)
    grad_weight = torch.sum(grad_out.float() * x_hat, dim=(0, 2))  # (C,)
    gb = grad_bias.view(shape)
    gw = grad_weight.view(shape)
    grad_x = (weight.view(shape) * rstd.view(shape) / L) * (
        L * grad_out.float() - gb - x_hat * gw
    )
    return grad_x.to(x.dtype), grad_weight, grad_bias
```

**独立性说明**：golden 直接调用 `torch.sum` 归约与逐元素运算，不含 NPU 设计中的 persistent 分核、S 轴拆段、部分和 workspace、finalize 合并、两步 `T.reduce`、`T.vbrc` 广播、fp32 中转链、分布式 epilogue 等任何算法特征；NPU 算法若与 golden 数值一致即证明迁移正确。另可用 `examples/TileOPs/tests/ops/test_batch_norm_bwd.py` 的 `BatchNormBwdTest` 的 `gen_inputs`（用 `torch.var_mean` 从 `x` 生成自洽的 mean/rstd）作为入口。

### 8.2 精度标准

来自 `examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_bwd_tolerance`（`grad_x` 逐元素；通道梯度用更宽的累加界 `sum_atol`）：

| dtype | atol | rtol | sum_atol（通道梯度） |
|-------|------|------|----------------------|
| float32 | 1e-5 | 1e-5 | 1e-4 |
| float16 | 1e-2 | 1e-2 | 1e-2 |
| bfloat16 | 2e-2 | 2e-2 | 2e-2 |

fp32 中转链对齐 golden opmath（D-1 verify_equiv：fp16/bf16 舍回后逐位一致）；探针最坏 max_abs_err 4.883e-04（fp16 grad_x）在容差内；本函数设计探针 fp32 grad_x 9.5e-07。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」（完整 L1/L2/Boundary 分层套件交由 tilelang-op-develop）。对照 = §8.1 golden（CPU）。

| # | Shape (N, C, S) | dtype | 覆盖 | 验证目标 | 容忍度 |
|---|-----------------|-------|------|---------|--------|
| L0-1 | (4, 128, 262144) | fp16 | split 主负载 | large-spatial（splits=4，dist=65536，bN=2/bS=4544；stats 尾子块、apply 尾子块；**非整除归约路径**） | 1e-2 |
| L0-2 | (4, 64, 4096) | fp16 | split 中档 | dist=512（splits=8），多段；bS=4096 覆盖全段 | 1e-2 |
| L0-3 | (8, 8, 65536) | fp16 | 小 C 高并行 | C=8 < 40 → split 提升并行度的目标场景（logical=32） | 1e-2 |
| L0-4 | (4, 128, 262144) | bf16 | fp32 中转 | large-spatial bf16（fp16/bf16 中转链） | 2e-2 |
| L0-5 | (4, 256, 65536) | fp32 | fp32 直算 | 段长非整除 + fp32 路径（bN=2/bS=4092） | 1e-5 |
| L0-6 | (1, 64, 131072) | fp16 | N=1 退化 | `dims=[0]` reduce 跳过路径（N=1，bN=1） | 1e-2 |
| L0-7 | (4, 4, 65537) | fp16 | 素数 S、尾段 | `S % dist ≠ 0`（splits=min(4,4)=4，dist=16384，rem=1，首段 16385）——**余额分配回归** | 1e-2 |
| L0-8 | (2, 3, 65536) | fp16 | 小 C 小 N | C=3（末核守卫）、N=2 | 1e-2 |
| L0-9 | (4, 128, 65536) | fp16 | splits 退化风险 | 验证 `splits=min(_split_seed,S)` 与 workspace 形状一致（splits=4） | 1e-2 |
| L0-10 | (2, 8, 0) | fp16 | 空张量 | `L=0`：须由 op 层 `numel()==0` 早退拦截（§9.1），kernel 不应收到 `S=0` | 精确零 |
| **L0-11** | **(13108, 128, 5)** | **fp16** | **均衡段（v1 阻塞项 2）** | `_holder` 判 split（L=65540≥_SPLIT_MIN_L）、splits=min(4,5)=4、dist=1/rem=1 → 段 real=[2,1,1,1]；**v0 形态在 sp=3 产生 real=-1 非法切片必失败**，v1 余额分配通过 | 1e-2 |
| **L0-12** | **(8192, 128, 8)** | **fp16** | **大 N UB（v1 阻塞项 3）** | split 服务区内大 N：v0 `bS=8` 的 `(N,bS)` 四路 fp32 超 UB 编译失败；v1 N-tiling（bN=1077, n_tile_N=8）编译通过 + 精度通过 | 1e-2 |
| **L0-13** | **(4, 128, 262144)** | **fp16** | **apply 全 S 覆盖（v1 阻塞项 1）** | splits=4>1：**显式逐元素比对整张 `grad_x`（含 `[ceil(dist/bS)·bS:]` 区段）**；v0 形态 apply 只写前 `ceil(dist/bS)·bS=ceil(65536/4544)·4544=15·4544=68,160` 列 → 尾部 `grad_x` 保持未初始化必失败；v1 `n_tile_S_apply=58` 覆盖全 S 通过 | 1e-2 |

**通过条件**：全部项通过 atol/rtol（`grad_x` 逐元素 + `grad_weight/grad_bias` 用 `sum_atol`）。L0-1/L0-2/L0-7 为尾子块回归必测点；**L0-11 门住阻塞项 2（均衡段）、L0-12 门住阻塞项 3（大 N UB）、L0-13 门住阻塞项 1（apply 全 S 覆盖）**（三者各自对应 v0 会失败/编译失败的路径）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **三段 workspace 契约** | 工厂 `splits` 与 wrapper 预分配 `[C,splits]` 不一致 → 形状错位 | 工厂取 wrapper 同源 `_split_seed` 语义；集成期核对（§0.2/§5.4） |
| **跨段交接依赖运行时流序** | 单 grid 无全局 barrier；若三段异流并发则读脏 | 三段由同一默认流顺序下发（§7.2）；集成期确认 wrapper 顺序调用 |
| **合并确定性** | atomics 累加顺序不定，破坏源确定性契约 | **不用 atomics**；workspace + 固定序 finalize（§0.6 R3） |
| **中间轴归约（非尾轴）** | 逐通道平面 strided gather | `T.copy` 双显式 slice（探针真机验证）；行跨步 `C·S` |
| **bf16 无 Vector 算术** | bf16 直算编译失败 | fp32 中转链（§0.6 R6，必需） |
| **N=1 / S=1 / splits=1 的退化 reduce** | size-1 轴上 reduce 非法（无维度变化） | 工厂层 `if N>1` / `if S>1` / `if splits>1` 分支跳过（§5.4） |
| **二维 UB 预算（v1）** | `bN`/`bS` 超预算 / auto-multibuffer 膨胀 | §4.5 二维公式 `bN·bS·BPE + bN·8 + 64 ≤ UB_SAFE`；回退更小 `bS`（相应增大 `bN` 覆盖；编译期溢出报错为天然 guard） |
| **大 N UB 溢出（v1 阻塞项 3）** | 只按固定 N 取 `bS` 时大 N shape（如 `(8192,128,8)`）超 UB | N-tiling：`bN = min(N, max(2, (UB_SAFE-SMALL)//(bS·BPE+8)))`；`N>1 ⇒ bN≥2` 使 `dims=[0]` 合法；L0-12 门住 |
| **apply 只覆盖部分 S（v1 阻塞项 1）** | 复用 stats 的 `n_tile_S_stats` 时 apply 只写前 `ceil(max_seg/bS)·bS` 列 | `n_tile_S_apply=ceil(S/bS)` **独立定义**（§3.2/§3.3）；L0-13 显式比对整张 `grad_x` 门住 |
| **段 extent 非空（v1 阻塞项 2）** | `s0=sp·chunk`+`min(chunk,S-s0)` 在 `splits>ceil(S/chunk)` 时产生 0/负 extent | 余额分配 `dist/rem` 保证每段 `real≥1`、恰覆盖 `[0,S)`（§5.4）；L0-11 门住 |
| **未初始化垃圾列/行（尾子块/尾 N-块）** | 尾子块 `[r2:]` 列 / 尾 N-块 `[rN:]` 行为未初始化/残留垃圾，无边 reduce 会将其计入 | 归约传 `size=[rN, r2]`（dims=[1]）与 `size=[rN, 1]`（dims=[0]）截断（`docs/Tilelang.language/规约操作/T.reduce.md` §2.1）；L0-1/L0-2/L0-7 门住 |
| **`splits` cap 到 S（并行度）** | 小 S 大 N 时 cap 后并行度不及 `C*splits` | 仅影响并行度；**正确性由余额分配保证（`real≥1`）**（§5.4/§1.6.3 #3）；一般负载 S 大 |
| **`dims=[0]` reduce 组合（v1 新增）** | `T.reduce(dims=[0], clear=False, size=[rN,1])` 组合未在本探针真机验证 | Stage 3 真机核对；不支持则回退 `reduce(clear=True)+T.vadd`（§3.3 注） |
| **空张量 `L=0`** | NPU op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py` 无 `numel()==0` 早退 → 空张量进入 kernel 构建（`S=0` 的 `T.copy`/`T.min`/`T.reduce` 退化非法） | 集成期在 NPU op/wrapper 增补 `numel()==0` 早退（返回 `empty_like(x)` + 零通道梯度，对齐 GPU op 语义，§0.1/§9.3/L0-10） |
| **roofline 常数 stale** | §1.6.0 估算引用 CONST-* 被 kb_stale_check 标记 | 估算仅作排序与 Stage 4 对账 |
| **三段负载不均** | stats 512 = 40·12+32；finalize/apply 128 = 40·3+8 | 不均 ≤ 2.5%；Stage 4 可评估分核粒度 |
| **inf/NaN 传播** | 全 inf 通道 → NaN 输出（公式自然传播） | 与 golden/源一致；测试用例为有限随机输入 |
| **wrapper `ascend_mode` 不一致** | Stage 0 wrapper `batch_norm_bwd.py` 现为 `ascend_mode = "Developer"`，本设计选 Expert | 集成期（Stage 5）把 wrapper 的 `ascend_mode` 改为 `"Expert"`；实现期 `ASCEND_MODE = "Expert"`（§2.3） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | bS 超预算 / auto-multibuffer 膨胀 | 编译失败 `ub overflow` | 降 bS；§4.5 公式 + guard |
| bf16 直传 v-op/reduce | 未做 vcast fp32 | 编译错误（dtype 不支持） | fp32 中转链 |
| reduce 退化 | N=1 / S=1 / splits=1 仍做对应 dims reduce | MLIR verify fail（无维度变化） | 工厂层 `if` 分支 |
| `clear=False` 未初始化 | 跨子块累加前未 `T.clear(red)` | 静默数值错误（`docs/Tilelang.language/规约操作/T.reduce.md` §2.3） | 累加前 `T.clear` |
| 尾子块/尾 N-块 reduce 垃圾列行 | 对完整 `(bN,bS)`/`(bN,1)` 做 reduce 未限 extent | 未初始化/残留列/行计入部分和 | reduce 传 `size=[rN, r2]`（dims=[1]）/`size=[rN, 1]`（dims=[0]）；L0 非整除用例门住 |
| apply 子块数复用 stats | apply 用 `n_tile_S_stats` 而非 `ceil(S/bS)` | `grad_x` 尾部列不写出（静默错误） | 分别定义 `n_tile_S_stats`/`n_tile_S_apply`；L0-13 比对整张 `grad_x` |
| 段 extent 取 `min(chunk,S-s0)` | 用 chunk 切分而非余额分配 | `splits>ceil(S/chunk)` 时负 extent → 非法切片/崩溃 | 余额分配 `dist/rem`（real≥1）；L0-11 |
| 大 N 不做 N-tiling | 只按固定 N 取 `bS` | 大 N shape 超 UB 编译失败 | 二维 `(bN,bS)` tiling；L0-12 |
| workspace `splits` 不一致 | 工厂 splits ≠ wrapper splits | 形状错位/越界 | 同源 `_split_seed`；集成核对 |
| 跨 dtype `T.copy` 当 cast | GM fp16 → UB fp32 一步 | 隐藏 VCast 舍入不可控（`docs/Tilelang.language/内存操作/T.copy.md` §3） | 显式 `T.vcast(rint)` + 同 dtype copy |
| `min()` 内建函数置于 trace 体 | prim_func 内调用 Python `min(a,b)` | trace 期报错（`Cannot use and/or/not operator to Expr`） | 编译期常量（`K/nl/dist/rem/bS/bN/n_tile_*` 等）在工厂层算，trace 体内只用 `T.min` 或已算好的常量 |
| if 体内新定义变量跨块引用 | TVM parser 作用域 | Undefined variable | if 体内只写预分配 buffer；dtype/held 分支置于工厂层 |
| 通道-段索引常量折叠遗漏 | `num_local_tasks` 非常量 | 动态循环边界 | host 层编译期常量（§5.5） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| large-spatial（S=262144） | split 主负载：splits=4，dist=65536，bN=2/bS=4544（stats `n_tile_S_stats=15`、apply `n_tile_S_apply=58`、`n_tile_N=2`）；尾子块/尾 N-块 `size`/slice 截断 |
| 小 C（C=8） | split 提升并行度（logical=C·splits=32），target 场景；L0-3 覆盖 |
| 小 S 大 N（如 (13108,128,5)） | 余额分配段 `dist=1/rem=1 → real=[2,1,1,1]`（**非空**，v1 阻塞项 2）；split 仍被 `_holder` 选中（L≥65536）；L0-11 覆盖 |
| 大 N 小 S（如 (8192,128,8)） | N-tiling：bN=1077、n_tile_N=8（v1 阻塞项 3），`(bN,bS)` 驻留 ≤ 预算；L0-12 覆盖 |
| S=1（退化） | `bS=min(…,S)=1`，平面 `(bN,1)`；跳过 `dims=[1]` reduce，改 `T.copy(g32[0:rN,0:1], red[0:rN,0:1])`（v1 修复建议 5）；split 一般不由 `_holder` 选中（判 `thread`），仅鲁棒性兜底 |
| N=1 | `bN=1`，`(1, bS)`；`dims=[0]` reduce 退化 → 工厂分支跳过（copy `red→gb`）；L0-6 覆盖 |
| splits=1（退化） | finalize 跳过 `dims=[1]` reduce，直接 copy 部分和 |
| 3D 输入 `(N, C, H, W)` | op 层 reshape `(N, C, H*W)` 后调 kernel（host 元数据视图） |
| 空张量 `L=0` | 集成期由 NPU op/wrapper 早退（`numel()==0` → `empty_like(x)` + 零通道梯度）；kernel 不接收 `S=0`（§0.1/§9.1） |
| C > 40 | finalize/apply persistent 分核（§5.5）；核内 serial 多通道 |
| 零方差通道（x 常量） | `x̂ = 0`；`gw = 0`；grad_x = scale·g（公式自然）；verify_equiv 角点已覆盖 |
| 全 inf / NaN 通道 | 自然传播（与源/golden 一致）；verify_equiv 角点已覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # D-1 等价性机器验证脚本（已执行，EQUIV_PASS）
├── history_version/
│   └── design_probe_split_phase.py    # D-3 设计期探针（真机 PASS，三段结构验证）
├── _batch_norm_bwd_split_kernel.py    # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md                   # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | O1+O2 EQUIV_PASS（21 案例）；raw-moment 拒绝证据（判别段 A） |
| `history_version/design_probe_split_phase.py` | ✅ 已完成并执行（真机 PASS） | 三段结构（floordiv/mod 索引 / 部分和 workspace / finalize / apply）验证 |
| `_batch_norm_bwd_split_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + 三段 kernel + golden + L0 测试 |

### 10.3 命名规范

- 项目目录名：`batch_norm_bwd`；算子目录名 / 实现文件：`_batch_norm_bwd_split_kernel`（snake_case，不裁剪不变换）
- Golden 函数：`golden_batch_norm_bwd(grad_out, x, weight, mean, rstd)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）——本文档
2. ✅ 等价性机器验证（verify_equiv.py，D-1）+ 设计期探针（D-3）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_batch_norm_bwd_split_kernel.py`（§3.3 伪代码 → 代码，三段工厂）+ L0 门槛测试（§8.3 全 13 项，含 L0-11/12/13 三个阻塞项回归点）
5. ⬜ 精度比对（vs golden，三输出）；集成期同步 wrapper `ascend_mode="Expert"` + `numel()==0` 早退 + 三段顺序下发核对（§9.1）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 接口参数不变 | ✅ | 工厂签名 `(N, C, S, dtype)` 与源一致；三段 `@T.prim_func` 参数名/顺序逐字保持 |
| out_idx 契约 | ✅ | `stats`/`finalize` 无 out_idx（workspace/输出由 host 预分配）；`apply` 保持 `out_idx=[-1]` 返回 `grad_x`（对齐 wrapper 调用） |
| threads 移除 | ✅ | 移除 `threads/num_per_thread/blocks`（工厂闭包内烘焙） |
| host 侧不改输入 | ✅ | 全部计算在 kernel 内；op 层仅 reshape view（+集成期 `numel()==0` 早退） |
| 从源码推断规格（不问用户） | ✅ | §0.1/§0.2 从源码 + manifest 解读；输出 shape 已回填标注依据 |
| 合并方式声明 | ✅ | **GM workspace `[C,splits]` + 第二阶段 finalize kernel，无 atomics**（§0.6 R3 / §3.3 注 / §7.2） |
| 迁移纪律（不照抄 GPU flat map/线程树/threads） | ✅ | §0.5/§0.6 四态判定 + NPU 重设计 |

---

## 11. 修订日志

**revision_index: 1（revision）**（上一版 revision_index: 0 / first_design）

修订触发：Stage 2 `REVIEW.md`「结论: 不通过」（3 阻塞 + 3 建议）。本版保留 v0 全部通过项，仅作 §0.0 所列定点修正：

1. **阻塞项 1（apply 子块上界）** → `n_tile_S_stats`/`n_tile_S_apply` 分别定义，apply 覆盖全 S（§3.2/§3.3/§5.2/§6.1）；L0-13 门住。
2. **阻塞项 2（段 extent 空段/负值）** → 余额分配 `dist/rem`（real≥1，恰覆盖 `[0,S)`）+ `if r2>0` 守卫（§3.3/§5.2/§5.4/§9.1）；L0-11 门住。
3. **阻塞项 3（大 N UB 溢出）** → 二维 `(bN,bS)` tiling + `dims=[0]` N-块累加（§0.6 R2/§3.3/§4.5/§5.2）；L0-12 门住。
4. **建议 4（合并序表述）** → §0.5 行 #3 / §0.6 R3 统一为 `T.reduce(dims=[1])` 硬件定序树。
5. **建议 5（S=1 退化分支）** → `T.copy(g32[0:rN,0:1], red[0:rN,0:1])` 切片修正（§3.3/§5.4）。
6. **建议 6（L0 覆盖）** → 新增 L0-11/L0-12/L0-13（§8.3）。

**关键设计决策摘要**（v1；#3/#5/#6/#12/#13 为本次修订）：

| # | 决策 | 依据 |
|---|------|------|
| 1 | 三段结构（stats 分块部分和 / finalize 合并 / apply 逐通道） | §0.3 / §1.4 / 源三工厂契约 |
| 2 | **跨段合并 = GM workspace `[C,splits]` + 第二阶段 finalize kernel，无 atomics** | §0.6 R3 / §7.2 / 源「merge 顺序无关」契约 |
| 3 | 段切分沿 S 轴仿射子段，**余额分配** `dist/rem`（`c=t//splits`、`sp=t%splits`，cap 到 S，每段 `real≥1`）；向量化轴 S | §1.6.3 #1 / 设计探针真机 + §0.0 阻塞项 2 |
| 4 | 一维 persistent 分核（stats `min(C*splits,40)`；finalize/apply `min(C,40)`；实查 20→40） | §0.6 R1 / §5.5 |
| 5 | 段内 lane 向量化累加 + 两步 `T.reduce`（`dims=[1]` `size=[rN,r2]` + `dims=[0]` `size=[rN,1]`，`clear=False`） | §0.6 R2/R5 / 探针 + v1 |
| 6 | 尾段/尾 N-块/尾子块 `T.min` + src/dst 双显式 slice + reduce `size` 截断 + `if r2>0` 守卫 | §0.6 R5 / §0.0 阻塞项 2 |
| 7 | fp16/bf16 → fp32 中转链（显式 `T.vcast(rint)`）；bf16 必需 | §0.6 R6 |
| 8 | 分布式 epilogue + 倒数乘（O1/O2），D-1 EQUIV_PASS（21 案例） | §1.6.1 |
| 9 | 保留中心化 `x̂` 归约（拒绝 raw-moment，抵消证据 fp32 违反率 0.1553） | §1.6.1 判别段 A |
| 10 | 编程模式 **Expert**（每段 `T.Scope("Vector")` + `T.alloc_ub`） | §2 / `conv1d_direct` Expert/Vector 先例 |
| 11 | 三段共用同一 `splits`（= wrapper `_split_seed` 语义） | §0.2 / §5.4 |
| 12 | **二维 `(bN,bS)` tiling**（`bN=min(N,max(2,…))`；`N>1⇒bN≥2`）+ N-块间 `dims=[0]` 累加 | §0.0 阻塞项 3 / §4.5 / §5.2 |
| 13 | **apply/ stats 子块上界分别定义**（`n_tile_S_apply=ceil(S/bS)` 独立于 `n_tile_S_stats`） | §0.0 阻塞项 1 / §3.2 / §6.1 |

**revision_index: 0（first_design）**

首次设计（migration-harness，`_batch_norm_bwd_split_kernel`，2026-10-14）。无上一版；无历史否决路径。
