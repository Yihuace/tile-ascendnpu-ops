# prefill_chunk_local_cumsum_bthd_tl 算子设计文档

> 函数 1 / 6。所属算子：`GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`）。
> 工具链：**CANN 9.0.0** / tilelang 0.1.2（dev root）/ Ascend910B3。版本戳见 §8。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

把 `g`（log 空间衰减门）按 **chunk（默认 64 token）** 在各序列内部做**前向累积和**，
累积在**每个序列起点处重新开始**；结果以 **fp32** 写出。

源实现要点（`gdn/prefill_prepare.py:22`）：

- 累积必须在 **fp32** 里做：相邻累计值相减会放大舍入误差，若把累计值本身舍入到输入 dtype，
  后续 `exp(gc[l]-gc[s])` 与递归状态都会失真。
- 等长输入（`B` 条、长度 `S`）在字节布局上等价于 packed 输入（`1 × B·S`，offsets 步长 `S`），
  因此等长调用也走同一 kernel。
- 可选 `gate_in_kernel`：`g` 是原始门，内核内先算
  `-exp(A_log)·softplus(g + dt_bias)` 再累积（本函数本批 **不启用**，见 §0.5）。
- 网格覆盖"offsets 能描述的最大 chunk 数"，超出实际 chunk 数的块直接退出——
  使 launch 范围与 offsets 的具体取值无关。

### 0.2 源算子输入输出

| 名称 | 形状 | dtype | 说明 |
|---|---|---|---|
| `g` | `[1, total_tokens, head]` | fp16 / bf16 | log 空间衰减门 |
| `cu_seqlens` | `[N+1]` int32 | int32 | 序列起点；等长时步长恒为 `S` |
| `a_log` | `[head]` fp32 | fp32 | 仅 `gate_in_kernel=True` |
| `dt_bias` | `[head]` fp32 | fp32 | 仅 `gate_in_kernel=True` |
| `out` | `[1, total_tokens, head]` | **fp32** | 逐 chunk 前向累积和 |

### 0.3 实现算法解读（怎么算）

对每个 `(sequence, chunk, head_slab)`：

1. 由 `cid` 解码出 `(seq, first_row_in_chunk)`，`base = cu_seqlens[seq] + first`，`end = cu_seqlens[seq+1]`；
2. 把该 chunk 的 `chunk_size × head_tile` 门值读入共享内存（越界行填 0）；
3. 沿 **行方向串行** 扫描：`acc[hid] += chunk_s[i, hid]`，就地写回 `chunk_s[i, hid] = acc[hid]`；
4. 若 `gate_in_kernel`，第 2 步顺带做 `softplus` 变换；
5. 写回 `out`（fp32）。

`head_tile = min(head, 64)`，`head > 64` 时按 slab 切分。

### 0.4 优化手段解读（为什么快）

- **一次读一次写**：`gate_in_kernel` 的变换放在"本来就要读每个门值并写回"的那一趟里，
  不额外搬运字节。
- **chunk 留在 shared**：`chunk_size × head_tile` 的这一块（64 行 × ≤64 头，16 KiB 级）
  驻留片上，扫描不再回 GM。
- **头部向量化**：`head` 方向 `T.Parallel`，扫描只在行方向串行，避免逐元素标量循环。
- **launch 范围与数据无关**：多出的块立即退出，避免按 offsets 动态改网格。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 源实现手段 | 硬件前提 | NPU 处置 |
|---|---|---|
| `T.alloc_shared` 驻留 chunk | CUDA shared mem | → `T.alloc_ub`（UB 192 KB） |
| `T.Parallel(chunk_size, head_tile)` | CUDA 线程块 | → `T.Parallel(head_tile)` 逐行（Vector 核） |
| 行方向 `T.Serial` 扫描 | 线程内串行 | → `T.cumsum(ub, ub, dim=0)`（NPU 有 VCumsumOp） |
| `tilelang.cdiv` / `GroupTiling` | 与硬件无关 | 保留（纯 Python 侧） |
| `gate_in_kernel` 分支 | — | **本批不启用**（in-scope 参数全默认），保留接口 |

**决策**：利用 NPU 原生 `T.cumsum`（`hivm.VCumsumOp`）替换手写的行扫描，
把"64 行串行依赖"交给硬件，避免 64 次串行迭代。

### 0.6 NPU 算法重设计

```
grid = (num_chunks, head_slabs)                      # 一个 block 负责一个 (chunk, head_slab)
  seq, base, end = decode(cid, cu_seqlens)           # 等长：base = seq*S + first
  g_ub   = alloc_ub([chunk_size, head_tile], in_dtype)
  f32_ub = alloc_ub([chunk_size, head_tile], float32)
  out_ub = alloc_ub([chunk_size, head_tile], float32)
  load g[0, base:base+chunk_size, slab] -> g_ub      # 越界行置 0
  vcast(g_ub, f32_ub)                                # bf16/fp16 -> fp32（cumsum 不支持 bf16）
  T.cumsum(f32_ub, out_ub, dim=0)                    # 行方向前向累积和
  store out_ub -> out[0, base:base+chunk_size, slab]
```

- **不跨 chunk 累积**：每个 chunk 独立清零（与源实现一致）。
- **尾部**：`S % 64 != 0` 时最后一块的有效行按 `base + i < end` 掩码，其余填 0
  （填 0 不改变累积和）。
- `bf16` 必须先 `vcast` 到 fp32：`T.cumsum` 的 Ascend 支持表为 int32/int64/fp16/fp32，
  **不支持 bf16**（`docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1）。

### 0.7 标杆实现

源实现 `prefill_prepare.py:22`（CUDA，`T.Parallel` + 行串行扫描）；
NPU 侧同类 Vector 参考：`tileops/kernels/norm/batch_norm_bwd/*`（`T.alloc_ub` + `T.Scope("Vector")` 写法）。

## 1. 概述

### 1.1 算子名称

`prefill_chunk_local_cumsum_bthd_tl`（GDN prefill 流水线第 1 个内核）。

### 1.2 功能描述

逐 chunk、逐序列重置的门值前向累积和，fp32 输出。

### 1.3 数学公式

对第 `q` 个序列的第 `c` 个 chunk、第 `h` 个头：

```
out[base + i, h] = Σ_{j=0..i} g[base + j, h]      i ∈ [0, min(C, end-base))
out[base + i, h] = 0                              base + i ≥ end
```

`base = cu_seqlens[q] + c·C`。

### 1.4 算法描述

1. 每个 block 处理一个 `(chunk, head_slab)`；
2. 载入 `C × head_tile` 门值，越界行置 0；
3. 转 fp32 后调用 `T.cumsum(dim=0)`；
4. 写回 fp32。

### 1.5 数据流图

`GM(g, fp16/bf16) → UB(fp16/bf16) → vcast → UB(fp32) → cumsum(dim=0) → UB(fp32) → GM(out, fp32)`

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

候选（**基线候选 = 源实现**：行方向串行扫描 + 头部并行）：

| 候选 | 等价/化简 | 在线 | 复杂度 | 硬件亲和 | 评价 |
|---|---|---|---|---|---|
| **C0 源实现（基线）** | 直接按定义累加，无化简 | 否 | 时间 O(C·H)，空间 O(C·H) | CUDA 线程块；NPU 上需 64 次串行迭代 | 基线 |
| C1 NPU `T.cumsum` | 同一数学定义，用 VCumsumOp 一次算完 | 否（单块内完成） | 时间 O(1) 调用 + 片上搬移，空间 O(C·H) | **高**：原生 `hivm.VCumsumOp`，无需手工串行 | **采纳** |
| C2 分块并行扫描（Blelloch） | 等价，需跨块前缀传递 | 是 | O(C) 步、log 级深度 | 中：块间需 GM 往返与同步 | 弃：C=64 太小，收益不抵同步 |
| C3 按 head 切片整段扫描 | 等价 | 是 | O(S·H) | 中：单块跨 chunk，破坏"每 chunk 重置" | 弃：与语义冲突 |

**调研结论**：采纳 **C1**（`T.cumsum`）。基线 C0 的"64 次行串行"在 Vector 核上是纯延迟开销，
而 C1 用硬件累积和指令一次完成；C2/C3 的并行化对 C=64 的规模不值得引入跨块同步。

#### 1.6.1 数学等价优化（公式级）

C1 与基线 C0 是**同一个定义**（逐元素前缀和），不存在代数变换：
`Σ_{j≤i}` 的定义未被改写，只是执行载体从"手工串行循环"换成"硬件累积和指令"。

**优化结论**：本函数**无优化空间**（数学层面无等价化简；加速来自指令替换而非公式改写），
因此不引入 `verify_equiv.py`。唯一必须保证的数值性质是**累积在 fp32 中进行**——
这一点由 `vcast → cumsum(fp32) → 写出 fp32` 保证，并在 §7 的 L0/L1 用 fp64 参考校验。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

- 源实现的 `T.Parallel(chunk_size, head_tile)` 二维并行 → NPU 改为**一维头向量化**
  `T.Parallel(head_tile)`，行方向交给 `T.cumsum`；
- 标量循环完全消除（无逐元素 Python/TIR 标量语句）；
- 掩码用 `T.if_then_else` 生成 0 填充，不产生分支发散。

**向量化结论**：向量化轴 = **head（最内层、连续）**；行方向不再由软件循环承担。

#### 1.6.3 向量化轴与数据布局决策

| 候选布局 | 整除性 | 尾 lane 浪费 | repack 代价 | UB 影响 | 结论 |
|---|---|---|---|---|---|
| **L0 `[C, head]` 行主序，按 head 向量化（基线输入布局）** | `head` ∈ {16,32,64}，32B 对齐 | 无（head_tile 取 64 或 head） | **零**（GM 布局即核内布局） | C=64、head=64、fp32 → 16 KiB | **采纳** |
| L1 `[head, C]` 转置后按 C 向量化 | C=64 可整除 | 无 | 需核内 `T.transpose`，多一趟 UB 往返 | 同上 | 弃：多一次重排，收益为零（cumsum 沿 C 本就是硬件指令） |
| L2 head 与 C 平铺为 1D 后分段 | 需 32B 对齐拼接 | 尾部可能浪费 | 零 | 相同 | 弃：1D 分段无法表达"逐 chunk 重置" |

**布局决策结论**：采用 **L0**——直接沿用 GM 的 `[tokens, head]` 行主序，
向量化轴取 head，`dim=0` 的累积和交给 `T.cumsum`；不做任何核内重排。

## 2. 编程模式选型

### 2.1 模式结论

**Developer**（纯 Vector，无 Cube / `T.gemm`）。

### 2.2 选型理由

本函数只有逐元素搬移、类型转换与前缀和，没有任何矩阵乘；Developer 模式由编译器自动管理
Vector 上下文，避免手写 `T.Scope("Vector")` 与 UB 生命周期。仓库内纯 Vector 内核
（`batch_norm_bwd` 等）同为 Developer。

### 2.3 模式影响

- 若上层算子整体走 Expert（GDN 主干为 Cube+Vector），**本内核可作为独立 launch 保持 Developer**；
- 不在本内核内混用 Cube，故不触碰 Expert 的双段约束。

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 公式 | 说明 |
|---|---|---|
| S1 | `x = g[base+i, h]` | 取门值 |
| S2 | `xf = cast_fp32(x)` | 累积必须在 fp32 |
| S3 | `y[i] = Σ_{j≤i} xf[j]` | chunk 内前缀和 |
| S4 | `out[base+i, h] = y[i]` | 写回 fp32 |

### 3.2 TileLang API 映射

| 步骤 | API | 备注 |
|---|---|---|
| 取数 | `T.copy(g[...], g_ub)` | 支持切片与越界掩码 |
| 转型 | `T.vcast(g_ub, f32_ub)` | bf16/fp16 → fp32 |
| 前缀和 | `T.cumsum(f32_ub, out_ub, dim=0)` | `hivm.VCumsumOp`；支持 fp32 |
| 写回 | `T.copy(out_ub, out[...])` | fp32 → fp32 |

### 3.3 计算伪代码

```python
with T.Kernel(num_chunks, num_head_slabs, is_npu=True) as (cid, slab):
    g_ub   = T.alloc_ub((chunk, head_tile), in_dtype)
    f32_ub = T.alloc_ub((chunk, head_tile), "float32")
    out_ub = T.alloc_ub((chunk, head_tile), "float32")
    T.copy(g[0, base:base+chunk, slab*head_tile:(slab+1)*head_tile], g_ub)
    T.vcast(g_ub, f32_ub)
    T.cumsum(f32_ub, out_ub, dim=0)
    T.copy(out_ub, out[0, base:base+chunk, slab*head_tile:(slab+1)*head_tile])
```

### 3.4 API 可行性确认

- `T.cumsum` 支持 fp32/fp16/int32/int64，**不支持 bf16** → 先 `vcast`（§0.6）；
- `T.vcast` 在 Vector 上下文内合法（Developer 模式自动管理）；
- 文档：`docs/Tilelang.language/排序操作/T.vcumsum.md`（§2.2.1 支持表、§2.4 示例）。

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

- 仅纯 Vector，无 Cube 环路 → 不触发 CG-2026-0026 类问题；
- UB 预算：C=64、head_tile=64、fp32 → 16 KiB × 2（f32_ub/out_ub）+ 8 KiB（g_ub）≈ 40 KiB < 192 KiB；
- `head > 64` 时按 slab 切分，`head_tile=64` 固定。

### 3.5.2 参考实现差异说明

源实现的"网格覆盖最大 chunk 数、多余块立即退出"依赖 GPU 的 block 语义；NPU 侧保留同样语义
（`if cid < num_valid_chunks` 立即返回），保证 launch 范围与 offsets 取值无关。

### 3.5.3 本项目同类实现参考

`tileops/kernels/norm/batch_norm_bwd/*`（Developer + `T.alloc_ub`）、
`tileops/kernels/reduction/isnan` 系（纯 Vector 逐元素）。

## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | 形状 | dtype | 对齐 |
|---|---|---|---|
| `g` | `[1, total_tokens, head]` | fp16 / bf16 | head ≥ 16 ⇒ 内层 ≥ 32B |

### 4.2 输出张量

| 名称 | 形状 | dtype |
|---|---|---|
| `out` | `[1, total_tokens, head]` | **float32** |

### 4.3 中间缓冲区

`g_ub`（in_dtype，`C × head_tile`）、`f32_ub`（fp32）、`out_ub`（fp32）。

### 4.4 内存搬运路径

`GM → UB（MTE2）→ vcast/cumsum（Vector）→ UB → GM（MTE3）`，单次读、单次写。

### 4.5 UB 内存预算

C=64、head_tile=64：`8 KiB（g_ub, fp16）+ 16 KiB + 16 KiB = 40 KiB`；bf16 同量级。

## 5. 核切分与任务映射（Tiling 策略）

### 5.1 ① 逻辑核数计算

- 任务空间：`num_chunks = ceildiv(total_tokens, chunk)`，`num_head_slabs = ceildiv(head, 64)`；
- 逻辑任务数 `T = num_chunks × num_head_slabs`；
- **逻辑核数** `K = min(T, physical_cores)`。

### 5.2 ② 物理核数依据

`NPUUtils.get().get_aicore_num()` 实查（Ascend910B3 单卡；Vector 核数由运行时给出，**不硬编码**、
不在设计期写死常量）。

### 5.3 ③ 规模判定与分核方案

- 若 `T ≤ K`：一核一块，**无需适配**（规模判定依据：每核任务数 = 1，无轮转）；
- 若 `T > K`：`block_id = task_id % K` 轮转，每核任务数为 **整数倍** 关系（`ceildiv(T, K)`）；
- 单核内部三段流水为 **核内串行**（取数 → cumsum → 写回），无跨核依赖、无跨核同步；
- 尾块（`total_tokens % chunk != 0`）不另起核，靠行掩码在最后一块内处理。

## 6. 精度与数值行为

- 累积在 **fp32** 中进行，输出 fp32；
- 与 fp64 参考对比：`max_abs_err ≤ 1e-5`（C=64、|g| ≤ 1）；
- 输入 fp16/bf16 的门值本身有量化误差，单调累加不额外放大（无相减抵消）。

## 7. 边界与异常处理

- `total_tokens` 非 chunk 整数倍：最后一块有效行 = `end - base`，其余行填 0（不改变前缀和）；
- `head` 非 64 倍数：最后一个 head slab 带列掩码；
- `num_chunks == 0`（空输入）：kernel 直接返回，不写越界内存。

## 8. 验证方案

### 8.1 Golden 函数说明

`gdn_cumsum_golden(g, cu_seqlens, chunk)`：纯 torch（CPU）实现，**fp64** 累积，
逐序列按 chunk 重置，与内核实现无共享代码：

```
for seq: for chunk: acc = 0
  for i in range(valid_rows): acc += g64[base+i]; out[base+i] = acc
  for i in range(valid_rows, chunk): out[base+i] = 0
```

### 8.2 L0 门禁测试计划

| 层 | 用例 | 判据 |
|---|---|---|
| **L0** | `S=64, head=16, fp16`（单 chunk） | vs fp64 `max_abs_err ≤ 1e-5` |
| **L0b** | `S=64, head=16, bf16`（vcast 路径） | `≤ 1e-2`（bf16 输入量化） |
| L1 | `S=192, head=64, fp16` | 每 chunk 重置正确 |
| L1b | `S=200, head=64`（尾块 8 行） | 尾块与越界行正确 |
| L2 | `S ∈ {4096,12288,32768}, head=16`，fp16/bf16 | 与源实现数值对照 |

### 8.3 通过标准

L0 + L0b 全绿 → 进入 L1；L1/L1b 全绿 → 进入 L2；全部通过 → Stage 3 出口 `[PRECISION_PASS]`。

## 9. 版本戳与参考

- 工具链：CANN **9.0.0**（`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0`）、
  bishengir-compile **1.1.0**、hivmc **0.2.0**、tilelang **0.1.2**（dev root `/home/developer/tilelang-mlir-ascend`）、
  torch 2.7.1+cpu、torch_npu 2.7.1.post4、Ascend910B3 × 1。
- 源实现：`/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py:22`
- 抽取产物：`tileops/kernels/linear_attention/gdn_prefill/_gdn_prefill_kernels.py`
- 文档：`docs/Tilelang.language/排序操作/T.vcumsum.md`（§2.2.1 支持表）
