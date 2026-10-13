# _argreduce_kernel（ArgmaxFwdOp）算子设计文档

> 迁移任务（harness）：GPU TileOPs `ArgmaxFwdOp` → TileLang-NPUIR。
> 源算子：`/home/tilelang/l00970450/TileOPs/tileops/kernels/reduction/argreduce.py`（`ArgreduceKernel` + `_argreduce_kernel` 工厂）。
> 本设计基于：源码全文解读（Phase M0）、算法族调研（Phase R）、硬件耦合性判定与 NPU 重设计（Phase M1）、算法级优化设计（Phase 2）、**设计期探针实测**（D-3，`history_version/design_probe_argmax.py` + `design_probe_argmax_dbg.py`）与**等价性机器验证**（D-1，`verify_equiv.py`）。
> 工具链戳：tilelang 0.1.2（dev root build 2026-09-24）+ CANN 8.5.0 + Ascend910B2C + torch 2.x（npu）。
> **v1 修订（2026-09-24，revision_index=1）**：按 Stage 2 REVIEW.md（不通过：3 阻塞 + 7 建议）定点修订，修订清单与防复发说明见下方「修订说明」；**算法选型（C1/C2 双路径）、E1–E4 数学等价优化、R1–R4 重设计、golden 独立性、Developer 模式等设计结论全部保留**（均为低成本定点修正，不推翻迁移决策）。上一版备份：`history_version/design_v0.md`。

## 修订说明（v1 相对 v0 的关键调整与防复发说明）

| # | 调整 | 涉及章节 | 为什么不会再犯同一错误（防复发机制） |
|---|------|---------|-----------------------------------|
| 阻塞 1 | 3d-non-last-axis 负载 M 修正 **512 → 524288**（manifest `[4,128,4096] dim:0` → `N=x.shape[0]=4、M=128×4096`；前次日志 `shape=(524288,256) bm=384 ≈194–205µs` 三源互证）；§5.2 新增**发射项驱动的扩展 block_m 阶梯**（基础阶梯 block 数 > 48×2 时启用 {16..2048}），3d 选定 bm=2048（256 block、48 核×6 任务、发射 ≈18.7µs）；roofline/L0-6/§5.5/§9.2 全部按真值重算（bm=8 反事实 ≈4.8ms 写入表内作 sanity 锚点） | §1.4/§1.6.0/§5.2/§5.5/§8.2/§9.2-R10/§6.3-⑤ | 负载表 M/N 一律从 manifest + Op 层 `_prepare_input` 推导并**附推导算式**（不再手抄数字）；发射项反事实（最差 bm）随表记录，规模判定与 bm 取值不自洽时立即可见 |
| 阻塞 2 | `compute_tile_n` 调用修正为**字节口径 + slab 折算**（对照 `_primitives.py` 源码逐参数核对：budget 单位字节、num_buffers 为同形大缓冲折算数；v0 把 budget 折成元素数又传 elem_bytes/num_buffers 构成二次折算）；新增 wrapper 侧**整除优先 + 0.9 预算裕度后置校正规则**（不依赖原语除数接受判据），两 dtype 推导链显式写入（fp16：cap 6400→5120；bf16：cap 4608+尾→4096），冻结值 5120/4096 不变，「公式→表值→测试 shape→roofline」四处一致 | §5.2/§1.6.0/§8.2/§9.2-R8 | 凡引用仓内原语，设计期必须打开其源码核对签名与**单位口径**（本错误即未读原语源码、凭名字猜语义所致）；推导链（原语返回值→后置校正→表值）显式落盘供 Stage 2 复算 |
| 阻塞 3 | 删除 §1.6.3-#3 的「`T.transpose` dtype 限 fp16」负向论断（与 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 全 dtype √ 表直接矛盾）；弃选理由改为单一量化依据：主选向量化轴 N 即 I/O 原生连续轴、**无 repack 需求**，任何核内转置纯开销零收益；代价引 pattern-library `layout.md` **PL-1.1-transpose-chain** 实测（4 次融合转置 ~5.4µs，verified）与 +4~8B/elem UB 缓冲开销 | §1.6.0 信息源/§1.6.3-#3/结论段 | 负向论断（不支持/代价高）必须**亲自打开所引文档核对条款原文** + 先查 pattern-library 已有实测（v0 信息源漏了 layout.md，本版已补）；弃选理由须量化且单一化，不堆叠未核实前提 |
| S1 | §0.3 步骤表补源码 L95 `T.fill(out_idx, 0)`（无匹配回退值）；§0.1 NaN 段补「全 NaN 行源实现输出 0」；§0.5 增「M 尾块守卫缺失（源缺陷）→ 重新设计」行，Stage 3 不得照抄源码无守卫 copy 形态 | §0.1/§0.3/§0.5 | 源码解读以「每条计算语句归入步骤表」为完成标准（本次重读源码逐行核对补齐） |
| S2 | §5.2 工厂期新增 `assert N <= 2**24`（E1/E3 fp32 索引精确域前提，超限 raise）；§8.2 L2 增断言行为记录用例 | §5.2/§8.2/§9.2-R11 | 等价性论证中的隐式前提显式化为工厂期断言 |
| S3 | §1.6.1 机器验证表 E3 行数 40 → **33**（与 verify_equiv.py 实际输出对账：3 dtype ×〔8 shape×tile 组合 + 2 角点 + 1 跨 tile tie〕；总数 114 与 0 失败不变） | §1.6.1 | 表内计数与脚本输出逐行可对账（Stage 2 重跑口径） |
| S4 | §4.5/§9.1-C-7 两个膨胀数据点绑定 buffer 清单：**160KB = 含未用 ext_brc 首版（10B/elem）→1.60×；128KB = 净版（8B/elem）→2.0×**（同 256KB 报错值、不同清单，即 C-3 成因）；预算纪律仍按 ×2 保守 | §4.5/§9.1-C-7 | 数据点必须绑定产生它的配置清单，防 Stage 3 误按 128KB×1.60 规划 |
| S5 | C6（N-split）Stage 4 裁决三件套补齐：**§1.6.4 新增备选完整结构**（partial/merge 两 kernel 伪代码、workspace 形状 (M,nchunk)、UB 预算、分核三要素、tie-break 与 E1 同构）+ **判定阈值**（C6 端到端 < 0.8× 主选则该 shape 类翻转，按 shape 分派）+ **回写路径**；补前次 round9 终版实测锚点（partial ≈15.7µs + merge ≈1.7µs，golden PASS ≈0.54× 主选）；「单 kernel 范围」论断改为可核对表述（工厂契约条款 + 前次已证明 wrapper 组合可行） | §1.6.3-#4/§1.6.4/§1.6.0-R4/§6.3-③/§9.2-R3 | 实验裁决候选必须同时给出结构、阈值、回写路径三件套；引用前次工件时正例（round9 终版）与反例（round8 首版）都要取 |
| S6 | §3.3 删除不可执行的 `B={...}[dtype][bm>=2]` 单行下标写法，改**两段式伪代码**（B_multi 解基础阶梯 → 发射项触发扩展阶梯 → B_bm1 回退 → 转 tiled，与 §5.2 同序，消除 B↔bm 循环依赖）；`test_reduce.py` 行号 L43→**L38**、`migration-analysis.md §5.3` 行 11→**行 13**（online 行） | §3.3/§1.6.2-#3/§0.5 末注 | 伪代码须可执行序；引用行号以亲自打开文件核对为准 |
| S7 | §1.6.0-R3 C1 行缓冲峰值 8–14 → **8–16B/elem**（补 fp32 bm=1 = 16；与 §4.5/§5.2 同口径：fp16 8/10、bf16 10/14、fp32 12/16） | §1.6.0-R3 | 跨表数字一律与 §4.5 B/elem 表对账 |

> §0 其余结论（语义/算法/优化手段/耦合性判定/R1–R4）经本次修订重读源码复核一致，未改动；§1.6.0 调研结论（C1+C2 选定、R1–R4 四问）与 §1.6.1 E1–E4（ALL EQUIV_PASS，114 行）不变。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对 2D 输入 `x ∈ (M, N)` 沿最后一维（N，规约维）求极值位置索引：

$$\mathrm{argmax}(x)[i] = \min\{\, j \in [0, N) \;:\; x[i,j] = \max_{k<N} x[i,k] \,\},\qquad \mathrm{argmin}(x)[i] = \min\{\, j \;:\; x[i,j] = \min_{k} x[i,k] \,\}$$

即 **first-occurrence（首个极值位置）语义**——tie 时返回最小索引，与 `torch.argmax/torch.argmin` 的首现保证一致（源码以「串行扫描 + 首个匹配 break」实现该语义；harness 测试 `test_argmax.py` 用 `torch.equal` **精确比对**索引）。

**规约语义**：规约轴 = N（Op 层已把任意 rank 输入 reshape/movedim 到 (M, N)，`dim=None` 全张量规约被 flatten 成 (1, numel)）；无 keepdim 语义进入 kernel（Op 层后处理）。max/min 为**精确选择运算**——无累加顺序问题、无舍入（这是 argmax 与 sum 类规约的本质差异，见 §1.6.1-E4）。

**dtype 语义**：输入 fp16/bf16/fp32；GPU 源码把数据先 cast 到 **fp32** 再做 reduce 与匹配（对 argmax 而言该 cast 数值上是非必要的，见 §0.5 处置）；输出恒 **int64**。

**边界语义**（从源码与 torch 语义共同确定）：
- **±inf**：正常参与比较（finite < +inf < ... ）；全 `-inf` 行 argmax=0（首个匹配）。
- **NaN**：torch 语义为 NaN 胜出（返回首个 NaN 索引）；**源 GPU 实现即不保证 torch 语义**（`T.reduce_max` 在 CUDA 上以 fmax 语义忽略 NaN，极值为非 NaN 最大值，扫描匹配的是该值）；**全 NaN 行**：fmax 全部忽略 → `row_extreme` 保持 `T.fill(-inf)` 初始化值 −inf → 扫描永不匹配 → 源实现输出 **0**（源码 L95 `T.fill(out_idx, T.cast(0,"int64"))` 的无匹配回退值，v1 补记）；NPU 侧实测（探针）：硬件 reduce **传播 NaN** → 匹配永不成立 → 输出哨兵 `2^30`（见 §9.2-R1）——全 NaN 行上 torch（首 NaN 索引）/ 源 GPU（0）/ NPU（2^30）三方各异。harness 测试全部使用 `randn` 输入（无 NaN），该边界不进入 L0 门禁。
- **±0.0**：kernel 按 IEEE 相等（`-0.0 == +0.0`）匹配，与 **torch CPU golden 一致**；torch-NPU 设备实现把 ±0 视为可区分（argmax 偏好 +0.0）——混合符号零且行极值恰为 0 的行上两者分歧（探针实测记录，§9.2-R2）。`randn` 输入下该位形概率为 0。
- **空维**：N=0 由 kernel 层前置拒绝（源码 `ArgreduceKernel` 构造即 raise）；M=0（非规约维含 0）→ 输出 (0,)，源实现未覆盖（grid=0），NPU 侧列为风险（§9.2-R6）。
- **非整除**：源 Op 层 host 侧 `F.pad` 到 `N_padded = align_up(N, 256)`，pad 值为**单位元**（argmax → `-inf`、argmin → `+inf`），保证 pad 列永不胜出——这是「pad 不改变结果」的语义依据（§0.6-R3 的基础）。

**host 侧语义**：reshape/transpose/pad/keepdim 均为 **Op 层实现手段**而非算子语义（`tileops/ops/reduction/reduce.py` 注释明言 kernel 转换完成后应消除 host pad——"until their kernels are converted"）；golden 不得复刻这些实现细节。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.argmax`，first-occurrence + int64 + IEEE 相等匹配 + torch NaN 语义为文档化边界）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(M, N_padded)`，`N_padded = align_up(N, 256)` | float16 / bfloat16 / float32 | 2D 输入；M=非规约维乘积，N=原始规约维长度（工厂参数携带**原始 N**）；GPU 契约下 Op 层已 pad（argmax pad `-inf`） |
| `out` | `(M,)` | int64 | 每行一个首现极值索引（迁移必需字段：输出 shape `(M,)`，无转置布局） |

工厂契约（harness 固定，接口约束）：`_argreduce_kernel(M, N, op_kind, dtype)` 返回 `_func(block_m, threads)→kernel(x)`；语义参数 `M/N/op_kind/dtype` **不变**，后端参数 `threads` 依 K9 去除（NPU 无 CUDA 线程概念）；`op_kind ∈ {argmax, argmin}`（本迁移单元的测试范围是 **ArgmaxFwdOp/argmax**；argmin 为独立迁移单元，但工厂接口保留双 kind 支持）。

**NPU 侧输入契约设计决策（§0.6-R3）**：主选把 NPU 侧 `ArgmaxFwdOp._kernel_handles_padding` 置 `True`（一行属性变更，属仓库既定演化方向），kernel 直接接收**原始 `(M, N)`**——消除 host `F.pad`（该 pad 对 dim=0 类负载造成最高 64× 流量膨胀，见 §5.5 实算）；回退方案（若 harness 冻结 Op 层）：kernel 按 `(M, N_padded)` 声明、只读前 N 列（同一 kernel 代码，工厂换声明宽度）。两案输出完全一致（pad 值为单位元）。

### 0.3 实现算法解读（怎么算）

GPU 源码核心（`_argreduce_kernel`，两步法）：

**计算步骤分解**（覆盖源码全部计算语句）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.copy(x[pid*bm, 0], shared_buf)`：block 行块 GM→SMEM 整块载入 | x (M,N_padded) | shared_buf (bm,N_padded) | 数据载入 |
| 2 | `T.Parallel(bm,N_padded)` 循环 cast fp32 | shared_buf | x_f32 fragment | 精度提升（对 argmax 非必要，§0.5） |
| 3a | argmax 分支：`T.fill(-inf)` + `T.reduce_max(x_f32, row_extreme, dim=1, clear=False)` | x_f32 | row_extreme (bm,) | 行最大值 |
| 3b | argmin 分支：`T.Parallel` 循环取负 → `reduce_max` → `T.Parallel` 循环取负还原 | x_f32 | neg_x fragment + row_extreme | 行最小值（-max(-x) 变换） |
| 3c | `T.fill(out_idx, T.cast(0, "int64"))`：扫描回退值初始化（源码 L95） | — | out_idx (bm,) int64 | **无匹配行的回退语义**（如全 NaN 行：row_extreme 保持 −inf、扫描永不匹配 → 输出 0，§0.1；v1 补记） |
| 4 | 串行扫描：`for i in T.Parallel(bm): for j in T.Serial(N): if x_f32[i,j]==row_extreme[i]: out_idx[i]=j; T.loop_break()` | x_f32, row_extreme | out_idx (bm,) int64 | **首现索引**（early-exit 平均 N/2 次标量比较；扫描上界为原始 N 而非 N_padded） |
| 5 | `T.copy(out_idx, out[pid*bm])` | out_idx | GM out | 输出搬运 |

**数据流**（源硬件视角）：`GM[x] →T.copy→ SMEM[shared_buf] →T.Parallel cast→ register/fragment[x_f32] →reduce/scan→ fragment[row_extreme,out_idx] →T.copy→ GM[out]`；argmin 分支多一个 neg_x fragment 往返。

**循环与并行结构**：`T.Kernel(ceildiv(M, block_m), threads=threads)` 一维 grid × CUDA 线程两级并行；block_m ∈ {1,2,4,8} 由 48KB SMEM 预算启发式选择（`SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)`），N<256 重 pad 时附加 `block_m × N_padded ≤ 2×threads` 布局约束；无流水（无 T.Pipelined）。

**host 侧逻辑**：Op 层 validate →（dim 归一/movedim/contiguous）reshape (M,N) → `F.pad` 到 N_padded（单位元填充）→ 调 kernel → 输出 reshape 回原始形状（keepdim 处理）；`functools.lru_cache` 按 (M,N,op_kind,dtype) 缓存编译产物。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target）；无特殊 pass_configs。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（256 元素对齐整块 T.copy） | 合并访存、一次载入多次使用 | 行块驻留 SMEM，reduce 与 scan 共享 | shared memory + 256 对齐的 copy 指令约束 | 硬件强相关 |
| 2 | 两步法（并行 reduce 找极值 + 串行扫首匹配） | 极值用并行归约（快），首现索引用标量扫描（语义必需） | reduce_max 树形归约 + early-exit 标量扫描 | CUDA 线程级并行的 reduce 原语 | 算法层可移植；scan 的标量形态硬件相关 |
| 3 | fp32 累加域 cast | 数值稳健（对 sum 类规约必要） | 全数据 cast fp32 后再 reduce | — | 对 argmax 非必要（max 精确选择，§1.6.1-E4） |
| 4 | argmin 的 `-max(-x)` 变换 | 复用 argmax 通路（一次实现双算子） | 取负 → reduce_max → 取负还原 | — | 纯数学恒等，可移植 |
| 5 | host pad + 单位元填充（-inf/+inf） | 满足 256 对齐的非整除处理 | Op 层 F.pad，pad 列不改变结果 | GPU copy 的 256 对齐需求 | 对齐约束硬件相关；单位元技巧可移植 |
| 6 | block_m SMEM 预算启发式 | 行块尽量大以摊薄 block 开销 | 48KB 预算 + [1,2,4,8] 阶梯 | SMEM 容量 | 预算值硬件相关 |
| 7 | lru_cache 工厂缓存 | 消重复编译 | Python 层 (M,N,kind,dtype) 键 | — | 可移植 |
| 8 | 重 pad 布局约束（`bm×N_padded ≤ 2×threads`） | 规避 TileLang GPU copy 布局推断限制 | 限制 block_m | GPU copy-layout 推断 | 硬件强相关（NPU 不适用） |

> 识别完整性：以上覆盖源码全部优化性结构（步骤分解表已把每条计算语句归入步骤 1–5，含 3a–3c 分支——v1 补齐 L95 `T.fill(out_idx,0)`）；无 warp shuffle / mma / swizzle / 异步流水等 GPU 深度优化（源码为朴素两步法）。**源实现缺陷（本次修正，Stage 3 不得照抄）**：L68/L103 的 `T.copy` 无 M 尾块守卫（`M % block_m ≠ 0` 时越界读/写；harness 用例 (129,512) 且 block_m=8 → 129%8=1 触发位形）——见 §0.5 对应行与 §5.4 三件套修正。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| first-occurrence argmax/argmin 语义（含 int64 输出） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = torch.argmax |
| 两步法结构（极值归约 → 首现索引） | 算法 | 无（纯算法层） | — | **保留** | 结构保留；两步的**实现形态**分别等价替换/重设计（下两行） |
| 步骤 3 极值归约 `T.reduce_max` | 算法 | CUDA 线程归约 | 有：Vector `T.reduce_max/min`（dim=1） | **等价替换** | `docs/Tilelang.language/规约操作/T.reduce_max.md`；**探针实测约束：reduce dst dtype 必须与 src 相同**（fp16 src→fp32 dst 静默产出错误值，`design_probe_argmax_dbg.py` 变体 B；文档 §2.4 的 dtype≠accum 示例在本工具链不成立）；bf16 不被 reduce 支持（`T.reduce.md` §2.2.1）→ 预 cast fp32 |
| 步骤 4 串行扫描 + loop_break | 算法 | CUDA 标量线程 early-exit | 无等价（NPU Vector 无 per-lane 标量 early-exit） | **重新设计** | → §0.6-R1：向量化「掩码索引取 min」：`cand[j] = j if x[j]==m else BIG; first = reduce_min(cand)`（E1，机器验证 EQUIV_PASS） |
| 步骤 2 fp32 cast（fp16/bf16 输入） | 算法/精度 | — | — | **舍弃（fp16）/ 保留改造（bf16）** | fp16：max/min 为精确选择运算，fp16 域 reduce 与匹配和 fp32 域**索引恒等**（E4，EQUIV_PASS），省 1 个全宽 pass + 2B/elem；bf16：reduce 不支持 + bf16→fp32 精确，保留 cast（`T.vcast` rint） |
| 步骤 3b `-max(-x)` argmin 变换 | 算法 | — | 有更优：直接 `T.reduce_min` | **等价替换** | 直接 reduce_min + 同一首现索引机制；索引恒等（E2，EQUIV_PASS）；省 2 个全宽 pass（取负+还原） |
| SMEM tiling / 256 对齐 copy | 优化 | shared memory | 有：UB（`T.alloc_shared`，192KB） | **等价替换** | GM→UB staging + fragment 计算（migration-analysis.md §5.3 行 1）；对齐需求降为 32B 尾轴（logsumexp N=300 直证） |
| host pad + 单位元（N 非整除处理） | 优化 | GPU 256 对齐 | NPU 可免（32B 对齐） | **重新设计** | → §0.6-R3：kernel 接收原始 N，尾 tile 静态特化；主选翻转 `_kernel_handles_padding=True` 消除 host pad |
| block×thread 两级并行 / threads 参数 | 优化 | CUDA 线程 | 无（本项目一维 Kernel） | **重新设计** | → §0.6-R2：一维 persistent grid（48 Vector 核，实查）+ `T.serial` 静态边界任务循环；threads 参数 K9 去除 |
| block_m 48KB SMEM 启发式 + 布局约束（#6/#8） | 优化 | SMEM 容量 / GPU copy 布局推断 | 换 UB 预算模型 | **重新设计** | → §0.6-R4：探针定标的 UB 预算公式（§5.2 表；auto-multi-buffer 膨胀 1.6–2.0× 实测 + 竞态安全手工预算 64KB/block 实测验证） |
| M 尾块守卫缺失（源实现 L68/L103 `T.copy` 无 `M % block_m` 边界检查 → 越界读/写风险，(129,512) bm=8 即触发） | 正确性（**源缺陷**，v1 补记） | GM 定长 block copy 无边界检查 | 有：`T.min` 标量 + src/dst 双显式 slice | **重新设计** | §5.4 三件套（real_m = T.min(bm, M−start) + 双 slice + 垃圾行不写出，VP-2026-0044 / logsumexp 先例）；**Stage 3 不得照抄源码的无守卫 copy 形态** |
| lru_cache 工厂缓存（#7） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> 每项处置依据：migration-analysis.md §5.3 映射表（行 1 SMEM→UB、行 7 block×thread→一维重设计、**行 13** online/算法层保留——v1 更正：行 11 为 grid-stride，online 行在 L163）+ 本仓 `docs/Tilelang.language/` API 文档 + 探针实测；无「舍弃后性能意图丢失」项——#3 的精度意图由 E4 的等价性论证承接（max 精确选择无需 fp32 域），#8 的布局规避意图随 GPU copy 约束消失而不需要。

### 0.6 NPU 算法重设计

**重设计项 R1：首现索引的向量化（源：串行扫描 + loop_break）**

- **源方案**：`T.Serial(N)` 标量扫描逐元素比较、命中即 `T.loop_break()`（平均 N/2 次标量操作；依赖 CUDA 线程标量执行模型；性能意图：early-exit 减少平均比较次数）。
- **NPU 新算法**：掩码索引取最小——
  $$\mathrm{first}[i] = \min_{j<N}\;\big(\,j \cdot \mathbb{1}[x[i,j]=m[i]] + \mathrm{BIG}\cdot \mathbb{1}[x[i,j]\neq m[i]]\,\big),\qquad \mathrm{BIG}=2^{30}$$
  实现：`T.reduce_max/min`（极值 m）→ 融合 `T.Parallel` 循环单遍生成 `cand[i,j] = if_then_else(x[i,j]==m[i], j, BIG)`（`T.if_then_else` 元素级条件有已验证测试先例 `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py`，探针 P1 端到端复证）→ `T.reduce_min(cand, first, dim=1)` → int64 出栈。early-exit 意图由「向量单元整行并行」承接（O(N) 向量 pass ≪ O(N/2) 标量操作）。
- **语义保持论证**：m 由实际数据归约得出、至少一个 j 命中（有限输入/±inf）；非命中候选为 BIG=2^30 > 任何合法 j（j < N ≤ 2^24，fp32 精确表示；BIG 为 2 的幂 fp32 精确）；min 取到**最小命中 j = 首现索引**，与源扫描逐元素等价（机器验证 E1：114 行 EQUIV_PASS，含 tie/全 -inf/±inf/亚正规/混合零/N=1..102400）。边界：NaN 行不命中（输出哨兵 2^30，§9.2-R1 文档化）。

**重设计项 R2：并行结构（源：block×thread 两级 + threads 参数）**

- **源方案**：一维 grid × CUDA threads；threads 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector 一维 persistent kernel——`T.Kernel(num_kernels, is_npu=True)`，`num_kernels = min(ceil(M/block_m), 48)`（48 = 24 AICore × 2，纯 Vector 翻倍，**实查**见 §5.5）；核内 `T.serial(静态上界)` 任务循环（grid-stride 映射）+ 动态 guard `if start < M`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例先例）；元素级并行全部交给 Vector 向量化（`T.Parallel` 融合循环 + reduce 原语）。**计算缓冲全部用 fragment**（shared 仅作 GM staging）——探针 P3 发现 shared 多消费者缓冲在 auto-multi-buffer 流水下当双缓冲预算超限时产生**非确定性数据竞争**（详见 §9.2-R4），fragment 计算形态在 1024 block 多波次下 3/3 稳定通过。
- **语义保持论证**：行块间无依赖（per-row 独立规约），任务循环只改变调度不改变计算；尾块 `real_m = T.min(bm, M - start)` + src/dst 双显式 slice（logsumexp/VP-2026-0044 三件套先例），垃圾行不写出。

**重设计项 R3：padding 策略（源：host F.pad 到 256 对齐 + 单位元）**

- **源方案**：Op 层 `F.pad(x, (0, N_padded-N), value=单位元)`，满足 GPU SMEM copy 256 元素对齐；kernel 声明 (M, N_padded) 但只扫描前 N 列。
- **NPU 新算法**：kernel 接收**原始 (M, N)**（主选：`ArgmaxFwdOp._kernel_handles_padding = True` 一行翻转，`tileops/ops/reduction/reduce.py` 模块注释明言这是仓库既定演化方向 "until their kernels are converted"）；非整除 tile 由**静态尾 tile 特化**处理（`tail = N - num_full×tile_n` 为工厂期常量，独立静态宽度代码路径，探针 P2 验证含 188 列尾 tile）；回退方案：kernel 声明 (M, N_padded)、T.copy 只读前 N 列。
- **语义保持论证**：pad 值为单位元（argmax:-inf / argmin:+inf），pad 列在原实现中本就不可能胜出且不被扫描——**消除 padding 不可能改变任何输出**（对任意输入 x：argmax(x) 与 argmax([x, -inf…]) 索引相同，含全 -inf 行——首现在实列）；回退方案同理由。收益：dim=0 类负载（3d-non-last-axis：`[4,128,4096] dim:0` → **M=524288、N=4**（`_prepare_input` 单维路径：N=x.shape[dim]=4、M=prod(其余)=128×4096）→ pad 到 N_padded=256：读侧 N 维 **64× 膨胀**（4.19MB → 268MB）；含写出的**总流量口径 ~32×**（268+4.19MB vs 8.4MB，§5.5/§1.6.0 实算——两个口径须区分））host pad 全消（§5.5 实算：前次任务该负载 pad 契约 ~194–205µs 主因即此）。

**重设计项 R4：block_m/tile_n 选择（源：48KB SMEM 启发式 + GPU 布局约束）**

- **源方案**：`SHARED_MEMORY_BUDGET_BYTES(48KB) // (N_padded × elem)` + [1,2,4,8] 阶梯 + 重 pad 布局约束。
- **NPU 新算法**：UB 预算公式（探针定标）：`block_m = max{p ∈ {1,2,4,8} : p·N·B(dtype) ≤ 64KB}`（B 为 fragment 形态手工字节数/元素，见 §5.2 表；64KB 为实测竞态安全手工预算——×2 双缓冲 ≈ 128KB ≪ 188KB 可用）；`N·B > 64KB 且 p=1 仍超` 时切 tiled 路径（强制 bm=1，§9.2-R3 编译器陷阱），`tile_n` 由 `_primitives.compute_tile_n`（仓内既有 NPU 原语，除数优先）按同预算选取。
- **语义保持论证**：纯配置选择，不影响语义；预算边界由探针编译+运行双向定标（通过点：fp16 bm=2@4096=64KB、fp32 bm=1@4096、bf16 bm=1@4096；失败点：fp16 bm=4@4096 编译溢出 256KB、fp32/bf16 bm=2@4096 竞态）。

### 0.7 标杆实现

- **源算子**：`/home/tilelang/l00970450/TileOPs/tileops/kernels/reduction/argreduce.py`（本设计 §0.1–0.4 解读对象）；Op 层 reshape/pad 策略：`/home/tilelang/l00970450/TileOPs/tileops/ops/reduction/argreduce.py` + `reduce.py`。
- **NPU 侧 harness 工件**：提取版 GPU 参考 `examples/TileOPs/tileops/kernels/reduction/argmax/_argmax_fwd_kernels.py`；已移植 wrapper（Part B）`examples/TileOPs/tileops/kernels/reduction/argmax/argmax.py`（`npub::argreduce_fwd` custom_op + `ArgreduceKernel`，K5–K9 已适配，`threads` 已去除）；Op 层 `examples/TileOPs/tileops/ops/reduction/argmax.py`；规格 `examples/TileOPs/tileops/manifest/reduction.yaml` ArgmaxFwdOp 键。
- **golden 基准**：`torch.argmax`（§8.1；harness 测试 `examples/TileOPs/tests/ops/test_argmax.py` 以 `x.argmax(dim=...)` 为 ref、`torch.equal` 精确比对）。
- **同族已验证参考**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/` 与 `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（row-reduction 迁移基准形态，VP-2026-0044）。
- **前次任务痕迹（重要背景）**：本算子目录存在前次 argmax 任务（Stage 3/4 已清理源码、残留 `repro/__pycache__` 与 `perf_opt/logs`）的两条工具链级陷阱与性能数据，已并入本设计（§9.2 与 §1.6.0 校准）：`TRAP-multitile-bm-gt1-segfault`（bm>1+多 tile → 编译器 SIGSEGV；CG-2026-0014 登记缺失，见 §9.4）、`CONST-vcast-f32-to-f32-corrupt`（vcast f32→f32 非恒等、破坏数据）；性能校准：lm-head 单核 tiled ~32.5µs、hidden-state fp16 bm=2 ~47µs / bf16 bm=1 ~55µs、3d（pad 契约）~200µs。

## 1. 概述

### 1.1 算子名称

`_argreduce_kernel`（承载 `ArgmaxFwdOp` 的 NPU kernel；接口同时保留 `argmin` kind）

### 1.2 功能描述

对 `(M, N)` 输入沿 N 维（规约维）求**首个**最大值（argmax）/最小值（argmin）位置，输出 `(M,)` int64 索引；first-occurrence tie-break，与 `torch.argmax` 精确一致。

### 1.3 数学公式

$$y[i] = \min\{\, j \in [0,N) : x[i,j] = \mathrm{ext}_i(x) \,\},\qquad \mathrm{ext} = \max\ \text{for argmax},\ \min\ \text{for argmin}$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解）：

$$m[i] = \mathrm{reduce}_{\max/\min,\,j}\,x[i,j];\qquad y[i] = \mathrm{int64}\Big(\min_{j<N}\ \mathrm{if\_then\_else}\big(x[i,j]=m[i],\ j,\ \mathrm{BIG}\big)\Big),\quad \mathrm{BIG}=2^{30}$$

大 N（超 UB 驻留）时的在线单遍递推（§1.6.0-C2，tile-0 初始化 + 严格大于更新）：

$$\text{tile}_t:\ (c_m, c_f)=\Big(\max_{j\in t}x[\cdot,j],\ \min_{j\in t}\mathrm{ite}(x[\cdot,j]=\textstyle\max_t, j, \mathrm{BIG})\Big);\quad (r_m,r_f)\leftarrow\begin{cases}(c_m,\,t{+}c_f) & c_m>r_m\\ (r_m,r_f) & \text{else}\end{cases}$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

按 N 驻留性双路径（§0.5/§0.6 决策 + §1.6.0 调研选定；与源算法差异逐条标注来源）：

**路径 S（resident，N × B(dtype) × bm ≤ 64KB，整行驻留 UB）**——覆盖 smoke (32,256)、hidden-state (2048,4096)、3d-non-last-axis（**M=524288, N=4**，dim=0 单维路径，v1 更正）等全部 manifest 负载：
1. `T.copy` GM 行块 → UB（src/dst 双显式 slice 到 real_m）→ `T.copy` UB → **fragment**；
2. `T.reduce_max/min`（argmin 直接 reduce_min，§0.6-R1/E2；fp16 原域、bf16 预 cast fp32、fp32 原域，E4）→ 行极值 m (bm,1)；
3. bm=1 时 `T.vbrc(m → (bm,N))`（编译器 (·,1) 不变操作数 bug 规避，§9.2-R5），bm≥2 直接 (bm,1) 索引；
4. 融合 `T.Parallel(bm,N)` 单遍生成候选 `cand[i,j] = if_then_else(x[i,j]==m[i(,j)], j, BIG)`（j 分支 fp32；向量化器自动生成 varange+vcast，探针 IR 证实）；
5. `T.reduce_min(cand, first, dim=1)` → 首现索引（fp32，精确整数域）；
6. `T.Parallel(bm)` 出栈 `out_ub[i] = cast(first[i,0], "int64")`（bm 元素、恒单发向量 op；基础阶梯 ≤8 / 扩展阶梯（§5.2，3d 类负载）至 2048，logsumexp 同款形态）→ `T.copy` → GM。

**路径 T（tiled online，N 超驻留上限）**——覆盖 lm-head (4,102400)：
1. tile 0（静态宽度 tile_n）：copy → fragment → `reduce_max` **直接写入 running_max** → 融合候选（局部 j）→ `reduce_min` **直接写入 running_idx**（tile-0 初始化，消除哨兵边界 bug——全 -inf 行在「仅严格大于更新」下会错误保持哨兵，tile-0 直写彻底规避，E3 论证）；
2. `T.serial(num_full-1)`（静态边界）逐全 tile：copy → fragment → chunk_max → vbrc（bm=1 必需）→ 融合候选（局部 j）→ reduce_min → chunk_first_local；`T.Parallel(bm)` 全局化 `chunk_global = t·tile_n + local`（纯算术、无 select codegen）；`vcmp(chunk_max, running_max, "gt")` + 两个 `vselect` 原位更新（严格大于才更新——升序 tile 下相等即保留更早索引，first-occurrence 保持）；
3. 静态尾 tile（`tail = N mod tile_n` 工厂期常量，独立静态宽度代码路径，探针 P2 验证）同 2；
4. 出栈同路径 S。

**与源算法差异**：串行扫描→向量化候选+reduce_min（§0.6-R1）；fp32 cast 舍弃（fp16，E4）；argmin 直译（E2）；host pad 消除（§0.6-R3）；persistent 一维分核（§0.6-R2）；block_m/tile_n UB 预算（§0.6-R4）。算法结构（两步：极值→首现索引）**保留**自源码（§0.5）。

### 1.5 数据流图

```
路径 S（resident，纯 Vector，单引擎单 kernel）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub] --T.copy--> FRAG[x_frag]
  --T.reduce_max/min(dim=1)--> FRAG[m (bm,1)]
  --[bm=1: T.vbrc]--> FRAG[ext_brc (bm,N)]
  --T.Parallel 融合(if_then_else, varange)--> FRAG[cand (bm,N) fp32]
  --T.reduce_min(dim=1)--> FRAG[first (bm,1) fp32]
  --T.Parallel(bm)+cast--> UB[out_ub int64] --T.copy(slice)--> GM[out]

路径 T（tiled online）：每 tile 重复 [GM→UB→FRAG→chunk 极值→候选→reduce_min→全局化→gt 更新]，
running_(max,idx) (bm,1) fragment 跨 tile 保持在片上；尾 tile 静态宽度独立缓冲。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：first-occurrence argmax/argmin 这一数学语义的**算法族**（GPU 源算法只是基线候选之一）。信息源：skill 参考表 `references/algorithm-candidates.md` + kb_search（命中 CASE-reduction-logsumexp / VP-2026-0044 / ALG-elementwise）+ 本仓 `examples/`（logsumexp single/tiled、ssd_chunk_scan 的 T.arange 用法）+ pattern-library（constants/traps/**layout——PL-1.1-transpose-chain 转置实测代价，v1 补**）+ 前次任务残留工件 + 结构判据分析。argmax 属规约类 → **完整调研**（R1 候选表 + R3 四口径对比 + R4 逐候选评估）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源两步法** | 并行 reduce 极值 + 串行扫描首匹配（early-exit） | — | — | ✅（基线） |
| 1 | **C1 向量化两步（驻留式）** | `first = reduce_min_j(ite(x_j=m, j, BIG))` | 数学恒等（m 必被命中；BIG>N 精确哨兵） | 串行标量扫描 → 2 个向量 pass；消除 loop_break 标量依赖 | ✅ |
| 2 | **C2 在线单遍（流式）** | running (max, first-idx) 对 + 严格大于更新（tile-0 初始化） | 数学恒等（比较-only，无值域算术；归纳论证见 §1.6.1-E3） | 大 N 单遍 GM 扫描；中间缓冲 O(N)→O(tile) | ✅ |
| 3 | C3 int64 复合键打包 | `key = monotonic_bits(x)·2³² + (2³²−1−j)` 单次 reduce_max 后解码 | 数学恒等（IEEE 位序单调映射） | 单 reduce 取代两步——但键构造需 vbitcast+vshl+vor/按位或链 ~5+ 全宽 pass，**净亏**（int64 支持链：`T.vcmp`✓/`T.reduce`✓ i64、`T.vshl/vor` 见 `docs/Tilelang.language/逻辑操作/`；fp32 键 mantissa 24bit < 值 11+j 17bit 打包需求，**结构上必须 int64**） | ❌（pass 数劣于 C1/C2） |
| 4 | C4 全排序取首 | argsort(x)[0] | 数学恒等 | O(N log N) ≫ O(N) | ❌（复杂度劣） |
| 5 | C5 argmin 的 `-max(-x)` 变换（源码方案） | 取负→argmax→还原 | 数学恒等（严格单调+保等） | 复用单实现；但 NPU 有 reduce_min → 变换冗余 | ✅（作为 E2 基线对照） |
| 6 | C6 跨核 N-split partial+merge 两段 kernel | 列分片并行 → workspace (M,nchunk)×(max,idx) → merge kernel 归并 | 数学恒等（merge 同 C1 逻辑跨 chunk） | 小 M 大 N 负载核利用率 4/48 → 48/48 | ⏸ Stage 4 候选（单 kernel 主选的理由与合并 tie-break 公式见 §1.6.3；备选完整结构 + 判定阈值 + 回写路径见 **§1.6.4**，v1 补齐三件套） |

**R2 在线算法**：**有**——argmax 的在线变体 = running (极值, 首现索引) 对（结构判据：argmax 可分解为 running 统计量 max + 可合并的 first-idx；与 online softmax 的 (max,sum) 对同构，logsumexp tiled 已验证同族结构）。收益口径：tiled 场景 GM 扫描 2 遍→**1 遍**（两遍式 pass1 找全局极值 + pass2 重读找首现）；中间缓冲 O(N)→O(tile)；UB 驻留可行性：tile 级 (bm,tile_n) 工作集 ≤ 64KB（§5.2 表）。驻留场景（路径 S）数据已在片上，「在线」无额外收益（片上两 pass 零 GM 代价）——**仅在 tiled 路径采纳**。

**R3 复杂度对比**（四口径；per-row 计，M 行并行摊派；带宽受限算子——roofline flops=M·N vs bytes=M·N·elem+M·8，访存主导）：

| 算法候选 | FLOPs（比较次数，口径：per row） | 访存量 Bytes（GM，口径：读 x + 写 out + 中间往返） | 扫描遍数（GM） | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|------|------|------|------|------|
| 基线（GPU 两步） | N（reduce）+ N/2 平均（标量扫描） | N·elem + 8（1 遍） | 1 | x_f32 fragment 4B/elem（驻留） | M/bm 行块独立，无跨核 |
| C1 驻留两步向量化 | N（reduce_max）+ N（融合候选：比较+选择）+ N（reduce_min）≈ 3N 向量操作 | N·elem + 8（1 遍） | 1 | x_frag B/2 + cand 4B ≈ **8–16B/elem**（fp16 8/10、bf16 10/14、fp32 12/16；bm≥2 / bm=1 含 ext_brc，与 §4.5 同口径——v1 补 fp32 bm=1=16） | 同上 |
| C2 在线单遍（tiled） | 每 tile：N_t（chunk max）+ N_t（候选）+ N_t（min）+ O(1) 更新 ≈ 3N 向量 + 3·n_tiles 次微操作 | N·elem + 8（**1 遍**）+ workspace 0 | **1** | O(tile)：(bm,tile_n)×(2..4B) ≈ 10–16B/elem·tile_n | 行独立；lm-head 类 M<48 时核利用率低（§1.6.3-#4 / §1.6.4 C6 承接） |
| C3 int64 打包 | N（键构造 ~5 pass）+ N（reduce）+ 解码 ≈ 7N | N·elem + 8（1 遍） | 1 | 键 int64 8B + 输入 B/2 ≈ 10–12B/elem | 同 C1 |
| （对照）tiled 两遍式 | 2N | **2N·elem** + 8（重读） | **2** | O(tile) | 同 C2 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 pattern-library/constants.md 条目 + 探针）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（串行扫描形态） | ❌ 标量 early-exit 扫描与 Vector 逐元素/规约模型不符（N/2 次标量 op × 0.5µs 发射 ≫ 向量 pass） | — | — | — | — | ❌ 形态需重设计（→C1） |
| C1 驻留两步向量化 | ✅ 纯 Vector（reduce_max/min + if_then_else 融合 + reduce_min 全向量原语；`T.reduce.md` dtype 表 ✓ fp16/fp32/i64，**bf16 ✗→预 cast**） | ✅ 驻留 ≤64KB/block（UB 192KB÷auto-multi-buffer 膨胀 1.6–2.0×〔CONST-capacity-910B2C + 探针实测 256.03KB/160KB=1.60×〕；fragment 计算形态规避多消费者竞态，探针 3/3 稳定） | ✅ 32B 尾轴对齐即可（logsumexp N=300 先例）；N 任意（静态尾列天然由驻留整行覆盖） | ✅ M/N/tile 边界全工厂期常量 | ✅ auto-multi-buffer 流水（探针验证）；无跨引擎往返 | ✅ **驻留路径选定** |
| C2 在线单遍 | ✅ 纯 Vector + (bm,1) 微操作链（vcmp/vselect） | ✅ tile 工作集 ≤64KB（fp16 tn≤6553、bf16/fp32 tn≤4096–4681） | ✅ tile_n 256 倍数由 compute_tile_n 除数优先；**静态尾 tile** 特化（kernel 侧掩码有 logsumexp 编译器 segfault 反例，故用静态特化） | ✅ num_full/tail/serial 边界全静态；**bm=1 强制**（TRAP-multitile-bm-gt1-segfault：bm>1+num_tiles>1 → 编译器 SIGSEGV，前次任务 repro 实证） | ✅ tile 循环 auto-multi-buffer（探针 P2b 96 block 多波次 3/3 稳定） | ✅ **tiled 路径选定** |
| C3 int64 打包 | ✅ Vector | ⚠️ 键 8B/elem 预算减半 | — | — | ⚠️ 键构造链多 pass | ❌ pass 数劣 |
| C6 N-split merge | ✅ | ⚠️ workspace GM 往返（M·nchunk·(elem+4)B ×2——fp16 lm-head 实例 ≈3.8KB，≪ x 读流量 800KB） | — | ✅ | ⚠️ 两 kernel 启动 + 合并正确性面（前次任务 round8_op9 首版 4/4 行 golden mismatch 实证该风险；round9 终版已修复：partial ≈15.7µs + merge ≈1.7µs，golden PASS） | ⏸ Stage 4（**完整备选结构 + 判定阈值（端到端 < 0.8× 主选则翻转）+ 回写路径见 §1.6.4**，v1 补齐三件套） |

**调研结论**：选定 **C1（驻留两步向量化）+ C2（在线单遍）按 N 驻留阈值双路径**的算法族。关键依据：R3 表——两口径（GM 遍数、向量 pass 数）均不劣于基线且消除标量扫描；R4——全 Vector 原语可达、驻留/tile 预算经探针实测定标、静态边界满足一维 Kernel 约束。与基线的结构差异一句话：**「极值归约 + 串行扫描」→「极值归约 + 掩码索引 reduce_min（驻留）/ 在线 running-(max,idx)（流式）」**。调研范围（「无更优替代」的查证范围）：algorithm-candidates.md 参考表（reduction 族行 + kb_search 命中条目 CASE-reduction-logsumexp/VP-2026-0044/ALG-elementwise）、examples/（logsumexp single+tiled、ssd_chunk_scan）、pattern-library（constants.md 5 条目、traps）、前次任务 perf_opt/repro 工件、结构判据（在线变体存在性/复合键打包可行性分析）、恒等变形空间（C3/C4/C5 全枚举）。源算法优化手段承接：§0.4 #2 两步结构保留；#3 fp32 cast 由 E4 论证舍弃（意图=数值稳健，对精确选择运算冗余）；#4 变换由 E2 替换为直接 reduce_min（意图=单实现双算子，保留——同一 kernel 双 kind 分支）；#5 单位元 pad 由 R3 消除（意图=非整除正确性，由静态尾 tile 承接）；#1 SMEM tiling 由 UB+fragment 承接。

**设计期 roofline（D-2，dispatch 代表 workload；协议：hardware-cost-model.md §2；常数带版本戳）**：

| workload（manifest） | 路径/配置 | 流量项 | 发射项（向量 op 数 × 0.5µs〔CONST-vector-launch-overhead〕÷48 核） | 估算下界（µs） |
|---|---|---|---|---|
| smoke (32,256) fp16 | S, bm=8, 4 block | 16KB÷(33.9GB/s×4核〔CONST-mte2-degradation 1M 档〕)≈0.1µs | 4×7op÷4核×0.5=3.5µs | **≈4** |
| lm-head (4,102400) fp16 | T, bm=1, tn=5120, 4 block | 800KB÷(30.2×4)≈6.6µs | (6+19×10)op×0.5≈98µs/核（4 核并行）→ 流水化后有效≈30–60µs（前次任务同类 25-tile 实测 32.5µs 校准） | **≈30–60** |
| hidden-state (2048,4096) fp16 | S, bm=2, 1024 block→persistent 48 | 16MB÷(30.2×48)=11.4µs | 1024×7op÷48×0.5≈74µs（前次任务同配置实测 47µs——0.5µs/op 为串行发射上界，流水化后部分隐藏） | **≈47–74** |
| hidden-state bf16 | S, bm=1, 2048 block | 16MB÷(30.2×48)=11.4µs | 2048×8op÷48×0.5≈171µs（前次实测 55µs——同上校准，发射常数在流水下高估 ~3×） | **≈55–100** |
| 3d-non-last-axis→(**M=524288,N=4**) fp16（**原始 N 契约**；dim=0 单维路径 N=4、M=128×4096，v1 更正） | S, bm=2048（扩展阶梯），256 block→persistent 48×6 | 读 4.19MB + 写 4.19MB = 8.4MB ÷ (30.2×48) ≈ 5.8µs（8.4MB 介于 1M–16M 档，取 16M 档保守值） | 256×7op ÷ 48核 × 0.5 ≈ **18.7µs** | **≈19**（前次 pad 契约实测 ~194–205µs——**~10× 改善**，host pad 消除；**反事实锚点**：若沿用基础阶梯 bm=8 → 65536 block → 发射项 ≈4.8ms，劣于 pad 契约 ~24×——即 §5.2 扩展阶梯的存在依据，v1 补） |

> 容量项 n/a（纯 Vector 单引擎、无跨引擎 S/P 往返，CONST-store-fixpipe-gm-only 不适用）。发射项以 CONST-vector-launch-overhead ~0.5µs/op 串行模型为上界；三处用前次任务同配置实测值（47µs/55µs/32.5µs）校准并标注偏差方向（流水化使实际低于串行模型 1.5–3×）。MTE2 曲线按总流量档取 HBM 口径（bench harness sets>1）。3d 行 bm=2048 为扩展阶梯预算最大值（§5.2），(2048,4) 窄内维形态未经探针覆盖（§9.2-R10，L0-6 先行验证）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象：§1.6.0 选定算法（C1+C2）的公式。四要素逐项；**等价性机器验证（D-1）**：`verify_equiv.py`（torch CPU，fp64 参照 + 输入 dtype 仿真 + 角点），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | 首现索引向量化（承接 §0.6-R1） | 串行扫描 `for j: if x[j]==m: break`（N/2 平均标量比较） | `first = reduce_min_j( ite(x_j = m, j, BIG) )`，BIG=2^30 | m 必被 ≥1 个 j 命中（有限/±inf 输入）；非命中候选 BIG=2^30（fp32 精确 2 的幂）> 任何 j<N≤2^24（fp32 精确整数域）；min 恰取最小命中 j = 首现。NaN 行不命中（输出哨兵）——NaN 为文档化边界非等价域（§9.2-R1） | 标量 N/2 次 op（每次 ~0.5µs 发射不可向量化）→ 2 个向量 pass（融合候选 1 + reduce_min 1）；GPU loop_break 依赖消除 |
| E2 | argmin 直译（替换源 `-max(-x)` 变换） | `argmin(x) = argmax(-x)`（取负 pass + reduce_max + 还原 pass） | `argmin(x) = first-match(min(x))`：直接 `T.reduce_min` + 同 E1 候选机制 | 取负严格单调且保等：x_i < x_j ⟺ −x_i > −x_j，x_i = x_j ⟺ −x_i = −x_j ⟹ min 的首现 j ≡ −x 的 max 的首现 j（索引恒等，不经值域算术） | 省 2 个全宽 pass（取负 + 还原）；NPU `T.reduce_min` 存在（`testing/npuir/reduction_ops/test_reduce.py` 先例） |
| E3 | 在线单遍递推（承接 §1.6.0-C2） | 两遍式：pass1 全局极值 → pass2 重读数据找首现（GM 2 遍） | running (r_m, r_f)：tile-0 直写初始化；后续 tile 严格大于才更新 `(c_m, t·tn+c_f)` | 归纳：处理完 tiles 0..t 后 (r_m, r_f) = (已见列的 max, 其首现)。c_m>r_m → 新极值来自本 tile、首现 = 本 tile 内首现（更早列值 ≤r_m<c_m 不可能命中）；c_m=r_m → 相等时保留更早索引（升序 tile 下本 tile 索引必大于已见索引）；c_m<r_m → 保持。tile-0 直写消除哨兵初始化（否则全 −inf 行 c_m=−inf=r_m 不触发严格大于、首现丢失——探针 P2 全 −inf 行用例覆盖）。比较-only：无值域算术 → 无舍入路径 | tiled 路径 GM 2 遍→1 遍（lm-head：−800KB 读流量 ≈ −6.6µs@4 核）；中间缓冲 O(N)→O(tile) |
| E4 | fp16 免 fp32 上抛（舍弃源全量 cast） | `m = reduce_max(cast(x, fp32))`，匹配在 fp32 域 | `m = reduce_max(x)`（fp16 原域），匹配 `x[j] == m`（fp16 域，同 dtype 比较） | max/min 为**精确选择**运算（无舍入）：fp16 域 max 的结果 = fp32 域 max 经 fp16↔fp32 单射往返的结果（fp16→fp32 精确单射保序保等）；等值比较在两域判定一致 ⟹ 首现索引恒等。bf16 不适用此优化（reduce 不支持 bf16，`T.reduce.md` §2.2.1——bf16 保留 cast fp32，非精度需求而是 API 需求）；fp32 输入无 cast | fp16 路径省 1 个全宽 pass + 2B/elem 缓冲（驻留预算翻倍：B 8→6B/elem 等效）；**注意探针实测：reduce 混合 dtype（fp16 src→fp32 dst）静默产出错误值——本优化的「同域比较」正是其规避形态** |

**机器验证结果表**（`verify_equiv.py`，2026-09-24 执行，exit 0；输入 dtype 仿真 + fp64 参照；shape 覆盖 (1,1)/(1,2)/(3,300)/(8,256)/(16,1024)/(7,4096)/(4,102400)×{fp16,bf16,fp32} + 角点行〔全−inf/全+inf/重复极值/±inf 混合/亚正规最大/混合符号零/dtype 极值〕×（tile_n ∈ 128..30000，含跨 tile tie / 尾 tile tie / 全 −inf））：

| 验证项 | 用例行数 | 最大偏差 | 违反率 | 结论 |
|--------|---------|---------|--------|------|
| E1 argmax/argmin（vs 串行扫描基线） | 48（含角点 ×3 dtype） | 0（索引精确相等） | 0/114 | **EQUIV_PASS** |
| E2 argmin 直译（vs −max(−x) 基线） | 24 | 0 | 0 | **EQUIV_PASS** |
| E3 在线递推（vs 直接 argmax 基线） | 33（3 dtype ×〔8 组 shape×tile_n + 2 角点 + 1 跨 tile tie〕；含全 −inf/尾 tile 唯一命中，v1 与脚本输出对账更正——原记 40） | 0 | 0 | **EQUIV_PASS** |
| E4 fp16 原域（vs fp32 上抛基线） | 8 | 0 | 0 | **EQUIV_PASS** |
| INFO：NaN 行为（等价域外，行为记录） | 1 | 3/4 行分歧（候选=哨兵 vs 基线=0/首 NaN）— **符合声明的边界行为** | — | INFO(diverge)，非门禁项 |
| **合计** | **114** | — | **0 项失败** | **ALL EQUIV_PASS** |

**优化结论**：采纳 E1–E4 共 4 项；优化后公式 = §1.3 第二、三式（供 §3.1 拆解）：驻留式 `first = reduce_min(ite(x=reduce_ext(x), j, 2^30))`（fp16/fp32 原域，bf16 预 cast fp32；argmin 用 reduce_min 直译）；流式在线递推（tile-0 直写 + 严格大于更新）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/` 或 `examples/`/`testing/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源码/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | 首现索引扫描（源步骤 4） | `T.Serial(N)` 标量比较 + `T.loop_break` | `T.Parallel` 融合候选（`T.if_then_else` 元素级，先例 `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py` + 探针 P1）+ `T.reduce_min(dim=1)`（`T.reduce.md` reduce_mode="min"） | ✅ | — |
| 2 | 极值归约（源步骤 3a） | （源已是 reduce 原语；朴素替代为标量扫描） | `T.reduce_max` / `T.reduce_min`（dim=1，`规约操作/T.reduce_max.md`；**同 dtype 约束**探针实证） | ✅ | — |
| 3 | argmin 取负/还原（源步骤 3b 两处全宽循环） | `T.Parallel` 逐元素取负 ×2 | 直接 `T.reduce_min`（E2；`testing/npuir/reduction_ops/test_reduce.py` L38 `T.reduce_min(c, s, clear=False)` 先例——v1 更正行号） | ✅（消除） | — |
| 4 | fp32 cast（源步骤 2） | `T.Parallel` 全宽 cast 循环 | fp16：**消除**（E4，原域 reduce+比较）；bf16：`T.vcast(rint)`（`数据类型转换操作/T.vcast.md` bf16→f32 行；**禁 f32→f32 vcast**——CONST-vcast-f32-to-f32-corrupt）；fp32：无 cast | ✅ | — |
| 5 | 索引序列生成 | 朴素：`T.Parallel` 赋值 `idx[i,j]=j` | 融合式免显式索引缓冲（向量化器对 `T.cast(j,"float32")` 自动生成 varange+vcast，探针 IR 证实）；链式备选：`T.arange(buf,[0,1],0)`（`创建操作/T.arange.md` + ssd_chunk_scan L352 先例） | ✅ | — |
| 6 | 极值广播到 (bm,N) | 朴素：(bm,1) 循环填充 | bm≥2：融合循环直接 (bm,1) 索引（i 随循环变化、非不变操作数——探针 P1 bm=4 验证）；bm=1：`T.vbrc`（`shape操作/T.vbrc.md` 张量广播 (bm,1)→(bm,N)；**bm=1 时 (·,1) 不变操作数触发编译器 i1 误型 bug**，探针 P2 定位，§9.2-R5） | ✅ | — |
| 7 | 在线更新（tiled） | 朴素：逐元素条件更新 | `T.vcmp("gt")`（`比较操作/T.vcmp.md`）+ `T.vselect` 原位 ×2（`条件操作/T.vselect.md`；探针 P2/P2b 验证）于 (bm,1) 微缓冲 | ✅ | — |
| 8 | tile 基址全局化（tiled） | — | `T.Parallel(bm)` 标量算术 `chunk_global = t·tn + local`（bm=1 恒为 1 次迭代；纯 TIR 算术无 select codegen） | ❌（保留标量） | tile 级顺序依赖载体（在线更新的跨 tile 跨携带状态）；bm=1（multitile 陷阱强制）下为单元素语句，非逐元素热点；替代形态（融合循环 value 分支含 `t·tn+j`）未探针验证、且有 logsumexp 条件含 serial 变量致 segfault 的同类 codegen 风险反例（`_logsumexp_kernel_tiled.py` L24-33 记录）——列为 Stage 4 简化候选 |
| 9 | int64 出栈 epilogue | — | `T.Parallel(bm)` Parallel cast（bm 元素）+ `T.copy` UB→GM | ❌（保留 Parallel 循环） | 形状收缩类：每 block 1 次发射的 (bm,1)→(bm,) 形状收缩，无 cast+收缩复合向量 API（vcast 不改 shape；logsumexp single L108-109 同款先例）；bm 取值随阶梯（基础 ≤8 / 扩展 §5.2 至 2048）但**恒为单发向量 op**（发射 1 op，非逐元素热点） |
| 10 | 尾块行数 real_m / 任务 guard | — | `T.min` 标量 + `if start < M` 动态 guard（`vcast.md` §2.4 示例先例） | ❌（保留标量） | 依赖运行时 pid 的 block 级边界处理（且已说明静态化方式：serial 上界静态 + guard 动态）；VP-2026-0044 三件套标准形态 |
| 11 | host 侧 block_m/tile_n/路径选择 | Python 工厂期计算 | — | ❌（保留） | host 元数据计算（不在 kernel 内；每 (M,N,dtype) 一次） |

**向量化结论**：逐元素计算已全部向量化（1–7，API 均有佐证）；保留 4 处标量/循环（8–11），理由类别：tile 级顺序依赖（#8）、形状收缩类（#9，恒单发向量 op）、动态边界（#10）、host 元数据（#11）——均非逐元素热点（发射项均 1 op 量级），逐项见上表。源实现中唯一的逐元素标量热点（串行扫描）已被 E1 向量化替代。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（(M,N) 行主序进 / (M,) int64 出）；**核内布局与 lane 映射是设计变量**。本算子 2D（M,N)——按规约类必选候选枚举。

**轴质量评分**（规约类：水平归约 vs 垂直扫描为正交决策，须分别枚举）：

| 评分项 | 水平归约（lane→N 轴，reduce dim=1） | 垂直扫描（lane→M 行，串行步进 N） | 说明 |
|--------|------|------|------|
| 整除性 | N 任意（reduce/if_then_else 原语处理任意宽度；logsumexp N=300 先例 + 探针 P1 N=300 直证）；尾 lane 浪费由硬件归约吸收 | M 任意同理 | 无差异 |
| 累加链形态 | 归约维即向量维（dim=1 连续 stride=1）——reduce 原生形态 ✓ | 归约维变串行步进维（N 次 (bm,1) 微操作 × 0.5µs 发射 = N/2·0.5µs/行——发射爆炸） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | 行块 (bm,N) 驻留（§5.2 预算表） | 同 | — |

**候选矩阵**（含 GPU 源轴选择对照——源码 thread/warp 轴是输入非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行块 × 全 N 驻留 + 水平归约**（lane→N，reduce dim=1；persistent 按 M 行块分核） | N（连续） | 无 | 驻留 ≤64KB/block（探针定标）；向量 pass 5–7/block；VP-2026-0044 row-reduction 基准形态（logsumexp+ada 两任务实证） | ✅ **主选** |
| 2 | 垂直扫描（lane→M，串行步进 N 维） | M | 无 | 归约维串行化 → 每行 N 次微操作发射（N=4096 → 2ms/行量级）——发射项灾难 | ❌（发射模型量化） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链 ×2（UB 级；`创建操作/T.transpose.md` **§2.2.1 dtype 表：int8/int16/int32/uint64/int64/fp16/fp32/bf16 全 √**（仅 uint8/uint16/uint32/bool ×）、§2.3 真实限制 = 仅二轴交换（3D 须拆链）+ `size` 参数不可用——本算子 2D，二轴交换即足；实测代价引 pattern-library `layout.md` **PL-1.1-transpose-chain**：4 次融合转置合计 ~5.4µs，status: verified〔v1 更正：删除 v0 的「dtype 限 fp16」错误论断〕〕） | **纯开销零收益**：主选 #1 的向量化轴 N 即 I/O 原生连续轴、无 repack 需求，任何核内转置不减少 GM 流量与向量 pass 数；另需 +2 个 (bm,N) UB 缓冲（按 §4.5 口径 +4~8B/elem，直接吃掉 hidden-state fp16 bm=2 的预算裕度）——即使 PL-1.1 实测代价小（~µs 级）也无任何收益可抵 | ❌ |
| 4 | N-split 跨核（C6：列分片 partial+merge 两段 kernel） | N（跨核切分） | workspace GM 往返 (M,nchunk)×2（fp16 lm-head 实例 ≈3.8KB） | 小 M 大 N（lm-head）核利用 4/48→48/48；**前次 round9 终版实测锚点（v1 补）**：partial ≈15.7µs + merge ≈1.7–2.7µs（合计 ≈17.4µs，golden PASS，`perf_opt/logs/final_round9_final_stage4_lm-head-argmax-fp16_argreduce_{partial,merge}.log`）≈ **0.54×** 主选 32.5µs；代价：+1 kernel 启动、workspace 往返、**合并 tie-break 正确性面**（前次任务 round8_op9 首版 4/4 行 mismatch 实证风险——终版已修复；正确合并式：`global_m = reduce_max(partial_m); first = reduce_min_c( ite(partial_m[c]=global_m, partial_first[c], BIG) )`——与 E1 同构，机器验证背书） | ⏸ **Stage 4 A/B**（完整备选结构 + 判定阈值〔C6 端到端 < 0.8× 主选则该 shape 类翻转〕+ 回写路径见 **§1.6.4**，v1 补齐三件套；另见 §6.3-③/§9.2-R3） |
| 5 | 融合 if_then_else 候选 vs vcmp/vselect 链（同轴 #1 内的两种 pass 组织） | N | 无 | 融合式：全宽 pass 4–5（copy/reduce/融合/reduce_min）+ 缓冲 6–14B/elem；链式（vbrc+vcmp+arange+vselect+reduce_min）：pass 7–8 + 缓冲 +4B/elem（fp16 bm=4@4096 预算即不敷——探针 p3a 首版编译溢出 256KB 的直接原因之一） | ✅ **融合式主选**（探针 P1/P3 实证；链式为 codegen 反例时的回退形态，已探针验证可用） |

**布局决策结论**：选定方案 #1+#5：核内布局 = 行块 (bm,N) 行主序驻留（UB staging + fragment 计算）、向量化轴 = **N 轴**（reduce dim=1、融合循环 T.Parallel(bm,N)）、repack 无、分核 = persistent 按 M 行块（§5.5）；该结论同步落入 §3.3 伪代码与 §6 循环结构（融合循环的内层向量维 = N）。弃选理由：#2 发射爆炸（量化）；#3 **纯开销零收益**——主选轴 N 即 I/O 原生连续轴、无 repack 需求（PL-1.1 实测 ~5.4µs 代价小亦无收益可抵；v1 更正：v0 的「dtype 缺口」论断与 `T.transpose.md` §2.2.1 全 √ 表矛盾，已删除）；#4 为 Stage 4 实验裁决项——**三件套齐备**（v1 补）：主选 = 当前证据下单 kernel 最优（正确性面最小、无 workspace）、备选完整结构与分核参数见 §1.6.4、判定阈值（C6 端到端 < 0.8× 主选则该 shape 类翻转，按 shape 分派）与回写路径同见 §1.6.4；「单 kernel 主选」的准确边界：`_argreduce_kernel` 工厂契约返回单 kernel、harness `ArgreduceKernel.forward` 单次发射（`argmax.py` L65）——C6 需 wrapper 层双发射 + workspace 管理（**前次 round9 已证明该形态可在 harness bench 内跑通且 golden PASS**，故 C6 采纳只改 wrapper 组合层、不改 kernel 工厂契约，非结构不可行）。

#### 1.6.4 实验裁决备选方案：C6 N-split partial+merge（Stage 4 A/B 三件套）⭐（v1 新增）

> §1.6.3-#4 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。主选 = §1.4 单 kernel 双路径（当前证据最优：正确性面最小、无 workspace 往返）；触发形态 = 小 M 大 N 负载（lm-head 类）核利用率 4/48。§3.3/§4/§6 只承载主选方案，本节结构独立成节、不与主选混排。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：`M ≤ 16 且 ceildiv(M, bm) < 48 且 N ≥ 32768`——lm-head (4,102400) 命中；hidden-state / 3d / smoke 不命中（走主选）。工厂 lru_cache 本就按 (M,N,dtype) 分派，C6 与主选可按 shape 共存（无全局二选一约束）。

**② 备选结构（两段 kernel，均为 Developer 单 kernel 形态；fp16 lm-head 实例参数：bm=M=4、tn=1280、nchunk=80）**：

```python
# ── partial kernel（argreduce_partial）：逻辑块 = nchunk 列分片，每块处理 (M, tn) ──
# tn/nchunk：§5.2 同款规则在 partial 预算下取值——fp16：slab=ub_slab_units(2, 2, 1)=4
#   （x_ub/x_frag fp16 + cand fp32），4×4×tn×2B ≤ 0.9×64KB → tn ≤ 1843 → N 的 256 倍数
#   整除因子取最大 → tn=1280、nchunk=80（102400=80×1280，零尾）；手工预算 40960B=40KB
with T.Kernel(min(nchunk, 48), is_npu=True) as (cid, _):        # 分核：逻辑 80 ≫ 48 → persistent
    x_ub   = T.alloc_shared((M, tn), dtype)                      # GM staging（§4.5 纪律②）
    x_frag = T.alloc_fragment((M, tn), dtype)                    # 计算缓冲（bm=M≥2，无需 ext_brc）
    m      = T.alloc_fragment((M, 1), dtype)
    cand   = T.alloc_fragment((M, tn), "float32")
    first  = T.alloc_fragment((M, 1), "float32")
    gidx   = T.alloc_fragment((M,), "float32")
    m_ub, gidx_ub = T.alloc_shared((M,), dtype), T.alloc_shared((M,), "float32")
    for s in T.serial(ceildiv(nchunk, num_kernels)):            # 静态上界（80/48 → 2）+ guard
        c = cid + s * num_kernels
        if c < nchunk:
            T.copy(x[0:M, c*tn:(c+1)*tn], x_ub[0:M, 0:tn])      # 列块 2D slice（tiled 路径同款）
            T.copy(x_ub, x_frag)
            T.reduce_max(x_frag, m, dim=1)                       # 驻留路径 S2–S5 同款链（E4：原域）
            for i, j in T.Parallel(M, tn):                       # 融合候选（局部 j）
                cand[i, j] = T.if_then_else(x_frag[i, j] == m[i, 0],
                                            T.cast(j, "float32"), T.float32(BIG))
            T.reduce_min(cand, first, dim=1)
            for i in T.Parallel(M):                              # 全局化（§1.6.2-#8 同款标量算术）
                gidx[i] = c * tn + first[i, 0]
                m_ub[i] = m[i, 0]                                # (M,1)→(M,) 形状收缩（#9 同款）
            T.copy(m_ub[0:M], ws_max[0:M, c])                    # workspace 列切片写出
            T.copy(gidx_ub[0:M], ws_idx[0:M, c])                 # （UB staging + slice copy 纪律）

# ── merge kernel（argreduce_merge）：单块归并，E1 同构 tie-break ──
with T.Kernel(1, is_npu=True) as (cid, _):                       # 分核：逻辑 1 ≤ 48，单块
    ws_max_f = T.alloc_fragment((M, nchunk), cdtype)             # workspace → fragment
    ws_idx_f = T.alloc_fragment((M, nchunk), "float32")
    g   = T.alloc_fragment((M, 1), cdtype)
    candm = T.alloc_fragment((M, nchunk), "float32")
    fout = T.alloc_fragment((M, 1), "float32")
    T.copy(ws_max[0:M, 0:nchunk], ws_max_f)                      # GM workspace → fragment
    T.copy(ws_idx[0:M, 0:nchunk], ws_idx_f)
    T.reduce_max(ws_max_f, g, dim=1)                             # 全局极值（行向，已验证形态）
    for i, c in T.Parallel(M, nchunk):                           # tie-break：与 E1 同构
        candm[i, c] = T.if_then_else(ws_max_f[i, c] == g[i, 0],
                                     ws_idx_f[i, c], T.float32(BIG))
    T.reduce_min(candm, fout, dim=1)                             # 首现 = 命中 chunk 的最小全局索引
    for i in T.Parallel(M):                                      # int64 出栈（主选 S6 同款）
        out_ub[i] = T.cast(fout[i, 0], "int64")
    T.copy(out_ub[0:M], out[0:M])
```

- **workspace（GM，wrapper 层分配）**：`ws_max (M, nchunk)` 计算域 dtype + `ws_idx (M, nchunk)` fp32——fp16 lm-head 实例 4×80×(2+4)B ≈ 1.9KB，往返 ≈3.8KB ≪ x 读流量 800KB。**tie-break 正确性**：ws_idx[i,c] = c·tn + 首现局部索引（已全局化），chunk 索引随列位单调递增 → 命中集上取 min 即全局首现（与 E1 机器验证同构；round8 首版 4/4 mismatch 的教训环节，round9 终版 golden PASS）。
- **UB 预算（partial，fp16 tn=1280）**：slab=4（x_ub/x_frag fp16 + cand fp32）× bm=4 × tn=1280 × 2B = 40960B = 40KB ≤ 0.9×64KB ✓；merge 侧 (M,nchunk)=(4,80) 微缓冲 ≪ 预算。dtype 路径与主选一致（fp16 原域 / bf16 vcast 预转 / fp32 原域，E4 适用；bf16 partial 无 max_brc〔bm=M≥2 直索引〕→ slab=ub_slab_units(2,1,2)=5（x_ub bf16 + x_work/cand fp32）→ 5×4×tn×2B ≤ 0.9×64KB → tn ≤ 1474 → 整除因子同取 **1280**、预算 51200B=50KB ✓；fp32 partial slab=ub_slab_units(4,3,0)=3 → tn=1024/nchunk=100）。
- **分核三要素（partial）**：逻辑核数 = nchunk（80）；物理核数 = 48（§5.5 实查同源）；规模判定 = 极大规模（80 > 48）→ persistent 48 + 核内 T.serial（静态上界 2 + guard）。merge：逻辑 1 ≤ 物理 → 单块直发。
- **同步**：两 kernel 内部 Developer 自动同步（§7 同款）；kernel 间依赖由 wrapper 顺序发射保证（无核间 sync 需求）。

**③ 实验裁决计划**：代表 shape = lm-head fp16 (4,102400) + lm-head bf16 (4,102400)；指标 = bench harness Task Duration（launches≥30 均值，partial+merge 两段之和）且 golden `torch.equal` 全 PASS 为前提；**判定阈值：C6 端到端 < 0.8× 主选单 kernel 同 shape 延迟 → 该 shape 类分派翻转**（前次 round9 实测锚点 ≈0.54×，若本轮主选实测维持 ≥30µs 则大概率翻转）；**回写路径**：实测数据与结论回写本节裁决记录 + RETROSPECTIVE.md，翻转时修订 §1.4/§3.3 增补 C6 分派分支（wrapper 双发射 + workspace 分配）并过 Stage 2 复审；Stage 3 实现期如需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（归约 + 逐元素融合候选 + 微操作比较选择），无 matmul、无 Cube | 不需要 L1/L0；`alloc_shared`→UB + `alloc_fragment` 即可 |
| 归约 | reduce_max/reduce_min（dim=1）+ reduce_min（候选）——Developer 模式可用（`testing/npuir/reduction_ops/` 全部 Developer 先例） | ✓ |
| 内存层级 | 仅 GM ↔ UB ↔ fragment，无跨引擎 | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作（C6 Stage 4 才引入跨核 merge） | Developer 自动同步即可 |
| 家族先例 | logsumexp single/tiled、ada_layer_norm（VP-2026-0044 row-reduction 基准形态）均为 Developer | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer 模式 | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging，编译器映射 UB）+ `T.alloc_fragment`（全部计算缓冲——竞态规避形态，§9.2-R4） |
| 计算方式 | v 前缀 API（vbrc/vcmp/vselect/vcast）+ reduce_max/reduce_min + `T.Parallel` 融合循环（if_then_else 元素级） |
| 同步 | Developer 自动同步（无手动 sync_block_set/wait） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

**驻留路径 S**（argmax；argmin 把 `reduce_max`→`reduce_min`）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| S1 | `x_ub ← x[block 行块, 0:N]`；`x_frag ← x_ub` | GM→UB→fragment（src/dst 双 slice 到 real_m） |
| S2 | `m = reduce_max(x_frag, dim=1)` | 行极值 (bm,1)；fp16/fp32 原域、bf16 先 `vcast(x_ub→x_work, rint)`（fp32 域） |
| S3 | bm=1：`ext = vbrc(m → (bm,N))`；bm≥2：融合循环直接索引 `m[i,0]` | 极值广播（编译器 bug 规避分叉，§9.2-R5） |
| S4 | `cand[i,j] = ite(x[i,j] == m[i(,j)], float32(j), 2^30)` | 融合候选单 pass（向量化器自动 varange+vcast 索引分支） |
| S5 | `first = reduce_min(cand, dim=1)` | 首现索引 (bm,1) fp32 |
| S6 | `out_ub[i] = int64(first[i,0])`；`out[block] ← out_ub[0:real_m]` | int64 出栈 + GM 写（slice） |

**流式路径 T**（tiled online；tile-0 为 S2/S4/S5 直写 running，全 tile/尾 tile 追加更新链）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| T1 | tile-0：`running_max = reduce_max(tile0)`；`running_idx = reduce_min(ite(tile0=running_max, j, BIG))` | 直写初始化（E3；局部 j 即全局 j） |
| T2 | 全 tile t：`c_m = reduce_max(tile_t)`；`c_f = reduce_min(ite(tile_t=c_m, j, BIG))`；`c_g = t·tn + c_f`；`if c_m > running_max: (running_max, running_idx) ← (c_m, c_g)` | 严格大于更新（vcmp "gt" + vselect ×2 原位） |
| T3 | 尾 tile（静态宽度 tail）：同 T2，基址 `num_full·tn` | 静态特化路径 |
| T4 | 同 S6 | 出栈 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| S1/T1 载入 | GM→UB | `T.copy(x[start:start+real_m, c0:c1], buf[0:real_m, 0:w])` | 双显式 slice | Developer | logsumexp single L79-82；探针 P1/P2 |
| S1' UB→fragment | UB→FRAG | `T.copy(x_ub, x_frag)`（同 dtype） | 全缓冲 | Developer | logsumexp L83；**禁 vcast f32→f32**（CONST-vcast-f32-to-f32-corrupt） |
| S2 bf16 预转 | bf16→fp32 | `T.vcast(x_ub, x_work, round_mode="rint")` | rint（bf16→f32 精确） | Developer | `数据类型转换操作/T.vcast.md` §2.2.1 |
| S2 极值 | 行极值 | `T.reduce_max(x_frag, m, dim=1)` / `T.reduce_min(...)` | dim=1；**dst dtype ≡ src dtype**（探针实证混合 dtype 静默错误） | Developer | `规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py` |
| S3 广播 | (bm,1)→(bm,N) | `T.vbrc(m, ext_brc)` | 张量广播（尺寸 1 维扩展） | Developer | `shape操作/T.vbrc.md` §2.2.2 |
| S4 候选 | ite 元素级 | `T.Parallel(bm,N)` + `T.if_then_else(x[i,j]==m, T.cast(j,"float32"), T.float32(2**30))` | 元素级条件（不含 serial 循环变量于**条件**） | Developer | `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py`；探针 P1 |
| S5 首现 | reduce_min dim=1 | `T.reduce_min(cand, first, dim=1)` | fp32 | Developer | `T.reduce.md`（reduce_mode="min"）；`test_tilelang_language_clamp.py` L71 |
| T2 比较/选择 | gt 更新 | `T.vcmp(c_m, r_m, cond, "gt")` + `T.vselect(cond, c_m, r_m, r_m)` + `T.vselect(cond, c_g, r_f, r_f)` | (bm,1) 原位 | Developer | `比较操作/T.vcmp.md`、`条件操作/T.vselect.md`；探针 P2/P2b 原位验证 |
| T2' 基址 | t·tn+local | `T.Parallel(bm)` + TIR 算术 `T.cast(t*tile_n,"float32") + local[i,0]` | 纯算术（bm=1 单元素） | Developer | 探针 P2 |
| S6 出栈 | fp32→int64 | `T.Parallel(bm)` + `T.cast(first[i,0], "int64")`；`T.copy(out_ub[0:real_m], out[start:start+real_m])` | bm 元素 Parallel cast（单发向量 op；基础阶梯 ≤8 / 扩展阶梯至 2048）+ slice 写 | Developer | logsumexp single L108-116 |
| Grid | persistent | `T.Kernel(num_kernels, is_npu=True)` + `T.serial(静态上界)` + `if start < M` | 一维 | Developer | `vcast.md` §2.4 示例（guard 先例）；VP-2026-0044 |

**dtype 路径矩阵**（S2/S3/S4 按输入 dtype 工厂期分派；TVMScript 禁止条件定义变量 → 每路径独立 prim_func，工厂选择——探针实证的组织方式）：

| 输入 dtype | 极值/比较域 | 预转 pass | 候选域 | B/elem（bm≥2 / bm=1 含 ext_brc） |
|-----------|------------|----------|--------|------------------|
| float16 | fp16 原域 | 无（E4） | fp32 | 8 / 10 |
| float32 | fp32 原域 | 无 | fp32 | 12 / 16 |
| bfloat16 | fp32（vcast 预转） | vcast rint | fp32 | 10 / 14 |

### 3.3 计算伪代码

TIR 原语函数签名中 dtype 直接作第二个位置参数（模板 §3.3 规则：不用 `dtype=` 关键字，避免 false-alarm）。

```python
# 工厂期（Python，全部静态量）：block_m 阶梯、tile_n、num_full/tail、num_kernels、serial 上界
def _argreduce_kernel(M, N, op_kind, dtype):          # 接口语义参数不变（harness 契约）
    # 两段式 bm 解析（v1 重写：v0 的 B 表单行下标写法不可执行且 B↔bm 循环依赖；
    # 与 §5.2 同序求解——先 B_multi 解基础/扩展阶梯，无解回退 B_bm1，仍无解转 tiled）
    B_multi = B_elem(dtype, bm_class="multi")         # §4.5 表 bm≥2 列：fp16 8 / fp32 12 / bf16 10
    block_m = max([p for p in (1, 2, 4, 8)
                   if p * N * B_multi <= 65536] or [None])    # 基础阶梯（源码/家族先例）
    if block_m is not None and ceildiv(M, block_m) > 96:      # 发射项驱动 → 扩展阶梯（§5.2，v1 新增）
        block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024, 2048)
                       if p * N * B_multi <= 65536] + [block_m])
    if block_m is None and N * B_elem(dtype, bm_class="bm1") <= 65536:
        block_m = 1                                   # bm=1 驻留（含 ext_brc 预算列：fp16 10/fp32 16/bf16 14）
    if block_m is None:                               # 仍无解 → 路径 T（tiled online）
        block_m = 1                                   # multitile 陷阱强制（§9.2-R3）
        tile_n = pick_tile_n(N, dtype)                # §5.2：compute_tile_n（字节口径）+ 整除/裕度后置校正
    assert N <= 2 ** 24                               # E1/E3 fp32 索引精确域前提（v1 新增，超限 raise）
    # ── 路径 S（resident）伪代码（fp16 形态；bf16/fp32 同构换域）──
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(bm):                                     # K9: threads 已去除
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub   = T.alloc_shared((bm, N), dtype)        # GM staging
                x_frag = T.alloc_fragment((bm, N), dtype)      # 计算缓冲（竞态规避）
                m      = T.alloc_fragment((bm, 1), dtype)
                ext    = T.alloc_fragment((bm, N), dtype)      # 仅 bm=1 使用（独立 prim_func 分派）
                cand   = T.alloc_fragment((bm, N), "float32")
                first  = T.alloc_fragment((bm, 1), "float32")
                out_ub = T.alloc_shared((bm,), "int64")
                for s in T.serial(num_local_tasks):            # 静态上界 + grid-stride
                    start = (cid + s * num_kernels) * bm
                    if start < M:                              # 动态 guard（静态化边界内）
                        real_m = T.min(bm, M - start)
                        T.copy(x[start:start+real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.copy(x_ub, x_frag)
                        T.reduce_max(x_frag, m, dim=1)         # argmin: reduce_min
                        # bm=1: T.vbrc(m, ext)；bm>=2: 融合循环直接 m[i,0]
                        for i, j in T.Parallel(bm, N):
                            cand[i, j] = T.if_then_else(
                                x_frag[i, j] == m[i, 0],        # bm=1 形态: ext[i, j]
                                T.cast(j, "float32"), T.float32(BIG))
                        T.reduce_min(cand, first, dim=1)
                        for i in T.Parallel(bm):
                            out_ub[i] = T.cast(first[i, 0], "int64")
                        T.copy(out_ub[0:real_m], out[start:start+real_m])
        return main
    return _func
# 路径 T（tiled online）：tile-0 直写 running → T.serial(num_full-1) 全 tile → 静态尾 tile
# → 出栈（§3.1 T1–T4；完整形态见 history_version/design_probe_argmax.py P2 kernel，探针已端到端验证）
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.reduce_max/min(dim=1)` | `docs/Tilelang.language/规约操作/T.reduce_max.md`、`T.reduce.md`；`testing/npuir/reduction_ops/test_reduce.py`、`test_tilelang_language_clamp.py` | ✅ 文档+测试+探针（同 dtype 约束为探针新增实证） |
| `T.if_then_else` 元素级于 `T.Parallel` | `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py`、`test_if_then_else_buffer_cond.py`（通过测试） | ✅ 测试+探针 P1/P2 端到端 |
| `T.vbrc`（张量广播 (bm,1)→(bm,N)） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✅ 文档+探针 P2 |
| `T.vcmp("gt")` / `T.vselect`（含原位） | `docs/Tilelang.language/比较操作/T.vcmp.md`、`条件操作/T.vselect.md` | ✅ 文档+探针 P2/P2b（(bm,1) 原位） |
| `T.vcast(bf16→f32, rint)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ 文档（f32→f32 禁用为前次任务 CONST 实证） |
| `T.arange`（备选索引链） | `docs/Tilelang.language/创建操作/T.arange.md`；ssd_chunk_scan L352 `T.arange(idx_j,[0,1],0)` | ✅ 先例（主选融合式不需要） |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/条件操作/T.vselect.md` §2.4 / `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例；logsumexp 家族 | ✅ 先例+探针 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | 48 = `get_aicore_num()×2` 实查（§5.5） |
| bf16 不被 reduce 支持 | Yes | vcast bf16→fp32 预转（`T.reduce.md` §2.2.1 dtype 表） |
| reduce 混合 dtype 静默错误 | Yes（探针新发现） | 同 dtype reduce + 同 dtype 比较（§3.2 约束注） |
| (·,1) 不变操作数 Parallel 条件 bug（bm=1） | Yes（探针新发现） | vbrc 同形化（§1.6.2-#6） |
| shared 多消费者竞态（auto-multi-buffer 预算超限） | Yes（探针新发现） | fragment 计算形态 + 64KB/block 手工预算（§9.2-R4） |
| multitile bm>1 编译器 SIGSEGV | Yes（tiled 路径） | tiled 强制 bm=1（§9.2-R3） |
| vcast f32→f32 数据破坏 | fp32 路径潜在 | 全部用同 dtype `T.copy`（§3.2 注） |
| L0C / 分形 / GEMM 非整除 | No | 无 Cube 计算 |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(n, is_npu=True)` | threads 去除（K9）；persistent + serial（§0.6-R2） |
| SMEM tiling | `alloc_shared` 48KB 预算 + 256 对齐 | UB 192KB + fragment（auto-multi-buffer 膨胀 1.6–2.0×） | §5.2 预算公式（探针定标） |
| 串行首现扫描 | `T.Serial` + `T.loop_break` | 向量化候选 + reduce_min（E1） | §0.6-R1 |
| host pad | Op 层 F.pad 256 对齐 | 原始 N + 静态尾 tile（主选 Op 翻转） | §0.6-R3 |
| jit 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | logsumexp 同款 |
| 精度域 | 全量 fp32 上抛 | fp16 原域（E4）/ bf16 fp32（API）/ fp32 原域 | §1.6.1-E4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（+DESIGN.md） | 高度相似（同族 row-reduction、Developer、reduce dim=1、尾块 T.min + 双 slice） | Kernel 结构、JIT 形态、尾块处理、L0 分层测试 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（+DESIGN.md） | 高度相似（N>UB 流式、T.serial tile 循环、在线递推、(bm,1) fragment 更新链、**尾 tile kernel 侧掩码 segfault 反例 → 本设计静态尾 tile 特化的依据**） | tile 循环结构、在线递推形态、vbrc+vadd 复制技巧 |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` L348-355 | 中（`T.arange(buf,[0,1],0)` 列索引生成先例） | 索引缓冲备选链 |
| `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py` | 中（元素级条件 if_then_else 向量化正例） | 融合候选构造可行性 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 高（`compute_tile_n`/`ub_slab_units`/`device_smem_budget`/`UB_SAFETY_RESERVE_BYTES` NPU 工具） | tile_n 选择与 UB 预算复用 |
| 前次任务残留 `examples/argmax/_argreduce_kernel/repro/__pycache__`、`examples/argmax/_argreduce_kernel/perf_opt/__pycache__`、`examples/argmax/_argreduce_kernel/perf_opt/logs/` | 直接相关（同算子前次 Stage 3/4 的陷阱与性能基线） | TRAP-multitile-bm-gt1-segfault、CONST-vcast-f32-to-f32-corrupt、性能校准数字（§1.6.0/§9） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)`（主选契约；回退契约 `(M, N_padded)`，N_padded=align_up(N,256)，只读前 N 列） | float16 / bfloat16 / float32 | 2D 连续行主序；M、N 为工厂期常量（无动态轴） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | int64 | 首现极值索引；`out_idx=[1]` 由 jit 分配返回 |

### 4.3 中间缓冲区（路径 S，fp16 形态；bf16/fp32 见 §3.2 矩阵）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | (bm, N) | 输入 dtype | UB（shared） | GM staging（唯一 shared 数据缓冲） |
| `x_frag` | (bm, N) | 输入 dtype | fragment | 计算源（竞态规避：多消费者全走 fragment） |
| `x_work`（仅 bf16） | (bm, N) | float32 | fragment | bf16→fp32 精确域 |
| `m` | (bm, 1) | 计算域 dtype | fragment | 行极值 |
| `ext_brc`（仅 bm=1） | (bm, N) | 计算域 dtype | fragment | 极值同形广播（编译器 bug 规避） |
| `cand` | (bm, N) | float32 | fragment | 掩码索引候选 |
| `first` | (bm, 1) | float32 | fragment | 首现索引 |
| `out_ub` | (bm,) | int64 | UB（shared） | 输出 staging |

路径 T 追加（全部 fragment，(bm,1) 类微缓冲）：`running_max/running_idx/chunk_max/chunk_first/chunk_global/cond_gt` + 尾 tile 独立静态宽度组 `x_tail/xw_tail/tail_brc/cand_tail/tail_max/tail_first/tail_global/cond_tail`（尾 tile 的 `x_tail` 为 shared staging，其余 fragment）。

### 4.4 内存搬运路径

```
路径 S（纯 Vector 单引擎）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub] --T.copy--> FRAG[x_frag]
  --T.reduce_max/min(dim=1)--> FRAG[m]
  --[bm=1: T.vbrc]--> FRAG[ext_brc]
  --T.Parallel 融合--> FRAG[cand] --T.reduce_min(dim=1)--> FRAG[first]
  --T.Parallel(bm) cast--> UB[out_ub] --T.copy(slice)--> GM[out]
无 L1/L0 参与；无 GM 中间往返；无 workspace。

路径 T：GM[x tile] --T.copy--> UB[x_ub] --T.copy/vcast--> FRAG --(chunk 链)--> FRAG[running_*]
（逐 tile 循环；running 状态全程片上驻留）
```

### 4.5 UB 内存预算（探针定标模型）

**预算公式**：`手工字节预算(block) ≤ 64KB`（竞态安全实测值；×2 auto-multi-buffer 双缓冲 ≈ 128KB ≤ 192KB−8KB reserve）。依据链：UB=192KB（CONST-capacity-910B2C，1572864 bits 报错文本反推同源）；auto-multi-buffer 膨胀实测两个数据点（v1 标注各自 buffer 清单，防误读为同配置两系数）：**1.60× = 含未用 ext_brc 的首版清单**（10B/elem：手工 160KB → 报错 requires 2097408 bits = 256.03KB）、**2.0× = 净版清单**（8B/elem：手工 128KB → 同 256KB 报错值）——两者是**不同 buffer 清单的配置**（正是 C-3「未用 alloc 不被 DCE」的成因），预算纪律仍按 ×2 保守取值；**预算超限不报错而是静默放弃双缓冲 → 多消费者 shared 缓冲数据竞争**（fp32/bf16 bm=2@4096 实测 15–25 行非确定 BIG 行；fragment 化 + 64KB 预算后 3/3 稳定）。

**B/elem 表**（fragment 形态，bm≥2 / bm=1（含 ext_brc））：

| dtype | x_ub | x_frag(+work) | cand | bm≥2 合计 | bm=1 合计 |
|-------|------|--------------|------|-----------|-----------|
| fp16 | 2 | 2 | 4 | **8** | 10 |
| fp32 | 4 | 4 | 4 | **12** | 16 |
| bf16 | 2 | 4 | 4 | **10** | 14 |

**逐 workload 预算验证**（驻留路径； tiled 见 §5.2）：

| 配置 | 手工预算 | ×2 双缓冲 | 结论（≤184KB 可用） | 探针证据 |
|------|---------|-----------|--------------------|----------|
| fp16 bm=2 @ N=4096 | 64KB | 128KB | ✅ | p3a 3/3 PASS |
| fp16 bm=4 @ N=4096 | 128KB | 256KB | ❌ 编译拒绝（requires 2097408 bits） | p3a 首版实测（净版清单 8B/elem → 2.0×；含未用 ext_brc 首版为 160KB → 1.60×，同报错值——两清单口径见上，v1 标注） |
| fp32 bm=1 @ N=4096 | 64KB | 128KB | ✅ | p3d 3/3 PASS |
| bf16 bm=1 @ N=4096 | 56KB | 112KB | ✅ | p3e 3/3 PASS |
| fp32 bm=2 @ N=4096（含未用 ext_brc） | 128KB | 256KB | ❌ 竞态（静默） | p3d 首版实测 15–25 行非确定 |
| bf16 bm=2 @ N=4096（含未用 ext_brc） | 112KB | 224KB | ❌ 竞态 | p3c 首版实测 |

> **实现纪律**（探针派生，Stage 3 必须遵守）：① 未用 fragment alloc **不会被 DCE**（计入 UB 规划）→ bm=1/bm≥2、bf16/非 bf16 必须独立 prim_func 分派，禁止「无条件 alloc + 条件使用」；② 计算一律读 fragment，shared 仅 GM staging；③ 预算按上表 B/elem 计算，宁小勿大（bm 降档优先于预算贴线）。

### 4.6 动态轴定义

无（M、N、dtype、op_kind 均工厂期静态；kernel 内全部循环边界静态，运行时仅 pid/task guard）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,) int64
def _func(block_m): ...                       # K9: threads 去除；block_m 由 wrapper config 传入
```

工厂缓存：沿用 harness `ArgreduceKernel` 的 kernel cache（(M,N) 键）+ 源码 `lru_cache` 语义（§0.4-#7 保留）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为逐元素/规约/比较选择（reduce_max/min、if_then_else 融合、reduce_min、vcmp/vselect），无 matmul；数据通路 GM↔UB↔fragment；无 Cube/L0。

### 5.2 Block 划分（block_m 阶梯 + 路径切换 + tile_n）

```python
# 工厂期（全部静态 Python 量）
UB_MANUAL_BUDGET = 65536                             # 字节（§4.5 竞态安全手工预算）
B_multi = B_elem(dtype, bm_class="multi")            # §4.5 表 bm≥2 列：fp16 8 / fp32 12 / bf16 10
B_bm1   = B_elem(dtype, bm_class="bm1")              # §4.5 表 bm=1 列（含 ext_brc）：fp16 10 / fp32 16 / bf16 14
# 路径切换（两段式，与 §3.3 同序）：驻留阈值 = 64KB 手工预算
block_m = max([p for p in (1, 2, 4, 8)
               if p * N * B_multi <= UB_MANUAL_BUDGET] or [None])   # 基础阶梯（源码/家族先例）
# 扩展阶梯（v1 新增，发射项驱动——REVIEW 阻塞 1）：基础阶梯下 block 数 > 48×2（两波）
# 时启用 {16..2048} 档（小 N 大行数负载；档位由预算过滤器自动封顶 = 64KB//(N·B_multi)）。
# 依据：3d 负载 (524288,4) fp16 预算允许 bm≤2048，若停在 bm=8 → 65536 block、
# 发射项 ≈4.8ms（劣于前次 pad 契约 ~200µs 约 24×，§1.6.0 反事实锚点）；
# 基础阶梯 {1,2,4,8} 承自 GPU 48KB SMEM 启发式，对小 N 大 M 严重不足。
if block_m is not None and ceildiv(M, block_m) > 96:
    block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024, 2048)
                   if p * N * B_multi <= UB_MANUAL_BUDGET] + [block_m])
if block_m is None and N * B_bm1 <= UB_MANUAL_BUDGET:   # bm≥2 无解 → bm=1（含 ext_brc 预算）试驻留
    block_m = 1
if block_m is None:                                  # 仍无解 → 路径 T（tiled online）
    block_m = 1                                      # multitile 陷阱强制（§9.2-R3）
    # compute_tile_n（仓内原语 _primitives.py L110）：budget 单位为【字节】、num_buffers 为
    # 同形 (bm,tile_n) 大缓冲折算数（ub_slab_units——v1 更正：v0 把 budget 折成元素数后又传
    # elem_bytes/num_buffers，二次折算导致 tn=512/4608 与表值差 10–16×）。
    # slab 按 §4.3 路径 T buffer 清单：fp16 3+1→5（x_ub/x_frag/max_brc + cand）、
    # bf16 1+3→7（x_ub + x_work/max_brc/cand）、fp32 4+0→4
    slab = ub_slab_units(elem_bytes, dtype_slabs, fp32_slabs)
    tile_n_cap = compute_tile_n(block_m=1, elem_bytes=elem_bytes, N_padded=N,
                                alignment=256, budget=UB_MANUAL_BUDGET,
                                num_buffers=slab)
    # wrapper 侧整除优先 + 裕度后置校正（v1 新增；不直接依赖原语除数接受判据——
    # 该判据仅在「除数不增加 tile 数」时生效，无法表达宁小勿大裕度纪律）：
    #   在候选集（N 的因子 ∩ 256 倍数 ∩ ≤ tile_n_cap ∩ slab·elem·tn ≤ 0.9·UB_MANUAL_BUDGET）
    #   中取最大者（零尾 tile + ≥10% 预算裕度）；候选集为空回退 tile_n_cap + 静态尾 tile
    tile_n = pick_divisor_with_margin(N, tile_n_cap, slab, elem_bytes,
                                      UB_MANUAL_BUDGET, margin=0.9)
    # 非整除残余 tail = N - num_full·tile_n 走静态尾 tile 特化（§5.4）
num_row_blocks = ceildiv(M, block_m)
assert N <= 2 ** 24    # v1 新增（S2）：E1/E3 fp32 索引精确域前提（§1.6.1）；超限 raise
```

**lm-head tile_n 推导链（冻结值复现，v1 显式落盘供复算）**：

| dtype | slab | compute_tile_n 返回（tile_n_cap） | 后置校正（整除因子 ∩ ≤cap ∩ ≤0.9×64KB 预算） | 冻结 tile_n / tiles / 手工预算 |
|-------|------|----------------------------------|---------------------------------------------|-------------------------------|
| fp16 | 5 | 6400（65536//10=6553→align_down 256→6400；6400 整除 102400 且 16 tile ≤ max_tiles → 原语接受除数返回 6400） | 6400 预算 64000B > 0.9×65536=58982B ✗ → 次大因子 5120（51200B ✓，20 tile 零尾） | **5120 / 20 / 50KB（51200B）** |
| bf16 | 7 | 4608（65536//14=4681→4608；最大整除因子 4096 给 25 tile > max_tiles 23 → 原语拒绝除数，返回 4608 + 1024 列尾 tile） | 4096 预算 57344B ≤ 58982B ✓（25 tile 零尾，消除原语返回值的 1024 列尾 tile） | **4096 / 25 / 56KB（57344B）** |

**逐 workload 参数表**（manifest 负载；M/N 由 manifest + `_prepare_input` 单维路径推导，v1 更正 3d 行）：

| workload | dtype | 路径 | block_m | tile_n | tiles | 手工预算 | 探针/前次证据 |
|----------|-------|------|---------|--------|-------|---------|---------------|
| smoke (32,256) | fp16/bf16/fp32 | S | 8 | — | — | 16/20/24KB | 预算宽裕（blocks=4 ≤ 96，扩展阶梯不触发） |
| lm-head (4,102400) | fp16 | T | 1 | **5120**（推导链见上表，零尾 tile） | 20 | 50KB（51200B） | 前次任务同负载 25 tile（tn=4096）实测 32.5µs 校准 |
| lm-head (4,102400) | bf16 | T | 1 | **4096**（推导链见上表，零尾 tile） | 25 | 56KB（57344B） | 同上（bf16） |
| hidden-state (2048,4096) | fp16 | S | **2** | — | — | 64KB | p3a 3/3 PASS；前次任务终选同 bm=2（47µs）；扩展档均超预算不触发 |
| hidden-state (2048,4096) | bf16 | S | **1** | — | — | 56KB | p3e 3/3 PASS；前次任务终选同 bm=1（55µs） |
| hidden-state (2048,4096) | fp32 | S | **1** | — | — | 64KB | p3d 3/3 PASS |
| 3d-non-last-axis → (**524288, 4**)（原始 N 契约；dim=0：N=4、M=128×4096，v1 更正） | fp16 | S | **2048**（扩展阶梯；预算上限 65536/(4×8)=2048） | — | — | 64KB | 原始契约总流量 8.4MB（读 4.19 + 写 4.19MB；前次 pad 契约 268MB 读 + 4.19MB 写 ≈ 269MB，总流量 ~32×——§0.6-R3 的 N 维 64× 为读侧口径）；(2048,4) 窄内维未经探针覆盖 → §9.2-R10 + L0-6 先行验证 |

**block_m 选择理由**：阶梯内取预算最大值——行块越大每 block 摊薄发射越优（CONST-vector-launch-overhead），受 64KB 竞态安全预算硬约束（§4.5）。基础阶梯 {1,2,4,8}（源码与家族先例）覆盖 N ≥ 512 类负载（hidden-state N=4096 fp16 → bm=2 即达 64KB 上限，扩展档全部超预算、不改变结果）；**小 N 大行数负载**由发射项门控（blocks > 96）启用扩展阶梯至 2048（档位 = 预算过滤器自动封顶；N ∈ (256, 1024) 的中等宽度负载基础阶梯已接近预算上限、扩展收益有限，如 Stage 4 遇该类大 M 负载可下调发射门控——发射项公式同 §1.6.0）。3d 负载选定 bm=2048 的依据：预算最大值规则（2048×4×8B = 64KB，与已验证配置 fp16 bm=2@4096 同为 64KB 手工预算/128KB 双缓冲包络）+ 发射项最优（256 block → 18.7µs；bm=1024 → 37µs、bm=512 → 75µs 仍可作 R10 异常时的降档备选）。

### 5.3 约束分析

- **对齐约束**：尾轴 32B 对齐由 `T.copy` 切片天然满足（logsumexp N=300 先例 + 探针 P1 直证）；N 无需 256 对齐（host pad 消除后契约即原始 N）。
- **UB 容量**：§4.5 预算表逐配置验证（✅/❌ 双向实测）。
- **L0/L1 容量**：不适用（纯 Vector）。
- **编译器陷阱约束**：tiled 路径 bm=1 强制（TRAP-multitile-bm-gt1-segfault）；bm=1 的 (·,1) 广播 bug 规避（§9.2-R5）。

### 5.4 注意事项（非整除/边界处理策略）

- **M 非整除（M % bm ≠ 0）**：`real_m = T.min(bm, M − start)` + src/dst 双显式 slice T.copy + 垃圾行不写出（VP-2026-0044 三件套；logsumexp/探针 P1 的 9%4=1 用例验证）。
- **N 非整除（驻留路径）**：整行驻留天然覆盖（无 N 分块）。
- **tile_n 非整除（tiled 路径）**：compute_tile_n 除数优先；残余 `tail` 走静态尾 tile 代码路径（宽度为工厂期常量；**不用 kernel 侧 if_then_else 掩码**——logsumexp tiled 的 serial 变量条件 segfault 反例）。
- **M=1**（dim=None 全张量负载）：bm=1、单 block、驻留路径（N=numel ≤ 预算时）或 tiled。
- **N=1**：驻留；小 N 扩展阶梯生效（大 M 时 bm 至 2048，发射门控同 §5.2）；候选唯一命中 j=0（verify_equiv (1,1)/(1,2) 用例）。
- **M=0**：grid=0 边界（§9.2-R6 风险记录，harness 测试不覆盖）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数（实查）**：`from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **24**（2026-09-24 03:22 实查，Ascend910B2C，与 CONST-aicore-910B2C 一致）。本算子**纯 Vector → 核数翻倍 = 24 × 2 = 48**。查询代码与返回值记录于本节（禁止文档假设值替代——本节即实查记录）。
- **逻辑核数**：`num_row_blocks = ceil(M / block_m)`（纯行并行，无 N 向分核；tiled 路径 bm=1 → num_row_blocks = M）。
- **规模判定与分核方案**（persistent 统一结构：`num_kernels = min(num_row_blocks, 48)`；核内 `T.serial(静态上界 = ceildiv(num_row_blocks, num_kernels))` + grid-stride 任务映射 `task = cid + s·num_kernels` + 动态 guard `if start < M`）：

| workload | num_row_blocks | 判定 | num_kernels | 核内任务数（静态上界） | 依据 |
|----------|---------------|------|-------------|----------------------|------|
| smoke (32,256) | 4（bm=8） | **逻辑 ≤ 物理（4 ≤ 48）** | 4 | 1 | 无需适配：行块独立、每核 ≤1 任务、无串行调度（单波） |
| lm-head fp16 | 4（bm=1, tiled） | 逻辑 ≤ 物理（4 ≤ 48） | 4 | 1 | 同上；核利用率 4/48——C6 N-split 为 Stage 4 候选（§1.6.3-#4），主选单 kernel 以合并正确性面最小化（前次 round8_op9 反例） |
| lm-head bf16 | 4 | 同上 | 4 | 1 | 同上 |
| hidden-state fp16 (2048,4096) | 1024（bm=2） | **极大规模（1024 ≫ 48，bm 受 UB 预算不可再增）** | **48** | **22**（ceildiv(1024,48)） | 固定 48 核 + 核内 T.serial 串行（静态边界 22 + guard）；消核启动开销与串行调度（VP-2026-0044 persistent 形态；ada Stage 4 +2.14x 实证） |
| hidden-state bf16 | 2048（bm=1） | 极大规模 | 48 | 43 | 同上 |
| hidden-state fp32 | 2048（bm=1） | 极大规模 | 48 | 43 | 同上 |
| 3d-non-last-axis (**524288,4**) | 256（bm=**2048**，扩展阶梯，v1 更正） | **极大规模（256 ≫ 48）** | **48** | **6**（ceildiv(256,48)=6；16 核 ×6 + 32 核 ×5，负载不均 6:5） | 固定 48 核 + 核内 T.serial（静态边界 6 + guard）；流量下界 ≈5.8µs、发射项 ≈18.7µs（§1.6.0）；**反事实锚点**：bm=8 → 65536 block、核内 1366 任务 → 发射 ≈4.8ms（劣于前次 pad 契约 ~200µs 约 24×）——扩展阶梯的存在依据（§5.2）；(2048,4) 窄内维风险 §9.2-R10，Stage 4 A/B bm∈{512,1024,2048}（§6.3-⑤） |

> 核内串行循环边界全部为**静态值**（工厂期 Python 整数：ceildiv(num_row_blocks, num_kernels)）；动态性仅存在于 guard（`if start < M`，vcast.md §2.4 先例形态）。分核参数（block_m/tile_n）选择公式见 §5.2；wrapper `default_config` 按 §5.2 公式适配（其 GPU 48KB SMEM 启发式弃用）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；消核启动开销 |
| N 方向（tiled 路径） | tile 迭代 | `T.serial(num_full − 1)`（静态；tile-0 剥离初始化） | 在线递推的跨 tile 顺序依赖（E3）；静态边界要求 |
| 元素级（候选生成） | 向量化 | `T.Parallel(bm, N)` / `T.Parallel(bm, tile_n)` / `T.Parallel(bm, tail)` | 融合 if_then_else 单 pass（§1.6.2-#1） |
| 元素级（极值/首现） | 向量化归约 | `T.reduce_max/min(dim=1)`、`T.reduce_min(dim=1)` | 原语（§3.2） |
| 出栈 | Parallel 向量（bm 元素，单发） | `T.Parallel(bm)` + `T.cast` | 形状收缩 (bm,1)→(bm,)，无 cast+收缩复合向量 API（§1.6.2-#9；bm 随阶梯 ≤8/至 2048，恒单发向量 op） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # 路径 S：单层任务循环
    for s in T.serial(num_local_tasks):              # 静态上界
        start = (cid + s * num_kernels) * block_m
        if start < M:                                # 动态 guard
            [§3.3 路径 S 主体：copy→frag→reduce→融合候选→reduce_min→出栈]
    # 路径 T：任务循环内嵌 tile 循环
    #   tile-0 直写初始化（无循环）
    #   for t in T.serial(num_full - 1): [chunk 链 + 严格大于更新]
    #   静态尾 tile（无循环，独立代码路径）
    #   出栈
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**（auto-multi-buffer 由 Developer 模式自动施加，探针已验证其在任务/tile 循环上的正确性与竞态边界——§9.2-R4 纪律）。Stage 4 候选（按前次任务 round3 "swp" 系列与本案发射项分析排序；v1 更新 ③⑤）：① `--enable-auto-multi-buffer=false` 显式分配结构（ssd 数据点：膨胀 1.12× → bm 可升档，需重验竞态）；② tiled 路径 tile 循环软件流水（tn/bm 扫描）；③ C6 N-split partial+merge（§1.6.3-#4；**备选完整结构 + 判定阈值〔端到端 < 0.8× 主选则翻转〕+ 回写路径见 §1.6.4**——前次 round9 终版实测 partial ≈15.7µs + merge ≈1.7µs ≈ 0.54× 主选）；④ 融合候选 value 分支内联全局索引 `t·tn+j`（消 base-add 微操作，需探针验证 serial 变量于 value 分支的 codegen）；⑤ **扩展阶梯 bm 扫描**（v1 新增，3d 类负载 bm ∈ {512, 1024, 2048} A/B——主选冻结 bm=2048〔预算最大值〕，为 §9.2-R10 窄内维风险的实证裁决；发射项理论序 19/37/75µs）。

### 6.4 尾块处理

见 §5.4（M 尾块：real_m + 双 slice + 垃圾行丢弃；N 尾 tile：静态宽度特化；guard 静态化边界）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输）。persistent 任务循环与 tile 循环内的生产者-消费者依赖由编译器自动插入（探针 P1/P2/P2b 在 1024 block / 96 block 多波次下的正确性为该自动同步的实证——前提为 §4.5 的 fragment 纪律）。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer` 环境变量，logsumexp 同款）。Stage 4 探索项：`--enable-auto-multi-buffer=false`（§6.3-①）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 §0.1 语义为唯一依据（torch.argmax，first-occurrence + int64 + torch NaN 语义为文档化边界），**不复刻 §0.6 NPU 算法**（无掩码候选/reduce_min/在线递推结构）——保证验证独立性。

```python
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmax") -> torch.Tensor:
    """PyTorch 参考实现（CPU）：torch.argmax/argmin first-occurrence 语义。
    输入任意 rank；与 harness ArgmaxFwdOp 的 dim/keepdim 语义对齐由 Op 层测试覆盖，
    kernel 级 golden 只承接 (M, N) → (M,) 契约。"""
    return x.argmax(dim=-1) if op_kind == "argmax" else x.argmin(dim=-1)
```

（harness pytest 侧 ref：`x.argmax(dim=...)` on device——§9.2-R2 记录其 ±0.0 设备实现差异对 randn 测试无影响。）

### 8.2 精度标准

**精确比对**（索引语义，无 atol/rtol 容差）：`torch.equal(out, ref)` 且 `out.dtype == torch.int64`（与 harness `test_argmax.py::_exact_compare` 同款）。数学依据：E1–E4 为索引恒等（§1.6.1 机器验证 0 偏差），任何容差放松都会掩盖 tie-break/哨兵类 bug。

**L0 门槛测试计划**（gate 用例；完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 按本节展开）：

| # | 用例 | 覆盖点 |
|---|------|--------|
| L0-1 | (32,256) fp16 / bf16 / fp32，bm=8 | smoke 负载 ×3 dtype |
| L0-2 | (4,102400) fp16，tiled tn=5120（§5.2 推导链冻结值，v1） | lm-head 负载 + tiled 路径 + 多 tile 在线递推 |
| L0-3 | (4,102400) bf16，tiled tn=4096（§5.2 推导链冻结值，v1） | bf16 tiled（vcast 预转 + 整除 tile_n） |
| L0-4 | (2048,4096) fp16 bm=2 | hidden-state + persistent 48×22 任务循环（多波次竞态回归） |
| L0-5 | (2048,4096) bf16 bm=1 / fp32 bm=1 | 预算边界 dtype 路径 |
| L0-6 | (4096,4) fp16 bm=2048 快速回归 + (**524288,4**) fp16 bm=2048（manifest 真 shape，原始 N 契约；v1 更正——v0 误记 (512,4)） | 3d-non-last-axis（dim=0 → Op 层 (M,N)，M=128×4096=524288）+ 小 N 大行数 + 扩展阶梯 + (bm,4) 窄内维形态（§9.2-R10 的先行验证：先用 (4096,4) 快速暴露编译/正确性问题，再跑真 shape 验证发射项/分核） |
| L0-7 | (128,300) fp16/bf16 + (129,512) fp16 | N 非对齐（300）+ M 尾块（129%bm）——harness basic 用例 |
| L0-8 | (1, N) 与 (N,) flatten（dim=None → M=1） | 单行负载 + M=1 驻留/tiled 分界 |
| L0-9 | 构造行：重复最大值（跨 tile tie）/ 全 −inf / 全 +inf / ±inf 混合 / 亚正规最大 / N=1、N=2 | 首现 tie-break + 边界值（verify_equiv 角点集的 NPU 侧复验） |
| L0-10 | argmin kind 全量镜像（(32,256)×3 dtype + (4,102400) fp16） | 接口双 kind 契约（E2 直译路径） |

L2（记录不阻断）：NaN 行为（§9.2-R1，断言**记录值** = 哨兵 2^30 而非崩溃）、混合符号零行（记录 vs torch-CPU/NPU 两基准）、M=0、(1,1) 极小、tile_n+1 恰跨分界、`N > 2^24` 工厂断言行为（构造 numel 超限张量验证 raise 而非静默失精，§5.2/S2）。

### 8.3 等价性验证工件

`verify_equiv.py`（随算子目录交付，已执行 ALL EQUIV_PASS，§1.6.1 结果表）——Stage 2 检视维度 8 重跑复核对象。

---

## 9. 风险点与注意事项

### 9.1 已知约束（工具链级，探针/前次任务实证）

| # | 约束 | 实证来源 | 设计内规避 |
|---|------|---------|-----------|
| C-1 | reduce 混合 dtype（fp16 src→fp32 dst）**静默产出错误值**（文档 §2.4 的 dtype≠accum 示例不成立） | 探针 dbg 变体 B（extreme 值错乱） | §3.2 强制同 dtype reduce + 同 dtype 比较 |
| C-2 | bm=1 时 Parallel 融合循环条件中的 (·,1) **不变操作数**被误 lower 为 `f16→i1` 广播（MLIR verify fail） | 探针 P2 首版（`arith.fptoui(f16→i1)` IR 定位） | §1.6.2-#6：bm=1 用 `T.vbrc` 同形化（独立 prim_func 分派） |
| C-3 | **未用 fragment alloc 不被 DCE**（计入 UB 规划、挤占双缓冲预算） | 探针 p3c/p3d（fp32/bf16 bm=2 含未用 ext_brc → 竞态） | §4.5 纪律①：bm/bf16 类独立 prim_func |
| C-4 | **shared 多消费者缓冲在 auto-multi-buffer 预算超限时静默竞态**（非确定 BIG 行；预算内则正常） | 探针 p3c/p3d 首版（15–25 行、跨 run 变化）vs fragment 版 3/3 稳定 | §4.5 纪律②：计算全走 fragment + 64KB/block 预算 |
| C-5 | multitile + bm>1 → 编译器 **SIGSEGV**（rc 139，无报错） | 前次任务 repro `TRAP-multitile-bm-gt1-segfault`（触发矩阵：bm=1 任意 tile ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗） | §5.2：tiled 路径强制 bm=1 |
| C-6 | `T.vcast` **f32→f32 非恒等**（identity diff ~0.5） | 前次任务 repro `CONST-vcast-f32-to-f32-corrupt` | §3.2：fp32 路径一律同 dtype `T.copy` |
| C-7 | auto-multi-buffer UB 膨胀 1.6–2.0×（结构相关） | 探针 p3a 两数据点（v1 绑定清单）：含未用 ext_brc 首版 160KB→256.03KB（**1.60×**）与净版 128KB→256KB（**2.0×**）——同报错值、不同 buffer 清单（C-3 成因），勿按 128KB×1.60 规划 | §4.5 预算模型按 ×2 保守计 |
| C-8 | `T.if_then_else` 条件引用 `T.serial` 循环变量 → select codegen **segfault** | logsumexp tiled 模块 docstring（L24-33） | 融合候选条件只含 buffer 元素；tile 基址走 value 算术（§1.6.2-#8） |

### 9.2 风险点

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | **NaN 行为分歧**：NPU reduce 传播 NaN → 匹配不成立 → 输出哨兵 2^30；torch（CPU 与 NPU 设备实现）返回首个 NaN 索引；源 GPU 实现（fmax 语义）返回非 NaN 极值位置——三方各异 | NaN 输入行索引错误（值 2^30） | 语义域文档化（§0.1）；harness 测试全 randn（无 NaN）不受影响；L2 记录行为；如需对齐 torch 可 Stage 4 加 `vcmp(x,x,"ne")` NaN 预检 pass（成本 +1 全宽 pass，暂不做） |
| R2 | **±0.0 混合零行 vs torch-NPU 设备实现分歧**：本 kernel IEEE 相等（与 torch-CPU golden 一致）；torch-NPU argmax 偏好 +0.0（探针实测三 dtype 一致） | 行极值恰为 0 且含混合符号零的行，harness pytest（device ref）可能不等 | randn 输入下概率 0（极值恰为 0 且 ±0 并存需构造）；kernel 级 golden（CPU）一致；L2 记录；不改设计（torch 自身 CPU/NPU 不一致，无单一正确基准） |
| R3 | tiled 路径 bm=1 由编译器陷阱（C-5）强制，行块不可放大 | lm-head 类负载核利用率 4/48 | 主选接受（发射项估算 30–60µs，前次同类实测 32.5µs 达标区间）；C6 N-split 为 Stage 4 备选（**完整结构/判定阈值〔端到端 < 0.8× 主选则该 shape 类翻转〕/回写路径见 §1.6.4**；前次 round9 实测锚点 partial+merge ≈17.4µs ≈ 0.54× 主选——若本轮主选实测维持 ≥30µs，Stage 4 大概率翻转，v1 补） |
| R4 | 竞态规避依赖 fragment 纪律 + 64KB 预算（经验值，非编译器契约） | 新 shape/新配置贴线时可能复现 C-4 | §4.5 双向实测表 + L0-4 多波次回归用例；实现纪律写入 §4.5；Stage 4 探索 auto-multi-buffer=false 结构 |
| R5 | (·,1) 广播 bug（C-2）的规避依赖 bm 分派正确性 | bm=1 路径误用直接索引 → 编译失败（显性，非静默） | 编译期失败即可发现（无静默风险）；L0-8 M=1 用例覆盖 bm=1 路径 |
| R6 | **M=0**（非规约维含 0）：grid=0 边界未验证 | 可能启动异常或空输出 | Op 层短路建议（Stage 3 在 wrapper forward 加 `if M == 0: return torch.empty(0, dtype=int64)`）；harness 测试不覆盖；L2 记录 |
| R7 | harness Op 层翻转（`_kernel_handles_padding=True`）若被冻结 | 回退契约（kernel 声明 N_padded + 读前 N 列）需 Stage 3 换宽度声明；host pad 开销回归（3d 负载 200µs 量级） | §0.2 已给两案完整规格；翻转依据仓库注释明示的演化方向，风险低 |
| R8 | bf16 tiled tn=4096 手工预算 57344B（56KB，slab=7×2B×4096）距 64KB 上限 12.5% 裕度（0.9 裕度规则内，§5.2 推导链——v1 更正预算口径；原语返回 4608+1024 列尾 tile 由后置校正消除） | 若编译器版本行为变化可能溢出/竞态 | tile_n 规则参数化（可降 tn=2048：32768B 预算 50% 裕度）；L0-3 回归 |
| R9 | 前次任务陷阱知识（C-5/C-6）的登记缺失（capability-gaps.md 无 CG-2026-0014 条目，源 repro .py 已清理仅存 pyc） | 知识流失风险（工具链升级后不可追溯） | 本设计 §9.1 全量收录 + RETROSPECTIVE 提交 Value Point（重新登记 + 版本戳重验要求） |
| R10 | **小 N + 大 M 的 (bm,4) 窄内维形态未经探针覆盖**（探针最窄 N=300；3d 负载 (2048,4) fragment 的 reduce dim=1 短行归约与 8B 行宽的向量效率未实证——v1 新增，REVIEW 阻塞 1） | 3d 负载可能编译失败/性能不及发射模型（§1.6.0 ≈19µs 为估算） | L0-6 双用例先行（(4096,4) 快速回归 → 真 shape 验证）；Stage 3 首跑记录实测；异常时扩展阶梯内降档（bm=1024 → 32KB/≈37µs、bm=512 → ≈75µs，仍 ≪ pad 契约 200µs）；Stage 4 A/B bm∈{512,1024,2048}（§6.3-⑤） |
| R11 | **N > 2^24 超界输入**（fp32 索引不可精确表示，E1/E3 等价域外；dim=None 全张量 numel > 16.7M 时触发——v1 新增，S2） | 静默索引失精 | §5.2 工厂期 `assert N <= 2**24`（超限 raise 并提示分块）；int64 候选域为超界扩展路径（`T.reduce.md` §2.2.1 int64 √ 可用，未实现——需要时按 [DESIGN_LIMIT] 路由）；manifest 最大 N=102400 安全；L2 记录断言行为 |

### 9.3 特殊场景处理

非整除分块（§5.4）；极小 shape（(1,1) 驻留 bm=1，verify_equiv 覆盖）；混合精度（per-dtype 路径矩阵 §3.2）；dim=None 全张量（Op 层 flatten → M=1，L0-8）；跨 tile tie（L0-9 构造行 + verify_equiv E3 专项）。

### 9.4 能力缺口登记（Compiler Capability Gap Rule）

本轮设计确认的缺口（Stage 1 登记，`occurrences` 待合入 `.agents/evolution/capability-gaps.md`；**C-5 前次已登记为 CG-2026-0014 但条目缺失，需补录**）：

| 缺口 | 层级 | 证据 |
|------|------|------|
| multitile bm>1 编译器 SIGSEGV（CG-2026-0014 补录） | Frontend/TileLangIR pass（向量化/分块） | 前次 repro pyc 触发矩阵 + 本设计规避 |
| reduce 混合 dtype 静默错误值 | BishengIR（VReduce 类型校验缺失） | 探针 dbg-B（vs `T.reduce_max.md` §2.4 文档示例矛盾——文档或实现需修正） |
| (·,1) 不变操作数 i1 误型广播（bm=1 Parallel 条件） | TileLangIR npu_loop_vectorize | 探针 P2 IR（`arith.fptoui(f16→i1)`） |
| auto-multi-buffer 预算超限静默竞态 | BishengIR PlanMemory/流水 | 探针 p3c/p3d 非确定失配 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/argmax/_argreduce_kernel/
├── DESIGN.md                                    # 本设计文档
├── verify_equiv.py                              # D-1 等价性机器验证（已执行 ALL EQUIV_PASS）
├── _argreduce_kernel.py                         # ⬜ Stage 3：算子实现（kernel + L0 测试）
├── RETROSPECTIVE.md                             # Stage 1 复盘（本次产出）
├── history_version/
│   ├── design_probe_argmax.py                   # D-3 设计期探针（P1/P2/P2b/P3 系列，已执行）
│   └── design_probe_argmax_dbg.py               # D-3 调试探针（混合 dtype 定位，已执行）
├── perf_opt/                                    # Stage 4 工作区（前次任务残留 logs/profiles 保留供校准）
└── repro/                                       # 前次任务 repro pyc 残留（C-5/C-6 证据）
```

harness 侧 Stage 3 落点：NPU kernel 实现写入 harness 集成包 `argmax_kernel/` 目录下的 `_argreduce_kernel.py`（Stage 3 交付目标，届时生成——当前尚未创建），wrapper（`argmax.py` Part B）改 import 指向之 + `default_config` 按 §5.2 公式适配 + Op 层 `_kernel_handles_padding=True` 翻转（§0.2 主选）。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（v1 修订版，2026-09-24） | 本文档（v0 备份于 history_version/design_v0.md；v1 按 Stage 2 REVIEW 3 阻塞 + 7 建议定点修订，见「修订说明」节） |
| `verify_equiv.py` | ✅ 已执行 | 114 行 EQUIV_PASS / 0 FAIL（§1.6.1 表） |
| `history_version/design_probe_argmax.py`（+`_dbg.py`） | ✅ 已执行 | 探针结论内嵌 §1.6.0/§4.5/§9.1 |
| `_argreduce_kernel.py` | ⬜ 待实现（Stage 3） | 按本设计 §3.3/§5/§6 实现 + L0 套件 §8.2 |
| `test__argreduce_kernel.py` | ⬜ 待实现（Stage 3，并入 `_argreduce_kernel.py` L0 段） | logsumexp 同款分层自测形态 |

### 10.3 命名规范

- 项目目录：`examples/argmax/`（project_name=argmax）；算子目录：`_argreduce_kernel/`（op_name，harness 工厂函数名）。
- 实现文件：`_argreduce_kernel.py`（与工厂函数同名的 Developer 模式自包含形态：kernel + golden + L0-L2 自测，logsumexp 先例）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性机器验证（verify_equiv.py，E1–E4 ALL PASS）
3. ✅ 设计期探针（P1/P2/P2b/P3：融合链/在线链/多波次/预算定标）
4. ⬜ 算子实现（`_argreduce_kernel.py`：路径 S/T 双 prim_func 族 + 工厂 + L0 测试）+ golden 精确比对（torch.equal）
5. ⬜ harness 接线（wrapper import + default_config + Op flag 翻转）+ `test_argmax.py` 全量 + `bench_argmax.py`

---

## 附录 A：设计期探针证据链（D-3 记录）

**探针文件**：`history_version/design_probe_argmax.py`（主）+ `design_probe_argmax_dbg.py`（混合 dtype 定位）；执行窗口 2026-09-24 03:50–04:15，工具链 tilelang 0.1.2（dev build）+ CANN 8.5.0 + Ascend910B2C。

| 探针 | 配置 | 结果 | 设计影响 |
|------|------|------|---------|
| P1 | resident 融合链 (9,300) fp16/fp32 bm=4（含全 −inf/tie/±0/尾行） | argmax 精确 PASS（argmin ±0 混合零 1 行分歧 → §9.2-R2） | 路径 S 端到端可行 |
| P2 | tiled 在线链 (3,700) tn=256（跨 tile tie/全 −inf/尾 tile 188 列/垃圾行） | 首版编译失败（C-2 i1 误型）→ vbrc 修复后 PASS | 路径 T 端到端可行 + C-2 发现 |
| P2b | tiled 多波次 (96,8192) tn=4096（96 block/48 核） | 3/3 PASS（fragment 形态） | 竞态安全实证 |
| P3a | (2048,4096) fp16：bm=4 编译拒绝（requires 2097408 bits = 1.60×膨胀）→ bm=2 fragment 形态 3/3 PASS | 预算定标 + C-3/C-4 发现 | §4.5 预算模型 |
| P3c/p3e | bf16：bm=2（含未用 ext_brc）竞态 17–22 行非确定 → bm=1 3/3 PASS | C-3/C-4 定证 | bf16 bm=1 配置 |
| P3d | fp32：bm=2 竞态（15–25 行非确定，3 run 变化）→ bm=1 3/3 PASS | 同上 | fp32 bm=1 配置 |
| dbg-B/C | reduce 混合 dtype 变体 B 极值错乱（0.75/117/13/−152 vs 3.09/2.81/...）；同 dtype 变体 C 精确 PASS | C-1 定证 | 同 dtype 纪律 |
| NaN 补测 | NaN 行 ×4 位形 | kernel=哨兵 2^30；torch CPU/NPU=首 NaN 索引；±inf 行精确 | §9.2-R1 |
| NPUUtils | `get_aicore_num()` → 24（2026-09-24 03:22） | Vector ×2 = 48 | §5.5 实查记录 |

**前次任务工件校准**（`perf_opt/logs/`，2026-09-23）：lm-head fp16 单核 tiled ~32.5µs（round7_op8）；hidden-state fp16 bm=2 ~47µs、bf16 bm=1 ~55µs（merged final）；3d（pad 契约 (524288,256) bm=384）~195–204µs（host pad 262KB 流量主因——本设计原始 N 契约消除）；round8_op9 N-split 首版 4/4 行 golden mismatch（C6 合并正确性风险实证）。

---

## 附录 B：性能反馈附录（[DESIGN_LIMIT] 归档，Stage 4 后追加）

> 本附录由 conductor 在 Stage 4 调优完成后按「TUNING→DESIGN 受控逆向反馈」路由**附录补记**追加（2026-09-24）；只追加、不改动 §0–§10 与附录 A 既有内容。完整证据见 `perf_opt/perf_feedback.md`。

**结论**：lm-head 类「小 M（≤16）大 N（≥32768、N 为 256 倍数）」负载的 §1.6.3-#4 单 kernel 主选存在设计层结构性天花板——§9.1 C-5（multitile bm>1 SIGSEGV）强制 tiled bm=1 → 分核数上界 = M = 4（4/48 核），实测单 kernel 25.89µs（fp16）/ 30.61µs（bf16）已贴 4 核 MTE2 带宽地板。预注册的 C6 N-split 备选在本轮 Stage 4 A/B 裁决中以 **2.20×（fp16）/ 2.60×（bf16）**结构性胜出（端到端 11.78µs / 11.75µs，均 < 0.8× 翻转阈值），并已在 `perf_opt/_argreduce_kernel.py` 最终版采纳（5 workload 几何平均 11.5× 加速）。

**归因章节**：§1.6.3-#4、§1.6.4（预注册实验裁决）、§5.5（tiled 分核上界）、§9.1 C-5；前次任务 round9 N-split 终版（partial ≈15.7µs + merge ≈1.7µs）为性能先例。

**路由说明**：本任务为 migration-harness（含用户显式追加的 Stage 4），按 `perf-feedback.md` §3「设计修订（路径 C）… harness 场景无 Stage 4，均不适用」→ 采用**附录补记**归档，不触发设计修订循环。C6 N-split 已在 perf_opt 最终版落地，未来同族「小 M 大 N」负载的设计（VP-2026-0044 形态消费者）应把该负载类路由到 N-split 结构。详见 `examples/argmax/_argreduce_kernel/perf_opt/perf_feedback.md`。
