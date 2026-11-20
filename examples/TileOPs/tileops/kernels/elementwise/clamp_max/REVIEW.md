# _make_clamp_tensor_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md
- 任务类型: 迁移（`migration` / `harness`，源仓 TileOPs 同构工程）
- 源算子: /tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py（`_make_clamp_tensor_kernel` @L3017、`ClampTensorFwdKernel` @L3227）；op 类 /tmp/opencode/TileOPs-ref/tileops/ops/elementwise/clamp.py（`ClampMaxFwdOp` @L209）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（6 条 warn 级观察随维度记录，不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS` — 于 `examples/clamp_max/_make_clamp_tensor_kernel/` 执行 `python3 verify_equiv.py`，输出 `OVERALL: EQUIV_PASS (14/14 checks passed)`（E1×6：fp16/bf16/fp32 各 candidate-vs-ref + candidate-vs-baseline；E2×2：fp16/bf16 native-vs-fp32transit；E3×3；E4×3），与 DESIGN.md §1.6.1 内嵌结果表**逐项一致**（表列 14 项全 `EQUIV_PASS`），无脚本缺失、无 `EQUIV_FAIL`、无与内嵌表不一致项。
- **design_calc_check** (`python3 .agents/tools/design_calc_check.py --design examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md`):
  - `ub_budget`: **skip** — 未从内存规划解析出分层 buffer 表（advisory checker）→ **机械跳过，人工已核**：§4.5 活跃 buffer 3 个（`x_ub`/`hi_ub`/`y_ub`），fp16/bf16@4096 或 fp32@2048 手工合计均 24KB，×1.7 膨胀 ≈41KB < 192KB（`CONST-capacity-910B2C`），余量充裕。
  - `l0c_budget`: **skip** — 无 Cube 路径 API（`T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope("Cube")`）→ 纯 Vector，无 L0C，**人工已核 n/a**。
  - `core_split`: **skip** — 未解析出 dims、declared logical cores = None → **机械跳过，人工已核**：§5.5 三要素齐全（① 逻辑核 1–4096；② 物理核 `NPUUtils.get().get_aicore_num()` 实查 = 20，×2 = 40，附查询代码与返回值；③ 判定极大规模 → `num_kernels = min(num_logical, 40)`、`num_local = T.ceildiv(num_logical, num_kernels) = 103` 静态常量）。
  - `r3_metrics`: **pass** — 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）均在 §1.6.0 R3 表内且基线在表。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源文件 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（L3017–L3265）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/clamp.py`（全 342 行）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/_base.py`（`_ClampTensorBase._expand_flat` L1232–L1236）、Stage 0 提取件 `examples/TileOPs/tileops/kernels/elementwise/clamp_max/_clamp_max_fwd_kernels.py` 逐项核对：

- **语义理解正确性（pass）**：§0.1「`y[i]=min(x[i],hi[i])` ≡ `torch.clamp_max`；NaN 传播：x 或 hi 为 NaN → NaN」与源注释 L3034–L3039（"If ``x``, ``lo``, or ``hi`` is NaN at a position, the output at that position is NaN…CUDA ``T.max``/``T.min`` do not propagate NaN by themselves…so we add explicit ``isnan`` guards"）一致。§0.1 判定「无规约、无跨元素依赖」与源逐元素 `T.Parallel` 结构一致。§0.2 I/O（两输入一输出、same dtype、输出 shape = `broadcast_shapes`、`out_idx=[2]`）与源 `ClampMaxFwdOp.__init__`（clamp.py L230–L238 `out_shape=broadcast_shapes`、kernel `has_min=False, has_max=True`）及 `_make_clamp_tensor_kernel` has_max-only 分支（L3197–L3220，`out_idx=[2]`、`y: T.Tensor((N,), dtype)`）一致，无转置/布局变换。
- **算法解读完整性（pass）**：§0.3 步骤 1–9 覆盖源全部计算语句——host `_expand_flat`（`_base.py` L1234–L1236 `t.expand(...).contiguous().view(-1)`，**注意**：同形时 `tuple(t.shape)==target_shape` → 无 expand，`contiguous` 为 no-op 纯视图，§0.5 判「同形路径合规」成立）；`block_size=threads*npt`、grid=`ceildiv(N,block_size)`（L3046、L3205）；`alloc_fragment` + `T.copy` 载入（L3206–L3209）；`T.Parallel(threads,npt)` 内 `k=i*npt+j` → `cast(f32)` → `T.min` → 2 条 `isnull` 守卫 → `Cast(dtype)`（L3210–L3219）；`T.copy` 写回（L3220）；Op 层 `view(out_shape)`（clamp.py L255）。**无遗漏、无臆造**。§0.3 观察「非 fp8 路径无显式尾块边界判断，依赖 `LegalizeSafeMemoryAccess`」与源 L3205–L3220 无 `if idx<N` 一致（仅 fp8 路径 L3116 有 `if idx < N`）。数据流图覆盖 `x_reg`/`hi_reg`/`y` 三 buffer。
- **优化手段覆盖完整性（pass）**：§0.4 表 7 项与源码显式优化逐项对应——`explicit_parallel`（`T.Parallel(threads,npt)`）、连续段 coalesced、`register_copy`（`alloc_fragment`+`T.copy`）、fp32 中转、显式 NaN 守卫、Python 编译期 `has_min/has_max/is_fp8` 分支、工厂闭包 + `lru_cache(maxsize=32)`（L3017）。源码**确无** SMEM tiling / warp shuffle / swizzle / 异步流水 / online 手段，§0.4 声明「对照清单无遗漏」成立。§0.3 步骤 3 的 `threads=512 / npt(fp32=4, 非 fp32=8)` 与 wrapper K11 冻结契约一致（`examples/TileOPs/.../clamp_max.py` `_DEFAULT_THREADS=512`、`_NPT_FP32=4`、`_NPT_NON_FP32=8`，block_size 2048/4096；源工厂默认 `threads=256` 被 `ClampTensorFwdKernel` 覆盖）。
- **耦合性判定合理性（pass）**：§0.5 每个条目四态处置 + 依据，判定统计自洽（保留 6 / 等价替换 3 / 重新设计 2 / 舍弃 3，逐行清点一致）。处置与 GPU→NPU 能力映射无矛盾：`threads=`/npt kwarg 判「舍弃」与 `is_npu=True` 一维 Kernel 一致；fp32 中转、NaN 守卫判「舍弃」均附 NPU 实测依据；无 warp-shuffle 类误判（源确无）。**舍弃项均写明理由**。
- **NPU 重设计可行性（pass）**：§0.6 三项重设计均有「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足 NPU 约束——R1 一条 `T.vmin`（原生 dtype）；R2 一维 `T.Kernel(num_kernels, is_npu=True)` + persistent 静态边界（非三维 Kernel、非超发）；R3 行分解广播内移（`T.vmin` 原生标量/一维广播 + `T.vbrc` 兜底，末维 stride ∈ {0,1} 论证成立）。语义保持论证为「数学恒等/值域恒等」，逐元素无累加顺序问题（论证正确）。**优化意图承接总表**（§0.4→NPU）逐项登记，fp32 中转/NaN 守卫「经论证后舍弃」而非静默丢弃。
- **设计与决策一致性（pass）**：§1.4/§3.1/§3.3/§4/§5/§6 描述的是重设计后算法——§3.3 伪代码**无 fp32 cast、无 `isnan`**；§6.2「核内只有一个 `T.serial(num_local)` 调度循环，无任何逐元素标量循环」。无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性（pass）**：§8.1 golden = `torch.clamp_max(input, max)` 广播直算，以 §0.1 语义为唯一依据，明确「不复刻 §0.6 NPU 算法（不模拟 `T.vmin`/分核/尾块）」，优先移植源测试 `test_special_elementwise_conformance.py::test_clamp_max_tensor`。精度对比不与设计自证。

**warn 级观察（不阻塞）**：§9.1 R1（`T.vmin` 文档 bf16「×」vs 探针实测 OK）与 R2（`VMinOp` NaN 传播为未文档化假设）均为**真实存在的能力不确定性**，但设计已附「探针实测 + 强制 L0 复测（L0-1 bf16 / L0-3 NaN 角点）+ 回退方案」，属已登记风险而非缺陷。

### 1. API 可行性 — pass

§3.2 API 映射逐条核对，均有 `docs/` 与 `examples/` 双佐证，且优先 v-prefix：
- `T.vmin`（`npuir_min` 别名）：`docs/Tilelang.language/比较操作/T.vmin.md` §2.1/§2.2.2/§2.4 + `testing/npuir/arith_ops/test_clamp_dev.py` 等先例；§2.2.2 明示「src2 可以使用标量」「仅一个维度不同且 size=1 → 带 brc 的 min」，与设计标量/一维广播路径一致。
- `T.min`（标量尾块）、`T.copy`（GM↔UB 切片）、`T.alloc_ub`（1D）、`T.ceildiv`、`T.Kernel(grid, is_npu=True)`、`T.serial` + `task=i·K+cid` + `if`、`@tilelang.jit(out_idx=[2], target="npuir")`——分别有 `T.min.md`/`T.copy.md`/`T.alloc_ub.md`/`T.ceildiv.md` 与 `examples/elementwise/vec_add_1d.py`、`T.vmin.md` §2.4（`@tilelang.jit(target="npuir")` Developer 形态）、`T.vcast.md` §2.4 persistent 分发先例。

**独立复核结论**：`T.vmin` 的标量 src2 与一维广播确为文档化能力（§2.2.2），非设计臆造；`T.vclamp` 弃用论据（展开为 `npuir_max`+`npuir_min`，2 op/元素）与 `T.vclamp.md` 一致，而 `T.vmin` 单 op 更优——弃选成立。无「凭记忆猜 API」。

**warn 级观察**：探针 Q-C2 实测的是 2-D `[M,N]` vs `[M,1]` 广播，而伪代码 §3.3 末维广播分支用 1-D `(1,)` vs `(block_size,)`。文档 §2.2.2 的规则（任一单维 size=1）在 1-D 上同样成立，且设计另备 `T.vbrc` 兜底，属可接受；建议 Stage 3 在 L0-2/L0-6 直接覆盖 1-D 形态。

### 2. 内存层级规划 — pass

§4.4 搬运路径为纯 Vector 单阶段 `GM[x]/GM[hi] → UB → T.vmin → UB → GM[y]`，无 L1/L0 参与、无中间 GM 往返、无 GM→L0 越级路径（正确，本算子无 Cube）。§4.3 每个 buffer 标注存储层级（`x_ub`/`hi_ub`/`y_ub`/`hi1` 均 UB）。§4.5 UB 预算：fp16/bf16@4096 或 fp32@2048 活跃 3 buffer 手工合计 24KB，×1.7（`CONST-capacity-910B2C`）≈41KB < 192KB；block_size 上限公式 `floor(192KB/(3×E×1.7))` 复核无误（fp16/bf16 ≤19267、fp32 ≤9634，wrapper 默认远低于上限）。§4.3 明示**无 fp32 upcast / 无 isnan / 无承载 dtype buffer**（采纳项 1/2 消除），与 §1.6.1 一致。`design_calc_check` 的 `ub_budget` skip 已人工复核通过（见机械复核段）。

### 3. Tiling 策略 — pass（分核三要素）

- **Block 划分（pass）**：§5.2 `block_size=4096(fp16/bf16)/2048(fp32)`，附 5 条理由（32B 对齐、向量宽度整除、UB 余量、发射项低于流量项、与 wrapper K11 冻结契约一致）。
- **对齐（pass）**：2048×4B=8KB / 4096×{2,4}B 均为 32B 整倍数；fp16/bf16 向量宽 ×8、fp32 ×4 整除。
- **非整除（pass）**：§6.4 `tail = T.min(block_size, N_total−off)` + runtime 边界 `T.copy` 切片，`T.vmin` 整 buffer 计算、仅 `[0:tail]` 写回，垃圾不落 GM。
- **分核三要素（pass）**：① 逻辑核数 `num_logical = ceildiv(N,block_size)` 逐 workload 列表（1 / 4096 / 4096 / 4096）；② 物理核数**实查**（`NPUUtils.get().get_aicore_num()` = 20，纯 Vector ×2 = 40，附查询命令与返回值，未用假设值）；③ 规模判定：`num_logical > 40` → persistent，`num_kernels = min(num_logical, 40)`、`num_local = T.ceildiv(num_logical, num_kernels)` **编译期静态**（103），尾核空转 <1%。与 `.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素一致。smoke（逻辑核 1 ≤ 40）退化为 1 核，处理正确。

### 4. 技术约束检测 — pass

§3.5.1 逐条对照本项目已知限制：无三维 Kernel（一维 grid）、无 GEMM 非整除、无 L0C 溢出、无 uint8/bool 输出（输出 `same_as(input)`）、无 `T.vcmp`；GPU 专用 API（`threads=`/`T.Parallel`）已由 §0.5 舍弃 / §0.6 重设计处置；物理核数限制由 §5.5 处理；host 侧输入操作约束（广播物化）由 §0.6 R3 内移处置。**无遗漏的阻塞约束**。`T.vmin` 文档 bf16「×」已单列（§3.5.1 末行）并附探针实测与回退。

### 5. 循环与同步 — pass

§6.1 循环结构明确（核间 `T.Kernel` / 核内 `T.serial(num_local)` 静态边界 / 逐元素**无循环**由 `T.vmin` 覆盖 / 尾块 `if task<num_logical` + `tail`）；§6.3 单阶段无 `T.Pipelined`，auto multi-buffer 重叠 MTE2/VEC；§6.4 尾块逻辑完整。§7 同步为 Developer 自动同步（`copy→vmin→copy` 数据依赖由编译器插入），每 task 独立、零跨核同步，纯 Vector 非 MixCV → 无需手动 `sync_block_set/wait`。**与 §2.1 编程模式（Developer）匹配**。§1.6.2 向量化结论「kernel 内无任何逐元素标量循环」与 §6 交叉核对一致。

### 6. 验证方案 — pass

§8.1 含 golden 草案（`torch.clamp_max` 广播直算，`equal_nan=True` 必需并说明理由）。§8.3 含 L0 门槛计划 8 项：L0-1 全 dtype 同形（**bf16 强制**复测 R1）、L0-2 两条 manifest 广播 workload + 0-dim 上界、L0-3 NaN 传播、L0-4 IEEE 角点、L0-5 非整除（N=4097）、L0-6 标量上界、L0-7 Op 层 shape 契约、L0-8 roofline 契约；形态/dtype 具体，覆盖两条广播 workload。§8.2 按 dtype 给 atol/rtol（fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5），**比默认更紧**（min 为操作数选择 → 预期 bit-exact，容差为余量），并额外断言 NaN 模式一致。与 manifest `ClampMaxFwdOp` signature（fp16|bf16|fp32、`same_as(input)`、`broadcast_shapes`、roofline `flops=N`/`bytes=3·N·elem_bytes`，`elementwise.yaml` L485–L522）一致。

### 7. 完整性与一致性 — pass

模板 11 章节齐全（§0 迁移分析含 0.1–0.7 全小节）。全文件无 `{placeholder}`/`TODO`/`待补充`（grep 零命中）。§3.5.3 引用的 5 类路径**实查全部存在**（`examples/elementwise/vec_add_1d.py`、`examples/ge/ge_fwd_kernel/DESIGN.md`、`examples/le/le_fwd_kernel/DESIGN.md`、`examples/TileOPs/.../mish/mish_kernel/`、`testing/npuir/arith_ops/test_clamp*.py`、`testing/npuir/parallel_ops/test_min_max_parallel.py`）。§0.7 关于调度 prompt 提到的 `tests/ops/test_clamp_max.py` / `benchmarks/ops/bench_clamp_max.py` 在源仓与 NPU 仓均缺失的说明——已实查 manifest `source.test/bench`（L529–L530）确指向缺失文件，与 `find` 结论一致。§3.5.2 参考实现差异表、§4.7 JIT 配置、§10 交付清单内部一致；API 映射 / 内存规划 / Tiling 三处描述相互自洽（block_size 值、out_idx、dtype 族全一致）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在（pass）**：含 1.6.0 调研、1.6.1 数学等价优化、1.6.2 向量化替代、1.6.3 向量化轴与布局决策四子节，结论行字面量齐全（`调研结论`/`优化结论`/`向量化结论`/`布局决策结论`）。
- **算法调研存在性与完整性（pass）**：四问齐全——R1 候选表**含基线候选 #0**（源公式）+ 候选 1–8（每项含等价性初判与收益方向）；R2 在线算法**明确结论「无在线变体需求（天然单遍）」+结构依据**（`y_i` 仅依赖 `(x_{off_x(i)}, hi_{off_hi(i)})`，无 running 统计量/跨元素依赖）；R3 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径写明）；R4 硬件亲和逐候选评估（单元匹配/容量/对齐/静态边界/流水融合）。深度与单步逐元素类匹配（轻量调研）。
- **调研结论正确性独立复核（pass）**：
  - *等价性*：采纳项 1（IEEE `minimum(x,NaN)=NaN` ≡ `torch.clamp_max`）与采纳项 2（`min(a,b)∈{a,b}`，fp16/bf16 在 fp32 精确可表示 → 原生 dtype bit-exact）经我独立推演**成立**；无精度稳定化隐患（单次 `min`，无累加/无 cancellation）。
  - *复杂度算术复算*：基线 8N 标量 op（2 cast↑+1 min+2 isnan+2 select+1 cast↓）核对无误；NPU 重设计 1 op/元素；访存 3N·E（同形）、广播路径免 `+2N·E`（cnn-feat `2×(12.85M×2)=51.4MB`、row-broadcast `2×(16.78M×2)=67.1MB`，与设计逐数一致）；流量项 `3N·E/1.05TB/s` 与发射项折算（1.05 = 1.26×40/48）复算一致。
  - *负向断言主动复核*：对照 `tilelang-op-design/references/algorithm-candidates.md` 命中行 `ALG-elementwise`（L98–L104）——R1「公共子表达式消除/除法乘倒数/cast 链合并」逐项已评估（无公共子式、无除法、**cast 链消除即采纳项 2 命中**）；R2「逐元素映射天然单遍」与本算子一致；R3「融合消除中间 GM 往返」已由 §0.6 R3 承接；R4「单遍 Vector 流 + dtype 路径」已评估。「无更优替代」列出 7 项调研范围（参考表命中行 / 恒等变形自查 / 本仓 elementwise 先例 / pattern-library elementwise.md+cases.md / kb_search 命中 PL-1.11+VP-2026-0087 / 源结构 / 文档偏差）。**未发现被漏掉的已知候选**（`min` 已是 1 op/元素下界；Welford/online/分块合并等归约类候选对无规约算子不适用）。
  - *亲和性淘汰依据*：基线「逐元素标量链落错单元」淘汰与 `VP-2026-0087`（vmin 族原生支持一般列广播）、`PL-1.11`（vmax/vmin 钳位不清除 NaN）实测条目一致；容量/带宽淘汰引用 `CONST-capacity-910B2C`；弃选 `T.vclamp`（2 op）依据 `T.vclamp.md` 亲自核对一致。
- **调研→设计一致性（pass）**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以 §1.6.0 选定的「裸 `T.vmin` 原生 dtype 单遍 + 一维 Kernel persistent + 广播内移」为口径；§1.6.1 采纳项与 §0.5「舍弃 fp32 中转/NaN 守卫」、§0.6 R1 一致；源优化手段（fp32 中转、NaN 守卫）被替换项**有承接说明**（§0.6 末尾总表），无静默丢弃。
- **数学等价优化完整性（pass）**：§1.6.1 对选定算法做了系统审视（8 类：恒等变形/稳定化/公共子式/算子降代换/访存削减/归约/cast 链/谓词守卫），采纳 2 项均四要素齐全【原式 → 优化后公式（`y=vmin(x,hi)`）→ 等价性论证 → 收益量化（每 task 向量 op 8→1：省 2 cast + 2 isnan + 2 select）】；其余类别显式结论「无进一步可采纳项」+依据（未用「无优化空间」误作豁免）。
- **等价性论证独立推演（pass）**：采纳项 1/2 推演成立（见上）；`verify_equiv.py` 重跑 14/14 `EQUIV_PASS`，与内嵌表一致。**warn 级观察**：E1 的 CPU 基线 `_guard_formula` 用 `torch.minimum`（本身已传播 NaN），故 E1 的 guard-elimination 部分在 CPU 上接近同义反复，无法单独证伪 NPU 侧行为——设计已**如实分工**（「CPU 机器验证锁定公式等价，探针锁定硬件行为」）并强制 L0-3，属已缓解的可接受安排，非等价性论证缺陷。
- **向量化替代覆盖完整性（pass）**：§1.6.2 盘点 9 个计算点（#1 截断本体、#2/#3 fp32 中转、#4 NaN 守卫、#5 上界广播、#6 行 base、#7 尾块、#8 tile 调度循环、#9 host 元数据），与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**；#1 用 `T.vmin`、#2–#4 **直接消除**、#5 用原生标量/广播；保留 4 处（行 base / 动态边界 / tile 调度 / host 元数据）逐项理由具体到本算子（block 级索引、动态边界、persistent 结构、host 元数据）。
- **不可替代理由充分性（pass）**：#6–#9 理由均具体（含 `vec_add_1d.py` 已验证模式、`ascend-constraints.md §4` host 禁改输入约束），非「实现简单」式泛泛而谈。
- **替代 API 可行性（pass）**：`T.vmin` 标量/广播有 `T.vmin.md` §2.2.2 佐证，`T.vbrc` 兜底有 `docs/Tilelang.language/shape操作/T.vbrc.md`，与维度 1 口径一致，优先 v-prefix。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 候选矩阵 6 项（展平一维+行分解 / 2-D 块化 tile / host 预广播 / 逐元素标量 / 位模式 / vclamp 两步），逐项给量化取舍（对齐、UB、每 task op 数、上游 GM 流量）；类别（逐元素/广播）必选候选「广播轴×向量轴对齐」已覆盖；弃选均量化（host 预广播广播路径 `+2N·E`、逐元素标量落错单元、位模式浮点不成立、vclamp 2 op）；主选 #1 有轴质量评分；**迁移任务对照了源隐式轴**（源 `T.Parallel(threads,npt)` 的连续 N 轴，NPU 独立评估后同构，但明确标注「源轴是输入非结论」）。
- **实验裁决模式完整性（pass）**：§1.6.3 判定依赖未实证常数（发射摊销律）→ 提供主选 #1 + 备选 #2（结构化：buffer 形状 `(R_blk,block)`、连续段载入、rank-2 `T.vmin`、尾块截断、分核三要素按 2-D 粒度重算 `num_logical=ceildiv(R,R_blk)×ceildiv(N,block)`、UB 预算）+ 裁决计划（代表 shape `[4096,4096]` fp16、指标 msprof Task Duration + vec_ratio + mte2_active_bw、阈值「备选 < 主选 × 0.97 才翻转」、回写 §1.6.3）+ shape 特化工厂逐 shape 分派评估。三件套齐全。
- **布局决策与实现一致（pass）**：§1.6.3 选定 `(block_size,)` UB buffer 与 §3.3 伪代码（buffer 形状 `(block_size,)`、`T.vmin` 操作数即整 buffer）、§4.5 UB 预算（3 buffer×(block_size,)×E×1.7）、§6 循环结构（核内无逐元素循环）三方一致。
- **与下游章节一致（pass）**：§3.1 以优化后公式 `y = vmin(x,hi)` 为输入；§6 无 §1.6.2 未论证的逐元素标量循环。

## 结论说明

全部 9 个检视维度（含阻塞级维度 0 与维度 8）均通过，无任何阻塞级 fail，故结论为**通过**。检视中记录的 6 条 warn 级观察（bf16/NaN 能力不确定性 R1/R2、E1 硬件验证分工、广播内移的 wrapper 签名扩展、1-D 广播探针未直接覆盖、备选 UB 估算 buffer 计数 2 vs 3 的微小口径差异）均已在 DESIGN.md §9.1/§1.6.3 中显式登记或有回退/复测计划，不构成阻塞，供 Stage 3 实现时按 L0 门槛（L0-1/L0-2/L0-3/L0-6）优先复核。
