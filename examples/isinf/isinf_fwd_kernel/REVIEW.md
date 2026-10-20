# isinf_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/isinf/isinf_fwd_kernel/DESIGN.md
- 任务类型: 迁移（scenario=migration，mode=harness）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/comparison.py（`IsinfFwdKernel`，复用 `_base.py` / `_builders.py` / `_op_body.py` / ops/elementwise/comparison.py）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **PASS**。`python3 examples/isinf/isinf_fwd_kernel/verify_equiv.py` 退出码 0；2 个采纳项 ×3 dtype 全部 `EQUIV_PASS`（item1 fp16/bf16/fp32 mismatches=0；item2 fp16/bf16/fp32 mismatches=0），判别段 A `(x-x)!=0` mismatches=399/403/459（=NaN 数，`DISCRIM_OK`）、判别段 B `NOT(isfinite)&NOT(isnan)` mismatches=0（`DISCRIM_OK`）。执行输出与 §1.6.1 内嵌结果表逐行一致（item1=0、item2=0、A=399/403/459、B=0）。
- design_calc_check: 以 README/脚本实际接口 `--design examples/isinf/isinf_fwd_kernel/DESIGN.md` 执行（任务提示中的 `--dir` 形参在已安装的 `.agents/tools/design_calc_check.py` 中不存在，脚本仅接受 `--design`；`--dir` 报 `error: the following arguments are required: --design`）。逐项 JSON checks：
  - `ub_budget` → **skip**：`no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)` —— **机械跳过，人工已核**：§4.5 逐 buffer 求和 = 53248 B（52KB，fp16/bf16 路径）/ 77824 B（76KB，fp32 路径），均 < 192KB（§4.5 已按 ×1.12 显式-alloc 带 ≈85KB、×1.7 悲观膨胀 ≈129KB 双口径核对，均安全）。
  - `l0c_budget` → **skip**：`no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design` —— 纯 Vector 无 Cube，正确跳过。
  - `core_split` → **skip**：`insufficient parsed dims {}; declared logical cores = 40` —— **机械跳过，人工已核**：脚本无 M/bm 可解析（纯 1-D 逐元素无 GEMM 网格）；§5.5 逻辑核数 `num_logical = ceil(N/block_size)`（16M→4096、256M→65536）、物理核数 `NPUUtils.get().get_aicore_num()` 实查 20 ×2（纯 Vector）= 40、核内串行边界 `num_local_tasks=ceil(num_logical/40)`（16M→103、256M→1639）三要素齐备且算术自洽。
  - `r3_metrics` → **pass**：`all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子代码（`comparison.py`、`_base.py`、`_builders.py`、`_op_body.py`、`_dtype.py`、`ops/elementwise/comparison.py`、`call_spec.py`）与已抽取参考 `examples/TileOPs/tileops/kernels/elementwise/isinf/_isinf_fwd_kernels.py`，与 DESIGN.md §0 逐项核对。

- **语义理解正确性 — pass**：源码 `IsinfFwdKernel.op_func`（`comparison.py:194-199`）= `T.isinf(T.cast(x, "float32"))`；§0.1 数学语义「±inf→True、NaN/有限/±0/次正规→False」与 `torch.isinf` 一致，且明确「有序 `eq` 遇 NaN 恒 false，故 NaN 正确判为 False」。dtype 语义核对：`FloatPredicateKernel(FloatUnaryKernel)`（`_base.py:500-510`）`SUPPORTED_DTYPES=FLOAT_DTYPES`（fp16/bf16/fp32，`_dtype.py:25-29`）、`OUTPUT_DTYPE=torch.bool`，与 §0.1/§0.2 一致。§0.2 I/O（input `[*S]` 浮点展平、output 同 shape bool）与 `UnaryKernel.forward`（`_base.py:253-256`：`flat(x)`→`_compiled_fn`→`_restore_output_dtype(result).reshape(x.shape)`）及 NPU `IsinfFwdOp._eager_forward`（`ops/elementwise/isinf.py`：`contiguous().reshape(-1)`→kernel→`reshape(orig_shape)`）一致。源码 `T.cast(x,"float32")` 被 §0.1 正确解读为「谓词可用性适配，非精度提升」。
- **算法解读完整性 — pass**：§0.3 七步（host 展平 / 分块 `T.copy` 装载 / `T.cast` 宽化 / `T.isinf` 谓词 / `if_then_else(pred,1,0)` int8 字节 / 分块写回 / host `view(bool)`+回塑）与源码计算语句一一对应：`make_unary_regcopy`（`_builders.py:84-121`）的 fragment `T.copy` 装载/写回 + `T.Parallel(threads,npt)` 逐元素；`store_unary_bool_as_int8`（`_op_body.py:50-72`）= `T.if_then_else(op_func(x), T.cast(1,"int8"), T.cast(0,"int8"))`；`_dtype.py:9` `BOOL_STORAGE_DTYPE="int8"`。无遗漏步骤、无臆造步骤；host 侧逻辑（展平视图 + view 回转）已列出。
- **优化手段覆盖完整性 — pass**：§0.4 五项（register_copy fragment 分块、coalesced 连续访问、`threads×npt` 两级并行、fp32 宽化谓词、bool 字节存储）均能在源码定位（`make_unary_regcopy`、`FloatPredicateKernel.DEFAULT_STRATEGY="register_copy"`、`UnaryKernel._get_effective_op_func` 的 bool 包裹）。未把实现层优化误判为算法层；明确声明源无 swizzle/mma/warp shuffle/online，经核对源码确实未使用。
- **耦合性判定合理性 — pass**：§0.5 每条四态处置（保留/等价替换/重新设计/舍弃）均附依据；「fragment→UB」「coalesced→MTE 连续搬运」「bool int8 字节存储」判等价替换（对照 `migration-analysis.md` §5.3 映射表 shared memory→UB、`_shared` 映射一致）；`T.isinf`+fp32 宽化判重新设计（NPU 无向量 `T.isinf`）；swizzle/mma/shuffle 判「舍弃（不适用）」（源未使用，无迁移对象，写明理由）。无「舍弃理由待补」项，与迁移分析三层模型（语义层永远保留）一致。
- **NPU 重设计可行性 — pass**：§0.6 三项重设计各有「源方案 → NPU 新算法 → 语义保持论证」：
  - R1（并行结构）：GPU `T.Kernel(ceildiv(N,bs),threads=128)`+`T.Parallel` → 一维 `T.Kernel(num_kernels,is_npu=True)` + `block_size` 折叠 + Vector 整块 v-prefix；满足「仅支持一维 Kernel」约束（`ascend-constraints`）。
  - R2（无限谓词）：`T.isinf(T.cast(x,"float32"))` → `(x==+inf)∨(x==-inf)` 经**有序 `eq`**（原生 dtype）；语义保持论证正确（独立推演见维度 8）。
  - R3（bool 输出表示）：`if_then_else` → 掩码→f16 `vcast`→`vadd` 合并→`vcast` 落字节；`+inf`/`-inf`/NaN 三分支逐项论证与源一致。
- **设计与决策一致性 — pass**：§11.1 逐条把 §0.5/§0.6 决策映射到 §1–§10（§3.3 伪代码 → R1；§3.2 步骤 2–3 → R2；§3.2 步骤 4–7 → R3；§4.3/§4.4 → fragment→UB 等价替换）。无「§0 判重设计、§3 照抄源方案」脱节；§11.2 逐项列出源优化手段的意图承接（无静默丢弃）。
- **golden 独立性 — pass**：§8.1 golden = `torch.isinf(x.detach().cpu().float())`，与源仓测试 `examples/TileOPs/tests/ops/test_isinf.py` 的 `ref_program = torch.isinf(...)` 同源；**不含** NPU 算法的任何要素（无 `vcmp "eq"`/`vadd` 掩码合并/`vcast` 字节链/int8 字节存储/persistent 结构），验证独立性成立。

### 1. API 可行性 — pass

- **API 存在性**：§3.2/§3.4 列出的每条 API 均有 `docs/` 或 `examples/` 佐证，逐条亲自打开文档核对：
  - `T.vcmp`：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1 `"eq"` 合法、§2.2.1 支持 fp16√/fp32√/bf16√、§2.3 输出 bool 且「不可直接写回 GM（i1 无法 load/store GM）」——与 §3.4 完全一致；源码 `tilelang/language/customize_npuir.py` 无 `unordered` 标志的断言与 CG-2026-0024 一致。
  - `T.vcast`：`docs/.../数据类型转换操作/T.vcast.md` §2.2.1 表列 `i1→f16`（√, rint）与 `f16→i8`（√, rint），**无 `i1→i8` 直连**——DESIGN 关于「须经 fp16 中转」的论断与文档一致。
  - `T.vadd`：`docs/.../数学操作/T.vadd.md` §2.2.1 fp16√，§2.3 支持 `dst` 与 `src0` 同 buffer（原地）——§3.2 `T.vadd(f_pos,f_neg,f_pos)` 合法。
  - `T.infinity`：`docs/.../创建操作/T.infinity.md` §2.2.1 fp16√/fp32√/**bf16×**——§3.4 论断一致。
  - `T.vor`/`T.vmax`/`T.vnot`/`T.vselect` 的 bool/int1 列均为 ×（`docs/.../逻辑操作/T.vor.md`、`/比较操作/T.vmax.md`、`/逻辑操作/T.vnot.md`、`/条件操作/T.vselect.md` §2.2.1），`T.vselect` int8/uint8 亦 ×——DESIGN 关于「掩码→byte 必须经 fp16 域合并」的解释成立。
  - `T.copy`（双显式 slice，GM↔UB Developer √）、`T.Kernel(一维,is_npu=True)`、`T.alloc_ub`、`T.ceildiv/T.min` 均有 docs 或同族 examples 佐证。
- **API 与模式匹配**：Developer 模式选自动同步 + v-prefix 整块 API，与 §2/§7 的自述一致。
- **v-prefix 优先**：全链使用 `vcmp/vcast/vadd`（v-prefix），无 npuir_ 遗留写法，符合 AGENTS.md 约定。
- bf16 常量构造 `T.cast(T.infinity("float32"),"bfloat16")` 属**未文档化假设**（`T.infinity` bf16 ×，`T.fill` 接受 Cast 标量表达式有 isnan repro 先例），DESIGN 已在 §3.4 显式标注并列入 §9.1 风险 + Stage 3 首验，符合负向/假设断言证据规则，不构成阻塞。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 GM→UB→GM 两级路径（无 Cube，故无 L1/L0A/L0B/L0C），无 GM→L0 直连等越级路径；数据仅 GM→UB 一次读入、UB→GM 一次写出，**无中间 GM 往返**；常量 ±inf 一次性填 UB（循环外）。
- **UB 预算**：§4.5 逐 buffer 求和人工复算 = fp16/bf16 路径 **52KB**、fp32 路径 **76KB**，与 DESIGN 表列一致；对目标平台 UB 容量 **192KB**（`constants.md` `CONST-capacity-910B2C`）有余量（×1.12 ≈85KB、×1.7 ≈129KB 均 < 192KB）。block_size=4096 的选择理由（比同族 isnan 多 2 个输入 dtype 常量 tile，取 4096 保证 fp32 在 1.7x 悲观口径下仍安全）成立。
- **L0C 容量**：无 Cube，不适用（`l0c_budget` 机械 skip 正确）。
- **Buffer 层级标注**：§4.3 每个 buffer 标注存储层级（全部 UB），dtype 明确（含 `m_pos`/`m_neg` bool、`f_pos`/`f_neg` f16、`y_ub` int8）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_size=4096` + 选择理由（UB 安全 + 32B 对齐 + 整除向量宽）。
- **对齐约束**：4096×{1,2,4}B 均 32B 整数倍；fp16/bf16 整除向量宽 ×8、fp32 ×4（4096=512×8=1024×4）。
- **非整除处理**：§6.4 `tail_size=T.min(block_size, remaining)` + `block_id < num_logical` guard，双显式 slice `T.copy` 仅搬有效元素，垃圾尾部不写回。
- **分核策略（三要素）**：对照 `.agents/skills/_shared/standards/core-split-strategy.md` §1：① 逻辑核数 `ceil(N/block_size)`（§5.5 表 16M→4096/256M→65536）；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查记录**（查询代码 + 返回值 20，纯 Vector ×2=40，§5.5 已注明实查日期/环境），未使用文档假设值——且明确指出与 `constants.md CONST-aicore-910B2C`（24）不一致并以实查为准；③ 规模判定与方案（小规模「无需适配」/极大规模「固定启动 40 内核 + 核内 `T.serial` 串行」，`num_local_tasks` 编译期常量满足静态边界）。核数与 block 取值自洽（16M→103、256M→1639 复算一致）。

### 4. 技术约束检测 — pass

- **三维 Kernel**：源即一维，本设计一维 `T.Kernel(num_kernels, is_npu=True)`，不涉及三维。
- **GEMM 非整除 / L0C 溢出**：无 GEMM/Cube，不适用。
- §3.5.1 逐条核查本项目已知限制：GPU 专用 API（`T.isinf`+宽化、`T.Parallel(threads,npt)`、bool i1 GM 存取）全部改写；`T.infinity` bf16 × 有对策（§3.3 由 fp32 inf 精确窄化）；条件性 `T.alloc_ub` 被 parser 拒（CG-2026-0020）已通过无条件分配规避。§9.2 常见错误表覆盖完整（含误用 `vcmp "ne"`、减法变体、bool 域合并、`T.infinity("bfloat16")`、尾块越界、`dtype=` 关键字 false alarm）。

### 5. 循环与同步 — pass（建议级）

- **循环结构**：§6.1 明确 block 级核并行（`T.Kernel`）/ tile 级核内串行（`T.serial(num_local_tasks)`，静态边界）/ 元素级向量化（v-prefix）。无逐元素标量循环，与 §1.6.2 一致。
- **同步与模式匹配**：§7 自动同步（Developer），无手动 set_flag/wait_flag，与 §2.1 Developer 模式一致。
- **尾块处理**：§6.4 说明非整除逻辑（与 §5.4 一致）。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 给出 `torch.isinf` 参考实现草案，且明确独立于 NPU 算法。
- **L0 测试计划**：§8.3 给出具体规则 shape（N=1048576）、三 dtype、精度标准（精确相等），并含负向用例（拒绝非 float dtype）。
- **精度标准**：§8.2 atol=rtol=0（谓词精确，与源 `_get_tolerances` 一致，§1.6.1 mismatches=0 佐证）；边界用例（±inf/NaN/±0/次正规/极值）与判别探针（全 NaN 须全 False）规划完整。

### 7. 完整性与一致性 — pass（建议级）

- **章节齐全**：模板 11 章节齐全，§0 及 0.1–0.7 小节齐全（grep 核对）。
- **无占位符**：grep `{placeholder}`/`TODO`/`待补充` 无命中。
- **同类引用真实**：§3.5.3/§0.7 引用的 `examples/isnan/isnan_fwd_kernel/DESIGN.md`、`.../repro/CONST-vcmp-ne-ordered.py`、`examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md`、`examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/{hardsigmoid,hardsigmoid_kernel/hardsigmoid_fwd_kernel.py,bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py}`、`.../isinf/_isinf_fwd_kernels.py`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/DESIGN.md` 均实际存在（Glob 核对）；manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `IsinfFwdOp` 条目存在。
- **内部一致**：API 映射（§3）、内存规划（§4）、Tiling（§5）三方描述相互一致（block_size=4096 全程一致；UB 预算与 buffer 表一致；分核方案与 §6 循环一致）。

### 8. 算法优化分析 — pass ⭐

**§1.6 章节存在性 — pass**：§1.6 含 §1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 布局决策四子节，各含结论行（调研结论/优化结论/向量化结论/布局决策结论）。

**算法调研存在性与完整性 — pass**：§1.6.0 四问齐全。R1 候选表**含基线候选**（#0 源公式 `T.isinf(T.cast(x,"float32"))`）+ 候选 1–5（参考表命中行 ALG-elementwise 的 CSE/除法乘倒数/cast 链合并思路已评估）；R2「在线算法」有**明确结论**「无（天然单遍即最优）」附结构依据（逐元素无跨元素依赖）；R3 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表、口径写清（机械 `r3_metrics`=pass）；R4 逐候选六维亲和性评估（单元匹配/容量/对齐/静态边界/流水/跨核）。「无更优替代」写明调研范围（algorithm-candidates.md elementwise 命中行 + 恒等变形自查 + 本工具链谓词族实测 + 源码 §0.3/§0.4）。轻量调研（单步逐元素）深度匹配。

**调研结论正确性（独立复核）— pass**：
- **等价性独立推演**：`isinf(x) ⟺ (x==+inf)∨(x==-inf)` 在 IEEE-754 **有序**比较下成立——逐情形验证：有限/±0/次正规 → 两定向 `eq` 均假 → False；`+inf` → `==+inf` 真 → True；`-inf` → `==-inf` 真 → True；NaN → 有序 `eq` 与 NaN 亦假 → False（与 `isinf(NaN)=False` 一致）。本工具链 `eq` 的有序语义由 `examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py` 掩码直读（`eq(NaN)=0`、`eq(非 NaN)=1`）与 CG-2026-0024 佐证（`npuir_cmp` 无 unordered 模态）。独立推演结论与 `verify_equiv.py` 重跑一致。
- **复杂度算术独立复算**：候选 1 与基线访存均 = `N·in_bytes + N·1`（读 N 元素 + 写 N 个 int8 字节），无中间 GM 往返（宽化仅 UB 级，不增 GM 流量）——与 §1.6.0 R3 表一致；扫描遍数均 = 1。设计期估算（16M f16/bf16 ≈41.7µs、16M f32 ≈69.4µs、256M ≈1.09ms）按 `constants.md CONST-mte2-degradation`（16M 30.2 GB/s/核、256M 18.4 GB/s/核）× 40 向量核复算自洽（50.3MB/1.208TB/s、83.9MB/1.208TB/s、805MB/0.736TB/s），且 DESIGN 已标注「未实测外推」。
- **负向断言主动复核**：①「无在线变体」——逐元素谓词无跨元素依赖，对照 `algorithm-candidates.md` ALG-elementwise（R2「逐元素映射天然单遍」）一致，无被漏掉的已知在线候选；②「无更低复杂度算法」——候选 2/3/5 与候选 1 同访存/遍数/缓冲量级（带宽受限），无更低量级候选；③ 未出现「无化简公式」的虚假负向断言（本设计恰恰做了一次化简）。
- **并列候选裁决**：候选 1 相对 2/3/5 由 R4 明确胜出（少一次宽化 cast/少 fp32 缓冲/无特殊函数/全 dtype 原生），无判定裕度并列争议，无需实验裁决三件套。
- **弃选论证前提核对（亲自查证）**：候选 2/3 依赖「`T.vabs` bf16 ×」——已打开 `docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1 核实（bf16 列 = ×）；候选 5 依赖「`vbitcast` bf16 ×」——已打开 `docs/.../数据类型转换操作/T.vbitcast.md` §2.2.1 核实（bf16 ×）；merge 走 fp16 域依赖「`T.vor`/`T.vmax`/`T.vnot`/`T.vselect` 不支持 bool」——已逐文档核实。弃选的前提与 API 文档一致，无「核内重排被写成 GM 级」类前提错误，无与 pattern-library 实测条目矛盾的负向论断。

**调研→设计一致性 — pass**：§1.6.0 选定算法（原生 dtype 双定向有序 `eq` 取并 + 掩码→f16 `vadd` 合并→字节链）实际驱动 §1.4 算法描述、§3.1 公式拆解（输入即 §1.6.1 优化后公式）、§6 循环结构（无逐元素标量循环）。迁移任务：§0.5/§0.6 处置与调研选族一致，源优化手段 #4「fp32 宽化谓词」被升级/替换后 §11.2 有逐项意图承接说明（非静默丢弃）。

**数学等价优化完整性（四要素）— pass**：§1.6.1 两项采纳项均含「原式 → 优化后公式 → 等价性论证 → 收益量化」。无优化空间部分给出依据（单次逐元素谓词，无公共子表达式/超越函数/中间 GM 往返；掩码→byte 为表示层必需步骤）。两掩码互斥论证支撑 `vadd` 合并（和恒为 0.0/1.0）。

**等价性论证独立推演 — pass**：项 1（消除特殊函数）为精确布尔恒等，独立推演成立；项 2（原生 dtype 消除宽化）等价性 = IEEE inf/NaN 表示在宽化下不变，独立推演成立；两者均经 `verify_equiv.py` 三 dtype 0 违反机器验证（重跑与内嵌表一致），判别段 A/B 证明减法变体被正确弃选、`NOT(isfinite)&NOT(isnan)` 备选式等价。

**向量化替代完整性 — pass**：§1.6.2 覆盖全部计算点（+inf 谓词/−inf 谓词/掩码合并/掩码→字节/搬入/搬出/常量预置）并逐点给出 v-prefix 替代；交叉核对 §3.3 伪代码与 §6 循环结构，§6 不出现 §1.6.2 未涵盖的逐元素标量循环。保留 2 处标量（block 起始索引 / 尾块大小）为 block 级任务映射与动态边界，非逐元素热点，理由具体到本算子且与同族先例一致。

**不可替代理由充分性 / 替代 API 佐证 — pass**：保留项理由属「block 级索引/动态边界」类，非泛泛而谈；替代 API（`vcmp`/`vcast`/`vadd`/`copy`/`fill`）均有 `docs/` 或本工具链谓词族实测佐证，优先 v-prefix。

**向量化轴与布局决策完整性 — pass**：§1.6.3 给出**单轴豁免**论证（输入展平 1-D，仅一个非 batch 维，无广播/窗口/归约/重排候选），符合 SKILL.md「纯 GEMV/单轴算子可写『仅一个候选轴』豁免」；整除性/尾 lane/repack 代价/UB 影响逐项量化；迁移对照说明了 GPU 轴选择是输入而非结论。

**布局决策与实现一致 / 与下游章节一致 — pass**：§1.6.3 选定「I/O 原生 1-D 连续布局 + 展平轴向量化」与 §3.3 伪代码、§4 buffer 形状（均 1-D）、§6 循环结构三方一致；§3.1 以 §1.6.1 优化后公式为输入；§6 与 §1.6.2 结论一致（无未论证标量循环）。

**额外焦点核对（本任务指定）**：① 有序 `eq` 谓词对 ±inf 正确、NaN/有限→False——独立推演 + `verify_equiv.py` 三 dtype 0 违反双重确认；② 未依赖 `vcmp "ne"` 或标量 `T.isinf`——§3.3 伪代码仅出现 `vcmp(...,"eq")`（两处），全链无 `"ne"`、无 `T.isinf`，§0.6 R2/§9.1/§9.2 明确弃用并把「误用 `vcmp "ne"`」列为常见错误；③ bf16 inf 常量路径——`T.infinity` bf16 × 经文档核实，`T.cast(T.infinity("float32"),"bfloat16")` 已显式标注为**未文档化假设**（§3.4/§9.1）并附 Stage 3 首验与回退（失败则回退 fp32 宽化比较），满足「doc-backed 或显式 undocumented-assumption」要求。
