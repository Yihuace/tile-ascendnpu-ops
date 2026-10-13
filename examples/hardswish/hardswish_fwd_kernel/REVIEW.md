# hardswish_fwd_kernel 设计文档检视报告（Stage 2 — 迁移任务）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/hardswish/hardswish_fwd_kernel/DESIGN.md
- 任务类型: 迁移（harness，GPU→Ascend NPU）
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`HardswishFwdKernel`，L151-166）+ `_base.py`（`FloatUnaryKernel`/`UnaryKernel`/`make_unary_regcopy` 承载路径）+ `_builders.py`（`make_unary_regcopy` L85-121）+ `_policy.py`（`_DEFAULT_THREADS=128`）+ `ops/elementwise/activations.py`（`HardswishFwdOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（含 3 项已如实标注的非阻塞说明，见维度 2/8）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。`python3 examples/hardswish/hardswish_fwd_kernel/verify_equiv.py` 退出码 0，输出与 §1.6.1 内嵌结果表**逐项一致**：采纳项 1「除法转乘倒数（`x*clamp(...)/6` → `x*clamp(...)*(1/6)`）」`max ulp 差 = 1.000 / 违反率 = 0 / EQUIV_PASS`；逐 dtype 舍入路径 float32 = 0.000/0/EQUIV_PASS、float16 = 0.000/0/EQUIV_PASS、bfloat16 = 0.000/0/EQUIV_PASS；总体结论「全部 EQUIV_PASS — 可采纳」。**无 EQUIV_FAIL**，脚本缺失/执行失败均不成立。
- **design_calc_check: `python3 .agents/tools/design_calc_check.py --design examples/hardswish/hardswish_fwd_kernel/DESIGN.md`** 输出 JSON `checks` 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）——**人工已核**：fp16/bf16 逐 buffer 求和 = `16384(x_ub)+32768(x_f32_ub)+32768(buf1_ub)+32768(buf2_ub)+16384(y_ub)` = **131072B（128KB）**，与 §4.5「总计（fp16/bf16 路径）」一致；fp32 全 buffer 显式 = `32768×5` = **163840B（160KB）**，与 §4.5 一致；fp32 复用链（`x_f32_ub`→`x_ub`、`y_ub`→`buf2_ub`）= `32768×3` = **98304B（96KB）**，与 §4.5 一致。三行逐 buffer 行求和闭合，无混合字节口径偏差。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API in this design"）——**人工已核**：纯 Vector 算子，无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope(Cube)`，L0C 不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 8192"）——**人工已核**：`declared logical cores = 8192` 为解析伪影（8192 是 `block_size`，§5.5 无该逻辑核数）。§5.5 分核算术独立复算自洽：逻辑核数 = `ceil(N/8192)`（smoke-1m 1048576→128、mbv3-stage2 9633792→1176、mbv3-stage3 6021120→735、fc-wide 8388608→1024）；物理核数实查 `NPUUtils.get().get_aicore_num()=20`，纯 Vector ×2 = 40；`num_kernels = min(num_logical, 40)`，`num_local_tasks = ceil(num_logical/40)`（128→4、1176→30、735→19、1024→26），均与 §5.5 表一致，边界全静态。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 全部源文件（`activations.py` / `_base.py` / `_builders.py` / `_policy.py` / `ops/elementwise/activations.py`）后逐项核对，逐条附源码证据。**

- **语义理解正确性**：§0.1「`y = x * clamp(x+3,0,6) * (1/6)`，中间 fp32」与源码 `HardswishFwdKernel.op_func`（`activations.py` L159-166）逐字一致——`three/six/zero/one_sixth = T.cast(..., "float32")`（四常数 fp32 上抛），`clamped = T.min(T.max(x + three, zero), six)`（**普通** min/max，非 `nan_min/nan_max`），`return x * clamped * one_sixth` ✓。dtype 语义「输入 fp16|bf16|fp32、输出 same_as(input)、中间 fp32」与 `FloatUnaryKernel`（`_base.py` L500-503，`SUPPORTED_DTYPES=FLOAT_DTYPES`、`OUTPUT_DTYPE=None`）一致 ✓。边界语义（NaN/±inf 由末尾 `x*` 因子天然传播）与源码选择一致 ✓。
- **算法解读完整性**：§0.3 八步（host 展平 → 分块装载 → `x+3` → `max(·,0)` → `min(·,6)` → `x*clamped*(1/6)` → 分块写回 → host 回塑）覆盖源码全部计算语句；`make_unary_regcopy`（`_builders.py` L85-121）的 `contiguous = _narrows(dtype, out_dtype) and N % block_size == 0`（L96）——hardswish 输出 dtype 与输入相同，`_narrows`（L35-37，`DataType(out_dtype).bits < DataType(dtype).bits`）为 False，故走 fragment 全块分支（L112-117：`alloc_fragment` → `T.copy` 装载 → `T.Parallel(threads, npt)` 逐元素 `op_func` → `T.copy` 写回），§0.3「不触发窄化分支，一律走 fragment 全块路径」**核对成立** ✓。host 侧（`UnaryKernel.forward` L253-256 `flat(x)`→`_compiled_fn`→`reshape(x.shape)`；Op 层 `_UnaryActivationMixin` inplace `copy_`）无遗漏/无臆造 ✓。
- **优化手段覆盖完整性**：§0.4 六项（register_copy / coalesced 向量化 / 除法转乘倒数 / fp32 上抛 / `threads×npt` / NaN 免处理 clamp）逐项对应源码（`DEFAULT_STRATEGY="register_copy"` L174；`_DEFAULT_THREADS=128` `_policy.py` L12；`one_sixth`；`T.cast fp32`；`T.Parallel(threads, npt)`；普通 `T.min/T.max`），硬件依赖标注正确，未把实现层优化误判为算法层 ✓。
- **耦合性判定合理性**：§0.5 每条目四态处置（保留 / 等价替换 / 重新设计 / 舍弃）齐全——计算语义、逐元素单遍、除法乘倒数、fp32 上抛、NaN 免处理 clamp 均「保留」（纯算法层）；`threads×npt`「重新设计」（CUDA 线程模型 → 一维 Kernel）；fragment/coalesced「等价替换」（UB `T.copy`）。**无舍弃项**，与 ascend-constraints「仅支持一维 Kernel」及 MTE 连续搬运能力不矛盾 ✓。
- **NPU 重设计可行性**：§0.6 R1（`threads×npt` → 一维 `T.Kernel(num_kernels, is_npu=True)` + `block_size` 折叠 + Vector 整块 v-prefix）含「源方案 → NPU 新算法 → 语义保持论证」三要素；语义保持论证以「每元素独立计算、无跨元素依赖、无规约」为据，且指出求值链 `x+3 → max → min → x*clamped → ×1/6` 全程 fp32 与源逐步一致，满足一维 Kernel / 内存层级 / 对齐 / 静态边界约束 ✓。§0.6 明确「无需 NaN 重设计项」，与 §0.5「NaN 免处理 clamp — 保留」一致 ✓。
- **设计与决策一致性**：§11.1/§11.2 逐项核对 §0 → §1–§10，且 §0.4 各优化手段的意图承接有说明（register_copy/coalesced → UB `T.copy`；`threads×npt` → Vector 整块；除法乘倒数 / fp32 上抛 / NaN 免处理直接保留），无「§0 判重设计、§3 照抄源方案」脱节 ✓。
- **golden 独立性**：§8.1 golden 以 §0.1 源语义为唯一依据移植 `torch.nn.functional.hardswish`（`x.float()`→`F.hardswish`→`.to(x.dtype)`），**不复刻** §0.6 的 NPU 算法（不含 vclamp/vmul fp32 链 / persistent 结构），与源仓参考实现 `examples/TileOPs/tests/ops/test_hardswish.py` 的 `ref_program = F.hardswish(x.float()).to(x.dtype)`（L38-39）一致，验证独立性成立 ✓。
- **专项复核（本 op 特有，任务要求）——「无 NaN 处理链」断言的源码与 torch 语义核对**：
  1. **源码证据**：`HardswishFwdKernel.op_func`（L165）用普通 `T.min/T.max`；同族 `HardsigmoidFwdKernel.op_func`（L183）用 `nan_min(nan_max(...))*one_sixth`——两者**确实不同**，且 Hardsigmoid docstring（L174「A NaN input gives NaN, as in torch」）显式说明其需 NaN 专用链。hardswish 输出含 `x *` 因子，源码因而无需专用链，§0.1/§0.4#6/§0.6 的解读与源码一致 ✓。
  2. **torch 语义实测**（本机 `F.hardswish`，fp32）：`NaN → NaN`、`+inf → +inf`、`-inf → NaN`、`-3 → -0.0`、`0 → 0.0`。
  3. **npuir 内核端**：计算链为 `vadd(x_f32,3)` → `vclamp(·,0,6)` → `vmul(x_f32 × buf2)` → `vmul(·×1/6)`（§3.3）。即便 `vclamp`（`vmax/vmin`，`CONST-vcmp-ne-nan-unordered` 记「丢 NaN」）把 NaN 夹成 0/6，随后的 `T.vmul(x_f32, buf2)` 中 `x_f32` 仍为 NaN，`NaN * 0 = NaN` → 输出 NaN；`-inf` → `vmul(-inf, 0) = NaN`；`+inf` → `vmul(+inf, 6)*(1/6) = +inf`。**三者与 `F.hardswish` golden 逐一吻合**，故「无 NaN 单点恢复链」的结论成立 ✓。
  4. 说明：`CONST-vcmp-ne-nan-unordered`（constants.md L245）把「hardswish 族」列为该 vcmp/vselect 技术的**可适用范围**（技术可用），并非「必需」；本设计「无需该链」与源算法（普通 min/max + `x*`）一致，不构成矛盾。

### 1. API 可行性 — pass

- API 存在性：§3.2/§3.4 列出的每条 API 均在 `docs/Tilelang.language/` 找到文档佐证，**亲自打开文档核对签名与 dtype**：`T.vadd(src0, src1, dst)`（`数学操作/T.vadd.md`：src1 可为 scalar；fp16/fp32 ✓、bf16 ×）✓；`T.vmul(src1, src2, dst)`（`数学操作/T.vmul.md`：src2 可为 tensor/scalar；fp16/fp32 ✓、bf16 ×）✓；`T.vclamp(src, dst, min_val, max_val)`（`比较操作/T.vclamp.md`）✓；`T.vcast(src, dst, round_mode)`（`数据类型转换操作/T.vcast.md`：f16→f32 仅 rint；f32→bf16 含 rint）✓；`T.copy` 双显式 slice（`内存操作/T.copy.md`：切片语法合法、尾块 extent 推断规则明确）✓；`T.ceildiv`（`数学操作/T.ceildiv.md`）✓。
- 与 §3.3 映射逐一核对：`T.vadd(x_f32_ub, 3.0, buf1_ub)`（标量在 src1 ✓）、`T.vmul(x_f32_ub, buf2_ub, buf1_ub)`（tensor×tensor ✓）、`T.vmul(buf1_ub, one_sixth, buf2_ub)`（标量在 src2 ✓）、`T.vclamp(buf1_ub, buf2_ub, 0.0, 6.0)` ✓、`T.vcast(..., round_mode="rint")` ✓。**无 vbrc/vcmp/vselect**（本算子无 NaN 专用链），与 §3.4 一致。
- API 与模式匹配：Developer 模式，全部为自动同步 API，无手动 `set_flag/wait_flag` ✓；v-prefix 优先 ✓。
- 非阻塞说明：§3.3 用 1-D `(block_size,)` buffer（docs 推荐 2-D），但同族已验证先例 `mish_kernel/mish.py` 与 `hardsigmoid_fwd_kernel.py` 均为 1-D buffer，佐证充分，**不阻塞**。

### 2. 内存层级规划 — pass

- 搬运路径：§4.4 GM↔UB 两级纯 Vector，数据 GM→UB 一次读入、UB→GM 一次写出，无 GM→L0 越级、无中间 GM 往返 ✓。
- UB 预算（**独立逐 buffer 复算，见「机械复核」**）：fp16/bf16 = **128KB**、fp32 全 buffer 显式 = **160KB**、fp32 复用链 = **96KB**，三行与 §4.5 总计行及逐 buffer 行闭合，无 VP-2026-0009 混合字节口径偏差 ✓。
- 对照 UB 容量 192KB（`CONST-capacity-910B2C`）：fp16/bf16 128KB、fp32 全 buffer 160KB 均 < 192KB；上偏系数引用 `CONST-capacity-910B2C`「auto-multi-buffer=false 下 BishengIR 实际需求 ≈ 手工预算 ×1.10–1.12」（fp16 128×1.12≈143KB、fp32 160×1.12≈179KB）✓。
- Buffer 层级标注：§4.3 五个 buffer 均标 UB ✓。
- 非阻塞说明（UB 膨胀假设）：`design_calc_check` 的 `capacity_constants.ub_inflation = 1.7`（auto-multi-buffer **默认**路径）。本设计基线用 `T.serial`（非 `T.Pipelined`、显式 `T.alloc_ub`），与已通过检视并落地的同族 `hardsigmoid_fwd_kernel`（fp16 路径 136KB，采用同一 ×1.10–1.12 口径）一致；若 Stage 3/4 启用 `T.Pipelined` 双缓冲（§6.3 Stage 4 可选）则需按膨胀后预算复核——此为 Stage 4 事项，**不阻塞 Stage 2**。设计已如实标注该前提。

### 3. Tiling 策略 — pass

- Block 划分：§5.2 `block_size=8192`，理由充分（mish Stage 4 实测最优值 `perf_opt/mish.py` `BLOCK_SIZE_CAST=BLOCK_SIZE_FP32=8192` 已亲自核对；32B 对齐；发射开销摊薄引用 `CONST-vector-launch-overhead`）✓。
- 对齐约束：8192×2B=16KB、8192×4B=32KB 均 32B 整数倍；整除 fp16/bf16 ×8、fp32 ×4 向量宽 ✓。
- 非整除处理：§5.4/§6.4 `tail_size = T.min(block_size, remaining)` + `block_id < num_logical` guard，仅 copy 有效元素 ✓。
- 分核策略三要素（`core-split-strategy.md` §1）齐全：
  - ① 逻辑核数计算 `num_logical = ceil(N/block_size)`，四 workload 算术**独立复算一致**（见「机械复核」core_split 人工核）✓。
  - ② 物理核数依据：`NPUUtils.get().get_aicore_num()` **实查 = 20**，纯 Vector 翻倍 = 40，记录了查询代码与实际返回值；与 `CONST-aicore-910B2C`（24）的差异已如实标注为设备型号差异，符合「一律以实查为准」✓。
  - ③ 规模判定与分核方案：小规模（逻辑核数 ≤ 40）「无需适配」；中大/极大规模固定启动 40 核 + 核内 `T.serial` 串行 `num_local_tasks = ceil(num_logical/40)`，内核总数 = 物理核数整数倍（1×），无负载不均，边界为编译期常量 ✓。
- 与 §5.5 表逐行核对：见「机械复核」，无矛盾。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（源即一维 `T.Kernel(ceildiv(N, block_size))`，`_builders.py` L100）✓。
- GEMM 非整除：不适用（无 GEMM）✓。
- L0C 溢出：不适用（无 Cube，`design_calc_check.l0c_budget` skip 已人工确认为不适用）✓。
- GPU 专用 API 改写：§3.5.1 列出 `T.get_thread_binding`/`T.vectorized`/`T.unroll`（`_builders.py` L102/105/109）+ bf16 原生算术缺失，均有转换方案 ✓。

### 5. 循环与同步 — pass

- 循环结构：§6.1/§6.2 明确 `T.Kernel`（核级并行）+ `T.serial(num_local_tasks)`（核内串行）+ Vector 整块 v-prefix（元素级），与 §5.5/§1.6.2 一致 ✓。
- 同步与模式匹配：§7 Developer 自动同步，无手动同步点，与 §2.1 模式选择一致 ✓。
- 尾块处理：§6.4 完整（tail_size + guard）✓。

### 6. 验证方案 — pass

- Golden 函数：§8.1 给出 `golden_hardswish`（torch `F.hardswish`，fp32 opmath），与源仓测试基准一致 ✓。
- L0 测试计划：§8.3 `N=1048576`、fp32/fp16/bf16 三 dtype、有限 `randn` 输入、`assert_close` ✓。
- 精度标准：§8.2 fp16 1e-3/1e-3、fp32 1e-5/1e-5、bf16 1.6e-2/1.6e-2——**与源仓 `test_hardswish.py::_get_tolerances`（L18-24）逐值一致**（亲自核对）✓。
- 边界说明：§8.3「特别说明」正确指出本算子无 NaN 专用链、NaN 由 `x*` 传播、且不得套用 hardsigmoid 的 NaN 断言，与维度 0 专项复核结论一致 ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：模板 11 章节齐全；迁移类 §0 含 0.1–0.7 全部小节 ✓。
- 无占位符：全文无 `{placeholder}`/`TODO`/`待补充`（grep 已核；§10.2 的「⬜ 待实现（Stage 3）」为交付清单的正常未来项）✓。
- 同类引用真实：§3.5.3 四条引用路径 **Glob 均存在**（`mish_kernel/mish.py`、`mish_kernel/perf_opt/mish.py`、`examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md`、`lerp_tensor_kernel/_make_lerp_tensor_kernel.py`）✓。
- 内部一致：API 映射（§3.2）、内存规划（§4.3-4.5）、Tiling（§5.2/5.5）三处描述相互一致，无矛盾 ✓。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：含 1.6.0 算法调研、1.6.1 数学等价优化、1.6.2 向量化替代分析、1.6.3 向量化轴与布局决策四子节 ✓。
- **算法调研存在性与完整性（§1.6.0）**：调研四问齐全——R1 候选表**含基线候选**（#0 源公式，#1 relu6 原生式、#2 除法乘倒数、#3 分段式）且标注是否纳入 R3；R2 有明确结论「无在线/离线之分（天然单遍）」并给结构依据；R3 复杂度对比表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表、口径写清；R4 逐候选六维硬件亲和评估。「无更优替代」写明调研范围（`algorithm-candidates.md` elementwise 命中行、恒等变形自查、pattern-library elementwise、源码 §0.3/§0.4）✓。
- **调研结论正确性（独立复核）**：
  - ① 等价性推演：见 §1.6.1 条。
  - ② 复杂度算术独立复算：FLOPs = **4N**（add 1 + two-sided clamp 1 + mul 1 + 缩放 1），与源测试 `test_hardswish.py` L81 `flops == 4 * N` 断言一致；访存 = **2N·elem_bytes**（读 N + 写 N），与源测试 L82 `mem_bytes == 2 * N * elem_bytes` 断言一致；扫描遍数 = 1、中间缓冲峰值 = O(block_size)（UB 驻留、**无中间 GM 往返**），复算成立 ✓。
  - ③ 亲和性判断与 `ascend-constraints`（一维 Kernel）及 pattern-library 实测条目不矛盾，负向淘汰（候选 3）有佐证 ✓。
  - ④ **负向断言复核**：对照 `algorithm-candidates.md` 的 `ALG-elementwise` 命中行（R1：公共子表达式消除 / 除法乘倒数 / cast 链合并；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返；R4：单遍 Vector 流）——**无被漏掉的已知候选**：hardswish 为单步逐元素，无归约（不涉 Welford/分块合并）、无 softmax 族、无 attention；CSE（`x+3` 仅用一次）、cast 链合并（仅一次上/下 vcast，无冗余链）已在 §1.6.0 R1/§1.6.1 结论中处理。「无更优于基线的化简公式」复核**成立**。
  - ⑤ R3 并列候选：四口径全同 → 无并列争议，无需实验裁决路径 ✓。
- **调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构（单遍、无逐元素标量循环）与 §1.6.0 选定算法（逐元素直接映射、fp32 中转、无 NaN 专用链）**同口径一致**，无「调研选 A 设计做 B」；源优化手段无被替换，故无意图承接中继缺失 ✓。
- **数学等价优化完整性（§1.6.1）**：采纳项 1（除法转乘倒数）具四要素「原式 `x*clamp(x+3,0,6)/6` → 优化后 `x*clamp(x+3,0,6)*(1/6)` → 等价性论证 → 收益量化（N 次除法 → 1 次编译期倒数 + N 次乘法）」，无其他可采纳项（无 CSE / 无超越函数 / 无 cast 链冗余 / 无中间 GM 往返）✓。
- **等价性论证正确性（独立推演）**：`/6` 与 `*(1/6)` 公共因子 `(x*clamp)` 与乘法次序不变，仅缩放算子形态不同；`1/6` 舍入后两式差值 ≤1 ulp，落在 §8.2 容差内。**verify_equiv 重跑与 §1.6.1 内嵌表逐项一致（EQUIV_PASS，无 EQUIV_FAIL）**，推演与机器验证互补成立 ✓。
- **向量化替代完整性（§1.6.2）**：覆盖全部计算点——5 处逐元素计算（`x+3` / clamp / `x*clamped` / `×(1/6)` / cast）**全部向量化**（替代 API `T.vadd`/`T.vclamp`/`T.vmul`/`T.vcast` 均有 docs 与 examples 佐证）；保留 2 处标量（block 起始索引、尾块大小）并给出可接受理由类别（block 级任务映射 / 动态边界，非逐元素热点）。与 §3.3 伪代码、§6 循环结构**交叉核对一致**（§6 不出现逐元素标量循环）✓。
- **不可替代理由充分性**：两处保留标量理由具体到本算子（每 block 一次、非逐元素热点、mish 同款尾块处理），非泛泛而谈 ✓。
- **替代 API 可行性**：替代方案 API 均有 `docs/Tilelang.language/` 佐证（见维度 1），优先 v-prefix，无幻造 API ✓。
- **向量化轴与布局决策完整性（§1.6.3）**：本算子展平为 1-D `(N,)`，仅一个非 batch 维（展平轴），给出**单轴豁免说明**（无广播轴/窗口轴/归约轴/重排布局候选），符合 SKILL「纯单轴算子可写明『仅一个候选轴』豁免」；整除性（8192 = 1024×8 = 2048×4）、无 repack、UB 影响、迁移对照（GPU `threads×npt` 是 CUDA 线程模型，非 NPU 结论轴）齐备 ✓。
- **弃选论证前提核对**：§1.6.1/§1.6.3 无 API 类否决项（无「重排代价过高」类 GM 级 transpose 误述）。§1.6.0 R4 对候选 3（分段/条件式）的淘汰理由为「`T.if_then_else`/条件域在 npuir 上会标量化」，引 `pattern-library traps`「条件构造标量化」类条目——**前提亲自核对一致**：`traps-compiler.md` 含 `TRAP-parallel2d-bm-ge2-scalarize`（L215-229，2D `T.Parallel` 融合 `if_then_else` 整体标量化）等条目，且源 `MishFwdKernel.op_func` docstring（`activations.py` L201「Clamping rather than selecting on it also keeps the element loop vectorised, which `T.if_then_else` does not」）**独立佐证**该前提，淘汰有理据 ✓。
- **布局决策与实现一致**：§1.6.3 选定「I/O 原生 1-D 连续布局 + 展平轴向量化」，§3.3 伪代码、§4.3 buffer（全 1-D）、§6 循环结构三方一致，无重排路径需实现 ✓。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不再出现逐元素标量循环）✓。
- 非阻塞说明：§1.6.1 将「优化前式 `/6`」标注为「torch `F.hardswish` 语义」，但源 docstring（`activations.py` L154-156）称 `F.hardswish` 本身即按乘倒数实现（finite inputs 逐位一致）——该处归因为表述性小事，两式数学恒等且 verify_equiv 已证等价，**不阻塞**。

---

## Review Result

- verdict: **通过（结论: 通过）**
- per-dimension:
  - 维度 0 源算子理解与迁移分析: **pass**（语义/算法/优化/耦合/重设计/一致性/golden 独立性 全部 pass；专项「无 NaN 链」断言经源码 + torch + npuir 链三方核对成立）
  - 维度 1 API 可行性: **pass**
  - 维度 2 内存层级规划: **pass**（UB 逐 buffer 复算闭合；UB 膨胀假设非阻塞说明）
  - 维度 3 Tiling 策略: **pass**（分核三要素齐全，算术复算自洽）
  - 维度 4 技术约束检测: **pass**
  - 维度 5 循环与同步: **pass**
  - 维度 6 验证方案: **pass**（容差与源仓逐值一致）
  - 维度 7 完整性与一致性: **pass**
  - 维度 8 算法优化分析: **pass**（调研四问齐全、负向断言复核无漏候选、复杂度复算一致、等价性独立推演 + verify_equiv 一致、向量化覆盖完整、单轴豁免成立、弃选前提核对一致）
- mechanical recheck outputs:
  - `verify_equiv.py`: PASS（1.000 ulp / 0 违反 / EQUIV_PASS；三 dtype 全 EQUIV_PASS；与 §1.6.1 内嵌表一致；无 EQUIV_FAIL）
  - `design_calc_check.py`（`--design`）: ub_budget=skip（人工已核三行求和闭合：128KB/160KB/96KB）、l0c_budget=skip（无 Cube，不适用）、core_split=skip（解析伪影，人工复算自洽）、r3_metrics=pass（四口径齐全）
- issues: 阻塞级 0；建议级 0（3 项非阻塞说明：UB 膨胀假设、1-D buffer、§1.6.1 归因表述）
- blocking_issues: 0

## Stage Result
- stage: 2
- project: hardswish
- operator: hardswish_fwd_kernel
- output: examples/hardswish/hardswish_fwd_kernel/REVIEW.md
- is_migration: true
- source_op_path: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py
- conclusion: 通过
- validation: pass
- validation_details:
  - 结论行存在: pass
  - 结论一致: pass
  - 维度完整（迁移 9）: pass
  - 维度 0 有源码证据: pass
  - 维度 8 有推演证据: pass
  - 建议完整: n/a
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义 pass / 算法 pass / 优化 pass / 耦合性 pass / 重设计 pass / 一致性 pass / golden独立性 pass
- algo_opt_findings: 调研完整性 pass / 调研结论正确性 pass / 调研→设计一致性 pass / 优化完整性 pass / 等价论证 pass / 向量化覆盖 pass / 不可替代理由 pass / 替代API佐证 pass / 与§3§6一致性 pass
- suggestion_count: 0
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md；.agents/skills/_shared/standards/gate-and-retry.md；.agents/skills/_shared/standards/core-split-strategy.md；.agents/skills/tilelang-op-design/references/algorithm-candidates.md；.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md, traps-compiler.md, elementwise.md
- summary: 迁移设计经亲自读源核对全部 9 维度通过，机械复核（verify_equiv + design_calc_check）一致，无阻塞级问题。
- issues: none
