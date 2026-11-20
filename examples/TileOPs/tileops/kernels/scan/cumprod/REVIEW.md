# `_cumulative_kernel`（CumprodFwdOp · 前缀积 scan）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/cumprod/_cumulative_kernel/DESIGN.md
- 任务类型: 迁移（harness，Developer 模式；scan/前缀积族）
- 源算子: /tmp/opencode/TileOPs-ref/tileops/kernels/reduction/cumulative.py（`_cumulative_kernel` 的 `op_kind="prod"` 分支，L176–L254）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（均不阻塞，逐条见各维度说明）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS（ALL EQUIV_PASS，gated pass=123 fail=0，exit 0）** —— 执行结果与 §1.6.1 内嵌结果表（「合计（gated）123，0 违反，ALL EQUIV_PASS」）完全一致。重跑逐行复核：E1/E2/E3 在 fp32/fp16/bf16 × rand-unity/rand-sign/with-zero/with-neg-zero/neg-inf/pos-inf/large-finite/subnormal/all-one/gt-one-growth/mixed-sign-long 全部 `viol=0 patt_mm=0`；`[near-max-ulp float32] ratio=290 patt_mm=1 INFO(EQUIV_FAIL)`、`[near-max-ulp float16] patt_mm=0 INFO(EQUIV_PASS)`、`[near-max-ulp bfloat16] patt_mm=0 INFO(EQUIV_PASS)`（均非 gated，与 §9 R3 所述的已知边界一致）。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/cumprod/_cumulative_kernel/DESIGN.md`）JSON checks 摘要：
  - `r3_metrics`: **pass** — all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)。
  - `ub_budget`: **skip** — 未从内存规划解析到逐层 buffer 表。**人工复核结论**：§4.5 逐 workload 预算表可手工复算——`B_ELEM = 4×4+1 = 17`（4 个 fp32 大缓冲 + 1 个 bool，与 §3.3 `B_ELEM=17` 一致），最坏行 non-aligned (512,300) `bm·bn·17 = 16×300×17 = 81600 B ≈ 81.6KB（十进制）`、smoke `9×512×17 = 78336 B ≈ 78.3KB`，×1.7 膨胀 ≈ 139KB ≤ 192KB（`CONST-capacity-910B2C` UB 192KB/AIV）✓；§4.5 表内各值与公式自洽（decimal KB 口径一致）。
  - `l0c_budget`: **skip** — 无 Cube 路径 API。**人工复核结论**：本设计无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/Cube Scope，L0C **n/a** ✓。
  - `core_split`: **skip** — 未解析到分核维度。**人工复核结论**：§5.5 三要素手算通过（见维度 3）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源算子全文（`cumulative.py` 574 行）逐条与 §0 核对，结论：Stage 1 **读懂**了源算子，耦合性判定合理，NPU 重设计可行且语义保持。证据：

- **语义理解正确性（pass）**：§0.1 的 `y[i,j]=∏_{k≤j}x[i,k]` 与源 `L240–243`（`acc[i]=acc[i]*tile_f32[i,j]; out_f32[i,j]=acc[i]`）逐字一致；fp32 计算域核对 `L214/L237`（load `T.cast(...,"float32")`）、`L195`（`acc` fp32）、`L247`（写回 cast 回 dtype）与 §0.1 描述一致；单位元语义核对 `L88`（`_identity = 1.0 if prod`）与 `L227` 一致；M 尾块 guard 核对 `L210/L233/L250`。输出 shape/布局：源 kernel 写 `(M, N_padded)`、Op 层 `L445` 裁 `y[:, :N]`，§0.1/§0.2 均如实记录（非转置、同 layout）✓。
- **算法解读完整性（pass）**：§0.3 步骤表 1–6（含 3a 快路径/3b masked 路径双分支）覆盖源 prod 分支全部计算语句，无遗漏、无臆造：步 1↔`L198–199`；步 2↔`L202`（`n_tiles=N_padded//block_n`）；步 3a↔`L208–214`；步 3b↔`L215–228`（单位元 `if_then_else`）；步 4↔`L240–243`；步 5↔`L246–247`；步 6↔`L248–252`。数据流图覆盖 `shared_in/tile_f32/out_f32/acc/shared_out` 全部 buffer。**并行 scan 分支排除正确**：§0.3 指出 `use_parallel = (M<128 and N>8192 and op_kind=="sum")`（源 `L354`）且源码注释 "cumprod has no parallel implementation"（`L352–353`），三遍 `_parallel_scan_*`（`L453–572`）不属 prod 单元——与源码一致。host 侧 `lru_cache`（`L63`）与 Op 层 `movedim/reshape/裁剪` 均已列出。
- **优化手段覆盖完整性（pass）**：§0.4 九项逐项标注硬件依赖，核对源：`_SMEM_PAD=8`（`L60`）、`align_up(N,256)`（`L87`/`_primitives.DEFAULT_ALIGNMENT=256`）、`block_m × threads`（`L93/L188`）、48KB 预算启发式（`L372–394`）、`lru_cache`（`L63`）均识别；未把实现层优化误判为算法层（#2 running scan 明标「算法层」，#5 bank padding 明标硬件强相关）。
- **耦合性判定合理性（pass）**：§0.5 四态处置（保留 4 / 等价替换 1 / 重新设计 3 / 舍弃 1）逐条有依据，与 `ascend-constraints.md` §1 及 `migration-analysis.md §5.3` 映射不矛盾——NPU 无 SMEM bank conflict（#5 舍弃有理由）、一维 Kernel（#7 重设计）、纯 Vector 核数翻倍实查（§0.6-R2）、无 host pad（符合 host 侧输入约束）。舍弃项（#5）写明理由，无「舍弃后性能意图丢失」项（#2/#4/#7/#8 意图均有承接对象）。
- **NPU 重设计可行性（pass）**：§0.6 R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素，新算法满足一维 Kernel / 内存层级（仅 GM↔UB）/ 尾轴 32B 对齐约束。R1 的 log 域变换等价条件与失效边界由本维度独立推演确认（见维度 8）。
- **设计与决策一致性（pass）**：§1.4/§3.1/§6 描述的均是「log 域 `T.cumsum` + 符号分离 + 一维 persistent + raw N slice 尾块 + 加性 carry」——即 §0.5/§0.6 决策后的 NPU 算法；无「§0 判重设计、§3 抄源方案」脱节。§1.4 末尾「与源算法差异」4 条逐条回指 §0.6-R1–R4。
- **golden 独立性（pass）**：§8.1 `x.float().cumprod(dim=-1).to(x.dtype)` 与 harness `CumprodFwdTest.ref_program`（`examples/TileOPs/tests/ops/test_cumulative.py`）一致，以 §0.1 源语义为唯一依据，**未复刻** §0.6 的 log 域/符号分离/persistent 数据流（§8.1 明示）✓。

> 注（非阻塞）：§0.2 与 §3.3/§4.7 对工厂返回函数的签名表述不一致，见维度 7 建议级问题 1。

### 1. API 可行性 — pass

§3.2 每条 API 均能在 `docs/Tilelang.language/` 找到佐证，且与 Developer 模式匹配（无手动 `set_flag/wait_flag`），优先 v-prefix：

- `T.cumsum`：`docs/Tilelang.language/排序操作/T.vcumsum.md`（签名 `T.cumsum(src,dst,dim,reverse)`、`hivm::VCumsumOp`、dtype 表 fp16/fp32 √、**bf16 ×**）+ `testing/npuir/scan_ops/test_cumsum_dev.py`（Developer 合法形态）；文档 §2.4 示例仅 dim=0，dim=1 由**实机探针 P1-A**（`history_version/design_probe_p1_logdomain_scan.py`，max_abs_err 4.77e-7）补充证实——`T.vfloor` 类「导出存在但 codegen 未分发」的风险已被探针规避。
- `T.vcast(src,dst,round_mode="floor")`：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 明确 **f32→f32 支持 floor**；探针 P2 实证。
- `T.vfloor` 不可用已核实：`tl.npuir_floor` 分发仅在 `src/target/codegen_npuir_dev_a5.cc:4215`（A5），`codegen_npuir_api.cc` 无该分发，且 `docs/Tilelang.language/` 无 `T.vfloor.md`——§3.4/§9 R2 论断成立。
- `T.vabs/T.vln/T.vexp`（`数学操作/`，fp16/fp32 √、bf16 × → 升域）、`T.vcmp`（`比较操作/T.vcmp.md` §2.3「bool 不可写 GM，须片上消费」与设计一致）、`T.vmul/T.vadd/T.vsub`（`T.vadd.md §2.2.2` 明确支持行广播 `[M,N]+[M,1]`、`§2.3` 支持原地 dst）、`T.reduce_sum`（`规约操作/T.reduce_sum.md`，fp32 √）、`T.copy`（`内存操作/T.copy.md`，Developer 跨 dtype 自动 `VCast(RINT)`）、`T.Kernel(一维,is_npu=True)`+`T.serial`（`编译器提示操作/T.Kernel.md`；amax 先例）均有佐证。§1.6.2-#6 的奇偶链替代 API 有 `T.vcast floor` + `T.vmul/T.vsub` 佐证。

### 2. 内存层级规划 — pass（机检 ub_budget 为 skip，已人工复核）

- 搬运路径完整（§4.4）：`GM[x] --T.copy(slice)--> UB` → 全计算在 UB → `T.copy(slice)--> GM[y]`；无 Cube/L1/L0 参与，无越级 GM→L0，无 GM 中间往返、无 workspace ✓。
- Buffer 层级标注（§4.3）：`x_f32/w_f32/l_f32/s_f32/cond/c_l/c_k/rs` 全部标 UB（`alloc_shared`）✓。
- UB 预算（§4.5）：`B_ELEM=17`（4 fp32 + 1 bool），最坏 81.6KB × 1.7 ≈ 139KB ≤ 192KB ✓（同「机械复核」手算）。已计入小 `(bm,1)` 缓冲；对 `TRAP-UB-multibuffer-inflation` 的「漏计小 buffer」陷阱有显式规避（discipline 段 + Stage 3 `ub overflow` 报文核对）。
- L0C：无 Cube → n/a。

### 3. Tiling 策略 — pass（机检 core_split 为 skip，已人工复核）

- **Block 划分**（§5.2）：`block_n=min(N,4096)`、`block_m=max(1,min(M,UB_BUDGET//(block_n·B_ELEM)))`，具体值与理由齐备；`n_tiles=ceildiv(N,block_n)` 决定 carry 路径 ✓。
- **对齐**：尾轴 32B 由 `T.copy` slice 天然满足（§5.3）✓。
- **非整除处理**：N 非整除 `tn=T.min(block_n,N−col0)` + src/dst 双显式 slice（§5.4）；M 非整除 `real_m` + 双 slice + 垃圾行不写出 ✓（纯 Vector 亦说明）。
- **分核策略三要素**（§5.5，对照 `_shared/standards/core-split-strategy.md §1`）：
  - ① 逻辑核数：`num_logical = ceil(M/block_m)`（scan 轴 N 不跨核，行间独立），逐 workload 给出 15/32/2048/64 ✓。
  - ② 物理核数依据：**实查记录齐全** —— `from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → `20`（2026-10-13，设备 `Ascend910_9362`，torch 2.7.1 + CANN 8.5.0），纯 Vector 翻倍 = 40；并显式声明**不沿用** `CONST-aicore-910B2C` 的 24 ✓。
  - ③ 规模判定与分核方案：逻辑 ≤ 物理（smoke/non-aligned →「无需适配」）、极大规模（hidden-state/long-seq → 固定 40 核 + 核内 `T.serial` 静态上界 52/2 + grid-stride + 动态 guard）✓；**核内串行边界为工厂期静态 Python 整数** ✓；负载不均（long-seq 2:1）有说明 ✓。
  - 核数与 block 取值自洽性：`block_n=4096 → block_m=1 → num_logical=2048 > 40 → 40 核 × 52`；`block_n=512 → block_m=9 → 15 ≤ 40`；`block_n=300 → block_m=16 → 32 ≤ 40`——均与 §5.2 公式一致 ✓。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 persistent，§5.5）✓。
- GEMM 非整除：无 GEMM ✓。
- L0C 溢出：无 Cube ✓。
- 其余约束（bf16 不被 cumsum/ln/exp 支持 → 升 fp32；`T.vfloor` 不可用 → `vcast floor`；bool 不写 GM；`T.Parallel` 禁读 GM；未用 alloc 不 DCE → 路径分派拆 prim_func）均在 §3.5.1/§5.3 逐项登记并给出处理 ✓，与 `ascend-constraints.md` 一致。

### 5. 循环与同步 — pass（含 1 项建议级）

- §6.1 明确各循环类型：M 向 `T.serial(num_local_tasks)`（persistent）、N 向 `T.serial(n_tiles)`（tile）、scan 用硬件 `T.cumsum`、逐元素用向量 op、carry 源用 `T.reduce_sum` ✓。
- 同步策略与模式匹配（§7）：Developer 自动同步，无手动 set_flag/wait_flag；`T.cumsum` src/dst 异缓冲无原地竞态；carry 依赖由编译器识别——与 Developer 模式一致 ✓。
- 尾块处理（§6.4）：`real_m`/`tn` + 双 slice；`N=0/M=0` 列 §9 R6 ✓。
- **建议级问题 2（不阻塞）**：§6.2 循环伪代码恒显示 `for t in T.serial(n_tiles)`，而 §3.3 将 `n_tiles==1` 单列分支（§6.1 亦写「单 tile 时省略」）。三处语义等价（`n_tiles=1` 循环退化为 1 次），但表述不一致，建议 §6.2 与 §3.3 统一为「统一 tile 循环（单 tile 自动退化）」（更简单且与 §3.4 carry 时序一致）。

### 6. 验证方案 — pass

- **Golden 函数**（§8.1）：`golden_cumprod(x, dim=-1) = x.float().cumprod(dim=-1).to(x.dtype)`，含 PyTorch 参考实现草案 ✓，且独立于 NPU 算法（见维度 0 golden 独立性）。
- **L0 门槛测试计划**（§8.3）：10 组用例覆盖 manifest 三 workload（含长 seq 多 tile carry 路径 L0-4）、N 非对齐 L0-5、M 尾块 L0-6、M=1/N=1 退化 L0-8、角点集（0/负号/交替符号/全 1/近 1）L0-9、多维 dim L0-10 ✓。
- **精度标准**（§8.2）：fp32 atol/rtol 1e-3；fp16/bf16 5e-2——与 harness `_cumprod_tol` 契约一致 ✓；并注明机器验证实际误差远优于容差（不放松亦不收紧）。

### 7. 完整性与一致性 — pass（含 2 项建议级）

- 章节齐全：模板 11 章齐全；迁移类 §0 且 0.1–0.7 小节齐全 ✓。
- 无占位符：全文无未填充占位符或待办标记 ✓（对 DESIGN.md 做三类占位 token 检索均为空）。
- 同类引用真实：§3.5.4 引用的 `examples/amax/_simple_reduce_kernel/DESIGN.md`、`examples/TileOPs/.../reduction/_primitives.py`、`testing/npuir/scan_ops/test_cumsum_dev.py`、`docs/.../T.vcumsum.md`、logsumexp 目录等**全部存在**（逐条 Stat 核对）；§10.1 交付清单与磁盘现存文件（`DESIGN.md`/`verify_equiv.py`/`history_version/design_probe_p1,p2*.py`/`RETROSPECTIVE.md`）一致。
- 内部一致：API 映射（§3.2）/内存规划（§4）/Tiling（§5）三处相互一致（buffer 形状 `(bm,bn)`+`(bm,1)`、`B_ELEM`、分核公式与 §3.3 伪代码一致）。
- **建议级问题 1（不阻塞，Stage 3 须先澄清）**：工厂返回函数签名不一致——§0.2 与 §9 R9 声明 harness 契约是 `_cumulative_kernel(M,N,op_kind,dtype) → _func(block_m, block_n) → kernel(x)`（`examples/TileOPs/tileops/kernels/scan/cumprod/cumprod.py` 的 `_cumulative_fwd_wrapped` 确实以 `(block_m, block_n)` 调用），但 §3.3/§4.7 的伪代码写作 `def _func():`（block 参数由工厂闭包固化）。二者只能取一：若照 §3.3 实现，harness 调用会因参数不匹配立即失败。建议 §3.3/§4.7 统一为 `def _func(block_m, block_n):`（内部可用 §5.2 公式覆盖/校验传入值），并在 §3.3 注明「harness 传入的 `block_m/block_n` 与 §5.2 NPU 公式的关系（沿用或重算）」。
- **建议级问题 3（不阻塞）**：§1.6.0 roofline 表中 non-aligned `(512,300)` 行记为 `bm=15 × bn=300`、tiles=35，而 §4.5/§5.2/§5.5 均给 `bm=16`、num_logical=32（`81920//(300×17)=16`）。建议将 roofline 行改为 `bm=16`、tiles=32、发射项 `32/40×16×0.5=6.4µs`（结论方向不变：发射主导）。同属 §1.6.1 INFO 行表述——「float32 / bfloat16 … 模式不一致 1」在重跑中仅 float32 出现 `patt_mm=1`（bf16 为 0），建议拆行如实标注。

### 8. 算法优化分析 — pass（含 2 项建议级）

**§1.6 章节存在性**：含 1.6.0/1.6.1/1.6.2/1.6.3 + 附加 1.6.4 ✓。

**算法调研存在性与完整性（pass）**：四问齐全——R1 候选表**含基线候选 0（源顺序 running 乘积 scan）**且评估 C1–C6（log 域 / Hillis-Steele / 两遍并行 scan / 矩阵物化 / exp2 换底 / 排序类），与 `algorithm-candidates.md` 的 `ALG-reduction`（scan 族的规约命中行：R1「max+sum 合并扫描」、R2「分块归约 + 合并（两阶段/树形）」→ 对应设计的 C3 三遍 scan、R4「水平 vs 垂直 lane 映射」）逐条对上，候选覆盖充分；R2 在线算法**有明确结论**（前缀积天然单遍在线，C1 的跨块 carry 即该在线变体的中间缓冲 O(N)→O(tile) 形态）；R3 复杂度**四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值 + NPU 真实成本口径「向量 op 数/核」，基线在表、口径注明，机检 `r3_metrics` pass）；R4 逐候选硬件亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）并附探针常数；**调研范围**写明（候选库命中行 + `kb_search` 命中 SSD chunk scan 同族 5 条 + 本仓 examples + 探针 + 恒等变形自查 + 互联网未触发）；深度与 scan 类复杂度匹配 ✓。

**调研结论正确性（独立复核，pass）**：
- ① 等价性独立推演（与 §1.6.1 同口径）：log 域变换 `y = exp(cumsum(ln|x|))·(-1)^{cumsum(x<0)}` 对 `x≠0` 为**精确恒等**——`∏x_k = (∏sign(x_k))·∏|x_k| = (-1)^{#neg}·exp(Σln|x_k|)`（乘法群→加法群同构）；`x=0`：`ln|0|=-inf` → 前缀和 `-inf` → `exp(-inf)=±0`，与 `torch.cumprod` 遇零后恒 `±0`（含符号零）一致；`NaN/±inf` 经 `-inf+inf=NaN` 与比较判据 `NaN<NaN=False` 与 torch 传播一致。符号用 `x<|x| ⟺ x<0` 判据（`x=-0.0` 下 `-0.0<+0.0=False`，不误计；`x=NaN` 不误计），整数计数奇偶**无舍入**。推演成立，未见失效边界（`near-max` 属数值往返而非代数）。
- ② 复杂度复算：基线 op 数 `(M/bm)·N/40 = 2048·4096/40 ≈ 2.1e5 op/核`（与 §1.6.0 一致）；C1 `(M·N/(bm·bn))·op_tile/40 = 2048·16/40 ≈ 820`；C2 `820·(2·12+4)/16 ≈ 1435`——算术复算通过；中间缓冲**未漏计 GM 往返**（纯 Vector 单 kernel、无 workspace、无 GM 中间往返，§1.5 明示，与「访存量 = 2·M·N·elem、扫描遍数 = 1」口径自洽）。
- ③ 亲和性判断：C1 全链 fp32 ✓、容量 ✓、尾轴 32B ✓、静态边界 ✓、auto-multi-buffer ✓；淘汰依据与 `ascend-constraints.md`、pattern-library constants 不矛盾；容量/带宽引用 `CONST-capacity-910B2C`/`CONST-vector-launch-overhead`/`CONST-mte2-degradation`（均实存于 `pattern-library/constants.md`）✓。
- ④ 负向断言主动复核：设计**未作**「无在线变体/无化简公式」的负向断言——反而明确「R2 在线变体存在且被采用」、「任何算法至少 N−1 次乘法（信息论下界），无降低乘法次数的代数化简」这一结构性论断经复核成立（scan 为结合非交换幺半群，无代数化简空间）；候选库 `ALG-reduction`/scan 命中行无被漏掉的已知候选（Blelloch work-efficient scan 由 C3 三遍覆盖，Hillis-Steele 由 C2 覆盖）✓。
- ⑤ 并列候选裁决路径：C1 vs C2 判定裕度依赖**未文档化常数**（超越函数 vs 普通 `vmul` 的 per-op 成本比），设计**未纸面拍板**，而是在 §1.6.4 给出主选 + 完整备选结构 + 实验裁决计划（代表 shape、指标、阈值 0.9×、回写路径）✓（详见下）。

**调研→设计一致性（pass）**：§1.6.0 选定 **C1**，§1.4 算法描述、§3.1 公式拆解（输入 = §1.6.1 优化后公式）、§6 循环结构（`T.cumsum` 硬件 scan + tile 循环）三方均为 C1 log 域；迁移侧 §0.5/§0.6 的 R1 处置与调研选族一致；源算法优化手段被替换后（源顺序 scan → log 域）§0.4 各优化项**逐项承接说明**齐全（#2→R1、#3→保留、#4→E3、#5→舍弃、#6→R3、#7/#8→R2/R4、#9→保留），无静默丢弃 ✓。

**数学等价优化完整性（pass）**：§1.6.1 对选定算法 C1 逐项四要素——E1（乘法 scan→log 域加法 scan）、E2（符号奇偶分离）、E3（单位元 pad 消除 + 跨块加性 carry）、E4（全 dtype fp32 域，保留源路径 + API 必要性）；每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」，E4 为保留项但以「下界 API 必需」给出论证，符合「无新优化时仍给结论与依据」✓。

**等价性论证独立推演（pass）**：E1/E2/E3 的恒等性经独立推演成立（见上①）；容差内等价已评估 dtype 舍入影响（fp32 max rel 3.5e-6 / fp16 4.3e-4 / bf16 3.9e-3，均 ≪ §8.2 容差）；**verify_equiv.py 重跑 123 gated 全 EQUIV_PASS** 与内嵌表一致 ✓。未知/已知失效边界已登记（near-max 往返上溢，§1.6.1 INFO + §9 R3，非 manifest 负载 + 给出 `T.vmin` 截断/首元素特判缓解路径）。

**向量化替代完整性（pass）**：§1.6.2 盘点 §1.4 算法全部循环/标量计算点 #1–#14；#1–#10（行内串行 scan、幅值/log/exp、符号判据、指示量、计数前缀、奇偶链、幅值×符号、GM↔UB 搬运、carry 注入、tile 内和）**全部向量化**并附 API 佐证（`T.cumsum` 探针 P1 实证、`T.vabs/vln/vexp/vcmp/vcast/vmul/vadd/vsub/reduce_sum/T.copy` 文档佐证）；#11–#14（persistent 任务循环、block 索引算术、尾块尺寸、host 元数据）保留且理由具体到本算子（分核调度串行 / block 级索引 / 运行时 pid 依赖边界 / host 计算，均非逐元素热点）✓。与 §3.3 伪代码、§6 循环结构交叉核对——§6 循环表中**无任何逐元素标量循环** ✓。

**不可替代理由充分性（pass）**：#11–#14 理由均具体（非「实现简单/照源码写」）✓。

**替代 API 可行性（pass）**：全部替代 API 有 `docs/`/`testing/`/实机探针佐证（与维度 1 同口径）✓。

**向量化轴与布局决策完整性（pass）**：§1.6.3 按算子类别（scan = 规约类顺序依赖变体）枚举候选矩阵 6 行：水平 scan（lane→N，`cumsum(dim=1)`）vs 垂直 scan（lane→M，串行步进 N）**两种 lane 映射正交枚举** + 核内转置（dim=0 绕行）+ Hillis-Steele + 三遍并行 scan + 缓冲组织（staging 直链 vs fragment 中转）；逐候选有量化评分（整除性/尾 lane 浪费/scan 链形态/repack/UB 影响）✓。**弃选有量化理由**：#2 垂直 scan（≈2.1e5 op/核 发射灾难）；#3 核内转置（**前提被探针 P1 推翻**——`cumsum(dim=1)` 直接可用，无需 dim=0 绕行；代价机制引用 `pattern-library/layout.md PL-1.1-transpose-chain` 实测链代价，未犯「核内重排写成 GM 级 transpose」之误）；并把源码 GPU 轴选择标注为「输入非结论」✓。布局决策与 §3.3/§4/§6 一致（buffer 形状 `(bm,bn)`、向量维 = N 轴、无 repack）✓。

**实验裁决模式完整性（pass）**：§1.6.4 三件套齐全——触发前提（判定依赖未文档化常数：超越 vs 普通 op 成本比）；① 分派条件（工厂期静态谓词 + per-shape 分派可行）；② 备选 C2 完整结构（缓冲 `y_ub/sh_ub`、`B_elem=8`、静态展开移位链、分核三要素重算、数值路径说明、**区间 slice 形态的 Stage 3 首日验证**）；③ 裁决计划（代表 shape (2048,4096)/(64,32768)、指标 bench Task Duration、**阈值「C2 < 0.9× C1 则分派翻转」**、回写路径）。设计期不额外做探针的理由（探针预算 ≤2 已用于最高风险 P1/P2；op 成本比属运行期画像）合理 ✓。

**布局决策与实现一致（pass）**：§1.6.3 选定布局与 §3.3 伪代码、§4.3 buffer 形状、§6 循环三方一致 ✓。

**与下游章节一致（pass）**：§3.1 以 §1.6.1 优化后公式（`y=exp(cumsum(ln|x|))·(-1)^{cumsum(x<0)}`）为输入拆解为 11 步；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）✓。

- **建议级问题 4（不阻塞，Stage 3 需明确）**：§3.4/R7 的 carry 更新取值时序（须在 `T.cumsum` 覆盖源缓冲前取 tile 局部 log 和/计数和）尚未落定实现形态；「独立缓冲保存 raw 值」会引入第 6 个 `(bm,bn)` 缓冲（`B_ELEM` 17→21），虽因多 tile 仅在 `bn=4096/bm=1`（long-seq）出现而对 manifest 各 shape 的 UB 预算无实际影响，但 §4.5 的 `B_ELEM=17` 全局公式未涵盖该路径。建议在 §3.3/§4.5 显式选定一种形态（推荐「折叠进 cumsum 首元素」以免新增缓冲）并说明末列抽取方式。
- **建议级问题 5（不阻塞）**：探针 P1-A 仅在 `(2,64)` 上证实 `T.cumsum(dim=1)`，而 manifest 主负载 scan 轴长 4096/32768。建议在 §9 R 中登记「长 scan 轴（≥4096）的 cumsum 原语长度上限未编码验证，交 Stage 3 L0-2/L0-4 首日确认」，以显式承接该残留风险。
