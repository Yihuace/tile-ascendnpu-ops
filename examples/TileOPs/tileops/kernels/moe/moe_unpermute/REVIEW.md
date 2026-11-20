# _make_unpermute_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/moe_unpermute/_make_unpermute_kernel/DESIGN.md
- 任务类型: 迁移（harness，Developer）
- 源算子: /tmp/opencode/TileOPs-ref/tileops/kernels/moe/unpermute.py（`_make_unpermute_kernel`，L33–L113）；op 层 /tmp/opencode/TileOPs-ref/tileops/ops/moe/routed_expert/unpermute.py
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2（已随各维度内联记录，均不阻断）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。`python examples/moe_unpermute/_make_unpermute_kernel/verify_equiv.py` 执行成功，输出 10 行（5 用例 × bf16/fp16）全部 `EQUIV_PASS`、`max_abs = 0.000e+00`、`viol = 0`、`ulp = 0`，末行 `EQUIV_RESULT: ALL_PASS`——与 DESIGN.md §1.6.1 内嵌结果表**逐格一致**（含「all -1」「every 3rd slot = -1」「all same slot」「K = 1」「random permutation」五类用例）。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/moe_unpermute/_make_unpermute_kernel/DESIGN.md`
  - `ub_budget`: **skip** —— 未解析到 per-level buffer 表（advisory）。**人工复核结论：pass**——§4.5 四 buffer 86016B（14336+28672+28672+14336）@H=7168 < 196608B（192KB，`CONST-capacity-910B2C`），按 ×1.10–1.12 放大后 ~96KB 仍 ≪ 192KB，auto-multi-buffer 已关闭（§7.3）无 ×1.7 膨胀。
  - `l0c_budget`: **skip** —— 无 `block_M × block_N`。**人工复核结论：不适用**——纯 Vector，无 `T.gemm`/L0C（§5.1/§3.5.1）。
  - `core_split`: **skip** —— 未解析到 dims。**人工复核结论：pass**——§5.5 三要素齐全：逻辑核数 = T；物理核数实查 `NPUUtils.get().get_aicore_num()`，本机复跑返回 **20**，纯 Vector 翻倍 = **40**（复跑命令与返回值一致，非假设值）；极大规模固定 40 核 + 核内 `T.serial(spc)`，`spc = ceildiv(T,40)` 编译期静态，末核 `if token < T` 守卫。
  - `r3_metrics`: **pass** —— 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐全且含基线 C0。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子全文（`kernels/moe/unpermute.py` L1–L223）与 op 层（`ops/moe/routed_expert/unpermute.py`），逐项与 §0 核对。

- **语义理解正确性 — pass**。§0.1 的数学语义与源逐句一致：`acc` 为 fp32 fragment（源 L74 `alloc_fragment([hidden_size], "float32")`），`src` 为 dtype fragment（L75），`fwd_idx` int32（L68）、`topk_weights` float32（L69），沿 K 升序 loop-carried 累加（L99），最后一次 `Cast(dtype, acc*scaling)` 落盘（L103–L108）。**关键判定「输出行唯一写者 → 无跨 token 累加」经源码确认成立**：kernel 为 `T.Kernel(num_tokens, threads=threads)`（L72，一 block 一 token），每 block 唯一执行 `T.copy(out_frag, output[token_idx, ...])`（L109）；同一 token 的多专家加权和是**核内 K 串行累加**，非跨 token/跨核 scatter-add。op 层 docstring 虽写「scatter-add」（ops 文件 L1/L16）、kernel doctring 写「Scatters」（L3），但**源实现实为逐 token gather + 加权归约**，§0.1 的判定正确而非沿袭误名。EP `-1` 语义（源 L88–L96 `safe=clamp(...,0)` + 权重置 0）与 `padded_batch_sum ≥ T*K` 边界（L126）、`top_k<2` 退化 serial（L84–85）均如实转录。
- **算法解读完整性 — pass**。§0.3 的 U1–U8 覆盖源全部计算语句（L86 flat、L87 raw、L90–92 safe、L93–96 weight、L97 copy、L98–99 Parallel 累加、L102–108 out_frag、L109 store），**无遗漏、无臆造**；数据流图覆盖 `fwd_idx`/`mm2_pad`/`topk_weights`/`output` 全部 buffer；host 侧逻辑（`forward` L172–223 的 `out` 分配与校验）已列（§0.1「host 侧语义」）。
- **优化手段覆盖完整性 — pass**。§0.4 的 O1–O9 逐项对应源结构/注释：O1 一 block 一 token（L72）、O2 thread 协作（docstring L41/L49–58）、O3 128-bit `uint4`（L54 VEC=8）、O4 fp32 fragment（L74）、O5 2-stage pipeline（L84–85）、O6 EP clamp+零权（L88–96）、O7 scaling 折叠（L103–105）、O9 向量化索引行 load 进 fragment（L97）；O8 `out_idx=[]` 见 L62 与 op `forward`。层级划分（算法/优化）与源码显式注释一致，无「实现层误判为算法层」。
- **耦合性判定合理性 — pass**。§0.5 每项四态处置 + 依据，逐项抽检与 `capability-gaps.md` / pattern-library 不矛盾：O1「重新设计」（一维核超发被串行调度，core-split-strategy.md §1）；O2「舍弃」（`TRAP-threads-kwarg-noop` L136–138：npuir 下 `threads=` 不生成 threadIdx 绑定）；O5「重新设计」（CG-2026-0001(a)：`T.Pipelined` 体内 runtime `if` → SIGSEGV，实测条目 L139–155）；O6「等价替换」（CG-2026-0028(c) 禁标量 `T.if_then_else`）；「跨 token scatter-add」项明确标注为**非源、新引入候选**并因「语义不需要 + 无原子原语」舍弃，界定清晰。
- **NPU 重设计可行性 — pass**。R1–R6 每项含「源方案 → NPU 新算法 → 语义保持论证」：R1 动态 slice `[raw:raw+1,0:H]` 即第 `raw` 行 + `vcast/vmul/vadd` 为源 `Cast + acc += Cast(src)*w` 的向量形态；R2 EP 跳过（§1.6.1 E1）；R3 分核只改任务归属不改结果，token 间独立；R4 online 为**必需**（两阶段 fp32 `K·H·4 = 229KB > 192KB`，算术复算成立）；R6 `fwd_idx` 标量只读 GM（跨 kernel，规避 CG-2026-0027）。新算法满足一维 Kernel、UB 容量、H·2 32B 对齐等 NPU 约束。
- **设计与决策一致性 — pass**。§1.4（固定核数 + 核内 `T.serial` + 标量 GM 读驱动动态 slice + UB fp32 online）、§3.1（以 §1.6.1 优化后公式为输入）、§3.2（`T.copy(mm2_pad[raw:raw+1,0:H], ub_src)` 动态源行）、§6（token `T.serial(spc)` / K `T.serial(K)`）描述的就是 §0.5/§0.6 决策后的 NPU 算法；无「§0 判重设计、§3 照抄源方案」的脱节。伪代码 §3.3 与源差异均在 §3.5.2 逐条注明。
- **golden 独立性 — pass**。§8.1 golden 以 §0.1 语义实现（纯 PyTorch 逐 token gather + fp32 累加 + 末次 scale + 单次 cast + EP `-1` 跳过），与 `examples/TileOPs/tests/ops/test_moe_unpermute.py` 的 `_ref_moe_unpermute`（L38–56）同构并补 EP 变体（L133–163 的 skip 语义），**未复刻** §0.6 的固定核数 / 核内串行 / 标量 GM 读结构。

### 1. API 可行性 — pass

§3.2/§3.4 列出的 API 均在 `tilelang/language/__init__.py` 与 `docs/Tilelang.language/` 找到佐证，且亲自运行确认存在：`T.vcast`（`npuir_cast as vcast`，L123）、`T.vmul`（`npuir_mul as vmul`，L81）、`T.vadd`（`npuir_add as vadd`，L73）、`T.vbrc`、`T.clear`、`T.alloc_ub`、`T.gather` 均 `hasattr=True`。v-prefix 优先明确（§3.1/§3.4），`npuir_xxx` 仅作兼容。`T.vcast` 的 `round_mode` **必须关键字传参**的坑（§9.1）与前端签名一致（`tilelang/language/customize_npuir.py` L750 `npuir_cast(src, dst, size=None, round_mode="rint")`，第 3 参是 `size`），处置正确。Developer 模式用自动同步（§7.1），无手动 flag；`T.vmul` 标量 src2（`docs/.../T.vmul.md` §2.2.2 + §2.3-2/4「仅 src2 可为标量」、in-place 合法）与 §3.2 用法吻合；`T.vadd` in-place（dst==src0）合法。API 与模式匹配。

### 2. 内存层级规划 — pass

§4.3 每个 buffer 标注存储层级（`ub_src`/`ub_f32`/`ub_acc`/`ub_out` 均 UB，无 L1/L0A/L0B/L0C，符合纯 Vector 判定）。§4.4 搬运路径仅 `GM → UB → GM`，无越级直连。§4.5 UB 预算 86016B @H=7168 < 192KB（机械 skip，人工复算：7168×2×2 + 7168×4×2 = 86016，pass），并考虑 `CONST-capacity-910B2C` 放大系数后仍 ~96KB。L0C 不适用（无 Cube）。

### 3. Tiling 策略 — pass

Block 划分明确（§5.2：逻辑任务 = 每输出 token 一行，共 T 个，每核 `spc` 个）。对齐约束满足（§5.3：行宽 `H·2` 为 32B 倍数，H∈{64,256,512,2048,3072,7168} 全部满足）。非整除处理明确（token 轴尾守卫 `if token < T`，§5.3/§5.4）。**分核策略三要素齐全且自洽**（§5.5，对照 `core-split-strategy.md` §1/§2.2）：① 逻辑核数 = T；② 物理核数**实查**——`NPUUtils.get().get_aicore_num()`，本机复跑返回值 **20**，纯 Vector 翻倍 = **40**（并给出查询代码，非文档假设/经验值；与同族先例 `_make_gather_kernel_nopad.py` L59–70 的 `VECTOR_CORES = aicore_num*2 = 40` 一致）；③ 规模判定 prefill `T=4096 ≫ 40` → 极大规模，固定 40 核 + 核内 `T.serial(spc=103)`，**循环边界静态**（`spc = ceildiv(T,nk)` 编译期常量），末核 `if token < T` 守卫；不超发逻辑核。

### 4. 技术约束检测 — pass

§3.5.1 逐条核对：未用三维 Kernel（一维 `T.Kernel(nk, is_npu=True)`）；不涉及 GEMM 非整除；无 L0C 溢出；规避 GM→GM（经 UB 中转，`docs/.../T.copy.md` §2.3-5 Developer 禁 GM→GM，已核对文档）；规避 `T.Pipelined` 体内 runtime `if`（CG-2026-0001a）；规避标量消费向量产出（CG-2026-0027）；规避标量 `T.if_then_else`（CG-2026-0028c）；不需要原子（CG-2026-0029/0030 不触发）。技术约束识别与规避方案完整。

### 5. 循环与同步 — pass

§6.1 明确各循环类型与 API（核级 `T.Kernel`、token 轴 `T.serial(spc)`、K 轴 `T.serial(K)`、元素轴 `T.vcast/vmul/vadd` 向量化），并说明**未使用** `T.Parallel`（主选）/`T.Pipelined`/`T.Persistent`。§6.3 说明不用 `T.Pipelined` 的依据（CG-2026-0001(a)）。§6.4 尾块处理完整（token 尾守卫 + EP 守卫，无 GEMM padding）。§7.1/7.2 同步模式 = Developer 自动同步（无手动 flag），与 §2.1 编程模式一致。**建议级观察**：§7.2/§9.1 指出同一 `ub_src` 跨 MTE2/MTE3 复用的 WAR 属 CG-2026-0009 类风险，依赖 auto-sync + D-3 探针 6 次逐位稳定，并预置备选（显式多版本 `ub_src`，R-2）——风险已识别且有缓解路径，不阻断。

### 6. 验证方案 — pass

§8.1 给出 PyTorch 参考实现草案（golden），以 §0.1 语义为唯一依据。§8.3 的 L0 门槛测试计划给出具体规则 shape（smoke T=1/K=8/H=512、tiny、top_k=1、EP 掩码、scaling、代表 prefill ×2、多运行稳定）、dtype 与判定。§8.2 精度标准：bf16 rtol/atol 1.6e-2、fp16 1e-3，依据 `examples/TileOPs/tests/ops/test_moe_unpermute.py` L107–108（核对一致，比 skill 默认 fp16 1e-2 更严）。

### 7. 完整性与一致性 — pass

迁移类 §0 及 0.1–0.7 小节齐全；§1–§10 模板章节齐全。全文不含任何未填充占位标记（对占位符、待办标记、待补写标记逐一检索均为零命中；「待实现」仅作交付状态标记，非占位符，§10.2/§10.4 的未实现项属正常交付清单）。§3.5.3 引用的 examples 路径**真实存在**（Glob 核对）：`examples/moe_permute_nopad/_make_gather_kernel_nopad/_make_gather_kernel_nopad.py`、`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py`、`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/DESIGN.md`。§0.7 引用的源 op / 测试 / manifest / D-3 探针 / verify_equiv 均存在。API 映射、内存规划、Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass（可机检子集：verify_equiv 重跑 ALL_PASS + r3_metrics pass）

- **§1.6 章节存在性 — pass**：含 1.6.0（调研四问）/1.6.1（数学等价优化）/1.6.2（向量化替代）/1.6.3（向量化轴与布局决策）四子节。
- **算法调研存在性与完整性 — pass**：R1 候选表含**基线 C0**（源算法，L363）且参考表命中行（`ALG-reduction`/`ALG-gemv`，`algorithm-candidates.md`；该文件确无 gather/unpermute 专条，设计已如实注明）已评估；R2 在线算法给出**明确结论**「有，且为必需」并给结构依据（running sum，源即 loop-carried `acc`，L374–378）；R3 复杂度表覆盖**四口径**（计算量/访存 Bytes/扫描遍数/缓冲峰值，基线在表 L385）；R4 逐候选硬件亲和评估（L399–410）。调研范围（参考表/pattern-library/源/examples 路径/CG 清单）写明，「无更优替代」有范围说明（L425–432）。
- **调研结论正确性（独立复核）— pass**：
  - *等价性*：E1「跳过 vs 零权」独立推演成立——`idx<0` 时源贡献 = `mm2_pad[0]·w·0`，fp32/fp64 中精确为 0，跳过 = 加 0，逐位相同；`verify_equiv.py` 重跑 ALL_PASS 佐证（不依赖设计自述）。
  - *复杂度复算*：逐口径独立复算成立——读 `T·K·H·2 = 470MB`、写 `T·H·2 = 58.7MB`、元数据 `2·T·K·4 = 262KB`，总 528.8MB；FLOPs `2TKH = 470M`；C4 `2·T·padded·H ≈ 1.9e12`、`P` 物化 268MB；C2 fp32 `K·H·4 = 229KB > 192KB`、bf16 114KB；UB 峰值 84KB。**未发现中间缓冲 GM 往返漏计**（本设计中间量全驻 UB，无 GM 往返；C2/C5 的中间缓冲已计入并据此否决）。
  - *负向断言复核*：逐条主动复核，均与参考表/源码/能力缺口一致，**未发现被漏掉的已知候选**——「无行 gather 向量原语」有 `docs/.../T.gather.md` §2.3「按最后一维取数」+ `TRAP-C10-strided-gather`（L29–43）佐证；「无可用原子 / 无 fetch-add」有 CG-2026-0029（标量载荷失败）与 CG-2026-0030（`atomic_add` 降级为 StoreOp 无返回值）佐证；「标量消费向量产出不可同步」有 CG-2026-0027（L1147–1184）佐证；「Pipelined 内 runtime if 崩溃」有 CG-2026-0001(a) 佐证。归约族（Welford/分块合并）与 gather 族候选空间已评估（C2/C8），K≤8 极小、online 即最优，两阶段被容量否决——断言与 `ALG-reduction` 条目无矛盾。
  - *亲和性淘汰依据*：C2 容量否决引用 `CONST-capacity-910B2C`（已核对条目）；C4 FLOPs 爆炸（复算成立）；C5 容量 + 转置代价；C6 CPU 同步；C3 无原语 + 非确定。
  - *并列候选裁决路径*：C1-varA 与 C1-varB（D-3 实测 610 vs 604µs，<2%）**未纸面拍板**，§1.6.0 给出主选 + 备选 + 实验裁决计划（代表 shape / 指标 / 判定阈值 ε=0.05 / 回写路径 / shape 分派可行性），符合「实验裁决模式」。
- **调研→设计一致性 — pass**：§1.6.0 选定 C1 实际驱动 §1.4 算法描述、§3.1 公式拆解（以优化后公式为输入）、§6 循环结构；§1.6.2/§1.6.3 与之一致；迁移侧 §0.5/§0.6 处置（重设计/等价替换/保留/舍弃）与调研选族一致，**源 O1–O9 优化手段意图逐项承接（§1.6.0 末段 + §0.6 R1/R3/R4），无静默丢弃**（O2 舍弃、O5/O9 形态替换附依据）。
- **数学等价优化完整性 — pass**：§1.6.1 对选定 C1 做系统审视，采纳 1 项（EP 跳过）具四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」（L488–490）；未采纳 5 项（scaling 折叠 / 除法降代换 / CSE / 访存削减 / 稳定化）逐项给结论与依据（L505–512），无「明显可优化项未审视」。
- **等价性论证正确性 — pass**：独立推演 E1 恒等式成立（加精确 0），舍入影响评估为「不引入额外 fp16/bf16 舍入，仍 fp32、k 升序」，与 §8.2 精度标准一致；未采纳项 1（scaling 折叠）以「舍入顺序不同 + 源契约为末次缩放」为由拒绝，论证成立。
- **向量化替代完整性 — pass**：§1.6.2 盘点 10 个循环/标量点，与 §3.3 伪代码、§6 循环结构**交叉核对无遗漏**——§3.3/§6 的每个循环（`T.serial(spc)`、`T.serial(K)`）与标量点（`raw` 读、`w` 读、尾/EP 守卫、核起点算术）均对应 §1.6.2 某行；行内元素（cast 提升 / 乘权重 / 累加 / 落盘 cast / scaling）全部向量化，**无「可向量化而未向量化」的逐元素标量循环**。
- **不可替代理由充分性 — pass**：6 处保留点逐项给具体理由（block 级任务映射 / tile 级 loop-carried 依赖 / 无行 gather 向量原语〔附 `T.gather.md` §2.3 + `TRAP-C10-strided-gather`〕/ CG-2026-0027 标量消费向量产出不可同步〔附缺口证据〕/ 动态边界控制流 / 标量操作数约束〔`T.vmul.md` §2.3-2〕），均非「实现简单/照源码写」式泛泛理由；且均为非逐元素热点。
- **替代 API 可行性 — pass**：`T.vcast`/`T.vmul`/`T.vadd`/`T.Parallel`（备选）均在 `tilelang/language/__init__.py` 与 docs 找到佐证（见维度 1），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 明确 I/O layout 为契约、核内布局为自由变量；给出三候选轴质量评分（token/H/K）；候选矩阵 L1–L5 覆盖「向量轴选择 × 索引处理 × 任务粒度」，其中 gather 类的**连续批量 gather 粒度**（L3 token 批处理）与「元素轴整行」均已枚举，逐候选有量化评分（整除性 / 累加链 / repack / UB），弃选（L4 容量 `K·H·4=229KB`、L5 容量+转置）有量化理由；L3 为 Stage 4 候选并登记 R-2，未把 I/O 契约当内部布局依据。**建议级观察**：§7.3 关闭 auto-multi-buffer 引用 `TRAP-UB-dynsubview-dominance`，该陷阱精确触发形态为「task 级 UB 行 buffer + 内层循环动态偏移 **UB** 切片」；本设计的动态偏移在 **GM 源侧**（`mm2_pad[raw:raw+1, 0:H]`），UB dst 为全宽 0 起点，与陷阱原文略有差异——但设计用「**可能**被重写为非支配 IR」的保守措辞，且该 `_PASS` 配置与同族先例一致，属安全防御性选择，不构成错误论断。
- **布局决策与实现一致 — pass**：§1.6.3 选定「元素轴 H 向量化 + token/padded 串行 + K online」与 §3.3 伪代码、§4 内存规划（buffer 形状 `(1,H)`）、§6 循环结构三方一致；累加循环内层向量维 = 元素轴 H；UB 预算按该布局计算。
- **与下游章节一致性 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（已向量化点不再出现对应逐元素标量循环；`T.serial(K)` 的保留理由与 §1.6.2 第 6 项吻合）。

## 说明

- 主选 C1-varA 与备选 C1-varB 的 D-3 实测（610µs / 866 GB/s vs 604µs / 876 GB/s）为设计期单进程均值，非 msprof（§9.4 R-5 已如实标注），不影响主选/备选裁决；`CONST-mte2-degradation` 地板比对与 R-1/R-2 风险登记逻辑核对无误。
- D-3 探针 `history_version/design_probe_unpermute.py` 亲自核对：其变体 A（vcast→vmul 标量→vadd in-place）与变体 B（单条 `T.Parallel` 融合）结构、scalar GM 读、动态 slice、`if raw>=0`、`nk=min(T,40)`、`spc` 静态边界与 §3.3 主选完全一致，golden 含 EP/scaling 变体——设计与探针无脱节。
