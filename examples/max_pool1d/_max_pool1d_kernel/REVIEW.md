# _max_pool1d_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/max_pool1d/_max_pool1d_kernel/DESIGN.md
- 任务类型: 迁移（harness 子模式，逐函数）
- 源算子: /workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py（函数 `_max_pool1d_kernel`，第 15–68 行）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **PASS**。`/workspace/venv/bin/python examples/max_pool1d/_max_pool1d_kernel/verify_equiv.py` → `[EQUIV] ALL PASS — 117 case-groups, 0 failures`；stdout 逐行与 §1.6.1 内嵌结果表（DESIGN.md 第 612–728 行）**逐行完全一致**（`diff` 无差异）。分组计数 E1 21 / C1 63 / C4 15 / C4x 6 / C4b 12 = 117，与内嵌表声明一致。
  - **空真守卫核对**：`mismatch()`（verify_equiv.py 第 200 行）断言两侧 `numel() > 0` 且 shape 相同；逐组复算 `out_l`，全部 26 条配置的输出长度均 > 0（E1/C1 的 `(1,8,8)` k3s4p0 ceil → `out=2`；C4 的 5 条均 `out==1`；C4x 两条均 `out==1`；C4b 的 4 条 `out==in`）。**无空真行**。
  - **前置条件核对**：C4 组第 297 行 `assert c4_condition(...)`（5 条全部满足 `L_out==1 ∧ p==0 ∧ d==1 ∧ k>=L_in`）；C4x 组第 326 行 `assert not c4_condition(...)`（2 条均 `out==1` 但 `d>1`，拒绝条件成立）；C4b 组第 354–358 行同时断言 `out==in` 且 `not c4_condition`。case 配置与其声明的前置条件一致。
  - **`pool_output_dim` ceil_mode 分支配对**：第 74 行的「末窗回退一窗」分支由 E1/C1 的 `(1,8,8)` k3s4p0 ceil（`out=3 → 2`）触发，机器验证覆盖了 helper 的第二处 ceil 修正。
- design_calc_check: `python3 .agents/tools/design_calc_check.py --design examples/max_pool1d/_max_pool1d_kernel/DESIGN.md`
  - `ub_budget`: **skip**（advisory checker 未解析逐 buffer 表）→ **人工已核**：逐 buffer 复算一般路径两代表档（见下），全部 ≤ 192 KiB。
  - `l0c_budget`: **skip**（未解析 block_M×block_N）→ **人工已核**：纯 Vector 无 Cube，L0C 不适用，非缺口。
  - `core_split`: **skip**（dims 解析不足）→ **人工已核**：见维度 3。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值均在 §1.6.0 R3 表内）。
  - 人工复算（KiB = 1024 B 口径）：`G = CH·LSPAN·(2·eb+9) + BL·CH·(25+2·eb)`
    - sincnet CH=64/BL=26/LSPAN=78/fp16：逐 buffer 求和 = 113 152 B = 110.5 KiB，×1.7 = 187.8 KiB ≤ 192 ✓（BL=27 → 117 504 → ×1.7 = 199 757 B ✗，故 BL=26 成立）；
    - ecg CH=64/BL=31/LSPAN=69/bf16：114 944 B = 112.3 KiB，×1.7 = 190.8 KiB ≤ 192 ✓（BL=32 → 118 464 → ×1.7 = 201 389 B ✗，故 BL=31 成立）；
    - C4 textcnn CH=64/L_in=128：91 840 B ×1.7 = 152.5 KiB ✓；C4b 长线回退阈值 (1,8,4096) → 111 411 B ✓、(1,8,16384) → 445 645 B ✗ 回退。
    - **逐 buffer 复算值与 §4.5 表逐行一致**（`row_src/row_dst/row_f32/eq_mask/eq_f32/acc/flag/zeros/infbuf/nanbuf/out32/cfinal/acc16/acc_t`）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源文件全文（`max_pool1d.py` 348 行）、helper `common.py`（`pool_output_dim` 第 80–97 行）、抽取件 `_max_pool1d_fwd_kernels.py`（100 行）、集成包 `max_pool1d.py` 与 `.migration_meta.json`，与 §0 逐项核对：

- **语义理解正确性** — pass。§0.1/§0.2 ↔ 源码：`max_val = T.cast(float("-inf"), accum_dtype)`（源第 49 行）＝ §0.1「初值 $-\infty$」；越界跳过 `if iw >= 0 and iw < l_in:`（源第 53 行）＝ §0.1「越界 lane 跳过 ≡ padding $-\infty$」；NaN 追踪 `if T.isnan(val): has_nan = True`（源第 55–56 行）+ 末尾 `T.if_then_else(has_nan, cast(nan), max_val)`（源第 59–63 行）＝ §0.1「NaN 胜出」；`out_l = pool_output_dim(...)`（源第 28 行）＝ §0.1 输出尺寸；`accum_dtype = "float"`（源第 27 行）＝ §0.1「fp32 比较域」；dtype 门禁 `{float16, bfloat16, float32}`（源第 147–150 行，**含 fp32**；集成包第 176–180 行同）＝ §0.1；I/O 恰好 3D NCL —— manifest `pool.yaml` 第 313–314 行原文 "unlike the adaptive pool family the Op layer accepts exactly 3D NCL input (no CHW unsqueeze)"、op 层 `ops/pool.py` 第 506–509 行 `if input.ndim != nd + 2: raise`，与 §0.1「不接受 2D C×L / 无 unsqueeze」**逐字吻合**。`{+inf,-inf}` 同窗取 `+inf`（max 语义）与 §0.1 一致。
- **算法解读完整性** — pass。§0.3 步骤表 6 步覆盖源全部计算语句：host `out_l`/`total_output`（源第 28–29 行）、grid（第 38 行）、lane 扁平解码（第 40–45 行）、状态初值（第 47–50 行）、窗口扫描含 cast/isnan/max（第 51–57 行）、结果 + 写出（第 59–64 行）。**无遗漏、无臆造**；数据流图覆盖 `x`/`out` 两个 buffer；host 侧逻辑（op 层规范化、lru_cache、kernel 缓存）已列。
- **优化手段覆盖完整性** — pass。§0.4 列 11 项，与源码核对：一维 block 网格 + 扁平映射（源第 38–45 行）、`T.Parallel(block_m)` lane（第 39 行）、静态窗口循环 + 守卫（第 51–53 行）、标量 max 链 + `-inf` 初值（第 49/57 行）、标量 NaN 追踪（第 55–56 行）、fp32 比较域 + 单次舍回（第 27/64 行）、逐元素 GM gather（第 54 行）、`T.alloc_var` 标量状态（第 47–48 行）、尾任务守卫（第 41 行）、`lru_cache`（第 15 行）、`-inf` 初值 + 越界跳过（第 49/53 行）。§0.4 同声明的「源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`」经源码逐行核对**属实**（源码无任何这些构造）。
- **耦合性判定合理性** — pass。§0.5 每项四态处置 + 依据；「重新设计」项（#2/#4/#7/#8）与「保留/等价替换」项的划分与源码硬件依赖一致；舍弃项声明为「无」并在 §0.6 复述理由（源码确无 GPU 专有手段）；依据引用 `migration-analysis.md §5.3`、`docs/Tilelang.language/`、pattern-library 实测条目。
- **NPU 重设计可行性** — pass。§0.6 R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素；R1 一维 `T.Kernel`、R2 C 轴向量 `vmax` + eq 掩码、R3 UB span 载入 + 核内转置、R4 结构特化；均满足 NPU 硬件约束（一维网格 ≤2 维、无 CUDA thread、UB 驻留、静态边界）；R2 论证「C 轴是独立轴、SIMD 不改变取值集合」成立。
- **设计与决策一致性** — pass。§1–§7 描述的是 §0.5/§0.6 决策后的算法：§1.4 步 2/3、§3.3 伪代码、§5.5 分核、§6 循环均为 `(n,o_blk,c_block)` 单元 + C 轴向量 + span 载入；无「§0 判重设计、§3 照抄源方案」的脱节（对照 §3.5.2 差异表逐行一致）。
- **golden 独立性** — pass。§8.1 golden 直接使用 `F.max_pool1d`（源仓测试基准 `MaxPool1dTest.ref_program`），未复刻 §0.6 的 NPU 数据流（无 transpose / 无 C 向量化 / 无 span 载入 / 无 C4-C4b 特化），验证独立。
- **host/op 层引用路径真实性** — pass。Glob 核对 §0/§0.3/§3.5.3 引用的路径逐条存在：集成包、抽取件、`.migration_meta.json`、manifest `pool.yaml`、`tests/ops/test_pool.py`、`benchmarks/ops/bench_pool.py`、同族前作 DESIGN/实现、`docs/Tilelang.language/` 各条目均 `OK`。未把共享 `ops/pool.py` 误引为单算子文件。
- **多输出/变体函数的迁移范围** — pass。`.migration_meta.json` 的 `extracted_functions = ["_max_pool1d_kernel"]`；§0 显式声明 `_max_pool1d_with_indices_kernel`（源第 212–288 行）与 `MaxPool1dWithIndicesKernel` 不在范围。Grep `indices` 命中位置全为范围排除说明（第 5/13/14/33/49/54/91/162 行），**无 indices 输出相关内容，无范围越界**。

### 1. API 可行性 — pass

§3.2 列出的每条 API 均有 `docs/Tilelang.language/` 佐证或 examples 实测用法（亲自核对文档）：`T.Kernel`（编译器提示操作/T.Kernel.md §2.3）、`T.vbrc`（shape操作/T.vbrc.md §2.2.2）、`T.clear`（内存操作/T.clear.md）、`T.copy`（内存操作/T.copy.md）、`T.transpose`（创建操作/T.transpose.md §2.3 —— 文档明确「只支持转置两个轴」）、`T.vcast`（数据类型转换操作/T.vcast.md §2.2.1，含 `f16/bf16→f32 rint`、`i1→f32 rint`）、`T.vmax`（比较操作/T.vmax.md §2.2.1，实测 doc 表 bf16 = ×，与 §3.4「bf16 须先上转」论断一致）、`T.vcmp`（比较操作/T.vcmp.md，`eq/ne` 枚举）、`T.vselect`（条件操作/T.vselect.md）、`T.vmul`/`T.vsub`（数学操作）、`T.infinity`（创建操作/T.infinity.md）、`T.reduce_max`（规约操作/T.reduce_max.md）。Developer 模式选自动同步（§7）。v-prefix 优先（`vadd/vmul/vexp/vcast/vbrc` 形态）满足 AGENTS.md API 约定。`T.reduce_min` 无独立文档页，设计如实标注（`ls docs/Tilelang.language/规约操作/` = {T.reduce.md, T.reduce_max.md, T.reduce_sum.md}），并以同族已验证 kernel 实测用法佐证（`max_pool2d/_max_pool2d_kernel.py` 第 516–517 行、`adaptive_max_pool2d/_adaptive_max_pool2d_kernel.py` 第 370–371 行均实测使用 `T.reduce_min`）——属文档缺口，非 API 不存在。

### 2. 内存层级规划 — pass

§4.4 给出完整 GM→UB→GM 路径，无 GM→L0 越级（纯 Vector 无 L0/L1 路径）。§4.3 逐 buffer 标注 UB 层级。§4.5 UB 预算经人工逐 buffer 复算（见「机械复核」段）：一般路径两代表档 110.5 / 112.3 KiB 原始，默认档 ×1.7 → 187.8 / 190.8 KiB ≤ 192 KiB；退化路径 C4/C4b 有独立不等式与回退（失败即回退一般路径，语义不变）；容量判据为**带字节数的不等式**而非代表 shape 外推，并给出 BL±1 的边界核验。KB/KiB 口径统一为 KiB 且并列原始字节便于复算，无混用。

### 3. Tiling 策略 — pass

- §5.2 给出 `CH`/`BL`/`LSPAN` 具体值与理由（`CH` 为 8 的倍数 ≤min(C,64)、`BL` 越大 halo 越小且受容量约束）。
- 对齐约束：向量轴 C 恒为 8 的整数倍（fp16/bf16 向量宽 8、fp32 宽 4，8 为两者公倍数）；`L_in`/`LSPAN` 仅作转置后 row 长度、不作向量轴。
- 非整除：任务/通道/输出块/窗口四类尾块分别由 `uid` 守卫 / `real_c` / `BL_real` / `iw` 守卫处理（§6.4）。
- **分核策略三要素齐全**（对照 `_shared/standards/core-split-strategy.md` §1）：
  - ① 逻辑核数：单元粒度明确定义（一般 `(n,o_blk,c_block)`；C4/C4b 整线 `(n,c_block)`），`num_logical` 唯一定义源在 §5.2/§5.5①，代表值 smoke 1 / sincnet 3 392 / textcnn 256 / ecg 1 056（人工复算：sincnet 32×53×2=3392、ecg 16×33×2=1056、textcnn 64×4=256 均一致）。
  - ② 物理核数依据：**实查**。设计记录查询代码 `NPUUtils.get().get_aicore_num()`；本次检视用 `/workspace/venv/bin/python` 独立重跑 → **返回值 20**（Ascend910B3），与设计一致。纯 Vector 翻倍 20×2=40（依据 `PL-1.21-vector-persistent-split` 的 `vector_cores = get_aicore_num()*2` 模板）；设计明确拒绝复用 `CONST-aicore-910B2C`=24。
  - ③ 规模判定与分核方案：逐 workload 判定（≤物理核 → 无需适配；极大规模 → persistent `num_kernels=min(num_logical,40)`、`num_local=ceil(num_logical/num_kernels)` 为 Python 编译期常量、`if uid<num_logical` 守卫）。人工复算 `num_local`：sincnet ceil(3392/40)=85、ecg ceil(1056/40)=27、textcnn ceil(256/40)=7，与 §5.5 表一致；`num_kernels ≤ num_logical` 保证无空核。
- **核内串行边界 `num_local` 为静态值**（编译期常量），满足标准 §1 要素③硬要求。

### 4. 技术约束检测 — pass

无三维 Kernel（一维 `T.Kernel`，任务维度在 kernel 内解码）；无 GEMM 整除问题；无 L0C 溢出（无 Cube）。§3.5.1 已知限制逐项检查齐全。

### 5. 循环与同步 — pass

§6 明确 `T.serial`（persistent / `k_W` / `BL`，均为编译期常量上界）与逐元素向量化的选择，未使用 `T.Pipelined`（理由充分：单阶段流、窗口循环次数少）。§7 同步策略与 Developer 模式一致（自动同步，无手动 `set_flag/wait_flag`）。尾块逻辑见 §6.4。

### 6. 验证方案 — pass

§8.1 含 golden 草案（`F.max_pool1d`，独立于 NPU 算法）；§8.2 精度标准 `atol=rtol=0`（三 dtype），偏离模板通用档 1e-2/1e-4 **有依据**（max 是被选元素原值、无累加无舍入，源仓 GPU 套件同用 0）；§8.3 含 L0 门槛计划 L0-1…L0-14，覆盖 manifest 各 shape、三种 dtype、NaN 门禁（L0-8）、±inf 门禁（L0-9）、C4 特化 + 容量回退（L0-11）、C4 dilation 守卫（L0-11b）、`ceil_mode` 回退（L0-11c）、C4b（L0-12）、通道/输出块尾（L0-13）、参数校验（L0-14）。

### 7. 完整性与一致性 — pass

章节齐全（§0 含 0.1–0.7 七小节；§1–§10）；无任何未填充占位符（对常见占位符 token 的 Grep 为空）；§3.5.3 引用的 examples/ 路径逐条真实存在；§3.5.3 引用本算子目录自身的设计文档为「同族参考」的表述已核对无诳；API 映射 / 内存规划 / Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性** — pass。含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策四个子节。
- **算法调研存在性与完整性** — pass。R1 候选表**含基线**（#0 源算法、#0' 基线向量化）且评估了 `algorithm-candidates.md` 的 `ALG-pooling` 命中行（R1 常数权重卷积 / sum-pool×常数）；R2 明确结论「有（基线本身已是单遍）」；R3 表覆盖四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），**基线在表**，口径写清（含「不含 UB 内搬运 / 中间缓冲不落 GM」）；R4 逐候选评估六项（单元匹配/容量/对齐/静态边界/流水/跨核）；调研结论含选定算法（#0'）与关键依据；「无更低复杂度算法族」附调研范围声明。调研深度与「窗口/归约类」匹配。
- **调研结论正确性（独立复核）** — pass。
  - ① 等价性：选定 #0' 与基线数学恒等（C 轴独立、L 分块只改访存），已由 E1 组机器验证（117 组逐位一致）；
  - ② 复杂度独立复算：manifest `roofline.flops = N·C·out_W·kW`、`bytes = (N·C·L_in + N·C·out_W)·eb` 与 §1.6.0 估算表逐 workload 复算一致（smoke 768 B、sincnet 27 960 320 B、textcnn 4 227 072 B、ecg 12 578 816 B；对应 0.6/22.2/3.35/9.98 µs @1.26 TB/s）；op 数复算一致（3392×168≈0.57M、256×13=3328、1056×322≈0.34M）；**无漏计 GM 往返**（设计明确中间缓冲不落 GM，纯 Vector 单 kernel）；
  - ③ 亲和性淘汰有佐证：候选 0「标量占比」引 `PL-1.2`/`layout.md` 诊断触发器的「98.3% 标量、换轴 5.6x」实测；容量/带宽淘汰引 `constants.md`（`CONST-mte2-degradation` / `CONST-capacity-910B2C` / `CONST-vector-launch-overhead`）；
  - ④ **负向断言主动复核**：候选 1/2（常数权重卷积 / sum-pool×常数）与候选 3（前缀最大值差分）为**数学不可能类**（$\max$ 非线性 / 无逆元），只须数学推演——已逐条推演且成立（候选 3 例 `x=[5,1,3]`，区间 `[1,3)` max=3 而 `P[3]−P[1]=5−5=0`，独立复核正确）；候选 7（可分离）1D 结构不适用属实。对照 `algorithm-candidates.md` 的 `ALG-pooling` 命中行与 `PL-1.22-maxreduce-nan-tracking`（`layout.md` 第 63–76 行），设计已评估全部命中候选，**未漏已知候选**；
  - ⑤ 并列候选裁决：判定裕度落在未实证常数（窄向量每-op 代价）上时，未纸面拍板，而是指向 §1.6.3 的 D-3 裁决三件套。
- **调研→设计一致性** — pass。§1.6.0 选定算法 #0' 实际驱动 §1.4（步 2 的 C 轴向量 + span 载入 + eq 掩码）、§3.1（以优化后公式为输入）、§6（累加内层向量维 = C、`kw`/`owl` serial）；迁移侧 §0.5/§0.6 处置与选族一致，源 §0.4 优化手段的意图承接在 §0.6 每项「承接」行逐条给出（#1→R1、#2→R1、#4/#5/#6→R2、#7→R3、#8→R1/R2），**无静默丢弃**。
- **数学等价优化完整性** — pass。§1.6.1 对选定算法做了系统审视（恒等变形 / 稳定化 / CSE / 访存削减 / 归约融合 / 结构特化）；采纳项 C1/C4/C4b 均有四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」+ E1 语义锚点；无优化空间处有显式结论（无除法/归一化、无高代价算子降代换、无未消解的公共子表达式、1D 无第二轴判据可外提）。
- **等价性论证独立推演** — pass（逐项）：
  - E1（越界跳过 ≡ $-\infty$ padding + 初值 $-\infty$）：$\max(-\infty,x)=x$，逐位等价，机器验证一致；
  - C1（钳制连续载入）：推演 `s0=max(0,min(base,L_in−LSPAN))` 下 `s0≤iw<s0+LSPAN ⟺ 0≤iw<L_in`（对窗口内 `iw∈[base,base+span)`）成立，且载入恒在界内——与 verify_equiv.py 第 152/156 行的逐 lane 断言一致；
  - C4（整线归约）：**独立逐变量验证输入覆盖集合**——`L_out==1 ∧ p_W==0 ∧ d_W==1` 时窗口 = `{0,…,k_W−1}`，覆盖条件 `(k_W−1)d_W+1=k_W≥L_in` ⇒ 窗口恰为 `[0,L_in)`，取值集合相同 ⇒ 逐位等价。**`d_W==1` 显式列入且被 C4x 机器锁住**（反例 `L=5,k=2,s=1,p=0,d=4`：窗口仅 `{0,4}`，整线 1.0 vs 正确 0.0，`[C4x-GUARD]` 实测输出一致）；`s_W` 因 `out=1` 不影响窗口，条件不依赖 `s_W`，正确；
  - C4b（恒等拷贝）：`k_W==1 ∧ d_W==1 ∧ p_W==0 ∧ s_W==1` ⇒ 窗口恰为单元素 ⇒ 恒等，独立推演成立（对 `ceil_mode` 亦成立：此时 `out=L_in`，第 94 行回退分支不触发）。
- **向量化替代覆盖完整性** — pass。§1.6.2 盘点 17 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对：累加器/标记初始化、取最大、cast、NaN 探测/判决、布局重排、单元级常量构造、C4 归约均已向量化；保留的 5 类（`kw`/`owl` serial、逐 lane 守卫、载入钳制、任务解码、host 元数据）逐项有理由，且 §6 不出现逐元素标量循环，与 §1.6.2 结论一致。
- **不可替代理由充分性** — pass。保留项理由具体到本算子（tile 级顺序依赖/循环携带累加维、运行时行索引、block 级索引、动态偏移写入陷阱 `TRAP-UB-dynsubview-dominance`、host/trace 元数据），无「实现简单」「照源码写」类空泛理由。
- **替代 API 可行性** — pass。替代方案 API 均有 `docs/Tilelang.language/` 条目或 examples 实测佐证（与维度 1 同口径），优先 v-prefix。
- **向量化轴与布局决策完整性** — pass。§1.6.3 按「窗口/池化类」枚举类别内必选候选：候选 1「I/O 原生 NCL / L 连续轴」、候选 2「核内 NLC / C 轴」均在矩阵内；逐候选有量化评分（整除性 / 尾 lane 浪费 / 累加链形态 / repack 代价 / UB 影响）；弃选有量化理由（候选 1 引 `PL-1.2` 98.3% 标量 + `s_W≠1` 跨步系数；候选 3 host permute 106–147µs `PL-1.3` 实测 > 本体 3–22µs；候选 4 数学不可行 + Cube 分形下限）。**弃选论证前提与所引 API 文档亲自核对一致**：`T.transpose` 文档 §2.3 确认「只支持转置两个轴」（核内 UB 级二轴交换，与 `PL-1.1` 实测 ~5.4µs 一致），否定论断与 pattern-library 实测条目无矛盾。迁移注：对照 §0.3 源的 GPU 并行轴选择（元素扁平 lane）独立评估并替换为 C 轴，未把 I/O 契约当内部布局依据。
- **布局决策与实现一致** — pass。§1.6.3 选定「核内 NLC（C 最内）+ 向量化轴 C + 核内 `T.transpose([1,0])` + L 轴 BL 分块 span 载入」，与 §3.3 伪代码（`row_src/row_dst/row_f32/eq_*/acc/flag` 形状、内层向量维 = C、`owl` 为 acc 行索引）、§4 内存规划（逐 buffer 形状与 UB 预算按重排布局算）、§6 循环结构（累加内层向量维 = C）三方一致。
- **编译期特化谓词的充分条件** — pass（带 1 条建议，见「非阻塞改进建议」）。§3.3 单一定义源：`IS_IDENTITY = (k_W==1 and d_W==1 and p_W==0 and s_W==1)`、`IS_GLOBAL = (L_out==1 and p_W==0 and d_W==1 and (k_W-1)*d_W+1 >= L_in)`。① 依赖变量全集逐项验证：C4 的关键 `d_W==1` **已显式列入**（§0.6 R4/§3.3 注释/C4x 组三处一致），窗口覆盖集合 `{k_w·d_W} ∩ [0,L_in)` 的集合关系论证成立；② 谓词在 §3.3 定义一次，§5.2/§5.5 引用该口径，无漂移；③ 负向守卫 C4x 组隔离了 `d_W` 子句（2 条仅违反 `d_W==1`、其余成立）。
- **与下游章节一致** — pass。§3.1 以 §1.6.1 优化后公式为输入（`base/span/LSPAN/s0`、eq 掩码、判决）；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。
- **D-3 实验裁决模式完整性** — pass。判定裕度依赖未实证常数（窄向量 op 每-op 代价——设计标注「未文档化假设」并说明 `CONST-vector-launch-overhead` 0.5µs/op 只是 bn=512 大 op 实测、不可外推）；提供三件套：主选（容量允许的最大 CH，`PL-1.4` 实测 CH64 最优为主证据）+ 备选（CH 减半 + BL 放大，结构完整，分核三要素按新 `num_logical` 重算：sincnet CH=32/BL=53 → num_cblk=3/num_oblk=26/num_logical=2496、persistent 40 核 ×63 任务，人工复算一致）+ 裁决计划（代表 shape、msprof op Task Duration、未知常数反解、10% 判定阈值、回写路径）；并评估按 shape 分派可行性。

## 非阻塞改进建议

> 结论为「通过」；以下为不阻塞 Stage 3 的建议。

1. **C4/C4b 特化谓词的负向守卫用例可补全（建议级）** — 位置 §1.6.1 C4x / verify_equiv.py C4x 组。
   - 现状：C4x 组 2 条配置仅隔离了 `d_W==1` 子句（`(1,8,5)` k2s1p0 d4、`(1,8,7)` k3s1p0 d3，均 `L_out==1 ∧ p==0 ∧ 覆盖成立`，仅违反 `d_W==1`），这与历史高危项（`d_W` 漏判致静默错值）对应，**已充分**；但 `C4` 条件中 `p_W==0` 与 `(k_W−1)·d_W+1 >= L_in` 两个子句**无隔离负例**，「移除任一子句即失败」对这两子句为空真（例如 `L_in=10,k=3,s=9,p=0,d=1`：`L_out==1 ∧ p==0 ∧ d==1` 但覆盖不足，窗口 `{0,1,2}` ≠ 整线；`L_in=5,k=5,p=2,s=5`：`L_out==1 ∧ d==1 ∧ k>=L_in` 但 `p≠0`，窗口 `{-2,…,2}` ∩ `[0,5)={0,1,2}` ≠ 整线）。
   - 说明：C4 条件**本身正确**（三子句齐全），故不影响正确性；此为机器守卫的健壮性缺口。
   - 建议：在 `verify_equiv.py` 的 C4x 组各补 1 条「仅违反 `p_W==0`」与「仅违反覆盖子句」的负例（断言 `not c4_condition(...)` 且一般路径与 `F.max_pool1d` 逐位一致），使每个子句都有隔离守卫。
2. **§0.6 承接行的小幅一致性润色（建议级）** — 位置 §1.6.0「调研结论」末段。该段写「#3/#4/#5/#6/#7/#11 的意图分别由 §0.6 R1/R2/R3 承接」，而 §0.4 中 #3/#5/#6/#11 属「保留」、#4/#7 属「重新设计」；逐项承接实际写在 §0.6 R1–R3 的「承接」行内（#1/#2→R1，#4/#5/#6→R2，#7→R3）。建议该段改为指向各「承接」行，避免读者将「保留项」误读为「由 R1/R2/R3 承接」。不影响语义与实现。
