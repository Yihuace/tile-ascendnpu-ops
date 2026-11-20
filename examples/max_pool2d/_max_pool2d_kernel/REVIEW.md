# _max_pool2d_kernel 设计文档检视报告（复检 revision v1）

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/max_pool2d/_max_pool2d_kernel/DESIGN.md`（**revision v1**；上一版备份 `history_version/design_v0.md`）
- 任务类型: 迁移（TileOPs 同构仓迁移，harness 子模式）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool2d.py`（函数 `_max_pool2d_kernel`，第 15–77 行；helper `common.py` 的 `pool_output_dim` 第 80–97 行）——本轮已**亲自重读源文件全 420 行 + `common.py` + 抽取件 `_max_pool2d_fwd_kernels.py`**
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 复检轮次: `revision_index = 1`（针对上一轮 REVIEW「不通过」的 1 阻塞 + 4 建议逐项闭合核对后再查新问题）
- 阻塞级问题数: **0**
- 建议级问题数: 3

## 机械复核（D-1/D-5）

1. **verify_equiv 重跑**：`/workspace/venv/bin/python examples/max_pool2d/_max_pool2d_kernel/verify_equiv.py`
   → 退出 0，末尾 `[EQUIV] ALL PASS — 69 case-groups, 0 failures`，并打印
   `[C4x-GUARD] general path = 0.0 == golden 0.0; whole-plane = 1.0 (excluded by the d==1 guard)`。
   **逐行比对 §1.6.1 内嵌结果表**（本 Reviewer 用脚本解析两边的数据行并逐字段比较）：
   内嵌表 **69** 条数据行 vs 重跑 **69** 条，字段 `(check/dtype/shape/kH,kW/kinds/worst|d|/viol/status)`
   **零处不一致**（分组计数也一致：E1 18 / C1 18 / C4 15 / C4x 6 / C4b 12；列宽等排版差异不计）。
   与 v0（57 组）相比**新增组真实存在且非空真**：C4x 组 6 行（`(1,8,5,5) k2s1p0 d4`、`(1,8,7,7) k3s1p0 d3` ×3 dtype）、
   E1 新增 ceil step-back 用例 3 行（`(1,8,8,8) k3s4p0 ceil`），原空真用例 `c4_cfg[4]=(1,8,5,6) k8s2` 已删除。
   空真防护已落到代码：`mismatch()` 首行 `assert cand64.numel() > 0 and base64.numel() > 0 and shape equal`（第 171–174 行）。
2. **design_calc_check**：`python3 .agents/tools/design_calc_check.py --design examples/max_pool2d/_max_pool2d_kernel/DESIGN.md`
   - `ub_budget`: **skip**（advisory checker 未解析到逐层级 buffer 表）→ **人工复核：已核，结论成立**。
     按 §4.5 逐 buffer 求和复算：resnet-stem CH=32/fp16 = **73 632 B**、vgg-block CH=64 = **73 792 B**、
     resnet-stem CH=64 = **147 264 B**、alexnet-ceil CH=64 = **71 680 B**、fp32 resnet-stem CH=32 = **95 136 B**、
     fp32 resnet-stem CH=64 = **190 272 B**（逐项与文档表/公式相同）；×1.7 后分别 = 125 174 / 125 446 / 250 349 /
     121 856 / 161 731 / 323 462 B，与 196 608 B(=192 KiB) 判据比较结果与文档一致（CH=64 被否、CH=32 落地）。
     退化路径复算：C4 判据 `CH·H_in·W_in·(elem_bytes+9) + CH·(8·H_in+20+elem_bytes+1)`；
     `(1,8,4,4)` CH=8 → 1 408+440 = **1 848 B**×1.7 = 3 142 B ✓、`(1,8,64,64)` → 360 448+4 280 = **364 728 B**×1.7 ✗ 回退 ✓；
     C4b `(1,8,4,4)` 256 B ✓ / `(1,8,64,64)` 65 536 B×1.7 = 111 411 B ✓ / `(1,8,112,112)` 200 704 B×1.7 ✗ 回退 ✓。
   - `l0c_budget`: **skip**（未解析到 `block_M×block_N`）→ **人工复核：n/a，已核**。§5.1 判定纯 Vector（无 `T.gemm`），
     L0C 不参与——与源码一致（源 kernel 第 15–77 行无任何 Cube/matmul 语句）。
   - `core_split`: **skip**（未解析到 dims）→ **人工复核：已核，结论成立**。`num_logical` 逐 workload 复算：
     smoke `1×8×1=8`、resnet-stem `32×56×2=3 584`、vgg-block `16×28×2=896`、alexnet-ceil `32×27×1=864`；
     `num_kernels=min(num_logical,40)` → 8/40/40/40；`num_local=ceil(num_logical/num_kernels)` → 1/90/23/22（**静态编译期常量**）。
   - `r3_metrics`: **pass**（四口径齐全）。
3. **物理核数实查**：`/workspace/venv/bin/python -c "from tilelang.utils.npu_utils import NPUUtils; print(NPUUtils.get().get_aicore_num())"`
   → **20**（本会话复跑；启动打印 dev root build 警告属环境噪声）。与 §5.5② 自述「实查 = 20（Ascend910B3）→ 纯 Vector ×2 = 40」**一致**，
   且与 `core-split-strategy.md` §1 要素②「纯 Vector 翻倍」要求一致；设计明确**拒绝**复用 `CONST-aicore-910B2C` 的 24 ✓。

## 上一轮（v0）检视问题闭合表

> 逐项给出闭合证据句（含 DESIGN.md 章节号 / 源码位置），据此把注意力放到新问题上。

| v0 问题 | 严重度 | 闭合证据 | 状态 |
|---|---|---|---|
| **问题 1**：C4 整平面归约充分条件漏 `dilation==1` → 编译期分支静默错值 | **阻塞** | ① §0.6 R4（第 253–268 行）改为**充分条件**表述并显式增补 `d_H==d_W==1`，附**反例一**（`H=W=5,k=2,s=1,p=0,d=4`，torch 值 0.0 vs 整平面 1.0）与**反例二**（`H=3,k=3,s=4,p=1` 窗口缺 `i_h=2`），并写明**不利用** `H_in==1` 特例放宽；② §1.6.1 C4 行（第 534 行）+「优化后公式」代码块（第 671–672 行）含 `d_H==1, d_W==1`；③ §3.3 `IS_GLOBAL`（第 858–861 行）含 `d_H == 1 and d_W == 1`；④ §5.2（第 1182–1184、1205–1206、1211 行）、§5.5①（第 1245–1248 行）、§8.3 L0-12/12b/12c、§9.4 第 4 条**六处同步**；⑤ `verify_equiv.py` 新增 C4x 反例组（6 行）+ HAZARD 演示行 + `assert not c4_condition(...)`。**独立复核**：(a) 充分性独立推演成立（`p=0∧d=1∧out=1` 迫使 `k==in`（否则 `out≤0`）⇒ 窗口 = `[0,k-1]×[0,k-1] ⊇ 平面`）；(b) 负向控制真实——本 Reviewer 实跑：去掉 `d` 两项后 `c4_condition` 对两条 C4x 配置均返回 True（`no-d-> True  with-d-> False`），即该 assert 确实会把 v0 的漏 dilation 错误形态拦下。 | **已闭合** |
| **问题 2**：C4 验证组含 1 条空真用例 + §1.6.1 覆盖陈述不准确 | 建议 | `c4_cfg[4]` 已由 `(1,8,5,6) k8s2`（`out=(0,0)`）替换为 `(1,8,8,8) k8s1p0`（`out=1×1`，非空）；`mismatch()` 增空真断言（第 171 行）；§1.6.1 说明行（第 552、559–560、651–653 行）同步更新为 69 组并写明替换。重跑 69 组全 PASS、0 空真。 | **已闭合** |
| **问题 3**：§4.5 fp32 派生数字不可复现 + KB 口径混用 | 建议 | §4.5（第 1116–1120 行）fp32/CH=64 更正为 `190 272 B（185.8 KiB）`、`×1.7 = 323 462 B = 315.9 KiB`（本 Reviewer 逐 buffer 复算 = 190 272 B ✓）；同节声明「统一 KiB（1 KiB=1024 B）并附原始字节」（第 1108–1109 行），逐项复算全部对得上。 | **已闭合** |
| **问题 4**：C4 容量判据声称「小项已计入」但公式未含 | 建议 | §4.5（第 1126–1134 行）判据改写为显式含小项：`CH·H_in·W_in·(elem_bytes+9) + CH·(8·H_in+20+elem_bytes+1)`，并逐项注明 `mid/midn/gmax/nmax/zero/nanbuf/out32/gout/cond` 的贡献；示例复算 1 848 B（小项占 24%）/ 364 728 B，本 Reviewer 复算一致。 | **已闭合** |
| **问题 5**：引用/命名细节（4 处） | 建议 | ① §0 抬头（第 12–14 行）改为「第 258–344 行的函数 … 与第 347–420 行的 wrapped op / 类」——对源码核对**正确**；② §1.6.3 候选 #1（第 727 行）已删「`T.reduce_max` 要求静态形状」，仅留「bf16 须先上转」（`grep -c "静态形状"` 除变更日志行外为 0）；③ §4.3（第 1063 行）`mf32p` → `mf32` 并注明「与一般路径复用同名」；④ §5.2 表（第 1203 行）写全 `k(3,3) s(3,3) p(1,1) d(2,1)`。**残留**（新发现，非 v0 项）：§4.3 同行的 `zero2`/`nanbuf2`/`out32b` 与 §3.3/§4.5 的 `zero`/`nanbuf`/`out32` 命名不一致——见「非阻塞改进建议 2」。 | **已闭合**（残留 1 处命名，见建议 2） |

## 检视详情

### 0. 源算子理解与迁移分析 — pass

逐行 Read 源 `max_pool2d.py`（420 行）+ `common.py` + 抽取件后逐项核对：

| 检查项 | 结论 | 源码核对证据（源码语句/行 + DESIGN 章节） |
|---|---|---|
| 语义理解正确性 | **pass** | ① 输出尺寸：源第 33–34 行 `pool_output_dim(...)` ↔ §0.1/§1.3「不得重算」，与 `common.py` 第 80–97 行**逐字一致**（`effective_kernel=d*(k-1)+1`、ceil 上取整、`(out-1)*stride>=input+padding → out-=1` 回退、`max(out,0)`）✓。② padding/越界：源第 63 行 `if ih>=0 and ih<h_in and iw>=0 and iw<w_in` + 第 57 行初值 `T.cast(float("-inf"), accum_dtype)`，§0.1「越界 lane 跳过 ≡ −∞ padding」独立推演成立（`max(−∞,x)=x`）✓。③ NaN：源第 65 行 `has_nan = has_nan \| T.isnan(val)` + 第 68–72 行 `if_then_else(has_nan,NaN,max_val)` ↔ §0.1「窗口含 NaN ⇒ 输出 NaN」✓。④ ±inf：源用 `T.max`，`{+inf,−inf}` 同窗 → `+inf`，§0.1 正确且显式排除求和类判据 ✓。⑤ dtype：源第 32 行 `accum_dtype="float"`、第 177–180 行门禁 `{float16,bfloat16,float32}`（**含 fp32**）↔ §0.1/§8.2 `atol=rtol=0` ✓。⑥ I/O：源 `x:(n,c_in,h_in,w_in)`/`out:(n,c_in,out_h,out_w)`、`out_idx=[1]`（第 37 行）↔ §0.2；4D-only 与 manifest 注释一致 ✓。 |
| 算法解读完整性 | **pass** | §0.3 步骤 1–6 覆盖源第 33–73 行**全部计算语句**：host 输出长度（33–35）、`T.Kernel(ceildiv(total_output,block_m), threads)`（44）、lane 解码 `%`/`//`（46–53）、状态初值（55–58）、窗口双循环 + 守卫 + `isnan` + `max`（59–66）、结果与写回（68–73）；无遗漏、无臆造。数据流图「GM→寄存器→GM，无 SMEM/L1」与源码一致（源无 shared/alloc_shared）✓；host 侧 `lru_cache(maxsize=32)`（15）、`compile_flags`（37）已列出 ✓。 |
| 优化手段覆盖完整性 | **pass** | §0.4 列 11 项；对照源码：无 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`（§0.4 末注与源码一致 ⇒「舍弃项：无」成立）✓；`T.Parallel(block_m)`（45）、`T.alloc_var` 标量状态（55–56）、逐元素 GM gather（64）、`lru_cache`（15）、`out_idx<total_output` 尾守卫（47）均识别且未把实现层/算法层混判 ✓。 |
| 耦合性判定合理性 | **pass** | §0.5 每条四态处置有依据，与 migration-analysis.md §5.3 映射表、`docs/` 条目、pattern-library 实测条目对应；CUDA threads/warp、寄存器比较链、逐元素 gather 均判「重新设计」，无把 GPU 专有手段判保留 ✓。 |
| NPU 重设计可行性 | **pass**（v0 为 fail，本轮闭合） | R1/R2/R3 论证成立（与 PL-1.2/PL-1.1 已验证形态同形；R2 只用 IEEE 正确的 `eq`，与 `TRAP-vcmp-ne-ordered-nan` 合法绕法一致）。**R4（C4）的语义保持论证现成立**：充分条件已含 `d_H==d_W==1`，独立推演证明窗口恰覆盖整平面（见维度 8）；C4 的 NaN 判决由独立 eq 掩码归约给出，不依赖 `reduce_max` 的 NaN 行为 ✓。 |
| 设计与决策一致性 | **pass**（v0 为 warn，本轮闭合） | §1–§7 对 R1/R2/R3 的落地与 §0.5/§0.6 一致；R4 的修正条件已同步到 §3.3 `IS_GLOBAL`、§5.2、§5.5、§8.3（不再「§0 判重设计、§3 照旧方案」）✓。 |
| golden 独立性 | **pass** | §8.1 golden 直接 `F.max_pool2d`（移植源仓基准 `MaxPool2dTest.ref_program`），**不含** C 轴向量化 / 核内转置 / C4-C4b 特化，以 §0.1 语义为唯一依据 ✓；fp32 中转后舍回对 max 无损，`atol=rtol=0` 与源套件一致 ✓。 |
| host/op 层引用路径真实性 | **pass** | 设计引用到的源文件路径（`kernels/pool/max_pool2d.py`、抽取件、manifest、`tests/ops/test_pool.py`、`benchmarks/ops/bench_pool.py`）逐条 `ls` 均存在 ✓；未引用不存在的 `ops/pool/...` 路径 ✓。 |
| 多输出/变体范围 | **pass** | 源同文件第 258–344 行 `_max_pool2d_with_indices_kernel`（`out_idx=[1,2]`）+ 第 347–420 行 wrapped op / 类；§0 抬头**显式声明排除**且行号现正确，与 manifest `ref_api`（`return_indices=False`）一致，无范围蔓延 ✓。 |

### 1. API 可行性 — pass

- §3.2/§3.4 列出的 14 个 API 全部在 `docs/Tilelang.language/` 有条目，dtype 表逐条核对一致（`T.vmax`/`T.reduce_max` fp16/fp32 ✓、**bf16 ✗**；`T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast` 支持 bf16；`T.vbrc` scalar→任意 shape、dst 须整 buffer；`T.clear` 1~5 维；`T.vmul`/`T.vsub` 支持原地；`T.transpose` **仅二轴交换**；`T.infinity` fp16/fp32 ✓、bf16 ✗）。
- bf16 缺口**已被设计显式识别**并给出路径（经 `T.vcast` 上转 fp32 计算，§3.4 注/§9.3），与「fp32 比较域」自洽 ✓。
- v-prefix 优先 ✓；Developer 模式自动同步 + 一维 `T.Kernel` ✓；`cond`/`cfinal`/`mb` 为片上 bool，不写 GM ✓。
- （唯一 API 级瑕疵为 §3.2 的 C4 判决 cmp 模式与 §3.3/§4.4 矛盾，见「非阻塞改进建议 1」——不影响 API 存在性。）

### 2. 内存层级规划 — pass

- 搬运路径完整、无越级：GM→UB（`T.copy`）→ 片上向量 op → UB→GM（`T.copy`），无 GM→L0 直连、无 Cube/L1/L0C（§4.4）✓；每个 buffer 标注层级（UB）与 shape/dtype（§4.3）✓。
- UB 预算：见「机械复核」第 2 项，逐 buffer 复算与文档表**逐项一致**；`resnet-stem` 因 `W_in=112` 被容量判据压到 CH=32 ✓（与 `PL-1.4` 实测同向）；退化路径 C4/C4b 容量**独立复算**且失败即回退 ✓；两档膨胀口径（1.7 / 1.10–1.12）并列 ✓。
- L0C：不适用（纯 Vector）✓；`out_idx=[1]` 单输出 ✓。
- v0 的两处 warn（fp32 数字不可复现、C4 小项未计入）已闭合 ✓。

### 3. Tiling 策略 — pass

- 分核策略三要素齐全（对照 `core-split-strategy.md` §1/§2.2）：① 逻辑核数按**编译期分支的单元粒度**定义（一般 `N×out_H×ceil(C/CH)`；C4/C4b 整平面 `N×ceil(C/CH)`，并显式说明 C4b 与通式分叉）✓；② 物理核数**实查**（查询代码+返回值 20 → ×2 = 40）✓，拒绝跨型号复用 24 ✓；③ 规模判定逐 workload 三选一，极大规模走 persistent，`num_local` 为 Python 编译期常量（**静态界**）✓，`num_kernels=min(num_logical, vector_cores)` 保证无空核、负载差 ≤1 任务 ✓。
- v0 的措辞瑕疵（§5.5 表「无法通过调小 block 缩减核数」方向相反）已删改 ✓。
- 核数与 block 取值自洽：CH 由容量判据 + 8 的倍数 + `≤min(C,64)` 决定，与 `num_logical` 算术一致（8/3 584/896/864，复算 ✓）。
- 对齐/非整除：C 轴恒为 8 的倍数（亦为 fp32 宽 4 的倍数）✓；`W_in` 仅作转置后 row 长不作向量轴 ✓；`real_c`/`uid`/`ih,iw` 三类尾块分别处理 ✓。
- 新增 C4x 反例行（`d>1`，`out=1×1`）按**一般网格口径**计 `num_logical=1`，与 `IS_GLOBAL=False` 一致 ✓（与源语义、§0.6 R4 反例一自洽）。

### 4. 技术约束检测 — pass

三维 Kernel 不使用（一维 + kernel 内解码，§3.5.1）✓；无 GEMM 非整除/L0C 溢出（无 Cube）✓；host 侧无输入数据改动（仅 shape/stride 元数据与缓存）✓，符合 `ascend-constraints.md` §4；动态 `T.serial` 边界不涉及（`k_H/k_W/out_W` 全编译期常量）✓；UB 容量已处理 ✓。

### 5. 循环与同步 — pass

- §6 明确 `T.serial`（persistent 任务循环 / `kh` / `kw` / `ow`）+ `T.Kernel` 的职责，全部上界为编译期常量；无 `T.Parallel` 误用、无 `T.Pipelined`（并给出不用理由与 Stage 4 探索候选）✓。
- **实现次序核对（逐条拍序）**：一般路径 op 序列 `vmax → vcmp(row,row,"eq") → vcast(cond,mf32) → vmul(flag,mf32,flag)`（§3.3 第 942–949 行）「掩码当轮生成当轮消费」次序**正确**；`vcmp(flag,zero,cfinal,"eq")` 位于全部 `vmax`/`vmul` **之后**（第 950 行）✓；`flag` 初值 1（0 吸收、1 中性）与 `x=x` 掩码语义使 `flag==0 ⟺ 窗口含 NaN` ✓；`vselect` 的 `Out` 独立、未用原位 `vselect` 累积 loop-carried 状态 ✓。以标量模拟器按序复算，与源「逐 lane 跳过越界 + 标量 NaN 或 + 标量 max」逐位一致，未发现恒真/恒假退化 ✓。
- 同步：Developer 自动同步，无跨引擎/跨核依赖（§7）✓。

### 6. 验证方案 — pass

- §8.1 golden = `F.max_pool2d`（源仓基准移植），独立于 NPU 算法 ✓；§8.2 按 dtype 给 `atol=rtol=0` 并给出依据（max 无累加、逐位精确，源仓套件同款）✓。
- §8.3 L0 计划现 17 条（新增 L0-12b「C4 dilation 守卫」、L0-12c「ceil_mode 回退」）：覆盖 manifest 三档 + fp32 + bf16 + 非方形/非单位 dilation + 非连续 + NaN/±inf 语义门禁 + C4/C4b 特化（**含 d>1 反例**）与容量回退 + 通道尾块 + 校验路径 ✓。
- v0 的 warn（C4/C4b 门禁只用 d=1、ceil 回退分支无用例）**已闭合**：L0-12b 用 `d4`/`d3` 断言不得走整平面；L0-12c 用 `(1,8,8,8) k3s4p0 ceil` 触发 helper 末窗回退（本 Reviewer 复算 `out=2` ✓，且与 torch 输出 shape 对照一致）✓。

### 7. 完整性与一致性 — pass

- 迁移模板 11 章节齐全，§0.1–0.7 小节齐全 ✓；无未填充占位符（§10.2「待实现」为 Stage 3 工件状态列，非占位符）✓；§3.5.3 引用的 examples/docs/pattern-library 路径**逐条 `ls` 全部存在**（15 条，含两个同族 DESIGN.md、`lerp_tensor`、`logsumexp`、`test_pool.py`、`pool.yaml`、`bench_pool.py`、`gpu_to_npu_adaptation.md`、`core-split-strategy.md`、`hardware-cost-model.md`、`algorithm-candidates.md`、`algorithm-research.md`、`migration-analysis.md`）✓。
- 内部一致性：C4 充分条件在 §0.5/§0.6/§1.4/§1.6.0/§1.6.1/§3.1/§3.3/§5.2/§5.5/§8.3/§9.4 **全文一致**（`grep` 核对：`d_H==d_W==1` / `d==1` 出现处描述一致，无残留旧式条件）✓；「向量轴 = C」在 API 映射/内存规划/Tiling/循环四处一致 ✓；KiB 口径在 §4.5 全节统一（`grep` 确认 §4.5 内不再出现十进制 KB 混用；§1.6.3/§5.3 的 "≈72 KB/192 KB" 属叙述性量级，与 72.1 KiB 差异 <0.2%，见建议 3）✓。
- 残留 1 处命名不一致（§4.3 `zero2/nanbuf2/out32b` vs §3.3/§4.5 `zero/nanbuf/out32`），非阻塞，见建议 2。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节齐全 ✓。

**① 算法调研存在性与完整性（pass）**：R1 候选表**含基线候选**（#0 源算法、#0' 基线向量化）且把 `algorithm-candidates.md` 的 `ALG-pooling` 命中行（R1 常数权重卷积/sum×常数、R2 滑窗流式、R3 窗口重叠重复读、R4 C 轴向量轴+跨步系数、`known_impl`/`kb_links`）逐项评估 ✓；R2 有明确结论（「有在线变体，基线本身单遍」）并给结构依据（源第 65–66 行 running 标记/running max 即直接证据，参考表「滑窗天然流式」互证）✓；R3 表**四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表、口径写清）✓；R4 逐候选六项评估 ✓；「无更低复杂度算法族」有调研范围声明（参考表命中行 + 恒等变形空间 + pattern-library + 本仓 examples）✓；互联网检索未使用并给出理由 ✓。

**② 调研结论正确性（独立复核，pass）**：
- 数学不等价负向断言（候选 1/2/4：max 非线性、无逆元 ⇒ 无常数权重卷积/无前缀最大值差分）**独立推演成立**（属数学事实级，按分级复核不要求 API 检索），与参考表原文互证 ✓。
- 复杂度**独立复算**：`FLOPs = N·C·out_H·out_W·k_H·k_W` 与 manifest `roofline.flops` 逐字一致；`bytes = (N·C·H_in·W_in + N·C·out_H·out_W)·eb` 四档数值复算一致（smoke 5 120 / resnet-stem 64 225 280 / vgg-block 16 056 320 / alexnet-ceil 15 376 384 B）；可分离 2D「不降反升」独立复算成立（resnet-stem 1.00 / vgg 1.50 / alexnet 1.01）；**中间缓冲未漏计 GM 往返**（本设计无跨引擎中间矩阵，容量项 0）✓。
- **负向断言主动复核**：候选 3（可分离）未漏——参考表 `ALG-pooling` 未列它，属恒等变形自查；「无在线变体」断言与源码 online 证据（源第 65–66 行已用 running 手段）不矛盾（设计恰**采用**在线单遍）✓。硬件亲和淘汰引用 `constants.md` 条目 ID（`CONST-mte2-degradation` / `CONST-capacity-910B2C` / `CONST-vector-launch-overhead`）✓。
- **等价性论证正确性（本轮重点，独立推演）**：
  - **C4 充分条件现成立**：`p=0 ∧ d=1 ∧ out=1` ⇒ `out=(in-k)/s+1≥1 ⇒ in≥k`（否则 `out≤0`），与覆盖条件 `k≥in` 合并得 `k==in`，故窗口 = `[0,k-1]=[0,in-1]` = 整平面，取值集合完全相同 ⇒ 逐位等价。**`d==1` 项确为必要**（`d>1` 时窗口 = `{k_h d_H}∩[0,in)` 为等差子集 ≠ 平面）✓。
  - **反例组确能捕获 v0 缺陷**：实跑确认去掉 `d` 两项后 `c4_condition` 对两条 C4x 配置返回 True（本应 False）⇒ `assert not c4_condition(...)` 会失败；且 HAZARD 演示断言 `gen==ref==0.0 ∧ wrong==1.0` 独立复现（torch + 泛型路径 0.0、整平面 1.0）✓。守卫分支代码**亲读**（`verify_equiv.py` 第 82–99、293–320、341–352 行），负向控制建模 = v0 历史错误形态（漏 dilation）✓。
  - **E1 覆盖 ceil 回退分支**：`(1,8,8,8) k3s4p0 ceil` 经 `pool_output_dim` 得 `out=2`（step-back 触发，本 Reviewer 复算 ✓），且与 `F.max_pool2d` 输出 shape 双向对照 ⇒ 锁住 helper 两处 ceil 修正 ✓。
  - C1 独立推演成立（`ih` 与 `k_w` 无关，是 `k_w` 循环不变量；合取可重排），并经机器逐位复核 ✓；C4b（`k=d=1,p=0,s=1` ⇒ 单元素窗、`out==in`）独立推演成立 ✓。

**③ 调研→设计一致性（pass）**：§1.6.0 选定算法（#0' C 轴向量化 + 向量 eq 掩码 NaN 追踪 + 核内融合转置 + 在线单遍；另采纳 #6 退化特化 → C4/C4b）实际驱动 §1.4、§3.1（以 §1.6.1 优化后公式为输入 ✓）、§6 ✓；§0.5/§0.6 处置与调研选族一致（R2=归约向量化、R3=C 轴布局、R4=候选 #6）✓；源 §0.4 的 11 项优化手段逐项有处置与意图承接（#3/#4/#5/#6/#7/#11 分别由 R1/R2/R3 承接），**无静默丢弃** ✓。

**④ 数学等价优化完整性（pass）**：§1.6.1 建立 E1 语义锚点 + 采纳 C1/C4/C4b，每项四要素（原式→优化后→等价性论证→收益）齐全 ✓；显式声明「无倒数乘/无高代价算子降代换/唯一 CSE 点已采纳」，并量化否决可分离 2D（≥ 基线 + 破坏单遍）✓；C4 的 `d==1` 现为条件组成部分、收益栏保留「不被 manifest 触发」标注 ✓。

**⑤ 向量化替代完整性（pass）**：§1.6.2 盘点 14 个计算点，与 §3.3 伪代码 + §6 循环结构**交叉核对无遗漏**：累加器/标记初始化（#1/#2）、取最大值（#3）、cast（#4/#7）、NaN 追踪（#5）、NaN 判决（#6）、布局重排（#8）、单元级常量构造（#14）均已向量化；保留的 #9/#10（tile 级循环携带依赖 + 运行时行索引）、#11（block 级任务映射）、#12/#13（host/trace 元数据）四类理由具体到本算子，非「实现简单」式托词 ✓；§6.1 明确「不出现逐元素标量循环」，与结论一致 ✓。替代 API 均有 `docs/` 佐证且优先 v-prefix ✓。

**⑥ 向量化轴与布局决策完整性（pass）**：§1.6.3 按「窗口/池化类」枚举**两类必选候选**（#1 I/O 原生 NCHW+最内连续轴、#2 核内重排+高整除性轴）+ 另两类（#3 host permute、#4 Cube/implicit-GEMM）✓；逐候选量化评分（整除性/尾 lane 浪费/累加链形态/repack 代价/UB 影响）✓；弃选有量化理由（#3 `PL-1.3` 实测 106–147µs > 本体 12–51µs；#1 `PL-1.2`/`PL-1.4` 实测跨步致 98.3% 标量；#4 数学不可行 + `out_W`<16 分形下限）✓，**未把 I/O 契约当作核内布局依据** ✓；迁移任务对照了源码隐式轴选择（GPU 逐元素 lane 是输入非结论）✓。D-3 实验裁决三件套齐全（主选 CH=容量允许最大 8 的倍数、备选 CH 减半且结构完整、代表 shape、指标 msprof Task Duration、反解未知常数方法、**判定阈值 10%**、回写路径、按 shape 分派评估）✓。布局决策与 §3.3/§4/§6 三方一致 ✓。

**⑦ 机械复核与内嵌表一致性（pass）**：`verify_equiv.py` 重跑 69 组全 `EQUIV_PASS` 且与内嵌表**逐字段零差异** ✓；`r3_metrics` pass ✓；C4x 组非空真且负向控制成立 ✓；E1 已含 ceil step-back ✓。

**结论**：维度 8 全部子项 pass——v0 的唯一阻塞项（C4 等价性论证）已闭合并经独立推演 + 机器回归双重确认。

## 非阻塞改进建议

> 以下为建议级（warn）事项，不影响「通过」结论，供 Stage 1 在后续修订/Stage 3 实现时顺手修正。

1. **§3.2 的 C4「NaN 判决」cmp 模式与 §3.3/§4.4 矛盾（建议优先修）**：§3.2（第 838 行）写 `cond=vcmp(nmax,zero,"gt")`，而 §3.3 伪代码（第 896 行）写 `T.vcmp(nmax, zero, cond, "eq")   # 0 即 has_nan`、§4.4（第 1077 行）写 `T.vcmp(eq vs 0)`。按 §3.3/§4.4，`nmax` ∈ {0,1} 且「1.0 iff 窗口无 NaN」，故 `cond`（= has_nan）应为 `nmax == 0`（`eq`）；§3.2 的 `"gt"`（= `nmax > 0` = 无 NaN）若被实现会**反转** C4 路径的 NaN 判决（`vselect(cond,nanbuf,gmax)` 会在无 NaN 时选 NaN）。§3.3 为冻结决策且 §4.4 一致，故以 §3.3 为准；建议把 §3.2 该格改为 `cmp_mod="eq"`。**理由**：虽是单格笔误且被 L0-12 类 NaN 门禁可捕获（非静默），但 §3.2 是 API 映射表，Stage 3 直接照抄会引入错误。
2. **§4.3 的 C4 buffer 命名与 §3.3/§4.5 不一致**：§4.3（第 1064 行）列 `zero2`/`nanbuf2`/`out32b`，而 §3.3 C4 分支（第 895–899 行）与 §4.5 小项口径（第 1130 行）均用 `zero`/`nanbuf`/`out32`。建议统一为同一组名（分支互斥，可注明「与一般路径同名但 shape 不同」）；此为 v0 已存在、本轮未触及的命名残留。
3. **两处轻微口径/覆盖精化（低优先）**：
   - §1.6.3（第 721 行）与 §5.3（第 1218 行）仍用叙述性「≈ 72 KB / 192 KB」，而 §4.5 已统一为 KiB（实值 72.1 KiB）。数值差异 <0.2% 不影响结论，建议加注「KiB」保持一致。
   - `verify_equiv.py` 的 C4x 两条配置均为**对称** dilation（`dH==dW`），故「移除 `d==1` 约束**任一轴**即失败」仅在移除 H 轴（或两轴同时移除）时成立——若未来出现「只漏 `dW==1`」的形态，该组不会失败。建议补一条非对称配置（如 `H=8,k_H=8,d_H=1` + `W=5,k_W=2,d_W=4`，`out=1×1` 且不含 `dW==1` 时 `c4` 误判为真）以独立锁住两轴；同时把 §1.6.1（第 555–556 行）「移除任一轴」措辞改为「移除 dilation 约束（两轴）」。（核心历史错误形态——v0 完全漏掉 dilation——本组已能捕获，故非阻塞。）
