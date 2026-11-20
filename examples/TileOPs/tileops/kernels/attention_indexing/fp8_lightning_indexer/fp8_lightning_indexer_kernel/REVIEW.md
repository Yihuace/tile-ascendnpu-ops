# _fp8_lightning_indexer_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/DESIGN.md
- 任务类型: 迁移（migration / harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`（主 kernel L19–L149；Op 类 `ops/fp8_lightning_indexer.py`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（均已内嵌于对应维度说明，不单列问题章节）
- 检视工具链: tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3`（dev root build）/ CANN 9.0.0 / Ascend910B3（实查复现 `get_aicore_num()=20`, `get_aivector_core_num()=40`）/ torch_npu 2.7.1.post4 / 2026-10-15

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。`python3 examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/verify_equiv.py` 输出
  `ALL_GATE_ITEMS EQUIV_PASS (L1 exact=PASS, L2 13/13 items pass)`；逐项与 §1.6.1「D-1 机器验证结果表」一致
  （E1 random `fp64_fwd_ulp=0.848 / corr_diff=6.66e-16 / viol=0 / nonfinite_eq=True`；角点 scale∈{0,−0,1e-4,1,−1,448,1e4}
  `≤0.889`；inf/nan 角点 `≤0.755`；subnormal `0.848`；E3 `0/0`；诊断 fold-into-K scale<0 `3.85e15 / 0.997 / viol=1.0`；L1 200 组有理数 diff=0）。
  无脚本缺失、无执行失败、无与内嵌表不一致项。
- **design_calc_check（`python3 .agents/tools/design_calc_check.py --design <path>`）JSON checks 摘要**：
  - `l0c_budget`: **pass** — `block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB`。
  - `r3_metrics`: **pass** — `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`。
  - `ub_budget`: **skip** — `no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)`。
    **人工复核（已核）**：DESIGN §4.5 手工小计 111,936 B（q_ub 32768 + w_ub 1024 + k_ub 8192 + scale_ub 256 + brc_ub 2048 +
    acc_ub 65536 + out_ub 2048 + 标量 64），与文中「≈111.9 KB」自洽（文中按 SI KB 计）；×1.10 ≈123 KB < 192 KB ✅。系数口径偏差见维度 2。
  - `core_split`: **skip** — `insufficient parsed dims {'M': 128, 'bn': 64}; declared logical cores = 32768`（解析口径产物，非设计声明值）。
    **人工复核（已核）**：DESIGN §5.5 明确 `num_logical_kernels = batch × ceildiv(seq_len, block_Q)` = 128/512/1024；
    物理核 `NPUUtils.get().get_aicore_num()=20`（本报告已独立实查复现）；无法对齐兜底 + 残留负载不均记录齐全（见维度 3）。
- 说明：以上两项机械 skip 均因本文档的表格/符号无法被检查器解析，已按要求转人工复核并给出结论，非机械通过。

## 检视详情

### 0. 源算子理解与迁移分析 — pass
（附 1 项建议级：golden scale 位置见下）

已亲自 Read 源算子 `fp8_lightning_indexer.py` 全文（L19–L149 + Op 类），与 §0 逐项核对：

- **语义理解正确性 — pass**。§0.1 公式 `Logits[b,s,t,g]=Σ_h relu(Sq[b,s,g·Hp+h,t])·W[s,g·Hp+h]·Scale[b,t,g]`、
  `Hp=heads//kv_group`（§0.1 ↔ 源 L64 `heads_per_group = heads // kv_group`）、**唯一归约轴 = 组内 head**
  （§0.1 ↔ 源 L140 `T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)`，`s_reshaped[bn,bq,hpg,G]` 的 `-2` 即 hpg）、
  **kv 轴纯 streaming 不归约**、**fp32 累加**（§0.1 ↔ 源 L43 `accum_dtype = T.float32`）、
  dtype（IndexQ/K `T.float8_e4m3fn` ↔ 源 L42；weights/scale/logits fp32 ↔ L43/L58–L60；cu_seqlen int32 ↔ L61–L62）、
  窗口边界语义（主 kernel 写块对齐超集、掩码属 `clean_logits_` ↔ 源 L82–L91 + L152–L182）均一致。
- **算法解读完整性 — pass**。§0.3 步骤表覆盖源全部计算语句：`cu_k_s_min/cu_k_e_max`（L88–L91）、
  Q/Weights hoist（L93–L94）、Pipelined（L96–L97）、scale tile（L100）、G==1 分支（L104–L114）、
  G>1 gather+gemm+scatter（L117–L132）、relu·w·scale（L134–L138）、reduce（L140）、写回（L142–L144），无遗漏；
  host 侧 Op `per_custom_dims_cast_to_fp8`（设备端 fp8 量化，Op L152–L160）与 wrapped `IndexQ.view`（纯 view）已列出。
- **优化手段覆盖完整性 — pass**。O1–O9 与源码/§0.5 对应：O1 fp8 TensorCore（L107–L114 + GemmWarpPolicy.FullCol L113）、
  O2 block_Q 行合并（L37–L38）、O3 hoist（L93–L94 在 Pipelined 外）、O4 softmax 流水（L96）、O5 `_MIN_PIPELINED_BLOCK_Q`
  死锁守卫（L13–L15/L40–L41）、O6 fp32 累加（L43）、O7 scale 归约内乘（L136–L138）、O8 块对齐超集写+独立 clean pass、
  O9 `TL_ENABLE_FAST_MATH`（L29）。未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass**。§0.5 每项均有四态处置 + 依据：**保留**（语义/hoist/fp32 累加）、**等价替换**
  （TensorCore→Cube `T.gemm`、CUDA 流水→NPU `T.Pipelined`、SMEM tile→UB/L1）、**重新设计**（fp8 载体→bf16/fp16、线程索引→block 级、
  scale 归约内→E1 提出）、**舍弃**（O5 守卫、`TL_ENABLE_FAST_MATH`、`GemmWarpPolicy`——理由均落在表内「NPU 对应方案/依据」列）。
  舍弃/重设计依赖的负向前提（fp8 不可用、NPU 无线程绑定）已逐条查证（见维度 8 与下文 FP8 复核），与 `ascend-constraints.md` 不矛盾。
- **NPU 重设计可行性 — pass**。R1/R2/R3 均含「源方案 → NPU 新算法 → 语义保持论证」：R1 载体替换论证了公式骨架不变、
  `has_scale=False` 的 identity 特化经 E3 逐位等价（ulp=0）、`-inf` 仍由 clean 单元写入；R2 线程索引→UB tile（同一集合逐元素映射）；
  R3 去守卫论证了 NPU `T.Pipelined` 无 warp 屏障（并如实标注为**结构判断**、列入 §9.1/§9.2 的 Stage 3 首编验证）。
  新算法满足一维/2D grid、UB/L0C 容量、分形对齐约束（§5.3）。
- **设计与决策一致性 — pass**。§1.4（1D grid / bf16 载体 / E1）、§3.2（`T.gemm(b_transpose=, initC=)` 无 `policy`）、
  §6（`T.Pipelined` 无 `_MIN_PIPELINED_BLOCK_Q` 约束）、§7（`pass_configs={}`）均为 §0.5/§0.6 决策后的形态，无「§0 判重设计、§3 照抄源」脱节。
- **golden 独立性 — warn（建议级）**。§8.1 直接移植源仓 `ref_program`（`test_fp8_lightning_indexer.py` L67–L94，已核对行号与内容：
  einsum → relu → 分组加权 → sum(dim=2) → permute → masked_fill(−inf)）与 `_validate_tensor_match`（L41–L65），
  **不复刻** §0.6 的 NPU 算法（载体 fp8→bf16、线程模型、E1 均未复刻到 golden 的数据流）→ 独立性成立。
  **建议**：§8.1 golden 对 `index_k_scale` 采取 `logits = score.sum(dim=2)` **之后**乘（源式应为归约内 `Σ_h(relu·w)·s`）——
  该位置恰为 §0.6/E1 采纳的变形形态；虽 E1 已被 verify_equiv 证为精确代数恒等、风险有限，仍建议把 scale 放回归约内以严格独立。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均在本仓 `docs/Tilelang.language/` 亲自核对到佐证：

- `T.gemm(src1, src2, dst, size=[], initC=, a_transpose=, b_transpose=)`（`线性代数操作/T.gemm.md` §2.1）——
  npuir 签名**确无 `policy` 参数**（印证 §0.5 条目 12 舍弃）；dtype 矩阵（§2.2.1）Ascend 支持 int8/fp16/fp32/bf16。
- `T.copy`：L0C fragment ↔ UB 局部拷贝（`内存操作/T.copy.md` §1 方向表「UB -> UB（含 L0C fragment 与 UB 之间的局部拷贝）」Developer √）；
  Developer 支持跨 dtype（§2.3 条 4，自动 VCast）；尾块动态 extent 借用（§2.2.2 条 2）；索引+size 形（§2.1）。
- `T.vrelu`（`数学操作/T.vrelu.md` §2.2.1：fp16/fp32 ✓、int32 ✓、**bf16 ✗**，与 §9.2 规避一致）。
- `T.vmul`（`数学操作/T.vmul.md` §2.2.2 四形：同形/行因子 `[M,N]*[M,1]`/`[1,N]*[1,1]`/标量，**确无 `[M,N]*[1,N]` 列因子**，印证 §1.6.2 #4）。
- `T.reduce_sum`（`规约操作/T.reduce.md` §2.2.1 fp16/fp32 ✓、bf16 ✗；§2.2.2 合法形 `(M,N,K)→(1,N,K)` 垂直归约）。
- `T.vbrc`（`shape操作/T.vbrc.md` §2.2.2 例 2 `(1,N)→(M,N)` 合法）。
- `T.Kernel(is_npu=True)`（`编译器提示操作/T.Kernel.md` §2.3：NPU 最多 2D block；§2.4 例 1 `cid` 拆 `(by,bx)`）。
- `T.Parallel`（`编译器提示操作/T.Parallel.md` §2.3：**下标记不能变换**、循环体内须为 UB Buffer——印证 §1.6.2 #11 的 API 限制）。
- `T.Pipelined` 动态 trip / `T.symbolic` / `T.ceildiv` / `T.min`：先例 `examples/indexer/indexer_fwd.py`（已核对 L29–L31/L64）。

v-prefix 优先（vrelu/vmul/vbrc）已贯彻；Developer 模式用自动同步（无手写 set_flag/wait_flag），与 §2.3/§7.1 一致。

### 2. 内存层级规划 — pass（附 1 项建议级）

- **搬运路径完整 — pass**：§4.4 给出 `GM→UB/L1→(Cube)→L0C→UB→GM` 全路径，无越级（无 GM→L0 直连）；L0C→UB 经 `T.copy`（上文 §1 方向表佐证）。
- **UB 预算 — pass（机械 skip，人工已核）**：手工 111,936 B（=109.3 KiB），×1.10 ≈123 KB < 192 KB ✅（明细见「机械复核」）。
- **L0C 容量 — pass（机械 pass）**：`acc = block_Q·H_p × block_N = 256×64 = 16384 elem × 4B = 64KB ≤ 128KB`，
  且 16384 ≤ 32768（`CONST-cube-gemm-l0c-traffic-floor`）✅。
- **层级标注 — pass**：§4.3 逐 buffer 标注 UB/L1/L0C。
- **建议级**：§4.5 把放大系数写作「auto-multi-buffer 膨胀 ≈1.10」，但 `CONST-capacity-910B2C` 语义为——
  **默认 auto-multi-buffer 开启时膨胀 ~1.7x；×1.10–1.12 是 `auto-multi-buffer=false` 下的实测需求**。按 1.7x 复算 UB ≈190 KB（<192 KB，但余量仅 ~1%），结论仍成立、非阻塞；
  建议订正系数标注口径，并在 §9.2/§8.3 增列 Stage 3 首编 UB 探针（`ub overflow, requires N bits` 反推实测占用），避免贴限设计不可编译。

### 3. Tiling 策略 — pass

- **Block 划分 — pass**：§5.2 给出主选 `(block_Q=8, block_N=64)` 及理由，并附 5 行候选矩阵（L0C/UB/K 重读三维量化，选/弃理由明确）。
- **对齐约束 — pass**：`block_N=64` fp32 尾轴 256B ≥32B；`index_dim=64` 分形对齐；`block_Q·H_p=256` M 对齐；
  §5.4/§9.3 处理了 `H_p<16` 的 M 提升与 `seq_len` 尾块。
- **非整除处理 — pass**：kv 尾块 `tail=T.min(bn, Skv−k0)` + 零起点 UB 切片（`T.copy.md` §2.2.2 条 2/条 4 佐证）+ `k_ub` 预清零防 NaN；`seq_len` 尾块行守卫；空窗口 `n_kv=0` 不写。
- **分核策略三要素 — pass**（对照 `core-split-strategy.md` §1/§2.2）：① 逻辑核 `batch×ceildiv(seq_len,block_Q)` = 128/512/1024；
  ② 物理核 `NPUUtils.get().get_aicore_num()` **实查 = 20**（本报告已独立复现：aicore=20、aivector=40、`Ascend910B3`），
  混合算子直接取返回值不翻倍，符合标准；③ 规模判定为极大规模，因 **Developer+gemm 使核内串行 persistent 不可用**
  （CG-2026-0010）走「无法对齐兜底」——2^n 逻辑核 vs 20=2²·5，`gcd=4` 无法整数倍对齐，
  并显式记录残留不均（full +1.6% / smoke +9.4%）。三要素与核数/block 自洽。
- **实验裁决三件套 — pass**：§5.2 主选 `(8,64)` + 备选 `(16,32)`（结构完整、分核按新粒度重算逻辑核 512）+ 裁决计划
  （代表 shape、msprof 指标、阈值「备选 < 主选×(1−5%)」、回写路径、shape 分派评估），满足「判定裕度依赖未实测常数」时的要求。

### 4. 技术约束检测 — pass

- **三维 Kernel — n/a**：本设计用 **1D** grid（`T.Kernel(batch*num_q_blocks, is_npu=True)`），未触 2D 上限，更无三维。
- **GEMM 非整除 — pass**：M/N 整除、kv 尾块见维度 3。
- **L0C 溢出 — pass**：见维度 2。
- **GPU 专用 API — pass**：`get_thread_binding`/`threads`/`GemmWarpPolicy`/`TL_ENABLE_FAST_MATH`/`T.float8_e4m3fn` 均已 §0.5 处置。

### 5. 循环与同步 — pass

- **循环结构 — pass**：§6.1 明确 `T.Kernel`（1D 非持久并行）/ `T.serial(kv_group)` / `T.Pipelined(n_kv)`（动态 trip）/ `T.serial(block_Q)` 归约 /
  逐元素一律 v-prefix 整 buffer 操作；**显式声明不用 `T.Parallel`**（源逐元素式已由 vrelu/vmul/vbrc/copy 承接），与 §1.6.2 结论一致。
- **同步与模式匹配 — pass**：§7 自动同步，与 Developer 模式一致；无手写 flag。
- **尾块处理 — pass**：§6.4 指向 §5.4，逻辑完备。

### 6. 验证方案 — pass

- **Golden 函数 — pass**：§8.1 含 PyTorch 参考实现（移植源 `ref_program`），输入域为未量化 bf16/fp16 原始张量 + 可选 scale。
- **L0 测试计划 — pass**：§8.3 六例覆盖 smoke/主 shape/非方形/G>1/scale 路径/尾块 + Boundary（空窗口、-inf 区）。
- **精度标准 — pass**：§8.2 沿用源契约——有限域 correlation diff ≤1e-3（源 `_validate_tensor_match` L64 核对一致）、
  非有限域 isfinite 掩码精确一致；bf16 辅助 rtol/atol 1e-2。
- **建议级**：见维度 0 的 golden scale 位置项（与 §8.1 同源，不重复计）。

### 7. 完整性与一致性 — pass（附 1 项建议级）

- **章节齐全 — pass**：迁移模板 §0（0.1–0.7 齐全）–§11 完整。
- **无占位符 — pass**：全篇未出现任何占位符标记或未完成待办字样（§3.3 的 `...` 为伪代码省略号，非占位符）。
- **同类引用真实 — pass**：Glob 核对 §3.5.3 引用的
  `examples/indexer/indexer_fwd.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`、`examples/deepseek_v4/example_fp8_gemm_kernel.py`、
  `examples/flash_attention/flash_attn_npuir_dev.py` **全部存在**；`deepseek_v32` 先例已核（`dtype="float16"` L38；
  `T.vmul(ub_relu_32, q_s_32, temp)` L143 在 `T.reduce(...dims=[0])` L145 前、`T.vmul(scores_sum, k_s_32, o_temp)` L146 在后——**E1 先例吻合**）；
  `indexer_fwd.py` 先例已核（L59–L62 hoist / L68 `b_transpose=True` / L70 L0C→UB / L71–L72 vrelu·vmul / L74–L80 逐行 reduce，**与 §3.3 同形**）。
- **内部一致 — pass**：API 映射（§3.2）、内存规划（§4.3–§4.5）、Tiling（§5.2）三处 buffer/尺寸相互一致。
- **建议级**：`attention_indexing.yaml` 第 61 行 `lightning-indexer-fp8-s8k-h32-d64` 仍以 `dtypes: [float8_e4m3fn]` 列为 workload，
  manifest 内**无受限标注**。DESIGN §9.1 已如实披露「该 workload 在缺口修复前不纳入 NPU 交付契约」（点名 label），
  但 manifest 本体未同步标注——建议在 manifest 该行加 `status: restricted`（或注释），避免 Stage 5 bench 将该 workload 纳入必过集合。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 布局决策四子节。
- **算法调研存在性与完整性 — pass**：
  - R1 候选表**含基线候选 #0（源算法）**，并列举 E1/E2/E3 与负向候选 4（scale 折进 K）/5（head 归约折进 Cube）/6（融合 clean）/7（物化 S）。
  - R2 **明确结论「无（不需要）」** + 结构依据（唯一归约轴为 tile 内 head，kv 轴无归约、无跨块 running 统计量），**与源码一致**（源无 online 手段）。
  - R3 **四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值），基线在表，口径写清。
  - R4 逐候选亲和性评估（计算单元/容量/对齐/静态边界/流水融合），含淘汰依据。
  - 结论含选定算法（候选 1+2+3 组合）+ 关键依据 + 调研范围 + **源优化手段 O1–O9 逐项意图承接**（无静默丢弃）。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性：E1 `Σ_h(a_h·s)=s·Σ_h a_h` 独立推演成立（s 对归约轴 h 为常量、relu 作用于未缩放 Sq，含负/零 s 亦成立）；
    机器验证复跑一致（L1 精确 diff=0，L2 13/13）。
  - ② 复杂度算术**独立复算全部吻合**：MAC `2·1·8192·8192·1·32·64=2.749e11=275 GFLOP` ✓；
    K 重读 `8192·64·2·1024=1.0737e9 B=1.07GB` ✓；out `8192·8192·4=268MB` ✓；总 ≈1.37GB ✓；L2 口径 ≈303MB ✓；
    E1 省乘 `8192·8192·31≈2.08e9` ✓；物化 S `±4.3GB×2=8.6GB` ✓；发射项 `3×2.147e9/1.75e12≈3.7ms` ✓、E1 前多一遍 ≈+1.2ms ✓。
    **中间缓冲 GM 往返无漏计**（§1.6.0 容量项明确 L0C→UB 直拷免 GM 往返，与 §4.4 一致）。
  - ③ 亲和性判断与 `CONST-cube-gemm-l0c-traffic-floor`（`bm·bn ≤ 32768`）、UB 容量一致；R4 淘汰项有佐证。
  - ④ **负向断言独立复核**：「无在线变体」——对照 `algorithm-candidates.md` ALG-attention（R2 online softmax 因**跨 kv 归约**才需要）
    与 ALG-reduction，本算子 kv 轴无归约，结论成立、与源码证据一致（源无 online）；「无化简公式/无更低复杂度」——ALG-attention R1「scale 融入 Q/K」
    正对应候选 4，设计已评估并以 `corr_diff=0.997`（s<0）与「把 1 次归约后乘换成 bn·D 次 K 元素乘」否决；候选 5 数学不可行、候选 7 访存爆炸，
    均予否定——**未发现被漏掉的已知候选**。
  - ⑤ 并列候选（主选/备选裕度小）不纸面拍板：§5.2 给出主选+备选+裁决计划与阈值；§1.6.3 给出量化弃选理由。
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解（步骤 8 `out=out·Scale`）、§6 循环结构均与 §1.6.0 选定算法同口径；
  §0.5/§0.6 与调研选族一致；迁移任务源算法优化手段替换均有承接说明。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法做了系统审视，采纳 E1/E3 各含四要素（原式→优化后→等价性论证→收益量化）；
  负向项（scale 折进 K、head 归约折进 Cube）有依据。
- **等价性论证正确性 — pass**：E1 代数恒等独立推演成立、与 §8.2 精度契约不冲突（E1 不改变数值语义）；
  E3 `x·1.0≡x` 逐位等价成立；**未经独立推演的等价性结论不存在**。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 §3.3/§6 中全部循环/标量点（#1 relu / #2 ×w / #3 head 归约 / #4 ×scale / #5/#6 装载 / #8 写回**已全部向量化**；
  #7 窗口并集 / #9 cid 解码 / #10 动态 trip / #11 G>1 gather **保留**）；
  交叉核对 §6——无 `T.Parallel` 逐元素标量循环残留（§6.1 显式声明），无 §1.6.2 未涵盖的点。
- **不可替代理由充分性 — pass**：#7 block 级索引（8 次标量比较）、#9 每核 1 次整数除模、#10 依赖动态 shape 边界、
  #11 `T.Parallel` 下标不可变换（**附文档 §2.3**）——逐项具体到本算子，无「实现简单」类理由。
- **替代 API 可行性 — pass**：替代 API 均有 `docs/`+`examples/` 佐证（见维度 1），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 对多轴算子（S/H/D/Skv）给出 4 候选矩阵（acc 朝向 / 归约轴 lane 映射 / host permute / 核内转置），
  逐候选量化，（弃选 #2 无收益且增写回重排、#3 host permute 净亏且违 `ascend-constraints.md` §4〔已核对：§4「host 禁止改输入真实内容/物理排布，仅允许只改 stride/shape 元数据的视图」〕、
  #4 `T.gemm b_transpose=True` 已支持无需转置）；spec 先验被正确当作契约而非内部布局依据。
  **弃选前提与 API 文档一致**（`T.alloc_shared`/`b_transpose` 依据 `T.gemm.md` §2.1）。
- **布局决策与 §3.3/§4/§6 一致 — pass**：选定 acc `[bq·H_p, bn]`、向量轴 `bn` ↔ §3.3 `acc[block_Q*H_p, block_N]` ↔ §4.3 缓冲形状 ↔ §6 最内维 `bn`，三方一致。
- **§3.1/§6 一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 不出现 §1.6.2 未论证的标量循环。

#### FP8 能力缺口负向断言专项复核（本次检视重点）— pass

对 DESIGN 的核心负向结论「FP8 为 NPU 能力缺口（CG-2026-0022），改 bf16/fp16 载体」在**三层**独立取证，全部复现成功：

1. **前端 dtype 名**：本报告实跑 `hasattr(T,'float8_e4m3fn')==False`、`hasattr(T,'e4m3_float8')==True`
   （`tilelang/language/ast/ir.py` L1431 `e4m3_float8 = func_gen(("E4M3Float8"))`）——源 `dtype=T.float8_e4m3fn` 在 npuir 前端不可解析。
2. **codegen**：本报告实跑最小 npuir gemm（`in_dtype="e4m3_float8"`）→ **`TVMError: Cannot convert type e4m3_float8`**；
   源码 `src/target/codegen_npuir_api.cc:493–594` `DTypetoMLIRType` 的 float 分支仅 case 16/32/64，8-bit float 落入 `default: fail` → L593 `LOG(FATAL)`，**确无 fp8 映射**。
3. **docs dtype 矩阵**：`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.2.1（Ascend：int8/fp16/fp32/bf16，无 fp8）与
   `数据类型转换操作/T.vcast.md` §2.2.1（无任何 fp8 源/目标格）；且 `grep -ri "e4m3\|float8\|fp8" docs/Tilelang.language/` **零命中**（已枚举全部 16 个子目录）。
4. **runtime 佐证**：本报告实跑 torch_npu 2.7.1.post4 / Ascend910B3——
   `npu fp16→fp8` 报 `aclnnInplaceCopy failed, error code is 561103`；`npu zeros(fp8)` 失败；仅 `torch.empty(fp8, npu)` 可用（与 CG-2026-0022 一致）。
   （**备注**：本次实跑的 `aclnnInplaceZero` 错误码为 `161002`，与 CG 条目归并书写的 `561103` 略有出入；属 CG 证据的精度瑕疵，不改变「设备端 fp8 产生/转换不可用」的实质结论。）

**唯一合理路径判定**：T.gemm/T.vcast 与 codegen 均无 fp8 通路 ⇒ 即便改用「uint8 载体 + 手工 e4m3 解码」也无法进入 Cube（解码后仍须转 bf16/fp16 才能 gemm），且设备端无法产生 fp8 量化输入（Op 层 `per_custom_dims_cast_to_fp8` 受阻）⇒
  改 **bf16/fp16 载体 + fp32 累加 + 可选 fp32 `index_k_scale`** 是唯一可行落地方案，未发现被漏掉的替代路径；
DESIGN §9.1 已如实披露受限范围（指名 manifest fp8 workload 不纳入交付契约），符合 `negative-claim-evidence.md` 的举证要求。

> 另附两条未在正文小节单列的建议级事项（不影响结论）：① §3.3 伪代码/§4.1 使用 `has_scale` 编译期开关，但**工厂签名契约 `(block_N, num_stages, block_Q)`（§3.3 注 + NPU wrapper）中未定义其传入方式**——建议明确 `has_scale` 的来源（工厂级 Python 常量/额外 kwarg）或声明当前契约下恒为 True，否则 Stage 3 无法据文实现 E3 特化；
> ② Tiling 主选 `block_Q=8`（§5.2）与 NPU wrapper `default_config` 的 `block_Q=4` 不一致——建议对齐或显式声明设计覆盖该默认值，使 §4.3/§4.5 的预算与实际运行配置对应。
