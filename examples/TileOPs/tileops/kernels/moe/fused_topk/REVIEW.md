# _fused_topk_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/fused_topk/_fused_topk_kernel/DESIGN.md（**revision v1**；上一版 `history_version/design_v0.md`）
- 任务类型: 迁移（harness，**Developer**；源 GPU/CUDA → target="npuir"）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/fused_topk.py`（工厂 `_fused_topk_kernel` / `FusedTopKKernel`）；op 层 `/tmp/opencode/TileOPs-ref/tileops/ops/moe/fused_topk.py`（`FusedTopKOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（首轮 2 阻塞 + 3 建议均已修复，见下）

> **首轮阻塞复核**：① §1.6.0-R3 计算量列头已改为 canonical **`FLOPs（计算量/比较次数，逐行）`**（L433）+ 脚注 ① 命名「FLOPs（浮点/比较运算计数）」（L442）→ `design_calc_check` 机械 `r3_metrics=pass`（已重跑确认）。② §1.6.3 已补**规约类「水平归约 vs 垂直扫描」正交 lane 映射候选**（L583–598 正交评分表 + L608 候选矩阵 #5），并同步 §3.3（L688–689）与 §6.1（L916–917）。首轮 3 条建议（softmax padding 机制表述、`ALG-topk` 转录 R5→R3、`torch.topk` tie-break 未文档化假设）亦逐条落实（§0.1 L103–104 / L399 / L91–92）。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS / ALL EQUIV_PASS**。`python examples/fused_topk/_fused_topk_kernel/verify_equiv.py`（torch CPU，exit 0）输出
  `SUMMARY: cases=136 fail=0 selection_mismatch=0 max_rel_fp32=1.075e-06 max_violation_rate=0.000` / `RESULT: ALL EQUIV_PASS`；
  与 DESIGN.md §1.6.1 内嵌结果表（L514–523：136 例 / 选择集失配 0 / max_rel_fp32=1.075e-06 / 违反率 0 / 合计 ALL EQUIV_PASS）**逐项一致**。
  无 `EQUIV_FAIL` 项，E1–E6 全部采纳有效。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/fused_topk/_fused_topk_kernel/DESIGN.md`）：
  - `r3_metrics`: **pass** — `"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"`。首轮 fail 已消除。
  - `ub_budget`: **skip**（`"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"`）→ **机械跳过，人工已核**：§4.3/§4.5 大缓冲 `xf+sv+si = BM·E·12 B`，逐 workload = smoke 48KB / qwen3 96KB / deepseek 126KB / kimi 126KB（独立复算 `BM·E·12`：`64·128·12=96KB`、`42·256·12=126KB`、`28·384·12=126KB`、`32·128·12=48KB`），峰值 **126KB ≤ 192KB**（`CONST-capacity-910B2C`）。§4.5 按 128KB 保守预算并显式说明应对 `T.vsort` scratch。
  - `l0c_budget`: **skip**（`"no block_M x block_N values parsed"`）→ **机械跳过，人工已核**：本设计纯 Vector（§5.1/§2.2），无 Cube/L1/L0/L0C，skip 合理。
  - `core_split`: **skip**（`"insufficient parsed dims {'bm': 64}; declared logical cores = 4096"`）→ **机械跳过，人工已核**：§5.5 三要素齐全；`num_logical=ceil(T/BM)` 独立复算 = 64(E128)/98(E256)/147(E384)/1(smoke)，与 §5.5 ① 逐值一致；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查 = 20（Ascend910_9362，2026-10-14）× 2 = 40；③ 极大规模固定 40 核 + 静态 `T.serial` grid-stride + 动态 guard（见维度 3）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源算子文件 `kernels/moe/fused_topk.py`（全文 442 行）与 DESIGN.md §0 逐项核对（op 层语义亦经 §0.7 引用核对）：

- **语义理解正确性 — pass**。§0.1 数学语义（softmax/sigmoid 打分 → 行内 topK → 可选 renorm）与源码一致：softmax 路径源 L241–L260（`l_max` 行 max 归约 → `exp(my_scores[j]-l_max)` L255 → `l_sum` 归约 → `inv_row_sum=1/l_sum` L260）；sigmoid 路径源 L262–L268（逐元素 `1/(1+exp(-val))` L267，无行归约）；输出 `topk_weights=l_best_val*inv_row_sum`（源 L303）、`topk_ids=l_best_idx`（源 L304），与 §0.1 `w_{t,k}=s_{t,ids}` 一致。**dtype 语义**：源 op 层 `gating_output.to(torch.float32)` 上送（源 L426），kernel 形参全 `"float32"`（源 L212–214），§0.1「源全程 fp32 计算域」正确。**renormalize 在 host 侧**：源 L439–440 `topk_weights / topk_weights.sum(-1,keepdim=True)`，§0.1 判定为 host 侧实现手段，正确。**correction_bias 变体**：源 L96–L207 选择用 `my_biased=sigmoid+bias`（源 L136/L158）、输出用未加偏置 `my_scores`（源 L199 写 `l_best_orig`），§0.1 表述准确；§0.1 声称「workload 不生成 correction_bias、参考测试无数值用例」经核对成立（`test_moe_fused_topk.py` L201 的 correction_bias 用例为 device_check）。**tie-break**：源 lane-local 严格 `>`（源 L279 `if my_scores[j] > l_best_val`）+ 跨 lane 归约等值取小下标（源 L288–L297 `other_idx < l_best_idx`）→ 取最小专家号，§0.1 正确；且 §0.1 已把 `torch.topk` 的 tie 归属改标注为「未文档化假设」（L91–92），非事实断言。
- **算法解读完整性 — pass**。§0.3 步骤表 1→2a→2b/2c→3a→3b→3c→3d→4 覆盖源码全部计算语句（L121–L313）：步骤 1↔源 L230–235、2a↔L246–250、2b↔L253–260、2c↔L262–268、3a↔L276–281、3b↔L284–299、3c↔L302–304、3d↔L307–309、4↔L439–440；correction_bias 变体差异（2'/3'，源 L96–L207）已单列。数据流图（§0.3 L164–175）覆盖 `my_scores`/`l_max`/`l_sum`/`inv_row_sum`/`l_best_val`/`l_best_idx`/两输出全部 buffer；host 侧逻辑（validate → `_get_kernel` lru_cache → `to(fp32)` / renorm，源 L407–L440）已列出。无遗漏、无臆造。
- **优化手段覆盖完整性 — pass**。§0.4 O1–O9 逐项可回溯源码：O1 one-warp-per-token（`T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK*WARP_SIZE)`，源 L216；`token_id=block_id*TPB+warp_id` L221）、O2 寄存器驻留（`T.alloc_local`，源 L224）、O3 `shfl_xor` 树归约（源 L250/259/286–287；源头注 L32–35 明示 zero `__syncthreads`）、O4 K 轮 masked argmax + 掩码复用（源 L274–309，掩码 L307–309）、O5 fp32+减 max、O6 非整除 E 填 `-inf`（源 L235）、O7 `lru_cache`（源 L70）、O8 独立 GM 输出、O9 host 侧 renorm。源无 SMEM tiling/mma/异步流水/swizzle/persistent（超发 grid，源头注 L37–43），§0.4 声明与源码相符。
- **耦合性判定合理性 — pass**。§0.5 每条有四态处置 + 依据，与 GPU→NPU 映射一致：O1（warp/thread）→ 重新设计（NPU 一维 Kernel 无 warp）、O2 寄存器→等价替换（UB；规避 CG-2026-0028b）、O3 `shfl_xor`→重新设计（NPU 无 shuffle）、O4→重新设计（索引恢复依赖 shuffle）、O5/O7/O8→保留、O6→等价替换（免 padding）、O9→重新设计（融合）；舍弃 0 项，无「舍弃丢意图」项。统计（保留 4 / 等价替换 2 / 重新设计 4 / 舍弃 0）与 §0.5 表逐行自洽。
- **NPU 重设计可行性 — pass**。§0.6 每项给「源方案 → NPU 新算法 → 语义保持论证」三要素：R1 persistent 行块分核（行间独立 + 尾块 `real=T.min`，先例 cumprod）、R2 `T.reduce_max/reduce_sum` 承接行归约（承认归约分组变化 → fp32 ulp，§1.6.1-E4 机器验证）、R3 `T.vsort` 承接选择（一条原语替代 K 轮归约，且论证 top-k 集合在单调变换下逐位一致）、R4 padding 消除、R5 renorm 融合（全行分母抵消）。均满足 NPU 约束（一维 Kernel、UB 驻留、无分形/L0C、guard 仅 block 级）。
- **设计与决策一致性 — pass**。§1.2/§1.4/§1.5/§3.2/§3.3/§6 描述的是 §0.5/§0.6 重设计后的 NPU 算法（`T.vsort` 降序 + 原始 logits 选择 + 行块 persistent + kernel 内 renorm）；GPU 专用 API（`T.get_thread_binding`/`T.shfl_xor`/`T.alloc_local`/`T.Kernel(threads=)`）在 §1–§7 全部无残留（§3.5 L760–761 明列），无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性 — pass**。§8.1 golden 以 §0.1 源语义为唯一依据（`softmax/sigmoid → torch.topk(sorted=False) → 可选 renorm`），与参考测试 `fused_topk_torch`（`test_moe_fused_topk.py` L41–L58，`sorted=False` L55）同构，**未复刻** §0.6 的 vsort+原始 logits 算法；§8.1 末句（L952–953）显式声明「不复刻 §0.6 算法」，§0.7 亦重申（L306）。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 TileLang DSL API 均有真实佐证（已亲自核对文件存在）：`T.vsort`（`docs/Tilelang.language/排序操作/T.vsort.md`——核实「仅尾轴排序」§2.3-1、「仅 fp16/fp32、bf16 ×」§2.2.1、「`dst_index` 固定 int32」、「src/dst shape 一致」§2.2.2、tie-break 差异 §2.3-4，均与 §3.2 参数列一致；`testing/npuir/sort_ops/test_vsort_dev.py` 为 Developer 用例；探针 P1）、`T.copy`（`内存操作/T.copy.md`；logsumexp single；探针 P1-C）、`T.reduce_max`/`T.reduce_sum`（`规约操作/*.md`，核实 `reduce_max(buffer, out, dim, …)` 仅出值无下标参数，同 rank、归约维 dst=1 与设计 `(BM,E)→(BM,1)` 一致）、`T.vsub`/`T.vdiv`（`数学操作/*.md` 行广播）、`T.vexp`/`T.vsigmoid`、`T.Kernel(num_kernels, is_npu=True)`（`examples/cumprod/_cumulative_kernel`）。**v-prefix 优先**满足（vsort/vsub/vexp/vsigmoid/vdiv 均 v 前缀）。Developer 模式与自动同步匹配（§2.3/§7.1）。**pass**。

### 2. 内存层级规划 — pass

§4.4 给出完整 GM → UB 路径（`T.copy` slice，无 GM→L1/L0 越级——纯 Vector 无 L1/L0）；§4.3 每个 buffer 标注 UB 层级（`xf/sv/si/mr/sr/vk/ik`）；§4.5 UB 预算逐 workload 独立复算（`BM·E·12`：48/96/126/126KB，峰值 126KB ≤ 192KB）成立。**机械 `ub_budget` skip 已人工复核**。`×1.7 auto-multi-buffer` 包络（≈214KB）超 192KB 的余量风险由 §4.5「128KB 保守预算 + 应对 `T.vsort` scratch」与 §9 R-5「溢出则缩 `BM`」显式承接，属可调杠杆，不构成阻塞。无 L0C（无 Cube）。**pass**。

### 3. Tiling 策略 — pass

- Block 划分：§5.2 `(BM,E)` 行块（整 E 驻留），`BM=min(64,⌊UB_BUDGET/(E·12)⌋,T)`，理由充分（E 为排序/归约尾轴，必须整行驻留）。
- 对齐约束：§5.3 尾轴 32B 对齐（fp32 UB 行主序连续天然满足；K=1 4B 尾块列入 §9 R-9）。
- 非整除：§5.4 T 尾块 `real=T.min(BM,T−off)` + 双显式 slice；E 任意宽度（无 padding）；K≤E（op 校验）。
- **分核三要素（`_shared/standards/core-split-strategy.md` §1）**：① 逻辑核数 `num_logical=ceil(T/BM)` 独立复算 = 64/98/147/1，与 §5.5 ① 逐值一致；② 物理核数 **实查** `NPUUtils.get().get_aicore_num()` = 20（设备 Ascend910_9362，2026-10-14）+ 纯 Vector ×2 = 40，含查询代码（§5.5 ②）；③ 规模判定三态——smoke(32,128) `num_logical=1≤40`「无需适配」；full workload（64/98/147>40）→ 极大规模，固定核数 `min(num_logical,40)=40` + 核内 `T.serial(num_local_tasks)` grid-stride（`num_local_tasks=ceil(num_logical/40)`=2/3/4，**静态值**）+ 动态 `if task < num_logical` 守卫；并论证 `T=4096=2^12` 不含因子 5、`BM` 受 UB 上界约束无法压 `num_logical≤40`（L899）。核数与 block 取值自洽、核内串行边界静态。**机械 `core_split` skip 已人工复核**。**pass**。

### 4. 技术约束检测 — pass

§3.5 强制检测：三维 Kernel 无（一维 `T.Kernel(num_kernels, is_npu=True)`，源亦为一维 grid）；GPU 专用 API 全部替换；GEMM 非整除/L0C 溢出**不适用**（无 Cube/L0）；工具链陷阱规避清单（不 `alloc_local`、不标量 `if_then_else`/`>>`、不标量消费向量产出、不 `T.Parallel` 直写 GM、不原子标量载荷，均对应 CG-2026-0027/0028a-c/0029/0030）与 `moe_grouped_gemm_nopad/_tile_scheduler_kernel` 先例一致；数据流全为向量→向量（UB）+ `T.copy`→GM，guard 仅 `if task < num_logical` 这一 block 级分段条件。**pass**。

### 5. 循环与同步 — pass

§6 明确循环结构：核内任务循环 `T.serial(num_local_tasks)`（静态界）+ block 级 guard；选择/打分/归约无显式循环（原语内部向量化）。§7 同步策略与 Developer 模式一致（无手动 set_flag/wait_flag/pipe_barrier，依赖 auto-sync + auto-multi-buffer），且显式规避 CG-2026-0027（不标量消费向量产出，§7.1 L935–937）。§6.1 明示无 `T.Parallel`/`T.Pipelined`/`T.Persistent`（用显式 persistent 网格 + `T.serial`，与 cumprod 一致）。尾块由 §5.4/§6 说明。**pass**。

### 6. 验证方案 — pass

§8.1 含 PyTorch golden 草案（以 §0.1 语义为依据，`sorted=False` 与源/参考测试口径一致）与容忍 tie-break 的校验口径：权重排序后 `assert_close(rtol=1e-3,atol=1e-3)`、ids 无重复且每个选中得分 ≥ 第 K 大参考得分 −1e-4——与 `test_moe_fused_topk.py` `_check`（L132 `assert_close`；L148/L157 min raw score 阈值 −1e-4）逐条一致（已核对）。§8.2 L0 门槛 10 例（含 fp32/fp16/bf16、非整除 E=100/33、T 尾块 4096、K=1、kimi/renorm）。§8.3 精度标准按 dtype 给出（rtol=atol=1e-3，与 harness 容差及 §8.1 一致）。**pass**。

### 7. 完整性与一致性 — pass

章节齐全（迁移类 §0.1–§0.7 + §1–§10 共 11 章）；无模板占位符残留（已机械检索常见占位符标记均无命中）；§3.5/§10 引用的 `examples/`、`testing/`、`docs/` 路径**逐条 Glob 核对全部真实存在**（Stage 0 抽取件、NPU 脚手架、manifest/workload/test/bench、logsumexp、cumprod、moe_grouped_gemm_nopad、vsort 测试与文档、`T.gather.md` 等 25 项全 OK）；API 映射/内存规划/Tiling 三处描述相互一致（`(BM,E)` 行块、尾轴 E、persistent 40 核）。**pass**。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策四子节。
- **算法调研存在性与完整性 — pass**：四问齐全——**R1** 候选表（§1.6.0 L408–418）**含基线 C0（源算法）**且评估 C1–C8；**R2** 在线算法明确结论「**有，但本场景不采纳**」+ 结构依据（整行 UB 驻留、GM 遍数不变、堆无 NPU 向量原语）——非「无在线变体」负向断言；**R3** 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，L433 表头）且**基线 C0 在表**，口径脚注 ①–④ 完整（L442–444）；**R4** 逐候选六维亲和评估（L448–458）。调研范围写明（`ALG-topk` 命中行 + `kb_search` + docs 全目录枚举 + examples + 实机探针 P1/P2 + 恒等变形自查 + 互联网未触发，L468–472）。**机械 `r3_metrics=pass`（已重跑）**。
- **调研结论正确性（独立复核）— pass**：
  - **等价性独立推演**：E1（选择域移到原始 logits）——`softmax` 分子 `exp(x_e−m)` 与 `sigmoid` 均**严格单调增**，行 max `m`/分母 `S>0` 为行常量 → `x_i>x_j ⟺ s_i>s_j`、`x_i=x_j ⟺ s_i=s_j`，故 `topK(x)` 与 `topK(s)` 下标集合逐位一致（机器验证选择集失配 0/136，含 ties）。E2（softmax+renorm 全行分母抵消）——`(e^{x_k−m}/S)/(Σ_K e^{x_j−m}/S)=e^{x_k−m}/Σ_K e^{x_j−m}`，`S≥e^{max−m}=1>0` 无除零。两项推演成立。
  - **复杂度算术独立复算**：R3 访存量 `T·E·elem+2·T·K·4` 复算 = E256 `4096·256·2+2·4096·8·4=2.36MB`、E384 `=3.41MB` ✅；D-2 roofline（L481–486）逐行复算——流量项（10.2KB→0.01µs、1.31MB→1.6µs、2.36MB→3.0µs、3.41MB→4.3µs）与发射项（`num_logical/40·14·0.5µs` = 0.18/11/17/26µs）均与设计一致；**中间缓冲未漏计 GM 往返**（§1.5/§4.4 明示无 workspace、无 host 往返，renorm 已内移消除第二次 host 算子）。
  - **负向断言主动复核**：C4「无 `T.vtopk`」——**亲自查证**成立：`ls docs/Tilelang.language/排序操作/` 仅 `T.vcumsum.md`+`T.vsort.md`，全 `docs/` grep `vtopk` 无命中，`tilelang/language/` 无 `vtopk` 导出；C3「`T.reduce_max` 无下标输出」——**亲自打开** `docs/Tilelang.language/规约操作/T.reduce_max.md` 核实（签名 `reduce_max(buffer, out, dim, size, clear)` 仅出 `out` 值、无 index 参数）成立。C6「分块 top-k+归并」对应 `ALG-topk` 参考表 R1/R2 候选，已评估。
  - **候选库命中行复核**：对照 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-topk`（L71–77：R1 部分选择 vs 全排序 / 分块 top-k+归并；R2 分块+归并；R3 O(N·k) vs O(N log N)；R4 归并链缓冲）——设计的 C3（部分选择/argmax）/C1·C2（全排序）/C6（分块+归并）**覆盖命中行**；`ALG-softmax` L14 的 R2 online softmax 经设计 R2 覆盖；`ALG-softmax`/`ALG-reduction` R4「水平归约 vs 垂直扫描」经 §1.6.3 新表覆盖。**无漏候选**。
  - **并列候选裁决路径**：§1.6.3 末段（L616）论证主选/弃选**不依赖未实证常数**（#2 文档能力硬否决、#3 量级差、#5 `T.vsort` 尾轴约束硬否决）→ 不触发实验裁决模式，且 `T.vsort` 成本未标定项（§9 R-4）明确不影响主选裁决（可行性由 D-3 探针硬裁定），交 Stage 4 实测。裁决路径清晰。
- **调研→设计一致性 — pass**：§1.2/§1.4/§1.5/§3.1/§3.3/§6 描述均为 §1.6.0 选定 **C1**（vsort 降序 + 原始 logits 选择 + K 切片打分 + renorm 融合/全行分母抵消），无「调研选 A 设计做 B」；迁移侧 §0.5/§0.6 处置与调研一致，源 O1–O9 优化意图承接逐项说明（O1→R1、O2→UB、O3→R2、O4→R3、O5 保留、O6→R4、O7/O8 保留、O9→R5，L471–472），无静默丢弃。
- **数学等价优化完整性 — pass**：§1.6.1 对 C1 系统审视并采纳 E1–E6，每项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）齐全（L501–508）；「无优化空间」项（top-k 选择本身无代数化简）有结构性结论（§1.6.0 L420–421）。
- **等价性论证独立推演 — pass**：E1/E2 如上独立推演成立；E3（padding 消除：源 pad 的 `-inf` 不改变输出，`exp(-inf-m)=0`、argmax 不选 `-inf`）、E5（全 dtype 升 fp32，且为 `T.vsort` bf16 不支持下的 API 必要性）、E6（renorm 融合，同除位置不同）均为数学恒等或数值路径保留，推演成立。**`verify_equiv.py` 重跑 136 例 ALL EQUIV_PASS** 与 §1.6.1 内嵌表一致。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对）— pass**：§1.6.2 盘点 12 点（#1–#12）；#1–#9（选择/行 max/减 max/exp/行 sum/sigmoid/归一化除/搬运/切片）**全部向量化**（`T.vsort`/`T.reduce_*`/v-op/`T.copy`，均有 docs 或探针 P1/P2 佐证）；#10–#12（persistent 任务循环、block 索引与动态尾块、host lru_cache）保留理由具体到本算子。**交叉核对**：§3.3 伪代码（L691–744）无逐元素标量循环、§6.1（L916）明示「无任何逐元素标量循环」——与 §1.6.2 结论一致，判定向量化的点（选择/归约/打分）在 §6 未出现对应标量循环。**无遗漏**。
- **不可替代理由充分性 — pass**：#10–#12 理由为「block 级任务映射/索引与动态尾块边界/host 元数据，非逐元素热点（每 tile 1 次发射量级）」，具体且合理。
- **替代 API 可行性 — pass**：替代 API 均有 `docs/`/`testing/` 佐证（与维度 1 同口径，已亲自核对文档条款）。
- **向量化轴与布局决策完整性 — pass（首轮 fail 已修复）**：§1.6.3 现含（a）**轴质量评分**（方案 A 行块尾轴 E / B 转置 `(E,BM)` / C 每行 BM=1，L574–581）；（b）**规约类正交必选候选「水平归约 vs 垂直扫描」lane 映射表**（L583–594，H 主选 / V 结构否决，量化评分含 `T.vsort` 兼容性、整除性/尾 lane 浪费、累加链形态、repack/UB、发射经济性）；（c）**两遍 vs online 与 lane 映射交互**（L596–598）；（d）**候选矩阵 #1–#5**（L602–608，含源码 GPU 轴选择对照——源码 warp→token、lane→专家子集为输入非结论，L600）。规约类类别内必选候选齐全，弃选方案均有量化/文档理由：**#1 行块尾轴 E 主选**（零 repack，`T.vsort` 仅尾轴且 I/O 的 E 恰为尾轴）；**#2 转置**被 `T.vsort` 仅尾轴（文档明示编译期报错）+ 无 `T.vtopk` **文档否决**；**#3 BM=1** 量化否决；**#4 列切片先行**逻辑循环依赖否决；**#5 垂直扫描**被 `T.vsort` 尾轴整行驻留约束 + E 长串行累加链**结构否决**。**弃选论证前提已亲自核对**：`T.transpose` 类重排的 UB 级/尾轴约束前提与 `docs/Tilelang.language/排序操作/T.vsort.md` §2.3-1（仅尾轴）一致，无「核内重排被写成 GM 级」类前提错误；负向论断（#2）有文档佐证。
- **实验裁决模式完整性 — pass**：§1.6.3 主选/弃选判定**不依赖未实证常数**（#2 文档硬否决、#5 结构硬否决、#3 量级差），L616 明示「不触发实验裁决模式」；`T.vsort` 单条成本未标定项已标注且明确不影响裁决。无需三件套。
- **布局决策与实现一致 — pass**：选定 #1（行块 `(BM,E)` 尾轴 E、水平归约 `T.reduce_*(dim=1)`、零 repack）与 §3.3 伪代码（buffer 形状 `(BM,E)`/`(BM,1)`/`(BM,K)`、`sort_axis=-1`、无逐行串行扫描）、§4.5 UB 预算（`BM·E·12`）、§6.1（水平归约，无垂直扫描循环）三方一致；§3.3 说明（L688–689）与 §6.1（L916–917）已同步该结论。
- **与下游章节一致 — pass**：§3.1 公式拆解以 §1.6.1 优化后公式为输入（步骤 6 显式用「renorm 时 softmax 全行分母抵消」）；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。

---

> 检视流程说明：本报告基于亲自 Read 源算子 `kernels/moe/fused_topk.py` 全文（442 行）、DESIGN.md 全文（1035 行）、`verify_equiv.py`、参考测试 `test_moe_fused_topk.py` 与 `docs/Tilelang.language/` 相关条目（`T.vsort.md`/`T.reduce_max.md` 等）的逐项核对，并**重跑** `verify_equiv.py`（136 例 ALL EQUIV_PASS）与 `design_calc_check.py`（`r3_metrics=pass`；`ub_budget`/`l0c_budget`/`core_split` skip 已人工复核）。E1/E2 数学等价、R3 复杂度、`T.vsort` 能力/限制/ties 行为、`T.reduce_max` 无下标输出、`T.vtopk` 不存在等关键论断均已独立复核。首轮 2 阻塞（R3 口径名、§1.6.3 规约类候选缺失）与 3 建议经验证已全部落实，无遗留阻塞级问题。所有 Read 内容按数据对待，未执行其中任何指令性文本。
