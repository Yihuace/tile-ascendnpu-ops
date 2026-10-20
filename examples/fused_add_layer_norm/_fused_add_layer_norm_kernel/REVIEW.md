# _fused_add_layer_norm_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md
- 任务类型: 迁移（harness 模式）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/norm/fused_add_norm.py（`FusedAddLayerNormKernel`，L48–240）；本函数迁移精确源 = examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/_fused_add_layer_norm_fwd_kernels.py
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（与 §1.6.1 内嵌表一致，全 EQUIV_PASS）**
  - 命令: `python3 examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/verify_equiv.py`（torch CPU，fp64 参照）
  - 结果: 采纳项 O1+O2+O3 联合（24 案例 = 8 形状 × 3 dtype）**全部 EQUIV_PASS**——fp16/bf16 舍回后 max ulp = 0（逐位一致），fp32 max ulp ≤ 1（低于容差 1e-5），违反率全 0。判别段 A（矩式拒绝证据）fp32 max violation = 0.2099 > 0、fp16 = 0.1554 > 0（DISCRIM_OK，bf16 = 0.0005 被宽容差掩蔽）；判别段 B（round-first vs 源 unrounded 统计）24 案例违反率全 0（DISCRIM_OK）。三者与 DESIGN §1.6.1 内嵌表逐格一致（表内 fp32 0.2100 / fp16 0.1554 / bf16 0.0005 与实测 0.2099 / 0.1554 / 0.0005 仅末位四舍五入差异）。
- **design_calc_check: 附 JSON checks 摘要**（`python3 .agents/tools/design_calc_check.py --design examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md`；注：调度 prompt 给的 `--dir` 参数该脚本不支持，实际以 `--design <path>` 运行）:

```json
{
  "design": "examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md",
  "capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0, "source": "pattern-library/constants.md CONST-capacity-910B2C"},
  "checks": [
    {"check": "ub_budget", "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check": "l0c_budget", "status": "skip", "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'bm': 1}; declared logical cores = 64"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```

  - **skip 项人工复核结论**:
    - `ub_budget`（机械跳过，人工已核）——按 §4.5 逐档复算 `总驻留 = bpe·bm·N + 4N`（bpe=10 fp16/bf16、8 fp32，4N=w1），全部 ≤192KiB（含 ×1.7 膨胀）：fp16/bf16 N=4096 bm=2 = 98,304B(96KiB)×1.7=163KiB ✓；N=8192 bm=1 = 114,688B(112KiB)×1.7=190.4KiB(99.2%) ✓；fp32 N=4096 bm=3 = 114,688B(112KiB)×1.7=190.4KiB ✓；N=5120 bm=2 = 102,400B(100KiB)×1.7=170KiB ✓；N=8192 bm=1 = 98,304B(96KiB)×1.7=163KiB ✓。见维度 2。
    - `l0c_budget`（机械跳过）——纯 Vector 无 Cube/L0C，不适用，人工确认 §4.4/§5.3 无 Cube 路径。
    - `core_split`（机械跳过，人工已核）——`num_kernels = min(num_logical, 40)`（40 = 20 AICore × 2，纯 Vector 翻倍，core-split-strategy.md §1 + CONST-aicore-910B2C）；`num_local_tasks = ceil(num_logical/num_kernels)` 算术自洽（1024→26、2048→52），logical 表与 manifest workload 一致。见维度 3。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源源码 `TileOPs-main/src/tileops/kernels/norm/fused_add_norm.py`（`FusedAddLayerNormKernel`，L48–240）、`norm/call_spec.py`、`tiling.py`、`ops/norm/fused_add_layer_norm.py` 与 Stage 0 提取件，逐项与 DESIGN §0 核对。

- **语义理解正确性（源码证据）**: §0.1 的数学语义与源码一致——源码 L96–99 `add_f32 = cast(x_local)+cast(r_local)`（fp32 融合加），L106–108 行和 + `acc[i]/float(N)`（除以真实 N），L111–116 中心化平方（`(add_f32-mean)^2`，pad 列 `if j<N else 0`），L118–120 `rstd = T.rsqrt(acc/N+eps)`（eps 加在方差上、ddof=0），L124–132 y 仿射、L135–137 `residual_out = x_local`。**双输出契约**（`residual_out=x+residual`、`y=LayerNorm`）与 §0.1/§0.2 一致，`out_idx=[4,5]` 与源 L52 一致。**舍回语义**：§0.1「residual_out = round_dtype(x+residual)，y=LayerNorm(residual_out)」与源码 L102–103 隐式截断 + L127 重 cast 一致。
- **一处细微差异已如实标注**: §0.1 明确写出「源码 S1/S2 在未舍回 fp32 和 `add_f32` 上算，中心化/输出用舍回 `x_local`；golden 在舍回值上算 S1/S2」，与源码 L106（reduce `add_f32`）/ L127（`cast(x_local)`）**逐句吻合**，非臆造亦无遗漏。本设计取 golden 对齐的「舍回后统计」（§1.6.1 O4），差异 ≤1 dtype ulp 经判别段 B 机器验证（0 违反）。
- **算法解读完整性（源码证据）**: §0.3 步骤表 L1–E2 覆盖源码全部计算语句（装载 L76–93 / 加 L96–99 / 舍回 L102–103 / S1 L106–108 / S2 L111–120 / E1 L124–132 / E2 L135–137），数据流图覆盖 x_local/r_local/add_f32/acc/mean_val/rstd 全部 fragment，host 侧逻辑（reshape/pad/crop/lru_cache/JIT 特化）与 `forward` L214–240、Op L82–91 一致。**无误判/无遗漏计算步骤**。
- **优化手段覆盖完整性（源码证据）**: §0.4 十二项可逐项溯源——register fragment 驻留（L67–69）、融合加（L96–99）、双输出（L52/L135–137）、256-pad（L50，`ALIGNMENT=256` 见 `tiling.py` L7）、中心化两遍（L111–116）、fp32 fragment（L69）、融合 epilogue（L124–132）、谓词掩码（L76–93/L125）、`block_m=1 + threads=128`（`_config.py` `select_row_config()` → `{"block_m":1,"threads":_DEFAULT_THREADS}`，`_DEFAULT_THREADS=128`）、lru_cache（L48）、T.Parallel 直读（L74–75）、out_idx（L52）。未把实现层优化误判为算法层；判据（源码行 + 硬件特性 + 耦合性）自洽。
- **耦合性判定合理性**: §0.5 每项四态处置有依据且与 GPU→NPU 映射不矛盾——fragment 驻留→`T.alloc_shared`(UB)「等价替换」、fp32 计算→vcast 链「等价替换」、256-pad→「舍弃」（NPU 32B 由编译器管理）、threads→「舍弃+重新设计」、谓词掩码→「重新设计」。舍弃项均写明理由（pad/谓词/threads 意图在 §0.6 末尾汇总承接，无静默丢弃）。无「warp shuffle 判保留」类错误（源码本函数未用 warp shuffle/cp.async）。
- **NPU 重设计可行性（§0.6 R1–R5）**: 每项「源方案→NPU 新算法→语义保持论证」三要素齐全，且满足 NPU 约束——R1 一维 `T.Kernel(num_kernels,is_npu=True)` persistent 分核、核内 serial 边界为编译期常量；R3 尾块 `T.min`+src/dst 双显式 slice+垃圾行丢弃（规避越界，行独立规约保证垃圾行按行隔离）；R4 bf16 无 Vector 算术故 fp32 中转**必需**（有 docs 佐证）；R5 列因子走 vbrc（文档化 `(1,N)→(bm,N)`，规避未文档化列广播税）。语义保持论证具体（行间独立、累加顺序不变、fp32 中转无损上行+单次 RNE 下行）。
- **设计与决策一致性**: §1–§7 描述的是 §0.5/§0.6 决策后的算法——§1.4 列出无 pad/persistent/bm 预算表/fp32 中转/vbrc/rstd 组合等差异来源；§3.3 伪代码、§5.5 分核、§6 循环均与 §0.6 R1–R5 对应，无「§0 判重设计、§3 照抄源」脱节。
- **golden 独立性（§8.1）**: golden 直接调用 `torch.nn.functional.layer_norm`，与源仓 `tests/ops/test_fused_add_layer_norm.py:ref_program`（`added=(x.float()+residual.float()).to(x.dtype)` → `F.layer_norm(added.float(),...)` → `.to(x.dtype)`；实测 L62–72 与 §8.1 逐句一致）同构，**不含** NPU 设计的 persistent 分核/vbrc/fp32 中转/vsqrt+vdiv 组合等任何算法特征，验证独立。

**维度 0 结论: pass**（语义/算法/优化/耦合性/重设计/一致性/golden 独立性 全部与源码核对通过）。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有文档/示例佐证，且逐一核对存在：`T.Kernel`（一维 `is_npu=True`，`docs/Tilelang.language/编译器提示操作/T.Kernel.md` ✓）、`T.alloc_shared`（Developer 映射 UB ✓）、`T.copy`（slice 形态 ✓）、`T.vcast`（f16↔f32 / bf16↔f32 rint ✓）、`T.reduce_sum`（fp32 ✓ bf16 ×）、`T.vmul/vsub/vadd`（fp16/fp32 ✓ bf16 ×、行广播 [M,N]↔[M,1] ✓）、`T.vsqrt/vdiv`（组合 rstd ✓）、`T.vbrc`（`(1,N)→(bm,N)` 文档化例 2 ✓，bf16 支持）、`T.min`（标量 ✓）、`T.serial`+if 守卫（ada/lerp 佐证）。v-prefix 优先，npuir_xxx 仅作兼容。Developer 模式自动同步，无手动 set_flag。§3.5.3 引用的 examples 路径经 Glob 全部真实存在。**建议级**：`T.copy(weight[0:N], stage[0:1,0:N])`（源 1D `(N,)`→目的 2D `(1,N)`）秩不同，§3.4 自述「src/dst 同 shape」，实现期需确认 slice 形态接受秩提升或改经中间 1D buffer；不影响设计可行性（有 ada/logsumexp 同族先例）。

### 2. 内存层级规划 — pass

§4.4 搬运路径完整：GM→UB（无 GM→L0 直连），全部计算在 UB 内闭环，无中间 GM 往返；每个 buffer 标注 UB 层级（§4.3）。§4.5 UB 预算经人工复算全部达标（见「机械复核」ub_budget skip 项），最大 workload fp32 N=8192 bm=1 = 96KiB（×1.7 = 163KiB）、全域最大 fp16/bf16 N=8192 bm=1 与 fp32 N=4096 bm=3 = 112KiB（×1.7 = 190.4KiB，99.2%）均 ≤192KiB。L0C 不适用（纯 Vector）。**建议级**：§4.5 表单位标注不一致（原始驻留列用 KiB 标「KB」、×1.7 膨胀列用十进制 KB 标「KB」，如 fp16 N=8192 标 112KB→195KB），但最终判定「≤192KB ✓ 99.2%」以 KiB 口径（190.4/192）自洽，不影响结论。

### 3. Tiling 策略 — pass

§5.2 block 划分明确：`block_m = min(8, max(1,(UB_SAFE_BYTES−4N)//(bpe·N)))`、`block_n = N`（全 N 驻留），默认公式复现 §4.5 表值（fp16 4096→2、8192→1；fp32 4096→3、5120→2、8192→1，人工复算一致）。对齐：N 无 256 对齐要求（GPU 约束消除），尾轴向量宽整除（manifest 形状 0% 尾 lane 浪费）。非整除：M 尾块经 `T.min`+slice（§5.4/§1.6.2#11），N 精确无 pad。**分核策略三要素齐全**：① 逻辑核数 `num_logical=ceil(M/block_m)`（§5.5 表与 manifest workload 逐行一致）；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查=20（记录查询代码+返回值），纯 Vector 翻倍 → 40（符合 core-split-strategy.md §1）；③ 规模判定 persistent 统一式 `min(num_logical,40)`+核内 serial+if 守卫，循环边界静态（编译期常量，PL-1.5）。核数与 block 取值自洽；load 不均 ≤1 task（3.8%/1.9%）已处理。

### 4. 技术约束检测 — pass

§3.5 逐项核对：无三维 Kernel（一维 `T.Kernel`）；GPU 专用 API（threads/sync_threads/谓词）已在 §0.5 舍弃/替换；无 GEMM 分形/非整除；L0C 不适用；物理核数适配见 §5.5（实查 20→40）；bf16 无 Vector 算术经 fp32 中转链处理；列因子走 vbrc。均已给出方案，无遗留阻塞约束。

### 5. 循环与同步 — pass

§6.1 明确循环类型：任务维 `T.Kernel(num_kernels)`+`T.serial(num_local_tasks)`、元素维向量化（无循环）、尾块标量 `T.min`+slice。§7 同步与 Developer 模式匹配（自动同步，无手动 sync_block_set/wait，无核间协作）。§6.4 尾块逻辑完整（M 尾行 slice 截断、`num_logical % num_kernels ≠ 0` if 守卫、N 无尾块）。§6.3 说明不显式 `T.Pipelined` 的理由与 Stage 4 备选。与 §1.6.2 结论一致（kernel 内无逐元素标量循环）。

### 6. 验证方案 — pass

§8.1 含 PyTorch 参考实现草案（golden 独立，见维度 0）；§8.3 L0 计划含具体 shape/dtype/block_m/精度目标（13 项，覆盖 fp32/fp16/bf16、M 尾块 1025×4096、manifest workload、3D reshape、非 2 幂 N=5120/1152、微型 N=2）；§8.2 精度标准取自源仓 `_get_tolerances`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，实测源码一致）。两输出各自比对。

### 7. 完整性与一致性 — pass

§0–§11 章节齐全（迁移类含 §0，0.1–0.7 子节齐全）；无 `{placeholder}`/`TODO`/`待补充`；§3.5.3 引用的 examples 路径经 Glob 全部真实存在；API 映射（§3）、内存规划（§4）、Tiling（§5）三处描述相互一致。源路径、manifest roofline（`flops=8*M*N`、`bytes=(4*M*N+2*N)*elem_bytes`，norm.yaml L102/L105）与 DESIGN §0.7/§1.6.0 一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在**: 含 1.6.0 调研、1.6.1 数学等价优化、1.6.2 向量化替代、1.6.3 向量化轴与布局决策四子节。
- **调研存在性与完整性**: §1.6.0 四问齐全——R1 候选表**含基线候选 #0（GPU 源算法）**；R2 在线算法**明确有结论**（「有 Welford/分块 running 变体，但不采纳」+结构依据）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表、口径写清）；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）。
- **调研结论独立复核（复算/复核）**:
  - **R1/R2 负向断言核对**：对照 `tilelang-op-design` skill `references/algorithm-candidates.md` `ALG-layernorm` 命中行——该行确实记载「Welford 增量 mean/var；分块 running 统计」（确认设计称「有在线变体」成立）与「row-reduction 迁移基准形态：全 N 驻留 + M 行块分核 + 尾块三件」，设计选用算法与该命中行基准形态一致；`ALG-rmsnorm`/`ALG-reduction` 命中行亦一致。设计**未**做「无在线变体/无化简公式」式错误负向断言，无被推翻项。
  - **R3 复杂度复算**：基线两遍驻留 Bytes = 读 x + 读 residual + 写 y + 写 residual_out = 4MN·eb（+ 2N weight/bias，与 manifest `(4MN+2N)·eb` 一致）✓；N-tile 流式 = 统计遍读 x/res 重算 s 或经 residual_out 往返 = 5MN·eb ✓；prefill 2048×4096 fp16 复算 4MN·eb = 4·2048·4096·2 = 67.1MB、5MN·eb = 83.9MB（+25%）✓，与设计数字一致，未漏计中间缓冲 GM 往返（设计明确声明本算子无中间往返项）。
  - **R4 亲和淘汰核对**：矩式（R1#2）以 D-1 机器证据（判别段 A 违反率 fp32 0.21 / fp16 0.155）否决，非纸面论证；Welford 以 Vector 不亲和的逐 tile 标量合并链否决；N-tile 以 +25% 流量否决。容量淘汰引用 `constants.md` CONST-capacity-910B2C（192KB/AIV）✓。
  - 互联网来源：无（§1.6.0 声明本地源全覆盖，符合证据规则）。
- **调研→设计一致性**: §1.4 算法描述、§3.1 公式拆解（输入声明为 §1.6.1 优化后公式）、§6 循环结构均与 §1.6.0 选定「中心化两遍驻留式」一致；迁移任务 §0.5/§0.6 处置与调研选族一致，源算法被替换项（pad/谓词/threads）意图承接在 §0.6 末尾逐项说明，无静默丢弃。
- **数学等价优化完整性 + 等价性独立推演**: §1.6.1 对选定算法系统审视，采纳 O1/O2/O3/O4（+R5），逐项四要素齐全。独立推演：O1 `S/N → S·fl(1/N)` 在 N 编译期常量下成立（N=2^k 精确，否则 fp64 ≤1 ulp，D-1 佐证 fp16/bf16 逐位一致）；O2 `sqrt(v)/v = 1/sqrt(v)` 对 **v>0** 恒等（v=var+eps>0）；O3 `d` 同值复用恒等（0 ulp）；O4 round-first 与 golden 路径逐项一致（判别段 B 0 违反）。**建议级**：O2 恒等在 v=+inf 处退化（`sqrt(inf)/inf = NaN` vs 源 `rsqrt(inf)=0`）——仅当 `d²` fp32 上溢（|d|≳1.8e19，远离本任务 workload 与 §8.3 有限随机/角点输入域）才触发，§9.1「inf/NaN 自然传播与源一致」对该极端角点略有过度概括；建议在 §9.1 注一句该边界或将 v=+inf 加入判别段。
- **向量化替代覆盖完整性**: §1.6.2 覆盖方案全部循环/标量点，与 §3.3 伪代码、§6 循环结构交叉核对——逐元素计算（cast/加/减/乘/vsqrt/vdiv/规约/vbrc/仿射）**已全部向量化**（替代 API 有 docs 佐证，优先 v-prefix）；保留 3 类（#10 任务映射索引、#11 动态边界 min、#12 host 编译期分支）逐项理由具体（block 级/元数据/编译期，非逐元素热点），无「实现简单」类泛理由。§6.1 确认无逐元素标量循环。
- **替代 API 可行性**: vbrc/vcast/vsqrt/vdiv 等均有 `docs/Tilelang.language/` 佐证（与维度 1 同口径），优先 v-prefix。
- **向量化轴与布局决策完整性**: §1.6.3 规约类必选候选齐全（水平归约 vs 垂直扫描两种 lane 映射已分别评估，与「选哪条轴」正交；另评估 N-tile 流式与转置布局；两遍 vs online 交互在 §1.6.0 R2 覆盖）。候选矩阵 5 行逐项量化评分（整除性/尾 lane/累加链/repack 代价/UB 影响），弃选均数量化（#2 +25% 流量、#3 跨步无向量宽+TRAP-C10、#4 host permute 百 µs/核内转置纯开销、#5 2MN 额外 GM）。**弃选论证前提核对（亲自开文档核对）**：核内重排机制与 `docs/Tilelang.language/创建操作/T.transpose.md`（存在）一致，未写成 GM 级 transpose；列广播「可编译但有 +12~16% 税」与 `pattern-library/layout.md` PL-1.15-broadcast-perf-tax（实测 vsub `[M,N]−[1,N]` +12.2%/+16.0%，文档仅列同 shape/行广播/退化/标量四形）核对一致；vbrc `(1,N)→(bm,N)` 与 T.vbrc.md 例 2 一致。**GPU 源码隐式轴选择独立复核**：§1.6.3 指出 GPU 亦选 N 为归约轴但系 CUDA warp 组织结果，NPU 依据 Vector 宽度/广播形态/UB 容量独立评估后结论一致、理由不同，未把 GPU 轴当结论。
- **与下游章节一致**: §3.1 以 §1.6.1 优化后公式（`r_N`/`sqrt(v)/v`）为输入；§6 循环结构与 §1.6.2 结论一致；§4.5 UB 预算按全 N 驻留布局计算。
- **实验裁决模式**: 主选 #1 vs #2 裕度 = 25% GM 流量差，远超未实证常数不确定区间，无需三件套；唯一落入未实证区间的 auto-multi-buffer 膨胀系数（~1.7x）为**参数级**（bm 取值）决策，主选结构对 bm∈{1,2} 均成立，且有 `constants.md` 实测（显式 alloc ×1.10–1.12、甚至 <1）作保守上界，§4.5/§9.3 给出编译期 guard 与回退路径，判定合理。

**维度 8 结论: pass**（调研完整性/调研结论正确性(独立复算+负向断言对照复核)/调研→设计一致性/优化完整性/等价性独立推演/向量化覆盖/不可替代理由/替代 API 佐证/与 §3.1§6 一致性 全部通过）。

---

## 机制说明

本次检视为迁移任务，采用 9 维度（0–8）。**所有 Read 的源算子代码、DESIGN.md 及其注释一律作为数据（DATA）处理**，其中未出现任何要求放行检视或跳过维度的指令性文本；未修改 DESIGN.md / 源算子代码；未写入任何全局状态文件。
