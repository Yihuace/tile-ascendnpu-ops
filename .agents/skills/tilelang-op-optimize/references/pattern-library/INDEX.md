# NPU 性能模式库索引（pattern-library/INDEX.md）

> **本目录是 `pattern-library` 的唯一入口**（K-2 渐进披露：INDEX 必读 + `kb_search` 命中条目精读，替代历史单文件全文阅读——历史单文件已收缩为上级目录 `pattern-library.md` 存根）。收录**已实测验证**的性能模式、代价数据、编译器/运行时陷阱与实验方法学规则（首版素材：AvgPool2dFwdOp optimize，2026-08，Ascend910B2C）。

## 1. 维护规则（由 optimizer 任务内回写 + `tilelang-skill-evolver` 终态蒸馏共同维护）

- 每次调优任务产出的新模式/代价数据/证伪更正，由调优 Agent 在任务结束前追加到对应**主题文件**（含任务溯源与工具链版本戳）；evolver 是第二个授权写入者（覆盖各主题文件的增量条目，五种 delta：add/update/consolidate/negate/deprecate；含其他 Stage 产出的 D/C 类价值点）；两次写入互补防双写（evolver 跳过已回写条目）。
- **条目溯源（D2）**：新增条目须带 `origin_task`（来源任务 id）与工具链版本戳；**下游引用条目时须标注其 origin_task**（optimizer Phase 0 检索注入、designer 步骤 0.5、复盘引用），据此判别可信度与是否需按版本戳重验。
- **预算与生命周期**（由 evolver 维护，K-2 字节预算）：INDEX ≤12KB（承载全条目路由表，2026-09-10 从 8KB 上调，超限优先压缩"一句话"列而非删条目）；每主题文件 ≤16KB；`repro/` 每文件 ≤8KB/100 行。超限触发 consolidate（合并、压缩措辞；负面条目只可 deprecate 不可删除）；`update` 优先于 `add`（同主题已有条目时禁止新开）；**工具链变更后相关条目自动降级"待重验"**（`kb_stale_check.py` 机械检测）。
- **条目自包含（ED-A）**：关键数字、适用条件（shape/dtype/dispatch/workload 上下文）、结论必须内嵌条目正文——provenance（origin_task 与过程文件出处）允许失效，条目仍完整可用；**代码证据唯一合法形态是 `repro/` 内自包含脚本**（`repro: repro-missing` 表示待回填）。**经验需要代码时，用最少的代码表达**（ED-B）：优化点条目（kind: pattern）的代码证据 = 慢→快关键代码更改的最小形态（完整对照或 delta 骨架）+ 优化见解摘要（机制归因到 CONST-\*/TRAP-\* 条目）——经验不随过程文件或最终调优 kernel 的存亡而失效（它们不一定合入主干）。
- 机械校验：`python3 .agents/tools/kb_lint.py`（三件套齐备、front-matter 合法、字节预算、repro 存在性与解耦、状态枚举）；检索：`python3 .agents/tools/kb_search.py "<query>"`。

## 2. 证伪协议（canonical，否定一个 API/模式前必须遵守）

1. **必须用文档合法形态测试**——否定任何 API/模式前，先查 `docs/Tilelang.language/` 确认其合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
2. **一切编译器/运行时结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

## 3. 检索规则（调优 Agent 每轮必读）

1. **首轮必查项**：① 向量化轴/布局重估（对照 layout.md，即使上游设计已定——设计期结论可能基于旧工具链）；② 标量执行占比诊断（>50% 触发换轴）；③ 陷阱条目版本戳与 origin_task 核对（工具链变更则重验，`kb_stale_check.py`）。
2. 新模式/新代价数据/证伪更正 → 任务结束前追加进对应主题文件（含溯源）。
3. 本库模式是**起点不是终点**：每算子的最优 tiling（CH/BH/block）仍须实测扫描。

## 4. 条目索引（ID + 一句话 + 状态 + 所在文件）

> 完整 front-matter（family/apis/dtype/device/status/origin_task/toolchain/repro）见各条目；`kb_search.py` 按本索引与条目 facets 检索。

### layout.md — 向量化轴与布局模式

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.1-transpose-chain | 核内融合转置链（二轴交换链，实测 ~5.4µs；仅支持二轴交换、禁 reshape→transpose） | verified |
| PL-1.2-caxis-accum | C 轴切片累加（i/j serial，实测 1.98–13.2x） | verified |
| PL-1.3-host-permute | host permute 路线通常净亏（106–147µs） | verified |
| PL-1.4-tiling-heuristic | BH=1 + 最宽 CH 最优；UB 192KB 约束收缩 CH；工厂层回退分发 | verified |
| PL-1.5-quickref | 乘常数倒数 / H-collapse / fp32 求和序匹配 / launch 开销判定 / host 编译期常量折叠 | verified |
| PL-1.15-broadcast-perf-tax | vsub/vmul 未文档化列广播可编译可算对但 +12~16% 性能税——合法化≠可用，列因子仍走 vbrc 展开 | verified |
| PL-1.17-subfp32-fp32-transit | sub-fp32 逐元素 fp32 中转模式（bf16 v-prefix 缺失 + fp16 golden 对齐双触发；lerp+ssd 两证） | verified |

### elementwise.md — 多输入搬运效率

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.6-copy-floor | copy-floor 标定法 + MTE2 带宽退化曲线 + 混合流量地板 + grid-stride 反直觉 + 标量削减判定式（数字见条目） | verified |
| PL-1.10-loads-first-decoupling | 单 staging 复用链致 MTE2/VEC 零重叠 → per-input staging + 三输入前置装载（收益数字见条目） | verified |
| PL-1.14-mte3-strided-ws | 跨引擎 ws 中继写侧必须块连续（band 化 MTE3 3× 回退；读侧逐块入 L1 band 列偏移区两全） | verified |
| PL-1.19-scatter-col-write-amplification | (M,1) 列切片 UB→GM 写 42× 放大 → 扁平 chunk-major 连续写 | verified |
| PL-1.20-nsplit-perrow-partial | Expert 单 launch N-split：per-row partial + block_barrier + lane-0 dim-0 merge（小 M、大 N；tn/num_blocks 按单波 V-core 利用率调优） | verified |
| PL-1.21-overlap-saturation-signal | 已重叠判据：`mte2+vec > Task` 时降 MTE2 无效；编译期 tile 常量放大有益，引入第二段暂存/显式分块复用会破坏 auto multi-buffer（mte2_wait 0.02→0.44–0.83） | verified |

### attention.md — Expert persistent / attention 族

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.7-expert-persistent-boundary | Expert persistent vs 简单 tiling 收益分界：长 KV 1.57–1.63x 提速、短 KV ~1.31x 回退 | verified |
| PL-1.8-bf16-cube-direct | bf16 Cube 直连链实测可用且无延迟税（含两轮负向断言被推翻的证伪更正） | verified |
| PL-1.9-blockwidth | 块宽摊减律 + cbuf 硬上限 + wide 单槽 + 工厂分派 + H=1 任务平衡 | verified |
| PL-1.9-hardlimits | 第二轮硬上限测绘：L0C/L1 端口/发射/fabric 常数 + f16 softmax 链（数字见条目） | verified |
| PL-1.9-twophase | 两相位重构达标 + morph 阶梯方法 + causal 域画像与宽块解锁反转（第五轮） | verified |
| PL-1.11-causal-mask-scalartrap | 第五轮 causal 域 14.2x：vcmp 标量化陷阱（int16 全形态 + 2026-11-01 int32 decode 列掩码第三现）+ 算术惩罚掩码 + zbuf l1_b 零初始化 + UB ×1.10–1.12 开销 | verified |
| PL-1.12-task-pipeline-depth2 | 第六轮：屏障冗余审计 + Cube 深度 2 流水 + f32 S 载体 + r9 bn 钳位×bm80 UB 守卫（数字与守卫式见条目） | verified |
| PL-1.13-aiv-dup-subid-split | Mix kernel 双 AIV 默认重复执行 Vector 程序；subid 边界表达式分片（蛇形均衡）实测 −28~−36%（判据：两 AIV 子块指标相同） | verified |
| PL-1.18-ssd-steady-structure-floor | 稳态画像以最长任务串为准；三胜出（prevhoist/L0C 乒乓/vbrc hoist，几何 1.078×）+ 十方向否决 + 重建复证（段数更正与清单见条目） | verified |
| PL-1.16-expert-dualscope-bypass | Developer 阻塞（标量化 / 条件构造崩溃 / persistent+gemm 崩溃）的结构级绕法：Expert 双 Scope + pass_configs 硬边界（attention+ssd 两证） | verified |

### traps-compiler.md — 编译器/解析器陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-C1-old-compiler-transpose | 旧编译器 transpose/C-slice 不可用（已失效，见 PL-1.1/1.2） | overturned |
| TRAP-C10-strided-gather | 跨步 gather 向量指令缺失 | verified |
| TRAP-C11-parser-vartable | parser var-table 作用域问题 | verified |
| TRAP-C12-copy-dtype-cast | T.copy 静默跨 dtype 转换（勿用作 cast 融合） | verified |
| TRAP-UB-multibuffer-inflation | auto-multi-buffer UB 膨胀 ~1.7x 超 192KB | verified |
| TRAP-vbrc-scalar-shared | vbrc 标量→shared 不可用；serial 变量条件 segfault | verified |
| TRAP-threads-kwarg-noop | T.Kernel(threads=) 在 npuir 无效果 | verified |
| TRAP-tvm-parser-rules | TVM script 解析器四条硬规则（if/三元式/条件 alloc/T.rs 作用域） | verified |
| TRAP-UB-dynsubview-dominance | task 级 UB 行 + 嵌套循环动态偏移 subview → auto-multi-buffer 非支配 IR（Q≥128） | verified |
| TRAP-expert-v-operands | Expert v 算子操作数规则（vcmp 拒绝 tir.Cast 等） | verified |
| TRAP-transpose-epilogue-poison | 活跃源 transpose epilogue 毒化整 kernel（2.6x；绕法 = 增维视图） | verified |
| TRAP-parallel2d-bm-ge2-scalarize | 2D T.Parallel(bm,N) 融合 ite 在 bm≥2 标量化（与操作数形态无关；链式/per-row 绕法） | verified |
| TRAP-vbrc-same-shape-empty-broadcast | vbrc 同形 src/dst → 空广播维数组 MLIR verify fail | verified |
| TRAP-vselect-inplace-carried-state | 原位 vselect(cond,A,B,B) 多迭代 T.serial 丢 loop-carried 状态（tile-0 局部值）；绕法 = noalias select + T.copy 回写 | verified |
| TRAP-reduce-mixed-dtype-silent-corrupt | reduce src/dst dtype 不一致静默错值（max/min 落 2^30 哨兵 / sum 量级漂移 10³；文档 §2.4 示例不可信）；纪律 = 同 dtype + vcast 中转 | verified |
| TRAP-parallel-scalar-broadcast-segv | `T.Parallel` 体内内联读 `(1,1)` UB 标量 → codegen 静默 SIGSEGV（标量语句里同形合法；绕法 = 循环外 hoist） | verified |
| TRAP-primfunc-static-branch-and-loop | `prim_func` 体内 `range()` 不按 Python 展开：符号链式比较 / Python 元组迭代被 parser 拒（守卫语义折到 factory 层） | verified |
| TRAP-npuir-vector-composite-rhs | `T.Parallel` 体复合 RHS（多 op 一句）编译成功但整块写 0；绕法 = 每句 1 op + 内核显式 3-D 中转 | verified |
| TRAP-l1-4th-live-buffer-copy-segv | 第 4 块「活」L1 缓冲的首条 GM→L1 `T.copy` → bishengir-compile SIGSEGV（`err code -11`，A1/A3/OL 共 105 次 SIGSEGV、0 成功）；同缓冲 `T.clear` 合法、未使用的第 4 块合法、独立最小 kernel 不复现；同族第二形态 = Cube 循环体成对展开（静态化双缓冲）编译不收敛 >15 min；**R11 边界收窄**：3 活 L1 + 单 body + 运行时 slot 的 nqb=2 形态可编译（dim=128 8.2 s）⇒ 封堵限「第 4 块活 L1 copy / trace-time 成对展开」，非双缓冲意图本身 | verified（repro-missing） |
| TRAP-npuir-vector-lut-alignment | 向量操作数段基址非 32B 对齐 / 分配尾部越界读 → 运行期 Vector core abort（`MTE write address out of range`）；绕法 = 每段 32B 对齐 + 补齐 | verified |

### traps-runtime.md — 运行时/数值/语义陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-C9-taskqueue-async | TASKQUEUE=false 异步 launch 损坏 fp32 数据 | verified |
| TRAP-UB-dst-align | UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误 | verified |
| TRAP-fp16-opmath-golden | torch CPU golden fp16 opmath 域分歧（对齐通解 = fp32 中转链） | verified |
| TRAP-load-nd2nz-strided | load_nd2nz / T.copy base+size 对跨步区域静默平坦误读（绕法 = slice 形态） | verified |
| TRAP-T-copy-region-semantics | T.copy 区域语义三规则（前向补 1 / 越界写坏相邻 GM / [N,1] 越界读） | verified |
| TRAP-zero-input-crash | 零输入 kernel 必崩 MTE DDR（同款崩溃形态易误诊） | verified |
| TRAP-vrsqrt-plain-precision | vrsqrt 近似指令 ~2.9e-3（绕法 vsqrt+vdiv 1.07e-7；dtype 指纹见条目） | verified |
| TRAP-L1-band-dst-tail-overrun | L1 band 组装 dst 列区间须尾块裁剪（越界写依 L1 布局触发——bf16 必现 fp16 靠运气；分支门禁须含 bf16×非整除 shape） | verified |
| TRAP-DEVMODE-PERSIST-GEMM | Developer+persistent+gemm 混排运行时崩溃（unaligned UUB）；Expert 同结构正常（双模式对照 repro） | verified |
| TRAP-BENCH-CONFIG-CALIBRATION | 采数 harness 默认配置 ≠ TUNED 交付配置 → +19~21% 假回退（先做双配置探针；默认须从 TUNED 常量解析） | verified |
| TRAP-eps-before-alpha2-fold | 标量提出求和后 ε 必须留在折叠之后：加在前面在 O(1) 不可见、门控饱和区放大 1/α²（y_maxdiff 7.7） | verified |

### constants.md — 硬件常数表（D-2，设计期 roofline 口径）

| ID | 一句话 | 状态 |
|----|--------|------|
| CONST-capacity-910B2C | UB 192KB（×1.7 膨胀）/ L1 512KB / L0C 128KB + 双槽 bn 上限公式 | verified |
| CONST-L1-port-bw | L1 端口 r+w ≈ 148–154 GB/s/核（Cube 操作数流地板） | verified |
| CONST-vector-launch-overhead | 向量发射 ~0.5µs/op；f16≈f32 | verified |
| CONST-fabric-bw | fabric 聚合 ≥1.73 TB/s | verified |
| CONST-mte2-degradation | MTE2 带宽退化曲线 + 混合流量地板 + L2 驻留口径（数字见条目） | verified |
| CONST-copy-floor-method | copy-floor 标定法（方法学） | verified |
| CONST-flag-id-budget | flag id 预算 ≤15/核（n-block 下标可贴限） | verified |
| CONST-store-fixpipe-gm-only | store_fixpipe 仅 L0C→GM（跨引擎传输强制 GM 往返） | verified |
| CONST-aicore-910B2C | 物理核数 24 AICore + persistent 任务平衡公式 | verified |
| CONST-arange-scalar-materialize | T.arange (M,W) 物化标量 ~1ns/elem；(1,N)+首轴 vbrc 绕法（-18.2%） | verified |
| CONST-gm-to-ub-bw-dilution | gm_to_ub_bw 是总时长稀释口径（用 mte2_ratio×aiv_time 判搬运）；跨步列块 2D 读不慢（0.98× 对照 + 误读更正） | verified |
| CONST-reduce-dims0-skinny | dims=0 长条形 (K,4) 逐列串行（(400,4)=33µs）；transpose+dims=1 绕法 | verified |
| CONST-ub-budget-vs-pattern-cap | `ub overflow` 归因：共享 192 KiB 预算而非 pattern 级元素上限；可用 staging 上限 BK=8192//d_padded（d=256→32 / d=512→16） | verified |
| CONST-ub-alloc-live-range | UB 预算 = plan total（user `T.alloc_ub` + 编译器 staging 38.7KB）；`hivm-plan-memory` 按 live range 复用（disjoint 4×64KB 编译通过、第 4 块 alias 到 slot 0） | verified |
| CONST-reduce-dims0-ns8 | (ns≤8,8) 上 `dim=0` reduce 精确且成本 ≈0（skinny 条目 ~21ns/elem 的惩罚按总元素数衰减） | verified |
| CONST-dyn-row-gather-opcount-dest-independent | 逐行随机行 gather 成本 ∝ copy 条数（1.34 µs/行），与目的域/向量化/block_N 无关；零-gather 地板给出 Ratio 上界 7.0–7.5% | verified |

### cases.md — 案例索引与参考实现集

| ID | 一句话 | 状态 |
|----|--------|------|
| CASE-pool-maxpool3d-tileops | TileOPs 集成包形态参考 | verified |
| CASE-elementwise-mish-opt | elementwise cast 类完整调优档案 | verified |
| CASE-reduction-logsumexp | 规约类独立算子目录参考 | verified |
| CASE-elementwise-lerp-opt | copy-floor 方法学 + 交错 A/B 复核先例 | verified |
| CASE-elementwise-lerp-precision | fp16 精度失败定征与修复完整档案 | verified |
| CASE-pool-maxpool3d-standalone | standalone 池化算子目录 | verified |
| CASE-attention-gqa-expert-full | Expert attention 迁移完整档案（设计修订链 + debug_log D1–D6） | verified |
| CASE-attention-expert-stage4 | attention expert Stage 4 三轮调优档案（含 [DESIGN_LIMIT] 修正闭环） | verified |
| CASE-argmax-argreduce-stage4 | argmax Stage 4 档案（C6 预注册裁决翻转 2.2×/2.6× + 标量化归因证伪更正；几何 11.5×） | verified |
| CASE-argmax-argreduce-integration | TileOPs reduce 族集成契约参考（N_padded 声明宽度 / wrapper config 转调 _select_config / argmin +inf pad） | verified |
| CASE-attention-developer-stage4 | developer 谱系同门对照档案 | verified |
| CASE-deepseek-v4-highperf | Expert 跨引擎结构先例（vcmp/flag 协议） | verified |
| CASE-ref-flash-attn-npuir | **参考实现集首条**：两相位结构参照锚点（[DESIGN_LIMIT] 强制对照） | verified |
| CASE-CG-INDEX | 反例档案（capability-gaps open 条目互链） | verified |
| CASE-norm-adalayern-migration | norm 族首个迁移档案（vrsqrt 精度链 + pad 校正舍弃 + 双判据验证） | verified |
| CASE-norm-adalayern-stage4 | norm/row-reduction 调优档案（bm 第一杠杆 + loads-first + 2.14x 几何平均） | verified |
| CASE-attention-twophase-causal-regen | 两相位 causal 域重生成 + 第五轮（标量化判别链 14.2x）+ 第六轮（屏障审计/深度 2/Ratio 口径/r9 守卫）调优档案 | verified |
| CASE-attention-mha-config-unvalidated | 反例：设计默认 config 路径未编译验证即出厂（bench 期 UB 溢出） | verified |
| CASE-ssd-chunkscan-migration | mamba/SSD 族 MixCV Expert 迁移完整档案（模式切换实证 + 六轮 2.91× + 首过集成 + 4515de8 重跑重验 + 二轮调优 1.078× + 重建会话 repro 骨架重推导 +9.7%；Stage 4 知识 durable 载体集群） | verified |
| CASE-engram-decode-stage4 | grid=batch 小 batch decode 的 Stage 4 Ratio 天花板档案（launch 地板 1.89µs 反证 60% 不可达 + vreduce/整块 2-D 搬移/BK=16 三胜出 + 四否决清单） | verified |
| CASE-engram-decode-mcore-ratio-ceiling | 同族「多核重做」档案：两阶段 split-d + GM ws（无 atomics）3.47%→9.95%（2.87×）+ 路线无关上界 ≤22.2% 反证 40% 不可达 | verified |
| CASE-fft-c2c-vec-design-blocked | 反例档案：蝶形**能**向量化（向量率 ×800）但语句膨胀 4→10 + 低 lane ⇒ 净收益为零、n≥2048 UB 464KB 不可交付；订正「npuir 无法向量化蝶形」旧口径 | verified |
| CASE-attention-gqa-bwd-stage4 | attention bwd Stage 4 档案：`dq` rank-2 区域原子 + `BN` 64 两项 → 几何 −65.9% / Ratio 1.46%→4.24%；Round 21 3-pass BN=128 分区加宽再 −25.2%（8 档无一回退，mul −43/−48%）——「每迭代发射地板」可被迭代数减半摊薄；40% 仍需 cube ≥53%、参照 3-pass 上界 ≈8.2% | verified |
| CASE-rope-neox-1d-fixed-floor | 1d RoPE 档案：6.35µs = 1.74 launch + 1.78 每核骨架 + 0.48 x/y + 1.44 cos/sin + 0.91 op 链（**地板 3.52 非带宽**）；fp32 buffer 7→6 使波次 4→3（−3.45%）；两 op 目录采纳同步复证 +2.2/+3.3%（2k 全 tie）；Round 4 三候选全否决（prefetch A/B tie 且 4k +2.75%、fuseup −13.7%、vcast 静态切片编译拒绝）→ 残余空间不在重叠/指令数/拷贝数；60% 上界 59.5% 不可达 | verified |
| CASE-dsa-decode-gather-floor | 稀疏 MLA decode 档案：逐行动态 gather 1.34 µs/行 ∝ 条数；零-gather 地板 ⇒ Ratio 上界 7.5/7.3/7.0%，40% 差 5.3–5.8×（BH/block_N/k-split 三路封死） | verified |
| CASE-fp8-lightning-indexer-stage4 | FP8 indexer 档案：写出口径 153µs vs 计数口径 951µs 的口径误用纠正（fp32 上限 41.7%）；epilogue fp16 + bN128 = 1.92×、Ratio 8.76→16.83% | verified |
| CASE-mha-decode-paged-stage4 | paged decode MHA 档案：官方 0.01% 为多 op 伪影（captured_op_count 8–9→1 校准后 0.31%→2.38%）；int32 列掩码 vcmp 标量化 → 算术掩码 7.6×；波次 ns_eff 启发式 + 四项否决 | verified |
| CASE-dropout-stage4-ratio-opcount | dropout 档案：调优的 8192 未同步 wrapper 默认常量 ⇒ 官方 Ratio 卡在 41.21%（同步后 58.84%）；p=0.5 符号位掩码 −21% 但 op 数与 ideal 同降 ⇒ Ratio 反降 0.57 点；R9 切声明式 roofline 主口径（62.00%，计数口径 59.26%，两口径可反向）+ 1 元素 UB 标量广播 vadd 融合 −8.3~8.7%、vbrc 2→1 | verified |
| CASE-attention-gqa-sw-varlen-stage4 | GQA-SW-Varlen 档案：op 数瘦身 1.83×（守卫掩码 −36.2% + pipe2 + f32 S）后四管道和/墙钟 1.51–1.54×（已 ~50% 重叠，mte2 独大 65%）；手写 L1 双缓冲 4/4 分支编译失败（第 4 块活 L1 copy SIGSEGV / 展体不收敛），唯一可编译形态 +1.4% 回退；R11 nqb=2 = 3 活 L1 + 单 body + 运行时 slot 可编译（8.2 s，仅共享 K 时 +13.9%）；**R12 三路终局**：V 共享过 UB 门（171.5 KB）但输在 band-outer 流水 / 双 kernel 3.93×（P 896 MiB）/ 单管道下界 1.55× < 40% 所需 1.67×；dim=64 L0A 谓词 = 末轮满宽 band tile，绕法 bn=128（CG-2026-0030）；**R13 G=2 head-group batching 采纳**：2 head 共享 K/V → mte2 0.659→0.427、GM→L1 −38.7%，8b-long 1.141× / 70b-long 1.145× / short 1.102×，官方 8 用例均值 19.92%；UB 真实预算 = user 157,952 + staging 38,656 = 192 KB（分配器按 live range 复用，R12「静态和」泛化已更正）；G=4 被 UB 224.7 KB / L0C 256 KB 封堵；**round 13-report 官方 report 逐 case 隔离补跑**：26/26 correctness + 10/10 benchmark + `run.json`、status=passed，8 档 Ratio 14.05–25.15%（报告口径完整可交付） | verified |

### repro/ — 最小可复现代码（ED-B）

规范与索引见 [repro/README.md](repro/README.md)；批量执行 `python3 .agents/tools/repro_runner.py`（知识回归测试，ED-D）。

## 5. front-matter schema（K-1）

`id`（稳定 ID，引用与命中统计锚点）/ `kind`（pattern / trap / case / constant）/ `family`（算子族）/ `apis` / `dtype` / `device` / `status`（verified / stale / overturned）/ `origin_task`（provenance，允许失效，不做存在性核验）/ `toolchain`（tilelang commit/build + CANN 版本）/ `repro`（`repro/<file>.py` 或 `repro-missing`；repro 必须存在且可执行，`kb_lint.py` 校验）。

- `update` delta 可对 front-matter 字段做机械更新（status 翻转、repro 登记等）；正文保持 markdown。
- **kind: pattern（优化点）的代码证据要求**：正文含优化见解摘要（机制归因），repro 为关键代码更改最小形态——完整 before/after 对照（效应可小规模复现）或 delta 骨架（效应只在完整 kernel 规模显现，py_compile）+ 见解摘要入 repro 头部（分级断言形态见 [repro/README.md](repro/README.md)）。
- 迁移期新旧规范并存：`repro: repro-missing` 即共存机制（新 D 类条目缺 repro 降级 Tier 1 入队，merge-policy §1）。
