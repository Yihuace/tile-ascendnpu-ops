# 算法候选库（algorithm-candidates）— 常见算子族替代算法参考表

> **版本化结构化候选库**（D-4）：由 `algorithm-research.md` §5 参考表升级而来——该文件保留调研方法论，**候选表本体以本文件为唯一事实源**。R1/R2 的命中行是**必查清单**（下界不是上界）：命中行的候选必须评估，但调研不得止步于本表——结构判据（algorithm-research.md §3 R2）与外部已知算法仍须过一遍。表内条目涉及 API 时仍须本地佐证（`examples/` / `docs/`）——**算法思路与 API 存在性是两回事**。
>
> **维护**：`tilelang-skill-evolver` 蒸馏 Stage 4 发现的"算法族 X 的候选 Y 实测收益 Z"时同步更新本表（P 类，Tier 1，target_doc 指向本文件）；任务内 optimizer 回写 pattern-library 的算法级发现同样在复盘表标注本表候选行。条目带 front-matter：`known_impl`（本仓已验证实现指针）/ `kb_links`（pattern-library 条目 ID——候选的实测代价/反例证据）。检索：`kb_search.py`（K-3 统一入口）。

---
id: ALG-softmax
family: [softmax, log-softmax, logsumexp]
known_impl: []
kb_links: [PL-1.9-hardlimits（f16 softmax 链实测）, PL-1.9-twophase（两遍式 S/P 物化先例）]
---

R1 等价化简候选：减 max 稳定化（必选）；log-softmax 直接由 logsumexp 表示；exp → exp2·log2e 缩放。R2 在线变体：online softmax（分块 running max/sum + 重缩放）。R3 复杂度要点：三遍（max/sum/normalize）vs 两遍 vs 单遍 online；行缓冲 O(row) vs O(tile)。R4 亲和要点：行内水平归约 vs 垂直扫描（与 §1.6.3 交互）；整行 UB 驻留；attention 场景 MixCV 融合。

---
id: ALG-layernorm
family: [layer_norm, batch_norm]
known_impl: [examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py（persistent 全 N 驻留 + loads-first，2.14x）]
kb_links: [PL-1.5-quickref（fp32 求和序匹配）, TRAP-vrsqrt-plain-precision（rsqrt 近似指令 2.9e-3，vsqrt+vdiv 绕法）, PL-1.10-loads-first-decoupling（staging 解耦）, CASE-norm-adalayern-stage4]
---

R1：`x/sqrt(var+ε)` → `x·rsqrt(...)`；单遍式 `var = E[x²]−(E[x])²`（须评估 fp32 累加与 catastrophic cancellation——ada_layer_norm 2026-09-10 实测定量证据入 queue VP-2026-0043，repro-missing 待补后合入）。R2：Welford 增量 mean/var；分块 running 统计。R3：两遍扫描 vs 单遍；统计量缓冲。R4：规约轴 lane 映射（见 §1.6.3）；fp32 累加路径。**row-reduction 迁移基准形态（两任务实测：logsumexp 2026-08 grid 形态 / ada_layer_norm 2026-09-10 persistent 形态）**：全 N 驻留 UB（流量 4MN 最小）+ 按 M 行块分核（persistent `num_kernels = min(ceil(M/bm), aicore×2)`，纯 Vector 核数翻倍实查）+ 尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限（fp16 约 N>11520，20 B/elem UB 律见 TRAP-UB-multibuffer-inflation）才需 N-tile 流式（+25% 流量，ada 设计期估算）。

---
id: ALG-rmsnorm
family: [rmsnorm]
known_impl: []
kb_links: [TRAP-vrsqrt-plain-precision（rsqrt 近似指令——rmsnorm 的 rsqrt 直用同样受影响）]
---

R1：无 mean 减法（相对 layer_norm 少一遍扫描）；rsqrt。R2：平方和单遍天然在线。R3：平方和单遍；无均值遍。R4：同 layer_norm（row-reduction 迁移基准形态见 ALG-layernorm R4：全 N 驻留 + M 行块分核 + 尾块三件，两任务实测）。

---
id: ALG-attention
family: [attention]
known_impl: [examples/flash_attention/flash_attn_npuir.py（两相位 flash 结构，第三轮 [DESIGN_LIMIT] 推翻者）]
kb_links: [PL-1.7-expert-persistent-boundary, PL-1.9-twophase, CASE-ref-flash-attn-npuir]
---

R1：scale 融入 Q/K；softmax 分母倒数乘法化。R2：flash 结构——online softmax + 分块 KV，O(N²) 中间矩阵 → O(N)。R3：materialize O(N²) 访存/缓冲 vs flash O(N) 缓冲（FLOPs 同阶）。R4：Cube GEMM 分块 + Vector online softmax 流水（MixCV）；L0C/UB 容量定 tile。**已知实测**：两相位结构（pass-1 全 S / pass-2 全 O_partial + Q hoist）四硬目标全达成且优于 per-block persistent 链（PL-1.9-twophase）；短 KV 分档 expert 形态回退 1.31x（PL-1.7，shape 分派权衡见 T-6 多 dispatch 汇总）。

---
id: ALG-conv
family: [conv]
known_impl: []
kb_links: [PL-1.2-caxis-accum, PL-1.1-transpose-chain]
---

R1：im2col + GEMM；implicit GEMM；Winograd（小核乘法次数↓加法次数↑，须按单元吞吐比评估）。R2：滑窗天然流式。R3：direct vs im2col——FLOPs 同阶、访存与单元利用率不同；Winograd 乘法按窗口比例下降。R4：Cube 路径分形对齐；Vector 直接窗口的 C 轴整除性；im2col 展开缓冲。

---
id: ALG-pooling
family: [pooling]
known_impl: [examples/TileOPs/tileops/kernels/pool/avg_pool2d/avg_pool2d_kernel/perf_opt/]
kb_links: [PL-1.2-caxis-accum, PL-1.3-host-permute, PL-1.4-tiling-heuristic]
---

R1：avg pool = 常数权重卷积；sum pool × 常数 = avg pool。R2：滑窗天然流式。R3：窗口重叠数据的重复读。R4：C 轴向量轴；跨步系数（见 §1.6.3）。**已知实测**：核内融合转置链（NCHW→NHWC）~5.4µs 可忽略；host permute 106–147µs 通常净亏（PL-1.1/1.3）。

---
id: ALG-reduction
family: [reduction, statistics]
known_impl: [examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/]
kb_links: [CASE-reduction-logsumexp, CASE-norm-adalayern-stage4（persistent 形态调优档案）]
---

R1：max + sum 合并扫描；和与平方和一次扫描。R2：分块归约 + 合并（两阶段 / 树形）；Welford。R3：扫描遍数；跨核归并代价。R4：水平 vs 垂直 lane 映射；跨核 sync 代价。row-reduction 族迁移基准形态（全 N 驻留 UB + M 行块分核 + 尾块 T.min 双显式 slice——logsumexp grid 形态 / ada_layer_norm persistent 形态两任务实测，详见 ALG-layernorm R4）。

---
id: ALG-topk
family: [top-k, sort]
known_impl: []
kb_links: []
---

R1：部分选择 vs 全排序；分块 top-k + 归并。R2：分块 + 归并。R3：O(N·k) vs O(N log N)。R4：归并链的核内结构与缓冲。

---
id: ALG-gemv
family: [gemv, matvec]
known_impl: []
kb_links: [TRAP-load-nd2nz-strided（Cube 路径装载形态约束）]
---

R1：GEMM 分形路径（Cube，load_nd2nz）vs Vector 规约累加。R2：split-K 分块累加。R3：Cube 利用率 vs 归约代价。R4：K 维分块与 L0 容量。

---
id: ALG-triangular
family: [triangular-solve, inverse]
known_impl: []
kb_links: []
---

R1：只需解不需逆时——直接求解替代显式求逆（同阶但常数更小、数值更稳）；对角/三角特例降阶。R2：—。R3：求逆 + 乘法 vs 单次求解；特例 O(n²)/O(n)。R4：Cube 三角 GEMM 支持。

---
id: ALG-elementwise
family: [elementwise, activation]
known_impl: [examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/]
kb_links: [TRAP-fp16-opmath-golden, PL-1.6-copy-floor, PL-1.5-quickref]
---

R1：公共子表达式消除；除法乘倒数；cast 链合并。R2：逐元素映射天然单遍。R3：融合消除中间 GM 往返（访存收益主导）。R4：单遍 Vector 流；dtype 路径。**已知实测**：sub-fp32 逐元素算子 fp32 中转链对齐 torch golden（≤1 ulp，TRAP-fp16-opmath-golden）；≥16M 档计算链被 MTE2 窗口完全隐藏（PL-1.6-copy-floor，计算链优化 ROI≈0）。

---
id: ALG-transpose
family: [transpose, repack]
known_impl: []
kb_links: [PL-1.1-transpose-chain, PL-1.3-host-permute, TRAP-transpose-epilogue-poison]
---

R1：核内融合转置链（UB 级，实测 ~µs 级）vs host permute（~百 µs 级）。R2：分块转置。R3：GM 流量不变，代价在向量管线开销。R4：T.transpose dtype 矩阵（fp16/fp32 ✓，bf16/整型 ×）；UB 容量。**已知反例**：epilogue 活跃源 transpose 毒化整 kernel 2.6x（TRAP-transpose-epilogue-poison，绕法 = 增维视图）。
