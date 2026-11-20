# _gemm_w4a16_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/gemm_w4a16/_gemm_w4a16_kernel/DESIGN.md`（**v2 · revision_index=2**，已按第 2 轮检视修订；上一版 `history_version/design_v1.md`）
- 任务类型: 迁移（harness，Developer / Cube+Vector MixCV，W4A16 融合反量化）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/gemm_w4a16.py`（`_gemm_w4a16_kernel`，L1–216）＋ op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py`（`GemmW4A16Op`，L352–512）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1（§1.6.0 R3/D-2 的 FLOPs 量纲笔误；非阻塞，已在维度 8 注明）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。`python3 examples/gemm_w4a16/_gemm_w4a16_kernel/verify_equiv.py` → `TOTAL 13 checks, 0 FAIL`，13 行全部 `EQUIV_PASS`（E1×2 / E2×8 / E3×2 / E4×1）；E4 重跑 `max |cand-golden| = 3.906e-03`（对照 atol=7e-2 / rtol=5e-2），与 §1.6.1 内嵌值一致。脚本循环为 dtype 外层、K 内层，行序与文档内嵌表不同但**集合逐行一致**（名称/dtype/ulp/违反率/结论全同）。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design <path>` JSON checks 摘要：

  | check | status | detail（原文摘要） | 人工复核结论 |
  |-------|--------|--------------------|--------------|
  | `ub_budget` | pass | `UB: 3 buffer entries sum = 0.8 KB; capacity = 192.0 KB; x1.7 auto-multi-buffer inflation budget = 113 KB` | **部分解析通过**：脚本仅命中 3 条 buffer（漏解析 §4.5 逐 buffer 表多数行）。**人工独立复算**（§4.5 声明形状，block_M=16/block_N=64/block_K=64/2-stage）：`4096 + 49152 + 16384 + 65536 + 16384 + 2048 = 153,600 B = 150.0 KiB`，与 §4.5 汇总行一致；×1.10–1.12 → **≈165–168 KiB < 192 KiB（196,608 B）** → 未溢出（×1.7 口径下的裕度风险见维度 2 的 warn）。 |
  | `l0c_budget` | pass | `block 16x16 x fp32 accum = 1.0 KB vs L0C 128 KB` | 通过（`16x16` 系解析器按 `block_M=16` 的顺序匹配所致）；真值 `16·64·4 = 4,096 B = 4.0 KiB < 128 KiB`，亦通过。 |
  | `core_split` | fail | `recomputed ceil(1/16) x ceil(8192/64) = 128 vs declared 20` | **解析器伪失败**：正则把 §5.5「② 物理核数依据…**实查值 = 20**」的**物理**核数当成了声明的**逻辑**核数。**人工核对**：§5.5 workload 表 decode 行声明逻辑核数 = 128，独立复算 `ceil(1/16)×ceil(8192/64) = 1×128 = 128` 一致；`num_launched = num_logical` 三处（§0.6 R3/§3.3/§5.5）一致，无真实缺陷。 |
  | `r3_metrics` | pass | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` | 通过（存在性检查；四口径**数值**的独立复算见维度 8）。 |

  无 `skip` 项。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文（`kernels/gemm_w4a16.py` L1–216）、op 类 `GemmW4A16Op` 关键段（`ops/gemm.py` L352–512）、打包/基准实现（`examples/TileOPs/tileops/workloads/gemm.py` L52–81、`examples/TileOPs/tests/ops/test_gemm.py` L138–212）与 NPU manifest（`examples/TileOPs/tileops/manifest/gemm.yaml` L80–137），逐项核对 §0：

- **语义理解正确性 — pass**。§0.1「`out = A·Wᵀ`、`W=(q−z)s`、`g=k//128`、fp16 输出 / fp32 累加」与源码一致：源码 `output_local = T.alloc_fragment(..., "float")`（L70，fp32 累加）、`T.gemm(..., transpose_B=True)`（L161，NT 形态）、`output: T.Tensor((m, n), dtype)`（L58）。打包约定「低 nibble = 偶 k、高 nibble = 奇 k」经直接 Read `workloads/gemm.py` L77 `packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)` 确认，且与 `ops/gemm.py` L356–358 docstring 一致。**输出 shape `(M,N)`** 与源码 `_infer_output_shapes`（L430–437）`return {"output": (activation_shape[0], packed_weight_shape[0])}` 一致；manifest `shape_rules` 末条 `output.shape == (activation.shape[0], packed_weight.shape[0])` 亦同。dtype 契约经源码实测：`_validate_dtypes`（L387–397）对 `activation.dtype != torch.float16` 抛错；`GroupW4A16Op.__init__`（L368–371）对 `group_size != GROUP_SIZE` 抛错。边界语义（`k % group_size == 0` L23–24、`group_size % block_k == 0` L47–48）与源码一致。m=1 无 GEMV 快路径的断言与源码相符（`GemmW4A16Op` 无 gemv 分支）。
- **算法解读完整性 — pass**。§0.3 步骤表（步骤 1–11）逐条对上源码计算语句：grid `T.Kernel(ceildiv(n,block_n), ceildiv(m,block_m), threads=128)`（L60–64）、alloc（L65–74）、`annotate_layout` swizzle（L76–81）、`T.clear`（L86）、`T.Pipelined`（L88）、A 谓词装载（L91–96）、packed 快/慢路径（L98–109）、元数据装载（L111–121）、`T.serial(chunk)` 解码（L123–155）、`T.gemm`（L157–163）、epilogue 守卫（L165–167）；host 侧 `forward`（L206–216）与 `lru_cache(maxsize=32)`（L15）均已列出。无遗漏步骤、无臆造步骤。
- **优化手段覆盖完整性 — pass**。§0.4 的 W1–W14 覆盖源码可识别的主要手段（INT4 打包契约、融合反量化 L123–163、`T.Pipelined` L88、A16 TC GEMM + fp32 累加 L157–163、线程级 chunk 解码 L123–155 及行内 `scale_row` 提升 L125–127、`policy=FullRow` L162、swizzle L76–81、谓词 + M/N 守卫 L91–96/165–167、n 整除快路径 L98、`lru_cache` L15、fast-math/nvcc L33–39）；W1/W2/W8 正确标为算法层本体、W7/W9/W10/W14 标为实现层硬件相关，未见层级误判。
- **耦合性判定合理性 — pass**。§0.5 逐项四态处置与 `migration-analysis.md` §5.3 一致：SMEM swizzle → 舍弃（NPU 无 bank conflict）、`policy=FullRow` → 舍弃（`T.gemm` 无 policy 参数）、线程级 chunk 解码 → 重新设计（`TRAP-threads-kwarg-noop`，源 `T.get_thread_binding(0)` L85 + `threads=128` L45 确为线程模型依赖）、私有 TIR 内建 → 重新设计（探针实测阻断）。舍弃项均写明理由，未静默丢弃。
- **NPU 重设计可行性 — pass**。§0.6 R1–R6 每项含「源方案 → NPU 新算法 → 语义保持论证」：R1（Vector 反量化 → Cube 消费，先例 `test_mix_kernel_bf16.py`）、R2（符号无关 cast-only 解包，探针 P2 bit-exact；本检视独立推演见维度 8）、R3（一维 grid + cid 双射）、R4（整块向量 op 取代线程分片）、R5（动态 extent 取代谓词，A/W 双清杜绝 `0·NaN`）、R6（`block_K | group_size` 不变式）。重设计满足一维 Kernel / 容量 / 分形 / 对齐约束；K 顺序累加、fp32 累加域不变，精度影响已论证。
- **设计与决策一致性 — pass**。§1–§7 落实的即 §0.5/§0.6 判定后的 NPU 算法（核内 cast-only 解包、UB 反量化缓冲 → `T.gemm`、一维 grid、舍弃 swizzle/policy/fast-math 均在 §3.3 伪代码与 §4 内存规划中体现），无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性 — pass**。§8.1 golden 以 §0.1 语义为唯一依据（§3.3 未参与）：`golden_gemm_w4a16` 从 packed nibble 重建 `q`、以 fp32 `(q−z)·s` 反量化后 `.to(activation.dtype)`，与源仓 `quantize_weight_int4`（`workloads/gemm.py` L69–78）+ `gen_inputs` 的 cast（L114）一致，等价于 `ref_program` 的 `torch.matmul(activation, self.dequantized_weight.T)`（`test_gemm.py` L143 实测）；**未复刻** §0.6 的 NPU 算法（fp16 域反量化 / 符号无关解包 / interleave）。容差 atol=7e-2 / rtol=5e-2 与 `test_gemm.py`（`test.check(..., atol=7e-2, rtol=5e-2)`）一致。
- **第 2 轮修订核对 — pass（源事实已更正）**：§0.3（L105）现将 chunk 循环更正为「源要求 `(block_n*block_k//2) % (threads*4) == 0`（整除而非相等，L49–L50）；默认 `2048 = 4×(128×4)` ⇒ `T.serial(...)` = `T.serial(4)`，即 chunk 循环 **4 次**」，与源码 L49–50（`!= 0` 抛错，即整除约束）与 L123（`T.serial(block_n*block_k//2 // (threads*4))`）逐字一致；v1 的「退化为 1 次」算术错误已消除。

### 1. API 可行性 — pass

§3.2/§3.4 列出的 API 均有佐证且实际存在：`T.Kernel(..., is_npu=True)`（`examples/gemm/example_gemm.py`）、`T.alloc_shared`/`T.alloc_fragment`（`docs/Tilelang.language/内存操作/T.alloc_shared.md`）、`T.copy`（动态 extent 尾块，`T.copy.md` §2.2.2 规则 2）、`T.Pipelined`、`T.clear`、`T.gemm(src1,src2,dst,initC=,b_transpose=)`（`T.gemm.md` §2.1/§2.3）。`T.npuir_cast` / `T.npuir_mul` / `T.npuir_sub` / `T.npuir_interleave` / `T.vbrc` 在 `tilelang/language/` 有导出且经探针 P2 实机跑通；`NPUUtils.get().get_aicore_num()` 在 `tilelang/utils/npu_utils.py`。v-prefix 优先（用 `T.vbrc` 显式广播而非隐式列广播，规避 `PL-1.15-broadcast-perf-tax`）。`T.Kernel(threads=)` 已移除。备选 L2' 的 `T.load_nd2nz`/`T.alloc_L1`/`T.alloc_L0C`/`T.store_fixpipe`/`sync_block_set` 均有 `docs/` 与 `examples/` 佐证。

### 2. 内存层级规划 — warn

- 搬运路径完整（GM→UB/L1→Cube→L0C→GM，无 GM→L0 越级，§4.4）；层级标注完整（§4.3 每个 buffer 标注 UB/L1/L0C）。
- **UB 预算 — 未溢出（余量偏紧）**：§4.5 逐 buffer 表 2-stage 合计 = 153,600 B = 150.0 KiB（本检视独立复算，与文档一致），×1.10–1.12 → **≈165–168 KiB < 192 KiB** → 满足「总和 ≤ 容量」判定。**warn 理由（残留风险，与上一轮同源）**：① 余量仅 ≈12–14%；② ×1.10–1.12 系数绑定 `CONST-capacity-910B2C` 的显式分配口径，而该条目同时记录 auto-multi-buffer 膨胀 ~1.7x——若反量化链缓冲实际走 auto-multi-buffer，则 153.6×1.7 ≈ 261 kB > 192 KiB 会溢出（设计已自述「`T.Pipelined` 是否对全部链缓冲做 auto-multi-buffer 属未文档化」，并给出缓冲复用 / 降 stage / `deq` 分块三条缓解，交 Stage 3 按 msprof 对账）。上一轮的 UB 口径不一致（153.6 kB vs 59 KB）与十进制 KB 混读**本轮已修复**（§1.6.0/§1.6.3 统一引用 §4.5，单位声明 `1 KiB = 1024 B`）。
- **L1 预算**：`(16·64 + 64·64)·2·2 = 20,480 B = 20.0 KiB < 512 KiB` ✓（独立复算一致）。**L0C**：`16·64·4 = 4,096 B = 4.0 KiB < 128 KiB` ✓（机械 `l0c_budget` pass）。

### 3. Tiling 策略 — pass

- Block 划分明确（`block_M=16, block_N=64, block_K=64, num_stages=2`，含理由）✓；尾轴对齐满足（`block_K/2=32` 元素 = 64 B、`block_K=64` 元素 = 128 B、`block_M/block_N ≥16` 分形）✓；非整除 M/N 用动态 extent + 写回守卫（禁 host padding）✓。
- **分核策略三要素齐全 — pass**：① 逻辑核数 `ceil(M/block_M)×ceil(N/block_N)`（§5.5 含 6 workload 逐项表，decode=128，本检视独立复算一致）；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查 = 20**（含查询代码与返回值；混合算子直接用返回值、不翻倍；并注明与 `CONST-aicore-910B2C` 记 24 核的差异及「换设备须重查」）；③ 规模判定与方案（小/中/大三档 + `block_N=208` 对齐到 40=2×20）✓；核内串行边界为编译期常量 ✓。
- **`num_launched` 一致 — pass**：§0.6 R3（`num_launched = num_logical`）、§3.3（`num_launched = num_logical`）、§5.5（`num_logical > 20：num_launched = num_logical（非 persistent）`）三处一致，非 persistent 且核内无 `T.serial` 多任务循环，规避 `CG-2026-0010`/`PL-1.16` 结构边界。
- **`deq_N` 分块自洽 — pass（第 2 轮建议已修复）**：§5.2/§5.5/§6.1 现统一写入**不变式 `deq_N | block_N` 且 `deq_N % 16 == 0`**；`block_N=208` 配套 `deq_N=16`（`208 // 16 = 13` 段，全覆盖），`block_N=176` 配套 `deq_N=16`（`11` 段，全覆盖）；明确 `W_sh` 是 `T.gemm` 的 src2、**必须整 `block_N`、不随 deq 分块缩小**；本检视独立复算 UB 峰值 = `1024+12288+4096+16384+53248+512 = 87,552 B = 85.5 KiB`，×1.10–1.12 → **≈94–96 KiB < 192 KiB**（含整块 `W_sh` 53,248 B），与 §5.5 末重算表一致；v1 的「UB 峰值降到 ≈60 KB」（`deq_N=32` 仅覆盖 192 行且未计 `W_sh`）已作废并注明原因。

### 4. 技术约束检测 — pass

未涉及三维 Kernel（一维 grid + cid）；GEMM 非整除已在 §0.6 R5/§5.4 处理（动态 extent，禁 host padding，符合 `references/ascend-constraints.md` §4）；L0C 不溢出（4.0 KiB）；fp16 累加禁忌已遵守（强制 fp32，`T.gemm.md` §2.3）；`T.gemm` 不支持 int32×int32 已规避（不走 int8 路径）。§3.5.1 已知限制检查表逐项覆盖。

### 5. 循环与同步 — pass

循环结构明确（§6.1：block 级并行 `T.Kernel`、tile 级 `T.Pipelined`、可选静态 `T.serial` deq 分块；tile 内用 `T.copy` / v-prefix 链 / `T.npuir_interleave` / `T.gemm`，**无逐元素标量循环**）；同步与模式匹配（§7 自动同步，Developer；CV 交接无手写 barrier / flag）；尾块处理明确（§6.4）。`T.Pipelined` 迭代数 `K/64 ∈ {4,…,1280}`，流水充分。deq 分块边界 `block_N // deq_N` 现与 `block_N` 整除（见维度 3），为编译期常量。

### 6. 验证方案 — pass

§8.1 含 PyTorch golden 草案（`quantize_weight_int4` + `golden_gemm_w4a16`，移植源仓参考实现，本检视核对语义一致）；§8.3 含 7 档 L0 门槛测试计划（smoke / m=1 decode / 非 2 幂 / 组内多 tile / 长 K），dtype 与 shape 明确；精度标准按 dtype 给出（fp16 atol=7e-2 / rtol=5e-2，来源 `test_gemm.py` 的 `test_gemm_w4a16` 实测一致）。

### 7. 完整性与一致性 — pass

- 章节齐全：§0.1–§0.7 + §1–§10 模板章节齐全，全文无未填充占位符（无花括号占位、无待办标记）。
- 同类引用真实：§3.5.3 引用的 12 个 `examples/`/`testing/` 路径经逐一确认**全部存在**（`test_mix_kernel_bf16.py`、`example_gemm.py`、`example_mixcv.py`、`mixcv_mixkernel.py`、`matmul_dynamic_shape.py`、`matmul.py`、`test_slice_interleave_dev.py`、`test_gemm.py`、`workloads/gemm.py`、`_gemm_w4_a16_kernels.py`、`examples/gemm/_gemm_kernel/DESIGN.md`）。
- 内部一致性 — pass（上一轮 3 处不一致本轮已修复）：① UB 数值统一为 §4.5 单一事实源（150.0 KiB / 165–168 KiB）；② `deq_N | block_N` 不变式 + UB 重算；③ §4.5 单位口径注明（`1 KiB = 1024 B`）。API 映射、内存规划、Tiling 三处描述相互一致。（§1.6.0 R3 的 FLOPs 量纲笔误见维度 8 的 warn。）

### 8. 算法优化分析 — warn

- **§1.6 章节存在性 — pass**：§1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节齐全。
- **算法调研存在性与完整性 — pass（第 2 轮阻塞已修复）**：R1 候选表含**基线候选**（候选 0 = 源算法），R1 现**逐候选建行**——`algorithm-candidates.md` 的 `ALG-gemv` 条目 R1 命中行的两个候选「**GEMM 分形路径（Cube，load_nd2nz）**」（= 基线 0）与「**Vector 规约累加**」（= 候选 9，m=1 gemv 形态）**均已进入 R1/R3/R4 量化对比**（本检视直接 Read 该条目 L80–86 核对「R1：GEMM 分形路径（Cube，load_nd2nz）vs Vector 规约累加」）。R2 有明确结论（「基线本身已是单遍在线形态」，依据「GEMM K 累加 = running accumulator + 逐元素映射单遍天然在线」，与 `algorithm-research.md` §3 R2 判据一致，且为**正向结构结论**而非空泛负向断言）。R3 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表，口径注明，机械 `r3_metrics` pass），候选 1/4 均计入 GM 往返。R4 逐候选六维评估，候选 9 补入「计算单元匹配 / 容量 / 对齐 / 静态边界 / 流水融合」六维，并给出量化淘汰依据（无访存收益 + Vector 吞吐劣势 + 仅 m=1）。「调研范围」段现明列命中行两候选均已评估。
- **调研结论正确性（独立复核） — warn**：
  - **等价性**：候选 9 与 R1 其它候选的等价性推演成立（候选 9 为同一 dot product、K 顺序累加，仅执行单元不同；E1–E4 见下）。**cast-only recipe 独立推演成立**：本检视按 `256 ≡ 0 (mod 16)` 对边界字节手算——byte 172（i8=−84）→ `h=floor(−84/16)=−6`、`lo=−84−16·(−6)=12`（≡172 mod 16 ✓）、`hi=−6−16·floor(−6/16)=−6+16=10`（=⌊172/16⌋ ✓）；byte 255（i8=−1）→ `lo=15`、`hi=15`；byte 127 → `lo=15`、`hi=7` —— 与无符号语义逐位一致，符号无关性成立。
  - **复杂度算术独立复算**：访存 `M·K·2 + N·K/2 + N·(K/128)·5 + M·N·2 = 361,938,944 B ≈ 361.9 MB` ✓（逐项复算一致）；候选 1 增量 `2·N·K·2 = 2.68 GB → 3.04 GB`（8.4×）✓；MAC 数 `N·K = 6.71e8` ✓；扫描遍数 1（W 打包读 1 遍）✓；中间缓冲峰值口径含 GM 往返（候选 1/4）✓，**未漏计**。**warn（量纲笔误）**：R3 基线与候选 9 的 **FLOPs 列数值大 1000×**——文档写「真值 `2·M·N·K = 1.34e12`、Cube 实算 `2·block_M·N·K = 2.15e13`、`block_M=64` 时 `8.6e13`」，本检视按 M=1/N=8192/K=81920 复算应为 `1.34e9 / 2.15e10 / 8.6e10`（manifest roofline `flops: "2 * m * n * k"` 佐证；且 `2×MAC = 2×6.71e8 = 1.34e9`，与文档自身同列的 MAC 数 6.7e8 自相矛盾）。此系数是**一致的量纲笔误**（三处同偏 10³），**不翻转任何结论**：D-2 的 Cube 延迟估计（300 TFLOPS → ≈72 µs、200 TFLOPS → ≈107 µs、`block_M=64` → ≈287–430 µs）恰由**正确量级**（2.15e10 / 8.6e10）算出，故设计「流量项 287 µs 主导、Cube 填充不翻转为主导项」的判据成立。
  - **负向断言复核**：R2「无独立在线变体」为**正向结构结论**（GEMM K 累加天然 running accumulator），对照 `algorithm-research.md` §3 R2 判据成立；「本仓无 W4A16/反量化 GEMM 专条」经 Read `algorithm-candidates.md` 属实（仅 `ALG-gemv` 命中）；未引用互联网候选，无需来源记录。
  - **亲和性淘汰依据**：与 `constants.md` / pattern-library 条目（`CONST-capacity-910B2C`、`CONST-mte2-degradation`、`CONST-store-fixpipe-gm-only`、`CONST-vector-launch-overhead`、`PL-1.13`/`PL-1.15`/`PL-1.16` 等）一致，被引条目真实存在。
  - **并列候选裁决**：候选 9 的淘汰裕度非「小裕度纸面拍板」——受同一 GM 流量下界约束（带宽受限，W 占 92.7%），候选 9 不降低主导项，属结构性淘汰；其余候选（仿射分解/split-K）亦各有明确裁决依据。
- **调研→设计一致性 — pass**：§1.6.0 选定「分块融合反量化 A16 Cube 直接法 + E1–E4」实际驱动 §1.4 算法描述、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构（顺序 k-tile 单遍）；迁移任务 §0.5/§0.6 处置与选族一致（不替换算法族，仅硬件组织重设计），源优化手段 W1/W2/W8 的意图承接逐项说明（W2→R1、W8→E3、W7→R2/R4），无静默丢弃。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法采纳 E1–E4，每项四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」齐全（收益量化到次数 / 字节，如 E3 元数据 cast `6.7e8 → 2.6e5` 次）。
- **等价性论证正确性（独立推演） — pass**：E1（fp32 累加器单次 RNE 到 fp16，重复舍入幂等：`round(round(x))=round(x)`）成立；E2（K 尾 A/W 双清 → 新增项 `0·0=0`，杜绝 `0·NaN`/`0·inf`）成立；E3（同值 cast 一次再广播 ≡ 逐元素取同一常量，同一元素同一算子）成立；E4（`q,z∈[0,15]` 在 fp16 精确，`s` 单次舍入 ≲1 ulp，实测 max abs 3.906e-3 ≪ atol 7e-2，违反率 0）成立。**verify_equiv.py 重跑 13/13 `EQUIV_PASS`，与内嵌表逐行一致**。
- **向量化替代完整性 — pass**：§1.6.2 覆盖全部计算点（#1–#11），与 §3.3 伪代码、§6 循环结构交叉核对无遗漏；矩阵乘归 Cube `T.gemm`（#1）、装载/写回归 `T.copy`（#2/#7）、解包/反量化归 v-prefix 链（#3–#6/#8）；保留 3 处标量 / 循环（#9 block 索引、#10 host/kernel 元数据、#11 tile 级顺序依赖），理由具体到本算子。§6 无逐元素标量循环，双向一致。
- **不可替代理由充分性 — pass**：#9/#10/#11 分别为 block 级任务映射、host/编译期元数据、`C_local` 跨 K-tile 携带依赖（归约需先物化全 K，破坏 tiling 与 UB 预算），均非逐元素热点，理由充分。
- **替代 API 可行性 — pass**：cast-only recipe 由探针 P2 机器验证 bit-exact（本检视独立推演亦成立）；`npuir_cast/mul/sub/interleave/vbrc` 均有源码导出佐证；位运算族（`vand`/`vshr`/`floordiv`）的**负面论断**（阻断/超时）附 CG-2026-0018/0019/0025 探针实测，未凭记忆。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按 Cube/MixCV 类别枚举 L1–L5（Developer 统一缓冲 / Expert 双 Scope / 核内 transpose / host 预解包 / epilogue 归属 Vector），逐候选量化评分（整除 / UB / repack 代价 / 向量轴）；弃选有量化理由（L2 GM 往返 8.4×、L3 `T.transpose` 核内二轴交换 + dtype 矩阵限制（已核对 `docs/Tilelang.language/创建操作/T.transpose.md`）、L4 `ascend-constraints.md` §4 禁止、L5 无收益 + GM 往返）；未把 I/O 契约当内部布局依据。
- **实验裁决模式完整性 — pass**：判定裕度依赖未实证常数（UB→L1 交接开销、Vector 链墙钟、Cube 吞吐）→ 三件套齐全：主选 L1；备选 L2'（结构达 Stage 4 可直接实现深度：buffer 形状 / UB 预算 / GM workspace 尺寸 / flag 配对 / sync 均已给、按新粒度重算）；裁决计划含代表 shape、指标（msprof latency / `vec_ratio` / `mte2_active_bw`）、判定阈值（备选 < 主选 ×0.95 才翻转，主选 UB 溢出/编译失败则无条件切）与回写路径，并评估了按 shape 分派主选/备选。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环与 §1.6.2 结论一致；§1.4 / §3.1 / §6 与 §1.6.0 选定算法（含候选 9 淘汰）同口径。

## 检视结论汇总

- **第 2 轮阻塞项修复确认**：§1.6.0 R1 已补 `ALG-gemv` 命中行必查候选「Vector 规约累加（m=1 gemv 形态）」（候选 9），并在 R3 补四口径、R4 补六维量化淘汰，同步更新「调研范围 / 关键依据 / §9.1.2-5」——**已修复**。
- **4 条建议修复确认**：① UB 口径统一为 KiB（§1.6.0/§1.6.3 引用 §4.5，150.0 KiB / 165–168 KiB）——已修复；② `deq_N | block_N` 不变式 + UB 重算（≈94–96 KiB，含整 `W_sh`，旧 60 KB 表述作废）——已修复；③ §4.5 单位注（`1 KiB = 1024 B`）——已修复；④ §0.3 chunk 循环更正为 4 次——已修复。
- **残留（非阻塞）**：§1.6.0 R3/D-2 的 FLOPs 量纲笔误（1.34e12/2.15e13/8.6e13 应为 1.34e9/2.15e10/8.6e10），记入维度 8 的 warn；不影响算法选择结论与 Stage 3 正确性，建议后续修订将三处指数改正为 10⁹/10¹⁰（或统一标注 GFLOP/TFLOP 量纲），避免 Stage 4 对账时误读。
