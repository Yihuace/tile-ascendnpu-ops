# _make_rope_neox_1d 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/rope_neox/_make_rope_neox_1d/DESIGN.md`
- 任务类型: 迁移（migration-harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/rope.py`（`_make_rope_neox_1d`，L47–L89）
- 抽取副本: `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/_rope_neox_kernels.py`
- 集成 wrapper: `examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py`
- 规格: `examples/TileOPs/tileops/manifest/position_encoding.yaml`（`RopeNeoxOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。`python3 examples/rope_neox/_make_rope_neox_1d/verify_equiv.py` 实跑输出 `TOTAL items = 30 ; EQUIV_FAIL = 0` / `OVERALL: ALL ITEMS EQUIV_PASS`，与 §1.6.1 内嵌结果表**逐行一致**（30/30 `EQUIV_PASS`；fp64 参照 `max_abs=0.0/ulp=0`；被拒 E1 诊断 fp16 `7.8125e-03`、bf16 `6.2500e-02` 与 §8.2 引用一致）。脚本存在、可执行、与内嵌表一致 → 等价性机器验证有效。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/rope_neox/_make_rope_neox_1d/DESIGN.md`）:
  - `ub_budget`: **pass** — 11 buffer 求和 = 36.0 KB；容量 192.0 KB；×1.7 auto-multi-buffer 膨胀预算 113 KB，within budget（与 §4.5 逐 buffer 表一致）。
  - `l0c_budget`: **pass** — 解析出 block 256×64 fp32 accum = 64.0 KB vs L0C 128 KB（本算子为纯 Vector，无 Cube/L0C 计算，此项不适用；机械通过不构成额外约束）。
  - `core_split`: **skip** — `insufficient parsed dims {'M': 256}`（工具无法从 §5.5 表格机械解析核数）。**机械跳过，人工已核**：§5.5 三要素齐全，逻辑核数与物理核取值自洽（详见维度 3）。
  - `r3_metrics`: **pass** — R3 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）均存在。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子 `/home/developer/TileOPs/tileops/kernels/rope.py`（L47–L89）全文，与 DESIGN §0 逐项核对。

- **语义理解正确性（pass）**：源码 L74–85 为 `freq_idx = col % half`；`c=cos_table[row,freq_idx]`、`s=sin_table[row,freq_idx]`；`val=x[row,col]`；`rotated = col<half ? -x[row,col+half] : x[row,col-half]`；`y[row,col]=val*c+rotated*s`。DESIGN §0.1 的定义式、`rot(x)` 分段式、`c/s` 来源逐项一致；§0.1「等价闭式（半段 split）」独立推演成立（`col=j` → `a_j c_j + (−b_j) s_j`；`col=half+j` → `b_j c_j + a_j s_j`）。规约语义「无规约」与源码一致（无累加轴）。dtype 语义「cos/sin 由 Op 层 fp32 计算后 `.to(dtype)`」经 `tileops/ops/rope.py::_base_freqs`（L99–L118）核对属实。边界语义（无分支特例、`x=0`→`y=rot*s`）与源码一致。
- **算法解读完整性（pass）**：§0.3 步骤表 1–5 覆盖源码全部计算语句（`flat_idx` / `row,col` / `freq_idx,c,s` / `val,rotated` / `y` 写回），**无遗漏、无臆造**；数据流图「纯寄存器、无显式 SMEM/fragment」与源码（无 `T.alloc_*`）一致；host 侧逻辑（`N_total`/`half`/`block_size` 编译期计算 + `out_idx=[3]` + Op 层 cos/sin 生成）已列出。
- **优化手段覆盖完整性（pass）**：§0.4 识别 7 项（`T.Parallel(threads,npt)` / flat `//,%` / `if_then_else` 配对 / `out_idx=[3]` / `lru_cache` / 隐式 cos-sin 复制 / `threads=256,npt` 对齐），与源码逐条对应，无遗漏、无误判层级。
- **耦合性判定合理性（pass）**：§0.5 每条有四态处置 + 依据，与 `TRAP-threads-kwarg-noop`（NPU 不消费 `threads=`）、`TRAP-C10-strided-gather`（跨步 gather 缺失，pattern-library/INDEX.md L76 `verified`）不矛盾；舍弃项（layout=2d / non_neox / llama31 / yarn / longrope）写明范围外理由，与 manifest 单 `RopeNeoxOp` 契约一致。
- **NPU 重设计可行性（pass）**：R1（一维核 + 核内网格跨步 + 2D UB tile）、R2（半段 2D 向量链 + fp32 中转）、R3（cos/sin 单次装载复用）均含「源方案 → NPU 新算法 → 语义保持论证」三要素；R2 的舍入路径论证（`(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)`）经独立推演成立，且与 `verify_equiv.py` #1/#2 三 dtype `max_abs=0/ulp=0` 一致。
- **设计与决策一致性（pass）**：§1.4 / §3.1 / §3.3 / §4 / §6 描述的是 §0.5/§0.6 决策后的 NPU 算法（半段向量链 + fp32 中转 + cos/sin 复用）；无「§0.5 判重设计、§3 照抄源 gather」的脱节。
- **golden 独立性（pass）**：§8.1 golden 依 §0.1 语义直接实现（`torch.cat([-x2,x1],-1)` + `x.float()*cos_full.float()`…），经 `tests/ops/test_rope.py::ref_rope_neox`（L62–L68，`_rotate_half_neox` = `cat([-x2,x1])` L46–L51）核对为源仓参考实现，**未复刻 §0.6 的半段向量链结构** → 验证独立。

### 1. API 可行性 — pass

§3.2 列出的 API 均在本仓有佐证：`T.vmul`/`T.vadd`/`T.vsub`（`docs/Tilelang.language/数学操作/*.md` §2.1/§2.3/§2.4，实跑先例 `testing/npuir/bf16_support_ops/test_vv_bf16.py`）、`T.vcast(src,dst,round_mode)`（`docs/.../数据类型转换操作/T.vcast.md` §2.1 参数表 + §2.2.1 dtype 矩阵）、`T.copy`（`docs/.../内存操作/T.copy.md` §2.4 示例 1/2 含 `T.min` 尾块）、`T.alloc_ub` 2D（`T.vmul.md` §2.4）、`T.Kernel(n, is_npu=True) as (cid,_)` + `T.serial`（`examples/elementwise/vec_add_2d.py` 实跑）、`T.min`（T.copy.md §2.4）。v-prefix 优先（全链 `vmul/vadd/vsub/vcast`），未使用 `npuir_xxx`。Developer 模式 + 自动同步，无 Expert 手动 set/wait。**verdict pass**。

### 2. 内存层级规划 — pass

§4.4 给出完整 GM→UB 路径（无 GM→L0 越级）；每个 buffer 标注层级（§4.3 全部 UB）；§4.5 UB 逐 buffer 预算最大情形 36 KB（fp16/bf16 d128，E=1024：4 dtype×2KB + 7 fp32×4KB），×1.7 膨胀 ≈ 61 KB < 192 KB。机械 `ub_budget` pass（36.0 KB）。无 Cube，L0C 不适用。**verdict pass**。

### 3. Tiling 策略 — pass

- **Block 划分（§5.2）**：`block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048），`block_rows = min(block_size//head_dim, seq_len)`，表格逐 dtype/head_dim 给出具体值（d64 fp16/bf16→32；d128 fp16/bf16→16；fp32 d64→16、d128→8），理由充分（对齐、UB 容量、发射项权衡）。
- **对齐约束（§5.3）**：`half` 段 fp16 128B/64B、fp32 128B，均 32B 整倍；向量宽整除 `half∈{32,64}`，尾 lane 浪费 0%。
- **非整除处理（§5.4）**：`tr = T.min(block_rows, seq_len - r0)` 动态行尾裁剪 + `block_rows=min(...,seq_len)` 防 `block_rows>seq_len`（`TRAP-ub-blockm-gt-M-overflow`）。
- **分核策略三要素（§5.5，对照 `core-split-strategy.md` §1/§2.2）**：① 逻辑核数 `ceil(seq_len/block_rows)` 逐 workload 列出（16/8/64/256）；② 物理核数**实查**——记录 `NPUUtils.get().get_aicore_num()`→20、`get_aivector_core_num()`→40（纯 Vector 翻倍），代码路径经 `tilelang/utils/npu_utils.py` L112–119 核对存在（`get_aivector_core_num = get_aicore_num()*2`），且与 `CONST-aicore-910_9362`（20 AIC / 40 AIV）一致；③ 规模判定四档 + 统一网格跨步 persistent 方案（`core_num=40`，循环上界 `ceildiv(n_num,40)` 静态），并**显式记录残留负载不均**（[2048,64] +25.0%、[4096,128] +9.4%；smoke 核占用 20–40% 空转）。核数与 block 取值自洽。机械 `core_split` skip → **人工已核**：三要素齐全、核数自洽、核内串行边界静态。
- **minor（建议级，见维度 7）**：§5.5 对 `[2048,64]` 标注「中等规模」却采用「极大规模」处置，术语标签略不精确，但方案与依据完整。

**verdict pass**。

### 4. 技术约束检测 — pass

不涉及三维 Kernel（一维 `T.Kernel(40, is_npu=True)`）；非 GEMM（无需 padding+crop，尾块由 §5.4 处理）；无 Cube → 无 L0C 溢出风险。§3.5.1 逐条核对本项目已知限制（`threads=` noop、条件 `T.alloc` 非法、v-op 动态 slice、`T.vbrc` 宽 1 广播不触发、无 fused multiply-add），处置与 `TRAP-*` 条目一致。**verdict pass**。

### 5. 循环与同步 — pass

§6.1 明确循环类型（核级 `T.Kernel` / tile 级 `T.serial` 网格跨步 / 元素级向量化整 buffer），无逐元素标量循环；§6.3 明确不用 `T.Pipelined` 的理由（单 tile 线性依赖、跨 tile 由 auto-multi-buffer 处理）；§6.4 尾块处理与 §5.4 一致；§7 同步策略与 Developer 模式匹配（自动同步，无手动 set/wait；无 CV 融合，§7.4 n/a）。**verdict pass**。

### 6. 验证方案 — pass

§8.1 给出 golden 草案（独立于 NPU 结构）；§8.2 精度标准按 dtype 给出（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与源仓 `_get_tolerances` L287–L293 一致）；§8.3 L0 计划含 manifest 全 workload、各 dtype、角点值（±inf/NaN/±0/次正规）、非整除尾块（2050）、极小 `seq_len`（1）、非 2 幂（1000）与 cos/sin 复用等价用例，判定标准明确。**verdict pass**。

### 7. 完整性与一致性 — warn

- 章节齐全（§0 含 0.1–0.7 全部小节 + §1–§10）；未检出任何占位符标记（已逐类检索确认）；§3.5.3 引用的 examples/ 路径全部真实存在（`examples/elementwise/vec_add_2d.py`、`examples/bitwise_not/bitwise_not_fwd_kernel/`、`examples/hardtanh/_make_hardtanh_kernel/`、`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/dropout/_make_dropout_kernel/`、`testing/npuir/bf16_support_ops/test_vv_bf16.py` 均核对存在）。
- **建议级问题 1（内部一致性）**：§3.3 工厂 `_make_rope_neox_1d(seq_len, head_dim, dtype, block_size=2048)` 在工厂体（`def kernel` 之外）由 **工厂参数 `block_size`** 计算 `block_rows = max(1, min(block_size//head_dim, seq_len))`，而内层 `kernel(blk)` 的 `blk` 在伪代码中**未被使用**。集成 wrapper 的 `_build_kernel` 以 3 个位置参数调用工厂（`block_size` 取默认 2048），`forward` 才把 `self.config["block_size"]`（`_DEFAULT_BLOCK_SIZE`：fp32=**1024**/fp16·bf16=2048）传给编译后 callable。→ **fp32 路径下 §5.2/§4.5/§5.5 声明的 `block_size=1024`→`block_rows=16` 不会被实际采用**（实际取 2048→32），wrapper 的 fp32 `block_size` 被静默忽略；§3.3 末尾「本设计完全对齐 wrapper 契约」的断言对 fp32 不成立。此为文档内部一致性问题，**不影响数值正确性**（两种 tiling 均有效、UB 44 KB 亦不溢出），故列为建议级。
- 其余内部一致（API 映射 / 内存规划 / Tiling 相互自洽）。

**verdict warn**（建议级，不阻塞）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性（pass）**：含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策四个子节。
- **算法调研存在性与完整性（pass）**：R1 候选表**含基线**（候选 #0 源算法）且列 #1–#5 及等价性初判/收益方向；R2 在线算法给出**明确结论「无」**并附结构依据（无 running 统计量、无跨元素依赖、源即单遍直写）；R3 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表、口径写明；R4 逐候选对照六项亲和清单；调研结论 + 调研范围写明。深度与复杂度匹配（参考 `ALG-elementwise` 并补足结构重排候选）。
- **调研结论正确性（独立复核，pass）**：
  - **等价性独立推演**：半段结构与源公式逐项等价（见维度 0）；`(−b)*s ≡ −(b*s)`、`t1−t2 ≡ t1+(−t2)` IEEE 精确成立 → 候选式与基线逐位一致（`verify_equiv` #1/#2 三 dtype `0.0 ulp`）。
  - **复杂度独立复算**：[4096,128] bf16 `x_elems=524288`、`cos_sin_elems=524288` → `(2·524288+524288)×2B = 3.0 MB` ✓；FLOPs 3/元素（4 mul+2 add per 2 输出）✓；#3 Gauss 3 乘 = 3 mul+5 add = 8 op/2 输出 = 4/元素 ✓；扫描 1 遍、缓冲 O(tile) ✓。**未发现中间缓冲 GM 往返漏计**（所有候选均无中间 GM 往返）。
  - **负向断言复核**：① 基线 gather 淘汰有 `TRAP-C10-strided-gather`（pattern-library/INDEX.md L76 `verified`，traps-compiler.md L41–43「跨步 gather 向量指令缺失」）佐证；② `T.npuir_gather` 存在但为显式索引搬移（`tilelang/language/customize_npuir.py` L1215–1235 核对属实）；③ `TRAP-expert-v-operands`（traps-compiler.md L160+「v-op 切片仅接受静态边界，运行时 Var 触发 parser `int(Var)` 拒绝」）支撑 L3 切片依赖判定与 v-op 整 buffer 纪律；④ `TRAP-conditional-alloc` / `TRAP-ub-blockm-gt-M-overflow` / `TRAP-vbrc-empty-broadcast` / `TRAP-tvm-parser-rules` / `TRAP-threads-kwarg-noop` 均实存。对照 `ALG-elementwise` 命中行（R1 CSE/乘倒数/cast 链合并、R2 天然单遍、R3 融合消 GM 往返、R4 单遍 Vector/dtype 路径），**未发现被漏掉的已知候选**（本算子无 RoPE 专门候选；「无在线变体」结论有结构依据成立；无除法/超越函数，candidate #4/#5 明确不适用）。
  - **亲和性淘汰依据**：容量/带宽淘汰引用 `constants.md`（`CONST-capacity-910B2C`、`CONST-vector-op-cost-910_9362`、`CONST-mte2-degradation`），且**逐条标注 stale/设备绑定 + 9362 适用性**（R-12），未与 pattern-library 实测条目矛盾。
  - **并列候选裁决**：L1/L3 判定裕度依赖未实证常数 → 交 §1.6.3 实验裁决（非纸面拍板），符合要求。
- **调研→设计一致性（pass）**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构与 §1.6.0 选定算法族（半段旋转 + fp32 中转 + cos/sin 复用）一致；§0.5/§0.6 处置与调研选族一致；源优化意图（§0.4 #1–#7）逐项承接说明（§0.6 R3 末段 + 1.6.0）无静默丢弃。
- **数学等价优化完整性（pass）**：§1.6.1 采纳 3 项，每项四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化）；不采纳 E1/E2/E3 各有理由；「公式层已无进一步计算量空间」有依据（每输出元素已是最少 3 算术 op；乘倒数/cast 链合并不适用）。
- **等价性论证独立推演（pass，含一项证据质量提示）**：#1/#2 推演成立且与机器验证一致；#3（cos/sin CSE）在数学上**平凡等价**（两半在源中本就映射同一 `freq_idx`，取同值，复用不改变数值）——独立推演成立。**提示（建议级问题 2）**：`verify_equiv.py` 的 `candidate_no_cse`（L69–70）直接 `return candidate(...)`，与 `candidate` 完全相同，故 #3 行是**自比较（恒 0）**，并未真正复核 CSE；建议改为「逐元素重取 cos/sin」的真实对照实现或删除该行。因该优化项经独立推演平凡成立，不影响结论。
- **向量化替代覆盖完整性（pass）**：§1.6.2 表 8 个计算点与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**——逐元素计算点 #1–#5 全部向量化（`vmul/vadd/vsub/vcast`）；保留 3 处（#6 核内调度循环 / #7 block 索引 / #8 动态行尾边界）类别合法（tile 级调度 + block 索引 + 依赖编译期 shape 的边界），§6 确认 kernel 内无逐元素标量循环。
- **不可替代理由充分性（pass）**：3 处保留逐项给出具体理由且指向先例（`vec_add_2d.py` / `PL-1.22`），非泛泛而谈。
- **替代 API 可行性（pass）**：全部替代 API 有 docs/examples 佐证，优先 v-prefix（与维度 1 同口径）。
- **向量化轴与布局决策完整性（pass）**：**2 个非 batch 维**（`seq_len`、`head_dim`）→ 给出 L1–L4 候选矩阵 + 逐候选量化评分（整除性/尾 lane/repack/UB）；L1 主选（原生行主布局 + `half` 内层向量轴，尾 lane 0%、无 repack）；L2（flat+gather）以 `TRAP-C10` 弃选；L4（核内转置）以「GM 流量不变 + repack 开销」弃选（依据 `ALG-transpose` R3，未把 I/O 契约当内部布局依据）；并明确「GPU 源码的并行轴是输入而非结论」。
- **实验裁决模式完整性（pass）**：判定依赖两个未实证常数（64B vs 128B 段效率、静态 2D 子切片 v-op 可用性）→ §1.6.3 给出**主选 L1 + 备选 L3（结构达 Stage 4 可实现深度：buffer 形状/UB 预算/循环结构/dtype 路径/分核三要素齐全）+ 裁决计划（代表 shape、msprof 指标、未知常数反解、判定阈值 ≥5%、回写路径）+ 按 shape 分派可行性评估**。
- **布局决策与实现一致（pass）**：L1 决策与 §3.3 伪代码（`a_ub/b_ub/c_ub/s_ub` 均 `(block_rows, half)`）、§4.5 UB 预算、§6 循环结构三方一致；无 repack 路径与无重排布局自洽。
- **与下游章节一致（pass）**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。
- **bf16 文档/示例分歧处置（pass）**：`docs/.../T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.2.1 表 Ascend 列 bf16 **✗**（已亲自逐格核对），而 `testing/npuir/bf16_support_ops/test_vv_bf16.py` 用 bf16 `T.vmul/T.vadd`（Developer 2D）实存 → 分歧属实。DESIGN 以「文档为准」（AGENTS.md）+ **本算子不依赖原生 bf16 路径**（sub-fp32 一律 fp32 中转，`T.vcast` 的 `bf16→f32`/`f32→bf16` 经 §2.2.1 核对为 `rint` 支持）处置，**成立且保守**（不阻塞，登记 R-3 + Stage 3 探针核对）。

**verdict pass**（含 1 项证据质量建议）。

---

## 附：关键风险观察（非问题，供 Stage 3 关注）

- **计算域正确性关键**：源 GPU kernel 在**输入 dtype 域**计算，而契约 golden 是 **fp32 opmath + 单次舍回**——异域，必须 fp32 中转；`verify_equiv` 诊断证明输入 dtype 域 fp16 `max_abs=7.81e-3 > 1e-3`，故 §0.6 R2 的 fp32 中转是**正确性要求**而非性能偏好。Stage 3 不得省略 `vcast`。
- **tiling 参数传递**：见维度 7 建议级问题 1，Stage 3 实现时应使内层 callable 真正消费 wrapper 传入的 `block_size`（例如在 `kernel(blk)` 内据 `blk` 计算 `block_rows`），以兑现 §5.2/§4.5/§5.5 的 fp32 tiling 声明。
