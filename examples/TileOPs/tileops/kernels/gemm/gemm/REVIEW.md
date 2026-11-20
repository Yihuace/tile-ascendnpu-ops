# _gemm_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/gemm/_gemm_kernel/DESIGN.md（revision v1，已按首轮检视修订）
- 任务类型: 迁移（harness，project=`gemm`，Cube 稠密矩阵乘族，Developer 模式）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py`（`_gemm_kernel`/`GemmKernel`，行 301–493）+ `/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py`（`GemmOp`，行 20–164）
- 上一版（首轮检视对象）: `examples/gemm/_gemm_kernel/history_version/design_v0.md`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2（均在维度 5 详情内，不作阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/gemm/_gemm_kernel/verify_equiv.py` → exit 0，逐行：
  - `E1-fused-vs-materialised/torch.float16` max_ulp `0.0` viol_rate `0.000000` **EQUIV_PASS**
  - `E1-fused-vs-materialised/torch.bfloat16` max_ulp `0.0` viol_rate `0.000000` **EQUIV_PASS**
  - `E2-ktail-zerofill/torch.float16` max_ulp `0.0` viol_rate `0.000000` **EQUIV_PASS**
  - `E2-ktail-zerofill/torch.bfloat16` max_ulp `0.0` viol_rate `0.000000` **EQUIV_PASS**
  - 末行 `ALL EQUIV_PASS`；信息性探针 `fp64→fp16 直接 vs fp64→fp32→fp16 双重舍入` 失配 `1599926`。
  - **与 §1.6.1 内嵌结果表（行 302–305）逐行一致（4/4 EQUIV_PASS）**，行名一致（`E1-fused-vs-materialised`）。E1 两条对照分支为**不同计算路径**（融合 store cast vs 显式 fp32 GM 物化 `gm.copy_` + `clone` 回读 + 分离 cast，脚本 L87–91），已非首轮的恒等式，该建议已落实。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/gemm/_gemm_kernel/DESIGN.md` JSON 逐项：
  - `l1_budget`: **pass**（`L1: 1 buffer entries sum = 128.0 KB; capacity = 512.0 KB`）；
  - `l0c_budget`: **pass**（`block 128x128 x fp32 accum = 64.0 KB vs L0C 128 KB`）；
  - `core_split`: **skip**（`insufficient parsed dims {'N': 2112, 'bm': 128, 'bn': 128}; declared logical cores = 128`）——机械无法解析一维折叠网格 + trans 布局，**机械跳过，人工已核**（见维度 3：三要素齐全、物理核数实查、`num_launched = num_logical` 四处自洽，无首轮的 `min(...)` 矛盾）；
  - `r3_metrics`: **pass**（`all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子 `tileops/kernels/gemm.py`（L301–493）与 `tileops/ops/gemm.py`（L20–164），与 §0.1–§0.7 逐项核对，逐检查项结论与源码证据如下：

- **语义理解正确性 — pass**：§0.1「`C = op(A) @ op(B)`；`trans_a=True` 时 `A` 存储 `[K,M]`、`trans_b=True` 时 `B` 存储 `[N,K]`」与源码 `a_shape = (k, m) if trans_a else (m, k)`、`b_shape = (n, k) if trans_b else (k, n)`（kernels/gemm.py L346–347）一致；输出 `C:[M,N]`、`same_as(a)` 与 `c: T.Tensor((m, n), dtype)`（L372）一致；「累加必须 fp32（`accum_dtype="float"`）」命中源码 L345；「K 尾 TMA 零填充、M/N 尾 epilogue 逐元素 `if` 守卫」与源码 docstring（L319–323）及 epilogue（L480–483）一致。默认 `(False, True)`（NT）与 op 层 `GemmOp.__init__(trans_b=True)`（ops/gemm.py L49）及 manifest `params.trans_b.default: true`（`manifest/gemm.yaml` L27）一致（源 `GemmKernel` 类自身默认 `trans_b=False`，L547——§0.1 取 op 层默认，正确）。**GEMV fast path 精确条件**（本版新增）§0.1/§9.3 写 `m==1 and not trans_a and trans_b`（lhs_row）或 `n==1 and not trans_a and not trans_b`（rhs_col），与源码 ops/gemm.py L107–108 逐字符一致。
- **算法解读完整性 — pass**：§0.3 七步计算分解覆盖源码全部计算语句——grid（L374–375）、`alloc_shared`×2 + `alloc_fragment`（L378–380）、`make_swizzled_layout`（L382–385）、`alloc_barrier`（L391–392）、`alloc_var` 迭代计数（L396–397）、producer `T.tma_copy`（L427–446）、consumer `T.wgmma_gemm(clear_accum=(ki==0))`（L463–471）、epilogue `T.Parallel` 守卫写（L480–483）、`dec/inc_max_nreg`（L407/L452）、`warpgroup_fence_operand`（L473）、`trace.*`（L408/426/447/454/460/462/480/487）。数据流图覆盖 `a_smem`/`b_smem`/`c_local`；host 侧（`GemmKernel.__init__` L556、`GemmKernel.forward` L569–579、`GemmOp.forward` L129–154、`_gemm_wrapped_kernel` L496–525）已列出。无遗漏、无臆造。
- **优化手段覆盖完整性 — pass**：§0.4 十四项与源码显式优化逐条对应（SMEM tiling / TMA / `num_stages=3` 环形缓冲 / 手工 WS `tx<128` 分流 L404 / WGMMA / swizzle / mbarrier / `setmaxnreg` / `warpgroup_fence_operand` / epilogue 守卫 / TMA K 零填 / `lru_cache` L301 / `T.Parallel` epilogue / `trace.*`），逐项标注硬件依赖，未把实现层误判为算法层（K 尾零填充/#11 与 epilogue 守卫/#10 标为算法层，其余标为实现层）。
- **耦合性判定合理性 — pass**：§0.5 每项有四态处置（保留/等价替换/重新设计/舍弃）+ 依据，与 GPU→NPU 能力映射（`migration-analysis.md` §5.3）及 `ascend-constraints.md` 不矛盾。舍弃项（swizzle #6 / `setmaxnreg` #8 / `warpgroup_fence_operand` #9 / `trace.*` #14 + helper）均写明理由（NPU 无 SMEM bank、无大寄存器堆、无 WGMMA、无 NPU tracer）；「WGMMA→`T.gemm`」与 `T.gemm.md` §1/§2.1（`a_transpose`/`b_transpose`/`initC`）一致；「TMA→MTE2/`T.copy`」「环形缓冲→`T.Pipelined`」有 `docs/开发指南.md` §2/§3 与 `examples/gemm/example_gemm.py` 佐证；`supported_archs=[90]`（L537）、`trace.out_idx`（L350）等 GPU-only 项处置齐全。统计 3 保留 / 8 等价替换 / 1 重新设计 / 5 舍弃，无静默丢弃。
- **NPU 重设计可行性 — pass**：§0.6 R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（WS→单核 `T.Pipelined`）保持 K 累加顺序与 fp32 域，数学等价；R2（2D grid→一维 `T.Kernel` + cid 分解 + 分核）满足 NPU 一维网格约束，且 `num_launched = num_logical` 保 `cid→(by,bx)` 双射（详见维度 3）；R3（WGMMA transpose flags→`T.gemm` transpose 参数 + 布局相关 tile 形状）与源码 `a_tile/b_tile`（L365–366）及 `T.gemm.md` §2.1 `size=[a,b,c]` 语义一致；R4（TMA 零填 + epilogue 守卫→动态 extent 切片 + A/B 双侧 K 尾 `T.clear` + 写回守卫）语义保持论证成立；R5（trace→去除）不影响数值与 I/O。新算法满足一维 Kernel、L1/L0C 层级、分形对齐、无 host padding 等硬件约束。
- **设计与决策一致性 — pass**：§1.2/§1.4/§1.5/§3.1–§3.5/§4/§5/§6/§7 描述的是 §0.5/§0.6 决策后的 NPU 算法（单核 `T.Pipelined` + `T.gemm` + 动态 extent 尾块）；§0.5 判「舍弃 swizzle」后 §3–§7 无布局重排步骤；§0.5 判「重新设计 #4 WS」后 §6.3 用 `T.Pipelined` 承接流水意图。无「§0 判重设计、§3 照抄源方案」脱节。首轮的 `num_launched` 跨节矛盾已消除（§0.6 R2 L154 / §3.3 L419–420 / §5.5 L655–661 / §6.2 L680–681 四处一致）。
- **golden 独立性 — pass**：§8.1 `golden_gemm`（`a.T if trans_a`、`b.T if trans_b`、`torch.matmul`）以 §0.1 源语义为唯一依据，与源测试 `GemmTest.ref_program`（`tests/ops/test_gemm.py` L28–33）逐行等价，**未复刻** §0.6 的 NPU 数据流（无动态 extent / K 尾清零 / VCast 细节），验证独立性成立。§8.2 容差 fp16 `1e-3/1e-3`、bf16 `1.6e-2/1.6e-2`、GEMV 边界 `1e-2/1e-2` 与 `test_gemm.py` L113–115/L124 逐值一致。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均有佐证：`T.Kernel(...,is_npu=True)`（`examples/gemm/example_gemm.py` L16、`docs/开发指南.md` §2）、`T.alloc_shared`/`T.alloc_fragment`（example_gemm.py L23–25）、`T.gemm(initC=,a_transpose=,b_transpose=)`（`docs/Tilelang.language/线性代数操作/T.gemm.md` §1/§2.1/§2.4）、`T.copy` 切片 + 动态 extent（`docs/Tilelang.language/内存操作/T.copy.md` §2.1 规则 2/4、§2.2.2、§2.4 示例 2）、`T.clear`（`T.clear.md`）、`T.Pipelined(...,num_stages=)`（example_gemm.py L27）、`NPUUtils.get().get_aicore_num()`（本机实查返回 20，§5.5 含查询代码与返回值）。Developer 模式全程用自动同步 API，未误用手动 `set_flag/wait_flag`，与 §2.1 模式选型匹配。备选 #2 的 Expert API（`T.alloc_L1`/`T.alloc_L0C`/`T.load_nd2nz`/`T.store_fixpipe`）有 `examples/gemm/matmul.py` 佐证且仅出现在备选。v-prefix 优先原则对本纯 Cube 路径不适用（无逐元素 Vector 链）。

> 附 warn（非阻塞）：§0.5#1/§3.2/§4.4 的 Developer `T.alloc_shared` + `T.copy` 作 **GM→L1** 通路，与 `T.copy.md` §1 方向表第 14 行「Developer `GM->L1` ×（用 `T.load_nd2nz`）」存在文档口径冲突；但官方 Developer Cube 示例 `examples/gemm/example_gemm.py`（`docs/开发指南.md` §2 同款）正以该形态装载。设计已在 §9.1#2 显式登记为「未文档化假设」、给出判决依据（以官方示例为准）与 Stage 3 首验证/回退备选 #2 的路径，属有据假设而非无据断言，不判 fail。Stage 3 首个 smoke 须先证该通路可编译，否则立即切 §1.6.3 备选 #2。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM→L1（`A_shared`/`B_shared`）→L0C（`C_local`，经 `T.gemm`）→GM（`T.copy` VCast），无 GM→L0C 越级路径。
- Buffer 层级标注齐全（§4.3/§4.5）：`A_shared`/`B_shared`=L1、`C_local`=L0C；UB 仅承载自动 VCast 写回中转（无主数据驻留）。
- **L1 预算**：`3×(128·128)·2B×2 = 196608B = 192KB < 512KB`（§4.5），按 `CONST-capacity-910B2C` 的 auto-multi-buffer ~1.12× 膨胀 ≈ 220KB 仍余量充足；机械 `l1_budget` **pass**。
- **L0C 容量**：`128·128·4 = 65536B = 64KB ≤ 128KB`（§4.5/§5.3）；机械 `l0c_budget` **pass**。
- fp32 累加器落 L0C（非寄存器堆），符合 NPU 层级（§0.5#8）。

### 3. Tiling 策略 — pass

**首轮阻塞项复核（已修复）**：首轮 fail 为 `num_launched = min(num_logical, num_physical)` 与 §5.5 主选方案 / §0.6 语义双射论证自相矛盾，且伪代码无核内 persistent 循环 → `num_logical > 20` 时漏算输出块。本版：
- §0.6 R2（L154）显式写 **`num_launched = num_logical`** 并注明「禁止用 `min(...)` 截断，否则 `num_logical > 物理核数` 时漏写输出块」；
- §3.3 伪代码（L419–420）注释 + `num_launched = num_logical`；
- §5.5（L655–661）删除 `min(...)` 公式，新增「网格数不变量」段，显式写「非 persistent 下 `num_launched` 必须恒等于 `num_logical`，否则漏写输出」，并把 `min(...)` 仅作为反例解释；
- §6.2（L680–681）注释 + 伪代码同步。
四处**自洽**，与源码 `example_gemm.py` L16（`T.Kernel(ceildiv(N,block_N)*ceildiv(M,block_M), is_npu=True)` 直接以逻辑块数为网格）一致。阻塞项已真正修复。

- **Block 划分**（§5.2）：`block_M=block_N=128`、`block_K=128`、`num_stages=3`，取值与 L0C/L1 约束自洽。
- **对齐**（§5.3）：`block_K/block_N=128` 为 16 元素（32B）倍数，Cube 分形 `≥16` 满足。
- **非整除处理**（§5.4/§6.4/§0.6 R4）：`m∈{16,32,64,96}`、`n=2112` 非整除用动态 extent 切片 + A/B 双侧 K 尾清零 + 写回守卫；host 禁 padding（`references/ascend-constraints.md` §4）。
- **分核三要素**（§5.5，对照 `core-split-strategy.md` §1）：
  1. ① **逻辑核数计算**：`num_logical = ceil(M/block_M)×ceil(N/block_N)`，按全部 13 个 manifest workload 列算。**独立复算全部正确**（smoke 1；square-1k 8×8=64；decode-gate-up 1×17=17；decode-down 1×56=56；prefill-gate-up 32×17=544；prefill-down 32×56=1792；prefill-attn-proj 32×32=1024；k-dominant 32×56=1792；wide-n 32×192=6144；mid-m16/m32 1×32=32；mid-m64 1×56=56；mid-m96 1×17=17；对照 `manifest/gemm.yaml` L35–55 的 m/n 值均一致）。
  2. ② **物理核数依据**：`NPUUtils.get().get_aicore_num()` 实查，§5.5 记录查询代码与实际返回值 **20**（Ascend910B2C，2026-10-13），Cube 算子直接用返回值不翻倍，符合标准 §1 要素②。**本次复跑确认返回值仍为 20**。
  3. ③ **规模判定与分核方案**：三档（小规模 ≤20 无需适配 / 中等规模对齐 20 整数倍 / 极大规模），并因 `TRAP-DEVMODE-PERSIST-GEMM`（`traps-runtime.md`，verified）明确 Developer 主选**不用 persistent**、固定 `num_launched = num_logical`、超发由运行时串行调度；Expert persistent 列为 Stage 4 备选（届时才 `num_launched = num_physical` + 静态边界 `T.serial`）。
  网格数与 block 取值自洽（`block_M·block_N ≤ 32768` 满足 L0C 128KB）。
- **机械 `core_split` skip → 人工已核**：机械无法解析一维折叠网格，人工确认三要素齐全、物理核数实查、`num_launched = num_logical` 自洽。

> 附 warn（非阻塞）：主选对极大规模（wide-n 6144）采用超发而非标准 §1 要素③ 的「固定核数 + 核内 `T.serial`」，属对已实测 trap 的**有据偏离**，设计已给权衡（K 迭代 8~128 充分、L2 复用摊薄启动开销）与 Expert 备选 + Stage 4 A/B；符合标准对 verified trap 的例外路径，不阻塞。

### 4. 技术约束检测 — pass

- 三维 Kernel：否（源二维 grid，设计一维 `T.Kernel` + cid 分解，§0.6 R2），无需 block_metadata。
- GEMM 非整除：§5.4/§6.4/§0.6 R4 已给核内动态 extent 方案；§3.5.1 逐条登记（含 host 禁 padding）。
- L0C 溢出：§4.5/§5.3 已核（64KB/128KB）。
- host 禁改输入/禁 padding：全部逻辑在 kernel 内（§0.3/§3.5.1/§4.4「宿主边界」）。
- `TRAP-DEVMODE-PERSIST-GEMM` 已识别并规避（§2.3/§5.5/§9.1#1）。

### 5. 循环与同步 — pass

- 循环结构（§6.1/§6.2）：block 级一维 `T.Kernel` 并行 + tile 级 `T.Pipelined(ceildiv(K,block_K), num_stages=3)` + `T.gemm`/`T.copy`/`T.clear` 隐式向量化；§6.1 声明「无逐元素标量循环」与 §1.6.2 一致。
- 同步（§7）：Developer 自动同步（`T.Pipelined` 自动插入 MTE2↔Cube 多缓冲同步、`T.copy` 自动依赖），与 §2.1 模式匹配；无手动 `barrier`/`sync_block`。
- 尾块处理（§6.4）：M/N 动态 extent 写回守卫、K 尾显式清零、对齐填充规避，均有伪代码对应。
- **首轮建议 4（B trans 分支）已落实**：§3.3 伪代码（L448–456）与 §6.2（L695–702）均补出 `trans_a`/`trans_b` 双分支（默认 NT 走 `B[n0:n0+tn, k0:k0+rK] → B_shared[0:tn,0:rK]`），§3.3 note（L465）覆盖 `trans_a`/`trans_b` 两侧展开，与 §4.3 `B_shared(block_N,block_K)`（NT）一致。
- **首轮建议 5a（K 尾清零落伪代码）已落实**：§3.3（L443–447）与 §6.2（L691–693）给出显式 `if rK < block_K: T.clear(A_shared); T.clear(B_shared)`，与 §6.4/§0.6 R4/E2 声明一致。
- **首轮建议 5b（GEMV 条件）已落实**：§0.1（L39）补全精确条件并注明默认 NT 下 `n==1` 不走 GEMV；§9.3（L830）同步。
- **首轮建议 2（CV 措辞）已落实**：§5.1（L612）与 §8.4（L799）均改为「后端自动插入的 `VCastOp`（Vector 单元）……隐式 Cube→Vector 协同，无需用户手写双 Scope」；全文 grep 无「无 Cube↔Vector 协同」残留，`store_fixpipe`/`enable_nz2nd` 仅出现在 Expert 备选。

> 附 warn（非阻塞，Stage 3 注意项）：§3.3/§6.2 把 `if rK < block_K: T.clear(...)` 放在 `T.Pipelined` 循环体内。`pattern-library` 记 CG-2026-0001 为 Developer codegen 能力缺口，其触发类含「Pipelined 体内条件构造」（attention.md L192）。本 manifest 的 `gcd(K)=128` 使该分支恒不触发，但编译器仍会看到循环体内的运行时谓词。因 K 与 block_K 均为编译期常量，建议 Stage 3 将 K 尾清零提升为 **trace 期 Python 分支**（如 `if k % block_K != 0:` 在工厂层判定后选构建变体），或先以 smoke 验证该条件体可编译，以规避该 Developer 崩溃类。属实现期风险提示，不影响本设计正确性。

### 6. 验证方案 — pass

- §8.1 `golden_gemm` 为 PyTorch 参考实现草案，以 §0.1 源语义为唯一依据（见维度 0 golden 独立性）。
- §8.3 L0 门槛含 smoke fp16/bf16、NN 方形（trans_b=False）、非整除 N（2112）、小 M（16/96）档，覆盖 M/N 尾块路径与两 dtype。
- §8.2 精度标准与 `tests/ops/test_gemm.py` 逐值一致（fp16 1e-3/1e-3、bf16 1.6e-2/1.6e-2，L113–115；GEMV 边界 1e-2/1e-2，L124）。
- 累加域 fp32 + 单次舍入（§1.6.1 E1）的误差来源说明合理，与 §8.2 容差自洽。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（含 0.1–0.7 七小节）+ §1–§10，另含「修订说明 v1」章。
- 全文**无占位符标记**（已检索设计模板定义的全部占位符 token，均无命中）。
- §3.5.3 引用的 examples 路径**全部真实存在**（`examples/gemm/{example_gemm.py,matmul.py,matmul_dynamic_shape.py}`、`examples/TileOPs/tileops/kernels/gemm/gemm/{gemm.py,_gemm_kernels.py}`、`examples/TileOPs/tests/ops/test_gemm.py`）；§0.2/§0.7 引用的 `manifest/gemm.yaml`、`workloads/gemm.py` 亦存在。
- §1.6/§1.6.3 引用的 `CONST-*`/`PL-1.*`/`TRAP-*` 条目均在 pattern-library 核对存在（`CONST-capacity-910B2C`/`CONST-L1-port-bw`/`CONST-fabric-bw`/`CONST-mte2-degradation`/`CONST-store-fixpipe-gm-only`/`PL-1.8-bf16-cube-direct`/`PL-1.1-transpose-chain`/`PL-1.3-host-permute`/`TRAP-DEVMODE-PERSIST-GEMM`/`TRAP-fp16-opmath-golden`）。
- 内部一致：首轮 (a) `num_launched` 矛盾已统一（维度 3）；(b) §8.4 CV 措辞已改（维度 5）；(c) B 装载 trans 分支已补（维度 5）；(d) GEMV 条件已补全（维度 5）。API 映射 / 内存规划 / Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与布局决策四子节，且 §1.6.3 有实质内容（Cube 类布局决策，非豁免）。
- **算法调研存在性与完整性（§1.6.0）— pass**：调研深度标注为「完整调研」并给检索入口。R1 候选表**含基线候选 #0（源分块 Cube 直接法）**，评估 Strassen/split-K/降精度/转置改写/epilogue cast 融合，逐项标等价性初判与是否入 R3；R2 有**明确结论**「无在线变体（基线即天然单遍）」并给结构依据（K 累加本身是 O(block) running accumulator，无需重缩放）+ 指出可选并行变体是 split-K（非 online）；R3 四口径表（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）基线在表、口径写明（乘加=2、访存为 GM 流量、理想口径假设 L2 复用）；R4 逐候选六项亲和评估（单元匹配/容量/对齐/静态边界/流水融合/跨核）；结论「选定基线 + E1 + E2」含关键依据与调研范围，并逐一说明源 14 项优化手段的承接口径。
- **调研结论正确性（独立复核）— pass**：
  - **复杂度算术独立复算**：基线 FLOPs `2·M·N·K` 正确；访存 `(M·K+N·K+M·N)·2`，代表 shape M=N=4096/K=7168 ⇒ `(29.36M+29.36M+16.78M)·2 = 150.99 MB` 正确；中间缓冲 `128·128·4 = 64 KB` 正确；扫描遍数 1 正确；split-K `M·N·P·4` partial 落 GM 正确；Strassen `~0.875·2MNK` 正确。**未漏计中间缓冲 GM 往返**。估算 `150.99MB/1.26TB/s ≈ 120µs`、`/1.73TB/s ≈ 87µs` 复算正确，常数版本戳与 `CONST-mte2-degradation`（2026-09-07 build）/`CONST-fabric-bw` 一致。
  - **负向断言逐条对照复核**：对照 `algorithm-candidates.md`（dense GEMM 无专列行；邻近族 `ALG-conv`（im2col/implicit GEMM/Winograd）与 `ALG-gemv`（Cube 分形 vs Vector 规约、split-K））——Strassen「Cube MAC 阵列乘法削减不转化为 throughput + 数值不稳」成立（MAC 阵列吞吐按 MMA 计，与乘法次数无关）；split-K「本 manifest 并行度已足、partial 落 GM 净亏」有 `CONST-store-fixpipe-gm-only`（跨引擎强制 GM 往返）佐证；降精度「fp16 dst 卡死」有 `T.gemm.md` §2.3 佐证；转置改写「`T.gemm` 原生支持 transpose」成立。**未发现明显漏掉的已知候选**（dense GEMM 的等价变形空间：结合律重排=split-K、快速矩阵乘=Strassen/Winograd 均已覆盖；Winograd 属 conv 专用，对稠密 GEMM 无适用性）。
  - **并列候选裁决路径**：R3 中判定裕度小的 split-K 已明确降级为「仅作 K-极大/M,N-极小场景备选」，主选无并列拍板问题。
- **调研→设计一致性 — pass**：§1.6.0 选定「基线单遍分块 Cube 直接法 + E1 + E2」实际驱动 §1.4 算法描述（L217「与 §1.6.0 调研选定算法一致」）、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构（`T.Pipelined` + `T.gemm`）。迁移任务 §0.5/§0.6 处置与调研选族一致（算法族沿用源算法，仅硬件组织重设计）；源算法优化意图承接逐项说明（E2 承接 K 尾零填充、R4 承接 epilogue 守卫、实现层四项态承接），无静默丢弃，无「选 A 做 B」。
- **数学等价优化完整性 — pass**：§1.6.1 两项采纳均有四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」。E1（epilogue cast 融合）：经首轮建议后**重新定位为非回归约束**（§1.6.0 R1-5 / §1.6.1），明确「源 epilogue 本就单次融合 cast，相对源算子 0 增量收益」，收益仅相对伪基线（额外物化 fp32）口径给出；E2（K 尾零填充，`K'=ceil(K/block_K)·block_K`）使 `block_K∤K` 免 host padding。
- **等价性论证独立推演 — pass**：
  - **E1**：融合 store 单次 RNE（对 fp32 累加器一次舍入）vs 显式 fp32 GM 物化（写）+ 回读（fp32 无损）+ 分离 cast（对同一 fp32 值一次 RNE）→ 两路径比特级一致，独立推演成立；依赖「必须经过 fp32 opmath 域」前提，与 `TRAP-fp16-opmath-golden`（fp64 直接 cast 双重舍入分歧）一致。`verify_equiv.py` 重跑 4/4 `EQUIV_PASS` 且与 §1.6.1 内嵌表一致。
  - **E2**：新增项 `op(A)[m,k]·op(B)[k,n]=0·0=0`（A、B 尾区**双侧**清零，杜绝 `0·NaN`/`0·inf`），仅改变 fp32 求和的加零操作，独立推演成立；脚本角点值（±inf/NaN/−0/subnormal）+ 边界 shape 覆盖，`EQUIV_PASS`。
- **向量化替代完整性 — pass**：§1.6.2 覆盖方案中全部循环/标量点——K 累加（Cube `T.gemm`）、A/B 装载（`T.copy`）、写回+cast（`T.copy`+VCast）、零填充（`T.clear`）均已替代；保留 3 处（#5 block 索引/任务映射、#6 host/kernel 元数据 `num_logical/num_launched/tm,tn,tk`、#7 tile 级顺序依赖 `T.Pipelined`）。**与 §3.3 伪代码、§6 循环结构交叉核对无遗漏**：§6.2 无逐元素标量循环，§6.1 声明与 §1.6.2 一致。
- **不可替代理由充分性 — pass**：3 处保留点逐项给出具体理由类别（block 级索引不构成张量、非逐元素热点；host 元数据编译期常量；`C_local` 跨 K-tile 携带依赖，归约需先物化全 K 破坏 tiling），非泛泛而谈，且无「有向量 API 可替代却未替代」的遗漏。
- **替代 API 可行性 — pass**：`T.copy`/`T.clear`/`T.gemm` 均在 `docs/Tilelang.language/` 找到佐证（与维度 1 同口径）；纯 Cube 路径无 v-prefix 需求。
- **向量化轴与布局决策完整性（§1.6.3）— pass**：Cube 类候选矩阵枚举 4 项（#1 Developer ND-preserving（`alloc_shared`+`T.copy`+`T.gemm`，框架自动 ND→NZ）/ #2 Expert 显式 NZ（`Scope("Cube")`+`alloc_L1/L0C`+`load_nd2nz/store_fixpipe`）/ #3 host permute / #4 显式 `T.transpose`），逐候选量化评分（整除性/repack 代价/UB·L0C 容量/流水融合）。弃选均附量化理由与 pattern-library 实测（host permute `PL-1.3` 106–147µs、`T.transpose` `PL-1.1` ~5.4µs、Expert NZ 引 `TRAP-DEVMODE-PERSIST-GEMM`）。正确声明 I/O layout（ND）是契约、核内 Cube 布局是设计变量；repack 代价机制与 `T.copy.md` §1/§2.3 一致（ND→NZ 在搬运内完成，非 GM 级全量重排）。**实验裁决三件套齐全**：主选 #1 + 备选 #2 结构（§3.3 备选伪代码，含 `Scope("Cube")`/`alloc_L1/L0C`/`load_nd2nz`/`store_fixpipe`，随 DESIGN 冻结进 Stage 4 A/B）+ 裁决计划（代表 shape `ds-v3-prefill-attn-proj`/`wide-n-24576`/`mid-m16-attn`、指标 kernel latency + 反解 NZ 转换开销、**判定阈值备选 < 主选 × 0.95**、回写路径）。判定裕度依赖的未实证常数（NZ 转换开销 vs 框架自动路径）已显式标注并纳入裁决。**未查证到与 pattern-library 实测矛盾的负向论断**。
- **布局决策与实现一致 — pass**：§1.6.3 选定布局（Developer ND-preserving，L0C fp32 累加，`T.Pipelined`）与 §3.3 伪代码、§4.3/§4.5（`A_shared`/`B_shared`=L1、`C_local`=L0C）、§6 循环结构三方一致；Cube 类无逐元素向量轴，epilogue 归属（VCastOp，Vector 单元）在 §5.1/§8.4 正确归属。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式（`K'=ceil(K/block_K)·block_K` + 单次舍入）为输入；§6 循环结构与 §1.6.2 结论一致（无 §1.6.2 未论证的标量循环）。

## 首轮阻塞与建议项复核小结

| 首轮项 | 严重程度 | v1 处置 | 复核结论 |
|--------|---------|---------|---------|
| 问题 1 `min(num_logical, num_physical)` 漏算输出块 | 阻塞 | §0.6 R2 / §3.3 / §5.5 / §6.2 统一 `num_launched = num_logical`，并加「网格数不变量」 | **已修复**（维度 3） |
| 问题 2 §8.4「无 Cube↔Vector 跨引擎流水」 | 建议 | §5.1/§8.4 改述「隐式 Cube→Vector 协同」 | 已落实（维度 5） |
| 问题 3 E1 strawman + 脚本恒等式 | 建议 | E1 重定位为非回归约束；脚本 E1 改双路径对照 | 已落实（维度 8 + 机械复核） |
| 问题 4 B 装载 trans 分支缺失 | 建议 | §3.3/§6.2 补 `trans_a`/`trans_b` 双分支 | 已落实（维度 5） |
| 问题 5 K 尾清零未落伪代码 / GEMV 条件不全 | 建议 | §3.3/§6.2 补显式清零分支；§0.1/§9.3 补全 GEMV 精确条件 | 已落实（维度 5） |

> 检视依据文件: `DESIGN.md`；源算子 `/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py`、`/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py`；`examples/TileOPs/tileops/{kernels/gemm/gemm/gemm.py,kernels/gemm/gemm/_gemm_kernels.py,manifest/gemm.yaml,workloads/gemm.py}`、`examples/TileOPs/tests/ops/test_gemm.py`；`docs/Tilelang.language/{内存操作/T.copy.md,内存操作/T.clear.md,线性代数操作/T.gemm.md}`、`docs/开发指南.md`；`examples/gemm/{example_gemm.py,matmul.py,matmul_dynamic_shape.py}`；`examples/gemm/_gemm_kernel/verify_equiv.py`；`.agents/skills/tilelang-op-design/references/{algorithm-candidates.md,algorithm-research.md}`；`.agents/skills/tilelang-op-optimize/references/pattern-library/{INDEX.md,constants.md,attention.md,traps-runtime.md}`；`.agents/skills/_shared/standards/core-split-strategy.md`；`.agents/tools/design_calc_check.py`。
