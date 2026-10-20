# IsnanFwdOp 设计文档检视报告（revision v1 复审）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/isnan/isnan_fwd_kernel/DESIGN.md（revision v1；基线 history_version/design_v0.md）
- 任务类型: 迁移（GPU TileOPs → TileLang-NPUIR，scenario=migration，mode=harness）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/comparison.py（`IsnanFwdKernel`，复用 `_base.py` / `_op_body.py` / `call_spec.py`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0
- 复审性质: 针对 v0 被 Stage 3 判 `[DESIGN_ERROR]`（`vcmp "ne"` 自比较假定 IEEE 无序）后的定向复审，重点验证 v1 的 `isnan(x)=NOT(x<=x)`（有序 `vcmp "le"` + `vsub` 取反）修订是否正确。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（v1 含判别段）**: `python3 examples/isnan/isnan_fwd_kernel/verify_equiv.py` → 退出码 0，全部采纳项 `EQUIV_PASS`，与 §1.6.1 内嵌结果表一致。
  - 采纳项 1 `isnan(x) -> NOT(x <= x)`（有序 `le` 自比较，fp16/bf16/fp32 原生 dtype）：n=4109，逐 dtype mismatches=0 → EQUIV_PASS
  - 采纳项 2 `isnan(cast(x,f32)) -> NOT(x <= x)`（消除宽化，原生 dtype）：n=4109，逐 dtype mismatches=0 → EQUIV_PASS
  - **判别段（v1 新增，v0 缺）**——有序 `ne` 自比较 `(x<x)|(x>x)` ≠ isnan，`~(x<=x)` == isnan：
    - float16：NaN 数 423，ordered-ne 失配 423（>0），le-取反失配 0 → DISCRIM_OK
    - bfloat16：NaN 数 443，ordered-ne 失配 443（>0），le-取反失配 0 → DISCRIM_OK
    - float32：NaN 数 402，ordered-ne 失配 402（>0），le-取反失配 0 → DISCRIM_OK
  - 结论：判别段确认「有序 `ne` 检不出 NaN、`le` 取反精确等于 isnan」，正是 v0 错误的机器拦截位；与 §1.6.1 内嵌表逐行一致。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/isnan/isnan_fwd_kernel/DESIGN.md`（脚本仅接受 `--design`；任务书的 `--dir` 非合法参数，按 skill 口径改用 `--design`）输出：
  ```json
  {
    "design": "examples/isnan/isnan_fwd_kernel/DESIGN.md",
    "capacity_constants": {
      "ub_kb": 192.0,
      "ub_inflation": 1.7,
      "l0c_kb": 128.0,
      "source": "pattern-library/constants.md CONST-capacity-910B2C"
    },
    "checks": [
      {"check": "ub_budget",  "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
      {"check": "l0c_budget", "status": "skip", "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"},
      {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {}; declared logical cores = 40"},
      {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
    ]
  }
  ```
  - `r3_metrics` pass（R3 四口径齐全）。
  - `ub_budget` / `l0c_budget` / `core_split` 三项 skip → **人工复核结论（机械跳过，人工已核）**：
    - UB 预算按 §4.5 手算复核通过：fp16/bf16 路径 = 16384+8192+16384+16384+16384+8192 = 81920 B = 80KB；fp32 路径 = 32768+8192+16384+16384+16384+8192 = 98304 B = 96KB，均 < 192KB（含 ×1.12 与 ×1.7 膨胀仍安全）。
    - 无 Cube → L0C 不适用。
    - 分核算术按 §5.5 手算复核通过：16M 逻辑核 2048、每核 ceil(2048/40)=52；256M 逻辑核 32768、每核 ceil(32768/40)=820，且 40 = 实查物理核 20 × 2（纯 Vector）。
- **本工具链设备级独立复测（额外焦点，亲自运行）**: NPU 可用（`torch.npu.is_available()=True`，`NPUUtils.get().get_aicore_num()=20`），`python3 examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py` → 退出码 0，fp32/fp16/bf16 三 dtype 全 `[PASS]`：直读 `vcmp` 掩码得 `ne(NaN)=0`（有序）、`eq/le/ge(NaN)=0` 且 `le(非 NaN)=1`，`isnan_via_le` 与 `torch.isnan` 全等。独立证实 v1 前提。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**源码核对证据**（亲自 Read `comparison.py`，并交叉核对 `_base.py` / `_op_body.py` / `_dtype.py`）：

- **语义理解正确（§0.1 ↔ 源码）**：源码 `IsnanFwdKernel.op_func(x) = T.isnan(T.cast(x, "float32"))`（`comparison.py:189-191`），继承 `FloatPredicateKernel`→`FloatUnaryKernel`，`SUPPORTED_DTYPES = FLOAT_DTYPES`（`_base.py:500-510`），`OUTPUT_DTYPE = torch.bool`（`_base.py:506-510`）。§0.1 的「NaN→True、±inf/有限→False、fp16/bf16/fp32 输入、bool 输出」与源码签名一致；§0.1 把 `T.cast(x,"float32")` 判为「谓词可用性适配、非精度提升」正确（宽化不改变 NaN 性，v1 的 verify_equiv 项 2 与设备 repro 双重证实）。
- **算法解读完整（§0.3 ↔ 源码）**：`store_unary_bool_as_int8` 包裹为 `T.if_then_else(op_func(x), T.cast(1,"int8"), T.cast(0,"int8"))`、`BOOL_STORAGE_DTYPE="int8"`（`_op_body.py:50-72`、`_dtype.py`）；`forward` host 侧 `flat(x)` → kernel → `result.view(torch.bool).reshape(...)`（`_base.py:100-101` 等）。§0.3 步骤 1–7 逐条对应，无遗漏、无臆造；GPU 抽取件 `_isnan_fwd_kernels.py` 与解读一致。
- **优化手段覆盖完整（§0.4 ↔ 源码）**：register_copy fragment、coalesced 连续访问、`threads×npt` 两级并行、fp32 宽化谓词、bool int8 字节存储五条均列出并逐条标注硬件依赖；swizzle/mma/warp shuffle 源码确未使用，判定「舍弃（不适用）」无双标。
- **耦合性判定合理（§0.5）**：逐条四态处置（保留 / 等价替换 / 重新设计 / 舍弃）+ 依据，与 GPU→NPU 映射一致；§0.5 已将 v0 的 `T.isnan+宽化 → 重新设计` 指向 **v1 的 `NOT(x<=x)`**（§0.0 调整表 #1）。
- **NPU 重设计可行且语义保持（§0.6）**：R1（一维 Kernel + block_size 折叠 + Vector 整块）、R2（`NOT(x<=x)`：`vcmp(x,x,"le")` 原生 dtype 有序自比较 + 取反）、R3（掩码→`vcast`(i1→f16)→`vsub(1.0−m)` 取反→`vcast`(f16→i8) 字节链）三要素齐全，均满足一维 Kernel、UB 层级、32B 对齐、静态循环边界；R2 语义保持论证「`le` 对非 NaN 恒真、仅 NaN 假 → 取反即 isnan」经独立推演 + verify_equiv + 设备 repro 三重成立。
- **§0.0 复审核心（v0→v1 修订正确性）**：§0.0 明确指出 v0 前提「Ascend `vcmp "ne"` 为 IEEE 无序」在本工具链不成立（`ne` 为有序 `ne≡lt∨gt`，NaN 自比较 false），并给出三条防再犯机制（按当前工具链实跑复核语义常数、比较模式须在目标后端实测、等价性验证加判别项）。**该修订方向正确且证据充分**（见维度 8 与机械复核）。
- **§1–§7 与迁移决策一致（§11.1/§11.2）**：无「§0 判重设计、§3 照抄源方案」脱节；源优化手段 #1–#5 意图逐项承接（#4 由 fp32 宽化谓词升级为原生 dtype 有序自比较取反，属等价化简而非静默丢弃）。
- **golden 独立（§8.1）**：`golden_isnan = torch.isnan(x.detach().cpu().float())`，移植源仓参考实现，不含 `vcmp` 自比较 / `vsub` 取反 / `vcast` 字节链 / persistent 结构，验证独立成立。
- **非阻塞观察**：源码 `isnan_exact` 分支（int/bool → 全 False，`AlwaysFalseFwdKernel`）未迁移，§11.3 明示 NPU 侧仅服务 float（int/bool 抛 `ValueError`），与 NPU manifest 的 float-only workloads（`elementwise.yaml:495-520`）一致，属明确的范围裁剪，非语义偏差。

### 1. API 可行性 — pass

- `T.vcmp(x_ub, x_ub, mask_ub, "le")`：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.1（`"le"` 合法）、§2.2.1（fp16/fp32/bf16 均 √）、§2.3（输出 bool、i1 不可写回 GM）；源码 `tilelang/language/customize_npuir.py:501-520` `npuir_cmp` 断言 `cmp_mod ∈ {eq,ne,lt,gt,ge,le}` 且**无 `unordered` 标志**（亲读）——佐证齐全，v1 选 `le` 有依据。
- `T.vcast(mask_ub, le_f16_ub, "rint")`（i1→f16）与 `T.vcast(inv_f16_ub, y_ub, "rint")`（f16→i8）：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 表确有 `i1→f16`、`f16→i8`（rint），且无 `i1→i8` 直连——设计两步 cast 论断成立。
- `T.vsub(ones_ub, le_f16_ub, inv_f16_ub)`：`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1（fp16 √）+ 设备 repro 实测；双 tensor 形态（非标量 src0），符合 §2.3 限制。
- `T.vnot` / `T.vselect` **弃用论断核对**：`T.vnot.md` §2.2.1（bool/int1 = ×）；`T.vselect.md` §2.2.1（int8/uint8 = ×）——确不能直接 not/select 出 int8，故经 fp16 域 `1.0−m` 取反，论证成立。
- `T.copy` 双显式 slice / `T.Kernel(一维, is_npu=True)` / `T.alloc_ub` / `T.fill`：均有 `docs/` 或同族 `examples/`（hardsigmoid、bitwise_not）佐证，且设备 repro 实跑通过。
- v-prefix 优先：设计全用 v-prefix（`vcmp/vcast/vsub`），符合 AGENTS.md API 约定；`T.isnan` 明确弃用（仅标量谓词）。

### 2. 内存层级规划 — pass

- §4.4 完整路径仅为 GM ↔ UB 两级（`T.copy` GM→UB、`vcmp`/`vcast`/`vsub` 片上、`T.copy` UB→GM），无 GM→L0 越级；无 Cube → 无 L1/L0A/L0B/L0C。
- §4.5 UB 预算：fp16/bf16 峰值 80KB、fp32 峰值 96KB，均 < 192KB（`CONST-capacity-910B2C`）；显式 alloc ×1.12 ≈ 107.5KB、保守 ×1.7 ≈ 163KB 仍安全。手算复核通过（见机械复核段）；`design_calc_check` 的 ub_budget 为机械 skip，已转人工核。
- §4.3 六个 buffer（`x_ub`/`mask_ub`/`le_f16_ub`/`ones_ub`/`inv_f16_ub`/`y_ub`）均标注 UB 层级。

### 3. Tiling 策略 — pass

- §5.2 `block_size=8192` 给出理由（同族实测最优、32B 对齐、UB 预算、摊薄发射开销）。
- 对齐：8192×1/2/4B 均 32B 整数倍，且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓。
- 非整除：`tail_size = T.min(block_size, remaining)` 尾块 + `block_id < num_logical` guard（§5.4/§6.4）。
- **分核策略三要素齐全（§5.5）**：① 逻辑核数 `num_logical = ceil(N/8192)`（16M→2048、256M→32768）；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查——设计记录返回 20，**本次复审亲自复测亦返回 20**（纯 Vector ×2 = 40），查询代码已记录；③ 规模判定与分核方案（极大规模档固定启 40 核 + 核内 `T.serial` 串行，16M→52、256M→820，边界编译期常量）。算术手算复核通过，满足 `core-split-strategy.md`。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（源即一维，§3.5.1）。
- GPU 专用 API（`T.isnan` 特殊函数、`T.cast` 宽化、`T.Parallel(threads,npt)`、bool i1 GM 存取）已全部改写为 NPU v-prefix + UB 路径（§3.5.1/§3.5.2）。
- GEMM 非整除 / L0C 溢出：不适用（无 Cube）；`design_calc_check` l0c_budget skip 记为人工确认不适用。

### 5. 循环与同步 — pass

- §6.1 循环结构明确：核级并行 `T.Kernel(num_kernels, is_npu=True)`、tile 级 `T.serial(num_local_tasks)`（静态边界）、元素级 v-prefix 整块（无逐元素标量循环）。
- §7 同步策略为 Developer 自动同步，与 §2 模式一致，无手动 `set_flag/wait_flag`。
- §6.4 尾块逻辑完备（`tail_size` + guard），与 §5.4/§1.6.2 一致。

### 6. 验证方案 — pass

- §8.1 Golden 函数为 `torch.isnan`（PyTorch 参考实现草案），明确以 §0.1 语义为唯一依据、不复刻 NPU 算法（独立性成立）。
- §8.2 精度标准：三 dtype `atol=rtol=0`（精确 bool），与源仓 `_get_tolerances` 一致。
- §8.3 L0 计划：`N=1048576` 三 dtype + 注入 NaN/±inf + 精确相等比对 + 非 float dtype 负向用例；**特别地**，§8.3/§9.3 设置「全 NaN 输入须全 True、全 ±inf 须全 False」判别用例，可在 Stage 3 直接拦截退回 v0 的 `ne` 错误——设计闭环良好。

### 7. 完整性与一致性 — pass

- 章节齐全：模板 11 章 + §0（0.1–0.7 全）+ §0.0（v0→v1 调整表）+ §11 一致性核对。
- 无占位符（`{placeholder}` / `TODO` / `待补充` 均无；§10.2 的「待实现」为交付状态描述，非占位符）。
- §3.5.3 引用路径经逐一核对**全部真实存在**（hardsigmoid DESIGN + NPU kernel、bitwise_not DESIGN + NPU kernel、`_isnan_fwd_kernels.py`、`isnan.py`、`ops/.../isnan.py`、`test_isnan.py`、`bench_isnan.py`、manifest `elementwise.yaml`）。
- 内部一致：§0.0 调整表 ↔ §0.5/§0.6 ↔ §1.4/§1.6.1/§3.1–§3.3/§4.4/§4.5/§9.1/§11 全部同步为 v1 的 `NOT(x<=x)`，无残留 v0 的 `vcmp "ne"` 表述（除 §0.0/§1.6.0/§9.1 作为「已推翻前提」的显式反例记录，属应有内容）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：§1.6 含 1.6.0 调研 / 1.6.1 等价优化 / 1.6.2 向量化替代 / 1.6.3 轴与布局决策，四子节齐全。

**算法调研存在性与完整性（§1.6.0）**：R1 候选表含**基线候选 #0**（源公式 `T.isnan(T.cast(x,"float32"))`），并覆盖候选 1（有序 `le` 取反，采纳）、候选 1b（`vcmp "ne"` 自比较，弃选并给出本工具链语义推翻依据）、候选 2（位域）、候选 3（`x-x!=0`）；R2 在线算法**明确结论「无（天然单遍）」并给结构依据**（逐元素无跨元素依赖，对照 `algorithm-research.md` §3 R2 判据）；R3 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）+ 可并行度，基线在表；R4 逐候选对照六项清单评估；「调研结论」与「无更优替代」调研范围均写明。

**调研结论正确性独立复核**：
- ① 等价性独立推演：`isnan(x) ⟺ NOT(x <= x)` 成立——IEEE-754 **有序**比较下仅 NaN 使 `x <= x` 为假（±inf 满足 `inf<=inf`、±0/次正规/有限均满足 `x<=x`），取反即 isnan。**独立复算**：`verify_equiv.py` 重跑 fp16/bf16/fp32 三 dtype 采纳项 mismatches=0；**设备级复测** `repro/CONST-vcmp-ne-ordered.py` 三 dtype 直读掩码 `ne(NaN)=0`、`le(NaN)=0`、`le(非 NaN)=1` 且 `NOT(x<=x)==torch.isnan` 全等——推演与实测一致。
- ② 复杂度算术独立复算：R3 访存口径 = 读 N×in_bytes + 写 N×1B，无中间 GM 往返（v1 仅多发一步片上 `vsub`，不增 GM 流量）；设计期估算复算——16M f16/bf16 `3N/1.208TB/s ≈ 41.7µs`、16M f32 `5N/1.208TB/s ≈ 69.4µs`、256M f16/bf16 `3N/0.736TB/s ≈ 1094µs`，与 §1.6.0 数字一致；`design_calc_check` r3_metrics pass。
- ③ 亲和性判断核对：主选纯 Vector 亲和（`VCmpOp` 三 dtype 支持、UB 驻留、32B 对齐、静态边界），与 ascend-constraints 不矛盾；容量淘汰引用 `constants.md CONST-capacity-910B2C`。
- ④ **负向断言主动复核（关键）**：设计的核心负向断言是「本工具链 `vcmp "ne"` 为有序不等、检不出 NaN，故弃用」。该断言**与 pattern-library 条目 `CONST-vcmp-ne-nan-unordered`（INDEX.md:116 / constants.md:227，status=verified）直接矛盾**，故本检视不采信设计自述，亲自在设备上重跑判别探针：`ne(NaN)` 自比较 = 0（有序），确证 v0 前提被推翻。进一步核对设计给出的推翻解释——旧 repro 非判别（其 hardsigmoid 链 `vadd→vclamp→vmul` 自身传播 NaN，断言 `torch.equal(isnan(x),isnan(y))` 恒过，不读出 `ne` 掩码）；亲读旧 repro 确认其确实**不直读掩码**（`CONST-vcmp-ne-nan-unordered.py:75-76` 用 `vselect(mask, buf2, buf1, buf1)`），设计对「旧 repro 不可判别」的判断成立。对照 `algorithm-candidates.md ALG-elementwise` 命中行，isnan 候选空间（自比较 / 位域 / 特殊函数 / exp==255）已覆盖，无已知候选被漏。
- ⑤ R3 并列候选：候选 1 相对基线在带宽受限下量级相同但**严格占优**（少一次宽化 cast、少一份 fp32 中转缓冲、无特殊函数；多一步片上 `vsub` 不增 GM 流量），非纸面拍板、无并列争议。

**弃选论证前提核对（强制，亲查 API 文档）**：
- 基线「NPU 无向量 `T.isnan`」——亲验 `T.isnan` 仅标量（签名 `(x, span=None)`）。
- 候选 1b 弃选前提「`vcmp "ne"` 有序」——**设备实测核对**（非仅文档）。
- 候选 2 淘汰前提「bf16 无 `vbitcast`」——亲验 `docs/Tilelang.language/数据类型转换操作/T.vbitcast.md` §2.2.1（fp16 √ / fp32 √ / bf16 ×）。
- §3.2 弃用 `vselect`（int8/uint8 不支持）——亲验 `T.vselect.md` §2.2.1；弃用 `vnot`（bool/int1 不支持）——亲验 `T.vnot.md` §2.2.1；`vcast` 无 `i1→i8` 直连——亲验 `T.vcast.md` §2.2.1 表。
- 结论：弃选/否决前提均亲自打开对应文档（或本工具链实测）核对一致，无「核内重排写成 GM 级」类代价机制矛盾。

**数学等价优化完整性（§1.6.1）**：对选定算法系统审视，采纳 2 项，每项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）齐全；优化结论明确「无进一步优化空间」并给依据（单次逐元素谓词、无 CSE、无超越函数、无中间 GM 往返）。
**等价性论证正确性**：两项独立推演成立（有序 `le` 取反恒等；宽化保持 NaN 性），与 §8.2 精确标准（atol=rtol=0）一致；`verify_equiv.py` 重跑与 §1.6.1 内嵌表一致。
**向量化替代完整性（§1.6.2）**：覆盖全部循环/标量点，并与 §3.3 伪代码、§6 交叉核对——`vcmp`/`vcast`/`vsub`/`T.copy` 取代逐元素标量与 `if_then_else`；保留 2 处标量（block 起始索引、尾块 `T.min`）逐项给出可接受类别理由（block 级任务映射 / 动态边界），§6 不出现逐元素标量循环。
**替代 API 佐证**：`vcmp`/`vcast`/`vsub`/`T.copy` 均有 `docs/Tilelang.language/` 与设备 repro 佐证（优先 v-prefix）。
**向量化轴与布局决策完整性（§1.6.3）**：输入展平为 1-D `(N,)`，仅一个非 batch 轴——按 skill Phase 2 第 3 项「单轴算子可写明『仅一个候选轴』豁免」给出豁免说明，合理；无 repack、无重排候选，向量化轴 = 展平轴（stride=1 连续），整除性与 UB 影响均已说明；迁移对照了源 GPU 的隐式并行轴（`threads×npt`）并判定不迁移。
**布局决策与实现一致**：§1.6.3 选定布局（1-D 连续、展平轴向量化）与 §3.3 伪代码、§4 内存规划（buffer 均 1-D）、§6 循环结构三方一致。
**与下游章节一致**：§3.1 以 §1.6.1 优化后公式（`pred=NOT(x<=x)`，原生 dtype）为输入；§6 循环结构与 §1.6.2 结论一致。

## Review Result

- stage: 2
- project: isnan
- operator: isnan_fwd_kernel
- output: examples/isnan/isnan_fwd_kernel/REVIEW.md
- is_migration: true
- source_op_path: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/comparison.py
- conclusion: 通过
- validation: pass
- validation_details:
  - 结论行存在: pass（`结论: 通过`）
  - 结论一致: pass（9 维度全 pass，无阻塞级问题）
  - 维度完整（迁移 9）: pass（维度 0–8）
  - 维度 0 有源码证据: pass（每项附 `comparison.py`/`_base.py`/`_op_body.py`/`_dtype.py` 源码位置 + DESIGN 章节号）
  - 维度 8 有推演证据: pass（等价性独立推演 + 复杂度复算 + 负向断言设备级复核 + 弃选前提亲查文档 + §3.3/§6 交叉核对）
  - 建议完整: n/a（结论通过）
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义 pass / 算法 pass / 优化 pass / 耦合性 pass / 重设计 pass / 一致性 pass / golden 独立性 pass
- algo_opt_findings: 调研完整性 pass / 调研结论正确性 pass / 调研→设计一致性 pass / 优化完整性 pass / 等价论证 pass / 向量化覆盖 pass / 不可替代理由 pass / 替代 API 佐证 pass / 与 §3 §6 一致性 pass
- suggestion_count: 0
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md; .agents/skills/_shared/standards/gate-and-retry.md; .agents/skills/tilelang-op-design/references/{algorithm-candidates.md,algorithm-research.md}; .agents/skills/tilelang-op-optimize/references/pattern-library/{INDEX.md,constants.md}; docs/Tilelang.language/{比较操作/T.vcmp.md,数据类型转换操作/T.vcast.md,数据类型转换操作/T.vbitcast.md,条件操作/T.vselect.md,逻辑操作/T.vnot.md,数学操作/T.vsub.md}
- summary: v1 以有序 `le` 自比较取反 `NOT(x<=x)` 替换 v0 被推翻的 `vcmp "ne"` 前提，等价性经 verify_equiv 判别段（ne 失配>0、le 取反失配=0）与设备级 repro（三 dtype）独立复核成立，9 维度全 pass。
- issues: none
