# _logical_reduce_kernel（承载 `AnyFwdOp` / `torch.any` 语义）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/any/_logical_reduce_kernel/DESIGN.md`
- 任务类型: **迁移**（scenario=migration, migration_mode=harness；9 维度检视）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`（基础工厂 `_logical_reduce_kernel` / `LogicalReduceKernel`，`op_kind="any"` 分支）
  - 辅助核对：`/workspace/TileOPs-gpu/tileops/ops/reduction/logical_reduce.py`、`/workspace/TileOPs-gpu/tileops/manifest/reduction.yaml#AnyFwdOp`、`/workspace/TileOPs-gpu/tileops/kernels/reduction/_primitives.py`、`/workspace/TileOPs-gpu/tests/ops/test_logical_reduce.py`
  - harness 侧工件：`examples/TileOPs/tileops/{kernels/reduction/any/any.py, kernels/reduction/any/_any_fwd_kernels.py, ops/reduction/any.py, ops/reduction/reduce.py, ops/_multidim.py, manifest/reduction.yaml, tests/ops/test_any.py, benchmarks/ops/bench_any.py, workloads/reduction.py, kernels/reduction/any/.migration_meta.json}`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: **7**（见「非阻塞改进建议」，均不影响结论）
- 工具链实查: tilelang `0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba`（与设计自述一致，`import tilelang` 实测）+ torch 2.9.0+cpu（`/workspace/venv/bin/python`）+ CANN 9.0.0 + Ascend910B3

## 机械复核（D-1/D-5）

### D-1 `verify_equiv.py` 重跑（含 §0 边界语义的 torch 实测锚）

- **解释器定位**：默认 `python3` **无 torch**（`ModuleNotFoundError: No module named 'torch'`），故用本仓 venv：`ls -la /workspace/venv/bin/python` → `/usr/bin/python3.10`，`import torch` → `2.9.0+cpu`。（同前作教训：不定位带 torch 的解释器会假性失败。）
- **执行**：`/workspace/venv/bin/python examples/any/_logical_reduce_kernel/verify_equiv.py` → **exit 0**，末行 `ALL EQUIV_PASS`。重跑输出与 `DESIGN.md §1.6.1` 内嵌结果表**逐格一致**：

| 组 | 设计表 | 重跑（实测） | 一致 |
|---|---|---|---|
| A1 补式 vs 基线 | 1440 行 / 0 违反 / EQUIV_PASS | 1440 / 0 / EQUIV_PASS | ✓ |
| A1c 对照（有序 NE） | 66 处差异 / CONTROL_OK | 1440 行 / **66** / CONTROL_OK | ✓ |
| A2 实列域 vs 256-pad | 1440 / 0 / EQUIV_PASS | 1440 / 0 / EQUIV_PASS | ✓ |
| A2b 窄行 32B pad | 1440 / 0 / EQUIV_PASS | 1440 / 0 / EQUIV_PASS | ✓ |
| A3 原生域 vs fp32 上抛（2¹⁶ 穷举 ×2） | 262144 / 0 / EQUIV_PASS | 262144 / 0 / EQUIV_PASS | ✓ |
| A4 分段 + running-min 合并 | 396 / 0 / EQUIV_PASS | 396 / 0 / EQUIV_PASS | ✓ |
| A5 微 epilogue `f32→f16→i8` | 1440 / 0 / EQUIV_PASS | 1440 / 0 / EQUIV_PASS | ✓ |
| 门禁合计 | 268300 / 0 失败 | **268300 / 0** | ✓ |
| INFO 行（NaN×4 + N=0） | 源/候选/torch 三方 True，有序 NE=False；`torch.any((3,0))=[False,False,False]` | 逐行一致 | ✓ |

- **判别性对照组合格**：A1c 实测 66 处差异（>0）⇒ 对照组有判别力；其中「NaN+零（无非零）」与「全 NaN」行为分歧行，与 `TRAP-vcmp-ne-ordered-nan` 的语义断言一致。
- **空真守卫逐组核对（无空真行）**：逐组复算比较元素数——A1/A2/A2b/A5 = 33 组随机（ΣM=270×3 dtype=810 行）+ 21 组角点（30 行/组=630 行）= 1440 行/组；A3 = 2 × 2¹⁶ = 262144；A4 = 5 个 (case, tn) 组合（3×64+3×32+3×16+30+30=396，`tn>n` 组按脚本设计跳过且**不**计入行数）。全部 >0 且 case 前置条件（分段覆盖 `[0,N)`、pad 宽度 = `align_up(N,256)`、特化条件成立）均满足 ⇒ **无空真 EQUIV_PASS**。
- **D-1 结论**：等价性机器验证完整、可复现、表内自洽，**PASS**（支持维度 8 的等价性论证）。

### D-5 `design_calc_check.py`（含 skip 项的人工逐配置复算）

命令：`python3 .agents/tools/design_calc_check.py --design examples/any/_logical_reduce_kernel/DESIGN.md`（exit 0，advisory）。JSON checks 摘要：

| check | status | detail | 处理 |
|---|---|---|---|
| `ub_budget` | **skip** | "no per-level buffer tables parsed from the memory plan" | **人工复算**（见下 ①②） |
| `l0c_budget` | **skip** | "no block_M x block_N values parsed" | 不适用（纯 Vector，无 L0C）——人工确认 |
| `core_split` | **skip** | "insufficient parsed dims {}; declared logical cores = None" | **人工复算**（见下 ③） |
| `r3_metrics` | **pass** | "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)" | 与 §1.6.0-R3 表逐列核对通过 |

> **skip ≠ pass**，下列为人工复算结论（本族连续三任务复现 checker 对「元素预算型 UB 表 / `ceil(M/bm)` 分核表」的 skip）：

1. **`E_cap` 算术复算**：`196608 // 2 // 12 = 8192` ✓（§4.5/§5.2）；硬上限 `196608/8 = 24576`（fp16）、`196608/12 = 16384`（bf16）✓；`HARD_CAP` fp32 取 16384 = 探针**最大通过实测点**（非理论上限，探针未测 20480 档）——作为“通过点”表述正确。
2. **逐配置 UB 自洽复算**（判据 = 每 block 元素数 ≤ `E_cap`=8192；表格逐行重算）：smoke `(32,256)` bm=8 → 2048 ✓；mask-4k `(32,4096)` bm=2 → 8192 ✓（= 上限档）；`(256,4096)` bm=2 → 8192 ✓；`(128,300)` bm=8 → 2400 ✓；`(64,512)dim=0` bm=8 → 512 ✓；`(4,64,512)dim=1` bm=128 → 8192 ✓；`(4,64,512)dim=0` bm=1024 → 4096 ✓；path C tn=8192 → 8192 ✓；`N=1` → 8 ✓。手工代理口径 `2·elem+1+4` B/元素（fp16/bf16 = 9、fp32 = 13）逐行复算一致；实测量价列（8/12 B/elem × 元素数）逐行复算一致。
   **关键独立证据**：以上**每个** `(dtype, bm, tn)` 组合都在同工具链探针里有**直接编译通过点**（fp16 (8,4096)/(2,4096)/(1,8192)/(1,16384)/(1,24576)、fp32 (2,4096)/(1,8192)/(1,16384)、bf16 (2,4096)/(1,8192)/(1,16384) 全部 OK）⇒ `E_cap` 判据经实测直接支持，非纸面推算。
3. **分核复算**（`num_logical = ceil(M/bm)`，`num_kernels = min(·,40)`，`local = ceil(num_logical/num_kernels)`）：smoke 32/8=4→4/1 ✓；mask-4k 32/2=16→16/1 ✓；mask-32k 32/1=32→32/1 ✓；3d-multidim 128/1=**128**→40/**4** ✓（M=128,N=16384 由 dim=[0,2] 推得，与 `flatten_for_multidim` 语义一致）；`(256,4096)` 256/2=128→40/4 ✓；`(64,512)dim=0` 512/8=64→40/2 ✓；`(4,64,512)dim=1` 2048/128=16→16 ✓；`(4,64,512)dim=0` 32768/1024=32→32 ✓；`(129,512)` ceil(129/8)=17→17 ✓；`(512,)` 1/1=1→1 ✓。**全部自洽**，且核内串行上界全为工厂期静态整数（满足 core-split-strategy §1-③）。
4. **UB 记账口径的 3 处文字不一致**（不影响判据，记建议 S2/S3/S4）：见「非阻塞改进建议」。

### 设计期容量探针重跑（D-3，独立证据，补 skip 之外）

`/workspace/venv/bin/python examples/any/_logical_reduce_kernel/history_version/design_probe_ub.py` → **exit 0**，两节均执行完（末行 `PROBE COMPLETE (both sections ran)`；`if __name__ == "__main__"` 位于文件末尾，**未**出现同族「输出段不全」问题）。

| 复现项 | 设计自述 | 重跑实测 | 一致 |
|---|---|---|---|
| fp16 单价（bm≥2 溢出点） | 8.00 B/元素 | `bm=8/tn=4096`、`bm=4/tn=8192`、`bm=2/tn=16384` 三档溢出均反解 **8.00**（needs 2097408 bits / 32768 元素） | ✓ |
| fp16 最大通过点 | 24576 | `tn=24576` **OK**；`tn=32768`（bm=1）溢出（needs 1573632 bits） | ✓ |
| fp32 单价 / 通过 / 溢出 | 8.00 / 16384 / 24576 | 溢出（24576）反解 **8.00**；16384 OK、24576 OVERFLOW | ✓ |
| bf16 单价 / 通过 / 溢出 | 12.00 / 16384 / 24576 | 24576 溢出反解 **12.00**（needs 2359296 bits）、32768 亦 12.00；16384 OK | ✓ |
| raw 窄行可编译（无需 pad） | N=2/4/64/300 均可编译 | 7/7 OK（N=64/4/2/300 × fp16/fp32） | ✓ |
| IR 形态 | `vcast bf16→f32` 出现在 `vcmp` 两操作数 | 溢出 dump 中可见 `hivm.hir.vcast %17/%18` + `hivm.hir.vcmp` + `hivm.hir.vreduce <min>` | ✓ |

- **独立补充验证（检视方新增，compile-only）**：把设计唯一标注「Stage 3 待实证」的 API `T.vmin` 与 path C 结构（`bm=1` + 静态 `T.serial(num_full)` + `vmin(acc,mn,new_acc)` + `copy(new_acc,acc)` + 静态 `tail` 变体 + 微 epilogue）编译实测：**5/5 OK**（fp32 `(32,32768)`/tn=8192/4 tile、fp16 `(128,16384)`/2 tile、fp16 静态尾块 `(64,8448)`/tail=256、fp32 非 256 尾块 `(16,10240)`/tail=2048、bf16 `(16,16384)`/2 tile，`AscendNPU IR compile success`）⇒ path C 结构与 `T.vmin` 合并链在**本工具链可编译**，设计的「待 Stage 3 实证」项已由检视方提前闭合并给出独立佐证（不改设计）。

### §0 边界语义表的 torch 可执行核对（以参照实现实测行为为锚，33/33 PASS）

`/workspace/venv/bin/python /tmp/opencode/any_boundary_check.py` → exit 0。逐项实测（与 §0.1 表逐格比对）：

| 边界项 | torch 实测 | §0.1 表 | 判定 |
|---|---|---|---|
| `N=0` 空归约 | `torch.any((3,0),dim=-1)=[False,False,False]`；对照 `torch.all=[True,True,True]` | 「空 OR 恒假 ⇒ False（与 all 相反）」 | ✓ **方向正确** |
| `dim=[]` / `dim=()` | 返回 `x.bool()`、shape 不变（no-op） | no-op、不进 kernel | ✓ |
| `dim=None` | 0-dim 标量（`ndim==0`） | Op 层 flatten 成 `(M=1,N=numel)` | ✓ |
| NaN 单元素 / **NaN+零** / 全 NaN / NaN+非零 | 全 **True** | NaN 计入非零 ⇒ True | ✓（判别行方向正确） |
| 全零（`0.0,-0.0`）/ 单个 `-0.0` | **False** | ±0 视为零 | ✓ |
| 亚正规（fp16 min subnormal） | True | 计入非零 | ✓ |
| `±inf` | True | 计入 | ✓ |
| complex（`0+0j`/`0-0j`/`0+1j`/`1+0j`/`nan+0j`/`0+nanj`） | 与 `(re!=0)|(im!=0)` **逐例一致** | Op 层 `to_logical_float32` 语义 | ✓ |
| bool / int64 输入、bool 预转 fp32 | True/False 与语义一致 | Op 层预转契约 | ✓ |
| `keepdim` False/True | shape `(2,)` / `(2,1)`，值一致 | Op 层 `_reshape_output` 处理 | ✓ |

### 设备实查复核（独立于设计自述）

`NPUUtils.get().get_aicore_num()` → **20**；`NPUUtils.get().get_aivector_core_num()` → **40**；`AscendArch().aivector_core_num` → 40、`.ub_cap` → **196608**、`.chip_name` → `Ascend910B`（`npu-smi` 910B3）——与 `DESIGN.md` 头部及 §5.5 的实查记录**逐项一致**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**语义核对（§0.1/§0.2 ↔ 源码 + torch 实测）**：源码 `logical_reduce.py:129-134` 的 `any` 分支为「`[x_f32 != 0.0]` 指示化（1.0/0.0）→ `T.reduce_max(dim=1)`」，输出载体 `int8`（`logical_reduce.py:102/148`，wrapper `out_raw.bool()`@`logical_reduce.py:262`）——§0.1「OR_j(x_j≠0)、y∈{True,False}^M、bool 契约 + int8 载体」**与源码一致**；`_pad_value_for_op("any")=0.0`（`logical_reduce.py:65-69`）与 §0.1/§0.5 一致；host 侧「`_kernel_handles_padding=True` ⇒ 不 pad、传 raw N」经 harness 侧 `ops/reduction` 目录下的 `any.py:80`（类属性 True）+ `ops/reduction/reduce.py:428-490`（`_kernel_handles_padding` 为真时跳过 `F.pad`）**实读确认**；`dim=None ⇒ (M=1,N=numel)` 由 `ops/_multidim.py:59-74`（`kept_shape=[]` ⇒ `[1, reduced]`）**实读确认**；`dim=[]/()` no-op 由 `ops/reduction/reduce.py:318-340`（`_maybe_noop` + `_noop_output_dtype()→bool`）确认。**判据性区分**（kind 契约先于语义，按前作教训）：manifest `AnyFwdOp.outputs.output.dtype = "bool"`（harness manifest:293）、`signature.params` 含 **`keepdim: bool`（default false）**（harness manifest:296）——设计**未照抄**姊妹 `count_nonzero` 的 int64 契约或 `all` 的 `reduce_min`+`1−max` 方向；`ops/reduction/reduce.py:376` 的 `elif op_kind in {"all","any"}` 被正确判为 **roofline 记账 kind 而非实现入口**。边界语义（NaN/±0/空 OR）经 torch 实测锚定（见机械复核），**方向全部正确**（尤其 `N=0` 空 OR = False，与 `all` 相反）。

**算法解读完整性（§0.3 ↔ 源码）**：步骤 1–6 与源码逐句对应且无遗漏/无臆造——步骤 1 ↔ `logical_reduce.py:93-95,104`；2a/2b ↔ `:113-126`；3 ↔ `:129-130`；4(any) ↔ `:132-134`；5 ↔ `:145-148`；6 ↔ `:151`；数据流覆盖 `shared_buf/x_f32/bool_vals/result/out_local` 全部 buffer；host 侧逻辑（validate→dim 归一→movedim→contiguous→reshape→**不 pad**→kernel→`_reshape_output`/`_maybe_noop`/`lru_cache`）与 `ops/reduction/reduce.py` 实读一致。**源缺陷识别正确**：步骤 2b/6 的 `T.copy` 均无 M 尾块守卫（`logical_reduce.py:122`、`:151` 定长 block copy），与 §0.4 末注一致。

**优化手段覆盖（§0.4 ↔ 源码）**：#1 SMEM tiling ✓（`:105,122-126`）、#2 256 对齐 pad + OR 中性掩码 ✓（`:93-95,113-119`）、#3 指示化 + 复用归约原语 ✓、#4 fp32 上抛 ✓（`:106,126`）、#5 `block_m` SMEM 启发式 `[1,2,4,8]` ✓（`:391-398`）、#6 `threads` ✓（`:98,104`，autotune `:420` 的 `[128,256]`）、#7 `lru_cache` ✓（`:75`）、#8 tiled 回退 ✓（`:166-241`，触发 `N_padded>32512 or N_padded·elem>smem_budget` @`:354-357`，`_primitives.MAX_SINGLE_TILE_COLS=32512` 实读）、#9 结果块 `T.copy` ✓、#10 autotune ✓（`:415-443`）——**除 #11 外逐项在源码中落实**（#11 见建议 S1）。**归约/在线证据核对**：源 tiled 工厂的 `acc[i]=T.max(acc[i],tile_acc[i])`（`:211-214`，且初值 `fill(acc,0.0)`@`:194-195`）确为「any 存在在线（运行合并）变体」的直接证据，§1.6.0-R2 引用成立。

**耦合性判定合理性（§0.5）**：14 行逐项四态处置（保留/等价替换/重新设计/舍弃）齐全且方向正确——`threads`/warp 级并行、GPU 256 对齐 pad、GPU copy-layout 约束判为**不可移植**，UB 替代 SMEM、一维 Kernel、元素预算阶梯判为**重设计**，`lru_cache`/语义层判为**保留**；无「warp shuffle 判保留」类错误；舍弃项均写明理由，且有「无舍弃后性能意图丢失」逐条承接说明（#4→A3 穷举、#2→实列域+raw 宽度可编译、#8→R4/C8 容量驱动列分块、#6/#10→harness K8/K9）。与 `migration-analysis.md §5.3`（shared memory→UB/L1）及 `ascend-constraints` 方向一致。

**NPU 重设计可行性（§0.6）**：R1–R6 各有「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 以 De Morgan 补式 + IEEE `eq` 规避 `vcmp("ne")` 的 ordered-NE 陷阱（NaN 三方一致，A1c 实证）；R2 一维 persistent + 静态上界 + 动态 guard（`T.vcast.md §2.4` 同款 guard 先例，实读确认）；R3 实列域不 pad（A2/A2b 中性性 + 探针 raw 宽度可编译）；R4 列分块以 **UB 元素预算**为判据、合并算子取 **min**（与 `all` 的 `vmax` 显式区分，防照抄）、累加器 min 单位元 **1.0**（与 `all` 的 0.0 相反）；R5 元素预算阶梯（替代字节/SMEM 启发式）；R6 尾块三件套修正源缺陷（先例 `_logsumexp_kernel_single.py:74-81,112-115` 实读确认同形态）。硬件约束满足：一维 `T.Kernel(num_kernels, is_npu=True)`、静态循环边界、UB 196608 B、无 L0/L1、bool 不落 GM。

**设计与决策一致性（§1–§7 ↔ §0.5/§0.6）**：§1.4 步骤 1–8、§3.2 映射表（10 行）、§3.3 伪代码、§4.3 buffer 清单、§5.2 路径切换、§6.1/§6.2 循环结构**全部**是决策后的 NPU 算法（`vcmp eq` + i1→f32 指示平面 + `reduce_min` + `(bm,)` 微缓冲补式 + 元素预算双路径 + 双显式 slice 尾块），**无**「§0 判重设计、§3 照抄源方案」的脱节；§11 姊妹差异表与 §9.2 常见错误表互为校验（四项反转 + 空归约单位元 + 出栈链 + 输出 dtype 逐条登记）。

**golden 独立性（§8.1）**：golden = `torch.any(x, dim, keepdim)`（主）/`x.bool().any(dim=-1)`（kernel 级），以 §0.1 语义为唯一依据，**未复刻** §0.6 的补式/指示平面/列分块；与 harness 参照 `tests/ops/test_any.py:163-165`（`x.bool().any(dim=-1)`，GPU 侧 `tests/ops/test_logical_reduce.py:144-145` 同）一致，比对口径 `torch.equal` + 断言 `dtype==bool`（`_exact_compare`@`test_any.py:168-177`）实读确认。

**host/op 层引用路径真实性 + 迁移范围**：§0/§0.3/§0.7 引用的源仓与 harness 路径**逐条 `ls` 核对存在**（18 条全部 OK，含 `_primitives.py`、`ops/reduction` 目录下的 `any.py`、`manifest/reduction.yaml`、`tests/ops/test_any.py`、`benchmarks/ops/bench_any.py`、`workloads/reduction.py`、`.migration_meta.json`、`_any_fwd_kernels.py`）；`.migration_meta.json` 的 `extracted_functions=["_logical_reduce_kernel"]` + `_any_fwd_kernels.py` 的 scope note（"only the baseline factory … tiled variant intentionally not extracted"）与 §0.5-#8/§0.7 的排除声明一致，**无范围蔓延、无多输出变体遗漏**（GPU 侧 `_logical_reduce_tiled_fwd_wrapped` 已显式排除）。kind 范围仅 `any`（all / count_nonzero 为独立迁移单元）已显式声明。

**结论**：维度 0 各检查项 **pass**（唯一瑕疵为 §0.4-#11 的来源标注错误，属建议级，不改变任何语义/重设计结论）。

### 1. API 可行性 — pass

- **API 存在性 + 文档佐证**（逐条实读 `docs/Tilelang.language/`）：`T.vcmp(A,B,C,cmp_mod)`（比较操作/T.vcmp.md：签名/`"eq"` 枚举/输出 bool 且**不可落 GM** §2.3 —— 设计据此把掩码留片上 ✓）；`T.vcast`（数据类型转换操作/T.vcast.md 表含 `i1→f32`、`f32→f16`、`f16→i8`，**确无 `f32→i8`** ⇒ 两步出栈的论证成立）；`T.reduce_min`（规约操作/T.reduce.md：`reduce_mode` 含 `min`、`clear` 参数、rank/单维 shape 规则、dtype 表 **bf16 = ×**（故归约域恒 f32 指示平面）、`clear=False` 静默错值约束）；`T.vsub`/`T.vmin`（数学操作/比较操作，fp32 ✓、**bf16 ✗** ⇒ `acc` 恒 f32）；`T.reshape`、`T.copy`、`T.fill`（须 `tir.const` 类型化常量）、`T.Kernel(n, is_npu=True)` + `T.serial(静态上界)` + guard（guard 形态先例 `T.vcast.md §2.4 示例1`，实读确认）。
- **v-prefix 优先**：全链 `vcmp/vcast/reduce_min/vsub/vmin` 均为 v-prefix ✓，无 `npuir_xxx` 回退。
- **模式匹配**：Developer 模式 + 自动同步，无手工 `set_flag/wait_flag` ✓。
- **实证强度**：path R 全链 3 dtype 编译通过（探针）；**path C 全链（含 `T.vmin` no-alias 合并 + 静态尾块变体）经检视方 5/5 编译通过**——设计唯一「Stage 3 待实证」项在 Stage 2 即被独立闭合（`T.vmin` 失败退路 `vsub+vmax` 亦已备）。

### 2. 内存层级规划 — pass

- 搬运路径完整：`GM[x] →(T.copy 双显式 slice, real_m)→ UB[x_ub] → UB[mask_eq]/UB[ind] → frag[mn/acc] → UB[out_ub] → GM[out]`，**无越级路径**（无 GM→L0/L1 直连，纯 Vector 无 Cube 段）✓；UB 明细（`§4.3`）逐 buffer 标注层级 ✓。
- **UB 预算**：`E_cap = 196608/2/12 = 8192` 元素算术正确，逐配置复算全通过（机械复核 ①②），且**每个配置都在探针里有直接编译通过点**；三层口径（手工代理 / 实测量价 / 硬上限）并列且判据明确取实测量价；退化路径（`dim=None` flatten、`N=1`）单列补行 ✓（`§4.5` 口径纪律 ①）；反例登记（fp16 32768 元素溢出）方向正确。
- L0C/L1 容量不适用（无 Cube）✓。
- 记账文字瑕疵 3 处（S2/S3/S4，均不改变容量判据）见建议节。

### 3. Tiling 策略 — pass

- Block 划分：`§5.2` 给出具体 `block_m`/`tn`/元素数/推导（阶梯 + 元素预算过滤器 + 扩展阶梯触发条件 `ceil(M/bm)>80`），理由明确（摊薄固定开销、受 `E_cap` 封顶）✓。
- 对齐：**无需 pad** 且由探针实测（raw N=2/4/64/300 编译通过）支持；`T.copy` 切片满足 32 B 尾轴 ✓。
- 非整除：M 尾块三件套（`real_m = T.min(bm, M−start)` + src/dst 双 slice + 垃圾行不写出，先例 logsumexp 实读）；N 实列域无 pad；path C 尾块为**工厂期静态宽度的独立代码块**（非 kernel 侧掩码）✓。
- **分核三要素**（对照 `core-split-strategy.md §1/§2.2`）：① 逻辑核数 `ceil(M/bm)` 算式 + 逐负载值 ✓；② 物理核数**实查**（`NPUUtils.get().get_aicore_num()=20`、纯 Vector 翻倍 `get_aivector_core_num()=40`，并 `AscendArch().aivector_core_num=40` 互证；检视方独立实查复现同值）✓；③ 规模判定与方案（≤40「无需适配」/ >40 固定 40 核 + 核内 `T.serial` 静态上界 + grid-stride + guard）✓；核数与 block 取值自洽（机械复核 ③ 逐负载复算一致）✓。
- 中等规模档 `(512,64)`（64 逻辑 vs 40 物理）采用 40 核 + 核内 2（极大规模式 persistent），并已登记「bm=16 对齐单波」为 Stage 4 候选 —— 与标准 §1-③ 第三选项一致，无负载不均超 2:1。

### 4. 技术约束检测 — pass

- 三维 Kernel：**不涉及**（一维 `T.Kernel(num_kernels, is_npu=True)`，M/N 分块均为核内结构）。
- `§3.5.1` 逐条核对 12 项已知限制：bool（i1）不可 load/store GM（掩码留片上 + int8 载体）；`reduce` 不支持 bf16（归约域恒 f32）；`T.vmin` 不支持 bf16（acc 恒 f32）；`T.vcast` 无 `f32→i8`（两步）；循环边界静态；UB 196608 B；L0/L1 不涉及；GEMM 非整除不涉及；尾轴 32B 对齐由探针证明不需 pad。全部处理方案与文档条款一致。

### 5. 循环与同步 — pass

- `§6.1/§6.2` 明确循环类型：M 方向 = 核内 `T.serial(静态上界)` + grid-stride + 动态 guard；N 方向（path C）= `T.serial(num_full)` 静态上界（online min 的循环携带状态）；尾块 = 无循环（工厂期生成独立代码块）；元素级 = 全向量化 ✓。
- 同步：自动同步（Developer 编译器管理 UB/fragment 依赖与 auto-multi-buffer），无手工 flag，无跨核/跨引擎依赖 ✓ 与模式匹配；`pass_configs={}` 与「不显式关闭 auto-multi-buffer」一致（与 `§4.5` 的 2x 口径自洽）✓。
- 尾块：`§6.4` 汇总 M 尾块 / N 向实列域 / path C 静态尾块 / guard 静态化边界 ✓。

### 6. 验证方案 — pass

- Golden：`§8.1` 含可执行 torch 参考（`torch.any`）且满足 `S3-GOLDEN-TORCH` 形态纪律；**独立性**已核（不复刻 NPU 算法）✓。
- L0 计划：L0-1（path R 代表 shape，含 M 尾块 `(129,512)`、退化 `(1,1)`/`(512,)`）/L0-2（判别性角点：全零行、单非零行、**「NaN+零（无非零）」**、全 NaN、±0、inf、亚正规、`N=1`）/L0-3（path C 代表 shape + **tn 不整除 N 的合成档**）/L0-4（分核边界 M=40/32/128 与 `%40≠0`）/L0-5（IR 复核向量化 + `hivm.hir.vreduce <min>` + `bm>=2` 未标量化）——覆盖设计所有分支与陷阱面 ✓。
- 精度标准：输出为 bool ⇒ 逐位精确 `torch.equal` + `dtype==bool`（与 harness `_exact_compare` 一致），并显式声明 atol/rtol **不适用** ✓。
- 判别性输入纪律（`randn` 零密度≈0 ⇒ any 恒 True 的假阳性）已显式登记并给出构造要求（~70% 置零 + 角点 + 两处 NaN 判别行），与 harness `workloads/reduction.py:57-89`（首行全零/次行全非零）互补 ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：`§0`（0.1–0.7 全）+ `§1`–`§11`（含 §1.6 及其 1.6.0/1.6.1/1.6.2/1.6.3/1.6.4）；无占位符（未完成标记、花括号占位、补写提示类文本，检视方 grep 0 命中）。
- 同类引用真实：`§3.5.3` 与 `§0.7` 引用 18 条路径**全部存在**（含 `examples/all`、`examples/count_nonzero` 姊妹单元、logsumexp single/tiled、argmin 同族、harness 全部工件）。
- 内部一致：`§0.2` 契约 ↔ `§3.5.2` 差异表 ↔ `§4` 数据规格 ↔ `§5.6` 集成前置核对 ↔ `§11` 姊妹差异表逐项一致（输出 dtype/载体、raw N 契约、keepdim、`_empty_dim_policy`、`default_config` 回落、msprof 锚点）。

### 8. 算法优化分析 — pass

- **§1.6 四子节齐全**（1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局 + 1.6.4 实验裁决）。
- **调研完整性（R1–R4）**：R1 候选表 **11 项含基线（#0 = GPU 源）**，覆盖直判式 / 幅度式 / 计数式 / vselect / 位域式 / 整数域 / 分段在线 / 跨核 N-split / 排序选择，**无「无更优替代」空白断言**；R2 给出**明确结论「有，且天然在线」**+ 结构判据（OR 可分解为结合/交换/幂等的 running min；逐元素判决无跨元素依赖）+ **源码证据**（源 tiled 工厂 `:197-222` 的运行合并，实读核对）+ 与 logsumexp 的遍数差异辨析（在线价值为容量而非遍数）；R3 **四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）含基线，`design_calc_check.r3_metrics = pass`；R4 六维逐候选评估 + 常数版本戳与「未实测假设」显式标注。
- **调研结论独立复核**：① 等价性——`OR_j(x_j≠0) ≡ ¬AND_j(x_j==0)`（De Morgan + 行内划分）在 2 值域成立，`[x==0]∈{0,1}` 的 `min` 与 `1−min` 恒精确（0/1 可精确表示）⇒ 推演成立（并被 D-1 的 268300 行 0 违反独立确认）；② 复杂度算术复算——基线 `2N_padded` FLOPs / `N_padded·elem+1` Bytes / 1 遍 vs C1 `2N` / `N·elem+1` / 1 遍 ✓，`N=300` 的 `512→300` = **−41.4%** ✓，C1+path C「仍 1 遍、GM 流量不增」✓，C9 的 workspace 往返（`M·nchunk·2B×2`，+2 遍）**已计入而未漏**✓，中间缓冲峰值口径（指示平面 1 个 vs 基线 2 个 4 B/elem 平面）✓；③ 亲和性——与 `ascend-constraints`/pattern-library 实测条目无矛盾（`PL-1.25` 形态骨架、`CONST-reduce-ub-per-element` 的 2-D 行归约 dtype 分档 8/12 B/elem、`TRAP-vcmp-ne-ordered-nan`、`TRAP-parallel2d-bm-ge2-scalarize` 逐条对照，且**明确界定 `CONST-reduce-ub-per-element` 的 3-D 整平面 ~128 B/元素模型对本形态不适用**并以本任务探针重标）；④ **负向断言主动复核**：C3「依赖 `reduce_max` NaN 传播未文档化」——我实读 `docs/Tilelang.language/规约操作/T.reduce.md`，**确无 NaN 条款** ⇒ 断言成立；C5「单价未实测」为透明的证据缺失声明（已登记 Stage 4 待标定）；C10/C9 为复杂度/收益量化否决；对照 `algorithm-candidates.md` 的 `ALG-reduction`（R1「max+sum 合并扫描」/R2「分块归约+合并、Welford」/R3「扫描遍数、跨核归并代价」/R4「水平 vs 垂直 lane 映射」）与 `ALG-elementwise`（CSE / cast 链合并）命中行——**未发现被漏掉的已知候选**（逻辑 OR 归约本无 Welford 类在线统计量需求，设计已说明 0/1 精确域使数值稳定性优化无对象）；⑤ R3 并列候选（C1 vs C5）给出裁决路径（Stage 4 先标定 C5 UB 单价）✓。
- **调研→设计一致性**：§1.4（算法描述）/§3.1（公式拆解以**优化后**公式 `y = 1 − min[x==0]` 为输入）/§6（循环结构与 `1.6.2` 结论一致，无残留逐元素标量循环）全部落地 C1 + 双路径；§0.4 优化手段逐项承接、**无静默丢弃**；迁移侧 §0.5/§0.6 处置与选族一致（`max→min` 的置换、补式方向、`vcmp("ne")` 的否决理由同源）。
- **数学等价优化完整性（§1.6.1）**：采纳 A1/A2/A2b/A3/A4/A5 共 6 项，**每项四要素齐全**（原式 → 优化后公式 → 等价性论证 → 收益量化），我逐项独立推演：A1（De Morgan + IEEE `eq` + `T.vnot` 无 bool 表项 ⇒ 补式必要）成立；A2/A2b（pad 值 0.0 的「命中零」指示量恒 1、`min(x,1)=x` ⇒ min 中性）成立，且同时证明 pad=1.0 会破坏「全零行」判据 ✓；A3（fp16/bf16→fp32 为精确单射、保 ±0/NaN ⇒ 零判决恒等）成立且穷举 2×2¹⁶ 位型验证；A4（分段覆盖 `[0,N)` 无重无漏 + min 结合/交换/幂等 + 累加器恒 0/1）成立；A5（`1−min∈{0,1}` 精确 + `f32→f16→i8(rint)` 对 0/1 位精确）成立。收益量化均有数字（省 1 个全宽 pass / 读流量 −41.4% / 补式落 `(bm,)` 微缓冲 vs +1 全宽平面）。
- **向量化替代完整性（§1.6.2）**：与 `§3.3` 伪代码、`§6` 循环结构**交叉核对无遗漏**——判决/指示化/归约/补式/出栈/常量构造/合并全向量化（#1–#8）；保留 3 处（#9 block 级任务 guard、#10 tile 级顺序依赖的循环控制、#11 host 元数据），理由具体到本算子且发射量均为每 block/tile 1 次量级（非逐元素热点），符合「不可替代理由充分」标准；源实现中的逐元素标量热点（`ite` 融合在 `bm≥2` 标量化、`T.Parallel(bm)` 出栈）识别正确并被替代。
- **替代 API 佐证**：全部 API 有 `docs/` 佐证（与维度 1 同口径），优先 v-prefix ✓。
- **向量化轴与布局决策（§1.6.3）**：按**规约类必选候选**枚举——水平归约（#1）vs 垂直扫描（#2）两条正交 lane 映射**分别评估**，并含 online 交互（#4/#9）、pass 组织（#5 显式向量链 vs 2D `ite` 融合）、补式落点（#6）、归约算子选择（#7）、跨核（#8）；逐候选量化评分（整除性 / 尾 lane 浪费 ≤2.3% / 累加链形态 / repack 代价 / UB 影响）。**弃选论证前提已亲自核对**：#3 核内转置明确引用 `docs/Tilelang.language/创建操作/T.transpose.md`（实读：`hivm.hir.vtranspose`、**UB 级**、二轴交换链；与 `ALG-transpose` 实测「核内转置链 ~µs 级」一致）⇒ 否决理由为「纯开销零收益、量化收益为零」而非「代价高」，**未出现 GM 级重排的稻草人**（符合前作教训）；#2 以发射模型量化否决；#6/#8 有量化依据。**决策与实现一致**：核内布局（行块 `(bm,·)` 行主序 UB 驻留）、向量化轴（N 轴，`reduce_min dim=1`）、补式落点（`(bm,)` 微缓冲）与 `§3.3`/`§4.3`/`§6` 三方一致。
- **实验裁决模式（§1.6.4）**：判定裕度依赖未实证常数（超预算单消费者竞态、发射单价双档 0.25–0.5 µs/op）时，给出**三件套**：① 工厂期静态分派谓词（`E_cap < N ≤ 硬上限(dtype)`，且因工厂按 `(M,N,dtype)` 分派 ⇒ 主选/备选可按 shape 共存，非全局二选一）；② 备选结构完整可落地（buffer 形状 / 128 KB 预算 / 无 tile 循环 / dtype 路径 / 分核三要素 / 伪代码逐条给出，达 Stage 4 深度）；③ 裁决计划（代表 shape + 指标 + **阈值 0.8×** + 回写路径 + 常数回填）✓。
- **编译期特化谓词的充分条件**：`path == "R"`、`tail > 0`、`num_full`、`tn` 均为工厂期 Python 常量，依赖变量集（`N`/`E_cap`/`dtype→HARD_CAP`/`M`）明确，每变体返回独立 prim_func（规避 `TRAP-primfunc-compile-time-if-buffer-def`）；负向守卫由 L0-1/L0-3/L0-4（含 tn 不整除 N 与非整 40 核负载）覆盖 ✓。
- 小结：唯一可改进项为 C3 成本栏未考虑 `reduce_mode="absmax"`（否决结论不受影响），见建议 S6。

## 非阻塞改进建议

> 以下均为**建议级**（不影响 `结论: 通过`，不阻塞进入 Stage 3）。建议在 Stage 3 落地或后续修订时顺手处理。

**S1（§0.4-#11 来源标注错误）**：`block_m × N_padded ≤ 2 × threads` 的 GPU copy-layout 约束**不在本迁移单元的源工厂**中——`logical_reduce.py` 全文无 `2 *`/`layout` 相关代码（grep 0 命中），该约束实为同族**另一算子** `tileops/kernels/reduction/argreduce.py:186-191`（docstring）中 `default_config` 的 `max_block_m_layout = (2*threads)//N_padded`（`:210`、`:240`）。建议把该行来源改注为 `argreduce.py`（并说明「非本单元源工厂」），或删除该行，以免读者据 §0.4 高估本单元源工厂的 GPU 依赖面。**不影响任何迁移决策**（其处置为「舍弃」，NPU 亦无该约束）。

**S2（§4.5 注③ 与表内数值矛盾）**：注③称「fp16 档手工代理（约 36 KB/8192 元素 ≈ 4.5 B/elem）」，而同节表格 `(256,4096)` fp16 行为「9 × 8192 = 73.7 KB」（9 B/elem）——同一配置两个数（疑为姊妹稿残留）。建议统一为 73.7 KB / 9 B/elem。

**S3（§4.5「单价」列的口径）**：fp16 行同格并列「实测单价 8.00」与「溢出反解 6.00 B/元素」，而探针重跑显示：fp16 `bm≥2`（4096/8192/16384 列宽）的 32768 元素溢出点均反解 **8.00**，唯 `bm=1/tn=32768` 反解 **6.00**（needs 1573632 bits = 196704 B）⇒ 「单价」并非严格线性常数（含一个 ~96 B 的常数项/形态相关项）。建议把该列改述为「通过点/溢出面反解区间」并注明探针 `(bm,tn)` 组合，避免 Stage 4 把 8 B/元素当作严格线性模型使用（`E_cap` 取最坏 dtype 12 B/元素的口径不受影响，且 8192 元素在三种 dtype 下都有直接编译通过点）。

**S4（§4.5 单位标注）**：手工代理列以 1 KB = 10³ B 呈现（如 26624 B → 「26.6 KB」），而 UB 容量按 196608 B = 192 KiB 计。建议统一为 KiB（或统一 10³ 并标注），消除口径歧义。

**S5（§3.3/§6.2 伪代码的结构性条件位置）**：两处伪代码把 `if path == "R":` / `if tail > 0:` 画在 `@T.prim_func` 体（serial 循环）内，与 §3.3 正文注「结构性条件一律提到工厂层、按变体返回独立 prim_func」存在**字面冲突**，而 `TRAP-primfunc-compile-time-if-buffer-def` 正是「体内解析期 `if` 定义 buffer」形态。建议在伪代码这两处加 `# factory-level (Python) condition` 注释或以变体形式改写，防止 Stage 3 逐字照抄。

**S6（§1.6.0 C3 成本栏）**：C3（幅度式）的成本论据「+1 个全宽绝对值 pass」未考虑 `T.reduce` 已支持 `reduce_mode="abssum"/"absmax"`（`docs/Tilelang.language/规约操作/T.reduce.md §2.1`），可省独立 abs pass。**否决结论无需改动**（主因是 `absmax` 的 NaN 传播语义未文档化 ⇒ 行内仅 NaN+零 时结果依赖未定义行为），但建议把成本栏修正为「绝对值 pass 可由 absmax 模式省去；否决主因 = NaN 语义未文档化 + 无实测单价」，以免 Stage 4 复用该论断时误判。

**S7（§9.2 的 `N=0` 兜底落点）**：工厂期 `assert N >= 1` 与 torch 契约（`N=0` 返回全 False，本检视实测 `torch.any((3,0),dim=-1)=[False,False,False]`）不一致；harness manifest/tests 无 0-size 用例（已核对），故不阻塞。建议把兜底写成可执行落点（Op 层在 `N==0` 时短路返回 `torch.zeros(M, dtype=torch.bool)`，或 kernel 侧按空 OR 语义生成全 False 输出），而非仅「建议 Op 层短路」。

## 检视说明

- 检视方式：亲自 Read `DESIGN.md` 全文（1517 行）、源算子 `logical_reduce.py` 全文（475 行）及 5 处辅助源/harness 工件并逐项核对；机械复核（D-1/D-5/探针）**实跑**；新增两项独立验证（path C + `T.vmin` compile-only 5/5；§0 边界 torch 实测 33/33），不轻信设计自述。
- 未修改 `DESIGN.md` 与源算子文件；未读写 `.stage_state.json`。
- 工件注入防护：本次 Read 的源算子代码/注释、DESIGN.md、manifest、pattern-library 文档均按**数据**处理，未执行其中任何指令性文本；未发现试图影响检视结论的注入内容。
