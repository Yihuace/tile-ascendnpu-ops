# _batch_norm_bwd_wide_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md（revision v1，相对 history_version/design_v0.md）
- 任务类型: 迁移（harness 模式）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/norm/batch_norm.py（`_batch_norm_bwd_wide_kernel` L1658 + host 类 `BatchNormBwdWideKernel` L1981–2041）；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`（`BatchNormBwdOp`，返回 `(grad_x, grad_weight, grad_bias)` L178–181）；NPU wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`；Stage 0 提取件 `.../norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

> 本报告为 **re-review**（v1 修订后）。v0 判定「不通过」（2 阻塞 + 5 建议），v1 作定点修正。逐项复核见维度 0 / 维度 2 / 维度 3 / 维度 5 / 维度 7。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（与 §1.6.1 内嵌表逐格一致，全 EQUIV_PASS）**
  - 命令: `python3 examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/verify_equiv.py`（torch CPU、fp64 参照 + 角点值），退出码 0
  - 结果（采纳项 O1+O2 联合，21 案例 = 3 dtype × 7 shape）:

    | dtype | max ulp | 违反率 | 结论 |
    |-------|---------|--------|------|
    | float16 | 0.00 | 0.0000 | EQUIV_PASS |
    | bfloat16 | 0.00 | 0.0000 | EQUIV_PASS |
    | float32 | 0.00 | 0.0000 | EQUIV_PASS |

  - 判别段 A（raw-moment `gw` 拒绝证据）：fp32 max 违反率 = **0.1553**（N=4,S=2048 大均值小方差），fp16/bf16 = 0.0000（宽容差掩蔽）→ **DISCRIM_OK**。三处数字与 DESIGN §1.6.1 内嵌表（0/0/0、判别段 0.1553）完全一致。
- **design_calc_check 重跑**（命令：`python3 .agents/tools/design_calc_check.py --design examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`）：

```json
{
  "design": "examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {
      "check": "ub_budget",
      "status": "skip",
      "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"
    },
    {
      "check": "l0c_budget",
      "status": "skip",
      "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"
    },
    {
      "check": "core_split",
      "status": "skip",
      "detail": "insufficient parsed dims {'N': 1}; declared logical cores = 40"
    },
    {
      "check": "r3_metrics",
      "status": "pass",
      "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"
    }
  ]
}
```

  - **skip 项人工复核结论**:
    - `ub_budget`（机械跳过，人工已核）——按 §4.5 的 `驻留 = BPE·N·bS`（**v1 统一 BPE = itemsize + 16**：fp16/bf16 = 18 = stage 2 + g32/xh32/prod/brc 各 4；**fp32 = 20** = stage 4 + 16，`DESIGN.md` L701/L462）复算：held 档最大 resnet-stage1 `8·1024·18 = 147,456 B`、tiled 档最大 `4·2272·18 = 163,584 B`、fp32 resnet-stage3 `4·784·20 = 62,720 B`，均 ≤ `UB_SAFE_BYTES=163,840`（160 KiB）且 ≤ 192 KiB 物理容量 ✓。**v0 的 fp32=16 低估已修正**；§4.5 表值 2272 与 §3.3/§5.2 公式同源可复推。
    - `l0c_budget`（机械跳过）——纯 Vector，无 Cube/L0C/L1，不适用；§4.4/§5.3 无 Cube 路径。
    - `core_split`（机械跳过，人工已核）——`num_kernels = min(C, 40)`、`num_local_tasks = ceil(C/40)`（64→40/2、128→40/4、256→40/7）算术自洽；`40 = get_aicore_num()(实查 20) × 2`（纯 Vector 翻倍）；核内 serial 边界为 host 编译期常量 ✓。
    - `r3_metrics`——pass，与 §1.6.0 R3 四口径表一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_bn_bwd_wide` L1682–1789、host `BatchNormBwdWideKernel` L1981–2041）、`ops/norm/batch_norm.py`、NPU op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`、NPU wrapper `.../batch_norm_bwd/batch_norm_bwd.py`、Stage 0 提取件、探针 `batch_norm_bwd_probe.py` + `PROBE.md`，逐项与 §0 核对。**两个 v0 阻塞项复核如下，均已修复。**

- **阻塞项 1（out_idx 返回序）— 已修复（源码 + 框架源码双证据）**：v1 §0.2（L74–78）保持源 `@T.prim_func` 参数名/顺序 `(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`，`out_idx` 改为 **`[7, 5, 6]`**，返回序 = 列表序 = `(grad_x, grad_weight, grad_bias)`。核对 tilelang 框架源码（真实文件 `/home/developer/tilelang-mlir-ascend/tilelang/jit/jit_npu.py`）：`_normalize_out_idx`（L40–59）**只**做列表化与负索引归一，**不排序**；`__call__` 返回 `[full_args[i] for i in self.out_idx]`（L1352）→ `[full_args[7], full_args[5], full_args[6]]` = `(grad_x, grad_weight, grad_bias)`；`num_inputs = total_params - len(out_idx)`（L1276–1278）= `8 - 3 = 5`。**与 op/wrapper 契约 `(grad_x, grad_weight, grad_bias)`（NPU op L178–181、wrapper L242–243 返回值）逐位一致**。三处口径一致：§0.2（L74–78）、§3.3（`@tilelang.jit(out_idx=[7, 5, 6], ...)` L472 + 参数注释 L487–489 + 注 L588）、§9.2（L947，「保持源参数序时 `out_idx` 必须写 `[7,5,6]`」——v0 判据倒置已改正）；并同步 §4.2/§4.7/§10.5。§3.3 注新增 wrapper 调用/解包约定（5 参调用 `grad_x, grad_weight, grad_bias = program(...)`），与 `num_inputs=5` 自洽；当前 wrapper（L241–242）仍为 7 参调用，已在 §9.1 列入集成期修改清单。
- **阻塞项 2（tiled 尾 tile reduce 垃圾列）— 已修复（含 API 文档证据）**：v1 §3.3 pass-1 两处 `T.reduce` 均显式传 **`size=[N, real]`**（L541 `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[N, real])`、L542 对 `prod`），`real = T.min(bS, S - s0)`（L522）截断到有效列 `[0, real)`；装载/写出仍用 src/dst 双显式 slice。核对文档 `docs/Tilelang.language/规约操作/T.reduce.md` §2.1：`size` 参数「控制 reduce 实际参与计算的数据范围」——用其限定归约 extent 与 v0 缺陷根因（对含未初始化垃圾列的完整 `(N,bS)` 做无边 `dims=[1]` 行和）对应；§3.3 后注（L588）已删除 v0 的错误「按列隔离」论证，改述为「slice 管 I/O、`size` 管归约，二者成对」。四处口径一致：§0.6 R3（L171）、§3.3（L539–542）、§5.4（L758）、§6.4（L816）、§9.2（L948）；§8.3 新增 L0-13（(4,64,4097)）与既有 L0-8/L0-9（按 bS=2272 均命中尾 tile）共同门住该路径。
- **语义理解正确性 — pass（源码证据）**：§0.1 三式与源码逐句一致——`grad_bias[c]=Σ grad_out`（L1750–1751）、`grad_weight[c]=Σ grad_out·x_hat`（L1719–1720/1752）、`grad_x = w·rstd/L·(L·g − gb − x̂·gw)`（L1754–1758 系数 + L1767–1771 写出）、`x_hat=(x−mean)·rstd`（L1716–1718）；元素映射 `l → [l//S, c, l%S]`（源码 L1708/1711 `head//S, bc, head%S`）与 §0.1「沿展平轴 `N·S`、原生 `{0,2}` 轴、通道轴为中间轴（逐通道平面 GM 行跨步 `C·S`、行内 stride=1）」一致。**累加顺序** §0.1 述「固定 shuffle 树 + per-warp 串行合并」与源码 L1734–1736（`butterfly_depth = 32.bit_length()-1 = 5`）+ L1740–1748 一致。**dtype 语义**「fp32 opmath + 单次舍回」与 `accum_dtype="float32"`（L1670）、`T.cast(..., dtype)`（L1767/1774）一致。**边界** `exact = steps*threads*num_per_thread==L`（L1677）+ 非整除 `if l < L`（L1724）一致。**空张量**：源码证据核实——GPU op `TileOPs-main/src/tileops/ops/norm/batch_norm.py` 有 `numel()==0` 早退，而本迁移所用的 **NPU op**（`examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py` forward L105–181）**确无**该守卫；v1 §0.1（L61）/§9.1（L929）/§9.3（L960）已明确该拦截须由 NPU op/wrapper 承接并列入集成清单，L0-14 覆盖。
- **算法解读完整性 — pass（源码证据）**：§0.3 步骤表 P0–E2 覆盖源码全部计算语句（P0 L1696–1698 / P1a-b L1710–1713 / P1c L1714–1720 / P1d L1721–1732 / R1 L1734–1736 / R2 L1738–1748 / W1 L1750–1752 / E0 L1755–1758 / E1 L1760–1773 / E2 L1774），无遗漏/无臆造；数据流图覆盖 go_held/x_held/acc/acc_xhat/warp_sum/params/out；host 侧逻辑（`_block_launch`、预分配 gw/gb、`lru_cache`、op 层 view）与 L1981–2041 / NPU op L159–181 一致。
- **优化手段覆盖完整性 — pass（源码证据）**：§0.4 十一项可逐项溯源（register 驻留 L1699–1700、shuffle 树 L1734、two-accumulator L1719–1720、固定序 L1734–1748、epilogue 预折叠 L1755–1758、fp32 opmath L1715、`T.vectorized` L1710/1773、`exact` 快路径 L1677/L1709、一通道一 block L1692、lru_cache L1657、out_idx L1674），未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass**：§0.5 每条四态处置（保留/等价替换/重新设计/舍弃）有依据，与 GPU→NPU 能力映射不矛盾——register 驻留→`T.alloc_ub` 等价替换、shuffle 树→`T.reduce` 重新设计、`threads`/`T.shfl_xor`/`T.sync_threads`→舍弃、一通道一 block→persistent 分核重新设计、谓词→`T.min`+双 slice 重新设计；舍弃项在 §0.6 末尾意图承接汇总（L186–190），无静默丢弃。
- **NPU 重设计可行性 — pass**：R1（persistent 分核，核内 serial 边界静态）、R2（UB 两步 `T.reduce`，探针真机 PASS）、R3（**已修正：`T.min`+双显式 slice + reduce `size=[N,real]` 截断**，语义保持论证具体）、R4（bf16 无 Vector 算术→fp32 中转必需，有 docs 佐证）、R5（单核 S-tile 两扫描，无跨核归并）均满足 NPU 约束（一维 Kernel、UB 层级、静态边界）且语义保持论证成立。
- **设计与决策一致性 — pass**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法；v0 的两处脱节（out_idx 返回序、§4.5 表 bS 与公式矛盾）均已消除。
- **golden 独立性 — pass**：§8.1 golden 直接调用 `torch`（`.float()`、`.sum(dim=(0,2))`、逐元素仿射），**不含** NPU 设计的 persistent 分核、两步 `T.reduce`、`T.vbrc` 广播、fp32 中转链、分布式 epilogue 等任何算法特征；以 §0.1 语义为唯一依据（移植源仓参考实现），验证独立 ✓。

**维度 0 结论: pass**（语义/算法/优化/耦合性/golden 独立性各项与源码核对通过；两个 v0 阻塞项——out_idx=`[7,5,6]` 返回序、pass-1 reduce `size=[N,real]` 截断——均已修复且三处口径一致）。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有文档佐证并经 Glob 核实路径存在：`T.Kernel(一维, is_npu=True)`（`编译器提示操作/T.Kernel.md`）、`T.alloc_ub`（`内存操作/T.alloc_ub.md`，Expert 显式 UB）、`T.copy` 双显式 slice（`内存操作/T.copy.md`，strided 区域 + §2.2.2 size-1 维折叠支持 3D→2D 切片）、`T.vcast(rint)` 四方向（`数据类型转换操作/T.vcast.md`）、`T.reduce` dims/clear/**size**（`规约操作/T.reduce.md` §2.1/§2.2.2，bf16 ×）、`T.reduce_sum`、`T.vbrc` `(1,1)→(N,bS)`（`shape操作/T.vbrc.md` §2.2.2）、`T.vmul/vsub/vadd`（fp16/fp32 ✓ bf16 ×，行/列/标量广播）、`T.min` 标量（`比较操作/T.min.md`）、`T.clear`（`内存操作/T.clear.md`）；§3.5.3 引用的 examples 路径全部真实存在。**Expert/Vector 形态有先例**：`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py`（Expert + 单 `T.Scope("Vector")` + `T.alloc_ub`，未手写 sync 的 `T.reduce`），`decision-tree.md` 明确 `T.reduce` 两模式可用。标量乘（`T.vmul(coef, L_inv, ...)`，Python 常量作 src2）由探针 `batch_norm_bwd_probe.py` L168/L169 真机 PASS 佐证。v-prefix 优先。

### 2. 内存层级规划 — pass

§4.4 完整给出 GM→UB 路径（无 GM→L0 直连），全部计算 UB 内闭环、无中间 GM 往返；§4.3 每 buffer 标注层级。§4.5 预算人工复算达标（见「机械复核」ub_budget）。**v0 的 fp32 BPE=16 低估已修正为 20**（§3.3 `bpe = elem_bytes + 16` L462、§4.5 L701 同源可复推），fp32 resnet-stage3 `4·784·20 = 62,720 B ≤ 160 KiB`；held/tiled 判定公式与 bS 反推公式同一来源，表值与公式可互推。Cube 路径不适用。

### 3. Tiling 策略 — pass

§5.2 block 划分明确（向量轴 = S 尾轴、归约内层维 `dims=[1]`、N 为外层 reduce）；对齐：行内 stride=1、fp16/bf16 向量宽 ×8 / fp32 ×4、尾 lane 浪费 ≤0.5%（§1.6.3）；非整除经 `T.min`+slice+reduce `size`（§3.3/§5.4）。**分核策略三要素齐全**：① 逻辑核数 `num_logical = C`（§5.5 表）；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查 = 20（记录查询代码 + 返回值，与 `constants.md` CONST-aicore-910B2C 2026-10-12 注一致），纯 Vector 翻倍 → 40；③ 极大规模分支 `num_kernels = min(C,40)`、核内 `T.serial(num_local_tasks)` grid-stride、`if c < C` 守卫、边界静态（host 编译期常量）。核数与 block 取值自洽，负载不均 ≤1 任务。**v0 的 §4.5 表 bS 与 §3.3/§5.2 公式矛盾已修正**——表值统一为公式值 2272（L712–713/715，`4·2272·18 = 163,584 ≤ 163,840`）。§5.5 已按 v1 建议 7 标注宽核实际服务范围（`_holder` 判 `thread`/`block` 者；large-spatial 默认路由 `bwd_split`，L787）。

### 4. 技术约束检测 — pass

§3.5 逐项核对：无三维 Kernel（一维 `T.Kernel`）；GPU 专用 API（`T.shfl_xor`/`T.shift_left`/`T.sync_threads`/`T.get_thread_binding`/`threads=`）已在 §0.5 舍弃/替换；无 GEMM 分形/非整除；L0C 不适用（纯 Vector）；物理核数适配见 §5.5（实查 20→40）；bf16 无 Vector 算术经 fp32 中转链（必需，`T.vmul.md` §2.2.1 bf16 列为 ×，已核对）；中间轴归约（非尾轴）经 strided 平面 gather + 两步 `T.reduce`（探针真机验证）。无遗留阻塞约束。

### 5. 循环与同步 — pass

§6.1 明确循环类型（任务维 `T.Kernel`+`T.serial`、S-tile 维 `T.serial`、元素维向量化、尾块标量 `T.min`+slice+reduce `size`）；§7 同步与 Expert 模式匹配——单 Vector scope 内顺序依赖链、编译器按 MTE2→V→MTE3 自动插 `pipe_barrier`，无跨引擎/跨核协作，无需手动 `set_flag/wait_flag`（conv1d Expert/Vector 先例佐证）；§6.3 说明不显式 `T.Pipelined` 的理由与 Stage 4 备选；§6.4 尾块逻辑**已与 §3.3 实现口径一致**（装载/写出双显式 slice **且** pass-1 两处 `T.reduce` 传 `size=[N, real]`，v0 的「文字说有效 extent、代码无边 reduce」不一致已消除）。

### 6. 验证方案 — pass

§8.1 含独立 PyTorch 参考实现草案（golden 独立性见维度 0）；§8.2 精度标准取自源仓 `batch_norm_bwd_tolerance`（fp32 1e-5 / fp16 1e-2 / bf16 2e-2，含通道梯度 `sum_atol`），与探针口径一致；§8.3 L0 计划 14 项覆盖 fp32/fp16/bf16、held/tiled、S=1/N=1 退化、非向量宽整数倍 S=900、微型 shape，三输出各自比对。**v1 补齐尾块回归**：L0-8（large-spatial，bS=2272 → 462 S-tile，尾 tile 1184）、L0-9（test，2 S-tile，尾 tile 1824）、L0-13（(4,64,4097)，尾 tile 1825）均命中非整除 reduce 路径，直接门住 v0 阻塞项 2；L0-14 覆盖空张量 `L=0`（op 层拦截）。计算：`1048576 % 2272 = 1184 ≠ 0`、`4096 % 2272 = 1824 ≠ 0`、`4097 % 2272 = 1825 ≠ 0` ✓。

### 7. 完整性与一致性 — pass

§0–§11 章节齐全（迁移类含 §0，0.1–0.7 子节齐全）；全文无占位符标记、无未填写段落；§3.5.3 引用的 examples 路径经 Glob 全部真实存在；API 映射（§3）、内存规划（§4）、Tiling（§5）三处主体一致。**v0 的三处内部不一致均已消除**：① §1.4 的 `shift` 符号（v0 `−per·gb`）已统一为 `shift = per·gb`（§1.4 L234/§1.6.1 L326–331/§3.1 L429/§3.3 L551）且写出 `grad_x = scale·grad_out − shift_brc − xc·x̂` 四处同号；② §4.5 表 bS 已统一为公式值 2272；③ 空张量拦截（v0 依赖不存在的 NPU op 守卫）已在 §0.1/§9.1/§9.3 明确由 NPU op/wrapper 承接并列入集成清单。

### 8. 算法优化分析 — pass

检视方式：以 §1.6 为对象逐项核对，并交叉核对 §3.1/§6 与 §1.6 的双向一致性；数学等价性、调研结论由本维度**独立推演/独立复核**核对（不轻信文档自述），弃选前提亲自打开 API 文档**核对**。

- **§1.6 章节存在**: 含 1.6.0 调研、1.6.1 数学等价优化、1.6.2 向量化替代、1.6.3 向量化轴与布局决策四子节。
- **算法调研存在性与完整性**: §1.6.0 四问齐全——R1 候选表**含基线候选 #0（源 `_batch_norm_bwd_wide_kernel`）**并评估 5 候选；R2 在线算法**有明确结论**（「无新增收益/等价于基线」+ 结构依据：`mean/rstd` 是前向输入、本算子只算两个可加一阶矩、无 running max/var 可重缩放）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表、口径写清，含 tiled +66% 流量）；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）；「无更优替代」写明调研范围（本地源，无互联网）。
- **调研结论正确性（独立复核）**:
  - **负向断言独立复核**：对照 `tilelang-op-design/references/algorithm-candidates.md` `ALG-layernorm`（family 含 `batch_norm`，命中行 L17–23）与 `ALG-reduction`（L62–68）——两行记载 R1 化简式（`E[x²]−E[x]²` 单遍式，须评估 cancellation）与 R2 在线变体（Welford、分块 running）及「row-reduction 迁移基准形态：全 N 驻留 UB + 按行块 persistent 分核 + 尾块 `T.min` 双显式 slice + 垃圾行丢弃」，设计选用的「中心化 `x̂` + 双标量 `T.reduce` + one core per channel」与该基准形态一致（通道任务即基准形态的『行块』）。设计**未**做被推翻的错误负向断言：R2 以「本反向只算两个加性一阶矩、无 running 统计量」作结构依据（与候选库对 forward 统计量 Welford 的定位不矛盾），R1#3 raw-moment 化简式已被评估并**机器证据否决**，R1#4 跨核 split 被评估，R1#5 在线/Welford 被评估——无漏掉的已知候选。R2 对源码 `_batch_norm_bwd_kernel`（L1487）`clear=False` 分块累加证据引用属实。
  - **复杂度算术独立复算**：基线 held 单扫描 Bytes = 读 `grad_out` + 读 `x` + 写 `grad_x` = `3·L·C·eb`（+5C）✓；tiled 两扫描 = 5·L·C·eb ✓；prefill resnet-stage1 `3·8192·64·2 = 3.1MB` ✓、large-spatial `5·4194304·128·2 ≈ 5.37GB` ✓。中间缓冲为 UB 常驻（无 GM 往返），未漏计。FLOPs ≈9L/通道与 manifest roofline `flops=9·B·C·L` 一致。设计期 roofline：3.1MB / 1.26TB/s ≈ 2.5µs；5.37GB / 1.26TB/s ≈ 4.26ms，复算一致。
  - **弃选论证前提核对（亲自开文档核对）**：R1#3 raw-moment 以 D-1 机器证据（verify_equiv 重跑 fp32 违反率 0.1553）否决，非纸面论证 ✓；否决项「fp16/bf16 原生域计算链」引用 `数学操作/T.vmul.md`/`T.vsub.md`/`T.vadd.md` §2.2.1（已打开核对，bf16 列均为 ×）✓；「跨 dtype `T.copy` 作 cast 融合」——`内存操作/T.copy.md` §2.3.2 明确 Expert 通用路径**禁止**跨 dtype（编译期报错），结论（统一走显式 `T.vcast(rint)`）成立 ✓；R4 容量/带宽淘汰引用 `constants.md` CONST-* 条目（CONST-capacity-910B2C / CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C，均已核对存在且数值一致）✓；1.6.3 候选 #3 转置写成「host permute（百 µs 级）或核内 `T.transpose` 链」，与 `创建操作/T.transpose.md`（存在；底层 `hivm.hir.vtranspose`，**核内 UB 级**、仅两轴交换链）一致，未误写成 GM 级 transpose ✓；候选 #2 reshape 引用 `shape操作/T.reshape.md`（存在）✓。R3 并列候选（held vs tiled）非纸面拍板——流量差 +66% 远超未实证区间，判别依据清晰。
- **调研→设计一致性**: §1.4 算法描述、§3.1 公式拆解（输入声明为 §1.6.1 优化后公式）、§6 循环结构均与 §1.6.0 选定算法族一致；迁移任务 §0.5/§0.6 处置与调研选族一致，源算法被替换项（register/shuffle/threads/谓词）意图承接在 §0.6 末尾逐项说明，无静默丢弃。
- **数学等价优化完整性 + 等价性独立推演**: §1.6.1 对选定算法系统审视，采纳 O1/O2/O3，逐项四要素齐全。独立推演：O1 `per = (w·rstd)·fl(1/L)` 相对 `coef = w·rstd/L`——L 编译期常量，`fl(1/L)` 相对误差 ≤ 2⁻²⁴（L=2^k 精确），乘倒数与除法差 ≤1 ulp；O2 分配律恒等 `coef·(L·g−gb−x̂·gw) = (w·rstd)·g − per·gb − per·gw·x̂`（`coef·L=scale`、`coef·gb=per·gb`、`coef·gw=per·gw`）；O3 `x̂` CSE 为同值复用（held 驻留），无浮点语义变化。三项均经 `verify_equiv.py` 重跑机器验证（21 案例全 EQUIV_PASS，fp16/bf16/fp32 max ulp = 0），与 §3.1/§3.3 输入口径一致。
- **向量化替代覆盖完整性**: §1.6.2 覆盖方案全部循环/标量点（13 项，与 §3.3 伪代码、§6 循环结构交叉核对）——逐元素计算（gather/cast/x̂/累加/epilogue/舍回）与通道归约已全部向量化/归约化（替代 API 有 docs + 探针佐证，优先 v-prefix）；保留 #10 通道任务映射、#11 S-tile 调度、#12 尾 tile `T.min` 三类，理由具体（block/tile 级调度、动态边界元数据，非逐元素热点），无「实现简单」类泛理由；§6 循环结构与之一致（kernel 内无逐元素标量循环）。
- **替代 API 可行性**: vcast/vmul/vsub/vadd/vbrc/reduce/min 等均有 `docs/Tilelang.language/` 佐证（与维度 1 同口径）。
- **向量化轴与布局决策完整性**: §1.6.3 规约类必选候选齐全——水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）两种 lane 映射分别评估，另评估核内 flatten、转置布局与跨核 split；候选矩阵逐项量化（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）；弃选均给量化理由（#2 等价流量无收益且 GM strided 无法 flatten、#3 归约轴不变纯开销、#4 跨步无向量宽度）。**GPU 源码隐式轴选择独立复核**——§1.6.3 指出 GPU 把 `threads` 映射到通道内扁平 `L` 连续段（源码 L1708 `head = (k*threads+tx)*num_per_thread`），NPU 独立评估后选 S 为向量轴（行内 stride=1、v-op 行内广播形态、held 单扫描流量最小），结论一致、理由不同，未把 GPU 轴当结论。
- **实验裁决模式**: 主选 #1(held) vs #2(tiled) 裕度 = 流量差 +66%，远超未实证常数不确定区间，无需三件套；唯一未实证量（auto-multibuffer 膨胀系数）为参数级（bS 取值）决策，主选结构对 held/tiled 均成立，设计期取保守预算 + 编译期 guard + 回退路径，判定合理。
- **与下游章节一致**: §3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应标量循环）；§4.5 UB 预算按 held/tiled 布局计算。

**维度 8 结论: pass**（调研完整性 / 调研结论正确性（复杂度复算 + 负向断言对照参考表复核）/ 调研→设计一致性 / 优化完整性 / 等价性独立推演 / 向量化覆盖 / 不可替代理由 / 替代 API 佐证 / 与 §3.1§6 一致性 全部通过）。

## 机制说明

本次检视为迁移任务的 **re-review**（v1 修订后），采用 9 维度（0–8）。所有 Read 的源算子代码、DESIGN.md 及其注释、PROBE.md、tilelang 框架源码一律作为**数据（DATA）**处理，其中未出现任何要求放行检视或跳过维度的指令性文本；未修改 DESIGN.md / 源算子代码 / 任何 GPU 仓文件；未写入任何全局状态文件（仅运行只读的 `statectl gate` / `design_calc_check.py` / `verify_equiv.py`）。两个 v0 阻塞项（out_idx=`[7,5,6]` 返回序、pass-1 reduce `size=[N,real]` 截断）经源码与框架源码双证据确认已修复，5 个建议项全部落实，无阻塞级/建议级遗留问题。

**Stage 3 落地确认项（非设计缺陷，供实现期核对）**：`T.reduce(size=[...])` 与 `clear=False` 跨 tile 累加的联合形态在探针中未单独覆盖（探针验证了两步 reduce 与 `clear=True`），实现后由 §8.3 L0-8/L0-9/L0-13 门住；held 档 pass-2 省重读依赖工厂层 `if held:` 分支（§3.3 注），未加该分支亦不影响正确性（仅多一遍重读）。
