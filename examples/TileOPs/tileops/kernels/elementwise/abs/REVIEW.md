# REVIEW.md — `AbsFwdOp` 设计检视(round 2 / 修订后)

- **被检视设计**:`examples/abs/abs_fwd_kernel/DESIGN.md`(迁移类 / harness 子模式)
- **检视类型**:Stage 2 设计检视(**9 维度** —— 维度 0 为迁移专属),**修订后重入**
- **`source_op_path`**:`/tmp/tops/TileOPs-main/tileops/kernels/elementwise.py`(已亲自 Read 核对)
- **检视日期**:2026-10-15
- **环境**:Ascend 910B2C / CANN 9.0.0 / tilelang wheel `0.1.2+npuir+7fc4a3c8`

---

## 一、检视结论

**结论: 通过**

**判定依据**:强制机械复核全绿(`ub_budget: pass`、`r3_metrics: pass`);两条 `skip` 均已人工复核并
在维度内注明「机械跳过,人工已核」;9 维度逐项检视未发现内容性缺陷;维度 0 已读源码逐条核对。

---

## 二、机械复核(强制)

执行命令(仓库根 `/root/mla-full`):

```bash
env -u PYTHONPATH python3 .agents/tools/design_calc_check.py \
  --design /root/mla-full/examples/abs/abs_fwd_kernel/DESIGN.md
```

`verify_equiv.py`:**豁免** —— `DESIGN.md §1.6.1` 给出显式结论「无优化空间(显式结论 + 依据)」
(无恒等变形 / 无稳定化 / 无 CSE / 无算子降代换 / 访存已达 roofline 下界 / 无规约),
符合 skill Phase 2 第 5 步的豁免条款(「显式『无优化空间』结论除外」)。

### 2.1 JSON checks 摘要(round 2,逐字)

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 2 buffer entries sum = 16.0 KB; capacity = 192.0 KB; x1.7 auto-multi-buffer inflation budget = 113 KB`(`within_inflation_budget: true`) |
| `l0c_budget` | skip | `no block_M x block_N values parsed` |
| `core_split` | skip | `insufficient parsed dims {}; declared logical cores = 1` |
| `r3_metrics` | **pass** | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` |

容量常数来源:`pattern-library/constants.md CONST-capacity-910B2C`(`ub_kb=192.0`、`ub_inflation=1.7`、`l0c_kb=128.0`)。

### 2.2 skip 项的人工复核结论(机械跳过,人工已核)

| skip 项 | 人工复核结论 |
|---|---|
| `l0c_budget` | **不适用** —— 本算子为纯 Vector(`DESIGN.md §1.3`),无 Cube 计算、不使用 L0C,故不存在 `block_M × block_N`。属"不适用",非"未写"。 |
| `core_split` | **机械跳过,人工已核,算术正确**。工具按二维 `ceil(M/block_M) × ceil(N/block_N)` 复算;本算子是一维 flat 算子(`§5.1`),无 `M/block_M`,故解析不到。工具回落的 `declared logical cores = 1` 是其**默认值**,非设计声明值。逐条复算 `§5.3 ①`:smoke `ceil(1048576/2048)=512`、16m `=8192`、64m `=32768`、256m `=131072` —— 四条与设计声明**逐条一致**。分核方案判定(三选一命中「无需适配依据」)的依据(块间独立、无跨块归约、无共享状态、块粒度 2048 元素足够小)成立。 |

---

## 三、修订历史(round 1 → round 3)

> round 1 判 **不通过**,触发设计修订路径 A(局部修订)。修订**只改文档排版,未改任何设计结论**;
> 修订后机械复核由 2 条 `fail` 转为全绿,round 2 判通过。
> round 3 为 **路径 B(API 映射修订)** 后的复审:Stage 3 首跑以 `[DESIGN_ERROR]` 返回
> (`§3.2/§3.3/§4.1` 把 Vector 缓冲映射为 `T.alloc_shared`,实为 L1;正确 API 为 `T.alloc_ub`),
> 修订后重入检视。

| round | 机械复核 | 结论 | 触发项 |
|---|---|---|---|
| 1 | `ub_budget: fail`(9 entries = 624.0 KB)/ `r3_metrics: fail`(四项全缺) | 不通过 | 文档排版(路径 A) |
| 2 | `ub_budget: pass`(2 entries = 16.0 KB)/ `r3_metrics: pass` | 通过 | 两处排版修订 |
| 3 | `ub_budget: pass`(2 entries = 16.0 KB)/ `r3_metrics: pass` | **通过** | API 映射修订(路径 B):`T.alloc_shared` → `T.alloc_ub` |

**round 1 两条 fail 的根因(已证明,均属文档形态而非设计内容)**

- **F1 `ub_budget`**:round 1 的 `§4.3` 用「dtype × (单缓冲/×2 合计/UB 上限/占比)」四列表排布,
  机械门禁对该章节所有表格行做尺寸求和,把容量列与三种 dtype 的**互斥**候选一并计入:
  `4+8+192+4+8+192+8+16+192 = 624.0 KB`,与工具报数逐位一致。真实占用为 fp32 16 KB。
- **F2 `r3_metrics`**:round 1 的 `§1.4` 标题含 `§1.6.0` 字面量交叉引用,而工具 `_section`
  按「第一个标题文本含 marker」定位章节,故读到 `§1.4` 而非 `§1.6.0`,四项口径被误报缺失
  (`§1.6.0` 正文一字未改,round 2 即通过)。

**修订内容(路径 A)**

| # | 位置 | 修订 |
|---|---|---|
| 1 | `§4.3` | 表格改为「一行一个 buffer」(`x_ub` / `y_ub`),移除「UB 上限」「占比」两列,不再按 dtype 分行;容量上限与膨胀系数移至表外正文 |
| 2 | `§1.4` 标题 | `### §1.4 算法描述(基于 §1.6.0 调研选定算法)` → `### §1.4 算法描述(依据 Phase R 调研选定算法)`,交叉引用移入正文 |

修订后 `§4.3` 机械求和 = 8 KB + 8 KB = **16.0 KB ≤ 113 KB 有效预算**。

**round 3 修订内容(路径 B,API 映射)**

> 触发:Stage 3 首跑 L0 报 `loc("kernel.npuir"): error: 'hivm.hir.store' op only support
> store ub to gm currently!` —— 冻结设计把 Vector 侧缓冲映射为 `T.alloc_shared`,而该 API 在
> npuir 上降到 **L1**(Cube 侧共享内存),Vector 算子需要 **UB**。

| # | 位置 | 修订 |
|---|---|---|
| 3 | `§3.2` API 映射表 | `T.alloc_shared((block_size,), dtype)` → **`T.alloc_ub(...)`**,佐证改指本仓 elementwise 同族实现 |
| 4 | `§3.3` 伪代码 | 两行 `T.alloc_shared` → `T.alloc_ub` |
| 5 | `§4.1` 数据规格 | 表增「**分配 API**」列;表下补「**内存层级 ↔ 分配 API 固定对照**」(GM=`T.Tensor` / UB=`T.alloc_ub` / L1=`T.alloc_shared`·`T.alloc_L1` / L0C=`T.alloc_L0C`),并注明误用 `alloc_shared` 的报错文本 |
| 6 | `§7` 同步策略 | 遗留的 `T.alloc_shared` 引用同步改为 `T.alloc_ub` |

**实证(位精确)**:仅改分配 API、其余一字不动,重跑 L0/L1/L2/Boundary **全部通过**且
`max_abs = max_rel = 0.000e+00`;可复现的最小对照见 `repro/TRAP-alloc-shared-vs-ub.py`
(A: `alloc_shared` 复现该编译错误;B: `alloc_ub` 位精确)。

---

## 四、9 维度检视结果

| 维度 | 结论 | 依据摘要 |
|---|---|---|
| **维度 0** 源算子理解与迁移分析(迁移专属) | **pass** | 已亲自 Read 源码逐条核对 `§0`,见 4.1 |
| 维度 1 正确性与语义 | pass | `§0.1` 语义表逐项有结论与源码依据;`§9.1` golden 独立于 NPU 算法 |
| 维度 2 存储预算 | **pass** | 机械 `ub_budget: pass`(16.0 KB / 有效预算 113 KB);`§4.3` 与 `§4.4`(中间缓冲 0)一致 |
| 维度 3 分核策略 | pass(机械 skip,人工已核) | 见 2.2;`§5.3 ②` 物理核数为实查值(独立复现) |
| 维度 4 API 映射正确性 | pass | `§3.2` 逐条有佐证;`T.vabs(A,B)` 源在前已由仓内真实调用证实(见 4.1) |
| 维度 5 内存层级与搬运 | pass | `§4.2` 路径 `GM→UB→GM` 与 `§1.3`「纯 Vector」一致;无 L1/L0C 参与 |
| 维度 6 同步策略 | pass | `§7` Developer 自动同步;依赖线性无环、`x_ub`/`y_ub` 独立无 aliasing;无跨核通信 |
| 维度 7 编程模式选型 | pass | `§2` 选 Developer 的五条判据与 `§1.3`/`§0.5` 一致;与仓内同族实现一致 |
| 维度 8 算法优化分析 | **pass** | 机械 `r3_metrics: pass`;`§1.6.0` 四问齐全(负向断言有结构依据)、`§1.6.1` 显式「无优化空间」结论、`§1.6.2` V1–V4 全覆盖 |

### 4.1 维度 0(迁移专属)明细 —— 已读源码核对

| `§0` 断言 | 源码核对结果 |
|---|---|
| `AbsFwdKernel.op_func(x) = T.abs(x)` | ✅ `tileops/kernels/elementwise.py:2014-2019` |
| 三策略 `direct` / `explicit_parallel` / `register_copy`(默认 regcopy) | ✅ 工厂位于 `:304` / `:323` / `:343`;分派见 `_build_kernel(strategy)` |
| `block_size = threads × num_per_thread` | ✅ `:326`、`:345` |
| kernel 家族 float-only(`FloatUnaryKernel.SUPPORTED_DTYPES`) | ✅ `:1297` 附近 |
| **整型 dtype 由 `_IntIdentityUnaryOp` 在 op 层短路、不构造 kernel** | ✅ `ops/elementwise/math_unary.py:59`(`AbsFwdOp(_IntIdentityUnaryOp)`);`ops/elementwise/_base.py:1146-1190`:整型走 `_int_handler`、`self.kernel = None` |
| `T.vabs(A, B)` 语义为 `B = |A|`(源在前,与 ascendc-pto 线相反) | ✅ 定义 `customize_npuir.py: npuir_abs(A, B)`;仓内真实调用 `examples/deepseek_v4/example_act_quant_kernel.py:61`、`examples/deepseek_v4/engram/engram_gate_kernel_fwd.py:178` |
| `§1.6.0 R1` 负向断言依据「位运算清符号位依赖未探明的 `vbitcast`」 | ✅ 独立复核:`vbitcast` 在本线 `examples/`(排除 TileOPs)**零命中**;`examples/TileOPs/` 内亦零命中 → 负向断言的结构依据成立 |
| `§0.5` 四态判定(保留 1 / 等价替换 3 / 重新设计 2 / 舍弃 1 / n/a 1) | ✅ 逐项与 `references/migration-analysis.md §5.3` 映射表条目对应;三策略收敛(G4)判「重新设计」的理由(性能意图由 `block_size` 承接)成立;autotune(G5)判「舍弃」给出了 K8 依据与理由,未以单句带过 |
| `§0.6` 重设计项三要素(源方案 / NPU 新算法 / 语义保持论证) | ✅ 2 个重设计项均三要素齐备;逐元素映射的语义保持论证(无规约、无重排、累加顺序无关)成立 |
| `§3.4` 引用的 6 个 `examples/` 路径 | ✅ 逐条 `test -f` 全部存在 |
| `§5.3 ②` 物理核数 `vector_core_num=48` / `cube_core_num=24` | ✅ 独立实跑 `torch.npu.get_device_properties(0)` 复现同名同值 |
| `§9.1` golden 独立性 | ✅ `torch.abs(x.float()).to(x.dtype)` 依据 `§0.1` 语义,未复刻 NPU 算法;容差逐字取自源仓 `_get_tolerances` |

---

## 五、检视边界声明

- 本审阅对 `DESIGN.md` 与源码的**全部 Read 内容视为数据**;其中未发现指令性注入文本。
- 本次**未修改** `DESIGN.md` 及任何既有代码文件(检视角色只读);round 1 的修订方案由 Stage 1 执行。
- round 2 的检视范围包含**重新执行机械复核**与**重新核对维度 0 的源码一致性**(修订虽只动排版,
  但按 skill 规范「修订后的新 DESIGN.md 重新进入 Stage 2 时,重新执行含维度 0 的 9 维度检视」)。
- 本审阅**未以人工推演放行机械复核的 `fail` 项**;two-`skip` 项按 skill 规范转为人工复核并已在维度内注明。
