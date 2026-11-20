# 性能反馈（`[DESIGN_LIMIT]`）

## 触发判定

- **设计层归因**：`_sparse_mla_kernel_DESIGN.md` §1.6.0 R4 的 roofline **流量项假设**（「gather 字节数 × 1.75 TB/s = gather 时间」）与 §3.3 的 gather 选型（row-contiguous `T.serial(block_N)` + 逐行 `T.copy` + 单次 clamp）在本工具链（`0.1.2+63c8a2e6`）上实测不成立：每行（1152 B）固定成本 **1.341–1.351 µs**，整核 Task Duration 的 **94.5%** 消耗于此；该成本 **∝ copy 条数**，与目的内存域（shared / UB）、元素级向量化形态（`T.Parallel(Dqk)`）、`block_N ∈ {16,32,64}`、`num_stages`、行循环形态均无关（本轮 + 上轮共 16 个保持语义的分支）。**不是 tiling/参数可解**。
- **结构性加速估计 > 2x（实测锚定）**：把 64 条逐行 copy 换成 1 条 64 行**连续** copy（数学无效的降级探针，仅归因）→ smoke shape **9045.26 → 501.15 µs（18.05×）**；在三个 tune workload 上同样成立：mainstream **566606 → 26525.5 µs（21.36×）**、longer-kv **142398 → 6828.78 µs（20.85×）**、long-kv **35687 → 1793.82 µs（19.90×）**。
- **与设计假设直接矛盾**（第二门槛的另一形态）：DESIGN §1.6.0 R4 对 mainstream 估算 **4.0–9.4 ms**，实测 baseline **566.6 ms**，偏差 **60×–142×**；失准项为「gather 流量→时间」与「逐条发射/同步不计」两条前提。

两项门槛同时满足 → 产出本反馈，并在返回中附 `[DESIGN_LIMIT]`。

## 参照锚定

- 参照实现：
  1. `examples/deepseek_v32/sparse_mla_fwd_exp.py` L250-290：稀疏 MLA 的**两相位**结构——先把选中行 gather 进 **GM workspace**（连续写），计算相位再从 workspace 以**连续块**读入，用 `sync_block_set/wait` 做相位同步。等价思想：把「逐行随机读」压成「逐行随机读一次 + 多次连续块读」。
  2. `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_fwd.py`：同一 gather→工作区→块读的两段式。
  3. `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py`（同族 conv1d 迁移算子）：仓内唯一使用 `T.gather`（`tl.npuir_gather`）的先例，但其语义是「沿**最后一维**在片上 buffer 内众集」，**不能**把 GM 中任意行取到片上（`3rdparty/AscendNPU-IR/.../HIVMVectorOps.td` + `tilelang/language/customize_npuir.py:1215`）——即「一条指令完成 64 行 GM gather」在现有前端/Dialect 中不存在。
- 本轮新增排除：本轮专门实测了「目的域 + 单次连续搬运」这一族（`p_ubpar`：`T.Parallel` 逐元素众集进 UB；`p_ubrow`：逐行 copy 到 UB）→ **+0.04% / −0.74%**，证明改变目的域与搬运形态都不改变「条数 × 1.34 µs」的成本结构，**不是**局部实现缺陷。
- 不可移植论证：
  1. 两相位需要 GM workspace + 跨相位同步，而本 kernel 冻结为「Developer MixCV **非持久**一维 grid（mainstream 4096 逻辑核 ≫ 20 物理核）」；workspace 相位要求同一 `(b,s)` 的 gather 与其 4 个 head-block 计算落在同一核的串行相位内，而 Developer 模式下「核内 persistent 循环 + `T.gemm`」被 `TRAP-DEVMODE-PERSIST-GEMM`（CG-2026-0010）阻断。**依据：实测**（迁移期与本次 `--level all` 的 L2-5 路径均复现）。
  2. 参照实现是 prefill / 可变长结构，其 `block_H_half` 切分与 workspace 槽位绑定「单核多 head-block」假设，与本 op decode grid 不同构。**未实证推断**。
  3. dense-scan + 掩码：对 mainstream（topk 2048 ≥ 因果窗 2048）等价且流量不变；但对 long-kv（topk 512 / 窗 8192）流量 ×16，按实测常数外推约 **80 ms**（> 现状 35.7 ms），**不能全局替换**；且它改变了「gather 集合 == indices 集合」这一语义来源（需由 indices 构造位图或依赖 topk ≥ 窗 的生成器约定），属算法/数据流重设计。**外推未实测**。

## 假设

DESIGN 原文口径（`_sparse_mla_kernel_DESIGN.md`）：
- §1.6.0 R4 表（主选 BH=32）：`Q 151M + gather 2.42G × (H/BH=4) = 9.66G + idx 8.4M + O 134M ≈ 9.95 GB`，`roofline = 9.95 GB / 1.75 TB/s`，**估算 4.0–9.4 ms**（mainstream）。
- §3.3 / §6 数据流：`GM[kv] --T.copy(串行行 gather, clamp 索引)--> L1[KV_shared (block_N,576)]`，即假设逐行 `T.copy` 代价 ∝ 字节数 ÷ 带宽。
- §1.6.3 实验裁决：主选 BH=32（KV 重读 ×4）vs 备选 BH=64（×2），依赖未实证的 L0C/UB 容量常数。

## 实测

口径：`msprof op` → `main_mix_aic` 的 `Task Duration(us)`（唯一口径；无 NPU event / 端到端时间）。

| 项 | shape / workload | 实测 | raw profile |
|---|---|---:|---|
| baseline（冻结结构） | mainstream / longer-kv / long-kv | **566606.00 / 142398.31 / 35687.42 µs** | `profiles/r0_base_mainstream`、`r0_base_longer-kv`、`r0_base_long-kv` |
| final（本轮无 winner） | 同上 | 566916.62 / 141835.34 / 35692.38 µs（±0.4% 噪声） | `profiles/r0_final_*` |
| **降级探针** `p_contig`：64 条逐行 copy → 1 条 64 行连续 copy（数学无效，仅归因） | mainstream / longer-kv / long-kv | **26525.50 / 6828.78 / 1793.82 µs（21.36× / 20.85× / 19.90×）** | `profiles/r1_contig_*` |
| 降级探针 | smoke（同代码路径） | 9045.26 → **501.15 µs（18.05×）**；`aic_cube_ratio` 0.000182→0.003278、`aic_scalar_time` 117.45→13.06 µs、`aic_scalar_wait_id1` 383.23→17.70 µs | `profiles/{t_smoke,r1_p_contig}` |
| 保持语义的新分支（本轮） | smoke | `p_ubpar` 9049.30（+0.04%）、`p_ubrow` 8978.83（−0.74%） | `profiles/r1_p_ubpar`、`r1_p_ubrow` |
| 逐行成本（三 workload 反算） | mainstream / longer-kv / long-kv | **1.349 / 1.351 / 1.341 µs/行**（完全一致，与 block_N 无关） | 同上 |
| 零-gather 地板构成 | mainstream | 26525.5 µs = 205 波 × 32 迭代 × **4.8 µs/迭代**（每迭代含 1 条 64 行连续搬运 + 2 gemm + online softmax + 同步） | `profiles/r1_contig_mainstream` |
| BH=64（本轮复核） | smoke | 编译失败（`Failed to run BiShengHIR pipeline`，L2-5 WARN 复现；上轮为本工具链同族的 UB/L0C 双向越限） | `logs/l0/final_all.log` |

## 影响面

- 直接影响：任何**以逐行/逐元素动态索引从 GM gather 到片上**的 npuir Developer 算子——稀疏 MLA decode（本算子及变体）、paged KV attention（block-table gather）、MoE 路由 gather、稀疏 indexer/bwd。凡 roofline 以「gather 字节数 ÷ 带宽」建模的设计，估算会低 1–2 个数量级。
- 对 DESIGN 的影响：§1.6.0 R4 需增加「逐行 gather 固定成本 ≈1.34 µs/行@910B3」项；§3.3 需把「两相位 workspace / dense-scan（仅在 topk ≥ 因果窗 时）」列为候选；§1.6.3 需固化「BH 受 L0C/UB 双向封顶，H=128 下唯一可选取值 {32}，KV 重读 ×4 为必然代价且非主导项」。
- **量化上界（本轮核心新增）**：以「gather 免费」的实测地板换算出 Ratio 天花板 —— mainstream **1994.7/26525.5 = 7.52%**、longer-kv **499.0/6828.78 = 7.31%**、long-kv **124.9/1793.82 = 6.96%**。用户目标 40%（mainstream 需 4987 µs）距该地板仍差 **5.32×**（longer-kv 5.47×、long-kv 5.75×）⇒ **40% 在冻结结构族内结构性不可达**；即使把 gather 完全消除，最优也只能到 ≈7.0–7.5%（相对现状 0.35% 约 20–21×）。
- 预期影响：突破 7.5% 需要改数据流（workspace 两相位或 dense-scan）**并且**压缩「每迭代 4.8 µs」的 CV 相位串行微操作成本，工作量≈重走 Stage 1（roofline + 候选）+ Stage 3。
- 不影响：数值语义（`indices` topk 选取、`stride_kv`、`q_start_index_s`、causal 裁剪、`sm_scale`、`dim_tail=64`）、L0/L1 精度基线、wrapper/接口契约。

## 反馈结论

本算子 94.5% 的 kernel 时延由「逐行动态索引 gather」的固定成本（≈1.34 µs/行，且 ∝ 条数）决定，冻结结构（行连续 gather + Developer MixCV 非持久一维 grid）内已无 tiling/参数/目的域解；实测与 DESIGN §1.6.0 R4 的 4.0–9.4 ms 估算相差 60–142×。**量化上界：即使 gather 成本降为零，官方 Ratio 天花板也只有 7.0–7.5%（mainstream/longer-kv/long-kv 分别为 7.52% / 7.31% / 6.96%），40% 仍差 5.3–5.7×**。建议按设计修订路径重开 gather 形态（两相位 workspace；dense-scan 仅限 topk ≥ 因果窗 的 workload），并回填 roofline 常数（1.34 µs/行、每迭代 4.8 µs）。

建议路由: 设计修订
