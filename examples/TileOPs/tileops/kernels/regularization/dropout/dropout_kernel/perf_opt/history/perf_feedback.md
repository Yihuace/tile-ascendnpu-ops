# 性能反馈（[DESIGN_LIMIT]）

> 任务: `dropout-_make_dropout_kernel-20261016T193233Z`（scenario=optimize, mode=full）
> 算子: `DropoutOp` / `_make_dropout_kernel`（`examples/TileOPs/tileops/kernels/regularization/dropout/dropout_kernel/`）
> 设备/工具链: Ascend910B3（20 AI core × 2 = 40 Vector core，UB 192 KB）/ CANN 9.0.0 / `tilelang 0.1.2+90fb7ae37f0be5e908d6317eb2b1e09aeb1ddc4b` / 2026-10-17

## 触发判定

- **设计层归因**: `_make_dropout_kernel_DESIGN.md` 的具体假设——
  - §1.6.0 结论：「选定 A5 核内无状态 counter-based hash … **无更优可行替代**」；「PRNG 的 ~9 op（§1.6.0 固有落差）」；D-2 roofline「发射项 = op 数 × 单 op 边际代价（9.15 µs/16m）= 16m fp16 估算下界 ≈192 µs，实测 366 µs，**这是本迁移的固有性能落差**」。
  - §1.6.1 结论：「**公式层面已无进一步计算量/访存量可优化空间**：… PRNG 的 ~9 op（§1.6.0 固有落差，无更优可行替代）」。
  - §9 R-1「发射/向量吞吐主导…固有性能落差」。
  - 剩余下界**不可由 tiling/参数解开**：`block_size` 已在 BiShengIR UB 上限（8192 ok / 10240 `ub overflow, requires 1802240 bits`），掩码段已是免 `vselect` 的最廉价整数链，op 数在「有软件 PRNG」的前提下已压到 12–13。下一档提升要求**前端可达的随机数原语**，即能力缺口 `CG-2026-0021`（`.agents/evolution/capability-gaps.md`；`dir(T)` 无 rng/rand 属性、`docs/Tilelang.language/` 全子目录无 RNG 条目、`tilelang/language/customize_npuir.py` 无 RNG 算子、BishengIR 仅 `core_named_ops.py` 的 `fill_rng_2d` 且前端不可达）。这是编译器/前端能力项，不是 tiling 参数。
- **实测与设计假设直接矛盾**（同口径 msprof op，`perf_records.jsonl`）:
  1. 「~9 op 无更优可行替代」被推翻：同一算法族内 **5 op** 的 counter hash（`c=(idx+seed); c*A; c^=(c<<13); c*B`）在 4–8 Mi 元素上 rate 偏差 < 3σ、block-χ²(dof 4095) = 3812–3946（Stage 3 的 9-op 链为 3974–4080，同档）⇒ 统计契约等价，而 TaskDur 降 **30.3%**（W1 57.20→39.85 µs）。
  2. 「已无进一步计算量可优化空间」被推翻：fp16 链由 20 op 压到 **13 op**（结构实测：seed 折入 counter −19.5%、fp16 存储域去 fp32 往返 −4.5%），端到端 W1 65.11→27.91 µs。
  3. §1.6.0 D-2 估算系统性失准：W1 估算 ≈173 µs vs 实测 baseline **65.11 µs（设计高估 2.66x）**；W2 ≈293 vs 158.36（1.85x）；W3 ≈180 vs 77.11（2.33x）。失准项为「发射项」——它引用 910_9362 绑定的 9.15 µs/16m-op 与 1.06 TB/s 口径，而本设备是 910B3；且线性外推未计入 `block_size` 摊薄（bs2048→8192 使 W1 降 56%，op 数不变）。
- **结构性加速估计**: **2.5–3.0x**（依据：同口径实测外推）。最终候选 W1 的 Vector 时间 ≈ 0.9612 × 27.913 µs ≈ 26.8 µs，对应 13 个向量 op（vec-bound，`aiv_mte2_ratio` 仅 0.20、`aiv_mte3_ratio` 0.09，DMA 完全被掩盖）；若前端 RNG 原语可用，13 op → 4 op（掩码转存储 dtype + `scale·mask` + `x·(scale·mask)`，PRNG/hash/threshold 段全部消除）⇒ Vector 段 ≈ 26.8 × 4/13 ≈ 8.2 µs，TaskDur ≈ max(流量地板 6.7 µs, 8.2 µs) ≈ **9–11 µs**，即 **2.5–3.0x**（> 2x 门槛）。

## 参照锚定

- **参照实现**（同族「Developer 模式一维 persistent 逐元素、1 读 1 写」的已知最优）:
  - `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/perf_opt/`（同机 910B3，16m fp16 27.96 µs = 2.40 TB/s / 16m fp32 52.45 µs = 2.56 TB/s）
  - `examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/perf_opt/`（16m fp16 25.44 µs = 2.64 TB/s / fp32 49.80 µs = 2.69 TB/s）
  - `examples/TileOPs/tileops/kernels/elementwise/elu/elu_kernel/perf_opt/`（copy-only 地板 16m fp16 26.98 µs = 2.49 TB/s）
  - 模式库条目与常数：`.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md`（`PL-1.22-persistent-1d-grid-stride` 模板）、`.../constants.md`（`CONST-elu-910B3-bw-floor`，verified：910B3 逐元素实用天花板 **2.49–2.67 TB/s**）
  - 本族已有档案：`.../pattern-library/cases.md` 的 `CASE-elementwise-dropout-rng-migration`（Stage 3 迁移档案）
- **不可移植论证**（逐条结构差异 + 依据）:
  1. **参照 kernel 无随机性**：leaky_relu/hardtanh/elu 的向量链全是数据独立算术（2–7 op：`vmul+vmax`、`vclamp`、`vexp+vsub+vselect`），**没有 counter→统计量的映射**。依据：上述 `perf_opt/` 内 kernel 体（实测文件）。
  2. **dropout 必须在核内逐元素生成独立掩码**：host 端物化掩码被 `DESIGN.md` §0.5/`ascend-constraints.md` §4（host 禁核心计算）否决，核外生成又会把流量抬到 3R:1W 以上。依据：`_make_dropout_kernel_DESIGN.md` §0.5 表 + §1.6.0 R1-A1 淘汰记录。
  3. **无前端可达的 RNG 原语**：`T.rng_init`/`T.rng_rand_float` 不存在（`CG-2026-0021`，四条独立复核证据见 DESIGN.md §1.6.0 R1 表）。因此那 8–9 个 op 的 PRNG+阈值链是**软件替代硬件 RNG** 的必然代价——参照 kernel 没有这一项，其 2.49–2.67 TB/s 的达成前提（op 数 2–7）在本算子结构上不可满足。
  4. **参照的搬运地板不可达**：即便在 DMA 完全隐藏的有利条件下（本算子 bs8192 时 `mte2_ratio` 仅 0.16–0.21），有效 R+W 仍只有 0.57–1.18 TB/s，说明瓶颈完全在 Vector 段而非搬运段；把参照的「静态整块拷贝」（`PL-1.34`）移植过来实测**打平 ≤0.5% 且抬高 UB**（本任务 Round 2 `c1_sbulk`/`u7_sb_diet` 行）——该模式的前提是 DMA 受限，不成立。
  5. 未找到任何「在 npuir 上以更少 op 生成逐元素随机掩码」的同族实现（本仓 `examples/` 无 RNG 算子先例，DESIGN.md §1.6.0 调研范围已逐项列出）⇒ 天花板结论限定于「有软件 PRNG」的冻结结构族，不主张绝对地板。

## 假设

DESIGN.md 原文（`_make_dropout_kernel_DESIGN.md`）：

- §1.6.0：「**调研结论**：选定「A5 核内无状态 counter-based hash + B1 算术 0/1 掩码乘 + fp32 中转计算域」；**无更优可行替代**。」
- §1.6.0 D-2：「**设计期估算（D-2）结论**：代表性 `16m fp16` 实测 **366 µs**（估算下界 ≈ 192 µs，流量地板 63 µs）——**发射/向量吞吐主导，约为带宽地板的 5.8×**。根因：软件 PRNG（~19–21 op）取代了源硬件 RNG（≈0 op），这是本迁移的**固有性能落差**。」
- §1.6.1：「**公式层面已无进一步计算量/访存量可优化空间**：访存固定在理论下界 `2N·eb`…；唯一「额外」成本是 fp16/bf16 的 2 次 `vcast`（bf16 强制、fp16 为统一掩码域所需）与 PRNG 的 ~9 op（§1.6.0 固有落差，**无更优可行替代**）。」
- §9 R-1：「**R-8**：发射/向量吞吐主导（op 数 19–21）… 性能天花板由向量 op 吞吐决定」。

## 实测

口径：`msprof op --kernel-name=main --warm-up=5 --launch-count=20/30 --aic-metrics=Default`，`OpBasicInfo.csv::Task Duration(us)`（AIV 均值），Ascend910B3 40 Vector core，kernel-only。原始数据：`perf_opt/profiles/`（baseline/final/各轮分支）；结构化记录：`perf_opt/perf_records.jsonl`。

| 断言 | 设计值/假设 | 实测 | 证据 |
|---|---|---:|---|
| `[1024,4096]` fp16 下界 | ≈173 µs（§1.6.0 D-2） | baseline **65.106**，final **27.913** | `profiles/baseline/W1/…`、`profiles/final/tokens-1k-hidden-4k-fp16-float16/final_bs8192/…` |
| `[1024,10240]` bf16 下界 | ≈293 µs | baseline **158.356**，final **73.899** | `profiles/baseline/W2/…`、`profiles/final/tokens-1k-hidden-10k-bf16-bfloat16/…` |
| `[1024,4096]` fp32 下界 | ≈180 µs | baseline **77.110**，final **28.561** | `profiles/baseline/W3/…`、`profiles/final/tokens-1k-hidden-4k-fp32-float32/…` |
| hash 链 op 数 | 「~9 op，无更优替代」 | 5 op 链统计等价（rate <3σ、block-χ² 3812–3946 @ dof 4095）；TaskDur −30.3%（W1 57.20→39.845；W2 136.12→94.379；W3 54.45→37.186） | `perf_opt/_hash_analysis.py` 输出 + `perf_records.jsonl` 的 `hashC` 行 |
| 掩码段/计算域 | 「掩码段最廉价」「fp16 必需 fp32 中转」 | `vabs`+`vclamp` 取代 `vand`+`vmax`+`vmin`（−1 op、−8 B/elem）；fp16 在 scale 可精确表示时存储域计算与 fp32 往返**逐位一致**且 −4.5% | `perf_opt/logs/r4/native_equiv.log`（`bit_identical=True`、`kept_max_err_vs_x2=0.000e+00`）、`perf_records.jsonl` 的 `u5_diet`/`f16native` 行 |
| 最终 op 数 | 19–21（设计） | **13**（fp16）/ **12**（fp32） | `opt_log.md` §3 op 数表 |
| Vector 段占比（决定外推） | — | final W1 `aiv_vec_ratio=0.9612`、`aiv_mte2_ratio=0.1992`、`aiv_mte3_ratio=0.0867` | `profiles/final/tokens-1k-hidden-4k-fp16-float16/final_bs8192/OPPROF_*/main/*/PipeUtilization_*.csv` |
| tiling 已到上限 | — | fp16/bf16/fp32 均 bs8192 ok、bs10240 `ub overflow, requires 1802240 bits while 1572864 bits available` | `opt_log.md` §2 Autotune 段 + `profiles/` 采集日志 |

## 影响面

- **受影响的选择/结论**（都在本算子设计文档内，均可「追加式」更正，不需要改算法）:
  1. `§1.6.0` 的 A5 hash 具体形态（9 op → 5 op）与「无更优可行替代」的措辞；
  2. `§1.6.1` 的「已无进一步计算量可优化空间」与「fp16 必须 fp32 中转」（应改为「scale 可精确表示时存储域更优」）；
  3. `§1.6.0 D-2` roofline 的设备口径（910_9362 常数用于 910B3 设备）与发射项线性外推；
  4. `§5.2`(分核/参数) 的默认 `block_size`（1024/2048/2048 → 8192/8192/8192，UB 二分实测）。
- **不改动的部分**: 算法族（核内 counter-based 无状态 hash + 算术 0/1 掩码乘 + 无 host 物化掩码）与全部正确性/契约设计（确定性重放、drop-rate 判据、`vshr`/`vselect` 禁用、内存层级）在本次调优中**保持不变且验证通过**。
- **影响范围**: 同族（regularization / 随机掩码类）后续算子的设计期 roofline 与「软件 PRNG op 预算」估计；`examples/` 内其他 910B3 逐元素算子若引用 910_9362 的 9.15 µs/16m-op 常数，其发射项同样需要重标定。
- **改设计预期影响的 Stage 与工作量**: 若走「设计修订」，需重跑 Stage 1（DESIGN 的 §1.6 调研结论与 D-2 roofline 更正）+ Stage 2 检视 + Stage 3 实现核对；若走「附录补记」，只需在 DESIGN.md 追加一节（含本文档路径），不触发下游重跑。conductor 侧无额外 Stage 4 重跑需求。

## 反馈结论

设计文档中「软件 PRNG ~9 op 无更优可行替代 / 公式层面已无优化空间 / 19–21 op 发射主导、固有落差」三处结论已被同口径实测**部分推翻并量化**：5-op counter hash 与 fp16 存储域把 op 数从 20 降到 13，端到端 65.11→27.91 µs（2.33x，同机同口径）；剩余到搬运地板的 2.5–3.0x 差距是「前端无 RNG 原语（CG-2026-0021）」的必然软件代价，**不是 tiling/参数可解**——建议把「5-op hash + fp16 存储域 + 8192 tile」作为该算子族的新基线写回设计侧，并把 `T.rng_*` 前端原语记为后续工具链升级时的回访项。

建议路由: **附录补记**（optimize 场景无 Stage 1，按 `_shared/standards/perf-feedback.md` §3，`设计修订` 路径 C 不适用；建议 conductor 向 `_make_dropout_kernel_DESIGN.md` 追加「性能反馈附录」一节引用本文件）
