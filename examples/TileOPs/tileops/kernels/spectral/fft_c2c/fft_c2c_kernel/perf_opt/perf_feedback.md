# 性能反馈（[DESIGN_LIMIT]）

> 算子：`FFTC2COp` / `_fft_c2c_kernel`（project `fft_c2c`，scenario `optimize`）
> 调优日志与原始 profile：`examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/perf_opt/opt_log.md`、
> `examples/TileOPs/tileops/kernels/spectral/fft_c2c/fft_c2c_kernel/perf_opt/perf_records.jsonl`

## 触发判定

- 设计层归因: `_fft_c2c_kernel_DESIGN.md` §1.6.2「向量化替代分析」的结论 *"逐元素计算（#1–#5）已全部向量化"*，与 §1.6.3「向量化轴与数据布局决策（阻塞级）」选定的布局（`T.Parallel(ng, h)`，内层轴 `i` 为向量化轴，整行 UB 常驻 + ping-pong）；配套的 §5 分核方案与 §9「B=1 低占用」风险项（`grid=B`，unbatched 时仅占用 1/40 核）。
- 结构性加速估计: **≥2x**（实测外推）。标量吞吐实测 ~0.34 Gop/s（每行 12 级 × 2048 蝶形 × ~10 fp op = 245,760 op 用时约 717.68 us），即使按向量单元 10% 效率外推也有 ~68x 的结构性余量；另有实测佐证：同一 kernel 在 batch=64 时为 23.4 us/行 对 unbatched 的 717.7 us/行，即 **30.7倍** 核占用差。
- 实测与设计假设矛盾: DESIGN §1.6.2 断言蝶形逐元素计算「已全部向量化」，实测 `aiv_vec_ratio = 1.20e-04`、`aiv_scalar_ratio = 0.99884`——99.9% 标量，向量化**完全未发生**。

上述两项条件同时满足，故产出本文件。需说明：断言的直接机制是**工具链能力缺口**（见「参照锚定」与「实测」），因此建议路由以「附录补记」为主，不构成本轮 Stage 4 的阻塞。

## 参照锚定

- 参照实现: **未找到参照，天花板结论置信度降级**。检索范围：本仓 `examples/`（无 FFT/DSP 族算子实现；`examples/TileOPs/` 内同为迁移算子的 elementwise/reduction 族均无逐级变几何结构）、`.agents/skills/tilelang-op-optimize/references/pattern-library/cases.md` 参考实现集（最近邻为 PL-1.1 二轴交换转置链、PL-1.2 C 轴切片累加，均为固定几何布局）、`tilelang` 语言层测试 `testing/python/language/test_tilelang_language_view.py`（`T.view` 官方用例，但**不在 `target="npuir"` 下运行**，不构成 npuir 可用性佐证）。
- 不可移植论证:
  1. PL-1.1 转置链（`pattern-library/layout.md`，实测 ~5.4 µs）解决的是**二轴交换**且依赖核内重排 buffer；FFT 每级需在 `[ng,2,h]` 与 `[2,ng,h]` 之间变几何（`ng`、`h` 每级不同），是 `log2(n)` 个不同的 reshape，超出二轴交换链的表达范围（依据：`layout.md` 对 PL-1.1 的适用边界条款）。
  2. 官方 `T.view` 形态（`testing/python/language/test_tilelang_language_view.py` 的 `A_viewed = T.view(A, new_shape)` + `T.copy`）在 npuir 上**实测失败**：`Check failed: (GetVarValue(op->buffer_var.get()) == mlir::Value{})`；同数据指针的 `T.Buffer(shape, dtype, src.data, scope=...)`（`tilelang/language/customize.py` 的 `view`/`reshape` 实现路径）同样失败；`T.reshape` 在 npuir 解析为 `npuir_reshape(src, dst)`（`tl.npuir_reshape` 算子，非别名）。因此参照实现中「核内重排/别名」这一关键手段**不可移植**——这是实测结论（依据：本轮 probe 集，见 opt_log.md §2.1）。
  3. 未实证假设（已标注）：是否存在完全不需要别名的变几何布局，本轮未穷举证明其不存在；证据仅为「无别名时每级需独立物理 buffer，`n=4096` 需约 768 KB UB 对 192 KB 可用」这一容量实测约束。

## 假设

DESIGN.md 中被本反馈指认的假设/选型：

1. §1.6.2 向量化替代分析：「逐元素计算（#1–#5）**已全部向量化**」——即认定 `T.Parallel(ng, h)` 的蝶形会经 npuir 向量化通道变为向量指令。
2. §1.6.3 布局决策（标注为「阻塞级」）：选定「整行 UB 常驻 + ping-pong」+「`T.Parallel(ng, h)` 内层轴 `i` 为向量化轴」，未采纳核内转置等备选（备选 #3 被否决）。
3. §5 分核方案 / §9 风险项：`grid = B`，`B=1`（unbatched）时仅占 1/40 物理核，被列为 Stage 4 裁决项。

## 实测

口径统一为 `msprof op Task Duration(us)`（唯一 kernel 时延口径）；`--launch-count=20 --warm-up=5`，取 20 次中位数；采集前后 `npu-smi info -t common -i 6` 的 Aicore Usage 均为 0%。

- 基线（`[4096]` complex64，`perf_opt/_fft_c2c_kernel.py`）：Task Duration 中位 **717.68 us**，`aiv_vec_ratio = 1.197e-04`、`aiv_scalar_ratio = 0.99884`、`aiv_ub_read_bw_vector = 0.0423 GB/s`；raw profile `perf_opt/profiles/baseline/round0/fft-4k-c64-unbatched_baseline/OPPROF_20261016224448_CWDVOLGCGCBRURVF`。
- 基线（`[64,4096]` complex64）：Task Duration 中位 **1498.32 us**，`aiv_vec_ratio = 1.157e-04`、`aiv_scalar_ratio = 0.99848`；raw profile `perf_opt/profiles/baseline/round0/fft-4k-c64-b64_baseline/OPPROF_20261017020209_GJXLNFPMMRTDTOOF`。
- 候选 r1_serialvec（`T.serial(g)` + `T.vectorized(i)`，L0 全通过）：Task Duration 中位 **2040.12 us**（较基线回退 2.84x），`aiv_vec_ratio = 4.6e-05`、`aiv_scalar_ratio = 0.99957`——TIR `kVectorized` 路径在 npuir 上同样不产生向量指令；raw profile `perf_opt/profiles/round1/serialvec_4k_unbatched/OPPROF_20261017020002_YMDNBAZLOCYSUVOR`。
- 候选 r1_vecviews（每级 `T.view` 重排为四个单 store 循环）在编译期失败：`tvm::tir::RemoveNoOp` 报 `variable S3i has already been defined, the Stmt is not SSA`；无有效 profile，故不写入 `perf_records.jsonl`。
- 约束来源（源码条款）：`src/transform/npu_loop_vectorize.cc` 的 `VectorizeSingleStatement`（第 1184 行）要求 store 索引为裸 `Var` 或标量常数，`IsSupportedVectorLoadIndex`（第 251 行）要求 load 索引为循环不变或单循环变量单位步长；基线蝶形的 `D[g*h+i]` / `S[g*2h+h+i]` 两条都不满足，必须重排布局，而重排所需的 UB 别名在 npuir 不可用。
- 最终态（无候选被采纳）：`perf_opt/_fft_c2c_kernel.py`（sha256 `623fd321e2befb95d08a3a2afc03f10c1d95d3de01b0e1fa6307c7e3c668c552`，kernel 体与只读基线逐字节相同）复测 719.53 us / 1497.34 us，`aiv_vec_ratio` 仍为 1.2e-04。

## 影响面

- 受影响算子族：任何**核内逐级变几何**的算法——FFT/Stockham/DIF-DIT 网络、多级 stencil、分阶段 tiling（每级段长/组数变化的展开式算法）。
- 受影响的 shape 范围：本算子全部 tune workload（`fft-4k-c64-unbatched-complex64`、`fft-4k-c64-b64-complex64`）；其中 unbatched（`B=1`）额外受 §5/§9 `grid=B` 单核占用限制。
- 改设计预期影响的 Stage 与工作量：若走「工具链补齐别名能力」路线，影响 **编译器层**（`src/transform/npu_loop_vectorize.cc` 的 store/load 索引规则，或 TileLang frontend 的 buffer alias 构造），不属 DESIGN/Stage 1 工作；若走「设计修订」路线（重选分核/分解方式，如 §1.6.0 #8 四步分解），影响 Stage 1 设计文档 + Stage 2 检视 + Stage 3 重实现 + Stage 4 重调优，且其转置/子 FFT 仍会撞上同一别名缺口，预期收益不确定。
- 建议同时按 `AGENTS.md`「Compiler Capability Gap Rule」在 `.agents/evolution/capability-gaps.md` 登记该缺口（层：TileLangIR pass / BisectorIR codegen；能力：UB buffer alias / view；证据：本轮 probe 集与 `src/transform/npu_loop_vectorize.cc` 条款），供跨任务 recurrence 统计与能力补齐提案使用——本 Agent 按本次调度的「产物只写 `perf_opt/`」边界未写入该文件。

## 反馈结论

该算子的性能天花板同时由两处 DESIGN 假设决定：§1.6.2/§1.6.3 假定 `T.Parallel(ng, h)` 蝶形会被 npuir 向量化（实测被推翻：`aiv_vec_ratio` 1.2e-04），而打破该地板所需的核内布局重排能力（UB buffer alias/view）在当前工具链上实测不可用，因此该天花板**非 tiling/参数可解**，且**不能仅靠设计层修订消除**。

建议路由: 附录补记（默认）。向 `_fft_c2c_kernel_DESIGN.md` 追加「性能反馈附录」小节（一段摘要 + 本文件路径引用，只追加不删改既有章节）；另建议由 conductor 判定是否按 `AGENTS.md` 能力缺口规则登记 `.agents/evolution/capability-gaps.md`。不建议走「设计修订」路径：本场景（`optimize`）无 Stage 1，且已知设计修订无法单独消除该地板。
