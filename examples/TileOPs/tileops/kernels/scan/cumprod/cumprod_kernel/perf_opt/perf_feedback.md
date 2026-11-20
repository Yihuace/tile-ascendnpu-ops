# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `_cumulative_kernel_DESIGN.md` §1.6.0（R4 硬件亲和性）/§1.6.3（布局决策 #1）/§6.1 把
  `T.cumsum` 当作**硬件 scan 原语**（"1 次 `T.cumsum` 原语替代 N 次串行向量 op"、"op_tile≈16"），整套
  C1 选型（log 域 `exp(cumsum(ln|x|))·(-1)^(cumsum(x<0))`）与其 roofline 估算均由此成立。该前提在本工具链
  上不成立：npnir `hivm.hir.vcumsum` 被降为**标量指令**，故 C1 的结构性成本是 `2·M·N` 次标量 element-step，
  非 tiling/参数可解（配置扫描 6765–7318 us 全落 ~1% 噪声带）。
- 实测与假设矛盾: DESIGN §1.6.0 估算最大负载 `hidden-state`（2048×4096 fp16）≈ **410 us**；实测 baseline
  **8422 us（20.5x，>2x 门槛）**，最优调优结果 **6766 us（16.5x）**。编译器对 `via VCumsumOp` 直接给出
  `will execute by scalar instruction with low effiency`，与"硬件 scan 原语一次完成"假设直接矛盾；profile 显示
  `aiv_scalar_ratio = 0.9898`、`aiv_vec_ratio = 0.0101`（99% 标量）。DESIGN §1.6.4 预注册的纯向量 C2 备选亦
  实测慢于 baseline（region-offset 数据流标量化），故结构性 >2x 加速无法在本设计族内兑现。

## 参照锚定

- 参照实现: 同族已在仓库落地的 scan 实现 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`
  （CASE-ssd-chunkscan-migration，MixCV Expert Cube+Vector chunk scan）；顺序 scan 基元
  `examples/TileOPs/tileops/kernels/reduction/_primitives.py`（其 `make_cumulative_scan` 与本题 baseline
  同型，无性能参照价值）；`docs/Tilelang.language/排序操作/T.vcumsum.md` 仅声明 dtype/shape 规格，无吞吐/向量性条款。
- 不可移植论证: SSD chunk scan 的 scan 维是**chunk 级状态递推**（少量 chunk 的 Cube gemm 双路径），并非
  cumprod 所需的**逐元素前缀积/前缀和**；移植到本算子需要把逐元素 scan 改写成 chunk 内 Cube 三角矩阵乘 +
  chunk 间传播乘，属算法族替换 + Expert MixCV 跨引擎重写，超出 Stage 4 调优授权范围。**未实证假设声明**：本
  阶段未实现该 Cube 路线，故"其能否 >2x 优于 scalar cumsum"为未实证推断，非实测结论；本反馈不据此断言绝对
  上限，仅断言**冻结的 C1（log 域 + `T.cumsum`）设计族**在当前工具链上的局部地板。未找到与本算子同形（纯
  Vector、逐元素、无 Cube 状态）的快速 scan 参照实现。

## 假设

`_cumulative_kernel_DESIGN.md` §1.6.0 调研结论与 §1.6.3 布局决策 #1：以 `T.cumsum(dim=1)`（`hivm::VCumsumOp`）
作为硬件向量 scan 原语，`op_tile≈16`，据此估算 `hidden-state` ≈410 us、`long-seq` ≈128 us，并给出
"发射/计算主导"而非"访存主导"的结论（§1.6.0 roofline 表）；§9 R4 亦列出"减少 op_tile"作为 Stage 4 杠杆。
DESIGN 自身已标注：若 0.5 us/op 常数外推偏差 >2x，Stage 4 应触发 `[DESIGN_LIMIT]` 归因。

## 实测

msprof op 口径（kernel `main`，launch-count 20，warm-up 5；device Ascend910_9362，40 向量核）：

| workload | dtype | baseline Task Duration (us) | final Task Duration (us) | 备注 |
|---|---|---:|---:|---|
| hidden-state-scan 2048×4096 | fp16 / bf16 | 8422.01 / 8425.57 | 6766.00 / 6767.38 | ratio 0.423→0.446 % |
| long-seq-scan 64×32768 | bf16 | 2185.68 | 2089.52 | ratio 0.483→0.438 %（向量 op 数下降） |
| non-aligned-cumprod 512×300 | fp16 / bf16 | 200.00 / 200.62 | 127.12 / 127.28 | ratio 0.429→0.444 % |

- 诊断信号: `aiv_scalar_ratio = 0.9898`、`aiv_vec_ratio = 0.0101`（hidden-state baseline）；
  编译期告警 `Op 'hivm.hir.vcumsum' will execute by scalar instruction with low effiency`。
- 对照实测（单测探针，2048×4096 fp16 同 tiling）：纯搬运 16.0 us；搬入+全缓冲 `T.vmul`+搬出 16.5 us；
  单次 region-copy + separate-dst 全缓冲 `T.vmul` 15.7 us；而 12 步 region-offset / 链式 region-copy 数据流
  20858–20886 us（全部标量化）。即：向量逐元素算力可用（~16 us 地板），但 scan 所需的移位/前缀数据流在本
  工具链上无法保持向量化。
- raw profile: `perf_opt/profiles/{baseline,final,candidate_ks}/{workload}/`；配置扫描与探针日志见
  `perf_opt/logs/`。
- DESIGN §1.6.4 预注册 C2 备选（纯向量 shift-multiply）实测：Kogge-Stone 变体 9856 us（hidden-state）、
  3038 us（long-seq）、473 us（non-aligned），**全 workload 慢于 baseline**，故 A/B 裁决维持 C1。

## 影响面

- 算子族/shape: 所有依赖 `T.cumsum` / `hivm.hir.vcumsum` 的 scan 类算子（cumsum/cumprod、SSD/linear-attention
  的混合状态 scan 中若走该原语的分支），以及任何把 scan 当 "1 原语/tile" 计入 op 预算的设计估算；本算子全
  shape（N 从 300 到 32768、fp16/bf16/fp32）均受同一标量地板约束。
- 改设计预期影响: 需修订 §1.6.0 roofline（常数校准为"cumsum 为标量"）、§1.6.3 布局决策 #1 与 §9 R4 杠杆
  清单；若要突破该地板，须走算法族替换（Cube 三角矩阵前缀积 / chunk 化两遍 scan + workspace），预计属于
  Stage 1（designer, mode=revision）工作量级。
- 交付影响: 本 Stage 4 交付仍为当前设计族内的最优版本（`perf_opt/_cumulative_kernel.py`，1.05–1.58x，
  Roofline % 不过 0.5%）；60% Roofline 目标在本设计族内结构性不可达。

## 反馈结论

`T.cumsum` 在本工具链被降为标量指令，使 C1（log 域硬件 scan）设计的真实上限是 `O(2·M·N)` 标量成本，
实测比 DESIGN 估算慢 16.5–20.5x 且不可由 tiling 改善——60% Roofline 目标结构性不可达，需设计层修订
（校准 roofline 常数或替换 scan 算法族）。

建议路由: 附录补记

## 修正附录（2026-10-22，v3 定稿后回填；正文按 append-only 保留不改）

本附录只登记两条「已被后续实测修正」的判断，不改变上文对**冻结 C1 设计族**（log 域 + 标量
`T.cumsum`）的结论——那条结论仍然成立：在该设计族内 1.05x-1.58x 就是地板。

### R1 「Cube 三角矩阵前缀」不再是未实证假设

上文「参照锚定 / 不可移植论证」把「chunk 内 Cube 三角矩阵乘 + chunk 间传播乘」标为**未实证假设**
（"本阶段未实现该 Cube 路线，故其能否 >2x 优于 scalar cumsum 为未实证推断"）。该路线已在 v2（2026-10-19）
与 v3（2026-10-22）实现并实测，口径仍为 `msprof op` `Task Duration(us)`（多 kernel 算子取同一次测量内
全部 kernel 时长之和，见 `perf_opt/run_suite_v3.py`）：

| workload | v1 baseline (us) | v3 定稿 (us) | 实测加速 |
|---|---:|---:|---:|
| hidden-state-scan fp16 2048x4096 | 8422.01 | 644.15 | 13.07x |
| hidden-state-scan bf16 2048x4096 | 8425.57 | 655.65 | 12.85x |
| long-seq-scan bf16 64x32768 | 2185.68 | 209.78 | 10.42x |
| non-aligned-cumprod fp16 512x300 | 200.00 | 39.18 | 5.10x |
| non-aligned-cumprod bf16 512x300 | 200.62 | 38.40 | 5.22x |

结论修正：上文「结构性 >2x 加速无法在本设计族内兑现」应读作**只对冻结的 C1 标量族成立**；换算法族
（Cube 三角前缀 + 层次化 carry）后实测 5.10x-13.07x，且 long-seq 不再需要回退 v1。定稿文件
`examples/TileOPs/tileops/kernels/scan/cumprod/cumprod_kernel/perf_opt/CumprodFwdOp.py`。

### R2 设计估算失准的归因需要补一项（不只是「cumsum 是标量」）

上文把 DESIGN roofline 的 20.5x 偏差全部归因于「`T.cumsum` 被降为标量指令」。v3 实测显示还有第二个
独立失准项，与 scan 原语无关：**多相位流水的中间量流量与访问粒度**。实测常数（Ascend910_9362，
CANN 9.0.0，2026-10-22 本机重验）：

- 中间量：v2/v3 需要两个 `(M, N)` fp32 缓冲（幅值前缀 + 符号计数前缀）= 每元素 8B 写 + 8B 读，
  是输入流量的 4 倍（fp16/bf16）。定稿版 hidden-state 的 phase 1 占 446/644us（69%）、phase 3 占 184us；
  两个相位都是 latency-bound（phase 1 全部 pipe ratio 约 0.001-0.18，block 时长 ~7us 对 ~0.3us 算术；
  phase 3 有效带宽 31 GB/s，峰值 1.8 TB/s）。
- 访问粒度：对 `(M, NR)` 缓冲取「单列」（bm 行 x 4B、行距 `NR*4`）是 ~1us/次的 latency-bound MTE2，
  实测 `aiv_mte2_ratio 0.875`、`aiv_mte2_active_bw 2.9 GB/s`（同 kernel 的 `aiv_vec_ratio` 仅 0.06）；
  改成整段 `(bm, ncp2)` tile 载入后同一相位 339us -> 84us（long-seq，Round 3）；再由 `BM` 32->64 与
  符号指示量去 `T.vdiv` 进一步降到 129us/73us（Round 3b）。
- 因此「设计期把 scan 当 1 原语/tile」之外，roofline 还应显式计：中间 fp32 缓冲的 4x 放大、以及
  「carry 按 chunk 逐元素取用」时的 DMA 粒度代价（每 block 一次小 DMA = 1us 量级固定成本）。

建议路由：附录补记（本附录即补记）。DESIGN 若修订，修订点是 §1.6.0 roofline 的中间量流量项与
§1.6.3 的 carry 布局决策，而不是推翻 Cube 三角前缀选型——后者已被实测支持。
