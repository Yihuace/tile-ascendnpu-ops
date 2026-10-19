# 当前工作状态（每轮更新）

> 本窗口：**19 算子统一 CANN 9.0.0 口径**。四个 8.5 残留（Add / Sub / Isnan / ClampMin）已完成端到端重测；Add / Sub 同时切到 `perf_opt`。
> 更新：2026-10-18 16:00 (+08:00) · 会话：9.0 口径统一
> **收口（2026-10-18 22:2x）**：最终交付报告 `examples/TileOPs/reports/FINAL_DELIVERY_20261018.md`；18/19 完整交付 + GDN correctness-first fallback。
> **环境更新（2026-10-19）**：当前运行环境为 **CANN 9.2.0 + Apr v1.2.0 补丁编译器 + hivmc 0.3.0**；19 批 conv1d/conv1d_bias/conv3d 已验证通过，MHA/SSD/GDN blocksolve 仍卡 hivmc 代际。

## 总览（严格口径；报告数据为 CANN 9.0.0 存档，当前运行环境已切 CANN 9.2.0/Apr）

**19 个接口：18 个完整交付；GDN 为 correctness-first fallback（6/6 tests），六 kernel 的 TileLang 端口待续。**

| 判定 | 数量 | 算子 |
|---|---|---|
| **全 shape 严格达标** | **1** | Sub（15/15） |
| 贴线 / 部分达标 | 4 | Add（10/11；`linear-bias bf16` 复测 63.7–69.4%，确认贴线未过）、Lerp（9/11）、Hardswish（2/6）、ClampScalar（1/11） |
| 未达标但**已采纳真实时延收益** | 12 | Hardsigmoid、ClampMin、Isinf、Isfinite、GeluTanh、GeluAndMul、FALN、BatchNormBwd、Conv1d、Conv1dBias、Conv3dBias、GLA |
| 未达标且未采纳 | 1 | Isnan（无收益，MTE2 带宽地板） |
| 正确性交付但 kernel 未收口 | 1 | GDN（reference fallback，6/6；2/6 kernel，CG-2026-0003；拆分探针见 PROBE_20261018） |

> 全 shape 达标从上一版“3 个”修正为“1 个”：Add 在 9.0 端到端下 10/11（`linear-bias bf16` 69.37%，复测 63.7–69.4%，确认未过线），ClampScalar 在 9.0 下仅 1/11（8.5 的 71.9–82.8% 不再引用）。
> 所有数字来自 `reports/tileops/<run_id>/run.json`；8.5.2 数据仅保留在 `MIGRATION_INDEX.md` 表 B 历史存档（不在本批范围）。

## 本轮 9.0 统一重测（4 个 8.5 残留）

| 算子 | 生效内核 | 正确性 | ratio（9.0 端到端） | 达标 | run_id |
|---|---|---|---:|---:|---|
| AddFwdOp | `perf_opt`（本轮切换） | 24/24 | 69.37–83.48% | 10/11 | `20261018_143726_708597` |
| SubFwdOp | `perf_opt`（本轮切换） | 24/24 | 70.70–84.00% | 15/15 | `20261018_145645_852546` |
| IsnanFwdOp | baseline（不采纳） | 9/9 | 52.91–79.33% | 1/5 | `20261018_152024_774098` |
| ClampMinFwdOp | 复用 ClampScalar `perf_opt` | 14/14 | 61.77–68.84% | 0/7 | `20261018_153708_107121` |

## P1：Conv1d 通用 kernel 优化（9.0，2026-10-18）

| 算子 | 生效内核 | 正确性 | bench | 关键收益 | run_id |
|---|---|---|---:|---|---|
| Conv1dFwdOp | `perf_opt`（通用 `conv1d`） | **27 passed + 1 skipped**（CG-2026-0003），0 failed | 15/17（2 例 flaky） | wav2vec2 **17.7×/17.2×**、encodec-init **4.2×/3.4×** | `20261018_180016_718940` |
| Conv1dBiasFwdOp | 同上（has_bias 路径） | **27 passed + 1 skipped**，0 failed | 8/9（1 例 flaky） | wav2vec2-bias **8.3×/8.9×**、encodec-init-bias 1.15×/1.20× | `20261018_180716_376489` |

> 手法：taps-in-K 单次 gemm + A 直读原生 weight + 恒等 x_pad 跳过 + 单通道 polyphase + N tile 512。
> **wav2vec2 的 CG-2026-0003 编译崩溃已被新 IR 形态绕开**；残余 bench 失败为 unit_stride 的 flaky SIGSEGV，单点重跑 2 passed。
> 新缺口：**CG-2026-0028**（Vector prelude「1D GM 切片 → 带列偏移 GM 子区域」写必崩），bias-in-K 折叠被迫放弃；**CG-2026-0029**（bias MixCV 的 ws 宽度必须 128，放大 N tile 触发 dual-entry）。bias 路径 240µs 的差距暂无法用 N tile 解决，已回退 128。

## P1：Conv3d unet 路径（9.0，2026-10-18）

| 算子 | 生效内核 | 正确性 | 关键收益 | run_id |
|---|---|---|---|---:|
| Conv3dBiasFwdOp | `perf_opt`（tap 打包 + padded-N 分块 + NDHWC 暂存） | **17/17**，0 failed | unet-encoder **6428 → 307µs（21.3×）**；r3d-stem **3096 → 2589µs（对 Stage-3 2.36×）**；video-down 持平 | `20261018_185919_687302` |

> **归因更正**：`c_in_pad=32` 的权重 L1 常驻**早已生效**（IR 中 27 次 `nd2nz` 在 task 循环外），旧结论“编译器仍逐 task 重载”不成立。真正病根是 **im2col 输入 copy 的 DMA 段数**（每 copy = `c_in_pad` 个 128B 小段、跨 359KB 平面；`aic_mte2_instructions=91774`、`GM_to_L1` 仅 5.63% 带宽）；两轮结构改造把段数降到 **973 条（−94×）** → 301.6µs。常驻反而把 L1 顶到 220KB、丢失多缓冲，是负杠杆。

## P1：GLA persistent 分核（9.0，2026-10-18）

| workload | k16 baseline | k32 current best | final（k32+persistent） | vs current best | vs 基线 |
|---|---:|---:|---:|---:|---:|
| serving-b1-h8 | 9.01 | 6.87 | 6.83 | −0.6%（tie） | −24.2% |
| serving-b1-h32 | 11.03 | 8.35 | 8.21 | −1.7%（tie） | −25.6% |
| serving-b1-h64 | 18.77 | 14.80 | **12.69** | **−14.3%** | −32.4% |
| serving-b8-h32 | 57.59 | 41.73 | **38.39** | **−8.0%** | −33.3% |
| serving-b8-h64 | 106.31 | 77.04 | **68.66** | **−10.9%** | −35.4% |

> run `20261018_215742_239583`，correctness **11/11**、report passed，ratio 3.10–20.16%。
> 分派规则：`B*H ≤ 40` 走原 flat 体（源码逐字节相同，零回退）；`B*H > 40` 才用 `T.Kernel(min(BH,40))` + grid-stride 核内 serial。
> 过程中修复真 bug：persistent 包裹 fp32 直拷路径会静默算错（max|Δo|=5.12）→ persistent 只在 transit 路径启用，fp32 恒走 flat。
> `T.Pipelined` 全落在噪声内 → advice 的 `latency bound:pipeline caused` 指 wave/依赖链，不是 copy-compute 未重叠；不再投入。

## 上一轮 11 个算子的结果（9.0.0 口径，仍有效）

| 算子 | 关键时延收益 | 采纳 | 状态 |
|---|---|---:|---|
| GLADecodeOp | persistent 后相对 k16 **−24.2~35.4%** | ✅ | DONE（latency-bound） |
| HardsigmoidFwdOp | 0（wash，判据失配） | ✅ | DONE |
| GeluTanhAndMulFwdOp | prefill −8.2% | ✅ | DONE |
| GeluAndMulFwdOp | prefill **−43.6%** | ✅ | DONE |
| IsnanFwdOp | 0（MTE2 地板） | ❌ | DONE |
| IsinfFwdOp | bf16 **−19.4~−19.9%** | ✅ | DONE |
| IsfiniteFwdOp | bf16 **−21.2%** | ✅ | DONE |
| FusedAddLayerNormFwdOp | 8b prefill **−23.2%** | ✅ | DONE |
| BatchNormBwdOp | **−33~−46%** | ✅ | DONE |
| Conv1dBiasFwdOp / Conv1dFwdOp | whisper **15×**、encodec-deep 3.6×；通用 kernel wav2vec2 **17.7×**、encodec-init **4.2×** | ✅ | DONE（形状路由 + taps-in-K） |
| Conv3dBiasFwdOp | r3d-stem **2.36×**、unet **21.3×** | ✅ | DONE |

## 必须记住的七件事

1. **CANN 9.0 编译器缺陷（CG-2026-0003, recurring, occurrences=4）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取 `getResult(0)` 越界；**9.0.0 与 9.2.0 官方包均有**，8.5.2 无（405/405）。
   - Conv1d/Conv1dBias 已用**形状路由**绕开（`dilation==1` 才走 unit-stride；unit-stride 仅 `c_in>=64` 用调优版）；通用 kernel 的 wav2vec2 由 taps-in-K 新 IR 形态绕开
   - GDN blocksolve 仍阻塞（触发点在算法本体的 L0C→GM→L1 回喂；2026-10-18 拆分探针：3 级解锁 `S=1024,H=4`，但 `S=64,H=2` 在 1/3/6 级、group、核数下均失败）
2. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降（GeluAndMul −43.6% 时延，ratio 43.2%→46.0%）。这类算子看 Task Duration，不看 ratio。
3. **BatchNormBwd 的 ratio 因 subid 去双计而下降**（15.3–28.2% → 13.0–25.5%），不可比，时延 −40%。
4. **口径策略：全部统一 CANN 9.0.0**（2026-10-18 用户口径）。8.5.2 数据仅在表 B 历史存档中保留，不再用于本批 19 接口的结论；跨版本数字不横比。
5. **CG-2026-0028（新）**：Vector prelude 里「1D GM 切片 → 带列偏移 GM 子区域」写必崩（`Unknown core type: llvm.func @malloc`），bias-in-K 折叠不可表达；bias 路径保留 Cube→ws→Vector 中继（`encodec-init-bias` 240µs vs nobias 66µs，差 3.7×）。
6. **Conv3d 的 L1 常驻归因已更正**：`c_in_pad=32` 常驻早已生效，不是杠杆；真正病根是 im2col 输入 copy 的 DMA 段数（128B 小段 × `c_in_pad`）。修段数（padded-N 分块 + NDHWC 暂存）拿到 unet 21.3×；常驻把 L1 顶到 220KB 会丢多缓冲，反成负杠杆。
7. **Clamp 族 9.0 回退已定性：工具链/环境整体下移，不是调优失效**。同链路 baseline vs tuned（`20261018_160303` vs `20261018_131331`）：tuned 仍快 **7.6–40%**（16M 30.7–32.9µs vs 38.3–43.1µs；256M 803–810µs vs 864–874µs）。9.0 baseline 自身只有 **45.35–64.37%**，所以 8.5 的 71.9–82.8% 不能跨版本引用；**保留 perf_opt**，不再重优化。

## 报告规范

- 模板：`examples/TileOPs/reports/optimize/REPORT_TEMPLATE.md`
- 16 份单算子报告已按 8 节统一（0 身份 / 1 结论 / 2 优化方法与过程〔2.1 诊断 + 2.2 方法 + 2.3 逐轮〕/ 3 结果 / 4 精度 / 5 时间成本 / 6 产物 / 7 天花板豁免缺口 / 8 建议）
- 旧版报告存各目录 `history_version/OPTIMIZE_REPORT_v0.md`
- 缺失报告：GDN（fallback 交付，benchmark 暂 skip）；ClampMin/Lerp 无独立报告（分别由 ClampScalar / 未优化覆盖）

## 环境纪律（2026-10-18 更新）

- **当前生效栈**：tilelang Apr v1.2.0 前端（`build`；Oct 在 `build.oct-bak`）+ CANN **9.2.0**（`bishengir-compile 1.2.0 8796a8ac + mmadL1 补丁` + `hivmc 0.3.0`）。
- **已验证**：conv1d_bias 23 passed/1 xfailed/4 xpassed；conv1d + conv3d 通过；MHA/SSD/GDN blocksolve 仍卡 `hivmc` external run（v1.2.0 tag 与 hivmc 0.3.0 不同代）。
- **回退**：`cd /home/developer/tilelang-mlir-ascend && bash switch_stack.sh oct && cp ~/.bashrc.oct-current ~/.bashrc`。

## 环境纪律

- 上机命令经 `.agents/tools/npu_lock.sh`；非交互 shell 先 `source perf_opt/env.sh`
- 长测试运行期间禁止编辑被测源文件（JIT 源映射）
- **所有 agent/subagent 一律 `deepseek/deepseek-flash`**（`AGENTS.md` Model Policy，禁止 pro）
- 新派发的 optimizer 必须在返回前自查 `statectl gate 4` 直到 pass

## 后续可选（按收益排序）

1. **FALN 设计层**：两次 row-reduce 改 Cube GEMV 卸载，20–40%
2. **GDN prefill**：判定性探针已完成（2026-10-18，`examples/gdn_prefill/probe/experiments/18_level_split/PROBE_20261018.md`）；结论是拆 kernel 只能解锁大 shape，小 shape 需设计级替代（纯 Vector 前处理 / 换 tile 布局）
3. Tier 2 进化提案（`VP-2026-0147`~`0162`）待人工审批
