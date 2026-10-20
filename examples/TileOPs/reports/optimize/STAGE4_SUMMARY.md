# Stage-4 优化分析与口径说明

> **数据表已归口**：19 接口的交付数据（正确性 / ratio / 达标 / 生效内核）见
> [`FINAL_DELIVERY_20261018.md`](../FINAL_DELIVERY_20261018.md) §2 —— 唯一交付口径，
> 机器可读版为 `perf_data/canonical.json`。逐算子的逐轮过程见 `reports/optimize/<Op>/OPTIMIZE_REPORT.md`。
> 本文件只保留**口径陷阱、天花板与能力缺口**（不复述数字）。

## 1. 三个口径陷阱（必须随数据一起读）

1. **ratio 与收益脱钩**：compute-bound 算子删指令时 `total_fops` 同步下降，ratio 几乎不动
   （GeluAndMul −43.6% 时延，ratio 仅 43.2%→46.0%）。这类算子以 Task Duration 判定。
2. **subid 分片使 ratio 减半**：BatchNormBwd 的 `counted_flops` 去双计 → ratio 从 15.3–28.2% 降到
   13.0–25.5%，但时延 −33~−46%。ratio 不可比（`VP-2026-0158`）。
3. **9.0 clamp 族整体下移（已定性）**：同链路 baseline 9.0 为 45.35–64.37%，tuned 为 58.21–70.16%（1/11）、
   ClampMin 61.77–68.84%（0/7）；tuned 仍比 baseline 快 **7.6–40%**。因此上一代工具链的
   71.9–82.8% / 69.8–77.2% 不能跨版本引用，但 **perf_opt 仍是正确选择**，不重优化。

## 2. 天花板与豁免（已举证）

| 类别 | 算子 | 证据 |
|---|---|---|
| MTE2 混合读写带宽地板 | Isnan / Isinf / Isfinite（fp16/fp32） | `aiv_mte2_ratio 0.81–0.90 ≫ aiv_vec_ratio 0.25–0.39`；4→3 pass 仅 −3.4%；40 核最优 |
| 计算受限 + 判据失配 | Hardsigmoid / GeluTanh / GeluAndMul | `aiv_vec_ratio 0.85–0.95`、`computility=Vec_FP32(~99%)` |
| latency-bound | GLA / FALN | `advice=latency bound:pipeline caused`；并行度受限 / 15+ pass + 2 reduce |
| fp32 中转结构天花板 | BatchNormBwd / FALN | fp32 中转 + Vec_MISC + 内存 |
| 编译器缺陷（非优化可解） | GDN blocksolve、Conv1d 的 1 个 bench 形状 | CG-2026-0003；GDN 用 K2 split 绕开，K5/CP 仍待续 |
| 微型 shape 豁免 | Hardsigmoid ×4、Gelu ×1、FALN ×2、GLA smoke | Task Duration ≤5µs 或启动开销 ≥50% |

## 3. 关键技术与能力缺口

**技术突破**

1. `vand(x, ~(-0.0))` 位运算求 `|x|`：绕开 bf16 无 `vabs`，predicate 族 6→4 pass
2. dtype 分化多项式阶数：fp16/bf16 容差 1e-3 无需 deg-20，GELU 族 −43%
3. subid 分片跨族迁移（MixCV → 纯 Vector 逐通道归约）：BatchNormBwd −40%
4. loads-first 双输入前置装载：FALN 8b prefill −23%
5. implicit-GEMM 权重 L1 常驻 + tap 打包：Conv3d r3d-stem 2×；Conv1d co 块切分 15×
6. 形状路由绕开编译器缺陷：`dilation` 分流 + `c_in>=64` 门，使 Conv1d 族在官方 9.0 上可交付
7. **im2col DMA 段数削减**：padded-N 多行分块 + NDHWC 暂存，把 `c_in_pad` 个 128B 小段降到 973 条，
   Conv3d unet 21.3×

**能力缺口**

- **CG-2026-0003（recurring, occurrences=4）**：`ConvertHFusionToHIVM` 对 buffer 形态 `mmadL1` 取
  `getResult(0)` 越界；9.0.0 与 9.2.0 官方包均带，上一代工具链无（405/405）。本仓按工具链绑定 xfail 处理
  （`tests/conftest.py`；根因见 `docs/cann9_adaptation.md` §9.3）。
- `T.gemm` 忽略 L1 切片偏移（TRAP 已入库）；L1 超配 → bishengir SIGSEGV（`n_max` 绕法）。
- predicate 缺 `i1→i8` 直转 / bool `vnot`（CG-2026-0027）。
- **CG-2026-0028**：Vector prelude「1D GM 切片 → 带列偏移 GM 子区域」写必崩
  （`Unknown core type: llvm.func @malloc`）；bias-in-K 折叠被迫放弃，bias 路径保留中继。
- **CG-2026-0029**：Conv1d bias MixCV 的 ws 宽度只能等于 `block_n=128`；放大到 160/192/256 触发
  `multiple entry functions`（AIV/AIC 双 `hacc.entry`），bias N tile 无法放大。
- **L1 常驻归因更正**：`c_in_pad=32` 常驻早已表达，不是杠杆；真正病根是 im2col 输入 copy 的 DMA 段数
  （91774→973 后 unet 21.3×）。常驻把 L1 顶到 220KB 会丢多缓冲，反成负杠杆。
- `statectl` 无法重开 optimize 场景 DONE 任务；多 kernel 包的 `S4-PERF-RECORDS-RECON` 单文件契约已由
  `kernel_opt_py_paths` 扩展支持（2026-10-18）。
