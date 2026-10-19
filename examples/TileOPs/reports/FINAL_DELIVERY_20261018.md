# TileOPs 19 接口迁移与优化 — 最终交付报告

> 日期：2026-10-18 · 工具链：**CANN 9.0.0**（bishengir 1.1.0 + hivmc 0.2.0）/ tilelang `0.1.2` dev root / Ascend910B3 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 / python 3.11.4。
> 口径：全部为 **9.0.0 端到端 `tileops-report` + `msprof op` Task Duration**；8.5.x 数据仅存 `MIGRATION_INDEX.md` 表 B 历史存档。

## 1. 完成度

| 维度 | 结果 |
|---|---|
| 接口交付 | **19 / 19 接口可用**；GDN in-scope 主路径（K1+K2+K6 split）已交付，K5/CP 与小 shape fallback 保留 |
| 正确性 | **19 / 19 接口**（GDN `test_gdn.py` **10/10**，含 in-scope + tail；Conv1d/Conv1dBias 为 27 passed + 1 skipped，0 failed） |
| 端到端报告 | **17 passed + 1 partial**（Conv1d：correctness 通过，bench 有 flaky unit_stride 用例） |
| 有生效调优 kernel | **15 / 18 已交付（83.3%）** |
| 单算子优化报告 | **16 份** |
| 全 shape 严格达标 | **1 / 19（5.3%，Sub）** |
| shape 级达标 | **40 / 136（29.4%）** |

> “严格达标”按各算子 AI 分类阈值（访存 70% / 算力 60%），微型 shape 豁免单列。

## 2. 19 接口状态

| # | 算子 | 交付 | 正确性 | 9.0 ratio | 达标（可调 shape） | 生效内核 |
|---|---|---|---:|---:|---:|---|
| 1 | HardsigmoidFwdOp | ✅ | 7/7 | 0.37–61.52% | 0/2 | perf_opt |
| 2 | ClampScalarFwdOp | ✅ | 11/11 | 58.21–70.16% | 1/11 | perf_opt |
| 3 | HardswishFwdOp | ✅ | 7/7 | 50.20–60.95% | 2/6 | baseline |
| 4 | AddFwdOp | ✅ | 24/24 | 69.37–83.48% | 10/11 | perf_opt |
| 5 | SubFwdOp | ✅ | 24/24 | 70.70–84.00% | **15/15** | perf_opt |
| 6 | GeluAndMulFwdOp | ✅ | 11/11 | 4.05–47.71% | 0/4 | perf_opt |
| 7 | GeluTanhAndMulFwdOp | ✅ | 11/11 | 2.22–47.63% | 0/4 | perf_opt |
| 8 | IsnanFwdOp | ✅ | 9/9 | 52.91–79.33% | 1/5 | baseline |
| 9 | IsinfFwdOp | ✅ | 9/9 | 41.45–78.13% | 1/5 | perf_opt |
| 10 | IsfiniteFwdOp | ✅ | 9/9 | 42.29–78.29% | 1/5 | perf_opt |
| 11 | FusedAddLayerNormFwdOp | ✅ | 14/14 | 0.54–43.89% | 0/5 | perf_opt |
| 12 | BatchNormBwdOp | ✅ | 12/12 | 11.91–24.44% | 0/6 | perf_opt |
| 13 | Conv1dFwdOp | ✅ | 27+1 skip | 0.04–1.79% | 0/15 | perf_opt |
| 14 | ClampMinFwdOp | ✅ | 14/14 | 61.77–68.84% | 0/7 | 复用 ClampScalar |
| 15 | LerpFwdOp（标量） | ✅ | 36/36 | 60.22–82.03% | 9/11 | baseline |
| 16 | Conv1dBiasFwdOp | ✅ | 27+1 skip | 0.03–1.73% | 0/8 | perf_opt |
| 17 | Conv3dBiasFwdOp | ✅ | 17/17 | 1.68–20.43% | 0/3 | perf_opt |
| 18 | GLADecodeOp | ✅ | 11/11 | 3.10–20.16% | 0/5 | perf_opt |
| 19 | GatedDeltaNetPrefillFwdOp | ✅（in-scope split） | 10/10 | —（benchmark skip） | — | K1+K2+K6 split；K5/CP 待续 |

## 3. 主要性能收益

| 算子 | 优化 | 收益 |
|---|---|---|
| Conv1d（通用 kernel） | taps-in-K + 直读原生 weight + 单通道 polyphase + N tile 512 | wav2vec2 **17.7×**、encodec-init **4.2×**；bias 路径 8.3× / 1.15× |
| Conv3d（unet） | padded-N 分块 + NDHWC 暂存（im2col DMA 段数 91774→973） | unet **21.3×**；r3d-stem 对 Stage-3 **2.36×** |
| GLA Decode | k_tile 32 + 超发 shape persistent 分核 | 相对 k16 **−24.2~35.4%**（persistent 再取 −8.0~14.3%） |
| GeluAndMul | dtype 分化 erf 阶数 | prefill **−43.6%** |
| BatchNormBwd | subid 分片（MixCV → Vector 归约） | **−33~−46%** |
| FusedAddLayerNorm | loads-first 双输入前置装载 | 8b prefill **−23.2%** |
| Conv1d unit_stride | co 块切分 + x_pad 跳过 | whisper **15×**、encodec-deep **3.6×** |
| Isfinite / Isinf | bf16 `vand` 6→4 pass | bf16 **−21.2% / −19.4~19.9%** |
| GeluTanhAndMul | fold + hoist `vbrc` | prefill −8.2% |

## 4. 未收口项与阻塞证据

| 项 | 状态 | 证据 |
|---|---|---|
| **GDN prefill** | in-scope 主路径已交付；K5/CP 待续 | K1+K2+K6 split 在 `S=64/K=64`、`S=100/K=64`、`S=3000/K=128` 对拍通过；`test_gdn.py` 10/10。`examples/gdn_prefill/correct_initial_states/PROBE_20261019.md` |
| **Conv1d bias 中继** | 阻塞 | CG-2026-0028（bias-in-K 不可表达）+ CG-2026-0029（ws 宽度必须 128，160/192/256 触发 AIV/AIC 双 `hacc.entry`）；`encodec-init-bias` 240µs vs nobias 66µs |
| **ClampScalar / ClampMin 9.0** | 已定性，保留 perf_opt | 9.0 baseline 45.35–64.37% vs tuned 58.21–70.16%；tuned 仍快 7.6–40%，是环境整体下移 |
| **Add `linear-bias bf16`** | 贴线 10/11 | 69.37%，复测 63.7–69.4%；不再追 |
| **Conv1d bench partial** | 既有缺口 | `encodec-deep-bias` 走未改动的 unit_stride，bishengir 偶发 SIGSEGV，单点重跑 2 passed |

## 5. 关键提交

| commit | 内容 |
|---|---|
| `250a273` | 18 个已交付接口统一 9.0.0 口径 |
| `bcf935a` | Clamp 9.0 归因（环境下移，非调优失效）+ Add 贴线复测 |
| `aff2403` | Conv1d 通用 kernel 17.7×/4.2× + gate 4 多 kernel 支持 |
| `f7d17dc` | Conv3d unet 21.3× |
| `802e819` | GDN 拆分判定性探针 |
| `14bcfe4` | GLA persistent 分核 + Conv1d bias N-tile 阻塞（CG-2026-0029） |

## 6. 复核入口

```bash
cd examples/TileOPs
# 正确性（整仓）
python -m pytest tests/ops -q
# 单算子 9.0 报告
python -m tileops.reporting.cli run --op <OpName> --prof-mode msprof
# 状态
python3 ../../.agents/tools/statectl.py show --dir tileops/kernels/<family>/<op>/<op>_kernel
python3 ../../.agents/tools/statectl.py gate 4 --dir tileops/kernels/<family>/<op>/<op>_kernel
```

- 迁移索引：`MIGRATION_INDEX.md`
- 当前状态：`STATUS.md`
- 总报告：`examples/TileOPs/reports/迁移与优化总报告.md`
- 优化总表：`examples/TileOPs/reports/optimize/STAGE4_SUMMARY.md`
- 剩余空间：`examples/TileOPs/reports/optimize/HEADROOM_ASSESSMENT.md`
- 横向对照：`examples/TileOPs/reports/横向对照_SOL.md`

## 7. 结论

- **迁移侧收口**：19/19 接口可用；GDN in-scope 主路径（K1+K2+K6 split）已交付，`test_gdn.py` **10/10**；K5/CP 与小 shape fallback 保留。
- **优化侧收口**：15/18 有生效调优 kernel；两个最大杠杆（Conv1d 17.7×、Conv3d 21.3×）与 GLA persistent 已落地。
- **未收口项均有明确证据**：GDN K5/CP 与性能数据、Conv1d bias（CG-2026-0028/0029）、FALN（高不确定设计项）；不建议在无新工具链/新证据前继续投入。
