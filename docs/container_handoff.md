# 换容器交接清单（8.5 容器 → 9.0 容器）

> **旧容器**：CANN **8.5.2**，工作全部完成并推送 GitHub（`main`，HEAD `04a22fd`）。
> **新容器**：CANN **9.0.0**。以下为"从旧容器搬到新容器"的清单 + 新容器开工步骤。

## 一、仓库内（已在 GitHub，`clone` 即得，无需手工搬）
- 仓库：`https://github.com/Yihuace/tile-ascendnpu-ops.git`（branch `main`）。
- 含：18/19 算子交付、5+ 优化（`perf_opt/`）、总报告/横向对照、任务书 §3.1 口径与达标线、`docs/*`（含 `cann9_adaptation.md`、`deliverables_index.md`、`thin_interface_status.md`）、GDN 设计/探针、pattern-library 回写、CG-2026-0018..0026。

## 二、**容器本地**（不在 repo，需在新容器重建/搬运）
| 项 | 旧容器路径 | 处理 |
|---|---|---|
| **tilelang dev-root + CANN9 shim**（关键） | `/home/developer/tilelang-mlir-ascend`；shim 在 `tilelang/jit/jit_npu.py::_npuir_to_bin_enable_npu_compile`（`linalg = re.sub(r"\s+syn_instr_mode\s*=\s*<[^>]*>", "", linalg)`） | 新容器若已装匹配 9.0 的 tilelang，**先验证 shim 是否仍需要**；否则按 `docs/cann9_adaptation.md` §3 打补丁 |
| **迁移 kit 总包** | `/mnt/workspace/cannlab-migration-kit_20261014.tgz`（3.4G，SHA256 `5b8a6850…4ed`） | 需**手动拷贝**出旧容器 |
| **GPU 源树** | `/mnt/workspace/TileOPs-main`（== 基线 `1238a9f`）+ `.migration_source_map.json` | 可 `clone tile-ai/TileOPs` 到 `1238a9f` 重建，或拷贝 |
| 工具/脚本 | `/home/developer/.local/toolchain`、`build_npuir_*.sh` | 视新容器是否自带 |

## 三、新容器开工步骤（CANN 9.0）
1. **clone 仓库** → `pip install -e ".[dev]"`（`examples/TileOPs`）。
2. **确认 tilelang/CANN**：`python -c "import tilelang,torch_npu"`；量 `ASCEND_HOME_PATH`。
3. **打/验 CANN9 shim**（`syn_instr_mode`）。
4. **跑正确性**：`pytest tests/ops -q`（旧容器 9.0 下 = 398/405；目标补到 405/405）。
5. **剩余 7 个 SEGV**：见 `docs/cann9_adaptation.md` §6（需匹配版 AscendNPU-IR）。
6. **再谈优化/收口**。

## 四、剩余工作（全部在 9.0 上做）
1. 打通 9.0：解决 7 个 SEGV（AscendNPU-IR 匹配版 + 重建）；**顺验 GDN 的 Cube 环路限制是否已修**（`initC` 累加 / `L0C→L1`）。
2. 全量重验：正确性 405/405；**性能全部重测**；**重验"删 NaN 恢复"**（9.0 的 `vmax` NaN 行为）。
3. 继续迁移/优化：GDN 收口（19/19）、conv1d 权重 L1 复用、**conv3d 优化**、D2 时间线补全。
4. 更新 `MIGRATION_INDEX`（8.5.2 作历史基线，9.0 为新基线）。

## 五、当前 8.5.2 基线（作为对照，勿丢）
- 正确性 **405/405**；5+ 优化（ClampScalar/Hardsigmoid/Gelu×2/Hardswish/conv1d）成果见 `reports/optimize/`。
- 关键性能（8.5.2）：conv1d whisper 43.7µs（ratio 14.2%）、GLADecode 2.5–14%、elementwise 访存类 72–83%。
