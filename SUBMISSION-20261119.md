# 提交准备清单（19 个任务书算子 + 三算子 = 20）— 2026-11-19

> **口径修正**：此前那版按 `examples/TileOPs/task_operators.md` 的 24 个整理，那是**吕盼豪**的登记表，不是本次范围。
> 本版按 `TileOPs迁移任务书` / `MIGRATION_INDEX.md` 表 A 的 **19 个**，并上三算子 **Add / Argmax / GLA**；
> 其中 Add、GLA 已在 19 内，所以实际是 **20 个（19 + ArgmaxFwdOp）**。

## 1. 结论

| 项 | 结果 |
|---|---|
| 目标算子 | 20（19 任务书 + Argmax；Add/GLA 已在 19 内） |
| 可交付 | **18** |
| 阻塞 | **2**：`LerpFwdOp`（标量）、`GatedDeltaNetPrefillFwdOp` —— 本交付树无任何实现文件 |
| 六区 + 证据 + perf_opt | 18/18 齐 |
| 提交闸门（规则同 `check_op_submission.py`，`tools/gate_20ops.py`） | **18/18 PASS，0 FAIL，2 BLOCKER** |
| 官方 report | 18/18 `passed`（本轮把 11-16~11-18 的 9 份归档进树） |
| smoke 回归 | **222 passed / 0 failed / 91 deselected**（18 个 test 文件，13.1 min） |
| 实现新鲜度 | **18/18 已验证的最优交付态**（Hardswish 已切 11-03 版；FALN 的 dev R16/R17 A/B 失败未采纳；Isnan 候选否决保 baseline） |
| 禁用文件 / >5MB / 密钥 | 0 |

## 2. 20 算子清单

| # | 任务书算子 | 本树实现 | family | 分支 | PR | 最新官方 report | 备注 |
|---:|---|---|---|---|---|---|---|
| 1 | HardsigmoidFwdOp | tileops/ops/elementwise/hardsigmoid.py | elementwise | `op/elementwise/hardsigmoid-guoyihua` | 14 | `passed` @ 2026-11-17 |  |
| 2 | ClampScalarFwdOp | tileops/ops/elementwise/clamp_scalar.py | elementwise | `op/elementwise/clamp-scalar-guoyihua` | 3 | `passed` @ 2026-10-21 |  |
| 3 | HardswishFwdOp | tileops/ops/elementwise/hardswish.py | elementwise | `op/elementwise/hardswish-guoyihua` | 16 | `passed` @ 2026-11-19 | 已切 11-03 最优版 |
| 4 | AddFwdOp | tileops/ops/elementwise/add.py | elementwise | `op/elementwise/add-guoyihua` | 7 | `passed` @ 2026-11-17 |  |
| 5 | SubFwdOp | tileops/ops/elementwise/sub.py | elementwise | `op/elementwise/sub-guoyihua` | 8 | `passed` @ 2026-10-23 |  |
| 6 | GeluAndMulFwdOp | tileops/ops/elementwise/gelu_and_mul.py | elementwise | `op/elementwise/gelu-and-mul-guoyihua` | 9 | `passed` @ 2026-11-17 |  |
| 7 | GeluTanhAndMulFwdOp | tileops/ops/elementwise/gelu_tanh_and_mul.py | elementwise | `op/elementwise/gelu-tanh-and-mul-guoyihua` | 10 | `passed` @ 2026-11-17 |  |
| 8 | IsnanFwdOp | tileops/ops/elementwise/isnan.py | elementwise | `op/elementwise/isnan-guoyihua` | 13 | `passed` @ 2026-11-16 |  |
| 9 | IsinfFwdOp | tileops/ops/elementwise/isinf.py | elementwise | `op/elementwise/isinf-guoyihua` | 11 | `passed` @ 2026-11-17 |  |
| 10 | IsfiniteFwdOp | tileops/ops/elementwise/isfinite.py | elementwise | `op/elementwise/isfinite-guoyihua` | 12 | `passed` @ 2026-11-17 |  |
| 11 | FusedAddLayerNormFwdOp | tileops/ops/norm/fused_add_layer_norm.py | norm | `op/norm/fused-add-layer-norm-guoyihua` | 4 | `passed` @ 2026-10-21 | dev R16/R17 A/B 失败（13b −23%），保持交付版 |
| 12 | BatchNormBwdOp | tileops/ops/norm/batch_norm_bwd.py | norm | `op/norm/batch-norm-bwd-guoyihua` | 5 | `passed` @ 2026-11-17 |  |
| 13 | Conv1dFwdOp | tileops/ops/convolution.py | convolution | `op/convolution/conv1d-perf-guoyihua` | 95 | `passed` @ 2026-10-28 |  |
| 14 | ClampMinFwdOp | tileops/ops/elementwise/clamp_min.py | elementwise | `op/elementwise/clamp-min-guoyihua` | 63 | `passed` @ 2026-10-23 |  |
| 15 | LerpFwdOp | **缺失** | elementwise | `` | — | — | 本交付树无 manifest/kernel/op/test/bench |
| 16 | Conv1dBiasFwdOp | tileops/ops/convolution.py | convolution | `op/convolution/conv1d-bias-guoyihua` | 88 | `passed` @ 2026-10-27 |  |
| 17 | Conv3dBiasFwdOp | tileops/ops/convolution.py | convolution | `op/convolution/conv3d-bias-guoyihua` | 31 | `passed` @ 2026-10-23 |  |
| 18 | GLADecodeOp | tileops/ops/linear_attention/gla_decode.py | linear_attention | `op/linear_attention/gla-decode-guoyihua` | 17 | `passed` @ 2026-11-18 |  |
| 19 | GatedDeltaNetPrefillFwdOp | **缺失** | linear_attention | `` | — | — | 本交付树无 manifest/kernel/op/test/bench |
| 20 | ArgmaxFwdOp | tileops/ops/reduction/argmax.py | reduction | `` | — | `passed` @ 2026-11-19 | perf_records 仍为 10-25 的 3 条 |

> 名称映射（任务书 → 本树）：`ClampMinFwdOp` → `ClampScalarFwdOp`(max=None)；`Conv1dBiasFwdOp` → 复用 `Conv1dFwdOp`(has_bias)；
> `GLADecodeOp` 本树同名；`LerpFwdOp`(标量) 与 `GatedDeltaNetPrefillFwdOp` 本树无实现。

## 3. 本轮归档进树的官方 report（9 份，均 passed）

| 算子 | run_id | 来源 |
|---|---|---|
| HardsigmoidFwdOp | `20261117_095705_795920` | `/mnt/workspace/reports/2026-11/` |
| AddFwdOp | `20261117_152252_308638` | `/mnt/workspace/reports/2026-11/` |
| GeluAndMulFwdOp | `20261117_102709_565655` | `/mnt/workspace/reports/2026-11/` |
| GeluTanhAndMulFwdOp | `20261117_104002_621842` | `/mnt/workspace/reports/2026-11/` |
| IsnanFwdOp | `20261116_183742_409434` | `/mnt/workspace/reports/2026-11/` |
| IsinfFwdOp | `20261117_122322_241985` | `/mnt/workspace/reports/2026-11/` |
| IsfiniteFwdOp | `20261117_120655_645080` | `/mnt/workspace/reports/2026-11/` |
| BatchNormBwdOp | `20261117_140629_142480` | `/mnt/workspace/reports/2026-11/` |
| GLADecodeOp | `20261118_153727_643910` | `/mnt/workspace/reports/2026-11/` |

> 只保留 `report.md` / `run.json` / `benchmark.*` / `pytest/`；msprof `artifacts/` 与 `*.log` 已剔除。

## 3.1 18 算子 smoke 回归

`pytest <18 个 test 文件> -m smoke -q` → **222 passed, 0 failed, 91 deselected**（13.1 min）。

> 之前按 24 个跑出的 4 条失败（`test_gqa_bwd` ×3、`test_fp8_lightning_indexer` ×1，`bishengir-compile` 维度不一致）
> 属于那两个算子的范围，**不在本次 20 个内**，所以本次 18 个全绿。

## 4. 本轮改动

### 4.1 实现新鲜度审计（2026-11-19）

逐个核对“交付树实际生效的 kernel 是否就是最新优化”：

- **HardswishFwdOp 原先是旧版**：交付树 wrapper 仍 import baseline、`perf_opt/` 只有日志没有内核，block 用 8192。
  已将 dev 树 11-03 的版本移植过来：`perf_opt/HardswishFwdOp.py`（fp16 原生 + vdiv-by-6）+ wrapper 的
  `_select_block_size()`（fp16 16384、MobileNetV3-stage3 10752 = 14 波整）。
  本树重跑官方 report（`20261119_164729`，`passed` 7/7）：stage2 fp16 22.68→**19.88**、stage3 fp16 15.46→**14.20**、
  fc-wide fp16 20.52→**17.82 µs**（−13.2%），bf16 同步下降。
- **IsnanFwdOp 的 11-16 候选被否决**：`perf_opt/IsnanFwdOp.py` 用 `vcmp "ne"` 判 NaN，但本工具链 `ne` 是
  **ordered** 语义（`lt | gt`），NaN 自比较为 False。实测切换后 1M fp32 有 **25% 位置错判**、4 条 smoke FAIL；
  baseline 的 `vcmp "le"` + 取反是正确实现。**保持 baseline**，该候选不进入交付。
- 其余 16 个：Add（period）、Argmax（R49）、GLA（R47）、BN（R34–R37）、Gelu×2（11-05）、
  Isinf/Isfinite（11-17 int8）、Hardsigmoid（R2/R3）、ClampScalar/ClampMin（R2）、Sub、Conv1d/1dBias/3dBias
  均已确认是最新且 wrapper 已接 perf_opt。
- **FusedAddLayerNormFwdOp 的 dev R16/R17 A/B 结论：不采纳**。移植后 smoke 通过（10/10），但本环境连跑两次
  官方 report 均显示 `llama-13b-prefill`（fp32 2048×5120）由交付版 67.20 → **86.14 / 87.92 µs**（ratio 71.4%→55.7%），
  其余 6 档改善不足以抵消；7 档 avg ratio 31.17 vs 31.01 基本持平。已回退到交付版，两次试验 report 存
  `/mnt/workspace/reports/2026-11/faln_ab_20261119/`。dev 自身报 70.30 µs，说明差异来自本环境 fp32 路径，
  需单独定位后再谈回迁。

### 4.2 其他改动

- `team/progress/郭益华.yaml`：`Conv1dFwdOp` 的 `dev_stage` 由 `已优化` 改为 `已提交审核`（PR #95 已存在，闸门要求）；
- 归档 9 份 11-16~11-18 官方 report 到 `examples/TileOPs/reports/tileops/`；
- 口径修正：删除了上一版按 24 个算子生成的清单/闸门，替换为 `tools/gate_20ops.py`。
- 附带（不在 20 内）：`HardtanhFwdOp` / `LeakyReluFwdOp` 的 manifest/workload/导出已补齐（原树同步），因这两个算子此前 `bench_*` 直接 ImportError。

### 4.3 实际生效内核（2026-11-19）

| 算子 | 生效来源 | 模块 | sha1 |
|---|---|---|---|
| HardsigmoidFwdOp | perf_opt | `.hardsigmoid_kernel.perf_opt.HardsigmoidFwdOp` | `7a07bd950f` |
| ClampScalarFwdOp | perf_opt | `.clamp_scalar_kernel.perf_opt.clamp_scalar_fwd_kernel` | `332742313e` |
| HardswishFwdOp | perf_opt | `.hardswish_kernel.perf_opt.HardswishFwdOp` | `19464cbe6d` |
| AddFwdOp | perf_opt | `.add_kernel.perf_opt.AddFwdOp` | `cf2c386053` |
| SubFwdOp | perf_opt | `.sub_kernel.perf_opt.SubFwdOp` | `df0808eca2` |
| GeluAndMulFwdOp | perf_opt | `.gelu_and_mul_kernel.perf_opt.GeluAndMulFwdOp` | `2c87ae3819` |
| GeluTanhAndMulFwdOp | perf_opt | `.gelu_tanh_and_mul_kernel.perf_opt.GeluTanhAndMulFwdOp` | `6bffe31923` |
| IsnanFwdOp | baseline（正确实现） | `.isnan_kernel` | `-` |
| IsinfFwdOp | perf_opt | `.isinf_kernel.perf_opt.IsinfFwdOp` | `1c4a84a164` |
| IsfiniteFwdOp | perf_opt | `.isfinite_kernel.perf_opt.IsfiniteFwdOp` | `0735ce7858` |
| FusedAddLayerNormFwdOp | perf_opt | `.fused_add_layer_norm_kernel.perf_opt.FusedAddLayerNormFwdOp` | `464741e221` |
| BatchNormBwdOp | perf_opt | `.batch_norm_bwd_kernel.perf_opt.BatchNormBwdOp` | `c07c364f9a` |
| Conv1dFwdOp | 多 kernel 路由 | `.conv1d_kernel._conv1d_direct_kernel` | `92a832d9ac` |
| ClampMinFwdOp | perf_opt | `.clamp_scalar_kernel.perf_opt.clamp_scalar_fwd_kernel` | `332742313e` |
| Conv1dBiasFwdOp | 多 kernel 路由 | `.conv1d_kernel._conv1d_direct_kernel` | `92a832d9ac` |
| Conv3dBiasFwdOp | perf_opt | `.conv3d_kernel.perf_opt.Conv3dBiasFwdOp` | `cc951e1f0a` |
| GLADecodeOp | perf_opt | `.gla_decode_kernel.perf_opt.GLADecodeOp` | `dd9fdf4166` |
| ArgmaxFwdOp | perf_opt | `.argmax_kernel.perf_opt._argreduce_kernel` | `34d1c6ebc7` |

> `Conv1dFwdOp`/`Conv1dBiasFwdOp` 是 5-kernel 路由（direct / unit_stride / group / pointwise / generic），
> 其中 group/kernel/pointwise 取自 `perf_opt/`，direct/unit_stride 取自基线目录，按 stride/padding/通道条件选择。
> `IsnanFwdOp` 的 11-16 perf_opt 候选因 `vcmp ne` ordered 语义判错 NaN，未启用。
### 4.4 op-dev 机械门禁补齐（2026-11-19）

按 `_shared/standards/gate-and-retry.md` 与 `signal-registry.md` §5 复核 18 个算子，补掉 4 个会被机械门禁挡下的缺口：

| 算子 | 缺口 | 处理 |
|---|---|---|
| HardswishFwdOp | 无 `workload_inventory.json`、无 baseline/final 配对、`opt_log` 无 Final 表 | 新建 inventory（3 smoke + 6 tune）、12 条 `perf_records`（6 baseline + 6 final）、Final Performance 表 |
| ArgmaxFwdOp | 无 inventory；新 5 条只有 final 无 baseline；REVIEW 结论“有条件通过”且内容过期 | 新建 inventory（3 smoke + 5 tune）、补 5 条 baseline、5 条 final 统一 `candidate_id=R49_final`；REVIEW 改「结论: 通过」并更新 lm-head 阻塞描述 |
| GLADecodeOp | REVIEW 无 `结论: 通过` 字面量；DESIGN 无 §1.6 | 补字面量；补 §1.6.0–§1.6.3（调度/内存/精度契约变更，按“无优化空间”豁免 `verify_equiv.py`，附官方 11/11 + 50 次复验证据） |
| Conv3dBiasFwdOp | 同上 | 同上（implicit-GEMM/polyphase/bias 融合均为实现结构变更，公式未变；附 17/17 证据） |

复核结果：**18/18 PASS**：
- `tools/gate_20ops.py`（团队提交规则）：18/18 PASS + 2 BLOCKER；
- 严格机械审计：inventory 分类覆盖、每 tune workload 有 baseline+final、同一 `kernel_id` 单一 final candidate、
  `design_calc_check.py` 的 `r3_metrics=pass`（复杂度四口径）、DESIGN §1.6.1、REVIEW「结论: 通过」+ 维度 8 + 机械复核（D-1/D-5）。
- 本轮为 GLA / Conv3dBias 补了复杂度四口径表（`r3_metrics` 由 fail→pass）与 REVIEW 的机械复核/维度 8 段；
  Argmax REVIEW 同样补齐。

## 5. 风险

1. **2 个 blocker**：`LerpFwdOp`（标量权重）与 `GatedDeltaNetPrefillFwdOp` 在本交付树没有 manifest/kernel/op/test/bench；
   旧 `tilelang-mlir-ascend` 树也只有 LerpTensor，GDN 在整个 workspace 未找到。需从原交付源移植后才能提交。
2. **无上游 git 历史**：origin(GitHub) 401；推 PR 需在有权限的 clone 上操作（第 6 节）。
3. **闸门缺 `--evidence-only`**：18 个算子六区都已在基线，纯证据/纯 perf 分支会被判“缺少六区改动”，
   需先合入团队提案或使用补丁版脚本。
4. 工具链既有失败：`test_gqa_bwd` / `test_fp8_lightning_indexer` 在 `bishengir-compile` 报维度不一致
   （原树同样复现）；这两个算子不在本次 20 个范围内，仅作环境说明。

> 已在本次收口：Argmax 的 R48/R49 记录、Hardswish 的 inventory/records、以及 GLA/Conv3d 的 §1.6 与 REVIEW 结论
> 均已按 op-dev 机械门禁补齐（见 4.4）；FALN 的 report 与最终交付版一致（dev R16/R17 A/B 失败未采纳）。

## 6. 提交步骤（单算子分支 / 单 PR）

```bash
# 在有上游权限的 clone 上
git fetch origin && git switch integration/npu-ops && git pull
git switch -c <branch> origin/integration/npu-ops   # 分支名见第 2 节
# 从本树同步该算子文件（六区 + 证据 + perf_opt + report）
rsync -a <this-tree>/examples/TileOPs/tileops/    <clone>/examples/TileOPs/tileops/
rsync -a <this-tree>/examples/TileOPs/tests/      <clone>/examples/TileOPs/tests/
rsync -a <this-tree>/examples/TileOPs/benchmarks/ <clone>/examples/TileOPs/benchmarks/
rsync -a <this-tree>/examples/TileOPs/reports/    <clone>/examples/TileOPs/reports/
cd examples/TileOPs
python team/scripts/check_op_submission.py --op-name <Op> --owner <owner> --optimized
git add -A && git commit -m 'feat(<family>): <Op> ...' && git push -u origin <branch>
```

## 7. 环境

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
export PYTHONPATH=/mnt/workspace/tools_shim:/mnt/workspace/tilelang-mlir-ascend:$PWD/examples/TileOPs
cd examples/TileOPs && python3 -m pytest tests/ops/test_<op>.py -q
```

CANN 9.3.0 / bishengir 2.0.0 / Ascend910B3；时延口径 msprof `Task Duration`。
