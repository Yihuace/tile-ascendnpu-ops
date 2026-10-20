# 迁移索引（MIGRATION_INDEX）

> TileOPs GPU→NPU 迁移总览。算子清单见《TileOPs迁移任务书.md》§5/§6；此处记录**状态 / 正确性 / 性能 / 报告指针 / 版本戳**。
> 维护约定：每完成一个算子（Stage 5 通过）更新本表一行；一批完成生成统一横评。
> **2026-10-18 修订**：CANN 9.0 重定基线下，全量 `tileops-report run --all`（约 4.5 h）判定为与 optimize 阶段的基线采集**重复劳动**，
> 改以**细粒度快扫**（每算子取元素量最小/最大两个非 smoke workload × 全 dtype）承接「9.0 是否打坏性能」这一唯一不可后补的问题，见 `docs/cann9_adaptation.md` §12–§13。

## A. 本批待迁（任务书 §5/§6，**19 接口全量交付**；交付名 = 任务书原名）

> **口径（2026-10-18 收尾）**
>
> - **正确性列 = CANN 9.0.0 重跑值**（工作流②，`docs/cann9_adaptation.md` §11，**逐文件串行**）。表 A 合计 **276 / 282 passed**：
>   `Conv1dFwdOp`（1/28 失败）与 `Conv1dBiasFwdOp`（5/28 失败）的失败用例**全部**为卷积族工具链缺陷
>   **CG-2026-0003**（`bishengir-compile` 带 LLVM crash backtrace 的 SIGSEGV），已按**工具链绑定 xfail**
>   （`tests/conftest.py`，仅 `CANN==9.0.0` 生效、`strict=False`，修好即 XPASS）处理；
>   故下表这 2 行**仍标 上一代工具链 全通过值**，9.0 下的有效通过数与失败清单以 §11.1 为准。
>   其余 **16 个已交付算子**在 9.0 与 上一代工具链 下均全通过；`GatedDeltaNetPrefillFwdOp` in-scope 主路径已交付（10/10）；K5/CP 待续。
>
> - **性能 ratio 列 = 上一代工具链 口径**（未标注行）。9.0「迁移态」快扫偏差见 §13（18 算子 / 77 用例：**中位 1.64%**、P90 10.13%，
>   56% 用例 ≤2%，**无系统回退**）。按 §13.6 判定：**大 shape 与 fp16/bf16 行可照用 上一代工具链 值**（±2%）；µs 级小用例仅作量级参考（±10%）；
>   例外需修正后再用 —— `BatchNormBwd` large-spatial/fp16 **+22.7%**、clamp 家族 fp32 **+7~8%**。
>
> - 9.0.0 正式报告（3 份，`20261017_*`）：`HardsigmoidFwdOp` / `ClampScalarFwdOp` / `HardswishFwdOp`（路径见下方各行报告指针）。
>
> 工具链版本戳见 §C 与 `versions.txt`。

| 算子 | family | 批次 | 状态 | 正确性 | 性能 ratio（9.0：迁移态 → 交付态） | 报告指针 |
|---|---|---|---|---|---|---|
| `HardsigmoidFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 7/7, max_abs_err 5.96e-08 | 55.6–59.7% → 0.37–61.52%（0/2 达标） | `examples/hardsigmoid/MIGRATION_REPORT.md`；run `reports/tileops/20261012_223321_664388_HardsigmoidFwdOp/` |
| `ClampScalarFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`） | 11/11, max_abs_err 0.00e+00 | 45.1–64.4% → 58.2–70.2%（1/11 达标） | `examples/TileOPs/reports/tileops/20261013_141507_226988_ClampScalarFwdOp/` |
| `HardswishFwdOp` | elementwise | 1a | ✅ **已完成**（harness, `phase=DONE`） | 7/7, max_abs_err 4.77e-07 | 50.2–61.0% → 50.2–61.0%（2/6 达标） | `examples/TileOPs/reports/tileops/20261013_150641_447235_HardswishFwdOp/` |
| `AddFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 24/24, max_abs_err 0.00e+00 | 69.5–82.3% → 69.4–83.5%（10/11 达标） | `examples/TileOPs/reports/tileops/20261013_162043_457669_AddFwdOp/` |
| `SubFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 24/24, max_abs_err 0.00e+00 | 68.1–83.4% → 70.7–84.0%（15/15 达标） | `examples/TileOPs/reports/tileops/20261013_170400_275674_SubFwdOp/` |
| `GeluAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`；1 轮设计修订） | 11/11, max_abs_err 1.43e-06(fp32)/0.0312(bf16) | 34.4–44.0% → 4.05–47.71%（0/4 达标） | `examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/` |
| `GeluTanhAndMulFwdOp` | elementwise | 1b | ✅ **已完成**（harness, `phase=DONE`） | 11/11, max_abs_err 0(fp32)/1.56e-02(bf16) | 23.3–49.7% → 2.22–47.63%（0/4 达标） | `examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/` |
| `IsnanFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`；1 设计修订） | 9/9, max_abs_err 0.00e+00 | 51.1–77.3% → 52.9–79.3%（1/5 达标） | `examples/TileOPs/reports/tileops/20261014_093134_398669_IsnanFwdOp/` |
| `IsinfFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 41.7–71.9% → 41.5–78.1%（1/5 达标） | `examples/TileOPs/reports/tileops/20261014_101823_741698_IsinfFwdOp/` |
| `IsfiniteFwdOp` | elementwise | 1c | ✅ **已完成**（harness, `phase=DONE`） | 9/9, max_abs_err 0.00e+00 | 41.3–73.0% → 42.3–78.3%（1/5 达标） | `examples/TileOPs/reports/tileops/20261014_105757_589000_IsfiniteFwdOp/` |
| `FusedAddLayerNormFwdOp` | norm | 2 | ✅ **已完成**（harness, `phase=DONE`） | 14/14, max_abs_err 1.56e-02(bf16) | 27.1–45.1% → 0.54–43.89%（0/5 达标） | `examples/TileOPs/reports/tileops/20261014_115534_389335_FusedAddLayerNormFwdOp/` |
| `BatchNormBwdOp` | norm | 3 | ✅ **已完成**（harness, `phase=DONE`；3 kernel: wide/split/general） | 12/12, avg max_abs_err 2.29e-03 | 15.3–28.2% → 11.9–24.4%（0/6 达标） | `examples/TileOPs/reports/tileops/20261015_162156_553625_BatchNormBwdOp/` |
| `Conv1dFwdOp` | convolution | 3 | ✅ **已完成**（harness, `phase=DONE`；5 kernel） | 28/28, avg max_abs_err 3.91e-02 | 0.07–0.41% → 0.04–1.79%（0/15 达标） | `examples/TileOPs/reports/tileops/20261015_162520_967795_Conv1dFwdOp/` |
| `ClampMinFwdOp` | elementwise | 1c | ✅ **已完成**（薄接口；复用 ClampScalar kernel，max=None；fp32 block_size 修正） | 14/14, max_abs_err 0.00e+00 | 49.4–65.5% → 61.8–68.8%（0/7 达标） | `examples/TileOPs/reports/tileops/20261014_214827_195753_ClampMinFwdOp/` |
| `LerpFwdOp`（标量权重） | elementwise | 1c | ✅ **已完成**（薄接口；新 npuir `lerp_scalar` 二元 kernel） | 36/36, fp16≤1.95e-3 / bf16≤1.56e-2 | 60.2–82.0% → 60.2–82.0%（9/11 达标） | `tests/ops/test_lerp_scalar.py` |
| `Conv1dBiasFwdOp` | convolution | 3 | ✅ **已完成**（薄接口；复用已迁 Conv1d 5 kernel 的 has_bias 路径） | 28/28, max_abs_err 2.50e-01(bf16) | 0.07–0.46% → 0.03–1.73%（0/8 达标） | `tests/ops/test_conv1d_bias.py` |
| `Conv3dBiasFwdOp` | convolution | 3 | ✅ **已完成**（新增 dense NCDHW implicit-GEMM + MixCV bias kernel；stride/dilation/pad 通用，W 轴 polyphase） | 17/17, max_abs_err 3.125e-02(fp16 r3d-stem) / 5.00e-01(bf16 unet-encoder) | 无正式迁移态（旧 run 无 msprof ratio；unet 0.726% 仅反推估算）→ 优化 1.68–20.43% | `examples/TileOPs/reports/tileops/20261018_185919_687302_Conv3dBiasFwdOp/` |
| `GLADecodeOp` | linear_attention | 4 | ✅ **已完成**（Vector kernel；K-tile 流式 state，bf16 fp32 中转） | 11/11, worst max_abs_err 1.51e-02(bf16) | 3.17–18.09% → 3.1–20.16%（0/5 达标） | `examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/` |
| `GatedDeltaNetPrefillFwdOp` | linear_attention | 4 | ✅ **in-scope 主路径已交付**：K1 cumsum + K2 split blocksolve + K6 split forward（K=V∈{64,128}，含尾 chunk）；K5/CP 仍 pending，fallback 保留 | **10/10**（`tests/ops/test_gdn.py`，含 in-scope + tail） | 0.03% → 0.36%（0/1 达标） | `tests/ops/test_gdn.py`；`gdn_split_*.py`；`GatedDeltaNetPrefillFwdOp/OPTIMIZE_REPORT.md` |

> 基线名映射（**交付名 = 任务书原名**；基线 commit `1238a9f` 注册名为映射参考，非交付名）：`ClampMinFwdOp`→`ClampScalarFwdOp`(max=None)；`LerpFwdOp`→`LerpScalar`/`LerpTensor`；`Conv1dBiasFwdOp`/`Conv3dBiasFwdOp`→基线 conv1d/conv3d（bias 可选）；`GLADecodeOp`→`GLARecurrentFwdOp`；`GatedDeltaNetPrefillFwdOp`→`GDNFwdOp`。裸 `Conv3dFwdOp` **不在 19 内**（任务书 §5）。

## B. 此前已完成（不在本批清单）

来源：`examples/TileOPs/reports/report.md`（统一 run `20261009_022152_829230_all`，环境 910B2C / CANN 上一代工具链 / git `84519be`）。

> ⚠️ **2026-10-17 决策：表 B 退出本次 CANN 9.0 重定基线范围。**
> 这 7 个算子**不属于本批迁移/优化目标**，后续阶段（②正确性 / ③性能 / ④记录）**一律不再纳入**，除非另行下令；
> 本节数据**维持 上一代工具链 口径不动**（不做 9.0 重测）。仅存档观察到的事实：
> `SSDChunkScanFwdOp`（test_mamba）在 9.0 下 `bishengir-compile` **挂死**（CPU 空转 >19min）、
> `MultiHeadAttentionFwdOp` 按此决策跳过、`MishFwdOp` 7/7 ✅。
> 详见 `docs/cann9_adaptation.md` §11.2。
> **版本戳收尾（工作流④）对本节不做 9.0 更新**——本节数据维持 上一代工具链 口径存档，不重测、不重写。

| 算子 | 正确性 | avg max_abs_err | shapes | ratio 范围 |
|---|---|---|---|---|
| `AdaLayerNormFwdOp` | 22/22 | 1.13e-02 | 5 | 0.41%–69.38% |
| `ArgmaxFwdOp` | 42/42 | 0.00e+00 | 5 | 5.44%–41.39% |
| `LerpTensorFwdOp` | 8/8 | 5.21e-08 | 7 | 67.38%–80.61% |
| `LogSumExpFwdOp` | 23/23 | 3.67e-08 | 6 | 3.23%–36.61% |
| `MishFwdOp` | 7/7 | 2.45e-05 | 6 | 54.01%–64.87% |
| `MultiHeadAttentionFwdOp` | 6/6 | 上一代工具链4e-04 | 8 | 14.34%–23.44% |
| `SSDChunkScanFwdOp` | 4/4 | 2.19e-04 | 11 | 0.13%–20.30% |

## C. 版本戳与状态文件

### C.1 工具链版本戳（机器可读副本：`versions.txt`）

| 项 | 版本 | 备注 |
|---|---|---|
| **CANN** | **9.0.0** | `bishengir-compile 1.1.0`（`428ab8fdab46`，2026-04-30）/ `hivmc 0.2.0`（`3af4c111df44`） |
| tilelang | `0.1.2` | 自建 dev root：新容器 `/mnt/workspace/tilelang-mlir-ascend`（旧容器 `/home/developer/tilelang-mlir-ascend`）@ `9309edb0`（2026-10-15） |
| AscendNPU-IR | `31f69036` | 子模块，`legacy-interface-342-g31f690369`（2026-01-14） |
| torch / torch_npu | `2.7.1+cpu` / `2.7.1.post4` | Python `3.11.4` |
| 设备 | Ascend910B3 | `ASCEND_VISIBLE_DEVICES=6`；`get_aicore_num()` 实查 **20**（纯 Vector 场景 ×2 = 40）；UB 192 KB |

- **本地补丁（dev root 内，not upstream）**：`tilelang/jit/jit_npu.py` 的 CANN9 `syn_instr_mode` strip shim（见 `docs/cann9_adaptation.md` §3），备份 `jit_npu.py.bak_cann9`。
- **取戳陷阱**：`tilelang.__version__` 的 `+<sha>` 后缀取自 **cwd 的 git**（非 tilelang 自身），取戳请以本表 / `versions.txt` 的 `commit` 字段为准。

### C.2 数据口径（2026-10-18 收尾）

- **正确性 = 9.0.0 口径**：工作流② 已完（`docs/cann9_adaptation.md` §11）——表 A 19 算子**逐文件串行**重跑，**276/282 passed**；
  6 个失败全部为卷积族 CG-2026-0003（已工具链绑定 xfail）；表 B 7 算子**退出**本次重定基线范围（§11.2）。
- **性能 = 混合口径**：工作流③ 以「细粒度快扫」**替代**全量 `tileops-report run --all`（§12：全量约 4.5 h，判定为与 optimize 基线采集重复劳动），
  完成 **18 算子 / 77 用例**（全量非 smoke 的 31%）「迁移态」偏差复核（§13：中位 1.64%、P90 10.13%，**无系统回退**）。
  9.0.0 正式报告 3 份（`HardsigmoidFwdOp` / `ClampScalarFwdOp` / `HardswishFwdOp`，`reports/tileops/20261017_*`）；
  其余算子按用户口径「**能衔接 上一代工具链 的一律衔接**」沿用 上一代工具链 ratio，例外见 §13.6（`BatchNormBwd` +22.7%、clamp 家族 fp32 +7~8%）。
- **取基线必先定「层」**（§13.2）：必须取**迁移态**而非「最新 run」（优化态），否则 `Hardsigmoid` 会得到 −42% 的**假偏差**（真值 −1.4%）。
- **未闭合项**：① ~~T6 ——「删 NaN 恢复」优化（`GeluAndMul`）在 9.0 下的重验~~ → **已关闭（2026-10-19）**：
  9.0 上假设成立（kernel 自检 `[Boundary] inf/nan` 全 PASS）、增益仍在（−8.8%，`opt_log.md §1`），
  并新增仓库级守卫 `tests/ops/test_gelu_and_mul.py::test_gelu_and_mul_special_values`（反向对照已验证可失败），见 `docs/cann9_adaptation.md` §12.5；
  ② §13.6 两条新异常立项（`BatchNormBwd` large-spatial/fp16、clamp 家族 fp32）；③ `GatedDeltaNetPrefillFwdOp` in-scope 主路径已交付（10/10）；K5/CP 待续。

- 状态机：`python3 .agents/tools/statectl.py show --dir examples/<op_slug>`；op 级 `migration show --dir examples/<op_slug>`。
- 报告口径：`examples/TileOPs/reports/report.md`（多算子统一 run）vs `examples/TileOPs/reports/tileops/<run_id>/`（单算子）。
- ~~待办缺口：顶层 `docs/` 缺失（skill 引用 `docs/开发指南.md`/`docs/Tilelang.language/` 不存在）→ `standards_check` 存量 SC-REF-PATH；补入后应转绿。~~
  → **已闭合**：本分支 `docs/` 已补齐（`开发指南.md` / `快速入门.md` / `安装指南.md` / `Tilelang.language/` 等）。
