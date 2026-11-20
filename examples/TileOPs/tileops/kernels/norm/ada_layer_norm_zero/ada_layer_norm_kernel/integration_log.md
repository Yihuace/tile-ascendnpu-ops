# Integration Log — `AdaLayerNormKernel`(覆盖 `AdaLayerNormFwdOp` 与 `AdaLayerNormZeroFwdOp`)

- **meta**:`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/.migration_meta.json`
- **integrated**:`_ada_layer_norm_kernel` → `tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py`(源自 `examples/ada_layer_norm/_ada_layer_norm_kernel/_ada_layer_norm_kernel.py`)
- **design_docs**:`_ada_layer_norm_kernel` → `tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md`(**未缺失**,补齐快照后重跑带上)
- **wrapper**:`tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py` —— `wrapper_rewritten: false`(**未重写**,见下)
- **ascend_mode**:`Developer`(**以环境变量形式生效**)
- **attempts**:1(集成侧首次即完成;**上游 Stage 3 零回退一次通过**)
- **smoke**:pass

## ⚠️ 关键披露:report 实际测的是 `perf_opt`,不是本次集成的 Stage 3 baseline

本算子的 wrapper 采用**双 import 切换块**(注释原文:"Exactly one source block below is active; toggle by swapping the comment")。集成后的实际状态是:

```
# --- baseline (Stage 3) ------------------------------------------------------
# from .ada_layer_norm_kernel import ALIGNMENT, _ada_layer_norm_kernel, _align_up
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import (
    ALIGNMENT, _ada_layer_norm_kernel, _align_up,
)
from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import (
    select_row_config as _select_row_config_tuned,
)
```

⇒ **生效的是 `perf_opt`(Stage 4 已调优变体),baseline 被注释掉。**

**为什么没有翻转开关(处置依据)**:wrapper 自己写明的策略是

> "Default policy: the perf_opt source becomes active once tuned drop-in kernels (same factory signatures) land at `.ada_layer_norm_kernel/perf_opt/{func}.py` **and pass their L0/L1 regression**; the baseline (Stage 3) source is active otherwise."

`perf_opt/_ada_layer_norm_kernel.py` **已存在**(且带 `perf_opt/opt_log.md`,来自早先手工阶段),且处于激活态 ⇒ 按该策略,它**已满足"落地 + 通过回归"**。**翻转会把一个已调优且已验证的变体降级**,不属于本次集成该做的改动。故:**不翻转,但如实记录。**

**由此产生的口径限定**:

| 项 | 由什么验证 |
|---|---|
| **本次集成的 Stage 3 baseline** | 由 **Stage 3 自带分层套件**验证:L0 **6/6**(两变体 × 3 dtype)、L1 **20/20**、L2 4/4、Boundary 16/16,零失败 |
| **集成侧 report 的 22/22 与 14/14** | 走的是**生效的 `perf_opt` 路径**,**不是** baseline |

⇒ **不得**把 report 的通过数当作"baseline 已在集成侧被验证"。两者是**不同的代码路径**。

## 验证结果(两 Op 的 report,均为今天 **20261016** 的采集)

| Op | report `run.json` | `status` | 正确性 | benchmark 失败数 |
|---|---|---|---|---|
| `AdaLayerNormFwdOp` | `reports/tileops/20261016_062902_893752_AdaLayerNormFwdOp/run.json` | **passed** | **22 / 22** | **0** |
| `AdaLayerNormZeroFwdOp` | `reports/tileops/20261016_064033_284305_AdaLayerNormZeroFwdOp/run.json` | **passed** | **14 / 14** | **0** |

### ⚠️ 一处采集事故(如实记录)

`AdaLayerNormZeroFwdOp` 的**首次采集**因上一轮会话被中断而**从未启动**(日志为空、无目录产出)。**重跑后完成**。因此**只认今天(20261016)目录下的 run.json** —— 仓库里存在 `reports/tileops/20261015_124806_589781_AdaLayerNormZeroFwdOp/`(**昨天**的旧 run),**不得**用它的 `14/14` 冒充本轮结论(两者数值巧合相同,但来源不同)。

### benchmark 说明

两个 Op 的 `benchmark_failed` 均为 **0**,`status=passed`。norm 族的 manifest 不含 268M 量级 workload(本算子最大 `(2048, 4096)`),故逐元素族那两条采集侧失败不出现。

## 调试历史

集成侧无调试(attempt 1 即完成)。**上游 Stage 3 亦零回退**(首次运行即 L0 6/6)。

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`;CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`;profiling mode:msprof

## 集成侧测试覆盖缺口

`tests/ops/test_ada_layer_norm.py` / `test_ada_layer_norm_zero.py` 不含**容量护栏行为**、**核内串行分支**、**`rstd` 精度**等类别(`tests/` 不在本 Agent 的可改范围内)。该缺口登记于 `examples/ada_layer_norm/RETROSPECTIVE.md`。

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**:harness.md §7.1 要求基于 `opencode export <session_id>` 生成,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**。按"无法确认时如实标注、不得伪造"的口径,**未伪造任何导出数据**,交由 conductor 处置。
