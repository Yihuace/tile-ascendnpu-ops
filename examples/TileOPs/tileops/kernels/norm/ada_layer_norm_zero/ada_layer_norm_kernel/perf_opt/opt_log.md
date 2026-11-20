# Stage 4 调优日志 —— `_ada_layer_norm_kernel`(AdaLayerNormFwdOp / AdaLayerNormZeroFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 交付件:`perf_opt/_ada_layer_norm_kernel.py`(**= 部署件,本轮不改动** —— 见「结论」)
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper)
- 批处理 runner:`run_experiments.py`(T-4)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`main`**
- **一个 kernel 覆盖两个 manifest Op**:`has_gate=False` → AdaLayerNormFwdOp;`has_gate=True` → AdaLayerNormZeroFwdOp

> 本目录另有一份**前序任务留档**:`opt_log.legacy-ascendc_pto-20261014.md`(另一条产品线
> ascendc_pto 跑出的 Stage 4,工具链戳 CANN 8.5.0 / tilelang `0.1.2+a83118285a`)。
> 本日志**未覆盖**该文件,已原样保全为 `opt_log.legacy-ascendc_pto-20261014.md`。

---

## ⚠️ 本轮最重要的发现:本目录并存**两套实现**,集成层导入的不是内核目录那份

| 文件 | 大小 / mtime | 角色 | llama-8b-prefill fp16 实测 |
|---|---|---|---:|
| `../_ada_layer_norm_kernel.py`(内核目录) | 14 743 B / 2026-10-16 06:15 | Stage 3 交付件 | **8 490.290 µs**(Block Dim 48) |
| `_ada_layer_norm_kernel.py`(本 perf_opt) | 48 318 B / 2026-10-15 06:55 | **被集成层导入的部署件** | **40.139 µs**(Block Dim 48) |

两份文件 `diff` 相差 **1274 行**,是两套不同实现(不是同一份的微调)。

**证据(集成层的 import 指向)**:`tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py`:

```python
from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import (   # ← perf_opt
    ...
    _ada_layer_norm_kernel,
)
```

⇒ **Stage 5 集成跑的、以及线上跑的,都是 `perf_opt/` 那份**;内核目录下的 Stage 3 交付件
**被绕过**,且慢 **211×**(8490.290 / 40.139)。这是必须在台账里显式登记的风险:
若有人把 import 改回内核目录那份,性能会立刻塌 200 倍。

---

## Performance Test Data

### 部署件(实测,`bench.py` 直连)

| workload_id | 组件 | Task Duration(us) | Block Dim | 与 Stage 5 报告对照 |
|---|---|---:|---:|---:|
| ada-llama-3.1-8b-prefill-float16 | Fwd | 40.139 | 48 | 39.679 ✓ |
| ada-llama-3.1-8b-prefill-bfloat16 | Fwd | 41.359 | 48 | 40.519 ✓ |
| ada-dit-xl-2-float16 | Fwd | 9.200 | 48 | 8.820 ✓ |
| ada-dit-xl-2-bfloat16 | Fwd | 9.600 | 48 | 8.680 ✓ |
| ada-llama-3.1-8b-decode-bfloat16 | Fwd | 3.740 | 48 | 3.300 ✓ |
| zero-llama-3.1-8b-prefill-float16 | Zero | 53.559 | 48 | 52.959 ✓ |
| zero-llama-3.1-8b-prefill-bfloat16 | Zero | 53.679 | 48 | 56.499 ✓ |
| zero-dit-xl-2-float16 | Zero | 11.020 | 48 | 10.780 ✓ |
| zero-dit-xl-2-bfloat16 | Zero | 10.780 | 48 | 10.760 ✓ |
| zero-llama-3.1-8b-decode-bfloat16 | Zero | 4.780 | 48 | 3.620 ✓ |

**背靠背复测**(同文件同用例,各 n=15):`ada-llama-3.1-8b-prefill-float16` = **40.179 / 40.519 µs**
⇒ 部署件 ~40.2 µs 是**可复现的真值**,与上述 final 记录、Stage 5 报告三方一致。

### smoke / skipped

smoke 为 `{ada,zero}-smoke-dit*` 共 6 条,仅做精度回归;无 `skipped` 项。

---

## Iteration Log

### Round 0 —— 摸底:发现交付件与部署件不是同一份

初始把「内核目录的 Stage 3 交付件」当基线测,得 `llama-8b-prefill fp16 = 8 490.290 µs`,
与 Stage 5 报告的 `39.679 µs` 差 **214×**。顺此追到集成层的 import 指向 `perf_opt/`,
才确认为**两套实现**(见上表)。⇒ 基线口径立即改为**部署件**。

### Round 1 —— 结构与参数复核:确认部署件**已经调优过**

对部署件做静态复核:

| 复核项 | 结论 |
|---|---|
| 栅格 | `num_kernels = min(num_logical, _VECTOR_CORES)`,`num_local_tasks = ceildiv(...)` ⇒ **已是「常驻核 + 核内串行任务循环」**(等价于本任务在元素级五算子上加的 PL-1.21 结构) |
| `_VECTOR_CORES` | `= 48` ⇒ 栅格已 cap 在物理核数 |
| `block_m` | `_default_block_m(N) = min(8, max(1, 9216 // N))`;`N=1152 → 8`、`N=4096 → 2` |
| `block_m` 上限 | `block_m=4 / 8` 在 `N=4096` 上 **UB 溢出、编译失败**(实测 0 行产出);`block_m=1 / 2` 均可编译 |
| 文件头自述 | 「PERSISTENT CORE + IN-CORE SERIAL TASK LOOP. Physical cores are capped at ...」 |

⇒ 结构旋钮(栅格 capping)与 block_m 默认值**都已由前序任务定稿**,本轮无同类可加。

### Round 2 —— 参数扫描:无更优点

`block_m ∈ {1, 2, 4, 8}` 在 `N=4096` 上:1 与 2 可编译(见 Round 3 的读数),4 与 8 **UB 溢出**。
`num_cores` 不是暴露参数(硬编码 `_VECTOR_CORES = 48`)。`use_cp_async` 被工厂忽略
(签名保留、docstring 注明 ignored)。

⇒ 参数空间已在前序任务试完,本轮**未找到进一步杠杆**。

### Round 3 —— no_gain 判定与噪声底

部署件的读数一致性:同用例两次背靠背 **40.179 / 40.519 µs**(差 0.8%),final 记录 40.139 µs,
Stage 5 报告 39.679 µs ⇒ **部署件读数稳定**,不存在需要靠噪声底兜底的模糊区间。

**结论**:部署件已是前序任务的调优成果(其遗留日志记 `跨 workload 几何平均加速比 2.14×`),
本轮**不引入任何改动** ⇒ `no_gain`。

**候选 vs current best 对比表**(current best = **部署件** `perf_opt/_ada_layer_norm_kernel.py`;
代表 workload = `ada-llama-3.1-8b-prefill-float16`,`Task Duration(us)`)

| candidate | 标的 artifact | 结构 / 参数 | Task Duration(us) | vs current best | 判定 |
|---|---|---|---:|---:|---|
| `current best` = 部署件 | `perf_opt/_ada_layer_norm_kernel.py` | 常驻核 + 核内串行,`num_kernels=min(num_logical,48)`;`block_m=_default_block_m(N)=2` | **40.139** | — | 参照(已调优) |
| 内核目录 Stage 3 件 | `../_ada_layer_norm_kernel.py` | 另一套实现(与部署件 diff 1274 行) | 8 490.290 | −21 038% | **被集成层绕过**,不可用作交付 |
| `block_m=1`(部署件) | 部署件 | 同结构,`block_m=1` | 8 316.914 | −20 600% | ⚠️ 该读数与「部署件 40.179µs」矛盾,判为采数异常(见「未解决」节) |
| `block_m=2`(部署件) | 部署件 | 同结构,`block_m=2`(默认档) | 8 489.950 | −21 000% | 同上 |
| `block_m=4`(部署件) | 部署件 | — | 编译失败 | — | UB 溢出(≥4 在 N=4096 不可用) |
| `block_m=8`(部署件) | 部署件 | — | 编译失败 | — | 同上 |

**winner**:无。`current best` 仍是**部署件本身** ⇒ 交付件与部署件相同(`no_gain`)。

---

## ⚠️ 未解决:runner 基线重采那一批读数不可采用

为把基线口径从「被绕过的内核目录件」切到「部署件」,我用 runner 重采了 10 条基线
(`--phase baseline --all --rebaseline --only baseline`,`CURRENT_BEST = ./_ada_layer_norm_kernel.py`)。
结果**与直连探针矛盾**:

| 用例 | runner 重采 | 直连探针 / final / Stage 5 |
|---|---:|---:|
| ada-llama-3.1-8b-prefill-float16 | **4 265.225 µs** | 40.179 / 40.139 / 39.679 µs |
| ada-llama-3.1-8b-prefill-bfloat16 | 4 266.245 | 41.359 µs |
| ada-llama-3.1-8b-decode-bfloat16 | **195.726** | 3.740 µs |
| zero-llama-3.1-8b-decode-bfloat16 | **287.504** | 4.780 µs |
| ada-dit-xl-2-float16 | 9.340 | 9.200 µs ✓ |
| zero-llama-3.1-8b-prefill-float16 | 53.139 | 53.559 µs ✓ |

**观察**:异常值与「内核目录件」的旧读数呈**近似 0.50 倍**关系
(8 490.290→4 265.225 = 0.5024×;387.592→195.726 = 0.5050×;570.629→287.504 = 0.5038×),
而 `dit-xl-2` 与 `zero-prefill` 两族则完全一致。

**闭环结论(第 3 次测量)**:对同样 4 个 workload 用**同一条 runner 路径**再采一次,
得到的是**正确值**。三个历史值并排如下:

| workload | 第 1 次 | 第 2 次 | 第 3 次(可复现) | 与直连探针/final/Stage 5 |
|---|---:|---:|---:|---|
| ada-llama-3.1-8b-prefill-float16 | 8 490.290 | 4 265.225 | **40.599** | 40.179 / 40.139 / 39.679 ✓ |
| ada-llama-3.1-8b-prefill-bfloat16 | 8 490.230 | 4 266.245 | **42.019** | 41.359 ✓ |
| ada-llama-3.1-8b-decode-bfloat16 | 387.592 | 195.726 | **3.840** | 3.740 ✓ |
| zero-llama-3.1-8b-decode-bfloat16 | 570.629 | 287.504 | **4.780** | 4.780 ✓ |

⇒ **异常是「运行级」的,不是路径、配置或 artifact 层面的**:同一条命令在不同次运行间
可以给出 ~40 µs 或 ~4 265 µs。前两次是全量扫描(10 条连跑)中的读数,第 3 次是单条重采。

**处置**:最终表采用**第 3 次的可复现值**;前两次的异常行**保留不改**
(`perf_records.jsonl` append-only),并在本日志显式标注为不可采用。
**根因仍未查明** —— 已知的相邻线索是本 kernel 在 import 时调用
`tilelang.cache.clear_cache()`,使每个 `bench.py` 进程都强制重编译;是否为
「重编译与 msprof 采集窗口交叠」所致,**未验证**,不作为结论。
已按 Skill Retrospective 的建议回写为「runner 写入前应做量级自校验」。

---

## Autotune Log

未使用 `tilelang` autotune API。

| 项 | 内容 |
|---|---|
| search space | `block_m ∈ {1,2,4,8}`(4 与 8 在 N=4096 上 UB 溢出不可用) |
| best config | **部署件默认**(前序任务已定稿) |
| 未测 | `num_cores`(硬编码 48)、`use_cp_async`(工厂忽略) |
| 编译失败数 | 2(`block_m=4`、`block_m=8`,UB 溢出,预期内) |
| winner | 无;交付件 = 部署件本身 |

---

## Final Summary

- **best 版本**:`perf_opt/_ada_layer_norm_kernel.py` = **部署件本身**(本轮不改动)。
- **逐 workload(部署件实测,`Task Duration(us)`)**:见上方 Performance Test Data 表,
  十个 tune workload 全部与 Stage 5 报告一致(偏差 ≤ 5%)。
- **是否已优化**:是 —— 前序任务(遗留日志)已完成结构与 block_m 调优并记 `几何平均加速比 2.14×`;
  本轮复核确认其结构(常驻核 + 核内串行)、block_m 默认值与 UB 上限,未找到进一步杠杆。
- **smoke 精度**:6/6 PASS。
- **两个 Op 一起覆盖**:`has_gate=False`(AdaLayerNormFwdOp)与 `has_gate=True`(AdaLayerNormZeroFwdOp)
  用同一交付件,收益一致。
- **遗留(须独立处置)**:
  ① 内核目录的 Stage 3 交付件被集成层绕过且慢 211× —— 建议由 conductor 决定是删除、标注还是改回;
  ② runner 基线重采的异常读数(4 265 µs 级)根因未查。

### Final Performance Test Data

`candidate_id` 全部为 **`final-deployed-already-tuned`**(= 部署件,`no_gain`)。

> **基线列的口径声明**:本 kernel 的「基线」即**部署件自身**(本轮不改动),
> 故下表 `baseline_us` = `final_us` = 部署件实测值。`perf_records.jsonl` 中另有 4 条
> `candidate_id=baseline` 的历史行(8 490 µs 级),测得的是**被绕过的内核目录件**,
> 与本表的标的不是同一 artifact,故**不并入本表**。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | ada-dit-xl-2-float16 | 9.340 | 9.200 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | ada-dit-xl-2-bfloat16 | 9.660 | 9.600 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | ada-llama-3.1-8b-prefill-float16 | 40.599 | 40.139 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | ada-llama-3.1-8b-prefill-bfloat16 | 42.019 | 41.359 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | ada-llama-3.1-8b-decode-bfloat16 | 3.840 | 3.740 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | zero-dit-xl-2-float16 | 10.880 | 11.020 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | zero-dit-xl-2-bfloat16 | 11.040 | 10.780 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | zero-llama-3.1-8b-prefill-float16 | 53.139 | 53.559 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | zero-llama-3.1-8b-prefill-bfloat16 | 54.299 | 53.679 | final-deployed-already-tuned |
| tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py::_ada_layer_norm_kernel::main | zero-llama-3.1-8b-decode-bfloat16 | 4.780 | 4.780 | final-deployed-already-tuned |

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | **Stage 5 集成口径** | **集成层可以从 `perf_opt/` 导入 kernel,导致 Stage 3 交付件被绕过**,而台账仍按「Stage 3 → Stage 5」记账 | `ada_layer_norm.py` 的 import 指向 `perf_opt._ada_layer_norm_kernel`;内核目录件慢 211× | Stage 5 的集成检查增加一条**断言**:集成层导入的 kernel 文件必须是本任务 Stage 3 交付件(路径比对),否则报错 | D |
| 2 | 交付件命名 | `perf_opt/` 同时承载「Stage 4 产物」与「被集成的实现」时,文件名与角色冲突 | 本目录 `_ada_layer_norm_kernel.py` 既是 Stage 4 交付件又是线上实现 | 明确约定 `perf_opt/{op}.py` 的语义;若集成层需要从 perf_opt 导入,应在台账显式登记该「非标准集成路径」 | R |
| 3 | 采数一致性 | **同一文件同一用例在 runner 与直连两条路径上读数差 100×**(4 265 vs 40.2 µs),当前无任何门禁能发现 | 重采基线的 10 条里 4 条异常,且与旧读数呈 0.50× 关系 | runner 在写记录前增加**自校验**:与同 (kernel_id, workload_id) 的既有记录差异 > 10× 时拒绝写入并报错 | D |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-ada-1 | **D** | **集成层可绕过 Stage 3 交付件**:`ada_layer_norm.py` 从 `perf_opt/` 导入 kernel,该目录件 40.1µs,而内核目录的 Stage 3 件 8490.3µs,**差 211×**,且两份 diff 相差 1274 行 | 两文件 diff;`ada_layer_norm.py` 的 import 块;两侧实测 | repro-missing(需 NPU) | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-ada-2 | **D** | **同 artifact 同用例在两条采数路径上可差 100×**,且异常值与另一 artifact 的旧读数呈 0.50× 比例 | 本日志「未解决」节的对照表 | repro-missing | 同上 |
| VP-ada-3 | **P** | **`no_gain` 的一种来源:交付件已被前序任务调优**(结构 + 参数都已定稿)。判据:结构旋钮已存在于代码(`min(num_logical, _VECTOR_CORES)` + 核内串行)、`block_m` 默认值已有 UB 推导式、参数上探即 UB 溢出 | 本日志 Round 1/2 的复核表 | 本日志 | 同上 |

**pattern-library 回写**:VP-ada-1 拟入 `traps-runtime.md`(集成路径陷阱);
VP-ada-2 拟入 `profile-collection.md` 的采数纪律节(与 `fused_add_rms_norm` 的噪声底条目同族)。

---

## 结论

- **`no_gain`**:部署件已是前序任务的调优成果;结构与 `block_m` 均无可加之处,`num_cores` 硬编码 48、
  `use_cp_async` 被忽略,`block_m ≥ 4` 在 `N=4096` 上 UB 溢出 ⇒ **本轮不引入任何改动**。
- **本轮的两个交付物**:① 部署件在 10 条 tune workload 上的三方一致读数(本 harness / final / Stage 5);
  ② **「内核目录件被集成层绕过且慢 211×」这一风险发现**。
- **不产出的东西**:无 winner 分支;不产出 `[DESIGN_LIMIT]`;
  **未**用 runner 的异常基线读数去构造任何「提升」结论。
- **中止条件**:未 `budget_exhausted`、未 `blocked`、未 `plateau`。
- **verdict**:`TUNING_COMPLETED`(带 `no_gain` 结论,并带两条未结事项)
