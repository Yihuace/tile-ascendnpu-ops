# Stage 4 调优日志 —— `exp_fwd_kernel`(ExpFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 基线(只读):`../exp_fwd_kernel.py`(Stage 3 `[PRECISION_PASS]`)
- 交付件:`exp_fwd_kernel.py`
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper;复用候选模块自带的 `golden_exp` 与其 `TOLERANCES`)
- 批处理 runner:`run_experiments.py`(T-4)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`main`**

> 环境备注:本轮中途会话丢失了 `LD_LIBRARY_PATH`(表现为 `npu-smi` 报 `libc_sec.so` 缺失、
> `torch_npu` 报 `Failed to load the backend extension`)。恢复方式:命令前 `. /usr/local/Ascend/cann-9.0.0/set_env.sh`。
> 该现象与算子无关,但不加恢复会得到「import 阶段失败」的假失败——排查时须先区分。

---

## Performance Test Data

### baseline(round 0,`phase=baseline`,artifact = Stage 3 `../exp_fwd_kernel.py`)

| kernel_id | workload_id | candidate_id | task_duration_us | profile_status |
|---|---|---|---:|---|
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-16m-float16 | baseline | 348.873 | valid |
| 同上 | elementwise-16m-bfloat16 | baseline | 348.573 | valid |
| 同上 | elementwise-16m-float32 | baseline | 348.613 | valid |
| 同上 | elementwise-64m-float16 | baseline | 1390.372 | valid |
| 同上 | elementwise-64m-bfloat16 | baseline | 1390.112 | valid |
| 同上 | elementwise-256m-float16 | baseline-launchable-256m-float16 | 1852.443 | valid(**非交付默认配置**,见下) |
| 同上 | elementwise-256m-bfloat16 | baseline-launchable-256m-bfloat16 | 1852.223 | valid(**非交付默认配置**,见下) |

命令模板:

```bash
msprof op --kernel-name=main \
  --output=profiles/<phase>/round<r>/<candidate>/<workload> \
  --launch-count=15 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python bench.py --impl <candidate.py> --case <workload_id>
```

### 256m 基线的口径声明(与 16m/64m 分开读)

交付默认(`block_size=2048`)在 256m 上**无法启动**(`N/2048 = 131072` 块越栅格上限,
与 abs/ceil/floor 同一陷阱)。本 kernel 的 UB 占用为 12 B/元素(fp16)/16 B/元素(fp32),
故可选档位比取整族宽:

| 用于 256m 的 block_size | 栅格 = N/bs | 结果 |
|---:|---:|---|
| 2048(交付默认) | 131072 | ❌ 越栅格 |
| **6144** | 43691 | ✅ **1852.443 µs ← 表中基线** |

### smoke / skipped 排除

| workload_id | 类别 | 排除原因 | 最终精度 |
|---|---|---|---|
| smoke-1m-float32 | smoke | 案例 ID 含独立 `smoke` 词 | PASS |
| smoke-1m-float16 | smoke | 同上 | PASS |
| smoke-1m-bfloat16 | smoke | 同上 | PASS |

无 `skipped` 项。

---

## ⚠️ 已知精度缺口(fp32,16m)——**先于本轮存在,非本轮引入**

`elementwise-16m-float32` 在**未修改的 Stage 3 文件**上就通不过其自身声明的容差
(`TOLERANCES[float32] = {atol: 1e-5, rtol: 1e-5}`)。A/B 证据(同一 `bench.py --check`,
golden 取候选模块自带的 `golden_exp`):

| 用例 | artifact | block_size | 结果 |
|---|---|---:|---|
| A | **Stage 3 原文件**(`../exp_fwd_kernel.py`) | 2048 | ❌ 2.1% 元素超差;最大绝对差 7.324e-3;最大相对差 **1.494756e-4** |
| B | 本轮候选 | 2048 | ❌ 1.1% 元素超差;最大绝对差 7.767e-3;最大相对差 **1.494754e-4** |
| B' | 本轮候选 | 4096 | ❌ 4.2% 超差;最大相对差 **1.494754e-4** |
| B'' | 本轮候选 | 8192 | ❌ 4.2% 超差;最大相对差 **1.494752e-4** |
| C | 本轮候选 | 16384(fp16) | ✅ PASS |

**判读**:① 最大相对差在 A/B/B'/B'' 上**是同一个数值**(1.4947e-4),与 `block_size` 无关
⇒ 这是 `vexp` 的**系统性逼近误差**,不是配置相关或结构相关的 bug;
② A 与 B 都失败 ⇒ **Stage 3 交付件本身**在该 workload 上超出其自声明容差;
③ 超差率在 A(2.1%)与 B(1.1%/4.2%)之间的差异源于每次随机输入的抽样不同,不是行为差异。

**处置**:**不**把它当作 Stage 4 的精度回归(结构对该差异的贡献为零,有 A/B 直接对照),
但**如实上报为 Stage 3 / DESIGN 层的精度缺口**,需独立处置(候选路径:放宽 fp32 容差并给出
理由、或改用更精确的 exp 实现)。本轮**未**修改 `DESIGN.md` / Stage 3 交付件(只读基线),
Stage 4 的 fp32 性能测量走 `phase=final`(不经 `--check`),故该缺口不阻塞本阶段交付。

fp16 / bf16 全部通过,本轮的调优结论以它们为准。

---

## Iteration Log

### Round 1 —— 结构改写:grid-stride `T.serial`

**现象**:`Block Dim = 8192`(16m)/`32768`(64m)/`131072`(256m);16m(348.87µs)与 64m
(1390.37µs)每元素耗时相同 ⇒ 瓶颈随规模线性;fp32 搬 2 倍字节仅多 −0.07% 时延 ⇒ 限制在「每元素」。

**候选优化点(单一)**:grid 收成 `num_cores` + 核内 grid-stride 串行;widen → vexp → narrow 链一字不动。

**合并非回退检查(全部 tune workload)**

| branch | workload_id | L0 | Task Duration(us) | vs current best | status |
|---|---|---:|---:|---:|---|
| r1_serial48 | elementwise-16m-float16 | pass | 64.979 | +81.4% | ok |
| r1_serial48 | elementwise-16m-bfloat16 | pass | 64.699 | +81.4% | ok |
| r1_serial48 | elementwise-16m-float32 | pass | 74.959 | +78.5% | ok |
| r1_serial48 | elementwise-64m-float16 | pass | 264.535 | +81.0% | ok |
| r1_serial48 | elementwise-64m-bfloat16 | pass | 261.075 | +81.2% | ok |
| r1_serial48 | elementwise-256m-float16 | pass | 1172.697 | 基线为可启动配置(1852.443) | ok |
| r1_serial48 | elementwise-256m-bfloat16 | pass | 1167.337 | 基线为可启动配置(1852.223) | ok |

**winner**:`r1_serial48`(7/7 无回退,并解锁 256m)。结构单独贡献 **−81.4%**,与 abs 的 −81.9%
基本一致 —— 两者同属「UB 缓冲少、每 tile 向量指令少」的一类(exp 只有 4 个缓冲、3 条指令)。

### Round 2 —— 结构后粗搜:`block_size × num_cores`

**搜索空间**:`block_size ∈ {2048, 4096, 8192}` × `num_cores ∈ {24, 48, 96, 192}`(12 组全测),
代表 shape = `elementwise-16m-float16`。参数分支用 `bench.py --check` 作配置级等价门禁。

**候选 vs current best 对比表**(current best = `r1_serial48`,16m fp16 = 64.979µs)

| block_size | num_cores | Task Duration(us) | vs r1_serial48 |
|---:|---:|---:|---:|
| 2048 | 24 | 122.958 | −89.2%(劣化) |
| 2048 | 48 | 65.159 | −0.3% |
| 2048 | 96 | 65.099 | −0.2% |
| 2048 | 192 | 66.379 | −2.2% |
| 4096 | 24 | 70.619 | −8.7%(劣化) |
| 4096 | 48 | 39.279 | **+39.6%** |
| 4096 | 96 | 39.919 | +38.6% |
| 4096 | 192 | 40.739 | +37.3% |
| 8192 | 24 | 45.559 | +29.9% |
| 8192 | 48 | 29.399 | **+54.8%** |
| 8192 | 96 | 29.079 | +55.1% |
| 8192 | 192 | **28.839** | **+55.6%** |

**读表结论**:同族同型——`num_cores=24` 全面劣化,≥48 进平区(≤2%);`block_size` 是主杠杆
(2048→4096→8192 单调变好,与 abs 同形)。**winner**:`bs=8192 / nc=192`(28.839µs)。

### Round 3 —— UB 上限探测 + 跨 dtype/shape 确认

**现象**:round 2 的 winner 只在 16m fp16 上测得;`bs=16384` 的 UB 余量须实测(fpx16 需 12 B/元素)。

| branch | workload | Task Duration(us) | 结论 |
|---|---|---:|---|
| bs=16384 / nc=48 | elementwise-16m-float16 | **26.559** | ✅ fp16 合法且更快 |
| bs=8192 / nc=48 | elementwise-16m-float32 | — | ❌ **精度失败**(见上节 A/B:系 Stage 3 已存在的 fp32 缺口,非本轮引入) |
| bs=8192 / nc=48 | elementwise-16m-bfloat16 | 28.099 | 合法 |
| bs=8192 / nc=48 | elementwise-64m-float16 | 132.637 | 合法 |
| bs=8192 / nc=48 | elementwise-256m-float16 | 800.044 | 合法 |

**为什么最终取 `bs=8192` 而非更快的 `bs=16384`**:本 kernel 在 fp32 下每元素占 16 B
(x/f/e/y 各 4 B),`16384 × 16 = 256 KB > 192 KB` UB ⇒ `bs=16384` **对 fp32 不合法**;
`bs=8192`(fp32 131 KB)对三种 dtype 均合法,故取「一档通吃」的 8192(代价仅 ~8%:
28.84 vs 26.56µs)。**不**按 dtype 在 factory 里分派 —— 避免把分派塞进 Python 工厂层。

**winner**:`bs=8192 / nc=192`。**中止条件**:结构 + 参数空间收束。

---

## Autotune Log

未使用 `tilelang` autotune API;round 2 用等价手动粗搜覆盖同一空间(12 组全测完)。

| 项 | 内容 |
|---|---|
| search space | `block_size ∈ {2048, 4096, 8192}` × `num_cores ∈ {24, 48, 96, 192}`(12 组全测) |
| best config | `block_size=8192`, `num_cores=192` → 28.839 µs |
| top-k | `8192/96` 29.079 · `8192/48` 29.399 · `4096/48` 39.279 |
| 未选组合 | `block_size=16384`:fp16/bf16 更快(26.559µs)但 **fp32 不合法**,故不取;`num_cores=24` 已证劣化 |
| 编译/精度失败数 | 编译失败 0;精度失败 1(**fp32 16m,Stage 3 既有缺口**,已单列) |
| winner 独立复测 | `perf_opt/exp_fwd_kernel.py`(默认 = 该组合)final 复测 16m fp16 = **28.479 µs** |
| Block Dim | 每个候选 = `num_cores` |

**平区裁决说明**:`nc≥48` 三档差异 ≤2%;定档 192 而非 48 的依据是**实测最低值**
(29.399 → 28.839,1.9%),且在 final 复测中方向一致(28.479µs)。
`num_cores=24` 的劣化是量级差异(2×),不属平区。

---

## Final Summary

**逐 workload 提升**(逐项对账见下节 `Final Performance Test Data`)

| workload_id | baseline_us | final_us | 提升 | 备注 |
|---|---:|---:|---:|---|
| elementwise-16m-float16 | 348.873 | 28.479 | **+91.8%(12.25×)** | |
| elementwise-16m-bfloat16 | 348.573 | 28.879 | +91.7%(12.07×) | |
| elementwise-16m-float32 | 348.613 | 48.539 | +86.1%(7.18×) | ⚠️ 该 workload 存在 Stage 3 既有 fp32 精度缺口(见前节) |
| elementwise-64m-float16 | 1390.372 | 132.737 | +90.5%(10.47×) | |
| elementwise-64m-bfloat16 | 1390.112 | 130.797 | +90.6%(10.63×) | |
| elementwise-256m-float16 | 1852.443 | 779.124 | +57.9% | 基线为可启动配置(bs=6144) |
| elementwise-256m-bfloat16 | 1852.223 | 775.784 | +58.1% | 同上 |

**带宽占峰比**(分母 = 1.8 TB/s;口径约束见 `constants.md` CONST-bw-occupancy-l2-resident)

| workload_id | baseline | final |
|---|---:|---:|
| elementwise-16m-float16 | 10.7% | **130.9%** |
| elementwise-16m-bfloat16 | 10.7% | **129.1%** |
| elementwise-16m-float32 | 21.4% | **153.6%** |
| elementwise-64m-float16 | 10.7% | **112.4%** |
| elementwise-64m-bfloat16 | 10.7% | **114.0%** |
| elementwise-256m-float16 | — | **38.3%** |
| elementwise-256m-bfloat16 | — | **38.4%** |

> 16m/64m 超过 100%(L2 驻留),256m 越过 L2 后为 38.3% —— 与 abs 同型。

### 设计估算 vs 实测偏差行(D-2 回填,DESIGN.md §1.6.0)

| workload | 设计估算下界(2·N·e ÷ 1.8 TB/s) | baseline 实测 | 偏差 | final 实测 | 相对估算下界 |
|---|---:|---:|---:|---:|---|
| elementwise-16m-float16 | 37.3 µs | 348.873 | **9.4×** | 28.479 | **优于下界 1.31×** |
| elementwise-16m-float32 | 74.6 µs | 348.613 | 4.7× | 48.539 | **优于下界 1.54×** |
| elementwise-64m-float16 | 149.1 µs | 1390.372 | 9.3× | 132.737 | **优于下界 1.12×** |

**失准项归属**:baseline 的 9.4× 归因于逐块派发开销(与 abs/ceil/floor 同因);final **优于**
设计估算下界 ⇒ 与 abs 同型,说明 1.8 TB/s 作分母偏保守(L2 驻留红利),而**不是**设计低估了本算子。

**smoke 最终精度**:3/3 PASS。**skipped**:无。**`merge_blocked`**:无。

### Final Performance Test Data

`candidate_id` 全部为 **`final-bs8192-nc192`**,时延来自 `perf_records.jsonl` 的 `phase=final` 记录。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-16m-float16 | 348.873 | 28.479 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-16m-bfloat16 | 348.573 | 28.879 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-16m-float32 | 348.613 | 48.539 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-64m-float16 | 1390.372 | 132.737 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-64m-bfloat16 | 1390.112 | 130.797 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-256m-float16 | 1852.443 | 779.124 | final-bs8192-nc192 |
| tileops/kernels/elementwise/exp/exp_kernel/exp_fwd_kernel.py::exp_fwd_kernel::main | elementwise-256m-bfloat16 | 1852.223 | 775.784 | final-bs8192-nc192 |

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | 精度门禁 vs 既有缺口 | **Stage 4 的 `--check` 会把 Stage 3 既有的精度缺口报成「本轮失败」**,若无 A/B 对照极可能被误判为优化引入的回归 | 首次在 `fp32/bs=8192` 报 `Tensor-likes are not close`;A/B 对照(Stage 3 原文件在 bs=2048 同样失败、最大相对差同一数值)才定性为既有缺口 | `profile-collection.md` / optimize SKILL 增加规则:**候选的精度失败必须先与 Stage 3 基线在相同配置下对照**,不得直接记为「本轮引入」 | R |
| 2 | 环境 | **会话可能丢失 `LD_LIBRARY_PATH`**,表现为 `npu-smi`/`torch_npu` 加载失败(报错与算子无关但会被当成算子错误) | 中断恢复后 `torch_npu` 报 `Failed to load the backend extension`;`. /usr/local/Ascend/cann-9.0.0/set_env.sh` 即恢复 | 环境自检前置:`bench.py` 在 import 前校验 CANN 库路径,缺失时给出明确的修复指引而非原始 traceback | R |
| 3 | 同族配方复用 | 复用收益仍未记录 | exp 的 round 1 结构贡献 −81.4%,与 abs 的 −81.9% 同为「少缓冲少数指令」一类;而 ceil/floor(10 缓冲)只有 −69% | 同 floor 提案:`reused_from` 字段 + 「每 tile 工作量 → 收益上限」的经验关系本轮回写 PL-1.21 | R |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-exp-1 | **P** | **PL-1.21 第四证**:exp 16m fp16 348.87→64.98µs(−81.4%),叠 `bs=8192/nc=192` 到 28.48µs(累计 −91.8%,12.25×)。**边界条件再确认**:收益与「每 tile 的 UB 缓冲数 / 向量指令数」负相关——exp 4 缓冲/3 指令 → −81%;ceil/floor 10 缓冲/12+ 指令 → −69% | baseline 7 + r1 7 + final 7(`perf_records.jsonl`) | `pattern-library/repro/PL-1.21-gridstride-serial.py` | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-exp-2 | **D** | **`vexp` 的 fp32 系统性逼近误差 ≈1.4947e-4 相对**,超过本仓 elementwise 族自声明的 `rtol=1e-5`;在 16m 规模上使 1.1–4.2% 元素超差,且**Stage 3 交付件同样失败**(与 block_size 无关,同一数值) | A/B 四组对照(见「已知精度缺口」节);原始断言文本 `Greatest relative difference: 0.00014947539602871984` | repro-missing(需 NPU;数值与判读已自包含) | 同上 |
| VP-exp-3 | **D** | **UB 预算决定「一档通吃」的边界**:exp 的 fp32 每元素 16 B ⇒ `bs=16384` 需 256 KB > 192 KB,故虽然 fp16 在 16384 上快 8%(26.56 vs 28.84µs),统一默认仍须取 8192 | round 3 的 bs=16384(fp16 过、fp32 不过)与 round 2 的 8192 档 | repro-missing | 同上 |

**pattern-library 回写**:VP-exp-1 作为 PL-1.21 第四证追加;VP-exp-2 拟入 `traps-runtime.md`
(VEXP 类)。**未**回写 VP-exp-3(属个案参数选择,已自包含于本日志)。

---

## 结论

- **中止条件**:结构 + 参数空间收束(3 轮 / 18 个候选分支;预算 10/30 未耗尽);未 `blocked`、未 `plateau`。
- **性能目标**:未指定逐 workload 数值目标 ⇒ 以「通过核内合并与全量非回退检查的实测最优结果」为结论。
- **`[DESIGN_LIMIT]`**:不产出。final 在 16m/64m 上**优于**设计估算下界(1.12–1.54×),
  参数/结构域内未见天花板;fp32 的精度缺口属**正确性**问题(已按既有缺口上报),不属性能天花板。
- **遗留(须独立处置)**:`elementwise-16m-float32` 的 Stage 3 既有精度缺口 —— 本轮以 A/B 证据定性并
  如实上报,**未掩盖、也未假装已修复**。
- **verdict**:`TUNING_COMPLETED`

---

# 追加(2026-10-28):Round 4 —— block_size 按 dtype 分化,把已被测到却丢掉的 −7.9% 捡回来

## 1. 问题:一个实测收益被"一个值通吃"的默认值吞掉了

Round 2-3 的扫描记录里本来就写着(注释原文):

```
bs=8192  nc=192  28.84us  <- chosen
bs=16384 nc=192  26.56us   (fp16/bf16 only)
```

即 **fp16/bf16 下 `bs=16384` 快 7.9%**,但因为 fp32 在 16384 时
`16384 × 16 B = 256 KB > 192 KB` 装不下(`x/f/e/y` 四个缓冲各 4 B),
交付件取了统一默认值 **8192** —— 于是 tune workload 里除 16m fp32 外全是
fp16/bf16,却都按 fp32 的上限跑。

## 2. 改动

```python
BLOCK_SIZE = 8192
BLOCK_SIZE_F16 = 16384          # fp16/bf16 专用(12 B/元素 ⇒ 恰好 192 KB)

def exp_fwd_kernel(N, dtype, output_dtype=None, block_size=None, num_cores=NUM_CORES):
    if block_size is None:
        _out = output_dtype or dtype
        block_size = BLOCK_SIZE_F16 if (dtype != "float32" and _out != "float32") else BLOCK_SIZE
```

`block_size=None` 时按 dtype 解析;**显式传参 `--block-size` 仍然优先**,所以参数扫描路径不受影响。

UB 账:

| dtype | 每元素 | bs=16384 需 | 判定 |
|---|---:|---:|---|
| fp32 | 16 B(x/f/e/y 各 4) | 256 KB | ✗ 超限 → 留 8192 |
| fp16/bf16 | 12 B(2+4+4+2) | **192 KB** | ✓ 恰好合规 |

## 3. 实测(msprof Task Duration)

| workload | A(bs=8192) | **B(dtype 分化)** | 变化 |
|---|---:|---:|---:|
| elementwise-16m-float16 | 29.161 µs | **26.541 µs** | **−8.98%** |
| elementwise-16m-bfloat16 | 29.341 µs | **27.281 µs** | **−7.02%** |
| elementwise-16m-float32 | 48.562 µs | 49.942 µs | +2.84%(噪声) |

**fp32 的 +2.84% 是噪声**:该路径解析出的 `block_size` 与改动前**完全相同**(仍 8192),
tracing 出的内核逐行一致,不存在因改动而变化的可能;在 5% 门槛内。

## 4. 采纳与验证

已采纳为交付件,备份 `/tmp/exp_backup.py`。

## 5. 验证

- 集成测试 **8 passed**
- 内核级 L0 **3 PASS / 0 FAIL**

> 注:L0 的 `max_diff` 每次运行不同(基线 fp16 1.600e+01 / fp32 3.125e-02,
> 改动后 fp16 8.000e+00 / fp32 5.000e-01)—— 因为 L0 用 `run_random` 随机数据,
> **跨运行不可比**,两次都 PASS。块尺寸对逐元素算子不改变数值结果。

---

# 追加(2026-10-30):Round 5 —— 块尺寸再加"规模"判据,修掉 1m 场景的真实回退

## 1. 问题:Round 4 的 dtype 分化在 1m 上是负收益(已实测)

Round 4 让 fp16/bf16 用 `bs=16384`。但在 **1m smoke 用例**上直接 A/B(同一份内核文件,只改 `--block-size`):

| workload | bs=8192 | bs=16384 | 变化 |
|---|---:|---:|---:|
| `smoke-1m-float16` | 9.92 µs | 10.76 µs | **+8.47%(更慢)** |
| `smoke-1m-bfloat16` | 9.92 µs | 10.56 µs | **+6.45%(更慢)** |

**根因**:1m / 16384 = **64 个 block**,而只有 48 个核 ⇒ 最后一轮只有 16/48 核有活,
尾部掩盖严重不足。这与 `floor` 上记过的那次抖动是同一个机制(注释里当时判为"抖动",
现在有直接 A/B 证据,应改成"小 shape 的真实代价")。

## 2. 改法:块尺寸按「dtype **且** 规模」选择

```python
BLOCK_SIZE     = 8192
BLOCK_SIZE_F16 = 16384
_BIG_MIN_N = 4 * 1024 * 1024      # 4m

if block_size is None:
    _out = output_dtype or dtype
    _big_ok = (dtype != "float32" and _out != "float32" and N >= _BIG_MIN_N)
    block_size = BLOCK_SIZE_F16 if _big_ok else BLOCK_SIZE
```

**Op 层同步**(`tileops/kernels/elementwise/exp/exp.py`)—— 否则 Op 层的
`_DEFAULT_BLOCK_SIZE` 会显式传值把内核逻辑覆盖掉(wrapper 入口陷阱):

```python
from .exp_kernel.perf_opt.exp_fwd_kernel import exp_fwd_kernel, _BIG_MIN_N

@property
def default_config(self) -> dict:
    _big_ok = (self.dtype_str != "float32" and self.N_total >= _BIG_MIN_N)
    return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str] if _big_ok else 8192}
```

**规则解析自检**(纯 Python):

| N | fp16 | bf16 | fp32 |
|---|---:|---:|---:|
| 1m | 8192 | 8192 | 8192 |
| 2m | 8192 | 8192 | 8192 |
| 4m | **16384** | **16384** | 8192 |
| 16m | **16384** | **16384** | 8192 |
| 256m | **16384** | **16384** | 8192 |

## 3. 复测(交付件默认配置,不传 `--block-size`)

| workload | 旧(全 8192) | **新(N 感知)** | 变化 |
|---|---:|---:|---:|
| `smoke-1m-float16` | 9.92 µs | **9.84 µs** | **−0.81%**(原 +8.47% 的回退已消除) |
| `elementwise-16m-float16` | 28.44 µs | **27.119 µs** | **−4.64%** |
| `elementwise-64m-float16` | 134.59 µs | **124.538 µs** | **−7.47%** |
| `elementwise-256m-float16` | 779.11 µs | **750.245 µs** | **−3.70%** |

**小 shape 不再回退、大 shape 完整保留收益。**

## 4. 验证

内核级 L0 **3 PASS / 0 FAIL**(fp16 `8.000e+00`、fp32 `1.562e-02`、bf16 `0.000e+00`)。

## 5. 教训

Round 4 里只按 dtype 分化是不够的——**块尺寸的最优值同时取决于 dtype 与规模**:
dtype 决定"每元素占多少 UB",规模决定"最后一个栅格轮次能否填满"。
两个条件任一不满足,放大块尺寸就是负收益。
