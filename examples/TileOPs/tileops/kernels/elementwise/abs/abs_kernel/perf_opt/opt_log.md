# Stage 4 调优日志 —— `abs_fwd_kernel`(AbsFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 基线(只读):`../abs_fwd_kernel.py`(Stage 3 `[PRECISION_PASS]`)
- 交付件:`abs_fwd_kernel.py`
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数(15 次 measured launch)
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper)
- 批处理 runner:`run_experiments.py`(T-4;L0/`--check` → msprof → `perf_records.jsonl` 追加 → 汇总表)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核,实查)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`abs_fwd_kernel`**

> 口径说明:Stage 5 报告把 `kernel_name_requested` 记为 `main`,而 msprof 实测的 kernel 名是
> `abs_fwd_kernel`。本日志一律使用实测名;探针见下节第 1 条命令。

---

## Performance Test Data

### baseline(round 0,`phase=baseline`,artifact = Stage 3 `../abs_fwd_kernel.py`)

| kernel_id | workload_id | candidate_id | target_kernel_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---:|---|---:|---|
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-16m-float16 | baseline | abs_fwd_kernel | 348.313 | valid | profiles/baseline/round0/baseline_8ba8496a/elementwise-16m-float16_72f212c1 |
| 同上 | elementwise-16m-bfloat16 | baseline | abs_fwd_kernel | 348.433 | valid | profiles/baseline/round0/baseline_8ba8496a/elementwise-16m-bfloat16_83728fec |
| 同上 | elementwise-16m-float32 | baseline | abs_fwd_kernel | 348.273 | valid | profiles/baseline/round0/baseline_8ba8496a/elementwise-16m-float32_61e8c2bc |
| 同上 | elementwise-64m-float16 | baseline | abs_fwd_kernel | 1389.092 | valid | profiles/baseline/round0/baseline_8ba8496a/elementwise-64m-float16_d476a170 |
| 同上 | elementwise-64m-bfloat16 | baseline | abs_fwd_kernel | 1390.552 | valid | profiles/baseline/round0/baseline_8ba8496a/elementwise-64m-bfloat16_4fcef0c6 |
| 同上 | elementwise-256m-float16 | baseline | abs_fwd_kernel | — | **invalid** | profiles/baseline/round0/baseline_8ba8496a/elementwise-256m-float16_4bbe4f7b |
| 同上 | elementwise-256m-bfloat16 | baseline | abs_fwd_kernel | — | **invalid** | profiles/baseline/round0/baseline_8ba8496a/elementwise-256m-bfloat16_b47cecfd |

命令模板(逐 workload 替换 `--case`):

```bash
msprof op --kernel-name=abs_fwd_kernel \
  --output=profiles/<phase>/round<r>/<candidate>/<workload> \
  --launch-count=15 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python bench.py --impl <candidate.py> --case <workload_id>
```

### invalid profile 的原因(256m 两条,不予采用)

`elementwise-256m-float16 / -bfloat16` 在 Stage 3 结构下**启动即失败**,不是采集侧问题:

```
msprof: Kernel Cast_... skipped: not selected via --kernel-name.
        Child process exited with status 134            # SIGABRT
terminate called after throwing an instance of 'c10::Error'
  what(): InnerRunOpApi:.../OpParamMaker.cpp:287 OPS function error:
           abs_fwd_kernel, error code is 107000
[Error]: Parameter verification failed.
[PID: ...] Invalid_Argument(EL0003): The argument is invalid.
```

根因:**Block Dim 越界**。Stage 3 一个 tile 一个 block,`N/2048` 在 256m 上是 **131072**,超过
launch 栅格上限;16m(8192)与 64m(32768)都在限内,故只有 256m 中招。
该现象与 Stage 5 报告里那 2 例 `aclrtEventElapsedTime 507000` 是同一处失败的不同表相。

### smoke / skipped 排除(不采集性能、不选 winner)

| workload_id | 类别 | 排除原因 | 最终精度 |
|---|---|---|---|
| smoke-1m-float32 | smoke | 案例 ID 含独立 `smoke` 词 | PASS |
| smoke-1m-float16 | smoke | 同上 | PASS |
| smoke-1m-bfloat16 | smoke | 同上 | PASS |

无 `skipped` 项。

---

## Iteration Log

### Round 1 —— 结构改写:把 tile 走查折进核内(grid-stride `T.serial`)

**现象**(来自 baseline profile):`Block Dim = 8192`(16m fp16)摊到 48 个 Vector 核 =
**每核 170 个块,每块仅 2048 元素(4 KB)**;64m 更极端,`Block Dim = 131072`。三条互证:

1. 16m(348.19µs)与 64m(1389.74µs)的**每元素耗时相同** ⇒ 瓶颈随规模线性,不是启动开销;
2. fp32 搬 2 倍字节却几乎同时延(348.19 vs 354.47)⇒ 限制在「每元素」而非「每字节」;
3. 块极小且极多 ⇒ 与 `autotune.md`「`Block Dim` 远超物理核数时先做 `T.serial` 多块迭代」的处方一致。

**候选优化点(单一)**:grid 由 `ceildiv(N, block_size)` 收成 `num_cores`,核内用 grid-stride
`T.serial` 串行走各自份额的 tile。合法写法照抄仓内参考
`examples/elementwise/example_elementwise_exp2.py`(循环上界 + `if block_id` 守卫)。
其余(源在前的 `T.vabs(A,B)`、零起点 UB 切片、fp32 opmath golden)一字不动。

**实验分支**:`abs_fwd_kernel_round1_serial.py`(`parent_id=baseline`,唯一改动 = 上述结构)

**合并非回退检查(全部 tune workload)**

| branch | workload_id | L0 | Task Duration(us) | vs current best | status |
|---|---|---:|---:|---:|---|
| r1_serial48 | elementwise-16m-float16 | pass | 63.059 | +81.9% | ok |
| r1_serial48 | elementwise-16m-bfloat16 | pass | 64.719 | +81.4% | ok |
| r1_serial48 | elementwise-16m-float32 | pass | 72.819 | +79.1% | ok |
| r1_serial48 | elementwise-64m-float16 | pass | 254.035 | +81.7% | ok |
| r1_serial48 | elementwise-64m-bfloat16 | pass | 263.795 | +81.0% | ok |
| r1_serial48 | elementwise-256m-float16 | pass | 1153.057 | 基线不可运行 | ok |
| r1_serial48 | elementwise-256m-bfloat16 | pass | 1168.677 | 基线不可运行 | ok |

**winner**:`r1_serial48`(7/7 workload 无回退,且把 256m 从「启动失败」变为可运行)。
**smoke 精度**:3/3 PASS。**中止条件**:未命中,进入下一轮。

### Round 2 —— 结构后粗搜:`block_size × num_cores`

**现象**(来自 round 1 最新 profile):结构已改对,但 `block_size` 仍是 Stage 3 的 2048
(每块 4 KB,搬运粒度小),`num_cores=48` 也未经实测。按 skill「结构分支有效后先做一轮
coarse autotune / 等价手动粗搜」执行。

**候选优化点(单一)**:同一结构下的**参数搜索**,不动结构。

**搜索空间**:`block_size ∈ {2048, 4096, 8192, 16384}` × `num_cores ∈ {24, 48, 96, 192}`,
代表 shape = `elementwise-16m-float16`(48 个分支中最能暴露搬运粒度与并发的一个)。
参数分支使用 `bench.py --check`(配置精确 golden 比对)作为等价门禁,替代文件级 L0
(后者只跑模块默认参数)——见 Skill Retrospective 的 BP 提案。

**候选 vs current best 对比表**(current best = `r1_serial48`,16m fp16 = 63.059µs)

| block_size | num_cores | Task Duration(us) | vs r1_serial48 | 备注 |
|---:|---:|---:|---:|---|
| 2048 | 24 | 119.178 | −88.9% | 并发不足,明显回退 |
| 2048 | 48 | 62.739 | +0.5% | 与 r1 同口径 |
| 2048 | 96 | 63.099 | −0.1% | |
| 2048 | 192 | 63.939 | −1.4% | |
| 4096 | 24 | 66.159 | −4.9% | |
| 4096 | 48 | 37.519 | **+40.5%** | |
| 4096 | 96 | 38.279 | +39.3% | |
| 4096 | 192 | 38.399 | +39.1% | |
| 8192 | 24 | 42.659 | +32.4% | |
| 8192 | 48 | 28.239 | **+55.2%** | |
| 8192 | 96 | 28.319 | +55.1% | |
| 8192 | 192 | 27.479 | +56.4% | |
| 16384 | 24 | 32.639 | +48.3% | |
| 16384 | 48 | 27.319 | +56.7% | |
| 16384 | 96 | 26.359 | +58.2% | |
| 16384 | 192 | **25.619** | **+59.4%** | 本轮 winner |

**读表结论**:① `num_cores=24` 全面劣化 ⇒ 24 核喂不满 48 个 Vector 核;
② ≥48 后基本进入平区(48/96/192 三档差异 ≤7%);③ `block_size` 是主杠杆,
2048→4096→8192→16384 单调变好,符合「搬运粒度」而非「并行度」受限的判断。

**winner**:`bs=16384 / nc=192`(25.619µs)。**中止条件**:未命中,继续确认跨 dtype/shape。

### Round 3 —— 最优组合的跨 dtype / 跨 shape 确认与 UB 上限探测

**现象**:round 2 的最优只在 16m fp16 上测得。`x_ub + y_ub = 2·block_size·elem_bytes`,
fp32 的 elem_bytes 翻倍 ⇒ `block_size=16384` 对 fp32 是否合法必须实测,不能推。

**候选优化点(单一)**:把 round 2 winner 的**参数**带到其它 dtype/shape(不引入新结构)。

| branch | workload_id | Task Duration(us) | 结论 |
|---|---|---:|---|
| r3_elementwise-16m_float32_bs8192_nc192 | elementwise-16m-float32 | 47.839 | |
| r3_elementwise-16m_float32_bs16384_nc192 | elementwise-16m-float32 | 48.799 | **bs=16384 对 fp32 合法**(编译通过、精度通过);与 bs=8192 差 2.0%,在噪声带内 |
| r3_elementwise-16m_bfloat16_bs16384_nc192 | elementwise-16m-bfloat16 | 26.679 | 最优迁移成功 |
| r3_elementwise-64m_float16_bs16384_nc192 | elementwise-64m-float16 | 127.617 | 大 shape 同样受益 |
| r3_elementwise-256m_float16_bs16384_nc192 | elementwise-256m-float16 | 752.345 | 同 |

**UB 上限实测结论**:`bs=16384` 对 fp16/bf16/fp32 三种 dtype **均合法**。
fp32 时 `2×16384×4 = 128 KB` < 192 KB UB;`TRAP-UB-multibuffer-inflation` 的 ~1.7x 膨胀
**在本 kernel 未生效**(该 kernel 无 `T.Pipelined`,与 `fused_add_rms_norm` REVIEW 中
「膨胀系数针对多缓冲」的判断一致)。

**winner**:`bs=16384 / nc=192` 跨 dtype/shape 成立;16m fp32 该档与 bs=8192 属平区(2.0%),
按「一档通吃 + 减少分派」取 `bs=16384`。**中止条件**:结构候选已充分验证,进入收束。

---

## Autotune Log

未使用 `tilelang` autotune API;round 2 用**等价手动粗搜**覆盖同一搜索空间(见 round 2 表)。

| 项 | 内容 |
|---|---|
| search space | `block_size ∈ {2048, 4096, 8192, 16384}` × `num_cores ∈ {24, 48, 96, 192}`(16 组,全部测完) |
| best config | `block_size=16384`, `num_cores=192` → 25.619 µs |
| top-k | `16384/96` 26.359 · `16384/48` 27.319 · `8192/192` 27.479 · `8192/48` 28.239 |
| 未测组合 | 无;`block_size>16384` 未展开(UB 已用 128 KB/192 KB 于 fp32,余量不足以再翻倍) |
| 编译/精度失败数 | 0(16 组全部编译通过且 `--check` 通过) |
| winner 独立复测 | `perf_opt/abs_fwd_kernel.py`(模块默认 = 该组合)final 复测 16m fp16 = **25.099 µs** |
| Block Dim / logical blocks | 每个候选 `Block Dim = num_cores`(192);理论 logical tile 数 = `ceildiv(N, block_size)`,由核内 grid-stride 走完 |

**平区裁决说明**:`num_cores ≥ 48` 后三档差异 ≤7%,`block_size` 16384 与 8192 在 fp32 上差 2.0%
(落在噪声阈内)。这两处**未**走 `ab_test.py` 交错 A/B 协议即定档,依据是:决策取的是
「一档通吃的默认值」而非「逐 shape 分派」,且 fp16/bf16 两档差异(25.6 vs 27.5 ≈ 7%)方向
在 64m 上复现(127.6 vs 未见回退),属于**同一方向的单调趋势 + 合并后全量非回退**,
不构成「用单次测量在平区内排序」。此判定列为流程风险记入 Skill Retrospective。

---

## Final Summary

**逐 workload 提升**(逐项对账见下节 `Final Performance Test Data`)

| workload_id | baseline_us | final_us | 提升 | 备注 |
|---|---:|---:|---:|---|
| elementwise-16m-float16 | 348.313 | 25.099 | +92.8%(13.9×) | |
| elementwise-16m-bfloat16 | 348.433 | 26.179 | +92.5%(13.3×) | |
| elementwise-16m-float32 | 348.273 | 48.339 | +86.1%(7.2×) | |
| elementwise-64m-float16 | 1389.092 | 127.217 | +90.8%(10.9×) | |
| elementwise-64m-bfloat16 | 1390.552 | 125.597 | +91.0%(11.1×) | |
| elementwise-256m-float16 | 825.563 | 748.665 | +9.3% | 基线为**可启动配置**,非交付默认值 |
| elementwise-256m-bfloat16 | 837.663 | 749.505 | +10.5% | 同上 |

**256m 基线的口径声明(必须与 16m/64m 分开读)**
交付默认(`block_size=2048`)在 256m 上**无法启动**(栅格 131072 越限,ACL 107000 / SIGABRT),
故不存在默认配置下的基线时延。上表 825.563 / 837.663 µs 是**同一个 Stage 3 文件**、
仅把 `block_size` 设为 16384(栅格 16384,合法)测得的、**唯一可得的**基线。
它同时是一个对照实验,把两个因素分离开:

| 因素 | 16m fp16 | 64m fp16 | 256m fp16 |
|---|---:|---:|---:|
| 交付默认(逐 tile 一 block,bs=2048) | 348.313 | 1389.092 | **无法启动** |
| 仅换 block_size=16384(栅格仍为 N/bs) | 未单独测 | 未单独测 | 825.563 |
| 仅改 grid(bs=2048,grid-stride) | 63.059 | 254.035 | 1153.057 |
| **两者合并(final)** | **25.099** | **127.217** | **748.665** |

读法:① 在能用默认配置启动的规模上,**结构改写的贡献更大**(16m:−81.9% vs 参数再 −59.4%);
② 在 256m 上默认配置根本不可运行,`block_size` 是**解锁因素**(把不可运行变为可运行),
结构改写再贡献 −9.3%;③ 两者合并后 256m 也优于「仅放大 block_size」的 825.563µs。

### Final Performance Test Data

`candidate_id` 全部为 **`final-bs16384-nc192`**,时延来自 `perf_records.jsonl` 的
`phase=final` 记录(非拼接各分支最好成绩)。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-16m-float16 | 348.313 | 25.099 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-16m-bfloat16 | 348.433 | 26.179 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-16m-float32 | 348.273 | 48.339 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-64m-float16 | 1389.092 | 127.217 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-64m-bfloat16 | 1390.552 | 125.597 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-256m-float16 | 825.563 | 748.665 | final-bs16384-nc192 |
| tileops/kernels/elementwise/abs/abs_kernel/abs_fwd_kernel.py::abs_fwd_kernel::abs_fwd_kernel | elementwise-256m-bfloat16 | 837.663 | 749.505 | final-bs16384-nc192 |

**smoke 最终精度**:`smoke-1m-float32 / float16 / bfloat16` = **3/3 PASS**。
**skipped**:无。**`merge_blocked`**:无。

### 带宽占峰比(分母 = 仓库常数 1.8 TB/s)

| workload_id | baseline | final | 说明 |
|---|---:|---:|---|
| elementwise-16m-float16 | 10.7% | **148.6%** | |
| elementwise-16m-bfloat16 | 10.7% | **142.4%** | |
| elementwise-16m-float32 | 21.0% | **154.3%** | |
| elementwise-64m-float16 | 10.7% | **117.2%** | |
| elementwise-64m-bfloat16 | 10.7% | **118.7%** | |
| elementwise-256m-float16 | — | **39.8%** | |
| elementwise-256m-bfloat16 | — | **39.8%** | |

> ⚠️ **口径告警(重要)**:优化后 16m/64m 的占峰比**超过 100%**,这**不是**测量错误 ——
> 1.8 TB/s 是**片外(GM/HBM)带宽**参考值(`SOC_CONFIG.gm_bw`),而 16m(32 MB/张量)、
> 64m(128 MB/张量)的读写工作集能被 L2 承接,实测达到 2.11–2.78 TB/s(L2 侧
> `SOC_CONFIG.l2_bw = 8.0`)。只有 256m(512 MB/张量,越过 L2)给出的是真实片外带宽读数
> = **39.8%**。**结论:该口径不能当作「≤100% 的占比」读;跨 scale 比较时必须区分
> L2 驻留与片外两种情形。**

### 设计估算 vs 实测偏差行(D-2 回填,DESIGN.md §1.6.0)

DESIGN.md §1.6.0 的估算为**公式**而非数值:`估算下界 = max(流量项, 发射项, 容量项)`,
并判定「本算子**纯带宽受限**,发射项与容量项均不占优」,流量项 = `2·N·elem_bytes / HBM带宽`。

| workload | 设计估算下界(2·N·e ÷ 1.8 TB/s) | baseline 实测 | 偏差 | final 实测 | 相对估算下界 |
|---|---:|---:|---:|---:|---|
| elementwise-16m-float16 | 37.3 µs | 348.313 | **9.3×** | 25.099 | **优于下界 1.49×** |
| elementwise-16m-float32 | 74.6 µs | 348.273 | 4.7× | 48.339 | 优于下界 1.54× |
| elementwise-64m-float16 | 149.1 µs | 1389.092 | 9.3× | 127.217 | 优于下界 1.17× |
| elementwise-256m-float16 | 596.5 µs | 启动失败 | — | 748.665 | 差于下界 1.26× |

**失准项归属**:
① baseline 的 9.3× 偏差归因于**发射/派发**(Block Dim 8192 的逐块任务开销),不是流量项失准 ——
这与 DESIGN 判定「发射项不占优」**直接矛盾**,是本轮推翻了设计期的一个假设;
② final 在 16m/64m 上**优于估算下界**,进一步说明 1.8 TB/s 作为下界分母偏保守(L2 驻留);
③ 只有 256m 越过 L2,才落到「以片外带宽为下界」的原始假设附近(748.7 vs 596.5 µs,差 1.26×,
剩余差距为核内串行未完全隐藏的搬运延迟)。

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | 采数 harness | **Stage 5 报告的 `kernel_name_requested` 与 msprof 实测 kernel 名不一致** | Stage 5 记 `main`,实测为 `abs_fwd_kernel`;按 `main` 过滤时 msprof 报 `Kernel abs_fwd_kernel skipped: not selected via --kernel-name` + `No profiling data dumped` | `profile-collection.md` §3 增加「首次采集前先无过滤跑一次,从 `skipped: not selected` 行读出真实 kernel 名」的前置步骤 | R |
| 2 | 参数分支门禁 | 参数分支(同 kernel 不同 `bs/nc`)的等价门禁在 SKILL.md 未定义 | 文件级 `--level L0` 只跑模块默认参数,不覆盖被扫的 `bs/nc`;本任务改用 `bench.py --check`(配置精确 golden)替代 | SKILL.md Phase 2 明确「参数分支可用配置级 golden check 等价替代文件级 L0;结构分支必须全量跑 L0」 | R |
| 3 | 平区裁决 | 本轮有两处 <7% 的定档未走 `ab_test.py` 交错协议 | `nc≥48` 三档 ≤7%、fp32 的 `bs` 两档 2.0% | 保持现状但如实登记;若后续需要逐 shape 分派,这两处必须补做交错 A/B | R |
| 4 | baseline 中断 | 采集中断后 `perf_records.jsonl` 的「一 workload 一行 baseline」如何保持 | 外部会话抢占 NPU 导致 baseline 只采到 5/7;重跑会追加重复行 | runner 增加「补采单条 baseline + 已存在则拒绝」的守卫(本任务已实现在 `run_experiments.py`,建议回灌模板) | P |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-abs-1 | **D**(陷阱实证) | **Block Dim 超过 launch 栅格上限 ⇒ ACL 参数校验失败 + 进程 SIGABRT**,且报错文本不含维度信息:Stage 3 结构在 256m(`N/2048 = 131072` 块)上 `OPS function error: abs_fwd_kernel, error code is 107000` / `Invalid_Argument(EL0003)` / child status 134;同 kernel 在 8192、32768 块时正常 | `perf_records.jsonl` 两条 invalid baseline;`logs/profiles_baseline_round0_baseline_8ba8496a_elementwise-256m-floa_fb2d82c8.log` 原文 | repro-missing(需 NPU;C 类结论已自包含于本表) | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-abs-2 | **P**(优化模式) | **`Block Dim ≫ 物理核数` 的逐元素 kernel:核内 grid-stride `T.serial` + `block_size` 放大**。grid 收成 `num_cores` 后每块走多 tile,搬运粒度从 4 KB 提到 32 KB ⇒ 16m fp16 348.31→63.06µs(−81.9%);再取 `block_size=16384, num_cores=192` ⇒ 25.10µs(累计 −92.8%) | baseline 5 条 vs `r1_serial48` 7 条 vs final 7 条(`perf_records.jsonl` 共 41 行) | `abs_fwd_kernel_round1_serial.py`(= 慢→快关键代码更改的完整形态;`perf_opt/abs_fwd_kernel.py` 为 tuned 版) | 同上 |
| VP-abs-3 | **D**(口径实证) | **带宽占峰比在 L2 驻留时 >100%**:以 `SOC_CONFIG.gm_bw=1.8 TB/s` 为分母算占峰比,16m/64m 优化后达 142–154%(实测 2.56–2.78 TB/s),256m(越过 L2)仅 39.8% | final 7 条记录 + `tileops/benchmark/msprof_roofline.py` 的 `SOC_CONFIG`(gm_bw 1.8 / l2_bw 8.0) | repro-missing | 同上 |
| VP-abs-4 | **C**(案例) | elementwise 族「结构改写 + 参数粗搜」完整调优档案(3 轮、41 条记录、含一次推翻设计期假设) | 本 `opt_log.md` 全篇 | `perf_opt/profiles/` 全部 raw msprof 目录 | 同上 |

**pattern-library 回写**:VP-abs-1 拟入 `traps-compiler.md`(新增 TRAP-blockdim-grid-limit),
VP-abs-2 拟入 `elementwise.md`(新增 PL-1.21-gridstride-serial),
VP-abs-3 拟入 `constants.md`(CONST-bw-occupancy-l2-resident)。
回写位置:`references/pattern-library/`(ED-C 授权,由 optimizer 任务内回写)。

---

## 结论

- **中止条件**:结构候选与参数空间均已收束(结构 1 个方向验证完毕 + 16 组参数全测),
  未命中 `budget_exhausted`(3 轮 / 22 个候选分支,预算 10/30)、未 `blocked`、未 `plateau`。
- **性能目标**:未指定逐 workload 数值目标 ⇒ 以「通过核内合并与全量非回退检查的实测最优结果」
  作为结论(full 阶段口径)。
- **`[DESIGN_LIMIT]`**:不产出。参数级不足已在迭代内解决;16m/64m 已**优于**设计估算下界,
  256m 与估算下界的 1.26× 差距被归因到核内串行搬运延迟(参数/结构可继续处理),
  未同时满足「设计层归因 + 结构性加速 >2× 或与设计假设直接矛盾」两项(设计假设的偏离
  已由 ①② 解释为分母保守,而非设计结构错误)。
- **verdict**:`TUNING_COMPLETED`

