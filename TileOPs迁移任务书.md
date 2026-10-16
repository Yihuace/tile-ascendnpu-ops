# TileOPs 迁移任务书（GPU → Ascend NPU）

单文件交付：给 Agent 的输入看 §1；执行与交付规范看 §2–§4；本次批次的事实看 §5–§7。
规则细节与仓库内权威文件冲突时，以 `{NPU_WORKSPACE}/.opencode/agents/tilelang-op-conductor.md`
及其 `conductor-scenarios/*.md`、`.agents/skills/_shared/standards/*` 为准。

---

## 0. 环境与前置

### 0.1 路径

| 变量 | 含义 |
|---|---|
| `{NPU_WORKSPACE}` | TileLang-Ascend 源码仓（已按 `install_ascend.sh` 编译安装） |
| `{NPU_TILEOPS}` | `{NPU_WORKSPACE}/examples/TileOPs`，迁移产物的落位根 |
| `{GPU_REPO}` | GPU 侧 TileOPs 仓（迁移的 source of truth） |
| `{OP_DIR}` | conductor 解析出的算子目录（harness 迁移为 `{NPU_TILEOPS}/tileops/kernels/{family}/{op_slug}`） |

### 0.2 环境清单（缺一不可）

Ubuntu 20.04+ / Python 3.9–3.11 / torch_npu（≥2.6.0）/ CANN 8.5.0（toolkit + 芯片 kernels）/ Ascend 设备（历史基线 Ascend910B2C）/ gcc ≥9.4、glibc ≥2.27、cmake ≥3.18、ninja ≥1.12 / msprof（性能唯一合法口径，event 计时已弃用）/
`cd {NPU_TILEOPS} && pip install -e ".[dev]"` 后 `tileops-report list` 可用 /
`python3 {NPU_WORKSPACE}/.agents/tools/{statectl,standards_check,kb_search}.py` 可运行。

### 0.3 必做前置：源仓适配层（不做则全批退化为 plain）

上游 TileOPs 是 `src/tileops/…` 布局且 manifest 无 `source:` 块，而工作流按
`{GPU_REPO}/tileops/manifest/*.yaml` + `source.*` 定位工件，因此必须先在
`{NPU_WORKSPACE}` 侧补三处（不动上游仓）：

1. conductor 的 harness 判定加 src-layout 回退：
   `MANIFEST_DIR=$([ -d {GPU_REPO}/tileops/manifest ] && echo {GPU_REPO}/tileops/manifest || echo {GPU_REPO}/src/tileops/manifest)`
2. `.agents/skills/add-npu-op/scripts/extract_tl_kernel.py` 的 `load_manifest_entry()` 增加
   `src/tileops/manifest` 回退，并在 `source` 缺失时回退读 §0.4 的映射文件。
3. 生成 `{GPU_REPO}/.migration_source_map.json`（§5 表的机器可读版），提供
   `op → {op_class, kernel_file, kernel_map, test, bench, workload}`。

改完自检：`python3 {NPU_WORKSPACE}/.agents/tools/standards_check.py check`。

### 0.4 判定失败的含义

harness 判定是四条 shell 测试（`&&` 串联，任一条失败即判 plain，且**不报错、静默降级**）：

```bash
cd {GPU_REPO}
test -d tileops/manifest  && echo "1 ok" || echo "1 FAIL"
test -d tests/ops         && echo "2 ok" || echo "2 FAIL"
test -d benchmarks/ops    && echo "3 ok" || echo "3 FAIL"
grep -rl "^<OpName>:" tileops/manifest/*.yaml
```

判成 plain ⇒ 无 Stage 0（7 文件脚手架）与 Stage 5（集成/统一报告），产物只落在
`examples/{op}/{func}/`，且 `statectl start 0|5` 会被 `E-NOT-IN-PLAN` 拦掉。**一旦判成
plain，必须停下上报卡点，不得继续。**

---

## 1. 命令（下达给 Agent）

```
【迁移任务】<OpName>     family = <elementwise | norm | reduction | ...>
gpu_repo_root = {GPU_REPO}

模式 = harness。若探测判成 plain，先停下告诉我卡在哪一条判定，不许静默降级。
交付 = §3 交付清单的全部证据 + §4 一页报告（缺 Stage 5 三护栏按未完成计）。
取数 = 所有数字来自命令输出或 run.json；禁止估算，禁止复述我给的数字。
纪律 = 「链路通过」≠「性能达标」；性能给 per-shape 表并与同类已迁移算子横向对照。
版本 = 报告必须带版本戳（tilelang build/commit + CANN + 设备型号 + torch/torch_npu + python）。
```

多算子批量时，逐个下达（每算子一份独立状态与报告），不要合并成一次任务。

---

## 2. 执行规则

| # | 规则 | 验证方式（要求贴实际输出） |
|---|---|---|
| R1 | 不静默降级：先跑 §0.4 四条判定并逐条报告；判成 plain 必须停下说明卡点 | §0.4 |
| R2 | 不跳阶段：Stage 0 →（1→2→3）×N 函数 → Stage 5；Stage 4 本批跳过；禁止越级调度 | `statectl show --dir {OP_DIR}` |
| R3 | 逐阶段过门禁：`complete` 内部强制 gate；异常走 `fail`，不绕行 | `statectl gate N`、`statectl verify --dir {OP_DIR}` |
| R4 | Stage 5 三护栏必核 | §2.1 |
| R5 | 证据边界：每个数字可回溯到某条命令或 run.json 字段 | 命令与字段同列 |
| R6 | 结论纪律：正确性节不掺性能；性能节必须给同类横向对照与反常信号 | §4 |
| R7 | 版本戳必填：无戳结论不得当作可复用事实 | §2.2 |
| R8 | 不可行即登记：按 capability-gap 规则登记并停下，不得改语义绕过 | `.agents/evolution/capability-gaps.md` |

状态机命令（工作目录 `{NPU_TILEOPS}`）：

```bash
CTL="python3 {NPU_WORKSPACE}/.agents/tools/statectl.py"
$CTL init --dir {OP_DIR} --project <project> --op <op_slug> \
     --scenario migration --migration-mode harness --stage-plan "0,1,2,3,5"
$CTL migration init --dir {OP_DIR} --op-slug <op_slug> --family <family> \
     --meta-path {OP_DIR}/.migration_meta.json --functions <f1> <f2> ...
$CTL start 0 && $CTL complete 0 --meta {OP_DIR}/.migration_meta.json
$CTL start 1 && $CTL complete 1     # designer → examples/{op_slug}/{func}/DESIGN.md
$CTL start 2 && $CTL complete 2     # reviewer → REVIEW.md（结论: 通过/不通过）
$CTL start 3 && $CTL complete 3     # developer → {func}.py（target="npuir"，L0/L1 必须真跑）
$CTL verify --dir {OP_DIR}
```

Stage 0 抽取脚本与 Stage 5 集成脚本：

```bash
python3 .agents/skills/add-npu-op/scripts/extract_tl_kernel.py \
    --op-name <OpName> --gpu-repo-root {GPU_REPO} --out <op_slug>/_extracted.py

cp -r tileops/kernels/<family>/<op_slug>/ tileops/kernels/<family>/<op_slug>.bak   # 幂等重跑会覆盖人工补丁
python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py \
    --meta tileops/kernels/<family>/<op_slug>/.migration_meta.json
```

### 2.1 Stage 5 集成三护栏（R4）

```bash
cd {NPU_TILEOPS}
grep -n "wrapper_rewritten"  tileops/kernels/<family>/<op_slug>/integration_log.md   # 必须 true
grep -n "already integrated" tileops/kernels/<family>/<op_slug>/integration_log.md   # 出现即人工复核
python -c "import importlib; w=importlib.import_module('tileops.kernels.<family>.<op_slug>'); print(w.<func>.__module__)"
# ↑ 必须指向 baseline kernel，不能是 perf_opt 旧谱系
```

判读规则：护栏 2 命中 `already integrated` 是**幂等重跑的常态信号**（脚本同时写出
`wrapper_rewrite_action`），不得单凭该关键字判失败——必须与护栏 3（`__module__` 指向
baseline）联合判定。真正的风险形态是「日志说已集成、但实际生效的是 `perf_opt` 旧谱系」。

### 2.2 版本戳（R7）

```bash
python -c "import tilelang,torch,torch_npu; print(tilelang.__version__, torch.__version__, torch_npu.__version__)"
npu-smi info | head -20      # 设备型号；另附 CANN 版本与 tilelang build/commit
```

---

## 3. 交付清单（提交自查，缺项即未完成）

| # | 交付项 | 证据形式 |
|---|---|---|
| D1 | 路由与判定 | scenario / migration_mode / stage_plan + §0.4 四条输出 |
| D2 | 流程与时间线 | `statectl show --dir {OP_DIR}`、`statectl verify --dir {OP_DIR}`、`.task_timeline.jsonl` 摘要 |
| D3 | Stage 0 产物 | 7 文件路径 + `.migration_meta.json` + 提取函数数 N |
| D4 | Tier 1 结构校验 | `python -c "import tileops"` / `ruff check` / `pytest tests/ops/test_<op_slug>.py --collect-only` 输出 |
| D5 | 设计与检视 | 每函数 DESIGN.md、REVIEW.md 路径 + `结论:` 字面量 |
| D6 | 精度门禁 | Stage 3 内嵌 L0/L1 的**实测**输出（非设计意图） |
| D7 | 集成护栏 | §2.1 三项结果 |
| D8 | 正确性事实 | `pytest tests/ops/test_<op_slug>.py -v` 的用例数/通过数/失败项；`max_abs_err` 及其解释（整型按位类为 0 属构造必然） |
| D9 | 性能事实 | `pytest benchmarks/ops/bench_<op_slug>.py -v` + `tileops-report run --op <OpName> --prof-mode msprof --kernel-name main` 的报告目录与 run_id；per-shape 表（Label/Latency/Ratio/Shape/DType/Kernel/BW）+ `msprof_case_count` / `profiler_fallback_count` / `invalid_profile_count`；**benchmark 用例构成表**（收集数 vs 进 msprof 数：benchmark 阶段以 `-m "not smoke"` 排除 smoke 例，见 `tileops/reporting/runner.py:100-122`；两数不等时必须逐例说明） |
| D10 | 版本戳 | §2.2 输出 |
| D11 | 未决项与风险 | capability gap（若有）/ 已知缺陷对本任务的影响 / 下一轮入口 |

run.json 取数字段：

```bash
grep -nE '"status"|"correctness_passed"|"correctness_tests"|"benchmark_passed"|"benchmark_tests"|"msprof_case_count"|"profiler_fallback_count"|"invalid_profile_count"|"max_abs_err"|"run_id"|"git_commit"' run.json
```

字段位置提示（实测）：`correctness_tests` 位于 `summary` 下，**顶层同名键不存在**；
非 git 工作区 `git_commit` 为 `null`；`benchmark_tests` 是用例数，
`summary.msprof_case_count` 是真正进 msprof 的用例数。

设备口径提示（实测）：`tileops/benchmark/msprof_roofline.py` 的 `SOC_CONFIG` 只登记
`Ascend910B2C / 910B1 / 910B / 910A`，未登记型号走 `DEFAULT_CONFIG`（`num_cores=48`、
`gm_bw=1.8`，见 :91-105、:168）。报告中的 Ratio/Bandwidth 若取自 msprof 解析值
（`benchmark_base.py` 的 `gm["ratio"] / gm["bw"]`），必须与该兜底常数区分标注，
否则默认核数/带宽会被误当作实查值。

### 3.1 性能口径（SOL，**以官方文档为准**）

性能唯一口径 = msprof `visualize_data.bin` 的 Roofline **`GM Read + Write`** 条目，其定义为标准 Roofline **SOL（speed-of-light）利用率**：

```
Ratio = Performance / min(AI × 带宽上限, 计算上限)
AI          = total_fops / total_bytes          # total_bytes = Σ(L2 读未命中) × 128
Performance = total_fops / task_duration
计算上限(computility) = (峰值 ppc × 指令占比) × 频率 × 核数   # 按本算子指令构成加权
fops        = Σ(FP32指令 × 64 + MISC指令 × 32)   # 计算单元指令计数，非"有用 FLOPs"
```

依据（仓库内官方文档）：`examples/TileOPs/docs/roofline_metrics_analysis.md`（公式逆向验证）、`examples/TileOPs/docs/roofline_types_and_latency_analysis.md`（多级 roofline + `advice` 字段 + 阈值）。**本任务书与上述文档冲突处，以文档为准。**

**关键（勿误读）**：
- `Ratio` 是 **SOL 利用率**，**不是"带宽利用率"**。分母 `min(AI×带宽, 计算上限)` 随算子 AI 切换：
  - `AI × 带宽 < 计算上限`（**访存受限**）→ Ratio = 带宽利用率；
  - `AI × 带宽 ≥ 计算上限`（**算力受限**）→ Ratio = 计算单元利用率（Cube 或 Vector）。
- 判据须**按 AI 分类**（平衡点 = 计算上限 / 带宽上限）；达标口径、瓶颈判定、阈值一律以官方文档为准，**不得**把 Ratio 当成"带宽利用率"。
- 报告须给出：AI、瓶颈类型（访存/算力受限）；如可得，附多级 roofline（Memory Unit / Memory Transfer / Pipeline）与 bin 的 `advice` 字段用于定位。

#### 达标线（按瓶颈类型，暂行）

| 类别 | 达标线（SOL 利用率 §3.1） | 依据 |
|---|---|---|
| **访存受限** | **≥ 70%** | 实测纯访存 elementwise 已达 72%–83%（Add/Sub/Lerp/ClampMin/ClampScalar），可达性已证；对齐"A/N Vector 70%"先例 |
| **算力受限（Vector）** | **≥ 60%** | 计算单元打满更难；Hardswish 61.6% 为当前参考上限 |
| **混合类（MixCV / Cube 主导）** | **≥ 40%（暂定）** | MixCV 有跨 scope/中继开销，天花板低于纯 Vector（官方 Mix 示例 GM ratio 25%、历史 SSDChunkScan 0.13–20.3%）；暂定，待用 MHA/SSD 标定后调整 |
| **微型 shape（2–4µs）** | **豁免** | 固定启动开销主导其耗时，带宽/算力利用率物理不可达（已论证） |

- **硬底线**：kernel 时间（`msprof op Task Duration`）**必须变短**——时间不降即未优化（判据不满足也不得硬凑）。
- **达标定义** = `kernel 时间下降` **且** `对应类别的 SOL 利用率 ≥ 对应线`。

报告禁开 pytest-xdist（`-n` / `--numprocesses` 会被 CLI 拒绝）；报告运行必须串行。

---

## 4. 报告骨架（一页，六节）

| 节 | 写什么 | 引用 |
|---|---|---|
| 1 身份与范围 | op_name / family / op_slug / gpu_repo_root / 报告目录与 run_id / 版本戳 | D10、D9 |
| 2 迁移流程证据 | 路由与判定、Stage 0 产物、Tier 1、逐函数 DESIGN+REVIEW、Stage 5 三护栏 | D1–D7 |
| 3 正确性事实 | 用例与通过率、误差指标及其是否构造必然、链路是否通过（不掺性能） | D8 |
| 4 性能事实 | msprof per-shape 表、采集计数、同类算子横向对照、反常信号、水位结论 | D9 |
| 5 未决项与风险 | capability gap、已知缺陷影响、优化入口 | D11 |
| 6 复核入口 | 全部命令与路径，逐条可照抄重跑 | 全部 |

---

## 5. 本次批次：算子核对与源定位（19 接口全量交付）

基线：`{GPU_REPO}` = 上游 TileOPs `1238a9f`（2026-10-12）。行号为定义起始行。

**本任务全量交付 19 个独立 Op 接口**，不以"底层 Kernel 复用"为由合并或删减。其中 13 个接口名与基线注册名一致；另 6 个在基线 manifest 中已发生**改名 / 合并 / 拆分**（改动均发生在基线之前），本任务将它们**作为独立 Op 接口交付**，交付名沿用本清单原名，并在「基线状态」列如实标注与基线实际注册条目的映射。

「基线状态」：`同名` = 1238a9f manifest 即为此名；`已改名`/`已合并`/`已拆分` = 该名字在基线中已不存在，本任务按独立接口重新暴露（底层复用对应 Kernel）。

| 算子（交付接口） | 基线状态 | manifest 条目（基线 1238a9f） | kernel 层 | op 层 | 交付说明 |
|---|---|---|---|---|---|
| `GeluAndMulFwdOp` | 同名 | elementwise_fused_gated.yaml:31 | kernels/elementwise/activations.py:386（GeluAndMulFwdKernel） | ops/elementwise/activations.py:273 | 独立交付 |
| `GeluTanhAndMulFwdOp` | 同名 | elementwise_fused_gated.yaml:56 | kernels/elementwise/activations.py:405 | ops/elementwise/activations.py:284 | 独立交付 |
| `IsnanFwdOp` | 同名 | elementwise_unary_math.yaml:424 | kernels/elementwise/comparison.py:186 | ops/elementwise/comparison.py:86 | 独立交付 |
| `IsinfFwdOp` | 同名 | elementwise_unary_math.yaml:443 | kernels/elementwise/comparison.py:194 | ops/elementwise/comparison.py:99 | 独立交付 |
| `IsfiniteFwdOp` | 同名 | elementwise_unary_math.yaml:462 | kernels/elementwise/comparison.py:202 | ops/elementwise/comparison.py:112 | 独立交付 |
| `ClampScalarFwdOp` | 同名 | elementwise_unary_activation.yaml:323 | kernels/elementwise/clamp.py:24（ClampFwdKernel） | ops/elementwise/clamp.py:74 | 独立交付（双边截断） |
| `ClampMinFwdOp` | 已合并（现 ClampScalarFwdOp / ClampTensorFwdOp） | elementwise_unary_activation.yaml:323（ClampScalarFwdOp） | kernels/elementwise/clamp.py:24 | ops/elementwise/clamp.py:74 | 独立交付 Op 接口（单边 min，复用 ClampFwdKernel） |
| `HardsigmoidFwdOp` | 同名 | elementwise_unary_activation.yaml:106 | kernels/elementwise/activations.py:169 | ops/elementwise/activations.py:113 | 独立交付 |
| `HardswishFwdOp` | 同名 | elementwise_unary_activation.yaml:82 | kernels/elementwise/activations.py:151 | ops/elementwise/activations.py:107 | 独立交付 |
| `FusedAddLayerNormFwdOp` | 同名 | norm.yaml:120 | kernels/norm/fused_add_norm.py:144 | ops/norm/fused_add_layer_norm.py:14 | 独立交付 |
| `AddFwdOp` | 同名 | elementwise_binary.yaml:98 | kernels/elementwise/arithmetic.py:43 | ops/elementwise/arithmetic.py:75 | 独立交付 |
| `LerpFwdOp`（标量插值） | 已拆分（现 LerpScalarFwdOp + LerpTensorFwdOp） | elementwise_binary.yaml:288（LerpScalarFwdOp） | kernels/elementwise/arithmetic.py:441 / :596 | ops/elementwise/arithmetic.py | 独立交付 Op 接口（标量 lerp，区别于张量版 `LerpTensorFwdOp`） |
| `SubFwdOp` | 同名 | elementwise_binary.yaml:127 | kernels/elementwise/arithmetic.py:53 | ops/elementwise/arithmetic.py:86 | 独立交付 |
| `BatchNormBwdOp` | 同名 | norm.yaml:250 | kernels/norm/batch_norm.py:2146（Wide:1981 / Split:2044） | ops/norm/batch_norm.py:182 | 独立交付 |
| `GLADecodeOp`（GLA decode） | 已改名（现 GLAFwdOp / GLARecurrentFwdOp） | linear_attention.yaml:528（GLAFwdOp）/ :396（GLARecurrentFwdOp） | kernels/linear_attention/gla/recurrent.py（GLADecodeKernel） | ops/linear_attention/gla/ | 独立交付 Op 接口 |
| `Conv3dBiasFwdOp` | 已合并（现 Conv3dFwdOp，bias 可选输入） | convolution.yaml:128（Conv3dFwdOp） | kernels/convolution/conv3d.py（Conv3dKernel / Group / Ndhwc） | ops/convolution.py:263 | 独立交付 Op 接口（带 bias） |
| `Conv1dFwdOp` | 同名 | convolution.yaml:6 | kernels/convolution/conv1d.py:745（另有 641/874/929/1074） | ops/convolution.py:75 | 独立交付 |
| `Conv1dBiasFwdOp` | 已合并（现 Conv1dFwdOp，bias 可选输入） | convolution.yaml:6（Conv1dFwdOp） | kernels/convolution/conv1d.py:745 | ops/convolution.py:75 | 独立交付 Op 接口（带 bias） |
| `GatedDeltaNetPrefillFwdOp` | 已改名（现 GDNFwdOp） | linear_attention.yaml:66（GDNFwdOp） | kernels/linear_attention/gdn/prefill.py:29（decode 见 gdn/decode.py:18） | ops/linear_attention/gdn.py:19 | 独立交付 Op 接口 |

**交付口径声明（纠偏）：**

1. 严禁以"底层 Kernel 复用"为由，在 L2 Python 层省略 `ClampMinFwdOp`、`LerpFwdOp`、`Conv1dBiasFwdOp`、`Conv3dBiasFwdOp`、`GLADecodeOp`、`GatedDeltaNetPrefillFwdOp` 等独立 Op 接口的封装与暴露。
2. 交付名沿用本清单原名；基线实际注册名（如 `GDNFwdOp` / `GLARecurrentFwdOp` / `GLAFwdOp`）**仅作映射参考**，不作为交付接口名。
3. `LerpFwdOp`（标量插值）与 `LerpTensorFwdOp`（张量插值）是两个独立算子；标量版必须独立开发与测试。
4. 上表 19 行为本任务接口清单。基线中的裸卷积 `Conv3dFwdOp`（无 bias）**不在**本 19 清单内；如另行需要，单列，不与本清单混计。

---

## 6. 分批与出口标准

**总计 19 个独立 Op 接口**（而非按当前注册名去重后的 16 个）。分批计划：

| 批 | 算子 | 出口标准（不满足不进下一批） |
|---|---|---|
| 1a 链路验证 | ClampScalarFwdOp、HardswishFwdOp | 各走完 Stage 0→5；`wrapper_rewritten=true`；报告正确性 100%；全程无需人工补丁 |
| 1b elementwise（常规） | HardsigmoidFwdOp、AddFwdOp、SubFwdOp、GeluAndMulFwdOp、GeluTanhAndMulFwdOp | 同批全绿 + 报告产出 |
| 1c elementwise（补齐） | **ClampMinFwdOp、LerpFwdOp** | 必须提供独立的 Python 测试用例与 API 入口；同批全绿 + 报告产出 |
| 1d 谓词算子 | IsnanFwdOp、IsinfFwdOp、IsfiniteFwdOp | 前置探针：bool 输出 + vcmp 在 npuir 上可落地；不可行则登记 gap 并停批 |
| 2 中风险 | FusedAddLayerNormFwdOp、BatchNormBwdOp | 验证 Norm 系列 row-reduction 模板（全 N 驻留 + M 行块分核 + 尾块 `T.min` 双显式 slice）在 NPU 的精度与性能基线 |
| 3 高风险（卷积） | Conv1dFwdOp、**Conv1dBiasFwdOp**、**Conv3dBiasFwdOp** | 带/不带 bias 的接口均需通过独立测试；每个先过可行性预研探针；不可行时登记 gap，只交付 plain 产物 |
| 4 高风险（线性注意力） | **GLADecodeOp**、**GatedDeltaNetPrefillFwdOp** | 每个先过可行性预研探针；重点关注 chunk 并行与状态传递在 NPU L1 Buffer 上的映射；不可行时登记 gap，只交付 plain 产物 |

---

## 7. 已知坑（必须遵守）

1. **plain 是静默降级**：判定失败不报错。开工前先跑 §0.4；若 §0.3 适配层未就位，整批会落到 plain。
2. **`source:` 缺失是第二段失败**：即使判定过了，`extract_tl_kernel.py` 也会因上游无 `source` 块抛
   `ValueError: op '...' has no 'source' section in manifest`——必须靠 §0.3 的映射文件补齐。
3. **`integrate_kernel.py` 有已知未修缺陷**（登记在 `.agents/evolution/queue.md`，状态 pending）：
   VP-2026-0032（不清理上一轮 `perf_opt/`，旧谱系静默生效）、VP-2026-0055（契约名缺失只 `[warn]`，
   smoke 期才 ImportError）、VP-2026-0056（extracted-import 契约缺口）、VP-2026-0064（幂等检查被
   注释行骗过，`wrapper_rewritten=false`）、VP-2026-0012（幂等重跑整文件覆盖人工补丁 → 重跑前先备份）、
   VP-2026-0140（`_select_config` 成对 import 需人工补）。§2.1 三护栏就是为这些设的。
4. **性能口径只有 msprof**：`profiler_fallback_count` 必须为 0；报告串行，禁 `-n`。
5. **版本戳规则**：缺陷与结论都绑定工具链版本，升级 tilelang/CANN 后旧结论必须重验。
6. **优化是另一次任务**：harness 迁移的 `stage_plan` 不含 Stage 4，优化需另发 `optimize` 场景
   （Stage 4 + 强制 L0/L1 回归）；迁移报告里不得把"可作为 baseline"写成"性能已达水位"。
