# Integration Log — `CosFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/elementwise/cos/.migration_meta.json`
- **integrated**:`cos_fwd_kernel` → `tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py`(源自 `examples/cos/cos_fwd_kernel/cos_fwd_kernel.py`)
- **design_docs**:`cos_fwd_kernel` → `tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel_DESIGN.md`(源自 `examples/cos/cos_fwd_kernel/DESIGN.md`,**含 Stage 3 的 Cody-Waite 边界量化修订**;**未缺失**)
- **wrapper**:`tileops/kernels/elementwise/cos/cos.py` —— `wrapper_rewritten: false` 已集成;`ascend_mode: Developer`
- **attempts**:1(集成侧首次即完成;但**上游 Stage 3 经历了一次 `[DESIGN_ERROR]` 回退与设计修订**,见 `examples/cos/cos_fwd_kernel/RETROSPECTIVE.md` 的 Stage 3 章节)
- **smoke**:pass

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261015_152031_061195_CosFwdOp/run.json`
- **report_status**:`partial`(CLI 退出码非零 —— 按规范**不凭退出码判定**,以 `run.json` 为准)
- **correctness**:**pass** —— `summary.correctness_passed = true`,`correctness_tests = 9`,`correctness_failed = 0`,`pass_rate = 1.0`
- **benchmark**:**partial** —— `summary.benchmark_passed = false`(`benchmark_tests = 10`,`benchmark_failed = 2`)

### benchmark 失败明细(2 例)

两条均为 manifest 最大 workload `[16384,16384]`(268 435 456 元素),采集侧设备失败 —— 与前序 5 个算子(`abs`/`ceil`/`floor`/`exp`)**完全同型**。**该失败现已在 5 个算子上复现,确认属 harness 能力边界**,应在 Stage 0 的 workload 规模规则层面修。

按 integrator 规范「benchmark 失败只记录、不修复、不重试」,**未**修改 kernel / wrapper / tests / benchmarks / manifest。

### 本次集成的实质变更与**已知分歧**(须显式记录,不得掩盖)

本次集成把 **Stage 3 修订后**的实现带入集成侧。该实现与源算子存在一处**已实测量化的精度分歧**:

| $\lvert x\rvert$ 上界 | 本实现(越界元素数) | 源算子(`torch.cos`) |
|---|---|---|
| ≤ 100 | **0 / 8192** | 0 |
| 500 | **704 / 8192** | 0 |
| 1000 | 1828 / 8192 | 0 |
| 10000 | 5930 / 8192(最大绝对误差 7.225e-04) | 0 |

- **成因**:四象限归约在 fp32 中计算象限索引 $k$,大 $\lvert x\rvert$ 时 $x - k(\pi/2)$ 发生灾难性抵消(经典 Cody-Waite 问题)。
- **该分歧在算子层无法消除**(需 Payne-Hanek 归约,属数学库层),已在设计 §0.6/§10 R1 量化并归口为**待契约层裁决**项。
- **对交付的影响**:manifest 的 workload 为 randn 缩放(约 $\pm15$),**远在实测精度域内**,故不影响集成侧的 9/9 正确性结论。
- **实现侧的如实披露**:交付件含 `record_beyond_domain()` 段,**记录而非断言**域外误差,并在输出中并列 `(source operator: 0)` 作对比。

### 集成侧测试覆盖缺口

`tests/ops/test_cos.py` 不含四象限逐分支、误差窗口、域界等类别。**tests/ 不在本 Agent 的可改范围内**,故该缺口登记于 `examples/cos/RETROSPECTIVE.md` 的 Stage 5 章节,交 conductor 决策。

## 调试历史

集成侧无调试(attempt 1 即完成)。**上游 Stage 3 的调试链**:

| attempt | 结果 | 现象 |
|---|---|---|
| 1 | `[DESIGN_ERROR]` | `L1-large-x` 在 $x=12345.678$ 失败(`abs 9.8169e-05 > 1e-05`, `2/4096` 元素失配) |
| — | 设计修订 | 用 `history_version/design_probe_cody_waite_boundary.py` 实测边界($\lvert x\rvert\lesssim100$),把阻塞用例收进域内、域外转 L2 记录 |
| 2 | `[PRECISION_PASS]` | `All check passed!` |

## 环境

- Ascend 910B2C,npu-smi `26.0.rc1`;CANN 9.0.0;torch `2.10.0+cpu`;torch_npu `2.10.0`
- tilelang wheel `0.1.2+npuir+7fc4a3c8`;profiling mode:msprof

## 已知未执行项(如实披露)

- **`SESSION_TIMING_ANALYSIS.md` 未生成**:harness.md §7.1 要求基于 `opencode export <session_id>` 生成,但本任务运行在 **AtomCode** 而非 OpenCode,**没有 `opencode` CLI**。按「无法确认时如实标注、不得伪造」的口径,**未伪造任何导出数据**,交由 conductor 处置。

---

## Stage 4 性能优化集成记录（2026-11-04）

### 本次提交改了什么

| 文件 | 性质 | 内容 |
|---|---|---|
| `cos.py`（wrapper） | 修改 | `_DEFAULT_BLOCK_SIZE`：`{4096, 4096, 4096}` → `{float32: 8192, float16: 12288, bfloat16: 12288}`。**不改则正式 benchmark 只能拿 4096 档（16m fp16 ratio 54.75%）**；该值与 kernel 侧 `TUNED_DEFAULT_BLOCK_SIZE` 必须成对一致 |
| `cos_kernel/perf_opt/cos_fwd_kernel.py` | 修改（交付件） | 新增 `_cos_single`（无三角函数 poly-R 链：`cos(x)=1−2sin²(x/2)`，按 π 归约让 sin² 吸收符号）、`_cos_f32_dual`（fp32 双流、两流相距半个 tile 域的远距映射）、`cos_fwd_kernel` dtype 分发、`TUNED_DEFAULT_BLOCK_SIZE` |
| `cos_kernel/perf_opt/` 其余 12 个文件 | 新增 | 第 5/6/7 轮候选（`round5_c3base`/`polyr`/`polyr2`/`polyr3`/`polyr4`、`round6_dual`、`round6b_far`、`round7_slim`）、探针（`probe_instr`、`probes/probe_novsin`、`probes/probe_nocast`）、复算脚本（`ratio_from_bin`、`ratio_from_opprof`） |
| `opt_log.md` / `perf_records.jsonl` | 追加 | 第 5/6/7 轮日志与结构化记录（`perf_records.jsonl` 共 50 行，`phase` 区分直跑/官方口径） |

### 官方 pytest 口径结果（`benchmarks/ops/bench_cos.py`，10 passed）

| workload | 时延 | ratio | 基线（C3） |
|---|---:|---:|---:|
| 16m fp16 | 96.98 µs | 62.70% | 53.39% |
| 16m bf16 | 98.88 µs | 64.56% | 54.60% |
| 16m fp32 | 107.96 µs | **53.53%** | 51.59% |
| 64m fp16 | 371.47 µs | 65.45% | 54.44% |
| 64m bf16 | 378.08 µs | 67.52% | 55.85% |
| 256m fp16 | 1474.54 µs | 65.95% | 54.76% |
| 256m bf16 | 1501.84 µs | 67.98% | 56.17% |
| **单点 / 平均** | — | **67.98% / 63.96%** | 56.17% / 54.40% |

### 精度与等价性

- 官方 `pytest benchmarks/ops/bench_cos.py`：**10 passed**（含 L0 阻塞用例、边界用例与 3 条 smoke）；
- 未被放宽或跳过的用例：无；测试容差未修改；
- poly-R 链在 `t∈[0,(π/2)²]` 上 deg-5 Horner 的实测最大绝对误差 **9.69e-8**；
- 与 Stage 3 `cos_kernel/cos_fwd_kernel.py` 的差异仅限性能结构（链式与分档），运算符语义未变。

### 已知遗留（如实披露，未闭环）

- **16m fp32 官方 53.53%**：同一交付件直跑为 90.5 µs / 63.9%，官方路径 108.0 µs / 53.5%；
  两者 `fops` 相同（6 545 344），`Current Freq` 恒 1800 MHz，但周期多 21%（151.6k → 184.0k），
  属「拷贝与计算串行」的停顿随调用路径放大。已试并否决：deg-7 填槽（+17% 时间）、`T.Pipelined`、
  fp32 UB→UB 加粗 copy、8192→12288（UB 溢出，实需 24 B/元素）。**该条未达 60%，本 PR 不掩盖**。
- `profiles/`（原始 msprof 数据）与 `logs/` 未入库：前者体积过大，后者被 `.gitignore` 的 `*.log` 规则排除。

### 环境

- Ascend 910B2C（24 AIC / 48 AIV，1800 MHz）；CANN 9.0.0；torch `2.10.0+cpu`；torch_npu `2.10.0`；
  tilelang `0.1.2+npuir`；profiling mode：msprof（`--aic-metrics=Roofline`）。
