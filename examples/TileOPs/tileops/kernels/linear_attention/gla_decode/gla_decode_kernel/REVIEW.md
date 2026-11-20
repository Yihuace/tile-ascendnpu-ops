# GLADecodeOp 复核记录

结论: 通过（2026-11-19 按 op-dev 门禁字面量规范化；逐项结论见下表）

## 复核范围

- Op：`tileops/ops/linear_attention/gla_decode.py`
- Kernel：`tileops/kernels/linear_attention/gla_decode/`
- Workload / Manifest / 测试 / Benchmark：`GLADecodeWorkload`、`manifest/linear_attention.yaml`、`tests/ops/test_gla_decode.py`、`benchmarks/ops/bench_gla_decode.py`

## 检查项与结论

| 检查项 | 结论 |
|---|---|
| 语义与公式 | 与 `fused_recurrent_gla` 单 token decode 一致；`scale` 非正时取 `DK**-0.5` |
| 输入/输出 | `q/k/v/gk/state` → `o/new_state`；BHD；state 函数式返回，无原地写 |
| dtype | 支持 fp16/bf16/fp32；bf16 输入 + fp32 累加；full shape 以 bf16 验证 |
| tiling | `k_tile=32` 在 192KB UB 内；`k_tile=64` 超容量，已否决 |
| 分派 | `B*H<=40` flat；`B*H>40` persistent 40 核；编译期分派 |
| 边界 | 小 shape smoke（B=2,H=4,DK=DV=64）通过；无空 tensor/异常 shape 崩溃 |
| 测试 | 11/11 通过；无 skip、无降低容差 |
| 精度 | 与 fp32 golden 对比通过；报告最大误差字段为 N/A（多输出/布尔统计口径） |
| 性能 | 5 个 serving shape ratio 3.17–19.94%；相对基线 Task Duration −24.6%~−35.2% |
| 风险 | SOL 低是 decode latency-bound 的结构结果，不是 tiling 退化；已在 PR 中如实说明 |

## 机械复核（D-1/D-5）

- **verify_equiv.py 重跑**：N/A —— GLADecodeOp 的采纳项为调度 / 内存别名 / 精度契约变更，没有减少算术的等价变形，
  按 `gate-and-retry.md` Stage 1「无优化空间」豁免（依据见 DESIGN §1.6.1）。
- **design_calc_check**：`python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/linear_attention/gla_decode/gla_decode_kernel/DESIGN.md` 输出：
  - `r3_metrics`: **pass**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值四口径齐全）
  - `ub_budget`: skip（无逐 buffer 表，已按 DESIGN §4 人工核对峰值 < 192 KB）
  - `l0c_budget`: skip（无 Cube 路径（纯 Vector））
  - `core_split`: skip（解析维度不足，分核三要素已在 DESIGN §5 人工核对）

### 8. 算法优化分析 — pass

- **调研四问**：DESIGN §1.6.0 已补等价化简、在线算法、复杂度四口径、硬件亲和，均带依据。
- **负向断言核对**：`k_tile=64` 超 192 KB UB、DV split +25.7%、b1-h8 5-block +45%，分别有 REVIEW tiling 项与
  `perf_opt/opt_log.md` R49 负结果佐证（DESIGN §1.6.3）。
- **独立复核**：官方 report **11/11 passed**；R46/R47 5 shape × 10 轮随机输入共 50 次，max abs err 8.28e-3 < 2e-2
  （`reports/2026-11/2026-11-18-三算子9.3固化.md` §3.2）；fp16 state 属精度契约变更，已在同文 §3.2 明示。
- **结论**：采纳项与 §1.6 逐项一致，无算术化简项，无需 `verify_equiv.py`。

## 结论

代码、测试与报告一致，未发现阻塞问题。建议按“迁移 + 优化”提交；性能判据采用 Task Duration，不宣称达到 70% GM SOL。
