# Conv3dBiasFwdOp 复核记录

结论: 通过（2026-11-19 按 op-dev 门禁字面量规范化；逐项结论见下表）

## 复核范围

- Op：`tileops/ops/convolution.py`（`Conv3dFwdOp` / `Conv3dBiasFwdOp`）
- Kernel：`tileops/kernels/convolution/conv3d/`
- Manifest / Workload / 测试 / Benchmark：`Conv3dBiasFwdOp`、`Conv3dWorkload`、`tests/ops/test_conv3d_bias.py`、`benchmarks/ops/bench_conv3d_bias.py`

## 检查项与结论

| 检查项 | 结论 |
|---|---|
| 语义 | 与 `F.conv3d` 一致；bias 必选、逐输出通道 |
| 布局/参数 | NCDHW；stride/padding/dilation 支持 int 与 3-tuple；groups 支持 |
| Kernel 路径 | dense NCDHW implicit GEMM + MixCV bias；NDHWC/padded-N 作为回退分支 |
| 权重复用 | R1 权重 L1 常驻，消除循环内重载；r3d-stem-bias 1.99×（第二轮 2.36×） |
| 回退 | 编译期 shape 级回退，video-down/unet 在无收益时走 baseline 等效实现 |
| 测试 | 17/17 通过（含 smoke、stride=2、非整除 shape）；无 skip |
| 性能 | r3d-stem-bias 6161.50→2588.7µs（2.36×）；unet 6428→301.6µs（21.3×）；video-down 87.00→86.42µs（噪声内） |
| 风险 | implicit GEMM 的 27-tap 重复读 + 小通道 16-fractal 填充是剩余结构上限；`bishengir-compile` 在部分 shape 上存在不稳定 |

## 机械复核（D-1/D-5）

- **verify_equiv.py 重跑**：N/A —— Conv3dBiasFwdOp 的采纳项为实现结构 / 数据布局 / 融合位置变更，没有减少算术的等价变形，
  按 `gate-and-retry.md` Stage 1「无优化空间」豁免（依据见 DESIGN §1.6.1）。
- **design_calc_check**：`python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/convolution/conv3d/conv3d_kernel/conv3d_bias_evidence/DESIGN.md` 输出：
  - `r3_metrics`: **pass**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值四口径齐全）
  - `ub_budget`: skip（无逐 buffer 表，已按 DESIGN §4 人工核对峰值 < 192 KB）
  - `l0c_budget`: skip（Cube 路径已在 DESIGN §2/§4 人工核对 L0C 128 KB 预算）
  - `core_split`: skip（解析维度不足，分核三要素已在 DESIGN §5 人工核对）

### 8. 算法优化分析 — pass

- **调研四问**：DESIGN §1.6.0 已补等价化简、在线算法（不适用）、复杂度四口径、硬件亲和，均带依据。
- **负向断言核对**：direct 7 层循环因向量化差/L1 复用低弃选（`MIGRATION_INDEX.md` 表 A Conv3dBias 行）；
  裸 `Conv3dFwdOp` 不在 19 个任务书算子内（任务书 §5）——均见 DESIGN §1.6.3。
- **独立复核**：官方 report **17/17 passed**，max_abs 3.125e-2（fp16 r3d-stem）/ 5.0e-1（bf16 unet-encoder），
  与 PyTorch conv3d golden 逐用例对齐（`reports/tileops/20261023_184106_697102_Conv3dBiasFwdOp`）。
- **结论**：implicit-GEMM / polyphase / bias 融合只改实现结构与数据布局，数学公式不变，无需 `verify_equiv.py`。

## 结论

代码、测试与报告一致；性能收益来自权重 L1 常驻与 tap 打包。建议按“迁移 + 优化”提交，但最终以本次端到端 run 是否通过为准。
