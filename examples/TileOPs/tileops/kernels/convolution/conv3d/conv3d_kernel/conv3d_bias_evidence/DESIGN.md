# Conv3dBiasFwdOp 设计说明

## 1. 语义

`torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)`，NCDHW 布局，bias 为必选逐输出通道张量：

```text
out[n, oc, od, oh, ow] = bias[oc]
    + Σ_{ci,kd,kh,kw} x[n, ci, id, ih, iw] * w[oc, ci, kd, kh, kw]
id = od*sD + kd*dD - pD   （ih/iw 同理）
```

## 1.6 算法调研与优化分析（历史补记，2026-11-19）

> 本设计成稿早于 op-dev 的 §1.6 模板；此处按现有 evidence 与官方 report 回填，不改变已交付实现。

### 1.6.0 算法调研（四问）

| 问题 | 结论 | 依据 |
|---|---|---|
| 等价化简公式 | 无可化简公式：NCDHW 卷积 `y[n,co,d,h,w] = bias[co] + Σ_{ci,kd,kh,kw} x·W` 无减少乘加的代数恒等式；优化只发生在实现结构与数据布局 | §1 语义、§2 实现结构 |
| 在线算法 | 不适用（非递推算子） | — |
| 复杂度 | `O(N·Cout·D·H·W·C·KD·KH·KW)`；访存以 x/W 的 GM→L1 与 L0C→GM 为主 | §4 性能模型 |
| 硬件亲和 | dense NCDHW implicit-GEMM 走 Cube；bias 在 MixCV epilogue 融合；W 轴 polyphase 展开处理 stride/dilation/pad | §2、§3 |

候选表（基线 → 采纳/否决）：

| 候选 | 结构 | 结论 |
|---|---|---|
| 基线 | direct 7 层循环（逐输出点累加） | 向量化差、L1 复用低，弃选 |
| 采纳 | dense NCDHW implicit-GEMM + MixCV bias kernel（W 轴 polyphase） | 交付版：官方 ratio 1.68–20.43%，17/17 passed |
| 备选 | 裸 Conv3d（无 bias） | **不在 19 内**，不交付 |

复杂度四口径：

| 口径 | 值 |
|---|---|
| FLOPs | `2·N·C_out·C_in_g·taps`（含 bias add，见 §4） |
| 访存 Bytes | `(N·C·D·H·W + C_out·C_in·KD·KH·KW + N·C_out·D·H·W)·elem_bytes`（x/W/y 各一次） |
| 扫描遍数 | **1**（implicit-GEMM 单遍，无多 pass 重扫） |
| 中间缓冲峰值 | L1 tile（BK/BH/BW 分块）+ L0C 累加，峰值 < 512 KB L1 / 128 KB L0C（§2） |

### 1.6 数学等价优化

| 原式 / 原形态 | 优化后 | 等价论证 | 收益 |
|---|---|---|---|
| direct 7 层循环 | implicit-GEMM + Cube | 卷积求和顺序重排，数学定义不变；L0C 累加与 golden 对齐 | 交付态 ratio 1.68–20.43% |
| 独立 bias 后处理 | bias 折进 MixCV epilogue | 线性算子可交换：`Σ(x·W) + b`，只改融合位置 | 省一趟 GM 读写 |
| 逐点 stride 索引 | W 轴 polyphase | 对固定 stride/dilation/pad 的采样点重排，采样集合不变 | 索引向量化 |

### 1.6.1 等价性验证

- 采纳项均为**实现结构 / 数据布局 / 融合位置**变更，数学公式未变，无减少算术的等价变形 →
  按标准「无优化空间」豁免 `verify_equiv.py`（历史算子补记）。
- 机器可复核证据：官方 report 正确性 **17/17 passed**，max_abs 3.125e-2（fp16 r3d-stem）/ 5.0e-1（bf16 unet-encoder），
  与 golden（PyTorch conv3d）逐用例对齐（`reports/tileops/20261023_184106_697102_Conv3dBiasFwdOp`）。

### 1.6.2 向量化替代分析

- W 轴 polyphase 与 bias epilogue 均为向量/Cube 指令；无逐元素标量循环（标量只在边界与参数计算）。

### 1.6.3 弃选行（佐证）

- direct 7 层循环弃选：向量化差、L1 复用低（`MIGRATION_INDEX.md` 表 A Conv3dBias 行）；
- 裸 `Conv3dFwdOp` 不在 19 个任务书算子内（`TileOPs迁移任务书` §5）。

## 2. 实现结构

- 单 kernel，dense NCDHW implicit GEMM（Cube）+ MixCV bias 中继。
- 把 `(kd, kh, kw, ci)` 展平为 GEMM 的 K 轴，输出空间 `(od, oh, ow)` 展平为 N/M 轴。
- 主要杠杆：
  - **tap 打包**：按 tap 切片组织 `T.copy`，承载 3-D 跨步/重叠 im2col；
  - **权重 L1 常驻**：按 `co` 块分核，在 task 循环前一次性载入 L1，消除循环内逐 task 权重重载（48384 次）；
  - **编译期形状级回退**：R3(NDHWC) / R2(padded-N) / R1 / Stage-3 四个兄弟分支，按 shape 选择，保证不劣于 baseline。
- dtype：fp16 / bf16；bias 走 Cube→Vector 交接。

## 3. 支持的参数

- `stride`、`padding`、`dilation` 支持 int 或 3-tuple；`padding` 支持 `'valid'` / `'same'`（`'same'` 要求 stride=1）。
- `groups` 支持分组卷积；非整除 shape 由回退分支覆盖。
- 输出 shape 按标准卷积公式推导。

## 4. 性能模型

- FLOPs：`2 * N * C_out * C_in_g * taps`，`taps` 为输出体上在界内的 tap 数（含 bias add）。
- Bytes：每个张量搬运一次；input 只在实际被窗口读到的位置计。
- 判据：Task Duration 下降 ≥3%（SOL 仅参考）。

## 5. 来源

- GPU 基线：`Conv3dFwdOp` / `Conv3dKernel`（TileOPs，commit `1238a9f4`）。
- 迁移方式：dense NCDHW + MixCV bias；NDHWC 变体作为同结果回退分支。
