# Conv3dBiasFwdOp 复盘

## 有效点

1. **权重 L1 常驻**：按 `co` 块分核，task 循环前一次性载入权重，`aic_mte2_time` 6050→3042µs，r3d-stem-bias 6161.50→3090.14µs（1.99×）。
2. **tap 打包**：把 27-tap im2col 组织成按 tap 的切片，减少重复地址计算。
3. **编译期 shape 回退**：R3(NDHWC)/R2(padded-N)/R1/Stage-3 四分支，保证无收益 shape 不劣于 baseline；第二轮 unet 6428→301.6µs（21.3×）。

## 未解决 / 限制

- video-down-bias 87.00→86.42µs（−0.7%，噪声内），回退 baseline。
- 27-tap 输入重复读 + `c_in=3→16` 的 5.3× FLOP/访存浪费是当前工具链下的结构上限。
- `bishengir-compile` 在部分 conv shape 上会崩（NPU IR opt），与具体 shape 相关且每次不同；需要编译器侧修复或回退。

## 经验

- 先把权重搬运从内层循环拿出去，比继续调 tiling 更有效。
- 对多形态 conv，编译期 shape 级回退比强行统一实现更稳。
- 小通道 conv 的 fractal 填充会显著放大无效计算，选型时要计入。
