# fused_add_rmsnorm

[English](README-EN.md)

基于 TileLang NPU 实现的融合缩放加法与 RMSNorm 算子。

```text
x = x1 * scale + x2
rstd = rsqrt(mean(x * x, dim=gamma_suffix_axes, keepdim=True) + eps)
y = x * rstd * gamma
```

输入支持任意维度。`gamma.shape` 定义参与归一化的末尾维度：

```text
M = input.shape 中除 gamma 后缀外各维度的乘积
N = gamma.shape 各维度的乘积
```

## 文件结构

```text
__init__.py               公开 Python API
fused_add_rmsnorm.py      算子分发与可运行示例
selector.py               kernel 与分块的自动选择
shape_utils.py            多维形状校验与二维展开
kernels/
  single_n.py            单行整载路径
  merge_n.py             小 N 整载路径及 FP32 双行回退
  multi_n.py             FP16 对齐整载路径
  split_d.py             大 N split-D 流式路径
  split_d_row_group.py   FP16/BF16 非对齐宽行分组路径
  normal.py              通用整载回退路径
  row_one.py             normal/multi-N 共用的单行调度
```

## API

```python
from examples.norm.fused_add_rms_norm import fused_add_rmsnorm, fused_add_rmsnorm_full

y, x_out = fused_add_rmsnorm(x1, x2, gamma, eps=1e-6, scale=1.0)
y, x_out, rstd = fused_add_rmsnorm_full(x1, x2, gamma, eps=1e-6, scale=1.0)
```

`fused_add_rmsnorm` 返回公开输出 `y` 和 `x_out`；
`fused_add_rmsnorm_full` 额外返回 `rstd`，用于正确性验证和调试。

输入必须位于 NPU，且 `x1`、`x2`、`gamma` 使用相同的 `float16`、
`bfloat16` 或 `float32` 数据类型。`x1` 与 `x2` 的形状必须完全相同，
`gamma.shape` 必须与输入形状的某个后缀一致。

## 验证结果

公开自动选择路径通过了 600 个正确性用例，其中 FP16、BF16、FP32 各
200 个。完整性能测试的 460 个用例均通过正确性检查和
`msprof op` profiling。逐用例计算 `torch_npu / TileLang` 设备时间比后，
算术平均为 **1.180x**；对应的 `TileLang / torch_npu` 算术平均为 0.868x。

## 性能样例

下表为 Ascend 910B 上的代表性输入。设备时间取自 `msprof op` 的
Task Duration，管道占用率为所有活跃 AIV 行的算术平均。

| M | N | 数据类型 | TileLang（μs） | torch_npu（μs） | torch/TL | VEC | MTE2 | MTE3 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4096 | BF16 | 4.800 | 6.200 | 1.292x | 33.7% | 29.6% | 19.6% |
| 256 | 4096 | BF16 | 11.680 | 14.741 | 1.262x | 50.8% | 34.9% | 37.3% |
| 512 | 4096 | FP16 | 12.961 | 13.941 | 1.076x | 49.4% | 34.0% | 30.2% |
| 2048 | 8192 | BF16 | 67.483 | 100.264 | 1.486x | 86.5% | 35.0% | 35.3% |
| 12288 | 1024 | FP16 | 41.042 | 51.762 | 1.261x | 83.7% | 80.9% | 66.6% |
| 12288 | 1024 | BF16 | 57.982 | 67.763 | 1.169x | 90.7% | 44.4% | 44.1% |
| 12288 | 4096 | BF16 | 309.912 | 373.135 | 1.204x | 69.2% | 49.9% | 37.8% |
| 8192 | 16384 | FP16 | 825.913 | 976.639 | 1.183x | 42.3% | 63.4% | 33.9% |
| 8192 | 32768 | FP16 | 1773.331 | 2575.223 | 1.452x | 38.7% | 68.9% | 34.7% |
| 12288 | 32768 | FP32 | 5597.724 | 7039.521 | 1.257x | 19.4% | 80.0% | 31.2% |

## 运行示例

在当前目录运行默认 FP16 示例：

```bash
python fused_add_rmsnorm.py
```

可通过命令行指定形状、数据类型、epsilon 和残差缩放系数：

```bash
python fused_add_rmsnorm.py \
  --m 128 --n 4096 --dtype bfloat16 --eps 1e-6 --scale 0.5
```

示例会导入 `torch_npu`，将 TileLang 的三个输出与 PyTorch FP32 参考实现
进行比较，并打印自动选择的实现和分块。运行示例需要可用的 TileLang
Ascend 环境和 NPU。
