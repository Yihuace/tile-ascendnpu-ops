# ClampMinFwdOp 设计说明

## 1. 语义

`torch.clamp_min(input, min)` 是 `torch.clamp(input, min, max=None)` 的单下界特化：

```text
y = max(x, min)
```

`min` 是构造期参数（Number），不是张量；输入展平为 1-D 后交给 kernel，输出恢复原 shape。

## 2. 实现结构

- 不新增 TileLang kernel；直接复用 `ClampScalarFwdKernel`（`max=None`）。
- 数据流：flatten → `ClampScalarFwdKernel` → reshape。
- dtype：float16 / bfloat16 / float32；比较在 fp32 进行，NaN 按 torch 语义传播。

## 3. 性能模型

- FLOPs：比较并选择，按 clamp/min-max 家族约定记 `1 * N`。
- Bytes：读 + 写，`2 * N * elem_bytes`。
- 判据：访存 ≥70%；R2 后 7/7 达标（72.50–78.39%）。

## 4. 依赖

- 依赖 `ClampScalarFwdKernel`，随 `ClampScalarFwdOp` 的 PR（#3）进入集成分支；本算子不复制 kernel 代码。
