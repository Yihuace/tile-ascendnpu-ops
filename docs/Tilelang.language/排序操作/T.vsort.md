# Tilelang.language.vsort

## 1. OP概述

简介：`tilelang.language.vsort` 该算子对输入张量 `src` 沿指定轴 `sort_axis` 进行**排序（sort）**操作，同时输出排序后的值张量 `dst_value` 与对应的原始下标张量 `dst_index`。

```python
T.vsort(src, dst_value, dst_index, descending=False, sort_axis=-1)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor，待排序数据 |
| `dst_value` | `tensor` | 输出tensor，排序后的值，dtype 与 `src` 相同 |
| `dst_index` | `tensor` | 输出tensor，排序后值对应的原始下标，dtype 为 int32 |
| `descending` | `bool` | (可选) 排序顺序，`False`=升序（默认），`True`=降序 |
| `sort_axis` | `int` | (可选) 排序轴，`-1`=尾轴（默认）。目前硬件仅支持尾轴排序 |

### 2.2 支持规格

#### 2.2.1 DataType支持

| | uint8 | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | ------ | ------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | × | × | × | × | × | √ | √ | × | × |

> `dst_index` 固定使用 int32。

#### 2.2.2 Shape支持

结论：输入 `src` 与输出 `dst_value`、`dst_index` 的 shape 必须完全一致（同 rank、同各维大小）。

### 2.3 特殊限制说明

1. **仅支持尾轴排序**：`sort_axis` 只能取 `-1` 或最后一维索引。非尾轴排序将在编译期报错。
2. **元素类型限制**：`src` 和 `dst_value` 仅支持 fp16/fp32；不支持 bf16（如需 bf16 请先 vcast 到 fp32）。
3. **下标类型**：`dst_index` 必须为 int32。
4. **重复元素**：当存在重复值时，排序后下标可能与 PyTorch `torch.sort` 的结果不完全一致（排序稳定性差异）。验证时按「值匹配 + 下标指向的值匹配」双重校验。

### 2.4 使用方法

示例：实现了对尾轴的升序排序（Developer 模式）

```python
@T.prim_func
def vsort_kernel(
    src: T.Tensor((M, N), dtype),
    dst_value: T.Tensor((M, N), dtype),
    dst_index: T.Tensor((M, N), "int32"),
):
    with T.Kernel(1, is_npu=True) as (cid, _):
        # Allocate UB memory (Developer 模式用 alloc_shared，Expert 模式用 alloc_ub)
        src_ub = T.alloc_shared((M, N), dtype)
        val_ub = T.alloc_shared((M, N), dtype)
        idx_ub = T.alloc_shared((M, N), "int32")

        # Copy data from GM to UB
        T.copy(src, src_ub)

        # Perform sort
        T.vsort(src_ub, val_ub, idx_ub, descending=False, sort_axis=-1)

        # Copy results back to GM
        T.copy(val_ub, dst_value)
        T.copy(idx_ub, dst_index)
```

Expert 模式示例：

```python
@T.prim_func
def vsort_kernel_exp(
    src: T.Tensor((M, N), dtype),
    dst_value: T.Tensor((M, N), dtype),
    dst_index: T.Tensor((M, N), "int32"),
):
    with T.Kernel(1, is_npu=True) as (cid, _):
        src_ub = T.alloc_ub((M, N), dtype)
        val_ub = T.alloc_ub((M, N), dtype)
        idx_ub = T.alloc_ub((M, N), "int32")

        T.copy(src, src_ub)
        T.vsort(src_ub, val_ub, idx_ub, descending=False, sort_axis=-1)
        T.copy(val_ub, dst_value)
        T.copy(idx_ub, dst_index)
```

### 2.5 Golden 参考

```python
import torch

def golden_vsort(src, descending=False):
    values, indices = torch.sort(src, dim=-1, descending=descending)
    return values, indices.to(torch.int32)
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::npuir_sort** 将被转换为 `hivm::VSortOp`（即 MLIR 中的 `hivm.hir.vsort`）。

生成的 MLIR 形态：

```mlir
hivm.hir.vsort ins(%src : memref<...>) outs(%dst_value, %dst_index : memref<...>, memref<...>) descending = false sort_axis = -1
```

底层 VSortOp 定义位于 `3rdparty/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td`。

## 4. 兼容性

- `T.npuir_sort` 为兼容别名，与 `T.vsort` 完全等价。
- 遵循 v 前缀 API 规范，新代码推荐使用 `T.vsort`。
