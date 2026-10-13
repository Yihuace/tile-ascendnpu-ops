# Tilelang.language.transpose

## 1. OP概述

简介：`tilelang.language.transpose`根据给定的维度排列对输入Tensor的维度进行转置。

```python
T.transpose(src, dst, permutation, size=[])
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src` | `tensor` | 输入Tensor |
| `dst`  | `tensor` | 输出Tensor |
| `permutation`  | `list` | 维度排列序列 |
| `size`  | `list` | 可选参数，手动指定shape。注意：当前实现传入 `size` 即触发 `IndexError: list assignment index out of range`，暂不可用，请保持默认空列表 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | √ | × | √ | × | √ | √ | √ | √ | √ | √ | × |

#### 2.2.2 Shape支持

输入`src`和输出`dst`的秩相同

### 2.3 特殊限制说明

- `T.transpose` 底层调用 `hivm.hir.vtranspose`，而 `hivm.hir.vtranspose` 当前只支持转置两个轴，因此 `T.transpose` 也只支持转置两个轴，即 `permutation` 只能是交换两个轴、其余轴保持原位的排列。
- 如果需要转置两个轴以上，可以分解为几次相邻轴交换的链。例如将 `(A, B, C)` 转置为 `(C, A, B)`（等价于 `permutation=[2, 0, 1]`），可分解为两次相邻轴交换：

```
(A, B, C) --[0, 2, 1]--> (A, C, B) --[1, 0, 2]--> (C, A, B)
```

参考实现如下：

```python
@tilelang.jit(target="npuir")
def transpose3d_kernel(A, B, C, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        src: T.Tensor((A, B, C), dtype),
        dst: T.Tensor((C, A, B), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((A, B, C), dtype)
            tmp_ub = T.alloc_shared((A, C, B), dtype)
            dst_ub = T.alloc_shared((C, A, B), dtype)

            T.copy(src, src_ub)
            T.transpose(src_ub, tmp_ub, permutation=[0, 2, 1])
            T.transpose(tmp_ub, dst_ub, permutation=[1, 0, 2])
            T.copy(dst_ub, dst)

    return main
```

### 2.4 使用方法

以下示例实现了一个transpose计算

```python
@tilelang.jit(target="npuir")
def transpose_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((N, M), dtype)

            T.copy(src, src_ub)
            T.transpose(src_ub, dst_ub, permutation=[1, 0])
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::transposeOp**将被编译为hivm::VTransposeOp
