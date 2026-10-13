# Tilelang.language.vcmp

## 1. OP概述

简介：`tilelang.language.vcmp` 对两个输入tensor进行元素级比较，返回比较结果

```python
T.vcmp(A, B, C, cmp_mod)
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型     | 说明                                                         |
| --------- | -------- | ------------------------------------------------------------ |
| `A`       | `tensor` | 第一个输入tensor                                             |
| `B`       | `tensor` | 第二个输入tensor                                             |
| `C`       | `tensor` | 输出tensor，存储比较结果（bool类型）                         |
| `cmp_mod` | `str`    | 比较模式，可选值："eq"(等于), "ne"(不等于), "lt"(小于), "gt"(大于), "ge"(大于等于), "le"(小于等于) |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | ×     | ×    | ×      | √     | ×      | √     | ×      | √     | √    | √    | √    | ×         |

#### 2.2.2 Shape支持

结论：输入tensor A和B必须具有相同的shape；输出tensor C的shape与输入tensor的shape相同

### 2.3 特殊限制说明

- cmp_mod必须是以下之一："eq", "ne", "lt", "gt", "ge", "le"
- 输入tensor A和B的shape必须相同
- 输出tensor C的dtype为bool类型
- 输出为 bool 类型时不可直接写回 GM（i1 无法 load/store GM，编译失败），需配合 `T.vselect` 等在片上使用

### 2.4 使用方法

以下示例实现了对两个形状为(M,N)的tensor进行大于等于比较，并通过 vselect 取较大值（bool 比较结果保留在片上，不写回 GM）

```python
@tilelang.jit(target="npuir")
def vcmp_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_ub = T.alloc_shared((M, N), dtype)
            B_ub = T.alloc_shared((M, N), dtype)
            cond_ub = T.alloc_shared((M, N), "bool")
            C_ub = T.alloc_shared((M, N), dtype)

            T.copy(A, A_ub)
            T.copy(B, B_ub)
            # vcmp 的 bool 结果保留在片上，配合 vselect 使用（bool 不可写回 GM）
            T.vcmp(A_ub, B_ub, cond_ub, "ge")
            T.vselect(cond_ub, A_ub, B_ub, C_ub)
            T.copy(C_ub, C)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vcmpOp**将被转换为`mlir::hivm::VCmpOp`
