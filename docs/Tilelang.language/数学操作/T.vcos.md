# tilelang.language.vcos

## 1. 概述

简介： `tilelang.language.vcos`用于计算张量的逐元素cos值

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 源张量  |
| `dst` | `tensor` | 目的张量 |

约束：`src`和`dst`应具有相同的形状和数据类型

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×   |

#### 2.2.2 Shape 支持

仅支持 1-5D tensor

### 2.3 特殊限制说明

`src` 的输入范围建议为 `[-1.0, 1.0]`（高精度），最大不超过 `[-2.0, 2.0]`。实测（fp32）：[-1,1] max_err ≈ 2.4e-5；[-2,2] max_err ≈ 6.0e-3；超出后误差急剧增大（[-4,4] max_err ≈ 1.36，[-10,10] 完全错误）

### 2.4 使用方法

以下示例实现了计算输入张量`input`中每个元素的cos值并输出到张量`output`中：

```python
@tilelang.jit(target='npuir')
def vec_cos(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            ub_input = T.alloc_ub((block_M, block_N), dtype)
            ub_output = T.alloc_ub((block_M, block_N), dtype)

            T.copy(Input[bx * block_M, by * block_N], ub_input)
            T.vcos(ub_input, ub_output)
            T.copy(ub_output, Output[bx * block_M, by * block_N])

    return main
```

### 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vcosOp**将被转换为hivm::VMulOp和hivm::VAddOp
