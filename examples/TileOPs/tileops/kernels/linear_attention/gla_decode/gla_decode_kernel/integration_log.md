# GLADecodeOp 集成记录

## 迁移方式

- direct implementation：由 `examples/gla_decode/probe` 直接实现，未走 extract/integrate harness。
- GPU 基线：`GLARecurrentFwdOp`（TileOPs，commit `1238a9f4`）；交付名为 `GLADecodeOp`。
- 源文件：
  - kernel：`tileops/kernels/linear_attention/gla_decode/`
  - op：`tileops/ops/linear_attention/gla_decode.py`
  - workload：`tileops/workloads/linear_attention.py`（仅 GLA 部分）
  - manifest：`tileops/manifest/linear_attention.yaml`（仅 `GLADecodeOp`）
  - test / bench：`tests/ops/test_gla_decode.py`、`benchmarks/ops/bench_gla_decode.py`

## 验证命令

```bash
pytest tests/ops/test_gla_decode.py -q
TILELANG_ASCEND_MODE=Developer python -m tileops.reporting.cli run --op GLADecodeOp --prof-mode msprof
```

## 端到端结果

- run：`20261023_180626_074474_GLADecodeOp`
- 正确性：11/11
- 性能：5 个 tune shape，ratio 3.17–19.94%（bf16）
- 报告状态：passed

## 说明

- 本次只提交 `GLADecodeOp`，未带入 GDN prefill 代码。
- `scale` 非正时按 `DK ** -0.5` 处理；`state` 以 `new_state` 返回。
