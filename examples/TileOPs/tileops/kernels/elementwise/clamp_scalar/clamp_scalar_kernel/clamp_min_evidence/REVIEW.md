# ClampMinFwdOp 复核记录

## 复核范围

- Op：`tileops/ops/elementwise/clamp_min.py`
- 复用 kernel：`tileops/kernels/elementwise/clamp_scalar/`（不新增 kernel）
- Manifest / Workload / 测试 / Benchmark：`ClampMinFwdOp`、`ClampMinWorkload`、`tests/ops/test_clamp_min.py`、`benchmarks/ops/bench_clamp_min.py`

## 检查项与结论

| 检查项 | 结论 |
|---|---|
| 语义 | `max(x, min)`，与 `torch.clamp_min` 一致 |
| 参数 | `min` 为构造期 Number；无张量参数 |
| 复用方式 | `ClampScalarFwdKernel(max=None)`；无新 kernel、无重复代码 |
| dtype / NaN | fp16/bf16/fp32；fp32 比较，NaN 传播 |
| 测试 | 14/14 通过；无 skip、无降低容差 |
| 性能 | 7/7 tune shape 达标，ratio 70.08–77.31%；延迟 −11.3%~−34.1% |
| 风险 | 依赖 ClampScalar kernel 先合入；#3 已合入，本分支基于最新 integration |
