# ClampMinFwdOp 复盘

## 结论

`clamp_min` 是 `clamp` 的单下界特化，直接复用 `ClampScalarFwdKernel` 是最小实现：没有新增 TileLang kernel，也没有复制 tiling/流水逻辑。

## 有效点

- 复用同族 kernel，行为与 `ClampScalarFwdOp` 一致，减少维护面。
- 继承 R2 的满块静态 extent + 尾块 epilogue + 流水结构，7 个 workload 全部达标（72.50–78.39%），延迟 −11.3%~−34.1%。
- 接口层只做 flatten/reshape 与参数校验，没有额外数据搬运。

## 限制

- 依赖 ClampScalar kernel 先合入集成分支（#3 已合入）。
- `block_size` / `num_stages` 已扫，无剩余杠杆。
