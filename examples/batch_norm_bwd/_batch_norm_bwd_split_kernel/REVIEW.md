# _batch_norm_bwd_split_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_split_kernel_DESIGN.md（revision_index 1，revision；上一版 `history_version/design_v0.md`）
- 任务类型: 迁移（harness 模式）
- 源算子: Stage 0 提取件 `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（`_batch_norm_bwd_split_kernel` L173–356：stats L191–252 / finalize L254–290 / apply L292–354）；GPU 原始 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_batch_norm_bwd_split_kernel` L1795 + host 类 `BatchNormBwdSplitKernel` L2044）；NPU wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 本轮性质: v0 结论「不通过」（3 阻塞 + 3 建议）后的**再检视（re-review）**
- 阻塞级问题数: 0
- 建议级问题数: 3（均为遗留/集成级，见各维度 warn 说明，不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（与 §1.6.1 内嵌表逐格一致，全 EQUIV_PASS）**
  - 命令: `python3 examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/verify_equiv.py`（torch CPU，fp64 参照 + 角点值），退出码 0。
  - 结果（采纳项 O1+O2 联合，21 案例 = 3 dtype × 7 shape：(32,1)/(8,1024)/(4,1024)/(4,784)/(4,2048)/(1,1)/(2,3)）：`float16 max ulp=0.00 / viol=0.0000`、`bfloat16 max ulp=0.00 / viol=0.0000`、`float32 max ulp=0.00 / viol=0.0000`，三 dtype 全 `EQUIV_PASS`——与 DESIGN §1.6.1 内嵌结果表（L350–353）逐格一致。
  - 判别段 A（raw-moment `gw` 拒绝证据）：`float32, N=4, S=2048` 违反率 = **0.1553**（`DISCRIM_OK`），fp16/bf16 = 0.0000（宽容差掩蔽）——与 §1.6.1 L353 的 0.1553 一致，非纸面论证。
- **design_calc_check 重跑**（命令：`python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_split_kernel_DESIGN.md`）：

```json
{
  "design": "examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_split_kernel_DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {"check": "ub_budget", "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check": "l0c_budget", "status": "skip", "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {}; declared logical cores = None"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```

  - **skip 项人工复核结论**：
    - `ub_budget`（机械跳过，人工已核）——§4.5 v1 二维口径 `bN·bS·BPE + bN·8 + 64 ≤ UB_SAFE(163840)`。独立复算：large-spatial fp16 `bS=4544, bN=2 → 2·4544·18+16=163,600` ✓；大 N `(8192,128,8)` fp16 `bN=1077,bS=8 → 1077·8·18+1077·8=163,704` ✓；fp32 `(4,256,65536) bS=4092,bN=2 → 163,696` ✓；N=1 `bS=9096 → 9096·18+8+64=163,800` ✓。**v0 的大 N 溢出缺口已由 N-tiling 封堵**（见维度 2/3；机制跳过的正是 v0 遗漏点，本轮改以人工公式复算拦截）。
    - `l0c_budget`（机械跳过）——纯 Vector，无 Cube/L0，不适用（§3.5.1/§4.4/§5.3）。
    - `core_split`（机械跳过，人工已核）——§5.5：stats 逻辑核 `C·splits=512`、finalize/apply 逻辑核 `C=128`，`num_kernels=min(logical,40)=40`，`num_local_tasks` = `ceil(512/40)=13` / `ceil(128/40)=4`；物理核数 `NPUUtils.get().get_aicore_num()` 实查 = 20（纯 Vector ×2 = 40）。算术自洽 ✓。
    - `r3_metrics`——pass，与 §1.6.0 R3 四口径表一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子全文——Stage 0 提取件 `.../norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（stats L191–252、finalize L254–290、apply L292–354）与 GPU 原始 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`、NPU wrapper `.../batch_norm_bwd/batch_norm_bwd.py`、op、探针 `design_probe_split_phase.py` + `probe/PROBE.md`，逐项与 DESIGN §0 核对。

- **语义理解正确性 — pass（源码证据）**：§0.1 两标量归约与 `grad_x` 公式与源码逐句一致——`grad_bias=Σ grad_out`（stats `sums += g` L232 + finalize `grad_bias[bc]=total[0]` L280）、`grad_weight=Σ grad_out·x̂`（`sums_xhat += g*x_hat` L233 + L281）、`x_hat=(x−mean)·rstd`（L230–231）、`grad_x = w·rstd/L·(L·g − gb − x̂·gw)`（finalize 系数 L284–288 + apply 输出 L326–331）。元素映射 `l → [l//S, c, l%S]`（源码 L226/L228 `head//S, bc, head%S`）与 §0.1「沿展平轴 `N·S`、通道轴为中间轴（逐通道平面行跨步 `C·S`、行内 stride=1）」一致。**累加顺序** §0.1 与源码 L212–213 注释（"the channel gradients must not depend on a merge order"）、finalize `for k in T.serial(splits): total[0]+=partial_sum[bc,k]`（L277–279）一致。**dtype 语义**「fp32 opmath + 单次舍回」与 `accum_dtype="float32"`（L187）一致。**边界** `end=T.min(start+chunk,L)`（L209）+ 逐元素 `if l<end`（L237）一致。**空张量**：源/npuir op 无 `numel()==0` 守卫，§0.1/§9.1/§9.3 明确由 NPU op/wrapper 集成期承接，L0-10 覆盖 ✓。
- **算法解读完整性 — pass（源码证据）**：§0.3 步骤表 S-P0–A-W1 覆盖源码全部计算语句（S-P0 L205–209 / S-P1a-b L225–228 / S-P1c L229–233 / S-P1d L234–243 / S-R1 L246–247 / S-W1 L248–250 / F-P0 L277–279 / F-W1 L280–281 / F-E0 L283–288 / A-P0 L317 / A-P1 L319–322 / A-P1b L323–324/L334 / A-E1 L326–331/L335–340 / A-W1 L341–342/L347–352），无遗漏、无臆造；数据流图覆盖全部 buffer；host 侧（`_split_seed`、`forward` 预分配 + 三段调用、`lru_cache`）已列出。
- **优化手段覆盖完整性 — pass（源码证据）**：§0.4 十二项可逐项溯源（`T.Kernel(C*splits, threads=)` L205、`partial_sum[C,splits]` workspace L202–203 + finalize、固定归约树 L212–213、two-accumulator `sums/sums_xhat` L232–233、`T.Parallel(1,threads)` L220、`alloc_fragment` 驻留 L214–215、`apply` 平铺 `base` L317、epilogue 预折叠 L283–288、fp32 opmath、`vector_holds_one_item` 快路径 L194/L224、`T.vectorized` L225、`lru_cache`），未把实现层优化误判为算法层。
- **耦合性判定合理性 — pass**：§0.5 每条四态处置有依据，与 GPU→NPU 映射不矛盾——register 驻留→`T.alloc_ub`、线程树→`T.reduce`、`threads`/`T.Parallel`/`T.get_thread_binding`/flat `T.Tensor` 别名→舍弃、`C*splits` 超发→persistent 分核、谓词→`T.min`+双 slice+reduce `size`；舍弃项意图承接在 §0.6 末尾逐项说明，无静默丢弃。
- **NPU 重设计可行性 — pass（本轮聚焦 v0 三阻塞项的定点修复）**：
  - **阻塞项 1（apply 复用 stats 的 `n_tile` 漏尾列）— 已修复**。v1 工厂层拆两常量：`n_tile_S_apply=-(-S//bS)=ceil(S/bS)`（§3.3 L531）与 `n_tile_S_stats=-(-max_seg//bS)`（L532）；apply 内层循环用 `T.serial(n_tile_S_apply)`（L707）、stats 用 `T.serial(n_tile_S_stats)`（L584），**不再共用**。独立复算：apply `tt∈[0,ceil(S/bS))`，`s0=tt·bS<S`，`real=T.min(bS,S-s0)≥1`，逐块写出 `[n0:n0+rN, c, s0:s0+real]` ⇒ 全 S 覆盖；large-spatial `S=262144,bS=4544 → n_tile_S_apply=58`（§4.5 L885）、stats `n_tile_S_stats=15`（L879），二者分离正确。
  - **阻塞项 2（段 extent 0/负）— 已修复**。v1 改余额分配 `dist=S//splits`、`rem=S%splits`、`s0=sp·dist+T.min(sp,rem)`、`s1=(sp+1)·dist+T.min(sp+1,rem)`、`real=s1-s0`（§3.3 L513–515/L573–574、§5.2 L922–923）。因 `splits=min(_split_seed,S)≤S ⇒ dist≥1`，每段 `real∈{dist,dist+1}≥1`，恰 `splits` 段覆盖 `[0,S)`。独立复算 v0 反例 `S=5,splits=4`：v1 段长 `[2,1,1,1]`（`s1(3)=5`），无 0/负 extent；子块 `r2=T.min(bS,s1-ss)` 并有 `if r2>0` 守卫（L587）。
  - **阻塞项 3（大 N UB 溢出）— 已修复**。v1 引入二维 `(bN,bS)` tiling（§3.3 L519–532）：`N>1` 时 `bN=min(N,max(2,(UB_SAFE-SMALL)//(bS·BPE+ROW_OVERHEAD)))`，核内新增 `for nn in T.serial(n_tile_N)`（L579/L704），`red/red2` 改 `(bN,1)`、N-块间以 `T.reduce(red,gb,dims=[0],clear=False,size=[rN,1])` 累加进 `(1,1)`（L613–617）。独立复算 v0 反例 `(8192,128,8)` fp16：`bS=min(4544,8)=8`，`bN=1077`，驻留 `1077·8·18+1077·8=163,704≤163,840` ✓（v0 该 shape `bS=8` 的 `(N,bS)` 四路 fp32 超 UB）。
  - **三处修复的跨章节一致性 — pass**：`n_tile_S_stats`/`n_tile_S_apply` 分离在 §3.2（L491）/§3.3（L530–532）/§5.2（L931–933）/§6.1（L995–1001）一致；余额分配在同一 4 处 + §9.1（L1134）一致；二维 `(bN,bS)` 口径在 §0.6 R2（L188）/§3.3（L518–532）/§4.5（L869–894）/§5.2/§6.1/§9.1 一致；§8.3 新增 L0-11（均衡段 `(13108,128,5)`）、L0-12（大 N `(8192,128,8)`）、L0-13（apply 全 S 显式比对整张 `grad_x`）分别门住三项（L1111–1115）。**未发现 §3.3 正文与 §4.5/§9.3 自相矛盾的残留**（v0 的 apply `n_tile=29 vs 116` 矛盾已消解，正文/伪代码/§4.5/§8.3 均为分离常量）。
- **golden 独立性 — pass**：§8.1 golden 直接调用 `torch.sum(dim=(0,2))`/`torch` 逐元素仿射，**不含** NPU 设计的 persistent 分核、S 轴拆段、部分和 workspace、finalize 合并、两步 `T.reduce`、`T.vbrc`、fp32 中转链、分布式 epilogue 等任何算法特征；以 §0.1 语义为唯一依据（移植源仓 `workloads/norm.py`）✓。`apply` 保持源 `out_idx=[-1]` 单输出（wrapper `grad_x = apply_()(...)`）；`stats/finalize` 无 out_idx，`grad_weight/grad_bias` 由 host 预分配——与源参数序（`finalize` 的 `grad_weight, grad_bias`，源 L258–259）逐位一致，未复现宽核 v0 的多输出返回序缺陷。
- **建议级（warn，遗留/集成契约）**：**工厂 `splits` cap 到 S 与 wrapper `_split_seed` 未 cap 的口径**。§3.3 L510–511 工厂 `splits=min(_split_seed,S)`；而 wrapper `BatchNormBwdSplitKernel.__init__`（`batch_norm_bwd.py` L277）取 `_split_seed(N,C,S)` **未 cap**，`forward`（L306–308）以该值分配 workspace `partial_sum(C,splits)`。当 `_split_seed>S`（需 `ceil(512/C)>S` 即 `C·S≲512`，且 `_holder` 判 split，如 `(32768,128,2)`）时，工厂 `partial_sum` 声明 `(C,2)` 与 wrapper 传入 `(C,4)` 形状错位。该缺口为 v0 亦存在的集成契约项（非 v1 引入），§0.2/§5.4/§9.1 已将其列为「三段共用 splits、集成期核对」，但**未显式要求 wrapper 侧同步 cap 到 S**。建议集成期将 wrapper 的 `splits` 统一改为 `min(_split_seed,S)`（与工厂同源同 cap）。判 warn：主服务负载 large-spatial 及全部 L0 用例均满足 `_split_seed≤S`，不影响本轮正确性主结论。

**维度 0 结论: pass**（语义/算法/优化/耦合性/重设计/golden 独立性均与源码核对通过；v0 三阻塞项经独立算术复算确认**已修复且跨 §3.3/§4.5/§5/§8.3 一致**；仅遗留一处集成期 splits-cap 口径 warn）。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有文档佐证且路径经 Glob 核实：`T.Kernel(一维, is_npu=True)`（`编译器提示操作/T.Kernel.md`）、`T.alloc_ub`（`内存操作/T.alloc_ub.md`，Expert 显式 UB）、`T.copy` 双显式 slice（`内存操作/T.copy.md`，strided 区域 + 中间轴标量索引，探针真机验证）、`T.vcast(rint)`（`数据类型转换操作/T.vcast.md`）、`T.reduce` dims/clear/**size**（`规约操作/T.reduce.md` §2.1 `size` 控制参与范围；§2.3 `clear=False` 前须初始化 dst）、`T.reduce_sum`、`T.vbrc` `(1,1)→(bN,bS)`（`shape操作/T.vbrc.md` §2.2.2）、`T.vmul/vsub/vadd`（fp16/fp32 ✓ bf16 ×）、`T.min` 标量（`比较操作/T.min.md`：标量运算、不支持 shape，与 §3.2/§3.3 的 `T.min(sp,rem)`/`T.min(bS,·)` 标量用法匹配）、`T.clear`、`T.serial`+`if` 守卫 + floordiv/mod 索引（`examples/fused_add_layer_norm/...` persistent 先例 + 本函数探针真机）、`T.atomic_add`（已核对可用但按确定性要求弃用）。v-prefix 优先 ✓；Expert/Vector 形态有先例 `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_direct_kernel.py`。**新用法 API 前提**：`T.reduce(dims=[0], clear=False, size=[rN,1])`（N-块累加，§3.3 L614）与 `T.reduce(dims=[1], clear=False, size=[rN,r2])` 同口径（`clear=False` 累加、`size` 限范围）——文档未禁止该组合；§3.3 注（L742）已声明其为 **Stage 3 真机核对项** 并给出回退（`dims=[0], clear=True` 得 `(1,1)` + `T.vadd(gb,part,gb)`）。API 存在性成立，判 pass（该组合的运行时行为属实现期核对，非 API 缺失）。

### 2. 内存层级规划 — pass

§4.4 完整给出 GM→UB 路径（无 GM→L0 直连），全部计算 UB 内闭环；§4.3 每 buffer 标注层级（stage/g32/xh32/prod/brc/red/red2/gb/gw/…全部 UB）。

- §4.5 v1 采用**二维驻留口径** `bN·bS·BPE + bN·ROW_OVERHEAD + SMALL ≤ UB_SAFE=163840`（BPE=itemsize+16：fp16/bf16=18、fp32=20；ROW_OVERHEAD=8=`red/red2` 各 `(bN,1)` fp32；SMALL=64）。独立复算四类代表 shape 均达标：large-spatial fp16 `163,600`、test `147,472`、大 N `163,704`、fp32 `163,696`、N=1 `163,800`（§4.5 L879–891）。
- **v0 问题 3（大 N 溢出）已封堵**：`N=8192,S=8` fp16 由 `bN=1077` 的 N-tiling 使 `(bN,bS)` 驻留收敛，不再出现「`bS` 被 `max(vw,·)` 抬回后 `(N,bS)` 超 UB」。
- **口径单一来源** ✓：§4.5 表值可由 §3.3/§5.2 公式（同 BPE、同 `bN`、同对齐）互推；`bN≥2`（`N>1`）保证 `dims=[0]` 归约合法。
- **warn（建议级）**：auto-multibuffer 膨胀（`constants.md` CONST-capacity-910B2C `ub_inflation=1.7`；RETROSPECTIVE TRAP-UB-multibuffer-inflation 实测 transit ≈22 B/元素）可能使实际驻留高于设计公式值。设计以 160 KiB（留 32 KiB 余量）+ §9.1 L1131 声明「Stage 4 实测超预算即降 bS（相应增大 bN 覆盖）」承接，属**参数级**风险、非结构缺陷，编译期 UB 溢出为天然 guard。

### 3. Tiling 策略 — pass

§5.2 block 划分（`splits`/`dist`/`rem`/`max_seg`/`bN`/`bS`/`n_tile_N`/`n_tile_S_stats`/`n_tile_S_apply` 全为工厂期常量）、向量轴 S、尾块 `T.min`、分核三要素（① 逻辑核数 stats=`C·splits=512`、finalize/apply=`C=128`；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查=20→纯 Vector 40；③ 极大规模 persistent 核内 serial、边界静态）齐备且算术自洽。

- **v0 问题 1（apply 子块上界）已修复**：`n_tile_S_apply=ceil(S/bS)` 独立定义（§5.2 L933），apply 循环覆盖全 S（§3.3 L707）；与 §4.5 `58`、§8.3 L0-13 一致。
- **v0 问题 2（段 extent）已修复**：余额分配保证 `real≥1`、恰覆盖 `[0,S)`（§5.2 L922–923、§5.4 L956）。
- **v0 问题 3（大 N）已修复**：`(bN,bS)` 二维分块（§5.2 L924–931），`N>1⇒bN≥2` 使 `dims=[0]` 合法。
- 非整除处理：段尾 `r2=T.min(bS,s1-ss)`（`if r2>0` 守卫）、尾 N-块 `rN=T.min(bN,N-n0)`、apply 尾子块 `real=T.min(bS,S-s0)`；I/O 双 slice 截断 + 归约 `size` 同步截断。S 尾轴对齐（fp16/bf16 ×8、fp32 ×4；manifest shapes 0% 浪费，§1.6.3）。判 pass。

### 4. 技术约束检测 — pass

§3.5 逐项核对：无三维 Kernel（三段各一维 `T.Kernel(num_kernels, is_npu=True)`）；GPU 专用 API 已列舍弃（`T.vectorized`/`T.get_thread_binding`/`T.Parallel(1,threads)`/`threads=`/flat `T.Tensor` 别名）；无 GEMM 分形；L0C 不适用（纯 Vector）；物理核数实查 20→40；bf16 无 Vector 算术经 fp32 中转（`T.vmul.md`/`T.vsub.md`/`T.vadd.md` §2.2.1 bf16 × 已核对）；中间轴归约（非尾轴）经 strided 平面 gather（探针验证）；跨 block 归约经 GM workspace + finalize（无 atomics）。v0 建议 5（`S=1` 退化分支 `T.copy(g32,red)` 形状不匹配）**已修复**为 `T.copy(g32[0:rN,0:1], red[0:rN,0:1])`（§3.3 L610–611、§5.4 L954）。判 pass。

### 5. 循环与同步 — pass

§6.1/§6.2 明确循环类型：任务维 `T.Kernel`+`T.serial`、stats 段内 `T.serial(n_tile_N)`×`T.serial(n_tile_S_stats)`（`if r2>0`）、apply 段内 `T.serial(n_tile_N)`×`T.serial(n_tile_S_apply)`、元素维向量化（`T.copy`/`T.vcast`/v-op/`T.reduce` 全 buffer 操作）、尾块标量 `T.min`+双 slice+`size`；§7 同步与 Expert 模式匹配（单 Vector scope 内编译器按 MTE2→V→MTE3 自动 `pipe_barrier`；三段间由运行时同一流顺序下发，无 atomics、无核间 flag）。

- **v0 问题 1 在循环层的修复已确认**：apply 的 S-子块循环上界改为独立的 `n_tile_S_apply`（§6.1 L1001），覆盖全 S；stats 用 `n_tile_S_stats`（L996）。
- **v0 问题 2 在循环层的修复已确认**：空/尾段由余额分配（`real≥1`）+ `if r2>0` 守卫消除，内层不再执行非法切片。
- 循环边界均为 host 编译期常量（`n_tile_N`/`n_tile_S_*`/`nlS`/`nlC`），满足 Ascend 静态边界约束。判 pass。

### 6. 验证方案 — pass

§8.1 golden 独立（维度 0）；§8.2 精度标准取自源仓 `batch_norm_bwd_tolerance`（fp32 1e-5 / fp16 1e-2 / bf16 2e-2，含通道梯度 `sum_atol`）；§8.3 L0 **十三项**覆盖 fp16/bf16/fp32、N=1/S=1 退化、素数 S、空张量，并新增：**L0-11**（`(13108,128,5)` 均衡段 `dist=1/rem=1 → [2,1,1,1]`，门住阻塞项 2）、**L0-12**（`(8192,128,8)` 大 N `bN=1077/n_tile_N=8`，门住阻塞项 3）、**L0-13**（`(4,128,262144)` 显式逐元素比对整张 `grad_x`，门住阻塞项 1；v0 形态 apply 只写前 `ceil(dist/bS)·bS=68,160` 列必失败）。v0 建议 6（L0 未门住缺陷路径）**已修复**。判 pass。

### 7. 完整性与一致性 — pass

- §0–§11 章节齐全（迁移类含 §0，0.1–0.7 子节齐全；§0.0 修订说明 + §11 修订日志完整）；全文无 `{placeholder}`/`TODO`/`待补充`；§3.5.3 引用的 examples 路径经 Glob **全部真实存在**（wide DESIGN、conv1d_direct、fused_add_layer_norm、logsumexp、lerp_tensor、probe、wrapper、workloads、manifest、test、bench）；op 输出契约 `(grad_x, grad_weight, grad_bias)` 在 §0.2/§3.2/§4.2/wrapper 四处一致。
- **v0 内部矛盾已消解**：§3.3/§6 的 apply 子块数原为 `ceil(chunk/bS)`（=29）与 §4.5/§9.3 的 `ceil(S/bS)`（=116）矛盾——v1 统一为 `n_tile_S_apply=ceil(S/bS)`（=58，因 v1 二维 bS 变更）单一来源，§3.2/§3.3/§5.2/§6.1/§9.1/§9.3 一致。
- v0 建议 4（合并序表述）**已修复**：§0.5 行 #3（L162）与 §0.6 R3（L196）统一表述为「`T.reduce(dims=[1])` 硬件定序树合并（确定性、无 atomics）；与源 `T.serial(k in splits)` 串行合并仅分组顺序不同，fp32 舍入差在 §8.2 容差内」。
- **warn（建议级）**：§9.1 L1125/§0.2 L88 述「工厂 splits 与 wrapper 预分配 `[C,splits]` 不一致 → 形状错位；工厂取 wrapper 同源 `_split_seed` 语义」——但工厂实际附带 `min(·,S)` cap 而 wrapper 未 cap（见维度 0 warn），措辞「同源」未覆盖 cap 差异。建议补一句「集成期 wrapper 亦须 cap 到 S」。不影响主结论。

### 8. 算法优化分析 — pass

检视方式：以 §1.6 为对象逐项核对，并交叉核对 §3.1/§3.3/§6 与 §1.6 的双向一致性；数学等价性、调研结论由本维度**独立推演/独立复核**核对（不轻信文档自述），弃选前提亲自打开 API 文档/候选库/常数库**核对**。

- **§1.6 章节存在**：含 1.6.0 调研、1.6.1 数学等价优化、1.6.2 向量化替代、1.6.3 向量化轴与布局决策四子节，各有结论行。
- **调研存在性与完整性 — pass**：§1.6.0 四问齐全——R1 候选表**含基线 #0（源 `_batch_norm_bwd_split_kernel`）**并评估 5 候选；R2 在线算法**有明确结论**（「无新增收益/等价于基线」+ 结构依据）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径写清）；R4 逐候选亲和评估；「无更优替代」写明调研范围（含结构判据，无互联网）。
- **调研结论正确性（独立复核）**：
  - **负向断言复核（对照候选库 + 源码证据）**：亲自读 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`——`ALG-reduction`（L62–68）R2 明确列「分块归约 + 合并（两阶段/树形）；Welford」，`ALG-layernorm`（L17–23，family 含 `batch_norm`）R1 列 `E[x²]−E[x]²` 单遍式（须评估 cancellation）、R2 列 Welford、R4 为 row-reduction 迁移基准形态。设计的负向论断逐条成立：R1#5 online/Welford——本反向的 `mean/rstd` 为前向输入，算子只算两个**可加一阶矩** `Σ grad_out`、`Σ grad_out·x̂`，无需要 rescale 的 running 统计量（对比 online softmax 的 running max/sum），「在线化」退化为 R1#0/#1 的「分段部分和 + 合并」结构（即候选库所述「分块归约+合并」），**非独立算法族**——与候选库定位不矛盾；R1#2 raw-moment——已被**机器证据**否决（复跑违反率 0.1553 > 0）。**无漏掉的已知候选，无与候选库/源码相矛盾的负向断言**。
  - **复杂度算术独立复算**：基线/主选 Bytes = stats 读 g+x（`2·L·C·eb`）+ apply 读 g+x（`2·L·C·eb`）+ 写 grad_x（`1·L·C·eb`）= `5·L·C·eb` ✓；FLOPs ≈9L 与 manifest roofline `flops=9·N·C·S` 一致 ✓；代表 large-spatial `5·4194304·128·2 ≈ 5.37GB` ✓、test `4×64×4096` fp16 `≈10.5MB` ✓；workspace `2·C·splits·4 = 2·128·4·4 = 4KB` ✓；中间缓冲为 UB 常驻（无 GM 往返），未漏计 ✓。
  - **弃选论证前提核对（亲自开文档/候选库核对）**：R1#2 raw-moment 以 D-1 机器证据否决 ✓；否决项「fp16/bf16 原生域计算链」引用 `数学操作/T.vmul.md` §2.2.1（已打开，bf16 列 ×）✓；「跨 dtype `T.copy` 作 cast 融合」引用 `内存操作/T.copy.md` §3（舍入不可控）✓；候选 #4 转置弃选写成「host permute 或核内 `T.transpose` 链」与 `创建操作/T.transpose.md`（UB 级、两轴交换）一致，未误写成 GM 级 transpose ✓；候选 #2 flat-L 弃选依据「非仿射索引，`T.copy` 未列非仿射」与 `内存操作/T.copy.md` 一致 ✓；R4 容量/带宽淘汰引用 `constants.md` CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C / CONST-capacity-910B2C（经核对四条目存在；CONST-aicore 条目 2026-10-12 注记实测 `get_aicore_num()`=20，与 §5.5 一致）✓。
  - **并列候选裁决路径**：主选 #1 vs one-core-per-channel 的裕度由**编译期已知的 C**决定（非未实证常数），无需三件套；唯一未实证量（auto-multibuffer 膨胀系数）为参数级（bS 取值）决策，判定合理（§1.6.3 实验裁决评估）。
- **调研→设计一致性 — pass**：§1.4、§3.1（输入声明为 §1.6.1 优化后公式）、§6 均与 §1.6.0 选定算法族一致；迁移 §0.5/§0.6 处置与调研选族一致，源算法被替换项（register/线程树/threads/谓词/flat map）意图承接在 §0.6 末尾逐项说明，无静默丢弃。
- **数学等价优化完整性 + 等价性独立推演 — pass**：§1.6.1 对选定算法系统审视，采纳 O1/O2/O3，逐项四要素齐全。独立推演：O1 `per=(w·rstd)·fl(1/L)` 对 `coef=w·rstd/L`——L 编译期常量，`fl(1/L)` 相对误差 ≤ 2⁻²⁴；O2 分配律恒等 `coef·(L·g−gb−x̂·gw)=(w·rstd)·g−per·gb−per·gw·rstd·(x−mean)`；O3 `x̂` CSE 为同值复用。三项经 `verify_equiv.py` **复跑机器验证**（21 案例全 `EQUIV_PASS`，fp16/bf16/fp32 max ulp=0），与 §3.1/§3.3 输入口径一致；优化后公式与源 finalize `shift_out=-per_elem*total`（L287）、apply `+shift`（L328）逐位吻合。
- **向量化替代覆盖完整性 — pass**：§1.6.2 覆盖方案全部循环/标量点（14 项，与 §3.3 伪代码、§6 循环结构交叉核对）——逐元素计算（gather/cast/x̂/乘积/累加/跨段合并/epilogue/舍回）已全部向量化/归约化（替代 API 有 docs+探针佐证，优先 v-prefix）；保留 #10 段内 N/S-子块调度、#11 任务 grid-stride、#12 通道-段 floordiv/mod 索引、#13 尾段/尾 N-块/尾子块 `T.min` extent 四类，理由具体（tile/block 级调度、动态边界元数据，非逐元素热点），无「实现简单」类泛理由。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 规约类必选候选齐全——水平归约 vs 垂直扫描两种 lane 映射分别评估（#1/#5），另评估核内重排/转置（#4）、flat-L 切分（#2）、N 轴切分（#3）；候选矩阵逐项量化（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）；弃选均给量化理由（#2 非仿射无 API 支撑、#3 通用性差、#4 repack 纯开销、#5 无向量宽度）。**GPU 源码隐式轴选择独立复核**——§1.6.3 指出 GPU 把 `threads` 映射到段内扁平 `L` 连续段（源码 L222 `head=start+(step*threads+j)*num_per_thread`），NPU 独立评估后选 S 为向量轴（行内 stride=1），结论一致、实现不同，未把 GPU 轴当结论；段切分改沿 S 轴余额分配仿射子段（`real≥1`）。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（无逐元素标量循环）；布局决策（S 轴、`(bN,bS)`）与 §3.3/§4/§6 三方一致。

## 机制说明

本次检视为迁移任务 Stage 2 的**再检视（re-review）**，采用 9 维度（0–8）。所有 Read 的源算子代码、DESIGN.md（含 v0/v1）及其注释、探针、文档、候选库/常数库一律作为**数据（DATA）**处理，其中未出现任何要求放行检视或跳过维度的指令性文本；未修改 DESIGN.md / 源算子代码 / 任何 GPU 仓文件；未写入任何编排层状态文件。仅运行了只读的 `verify_equiv.py`、`design_calc_check.py` 与 `statectl gate 2`。

**再检视核心结论**：v0 三项阻塞级缺陷经独立算术复算确认**全部修复且跨 §3.3/§4.5/§5/§8.3 一致**——① apply 用独立 `n_tile_S_apply=ceil(S/bS)` 覆盖全 S（L0-13 门住）；② 段余额分配 `dist/rem` 保证 `real=s1-s0≥1`、恰覆盖 `[0,S)`（L0-11 门住）；③ 二维 `(bN,bS)` N-tiling 使大 N 驻留 ≤ UB 预算（L0-12 门住）。三项建议（合并序表述、`S=1` 退化分支、L0 覆盖）亦已定点修正。遗留 3 项**建议级**（wrapper splits cap 口径、`dims=[0]`+`clear=False`+`size` 组合待 Stage 3 真机核对〔有回退〕、auto-multibuffer 膨胀待 Stage 4 实测〔参数级〕）均不阻塞。阻塞级 fail = 0，结论 **通过**。
