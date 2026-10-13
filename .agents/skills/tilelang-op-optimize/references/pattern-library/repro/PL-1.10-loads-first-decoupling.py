"""PL-1.10-loads-first-decoupling -- delta repro (ED-B, delta form).

Entry: pattern-library/elementwise.md PL-1.10-loads-first-decoupling
(kind: pattern; effect only manifests at full-kernel scale -- this file
is the minimal before/after code-change skeleton, py_compile-verified;
see the entry body for the msprof numbers).

One-line: a single reused staging buffer serializes MTE2 and VEC
(load -> vcast -> load -> vcast -> load ...); dedicated per-input
staging with ALL loads hoisted to the top of the persistent serial body
lets the auto-multi-buffer pass overlap the input stream with the
vector chain (ada_layer_norm prefill fp16 58.3 -> 39.8 us, -31.7%).

Optimization insight (why slow -> fast):
  With a shared `stage` reused x -> scale -> shift, the dataflow forces
  MTE2(x) -> VEC(vcast x) -> MTE2(scale) -> VEC(vcast scale) ->
  MTE2(shift) -> VEC ... : every GM load waits for the previous vector
  op to release the buffer, so the MTE2 and Vector pipes alternate
  instead of overlapping (measured: per-core pipe busy times sum to ~
  the wall time). Giving each input its own staging buffer removes the
  fake dependency; placing all three T.copy at the top of the serial
  body gives the scheduler a pure MTE2 phase to run concurrently with
  the previous iteration's vector phase (the auto-multi-buffer pass
  materializes the 2nd buffer instance that makes this legal -- TRAP-
  UB-multibuffer-inflation: budget the ~20 B/elem transit / ~26 B/elem
  fp32 platform footprint, NOT resident-set x factor). Prerequisite:
  the output staging may reuse the x-in staging buffer (disjoint live
  ranges: x-in dies at the first vcast, out staging starts at the last
  vcast). Same family evidence: PL-1.6 (lerp 3-loads-up-front fully
  hides its vector chain in the MTE2 window at >=16M elements).

First verified: 2026-09-10, ada_layer_norm Stage 4 (tilelang
0.1.2+a83118285a, Ascend910B2C, CANN 8.5.0).
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z (task-id corrected by evolver 2026-09-10: stage_state.json is authoritative)
Re-verification log: (append-only)
"""

import tilelang.language as T


# ---- SLOW (before): one shared staging buffer, mid-chain loads: each
# GM load waits for the previous v-op to release `stage` ----
@T.prim_func
def slow_main(
    x: T.Tensor([1024, 1152], "float16"),
    scale: T.Tensor([1024, 1152], "float16"),
    shift: T.Tensor([1024, 1152], "float16"),
    _dummy: T.Tensor([1], "float16"),
    y: T.Tensor([1024, 1152], "float16"),
):
    with T.Kernel(48, is_npu=True) as (cid, _):
        stage = T.alloc_shared((8, 1152), "float16")  # shared: x->scale->shift->out
        x_f32 = T.alloc_shared((8, 1152), "float32")
        sq_f32 = T.alloc_shared((8, 1152), "float32")
        for i in T.serial(3):
            block_id = i * 48 + cid
            if block_id < 128:
                off_m = block_id * 8
                T.copy(x[off_m : off_m + 8, 0:1152], stage[0:8, 0:1152])
                T.vcast(stage, x_f32, round_mode="rint")
                # ... reduce / center / reduce / rstd chain on x_f32 ...
                T.copy(
                    scale[off_m : off_m + 8, 0:1152], stage[0:8, 0:1152]
                )  # mid-chain
                T.vcast(stage, sq_f32, round_mode="rint")
                # ... vmul ...
                T.copy(
                    shift[off_m : off_m + 8, 0:1152], stage[0:8, 0:1152]
                )  # mid-chain
                T.vcast(stage, sq_f32, round_mode="rint")
                # ... vadd, downcast, store ...


# ---- FAST (after): per-input staging + ALL loads hoisted to the top:
# three T.copy as one MTE2 phase, uninterrupted vector chain, x-in
# staging reused as output staging (disjoint live ranges) ----
@T.prim_func
def fast_main(
    x: T.Tensor([1024, 1152], "float16"),
    scale: T.Tensor([1024, 1152], "float16"),
    shift: T.Tensor([1024, 1152], "float16"),
    _dummy: T.Tensor([1], "float16"),
    y: T.Tensor([1024, 1152], "float16"),
):
    with T.Kernel(48, is_npu=True) as (cid, _):
        stage_x = T.alloc_shared((8, 1152), "float16")  # x-in / y-out dual role
        stage_s = T.alloc_shared((8, 1152), "float16")  # scale
        stage_h = T.alloc_shared((8, 1152), "float16")  # shift
        x_f32 = T.alloc_shared((8, 1152), "float32")
        sq_f32 = T.alloc_shared((8, 1152), "float32")
        for i in T.serial(3):
            block_id = i * 48 + cid
            if block_id < 128:
                off_m = block_id * 8
                # 1. all three loads up-front (one MTE2 phase)
                T.copy(x[off_m : off_m + 8, 0:1152], stage_x[0:8, 0:1152])
                T.copy(scale[off_m : off_m + 8, 0:1152], stage_s[0:8, 0:1152])
                T.copy(shift[off_m : off_m + 8, 0:1152], stage_h[0:8, 0:1152])
                # 2. uninterrupted vector chain
                T.vcast(stage_x, x_f32, round_mode="rint")
                # ... reduce / center / reduce / rstd ...
                T.vcast(stage_s, sq_f32, round_mode="rint")
                # ... vmul ...
                T.vcast(stage_h, sq_f32, round_mode="rint")
                # ... vadd ...
                T.vcast(x_f32, stage_x, round_mode="rint")  # out staging reuse
                # 3. single store
                T.copy(stage_x[0:8, 0:1152], y[off_m : off_m + 8, 0:1152])
