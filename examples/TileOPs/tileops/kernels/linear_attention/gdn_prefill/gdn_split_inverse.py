"""K2 split prototype: 6-level Neumann inverse with Cube + Vector kernels.

A = (I - N)^{-1} = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32)
  for 64x64 strictly-lower N (N^64 = 0).

Per level:
  Cube kernel:   tmp = P @ T,  P2 = P @ P
  Vector kernel: T2 = T + tmp
This avoids the fused MixCV / chained-L0C path that triggers CG-2026-0003.
"""

import argparse

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2, 3], target="npuir")
def seg_cube(tasks: int, C: int = 64, dtype: str = "float32"):
    @T.prim_func
    def seg_cube_main(
        T_in: T.Tensor((tasks, C, C), dtype),
        P_in: T.Tensor((tasks, C, C), dtype),
        tmp_out: T.Tensor((tasks, C, C), dtype),
        p2_out: T.Tensor((tasks, C, C), dtype),
    ):
        with T.Kernel(tasks, is_npu=True) as (t, _), T.Scope("Cube"):
            p_l1 = T.alloc_L1([C, C], dtype)
            t_l1 = T.alloc_L1([C, C], dtype)
            tmp_l0 = T.alloc_L0C([C, C], dtype)
            p2_l0 = T.alloc_L0C([C, C], dtype)
            T.copy(T_in[t, 0:C, 0:C], t_l1)
            T.copy(P_in[t, 0:C, 0:C], p_l1)
            T.gemm(p_l1, t_l1, tmp_l0, size=[C, C, C], initC=True)
            T.gemm(p_l1, p_l1, p2_l0, size=[C, C, C], initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(tmp_l0, tmp_out[t, 0:C, 0:C])
                T.copy(p2_l0, p2_out[t, 0:C, 0:C])

    return seg_cube_main


@tilelang.jit(out_idx=[2], target="npuir")
def seg_add(tasks: int, C: int = 64, dtype: str = "float32"):
    @T.prim_func
    def seg_add_main(
        T_in: T.Tensor((tasks, C, C), dtype),
        tmp: T.Tensor((tasks, C, C), dtype),
        T_out: T.Tensor((tasks, C, C), dtype),
    ):
        with T.Kernel(tasks, is_npu=True) as (t, _):
            t_ub = T.alloc_ub((C, C), dtype)
            tmp_ub = T.alloc_ub((C, C), dtype)
            out_ub = T.alloc_ub((C, C), dtype)
            T.copy(T_in[t, 0:C, 0:C], t_ub)
            T.copy(tmp[t, 0:C, 0:C], tmp_ub)
            T.vadd(t_ub, tmp_ub, out_ub)
            T.copy(out_ub, T_out[t, 0:C, 0:C])

    return seg_add_main


def split_inverse(n: torch.Tensor, levels: int = 6):
    """n: [tasks, C, C] fp32 NPU tensor, strictly lower. Returns A [tasks, C, C]."""
    tasks, C, _ = n.shape
    eye = torch.eye(C, dtype=torch.float32, device=n.device)
    Tmat = eye.unsqueeze(0).repeat(tasks, 1, 1).contiguous()
    P = n.contiguous()
    k_cube = seg_cube(tasks, C, "float32")
    k_add = seg_add(tasks, C, "float32")
    for _ in range(levels):
        tmp, P2 = k_cube(Tmat, P)
        Tmat = k_add(Tmat, tmp)
        P = P2
    return Tmat


def _golden(n: torch.Tensor):
    C = n.shape[-1]
    eye = torch.eye(C, dtype=torch.float64)
    return torch.linalg.solve(eye - n.double(), eye.expand(n.shape[0], C, C))


def run_case(tasks=2, C=64, valid=None, seed=0):
    if valid is None:
        valid = C
    torch.manual_seed(seed)
    n_cpu = torch.randn(tasks, C, C, dtype=torch.float64)
    n_cpu = torch.tril(n_cpu, -1) * 0.15
    if valid < C:
        n_cpu[:, valid:, :] = 0.0
        n_cpu[:, :, valid:] = 0.0
    n = n_cpu.float().npu()
    a = split_inverse(n)
    torch.npu.synchronize()
    ref = _golden(n_cpu)
    err = float((a.cpu().double() - ref).abs().max())
    print(f"[K2-split] tasks={tasks} C={C} valid={valid} max_abs_err={err:.3e}")
    assert err < 2e-3, f"split inverse mismatch: {err:.3e}"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--C", type=int, default=64)
    args = ap.parse_args()
    run_case(args.tasks, args.C, args.C, seed=0)
    run_case(args.tasks, args.C, 36, seed=1)
    print("[K2-split] PASS")


if __name__ == "__main__":
    main()
