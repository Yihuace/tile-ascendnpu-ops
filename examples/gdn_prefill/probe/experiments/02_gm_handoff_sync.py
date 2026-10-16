import os; os.environ.setdefault("TILELANG_ASCEND_MODE","Expert")
import tilelang, tilelang.language as T, torch, torch_npu
C,D=16,16
@tilelang.jit(out_idx=[2], target="npuir")
def build2():
    @T.prim_func
    def main(k: T.Tensor((1,C,D),"float16"), s: T.Tensor((1,D,D),"float16"),
             out: T.Tensor((1,C,D),"float32"), kk_ws: T.Tensor((1,C,C),"float32")):
        with T.Kernel(1, is_npu=True) as (cid,_):
            with T.Scope("Cube"):
                l1k=T.alloc_L1([C,D],"float16"); l1k2=T.alloc_L1([C,D],"float16")
                l0=T.alloc_L0C([C,C],"float32")
                T.copy(k[0,0:C,0:D], l1k); T.copy(k[0,0:C,0:D], l1k2)
                T.gemm(l1k, l1k2, l0, size=[C,D,C], b_transpose=True, initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0, kk_ws[0,0:C,0:C])
                    T.sync_block_set(0)
            with T.Scope("Vector"):
                ub=T.alloc_ub((C,C),"float32"); uc=T.alloc_ub((C,D),"float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(kk_ws[0,0:C,0:C], ub)
                T.vmul(ub, 2.0, ub)
                T.copy(ub[0:C,0:D], uc)
                T.copy(uc, out[0,0:C,0:D])
    return main
torch.manual_seed(1)
k2=torch.randn(1,C,D,dtype=torch.float16).npu(); s2=torch.randn(1,D,D,dtype=torch.float16).npu()
kkw2=torch.zeros(1,C,C,dtype=torch.float32).npu()
out2=build2()(k2,s2,kkw2)
ref2=(k2[0].cpu().float()@k2[0].cpu().float().T)*2
print("with sync: VMUL err", float((out2[0].cpu()-ref2[:, :D]).abs().max()))
