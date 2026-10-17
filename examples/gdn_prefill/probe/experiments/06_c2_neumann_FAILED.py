import os; os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer
import tilelang, tilelang.language as T, torch, torch_npu
C,DV=16,16; K=C-1
@tilelang.jit(out_idx=[1], target="npuir")
def build():
    @T.prim_func
    def main(sn: T.Tensor((1,C,C),"float16"), eye: T.Tensor((1,C,C),"float16"), rh: T.Tensor((1,C,DV),"float16"),
             W: T.Tensor((1,C,DV),"float32"), pgm: T.Tensor((1,C,C),"float32"), mgm: T.Tensor((1,C,C),"float32")):
        with T.Kernel(1, is_npu=True) as (cid,_):
            with T.Scope("Cube"):
                lsn=T.alloc_L1([C,C],"float16"); li=T.alloc_L1([C,C],"float16"); lrh=T.alloc_L1([C,DV],"float16")
                lp=T.alloc_L1([C,C],"float16"); lm=T.alloc_L1([C,C],"float16")
                l0p=T.alloc_L0C([C,C],"float32"); l0m=T.alloc_L0C([C,C],"float32"); l0w=T.alloc_L0C([C,DV],"float32")
                T.copy(sn[0,0:C,0:C],lsn); T.copy(eye[0,0:C,0:C],li); T.copy(rh[0,0:C,0:DV],lrh)
                T.gemm(lsn,li,l0p,size=[C,C,C],initC=True)     # P=Sn
                T.gemm(li,li,l0m,size=[C,C,C],initC=True)      # M=I
                T.copy(l0p,pgm[0,0:C,0:C]); T.copy(pgm[0,0:C,0:C],lp)
                for j in T.serial(K):
                    T.gemm(lp,li,l0m,size=[C,C,C],initC=False)   # M+=P
                    T.gemm(lsn,lp,l0p,size=[C,C,C],initC=True)   # P=Sn@P
                    T.copy(l0p,pgm[0,0:C,0:C]); T.copy(pgm[0,0:C,0:C],lp)
                T.copy(l0m,mgm[0,0:C,0:C]); T.copy(mgm[0,0:C,0:C],lm)
                T.gemm(lm,lrh,l0w,size=[C,C,DV],initC=True)
                T.copy(l0w,W[0,0:C,0:DV])
    return main
torch.manual_seed(0)
# strict lower Sn
Sns=torch.tril(torch.randn(C,C),-1).half()
eyes=torch.eye(C).half()
rhs=torch.randn(C,DV).half()
pg=torch.zeros(1,C,C).npu(); mg=torch.zeros(1,C,C).npu()
W=build()(Sns[None].npu(),eyes[None].npu(),rhs[None].npu(),pg,mg)
ref=torch.linalg.inv(torch.eye(C)+Sns.float())@rhs.float()
print("W err", float((W[0].cpu()-ref).abs().max()))
print("M err", float((mg[0].cpu()-(torch.eye(C)+Sns.float()@torch.eye(C)+ (torch.linalg.inv(torch.eye(C)+Sns.float())-torch.eye(C)))).abs().max()))
