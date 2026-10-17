import os; os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer
import tilelang, tilelang.language as T, torch, torch_npu
C=16
@tilelang.jit(out_idx=[1,2,3], target="npuir")
def build():
    @T.prim_func
    def main(g: T.Tensor((1,C),"float16"), gcw: T.Tensor((1,C),"float32"),
             Gs: T.Tensor((1,C,C),"float32"), Gi: T.Tensor((1,C,C),"float32")):
        with T.Kernel(1, is_npu=True) as (cid,_):
            with T.Scope("Vector"):
                g16=T.alloc_ub((C,1),"float16"); gc=T.alloc_ub((C,1),"float32"); gcR=T.alloc_ub((1,C),"float32")
                gcL=T.alloc_ub((C,C),"float32"); gcS=T.alloc_ub((C,C),"float32")
                E=T.alloc_ub((C,C),"float32"); d=T.alloc_ub((C,C),"float32"); m=T.alloc_ub((C,C),"float32")
                ub=T.alloc_ub((C,C),"float32"); iL=T.alloc_ub((C,C),"float32"); iS=T.alloc_ub((C,C),"float32")
                T.copy(g[0,0:C], g16[0:C,0]); T.vcast(g16,gc,round_mode="rint"); T.cumsum(gc,gc,dim=0)
                T.copy(gc[0:C,0], gcw[0,0:C]); T.copy(gc[0:C,0], gcR[0,0:C])
                T.vbrc(gc,gcL); T.vbrc(gcR,gcS)
                T.arange(iL,[1,0],0); T.arange(iS,[0,1],0); T.vsub(iL,iS,d)
                T.vsub(gcL,gcS,E); T.vexp(E,E)
                T.vmin(d,1.0,m); T.vmax(m,0.0,m); T.vmul(E,m,ub); T.copy(ub, Gs[0,0:C,0:C])
                T.vadd(d,1.0,m); T.vmin(m,1.0,m); T.vmax(m,0.0,m); T.vmul(E,m,ub); T.copy(ub, Gi[0,0:C,0:C])
    return main
torch.manual_seed(0); g=(-torch.rand(1,C)*0.3).to(torch.float16)
gcw,Gs,Gi=build()(g.npu())
gcr=torch.cumsum(g[0].float(),0); l=torch.arange(C).float(); d=l[:,None]-l[None,:]
E=torch.exp(gcr[:,None]-gcr[None,:]); Gsr=E*(d>0).float(); Gir=E*(d>=0).float()
print("gc", float((gcw[0].cpu()-gcr).abs().max()))
print("Gstrict", float((Gs[0].cpu()-Gsr).abs().max()))
print("Gincl", float((Gi[0].cpu()-Gir).abs().max()))
