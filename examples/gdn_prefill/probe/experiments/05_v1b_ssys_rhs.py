import os; os.environ.setdefault("TILELANG_ASCEND_MODE","Expert")
import tilelang, tilelang.language as T, torch, torch_npu
C,D,DV=16,16,16
@tilelang.jit(out_idx=[6,7], target="npuir")
def build():
    @T.prim_func
    def main(q: T.Tensor((1,C,D),"float16"), k: T.Tensor((1,C,D),"float16"), v: T.Tensor((1,C,DV),"float16"),
             g: T.Tensor((1,C),"float16"), beta: T.Tensor((1,C),"float16"), s: T.Tensor((1,D,DV),"float16"),
             sn16: T.Tensor((1,C,C),"float16"), rhs: T.Tensor((1,C,DV),"float32"),
             kk: T.Tensor((1,C,C),"float32"), ks: T.Tensor((1,C,DV),"float32")):
        with T.Kernel(1, is_npu=True) as (cid,_):
            with T.Scope("Cube"):
                l1k=T.alloc_L1([C,D],"float16"); l1s=T.alloc_L1([D,DV],"float16")
                lcc=T.alloc_L0C([C,C],"float32"); lcd=T.alloc_L0C([C,DV],"float32")
                T.copy(k[0,0:C,0:D],l1k); T.copy(s[0,0:D,0:DV],l1s)
                T.gemm(l1k,l1k,lcc,size=[C,D,C],b_transpose=True,initC=True)
                with T.rs("PIPE_FIX"): T.copy(lcc,kk[0,0:C,0:C]); T.sync_block_set(0)
                T.gemm(l1k,l1s,lcd,size=[C,D,DV],initC=True)
                with T.rs("PIPE_FIX"): T.copy(lcd,ks[0,0:C,0:DV]); T.sync_block_set(1)
            with T.Scope("Vector"):
                g16=T.alloc_ub((C,1),"float16"); gc=T.alloc_ub((C,1),"float32"); gcR=T.alloc_ub((1,C),"float32")
                gcL=T.alloc_ub((C,C),"float32"); gcS=T.alloc_ub((C,C),"float32"); gexp=T.alloc_ub((C,1),"float32")
                E=T.alloc_ub((C,C),"float32"); d=T.alloc_ub((C,C),"float32"); m=T.alloc_ub((C,C),"float32")
                ssis=T.alloc_ub((C,C),"float32"); KK=T.alloc_ub((C,C),"float32")
                b1=T.alloc_ub((C,1),"float32"); bM=T.alloc_ub((C,C),"float32"); bV=T.alloc_ub((C,DV),"float32")
                KS=T.alloc_ub((C,DV),"float32"); vf=T.alloc_ub((C,DV),"float32"); rhsv=T.alloc_ub((C,DV),"float32")
                gexM=T.alloc_ub((C,DV),"float32"); iL=T.alloc_ub((C,C),"float32"); iS=T.alloc_ub((C,C),"float32")
                snh=T.alloc_ub((C,C),"float16"); v16=T.alloc_ub((C,DV),"float16")
                T.copy(g[0,0:C],g16[0:C,0]); T.vcast(g16,gc,round_mode="rint"); T.cumsum(gc,gc,dim=0)
                T.copy(gc[0:C,0],gcR[0,0:C]); T.vbrc(gc,gcL); T.vbrc(gcR,gcS)
                T.arange(iL,[1,0],0); T.arange(iS,[0,1],0); T.vsub(iL,iS,d)
                T.vsub(gcL,gcS,E); T.vexp(E,E)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0); T.copy(kk[0,0:C,0:C],KK)
                    T.sync_block_wait(1); T.copy(ks[0,0:C,0:DV],KS)
                    T.copy(v[0,0:C,0:DV],v16); T.vcast(v16,vf,round_mode="rint")
                T.copy(beta[0,0:C], g16[0:C,0]); T.vcast(g16,b1,round_mode="rint")
                T.vexp(gc,gexp); T.vbrc(b1,bM); T.vbrc(b1,bV); T.vbrc(gexp,gexM)
                T.vmin(d,1.0,m); T.vmax(m,0.0,m); T.vmul(E,m,m)   # G strict
                T.vmul(bM,m,m); T.vmul(m,KK,ssis)                  # S_sys
                T.vcast(ssis,snh); T.copy(snh, sn16[0,0:C,0:C])
                T.vmul(KS,gexM,KS); T.vsub(vf,KS,rhsv); T.vmul(rhsv,bV,rhsv)
                T.copy(rhsv, rhs[0,0:C,0:DV])
    return main
torch.manual_seed(0); scale=D**-0.5
q=torch.randn(1,C,D).half(); k=torch.randn(1,C,D).half(); v=torch.randn(1,C,DV).half()
g=(-torch.rand(1,C)*0.3).half(); beta=torch.rand(1,C).half(); s=torch.randn(1,D,DV).half()
kkw=torch.empty(1,C,C,dtype=torch.float32).npu(); ksw=torch.empty(1,C,DV,dtype=torch.float32).npu()
sn16,rhs=build()(q.npu(),k.npu(),v.npu(),g.npu(),beta.npu(),s.npu(),kkw,ksw)
kf,kq,vf,bf=k[0].float(),q[0].float(),v[0].float(),beta[0].float()
gc=torch.cumsum(g[0].float(),0); E=torch.exp(gc[:,None]-gc[None,:])
l=torch.arange(C).float(); d=l[:,None]-l[None,:]
Gst=E*((d>0).float())
KK=kf@kf.T; KS=kf@s[0].float()
Ssys=bf[:,None]*Gst*KK
rhs_ref=bf[:,None]*(vf-KS*torch.exp(gc)[:,None])
print("S_sys(before cast) err", float((sn16[0].cpu().float()-Ssys).abs().max()))
print("rhs err", float((rhs[0].cpu()-rhs_ref).abs().max()))
