import os; os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer
import tilelang, tilelang.language as T, torch, torch_npu
C,D,DV=16,16,16
@tilelang.jit(out_idx=[3,4,5,6], target="npuir")
def build():
    @T.prim_func
    def main(q: T.Tensor((1,C,D),"float16"), k: T.Tensor((1,C,D),"float16"), s: T.Tensor((1,D,DV),"float16"),
             kk: T.Tensor((1,C,C),"float32"), qk: T.Tensor((1,C,C),"float32"),
             ks: T.Tensor((1,C,DV),"float32"), qs: T.Tensor((1,C,DV),"float32")):
        with T.Kernel(1, is_npu=True) as (cid,_):
            with T.Scope("Cube"):
                l1k=T.alloc_L1([C,D],"float16"); l1q=T.alloc_L1([C,D],"float16"); l1s=T.alloc_L1([D,DV],"float16")
                lcc=T.alloc_L0C([C,C],"float32"); lcd=T.alloc_L0C([C,DV],"float32")
                T.copy(k[0,0:C,0:D], l1k); T.copy(q[0,0:C,0:D], l1q); T.copy(s[0,0:D,0:DV], l1s)
                T.gemm(l1k, l1k, lcc, size=[C,D,C], b_transpose=True, initC=True)
                T.copy(lcc, kk[0,0:C,0:C])
                T.gemm(l1q, l1k, lcc, size=[C,D,C], b_transpose=True, initC=True)
                T.copy(lcc, qk[0,0:C,0:C])
                T.gemm(l1k, l1s, lcd, size=[C,D,DV], initC=True)
                T.copy(lcd, ks[0,0:C,0:DV])
                T.gemm(l1q, l1s, lcd, size=[C,D,DV], initC=True)
                T.copy(lcd, qs[0,0:C,0:DV])
    return main
torch.manual_seed(0)
q=torch.randn(1,C,D,dtype=torch.float16).npu(); k=torch.randn(1,C,D,dtype=torch.float16).npu()
s=torch.randn(1,D,DV,dtype=torch.float16).npu()
kk,qk,ks,qs = build()(q,k,s)
def e(a,b): return float((a.cpu()[0].float()-b).abs().max())
print("KK", e(kk, k[0].cpu().float()@k[0].cpu().float().T))
print("QK", e(qk, q[0].cpu().float()@k[0].cpu().float().T))
print("KS", e(ks, k[0].cpu().float()@s[0].cpu().float()))
print("QS", e(qs, q[0].cpu().float()@s[0].cpu().float()))
