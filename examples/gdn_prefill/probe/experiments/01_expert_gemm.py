import os; os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer
import tilelang, tilelang.language as T, torch, torch_npu
def build(C=64,D=64,bt=False):
    @tilelang.jit(out_idx=[2], target="npuir")
    def f():
        @T.prim_func
        def main(a: T.Tensor((C,D),"float16"), b: T.Tensor((C,D),"float16"), c: T.Tensor((C,C),"float32")):
            with T.Kernel(1, is_npu=True) as (cid,_):
                with T.Scope("Cube"):
                    la=T.alloc_L1([C,D],"float16"); lb=T.alloc_L1([C,D],"float16"); lc=T.alloc_L0C([C,C],"float32")
                    T.copy(a, la)
                    T.copy(b, lb)
                    T.gemm(la, lb, lc, initC=True, b_transpose=bt, size=[C,D,C])
                    with T.rs("PIPE_FIX"):
                        T.copy(lc, c)
        return main
    return f()
torch.manual_seed(0)
a=torch.randn(64,64,dtype=torch.float16).npu(); b=torch.randn(64,64,dtype=torch.float16).npu()
for bt in (False, True):
    try:
        k=build(bt=bt); c=k(a,b); torch.npu.synchronize()
        ref=(a.cpu().float() @ (b.cpu().float().T if bt else b.cpu().float()))
        print(f"b_transpose={bt}: OK max_err={float((c.cpu()-ref).abs().max()):.3e}")
    except Exception as e:
        print(f"b_transpose={bt}: FAIL {str(e)[:80]}")
