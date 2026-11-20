import importlib.util, sys
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["probe"]=m; spec.loader.exec_module(m)
k = m._gqa_bwd_kernel(2,32,8,512,128,True,'float16')
print("COMPILE_OK", sys.argv[1])
