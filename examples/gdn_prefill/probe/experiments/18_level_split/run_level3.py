import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "prefill_blocksolve_A_bthd"))

import torch
import torch_npu  # noqa: F401

import level3_variant as m


def run_case(total_tokens, head, key_head, dim_k, dtype=torch.float16, cores=0):
    print(f"[probe] tokens={total_tokens} head={head} kh={key_head} dk={dim_k} cores={cores}", flush=True)
    k, g, beta = m._mk_inputs(total_tokens, head, key_head, dim_k, dtype, 0)
    s_pad = ((total_tokens + m.CHUNK - 1) // m.CHUNK) * m.CHUNK
    a_pad = torch.full((s_pad, head, m.CHUNK), 1024.0, dtype=dtype, device="npu")
    try:
        kwargs = {} if not cores else {"num_cores": cores}
        A, _ = m.prefill_blocksolve_A_bthd(
            k.npu(), g.npu(), beta.npu(), None, m.CHUNK, True, False, False, False,
            total_tokens, a_pad, **kwargs,
        )
        torch.npu.synchronize()
        print(f"[probe] COMPILED_AND_RAN shape={tuple(A.shape)}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:240]}", flush=True)
        return False


if __name__ == "__main__":
    ok = [run_case(64, 2, 1, 64), run_case(1024, 4, 1, 128)]
    print("[probe] result", ok, flush=True)
