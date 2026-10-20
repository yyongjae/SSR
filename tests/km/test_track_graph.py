"""km 10/20: the CUDA-graphed track() == the eager _track() bitwise (outputs and input gradients), every call shape of
the c64 training step: model forward (float, full), corridor_plan_loss (float, full, grad), labels (double, full +
angular), human (double, full + angular, N = B).  python tests/km/test_track_graph.py"""
import time
import numpy as np
import torch
from navsim.agents.para_ssr.modules import torch_tracker as TT

torch.manual_seed(0)
dev = "cuda"
anc = torch.tensor(np.load("resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32, device=dev)
B, K = 4, anc.shape[0]


def sample():
    off = torch.randn(B, K, 8, 3, device=dev) * torch.tensor([0.3, 0.2, 0.03], device=dev)
    st = torch.cat([torch.rand(B, 1, device=dev) * 15, torch.randn(B, 1, device=dev) * 0.2,
                    torch.randn(B, 2, device=dev)], -1)
    ego = torch.cat([st, torch.randn(B, 4, device=dev) * 0.05], -1)                       # (+ steer, rate, w, dw)
    return anc[None] + off, st, ego


ok = True
for it in range(3):                                                                     # 3 batches: replays, not just capture
    cand, st, ego = sample()
    init4 = torch.cat([st, st.new_zeros(B, 3)], -1).repeat_interleave(K, 0)
    cases = {
        "forward float full": (cand.reshape(-1, 8, 3), init4, dict(full=True)),
        "labels double full+angular": (cand.reshape(-1, 8, 3).double(), ego.double().repeat_interleave(K, 0),
                                       dict(full=True, angular=True)),
        "human double full+angular": (cand[:, 0].double(), ego.double(), dict(full=True, angular=True)),
    }
    for name, (p, i, kw) in cases.items():
        with torch.no_grad():
            a, b = TT._track(p, i, **kw), TT.track(p, i, **kw)
        same = torch.equal(a, b)
        ok &= same
        print(f"batch {it} {name:28s} bitwise {same}  max|d| {(a - b).abs().max().item():.3g}")
    w = torch.randn(B * K, 41, 7, device=dev)
    pa = cand.reshape(-1, 8, 3).clone().requires_grad_(True)
    pb = cand.reshape(-1, 8, 3).clone().requires_grad_(True)
    oa = TT._track(pa, init4, full=True); (oa * w).sum().backward()
    ob = TT.track(pb, init4, full=True); (ob * w).sum().backward()
    same = torch.equal(oa, ob) and torch.equal(pa.grad, pb.grad)
    ok &= same
    print(f"batch {it} {'plan loss float full grad':28s} bitwise {same}  max|d out| {(oa - ob).abs().max().item():.3g}"
          f"  max|d grad| {(pa.grad - pb.grad).abs().max().item():.3g}")
for name, f in (("eager", TT._track), ("graph", TT.track)):
    p, i = cand.reshape(-1, 8, 3).double(), ego.double().repeat_interleave(K, 0)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            f(p, i, full=True, angular=True)
    torch.cuda.synchronize(); print(f"{name}: {(time.time() - t0) / 10 * 1e3:.1f} ms per call (labels shape)")
print("ALL BITWISE" if ok else "MISMATCH")
