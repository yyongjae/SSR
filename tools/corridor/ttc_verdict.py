#!/usr/bin/env python
"""km 10/15: TTC from the corridor's dynamic field (official PDMScorer._calculate_ttc re-done on occupancy planes).
Official: at every 0.1 s step t, the ego footprint shifted by speed * heading * dt, dt in {0, .3, .6, .9} s, vs the
agents at t + dt; counts if the agent is ahead (relative angle < 30 deg) [or, in intersections / multi-lane, not
behind (< 150 deg)].  Here: footprint sample points vs occupancy pixels, angle of the hit point from the rear axle.
  GT : 0.1 s planes (corridor_dt01, t + dt <= 4 s);  pred: the 8 swept 0.5 s planes, p > 0.5.
Evaluation only (official TTC of the candidates is used only to measure agreement / selection).
"""
import numpy as np, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsim.agents.para_ssr.modules.torch_tracker import track
from navsim.agents.para_ssr.modules.corridor_head import FRONT, REAR, HALF_W
torch.set_num_threads(4)
E = sys.argv[1]; N = sys.argv[2] if len(sys.argv) > 2 else "400"
z = np.load(f"{E}/official_all_{N}.npz"); rows = z["rows"]
traj = np.load(f"{E}/traj.npy", mmap_mode="r"); status = np.load(f"{E}/status.npy", mmap_mode="r")
cocc = np.load(f"{E}/cocc.npy", mmap_mode="r")
G = np.load("/home/external-user/ssd/kyungmin_ck/corridor_dt01/navtrain_val/dyn_occ.npy", mmap_mode="r")
gx, gy = np.meshgrid(np.linspace(-REAR, FRONT, 9), np.linspace(-HALF_W, HALF_W, 5), indexing="ij")
gx, gy = gx.ravel(), gy.ravel()                                                      # 45 footprint points
DTS = (0, 3, 6, 9)
def hits(occ_at, x, y, h, v, ang):
    """occ_at(tau) -> bool [100, 200] or None; x,y,h,v [K, 41]; -> bool [K] TTC infraction."""
    K = x.shape[0]; bad = np.zeros(K, bool)
    c, s = np.cos(h), np.sin(h)
    for t in range(1, 41):
        mov = v[:, t] >= 5e-3
        for d in DTS:
            tau = t + d
            o = occ_at(tau)
            if o is None:
                continue
            sh = v[:, t] * d * 0.1
            px = x[:, t, None] + c[:, t, None] * (gx + sh[:, None]) - s[:, t, None] * gy
            py = y[:, t, None] + s[:, t, None] * (gx + sh[:, None]) + c[:, t, None] * gy
            row = np.floor(px / 0.32).astype(int); col = np.floor((-py + 32) / 0.32).astype(int)
            inr = (row >= 0) & (row < 100) & (col >= 0) & (col < 200)
            hit = np.zeros_like(inr); hit[inr] = o[row[inr], col[inr]]
            rel = np.abs(np.arctan2(np.sin(np.arctan2(py - y[:, t, None], px - x[:, t, None]) - h[:, t, None]),
                                    np.cos(np.arctan2(py - y[:, t, None], px - x[:, t, None]) - h[:, t, None])))
            bad |= (hit & (rel < np.deg2rad(ang)) & mov[:, None]).any(1)
    return bad
out = {k: np.zeros((len(rows), 256), bool) for k in ("gt30", "gt150", "pred30", "pred150")}
for i, r in enumerate(rows):
    tr = torch.from_numpy(np.array(traj[r], np.float32)); init = torch.zeros(256, 7); init[:, :4] = torch.from_numpy(np.array(status[r], np.float32))
    st = track(tr, init).numpy().astype(np.float64)
    x, y, h = st[..., 0], st[..., 1], st[..., 2]
    v = np.zeros_like(x); v[:, 1:] = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)) / 0.1; v[:, 0] = np.hypot(*np.array(status[r][:2], np.float64))
    g = np.unpackbits(np.asarray(G[r]), axis=-1).astype(bool)                       # [40, 100, 200], plane k = (k+1)*0.1 s
    p = np.asarray(cocc[r]) > 127                                                    # [8, 100, 200], plane j = ((j)*.5, (j+1)*.5]
    gat = lambda tau: g[tau - 1] if 1 <= tau <= 40 else None
    pat = lambda tau: p[(tau - 1) // 5] if 1 <= tau <= 40 else None
    for a in (30, 150):
        out[f"gt{a}"][i] = hits(gat, x, y, h, v, a); out[f"pred{a}"][i] = hits(pat, x, y, h, v, a)
    if i % 50 == 0:
        print(i, flush=True)
np.savez(f"{E}/ttc_verdict_{N}.npz", rows=rows, **out)
print("saved")
