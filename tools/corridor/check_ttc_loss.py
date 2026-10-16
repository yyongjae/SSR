#!/usr/bin/env python
"""km 10/15: geometry check of the TTC term of corridor_plan_loss (margin 0, GT targets as the builder makes them)
on the model's candidates of an eval dump vs their official TTC / NC / DAC (oracle_breakdown npz).  Evaluation only."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsim.agents.para_ssr.modules.corridor_head import _occ_dist, corridor_plan_loss
torch.set_num_threads(6)
E = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
z = np.load(f"{E}/official_all_400.npz"); rows, O = z["rows"][:n], z["O"][:n]
traj = np.load(f"{E}/traj.npy", mmap_mode="r"); status = np.load(f"{E}/status.npy", mmap_mode="r")
G = "/home/external-user/ssd/kyungmin_ck/corridor_dt01/navtrain_val"
S = np.load(f"{G}/static_sdf.npy", mmap_mode="r"); D = np.load(f"{G}/dyn_occ.npy", mmap_mode="r")
vs, vo, vt = [], [], []
for r in rows:
    tg = {"corridor_sdf": torch.tensor(np.asarray(S[r], np.float32))[None],
          "corridor_occ_dist": torch.tensor(_occ_dist(np.unpackbits(np.asarray(D[r]), axis=-1).astype(bool)).astype(np.float32))[None],
          "corridor_valid": torch.ones(1)}
    pr = {"trajectory_anchors": torch.tensor(np.array(traj[r], np.float32)), "trajectory_offset": torch.zeros(1, 256, 8, 3)}
    a, b_, c = corridor_plan_loss(pr, tg, torch.tensor(np.array(status[r], np.float32))[None], 0.0, 0.0, 3.0, 1.0, True)
    vs.append(a[0].numpy()); vo.append(b_[0].numpy()); vt.append(c[0].numpy())
vs, vo, vt = (np.stack(v) == 0 for v in (vs, vo, vt))
ttc, nc, dac = O[..., 4] == 1, O[..., 1] == 1, O[..., 2] == 1
both = vo & vt
for nm, a, o in (("static vs DAC", vs, dac), ("occ vs NC", vo, nc), ("ttc term vs TTC", vt, ttc), ("occ&ttc vs TTC", both, ttc)):
    print(f"{nm:16s} agree {100*(a==o).mean():.2f}  false-pass {100*(a&~o).mean():.2f}  false-fail {100*(~a&o).mean():.2f}")
