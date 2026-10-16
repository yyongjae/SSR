#!/usr/bin/env python
"""km 10/15: geometry check of corridor_plan_loss: hinge == 0 (margin 0) on the 256 raw anchors of navtrain_val tokens
vs the official DAC / NC labels (fidelity_pred npz), using the GT corridor targets exactly as the target builder does."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)


def main() -> int:
    import pandas as pd
    import torch
    from navsim.agents.para_ssr.modules.corridor_head import _occ_dist, corridor_plan_loss  # noqa: F401
    from navsim.agents.para_ssr.modules import corridor_head as CH
    from tools.ck.data.common import MC_ROOT
    from tools.corridor.build_corridor import load_mc
    from tools.corridor.fidelity import ANCH
    from tools.corridor.fidelity_pred import K, PRED, ROOT, SPLIT

    n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    toks = pd.read_parquet(f"{ROOT}/packed/{SPLIT}/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{K}/corridor/{SPLIT}/ok.npy")
    for v in PRED.values():
        ok &= np.load(f"{v}/ok.npy")
    rows = np.sort(np.random.default_rng(0).choice(np.where(ok)[0], 600, replace=False))
    F = np.load(f"{K}/km_runs/corridor_head/fidelity_pred_{SPLIT}.npz")
    dac_off, nc_off = F["dac"].reshape(600, 256), F["nc"].reshape(600, 256)
    where = {}
    for log in os.scandir(MC_ROOT[SPLIT]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    S = np.load(f"{K}/corridor_dt01/{SPLIT}/static_sdf.npy", mmap_mode="r")
    D = np.load(f"{K}/corridor_dt01/{SPLIT}/dyn_occ.npy", mmap_mode="r")
    A = torch.tensor(np.load(ANCH), dtype=torch.float32, device="cuda")
    vs, vo, ds, ns = [], [], [], []
    for j, r in enumerate(rows[:n_tok]):
        dcs = load_mc(where[toks[r]]).ego_state.dynamic_car_state
        status = torch.tensor([[dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y,
                                dcs.rear_axle_acceleration_2d.x, dcs.rear_axle_acceleration_2d.y]], device="cuda")
        tg = {"corridor_sdf": torch.tensor(np.asarray(S[r], np.float32), device="cuda")[None],
              "corridor_occ_dist": torch.tensor(_occ_dist(np.unpackbits(np.asarray(D[r]), axis=-1).astype(bool))
                                                .astype(np.float32), device="cuda")[None],
              "corridor_valid": torch.ones(1, device="cuda")}
        pred = {"trajectory_anchors": A, "trajectory_offset": torch.zeros(1, 256, 8, 3, device="cuda")}
        # per-candidate hinge with margin 0 (re-implemented via the loss on one candidate set)
        traj = A[None]
        from navsim.agents.para_ssr.modules.torch_tracker import track
        init = torch.cat([status, torch.zeros(1, 3, device="cuda")], -1).repeat(256, 1)
        st = track(traj[0], init)[None, :, 1:]
        x, y, h = st[..., 0], st[..., 1], st[..., 2]
        c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
        cx = torch.tensor([CH.FRONT, CH.FRONT, -CH.REAR, -CH.REAR], device="cuda")
        cy = torch.tensor([CH.HALF_W, -CH.HALF_W, -CH.HALF_W, CH.HALF_W], device="cuda")
        px, py = x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy
        val, inr = CH._sample(tg["corridor_sdf"][:, None], torch.stack([px, py], -1).reshape(1, 1, -1, 2))
        vs.append((((0 - val).clamp(min=0) * inr).reshape(256, -1).amax(-1) == 0).cpu().numpy())
        fx, fy = torch.meshgrid(torch.linspace(0.0, CH.FRONT, 4, device="cuda"),
                                torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device="cuda"), indexing="ij")
        fx, fy = fx.reshape(-1), fy.reshape(-1)
        qx, qy = x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy
        q = torch.stack([qx, qy], -1).permute(0, 2, 1, 3, 4).reshape(1, 40, -1, 2)
        dv, di = CH._sample(tg["corridor_occ_dist"][None, ...][0][None] if False else tg["corridor_occ_dist"], q)
        vo.append((((0 - dv).clamp(min=0) * di).reshape(40, 256, 12).amax(-1).amax(0) == 0).cpu().numpy())
        ds.append(dac_off[j]); ns.append(nc_off[j])
        if j == 0:
            l, lg = corridor_plan_loss(pred, tg, status, 0.1, 0.3, 3.0)
            print("loss on raw anchors (token 0):", float(l), {k: round(float(v), 4) for k, v in lg.items()})
    vs, vo, ds, ns = map(np.concatenate, (vs, vo, ds, ns))
    print(f"tokens {n_tok}: static hinge==0 vs official DAC agree {100 * (vs == ds).mean():.2f}% "
          f"(false-pass {100 * (vs & ~ds).mean():.2f}, false-fail {100 * (~vs & ds).mean():.2f}); "
          f"occ hinge==0 (front half) vs official NC agree {100 * (vo == ns).mean():.2f}% "
          f"(false-pass {100 * (vo & ~ns).mean():.2f}, false-fail {100 * (~vo & ns).mean():.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
