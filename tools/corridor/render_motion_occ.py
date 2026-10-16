#!/usr/bin/env python
"""km 10/16: test A — the occupancy the det/motion head implies, on the corridor grid, for the corridor verdict.

From dump_corr_model.py --det (top-64 queries): agent a with detection probability p_a (max class sigmoid), box W x L,
yaw, and 6 motion modes (per-step offsets at 0.5 s, mode weights pi_m = sigmoid(logit) / sum) is rendered at
t = (k + 1) * 0.1 s, k < 40 (the GT corridor_dt01 planes): centre linearly interpolated between the 0.5 s points,
heading = direction of the 0.5 s segment when it moves > 0.25 m, else the box yaw; static classes (cone, barrier,
czone sign, generic object) do not move.  Modes are exclusive within an agent, agents independent:
    occ(cell, t) = 1 - prod_a (1 - p_a * sum_m pi_m [cell centre inside box_{a,m}(t)])
  --motion cv : constant velocity from the box (vx, vy), one mode (diagnostic: does the motion head beat CV?)
  --det-mode hard : p_a -> [p_a >= 0.3]

  <dump>/<split>/mocc_<tag>.npy u8 [N, 40, 100, 200] (probability * 255), moccdone_<tag>.npy
  CUDA_VISIBLE_DEVICES=1 $PY tools/corridor/render_motion_occ.py --dump <dir> --split navtrain_val [--motion cv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

H, W, RES, T = 100, 200, 0.32, 40
STATIC = (3, 4, 5, 6)              # DET_CLASS_NAMES: traffic_cone, barrier, czone_sign, generic_object
WIN = 32                           # half window in cells (10.2 m)


def render(box, cls, traj, tcls, motion="head", det_mode="prob"):
    """one token: box [A, 10], cls [A, 7], traj [A, M, 8, 2], tcls [A, M] (torch, cuda) -> occ [40, H, W]"""
    import torch
    dev = box.device
    p, k = cls.max(-1)
    if det_mode == "hard":
        p = (p >= 0.3).float()
    keep = p > 0.02
    box, p, k, traj, tcls = box[keep], p[keep], k[keep], traj[keep], tcls[keep]
    A = box.shape[0]
    if A == 0:
        return torch.zeros(T, H, W, device=dev)
    c0 = box[:, :2]                                                                     # SSR (x right, y fwd)
    wd, ln = box[:, 2].exp(), box[:, 3].exp()
    d0 = torch.stack([-box[:, 6], -box[:, 7]], -1)                                      # heading in SSR axes
    d0 = d0 / d0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    if motion == "cv":
        traj = (box[:, None, None, 8:10] * 0.5).expand(A, 1, 8, 2)
        pi = torch.ones(A, 1, device=dev)
    else:
        pi = tcls.sigmoid()
        pi = pi / pi.sum(-1, keepdim=True).clamp(min=1e-6)
    st = torch.tensor([kk in STATIC for kk in k.tolist()], device=dev)
    traj = torch.where(st[:, None, None, None], torch.zeros_like(traj), traj)
    M = traj.shape[1]
    pts = torch.cat([c0[:, None, None].expand(A, M, 1, 2), c0[:, None, None] + traj.cumsum(2)], 2)   # [A, M, 9, 2]
    tt = (torch.arange(T, device=dev) + 1) / 5.0                                         # in 0.5 s units, 0.2 .. 8
    i0 = tt.floor().long().clamp(max=7)
    f = (tt - i0)[None, None, :, None]
    ctr = pts[:, :, i0] * (1 - f) + pts[:, :, i0 + 1] * f                                # [A, M, T, 2]
    seg = pts[:, :, i0 + 1] - pts[:, :, i0]
    sn = seg.norm(dim=-1, keepdim=True)
    dirv = torch.where(sn > 0.25, seg / sn.clamp(min=1e-6), d0[:, None, None].expand_as(seg))
    # local window of cells around each centre
    col = torch.round((ctr[..., 0] + 32.0) / RES - 0.5).long()
    row = torch.round(ctr[..., 1] / RES - 0.5).long()
    o = torch.arange(-WIN, WIN, device=dev)
    rr = row[..., None, None] + o[:, None]                                                # [A, M, T, 2W, 1]
    cc = col[..., None, None] + o[None, :]                                                # [A, M, T, 1, 2W]
    rel_x = (-32.0 + (cc + 0.5) * RES) - ctr[..., 0, None, None]
    rel_y = ((rr + 0.5) * RES) - ctr[..., 1, None, None]
    dx, dy = dirv[..., 0, None, None], dirv[..., 1, None, None]
    u = rel_x * dx + rel_y * dy                                                           # along the length
    v = -rel_x * dy + rel_y * dx
    inside = (u.abs() <= ln[:, None, None, None, None] / 2) & (v.abs() <= wd[:, None, None, None, None] / 2)
    ok = inside & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    val = (p[:, None, None, None, None] * pi[:, :, None, None, None]) * ok.float()       # [A, M, T, 2W, 2W]
    idx = (rr.clamp(0, H - 1) * W + cc.clamp(0, W - 1)).expand_as(val)
    G = torch.zeros(A, T, H * W, device=dev)
    G.scatter_add_(2, idx.permute(0, 2, 1, 3, 4).reshape(A, T, -1), val.permute(0, 2, 1, 3, 4).reshape(A, T, -1))
    occ = 1 - torch.prod(1 - G.clamp(0, 1), 0)
    return occ.reshape(T, H, W)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--motion", default="head", choices=("head", "cv"))
    ap.add_argument("--det-mode", default="prob", choices=("prob", "hard"))
    a = ap.parse_args(argv)
    import torch
    d = Path(a.dump) / a.split
    tag = f"{a.motion}_{a.det_mode}"
    done = np.zeros(0, bool)
    for q in d.glob("detdone_*.npy"):
        done = np.load(q) if not len(done) else done | np.load(q)
    rows = np.where(done)[0]
    n = len(done)
    D = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("det_box", "det_cls", "det_traj", "det_tcls")}
    fo = d / f"mocc_{tag}.npy"
    if not fo.exists():
        np.lib.format.open_memmap(fo, "w+", np.uint8, (n, T, H, W)).flush()
    O = np.load(fo, mmap_mode="r+")
    dev = torch.device("cuda")
    t0 = time.time()
    for j, r in enumerate(rows):
        g = {k: torch.tensor(np.asarray(v[r], np.float32), device=dev) for k, v in D.items()}
        with torch.no_grad():
            occ = render(g["det_box"], g["det_cls"], g["det_traj"], g["det_tcls"], a.motion, a.det_mode)
        O[r] = np.rint(occ.cpu().numpy() * 255).astype(np.uint8)
        if j % 500 == 0:
            print(f"{j + 1}/{len(rows)} {(j + 1) / (time.time() - t0):.1f} tok/s", flush=True)
    O.flush()
    m = np.zeros(n, bool)
    m[rows] = True
    np.save(d / f"moccdone_{tag}.npy", m)
    print(f"saved {fo} ({len(rows)} rows) in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
