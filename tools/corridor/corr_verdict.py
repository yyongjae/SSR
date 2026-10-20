#!/usr/bin/env python
"""km 10/15: per-candidate corridor verdicts of a dump (dump_corr_model.py), every candidate tracked by the torch port of
the official LQR + bicycle simulation (status-only initial state, as at inference).

  <dump>/<split>/verdict.npz:
     pred_sdf_min [N, 256]  min over 40 steps x 4 corners of the PREDICTED static SDF (in-range points; +8 if none)
     pred_occ_max [N, 256]  max over 40 steps x 15 footprint points of the PREDICTED swept occupancy probability
     gt_sdf_min / gt_occ_hit  the same with the GT corridor (navtrain splits only; oracle upper bound, never deployed)
     oor [N, 256]           fraction of footprint points beyond the raster (x > 32 m or |y| > 32 m)
   km 10/15 (4), v3:
     pred_ttc_max [N, 256]  official-TTC-style: front half of the footprint at step t shifted along the heading by
                            speed * {0.3, 0.6, 0.9} s vs the predicted occupancy at t + dt (moving ego only), max prob
     comfort [N, 256]       exact: official ego_is_comfortable on the tracked states (no learning)
     gt_occ_hit01 / gt_ttc_hit  GT at 0.1 s planes (oracle)
   the predicted occupancy may have 8 swept planes (0.5 s) or 40 (0.1 s): step -> plane accordingly.

  km 10/16 (test A): --occ motion_<tag> | max_<tag> uses the occupancy rendered from the det/motion head
  (render_motion_occ.py, mocc_<tag>.npy, 40 planes) alone or max-combined with the corridor head's; -> verdict_<occ>.npz

  CUDA_VISIBLE_DEVICES=0 $PY tools/corridor/corr_verdict.py --dump <dir> --split navtrain_val
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

K = os.environ.get("KM_ROOT", "/home/external-user/ssd/kyungmin_ck")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--subset", type=int, default=0, help="only corr_select's --n rows (rng(0) over all rows)")
    ap.add_argument("--occ", default="model", help="model | motion_<tag> | max_<tag> (tag of mocc_<tag>.npy)")
    a = ap.parse_args(argv)
    import torch
    from navsim.agents.para_ssr.modules import corridor_head as CH
    from navsim.agents.para_ssr.modules.torch_tracker import track

    d = Path(a.dump) / a.split
    done = np.zeros(0, bool)
    for p in d.glob("done_*.npy"):
        done = np.load(p) if not len(done) else done | np.load(p)
    rows = np.where(done)[0]
    if a.subset:
        rows = np.intersect1d(rows, np.random.default_rng(0).choice(np.arange(len(done)), a.subset, replace=False))
    T = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("traj", "status", "csdf", "cocc")}
    if a.occ != "model":
        T["mocc"] = np.load(d / f"mocc_{a.occ.split('_', 1)[1]}.npy", mmap_mode="r")
    has_route = (d / "croute.npy").exists()
    if has_route:                                  # km 10/16: predicted route progress (EP transport)
        T["croute"] = np.load(d / "croute.npy", mmap_mode="r")
        ANC = None
    has_gt = a.split.startswith("navtrain")
    if has_gt:
        GS = np.load(f"{K}/corridor_dt01/{a.split}/static_sdf.npy", mmap_mode="r")
        GD = np.load(f"{K}/corridor_dt01/{a.split}/dyn_occ.npy", mmap_mode="r")
        gok = np.load(f"{K}/corridor_dt01/{a.split}/ok.npy")
    n = len(done)
    out = {k: np.full((n, 256), np.nan, np.float32) for k in ("pred_sdf_min", "pred_occ_max", "oor")}
    if has_gt:
        out.update({k: np.full((n, 256), np.nan, np.float32) for k in ("gt_sdf_min", "gt_occ_hit")})
    dev = torch.device("cuda")
    cx = torch.tensor([CH.FRONT, CH.FRONT, -CH.REAR, -CH.REAR], device=dev)
    cy = torch.tensor([CH.HALF_W, -CH.HALF_W, -CH.HALF_W, CH.HALF_W], device=dev)
    fx, fy = torch.meshgrid(torch.linspace(-CH.REAR, CH.FRONT, 5, device=dev),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=dev), indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    P = T["cocc"].shape[1]
    plane = CH.occ_plane_index(P, dev)                                                      # plane of each step

    def ttc_idx(pl, dd):  # km 10/18: 49 head planes reach 4.9 s; 40 / 8 planes (and the 40-plane mocc) hold 4.0 s
        st_ = torch.arange(40, device=dev) + dd
        return CH.ttc_plane_index(P, st_) if pl is plane else pl[st_.clamp(max=39)]
    gx, gy = torch.meshgrid(torch.linspace(0.0, CH.FRONT, 4, device=dev),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=dev), indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)                                                 # front half (TTC)
    for k in ("pred_ttc_max", "comfort") + (("prog_cand", "prog_anchor") if has_route else ()):
        out[k] = np.full((n, 256), np.nan, np.float32)
    if has_gt:
        out.update({k: np.full((n, 256), np.nan, np.float32) for k in ("gt_occ_hit01", "gt_ttc_hit")})
    TP = np.arange(41, dtype=np.float64) * 0.1
    for i in range(0, len(rows), a.bs):
        r = rows[i:i + a.bs]
        b = len(r)
        traj = torch.tensor(np.asarray(T["traj"][r], np.float32), device=dev)
        stat = torch.tensor(np.asarray(T["status"][r], np.float32), device=dev)
        init = torch.cat([stat, torch.zeros(b, 3, device=dev)], -1).repeat_interleave(256, 0)
        with torch.no_grad():
            full = track(traj.reshape(-1, 8, 3), init, full=True).reshape(b, 256, 41, 7)
            st = full[:, :, 1:]
            x, y, h = st[..., 0], st[..., 1], st[..., 2]
            spd = st[..., 3]                                                                   # [b, 256, 40]
            c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
            pc = torch.stack([x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy], -1)   # [b,256,40,4,2]
            pf = torch.stack([x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy], -1)   # [b,256,40,15,2]

            def smin(field):
                v, inr = CH._sample(field[:, None], pc.reshape(b, 1, -1, 2))
                v = torch.where(inr, v, torch.full_like(v, CH.CLIP))
                return v.reshape(b, 256, -1).amin(-1)

            def omax(planes, pl=None):                                                         # planes [b, P, H, W]
                q = pf.permute(0, 2, 1, 3, 4)                                                 # [b, 40, 256, 15, 2]
                f = planes[:, plane if pl is None else pl]                                     # [b, 40, H, W]
                v, inr = CH._sample(f, q.reshape(b, 40, -1, 2))
                v = torch.where(inr, v, torch.zeros_like(v))
                return v.reshape(b, 40, 256, 15).amax(-1).amax(1), inr.reshape(b, 40, 256, 15)

            def tmax(planes, pl):                                                              # TTC projection
                best = torch.zeros(b, 256, device=dev)
                mov = (spd >= 5e-3).float()
                for dd in (3, 6, 9):
                    sh = (spd * dd * 0.1)[..., None]
                    q = torch.stack([x[..., None] + c * (gx + sh) - s * gy, y[..., None] + s * (gx + sh) + c * gy], -1)
                    f = planes[:, ttc_idx(pl, dd)]
                    v, inr = CH._sample(f, q.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2))
                    v = torch.where(inr, v, torch.zeros_like(v)).reshape(b, 40, 256, 12).amax(-1).permute(0, 2, 1)
                    best = torch.maximum(best, (v * mov).amax(-1))
                return best

            out["pred_sdf_min"][r] = smin(torch.tensor(np.asarray(T["csdf"][r], np.float32), device=dev)).cpu().numpy()
            src = []
            if not a.occ.startswith("motion"):
                src.append((torch.tensor(np.asarray(T["cocc"][r], np.float32) / 255.0, device=dev), plane))
            if a.occ != "model":
                src.append((torch.tensor(np.asarray(T["mocc"][r], np.float32) / 255.0, device=dev),
                            torch.arange(40, device=dev)))
            om, inr = omax(*src[0])
            tm = tmax(*src[0])
            for po, pl in src[1:]:
                om, tm = torch.maximum(om, omax(po, pl)[0]), torch.maximum(tm, tmax(po, pl))
            out["pred_occ_max"][r] = om.cpu().numpy()
            out["pred_ttc_max"][r] = tm.cpu().numpy()
            fs = full.reshape(b * 256, 41, 7).double().cpu().numpy()
            S = np.zeros((b * 256, 41, StateIndex.size()), np.float64)
            for j, ix in enumerate((StateIndex.X, StateIndex.Y, StateIndex.HEADING, StateIndex.VELOCITY_X,
                                    StateIndex.VELOCITY_Y, StateIndex.ACCELERATION_X, StateIndex.ACCELERATION_Y)):
                S[..., ix] = fs[..., j]
            out["comfort"][r] = ego_is_comfortable(S, TP).all(-1).reshape(b, 256)
            if has_route:
                # progress (m) along the predicted route field at the tracked end pose: candidate (anchor + offset)
                # and its raw anchor (tracked the same way) -> EP transport ratio in corr_select
                if ANC is None:
                    from tools.corridor.corr_select import ANCHOR_FILE
                    ANC = torch.tensor(np.load(ANCHOR_FILE)[..., :3], dtype=torch.float32, device=dev)
                rf = torch.tensor(np.asarray(T["croute"][r], np.float32), device=dev)[:, None]          # [b,1,h,w]
                rf = torch.nn.functional.interpolate(rf, scale_factor=2, mode="bilinear", align_corners=False)
                ea = track(ANC.repeat(b, 1, 1), init).reshape(b, 256, 41, 3)[:, :, -1, :2]
                pc_, _ = CH._sample(rf, st[:, :, -1, :2].reshape(b, 1, 256, 2))
                pa_, _ = CH._sample(rf, ea.reshape(b, 1, 256, 2))
                out["prog_cand"][r] = pc_.reshape(b, 256).cpu().numpy()
                out["prog_anchor"][r] = pa_.reshape(b, 256).cpu().numpy()
            out["oor"][r] = (1 - inr.float().mean((1, 3))).cpu().numpy()
            if has_gt:
                g = gok[r]
                out["gt_sdf_min"][r] = np.where(g[:, None], smin(torch.tensor(np.asarray(GS[r], np.float32), device=dev)
                                                                  ).cpu().numpy(), np.nan)
                bits = np.unpackbits(np.asarray(GD[r]), axis=-1).reshape(b, 8, 5, 100, 200).max(2)      # swept 8
                gm, _ = omax(torch.tensor(bits.astype(np.float32), device=dev), torch.arange(40, device=dev) // 5)
                out["gt_occ_hit"][r] = np.where(g[:, None], gm.cpu().numpy(), np.nan)
                g01 = torch.tensor(np.unpackbits(np.asarray(GD[r]), axis=-1).astype(np.float32), device=dev)
                st40 = torch.arange(40, device=dev)
                gm, _ = omax(g01, st40)
                out["gt_occ_hit01"][r] = np.where(g[:, None], gm.cpu().numpy(), np.nan)
                out["gt_ttc_hit"][r] = np.where(g[:, None], tmax(g01, st40).cpu().numpy(), np.nan)
        if i % (a.bs * 100) == 0:
            print(f"{i + b}/{len(rows)}", flush=True)
    vf = "verdict.npz" if a.occ == "model" else f"verdict_{a.occ}.npz"
    np.savez(d / vf, rows=rows, **out)
    print(f"saved {d}/{vf} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
