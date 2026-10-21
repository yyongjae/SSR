#!/usr/bin/env python
"""Can the model's OWN predicted vector map reproduce the PDM DAC label?

For every navtest token the model trajectory (and, on a subset, all 256 anchors) is turned into
0.1 s footprint corners, and a drivable-area margin is computed from three map sources:
  cache   signed distance to the GT drivable polygons of the v1 metric cache (reference; report 42 §7)
  gt_pl   GT road-contour polylines of the aux record (same ROI the map head sees)
  pred    predicted road-contour polylines of the aux record (score >= --map-score)
Polylines are open boundary pieces without a reliable orientation, so their sign comes from the
crossing parity of the segment ego-origin -> corner (the ego origin is on the road at t=0):
odd crossings = outside.  Unsigned distance is reported too.  Only time steps inside the map ROI
(x_right in [-32, 32], y_forward in [0, 32], PARA-SSR frame) count; a trajectory with no step in the
ROI gets nan.  Labels: official DAC of the trajectory (PDMS csv) and of the anchors (scorer re-run
with the metric cache, 256 per token in one batch).

usage: python pred_map_dac_margin.py --records <aux records> --traj <dump.pkl> --pdm-csv <official csv>
           --out <csv> [--anchor-tokens 500] [--workers 24]
"""
from __future__ import annotations

import argparse
import lzma
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("SSR_REPO", Path.home() / "SSR"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(REPO / "work_dirs"))

ROAD = 0
X0, X1, Y0, Y1 = -32.0, 32.0, 0.0, 32.0           # PARA-SSR map ROI (x_right, y_forward)
LEN, WID, RA2C = 5.176, 2.297, 1.461               # pacifica
DT, T = 0.1, 41
_G = {}


def _init(cache_dir, anchors_path, map_score):
    import torch
    torch.set_num_threads(1)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/pdm_scoring"), version_base="1.2"):
        cfg = compose(config_name="default_run_pdm_score")
    _G["sim"] = instantiate(cfg.simulator)
    _G["scorer"] = instantiate(cfg.scorer)
    _G["cache_paths"] = {p.parent.name: p for p in Path(cache_dir).rglob("metric_cache.pkl")}
    _G["anchors"] = np.load(anchors_path).astype(np.float64)
    _G["map_score"] = map_score


def _corners_navsim(poses8):
    """[8,3] navsim ego poses (0.5 s) -> corners [T,4,2] in the PARA-SSR map frame (x_right, y_forward)."""
    t8 = np.arange(1, 9) * 0.5
    ts = np.arange(T) * DT
    x = np.interp(ts, np.r_[0, t8], np.r_[0, poses8[:, 0]])
    y = np.interp(ts, np.r_[0, t8], np.r_[0, poses8[:, 1]])
    h = np.interp(ts, np.r_[0, t8], np.r_[0, np.unwrap(np.r_[0, poses8[:, 2]])[1:]])
    c, s = np.cos(h), np.sin(h)
    cx, cy = x + RA2C * c, y + RA2C * s                       # box centre from the rear axle
    hl, hw = LEN / 2, WID / 2
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    cx_ = cx[:, None] + local[None, :, 0] * c[:, None] - local[None, :, 1] * s[:, None]
    cy_ = cy[:, None] + local[None, :, 0] * s[:, None] + local[None, :, 1] * c[:, None]
    # navsim (x forward, y left) -> PARA-SSR map (x_right, y_forward)
    return np.stack([-cy_, cx_], axis=-1)


def _polyline_margin(corners, lines):
    """corners [N,4,2]; lines list of [20,2] arrays -> (signed_min, unsigned_min) over corners in the ROI."""
    import shapely
    from shapely.geometry import LineString, MultiLineString
    n = corners.shape[0]
    inroi = (corners[..., 0] >= X0) & (corners[..., 0] <= X1) & (corners[..., 1] >= Y0) & (corners[..., 1] <= Y1)
    inroi = inroi.all(axis=1)                                   # time step counts only if the whole box is in the ROI
    if not inroi.any():
        return np.nan, np.nan, 0
    if not lines:
        return np.nan, np.nan, int(inroi.sum())
    boundary = MultiLineString([LineString(l) for l in lines if len(l) >= 2])
    pts = corners[inroi].reshape(-1, 2)
    dist = shapely.distance(shapely.points(pts), boundary)
    origin = np.array([0.0, 0.0])
    segs = shapely.linestrings(np.stack([np.broadcast_to(origin, pts.shape), pts], axis=1))
    inter = shapely.intersection(segs, boundary)
    n_cross = np.array([0 if g.is_empty else (1 if g.geom_type == "Point" else len(getattr(g, "geoms", [g]))) for g in inter])
    outside = (n_cross % 2) == 1
    signed = np.where(outside, -dist, dist)
    return float(signed.min()), float(dist.min()), int(inroi.sum())


def _cache_margin(corners_glob_fn, mc):
    """signed distance to the metric-cache drivable union (global frame)."""
    import shapely
    from shapely.ops import unary_union
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    dm = mc.drivable_area_map
    idx = dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                      SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    if not idx:
        return None
    U = unary_union([dm._geometries[i] for i in idx])
    if U.is_empty:
        return None

    def margin(corners):                                       # [K,T,4,2] global
        flat = corners.reshape(-1, 2)
        d = shapely.distance(shapely.points(flat), U.boundary)
        inside = shapely.contains_xy(U, flat[:, 0], flat[:, 1])
        return np.where(inside, d, -d).reshape(corners.shape[:3]).min(axis=(1, 2))
    return margin


def _one(args):
    token, rec_path, traj, do_anchors = args
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score_multi_trajs, transform_trajectory
    sim, scorer = _G["sim"], _G["scorer"]
    try:
        rec = np.load(rec_path, allow_pickle=True)
        keep = rec["map_pred_scores"] >= _G["map_score"]
        pred_lines = [p for p, l in zip(rec["map_pred_points"][keep], rec["map_pred_labels"][keep]) if l == ROAD]
        gt_lines = [p for p, l in zip(rec["map_gt_points"], rec["map_gt_labels"]) if l == ROAD]
        rows = []
        cands = [("traj", np.asarray(traj, np.float64))]
        if do_anchors:
            cands += [(f"a{k}", a) for k, a in enumerate(_G["anchors"])]

        # cache reference + official anchor labels
        cache_margin = None
        off_dac = {}
        mc = None
        cp = _G["cache_paths"].get(token)
        if cp is not None:
            with lzma.open(cp, "rb") as f:
                mc = pickle.load(f)
            margin_fn = _cache_margin(None, mc)
            ego0 = mc.ego_state
            ps = sim.proposal_sampling
            states = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(p.astype(np.float32)), ego0), ps, ego0.time_point)
                               for _, p in cands])
            scorer.score_proposals(states.copy(), mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
            if margin_fn is not None:
                cm = margin_fn(scorer._ego_coords[:, :, :4, :].copy())
                cache_margin = {name: float(cm[i]) for i, (name, _) in enumerate(cands)}
            if do_anchors:
                off = pdm_score_multi_trajs(mc, [a.astype(np.float32) for a in _G["anchors"]], ps, sim, scorer)
                off_dac = {f"a{k}": float(v) for k, v in enumerate(off.drivable_area_compliance)}

        for name, poses in cands:
            corners = _corners_navsim(poses)
            ps_, pu_, n_in = _polyline_margin(corners, pred_lines)
            gs_, gu_, _ = _polyline_margin(corners, gt_lines)
            rows.append(dict(token=token, cand=name, n_in_roi=n_in, n_pred_road=len(pred_lines), n_gt_road=len(gt_lines),
                             pred_signed=ps_, pred_unsigned=pu_, gtpl_signed=gs_, gtpl_unsigned=gu_,
                             cache_signed=(cache_margin or {}).get(name, np.nan),
                             off_dac_anchor=off_dac.get(name, np.nan)))
        return rows
    except Exception as exc:  # noqa: BLE001
        import traceback
        return [dict(token=token, cand="ERROR", err=repr(exc) + traceback.format_exc()[-600:])]


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(score)
    label = np.asarray(label, bool)[m]
    score = np.asarray(score)[m]
    if m.sum() == 0 or label.all() or (~label).all():
        return float("nan"), int(m.sum())
    return float(roc_auc_score(label, score)), int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--pdm-csv", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=REPO / "data/exp/metric_cache")
    ap.add_argument("--anchors", type=Path, default=REPO / "data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map-score", type=float, default=0.3)
    ap.add_argument("--anchor-tokens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    trajs = pickle.load(open(a.traj, "rb"))["trajectories"]
    recs = {p.stem: p for p in a.records.glob("*.npz")}
    tokens = sorted(set(trajs) & set(recs))
    rng = np.random.default_rng(a.seed)
    anchor_tokens = set(rng.choice(tokens, size=min(a.anchor_tokens, len(tokens)), replace=False).tolist())
    print(f"{len(tokens)} tokens; anchors on {len(anchor_tokens)}", flush=True)
    jobs = [(t, recs[t], trajs[t], t in anchor_tokens) for t in tokens]
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(str(a.cache), str(a.anchors), a.map_score)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    err = df[df.cand == "ERROR"]
    if len(err):
        print(f"{len(err)} tokens failed; first:\n{err.iloc[0]['err']}")
        df = df[df.cand != "ERROR"].drop(columns=["err"], errors="ignore")
    df.to_csv(a.out, index=False)

    pdm = pd.read_csv(a.pdm_csv)
    pdm = pdm[pdm.token.notna()].set_index("token")
    tr = df[df.cand == "traj"].set_index("token").join(pdm[["drivable_area_compliance"]], how="inner")
    y = tr.drivable_area_compliance < 1
    print(f"\n--- model trajectory, {len(tr)} tokens, DAC fail {int(y.sum())}; mean pred road polylines {tr.n_pred_road.mean():.1f}, "
          f"GT {tr.n_gt_road.mean():.1f}; no ROI step {int((tr.n_in_roi == 0).sum())} ---")
    for col in ("cache_signed", "gtpl_signed", "gtpl_unsigned", "pred_signed", "pred_unsigned"):
        v, n = auc(-tr[col].values, y.values)
        acc = ((tr[col] <= 0) == y)[np.isfinite(tr[col])].mean() if "signed" in col else float("nan")
        print(f"  {col:14s} AUC {v:.4f} (n={n})" + (f"  acc@0 {acc:.4f}" if np.isfinite(acc) else ""))
    yy = y.values
    for col in ("gtpl_signed", "pred_signed"):
        m = np.isfinite(tr[col].values)
        fn = ((tr[col].values > 0) & yy & m).sum()
        fp = ((tr[col].values <= 0) & ~yy & m).sum()
        print(f"  {col}: DAC-fail missed (margin>0) {int(fn)}/{int((yy & m).sum())}, DAC-pass flagged (margin<=0) {int(fp)}/{int((~yy & m).sum())}")

    an = df[df.cand != "traj"]
    an = an[np.isfinite(an.off_dac_anchor)]
    if len(an):
        ya = an.off_dac_anchor < 1
        print(f"\n--- anchors, {an.token.nunique()} tokens x 256 = {len(an)} rows, official DAC fail rate {ya.mean():.3f} ---")
        for col in ("cache_signed", "gtpl_signed", "gtpl_unsigned", "pred_signed", "pred_unsigned"):
            v, n = auc(-an[col].values, ya.values)
            acc = ((an[col] <= 0) == ya)[np.isfinite(an[col])].mean() if "signed" in col else float("nan")
            print(f"  {col:14s} AUC {v:.4f} (n={n})" + (f"  acc@0 {acc:.4f}" if np.isfinite(acc) else ""))
        # per-token agreement between predicted-map margin and official anchor DAC: how often does the predicted map
        # rank the official-safe anchors above the unsafe ones (token-level AUC)
        per = []
        for t, g in an.groupby("token"):
            v, n = auc(-g.pred_signed.values, (g.off_dac_anchor < 1).values)
            if np.isfinite(v):
                per.append(v)
        per = np.array(per)
        print(f"  per-token pred_signed AUC: median {np.median(per):.3f}, p10 {np.percentile(per, 10):.3f}, tokens with AUC<0.7: {(per < 0.7).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
