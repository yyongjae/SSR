"""Measure (do not assume) how the ReSMap map-teacher cache maps onto the refiner's S grid and N frame.

Ground truth (stage-T data, independent of the teacher): the drivable-area SDF (packed/<split>/sdf.npy, E grid, sdf.py)
and the route centerline (packed/<split>/cl_xy.npy, metric-cache PDMPath vertices in N).  Both are in the N frame
(NAVSIM ego at t0, REAR-AXLE origin, x forward, y left).

Checks (raw memmaps are read here, NOT through resmap_cache.py, so the loader is tested against an independent reader)
  1. seg (4, 200, 100) raw logits: 8 hypotheses (which array axis is forward x swap, flip of axis a, flip of axis b)
     -> metric cell centres -> road logit vs SDF inside (IoU of road>0 vs sdf>0, AUC, Pearson r with clip(sdf, +-3))
     and centerline logit at the GT route-centerline points (hit rate logit>0, mean logit, mean logit by distance
     band of the cell centre to the GT centerline).
  2. vectors (100, 20, 2) in [0, 1]: 8 hypotheses (swap u/v, flip u, flip v) -> metres -> road-class points: median
     |sdf|; centerline-class points: GT-route-centerline -> nearest predicted centerline point (one-sided chamfer).
  3. forward / lateral origin: with the winning hypotheses, scan a shift (dx, dy) added to the teacher's metric
     coordinates (teacher point p in its own frame is at p + (dx, dy) in N) and report the arg-optimum of each metric.
  4. bev (256, 100, 50): (a) native layout: ridge probe bev -> seg road/centerline logit (2x2 mean-pooled to 100x50)
     with the target flipped along neither / each / both axes (held-out R^2); (b) geometric: ridge probe bev ->
     sdf>0 under the 8 hypotheses for the bev array (held-out AUC).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/validate_resmap_axes.py \
      --n-random 200 --n-curve 100 --out report/refiner_T/resmap_axes.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner.sdf import sample_sdf_np  # noqa: E402

RESMAP = Path("/home/external-user/datasets/teacher_cache/resmap")
DATA_ROOT = Path("/home/external-user/ssd/yongjae_refiner")
X_ROI, Y_ROI = 32.0, 32.0            # forward 0..32, lateral +-32 (README / meta pc_range)


# ----------------------------------------------------------------------------------------------- hypotheses
def cell_xy(shape_ab, swap, flip_a, flip_b):
    """Metric centres [A, B, 2] (x forward, y left) of an (A, B) array under a hypothesis.
    swap=0: axis a = lateral, axis b = forward.  swap=1: axis a = forward, axis b = lateral.
    Unflipped: forward index 0 = x 0 (nearest the ego); lateral index 0 = y +32 (LEFT)."""
    A, B = shape_ab
    na, nb = (np.arange(A) + 0.5) / A, (np.arange(B) + 0.5) / B        # fractions in (0, 1)
    fa = 1 - na if flip_a else na
    fb = 1 - nb if flip_b else nb
    if swap == 0:
        y = Y_ROI - 2 * Y_ROI * fa                       # a -> lateral, index 0 = left
        x = X_ROI * fb                                   # b -> forward
        X, Y = np.meshgrid(x, y, indexing="xy")          # [A, B]
    else:
        x = X_ROI * fa
        y = Y_ROI - 2 * Y_ROI * fb
        Y, X = np.meshgrid(y, x, indexing="xy")          # [A, B]
    return np.stack([X, Y], -1)


HYP = {f"swap{s}_flipA{a}_flipB{b}": (s, a, b) for s, a, b in itertools.product((0, 1), repeat=3)}


def vec_xy(v, swap, flip_u, flip_v):
    """vectors [..., 2] in [0, 1] -> metres.  swap=0: (u, v) = (x/32, (y+32)/64 ... per flips); swap=1: roles exchanged.
    Unflipped: forward fraction f -> x = 32 f; lateral fraction g -> y = -32 + 64 g (g = 0 is the RIGHT edge)."""
    a, b = (v[..., 1], v[..., 0]) if swap else (v[..., 0], v[..., 1])
    a = 1 - a if flip_u else a
    b = 1 - b if flip_v else b
    return np.stack([X_ROI * a, -Y_ROI + 2 * Y_ROI * b], -1)


# ----------------------------------------------------------------------------------------------- data
class Raw:
    """Independent raw reader (memmap per shard)."""

    def __init__(self, root=RESMAP):
        self.root = Path(root)
        self.index = json.load(open(self.root / "index.json"))
        self.mm = {}

    def get(self, tok, field):
        sh, row = self.index[tok]
        k = (field, sh)
        if k not in self.mm:
            self.mm[k] = np.load(self.root / field / f"{sh}.npy", mmap_mode="r")
        return np.asarray(self.mm[k][row])


def auc(score, label):
    from sklearn.metrics import roc_auc_score

    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    return float(roc_auc_score(label, score))


def densify(cl):
    """Polyline vertices [n, 2] -> points every <= 0.1 m."""
    out = [cl[:1]]
    for p, q in zip(cl[:-1], cl[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(q - p) / 0.1)))
        out.append(p + (q - p) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


def pick_tokens(split, n_random, n_curve, seed):
    import yaml

    tl = set(yaml.safe_load(open(REPO / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"))
             ["train_logs"])
    pdir = DATA_ROOT / "packed" / split
    idx = pd.read_parquet(pdir / "index.parquet")
    done = np.load(pdir / "done.npy", mmap_mode="r")
    ok = (np.asarray(done[:, 4]) == 1) & (np.asarray(done[:, 5]) == 1) & (np.asarray(done[:, 0]) == 1)
    if split in ("train", "dev"):
        ok &= idx.log.isin(tl).values                      # ReSMap root = train_logs only
    h = np.load(pdir / "human_traj.npy", mmap_mode="r")
    yaw4 = np.abs(np.asarray(h[:, -1, 2]))
    rng = np.random.default_rng(seed)
    curve = np.flatnonzero(ok & (yaw4 > 0.5))
    straight = np.flatnonzero(ok & ~(yaw4 > 0.5))
    c = rng.choice(curve, min(n_curve, len(curve)), replace=False)
    r = rng.choice(straight, n_random, replace=False)
    rows = np.concatenate([r, c])
    return idx.iloc[rows].assign(row=rows, curve=np.r_[np.zeros(len(r), bool), np.ones(len(c), bool)],
                                 yaw4=yaw4[rows]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- checks
def seg_metrics(seg, sdf, cl_pts, xy, shift=(0.0, 0.0)):
    """seg [4, A, B] logits under metric centres xy [A, B, 2] (+shift) vs GT."""
    P = xy + np.asarray(shift)
    s, ok = sample_sdf_np(sdf, P)
    road = seg[0][ok].astype(np.float64)
    inside = s[ok] > 0
    pr = road > 0
    inter, uni = (pr & inside).sum(), (pr | inside).sum()
    out = dict(inter=int(inter), union=int(uni), road=road, inside=inside, sdfc=np.clip(s[ok], -3, 3))
    # centerline: nearest cell of each GT point (inverse by KD-tree on the cell centres)
    from scipy.spatial import cKDTree

    tree = cKDTree(P.reshape(-1, 2))
    d, j = tree.query(cl_pts)
    cl_logit = seg[2].reshape(-1)[j].astype(np.float64)
    keep = d < 0.5                                         # GT point inside the ROI
    out["cl_hit"] = cl_logit[keep]
    # distance of every cell centre to the GT centerline -> band means of the centerline logit
    t2 = cKDTree(cl_pts)
    dc, _ = t2.query(P.reshape(-1, 2), distance_upper_bound=50)
    out["cl_band"] = (seg[2].reshape(-1).astype(np.float64), dc)
    return out


BANDS = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]


def summarise_seg(recs):
    I = sum(r["inter"] for r in recs)
    U = sum(r["union"] for r in recs)
    road = np.concatenate([r["road"] for r in recs])
    ins = np.concatenate([r["inside"] for r in recs])
    sc = np.concatenate([r["sdfc"] for r in recs])
    hit = np.concatenate([r["cl_hit"] for r in recs])
    lg = np.concatenate([r["cl_band"][0] for r in recs])
    dc = np.concatenate([r["cl_band"][1] for r in recs])
    tok_iou = [r["inter"] / max(r["union"], 1) for r in recs]
    band = {f"{a}-{b}m": float(lg[(dc >= a) & (dc < b)].mean()) if ((dc >= a) & (dc < b)).any() else None
            for a, b in BANDS}
    return dict(iou_pooled=I / max(U, 1), iou_tok_median=float(np.median(tok_iou)), road_auc=auc(road, ins),
                road_r=float(np.corrcoef(road, sc)[0, 1]), cl_hit_rate=float((hit > 0).mean()),
                cl_mean_logit_at_gt=float(hit.mean()), cl_mean_logit_all=float(lg.mean()),
                cl_logit_by_dist=band, n_cl_pts=int(len(hit)))


def vec_metrics(vec, sc, lab, sdf, cl_pts, hyp, shift=(0.0, 0.0), thr=0.5):
    """road-class points: |sdf|; centerline: GT -> nearest predicted centerline point."""
    from scipy.spatial import cKDTree

    xy = vec_xy(vec.astype(np.float64), *hyp) + np.asarray(shift)
    out = {}
    road = xy[(lab == 0) & (sc > thr)].reshape(-1, 2)
    if len(road):
        s, ok = sample_sdf_np(sdf, road)
        out["road_abs_sdf"] = np.abs(s[ok])
    cl = xy[(lab == 2) & (sc > thr)].reshape(-1, 2)
    if len(cl):
        dense = np.concatenate([densify(p) for p in xy[(lab == 2) & (sc > thr)]])
        g = cl_pts[(cl_pts[:, 0] > 2) & (cl_pts[:, 0] < 30) & (np.abs(cl_pts[:, 1]) < 30)]
        if len(g):
            d, _ = cKDTree(dense).query(g)
            out["cl_gt2pred"] = d
    return out


def summ_vec(recs):
    a = [r["road_abs_sdf"] for r in recs if "road_abs_sdf" in r]
    c = [r["cl_gt2pred"] for r in recs if "cl_gt2pred" in r]
    a = np.concatenate(a) if a else np.zeros(0)
    c = np.concatenate(c) if c else np.zeros(0)
    tok_c = [float(np.median(r["cl_gt2pred"])) for r in recs if "cl_gt2pred" in r]
    return dict(road_abs_sdf_median=float(np.median(a)) if len(a) else None,
                road_frac_within_0p5m=float((a < 0.5).mean()) if len(a) else None,
                cl_gt2pred_median=float(np.median(c)) if len(c) else None,
                cl_gt2pred_mean=float(np.mean(np.minimum(c, 10))) if len(c) else None,
                cl_tok_median_of_medians=float(np.median(tok_c)) if tok_c else None,
                n_road_pts=int(len(a)), n_cl_pts=int(len(c)))


def ridge_fit(X, Y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ W


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-curve", type=int, default=100)
    ap.add_argument("--n-probe", type=int, default=120, help="tokens for the bev probes (half fit, half eval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resmap-root", default=None, help="default: RESMAP (navtrain) or RESMAP/navtest for --split navtest")
    ap.add_argument("--out", default=str(REPO / "report/refiner_T/resmap_axes.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    toks = pick_tokens(a.split, a.n_random, a.n_curve, a.seed)
    pdir = DATA_ROOT / "packed" / a.split
    SDF = np.load(pdir / "sdf.npy", mmap_mode="r")
    CLX = np.load(pdir / "cl_xy.npy", mmap_mode="r")
    CLN = np.load(pdir / "cl_n.npy", mmap_mode="r")
    raw = Raw(a.resmap_root or (RESMAP / "navtest" if a.split == "navtest" else RESMAP))
    data = []
    for r in toks.itertuples():
        n = min(int(CLN[r.row]), CLX.shape[1])
        cl = densify(np.asarray(CLX[r.row][:n], np.float64))
        cl = cl[(cl[:, 0] >= 0) & (cl[:, 0] <= X_ROI) & (np.abs(cl[:, 1]) <= Y_ROI)]
        data.append(dict(token=r.token, curve=bool(r.curve), sdf=np.asarray(SDF[r.row], np.float32), cl=cl,
                         seg=raw.get(r.token, "seg").astype(np.float32), vec=raw.get(r.token, "vectors"),
                         sc=raw.get(r.token, "scores").astype(np.float32), lab=raw.get(r.token, "labels")))
    print(f"loaded {len(data)} tokens ({toks.curve.sum()} curves, {toks.log.nunique()} logs) {time.time()-t0:.0f}s",
          flush=True)
    res = dict(tokens=dict(n=len(data), n_curve=int(toks.curve.sum()), n_logs=int(toks.log.nunique()),
                           split=a.split, seed=a.seed, resmap_root=str(raw.root), list=toks.token.tolist()))

    # 1. seg 8 hypotheses
    seg_shape = data[0]["seg"].shape[1:]
    seg_tab = {}
    for name, h in HYP.items():
        xy = cell_xy(seg_shape, *h)
        recs = [seg_metrics(d["seg"], d["sdf"], d["cl"], xy) for d in data]
        seg_tab[name] = summarise_seg(recs)
        seg_tab[name]["curves_only"] = {k: v for k, v in summarise_seg([r for r, d in zip(recs, data) if d["curve"]]).items()
                                        if k in ("iou_pooled", "road_auc", "cl_hit_rate")}
        print("seg", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in seg_tab[name].items()
                            if k in ("iou_pooled", "road_auc", "road_r", "cl_hit_rate", "cl_mean_logit_at_gt")},
              flush=True)
    res["seg_hypotheses"] = seg_tab
    best_seg = max(seg_tab, key=lambda k: seg_tab[k]["iou_pooled"])
    res["seg_best"] = best_seg

    # 2. vectors 8 hypotheses
    vec_tab = {}
    for name, h in HYP.items():
        recs = [vec_metrics(d["vec"], d["sc"], d["lab"], d["sdf"], d["cl"], h) for d in data]
        vec_tab[name] = summ_vec(recs)
        print("vec", name, vec_tab[name], flush=True)
    res["vec_hypotheses"] = vec_tab
    best_vec = min(vec_tab, key=lambda k: vec_tab[k]["cl_gt2pred_median"])
    res["vec_best"] = best_vec

    # 3. origin shift scans with the winners
    xy = cell_xy(seg_shape, *HYP[best_seg])
    scan = {}
    for axis, rng_ in (("dx", np.round(np.arange(-3.0, 3.001, 0.1), 2)), ("dy", np.round(np.arange(-1.5, 1.501, 0.1), 2))):
        rows = []
        for v in rng_:
            sh = (v, 0.0) if axis == "dx" else (0.0, v)
            recs = [seg_metrics(d["seg"], d["sdf"], d["cl"], xy, sh) for d in data]
            s = summarise_seg(recs)
            vr = summ_vec([vec_metrics(d["vec"], d["sc"], d["lab"], d["sdf"], d["cl"], HYP[best_vec], sh) for d in data])
            rows.append(dict(shift=float(v), seg_iou=s["iou_pooled"], seg_road_auc=s["road_auc"],
                             seg_cl_mean_logit_at_gt=s["cl_mean_logit_at_gt"],
                             vec_road_abs_sdf_median=vr["road_abs_sdf_median"], vec_cl_gt2pred_mean=vr["cl_gt2pred_mean"],
                             vec_cl_gt2pred_median=vr["cl_gt2pred_median"]))
        df = pd.DataFrame(rows)
        scan[axis] = dict(table=rows, argopt=dict(
            seg_iou=float(df["shift"][df.seg_iou.idxmax()]), seg_road_auc=float(df["shift"][df.seg_road_auc.idxmax()]),
            seg_cl_mean_logit_at_gt=float(df["shift"][df.seg_cl_mean_logit_at_gt.idxmax()]),
            vec_road_abs_sdf_median=float(df["shift"][df.vec_road_abs_sdf_median.idxmin()]),
            vec_cl_gt2pred_mean=float(df["shift"][df.vec_cl_gt2pred_mean.idxmin()])))
        print("scan", axis, scan[axis]["argopt"], flush=True)
    res["origin_scan"] = scan

    # 4. bev probes
    rng = np.random.default_rng(a.seed + 1)
    pr = rng.choice(len(data), min(a.n_probe, len(data)), replace=False)
    fit_i, ev_i = pr[: len(pr) // 2], pr[len(pr) // 2:]
    bev = {i: raw.get(data[i]["token"], "bev").astype(np.float32) for i in pr}
    bshape = bev[pr[0]].shape[1:]
    # 4a native layout vs pooled seg (road ch 0, centerline ch 2)
    def pooled(s):
        C, A, B = s.shape
        return s.reshape(C, A // 2, 2, B // 2, 2).mean((2, 4))
    sub = rng.choice(bshape[0] * bshape[1], 1500, replace=False)
    native = {}
    for fl in ("none", "flipA", "flipB", "both"):
        def tgt(s, fl=fl):
            p = pooled(s)[[0, 2]]
            if fl in ("flipA", "both"):
                p = p[:, ::-1]
            if fl in ("flipB", "both"):
                p = p[:, :, ::-1]
            return p.reshape(2, -1).T
        Xf = np.concatenate([bev[i].reshape(256, -1).T[sub] for i in fit_i])
        Yf = np.concatenate([tgt(data[i]["seg"])[sub] for i in fit_i])
        f = ridge_fit(Xf, Yf)
        Xe = np.concatenate([bev[i].reshape(256, -1).T for i in ev_i])
        Ye = np.concatenate([tgt(data[i]["seg"]) for i in ev_i])
        P = f(Xe)
        native[fl] = dict(r2_road=r2(Ye[:, 0], P[:, 0]), r2_centerline=r2(Ye[:, 1], P[:, 1]))
        print("bev-native", fl, native[fl], flush=True)
    res["bev_native_probe"] = native
    # 4b geometric: bev -> sdf>0 under 8 hypotheses
    geo = {}
    for name, h in HYP.items():
        xyb = cell_xy(bshape, *h)
        def lab_of(i):
            s, ok = sample_sdf_np(data[i]["sdf"], xyb)
            return (s > 0).reshape(-1), ok.reshape(-1)
        Xf, Yf = [], []
        for i in fit_i:
            y, ok = lab_of(i)
            X = bev[i].reshape(256, -1).T
            Xf.append(X[sub][ok[sub]])
            Yf.append(y[sub][ok[sub]].astype(np.float64))
        f = ridge_fit(np.concatenate(Xf), np.concatenate(Yf)[:, None])
        sc_, lb = [], []
        for i in ev_i:
            y, ok = lab_of(i)
            sc_.append(f(bev[i].reshape(256, -1).T[ok])[:, 0])
            lb.append(y[ok])
        geo[name] = dict(auc=auc(np.concatenate(sc_), np.concatenate(lb)))
        print("bev-geo", name, geo[name], flush=True)
    res["bev_geo_probe"] = geo
    res["bev_geo_best"] = max(geo, key=lambda k: geo[k]["auc"])
    res["sec"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", a.out, "best seg", best_seg, "best vec", best_vec, "best bev", res["bev_geo_best"], flush=True)


if __name__ == "__main__":
    main()
