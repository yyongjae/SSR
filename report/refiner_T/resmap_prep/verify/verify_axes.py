"""Verifier: ReSMap seg/bev axis convention vs independent metric-cache GT (gt_extract.py output).
Reads the ReSMap shards directly (own mmap + index.json), NOT via resmap_cache.py.  CPU only.
usage: verify_axes.py <gt_dir> <out_dir>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R = Path("/home/external-user/datasets/teacher_cache/resmap")
IDX = {"navtrain": json.load(open(R / "index.json")), "navtest": json.load(open(R / "navtest/index.json"))}
ROOT = {"navtrain": R, "navtest": R / "navtest"}
_mm = {}


def raw(subset, token, field):
    shard, row = IDX[subset][token]
    k = (subset, field, shard)
    if k not in _mm:
        _mm[k] = np.load(ROOT[subset] / field / f"{shard}.npy", mmap_mode="r")
    return np.asarray(_mm[k][row], np.float32)


# hypotheses on RAW [.., a (lateral, 200|100), b (forward, 100|50)] -> S [.., rows = forward, cols = lateral(col0=left)]
HYP = {
    "chosen(transpose)": lambda x: np.swapaxes(x, -1, -2),
    "lateral_mirror": lambda x: np.swapaxes(x, -1, -2)[..., :, ::-1],
    "forward_flip": lambda x: np.swapaxes(x, -1, -2)[..., ::-1, :],
    "both_flip": lambda x: np.swapaxes(x, -1, -2)[..., ::-1, ::-1],
}
PI, PJ = 12, 6          # GT pad cells (rows, cols)
C = 0.32
xs = (np.arange(100) + 0.5) * C
ys = 32 - (np.arange(200) + 0.5) * C
CELL_XY = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)


def load_gt(t):
    z = np.load(GT / f"{t}.npz")
    sh = (124, 212)
    d = {k: np.unpackbits(z[k])[: sh[0] * sh[1]].reshape(sh).astype(bool) for k in ("dac", "lane", "isec")}
    d.update(cl=z["cl"].astype(np.float64), speed=float(z["speed"]), subset=str(z["subset"]))
    return d


def auc(score, lab):
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    npos = lab.sum(); nneg = len(lab) - npos
    return float((r[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def seg_metrics(S, g, di=0, dj=0):
    """S: [4, 100, 200] seg logits in S orientation; g: GT dict; shift (di, dj) in cells of GT relative to teacher."""
    dac = g["dac"][PI + di: PI + di + 100, PJ + dj: PJ + dj + 200]
    road = S[0] > 0
    inter, uni = (road & dac).sum(), (road | dac).sum()
    out = dict(inter=int(inter), union=int(uni), iou=inter / max(uni, 1))
    cl = g["cl"]
    cl = cl[(cl[:, 0] > 0) & (cl[:, 0] < 32) & (np.abs(cl[:, 1]) < 32)]
    if di == 0 and dj == 0 and len(cl) > 10:
        r = np.clip((cl[:, 0] / C - 0.5).round().astype(int), 0, 99)
        c = np.clip(((32 - cl[:, 1]) / C - 0.5).round().astype(int), 0, 199)
        lg = S[2, r, c]
        out.update(cl_logit=float(lg.mean()), cl_hit=float((lg > 0).mean()))
        pred = CELL_XY[(S[2] > 0).reshape(-1)]
        if len(pred):
            dd, _ = cKDTree(pred).query(cl)
            out.update(cl_gt2pred_med=float(np.median(dd)))
    return out


# ------------------------------------------------------------------------------------------------ load everything
jobs = pd.read_parquet(GT / "_jobs.parquet")
rows = []
SEG, GTS = {}, {}
for t, sub in zip(jobs.token, jobs.subset):
    g = load_gt(t)
    seg = raw(sub, t, "seg")              # (4, 200, 100)
    SEG[t], GTS[t] = seg, g
    cl = g["cl"]
    roi = cl[(cl[:, 0] > 0) & (cl[:, 0] < 32) & (np.abs(cl[:, 1]) < 32)]
    # curve: heading of the route centerline at its last in-ROI point vs near the ego
    head = np.nan
    if len(roi) > 20:
        a0 = np.arctan2(*(roi[5] - roi[0])[::-1]); a1 = np.arctan2(*(roi[-1] - roi[-6])[::-1])
        head = float(np.angle(np.exp(1j * (a1 - a0))))
    isec = g["isec"][PI:PI + 100, PJ:PJ + 200].mean()
    rec = dict(token=t, subset=sub, speed=g["speed"], cl_head=head, isec_frac=float(isec))
    for h, f in HYP.items():
        m = seg_metrics(f(seg), g)
        rec.update({f"{h}:{k}": v for k, v in m.items()})
    rows.append(rec)
df = pd.DataFrame(rows)
df["kind"] = np.where(df.cl_head > 0.4, "left", np.where(df.cl_head < -0.4, "right",
                      np.where(df.isec_frac > 0.08, "intersection", "other")))
df.to_parquet(OUT / "per_token_seg.parquet")


def summ(d):
    o = {}
    for h in HYP:
        o[h] = dict(road_iou_pooled=float(d[f"{h}:inter"].sum() / d[f"{h}:union"].sum()),
                    road_iou_token_median=float(d[f"{h}:iou"].median()),
                    cl_logit_mean=float(d[f"{h}:cl_logit"].mean()), cl_hit=float(d[f"{h}:cl_hit"].mean()),
                    cl_gt2pred_median_m=float(d[f"{h}:cl_gt2pred_med"].median()))
    return o


res = {"n": {s: int((df.subset == s).sum()) for s in ("navtrain", "navtest")},
       "kinds": df.groupby(["subset", "kind"]).size().to_dict()}
res["kinds"] = {f"{a}/{b}": int(v) for (a, b), v in res["kinds"].items()}
for s in ("navtrain", "navtest"):
    d = df[df.subset == s]
    res[s] = {"all": summ(d)}
    for k in ("left", "right", "intersection"):
        if (d.kind == k).sum():
            res[s][k] = summ(d[d.kind == k])
    # pooled AUC of the road logit (subsample cells)
    rng = np.random.default_rng(0)
    au = {}
    for h, f in HYP.items():
        sc, lb = [], []
        for t in d.token:
            S = f(SEG[t])[0].reshape(-1); G = GTS[t]["dac"][PI:PI + 100, PJ:PJ + 200].reshape(-1)
            k = rng.choice(S.size, 500, replace=False)
            sc.append(S[k]); lb.append(G[k])
        au[h] = auc(np.concatenate(sc), np.concatenate(lb))
    res[s]["road_auc"] = au
    # per-token: fraction of tokens where the chosen hypothesis has the highest road IoU
    ious = np.stack([d[f"{h}:iou"].values for h in HYP], 1)
    res[s]["chosen_best_frac"] = float((ious.argmax(1) == 0).mean())
    res[s]["chosen_iou_p05_min"] = [float(np.percentile(ious[:, 0], 5)), float(ious[:, 0].min())]

# ------------------------------------------------------------------------------------------------ origin shift scan
for s in ("navtrain", "navtest"):
    toks = df[df.subset == s].token.values
    grid = {}
    best = []
    for t in toks:
        S = HYP["chosen(transpose)"](SEG[t])
        g = GTS[t]
        tab = np.zeros((25, 13))
        for a, di in enumerate(range(-12, 13)):
            for b, dj in enumerate(range(-6, 7)):
                m = seg_metrics(S, g, di, dj)
                tab[a, b] = m["iou"]
                grid.setdefault((di, dj), [0, 0])
                grid[(di, dj)][0] += m["inter"]; grid[(di, dj)][1] += m["union"]
        a, b = np.unravel_index(tab.argmax(), tab.shape)
        best.append((t, g["speed"], (a - 12) * C, (b - 6) * C, tab[12, 6], tab.max()))
    pooled = {k: v[0] / v[1] for k, v in grid.items()}
    kb = max(pooled, key=pooled.get)
    bdf = pd.DataFrame(best, columns=["token", "speed", "best_dx", "best_dy_right", "iou0", "iou_best"])
    mv = bdf[bdf.speed > 5]
    res[s]["shift_scan"] = dict(
        pooled_best_dx_m=kb[0] * C, pooled_best_dy_m=-kb[1] * C, pooled_iou_at_0=pooled[(0, 0)],
        pooled_iou_at_best=pooled[kb],
        pooled_iou_dx=[(di * C, round(pooled[(di, 0)], 4)) for di in range(-6, 7, 2)],
        pooled_iou_dy=[(-dj * C, round(pooled[(0, dj)], 4)) for dj in range(-4, 5, 2)],
        moving_n=int(len(mv)), moving_best_dx_median=float(mv.best_dx.median()),
        moving_best_dx_absgt1m_frac=float((mv.best_dx.abs() > 1.0).mean()),
        corr_speed_bestdx=float(np.corrcoef(mv.speed, mv.best_dx)[0, 1]) if len(mv) > 3 else None,
        tokens_iou0_lt_0p5=bdf[bdf.iou0 < 0.5][["token", "speed", "best_dx", "best_dy_right", "iou0", "iou_best"]]
        .round(3).values.tolist())
    bdf.to_parquet(OUT / f"shift_best_{s}.parquet")

json.dump(res, open(OUT / "seg_axes_verify.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
