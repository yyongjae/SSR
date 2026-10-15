"""Verifier: ReSMap bev (256, 100 lateral, 50 forward) uses the same convention as seg.
Per-cell ridge probe (1x1, 256 ch + bias) bev -> GT DAC drivable (metric-cache GT, 2x2-pooled to 0.64 m) under the
chosen transform vs lateral mirror vs forward flip vs both; fit on 70 navtrain tokens, test on 30 held-out navtrain +
30 navtest tokens.  Also bev -> seg road/centerline logit (seg fixed in the chosen transform) and the BEVFusion S grid
(data.TeacherCache path, teacher_to_s_grid) -> same GT, to check the two arms land on the same S grid.
usage: verify_bev_probe.py <gt_dir> <out_dir>"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

GT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = Path("/home/external-user/datasets/teacher_cache/resmap")
IDX = {"navtrain": json.load(open(R / "index.json")), "navtest": json.load(open(R / "navtest/index.json"))}
ROOT = {"navtrain": R, "navtest": R / "navtest"}
BF = {"navtrain": Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100"),
      "navtest": Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100")}


def raw(sub, t, field):
    shard, row = IDX[sub][t]
    return np.asarray(np.load(ROOT[sub] / field / f"{shard}.npy", mmap_mode="r")[row], np.float32)


def bevfusion_s(sub, t):
    with np.load(BF[sub] / "samples" / t[:2] / f"{t}.npz") as z:
        x = z["bev_feature"].astype(np.float32)        # (256, 50 x_fwd, 100 y_left ascending)
    return x[:, :, ::-1]                               # S grid (col 0 = left), as adapters.teacher_to_s_grid


HYP = {"chosen": lambda x: np.swapaxes(x, -1, -2), "lateral_mirror": lambda x: np.swapaxes(x, -1, -2)[..., ::-1],
       "forward_flip": lambda x: np.swapaxes(x, -1, -2)[..., ::-1, :],
       "both_flip": lambda x: np.swapaxes(x, -1, -2)[..., ::-1, ::-1]}


def gt_target(t):
    z = np.load(GT / f"{t}.npz")
    dac = np.unpackbits(z["dac"])[: 124 * 212].reshape(124, 212).astype(np.float32)[12:112, 6:206]
    return dac.reshape(50, 2, 100, 2).mean((1, 3)) > 0.5


def pool(x):
    return x.reshape(x.shape[0], 50, 2, 100, 2).mean((2, 4))


def auc(score, lab):
    o = np.argsort(score, kind="mergesort"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    p = lab.sum(); n = len(lab) - p
    return float((r[lab].sum() - p * (p + 1) / 2) / (p * n))


jobs = pd.read_parquet(GT / "_jobs.parquet")
tr = jobs[jobs.subset == "navtrain"].token.values[:100]
nt = jobs[jobs.subset == "navtest"].token.values[:30]
sets = {"fit": [("navtrain", t) for t in tr[:70]], "test_train": [("navtrain", t) for t in tr[70:]],
        "test_navtest": [("navtest", t) for t in nt]}
data = {k: [(s, t, raw(s, t, "bev"), raw(s, t, "seg"), gt_target(t)) for s, t in v] for k, v in sets.items()}


def feats(items, fn):
    X = np.concatenate([fn(s, t, b).reshape(256, -1).T for s, t, b, _, _ in items])
    return np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)


def ridge(X, Y, lam=1.0):
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-3
    Xn = X.copy(); Xn[:, :-1] = (Xn[:, :-1] - mu) / sd
    A = Xn.T.astype(np.float64) @ Xn; A[np.diag_indices_from(A)] += lam; A[-1, -1] -= lam
    W = np.linalg.solve(A, Xn.T.astype(np.float64) @ Y)
    return lambda Z: ((np.concatenate([(Z[:, :-1] - mu) / sd, Z[:, -1:]], 1)) @ W)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


res = {"n_fit": 70, "n_test_train": 30, "n_test_navtest": 30}
# (1) ReSMap bev -> GT drivable, and bev -> seg (seg in chosen transform, pooled to 50x100)
for h, f in HYP.items():
    fn = lambda s, t, b: f(b)
    Xf = feats(data["fit"], fn)
    yg = np.concatenate([d[4].reshape(-1) for d in data["fit"]]).astype(np.float64)[:, None]
    ys = np.concatenate([pool(HYP["chosen"](d[3])[[0, 2]]).reshape(2, -1).T for d in data["fit"]])
    pg, ps = ridge(Xf, yg), ridge(Xf, ys)
    out = {}
    for k in ("test_train", "test_navtest"):
        Xt = feats(data[k], fn)
        lab = np.concatenate([d[4].reshape(-1) for d in data[k]])
        p = pg(Xt)[:, 0]
        pr = p > 0.5
        yst = np.concatenate([pool(HYP["chosen"](d[3])[[0, 2]]).reshape(2, -1).T for d in data[k]])
        pst = ps(Xt)
        out[k] = dict(gt_iou=float((pr & lab).sum() / (pr | lab).sum()), gt_auc=auc(p, lab),
                      seg_road_r2=r2(yst[:, 0], pst[:, 0]), seg_cl_r2=r2(yst[:, 1], pst[:, 1]))
    res[f"resmap_bev:{h}"] = out
    print(h, out, flush=True)

# (2) BEVFusion S grid (existing arm) -> the same GT; own orientation and its mirror
for h, g in {"bevfusion_S": lambda x: x, "bevfusion_S_lateral_mirror": lambda x: x[..., ::-1]}.items():
    fn = lambda s, t, b: g(bevfusion_s(s, t))
    Xf = feats(data["fit"], fn)
    yg = np.concatenate([d[4].reshape(-1) for d in data["fit"]]).astype(np.float64)[:, None]
    pg = ridge(Xf, yg)
    out = {}
    for k in ("test_train", "test_navtest"):
        Xt = feats(data[k], fn)
        lab = np.concatenate([d[4].reshape(-1) for d in data[k]])
        p = pg(Xt)[:, 0]; pr = p > 0.5
        out[k] = dict(gt_iou=float((pr & lab).sum() / (pr | lab).sum()), gt_auc=auc(p, lab))
    res[h] = out
    print(h, out, flush=True)

json.dump(res, open(OUT / "bev_probe_verify.json", "w"), indent=1)
