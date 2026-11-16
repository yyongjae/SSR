#!/usr/bin/env python
"""Fidelity gate of regenerated camera-teacher caches: do they reproduce the caches the teachers' norm files and refiner
teachers were made from, and are their BEV axes the contract's?

  norm      --det D | --map D --ref N [--split-tokens P]   per-channel mean / std of the 2048 norm tokens
            (refiner/data.compute_teacher_norm: seed 0 over the sorted navtrain_train tokens, token_hash
            ad2931d1b6607238) against a norm_v15 file (resources/km_corridor/norm_{bevformer,mask2map}_100x100.npz):
            z = |mean - mean_ref| / std_ref and the relative std error |std / std_ref - 1| per channel (max, mean)
  det-axes  --det D --infos I --tokens T [--sim-tokens S --sim-infos J]   BEVFormer: (a) predicted boxes against the
            GT boxes of the info pkl (recall at 2 m, centre error) as stored and mirrored; (b) a linear probe from
            bev_feature cells to "a GT vehicle centre is in this cell", trained on half the real tokens and scored
            (AUC) on the rest, under the cache's axes and the 3 swapped / flipped ones (and on SimScale tokens)
  map-axes  --map D --infos I [--infos J ...] --tokens T     Mask2Map: seg (sigmoid >= 0.5) against the map GT
            rasterised at each token's pose (Mask2Map tools/navsim/navsim_map_gt.py, NavsimBEVRasterizer), IoU per
            class under the cache's axes and the 3 swapped / flipped ones
            (needs nuplan-devkit and NUPLAN_MAPS_ROOT, and the Mask2Map checkout: M2M_SRC or --m2m)
SSR python, CPU only.  Prints one JSON report (and writes it with --out).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
os.environ["CK_BEV64"] = "1"                     # the 64 m readers, before the refiner modules are imported

import numpy as np  # noqa: E402

NORM_HASH = "ad2931d1b6607238"
TRAIN_TOKENS = "splits/e2e_train_trainlogs.parquet"
HYPOTHESES = ("as_cached", "flip_rows", "flip_cols", "transpose")


def _read(path) -> List[str]:
    return [t.strip() for t in Path(path).read_text().splitlines() if t.strip()]


def _hyp(x: np.ndarray, h: str) -> np.ndarray:
    """One axis hypothesis on the last two axes of x (square grids)."""
    if h == "as_cached":
        return x
    if h == "flip_rows":
        return x[..., ::-1, :]
    if h == "flip_cols":
        return x[..., :, ::-1]
    return np.swapaxes(x, -1, -2)


# ----------------------------------------------------------------------------------------------- norm
def norm_report(cache, train_tokens: Sequence[str], ref: Path) -> dict:
    from navsim.agents.para_ssr.refiner.data import compute_teacher_norm
    from navsim.agents.para_ssr.refiner.adapters import load_norm
    mean, std, info = compute_teacher_norm(cache, list(train_tokens), 2048, seed=0)
    m0, s0, meta0 = load_norm(ref)
    z = np.abs(mean.astype(np.float64) - m0) / s0
    rs = np.abs(std.astype(np.float64) / s0 - 1.0)
    return dict(token_hash=info["token_hash"], token_hash_ok=info["token_hash"] == NORM_HASH, n_tokens=info["n_tokens"],
                n_pool=info["n_pool"], ref=str(ref), ref_token_hash=meta0.get("token_hash"),
                z_mean_diff=dict(max=float(z.max()), mean=float(z.mean()), p99=float(np.quantile(z, 0.99)),
                                 argmax=int(z.argmax())),
                rel_std_err=dict(max=float(rs.max()), mean=float(rs.mean()), p99=float(np.quantile(rs, 0.99)),
                                 argmax=int(rs.argmax())),
                frac_zero_mean=info["frac_zero_mean"], ref_frac_zero_mean=meta0.get("frac_zero_mean"),
                mean_corr=float(np.corrcoef(mean, m0)[0, 1]), std_corr=float(np.corrcoef(std, s0)[0, 1]))


def cmd_norm(a) -> dict:
    import pandas as pd
    from navsim.common import ssr_paths as SP
    tr = pd.read_parquet(a.split_tokens or Path(SP.SSR_DATA_ROOT) / TRAIN_TOKENS)
    toks = tr.token.astype(str).tolist()
    if a.det:
        from navsim.agents.para_ssr.refiner.data import TeacherCache
        cache = TeacherCache(a.det, expect_sha_head=a.det_sha_head)
    else:
        from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
        cache = ResmapCache(a.map, expect_sha256=a.map_sha256, expect_split="none")
    return norm_report(cache, toks, Path(a.ref))


def norm_tokens(train_tokens: Sequence[str], n: int = 2048, seed: int = 0) -> List[str]:
    """The tokens compute_teacher_norm samples (sorted pool, seeded choice, sorted)."""
    tokens = sorted(set(train_tokens))
    rng = np.random.default_rng(seed)
    pick = tokens if len(tokens) <= n else [tokens[i] for i in sorted(rng.choice(len(tokens), n, replace=False))]
    assert len(tokens) != 85109 or hashlib.sha256("\n".join(pick).encode()).hexdigest()[:16] == NORM_HASH
    return pick


# ----------------------------------------------------------------------------------------------- DET axes
def bev_infos(path) -> Dict[str, dict]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return {i["token"]: i for i in (d["infos"] if isinstance(d, dict) else d)}


def gt_centres(info: dict, classes=("vehicle",)) -> np.ndarray:
    """GT box centres (x forward, y left) in the 64 m front ROI of one BEVFormer info, of the given classes."""
    b = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7)
    keep = np.isin(np.asarray(info["gt_names"]).astype(str), classes)
    xy = b[keep, :2]
    return xy[(xy[:, 0] > 0) & (xy[:, 0] < 64) & (np.abs(xy[:, 1]) < 32)]


def box_match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float = 0.3, radius: float = 2.0):
    """(matched GT, GT, centre errors of matched): greedy nearest predicted centre (score >= thr) within radius."""
    p = pred[scores >= thr, :2]
    if len(gt) == 0:
        return 0, 0, []
    if len(p) == 0:
        return 0, len(gt), []
    d = np.linalg.norm(gt[:, None, :] - p[None, :, :], axis=-1)
    used, errs = set(), []
    for g in np.argsort(d.min(1)):
        j = [k for k in np.argsort(d[g]) if k not in used and d[g, k] <= radius]
        if j:
            used.add(j[0])
            errs.append(float(d[g, j[0]]))
    return len(errs), len(gt), errs


def det_boxes(det: Path, tokens: Sequence[str], infos: Dict[str, dict]) -> dict:
    """Predicted vs GT vehicle centres, as stored and with y mirrored (a lateral flip would show as recall loss)."""
    out = {}
    for name, sign in (("as_cached", 1.0), ("mirror_y", -1.0)):
        m = n = 0
        errs: List[float] = []
        for t in tokens:
            with np.load(det / "samples" / t[:2] / f"{t}.npz", allow_pickle=False) as z:
                pb, ps, pl = z["pred_boxes_3d"], z["pred_scores_3d"], z["pred_labels_3d"]
            keep = pl == 0                          # vehicle
            pred = pb[keep].astype(np.float64)
            pred[:, 1] *= sign
            a, b, e = box_match(pred, ps[keep], gt_centres(infos[t]))
            m, n, errs = m + a, n + b, errs + e
        out[name] = dict(recall_2m=m / max(n, 1), n_gt=n, centre_err_median=float(np.median(errs)) if errs else None)
    return out


def det_probe(det: Path, train: Sequence[str], test_sets: Dict[str, Sequence[str]], infos: Dict[str, dict],
              cells_per_token: int = 400, seed: int = 0) -> dict:
    """AUC of a logistic probe bev cell -> 'a GT vehicle centre lies in this 0.64 m cell' (cache layout (C, x, y_left
    ascending)), fit on `train` under each axis hypothesis and scored on each test set under the same hypothesis."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def label(info):
        lab = np.zeros((100, 100), bool)
        xy = gt_centres(info)
        i = np.clip((xy[:, 0] / 0.64).astype(int), 0, 99)
        j = np.clip(((xy[:, 1] + 32.0) / 0.64).astype(int), 0, 99)
        lab[i, j] = True
        return lab

    def cells(tokens, h, rng, n=None):
        X, Y = [], []
        for t in tokens:
            with np.load(det / "samples" / t[:2] / f"{t}.npz", allow_pickle=False) as z:
                bev = z["bev_feature"].astype(np.float32)
            f = _hyp(bev, h).reshape(256, -1).T
            y = label(infos[t]).reshape(-1)
            pos = np.flatnonzero(y)
            neg = rng.choice(np.flatnonzero(~y), n or cells_per_token, replace=False)
            idx = np.concatenate([pos, neg])
            X.append(f[idx])
            Y.append(y[idx])
        return np.concatenate(X), np.concatenate(Y)

    out = {}
    for h in HYPOTHESES:
        rng = np.random.default_rng(seed)
        X, Y = cells(train, h, rng)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1).fit((X - mu) / sd, Y)
        out[h] = {}
        for name, toks in test_sets.items():
            Xt, Yt = cells(toks, h, np.random.default_rng(seed + 1))
            out[h][name] = float(roc_auc_score(Yt, clf.decision_function((Xt - mu) / sd)))
    return out


def cmd_det_axes(a) -> dict:
    det = Path(a.det)
    infos = bev_infos(a.infos)
    toks = [t for t in _read(a.tokens) if (det / "samples" / t[:2] / f"{t}.npz").is_file()][:a.n]
    half = len(toks) // 2
    rep = {"n_real": len(toks), "boxes_real": det_boxes(det, toks, infos)}
    tests = {"real_heldout": toks[half:]}
    if a.sim_tokens:
        sinfo = bev_infos(a.sim_infos)
        st = [t for t in _read(a.sim_tokens) if (det / "samples" / t[:2] / f"{t}.npz").is_file()][:a.n]
        infos.update(sinfo)
        rep.update(n_sim=len(st), boxes_sim=det_boxes(det, st, infos))
        tests["sim"] = st
    rep["probe_auc"] = det_probe(det, toks[:half], tests, infos)
    return rep


# ----------------------------------------------------------------------------------------------- MAP axes
def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def map_gt_masks(m2m: Path, infos: Sequence[dict]) -> Dict[str, np.ndarray]:
    """(4, 200, 200) uint8 map GT per info in Mask2Map's raw frame (row 0 = y -32 m, col 0 = x 0 m), from nuPlan's
    map API at the info's ego pose, with Mask2Map's own extractor and rasteriser."""
    from navsim.common import ssr_paths as SP
    gt = _load_by_path("m2m_navsim_map_gt", m2m / "tools/navsim/navsim_map_gt.py")
    mu = _load_by_path("m2m_navsim_map_utils", m2m / "projects/mmdet3d_plugin/datasets/navsim_map_utils.py")
    gt.MAP_ROOT = SP.NUPLAN_MAPS_ROOT
    ras = mu.NavsimBEVRasterizer((64.0, 64.0), (200, 200), 3)
    ex: Dict[str, object] = {}
    out = {}
    for info in infos:
        city = info["location"]
        if city not in ex:
            ex[city] = gt.Extractor(city)
        t = info["e2g_translation"]
        g = ex[city](float(t[0]), float(t[1]), gt.heading_of(info["e2g_rotation"]))
        geoms = {}
        for k, c in enumerate(gt.CLASSES):
            if c == "centerline":
                from shapely.geometry import LineString
                geoms[k] = [LineString(x) for x in g[c] if len(x) >= 2]
            else:
                from shapely.geometry import Polygon
                geoms[k] = [Polygon(r[0], [h for h in r[1:] if len(h) >= 3]) for r in g[c] if len(r[0]) >= 3]
        out[info["token"]] = ras(geoms, 4)
    return out


def cmd_map_axes(a) -> dict:
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
    m2m = Path(a.m2m or os.environ.get("M2M_SRC", ""))
    rc = ResmapCache(a.map, expect_sha256=a.map_sha256, expect_split="none")
    infos = {}
    for p in a.infos:
        with open(p, "rb") as fh:
            infos.update({i["token"]: i for i in pickle.load(fh)})
    rep = {}
    for name, path in (("real", a.tokens), ("sim", a.sim_tokens)):
        if not path:
            continue
        toks = [t for t in _read(path) if rc.has(t) and t in infos][:a.n]
        masks = map_gt_masks(m2m, [infos[t] for t in toks])
        inter = {h: np.zeros(4) for h in HYPOTHESES}
        union = {h: np.zeros(4) for h in HYPOTHESES}
        for t in toks:
            seg = rc.load_seg(t, s_grid=False).astype(np.float32)          # cache: row 0 = y +32 m
            raw = (1.0 / (1.0 + np.exp(-seg[:, ::-1, :]))) >= 0.5          # back to Mask2Map's raw rows (row 0 = y -32)
            g = masks[t].astype(bool)
            for h in HYPOTHESES:
                p = _hyp(raw, h)
                inter[h] += (p & g).reshape(4, -1).sum(1)
                union[h] += (p | g).reshape(4, -1).sum(1)
        rep[name] = dict(n=len(toks), iou={h: dict(zip(("road", "walkway", "centerline", "crosswalk"),
                                                        (inter[h] / np.maximum(union[h], 1)).round(4).tolist()))
                                            for h in HYPOTHESES})
    return rep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("norm")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--det")
    g.add_argument("--map")
    s.add_argument("--ref", required=True)
    s.add_argument("--split-tokens", default=None, help="navtrain_train token table (default the SSR_DATA_ROOT one)")
    s = sub.add_parser("det-axes")
    s.add_argument("--det", required=True)
    s.add_argument("--infos", required=True)
    s.add_argument("--tokens", required=True)
    s.add_argument("--sim-tokens", default=None)
    s.add_argument("--sim-infos", default=None)
    s.add_argument("--n", type=int, default=200)
    s = sub.add_parser("map-axes")
    s.add_argument("--map", required=True)
    s.add_argument("--infos", nargs="+", required=True)
    s.add_argument("--tokens", required=True)
    s.add_argument("--sim-tokens", default=None)
    s.add_argument("--m2m", default=None)
    s.add_argument("--n", type=int, default=50)
    for s in sub.choices.values():
        s.add_argument("--det-sha-head", default="4ca4f1aa55011beb")
        s.add_argument("--map-sha256", default="579665a7309beb17621dcbad511031776015f0a58c65a2beee1ee58ed45771e4")
        s.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rep = {"norm": cmd_norm, "det-axes": cmd_det_axes, "map-axes": cmd_map_axes}[a.cmd](a)
    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
