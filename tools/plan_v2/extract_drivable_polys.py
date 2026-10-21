"""Dump the scorer's own drivable polygons, in the ego frame at t=0, for a list of tokens.

Runs against the navsim v2 worktree, because that is what pickled the metric caches (their
`MapParameters` / `SceneFrameType` do not exist in this repo).  Any consumer in THIS repo reads the
npz instead of the pickles, so the two navsim packages never share a process.

    PYTHONPATH=~/kyungmin/navsim_v2 python tools/plan_v2/extract_drivable_polys.py \
        --out <file.npz> [--scenes 60] [--workers 8]
"""
import argparse
from pathlib import Path

import numpy as np

CACHE = Path("/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navtest")
SCORES = Path("/home/external-user/kyungmin/SSR-v2/data/planning_vb/navtest_refined_scores.npz")


def one(token):
    import lzma
    import pickle

    hits = list(CACHE.rglob(f"{token}/metric_cache.pkl"))
    if not hits:
        return None
    with lzma.open(hits[0], "rb") as f:
        mc = pickle.load(f)
    ra = mc.ego_state.rear_axle
    ox, oy, oh = float(ra.x), float(ra.y), float(ra.heading)
    c, s = np.cos(-oh), np.sin(-oh)
    R = np.array([[c, -s], [s, c]])
    out = []
    for g in getattr(mc.drivable_area_map, "_geometries", []):
        try:
            p = np.asarray(g.exterior.coords, np.float64)[:, :2]
            out.append(((p - [ox, oy]) @ R.T).astype(np.float32))
        except Exception:
            continue
    return (token, out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tokens", default="", help="file with one token per line; overrides sampling")
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor

    z = np.load(SCORES, allow_pickle=True)
    ci = [str(c) for c in z["columns"]].index("drivable_area_compliance")
    tokens = [str(t) for t in z["tokens"]]
    dac = np.asarray(z["scores"])[:, :256, ci]
    # only scenes the listwise loss can even be defined on: some feasible and some infeasible
    n = dac.sum(1)
    if args.tokens:
        want = [t.strip() for t in open(args.tokens) if t.strip()]
        at = {t: i for i, t in enumerate(tokens)}
        cand = [at[t] for t in want if t in at]
        args.scenes = len(cand)
    else:
        cand = [i for i in np.random.default_rng(0).permutation(len(tokens))
                if 0 < n[i] < dac.shape[1]]

    saved, took = {}, []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in zip(cand, ex.map(one, [tokens[i] for i in cand[: args.scenes * 3]])):
            if r is None:
                continue
            tok, polys = r
            for j, p in enumerate(polys):
                saved[f"{tok}|{j}"] = p
            took.append(i)
            if len(took) >= args.scenes:
                break

    saved["__tokens__"] = np.array([tokens[i] for i in took])
    saved["__dac__"] = dac[took].astype(np.float32)
    np.savez_compressed(args.out, **saved)
    print(f"{len(took)} 장면, 폴리곤 {len(saved) - 2}개 -> {args.out}")


if __name__ == "__main__":
    main()
