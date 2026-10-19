#!/usr/bin/env python
"""refiner v4_3 Phase 0 oracle -- criterion 1: 'outside the GT tube <=> official fail' on the EXECUTED states of the 96
candidates (par16 + valid var80) of the 6,000 F1 navtrain_val tokens.  CPU only, no scorer call (the official labels
are the F1 exec labels, bit-exact).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/run_crit1.py [--workers 28] [--limit N]

Per token (oracle_common.parent_fields on the 16 parents, then every candidate on its parent's grid):
  grid (learnable representation; RT.executed_slack, hit = slack < 0)
    g{10,25,50}_{nc,ttc}   nearest cell, 0.1 / 0.25 (main) / 0.5 s bins, TTC with the LON channel (not behind from the
                           current ego); g25_dac = executed corners on D (SDF at the grid points)
    g25a_{nc,ttc,dac}      read 'any4' (conservative), 0.25 s
    g25b_ttc               TTC on B (not-behind from the projected cell, no LON), 0.25 s
    g25s_nc / g25sa_nc     stopped-aware NC (oracle_common.stopped_aware_B01: stopped / static objects block without
                           the not-behind proxy, the official is_track_stopped -> at fault), nearest / any4
    p25_{nc,ttc}           point target O at the 9 footprint points, 0.25 s
    g25_slack_{nc,ttc,dac}, g25_noog (executed states off the parent grid)
  exact (grid-free; oracle_common.exact_flags)
    x_{nc,ttc}             proxy rules (= the field's, on exact boxes); x_dac = stored-SDF corners < 0; x_min_corner
    x_nc_sa                stopped-aware NC proxy on exact boxes
    x_{nc,ttc}_lo / _hi    official classification emulation (lane-map bounds); x_nc_val_lo / _hi (0 / 0.5 / 1)
  UNKNOWN (official failures the GT store cannot see):  unk_nc / unk_ttc = the official nc_track / ttc_track (label
    shards) is not in the GT store, or the failure time (+ 0.9 s for TTC) exceeds the annotated horizon t_avail;
    unk_dac = official DAC fail with no readable corner below 0 and some corner outside the E grid.
  token: gt_ok, sdf_ok, t_avail, n_obj, top1_min_corner (stored SDF, top-1 executed; near curb < 1 m), top1_obj_dmin
    (min distance OBS GT centre -> top-1 executed rear axle over 0..4 s; lane-only >= 30 m), sec.
Arrays are [n_tok, 96] (invalid columns: valid False, flags False).  Shards OUT/crit1/shard_XXX.npz (resumable),
merged to OUT/crit1/rows.npz + OUT/crit1/run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_common as OC  # noqa: E402
import numpy as np  # noqa: E402

OUTD = OC.OUT / "crit1"
_G: dict = {}
BOOL_KEYS = ("valid", "g10_nc", "g10_ttc", "g25_nc", "g25_ttc", "g25_dac", "g50_nc", "g50_ttc", "g25a_nc", "g25a_ttc",
             "g25a_dac", "g25b_ttc", "g25s_nc", "g25sa_nc", "p25_nc", "p25_ttc", "x_nc", "x_nc_sa", "x_ttc", "x_dac",
             "x_nc_lo", "x_nc_hi", "x_ttc_lo", "x_ttc_hi", "unk_nc", "unk_ttc", "unk_dac")
FLOAT_KEYS = ("g25_slack_nc", "g25_slack_ttc", "g25_slack_dac", "x_min_corner", "x_nc_val_lo", "x_nc_val_hi",
              "g25_noog")
TOK_KEYS = ("gt_ok", "sdf_ok", "t_avail", "n_obj", "top1_min_corner", "top1_obj_dmin", "sec")


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    _G["D"] = OC.Data()


def process(i: int) -> dict:
    t0 = time.time()
    T = OC.token_inputs(_G["D"], i)
    out = {k: np.zeros(OC.N96, bool) for k in BOOL_KEYS}
    out.update({k: np.full(OC.N96, np.nan) for k in FLOAT_KEYS})
    gt, sdf = T["gt"], T["sdf"]
    tok = dict(gt_ok=gt is not None, sdf_ok=sdf is not None, t_avail=np.nan if gt is None else gt["t_avail"],
               n_obj=0 if gt is None else gt["boxes"].shape[0], top1_min_corner=np.nan, top1_obj_dmin=np.nan)
    if sdf is None:
        tok["sec"] = time.time() - t0
        return dict(i=i, row=T["row"], **out, **tok)
    cols = np.flatnonzero(T["valid96"])
    idx = OC.parent_of(cols)
    X = T["X"][cols]
    F = OC.parent_fields(T["par"], gt, sdf)
    out["valid"][cols] = True
    for tb, tag in ((0.1, "g10"), (0.25, "g25"), (0.5, "g50")):
        g = OC.grid_eval(X, F, idx, tb=tb, dac="grid" if tag == "g25" else None)
        out[f"{tag}_nc"][cols] = g["hit_nc"]
        out[f"{tag}_ttc"][cols] = g["hit_ttc"]
        if tag == "g25":
            out["g25_dac"][cols] = g["hit_dac"]
            for k in ("nc", "ttc", "dac"):
                out[f"g25_slack_{k}"][cols] = g[f"slack_{k}"]
            out["g25_noog"][cols] = g["n_oog"]
    g = OC.grid_eval(X, F, idx, tb=0.25, read="any4")
    for k in ("nc", "ttc", "dac"):
        out[f"g25a_{k}"][cols] = g[f"hit_{k}"]
    out["g25b_ttc"][cols] = OC.grid_eval(X, F, idx, tb=0.25, ttc_lon=False, dac=None)["hit_ttc"]
    Fs = dict(F, B01=OC.stopped_aware_B01(F["frn"], gt))
    out["g25s_nc"][cols] = OC.grid_eval(X, Fs, idx, tb=0.25, dac=None)["hit_nc"]
    out["g25sa_nc"][cols] = OC.grid_eval(X, Fs, idx, tb=0.25, read="any4", dac=None)["hit_nc"]
    p = OC.point_eval(X, F, idx, tb=0.25)
    out["p25_nc"][cols], out["p25_ttc"][cols] = p["hit_nc"], p["hit_ttc"]
    x = OC.exact_flags(X, gt, sdf)
    for k in ("nc", "ttc"):
        out[f"x_{k}"][cols] = x[f"{k}_proxy"]
        out[f"x_{k}_lo"][cols] = x[f"{k}_emul_lo"]
        out[f"x_{k}_hi"][cols] = x[f"{k}_emul_hi"]
    out["x_dac"][cols] = x["dac"]
    out["x_nc_sa"][cols] = x["nc_proxy_sa"]
    out["x_min_corner"][cols] = x["min_corner"]
    out["x_nc_val_lo"][cols], out["x_nc_val_hi"][cols] = x["nc_val_lo"], x["nc_val_hi"]
    # UNKNOWN bucket of the official failures
    e = T["exec"]
    store = set(gt["track"].tolist()) if gt is not None else set()
    t_av = gt["t_avail"] if gt is not None else 0.0
    for q, c in enumerate(cols):
        trk = T["tracks"].get(int(c), ("", ""))
        if e[c, OC.EC["nc"]] < 1:
            tn = 0.1 * e[c, OC.EC["nc_t"]]
            out["unk_nc"][c] = (trk[0] not in store) or tn > t_av + 1e-6
        if e[c, OC.EC["ttc"]] < 1:
            tt = 0.1 * e[c, OC.EC["ttc_t"]] + 0.9
            out["unk_ttc"][c] = (trk[1] not in store) or tt > t_av + 1e-6
    cor = OC.ego_corners(OC.torch.as_tensor(X[..., :3])).numpy()
    _, okc = OC.sample_sdf_np(sdf, cor)
    unread = (~okc).any((-1, -2))
    out["unk_dac"][cols] = (e[cols, OC.EC["dac"]] < 1) & (x["min_corner"] >= 0) & unread
    sb = OC.scene_bins(T["X"][0], gt, sdf)
    tok.update(top1_min_corner=sb["min_corner"], top1_obj_dmin=sb["obj_dmin"], sec=time.time() - t0)
    return dict(i=i, row=T["row"], **out, **tok)


def stack(res: list) -> dict:
    res = sorted(res, key=lambda d: d["i"])
    o = {"i": np.array([d["i"] for d in res]), "row": np.array([d["row"] for d in res])}
    for k in BOOL_KEYS + FLOAT_KEYS:
        o[k] = np.stack([d[k] for d in res])
    for k in TOK_KEYS:
        o[k] = np.array([d[k] for d in res], np.float64)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=250)
    a = ap.parse_args()
    assert a.workers <= 28
    OUTD.mkdir(parents=True, exist_ok=True)
    n = len(np.load(OC.PREP / "f1_rows.npy"))
    items = list(range(n if a.limit <= 0 else min(a.limit, n)))
    shards = [items[k:k + a.shard] for k in range(0, len(items), a.shard)]
    tag = "" if a.limit <= 0 else f"_lim{a.limit}"
    todo = [(j, s) for j, s in enumerate(shards) if not (OUTD / f"shard{tag}_{j:03d}.npz").is_file()]
    t0 = time.time()
    print(f"{time.strftime('%F %T')} UTC tokens {len(items)} shards {len(shards)} todo {len(todo)}", flush=True)
    with Pool(a.workers, initializer=init, maxtasksperchild=100) as pool:
        for j, s in todo:
            res = pool.map(process, s, chunksize=1)
            np.savez(OUTD / f"shard{tag}_{j:03d}.npz", **stack(res))
            print(f"{time.strftime('%F %T')} shard {j} done ({time.time() - t0:.0f} s)", flush=True)
    parts = [dict(np.load(OUTD / f"shard{tag}_{j:03d}.npz")) for j in range(len(shards))]
    cat = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    np.savez(OUTD / f"rows{tag}.npz", **cat)
    (OUTD / f"run{tag}.json").write_text(json.dumps(dict(n=len(items), sec=time.time() - t0,
                                                          mean_sec_token=float(np.mean(cat["sec"])),
                                                          gt_missing=int((cat["gt_ok"] == 0).sum()),
                                                          sdf_missing=int((cat["sdf_ok"] == 0).sum())), indent=1))
    print("done", len(items), f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
