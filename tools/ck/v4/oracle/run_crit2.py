#!/usr/bin/env python
"""refiner v4_3 Phase 0 oracle -- criterion 2: minimal computed edit of the v2 top-1 parent (column 0) when it is outside
the GT tube, scored by the OFFICIAL scorer; plus the curvature speed-limit probe.  CPU only.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/run_crit2.py [--workers 28] [--limit N]

Per token (oracle_common.compute_edit, module docstring there for the rule):
  configs CFGS (12): tube 'grid' (the learnable representation: GT B / LON / D on the top-1 parent's grid, 0.25 s bins,
    nearest read -- the main) and 'exact' (grid-free, upper bound) x margins (m_t s, m_l m) in MARGIN_CONFIGS (main
    (0.2, 0.3); sensitivity (0.1, 0.3), (0.3, 0.3), (0.2, 0.2), (0.2, 0.5)), + the two main configs with best_effort
    (an empty tube keeps the largest edit instead of no edit).
  For each config: kind (KINDS index; 'empty_*' = no feasible edit, 'noconv_lon' = the grid critical-station loop
    ran out of MAX_ROUNDS with every round feasible -- both submit the top-1 unedited), a (m/s^2), D (m),
    inside_after (the submitted trajectory is inside that tube),
    outside0 [3] (top-1 nc / ttc / dac hits of that tube), rounds, t_out, delta, traj [8, 3] float32, lab [11] = the
    official labels of the submitted trajectory (SCORE_COLS: nc, dac, ddc, ep, ttc, comfort, pdms, raw_progress,
    dac / nc / ttc time idx; the F1 top-1 labels when no edit).
  speed cap probe: cap_over (top-1 planned faster than v_max at some in-plan station), cap_n_over, cap_vmax_min,
    cap_dv_max, cap_lab [11] (official labels of the re-timed top-1; NaN when not over).
  parity: every scored token also re-scores the top-1 and the DET-head trajectory (F1 columns 0 / 96) in the same
    call; par_max_diff = max |official - F1 exec| over both (F1 stores float32).
Shards OUT/crit2/shard_XXX.npz (resumable) -> OUT/crit2/rows.npz + run.json.
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

OUTD = OC.OUT / "crit2"
KINDS = ("inside", "slow", "short", "lon", "lat", "lon+lat", "empty_lon", "empty_lat",
         "noconv_lon")          # appended last: the earlier indices are unchanged
CFGS = ([("grid", mt, ml, False) for mt, ml in OC.MARGIN_CONFIGS]
        + [("exact", mt, ml, False) for mt, ml in OC.MARGIN_CONFIGS]
        + [("grid", OC.MARGIN_T, OC.MARGIN_L, True), ("exact", OC.MARGIN_T, OC.MARGIN_L, True)])
CFG_NAMES = tuple(OC.cfg_name(*c) for c in CFGS)
NC_ = len(CFGS)
_G: dict = {}


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    _G["D"] = OC.Data()
    _G["S"] = OC.OfficialScorer()
    _G["lat"] = np.load(OC.T10 / "lat_traj16.npy", mmap_mode="r")


def process(i: int) -> dict:
    t0 = time.time()
    T = OC.token_inputs(_G["D"], i)
    top1 = T["par32"][0]
    lab0 = OC.exec_to_score(T["exec"][0])
    o = dict(i=i, row=T["row"], kind=np.zeros(NC_, np.int8), a=np.zeros(NC_), D=np.zeros(NC_),
             inside_after=np.zeros(NC_, bool), outside0=np.zeros((NC_, 3), bool), rounds=np.zeros(NC_, np.int8),
             t_out=np.full(NC_, np.nan), delta=np.full(NC_, np.nan), traj=np.repeat(top1[None], NC_, 0),
             lab=np.repeat(lab0[None], NC_, 0), edited=np.zeros(NC_, bool), cap_over=False, cap_n_over=0,
             cap_vmax_min=np.nan, cap_dv_max=np.nan, cap_lab=np.full(11, np.nan), par_max_diff=np.nan, ok=False,
             n_scored=0)
    gt, sdf = T["gt"], T["sdf"]
    if sdf is None or not T["fin"][0]:
        o["sec"] = time.time() - t0
        return o
    F = OC.parent_fields(T["par"][:1], gt, sdf, point=False)
    tubes = {"grid": OC.GridTube(F), "exact": OC.ExactTube(F, gt, sdf)}
    cache = {}
    for q, (tube, mt, ml, best) in enumerate(CFGS):
        key = (tube, mt, ml)
        if key not in cache:
            cache[key] = OC.compute_edit(top1, T["v0"], T["init"], T["X"][0], tubes[tube], mt, ml)
        r = cache[key]
        tr = r["traj_best"] if best else r["traj"]
        o["kind"][q] = KINDS.index(r["kind"])
        o["a"][q], o["D"][q] = r["a"], r["D"]
        o["inside_after"][q] = r["inside_after"]
        o["outside0"][q] = r["outside0"]
        o["rounds"][q], o["t_out"][q], o["delta"][q] = r["rounds"], r["t_out"], r["delta"]
        o["traj"][q] = tr
        o["edited"][q] = not np.array_equal(tr, top1)
    cap, info = OC.speed_cap_traj(T["par"][:1], F, sdf)
    o.update(cap_over=info["over"], cap_n_over=info["n_over"], cap_vmax_min=info["vmax_min"],
             cap_dv_max=info["dv_max"])
    # one official call: [top1, head, unique edits..., cap]
    uniq, where = [], {}
    for q in np.flatnonzero(o["edited"]):
        b = o["traj"][q].tobytes()
        if b not in where:
            where[b] = len(uniq)
            uniq.append(o["traj"][q])
    batch = [top1, np.asarray(_G["lat"][T["row"], 0], np.float32)] + uniq + ([cap] if cap is not None else [])
    if len(batch) > 2:
        lab = _G["S"].score(T["log"], T["token"], np.stack(batch))
        o["n_scored"] = len(batch)
        o["par_max_diff"] = float(max(np.abs(lab[0] - lab0).max(),
                                      np.abs(lab[1] - OC.exec_to_score(T["exec"][OC.COL_HEAD])).max()))
        for q in np.flatnonzero(o["edited"]):
            o["lab"][q] = lab[2 + where[o["traj"][q].tobytes()]]
        if cap is not None:
            o["cap_lab"] = lab[-1]
    o["ok"] = True
    o["sec"] = time.time() - t0
    return o


def stack(res: list) -> dict:
    res = sorted(res, key=lambda d: d["i"])
    return {k: np.stack([np.asarray(d[k]) for d in res]) for k in res[0]}


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
    print(f"{time.strftime('%F %T')} UTC tokens {len(items)} shards {len(shards)} todo {len(todo)} cfgs {CFG_NAMES}",
          flush=True)
    with Pool(a.workers, initializer=init, maxtasksperchild=100) as pool:
        for j, s in todo:
            res = pool.map(process, s, chunksize=1)
            np.savez(OUTD / f"shard{tag}_{j:03d}.npz", **stack(res))
            print(f"{time.strftime('%F %T')} shard {j} done ({time.time() - t0:.0f} s)", flush=True)
    parts = [dict(np.load(OUTD / f"shard{tag}_{j:03d}.npz")) for j in range(len(shards))]
    cat = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    np.savez(OUTD / f"rows{tag}.npz", **cat, cfg_names=np.array(CFG_NAMES), kinds=np.array(KINDS))
    (OUTD / f"run{tag}.json").write_text(json.dumps(dict(
        n=len(items), sec=time.time() - t0, cfgs=CFG_NAMES, kinds=KINDS, n_ok=int(cat["ok"].sum()),
        n_scored_tokens=int((cat["n_scored"] > 0).sum()),
        parity_max_abs_diff=float(np.nanmax(cat["par_max_diff"])) if np.isfinite(cat["par_max_diff"]).any() else None),
        indent=1))
    print("done", len(items), f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
