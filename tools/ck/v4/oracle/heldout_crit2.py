#!/usr/bin/env python
"""refiner v4_3 Phase 0b -- criterion 2 (informational): minimal computed edit of the v2 top-1 parent (column 0) when it
is outside the R2 GT tube (heldout_common.GridTubeR2), scored by the OFFICIAL scorer, vs the DET lateral head on the
same parent; on the 12,179 held-out tokens.  CPU only.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/heldout_crit2.py [--workers 28] [--limit N]

Configs CFGS (oracle_common.compute_edit, the Phase 0 rule; tube + margins (m_t s, m_l m)):
  r2_t0.2_l0.3        the Phase 0b criterion 2 config (R2 tube, the spec's margins m_t 0.2 s, m_l 0.3 m)
  r2_* (4)            margin sensitivity as Phase 0 (MARGIN_CONFIGS[1:])
  r2_t0.2_l0.3_best   best effort (an empty tube keeps the largest edit)
  grid_t0.2_l0.3      Phase 0's main grid tube on the held-out tokens (context: replication, no verdict)
  exact_t0.2_l0.3     Phase 0's grid-free exact tube (context: upper bound)
Per config: as run_crit2 (kind, a, D, inside_after, outside0, rounds, t_out, delta, traj, lab [11] official labels of
the submitted trajectory, edited).  One official call per token with an edit: [top1, head, unique edits]; the top-1 /
head rows give the parity vs the stored labels (par_max_diff).  No speed-cap probe (not part of Phase 0b).
Shards HO/crit2/shard_XXX.npz (resumable) -> HO/crit2/rows.npz + run.json.
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
import heldout_common as H  # noqa: E402
import oracle_common as OC  # noqa: E402
import run_crit2 as C2  # noqa: E402  (KINDS only)
import numpy as np  # noqa: E402

OUTD = H.HO / "crit2"
KINDS = C2.KINDS
MT, ML = OC.MARGIN_T, OC.MARGIN_L
CFGS = ([("r2", mt, ml, False) for mt, ml in OC.MARGIN_CONFIGS] + [("r2", MT, ML, True)]
        + [("grid", MT, ML, False), ("exact", MT, ML, False)])
CFG_NAMES = tuple(OC.cfg_name(*c) for c in CFGS)
MAIN_CFG = OC.cfg_name("r2", MT, ML)
NC_ = len(CFGS)
_G: dict = {}


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    _G["D"] = H.HeldoutData()
    _G["S"] = OC.OfficialScorer()
    _G["lat"] = np.load(H.LAT16, mmap_mode="r")


def process(i: int) -> dict:
    t0 = time.time()
    T = OC.token_inputs(_G["D"], i)
    top1 = T["par32"][0]
    lab0 = OC.exec_to_score(T["exec"][0])
    o = dict(i=i, row=T["row"], kind=np.zeros(NC_, np.int8), a=np.zeros(NC_), D=np.zeros(NC_),
             inside_after=np.zeros(NC_, bool), outside0=np.zeros((NC_, 3), bool), rounds=np.zeros(NC_, np.int8),
             t_out=np.full(NC_, np.nan), delta=np.full(NC_, np.nan), traj=np.repeat(top1[None], NC_, 0),
             lab=np.repeat(lab0[None], NC_, 0), edited=np.zeros(NC_, bool), par_max_diff=np.nan, ok=False,
             n_scored=0)
    gt, sdf = T["gt"], T["sdf"]
    if sdf is None or not T["fin"][0]:
        o["sec"] = time.time() - t0
        return o
    F = OC.parent_fields(T["par"][:1], gt, sdf, point=False)
    tubes = {"grid": OC.GridTube(F), "exact": OC.ExactTube(F, gt, sdf), "r2": H.GridTubeR2(H.r2_from(F, gt))}
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
    uniq, where = [], {}
    for q in np.flatnonzero(o["edited"]):
        b = o["traj"][q].tobytes()
        if b not in where:
            where[b] = len(uniq)
            uniq.append(o["traj"][q])
    batch = [top1, np.asarray(_G["lat"][T["row"], 0], np.float32)] + uniq
    if len(batch) > 2:
        lab = _G["S"].score(T["log"], T["token"], np.stack(batch))
        o["n_scored"] = len(batch)
        o["par_max_diff"] = float(max(np.abs(lab[0] - lab0).max(),
                                      np.abs(lab[1] - OC.exec_to_score(T["exec"][OC.COL_HEAD])).max()))
        for q in np.flatnonzero(o["edited"]):
            o["lab"][q] = lab[2 + where[o["traj"][q].tobytes()]]
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
    n = len(np.load(H.HO_PREP / "ho_rows.npy"))
    items = list(range(n if a.limit <= 0 else min(a.limit, n)))
    shards = [items[k:k + a.shard] for k in range(0, len(items), a.shard)]
    tag = "" if a.limit <= 0 else f"_lim{a.limit}"
    todo = [(j, s) for j, s in enumerate(shards) if not (OUTD / f"shard{tag}_{j:03d}.npz").is_file()]
    H.HeldoutData()                      # fail fast here (a failing Pool initializer respawns forever)
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
        parity_max_abs_diff=float(np.nanmax(cat["par_max_diff"])) if np.isfinite(cat["par_max_diff"]).any() else None,
        finished=H.kst()), indent=1))
    print("done", len(items), f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
