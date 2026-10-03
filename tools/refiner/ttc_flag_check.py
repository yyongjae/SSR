#!/usr/bin/env python
"""Sanity check of the TTC surrogate flag against the official TTC label (PRESTATED_DECISION_RULE AMENDMENT 4 (1)).

NOT the m_ttc selection (AMENDMENT 4 (2)); a small agreement check on TRAIN data only.  Reads the D1-b sweep pool
(<data>/m8_recheck/tokens.parquet, one token per train log; labels_pool.parquet = the stored official labels of its
13-draft banks), drafts/train, objects/train and human/train.  Dev / navtest paths are refused (m8_recheck._train_only).

Per token: the stored bank (k = 0 human identity + the VALID perturbed drafts; invalid slots skipped, as m8_recheck),
dense references (geometry.dense_reference, the positions the LQR tracker follows; the official labels were scored on
the tracked states), scene = surrogate.scene_from_numpy(objects, human_traj=human) (0..5 s objects + human mask).
Margin-free statistics per draft:
  col_g   min smooth separation of C_col (default config: not-behind, SAT human mask) -> col flag at m <=> col_g < m
  ttc_g   min smooth separation of C_ttc (projected boxes, delta 0.3 / 0.6 / 0.9 s)     -> TTC flag at m <=> ttc_g < m
  ttc_g_stop  the same with the official stopped-ego skip (cfg.ttc_min_speed = 5e-3 m/s; diagnostic only)
Printed confusion vs official ttc < 1 (all bank drafts, and human drafts k = 0 separately):
  proj  : ttc_g < m_ttc                                   (the new term's own flag)
  combo : col_g < m_col OR ttc_g < m_ttc                  (AMENDMENT 4 (2) TTC flag; m_col 0.15)

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python tools/refiner/ttc_flag_check.py --n 100 [--workers 2]
      [--m-ttc 0.0] [--m-col 0.15] [--out rows.parquet]
Writes nothing unless --out is given.  CPU only, <= 2 workers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))

import m8_recheck as R  # noqa: E402

POOL = R.OUT / "tokens.parquet"
LABELS_POOL = R.OUT / "labels_pool.parquet"
STOP_SPEED = 5e-3                         # official stopped_speed_threshold (pdm_scorer TTC skip)

_W: dict = {}


def _init():
    import torch
    torch.set_num_threads(1)
    R._train_only(R.HUMAN)
    h = np.load(R.HUMAN)
    _W["h"] = {k: h[k] for k in ("traj", "frame_gap")}
    _W["hidx"] = {t: i for i, t in enumerate(h["tokens"].tolist())}


def token_rows(task):
    import torch
    from navsim.agents.para_ssr.refiner import gt_future as GF
    from navsim.agents.para_ssr.refiner import surrogate as SU
    from navsim.agents.para_ssr.refiner.geometry import dense_reference
    token, log = task
    dpath, opath = R.DRAFT_DIR / f"{token}.npz", R.OBJ_DIR / f"{token}.npz"
    for p in (dpath, opath):
        R._train_only(p)
    with np.load(dpath) as z:
        drafts = np.asarray(z["drafts"], np.float32)
        fam = np.asarray(z["family"]).astype(int)
        valid = np.asarray(z["valid"]).astype(bool)
        assert str(z["split"]) == "train", str(z["split"])
    i = _W["hidx"][token]
    assert not bool(_W["h"]["frame_gap"][i])
    objs = GF.load_objects(opath)
    scene = SU.collate_scenes([SU.scene_from_numpy(objs, human_traj=_W["h"]["traj"][i])])
    bank_k = [0] + [k for k in range(1, 13) if valid[k]]
    dense = dense_reference(torch.as_tensor(drafts[bank_k].astype(np.float64)))
    idx = torch.zeros(len(bank_k), dtype=torch.long)
    with torch.no_grad():
        col = SU.collision_cost(dense, scene, idx, SU.SurrogateConfig())
        ttc = SU.ttc_cost(dense, scene, idx, SU.SurrogateConfig(), details=True)
        stp = SU.ttc_cost(dense, scene, idx, SU.SurrogateConfig(ttc_min_speed=STOP_SPEED), details=True)
    return [dict(token=token, log=log, k=int(k), fam=int(fam[k]), col_g=float(col["gmin"][j]),
                 ttc_g=float(ttc["gmin"][j]), ttc_g_stop=float(stp["gmin"][j]), ttc_first_n=int(ttc["first_n"][j]),
                 n_obj=int(objs["kf"].shape[0])) for j, k in enumerate(bank_k)]


def confusion(flag, fail) -> dict:
    flag, fail = np.asarray(flag, bool), np.asarray(fail, bool)
    tp, fp = int((flag & fail).sum()), int((flag & ~fail).sum())
    fn, tn = int((~flag & fail).sum()), int((~flag & ~fail).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=tp + fp + fn + tn,
                recall=(tp / (tp + fn) if tp + fn else None), false_alarm=(fp / (fp + tn) if fp + tn else None),
                precision=(tp / (tp + fp) if tp + fp else None))


def summarize(df: pd.DataFrame, m_ttc: float, m_col: float) -> dict:
    fail = (df.ttc < 1).values
    proj = (df.ttc_g < m_ttc).values
    proj_stop = (df.ttc_g_stop < m_ttc).values
    col = (df.col_g < m_col).values
    hum = (df.k == 0).values
    out = {"n_tokens": int(df.token.nunique()), "n_drafts": int(len(df)), "m_ttc": m_ttc, "m_col": m_col,
           "official_ttc_fail": int(fail.sum()), "official_ttc_fail_human": int(fail[hum].sum())}
    for name, f in (("proj", proj), ("combo", col | proj), ("col_only", col), ("proj_stopskip", proj_stop),
                    ("combo_stopskip", col | proj_stop)):
        out[name] = {"bank": confusion(f, fail), "human_k0": confusion(f[hum], fail[hum])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=100, help="tokens (first n of a seeded permutation of the pool)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-ttc", type=float, default=0.0)
    ap.add_argument("--m-col", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional parquet of per-draft rows")
    a = ap.parse_args(argv)
    if a.workers > 2:
        raise SystemExit("--workers must be <= 2 (shared machine)")
    for p in (POOL, LABELS_POOL):
        R._train_only(p)
    toks = pd.read_parquet(POOL)
    rng = np.random.default_rng(a.seed)
    toks = toks.iloc[np.sort(rng.permutation(len(toks))[:a.n])]
    tasks = list(zip(toks.token, toks.log))
    t0 = time.time()
    if a.workers <= 1:
        _init()
        res = list(map(token_rows, tasks))
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init) as pool:
            res = pool.map(token_rows, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    lab = pd.read_parquet(LABELS_POOL, columns=["token", "k", "ttc", "nc"])
    df = df.merge(lab, on=["token", "k"], how="left", validate="one_to_one")
    assert df.ttc.notna().all()
    if a.out:
        df.to_parquet(a.out)
    s = summarize(df, a.m_ttc, a.m_col)
    s["sec"] = round(time.time() - t0, 1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
