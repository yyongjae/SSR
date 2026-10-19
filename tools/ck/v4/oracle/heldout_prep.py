#!/usr/bin/env python
"""refiner v4_3 Phase 0b -- step 0: held-out inputs (CPU, read-only on every input; writes HO/prep only).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/heldout_prep.py [--workers 28]

Order (each check must pass before the next step; heldout_common docstring for the arrays):
  1. LQR pipeline parity on 50 F1 tokens (seeded, seed 0): the held-out code path (heldout_common.sim96: packed cand +
     r34 traj80 cast to float32, ego (status v0, a0, 0), ck/v4/lqr float64) vs the F1 stored official states
     (f1_sim, float32): max |dxy| <= 1e-4 m (else abort).  Also: F1 init == ego_init(status) on all 6,000 F1 rows.
  2. Label assembly parity: heldout_common.build_exec on the 6,000 F1 rows == F1 exec[:, :97] bit for bit (else abort).
  3. Held-out arrays: ho_rows / ho_tokens / ho_init / ho_fin / ho_exec / ho_sim (float64, [N, 96, 41, 6]) /
     ho_tracks.parquet (official nc_track / ttc_track of every NC / TTC failure among the 96, run_crit1 UNKNOWN bucket).
  prep.json: provenance + the check numbers.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_PARITY = 50
PARITY_TOL_M = 1e-4
_G: dict = {}


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    _G["cand"] = np.load(OC.PACK / "cand.npy", mmap_mode="r")
    _G["var"] = np.load(H.TRAJ80, mmap_mode="r")
    _G["status"] = np.load(H.STATUS, mmap_mode="r")


def sim_row(row: int):
    ini = H.ego_init(_G["status"][row:row + 1])[0]
    X, fin = H.sim96(_G["cand"][row], _G["var"][row], ini)
    return row, X, fin


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    assert a.workers <= 28
    H.HO_PREP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = {"started": H.kst(), "f1_rows": str(OC.PREP / "f1_rows.npy")}
    tok = pd.read_parquet(OC.PACK / "tokens.parquet")
    rows_f1 = H.f1_rows()
    rows = H.heldout_rows(len(tok))
    assert len(np.intersect1d(rows, rows_f1)) == 0 and len(rows) + len(rows_f1) == len(tok)
    logs = tok["log"].astype(str).to_numpy()
    rep.update(n_packed=len(tok), n_f1=len(rows_f1), n_heldout=len(rows), n_logs_heldout=len(set(logs[rows])),
               n_logs_shared=len(set(logs[rows]) & set(logs[rows_f1])))
    status = np.load(H.STATUS, mmap_mode="r")

    # 1. LQR parity on F1 tokens (+ init source)
    f1_init = np.load(OC.PREP / "f1_init.npy")
    ini_f1 = H.ego_init(status[rows_f1])
    rep["init_f1_max_abs_diff"] = float(np.abs(ini_f1 - f1_init).max())
    assert rep["init_f1_max_abs_diff"] == 0.0, rep["init_f1_max_abs_diff"]
    f1_sim = np.load(OC.PREP / "f1_sim.npy", mmap_mode="r")
    f1_fin = np.load(OC.PREP / "f1_fin.npy", mmap_mode="r")
    pick = np.sort(np.random.default_rng(0).choice(len(rows_f1), N_PARITY, replace=False))
    init()
    d = dict(dxy=[], dh=[], dv=[], da=[], dsteer=[])
    n_tr = 0
    for i in pick:
        _, X, fin = sim_row(int(rows_f1[i]))
        assert np.array_equal(fin, np.asarray(f1_fin[i, :OC.N96])), i
        R = np.asarray(f1_sim[i, :OC.N96], np.float64)
        d["dxy"].append(np.hypot(X[..., 0] - R[..., 0], X[..., 1] - R[..., 1]).max())
        d["dh"].append(np.abs(wrap(X[..., 2] - R[..., 2])).max())
        for k, c in (("dv", 3), ("da", 4), ("dsteer", 5)):
            d[k].append(np.abs(X[..., c] - R[..., c]).max())
        n_tr += OC.N96
    par = {f"max_{k}": float(np.max(v)) for k, v in d.items()}
    par.update(n_tokens=N_PARITY, n_trajectories=n_tr, f1_index=pick.tolist(), tol_m=PARITY_TOL_M,
               ok=bool(par["max_dxy"] <= PARITY_TOL_M))
    rep["lqr_parity_f1"] = par
    print("LQR parity on F1:", json.dumps(par)[:400], flush=True)
    assert par["ok"], par

    # 2. label assembly parity on the F1 rows
    shards = {g: H.read_shards(g) for g in ("par", "var", "head")}
    E_f1 = H.build_exec(rows_f1, shards)
    f1_exec = np.load(OC.PREP / "f1_exec.npy", mmap_mode="r")
    ref = np.asarray(f1_exec[:, :H.N97])
    rep["label_assembly_f1"] = dict(bit_identical=bool(np.array_equal(E_f1, ref)),
                                    max_abs_diff=float(np.abs(E_f1.astype(np.float64) - ref).max()),
                                    n_rows=len(rows_f1), n_cols=H.N97)
    print("label assembly vs F1 exec:", rep["label_assembly_f1"], flush=True)
    assert rep["label_assembly_f1"]["bit_identical"]

    # 3. held-out arrays
    np.save(H.HO_PREP / "ho_rows.npy", rows)
    np.save(H.HO_PREP / "ho_tokens.npy", tok["token"].astype(str).to_numpy()[rows].astype("U"))   # fixed-width (np.load without pickle)
    np.save(H.HO_PREP / "ho_init.npy", H.ego_init(status[rows]))
    E = H.build_exec(rows, shards)
    np.save(H.HO_PREP / "ho_exec.npy", E)
    lat = np.load(H.LAT16, mmap_mode="r")
    fin_head = np.isfinite(np.asarray(lat[rows, 0])).reshape(len(rows), -1).all(1)
    tmp = H.HO_PREP / "ho_sim.tmp.npy"
    sim = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float64, shape=(len(rows), OC.N96, 41, 6))
    fin = np.zeros((len(rows), H.N97), bool)
    fin[:, OC.COL_HEAD] = fin_head
    pos = {int(r): q for q, r in enumerate(rows)}
    ts = time.time()
    with Pool(a.workers, initializer=init, maxtasksperchild=2000) as pool:
        for k, (r, X, f) in enumerate(pool.imap_unordered(sim_row, rows.tolist(), chunksize=16)):
            q = pos[r]
            sim[q] = X
            fin[q, :OC.N96] = f
            if k % 2000 == 0:
                print(f"{time.strftime('%F %T')} sim {k}/{len(rows)} ({time.time() - ts:.0f} s)", flush=True)
    sim.flush()
    del sim
    os.replace(tmp, H.HO_PREP / "ho_sim.npy")
    np.save(H.HO_PREP / "ho_fin.npy", fin)
    rep["sim"] = dict(shape=[len(rows), OC.N96, 41, 6], dtype="float64", sec=time.time() - ts,
                      n_nonfinite_traj=int((~fin[:, :OC.N96]).sum()), n_nonfinite_head=int((~fin_head).sum()))
    # tracks of the official NC / TTC failures (UNKNOWN bucket)
    rs = set(rows.tolist())
    parts = []
    for g, off in (("par", 0), ("var", OC.N_PAR)):
        t = shards[g]
        t = t[t["row"].isin(rs) & ((t["nc"] < 1) | (t["ttc"] < 1))]
        parts.append(t.assign(col=t["k"].astype(int) + off)[["row", "col", "nc", "ttc", "nc_track", "nc_obj_type",
                                                              "ttc_track", "nc_time_idx", "ttc_time_idx"]])
    tr = pd.concat(parts, ignore_index=True)
    tr.to_parquet(H.HO_PREP / "ho_tracks.parquet", index=False)
    rep.update(tracks_rows=int(len(tr)), tracks_tokens=int(tr["row"].nunique()),
               label_sources={g: str(p) for g, p in H.LAB_NPY.items()},
               label_shards={g: str(p) for g, p in H.LAB_SHARD_DIRS.items()},
               traj_sources=dict(par=str(OC.PACK / "cand.npy"), var=str(H.TRAJ80), head=str(H.LAT16) + " [:, 0]"),
               finished=H.kst(), sec=time.time() - t0)
    (H.HO_PREP / "prep.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "lqr_parity_f1"}, indent=1))


if __name__ == "__main__":
    main()
