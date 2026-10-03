#!/usr/bin/env python3
"""Stage-E diagnostics (H3 under-correction / H4 direction): counterfactual control swaps on E2's own navtest tau0.

Inputs (read only)
  student : /home/external-user/ssd/yongjae_refiner/stageE_diag/e2_navtest_dump.npz (E2 last.ckpt; tau0, z_lon, w_lat, v0)
  teachers: <same_draft>/{R_T4,R_M4}/pred.npz = tools/refiner/refine_external_drafts.py predict of the frozen run-4 R_T /
            R_M on e2_tau0_navtest_trajectories.pkl (theta 0, ckpt_best, fp16 autocast), produced by the sibling
            same-draft analysis; reused here (pkl sha256 checked).
Every arm is decode(tau0, z_lon, w_lat, v0, mode='A') on CPU fp32 with the student's v0 (== the packed v0 to 1e-6):
  S          student as evaluated (checked == dump tau_final)
  T, M       teacher controls re-decoded (checked against the teacher pred tau1)
  TlonSlat   z_lon of R_T, w_lat of the student      SlonTlat  z_lon of the student, w_lat of R_T
  MlonSlat   z_lon of R_M, w_lat of the student      SlonMlat  z_lon of the student, w_lat of R_M
  Sx2, Sx3   student corrections amplified k x in the decoded-control space: Q' = k Q (pre-clamp c' = k c, so the
             decoded c_lon' = k c_lon; a non-live draft stays non-live), e' = k e; both clipped to 0.999 of their tanh
             bound (A_DEC / A_UP, D_MAX) before inverting to z / w.  The curvature projection alpha is re-applied.
Output: <out>/swaps.npz (tokens, arms, drafts [N, K, 8, 3] f32), <out>/tokens.parquet, <out>/swaps_meta.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.refiner.decoder import (A_DEC, A_UP, D_MAX, decode, lon_c_from_q,  # noqa: E402
                                                    lon_q_from_z)

DIAG = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
SAME = DIAG / "same_draft"
OUT = DIAG / "teacher_on_e2"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dec(tau0, z, w, v0, bs=2048):
    out = []
    for i in range(0, len(tau0), bs):
        d = decode(torch.as_tensor(tau0[i:i + bs]).float(), torch.as_tensor(z[i:i + bs]).float(),
                   torch.as_tensor(w[i:i + bs]).float(), v0=torch.as_tensor(v0[i:i + bs]).float(), mode="A")
        out.append(d["traj"].numpy())
    return np.concatenate(out).astype(np.float32)


def amplify(z, w, k):
    q = lon_q_from_z(torch.as_tensor(z).double()).numpy() * k
    q = np.clip(q, -0.999 * A_DEC, 0.999 * A_UP)
    z2 = np.where(q < 0, np.arctanh(q / A_DEC), np.arctanh(q / A_UP))
    e = np.clip(D_MAX * np.tanh(np.asarray(w, np.float64)) * k, -0.999 * D_MAX, 0.999 * D_MAX)
    return z2.astype(np.float32), np.arctanh(e / D_MAX).astype(np.float32)


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    D = np.load(DIAG / "e2_navtest_dump.npz")
    tokens = D["tokens"].astype(str)
    tau0, v0 = D["tau0"].astype(np.float32), D["v0"].astype(np.float32)
    zS, wS = D["z_lon"].astype(np.float32), D["w_lat"].astype(np.float32)
    pkl_sha = sha(DIAG / "e2_tau0_navtest_trajectories.pkl")
    T = {}
    for name in ("R_T4", "R_M4"):
        m = json.loads((SAME / name / "predict_meta.json").read_text())
        assert m["drafts_pkl_sha256"] == pkl_sha, name
        p = np.load(SAME / name / "pred.npz")
        idx = {t: i for i, t in enumerate(p["tokens"].astype(str))}
        r = np.array([idx[t] for t in tokens])
        assert np.array_equal(p["tau0"][r, 0], tau0), f"{name}: fed draft != E2 tau0"
        T[name] = dict(z=p["z_lon"][r, 0].astype(np.float32), w=p["w_lat"][r, 0].astype(np.float32),
                       tau1=p["tau1"][r, 0].astype(np.float32))
    arms = {}
    arms["S"] = dec(tau0, zS, wS, v0)
    chk = {"S_vs_dump_tau_final_maxabs": float(np.abs(arms["S"] - D["tau_final"]).max())}
    arms["T"] = dec(tau0, T["R_T4"]["z"], T["R_T4"]["w"], v0)
    arms["M"] = dec(tau0, T["R_M4"]["z"], T["R_M4"]["w"], v0)
    chk["T_vs_pred_tau1_maxabs"] = float(np.abs(arms["T"] - T["R_T4"]["tau1"]).max())
    chk["M_vs_pred_tau1_maxabs"] = float(np.abs(arms["M"] - T["R_M4"]["tau1"]).max())
    arms["TlonSlat"] = dec(tau0, T["R_T4"]["z"], wS, v0)
    arms["SlonTlat"] = dec(tau0, zS, T["R_T4"]["w"], v0)
    arms["MlonSlat"] = dec(tau0, T["R_M4"]["z"], wS, v0)
    arms["SlonMlat"] = dec(tau0, zS, T["R_M4"]["w"], v0)
    for k in (2, 3):
        z2, w2 = amplify(zS, wS, k)
        arms[f"Sx{k}"] = dec(tau0, z2, w2, v0)
    names = list(arms)
    drafts = np.stack([arms[n] for n in names], 1)
    assert np.isfinite(drafts).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "swaps.npz", tokens=tokens, arms=np.array(names), drafts=drafts)
    pd.DataFrame({"token": tokens, "log": D["log"].astype(str)}).to_parquet(OUT / "tokens.parquet", index=False)
    meta = dict(arms=names, n=int(len(tokens)), checks=chk, e2_tau0_pkl_sha256=pkl_sha,
                teacher_preds={n: str(SAME / n / "pred.npz") for n in T},
                teacher_pred_sha256={n: sha(SAME / n / "pred.npz") for n in T},
                dump_sha256=sha(DIAG / "e2_navtest_dump.npz"), sec=round(time.time() - t0, 1),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUT / "swaps_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
