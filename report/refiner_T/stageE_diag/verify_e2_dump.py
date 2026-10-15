#!/usr/bin/env python3
"""Verify the E2 navtest dump: batched-scorer scores of [tau0, tau_final] vs the official E2_tau0 / E2 CSVs (per token
and means), plus a few descriptive stats of the student refiner outputs.  -> e2_dump_verify.json (stdout too)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/external-user/yongjae/SSR")
D = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
OUT = REPO / "report/refiner_T/stageE_diag/e2_dump_verify.json"
CSV = {1: REPO / "work_dirs/eval/stageE_E2/2026.09.30.20.42.16.csv",
       0: REPO / "work_dirs/eval/stageE_E2_tau0/2026.09.30.20.34.56.csv"}
E0CSV = REPO / "work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv"
COLS = {"pdms": "score", "nc": "no_at_fault_collisions", "dac": "drivable_area_compliance",
        "ddc": "driving_direction_compliance", "ttc": "time_to_collision_within_bound", "ep": "ego_progress",
        "comfort": "comfort"}

z = np.load(D / "e2_navtest_dump.npz")
s = pd.read_parquet(D / "scores.parquet")
res = {"n_dump": int(len(z["tokens"])), "score_rows": int(len(s)), "score_errors": int((s.k < 0).sum()),
       "rec_ok_all": bool(s.rec_ok.all())}
for k, name in ((0, "tau0"), (1, "tau_final")):
    c = pd.read_csv(CSV[k])
    c = c[c.token != "average"]
    m = s[s.k == k].merge(c, on="token", suffixes=("", "_off"))
    r = {"n_matched": int(len(m)), "n_official": int(len(c)), "official_valid_all": bool(c.valid.all())}
    for ours, off in COLS.items():
        a, b = m[ours].astype(float), m[off].astype(float)
        r[f"{ours}_ours_x100"] = round(100 * a.mean(), 4)
        r[f"{ours}_official_x100"] = round(100 * b.mean(), 4)
        r[f"{ours}_max_abs_diff"] = float((a - b).abs().max())
        r[f"{ours}_n_diff_gt_1e-9"] = int(((a - b).abs() > 1e-9).sum())
    res[name] = r
# student refiner descriptives
tf, t0 = z["tau_final"], z["tau0"]
disp = np.linalg.norm(tf[..., :2] - t0[..., :2], axis=-1)
res["student"] = {
    "frac_tau_final_eq_tau0": float((tf == t0).all((1, 2)).mean()),
    "mean_disp_m": float(disp.mean()), "mean_max_disp_m": float(disp.max(1).mean()),
    "frac_max_disp_gt_0.5m": float((disp.max(1) > 0.5).mean()),
    "lon_live_frac": float(z["lon_live"].mean()), "alpha_mean": float(z["alpha"].mean()),
    "beta_mean": float(z["beta"].mean()), "lat_on_frac": float(z["lat_on"].mean()),
    "p_g_mean": float(z["p_g"].mean()), "p_g_ge_0.5_frac": float((z["p_g"] >= 0.5).mean()),
    "z_lon_mean": float(z["z_lon"].mean()), "z_lon_frac_pos": float((z["z_lon"] > 0).mean()),
    "c_lon2_mean": float(z["c_lon"][:, 2:].mean()), "abs_e_lat2_mean": float(np.abs(z["e_lat"][:, 2:]).mean()),
}
# E2 tau0 vs E0 draft
with open(REPO / "work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl", "rb") as f:
    import pickle
    e0 = pickle.load(f)["trajectories"]
e0a = np.stack([e0[t] for t in z["tokens"]])
dd = np.linalg.norm(e0a[..., :2] - t0[..., :2], axis=-1)
res["e2_tau0_vs_e0_draft"] = {"mean_l2_m": float(dd.mean()), "mean_final_l2_m": float(dd[:, -1].mean()),
                              "frac_final_l2_gt_1m": float((dd[:, -1] > 1).mean())}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
