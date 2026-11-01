#!/usr/bin/env python3
"""H8 / H9 analysis: arm means and paired log-cluster bootstrap contrasts (10,000 draws, seed 0, percentile 95 %,
136 navtest logs; points x100) from batched-scorer parquets and official CSVs, plus draft-level geometry diffs between
inference paths.  -> pipeline.json (stdout summary).  Read-only on every existing file."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/external-user/yongjae/SSR")
SSD = Path("/home/external-user/ssd/yongjae_refiner")
E0R = SSD / "e0_teacher_refine"
DG = SSD / "stageE_diag"
PL = DG / "pipeline"
OUT = REPO / "report/refiner_T/stageE_diag/pipeline.json"
MET = ["pdms", "nc", "dac", "ddc", "ttc", "ep", "comfort"]
CSVCOLS = {"pdms": "score", "nc": "no_at_fault_collisions", "dac": "drivable_area_compliance",
           "ddc": "driving_direction_compliance", "ttc": "time_to_collision_within_bound", "ep": "ego_progress",
           "comfort": "comfort"}


def boot(diff, logs, n=10000, seed=0):
    codes, inv = np.unique(logs, return_inverse=True)
    s = np.bincount(inv, weights=diff)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(codes), size=(n, len(codes)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * float(np.percentile(est, 2.5)), 3), round(100 * float(np.percentile(est, 97.5)), 3)]


def pq(path, k=0):
    s = pd.read_parquet(path)
    s = s[s.k == k]
    assert (s.error.isna() | (s.error == "")).all() if "error" in s else True
    return s.set_index("token")[MET + ["log"]].sort_index()


def csv(path):
    d = pd.read_csv(path)
    d = d[d.token != "average"]
    d = d.set_index("token")
    out = d[[CSVCOLS[m] for m in MET]].astype(float)
    out.columns = MET
    return out.sort_index()


def _pred(p):
    z = np.load(p)
    t = z["tokens"].astype(str)
    x = z["tau1"]
    x = x[:, 0] if x.ndim == 4 else x
    t0 = z["tau0"]
    t0 = t0[:, 0] if t0.ndim == 4 else t0
    return {s: i for i, s in enumerate(t)}, x, t0, z


def geom(a, b):
    ia, xa, ta, _ = _pred(a)
    ib, xb, tb, _ = _pred(b)
    toks = [t for t in ia if t in ib]
    i = np.array([ia[t] for t in toks])
    j = np.array([ib[t] for t in toks])
    d = np.linalg.norm(xa[i][..., :2] - xb[j][..., :2], axis=-1).max(1)
    return {"n": len(toks), "fed_drafts_equal": bool(np.array_equal(ta[i], tb[j])), "max_m": float(d.max()),
            "p99_m": float(np.quantile(d, .99)), "mean_m": float(d.mean()), "frac_gt_1cm": float((d > 0.01).mean()),
            "bitwise_equal_frac": float((xa[i] == xb[j]).all((1, 2)).mean())}


def checks(A):
    import sys
    sys.path.insert(0, str(REPO))
    import torch
    from navsim.agents.para_ssr.refiner import data as RD
    from navsim.agents.para_ssr.refiner.decoder import decode
    out = {}
    # (1) ego inputs: E2 eval path (status_feature -> e2e.ego_inputs, recorded in the dump) vs packed navtest (stage T)
    z = np.load(DG / "e2_navtest_dump.npz")
    P = RD.PackedSplit("navtest")
    idx = {t: i for i, t in enumerate(P.index.token.astype(str).values)}
    rows = np.array([idx[t] for t in z["tokens"].astype(str)])
    e = {}
    for k in ("v0", "a0", "eds"):
        e[f"{k}_max_abs_diff"] = float(np.abs(z[k].astype(np.float64) - np.asarray(P.arrays[k])[rows].astype(np.float64)).max())
    e["cmd_equal_frac"] = float((z["cmd"].astype(int) == np.asarray(P.arrays["cmd"])[rows].astype(int)).mean())
    e["status_feature_vs_packed_eds_max_abs_diff"] = float(np.abs(z["status_feature"][:, 4:].astype(np.float64)
                                                                  - np.asarray(P.arrays["eds"])[rows]).max())
    out["ego_inputs"] = e
    # (2) straight-through slope: forward independent of lon_st_slope; CPU re-decode == stored tau_final
    t = torch.tensor(z["tau0"]).float()
    zl, wl, v0 = (torch.tensor(z[k]).float() for k in ("z_lon", "w_lat", "v0"))
    r0 = decode(t, zl, wl, v0=v0, mode="A", lon_st_slope=0.0)["traj"].numpy()
    r1 = decode(t, zl, wl, v0=v0, mode="A", lon_st_slope=0.1)["traj"].numpy()
    out["decode_slope"] = {"slope0_vs_0.1_max_abs": float(np.abs(r0 - r1).max()),
                           "cpu_redecode_vs_stored_tau_final_max_abs": float(np.abs(r0 - z["tau_final"]).max())}
    # (3) inference-path geometry: fp32 vs fp16 (stage-T path), stage-E code path vs stage-T path
    out["geometry"] = {
        "E0+R_T4 fp32 vs fp16": geom(PL / "E0_RT4_fp32/pred.npz", E0R / "R_T4/pred.npz"),
        "E0+R_M4 fp32 vs fp16": geom(PL / "E0_RM4_fp32/pred.npz", E0R / "R_M4/pred.npz"),
        "E0+R_T4 stageE-path(fp32) vs stageT-path fp32": geom(PL / "E0_RT4_stageE/pred.npz", PL / "E0_RT4_fp32/pred.npz"),
        "E0+R_M4 stageE-path(fp32) vs stageT-path fp32": geom(PL / "E0_RM4_stageE/pred.npz", PL / "E0_RM4_fp32/pred.npz"),
        "E2t0+R_T4 stageE-path(fp32) vs stageT-path fp16": geom(PL / "E2t0_RT4_stageE/pred.npz", PL / "E2t0_RT4/pred.npz"),
        "E2t0+R_M4 stageE-path(fp32) vs stageT-path fp16": geom(PL / "E2t0_RM4_stageE/pred.npz", PL / "E2t0_RM4/pred.npz"),
    }
    # (4) refiner activity on each draft set
    act = {}
    for n, p in (("E0+R_T4", E0R / "R_T4/pred.npz"), ("E0+R_M4", E0R / "R_M4/pred.npz"),
                 ("E0+R_none4", E0R / "R_none4/pred.npz"), ("E2t0+R_T4", PL / "E2t0_RT4/pred.npz"),
                 ("E2t0+R_M4", PL / "E2t0_RM4/pred.npz"), ("E2t0+R_none4", PL / "E2t0_Rnone4/pred.npz")):
        _, x1, x0, q = _pred(p)
        d = np.linalg.norm(x1[..., :2] - x0[..., :2], axis=-1).max(1)
        act[n] = {"lon_live": float(q["lon_live"].mean()), "mean_max_disp_m": float(d.mean()),
                  "frac_disp_gt_0.5m": float((d > 0.5).mean()), "mean_abs_e_lat2": float(np.abs(q["e_lat"][:, 0, 2:]).mean())}
    d = np.linalg.norm(z["tau_final"][..., :2] - z["tau0"][..., :2], axis=-1).max(1)
    act["E2 student (E2t0 -> tau_final)"] = {"lon_live": float(z["lon_live"].mean()), "mean_max_disp_m": float(d.mean()),
                                             "frac_disp_gt_0.5m": float((d > 0.5).mean()),
                                             "mean_abs_e_lat2": float(np.abs(z["e_lat"][:, 2:]).mean())}
    out["refiner_activity"] = act
    # (5) the tokens whose E0 multipliers differ between the pkl (bs 8 dump) and the official CSV (bs 1)
    a, b = A["E0_pkl"], A["E0_csv"]
    flip = [(t, {m: [float(b.at[t, m]), float(a.at[t, m])] for m in ("nc", "dac", "ttc", "pdms")})
            for t in a.index if any(a.at[t, m] != b.at[t, m] for m in ("nc", "dac", "ddc", "ttc", "comfort"))]
    out["E0_pkl_vs_csv_flips (csv, pkl)"] = dict(flip)
    for n in ("E0+R_T4_fp16", "E0+R_M4_fp16"):
        out[f"{n} on flip tokens"] = {t: float(A[n].at[t, "pdms"]) for t, _ in flip}
    return out


def main():
    A = {}
    A["E0_csv"] = csv(REPO / "work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv")
    A["E2_csv"] = csv(REPO / "work_dirs/eval/stageE_E2/2026.09.30.20.42.16.csv")
    A["E2t0_csv"] = csv(REPO / "work_dirs/eval/stageE_E2_tau0/2026.09.30.20.34.56.csv")
    A["E0_pkl"] = pq(E0R / "E0/scores.parquet")
    for n in ("R_T4", "R_M4", "R_none4"):
        A[f"E0+{n}_fp16"] = pq(E0R / f"{n}/scores.parquet")
    A["E2t0_pkl"] = pq(DG / "scores.parquet", 0)
    A["E2_pkl"] = pq(DG / "scores.parquet", 1)
    for grp in ("e0", "e2t0", "h9a", "h9b"):
        p = PL / f"score_{grp}"
        if (p / "scores.parquet").exists():
            arms = json.loads((p / "arms.json").read_text())["k"]
            for k, n in arms.items():
                A[n] = pq(p / "scores.parquet", int(k))
    logs = A["E0_pkl"]["log"]
    toks = logs.index
    for n, d in A.items():
        assert len(d) == 12146 and d.index.equals(toks), (n, len(d))
    L = logs.values
    res = {"arms": {}, "contrasts": {}}
    for n, d in A.items():
        res["arms"][n] = {m: round(100 * float(d[m].mean()), 3) for m in MET}
        res["arms"][n]["n_pdms0"] = int((d.pdms == 0).sum())

    def con(a, b, name=None):
        if a not in A or b not in A:
            return
        r = {}
        for m in ("pdms", "nc", "dac", "ttc", "ep"):
            df = (A[a][m] - A[b][m]).values
            r[m] = {"diff": round(100 * float(df.mean()), 3), "ci95": boot(df, L)}
        pa, pb = A[a].pdms.values, A[b].pdms.values
        r["fixed_0_to_pos"] = int(((pb == 0) & (pa > 0)).sum())
        r["broken_pos_to_0"] = int(((pb > 0) & (pa == 0)).sum())
        r["n_tokens_pdms_diff_gt_1e-6"] = int((np.abs(pa - pb) > 1e-6).sum())
        res["contrasts"][name or f"{a} - {b}"] = r

    C = [("E0_pkl", "E0_csv"), ("E2_pkl", "E2_csv"), ("E2t0_pkl", "E2t0_csv"),
         ("E0+R_T4_fp32", "E0+R_T4_fp16"), ("E0+R_M4_fp32", "E0+R_M4_fp16"),
         ("E0+R_T4_stageE", "E0+R_T4_fp32"), ("E0+R_M4_stageE", "E0+R_M4_fp32"),
         ("E0+R_T4_fp16", "E0_pkl"), ("E0+R_M4_fp16", "E0_pkl"),
         ("E2_pkl", "E0_pkl"), ("E2_pkl", "E0+R_T4_fp16"), ("E2_pkl", "E0+R_M4_fp16"),
         ("E2t0_pkl", "E0_pkl"),
         ("E2t0+R_T4_fp16", "E2t0_pkl"), ("E2t0+R_M4_fp16", "E2t0_pkl"), ("E2t0+R_none4_fp16", "E2t0_pkl"),
         ("E2_pkl", "E2t0_pkl"),
         ("E2_pkl", "E2t0+R_T4_fp16"), ("E2_pkl", "E2t0+R_M4_fp16"),
         ("E2t0+R_T4_fp16", "E0+R_T4_fp16"), ("E2t0+R_M4_fp16", "E0+R_M4_fp16"),
         ("E2t0+R_T4_stageE", "E2t0+R_T4_fp16"), ("E2t0+R_M4_stageE", "E2t0+R_M4_fp16"),
         ("E0ep28_pkl", "E0_pkl"), ("E2ep28_pkl", "E2_pkl"), ("E2t0ep28_pkl", "E2t0_pkl"),
         ("E2ep28_pkl", "E2t0ep28_pkl"), ("E2ep28_pkl", "E0ep28_pkl"), ("E2t0ep28_pkl", "E0ep28_pkl"),
         ("E0ep28+R_T4_fp16", "E0ep28_pkl"), ("E2ep28_pkl", "E0ep28+R_T4_fp16")]
    for a, b in C:
        con(a, b)
    res["checks"] = checks(A)
    OUT.write_text(json.dumps(res, indent=1))
    for n, v in res["arms"].items():
        print(f"{n:22s} PDMS {v['pdms']:.3f} NC {v['nc']:.2f} DAC {v['dac']:.2f} TTC {v['ttc']:.2f} EP {v['ep']:.2f} "
              f"pdms0 {v['n_pdms0']}")
    for n, r in res["contrasts"].items():
        p = r["pdms"]
        print(f"{n:40s} {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] fixed {r['fixed_0_to_pos']} "
              f"broken {r['broken_pos_to_0']} ndiff {r['n_tokens_pdms_diff_gt_1e-6']}")


if __name__ == "__main__":
    main()
