"""Descriptive (NOT decisive) breakdown of the stage-T dev comparison at the selected theta (0 for both arms).
Reads runs/stageT_{T,none}_fold0_seed0/eval_dev/report_rows.parquet; writes decision_diag/diag.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/refiner")
import stageT_decision as SD
RUNS = Path("/home/external-user/ssd/yongjae_refiner/runs")
TAG = sys.argv[1] if len(sys.argv) > 1 else "stageT"
SELF = sys.argv[2] if len(sys.argv) > 2 else "report/refiner_T/crossfit_selection.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "report/refiner_T/decision_diag/diag.json"
sel = json.load(open(SELF))
F = {}
for arm in ("T", "none"):
    th = sel["arms"][arm]["candidates"][TAG]["theta"]
    F[arm] = SD.final_outcomes(SD.load_rows(RUNS / f"{TAG}_{arm}_fold0_seed0" / "eval_dev"), th).set_index(["token", "k"])
idx = F["T"].index.intersection(F["none"].index)
T, N = F["T"].loc[idx], F["none"].loc[idx]
out = {"n": int(len(idx))}
def rate(x): return round(100 * float(np.mean(x)), 3)
for m in ("nc", "ttc", "dac", "ddc", "comfort"):
    out[f"fail_{m}_pp"] = dict(orig=rate(T[f"{m}_orig"] < 1), T=rate(T[f"{m}_final"] < 1), none=rate(N[f"{m}_final"] < 1))
out["ep_mean"] = dict(orig=float(T.ep_orig.mean()), T=float(T.ep_final.mean()), none=float(N.ep_final.mean()))
out["pdms_mean"] = dict(orig=float(T.pdms_orig.mean()), T=float(T.pdms_final.mean()), none=float(N.pdms_final.mean()))
# EP among drafts that pass all multiplicative metrics before and after (pure slowdown cost)
mult = lambda D, s: (D[f"nc_{s}"] >= 1) & (D[f"dac_{s}"] >= 1) & (D[f"ddc_{s}"] >= 1)
both = mult(T, "orig") & mult(T, "final") & mult(N, "final")
out["ep_loss_points_on_always_passing"] = dict(n=int(both.sum()), T=float(100 * (T.ep_orig - T.ep_final)[both].mean()),
                                               none=float(100 * (N.ep_orig - N.ep_final)[both].mean()))
# NC/TTC transitions: fixed by one arm only
fo = T.fail_ncttc_orig > 0
out["ncttc_orig_fail_n"] = int(fo.sum())
out["ncttc_fixed"] = dict(T=int((fo & (T.fail_ncttc == 0)).sum()), none=int((fo & (N.fail_ncttc == 0)).sum()),
                          both=int((fo & (T.fail_ncttc == 0) & (N.fail_ncttc == 0)).sum()),
                          T_only=int((fo & (T.fail_ncttc == 0) & (N.fail_ncttc > 0)).sum()),
                          none_only=int((fo & (T.fail_ncttc > 0) & (N.fail_ncttc == 0)).sum()))
out["ncttc_new"] = dict(T=int((~fo & (T.fail_ncttc > 0)).sum()), none=int((~fo & (N.fail_ncttc > 0)).sum()))
dfo = T.dac_orig < 1
out["dac_fixed"] = dict(orig_fail=int(dfo.sum()), T=int((dfo & (T.dac_final >= 1)).sum()), none=int((dfo & (N.dac_final >= 1)).sum()))
fam = {}
for f, g in T.groupby("family"):
    n = N.loc[g.index]
    fam[f] = dict(n=len(g), ncttc_orig=rate(g.fail_ncttc_orig), ncttc_T=rate(g.fail_ncttc), ncttc_none=rate(n.fail_ncttc),
                  dac_orig=rate(g.dac_orig < 1), dac_T=rate(g.dac_final < 1), dac_none=rate(n.dac_final < 1),
                  ep_T=round(float(g.ep_final.mean()), 4), ep_none=round(float(n.ep_final.mean()), 4))
out["by_family"] = fam
Path(OUT).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# ---- part 2: what the arms output (displacement of tau1 vs tau0) and NC / TTC paired log-bootstrap CIs (descriptive)
P = {a: np.load(RUNS / f"{TAG}_{a}_fold0_seed0" / "eval_dev" / "pred.npz") for a in ("T", "none")}
out2 = {}
for a, d in P.items():
    v = d["draft_valid"]
    t0, t1 = d["tau0"][v], d["tau1"][v]
    fam = d["family"][v]
    # longitudinal shortfall at 4 s (arc length difference) and max lateral displacement
    arc = lambda t: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(t[:, :1, :2]), t[:, :, :2]], 1), axis=1), axis=-1).sum(1)
    short = arc(t0) - arc(t1)
    lat = np.linalg.norm(t1[:, :, :2] - t0[:, :, :2], axis=-1).max(1)
    rec = dict(mean_arc_short_m=float(short.mean()), frac_short_gt_1m=float((short > 1).mean()),
               mean_max_disp_m=float(lat.mean()), frac_disp_gt_0p3m=float((lat > 0.3).mean()), by_family={})
    for f in np.unique(fam):
        m = fam == f
        rec["by_family"][int(f)] = dict(arc_short_m=round(float(short[m].mean()), 3), max_disp_m=round(float(lat[m].mean()), 3))
    out2[a] = rec
rng = np.random.default_rng(0)
logs = T.log.values; ul, inv = np.unique(logs, return_inverse=True)
def boot(x, B=10000):
    s = np.bincount(inv, weights=x, minlength=len(ul)); c = np.bincount(inv, minlength=len(ul))
    w = rng.multinomial(len(ul), np.full(len(ul), 1 / len(ul)), size=B)
    est = (w @ s) / (w @ c)
    return dict(mean=float(100 * x.mean()), lo=float(100 * np.quantile(est, .025)), hi=float(100 * np.quantile(est, .975)))
out2["nc_fail_T_minus_none_pp"] = boot((T.nc_final < 1).astype(float).values - (N.nc_final < 1).astype(float).values)
out2["ttc_fail_T_minus_none_pp"] = boot((T.ttc_final < 1).astype(float).values - (N.ttc_final < 1).astype(float).values)
out2["dac_fail_T_minus_none_pp"] = boot((T.dac_final < 1).astype(float).values - (N.dac_final < 1).astype(float).values)
out["part2"] = out2
Path(OUT).write_text(json.dumps(out, indent=1))
print(json.dumps(out2, indent=1))
