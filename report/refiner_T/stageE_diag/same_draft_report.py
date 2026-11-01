#!/usr/bin/env python
"""Same-draft test: is E2 < E0 + R_T / R_M because the student refiner is weaker (H1) or because E2's drafts differ (H2)?

Arms (all scored with tools/refiner/score_trajectories.py, paired on the 12,146 navtest tokens):
  E0            : PARA-SSR interaction_final pkl (e0_teacher_refine/E0)
  E0+R_X        : frozen stage-T refiner X on E0's drafts (e0_teacher_refine/R_X), theta 0
  E2_tau0       : E2 planner draft (stageE_diag/scores.parquet k=0; == official stageE_E2_tau0 csv)
  E2_final      : E2 student refiner output (k=1; == official stageE_E2 csv)
  E2tau0+R_X    : frozen refiner X on E2's tau0 drafts (stageE_diag/same_draft/R_X), theta 0
CIs: log-cluster paired bootstrap (stageE_compare.boot semantics: resample logs, 10,000 draws, seed 0, percentile 95 %);
ratios (share of the gap) use the same log resamples (ratio of bootstrap means).
fail_any = nc < 1 | dac < 1 | ddc < 1 | ttc < 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(REPO / "tools/refiner"))
from liveness import arc_len  # noqa: E402

E0B = Path("/home/external-user/ssd/yongjae_refiner/e0_teacher_refine")
DG = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
SD = DG / "same_draft"
OUT = REPO / "report/refiner_T/stageE_diag"
M = {"PDMS": "pdms", "NC": "nc", "DAC": "dac", "DDC": "ddc", "TTC": "ttc", "EP": "ep", "C": "comfort"}
CORE = ("nc", "dac", "ddc", "ttc")
REF = ("R_T4", "R_M4", "R_none4", "R_T3")
NB = 10000


def sc(path, k=0):
    s = pd.read_parquet(path)
    assert (s["error"].fillna("").astype(str) == "").all(), path
    s = s[s.k == k].set_index("token")
    return s[list(M.values()) + ["log"]]


def load():
    A = {"E0": sc(E0B / "E0/scores.parquet")}
    for r in REF:
        A[f"E0+{r}"] = sc(E0B / r / "scores.parquet")
    A["E2_tau0"] = sc(DG / "scores.parquet", 0)
    A["E2_final"] = sc(DG / "scores.parquet", 1)
    for r in REF:
        A[f"E2+{r}"] = sc(SD / r / "scores.parquet")
    tok = sorted(set.intersection(*[set(d.index) for d in A.values()]))
    A = {k: v.loc[tok] for k, v in A.items()}
    logs = A["E0"]["log"].values
    for v in A.values():
        assert (v["log"].values == logs).all()
    return A, np.asarray(tok), logs


class Boot:
    def __init__(self, logs, n=NB, seed=0):
        codes, self.inv = np.unique(logs, return_inverse=True)
        self.c = np.bincount(self.inv).astype(float)
        self.idx = np.random.default_rng(seed).integers(0, len(codes), size=(n, len(codes)))
        self.cs = self.c[self.idx].sum(1)

    def draws(self, x):
        s = np.bincount(self.inv, weights=x)
        return s[self.idx].sum(1) / self.cs

    def ci(self, x):
        d = self.draws(x)
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    def ratio_ci(self, num, den):
        r = self.draws(num) / self.draws(den)
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def fails(d, keys=CORE):
    return np.any(np.stack([d[k].values < 1 for k in keys], -1), -1)


def contrast(bt, A, B):
    r = {}
    for n, c in M.items():
        d = A[c].values - B[c].values
        r[n] = dict(diff=float(d.mean()), ci95=bt.ci(d))
    fa, fb = fails(A), fails(B)
    pa, pb = fails(A, CORE + ("comfort",)), fails(B, CORE + ("comfort",))
    r.update(fixed=int((fb & ~fa).sum()), new_fail=int((~pb & pa).sum()), fail_any_A=int(fa.sum()),
             fail_any_B=int(fb.sum()), helped=int((A.pdms.values > B.pdms.values).sum()),
             harmed=int((A.pdms.values < B.pdms.values).sum()))
    for k in ("nc", "dac", "ttc"):
        r[f"fixed_{k}"] = int(((B[k].values < 1) & (A[k].values >= 1)).sum())
        r[f"new_{k}"] = int(((B[k].values >= 1) & (A[k].values < 1)).sum())
    return r


def mod_stats():
    """correction size on E2's tau0 drafts: student (dump) vs frozen refiners (pred.npz), same drafts."""
    D = np.load(DG / "e2_navtest_dump.npz")
    idx = {t: i for i, t in enumerate(D["tokens"].astype(str))}
    out = {}

    def blk(tau0, tau1, c, e, z, live):
        disp = np.linalg.norm(tau1[..., :2] - tau0[..., :2], axis=-1).max(-1)
        short = arc_len(tau0.astype(np.float64)) - arc_len(tau1.astype(np.float64))
        return dict(lon_live=float(live.mean()), short_mean_m=float(short.mean()), short_gt_0p5=float((short > .5).mean()),
                    lat_final_gt_0p1=float((np.abs(e[:, -1]) > .1).mean()), e_lat_abs_mean=float(np.abs(e[:, 2:]).mean()),
                    c_lon_mean=float(c[:, 2:].mean()), disp_mean_m=float(disp.mean()), disp_gt_0p5=float((disp > .5).mean()),
                    z_lon_mean=float(z.mean()))
    P = {r: np.load(SD / r / "pred.npz") for r in REF}
    j = np.array([idx[t] for t in P["R_T4"]["tokens"]])
    out["student"] = blk(D["tau0"][j], D["tau_final"][j], D["c_lon"][j], D["e_lat"][j], D["z_lon"][j],
                         D["lon_live"][j].astype(bool))
    for r in REF:
        p = P[r]
        assert (p["tokens"] == P["R_T4"]["tokens"]).all()
        assert np.array_equal(p["tau0"][:, 0], D["tau0"][j])
        out[r] = blk(p["tau0"][:, 0], p["tau1"][:, 0], p["c_lon"][:, 0], p["e_lat"][:, 0], p["z_lon"][:, 0],
                     p["lon_live"][:, 0].astype(bool))
    # KD-space agreement (12 decoded controls c_lon[2:], e_lat[2:], the E2 KD space) on navtest
    ctl = lambda c, e: np.concatenate([c[:, 2:], e[:, 2:]], 1)
    S = ctl(D["c_lon"][j], D["e_lat"][j])
    Ct = ctl(P["R_T4"]["c_lon"][:, 0], P["R_T4"]["e_lat"][:, 0])
    Cm = ctl(P["R_M4"]["c_lon"][:, 0], P["R_M4"]["e_lat"][:, 0])
    kd = lambda X: dict(l1_T=float(np.abs(X - Ct).mean()), l1_M=float(np.abs(X - Cm).mean()),
                        kd=float((np.abs(X - Ct).mean() + np.abs(X - Cm).mean()) / 2))
    mid = (Ct + Cm) / 2
    agree = {"student": kd(S), "zero_correction": kd(np.zeros_like(S)), "teacher_midpoint": kd(mid),
             "T_vs_M_l1": float(np.abs(Ct - Cm).mean())}
    lt, lm, ls = (P["R_T4"]["lon_live"][:, 0] > 0), (P["R_M4"]["lon_live"][:, 0] > 0), (D["lon_live"][j] > 0)
    at, am, as_ = (np.abs(Ct[:, -1]) > .1), (np.abs(Cm[:, -1]) > .1), (np.abs(S[:, -1]) > .1)
    agree["lon_live"] = dict(T=float(lt.mean()), M=float(lm.mean()), both=float((lt & lm).mean()),
                             exactly_one=float((lt ^ lm).mean()), S=float(ls.mean()),
                             S_given_both=float(ls[lt & lm].mean()), S_given_one=float(ls[lt ^ lm].mean()),
                             S_given_none=float(ls[~lt & ~lm].mean()))
    agree["lat_final_gt_0p1"] = dict(T=float(at.mean()), M=float(am.mean()), both=float((at & am).mean()),
                                     exactly_one=float((at ^ am).mean()), S=float(as_.mean()),
                                     S_given_both=float(as_[at & am].mean()), S_given_one=float(as_[at ^ am].mean()),
                                     S_given_none=float(as_[~at & ~am].mean()))
    for k, nm in ((5, "c_lon_7"), (11, "e_lat_7")):
        x, y = mid[:, k], S[:, k]
        agree[f"shrink_{nm}"] = dict(slope_S_on_mid=float((x * y).sum() / (x * x).sum()), corr=float(np.corrcoef(x, y)[0, 1]))
    groups = dict(both_active=(lt & lm) | (at & am), one_active=((lt ^ lm) | (at ^ am)) & ~((lt & lm) | (at & am)))
    groups["none_active"] = ~(groups["both_active"] | groups["one_active"])
    return out, agree, P["R_T4"]["tokens"].astype(str), groups


def main():
    A, tok, logs = load()
    bt = Boot(logs)
    n = len(tok)
    res = {"n_tokens": n, "n_logs": int(len(set(logs))), "bootstrap": "log-cluster paired, 10,000 draws, seed 0, pct 95 %",
           "arms": {k: {m: float(v[c].mean()) for m, c in M.items()} | {"fail_any": int(fails(v).sum())}
                    for k, v in A.items()}}
    for r in REF:
        res.setdefault("predict_meta", {})[r] = json.loads((SD / r / "predict_meta.json").read_text())
    C = {}
    pairs = [("E2_final", "E2_tau0"), ("E2_tau0", "E0"), ("E2_final", "E0")]
    for r in REF:
        pairs += [(f"E2+{r}", "E2_final"), (f"E2+{r}", "E2_tau0"), (f"E0+{r}", "E0"), (f"E0+{r}", "E2_final"),
                  (f"E0+{r}", f"E2+{r}")]
    pairs += [("E2+R_T4", "E2+R_M4")]
    for a, b in pairs:
        C[f"{a} - {b}"] = contrast(bt, A[a], A[b])
    # gain on E2 drafts minus gain on E0 drafts (are E2 drafts "harder to refine"?)
    for r in REF:
        d = (A[f"E2+{r}"].pdms.values - A["E2_tau0"].pdms.values) - (A[f"E0+{r}"].pdms.values - A["E0"].pdms.values)
        C[f"gain({r})@E2 - gain({r})@E0"] = {"PDMS": dict(diff=float(d.mean()), ci95=bt.ci(d))}
    d = (A["E2_final"].pdms.values - A["E2_tau0"].pdms.values) - (A["E0+R_T4"].pdms.values - A["E0"].pdms.values)
    C["gain(student)@E2 - gain(R_T4)@E0"] = {"PDMS": dict(diff=float(d.mean()), ci95=bt.ci(d))}
    res["contrasts"] = C

    # gap decomposition, per teacher X: gap = (E0+X) - E2_final
    dec = {}
    for r in ("R_T4", "R_M4", "R_T3"):
        g = A[f"E0+{r}"].pdms.values - A["E2_final"].pdms.values
        draft_raw = A["E0"].pdms.values - A["E2_tau0"].pdms.values                  # E0 - E2_tau0
        inter = ((A[f"E0+{r}"].pdms.values - A["E0"].pdms.values)
                 - (A[f"E2+{r}"].pdms.values - A["E2_tau0"].pdms.values))           # gain_X@E0 - gain_X@E2
        refi = A[f"E2+{r}"].pdms.values - A["E2_final"].pdms.values                 # teacher vs student, same drafts
        draft_via_X = draft_raw + inter                                             # (E0+X) - (E2+X)
        assert np.allclose(draft_raw + inter + refi, g)
        comp = {}
        for nm, x in (("gap", g), ("draft_raw", draft_raw), ("draft_x_refiner_interaction", inter),
                      ("draft_total_through_X", draft_via_X), ("refiner_same_draft", refi)):
            comp[nm] = dict(diff=float(x.mean()), ci95=bt.ci(x))
            if nm != "gap":
                comp[nm]["share_of_gap"] = float(x.mean() / g.mean())
                comp[nm]["share_ci95"] = bt.ratio_ci(x, g)
        dec[r] = comp
    res["gap_decomposition"] = dec

    # failing-token pools
    f0, f2 = fails(A["E0"]), fails(A["E2_tau0"])
    pools = {"E0_fail_only": f0 & ~f2, "both_fail": f0 & f2, "E2tau0_fail_only": ~f0 & f2, "neither": ~f0 & ~f2}
    P = {"counts": {k: int(v.sum()) for k, v in pools.items()},
         "E0_fail": int(f0.sum()), "E2tau0_fail": int(f2.sum()),
         "jaccard": float((f0 & f2).sum() / (f0 | f2).sum())}
    for k in ("nc", "dac", "ttc"):
        a0, a2 = A["E0"][k].values < 1, A["E2_tau0"][k].values < 1
        P[f"{k}_fail"] = dict(E0=int(a0.sum()), E2tau0=int(a2.sum()), both=int((a0 & a2).sum()))
    arms_on = {"E0": ["E0+R_T4", "E0+R_M4", "E0+R_none4"],
               "E2_tau0": ["E2_final", "E2+R_T4", "E2+R_M4", "E2+R_none4", "E2+R_T3"]}
    tab = {}
    for base, arms in arms_on.items():
        fb = fails(A[base])
        pb = fails(A[base], CORE + ("comfort",))
        for a in arms:
            fa = fails(A[a])
            pa = fails(A[a], CORE + ("comfort",))
            row = {"base": base, "fixed": int((fb & ~fa).sum()), "fix_rate": float((fb & ~fa).sum() / fb.sum()),
                   "new_fail": int((~pb & pa).sum()), "net": int((fb & ~fa).sum() - (~fb & fa).sum()),
                   "fail_after": int(fa.sum())}
            dp = A[a].pdms.values - A[base].pdms.values
            for pn, m in pools.items():
                row[f"dPDMS_pts_{pn}"] = float(dp[m].mean() * 100)
                row[f"contrib_pts_{pn}"] = float(dp[m].sum() / n * 100)
                if pn != "neither":
                    fbm = fb & m
                    row[f"fixed_{pn}"] = int((fbm & ~fa).sum())
                    row[f"base_fail_{pn}"] = int(fbm.sum())
            tab[a] = row
    P["refiners"] = tab
    # student vs teacher fix overlap on E2 drafts
    fs, ft, fm = ~fails(A["E2_final"]), ~fails(A["E2+R_T4"]), ~fails(A["E2+R_M4"])
    P["fix_overlap_on_E2tau0_fails"] = dict(
        n=int(f2.sum()), student=int((f2 & fs).sum()), R_T4=int((f2 & ft).sum()), R_M4=int((f2 & fm).sum()),
        student_and_T=int((f2 & fs & ft).sum()), T_only=int((f2 & ft & ~fs).sum()), student_only_vs_T=int((f2 & fs & ~ft).sum()),
        T_or_M=int((f2 & (ft | fm)).sum()), student_not_T_or_M=int((f2 & fs & ~ft & ~fm).sum()),
        T_or_M_not_student=int((f2 & (ft | fm) & ~fs).sum()))
    res["fail_pools"] = P

    # correction size + KD-space agreement, and the teacher-agreement strata
    ms, agree, mtok, groups = mod_stats()
    res["modification_on_E2tau0"] = ms
    res["kd_agreement_navtest"] = agree
    pos = pd.Series(np.arange(n), index=tok)
    gi = pos.loc[mtok].values
    strata = {}
    for gname, gm in groups.items():
        sel = np.zeros(n, bool)
        sel[gi[gm]] = True
        row = {"n": int(sel.sum())}
        for a in ("E2_final", "E2+R_T4", "E2+R_M4", "E2+R_none4"):
            row[a] = float((A[a].pdms.values[sel] - A["E2_tau0"].pdms.values[sel]).mean() * 100)
            row[f"{a}_contrib"] = float((A[a].pdms.values[sel] - A["E2_tau0"].pdms.values[sel]).sum() / n * 100)
        strata[gname] = row
    res["strata_teacher_agreement"] = strata

    # per city (same-draft contrast)
    sp = pd.read_parquet("/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet")
    city_col = next((c for c in ("city", "location", "map_name") if c in sp.columns), None)
    if city_col:
        cm = sp.drop_duplicates("token").set_index("token")[city_col].reindex(tok).values
        pc = {}
        for c in sorted(set(cm) - {None}):
            m = cm == c
            bc = Boot(logs[m])
            row = {"n": int(m.sum())}
            for a, b in (("E2+R_T4", "E2_final"), ("E2+R_M4", "E2_final"), ("E2_tau0", "E0"), ("E0+R_T4", "E2_final")):
                d = A[a].pdms.values[m] - A[b].pdms.values[m]
                row[f"{a} - {b}"] = dict(diff=float(d.mean()), ci95=bc.ci(d))
            pc[str(c)] = row
        res["per_city"] = pc
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "same-draft.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("arms", "gap_decomposition")}, indent=1))




def train_log_summary(path=Path("/home/external-user/yongjae/SSR/work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0/stageE_steps.jsonl")):
    """per-epoch means of the stage-E training diagnostics (added to same-draft.json under 'train_log')."""
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    cols = ["kd/loss", "kd/l1_0", "kd/l1_1", "kd/w_ema", "kd/weighted", "ref/L_sur_weighted", "ref/live",
            "kd/teacher_live_0", "kd/teacher_live_1", "ref/dist_final_tau0", "ref/zdead", "ref/frac_draft_human",
            "ref/gshare_e0", "ref/gshare_sur", "ref/gshare_kd", "ref/vshare_sur", "ref/vshare_kd", "loss_plan_reg",
            "ref/t_col", "ref/t_dac", "ref/t_ttc"]
    g = df.groupby("epoch")[[c for c in cols if c in df]].mean()
    return {int(e): {k: float(v) for k, v in row.items()} for e, row in g.loc[[0, 4, 5, 10, 20, 29]].iterrows()}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--train-log":
    p = OUT / "same-draft.json"
    r = json.loads(p.read_text())
    r["train_log"] = train_log_summary()
    p.write_text(json.dumps(r, indent=1))
    print(json.dumps(r["train_log"][29], indent=1))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()
