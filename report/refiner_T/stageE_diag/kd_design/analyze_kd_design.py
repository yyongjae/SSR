#!/usr/bin/env python
"""H5 / H6 analysis (stage-E E2 < E0 + R_T diagnosis).  Descriptive.

Inputs
  student : /ssd/.../stageE_diag/e2_navtest_dump.npz (E2 last.ckpt, fp32; c_lon / e_lat decoded controls)
  teachers: /ssd/.../stageE_diag/kd_design/R_{T4,M4}_fp32/pred.npz (E2 KD teachers, fp32 as in E2 training, on E2's
            own navtest tau0 drafts; the decoded controls are exactly the E2 KD targets for these drafts)
  scores  : kd_design/consensus/scores.parquet (k: 0 R_T4 fp16, 1 R_M4 fp16, 2 R_T4 fp32, 3 R_M4 fp32, 4 MID, 5 WEAK)
            stageE_diag/scores.parquet (k 0 E2 tau0, 1 E2 tau_final), e0_teacher_refine/{E0,R_T4,R_M4}/scores.parquet
  training: kd_design/train_mix.json (train_mix.py)
KD controls = c_lon[2:] (m/s, <= 0) and e_lat[2:] (m); KD loss = mean over 2 teachers of mean |S - teacher| (L1).
With two teachers, every point of the element-wise interval [min(T, M), max(T, M)] minimises the KD term (its
subgradient is 0 inside), so inside the interval only the surrogate positions the student.
Bootstrap: paired by token, 136 navtest log clusters, 10,000 draws, seed 0, percentile 95 % CI (stageE_compare.boot).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/external-user/yongjae/SSR/tools/refiner")
from stageE_compare import boot  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
B = D / "kd_design"
R = Path("/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/kd_design")
E0R = Path("/home/external-user/ssd/yongjae_refiner/e0_teacher_refine")
NB = 10000
TOL = 1e-3      # element tolerance for "between" (m/s or m)
DIS = 0.02      # an element counts as a teacher disagreement when |T - M| > DIS
BRK = 0.1       # material braking: min c_lon < -BRK m/s
LATM = 0.1      # material lateral: max |e_lat| > LATM m

# ------------------------------------------------------------------ load + align
S = dict(np.load(D / "e2_navtest_dump.npz"))
tok = S["tokens"]
order = {t: i for i, t in enumerate(tok)}


def aligned(n):
    d = dict(np.load(B / n / "pred.npz"))
    idx = np.array([order[t] for t in d["tokens"]])
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))
    return {k: (v[inv] if isinstance(v, np.ndarray) and v.shape[:1] == (len(idx),) else v) for k, v in d.items()}


T, M = aligned("R_T4_fp32"), aligned("R_M4_fp32")
assert np.array_equal(T["tokens"], tok) and np.array_equal(M["tokens"], tok)
assert np.array_equal(T["tau0"][:, 0], S["tau0"])
logs = S["log"]
N = len(tok)
out = {"n_tokens": int(N), "n_logs": int(len(set(logs))), "params": dict(TOL=TOL, DIS=DIS, BRK=BRK, LATM=LATM, n_boot=NB)}

# ego inputs: student (status_feature) vs teachers (packed split)
out["ego_input_check"] = {k: float(np.abs(S[k].astype(float) - T[k].astype(float)).max()) for k in ("v0", "a0", "eds", "cmd")}

cS = S["c_lon"][:, 2:].astype(float); eS = S["e_lat"][:, 2:].astype(float)
cT = T["c_lon"][:, 0, 2:].astype(float); eT = T["e_lat"][:, 0, 2:].astype(float)
cM = M["c_lon"][:, 0, 2:].astype(float); eM = M["e_lat"][:, 0, 2:].astype(float)
XS, XT, XM = np.c_[cS, eS], np.c_[cT, eT], np.c_[cM, eM]


def ci_mean(x, lg=logs):
    x = np.asarray(x, float)
    lo, hi = boot(x, lg, NB)
    return [float(x.mean()), lo, hi]


# ------------------------------------------------------------------ activity (navtest, E2 tau0 drafts)
def brake(c):
    return c.min(1) < -BRK


def live(c):
    return (c < 0).any(1)


def latm(e):
    return np.abs(e).max(1) > LATM


act = {}
for nm, c, e in (("student", cS, eS), ("R_T", cT, eT), ("R_M", cM, eM)):
    act[nm] = {"lon_live": ci_mean(live(c)), "brake>0.1": ci_mean(brake(c)), "lat>0.1": ci_mean(latm(e)),
               "mean_depth_ms": ci_mean(-c.min(1)), "mean_abs_e": ci_mean(np.abs(e).mean(1))}
out["navtest_activity_on_E2_tau0"] = act

# ------------------------------------------------------------------ KD loss on navtest (generalisation)
l1T = np.abs(XS - XT).mean(1); l1M = np.abs(XS - XM).mean(1)
out["navtest_kd_l1"] = {"R_T": ci_mean(l1T), "R_M": ci_mean(l1M), "mean2": ci_mean((l1T + l1M) / 2),
                        "R_T_lon": ci_mean(np.abs(cS - cT).mean(1)), "R_T_lat": ci_mean(np.abs(eS - eT).mean(1)),
                        "R_M_lon": ci_mean(np.abs(cS - cM).mean(1)), "R_M_lat": ci_mean(np.abs(eS - eM).mean(1))}

# ------------------------------------------------------------------ teacher disagreement
bT, bM = brake(cT), brake(cM)
lon_cat = np.select([bT & bM, bT & ~bM, ~bT & bM], ["both", "T_only", "M_only"], "none")
kT, kM = latm(eT), latm(eM)
# lateral sign at the largest offset
sT = np.sign(eT[np.arange(N), np.abs(eT).argmax(1)]); sM = np.sign(eM[np.arange(N), np.abs(eM).argmax(1)])
lat_cat = np.select([kT & kM & (sT == sM), kT & kM & (sT != sM), kT & ~kM, ~kT & kM],
                    ["both_same", "both_opposite", "T_only", "M_only"], "none")
dTM = np.abs(XT - XM).mean(1)
dTM_lon, dTM_lat = np.abs(cT - cM).mean(1), np.abs(eT - eM).mean(1)
el_opp = (np.sign(eT) * np.sign(eM) < 0) & (np.abs(eT) > 0.05) & (np.abs(eM) > 0.05)
dis = {"lon_category_frac": {k: float((lon_cat == k).mean()) for k in ("both", "T_only", "M_only", "none")},
       "lat_category_frac": {k: float((lat_cat == k).mean()) for k in ("both_same", "both_opposite", "T_only", "M_only", "none")},
       "token_L1_T_vs_M": ci_mean(dTM), "token_L1_T_vs_M_lon": ci_mean(dTM_lon), "token_L1_T_vs_M_lat": ci_mean(dTM_lat),
       "frac_tokens_dTM>0.02": float((dTM > 0.02).mean()), "frac_tokens_dTM>0.05": float((dTM > 0.05).mean()),
       "frac_tokens_opposite_lat_elem(|e|>0.05 both)": float(el_opp.any(1).mean()),
       "lon_sign_note": "c_lon <= 0 for both teachers (mode A): lon disagreement is only in magnitude / which one brakes",
       "corr_depth_T_M": float(np.corrcoef(-cT.min(1), -cM.min(1))[0, 1]),
       "corr_elat_T_M": float(np.corrcoef(eT.ravel(), eM.ravel())[0, 1])}
out["teacher_disagreement"] = dis

# ------------------------------------------------------------------ student position relative to the teachers
lo, hi = np.minimum(XT, XM), np.maximum(XT, XM)
disel = (hi - lo) > DIS
between = (XS >= lo - TOL) & (XS <= hi + TOL)
# weak / strong side per element: lon weak = max(c) (less braking); lat same sign: weak = smaller |e|; opposite sign: 0
weak = np.c_[np.maximum(cT, cM), np.where(np.abs(eT) < np.abs(eM), eT, eM)]
strong = np.c_[np.minimum(cT, cM), np.where(np.abs(eT) >= np.abs(eM), eT, eM)]
same = np.c_[np.ones_like(cT, bool), np.sign(eT) * np.sign(eM) >= 0]
lam = np.where(disel & same, (XS - weak) / np.where(np.abs(strong - weak) > 0, strong - weak, np.nan), np.nan)
under = disel & same & (lam < -TOL / DIS)       # beyond the weaker teacher, toward no correction
over = disel & same & (lam > 1 + TOL / DIS)
pos = {}
for nm, cols in (("all", slice(None)), ("lon", slice(0, 6)), ("lat", slice(6, 12))):
    m = disel[:, cols]
    l_ = lam[:, cols][m & same[:, cols]]
    pos[nm] = {"n_disagree_elements": int(m.sum()),
               "frac_between": float(between[:, cols][m].mean()) if m.any() else None,
               "frac_under(beyond weaker, toward 0)": float(under[:, cols][m].mean()) if m.any() else None,
               "frac_over(beyond stronger)": float(over[:, cols][m].mean()) if m.any() else None,
               "lambda_median(0=weak,1=strong)": float(np.nanmedian(l_)) if l_.size else None,
               "lambda_quartiles": [float(x) for x in np.nanpercentile(l_, [25, 75])] if l_.size else None,
               "frac_lambda<0.5": float((l_ < 0.5).mean()) if l_.size else None}
# closer to which teacher (token level, all 12 controls)
closer_T = l1T < l1M
pos["token_closer_to_T_frac"] = ci_mean(closer_T)
pos["token_ratio_betweenness"] = ci_mean(np.where(dTM > 1e-6, dTM / np.maximum(l1T + l1M, 1e-12), 1.0))
# braking-magnitude position on tokens where at least one teacher brakes
anyb = bT | bM
dS, dT_, dM_ = -cS.min(1), -cT.min(1), -cM.min(1)
pos["lon_on_teacher_brake_tokens"] = {
    "n": int(anyb.sum()), "student_depth": float(dS[anyb].mean()), "R_T_depth": float(dT_[anyb].mean()),
    "R_M_depth": float(dM_[anyb].mean()), "student_brakes_frac": float(brake(cS)[anyb].mean()),
    "by_cat": {k: {"n": int((lon_cat == k).sum()), "student_brake_frac": float(brake(cS)[lon_cat == k].mean()),
                   "student_depth": float(dS[lon_cat == k].mean()), "R_T_depth": float(dT_[lon_cat == k].mean()),
                   "R_M_depth": float(dM_[lon_cat == k].mean())} for k in ("both", "T_only", "M_only", "none")}}
anyl = kT | kM
pos["lat_on_teacher_lat_tokens"] = {
    "n": int(anyl.sum()), "by_cat": {k: {"n": int((lat_cat == k).sum()),
                                         "student_lat_frac": float(latm(eS)[lat_cat == k].mean()),
                                         "student_max|e|": float(np.abs(eS).max(1)[lat_cat == k].mean()),
                                         "R_T_max|e|": float(np.abs(eT).max(1)[lat_cat == k].mean()),
                                         "R_M_max|e|": float(np.abs(eM).max(1)[lat_cat == k].mean()),
                                         "student_sign_agrees_T": float((np.sign(eS[np.arange(N), np.abs(eT).argmax(1)]) == sT)[lat_cat == k].mean()),
                                         "student_sign_agrees_M": float((np.sign(eS[np.arange(N), np.abs(eM).argmax(1)]) == sM)[lat_cat == k].mean())}
                                     for k in ("both_same", "both_opposite", "T_only", "M_only", "none")}}
# magnitude ratio student / teacher-mean on tokens where the teachers correct (agreeing vs disagreeing)
mag = lambda X: np.abs(X).mean(1)
tm = (mag(XT) + mag(XM)) / 2
for nm, sel in (("teachers_agree(dTM<=0.02)", (dTM <= 0.02) & (tm > 0.02)), ("teachers_disagree(dTM>0.02)", (dTM > 0.02) & (tm > 0.02))):
    pos[f"magnitude_ratio_student_over_teacher_mean[{nm}]"] = {"n": int(sel.sum()),
                                                              "ratio_of_means": float(mag(XS)[sel].mean() / tm[sel].mean())}
out["student_position"] = pos

# ------------------------------------------------------------------ scores
sc = pd.read_parquet(B / "consensus" / "scores.parquet")
assert (sc.error.isna() | (sc.error == "")).all() and sc.rec_ok.all()
names = {0: "E2tau0+R_T4", 1: "E2tau0+R_M4", 2: "E2tau0+R_T4fp32", 3: "E2tau0+R_M4fp32", 4: "E2tau0+MID", 5: "E2tau0+WEAK"}
P = {}
for k, nm in names.items():
    P[nm] = sc[sc.k == k].set_index("token").loc[tok]
s2 = pd.read_parquet(D / "scores.parquet")
P["E2tau0"] = s2[s2.k == 0].set_index("token").loc[tok]
P["E2"] = s2[s2.k == 1].set_index("token").loc[tok]
for nm in ("E0", "R_T4", "R_M4"):
    x = pd.read_parquet(E0R / nm / "scores.parquet").set_index("token").loc[tok]
    P["E0" if nm == "E0" else f"E0+{nm}"] = x
MET = ["pdms", "nc", "dac", "ttc", "ep"]
out["arm_means_x100"] = {a: {m: float(100 * P[a][m].mean()) for m in MET} for a in P}


def contrast(a, b, sel=None):
    r = {}
    for m in MET:
        d = 100 * (P[a][m].values - P[b][m].values)
        if sel is not None:
            d = np.where(sel, d, 0.0)          # contribution to the all-token mean
        r[m] = ci_mean(d)
    return r


C = {}
for a, b in (("E0+R_T4", "E2"), ("E0+R_T4", "E2tau0+R_T4"), ("E2tau0+R_T4", "E2"), ("E2tau0+R_M4", "E2"),
             ("E2tau0+MID", "E2"), ("E2tau0+WEAK", "E2"), ("E2tau0+R_T4", "E2tau0+MID"), ("E2tau0+R_T4", "E2tau0+WEAK"),
             ("E2tau0+R_M4", "E2tau0+MID"), ("E2tau0+R_T4", "E2tau0+R_M4"), ("E2", "E2tau0"), ("E2tau0+R_T4", "E2tau0"),
             ("E2tau0+MID", "E2tau0"), ("E2tau0+R_T4fp32", "E2tau0+R_T4"), ("E0", "E2tau0"), ("E0+R_T4", "E0")):
    C[f"{a} - {b}"] = contrast(a, b)
out["contrasts_x100"] = C

# gap decomposition (PDMS points): E0+R_T4 - E2 = draft + consensus + imitation
g = lambda a, b: C[f"{a} - {b}"]["pdms"][0]
gap = g("E0+R_T4", "E2")
dec = {"gap_E0+R_T4_minus_E2": gap,
       "draft: (E0+R_T4) - (E2tau0+R_T4)": g("E0+R_T4", "E2tau0+R_T4"),
       "same-draft teacher-vs-student: (E2tau0+R_T4) - E2": g("E2tau0+R_T4", "E2"),
       "  of which consensus (H5 midpoint): (E2tau0+R_T4) - (E2tau0+MID)": g("E2tau0+R_T4", "E2tau0+MID"),
       "  of which imitation residual: (E2tau0+MID) - E2": g("E2tau0+MID", "E2"),
       "H5 upper bound (weakest KD-optimal point): (E2tau0+R_T4) - (E2tau0+WEAK)": g("E2tau0+R_T4", "E2tau0+WEAK")}
dec["H5_share_of_gap_midpoint"] = dec["  of which consensus (H5 midpoint): (E2tau0+R_T4) - (E2tau0+MID)"] / gap
dec["H5_share_of_gap_upper(WEAK)"] = dec["H5 upper bound (weakest KD-optimal point): (E2tau0+R_T4) - (E2tau0+WEAK)"] / gap
out["gap_decomposition_pdms_pts"] = dec

# ------------------------------------------------------------------ missed fixes vs disagreement
fail = lambda a: (P[a][["nc", "dac", "ddc", "ttc"]].values < 1).any(1)
f0 = fail("E2tau0")
fixT, fixM, fixS = f0 & ~fail("E2tau0+R_T4"), f0 & ~fail("E2tau0+R_M4"), f0 & ~fail("E2")
fixMID, fixW = f0 & ~fail("E2tau0+MID"), f0 & ~fail("E2tau0+WEAK")
mf = {"n_E2tau0_fail_any": int(f0.sum()), "fixed_by": {"R_T4": int(fixT.sum()), "R_M4": int(fixM.sum()), "student(E2)": int(fixS.sum()),
                                                      "MID": int(fixMID.sum()), "WEAK": int(fixW.sum()),
                                                      "R_T4_or_R_M4": int((fixT | fixM).sum())},
      "new_fail": {a: int((~f0 & fail(a)).sum()) for a in ("E2tau0+R_T4", "E2tau0+R_M4", "E2", "E2tau0+MID", "E2tau0+WEAK")}}
groups = {"both_fix": fixT & fixM, "T_only_fix": fixT & ~fixM, "M_only_fix": ~fixT & fixM, "neither": f0 & ~fixT & ~fixM}
mf["by_teacher_fix_group"] = {k: {"n": int(v.sum()), "student_fix_rate": float(fixS[v].mean()) if v.any() else None,
                                  "MID_fix_rate": float(fixMID[v].mean()) if v.any() else None,
                                  "WEAK_fix_rate": float(fixW[v].mean()) if v.any() else None,
                                  "mean_dTM": float(dTM[v].mean()) if v.any() else None,
                                  "student_between_frac(disagree elems)": float(between[v][disel[v]].mean()) if disel[v].any() else None,
                                  "student_closer_to_T": float(closer_T[v].mean()) if v.any() else None}
                              for k, v in groups.items()}
# teacher-fixed tokens split by disagreement (median dTM over teacher-fixed tokens)
tf = fixT | fixM
if tf.any():
    med = float(np.median(dTM[tf]))
    for nm, sel in (("teacher_fixed & dTM<=median", tf & (dTM <= med)), ("teacher_fixed & dTM>median", tf & (dTM > med))):
        mf[nm] = {"n": int(sel.sum()), "student_fix_rate": float(fixS[sel].mean()), "MID_fix_rate": float(fixMID[sel].mean()),
                  "median_dTM": med}
out["missed_fixes"] = mf

# ------------------------------------------------------------------ same-draft gap by disagreement bin
qs = np.quantile(dTM, [0, .5, .75, .9, 1.0])
bins = np.clip(np.searchsorted(qs, dTM, side="right") - 1, 0, 3)
lab = ["q0-50", "q50-75", "q75-90", "q90-100"]
byb = {}
for b in range(4):
    sel = bins == b
    byb[lab[b]] = {"n": int(sel.sum()), "dTM_range": [float(qs[b]), float(qs[b + 1])],
                   "contrib (E2tau0+R_T4)-E2": ci_mean(100 * np.where(sel, P["E2tau0+R_T4"].pdms.values - P["E2"].pdms.values, 0)),
                   "contrib (E2tau0+R_T4)-MID": ci_mean(100 * np.where(sel, P["E2tau0+R_T4"].pdms.values - P["E2tau0+MID"].pdms.values, 0)),
                   "contrib MID-E2": ci_mean(100 * np.where(sel, P["E2tau0+MID"].pdms.values - P["E2"].pdms.values, 0)),
                   "E2tau0_fail_rate": float(f0[sel].mean())}
out["same_draft_gap_by_disagreement_bin"] = byb


# ------------------------------------------------------------------ gap by teacher agreement group
lon_ag = np.isin(lon_cat, ["both", "none"]); lat_ag = np.isin(lat_cat, ["both_same", "none"])
corr_any = bT | bM | kT | kM
grp = np.where(~corr_any, "no_teacher_correction", np.where(lon_ag & lat_ag, "teachers_agree_correct", "teachers_disagree"))
gg = {}
for g_ in ("no_teacher_correction", "teachers_agree_correct", "teachers_disagree"):
    sel = grp == g_
    r = {"n": int(sel.sum()), "E2tau0_fail_rate": float(f0[sel].mean()), "student_fix_rate_of_fails": float(fixS[sel & f0].sum() / max(1, (sel & f0).sum())),
         "MID_fix_rate_of_fails": float(fixMID[sel & f0].sum() / max(1, (sel & f0).sum())),
         "WEAK_fix_rate_of_fails": float(fixW[sel & f0].sum() / max(1, (sel & f0).sum())),
         "R_T_fix_rate_of_fails": float(fixT[sel & f0].sum() / max(1, (sel & f0).sum()))}
    for a, b in (("E2tau0+R_T4", "E2"), ("E2tau0+MID", "E2"), ("E2tau0+WEAK", "E2"), ("E2tau0+MID", "E2tau0+WEAK"), ("E2tau0+R_T4", "E2tau0+MID")):
        r[f"contrib {a} - {b}"] = ci_mean(100 * np.where(sel, P[a].pdms.values - P[b].pdms.values, 0))
    gg[g_] = r
out["gap_by_agreement_group"] = gg

# ------------------------------------------------------------------ H6: training vs navtest
tm_ = json.load(open(R / "train_mix.json"))
bt = tm_["by_type"]["ep25-29"]["metrics"]; bt_all = tm_["by_type"]["ep5-29"]["metrics"]
h6 = {"train_mix": tm_["mix_by_phase"],
      "train_tau0_ep25-29": {"R_T_live": bt["kd/teacher_live_0"]["pure_tau0"], "R_M_live": bt["kd/teacher_live_1"]["pure_tau0"],
                             "student_live": bt["ref/live"]["pure_tau0"], "kd_l1_T": bt["kd/l1_0"]["pure_tau0"],
                             "kd_l1_M": bt["kd/l1_1"]["pure_tau0"]},
      "train_human_ep25-29": {"R_T_live": bt["kd/teacher_live_0"]["pure_human"], "R_M_live": bt["kd/teacher_live_1"]["pure_human"],
                              "student_live": bt["ref/live"]["pure_human"], "kd_l1_T": bt["kd/l1_0"]["pure_human"],
                              "kd_l1_M": bt["kd/l1_1"]["pure_human"]},
      "navtest_tau0": {"R_T_live": act["R_T"]["lon_live"], "R_M_live": act["R_M"]["lon_live"], "student_live": act["student"]["lon_live"],
                       "kd_l1_T": out["navtest_kd_l1"]["R_T"], "kd_l1_M": out["navtest_kd_l1"]["R_M"]},
      "E0_collision_fail_report32": {"navtrain_train_scenes": 0.70, "navtrain_heldout_logs": 1.50, "navtest": 2.30,
                                     "src": "report/32_cause_and_correction_tests.md L253"},
      "E2tau0_navtest_fail_rates_x100": {m: float(100 * (P["E2tau0"][m] < 1).mean()) for m in ("nc", "dac", "ttc")},
      "E2tau0_navtest_fail_any_x100": float(100 * f0.mean())}
st, nt = h6["train_tau0_ep25-29"], h6["navtest_tau0"]
h6["ratio_student_live_over_teacher_mean_live"] = {
    "train_tau0": st["student_live"][0] / ((st["R_T_live"][0] + st["R_M_live"][0]) / 2),
    "train_human": h6["train_human_ep25-29"]["student_live"][0] / ((h6["train_human_ep25-29"]["R_T_live"][0] + h6["train_human_ep25-29"]["R_M_live"][0]) / 2),
    "navtest_tau0": nt["student_live"][0] / ((nt["R_T_live"][0] + nt["R_M_live"][0]) / 2)}
h6["kd_l1_navtest_over_train_tau0"] = {"R_T": nt["kd_l1_T"][0] / st["kd_l1_T"][0], "R_M": nt["kd_l1_M"][0] / st["kd_l1_M"][0]}
# student under-braking where the teachers brake on navtest (conditional braking recall)
h6["navtest_student_brake_recall"] = {"given_R_T_live": float(live(cS)[live(cT)].mean()),
                                      "given_R_M_live": float(live(cS)[live(cM)].mean()),
                                      "given_both_live": float(live(cS)[live(cT) & live(cM)].mean()),
                                      "given_neither_live(false_pos)": float(live(cS)[~live(cT) & ~live(cM)].mean())}
out["H6"] = h6

json.dump(out, open(R.parent / "kd-design.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
