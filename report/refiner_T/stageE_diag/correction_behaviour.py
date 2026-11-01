#!/usr/bin/env python3
"""Stage-E diagnostics, hypotheses H3 (the E2 student under-corrects) and H4 (its corrections point elsewhere).

Same draft for everyone: E2's own navtest tau0 (e2_navtest_dump.npz).  Student S = E2's refiner as evaluated; teachers
T = frozen run-4 R_T (BEVFusion), M = frozen run-4 R_M (ReSMap), both run on E2's tau0 by refine_external_drafts.py
(same_draft/{R_T4,R_M4}/pred.npz).  Control swaps / amplifications and their official scores:
teacher_on_e2/{swaps.npz, scores.parquet} (build_control_swaps.py).  Descriptive; log-cluster paired bootstrap
(10,000 draws, seed 0, percentile 95 %) over the 136 navtest logs for every mean / rate / difference.

Controls compared = the KD controls of stage E (kd_space 'decoded'): c_lon[2:] (m/s speed offsets, mode-A clamp, <= 0)
and e_lat[2:] (m lateral offsets).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/external-user/yongjae/SSR")
DIAG = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
SAME = DIAG / "same_draft"
SW = DIAG / "teacher_on_e2"
OBJ = Path("/home/external-user/ssd/yongjae_refiner/objects/navtest")
RUN = REPO / "work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0"
OUT_DIR = REPO / "report/refiner_T/stageE_diag"
CORE = ("nc", "dac", "ddc", "ttc")
NB = 10000
GAP_E0RT_E2 = None  # filled from the reports


# ------------------------------------------------------------------------------------------------ bootstrap
class Boot:
    def __init__(self, logs, n=NB, seed=0):
        self.codes, self.inv = np.unique(logs, return_inverse=True)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.codes), size=(n, len(self.codes)))
        # multiplicity matrix W [n, L]
        self.W = np.zeros((n, len(self.codes)), np.float32)
        for j in range(len(self.codes)):
            self.W[:, j] = (idx == j).sum(1)

    def mean(self, x, m=None):
        """mean of x over tokens in mask m, with log-cluster CI -> dict(v, lo, hi, n)."""
        x = np.asarray(x, np.float64)
        m = np.ones(len(x), bool) if m is None else np.asarray(m, bool)
        n = int(m.sum())
        if n == 0:
            return dict(v=None, lo=None, hi=None, n=0)
        s = np.bincount(self.inv, weights=np.where(m, x, 0.0), minlength=len(self.codes))
        c = np.bincount(self.inv, weights=m.astype(float), minlength=len(self.codes))
        num, den = self.W @ s, self.W @ c
        ok = den > 0
        est = num[ok] / den[ok]
        return dict(v=float(x[m].mean()), lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)),
                    n=n)

    def total(self, x, m=None):
        """sum_{tokens in m} x / N_all (a contribution to an all-token mean), CI."""
        x = np.asarray(x, np.float64)
        m = np.ones(len(x), bool) if m is None else np.asarray(m, bool)
        s = np.bincount(self.inv, weights=np.where(m, x, 0.0), minlength=len(self.codes))
        c = np.bincount(self.inv, minlength=len(self.codes)).astype(float)
        est = (self.W @ s) / (self.W @ c)
        return dict(v=float(np.where(m, x, 0).sum() / len(x)), lo=float(np.percentile(est, 2.5)),
                    hi=float(np.percentile(est, 97.5)), n=int(m.sum()))


def arc(tr):
    p = np.concatenate([np.zeros_like(tr[:, :1, :2]), tr[:, :, :2]], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1).sum(1)


def lat_dev(tau0, tau1):
    """max over the 8 poses of the distance of tau1's pose to tau0's polyline (origin + 8 poses, last segment
    extended 20 m): the lateral displacement independent of the along-path (speed) change."""
    P = np.concatenate([np.zeros_like(tau0[:, :1, :2]), tau0[:, :, :2]], 1).astype(np.float64)
    d_end = P[:, -1] - P[:, -2]
    nrm = np.linalg.norm(d_end, axis=-1, keepdims=True)
    ext = P[:, -1] + 20.0 * d_end / np.maximum(nrm, 1e-6)
    P = np.concatenate([P, ext[:, None]], 1)
    A, B = P[:, :-1], P[:, 1:]                       # [N, S, 2]
    X = tau1[:, :, None, :2].astype(np.float64)      # [N, 8, 1, 2]
    AB = (B - A)[:, None]
    t = ((X - A[:, None]) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9)
    t = np.clip(t, 0, 1)
    C = A[:, None] + t[..., None] * AB
    d = np.linalg.norm(X - C, axis=-1).min(-1)       # [N, 8]
    return d.max(-1)


def fails(s):
    return np.any(np.stack([s[k].values < 1 for k in CORE], -1), -1)


def track_speed(tok, track, t_idx):
    try:
        z = np.load(OBJ / f"{tok}.npz", allow_pickle=False)
    except Exception:
        return np.nan
    tr = z["track"].astype(str)
    j = np.flatnonzero(tr == track)
    if len(j) == 0:
        return np.nan
    j = j[0]
    kf = z["kf"][j]
    k = int(round(max(t_idx, 0) / 5.0))
    k = min(k, kf.shape[0] - 1)
    if kf[k, 5] > 0:
        return float(np.hypot(kf[k, 3], kf[k, 4]))
    f = z["first"][j]
    return float(np.hypot(f[3], f[4]))


def load_scores(path, k=None, tokens=None, arm_names=None):
    s = pd.read_parquet(path)
    assert (s["error"].fillna("").astype(str) == "").all(), path
    out = {}
    for kk, g in s.groupby("k"):
        g = g.set_index("token").reindex(tokens)
        assert g.pdms.notna().all(), (path, kk)
        out[kk if arm_names is None else arm_names[kk]] = g
    return out


def r(x, nd=4):
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return x


# ------------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    D = np.load(DIAG / "e2_navtest_dump.npz")
    tokens = D["tokens"].astype(str)
    logs = D["log"].astype(str)
    N = len(tokens)
    bt = Boot(logs)
    tau0 = D["tau0"].astype(np.float64)
    ctrl = {"S": dict(c=D["c_lon"][:, 2:].astype(np.float64), e=D["e_lat"][:, 2:].astype(np.float64),
                      z=D["z_lon"].astype(np.float64), tau1=D["tau_final"].astype(np.float64))}
    for name, key in (("R_T4", "T"), ("R_M4", "M")):
        p = np.load(SAME / name / "pred.npz")
        idx = {t: i for i, t in enumerate(p["tokens"].astype(str))}
        rr = np.array([idx[t] for t in tokens])
        assert np.array_equal(p["tau0"][rr, 0], D["tau0"])
        ctrl[key] = dict(c=p["c_lon"][rr, 0, 2:].astype(np.float64), e=p["e_lat"][rr, 0, 2:].astype(np.float64),
                         z=p["z_lon"][rr, 0].astype(np.float64), tau1=p["tau1"][rr, 0].astype(np.float64))
    for k, v in ctrl.items():
        v["live"] = (v["c"] < 0).any(-1)
        v["short"] = arc(tau0) - arc(v["tau1"])
        v["lat"] = lat_dev(tau0, v["tau1"])
        v["emax"] = np.abs(v["e"]).max(-1)
        v["disp"] = np.linalg.norm(v["tau1"][..., :2] - tau0[..., :2], axis=-1).max(-1)
    assert np.array_equal(ctrl["S"]["live"], D["lon_live"].astype(bool))

    # ---------------------------------------------------------------- scores
    sE = load_scores(DIAG / "scores.parquet", tokens=tokens)          # k0 tau0, k1 tau_final (official == csv)
    sw = np.load(SW / "swaps.npz")
    arm_names = [str(a) for a in sw["arms"]]
    assert np.array_equal(sw["tokens"].astype(str), tokens)
    sS = load_scores(SW / "scores.parquet", tokens=tokens, arm_names=arm_names)
    S0, S1 = sE[0], sE[1]
    f0 = fails(S0)
    res = dict(n_tokens=N, n_logs=int(len(set(logs))), bootstrap="log-cluster paired, 10,000 draws, seed 0, 95 %")
    res["check_S_redecoded_vs_official"] = dict(
        pdms_official=float(S1.pdms.mean()), pdms_redecoded=float(sS["S"].pdms.mean()),
        n_tokens_pdms_diff_gt_1e6=int((np.abs(S1.pdms.values - sS["S"].pdms.values) > 1e-6).sum()))
    # arm table
    arms = {"E2_tau0": S0, "S (E2 final)": S1}
    arms.update({a: sS[a] for a in arm_names if a != "S"})
    res["arms"] = {}
    for a, g in arms.items():
        res["arms"][a] = dict(PDMS=bt.mean(g.pdms.values * 100), NC=float(g.nc.mean() * 100),
                              DAC=float(g.dac.mean() * 100), TTC=float(g.ttc.mean() * 100),
                              EP=float(g.ep.mean() * 100), fail_any=int(fails(g).sum()))
    base = sS["S"].pdms.values  # contrasts against the re-decoded S (same decode path as the swaps)
    res["contrasts_vs_S"] = {}
    for a in arm_names:
        if a == "S":
            continue
        d = (sS[a].pdms.values - base) * 100
        fa, fs = fails(sS[a]), fails(sS["S"])
        res["contrasts_vs_S"][a] = dict(dPDMS=bt.mean(d), fixed_vs_S=int((fs & ~fa).sum()),
                                        new_fail_vs_S=int((~fs & fa).sum()),
                                        dNC=float((sS[a].nc - sS["S"].nc).mean() * 100),
                                        dDAC=float((sS[a].dac - sS["S"].dac).mean() * 100),
                                        dTTC=float((sS[a].ttc - sS["S"].ttc).mean() * 100),
                                        dEP=float((sS[a].ep - sS["S"].ep).mean() * 100))
    res["contrasts_vs_tau0"] = {a: bt.mean((sS[a].pdms.values - S0.pdms.values) * 100) for a in arm_names}

    # ---------------------------------------------------------------- failure types on tau0
    nc0 = S0.nc.values
    spd = np.full(N, np.nan)
    for i in np.flatnonzero(nc0 == 0):
        spd[i] = track_speed(tokens[i], str(S0.nc_track.values[i]), int(S0.nc_time_idx.values[i]))
    ftype = {
        "NC_agent_moving": (nc0 == 0) & (spd > 0.5),
        "NC_agent_stationary": (nc0 == 0) & ~(spd > 0.5),
        "NC_static_object": nc0 == 0.5,
        "TTC": (S0.ttc.values == 0) & (nc0 == 1),
        "DAC": S0.dac.values == 0,
    }
    res["failure_type_note"] = ("types on E2 tau0, non-exclusive; NC_agent = nc 0 (vehicle / pedestrian / bicycle), moving "
                                "= collided track speed > 0.5 m/s at the collision keyframe (objects/navtest kf), "
                                "NC_static_object = nc 0.5 (cone / barrier / generic); TTC = ttc 0 with nc 1; "
                                f"n with unknown track speed {int(((nc0 == 0) & np.isnan(spd)).sum())}")

    # ---------------------------------------------------------------- control comparison
    S, T, M = ctrl["S"], ctrl["T"], ctrl["M"]
    allm = np.ones(N, bool)

    def behaviour(m):
        out = {}
        for k in ("S", "T", "M"):
            v = ctrl[k]
            out[k] = dict(brake=bt.mean(v["live"], m), short4_mean_m=bt.mean(v["short"], m),
                          short4_gt_0p5=bt.mean(v["short"] > 0.5, m), lat_mean_m=bt.mean(v["lat"], m),
                          lat_gt_0p1=bt.mean(v["lat"] > 0.1, m), lat_gt_0p5=bt.mean(v["lat"] > 0.5, m),
                          emax_mean_m=bt.mean(v["emax"], m), disp_gt_0p5=bt.mean(v["disp"] > 0.5, m))
        return out

    res["behaviour"] = {"all": behaviour(allm), "tau0_fail_any": behaviour(f0), "tau0_pass": behaviour(~f0)}
    res["behaviour_by_failure_type"] = {k: behaviour(m) for k, m in ftype.items()}
    res["diff_S_minus_T"] = {
        "brake": bt.mean(S["live"].astype(float) - T["live"], allm),
        "short4_m": bt.mean(S["short"] - T["short"], allm),
        "lat_m": bt.mean(S["lat"] - T["lat"], allm),
        "brake_on_fail": bt.mean(S["live"].astype(float) - T["live"], f0),
        "short4_on_fail_m": bt.mean(S["short"] - T["short"], f0),
        "lat_on_fail_m": bt.mean(S["lat"] - T["lat"], f0)}
    res["diff_S_minus_M"] = {
        "brake": bt.mean(S["live"].astype(float) - M["live"], allm),
        "short4_m": bt.mean(S["short"] - M["short"], allm),
        "lat_m": bt.mean(S["lat"] - M["lat"], allm),
        "brake_on_fail": bt.mean(S["live"].astype(float) - M["live"], f0),
        "short4_on_fail_m": bt.mean(S["short"] - M["short"], f0),
        "lat_on_fail_m": bt.mean(S["lat"] - M["lat"], f0)}

    def agreement(X, m=allm):
        o = {}
        o["L1_lon"] = bt.mean(np.abs(S["c"] - X["c"]).mean(-1), m)
        o["L1_lat"] = bt.mean(np.abs(S["e"] - X["e"]).mean(-1), m)
        o["P_Slive_given_Xlive"] = bt.mean(S["live"], m & X["live"])
        o["P_Xlive_given_Slive"] = bt.mean(X["live"], m & S["live"])
        o["P_Slive_given_not_Xlive"] = bt.mean(S["live"], m & ~X["live"])
        # longitudinal gain on the teacher's live drafts: <c_S, c_X> / |c_X|^2
        mx = m & X["live"]
        g_lon = (S["c"] * X["c"]).sum(-1) / np.maximum((X["c"] ** 2).sum(-1), 1e-12)
        o["lon_gain_on_Xlive_median"] = float(np.median(g_lon[mx])) if mx.any() else None
        o["lon_gain_on_Xlive_mean"] = bt.mean(np.clip(g_lon, -5, 5), mx)
        o["short_ratio_S_over_X_on_Xshort_gt_0p5"] = bt.mean(
            np.clip(S["short"] / np.maximum(X["short"], 1e-6), -5, 5), m & (X["short"] > 0.5))
        # lateral: dominant knot of the teacher
        act = m & (X["emax"] > 0.1)
        j = np.abs(X["e"]).argmax(-1)
        eS_j, eX_j = S["e"][np.arange(N), j], X["e"][np.arange(N), j]
        same = (np.sign(eS_j) == np.sign(eX_j)) & (np.abs(eS_j) >= 0.05)
        opp = (np.sign(eS_j) != np.sign(eX_j)) & (np.abs(eS_j) >= 0.05)
        none = np.abs(eS_j) < 0.05
        o["lat_active_n"] = int(act.sum())
        o["lat_same_sign"] = bt.mean(same, act)
        o["lat_opposite_sign"] = bt.mean(opp, act)
        o["lat_S_negligible(<5cm)"] = bt.mean(none, act)
        nS, nX = np.linalg.norm(S["e"], axis=-1), np.linalg.norm(X["e"], axis=-1)
        cos = (S["e"] * X["e"]).sum(-1) / np.maximum(nS * nX, 1e-12)
        g_lat = (S["e"] * X["e"]).sum(-1) / np.maximum(nX ** 2, 1e-12)
        both = act & (nS > 0.05)
        o["lat_cos_when_both_active_mean"] = bt.mean(cos, both)
        o["lat_cos_lt_0_when_both_active"] = bt.mean(cos < 0, both)
        o["lat_gain_on_Xactive_median"] = float(np.median(g_lat[act])) if act.any() else None
        o["lat_gain_on_Xactive_mean"] = bt.mean(np.clip(g_lat, -5, 5), act)
        return o, g_lon, g_lat, cos

    agT, glonT, glatT, cosT = agreement(T)
    agM, glonM, glatM, cosM = agreement(M)
    res["agreement"] = {"T": agT, "M": agM}
    res["agreement_on_tau0_fail"] = {"T": agreement(T, f0)[0], "M": agreement(M, f0)[0]}
    # teacher-teacher reference
    res["teacher_vs_teacher"] = dict(
        L1_lon=bt.mean(np.abs(T["c"] - M["c"]).mean(-1)), L1_lat=bt.mean(np.abs(T["e"] - M["e"]).mean(-1)),
        P_Mlive_given_Tlive=bt.mean(M["live"], T["live"]), P_Tlive_given_Mlive=bt.mean(T["live"], M["live"]))
    # two-teacher L1 target: where the teachers disagree on braking
    res["student_brake_by_teacher_agreement"] = {
        "both_live": bt.mean(S["live"], T["live"] & M["live"]),
        "T_only_live": bt.mean(S["live"], T["live"] & ~M["live"]),
        "M_only_live": bt.mean(S["live"], ~T["live"] & M["live"]),
        "neither_live": bt.mean(S["live"], ~T["live"] & ~M["live"])}
    act_T, act_M = T["emax"] > 0.1, M["emax"] > 0.1
    cosTM = (T["e"] * M["e"]).sum(-1) / np.maximum(np.linalg.norm(T["e"], axis=-1) * np.linalg.norm(M["e"], axis=-1),
                                                     1e-12)
    res["student_lat_by_teacher_agreement"] = {
        "both_active_same_dir(cos>0.5)": bt.mean(S["emax"], act_T & act_M & (cosTM > 0.5)),
        "both_active_conflict(cos<0)": bt.mean(S["emax"], act_T & act_M & (cosTM < 0)),
        "T_only_active": bt.mean(S["emax"], act_T & ~act_M),
        "M_only_active": bt.mean(S["emax"], ~act_T & act_M),
        "T_emax_same_masks": {"both_same": bt.mean(T["emax"], act_T & act_M & (cosTM > 0.5)),
                              "T_only": bt.mean(T["emax"], act_T & ~act_M)}}

    # ---------------------------------------------------------------- outcome split (fix / miss)
    def outcome(Xname):
        sx = sS[Xname]
        fx, fs = fails(sx), fails(S1)
        grp = {"X_fixes_S_not": f0 & ~fx & fs, "S_fixes_X_not": f0 & fx & ~fs, "both_fix": f0 & ~fx & ~fs,
               "neither_fixes": f0 & fx & fs, "X_breaks_S_not": ~f0 & fx & ~fs, "S_breaks_X_not": ~f0 & fs & ~fx}
        X = ctrl[Xname]
        o = {}
        for gname, m in grp.items():
            o[gname] = dict(n=int(m.sum()),
                            S_brake=bt.mean(S["live"], m), X_brake=bt.mean(X["live"], m),
                            S_short4=bt.mean(S["short"], m), X_short4=bt.mean(X["short"], m),
                            S_lat=bt.mean(S["lat"], m), X_lat=bt.mean(X["lat"], m),
                            lat_opposite=bt.mean(((np.sign(S["e"][np.arange(N), np.abs(X["e"]).argmax(-1)])
                                                   != np.sign(X["e"][np.arange(N), np.abs(X["e"]).argmax(-1)]))
                                                  & (np.abs(S["e"][np.arange(N), np.abs(X["e"]).argmax(-1)]) >= 0.05)),
                                                 m & (X["emax"] > 0.1)),
                            S_dead_all_z_gt3=bt.mean((S["z"] > 3).all(-1), m),
                            dPDMS_X_minus_S_contrib=bt.total((sx.pdms.values - S1.pdms.values) * 100, m))
        per_type = {}
        for tname, tm in ftype.items():
            per_type[tname] = dict(n=int(tm.sum()), S_fix=bt.mean(~fs, tm), X_fix=bt.mean(~fx, tm),
                                   X_fixes_S_not=int((tm & ~fx & fs).sum()), S_fixes_X_not=int((tm & fx & ~fs).sum()),
                                   S_brake=bt.mean(S["live"], tm), X_brake=bt.mean(X["live"], tm),
                                   S_lat=bt.mean(S["lat"], tm), X_lat=bt.mean(X["lat"], tm),
                                   dPDMS_X_minus_S_contrib=bt.total((sx.pdms.values - S1.pdms.values) * 100, tm))
        return o, per_type

    res["outcome_vs_T"], res["failure_type_vs_T"] = outcome("T")
    res["outcome_vs_M"], res["failure_type_vs_M"] = outcome("M")

    # ---------------------------------------------------------------- z_lon dead zone
    zS = S["z"]
    dz = {}
    for thr in (2.0, 3.0, 5.0):
        dead = (zS > thr).all(-1)
        dz[f"all_z_gt_{thr:g}"] = dict(
            S_frac=bt.mean(dead), T_frac=bt.mean((T["z"] > thr).all(-1)), M_frac=bt.mean((M["z"] > thr).all(-1)),
            S_frac_on_Tlive=bt.mean(dead, T["live"]), S_frac_on_tau0_fail=bt.mean(dead, f0))
    dz["mean_relu_z_sq"] = dict(S=bt.mean((np.maximum(zS, 0) ** 2).mean(-1)),
                                T=bt.mean((np.maximum(T["z"], 0) ** 2).mean(-1)),
                                M=bt.mean((np.maximum(M["z"], 0) ** 2).mean(-1)))
    dz["z_mean_per_knot"] = dict(S=zS.mean(0).tolist(), T=T["z"].mean(0).tolist(), M=M["z"].mean(0).tolist())
    dz["S_min_z_percentiles(5,25,50,75,95)"] = np.percentile(zS.min(-1), [5, 25, 50, 75, 95]).tolist()
    dz["tanh_prime_at_S_min_z_median"] = float(1 - np.tanh(np.median(zS.min(-1))) ** 2)
    # correlation with missed fixes (vs T): among tau0 failures that T fixes
    fT, fs = fails(sS["T"]), fails(S1)
    fixable = f0 & ~fT
    dead3 = (zS > 3).all(-1)
    dz["missed_fix_vs_T"] = dict(
        n_T_fixable=int(fixable.sum()),
        P_S_miss_given_dead3=bt.mean(fs, fixable & dead3), P_S_miss_given_not_dead3=bt.mean(fs, fixable & ~dead3),
        n_dead3=int((fixable & dead3).sum()), n_not_dead3=int((fixable & ~dead3).sum()),
        P_dead3_given_T_brakes_and_fixes=bt.mean(dead3, fixable & T["live"]),
        P_S_miss_given_T_brakes_and_S_dead3=bt.mean(fs, fixable & T["live"] & dead3),
        P_S_miss_given_T_brakes_and_S_not_dead3=bt.mean(fs, fixable & T["live"] & ~dead3))
    res["z_dead_zone"] = dz

    # ---------------------------------------------------------------- train vs test KD distance
    st = pd.read_json(RUN / "stageE_steps.jsonl", lines=True)
    last = st[st.epoch >= 25]
    tr = {}
    for nh, g in [("tau0_only(n_human=0)", last[last["ref/n_human"] == 0]), ("all", last)]:
        v = {}
        for k in ("kd/l1_lon_0", "kd/l1_lat_0", "kd/l1_lon_1", "kd/l1_lat_1", "ref/live", "kd/teacher_live_0",
                  "kd/teacher_live_1", "ref/zdead", "ref/lat_m", "ref/short_m"):
            x = g[k].values.astype(float)
            rng = np.random.default_rng(0)
            bm = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)])
            v[k] = dict(v=float(x.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))
        v["n_microbatches"] = int(len(g))
        tr[nh] = v
    res["train_epochs_25_29"] = tr
    res["test_navtest_tau0"] = {
        "kd/l1_lon_0(T)": bt.mean(np.abs(S["c"] - T["c"]).mean(-1)), "kd/l1_lat_0(T)": bt.mean(np.abs(S["e"] - T["e"]).mean(-1)),
        "kd/l1_lon_1(M)": bt.mean(np.abs(S["c"] - M["c"]).mean(-1)), "kd/l1_lat_1(M)": bt.mean(np.abs(S["e"] - M["e"]).mean(-1)),
        "ref/live(S)": bt.mean(S["live"]), "teacher_live_0(T)": bt.mean(T["live"]), "teacher_live_1(M)": bt.mean(M["live"]),
        "ref/zdead(S)": bt.mean((np.maximum(zS, 0) ** 2).mean(-1)),
        "ref/lat_m(S; max|d|~max lateral dev)": bt.mean(S["lat"]), "ref/short_m(S)": bt.mean(S["short"])}
    res["train_test_note"] = ("train: rank-0 micro-batches (4 samples) of epochs 25-29 with no human draft (all 4 drafts = "
                              "unperturbed sg(tau0) on navtrain), teachers fp32 on navtrain caches; CI = i.i.d. "
                              "micro-batch bootstrap (2,000).  test: navtest, E2 tau0, teachers fp16 autocast on navtest "
                              "caches (BEVFusion cache_val_50x100 / ReSMap navtest).  ref/lat_m in training = mean max|d| "
                              "(decoded offset), test uses the pose-to-path deviation (~ max|d| at the poses).")

    # ---------------------------------------------------------------- token-level attribution of the refiner gap
    attr = {}
    for Xname in ("T", "M"):
        X = ctrl[Xname]
        sx = sS[Xname]
        d = (sx.pdms.values - S1.pdms.values) * 100
        lon_act = X["live"] & (X["short"] > 0.1)
        lat_act = X["lat"] > 0.1
        rl = np.where(lon_act, S["short"] / np.maximum(X["short"], 1e-6), np.nan)
        glat = glatT if Xname == "T" else glatM
        cosx = cosT if Xname == "T" else cosM
        lat_opp = lat_act & (np.linalg.norm(S["e"], axis=-1) > 0.05) & (cosx < 0)
        lat_under = lat_act & ~lat_opp & (glat < 0.75)
        lon_under = lon_act & (rl < 0.75)
        cls = np.full(N, "X_passive", object)
        cls[(lon_act | lat_act)] = "S_matches_or_exceeds"
        cls[lon_under | lat_under] = "under"
        cls[lat_opp] = "direction"
        # S active where X is passive (S over-corrects / acts alone)
        s_alone = ~(lon_act | lat_act) & ((S["live"] & (S["short"] > 0.1)) | (S["lat"] > 0.1))
        cls[s_alone] = "S_acts_X_passive"
        a = {"refiner_gap_X_minus_S": bt.mean(d)}
        for c in ("under", "direction", "S_matches_or_exceeds", "S_acts_X_passive", "X_passive"):
            m = cls == c
            a[c] = dict(n=int(m.sum()), contrib=bt.total(d, m), mean_d=bt.mean(d, m) if m.any() else None)
        # under split by channel
        a["under_lon_only"] = dict(n=int((lon_under & ~lat_under & ~lat_opp).sum()),
                                   contrib=bt.total(d, lon_under & ~lat_under & ~lat_opp))
        a["under_lat_only"] = dict(n=int((lat_under & ~lon_under & ~lat_opp).sum()),
                                   contrib=bt.total(d, lat_under & ~lon_under & ~lat_opp))
        a["under_both"] = dict(n=int((lat_under & lon_under & ~lat_opp).sum()),
                               contrib=bt.total(d, lat_under & lon_under & ~lat_opp))
        a["definitions"] = ("X lon-active = X brakes and shortens > 0.1 m at 4 s; X lat-active = pose-to-path deviation "
                            "> 0.1 m; direction = X lat-active, |e_S| > 5 cm and cos(e_S, e_X) < 0; under = not direction "
                            "and (lon-active with S/X shortening < 0.75, or lat-active with lateral gain <e_S,e_X>/|e_X|^2 "
                            "< 0.75); S_acts_X_passive = X inactive in both channels but S active; contrib = sum over "
                            "the class of (PDMS_X - PDMS_S) / N_all (points)")
        attr[Xname] = a
    res["attribution"] = attr

    # ---------------------------------------------------------------- gap bookkeeping
    e0r = json.loads((REPO / "report/refiner_T/e0_teacher_refine/e0_teacher_refine.json").read_text())
    pd_e0rt, pd_e0rm = e0r["arms"]["R_T4"]["PDMS"] * 100, e0r["arms"]["R_M4"]["PDMS"] * 100
    pd_e2 = float(S1.pdms.mean() * 100)
    pd_T_e2 = float(sS["T"].pdms.mean() * 100)
    pd_M_e2 = float(sS["M"].pdms.mean() * 100)
    res["gap"] = dict(E0_RT4=pd_e0rt, E0_RM4=pd_e0rm, E2_final=pd_e2, E2tau0_RT4=pd_T_e2, E2tau0_RM4=pd_M_e2,
                      total_gap_T=pd_e0rt - pd_e2, draft_part_T=pd_e0rt - pd_T_e2, refiner_part_T=pd_T_e2 - pd_e2,
                      total_gap_M=pd_e0rm - pd_e2, draft_part_M=pd_e0rm - pd_M_e2, refiner_part_M=pd_M_e2 - pd_e2,
                      note="E0+R_X uses the E0 pkl scored with the batched scorer (e0_teacher_refine); E2tau0+R_X = "
                           "teacher re-decoded with the student's v0 (== packed v0) on E2 tau0 (teacher_on_e2 arm T/M)")
    e0rt = pd.read_parquet(REPO.parent.parent / "ssd/yongjae_refiner/e0_teacher_refine/R_T4/scores.parquet")
    e0rt = e0rt[e0rt.k == 0].set_index("token").reindex(tokens)
    assert e0rt.pdms.notna().all()
    p_e0rt = e0rt.pdms.values * 100
    res["gap_ci"] = {
        "total_E0RT4_minus_E2": bt.mean(p_e0rt - S1.pdms.values * 100),
        "draft_E0RT4_minus_E2tau0RT4": bt.mean(p_e0rt - sS["T"].pdms.values * 100),
        "refiner_E2tau0RT4_minus_E2": bt.mean((sS["T"].pdms.values - S1.pdms.values) * 100),
        "Sx2_minus_E0RT4": bt.mean(sS["Sx2"].pdms.values * 100 - p_e0rt),
        "Sx2_minus_T_same_draft": bt.mean((sS["Sx2"].pdms.values - sS["T"].pdms.values) * 100),
        "Sx3_minus_T_same_draft": bt.mean((sS["Sx3"].pdms.values - sS["T"].pdms.values) * 100),
        "Sx2_minus_M_same_draft": bt.mean((sS["Sx2"].pdms.values - sS["M"].pdms.values) * 100)}
    res["sec"] = round(time.time() - t_start, 1)
    res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "correction-behaviour.json").write_text(json.dumps(r(res), indent=1))
    print("wrote", OUT_DIR / "correction-behaviour.json", res["sec"], "s")


if __name__ == "__main__":
    main()
