#!/usr/bin/env python
"""refiner v4_3 Phase 0b -- analysis of the pre-registered held-out run (docs/refiner_v4/V4_3_SPEC.md 'Phase 0b'):
criterion S (primary), criterion 1 with R2 (secondary), criterion 2 with the R2 tube (informational), data checks ->
HO/results.json + HO/summary.md (verdict first).  CPU only; reads HO/prep, HO/checks, HO/crit1, HO/crit2, the Phase 0
F1 outputs (context only) and the read-only inputs of oracle_common.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/heldout_analyze.py

Definitions (fixed with the code before any held-out number was computed; AMBIGUITIES lists the readings)
  official fail: nc < 1 (0 agent, 0.5 static), ttc < 1, dac < 1; joint = any; fails3 = joint.
  R2 inside (a row): no R2 hit on NC, TTC, DAC (slacks >= 0).
  deployment blend: select.ck_final(prob, im) (SEL_W default, beta 1 = refiner v2 'pool96_old_b1'; ck2T10/r34
    prob16 / prob80 (the DET teacher critic, every navtrain_val row), v2_im of the parent for its variants) over the
    valid 96, argmax in the stored pool96 order (parent first) -- Phase 0's selection-oracle code.
  criterion S: switch = v2 top-1 if inside, else the best-blend row among the inside rows (none inside -> the blend
    pick); dPDMS = PDMS(switch) - PDMS(blend) (x 100) per token, paired log-clustered bootstrap
    (tools/ck/eval_ck.log_bootstrap, 2,000 resamples, seed 0, 95 % percentile CI); break = P(switch pick fails NC /
    TTC / DAC | blend pick passes all three).  PASS iff mean >= +1.0 AND CI lower bound > 0 AND break <= 1.3 %.
  criterion 1: per key (NC, TTC, DAC) and joint over the 96 executed rows of every token: fail recall
    P(outside | fail) >= 0.90 AND pass recall P(inside | pass) >= 0.90 on the point estimates (8 bars); CIs =
    log-clustered bootstrap of the ratio estimator (analyze.LogBoot).
  criterion 2 (informational): the R2 computed edit (heldout_crit2 'r2_t0.2_l0.3') vs the DET lateral head on the v2
    top-1, Phase 0's rule (dPDMS >= +0.5, CI lower bound > 0, break <= 1.3 % and <= the head's break).
  Every rate below also carries a log-clustered bootstrap CI (token-level ratio estimator, the same log weights).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import heldout_common as H  # noqa: E402
import oracle_common as OC  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import (BAR, I_C, I_DAC, I_DDC, I_EP, I_NC, I_PDMS, I_TTC, K4, KCOL, KEYS, LogBoot, boot, ci, f3, f4,  # noqa: E402
                     fails3, rc, rcell)
from navsim.agents.para_ssr.ck.select import ck_final  # noqa: E402

S_MEAN_BAR = 1.0
BREAK_BAR = 0.013
C2_MEAN_BAR = 0.5
R2 = H.R2_NAME
R2_KEYS = ("r2_nc", "r2_ttc", "r2_dac")
MAIN0_KEYS = A.REPS[A.MAIN_REP]
EXACT_KEYS = A.REPS[A.EXACT_REP]
REPS_HO = {R2: R2_KEYS, **A.REPS}
ROLE_HO = {R2: "Phase 0b main", **{k: f"Phase 0 {v}" for k, v in A.ROLE.items()}}
MAIN_CFG = "r2_t0.2_l0.3"
P0_SWITCH = "switch: top-1 if inside Phase 0 main grid tube else filter + blend"

AMBIGUITIES = [
    ("Tube membership vs the margins 'm_t 0.2 s, m_l 0.3 m' listed under R2",
     "A row is inside the R2 tube iff it has no NC / TTC / DAC hit (slack >= 0), as Phase 0's criterion 1 and "
     "selection oracle; m_t / m_l are the computed edit's margins (criterion 2), as Phase 0's main edit config. Not "
     "used: membership with slack >= m_t / m_l."),
    ("'stopped-aware B' and the TTC channel U",
     "Stopped-aware applies to B (the NC channel) only, exactly oracle_common.stopped_aware_B01 (it returns B); U / "
     "LON is unchanged (the spec names B; the official TTC rule has no stopped-track exception). R2 = Phase 0's post "
     "hoc 'grid stopped-aware NC + read any4' (g25sa_nc, g25a_ttc, g25a_dac); heldout_crit1 recomputes it on its own "
     "code path and counts disagreements."),
    ("'any4 read for B and U' and DAC",
     "any4 = OR over the 2 x 2 grid points around each read point (range_targets._cells), for the NC read of B and "
     "every TTC projection read of U; DAC stays the bilinear read of D ('DAC on D bilinear'), unchanged by the read."),
    ("Criterion S when the top-1 is outside and no row of the 96 is inside",
     "The switch falls back to the blend pick over the 96 (Phase 0's selection-oracle rule 'none inside -> the blend "
     "pick'); such tokens contribute dPDMS 0."),
    ("Units of 'mean dPDMS >= +1.0'",
     "PDMS x 100 (Phase 0 reports every PDMS and dPDMS on the 0-100 scale; criterion 2's +0.5 is on the same scale)."),
    ("Criterion S break",
     "Numerator: tokens whose switch pick fails NC, TTC or DAC; denominator: tokens whose blend pick passes all "
     "three (DDC / EP / comfort not part of break), as written."),
    ("Blend tie-breaking",
     "argmax in the stored pool96 order (parent first), Phase 0's rule; agreement with the stored pool96_old_b1 "
     "picks on the held-out rows is reported."),
    ("Token set / 'all 96 executed rows'",
     "Held-out tokens with a GT object file, an SDF and finite top-1 / DET-head trajectories (Phase 0's token mask); "
     "the 96 = par16 + valid80 variants with finite trajectories (valid80 False = a lateral variant of a parent with "
     "S_8 < 3 m, a duplicate of the parent)."),
    ("Criterion 2 any4 critical step (Phase 0 left it open)",
     "Blocked profile = OR over the minimising read's 4 grid points (NC: stopped-aware B, TTC: U >= -1.3 - |v| delta); "
     "t_out = the end of its last blocked bin (+inf if the last bin is blocked); critical arc length s_crit = "
     "max(0, (j0 - 0.5) ds), j0 = the lower station of the 4 points (an any4 read at arc length s reads stations "
     "floor(s / ds - 0.5) and + 1, so s_crit is where the read first reaches those points: the any4 analogue of Phase "
     "0's s_crit = j ds); the rest of the edit rule unchanged (heldout_common.GridTubeR2)."),
    ("Executed states / ego",
     "ck/v4/lqr float64 (fit 'reduced') from (status v0, status a0, steer0 0) -- the lqr docstring's ego; checked "
     "against F1's official states and against the official simulator on held-out tokens (Checks)."),
    ("Critic probabilities",
     "ck2T10/r34 prob16 / prob80 already cover all 18,179 navtrain_val rows (done.npy all True, one inference run, the "
     "file Phase 0 used); no new inference was needed."),
    ("Bootstrap",
     "214 log clusters, 2,000 resamples, seed 0, 95 % percentile intervals (Phase 0's); differences paired per token; "
     "recalls / break / fix as ratio estimators over the resampled logs."),
]


# ----------------------------------------------------------------------------------------------- helpers
def pdms(L):
    return 100.0 * L[..., I_PDMS]


def pool_order():
    col = np.arange(OC.N96)
    key = np.where(col < OC.N_PAR, col * 6, ((col - OC.N_PAR) // 5) * 6 + (col - OC.N_PAR) % 5 + 1)
    return np.argsort(key), key


ORDER, POOL96_KEY = pool_order()


def amax(x):
    return ORDER[np.argmax(x[:, ORDER], 1)]


def blend_scores(rows: np.ndarray, V: np.ndarray) -> np.ndarray:
    prob16 = np.load(OC.T10 / "prob16.npy", mmap_mode="r")
    prob80 = np.load(OC.T10 / "prob80.npy", mmap_mode="r")
    im = np.load(OC.PACK / "v2_im.npy", mmap_mode="r")
    prob = np.concatenate([np.asarray(prob16[rows], np.float64), np.asarray(prob80[rows], np.float64)], 1)
    im16 = np.asarray(im[rows], np.float64)
    im96 = np.concatenate([im16, np.repeat(im16, 5, 1)], 1)
    return np.where(V, ck_final(prob, im96), -np.inf)


def rate(num: np.ndarray, den: np.ndarray, lbt: LogBoot) -> dict:
    """token-level ratio sum(num) / sum(den) + its log-clustered bootstrap CI."""
    n, d = int(num.sum()), int(den.sum())
    lo, hi = lbt.ratio_ci(num[:, None], den[:, None])
    return dict(rate=n / d if d else float("nan"), n=n, den=d, ci=[lo, hi])


def pct(r: dict, nd: int = 2) -> str:
    if not np.isfinite(r["rate"]):
        return "n/a"
    return f"{100 * r['rate']:.{nd}f} % [{100 * r['ci'][0]:.{nd}f}, {100 * r['ci'][1]:.{nd}f}] ({r['n']}/{r['den']})"


def inside_of(r1: dict, keys) -> np.ndarray:
    a, b, c = keys
    return ~(r1[a] | r1[b] | r1[c]) & r1["valid"]


# ----------------------------------------------------------------------------------------------- criterion S
def selection(r1: dict, E: np.ndarray, rows: np.ndarray, logs: np.ndarray, tokm: np.ndarray, bins: dict,
              r2c: dict = None) -> dict:
    """criterion S + context picks; r2c = heldout_crit2 rows (edit brackets) or None."""
    V = r1["valid"]
    s = blend_scores(rows, V)
    lab = E[:, :OC.N96]
    ar = np.arange(len(lab))
    top1, head = lab[:, 0], E[:, OC.COL_HEAD]
    lbt = LogBoot(logs, 1)
    blend = amax(s)
    Lb = lab[ar, blend]
    pass_b = ~fails3(Lb)
    m = tokm
    picks = {"v2 top-1": np.zeros(len(lab), np.int64), "blend over 96 (pool96_old_b1)": blend}
    ins = {"R2": inside_of(r1, R2_KEYS), "Phase 0 main grid": inside_of(r1, MAIN0_KEYS),
           "exact proxy (grid-free)": inside_of(r1, EXACT_KEYS)}
    filt = {}
    for tag, inside in ins.items():
        sf = np.where(inside, s, -np.inf)
        f = np.where(inside.any(1), amax(sf), blend)
        filt[tag] = f
        picks[f"switch: top-1 if inside {tag} tube else filter + blend"] = np.where(inside[:, 0], 0, f)
        picks[f"{tag} tube filter + blend"] = f
    picks["label oracle over 96"] = np.argmax(np.where(V, lab[..., I_PDMS], -np.inf), 1)
    main_pick = "switch: top-1 if inside R2 tube else filter + blend"

    def metrics(L, ix=None):
        o = dict(pdms=float(pdms(L)[m].mean()), d_vs_blend=boot(pdms(L)[m] - pdms(Lb)[m], logs[m]),
                 d_vs_top1=boot(pdms(L)[m] - pdms(top1)[m], logs[m]),
                 d_vs_head=boot(pdms(L)[m] - pdms(head)[m], logs[m]),
                 break_vs_blend=rate(fails3(L) & m & pass_b, m & pass_b, lbt),
                 fix_vs_blend=rate(~fails3(L) & m & ~pass_b, m & ~pass_b, lbt),
                 fail3=rate(fails3(L) & m, m, lbt), ep=float(100 * L[m, I_EP].mean()),
                 fail_nc=float((L[m, I_NC] < 1).mean()), fail_ttc=float((L[m, I_TTC] < 1).mean()),
                 fail_dac=float((L[m, I_DAC] < 1).mean()))
        if ix is not None:
            o["frac_changed_vs_top1"] = float((ix != 0)[m].mean())
            o["frac_differs_from_blend"] = float((ix != blend)[m].mean())
        return o

    out = {"n_tokens": int(m.sum()), "picks": {nm: metrics(lab[ar, ix], ix) for nm, ix in picks.items()}}
    # the verdict
    sw = picks[main_pick]
    Ls = lab[ar, sw]
    d = boot(pdms(Ls)[m] - pdms(Lb)[m], logs[m])
    brk = out["picks"][main_pick]["break_vs_blend"]
    parts = dict(mean_ge_1p0=bool(d["mean"] >= S_MEAN_BAR), ci_lo_gt_0=bool(d["lo"] > 0),
                 break_le_1p3=bool(brk["rate"] <= BREAK_BAR))
    out["criterion_S"] = dict(pick=main_pick, d_vs_blend=d, break_=brk, verdict=bool(all(parts.values())),
                              verdict_parts=parts)
    # decomposition
    inR2 = ins["R2"]
    t_in = m & inR2[:, 0]
    t_out_some = m & ~inR2[:, 0] & inR2.any(1)
    t_none = m & ~inR2[:, 0] & ~inR2.any(1)
    dd = pdms(Ls) - pdms(Lb)
    out["decomposition"] = {
        nm: dict(n=int(mm.sum()), frac=float(mm.sum() / m.sum()),
                 d_mean_on_subset=float(dd[mm].mean()) if mm.any() else float("nan"),
                 contribution_per_token=float(dd[mm].sum() / m.sum()),
                 d_on_subset=boot(dd[mm], logs[mm]) if mm.any() else None,
                 frac_pick_differs_from_blend=float((sw != blend)[mm].mean()) if mm.any() else float("nan"))
        for nm, mm in (("top-1 inside (switch = top-1)", t_in), ("top-1 outside, some row inside (filter)", t_out_some),
                       ("top-1 outside, no row inside (blend)", t_none))}
    out["frac_rows_inside_R2"] = float(inR2[m].sum() / r1["valid"][m].sum())
    out["scene_bins"] = {bn: dict(n=int((m & bm).sum()), d_vs_blend=boot(dd[m & bm], logs[m & bm]),
                                  break_=rate(fails3(Ls) & m & bm & pass_b, m & bm & pass_b, lbt))
                         for bn, bm in bins.items()}
    # stored picks reproduction
    try:
        stored = np.load(OC.T10 / "picks.npz")["pool96_old_b1"][rows]
        out["blend_vs_stored_pool96_old_b1"] = dict(
            identical_index=float((POOL96_KEY[blend] == stored)[m].mean()),
            n_differ=int(((POOL96_KEY[blend] != stored) & m).sum()))
    except Exception as e:  # noqa: BLE001
        out["blend_vs_stored_pool96_old_b1"] = dict(error=str(e))
    # computed edit as an extra candidate (brackets, as Phase 0; informational)
    if r2c is not None:
        names = [str(x) for x in r2c["cfg_names"]]
        q = names.index(MAIN_CFG)
        el = r2c["lab"][:, q]
        ed = r2c["edited"][:, q].astype(bool)
        e_in = r2c["inside_after"][:, q].astype(bool) & ed
        mm = m & r2c["ok"].astype(bool)
        s2 = np.concatenate([s, np.where(ed, s[:, 0], -np.inf)[:, None]], 1)
        in2 = np.concatenate([inR2, e_in[:, None]], 1)
        sf = np.where(in2, s2, -np.inf)
        i96 = amax(sf[:, :OC.N96])
        ix = np.where(in2.any(1), np.where(sf[:, OC.N96] > sf[ar, i96], OC.N96, i96), blend)
        La = np.where((ix == OC.N96)[:, None], el, lab[ar, np.minimum(ix, OC.N96 - 1)])
        take = ~inR2[:, 0] & e_in
        Lp = np.where(inR2[:, :1], top1, np.where(take[:, None], el, Ls))
        out["edit_brackets"] = {
            "edit-averse (R2 filter + blend over 96 + edit; the edit carries the top-1's full blend score)": dict(
                pdms=float(pdms(La)[mm].mean()), d_vs_blend=boot(pdms(La)[mm] - pdms(Lb)[mm], logs[mm]),
                d_vs_filter=boot(pdms(La)[mm] - pdms(lab[ar, filt['R2']])[mm], logs[mm]),
                break_vs_blend=rate(fails3(La) & mm & pass_b, mm & pass_b, lbt),
                frac_edit_picked=float((ix == OC.N96)[mm].mean())),
            "edit-preferred (top-1 if inside, else the R2 edit if inside after, else filter + blend)": dict(
                pdms=float(pdms(Lp)[mm].mean()), d_vs_blend=boot(pdms(Lp)[mm] - pdms(Lb)[mm], logs[mm]),
                d_vs_switch=boot(pdms(Lp)[mm] - pdms(Ls)[mm], logs[mm]),
                break_vs_blend=rate(fails3(Lp) & mm & pass_b, mm & pass_b, lbt),
                frac_edit_picked=float(take[mm].mean())),
        }
    return out


# ----------------------------------------------------------------------------------------------- criterion 1
def crit1(r1: dict, E: np.ndarray, tokm: np.ndarray, bins: dict, logs: np.ndarray) -> dict:
    V = r1["valid"] & tokm[:, None]
    lb = LogBoot(logs, OC.N96)
    col = np.broadcast_to(np.arange(OC.N96)[None], V.shape)
    fail = {k: E[..., KCOL[k]] < 1 for k in KEYS}
    fail["joint"] = fail["nc"] | fail["ttc"] | fail["dac"]
    unk = {k: r1[f"unk_{k}"] for k in KEYS}
    unk["joint"] = unk["nc"] | unk["ttc"] | unk["dac"]
    subsets = {"all96": V, "top1": V & (col == 0), "par16": V & (col < OC.N_PAR), "var80": V & (col >= OC.N_PAR)}
    out = {"n_tokens": int(tokm.sum()), "n_rows_all96": int(V.sum()), "n_logs": lb.L, "n_boot": A.N_BOOT, "reps": {}}
    for name, (a, b, c) in REPS_HO.items():
        hit = {"nc": r1[a], "ttc": r1[b], "dac": r1[c]}
        hit["joint"] = hit["nc"] | hit["ttc"] | hit["dac"]
        rr = {"role": ROLE_HO[name], "flags": [a, b, c]}
        for sn, mm in subsets.items():
            rr[sn] = {k: rc(hit[k], fail[k], mm, lb if (sn == "all96" or name == R2) else None) for k in hit}
        rr["all96_excl_unknown"] = {k: rc(hit[k], fail[k], V & ~unk[k]) for k in hit}
        rr["verdict_all96"] = all(rr["all96"][k]["ok"] for k in K4)
        a96 = rr["all96"]
        rr["ci_clear_pass"] = all(a96[k][c_][0] >= BAR for k in K4 for c_ in ("fail_ci", "pass_ci"))
        rr["ci_clear_fail"] = any(a96[k][c_][1] < BAR for k in K4 for c_ in ("fail_ci", "pass_ci"))
        if name in (R2, A.MAIN_REP, A.EXACT_REP):
            rr["scene_bins"] = {bn: {k: rc(hit[k], fail[k], V & bm[:, None], lb if name == R2 else None) for k in hit}
                                for bn, bm in bins.items()}
        out["reps"][name] = rr
    # R2 mismatch buckets: grid-free counterparts (stopped-aware exact NC proxy, exact TTC proxy, stored-SDF DAC) and
    # the official-rule emulation bounds
    hits = {"nc": r1["r2_nc"], "ttc": r1["r2_ttc"], "dac": r1["r2_dac"]}
    ref = {"nc": r1["x_nc_sa"], "ttc": r1["x_ttc"], "dac": r1["x_dac"]}
    bk = {}
    for k in KEYS:
        fn = fail[k] & ~hits[k] & V
        fp = ~fail[k] & hits[k] & V
        d = dict(FN=int(fn.sum()), FN_unknown=int((fn & unk[k]).sum()),
                 FN_grid_free_counterpart_sees=int((fn & ~unk[k] & ref[k]).sum()),
                 FN_grid_free_counterpart_misses=int((fn & ~unk[k] & ~ref[k]).sum()),
                 FP=int(fp.sum()), FP_grid_free_counterpart_passes=int((fp & ~ref[k]).sum()),
                 FP_grid_free_counterpart_hits=int((fp & ref[k]).sum()))
        if k in ("nc", "ttc"):
            lo, hi = r1[f"x_{k}_lo"], r1[f"x_{k}_hi"]
            d["FN_counterpart_misses_only_official_rules_see"] = int((fn & ~unk[k] & ~ref[k] & (lo | hi)).sum())
            d["FP_counterpart_hits_rule_not_at_fault"] = int((fp & ref[k] & ~lo).sum())
        bk[k] = d
    out["buckets_r2_all96"] = bk
    ddc = (E[..., I_DDC] < 1) & V
    jh = hits["nc"] | hits["ttc"] | hits["dac"]
    out["ddc_r2"] = dict(n_rows_ddc_fail=int(ddc.sum()), frac_rows=float(ddc.sum() / V.sum()),
                         ddc_only_fail=int((ddc & ~fail["joint"]).sum()),
                         ddc_only_and_inside=int((ddc & ~fail["joint"] & ~jh).sum()),
                         joint4=rc(jh, fail["joint"] | (E[..., I_DDC] < 1), V))
    out["unknown_rows"] = {k: int((unk[k] & V).sum()) for k in unk}
    out["n_oog_rows_r2"] = int(((r1["r2_noog"] > 0) & V).sum())
    return out


# ----------------------------------------------------------------------------------------------- criterion 2
def crit2(r2: dict, E: np.ndarray, logs: np.ndarray, tokm: np.ndarray, bins: dict, r1: dict) -> dict:
    names = [str(x) for x in r2["cfg_names"]]
    kinds = [str(x) for x in r2["kinds"]]
    m = r2["ok"].astype(bool) & tokm
    top1, head = E[:, 0], E[:, OC.COL_HEAD]
    pass1 = ~fails3(top1)
    lbt = LogBoot(logs, 1)
    out = {"n_tokens": int(m.sum()), "n_top1_pass3": int((m & pass1).sum()), "n_top1_fail3": int((m & ~pass1).sum())}
    for nm, lab in (("top1", top1), ("head", head)):
        out[nm] = dict(pdms=float(pdms(lab)[m].mean()), ep=float(100 * lab[m, I_EP].mean()),
                       fail3=rate(fails3(lab) & m, m, lbt))
    out["head"]["break_"] = rate(fails3(head) & m & pass1, m & pass1, lbt)
    out["head"]["fix"] = rate(~fails3(head) & m & ~pass1, m & ~pass1, lbt)
    out["head_vs_top1"] = boot(pdms(head)[m] - pdms(top1)[m], logs[m])
    hb = out["head"]["break_"]["rate"]
    out["configs"] = {}
    for q, nm in enumerate(names):
        lab = r2["lab"][:, q]
        k = r2["kind"][:, q]
        edited = r2["edited"][:, q].astype(bool)
        outside = k != kinds.index("inside")
        empty = np.isin(k, [kinds.index("empty_lon"), kinds.index("empty_lat")])
        noconv = k == kinds.index("noconv_lon")
        skip = np.isin(k, [kinds.index("slow"), kinds.index("short")])
        lon_e = m & edited & np.isin(k, [kinds.index("lon"), kinds.index("lon+lat")])
        lat_e = m & edited & np.isin(k, [kinds.index("lat"), kinds.index("lon+lat")])
        brk = rate(fails3(lab) & m & pass1, m & pass1, lbt)
        dh = boot(pdms(lab)[m] - pdms(head)[m], logs[m])
        c = dict(pdms=float(pdms(lab)[m].mean()), ep=float(100 * lab[m, I_EP].mean()),
                 d_vs_head=dh, d_vs_top1=boot(pdms(lab)[m] - pdms(top1)[m], logs[m]), break_=brk,
                 fix=rate(~fails3(lab) & m & ~pass1, m & ~pass1, lbt),
                 frac_outside=rate(outside & m, m, lbt), frac_edited=rate(edited & m, m, lbt),
                 frac_empty_of_outside_editable=rate(empty & m & outside & ~skip, m & outside & ~skip, lbt),
                 n_noconv=int((noconv & m).sum()), kinds={kn: int((k[m] == i).sum()) for i, kn in enumerate(kinds)},
                 outside_of_top1_fail=rate(outside & m & ~pass1, m & ~pass1, lbt),
                 outside_of_top1_pass=rate(outside & m & pass1, m & pass1, lbt),
                 n_lon_edits=int(lon_e.sum()), n_lat_edits=int(lat_e.sum()),
                 lon_edit_a_median=float(np.median(r2["a"][lon_e, q])) if lon_e.any() else float("nan"),
                 lat_edit_absD_median=float(np.median(np.abs(r2["D"][lat_e, q]))) if lat_e.any() else float("nan"),
                 inside_after_edited=float(r2["inside_after"][m & edited, q].mean()) if (m & edited).any()
                 else float("nan"),
                 edited_pass3=float((~fails3(lab))[m & edited].mean()) if (m & edited).any() else float("nan"),
                 d_vs_head_edited_tokens=boot((pdms(lab) - pdms(head))[m & edited], logs[m & edited])
                 if (m & edited).any() else None,
                 fail_comfort=float((lab[m, I_C] < 1).mean()))
        parts = dict(mean_ge_0p5=bool(dh["mean"] >= C2_MEAN_BAR), ci_lo_gt_0=bool(dh["lo"] > 0),
                     break_le_1p3=bool(brk["rate"] <= BREAK_BAR), break_le_head=bool(brk["rate"] <= hb))
        c["verdict"] = bool(all(parts.values()))
        c["verdict_parts"] = parts
        if nm == MAIN_CFG:
            c["scene_bins"] = {bn: dict(n=int((m & bm).sum()), d_vs_head=boot((pdms(lab) - pdms(head))[m & bm],
                                                                             logs[m & bm]),
                                        break_=rate(fails3(lab) & m & bm & pass1, m & bm & pass1, lbt),
                                        head_break=rate(fails3(head) & m & bm & pass1, m & bm & pass1, lbt))
                               for bn, bm in bins.items()}
            lon = lon_e
            to, dl = r2["t_out"][:, q], r2["delta"][:, q]
            c["lon_t_out_inf"] = int((lon & ~np.isfinite(to)).sum())
            c["critical_delta_hist"] = {f"{x:.1f} s": int((lon & np.isclose(dl, x)).sum()) for x in (0.0, 0.3, 0.6, 0.9)}
            # tube consistency: the R2 tube's top-1 flags == heldout_crit1's R2 flags of column 0
            o0 = r2["outside0"][:, q]
            c1 = np.stack([r1["r2_nc"][:, 0], r1["r2_ttc"][:, 0], r1["r2_dac"][:, 0]], 1)
            c["tube_flags_vs_crit1_mismatch_tokens"] = int((m[:, None] & (o0 != c1)).any(1).sum())
        out["configs"][nm] = c
    out["parity_max_abs_diff_vs_labels"] = float(np.nanmax(r2["par_max_diff"]))
    return out


# ----------------------------------------------------------------------------------------------- report
def bars(rr: dict):
    o = []
    for k in K4:
        x = rr["all96"][k]
        for side, ck in (("fail", "fail_ci"), ("pass", "pass_ci")):
            lo, hi = x.get(ck, [float("nan")] * 2)
            o.append((f"{k.upper()} {side}", x[f"{side}_recall"], lo, hi))
    return o


def below(rr: dict, which: str = "point") -> str:
    """bars under BAR: 'point' (the estimate, the verdict) or 'lo' (the CI lower bound: the bar is inside the CI)."""
    j = {"point": 1, "lo": 2}[which]
    xs = [f"{b[0]} {b[1]:.4f} [{b[2]:.3f}, {b[3]:.3f}]" for b in bars(rr) if b[j] < BAR]
    return ", ".join(xs) if xs else "none"


def summary_md(R: dict) -> str:
    S, c1, c2, ck = R["selection"], R["criterion1"], R["criterion2"], R["checks"]
    cs = S["criterion_S"]
    r2r = c1["reps"][R2]
    v1 = r2r["verdict_all96"]
    c2m = c2["configs"][MAIN_CFG]
    sw = S["picks"][cs["pick"]]
    bl = S["picks"]["blend over 96 (pool96_old_b1)"]
    L = ["# refiner v4_3 Phase 0b -- pre-registered held-out run (R2 tube, criterion S / 1 / 2)", "",
         f"Generated {R['generated']}. CPU only, no training, no GPU. Pre-registration: docs/refiner_v4/V4_3_SPEC.md "
         "'Phase 0b -- re-registration' (written before any held-out number). Code: tools/ck/v4/oracle/heldout_*.py "
         f"(SSR-v4, branch refiner-v4). Numbers: results.json in this folder. Held-out: {R['n_heldout']:,} navtrain_val "
         f"tokens not in the F1 6,000 sample ({S['n_tokens']:,} used), token-disjoint, same {R['n_logs']} logs. "
         "PDMS x 100. [95 % CI] = log-clustered bootstrap (214 logs, 2,000 resamples, seed 0).", "",
         "## Pre-registered verdict", "",
         f"- **Criterion S (primary; selection with the R2 GT tube): {'PASS' if cs['verdict'] else 'FAIL'}** -- "
         f"switch - blend dPDMS {ci(cs['d_vs_blend'])} (bar >= +1.0 and CI lower bound > 0: "
         f"{'yes' if cs['verdict_parts']['mean_ge_1p0'] and cs['verdict_parts']['ci_lo_gt_0'] else 'no'}); break "
         f"{pct(cs['break_'])} (bar <= 1.3 %: {'yes' if cs['verdict_parts']['break_le_1p3'] else 'no'}). "
         f"PDMS blend {bl['pdms']:.2f} -> switch {sw['pdms']:.2f}.",
         f"- **Criterion 1 (secondary; R2 fidelity, all 96 executed rows): {'PASS' if v1 else 'FAIL'}** -- "
         + ", ".join(f"{k.upper()} {rcell(r2r['all96'][k])}" for k in K4)
         + f" (fail / pass recall). Below the 0.90 bar (point estimates, the pre-registered verdict): {below(r2r)}; "
         f"bar inside the 95 % CI (lower bound < 0.90): {below(r2r, 'lo')}"
         + (" -- a marginal pass on the pass-recall side." if v1 and below(r2r, 'lo') != "none" else "."),
         f"- **Criterion 2 (informational; R2 computed edit of the v2 top-1 vs the DET lateral head): "
         f"{'PASS' if c2m['verdict'] else 'FAIL'} on Phase 0's rule** -- edit - head {ci(c2m['d_vs_head'])}, break "
         f"{pct(c2m['break_'])} vs head {pct(c2['head']['break_'])}.",
         "- **Decision (pre-registered rule)**: " + (
             "**S passes -> (c)**: retrain the DET / MAP range teachers with R2 targets and measure how much of the GT "
             "selection gain their learned fields recover (fixed-pool critic-only, val decides, navtest reported). GPU "
             "training needs the user's approval (GPUs 0-3 only); nothing was launched." if cs["verdict"] else
             "**S fails -> stop idea 3 and report.**")
         + (" Criterion 1 " + ("passes at the pre-registered point estimates (the 0.90 bars on NC / TTC / DAC / joint over "
                                "the executed 96), marginally on the pass-recall side (TTC / joint CIs include 0.90)."
                                if v1 else
                                "fails: per the pre-registration the learned tube is known to be an approximate "
                                "selection signal, not a faithful copy of the official rules.")),
         "- Every tube here is GT (privileged future objects + the official SDF): the S gain is an upper bound for a "
         "learned field, which is what (c) measures.",
         "- Context (no verdict): Phase 0's main grid tube in the same switch on the same held-out tokens gives "
         f"{ci(S['picks'][P0_SWITCH]['d_vs_blend'])} (break {pct(S['picks'][P0_SWITCH]['break_vs_blend'])}) vs R2 "
         f"{ci(cs['d_vs_blend'])}: the switch's selection gain does not come from the two R2 changes (they buy the "
         f"criterion-1 fidelity: Phase 0 main on these rows {'passes' if c1['reps'][A.MAIN_REP]['verdict_all96'] else 'fails'}"
         f", below the bar {below(c1['reps'][A.MAIN_REP])}). The filter without the switch (R2 tube filter + blend, "
         f"not pre-registered) gives {ci(S['picks']['R2 tube filter + blend']['d_vs_blend'])}; when the top-1 is inside, "
         f"keeping it is worth {ci(S['decomposition']['top-1 inside (switch = top-1)']['d_on_subset'])} vs the blend "
         "on those tokens.", "",
         "## Ambiguities in the pre-registration and how they were resolved (fixed in code before any held-out number)",
         ""]
    L += [f"{i + 1}. **{a}** -- {b}" for i, (a, b) in enumerate(AMBIGUITIES)]
    L += ["", "## Checks (data path; all before the criteria)", "",
          f"- LQR pipeline vs F1's official states, {ck['lqr_parity_f1']['n_tokens']} F1 tokens "
          f"({ck['lqr_parity_f1']['n_trajectories']:,} trajectories, the held-out code path): max |dxy| "
          f"{ck['lqr_parity_f1']['max_dxy']:.1e} m (bar 1e-4), |dheading| {ck['lqr_parity_f1']['max_dh']:.1e} rad, |dv| "
          f"{ck['lqr_parity_f1']['max_dv']:.1e} m/s; ego init (status v0, a0, 0) == F1 init on all 6,000 F1 rows "
          f"(max diff {ck['init_f1_max_abs_diff']}).",
          f"- Label assembly (labels.npy + shard time indices; cand, r34 variants, DET head = labels_stack 'v2_lat' == "
          f"lat_traj16[:, 0] bitwise) == F1 exec on the 6,000 F1 rows x 97 columns: "
          f"{'bit-identical' if ck['label_assembly_f1']['bit_identical'] else 'DIFFERS'}.",
          f"- Official simulator + scorer on {ck['rescore']['n_tokens']} held-out tokens ({ck['rescore']['n_trajectories']:,}"
          f" trajectories, one call per token): states vs ck/v4/lqr max |dxy| {ck['rescore']['states_max_dxy_m']:.1e} m, "
          f"|dheading| {ck['rescore']['states_max_dheading_rad']:.1e} rad, |dv| {ck['rescore']['states_max_dv']:.1e} m/s; "
          f"labels vs the stored files max |diff| nc/dac/ttc {max(ck['rescore']['label_max_abs_diff'][k] for k in ('nc', 'dac', 'ttc'))}, "
          f"pdms {ck['rescore']['label_max_abs_diff']['pdms']:.1e} (float32 storage), time-index mismatches "
          f"{ck['rescore']['time_idx_mismatches']}.",
          f"- Executed-state flags vs official labels on ALL held-out rows (the Phase 0 grid-free checks; F1 value in "
          f"brackets): {ck['flag_agreement_text']}.",
          f"- R2 own code path vs Phase 0's 'grid stopped-aware NC + read any4' flags: {ck['r2_code_mismatch_rows']} "
          f"differing rows; criterion-2 R2 tube top-1 flags vs criterion-1 R2 flags: "
          f"{c2m['tube_flags_vs_crit1_mismatch_tokens']} differing tokens; official re-score of the top-1 / head in the "
          f"criterion-2 calls vs stored labels: max |diff| {c2['parity_max_abs_diff_vs_labels']:.1e}.",
          f"- Blend pick vs the stored pool96_old_b1 picks on the held-out tokens: "
          + (f"{100 * S['blend_vs_stored_pool96_old_b1']['identical_index']:.2f} % identical index "
             f"({S['blend_vs_stored_pool96_old_b1']['n_differ']} differ)." if 'identical_index' in
             S['blend_vs_stored_pool96_old_b1'] else str(S['blend_vs_stored_pool96_old_b1'])),
          f"- UNKNOWN rows (official failure object absent from the GT store / beyond its horizon; DAC corners outside the "
          f"SDF grid): {c1['unknown_rows']}.", "",
          "## Criterion S -- selection with the R2 GT tube (held-out)", "",
          "| pick | PDMS | vs blend | vs top-1 | vs head | break vs blend | fix of blend fails | fail NC/TTC/DAC | differs from blend |",
          "|---|---|---|---|---|---|---|---|---|"]
    for nm, p in S["picks"].items():
        L.append(f"| {nm}{' **(criterion S)**' if nm == cs['pick'] else ''} | {p['pdms']:.2f} | {ci(p['d_vs_blend'])} | "
                 f"{ci(p['d_vs_top1'])} | {ci(p['d_vs_head'])} | {pct(p['break_vs_blend'])} | {pct(p['fix_vs_blend'], 1)} | "
                 f"{pct(p['fail3'], 1)} | {100 * p['frac_differs_from_blend']:.1f} % |")
    L += ["", "Where the switch gain comes from (tokens of the criterion-S set):", "",
          "| subset | tokens | share | dPDMS on subset | contribution per token | pick differs from blend |",
          "|---|---|---|---|---|---|"]
    for nm, d in S["decomposition"].items():
        L.append(f"| {nm} | {d['n']:,} | {100 * d['frac']:.1f} % | {ci(d['d_on_subset']) if d['d_on_subset'] else 'n/a'} | "
                 f"{d['contribution_per_token']:+.2f} | {100 * d['frac_pick_differs_from_blend']:.1f} % |")
    L += ["", f"Rows inside the R2 tube: {100 * S['frac_rows_inside_R2']:.1f} % of the valid 96.", "",
          "| scene bin | tokens | switch - blend | break vs blend |", "|---|---|---|---|"]
    for bn, b in S["scene_bins"].items():
        L.append(f"| {bn} | {b['n']:,} | {ci(b['d_vs_blend'])} | {pct(b['break_'])} |")
    if "edit_brackets" in S:
        L += ["", "Computed R2 edit as an extra candidate (no critic score of the edit exists; the two extremes, as Phase 0):", ""]
        for nm, b in S["edit_brackets"].items():
            ref = "d_vs_filter" if "d_vs_filter" in b else "d_vs_switch"
            L.append(f"- {nm}: PDMS {b['pdms']:.2f}, vs blend {ci(b['d_vs_blend'])}, vs "
                     f"{'R2 filter + blend' if ref == 'd_vs_filter' else 'the switch'} {ci(b[ref])}, break "
                     f"{pct(b['break_vs_blend'])}, edit picked {100 * b['frac_edit_picked']:.2f} %.")
    f1s = R.get("f1_context", {}).get("criterion_S")
    if f1s:
        L += ["", f"Context (not the verdict): the same criterion S on the F1 6,000 tokens, where R2 was chosen post hoc -- "
              f"switch - blend {ci(f1s['d_vs_blend'])}, break {pct(f1s['break_'])}."]
    L += ["", "## Criterion 1 -- outside the tube <=> official fail (held-out, executed 96)", "",
          f"Tokens {c1['n_tokens']:,}, rows {c1['n_rows_all96']:,}. Verdict on the point estimates (8 bars >= 0.90). The "
          "Phase 0 representations are read on the same held-out rows for context (their roles refer to Phase 0; no "
          "verdict uses them).", "",
          "| representation | role | NC fail / pass | TTC fail / pass | DAC fail / pass | joint fail / pass | 8 bars? |",
          "|---|---|---|---|---|---|---|"]
    for nm, rr in c1["reps"].items():
        a = rr["all96"]
        L.append(f"| {nm}{' (= R2 flags, Phase 0 code path)' if nm == A.ANY4_SA_REP else ''} | {rr['role']} | "
                 + " | ".join(rcell(a[k]) for k in K4)
                 + f" | {'yes' if rr['verdict_all96'] else 'no'}{' (CI-clear)' if rr['ci_clear_pass'] else ''} |")
    L += ["", "R2 by subset (fail / pass recall [CI], n_fail):", "", "| subset | NC | TTC | DAC | joint |", "|---|---|---|---|---|"]
    for sn in ("top1", "par16", "var80", "all96"):
        a = r2r[sn]
        L.append(f"| {sn} | " + " | ".join(f"{rcell(a[k])} ({a[k]['n_fail']:,})" for k in K4) + " |")
    L += ["", "R2 scene bins (all 96; fail / pass recall [CI], n_fail):", "", "| bin | NC | TTC | DAC | joint |",
          "|---|---|---|---|---|"]
    for bn, a in r2r["scene_bins"].items():
        L.append(f"| {bn} | " + " | ".join(f"{rcell(a[k])} ({a[k]['n_fail']:,})" for k in K4) + " |")
    ex = r2r["all96_excl_unknown"]
    L += ["", "R2 excluding UNKNOWN failures: " + ", ".join(f"{k.upper()} {f3(ex[k]['fail_recall'])} / "
                                                         f"{f4(ex[k]['pass_recall'])}" for k in K4) + ".", "",
          "R2 mismatch buckets (all 96; grid-free counterpart = the same rule on exact boxes: stopped-aware exact NC "
          "proxy, exact TTC proxy, stored-SDF corners):", ""]
    for k, d in c1["buckets_r2_all96"].items():
        L.append(f"- {k}: " + ", ".join(f"{kk} {vv:,}" for kk, vv in d.items()))
    dd = c1["ddc_r2"]
    L += [f"- DDC (not a tube key): {dd['n_rows_ddc_fail']:,} rows fail DDC ({100 * dd['frac_rows']:.2f} %), "
          f"{dd['ddc_only_fail']:,} of them pass NC / TTC / DAC, {dd['ddc_only_and_inside']:,} of those inside R2; joint "
          f"incl. DDC: {f3(dd['joint4']['fail_recall'])} / {f4(dd['joint4']['pass_recall'])}. Executed rows with states "
          f"off the parent grid: {c1['n_oog_rows_r2']:,}.", "",
          "## Criterion 2 -- R2 computed edit of the v2 top-1 vs the DET lateral head (informational)", "",
          f"Tokens {c2['n_tokens']:,} (top-1 passes NC/TTC/DAC on {c2['n_top1_pass3']:,}, fails on {c2['n_top1_fail3']:,}). "
          f"v2 top-1 PDMS {c2['top1']['pdms']:.2f}; DET lateral head {c2['head']['pdms']:.2f} (head - top-1 "
          f"{ci(c2['head_vs_top1'])}); head break {pct(c2['head']['break_'])}, head fix {pct(c2['head']['fix'], 1)}.", "",
          "| config | PDMS | edit - head | edit - top-1 | break | fix | outside | edited | empty (of editable outside) | EP | Phase 0 rule |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, c in c2["configs"].items():
        L.append(f"| {nm}{' **(criterion 2)**' if nm == MAIN_CFG else ''} | {c['pdms']:.2f} | {ci(c['d_vs_head'])} | "
                 f"{ci(c['d_vs_top1'])} | {pct(c['break_'])} | {pct(c['fix'], 1)} | {pct(c['frac_outside'])} | "
                 f"{pct(c['frac_edited'])} | {pct(c['frac_empty_of_outside_editable'], 1)} | {c['ep']:.2f} | "
                 f"{'PASS' if c['verdict'] else 'fail'} |")
    L += ["", f"{MAIN_CFG}: kinds {c2m['kinds']}; noconv {c2m['n_noconv']}; outside of failing top-1 "
          f"{pct(c2m['outside_of_top1_fail'], 1)}, of passing top-1 {pct(c2m['outside_of_top1_pass'])}; lon edits "
          f"{c2m['n_lon_edits']} (median a {c2m['lon_edit_a_median']:.2f} m/s^2; t_out = +inf in {c2m['lon_t_out_inf']}; "
          f"critical read delta {c2m['critical_delta_hist']}), lat edits {c2m['n_lat_edits']} (median |D| "
          f"{c2m['lat_edit_absD_median']:.2f} m); edited tokens inside after {f3(c2m['inside_after_edited'])}, pass "
          f"NC/TTC/DAC {f3(c2m['edited_pass3'])}, edit - head on edited tokens "
          f"{ci(c2m['d_vs_head_edited_tokens']) if c2m['d_vs_head_edited_tokens'] else 'n/a'}; comfort fail "
          f"{100 * c2m['fail_comfort']:.2f} %.", "", "| scene bin | n | edit - head | break | head break |",
          "|---|---|---|---|---|"]
    for bn, b in c2m["scene_bins"].items():
        L.append(f"| {bn} | {b['n']:,} | {ci(b['d_vs_head'])} | {pct(b['break_'])} | {pct(b['head_break'])} |")
    L += ["", "## Notes", ""] + [f"- {x}" for x in R["notes"]]
    return "\n".join(L) + "\n"


def flag_agreement(c1: dict, f1: dict) -> str:
    """official-rule consistency of the executed states: grid-free reps on held-out vs F1 (Phase 0 results)."""
    out = []
    for nm, keys in ((A.EXACT_REP, ("dac",)), ("exact official-rule emulation hi (lane map: always)", ("nc", "ttc"))):
        a = c1["reps"][nm]["all96"]
        b = f1["criterion1"]["reps"][nm]["all96"] if f1 else None
        for k in keys:
            lab = {"dac": "DAC = stored-SDF corner < 0", "nc": "NC official-rule emulation (hi)",
                   "ttc": "TTC official-rule emulation (hi)"}[k]
            s = f"{lab} {f3(a[k]['fail_recall'])} / {f4(a[k]['pass_recall'])}"
            if b:
                s += f" [{f3(b[k]['fail_recall'])} / {f4(b[k]['pass_recall'])}]"
            out.append(s)
    return "; ".join(out) + " (fail / pass recall vs the official label)"


def f1_context_S(bins_fn) -> dict:
    """criterion S on the F1 tokens with the R2 flags of Phase 0's crit1 rows (context only)."""
    r1 = dict(np.load(OC.OUT / "crit1/rows.npz"))
    r1 = dict(r1, r2_nc=r1["g25sa_nc"], r2_ttc=r1["g25a_ttc"], r2_dac=r1["g25a_dac"])
    rows = np.load(OC.PREP / "f1_rows.npy")
    E = np.load(OC.PREP / "f1_exec.npy").astype(np.float64)
    fin = np.load(OC.PREP / "f1_fin.npy")
    tok = pd.read_parquet(OC.PACK / "tokens.parquet")
    logs = tok["log"].astype(str).to_numpy()[rows]
    tokm = (r1["gt_ok"] > 0) & (r1["sdf_ok"] > 0) & fin[:, 0] & fin[:, OC.COL_HEAD]
    S = selection(r1, E, rows, logs, tokm, bins_fn(r1, rows))
    return {"n_tokens": S["n_tokens"], "criterion_S": S["criterion_S"],
            "switch_pdms": S["picks"][S["criterion_S"]["pick"]]["pdms"],
            "blend_pdms": S["picks"]["blend over 96 (pool96_old_b1)"]["pdms"]}


def main():
    rows = np.load(H.HO_PREP / "ho_rows.npy")
    E = np.load(H.HO_PREP / "ho_exec.npy").astype(np.float64)
    fin = np.load(H.HO_PREP / "ho_fin.npy")
    tok = pd.read_parquet(OC.PACK / "tokens.parquet")
    logs = tok["log"].astype(str).to_numpy()[rows]
    r1 = dict(np.load(H.HO / "crit1/rows.npz"))
    r2 = dict(np.load(H.HO / "crit2/rows.npz"))
    assert np.array_equal(r1["i"], np.arange(len(rows))) and np.array_equal(r2["i"], np.arange(len(rows)))
    assert np.array_equal(r1["row"], rows) and np.array_equal(r2["row"], rows)
    tokm = (r1["gt_ok"] > 0) & (r1["sdf_ok"] > 0) & fin[:, 0] & fin[:, OC.COL_HEAD]
    lead = pd.read_parquet(OC.LEAD)

    def bins_fn(r, rw):
        lg = lead["lead_gap"].to_numpy()[rw]
        return {"lead 10-25 m": (lead["has_lead"].to_numpy()[rw] == 1) & (lg >= 10) & (lg <= 25),
                "near curb (top-1 min corner SDF < 1 m)": r["top1_min_corner"] < 1.0,
                "lane-only (no object within 30 m)": r["top1_obj_dmin"] >= 30.0}

    bins = bins_fn(r1, rows)
    R = {"generated": H.kst(), "split": OC.SPLIT, "n_heldout": int(len(rows)), "n_tokens_used": int(tokm.sum()),
         "n_logs": int(len(set(logs))), "scene_bin_counts": {k: int((v & tokm).sum()) for k, v in bins.items()},
         "preregistration": "docs/refiner_v4/V4_3_SPEC.md 'Phase 0b -- re-registration'",
         "ambiguities": [dict(item=a, resolution=b) for a, b in AMBIGUITIES]}
    R["selection"] = selection(r1, E, rows, logs, tokm, bins, r2)
    R["criterion1"] = crit1(r1, E[:, :OC.N96], tokm, bins, logs)
    R["criterion2"] = crit2(r2, E, logs, tokm, bins, r1)
    R["verdict"] = dict(criterion_S=R["selection"]["criterion_S"]["verdict"],
                        criterion_1=R["criterion1"]["reps"][R2]["verdict_all96"],
                        criterion_2_informational=R["criterion2"]["configs"][MAIN_CFG]["verdict"],
                        decision="S passes -> (c): retrain the DET / MAP range teachers with R2 targets (needs GPU "
                                 "approval)" if R["selection"]["criterion_S"]["verdict"] else
                                 "S fails -> stop idea 3 and report")
    try:
        f1 = json.loads((OC.OUT / "results.json").read_text())
    except Exception:  # noqa: BLE001
        f1 = None
    try:
        R["f1_context"] = f1_context_S(bins_fn)
    except Exception as e:  # noqa: BLE001
        R["f1_context"] = {"error": str(e)}
    prep = json.loads((H.HO_PREP / "prep.json").read_text())
    resc = json.loads((H.HO / "checks/rescore.json").read_text())
    run1 = json.loads((H.HO / "crit1/run.json").read_text())
    run2 = json.loads((H.HO / "crit2/run.json").read_text())
    R["checks"] = dict(lqr_parity_f1=prep["lqr_parity_f1"], init_f1_max_abs_diff=prep["init_f1_max_abs_diff"],
                       label_assembly_f1=prep["label_assembly_f1"], rescore=resc,
                       r2_code_mismatch_rows=run1["r2_code_mismatch_rows"],
                       gt_missing=run1["gt_missing"], sdf_missing=run1["sdf_missing"],
                       flag_agreement_text=flag_agreement(R["criterion1"], f1))
    R["notes"] = [
        "Executed states: ck/v4/lqr float64 (HO/prep/ho_sim.npy, 2.3 GB, deletable / reproducible with heldout_prep.py).",
        "Official labels: the stored label files (cand, r34 variants, ck2T10 stack 'v2_lat'); criterion-2 edits scored by "
        "tools/refiner/score_trajectories.score_token (plain navsim classes, one call per token with an edit).",
        "Criterion S picks use no label: the R2 flags are GT-field reads of the executed states; the comparator is the "
        "deployment blend over the same 96.",
        "The Phase 0 representations on the held-out rows (criterion 1 table) and the F1 criterion S are context; the "
        "pre-registered verdict uses only R2 on the held-out tokens.",
        "EPDMS: not computed (spec: no navtrain_val v2.2 metric cache; not approved).",
        f"Runtime (<= 28 CPU workers, nice 10): prep {prep['sec']:.0f} s, criterion 1 {run1['sec']:.0f} s, criterion 2 "
        f"{run2['sec']:.0f} s ({run2['n_scored_tokens']:,} tokens re-scored), official re-score check {resc['sec']:.0f} s.",
    ]
    (H.HO / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (H.HO / "summary.md").write_text(summary_md(R))
    print(summary_md(R))


if __name__ == "__main__":
    main()
