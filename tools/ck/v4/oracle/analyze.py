#!/usr/bin/env python
"""refiner v4_3 Phase 0 oracle -- analysis: criterion 1 + criterion 2 verdicts, buckets, scene bins, margin sensitivity,
speed-cap probe, selection-only oracle -> OUT/results.json + OUT/summary.md.  CPU only, reads OUT/prep, OUT/crit1,
OUT/crit2 and the read-only inputs listed in oracle_common.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/analyze.py

Definitions
  official fail: nc < 1 (0 agent, 0.5 static: both fail), ttc < 1, dac < 1; joint = any of the three (DDC is not a tube
    key: its failures are reported as their own bucket, and 'joint4' adds them to the failures).
  criterion 1 (two-sided, spec): on the 96 (par16 + valid var80; executed states), per key and joint, fail recall =
    P(outside | official fail) >= 0.9 AND pass recall = P(inside | official pass) >= 0.9 (point estimates: the
    pre-registered verdict).  'excl. UNKNOWN' drops the official failures the GT store cannot see (run_crit1 unk_*).
    95 % CIs: log-clustered bootstrap of the ratio estimator (resample the logs with replacement, recall = the sum of
    the per-log tp (tn) over the sum of the per-log n_fail (n_pass); 2,000 resamples, seed 0, the same multinomial
    log weights as tools/ck/eval_ck.log_bootstrap); 'ci_clear_pass' = every lower bound >= 0.9, 'ci_clear_fail' =
    some upper bound < 0.9 (descriptive only).  Representation roles (REPS): 'main' (pre-registered), 'ablation'
    (spec-listed bin size / point target; TTC without LON), 'post hoc' (chosen after seeing the misses), and
    'grid-free' (exact boxes, no B / D grid: locate misses, not a trainable tube target).
  criterion 2 (spec): per token on the v2 top-1 parent, dPDMS = PDMS(edit) - PDMS(DET lateral head on the same
    parent) (x 100, executed labels), paired log-clustered bootstrap (tools/ck/eval_ck.log_bootstrap, 2,000, seed 0):
    pass iff mean >= +0.5 and CI lower bound > 0, and break = P(edit fails NC or TTC or DAC | the top-1 passes all
    three) <= 1.3 % and <= the head's break on the same tokens.
  scene bins (token level): lead = ck/lead has_lead and 10 <= lead_gap <= 25 m; near curb = top-1 executed min corner
    SDF < 1 m; lane-only = no OBS GT box centre within 30 m of the top-1 executed rear axle over 0..4 s.
  selection-only oracle: the deployment blend score = select.ck_final(prob, v2_im) (SEL_W default, beta 1 = refiner v2
    'old_b1'; DET teacher ck2T10 critic probabilities, variants inherit the parent's im) among the 96; with the GT
    tube flags as a filter (pick the best-blend candidate inside the tube; none inside -> the blend pick); also
    'switch' (keep the top-1 when it is inside, else the filtered pick) and two extremes for the edit as an extra
    candidate (no critic score of the edit exists here; spec M5 scores it with the critic and lets it inherit only
    v2_final / im): 'select + edit' = the main computed edit joins the pool with its tube flag and the top-1's FULL
    blend score (incl. the outside top-1's own critic probability: the edit-averse extreme, it wins only strictly), and
    'edit-preferred bound' = the top-1 when inside, else the edit when it is inside after editing, else filter + blend
    (the edit-favouring extreme).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_common as OC  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck.eval_ck import log_bootstrap  # noqa: E402
from navsim.agents.para_ssr.ck.select import ck_final  # noqa: E402

I_NC, I_DAC, I_DDC, I_EP, I_TTC, I_C, I_PDMS = 0, 1, 2, 3, 4, 5, 6
KEYS = ("nc", "ttc", "dac")
KCOL = {"nc": I_NC, "ttc": I_TTC, "dac": I_DAC}


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M KST", time.gmtime(time.time() + 9 * 3600))


class LogBoot:
    """Log-clustered bootstrap of row-level ratios (module docstring 'criterion 1'): rows [n_tok, C] map to the log of
    their token; W [N_BOOT, L] multinomial log counts drawn as tools/ck/eval_ck.log_bootstrap (seed BOOT_SEED)."""

    def __init__(self, logs_tok: np.ndarray, n_cols: int):
        ul, inv = np.unique(np.asarray(logs_tok).astype(str), return_inverse=True)
        self.L = len(ul)
        self.inv = np.broadcast_to(inv[:, None], (len(inv), n_cols))
        rng = np.random.default_rng(BOOT_SEED)
        self.W = rng.multinomial(self.L, np.full(self.L, 1.0 / self.L), size=N_BOOT).astype(np.float64)   # [B, L]

    def ratio_ci(self, num: np.ndarray, den: np.ndarray, alpha: float = 0.05) -> list:
        """bool masks num (subset of den) [n_tok, C] -> [lo, hi] of sum(num) / sum(den) over resampled logs."""
        if not den.any():
            return [float("nan"), float("nan")]
        n = self.W @ np.bincount(self.inv[num], minlength=self.L).astype(np.float64)
        d = self.W @ np.bincount(self.inv[den], minlength=self.L).astype(np.float64)
        est = np.where(d > 0, n / np.maximum(d, 1.0), np.nan)
        lo, hi = np.nanquantile(est, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]


def rc(hit: np.ndarray, fail: np.ndarray, m: np.ndarray, lb: Optional[LogBoot] = None) -> dict:
    """fail / pass recall of 'outside (hit) <=> official fail' on the rows m (+ log-bootstrap CIs with lb)."""
    f, p = fail & m, ~fail & m
    nf, npss = int(f.sum()), int(p.sum())
    tp, tn = int((hit & f).sum()), int((~hit & p).sum())
    fr = tp / nf if nf else float("nan")
    pr = tn / npss if npss else float("nan")
    o = dict(n_fail=nf, n_pass=npss, tp=tp, tn=tn, fail_recall=fr, pass_recall=pr,
             ok=bool(nf and npss and fr >= BAR and pr >= BAR))
    if lb is not None:
        o["fail_ci"] = lb.ratio_ci(hit & f, f)
        o["pass_ci"] = lb.ratio_ci(~hit & p, p)
    return o


def boot(d: np.ndarray, logs: np.ndarray) -> dict:
    b = log_bootstrap(np.asarray(d, np.float64), logs, 2000, 0)
    return {k: (float(v) if isinstance(v, float) else v) for k, v in b.items()}


def fails3(lab: np.ndarray) -> np.ndarray:
    return (lab[..., I_NC] < 1) | (lab[..., I_TTC] < 1) | (lab[..., I_DAC] < 1)


# ----------------------------------------------------------------------------------------------- criterion 1
REPS = {   # name -> (nc flag, ttc flag, dac flag) keys of crit1/rows.npz
    "grid_main (0.25 s, nearest, box, TTC LON, DAC on D)": ("g25_nc", "g25_ttc", "g25_dac"),
    "grid 0.1 s bins": ("g10_nc", "g10_ttc", "g25_dac"),
    "grid 0.5 s bins": ("g50_nc", "g50_ttc", "g25_dac"),
    "grid read any4": ("g25a_nc", "g25a_ttc", "g25a_dac"),
    "grid TTC on B (no LON)": ("g25_nc", "g25b_ttc", "g25_dac"),
    "grid point target (O at 9 footprint points)": ("p25_nc", "p25_ttc", "g25_dac"),
    "grid stopped-aware NC (nearest)": ("g25s_nc", "g25_ttc", "g25_dac"),
    "grid stopped-aware NC + read any4": ("g25sa_nc", "g25a_ttc", "g25a_dac"),
    "exact proxy (no grid, field rules)": ("x_nc", "x_ttc", "x_dac"),
    "exact proxy, stopped-aware NC": ("x_nc_sa", "x_ttc", "x_dac"),
    "exact official-rule emulation lo (lane map: off-road only)": ("x_nc_lo", "x_ttc_lo", "x_dac"),
    "exact official-rule emulation hi (lane map: always)": ("x_nc_hi", "x_ttc_hi", "x_dac"),
}
MAIN_REP = "grid_main (0.25 s, nearest, box, TTC LON, DAC on D)"
EXACT_REP = "exact proxy (no grid, field rules)"
ANY4_REP = "grid read any4"
ANY4_SA_REP = "grid stopped-aware NC + read any4"
SA_REP = "grid stopped-aware NC (nearest)"
ROLE = {   # module docstring 'Representation roles'; only 'main' carries the pre-registered verdict
    MAIN_REP: "main", "grid 0.1 s bins": "ablation", "grid 0.5 s bins": "ablation", ANY4_REP: "post hoc",
    "grid TTC on B (no LON)": "ablation", "grid point target (O at 9 footprint points)": "ablation",
    SA_REP: "post hoc", ANY4_SA_REP: "post hoc", EXACT_REP: "grid-free", "exact proxy, stopped-aware NC": "grid-free",
    "exact official-rule emulation lo (lane map: off-road only)": "grid-free",
    "exact official-rule emulation hi (lane map: always)": "grid-free",
}
assert set(ROLE) == set(REPS)
N_BOOT, BOOT_SEED = 2000, 0
BAR = 0.9


def crit1(r1: dict, E: np.ndarray, tokm: np.ndarray, bins: dict, logs: np.ndarray) -> dict:
    V = r1["valid"] & tokm[:, None]
    lb = LogBoot(logs, OC.N96)
    col = np.broadcast_to(np.arange(OC.N96)[None], V.shape)
    fail = {k: E[..., KCOL[k]] < 1 for k in KEYS}
    fail["joint"] = fail["nc"] | fail["ttc"] | fail["dac"]
    unk = {k: r1[f"unk_{k}"] for k in KEYS}
    unk["joint"] = unk["nc"] | unk["ttc"] | unk["dac"]
    subsets = {"all96": V, "top1": V & (col == 0), "par16": V & (col < OC.N_PAR), "var80": V & (col >= OC.N_PAR)}
    out = {"n_tokens": int(tokm.sum()), "n_rows_all96": int(V.sum()), "n_logs": lb.L, "n_boot": N_BOOT, "reps": {}}
    K4 = ("nc", "ttc", "dac", "joint")
    for name, (a, b, c) in REPS.items():
        hit = {"nc": r1[a], "ttc": r1[b], "dac": r1[c]}
        hit["joint"] = hit["nc"] | hit["ttc"] | hit["dac"]
        rr = {"role": ROLE[name]}
        for sn, m in subsets.items():
            rr[sn] = {k: rc(hit[k], fail[k], m, lb) for k in hit}
            rr[sn + "_excl_unknown"] = {k: rc(hit[k], fail[k], m & ~unk[k]) for k in hit}
        rr["verdict_all96"] = all(rr["all96"][k]["ok"] for k in K4)
        rr["verdict_all96_excl_unknown"] = all(rr["all96_excl_unknown"][k]["ok"] for k in K4)
        a96 = rr["all96"]
        rr["ci_clear_pass"] = all(a96[k][c_][0] >= BAR for k in K4 for c_ in ("fail_ci", "pass_ci"))
        rr["ci_clear_fail"] = any(a96[k][c_][1] < BAR for k in K4 for c_ in ("fail_ci", "pass_ci"))
        if name in (MAIN_REP, EXACT_REP):
            rr["scene_bins"] = {bn: {k: rc(hit[k], fail[k], V & bm[:, None]) for k in hit} for bn, bm in bins.items()}
        out["reps"][name] = rr
    # mismatch buckets of the main grid representation and the exact proxy (all96)
    bk = {}
    for name, pre in ((MAIN_REP, "g25"), (EXACT_REP, "x")):
        a, b, c = REPS[name]
        hits = {"nc": r1[a], "ttc": r1[b], "dac": r1[c]}
        d = {}
        for k in KEYS:
            fn = fail[k] & ~hits[k] & V
            fp = ~fail[k] & hits[k] & V
            if k == "dac":
                xh = r1["x_dac"]
                d[k] = dict(FN=int(fn.sum()), FN_unknown=int((fn & unk[k]).sum()),
                            FN_exact_sees=int((fn & ~unk[k] & xh).sum()),
                            FN_other=int((fn & ~unk[k] & ~xh).sum()),
                            FP=int(fp.sum()), FP_exact_pass=int((fp & ~xh).sum()), FP_exact_hit=int((fp & xh).sum()))
                continue
            xp, lo, hi = r1[f"x_{k}"], r1[f"x_{k}_lo"], r1[f"x_{k}_hi"]
            d[k] = dict(FN=int(fn.sum()), FN_unknown=int((fn & unk[k]).sum()),
                        FN_exact_proxy_sees=int((fn & ~unk[k] & xp).sum()),
                        FN_only_official_rules_see=int((fn & ~unk[k] & ~xp & (lo | hi)).sum()),
                        **({"FN_stopped_aware_exact_sees": int((fn & ~unk[k] & ~xp & r1["x_nc_sa"]).sum())}
                           if k == "nc" else {}),
                        FN_other=int((fn & ~unk[k] & ~xp & ~lo & ~hi).sum()),
                        FP=int(fp.sum()), FP_exact_proxy_pass=int((fp & ~xp).sum()),
                        FP_rule_not_at_fault=int((fp & xp & ~lo).sum()),
                        FP_all_rules_hit=int((fp & xp & lo).sum()))
        bk[name] = d
    out["buckets_all96"] = bk
    # DDC bucket: failures the tube cannot see by construction
    ddc = (E[..., I_DDC] < 1) & V
    jh = r1["g25_nc"] | r1["g25_ttc"] | r1["g25_dac"]
    out["ddc"] = dict(n_rows_ddc_fail=int(ddc.sum()), frac_rows=float(ddc.sum() / V.sum()),
                      ddc_only_fail=int((ddc & ~fail["joint"]).sum()),
                      ddc_only_and_main_inside=int((ddc & ~fail["joint"] & ~jh).sum()),
                      joint4_main=rc(jh, fail["joint"] | (E[..., I_DDC] < 1), V))
    out["unknown_rows"] = {k: int((unk[k] & V).sum()) for k in unk}
    out["n_oog_rows_main"] = int(((r1["g25_noog"] > 0) & V).sum())
    return out


# ----------------------------------------------------------------------------------------------- criterion 2
def crit2(r2: dict, E: np.ndarray, P: np.ndarray, logs: np.ndarray, tokm: np.ndarray, bins: dict,
          has_lead: np.ndarray) -> dict:
    names = [str(x) for x in r2["cfg_names"]]
    kinds = [str(x) for x in r2["kinds"]]
    m = r2["ok"].astype(bool) & tokm
    top1, head = E[:, 0], E[:, OC.COL_HEAD]
    pdms = lambda lab: 100.0 * lab[..., I_PDMS]
    pass1 = ~fails3(top1)
    out = {"n_tokens": int(m.sum()), "top1": {}, "head": {}, "configs": {}}
    for nm, lab in (("top1", top1), ("head", head)):
        out[nm] = dict(pdms=float(pdms(lab)[m].mean()), ep=float(100 * lab[m, I_EP].mean()),
                       fail_nc=float((lab[m, I_NC] < 1).mean()), fail_ttc=float((lab[m, I_TTC] < 1).mean()),
                       fail_dac=float((lab[m, I_DAC] < 1).mean()),
                       break_=float(fails3(lab)[m & pass1].mean()) if nm == "head" else 0.0,
                       fix=float((~fails3(lab))[m & ~pass1].mean()) if nm == "head" else 0.0)
    out["head_vs_top1"] = boot(pdms(head)[m] - pdms(top1)[m], logs[m])
    out["n_top1_pass3"] = int((m & pass1).sum())
    out["n_top1_fail3"] = int((m & ~pass1).sum())
    head_break = out["head"]["break_"]
    for q, nm in enumerate(names):
        lab = r2["lab"][:, q]
        k = r2["kind"][:, q]
        edited = r2["edited"][:, q].astype(bool)
        outside = k != kinds.index("inside")
        empty = np.isin(k, [kinds.index("empty_lon"), kinds.index("empty_lat")])
        noconv = (k == kinds.index("noconv_lon")) if "noconv_lon" in kinds else np.zeros_like(empty)
        skip = np.isin(k, [kinds.index("slow"), kinds.index("short")])
        lon_e = m & edited & np.isin(k, [kinds.index("lon"), kinds.index("lon+lat")])    # edits with an a
        lat_e = m & edited & np.isin(k, [kinds.index("lat"), kinds.index("lon+lat")])    # edits with a D
        brk = float(fails3(lab)[m & pass1].mean())
        dh = boot(pdms(lab)[m] - pdms(head)[m], logs[m])
        c = dict(
            pdms=float(pdms(lab)[m].mean()), ep=float(100 * lab[m, I_EP].mean()),
            fail_nc=float((lab[m, I_NC] < 1).mean()), fail_ttc=float((lab[m, I_TTC] < 1).mean()),
            fail_dac=float((lab[m, I_DAC] < 1).mean()), fail_comfort=float((lab[m, I_C] < 1).mean()),
            d_vs_head=dh, d_vs_top1=boot(pdms(lab)[m] - pdms(top1)[m], logs[m]),
            break_=brk, n_break=int((fails3(lab) & m & pass1).sum()),
            fix=float((~fails3(lab))[m & ~pass1].mean()),
            frac_outside=float(outside[m].mean()), frac_edited=float(edited[m].mean()),
            frac_empty_all=float(empty[m].mean()),
            frac_empty_of_outside_editable=float(empty[m & outside & ~skip].mean()) if (m & outside & ~skip).any()
            else float("nan"),
            n_noconv=int((noconv & m).sum()), frac_noconv_all=float(noconv[m].mean()),
            frac_skip_slow_short=float(skip[m].mean()),
            kinds={kn: int((k[m] == i).sum()) for i, kn in enumerate(kinds)},
            outside_top1_fail=float(fails3(top1)[m & outside].mean()) if (m & outside).any() else float("nan"),
            outside_of_top1_fail=float(outside[m & ~pass1].mean()), outside_of_top1_pass=float(outside[m & pass1].mean()),
            n_lon_edits=int(lon_e.sum()), n_lat_edits=int(lat_e.sum()),
            lon_edit_a_mean=float(r2["a"][lon_e, q].mean()) if lon_e.any() else float("nan"),
            lon_edit_a_median=float(np.median(r2["a"][lon_e, q])) if lon_e.any() else float("nan"),
            lat_edit_absD_mean=float(np.abs(r2["D"][lat_e, q]).mean()) if lat_e.any() else float("nan"),
            lat_edit_absD_median=float(np.median(np.abs(r2["D"][lat_e, q]))) if lat_e.any() else float("nan"),
            d_ep_edited=float(100 * (lab[m & edited, I_EP] - top1[m & edited, I_EP]).mean()) if (m & edited).any()
            else float("nan"),
            inside_after_edited=float(r2["inside_after"][m & edited, q].mean()) if (m & edited).any() else float("nan"),
            edited_pass3=float((~fails3(lab))[m & edited].mean()) if (m & edited).any() else float("nan"),
            d_vs_head_edited_tokens=boot((pdms(lab) - pdms(head))[m & edited], logs[m & edited]) if (m & edited).any()
            else None,
        )
        c["verdict"] = bool(dh["mean"] >= 0.5 and dh["lo"] > 0 and brk <= 0.013 and brk <= head_break)
        c["verdict_parts"] = dict(mean_ge_0p5=bool(dh["mean"] >= 0.5), ci_lo_gt_0=bool(dh["lo"] > 0),
                                  break_le_1p3=bool(brk <= 0.013), break_le_head=bool(brk <= head_break))
        if nm in ("grid_t0.2_l0.3", "exact_t0.2_l0.3"):
            c["scene_bins"] = {}
            for bn, bm in bins.items():
                mm = m & bm
                c["scene_bins"][bn] = dict(
                    n=int(mm.sum()), d_vs_head=boot(pdms(lab)[mm] - pdms(head)[mm], logs[mm]),
                    break_=float(fails3(lab)[mm & pass1].mean()) if (mm & pass1).any() else float("nan"),
                    head_break=float(fails3(head)[mm & pass1].mean()) if (mm & pass1).any() else float("nan"),
                    frac_outside=float(outside[mm].mean()), frac_empty=float(empty[mm].mean()))
            # extra (not the spec comparison): edit where outside, head elsewhere
            mix = np.where(edited[:, None], lab, head)
            c["edit_else_head"] = dict(pdms=float(pdms(mix)[m].mean()),
                                       d_vs_head=boot(pdms(mix)[m] - pdms(head)[m], logs[m]),
                                       break_=float(fails3(mix)[m & pass1].mean()))
        out["configs"][nm] = c
    out["parity_max_abs_diff_vs_F1"] = float(np.nanmax(r2["par_max_diff"]))
    act = {}
    for tube in ("grid", "exact"):
        q0 = names.index(f"{tube}_t0.2_l0.3")
        lon = m & np.isin(r2["kind"][:, q0], [kinds.index("lon"), kinds.index("lon+lat")])
        a = dict(n_lon_edits=int(lon.sum()))
        if tube == "grid":
            # the last round's critical read: delta = its TTC projection (0 = the NC read), t_out = +inf when the
            # cell is blocked through the last bin [4.75, 5.0) s (a read reaches at most 4.0 + 0.9 s: same effect)
            dl, to = r2["delta"][:, q0], r2["t_out"][:, q0]
            inf = lon & ~np.isfinite(to)
            a["n_lon_t_out_inf"] = int(inf.sum())
            a["critical_delta_hist"] = {f"{d:.1f} s": int((lon & np.isclose(dl, d)).sum())
                                        for d in sorted(set(np.round(dl[lon], 3).tolist()), reverse=True)}
            a["critical_delta_hist_t_out_inf"] = {f"{d:.1f} s": int((inf & np.isclose(dl, d)).sum())
                                                  for d in sorted(set(np.round(dl[lon], 3).tolist()), reverse=True)}
            a["n_lon_t_out_finite"] = int((lon & np.isfinite(to)).sum())
            a["n_lon_t_out_inf_has_lead"] = int((inf & has_lead).sum())
            for nm in names:
                if nm.startswith("grid_") and nm.endswith("_l0.3") and nm != "grid_t0.2_l0.3":
                    a[f"n_finite_run_edits_a_differs_{nm}"] = int((lon & np.isfinite(to)
                                                                    & (r2["a"][:, names.index(nm)] != r2["a"][:, q0])).sum())
                    a[f"n_t_out_inf_edits_a_differs_{nm}"] = int((inf & (r2["a"][:, names.index(nm)] != r2["a"][:, q0])).sum())
        for nm in names:
            if nm.startswith(tube + "_") and nm != f"{tube}_t0.2_l0.3" and not nm.endswith("_best"):
                q = names.index(nm)
                a[f"{nm}: tokens with a different PDMS than the main"] = int(
                    (m & (r2["lab"][:, q, I_PDMS] != r2["lab"][:, q0, I_PDMS])).sum())
        act[tube] = a
    out["margin_activity"] = act
    # speed-cap probe
    S = m & r2["cap_over"].astype(bool) & np.isfinite(r2["cap_lab"][:, I_PDMS])
    cl = r2["cap_lab"]
    flips = S & (P[:, 0, I_DAC] == 1) & (top1[:, I_DAC] < 1)
    dacf = S & (top1[:, I_DAC] < 1)
    sp = dict(
        n_over=int(S.sum()), frac_over=float(S.sum() / m.sum()),
        top1_dac_fail=float((top1[S, I_DAC] < 1).mean()), cap_dac_fail=float((cl[S, I_DAC] < 1).mean()),
        n_exec_dac_fail=int(dacf.sum()), fixed_exec_dac_fail=float((cl[dacf, I_DAC] == 1).mean()) if dacf.any()
        else float("nan"),
        n_tracking_flips=int(flips.sum()),
        n_tracking_flips_all_tokens=int((m & (P[:, 0, I_DAC] == 1) & (top1[:, I_DAC] < 1)).sum()),
        fixed_tracking_flips=float((cl[flips, I_DAC] == 1).mean()) if flips.any() else float("nan"),
        break_=float(fails3(cl)[S & pass1].mean()) if (S & pass1).any() else float("nan"),
        d_ep=float(100 * (cl[S, I_EP] - top1[S, I_EP]).mean()),
        d_pdms=boot(pdms(cl)[S] - pdms(top1)[S], logs[S]),
        d_pdms_per_token_all=float((pdms(cl)[S] - pdms(top1)[S]).sum() / m.sum()),
        comfort_fail_top1=float((top1[S, I_C] < 1).mean()), comfort_fail_cap=float((cl[S, I_C] < 1).mean()),
        d_pdms_on_dac_fail=float((pdms(cl) - pdms(top1))[dacf].mean()) if dacf.any() else float("nan"),
        d_pdms_on_pass=float((pdms(cl) - pdms(top1))[S & pass1].mean()) if (S & pass1).any() else float("nan"),
        d_ep_on_pass=float(100 * (cl[S & pass1, I_EP] - top1[S & pass1, I_EP]).mean()) if (S & pass1).any()
        else float("nan"))
    for tag, sub in (("no room (v_max = 0 at some in-plan station)", S & (r2["cap_vmax_min"] <= 0.0)),
                     ("curvature-limited (v_max > 0 at every in-plan station)", S & (r2["cap_vmax_min"] > 0.0))):
        df = sub & (top1[:, I_DAC] < 1)
        sp[tag] = dict(n=int(sub.sum()),
                       top1_dac_fail=float((top1[sub, I_DAC] < 1).mean()) if sub.any() else float("nan"),
                       fixed_exec_dac_fail=float((cl[df, I_DAC] == 1).mean()) if df.any() else float("nan"),
                       break_=float(fails3(cl)[sub & pass1].mean()) if (sub & pass1).any() else float("nan"),
                       d_ep=float(100 * (cl[sub, I_EP] - top1[sub, I_EP]).mean()) if sub.any() else float("nan"),
                       d_pdms=boot(pdms(cl)[sub] - pdms(top1)[sub], logs[sub]) if sub.any() else None,
                       comfort_fail_cap=float((cl[sub, I_C] < 1).mean()) if sub.any() else float("nan"))
    out["speed_cap_probe"] = sp
    return out


# ----------------------------------------------------------------------------------------------- selection oracle
SEL_EDIT_AVERSE = "GT {} tube filter + blend over 96 + edit (edit-averse: the top-1's full blend score)"
SEL_EDIT_PREF = "edit-preferred bound: top-1 if inside {} tube, else the edit if inside after, else filter + blend"


def selection(r1: dict, r2: dict, E: np.ndarray, rows: np.ndarray, logs: np.ndarray, tokm: np.ndarray) -> dict:
    prob16 = np.load(OC.T10 / "prob16.npy", mmap_mode="r")
    prob80 = np.load(OC.T10 / "prob80.npy", mmap_mode="r")
    im = np.load(OC.PACK / "v2_im.npy", mmap_mode="r")
    prob = np.concatenate([np.asarray(prob16[rows], np.float64), np.asarray(prob80[rows], np.float64)], 1)
    im16 = np.asarray(im[rows], np.float64)
    im96 = np.concatenate([im16, np.repeat(im16, 5, 1)], 1)
    s = ck_final(prob, im96)
    V = r1["valid"]
    s = np.where(V, s, -np.inf)
    # argmax in the stored pool96 order (parent k: id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5 at k * 6 + v) so that ties
    # resolve as select.json pool96_old_b1 (parent first): 99.98 % identical picks, identical PDMS on these tokens
    col = np.arange(OC.N96)
    order = np.argsort(np.where(col < OC.N_PAR, col * 6, ((col - OC.N_PAR) // 5) * 6 + (col - OC.N_PAR) % 5 + 1))
    amax = lambda x: order[np.argmax(x[:, order], 1)]
    lab = E[:, :OC.N96]
    ar = np.arange(len(lab))
    pd_ = lambda L: 100.0 * L[..., I_PDMS]
    names = [str(x) for x in r2["cfg_names"]]
    q_main = names.index("grid_t0.2_l0.3")
    q_ex = names.index("exact_t0.2_l0.3")
    m = tokm & r2["ok"].astype(bool)
    top1, head = lab[:, 0], E[:, OC.COL_HEAD]
    pass1 = ~fails3(top1)
    picks = {"v2 top-1": np.zeros(len(lab), np.int64), "blend over 96 (pool96_old_b1)": amax(s)}
    for tag, inside in (("grid", ~(r1["g25_nc"] | r1["g25_ttc"] | r1["g25_dac"]) & V),
                        ("exact", ~(r1["x_nc"] | r1["x_ttc"] | r1["x_dac"]) & V)):
        sf = np.where(inside, s, -np.inf)
        anyin = inside.any(1)
        filt = np.where(anyin, amax(sf), amax(s))
        picks[f"GT {tag} tube filter + blend"] = filt
        picks[f"switch: top-1 if inside {tag} tube else filter + blend"] = np.where(inside[:, 0], 0, filt)
    picks["label oracle over 96"] = np.argmax(np.where(V, lab[..., I_PDMS], -np.inf), 1)
    out = {"n_tokens": int(m.sum()), "picks": {}}
    for nm, ix in picks.items():
        L = lab[ar, ix]
        out["picks"][nm] = dict(pdms=float(pd_(L)[m].mean()), d_vs_top1=boot(pd_(L)[m] - pd_(top1)[m], logs[m]),
                                d_vs_head=boot(pd_(L)[m] - pd_(head)[m], logs[m]),
                                break_=float(fails3(L)[m & pass1].mean()), frac_changed=float((ix != 0)[m].mean()),
                                fail3=float(fails3(L)[m].mean()))
    # select + edit: the main edit joins the pool (its tube flag, the top-1's blend score)
    for tag, q, inside in (("grid", q_main, ~(r1["g25_nc"] | r1["g25_ttc"] | r1["g25_dac"]) & V),
                           ("exact", q_ex, ~(r1["x_nc"] | r1["x_ttc"] | r1["x_dac"]) & V)):
        el = r2["lab"][:, q]
        ed = r2["edited"][:, q].astype(bool)
        e_in = r2["inside_after"][:, q].astype(bool) & ed
        s2 = np.concatenate([s, np.where(ed, s[:, 0], -np.inf)[:, None]], 1)
        in2 = np.concatenate([inside, e_in[:, None]], 1)
        sf = np.where(in2, s2, -np.inf)
        i96 = amax(sf[:, :OC.N96])                    # the edit is the last pool entry: it wins only strictly
        ix = np.where(in2.any(1), np.where(sf[:, OC.N96] > sf[ar, i96], OC.N96, i96), amax(s))
        L = np.where((ix == OC.N96)[:, None], el, lab[ar, np.minimum(ix, OC.N96 - 1)])
        out["picks"][SEL_EDIT_AVERSE.format(tag)] = dict(
            pdms=float(pd_(L)[m].mean()), d_vs_top1=boot(pd_(L)[m] - pd_(top1)[m], logs[m]),
            d_vs_head=boot(pd_(L)[m] - pd_(head)[m], logs[m]), break_=float(fails3(L)[m & pass1].mean()),
            frac_changed=float((ix != 0)[m].mean()), frac_edit_picked=float((ix == OC.N96)[m].mean()),
            fail3=float(fails3(L)[m].mean()))
        # edit-preferred bound (the other extreme): top-1 if inside, else the edit if inside after editing, else
        # filter + blend
        sw = picks[f"switch: top-1 if inside {tag} tube else filter + blend"]
        take_e = ~inside[:, 0] & e_in
        Lp = np.where(inside[:, :1], top1, np.where(take_e[:, None], el, lab[ar, sw]))
        out["picks"][SEL_EDIT_PREF.format(tag)] = dict(
            pdms=float(pd_(Lp)[m].mean()), d_vs_top1=boot(pd_(Lp)[m] - pd_(top1)[m], logs[m]),
            d_vs_head=boot(pd_(Lp)[m] - pd_(head)[m], logs[m]), break_=float(fails3(Lp)[m & pass1].mean()),
            frac_changed=float((~inside[:, 0] & (take_e | (sw != 0)))[m].mean()),
            frac_edit_picked=float(take_e[m].mean()), fail3=float(fails3(Lp)[m].mean()))
        out[f"edit_preferred_minus_switch_{tag}"] = boot(pd_(Lp)[m] - pd_(lab[ar, sw])[m], logs[m])
        Lf = lab[ar, picks[f"GT {tag} tube filter + blend"]]
        out[f"edit_averse_minus_filter_{tag}"] = boot(pd_(L)[m] - pd_(Lf)[m], logs[m])
        # selection gain vs edit gain on the same tokens: edit-only vs the switch selection
        out[f"edit_minus_switch_selection_{tag}"] = boot(pd_(el)[m] - pd_(lab[ar, sw])[m], logs[m])
    return out


# ----------------------------------------------------------------------------------------------- report
def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def ci(b):
    return f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]"


K4 = ("nc", "ttc", "dac", "joint")


def rcell(x: dict) -> str:
    """'fail [lo, hi] / pass [lo, hi]' of one rc() dict (CIs when present)."""
    f = f3(x["fail_recall"]) + (f" [{f3(x['fail_ci'][0])}, {f3(x['fail_ci'][1])}]" if "fail_ci" in x else "")
    p = f4(x["pass_recall"]) + (f" [{f4(x['pass_ci'][0])}, {f4(x['pass_ci'][1])}]" if "pass_ci" in x else "")
    return f"{f} / {p}"


def bars(rr: dict) -> list:
    """the 8 bars of one representation (all 96) -> [(label, point, lo, hi)]."""
    o = []
    for k in K4:
        x = rr["all96"][k]
        for side, ck in (("fail", "fail_ci"), ("pass", "pass_ci")):
            lo, hi = x.get(ck, [float("nan")] * 2)
            o.append((f"{k.upper()} {side}", x[f"{side}_recall"], lo, hi))
    return o


def bar_txt(b) -> str:
    return f"{b[0]} {b[1]:.4f} [{b[2]:.3f}, {b[3]:.3f}]"


def below(rr: dict, which: str = "point") -> str:
    """bars under BAR: 'point' (the estimate), 'lo' (CI lower bound: inside the CI), 'hi' (even the upper bound)."""
    j = {"point": 1, "lo": 2, "hi": 3}[which]
    xs = [bar_txt(b) for b in bars(rr) if b[j] < BAR]
    return ", ".join(xs) if xs else "none"


def summary_md(R: dict) -> str:
    c1, c2, sel = R["criterion1"], R["criterion2"], R["selection_oracle"]
    L = [f"# refiner v4_3 Phase 0 -- GT allowed-range oracle (navtrain_val)", "",
         f"Generated {R['generated']}. CPU only, no training. Source: tools/ck/v4/oracle (SSR-v4, branch refiner-v4). "
         f"Numbers: results.json in this folder. Decision split navtrain_val (6,000 F1 tokens, seed 0); navtest not used.",
         "", "## Verdict", ""]
    reps = c1["reps"]
    v1m = reps[MAIN_REP]["verdict_all96"]
    v1x = reps[EXACT_REP]["verdict_all96"]
    v2m = c2["configs"]["grid_t0.2_l0.3"]["verdict"]
    v2x = c2["configs"]["exact_t0.2_l0.3"]["verdict"]
    learn_pass = [nm for nm, rr in reps.items() if rr["verdict_all96"] and rr["role"] not in ("main", "grid-free")]
    free_pass = [nm for nm, rr in reps.items() if rr["verdict_all96"] and rr["role"] == "grid-free"]

    def keys4(nm):
        a = reps[nm]["all96"]
        return ", ".join(f"{k.upper()} {rcell(a[k])}" for k in K4)

    cm, cx = c2["configs"]["grid_t0.2_l0.3"], c2["configs"]["exact_t0.2_l0.3"]
    ho = R["heldout"]
    L += [f"- Criterion 1, pre-registered main (learnable grid representation, all 96): **{'PASS' if v1m else 'FAIL'}** "
          f"-- {keys4(MAIN_REP)} (fail / pass recall [95 % log-bootstrap CI, {c1['n_logs']} logs]). Below the 0.9 bar: "
          f"{below(reps[MAIN_REP])}; below it even at the CI upper bound: {below(reps[MAIN_REP], 'hi')}.",
          f"- Criterion 1, grid-free exact box check, same rules (upper bound, not a learnable representation): "
          f"**{'PASS' if v1x else 'FAIL'}** -- below the bar: {below(reps[EXACT_REP])}.",
          f"- Criterion 2, main (grid tube, m_t 0.2 s, m_l 0.3 m): **{'PASS' if v2m else 'FAIL'}** -- edit - head "
          f"{ci(cm['d_vs_head'])}, break {100 * cm['break_']:.2f} % vs head {100 * c2['head']['break_']:.2f} %; exact tube: "
          f"**{'PASS' if v2x else 'FAIL'}** ({ci(cx['d_vs_head'])}, break {100 * cx['break_']:.2f} %).",
          "- **Decision** (pre-registered rule 'both pass -> ask about teacher retraining; either fails -> report and drop "
          "idea 3'): " + ("**both pass -> ask the user about teacher retraining.**" if (v1m and v2m) else
                          "**per the pre-registered rule idea 3 is dropped.**"),
          "- Post hoc (chosen after seeing the misses, same 6,000 tokens; does not reopen the verdict):"]
    if learn_pass:
        L += [f"  - Learnable post hoc variants clearing the eight point bars: {len(learn_pass)} -- "
              + "; ".join(f"'{nm}', inside its 95 % CI on {below(reps[nm], 'lo')}"
                          + (" (clears every bar even at the CI lower bound)" if reps[nm]["ci_clear_pass"] else "")
                          for nm in learn_pass) + "."]
    else:
        L += ["  - No learnable post hoc variant clears the eight point bars."]
    L += [f"  - Its stopped-aware version ('{ANY4_SA_REP}'): **{'PASS' if reps[ANY4_SA_REP]['verdict_all96'] else 'FAIL'}**"
          f" (below the bar: {below(reps[ANY4_SA_REP])}); stopped-aware NC with the nearest read ('{SA_REP}'): "
          f"**{'PASS' if reps[SA_REP]['verdict_all96'] else 'FAIL'}** (below the bar: {below(reps[SA_REP])}).",
          f"  - Grid-free checks that pass: {', '.join(repr(x) for x in free_pass) if free_pass else 'none'} -- exact "
          "boxes with no B / D grid, i.e. not a trainable tube target; they locate the misses (grid read, and the "
          "official at-fault rule for stopped / static objects that the not-behind proxy drops), they do not rescue the "
          "criterion.",
          "  - Criterion 2 was not run with an any4 tube (CFGS: grid-nearest and exact only).",
          "- Accepting any4 or a stopped-aware B would be a NEW pre-registration: fix it first (representation, read, "
          "bars, decision rule), then re-run criteria 1 and 2 (with an any4 GridTube config added to CFGS) on the "
          f"navtrain_val tokens outside the F1 6,000 ({ho['n_tokens']:,} available in packed/navtrain_val; token-disjoint "
          + (f"but from the same {ho['n_logs']} logs: every log of the split is in the F1 sample"
             if ho['n_logs_shared'] == ho['n_logs'] else
             f"; {ho['n_logs_shared']} of the {ho['n_logs']} logs are shared with the F1 sample")
          + "; executed states via ck/v4/lqr).",
          "", "## Criterion 1 -- outside the GT tube <=> official fail (executed states, the 96)", "",
          f"Tokens {c1['n_tokens']}, rows (par16 + valid var80) {c1['n_rows_all96']}. "
          "Fail recall = P(outside | official fail), pass recall = P(inside | official pass); bar 0.9 / 0.9 on the point "
          f"estimates (the pre-registered verdict). [95 % CI]: log-clustered bootstrap, {c1['n_logs']} logs, "
          f"{c1['n_boot']:,} resamples, ratio estimator. Role: main = pre-registered; ablation = spec-listed variant; "
          "post hoc = chosen after seeing the misses; grid-free = exact boxes (not a learnable target).", "",
          "| representation | role | NC fail / pass | TTC fail / pass | DAC fail / pass | joint fail / pass | pass? |",
          "|---|---|---|---|---|---|---|"]
    for nm, rr in reps.items():
        a = rr["all96"]
        L.append(f"| {nm} | {rr['role']} | " + " | ".join(rcell(a[k]) for k in K4)
                 + f" | {'yes' if rr['verdict_all96'] else 'no'}{' (CI-clear)' if rr['ci_clear_pass'] else ''} |")
    L += ["", "Excluding UNKNOWN failures (official object absent from the GT store / beyond its horizon; DAC corners "
          f"outside the SDF grid): rows {c1['unknown_rows']}.", "",
          "| representation | NC | TTC | DAC | joint | pass? |", "|---|---|---|---|---|---|"]
    for nm in (MAIN_REP, EXACT_REP):
        a = c1["reps"][nm]["all96_excl_unknown"]
        L.append(f"| {nm} | " + " | ".join(f"{f3(a[k]['fail_recall'])} / {f4(a[k]['pass_recall'])}"
                                         for k in ("nc", "ttc", "dac", "joint"))
                 + f" | {'yes' if c1['reps'][nm]['verdict_all96_excl_unknown'] else 'no'} |")
    L += ["", "Subsets (main grid representation; fail / pass recall):", "",
          "| subset | NC | TTC | DAC | joint |", "|---|---|---|---|---|"]
    for sn in ("top1", "par16", "var80", "all96"):
        a = c1["reps"][MAIN_REP][sn]
        L.append(f"| {sn} | " + " | ".join(f"{f3(a[k]['fail_recall'])} / {f4(a[k]['pass_recall'])} (n_fail {a[k]['n_fail']})"
                                           for k in ("nc", "ttc", "dac", "joint")) + " |")
    L += ["", "Exact proxy by subset:", "", "| subset | NC | TTC | DAC | joint |", "|---|---|---|---|---|"]
    for sn in ("top1", "par16", "var80", "all96"):
        a = c1["reps"][EXACT_REP][sn]
        L.append(f"| {sn} | " + " | ".join(f"{f3(a[k]['fail_recall'])} / {f4(a[k]['pass_recall'])}"
                                           for k in ("nc", "ttc", "dac", "joint")) + " |")
    L += ["", "Mismatch buckets (all 96):", ""]
    for nm, d in c1["buckets_all96"].items():
        L.append(f"- {nm}: " + "; ".join(f"{k} " + ", ".join(f"{kk} {vv}" for kk, vv in v.items()) for k, v in d.items()))
    dd = c1["ddc"]
    L += ["", f"- DDC (not a tube key): {dd['n_rows_ddc_fail']} rows fail DDC ({100 * dd['frac_rows']:.2f} %), "
          f"{dd['ddc_only_fail']} of them pass NC / TTC / DAC, {dd['ddc_only_and_main_inside']} of those are inside the "
          f"main tube; joint incl. DDC (main): fail {f3(dd['joint4_main']['fail_recall'])} / pass "
          f"{f4(dd['joint4_main']['pass_recall'])}.",
          f"- Executed rows with states off the parent grid (main): {c1['n_oog_rows_main']}.", "",
          "Scene bins (all 96, fail / pass recall):", "", "| bin | rep | NC | TTC | DAC | joint |", "|---|---|---|---|---|---|"]
    for nm in (MAIN_REP, EXACT_REP):
        for bn, a in c1["reps"][nm]["scene_bins"].items():
            L.append(f"| {bn} | {nm.split(' (')[0]} | " + " | ".join(
                f"{f3(a[k]['fail_recall'])} / {f4(a[k]['pass_recall'])} ({a[k]['n_fail']})" for k in ("nc", "ttc", "dac", "joint")) + " |")
    L += ["", "## Criterion 2 -- minimal computed edit of the v2 top-1 vs the DET lateral head", "",
          f"Tokens {c2['n_tokens']} (top-1 passes NC/TTC/DAC on {c2['n_top1_pass3']}, fails on {c2['n_top1_fail3']}). "
          f"Official scorer parity on the re-scored top-1 / head trajectories: max |diff| {c2['parity_max_abs_diff_vs_F1']:.1e} "
          f"(F1 labels are float32). PDMS x 100.", "",
          f"- v2 top-1 PDMS {c2['top1']['pdms']:.2f}; DET lateral head {c2['head']['pdms']:.2f} "
          f"(head - top-1 {ci(c2['head_vs_top1'])}); head break {100 * c2['head']['break_']:.2f} %, head fix "
          f"{100 * c2['head']['fix']:.1f} % of failing top-1.", "",
          "| config | PDMS | dPDMS vs head [95% CI] | dPDMS vs top-1 | break % (n) | fix % | outside % | edited % | empty % (of editable outside) | EP | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, c in c2["configs"].items():
        L.append(f"| {nm} | {c['pdms']:.2f} | {ci(c['d_vs_head'])} | {ci(c['d_vs_top1'])} | {100 * c['break_']:.2f} "
                 f"({c['n_break']}) | {100 * c['fix']:.1f} | {100 * c['frac_outside']:.2f} | {100 * c['frac_edited']:.2f} | "
                 f"{100 * c['frac_empty_of_outside_editable']:.1f} | {c['ep']:.2f} | {'PASS' if c['verdict'] else 'fail'} |")
    for nm in ("grid_t0.2_l0.3", "exact_t0.2_l0.3"):
        c = c2["configs"][nm]
        L += ["", f"{nm}: empty tube {100 * c['frac_empty_all']:.2f} % of all tokens "
              f"({100 * c['frac_empty_of_outside_editable']:.1f} % of the outside tokens with v0 >= 2 m/s and S_8 >= 3 m); "
              f"lon loop not converged (MAX_ROUNDS {OC.MAX_ROUNDS}, no edit, not counted as empty) {c['n_noconv']} tokens; "
              f"kinds {c['kinds']}; outside of failing top-1 {100 * c['outside_of_top1_fail']:.1f} %, of passing "
              f"top-1 {100 * c['outside_of_top1_pass']:.2f} %; lon edits (lon, lon+lat; n {c['n_lon_edits']}): a mean "
              f"{c['lon_edit_a_mean']:.2f} / median {c['lon_edit_a_median']:.2f} m/s^2; lat edits (lat, lon+lat; n "
              f"{c['n_lat_edits']}): |D| mean {c['lat_edit_absD_mean']:.2f} / median {c['lat_edit_absD_median']:.2f} m; "
              f"edited tokens: dEP {c['d_ep_edited']:+.2f}, inside the tube after {f3(c['inside_after_edited'])}, "
              f"pass NC/TTC/DAC {f3(c['edited_pass3'])}; dPDMS vs head on edited tokens only "
              f"{ci(c['d_vs_head_edited_tokens']) if c['d_vs_head_edited_tokens'] else 'n/a'}; extra (not the spec "
              f"comparison) edit-where-outside else head: {c['edit_else_head']['pdms']:.2f}, vs head "
              f"{ci(c['edit_else_head']['d_vs_head'])}, break {100 * c['edit_else_head']['break_']:.2f} %.", "",
              "| scene bin | n | dPDMS vs head | break % | head break % | outside % | empty % |", "|---|---|---|---|---|---|---|"]
        for bn, b in c["scene_bins"].items():
            L.append(f"| {bn} | {b['n']} | {ci(b['d_vs_head'])} | {100 * b['break_']:.2f} | {100 * b['head_break']:.2f} | "
                     f"{100 * b['frac_outside']:.2f} | {100 * b['frac_empty']:.2f} |")
    ma = c2["margin_activity"]
    mg = ma["grid"]
    L += ["", f"Margin sensitivity: m_t is mostly inactive because the critical cells are blocked through the last bin "
          f"[4.75, 5.0) s (t_out = +inf in {mg['n_lon_t_out_inf']} of the {mg['n_lon_edits']} grid lon edits; a read "
          "reaches at most 4.0 + 0.9 s, so the edit must keep the read point short of the station for the whole horizon "
          "whatever m_t) -- mostly 0.9 s TTC projections into cells an object still occupies in the last bin (a lead "
          f"per ck/lead on {mg['n_lon_t_out_inf_has_lead']} of the {mg['n_lon_t_out_inf']}); m_t only acts on finite "
          f"runs ({mg['n_lon_t_out_finite']} edits, e.g. crossing traffic): it changes a on "
          f"{mg['n_finite_run_edits_a_differs_grid_t0.1_l0.3']} (m_t 0.1 s) / "
          f"{mg['n_finite_run_edits_a_differs_grid_t0.3_l0.3']} (0.3 s) of them (t_out = +inf edits: "
          f"{mg['n_t_out_inf_edits_a_differs_grid_t0.1_l0.3']} / {mg['n_t_out_inf_edits_a_differs_grid_t0.3_l0.3']}). "
          "Critical read of the last round, by TTC projection delta (0 s = the NC read): "
          f"{mg['critical_delta_hist']}; of them t_out = +inf: {mg['critical_delta_hist_t_out_inf']}. Tokens whose "
          "PDMS differs from the main config: "
          + "; ".join(f"{k.split(':')[0]} {v}" for t in ("grid", "exact") for k, v in ma[t].items() if ":" in k) + "."]
    sp = c2["speed_cap_probe"]
    L += ["", "## Curvature speed-limit probe (top-1 planned faster than v_max at an in-plan station)", "",
          f"- {sp['n_over']} tokens ({100 * sp['frac_over']:.1f} %). Executed DAC fail {100 * sp['top1_dac_fail']:.1f} % -> "
          f"{100 * sp['cap_dac_fail']:.1f} % capped; fixes {100 * sp['fixed_exec_dac_fail']:.1f} % of the {sp['n_exec_dac_fail']} "
          f"executed DAC failures and {100 * sp['fixed_tracking_flips']:.1f} % of the {sp['n_tracking_flips']} tracking flips "
          f"(plan pass -> exec fail; {sp['n_tracking_flips_all_tokens']} flips on all tokens).",
          f"- Cost: break {100 * sp['break_']:.2f} % of passing top-1, dEP {sp['d_ep']:+.2f} (on passing {sp['d_ep_on_pass']:+.2f}), "
          f"dPDMS {ci(sp['d_pdms'])} on the probe tokens ({sp['d_pdms_per_token_all']:+.3f} per token over all), "
          f"on DAC-failing {sp['d_pdms_on_dac_fail']:+.2f}, on passing {sp['d_pdms_on_pass']:+.2f}; comfort fail "
          f"{100 * sp['comfort_fail_top1']:.1f} % -> {100 * sp['comfort_fail_cap']:.1f} %."]
    for tag, d in sp.items():
        if isinstance(d, dict) and "fixed_exec_dac_fail" in d:
            L.append(f"- {tag}: n {d['n']}, top-1 DAC fail {100 * d['top1_dac_fail']:.1f} %, fixed "
                     f"{100 * d['fixed_exec_dac_fail']:.1f} %, break {100 * d['break_']:.2f} %, dEP {d['d_ep']:+.2f}, dPDMS "
                     f"{ci(d['d_pdms']) if d['d_pdms'] else 'n/a'}, comfort fail {100 * d['comfort_fail_cap']:.1f} %.")
    L += ["", "## Selection-only oracle (GT tube flags + deployment blend score)", "",
          "| pick | PDMS | vs top-1 | vs head | break % | changed % | edit picked % |", "|---|---|---|---|---|---|---|"]
    for nm, p in sel["picks"].items():
        L.append(f"| {nm} | {p['pdms']:.2f} | {ci(p['d_vs_top1'])} | {ci(p['d_vs_head'])} | {100 * p['break_']:.2f} | "
                 f"{100 * p['frac_changed']:.1f} | {100 * p.get('frac_edit_picked', 0.0):.2f} |")
    L += ["", f"- Edit gain over selection (same tokens): main grid edit - switch selection "
          f"{ci(sel['edit_minus_switch_selection_grid'])}; exact {ci(sel['edit_minus_switch_selection_exact'])}.",
          "- The edit as an extra candidate, both extremes (no critic score of the edit exists here): edit-averse "
          "('+ edit' rows: the edit carries the outside top-1's full blend score incl. its own critic probability, so "
          "its pick rate is a construction artifact) vs filter + blend: grid "
          f"{ci(sel['edit_averse_minus_filter_grid'])}, exact {ci(sel['edit_averse_minus_filter_exact'])}; "
          "edit-preferred bound (the edit whenever the top-1 is outside and the edit is inside after editing) vs the "
          f"switch selection: grid {ci(sel['edit_preferred_minus_switch_grid'])}, exact "
          f"{ci(sel['edit_preferred_minus_switch_exact'])}. Under either extreme the edit adds at most "
          + ", ".join(f"{max(sel[f'edit_averse_minus_filter_{t}']['mean'], sel[f'edit_preferred_minus_switch_{t}']['mean']):+.2f} "
                      f"({t})" for t in ("grid", "exact"))
          + " PDMS on top of selection among the 96 with GT flags, against the edit alone - switch above: the gain "
          "is in the selection, not in the edit.", "",
          "## EPDMS", "", "- " + R["epdms"], "", "## Notes", ""] + [f"- {x}" for x in R["notes"]]
    return "\n".join(L) + "\n"


def main():
    E = np.load(OC.PREP / "f1_exec.npy").astype(np.float64)
    P = np.load(OC.PREP / "f1_plan.npy").astype(np.float64)
    rows = np.load(OC.PREP / "f1_rows.npy")
    fin = np.load(OC.PREP / "f1_fin.npy")
    tok = pd.read_parquet(OC.PACK / "tokens.parquet")
    logs = tok["log"].astype(str).to_numpy()[rows]
    r1 = dict(np.load(OC.OUT / "crit1/rows.npz"))
    r2 = dict(np.load(OC.OUT / "crit2/rows.npz"))
    assert np.array_equal(r1["i"], np.arange(len(rows))) and np.array_equal(r2["i"], np.arange(len(rows)))
    tokm = (r1["gt_ok"] > 0) & (r1["sdf_ok"] > 0) & fin[:, 0] & fin[:, OC.COL_HEAD]
    lead = pd.read_parquet(OC.LEAD)
    lg = lead["lead_gap"].to_numpy()[rows]
    bins = {"lead 10-25 m": (lead["has_lead"].to_numpy()[rows] == 1) & (lg >= 10) & (lg <= 25),
            "near curb (top-1 min corner SDF < 1 m)": r1["top1_min_corner"] < 1.0,
            "lane-only (no object within 30 m)": r1["top1_obj_dmin"] >= 30.0}
    R = {"generated": kst(), "split": OC.SPLIT, "n_tokens_f1": int(len(rows)), "n_tokens_used": int(tokm.sum()),
         "scene_bin_counts": {k: int((v & tokm).sum()) for k, v in bins.items()}}
    R["criterion1"] = crit1(r1, E[:, :OC.N96], tokm, bins, logs)
    R["criterion2"] = crit2(r2, E, P, logs, tokm, bins, lead["has_lead"].to_numpy()[rows] == 1)
    R["selection_oracle"] = selection(r1, r2, E, rows, logs, tokm)
    R["epdms"] = ("not computed (spec item 'EPDMS next to PDMS' still open): the EPDMS scorer of "
                  "teacher_vs_student/epdms (perception_eval/epdms/run_epdms.sh, official NAVSIM v2.2 one-stage) is "
                  "wired to the navtest v2.2 metric cache only (metric_cache_v22_navtest); no navtrain_val v2.2 metric "
                  "cache exists, and building one is a new data root. EC also needs the adjacent frame's trajectory, "
                  "which a per-token oracle edit does not have. Question for the user: approve building a navtrain_val "
                  "v2.2 metric cache, or accept PDMS-only for Phase 0 (not built without approval).")
    ho = np.setdiff1d(np.arange(len(tok)), rows)
    R["heldout"] = dict(n_tokens=int(len(ho)), n_logs=int(tok["log"].nunique()),
                        n_logs_shared=int(len(set(tok["log"].iloc[ho]) & set(tok["log"].iloc[rows]))))
    run1 = json.loads((OC.OUT / "crit1/run.json").read_text())
    run2 = json.loads((OC.OUT / "crit2/run.json").read_text())
    R["notes"] = [
        "Official labels = F1 exec (official PDMSimulator + PDMScorer, bit-exact vs the label shards); edits scored by "
        "tools/refiner/score_trajectories.score_token in one call per token (plain navsim classes).",
        "Criterion 1 main reproduces the stage-1 range_targets validation exactly on top-1 / all 16 parents.",
        "Edits: lon = 0.8 s ramp deceleration of variants.make_variants along the same path; lat = early smoothstep "
        "(onset 0.2 S_8, flat end, decoder curvature projection); both via the same decoder as the rule variants.",
        f"Grid tube edit rule = spec M5 (critical station, t_out_block + m_t, iterated up to MAX_ROUNDS = "
        f"{OC.MAX_ROUNDS}; an exhausted loop is 'noconv_lon', reported apart from the empty tube); the exact tube has "
        "no stations, so its lon edit is the smallest |a| on the 0.05 m/s^2 grid passing the exact proxy with object "
        "samples dilated by m_t (oracle_common docstring).",
        "The selection-only oracle uses the DET teacher critic (ck2T10) probabilities -- the only critic with val "
        "outputs on these 96 -- in the deployment blend (SEL_W default with im, beta 1); its blend pick reproduces the "
        "stored pool96_old_b1 picks (99.98 % identical index, identical PDMS).",
        "'Edit iff outside the GT tube' is the spec's oracle definition of criterion 2, not a deployment gate: in "
        "deployment the edit is an extra candidate. Its critic score is not available here (spec M5: scored by the "
        "critic, inheriting only v2_final / im), so the selection oracle brackets it: edit-averse ('+ edit' rows, the "
        "edit carries the top-1's full blend score; picked on "
        f"{100 * R['selection_oracle']['picks'][SEL_EDIT_AVERSE.format('grid')]['frac_edit_picked']:.2f} % of the "
        "tokens, a construction artifact) and edit-preferred (picked on "
        f"{100 * R['selection_oracle']['picks'][SEL_EDIT_PREF.format('grid')]['frac_edit_picked']:.2f} %).",
        f"Bootstrap: {R['criterion2']['configs']['grid_t0.2_l0.3']['d_vs_head']['n_logs']} log clusters, 2,000 resamples, "
        "seed 0. Every edit, tube and label here uses GT (privileged): an upper bound for a learned predictor.",
        "UNKNOWN = 0 rows: every official NC / TTC failure object (label-shard track token) is in the GT store and "
        "within its horizon, and no official DAC failure is invisible to the stored SDF for lack of grid coverage.",
        f"Runtime (28 CPU workers, nice 10): criterion 1 {run1['sec']:.0f} s, criterion 2 {run2['sec']:.0f} s "
        f"({run2['n_scored_tokens']:,} tokens re-scored); OUT/prep holds a 0.66 GB .npy copy of the F1 sim array (only "
        "for mmap; deletable).",
    ]
    (OC.OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (OC.OUT / "summary.md").write_text(summary_md(R))
    print(summary_md(R))


if __name__ == "__main__":
    main()
