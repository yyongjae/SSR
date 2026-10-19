"""refiner v5 corrector-A statistics on official labels (pure numpy; shared by tools/ck/v5/eval_refine_gate.py, the
Stage-1 (3) gate, and tools/ck/e2e2/eval_ck2.py's 'v5' block of an e2e student).

Labels are [n, 9] rows in LABEL_COLS order (navsim/agents/para_ssr/ck/constants.py: nc, dac, ep, ttc, comfort, ddc, pdms,
raw_progress, pdm_progress_eff), one row per token for a REFERENCE trajectory (the draft / the no-edit trajectory) and a
NEW one (tau1), paired by token.

  boot100(new, ref, logs)        paired log-cluster bootstrap (tools/ck/eval_ck.log_bootstrap) of mean(100 new - 100 ref)
  break_fix(ref, new, rule)      break = P(new fails | ref passes), fix = P(new passes | ref fails) with the counts;
                                 rule 'pdms0': pass = PDMS > 0 (a multiplicative NC / DAC / DDC zero), 'fails3': pass =
                                 NC, TTC and DAC all 1 (tools/ck/v4/eval_range_teacher.fails3, the v4_3 break rule)
  key_counts(ref, new)           per key (nc, dac, ttc, ddc, comfort; fail = value < 1) and PDMS == 0: fail counts of both,
                                 breaks, fixes; EP means and the EP loss mean(ref - new)
  mu_stats(mu)                   corrector controls mu [n, 12] (z_lon 6 + w_lat 6): |mu| = L_inf per token (mean, median,
                                 max, share > MU_THR), per-control mean |mu|
  moved(a, b)                    [n] True where two trajectory sets [n, 8, 3] differ by more than MOVE_M in x / y
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

KEYS5 = ("nc", "dac", "ttc", "ddc", "comfort")
BREAK_RULES = ("pdms0", "fails3")
MU_THR = 0.05            # |mu| > 0.05 (report 97 section 5-3 gate 2: '|u| > 0.05 on >= 3 % of tokens')
MOVE_M = 1e-3            # a trajectory 'moves' when some pose differs by more than 1 mm


def _lbl() -> Dict[str, int]:
    from navsim.agents.para_ssr.ck.constants import LBL
    return LBL


def boot100(new: np.ndarray, ref: np.ndarray, logs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict:
    """paired log-cluster bootstrap (eval_ck.log_bootstrap: logs resampled, ratio over tokens) of mean(100 * new -
    100 * ref) -> {mean, lo, hi, n, n_logs} in x 100 units (non-finite pairs dropped by log_bootstrap)."""
    from tools.ck.eval_ck import log_bootstrap
    d = 100.0 * np.asarray(new, np.float64) - 100.0 * np.asarray(ref, np.float64)
    return log_bootstrap(d, np.asarray(logs).astype(str), n_boot, seed)


def passes(lab: np.ndarray, rule: str) -> np.ndarray:
    """[n, 9] -> [n] bool pass under the break rule."""
    L = _lbl()
    lab = np.asarray(lab)
    if rule == "pdms0":
        return lab[:, L["pdms"]] > 0
    if rule == "fails3":
        from tools.ck.v4.eval_range_teacher import fails3
        return ~fails3(lab)
    raise ValueError(f"break rule {rule!r} not in {BREAK_RULES}")


def break_fix(lab_ref: np.ndarray, lab_new: np.ndarray, rule: str) -> Dict[str, Any]:
    """-> {rule, n_break, den (reference passes), rate, n_fix, fix_den (reference fails), fix_rate, n_new_fail}."""
    p_ref, p_new = passes(lab_ref, rule), passes(lab_new, rule)
    nb, den = int((p_ref & ~p_new).sum()), int(p_ref.sum())
    nf, fden = int((~p_ref & p_new).sum()), int((~p_ref).sum())
    return {"rule": rule, "n_break": nb, "den": den, "rate": nb / den if den else float("nan"), "n_fix": nf,
            "fix_den": fden, "fix_rate": nf / fden if fden else float("nan"), "n_new_fail": int((~p_new).sum())}


def key_counts(lab_ref: np.ndarray, lab_new: np.ndarray) -> Dict[str, Any]:
    """per key of KEYS5 and 'pdms0' (PDMS == 0): {fail_ref, fail_new, break (ref pass -> new fail), fix}; 'ep':
    {mean_ref, mean_new, loss = mean(ref - new)} (0-1 units)."""
    L = _lbl()
    lab_ref, lab_new = np.asarray(lab_ref, np.float64), np.asarray(lab_new, np.float64)
    out: Dict[str, Any] = {}
    fails = {k: (lab_ref[:, L[k]] < 1, lab_new[:, L[k]] < 1) for k in KEYS5}
    fails["pdms0"] = (lab_ref[:, L["pdms"]] == 0, lab_new[:, L["pdms"]] == 0)
    for k, (fr, fn) in fails.items():
        out[k] = {"fail_ref": int(fr.sum()), "fail_new": int(fn.sum()), "break": int((~fr & fn).sum()),
                  "fix": int((fr & ~fn).sum())}
    e_ref, e_new = lab_ref[:, L["ep"]], lab_new[:, L["ep"]]
    n = len(e_ref)
    out["ep"] = {"mean_ref": float(e_ref.mean()) if n else float("nan"),
                 "mean_new": float(e_new.mean()) if n else float("nan"),
                 "loss": float((e_ref - e_new).mean()) if n else float("nan")}
    return out


def mu_stats(mu: np.ndarray, thr: float = MU_THR) -> Dict[str, Any]:
    """mu [n, C] -> {n, linf_mean, linf_median, linf_max, thr, frac_gt_thr, per_control_mean_abs [C]}."""
    m = np.abs(np.asarray(mu, np.float64))
    n = int(m.shape[0]) if m.ndim == 2 else 0
    inf = m.max(1) if n else np.zeros(0)
    nan = float("nan")
    return {"n": n, "linf_mean": float(inf.mean()) if n else nan, "linf_median": float(np.median(inf)) if n else nan,
            "linf_max": float(inf.max()) if n else nan, "thr": float(thr),
            "frac_gt_thr": float((inf > thr).mean()) if n else nan,
            "per_control_mean_abs": [float(x) for x in m.mean(0)] if n else []}


def moved(a: np.ndarray, b: np.ndarray, tol: float = MOVE_M) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (np.abs(a[..., :2] - b[..., :2]) > tol).reshape(len(a), -1).any(1)
