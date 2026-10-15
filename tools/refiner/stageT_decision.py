#!/usr/bin/env python
"""Stage-T decision analysis: cross-fitted theta selection (train OOF) and the pre-stated R_T vs R_none comparison (dev).

Implements report/refiner_T/PRESTATED_DECISION_RULE.txt on the per-draft rows that eval_refiner.py writes
(<run>/<eval dir>/report_rows.parquet: token, k, family, valid, p_g, <metric>_orig, <metric>_tau1; <eval dir>/tokens.parquet
gives the log of each token).  No model is run here; CPU only, seconds.

  # 1. theta (and, if several --wtags, loss weights) per arm from the out-of-fold rows of the cross-fitting runs
  python tools/refiner/stageT_decision.py select --arms T,none --seeds 0,1,2 --folds 0,1,2,3,4 --wtags stageT \
      --budget-ep 0.5 --out report/refiner_T/crossfit_selection.json
  # 2. dev comparison of the final runs (trained on all folds) at the selected theta
  python tools/refiner/stageT_decision.py compare --selection report/refiner_T/crossfit_selection.json \
      --seeds 0,1,2 --n-boot 10000 --out report/refiner_T/decision_dev.json

Run naming (train_refiner.py): <runs>/<wtag>_<arm>_fold<k>_seed<s>/; OOF rows: eval_train_fold<k>/, dev: eval_dev/.

select  per (wtag, arm): for every theta of the grid, the OOF rows of all folds are pooled per seed and the metrics are
        averaged over seeds: ep_loss_points = 100 mean(ep_orig - ep_final), d_pdms_points = 100 mean(pdms_final - pdms_orig)
        (valid drafts; final = tau1 if p_g >= theta else the original).  theta_arm = the SMALLEST theta with
        ep_loss_points <= budget (the most correcting gate within the progress budget; = eval_refiner theta_at_budget).
        With several wtags (loss-weight candidates), the wtag with the largest OOF d_pdms_points at its theta is chosen,
        per arm.  Train pool only; dev is not read.
compare per arm: dev rows of the final runs (<wtag>_<arm>_fold-1_seed<s>) at theta_arm; per draft, the per-seed outcomes
        are averaged over seeds; drafts are paired by (token, k) across arms.  Log-cluster paired bootstrap (resample dev
        logs with replacement, --n-boot, seed 0), 95 % percentile CIs.  Endpoints (R_T - R_none unless noted):
          P1   official PDMS difference [points, x100]
          P2   NC+TTC failure REDUCTION = fail_none - fail_T [pp] (positive = R_T better)   (NC < 1 or TTC < 1)
          P2ni DAC, DDC failure excess fail_T - fail_none [pp]; non-inferior iff upper CI <= margin (0.5 pp)
          P3   new-failure excess new_T - new_none [pp] (new = original passes NC, DAC, DDC, TTC, comfort and the final
               fails any of them); non-inferior iff upper CI <= margin (0.5 pp)
          sanity: some arm improves PDMS over "no correction" (final - original) with lower CI > 0
        Outcome (in this order): INVALID (sanity fails) / PASS (P1 lo > 0 and P2 lo > 0 and P3 non-inferior) /
        WORSE (P1 hi < 0) / EQUIVALENT (P1 CI inside +-0.2 points and P2 CI inside +-0.5 pp) / INCONCLUSIVE.
        DAC/DDC non-inferiority and the dev EP loss vs the budget are reported next to the outcome (the rule's PASS line
        does not list them; flagged, not silently folded in).  Family-wise results are reported, not decisive.
--budget-def (PRESTATED_DECISION_RULE AMENDMENT 3 (2), "B+"): which drafts the EP loss of the budget is measured on.
        all     (default; run 1): every valid draft, ep_loss_points = 100 mean(ep_orig - ep_final).
        passing : only drafts that pass NC, DAC and DDC (each >= 1) both before (orig) and after (final at theta) the
                  correction (pure slowdown cost; the official EP is 0 for a draft failing a multiplicative metric, which
                  made the 'all' budget non-binding in run 1).  Per seed, the mean over that seed's passing drafts, then
                  averaged over seeds; n_passing = seed-averaged count.
        pick_theta and ep_loss_points use the chosen definition; ep_loss_points_all / ep_loss_points_passing /
        n_passing are reported for both.  select records budget_def; compare uses the selection's budget_def (an
        explicit --budget-def that differs from it is refused) for the dev ep_loss_points / within_budget report.
Defaults marked [default] in the rule (budget 0.5 points, margins 0.5 pp / 0.2 points, 3 seeds, 10,000 resamples) are
arguments here.
RUN 4 (PRESTATED_DECISION_RULE AMENDMENT 6):
  select  --arms T,M,TM,none: per-arm theta exactly as above (nothing arm-specific; --budget-def passing).
  compare --pairs "T:none,M:none,TM:T,TM:M" (A:B = A minus B): for every pair the endpoints, sanity and outcome of the
          two-arm rule above with arm A in the role of R_T and arm B in the role of R_none; --ci-level L (default 0.95;
          run 4: 0.9875 = Bonferroni for 4 comparisons) sets EVERY CI (P1, P2, P2ni DAC / DDC, P3, sanity) to the
          two-sided L percentile interval [(1-L)/2, (1+L)/2].  Sanity is evaluated per pair (either arm of the pair
          improves PDMS over no correction with lower CI > 0); 'sanity_any_arm' (any of all arms) is reported too.
          Drafts are paired by (token, k) per pair (inner join, as the two-arm path; unpaired counts reported).
          Output: {pairs: {"A:B": result}, outcomes: {"A:B": outcome}, arms: per-arm info, ci_level, ...}.
          A pair element may name another eval directory of that arm's run: ARM@EVAL (e.g. "TM@eval_dev_drop_det:TM",
          "T@eval_dev_shuffle:T") -- the descriptive teacher-shuffle / branch-drop contrasts, same theta (the arm's
          selection), same drafts, same CI level; without @ the element uses --eval-name.
  Without --pairs the old two-arm CLI (T vs none) runs the old code path; its output is unchanged (a 'ci_level' key
  is added only when --ci-level is given explicitly).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = Path("/home/external-user/ssd/yongjae_refiner/runs")
METRICS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")
THETAS = tuple(np.round(np.linspace(0.0, 1.0, 101), 4))
from navsim.agents.para_ssr.refiner.decoder import FAMILY_NAME  # noqa: E402  (family code -> name)


# ----------------------------------------------------------------------------------------------- rows
def run_dir(runs: Path, wtag: str, arm: str, fold: int, seed: int) -> Path:
    return Path(runs) / f"{wtag}_{arm}_fold{fold}_seed{seed}"


def load_rows(eval_dir: Path) -> pd.DataFrame:
    """report_rows.parquet + log (tokens.parquet) of one evaluation directory."""
    df = pd.read_parquet(Path(eval_dir) / "report_rows.parquet")
    tk = pd.read_parquet(Path(eval_dir) / "tokens.parquet")[["token", "log"]].drop_duplicates("token")
    df = df.merge(tk, on="token", how="left", validate="many_to_one")
    if df.log.isna().any():
        raise ValueError(f"{eval_dir}: tokens without a log")
    return df


def final_outcomes(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    """Per draft (valid rows with finite scores): final metrics at theta and the derived failure indicators."""
    mod = df.p_g.to_numpy() >= theta
    out = pd.DataFrame({"token": df.token.values, "k": df.k.values, "log": df.log.values, "family": df.family.values})
    ok = np.array(df.valid.to_numpy(bool), copy=True)          # never a view: ok &= ... must not touch df
    for m in METRICS:
        o = df[f"{m}_orig"].to_numpy(np.float64)
        f = np.where(mod, df[f"{m}_tau1"].to_numpy(np.float64), o)
        out[f"{m}_orig"], out[f"{m}_final"] = o, f
        ok &= np.isfinite(o) & np.isfinite(f)
    out["modified"] = mod
    fail = lambda s, ks: np.any(np.stack([out[f"{k}_{s}"].to_numpy() < 1 for k in ks], -1), -1)
    out["fail_ncttc"] = fail("final", ("nc", "ttc")).astype(float)
    out["fail_ncttc_orig"] = fail("orig", ("nc", "ttc")).astype(float)
    out["fail_dac"] = (out.dac_final < 1).astype(float)
    out["fail_ddc"] = (out.ddc_final < 1).astype(float)
    allk = ("nc", "dac", "ddc", "ttc", "comfort")
    out["new_fail"] = (~fail("orig", allk) & fail("final", allk)).astype(float)
    out["ep_loss"] = out.ep_orig - out.ep_final
    core = lambda s: np.all(np.stack([out[f"{k}_{s}"].to_numpy() >= 1 for k in ("nc", "dac", "ddc")], -1), -1)
    out["pass_both"] = core("orig") & core("final")                 # budget_def 'passing' (NC, DAC, DDC before + after)
    out["d_pdms"] = out.pdms_final - out.pdms_orig
    return out[ok].reset_index(drop=True)


BUDGET_DEFS = ("all", "passing")


def ep_loss_points(f: pd.DataFrame, budget_def: str = "all") -> float:
    """EP loss [points] of one seed's final_outcomes frame under budget_def (module docstring); nan if no draft counts."""
    if budget_def == "all":
        return 100 * f.ep_loss.mean()
    if budget_def == "passing":
        p = f.pass_both.to_numpy(bool)
        return 100 * float(f.ep_loss.to_numpy()[p].mean()) if p.any() else float("nan")
    raise ValueError(f"budget_def {budget_def!r}")


def sweep(rows_by_seed: Sequence[pd.DataFrame], thetas: Sequence[float] = THETAS, budget_def: str = "all") -> pd.DataFrame:
    """theta -> seed-averaged pooled metrics (points / pp); ep_loss_points under budget_def."""
    rec = []
    for th in thetas:
        per = []
        for df in rows_by_seed:
            f = final_outcomes(df, th)
            per.append(dict(ep_loss_points=ep_loss_points(f, budget_def), d_pdms_points=100 * f.d_pdms.mean(),
                            pdms=f.pdms_final.mean(), modified_frac=f.modified.mean(), n=len(f),
                            fail_ncttc_pp=100 * f.fail_ncttc.mean(), new_fail_pp=100 * f.new_fail.mean(),
                            ep_loss_points_all=ep_loss_points(f, "all"),
                            ep_loss_points_passing=ep_loss_points(f, "passing"), n_passing=int(f.pass_both.sum())))
        r = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        r["theta"] = float(th)
        rec.append(r)
    return pd.DataFrame(rec)


def pick_theta(sw: pd.DataFrame, budget_ep: float) -> Optional[float]:
    ok = sw[sw.ep_loss_points <= budget_ep + 1e-12]
    return float(ok.theta.min()) if len(ok) else None


# ----------------------------------------------------------------------------------------------- select
def select(runs: Path, wtags: Sequence[str], arms: Sequence[str], seeds: Sequence[int], folds: Sequence[int],
           budget_ep: float, thetas: Sequence[float] = THETAS, budget_def: str = "all") -> Dict:
    if budget_def not in BUDGET_DEFS:
        raise ValueError(f"budget_def {budget_def!r}")
    res = dict(budget_ep_points=budget_ep, budget_def=budget_def, seeds=list(seeds), folds=list(folds),
               wtags=list(wtags), arms={})
    for arm in arms:
        cand = {}
        for wt in wtags:
            by_seed, missing = [], []
            for s in seeds:
                parts = []
                for k in folds:
                    d = run_dir(runs, wt, arm, k, s) / f"eval_train_fold{k}"
                    if (d / "report_rows.parquet").is_file():
                        parts.append(load_rows(d))
                    else:
                        missing.append(str(d))
                if parts:
                    by_seed.append(pd.concat(parts, ignore_index=True))
            if missing or not by_seed:
                cand[wt] = dict(missing=missing, complete=False)
                continue
            sw = sweep(by_seed, thetas, budget_def)
            th = pick_theta(sw, budget_ep)
            at = sw[sw.theta == th].iloc[0].to_dict() if th is not None else None
            cand[wt] = dict(complete=True, theta=th, at_theta=at, n_oof_drafts=int(np.mean([len(b) for b in by_seed])),
                            sweep=sw.round(6).to_dict(orient="list"))
        done = {w: c for w, c in cand.items() if c.get("complete") and c.get("theta") is not None}
        best = max(done, key=lambda w: done[w]["at_theta"]["d_pdms_points"]) if done else None
        res["arms"][arm] = dict(candidates=cand, wtag=best, theta=done[best]["theta"] if best else None,
                                complete=all(c.get("complete") for c in cand.values()))
    return res


# ----------------------------------------------------------------------------------------------- compare
def ci_percentiles(level: float = 0.95):
    """Two-sided percentile bounds [q, 100 - q], q = 50 (1 - level), rounded to 12 decimals so that 0.95 gives exactly
    [2.5, 97.5] (the pre-run-4 literal) and 0.9875 gives [0.625, 99.375]."""
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"ci level {level} not in (0, 1)")
    q = round(50.0 * (1.0 - float(level)), 12)
    return [q, round(100.0 - q, 12)]


def cluster_boot(diff: np.ndarray, logs: np.ndarray, n_boot: int, seed: int = 0, level: float = 0.95) -> Dict:
    """Log-cluster bootstrap of mean(diff) over drafts: resample logs with replacement.  -> mean, lo, hi (two-sided
    `level` percentile interval; 95 % by default)."""
    u, inv = np.unique(logs, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    rng = np.random.default_rng(seed)
    L = len(u)
    stats = np.empty(n_boot)
    for i0 in range(0, n_boot, 1000):
        m = min(1000, n_boot - i0)
        c = rng.multinomial(L, np.full(L, 1.0 / L), size=m).astype(np.float64)
        stats[i0:i0 + m] = (c @ s) / np.maximum(c @ n, 1.0)
    lo, hi = np.percentile(stats, ci_percentiles(level))
    return dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi), n_logs=int(L), n=int(len(diff)))


def seed_average(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    cols = ["pdms_final", "pdms_orig", "fail_ncttc", "fail_dac", "fail_ddc", "new_fail", "ep_loss", "d_pdms", "modified"]
    base = frames[0][["token", "k", "log", "family"]]
    keys = pd.MultiIndex.from_frame(base[["token", "k"]])
    acc = None
    for f in frames:
        g = f.set_index(["token", "k"]).reindex(keys)[cols].astype(float)
        acc = g if acc is None else acc + g
    out = base.copy()
    for c in cols:
        out[c] = (acc[c] / len(frames)).to_numpy()
    return out.dropna().reset_index(drop=True)


def compare(runs: Path, selection: Dict, seeds: Sequence[int], n_boot: int = 10000, margin_pp: float = 0.5,
            eq_points: float = 0.2, eq_pp: float = 0.5, final_fold: int = -1, budget_def: Optional[str] = None,
            eval_name: str = "eval_dev", ci_level: Optional[float] = None) -> Dict:
    """budget_def None -> the selection's (a selection without one = 'all', run 1); an explicit different one is
    refused (the dev budget report must use the definition theta was selected with).
    eval_name: the per-run eval subdirectory whose rows are compared (eval_dev; eval_navtest = AMENDMENT 5 final
    navtest evaluation with the same frozen models and theta)."""
    sel_def = selection.get("budget_def", "all")
    if budget_def is None:
        budget_def = sel_def
    if budget_def != sel_def:
        raise ValueError(f"budget_def {budget_def!r} != the selection's {sel_def!r}")
    if budget_def not in BUDGET_DEFS:
        raise ValueError(f"budget_def {budget_def!r}")
    arms = ("T", "none")
    per = {}
    info = {}
    for arm in arms:
        sel = selection["arms"][arm]
        th, wt = sel["theta"], sel["wtag"]
        if th is None:
            raise ValueError(f"no theta selected for arm {arm}")
        frames = [final_outcomes(load_rows(run_dir(runs, wt, arm, final_fold, s) / eval_name), th) for s in seeds]
        per[arm] = seed_average(frames)
        ep_all = 100 * float(per[arm].ep_loss.mean())
        ep_pass = float(np.mean([ep_loss_points(f, "passing") for f in frames]))
        ep_used = ep_all if budget_def == "all" else ep_pass
        info[arm] = dict(theta=th, wtag=wt, n_drafts=len(per[arm]), budget_def=budget_def,
                         ep_loss_points=ep_used, ep_loss_points_all=ep_all, ep_loss_points_passing=ep_pass,
                         n_passing=float(np.mean([f.pass_both.sum() for f in frames])),
                         d_pdms_points=100 * float(per[arm].d_pdms.mean()),
                         pdms=float(per[arm].pdms_final.mean()), pdms_orig=float(per[arm].pdms_orig.mean()),
                         modified_frac=float(per[arm].modified.mean()),
                         within_budget=bool(ep_used <= selection["budget_ep_points"] + 1e-12))
    m = per["T"].merge(per["none"], on=["token", "k", "log", "family"], suffixes=("_T", "_none"), validate="one_to_one")
    logs = m.log.to_numpy()
    lvl = 0.95 if ci_level is None else float(ci_level)
    B = lambda d: cluster_boot(np.asarray(d, np.float64), logs, n_boot, level=lvl)
    ep = {}
    ep["P1_d_pdms_points"] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m.pdms_final_T - m.pdms_final_none).items()}
    ep["P2_ncttc_reduction_pp"] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m.fail_ncttc_none - m.fail_ncttc_T).items()}
    ep["P2ni_dac_excess_pp"] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m.fail_dac_T - m.fail_dac_none).items()}
    ep["P2ni_ddc_excess_pp"] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m.fail_ddc_T - m.fail_ddc_none).items()}
    ep["P3_new_fail_excess_pp"] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m.new_fail_T - m.new_fail_none).items()}
    san = {arm: {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in B(m[f"d_pdms_{arm}"]).items()} for arm in arms}
    sanity = any(s["lo"] > 0 for s in san.values())
    p1, p2, p3 = ep["P1_d_pdms_points"], ep["P2_ncttc_reduction_pp"], ep["P3_new_fail_excess_pp"]
    p3_ni = p3["hi"] <= margin_pp
    dacddc_ni = ep["P2ni_dac_excess_pp"]["hi"] <= margin_pp and ep["P2ni_ddc_excess_pp"]["hi"] <= margin_pp
    if not sanity:
        outcome = "INVALID (sanity: no arm improves PDMS over no correction with lower CI > 0)"
    elif p1["lo"] > 0 and p2["lo"] > 0 and p3_ni:
        outcome = "PASS"
    elif p1["hi"] < 0:
        outcome = "WORSE"
    elif -eq_points <= p1["lo"] and p1["hi"] <= eq_points and -eq_pp <= p2["lo"] and p2["hi"] <= eq_pp:
        outcome = "EQUIVALENT"
    else:
        outcome = "INCONCLUSIVE"
    fam = {}
    for c, g in m.groupby("family"):
        fam[FAMILY_NAME.get(int(c), str(c))] = dict(
            n=int(len(g)), d_pdms_points_T_minus_none=100 * float((g.pdms_final_T - g.pdms_final_none).mean()),
            d_pdms_points_T=100 * float(g.d_pdms_T.mean()), d_pdms_points_none=100 * float(g.d_pdms_none.mean()),
            ncttc_reduction_pp=100 * float((g.fail_ncttc_none - g.fail_ncttc_T).mean()))
    res = dict(outcome=outcome, sanity=dict(ok=sanity, **{f"d_pdms_vs_orig_points_{a}": san[a] for a in arms}),
               endpoints=ep, p3_non_inferior=bool(p3_ni), dac_ddc_non_inferior=bool(dacddc_ni),
               arms=info, n_paired_drafts=int(len(m)), n_logs=int(len(np.unique(logs))), seeds=list(seeds),
               n_boot=n_boot, margin_pp=margin_pp, eq_points=eq_points, eq_pp=eq_pp, budget_def=budget_def,
               eval_name=eval_name, by_family=fam)
    if ci_level is not None:
        res["ci_level"] = lvl
    return res


# ----------------------------------------------------------------------------------------------- compare (run 4)
def parse_pairs(spec: str):
    """'T:none,M:none,TM:T,TM:M' -> [('T', 'none'), ...] (A:B = A minus B)."""
    out = []
    for p in [x for x in spec.split(",") if x.strip()]:
        a, b = [y.strip() for y in p.split(":")]
        if not a or not b or a == b:
            raise ValueError(f"bad pair {p!r}")
        out.append((a, b))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate pairs in {spec!r}")
    return out


def arm_rows(runs: Path, selection: Dict, arm: str, seeds: Sequence[int], final_fold: int, eval_name: str,
             budget_def: str):
    """-> (seed-averaged final outcomes at the arm's theta, per-arm info) -- the per-arm half of compare()."""
    sel = selection["arms"][arm]
    th, wt = sel["theta"], sel["wtag"]
    if th is None:
        raise ValueError(f"no theta selected for arm {arm}")
    frames = [final_outcomes(load_rows(run_dir(runs, wt, arm, final_fold, s) / eval_name), th) for s in seeds]
    per = seed_average(frames)
    ep_all = 100 * float(per.ep_loss.mean())
    ep_pass = float(np.mean([ep_loss_points(f, "passing") for f in frames]))
    ep_used = ep_all if budget_def == "all" else ep_pass
    info = dict(theta=th, wtag=wt, n_drafts=len(per), budget_def=budget_def,
                ep_loss_points=ep_used, ep_loss_points_all=ep_all, ep_loss_points_passing=ep_pass,
                n_passing=float(np.mean([f.pass_both.sum() for f in frames])),
                d_pdms_points=100 * float(per.d_pdms.mean()),
                pdms=float(per.pdms_final.mean()), pdms_orig=float(per.pdms_orig.mean()),
                modified_frac=float(per.modified.mean()),
                within_budget=bool(ep_used <= selection["budget_ep_points"] + 1e-12))
    return per, info


def pair_result(pa: pd.DataFrame, pb: pd.DataFrame, a: str, b: str, n_boot: int, level: float, margin_pp: float,
                eq_points: float, eq_pp: float) -> Dict:
    """The two-arm rule for arm a (role R_T) minus arm b (role R_none) on their paired drafts."""
    m = pa.merge(pb, on=["token", "k", "log", "family"], suffixes=("_A", "_B"), validate="one_to_one")
    logs = m.log.to_numpy()
    sc = lambda r: {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in r.items()}
    B = lambda d: sc(cluster_boot(np.asarray(d, np.float64), logs, n_boot, level=level))
    ep = {"P1_d_pdms_points": B(m.pdms_final_A - m.pdms_final_B),
          "P2_ncttc_reduction_pp": B(m.fail_ncttc_B - m.fail_ncttc_A),
          "P2ni_dac_excess_pp": B(m.fail_dac_A - m.fail_dac_B),
          "P2ni_ddc_excess_pp": B(m.fail_ddc_A - m.fail_ddc_B),
          "P3_new_fail_excess_pp": B(m.new_fail_A - m.new_fail_B)}
    san = {a: B(m.d_pdms_A), b: B(m.d_pdms_B)}
    sanity = any(x["lo"] > 0 for x in san.values())
    p1, p2, p3 = ep["P1_d_pdms_points"], ep["P2_ncttc_reduction_pp"], ep["P3_new_fail_excess_pp"]
    p3_ni = p3["hi"] <= margin_pp
    dacddc_ni = ep["P2ni_dac_excess_pp"]["hi"] <= margin_pp and ep["P2ni_ddc_excess_pp"]["hi"] <= margin_pp
    if not sanity:
        outcome = "INVALID (sanity: no arm improves PDMS over no correction with lower CI > 0)"
    elif p1["lo"] > 0 and p2["lo"] > 0 and p3_ni:
        outcome = "PASS"
    elif p1["hi"] < 0:
        outcome = "WORSE"
    elif -eq_points <= p1["lo"] and p1["hi"] <= eq_points and -eq_pp <= p2["lo"] and p2["hi"] <= eq_pp:
        outcome = "EQUIVALENT"
    else:
        outcome = "INCONCLUSIVE"
    fam = {}
    for c, g in m.groupby("family"):
        fam[FAMILY_NAME.get(int(c), str(c))] = {
            "n": int(len(g)), f"d_pdms_points_{a}_minus_{b}": 100 * float((g.pdms_final_A - g.pdms_final_B).mean()),
            f"d_pdms_points_{a}": 100 * float(g.d_pdms_A.mean()), f"d_pdms_points_{b}": 100 * float(g.d_pdms_B.mean()),
            "ncttc_reduction_pp": 100 * float((g.fail_ncttc_B - g.fail_ncttc_A).mean())}
    return dict(a=a, b=b, outcome=outcome, sanity=dict(ok=sanity, **{f"d_pdms_vs_orig_points_{x}": san[x] for x in (a, b)}),
                endpoints=ep, dac_excess_pp=ep["P2ni_dac_excess_pp"], p3_non_inferior=bool(p3_ni),
                dac_ddc_non_inferior=bool(dacddc_ni), n_paired_drafts=int(len(m)), n_logs=int(len(np.unique(logs))),
                n_unpaired_a=int(len(pa) - len(m)), n_unpaired_b=int(len(pb) - len(m)), ci_level=float(level),
                by_family=fam)


def compare_pairs(runs: Path, selection: Dict, pairs, seeds: Sequence[int], n_boot: int = 10000,
                  margin_pp: float = 0.5, eq_points: float = 0.2, eq_pp: float = 0.5, final_fold: int = -1,
                  budget_def: Optional[str] = None, eval_name: str = "eval_dev", ci_level: float = 0.95) -> Dict:
    """Run-4 multi-arm comparison (module docstring).  pairs: [(A, B), ...] or the --pairs string."""
    if isinstance(pairs, str):
        pairs = parse_pairs(pairs)
    sel_def = selection.get("budget_def", "all")
    if budget_def is None:
        budget_def = sel_def
    if budget_def != sel_def:
        raise ValueError(f"budget_def {budget_def!r} != the selection's {sel_def!r}")
    if budget_def not in BUDGET_DEFS:
        raise ValueError(f"budget_def {budget_def!r}")
    ci_percentiles(ci_level)
    labels = sorted({x for p in pairs for x in p}, key=lambda x: [y for p in pairs for y in p].index(x))
    per, info = {}, {}
    for lab in labels:
        arm, ev = lab.split("@", 1) if "@" in lab else (lab, eval_name)
        per[lab], info[lab] = arm_rows(runs, selection, arm, seeds, final_fold, ev, budget_def)
        if "@" in lab:
            info[lab]["eval_name"] = ev
    res = {f"{a}:{b}": pair_result(per[a], per[b], a, b, n_boot, ci_level, margin_pp, eq_points, eq_pp)
           for a, b in pairs}
    return dict(outcomes={k: v["outcome"] for k, v in res.items()}, pairs=res, arms=info,
                sanity_any_arm=any(r["sanity"][f"d_pdms_vs_orig_points_{x}"]["lo"] > 0 for r in res.values()
                                   for x in (r["a"], r["b"])),
                ci_level=float(ci_level), ci_percentiles=ci_percentiles(ci_level), seeds=list(seeds), n_boot=n_boot,
                margin_pp=margin_pp, eq_points=eq_points, eq_pp=eq_pp, budget_def=budget_def, eval_name=eval_name,
                final_fold=final_fold)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "compare"])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--wtags", default="stageT", help="comma list of run tags (one per loss-weight candidate)")
    ap.add_argument("--arms", default="T,none")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--budget-ep", type=float, default=0.5)
    ap.add_argument("--budget-def", choices=list(BUDGET_DEFS), default=None,
                    help="EP-loss budget drafts: all (run 1) | passing (NC, DAC, DDC pass before and after; AMENDMENT 3). "
                         "select: default all; compare: default = the selection's budget_def (a different one is refused)")
    ap.add_argument("--selection", default=str(REPO / "report/refiner_T/crossfit_selection.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--margin-pp", type=float, default=0.5)
    ap.add_argument("--eq-points", type=float, default=0.2)
    ap.add_argument("--eq-pp", type=float, default=0.5)
    ap.add_argument("--final-fold", type=int, default=-1,
                    help="fold of the runs whose eval_dev rows are compared (-1 = trained on all folds; 0 = the reduced "
                         "single-run plan: the fold-0 run both selects theta on its OOF fold and is evaluated on dev)")
    ap.add_argument("--eval-name", default="eval_dev",
                    help="compare: per-run eval subdirectory to compare (default eval_dev; eval_navtest = AMENDMENT 5)")
    ap.add_argument("--pairs", default=None,
                    help='run 4: comma list A:B (A minus B), e.g. "T:none,M:none,TM:T,TM:M"; default: the two-arm T vs none')
    ap.add_argument("--ci-level", type=float, default=None,
                    help="two-sided CI level of every CI used by the outcome rule (default 0.95; run 4: 0.9875)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",") if x != ""]
    if a.cmd == "select":
        res = select(Path(a.runs), a.wtags.split(","), a.arms.split(","), ints(a.seeds), ints(a.folds), a.budget_ep,
                     budget_def=a.budget_def or "all")
    else:
        sel = json.loads(Path(a.selection).read_text())
        try:
            if a.pairs:
                res = compare_pairs(Path(a.runs), sel, a.pairs, ints(a.seeds), a.n_boot, a.margin_pp, a.eq_points,
                                    a.eq_pp, a.final_fold, budget_def=a.budget_def, eval_name=a.eval_name,
                                    ci_level=0.95 if a.ci_level is None else a.ci_level)
            else:
                res = compare(Path(a.runs), sel, ints(a.seeds), a.n_boot, a.margin_pp, a.eq_points, a.eq_pp,
                              a.final_fold, budget_def=a.budget_def, eval_name=a.eval_name, ci_level=a.ci_level)
        except ValueError as e:
            if "budget_def" in str(e):
                raise SystemExit(str(e))
            raise
        res["selection_file"] = a.selection
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    short = {k: v for k, v in res.items() if k not in ("by_family",)}
    if a.cmd == "compare" and a.pairs:
        short = {k: v for k, v in res.items() if k != "pairs"}
        short["pairs"] = {k: {kk: vv for kk, vv in v.items() if kk != "by_family"} for k, v in res["pairs"].items()}
    if a.cmd == "select":
        short = {arm: {k: v for k, v in d.items() if k != "candidates"} for arm, d in res["arms"].items()}
    print(json.dumps(short, indent=1, default=float))


if __name__ == "__main__":
    main()
