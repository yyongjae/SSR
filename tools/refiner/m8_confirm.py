#!/usr/bin/env python
"""M8 disjoint-pool CONFIRMATION of the train-split margin selection (tools/refiner/m8_recheck.py, STATUS.md D1).

Pre-stated rule: report/refiner_T/m8_recheck/confirm/CONFIRM_RULE.txt (written before any number of this run).
Setting under test S* = (m_col 0.15, m_dac 0.10, SAT mask) = m8_recheck sid 14; reference D = code default (0.3, 0.2, SAT)
= sid 23.  Reads ONLY stage-T train-split data (the m8_recheck train-only guard); dev tokens and navtest are never read.

Pool: train-split tokens (m8_recheck.eligible_tokens) whose LOG is not among the sweep pool's logs.  Only 91 train logs are
left, so the pool takes up to c tokens per log (c = smallest cap giving >= --n tokens; seed 0) and every CI is a
log-cluster bootstrap.

Subcommands
  select    --n 800 --seed 0     -> <out>/tokens.parquet, <out>/labels_pool.parquet,
                                    <report>/{pool_disjointness.json, tokens_sweep.csv, tokens_confirm.csv}
  (run      m8_recheck.py run --out <out> --workers 2          unchanged sweep code on the confirmation pool)
  (summ.    m8_recheck.py summarize --out <out> --report <report>)
  analyze                        -> <report>/{confirm.json, confirm_table.csv}  (verdict per CONFIRM_RULE.txt)
CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import m8_recheck as R  # noqa: E402

SWEEP_OUT = R.OUT                                    # /home/external-user/ssd/yongjae_refiner/m8_recheck
OUT = R.OUT / "confirm"
REPORT_DIR = R.REPORT_DIR / "confirm"
SEL = dict(mask="sat", m_col=0.15, m_dac=0.1)        # S*
DEF = dict(mask="sat", m_col=0.3, m_dac=0.2)         # D (SurrogateConfig() defaults)
N_BOOT = 2000


def sid_of(s: dict) -> int:
    m = [x["sid"] for x in R.SETTINGS if x["mask"] == s["mask"] and x["m_col"] == s["m_col"] and x["m_dac"] == s["m_dac"]]
    assert len(m) == 1, s
    return m[0]


# ----------------------------------------------------------------------------------------------- select
def cap_for(n_per_log: pd.Series, n: int) -> int:
    """smallest per-log cap c with sum(min(n_log, c)) >= n (or the largest log size if the pool is smaller)."""
    for c in range(1, int(n_per_log.max()) + 1):
        if int(np.minimum(n_per_log, c).sum()) >= n:
            return c
    return int(n_per_log.max())


def pick_capped(s: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, int]:
    """up to c tokens per log (c = cap_for), drawn without replacement; logs sorted, tokens sorted within a log."""
    s = s.sort_values(["log", "token"]).reset_index(drop=True)
    c = cap_for(s.groupby("log").size(), n)
    rng = np.random.default_rng(seed)
    pick = []
    for _, g in s.groupby("log", sort=True):
        k = min(len(g), c)
        pick += sorted(g.index[rng.choice(len(g), k, replace=False)].tolist())
    return s.loc[pick].reset_index(drop=True), c


def disjoint_proof(sweep: pd.DataFrame, conf: pd.DataFrame) -> dict:
    tok_i = sorted(set(sweep.token) & set(conf.token))
    log_i = sorted(set(sweep.log) & set(conf.log))
    return {"sweep": {"n_tokens": int(len(sweep)), "n_logs": int(sweep.log.nunique())},
            "confirm": {"n_tokens": int(len(conf)), "n_logs": int(conf.log.nunique())},
            "token_intersection": tok_i, "log_intersection": log_i,
            "disjoint_tokens": not tok_i, "disjoint_logs": not log_i}


def cmd_select(a):
    out, rep = Path(a.out), Path(a.report)
    sweep = pd.read_parquet(Path(a.sweep_out) / "tokens.parquet")
    e = R.eligible_tokens()
    rest = e[~e.log.isin(set(sweep.log))]
    p, c = pick_capped(rest, a.n, a.seed)
    proof = disjoint_proof(sweep, p)
    assert proof["disjoint_tokens"] and proof["disjoint_logs"], proof
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    p[["token", "log", "frame_idx"]].to_parquet(out / "tokens.parquet")
    lab = pd.read_parquet(R.LABELS, columns=["token", "k", *R.LAB_KEYS, "family"])
    lab[lab.token.isin(set(p.token))].reset_index(drop=True).to_parquet(out / "labels_pool.parquet")
    sweep[["token", "log"]].sort_values(["log", "token"]).to_csv(rep / "tokens_sweep.csv", index=False)
    p[["token", "log"]].to_csv(rep / "tokens_confirm.csv", index=False)
    proof.update(eligible=int(len(e)), eligible_logs=int(e.log.nunique()), unused_eligible=int(len(rest)),
                 unused_logs=int(rest.log.nunique()), cap_per_log=int(c), seed=int(a.seed), target_n=int(a.n),
                 tokens_per_log=p.groupby("log").size().describe().to_dict(), created=time.strftime("%F %T"))
    (rep / "pool_disjointness.json").write_text(json.dumps(proof, indent=1, default=float))
    print(json.dumps({k: v for k, v in proof.items() if k not in ("tokens_per_log",)}, default=float))


# ----------------------------------------------------------------------------------------------- analyze helpers
def load_shards(out: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("part-*.parquet"))], ignore_index=True)
    df = df[df.error == ""].copy()
    df["sid"] = df.sid.astype(int)
    return df


def _cluster_sums(log, cols: dict) -> tuple[np.ndarray, dict]:
    codes, uniq = pd.factorize(pd.Series(np.asarray(log)))
    return codes, {k: np.bincount(codes, weights=np.asarray(v, float), minlength=len(uniq)) for k, v in cols.items()}


def boot_ratio_diff(log, num_a, den_a, num_b, den_b, n=N_BOOT, seed=0) -> dict:
    """paired log-cluster bootstrap of sum(num_a)/sum(den_a) - sum(num_b)/sum(den_b) (rows share the log column)."""
    _, s = _cluster_sums(log, dict(na=num_a, da=den_a, nb=num_b, db=den_b))
    L = len(s["na"])
    cnt = np.random.default_rng(seed).multinomial(L, np.full(L, 1.0 / L), size=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ra, rb = (cnt @ s["na"]) / (cnt @ s["da"]), (cnt @ s["nb"]) / (cnt @ s["db"])
    v = (ra - rb)[np.isfinite(ra - rb)]
    pa = s["na"].sum() / s["da"].sum() if s["da"].sum() else float("nan")
    pb = s["nb"].sum() / s["db"].sum() if s["db"].sum() else float("nan")
    return {"a": float(pa), "b": float(pb), "diff": float(pa - pb), "k_a": int(s["na"].sum()), "n_a": int(s["da"].sum()),
            "k_b": int(s["nb"].sum()), "n_b": int(s["db"].sum()),
            "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2}


def boot_indep_diff(log_a, num_a, den_a, log_b, num_b, den_b, n=N_BOOT, seed=0) -> dict:
    """independent log bootstraps of two pools: CI of ratio_a - ratio_b."""
    def draws(log, num, den, sd):
        _, s = _cluster_sums(log, dict(n=num, d=den))
        L = len(s["n"])
        cnt = np.random.default_rng(sd).multinomial(L, np.full(L, 1.0 / L), size=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (cnt @ s["n"]) / (cnt @ s["d"]), s["n"].sum() / max(s["d"].sum(), 1e-12)
    va, pa = draws(log_a, num_a, den_a, seed)
    vb, pb = draws(log_b, num_b, den_b, seed + 1)
    v = (va - vb)[np.isfinite(va - vb)]
    return {"a": float(pa), "b": float(pb), "diff": float(pa - pb),
            "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2}


def criteria_arrays(df: pd.DataFrame, s: dict) -> dict:
    """(log, num, den) arrays of the four pre-stated criteria of setting s (same definitions as setting_metrics)."""
    mc, md, mk = s["m_col"], s["m_dac"], s["mask"]
    bank = df[df.pool == "bank"]
    human = bank[bank.k == 0]
    g = df[(df.pool == "guided") & (df.sid == sid_of(s)) & df.modified]
    s0 = bank.set_index(["token", "k"]).loc[list(zip(g.token, g.k))]
    sf = R._flag(s0, "col", mc, mk) & ~R._flag(g, "col", mc, mk)
    fixed = (s0.nc < 1).to_numpy() & (g.nc == 1).to_numpy()
    return {"A1": (bank.log.to_numpy(), R._flag(bank, "col", mc, mk), (bank.nc < 1).to_numpy()),
            "A2": (human.log.to_numpy(), R._flag(human, "col", mc, mk), (human.nc == 1).to_numpy()),
            "A3": (g.log.to_numpy(), fixed, sf),
            "DAC_human_FA": (human.log.to_numpy(), R._flag(human, "dac", md, mk), (human.dac == 1).to_numpy())}


def paired_arrays(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """one row per valid perturbed source (token, k): official before / after guided(s) + surrogate flags."""
    mc, md, mk = s["m_col"], s["m_dac"], s["mask"]
    bank = df[df.pool == "bank"].set_index(["token", "k"])
    g = df[(df.pool == "guided") & (df.sid == sid_of(s))].sort_values(["token", "k"])
    s0 = bank.loc[list(zip(g.token, g.k))]
    return pd.DataFrame({
        "token": g.token.to_numpy(), "k": g.k.to_numpy(), "log": g.log.to_numpy(), "modified": g.modified.to_numpy(),
        "nc0": (s0.nc < 1).to_numpy(), "nc1": (g.nc < 1).to_numpy(),
        "dac0": (s0.dac < 1).to_numpy(), "dac1": (g.dac < 1).to_numpy(),
        "ddc0": (s0.ddc < 1).to_numpy(), "ddc1": (g.ddc < 1).to_numpy(),
        "ttc0": (s0.ttc < 1).to_numpy(), "ttc1": (g.ttc < 1).to_numpy(),
        "pdms0": s0.pdms.to_numpy(), "pdms1": g.pdms.to_numpy(), "ep0": s0.ep.to_numpy(), "ep1": g.ep.to_numpy(),
        "c0": R._flag(s0, "col", mc, mk), "c1": R._flag(g, "col", mc, mk),
        "q0": R._flag(s0, "dac", md, mk), "q1": R._flag(g, "dac", md, mk),
        "t_c1": (g[f"t_g_s_{mk}"] < mc).to_numpy()})


def paired_compare(pa: pd.DataFrame, pb: pd.DataFrame) -> dict:
    """paired S* (a) - D (b) on the same sources: safety quantities with log-bootstrap CIs of the difference."""
    assert (pa.token.to_numpy() == pb.token.to_numpy()).all() and (pa.k.to_numpy() == pb.k.to_numpy()).all()
    log = pa.log.to_numpy()
    one = np.ones(len(pa), bool)

    def q(num_a, den_a, num_b, den_b, sd):
        return boot_ratio_diff(log, num_a & den_a, den_a, num_b & den_b, den_b, seed=sd)
    out = {
        "new NC failure rate (all sources)": q(~pa.nc0 & pa.nc1, one, ~pb.nc0 & pb.nc1, one, 1),
        "new DAC failure rate (all sources)": q(~pa.dac0 & pa.dac1, one, ~pb.dac0 & pb.dac1, one, 2),
        "new DDC failure rate (all sources)": q(~pa.ddc0 & pa.ddc1, one, ~pb.ddc0 & pb.ddc1, one, 3),
        "new TTC failure rate (all sources)": q(~pa.ttc0 & pa.ttc1, one, ~pb.ttc0 & pb.ttc1, one, 4),
        "residual NC failure rate after correction (all sources)": q(pa.nc1, one, pb.nc1, one, 5),
        "P(NC fixed | source NC fail)": q(~pa.nc1, pa.nc0, ~pb.nc1, pb.nc0, 6),
        "P(NC still fails | surrogate says fixed)": q(pa.nc1, pa.c0 & ~pa.c1, pb.nc1, pb.c0 & ~pb.c1, 7),
        "P(NC fails after | modified)": q(pa.nc1, pa.modified, pb.nc1, pb.modified, 8),
        "P(NC fails after | modified & source NC fail)": q(pa.nc1, pa.modified & pa.nc0, pb.nc1,
                                                          pb.modified & pb.nc0, 9),
        "raw-invisible residual NC failure rate (all sources)": q(pa.nc1 & ~pa.c1, one, pb.nc1 & ~pb.c1, one, 10),
        "... of which tracked states flag (rate)": q(pa.nc1 & ~pa.c1 & pa.t_c1, one, pb.nc1 & ~pb.c1 & pb.t_c1, one, 11),
        "residual TTC failure rate after correction (all sources)": q(pa.ttc1, one, pb.ttc1, one, 12),
        "P(TTC fixed | source TTC fail)": q(~pa.ttc1, pa.ttc0, ~pb.ttc1, pb.ttc0, 13),
        "residual DAC failure rate after correction (all sources)": q(pa.dac1, one, pb.dac1, one, 14),
        "modified rate": q(pa.modified, one, pb.modified, one, 15),
    }
    def mean_diff(xa, xb, sel, sd):
        _, s = _cluster_sums(log, dict(a=np.where(sel, xa, 0.0), b=np.where(sel, xb, 0.0), n=sel))
        L = len(s["n"])
        cnt = np.random.default_rng(sd).multinomial(L, np.full(L, 1.0 / L), size=N_BOOT)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (cnt @ s["a"] - cnt @ s["b"]) / (cnt @ s["n"])
        v = v[np.isfinite(v)]
        n = s["n"].sum()
        return {"a": float(s["a"].sum() / n), "b": float(s["b"].sum() / n), "diff": float((s["a"].sum() - s["b"].sum()) / n),
                "n": int(n), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    ok0 = (~pa.nc0 & ~pa.dac0 & ~pa.ddc0).to_numpy()
    out["mean dPDMS (S* - D)"] = mean_diff(pa.pdms1 - pa.pdms0, pb.pdms1 - pb.pdms0, one, 16)
    out["mean dPDMS | source passes NC, DAC, DDC"] = mean_diff(pa.pdms1 - pa.pdms0, pb.pdms1 - pb.pdms0, ok0, 17)
    out["mean dPDMS | source fails NC, DAC or DDC"] = mean_diff(pa.pdms1 - pa.pdms0, pb.pdms1 - pb.pdms0, ~ok0, 18)
    out["mean dEP | source passes NC, DAC, DDC"] = mean_diff(pa.ep1 - pa.ep0, pb.ep1 - pb.ep0, ok0, 19)
    out["n_sources"] = int(len(pa))
    return out


def criteria_row(m: dict) -> dict:
    return {c: {"p": m[c]["p"], "k": m[c]["k"], "n": m[c]["n"], "cp": m[c]["ci"], "boot_log": m[c]["boot_log"]}
            for c in ("A1", "A2", "A3", "DAC_human_FA", "DAC_recall", "DAC_fpr", "DAC_A3")}


THR = {"A1": (R.A1_MIN, ">="), "A2": (R.A2_MAX, "<="), "A3": (R.A3_MIN, ">="), "DAC_human_FA": (R.DAC_HFA_MAX, "<=")}


def fragile(m: dict) -> dict:
    """criteria whose log-bootstrap 95% CI contains the threshold."""
    return {c: bool(m[c]["boot_log"][0] <= t <= m[c]["boot_log"][1]) for c, (t, _) in THR.items()}


def boot_pass_frac(log, num, den, thr: float, op: str, n=N_BOOT, seed=0) -> float:
    """fraction of log-bootstrap resamples in which sum(num)/sum(den) meets the threshold (op '>=' or '<=')."""
    _, s = _cluster_sums(log, dict(n=num & den, d=den))
    L = len(s["n"])
    cnt = np.random.default_rng(seed).multinomial(L, np.full(L, 1.0 / L), size=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (cnt @ s["n"]) / (cnt @ s["d"])
    v = v[np.isfinite(v)]
    return float(np.mean(v >= thr) if op == ">=" else np.mean(v <= thr))


def rule_on_tracked(tt: pd.DataFrame) -> dict:
    """SELECTION_RULE order applied to the tracked-state CEILING table (grid_tracked.csv; indicative only: the pairs
    were optimised with the raw-reference surrogate, not with an M7b tracker)."""
    t = tt.copy()
    t["feas_t"] = (t.A1_t >= R.A1_MIN) & (t.A2_t <= R.A2_MAX) & (t.A3_t >= R.A3_MIN) & (t.DAC_hFA_t <= R.DAC_HFA_MAX)
    t["mask_pref"] = t["mask"].map(R.MASK_PREF)
    F = t[t.feas_t].sort_values(["m_col", "m_dac", "mask_pref"], ascending=False)
    top = F.iloc[0].to_dict() if len(F) else None
    return {"n_feasible": int(len(F)), "choice": top,
            "feasible": [f"({r.m_col:g}, {r.m_dac:g}, {r['mask']})" for _, r in F.iterrows() if r["mask"] == "sat"]}


def verdict(feas_sel: dict, guard: dict) -> dict:
    g_nc = guard["new NC failure rate (all sources)"]["ci"][0] > 0
    g_dac = guard["new DAC failure rate (all sources)"]["ci"][0] > 0
    ok = bool(feas_sel["all"] and not g_nc and not g_dac)
    return {"verdict": "CONFIRMED" if ok else "NOT CONFIRMED", "criteria_point": feas_sel["criteria"],
            "guard_NC_worse_than_D": bool(g_nc), "guard_DAC_worse_than_D": bool(g_dac)}


def both_pool_choice(tab_s: pd.DataFrame, tab_c: pd.DataFrame, guard_ok: dict | None = None) -> dict:
    t = tab_c.merge(tab_s[["sid", "feasible"]].rename(columns={"feasible": "feasible_sweep"}), on="sid")
    t["feasible_both"] = t.feasible & t.feasible_sweep
    if guard_ok is not None:
        t["feasible_both"] &= t.sid.map(lambda x: bool(guard_ok.get(int(x), True)))
    t["mask_pref"] = t["mask"].map(R.MASK_PREF)
    F = t[t.feasible_both].sort_values(["m_col", "m_dac", "mask_pref"], ascending=False)
    return {"feasible_both_sids": [int(x) for x in F.sid],
            "feasible_both": [f"({r.m_col:g}, {r.m_dac:g}, {r['mask']})" for _, r in F.iterrows()],
            "rule_choice_sid": int(F.sid.iloc[0]) if len(F) else None}


# ----------------------------------------------------------------------------------------------- analyze
def cmd_analyze(a):
    out, sweep_out, rep = Path(a.out), Path(a.sweep_out), Path(a.report)
    dc, ds = load_shards(out), load_shards(sweep_out)
    sid_sel, sid_def = sid_of(SEL), sid_of(DEF)
    tab_c = pd.read_csv(rep / "grid_table.csv")
    tab_s = pd.read_csv(R.REPORT_DIR / "grid_table.csv")
    summ_c = json.loads((rep / "m8_recheck.json").read_text())
    assert summ_c["n_tokens"] == dc.token.nunique() and summ_c["n_rows"] == len(dc), \
        "grid_table / m8_recheck.json are stale: rerun m8_recheck.py summarize on the same shards first"
    res = {"created": time.strftime("%F %T"), "rule_file": str(rep / "CONFIRM_RULE.txt"),
           "pool": json.loads((rep / "pool_disjointness.json").read_text()),
           "confirm_run": {k: summ_c[k] for k in ("n_tokens", "n_logs", "n_rows", "n_error_tokens", "bank",
                                                  "timing_s_per_token")},
           "S*": {"sid": sid_sel, **SEL}, "D": {"sid": sid_def, **DEF}}
    msel = R.setting_metrics(dc, {**SEL, "sid": sid_sel})
    mdef = R.setting_metrics(dc, {**DEF, "sid": sid_def})
    fsel, fdef = R.feasible(msel), R.feasible(mdef)
    pa, pb = paired_arrays(dc, SEL), paired_arrays(dc, DEF)
    guard = paired_compare(pa, pb)
    res["verdict"] = verdict(fsel, guard)
    res["S*_confirm"] = {"criteria": criteria_row(msel), "feasible": fsel, "fragile_CI_contains_threshold": fragile(msel),
                         "A3_decomp": msel["A3_decomp"], "effect_all_sources": msel["effect_all_sources"],
                         "residual_nc_gap": msel["residual_nc_gap"], "residual_dac_gap": msel["residual_dac_gap"]}
    res["D_confirm"] = {"criteria": criteria_row(mdef), "feasible": fdef, "fragile_CI_contains_threshold": fragile(mdef),
                        "A3_decomp": mdef["A3_decomp"], "effect_all_sources": mdef["effect_all_sources"],
                        "residual_nc_gap": mdef["residual_nc_gap"], "residual_dac_gap": mdef["residual_dac_gap"]}
    res["paired_S*_minus_D_confirm"] = guard
    # sweep pool, same quantities (for the sweep-vs-confirm comparison)
    res["paired_S*_minus_D_sweep"] = paired_compare(paired_arrays(ds, SEL), paired_arrays(ds, DEF))
    # sweep vs confirm per criterion
    svc = {}
    for name, s in (("S*", SEL), ("D", DEF)):
        ac, as_ = criteria_arrays(dc, s), criteria_arrays(ds, s)
        svc[name] = {c: boot_indep_diff(ac[c][0], ac[c][1] & ac[c][2], ac[c][2], as_[c][0], as_[c][1] & as_[c][2],
                                        as_[c][2], seed=20 + i)
                     for i, c in enumerate(THR)}
        for c in THR:
            svc[name][c]["note"] = "a = confirm, b = sweep, diff = confirm - sweep"
    res["sweep_vs_confirm"] = svc
    # selection rule re-applied
    res["rule_on_confirm_alone"] = summ_c["selection"]
    guard_ok = {}
    for s in R.SETTINGS:
        gq = paired_compare(paired_arrays(dc, s), pb)
        guard_ok[s["sid"]] = not (gq["new NC failure rate (all sources)"]["ci"][0] > 0
                                  or gq["new DAC failure rate (all sources)"]["ci"][0] > 0)
    res["guard_ok_vs_D_by_sid"] = {str(k): v for k, v in guard_ok.items()}
    res["rule_on_both_pools"] = both_pool_choice(tab_s, tab_c, guard_ok)
    # fallback (CONFIRM_RULE: used for training only if S* is NOT CONFIRMED): the both-pools rule choice
    fb = res["rule_on_both_pools"]["rule_choice_sid"]
    if fb is not None and fb != sid_sel:
        sfb = {k: v for k, v in R.SETTINGS[fb].items() if k != "sid"}
        assert R.SETTINGS[fb]["sid"] == fb
        blk = {"setting": {"sid": fb, **sfb}}
        for pool, dd in (("confirm", dc), ("sweep", ds)):
            m = R.setting_metrics(dd, {**sfb, "sid": fb})
            blk[pool] = {"criteria": criteria_row(m), "feasible": R.feasible(m),
                         "fragile_CI_contains_threshold": fragile(m), "A3_decomp": m["A3_decomp"],
                         "effect_all_sources": m["effect_all_sources"], "residual_nc_gap": m["residual_nc_gap"],
                         "paired_minus_D": paired_compare(paired_arrays(dd, sfb), paired_arrays(dd, DEF))}
        ac = criteria_arrays(dc, sfb)
        blk["boot_pass_fraction_confirm"] = {c: boot_pass_frac(ac[c][0], ac[c][1], ac[c][2], t, op, seed=60 + i)
                                             for i, (c, (t, op)) in enumerate(THR.items())}
        mt = R.setting_metrics(dc, {**sfb, "sid": fb}, tracked=True)
        blk["tracked_ceiling_confirm"] = {c: {"p": mt[c]["p"], "k": mt[c]["k"], "n": mt[c]["n"]}
                                          for c in ("A1", "A2", "A3", "DAC_recall", "DAC_human_FA", "DAC_A3")}
        res["fallback_both_pools"] = blk
    # tracked-state ceiling (ideal M7b) for S* and D on the confirm pool
    tr = {}
    for name, s, sid in (("S*", SEL, sid_sel), ("D", DEF, sid_def)):
        mt = R.setting_metrics(dc, {**s, "sid": sid}, tracked=True)
        tr[name] = {c: {"p": mt[c]["p"], "k": mt[c]["k"], "n": mt[c]["n"], "boot_log": mt[c]["boot_log"]}
                    for c in ("A1", "A2", "A3", "DAC_recall", "DAC_human_FA", "DAC_A3")}
    res["tracked_ceiling_confirm"] = tr
    # bootstrap fraction of resamples that pass each criterion (confirm pool)
    bp = {}
    for name, s in (("S*", SEL), ("D", DEF)):
        ac = criteria_arrays(dc, s)
        bp[name] = {c: boot_pass_frac(ac[c][0], ac[c][1], ac[c][2], t, op, seed=40 + i)
                    for i, (c, (t, op)) in enumerate(THR.items())}
    res["boot_pass_fraction_confirm"] = bp
    # tracked-state ceiling rule (D1-b / M7b context; indicative only)
    res["rule_on_tracked_ceiling"] = {"sweep": rule_on_tracked(pd.read_csv(R.REPORT_DIR / "grid_tracked.csv")),
                                      "confirm": rule_on_tracked(pd.read_csv(rep / "grid_tracked.csv"))}
    # compact table: S* and D x (sweep, confirm)
    rows = []
    for name, s, sid in (("S*", SEL, sid_sel), ("D", DEF, sid_def)):
        for pool, tab in (("sweep", tab_s), ("confirm", tab_c)):
            r = tab[tab.sid == sid].iloc[0].to_dict()
            rows.append({"setting": name, "pool": pool, **{k: r[k] for k in (
                "m_col", "m_dac", "mask", "A1", "A1_lo", "A1_hi", "A2", "A2_lo", "A2_hi", "A3", "A3_lo", "A3_hi", "A3_k",
                "A3_n", "DAC_hFA", "DAC_hFA_lo", "DAC_hFA_hi", "DAC_rec", "DAC_fpr", "DAC_A3", "nc_fixed", "nc_new",
                "dac_fixed", "dac_new", "res_nc", "res_nc_invisible", "res_nc_inv_tracked", "feasible")}})
    pd.DataFrame(rows).to_csv(rep / "confirm_table.csv", index=False, float_format="%.4f")
    (rep / "confirm.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"verdict": res["verdict"], "rule_on_confirm_alone": res["rule_on_confirm_alone"].get("setting"),
                      "rule_on_both_pools": res["rule_on_both_pools"]}, indent=1, default=float))
    print(f"-> {rep}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("select", cmd_select), ("analyze", cmd_analyze)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--out", default=str(OUT))
        p.add_argument("--sweep-out", default=str(SWEEP_OUT))
        p.add_argument("--report", default=str(REPORT_DIR))
        if name == "select":
            p.add_argument("--n", type=int, default=800)
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
