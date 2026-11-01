#!/usr/bin/env python
"""m_ttc selection on the TRAIN D1-b sweep pool (PRESTATED_DECISION_RULE AMENDMENT 4 (2)).

Pre-stated rule and implementation details: report/refiner_T/ttc_select/SELECTION_RULE.txt (written before any number).
Reads ONLY the D1-b sweep pool (<data>/m8_recheck/tokens.parquet, 800 train tokens / 800 logs), its labels snapshot
(labels_pool.parquet), drafts/train, objects/train, human/train and the sweep shards (cross-check).  Dev / navtest paths are
refused (m8_recheck._train_only).

Per token (ttc_flag_check.token_rows loading): bank = k = 0 human identity + the VALID perturbed drafts k = 1..12 (invalid
slots skipped), raw dense references, scene = scene_from_numpy(objects 0..5 s, human_traj).  Margin-free per-draft columns:
  col_g        C_col gmin, SurrogateConfig()              -> collision flag at m_col 0.15 <=> col_g < 0.15
  ttc_g        C_ttc gmin, SurrogateConfig()              -> projected flag at m <=> ttc_g < m
  ttc_g_stop   C_ttc gmin with the official stopped-ego skip (ttc_min_speed 5e-3)   (diagnostic)
  ttc_g_n0     C_ttc gmin with n_from = 0                                           (diagnostic)
TTC flag(m) = col_g < M_COL_FLAG or ttc_g < m.  T1 = P(flag | ttc < 1) over bank drafts; T2 = P(flag | k = 0, ttc == 1).
Feasible = T1 >= 0.70 and T2 <= 0.02 (point estimates); choice = the largest feasible m_ttc, else fewest failed criteria,
then smallest shortfall, then the larger margin.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python tools/refiner/ttc_select.py run --workers 2
  ... python tools/refiner/ttc_select.py summarize
CPU only; torch 1 thread per worker; <= 2 workers (shared machine).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))

import m8_recheck as R  # noqa: E402
import ttc_flag_check as TF  # noqa: E402

OUT = R.DATA / "ttc_select"
REPORT_DIR = ROOT / "report/refiner_T/ttc_select"
POOL, LABELS_POOL, SHARDS = TF.POOL, TF.LABELS_POOL, R.OUT / "shards"
M_TTCS = (0.0, 0.05, 0.10, 0.15)
M_COL_FLAG = 0.15
T1_MIN, T2_MAX = 0.70, 0.02


# ----------------------------------------------------------------------------------------------- per token
def token_rows(task):
    import torch
    from navsim.agents.para_ssr.refiner import gt_future as GF
    from navsim.agents.para_ssr.refiner import surrogate as SU
    from navsim.agents.para_ssr.refiner.geometry import dense_reference
    token, log = task
    t0 = time.time()
    try:
        dpath, opath = R.DRAFT_DIR / f"{token}.npz", R.OBJ_DIR / f"{token}.npz"
        for p in (dpath, opath):
            R._train_only(p)
        with np.load(dpath) as z:
            drafts = np.asarray(z["drafts"], np.float32)
            fam = np.asarray(z["family"]).astype(int)
            valid = np.asarray(z["valid"]).astype(bool)
            assert str(z["split"]) == "train", str(z["split"])
        i = TF._W["hidx"][token]
        assert not bool(TF._W["h"]["frame_gap"][i])
        objs = GF.load_objects(opath)
        scene = SU.collate_scenes([SU.scene_from_numpy(objs, human_traj=TF._W["h"]["traj"][i])])
        bank_k = [0] + [k for k in range(1, 13) if valid[k]]
        dense = dense_reference(torch.as_tensor(drafts[bank_k].astype(np.float64)))
        idx = torch.zeros(len(bank_k), dtype=torch.long)
        with torch.no_grad():
            col = SU.collision_cost(dense, scene, idx, SU.SurrogateConfig())
            ttc = SU.ttc_cost(dense, scene, idx, SU.SurrogateConfig(), details=True)
            stp = SU.ttc_cost(dense, scene, idx, SU.SurrogateConfig(ttc_min_speed=TF.STOP_SPEED), details=True)
            n0 = SU.ttc_cost(dense, scene, idx, SU.SurrogateConfig(n_from=0), details=True)
        sec = time.time() - t0
        return [dict(token=token, log=log, k=int(k), fam=int(fam[k]), col_g=float(col["gmin"][j]),
                     ttc_g=float(ttc["gmin"][j]), ttc_g_hard=float(ttc["gmin_hard"][j]),
                     ttc_g_stop=float(stp["gmin"][j]), ttc_g_n0=float(n0["gmin"][j]),
                     ttc_first_n=int(ttc["first_n"][j]), n_obj=int(objs["kf"].shape[0]), error="",
                     sec=(sec if j == 0 else np.nan)) for j, k in enumerate(bank_k)]
    except Exception as e:  # noqa: BLE001
        import traceback
        return [dict(token=token, log=log, k=-1, error=f"{type(e).__name__}: {e} | {traceback.format_exc()[-800:]}")]


def cmd_run(a):
    for p in (POOL, LABELS_POOL):
        R._train_only(p)
    toks = pd.read_parquet(POOL)
    if a.limit:
        toks = toks.iloc[: a.limit]
    tasks = list(zip(toks.token, toks.log))
    print(f"[ttcsel] {len(tasks)} tokens, workers {a.workers}", flush=True)
    t0, rows, n = time.time(), [], 0
    if a.workers <= 1:
        TF._init()
        it = map(token_rows, tasks)
    else:
        from multiprocessing import Pool
        pool = Pool(a.workers, initializer=TF._init)
        it = pool.imap_unordered(token_rows, tasks, chunksize=1)
    for r in it:
        rows.extend(r)
        n += 1
        if n % 50 == 0:
            el = time.time() - t0
            print(f"[ttcsel] {n}/{len(tasks)} {el:.0f}s ETA {el / n * (len(tasks) - n) / 60:.1f} min", flush=True)
    if a.workers > 1:
        pool.close()
        pool.join()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "rows.parquet")
    print(f"[ttcsel] finished {n} tokens in {time.time() - t0:.0f}s -> {out / 'rows.parquet'}", flush=True)


# ----------------------------------------------------------------------------------------------- statistics
def _stat(num, den, log) -> dict:
    import validate_surrogate as V
    num, den = np.asarray(num, bool), np.asarray(den, bool)
    r = V._rate(num, den)
    r["boot_log"] = V._boot(np.asarray(log)[den], num[den], np.ones(int(den.sum()))) if den.any() else [np.nan] * 2
    return r


def flags(df: pd.DataFrame, m_ttc: float, kind: str = "combo", col: str = "ttc_g") -> np.ndarray:
    fc = (df.col_g < M_COL_FLAG).to_numpy()
    fp = (df[col] < m_ttc).to_numpy()
    return {"combo": fc | fp, "col_only": fc, "proj_only": fp}[kind]


def t1_t2(df: pd.DataFrame, f: np.ndarray) -> dict:
    """T1 over all bank drafts with ttc < 1, T2 over human drafts (k = 0) with ttc == 1, for per-row flags f."""
    fail = (df.ttc < 1).to_numpy()
    hum = (df.k == 0).to_numpy()
    log = df.log.to_numpy()
    return {"T1": _stat(f, fail, log), "T2": _stat(f[hum], ~fail[hum], log[hum]),
            "T1_perturbed_only": _stat(f, fail & ~hum, log), "FA_bank": _stat(f, ~fail, log)}


def feasible(t1: float, t2: float) -> dict:
    c = {"T1": bool(t1 >= T1_MIN), "T2": bool(t2 <= T2_MAX)}
    return {"criteria": c, "all": all(c.values()), "n_failed": int(sum(not v for v in c.values())),
            "shortfall": float(max(0.0, T1_MIN - t1) + max(0.0, t2 - T2_MAX))}


def choose(table: pd.DataFrame) -> dict:
    """pre-stated rule on a table with columns m_ttc, T1, T2 (point estimates)."""
    t = table.copy()
    fz = [feasible(a, b) for a, b in zip(t.T1, t.T2)]
    t["feasible"] = [f["all"] for f in fz]
    t["n_failed"] = [f["n_failed"] for f in fz]
    t["shortfall"] = [f["shortfall"] for f in fz]
    F = t[t.feasible]
    if len(F):
        return {"rule": "feasible: largest m_ttc", "m_ttc": float(F.m_ttc.max()), "n_feasible": int(len(F)),
                "feasible_m_ttc": sorted(float(x) for x in F.m_ttc)}
    c = t.sort_values(["n_failed", "shortfall", "m_ttc"], ascending=[True, True, False])
    return {"rule": "none feasible: fewest failed criteria, then smallest shortfall, then larger margin",
            "m_ttc": float(c.m_ttc.iloc[0]), "n_feasible": 0, "limitation": True}


def _short(r: dict) -> dict:
    return {"p": r["p"], "k": r["k"], "n": r["n"], "cp95": r["ci"], "boot_log95": r["boot_log"]}


def cmd_summarize(a):
    from navsim.agents.para_ssr.refiner import decoder as D
    for p in (POOL, LABELS_POOL, SHARDS):
        R._train_only(p)
    df = pd.read_parquet(Path(a.out) / "rows.parquet")
    err = df[df.error != ""]
    df = df[df.error == ""].copy()
    toks = pd.read_parquet(POOL)
    lab = pd.read_parquet(LABELS_POOL, columns=["token", "k", "ttc", "nc", "family"])
    df = df.merge(lab, on=["token", "k"], how="left", validate="one_to_one")
    assert df.ttc.notna().all()
    assert (df.fam == df.family).all(), "draft family != label family"
    # cross-check with the D1-b sweep shards (bank rows): official re-score ttc and collision gmin g_s_sat
    sh = pd.concat([pd.read_parquet(p, columns=["token", "k", "pool", "ttc", "g_s_sat", "error"])
                    for p in sorted(SHARDS.glob("part-*.parquet"))], ignore_index=True)
    sh = sh[(sh.error == "") & (sh.pool == "bank")].rename(columns={"ttc": "ttc_sweep"})
    x = df.merge(sh[["token", "k", "ttc_sweep", "g_s_sat"]], on=["token", "k"], how="left", validate="one_to_one")
    both_inf = np.isinf(x.col_g) & np.isinf(x.g_s_sat)
    dg = np.where(both_inf, 0.0, np.abs(x.col_g - x.g_s_sat))
    xcheck = {"rows_in_sweep_bank": int(x.ttc_sweep.notna().sum()), "rows": int(len(x)),
              "ttc label != sweep re-score": int((x.ttc_sweep.notna() & (x.ttc != x.ttc_sweep)).sum()),
              "max |col_g - sweep g_s_sat|": float(np.nanmax(dg)) if len(dg) else float("nan"),
              "col flag@0.15 disagreements": int(((x.col_g < M_COL_FLAG) != (x.g_s_sat < M_COL_FLAG))
                                                 [x.g_s_sat.notna()].sum())}
    fail = (df.ttc < 1).to_numpy()
    hum = (df.k == 0).to_numpy()
    Rj = {"created": time.strftime("%F %T"), "rule_file": str(REPORT_DIR / "SELECTION_RULE.txt"),
          "n_pool_tokens": int(len(toks)), "n_tokens": int(df.token.nunique()), "n_logs": int(df.log.nunique()),
          "n_error_tokens": int(err.token.nunique()), "errors": err.error.head(5).tolist(),
          "n_bank_drafts": int(len(df)), "n_human": int(hum.sum()), "n_perturbed": int((~hum).sum()),
          "official_ttc_fail_bank": int(fail.sum()), "official_ttc_fail_human": int((fail & hum).sum()),
          "official_ttc_fail_perturbed": int((fail & ~hum).sum()),
          "cross_check_vs_sweep": xcheck, "m_col_flag": M_COL_FLAG, "grid": list(M_TTCS),
          "thresholds": {"T1_min": T1_MIN, "T2_max": T2_MAX}}
    rows, per = [], {}
    fam_names = {int(k): D.FAMILY_NAME.get(int(k), str(k)) for k in sorted(df.fam.unique())}
    for m in M_TTCS:
        f = flags(df, m)
        s = t1_t2(df, f)
        fz = feasible(s["T1"]["p"], s["T2"]["p"])
        d = {"combo": {k: _short(v) for k, v in s.items()}, "feasible": fz}
        d["proj_only"] = {k: _short(v) for k, v in t1_t2(df, flags(df, m, "proj_only")).items()}
        d["diag_stopskip_combo"] = {k: _short(v) for k, v in
                                    t1_t2(df, flags(df, m, col="ttc_g_stop")).items() if k in ("T1", "T2")}
        d["diag_nfrom0_combo"] = {k: _short(v) for k, v in
                                  t1_t2(df, flags(df, m, col="ttc_g_n0")).items() if k in ("T1", "T2")}
        d["T1_by_family"] = {fam_names[fm]: _short(_stat(f[(df.fam == fm).to_numpy()],
                                                         fail[(df.fam == fm).to_numpy()],
                                                         df.log.to_numpy()[(df.fam == fm).to_numpy()]))
                             for fm in fam_names if (fail & (df.fam == fm).to_numpy()).any()}
        per[str(m)] = d
        rows.append(dict(m_ttc=m, T1=s["T1"]["p"], T1_k=s["T1"]["k"], T1_n=s["T1"]["n"],
                         T1_boot_lo=s["T1"]["boot_log"][0], T1_boot_hi=s["T1"]["boot_log"][1],
                         T1_cp_lo=s["T1"]["ci"][0], T1_cp_hi=s["T1"]["ci"][1],
                         T2=s["T2"]["p"], T2_k=s["T2"]["k"], T2_n=s["T2"]["n"],
                         T2_boot_lo=s["T2"]["boot_log"][0], T2_boot_hi=s["T2"]["boot_log"][1],
                         T2_cp_lo=s["T2"]["ci"][0], T2_cp_hi=s["T2"]["ci"][1],
                         FA_bank=s["FA_bank"]["p"], FA_bank_k=s["FA_bank"]["k"], FA_bank_n=s["FA_bank"]["n"],
                         T1_proj_only=d["proj_only"]["T1"]["p"], T2_proj_only=d["proj_only"]["T2"]["p"],
                         feasible=fz["all"], n_failed=fz["n_failed"], shortfall=fz["shortfall"]))
    col_only = t1_t2(df, flags(df, 0.0, "col_only"))
    Rj["col_flag_alone_m0.15"] = {k: _short(v) for k, v in col_only.items()}
    Rj["col_flag_alone_T1_by_family"] = {
        fam_names[fm]: _short(_stat(flags(df, 0.0, "col_only")[(df.fam == fm).to_numpy()],
                                    fail[(df.fam == fm).to_numpy()], df.log.to_numpy()[(df.fam == fm).to_numpy()]))
        for fm in fam_names if (fail & (df.fam == fm).to_numpy()).any()}
    table = pd.DataFrame(rows)
    sel = choose(table)
    Rj["selection"] = sel
    Rj["chosen"] = per[str(sel["m_ttc"])]
    Rj["per_m_ttc"] = per
    Rj["timing_s_per_token"] = float(df.sec.dropna().mean()) if "sec" in df else float("nan")
    rep = Path(a.report)
    rep.mkdir(parents=True, exist_ok=True)
    col_row = dict(m_ttc="col_only", T1=col_only["T1"]["p"], T1_k=col_only["T1"]["k"], T1_n=col_only["T1"]["n"],
                   T1_boot_lo=col_only["T1"]["boot_log"][0], T1_boot_hi=col_only["T1"]["boot_log"][1],
                   T1_cp_lo=col_only["T1"]["ci"][0], T1_cp_hi=col_only["T1"]["ci"][1],
                   T2=col_only["T2"]["p"], T2_k=col_only["T2"]["k"], T2_n=col_only["T2"]["n"],
                   T2_boot_lo=col_only["T2"]["boot_log"][0], T2_boot_hi=col_only["T2"]["boot_log"][1],
                   T2_cp_lo=col_only["T2"]["ci"][0], T2_cp_hi=col_only["T2"]["ci"][1],
                   FA_bank=col_only["FA_bank"]["p"], FA_bank_k=col_only["FA_bank"]["k"],
                   FA_bank_n=col_only["FA_bank"]["n"])
    pd.concat([table, pd.DataFrame([col_row])], ignore_index=True).to_csv(rep / "ttc_select.csv", index=False,
                                                                          float_format="%.4f")
    (rep / "ttc_select.json").write_text(json.dumps(Rj, indent=1, default=float))
    print(json.dumps({"selection": sel, "cross_check": xcheck, "n_error_tokens": Rj["n_error_tokens"]}, indent=1,
                     default=float))
    print(f"-> {rep}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("summarize")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--report", default=str(REPORT_DIR))
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) > 2:
        raise SystemExit("<= 2 workers (shared machine)")
    for p in (a.out, getattr(a, "report", "")):
        if "/runs/" in str(p) or "eval_dev" in str(p):
            raise SystemExit(f"refusing to write under {p}")
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
