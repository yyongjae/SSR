#!/usr/bin/env python
"""Evaluate CK selection / correction against plain v2 with official labels (report 44 §6-7; contract
pipeline.eval_ck).

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/eval_ck.py --run ckS_p1 --split navtrain_val \
    [--infer CK_DATA/infer/ckS_p1/navtrain_val] [--labels-corr CK_DATA/labels/navtrain_val/corr_ckS_p1] \
    [--fix-from CK_DATA/eval/ckS_p1/navtrain_val/metrics.json]

Variants (official per-trajectory labels, no new scoring here):
  v2        cand 0 (= the submitted v2 trajectory)
  oracle16  label-best of the 16 candidates (upper bound of selection)
  a         argmax blend(beta) = (1-beta) v2_final + beta ck_final(CK prob, v2 im) over the 16 originals
  b         a's candidate with the CK correction applied (corr labels)
  c         argmax over originals U corrected (corrected: own CK prob, the original's im / v2_final)
  oracle32  label-best of originals U corrected (if corr labels)
beta grid = constants.BETA_GRID on navtrain_val; the best beta per variant (mean PDMS) is recorded; with --fix-from
only those betas (+ beta 1) are reported (navtest).
Metrics: PDMS, NC/DAC/EP/TTC/C means, fail rates (nc, dac, ttc, ddc < 1), lead-decel NC|TTC failure rate
(has_lead == 1, D_1 == 1, censored excluded) with n, delta vs v2 with a paired log-cluster bootstrap 95% CI.
Token set: packed ok, CK inference done, all 16 cand labels ok (and all 16 corr labels ok when corr labels are
given) -- every variant on the same tokens.  navtest: v2 mean pdms over every cand-0-labelled token is checked
against the CSV (0.88137 +- 0.002 -> flag v2_mismatch) and per token against the CSV score.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

METRIC_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ----------------------------------------------------------------------------------------------- bootstrap
def log_bootstrap(diff: np.ndarray, logs: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired log-cluster bootstrap of mean(diff): resample logs with replacement (ratio estimator over tokens).
    -> dict(mean, lo, hi, n, n_logs)."""
    diff = np.asarray(diff, np.float64)
    logs = np.asarray(logs)
    m = np.isfinite(diff)
    diff, logs = diff[m], logs[m]
    n = len(diff)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_logs": 0}
    ul, inv = np.unique(logs, return_inverse=True)
    L = len(ul)
    S = np.bincount(inv, weights=diff, minlength=L)
    C = np.bincount(inv, minlength=L).astype(np.float64)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(np.float64)    # [B, L]
    num, den = W @ S, W @ C
    est = num / np.maximum(den, 1e-12)
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(n), "n_logs": int(L)}


# ----------------------------------------------------------------------------------------------- metrics
def row_metrics(lab: np.ndarray, lead_mask: Optional[np.ndarray]) -> Dict[str, float]:
    """lab [n, 9] (LABEL_COLS) of the chosen trajectories -> metric dict."""
    from navsim.agents.para_ssr.ck import constants as Cn
    L = Cn.LBL
    out = {"n": int(len(lab))}
    for c in METRIC_COLS:
        out[c] = float(np.mean(lab[:, L[c]])) if len(lab) else float("nan")
    for c in ("nc", "dac", "ttc", "ddc"):
        out[f"fail_{c}"] = float(np.mean(lab[:, L[c]] < 1)) if len(lab) else float("nan")
    fail = (lab[:, L["nc"]] < 1) | (lab[:, L["ttc"]] < 1)
    out["fail_nc_or_ttc"] = float(fail.mean()) if len(lab) else float("nan")
    if lead_mask is not None:
        out["lead_n"] = int(lead_mask.sum())
        out["lead_fail_nc_ttc"] = float(fail[lead_mask].mean()) if lead_mask.any() else float("nan")
    return out


def choose(v2_final, v2_im, prob, prob_corr, beta, have_corr: bool) -> Dict[str, np.ndarray]:
    from navsim.agents.para_ssr.ck.select import select_all
    sel = select_all(np.asarray(v2_final, np.float64), np.asarray(v2_im, np.float64), np.asarray(prob, np.float64),
                     np.asarray(prob_corr if prob_corr is not None else prob, np.float64), float(beta))
    out = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, np.int64) for k, v in sel.items()}
    if not have_corr:
        out.pop("b", None)
        out.pop("c", None)
    return out


def evaluate_arrays(lab: np.ndarray, lab_corr: Optional[np.ndarray], prob: np.ndarray, prob_corr: Optional[np.ndarray],
                    v2_final: np.ndarray, v2_im: np.ndarray, logs: np.ndarray, lead_mask: Optional[np.ndarray],
                    betas: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Dict:
    """Core of eval_ck on aligned arrays (tokens already filtered).  lab / lab_corr [n, K, 9] (LABEL_COLS),
    prob / prob_corr [n, K, 5] CK probabilities, v2_final / v2_im [n, K].  -> {'rows': [...], 'chosen': {...}}."""
    from navsim.agents.para_ssr.ck import constants as Cn
    P = Cn.LBL["pdms"]
    n, K = lab.shape[:2]
    ar = np.arange(n)
    have_corr = lab_corr is not None and prob_corr is not None
    base = lab[:, 0]
    rows, chosen_idx = [], {}

    def add(variant: str, beta, idx: np.ndarray, chosen_lab: np.ndarray):
        m = row_metrics(chosen_lab, lead_mask)
        m.update(variant=variant, beta=beta, frac_changed=float(np.mean(idx != 0)))
        if variant in ("c", "oracle32"):
            m["frac_from_corr"] = float(np.mean(idx >= K))
        if variant != "v2":
            d = chosen_lab[:, P] - base[:, P]
            m["d_pdms"] = log_bootstrap(d, logs, n_boot, seed)
            if lead_mask is not None and lead_mask.any():
                fb = ((base[:, Cn.LBL["nc"]] < 1) | (base[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
                fv = ((chosen_lab[:, Cn.LBL["nc"]] < 1) | (chosen_lab[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
                m["d_lead_fail"] = log_bootstrap((fv - fb)[lead_mask], logs[lead_mask], n_boot, seed)
        rows.append(m)
        chosen_idx[(variant, beta)] = idx

    add("v2", None, np.zeros(n, np.int64), base)
    o16 = lab[..., P].argmax(1)
    add("oracle16", None, o16, lab[ar, o16])
    if have_corr:
        pool = np.concatenate([lab, lab_corr], 1)
        o32 = pool[..., P].argmax(1)
        add("oracle32", None, o32, pool[ar, o32])
    for b in betas:
        sel = choose(v2_final, v2_im, prob, prob_corr, b, have_corr)
        ia = sel["a"]
        add("a", float(b), ia, lab[ar, ia])
        if have_corr:
            ib = sel["b"]
            add("b", float(b), ib, lab_corr[ar, ib])
            ic = sel["c"]
            pool = np.concatenate([lab, lab_corr], 1)
            add("c", float(b), ic, pool[ar, ic])
    best = {}
    for v in ("a", "b", "c"):
        cands = [r for r in rows if r["variant"] == v]
        if cands:
            best[v] = max(cands, key=lambda r: (r["pdms"], -abs(1.0 - r["beta"])))["beta"]
    return {"rows": rows, "chosen_idx": chosen_idx, "best_beta": best}


# ----------------------------------------------------------------------------------------------- io
def load_labels(d: Path, n: int):
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    if lab.shape[0] < n:
        pad = np.full((n - lab.shape[0],) + lab.shape[1:], np.nan, np.float32)
        lab = np.concatenate([np.asarray(lab), pad])
        ok = np.concatenate([np.asarray(ok), np.zeros((n - ok.shape[0],) + ok.shape[1:], bool)])
    return np.asarray(lab[:n], np.float64), np.asarray(ok[:n], bool)


def lead_flags(split: str, tokens: Sequence[str]):
    p = U.ck_data() / "lead" / f"{split}.parquet"
    if not p.is_file():
        return None, None
    df = pd.read_parquet(p).drop_duplicates("token").set_index("token")
    t = pd.Index(tokens)
    has = df["has_lead"].reindex(t).to_numpy(np.float64)
    d1 = df["D_1"].reindex(t).to_numpy(np.float64)
    m = (has == 1) & (d1 == 1)
    if "censored" in df.columns:
        cen = df["censored"].reindex(t).fillna(True).astype(bool).to_numpy()
        m &= ~cen
    return m, str(p)


def csv_check(tokens: Sequence[str], pdms0: np.ndarray, ok0: np.ndarray) -> Dict:
    from navsim.agents.para_ssr.ck import constants as Cn
    out = {"csv": str(Cn.V2_NAVTEST_CSV), "csv_pdms": Cn.V2_NAVTEST_PDMS, "tol": Cn.V2_NAVTEST_TOL}
    mean_all = float(np.mean(pdms0[ok0])) if ok0.any() else float("nan")
    out.update(v2_mean_pdms=mean_all, n=int(ok0.sum()), diff=mean_all - Cn.V2_NAVTEST_PDMS)
    out["v2_mismatch"] = bool(not (abs(out["diff"]) <= Cn.V2_NAVTEST_TOL)) or int(ok0.sum()) < 12146
    try:
        csv = pd.read_csv(Cn.V2_NAVTEST_CSV)
        csv = csv[csv.token != "average"].set_index("token")
        s = csv["score"].reindex(pd.Index(tokens)).to_numpy(np.float64)
        m = ok0 & np.isfinite(s)
        dd = np.abs(pdms0[m] - s[m])
        out.update(per_token_n=int(m.sum()), per_token_mae=float(dd.mean()) if m.any() else None,
                   per_token_max=float(dd.max()) if m.any() else None,
                   per_token_frac_gt_1e3=float((dd > 1e-3).mean()) if m.any() else None,
                   csv_mean_same_tokens=float(s[m].mean()) if m.any() else None)
    except Exception as e:     # noqa: BLE001
        out["per_token_error"] = f"{type(e).__name__}: {e}"
    if out.get("per_token_n", 0) < 12146:
        out["note"] = "partial navtest (smoke / missing rows): v2_mismatch flag judged on the full split only"
    return out


def fmt_ci(d):
    if not isinstance(d, dict) or d.get("n", 0) == 0:
        return ""
    return f"{100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


def table_md(res: Dict, split: str, run: str, info: Dict) -> str:
    L = [f"# CK eval: {run} / {split}", "",
         f"토큰 {info['n_eval']} / 전체 {info['n_total']} (packed ok, 추론 완료, 후보 라벨 16개 모두 ok"
         f"{', 교정본 라벨 16개 ok' if info['have_corr'] else ''}). log {info['n_logs']}. bootstrap {info['n_boot']}회"
         f"(log 군집). PDMS는 0-100 단위.", ""]
    if info.get("v2_check"):
        c = info["v2_check"]
        L += [f"v2 점검: 평균 pdms {100 * c['v2_mean_pdms']:.2f} vs CSV {100 * c['csv_pdms']:.3f} "
              f"(차 {100 * c['diff']:+.3f}, n {c['n']}), 토큰별 MAE {c.get('per_token_mae')}, "
              f"v2_mismatch={c['v2_mismatch']}", ""]
    L += ["| 변형 | β | PDMS | ΔPDMS [95% CI] | NC | DAC | EP | TTC | C | NC 실패 | DAC 실패 | TTC 실패 | "
          "앞차 감속 NC/TTC 실패 (n) | Δ앞차 실패 [95% CI] | 바뀐 비율 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        b = "" if r["beta"] is None else f"{r['beta']:g}"
        lead = (f"{100 * r['lead_fail_nc_ttc']:.2f} ({r['lead_n']})" if "lead_fail_nc_ttc" in r
                and r.get("lead_n") else "-")
        L.append(f"| {r['variant']} | {b} | {100 * r['pdms']:.2f} | {fmt_ci(r.get('d_pdms'))} | "
                 f"{100 * r['nc']:.2f} | {100 * r['dac']:.2f} | {100 * r['ep']:.2f} | {100 * r['ttc']:.2f} | "
                 f"{100 * r['comfort']:.2f} | {100 * r['fail_nc']:.2f} | {100 * r['fail_dac']:.2f} | "
                 f"{100 * r['fail_ttc']:.2f} | {lead} | {fmt_ci(r.get('d_lead_fail'))} | "
                 f"{100 * r['frac_changed']:.1f} |")
    L += ["", f"검증 로그에서 고른 β: {res.get('best_beta')}" + (f" (고정 출처 {info['fix_from']})"
                                                           if info.get("fix_from") else "")]
    return "\n".join(L) + "\n"


def main(argv=None):
    from navsim.agents.para_ssr.ck import constants as Cn

    ap = argparse.ArgumentParser(description="CK evaluation vs v2", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--run", required=True, help="run name (labels / output naming)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--infer", default=None, help="default CK_DATA/infer/<run>/<split>")
    ap.add_argument("--labels-corr", default=None, help="default CK_DATA/labels/<split>/corr_<run> if present; 'none'")
    ap.add_argument("--beta-grid", default=",".join(f"{b:g}" for b in Cn.BETA_GRID))
    ap.add_argument("--fix-from", default=None, help="metrics.json of the navtrain_val eval (its best betas)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default CK_DATA/eval/<run>/<split>")
    ap.add_argument("--no-per-token", action="store_true")
    a = ap.parse_args(argv)
    root = U.ck_data()
    infer = Path(a.infer) if a.infer else root / "infer" / a.run / a.split
    out = Path(a.out) if a.out else root / "eval" / a.run / a.split
    pdir = root / "packed" / a.split
    tdf = pd.read_parquet(pdir / "tokens.parquet")
    N = len(tdf)
    pok = np.load(pdir / "ok.npy") if (pdir / "ok.npy").is_file() else np.ones(N, bool)
    v2_final = np.load(pdir / "v2_final.npy", mmap_mode="r")
    v2_im = np.load(pdir / "v2_im.npy", mmap_mode="r")
    lab, lok = load_labels(root / "labels" / a.split / "cand", N)
    itok = pd.read_parquet(infer / "tokens.parquet")
    irow = itok["row"].to_numpy(np.int64) if "row" in itok.columns else np.arange(len(itok))
    idone = np.load(infer / "done.npy")
    s_log = np.load(infer / "score_logit.npy", mmap_mode="r")
    c_log = np.load(infer / "corr_score_logit.npy", mmap_mode="r") if (infer / "corr_score_logit.npy").is_file() \
        else None
    # map infer rows -> packed rows
    rows = irow[idone]
    ipos = np.flatnonzero(idone)
    lc_dir = None
    if a.labels_corr != "none":
        lc_dir = Path(a.labels_corr) if a.labels_corr else root / "labels" / a.split / f"corr_{a.run}"
        if not (lc_dir / "labels.npy").is_file():
            print(f"[eval] no corr labels at {lc_dir}: variants b / c / oracle32 skipped", flush=True)
            lc_dir = None
    have_corr = lc_dir is not None and c_log is not None
    lab_c, lok_c = load_labels(lc_dir, N) if have_corr else (None, None)
    keep = pok[rows] & lok[rows].all(1)
    if have_corr:
        keep &= lok_c[rows].all(1)
    rows, ipos = rows[keep], ipos[keep]
    tokens = tdf.token.to_numpy()[rows]
    logs = tdf.log.to_numpy()[rows]
    lead_mask, lead_src = lead_flags(a.split, tokens)
    prob = sigmoid(np.asarray(s_log[ipos], np.float32))
    prob_c = sigmoid(np.asarray(c_log[ipos], np.float32)) if have_corr else None
    betas = [float(x) for x in a.beta_grid.split(",") if x.strip()]
    fixed = None
    if a.fix_from:
        fixed = (U.read_json(a.fix_from) or {}).get("best_beta") or {}
        betas = sorted(set([float(v) for v in fixed.values()] + [1.0]))
    res = evaluate_arrays(lab[rows], lab_c[rows] if have_corr else None, prob, prob_c, np.asarray(v2_final[rows]),
                          np.asarray(v2_im[rows]), logs, lead_mask, betas, a.bootstrap, a.seed)
    if fixed is not None:
        res["best_beta"] = {k: float(v) for k, v in fixed.items()}
    info = dict(run=a.run, split=a.split, infer=str(infer), labels_cand=str(root / "labels" / a.split / "cand"),
                labels_corr=str(lc_dir) if lc_dir else None, have_corr=have_corr, n_total=int(N),
                n_infer_done=int(idone.sum()), n_eval=int(len(rows)), n_logs=int(len(np.unique(logs))),
                n_boot=a.bootstrap, betas=betas, fix_from=a.fix_from, lead_src=lead_src,
                lead_n=int(lead_mask.sum()) if lead_mask is not None else None, infer_meta=U.read_json(infer / "meta.json"),
                created=U.now())
    if a.split == "navtest":
        ok0 = pok & lok[:, 0]
        info["v2_check"] = csv_check(tdf.token.to_numpy(), lab[:, 0, Cn.LBL["pdms"]], ok0)
    metrics = dict(info, best_beta=res["best_beta"], rows=res["rows"],
                   v2_mismatch=bool(info.get("v2_check", {}).get("v2_mismatch", False)))
    out.mkdir(parents=True, exist_ok=True)
    U.write_json(out / "metrics.json", metrics)
    if not a.no_per_token:
        recs = []
        for (variant, beta), idx in res["chosen_idx"].items():
            if variant in ("b",):
                L_ = lab_c[rows][np.arange(len(rows)), idx]
            elif variant in ("c", "oracle32"):
                L_ = np.concatenate([lab[rows], lab_c[rows]], 1)[np.arange(len(rows)), idx]
            else:
                L_ = lab[rows][np.arange(len(rows)), idx]
            df = pd.DataFrame({"token": tokens, "log": logs, "variant": variant,
                               "beta": np.nan if beta is None else beta, "chosen": idx.astype(np.int16)})
            for c in Cn.LABEL_COLS:
                df[c] = L_[:, Cn.LBL[c]].astype(np.float32)
            recs.append(df)
        pt = pd.concat(recs, ignore_index=True)
        tmp = out / ".per_token.tmp.parquet"
        pt.to_parquet(tmp, index=False)
        tmp.replace(out / "per_token.parquet")
    (out / "table.md").write_text(table_md(res, a.split, a.run, info))
    print((out / "table.md").read_text(), flush=True)
    print(f"[eval] -> {out}", flush=True)


if __name__ == "__main__":
    main()
