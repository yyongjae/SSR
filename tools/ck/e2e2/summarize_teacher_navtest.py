#!/usr/bin/env python
"""Navtest summary of the CK2 teachers (ck2T DET / ck2M MAP, ckpt_last): GT-free planner, score quality, reference A
(v2 r34 top-16; user 2026-10-08 "참고용 평가도 2,3에서 같이 진행해"), old CK teachers, lead-decel slice.

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
    /venv/ssr/bin/python tools/ck/e2e2/summarize_teacher_navtest.py

Reads saved outputs only (no model, no GPU).  Every candidate choice was made earlier without GT / labels:
  eval_teacher_navtest_gtfree.py  -> <D>/<t>/gtfree/{p256_idx, g2_top16, g2_valid96, p96_col, final_traj, g1_prob, g2_prob96}
  eval_teacher_navtest_r34.py     -> <D>/<t>/r34/{picks.npz, score_logit16/80, final_traj, score_stack}
The official labels (tools/ck/data/label_cands.py) are read here for the metrics only.  The single choice made in this
script is the context row "old CK-only a" (argmax of the CK-only score over the old teacher's saved probabilities).

GT-free planner rows (labels of the submitted trajectory = <D>/labels/ck2eval_gtfree_<t>_final5, columns
  P256, P256_lat, P96_off, P96_on, P96_onexl):
  P256           argmax of the CK-only score over the 256 raw anchors
  P256_lat       P256 + lateral head (z_lon = 0)
  P96            (lateral none = P96_off) argmax over the 96 pool (own top-16 + 5 variants, invalid lateral excluded)
  P96_lat_always (P96_on), P96_lat_skiplatvar (P96_onexl: no lateral when the pick is l-0.5 / l+0.5)
  oracle256 / oracle_top16 / oracle96  label-best of the 256 anchors / the own top-16 / the valid 96 pool
Reference A rows (recomputed from picks.npz + labels; equal to <t>/r34/metrics.json, checked): v2, oracle16, oracle96,
  r34_{ck_noim, old_b1, old_b0.5}_{a,b}, pool96_{ck_noim, old_b1}_{a,b}, v2_lat; old teachers (ckT_p1 / ckM_p1
  per_token.parquet): a(1), b(1) and the context row old CK-only a.
Per row: PDMS (+ 95% CI), NC / DAC / EP / TTC / C / DDC means, fail rates, delta vs v2 r34 per token (paired) with the
  log-clustered bootstrap (2000, seed 0, tools/ck/eval_ck.log_bootstrap), better / worse token fractions, lead-decel
  slice (CK_DATA/lead/navtest.parquet: has_lead & D_1 & not censored): PDMS, NC|TTC failure rate, its delta vs v2.
EP target of the score-quality metrics (report 48 task 5c): every prediction-vs-label metric of the CK EP head (EP MAE
  / corr / BCE, the EP pair accuracy, the within-token EP Pearson) compares against ep_target.ck_targets(labels, the
  teacher run's ep_target) -- config.json of the run that holds the evaluated checkpoint ('official' when the key is
  missing, the old CK1 teachers are 'official'), so a decoupled-EP teacher is scored against the decoupled EP.  The
  official PDMS and the official sub-scores of the submitted trajectories (rows, per_token) are unchanged.
Score quality (train_ck2.key_metrics: fail AUC per key nc / dac / ttc / comfort, pass AUC, EP MAE / corr, BCE) on
  the 256 raw anchors, the own top-16, the valid GT-free variants (per type too), the r34 top-16, the r34 variants and
  (old teacher) the r34 top-16; parent-vs-variant pair accuracy per key and PDMS (train_ck2.raw_metrics definition:
  sign(dp) == sign(dy) on pairs with |dy| > 1e-6; PDMS uses the CK-only score); within-token Spearman of the CK-only
  score vs label PDMS.
Outputs <D> = CK_DATA/ck2/eval/navtest: metrics.json, table.md, per_token.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import EP_KEY_IDX, ck_targets, run_ep_target  # noqa: E402
from navsim.agents.para_ssr.ck.select import ck_final  # noqa: E402
from navsim.agents.para_ssr.ck.select2 import VNAMES  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.eval_ck import csv_check, lead_flags, log_bootstrap, row_metrics  # noqa: E402
from tools.ck.train_ck2 import key_metrics  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
D = CK_DATA / "ck2" / "eval" / "navtest"
SPLIT = "navtest"
TEACHERS = {"ck2T": "ckT_p1", "ck2M": "ckM_p1"}
K16, NV, NVAR = 16, 6, 5
P = Cn.LBL["pdms"]
CKI = list(Cn.CK_LABEL_IDX)
W_NOIM = (0.0,) + tuple(Cn.SEL_W[1:])
KST = 9 * 3600
GTFREE_MODES = (("P256", 0), ("P256_lat", 1), ("P96", 2), ("P96_lat_always", 3), ("P96_lat_skiplatvar", 4))
GTFREE_LABEL = {"P256": "P256 (256 anchor argmax)", "P256_lat": "P256 + lateral",
                "P96": "P96 (lateral 없음 = P96_off)", "P96_lat_always": "P96 + lateral 항상 (P96_on)",
                "P96_lat_skiplatvar": "P96 + lateral, l± 변형이면 생략 (P96_onexl)",
                "oracle_top16": "oracle own top-16", "oracle96": "oracle 96 pool", "oracle256": "oracle 256 anchors"}
R34_MODES = ("r34_ck_noim", "r34_old_b1", "r34_old_b0.5", "pool96_ck_noim", "pool96_old_b1")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def sel_score(prob) -> np.ndarray:
    p = np.asarray(prob, np.float64)
    s = np.asarray(ck_final(p, np.ones(p.shape[:-1]), W_NOIM), np.float64)
    return np.where(np.isfinite(s), s, -np.inf)


def to_pool(x16: np.ndarray, x80: np.ndarray) -> np.ndarray:
    n, tail = x16.shape[0], x16.shape[2:]
    out = np.empty((n, K16, NV) + tail, np.result_type(x16, x80))
    out[:, :, 0] = x16
    out[:, :, 1:] = x80.reshape((n, K16, NVAR) + tail)
    return out.reshape((n, K16 * NV) + tail)


def load_lab(d: Path, k: int):
    lab = np.load(d / "labels.npy")
    ok = np.load(d / "ok.npy").astype(bool)
    assert lab.shape[1] == k and lab.shape[2] == len(Cn.LABEL_COLS), (d, lab.shape)
    meta = U.read_json(d / "meta.json") or {}
    src = {"dir": str(d.resolve()), "k": k, "assembled": meta.get("assembled"), "traj_path": meta.get("traj_path"),
           "traj_sha16": meta.get("traj_sha16"), "n_err_tokens": meta.get("n_err_tokens"),
           "n_rec_not_ok": meta.get("n_rec_not_ok"),
           "scorer_sha16": (next(iter((meta.get("scorer_sha256") or {}).values()), "") or "")[:16]}
    return lab.astype(np.float64), ok, src


def teacher_ep_target(ckpt) -> str:
    """EP target of the teacher run holding `ckpt` (its config.json; a run written before --ep-target = 'official')."""
    cj = Path(ckpt).parent / "config.json"
    if not cj.is_file():
        raise SystemExit(f"{cj} missing: cannot tell the EP target of the teacher checkpoint {ckpt}")
    return run_ep_target(json.loads(cj.read_text()))


def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, (a * b).sum(1) / np.where(den > 1e-12, den, 1.0), np.nan)


def within_token(prob: np.ndarray, lab: np.ndarray, pick: Optional[np.ndarray] = None,
                 ep_target: str = "official") -> Dict:
    """Per-token ranking quality over a candidate set: Spearman(CK-only score, official label PDMS), Pearson(p_ep, y_ep)
    with y_ep = the EP target of the teacher (ck_targets(lab, ep_target); 'official' = the label 'ep' column)."""
    from scipy.stats import rankdata
    s = sel_score(prob)
    y = lab[..., P]
    sp = rowcorr(rankdata(s, axis=1), rankdata(y, axis=1))
    ep = rowcorr(np.asarray(prob[..., Cn.CK_KEYS.index("ep")], np.float64),
                 np.asarray(ck_targets(lab, ep_target)[..., EP_KEY_IDX], np.float64))
    out = {"spearman_sel_pdms_mean": float(np.nanmean(sp)), "spearman_n_tokens": int(np.isfinite(sp).sum()),
           "ep_pearson_within_mean": float(np.nanmean(ep)), "ep_pearson_n_tokens": int(np.isfinite(ep).sum())}
    if pick is not None:
        ch = y[np.arange(len(y)), pick]
        out["top1_is_oracle_frac"] = float((ch >= y.max(1) - 1e-9).mean())
    return out


def pair_metrics(prob96: np.ndarray, lab96: np.ndarray, valid96: np.ndarray, ep_target: str = "official") -> Dict:
    """Parent (identity column k * 6) vs its valid variants: sign agreement of dp / dy per key (dy of the CK targets =
    ck_targets(lab, ep_target): the EP key against the teacher's EP target) and of the CK-only score difference vs the
    official PDMS difference (pairs with |dy| > 1e-6), overall and per variant type."""
    n = prob96.shape[0]
    c = np.arange(K16 * NV)
    parent = (c // NV) * NV
    vt = c % NV
    pair = valid96 & (vt != 0)[None]
    Pp, Yp = prob96[:, parent], lab96[:, parent]
    Y5 = np.asarray(ck_targets(lab96, ep_target), np.float64)        # [n, 96, 5] CK targets (CK_KEYS order)
    Y5p = Y5[:, parent]
    out: Dict = {"pair_n": int(pair.sum()), "per_type": {}, "ep_target": ep_target}
    sel = sel_score(prob96)
    dsel = sel - sel[:, parent]

    def acc(dp, dy, m):
        m = m & (np.abs(dy) > 1e-6)
        return (float((np.sign(dp[m]) == np.sign(dy[m])).mean()) if m.any() else float("nan")), int(m.sum())

    for j, k in enumerate(Cn.CK_KEYS):
        out[f"pair_acc_{k}"], out[f"pair_n_{k}"] = acc(prob96[..., j] - Pp[..., j], Y5[..., j] - Y5p[..., j], pair)
    dy = lab96[..., P] - Yp[..., P]
    out["pair_acc_pdms"], out["pair_n_pdms"] = acc(dsel, dy, pair)
    for v in range(1, NV):
        m = pair & (vt == v)[None]
        d = {"n": int(m.sum())}
        for j, k in enumerate(Cn.CK_KEYS):
            d[f"acc_{k}"], d[f"n_{k}"] = acc(prob96[..., j] - Pp[..., j], Y5[..., j] - Y5p[..., j], m)
        d["acc_pdms"], d["n_pdms"] = acc(dsel, dy, m)
        d["frac_var_better_pdms"] = float((dy[m] > 1e-6).mean()) if m.any() else float("nan")
        d["frac_var_worse_pdms"] = float((dy[m] < -1e-6).mean()) if m.any() else float("nan")
        out["per_type"][VNAMES[v]] = d
    return out


def km(prob: np.ndarray, lab: np.ndarray, mask: Optional[np.ndarray] = None, ep_target: str = "official") -> Dict:
    """train_ck2.key_metrics against the CK targets ck_targets(lab, ep_target) ('official' = lab[..., CK_LABEL_IDX]
    exactly; 'decoupled' replaces only the EP column)."""
    p = np.asarray(prob, np.float64)
    y = np.asarray(ck_targets(lab, ep_target), np.float64)
    if mask is not None:
        p, y = p[mask], y[mask]
    p, y = p.reshape(-1, 5), y.reshape(-1, 5)
    return key_metrics(p, y, np.ones(len(p), bool))


class Rows:
    """Metric rows of one section on the common token set (paired with v2 r34 cand 0)."""

    def __init__(self, base: np.ndarray, logs: np.ndarray, lead: Optional[np.ndarray], n_boot: int):
        self.base, self.logs, self.lead, self.n_boot = base, logs, lead, n_boot
        self.fb = ((base[:, Cn.LBL["nc"]] < 1) | (base[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)

    def row(self, mode: str, lab: np.ndarray, paired: bool = True, extra: Optional[Dict] = None) -> Dict:
        m = row_metrics(lab, self.lead)
        m = {"mode": mode, **m}
        pdms = lab[:, P]
        m["pdms_ci"] = log_bootstrap(pdms, self.logs, self.n_boot, 0)
        if self.lead is not None and self.lead.any():
            m["lead_pdms"] = float(pdms[self.lead].mean())
        if paired:
            d = pdms - self.base[:, P]
            m["d_pdms"] = log_bootstrap(d, self.logs, self.n_boot, 0)
            m["better_vs_v2"] = float((d > 1e-9).mean())
            m["worse_vs_v2"] = float((d < -1e-9).mean())
            if self.lead is not None and self.lead.any():
                fv = ((lab[:, Cn.LBL["nc"]] < 1) | (lab[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
                m["d_lead_fail"] = log_bootstrap((fv - self.fb)[self.lead], self.logs[self.lead], self.n_boot, 0)
        if extra:
            m.update(extra)
        return m

    def diff(self, name: str, a: np.ndarray, b: np.ndarray, desc: str = "") -> Dict:
        d = a[:, P] - b[:, P]
        out = {"name": name, "desc": desc, "d_pdms": log_bootstrap(d, self.logs, self.n_boot, 0),
               "a_better": float((d > 1e-9).mean()), "a_worse": float((d < -1e-9).mean())}
        if self.lead is not None and self.lead.any():
            fa = ((a[:, Cn.LBL["nc"]] < 1) | (a[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
            fb = ((b[:, Cn.LBL["nc"]] < 1) | (b[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
            out["d_lead_fail"] = log_bootstrap((fa - fb)[self.lead], self.logs[self.lead], self.n_boot, 0)
        return out


# ----------------------------------------------------------------------------------------------- table helpers
def f2(x, s=100.0, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{s * x:.{nd}f}"


def ci(d, signed=True):
    if not isinstance(d, dict) or d.get("n", 0) == 0:
        return ""
    fmt = "+.2f" if signed else ".2f"
    return f"{100 * d['mean']:{fmt}} [{100 * d['lo']:{fmt}}, {100 * d['hi']:{fmt}}]"


def a3(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.3f}"


ROW_HDR = ("| teacher | 모드 | PDMS [95% CI] | ΔPDMS vs v2 [95% CI] | 나음/나쁨 % | NC | DAC | EP | TTC | C | NC 실패 | "
           "DAC 실패 | TTC 실패 | 앞차감속 PDMS | 앞차감속 NC/TTC 실패 (n) | Δ앞차 실패 [95% CI] |")
ROW_SEP = "|" + "---|" * 16


def row_line(t: str, label: str, m: Dict) -> str:
    bw = (f"{100 * m['better_vs_v2']:.1f} / {100 * m['worse_vs_v2']:.1f}" if "better_vs_v2" in m else "")
    lead = f"{f2(m.get('lead_fail_nc_ttc'))} ({m.get('lead_n')})" if m.get("lead_n") else "-"
    return (f"| {t} | {label} | {ci(m['pdms_ci'], False)} | {ci(m.get('d_pdms'))} | {bw} | {f2(m['nc'])} | "
            f"{f2(m['dac'])} | {f2(m['ep'])} | {f2(m['ttc'])} | {f2(m['comfort'])} | {f2(m['fail_nc'])} | "
            f"{f2(m['fail_dac'])} | {f2(m['fail_ttc'])} | {f2(m.get('lead_pdms'))} | {lead} | "
            f"{ci(m.get('d_lead_fail'))} |")


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(D))
    ap.add_argument("--no-per-token", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not str(out.resolve()).startswith(str(D.resolve())):
        raise SystemExit(f"--out must be under {D}")
    t0 = time.time()
    print(f"[{kst()}] summary start", flush=True)

    pdir = CK_DATA / "packed" / SPLIT
    tdf = pd.read_parquet(pdir / "tokens.parquet")
    N = len(tdf)
    pok = np.load(pdir / "ok.npy").astype(bool)
    tok_all, log_all = tdf.token.to_numpy(), tdf.log.to_numpy()

    # ---------------------------------------------------------------- labels
    src = {}
    lab256, ok256, src["raw256"] = load_lab(CK_DATA / "labels" / SPLIT / "raw256", 256)
    labc, okc, src["cand"] = load_lab(CK_DATA / "labels" / SPLIT / "cand", K16)
    labvr, okvr, src["r34_var80"] = load_lab(D / "labels" / "ck2eval_r34_var80", K16 * NVAR)
    valid80 = np.load(D / "r34_variants" / "valid80.npy").astype(bool)
    L = {}
    for t in TEACHERS:
        L[t] = {}
        L[t]["var80"] = load_lab(D / "labels" / f"ck2eval_gtfree_{t}_var80", K16 * NVAR)
        L[t]["final5"] = load_lab(D / "labels" / f"ck2eval_gtfree_{t}_final5", 5)
        sel = U.read_json(D / t / "r34" / "select.json")
        L[t]["stack"] = load_lab(D / "labels" / f"ck2eval_{t}_r34_stack", len(sel["stack_names"]))
        L[t]["select"] = sel
        for k in ("var80", "final5", "stack"):
            src[f"{t}_{k}"] = L[t][k][2]

    # ---------------------------------------------------------------- token set (common to both teachers)
    keep = pok & ok256.all(1) & okc.all(1) & (okvr | ~valid80).all(1)
    G = {}
    for t in TEACHERS:
        g = D / t / "gtfree"
        G[t] = {k: np.load(g / f"{k}.npy") for k in ("done", "bev_ok", "p256_idx", "p96_col", "g2_top16",
                                                       "g2_valid96")}
        assert (pd.read_parquet(g / "tokens.parquet").token.to_numpy() == tok_all).all()
        assert (pd.read_parquet(D / t / "r34" / "tokens.parquet").token.to_numpy() == tok_all).all()
        vvar = G[t]["g2_valid96"].reshape(N, K16, NV)[:, :, 1:].reshape(N, K16 * NVAR)
        rdone = np.load(D / t / "r34" / "done.npy").astype(bool) & np.load(D / t / "r34" / "bev_ok.npy").astype(bool)
        keep &= G[t]["done"].astype(bool) & G[t]["bev_ok"].astype(bool) & rdone
        keep &= (L[t]["var80"][1] | ~vvar).all(1) & L[t]["final5"][1].all(1) & L[t]["stack"][1].all(1)
    r = np.flatnonzero(keep)
    n = len(r)
    ar = np.arange(n)
    tokens, logs = tok_all[r], log_all[r]
    lead, lead_src = lead_flags(SPLIT, tokens)
    LC = labc[r]
    base = LC[:, 0]
    R = Rows(base, logs, lead, a.bootstrap)
    v2_check = csv_check(tok_all, labc[:, 0, P], pok & okc[:, 0])
    print(f"[{kst()}] tokens {n}/{N}, logs {len(np.unique(logs))}, lead {None if lead is None else int(lead.sum())}",
          flush=True)

    M: Dict = dict(created=kst(), script=str(Path(__file__).resolve()), split=SPLIT, n_total=N, n_eval=n,
                   n_logs=int(len(np.unique(logs))), n_boot=a.bootstrap, bootstrap="tools/ck/eval_ck.log_bootstrap "
                   "(paired, log-clustered ratio estimator, seed 0)",
                   token_set="packed ok & both teachers: gtfree done & bev_ok, r34 done & bev_ok; raw256 / cand / "
                             "final5 / stack labels ok; var80 labels ok on every valid column",
                   units="fractions 0-1 in json (table: x100)", lead_src=lead_src,
                   lead_def="has_lead == 1 & D_1 == 1 & not censored (tools/ck/eval_ck.lead_flags)",
                   lead_n=None if lead is None else int(lead.sum()), v2_check=v2_check, label_sources=src,
                   selection_w_noim=W_NOIM,
                   gt_free=("candidate choices were made by eval_teacher_navtest_gtfree.py / eval_teacher_navtest_r34.py "
                            "from CK probabilities (+ v2 im / final for the old-formula rows) only; labels used here "
                            "for metrics only; independent re-check: tools/ck/e2e2/check_teacher_navtest_headline.py"),
                   teachers={}, cross_teacher=[], headline_pdms={})
    per_tok: List[pd.DataFrame] = []
    chosen: Dict = {}

    def keep_pt(t, section, mode, lab, idx=None):
        chosen[(t, section, mode)] = lab
        if a.no_per_token:
            return
        df = pd.DataFrame({"token": tokens, "log": logs, "teacher": t, "section": section, "mode": mode,
                           "chosen": (np.full(n, -1) if idx is None else idx).astype(np.int16)})
        for c in Cn.LABEL_COLS:
            df[c] = lab[:, Cn.LBL[c]].astype(np.float32)
        per_tok.append(df)

    for t, old in TEACHERS.items():
        tm: Dict = {"old_teacher": old}
        gdir, rdir = D / t / "gtfree", D / t / "r34"
        gmeta = U.read_json(gdir / "meta.json")
        tm["ckpt"], tm["ckpt_sha16"] = gmeta["ckpt"], gmeta["ckpt_sha16"]
        ept = tm["ep_target"] = teacher_ep_target(gmeta["ckpt"])        # EP of the score-quality metrics
        imeta = U.read_json(rdir / "infer_meta.json")
        tm["r34_ckpt_sha16"] = imeta["ckpt_sha16"][:16]
        chk: Dict = {"r34_ckpt_same_as_gtfree": tm["r34_ckpt_sha16"] == tm["ckpt_sha16"]}

        # ============================================================ GT-free planner
        p256 = G[t]["p256_idx"][r].astype(np.int64)
        col = G[t]["p96_col"][r].astype(np.int64)
        top16 = G[t]["g2_top16"][r].astype(np.int64)
        valid96 = G[t]["g2_valid96"][r].astype(bool)
        L256 = lab256[r]
        LV = L[t]["var80"][0][r]
        LF = L[t]["final5"][0][r]
        LP96 = to_pool(L256[ar[:, None], top16], LV)
        allowed96 = valid96 | (np.arange(K16 * NV) % NV == 0)[None]
        is_latvar = np.isin(col % NV, (4, 5))
        chk["P256_final5_eq_raw256_pick"] = int((LF[:, 0] == L256[ar, p256]).all(1).sum())
        chk["P96off_final5_eq_pool_pick"] = int((LF[:, 2] == LP96[ar, col]).all(1).sum())
        chk["P96onexl_eq_rule"] = int((LF[:, 4] == np.where(is_latvar[:, None], LF[:, 2], LF[:, 3])).all(1).sum())
        chk["p96_col_allowed"] = int(allowed96[ar, col].sum())
        chk["n"] = n
        rows = [R.row("v2", base, paired=False)]
        keep_pt(t, "gtfree", "v2", base, np.zeros(n, np.int64))
        for mode, j in GTFREE_MODES:
            ex = {"final5_col": j}
            if mode.startswith("P96"):
                ex.update(frac_identity=float((col % NV == 0).mean()), frac_same_as_P256=float((col == 0).mean()),
                          frac_latvar=float(is_latvar.mean()),
                          share={VNAMES[v]: float((col % NV == v).mean()) for v in range(NV)})
            rows.append(R.row(mode, LF[:, j], extra=ex))
            keep_pt(t, "gtfree", mode, LF[:, j], p256 if mode.startswith("P256") else col)
        o16 = L256[ar[:, None], top16][..., P].argmax(1)
        oo = {"oracle_top16": (L256[ar, top16[ar, o16]], top16[ar, o16]),
              "oracle96": None, "oracle256": None}
        o96 = np.where(allowed96, LP96[..., P], -np.inf).argmax(1)
        oo["oracle96"] = (LP96[ar, o96], o96)
        o256 = L256[..., P].argmax(1)
        oo["oracle256"] = (L256[ar, o256], o256)
        for mode, (lab, idx) in oo.items():
            rows.append(R.row(mode, lab))
            keep_pt(t, "gtfree", mode, lab, idx)
        ft = np.load(gdir / "final_traj.npy", mmap_mode="r")
        ftr = np.asarray(ft[r], np.float64)
        dev = lambda i, j: np.linalg.norm(ftr[:, i, :, :2] - ftr[:, j, :, :2], axis=-1).max(1)  # noqa: E731
        d256, d96 = dev(1, 0), dev(3, 2)
        tm["gtfree"] = {"rows": rows, "counts": gmeta.get("counts"), "inference": gmeta.get("passes"),
                        "lat_dev": {"P256_lat_vs_P256_maxdev_m_mean": float(d256.mean()),
                                    "P256_lat_frac_maxdev_gt_0p1m": float((d256 > 0.1).mean()),
                                    "P96_on_vs_off_maxdev_m_mean": float(d96.mean()),
                                    "P96_on_frac_maxdev_gt_0p1m": float((d96 > 0.1).mean())}}

        # ============================================================ score quality
        g1p = np.load(gdir / "g1_prob.npy", mmap_mode="r")
        g1p = np.asarray(g1p[r], np.float32)
        p96 = np.asarray(np.load(gdir / "g2_prob96.npy", mmap_mode="r")[r], np.float32)
        vt = np.arange(K16 * NV) % NV
        varmask = valid96 & (vt != 0)[None]
        sq: Dict = {"ep_target": ept, "old_ep_target": "official"}
        sq["gtfree_anchor256"] = km(g1p, L256, ep_target=ept)
        sq["gtfree_own_top16"] = km(g1p[ar[:, None], top16], L256[ar[:, None], top16], ep_target=ept)
        sq["gtfree_variants"] = km(p96, LP96, varmask, ep_target=ept)
        sq["gtfree_variants_per_type"] = {VNAMES[v]: km(p96, LP96, varmask & (vt == v)[None], ep_target=ept)
                                          for v in range(1, NV)}
        sq["gtfree_pairs"] = pair_metrics(p96, LP96, valid96, ep_target=ept)
        sq["gtfree_within_token_256"] = within_token(g1p, L256, p256, ep_target=ept)
        sq["gtfree_within_token_top16"] = within_token(g1p[ar[:, None], top16], L256[ar[:, None], top16],
                                                       np.zeros(n, np.int64), ep_target=ept)
        del g1p
        prob16 = sigmoid(np.load(rdir / "score_logit16.npy", mmap_mode="r")[r])
        prob80 = sigmoid(np.load(rdir / "score_logit80.npy", mmap_mode="r")[r])
        LVr = labvr[r]
        v80 = valid80[r]
        LPr = to_pool(LC, LVr)
        validp = to_pool(np.ones((n, K16), bool), v80)
        sq["r34_top16"] = km(prob16, LC, ep_target=ept)
        sq["r34_variants"] = km(prob80, LVr, v80, ep_target=ept)
        sq["r34_variants_per_type"] = {VNAMES[v]: km(prob80, LVr, v80 & ((np.arange(K16 * NVAR) % NVAR) == v - 1)[None],
                                                     ep_target=ept) for v in range(1, NV)}
        sq["r34_pairs"] = pair_metrics(to_pool(prob16, prob80), LPr, validp, ep_target=ept)
        sq["r34_within_token_top16"] = within_token(prob16, LC, ep_target=ept)
        olog = np.load(CK_DATA / "infer" / old / SPLIT / "score_logit.npy", mmap_mode="r")
        otok = pd.read_parquet(CK_DATA / "infer" / old / SPLIT / "tokens.parquet").token.to_numpy()
        assert (otok == tok_all).all() and np.load(CK_DATA / "infer" / old / SPLIT / "done.npy")[r].all()
        oprob = sigmoid(np.asarray(olog[r], np.float32))
        sq["old_r34_top16"] = km(oprob, LC)                    # old CK1 teachers: official EP (no --ep-target)
        sq["old_r34_within_token_top16"] = within_token(oprob, LC)
        tm["score_quality"] = sq

        # ============================================================ reference A (r34)
        pz = np.load(rdir / "picks.npz")
        prow = pz["rows"]
        pos = np.full(N, -1, np.int64)
        pos[prow] = np.arange(len(prow))
        assert (pos[r] >= 0).all()
        LS = L[t]["stack"][0][r]
        sidx = {nm: j for j, nm in enumerate(L[t]["select"]["stack_names"])}
        arows = [R.row("v2", base, paired=False)]
        o16r = LC[..., P].argmax(1)
        arows.append(R.row("oracle16", LC[ar, o16r]))
        o96r = np.where(validp, LPr[..., P], -np.inf).argmax(1)
        arows.append(R.row("oracle96", LPr[ar, o96r]))
        keep_pt(t, "refA", "oracle16", LC[ar, o16r], o16r)
        keep_pt(t, "refA", "oracle96", LPr[ar, o96r], o96r)
        for mode in R34_MODES:
            idx = pz[mode][pos[r]].astype(np.int64)
            isr = mode.startswith("r34")
            la = LC[ar, idx] if isr else LPr[ar, idx]
            lb = LS[:, sidx[f"{mode}_b"]]
            ex = {"frac_changed": float((idx != 0).mean())}
            if not isr:
                ex["share"] = {VNAMES[v]: float((idx % NV == v).mean()) for v in range(NV)}
                ex["frac_parent_not_v2"] = float((idx // NV != 0).mean())
            arows.append(R.row(f"{mode}_a", la, extra=ex))
            arows.append(R.row(f"{mode}_b", lb, extra=ex))
            keep_pt(t, "refA", f"{mode}_a", la, idx)
            keep_pt(t, "refA", f"{mode}_b", lb, idx)
        lv2 = LS[:, sidx["v2_lat"]]
        arows.append(R.row("v2_lat", lv2))
        keep_pt(t, "refA", "v2_lat", lv2, np.zeros(n, np.int64))
        # cross-check with the r34 stage-eval metrics.json
        rm = U.read_json(rdir / "metrics.json") or {}
        ref = {x["mode"]: x["pdms"] for x in rm.get("rows", [])}
        dd = [abs(ref[x["mode"]] - x["pdms"]) for x in arows if x["mode"] in ref]
        chk["refA_vs_r34_metrics_n_modes"] = len(dd)
        chk["refA_vs_r34_metrics_maxabs"] = float(max(dd)) if dd else None
        chk["refA_r34_metrics_n_eval"] = rm.get("n_eval")
        # old teacher rows
        opt = pd.read_parquet(CK_DATA / "eval" / old / SPLIT / "per_token.parquet")
        orows = []

        def old_lab(variant, beta):
            q = opt[(opt.variant == variant) & ((opt.beta.isna()) if beta is None else (opt.beta == beta))]
            q = q.drop_duplicates("token").set_index("token").reindex(pd.Index(tokens))
            assert q["pdms"].notna().all(), (old, variant, beta)
            return q[list(Cn.LABEL_COLS)].to_numpy(np.float64), q["chosen"].to_numpy(np.int64)

        ov2, _ = old_lab("v2", None)
        chk["old_v2_vs_v2_maxabs"] = float(np.abs(ov2 - base).max())
        for variant, beta in (("a", 1.0), ("b", 1.0)):
            lab, idx = old_lab(variant, beta)
            orows.append(R.row(f"{old} {variant}({beta:g})", lab, extra={"frac_changed": float((idx != 0).mean())}))
            keep_pt(t, "old", f"{variant}({beta:g})", lab, idx)
        oidx = np.argmax(sel_score(oprob), 1)
        orows.append(R.row(f"{old} CK-only a (context)", LC[ar, oidx], extra={"frac_changed": float((oidx != 0).mean())}))
        keep_pt(t, "old", "ck_noim_a", LC[ar, oidx], oidx)
        om = U.read_json(CK_DATA / "eval" / old / SPLIT / "metrics.json") or {}
        oref = {f"{x['variant']}({x['beta']:g})": x["pdms"] for x in om.get("rows", []) if x.get("beta") is not None}
        chk["old_rows_vs_old_metrics_maxabs"] = float(max(abs(oref[x["mode"].split(" ")[1]] - x["pdms"])
                                                          for x in orows[:2]))
        tm["refA"] = {"rows": arows, "old_rows": orows,
                      "old_metrics_json": str(CK_DATA / "eval" / old / SPLIT / "metrics.json")}

        # ============================================================ paired comparisons
        c_ = lambda s, m: chosen[(t, s, m)]  # noqa: E731
        pairs = [
            ("P96 - P256", c_("gtfree", "P96"), c_("gtfree", "P256"), "변형 pool 효과 (lateral 없음)"),
            ("P256_lat - P256", c_("gtfree", "P256_lat"), c_("gtfree", "P256"), "lateral 효과 (P256)"),
            ("P96_lat_always - P96", c_("gtfree", "P96_lat_always"), c_("gtfree", "P96"), "lateral 항상"),
            ("P96_lat_skiplatvar - P96", c_("gtfree", "P96_lat_skiplatvar"), c_("gtfree", "P96"),
             "lateral, l± 변형이면 생략"),
            ("P256 - r34_ck_noim_a", c_("gtfree", "P256"), c_("refA", "r34_ck_noim_a"),
             "256 anchor 단독 vs r34 top-16 CK-only"),
            ("P96 - pool96_ck_noim_a", c_("gtfree", "P96"), c_("refA", "pool96_ck_noim_a"),
             "own top-16 pool vs r34 pool (둘 다 CK-only, lateral 없음)"),
            (f"r34_old_b1_a - {old} a(1)", c_("refA", "r34_old_b1_a"), c_("old", "a(1)"), "같은 식(β1), teacher만 다름"),
            (f"r34_old_b1_b - {old} b(1)", c_("refA", "r34_old_b1_b"), c_("old", "b(1)"),
             "같은 선택식 + 각자 교정 (ck2: lateral only, old: lon+lat)"),
            (f"r34_ck_noim_a - {old} CK-only a", c_("refA", "r34_ck_noim_a"), c_("old", "ck_noim_a"),
             "CK-only 선택, teacher만 다름"),
            (f"r34_ck_noim_b - {old} b(1)", c_("refA", "r34_ck_noim_b"), c_("old", "b(1)"), ""),
            (f"pool96_ck_noim_b - {old} b(1)", c_("refA", "pool96_ck_noim_b"), c_("old", "b(1)"), ""),
            (f"P96 - {old} b(1)", c_("gtfree", "P96"), c_("old", "b(1)"), "GT-free 단독 vs 이전 teacher 최고 행"),
        ]
        tm["paired"] = [R.diff(nm, x, y, desc) for nm, x, y, desc in pairs]
        tm["checks"] = chk
        M["teachers"][t] = tm
        M["headline_pdms"][t] = {**{f"gtfree/{x['mode']}": x["pdms"] for x in rows},
                                 **{f"refA/{x['mode']}": x["pdms"] for x in arows},
                                 **{f"old/{x['mode']}": x["pdms"] for x in orows}}
        print(f"[{kst()}] {t}: " + ", ".join(f"{x['mode']} {100 * x['pdms']:.2f}" for x in rows) + f"; checks {chk}",
              flush=True)

    # ---------------------------------------------------------------- cross teacher (same tokens, paired)
    for s, mname in (("gtfree", "P256"), ("gtfree", "P96"), ("gtfree", "P96_lat_skiplatvar"), ("refA", "r34_ck_noim_a"),
                     ("refA", "r34_old_b1_b"), ("refA", "pool96_ck_noim_b")):
        M["cross_teacher"].append(R.diff(f"ck2T - ck2M: {s}/{mname}", chosen[("ck2T", s, mname)],
                                         chosen[("ck2M", s, mname)]))
    pT, pM = G["ck2T"]["p256_idx"][r], G["ck2M"]["p256_idx"][r]
    M["agreement"] = {"P256_anchor_same": float((pT == pM).mean())}

    # ---------------------------------------------------------------- write
    out.mkdir(parents=True, exist_ok=True)
    M["sec"] = round(time.time() - t0, 1)
    M["finished"] = kst()
    U.write_json(out / "metrics.json", M)
    if per_tok:
        tmp = out / ".per_token.tmp.parquet"
        pd.concat(per_tok, ignore_index=True).to_parquet(tmp, index=False)
        tmp.replace(out / "per_token.parquet")
    (out / "table.md").write_text(render(M))
    print((out / "table.md").read_text(), flush=True)
    print(f"[{kst()}] -> {out} ({M['sec']} s)", flush=True)
    return 0


def render(M: Dict) -> str:
    T = list(M["teachers"])
    c = M["v2_check"]
    Lm = ["# CK2 teacher navtest 평가 요약 (ck2T DET / ck2M MAP, ckpt_last)", "",
          f"생성 {M['created']}. 토큰 {M['n_eval']} / 전체 {M['n_total']} ({M['token_set']}). log {M['n_logs']}. "
          f"bootstrap {M['n_boot']}회 (log 군집, paired, seed 0). PDMS 등 0-100 단위. [실측]", "",
          "- 후보 선택은 GT/라벨 없이 이미 끝난 결과(p256_idx, p96_col, picks.npz)를 읽기만 함. 라벨은 지표 계산에만 사용. "
          "이 스크립트에서 새로 고르는 것은 참고 행 `old CK-only a` 하나뿐 (이전 teacher의 저장된 확률로 CK-only argmax).",
          f"- v2 r34 점검: 평균 PDMS {100 * c['v2_mean_pdms']:.3f} vs CSV {100 * c['csv_pdms']:.3f} (n {c['n']}, "
          f"토큰별 MAE {c.get('per_token_mae'):.2e}), v2_mismatch={c['v2_mismatch']}.",
          f"- 선택 점수 (CK-only) = 0.5 log NC + 0.5 log DAC + 1.0 log(5 TTC + 2 C + 5 EP), eps 1e-6.",
          f"- 앞차 감속 slice: {M['lead_def']}, n {M['lead_n']} ({M['lead_src']}).",
          "- ckpt sha16: " + ", ".join(f"{t} {M['teachers'][t]['ckpt_sha16']}" for t in T),
          "- 점수 품질의 EP 목표 (EP MAE / corr / BCE, EP 쌍 정확도, 토큰 내 EP Pearson): teacher run의 ep_target "
          "(ck_targets) = " + ", ".join(f"{t} {M['teachers'][t].get('ep_target', 'official')}" for t in T)
          + "; 이전 teacher official. PDMS와 제출 궤적의 sub-score는 공식 라벨 그대로.", ""]

    Lm += ["## 1. GT-free planner (teacher 단독, 256 raw anchor 출발)", "",
           "P256 = 256 anchor 중 CK-only 점수 argmax. P96 = own top-16 + 변형 5개씩 (a-1.0, a-0.5, a+0.5, l-0.5, l+0.5; "
           "무효 lateral 제외) 중 argmax. lateral = lateral head 교정 (z_lon = 0). 라벨 = 제출 궤적의 공식 라벨 "
           "(final5). ΔPDMS는 같은 토큰의 v2 r34 대비 paired.", "", ROW_HDR, ROW_SEP]
    for t in T:
        for m in M["teachers"][t]["gtfree"]["rows"]:
            Lm.append(row_line(t, GTFREE_LABEL.get(m["mode"], m["mode"]), m))
    Lm.append("")
    for t in T:
        g = M["teachers"][t]["gtfree"]
        p96 = next(x for x in g["rows"] if x["mode"] == "P96")
        ld = g["lat_dev"]
        Lm.append(f"- {t} P96 선택 비율: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in p96["share"].items())
                  + f"; P96 = P256 그대로 {100 * p96['frac_same_as_P256']:.1f}%. lateral 이동량(최대 xy 편차 평균): "
                  f"P256 {ld['P256_lat_vs_P256_maxdev_m_mean']:.3f} m (>0.1 m {100 * ld['P256_lat_frac_maxdev_gt_0p1m']:.1f}%), "
                  f"P96 {ld['P96_on_vs_off_maxdev_m_mean']:.3f} m (>0.1 m {100 * ld['P96_on_frac_maxdev_gt_0p1m']:.1f}%).")
    Lm.append(f"- P256 anchor ck2T/ck2M 일치율 {100 * M['agreement']['P256_anchor_same']:.1f}%.")
    Lm += ["", "### 1b. paired 비교 (A − B, 같은 토큰)", "",
           "| teacher | 비교 | 설명 | ΔPDMS [95% CI] | A 나음 / A 나쁨 % | Δ앞차 NC/TTC 실패 [95% CI] |", "|---|---|---|---|---|---|"]
    for t in T:
        for d in M["teachers"][t]["paired"]:
            Lm.append(f"| {t} | {d['name']} | {d['desc']} | {ci(d['d_pdms'])} | {100 * d['a_better']:.1f} / "
                      f"{100 * d['a_worse']:.1f} | {ci(d.get('d_lead_fail'))} |")
    for d in M["cross_teacher"]:
        Lm.append(f"| - | {d['name']} | teacher 비교 | {ci(d['d_pdms'])} | {100 * d['a_better']:.1f} / "
                  f"{100 * d['a_worse']:.1f} | {ci(d.get('d_lead_fail'))} |")

    Lm += ["", "## 2. 점수 품질 (navtest, 공식 라벨; EP 열은 각 teacher run의 ep_target 기준)", "",
           "fail AUC = 1 − p 로 실패(라벨 < 1) 구분, pass AUC = 4개 실패 키 확률 곱으로 전부 통과 구분 (train_ck2.key_metrics). "
           "EP corr / MAE 는 후보 전체 pooled. 변형은 유효한 것만 (무효 lateral 제외).", "",
           "| teacher | 후보 집합 | n | AUC fail NC | DAC | TTC | C | AUC 평균 | AUC pass | pass 비율 | EP corr | EP MAE | "
           "BCE NC | BCE DAC | BCE TTC |", "|" + "---|" * 15]
    sets = (("gtfree_anchor256", "256 raw anchor"), ("gtfree_own_top16", "own top-16"),
            ("gtfree_variants", "GT-free 변형 (유효)"), ("r34_top16", "r34 top-16"), ("r34_variants", "r34 변형 (유효)"),
            ("old_r34_top16", "이전 teacher r34 top-16"))
    for t in T:
        sq = M["teachers"][t]["score_quality"]
        for k, lab in sets:
            q = sq[k]
            who = f"{t} ({M['teachers'][t]['old_teacher']})" if k.startswith("old") else t
            Lm.append(f"| {who} | {lab} | {q.get('n_cand')} | {a3(q.get('auc_fail_nc'))} | {a3(q.get('auc_fail_dac'))} | "
                      f"{a3(q.get('auc_fail_ttc'))} | {a3(q.get('auc_fail_comfort'))} | {a3(q.get('auc_mean'))} | "
                      f"{a3(q.get('auc_pass'))} | {f2(q.get('pass_rate'), 100, 1)} | {a3(q.get('ep_corr'))} | "
                      f"{a3(q.get('ep_mae'))} | {a3(q.get('bce_nc'))} | {a3(q.get('bce_dac'))} | {a3(q.get('bce_ttc'))} |")
    Lm += ["", "변형 종류별 pass AUC / pass 비율 / fail AUC (NC, DAC, TTC):", "",
           "| teacher | 집합 | " + " | ".join(VNAMES[1:]) + " |", "|---|---|" + "---|" * (NV - 1)]
    for t in T:
        sq = M["teachers"][t]["score_quality"]
        for k, lab in (("gtfree_variants_per_type", "GT-free"), ("r34_variants_per_type", "r34")):
            cells = []
            for v in VNAMES[1:]:
                q = sq[k][v]
                cells.append(f"{a3(q.get('auc_pass'))} / {f2(q.get('pass_rate'), 100, 1)}% / "
                             f"{a3(q.get('auc_fail_nc'))}, {a3(q.get('auc_fail_dac'))}, {a3(q.get('auc_fail_ttc'))}")
            Lm.append(f"| {t} | {lab} | " + " | ".join(cells) + " |")
    Lm += ["", "부모-변형 쌍 순위 정확도 (sign(Δp) = sign(Δy), |Δy| > 1e-6 쌍만; PDMS는 CK-only 점수 차 vs PDMS 차):", "",
           "| teacher | 집합 | 쌍 수 | NC (n) | DAC (n) | EP (n) | TTC (n) | C (n) | PDMS (n) | " +
           " | ".join(f"PDMS {v}" for v in VNAMES[1:]) + " |", "|" + "---|" * (9 + NV - 1)]
    for t in T:
        sq = M["teachers"][t]["score_quality"]
        for k, lab in (("gtfree_pairs", "GT-free"), ("r34_pairs", "r34")):
            q = sq[k]
            keys = list(Cn.CK_KEYS) + ["pdms"]
            cells = [f"{f2(q.get(f'pair_acc_{x}'), 100, 1)} ({q.get(f'pair_n_{x}')})" for x in keys]
            pt = [f"{f2(q['per_type'][v].get('acc_pdms'), 100, 1)} ({q['per_type'][v].get('n_pdms')})"
                  for v in VNAMES[1:]]
            Lm.append(f"| {t} | {lab} | {q['pair_n']} | " + " | ".join(cells) + " | " + " | ".join(pt) + " |")
    Lm += ["", "토큰 내 순위 (CK-only 점수 vs 라벨 PDMS Spearman, 토큰 평균; EP Pearson 토큰 평균; top-1 = 고른 후보가 "
           "그 집합의 oracle과 같은 PDMS인 비율):", "",
           "| teacher | 집합 | Spearman (n tok) | EP Pearson (n tok) | top-1 |", "|---|---|---|---|---|"]
    for t in T:
        sq = M["teachers"][t]["score_quality"]
        for k, lab in (("gtfree_within_token_256", "256 raw anchor"), ("gtfree_within_token_top16", "own top-16"),
                       ("r34_within_token_top16", "r34 top-16"), ("old_r34_within_token_top16", "이전 teacher r34 top-16")):
            q = sq[k]
            Lm.append(f"| {t} | {lab} | {a3(q['spearman_sel_pdms_mean'])} ({q['spearman_n_tokens']}) | "
                      f"{a3(q['ep_pearson_within_mean'])} ({q['ep_pearson_n_tokens']}) | "
                      f"{f2(q.get('top1_is_oracle_frac'), 100, 1)} |")

    Lm += ["", "## 3. 참고 A: v2 r34 top-16 위에서 teacher 선택 (+ 이전 teacher)", "",
           "_a = 선택 궤적 그대로, _b = 선택 궤적에 lateral 교정. ck_noim = CK-only 점수, old_b1 / old_b0.5 = 이전 식 "
           "(v2 im 포함, β). pool96 = r34 16개 + 변형 80개. 이전 teacher 행은 eval/<old>/navtest/per_token.parquet "
           "(a(1) = 이전 식 β1, b(1) = a(1) + 이전 교정). `CK-only a (context)` 는 이전 teacher 확률로 이 스크립트가 계산.",
           "", ROW_HDR, ROW_SEP]
    for t in T:
        A = M["teachers"][t]["refA"]
        for m in A["rows"] + A["old_rows"]:
            Lm.append(row_line(t, m["mode"], m))
    Lm.append("")
    for t in T:
        for m in M["teachers"][t]["refA"]["rows"]:
            if "share" in m and m["mode"].endswith("_a"):
                Lm.append(f"- {t} {m['mode'][:-2]} 선택 비율: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in m["share"].items())
                          + f"; 부모가 v2가 아닌 비율 {100 * m['frac_parent_not_v2']:.1f}%")

    Lm += ["", "## 4. 앞차 감속 slice 요약", "",
           f"lead 플래그 있음 ({M['lead_src']}, n {M['lead_n']}). 위 표의 마지막 세 열이 slice 값. 주요 행:", "",
           "| teacher | 모드 | slice PDMS | NC/TTC 실패 % | Δ실패 vs v2 [95% CI] |", "|---|---|---|---|---|"]
    for t in T:
        tm = M["teachers"][t]
        pick = [x for x in tm["gtfree"]["rows"] if x["mode"] in ("v2", "P256", "P96", "P96_lat_always",
                                                                     "P96_lat_skiplatvar", "oracle256")]
        pick += [x for x in tm["refA"]["rows"] if x["mode"] in ("r34_ck_noim_a", "r34_old_b1_a", "r34_old_b1_b",
                                                                 "pool96_ck_noim_a", "pool96_old_b1_b")]
        pick += tm["refA"]["old_rows"]
        for m in pick:
            Lm.append(f"| {t} | {m['mode']} | {f2(m.get('lead_pdms'))} | {f2(m.get('lead_fail_nc_ttc'))} | "
                      f"{ci(m.get('d_lead_fail'))} |")

    Lm += ["", "## 5. 점검", ""]
    for t in T:
        Lm.append(f"- {t}: " + json.dumps(M["teachers"][t]["checks"], ensure_ascii=False))
    Lm += ["- 라벨 출처: " + ", ".join(f"{k} ({v['assembled']}, err {v['n_err_tokens']}, scorer {v['scorer_sha16']})"
                                    for k, v in M["label_sources"].items()),
           f"- 독립 재계산: tools/ck/e2e2/check_teacher_navtest_headline.py -> {D}/check_headline.json", ""]
    return "\n".join(Lm) + "\n"


if __name__ == "__main__":
    sys.exit(main())
