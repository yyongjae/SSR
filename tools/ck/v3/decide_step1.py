#!/usr/bin/env python
"""refiner-v3 Step-1 decision: metrics and the pre-registered GO rule (SPEC section 8; builder LOSS+EVAL,
2026-10-11 KST).

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python tools/ck/v3/decide_step1.py \
      --a D3/infer/v3T10_a_learned --b D3/infer/v3T10_b_ca --c D3/infer/ck2T10 [--d D3/infer/v3T10_d_oracle_PRIV] \
      [--out D3/decide/step1.json]

CPU only, read-only on every input, writes only --out (default D3/decide/step1.json; D3 =
/home/external-user/ssd/yongjae_refiner/ck/v3).  navtest is never read.  Each --a/--b/--c/--d is an infer dir of
tools/ck/v3/infer_val_v3.py (prob16 [N,16,5], prob80 [N,80,5], done [N], infer_meta.json; N = 18,179 packed
navtrain_val rows).  Arm (c) must be a FRESH r34 dump of ck2T10 (infer_val_v3.py --run ck2/train/ck2T10): the
sd_plugin ck2T10 dump is on SafeDrive candidates (its own-DAC top-16 PDMS is 82.78, not ck2T10's 90.32) [실측].

Protocol (all arms, ckpt_last, navtrain_val r34 top-16, no best-epoch selection):
  decision PDMS  p = arm prob16 with p_dac FROZEN = lead_remedy/infer/ck2T/prob16[..., 1] (the ck2T10 recipe does not
                 train DAC); pick = argmax ck_final(p, im = 1, w = noim (0, .5, .5, 1)) over the 16 candidates
                 (first maximum = index order, = lib.select with tie arange(16)); PDMS = official
                 labels/navtrain_val/cand/labels.npy[:, :16, 6] of the pick.  Delta = 100 x per-token difference,
                 CI = paired log-cluster bootstrap (eval_ck.log_bootstrap = lib.boot, 2,000 resamples, seed 0).
                 Variants (descriptive): w_dac = 0 (w = 0, .5, 0, 1); p_dac = packed v2_sim[..., 1]; the arm's own DAC.
  lead-decel     lead_decel == 1 in ck/lead/navtrain_val.parquet (1,605 tokens).  TTC AUC = roc_auc(ttc < 1,
                 1 - p_ttc) over all 16 candidates of those tokens (NC AUC likewise); NC|TTC fail = % of those tokens
                 whose pick has nc < 1 or ttc < 1.
  hazard pref    pool96 on the labelled 3,000-token sample (lead_remedy/infer/sample3000/labels96.npz, column
                 c = 6 k + v): pairs = parent (v = 0) safe (nc >= 1 and ttc >= 1), its a-1.0 variant (v = 1) unsafe,
                 both allowed (valid96 | identity) with finite labels (ok96); pref = share of pairs with
                 log p_nc + log p_ttc strictly higher for the parent (A_novelty cv_st_baseline.py 435-467).  Paired
                 CI: log-cluster bootstrap of the per-pair indicator difference.
GO iff all (SPEC 8, as revised by the main session's review decision D4, 2026-10-11 KST):
  G1  Delta PDMS (a - b) >= +0.20 and its CI lower bound > 0.  ONE seed per arm: the CI covers token / log sampling,
      NOT training-seed noise (recorded in the output); if |Delta - 0.20| <= 0.15 the output says a second seed of
      (a) and (b) is to be run before the decision is final (second_seed_required).  Reported next to G1 (not
      gating): the same delta on the full-history subset and split by past-frame cache source (review causality
      findings 2, 3).
  G2  lead-decel TTC AUC of (a) >= .855 (absolute) -- plus the paired deltas (a) - (b), (a) - (c) with log-cluster
      bootstrap CIs and the (b), (c) values next to the threshold (reported, not gating)
  G3  lead-decel NC|TTC fail of (a) <= 6.97 % (absolute) -- plus (a) - (b), (a) - (c) with CIs (reported)
  G4  a-1.0 hazard preference: pre-registered NON-INFERIORITY of (a) vs (c): point estimate of pref(a) - pref(c)
      >= -0.01 and its CI lower bound >= -0.03; (a) - (b) reported
Descriptive: (b) - (c) (expected ~ +0.38; ~ 0 = the branch, not the forecaster, is broken), (a) - (c), (d) - (b)
(oracle headroom, PRIVILEGED, never a gate input), every metric on the full-history subset (frame_valid all True in
the clean val object pack, when built), forecast metrics of (a) vs CA when the infer dirs carry fc_metrics.json.

Arm checks (refuse unless --no-arm-check, which marks the result valid_for_decision = false): infer_meta split
navtrain_val, which last, r34 candidates (cand_sha16 of packed cand.npy) and the lead_remedy variants
(variants_sha16), temporal forecaster a learned / b ca / d oracle_gt, (c) temporal off; privileged only for (d);
the inference reproduce check passed (infer_meta check.ok, review fix D6); PROVENANCE (review fix D5): arms a, b, d
must share the training-code sha, K_OBJ, the train / val object-pack shas and the CA-scales sha
(tools/ck/v3/provenance.compare on infer_meta provenance.train), arm a's forecaster must be pre-trained on the same
train pack, and each arm's inference must have read the val pack it was trained with.  A differing worktree sha is
a warning only.  The frozen-DAC file's token order (lead_remedy/infer/ck2T/tokens.parquet) must equal the packed
order.
A token set smaller than all 18,179 rows also marks the result invalid (--allow-partial to run anyway).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
D3 = CK / "v3"
LR = CK / "ck2" / "analysis" / "lead_remedy"
SPLIT = "navtrain_val"
N_VAL = 18179
K16, NV, NVAR = 16, 6, 5
G96 = K16 * NV
V_A10 = 1                                      # VNAMES index of a-1.0 (id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)
FROZEN_DAC = LR / "infer" / "ck2T" / "prob16.npy"
VARIANTS = LR / "infer" / "variants" / "traj80.npy"
POOL = LR / "infer" / "sample3000" / "labels96.npz"
LABELS = CK / "labels" / SPLIT / "cand"
PACK = CK / "packed" / SPLIT
LEAD = CK / "lead" / f"{SPLIT}.parquet"
OBJPACK_VAL = D3 / "objpack_k96" / "bevfusion" / SPLIT          # K_OBJ 96 (review fix D1)
FROZEN_DAC_TOKENS = LR / "infer" / "ck2T" / "tokens.parquet"
W_NOIM = (0.0, 0.5, 0.5, 1.0)
W_NODAC = (0.0, 0.5, 0.0, 1.0)
LBL = {k: i for i, k in enumerate(("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress",
                                   "pdm_progress_eff"))}
PK = {k: i for i, k in enumerate(("nc", "dac", "ep", "ttc", "comfort"))}      # CK_KEYS order of prob16 / prob80
N_BOOT, SEED = 2000, 0
GO_RULE = {"G1": {"pair": "a-b", "min_delta_pdms": 0.20, "ci_lo_gt": 0.0, "second_seed_band": 0.15,
                  "note": "one seed per arm: seed noise not covered by the CI; a second seed of (a) and (b) is run "
                          "if |Delta - 0.20| <= 0.15 (main-session decision D4)"},
           "G2": {"arm": "a", "min_lead_ttc_auc": 0.855, "report": ["a-b", "a-c"]},
           "G3": {"arm": "a", "max_lead_ncttc_fail_pct": 6.97, "report": ["a-b", "a-c"]},
           "G4": {"pair": "a-c", "min_delta": -0.01, "min_ci_lo": -0.03, "report": ["a-b"],
                  "note": "pre-registered non-inferiority (main-session decision D4)"}}
PAIRS = ("a-b", "b-c", "a-c", "d-b", "d-a")
EXPECT_FORECASTER = {"a": "learned", "b": "ca", "d": "oracle_gt"}
VERSION = "v3_decide_2"          # 2: review fixes D4 (gates), D5 (provenance), D6 (check)
KST = 9 * 3600


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def sha16(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- core metrics
def boot(diff, logs, n_boot: int = N_BOOT, seed: int = SEED) -> Dict:
    """Paired log-cluster bootstrap of mean(diff) (tools/ck/eval_ck.log_bootstrap == B_ceiling lib.boot)."""
    from tools.ck.eval_ck import log_bootstrap
    return log_bootstrap(np.asarray(diff, np.float64), np.asarray(logs), n_boot=n_boot, seed=seed)


def with_dac(p: np.ndarray, p_dac: Optional[np.ndarray]) -> np.ndarray:
    """copy of p [.., K, 5] (float64) with the DAC column replaced (None: unchanged copy)."""
    q = np.array(p, np.float64, copy=True)
    if p_dac is not None:
        q[..., PK["dac"]] = np.asarray(p_dac, np.float64)
    return q


def select_pick(p: np.ndarray, w=W_NOIM) -> np.ndarray:
    """argmax over K of ck_final(p, im = 1, w) (navsim ck.select.ck_final), non-finite scores -> -inf, first maximum."""
    from navsim.agents.para_ssr.ck.select import ck_final
    s = np.asarray(ck_final(np.asarray(p, np.float64), np.ones(p.shape[:-1]), w), np.float64)
    s = np.where(np.isfinite(s), s, -np.inf)
    return s.argmax(-1).astype(np.int64)


def roc_auc(y, s) -> float:
    """ROC AUC of score s for the positive class y (rank-based, average ties; = sklearn roc_auc_score)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool).reshape(-1)
    s = np.asarray(s, np.float64).reshape(-1)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_boot(y: np.ndarray, scores: Dict[str, np.ndarray], logs: np.ndarray, pairs, n_boot: int = N_BOOT,
             seed: int = SEED, alpha: float = 0.05, chunk: int = 50) -> Dict[str, Dict]:
    """Paired log-cluster bootstrap of ROC-AUC differences.  y [n] bool (positive class), scores {arm: [n]} (higher
    = more positive; the same candidates for every arm), logs [n] (cluster of each candidate).  Resampling weights =
    the multinomial log weights of eval_ck.log_bootstrap (same rng and seed); AUC with integer weights
    (Mann-Whitney, ties count 1/2), identical to roc_auc for unit weights.
    -> {pair 'x-z': dict(mean = AUC_x - AUC_z (unweighted), lo, hi, n, n_logs)}."""
    y = np.asarray(y, bool).reshape(-1)
    logs = np.asarray(logs).reshape(-1)
    ul, inv = np.unique(logs, return_inverse=True)
    L = len(ul)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(np.float64)       # [B, L]
    est = {}
    for k, sc in scores.items():
        sc = np.asarray(sc, np.float64).reshape(-1)
        o = np.argsort(sc, kind="stable")
        ss, yy, gg = sc[o], y[o], inv[o]
        _, gid = np.unique(ss, return_inverse=True)                                    # tie groups, ascending
        G = int(gid.max()) + 1 if len(gid) else 0
        # per tie group and log: positive / negative candidate counts -> group sums are W @ counts
        cp = np.zeros((G, L))
        cn = np.zeros((G, L))
        np.add.at(cp, (gid[yy], gg[yy]), 1.0)
        np.add.at(cn, (gid[~yy], gg[~yy]), 1.0)
        out = np.empty(n_boot)
        for b0 in range(0, n_boot, chunk):
            Wb = W[b0:b0 + chunk]                                                        # [b, L]
            Gp, Gn = Wb @ cp.T, Wb @ cn.T                                                # [b, G]
            below = np.cumsum(Gn, 1) - Gn
            num = (Gp * (below + 0.5 * Gn)).sum(1)
            out[b0:b0 + chunk] = num / np.maximum(Gp.sum(1) * Gn.sum(1), 1e-12)
        est[k] = out
    res = {}
    for pr in pairs:
        x, z = pr.split("-")
        if x not in scores or z not in scores:
            continue
        d = est[x] - est[z]
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        res[pr] = {"mean": float(roc_auc(y, scores[x]) - roc_auc(y, scores[z])), "lo": float(lo), "hi": float(hi),
                   "n": int(len(y)), "n_logs": int(L)}
    return res


def to_pool(x16: np.ndarray, x80: np.ndarray) -> np.ndarray:
    """[n, 16, ..] + [n, 80, ..] -> [n, 96, ..], column c = 6 k + v (v = 0 identity, 1..5 variants k*5 + v - 1)
    (= B_ceiling lib.to_pool)."""
    n = x16.shape[0]
    tail = x16.shape[2:]
    o = np.empty((n, K16, NV) + tail, np.result_type(x16, x80))
    o[:, :, 0] = x16
    o[:, :, 1:] = np.asarray(x80).reshape((n, K16, NVAR) + tail)
    return o.reshape((n, G96) + tail)


def safe_of(lab: np.ndarray) -> np.ndarray:
    return (lab[..., LBL["nc"]] >= 1) & (lab[..., LBL["ttc"]] >= 1)


def hazard_pairs(lab96: np.ndarray, ok96: np.ndarray, valid96: np.ndarray,
                 v: int = V_A10) -> Tuple[np.ndarray, np.ndarray]:
    """-> (i, k) int arrays: sample token i, parent k with column 6 k safe, column 6 k + v unsafe, both allowed and
    finite-labelled."""
    n = lab96.shape[0]
    allowed = valid96.astype(bool) | (np.arange(G96) % NV == 0)[None]
    fin = ok96.astype(bool) & np.isfinite(lab96[..., [LBL["nc"], LBL["ttc"]]]).all(-1) & allowed
    safe = safe_of(lab96)
    par = np.arange(K16) * NV
    var = par + v
    m = fin[:, par] & fin[:, var] & safe[:, par] & ~safe[:, var]             # [n, 16]
    ii, kk = np.nonzero(m)
    assert ii.max(initial=-1) < n
    return ii.astype(np.int64), kk.astype(np.int64)


def hazard_indicator(p96: np.ndarray, ii: np.ndarray, kk: np.ndarray, v: int = V_A10) -> np.ndarray:
    """1.0 where log p_nc + log p_ttc of the parent is strictly higher than the variant's, per pair."""
    q = np.log(np.asarray(p96[..., PK["nc"]], np.float64)) + np.log(np.asarray(p96[..., PK["ttc"]], np.float64))
    a, b = q[ii, kk * NV], q[ii, kk * NV + v]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite arm probabilities on hazard pairs")
    return (a > b).astype(np.float64)


# ----------------------------------------------------------------------------------------------- data
@dataclass
class Data:
    """Aligned decision inputs (rows = the evaluated token set, packed order)."""
    tokens: np.ndarray            # [N] str
    logs: np.ndarray              # [N] str
    lab: np.ndarray               # [N, 16, 9] official labels of the r34 top-16
    lead: np.ndarray              # [N] bool (lead_decel == 1)
    p_dac: np.ndarray             # [N, 16] frozen DAC probability (ck2T prob16[..., 1])
    v2_dac: np.ndarray            # [N, 16] packed v2_sim[..., 1]
    pool_rows: np.ndarray         # [n] rows (into the N rows) of the labelled pool sample
    lab96: np.ndarray             # [n, 96, 9]
    ok96: np.ndarray              # [n, 96]
    valid96: np.ndarray           # [n, 96]
    full_hist: Optional[np.ndarray] = None     # [N] bool, all 3 frames valid in the clean val object pack
    fut_src: Optional[np.ndarray] = None       # [N] bool, a valid past frame came from the '_future'-named cache
    p_ck2T: Optional[np.ndarray] = None        # [N, 16, 5] ck2T prob16 (frozen DAC / EP / C of the NC-TTC-only view)
    rows: Optional[np.ndarray] = None          # [N] packed rows of the evaluated tokens
    n_total: int = N_VAL
    info: Dict = field(default_factory=dict)


@dataclass
class Arm:
    p16: np.ndarray               # [N, 16, 5] (aligned with Data rows)
    p80: np.ndarray               # [N, 80, 5]
    meta: Dict = field(default_factory=dict)
    fc_metrics: Optional[Dict] = None


def load_infer(d: Path) -> Dict:
    d = Path(d)
    out = {"dir": str(d), "meta": json.load(open(d / "infer_meta.json")) if (d / "infer_meta.json").is_file() else {}}
    for k in ("16", "80"):
        if (d / f"prob{k}.npy").is_file():
            out[f"p{k}"] = np.load(d / f"prob{k}.npy", mmap_mode="r")
        else:
            z = np.load(d / f"score_logit{k}.npy", mmap_mode="r")
            out[f"p{k}"] = 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))
    out["done"] = np.load(d / "done.npy").astype(bool)
    out["prob16_sha16"] = sha16(d / "prob16.npy") if (d / "prob16.npy").is_file() else None
    fm = d / "fc_metrics.json"
    out["fc_metrics"] = json.load(open(fm)) if fm.is_file() else None
    if (d / "tokens.parquet").is_file():
        import pandas as pd
        out["tokens"] = pd.read_parquet(d / "tokens.parquet").token.astype(str).to_numpy()
    return out


def load_data(done: np.ndarray) -> Data:
    """Real navtrain_val inputs restricted to rows with every arm done, 16 cand labels ok and finite."""
    import pandas as pd
    tdf = pd.read_parquet(PACK / "tokens.parquet")
    N = len(tdf)
    if N != N_VAL or len(done) != N:
        raise SystemExit(f"packed {SPLIT} has {N} rows, done {len(done)} (expected {N_VAL})")
    lab = np.load(LABELS / "labels.npy", mmap_mode="r")[:, :K16].astype(np.float64)
    lok = np.load(LABELS / "ok.npy")[:, :K16].astype(bool)
    rows = np.flatnonzero(done & lok.all(1) & np.isfinite(lab).all((1, 2)))
    ld = pd.read_parquet(LEAD).drop_duplicates("token").set_index("token")
    tok = tdf.token.astype(str).to_numpy()
    lead = ld["lead_decel"].reindex(pd.Index(tok)).fillna(0).to_numpy() == 1
    p_dac = np.load(FROZEN_DAC, mmap_mode="r")[..., PK["dac"]].astype(np.float64)
    v2d = np.load(PACK / "v2_sim.npy", mmap_mode="r")[..., 1].astype(np.float64)
    z = np.load(POOL)
    srow = z["rows"].astype(np.int64)
    if not (z["tokens"].astype(str) == tok[srow]).all():
        raise SystemExit(f"{POOL} tokens do not match packed rows")
    pos = np.full(N, -1, np.int64)
    pos[rows] = np.arange(len(rows))
    keep = pos[srow] >= 0
    fh, fs = None, None
    if (OBJPACK_VAL / "frame_valid.npy").is_file():
        fv = np.load(OBJPACK_VAL / "frame_valid.npy")
        ptok = pd.read_parquet(OBJPACK_VAL / "tokens.parquet").token.astype(str).to_numpy()
        if len(ptok) != N or not (ptok == tok).all():
            raise SystemExit(f"{OBJPACK_VAL} token order != packed {SPLIT}")
        fh = fv.all(1)[rows]
        src = np.load(OBJPACK_VAL / "frame_src.npy")
        fs = ((src[:, 1:] == 2) & fv[:, 1:]).any(1)[rows]
    if FROZEN_DAC_TOKENS.is_file():
        ftok = pd.read_parquet(FROZEN_DAC_TOKENS).token.astype(str).to_numpy()
        if len(ftok) != N or not (ftok == tok).all():
            raise SystemExit(f"{FROZEN_DAC_TOKENS}: token order != packed {SPLIT} (frozen DAC would be misaligned)")
    else:
        raise SystemExit(f"{FROZEN_DAC_TOKENS} missing: cannot check the frozen-DAC token order")
    p_ck2T = np.load(FROZEN_DAC, mmap_mode="r").astype(np.float64)
    info = {"labels": str(LABELS), "lead": str(LEAD), "frozen_dac": str(FROZEN_DAC), "frozen_dac_sha16":
            sha16(FROZEN_DAC), "pool": str(POOL), "pool_sha16": sha16(POOL), "v2_sim": str(PACK / "v2_sim.npy"),
            "objpack_val": str(OBJPACK_VAL) if fh is not None else None, "n_pool_dropped": int((~keep).sum()),
            "frozen_dac_tokens_checked": True}
    return Data(tokens=tok[rows], logs=tdf.log.astype(str).to_numpy()[rows], lab=lab[rows], lead=lead[rows],
                p_dac=p_dac[rows], v2_dac=v2d[rows], pool_rows=pos[srow[keep]],
                lab96=z["lab96"][keep].astype(np.float64),
                ok96=z["ok96"][keep].astype(bool), valid96=z["valid96"][keep].astype(bool), full_hist=fh, rows=rows,
                fut_src=fs, p_ck2T=p_ck2T[rows], n_total=N, info=info)


# ----------------------------------------------------------------------------------------------- arm checks
def check_arm_meta(key: str, meta: Dict, ref: Optional[Dict] = None) -> List[str]:
    """Problems (empty = ok) of an infer_meta.json for arm key a / b / c / d (module docstring)."""
    bad = []
    t = meta.get("temporal") or {}
    on = bool(t.get("on"))
    priv = bool(meta.get("privileged", False)) or bool(t.get("privileged", False))
    if meta.get("split", SPLIT) != SPLIT:
        bad.append(f"split {meta.get('split')!r}")
    if meta.get("which", "last") != "last":
        bad.append(f"which {meta.get('which')!r} (decision uses ckpt_last)")
    if ref is not None:
        for k in ("cand_sha16", "variants_sha16"):
            if meta.get(k) != ref.get(k):
                bad.append(f"{k} {meta.get(k)!r} != {ref.get(k)!r}")
    if key == "c":
        if on:
            bad.append("arm c must be temporal off")
    else:
        if not on:
            bad.append(f"arm {key} must be temporal on")
        elif t.get("forecaster") != EXPECT_FORECASTER[key]:
            bad.append(f"arm {key} forecaster {t.get('forecaster')!r} != {EXPECT_FORECASTER[key]!r}")
    if priv != (key == "d"):
        bad.append(f"arm {key} privileged={priv} (only arm d may be, and must be, privileged)")
    chk = meta.get("check") or {}
    if chk.get("ok") is not True:
        bad.append(f"arm {key}: inference reproduce check not ok ({chk.get('rule', 'missing')}; infer_meta check)")
    if on:
        pv = meta.get("provenance") or {}
        tr, inf = pv.get("train") or {}, pv.get("infer") or {}
        if not tr:
            bad.append(f"arm {key}: no training provenance in the run's config.json (pre-fix run?)")
        else:
            vt = (((tr.get("data") or {}).get("packs") or {}).get(SPLIT) or {}).get("build_sha256")
            vi = (((inf.get("data") or {}).get("packs") or {}).get(SPLIT) or {}).get("build_sha256")
            if vt is None or vt != vi:
                bad.append(f"arm {key}: inference read val pack {vi} but the run was trained / evaluated with {vt}")
    return bad


def provenance_check(raw: Dict[str, Dict]) -> Dict[str, List[str]]:
    """review fix D5: arms a / b / d must share code and data (tools/ck/v3/provenance.compare)."""
    from tools.ck.v3 import provenance as PV
    provs = {k: ((r["meta"].get("provenance") or {}).get("train") or {}) for k, r in raw.items() if k in ("a", "b", "d")}
    return PV.compare(provs)


# ----------------------------------------------------------------------------------------------- decision
def arm_metrics(data: Data, arm: Arm) -> Tuple[Dict, Dict]:
    """-> (metrics, per-token / per-pair vectors for the paired deltas)."""
    lab = data.lab
    n = len(lab)
    ar = np.arange(n)
    pd_ = lab[..., LBL["pdms"]]
    fail = ~safe_of(lab)
    p16 = np.asarray(arm.p16, np.float64)
    picks = {"decision": select_pick(with_dac(p16, data.p_dac), W_NOIM),
             "wdac0": select_pick(with_dac(p16, None), W_NODAC),
             "v2dac": select_pick(with_dac(p16, data.v2_dac), W_NOIM),
             "owndac": select_pick(with_dac(p16, None), W_NOIM)}
    if data.p_ck2T is not None:          # descriptive: only NC / TTC from the arm, DAC / EP / C frozen from ck2T
        q = np.array(data.p_ck2T, np.float64, copy=True)
        q[..., PK["nc"]], q[..., PK["ttc"]] = p16[..., PK["nc"]], p16[..., PK["ttc"]]
        picks["ncttc_only"] = select_pick(q, W_NOIM)
    vec = {f"pdms_{k}": 100.0 * pd_[ar, c] for k, c in picks.items()}
    vec["fail"] = 100.0 * fail[ar, picks["decision"]].astype(np.float64)
    m: Dict = {k: float(v.mean()) for k, v in vec.items() if k.startswith("pdms_")}
    m["pdms"] = m["pdms_decision"]
    m["ncttc_fail_pct"] = float(vec["fail"].mean())
    m["frac_changed"] = float((picks["decision"] != 0).mean())
    L = data.lead
    y_ttc = lab[L][..., LBL["ttc"]] < 1
    y_nc = lab[L][..., LBL["nc"]] < 1
    m["lead"] = {"n": int(L.sum()), "ttc_auc": roc_auc(y_ttc, 1 - p16[L][..., PK["ttc"]]),
                 "nc_auc": roc_auc(y_nc, 1 - p16[L][..., PK["nc"]]),
                 "ncttc_fail_pct": float(vec["fail"][L].mean()) if L.any() else float("nan"),
                 "pdms": float(vec["pdms_decision"][L].mean()) if L.any() else float("nan")}
    m["all_ttc_auc"] = roc_auc(lab[..., LBL["ttc"]] < 1, 1 - p16[..., PK["ttc"]])
    m["all_nc_auc"] = roc_auc(lab[..., LBL["nc"]] < 1, 1 - p16[..., PK["nc"]])
    # hazard preference (pool96 sample)
    ii, kk = hazard_pairs(data.lab96, data.ok96, data.valid96)
    pr = data.pool_rows
    p96 = to_pool(np.asarray(arm.p16, np.float64)[pr], np.asarray(arm.p80, np.float64)[pr])
    ind = hazard_indicator(p96, ii, kk)
    vec["hazard"] = ind
    m["hazard_a-1.0"] = {"n_pairs": int(len(ind)), "pref": float(ind.mean()) if len(ind) else float("nan"),
                         "ci": boot(ind, data.logs[pr][ii])}
    if data.fut_src is not None:
        fs = data.fut_src
        m["past_src"] = {"n_future_cache": int(fs.sum()), "pdms_future_cache": float(vec["pdms_decision"][fs].mean())
                         if fs.any() else float("nan"), "pdms_current_only": float(vec["pdms_decision"][~fs].mean())}
    if data.full_hist is not None:
        fh = data.full_hist
        m["full_history"] = {"n": int(fh.sum()), "pdms": float(vec["pdms_decision"][fh].mean()),
                             "lead_n": int((fh & L).sum()),
                             "lead_ttc_auc": roc_auc(lab[fh & L][..., LBL["ttc"]] < 1, 1 - p16[fh & L][..., PK["ttc"]]),
                             "lead_ncttc_fail_pct": (float(vec["fail"][fh & L].mean()) if (fh & L).any()
                                                     else float("nan"))}
    if arm.fc_metrics is not None:
        m["forecast"] = arm.fc_metrics
    return m, vec


def decide(data: Data, arms: Dict[str, Arm], n_boot: int = N_BOOT, seed: int = SEED,
           arm_check: Optional[Dict[str, List[str]]] = None, prov_check: Optional[Dict] = None) -> Dict:
    """Section-8 metrics, paired deltas and the GO rule on aligned inputs (synthetic or real).  The result is
    valid_for_decision only with arm checks run and clean (arm_check not None, no problems) on the full token set."""
    for k in ("a", "b", "c"):
        if k not in arms:
            raise ValueError(f"arm {k} missing (a, b, c required; d optional)")
    M, V = {}, {}
    for k, arm in arms.items():
        M[k], V[k] = arm_metrics(data, arm)
    logs = data.logs
    L = data.lead
    deltas = {}
    for pr in PAIRS:
        x, y = pr.split("-")
        if x not in arms or y not in arms:
            continue
        d = {kk: boot(V[x][kk] - V[y][kk], logs, n_boot, seed)
             for kk in ("pdms_decision", "pdms_wdac0", "pdms_v2dac", "pdms_owndac", "pdms_ncttc_only") if kk in V[x]}
        if data.fut_src is not None:
            fs = data.fut_src
            d["pdms_past_future_cache"] = boot((V[x]["pdms_decision"] - V[y]["pdms_decision"])[fs], logs[fs], n_boot, seed)
            d["pdms_past_current_only"] = boot((V[x]["pdms_decision"] - V[y]["pdms_decision"])[~fs], logs[~fs], n_boot,
                                               seed)
        d["lead_fail_pp"] = boot((V[x]["fail"] - V[y]["fail"])[L], logs[L], n_boot, seed)
        pl = logs[data.pool_rows][hazard_pairs(data.lab96, data.ok96, data.valid96)[0]]
        d["hazard_pref"] = boot(V[x]["hazard"] - V[y]["hazard"], pl, n_boot, seed)
        if data.full_hist is not None:
            fh = data.full_hist
            d["full_history_pdms"] = boot((V[x]["pdms_decision"] - V[y]["pdms_decision"])[fh], logs[fh], n_boot, seed)
        deltas[pr] = d
    # lead-decel AUC deltas (G2 report): paired log-cluster bootstrap over the lead tokens' candidates
    Lr = np.flatnonzero(L)
    clog = np.repeat(logs[Lr], K16)
    auc_d = {}
    for key, col in (("ttc", "ttc"), ("nc", "nc")):
        yy = (data.lab[Lr][..., LBL[col]] < 1).reshape(-1)
        sc = {k: (1.0 - np.asarray(arms[k].p16, np.float64)[Lr][..., PK[col]]).reshape(-1) for k in arms}
        auc_d[key] = auc_boot(yy, sc, clog, [p for p in PAIRS if all(q in arms for q in p.split("-"))], n_boot, seed)
    for pr in deltas:
        deltas[pr]["lead_ttc_auc"] = auc_d["ttc"].get(pr)
        deltas[pr]["lead_nc_auc"] = auc_d["nc"].get(pr)
    g1 = deltas["a-b"]["pdms_decision"]
    g4 = deltas["a-c"]["hazard_pref"]
    R = GO_RULE
    near = bool(abs(g1["mean"] - R["G1"]["min_delta_pdms"]) <= R["G1"]["second_seed_band"])
    side = {pr: {"lead_ttc_auc": deltas[pr]["lead_ttc_auc"], "lead_fail_pp": deltas[pr]["lead_fail_pp"]}
            for pr in ("a-b", "a-c") if pr in deltas}
    gates = {
        "G1": {"value": g1["mean"], "lo": g1["lo"], "hi": g1["hi"],
               "pass": bool(g1["mean"] >= R["G1"]["min_delta_pdms"] and g1["lo"] > R["G1"]["ci_lo_gt"]),
               "rule": "Delta PDMS (a - b) >= +0.20 and CI lo > 0 (one seed: CI excludes seed noise)",
               "second_seed_required": near,
               "co_report": {k: deltas["a-b"].get(k) for k in ("full_history_pdms", "pdms_past_future_cache",
                                                               "pdms_past_current_only", "pdms_ncttc_only")}},
        "G2": {"value": M["a"]["lead"]["ttc_auc"],
               "pass": bool(M["a"]["lead"]["ttc_auc"] >= R["G2"]["min_lead_ttc_auc"]),
               "rule": "lead-decel TTC AUC (a) >= .855 (absolute)",
               "values": {k: M[k]["lead"]["ttc_auc"] for k in M},
               "deltas": {pr: v["lead_ttc_auc"] for pr, v in side.items()}},
        "G3": {"value": M["a"]["lead"]["ncttc_fail_pct"],
               "pass": bool(M["a"]["lead"]["ncttc_fail_pct"] <= R["G3"]["max_lead_ncttc_fail_pct"]),
               "rule": "lead-decel NC|TTC fail (a) <= 6.97 % (absolute)",
               "values": {k: M[k]["lead"]["ncttc_fail_pct"] for k in M},
               "deltas": {pr: v["lead_fail_pp"] for pr, v in side.items()}},
        "G4": {"value": g4["mean"], "lo": g4["lo"], "hi": g4["hi"],
               "pass": bool(g4["mean"] >= R["G4"]["min_delta"] and g4["lo"] >= R["G4"]["min_ci_lo"]),
               "pref": {k: M[k]["hazard_a-1.0"]["pref"] for k in M},
               "vs_b": deltas["a-b"]["hazard_pref"],
               "rule": "a-1.0 hazard pref non-inferiority vs (c): pref(a) - pref(c) >= -0.01 and CI lo >= -0.03"},
    }
    go = all(g["pass"] for g in gates.values())
    partial = len(data.lab) < data.n_total
    problems = {k: v for k, v in (arm_check or {}).items() if v}
    prov_bad = bool(prov_check is not None and prov_check.get("errors"))
    valid = (arm_check is not None) and (prov_check is not None) and not problems and not prov_bad and not partial
    notes = []
    if partial:
        notes.append(f"token set {len(data.lab)} < {data.n_total} rows: not the pre-registered set")
    if arm_check is None:
        notes.append("arm checks skipped (--no-arm-check): not a decision")
    if problems:
        notes.append(f"arm check problems: {problems}")
    if near:
        notes.append(f"G1 delta {g1['mean']:+.3f} is within {R['G1']['second_seed_band']} of the +0.20 threshold: run a "
                     f"second seed of (a) and (b) before the decision is final (D4)")
    notes.append("single seed per arm: the G1 CI covers log / token sampling, not training-seed noise")
    bc = deltas.get("b-c", {}).get("pdms_decision")
    if bc is not None and bc["lo"] <= 0.0:
        notes.append("(b) - (c) CI includes or is below 0: the temporal branch itself (not the forecaster) may be "
                     "broken; expected about +0.38 [추정 from the CA stack]")
    if prov_check is not None:
        if prov_check.get("errors"):
            notes.append(f"provenance errors: {prov_check['errors']}")
        if prov_check.get("warnings"):
            notes.append(f"provenance warnings: {prov_check['warnings']}")
    return {"version": VERSION, "n": int(len(data.lab)), "n_total": int(data.n_total),
            "n_logs": int(len(np.unique(logs))), "n_lead": int(L.sum()),
            "n_hazard_pairs": int(M["a"]["hazard_a-1.0"]["n_pairs"]),
            "arms": M, "deltas": deltas, "gates": gates, "go": bool(go and valid),
            "decision": ("GO" if go else "NO-GO") if valid else ("INVALID (would be GO)" if go else
                                                                 "INVALID (would be NO-GO)"),
            "valid_for_decision": bool(valid), "rule": GO_RULE, "n_boot": n_boot, "seed": seed, "notes": notes,
            "arm_check": arm_check, "provenance_check": prov_check, "info": data.info}


# ----------------------------------------------------------------------------------------------- main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def reference_shas() -> Dict:
    return {"cand_sha16": sha16(PACK / "cand.npy"), "variants_sha16": sha16(VARIANTS)}


def run(dirs: Dict[str, str], out: Optional[Path], arm_check: bool = True, allow_partial: bool = False,
        n_boot: int = N_BOOT, seed: int = SEED) -> Dict:
    import pandas as pd
    t0 = time.time()
    raw = {k: load_infer(Path(v)) for k, v in dirs.items() if v}
    done = np.ones(N_VAL, bool)
    for k, r in raw.items():
        if r["p16"].shape != (N_VAL, K16, 5) or r["p80"].shape != (N_VAL, K16 * NVAR, 5) or r["done"].shape != (N_VAL,):
            raise SystemExit(f"arm {k}: {r['dir']} shapes {r['p16'].shape} / {r['p80'].shape} / {r['done'].shape}")
        done &= r["done"]
    data = load_data(done)
    ptok = pd.read_parquet(PACK / "tokens.parquet").token.astype(str).to_numpy()
    for k, r in raw.items():
        if "tokens" in r and (len(r["tokens"]) != len(ptok) or not (r["tokens"] == ptok).all()):
            raise SystemExit(f"arm {k}: {r['dir']}/tokens.parquet order != packed {SPLIT}")
    checks, prov = None, None
    if arm_check:
        ref = reference_shas()
        checks = {k: check_arm_meta(k, r["meta"], ref) for k, r in raw.items()}
        bad = {k: v for k, v in checks.items() if v}
        if bad:
            raise SystemExit(f"arm checks failed (use --no-arm-check for a diagnostic, invalid run): {bad}")
        prov = provenance_check(raw)
        if prov["errors"]:
            raise SystemExit(f"provenance mismatch between arms (D5; --no-arm-check for a diagnostic, invalid run): "
                             f"{prov['errors']}")
    if len(data.lab) < N_VAL and not allow_partial:
        raise SystemExit(f"only {len(data.lab)} of {N_VAL} rows evaluable (--allow-partial for a diagnostic run)")
    rows = data.rows
    arms = {k: Arm(p16=np.asarray(r["p16"], np.float64)[rows], p80=np.asarray(r["p80"], np.float64)[rows],
                   meta=r["meta"], fc_metrics=r["fc_metrics"]) for k, r in raw.items()}
    res = decide(data, arms, n_boot, seed, arm_check=checks, prov_check=prov)
    res["inputs"] = {k: {"dir": r["dir"], "prob16_sha16": r["prob16_sha16"],
                         "meta": {kk: r["meta"].get(kk) for kk in ("teacher", "run", "which", "ckpt_sha16", "temporal",
                                                                   "privileged", "cand_sha16", "variants_sha16")}}
                     for k, r in raw.items()}
    res["created"] = kst()
    res["sec"] = round(time.time() - t0, 1)
    res["code_sha16"] = sha16(__file__)
    if out is not None:
        write_json(out, res)
    return res


def summary_lines(res: Dict) -> List[str]:
    A = res["arms"]
    out = [f"decision: {res['decision']}  (n {res['n']}, logs {res['n_logs']}, lead {res['n_lead']}, hazard pairs "
           f"{res['n_hazard_pairs']})"]
    for k in sorted(A):
        a = A[k]
        out.append(f"  arm {k}: PDMS {a['pdms']:.3f} (w_dac0 {a['pdms_wdac0']:.3f}, v2dac {a['pdms_v2dac']:.3f}, own "
                   f"{a['pdms_owndac']:.3f})  lead TTC AUC {a['lead']['ttc_auc']:.4f}  lead fail "
                   f"{a['lead']['ncttc_fail_pct']:.2f} %  a-1.0 pref {a['hazard_a-1.0']['pref']:.4f}")
    for pr, d in res["deltas"].items():
        x = d["pdms_decision"]
        out.append(f"  {pr}: dPDMS {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]  dpref "
                   f"{d['hazard_pref']['mean']:+.4f} [{d['hazard_pref']['lo']:+.4f}, {d['hazard_pref']['hi']:+.4f}]")
    for g, v in res["gates"].items():
        extra = ""
        if v.get("lo") is not None:
            extra = f" [{v['lo']:+.4f}, {v['hi']:+.4f}]"
        if g in ("G2", "G3"):
            extra += "  values " + ", ".join(f"{k} {x:.4f}" for k, x in sorted(v["values"].items()) if x is not None)
            extra += "  " + ", ".join(f"{pr} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                                      for pr, d in v["deltas"].items() if d)
        if g == "G1" and v.get("second_seed_required"):
            extra += "  SECOND SEED REQUIRED (within 0.15 of the threshold)"
        out.append(f"  {g} {'PASS' if v['pass'] else 'fail'}: {v['rule']} -> {v['value']:.4f}{extra}")
    for n in res.get("notes", []):
        out.append(f"  note: {n}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--d", default=None)
    ap.add_argument("--out", default=str(D3 / "decide" / "step1.json"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-arm-check", action="store_true", help="diagnostic only: result is valid_for_decision false")
    ap.add_argument("--allow-partial", action="store_true", help="diagnostic only: fewer than 18,179 rows")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not str(out.resolve()).startswith(str(D3.resolve())) and not str(out.resolve()).startswith("/tmp/"):
        raise SystemExit(f"--out must be under {D3} (or /tmp for tests)")
    if a.n_boot != N_BOOT or a.seed != SEED:
        print(f"[warn] n_boot {a.n_boot} / seed {a.seed} differ from the pre-registered {N_BOOT} / {SEED}")
    res = run({"a": a.a, "b": a.b, "c": a.c, "d": a.d}, out, arm_check=not a.no_arm_check,
              allow_partial=a.allow_partial, n_boot=a.n_boot, seed=a.seed)
    print("\n".join(summary_lines(res)))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
