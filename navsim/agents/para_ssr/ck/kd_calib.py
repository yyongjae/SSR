"""CK2 KD probability calibration (user decision 2026-10-08 ~22:35 KST): per teacher run and per KD key, Platt scaling
of the teacher's score LOGIT before the sigmoid and before the DET / MAP / mean KD combination (kd.combine_teacher).

  p_cal_k = sigmoid(a_k * z_k + b_k),   a_k > 0  (monotone: the per-key ranking / AUC of the teacher is unchanged)

Fit (tools/ck/e2e2/fit_kd_calib.py): minimise the soft-target BCE (= NLL, a proper scoring rule; NOT PDMS) of the
teacher's predictions against the labels the student is trained on (official NC / DAC / TTC labels, NC 0.5 -> soft
target; EP = ep_target.ck_targets with the teacher run's own ep_target), on the v2 r34 top-16 executed candidates of
navtrain_train tokens (CK 'cand' labels) -- the KD-like candidates the teacher never trained on.  Both teachers use the
train split (no ReSMap navtrain_val features).  Held-out check = log-level 2-fold split (fit on one half, NLL / ECE on
the other), then the final parameters are refitted on all fit tokens.

Keys: the KD keys (default ck_e2e2.kd_score_keys nc, dac, ep, ttc) whose KD source (constants.KD_SCORE_SOURCE) involves
this teacher: DET (arm T) nc, ep, ttc; MAP (arm M) dac, ep.  Every other key keeps the identity (a = 1, b = 0) and is
flagged 'not_fitted' (a * z + b is then exactly z, so the uncalibrated value is reproduced bit for bit).

Numerics: 2-parameter Newton (float64) on the convex logit-form BCE  mean(softplus(s) - y s), s = a z + b, with a tiny
ridge RIDGE * ((a - 1)^2 + b^2) (keeps the optimum finite on separable data; negligible otherwise) and the constraint
a >= A_MIN (convex objective: when the unconstrained optimum has a < A_MIN the constrained one is on a = A_MIN, refitted
in b alone).  ECE: 15 equal-width bins on [0, 1] (ece) and 15 equal-count bins (ece_mass), |mean p - mean y| weighted
by the bin share; soft labels allowed.

Online mode (OnlineKDCalib, ck_e2e2.kd_calib.mode 'online', user 2026-11-02 option B): the same per-key Platt model
fitted INSIDE the student run, without a file and without any other trained planner -- the pairs are the run's own
labelled candidates (warm-up: the GT-based KD candidates; on-policy: the previous generation's label set with labeler2
official labels), binned, all-reduced and refitted refits_per_epoch times per epoch (class docstring).

File (JSON, one per teacher run and ckpt, default <CK_DATA>/ck2/kd_calib/<run>__<which>.json): version, method, run
(absolute), run_name, which, arm, teacher, ckpt, ckpt_sha16, ep_target, keys (CK_KEYS), kd_keys, fitted_keys,
params {key: {a, b, status}}, metrics, data, inference, fit_options, code, created.  The student (online2.TeacherPair2)
and the launcher (launch_util2) refuse a file whose run / which / ckpt sha16 / ep_target / arm differ from the loaded
teacher, or that does not fit every KD key the teacher is a source of.
No tools imports (navsim side).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import CK_DATA, CK_KEYS, KD_SCORE_SOURCE

CALIB_VERSION = "ck2_kd_calib_v1"
METHOD = "platt_logit_nll"
RULE = ("p_cal = sigmoid(a_k * z_k + b_k), a_k > 0, per teacher run and key; fitted by minimising the soft-target BCE "
        "(NLL) against the student's label targets on navtrain_train v2 r34 top-16 candidates; identity for keys "
        "not used in KD")
TEACHER_OF_ARM = {"T": "det", "M": "map"}
ARM_OF_TEACHER = {v: k for k, v in TEACHER_OF_ARM.items()}
KD_KEYS_DEFAULT = ("nc", "dac", "ep", "ttc")            # = CKE2E2Config.kd_score_keys default (no comfort KD)
A_MIN = 1e-3
RIDGE = 1e-6
N_BINS = 15
STATUS = ("fitted", "not_fitted")
KD_CALIB_ROOT = Path(CK_DATA) / "ck2" / "kd_calib"


def calib_path(run: str, which: str = "last", root=None) -> Path:
    """default file of a teacher run: <root or KD_CALIB_ROOT>/<run name>__<which>.json"""
    return Path(root or KD_CALIB_ROOT) / f"{Path(str(run)).name}__{which}.json"


def fit_keys(arm: str, kd_keys: Sequence[str] = KD_KEYS_DEFAULT) -> list:
    """KD keys whose KD target uses this teacher (KD_SCORE_SOURCE 'det' / 'map' / 'mean'), CK_KEYS order.
    T -> [nc, ep, ttc], M -> [dac, ep] for the default KD keys."""
    if arm not in TEACHER_OF_ARM:
        raise ValueError(f"arm must be one of {tuple(TEACHER_OF_ARM)}, got {arm!r}")
    t = TEACHER_OF_ARM[arm]
    bad = [k for k in kd_keys if k not in CK_KEYS]
    if bad:
        raise ValueError(f"unknown KD key(s) {bad} (CK_KEYS {list(CK_KEYS)})")
    return [k for k in CK_KEYS if k in kd_keys and KD_SCORE_SOURCE[k] in (t, "mean")]


# ----------------------------------------------------------------------------------------------- numerics
def _sig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _clean(z, y) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    if z.shape != y.shape:
        raise ValueError(f"z {z.shape} and y {y.shape} differ")
    m = np.isfinite(z) & np.isfinite(y)
    y = y[m]
    if len(y) and (y.min() < 0.0 or y.max() > 1.0):
        raise ValueError(f"labels must lie in [0, 1], got [{y.min()}, {y.max()}]")
    return z[m], y


def bce_logit(z, y, a: float = 1.0, b: float = 0.0) -> float:
    """mean soft-target BCE of sigmoid(a z + b) vs y, exact logit form mean(softplus(s) - y s) (float64)."""
    z, y = _clean(z, y)
    if not len(z):
        return float("nan")
    s = a * z + b
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


def ece(p, y, n_bins: int = N_BINS, mode: str = "width") -> float:
    """expected calibration error: sum_bins (n_b / N) |mean p_b - mean y_b|; 'width' = equal-width bins on [0, 1],
    'mass' = equal-count bins (sorted by p).  Soft labels allowed."""
    p, y = _clean(p, y)
    n = len(p)
    if not n:
        return float("nan")
    if mode == "width":
        idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
        cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
        sp = np.bincount(idx, weights=p, minlength=n_bins)
        sy = np.bincount(idx, weights=y, minlength=n_bins)
        nz = cnt > 0
        return float(np.sum(np.abs(sp[nz] - sy[nz])) / n)
    if mode == "mass":
        o = np.argsort(p, kind="stable")
        tot = 0.0
        for part in np.array_split(o, min(n_bins, n)):
            if len(part):
                tot += abs(float(p[part].sum() - y[part].sum()))
        return float(tot / n)
    raise ValueError(f"ece mode must be width | mass, got {mode!r}")


def summarize(z, y, a: float = 1.0, b: float = 0.0) -> Dict[str, float]:
    """NLL / ECE / mean predicted vs mean label of p = sigmoid(a z + b) (a = 1, b = 0: the raw teacher)."""
    z, y = _clean(z, y)
    if not len(z):
        return {"n": 0}
    p = _sig(a * z + b)
    return {"n": int(len(z)), "nll": bce_logit(z, y, a, b), "ece": ece(p, y), "ece_mass": ece(p, y, mode="mass"),
            "mean_pred": float(p.mean()), "mean_label": float(y.mean())}


def nll_const(y) -> float:
    """BCE of the constant prediction mean(y) (the label-prior baseline)."""
    y = np.asarray(y, np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if not len(y):
        return float("nan")
    q = float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
    return float(-np.mean(y * math.log(q) + (1 - y) * math.log(1 - q)))


def _newton(z, y, th, free, ridge, max_iter, tol, w=None):
    """Damped Newton on f(a, b) = mean(softplus(az + b) - y (az + b)) + ridge ((a - 1)^2 + b^2) over the `free`
    coordinates (backtracking Armijo).  w (optional, >= 0, same shape as z): sample weights, the means are then
    weighted means (w None = the unweighted path, unchanged).  -> (theta, f, iterations, converged)."""
    ctr = np.array([1.0, 0.0])
    if w is None:
        mean = np.mean
    else:
        wn = np.asarray(w, np.float64) / float(np.sum(w))

        def mean(x):
            return float(np.sum(wn * x))

    def f(t):
        s = t[0] * z + t[1]
        return float(mean(np.logaddexp(0.0, s) - y * s) + ridge * np.sum((t - ctr) ** 2))

    th = np.array(th, np.float64)
    fv = f(th)
    free = list(free)
    for it in range(1, max_iter + 1):
        s = th[0] * z + th[1]
        p = _sig(s)
        r = p - y
        h = p * (1.0 - p)
        g = np.array([mean(r * z), mean(r)]) + 2 * ridge * (th - ctr)
        H = np.array([[mean(h * z * z), mean(h * z)], [mean(h * z), mean(h)]]) + 2 * ridge * np.eye(2)
        gf, Hf = g[free], H[np.ix_(free, free)]
        try:
            d = -np.linalg.solve(Hf + 1e-12 * np.eye(len(free)), gf)
        except np.linalg.LinAlgError:
            d = -gf
        step, ok = 1.0, False
        for _ in range(60):
            cand = th.copy()
            cand[free] = th[free] + step * d
            fc = f(cand)
            if np.isfinite(fc) and fc <= fv + 1e-4 * step * float(gf @ d):
                ok = True
                break
            step *= 0.5
        if not ok:
            return th, fv, it, True                      # no descent possible: at the optimum to numerical precision
        dec = fv - fc
        th, fv = cand, fc
        if np.max(np.abs(step * d)) < 1e-10 or dec < tol:
            return th, fv, it, True
    return th, fv, max_iter, False


def fit_platt(z, y, a_min: float = A_MIN, ridge: float = RIDGE, max_iter: int = 100,
              tol: float = 1e-13) -> Dict[str, Any]:
    """Platt parameters (a, b) minimising the soft-target BCE of sigmoid(a z + b) vs y (labels in [0, 1]), a >= a_min.
    -> {a, b, nll (data term at the optimum), iters, converged, at_bound, n}."""
    z, y = _clean(z, y)
    if len(z) < 2:
        raise ValueError(f"fit_platt needs >= 2 finite samples, got {len(z)}")
    th, _, it, conv = _newton(z, y, (1.0, 0.0), (0, 1), ridge, max_iter, tol)
    at_bound = False
    if not th[0] >= a_min:
        th, _, it2, conv = _newton(z, y, (a_min, th[1]), (1,), ridge, max_iter, tol)
        th[0] = a_min
        it += it2
        at_bound = True
    a, b = float(th[0]), float(th[1])
    return {"a": a, "b": b, "nll": bce_logit(z, y, a, b), "iters": int(it), "converged": bool(conv),
            "at_bound": at_bound, "n": int(len(z))}


def fit_platt_weighted(z, y, w, a_min: float = A_MIN, ridge: float = RIDGE, max_iter: int = 100,
                       tol: float = 1e-13) -> Dict[str, Any]:
    """fit_platt with sample weights w >= 0 (binned data: z = bin centre, y = mean label of the bin, w = its count).
    Samples with w <= 0 or non-finite z / y / w are dropped.  -> {a, b, nll (weighted data term), iters, converged,
    at_bound, n (= sum of the kept weights)}."""
    z = np.asarray(z, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)
    if not z.shape == y.shape == w.shape:
        raise ValueError(f"z {z.shape}, y {y.shape} and w {w.shape} differ")
    m = np.isfinite(z) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    z, y, w = z[m], y[m], w[m]
    if len(y) and (y.min() < 0.0 or y.max() > 1.0):
        raise ValueError(f"labels must lie in [0, 1], got [{y.min()}, {y.max()}]")
    if len(z) < 2:
        raise ValueError(f"fit_platt_weighted needs >= 2 samples with weight > 0, got {len(z)}")
    th, _, it, conv = _newton(z, y, (1.0, 0.0), (0, 1), ridge, max_iter, tol, w=w)
    at_bound = False
    if not th[0] >= a_min:
        th, _, it2, conv = _newton(z, y, (a_min, th[1]), (1,), ridge, max_iter, tol, w=w)
        th[0] = a_min
        it += it2
        at_bound = True
    a, b = float(th[0]), float(th[1])
    return {"a": a, "b": b, "nll": binned_nll(z, w * y, w, a, b), "iters": int(it), "converged": bool(conv),
            "at_bound": at_bound, "n": float(w.sum())}


def binned_nll(c, sy, n, a: float = 1.0, b: float = 0.0) -> float:
    """soft-target BCE of sigmoid(a c + b) on binned data: c bin centres, sy label sums, n counts (per bin)."""
    c, sy, n = (np.asarray(x, np.float64).reshape(-1) for x in (c, sy, n))
    N = float(n.sum())
    if N <= 0:
        return float("nan")
    s = a * c + b
    return float(np.sum(n * np.logaddexp(0.0, s) - sy * s) / N)


def binned_mean_pred(c, n, a: float = 1.0, b: float = 0.0) -> float:
    c, n = (np.asarray(x, np.float64).reshape(-1) for x in (c, n))
    N = float(n.sum())
    return float(np.sum(n * _sig(a * c + b)) / N) if N > 0 else float("nan")


def heldout_2fold(z, y, fold, **kw) -> Dict[str, Any]:
    """fold [M] in {0, 1} (log-level split): fit on fold f, predict the other fold, for f = 0, 1 -> out-of-fold
    predictions over all samples; NLL / ECE of the raw teacher (before) and of the out-of-fold Platt (after).
    A fold without samples -> {'skipped': reason}."""
    z = np.asarray(z, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    fold = np.asarray(fold).reshape(-1).astype(np.int64)
    m = np.isfinite(z) & np.isfinite(y)
    z, y, fold = z[m], y[m], fold[m]
    if not ((fold == 0).sum() >= 2 and (fold == 1).sum() >= 2):
        return {"skipped": f"a fold has < 2 samples (fold sizes {int((fold == 0).sum())} / {int((fold == 1).sum())})"}
    s_oof = np.empty_like(z)
    folds = []
    for f in (0, 1):
        tr, te = fold == f, fold != f
        r = fit_platt(z[tr], y[tr], **kw)
        s_oof[te] = r["a"] * z[te] + r["b"]
        folds.append({"fit_fold": f, "a": r["a"], "b": r["b"], "n_fit": int(tr.sum()), "n_eval": int(te.sum()),
                      "nll_before": bce_logit(z[te], y[te]), "nll_after": bce_logit(z[te], y[te], r["a"], r["b"]),
                      "nll_const_fit": _nll_q(y[te], float(np.mean(y[tr])))})
    before = summarize(z, y)
    after = summarize(s_oof, y)                        # s_oof is already a logit: a = 1, b = 0
    return {"n": int(len(z)), "before": before, "after": after, "nll_const": nll_const(y), "folds": folds}


def _nll_q(y: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 1e-12, 1 - 1e-12))
    return float(-np.mean(y * math.log(q) + (1 - y) * math.log(1 - q)))


def log_folds(logs, seed: int = 0) -> np.ndarray:
    """log-level 2-fold assignment: the unique logs in a seeded random order, first half -> 0, rest -> 1.
    logs [T] (one entry per token) -> fold [T] int64."""
    logs = np.asarray(logs).astype(str)
    u = np.unique(logs)
    perm = np.random.default_rng(seed).permutation(len(u))
    f_of = {u[i]: int(r >= (len(u) + 1) // 2) for r, i in enumerate(perm)}
    return np.asarray([f_of[x] for x in logs], np.int64)


# ----------------------------------------------------------------------------------------------- apply
def apply_platt(logit, a, b):
    """logit [..., 5] (CK_KEYS) -> a * logit + b per key (torch: a, b moved to the logit's device / dtype; numpy:
    float32 result for float32 input).  Identity keys (a = 1, b = 0) give exactly the input values."""
    try:
        import torch
        if isinstance(logit, torch.Tensor):
            a_t = torch.as_tensor(a, dtype=logit.dtype, device=logit.device)
            b_t = torch.as_tensor(b, dtype=logit.dtype, device=logit.device)
            return logit * a_t + b_t
    except ImportError:   # pragma: no cover
        pass
    x = np.asarray(logit)
    dt = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    return (x * np.asarray(a, dt) + np.asarray(b, dt)).astype(dt)


# ----------------------------------------------------------------------------------------------- file
def file_sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_record(*, run: str, which: str, arm: str, ckpt: str, ckpt_sha16: str, ep_target: str,
                 params: Dict[str, Dict[str, Any]], kd_keys: Sequence[str] = KD_KEYS_DEFAULT,
                 **extra) -> Dict[str, Any]:
    """the calibration record (validated): params {key: {a, b}} for the FITTED keys only (identity elsewhere);
    extra = metrics / data / inference / fit_options / code / created ..."""
    from .ep_target import check_ep_target
    fk = fit_keys(arm, kd_keys)
    unknown = sorted(set(params) - set(CK_KEYS))
    if unknown:
        raise ValueError(f"params for unknown key(s) {unknown}")
    extra_keys = sorted(set(params) - set(fk))
    if extra_keys:
        raise ValueError(f"params for key(s) {extra_keys} that are not KD keys of teacher {TEACHER_OF_ARM[arm]} "
                         f"(fitted keys {fk})")
    missing = [k for k in fk if k not in params]
    if missing:
        raise ValueError(f"no params for the KD key(s) {missing} of teacher {TEACHER_OF_ARM[arm]}")
    P = {}
    for k in CK_KEYS:
        if k in fk:
            P[k] = {"a": float(params[k]["a"]), "b": float(params[k]["b"]), "status": "fitted"}
        else:
            P[k] = {"a": 1.0, "b": 0.0, "status": "not_fitted"}
    rec = {"version": CALIB_VERSION, "method": METHOD, "rule": RULE, "run": str(Path(str(run)).absolute()),
           "run_name": Path(str(run)).name, "which": str(which), "arm": arm, "teacher": TEACHER_OF_ARM[arm],
           "ckpt": str(ckpt), "ckpt_sha16": str(ckpt_sha16), "ep_target": check_ep_target(ep_target),
           "keys": list(CK_KEYS), "kd_keys": [k for k in CK_KEYS if k in kd_keys], "fitted_keys": fk, "params": P}
    rec.update(extra)
    validate_record(rec)
    return rec


def validate_record(rec: Dict[str, Any], where: str = "") -> None:
    """schema check of a calibration record; ValueError on anything malformed."""
    w = f"{where}: " if where else ""
    if not isinstance(rec, dict):
        raise ValueError(f"{w}calibration record must be a mapping")
    if rec.get("version") != CALIB_VERSION or rec.get("method") != METHOD:
        raise ValueError(f"{w}version / method {rec.get('version')!r} / {rec.get('method')!r} != "
                         f"{CALIB_VERSION!r} / {METHOD!r}")
    for k in ("run", "which", "arm", "teacher", "ckpt_sha16", "ep_target", "params", "keys", "fitted_keys"):
        if k not in rec:
            raise ValueError(f"{w}missing field {k!r}")
    if rec["arm"] not in TEACHER_OF_ARM or rec["teacher"] != TEACHER_OF_ARM[rec["arm"]]:
        raise ValueError(f"{w}arm / teacher {rec['arm']!r} / {rec['teacher']!r} inconsistent")
    if list(rec["keys"]) != list(CK_KEYS):
        raise ValueError(f"{w}keys {rec['keys']} != CK_KEYS {list(CK_KEYS)}")
    P = rec["params"]
    if set(P) != set(CK_KEYS):
        raise ValueError(f"{w}params keys {sorted(P)} != CK_KEYS")
    fitted = []
    for k in CK_KEYS:
        p = P[k]
        a, b, st = p.get("a"), p.get("b"), p.get("status")
        if st not in STATUS:
            raise ValueError(f"{w}params.{k}.status {st!r} not in {STATUS}")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a)
                and math.isfinite(b) and a > 0):
            raise ValueError(f"{w}params.{k}: need finite a > 0 and finite b, got a={a!r} b={b!r}")
        if st == "not_fitted" and (a != 1.0 or b != 0.0):
            raise ValueError(f"{w}params.{k} is 'not_fitted' but not the identity (a={a}, b={b})")
        if st == "fitted":
            fitted.append(k)
    if fitted != list(rec["fitted_keys"]):
        raise ValueError(f"{w}fitted_keys {rec['fitted_keys']} != keys with status 'fitted' {fitted}")


def load_calib(path) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.is_file():
        raise FileNotFoundError(f"KD calibration file {p} missing")
    rec = json.loads(p.read_text())
    validate_record(rec, str(p))
    return rec


def _same_path(a: str, b: str) -> bool:
    return Path(str(a)).resolve() == Path(str(b)).resolve()


def check_calib(rec: Dict[str, Any], *, run: str, which: str, ckpt_sha16: Optional[str], ep_target: str, arm: str,
                kd_keys: Optional[Sequence[str]] = None, where: str = "") -> None:
    """refuse (ValueError) a record that was not fitted for exactly this teacher: arm, run (resolved path), which,
    ckpt sha16, ep_target, and every KD key (kd_keys; None = KD_KEYS_DEFAULT) this teacher is a source of must be
    fitted."""
    from .ep_target import check_ep_target
    w = f"{where}: " if where else ""
    validate_record(rec, where)
    if rec["arm"] != arm:
        raise ValueError(f"{w}calibration is for arm {rec['arm']!r} ({rec['teacher']}), the teacher is arm {arm!r}")
    if not _same_path(rec["run"], run):
        raise ValueError(f"{w}calibration run {rec['run']} != teacher run {run}")
    if str(rec["which"]) != str(which):
        raise ValueError(f"{w}calibration which {rec['which']!r} != teacher which {which!r}")
    if not ckpt_sha16 or str(rec["ckpt_sha16"]) != str(ckpt_sha16):
        raise ValueError(f"{w}calibration ckpt sha16 {rec['ckpt_sha16']} != the loaded teacher's {ckpt_sha16} "
                         f"(checkpoint changed: refit with fit_kd_calib.py)")
    if check_ep_target(rec["ep_target"]) != check_ep_target(ep_target):
        raise ValueError(f"{w}calibration ep_target {rec['ep_target']!r} != the teacher's ep_target {ep_target!r}")
    need = fit_keys(arm, list(kd_keys) if kd_keys is not None else list(KD_KEYS_DEFAULT))
    miss = [k for k in need if rec["params"][k]["status"] != "fitted"]
    if miss:
        raise ValueError(f"{w}KD key(s) {miss} of teacher {TEACHER_OF_ARM[arm]} are not fitted in the calibration "
                         f"(fitted {rec['fitted_keys']}): refit with fit_kd_calib.py --kd-keys")


def calib_vectors(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(a [5], b [5]) float32 in CK_KEYS order."""
    a = np.asarray([rec["params"][k]["a"] for k in CK_KEYS], np.float32)
    b = np.asarray([rec["params"][k]["b"] for k in CK_KEYS], np.float32)
    return a, b


def load_teacher_calib(path, *, run: str, which: str, ckpt_sha16: Optional[str], ep_target: str, arm: str,
                       kd_keys: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """load + check_calib -> (a [5], b [5] float32, info for teachers.json / logs)."""
    rec = load_calib(path)
    check_calib(rec, run=run, which=which, ckpt_sha16=ckpt_sha16, ep_target=ep_target, arm=arm, kd_keys=kd_keys,
                where=str(path))
    a, b = calib_vectors(rec)
    held = {}
    for k in rec["fitted_keys"]:
        h = ((rec.get("metrics") or {}).get(k) or {}).get("heldout") or {}
        if "before" in h and "after" in h:
            held[k] = {"nll_before": h["before"].get("nll"), "nll_after": h["after"].get("nll"),
                       "ece_before": h["before"].get("ece"), "ece_after": h["after"].get("ece")}
    info = {"path": str(path), "sha16": file_sha16(path), "run": rec["run"], "which": rec["which"],
            "arm": rec["arm"], "ckpt_sha16": rec["ckpt_sha16"], "ep_target": rec["ep_target"],
            "created": rec.get("created"), "fitted_keys": list(rec["fitted_keys"]),
            "params": {k: [rec["params"][k]["a"], rec["params"][k]["b"]] for k in CK_KEYS}, "heldout": held}
    return a, b, info


# ----------------------------------------------------------------------------------------------- online (mode B)
ONLINE_VERSION = "ck2_kd_calib_online_v1"
CALIB_TEACHERS = ("det", "map")


class OnlineKDCalib:
    """KD calibration fitted INSIDE the student run (ck_e2e2.kd_calib.mode 'online', user 2026-11-02 option B): no
    calibration file, no other trained planner.  The pairs are (raw teacher score logit z, the student's own label
    target y) on candidates that carry official labels in the run itself:
      - warm-up / warmup_record: the step's KD candidates = the label set (GT-based raw anchors + rule variants);
      - on-policy: the label set G of the student's own previous generation (16 identities + variants, labeler2
        official labels; only rows with a generation, G_src prev / older, not the GT-based fallback), scored by the
        teachers in an extra no-grad pass.
    Same model as the file mode: p = sigmoid(a_k z_k + b_k) per teacher and fitted key (DET nc / ep / ttc, MAP dac /
    ep for the default KD keys; identity elsewhere), soft-target BCE (NLL) fit with a >= A_MIN.
    Accumulation: z binned on [-zmax, zmax] (bins equal-width bins, clipped at the ends), per bin the count and the
    label sum (exact for the BCE objective up to the bin width) -> one fixed-size float64 tensor per rank.
    Refit schedule: refits_per_epoch windows per epoch (slot = floor(epoch_frac * refits_per_epoch), identical on every
    rank); at the first micro-batch of a new slot the window is all-reduced (sum over ranks), folded into an EMA
    (decay exp(-1 / (refits_per_epoch * ema_epochs)) per window), and every fitted key whose EMA count >= min_n is
    refitted; the new (a, b) apply from that micro-batch on (lag of one window, like the labeler's label_lag).
    reset_onpolicy: the first on-policy window replaces the warm-up EMA (the candidate distribution changes there).
    Before the first refit the parameters are the identity (raw teacher).  An empty window (e.g. the partial window
    lost on a resume, since only rank 0's state is checkpointed) leaves the EMA and the parameters unchanged.
    Every refit record carries the window's NLL / mean prediction of the raw teacher and of the parameters that were in
    use during the window (fitted before it: out-of-sample) and of the new parameters (in-sample)."""

    def __init__(self, kd_keys: Sequence[str] = KD_KEYS_DEFAULT, refits_per_epoch: int = 10, ema_epochs: float = 1.0,
                 bins: int = 4096, zmax: float = 20.0, min_n: float = 2000.0, reset_onpolicy: bool = True):
        self.kd_keys = [k for k in CK_KEYS if k in kd_keys]
        self.fit = {t: fit_keys(ARM_OF_TEACHER[t], self.kd_keys) for t in CALIB_TEACHERS}
        self.R = int(refits_per_epoch)
        self.ema_epochs = float(ema_epochs)
        self.nb = int(bins)
        self.zmax = float(zmax)
        self.min_n = float(min_n)
        self.reset_onpolicy = bool(reset_onpolicy)
        if self.R < 1 or not self.ema_epochs > 0 or self.nb < 16 or not self.zmax > 0 or not self.min_n >= 2:
            raise ValueError(f"OnlineKDCalib: need refits_per_epoch >= 1, ema_epochs > 0, bins >= 16, zmax > 0, "
                             f"min_n >= 2; got {self.R}, {self.ema_epochs}, {self.nb}, {self.zmax}, {self.min_n}")
        self.decay = math.exp(-1.0 / (self.R * self.ema_epochs))
        self.width = 2.0 * self.zmax / self.nb
        self.centers = (np.arange(self.nb, dtype=np.float64) + 0.5) * self.width - self.zmax
        self.key_idx = {t: [CK_KEYS.index(k) for k in self.fit[t]] for t in CALIB_TEACHERS}
        self.ema = np.zeros((len(CALIB_TEACHERS), len(CK_KEYS), self.nb, 2), np.float64)
        self.a = np.ones((len(CALIB_TEACHERS), len(CK_KEYS)), np.float64)
        self.b = np.zeros((len(CALIB_TEACHERS), len(CK_KEYS)), np.float64)
        self.slot: Optional[int] = None
        self.ema_phase: Optional[str] = None
        self.n_refits = 0
        self.version = 0                      # bumped whenever (a, b) change (TeacherPair2 re-reads them then)
        self.last: Dict[str, Any] = {}
        self._win = None                      # torch float64 [2, 5 * nb, 2] (count, label sum) of this rank's window

    # ------------------------------------------------------------------------------------------ accumulate
    def _win_t(self, device):
        import torch
        if self._win is None:
            self._win = torch.zeros(len(CALIB_TEACHERS), len(CK_KEYS) * self.nb, 2, dtype=torch.float64,
                                    device=device)
        elif self._win.device != torch.device(device):
            self._win = self._win.to(device)
        return self._win

    def add(self, teacher: str, z, y, mask) -> int:
        """z [..., 5] RAW (uncalibrated) teacher score logits, y [..., 5] label targets in [0, 1], mask [...] bool
        (candidate usable for this teacher) -> pairs added (fitted keys x candidates with finite z and y)."""
        import torch
        ti = CALIB_TEACHERS.index(teacher)
        ks = self.key_idx[teacher]
        if not ks:
            return 0
        with torch.no_grad():
            m = mask.bool().reshape(-1)
            zz = z.detach().reshape(-1, len(CK_KEYS))[m][:, ks].double()
            yy = y.detach().to(zz.device).reshape(-1, len(CK_KEYS))[m][:, ks].double()
            if not zz.numel():
                return 0
            ok = torch.isfinite(zz) & torch.isfinite(yy) & (yy >= 0) & (yy <= 1)
            idx = ((zz.clamp(-self.zmax, self.zmax) + self.zmax) / self.width).floor().long().clamp(0, self.nb - 1)
            flat = idx + torch.as_tensor(ks, device=zz.device, dtype=torch.long)[None] * self.nb
            vals = torch.stack([torch.ones_like(yy), yy], -1)
            w = self._win_t(zz.device)
            w[ti].index_add_(0, flat[ok], vals[ok])
            return int(ok.sum())

    # ------------------------------------------------------------------------------------------ refit
    def slot_of(self, epoch_frac: float) -> int:
        return int(math.floor(float(epoch_frac) * self.R + 1e-9))

    def maybe_refit(self, epoch_frac: float, device, phase_of_epoch, reduce: bool = True) -> Optional[Dict[str, Any]]:
        """call once per training micro-batch on EVERY rank (before the teachers run): refit when the slot changed.
        phase_of_epoch: epoch -> 'warmup' | 'warmup_record' | 'onpolicy' (the phase of the finished window).
        -> the refit record, or None when no refit happened."""
        s = self.slot_of(epoch_frac)
        if self.slot is None:
            self.slot = s
            return None
        if s == self.slot:
            return None
        prev, self.slot = self.slot, s
        return self.refit(device, phase_of_epoch(prev // self.R), reduce=reduce, window_slot=prev)

    def refit(self, device, window_phase: str, reduce: bool = True, window_slot: Optional[int] = None
              ) -> Dict[str, Any]:
        import torch
        w = self._win_t(device)
        if reduce:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(w)
        win = w.detach().cpu().numpy().reshape(len(CALIB_TEACHERS), len(CK_KEYS), self.nb, 2).copy()
        w.zero_()
        rec: Dict[str, Any] = {"version": ONLINE_VERSION, "window_slot": window_slot, "window_phase": window_phase,
                               "refit": self.n_refits + 1}
        n_fit_keys = sum(win[i, j, :, 0].sum() for i, t in enumerate(CALIB_TEACHERS) for j in self.key_idx[t])
        if n_fit_keys <= 0:
            rec["skipped"] = "empty window"
            self.last = rec
            return rec
        reset = bool(self.reset_onpolicy and window_phase == "onpolicy" and self.ema_phase != "onpolicy")
        if reset:
            self.ema[:] = 0.0
        self.ema = self.decay * self.ema + win
        self.ema_phase = window_phase
        rec["reset"] = reset
        keys: Dict[str, Any] = {}
        changed = False
        for i, t in enumerate(CALIB_TEACHERS):
            for j in self.key_idx[t]:
                k = CK_KEYS[j]
                wn, wy = win[i, j, :, 0], win[i, j, :, 1]
                n, sy = self.ema[i, j, :, 0], self.ema[i, j, :, 1]
                a0, b0 = float(self.a[i, j]), float(self.b[i, j])
                r: Dict[str, Any] = {"n_win": float(wn.sum()), "n_ema": float(n.sum()), "a_prev": a0, "b_prev": b0}
                if wn.sum() > 0:
                    r.update(mean_label_win=float(wy.sum() / wn.sum()),
                             mean_pred_raw_win=binned_mean_pred(self.centers, wn),
                             mean_pred_prev_win=binned_mean_pred(self.centers, wn, a0, b0),
                             nll_raw_win=binned_nll(self.centers, wy, wn),
                             nll_prev_win=binned_nll(self.centers, wy, wn, a0, b0))
                r["fitted"] = False
                if n.sum() >= self.min_n:
                    nz = n > 0
                    try:
                        f = fit_platt_weighted(self.centers[nz], np.clip(sy[nz] / n[nz], 0.0, 1.0), n[nz])
                    except ValueError as e:
                        r["error"] = str(e)
                        f = None
                    if f is not None and math.isfinite(f["a"]) and math.isfinite(f["b"]) and f["a"] > 0:
                        self.a[i, j], self.b[i, j] = f["a"], f["b"]
                        changed = changed or (f["a"] != a0 or f["b"] != b0)
                        r.update(fitted=True, a=f["a"], b=f["b"], converged=f["converged"], at_bound=f["at_bound"],
                                 nll_ema=f["nll"])
                        if wn.sum() > 0:
                            r.update(mean_pred_new_win=binned_mean_pred(self.centers, wn, f["a"], f["b"]),
                                     nll_new_win=binned_nll(self.centers, wy, wn, f["a"], f["b"]))
                keys[f"{t}/{k}"] = r
        rec["keys"] = keys
        self.n_refits += 1
        if changed:
            self.version += 1
        self.last = rec
        return rec

    # ------------------------------------------------------------------------------------------ apply / state
    def params(self, teacher: str) -> Tuple[np.ndarray, np.ndarray]:
        """(a [5], b [5]) float32 of a teacher in CK_KEYS order (identity on the keys it does not calibrate)."""
        i = CALIB_TEACHERS.index(teacher)
        return self.a[i].astype(np.float32), self.b[i].astype(np.float32)

    def summary(self) -> Dict[str, Any]:
        return {"version": ONLINE_VERSION, "n_refits": self.n_refits, "slot": self.slot, "ema_phase": self.ema_phase,
                "params": {f"{t}/{CK_KEYS[j]}": [float(self.a[i, j]), float(self.b[i, j])]
                           for i, t in enumerate(CALIB_TEACHERS) for j in self.key_idx[t]},
                "n_ema": {f"{t}/{CK_KEYS[j]}": float(self.ema[i, j, :, 0].sum())
                          for i, t in enumerate(CALIB_TEACHERS) for j in self.key_idx[t]}}

    def options(self) -> Dict[str, Any]:
        return {"kd_keys": list(self.kd_keys), "refits_per_epoch": self.R, "ema_epochs": self.ema_epochs,
                "bins": self.nb, "zmax": self.zmax, "min_n": self.min_n, "reset_onpolicy": self.reset_onpolicy}

    def state_dict(self) -> Dict[str, Any]:
        """torch tensors / python scalars only (weights_only-loadable).  The current window is rank-local and is not
        saved (a resume drops it; see the class docstring)."""
        import torch
        return {"version": ONLINE_VERSION, "options": self.options(), "ema": torch.from_numpy(self.ema.copy()),
                "a": torch.from_numpy(self.a.copy()), "b": torch.from_numpy(self.b.copy()), "slot": self.slot,
                "ema_phase": self.ema_phase, "n_refits": int(self.n_refits), "pversion": int(self.version)}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        if sd.get("version") != ONLINE_VERSION:
            raise ValueError(f"online KD calibration state version {sd.get('version')!r} != {ONLINE_VERSION!r}")
        o = dict(sd.get("options") or {})
        if o != self.options():
            raise ValueError(f"online KD calibration state options {o} != this run's {self.options()}")
        self.ema = np.asarray(sd["ema"].cpu().numpy() if hasattr(sd["ema"], "cpu") else sd["ema"], np.float64).copy()
        self.a = np.asarray(sd["a"].cpu().numpy() if hasattr(sd["a"], "cpu") else sd["a"], np.float64).copy()
        self.b = np.asarray(sd["b"].cpu().numpy() if hasattr(sd["b"], "cpu") else sd["b"], np.float64).copy()
        self.slot = None if sd.get("slot") is None else int(sd["slot"])
        self.ema_phase = sd.get("ema_phase")
        self.n_refits = int(sd.get("n_refits", 0))
        self.version = int(sd.get("pversion", 0)) + 1          # force TeacherPair2 to re-read the parameters
        if self._win is not None:
            self._win.zero_()


__all__ = ["CALIB_VERSION", "METHOD", "RULE", "TEACHER_OF_ARM", "ARM_OF_TEACHER", "KD_KEYS_DEFAULT", "A_MIN", "RIDGE",
           "KD_CALIB_ROOT", "calib_path", "fit_keys", "bce_logit", "ece", "summarize", "nll_const", "fit_platt",
           "fit_platt_weighted", "binned_nll", "binned_mean_pred", "heldout_2fold", "log_folds", "apply_platt",
           "file_sha16", "build_record", "validate_record", "load_calib", "check_calib", "calib_vectors",
           "load_teacher_calib", "ONLINE_VERSION", "CALIB_TEACHERS", "OnlineKDCalib"]
