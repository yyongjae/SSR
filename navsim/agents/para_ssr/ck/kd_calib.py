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


def _newton(z, y, th, free, ridge, max_iter, tol):
    """Damped Newton on f(a, b) = mean(softplus(az + b) - y (az + b)) + ridge ((a - 1)^2 + b^2) over the `free`
    coordinates (backtracking Armijo).  -> (theta, f, iterations, converged)."""
    ctr = np.array([1.0, 0.0])

    def f(t):
        s = t[0] * z + t[1]
        return float(np.mean(np.logaddexp(0.0, s) - y * s) + ridge * np.sum((t - ctr) ** 2))

    th = np.array(th, np.float64)
    fv = f(th)
    free = list(free)
    for it in range(1, max_iter + 1):
        s = th[0] * z + th[1]
        p = _sig(s)
        r = p - y
        h = p * (1.0 - p)
        g = np.array([np.mean(r * z), np.mean(r)]) + 2 * ridge * (th - ctr)
        H = np.array([[np.mean(h * z * z), np.mean(h * z)], [np.mean(h * z), np.mean(h)]]) + 2 * ridge * np.eye(2)
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


__all__ = ["CALIB_VERSION", "METHOD", "RULE", "TEACHER_OF_ARM", "ARM_OF_TEACHER", "KD_KEYS_DEFAULT", "A_MIN", "RIDGE",
           "KD_CALIB_ROOT", "calib_path", "fit_keys", "bce_logit", "ece", "summarize", "nll_const", "fit_platt",
           "heldout_2fold", "log_folds", "apply_platt", "file_sha16", "build_record", "validate_record", "load_calib",
           "check_calib", "calib_vectors", "load_teacher_calib"]
