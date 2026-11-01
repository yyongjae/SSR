#!/usr/bin/env python
"""Mode-A deceleration liveness of an evaluated stage-T refiner run (PRESTATED_DECISION_RULE AMENDMENT 3, (1) and (3)).

  python tools/refiner/liveness.py --run <runs>/<run> --eval eval_train_fold0 [--min-liveness 0.01] [--out <json>]

Reads <run>/<eval>/pred.npz (eval_refiner.py predict: z_lon [N, K, 6] raw lon controls as float32, draft_valid [N, K],
family [N, K], tau0 / tau1 [N, K, 8, 3]); no model, no scores, CPU only, seconds.  --eval is REQUIRED (no default):
the liveness gate is evaluated on the train OOF fold before dev is touched; pointing it at eval_dev is an explicit act.

Per valid draft (draft_valid):
  live      = any c_i < 0 of the decoder's pre-clamp mode-A control points c = lon_c_from_q(lon_q_from_z(z_lon))
              (decoder.lon_live, float32 as in eval_refiner; equals (decode(...)['c_lon'] < 0).any(-1)).  A draft that
              is not live decodes with dv == 0 (no deceleration).
  zlon_pos  = all 6 raw z_lon > 0 (the run-1 diagnostic, DECISION_RUN1.md).
  short_m   = arc(tau0) - arc(tau1), arc = polyline length origin -> 8 poses (the 4 s arc length; the same formula as
              report/refiner_T/decision_diag/diag.py).  Ungated: tau1 is the decoded correction whatever p_g is.
Output (printed; written to --out only when given -- nothing is written into the run directory by default, so the
tool can be pointed at run-1 evidence read-only): liveness (fraction of valid drafts live),
frac_zlon_all_pos, frac_short_gt_0p5m, mean_short_m, n_valid, per family (decoder.FAMILY_NAME) the same fractions,
min_liveness and gate_pass = liveness >= min_liveness (AMENDMENT 3 threshold 1 %).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner.decoder import FAMILY_NAME, lon_live  # noqa: E402

SHORT_M = 0.5            # arc-length shortening threshold at 4 s [m]
MIN_LIVENESS = 0.01      # AMENDMENT 3: liveness >= 1 % in both arms


def arc_len(traj: np.ndarray) -> np.ndarray:
    """[..., 8, >=2] poses -> [...] polyline length from the origin through the 8 poses [m]."""
    xy = np.asarray(traj, np.float64)[..., :2]
    P = np.concatenate([np.zeros_like(xy[..., :1, :]), xy], -2)
    return np.linalg.norm(np.diff(P, axis=-2), axis=-1).sum(-1)


def _stats(live: np.ndarray, zpos: np.ndarray, short: np.ndarray) -> Dict:
    n = int(live.size)
    f = lambda x: float(x.mean()) if n else float("nan")
    return dict(n=n, liveness=f(live), n_live=int(live.sum()), frac_zlon_all_pos=f(zpos),
                frac_short_gt_0p5m=f(short > SHORT_M), mean_short_m=f(short))


def liveness_from_pred(P, min_liveness: float = MIN_LIVENESS) -> Dict:
    """P: mapping with z_lon [N, K, 6], draft_valid [N, K], family [N, K], tau0 / tau1 [N, K, 8, 3] -> result dict."""
    z = np.asarray(P["z_lon"])
    valid = np.asarray(P["draft_valid"]).astype(bool)
    live = lon_live(torch.as_tensor(z.astype(np.float32))).numpy()
    zpos = (z > 0).all(-1)
    short = arc_len(P["tau0"]) - arc_len(P["tau1"])
    res = _stats(live[valid], zpos[valid], short[valid])
    res["n_valid"] = res.pop("n")
    fam = np.asarray(P["family"])[valid]
    res["by_family"] = {FAMILY_NAME.get(int(c), str(int(c))): _stats(live[valid][fam == c], zpos[valid][fam == c],
                                                                     short[valid][fam == c])
                        for c in np.unique(fam)}
    res["short_threshold_m"] = SHORT_M
    res["min_liveness"] = float(min_liveness)
    res["gate_pass"] = bool(res["n_valid"] > 0 and res["liveness"] >= min_liveness)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (runs/<run>)")
    ap.add_argument("--eval", required=True,
                    help="evaluation subdirectory of the run holding pred.npz, e.g. eval_train_fold0 (no default: the "
                         "liveness gate precedes any dev evaluation)")
    ap.add_argument("--min-liveness", type=float, default=MIN_LIVENESS, help="gate threshold on the liveness fraction")
    ap.add_argument("--out", default=None, help="output json (default: print only, write nothing)")
    a = ap.parse_args(argv)
    d = Path(a.run) / a.eval
    with np.load(d / "pred.npz", allow_pickle=False) as P:
        res = liveness_from_pred({k: P[k] for k in ("z_lon", "draft_valid", "family", "tau0", "tau1")}, a.min_liveness)
    res = dict(run=str(a.run), eval=a.eval, **res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
