#!/usr/bin/env python
"""Selection over the 96 rule pool + the K corrector edits (refiner v4 A) vs the 96 alone, on an eval root written by
tools/ck/e2e2/eval_e2e2.py (stages dump + label of a student with ck_e2e2.v4.corrector true).  CPU only.

  cd /workspace/yongjae/SSR-v4 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/v4/eval_edits.py \
      --run-dir <RUN> --split navtrain_val [--betas 0.25,0.5,0.75,1.0] [--lat-mode on] [--sel-w default]

Inputs (<RUN>/eval/root, eval_e2e.Paths layout): packed/<split>/{cand96, valid96, v2_final, v2_im, ok, tokens.parquet},
  infer/<RUN>/<split>/{score_logit [N, 96, 5], done, edit_score_logit [N, K, 5], edit_src [N, K]},
  labels/<split>/{pool96_<RUN>, lat96_<RUN>, edit_<RUN>} (label_cands: labels [N, *, 9], ok [N, *]).
Rule (per beta of the grid, variant set 'all', the same selection weights as eval_ck2):
  96:   select2.select96 -> chosen labels with the lateral mode (select2.chosen_labels; default 'on').
  96+K: the 96 scores (select2.pool_scores) and the edits' scores (ck_final of the edit's predicted probabilities
        with the v2 final / im of its source's top-16 parent, blended by beta like the variants) -> argmax over 96 + K;
        an edit pick submits the edit itself (its own lateral controls, no lateral head on top) -> edit labels.
Tokens: eval_ck2.token_mask & every edit labelled ok.  Output: <RUN>/eval/edits_<split>.json (+ a printed table):
  per beta PDMS 96 / 96+K, paired log bootstrap of the difference, edit pick rate, PDMS of the picked edits vs their
  sources, share of edits better / worse than their source.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.select import blend, ck_final  # noqa: E402
from tools.ck.e2e import eval_e2e as E1  # noqa: E402
from tools.ck.e2e2 import eval_ck2 as EV  # noqa: E402


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def evaluate(packed: Dict[str, np.ndarray], infer: Dict[str, np.ndarray], lab: Dict[str, np.ndarray], logs,
             betas, lat_mode: str = "on", sel_w="default", n_boot: int = 2000) -> Dict[str, Any]:
    """Arrays restricted to the evaluation tokens (see the module docstring for keys)."""
    P = Cn.LBL["pdms"]
    w = S2.resolve_w(sel_w)
    n, K = infer["edit_src"].shape
    ar = np.arange(n)
    prob96 = _sig(infer["score_logit"])
    probe = _sig(infer["edit_score_logit"])
    src = infer["edit_src"].astype(np.int64)
    par = src // S2.NV
    allow = S2.allowed_mask(packed["valid96"], S2.VARIANT_SETS["all"])
    out = {"n": int(n), "k": int(K), "lat_mode": lat_mode, "sel_w": list(map(float, w)), "rows": []}
    e_p = lab["edit"][..., P]
    s_p = lab["pool"][ar[:, None], src, P]
    out["edit_vs_src"] = {"better": float(np.mean(e_p > s_p + 1e-9)), "worse": float(np.mean(e_p < s_p - 1e-9)),
                          "mean_d_pdms": float(np.mean(e_p - s_p))}
    for beta in betas:
        s96 = S2.pool_scores(packed["v2_final"], packed["v2_im"], prob96, beta, w)
        s96 = np.where(allow & np.isfinite(s96), s96, -np.inf)
        idx96 = S2.select96(packed["v2_final"], packed["v2_im"], prob96, packed["valid96"], beta,
                            S2.VARIANT_SETS["all"], None, w=w)
        ch96 = S2.chosen_labels(lab["pool"], lab["lat"], idx96, lat_mode)
        v2f = np.take_along_axis(packed["v2_final"], par, 1)
        v2i = np.take_along_axis(packed["v2_im"], par, 1)
        se = blend(v2f, ck_final(probe, v2i, w), beta)
        se = np.where(np.isfinite(se), se, -np.inf)
        both = np.concatenate([s96, se], 1)
        j = both.argmax(1)
        is_e = j >= S2.G_K
        je = np.clip(j - S2.G_K, 0, K - 1)
        ch = np.where(is_e[:, None], lab["edit"][ar, je], S2.chosen_labels(lab["pool"], lab["lat"],
                                                                         np.where(is_e, idx96, j), lat_mode))
        d = EV.log_bootstrap(ch[:, P] - ch96[:, P], logs, n_boot)
        pe = is_e.mean()
        row = {"beta": float(beta), "pdms_96": float(ch96[:, P].mean()), "pdms_96k": float(ch[:, P].mean()),
               "d_pdms": d, "edit_pick_rate": float(pe),
               "edit_pick_pdms": float(ch[is_e, P].mean()) if is_e.any() else None,
               "edit_pick_src_pdms": float(lab["pool"][ar[is_e], src[ar[is_e], je[is_e]], P].mean())
               if is_e.any() else None}
        out["rows"].append(row)
    return out


def main(argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True, choices=["navtrain_val", "navtest"])
    ap.add_argument("--betas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--lat-mode", default="on", choices=S2.LAT_MODES)
    ap.add_argument("--sel-w", default="default")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    P = E1.Paths(Path(a.run_dir), a.split)
    import pandas as pd
    tdf = pd.read_parquet(P.packed / "tokens.parquet")
    n = len(tdf)
    ld = lambda d, k: np.load(d / f"{k}.npy", mmap_mode="r")
    if not (P.infer / "edit_traj.npy").is_file():
        raise SystemExit(f"{P.infer}: no edit_traj.npy (not a corrector student, or dump missing)")
    lab_pool, ok_pool = EV.load_labels(P.labels / f"pool96_{P.name}", n)
    lab_lat, ok_lat = EV.load_labels(P.labels / f"lat96_{P.name}", n)
    de = P.labels / f"edit_{P.name}"
    lab_e = np.asarray(np.load(de / "labels.npy", mmap_mode="r"), np.float64)
    ok_e = np.asarray(np.load(de / "ok.npy", mmap_mode="r"), bool)
    valid96 = np.asarray(ld(P.packed, "valid96"), bool)
    m = EV.token_mask(ld(P.packed, "ok"), ld(P.infer, "done"), valid96, ok_pool, ok_lat) & ok_e.all(1)
    rows = np.flatnonzero(m)
    packed = {"valid96": valid96[rows], "v2_final": np.asarray(ld(P.packed, "v2_final"), np.float64)[rows],
              "v2_im": np.asarray(ld(P.packed, "v2_im"), np.float64)[rows]}
    infer = {k: np.asarray(ld(P.infer, k))[rows] for k in ("score_logit", "edit_score_logit", "edit_src")}
    lab = {"pool": lab_pool[rows], "lat": lab_lat[rows], "edit": lab_e[rows]}
    res = evaluate(packed, infer, lab, tdf["log"].astype(str).to_numpy()[rows],
                   [float(x) for x in a.betas.split(",")], a.lat_mode, a.sel_w, a.bootstrap)
    res.update(run=str(a.run_dir), split=a.split, n_split=int(n), created=EV.kst())
    out = P.eval / f"edits_{a.split}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{a.split}: {res['n']} tokens, K {res['k']}; edits vs source: {res['edit_vs_src']}")
    for r in res["rows"]:
        d = r["d_pdms"]
        print(f"beta {r['beta']:.2f}: 96 {100 * r['pdms_96']:.2f}  96+K {100 * r['pdms_96k']:.2f}  "
              f"d {100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]  edit picks "
              f"{100 * r['edit_pick_rate']:.1f}%")
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
