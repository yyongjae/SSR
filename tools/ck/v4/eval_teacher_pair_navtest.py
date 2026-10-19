#!/usr/bin/env python
"""navtest reference A for a DET + MAP teacher PAIR (the planner the KD student imitates): each key's probability from
the teacher that was trained on it (DET nc, ttc, comfort; MAP dac; EP = mean of the two teachers' EP probabilities,
both are EP sources of the KD), then the GT-free selection modes of eval_teacher_navtest_r34.select_modes on the v2
r34 top-16 (+ the 80 rule variants for the pool96 modes).  Only the '_a' trajectories (the pick as is, no lateral
head): their official labels already exist (packed r34 candidates + the variant labels), nothing is re-scored.
Two pairs are compared paired on the same tokens (log-clustered bootstrap, tools/ck/eval_ck.log_bootstrap).  CPU.

  cd /workspace/yongjae/SSR-v4 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 /venv/ssr/bin/python \
      tools/ck/v4/eval_teacher_pair_navtest.py --root <eval_teacher_navtest_r34 out-root> \
      --pair-a ck2T10,ck2M10 --pair-b v4T10_haz,v4M10_haz
Inputs: <root>/<teacher>/r34/{prob16, prob80, done}.npy (stage select), <root>/r34_variants/{valid80.npy, labels},
  CK_DATA/packed/navtest, CK_DATA/labels/navtest/cand.  Output: <root>/pair_<a>_vs_<b>.{json,md}.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS  # noqa: E402
from tools.ck.e2e2 import eval_teacher_navtest_r34 as R  # noqa: E402

DET_KEYS, MAP_KEYS = ("nc", "ttc", "comfort"), ("dac",)


def pair_prob(pd: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """[..., 5] DET / MAP probabilities -> the pair's [..., 5] (CK_KEYS order; EP = mean)."""
    out = np.empty_like(pd)
    for i, k in enumerate(CK_KEYS):
        out[..., i] = pd[..., i] if k in DET_KEYS else pm[..., i] if k in MAP_KEYS else 0.5 * (pd[..., i] + pm[..., i])
    return out


def main(argv=None):
    from tools.ck.eval_ck import lead_flags, log_bootstrap, row_metrics
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pair-a", required=True, help="det,map run names (baseline)")
    ap.add_argument("--pair-b", required=True, help="det,map run names")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    root = Path(a.root)
    vd = root / "r34_variants"
    pdir, tdf, pok = R.packed()
    N = len(tdf)
    pairs = {"a": a.pair_a.split(","), "b": a.pair_b.split(",")}
    done = np.ones(N, bool)
    probs = {}
    for key, (det, mp) in pairs.items():
        P = {}
        for t in (det, mp):
            td = root / t / "r34"
            done &= np.load(td / "done.npy")
            P[t] = (np.load(td / "prob16.npy"), np.load(td / "prob80.npy"))
        probs[key] = (pair_prob(P[det][0], P[mp][0]), pair_prob(P[det][1], P[mp][1]))
    labc, okc = R.load_lab(R.CK_DATA / "labels" / R.SPLIT / "cand", N, R.K16)
    labv, okv = R.load_lab(vd / "labels", N, R.K16 * R.NVAR)
    valid80 = np.load(vd / "valid80.npy")
    keep = pok & done & okc.all(1) & (okv | ~valid80).all(1)
    r = np.flatnonzero(keep)
    n = len(r)
    ar = np.arange(n)
    tokens, logs = tdf.token.to_numpy()[r], tdf.log.to_numpy()[r]
    lead_mask, _ = lead_flags(R.SPLIT, tokens)
    v2_final = np.asarray(np.load(pdir / "v2_final.npy", mmap_mode="r")[r], np.float64)
    v2_im = np.asarray(np.load(pdir / "v2_im.npy", mmap_mode="r")[r], np.float64)
    LC, LP = labc[r], R.to_pool(labc[r], labv[r])
    PD = Cn.LBL["pdms"]
    res = {"n": int(n), "n_split": int(N), "pairs": pairs, "keys": {"det": DET_KEYS, "map": MAP_KEYS, "ep": "mean"},
           "modes": {}}
    chosen = {}
    for key in ("a", "b"):
        p16, p80 = probs[key]
        picks = R.select_modes(p16[r].astype(np.float64), p80[r].astype(np.float64), valid80[r], v2_final, v2_im)
        for mode, idx in picks.items():
            lab = (LC if mode.startswith("r34") else LP)[ar, idx]
            chosen[(key, mode)] = lab
            m = row_metrics(lab, lead_mask)
            m["frac_changed"] = float(np.mean(idx != 0))
            if not mode.startswith("r34"):
                from navsim.agents.para_ssr.ck.select2 import VNAMES
                m["sel_share"] = {VNAMES[t]: float(np.mean(idx % R.NV == t)) for t in range(R.NV)}
            res["modes"].setdefault(mode, {})[key] = m
    v2 = row_metrics(LC[:, 0], lead_mask)
    res["v2"] = v2
    for mode in res["modes"]:
        d = chosen[("b", mode)][:, PD] - chosen[("a", mode)][:, PD]
        res["modes"][mode]["b_minus_a"] = log_bootstrap(d, logs, a.bootstrap)
        res["modes"][mode]["b_minus_a"]["better"] = float(np.mean(d > 1e-9))
        res["modes"][mode]["b_minus_a"]["worse"] = float(np.mean(d < -1e-9))
    name = f"pair_{'+'.join(pairs['a'])}_vs_{'+'.join(pairs['b'])}"
    (root / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    lines = [f"# teacher pair navtest reference A: {'+'.join(pairs['a'])} (a) vs {'+'.join(pairs['b'])} (b)", "",
             f"tokens {n} / {N}; keys: DET {', '.join(DET_KEYS)}, MAP {', '.join(MAP_KEYS)}, EP = mean; '_a' picks "
             f"(no lateral head); PDMS 0-100; v2 {100 * v2['pdms']:.2f}", "",
             "| mode | PDMS a | PDMS b | b - a [95% CI] | NC a / b | DAC a / b | EP a / b | TTC a / b | changed a / b |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, d in res["modes"].items():
        A, B, D = d["a"], d["b"], d["b_minus_a"]
        f = lambda k: f"{100 * A[k]:.2f} / {100 * B[k]:.2f}"
        lines.append(f"| {mode} | {100 * A['pdms']:.2f} | {100 * B['pdms']:.2f} | {100 * D['mean']:+.2f} "
                     f"[{100 * D['lo']:+.2f}, {100 * D['hi']:+.2f}] | {f('nc')} | {f('dac')} | {f('ep')} | {f('ttc')} | "
                     f"{100 * A['frac_changed']:.1f} / {100 * B['frac_changed']:.1f} |")
    (root / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
