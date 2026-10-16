#!/usr/bin/env python
"""km: selection rules + PDMS / EPDMS of critics dumped by critic_train.py --eval (CPU).

  $PY tools/ck/km/critic_eval.py --root <RUN>/eval/root --labv2 <labv2> --split navtrain_val \
      --single C=<dir> T=<dir> --pair F=<dirA>,<dirB> [--export <name> --out <npz dir>]

Every rule uses CK2's deployment (select2.select96: v2_final / v2_im blend, beta 0.5, variant set all) with the
critic probabilities p [N, 96, 5] replaced:
  single critic X       : p_X
  fold pair (A, B)      : avg  = (p_A + p_B) / 2                         (ensemble control)
                          min  = min(p_A, p_B) per key                    (pessimistic double estimate)
                          xA   = among the top-8 columns of A's deployed score, the best by B's deployed score
                          xB   = the same with A and B swapped               (select by one, verify by the other)
Scores: official PDMS of the submitted trajectory (lat96 labels) and exact one-stage EPDMS of the selected sequence
(epdms_offline, v2 labels), both vs CK2 with paired log bootstrap CIs; crit = mean(dPDMS, dEPDMS).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from tools.ck.km.epdms_offline import V2Labels  # noqa: E402
from tools.ck.km.ft_trunk import Pool, log  # noqa: E402

BETA = 0.5
P_PDMS = 6
TOPK = 8


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def load_prob(d: Path, split: str, rows):
    lg = np.load(Path(d) / f"logits_{split}.npy", mmap_mode="r")
    return sig(lg[rows])


def deployed(P: Pool, rows, prob):
    s = S2.pool_scores(P.v2_final[rows], P.v2_im[rows], prob, BETA)
    ok = S2.allowed_mask(P.valid[rows]) & np.isfinite(s)
    return np.where(ok, s, -np.inf)


def select(P, rows, prob):
    return S2.select96(P.v2_final[rows], P.v2_im[rows], prob, P.valid[rows], BETA)


def cross(sa, sb):
    """top-TOPK columns of sa, best by sb (ties / non-finite -> best by sa)."""
    top = np.argsort(-sa, 1)[:, :TOPK]
    vb = np.take_along_axis(sb, top, 1)
    j = np.argmax(vb, 1)
    return top[np.arange(len(sa)), j]


def score(P: Pool, V: V2Labels, rows, idx, base, n_boot=1000, seed=0):
    N = len(P.tokens)
    ar = np.arange(len(rows))
    lab = P.lab_lat[rows]
    p, p0 = lab[ar, idx, P_PDMS], lab[ar, base, P_PDMS]
    full, full0 = np.zeros(N, np.int64), np.zeros(N, np.int64)
    full[rows], full0[rows] = idx, base
    e, e0 = V.sequence_epdms(full, rows), V.sequence_epdms(full0, rows)
    logs = P.tokens.log.astype(str).to_numpy()[rows]
    u, inv = np.unique(logs, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = {"n": int(len(rows)), "pdms": float(np.mean(p)), "pdms_ck2": float(np.mean(p0)),
           "epdms": float(np.nanmean(e)), "epdms_ck2": float(np.nanmean(e0)), "changed": float(np.mean(idx != base))}
    for k, (a, b) in {"pdms": (p, p0), "epdms": (e, e0)}.items():
        ok = np.isfinite(a) & np.isfinite(b)
        ds, ns = np.bincount(inv[ok], (a - b)[ok], len(u)), np.bincount(inv[ok], minlength=len(u))
        bs = [ds[s].sum() / ns[s].sum() for s in (rng.integers(0, len(u), len(u)) for _ in range(n_boot))]
        out[f"ci_{k}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["d_pdms"], out["d_epdms"] = out["pdms"] - out["pdms_ck2"], out["epdms"] - out["epdms_ck2"]
    out["crit"] = 0.5 * (out["d_pdms"] + out["d_epdms"])
    for k, j in {"nc": 0, "dac": 1, "ep": 2, "ttc": 3}.items():
        out[k] = float(lab[ar, idx, j].mean())
    return out


def fmt(n, o):
    return (f"{n:10s} PDMS {100 * o['pdms']:.2f} ({100 * o['d_pdms']:+.2f} [{100 * o['ci_pdms'][0]:+.2f} "
            f"{100 * o['ci_pdms'][1]:+.2f}])  EPDMS {100 * o['epdms']:.2f} ({100 * o['d_epdms']:+.2f} "
            f"[{100 * o['ci_epdms'][0]:+.2f} {100 * o['ci_epdms'][1]:+.2f}])  crit {100 * o['crit']:+.3f}  "
            f"NC {100 * o['nc']:.2f} DAC {100 * o['dac']:.2f} EP {100 * o['ep']:.2f} TTC {100 * o['ttc']:.2f} "
            f"changed {o['changed']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--labv2", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--single", nargs="*", default=[], help="NAME=dir")
    ap.add_argument("--pair", nargs="*", default=[], help="NAME=dirA,dirB")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", nargs="*", default=[], help="rule names whose selection is saved (idx, rows)")
    a = ap.parse_args(argv)
    P = Pool(Path(a.root), a.run, a.split, need_pool=False)
    V = V2Labels(Path(a.labv2) / a.split)
    v2ok = V.meta.done.to_numpy() & (V.meta.err.to_numpy() == 0)
    rows = P.rows[v2ok[P.rows]]
    if len(rows) < len(P.rows):
        log(f"{a.split}: {len(P.rows) - len(rows)} rows without v2 labels excluded")
    base = select(P, rows, sig(P.logit[rows]))
    sels = {"CK2": base}
    for s in a.single:
        n, d = s.split("=")
        sels[n] = select(P, rows, load_prob(Path(d), a.split, rows))
    for s in a.pair:
        n, d = s.split("=")
        da, db = d.split(",")
        pa, pb = load_prob(Path(da), a.split, rows), load_prob(Path(db), a.split, rows)
        sels[f"{n}_avg"] = select(P, rows, 0.5 * (pa + pb))
        sels[f"{n}_min"] = select(P, rows, np.minimum(pa, pb))
        sa, sb = deployed(P, rows, pa), deployed(P, rows, pb)
        sels[f"{n}_A"], sels[f"{n}_B"] = select(P, rows, pa), select(P, rows, pb)
        sels[f"{n}_xA"], sels[f"{n}_xB"] = cross(sa, sb), cross(sb, sa)
    res = {}
    for n, idx in sels.items():
        res[n] = score(P, V, rows, idx, base)
        log(f"{a.split} " + fmt(n, res[n]))
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / f"rules_{a.split}.json").write_text(json.dumps(res, indent=1))
        for n in a.export:
            np.save(o / f"idx_{n}_{a.split}.npy", sels[n])
            np.save(o / f"rows_{n}_{a.split}.npy", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
