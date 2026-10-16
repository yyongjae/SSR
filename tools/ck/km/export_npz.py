#!/usr/bin/env python
"""km: submitted trajectories of a pool selection -> npz {tokens, trajectory} for SSR-v2 data/logs/score_npz.sh
(standard navsim pdm_score PDMS + v2.2 EPDMS on every navtest token).

  export_npz.py --root <RUN>/eval/root --run <RUN name> --split navtest --out x.npz
                (--beta 0.5 --set all | --idx <idx_navtest.npy> --rows <rows_navtest.npy>) [--lat-mode on]

The selected column's laterally corrected trajectory (lat_mode 'on', = CK2's eval); --idx / --rows = a listwise
selection over the evaluated rows (listwise_train.py); a row outside them keeps CK2's column (none expected).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--set", default="all")
    ap.add_argument("--lat-mode", default="on")
    ap.add_argument("--idx", default="")
    ap.add_argument("--rows", default="")
    a = ap.parse_args(argv)
    root = Path(a.root)
    pk, inf = root / "packed" / a.split, root / "infer" / a.run / a.split
    toks = pd.read_parquet(pk / "tokens.parquet").token.astype(str).to_numpy()
    cand, lat = np.load(pk / "cand96.npy"), np.load(inf / "lat_traj.npy")
    prob = 1.0 / (1.0 + np.exp(-np.load(inf / "score_logit.npy").astype(np.float64)))
    idx = S2.select96(np.load(pk / "v2_final.npy"), np.load(pk / "v2_im.npy"), prob, np.load(pk / "valid96.npy"),
                      a.beta, S2.VARIANT_SETS[a.set])
    if a.idx:
        rows = np.load(a.rows)
        sel = np.load(a.idx)
        assert len(rows) == len(sel)
        print(f"listwise selection on {len(rows)} of {len(idx)} rows; differs from CK2 on {np.mean(idx[rows] != sel):.4f}")
        idx[rows] = sel
    traj, src = S2.final_traj(cand, lat, idx, a.lat_mode)
    traj = np.asarray(traj, np.float32)
    assert np.isfinite(traj).all(), "non-finite submitted trajectory"
    np.savez(a.out, tokens=toks, trajectory=traj, sel_idx=idx, checkpoint=str(root))
    print(f"{len(toks)} trajectories -> {a.out} (lat {np.mean(src == 'lat'):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
