#!/usr/bin/env python
"""km 10/16: write the trajectory a pre-declared corridor selection rule picks (corr_select.py rules) for every
dumped token as {tokens, trajectory} npz, for the official PDMS / EPDMS scorers (SSR-v2/data/logs/score_npz.sh).

  $PY tools/corridor/select_npz.py --dump <dir> --split navtest --rule pred_multiply_t0.25 --out <npz>
  rules: v2 | model (the forward's corridor_select_rule) | pred_{replace,multiply}_t{0.25,0.5} | v3_veto_ttc* / v3_dacveto_ncsoft_ttc* (pred only)
         refiner v6c (dump --v6): v6 (mode B: R-J's switch ? tau_v6 : tau_base) | v6A (always tau_v6) | v6base (tau_base,
         read from the f32 dump, = model up to the f16 rounding of traj.npy)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

from tools.corridor.corr_select import ROOT, scores, scores_v3  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd
    d = Path(a.dump) / a.split
    D = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("traj", "sim", "im", "final")}
    if (d / "verdict.npz").exists():
        V = dict(np.load(d / "verdict.npz"))
        rows = V["rows"]
    else:                              # km 10/19 (C): no field verdicts; every dumped token
        assert a.rule in ("v2", "model", "v6", "v6A", "v6base"), f"{a.rule} needs verdict.npz (corr_verdict.py)"
        done = np.zeros(len(D["traj"]), bool)
        for f in d.glob("done_*.npy"):
            done |= np.load(f)
        rows = np.flatnonzero(done)
    if a.rule == "v2":
        ch = np.asarray(D["final"][rows]).argmax(1)
    elif a.rule in ("model", "v6", "v6A", "v6base"):   # km 10/17: the forward's own choice (corridor_select_rule)
        ch = np.asarray(np.load(d / "msel.npy", mmap_mode="r")[rows], np.int64)
    elif a.rule.startswith("pred_"):
        _, mode, t = a.rule.split("_")
        ch = scores(D, V, rows, "pred", mode, float(t[1:])).argmax(1)
    else:
        soft = "_dacveto_ncsoft_" in a.rule
        ts = a.rule.split("_ttc")[1]
        ch = scores_v3(D, V, rows, "pred", ts, nc_mode="soft" if soft else "hard").argmax(1)
    toks = pd.read_parquet(f"{ROOT}/packed/{a.split}/tokens.parquet").token.astype(str).to_numpy()
    if a.rule in ("v6", "v6A", "v6base"):
        base = np.asarray(np.load(d / "v6_base.npy", mmap_mode="r")[rows], np.float32)
        v6 = np.asarray(np.load(d / "v6_traj.npy", mmap_mode="r")[rows], np.float32)
        sw = np.asarray(np.load(d / "v6_switch.npy", mmap_mode="r")[rows]).astype(bool)
        sw = np.ones_like(sw) if a.rule == "v6A" else (np.zeros_like(sw) if a.rule == "v6base" else sw)
        traj = np.where(sw[:, None, None], v6, base)
        np.savez_compressed(a.out, tokens=toks[rows], trajectory=traj, rule=a.rule, dump=str(d))
        print(f"{a.rule}: {len(rows)} tokens -> {a.out} (tau_v6 submitted on {sw.mean() * 100:.1f}%)")
        return 0
    traj = np.stack([np.asarray(D["traj"][r], np.float32)[c] for r, c in zip(rows, ch)])
    np.savez_compressed(a.out, tokens=toks[rows], trajectory=traj, rule=a.rule, dump=str(d))
    print(f"{a.rule}: {len(rows)} tokens -> {a.out} (same choice as v2 "
          f"{(ch == np.asarray(D['final'][rows]).argmax(1)).mean() * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
