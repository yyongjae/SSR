#!/usr/bin/env python
"""km 10/19 (C): modules/exact_verdict.py on the fixed torch tracker vs the OFFICIAL per-candidate PDMS labels.
  $PY tools/corridor/check_exact.py <dump dir with navtest/{traj,status}.npy + official_all_ext.npz> <exact dir> [n]"""
import sys
from pathlib import Path

import numpy as np
import torch

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)
from navsim.agents.para_ssr.modules import exact_verdict as EV  # noqa: E402
from navsim.agents.para_ssr.modules.torch_tracker import track  # noqa: E402


def main():
    D, X = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.set_num_threads(4)
    A = np.load(f"{D}/official_all_ext.npz")
    keys, O, rows = list(A["keys"]), A["O"], A["rows"]
    T, S = (np.load(f"{D}/navtest/{k}.npy", mmap_mode="r") for k in ("traj", "status"))
    tab = {k: (np.load(f"{X}/navtest/{k}.npy", mmap_mode="r"), np.load(f"{X}/navtest/{k}_off.npy")) for k in EV.TABLES}
    ok = np.load(f"{X}/navtest/ok.npy")
    sel = [i for i, r in enumerate(rows) if ok[r]][: n or None]
    res = {k: [] for k in ("nc", "dac", "ttc", "progress", "comfort")}
    for j, i in enumerate(sel):
        r = rows[i]
        t = EV.pack_token(*(np.asarray(v[o[r]:o[r + 1]], np.float32) for v, o in tab.values()))
        t = {k: v[None] for k, v in t.items()}
        init = torch.cat([torch.tensor(np.asarray(S[r], np.float32)), torch.zeros(3)])[None].repeat(256, 1)
        st = track(torch.tensor(np.asarray(T[r], np.float64)), init.double(), full=True).float()[None]
        v = EV.exact_verdicts(st, t)
        for k in res:
            res[k].append(v[k][0].numpy())
        if j % 100 == 0:
            print(f"{j}/{len(sel)}", flush=True)
    off = {k: O[sel][..., keys.index(k)] for k in ("nc", "dac", "ttc", "comfort", "raw_progress")}
    for k, ok_ in (("nc", "nc"), ("dac", "dac"), ("ttc", "ttc"), ("comfort", "comfort")):
        m, o = np.stack(res[k]), off[ok_]
        mf, of = m < 1, o < 1
        print(f"{k:<8} agree {(mf == of).mean() * 100:7.3f}% | catch {mf[of].mean() * 100:6.2f}% | false flag {mf[~of].mean() * 100:6.3f}%"
              f" | exact value {(m == o).mean() * 100:7.3f}%")
    p, o = np.stack(res["progress"]), off["raw_progress"]
    print(f"progress |err| median {np.median(np.abs(p - o)):.4f} m, p99 {np.percentile(np.abs(p - o), 99):.4f} m, max {np.abs(p - o).max():.3f} m")


if __name__ == "__main__":
    main()
