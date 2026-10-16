#!/usr/bin/env python
"""km: label_v2.py shards vs an official run_pdm_score_from_trajectories CSV of a selection (reproduction check).

v2 env:  $PY tools/ck/km/check_v2_labels.py --shards <labv2>/navtest/shards --npz <export npz with sel_idx> --csv <csv>
For every token in the finished shards: the 8 sub-scores at the selected column vs the CSV row, and (frames whose
previous frame is also finished) the official two-frame EC recomputed from the stored states vs the CSV, and the final
EPDMS (epdms_offline formula) vs the CSV 'score'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tokens", required=True, help="packed tokens.parquet of the split")
    a = ap.parse_args(argv)
    from navsim.planning.script.run_pdm_score_one_stage import infer_start_adjacent_mapping
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    tok = pd.read_parquet(a.tokens)
    z = np.load(a.npz, allow_pickle=True)
    sel = dict(zip(z["tokens"].astype(str), z["sel_idx"]))
    csv = pd.read_csv(a.csv).set_index("token")
    rows, lab, st, start = [], [], [], []
    for p in sorted(Path(a.shards).glob("*.npz")):
        s = np.load(p)
        rows.append(s["row"]); lab.append(s["lab"]); st.append(s["states"]); start.append(s["start_time"])
    rows, lab, st, start = map(np.concatenate, (rows, lab, st, start))
    toks = tok.token.astype(str).to_numpy()[rows]
    j = np.array([sel[t] for t in toks])
    mine = lab[np.arange(len(rows)), j, :8].astype(np.float64)
    off = csv.loc[toks, list(COLS)].to_numpy(np.float64)
    d = np.abs(mine - off)
    print(f"{len(rows)} tokens; max |diff| per sub-score:", dict(zip(COLS, d.max(0).round(6))))
    df = pd.DataFrame({"token": toks, "log_name": tok.log.astype(str).to_numpy()[rows], "start_time": start,
                       "frame_type": 0})
    mp = infer_start_adjacent_mapping(df)
    pos = {t: i for i, t in enumerate(toks)}
    ec_m, ec_o, sc_m, sc_o = [], [], [], []
    for cur, prv in mp.items():
        i, k = pos[cur], pos[prv]
        n = round((start[i] - start[k]) / 0.1)
        c, q = st[i, j[i]].astype(np.float64)[:-n], st[k, j[k]].astype(np.float64)[n:]
        e = float(ego_is_two_frame_extended_comfort(c[None], q[None], np.arange(len(c)) * 0.1)[0])
        ec_m.append(e); ec_o.append(csv.loc[cur, "two_frame_extended_comfort"])
        w = 5 * mine[i, 4] + 5 * mine[i, 5] + 2 * mine[i, 6] + 2 * mine[i, 7]
        sc_m.append(lab[i, j[i], 8] * (w + 2 * e) / 16); sc_o.append(csv.loc[cur, "score"])
    ec_m, ec_o, sc_m, sc_o = map(np.asarray, (ec_m, ec_o, sc_m, sc_o))
    print(f"{len(ec_m)} adjacent frames: EC equal {np.mean(ec_m == ec_o):.6f}, EPDMS max |diff| "
          f"{np.abs(sc_m - sc_o).max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
