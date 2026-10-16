"""km: exact offline one-stage EPDMS of a selection sequence from label_v2.py / ec_v2.py arrays (numpy only).

Per frame t with selected column i (and the previous frame p's selected column j when ec_v2 found one):
  EPDMS_t = mult[t, i] * (5 ep + 5 ttc + 2 lk + 2 hc [+ 2 ec[t, i, j]]) / (14 [+ 2])
= run_pdm_score_one_stage.compute_final_scores (EC weight 0 when the frame has no previous frame).  Checked against the
official run_pdm_score_from_trajectories CSV in check_epdms_offline (tools/ck/km/listwise_cv2.py --check).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAB = ("nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc", "mult")
L = {k: i for i, k in enumerate(LAB)}
W = {"ep": 5.0, "ttc": 5.0, "lk": 2.0, "hc": 2.0, "ec": 2.0}


class V2Labels:
    def __init__(self, d: Path):
        d = Path(d)
        self.lab = np.load(d / "lab.npy", mmap_mode="r")
        self.ec = np.load(d / "ec.npy", mmap_mode="r")
        self.prev = np.load(d / "prev_row.npy")
        self.meta = pd.read_parquet(d / "meta.parquet")
        base = sum(W[k] * np.asarray(self.lab[..., L[k]], np.float64) for k in ("ep", "ttc", "lk", "hc"))
        self.mult = np.asarray(self.lab[..., L["mult"]], np.float64)
        self.base = base                      # [N, 96] weighted sum without EC

    def ec_of(self, rows: np.ndarray, sel: np.ndarray, sel_prev: np.ndarray) -> np.ndarray:
        """EC of (rows, sel) given the previous frame's selection sel_prev (-1 = no previous) -> float, NaN = none."""
        out = np.full(len(rows), np.nan)
        m = sel_prev >= 0
        v = np.asarray(self.ec[rows[m], sel[m], sel_prev[m]], np.float64)
        v[v == 255] = np.nan
        out[m] = v
        return out

    def epdms_given_prev(self, rows: np.ndarray, sel_prev: np.ndarray) -> np.ndarray:
        """[R, 96] EPDMS of every column of rows, the previous frame's choice fixed to sel_prev (-1 = none)."""
        b = self.base[rows]
        m = sel_prev >= 0
        out = b / 14.0
        if m.any():
            e = np.asarray(self.ec[rows[m], :, sel_prev[m]], np.float64)          # [R', 96]
            ok = e != 255
            out[m] = np.where(ok, (b[m] + W["ec"] * e) / 16.0, b[m] / 14.0)
        return self.mult[rows] * out

    def sequence_epdms(self, sel_full: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """per-row EPDMS of a full selection array sel_full [N] (column per row) on rows."""
        p = self.prev[rows]
        sp = np.where(p >= 0, sel_full[np.maximum(p, 0)], -1)
        ec = self.ec_of(rows, sel_full[rows], sp)
        b = self.base[rows, sel_full[rows]]
        w = np.where(np.isfinite(ec), (b + W["ec"] * np.nan_to_num(ec)) / 16.0, b / 14.0)
        return self.mult[rows, sel_full[rows]] * w
