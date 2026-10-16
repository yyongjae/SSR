"""refiner v4_3 Phase 0b (docs/refiner_v4/V4_3_SPEC.md 'Phase 0b -- re-registration') -- held-out data + the R2 tube.
New file; the Phase 0 scripts (prep / run_crit1 / run_crit2 / analyze) and their outputs are untouched.  CPU only,
read-only on every input; writes only under HO = CK/v4/analysis/range_oracle/heldout_r2.

Held-out tokens: the navtrain_val packed rows NOT in the F1 6,000 sample (OC.PREP/f1_rows.npy) = 12,179 rows, sorted;
token-disjoint from F1 (limitation: the same 214 logs).

HeldoutData has the attributes of oracle_common.Data, so oracle_common.token_inputs / run_crit1.process run unchanged
on it (index i = position in the held-out row list):
  rows [N] (packed rows), tokens [N], logs (per PACKED row, as Data), cand / valid80 / v0 (packed / r34_variants, full),
  sim [N, 96, 41, 6] float64 = ck/v4/lqr.simulate (float64, fit 'reduced') of par16 + var80 cast to float32 (a
      non-finite trajectory -> zeros, as F1) from ego (v0, a0, 0) = (status[..., 4], status[..., 6], 0) (prep).
  exec [N, 97, 11] float32 (EXEC_COLS: nc, dac, ddc, ep, ttc, comfort, pdms, raw, first_off, nc_t, ttc_t) = the
      OFFICIAL labels: columns 0..15 labels/navtrain_val/cand, 16..95 r34_variants/labels (16 + 5 k + v - 1, the F1
      order), 96 = the DET lateral head on the v2 top-1 (ck2T10/r34 labels_stack column 5 'v2_lat' ==
      lat_traj16[:, 0] bitwise); values from labels.npy, time indices (dac / nc / ttc_time_idx) from the label shards.
      prep checks the same assembly against F1 exec[:, :97] on the 6,000 F1 rows (bit-exact).
  plan: NaN (no un-tracked reference scoring for the held-out rows; the speed-cap probe is not part of Phase 0b).
  fin [N, 97] (finite trajectories), init [N, 3] float32 (v0, a0, 0), tracks (official nc_track / ttc_track of every
      NC / TTC failure, prep tracks.parquet: the UNKNOWN bucket of run_crit1).

R2 (spec 'Representation R2'): the Phase 0 main grid representation (corridor v2 parent grid, 0.25 s x 20 bins, box
  target, U = the LON channel for TTC (not-behind from the current ego), DAC on D bilinear) with (1) stopped-aware B
  (oracle_common.stopped_aware_B01 / stopped_mask: stopped / static objects block without the not-behind filter, moving
  objects with it; U / LON unchanged -- the spec names B only, and the official TTC rule has no stopped-track
  exception) and (2) read 'any4' for B and U (OR over the 2 x 2 grid points around the read point; the DAC read is
  bilinear on D in both reads).  On F1 this is exactly Phase 0's post hoc 'grid stopped-aware NC + read any4'
  (g25sa_nc, g25a_ttc, g25a_dac); heldout_crit1 recomputes it through r2_fields / r2_eval and counts disagreements.
  Tube membership (criteria S and 1): inside <=> no NC, TTC or DAC hit (slack >= 0), as in Phase 0; the margins
  m_t 0.2 s / m_l 0.3 m are the computed edit's margins (criterion 2), as in Phase 0 (summary 'Ambiguities').

GridTubeR2 (criterion 2): oracle_common.GridTube on the R2 fields with the any4 read in check() and an any4 critical
  step (Phase 0 left it open; fixed here before any held-out number): the minimising read's 4 grid points
  (range_targets._cells read='any4': stations j0, j0 + 1 x offsets i0, i0 + 1) -> blocked profile = OR over the 4
  points (NC: stopped-aware B; TTC: LON >= -1.3 - |v_n| delta), t_out = the end of its last blocked bin (+inf if the
  last bin is blocked, 0 if none: as Phase 0), and the critical arc length s_crit = max(0, (j0 - 0.5) ds) = where the
  read point first reads station j0's grid points (an any4 read at arc length s reads stations floor(s / ds - 0.5) and
  + 1) -- the any4 analogue of Phase 0's nearest rule (s_crit = j ds = where the read enters cell j).  Everything else
  (lon search grid + bisection, MAX_ROUNDS 30 critical rounds, lateral edit, empty / noconv kinds) is GridTube's.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_common as OC  # noqa: E402  (sets CUDA_VISIBLE_DEVICES='', OMP / MKL 1 thread)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RT = OC.RT

# ----------------------------------------------------------------------------------------------- paths
HO = OC.OUT / "heldout_r2"
HO_PREP = HO / "prep"
STACK = OC.T10 / "labels_stack"           # ck2T10/r34 stack labels (column 5 = 'v2_lat' = lat_traj16[:, 0])
STACK_HEAD_K = 5
TRAJ80 = OC.VAR / "traj80.npy"
LAT16 = OC.T10 / "lat_traj16.npy"
STATUS = OC.PACK / "status.npy"
LAB_NPY = {"par": OC.CKD / "labels" / OC.SPLIT / "cand/labels.npy", "var": OC.VAR / "labels/labels.npy",
           "head": STACK / "labels.npy"}
LAB_SHARD_DIRS = {"par": OC.LAB_SHARDS["par"], "var": OC.LAB_SHARDS["var"], "head": STACK / "shards"}
LABEL_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress", "pdm_progress_eff")
LAB_TO_EXEC = (0, 1, 5, 2, 3, 4, 6, 7)     # labels.npy column of EXEC_COLS[0..7] (nc dac ddc ep ttc comfort pdms raw)
N97 = OC.N96 + 1                           # 96 + the DET lateral head (column OC.COL_HEAD = 96)

R2_NAME = "R2 (stopped-aware B + any4 read of B / U, 0.25 s, box, TTC on U, DAC on D)"
R2_TBIN = OC.TBIN_MAIN


def kst() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


def f1_rows() -> np.ndarray:
    return np.load(OC.PREP / "f1_rows.npy")


def heldout_rows(n_packed: Optional[int] = None) -> np.ndarray:
    """packed navtrain_val rows not in the F1 sample, sorted."""
    if n_packed is None:
        n_packed = len(pd.read_parquet(OC.PACK / "tokens.parquet"))
    return np.setdiff1d(np.arange(n_packed), f1_rows())


def ego_init(status_rows: np.ndarray) -> np.ndarray:
    """packed status [n, 8] -> lqr ego [n, 3] float32 (v0 = status[4], a0 = status[6], steer0 = 0; lqr docstring)."""
    st = np.asarray(status_rows, np.float32)
    return np.stack([st[:, 4], st[:, 6], np.zeros(len(st), np.float32)], 1)


def token_trajs96(cand_row: np.ndarray, var_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """par16 + var80 of one packed row -> (trajs [96, 8, 3] float32 with non-finite rows zeroed (F1 process), fin [96])."""
    tr = np.concatenate([np.asarray(cand_row, np.float32), np.asarray(var_row, np.float32)], 0)
    fin = np.isfinite(tr).reshape(len(tr), -1).all(1)
    return np.where(fin[:, None, None], tr, 0.0).astype(np.float32), fin


def sim96(cand_row: np.ndarray, var_row: np.ndarray, init_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """-> (executed states [96, 41, 6] float64 (oracle_common.simulate = ck/v4/lqr float64), fin [96])."""
    tr, fin = token_trajs96(cand_row, var_row)
    return OC.simulate(tr, init_row), fin


# ----------------------------------------------------------------------------------------------- labels
def read_shards(group: str, columns=("row", "k", "nc", "ttc", "dac_time_idx", "nc_time_idx", "ttc_time_idx",
                                     "nc_track", "nc_obj_type", "ttc_track")) -> pd.DataFrame:
    import glob
    fs = sorted(glob.glob(str(LAB_SHARD_DIRS[group] / "c*.parquet")))
    t = pd.concat([pd.read_parquet(f, columns=list(columns)) for f in fs], ignore_index=True)
    if group == "head":
        t = t[t["k"] == STACK_HEAD_K]
    dup = t.duplicated(["row", "k"]).sum()
    assert dup == 0, (group, dup)
    return t


def build_exec(rows: np.ndarray, shards: Dict[str, pd.DataFrame]) -> np.ndarray:
    """official labels of the 97 columns of the packed rows -> [n, 97, 11] float32 in EXEC_COLS order."""
    n = len(rows)
    out = np.full((n, N97, 11), np.nan, np.float32)
    pos = np.full(int(max(rows.max(), 0)) + 1, -1, np.int64)
    pos[rows] = np.arange(n)
    spans = {"par": (0, OC.N_PAR, None), "var": (OC.N_PAR, OC.N_VAR, None), "head": (OC.COL_HEAD, 1, STACK_HEAD_K)}
    for g, (c0, K, kk) in spans.items():
        L = np.load(LAB_NPY[g], mmap_mode="r")
        v = np.asarray(L[rows], np.float32)                                   # [n, K_all, 9]
        v = v[:, kk:kk + 1] if kk is not None else v
        assert v.shape[1] == K, (g, v.shape)
        out[:, c0:c0 + K, :8] = v[..., list(LAB_TO_EXEC)]
        t = shards[g]
        t = t[(t["row"] <= pos.size - 1)]
        t = t[pos[t["row"].to_numpy()] >= 0]
        r = pos[t["row"].to_numpy()]
        k = t["k"].to_numpy().astype(np.int64) - (kk if kk is not None else 0)
        assert len(t) == n * K, (g, len(t), n * K)
        for j, c in enumerate(("dac_time_idx", "nc_time_idx", "ttc_time_idx")):
            out[r, c0 + k, 8 + j] = t[c].to_numpy().astype(np.float32)
    assert np.isfinite(out).all()
    return out


# ----------------------------------------------------------------------------------------------- data
class HeldoutData:
    """oracle_common.Data interface on the held-out rows (module docstring); arrays from HO_PREP (heldout_prep.py)."""

    def __init__(self):
        self.rows = np.load(HO_PREP / "ho_rows.npy")
        self.tokens = np.load(HO_PREP / "ho_tokens.npy")
        self.sim = np.load(HO_PREP / "ho_sim.npy", mmap_mode="r")
        self.exec = np.load(HO_PREP / "ho_exec.npy", mmap_mode="r")
        self.fin = np.load(HO_PREP / "ho_fin.npy", mmap_mode="r")
        self.init = np.load(HO_PREP / "ho_init.npy", mmap_mode="r")
        self.plan = np.broadcast_to(np.float32(np.nan), (len(self.rows), N97, 11))
        self.cand = np.load(OC.PACK / "cand.npy", mmap_mode="r")
        self.valid80 = np.load(OC.VAR / "valid80.npy", mmap_mode="r")
        self.v0 = np.load(OC.VAR / "v0.npy", mmap_mode="r")
        tok = pd.read_parquet(OC.PACK / "tokens.parquet")
        self.logs = tok["log"].astype(str).to_numpy()
        assert np.array_equal(tok["token"].astype(str).to_numpy()[self.rows], self.tokens.astype(str))
        assert self.sim.shape[:2] == (len(self.rows), OC.N96) and self.exec.shape[:2] == (len(self.rows), N97)
        self.tracks: Dict[int, Dict[int, Tuple[str, str]]] = {}
        tr = HO_PREP / "ho_tracks.parquet"
        if tr.is_file():
            t = pd.read_parquet(tr)
            for r, c, a, b in zip(t["row"].to_numpy(), t["col"].to_numpy(), t["nc_track"].to_numpy(),
                                  t["ttc_track"].to_numpy()):
                self.tracks.setdefault(int(r), {})[int(c)] = (str(a), str(b))


# ----------------------------------------------------------------------------------------------- R2
def r2_from(F: dict, gt: Optional[dict]) -> dict:
    """Phase 0 parent fields F (oracle_common.parent_fields) -> the R2 fields: B01 = stopped-aware (0.1 s), the rest
    (L01 = U / LON, D, frames) unchanged."""
    return dict(F, B01=OC.stopped_aware_B01(F["frn"], gt))


def r2_eval(X: np.ndarray, F2: dict, idx: np.ndarray, per_state: bool = False) -> Dict[str, np.ndarray]:
    """R2 read of executed states X [C, 41, 6] on the R2 fields of their parents: NC on stopped-aware B, TTC on U
    (LON, current-ego not-behind), both read 'any4' at 0.25 s bins; DAC on D (bilinear)."""
    return OC.grid_eval(X, F2, idx, tb=R2_TBIN, read="any4", ttc_lon=True, dac="grid", per_state=per_state)


class GridTubeR2(OC.GridTube):
    """R2 GT tube of one parent (module docstring): built from the R2 fields (r2_from), any4 read and critical step."""
    kind = "r2"

    def __init__(self, F2: dict, p: int = 0, tb: float = R2_TBIN):
        super().__init__(F2, p, tb)

    def check(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        V = X.shape[0]
        o = RT.executed_slack(OC._t(X), self.fr1, OC._t(self.B), self.field, index=OC.torch.zeros(V, dtype=OC.torch.long),
                              sdf_index=self.sidx, tbin_s=self.tb, lon_field=OC._t(self.LON), per_state=True,
                              read="any4")
        return {k: v.numpy() for k, v in o.items()}

    def critical(self, X1: np.ndarray, o: dict, q: int = 0) -> Tuple[float, float, float]:
        """(delta, s_crit, t_out) of candidate q's NC (else TTC) minimising any4 read (module docstring)."""
        if o["hit_nc"][q]:
            n, kd = int(np.argmin(o["state_nc"][q])), 0
        else:
            n, kd = np.unravel_index(int(np.argmin(o["state_ttc"][q])), o["state_ttc"][q].shape)
            n, kd = int(n), int(kd)
        delta = RT.TTC_DELTAS[kd]
        x = X1[q, n]
        vabs = abs(float(x[3]))
        pt = x[:2] + vabs * delta * np.array([math.cos(x[2]), math.sin(x[2])])
        _, _, _, jn, in_ = RT._cells(self.fr1, OC._t(pt)[None, None], read="any4")
        jn, in_ = jn[0, 0].numpy(), in_[0, 0].numpy()                              # [4] stations / offsets
        if kd == 0:
            prof = self.B[0, jn, in_].any(0)                                         # [M] OR over the 4 points
        else:
            prof = (self.LON[0, jn, in_] >= RT.BEHIND_M - vabs * delta).any(0)
        nz = np.flatnonzero(prof)
        if nz.size == 0:
            t_out = 0.0
        elif nz[-1] == self.M - 1:
            t_out = math.inf
        else:
            t_out = (nz[-1] + 1) * self.tb
        ds = float(self.fr1["ds"][0])
        j0 = int(jn.min())
        return float(delta), max(0.0, (j0 - 0.5) * ds), float(t_out)
