"""refiner v4_3 Phase 0 oracle (docs/refiner_v4/V4_3_SPEC.md 'Phase 0 oracle', SSR/report/84 section 5) -- shared
library: data access, GT range fields on the parent grid, grid-free exact GT checks, computed minimal edits, speed-cap
probe and the official scorer.  CPU only (CUDA_VISIBLE_DEVICES= set here), read-only on every input; the scripts write
only under OUT.

Inputs (navtrain_val, the decision split; all read-only)
  F1 sample  CK/ck2/analysis/detmap_path_ideas/F1/out/val/run.npz (prep.py copies it to OUT/prep/f1_*.npy for mmap):
             6,000 seeded tokens (rows into packed/navtrain_val), 112 trajectories each = par16 (v2 r34 top-16,
             column 0 = v2 top-1) + var80 (column 16 + 5 k + v - 1: the rule variants a-1.0, a-0.5, a+0.5, l-0.5,
             l+0.5 of parent k) + latT16 (column 96 + k: the DET lateral head ck2T10 on parent k); sim [.., 41, 6] =
             the OFFICIAL PDMSimulator states (x, y, heading, v, a, steer; N frame, t = 0.1 n), exec [.., 11] = the
             official labels on them (nc, dac, ddc, ep, ttc, comfort, pdms, raw progress, first off-road idx,
             nc_time_idx, ttc_time_idx; bit-exact vs the stored label shards), plan [.., 11] (the same scorer on the
             un-tracked reference), init [3] (v0, a0, steer0 of the metric-cache ego).
  packed     CK/packed/navtrain_val: cand.npy [N, 16, 8, 3] (= F1 par), tokens.parquet (token, log), v2_final / v2_im
             [N, 16] (deployment blend inputs).
  variants   r34_variants: valid80 [N, 80] (False = a lateral variant on a parent with S_8 < 3 m: a duplicate of the
             parent, excluded from 'the 96'), v0.npy [N] (|v| at t0, the make_variants input), labels/shards (+ cand
             label shards): official nc_track / ttc_track of every failure (prep.py -> OUT/prep/tracks.parquet).
  critic     ck2T10/r34 prob16 / prob80 [N, 16|80, 5] (the DET teacher critic; the only critic with val outputs on
             these 96) for the deployment blend score (selection-only oracle).
  GT         CK/v3/gt_objects/navtrain_val/<tok>.npz (gt_future_v1: metric-cache rule, 0..5 s, 360 deg) queried on the
             0.1 s grid 0..4.9 s (RT.fine_times); SDF sdf/navtrain_val/<tok>.npz (official DAC layers, + inside).

Frames / units: N frame at t0 (rear axle origin, x forward, y left, heading CCW; m, s, rad).  Ego box = the official
  one (half length 2.588, half width 1.1485, centre RA2C = 1.461 m ahead of the rear axle).

Representations (criterion 1)
  (i) learnable = GT fields on each candidate's PARENT grid (refiner.corridor version 2, 48 stations x 17 offsets,
      cell heading = the parent's tangent): blocked field B (RT.blocked_field_np: ego box at the cell overlaps an OBS GT
      box, not-behind proxy), LON (lon_max channel: TTC not-behind from the CURRENT ego), D = the SDF at the grid
      points (unreadable points -> +SDF_CLIP: no DAC information, like the unreadable corners of the 'sdf' read); read
      at the executed states by RT.executed_slack (nearest cell; 'any4' as a variant), bins of 0.1 / 0.25 (main) /
      0.5 s pooled exactly from the 0.1 s field (OR for B, max for LON: identical to blocked_field at that bin).
      Point target (variant): O = the grid POINT lies inside an OBS GT box (no not-behind: a point has no object
      reference), read at the 9 footprint points (3 along x 3 across, ck/v4/evidence FOOT) of the executed box.
  (ii) exact = no grid: the executed ego box (and its TTC projections |v| delta along the executed heading) against the
      GT boxes at the same 0.1 s sample by SAT (RT.box_gaps_np) -- rules 'proxy' (the field's: not-behind proxy -1.3 m
      from the box centre for NC, from the CURRENT ego centre for TTC; stopped-ego skips 0.05 / 5e-3 m/s) and an
      emulation of the official classification ('emul'): NC get_collision_type order (stopped ego -> not at fault,
      stopped / non-agent track (first-appearance speed <= 0.05) -> at fault, rear by angle > 150 deg -> not at fault,
      front bumper segment FL-FR intersects -> at fault, else lateral: at fault iff in multiple lanes or off-road) with
      the per-proposal collided-track memory; TTC: ahead (angle < 30 deg) or (multiple lanes / off-road / intersection
      and not behind), memory for the rest; the lane / intersection map is not in the GT store -> two bounds: 'lo'
      (only off-road = an executed corner with SDF < 0 counts) and 'hi' (always counts).  DAC exact = the min over the
      41 states x 4 corners of the stored SDF < 0 (unreadable corners excluded).

Computed edit (criterion 2, spec M5 with GT fields; compute_edit)
  On the v2 top-1 parent (column 0), with a tube = GridTube (learnable, main) or ExactTube (grid-free):
  skip (no edit) if inside the tube, or v0 < 2 m/s, or S_8 < 3 m;  NC/TTC-limited -> the smallest deceleration magnitude
  a of the 0.8 s ramp (variants.make_variants speeds=(a,) along the same path; grid A_GRID = 0.05 .. 3.95, 3.999 m/s^2
  then 6 bisection steps in the bracket) so that the executed (ck/v4/lqr, float64, parity 2.4e-5 m) arrival at the
  critical station >= t_out_block + m_t (m_t 0.2 s main); arrival = the first t_n + delta at which the read point
  (rear axle, or its TTC projection |v_n| delta) reaches the critical station along the parent frame (linear in n);
  critical station / t_out_block (GridTube): the cell of the minimising read (executed_slack per_state argmin),
  t_out = the end of that cell's last blocked bin (the NC profile B, or for a TTC read LON >= -1.3 - |v_n| delta; +inf
  if the last bin is blocked: never reach the station within the horizon); repeated (max MAX_ROUNDS = 30) while the
  edited trajectory is still NC/TTC-outside (a new critical station; |a| only grows); a loop that runs out of rounds
  with every round feasible is 'noconv_lon' (no edit, like 'empty_lon', but reported apart from the empty tube; at
  MAX_ROUNDS 5, 26 val tokens exhausted the loop: with 30, 3 become feasible lon edits (F1 idx 692 / 2061 / 2062),
  23 end in a truly infeasible round (<= 14 rounds), none exhausts 30).  ExactTube (no stations):
  the same margin applied everywhere -- the smallest |a| on A_GRID whose executed states pass the exact NC / TTC proxy
  with every object sample dilated by m_t (a place stays blocked m_t after the object left it; no bisection: a
  contact-free boundary refined to 1e-3 m/s^2 would sit within float noise of the official polygons).
  DAC-limited (after the lon edit) -> the smallest early smoothstep offset |D| (decoder.smoothstep_lat, onset
  0.2 S_8 = the rule l+-0.5 shape, make_variants lats=(D,) [combine 'cross' with the lon a]; D_GRID 0.05 .. 1.95,
  1.999 m) toward the side with the larger margin at the critical DAC station (RT.margins of the tube's field; centred
  box already outside: the side whose +-1 m shift reads the larger min corner field) until the executed min corner
  margin >= m_l (0.3 m main).  No feasible edit = 'empty tube' -> no edit (spec); best_effort=True keeps the largest
  edit instead (reported separately).

Speed-cap probe (speed_cap_traj): v_max (RT.v_max, median tracking-error table + 0.3 m, outside margin from the SDF
  margins of the parent) per station; if the planned segment speed exceeds it at an in-plan station, the parent is
  re-timed along the SAME path with v <= max(v_max, V_FLOOR = 1 m/s) (never faster than planned).

Official scorer: tools/refiner/score_trajectories.score_token (one simulate + score call per token, the official
  2-proposal values per trajectory, bit-exact; plain navsim classes, record=False as F1) on the navtrain_val metric
  cache (tools/ck/data/common.mc_path).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.ck.data import common as CM  # noqa: E402  (pins navsim to this worktree)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.v4 import lqr as LQ  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import range_targets as RT  # noqa: E402
from navsim.agents.para_ssr.ck.variants import make_variants  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.raster import frenet  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import N_LAT, N_ST, corridor_geometry  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import HALF_LEN, HALF_WID, RA2C, DraftPath  # noqa: E402
from navsim.agents.para_ssr.refiner.sdf import SDF_CLIP, ego_corners, load_sdf, sample_sdf_np  # noqa: E402

torch.set_num_threads(1)
F64 = torch.float64

# ----------------------------------------------------------------------------------------------- paths
DATA = Path("/home/external-user/ssd/yongjae_refiner")
CKD = DATA / "ck"
SPLIT = "navtrain_val"
F1_DIR = CKD / "ck2/analysis/detmap_path_ideas/F1/out/val"
PACK = CKD / "packed" / SPLIT
INFER = CKD / "ck2/analysis/refiner_dm_traits/infer" / SPLIT
VAR = INFER / "r34_variants"
T10 = INFER / "ck2T10/r34"
GT_OBJ = CKD / "v3/gt_objects" / SPLIT
SDF_DIR = DATA / "sdf" / SPLIT
LAB_SHARDS = {"par": CKD / "labels" / SPLIT / "cand/shards", "var": VAR / "labels/shards"}
LEAD = CKD / "lead" / f"{SPLIT}.parquet"
OUT = CKD / "v4/analysis/range_oracle"
PREP = OUT / "prep"

# ----------------------------------------------------------------------------------------------- constants
N_PAR, N_VAR, N96 = 16, 80, 96
COL_HEAD = 96                     # F1 latT column of parent 0 = the DET lateral head on the v2 top-1
VNAMES = ("a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")
EXEC_COLS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "raw", "first_off", "nc_t", "ttc_t")
EC = {c: i for i, c in enumerate(EXEC_COLS)}
SCORE_COLS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "raw_progress", "dac_time_idx", "nc_time_idx",
              "ttc_time_idx")          # official labels of a scored edit (same order as EXEC_COLS)
TBINS = (0.1, 0.25, 0.5)
TBIN_MAIN = 0.25
CORRIDOR_VERSION = 2
HEADING = "parent"
TTC_K = (0, 3, 6, 9)              # official future_time_idcs (0.1 s steps)
ANG_AHEAD = math.radians(30.0)    # nuplan is_agent_ahead
ANG_BEHIND = math.radians(150.0)  # nuplan is_agent_behind
STOP_TRACK = 5e-2                 # is_track_stopped threshold [m/s]
FOOT = tuple((a, c) for a in (-HALF_LEN, 0.0, HALF_LEN) for c in (-HALF_WID, 0.0, HALF_WID))   # 9 points / centre
A_GRID = tuple(-round(0.05 * k, 4) for k in range(1, 80)) + (-3.999,)    # lon edit magnitudes (|a| ascending)
D_GRID = tuple(round(0.05 * k, 4) for k in range(1, 40)) + (1.999,)      # lat edit magnitudes
N_BISECT = 6
MAX_ROUNDS = 30                   # critical-station rounds of GridTube.lon_edit (exhausted -> 'noconv_lon')
V0_MIN = 2.0                      # no edit below this ego speed [m/s]
S8_MIN = 3.0                      # no edit below this draft length [m]
V_FLOOR = 1.0                     # speed-cap probe floor [m/s]
MARGIN_T, MARGIN_L = 0.2, 0.3     # main edit margins (s, m)
MARGIN_CONFIGS = ((0.2, 0.3), (0.1, 0.3), (0.3, 0.3), (0.2, 0.2), (0.2, 0.5))   # first = main
T_N = np.arange(41, dtype=np.float64) * 0.1


def cfg_name(tube: str, mt: float, ml: float, best: bool = False) -> str:
    return f"{tube}_t{mt:.1f}_l{ml:.1f}" + ("_best" if best else "")


# ----------------------------------------------------------------------------------------------- data
class Data:
    """Per-process lazily opened arrays (mmap); F1 arrays from OUT/prep (prep.py)."""

    def __init__(self):
        self.rows = np.load(PREP / "f1_rows.npy")
        self.tokens = np.load(PREP / "f1_tokens.npy")
        self.sim = np.load(PREP / "f1_sim.npy", mmap_mode="r")
        self.exec = np.load(PREP / "f1_exec.npy", mmap_mode="r")
        self.plan = np.load(PREP / "f1_plan.npy", mmap_mode="r")
        self.fin = np.load(PREP / "f1_fin.npy", mmap_mode="r")
        self.init = np.load(PREP / "f1_init.npy", mmap_mode="r")
        self.cand = np.load(PACK / "cand.npy", mmap_mode="r")
        self.valid80 = np.load(VAR / "valid80.npy", mmap_mode="r")
        self.v0 = np.load(VAR / "v0.npy", mmap_mode="r")
        tok = pd.read_parquet(PACK / "tokens.parquet")
        self.logs = tok["log"].astype(str).to_numpy()
        assert np.array_equal(tok["token"].astype(str).to_numpy()[self.rows], self.tokens.astype(str))
        tr = PREP / "tracks.parquet"
        self.tracks: Dict[int, Dict[int, Tuple[str, str]]] = {}
        if tr.is_file():
            t = pd.read_parquet(tr)
            for r, c, a, b in zip(t["row"].to_numpy(), t["col"].to_numpy(), t["nc_track"].to_numpy(),
                                  t["ttc_track"].to_numpy()):
                self.tracks.setdefault(int(r), {})[int(c)] = (str(a), str(b))


def load_gt(token: str) -> Optional[dict]:
    """GT objects of a token on the 0.1 s grid 0..4.9 s -> dict boxes [A, 50, 5] (cx, cy, heading, L, W; float64), obs
    [A, 50] bool (metric-cache rule), is_agent [A], vfirst [A] (first-appearance speed: is_track_stopped), track [A],
    t_avail (s), obj (raw); None if the token has no GT file."""
    p = GT_OBJ / f"{token}.npz"
    if not p.is_file():
        return None
    obj = GF.load_objects(p)
    boxes, st = GF.query(obj, RT.fine_times(), dtype=np.float64)
    meta, first = np.asarray(obj["meta"]), np.asarray(obj["first"], np.float64)
    return dict(obj=obj, boxes=boxes, obs=st == GF.OBS, is_agent=meta[:, GF.MT_AGENT] > 0,
                vfirst=np.hypot(first[:, GF.FI_VX], first[:, GF.FI_VY]),
                track=np.asarray(obj["track"]).astype(str), t_avail=float(GF.t_avail(obj)))


def load_sdf_np(token: str) -> Optional[np.ndarray]:
    p = SDF_DIR / f"{token}.npz"
    return load_sdf(p).astype(np.float64) if p.is_file() else None


def token_inputs(D: Data, i: int) -> dict:
    """F1 index i -> everything one worker needs for that token (float64 copies)."""
    row = int(D.rows[i])
    tok = str(D.tokens[i])
    valid96 = np.concatenate([np.ones(N_PAR, bool), np.asarray(D.valid80[row], bool)])
    fin = np.asarray(D.fin[i], bool)
    return dict(i=i, row=row, token=tok, log=str(D.logs[row]), par=np.asarray(D.cand[row], np.float64),
                par32=np.asarray(D.cand[row], np.float32), X=np.asarray(D.sim[i], np.float64),
                exec=np.asarray(D.exec[i], np.float64), plan=np.asarray(D.plan[i], np.float64), fin=fin,
                valid96=valid96 & fin[:N96], init=np.asarray(D.init[i], np.float64), v0=float(D.v0[row]),
                tracks=D.tracks.get(row, {}), gt=load_gt(tok), sdf=load_sdf_np(tok))


def parent_of(col: np.ndarray) -> np.ndarray:
    """F1 column (0..95) -> parent index (0..15)."""
    col = np.asarray(col)
    return np.where(col < N_PAR, col, (col - N_PAR) // 5)


# ----------------------------------------------------------------------------------------------- fields
def point_field_np(frn: Dict[str, np.ndarray], boxes: np.ndarray, obs: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Point occupancy O [P, 48, 17, T] bool: grid point q_ij inside an OBS GT box (closed) at sample t (exact; boxes
    culled by the grid's bounding box grown by the box half diagonal)."""
    pts = frn["pts"]
    P = pts.shape[0]
    A, T = boxes.shape[:2]
    out = np.zeros((P, N_ST * N_LAT, T), bool)
    if A == 0:
        return out.reshape(P, N_ST, N_LAT, T)
    hl, hw = 0.5 * boxes[..., 3], 0.5 * boxes[..., 4]
    r = np.hypot(hl, hw) + 1e-6
    for p in range(P):
        q = pts[p].reshape(-1, 2)
        lo, hi = q.min(0), q.max(0)
        inb = (obs & (boxes[..., 0] >= lo[0] - r) & (boxes[..., 0] <= hi[0] + r)
               & (boxes[..., 1] >= lo[1] - r) & (boxes[..., 1] <= hi[1] + r))
        ai, ti = np.nonzero(inb)
        for k0 in range(0, ai.size, chunk):
            a, t = ai[k0:k0 + chunk], ti[k0:k0 + chunk]
            b = boxes[a, t]
            c, s = np.cos(b[:, 2])[:, None], np.sin(b[:, 2])[:, None]
            rx = q[None, :, 0] - b[:, None, 0]
            ry = q[None, :, 1] - b[:, None, 1]
            ins = (np.abs(rx * c + ry * s) <= hl[a, t][:, None]) & (np.abs(-rx * s + ry * c) <= hw[a, t][:, None])
            kk, qq = np.nonzero(ins)
            out[p, qq, t[kk]] = True
    return out.reshape(P, N_ST, N_LAT, T)


def parent_fields(par: np.ndarray, gt: Optional[dict], sdf: np.ndarray, point: bool = True) -> dict:
    """GT fields of the P parents [P, 8, 3] (module docstring (i)): geom, fr (torch frame), frn (numpy), B01 / L01 /
    O01 [P, 48, 17, 50] (0.1 s = per sample), D [P, 48, 17] (SDF at the grid points, unreadable -> +SDF_CLIP),
    D_ok."""
    g = corridor_geometry(torch.as_tensor(par, dtype=F64), version=CORRIDOR_VERSION)
    fr = RT.parent_frame(g, HEADING)
    frn = RT.frame_np(fr)
    P = par.shape[0]
    if gt is None or gt["boxes"].shape[0] == 0:
        B01 = np.zeros((P, N_ST, N_LAT, 50), bool)
        L01 = np.full((P, N_ST, N_LAT, 50), -np.inf)
        O01 = B01.copy() if point else None
    else:
        B01, L01 = RT.blocked_field_np(frn, gt["boxes"], gt["obs"], tbin_s=0.1, lon_max=True)
        O01 = point_field_np(frn, gt["boxes"], gt["obs"]) if point else None
    Dv, Dok = sample_sdf_np(sdf, frn["pts"])
    return dict(geom=g, fr=fr, frn=frn, B01=B01, L01=L01, O01=O01, D=np.where(Dok, Dv, SDF_CLIP), D_ok=Dok)


def stopped_mask(gt: Optional[dict]) -> np.ndarray:
    """[A] bool: the official is_track_stopped of the collision classification (non-agent, or first-appearance speed
    <= 0.05 m/s) -- such a collision is at fault whatever the geometry (checked BEFORE the rear test)."""
    if gt is None:
        return np.zeros(0, bool)
    return (~gt["is_agent"]) | (gt["vfirst"] <= STOP_TRACK)


def stopped_aware_B01(frn: Dict[str, np.ndarray], gt: Optional[dict]) -> np.ndarray:
    """variant of B (0.1 s): moving agents with the not-behind proxy OR stopped / static objects without it
    [P, 48, 17, 50] bool (two RT.blocked_field_np calls on the object subsets)."""
    P = frn["pts"].shape[0]
    out = np.zeros((P, N_ST, N_LAT, 50), bool)
    if gt is None or gt["boxes"].shape[0] == 0:
        return out
    st = stopped_mask(gt)
    for sub, bm in ((st, None), (~st, RT.BEHIND_M)):
        if sub.any():
            out |= RT.blocked_field_np(frn, gt["boxes"][sub], gt["obs"][sub], tbin_s=0.1, behind_m=bm)
    return out


def pool_time(F01: np.ndarray, tb: float, how: str) -> np.ndarray:
    """0.1 s per-sample field [..., 50] -> bins of tb (RT.time_bins): 'any' (B, O) or 'max' (LON)."""
    if abs(tb - 0.1) < 1e-9:
        return F01
    mt = RT.time_bins(RT.fine_times(), tb)
    M = RT.n_bins_of(tb)
    if how == "any":
        return np.stack([F01[..., mt == m].any(-1) for m in range(M)], -1)
    return np.stack([F01[..., mt == m].max(-1) for m in range(M)], -1)


def _t(x, dtype=F64):
    return torch.as_tensor(np.ascontiguousarray(x)).to(dtype) if np.asarray(x).dtype != bool \
        else torch.as_tensor(np.ascontiguousarray(x))


def grid_eval(X: np.ndarray, F: dict, idx: np.ndarray, tb: float = TBIN_MAIN, read: str = "nearest",
              ttc_lon: bool = True, dac: Optional[str] = "grid", sdf: Optional[np.ndarray] = None,
              per_state: bool = False) -> Dict[str, np.ndarray]:
    """RT.executed_slack of executed states X [C, 41, 6] on the parents' GT fields (candidate c -> parent idx[c]);
    dac 'grid' (D) | 'sdf' (the stored SDF) | None."""
    B = pool_time(F["B01"], tb, "any")
    LON = pool_time(F["L01"], tb, "max") if ttc_lon else None
    C = X.shape[0]
    field, sidx = None, None
    if dac == "grid":
        field = _t(F["D"])
    elif dac == "sdf":
        field = _t(sdf)[None]
        sidx = torch.zeros(C, dtype=torch.long)
    o = RT.executed_slack(_t(X), F["fr"], _t(B), field, index=torch.as_tensor(idx).long(), sdf_index=sidx,
                          tbin_s=tb, read=read, lon_field=None if LON is None else _t(LON), per_state=per_state)
    return {k: v.numpy() for k, v in o.items()}


def point_eval(X: np.ndarray, F: dict, idx: np.ndarray, tb: float = TBIN_MAIN) -> Dict[str, np.ndarray]:
    """Point target: O read at the 9 footprint points of each executed box (and their TTC projections) -> hit_nc,
    hit_ttc [C] (OR over the points)."""
    O = _t(pool_time(F["O01"], tb, "any"))
    h = X[..., 2]
    eh = np.stack([np.cos(h), np.sin(h)], -1)
    nh = np.stack([-np.sin(h), np.cos(h)], -1)
    ce = X[..., :2] + RA2C * eh
    hit_nc = np.zeros(X.shape[0], bool)
    hit_ttc = np.zeros(X.shape[0], bool)
    ix = torch.as_tensor(idx).long()
    for a, c in FOOT:
        Xp = X.copy()
        Xp[..., :2] = ce + a * eh + c * nh
        o = RT.executed_slack(_t(Xp), F["fr"], O, None, index=ix, tbin_s=tb)
        hit_nc |= o["hit_nc"].numpy()
        hit_ttc |= o["hit_ttc"].numpy()
    return {"hit_nc": hit_nc, "hit_ttc": hit_ttc}


# ----------------------------------------------------------------------------------------------- exact checks
def corner_sdf(X: np.ndarray, sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """executed states [C, 41, >=3] -> (min readable corner SDF [C, 41] (+inf none readable), offroad [C, 41] = some
    readable corner < 0)."""
    cor = ego_corners(torch.as_tensor(X[..., :3])).numpy()                       # [C, 41, 4, 2]
    v, ok = sample_sdf_np(sdf, cor)
    vm = np.where(ok, v, np.inf).min(-1)
    return vm, (ok & (v < 0)).any(-1)


def exact_events(X: np.ndarray, gt: Optional[dict], ks: Sequence[int] = TTC_K, dilate: int = 0) -> Dict[str, np.ndarray]:
    """All SAT overlaps of the executed ego box of each state n (shifted |v_n| 0.1 k along the executed heading for TTC
    step k) with OBS GT boxes at sample n + k -> event arrays c, n, k, a [E] (candidate, state, step, object), lon_box
    (object centre along the heading from the (shifted) ego box centre), lon_cur (from the CURRENT ego box centre),
    ang (relative angle rear axle n -> object centre, rad), box [E, 5].  Exact up to the culling radius (half
    diagonals + 1 mm).  dilate = d > 0 (edit margin only): also the samples n + k - 1 .. n + k - d (clipped at 0), i.e.
    a place counts as blocked until d x 0.1 s after the object left it."""
    keys = ("c", "n", "k", "a")
    empty = {k: np.zeros(0, np.int64) for k in keys}
    empty.update(lon_box=np.zeros(0), lon_cur=np.zeros(0), ang=np.zeros(0), box=np.zeros((0, 5)))
    if gt is None or gt["boxes"].shape[0] == 0:
        return empty
    bx, obs = gt["boxes"], gt["obs"]
    C, NT = X.shape[:2]
    xy, h = X[..., :2], X[..., 2]
    v = np.abs(X[..., 3])
    eh = np.stack([np.cos(h), np.sin(h)], -1)
    ce0 = xy + RA2C * eh
    rad = RT.R_EGO + 0.5 * np.hypot(bx[..., 3], bx[..., 4]) + 1e-3             # [A, 50]
    parts = []
    for k, back in ((k, b) for k in ks for b in range(int(dilate) + 1)):
        ra = xy + (v * 0.1 * k)[..., None] * eh
        ce = ra + RA2C * eh
        tt = np.clip(np.arange(NT) + k - back, 0, bx.shape[1] - 1)
        co = bx[:, tt, :2]                                                         # [A, 41, 2]
        dd = np.hypot(ce[:, :, None, 0] - co.transpose(1, 0, 2)[None, :, :, 0],
                      ce[:, :, None, 1] - co.transpose(1, 0, 2)[None, :, :, 1])  # [C, 41, A]
        near = (dd <= rad[:, tt].T[None]) & obs[:, tt].T[None]
        c, n, a = np.nonzero(near)
        if c.size == 0:
            continue
        pose = np.concatenate([ra[c, n], h[c, n][:, None]], -1)
        box = bx[a, tt[n]]
        ov = RT.box_gaps_np(pose, box).max(-1) <= 0.0
        c, n, a, box = c[ov], n[ov], a[ov], box[ov]
        if c.size == 0:
            continue
        e = eh[c, n]
        rel = box[:, :2] - xy[c, n]
        nr = np.maximum(np.linalg.norm(rel, axis=-1), 1e-12)
        parts.append(dict(c=c, n=n, k=np.full(c.shape, k), a=a,
                          lon_box=((box[:, :2] - ce[c, n]) * e).sum(-1), lon_cur=((box[:, :2] - ce0[c, n]) * e).sum(-1),
                          ang=np.arccos(np.clip((rel * e).sum(-1) / nr, -1.0, 1.0)), box=box))
    if not parts:
        return empty
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


def _seg_box(p0: np.ndarray, p1: np.ndarray, box: np.ndarray) -> bool:
    """segment p0 -> p1 intersects the closed oriented box (cx, cy, h, L, W) (Liang-Barsky in the box frame)."""
    c, s = math.cos(box[2]), math.sin(box[2])
    a = np.array([(p0[0] - box[0]) * c + (p0[1] - box[1]) * s, -(p0[0] - box[0]) * s + (p0[1] - box[1]) * c])
    b = np.array([(p1[0] - box[0]) * c + (p1[1] - box[1]) * s, -(p1[0] - box[0]) * s + (p1[1] - box[1]) * c])
    hl, hw = 0.5 * box[3], 0.5 * box[4]
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] + hl), (d[0], hl - a[0]), (-d[1], a[1] + hw), (d[1], hw - a[1])):
        if p == 0.0:
            if q < 0.0:
                return False
        else:
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def exact_flags(X: np.ndarray, gt: Optional[dict], sdf: np.ndarray, ev: Optional[dict] = None,
                behind_m: float = RT.BEHIND_M, emul: bool = True, dilate: int = 0) -> Dict[str, np.ndarray]:
    """Grid-free GT checks of executed states X [C, 41, 6] (module docstring (ii)) -> bool [C]: nc_proxy, ttc_proxy,
    dac (stored SDF corners), nc_emul_lo / nc_emul_hi (fail; nc_val_* in {0, 0.5, 1}), ttc_emul_lo / ttc_emul_hi
    (emul=True); first proxy hit nc_n [C] / ttc_n, ttc_kstep (-1 none); min_corner [C]; offroad [C, 41].  dilate:
    exact_events (edit margin)."""
    C = X.shape[0]
    ev = exact_events(X, gt, dilate=dilate) if ev is None else ev
    v = np.abs(X[..., 3])
    vm, off = corner_sdf(X, sdf)
    out = {"dac": vm.min(-1) < 0.0, "min_corner": vm.min(-1), "offroad": off}
    c, n, k, a = ev["c"], ev["n"], ev["k"], ev["a"]
    m_nc = (k == 0) & (v[c, n] > RT.STOP_NC) & (ev["lon_box"] >= behind_m)
    m_tt = (v[c, n] >= RT.STOP_TTC) & (ev["lon_cur"] >= behind_m)
    stp = stopped_mask(gt)
    m_ns = (k == 0) & (v[c, n] > RT.STOP_NC) & ((ev["lon_box"] >= behind_m) | (stp[a] if a.size else False))
    hit_s = np.zeros(C, bool)
    hit_s[c[m_ns]] = True
    out["nc_proxy_sa"] = hit_s                      # stopped-aware variant (stopped / static: no not-behind filter)
    for key, m in (("nc", m_nc), ("ttc", m_tt)):
        hit = np.zeros(C, bool)
        hit[c[m]] = True
        out[f"{key}_proxy"] = hit
        first_n = np.full(C, -1, np.int64)
        first_k = np.full(C, -1, np.int64)
        if m.any():
            order = np.lexsort((k[m], n[m]))
            cm, nm, km = c[m][order], n[m][order], k[m][order]
            u, i0 = np.unique(cm, return_index=True)
            first_n[u], first_k[u] = nm[i0], km[i0]
        out[f"{key}_n"], out[f"{key}_kstep"] = first_n, first_k
    if not emul:
        return out
    # official classification emulation (two bounds for the lane / intersection map)
    is_agent, vfirst = (gt["is_agent"], gt["vfirst"]) if gt is not None else (np.zeros(0, bool), np.zeros(0))
    cor = ego_corners(torch.as_tensor(X[..., :3])).numpy()                       # FL, RL, RR, FR
    for bound in ("lo", "hi"):
        val = np.ones(C)
        tfail = np.zeros(C, bool)
        for cc in np.unique(c):
            # NC
            sel = np.flatnonzero((c == cc) & (k == 0))
            sel = sel[np.lexsort((a[sel], n[sel]))]
            mem = set()
            for e in sel:
                ai, ni = int(a[e]), int(n[e])
                if ai in mem:
                    continue
                if v[cc, ni] <= RT.STOP_NC:
                    mem.add(ai)
                    continue
                if (not is_agent[ai]) or vfirst[ai] <= STOP_TRACK:
                    fault = True
                elif ev["ang"][e] > ANG_BEHIND:
                    fault = False
                elif _seg_box(cor[cc, ni, 0], cor[cc, ni, 3], ev["box"][e]):
                    fault = True
                else:
                    fault = bound == "hi" or bool(off[cc, ni])
                if fault:
                    val[cc] = min(val[cc], 0.0 if is_agent[ai] else 0.5)
                else:
                    mem.add(ai)
            # TTC
            sel = np.flatnonzero(c == cc)
            sel = sel[np.lexsort((a[sel], k[sel], n[sel]))]
            mem = set()
            for e in sel:
                ai, ni = int(a[e]), int(n[e])
                if ai in mem or v[cc, ni] < RT.STOP_TTC:
                    continue
                xarea = bound == "hi" or bool(off[cc, ni])
                if ev["ang"][e] < ANG_AHEAD or (xarea and not ev["ang"][e] > ANG_BEHIND):
                    tfail[cc] = True
                else:
                    mem.add(ai)
        out[f"nc_val_{bound}"] = val
        out[f"nc_emul_{bound}"] = val < 1.0
        out[f"ttc_emul_{bound}"] = tfail
    return out


# ----------------------------------------------------------------------------------------------- simulation / decode
def decode(par32: np.ndarray, v0: float, speeds: Sequence[float] = (), lats: Sequence[float] = (),
           combine: str = "separate") -> np.ndarray:
    """variants.make_variants of ONE parent [8, 3] (float32) -> [V, 8, 3] float32 (index 0 = the parent; module
    docstring of variants for the V order)."""
    tau = torch.as_tensor(np.asarray(par32, np.float32))[None, None]
    o = make_variants(tau, torch.tensor([float(v0)], dtype=F64), speeds=tuple(speeds), lats=tuple(lats),
                      combine=combine)
    return o["traj"][0, 0].numpy()


def simulate(trajs: np.ndarray, init: np.ndarray) -> np.ndarray:
    """official PDM simulation (ck/v4/lqr, float64) of float32-cast trajectories [K, 8, 3] -> states [K, 41, 6]."""
    t = torch.as_tensor(np.asarray(trajs, np.float32).astype(np.float64))
    return LQ.simulate(t, torch.as_tensor(np.asarray(init, np.float64))[None], dtype=F64).numpy()


def _frame1(fr: Dict[str, torch.Tensor], p: int = 0) -> Dict[str, torch.Tensor]:
    return RT.gather_frame(fr, torch.tensor([p]))


def arrival(X: np.ndarray, fr1: Dict[str, torch.Tensor], delta: float, s_crit: float) -> np.ndarray:
    """[V] first time t_n + delta at which the read point (rear axle shifted |v_n| delta along the executed heading)
    reaches arc length s_crit on the parent frame fr1 (linear between states); +inf if never within the 41 states."""
    V = X.shape[0]
    h = X[..., 2]
    pts = X[..., :2] + (np.abs(X[..., 3]) * delta)[..., None] * np.stack([np.cos(h), np.sin(h)], -1)
    frv = {k: v.expand(V, *v.shape[1:]) for k, v in fr1.items()}
    s = frenet(_t(pts), frv)["s"].numpy()                                         # [V, 41]
    reach = s >= s_crit
    out = np.full(V, np.inf)
    for q in range(V):
        nz = np.flatnonzero(reach[q])
        if nz.size == 0:
            continue
        n = int(nz[0])
        if n == 0:
            out[q] = delta
            continue
        s0, s1 = s[q, n - 1], s[q, n]
        f = (s_crit - s0) / (s1 - s0) if s1 > s0 else 1.0
        out[q] = T_N[n - 1] + 0.1 * float(np.clip(f, 0.0, 1.0)) + delta
    return out


# ----------------------------------------------------------------------------------------------- tubes
class GridTube:
    """GT tube of one parent on its grid (learnable representation, main setting: 0.25 s bins, nearest read, TTC not
    behind from the current ego via LON, DAC on D)."""
    kind = "grid"

    def __init__(self, F: dict, p: int = 0, tb: float = TBIN_MAIN):
        self.fr1 = _frame1(F["fr"], p)
        self.tb = tb
        self.B = pool_time(F["B01"][p:p + 1], tb, "any")
        self.LON = pool_time(F["L01"][p:p + 1], tb, "max")
        self.field = _t(F["D"][p:p + 1])
        self.sidx = None
        self.M = self.B.shape[-1]

    def check(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        V = X.shape[0]
        o = RT.executed_slack(_t(X), self.fr1, _t(self.B), self.field, index=torch.zeros(V, dtype=torch.long),
                              sdf_index=self.sidx, tbin_s=self.tb, lon_field=_t(self.LON), per_state=True)
        return {k: v.numpy() for k, v in o.items()}

    def critical(self, X1: np.ndarray, o: dict, q: int = 0) -> Tuple[float, float, float]:
        """(delta, s_crit, t_out) of candidate q's NC (else TTC) minimising read (module docstring)."""
        if o["hit_nc"][q]:
            n, kd = int(np.argmin(o["state_nc"][q])), 0
        else:
            n, kd = np.unravel_index(int(np.argmin(o["state_ttc"][q])), o["state_ttc"][q].shape)
            n, kd = int(n), int(kd)
        delta = RT.TTC_DELTAS[kd]
        x = X1[q, n]
        vabs = abs(float(x[3]))
        pt = x[:2] + vabs * delta * np.array([math.cos(x[2]), math.sin(x[2])])
        j, i, ok = RT._cells(self.fr1, _t(pt)[None, None])
        j, i = int(j[0, 0]), int(i[0, 0])
        if kd == 0:
            prof = self.B[0, j, i]
        else:
            prof = self.LON[0, j, i] >= RT.BEHIND_M - vabs * delta
        nz = np.flatnonzero(prof)
        if nz.size == 0:
            t_out = 0.0
        elif nz[-1] == self.M - 1:
            t_out = math.inf
        else:
            t_out = (nz[-1] + 1) * self.tb
        ds = float(self.fr1["ds"][0])
        return float(delta), j * ds, float(t_out)

    def lon_edit(self, par32, v0, init, X, o, mt):
        """critical-station rule (module docstring), repeated while still NC / TTC outside -> (ok, a, X, o, info);
        info: rounds, t_out / delta (of the last round), noconv (True: MAX_ROUNDS exhausted, every round feasible --
        not an empty tube)."""
        a = 0.0
        info = dict(rounds=0, t_out=np.nan, delta=np.nan, noconv=False)
        for r in range(MAX_ROUNDS):
            delta, s_crit, t_out = self.critical(X, o)
            info.update(rounds=r + 1, t_out=t_out, delta=delta)
            a_new, X_new = _lon_search(par32, v0, init, self, delta, s_crit, t_out, mt, abs(a))
            if a_new is None:
                return False, a, X, o, info
            a, X = a_new, X_new
            o = self.check(X)
            if not (o["hit_nc"][0] or o["hit_ttc"][0]):
                return True, a, X, o, info
        info["noconv"] = True
        return False, a, X, o, info

    def dac_station(self, X1: np.ndarray, o: dict, q: int = 0) -> int:
        n = int(round(float(o["crit_time"][q, RT.I_DAC]) / 0.1)) if o["crit_time"][q, RT.I_DAC] >= 0 else 0
        s = frenet(_t(X1[q:q + 1, n:n + 1, :2]), self.fr1)["s"].numpy()[0, 0]
        ds = float(self.fr1["ds"][0])
        return int(np.clip(math.floor(s / ds), 0, N_ST - 1))

    def dac_direction(self, X1: np.ndarray, o: dict, q: int = 0) -> int:
        """+1 (left) / -1 (right): the side of the larger margin at the critical DAC station."""
        j = self.dac_station(X1, o, q)
        mg = RT.margins(self.field, self.fr1, sdf_index=self.sidx)
        if bool(mg["valid"][0, j]) and float(mg["f0"][0, j]) > 0:
            return 1 if float(mg["m_left"][0, j]) >= float(mg["m_right"][0, j]) else -1
        p, n_, h = self.fr1["p"][0, j], self.fr1["n"][0, j], self.fr1["h"][0, j]
        f = []
        for sg in (1.0, -1.0):
            pose = torch.cat([p + sg * 1.0 * n_, h[None]])[None, None]
            cor = ego_corners(pose).reshape(1, 4, 2)
            vv, ok = RT.read_field(self.field, self.fr1, cor, sdf_index=self.sidx)
            f.append(float(torch.where(ok, vv, torch.full_like(vv, -math.inf)).min()))
        return 1 if f[0] >= f[1] else -1


class ExactTube(GridTube):
    """Grid-free GT tube of one parent: exact SAT proxy rules for NC / TTC (exact_flags), the stored SDF corners for
    DAC; the parent frame only measures arrival (arc length) and the DAC direction (RT.margins on the SDF)."""
    kind = "exact"

    def __init__(self, F: dict, gt: Optional[dict], sdf: np.ndarray, p: int = 0):
        self.fr1 = _frame1(F["fr"], p)
        self.gt = gt
        self.sdf = sdf
        self.field = _t(sdf)[None]
        self.sidx = torch.zeros(1, dtype=torch.long)
        self.dummyB = torch.zeros(1, N_ST, N_LAT, 1, dtype=torch.bool)

    def check(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        V = X.shape[0]
        f = exact_flags(X, self.gt, self.sdf)
        o = RT.executed_slack(_t(X), self.fr1, self.dummyB, self.field, index=torch.zeros(V, dtype=torch.long),
                              sdf_index=torch.zeros(V, dtype=torch.long), tbin_s=5.0)
        return {"hit_nc": f["nc_proxy"], "hit_ttc": f["ttc_proxy"], "hit_dac": o["hit_dac"].numpy(),
                "slack_dac": o["slack_dac"].numpy(), "crit_time": o["crit_time"].numpy(),
                "nc_n": f["nc_n"], "ttc_n": f["ttc_n"], "ttc_kstep": f["ttc_kstep"]}

    def lon_edit(self, par32, v0, init, X, o, mt):
        """scan rule: the smallest |a| on A_GRID whose executed states pass the exact NC / TTC proxy with the object
        samples dilated by mt (a place stays blocked mt after the object left it = arrival >= t_out + mt everywhere;
        no bisection: a contact-free boundary found to 1e-3 m/s^2 would sit within float noise of the official
        polygons) -> (ok, a, X, o, info)."""
        T = decode(par32, v0, speeds=A_GRID)[1:]
        Xs = simulate(T, init)
        f = exact_flags(Xs, self.gt, self.sdf, emul=False, dilate=int(round(mt / 0.1)))
        ok = ~(f["nc_proxy"] | f["ttc_proxy"])
        info = dict(rounds=1, t_out=np.nan, delta=np.nan, noconv=False)
        q = int(np.argmax(ok)) if ok.any() else len(A_GRID) - 1
        X1 = Xs[q:q + 1]
        return bool(ok.any()), float(A_GRID[q]), X1, self.check(X1), info


# ----------------------------------------------------------------------------------------------- computed edit
def _lon_search(par32, v0, init, tube, delta, s_crit, t_out, mt, a_min: float):
    """smallest |a| >= a_min on A_GRID (+ bisection) with arrival >= t_out + mt -> (a or None, states of a)."""
    grid = [a for a in A_GRID if abs(a) >= a_min - 1e-12]
    if not grid:
        return None, None
    T = decode(par32, v0, speeds=grid)[1:]
    X = simulate(T, init)
    arr = arrival(X, tube.fr1, delta, s_crit)
    ok = arr >= t_out + mt
    if not ok.any():
        return None, None
    q = int(np.argmax(ok))
    a_hi, X_hi = grid[q], X[q:q + 1]
    a_lo = grid[q - 1] if q > 0 else (-a_min if a_min > 0 else 0.0)
    for _ in range(N_BISECT):
        mid = 0.5 * (a_lo + a_hi)
        if mid == 0.0 or abs(mid - a_hi) < 1e-6:
            break
        Xm = simulate(decode(par32, v0, speeds=(mid,))[1:2], init)
        if arrival(Xm, tube.fr1, delta, s_crit)[0] >= t_out + mt:
            a_hi, X_hi = mid, Xm
        else:
            a_lo = mid
    return float(a_hi), X_hi


def compute_edit(par32: np.ndarray, v0: float, init: np.ndarray, X0: np.ndarray, tube: GridTube,
                 mt: float = MARGIN_T, ml: float = MARGIN_L) -> dict:
    """Minimal computed edit of one parent (module docstring) -> dict kind ('inside' | 'slow' | 'short' | 'lon' |
    'lat' | 'lon+lat' | 'empty_lon' | 'empty_lat' | 'noconv_lon'), a, D, traj [8, 3] float32 (the edit; the parent
    itself when no edit), traj_best (best effort for the empty / noconv cases, else = traj), outside0 [3] (nc, ttc, dac
    hits of the parent), inside_after (the returned traj is inside the tube), rounds, t_out, delta, noconv."""
    X0 = X0[None] if X0.ndim == 2 else X0
    o0 = tube.check(X0)
    hit0 = np.array([o0["hit_nc"][0], o0["hit_ttc"][0], o0["hit_dac"][0]], bool)
    res = dict(kind="inside", a=0.0, D=0.0, traj=par32.copy(), traj_best=par32.copy(), outside0=hit0,
               inside_after=not hit0.any(), rounds=0, t_out=np.nan, delta=np.nan, lat_dir=0)
    if not hit0.any():
        return res
    S8 = float(DraftPath(torch.as_tensor(par32.astype(np.float64))[None]).S[0, -1])
    if v0 < V0_MIN:
        res["kind"] = "slow"
        return res
    if S8 < S8_MIN:
        res["kind"] = "short"
        return res
    a, X, o = 0.0, X0, o0
    if hit0[0] or hit0[1]:
        ok, a, X, o, info = tube.lon_edit(par32, v0, init, X0, o0, mt)
        res.update(info)
        if not ok:
            res.update(kind="noconv_lon" if info["noconv"] else "empty_lon",
                       traj_best=decode(par32, v0, speeds=(A_GRID[-1],))[1])
            return res
    lon_traj = decode(par32, v0, speeds=(a,))[1] if a != 0.0 else par32.copy()
    D = 0.0
    if o["hit_dac"][0]:
        sg = tube.dac_direction(X, o)
        res["lat_dir"] = sg
        lats = tuple(sg * d for d in D_GRID)
        if a != 0.0:
            T = decode(par32, v0, speeds=(a,), lats=lats, combine="cross")[2 + len(lats):]
        else:
            T = decode(par32, v0, lats=lats)[1:]
        XL = simulate(T, init)
        oL = tube.check(XL)
        okL = oL["slack_dac"] >= ml
        if not okL.any():
            qb = int(np.argmax(oL["slack_dac"]))
            res.update(kind="empty_lat", a=a, traj_best=T[qb], D=0.0)
            return res
        q = int(np.argmax(okL))
        D, final, Xf = float(lats[q]), T[q], XL[q:q + 1]
    else:
        final, Xf = lon_traj, X
    of = tube.check(Xf)
    kind = "lon+lat" if (a != 0.0 and D != 0.0) else ("lon" if a != 0.0 else "lat")
    res.update(kind=kind, a=a, D=D, traj=final, traj_best=final,
               inside_after=not (of["hit_nc"][0] or of["hit_ttc"][0] or of["hit_dac"][0]))
    return res


# ----------------------------------------------------------------------------------------------- speed-cap probe
def speed_cap_traj(par: np.ndarray, F: dict, sdf: np.ndarray, p: int = 0) -> Tuple[Optional[np.ndarray], dict]:
    """Speed-capped re-timing of parent p (module docstring) -> (traj [8, 3] float32 or None if never over v_max,
    info: over (bool), n_over (stations), vmax_min (m/s over the in-plan stations), dv_max (max planned - cap))."""
    fr1 = _frame1(F["fr"], p)
    mg = RT.margins(_t(sdf)[None], fr1, sdf_index=torch.zeros(1, dtype=torch.long))
    kap = fr1["kappa"]
    vmax = RT.v_max(kap, RT.outside_margin(kap, mg["m_left"], mg["m_right"]))[0].numpy()    # [48]
    path = DraftPath(torch.as_tensor(par[p:p + 1], dtype=F64))
    S = path.S[0].numpy()                                                         # [9] knots, S_0 = 0
    u = np.diff(S) / 0.5                                                          # [8] planned segment speeds
    s_st = fr1["s"][0].numpy()
    ds = float(fr1["ds"][0])
    inplan = (s_st <= S[-1]) & mg["valid"][0].numpy()
    seg = np.clip(np.searchsorted(S, s_st, side="left") - 1, 0, 7)
    over = inplan & (u[seg] > vmax + 1e-6)
    info = dict(over=bool(over.any()), n_over=int(over.sum()),
                vmax_min=float(vmax[inplan].min()) if inplan.any() else np.nan,
                dv_max=float((u[seg] - vmax)[over].max()) if over.any() else 0.0)
    if not over.any() or S[-1] < 1e-3:
        return None, info
    # planned first-arrival time t_p(s) (segment k-1 -> k of positive length holds s in (S_{k-1}, S_k]) + the delay the
    # cap adds, int max(0, 1 / v_cap - 1 / v_plan) ds: planned waits (zero-length segments) are kept, never faster
    sg = np.linspace(0.0, S[-1], max(int(S[-1] / 0.05), 2) + 1)
    kk = np.clip(np.searchsorted(S, sg, side="left"), 1, 8)                      # knot index k of s's segment
    vp = u[kk - 1]
    t_p = np.where(sg > 0, 0.5 * (kk - 1) + (sg - S[kk - 1]) / np.maximum(vp, 1e-9), 0.0)
    jj = np.clip(np.floor(sg / ds).astype(np.int64), 0, N_ST - 1)
    cap = np.where(inplan[jj], np.maximum(vmax[jj], V_FLOOR), np.inf)
    vc = np.maximum(np.minimum(vp, cap), 1e-3)
    extra = np.maximum(1.0 / vc - 1.0 / np.maximum(vp, 1e-9), 0.0)
    tc = t_p + np.concatenate([[0.0], np.cumsum(np.diff(sg) * 0.5 * (extra[1:] + extra[:-1]))])
    tc = np.maximum.accumulate(tc)
    s_new = np.interp(np.arange(1, 9) * 0.5, tc, sg, right=S[-1])
    pe = path.eval(torch.as_tensor(s_new, dtype=F64)[None])
    traj = torch.cat([pe.xy[0], pe.heading[0][:, None]], -1).numpy().astype(np.float32)
    return traj, info


# ----------------------------------------------------------------------------------------------- scene bins
def scene_bins(X_top1: np.ndarray, gt: Optional[dict], sdf: np.ndarray) -> Dict[str, float]:
    """token-level scene descriptors from the v2 top-1 executed states: min_corner (stored SDF, m; near curb < 1 m),
    obj_dmin (min over t_n, n = 0..40, of the distance OBS GT box centre -> executed rear axle, m; lane-only >= 30 m;
    +inf without objects)."""
    vm, _ = corner_sdf(X_top1[None], sdf)
    dmin = np.inf
    if gt is not None and gt["boxes"].shape[0]:
        c = gt["boxes"][:, :41, :2]
        d = np.hypot(c[..., 0] - X_top1[None, :, 0], c[..., 1] - X_top1[None, :, 1])
        d = np.where(gt["obs"][:, :41], d, np.inf)
        dmin = float(d.min())
    return {"min_corner": float(vm.min()), "obj_dmin": dmin}


# ----------------------------------------------------------------------------------------------- official scorer
class OfficialScorer:
    """tools/refiner/score_trajectories.score_token on the navtrain_val metric cache (plain navsim classes)."""

    def __init__(self):
        p = str(REPO / "tools/refiner")
        if p not in sys.path:
            sys.path.insert(0, p)
        import score_trajectories as ST  # noqa: E402
        self.ST = ST
        self.sim, self.scorer = ST.build_simulator_scorer(record=False)

    def score(self, log: str, token: str, trajs: np.ndarray) -> np.ndarray:
        """[K, 8, 3] -> [K, 11] (SCORE_COLS: nc, dac, ddc, ep, ttc, comfort, pdms, raw_progress, dac / nc / ttc time
        idx) float64."""
        mc = self.ST.load_metric_cache(CM.mc_path(SPLIT, log, token))
        res = self.ST.score_token(mc, np.asarray(trajs, np.float32), self.sim, self.scorer)
        return np.array([[float(r[c]) for c in SCORE_COLS] for r in res], np.float64)


def exec_to_score(e: np.ndarray) -> np.ndarray:
    """F1 exec [.., 11] (EXEC_COLS) -> SCORE_COLS order (first_off -> dac_time_idx)."""
    return np.asarray(e, np.float64)[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
