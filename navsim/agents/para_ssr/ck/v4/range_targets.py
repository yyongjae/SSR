"""refiner v4_3 M2 (docs/refiner_v4/V4_3_SPEC.md): GT range fields on the PARENT grid and their 0-parameter reductions.

Frames / units: NAVSIM N frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x; metres, seconds, rad).
Parent grid (refiner.corridor.corridor_geometry, any corridor version): 48 stations s_j = (j + 0.5) ds on the parent's
  smooth path Gamma_p (ds = max(1 m, S_look / 48)) x 17 lateral offsets d_i = -4.8 + 0.6 i (left +) along the path's
  left unit normal n_j:  q_ij = Gamma_p(s_j) + d_i n_j  (= CorridorGeom.points [P, 48, 17, 2]).  A "cell" (j, i) is that
  grid point; station j's arc-length bin is [j ds, (j + 1) ds).
Ego box (OFFICIAL, = metric-cache vehicle / refiner.geometry / sdf.ego_corners): half length 2.588, half width 1.1485,
  box centre RA2C = 1.461 m ahead of the rear axle.  On the grid the REAR AXLE sits at q_ij with heading h_j:
  heading='parent' (default) = the tangent heading of Gamma_p at s_j, 'draft' = the parent's draft heading h0(s_j)
  (pose headings interpolated in arc length; DraftPath.eval 'heading').
Time: GT objects are read on the 0.1 s grid t_n = 0.1 n (fine_times(): n = 0..49, 0 .. 4.9 s -- the latest object
  time the official scorer reads is the TTC look-up 4.0 + 0.9 s) and max-pooled into M bins of tbin_s (default 0.25 s,
  M = round(T_END / tbin_s) = 20 bins to 5.0 s): bin m = [m tbin_s, (m + 1) tbin_s), a sample t joins bin
  floor(t / tbin_s + 1e-6).

Fields
  blocked_field(parent_geom, boxes, state, t_grid, ...) -> B [P, 48, 17, M] bool
      B[p, j, i, m] = for some sample t in bin m and some GT object a of the parent's scene with state OBS (metric-cache
      rule, refiner.gt_future.query / query_torch): the ego box at cell (j, i) overlaps a's box (exact SAT,
      refiner.surrogate.box_gaps, max gap <= 0: touching counts, as shapely 'intersects') AND a is not behind the ego
      (object centre >= BEHIND_M = -1.3 m along the cell heading from the ego box centre: the at-fault proxy of
      surrogate.C_col for the official 'is_agent_behind -> rear collision, not at fault').
      Computed sparsely (exact; superset culls, no approximation): object centres outside the parent grid's bounding box
      grown by r_ego + r_obj (half diagonals) are dropped, then per surviving (parent, object, time) the stations whose
      lateral segment {Gamma(s_j) + RA2C e(h_j) + d n_j : |d| <= 4.8} is farther than r_ego + r_obj from the object
      centre, then the 17 SAT tests of each remaining station.  torch (training: online from ref_obj_*,
      gt_boxes_from_ref) and a float64 numpy twin blocked_field_np (oracle; identical booleans,
      tests/test_v4_range_targets.py).
      lon_max=True (opt-in) also returns LON [P, 48, 17, M] = the max over the SAT-overlapping OBS objects (no
      not-behind filter) of lon = (c_obj - c_cell) . e(h_j), -inf where nothing overlaps; B == (LON >= BEHIND_M)
      exactly.  B alone fixes the not-behind reference at the CELL's box; LON lets executed_slack move it (TTC below).
  margins(D or sdf, parent_geom) -> m_left, m_right [P, 48] (m), L [P] (m)
      f_j(delta) = min over the 4 corners of the ego box with rear axle at Gamma(s_j) + delta n_j, heading h_j, of the
      field: 'grid' D [P, 48, 17] (a predicted / GT margin on the parent grid points; corners mapped to the grid by the
      Frenet projection on the parent's station frame, temporal.raster.frenet -- evaluated every PROJ_STEP = 0.3 m of
      the scan and linear in between, the corners move on straight lines as delta changes -- then bilinear,
      evidence.grid_read; arc lengths clamped to [s_0, s_47] -- the grid has no row behind s_0 = ds / 2 or beyond
      s_47 -- lateral offsets beyond 4.8 m unreadable) or 'sdf' [S, 320, 256] (the official DAC SDF, sdf.sample_sdf at
      the corner points; outside the E grid unreadable).  Scan delta = 0, +-step, .. +-lat_max (step 0.05 m,
      lat_max 4.8 m):
        m_left (m_right) = the first crossing of f <= 0 (or unreadable) going left (right) from delta = 0, linearly
          interpolated between the last positive and the first non-positive sample (an unreadable sample: crossing at
          the last positive one, flagged m_unread: the edge of the field, not a boundary -- with 'grid' D a straight
          parent reads at most ~3.65 m of shift); lat_max if f stays positive (m_capped);  f_j(0) <= 0 (centred box
          already outside): m_left = m_right = f_j(0) (<= 0: how far the centred box is outside); f_j(0) unreadable:
          0, valid False (and m_unread).
        L = arc length where the centred box first exits: the first station j with f_j(0) <= 0 or unreadable,
          L = s_{j-1} + ds f_{j-1}(0) / (f_{j-1}(0) - f_j(0)) (s_{j-1} if f_j(0) unreadable; 0 if j = 0); 48 ds (grid
          end) if no station exits (L_exit False).
  v_max(kappa, m_out, fit) -> [P, 48] m/s
      largest v with e(v |kappa|) + TRACK_MARGIN (0.3 m) <= m_out, e = the expected tracking path error of the
      official LQR + bicycle as a function of the trajectory's maximum yaw rate (piecewise-linear table TRACK_ERR_MEDIAN
      below; held at its first value below the first knot, extended with the last slope beyond the last knot): 0 if
      m_out - 0.3 < e(0), V_CAP (40 m/s) where kappa = 0 or the limit exceeds it.  m_out = the margin on the OUTSIDE of
      the turn (outside_margin: kappa > 0 -> m_right; the tracker executes turns wide, F1 notes T2).  comfort_prior=True
      also caps v by the yaw-rate / lateral-acceleration limits of refiner.geometry.kappa_limit (0.95 / v, 4.89 / v^2;
      its 0.213 /m human-curvature cap is a path property, not a speed limit, and is not used).  Off by default.
  windows(B) -> t_in_block, t_out_block [P, 48, 17] (s)
      per cell: t_in_block = the start of its first blocked bin (+inf if never blocked: the fast side, pass before it);
      t_out_block = the end of its LAST blocked bin, i.e. the time from which the cell stays free to the end of the
      horizon (0 if never blocked = free from t0; +inf if the last (known) bin is blocked: never free again within the
      horizon, e.g. a stopped lead).  Arriving at a cell at t >= t_out_block is free for the rest of the horizon (the
      M5 longitudinal edit target arrival >= t_out_block + 0.2 s).  n_runs = number of separate blocked runs (1 for a
      single passing / stopped object).  Reading of the spec sentence: "+inf if never blocked" is taken to describe
      t_in_block and "0 if free from 0" t_out_block (a never-blocked cell needs no wait) -- to be confirmed by the
      designer.  known [P, M]: bins after the last known bin are ignored; a parent with NO known bin is unknown, not
      free: t_in_block = 0, t_out_block = +inf.
  executed_slack(states [B, 41, 6], parent_geom, B, D or sdf) -> per candidate on its EXECUTED (LQR-tracked) states:
      states x, y, heading, v (+ a, steer) in N at t_n = 0.1 n, n = 0..40 (ck/v4/lqr.simulate, F1 run.npz 'sim').
      Rear axle (x, y) -> (s, d) on the parent's station frame (frenet) -> cell (j = floor(s / ds), i = round((d + 4.8)
      / 0.6); read='any4': the OR of the 4 surrounding grid points, conservative); outside the grid (s < -ds / 2,
      s >= 48 ds, |d| > 5.1 m): not read (n_oog).  Read times t_n + delta are binned in float64 whatever the dtype
      (float32 t / 0.1 puts 3.7 / 4.2 / 4.7 s one bin early).  Per read state, with the cell's blocked runs [a, b)
      (bin edges, s):
        in a run   (t in [a, b)): slack_n = t - b < 0   (how much later the ego would have to arrive to clear it;
                                  -slack_cap if b = +inf, the run reaches the last bin)
        free       : slack_n = min(t - b_prev, a_next - t) >= 0 (time distance to the nearest blocked time of the cell;
                     slack_cap if the cell is never blocked; clamped at 0: with T_EPS a sample can sit 1e-16 s before
                     its bin's start, 0.1 s bins only)
      Speeds are |v| (official speeds = hypot(vx, vy)): for the skips below and the TTC projection (a reversing state
      is projected forward along its heading).
      NC  slack_nc = min_n slack_n over the executed states (ego speed <= STOP_NC = 0.05 m/s skipped: the official
          STOPPED_EGO collision is not at fault, pdm_scorer_utils.get_collision_type); hit_nc = slack_nc < 0 <=> the
          executed path enters a blocked cell at a blocked time.
      TTC slack_ttc = the same over the states projected |v_n| delta along the executed heading h_n, delta in
          TTC_DELTAS = (0, 0.3, 0.6, 0.9) s (official future_time_idcs 0, 3, 6, 9), read at bin(t_n + delta); speed <
          STOP_TTC = 5e-3 m/s skipped (official stopped_speed_threshold).  crit_time = the EGO time t_n of the flagged
          step (the official ttc_time_idx convention), crit_station = the station of the projected cell.
          Not-behind reference: B tests it from the PROJECTED cell's box, but the official _calculate_ttc tests
          is_agent_ahead / is_agent_behind from the CURRENT rear axle, so an object between the current ego and the
          projected box is 'behind' for B and 'ahead' officially.  lon_field=LON (opt-in) reads the projection's cell
          as LON >= BEHIND_M - |v_n| delta (= the not-behind test from the current ego box centre, exact up to the
          cell rounding and the parent-vs-executed heading; delta 0 = B).  A predicted-field readout (M4) needs the
          same second channel (B without not-behind, or LON bins) to do this.
      DAC slack_dac = min over states and the 4 corners (executed heading, exact) of the field ('grid' D read as in
          margins, 'sdf' sample_sdf; unreadable corners excluded, no readable corner at all -> +SDF cap 10 m); hit_dac
          = slack_dac < 0 (official DAC fails iff a corner of a tracked state is outside the drivable layers <=> SDF
          < 0).
      crit_station / crit_time [B, 3] (nc, ttc, dac) of the minimising state (-1 / -1.0 when slack_nc / slack_ttc =
      slack_cap: nothing blocked was ever near, or when no DAC corner is readable; the DAC station is -1 for a state
      off the parent grid); slack_* clamped to [-slack_cap, slack_cap] (5 s) for NC / TTC.

What the field does NOT model (official exceptions; the oracle reports them separately)
  * at-fault classification beyond 'not behind': lateral contacts count only when the ego is in multiple lanes or off
    the drivable area (needs the lane map), ACTIVE_FRONT needs the front bumper segment, stopped TRACKS are at fault
    even from behind; the field counts every not-behind overlap.
  * the official per-proposal collided-track memory (after a non-at-fault contact the track is ignored for the rest of
    the trajectory) and the TTC 'ahead within 30 deg / multiple lanes / intersection' rule.
  * the ego heading on the grid is the parent's (the executed heading differs; p90 0.14 rad for v2 top-1, F1).
  * NC 0.5 (static objects) and 0 (agents) are not distinguished: both are failures (y < 1); obj_mask restricts the
    object set if a caller needs agents only.

Tracking-error table (v_max) -- provenance
  F1 (CK/ck2/analysis/detmap_path_ideas/F1/out/val/feats.parquet, notes.md T2): v2 r34 top-16 parents of 6,000 seeded
  navtrain_val tokens (vtype 'par', 96,000 rows), x = max_yaw_rate (max_k |wrap(h_k - h_{k-1})| / 0.5 of the submitted
  poses, rad/s), y = max_path (max over the 41 executed states of the official PDMSimulator of the rear axle's distance
  to the submitted polyline extended 30 m straight, m).  fit_track_table: bins [0, .02, .05, .075, .1, .15, .2, .3, .4,
  .5, .7) rad/s, knot = (median x, median y) per bin (n 1,650 .. 20,364 per bin), monotone (cummax).  Bins >= 0.7 rad/s
  (n 283 / 238 / 2,371) are not used: their median yaw rate (0.8 / 1.2 / 4.8 rad/s) is heading jitter of near-stopped
  drafts (a 0.5 s heading flip), not turning, and their path error is not monotone (0.84 / 1.18 / 1.37 m).
  The p90 table of the same bins is kept as TRACK_ERR_P90 (conservative option).  Generated 2026-10-16 (KST);
  tests/test_v4_range_targets.py::test_track_table_provenance refits it from the parquet.

Validation on real data (2026-10-16 KST; nothing tuned on labels -- the 'main' setting was fixed before the run:
corridor version 2, heading 'parent', 0.25 s bins, nearest cell, not-behind proxy, official stopped-ego skips)
  data: 6,000 F1 navtrain_val tokens (seed 0), their v2 r34 top-16 parents, the F1 executed states (official
  PDMSimulator, labels bit-exact), GT objects CK/v3/gt_objects/navtrain_val, official SDF sdf/navtrain_val; official
  labels = F1 exec (equal to the cand label shards: 0 nc / nc_time_idx mismatches).  'hit' vs official fail (< 1):
                               v2 top-1 (6,000)                           all 16 parents (96,000)
    NC  field (main)           fail recall 0.900 (45/50), pass 0.9931       0.846 (1,979/2,340), pass 0.9821
    TTC field (main)           0.772 (159/206), pass 0.9858                 0.794 (4,905/6,174), pass 0.9709
    TTC lon_field (current-ego 0.820 (169/206), pass 0.9841                 0.871 (5,378/6,174), pass 0.9653
        not-behind; opt-in)
    DAC executed corners on D  0.949 (260/274), pass 0.9998                 0.905 (5,075/5,606), pass 0.9986
        (D = the SDF at the parent grid points; on the SDF itself 0.989 / 1.000, the official-equivalent check)
    joint (NC or TTC or DAC)   0.878 (410/467), pass 0.9859                 0.860, pass 0.9710
    joint with TTC lon_field   0.899 (420/467), pass 0.9841                 0.902 (10,280/11,403), pass 0.9656
  decomposition (TTC; review 2026-10-16): on the exact executed boxes (no grid) TTC with the not-behind test from the
    CURRENT ego centre (the official reference) gives 1.000 / 0.9990 (all 16 0.981 / 0.9931), from the PROJECTED box
    centre (B's convention) 0.951 / 0.9990 (all 16 0.887 / 0.9950): of the 1,269 all-16 TTC misses of the field
    (main), the reference point alone accounts for 581 (46 %) on exact boxes, not the grid; lon_field recovers 473
    (37 %; 0 NC changes, 973 TTC hits added, none removed).  The remaining TTC loss is the grid (cell rounding, parent
    heading).  NC and TTC over all 16 parents (0.846 / 0.871) stay below the 0.9 bar of oracle criterion 1.
  decomposition (top-1 NC): the same rules on the exact executed boxes (no grid) give 0.980 / 0.9987, and an
    emulation of the official at-fault classification on the GT boxes (stopped track / rear by angle > 150 deg / front
    bumper / lateral when off-road, collided-track memory) gives 0.96 - 1.00 / 0.9993 - 1.000 (the
    single-lane lateral rule needs the lane map: the two bounds) -> the GT boxes reproduce the official labels and the
    loss is the grid: of 41 field false alarms 33 are grid-only (exact boxes free), 5 rear / 3 lateral contacts the
    official calls not at fault; of 5 misses 4 are grid misses, 1 the not-behind proxy (all 16: FP 1,283 grid-only,
    328 rear, 66 lateral; FN 239 grid, 122 not-behind proxy).  The first field hit lies within 0.25 s of the official
    nc_time_idx for 91 % of the detected top-1 failures (100 % within 0.5 s; all 16 parents 74 % / 92 %).
  sensitivity (top-1 NC fail / pass recall): 0.1 s bins 0.880 / 0.9945, 0.5 s 0.900 / 0.9919, heading 'draft'
    0.920 / 0.9946, corridor version 1 0.900 / 0.9862, no not-behind 0.920 / 0.9911, no stopped-ego skip identical;
    read 'any4' (post hoc, conservative) NC 0.980 / 0.9492, TTC 0.903 / 0.9285 (all 16: NC 0.932 / 0.9342; with
    lon_field TTC 0.932 / 0.9225, all 16 0.955 / 0.9016).  TTC lon_field at 0.1 / 0.5 s bins: 0.816 / 0.9855,
    0.830 / 0.9788 (all 16 0.863 / 0.9689, 0.881 / 0.9569).  Review fixes (|v|, float64 read bins, DAC crit) change
    no main hit (0 flips of 96,000); the free-slack clamp at 0 removes 35 spurious 0.1 s-bin TTC hits (all 16: 6 of
    them on failures; slack -1e-16 at a bin start), none at 0.25 / 0.5 s.
  tube test (executed rear axle outside [-m_right, m_left] of its station, parent heading): DAC fail recall 0.869 /
    pass 0.9387 on D, 0.898 / 0.9590 on the SDF (top-1): the parent-heading tube is a weaker DAC proxy than reading
    the executed corners.  v_max (descriptive): top-1 parents planned faster than v_max at some in-plan station
    (sdf margins) fail DAC in 18.4 % (no room, m_out < 0.33 m: 29.7 %; curvature-limited: 8.7 %) vs 0.3 % otherwise.
  twins: numpy vs torch float64 identical on 300 tokens (78.3 M cells); torch float32 differs in 7 cells (SAT ties).
    LON: B == (LON >= BEHIND_M) on all 6,000 tokens (numpy) and on 150 (torch float32); torch float64 vs numpy LON:
    same finite cells, max |diff| 1.8e-15 m (150 tokens); float32 vs float64 executed_slack: 0 hit differences.
  cost per token, 16 parents, M = 20 (CPU, 1 thread, 100 val tokens, mean A 57): blocked_field float32 53 ms (p90 131,
    max 260), gt_boxes_from_ref 0.5 ms, margins 'grid' 127 ms (p90 167), executed_slack (NC + TTC + DAC on the SDF)
    6 ms; RSS growth on the 10 heaviest scenes (A 242 - 310) 13 MB (field) / 64 MB (margins).  lon_max=True adds
    ~8 % to blocked_field (84 -> 91 ms mean on a loaded machine, same 100 tokens interleaved), lon_field nothing to
    executed_slack (6 ms); LON [16, 48, 17, 20] float32 = 1.0 MB.  Output B [16, 48, 17,
    20] = 261,120 cells (255 KB bool, 1.0 MB as float32 logits); 12 % of the cells are blocked on average (p90 30 %).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ...refiner.corridor import DS_MIN, LAT_MAX, N_LAT, N_ST, lateral_offsets
from ...refiner.geometry import HALF_LEN, HALF_WID, LAT_ACC_MAX, RA2C, YAW_RATE_MAX
from ...refiner.sdf import SDF_CLIP, ego_corners, sample_sdf
from ...refiner.surrogate import BEHIND_M, box_gaps
from ..temporal.raster import frenet
from .evidence import grid_read

Tensor = torch.Tensor

# ----------------------------------------------------------------------------------------------- constants
OBS = 1                          # gt_future.OBS
DT_FINE = 0.1                    # [s] object sample spacing (official 0.1 s grid)
N_FINE = 50                      # samples 0 .. 4.9 s (TTC reads object time 40 + 9 = 49)
TBIN_S = 0.25                    # [s] default time bin
T_END = 5.0                      # [s] bins cover [0, T_END)
LAT_STEP = 2 * LAT_MAX / (N_LAT - 1)            # 0.6 m
R_EGO = math.hypot(HALF_LEN, HALF_WID)          # ego half diagonal [m]
CULL_EPS = 1e-3                  # [m] cull slack: the culls stay supersets under float32 rounding (SAT ties only)
T_EPS = 1e-6                     # [s] bin assignment tolerance
CHUNK = {"cpu": 1 << 16, "cuda": 1 << 22}       # blocked_field elements per temporary (exact for any value)
TTC_DELTAS = (0.0, 0.3, 0.6, 0.9)               # official future_time_idcs (0, 3, 6, 9) x 0.1 s
STOP_NC = 5e-2                   # [m/s] get_collision_type stopped ego (not at fault)
STOP_TTC = 5e-3                  # [m/s] default_scoring_parameters stopped_speed_threshold (TTC skip)
SLACK_CAP = 5.0                  # [s] NC / TTC slack clamp
SCAN_STEP = 0.05                 # [m] margins lateral scan step
PROJ_STEP = 0.3                  # [m] margins 'grid': Frenet projection of the scan corners every 0.3 m, linear between
TRACK_MARGIN = 0.3               # [m] v_max safety margin on top of the expected tracking error
V_CAP = 40.0                     # [m/s] v_max where nothing limits

# tracking path error vs max yaw rate (module docstring 'Tracking-error table'): (yaw rate rad/s, path error m)
TRACK_YAW = (0.013, 0.033, 0.061, 0.086, 0.123, 0.173, 0.248, 0.344, 0.437, 0.544)
TRACK_ERR_MEDIAN = (0.032, 0.080, 0.160, 0.217, 0.300, 0.410, 0.564, 0.745, 0.872, 1.010)
TRACK_ERR_P90 = (0.063, 0.150, 0.270, 0.355, 0.498, 0.640, 0.858, 1.072, 1.176, 1.344)
TRACK_BIN_EDGES = (0.0, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
TRACK_FIT = (TRACK_YAW, TRACK_ERR_MEDIAN)
TRACK_FIT_P90 = (TRACK_YAW, TRACK_ERR_P90)
I_NC, I_TTC, I_DAC = 0, 1, 2     # crit_* column order


def fine_times(n: int = N_FINE, dt: float = DT_FINE) -> np.ndarray:
    """object sample times t_n = 0.1 n [s], n = 0..n-1 (float64)."""
    return np.arange(int(n), dtype=np.float64) * float(dt)


def n_bins_of(tbin_s: float, t_end: float = T_END) -> int:
    """M = round(t_end / tbin_s) (20 for 0.25 s, 50 for 0.1 s, 10 for 0.5 s)."""
    return int(round(float(t_end) / float(tbin_s)))


def time_bins(t, tbin_s: float = TBIN_S) -> np.ndarray:
    """bin index floor(t / tbin_s + 1e-6) of sample times t (numpy int64; T_EPS keeps 0.5 / 0.25 in bin 2)."""
    return np.floor(np.asarray(t, np.float64) / float(tbin_s) + T_EPS).astype(np.int64)


# ----------------------------------------------------------------------------------------------- parent frame
def parent_frame(pgeom, heading: str = "parent") -> Dict[str, Tensor]:
    """Station frame of the parents (dtype of the geometry; no_grad): pts [P, 48, 17, 2] (= points, rear axle of every
    cell), p (station centreline Gamma(s_j)), u (unit tangent), n (unit left normal) [P, 48, 2], h [P, 48] (cell heading
    per `heading`), s [P, 48], ds [P].  A dict passes through (already a frame, e.g. gathered per candidate)."""
    if isinstance(pgeom, dict):
        return pgeom
    if heading not in ("parent", "draft"):
        raise ValueError(f"heading {heading!r} not in ('parent', 'draft')")
    with torch.no_grad():
        pe = pgeom.path.eval(pgeom.s)
        h = pe.tangent_heading if heading == "parent" else pe.heading
        ds = torch.clamp(pgeom.S_look / N_ST, min=DS_MIN)
        return {"pts": pgeom.points, "p": pe.xy, "u": pe.tangent, "n": pe.normal, "h": h, "s": pgeom.s, "ds": ds,
                "kappa": pe.kappa}


def gather_frame(fr: Dict[str, Tensor], index: Tensor) -> Dict[str, Tensor]:
    """frame rows of parents index [B] (candidate b -> its parent's row)."""
    return {k: v[index] for k, v in fr.items()}


def _index(index, B: int, S: int, device) -> Tensor:
    if index is None:
        if S == B:
            return torch.arange(B, device=device)
        if S == 1:
            return torch.zeros(B, dtype=torch.long, device=device)
        raise ValueError(f"{S} rows for {B} parents / candidates: pass index")
    index = torch.as_tensor(index, device=device).long().reshape(-1)
    if index.shape[0] != B:
        raise ValueError(f"index must be [{B}], got {tuple(index.shape)}")
    return index


# ----------------------------------------------------------------------------------------------- GT objects
def gt_boxes_from_ref(ref: Dict[str, Tensor], tok: Tensor, t_grid=None, device=None):
    """GT boxes of tokens tok [N] from the batch's ref_obj_* tensors (as losses.evidence_aux; gt_future.query_torch,
    metric-cache rule) at t_grid [T] (default fine_times(), 0 .. 4.9 s) -> boxes [N, A, T, 5] (cx, cy, heading, L, W;
    float64), obs [N, A, T] bool, is_agent [N, A] bool, t_avail [N] (s)."""
    from ...refiner import gt_future as GF
    dev = device if device is not None else ref["ref_obj_kf"].device
    tok = torch.as_tensor(tok, device=ref["ref_obj_kf"].device).long()
    n = ref["ref_obj_n"][tok].long()
    a = int(max(int(n.max()) if n.numel() else 0, 1))
    kf = ref["ref_obj_kf"][tok][:, :a].to(dev)
    first = ref["ref_obj_first"][tok][:, :a].to(dev)
    meta = ref["ref_obj_meta"][tok][:, :a].to(dev).long()
    valid = torch.arange(a, device=dev)[None] < n.to(dev)[:, None]
    t = fine_times() if t_grid is None else t_grid
    n_kf = ref["ref_obj_n_kf"][tok].to(dev)
    with torch.no_grad():
        boxes, st = GF.query_torch(kf.double(), first.double(), meta, t, valid=valid, n_kf=n_kf)
    t_av = ((n_kf.double() - 1) * GF.KF_DT).clamp(max=GF.T_MAX)
    return boxes, st.to(torch.int64) == OBS, (meta[..., GF.MT_AGENT] > 0) & valid, t_av


# ----------------------------------------------------------------------------------------------- blocked field
def _cull_bbox(c0: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
    """bounding box [P, 2] lo / hi of the ego box centres of all cells (segment end points c0 +- 4.8 n)."""
    e = torch.cat([c0 - LAT_MAX * n, c0 + LAT_MAX * n], 1)                       # [P, 96, 2]
    return e.min(1).values, e.max(1).values


@torch.no_grad()
def blocked_field(parent_geom, boxes: Tensor, state: Tensor, t_grid=None, index=None, tbin_s: float = TBIN_S,
                  n_bins: Optional[int] = None, heading: str = "parent", behind_m: Optional[float] = BEHIND_M,
                  obj_mask: Optional[Tensor] = None, t_avail: Optional[Tensor] = None, details: bool = False,
                  chunk: Optional[int] = None, lon_max: bool = False):
    """Blocked field B [P, 48, 17, M] bool of P parents (module docstring 'Fields').

    parent_geom : CorridorGeom of the P parents (or a parent_frame dict).
    boxes       : [S, A, T, 5] GT boxes (cx, cy, heading, L, W) of S scenes at t_grid [T] (s);
    state       : [S, A, T] int8 (OBS = 1) or bool (True = OBS); t_grid default fine_times() (0 .. 4.9 s);
                  index [P]: scene row of parent p (default arange if S == P, zeros if S == 1);
                  obj_mask [S, A] bool optional object subset (e.g. agents only).
    tbin_s / n_bins: time bins (M = round(5.0 / tbin_s) by default); samples beyond the last bin are ignored.
    t_avail [S] optional annotated horizon (s): bins with a sample t > t_avail are not 'known' (details only).
    behind_m    : not-behind threshold (None: every overlap counts).
    details     : False -> B;  True -> dict(blocked=B, fine [P, 48, 17, T] bool (per sample), known [P, M] bool,
                  n_pairs (SAT pairs evaluated), n_cand (station candidates)).
    chunk       : elements per temporary (default CHUNK by device: CPU 2^16, CUDA 2^22; exact for any value).
    lon_max     : opt-in (default off: outputs unchanged) LON [P, 48, 17, M] (geometry dtype, m) = per cell and bin the
                  max over the SAT-overlapping OBS objects (NO not-behind filter) of the object centre's along-heading
                  offset (c_obj - c_cell) . e(h_j) from the cell's ego box centre, -inf where nothing overlaps.  Then
                  B == (LON >= behind_m) exactly (B == LON > -inf for behind_m None), and the not-behind test can be
                  re-read from another reference point (executed_slack(lon_field=): TTC from the CURRENT ego).
                  Returned as (B, LON) (details: dict key 'lon_max').
    """
    fr = parent_frame(parent_geom, heading)
    pts, n_, h = fr["pts"], fr["n"], fr["h"]
    P = pts.shape[0]
    dev, dt = pts.device, pts.dtype
    S, A, T = boxes.shape[:3]
    chunk = int(CHUNK.get(dev.type, CHUNK["cpu"]) if chunk is None else chunk)
    idx = _index(index, P, S, dev)
    t = torch.as_tensor(fine_times() if t_grid is None else t_grid, dtype=torch.float64, device=dev).reshape(-1)
    if t.shape[0] != T:
        raise ValueError(f"t_grid has {t.shape[0]} times, boxes {T}")
    M = n_bins_of(tbin_s) if n_bins is None else int(n_bins)
    mt = torch.floor(t / float(tbin_s) + T_EPS).long()                            # [T]
    t_ok = (mt >= 0) & (mt < M)
    obs = state.bool() if state.dtype == torch.bool else (state.to(torch.int64) == OBS)
    obs = obs.to(dev)
    if obj_mask is not None:
        obs = obs & obj_mask.to(dev).bool()[..., None]
    obs = obs & t_ok[None, None, :]
    bx = boxes.to(device=dev, dtype=dt)
    eh = torch.stack([torch.cos(h), torch.sin(h)], -1)                            # [P, 48, 2]
    c0 = fr["p"] + RA2C * eh                                                      # centred-cell box centres
    lo, hi = _cull_bbox(c0, n_)
    blocked = torch.zeros(P, N_ST, N_LAT, M, dtype=torch.bool, device=dev)
    fine = torch.zeros(P, N_ST, N_LAT, T, dtype=torch.bool, device=dev) if details else None
    lonf = torch.full((P * N_ST * N_LAT * M,), -math.inf, dtype=dt, device=dev) if lon_max else None   # flat LON
    n_pairs = n_cand = 0
    d = lateral_offsets(dev, dt)                                                  # [17]
    # 1. object level: parent p, object a, time t with a OBS and its centre inside the grown bounding box of p's cells
    for p0 in range(0, P, max(1, chunk // max(A * T, 1))):
        p1 = min(P, p0 + max(1, chunk // max(A * T, 1)))
        b = bx[idx[p0:p1]]                                                        # [p, A, T, 5]
        r = R_EGO + 0.5 * torch.hypot(b[..., 3], b[..., 4]) + CULL_EPS
        inb = ((b[..., 0] >= lo[p0:p1, 0, None, None] - r) & (b[..., 0] <= hi[p0:p1, 0, None, None] + r)
               & (b[..., 1] >= lo[p0:p1, 1, None, None] - r) & (b[..., 1] <= hi[p0:p1, 1, None, None] + r))
        pi, ai, ti = torch.nonzero(inb & obs[idx[p0:p1]], as_tuple=True)
        pi = pi + p0
        # 2. station level: distance of the object centre to station j's segment of ego box centres
        for k0 in range(0, pi.numel(), max(1, chunk // N_ST)):
            k1 = min(pi.numel(), k0 + max(1, chunk // N_ST))
            pk, ak, tk = pi[k0:k1], ai[k0:k1], ti[k0:k1]
            ob = bx[idx[pk], ak, tk]                                              # [k, 5]
            rk = R_EGO + 0.5 * torch.hypot(ob[:, 3], ob[:, 4]) + CULL_EPS
            rel = ob[:, None, :2] - c0[pk]                                        # [k, 48, 2]
            nn_ = n_[pk]
            bl = (rel * nn_).sum(-1)
            perp = rel - bl[..., None] * nn_
            dseg = torch.hypot(torch.linalg.vector_norm(perp, dim=-1), torch.clamp(bl.abs() - LAT_MAX, min=0.0))
            kk, jj = torch.nonzero(dseg <= rk[:, None], as_tuple=True)
            n_cand += int(kk.numel())
            # 3. cell level: exact SAT of the 17 lateral offsets + not-behind
            for q0 in range(0, kk.numel(), max(1, chunk // N_LAT)):
                q1 = min(kk.numel(), q0 + max(1, chunk // N_LAT))
                kq, jq = kk[q0:q1], jj[q0:q1]
                pq = pk[kq]
                obq = ob[kq]                                                      # [q, 5]
                ra = pts[pq, jq]                                                  # [q, 17, 2] rear axles
                hq = h[pq, jq][:, None].expand(-1, N_LAT)
                ego = torch.cat([ra, hq[..., None]], -1)                          # [q, 17, 3]
                sat = box_gaps(ego, obq[:, None, :]).max(-1).values <= 0.0          # [q, 17]
                hit = sat
                if behind_m is not None or lon_max:
                    ce = ra + RA2C * eh[pq, jq][:, None, :]
                    lon = ((obq[:, None, :2] - ce) * eh[pq, jq][:, None, :]).sum(-1)
                    if behind_m is not None:
                        hit = sat & (lon >= behind_m)
                    if lon_max:
                        qs, is_ = torch.nonzero(sat, as_tuple=True)
                        if qs.numel():
                            lin = ((pq[qs] * N_ST + jq[qs]) * N_LAT + is_) * M + mt[tk[kq[qs]]]
                            lonf.scatter_reduce_(0, lin, lon[qs, is_], reduce="amax", include_self=True)
                n_pairs += int(hit.numel())
                qq, ii = torch.nonzero(hit, as_tuple=True)
                if qq.numel():
                    pf, jf, tf = pq[qq], jq[qq], tk[kq[qq]]
                    blocked[pf, jf, ii, mt[tf]] = True
                    if fine is not None:
                        fine[pf, jf, ii, tf] = True
    if lonf is not None:
        lonf = lonf.reshape(P, N_ST, N_LAT, M)
    if not details:
        return (blocked, lonf) if lon_max else blocked
    if t_avail is None:
        known = torch.ones(P, M, dtype=torch.bool, device=dev)
    else:
        ta = torch.as_tensor(t_avail, device=dev, dtype=torch.float64).reshape(-1)[idx]          # [P]
        late = (t[None, :] > ta[:, None] + T_EPS) & t_ok[None, :]                                 # [P, T]
        bad = torch.zeros(P, M, dtype=torch.bool, device=dev)
        pp, tt = torch.nonzero(late, as_tuple=True)
        bad[pp, mt[tt]] = True
        known = ~bad
    out = {"blocked": blocked, "fine": fine, "known": known, "n_pairs": n_pairs, "n_cand": n_cand}
    if lon_max:
        out["lon_max"] = lonf
    return out


def box_gaps_np(ego_poses: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """float64 numpy twin of surrogate.box_gaps (same formula and order): rear-axle poses [..., 3], boxes [..., 5] ->
    SAT gaps [..., 4]."""
    he = ego_poses[..., 2]
    ce, se = np.cos(he), np.sin(he)
    ex = ego_poses[..., 0] + RA2C * ce
    ey = ego_poses[..., 1] + RA2C * se
    hj = boxes[..., 2]
    cj, sj = np.cos(hj), np.sin(hj)
    lh, wh = 0.5 * boxes[..., 3], 0.5 * boxes[..., 4]
    dx, dy = boxes[..., 0] - ex, boxes[..., 1] - ey
    cd = np.abs(ce * cj + se * sj)
    sd = np.abs(ce * sj - se * cj)
    g_e1 = np.abs(dx * ce + dy * se) - HALF_LEN - (lh * cd + wh * sd)
    g_e2 = np.abs(-dx * se + dy * ce) - HALF_WID - (lh * sd + wh * cd)
    g_f1 = np.abs(dx * cj + dy * sj) - lh - (HALF_LEN * cd + HALF_WID * sd)
    g_f2 = np.abs(-dx * sj + dy * cj) - wh - (HALF_LEN * sd + HALF_WID * cd)
    return np.stack([g_e1, g_e2, g_f1, g_f2], -1)


def frame_np(fr: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
    """parent_frame dict -> float64 numpy arrays (input of blocked_field_np)."""
    return {k: v.detach().cpu().double().numpy() for k, v in fr.items()}


def blocked_field_np(fr: Dict[str, np.ndarray], boxes: np.ndarray, obs: np.ndarray, t_grid=None,
                     tbin_s: float = TBIN_S, n_bins: Optional[int] = None, behind_m: Optional[float] = BEHIND_M,
                     fine: bool = False, lon_max: bool = False):
    """float64 numpy twin of blocked_field for P parents of ONE scene (oracle): fr = frame_np(parent_frame(...)),
    boxes [A, T, 5], obs [A, T] bool at t_grid [T] (default fine_times()) -> B [P, 48, 17, M] bool.  fine=True adds the
    per sample field [P, 48, 17, T], lon_max=True the LON field [P, 48, 17, M] float64 (blocked_field 'lon_max'); with
    either, a tuple (B, [fine], [LON]) in that order."""
    pts, n_, h, p = fr["pts"], fr["n"], fr["h"], fr["p"]
    P = pts.shape[0]
    A, T = boxes.shape[:2]
    t = fine_times() if t_grid is None else np.asarray(t_grid, np.float64).reshape(-1)
    M = n_bins_of(tbin_s) if n_bins is None else int(n_bins)
    mt = time_bins(t, tbin_s)
    obs = np.asarray(obs, bool) & ((mt >= 0) & (mt < M))[None, :]
    bx = np.asarray(boxes, np.float64)
    eh = np.stack([np.cos(h), np.sin(h)], -1)
    c0 = p + RA2C * eh
    e = np.concatenate([c0 - LAT_MAX * n_, c0 + LAT_MAX * n_], 1)
    lo, hi = e.min(1), e.max(1)
    out = np.zeros((P, N_ST, N_LAT, M), bool)
    out_f = np.zeros((P, N_ST, N_LAT, T), bool) if fine else None
    out_l = np.full((P, N_ST, N_LAT, M), -np.inf) if lon_max else None
    r = R_EGO + 0.5 * np.hypot(bx[..., 3], bx[..., 4]) + CULL_EPS                   # [A, T]
    for pp in range(P):
        inb = ((bx[..., 0] >= lo[pp, 0] - r) & (bx[..., 0] <= hi[pp, 0] + r)
               & (bx[..., 1] >= lo[pp, 1] - r) & (bx[..., 1] <= hi[pp, 1] + r) & obs)
        ai, ti = np.nonzero(inb)
        if ai.size == 0:
            continue
        ob = bx[ai, ti]                                                           # [k, 5]
        rel = ob[:, None, :2] - c0[pp][None]                                      # [k, 48, 2]
        bl = (rel * n_[pp][None]).sum(-1)
        perp = rel - bl[..., None] * n_[pp][None]
        dseg = np.hypot(np.linalg.norm(perp, axis=-1), np.maximum(np.abs(bl) - LAT_MAX, 0.0))
        kk, jj = np.nonzero(dseg <= r[ai, ti][:, None])
        if kk.size == 0:
            continue
        obq = ob[kk]
        ra = pts[pp, jj]                                                          # [q, 17, 2]
        ego = np.concatenate([ra, np.broadcast_to(h[pp, jj][:, None, None], ra.shape[:2] + (1,))], -1)
        sat = box_gaps_np(ego, obq[:, None, :]).max(-1) <= 0.0
        hit = sat
        if behind_m is not None or lon_max:
            ce = ra + RA2C * eh[pp, jj][:, None, :]
            lon = ((obq[:, None, :2] - ce) * eh[pp, jj][:, None, :]).sum(-1)
            if behind_m is not None:
                hit = sat & (lon >= behind_m)
            if lon_max:
                qs, is_ = np.nonzero(sat)
                np.maximum.at(out_l, (np.full(qs.shape, pp), jj[qs], is_, mt[ti[kk[qs]]]), lon[qs, is_])
        qq, ii = np.nonzero(hit)
        out[pp, jj[qq], ii, mt[ti[kk[qq]]]] = True
        if fine:
            out_f[pp, jj[qq], ii, ti[kk[qq]]] = True
    if not (fine or lon_max):
        return out
    return (out,) + ((out_f,) if fine else ()) + ((out_l,) if lon_max else ())


# ----------------------------------------------------------------------------------------------- field reads
def _field_kind(field: Tensor) -> str:
    if field.dim() == 3 and tuple(field.shape[1:]) == (N_ST, N_LAT):
        return "grid"
    if field.dim() == 3 and tuple(field.shape[1:]) == (320, 256):
        return "sdf"
    raise ValueError(f"field must be D [P, {N_ST}, {N_LAT}] or sdf [S, 320, 256], got {tuple(field.shape)}")


def read_field(field: Tensor, fr: Dict[str, Tensor], xy: Tensor, sdf_index: Optional[Tensor] = None):
    """Field values at N-frame points xy [B, Q, 2] of rows b (fr = the rows' parent frames [B, ...]):
    'grid' D [B, 48, 17] (Frenet on the row's station frame, arc length clamped to [s_0, s_47], bilinear, |d| > 4.8 m
    unreadable) | 'sdf' [S, 320, 256] (sample_sdf, sdf_index [B] -> field row).  -> (value [B, Q], valid [B, Q])."""
    kind = _field_kind(field)
    if kind == "sdf":
        v, ok = sample_sdf(field, xy, sdf_index)
        return v.to(xy.dtype), ok
    B, Q = xy.shape[:2]
    fp = frenet(xy, fr)
    s_st = fr["s"]
    s = torch.minimum(torch.maximum(fp["s"], s_st[:, :1]), s_st[:, -1:])
    v, ok = grid_read(field.to(xy.dtype)[..., None], s_st.to(xy.dtype), s, fp["d"])
    return v[..., 0], ok & fp["ok"]


@torch.no_grad()
def margins(field: Tensor, parent_geom, sdf_index=None, heading: str = "parent", step: float = SCAN_STEP,
            lat_max: float = LAT_MAX, proj_step: float = PROJ_STEP) -> Dict[str, Tensor]:
    """Drivable tube of the P parents (module docstring): field D [P, 48, 17] ('grid') or sdf [S, 320, 256] ('sdf',
    sdf_index [P] -> field row, default as sample_sdf).  -> dict m_left, m_right [P, 48] (m), valid [P, 48] (centred box
    readable), m_capped_left / m_capped_right [P, 48] (no crossing within lat_max), m_unread_left / m_unread_right
    [P, 48] (the scan stopped at an UNREADABLE sample -- 'grid': a corner beyond |d| = 4.8 m, i.e. the edge of the grid,
    about 3.65 m of rear-axle shift on a straight parent; 'sdf': outside the E grid -- not at f <= 0: the margin is a
    lower bound, not a curb; includes ~valid), f0 [P, 48] (min corner field of the centred box), L [P] (m), L_exit [P]
    (a station's centred box exits / is unreadable), L_station [P] (-1 none)."""
    fr = parent_frame(parent_geom, heading)
    p, n_, h = fr["p"], fr["n"], fr["h"]
    P = p.shape[0]
    dt, dev = p.dtype, p.device
    K = int(round(float(lat_max) / float(step)))
    dl = torch.arange(-K, K + 1, device=dev, dtype=dt) * float(step)               # [2K + 1]
    ra = p[:, :, None, :] + dl[None, None, :, None] * n_[:, :, None, :]           # [P, 48, D, 2]
    pose = torch.cat([ra, h[:, :, None, None].expand(P, N_ST, dl.numel(), 1)], -1)
    cor = ego_corners(pose)                                                        # [P, 48, D, 4, 2]
    kind = _field_kind(field)
    if kind == "grid" and field.shape[0] != P:
        raise ValueError(f"D has {field.shape[0]} rows for {P} parents")
    if kind == "sdf":
        si = _index(sdf_index, P, field.shape[0], dev)
        v, o = sample_sdf(field, cor.reshape(P, -1, 2), si)
        v = v.to(dt).reshape(P, N_ST, dl.numel(), 4)
        o = o.reshape(P, N_ST, dl.numel(), 4)
    else:
        # corners move linearly in xy with delta: Frenet (s, d) on a coarse delta grid (proj_step), linear in between
        Kc = int(round(float(lat_max) / float(proj_step)))
        dlc = torch.arange(-Kc, Kc + 1, device=dev, dtype=dt) * float(proj_step)
        rac = p[:, :, None, :] + dlc[None, None, :, None] * n_[:, :, None, :]
        corc = ego_corners(torch.cat([rac, h[:, :, None, None].expand(P, N_ST, dlc.numel(), 1)], -1))
        sc, dc, oc = [], [], []
        for p0 in range(0, P, 8):
            p1 = min(P, p0 + 8)
            fp = frenet(corc[p0:p1].reshape(p1 - p0, -1, 2), {k: v[p0:p1] for k, v in fr.items()})
            sc.append(fp["s"]), dc.append(fp["d"]), oc.append(fp["ok"])
        shp = (P, N_ST, dlc.numel(), 4)
        sc, dc, oc = (torch.cat(x, 0).reshape(shp) for x in (sc, dc, oc))
        u = (dl + float(lat_max)) / float(proj_step)
        i0 = torch.floor(u + 1e-9).long().clamp(0, 2 * Kc - 1)
        w = (u - i0.to(dt)).clamp(0.0, 1.0)[None, None, :, None]
        lerp = lambda x: x[:, :, i0] * (1 - w) + x[:, :, i0 + 1] * w
        s_st = fr["s"]
        sq = torch.minimum(torch.maximum(lerp(sc).reshape(P, -1), s_st[:, :1]), s_st[:, -1:])
        v, o = grid_read(field.to(dt)[..., None], s_st, sq, lerp(dc).reshape(P, -1))
        v = v[..., 0].reshape(P, N_ST, dl.numel(), 4)
        o = o.reshape(P, N_ST, dl.numel(), 4) & oc[:, :, i0] & oc[:, :, i0 + 1]
    ninf = torch.full((), -math.inf, dtype=dt, device=dev)
    f = torch.where(o.all(-1), v.min(-1).values, ninf)                            # [P, 48, D]; -inf unreadable
    f0 = f[..., K]
    valid = torch.isfinite(f0)

    def side(fs: Tensor):
        """fs [P, 48, K + 1] from delta = 0 outward -> (margin, capped, unread)."""
        pos = fs > 0
        n_lead = torch.cumprod(pos.to(torch.int64), -1).sum(-1)                   # leading positives
        capped = n_lead == K + 1
        kl = (n_lead - 1).clamp(min=0)
        kf = n_lead.clamp(max=K)
        fl = torch.gather(fs, -1, kl[..., None])[..., 0]
        ff = torch.gather(fs, -1, kf[..., None])[..., 0]
        frac = torch.where(torch.isfinite(ff), fl / torch.clamp(fl - ff, min=1e-12), torch.zeros_like(fl))
        m = kl.to(dt) * float(step) + float(step) * frac.clamp(0.0, 1.0)
        m = torch.where(capped, torch.full_like(m, float(lat_max)), m)
        m = torch.where(n_lead == 0, torch.where(valid, f0, torch.zeros_like(f0)), m)
        return m, capped, ~torch.isfinite(ff) & ~capped                           # unread: the scan hit the field edge

    m_left, cap_l, unr_l = side(f[..., K:])
    m_right, cap_r, unr_r = side(torch.flip(f[..., :K + 1], [-1]))
    # L: first station whose centred box exits (f0 <= 0) or is unreadable
    ex = ~(f0 > 0)                                                                # [P, 48]
    has = ex.any(-1)
    j = torch.where(has, ex.to(torch.int64).argmax(-1), torch.full((P,), -1, device=dev, dtype=torch.int64))
    jc = j.clamp(min=1)
    s_st, ds = fr["s"], fr["ds"]
    f_prev = torch.gather(f0, 1, (jc - 1)[:, None])[:, 0]
    f_j = torch.gather(f0, 1, j.clamp(min=0)[:, None])[:, 0]
    s_prev = torch.gather(s_st, 1, (jc - 1)[:, None])[:, 0]
    frac = torch.where(torch.isfinite(f_j), f_prev / torch.clamp(f_prev - f_j, min=1e-12), torch.zeros_like(f_prev))
    L = s_prev + ds * frac.clamp(0.0, 1.0)
    L = torch.where(j == 0, torch.zeros_like(L), L)
    L = torch.where(has, L, N_ST * ds)
    return {"m_left": m_left, "m_right": m_right, "valid": valid, "m_capped_left": cap_l, "m_capped_right": cap_r,
            "m_unread_left": unr_l, "m_unread_right": unr_r,
            "f0": torch.where(valid, f0, torch.zeros_like(f0)), "L": L, "L_exit": has, "L_station": j}


# ----------------------------------------------------------------------------------------------- speed limit
def fit_track_table(yaw: np.ndarray, err: np.ndarray, edges: Sequence[float] = TRACK_BIN_EDGES,
                    q: float = 50.0, min_n: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear tracking-error table (module docstring): per bin [edges[k], edges[k + 1]) with >= min_n rows,
    knot = (median yaw, q-th percentile error); errors made non-decreasing (cummax).  -> (yaw knots, err knots, n)."""
    yaw = np.asarray(yaw, np.float64)
    err = np.asarray(err, np.float64)
    ok = np.isfinite(yaw) & np.isfinite(err)
    yaw, err = yaw[ok], err[ok]
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (yaw >= lo) & (yaw < hi)
        if int(m.sum()) < int(min_n):
            continue
        xs.append(float(np.median(yaw[m])))
        ys.append(float(np.percentile(err[m], q)))
        ns.append(int(m.sum()))
    return np.asarray(xs), np.maximum.accumulate(np.asarray(ys)), np.asarray(ns)


def track_error(yaw: Tensor, fit=TRACK_FIT) -> Tensor:
    """expected tracking path error e(yaw rate) [m] (table held below the first knot, last slope beyond the last)."""
    xk = torch.as_tensor(fit[0], dtype=yaw.dtype, device=yaw.device)
    yk = torch.as_tensor(fit[1], dtype=yaw.dtype, device=yaw.device)
    w = yaw.abs()
    i = torch.searchsorted(xk, w.contiguous().reshape(-1), right=True).reshape(w.shape).clamp(1, len(xk) - 1)
    x0, x1, y0, y1 = xk[i - 1], xk[i], yk[i - 1], yk[i]
    e = y0 + (w - x0) * (y1 - y0) / (x1 - x0)
    return torch.where(w <= xk[0], yk[0].expand_as(w), e)


def _track_inverse(y: Tensor, fit=TRACK_FIT) -> Tensor:
    """largest yaw rate w with e(w) <= y for y >= e(0) = yk[0] (the table is strictly increasing from its first knot,
    extended with the last slope); -1 where y < yk[0] (no yaw rate qualifies)."""
    xk = torch.as_tensor(fit[0], dtype=y.dtype, device=y.device)
    yk = torch.as_tensor(fit[1], dtype=y.dtype, device=y.device)
    K = xk.numel()
    k = (torch.searchsorted(yk, y.contiguous().reshape(-1), right=True).reshape(y.shape) - 1).clamp(0, K - 2)
    x0, x1, y0, y1 = xk[k], xk[k + 1], yk[k], yk[k + 1]
    w = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return torch.where(y < yk[0], torch.full_like(w, -1.0), w)


def outside_margin(kappa: Tensor, m_left: Tensor, m_right: Tensor) -> Tensor:
    """margin on the outside of the turn: kappa > 0 (left turn) -> m_right, kappa < 0 -> m_left, 0 -> min."""
    return torch.where(kappa > 0, m_right, torch.where(kappa < 0, m_left, torch.minimum(m_left, m_right)))


def v_max(kappa: Tensor, m_out: Tensor, fit=TRACK_FIT, margin: float = TRACK_MARGIN, comfort_prior: bool = False,
          v_cap: float = V_CAP) -> Tensor:
    """largest speed [m/s] whose expected tracking error e(v |kappa|) + margin stays <= m_out (module docstring);
    kappa, m_out [P, 48] (any shape)."""
    w = _track_inverse(m_out - float(margin), fit)
    k = kappa.abs()
    v = torch.where(k > 0, w / torch.where(k > 0, k, torch.ones_like(k)), torch.full_like(w, float(v_cap)))
    v = torch.where(w < 0, torch.zeros_like(v), v)
    if comfort_prior:
        kk = torch.where(k > 0, k, torch.ones_like(k))
        vc = torch.minimum(YAW_RATE_MAX / kk, torch.sqrt(LAT_ACC_MAX / kk))
        v = torch.where(k > 0, torch.minimum(v, vc), v)
    return v.clamp(max=float(v_cap))


# ----------------------------------------------------------------------------------------------- windows
def windows(B: Tensor, tbin_s: float = TBIN_S, known: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """B [P, 48, 17, M] bool -> t_in_block, t_out_block [P, 48, 17] (s; module docstring), n_runs [P, 48, 17] int64.
    known [P, M] optional: bins after the last known bin are ignored and a blocked last known bin gives +inf; a parent
    with NO known bin gets t_in_block = 0, t_out_block = +inf on every cell (unknown = blocked over the horizon; n_runs
    0)."""
    Bb = B.bool()
    P, M = Bb.shape[0], Bb.shape[-1]
    dev = Bb.device
    m = torch.arange(M, device=dev)
    if known is not None:
        kn = known.bool().to(dev)
        last_known = torch.where(kn.any(-1), M - 1 - torch.flip(kn, [-1]).to(torch.int64).argmax(-1),
                                 torch.full((P,), -1, device=dev, dtype=torch.int64))      # [P]
        Bb = Bb & (m[None] <= last_known[:, None])[:, None, None, :]
    else:
        last_known = torch.full((P,), M - 1, device=dev, dtype=torch.int64)
    anyb = Bb.any(-1)
    first = Bb.to(torch.int64).argmax(-1)
    last = M - 1 - torch.flip(Bb, [-1]).to(torch.int64).argmax(-1)
    inf = torch.full(anyb.shape, math.inf, dtype=torch.float64, device=dev)
    t_in = torch.where(anyb, first.to(torch.float64) * float(tbin_s), inf)
    t_out = torch.where(anyb, (last + 1).to(torch.float64) * float(tbin_s), torch.zeros_like(inf))
    t_out = torch.where(anyb & (last >= last_known[:, None, None]), inf, t_out)
    # a parent with no known bin (GT horizon < the first bin) is unknown, not free: blocked from t0, never free
    unk = (last_known < 0)[:, None, None]
    t_in = torch.where(unk, torch.zeros_like(t_in), t_in)
    t_out = torch.where(unk, inf, t_out)
    rise = Bb[..., 1:] & ~Bb[..., :-1]
    n_runs = rise.to(torch.int64).sum(-1) + Bb[..., 0].to(torch.int64)
    return {"t_in_block": t_in, "t_out_block": t_out, "n_runs": n_runs}


# ----------------------------------------------------------------------------------------------- executed slack
def _cells(fr: Dict[str, Tensor], xy: Tensor, read: str = "nearest"):
    """points xy [B, Q, 2] -> station j, lateral i [B, Q] (long; the nearest cell) and in-grid mask (module docstring);
    read 'any4' also returns the 4 surrounding grid points jn, in [B, Q, 4] (clamped to the grid)."""
    fp = frenet(xy, fr)
    ds = fr["ds"].to(xy.dtype)[:, None]
    s, d = fp["s"], fp["d"]
    j = torch.floor(s / ds).clamp(min=0)
    i = torch.round((d + LAT_MAX) / LAT_STEP)
    ok = fp["ok"] & (s >= -0.5 * ds) & (s < N_ST * ds) & (i >= 0) & (i <= N_LAT - 1)
    j = torch.where(ok, j, torch.zeros_like(j)).long().clamp(max=N_ST - 1)
    i = torch.where(ok, i, torch.zeros_like(i)).long().clamp(0, N_LAT - 1)
    if read == "nearest":
        return j, i, ok
    if read != "any4":
        raise ValueError(f"read {read!r} not in ('nearest', 'any4')")
    us = torch.where(ok, s / ds - 0.5, torch.zeros_like(s))
    ud = torch.where(ok, (d + LAT_MAX) / LAT_STEP, torch.zeros_like(d))
    j0, i0 = torch.floor(us).long(), torch.floor(ud).long()
    jn = torch.stack([j0, j0 + 1, j0, j0 + 1], -1).clamp(0, N_ST - 1)
    in_ = torch.stack([i0, i0, i0 + 1, i0 + 1], -1).clamp(0, N_LAT - 1)
    return j, i, ok, jn, in_


def _time_slack(prof: Tensor, t: Tensor, tbin_s: float, cap: float) -> Tensor:
    """prof [..., M] bool (a cell's blocked bins), t [...] (s) -> signed slack [...] (module docstring).  The bin of t
    is floor(t / tbin_s + T_EPS) in t's dtype: pass float64 times (executed_slack does) -- in float32, t / 0.1 rounds
    below the integer for t = 3.7 / 4.2 / 4.7 s (one bin early).  A read time outside [0, M tbin_s) is never 'in' a
    run; its free slack sees every bin (the run searches use the unclamped bin).  Free slack >= 0."""
    M = prof.shape[-1]
    dt = t.dtype
    m = torch.floor(t / float(tbin_s) + T_EPS).long()
    inr = (m >= 0) & (m < M)
    mc = m.clamp(0, M - 1)
    ar = torch.arange(M, device=prof.device)
    blk = torch.gather(prof, -1, mc[..., None])[..., 0] & inr
    capt = torch.full_like(t, float(cap))
    # end of the run containing m (only used where m is in range, so mc = m): first free bin > m (M if none -> +inf)
    free_after = (~prof) & (ar > mc[..., None])
    has_end = free_after.any(-1)
    b_end = torch.where(has_end, free_after.to(torch.int64).argmax(-1), torch.full_like(mc, M))
    s_in = torch.where(has_end, t - b_end.to(dt) * float(tbin_s), -capt)
    # free: nearest blocked bin before (its end) / after (its start), searched from the UNCLAMPED bin m
    before = prof & (ar < m[..., None])
    after = prof & (ar > m[..., None])
    hb, ha = before.any(-1), after.any(-1)
    last_b = M - 1 - torch.flip(before, [-1]).to(torch.int64).argmax(-1)
    first_a = after.to(torch.int64).argmax(-1)
    d_b = torch.where(hb, t - (last_b + 1).to(dt) * float(tbin_s), capt)
    d_a = torch.where(ha, first_a.to(dt) * float(tbin_s) - t, capt)
    # a free read is never a hit: t joins its bin with T_EPS, so t can lie up to T_EPS tbin_s BEFORE the bin's start
    # (0.1 s bins: 24 of the 205 NC / TTC read times, e.g. 0.3 = 2.9999999999999996 x 0.1) and d_b = -1e-16
    s_free = torch.minimum(torch.minimum(d_b, d_a), capt).clamp(min=0.0)
    return torch.where(blk, s_in, s_free).clamp(-float(cap), float(cap))


@torch.no_grad()
def executed_slack(states: Tensor, parent_geom, B_field: Optional[Tensor], field: Optional[Tensor] = None, index=None,
                   sdf_index=None, heading: str = "parent", tbin_s: float = TBIN_S,
                   deltas: Sequence[float] = TTC_DELTAS, stop_nc: Optional[float] = STOP_NC,
                   stop_ttc: Optional[float] = STOP_TTC, slack_cap: float = SLACK_CAP,
                   dt_state: float = DT_FINE, per_state: bool = False, read: str = "nearest",
                   lon_field: Optional[Tensor] = None, behind_m: Optional[float] = BEHIND_M) -> Dict[str, Tensor]:
    """Slack of B candidates on their executed states (module docstring 'executed_slack').

    states [B, 41, >= 4] (x, y, heading, v, ...; N frame, t = 0.1 n); parent_geom of the P parents (or a frame dict),
    B_field [P, 48, 17, M] bool (blocked_field; a FLOAT field is read as probabilities sigmoid(B): blocked where
    > 0.5 -- threshold logits yourself at 0; None: LON >= behind_m from lon_field), field: D [P, 48, 17] |
    sdf [S, 320, 256] | None (no DAC), index [B] candidate -> parent row (default arange if P == B), sdf_index [B] ->
    sdf row.
    lon_field [P, 48, 17, M] (opt-in, blocked_field(lon_max=True)): TTC reads the not-behind test w.r.t. the CURRENT
    ego (the official is_agent_ahead / is_agent_behind reference, the state's rear axle) instead of the projected
    cell's box: the profile of state n at projection delta is LON >= behind_m - |v_n| delta (delta 0 = B); behind_m
    must be the field's (None: LON > -inf, every overlap).  None (default): TTC reads B at the projected cells.  NC
    always reads B.
    -> slack_nc, slack_ttc [B] (s), slack_dac [B] (m; +10 without field / no readable corner), hit_nc / hit_ttc /
    hit_dac [B] bool, crit_station [B, 3] long, crit_time [B, 3] (s) (columns nc, ttc, dac; -1 / -1.0 none; the DAC
    station is -1 for a state off the parent grid), n_oog [B] (states off the grid);
    per_state: also state_nc [B, 41], state_ttc [B, 41, K] (per-state slack, s), state_dac [B, 41] (min readable
    corner, m; 10 if none).
    read: 'nearest' (default) cell of each state | 'any4' (conservative: blocked if any of the 4 surrounding grid
    points is blocked).
    Speeds are |v| (official speeds = hypot(vx, vy) >= 0: stopped-ego skips, and a reversing state is projected FORWARD
    along its heading); read times / bins are float64 whatever the dtype (slack values in the geometry dtype)."""
    fr_p = parent_frame(parent_geom, heading)
    Bn, NT = states.shape[:2]
    P = fr_p["p"].shape[0]
    dev = states.device
    dt = fr_p["p"].dtype
    idx = _index(index, Bn, P, dev)
    fr = gather_frame(fr_p, idx)
    if B_field is None:
        if lon_field is None:
            raise ValueError("executed_slack needs B_field or lon_field")
        Bp = (lon_field > -math.inf) if behind_m is None else (lon_field >= behind_m)
    else:
        Bp = (B_field > 0.5) if B_field.is_floating_point() else B_field.bool()   # [P, 48, 17, M] (parent rows)
    if tuple(Bp.shape[:3]) != (P, N_ST, N_LAT):
        raise ValueError(f"B_field must be [{P}, {N_ST}, {N_LAT}, M], got {tuple(Bp.shape)}")
    if lon_field is not None and lon_field.shape != Bp.shape:
        raise ValueError(f"lon_field {tuple(lon_field.shape)} != B_field {tuple(Bp.shape)}")
    st = states.to(dt)
    xy, hd = st[..., :2], st[..., 2]
    spd = st[..., 3].abs()                                                         # [B, 41] official speed >= 0
    tn64 = torch.arange(NT, device=dev, dtype=torch.float64) * float(dt_state)     # [41] read times (float64 bins)
    tn = tn64.to(dt)
    bidx = torch.arange(Bn, device=dev)
    out: Dict[str, Tensor] = {}
    crit_s = torch.full((Bn, 3), -1, dtype=torch.long, device=dev)
    crit_t = torch.full((Bn, 3), -1.0, dtype=dt, device=dev)

    def key_slack(pts: Tensor, tt: Tensor, use: Tensor, thr: Optional[Tensor] = None, lon: bool = False):
        """pts [B, Q, 2], tt [Q] float64 times, use [B, Q]; lon: read lon_field >= thr [B, Q] (thr None: > -inf)
        instead of B -> (slack [B, Q], station [B, Q], ok [B, Q])."""
        if read == "nearest":
            j, i, ok = _cells(fr, pts)
            jr, ir = j[..., None], i[..., None]                                    # [B, Q, 1]
        else:
            j, i, ok, jr, ir = _cells(fr, pts, read)                               # [B, Q, 4] surrounding points
        pr_ = idx[:, None, None]
        if not lon:
            prof = Bp[pr_, jr, ir]                                                 # [B, Q, c, M]
        elif thr is None:
            prof = lon_field[pr_, jr, ir] > -math.inf
        else:
            prof = lon_field[pr_, jr, ir] >= thr[:, :, None, None]
        prof = prof.any(-2)                                                        # OR over the read points
        sl = _time_slack(prof, tt.expand(pts.shape[:2]), tbin_s, slack_cap).to(dt)
        sl = torch.where(ok & use, sl, torch.full_like(sl, float(slack_cap)))
        return sl, j, ok

    # NC
    use_nc = torch.ones_like(spd, dtype=torch.bool) if stop_nc is None else (spd > float(stop_nc))
    sl, j, ok = key_slack(xy, tn64, use_nc)
    out["n_oog"] = (~ok).sum(-1)
    s_nc, n_nc = sl.min(-1)
    if per_state:
        out["state_nc"] = sl
    out["slack_nc"] = s_nc
    out["hit_nc"] = s_nc < 0
    has = s_nc < float(slack_cap)
    crit_s[:, I_NC] = torch.where(has, j[bidx, n_nc], crit_s[:, I_NC])
    crit_t[:, I_NC] = torch.where(has, tn[n_nc], crit_t[:, I_NC])
    # TTC (delta 0 included, official): projection |v_n| delta along the executed heading, read at t_n + delta
    dl64 = torch.as_tensor([float(x) for x in deltas], device=dev, dtype=torch.float64)   # [K]
    dl = dl64.to(dt)
    K = dl.numel()
    eh = torch.stack([torch.cos(hd), torch.sin(hd)], -1)                           # [B, 41, 2]
    pr = xy[:, :, None, :] + (spd[..., None, None] * dl[None, None, :, None]) * eh[:, :, None, :]   # [B, 41, K, 2]
    tt64 = (tn64[:, None] + dl64[None, :]).reshape(-1)                              # [41 K]
    use_t = torch.ones_like(spd, dtype=torch.bool) if stop_ttc is None else (spd >= float(stop_ttc))
    thr = None
    if lon_field is not None and behind_m is not None:
        # not-behind w.r.t. the current ego centre: (c_obj - c_cur) . e ~ LON + |v_n| delta >= behind_m
        thr = (float(behind_m) - spd[:, :, None] * dl[None, None, :]).reshape(Bn, -1)
    sl_t, j_t, _ = key_slack(pr.reshape(Bn, NT * K, 2), tt64, use_t[:, :, None].expand(Bn, NT, K).reshape(Bn, -1),
                             thr, lon=lon_field is not None)
    s_tt, q_tt = sl_t.min(-1)
    if per_state:
        out["state_ttc"] = sl_t.reshape(Bn, NT, K)
    out["slack_ttc"] = s_tt
    out["hit_ttc"] = s_tt < 0
    has = s_tt < float(slack_cap)
    crit_s[:, I_TTC] = torch.where(has, j_t[bidx, q_tt], crit_s[:, I_TTC])
    crit_t[:, I_TTC] = torch.where(has, tn[q_tt // K], crit_t[:, I_TTC])
    # DAC
    if field is None:
        out["slack_dac"] = torch.full((Bn,), float(SDF_CLIP), dtype=dt, device=dev)
        out["hit_dac"] = torch.zeros(Bn, dtype=torch.bool, device=dev)
    else:
        kind = _field_kind(field)
        cor = ego_corners(st[..., :3]).reshape(Bn, NT * 4, 2)                     # [B, 41*4, 2]
        if kind == "grid":
            if field.shape[0] != P:
                raise ValueError(f"D has {field.shape[0]} rows for {P} parents")
            fv, fo = read_field(field[idx], fr, cor)
        else:
            fv, fo = read_field(field, fr, cor, _index(sdf_index, Bn, field.shape[0], dev))
        fv = torch.where(fo, fv, torch.full_like(fv, math.inf)).reshape(Bn, NT, 4).min(-1).values   # inf: unreadable
        s_d, n_d = fv.min(-1)
        has_d = torch.isfinite(s_d)                                                # some corner readable
        clip = torch.full_like(s_d, float(SDF_CLIP))
        if per_state:
            out["state_dac"] = torch.where(torch.isfinite(fv), fv, torch.full_like(fv, float(SDF_CLIP)))
        out["slack_dac"] = torch.where(has_d, s_d, clip)
        out["hit_dac"] = has_d & (s_d < 0)
        crit_s[:, I_DAC] = torch.where(has_d & ok[bidx, n_d], j[bidx, n_d], crit_s[:, I_DAC])
        crit_t[:, I_DAC] = torch.where(has_d, tn[n_d], crit_t[:, I_DAC])
    out["crit_station"] = crit_s
    out["crit_time"] = crit_t
    return out
