"""refiner v4_3 M4 readout 'range' (V4Cfg.readout 'range'; docs/refiner_v4/V4_3_SPEC.md M4, config.py docstring).

Frames / units: NAVSIM N frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x; metres, seconds).
Parent grid (range_targets docstring): 48 stations s_j = (j + 0.5) ds on the parent's smooth path x 17 lateral offsets
d_i = -4.8 + 0.6 i; a cell (j, i) holds the official ego box with its REAR AXLE at the grid point, heading along the
parent path.  Time bins m = [m tbin, (m + 1) tbin), tbin = V4Cfg.ev_tbin_s, M = ev_tbins (M tbin = 5.0 s).

Scene fields (learned, candidate-free, once per UNIQUE parent: CKNet.score_set's parent dedupe rows R = T * U):
  blk [R, 48, 17, 2M] logits, channels [0, M) = B, [M, 2M) = U (+ [2M, 3M) = Bp with rg_stopped_aware, below):
      B[j, i, m]  the box at cell (j, i) overlaps a GT object at a time of bin m that is NOT behind the box (object
                  centre >= BEHIND_M = -1.3 m along the cell heading from the box centre) -- target
                  range_targets.blocked_field (the at-fault proxy; NC and TTC delta 0 read it);
      U[j, i, m]  the same overlap WITHOUT the not-behind filter -- target blocked_field(behind_m=None) (= LON > -inf of
                  lon_max=True).
      Why a second channel: the official TTC tests not-behind from the CURRENT rear axle (pdm _calculate_ttc), while B
      fixes it at the projected cell's box, |v| delta ahead: an object between the current ego and the projected box is
      'behind' for B and 'ahead' officially (range_targets 'executed_slack', 46 % of the field's all-16 TTC misses on
      exact boxes).  Exact = LON >= BEHIND_M - |v| delta; a not-behind test at a shifted threshold needs the
      along-offset LON, which U bounds: for |v| delta >= ~3.7 m (box half length + an object half length - 1.3 m)
      every overlap is ahead (exact = U), below it B is the closer of the two.  Rule: TTC delta > 0 reads U where
      |v| delta >= U_SWITCH_M = 2.0 m, else B.  GT agreement (channel_study, 2026-10-16 KST, 6,000 F1 navtrain_val
      tokens, v2 r34 top-16 parents, corridor v2, heading 'parent', 0.25 s bins, nearest cell, official stopped-ego
      skips, F1 executed states, official TTC labels; TTC fail / pass recall):
                                       top-1 (206 fails)      all 16 (6,174 fails)   agreement with exact (all 16)
        B only (field convention)      0.7718 / 0.9858        0.7945 / 0.9709        0.98986
        exact LON (lon_field)          0.8204 / 0.9841        0.8711 / 0.9653        1
        B + U, U for every delta > 0   0.8204 / 0.9829        0.8826 / 0.9601        0.99436
        B + U, U if |v| delta >= 2 m   0.8204 / 0.9841        0.8709 / 0.9655        0.99756   <- used
        B + U, switch 1 m / 3 m        all-16 0.8785 / 0.9624, 0.8596 / 0.9672       0.99628 / 0.99708
        3 channels B, LON>=-2.3, U     all-16 0.8733 / 0.9650 (floor read)            0.99813
        5 channels (1 m steps, round)  all-16 0.8719 / 0.9653                         0.99942
      (top-1 identical to exact for every two-channel switch >= 1 m; cell rates B 12.3 %, U 16.9 %).  The 2 m switch
      is the exact rule up to 0.24 % of the candidates at 2 x M outputs; more LON bins gain <= 0.19 %.
  With V4Cfg.rg_stopped_aware (Phase 0b R2 (1)) blk has a THIRD group, channels [2M, 3M) = Bp:
      B[j, i, m]  is then the stopped-aware B (target range_targets.blocked_field(stopped_aware=True): stopped /
                  static objects block without the not-behind filter) -- read by NC only;
      Bp[j, i, m] the plain not-behind B (target LON >= BEHIND_M of the same lon_max call = the B above without R2):
                  every TTC read that would read B (delta 0, and |v| delta < U_SWITCH_M) reads Bp.
      Why: R2 changes NC only (the official TTC rule has no stopped-track exception, tools/ck/v4/oracle/heldout_common
      'U / LON unchanged'), so a TTC read of the stopped-aware B learns an exception the scorer does not have (review
      of cprep, 1,500 F1 tokens on GT fields: TTC pass recall 0.8422 vs 0.8525 for the R2 GT tube, joint NC | TTC
      disagreement with the tube 0.90 % vs 0.59 % with the plain B; NC identical).  With Bp the TTC reads are those of
      the readout without R2 (bitwise on the same fields, tests/test_v4_r2.py) and NC is the R2 GT reduction.
  margin [R, 48, 17] (m): signed distance of the grid point to the official DAC boundary (+ inside; target the
      official SDF at the grid points, as evidence 'D').
  Heads: per-cell MLPs on [Hc (128 corridor conv channels), x'_j (station token)] (the evidence readout's inputs),
  evaluated as 1x1 convolutions on Hc plus the per-station token term (no [R, 48, 17, 128 + d] concatenation; the same
  function as nn.Sequential(Linear, GELU, Linear) on the concatenation up to float summation order).

Candidate states (no parameters, no gradient; candidate_states):
  exec_model 'lqr': the official LQR tracker + bicycle (ck/v4/lqr.simulate, float32) from ego (v0, a0) =
      (status[..., 4], status[..., 6]) -> Q = 41 states (x, y, heading, v, a, steer) at t = 0.1 n;
  exec_model 'plan': the origin + the 8 planned poses at t = 0.5 k (Q = 9), v_k = |p_k - p_{k-1}| / 0.5 (v_0 = vx);
  exec_model 'ref': the tracker's 41-state reference, no tracker (lqr.reference_rollout; Q = 41, t = 0.1 n).
  Points projected on the parent's station frame (temporal.raster.frenet, exact for any point set) -> (s, d); an NC /
  TTC point is on the grid when s in [-ds / 2, 48 ds) and d in [-5.1, 5.1) (range_targets._cells' nearest-cell
  domain), a DAC corner when |d| <= 4.8 m at any arc length (range_targets.read_field 'grid').  B / U are read at the
  NEAREST cell (station floor(s / ds), lateral round((d + 4.8) / 0.6): the GT reduction's own read, so a field equal to
  its target reproduces the oracle; a bilinear read bleeds a blocked region half a cell into its free neighbours), D
  bilinearly in (s, d) with s clamped to [s_0, s_47] (as read_field; D is a continuous distance); the time bin is
  floor(t / tbin + 1e-6) (float64).
  V4Cfg.rg_read 'any4' (Phase 0b R2 (2), opt-in): B / U are read at the 2 x 2 grid points around each point
  (range_targets._cells 'any4'; any4_cells): the thresholded slacks OR the 4 cells' profiles (hard_slack; = range_targets
  .executed_slack(read='any4') on the same fields) and the soft evidence below runs its soft max over the points x
  their distinct cells (soft_read: a soft OR of the 4 cells inside the soft max over the states; set_prior then
  subtracts tau ln(4 Q)); D stays bilinear.
Evidence (all differentiable in the fields; spec M4):
  NC   z_nc  = smax over the states with |v| > 0.05 m/s (official stopped-ego skip) of B(s_n, d_n, bin(t_n))
  TTC  z_ttc = smax over states with |v| >= 5e-3 m/s and delta in (0, 0.3, 0.6, 0.9) s of B-or-U (rule above; Bp in
       place of B with rg_stopped_aware) at the rear axle projected |v_n| delta along the executed heading h_n,
       bin(t_n + delta)
  DAC  z_dac = smax over states x the 4 official corners (executed heading) of -D(corner) / eps, eps = exp(log_eps)
       (init ev_eps, >= 0.01 m)
  smax = evidence.smax (tau 0.5 logsumexp; no readable point -> -20), lp_k = log P_pass,k = logsigmoid(-z_k), score
  logit_k = -z_k (= logit(P_pass)) + residual_k(r_in, g_c) (zero-initialised MLP, ev_resid), r_in = [lp_nc, lp_ttc,
  lp_dac, slack_nc, slack_ttc, slack_dac (clamped +-10 m), overspeed] / 5.
  Slacks / critical points (the GT conventions of range_targets.executed_slack; slack_nc / slack_ttc / crit_* on the
  THRESHOLDED fields B > 0, U > 0 (logit 0 = probability 0.5), nearest cell, no gradient): slack_nc / slack_ttc [s]
  (time to the nearest blocked time of the read cell, < 0 inside a blocked run = how much later the ego would have to
  arrive; capped +-5 s), slack_dac [m] (= min_margin: min readable corner margin, the hard min of the bilinear D over
  the states x 4 corners; +10 m if no corner is readable; NOT detached: a sub-gradient flows into D at the arg-min
  corner through the residual input r_in and the haz output, as the evidence readout's min_marg -- hard_evidence's
  slack_dac, used by the edits, is the no-grad twin on the detached D), crit_station / crit_lat / crit_time [B, 3]
  (columns nc, ttc, dac; -1 / -1.0 none; TTC: the projected cell, the EGO time of the state), crit_chan [B] (channel
  group the TTC critical read used: 0 B, 1 U, 2 Bp (rg_stopped_aware)).
  vmax 'feature': overspeed = max over the moving on-grid states of relu(|v_n| - v_max(station of s_n)), v_max =
  range_targets.v_max(parent curvature, outside margin) with the margins of grid_margins on the predicted D (detached).
EP / comfort / lateral heads: as the evidence readout (stations read = readout.read_mask on the candidate's PLANNED
  schedule t_in, sibling.on_parent), lateral weights w = softmax over the read stations of -D(s_j, d_c(j)) / eps
  (detached); ep_evidence: EP input += detached [lp_nc, lp_ttc, lp_dac, slack_nc, slack_ttc, slack_dac] / 5, the
  executed 4 s progress ratio S_exec / max(P_ref_hat, S_exec, 5 m) (lqr.executed_progress; 1 without P_ref) and the
  margin-weighted candidate-position features sum_j w_j x'_j, sum_j w_j Hc(j, d_c(j)) (the lateral head's inputs).
Not modelled (as the GT field, range_targets 'What the field does NOT model'): at-fault classes beyond not-behind, the
  collided-track memory, the TTC 30 deg / lane rules, the executed heading on the grid (cells hold the parent heading).
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...refiner.corridor import LAT_MAX, N_LAT, N_ST
from ...refiner.decoder import N_FREE
from ...refiner.geometry import HALF_LEN, HALF_WID, RA2C
from ...refiner.sdf import SDF_CLIP, ego_corners
from ..constants import CK_KEYS
from ..temporal.raster import frenet
from . import lqr as LQR
from . import range_targets as RT
from .config import V4Cfg
from .evidence import SMAX_TAU, smax
from .readout import EP_MIN_DIST, HC_CH, I_C, I_DAC, I_EP, I_NC, I_TTC, LAT_STEP, _mlp, read_mask

Tensor = torch.Tensor
TTC_DELTAS = RT.TTC_DELTAS                       # (0, 0.3, 0.6, 0.9) s, official future_time_idcs x 0.1 s
U_SWITCH_M = 2.0                                 # [m] TTC reads U where |v| delta >= this (module docstring)
CH_B, CH_U, CH_BP = 0, 1, 2                      # blk channel groups [g M, (g + 1) M): B | U | Bp (rg_stopped_aware)
STOP_NC, STOP_TTC = RT.STOP_NC, RT.STOP_TTC      # [m/s] official stopped-ego skips (NC 0.05, TTC 5e-3)
SLACK_CAP = RT.SLACK_CAP                         # [s]
T_EPS = RT.T_EPS
N_RES = 7                                        # residual inputs besides g_c
N_EP_EV = 7                                      # ep_evidence scalars (+ d + 128 margin-weighted features)
MARGIN_STEP = 0.1                                # [m] grid_margins lateral scan step
A_FRONT, A_REAR = RA2C + HALF_LEN, RA2C - HALF_LEN    # box front / rear edge ahead of the rear axle [m]
I_CNC, I_CTTC, I_CDAC = RT.I_NC, RT.I_TTC, RT.I_DAC   # crit_* columns (0 nc, 1 ttc, 2 dac)


# ----------------------------------------------------------------------------------------------- states
_MEMO: Optional[list] = None                     # open shared_states() scope: [(key, states)], at most _MEMO_N entries
_MEMO_N = 4


@contextlib.contextmanager
def shared_states():
    """Within the scope, candidate_states returns the states it already computed for EQUAL inputs (same exec model,
    torch.equal cand / vx / ax after its finite / nan cleaning) instead of rolling them out again -- TeacherPair2.run:
    the DET and MAP teachers score the same candidates from the same ego, one LQR rollout instead of two (2026-10-20
    profile: 4,741 GPU launches per rollout).  Exact: the cached tensor is returned as a clone."""
    global _MEMO
    prev, _MEMO = _MEMO, []
    try:
        yield
    finally:
        _MEMO = prev


def _memo_get(key):
    if _MEMO is None:
        return None
    m, c, v0, a0 = key
    for (m2, c2, v2, a2), st in _MEMO:
        if (m2 == m and c2.shape == c.shape and c2.dtype == c.dtype and c2.device == c.device and v2.shape == v0.shape
                and torch.equal(c2, c) and torch.equal(v2, v0) and torch.equal(a2, a0)):
            return st.clone()
    return None


def _memo_put(key, st) -> None:
    if _MEMO is not None:                        # a private copy: no consumer's in-place write reaches the next one
        _MEMO.append((key, st.clone()))
        del _MEMO[:-_MEMO_N]


@torch.no_grad()
def candidate_states(cand: Tensor, vx: Tensor, ax: Tensor, model: str = "lqr") -> Tuple[Tensor, Tensor]:
    """cand [B, 8, 3] (N frame, t = 0.5 k s), ego vx / ax [B] (status[..., 4] / status[..., 6]: the rear-axle
    longitudinal velocity / acceleration, lqr docstring) -> (states [B, Q, 6] f32 (x, y, heading, v, a, steer), t [Q]
    float64 s).  'lqr': lqr.simulate (Q 41, t = 0.1 n); 'ref': lqr.reference_rollout (Q 41, the tracker's reference,
    no tracker); 'plan': origin + the 8 poses (Q 9, t = 0.5 k; v_k = |p_k - p_{k-1}| / 0.5, v_0 = vx, a = steer = 0).
    Non-finite rows are simulated as zeros (as the scorer callers do).  'lqr' / 'ref' reuse equal inputs' states inside
    a shared_states() scope."""
    B = cand.shape[0]
    c = cand.float()
    fin = torch.isfinite(c).flatten(1).all(1)
    c = torch.where(fin[:, None, None], c, torch.zeros_like(c))
    v0 = torch.nan_to_num(vx.float().reshape(B), nan=0.0)
    a0 = torch.nan_to_num(ax.float().reshape(B), nan=0.0)
    if model in LQR.EXECUTORS:
        key = (model, c, v0, a0)
        st = _memo_get(key)
        if st is None:
            st = LQR.executor(model)(c, torch.stack([v0, a0], -1), dtype=torch.float32)
            _memo_put(key, st)
        return st, torch.arange(LQR.NT, dtype=torch.float64, device=c.device) * LQR.DT
    if model != "plan":
        raise ValueError(f"exec model {model!r} not in ('plan', 'lqr', 'ref')")
    pose = torch.cat([torch.zeros(B, 1, 3, dtype=c.dtype, device=c.device), c], 1)          # [B, 9, 3]
    v = torch.cat([v0[:, None], torch.linalg.vector_norm(pose[:, 1:, :2] - pose[:, :-1, :2], dim=-1) / 0.5], 1)
    st = torch.cat([pose, v[..., None], torch.zeros(B, 9, 2, dtype=c.dtype, device=c.device)], -1)
    return st, torch.arange(9, dtype=torch.float64, device=c.device) * 0.5


# ----------------------------------------------------------------------------------------------- grid reads
def grid_coords(fr: Dict[str, Tensor], xy: Tensor, domain: str = "cell") -> Dict[str, Tensor]:
    """Points xy [B, P, 2] of rows b (fr = the rows' station frames, range_targets.parent_frame rows) -> s, d [B, P]
    (frenet), us = s / ds - 0.5, ud = (d + 4.8) / 0.6 (fractional station / lateral index), ok [B, P]; us / ud zeroed
    where not ok.  domain 'cell' (NC / TTC; range_targets._cells): on the grid when s in [-ds / 2, 48 ds) and d in
    [-5.1, 5.1); 'field' (DAC corners; range_targets.read_field 'grid'): any arc length (clamped to [s_0, s_47] by the
    read), |d| <= 4.8 m.  No gradient (states are inputs)."""
    with torch.no_grad():
        fp = frenet(xy, fr)
        return _domain(fp["s"], fp["d"], fp["ok"], fr["ds"].to(xy.dtype)[:, None], domain)


def _domain(s: Tensor, d: Tensor, ok_fp: Tensor, ds: Tensor, domain: str) -> Dict[str, Tensor]:
    ud = (d + LAT_MAX) / LAT_STEP
    if domain == "cell":
        half = 0.5 * LAT_STEP
        ok = ok_fp & (s >= -0.5 * ds) & (s < N_ST * ds) & (d >= -LAT_MAX - half) & (d < LAT_MAX + half)
    elif domain == "field":
        ok = ok_fp & (ud >= 0) & (ud <= N_LAT - 1)
    else:
        raise ValueError(f"domain {domain!r} not in ('cell', 'field')")
    us = torch.where(ok, s / ds - 0.5, torch.zeros_like(s))
    return {"s": s, "d": d, "us": us, "ud": torch.where(ok, ud, torch.zeros_like(ud)), "ok": ok, "ds": ds}


def read_cells(G: Tensor, rows: Tensor, us: Tensor, ud: Tensor, ch) -> Tensor:
    """Bilinear read of the parent-row grid G [R, 48, 17, C] (contiguous) at fractional indices us / ud [B, P]
    (clamped to the grid) of channel ch ([B, P] long or an int) for candidate rows rows [B] -> [B, P] (G's dtype;
    differentiable in G)."""
    R, NS, NL, C = G.shape
    usc = us.clamp(0.0, NS - 1.0)
    udc = ud.clamp(0.0, NL - 1.0)
    j0 = torch.floor(usc).long().clamp(0, NS - 2)
    i0 = torch.floor(udc).long().clamp(0, NL - 2)
    a = (usc - j0.to(usc.dtype)).clamp(0.0, 1.0).to(G.dtype)
    b = (udc - i0.to(udc.dtype)).clamp(0.0, 1.0).to(G.dtype)
    base = ((rows.reshape(-1, 1) * NS + j0) * NL + i0) * C + (ch if torch.is_tensor(ch) else int(ch))
    flat = G.reshape(-1)
    return (flat[base] * (1 - a) * (1 - b) + flat[base + NL * C] * a * (1 - b) + flat[base + C] * (1 - a) * b
            + flat[base + (NL + 1) * C] * a * b)


def nearest_cells(gc: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """nearest cell (range_targets._cells 'nearest') of grid_coords points: station j = floor(s / ds), lateral i =
    round(ud), both clamped to the grid (0 where not ok) -> j, i [B, P] long."""
    ok = gc["ok"]
    j = torch.where(ok, torch.floor(gc["s"] / gc["ds"]), torch.zeros_like(gc["s"])).long().clamp(0, N_ST - 1)
    i = torch.where(ok, torch.round(gc["ud"]), torch.zeros_like(gc["ud"])).long().clamp(0, N_LAT - 1)
    return j, i


def read_nearest(G: Tensor, rows: Tensor, j: Tensor, i: Tensor, ch) -> Tensor:
    """G [R, 48, 17, C] (contiguous) at cells (j, i) [B, P] (long) of channel ch ([B, P] long or an int) for rows
    [B] -> [B, P] (differentiable in G)."""
    R, NS, NL, C = G.shape
    idx = ((rows.reshape(-1, 1) * NS + j) * NL + i) * C + (ch if torch.is_tensor(ch) else int(ch))
    return G.reshape(-1)[idx]


def any4_cells(gc: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """the 2 x 2 grid points around grid_coords points (range_targets._cells 'any4': j0 = floor(s / ds - 0.5), i0 =
    floor((d + 4.8) / 0.6), points (j0, i0), (j0 + 1, i0), (j0, i0 + 1), (j0 + 1, i0 + 1), clamped to the grid) ->
    jn, in_ [B, P, 4] long, distinct [B, P, 4] bool (False for a clamped duplicate of an earlier point: the soft-OR
    counts each cell once; the hard OR does not care)."""
    us, ud = gc["us"], gc["ud"]                                          # zeroed where not ok (_domain)
    j0, i0 = torch.floor(us).long(), torch.floor(ud).long()
    jn = torch.stack([j0, j0 + 1, j0, j0 + 1], -1).clamp(0, N_ST - 1)
    in_ = torch.stack([i0, i0, i0 + 1, i0 + 1], -1).clamp(0, N_LAT - 1)
    dj, di = jn[..., 1] != jn[..., 0], in_[..., 2] != in_[..., 0]
    distinct = torch.stack([torch.ones_like(dj), dj, di, dj & di], -1)
    return jn, in_, distinct


def soft_read(G: Tensor, rows: Tensor, gc: Dict[str, Tensor], ch: Tensor, valid: Tensor, read: str = "nearest",
              tau: float = SMAX_TAU) -> Tensor:
    """Differentiable read of the parent-row logits G [R, 48, 17, C] at the points gc [B, P] (channel ch [B, P]) and
    soft maximum over the valid points (valid [B, P]) -> z [B]: 'nearest' = smax over the points of the nearest cell;
    'any4' = smax over the points x their distinct 2 x 2 cells (a soft OR of the 4 cells, nested in the soft max over
    the states: one tau-logsumexp over both)."""
    if read == "nearest":
        return smax(read_nearest(G, rows, *nearest_cells(gc), ch), valid, tau)
    if read != "any4":
        raise ValueError(f"read {read!r} not in ('nearest', 'any4')")
    jn, in_, distinct = any4_cells(gc)
    Bn, P = valid.shape
    chx = ch[..., None].expand(Bn, P, 4) if torch.is_tensor(ch) else ch
    v = read_nearest(G, rows, jn.reshape(Bn, P * 4), in_.reshape(Bn, P * 4),
                     chx.reshape(Bn, P * 4) if torch.is_tensor(chx) else chx)
    return smax(v, (valid[..., None] & distinct).reshape(Bn, P * 4), tau)


def lateral_read_rows(Hc: Tensor, rows: Tensor, d: Tensor) -> Tensor:
    """readout.lateral_read on parent rows without gathering Hc per candidate: Hc [R, C, 48, 17], rows [B], d [B, 48]
    (m, clamped +-4.8) -> [B, 48, C] (linear between the two lateral offsets around d; differentiable in Hc)."""
    B = rows.shape[0]
    u = ((d.float().clamp(-LAT_MAX, LAT_MAX) + LAT_MAX) / LAT_STEP)
    i0 = torch.floor(u).long().clamp(0, N_LAT - 2)
    w = (u - i0.float()).clamp(0.0, 1.0).to(Hc.dtype)[..., None]
    H = Hc.permute(0, 2, 3, 1)                                                     # [R, 48, 17, C] view
    r = rows.reshape(B, 1).expand(B, N_ST)
    jj = torch.arange(N_ST, device=Hc.device)[None].expand(B, N_ST)
    f0 = H[r, jj, i0]
    f1 = H[r, jj, i0 + 1]
    return f0 + w * (f1 - f0)


def time_bin(t: Tensor, tbin: float) -> Tensor:
    """bin index floor(t / tbin + 1e-6) of float64 times (range_targets.time_bins)."""
    return torch.floor(t.double() / float(tbin) + T_EPS).long()


@torch.no_grad()
def hard_slack(blk_b: Tensor, rows: Tensor, gc: Dict[str, Tensor], t: Tensor, chan, use: Tensor, M: int, tbin: float,
               cap: float = SLACK_CAP, read: str = "nearest") -> Tuple[Tensor, Tensor, Tensor]:
    """Per-point signed time slack on the thresholded fields (range_targets._time_slack): blk_b [R, 48, 17, G M] bool
    (B | U [| Bp], G = 2 or 3 channel groups), points gc (grid_coords, [B, P]), read times t [P] or [B, P] (float64),
    chan [B, P] (group: 0 B, 1 U, 2 Bp) or an int, use [B, P]; read 'nearest' (the nearest cell's profile) | 'any4' (the
    OR of the 2 x 2 cells' profiles, = range_targets.executed_slack(read='any4')) -> (slack [B, P] f32 (cap where not
    read), station j, lateral i [B, P] long: the NEAREST cell in both modes, as executed_slack's crit_station)."""
    Bn, P = gc["s"].shape
    ok = gc["ok"]
    j, i = nearest_cells(gc)
    rr = rows.reshape(Bn, 1).expand(Bn, P)
    G = blk_b.shape[-1] // M
    if G * M != blk_b.shape[-1] or G not in (2, 3):
        raise ValueError(f"hard_slack: blk has {blk_b.shape[-1]} channels, not 2 or 3 groups of M = {M}")
    if read == "nearest":
        prof2 = blk_b[rr, j, i]                                                         # [B, P, G M]
    elif read == "any4":
        jn, in_, _ = any4_cells(gc)
        prof2 = blk_b[rr[..., None].expand(Bn, P, 4), jn, in_]                          # [B, P, 4, G M]
    else:
        raise ValueError(f"read {read!r} not in ('nearest', 'any4')")
    if torch.is_tensor(chan):
        c_ = chan.long()[..., None] if read == "nearest" else chan.long()[..., None, None]
        prof = prof2[..., :M]
        for g in range(1, G):                                                           # group g where chan == g
            prof = torch.where(c_ == g, prof2[..., g * M:(g + 1) * M], prof)
    else:
        g = int(chan)
        prof = prof2[..., g * M:(g + 1) * M]
    if read == "any4":
        prof = prof.any(-2)                                                             # OR over the 4 cells
    tt = t.double().expand(Bn, P) if t.dim() == 1 else t.double()
    sl = RT._time_slack(prof, tt, tbin, cap).float()
    return torch.where(ok & use, sl, torch.full_like(sl, float(cap))), j, i


@torch.no_grad()
def grid_margins(D: Tensor, ds: Tensor, step: float = MARGIN_STEP, lat_max: float = LAT_MAX) -> Dict[str, Tensor]:
    """Fast drivable tube of P parents from a margin field on their own grid (the edits' and vmax's version of
    range_targets.margins 'grid'): the box with its rear axle at (s_j, delta) is approximated in the station frame
    (corners at s_j + (-1.127, +4.049) m along, delta +- 1.1485 m across; the path's curvature and the heading
    relative to the path ignored), f_j(delta) = min over the 4 corners of the bilinear D (s clamped to [s_0, s_47]; a
    corner beyond |d| = 4.8 m is unreadable).  Scan delta = 0, step, .., lat_max on each side -> m_left, m_right
    [P, 48] (m) with range_targets.margins' rules (first crossing f <= 0 linearly interpolated; an unreadable sample
    ends the scan at the last positive one; lat_max if f stays positive; f_j(0) <= 0 -> f_j(0)), f0 [P, 48]."""
    P = D.shape[0]
    dev = D.device
    K = int(round(float(lat_max) / float(step)))
    dl = torch.arange(K + 1, device=dev, dtype=torch.float32) * float(step)            # [K + 1]
    j = torch.arange(N_ST, device=dev, dtype=torch.float32)
    dsf = ds.float().reshape(P, 1, 1, 1)
    us = torch.stack([j[None, :, None] + A_FRONT / dsf[..., 0], j[None, :, None] + A_REAR / dsf[..., 0]], -1)
    us = us.expand(P, N_ST, K + 1, 2)                                                  # [P, 48, K + 1, 2]

    def side(sign: float):
        lat = sign * dl[None, None, :, None] + torch.tensor([HALF_WID, -HALF_WID], device=dev)   # [1, 1, K+1, 2]
        ud = ((lat + LAT_MAX) / LAT_STEP).expand(P, N_ST, K + 1, 2)
        readable = ((ud >= -1e-6) & (ud <= N_LAT - 1 + 1e-6)).all(-1)               # [P, 48, K + 1] both sides
        uu = us[..., :, None].expand(P, N_ST, K + 1, 2, 2).reshape(P, -1)
        vv = ud[..., None, :].expand(P, N_ST, K + 1, 2, 2).reshape(P, -1)
        v = read_cells(D.float()[..., None].contiguous(), torch.arange(P, device=dev), uu, vv, 0)
        f = v.reshape(P, N_ST, K + 1, 4).min(-1).values
        f = torch.where(readable, f, torch.full_like(f, -math.inf))
        pos = f > 0
        n_lead = torch.cumprod(pos.to(torch.int64), -1).sum(-1)
        capped = n_lead == K + 1
        kl = (n_lead - 1).clamp(min=0)
        kf = n_lead.clamp(max=K)
        fl = torch.gather(f, -1, kl[..., None])[..., 0]
        ff = torch.gather(f, -1, kf[..., None])[..., 0]
        frac = torch.where(torch.isfinite(ff), fl / torch.clamp(fl - ff, min=1e-12), torch.zeros_like(fl))
        m = kl.float() * float(step) + float(step) * frac.clamp(0.0, 1.0)
        m = torch.where(capped, torch.full_like(m, float(lat_max)), m)
        f0 = f[..., 0]
        return torch.where(n_lead == 0, torch.where(torch.isfinite(f0), f0, torch.zeros_like(f0)), m), f0

    m_left, f0 = side(1.0)
    m_right, _ = side(-1.0)
    return {"m_left": m_left, "m_right": m_right, "f0": torch.where(torch.isfinite(f0), f0, torch.zeros_like(f0))}


def _ttc_points(st: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """states [B, Q, >= 4] -> projected rear axles [B, Q, 4, 2] (|v| delta along the heading), shift |v| delta
    [B, Q, 4], |v| [B, Q]."""
    xy, hd = st[..., :2], st[..., 2]
    spd = st[..., 3].abs()
    dl = torch.as_tensor(TTC_DELTAS, dtype=st.dtype, device=st.device)
    eh = torch.stack([torch.cos(hd), torch.sin(hd)], -1)
    shift = spd[..., None] * dl
    return xy[:, :, None, :] + shift[..., None] * eh[:, :, None, :], shift, spd


@torch.no_grad()
def read_geometry(fr_rows: Dict[str, Tensor], states: Tensor, t: Tensor, M: int, tbin: float,
                  ttc_b_chan: int = CH_B) -> Dict[str, Tensor]:
    """Every point the range readout reads for B candidates (rows of fr_rows = their parents' frames): states
    [B, Q, 6] at t [Q] (float64) -> grid coordinates of the NC points (rear axles, [B, Q]), the TTC points ([B, Q * 4],
    state-major, delta-minor), the DAC corners ([B, Q * 4], FL RL RR FR), their time bins, read masks (stopped-ego
    skips, on grid, time bin in [0, M)) and the TTC channel group (CH_U = 1 where delta > 0 and |v| delta >= U_SWITCH_M,
    else ttc_b_chan: CH_B = 0 (default) or CH_BP = 2, the plain not-behind B of an rg_stopped_aware readout)."""
    if int(ttc_b_chan) not in (CH_B, CH_BP):
        raise ValueError(f"ttc_b_chan {ttc_b_chan} not in ({CH_B}, {CH_BP})")
    B, Q = states.shape[:2]
    st = states.float()
    pr, shift, spd = _ttc_points(st)
    cor = ego_corners(st[..., :3])
    nd = len(TTC_DELTAS)
    pts = torch.cat([st[..., :2], pr.reshape(B, Q * nd, 2), cor.reshape(B, Q * 4, 2)], 1)
    fp = frenet(pts, fr_rows)                                                      # one projection of every point
    ds = fr_rows["ds"].to(pts.dtype)[:, None]
    sl = lambda a, b, dom: _domain(fp["s"][:, a:b], fp["d"][:, a:b], fp["ok"][:, a:b], ds, dom)
    g_nc, g_tt = sl(0, Q, "cell"), sl(Q, Q + Q * nd, "cell")
    g_dc = sl(Q + Q * nd, Q + Q * nd + Q * 4, "field")
    dl = torch.as_tensor(TTC_DELTAS, dtype=torch.float64, device=st.device)
    t_tt = (t.double()[:, None] + dl[None]).reshape(-1)                                # [Q * 4]
    m_nc, m_tt = time_bin(t, tbin), time_bin(t_tt, tbin)
    in_nc, in_tt = (m_nc >= 0) & (m_nc < M), (m_tt >= 0) & (m_tt < M)
    use_nc = spd > STOP_NC
    use_tt = (spd >= STOP_TTC)[..., None].expand(B, Q, nd).reshape(B, -1)
    use_u = ((shift >= U_SWITCH_M) & (dl.to(st.dtype) > 0)).reshape(B, -1)
    chan = use_u.long() if int(ttc_b_chan) == CH_B else torch.where(use_u, CH_U, int(ttc_b_chan)).long()
    return {"nc": g_nc, "ttc": g_tt, "dac": g_dc, "t_nc": t.double(), "t_ttc": t_tt,
            "m_nc": m_nc.clamp(0, M - 1), "m_ttc": m_tt.clamp(0, M - 1),
            "rd_nc": g_nc["ok"] & use_nc & in_nc[None], "use_nc": use_nc & in_nc[None],
            "rd_ttc": g_tt["ok"] & use_tt & in_tt[None], "use_ttc": use_tt & in_tt[None], "chan": chan, "spd": spd}


@torch.no_grad()
def hard_evidence(blk_b: Tensor, D: Tensor, rows: Tensor, geo: Dict[str, Tensor], M: int, tbin: float,
                  read: str = "nearest") -> Dict:
    """Slacks and critical points of B candidates on the thresholded parent fields (module docstring): blk_b
    [R, 48, 17, 2M | 3M] bool (B | U [| Bp]; NC reads group 0, TTC the groups of geo 'chan'), D [R, 48, 17] margin (m),
    rows [B], geo = read_geometry(...) (ttc_b_chan CH_BP for a 3-group field), read (hard_slack: B / U / Bp read
    'nearest' | 'any4'; D is bilinear in both) -> slack_nc, slack_ttc [B] (s), slack_dac [B] (m; +10 if no readable
    corner), crit_station / crit_lat [B, 3] long, crit_time [B, 3] (s), crit_chan [B] long, state_dac [B, Q] (min
    readable corner margin per state, +inf none)."""
    Bn = rows.shape[0]
    Q = geo["nc"]["s"].shape[1]
    nd = len(TTC_DELTAS)
    dev = rows.device
    bidx = torch.arange(Bn, device=dev)
    tq = geo["t_nc"].float()
    crit_s = torch.full((Bn, 3), -1, dtype=torch.long, device=dev)
    crit_i = torch.full((Bn, 3), -1, dtype=torch.long, device=dev)
    crit_t = torch.full((Bn, 3), -1.0, dtype=torch.float32, device=dev)
    sn, jn, in_ = hard_slack(blk_b, rows, geo["nc"], geo["t_nc"], 0, geo["use_nc"], M, tbin, read=read)
    s_nc, q_nc = sn.min(-1)
    has = s_nc < SLACK_CAP
    crit_s[:, I_CNC] = torch.where(has, jn[bidx, q_nc], crit_s[:, I_CNC])
    crit_i[:, I_CNC] = torch.where(has, in_[bidx, q_nc], crit_i[:, I_CNC])
    crit_t[:, I_CNC] = torch.where(has, tq[q_nc], crit_t[:, I_CNC])
    stt, jt, it = hard_slack(blk_b, rows, geo["ttc"], geo["t_ttc"], geo["chan"], geo["use_ttc"], M, tbin, read=read)
    s_tt, q_tt = stt.min(-1)
    has = s_tt < SLACK_CAP
    crit_s[:, I_CTTC] = torch.where(has, jt[bidx, q_tt], crit_s[:, I_CTTC])
    crit_i[:, I_CTTC] = torch.where(has, it[bidx, q_tt], crit_i[:, I_CTTC])
    crit_t[:, I_CTTC] = torch.where(has, tq[q_tt // nd], crit_t[:, I_CTTC])
    chan = torch.where(has, geo["chan"][bidx, q_tt], torch.zeros_like(q_tt))
    g = geo["dac"]
    Dv = read_cells(D.float()[..., None].contiguous(), rows, g["us"], g["ud"], 0)
    Dv = torch.where(g["ok"], Dv, torch.full_like(Dv, math.inf)).reshape(Bn, Q, 4).min(-1).values   # [B, Q]
    s_d, n_d = Dv.min(-1)
    has_d = torch.isfinite(s_d)
    on = geo["nc"]["ok"][bidx, n_d]
    crit_s[:, I_CDAC] = torch.where(has_d & on, jn[bidx, n_d], crit_s[:, I_CDAC])
    crit_i[:, I_CDAC] = torch.where(has_d & on, in_[bidx, n_d], crit_i[:, I_CDAC])
    crit_t[:, I_CDAC] = torch.where(has_d, tq[n_d], crit_t[:, I_CDAC])
    return {"slack_nc": s_nc, "slack_ttc": s_tt, "slack_dac": torch.where(has_d, s_d, torch.full_like(s_d, SDF_CLIP)),
            "crit_station": crit_s, "crit_lat": crit_i, "crit_time": crit_t, "crit_chan": chan, "state_dac": Dv}


@torch.no_grad()
def field_hits(blk_b: Tensor, D: Tensor, fr: Dict[str, Tensor], rows: Tensor, states: Tensor, t: Tensor, M: int,
               tbin: float, read: str = "nearest", ttc_b_chan: Optional[int] = None) -> Dict[str, Tensor]:
    """The readout's own thresholded rule on GIVEN fields (e.g. GT fields: the 'readout-rule perfect-field' tube of a
    teacher whose fields equal their targets): blk_b [R, 48, 17, 2M | 3M] bool (B | U [| Bp]), D [R, 48, 17] (m), fr =
    range_targets.parent_frame of the R rows (read in float32, as RangeReadout.forward), rows [B] (candidate -> row),
    states [B, Q, 6] at t [Q] (float64 s; candidate_states), read 'nearest' | 'any4', ttc_b_chan (None: CH_BP for a
    3-group field, else CH_B) -> hard_evidence(...) + hit_nc / hit_ttc / hit_dac [B] bool (slack < 0)."""
    G = blk_b.shape[-1] // int(M)
    ch = (CH_BP if G == 3 else CH_B) if ttc_b_chan is None else int(ttc_b_chan)
    fr_rows = RT.gather_frame({k: fr[k].float() for k in ("p", "u", "n", "s", "ds")}, rows)
    geo = read_geometry(fr_rows, states.float(), t, int(M), float(tbin), ttc_b_chan=ch)
    h = hard_evidence(blk_b.bool(), D.float(), rows, geo, int(M), float(tbin), read=read)
    h.update(hit_nc=h["slack_nc"] < 0, hit_ttc=h["slack_ttc"] < 0, hit_dac=h["slack_dac"] < 0)
    return h


# ----------------------------------------------------------------------------------------------- readout
class RangeReadout(nn.Module):
    """V4Cfg.readout == 'range' heads (module docstring).  Built under the caller's seeded RNG."""

    def __init__(self, cfg: V4Cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        d = d_model
        M = int(cfg.ev_tbins)
        h = int(cfg.rg_hidden)
        self.M = M
        self.tbin = float(cfg.ev_tbin_s)
        # channel groups: B | U, + Bp (the plain not-behind B, TTC's B read) with rg_stopped_aware (module docstring)
        self.n_groups = 3 if cfg.rg_stopped_aware else 2
        self.ttc_b_chan = CH_BP if cfg.rg_stopped_aware else CH_B
        self.blk_head = _mlp(HC_CH + d, h, self.n_groups * M)   # [B bins | U bins (| Bp bins)] logits per cell
        self.marg_head = _mlp(HC_CH + d, h, 1)                  # D [m]
        # PRIVILEGED GT-object input (V4Cfg.rg_priv, config docstring): [R, 48, 17, M] constant-velocity occupancy as a
        # zero-initialised 1x1-conv term of both per-cell MLPs (off: no module, the heads are bitwise unchanged)
        # (plain zero parameters, no RNG draw: the seeded modules after them are bitwise those of the plain net)
        self.priv_blk_w = self.priv_marg_w = None
        if cfg.rg_priv != "off":
            self.priv_blk_w = nn.Parameter(torch.zeros(h, M, 1, 1))
            self.priv_marg_w = nn.Parameter(torch.zeros(h, M, 1, 1))
        self.log_eps = nn.Parameter(torch.tensor(math.log(float(cfg.ev_eps))))
        self.resid = _mlp(N_RES + d, h, 3, zero_last=True) if cfg.ev_resid else None
        ep_in = 3 * d + 3 + ((N_EP_EV + d + HC_CH) if cfg.ep_evidence else 0)
        self.ep_head = _mlp(ep_in, int(cfg.ep_ev_hidden if cfg.ep_evidence else cfg.ep_hidden), 1)
        self.comfort_head = _mlp(d, int(cfg.comfort_hidden), 1)
        self.lat_head = _mlp(2 * d + HC_CH, int(cfg.lat_hidden), N_FREE, zero_last=True)

    # refiner v5 (ck/v5; off unless CKNet calls enable_v5): extra residual inputs, relation log-pass terms, per-state
    # reads Z and the executed-corner margins
    v5_on = False

    def enable_v5(self, n_extra: int, rel: bool, unknown_ev: bool, z_stride: int) -> None:
        """refiner v5 (ck/v5/config.py): the residual MLP takes n_extra more inputs after [ev(7), g] (rebuilt here; the
        caller seeds the RNG; a warm start from a v4_3 run copies the old columns and zeroes the new ones), rel adds the
        direct log-pass weights rel_w [2] (NC, TTC; zero-initialised), forward() then also returns zq / unknown_frac /
        corner_D / corner_ok."""
        if self.resid is None:
            raise ValueError("v5 needs the range readout's residual MLP (V4Cfg.ev_resid)")
        d = self.resid[0].in_features - N_RES
        if int(n_extra) > 0:
            self.resid = _mlp(N_RES + d + int(n_extra), int(self.cfg.rg_hidden), 3, zero_last=True)
        self.v5_on, self.v5_n_extra, self.v5_rel = True, int(n_extra), bool(rel)
        self.v5_unknown, self.v5_stride = bool(unknown_ev), int(z_stride)
        self.rel_w = nn.Parameter(torch.zeros(2)) if rel else None

    @torch.no_grad()
    def set_prior(self, prior: Sequence[float]) -> None:
        """B / U logits biased so that a uniform field read at every executed state gives the NC pass prior (logit
        -logit(p_nc) - tau ln Q, Q = 41 states for 'lqr' / 'ref', 9 for 'plan'), margins +2 m, EP / comfort heads
        logit(prior); NC / TTC / DAC pass rates then follow from the evidence (the residual starts at 0)."""
        p = np.clip(np.asarray(prior, np.float64), 1e-4, 1 - 1e-4)
        q = LQR.NT if self.cfg.exec_model in LQR.EXECUTORS else 9
        if self.cfg.rg_read == "any4":                  # the soft max also runs over the 4 cells of each read
            q *= 4
        self.blk_head[-1].bias.fill_(-math.log(p[I_NC] / (1 - p[I_NC])) - SMAX_TAU * math.log(q))
        self.marg_head[-1].bias.fill_(2.0)
        for head, i in ((self.ep_head, I_EP), (self.comfort_head, I_C)):
            head[-1].bias.fill_(math.log(p[i] / (1 - p[i])))

    @staticmethod
    def _cell_mlp(mlp: nn.Sequential, xs: Tensor, Hc: Tensor, hp: Optional[Tensor] = None) -> Tensor:
        """mlp (Linear(128 + d, h), GELU, Linear(h, o)) on [Hc(j, i) || x'_j] of every cell, as 1x1 convolutions ->
        [R, 48, 17, o] (f32).  hp [R, h, 48, 17]: an extra pre-activation term (the privileged input's 1x1 conv)."""
        W1, b1, W2, b2 = mlp[0].weight, mlp[0].bias, mlp[2].weight, mlp[2].bias
        hc = F.conv2d(Hc, W1[:, :HC_CH, None, None])                                  # [R, h, 48, 17]
        hx = F.linear(xs, W1[:, HC_CH:], b1)                                          # [R, 48, h]
        pre = hc + hx.transpose(1, 2)[..., None].to(hc.dtype)
        if hp is not None:
            pre = pre + hp.to(hc.dtype)
        z = F.gelu(pre)
        return F.conv2d(z, W2[:, :, None, None], b2).permute(0, 2, 3, 1).float().contiguous()

    def scene(self, xs: Tensor, Hc: Tensor, priv: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """xs [R, 48, d], Hc [R, 128, 48, 17] (parent rows) -> blk [R, 48, 17, 2M] logits (B | U; [R, 48, 17, 3M]
        B | U | Bp with rg_stopped_aware), margin [R, 48, 17] (m), both f32 contiguous.  priv [R, 48, 17, M]
        (range_targets.priv_channels): required iff V4Cfg.rg_priv != 'off'."""
        hb = hm = None
        if self.priv_blk_w is not None:
            if priv is None:
                raise ValueError(f"range readout with rg_priv {self.cfg.rg_priv!r}: scene() needs priv [R, 48, 17, M] "
                                 f"(range_targets.priv_channels of the batch's ref_obj_*)")
            if tuple(priv.shape) != (Hc.shape[0], N_ST, N_LAT, self.M):
                raise ValueError(f"range readout: priv {tuple(priv.shape)} vs [{Hc.shape[0]}, {N_ST}, {N_LAT}, {self.M}]")
            pv = priv.permute(0, 3, 1, 2).to(Hc.dtype).contiguous()                   # [R, M, 48, 17]
            hb, hm = F.conv2d(pv, self.priv_blk_w.to(pv.dtype)), F.conv2d(pv, self.priv_marg_w.to(pv.dtype))
        elif priv is not None:
            raise ValueError("priv given to a range readout without rg_priv")
        return {"blk": self._cell_mlp(self.blk_head, xs, Hc, hb),
                "margin": self._cell_mlp(self.marg_head, xs, Hc, hm)[..., 0]}

    def zq_at(self, fields: Dict[str, Tensor], rows: Tensor, fr: Dict[str, Tensor], states: Tensor, t: Tensor
              ) -> Tensor:
        """refiner v5: the per-state reads zq [B, Q', 4] of these fields at the given states, no scoring -- forward()'s
        zq for the same (fields, rows, fr, states, t) bit for bit.  The teachers' Z-KD targets at the no-LQR student's
        states (CKNet.score_set zq_exec, TeacherPair2.run): the same cells as the student's own reads."""
        from ..v5.zread import state_reads, zq_of
        if not self.v5_on:
            raise ValueError("range readout: zq_at needs the v5 per-state reads (v5 on)")
        with torch.autocast(device_type=states.device.type, enabled=False):
            fr_rows = RT.gather_frame({k: fr[k].float() for k in ("p", "u", "n", "s", "ds")}, rows)
            st = states.float()
            geo = read_geometry(fr_rows, st, t, self.M, self.tbin, ttc_b_chan=self.ttc_b_chan)
            sr = state_reads(fields["blk"], fields["margin"], rows, geo, self.M, self.cfg.rg_read, states=st)
            return zq_of(sr, self.v5_stride)

    def forward(self, fields: Dict[str, Tensor], rows: Tensor, fr: Dict[str, Tensor], states: Tensor, t: Tensor,
                xs: Tensor, xd: Tensor, Hc: Tensor, g: Tensor, S8: Tensor, d_st: Tensor, t_in: Tensor,
                t_leave: Tensor, pref: Optional[Tensor] = None, v5: Optional[Dict[str, Tensor]] = None
                ) -> Dict[str, Tensor]:
        """fields = scene(...) of the R parent rows, rows [B] (candidate -> parent row), fr = range_targets.parent_frame
        of the R rows, states [B, Q, 6] at t [Q] (candidate_states), xs [B, 48, d] / xd [B, d] (the candidate's
        parent's station / draft tokens), Hc [R, 128, 48, 17] (parent rows), g [B, d] (draft_mlp of the candidate),
        S8 [B] (candidate draft length), d_st / t_in / t_leave [B, 48] (sibling.on_parent: planned lateral offset and
        station-bin entry / leave times), pref [B] log1p P_ref | None -> dict (module docstring): score_logit [B, 5],
        w_lat [B, 6], lp_nc / lp_ttc / lp_dac, slack_nc / slack_ttc / slack_dac, min_margin, exec_ratio, overspeed [B],
        crit_station / crit_lat [B, 3] long, crit_time [B, 3], crit_chan [B], mask / t_out [B, 48]."""
        B = rows.shape[0]
        dt = xs.dtype
        dev = xs.device
        M = self.M
        blk, D = fields["blk"], fields["margin"]
        if blk.shape[-1] != self.n_groups * M:
            raise ValueError(f"range readout: blk has {blk.shape[-1]} channels, expected {self.n_groups} x {M} "
                             f"(rg_stopped_aware {bool(self.cfg.rg_stopped_aware)})")
        with torch.autocast(device_type=dev.type, enabled=False):
            fr_rows = RT.gather_frame({k: fr[k].float() for k in ("p", "u", "n", "s", "ds")}, rows)
            st = states.float()
            geo = read_geometry(fr_rows, st, t, M, self.tbin, ttc_b_chan=self.ttc_b_chan)
            # ---- differentiable evidence on the fields
            gn, gt_, gd = geo["nc"], geo["ttc"], geo["dac"]
            rd = self.cfg.rg_read
            z_nc = soft_read(blk, rows, gn, geo["m_nc"][None].expand_as(gn["us"]), geo["rd_nc"], rd)
            z_ttc = soft_read(blk, rows, gt_, geo["m_ttc"][None] + M * geo["chan"], geo["rd_ttc"], rd)
            Dc = read_cells(D[..., None], rows, gd["us"], gd["ud"], 0)                       # [B, 4Q]
            eps = torch.exp(self.log_eps.float()).clamp(min=0.01)
            z_dac = smax(-Dc / eps, gd["ok"])
            lp_nc, lp_ttc, lp_dac = F.logsigmoid(-z_nc), F.logsigmoid(-z_ttc), F.logsigmoid(-z_dac)
            slack_dac = torch.where(gd["ok"], Dc, torch.full_like(Dc, math.inf)).min(-1).values
            slack_dac = torch.where(torch.isfinite(slack_dac), slack_dac, torch.full_like(slack_dac, SDF_CLIP))
            # ---- thresholded fields: slacks / critical points (no gradient)
            hard = hard_evidence(blk.detach() > 0, D.detach(), rows, geo, M, self.tbin, read=rd)
            v5o = None
            if self.v5_on:                                  # refiner v5: per-state reads Z (ck/v5/zread.py)
                from ..v5.zread import state_reads, zq_of
                sr = state_reads(blk, D, rows, geo, M, rd, states=st)
                v5o = {"zq": zq_of(sr, self.v5_stride), "unknown_frac": sr["unknown"].float().mean(-1),
                       "corner_D": Dc, "corner_ok": gd["ok"]}
            S_exec = LQR.executed_progress(st)[0]
            overspeed = torch.zeros(B, dtype=torch.float32, device=dev)
            if self.cfg.vmax == "feature":
                overspeed = self.overspeed(D.detach(), fr, rows, geo)
        logit = torch.zeros(B, len(CK_KEYS), dtype=torch.float32, device=dev)
        logit[:, I_NC] = -z_nc
        logit[:, I_TTC] = -z_ttc
        logit[:, I_DAC] = -z_dac
        five = lambda x: (x.float() / 5.0)
        ev = torch.stack([five(lp_nc), five(lp_ttc), five(lp_dac), five(hard["slack_nc"]), five(hard["slack_ttc"]),
                          five(slack_dac.clamp(-SDF_CLIP, SDF_CLIP)), five(overspeed).clamp(max=3.0)], -1)
        if self.resid is not None:
            rin = [ev.to(dt), g]
            if v5o is not None and self.v5_n_extra:
                ext = []
                if self.v5_rel:
                    if v5 is None or v5.get("ev_extra") is None:
                        raise ValueError("range readout with the v5 relation channel: forward(..., v5={'ev_extra', "
                                         "'rel_lp'}) is required")
                    ext.append(v5["ev_extra"].float())
                if self.v5_unknown:
                    ext.append(v5o["unknown_frac"][:, None])
                rin.append(torch.cat(ext, -1).to(dt))
            r = self.resid(torch.cat(rin, -1)).float()
            logit[:, I_NC] = logit[:, I_NC] + r[:, 0]
            logit[:, I_TTC] = logit[:, I_TTC] + r[:, 1]
            logit[:, I_DAC] = logit[:, I_DAC] + r[:, 2]
        if v5o is not None and self.v5_rel:              # direct relation log-pass terms (zero-initialised weights)
            lp_rel = v5["rel_lp"].float()
            logit[:, I_NC] = logit[:, I_NC] + self.rel_w[0].float() * lp_rel[:, 0]
            logit[:, I_TTC] = logit[:, I_TTC] + self.rel_w[1].float() * lp_rel[:, 1]
        # ---- stations read on the planned schedule (EP / lateral heads, v4_2 keys)
        H = float(self.cfg.horizon)
        m = read_mask(t_in, H)
        t_out = torch.where(m, torch.minimum(t_leave.float(), torch.full_like(t_leave.float(), H)),
                            torch.zeros_like(t_leave.float()))
        mf = m.to(dt)[..., None]
        xs_mean = (xs * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        # margin-weighted stations along the candidate's planned lateral offset
        jj = torch.arange(N_ST, device=dev, dtype=torch.float32)[None].expand(B, N_ST)
        ud = (d_st.float().clamp(-LAT_MAX, LAT_MAX) + LAT_MAX) / LAT_STEP
        dpath = read_cells(D[..., None], rows, jj, ud, 0)                                   # [B, 48]
        w = torch.where(m, -dpath.detach() / eps.detach(), torch.full_like(dpath, -1e4))
        w = torch.softmax(w, -1)
        w = torch.where(m.any(-1, keepdim=True), w, torch.zeros_like(w)).to(dt)
        f = lateral_read_rows(Hc, rows, d_st).to(dt)
        wxs, wf = (w[..., None] * xs).sum(1), (w[..., None] * f).sum(1)
        S = S8.float().clamp(min=0.0)
        if pref is None:
            lp_ref, ratio, ratio_x = torch.zeros_like(S), torch.ones_like(S), torch.ones_like(S)
        else:
            lp_ref = pref.detach().float()
            P = torch.expm1(lp_ref).clamp(min=0.0)
            ratio = S / torch.maximum(torch.maximum(P, S), torch.full_like(S, EP_MIN_DIST))
            Sx = S_exec.float().clamp(min=0.0)
            ratio_x = Sx / torch.maximum(torch.maximum(P, Sx), torch.full_like(Sx, EP_MIN_DIST))
        ep_parts = [g, xd, xs_mean, torch.stack([torch.log1p(S) / 4.0, lp_ref / 4.0, ratio], -1).to(dt)]
        if self.cfg.ep_evidence:
            ep_parts += [torch.cat([ev[:, :6], ratio_x[:, None]], -1).detach().to(dt), wxs.detach(), wf.detach()]
        logit[:, I_EP] = self.ep_head(torch.cat(ep_parts, -1))[:, 0].float()
        logit[:, I_C] = self.comfort_head(g)[:, 0].float()
        w_lat = self.lat_head(torch.cat([g, wxs, wf], -1)).float()
        out = {"score_logit": logit, "w_lat": w_lat, "lp_nc": lp_nc, "lp_ttc": lp_ttc, "lp_dac": lp_dac,
               "slack_nc": hard["slack_nc"], "slack_ttc": hard["slack_ttc"], "slack_dac": slack_dac,
               "min_margin": slack_dac, "exec_ratio": ratio_x, "S_exec": S_exec, "overspeed": overspeed,
               "mask": m, "t_out": t_out}
        out.update({k: hard[k] for k in ("crit_station", "crit_lat", "crit_time", "crit_chan")})
        if v5o is not None:
            out.update(v5o)
        return out

    @torch.no_grad()
    def overspeed(self, D: Tensor, fr: Dict[str, Tensor], rows: Tensor, geo: Dict[str, Tensor]) -> Tensor:
        """vmax 'feature': max over the moving on-grid states of relu(|v_n| - v_max(row, station of s_n)) [B] (m/s);
        v_max = range_targets.v_max(parent curvature at the stations, outside margin of grid_margins(D))."""
        mg = grid_margins(D, fr["ds"])
        kap = fr["kappa"].float()
        vm = RT.v_max(kap, RT.outside_margin(kap, mg["m_left"], mg["m_right"]))      # [R, 48]
        g = geo["nc"]
        j = torch.where(g["ok"], torch.floor(g["s"] / g["ds"]), torch.zeros_like(g["s"])).long().clamp(0, N_ST - 1)
        v_lim = vm[rows.reshape(-1, 1).expand_as(j), j]
        over = torch.relu(geo["spd"] - v_lim)
        return torch.where(g["ok"] & (geo["spd"] > STOP_NC), over, torch.zeros_like(over)).amax(-1)
