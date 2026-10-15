"""M7 differentiable surrogate losses of the stage-T refiner (IMPL_SPEC §3.7).  torch, batched.

Every term is computed on the 41-point dense reference (t = 0.1 n, n = 0..40) of a trajectory, i.e. the positions the
official scorer's LQR tracker is asked to follow (geometry.dense_reference; decoder.decode()['dense']).

Frames, units, time
  N frame  : NAVSIM ego frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x; metres, radians).
  Ego box  : Pacifica, half length 2.588 m, half width 1.1485 m, box centre 1.461 m ahead of the rear axle
             (geometry.HALF_LEN / HALF_WID / RA2C; = sf_common, sdf.ego_corners, the metric-cache vehicle).
  Objects  : boxes (cx, cy, heading, L, W) in N at the 41 dense times + state OBS / ABSENT_OFFICIAL, from
             gt_future.query / query_torch (metric-cache rule, verified to 7.6e-6 m).
  Batching : per-token ("scene") tensors have a leading S axis; drafts have a leading B axis and ``index`` [B] maps
             draft b -> scene row (e.g. the 13 drafts of a token share one scene).  Default: arange(B) if S == B,
             zeros if S == 1.
  Points   : all costs are means over n = n_from..40 (N_FROM = 1): the t0 pose (n = 0) is the same for every
             draft of a token, so it carries no gradient and would only add a draft-independent constant.

Terms (per draft, [B])
  C_col  collision with GT future objects, box-box smooth signed separation
           gap_u = |d . u| - r_ego(u) - r_obj(u) for the 4 box axes u (d = centre offset; r = support half-extent)
           g_hard = max_u gap_u (> 0 separated, a lower bound of the Euclidean gap; <= 0 overlapping, -g_hard =
           minimum SAT penetration depth), g = T logsumexp(gap / T) - T ln 4 (T = 0.05 m; the bias correction puts g in
           [g_hard - T ln 4, g_hard]).
           C_col = mean_n sum_j w_j m_j(n) beta softplus((m_col - g_jn) / beta), m_col = 0.3 m, beta = 0.1 m,
           w_j = 1 (agent: vehicle / pedestrian / bicycle) or 0.5 (static; official NC 0 vs 0.5),
           m_j(n) = OBS  AND  not behind (object centre >= -1.3 m along the ego heading from the ego centre)
                    AND NOT human_overlap(j, n)  (the HUMAN footprint overlaps object j at n: g_hard_human <= 0).
  C_ttc  time-to-collision (PRESTATED_DECISION_RULE AMENDMENT 4 (1); mirrors pdm_scorer._calculate_ttc; weight 'ttc',
           DEFAULT_WEIGHTS 0 -> off unless enabled): for n = n_from..40 and delta_k in ttc_deltas = (0.3, 0.6, 0.9) s the
           ego box at pose_n is translated by v_n delta_k along h_n (v_n = ref_speed, forward difference of the dense
           poses; h_n = dense heading) and compared with the objects at dense time n + 10 delta_k (objects 0..5 s =
           51 dense times, SceneBatch.boxes_ttc / obs_ttc; pairs whose object time lies beyond the stored times or where
           the object is not OBS -- e.g. t > t_avail -- are EXCLUDED, never clamped):
             C_ttc = mean_n sum_k sum_j w_j m_jnk beta softplus((m_ttc - g_jnk) / beta),  m_ttc = 0 m (code default)
           g = the same smooth box separation, w_j the same object weights, m_jnk = OBS(j, n + k) AND not behind (object
           centre at n + k >= -1.3 m along the UNPROJECTED heading h_n from the UNPROJECTED ego centre at n: the
           official is_agent_ahead / is_agent_behind use the rear axle at time_idx vs the object at time_idx +
           future_idx) AND NOT human_overlap_ttc(j, n, k) (the human's own PROJECTED box -- human speed / heading of its
           dense reference, same projection -- overlaps object j at n + k, exact SAT, touching counts).  delta = 0 is
           C_col.  At v_n = 0 the projection is exactly 0.  Not mirrored (AMENDMENT 4): already-collided tracks,
           red-light tokens, the 30 deg "ahead" cone / multiple-lane / intersection rule (approximated by not-behind),
           and the stopped-ego skip (speed < 5e-3 m/s; optional cfg.ttc_min_speed, default off).
  C_dac  drivable area: mean_n sum_corners valid * beta softplus((m_dac - SDF(corner)) / beta), m_dac = 0.2 m,
           bilinear SDF on the E grid (sdf.sample_sdf); corners outside the grid are excluded and counted (n_oog).
  L_prog progress, aligned with the official EP: P(tau) = relu(proj(c_40) - proj(c_0)) with c = ego CENTRE and proj =
           arc length of the nearest point of the metric-cache route centerline (= pdm_scorer._calculate_progress on
           the reference instead of the tracked states); P_pdm = pdm_progress_eff (PDM-Closed raw progress x its
           multiplicative metrics, score_trajectories); mode A:
               L_prog = relu(min(P(tau0), P_pdm) - P(tau1)) / max(P_pdm, 5.0)
           and 0 for drafts whose footprint enters UNKNOWN space (no progress reward can be claimed there).
           ep_surrogate() is the official EP rule applied to (P, P_pdm).
  C_cmf  comfort, penalty sum_terms mean_points relu(|x| - f lim)^2 / lim^2 (asymmetric for lon accel) with the PDMS
           v1 limits (lon accel [-4.05, 2.40], lon jerk 4.13, lat accel 4.89, yaw rate 0.95, yaw accel 1.93):
           keyframe terms of the output trajectory (0.5 s, t0 state included through v0):
               u_k = |P_{k+1} - P_k| / 0.5 (P_0 = origin), lon accel (u_0 - v0) / 0.25 and (u_k - u_{k-1}) / 0.5,
               lon jerk (a_k - a_{k-1}) / 0.5 (k >= 2), yaw rate w_k = dh_k / 0.5, lat accel u_k w_k,
               yaw accel (w_k - w_{k-1}) / 0.5;             factor f = CMF_FRAC_KF = 1.0 (see deviations)
           analytic terms of the correction (decoder spline, dense 0.1 s): da (accel offset, lon accel limits),
               jerk (lon jerk limit), v1^2 kappa_new (lat accel limit);    factor f = CMF_FRAC_AN = 0.9.
  C_mod  modification: mean_n |s1 - s0| / 5 m + mean_n |d| / 1 m.
  L_gate BCE(p_g, y), y = 1[official NC < 1 or DAC < 1 or DDC < 1] (TTC optional), pos_weight = min((1 - pi) / pi,
           10); GateHead detaches its input so that the gate never back-propagates into the trunk.

UNKNOWN (gt_future.unknown_space, torch twin unknown_footprint): an ego corner at n is UNKNOWN iff t_n > t_avail, or it
  is > 75 m from the GT ego at t_n, or (extension, default) > R - 10 m from the t0 origin.  Objects are only counted
  where OBS, so "unknown" is never "free": a draft whose footprint enters UNKNOWN space gets unknown = True and its
  progress term is dropped (L_prog = 0).  Measured rate on the dev M8 bank: report/refiner_T/surrogate_validation.json.

Deviations from IMPL_SPEC §3.7 (documented; interfaces as specified, extra options only)
  1. Comfort keyframe factor: the keyframe terms use f = 1.0 (penalty starts AT the official limit), the analytic
     terms keep f = 0.9.  With the spec's f = 0.9 the keyframe lon-accel term alone flags 1.92% (train, 23,820) /
     1.63% (dev, 7,930) of human trajectories -- above the spec's own acceptance (<= 1%): humans starting from
     stand-still accelerate at 2.2-2.5 m/s^2 for ~1 s (median v0 of those tokens 1.1 m/s), which the official LQR +
     whole-horizon Savitzky-Golay comfort check does not fail (official human comfort failures 0.19% on E's 9,000).
     Cubic / quartic least-squares speed fits (official-like smoothing) do not help (1.7-2.3%).  f = 1.0 gives the
     rates in tools/refiner/validate_surrogate.py comfort_human_rates() -> the M8 JSON (acceptance test in
     tools/refiner/tests/test_surrogate.py::test_comfort_human_acceptance).
  2. Comfort jerk terms that involve the t0 state are NOT used: (a_first - a0) / 0.125 flags 57% and
     (a_1 - a_first) / 0.375 flags 5.6% of humans (the metric-cache v0 and the logged positions disagree by up to
     ~0.5 m/s at t0; |a_first - a0| p99 1.8 m/s^2).  The t0 state enters through the first accel (u_0 - v0) / 0.25;
     a0 is accepted for interface compatibility but only used in keyframe_comfort(..., use_a0=True).
  3. Human-overlap mask uses the EXACT SAT gap of the human footprint (g_hard_human <= 0; touching counts, like
     shapely intersects / sf_common.sat_overlap), not the smooth g: the smooth g is below g_hard by up to T ln 4 and
     would also remove non-overlapping pairs the human passed within 7 cm.
  4. All costs average over n = 1..40 (not 0..40), see above.
  5. Progress uses a cropped centerline (vertices within [-30, +250] m of arc length around the t0 ego centre,
     centerline_from_metric_cache).  M8: equal to shapely on the full line (|diff| < 1e-6 m) on 32,265 of 32,269
     trajectories; 1 was a 38 m/s draft beyond the first 150 m window (fixed by the 250 m window, re-checked 6e-11 m);
     3 are constant-velocity drafts that leave a curving route, where the official full-line projection jumps to a
     part of the route 200-300 m ahead (official raw progress 207-301 m in 4 s; those drafts fail DAC, mult = 0, so
     the official EP is 0 either way) -- the crop keeps the local progress there.
  6. Not implemented (not in §3.7): the DDC surrogate (DDC stays in the gate label only).  TTC was added later
     (C_ttc above, AMENDMENT 4; weight 0 by default, so the §3.7 loss is unchanged).
  Weights / temperatures are the spec values; nothing was tuned on the M8 results.

M8 validation (tools/refiner/validate_surrogate.py -> report/refiner_T/surrogate_validation.json; 800 dev tokens / 212
held-out logs, 32,269 officially scored trajectories: 13-draft bank, E's student draft + rule corrections,
surrogate-guided and random mode-A corrections; pre-stated acceptance of IMPL_SPEC §3.7):
  A1 NC recall at m_col (bank, 497 NC failures)             0.948 [CP 0.924, 0.966]   PASS (>= 0.70)
  A2 human false alarm at m_col (800 human drafts)           0.0275 [0.017, 0.041]    FAIL (<= 0.02); 0 at margin 0
  A3 P(official fixed | surrogate fixed), guided pairs       0.510 [0.472, 0.548]     FAIL (>= 0.6)
  A3 fails on the flag's PRECISION at the 0.3 m margin, not on fixes: of 684 surrogate "fixes", 379 had an official
  NC failure (349 = 92% officially fixed), 305 were near misses the official NC passes (95 of them TTC failures).
  Guided corrections created 1 new NC failure in 2,303 officially passing drafts; sign(dC_col) agrees with the official
  change in 402/403.  After guided correction, 37 of the 82 remaining NC failures are invisible on the raw reference
  but 36 of those are flagged on the LQR-tracked states (the M7b gap).  All gradients finite; a 1e-4 step along
  -grad C_col lowers C_col for 98.1% of 3,953 sources (1.7% unchanged: mode-A clamp dead zone; 0.2% higher: boolean
  mask flips).  In mode A at z = 0 the longitudinal gradient is HALF the braking-side slope (decoder Q = A tanh z has
  A_UP = 2 for z >= 0, A_DEC = 4 for z < 0; autograd takes the z >= 0 side) -- sign correct, magnitude x0.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (DT, HALF_LEN, HALF_WID, JERK_MAX, LAT_ACC_MAX, LON_ACC_MAX, LON_ACC_MIN, LON_JERK_MAX, NK,
                       RA2C, T_POSE, YAW_ACC_MAX, YAW_RATE_MAX, dense_reference, safe_norm, unwrap_anchored)
from .sdf import ego_corners, sample_sdf

# ----------------------------------------------------------------------------------------------- constants
TEMP = 0.05              # smoothmax temperature [m]
M_COL = 0.3              # collision margin [m]
BETA = 0.1               # softplus scale [m]
BEHIND_M = -1.3          # object centre must be >= this far along the ego heading from the ego centre [m]
W_AGENT, W_STATIC = 1.0, 0.5
M_DAC = 0.2              # drivable-area margin [m]
PROG_THR = 5.0           # official progress_distance_threshold (default_scoring_parameters.yaml)
CMF_FRAC_KF = 1.0        # keyframe comfort threshold factor (deviation 1)
CMF_FRAC_AN = 0.9        # analytic comfort threshold factor (spec)
MOD_S, MOD_D = 5.0, 1.0  # C_mod scales [m]
N_FROM = 1               # first dense index used by the costs
REACH_M = 75.0           # annotation reach around the GT ego (gt_future.REACH_M)
RADIUS_MARGIN = 10.0     # UNKNOWN margin inside R (gt_future.RADIUS_MARGIN); None = spec definition
CL_BACK, CL_AHEAD = 30.0, 250.0   # centerline crop window around the t0 ego-centre projection [m]
GATE_POS_WEIGHT_CAP = 10.0
M_TTC = 0.0              # TTC margin [m] (AMENDMENT 4; chosen on train flags before run 3)
TTC_DELTAS = (0.3, 0.6, 0.9)   # TTC projection horizons [s] (official future_time_idcs 3, 6, 9 x 0.1 s)
NK_OBJ = 51              # dense object times 0 .. 5.0 s (objects for n + 10 delta, n <= 40)
DEFAULT_WEIGHTS = {"col": 1.0, "dac": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1, "gate": 0.5,   # architecture §M7
                   "ttc": 0.0}                                  # AMENDMENT 4: off by default (runs 1 / 2 unchanged)
OBS = 1

LN4 = math.log(4.0)


@dataclass
class SurrogateConfig:
    """Surrogate hyper-parameters (defaults = IMPL_SPEC §3.7 + the documented deviations)."""
    m_col: float = M_COL
    beta: float = BETA
    temp: float = TEMP
    bias_correct: bool = True
    behind_m: Optional[float] = BEHIND_M      # None disables the not-behind mask
    w_agent: float = W_AGENT
    w_static: float = W_STATIC
    use_human_mask: bool = True
    min_speed: Optional[float] = None         # optional: ignore (j, n) where the reference speed <= min_speed
    m_dac: float = M_DAC
    prog_thr: float = PROG_THR
    cmf_frac_kf: float = CMF_FRAC_KF
    cmf_frac_an: float = CMF_FRAC_AN
    mod_s: float = MOD_S
    mod_d: float = MOD_D
    n_from: int = N_FROM
    reach: float = REACH_M
    radius_margin: Optional[float] = RADIUS_MARGIN
    m_ttc: float = M_TTC                      # TTC margin (scalar, or a tensor broadcastable to [B, A, T', K])
    ttc_deltas: Tuple[float, ...] = TTC_DELTAS
    ttc_min_speed: Optional[float] = None     # optional: ignore (j, n, k) where v_n <= ttc_min_speed (official 5e-3)


# ----------------------------------------------------------------------------------------------- helpers
def _index(index, B: int, S: int, device) -> torch.Tensor:
    if index is None:
        if S == B:
            return torch.arange(B, device=device)
        if S == 1:
            return torch.zeros(B, dtype=torch.long, device=device)
        raise ValueError(f"{S} scenes for {B} drafts: pass index")
    index = torch.as_tensor(index, device=device).long().reshape(-1)
    if index.shape[0] != B:
        raise ValueError(f"index must be [{B}], got {tuple(index.shape)}")
    return index


def ego_centre(poses: torch.Tensor) -> torch.Tensor:
    """rear-axle poses [..., 3] -> ego box centres [..., 2]."""
    h = poses[..., 2]
    return torch.stack([poses[..., 0] + RA2C * torch.cos(h), poses[..., 1] + RA2C * torch.sin(h)], -1)


def ref_speed(dense: torch.Tensor) -> torch.Tensor:
    """speed of the reference at each dense point [B, T] (forward difference; last = previous)."""
    v = safe_norm(dense[:, 1:, :2] - dense[:, :-1, :2]) / DT
    return torch.cat([v, v[:, -1:]], 1)


# ----------------------------------------------------------------------------------------------- box separation
def box_gaps(ego_poses: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """SAT gaps along the 4 box axes between the ego box of rear-axle poses [..., 3] and object boxes [..., 5]
    (cx, cy, heading, L, W); shapes broadcast.  Returns [..., 4] (axes: ego long, ego lat, obj long, obj lat)."""
    he = ego_poses[..., 2]
    ce, se = torch.cos(he), torch.sin(he)
    ex = ego_poses[..., 0] + RA2C * ce
    ey = ego_poses[..., 1] + RA2C * se
    hj = boxes[..., 2]
    cj, sj = torch.cos(hj), torch.sin(hj)
    lh, wh = 0.5 * boxes[..., 3], 0.5 * boxes[..., 4]
    dx, dy = boxes[..., 0] - ex, boxes[..., 1] - ey
    cd = torch.abs(ce * cj + se * sj)          # |cos(hj - he)|
    sd = torch.abs(ce * sj - se * cj)          # |sin(hj - he)|
    g_e1 = torch.abs(dx * ce + dy * se) - HALF_LEN - (lh * cd + wh * sd)
    g_e2 = torch.abs(-dx * se + dy * ce) - HALF_WID - (lh * sd + wh * cd)
    g_f1 = torch.abs(dx * cj + dy * sj) - lh - (HALF_LEN * cd + HALF_WID * sd)
    g_f2 = torch.abs(-dx * sj + dy * cj) - wh - (HALF_LEN * sd + HALF_WID * cd)
    return torch.stack([g_e1, g_e2, g_f1, g_f2], -1)


def box_separation(ego_poses: torch.Tensor, boxes: torch.Tensor, temp: float = TEMP, bias_correct: bool = True):
    """(g_smooth, g_hard) [...]: smooth signed separation (logsumexp over the 4 axis gaps, minus temp ln 4 if
    bias_correct) and the exact SAT max gap (no gradient needed; returned detached)."""
    gaps = box_gaps(ego_poses, boxes)
    g = temp * torch.logsumexp(gaps / temp, -1)
    if bias_correct:
        g = g - temp * LN4
    return g, gaps.detach().max(-1).values


def human_overlap_mask(human_dense: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """[S, T, 3] human reference, [S, A, T, 5] boxes -> bool [S, A, T]: the human footprint overlaps (exact SAT,
    touching counts) object j at time n."""
    with torch.no_grad():
        return box_gaps(human_dense[:, None], boxes).max(-1).values <= 0.0


# ----------------------------------------------------------------------------------------------- scene batch
@dataclass
class SceneBatch:
    """Per-token tensors (leading S).  Missing parts may be None (the corresponding term is then skipped).

    boxes [S, A, 41, 5], obs [S, A, 41] bool, is_agent [S, A] bool, human_overlap [S, A, 41] bool,
    track (list of per-scene numpy <U16 arrays, for diagnostics only), sdf [S, 320, 256], centerline [S, L, 2],
    cl_valid [S, L] bool, p_pdm [S], v0 [S], a0 [S], gt_ego [S, 41, 2], t_avail [S], R [S].
    TTC (C_ttc): boxes_ttc [S, A, 51, 5] / obs_ttc [S, A, 51] bool = the same objects at t = 0.1 m, m = 0..50 (0..5 s;
    None -> ttc_cost uses boxes / obs, i.e. only object times <= 4 s, the rest excluded), human_dense [S, 41, 3] = the
    human dense reference (its projected boxes give the TTC human-overlap mask; None -> no TTC human mask)."""
    boxes: Optional[torch.Tensor] = None
    obs: Optional[torch.Tensor] = None
    is_agent: Optional[torch.Tensor] = None
    human_overlap: Optional[torch.Tensor] = None
    sdf: Optional[torch.Tensor] = None
    centerline: Optional[torch.Tensor] = None
    cl_valid: Optional[torch.Tensor] = None
    p_pdm: Optional[torch.Tensor] = None
    v0: Optional[torch.Tensor] = None
    a0: Optional[torch.Tensor] = None
    gt_ego: Optional[torch.Tensor] = None
    t_avail: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None
    track: Optional[list] = None
    boxes_ttc: Optional[torch.Tensor] = None
    obs_ttc: Optional[torch.Tensor] = None
    human_dense: Optional[torch.Tensor] = None

    @property
    def n(self) -> int:
        for f_ in fields(self):
            v = getattr(self, f_.name)
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        return 0

    def to(self, device=None, dtype=None) -> "SceneBatch":
        out = {}
        for f_ in fields(self):
            v = getattr(self, f_.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device) if device is not None else v
                if dtype is not None and v.is_floating_point() and f_.name != "sdf":
                    v = v.to(dtype)
            out[f_.name] = v
        return SceneBatch(**out)


def scene_from_numpy(objs: Optional[Dict] = None, sdf=None, centerline=None, human_traj=None, p_pdm=None, v0=None,
                     a0=None, dtype=torch.float64) -> Dict:
    """One token's scene dict (numpy inputs) -> dict of torch tensors (no batch axis) for collate_scenes.
    objs: gt_future.load_objects dict; sdf: [320, 256]; centerline: [L, 2] N frame; human_traj: [8, 3]."""
    from . import gt_future as G
    out: Dict = {}
    t = np.arange(NK) * DT
    if objs is not None:
        boxes, state = G.query(objs, t, dtype=np.float64)
        out["boxes"] = torch.as_tensor(boxes, dtype=dtype)
        out["obs"] = torch.as_tensor(state == OBS)
        out["is_agent"] = torch.as_tensor(np.asarray(objs["meta"])[:, G.MT_AGENT] > 0)
        out["track"] = np.asarray(objs["track"])
        out["gt_ego"] = torch.as_tensor(G.gt_ego_xy(objs, t), dtype=dtype)
        out["t_avail"] = torch.as_tensor(G.t_avail(objs), dtype=dtype)
        out["R"] = torch.as_tensor(float(objs["R"]), dtype=dtype)
        bt, st = G.query(objs, np.arange(NK_OBJ) * DT, dtype=np.float64)     # TTC: object times 0..5 s
        out["boxes_ttc"] = torch.as_tensor(bt, dtype=dtype)
        out["obs_ttc"] = torch.as_tensor(st == OBS)
        if human_traj is not None:
            hd = dense_reference(torch.as_tensor(np.asarray(human_traj, np.float32)).to(dtype)[None])
            out["human_overlap"] = human_overlap_mask(hd, out["boxes"][None])[0]
            out["human_dense"] = hd[0]
    if sdf is not None:
        out["sdf"] = torch.as_tensor(np.asarray(sdf))
    if centerline is not None:
        out["centerline"] = torch.as_tensor(np.asarray(centerline), dtype=dtype)
    for k, v in (("p_pdm", p_pdm), ("v0", v0), ("a0", a0)):
        if v is not None:
            out[k] = torch.as_tensor(float(v), dtype=dtype)
    return out


def collate_scenes(scenes: Sequence[Dict], a_max: Optional[int] = None, l_max: Optional[int] = None) -> SceneBatch:
    """Pad a list of scene_from_numpy dicts into a SceneBatch (objects padded with obs = False, centerlines with
    cl_valid = False)."""
    S = len(scenes)
    sb = {}
    if all("boxes" in s for s in scenes):
        A = max([s["boxes"].shape[0] for s in scenes] + [1]) if a_max is None else int(a_max)
        ref = scenes[0]["boxes"]
        boxes = ref.new_zeros(S, A, NK, 5)
        obs = torch.zeros(S, A, NK, dtype=torch.bool)
        agent = torch.zeros(S, A, dtype=torch.bool)
        hov = torch.zeros(S, A, NK, dtype=torch.bool)
        for i, s in enumerate(scenes):
            m = min(A, s["boxes"].shape[0])
            boxes[i, :m], obs[i, :m], agent[i, :m] = s["boxes"][:m], s["obs"][:m], s["is_agent"][:m]
            if "human_overlap" in s:
                hov[i, :m] = s["human_overlap"][:m]
        sb.update(boxes=boxes, obs=obs, is_agent=agent, human_overlap=hov, track=[s.get("track") for s in scenes],
                  gt_ego=torch.stack([s["gt_ego"] for s in scenes]), t_avail=torch.stack([s["t_avail"] for s in scenes]),
                  R=torch.stack([s["R"] for s in scenes]))
        if all("boxes_ttc" in s for s in scenes):
            nt = scenes[0]["boxes_ttc"].shape[1]
            bt = ref.new_zeros(S, A, nt, 5)
            ot = torch.zeros(S, A, nt, dtype=torch.bool)
            for i, s in enumerate(scenes):
                m = min(A, s["boxes_ttc"].shape[0])
                bt[i, :m], ot[i, :m] = s["boxes_ttc"][:m], s["obs_ttc"][:m]
            sb.update(boxes_ttc=bt, obs_ttc=ot)
        if all("human_dense" in s for s in scenes):
            sb["human_dense"] = torch.stack([s["human_dense"] for s in scenes])
    if all("sdf" in s for s in scenes):
        sb["sdf"] = torch.stack([s["sdf"] for s in scenes])
    if all("centerline" in s for s in scenes):
        L = max(s["centerline"].shape[0] for s in scenes) if l_max is None else int(l_max)
        cl = scenes[0]["centerline"].new_zeros(S, L, 2)
        cv = torch.zeros(S, L, dtype=torch.bool)
        for i, s in enumerate(scenes):
            m = min(L, s["centerline"].shape[0])
            cl[i, :m], cv[i, :m] = s["centerline"][:m], True
        sb.update(centerline=cl, cl_valid=cv)
    for k in ("p_pdm", "v0", "a0"):
        if all(k in s for s in scenes):
            sb[k] = torch.stack([s[k] for s in scenes])
    return SceneBatch(**sb)


def centerline_from_metric_cache(mc, back: float = CL_BACK, ahead: float = CL_AHEAD) -> np.ndarray:
    """Route centerline (metric_cache.centerline, PDMPath, global) -> float64 [L, 2] vertices in N, cropped to the
    arc-length window [s_c - back, s_c + ahead] around the projection s_c of the t0 ego centre (one extra vertex on
    each side).  Arc-length differences along the crop equal those along the full line wherever the nearest point
    lies inside the window."""
    from shapely.geometry import Point
    ra = mc.ego_state.rear_axle
    arr = np.asarray(mc.centerline._states_se2_array[:, :2], np.float64)
    x0, y0, h0 = float(ra.x), float(ra.y), float(ra.heading)
    c, s = math.cos(h0), math.sin(h0)
    pts = np.stack([c * (arr[:, 0] - x0) + s * (arr[:, 1] - y0), -s * (arr[:, 0] - x0) + c * (arr[:, 1] - y0)], 1)
    prog = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arr, axis=0).T))])
    centre = Point(x0 + RA2C * c, y0 + RA2C * s)
    s_c = float(mc.centerline.linestring.project(centre))
    i0 = max(0, int(np.searchsorted(prog, s_c - back, side="left")) - 1)
    i1 = min(len(prog), int(np.searchsorted(prog, s_c + ahead, side="right")) + 1)
    return pts[i0:i1]


# ----------------------------------------------------------------------------------------------- collision
def collision_cost(dense: torch.Tensor, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig(),
                   details: bool = True) -> Dict[str, torch.Tensor]:
    """C_col of drafts with dense references [B, 41, 3] (see module docstring).

    Returns dict: cost [B]; if details: viol [B] = max over counted (j, n) of (m_col - g) (-inf if none; > 0 <=> the
    draft is flagged at the training margin), gmin [B] (smooth g), gmin_hard [B] (exact SAT gap), hard_overlap [B]
    (some counted pair with g_hard <= 0), first_n [B] (first dense index with a counted g < m_col, -1 if none),
    first_obj [B] (object with the smallest g at first_n, -1), n_human_masked [B] (pairs that would be counted and
    within the margin but are removed by the human mask), g [B, A, T'], mask [B, A, T'] (T' = 41 - n_from)."""
    B = dense.shape[0]
    dev, dt = dense.device, dense.dtype
    if scene.boxes is None or scene.boxes.shape[1] == 0:
        z = dense.new_zeros(B)
        out = {"cost": z}
        if details:
            neg = torch.full((B,), -math.inf, dtype=dt, device=dev)
            out.update(viol=neg, gmin=-neg, gmin_hard=-neg, hard_overlap=torch.zeros(B, dtype=torch.bool, device=dev),
                       first_n=torch.full((B,), -1, device=dev), first_obj=torch.full((B,), -1, device=dev),
                       n_human_masked=torch.zeros(B, dtype=torch.long, device=dev))
        return out
    idx = _index(index, B, scene.boxes.shape[0], dev)
    n0 = cfg.n_from
    boxes = scene.boxes.to(device=dev, dtype=dt)[idx][:, :, n0:]            # [B, A, T', 5]
    ego = dense[:, None, n0:, :]                                            # [B, 1, T', 3]
    g, g_hard = box_separation(ego, boxes, cfg.temp, cfg.bias_correct)       # [B, A, T']
    mask = scene.obs.to(dev)[idx][:, :, n0:].bool()
    with torch.no_grad():
        if cfg.behind_m is not None:
            he = dense[:, None, n0:, 2]
            c = ego_centre(dense[:, None, n0:, :])
            lon = (boxes[..., 0] - c[..., 0]) * torch.cos(he) + (boxes[..., 1] - c[..., 1]) * torch.sin(he)
            mask = mask & (lon >= cfg.behind_m)
        if cfg.min_speed is not None:
            mask = mask & (ref_speed(dense)[:, None, n0:] > cfg.min_speed)
        pre_human = mask
        if cfg.use_human_mask and scene.human_overlap is not None:
            mask = mask & ~scene.human_overlap.to(dev)[idx][:, :, n0:].bool()
    w = torch.where(scene.is_agent.to(dev)[idx].bool(), torch.full((), cfg.w_agent, dtype=dt, device=dev),
                    torch.full((), cfg.w_static, dtype=dt, device=dev))     # [B, A]
    pen = cfg.beta * F.softplus((cfg.m_col - g) / cfg.beta)
    pen = torch.where(mask, pen, torch.zeros_like(pen))
    cost = (w[:, :, None] * pen).sum(1).mean(1)
    out = {"cost": cost}
    if details:
        with torch.no_grad():
            inf = torch.full((), math.inf, dtype=dt, device=dev)
            gm = torch.where(mask, g.detach(), inf)
            ghm = torch.where(mask, g_hard, inf)
            gmin = gm.flatten(1).min(1).values
            out["gmin"] = gmin
            out["gmin_hard"] = ghm.flatten(1).min(1).values
            out["viol"] = cfg.m_col - gmin
            out["hard_overlap"] = (ghm <= 0).flatten(1).any(1)
            vt = (gm < cfg.m_col).any(1)                                   # [B, T']
            has = vt.any(1)
            fn = torch.where(has, vt.float().argmax(1), torch.zeros_like(has, dtype=torch.long))
            gat = gm[torch.arange(B, device=dev), :, fn]                    # [B, A]
            out["first_n"] = torch.where(has, fn + n0, torch.full_like(fn, -1))
            out["first_obj"] = torch.where(has, gat.argmin(1), torch.full_like(fn, -1))
            removed = pre_human & ~mask & (g.detach() < cfg.m_col)
            out["n_human_masked"] = removed.flatten(1).sum(1)
            out["g"], out["mask"] = g, mask
    return out


# ----------------------------------------------------------------------------------------------- time to collision
def ttc_offsets(deltas: Sequence[float]) -> Tuple[int, ...]:
    """projection horizons [s] -> dense index offsets 10 delta (must be multiples of DT)."""
    out = []
    for d in deltas:
        k = int(round(float(d) / DT))
        if abs(k * DT - float(d)) > 1e-9 or k < 0:
            raise ValueError(f"TTC horizon {d} s is not a non-negative multiple of {DT} s")
        out.append(k)
    return tuple(out)


def project_poses(dense: torch.Tensor, deltas: Sequence[float]) -> torch.Tensor:
    """dense references [B, T, 3] -> constant-velocity projected rear-axle poses [B, T, K, 3]: pose_n translated by
    v_n delta_k along h_n (v_n = ref_speed, h_n = dense heading; heading unchanged).  v_n = 0 -> exactly pose_n."""
    v = ref_speed(dense)                                                   # [B, T]
    h = dense[..., 2]
    d = torch.as_tensor([float(x) for x in deltas], dtype=dense.dtype, device=dense.device)
    s = v[..., None] * d                                                   # [B, T, K]
    x = dense[..., 0, None] + s * torch.cos(h)[..., None]
    y = dense[..., 1, None] + s * torch.sin(h)[..., None]
    return torch.stack([x, y, h[..., None].expand_as(x)], -1)


def _ttc_objects(scene: SceneBatch):
    """(boxes [S, A, To, 5], obs [S, A, To]) for C_ttc: the 0..5 s objects if present, else the 41-time ones."""
    if scene.boxes_ttc is not None:
        return scene.boxes_ttc, scene.obs_ttc
    return scene.boxes, scene.obs


def _ttc_time_index(n0: int, T: int, offs: Sequence[int], To: int, device):
    """object time index [T', K] = n + off_k (clamped only for gathering) and in-range mask [T', K] (n + off_k < To)."""
    n = torch.arange(n0, T, device=device)
    tt = n[:, None] + torch.as_tensor(offs, device=device, dtype=torch.long)[None]
    return tt.clamp(max=To - 1), tt < To


@torch.no_grad()
def human_ttc_overlap_mask(human_dense: torch.Tensor, boxes: torch.Tensor, obs: Optional[torch.Tensor] = None,
                           deltas: Sequence[float] = TTC_DELTAS, n_from: int = N_FROM) -> torch.Tensor:
    """[S, T, 3] human dense reference, [S, A, To, 5] object boxes over To dense times -> bool [S, A, T - n_from, K]:
    the human's PROJECTED box at n (projection delta_k, project_poses) overlaps object j at n + 10 delta_k (exact SAT,
    touching counts; float64).  Pairs with n + 10 delta_k >= To are False; obs (optional) is not applied here."""
    T, To = human_dense.shape[1], boxes.shape[2]
    tix, inr = _ttc_time_index(n_from, T, ttc_offsets(deltas), To, boxes.device)
    hp = project_poses(human_dense.to(torch.float64), deltas)[:, None, n_from:]        # [S, 1, T', K, 3]
    bx = boxes.to(torch.float64)[:, :, tix]                                          # [S, A, T', K, 5]
    return (box_gaps(hp, bx).max(-1).values <= 0.0) & inr


def ttc_cost(dense: torch.Tensor, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig(),
             details: bool = False, flag_margin=None) -> Dict[str, torch.Tensor]:
    """C_ttc of drafts with dense references [B, 41, 3] (module docstring; AMENDMENT 4 (1)).

    Returns dict: cost [B]; if details: viol [B] = max over counted (j, n, k) of (m_ttc - g) (-inf if none), gmin [B]
    (smooth g), gmin_hard [B] (exact SAT gap), hard_overlap [B], flag [B] = gmin < flag_margin (default cfg.m_ttc; the
    AMENDMENT 4 (2) projected flag), first_n [B] (first dense index with a counted g < m_ttc, -1 if none), first_k [B]
    (horizon index of the smallest g at first_n, -1), first_obj [B] (object of the smallest g at first_n, -1),
    n_human_masked [B], n_pairs [B] (counted (j, n, k) pairs), g / mask [B, A, T', K] (T' = 41 - n_from)."""
    B = dense.shape[0]
    dev, dt = dense.device, dense.dtype
    boxes_all, obs_all = _ttc_objects(scene)
    offs = ttc_offsets(cfg.ttc_deltas)
    margin = cfg.m_ttc if flag_margin is None else flag_margin
    if boxes_all is None or boxes_all.shape[1] == 0 or len(offs) == 0:
        out = {"cost": dense.new_zeros(B)}
        if details:
            neg = torch.full((B,), -math.inf, dtype=dt, device=dev)
            m1 = torch.full((B,), -1, device=dev)
            out.update(viol=neg, gmin=-neg, gmin_hard=-neg, hard_overlap=torch.zeros(B, dtype=torch.bool, device=dev),
                       flag=torch.zeros(B, dtype=torch.bool, device=dev), first_n=m1, first_k=m1.clone(),
                       first_obj=m1.clone(), n_human_masked=torch.zeros(B, dtype=torch.long, device=dev),
                       n_pairs=torch.zeros(B, dtype=torch.long, device=dev))
        return out
    idx = _index(index, B, boxes_all.shape[0], dev)
    n0, T, To = cfg.n_from, dense.shape[1], boxes_all.shape[2]
    tix, inr = _ttc_time_index(n0, T, offs, To, dev)                        # [T', K]
    boxes = boxes_all.to(device=dev, dtype=dt)[idx][:, :, tix]              # [B, A, T', K, 5]
    ego = project_poses(dense, cfg.ttc_deltas)[:, None, n0:]                # [B, 1, T', K, 3]
    g, g_hard = box_separation(ego, boxes, cfg.temp, cfg.bias_correct)       # [B, A, T', K]
    mask = obs_all.to(dev)[idx][:, :, tix].bool() & inr
    with torch.no_grad():
        if cfg.behind_m is not None:                                        # unprojected pose at n vs object at n + k
            he = dense[:, None, n0:, None, 2]
            c = ego_centre(dense[:, None, n0:, None, :])
            lon = (boxes[..., 0] - c[..., 0]) * torch.cos(he) + (boxes[..., 1] - c[..., 1]) * torch.sin(he)
            mask = mask & (lon >= cfg.behind_m)
        if cfg.ttc_min_speed is not None:
            mask = mask & (ref_speed(dense)[:, None, n0:, None] > cfg.ttc_min_speed)
        pre_human = mask
        if cfg.use_human_mask and scene.human_dense is not None:
            hm = human_ttc_overlap_mask(scene.human_dense.to(dev), boxes_all.to(dev), None, cfg.ttc_deltas, n0)
            mask = mask & ~hm[idx]
    w = torch.where(scene.is_agent.to(dev)[idx].bool(), torch.full((), cfg.w_agent, dtype=dt, device=dev),
                    torch.full((), cfg.w_static, dtype=dt, device=dev))     # [B, A]
    pen = cfg.beta * F.softplus((cfg.m_ttc - g) / cfg.beta)
    pen = torch.where(mask, pen, torch.zeros_like(pen))
    cost = (w[:, :, None, None] * pen).sum((1, 3)).mean(1)
    out = {"cost": cost}
    if details:
        with torch.no_grad():
            inf = torch.full((), math.inf, dtype=dt, device=dev)
            gm = torch.where(mask, g.detach(), inf)
            ghm = torch.where(mask, g_hard, inf)
            gmin = gm.flatten(1).min(1).values
            out.update(gmin=gmin, gmin_hard=ghm.flatten(1).min(1).values, viol=cfg.m_ttc - gmin,
                       hard_overlap=(ghm <= 0).flatten(1).any(1), flag=gmin < margin)
            vt = (gm < cfg.m_ttc).any(1).any(-1)                            # [B, T']
            has = vt.any(1)
            fn = torch.where(has, vt.float().argmax(1), torch.zeros_like(has, dtype=torch.long))
            gat = gm[torch.arange(B, device=dev), :, fn]                    # [B, A, K]
            am = gat.flatten(1).argmin(1)
            K = gat.shape[-1]
            m1 = torch.full_like(fn, -1)
            out["first_n"] = torch.where(has, fn + n0, m1)
            out["first_obj"] = torch.where(has, am // K, m1)
            out["first_k"] = torch.where(has, am % K, m1)
            removed = pre_human & ~mask & (g.detach() < cfg.m_ttc)
            out["n_human_masked"] = removed.flatten(1).sum(1)
            out["n_pairs"] = mask.flatten(1).sum(1)
            out["g"], out["mask"] = g, mask
    return out


# ----------------------------------------------------------------------------------------------- drivable area
def dac_cost(dense: torch.Tensor, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig(),
             details: bool = True) -> Dict[str, torch.Tensor]:
    """C_dac of drafts [B, 41, 3]: mean_n sum_corners valid * beta softplus((m_dac - sdf) / beta).
    details: sdf_min [B] (min corner SDF over valid corners, +inf if none), viol [B] = m_dac - sdf_min, first_n [B]
    (first dense index with a valid corner below m_dac), n_oog [B] (corner samples outside the grid)."""
    B = dense.shape[0]
    if scene.sdf is None:
        return {"cost": dense.new_zeros(B)}
    idx = _index(index, B, scene.sdf.shape[0], dense.device)
    n0 = cfg.n_from
    val, valid = sample_sdf(scene.sdf, ego_corners(dense[:, n0:]), idx)     # [B, T', 4]
    pen = cfg.beta * F.softplus((cfg.m_dac - val) / cfg.beta)
    pen = torch.where(valid, pen, torch.zeros_like(pen))
    out = {"cost": pen.sum(-1).mean(1)}
    if details:
        with torch.no_grad():
            inf = torch.full((), math.inf, dtype=val.dtype, device=val.device)
            vm = torch.where(valid, val.detach(), inf)
            smin = vm.flatten(1).min(1).values
            vt = (vm < cfg.m_dac).any(-1)
            has = vt.any(1)
            out.update(sdf_min=smin, viol=cfg.m_dac - smin, n_oog=(~valid).flatten(1).sum(1),
                       first_n=torch.where(has, vt.float().argmax(1) + n0, torch.full((B,), -1, device=dense.device)),
                       hard_out=(vm < 0).flatten(1).any(1))
    return out


# ----------------------------------------------------------------------------------------------- progress
def project_on_polyline(p: torch.Tensor, cl: torch.Tensor, cl_valid: Optional[torch.Tensor] = None):
    """Arc length of the nearest point of polylines cl [B, L, 2] (valid vertices cl_valid [B, L]) for points
    p [B, M, 2] (same semantics as shapely LineString.project: nearest segment, first on ties).  Differentiable in p
    (d s / d p = unit segment direction inside a segment, 0 at a clamped end).  Returns (s [B, M], dist [B, M]);
    s = 0, dist = inf for a polyline with no valid segment."""
    a, b = cl[:, :-1], cl[:, 1:]                                           # [B, L-1, 2]
    seg = b - a
    ok = torch.ones(seg.shape[:2], dtype=torch.bool, device=cl.device) if cl_valid is None else \
        (cl_valid[:, :-1] & cl_valid[:, 1:])
    len2 = (seg * seg).sum(-1)
    seglen = torch.sqrt(len2)
    seglen = torch.where(ok, seglen, torch.zeros_like(seglen))
    S = torch.cat([seglen.new_zeros(seglen.shape[0], 1), torch.cumsum(seglen, 1)], 1)   # [B, L]
    rel = p[:, :, None, :] - a[:, None]                                    # [B, M, L-1, 2]
    pos = len2 > 0
    tt = torch.where(pos[:, None], (rel * seg[:, None]).sum(-1) / torch.where(pos, len2, torch.ones_like(len2))[:, None],
                     torch.zeros_like(rel[..., 0]))
    tt = tt.clamp(0.0, 1.0)
    q = a[:, None] + tt[..., None] * seg[:, None]
    d2 = ((p[:, :, None, :] - q) ** 2).sum(-1)
    d2 = torch.where(ok[:, None], d2, torch.full_like(d2, math.inf))
    i = d2.detach().argmin(-1)                                             # [B, M]
    t_i = torch.gather(tt, 2, i[..., None])[..., 0]
    s = torch.gather(S[:, None, :-1].expand(-1, p.shape[1], -1), 2, i[..., None])[..., 0] + \
        t_i * torch.gather(seglen[:, None].expand(-1, p.shape[1], -1), 2, i[..., None])[..., 0]
    dist = torch.gather(d2, 2, i[..., None])[..., 0].detach().sqrt()
    any_ok = ok.any(1)[:, None]
    s = torch.where(any_ok, s, torch.zeros_like(s))
    return s, dist


def progress(dense: torch.Tensor, scene: SceneBatch, index=None) -> torch.Tensor:
    """Raw progress P [B] of drafts [B, 41, 3]: relu(proj(centre_40) - proj(centre_0)) along the scene centerline
    (official pdm_scorer._calculate_progress applied to the reference)."""
    B = dense.shape[0]
    idx = _index(index, B, scene.centerline.shape[0], dense.device)
    cl = scene.centerline.to(device=dense.device, dtype=dense.dtype)[idx]
    cv = None if scene.cl_valid is None else scene.cl_valid.to(dense.device)[idx]
    c = ego_centre(dense[:, [0, -1]])
    s, _ = project_on_polyline(c, cl, cv)
    return torch.relu(s[:, 1] - s[:, 0])


def ep_surrogate(P: torch.Tensor, p_pdm: torch.Tensor, thr: float = PROG_THR) -> torch.Tensor:
    """Official EP rule for a draft with multiplicative metrics = 1: m = max(P_pdm, P); EP = P / m if m > thr else 1."""
    m = torch.maximum(p_pdm, P)
    big = m > thr
    return torch.where(big, P / torch.where(big, m, torch.ones_like(m)), torch.ones_like(P))


def progress_loss(P1: torch.Tensor, P0: torch.Tensor, p_pdm: torch.Tensor, thr: float = PROG_THR,
                  drop: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mode A: relu(min(P0, P_pdm) - P1) / max(P_pdm, thr); 0 where drop (UNKNOWN footprint)."""
    L = torch.relu(torch.minimum(P0, p_pdm) - P1) / torch.clamp(p_pdm, min=thr)
    if drop is not None:
        L = torch.where(drop.bool(), torch.zeros_like(L), L)
    return L


# ----------------------------------------------------------------------------------------------- UNKNOWN
@torch.no_grad()
def unknown_footprint(dense: torch.Tensor, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig()):
    """bool [B]: some ego corner at n = n_from..40 is in UNKNOWN space (gt_future.unknown_space rule)."""
    B = dense.shape[0]
    if scene.gt_ego is None:
        return torch.zeros(B, dtype=torch.bool, device=dense.device)
    idx = _index(index, B, scene.gt_ego.shape[0], dense.device)
    n0 = cfg.n_from
    cor = ego_corners(dense[:, n0:])                                       # [B, T', 4, 2]
    t = torch.arange(n0, NK, device=dense.device, dtype=dense.dtype) * DT
    ta = scene.t_avail.to(device=dense.device, dtype=dense.dtype)[idx]
    unk = (t[None] > ta[:, None] + 1e-6)[..., None].expand(-1, -1, 4)
    ge = scene.gt_ego.to(device=dense.device, dtype=dense.dtype)[idx][:, n0:, None, :]
    unk = unk | (torch.linalg.norm(cor - ge, dim=-1) > cfg.reach)
    if cfg.radius_margin is not None:
        R = scene.R.to(device=dense.device, dtype=dense.dtype)[idx]
        unk = unk | (torch.linalg.norm(cor, dim=-1) > (R - cfg.radius_margin)[:, None, None])
    return unk.flatten(1).any(1)


# ----------------------------------------------------------------------------------------------- comfort
def keyframe_comfort(traj: torch.Tensor, v0: Optional[torch.Tensor] = None, a0: Optional[torch.Tensor] = None,
                     use_a0: bool = False) -> Dict[str, torch.Tensor]:
    """0.5 s keyframe comfort quantities of trajectories [B, 8, 3] (+ ego v0 [B]; a0 only with use_a0):
    lon_acc [B, 8] ((u_0 - v0) / 0.25, then (u_k - u_{k-1}) / 0.5; 7 values without v0), lon_jerk [B, 6]
    ((a_k - a_{k-1}) / 0.5 over the position-only accelerations; + (a_first - a0) / 0.125 first if use_a0),
    lat_acc [B, 8], yaw_rate [B, 8], yaw_acc [B, 7]."""
    B = traj.shape[0]
    P = torch.cat([traj.new_zeros(B, 1, 3), traj], 1)
    u = safe_norm(P[:, 1:, :2] - P[:, :-1, :2]) / T_POSE                   # [B, 8]
    a_k = (u[:, 1:] - u[:, :-1]) / T_POSE                                  # [B, 7] at t = 0.5 k
    acc = a_k
    jerk = (a_k[:, 1:] - a_k[:, :-1]) / T_POSE                             # [B, 6]
    if v0 is not None:
        v0 = torch.as_tensor(v0, dtype=traj.dtype, device=traj.device).reshape(B)
        a_first = (u[:, 0] - v0) / (T_POSE / 2)
        acc = torch.cat([a_first[:, None], a_k], 1)
        if use_a0 and a0 is not None:
            a0 = torch.as_tensor(a0, dtype=traj.dtype, device=traj.device).reshape(B)
            jerk = torch.cat([((a_first - a0) / (T_POSE / 4))[:, None], jerk], 1)
    h = unwrap_anchored(P[..., 2])
    w = (h[:, 1:] - h[:, :-1]) / T_POSE
    return {"lon_acc": acc, "lon_jerk": jerk, "lat_acc": u * w, "yaw_rate": w, "yaw_acc": (w[:, 1:] - w[:, :-1]) / T_POSE}


def analytic_comfort(dec: Dict[str, torch.Tensor], n_from: int = N_FROM) -> Dict[str, torch.Tensor]:
    """Correction-spline comfort quantities from decoder.decode(): da, jerk, lat_acc = v1^2 kappa_new [B, 41 - n_from]."""
    return {"lon_acc": dec["da"][:, n_from:], "lon_jerk": dec["jerk"][:, n_from:],
            "lat_acc": dec["v"][:, n_from:] ** 2 * dec["kappa"][:, n_from:]}


_LIMITS = {"lon_acc": (LON_ACC_MIN, LON_ACC_MAX), "lon_jerk": LON_JERK_MAX, "lat_acc": LAT_ACC_MAX,
           "yaw_rate": YAW_RATE_MAX, "yaw_acc": YAW_ACC_MAX, "jerk": JERK_MAX}


def comfort_penalty(terms: Dict[str, torch.Tensor], frac: float) -> Dict[str, torch.Tensor]:
    """sum_terms mean_points relu(|x| - frac lim)^2 / lim^2 (lon_acc: asymmetric).  Returns dict(cost [B],
    ratio [B] = max over terms/points of |x| / lim (signed side for lon_acc), flag [B] = some point beyond frac lim)."""
    cost, ratio = None, None
    for k, x in terms.items():
        lim = _LIMITS[k]
        if isinstance(lim, tuple):
            lo, hi = -lim[0], lim[1]
            c = (torch.relu(x - frac * hi) ** 2 / hi ** 2 + torch.relu(-x - frac * lo) ** 2 / lo ** 2).mean(1)
            r = torch.maximum(x / hi, -x / lo).amax(1)
        else:
            c = (torch.relu(x.abs() - frac * lim) ** 2 / lim ** 2).mean(1)
            r = x.abs().amax(1) / lim
        cost = c if cost is None else cost + c
        ratio = r if ratio is None else torch.maximum(ratio, r)
    return {"cost": cost, "ratio": ratio.detach(), "flag": (ratio > frac).detach()}


def comfort_cost(traj: torch.Tensor, v0=None, a0=None, dec: Optional[Dict] = None,
                 cfg: SurrogateConfig = SurrogateConfig()) -> Dict[str, torch.Tensor]:
    """C_cmf = keyframe penalty (factor cfg.cmf_frac_kf) + analytic penalty (cfg.cmf_frac_an, only with a decoder
    output ``dec``).  Returns cost [B], kf_flag, kf_ratio, an_flag, an_ratio (flags False without dec)."""
    kf = comfort_penalty(keyframe_comfort(traj, v0, a0), cfg.cmf_frac_kf)
    out = {"cost": kf["cost"], "kf_flag": kf["flag"], "kf_ratio": kf["ratio"]}
    if dec is not None:
        an = comfort_penalty(analytic_comfort(dec, cfg.n_from), cfg.cmf_frac_an)
        out.update(cost=kf["cost"] + an["cost"], an_flag=an["flag"], an_ratio=an["ratio"])
    else:
        out.update(an_flag=torch.zeros_like(kf["flag"]), an_ratio=torch.zeros_like(kf["ratio"]))
    return out


# ----------------------------------------------------------------------------------------------- modification
def modification_cost(dec: Dict[str, torch.Tensor], cfg: SurrogateConfig = SurrogateConfig()) -> torch.Tensor:
    """C_mod = mean_n |s1 - s0| / mod_s + mean_n |d| / mod_d over n = n_from..40."""
    n0 = cfg.n_from
    return (dec["s"][:, n0:] - dec["s0"][:, n0:]).abs().mean(1) / cfg.mod_s + dec["d"][:, n0:].abs().mean(1) / cfg.mod_d


# ----------------------------------------------------------------------------------------------- gate
def gate_labels(nc, dac, ddc, ttc=None, use_ttc: bool = False) -> torch.Tensor:
    """y = 1[NC < 1 or DAC < 1 or DDC < 1 (or TTC < 1 if use_ttc)] as float."""
    y = (torch.as_tensor(nc) < 1) | (torch.as_tensor(dac) < 1) | (torch.as_tensor(ddc) < 1)
    if use_ttc:
        if ttc is None:
            raise ValueError("use_ttc needs ttc")
        y = y | (torch.as_tensor(ttc) < 1)
    return y.float()


def gate_pos_weight(prior: float, cap: float = GATE_POS_WEIGHT_CAP) -> float:
    """pos_weight = min((1 - pi) / pi, cap) for a positive rate pi (from the TRAIN labels)."""
    prior = float(prior)
    if prior <= 0:
        return float(cap)
    return float(min((1.0 - prior) / prior, cap))


def gate_loss(logit: torch.Tensor, y: torch.Tensor, prior: Optional[float] = None,
              pos_weight: Optional[float] = None, cap: float = GATE_POS_WEIGHT_CAP,
              weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean BCE-with-logits of the gate.  pos_weight: given, or min((1 - prior) / prior, cap) from ``prior`` (the
    positive rate of the training labels), or 1.  weight [B]: optional per-draft weights (e.g. UNKNOWN negatives)."""
    if pos_weight is None:
        pos_weight = gate_pos_weight(prior, cap) if prior is not None else 1.0
    pw = torch.as_tensor(min(float(pos_weight), cap), dtype=logit.dtype, device=logit.device)
    l = F.binary_cross_entropy_with_logits(logit, y.to(logit.dtype), pos_weight=pw, reduction="none")
    if weight is not None:
        return (l * weight).sum() / torch.clamp(weight.sum(), min=1e-12)
    return l.mean()


class GateHead(nn.Module):
    """Gate head on DETACHED trunk features (IMPL_SPEC §3.7 / §3.8: the gate BCE never back-propagates into the trunk).
    in_dim -> hidden -> 1 (logit)."""

    def __init__(self, in_dim: int, hidden: int = 192):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        return self.mlp(trunk.detach())[..., 0]


# ----------------------------------------------------------------------------------------------- total
def evaluate_trajectories(traj: torch.Tensor, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig(),
                          dense: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Surrogate terms that need only a trajectory [B, 8, 3] (no decoder output): col, dac, ttc, progress P, ep_sur,
    unknown, keyframe comfort.  ``dense`` overrides the reference (e.g. LQR-tracked states [B, 41, 3] in N)."""
    if dense is None:
        dense = dense_reference(traj)
    B = dense.shape[0]
    out = {"dense": dense, "col": collision_cost(dense, scene, index, cfg), "dac": dac_cost(dense, scene, index, cfg),
           "ttc": ttc_cost(dense, scene, index, cfg, details=True), "unknown": unknown_footprint(dense, scene, index, cfg)}
    if scene.centerline is not None:
        out["P"] = progress(dense, scene, index)
        if scene.p_pdm is not None:
            idx = _index(index, B, scene.p_pdm.shape[0], dense.device)
            out["ep_sur"] = ep_surrogate(out["P"], scene.p_pdm.to(device=dense.device, dtype=dense.dtype)[idx])
    v0 = a0 = None
    if scene.v0 is not None:
        idx = _index(index, B, scene.v0.shape[0], dense.device)
        v0 = scene.v0.to(device=dense.device, dtype=dense.dtype)[idx]
        a0 = None if scene.a0 is None else scene.a0.to(device=dense.device, dtype=dense.dtype)[idx]
    out["cmf"] = comfort_cost(traj, v0, a0, None, cfg)
    return out


def surrogate_terms(dec: Dict[str, torch.Tensor], tau0: torch.Tensor, scene: SceneBatch, index=None,
                    cfg: SurrogateConfig = SurrogateConfig(), details: bool = False,
                    ttc_grad: bool = True) -> Dict[str, torch.Tensor]:
    """All M7 terms of decoded corrections.  dec = decoder.decode(tau0, z, w, v0, ...) output, tau0 [B, 8, 3] the
    drafts.  Returns dict of [B] tensors: col, dac, prog, cmf, mod, ttc, P1, P0, unknown (+ 'details' sub-dicts).
    ttc_grad=False computes the TTC term under no_grad (same values, no autograd graph: for a TTC weight of 0, where
    it is only logged)."""
    B = tau0.shape[0]
    dense = dec["dense"]
    col = collision_cost(dense, scene, index, cfg, details=details)
    dac = dac_cost(dense, scene, index, cfg, details=details)
    with torch.set_grad_enabled(torch.is_grad_enabled() and ttc_grad):
        ttc = ttc_cost(dense, scene, index, cfg, details=details)
    unk = unknown_footprint(dense_reference(tau0.to(dense.dtype)), scene, index, cfg)
    if scene.centerline is not None and scene.p_pdm is not None:
        with torch.no_grad():
            P0 = progress(dense_reference(tau0.to(dense.dtype)), scene, index)
        P1 = progress(dense, scene, index)
        idx = _index(index, B, scene.p_pdm.shape[0], dense.device)
        prog = progress_loss(P1, P0, scene.p_pdm.to(device=dense.device, dtype=dense.dtype)[idx], cfg.prog_thr, unk)
    else:
        P0 = P1 = prog = dense.new_zeros(B)
    v0 = a0 = None
    if scene.v0 is not None:
        idx = _index(index, B, scene.v0.shape[0], dense.device)
        v0 = scene.v0.to(device=dense.device, dtype=dense.dtype)[idx]
        a0 = None if scene.a0 is None else scene.a0.to(device=dense.device, dtype=dense.dtype)[idx]
    cmf = comfort_cost(dec["traj"], v0, a0, dec, cfg)
    out = {"col": col["cost"], "dac": dac["cost"], "prog": prog, "cmf": cmf["cost"], "mod": modification_cost(dec, cfg),
           "ttc": ttc["cost"], "P1": P1, "P0": P0, "unknown": unk}
    if details:
        out["details"] = {"col": col, "dac": dac, "cmf": cmf, "ttc": ttc}
    return out


def surrogate_loss(dec, tau0, scene: SceneBatch, index=None, cfg: SurrogateConfig = SurrogateConfig(),
                   weights: Optional[Dict[str, float]] = None, gate_logit: Optional[torch.Tensor] = None,
                   gate_y: Optional[torch.Tensor] = None, gate_prior: Optional[float] = None,
                   draft_weight: Optional[torch.Tensor] = None):
    """Weighted total (scalar, mean over drafts) + the per-draft terms.  weights default DEFAULT_WEIGHTS; the gate
    term is added when gate_logit and gate_y are given (use GateHead so it cannot reach the trunk).  The TTC term is
    added only when its weight is non-zero (weight 0 / absent -> the per-draft total is bit-identical to the pre-TTC
    code, even where C_ttc is not finite)."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    t = surrogate_terms(dec, tau0, scene, index, cfg, ttc_grad=bool(w.get("ttc", 0.0)))
    per = sum(w.get(k, 0.0) * t[k] for k in ("col", "dac", "prog", "cmf", "mod"))
    if w.get("ttc", 0.0):
        per = per + w["ttc"] * t["ttc"]
    if draft_weight is not None:
        total = (per * draft_weight).sum() / torch.clamp(draft_weight.sum(), min=1e-12)
    else:
        total = per.mean()
    if gate_logit is not None and gate_y is not None:
        t["gate"] = gate_loss(gate_logit, gate_y, prior=gate_prior)
        total = total + w.get("gate", 0.0) * t["gate"]
    t["per_draft"] = per
    return total, t
