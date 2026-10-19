"""refiner v4_3 M5 computed edits (V4Cfg.edits 'computed'; docs/refiner_v4/V4_3_SPEC.md M5): 0-parameter edits of the
critic's shortlist, found on the PREDICTED range fields (ck/v4/range_readout.py) with the official LQR tracker's
executed states (ck/v4/lqr.py), returned as EXTRA candidates (the corrector's K slots) -- never a gate: the pool is not
changed, the edit only joins the set the critic picks from (96 + K).

Frames / units: N frame (x forward, y left; m, s, m/s^2).  Fields of the source's PARENT (the score_set pool output,
once per parent row): B / U (/ Bp) = blk > 0 (logit 0, probability 0.5), D = margin; windows(B) per cell
(range_targets.windows: t_out_block = the end of the cell's last blocked bin, +inf if the last bin is blocked).
The fields are read with the net's own rules (V4Cfg; range_readout.read_geometry / hard_evidence, the readout's
thresholded twin): rg_stopped_aware -> a third group Bp (haz bp_logit_u): NC reads B (= the stopped-aware B), every TTC
read that would read B (delta 0 and |v| delta < U_SWITCH_M = 2 m) reads Bp, U above the switch (read_geometry
ttc_b_chan CH_BP); rg_read 'any4' -> the tube check and the critical reads OR the 2 x 2 cells around each read point
(hard_evidence read='any4' = range_targets.executed_slack(read='any4')), and the longitudinal target of a critical read
follows the any4 critical step of the Phase 0b oracle (tools/ck/v4/oracle/heldout_common.GridTubeR2.critical): the
read's 4 cells (range_readout.any4_cells), t_out_block = the end of the last blocked bin of the OR of their profiles (of
the read's channel group), and the critical arc length max(0, (jb - 0.5) ds) = where an any4 read first touches station
jb (an any4 read at s reads stations floor(s / ds - 0.5) and + 1), jb = the lowest station of the 4 cells that holds a
blocked bin (any4_targets).  Two differences to the oracle's critical step (Phase 0b criterion 2 only): it takes the
lowest station of the 4 whether blocked or not (the same cut when the lower station of the read is blocked, one
station (ds ~ 1 m) earlier -- a harsher brake -- when only the upper one is), and for a TTC critical read at delta 0
it takes the window of the stopped-aware B, where these edits read Bp (= what the oracle's own check() reads TTC on
there: the plain not-behind B, LON >= -1.3 m).
With the defaults (nearest, no Bp) every call below is the R1 code path, bit for bit.

Shortlist (spec M5 "the critic's top-edit_k PARENTS"): the sources are the pool columns that ARE their parent (score_set
haz is_self, bitwise equal to the parent; one column per distinct parent, parent_columns), ranked by sel_score when
given -- the deployment selection score -- else the critic-only corrector.critic_select_score; variant columns, repeats
of a parent and invalid / non-finite columns are excluded (with fewer than k eligible parents the remaining slots hold
an excluded column bitwise with src_valid False, edit_kind 0: callers mask them).  So the rule family below is applied
to a parent, never on top of a speed / lateral variant, and two siblings of one parent never take two slots.
Per shortlisted source c:
  * no edit (kind 0, the slot holds the source itself, bitwise) when the ego speed v0 = |(vx, vy)| < 2 m/s, the source's
    draft length S_8 < 3 m (decoder S_LAT_MIN), the source is invalid / non-finite, the source is inside the predicted
    tube (its LQR states read slack_nc, slack_ttc >= 0 s and slack_dac >= 0 m, range_readout.hard_evidence), or the
    tube is empty (no edit of the family below reaches the target);
  * NC / TTC-limited (slack_nc < 0 or slack_ttc < 0): longitudinal edit = the CK2 speed-variant family (variants
    make_variants (a, 0): decoder mode A, Q_1..Q_6 = a, i.e. da ramps 0 -> a over the first 0.8 s B-spline span, then
    holds) with the SMALLEST magnitude a in [-3.96, 0) (0.99 A_DEC; tanh bound) such that the executed arrival at the
    critical cell is >= t_out_block + edit_dt: NC = the first time the executed rear axle's station coordinate (Frenet
    on the parent frame) reaches the cell's bin start j ds; TTC = min over delta in (0, 0.3, 0.6, 0.9) s of (the first
    time the state projected |v| delta along its heading reaches j ds) + delta, against the window of the channel the
    critical read used (B, or U where |v| delta >= 2 m; Bp in place of B with rg_stopped_aware; with rg_read 'any4' the
    window of the OR of the read's 4 cells and the cut (jb - 0.5) ds, header).  Search: a coarse scan A_SCAN then
    N_BISECT bisection steps in the first satisfying bracket (resolution 0.5 / 2^5 = 0.016 m/s^2; arrival is monotone
    in a: a lower speed profile reaches every station later).  Up to MAX_ROUNDS rounds: if the edited trajectory still
    hits B / U (/ Bp) somewhere (a later critical cell), the search continues beyond the current a for the new cell;
  * DAC-limited (slack_dac < 0 on the profile after the longitudinal edit): lateral edit = the rule l+-0.5 shape
    (variants w_rule: decoder.smoothstep_lat with onset s_on = 0.2 S_8, flat end at S_8, projected on the lateral
    B-spline, lat_len = S_8) with the smallest |D| in (0, 2] m, on EITHER side, whose executed min corner margin
    (corner_margin: hard min over the 41 states x 4 corners of the bilinear D) is >= edit_dlat (lateral_search: the
    scan D_SCAN + bisection run for D > 0 (left) and D < 0 (right) in one batch, resolution 0.25 / 2^5 = 0.008 m; the
    smaller |D| wins, a tie goes left).  Not "toward the larger grid margin": when the centred box is already outside
    at the critical station both grid margins equal f0 (range_targets.margins rule), and the margins about the
    parent's centreline ignore where the executed path actually is, so the side is decided by the executed check
    itself (the spec text "toward the larger margin" is the common case of this rule);
  * both: longitudinal first, the lateral search on the edited profile (one decode with both controls, as the CK2
    combo variants).
  edit_kind [T, k]: 0 none, 1 lon, 2 lat, 3 both.  u_rule [T, k, 2] = (a_lon, d_lat) (0 where not edited).
Corrector A (V4Cfg.corrector, optional, off in the first student run): a zero-initialised residual on top of the
computed edit -- the corrector's controls u [T, k, 12] (decoder mode B, corrector.decode_edits) on the computed edit
edit_base, from the SOURCE's corrector context (pool_out['ctx'], score_set(..., return_ctx=True)); mu = 0 at init = the
computed edit.  'u' keeps correct_topk's meaning (the 12 corrector controls the rec2 edit block / labeler2 / eval_e2e2 /
corrector_logp store and replay: corrector_logp(src = the pool source, u) and decode_edits(edit_base, u)); without the
corrector it is 0 (the identity residual) and edit_traj = edit_base bitwise.
The edits are re-scored by the same critic on their source's parent corridor (the pool's parent pass when pool_out
carries 'pctx': no second trunk.stations); the caller gives each edit its source's v2_final / im (src_idx) for the
blended selection over the pool + the edits and labels them with the pool (labeler2 96 + K).

Approximations (the M5 rule is a target, the critic judges the result): the field's cell heading is the parent's;
arrival is measured at one critical cell per round (any4: one critical read's 4 cells); the TTC arrival is the min over
every delta even where only some deltas read the critical read's group (B / Bp below the 2 m switch, U above; the same
rule in R1 and R2); the lateral search does not re-check NC / TTC (the critic scores the edit).  Measured behaviour
worth knowing: braking can clear a DAC violation by itself (a shorter executed path, and the tracker follows a lateral
offset more slowly at the lower speed), so the lateral need is re-read after the longitudinal edit; a box that is
outside before the rule shape's onset (0.2 S_8) cannot be fixed by the lateral family (empty lateral tube, edit_need 2
without edit_kind 2).
Known fragility of the lon target (spec rule, kept; designer decision, review 2026-10-16 KST): t_out_block is the end
of the cell's LAST blocked bin (+inf if the last bin is blocked), so on PREDICTED fields one spurious late blocked bin
(a free gap after the real run, then noise at e.g. 4.5 s or in bin M - 1) moves the target to that bin's end or +inf and
forces a harsher brake / a stop although clearing the run that holds the critical read would have been enough.
Runtime (CPU, 1 thread, tests/test_v4_edits.py::test_cpu_runtime_T16x96): T = 16 x 96 pool, critic top-8 parents per
token, worst case (all 128 sources run the lon rounds and the two-sided lat search) 1.0 .. 1.7 s on top of the pool's
~5.0 s score_set (2026-10-16 KST, 4 runs).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ...refiner.decoder import A_DEC, D_MAX, N_FREE, S_LAT_MIN, decode, project_lat, smoothstep_lat
from ...refiner.geometry import DraftPath
from ...refiner.sdf import SDF_CLIP, ego_corners
from ..temporal.raster import frenet
from ..variants import S_ON_FRAC
from . import lqr as LQR
from . import range_targets as RT
from .range_readout import (CH_B, CH_BP, CH_U, TTC_DELTAS, _ttc_points, any4_cells, grid_coords, hard_evidence,
                            hard_slack, read_cells, read_geometry)

Tensor = torch.Tensor
KIND_NONE, KIND_LON, KIND_LAT, KIND_BOTH = 0, 1, 2, 3
V0_MIN = 2.0                                     # [m/s] no edit below this ego speed
S8_MIN = S_LAT_MIN                               # [m] no edit for drafts shorter than this (3 m)
A_MIN = -0.99 * A_DEC                            # [m/s^2] strongest deceleration (tanh bound of decoder mode A)
A_SCAN = (-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, A_MIN)
D_SCAN = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
N_BISECT = 5
MAX_ROUNDS = 3
I_NC, I_TTC, I_DAC = RT.I_NC, RT.I_TTC, RT.I_DAC
FIELD_KEYS = ("occ_logit_u", "ovl_logit_u", "bp_logit_u")      # score_set haz groups B | U | Bp (CH_B, CH_U, CH_BP)
A4_KEYS = ("a4_cut_nc", "a4_tout_nc", "a4_station_nc", "a4_cut_ttc", "a4_tout_ttc", "a4_station_ttc")


# ----------------------------------------------------------------------------------------------- decode
_E1: Dict[float, np.ndarray] = {}


def lat_unit(s_on_frac: float = S_ON_FRAC) -> np.ndarray:
    """e(1) [6]: the lateral B-spline controls of the rule shape (smoothstep onset s_on_frac, flat end) for D = 1 m
    before the |e| <= 0.98 D_MAX clip; project_lat is linear, so e(D) = clip(D e(1)) = variants.w_rule's e."""
    if s_on_frac not in _E1:
        e1 = project_lat(smoothstep_lat(1.0, s_on_frac, 1.0))
        assert float(np.abs(e1).max()) < 0.98 * D_MAX - 1e-6          # D = 1 m is inside project_lat's clip
        _E1[s_on_frac] = e1
    return _E1[s_on_frac]


def rule_controls(a: Tensor, D: Tensor) -> Tuple[Tensor, Tensor]:
    """(a [n] m/s^2 <= 0, D [n] m) -> raw decoder controls z [n, 6] (= variants.z_speed: atanh(a / A_DEC)), w [n, 6]
    (= variants.w_rule(D): atanh(clip(D e(1), +-0.98 D_MAX) / D_MAX)), in a's dtype."""
    dt, dev = a.dtype, a.device
    z = torch.atanh((a.clamp(min=A_MIN) / A_DEC)).reshape(-1, 1).expand(-1, N_FREE)
    e1 = torch.as_tensor(lat_unit(), dtype=dt, device=dev)
    e = (D.reshape(-1, 1).to(dt) * e1).clamp(-0.98 * D_MAX, 0.98 * D_MAX)
    return z.contiguous(), torch.atanh(e / D_MAX)


@torch.no_grad()
def decode_rule(src: Tensor, a: Tensor, D: Tensor, v0: Tensor) -> Tensor:
    """Rule-family edits of src [n, 8, 3]: speed offset a [n] (decoder mode A; 0 = none) + lateral shift D [n] (0 =
    none) -> [n, 8, 3] float32 (decoded in float64 as the CK2 variants; rows with a = D = 0 are src bitwise)."""
    n = src.shape[0]
    if n == 0:
        return src.float().clone()
    s64 = src.double()
    z, w = rule_controls(a.double(), D.double())
    S8 = DraftPath(s64).knots()[:, -1]
    o = decode(s64, z, w, v0=v0.double().reshape(n), mode="A", lat_len=S8)
    same = (a == 0) & (D == 0)
    return torch.where(same.reshape(n, 1, 1), src.float(), o["traj"].float())


# ----------------------------------------------------------------------------------------------- search
def scan_bisect(ok_fn: Callable[[Tensor, Tensor], Tensor], rows: Tensor, start: Tensor, scan: Tensor,
                n_bisect: int = N_BISECT) -> Tuple[Tensor, Tensor]:
    """Smallest |x| beyond |start| along the signed scan direction satisfying ok_fn: rows [m] (indices into the
    caller's arrays), start [m] (current value, assumed not ok), scan [m, G] (signed values of increasing magnitude)
    -> (x [m] (start where none), found [m]).  ok_fn(idx [q], x [q]) -> ok [q]: one batched call for the scan (values
    with |x| <= |start| skipped), then n_bisect calls on the bracket (last not-ok, first ok]."""
    m, G = scan.shape
    if m == 0:
        return start.clone(), torch.zeros(0, dtype=torch.bool, device=start.device)
    beyond = scan.abs() > start.abs()[:, None] + 1e-9
    ri, gi = torch.nonzero(beyond, as_tuple=True)
    okm = torch.zeros(m, G, dtype=torch.bool, device=start.device)
    if ri.numel():
        okm[ri, gi] = ok_fn(rows[ri], scan[ri, gi])
    found = okm.any(-1)
    first = okm.to(torch.int64).argmax(-1)
    hi = torch.gather(scan, 1, first[:, None])[:, 0]
    prev = torch.gather(scan, 1, (first - 1).clamp(min=0)[:, None])[:, 0]
    use_prev = (first > 0) & torch.gather(beyond, 1, (first - 1).clamp(min=0)[:, None])[:, 0]
    lo = torch.where(use_prev, prev, start)
    sel = torch.nonzero(found, as_tuple=True)[0]
    for _ in range(int(n_bisect)):
        if sel.numel() == 0:
            break
        mid = 0.5 * (lo[sel] + hi[sel])
        ok = ok_fn(rows[sel], mid)
        hi[sel] = torch.where(ok, mid, hi[sel])
        lo[sel] = torch.where(ok, lo[sel], mid)
    return torch.where(found, hi, start), found


def first_crossing(q: Tensor, cut: Tensor, t: Tensor) -> Tensor:
    """First time q [n, Q] (a station coordinate per state, s at t [Q]) reaches cut [n] (linear between states; 0 if
    q_0 >= cut; +inf if never) -> [n] float64."""
    q = q.double()
    c = cut.double()[:, None]
    reach = q >= c
    anyr = reach.any(-1)
    n1 = reach.to(torch.int64).argmax(-1)
    n0 = (n1 - 1).clamp(min=0)
    q1 = torch.gather(q, 1, n1[:, None])[:, 0]
    q0 = torch.gather(q, 1, n0[:, None])[:, 0]
    td = t.double()
    t1, t0 = td[n1], td[n0]
    frac = torch.where(q1 > q0, (c[:, 0] - q0) / torch.where(q1 > q0, q1 - q0, torch.ones_like(q1)),
                       torch.zeros_like(q1)).clamp(0.0, 1.0)
    ta = torch.where(n1 == 0, torch.zeros_like(t1), t0 + frac * (t1 - t0))
    return torch.where(anyr, ta, torch.full_like(ta, math.inf))


@torch.no_grad()
def arrival_times(states: Tensor, fr: Dict[str, Tensor], cut: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """states [n, Q, 6] of rows with station frames fr [n] and cut [n] (m, station coordinate of the critical cell's
    bin start) -> (t_nc [n]: first time the rear axle reaches cut, t_ttc [n]: min over TTC_DELTAS of (first time the
    state projected |v| delta along its heading reaches cut) + delta), float64, +inf if never within the states."""
    n, Q = states.shape[:2]
    st = states.float()
    pr, _, _ = _ttc_points(st)                                                     # [n, Q, 4, 2]
    s = frenet(pr.reshape(n, Q * len(TTC_DELTAS), 2), fr)["s"].reshape(n, Q, len(TTC_DELTAS))
    s = torch.where(torch.isfinite(s), s, torch.full_like(s, -1e9))
    s = torch.cummax(s, 1).values                                                  # "reaches" = first crossing
    tt = [first_crossing(s[..., k], cut, t) + float(dl) for k, dl in enumerate(TTC_DELTAS)]
    return tt[0], torch.stack(tt, -1).min(-1).values


@torch.no_grad()
def corner_margin(states: Tensor, fr_rows: Dict[str, Tensor], D: Tensor, rows: Tensor) -> Tensor:
    """Executed min corner margin [n] (m) of states [n, Q, 6] (frames fr_rows [n] of their parents' station grids):
    hard min over the Q states x the 4 official corners (executed heading) of the bilinear D [R, 48, 17] of rows [n]
    (range_readout 'field' domain; SDF_CLIP when no corner is readable)."""
    n = states.shape[0]
    cor = ego_corners(states[..., :3].float()).reshape(n, -1, 2)
    gc = grid_coords(fr_rows, cor, "field")
    dv = read_cells(D[..., None], rows, gc["us"], gc["ud"], 0)
    mn = torch.where(gc["ok"], dv, torch.full_like(dv, math.inf)).min(-1).values
    return torch.where(torch.isfinite(mn), mn, torch.full_like(mn, SDF_CLIP))


def lateral_ok(src: Tensor, a_lon: Tensor, v0: Tensor, lqr_ego: Tensor, fr: Dict[str, Tensor], D: Tensor,
               d_min: float) -> Callable[[Tensor, Tensor], Tensor]:
    """ok_fn of the lateral search (scan_bisect / lateral_search) for n rows: src [n, 8, 3] (N frame), a_lon [n]
    (m/s^2, the longitudinal edit decoded with the shift; 0 none), v0 [n] (m/s), lqr_ego [n, 2] (lqr.simulate's ego),
    fr = station frames of the n rows' parents (range_targets.parent_frame rows), D [n, 48, 17] (m) their margin fields
    -> ok(rr [q], d [q] m) = corner_margin(LQR states of decode_rule(src[rr], a_lon[rr], d)) >= d_min [q] bool."""
    def ok(rr: Tensor, d: Tensor) -> Tensor:
        tr = decode_rule(src[rr], a_lon[rr], d.float(), v0[rr])
        st = LQR.simulate(tr, lqr_ego[rr], dtype=torch.float32)
        return corner_margin(st, RT.gather_frame(fr, rr), D, rr) >= float(d_min)
    return ok


def lateral_search(ok_fn: Callable[[Tensor, Tensor], Tensor], rows: Tensor,
                   n_bisect: int = N_BISECT) -> Tuple[Tensor, Tensor]:
    """Smallest |D| in (0, D_SCAN[-1]] m on EITHER side satisfying ok_fn (lateral_ok) for rows [m]: one scan_bisect
    over the rows twice (D > 0 left, D < 0 right; same number of batched calls, twice the rows) -> (D [m] (0 where
    none), found [m]); the side with the smaller |D| wins, a tie goes left.  Decided by the executed check, not by a
    margin heuristic (module docstring)."""
    m = rows.numel()
    dev = rows.device
    if m == 0:
        return torch.zeros(0, device=dev), torch.zeros(0, dtype=torch.bool, device=dev)
    sp = torch.as_tensor(D_SCAN, dtype=torch.float32, device=dev)[None].expand(m, -1)
    d2, f2 = scan_bisect(ok_fn, torch.cat([rows, rows]), torch.zeros(2 * m, device=dev),
                         torch.cat([sp, -sp]).contiguous(), n_bisect)
    d_p, d_n, f_p, f_n = d2[:m], d2[m:], f2[:m], f2[m:]
    use_n = f_n & (~f_p | (d_n.abs() < d_p.abs()))
    return torch.where(use_n, d_n, torch.where(f_p, d_p, torch.zeros_like(d_p))), f_p | f_n


@torch.no_grad()
def parent_columns(pool: Tensor, is_self: Tensor) -> Tensor:
    """[T, N] bool: the pool columns [T, N, 8, 3] that ARE their parent (score_set haz is_self [T, N], bitwise equal to
    the parent), one column per distinct parent (the first; later columns equal to an earlier parent column -- e.g. the
    S_8 < 3 m lateral variants, which equal their parent -- are dropped)."""
    T, N = is_self.shape
    s = is_self.bool()
    key = pool.float().reshape(T, N, -1)
    eq = s[:, :, None] & s[:, None, :]                                               # [T, N, N]
    for c in range(key.shape[-1]):                                                   # peak memory T N^2 (not x24)
        eq &= key[:, :, None, c] == key[:, None, :, c]
    ar = torch.arange(N, device=pool.device)
    earlier = (eq & (ar[None, None, :] < ar[None, :, None])).any(-1)
    return s & ~earlier


@torch.no_grad()
def any4_targets(blk_b: Tensor, rows: Tensor, geo: Dict[str, Tensor], M: int, tbin: float,
                 ds: Tensor) -> Dict[str, Tensor]:
    """Longitudinal targets of the NC and the TTC critical read under rg_read 'any4' (module header; the Phase 0b
    oracle's GridTubeR2.critical step on the thresholded predicted fields): blk_b [R, 48, 17, G M] bool (B | U [| Bp]),
    rows [n], geo = read_geometry(...) of the n rows, ds [n] (m, the rows' station spacing) -> per key k in (nc, ttc):
    a4_station_k [n] long (the nearest station of the critical read = hard_evidence crit_station), a4_cut_k [n] f32
    (m, max(0, (jb - 0.5) ds), jb = the lowest station among the read's 2 x 2 cells with a blocked bin in the read's
    channel group, the lowest of the 4 if none), a4_tout_k [n] float64 (s, range_targets.windows t_out_block of the OR
    of the 4 cells' profiles in that group: 0 if none blocked, +inf if the last bin is).  The critical read = the
    arg-min point of the same range_readout.hard_slack call as hard_evidence(read='any4') (first minimum); rows without
    a hit of that key get unused values."""
    n = rows.shape[0]
    dev = rows.device
    bidx = torch.arange(n, device=dev)
    mi = torch.arange(int(M), device=dev)
    out: Dict[str, Tensor] = {}
    for key, chan in (("nc", 0), ("ttc", geo["chan"])):
        g = geo[key]
        sl, j, _ = hard_slack(blk_b, rows, g, geo[f"t_{key}"], chan, geo[f"use_{key}"], M, tbin, read="any4")
        q = sl.min(-1).indices                                                        # hard_evidence's critical point
        jn, in_, _ = any4_cells(g)                                                    # [n, P, 4]
        jq, iq = jn[bidx, q], in_[bidx, q]                                            # [n, 4]
        grp = (torch.zeros(n, dtype=torch.long, device=dev) if not torch.is_tensor(chan)
               else chan.long()[bidx, q])
        prof = blk_b[rows.reshape(n, 1).expand(n, 4), jq, iq]                         # [n, 4, G M]
        sel = (grp[:, None] * int(M) + mi[None])[:, None, :].expand(n, 4, int(M))
        prof4 = torch.gather(prof, 2, sel)                                            # [n, 4, M] the read's group
        out[f"a4_tout_{key}"] = RT.windows(prof4.any(1)[:, None, None, :], tbin)["t_out_block"][:, 0, 0]
        blk4 = prof4.any(-1)                                                          # [n, 4] cells with a blocked bin
        jb = torch.where(blk4, jq, torch.full_like(jq, 1 << 20)).min(-1).values
        jb = torch.where(blk4.any(-1), jb, jq.min(-1).values)
        out[f"a4_cut_{key}"] = ((jb.float() - 0.5) * ds.float()).clamp(min=0.0)
        out[f"a4_station_{key}"] = j[bidx, q]
    return out


# ----------------------------------------------------------------------------------------------- main
@torch.no_grad()
def _gather_sources(pool_out: Dict, idx: Tensor, parents: Tensor, corridor_version: int, M: int, groups: int = 2):
    """per source (n = T k rows): fields blk [n, 48, 17, groups M] (logits B | U (| Bp with groups 3, the net's
    rg_stopped_aware)), D [n, 48, 17] of its parent, the parent's station frame (pool_out 'pctx' rg_frame rows, else
    corridor_geometry of the gathered parents)."""
    h = pool_out["haz"]
    T, k = idx.shape
    if int(groups) not in (2, 3):
        raise ValueError(f"computed_edits: {groups} field groups (B | U, or B | U | Bp)")
    keys = FIELD_KEYS[:int(groups)]
    miss = [kk for kk in keys if kk not in h]
    if miss or (int(groups) == 2 and FIELD_KEYS[2] in h):
        raise ValueError(f"computed_edits: pool_out haz has {sorted(kk for kk in FIELD_KEYS if kk in h)}, the net's "
                         f"rg_stopped_aware needs {list(keys)}")
    pi = torch.gather(h["parent_index"], 1, idx)                                    # [T, k] parent row of each source
    U = h["occ_logit_u"].shape[1]
    rows = (torch.arange(T, device=idx.device)[:, None] * U + pi).reshape(-1)
    per = lambda a: a.reshape(T * U, *a.shape[2:]).index_select(0, rows).detach().float()
    blk = torch.cat([per(h[kk]) for kk in keys], -1).contiguous()                  # [n, 48, 17, groups M]
    D = per(h["margin_u"]).contiguous()
    pc = pool_out.get("pctx")
    if pc is not None and "rg_frame" in pc:
        fr = RT.gather_frame(pc["rg_frame"], rows)
    else:
        from ...refiner.corridor import corridor_geometry
        g4 = torch.gather(parents.float(), 1, idx[..., None, None].expand(T, k, 8, 3)).reshape(-1, 8, 3)
        fr = RT.parent_frame(corridor_geometry(g4, version=corridor_version))
    fr = {kk: v.float() for kk, v in fr.items()}
    return blk, D, fr, pi


def computed_edits(net, feat: Tensor, mem: Tensor, ego, pool: Tensor, parents: Tensor, pool_out: Dict,
                   k: Optional[int] = None, valid: Optional[Tensor] = None, sel_score: Optional[Tensor] = None,
                   pref: Optional[Tensor] = None, sample: bool = False,
                   generator: Optional[torch.Generator] = None) -> Dict:
    """CKNet.computed_edits (module docstring).  pool [T, N, 8, 3] with parents [T, N, 8, 3], pool_out =
    net.score_set(feat, mem, ego, pool, parents, return_pctx=True[, return_ctx=True with the corrector]) of a range
    critic, valid [T, N] / sel_score [T, N] of the pool columns -> dict (the correct_topk keys src_idx, u, edit_traj,
    edit_score_logit, edit_w_lat, edit_haz with their shapes, + the rule's own keys): src_idx [T, k] (pool columns,
    parents only: parent_columns), edit_traj [T, k, 8, 3] f32, edit_score_logit [T, k, 5], edit_w_lat [T, k, 6],
    edit_haz (score_set 'haz' of the edits), u [T, k, 12] (the corrector's controls on edit_base, correct_topk's u;
    0 without the corrector), edit_base [T, k, 8, 3] (the computed edit before the corrector residual; = edit_traj
    without it), u_rule [T, k, 2] = (a_lon m/s^2, d_lat m), a_lon / d_lat [T, k], edit_kind [T, k] int64 (0 none /
    1 lon / 2 lat / 3 both), edit_need [T, k] (the same code for the keys that were outside the predicted tube and
    searched: need & ~kind = no edit of the family reached the target, an empty tube), src_valid [T, k] (an eligible
    source: a valid, finite parent column; False = a filler slot), edit_clear [T, k] (the final trajectory reads
    inside the predicted tube), src_score [T, k] (the shortlist score); with the corrector also mu / log_sig
    [T, k, 12], logp [T, k], ctx."""
    from .corrector import N_U, critic_select_score, decode_edits
    cfg = net.v4
    if not cfg.range:
        raise ValueError("computed_edits needs readout 'range' (the predicted fields)")
    T, N = pool.shape[:2]
    kk = min(int(k or cfg.edit_k), N)
    dev = pool.device
    M = int(cfg.ev_tbins)
    tbin = float(cfg.ev_tbin_s)
    sa = bool(cfg.rg_stopped_aware)                 # R2 (1): + the Bp group, TTC's B read (module header)
    read = str(cfg.rg_read)                         # R2 (2): 'nearest' | 'any4'
    ttc_b = CH_BP if sa else CH_B
    sc = (critic_select_score(pool_out["score_logit"].detach()) if sel_score is None else sel_score.detach().float())
    vmask = torch.ones(T, N, dtype=torch.bool, device=dev) if valid is None else valid.bool().to(dev)
    vmask = vmask & torch.isfinite(pool.float()).flatten(2).all(-1)
    vmask = vmask & parent_columns(pool, pool_out["haz"]["is_self"]).to(dev)        # spec M5: the top-k PARENTS
    sc = torch.where(vmask, sc, torch.full_like(sc, -1e9))
    idx = torch.topk(sc, kk, dim=1).indices                                          # [T, k]
    g4 = lambda a: torch.gather(a, 1, idx.reshape(T, kk, *([1] * (a.dim() - 2))).expand(T, kk, *a.shape[2:]))
    src = g4(pool.float())
    n = T * kk
    flat = src.reshape(n, 8, 3)
    fin = torch.isfinite(flat).flatten(1).all(1)
    flat_s = torch.where(fin[:, None, None], flat, torch.zeros_like(flat))              # simulated / decoded rows
    v0, _, eds, _ = ego
    rep = lambda a: a.float().reshape(T, 1).repeat_interleave(kk, 0)[:, 0]
    v0n, vx, ax = rep(torch.nan_to_num(v0.float(), nan=0.0)), rep(eds[:, 0]), rep(eds[:, 2])
    lqr_ego = torch.stack([torch.nan_to_num(vx, nan=0.0), torch.nan_to_num(ax, nan=0.0)], -1)
    tq = torch.arange(LQR.NT, dtype=torch.float64, device=dev) * LQR.DT
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        blk, D, fr, _ = _gather_sources(pool_out, idx, parents, net.corridor_version, M, 3 if sa else 2)
        blk_b = blk > 0
        if read == "nearest":                       # per-cell windows (any4: of the critical read's 4 cells, below)
            win_b, win_u = RT.windows(blk_b[..., :M], tbin), RT.windows(blk_b[..., M:2 * M], tbin)
            win_p = RT.windows(blk_b[..., 2 * M:], tbin) if sa else None
        rows_all = torch.arange(n, device=dev)
        sim = lambda tr, r: LQR.simulate(tr, lqr_ego[r], dtype=torch.float32)

        def evidence(states: Tensor, r: Tensor, crit: bool = False) -> Dict:
            geo = read_geometry(RT.gather_frame(fr, r), states, tq, M, tbin, ttc_b_chan=ttc_b)
            e = hard_evidence(blk_b, D, r, geo, M, tbin, read=read)
            if crit and read == "any4":             # the any4 targets of the critical reads (A4_KEYS)
                e.update(any4_targets(blk_b, r, geo, M, tbin, fr["ds"][r]))
            return e

        ev = evidence(sim(flat_s, rows_all), rows_all, crit=True)
        S8 = DraftPath(flat_s.double()).knots()[:, -1].float()
        elig = (v0n >= V0_MIN) & (S8 >= S8_MIN) & torch.gather(vmask, 1, idx).reshape(n) & fin
        a_lon = torch.zeros(n, dtype=torch.float32, device=dev)
        d_lat = torch.zeros(n, dtype=torch.float32, device=dev)
        lon_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        # ---------------------------------------------------------------- longitudinal (NC / TTC)
        need = elig & ((ev["slack_nc"] < 0) | (ev["slack_ttc"] < 0))
        ev_cur = ev
        for _ in range(MAX_ROUNDS):
            r = torch.nonzero(need, as_tuple=True)[0]
            if r.numel() == 0:
                break
            use_nc = ev_cur["slack_nc"][r] < 0
            use_tt = ev_cur["slack_ttc"][r] < 0
            if read == "any4":                      # the read's 4 cells: OR window, cut (jb - 0.5) ds (any4_targets)
                cut_nc, cut_tt = ev_cur["a4_cut_nc"][r], ev_cur["a4_cut_ttc"][r]
                tout_nc, wtt = ev_cur["a4_tout_nc"][r], ev_cur["a4_tout_ttc"][r]
            else:
                cs, ci = ev_cur["crit_station"][r], ev_cur["crit_lat"][r]
                ds = fr["ds"][r]
                cut_nc = cs[:, I_NC].clamp(min=0).float() * ds
                cut_tt = cs[:, I_TTC].clamp(min=0).float() * ds
                jn, in_ = cs[:, I_NC].clamp(min=0), ci[:, I_NC].clamp(min=0)
                jt, it = cs[:, I_TTC].clamp(min=0), ci[:, I_TTC].clamp(min=0)
                tout_nc = win_b["t_out_block"][r, jn, in_]                             # [m] float64, +inf possible
                ch = ev_cur["crit_chan"][r]                 # TTC window of the group its critical read used
                wtt = torch.where(ch == CH_U, win_u["t_out_block"][r, jt, it], win_b["t_out_block"][r, jt, it])
                if sa:
                    wtt = torch.where(ch == CH_BP, win_p["t_out_block"][r, jt, it], wtt)
            pos = torch.full((n,), -1, dtype=torch.long, device=dev)
            pos[r] = torch.arange(r.numel(), device=dev)

            def ok_lon(rr: Tensor, a: Tensor) -> Tensor:
                q = pos[rr]
                tr = decode_rule(flat_s[rr], a.float(), torch.zeros_like(a.float()), v0n[rr])
                st = sim(tr, rr)
                frr = RT.gather_frame(fr, rr)
                ok = torch.ones(rr.shape[0], dtype=torch.bool, device=dev)
                tn, _ = arrival_times(st, frr, cut_nc[q], tq)
                ok_nc = tn >= tout_nc[q] + float(cfg.edit_dt) - 1e-9
                _, tt = arrival_times(st, frr, cut_tt[q], tq)
                ok_tt = tt >= wtt[q] + float(cfg.edit_dt) - 1e-9
                ok = ok & torch.where(use_nc[q], ok_nc, torch.ones_like(ok)) & torch.where(use_tt[q], ok_tt,
                                                                                          torch.ones_like(ok))
                return ok

            scan = torch.as_tensor(A_SCAN, dtype=torch.float32, device=dev)[None].expand(r.numel(), -1).contiguous()
            a_new, found = scan_bisect(ok_lon, r, a_lon[r], scan)
            a_lon[r] = torch.where(found, a_new, a_lon[r])
            lon_ok[r] = lon_ok[r] | found
            need = torch.zeros_like(need)
            rf = r[found]
            if rf.numel() == 0:
                break
            e1 = evidence(sim(decode_rule(flat_s[rf], a_lon[rf], torch.zeros_like(a_lon[rf]), v0n[rf]), rf), rf,
                          crit=True)
            still = (e1["slack_nc"] < 0) | (e1["slack_ttc"] < 0)
            # rows that still hit B / U (/ Bp) at another (later) critical cell continue beyond their a next round
            ev_cur = {kk_: v.clone() for kk_, v in ev_cur.items()}
            for kk_ in ("slack_nc", "slack_ttc", "crit_station", "crit_lat", "crit_chan") + (
                    A4_KEYS if read == "any4" else ()):
                ev_cur[kk_][rf] = e1[kk_]
            need[rf] = still & (a_lon[rf] > A_MIN + 1e-6)
        # ---------------------------------------------------------------- lateral (DAC) on the edited profile
        lr = torch.nonzero(lon_ok, as_tuple=True)[0]
        slack_dac = ev["slack_dac"].clone()
        if lr.numel():
            e1 = evidence(sim(decode_rule(flat_s[lr], a_lon[lr], torch.zeros_like(a_lon[lr]), v0n[lr]), lr), lr)
            slack_dac[lr] = e1["slack_dac"]
        need_l = elig & (slack_dac < 0)
        r = torch.nonzero(need_l, as_tuple=True)[0]
        lat_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        if r.numel():                                   # either side, the smaller |D| (lateral_search docstring)
            d_new, found = lateral_search(lateral_ok(flat_s, a_lon, v0n, lqr_ego, fr, D, cfg.edit_dlat), r)
            d_lat[r] = torch.where(found, d_new, d_lat[r])
            lat_ok[r] = found
        kind = lon_ok.long() * KIND_LON + lat_ok.long() * KIND_LAT
        need_k = (elig & ((ev["slack_nc"] < 0) | (ev["slack_ttc"] < 0))).long() * KIND_LON + need_l.long() * KIND_LAT
        a_lon = torch.where(lon_ok, a_lon, torch.zeros_like(a_lon))
        edit = decode_rule(flat_s, a_lon, d_lat, v0n)
        edit = torch.where((kind > 0).reshape(n, 1, 1), edit, flat)                  # no edit: the source bitwise
        e2 = evidence(sim(torch.where(fin[:, None, None], edit, flat_s), rows_all), rows_all)
        clear = (e2["slack_nc"] >= 0) & (e2["slack_ttc"] >= 0) & (e2["slack_dac"] >= 0)
    edit_traj = edit.reshape(T, kk, 8, 3)
    out = {"src_idx": idx, "src_score": torch.gather(sc, 1, idx), "src_valid": torch.gather(vmask, 1, idx),
           "a_lon": a_lon.reshape(T, kk), "d_lat": d_lat.reshape(T, kk),
           "u_rule": torch.stack([a_lon, d_lat], -1).reshape(T, kk, 2), "edit_kind": kind.reshape(T, kk),
           "edit_need": need_k.reshape(T, kk), "edit_clear": clear.reshape(T, kk), "edit_base": edit_traj,
           "u": torch.zeros(T, kk, N_U, dtype=torch.float32, device=dev)}      # identity residual (no corrector)
    if net.corrector is not None:                       # optional zero-init residual on the computed edit (corrector A)
        if "ctx" not in pool_out:
            raise ValueError("computed_edits with the corrector: pool_out needs score_set(..., return_ctx=True)")
        ctx = torch.cat([g4(pool_out["ctx"]), g4(pool_out["score_logit"].detach().float()),
                         (v0.float() / 15.0).reshape(T, 1, 1).expand(T, kk, 1)], -1)
        a = net.corrector.act(ctx, sample=sample, generator=generator)
        edit_traj = decode_edits(edit_traj, a["u"], v0)
        out.update(u=a["u"], mu=a["mu"], log_sig=a["log_sig"], logp=a["logp"], ctx=ctx)
    if pool_out.get("pctx") is not None:                # the pool's parent pass (bit for bit the re-run, score_set)
        oe = net.score_set(feat, mem, ego, edit_traj, pref=pref, parent_index=g4(pool_out["parent_index"]),
                           pctx=pool_out["pctx"])
    else:
        oe = net.score_set(feat, mem, ego, edit_traj, g4(parents.float()), pref=pref)
    out.update(edit_traj=edit_traj, edit_score_logit=oe["score_logit"], edit_w_lat=oe["w_lat"],
               edit_haz=oe.get("haz"))
    return out
