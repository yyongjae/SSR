"""Per-step official labels of tracked paths (report 124 §3-3, detail D3).

The official v1 scorer aggregates over the 40 future steps (first at-fault overlap of a track, any TTC hit, any
off-road corner).  The refiner needs the same verdicts per step, before that aggregation:

  ovl     [B, K, 41]  at-fault overlap of the ego box at step t with some object at t (exact_verdict's NC rule
                      without the per-proposal track memory; step 0 = 0)
  ttc     [B, K, 41]  the TTC rule at step t: the box shifted by speed * dt (dt 0, 0.3, 0.6, 0.9 s) hits an object at
                      t + dt that is ahead (< 30 deg) or not behind while the ego is in multiple lanes / non-drivable /
                      an intersection (exact_verdict's TTC rule without the track memory)
  gap     [B, K, 41]  SIGNED separation (m; < 0 = penetration depth) between those shifted boxes and the candidate
                      objects at t + dt, minimised over (object, dt)
  vapp    [B, K, 41]  closing speed (m/s, + = approaching) between the ego box centre and that object
  pos     [B, K, 41, 2] that object's box centre at t + dt*, in the frame of the (unshifted) ego state at t
                      (x forward, y left), m
  obj     [B, K, 41]  1 where a candidate object exists; where it does not, gap = pos_x = NO_OBJ_M, pos_y = 0,
                      vapp = 0 (fixed values, report 124 §2-3: no mask, so 'no object' is distinguishable)
  offroad [B, K, 41]  some corner outside every drivable polygon (official DAC, per step)
  valid   [B, K, 41]  the step has object labels: table present and t <= the last observed object step (nobs)
  valid_off [B, K, 41] the step has an off-road label: table present (DAC does not depend on nobs)
Candidate objects (gap / vapp / pos): pass the official TTC direction rule (ahead < 30 deg, or not behind while the
ego is in multiple lanes / non-drivable / an intersection) and were inside the input window (0 <= x <= 64 m,
|y| <= 32 m, ego frame at t0) at their first observed step; the same set for the GT labels and for T-E.

The step labels ignore the scorer's track memory (the first overlap of a track decides whether it counts), so a step
label can be positive where the aggregated official NC / TTC is 1 (over-labelling, logged by the caller).  The
direction test that matters is "step positive -> candidate failed" for overlaps that are a track's first.

DESIGN (not official): gap is the separating-axis distance of two oriented boxes (exact 0 when they overlap, a lower
bound of the Euclidean distance otherwise); vapp uses the object's displacement to the next observed step of the same
track (0 if stopped or not observed again).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..modules import exact_verdict as EV

DTS = (0, 3, 6, 9)                      # TTC projection, 0.1 s steps
NO_OBJ_M = 20.0                         # gap / pos_x when there is no candidate object
WIN_X, WIN_Y = 64.0, 32.0               # the input window (front-only 64 m BEV)


def _sep(c1, u1, e1, c2, u2, e2) -> torch.Tensor:
    """oriented boxes (centre, unit axis, half extents (along, across)) -> signed separating-axis distance: > 0 a lower
    bound of the Euclidean gap, <= 0 overlap (minus the smallest penetration along the four axes)"""
    v1 = torch.stack([-u1[..., 1], u1[..., 0]], -1)
    v2 = torch.stack([-u2[..., 1], u2[..., 0]], -1)
    d = c2 - c1
    out = None
    for a in (u1, v1, u2, v2):
        dist = (d * a).sum(-1).abs()
        r1 = e1[..., 0] * (u1 * a).sum(-1).abs() + e1[..., 1] * (v1 * a).sum(-1).abs()
        r2 = e2[..., 0] * (u2 * a).sum(-1).abs() + e2[..., 1] * (v2 * a).sum(-1).abs()
        g = dist - r1 - r2
        out = g if out is None else torch.maximum(out, g)
    return out


def _tables(t: Dict[str, torch.Tensor], i: int, prefix: str = "ex1_") -> Dict[str, torch.Tensor]:
    """sample i of the packed tables (exact_score.pack_full with prefix) -> exact_verdict's 'ex_*' names"""
    return {"ex_" + kk[len(prefix):]: v[i] for kk, v in t.items() if kk.startswith(prefix)}


@torch.no_grad()
def exact_step_masks(st: torch.Tensor, t: Dict[str, torch.Tensor], prefix: str = "ex1_", ttc_last: int = 40,
                     window: bool = True) -> Dict[str, torch.Tensor]:
    """st [B, K, 41, >=7] tracked paths (exact init, as official_scores) ; t the packed tables -> the dict above.
    Rows whose table is missing (t[prefix + 'valid'] == 0) get valid = 0 and zero labels."""
    b, k = st.shape[:2]
    dev = st.device
    st = st[..., :7].float()
    z = torch.zeros(b, k, 41, device=dev)
    pos0 = torch.zeros(b, k, 41, 2, device=dev)
    pos0[..., 0] = NO_OBJ_M
    out = {"ovl": z.clone(), "ttc": z.clone(), "gap": torch.full_like(z, NO_OBJ_M), "vapp": z.clone(),
           "pos": pos0, "obj": z.clone(), "offroad": z.clone(), "valid": z.clone(), "valid_off": z.clone()}
    ee = EV.dconst((EV.HALF_L, EV.HALF_W), dev)
    eb = EV.dconst((0.0, EV.HALF_W), dev)
    cos30, cos150 = float(np.cos(np.deg2rad(30))), float(np.cos(np.deg2rad(150)))
    steps = torch.arange(41, device=dev)
    for i in range(b):
        if float(t[prefix + "valid"][i]) < 0.5:
            continue
        g = _tables(t, i, prefix)
        s = st[i]                                                                          # [K, 41, 7]
        nondrv, multi, inter_ = EV.ego_areas(s, g["ex_seg"][g["ex_smask"]], g["ex_lane"][g["ex_lmask"]],
                                             g["ex_int"][g["ex_imask"]])
        # exact_verdict.pack_token sets nobs = the largest STORED row time; objects stored once with t = 0 (constant)
        # are copied to every step, so a token whose objects are all constant has nobs = 0 although they exist at
        # every step (the baseline official NC / TTC then ignore them: a label-engine issue reported in report 125).
        # The step labels use the last step that holds any object row instead.
        am = g["ex_amask"]
        last = int(am.any(-1).nonzero().max()) if bool(am.any()) else int(g["ex_nobs"])
        nobs = max(int(g["ex_nobs"]), last)
        out["offroad"][i] = nondrv.float()
        out["valid"][i] = (steps <= nobs).float()[None].expand(k, -1)
        out["valid_off"][i] = 1.0
        m = g["ex_amask"]                                                                  # [50, A]
        na = int(m.sum(0).nonzero().max()) + 1 if m.any() else 0
        if na == 0:
            continue
        m = m[:, :na]
        ag = g["ex_agents"][:, :na]                                                        # [50, A, 8]
        ac, ah, ae = ag[..., :2], ag[..., 2], ag[..., 3:5]
        au = torch.stack([torch.cos(ah), torch.sin(ah)], -1)
        tid, stopped = ag[..., 6].long(), ag[..., 7] > 0.5
        tid = torch.where(m, tid, torch.full_like(tid, -1))
        # candidate tracks: inside the input window at their first observed step (static rows sit at every step)
        inwin = (ac[..., 0] >= 0) & (ac[..., 0] <= WIN_X) & (ac[..., 1].abs() <= WIN_Y)
        if window:
            ntr = int(tid.max()) + 1 if bool(m.any()) else 0
            first = torch.full((max(ntr, 1),), EV.T_OBS, dtype=torch.long, device=dev)
            sidx = torch.arange(EV.T_OBS, device=dev)[:, None].expand_as(tid)
            first.scatter_reduce_(0, tid[m], sidx[m], reduce="amin")
            ok_first = torch.zeros(max(ntr, 1), dtype=torch.uint8, device=dev)          # (bool scatter: no CUDA kernel)
            at_first = m & (sidx == first[tid.clamp(min=0)])
            ok_first.scatter_reduce_(0, tid[at_first], inwin[at_first].to(torch.uint8), reduce="amax")
            cand_tr = ok_first[tid.clamp(min=0)].bool() & m                                # [50, A]
        else:
            cand_tr = m
        c, u = EV._ego_box(s)                                                              # [K, 41, 2]
        ra = s[..., :2]
        spd = torch.hypot(s[..., 3], s[..., 4])                                            # [K, 41]
        area = multi | nondrv
        tmax = torch.arange(EV.T_OBS, device=dev) <= nobs
        # ---- overlap (NC rule) at steps 1..40
        ti = torch.arange(1, 41, device=dev)
        A_ = (lambda x: x[ti][None])
        hit = EV._overlap(c[:, 1:, None], u[:, 1:, None], ee, A_(ac), A_(au), A_(ae)) & A_(m) & tmax[ti][None, :, None]
        rel = EV._rel(ra[:, 1:, None], u[:, 1:, None], A_(ac))
        behind = rel < cos150
        front = EV._overlap(c[:, 1:, None] + u[:, 1:, None] * EV.HALF_L, u[:, 1:, None], eb, A_(ac), A_(au), A_(ae))
        ego_stop = (spd[:, 1:] <= 5e-2)[..., None]
        fault = ~ego_stop & (A_(stopped) | (~behind & (front | area[:, 1:, None])))
        out["ovl"][i, :, 1:] = (hit & fault).any(-1).float()
        # ---- TTC rule, gap, nearest relevant object
        best = torch.full((k, 41), float("inf"), device=dev)
        bslot = torch.zeros(k, 41, dtype=torch.long, device=dev)
        bstep = torch.zeros(k, 41, dtype=torch.long, device=dev)
        ttc_hit = torch.zeros(k, 41, dtype=torch.bool, device=dev)
        for dd in DTS:
            ct = steps + dd
            okt = (ct >= 1) & (ct < EV.T_OBS) & (ct <= nobs)
            cc = ct.clamp(max=EV.T_OBS - 1)
            B_ = (lambda x: x[cc][None])
            cs = c + u * (spd * dd * 0.1)[..., None]
            hh = EV._overlap(cs[:, :, None], u[:, :, None], ee, B_(ac), B_(au), B_(ae)) & B_(m)
            hh &= (okt[None] & (spd >= EV.STOPPED) & (steps <= ttc_last)[None])[..., None]
            r = EV._rel(ra[:, :, None], u[:, :, None], B_(ac))
            f = (r > cos30) | ((area | inter_)[..., None] & ~(r < cos150))
            ttc_hit |= (hh & f).any(-1)
            # nearest TTC-relevant object (direction rule f), any speed
            sep = _sep(cs[:, :, None], u[:, :, None], ee, B_(ac), B_(au), B_(ae))          # [K, 41, A]
            ok = B_(cand_tr) & f & okt[None, :, None]
            sep = torch.where(ok, sep, torch.full_like(sep, float("inf")))
            v, j = sep.min(-1)
            upd = v < best
            best = torch.where(upd, v, best)
            bslot = torch.where(upd, j, bslot)
            bstep = torch.where(upd, cc[None].expand(k, -1), bstep)
        out["ttc"][i] = ttc_hit.float()
        has = torch.isfinite(best)
        out["obj"][i] = has.float()
        out["gap"][i] = torch.where(has, best.clamp(max=NO_OBJ_M), torch.full_like(best, NO_OBJ_M))
        # object centre at its step, in the ego frame of step t
        oc = ac[bstep, bslot]                                                              # [K, 41, 2]
        d = oc - ra
        hcos, hsin = torch.cos(s[..., 2]), torch.sin(s[..., 2])
        px = d[..., 0] * hcos + d[..., 1] * hsin
        py = -d[..., 0] * hsin + d[..., 1] * hcos
        nopos = torch.zeros_like(d)
        nopos[..., 0] = NO_OBJ_M
        out["pos"][i] = torch.where(has[..., None], torch.stack([px, py], -1), nopos)
        # closing speed: ego box-centre velocity (central difference) vs the same track's displacement to step + 1
        ce = c
        ve = torch.zeros_like(ce)
        ve[:, 1:-1] = (ce[:, 2:] - ce[:, :-2]) / 0.2
        ve[:, 0] = (ce[:, 1] - ce[:, 0]) / 0.1
        ve[:, -1] = (ce[:, -1] - ce[:, -2]) / 0.1
        tsel = tid[bstep, bslot]                                                           # [K, 41]
        nxt = (bstep + 1).clamp(max=EV.T_OBS - 1)
        same = (tid[nxt] == tsel[..., None]) & m[nxt]                                      # [K, 41, A]
        has_n = same.any(-1) & (bstep + 1 < EV.T_OBS) & ((bstep + 1) <= nobs)
        jn = same.float().argmax(-1)
        vo = torch.where((has_n & ~stopped[bstep, bslot])[..., None], (ac[nxt, jn] - oc) / 0.1, torch.zeros_like(oc))
        nvec = oc - ce
        nvec = nvec / nvec.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        out["vapp"][i] = torch.where(has, ((ve - vo) * nvec).sum(-1), torch.zeros_like(best))
    return out


def stride2(lab: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """41 steps (0.1 s) -> the 21 refiner times (0, 0.2, .., 4.0 s), max-pooled over the odd step that the
    subsampling would drop (a 0.1 s positive at an odd step stays visible: bin max over (t - 0.1, t]), continuous
    channels at the even step"""
    out = {}
    for kk, v in lab.items():
        if kk in ("ovl", "ttc", "offroad"):
            ev = v[..., 0::2]
            od = torch.cat([torch.zeros_like(v[..., :1]), v[..., 1::2]], -1)                 # step 2j - 1 -> slot j
            out[kk] = torch.maximum(ev, od)
        elif kk == "pos":
            out[kk] = v[..., 0::2, :]
        else:
            out[kk] = v[..., 0::2]
    return out
