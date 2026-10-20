"""km 10/19 (C): candidate scorer -- the verdict of every one of the model's OWN candidates (anchor + offset, tracked
by the official-equivalent torch LQR port), trained on the EXACT verdict of exactly those tracked candidates
(modules/exact_verdict.py: official NC / DAC / TTC / progress on the metric-cache geometry).  Nothing in it is an
anchor label: v2's im / sim heads (anchor-label scores) are not read, the anchor query is not an input.

Per candidate k (all K = 256), from the tracked states st [B, K, 41, 7] (detached):
  step tokens at the 9 times 0, 0.5, .., 4.0 s:
    kinematics  x, y (/ 32 m), cos h, sin h, speed (/ 10), accel (/ 4), Fourier(x, y)
    bev         bev_embed sampled bilinearly at the box centre and its 4 corners (5 x C -> C)
    read-out    (use_fields) the predicted corridor along the footprint over that 0.5 s window: min corner SDF (m),
                max footprint occupancy, max TTC-shifted occupancy (dt 0.3 / 0.6 / 0.9 s), as in corridor_select
  -> MLP over the 9 step tokens -> candidate token -> cross-attention to the det/motion agent-mode tokens (detached,
     as the corridor head's AgentCrossAttn) -> self-attention across the K candidates -> heads:
     nc (3 classes: NC = 1 / 0.5 / 0), dac (logit of a DAC failure), ttc (logit of a TTC failure), progress (m)
Loss (cand_scorer_loss): CE + BCE + BCE unweighted (calibrated probabilities: the selector multiplies them),
smooth-L1 on progress / 10 m, mean over every candidate of the samples with exact GT.
Selection (expected_pdms): E[PDMS_k] = E[NC] * P(DAC) * (5 EP + 5 P(TTC) + 2 C) / 12, C = exact comfort of the
tracked states, EP = progress / max_j (progress_j * E[NC_j] * P(DAC_j)) (the official EP normalisation by the best
feasible progress; 1 below 0.1 m).  With a perfect verdict this is the furthest comfortable safe candidate
(navtest, 32 m ep30 candidates, exact GT: 97.58 vs the best-of-256 97.74).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import corridor_head as CH
from .corridor_head import _fourier
from .exact_verdict import CENTRE, HALF_L, dconst, exact_comfort

STEPS = torch.arange(0, 41, 5)                     # 0, 0.5, .., 4.0 s
# per-candidate heads (km 10/19 C v3): every official term of exact_score.official_scores that is not computable
# from the candidate alone; 3-class heads = value 1 / 0.5 / 0, 1-wide heads = logit of a FAILURE (value 0)
HEADS = {"nc1": 3, "dac1": 1, "ddc1": 3, "ttc1": 1,
         "nc2": 3, "dac2": 1, "ddc2": 3, "tlc2": 1, "ttc2": 1, "lk2": 1, "hc2": 1, "prog": 1}
TRI = ("nc1", "ddc1", "nc2", "ddc2")
BIN = ("dac1", "ttc1", "dac2", "tlc2", "ttc2", "lk2", "hc2")


def _bev_sample(bev: torch.Tensor, bev_h: int, bev_w: int, fwd_m: float, pts: torch.Tensor) -> torch.Tensor:
    """bev [B, H*W, C] (row = forward 0..fwd_m, col = SSR x right -32..32 m); pts [B, N, 2] NAVSIM ego (x fwd, y left)
    -> [B, N, C] bilinear (zeros outside)"""
    b, n, c = bev.shape
    img = bev.transpose(1, 2).reshape(b, c, bev_h, bev_w)
    u = (-pts[..., 1] + 32.0) / 64.0 * 2 - 1
    v = pts[..., 0] / fwd_m * 2 - 1
    g = torch.stack([u, v], -1)[:, :, None]                                            # [B, N, 1, 2]
    return F.grid_sample(img, g.to(img.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)[..., 0].transpose(1, 2)


def field_readout(st: torch.Tensor, sdf: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
    """predicted corridor along the tracked footprints, per 0.1 s step -> [B, K, 41, 3]: (min corner SDF (m),
    max occupancy under the box (15 points), max occupancy under the box shifted by speed * dt, dt 0.3/0.6/0.9 s,
    vs the plane of t + dt); step 0 = 0 occupancy.  sdf [B, H, 200], occ [B, P, H, 200] probabilities."""
    b, k = st.shape[:2]
    dev = st.device
    x, y, h, spd = st[..., 0], st[..., 1], st[..., 2], torch.hypot(st[..., 3], st[..., 4])
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    cx = dconst((CH.FRONT, CH.FRONT, -CH.REAR, -CH.REAR), dev)
    cy = dconst((CH.HALF_W, -CH.HALF_W, -CH.HALF_W, CH.HALF_W), dev)
    pc = torch.stack([x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy], -1)       # [B, K, 41, 4, 2]
    v, inr = CH._sample(sdf.float()[:, None], pc.reshape(b, 1, -1, 2))
    sdf_min = torch.where(inr, v, torch.full_like(v, CH.CLIP)).reshape(b, k, 41, 4).amin(-1)
    fx, fy = torch.meshgrid(torch.linspace(-CH.REAR, CH.FRONT, 5, device=dev),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=dev), indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    npl = occ.shape[1]
    occ1 = ttc1 = None                                     # out of place: the fields carry gradient
    steps = torch.arange(1, 41, device=dev)
    for dd in (0, 3, 6, 9):
        sh = (spd[:, :, 1:] * dd * 0.1)[..., None]
        q = torch.stack([x[:, :, 1:, None] + c[:, :, 1:] * (fx + sh) - s[:, :, 1:] * fy,
                         y[:, :, 1:, None] + s[:, :, 1:] * (fx + sh) + c[:, :, 1:] * fy], -1)       # [B, K, 40, 15, 2]
        q = q.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2)
        pl = (steps + dd - 1).clamp(max=npl - 1) if npl >= 40 else ((steps + dd).clamp(max=40) - 1) // (40 // npl)
        val, ti = CH._sample(occ.float()[:, pl], q)
        val = torch.where(ti, val, torch.zeros_like(val)).reshape(b, 40, k, 15).amax(-1).permute(0, 2, 1)
        if dd == 0:
            occ1 = val
        else:
            ttc1 = val.clamp(min=0) if ttc1 is None else torch.maximum(ttc1, val)
    z = sdf_min.new_zeros(b, k, 1)
    return torch.stack([sdf_min, torch.cat([z, occ1], 2), torch.cat([z, ttc1], 2)], -1)


class CandidateScorer(nn.Module):
    def __init__(self, cin: int, bev_h: int, bev_w: int, fwd_m: float, hid: int = 256, use_fields: bool = True,
                 agent_k: int = 32, nf: int = 8, layers: int = 2):
        super().__init__()
        self.bev_h, self.bev_w, self.fwd_m = bev_h, bev_w, fwd_m
        self.use_fields, self.agent_k, self.nf = use_fields, agent_k, nf
        nkin = 6 + 4 * nf
        self.kin = nn.Sequential(nn.Linear(nkin, hid), nn.ReLU(inplace=True), nn.Linear(hid, hid))
        self.bev = nn.Sequential(nn.Linear(5 * cin, hid), nn.ReLU(inplace=True), nn.Linear(hid, hid))
        self.fld = nn.Sequential(nn.Linear(3, hid), nn.ReLU(inplace=True), nn.Linear(hid, hid)) if use_fields else None
        self.step_pe = nn.Parameter(torch.zeros(len(STEPS), hid))
        self.merge = nn.Sequential(nn.Linear(len(STEPS) * hid, 2 * hid), nn.ReLU(inplace=True), nn.Linear(2 * hid, hid))
        self.ag_tok = nn.Sequential(nn.Linear(cin + 10 + 7 + 9 * 4 * 16 + 1, hid), nn.ReLU(inplace=True), nn.Linear(hid, hid))
        self.cross = nn.ModuleList([nn.MultiheadAttention(hid, 8, batch_first=True) for _ in range(layers)])
        self.selfa = nn.ModuleList([nn.MultiheadAttention(hid, 8, batch_first=True) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hid, 2 * hid), nn.ReLU(inplace=True), nn.Linear(2 * hid, hid))
                                  for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(hid) for _ in range(3)]) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(inplace=True), nn.Linear(hid, sum(HEADS.values())))
        self.tok_head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(inplace=True), nn.Linear(hid, 2))  # eff1, eff2

    def _agents(self, det: Dict[str, torch.Tensor]) -> torch.Tensor:
        """det/motion top-k agent-mode tokens (detached), the corridor head's AgentCrossAttn features"""
        cls = det["all_cls_scores"][-1].float().sigmoid().detach()
        box = det["all_bbox_preds"][-1].float().detach()
        traj, tcls, mh = (det[k].float().detach() for k in ("traj_preds", "traj_cls_preds", "motion_hidden"))
        top = cls.amax(-1).topk(min(self.agent_k, cls.shape[1]), dim=1).indices

        def g(x):
            return torch.gather(x, 1, top.view(*top.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))

        cls, box, traj, tcls, mh = g(cls), g(box), g(traj), g(tcls), g(mh)
        m = traj.shape[2]
        c0 = box[..., :2] / 32.0                                                            # SSR (x right, y fwd)
        pts = torch.cat([c0[:, :, None, None].expand(-1, -1, m, 1, 2), c0[:, :, None, None] + traj.cumsum(3) / 32.0], 3)
        bcode = torch.cat([c0, box[..., 2:8], box[..., 8:10] / 10.0], -1)
        feat = torch.cat([mh, bcode[:, :, None].expand(-1, -1, m, -1), cls[:, :, None].expand(-1, -1, m, -1),
                          _fourier(pts, 16).flatten(-2), tcls[..., None]], -1)
        return self.ag_tok(feat.flatten(1, 2))                                             # [B, k*M, hid]

    def forward(self, st: torch.Tensor, bev: torch.Tensor, det: Dict[str, torch.Tensor],
                fields: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """st [B, K, 41, 7] tracked candidates (detached) -> {cand_nc_logits [B, K, 3], cand_dac_logit, cand_ttc_logit,
        cand_progress [B, K]}"""
        b, k = st.shape[:2]
        dev = st.device
        idx = STEPS.to(dev)
        s9 = st[:, :, idx]                                                                  # [B, K, 9, 7]
        x, y, h = s9[..., 0], s9[..., 1], s9[..., 2]
        spd, acc = torch.hypot(s9[..., 3], s9[..., 4]), s9[..., 5]
        # Fourier in the SSR frame (x right, y fwd) / 32 m, as the agent tokens
        kin = torch.cat([torch.stack([x / 32, y / 32, torch.cos(h), torch.sin(h), spd / 10, acc / 4], -1),
                         _fourier(torch.stack([-y, x], -1) / 32.0, self.nf)], -1)
        tok = self.kin(kin)                                                                 # [B, K, 9, hid]
        u = torch.stack([torch.cos(h), torch.sin(h)], -1)
        vv = torch.stack([-u[..., 1], u[..., 0]], -1)
        cen = torch.stack([x, y], -1) + CENTRE * u
        off = dconst(((0, 0), (HALF_L, CH.HALF_W), (HALF_L, -CH.HALF_W), (-HALF_L, -CH.HALF_W), (-HALF_L, CH.HALF_W)),
                     dev, st.dtype)
        pts = cen[..., None, :] + u[..., None, :] * off[:, :1] + vv[..., None, :] * off[:, 1:]   # [B, K, 9, 5, 2]
        smp = _bev_sample(bev, self.bev_h, self.bev_w, self.fwd_m, pts.reshape(b, -1, 2))       # [B, K*45, C]
        tok = tok + self.bev(smp.reshape(b, k, len(STEPS), -1))
        if self.fld is not None:
            assert fields is not None, "use_fields needs the corridor head outputs"
            ro = field_readout(st, fields["corridor_sdf"], torch.sigmoid(fields["corridor_occ"].float()))   # [B, K, 41, 3]
            # step 0, then the windows (t - 0.5 s, t]: SDF min, occupancies max
            w_sdf = torch.cat([ro[:, :, :1, :1], ro[:, :, 1:, :1].reshape(b, k, 8, 5, 1).amin(3)], 2)
            w_occ = torch.cat([ro[:, :, :1, 1:], ro[:, :, 1:, 1:].reshape(b, k, 8, 5, 2).amax(3)], 2)
            tok = tok + self.fld(torch.cat([w_sdf / CH.CLIP, w_occ], -1))
        q = self.merge((tok + self.step_pe).flatten(2))                                         # [B, K, hid]
        kv = self._agents(det) if det is not None else None
        for ca, sa, ff, (n1, n2, n3) in zip(self.cross, self.selfa, self.ffn, self.norms):
            if kv is not None:
                q = n1(q + ca(q, kv, kv, need_weights=False)[0])
            q = n2(q + sa(q, q, q, need_weights=False)[0])
            q = n3(q + ff(q))
        o = self.head(q).float()
        out, j = {}, 0
        for nm, w in HEADS.items():
            out["cand_" + nm] = o[..., j:j + w] if w > 1 else o[..., j]
            j += w
        out["cand_prog"] = 10.0 * out["cand_prog"]
        e = 10.0 * self.tok_head(q.mean(1)).float()                                          # [B, 2]
        out["cand_eff1"], out["cand_eff2"] = e[:, 0], e[:, 1]
        return out


def cand_scorer_loss(pred: Dict[str, torch.Tensor], ex: Dict[str, torch.Tensor], prog_weight: float = 1.0):
    """ex = exact_score.official_scores(...) of the same tracked candidates (NaN where the token has no table) ->
    (loss, logs).  3-class heads: class 0 / 1 / 2 = value 1 / 0.5 / 0; binary heads predict a FAILURE (value 0)."""
    loss, logs = 0.0, {}
    for nm in TRI + BIN + ("prog",):
        key = nm if nm != "prog" else "prog1"
        y = ex[key]
        m = torch.isfinite(y)
        if not m.any():
            logs[f"cand/{nm}"] = y.new_tensor(0.0)
            continue
        if nm in TRI:
            cls = torch.where(y >= 1, 0, torch.where(y > 0, 1, 2)).long()
            l = F.cross_entropy(pred["cand_" + nm][m], cls[m])
            with torch.no_grad():
                logs[f"cand/acc_{nm}"] = ((pred["cand_" + nm][m].argmax(-1) > 0) == (cls[m] > 0)).float().mean()
                logs[f"cand/pos_{nm}"] = (cls[m] > 0).float().mean()
        elif nm in BIN:
            yf = (y[m] < 1).float()
            l = F.binary_cross_entropy_with_logits(pred["cand_" + nm][m], yf)
            with torch.no_grad():
                logs[f"cand/acc_{nm}"] = ((pred["cand_" + nm][m] > 0).float() == yf).float().mean()
                logs[f"cand/pos_{nm}"] = yf.mean()
        else:
            l = prog_weight * F.smooth_l1_loss(pred["cand_prog"][m] / 10, y[m] / 10, beta=0.5)
        loss = loss + l
        logs[f"cand/{nm}"] = l.detach()
    for nm in ("eff1", "eff2"):                                                  # token-level: PDM-Closed effective progress
        y = ex[nm][:, 0]
        m = torch.isfinite(y)
        if m.any():
            l = prog_weight * F.smooth_l1_loss(pred["cand_" + nm][m] / 10, y[m] / 10, beta=0.5)
            loss = loss + l
            logs[f"cand/{nm}"] = l.detach()
    return loss, logs


def _e3(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), -1)
    return p[..., 0] + 0.5 * p[..., 1]


def _p(logit: torch.Tensor) -> torch.Tensor:
    return 1 - torch.sigmoid(logit.float())


@torch.no_grad()
def expected_official(pred: Dict[str, torch.Tensor], st: torch.Tensor) -> Dict[str, torch.Tensor]:
    """-> {pdms1, epdms2 [B, K] expected official scores, c1 [B, K]}.  The official formulas of exact_score with every
    predicted term replaced by its expectation; DESIGN (not official): the terms are treated as independent and EP uses
    the predicted progress / PDM-Closed effective progress as point values.  EC is not in epdms2 (excluded by design)."""
    prog = pred["cand_prog"].float().clamp(min=0)
    c1 = exact_comfort(st[..., :7]).float()
    m1 = _e3(pred["cand_nc1"]) * _p(pred["cand_dac1"]) * _e3(pred["cand_ddc1"])
    eff1 = pred["cand_eff1"].float().clamp(min=0)[:, None]
    masked = m1 * prog
    norm1 = torch.maximum(masked, eff1)
    ep1 = torch.where(norm1 > 5.0, masked / norm1.clamp(min=1e-6), torch.ones_like(prog))   # official yaml: 5 m (km 10/20)
    pdms1 = m1 * (5 * ep1 + 5 * _p(pred["cand_ttc1"]) + 2 * c1) / 12
    m2 = _e3(pred["cand_nc2"]) * _p(pred["cand_dac2"]) * _e3(pred["cand_ddc2"]) * _p(pred["cand_tlc2"])
    eff2 = pred["cand_eff2"].float().clamp(min=0)[:, None]
    norm2 = torch.maximum(m2 * prog, eff2)
    ep2 = torch.where(norm2 > 5.0, (prog / norm2.clamp(min=1e-6)).clamp(0, 1), torch.ones_like(prog))
    epdms2 = m2 * (5 * ep2 + 5 * _p(pred["cand_ttc2"]) + 2 * _p(pred["cand_lk2"]) + 2 * _p(pred["cand_hc2"])) / 14
    return {"pdms1": pdms1, "epdms2": epdms2, "c1": c1}


@lru_cache(maxsize=None)
def _ec_ops(n: int, dt: float = 0.1):
    """the official two-frame extended comfort features (pdm_comfort_metrics.extract_features) as [n, n] filters
    (out = x @ M): jerk = savgol(|a|, 8, 2) -> d/dt (window 15, poly 2); yaw accel = d2/dt2 of the unwrapped heading
    (window 5, poly 2)"""
    from scipy.signal import savgol_filter
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as cm
    eye, tp = np.eye(n), np.arange(n) * dt
    acc = savgol_filter(eye, polyorder=2, window_length=min(8, n), axis=-1)
    jerk = acc @ cm._approximate_derivatives(eye, tp, window_length=min(15, n), poly_order=2, deriv_order=1)
    yaw_acc = cm._approximate_derivatives(eye, tp, poly_order=2, deriv_order=2)
    return torch.tensor(jerk, dtype=torch.float64), torch.tensor(yaw_acc, dtype=torch.float64)


_EC_OPS_DEV = {}


def ec_roughness(st: torch.Tensor) -> torch.Tensor:
    """km 10/20 (C v3.1): st [..., n, >=7] tracked candidates -> R [...] = rms(jerk) / 0.5 + rms(yaw accel) / 0.1, the
    candidate's own official EC features normalised by the EC thresholds (EC compares these features of two
    consecutive plans; a smooth plan keeps them small).  DESIGN (not official): a single-frame proxy of EC."""
    key = (st.shape[-2], st.device)
    if key not in _EC_OPS_DEV:
        _EC_OPS_DEV[key] = tuple(m.to(st.device) for m in _ec_ops(st.shape[-2]))
    mj, my = _EC_OPS_DEV[key]
    s = st.double()
    h = s[..., 2]
    hu = h.clone()
    hu[..., 1:] = h[..., 1:] - 2 * np.pi * torch.cumsum(torch.round(torch.diff(h, dim=-1) / (2 * np.pi)), -1)
    j = torch.hypot(s[..., 5], s[..., 6]) @ mj
    ya = hu @ my
    return (j.pow(2).mean(-1).sqrt() / 0.5 + ya.pow(2).mean(-1).sqrt() / 0.1).float()


def select_official(ex: Dict[str, torch.Tensor], eps: float, lam: float = 0.0, rough: torch.Tensor = None) -> torch.Tensor:
    """argmax of expected EPDMS (- lam * rough, km 10/20 C v3.1: EC is not in the expected EPDMS) among the
    candidates within eps of the best expected PDMS -> index [B] (eps = 0: the expected-PDMS argmax alone)"""
    p1 = ex["pdms1"]
    keep = p1 >= p1.amax(1, keepdim=True) - eps
    val = ex["epdms2"] if lam == 0 or rough is None else ex["epdms2"] - lam * rough
    return torch.where(keep, val, torch.full_like(p1, float("-inf"))).argmax(1)
