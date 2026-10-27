"""R-E (per-step risk), R-C (corrector) + decoder, R-J (row-wise judge) of refiner v6c (report 124 §2).

Shapes (B batch, R rows = paths per sample, T = 21 times 0, 0.2, .., 4.0 s):
  states S [B, R, 41, 7] (x, y, h, vx, vy, ax, ay; rear axle, ego frame at t0, 0.1 s; index 0 = now)
  E       [B, R, T, 12]  learned 7 (ovl logit, ttc logit, gap m, vapp m/s, px m, py m, offroad logit) + road margin 5
  M       [B, R, T, 5]   in-range mask of the 5 footprint points (centre + 4 corners)

Frames: BEV / corridor fields have row = forward (0 .. 64 m), col = SSR x right (-32 .. 32 m), as
modules/candidate_scorer._bev_sample and modules/corridor_head._sample; path points are NAVSIM ego (x fwd, y left).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import corridor_head as CH
from ..modules.candidate_scorer import _bev_sample, field_readout
from ..modules.corridor_head import _fourier
from ..modules.exact_verdict import CENTRE, HALF_L, dconst

T21 = torch.arange(0, 41, 2)                      # the refiner times among the 41 tracked states
N_LEARN = 7                                       # learned E channels
E_NAMES = ("ovl", "ttc", "gap", "vapp", "px", "py", "offroad")
LOGIT_CH = (0, 1, 6)
CONT_CH = (2, 3, 4, 5)
CONT_NORM = (10.0, 5.0, 10.0, 10.0)               # gap, px, py in 10 m units, vapp in 5 m/s units (report 124 §2-3)
CONT_CLIP = 4.0                                   # normalised continuous values are clipped to +-4 (loss and input)


def cont_norm(x: torch.Tensor) -> torch.Tensor:
    """[..., 4] continuous channels in m / m/s -> normalised, clipped units"""
    return (x / x.new_tensor(CONT_NORM)).clamp(-CONT_CLIP, CONT_CLIP)


def _mlp(a: int, b: int, hid: Optional[int] = None) -> nn.Sequential:
    hid = hid or b
    return nn.Sequential(nn.Linear(a, hid), nn.ReLU(inplace=True), nn.Linear(hid, b))


def footprint5(s: torch.Tensor) -> torch.Tensor:
    """states [..., 7] -> 5 footprint points [..., 5, 2] (box centre, FL, FR, RR, RL), NAVSIM ego frame"""
    h = s[..., 2]
    u = torch.stack([torch.cos(h), torch.sin(h)], -1)
    v = torch.stack([-u[..., 1], u[..., 0]], -1)
    cen = s[..., :2] + CENTRE * u
    off = dconst(((0.0, 0.0), (HALF_L, CH.HALF_W), (HALF_L, -CH.HALF_W), (-HALF_L, -CH.HALF_W), (-HALF_L, CH.HALF_W)),
                 s.device, s.dtype)
    return cen[..., None, :] + u[..., None, :] * off[:, :1] + v[..., None, :] * off[:, 1:]


def path_reads(S: torch.Tensor, bev: torch.Tensor, sdf: torch.Tensor, occ_prob: torch.Tensor, bev_h: int,
               bev_w: int, fwd_m: float) -> Dict[str, torch.Tensor]:
    """parameter-free reads along the paths.  S [B, R, 41, 7]; bev [B, H*W, C]; sdf [B, Hf, 200] (m);
    occ_prob [B, P, Hf, 200] PROBABILITIES (report 124 §3-1 exchange convention) ->
      ro [B, R, T, 3] field_readout at the 21 times (min corner SDF / CLIP, max occupancy, max TTC occupancy),
      road [B, R, T, 5] SDF at the 5 footprint points / CLIP, M [B, R, T, 5] in range (and not behind the ego),
      smp [B, R, T, 5 C] BEV samples at the 5 points, kin [B, R, T, 6] kinematics"""
    b, r = S.shape[:2]
    t = T21.to(S.device)
    ro = field_readout(S, sdf, occ_prob)[:, :, t]                                          # [B, R, T, 3]
    ro = torch.cat([ro[..., :1] / CH.CLIP, ro[..., 1:]], -1)
    s21 = S[:, :, t]                                                                       # [B, R, T, 7]
    pts = footprint5(s21)                                                                  # [B, R, T, 5, 2]
    val, inr = CH._sample(sdf.float()[:, None], pts.reshape(b, 1, -1, 2))
    inr = inr & (pts.reshape(b, 1, -1, 2)[..., 0] >= 0)
    road = torch.where(inr, val, torch.full_like(val, CH.CLIP)).reshape(b, r, len(t), 5) / CH.CLIP
    M = inr.reshape(b, r, len(t), 5).float()
    smp = _bev_sample(bev, bev_h, bev_w, fwd_m, pts.reshape(b, -1, 2)).reshape(b, r, len(t), -1)
    spd = torch.hypot(s21[..., 3], s21[..., 4])
    kin = torch.stack([s21[..., 0] / 32, s21[..., 1] / 32, torch.cos(s21[..., 2]), torch.sin(s21[..., 2]),
                       spd / 10, s21[..., 5] / 4], -1)
    return {"ro": ro, "road": road, "M": M, "smp": smp, "kin": kin}


def agent_features(det: Dict[str, torch.Tensor], k: int) -> torch.Tensor:
    """det/motion top-k agent-mode features (detached), as the corridor head's AgentCrossAttn / the scorer's _agents
    build them -> [B, k*M, cin + 10 + 7 + 9*4*16 + 1] (no parameters: each consumer owns its own tokenizer)"""
    cls = det["all_cls_scores"][-1].float().sigmoid().detach()
    box = det["all_bbox_preds"][-1].float().detach()
    traj, tcls, mh = (det[kk].float().detach() for kk in ("traj_preds", "traj_cls_preds", "motion_hidden"))
    top = cls.amax(-1).topk(min(k, cls.shape[1]), dim=1).indices

    def g(x):
        return torch.gather(x, 1, top.view(*top.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))

    cls, box, traj, tcls, mh = g(cls), g(box), g(traj), g(tcls), g(mh)
    m = traj.shape[2]
    c0 = box[..., :2] / 32.0
    pts = torch.cat([c0[:, :, None, None].expand(-1, -1, m, 1, 2), c0[:, :, None, None] + traj.cumsum(3) / 32.0], 3)
    bcode = torch.cat([c0, box[..., 2:8], box[..., 8:10] / 10.0], -1)
    feat = torch.cat([mh, bcode[:, :, None].expand(-1, -1, m, -1), cls[:, :, None].expand(-1, -1, m, -1),
                      _fourier(pts, 16).flatten(-2), tcls[..., None]], -1)
    return feat.flatten(1, 2)


class RiskNet(nn.Module):
    """R-E (student) and T-E (teacher, use_agents=False, n_area = 3 path-wise GT area flags) share this class.
    Per time: token = BEV-sample MLP + read-out MLP + time embedding; -> 1 cross-attention to the agent tokens
    (student only) -> 1 temporal transformer block -> head of the 7 learned channels.  H = the tokens after the
    temporal block (the ablation input of R-C, never its default input)."""

    def __init__(self, cin: int = 256, hid: int = 192, agent_dim: int = 0, n_area: int = 0, heads: int = 8):
        super().__init__()
        self.n_area = n_area
        self.bev = _mlp(5 * cin, hid)
        self.feat = _mlp(3 + 5 + 5 + 6 + n_area, hid)
        self.time_pe = nn.Parameter(torch.zeros(len(T21), hid))
        nn.init.normal_(self.time_pe, std=0.02)
        self.ag_tok = _mlp(agent_dim, hid) if agent_dim > 0 else None
        self.cross = nn.MultiheadAttention(hid, heads, batch_first=True) if agent_dim > 0 else None
        self.n_cross = nn.LayerNorm(hid) if agent_dim > 0 else None
        self.temporal = nn.TransformerEncoderLayer(hid, heads, 2 * hid, dropout=0.0, batch_first=True)
        self.head = _mlp(hid, N_LEARN)

    def forward(self, reads: Dict[str, torch.Tensor], agents: Optional[torch.Tensor] = None,
                area: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """reads of path_reads; agents [B, N, agent_dim] (detached features); area [B, R, T, n_area] ->
        {E [B, R, T, 12] (learned 7 raw: logits / metres, then road 5), H [B, R, T, hid]}"""
        b, r, t = reads["kin"].shape[:3]
        x = torch.cat([reads["ro"], reads["road"], reads["M"], reads["kin"]] + ([area] if self.n_area else []), -1)
        tok = self.bev(reads["smp"].to(self.bev[0].weight.dtype)) + self.feat(x) + self.time_pe
        tok = tok.reshape(b, r * t, -1)
        if self.cross is not None and agents is not None:
            kv = self.ag_tok(agents)
            tok = self.n_cross(tok + self.cross(tok, kv, kv, need_weights=False)[0])
        h = self.temporal(tok.reshape(b * r, t, -1)).reshape(b, r, t, -1)
        o = self.head(h).float()
        o = torch.cat([o[..., :2], o[..., 2:6] * o.new_tensor(CONT_NORM), o[..., 6:7]], -1)
        return {"E": torch.cat([o, reads["road"].float()], -1), "H": h}


def e_features(E: torch.Tensor) -> torch.Tensor:
    """E [.., 12] (raw) -> the bounded features R-C / R-J read: probabilities of the 3 logits, continuous normalised,
    road margins as they are (already / CLIP)"""
    p = torch.sigmoid(E[..., [0, 1, 6]])
    c = cont_norm(E[..., 2:6])
    return torch.cat([p, c, E[..., 7:]], -1)


def geom_features(S: torch.Tensor) -> torch.Tensor:
    """S [B, R, 41, 7] -> [B, R, T, 7] (speed / 10, accel / 4, curvature, cos h, sin h, x / 32, y / 32) at the 21 times"""
    s = S[:, :, T21.to(S.device)]
    spd = torch.hypot(s[..., 3], s[..., 4])
    hd = torch.diff(S[..., 2], dim=-1, prepend=S[..., :1, 2])
    hd = torch.atan2(torch.sin(hd), torch.cos(hd))[:, :, T21.to(S.device)] / 0.1
    kappa = (hd / spd.clamp(min=1.0)).clamp(-1, 1)
    return torch.stack([spd / 10, s[..., 5] / 4, kappa, torch.cos(s[..., 2]), torch.sin(s[..., 2]),
                        s[..., 0] / 32, s[..., 1] / 32], -1)


class Corrector(nn.Module):
    """R-C: per-time tokens c_t = MLP([E_t features 12, M_t 5, geometry_t 7, Fourier(t)]); 6 longitudinal + 6 lateral
    control queries (+ ego status / command) read the 21 tokens by attention; the last layer is zero-initialised,
    so Delta = 0 and tau_v6 == tau_base bitwise at the start.  use_h: the H ablation arms (E+H / H-only)."""

    def __init__(self, hid: int = 192, h_dim: int = 192, use_e: bool = True, use_h: bool = False, layers: int = 2,
                 heads: int = 8, n_cmd: int = 4):
        super().__init__()
        self.use_e, self.use_h = use_e, use_h
        nin = (12 + 5 + 7 if use_e else 7) + 8
        self.tok = _mlp(nin, hid)
        self.h_in = nn.Linear(h_dim, hid) if use_h else None
        self.query = nn.Parameter(torch.zeros(12, hid))
        nn.init.normal_(self.query, std=0.02)
        self.ego = _mlp(4 + n_cmd, hid)
        self.attn = nn.ModuleList([nn.MultiheadAttention(hid, heads, batch_first=True) for _ in range(layers)])
        self.ffn = nn.ModuleList([_mlp(hid, hid, 2 * hid) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.out = nn.Linear(hid, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, E: torch.Tensor, M: torch.Tensor, S: torch.Tensor, ego: torch.Tensor, cmd: torch.Tensor,
                H: Optional[torch.Tensor] = None) -> torch.Tensor:
        """E [B, T, 12], M [B, T, 5], S [B, 41, 7] (tau_base states), ego [B, 4], cmd [B, n_cmd]; all DETACHED by
        the caller -> raw controls [B, 12] (z_lon 6, w_lat 6)"""
        t = T21.to(E.device).to(E.dtype) * 0.1 / 4.0
        tf = _fourier(torch.stack([t, t], -1), 2)                                          # [T, 8]
        g = geom_features(S[:, None])[:, 0]
        parts = ([e_features(E), M] if self.use_e else []) + [g, tf[None].expand(E.shape[0], -1, -1)]
        tok = self.tok(torch.cat(parts, -1))
        if self.use_h:
            assert H is not None, "the H arm needs R-E's H (detached)"
            tok = tok + self.h_in(H)
        q = self.query[None].expand(E.shape[0], -1, -1) + self.ego(torch.cat([ego, cmd], -1))[:, None]
        for at, ff, n1, n2 in zip(self.attn, self.ffn, self.n1, self.n2):
            q = n1(q + at(q, tok, tok, need_weights=False)[0])
            q = n2(q + ff(q))
        return self.out(q)[..., 0].float()                                                 # [B, 12]


def decode_v6(tau_base: torch.Tensor, ctrl: torch.Tensor, v0: torch.Tensor, a_max: float = 1.0, d_max: float = 0.5,
              mode: str = "A", lon_st_slope: float = 0.0, anchor_h: Optional[torch.Tensor] = None
              ) -> Dict[str, torch.Tensor]:
    """raw controls [B, 12] -> tau_v6 through refiner/decoder.decode (mode A: deceleration only).  The decoder's own
    amplitudes (A_DEC 4 m/s^2 on the decel side, D_MAX 2 m) are mapped down in tanh space:
    A_DEC tanh(z') = a_max tanh(z)  <=>  z' = atanh((a_max / A_DEC) tanh(z)); same for the lateral side.
    z = w = 0 -> z' = w' = 0 -> tau_base bitwise (decoder anchoring).
    Heading (anchor_h [B, 8] given, report 124 D6): cf's candidates take the heading from the Bezier tangent of their
    xy (anchor_planner heading_from_xy, train.sh plan_heading_from_xy=true, fallback = the anchor heading), and
    tau_base = anchors + (bezier - anchors) in float32 is not bitwise a re-run of bezier_xyyaw.  So the edit is added:
    h_v6 = h_base + wrap(bezier(xy_v6) - bezier(xy_base)), which is h_base bitwise at Delta = 0."""
    from ..refiner import decoder as DEC
    from ..modules.kinematics import bezier_xyyaw
    z, w = ctrl[:, :6], ctrl[:, 6:]
    ra, rd = a_max / DEC.A_DEC, d_max / DEC.D_MAX
    if not (0 < ra < 1 and 0 < rd < 1):
        raise ValueError("a_max / d_max must be below the decoder bounds")
    zp = torch.atanh(ra * torch.tanh(z.to(tau_base.dtype)))
    wp = torch.atanh(rd * torch.tanh(w.to(tau_base.dtype)))
    out = DEC.decode(tau_base, zp, wp, v0=v0.to(tau_base.dtype), mode=mode, lon_st_slope=lon_st_slope)
    if anchor_h is not None:
        xy = out["traj"][..., :2]
        fb = anchor_h.to(tau_base.dtype)
        dh = bezier_xyyaw(xy, fb)[..., 2] - bezier_xyyaw(tau_base[..., :2], fb)[..., 2]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))
        out["traj"] = torch.cat([xy, (tau_base[..., 2] + dh)[..., None]], -1)
        out["heading_mode"] = "bezier_delta"
    out["lon_dead"] = None
    if mode == "A":                                                                        # mode-A dead zone (report 124 §4-4)
        with torch.no_grad():                                                              # (mode B: c > 0 is a live speed-up)
            c_pre = DEC.lon_c_from_q(DEC.lon_q_from_z(zp))                                 # c_0 = c_1 = 0
            out["lon_dead"] = ((c_pre > 0).any(-1) & ~(c_pre < 0).any(-1)).float()
    return out


class RowJudge(nn.Module):
    """R-J: one row = one path, read alone (set-independent): E features, M, geometry and a Delta summary at the 21
    times -> MLP over the flattened times -> NC 3-class logits, DAC failure logit, TTC failure logit, progress (m).
    eff (the PDM-Closed effective progress, EP's denominator) comes from a separate token head on the ego state /
    command and the tau_base geometry (V19 default; the scorer's cand_eff1 is the parity arm)."""

    def __init__(self, hid: int = 192, n_cmd: int = 4):
        super().__init__()
        nin = (12 + 5 + 7 + 2) * len(T21) + 4 + n_cmd
        self.mlp = nn.Sequential(nn.Linear(nin, 2 * hid), nn.ReLU(inplace=True), nn.Linear(2 * hid, hid),
                                 nn.ReLU(inplace=True), nn.Linear(hid, 6))
        self.eff = nn.Sequential(nn.Linear(7 * len(T21) + 4 + n_cmd, hid), nn.ReLU(inplace=True), nn.Linear(hid, 1))

    def forward(self, E: torch.Tensor, M: torch.Tensor, S: torch.Tensor, dsum: torch.Tensor, ego: torch.Tensor,
                cmd: torch.Tensor, S_base: torch.Tensor) -> Dict[str, torch.Tensor]:
        """E [B, R, T, 12], M [B, R, T, 5], S [B, R, 41, 7], dsum [B, R, T, 2] (lon / lat edit of the row vs tau_base),
        ego [B, 4], cmd [B, n_cmd], S_base [B, 41, 7]; all detached -> nc [B, R, 3], dac [B, R], ttc [B, R],
        prog [B, R] (m), eff [B] (m)"""
        b, r = E.shape[:2]
        g = geom_features(S)
        x = torch.cat([e_features(E), M, g, dsum], -1).flatten(2)
        ec = torch.cat([ego, cmd], -1)
        o = self.mlp(torch.cat([x, ec[:, None].expand(-1, r, -1)], -1)).float()
        eff = 10.0 * self.eff(torch.cat([geom_features(S_base[:, None])[:, 0].flatten(1), ec], -1)).float()[..., 0]
        return {"nc": o[..., :3], "dac": o[..., 3], "ttc": o[..., 4], "prog": 10.0 * o[..., 5], "eff": eff}


def expected_pdms(j: Dict[str, torch.Tensor], c: torch.Tensor) -> Dict[str, torch.Tensor]:
    """R-J terms -> expected PDMS per row, the formula of candidate_scorer.expected_official (report 124 §2-5a):
    m = E[NC] (1 - P(DAC fail)) * DDC (DDC := 1), masked = m max(prog, 0), ep = masked / max(masked, eff) if that
    max > 5 m else 1, PDMS = m (5 ep + 5 (1 - P(TTC fail)) + 2 C) / 12.  c [B, R] = comfort of each row's states."""
    p = torch.softmax(j["nc"].float(), -1)
    m = (p[..., 0] + 0.5 * p[..., 1]) * (1 - torch.sigmoid(j["dac"].float()))
    prog = j["prog"].float().clamp(min=0)
    masked = m * prog
    norm = torch.maximum(masked, j["eff"].float().clamp(min=0)[:, None])
    ep = torch.where(norm > 5.0, masked / norm.clamp(min=1e-6), torch.ones_like(prog))
    pdms = m * (5 * ep + 5 * (1 - torch.sigmoid(j["ttc"].float())) + 2 * c) / 12
    return {"pdms": pdms, "m": m, "ep": ep}


def n_params(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())
