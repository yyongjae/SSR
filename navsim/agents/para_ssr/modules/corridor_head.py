"""km 10/15: corridor head + GT corridor targets.

Head: bev_embed [B, H*W, C] (row = forward, col = SSR x right, 0.64 m) -> 2x upsample -> 100 x 200 at 0.32 m:
  corridor_sdf [B, 100, 200]     static drivable signed distance (m, + inside), CLIP * tanh(y / CLIP)
  corridor_occ [B, 8, 100, 200]  agent-occupancy logits, plane k = union over t in ((k)*0.5, (k+1)*0.5] s
Targets (tools/corridor/build_corridor_dt01.py, metric cache = what the official scorer reads), looked up by the
current-frame token in the packed navtrain_train / navtrain_val token tables.
Loss: boundary-weighted smooth-L1 on the SDF + BCE (pos_weight 4) on occupancy (tools/corridor/train_corridor.py).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

CLIP = 8.0
T_SWEPT = 8
SPLITS = ("navtrain_train", "navtrain_val")


def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


def _fourier(xy: torch.Tensor, nf: int = 16) -> torch.Tensor:
    """xy [..., 2] in units of 32 m -> [..., 4 nf] sin / cos features (periods 64 m .. ~1 m)"""
    f = (2.0 ** torch.linspace(0, 6, nf, device=xy.device, dtype=xy.dtype)) * np.pi
    a = xy[..., None] * f                                                           # [..., 2, nf]
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class AgentCrossAttn(nn.Module):
    """km 10/16 (v3.1, test B): the corridor's BEV-pixel features attend to the det/motion head's agent-mode tokens
    (top-k queries by detection score x 6 modes: motion latent + box + class probability + Fourier-encoded current
    and 8 future centres + mode logit), so the occupancy planes can use the specialist's object-level motion.  The
    output enters through a zero-initialised 1x1 conv: at step 0 the head equals the v3 corridor head.
    Pixel and agent positions share one frame (SSR x right, y forward, / 32 m)."""

    def __init__(self, hid: int, cin: int, bev_h: int, bev_w: int, k: int = 32, layers: int = 2,
                 detach: bool = True, nf: int = 16):
        super().__init__()
        self.k, self.detach, self.nf = k, detach, nf
        self.tok = nn.Sequential(nn.Linear(cin + 10 + 7 + 9 * 4 * nf + 1, hid), nn.ReLU(inplace=True),
                                 nn.Linear(hid, hid))
        self.pix = nn.Linear(4 * nf, hid)
        cell = 32.0 / bev_h                                                             # 0.64 m
        fwd = (torch.arange(bev_h) + 0.5) * cell
        ssr_x = -32.0 + (torch.arange(bev_w) + 0.5) * (64.0 / bev_w)
        gy, gx = torch.meshgrid(fwd, ssr_x, indexing="ij")
        self.register_buffer("grid", torch.stack([gx, gy], -1).reshape(-1, 2) / 32.0, persistent=False)
        self.attn = nn.ModuleList([nn.MultiheadAttention(hid, 8, batch_first=True) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hid, 2 * hid), nn.ReLU(inplace=True), nn.Linear(2 * hid, hid))
                                  for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.proj = nn.Conv2d(hid, hid, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, det: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, c, h, w = z.shape
        cls = det["all_cls_scores"][-1].float().sigmoid()                                  # [B, Q, 7]
        box = det["all_bbox_preds"][-1].float()                                            # [B, Q, 10]
        traj, tcls, mh = det["traj_preds"].float(), det["traj_cls_preds"].float(), det["motion_hidden"].float()
        if self.detach:
            cls, box, traj, tcls, mh = (x.detach() for x in (cls, box, traj, tcls, mh))
        top = cls.amax(-1).topk(min(self.k, cls.shape[1]), dim=1).indices                 # [B, k]

        def g(x):
            return torch.gather(x, 1, top.view(*top.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))

        cls, box, traj, tcls, mh = g(cls), g(box), g(traj), g(tcls), g(mh)
        m = traj.shape[2]
        c0 = box[..., :2] / 32.0                                                           # [B, k, 2]
        pts = torch.cat([c0[:, :, None, None].expand(-1, -1, m, 1, 2),
                         c0[:, :, None, None] + traj.cumsum(3) / 32.0], 3)                 # [B, k, M, 9, 2]
        bcode = torch.cat([c0, box[..., 2:8], box[..., 8:10] / 10.0], -1)                  # [B, k, 10]
        feat = torch.cat([mh, bcode[:, :, None].expand(-1, -1, m, -1), cls[:, :, None].expand(-1, -1, m, -1),
                          _fourier(pts, self.nf).flatten(-2), tcls[..., None]], -1)
        kv = self.tok(feat.flatten(1, 2))                                                  # [B, k*M, hid]
        q0 = z.flatten(2).transpose(1, 2)                                                  # [B, h*w, hid]
        pe = self.pix(_fourier(self.grid.to(z.dtype), self.nf))[None]
        x = q0
        for at, ff, n1, n2 in zip(self.attn, self.ffn, self.n1, self.n2):
            x = n1(x + at(x + pe, kv, kv, need_weights=False)[0])
            x = n2(x + ff(x))
        return z + self.proj((x - q0).transpose(1, 2).reshape(b, c, h, w))


class CorridorHead(nn.Module):
    def __init__(self, cin: int, bev_h: int, bev_w: int, hid: int = 192, t: int = T_SWEPT, route: bool = False,
                 agent_attn: bool = False, agent_k: int = 32, agent_detach: bool = True):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        # km 10/16 (v3.1): cross-attention to the det/motion head's agent tokens (None = the v3 head)
        self.agent = AgentCrossAttn(hid, cin, bev_h, bev_w, agent_k, detach=agent_detach) if agent_attn else None
        # km 10/16 (v3-64): route progress / distance channels at the BEV resolution (0.64 m)
        self.route = nn.Conv2d(hid, 2, 1) if route else None
        self.inp = nn.Sequential(nn.Conv2d(cin, hid, 1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True))
        self.low = nn.Sequential(_blk(hid, hid), _blk(hid, hid), _blk(hid, hid))
        self.up = nn.Sequential(nn.ConvTranspose2d(hid, hid // 2, 2, stride=2), nn.BatchNorm2d(hid // 2),
                                nn.ReLU(inplace=True), _blk(hid // 2, hid // 2))
        self.out = nn.Conv2d(hid // 2, 1 + t, 1)

    def forward(self, bev_embed: torch.Tensor, det: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, n, c = bev_embed.shape
        x = bev_embed.transpose(1, 2).reshape(b, c, self.bev_h, self.bev_w)
        z = self.low(self.inp(x))
        if self.agent is not None:
            assert det is not None and "motion_hidden" in det, "agent_attn needs det_out with return_hidden=True"
            z = self.agent(z, det)
        y = self.out(self.up(z))
        out = {"corridor_sdf": CLIP * torch.tanh(y[:, 0] / CLIP), "corridor_occ": y[:, 1:]}
        if self.route is not None:
            r = self.route(z)
            out["corridor_route_s"], out["corridor_route_d"] = 10.0 * r[:, 0], F.softplus(r[:, 1])
        return out


def corridor_loss(pred: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], occ_weight: float = 1.0,
                  route_weight: float = 0.0):
    """-> (loss, logs).  Samples without a target (corridor_valid = 0) contribute 0 but keep the graph."""
    sp, op = pred["corridor_sdf"].float(), pred["corridor_occ"].float()
    s, o = targets["corridor_sdf"].float(), targets["corridor_occ"].float()
    v = targets["corridor_valid"].float()
    nv = v.sum().clamp(min=1.0)
    w = 1.0 + 4.0 * torch.exp(-s.abs())
    ls = ((w * F.smooth_l1_loss(sp, s, beta=0.5, reduction="none")).mean((1, 2)) * v).sum() / nv
    lo = (F.binary_cross_entropy_with_logits(op, o, pos_weight=torch.tensor(4.0, device=op.device),
                                             reduction="none").mean((1, 2, 3)) * v).sum() / nv
    loss = ls + occ_weight * lo + 0.0 * (sp.sum() + op.sum())
    with torch.no_grad():
        band = (s.abs() < 1) & (v[:, None, None] > 0)
        sign = ((sp > 0) == (s > 0))[band].float().mean() if band.any() else sp.new_tensor(0.0)
    logs = {"corr/sdf": ls.detach(), "corr/occ": lo.detach(), "corr/sign_band1m": sign}
    if route_weight > 0 and "corridor_route_s" in pred and "corridor_route_s" in targets:
        # progress along the route (what EP measures), supervised near the route (distance < 6 m), in 10 m units
        rs, rd = pred["corridor_route_s"].float(), pred["corridor_route_d"].float()
        ts, td = targets["corridor_route_s"].float(), targets["corridor_route_d"].float()
        near = (td < 6.0).float() * v[:, None, None]
        lrs = (F.smooth_l1_loss(rs / 10, ts / 10, beta=0.5, reduction="none") * near).sum() / near.sum().clamp(min=1.0)
        lrd = ((F.smooth_l1_loss(rd, td, beta=0.5, reduction="none")).mean((1, 2)) * v).sum() / nv
        loss = loss + route_weight * (lrs + 0.2 * lrd) + 0.0 * (rs.sum() + rd.sum())
        logs.update({"corr/route_s": lrs.detach(), "corr/route_d": lrd.detach()})
    return loss, logs


class CorridorTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self._dir = str(config.corridor_dir)
        self._troot = str(config.corridor_tokens_root)
        self._planes = int(getattr(config, "corridor_occ_planes", T_SWEPT))
        self._route = str(config.corridor_route_dir) if float(getattr(config, "corridor_route_weight", 0.0)) > 0 else ""
        self._arr = None

    def get_unique_name(self) -> str:
        name = "para_ssr_corridor_dt01_swept8" if self._planes == T_SWEPT else f"para_ssr_corridor_dt01_p{self._planes}"
        return name + ("_route" if self._route else "")

    def _open(self):
        if self._arr is None:  # lazily: the builder is pickled into workers
            import pandas as pd
            self._index, self._arr = {}, {}
            for sp in SPLITS:
                toks = pd.read_parquet(f"{self._troot}/{sp}/tokens.parquet").token.astype(str).to_numpy()
                ok = np.load(f"{self._dir}/{sp}/ok.npy")
                self._index.update({t: (sp, i) for i, t in enumerate(toks) if ok[i]})
                self._arr[sp] = (np.load(f"{self._dir}/{sp}/static_sdf.npy", mmap_mode="r"),
                                 np.load(f"{self._dir}/{sp}/dyn_occ.npy", mmap_mode="r"))
                if self._route:
                    self._arr[sp] += (np.load(f"{self._route}/{sp}/route_s.npy", mmap_mode="r"),
                                      np.load(f"{self._route}/{sp}/route_d.npy", mmap_mode="r"))

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        self._open()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        hit = self._index.get(token)
        if hit is None:
            out = {"corridor_sdf": torch.zeros(100, 200), "corridor_occ": torch.zeros(self._planes, 100, 200),
                   "corridor_valid": torch.tensor(0.0)}
            if self._route:
                out.update({"corridor_route_s": torch.zeros(50, 100), "corridor_route_d": torch.full((50, 100), 20.0)})
            return out
        S, D = self._arr[hit[0]][:2]
        r = hit[1]
        route = {}
        if self._route:
            RS, RD = self._arr[hit[0]][2:]
            route = {"corridor_route_s": torch.from_numpy(np.asarray(RS[r], np.float32)),
                     "corridor_route_d": torch.from_numpy(np.asarray(RD[r], np.float32))}
        d = np.asarray(D[r])                                                   # [40, 100, 25] packed bits, 0.1 s
        if self._planes != d.shape[0]:                                         # swept over 40 / planes steps
            d = np.bitwise_or.reduce(d.reshape(self._planes, -1, *d.shape[1:]), axis=1)
        return {"corridor_sdf": torch.from_numpy(np.asarray(S[r], np.float32)),
                "corridor_occ": torch.from_numpy(np.unpackbits(d, axis=-1).astype(np.float32)),
                "corridor_valid": torch.tensor(1.0), **route}


# ---------------------------------------------------------------------------------------------------- constraint
FRONT, REAR, HALF_W = 4.049, 1.127, 2.297 / 2          # Pacifica footprint about the rear axle


def _occ_dist(planes: np.ndarray, res: float = 0.32) -> np.ndarray:
    """bool [T, H, W] occupancy -> f16 signed distance (m, + free, - inside an agent), clipped to CLIP."""
    import cv2
    out = np.empty(planes.shape, np.float16)
    for k, occ in enumerate(planes):
        free = (~occ).astype(np.uint8)
        if free.all():
            out[k] = CLIP
            continue
        d_in = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_out = cv2.distanceTransform(1 - free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[k] = np.clip(np.where(free > 0, d_in - 0.5, -(d_out - 0.5)) * res, -CLIP, CLIP)
    return out


class CorridorPlanTargetBuilder(CorridorTargetBuilder):
    """Adds corridor_occ_dist [40, 100, 200] (0.1 s planes, signed distance to agents) for the plan constraint."""

    def get_unique_name(self) -> str:
        return "para_ssr_corridor_dt01_plan"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        out = super().compute_targets(scene)
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        hit = self._index.get(token)
        if hit is None:
            out["corridor_occ_dist"] = torch.full((40, 100, 200), CLIP)
            return out
        d = np.unpackbits(np.asarray(self._arr[hit[0]][1][hit[1]]), axis=-1).astype(bool)   # [40, 100, 200]
        out["corridor_occ_dist"] = torch.from_numpy(_occ_dist(d).astype(np.float32))
        return out


def _sample(field: torch.Tensor, pts: torch.Tensor) -> tuple:
    """field [B, P, H, 200] (row fwd 0..H*0.32 m, col SSR x right -32..32 m); pts [B, P, N, 2] NAVSIM ego (x fwd,
    y left) -> (bilinear values [B, P, N], in-range mask (x <= H*0.32 m, |y| <= 32 m))."""
    fwd = field.shape[-2] * 0.32                                   # 32 m (100 rows) or 64 m (200 rows)
    u = (-pts[..., 1] + 32.0) / 64.0 * 2 - 1
    v = pts[..., 0] / fwd * 2 - 1
    b, p, n = pts.shape[:3]
    g = torch.stack([u, v], -1).reshape(b * p, n, 1, 2)
    val = F.grid_sample(field.reshape(b * p, 1, *field.shape[2:]).float(), g.float(), mode="bilinear",
                        padding_mode="border", align_corners=False).reshape(b, p, n)
    return val, (v <= 1) & (u.abs() <= 1)


def corridor_plan_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                       status: torch.Tensor, static_margin: float, occ_margin: float, cap: float,
                       ttc_weight: float = 0.0, per_candidate: bool = False, prog_weight: float = 0.0,
                       prog_cap: float = 1.2, comfort_weight: float = 0.0):
    """Every candidate's tracked footprint vs the GT corridor -> (loss, logs).
    status [B, 4] = (vx, vy, ax, ay) of the rear axle, NAVSIM ego frame (the model's status_feature).
    ttc_weight > 0 adds the TTC term: official PDMScorer._calculate_ttc projects the footprint at step t along the
    heading by speed * dt, dt in {0.3, 0.6, 0.9} s, against the agents at t + dt (planes past 4.0 s: the 4.0 s plane;
    a stopped ego (< 5e-3 m/s) is exempt).
    km 10/16 (v3.1):
      prog_weight > 0 : progress reward inside the safe field — every candidate except the imitation winner (the
        anchor nearest the GT trajectory, the only offset the imitation loss binds) that passes the GT corridor
        (static, occupancy, TTC hinges all 0, comfortable, end within 4 m of the route) is rewarded by its route
        progress at the tracked end pose (GT route_s field = centerline.project(end) - project(ego), what EP
        measures), saturating at prog_cap x the GT trajectory's progress (EP is normalised by the best progress, so
        going further than that earns nothing).
      comfort_weight > 0 : hinge on the official comfort thresholds of the tracked states (lon accel [-4.05, 2.40],
        lat accel |vx * yaw rate| <= 4.89), every candidate, so the reward cannot buy progress with discomfort."""
    from .torch_tracker import track

    traj = predictions["trajectory_anchors"][None].to(predictions["trajectory_offset"].dtype) \
        + predictions["trajectory_offset"]                                                 # [B, K, 8, 3]
    b, k = traj.shape[:2]
    init = torch.cat([status.float(), status.new_zeros(b, 3).float()], -1)
    full = track(traj.reshape(b * k, 8, 3).float(), init.repeat_interleave(k, 0), full=True).reshape(b, k, 41, 7)
    st = full[:, :, 1:]
    x, y, h = st[..., 0], st[..., 1], st[..., 2]                                          # [B, K, 40]
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    # static: the 4 corners at every 0.1 s step
    cx = torch.tensor([FRONT, FRONT, -REAR, -REAR], device=x.device)
    cy = torch.tensor([HALF_W, -HALF_W, -HALF_W, HALF_W], device=x.device)
    px, py = x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy               # [B, K, 40, 4]
    sdf = targets["corridor_sdf"][:, None]                                                 # [B, 1, 100, 200]
    val, inr = _sample(sdf, torch.stack([px, py], -1).reshape(b, 1, k * 40 * 4, 2))
    val, inr = val.reshape(b, k, 40, 4), inr.reshape(b, k, 40, 4)
    v_static = ((static_margin - val).clamp(0, cap) * inr).amax(-1).mean(-1)               # [B, K]
    # dynamic: front half of the footprint (3 x 4 points) vs the agents at the same 0.1 s step
    fx = torch.linspace(0.0, FRONT, 4, device=x.device)
    fy = torch.linspace(-HALF_W, HALF_W, 3, device=x.device)
    fx, fy = torch.meshgrid(fx, fy, indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    qx, qy = x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy               # [B, K, 40, 12]
    q = torch.stack([qx, qy], -1).permute(0, 2, 1, 3, 4).reshape(b, 40, k * 12, 2)         # plane = time step
    odist = targets["corridor_occ_dist"]
    npl = odist.shape[1]
    dval, dinr = _sample(odist[:, :40], q)
    dval, dinr = dval.reshape(b, 40, k, 12), dinr.reshape(b, 40, k, 12)
    v_occ = ((occ_margin - dval).clamp(0, cap) * dinr).amax(-1).mean(1)                    # [B, K]
    v_ttc = torch.zeros_like(v_occ)
    if ttc_weight > 0:
        p0 = torch.cat([torch.zeros_like(x[..., :1]), x], -1), torch.cat([torch.zeros_like(y[..., :1]), y], -1)
        spd = torch.hypot(p0[0].diff(dim=-1), p0[1].diff(dim=-1)).detach() / 0.1          # [B, K, 40]
        moving = (spd >= 5e-3).float()
        steps = torch.arange(40, device=x.device)
        for d in (3, 6, 9):
            sh = (spd * d * 0.1)[..., None]                                                # [B, K, 40, 1]
            tx, ty = qx + c * sh, qy + s * sh
            tq = torch.stack([tx, ty], -1).permute(0, 2, 1, 3, 4).reshape(b, 40, k * 12, 2)
            fld = odist[:, (steps + d).clamp(max=npl - 1)]                                 # agents at t + dt
            tv, ti = _sample(fld, tq)
            tv, ti = tv.reshape(b, 40, k, 12), ti.reshape(b, 40, k, 12)
            hv = ((occ_margin - tv).clamp(0, cap) * ti).amax(-1).permute(0, 2, 1) * moving  # [B, K, 40]
            v_ttc = torch.maximum(v_ttc, hv.mean(-1))
    if per_candidate:                                                                      # checks only
        return v_static, v_occ, v_ttc
    valid = targets["corridor_valid"].float()[:, None]
    nv = (valid.sum() * k).clamp(min=1.0)
    l_s, l_o, l_t = (v_static * valid).sum() / nv, (v_occ * valid).sum() / nv, (v_ttc * valid).sum() / nv
    with torch.no_grad():
        viol_s = ((v_static > 0).float() * valid).sum() / nv
        viol_o = ((v_occ > 0).float() * valid).sum() / nv
        viol_t = ((v_ttc > 0).float() * valid).sum() / nv
    loss = l_s + l_o + ttc_weight * l_t
    logs = {"corr_plan/static": l_s.detach(), "corr_plan/occ": l_o.detach(),
            "corr_plan/ttc": l_t.detach(), "corr_plan/frac_static_viol": viol_s,
            "corr_plan/frac_occ_viol": viol_o, "corr_plan/frac_ttc_viol": viol_t}
    if comfort_weight > 0 or prog_weight > 0:
        vx, ax = st[..., 3], st[..., 5]
        yaw_rate = _wrap_t(torch.cat([h[..., :1], h], -1).diff(dim=-1)) / 0.1             # [B, K, 40]
        lat = vx * yaw_rate
        v_comf = (F.relu(ax - 2.40) + F.relu(-4.05 - ax) + F.relu(lat.abs() - 4.89)).amax(-1)   # [B, K]
        if comfort_weight > 0:
            l_c = (v_comf * valid).sum() / nv
            loss = loss + comfort_weight * l_c
            logs["corr_plan/comfort"] = l_c.detach()
            logs["corr_plan/frac_comfort_viol"] = (((v_comf > 0).float() * valid).sum() / nv).detach()
    if prog_weight > 0 and "corridor_route_s" in targets and "trajectory" in targets:
        rs = targets["corridor_route_s"].float()[:, None]
        rd = targets["corridor_route_d"].float()[:, None]
        if rs.shape[-1] != 200:                                                            # route at 0.64 m
            rs = F.interpolate(rs, scale_factor=2, mode="bilinear", align_corners=False)
            rd = F.interpolate(rd, scale_factor=2, mode="bilinear", align_corners=False)
        s_end, in_end = _sample(rs, torch.stack([x[..., -1], y[..., -1]], -1)[:, None])     # [B, 1, K]
        d_end, _ = _sample(rd, torch.stack([x[..., -1], y[..., -1]], -1)[:, None])
        s_end, in_end, d_end = s_end[:, 0], in_end[:, 0], d_end[:, 0]
        gt = targets["trajectory"].float()                                                  # [B, 8, 3]
        s_gt, _ = _sample(rs, gt[:, None, -1:, :2])                                          # [B, 1, 1]
        s_cap = prog_cap * s_gt[:, 0].clamp(min=2.0)                                         # [B, 1]
        anc = predictions["trajectory_anchors"].float().reshape(1, k, -1)
        winner = torch.cdist(gt.reshape(b, 1, -1), anc.expand(b, -1, -1))[:, 0].argmin(-1)   # [B]
        with torch.no_grad():
            ok = (v_static == 0) & (v_occ == 0) & (d_end < 4.0) & in_end
            if ttc_weight > 0:
                ok &= v_ttc == 0
            if comfort_weight > 0:
                ok &= v_comf == 0
            ok[torch.arange(b, device=x.device), winner] = False
            w = ok.float() * valid
        reward = torch.minimum(s_end, s_cap) / 10.0                                        # 10 m units
        l_p = -(reward * w).sum() / w.sum().clamp(min=1.0)
        loss = loss + prog_weight * l_p
        logs["corr_plan/prog_reward"] = (-l_p).detach()
        logs["corr_plan/frac_prog_eligible"] = (w.sum() / nv).detach()
        logs["corr_plan/prog_vs_gt"] = (((s_end / s_gt[:, 0].clamp(min=2.0)) * w).sum() / w.sum().clamp(min=1.0)).detach()
    return loss, logs


def _wrap_t(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def corridor_verdict_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                          status: torch.Tensor, n_anchor: int = 64, tau: float = 0.25, ttc: bool = True):
    """km 10/16 (v3.1): supervise the PREDICTED corridor at exactly what the selector reads — the verdict of tracked
    footprints — with the GT corridor's verdict of the same footprints.  Query trajectories are model-free: n_anchor
    random raw anchors (fixed vocabulary, no offsets) + the GT trajectory, tracked from the sample's status.
      DAC: BCE(sigmoid(min corner pred SDF / tau), [min corner GT SDF > 0]), in-range points only
      NC : BCE(max footprint pred occupancy, [GT footprint hits an agent at the same 0.1 s step])
      TTC: the same with the front half shifted by speed * {0.3, 0.6, 0.9} s against t + dt (moving ego)
    positives / negatives balanced per term.  -> (loss, logs)"""
    from .torch_tracker import track
    anc = predictions["trajectory_anchors"].float()                                          # [K, 8, 3]
    b = status.shape[0]
    idx = torch.randperm(anc.shape[0], device=anc.device)[:n_anchor]
    q = torch.cat([anc[idx][None].expand(b, -1, -1, -1), targets["trajectory"].float()[:, None]], 1)   # [B, Q, 8, 3]
    nq = q.shape[1]
    init = torch.cat([status.float(), status.new_zeros(b, 3).float()], -1)
    with torch.no_grad():
        st = track(q.reshape(b * nq, 8, 3), init.repeat_interleave(nq, 0), full=True).reshape(b, nq, 41, 7)[:, :, 1:]
    x, y, h, spd = st[..., 0], st[..., 1], st[..., 2], st[..., 3]
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    dev = x.device
    cx = torch.tensor([FRONT, FRONT, -REAR, -REAR], device=dev)
    cy = torch.tensor([HALF_W, -HALF_W, -HALF_W, HALF_W], device=dev)
    pc = torch.stack([x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy], -1).reshape(b, 1, -1, 2)
    ps, inr = _sample(predictions["corridor_sdf"].float()[:, None], pc)
    gs, _ = _sample(targets["corridor_sdf"].float()[:, None], pc)
    inr = inr.reshape(b, nq, -1)
    big = torch.full_like(ps.reshape(b, nq, -1), CLIP)
    p_min = torch.where(inr, ps.reshape(b, nq, -1), big).amin(-1)
    g_min = torch.where(inr, gs.reshape(b, nq, -1), big).amin(-1)
    m_dac = inr.any(-1).float() * targets["corridor_valid"].float()[:, None]
    y_dac = (g_min > 0).float()

    def bal_bce(logit_or_p, yv, m, is_prob=False):
        if is_prob:
            l = F.binary_cross_entropy(logit_or_p.clamp(1e-4, 1 - 1e-4), yv, reduction="none")
        else:
            l = F.binary_cross_entropy_with_logits(logit_or_p, yv, reduction="none")
        pos, neg = m * yv, m * (1 - yv)
        return 0.5 * ((l * pos).sum() / pos.sum().clamp(min=1) + (l * neg).sum() / neg.sum().clamp(min=1))

    l_dac = bal_bce(p_min / tau, y_dac, m_dac)
    fx, fy = torch.meshgrid(torch.linspace(-REAR, FRONT, 5, device=dev), torch.linspace(-HALF_W, HALF_W, 3, device=dev),
                            indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    pf = torch.stack([x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy], -1)  # [B, Q, 40, 15, 2]
    po = torch.sigmoid(predictions["corridor_occ"].float())                                  # [B, P, H, W]
    npl = po.shape[1]
    plane = torch.arange(40, device=dev) // (40 // npl)
    od = targets["corridor_occ_dist"].float()
    qf = pf.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2)
    pv, pin = _sample(po[:, plane], qf)
    gv, _ = _sample(od[:, :40], qf)
    pv = torch.where(pin, pv, torch.zeros_like(pv)).reshape(b, 40, nq, 15).amax(-1).amax(1)  # [B, Q]
    gh = (torch.where(pin, gv, torch.full_like(gv, CLIP)).reshape(b, 40, nq, 15).amin(-1).amin(1) < 0).float()
    m = targets["corridor_valid"].float()[:, None].expand(-1, nq)
    l_nc = bal_bce(pv, gh, m, is_prob=True)
    loss, logs = l_dac + l_nc, {}
    if ttc:
        gx, gy = torch.meshgrid(torch.linspace(0.0, FRONT, 4, device=dev), torch.linspace(-HALF_W, HALF_W, 3, device=dev),
                                indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        mov = (spd >= 5e-3).float()                                                          # [B, Q, 40]
        pt = torch.zeros(b, nq, device=dev)
        gtt = torch.full((b, nq), CLIP, device=dev)
        steps = torch.arange(40, device=dev)
        for dd in (3, 6, 9):
            sh = (spd * dd * 0.1)[..., None]
            tq = torch.stack([x[..., None] + c * (gx + sh) - s * gy, y[..., None] + s * (gx + sh) + c * gy], -1)
            tq = tq.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2)
            tv, ti = _sample(po[:, plane[(steps + dd).clamp(max=39)]], tq)
            tv = torch.where(ti, tv, torch.zeros_like(tv)).reshape(b, 40, nq, 12).amax(-1).permute(0, 2, 1)
            pt = torch.maximum(pt, (tv * mov).amax(-1))
            gd, _ = _sample(od[:, (steps + dd).clamp(max=od.shape[1] - 1)], tq)
            gd = torch.where(ti, gd, torch.full_like(gd, CLIP)).reshape(b, 40, nq, 12).amin(-1).permute(0, 2, 1)
            gd = torch.where(mov > 0, gd, torch.full_like(gd, CLIP))
            gtt = torch.minimum(gtt, gd.amin(-1))
        l_ttc = bal_bce(pt, (gtt < 0).float(), m, is_prob=True)
        loss = loss + l_ttc
        logs["corr_verd/ttc"] = l_ttc.detach()
        logs["corr_verd/ttc_pos"] = (gtt < 0).float().mean()
    with torch.no_grad():
        acc_dac = (((p_min > 0).float() == y_dac).float() * m_dac).sum() / m_dac.sum().clamp(min=1)
        acc_nc = ((pv >= 0.5).float() == gh).float().mean()
    logs.update({"corr_verd/dac": l_dac.detach(), "corr_verd/nc": l_nc.detach(), "corr_verd/acc_dac": acc_dac,
                 "corr_verd/acc_nc": acc_nc, "corr_verd/nc_pos": gh.mean()})
    return loss, logs


# ---------------------------------------------------------------------------------------------------- v3-64 GT
def raster_agents(rows: np.ndarray, planes: int = 49, h: int = 200, w: int = 200, res: float = 0.32) -> np.ndarray:
    """ragged agent rows [M, 9] (t index, 4 corners (fwd, left)) -> bool [planes, h, w], cell centre inside (the
    convention of build_corridor.py; IoU 0.996 vs its shapely raster)."""
    occ = np.zeros((planes, h, w), bool)
    for row in rows.astype(np.float32):
        t = int(row[0])
        if not 1 <= t <= planes:
            continue
        p = row[1:].reshape(4, 2)
        u, v = (-p[:, 1] + 32.0) / res - 0.5, p[:, 0] / res - 0.5                     # col, row in cell units
        j0, j1 = max(int(np.floor(u.min())), 0), min(int(np.ceil(u.max())), w - 1)
        i0, i1 = max(int(np.floor(v.min())), 0), min(int(np.ceil(v.max())), h - 1)
        if j0 > j1 or i0 > i1:
            continue
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        cr = np.stack([(u[(k + 1) % 4] - u[k]) * (ii - v[k]) - (v[(k + 1) % 4] - v[k]) * (jj - u[k]) for k in range(4)])
        occ[t - 1, i0:i1 + 1, j0:j1 + 1] |= (cr >= 0).all(0) | (cr <= 0).all(0)
    return occ


class Corridor64TargetBuilder(AbstractTargetBuilder):
    """tools/corridor/build_corridor64.py: forward 0..64 m.  corridor_sdf [200, 200], corridor_occ [P, 200, 200]
    (P = 40: 0.1 s, or 8 swept), corridor_route_s / _d [100, 100], corridor_valid; with plan=True also
    corridor_occ_dist [49, 200, 200] (0.1 .. 4.9 s) for the constraint loss (TTC projects past 4.0 s)."""

    def __init__(self, config, plan: bool = False):
        self._dir = str(config.corridor_dir)
        self._troot = str(config.corridor_tokens_root)
        self._planes = int(getattr(config, "corridor_occ_planes", 40))
        self._plan = plan
        self._arr = None

    def get_unique_name(self) -> str:
        return f"para_ssr_corridor64_p{self._planes}{'_plan' if self._plan else ''}"

    def _open(self):
        if self._arr is None:
            import pandas as pd
            self._index, self._arr = {}, {}
            for sp in SPLITS:
                toks = pd.read_parquet(f"{self._troot}/{sp}/tokens.parquet").token.astype(str).to_numpy()
                ok = np.load(f"{self._dir}/{sp}/ok.npy")
                self._index.update({t: (sp, i) for i, t in enumerate(toks) if ok[i]})
                self._arr[sp] = {k: np.load(f"{self._dir}/{sp}/{k}.npy", mmap_mode="r")
                                 for k in ("static_sdf", "route_s", "route_d", "agent_off", "agent_rows")}

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        self._open()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        hit = self._index.get(token)
        if hit is None:
            out = {"corridor_sdf": torch.zeros(200, 200), "corridor_occ": torch.zeros(self._planes, 200, 200),
                   "corridor_route_s": torch.zeros(100, 100), "corridor_route_d": torch.full((100, 100), 20.0),
                   "corridor_valid": torch.tensor(0.0)}
            if self._plan:
                out["corridor_occ_dist"] = torch.full((49, 200, 200), CLIP)
            return out
        A = self._arr[hit[0]]
        r = hit[1]
        occ = raster_agents(np.asarray(A["agent_rows"][A["agent_off"][r]:A["agent_off"][r + 1]]))   # [49, 200, 200]
        o40 = occ[:40]
        if self._planes != 40:
            o40 = o40.reshape(self._planes, -1, 200, 200).any(1)
        out = {"corridor_sdf": torch.from_numpy(np.asarray(A["static_sdf"][r], np.float32)),
               "corridor_occ": torch.from_numpy(o40.astype(np.float32)),
               "corridor_route_s": torch.from_numpy(np.asarray(A["route_s"][r], np.float32)),
               "corridor_route_d": torch.from_numpy(np.asarray(A["route_d"][r], np.float32)),
               "corridor_valid": torch.tensor(1.0)}
        if self._plan:
            out["corridor_occ_dist"] = torch.from_numpy(_occ_dist(occ).astype(np.float32))
        return out
