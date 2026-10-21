"""Factorized candidate planner: path x speed vocabulary, coarse-to-fine scoring, PDM heads.

Stage 1 (paths)   every path anchor samples the BEV (+ predicted road SDF and
                  predicted future occupancy) at ``num_stations`` stations along
                  itself, pools them with the ego/scene token and gets a logit.
                  Top ``topk_path`` paths continue.
Stage 2 (speeds)  every (kept path, speed profile) pair is composed into poses;
                  each pose reads the predicted occupancy of ITS OWN time step
                  (the time-shifted state the v13 critic could not see), the
                  SDF under its centre, its speed, and gets a pair logit.  Speeds
                  are ranked by the log-sum-exp over the kept paths; top
                  ``topk_speed`` continue.
Stage 3 (fine)    the ``topk_path x topk_speed`` compositions (plus, in
                  training, the 256 WoTE anchors with exact PDM labels) are
                  encoded per step (BEV feature, occupancy now and 0.3/0.6/0.9 s
                  ahead, SDF at the four footprint corners, pose, speed), passed
                  through a per-candidate encoder and a set encoder over all
                  candidates, and scored by five PDM heads (NC, DAC, TTC, C
                  logits; EP in [0, 1]) plus an imitation logit.  The selection
                  score is log PDMS(predicted) + im_weight * log softmax(im).
                  A small offset head refines the top ``topk`` candidates;
                  the heading of the refined path is the Bezier tangent.

All trajectories are in the NAVSIM ego frame (x forward, y left, rear axle).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
from navsim.agents.para_ssr.plan_map import footprint_corners

from .proxy import TTC_HORIZONS, bev_grid, sample
from .vocab import compose, load_vocab, path_headings

SIM_KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort")
HEAD_KEYS = ("nc", "dac", "ttc", "comfort")      # order of the 4 BCE logits
DT = 0.5


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


class FactorizedPlanner(nn.Module):
    def __init__(
        self,
        path_file: str,
        speed_file: str,
        pc_range: Sequence[float],
        embed_dims: int = 256,
        fut_ts: int = 8,
        num_heads: int = 8,
        station_m: float = 1.0,
        num_stations: int = 16,
        topk_path: int = 20,
        topk_speed: int = 20,
        topk: int = 6,
        im_weight: float = 0.1,
        fine_layers: int = 2,
        dropout: float = 0.1,
        extra_anchor_file: Optional[str] = None,
    ):
        super().__init__()
        paths, speeds = load_vocab(path_file, speed_file)
        if extra_anchor_file:
            import numpy as np
            anchors = torch.tensor(np.load(extra_anchor_file), dtype=torch.float32)
            if anchors.ndim != 3 or anchors.shape[1:] != (fut_ts, 3):
                raise ValueError(f"extra anchors must be [M, {fut_ts}, 3], got {tuple(anchors.shape)}")
            self.register_buffer("extra_anchors", anchors)
        else:
            self.extra_anchors = None
        if speeds.shape[1] != fut_ts:
            raise ValueError(f"speed profiles have {speeds.shape[1]} steps, model has fut_ts={fut_ts}")
        self.register_buffer("paths", paths)                                  # [P, S+1, 2]
        self.register_buffer("path_heading", path_headings(paths))            # [P, S+1]
        self.register_buffer("speeds", speeds)                                # [V, T]
        self.num_paths, self.num_speeds = paths.shape[0], speeds.shape[0]
        self.num_station_all = paths.shape[1]
        self.station_m = float(station_m)
        self.fut_ts = int(fut_ts)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.topk_path = min(int(topk_path), self.num_paths)
        self.topk_speed = min(int(topk_speed), self.num_speeds)
        self.topk = int(topk)
        self.im_weight = float(im_weight)
        C = embed_dims
        self.embed_dims = C

        idx = torch.linspace(0, self.num_station_all - 1, int(num_stations)).round().long()
        self.register_buffer("station_idx", idx)
        self.num_stations = int(num_stations)

        # ---- stage 1: paths
        self.path_embed = _mlp(2 * self.num_station_all, 256, C)
        self.station_pos = nn.Embedding(self.num_stations, C)
        self.station_proj = nn.Linear(C + 3, C)          # bev | sdf | occ(any t) | inside
        self.path_norm_q = nn.LayerNorm(C)
        self.path_norm_kv = nn.LayerNorm(C)
        self.path_pool = nn.MultiheadAttention(C, num_heads, dropout=dropout, batch_first=True)
        self.path_norm_out = nn.LayerNorm(C)
        self.path_ffn = _mlp(C, 2 * C, C)
        self.path_score = _mlp(C, 128, 1)

        # ---- stage 2: speeds
        self.speed_embed = _mlp(self.fut_ts, 128, C)
        self.pair_step_proj = nn.Linear(4 * self.fut_ts, C)   # per step: occ(t) | sdf centre | speed | inside
        self.pair_mlp = _mlp(2 * C, C, 1)

        # ---- stage 3: fine
        n_scalar = 1 + len(TTC_HORIZONS) + 4 + 4 + 2 + 1      # occ | occ ahead | sdf corners | x y cos sin | speed acc | inside
        self.step_proj = nn.Linear(C + n_scalar, C)
        self.step_pos = nn.Embedding(self.fut_ts, C)
        self.cand_embed = _mlp(3 * self.fut_ts, 256, C)
        self.cand_encoder = nn.TransformerEncoderLayer(C, num_heads, dim_feedforward=2 * C, dropout=dropout,
                                                       batch_first=True, norm_first=True)
        set_layer = nn.TransformerEncoderLayer(C, num_heads, dim_feedforward=2 * C, dropout=dropout,
                                               batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(set_layer, num_layers=int(fine_layers))
        self.set_norm = nn.LayerNorm(C)
        self.score_head = _mlp(C, 128, len(HEAD_KEYS))
        self.ep_head = _mlp(C, 128, 1)
        self.im_head = _mlp(C, 128, 1)
        self.offset_head = nn.Sequential(nn.Linear(C, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 2 * self.fut_ts))
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    # ------------------------------------------------------------------ #
    def compose_pairs(self, path_idx: torch.Tensor, speed_idx: torch.Tensor) -> torch.Tensor:
        """``path_idx [B, Kp]``, ``speed_idx [B, Kv]`` -> poses ``[B, Kp, Kv, T, 3]``."""
        p = self.paths[path_idx].unsqueeze(2)                   # [B, Kp, 1, S+1, 2]
        h = self.path_heading[path_idx].unsqueeze(2)            # [B, Kp, 1, S+1]
        s = self.speeds[speed_idx].unsqueeze(1)                 # [B, 1, Kv, T]
        return compose(p, h, s, self.station_m)

    def _bev_sample(self, bev: torch.Tensor, pts: torch.Tensor):
        """``bev [B, C, H, W]``, nav pts ``[B, N, 2]`` -> features ``[B, N, C]``, inside ``[B, N]``."""
        grid, inside = bev_grid(pts, self.pc_range)
        feat = sample(bev, grid)
        return feat * inside.unsqueeze(-1).to(feat.dtype), inside

    def _occ_at_step(self, occ: torch.Tensor, pts: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """occ probabilities ``[B, T, H, W]``, pts ``[B, N, 2]``, step ``[N]`` -> ``[B, N]``."""
        grid, inside = bev_grid(pts, self.pc_range)
        vals = sample(occ, grid)                                   # [B, N, T]
        out = vals.gather(2, step.view(1, -1, 1).expand(vals.shape[0], -1, 1))[..., 0]
        return out * inside.to(out.dtype)

    # ------------------------------------------------------------------ #
    def stage_paths(self, bev, sdf, occ_any, ego) -> torch.Tensor:
        """-> path logits ``[B, P]``."""
        B, C = ego.shape
        P = self.num_paths
        stations = self.paths[:, self.station_idx]                                     # [P, Ns, 2]
        pts = stations.reshape(1, P * self.num_stations, 2).expand(B, -1, -1)
        feat, inside = self._bev_sample(bev, pts)                                      # [B, P*Ns, C]
        extra = torch.cat([sample(sdf, bev_grid(pts, self.pc_range)[0]),
                           sample(occ_any, bev_grid(pts, self.pc_range)[0]),
                           inside.unsqueeze(-1).to(feat.dtype)], dim=-1)                # [B, P*Ns, 3]
        tok = self.station_proj(torch.cat([feat, extra], dim=-1)).reshape(B, P, self.num_stations, C)
        tok = tok + self.station_pos.weight.view(1, 1, self.num_stations, C)
        q = self.path_embed(self.paths.reshape(1, P, -1).expand(B, -1, -1)) + ego.unsqueeze(1)   # [B, P, C]
        q2 = self.path_norm_q(q).reshape(B * P, 1, C)
        kv = self.path_norm_kv(tok).reshape(B * P, self.num_stations, C)
        pooled, _ = self.path_pool(q2, kv, kv, need_weights=False)
        h = q + pooled.reshape(B, P, C)
        h = h + self.path_ffn(self.path_norm_out(h))
        self._path_feat = h
        return self.path_score(h).squeeze(-1)

    def stage_speeds(self, path_idx, path_feat, sdf, occ, ego) -> torch.Tensor:
        """-> pair logits ``[B, Kp, V]``; the composed poses are kept on ``self._pair_poses``."""
        B, Kp = path_idx.shape
        V, T, C = self.num_speeds, self.fut_ts, self.embed_dims
        all_speed = torch.arange(V, device=path_idx.device).view(1, V).expand(B, -1)
        poses = self.compose_pairs(path_idx, all_speed)                                  # [B, Kp, V, T, 3]
        pts = poses[..., :2].reshape(B, Kp * V * T, 2)
        step = torch.arange(T, device=pts.device).repeat(Kp * V)
        occ_t = self._occ_at_step(occ, pts, step)                                       # [B, Kp*V*T]
        grid, inside = bev_grid(pts, self.pc_range)
        sdf_c = sample(sdf, grid)[..., 0]
        speed = self.speeds[all_speed]                                                   # [B, V, T] distances
        speed = torch.diff(torch.cat([speed.new_zeros(B, V, 1), speed], dim=-1), dim=-1) / DT
        speed = speed.view(B, 1, V, T).expand(B, Kp, V, T).reshape(B, -1)
        scal = torch.stack([occ_t, sdf_c / 10.0, speed / 20.0, inside.to(occ_t.dtype)], dim=-1)   # [B, Kp*V*T, 4]
        scal = scal.reshape(B, Kp, V, T * 4)
        q = (self.speed_embed(self.speeds).view(1, 1, V, C) + path_feat.unsqueeze(2) + ego.view(B, 1, 1, C))
        h = torch.cat([q.expand(B, Kp, V, C), self.pair_step_proj(scal)], dim=-1)
        self._pair_poses = poses
        return self.pair_mlp(h).squeeze(-1)

    def encode_candidates(self, cand, bev, sdf, occ, ego, v0) -> torch.Tensor:
        """``cand [B, N, T, 3]`` -> candidate features ``[B, N, C]`` (after the set encoder)."""
        B, N, T, _ = cand.shape
        C = self.embed_dims
        xy = cand[..., :2]
        pts = xy.reshape(B, N * T, 2)
        feat, inside = self._bev_sample(bev, pts)                                        # [B, N*T, C]
        step = torch.arange(T, device=cand.device).repeat(N)
        occ_now = self._occ_at_step(occ, pts, step)
        speed = torch.diff(torch.cat([xy.new_zeros(B, N, 1, 2), xy], dim=-2), dim=-2).norm(dim=-1) / DT   # [B, N, T]
        acc = torch.diff(torch.cat([v0.view(B, 1, 1).expand(B, N, 1).to(speed.dtype), speed], dim=-1), dim=-1) / DT
        heading = cand[..., 2]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        ahead = []
        for tau in TTC_HORIZONS:
            p_tau = (xy + direction * (speed * tau).unsqueeze(-1)).reshape(B, N * T, 2)
            s_tau = (torch.arange(T, device=cand.device).float() + tau / DT).round().long().clamp(max=T - 1).repeat(N)
            ahead.append(self._occ_at_step(occ, p_tau, s_tau))
        corners = footprint_corners(cand).reshape(B, N * T * 4, 2)
        sdf_corner = sample(sdf, bev_grid(corners, self.pc_range)[0])[..., 0].reshape(B, N * T, 4)
        scal = torch.cat([
            occ_now.unsqueeze(-1), torch.stack(ahead, dim=-1), sdf_corner / 10.0,
            (xy / 32.0).reshape(B, N * T, 2), direction.reshape(B, N * T, 2),
            (speed / 20.0).reshape(B, N * T, 1), (acc / 5.0).reshape(B, N * T, 1),
            inside.unsqueeze(-1).to(feat.dtype),
        ], dim=-1)
        tok = self.step_proj(torch.cat([feat, scal], dim=-1)).reshape(B, N, T, C)
        tok = tok + self.step_pos.weight.view(1, 1, T, C)
        ctok = (self.cand_embed(cand.reshape(B, N, -1) / 32.0) + ego.view(B, 1, C)).unsqueeze(2)      # [B, N, 1, C]
        seq = torch.cat([ctok, tok], dim=2).reshape(B * N, T + 1, C)
        seq = self.cand_encoder(seq)
        cfeat = seq[:, 0].reshape(B, N, C)
        set_in = torch.cat([ego.view(B, 1, C), cfeat], dim=1)
        set_out = self.set_norm(self.set_encoder(set_in))
        return set_out[:, 1:]

    def heads(self, feat: torch.Tensor):
        logits = self.score_head(feat)                     # [B, N, 4]
        ep_logit = self.ep_head(feat).squeeze(-1)
        im_logit = self.im_head(feat).squeeze(-1)
        return logits, ep_logit, im_logit

    def selection_score(self, logits, ep_logit, im_logit) -> torch.Tensor:
        p = torch.sigmoid(logits.float())
        nc, dac, ttc, cf = p.unbind(-1)
        ep = torch.sigmoid(ep_logit.float())
        pdms = nc * dac * (5 * ttc + 5 * ep + 2 * cf) / 12.0
        im = torch.log_softmax(im_logit.float(), dim=-1)
        return torch.log(pdms + 1e-6) + self.im_weight * im, pdms

    # ------------------------------------------------------------------ #
    def forward(
        self,
        bev: torch.Tensor,
        ego: torch.Tensor,
        ego_status: torch.Tensor,
        sdf: torch.Tensor,
        occ: torch.Tensor,
        use_extra: bool = False,
        gt_path: Optional[torch.Tensor] = None,
        gt_speed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """bev ``[B, C, H, W]``; ego ``[B, C]`` scene/ego token; ego_status ``[B, 4]`` (vx, vy, ax, ay);
        sdf ``[B, 1, H, W]`` predicted road SDF (metres); occ ``[B, T, H, W]`` predicted occupancy
        probabilities; use_extra scores the 256 exact-label anchors as well (training);
        gt_path/gt_speed ``[B]`` indices forced into the kept sets (training)."""
        B = bev.shape[0]
        T = self.fut_ts
        v0 = ego_status[:, 0]
        occ_any = occ.amax(dim=1, keepdim=True)
        extra = None
        if use_extra and self.extra_anchors is not None:
            extra = self.extra_anchors.unsqueeze(0).expand(B, -1, -1, -1)

        path_logit = self.stage_paths(bev, sdf, occ_any, ego)
        path_idx = path_logit.topk(self.topk_path, dim=-1).indices
        if gt_path is not None:
            path_idx = _force_include(path_idx, gt_path)
        path_feat = self._path_feat.gather(1, path_idx.unsqueeze(-1).expand(-1, -1, self.embed_dims))

        pair_logit = self.stage_speeds(path_idx, path_feat, sdf, occ, ego)               # [B, Kp, V]
        speed_logit = torch.logsumexp(pair_logit, dim=1)                                  # [B, V]
        speed_idx = speed_logit.topk(self.topk_speed, dim=-1).indices
        if gt_speed is not None:
            speed_idx = _force_include(speed_idx, gt_speed)
        poses = self._pair_poses.gather(2, speed_idx.view(B, 1, -1, 1, 1).expand(-1, self.topk_path, -1, T, 3))
        cand = poses.reshape(B, self.topk_path * self.topk_speed, T, 3)
        N = cand.shape[1]

        scored = cand if extra is None else torch.cat([cand, extra.to(cand.dtype)], dim=1)
        feat = self.encode_candidates(scored, bev, sdf, occ, ego, v0)
        logits, ep_logit, im_logit = self.heads(feat)
        final, pdms = self.selection_score(logits[:, :N], ep_logit[:, :N], im_logit[:, :N])

        k = min(self.topk, N)
        top = final.topk(k, dim=-1).indices                                               # [B, k]
        top_feat = feat[:, :N].gather(1, top.unsqueeze(-1).expand(-1, -1, self.embed_dims))
        top_cand = cand.gather(1, top[:, :, None, None].expand(-1, -1, T, 3))
        offset = self.offset_head(top_feat).view(B, k, T, 2)
        refined_xy = top_cand[..., :2] + offset
        refined = bezier_xyyaw(refined_xy, top_cand[..., 2])                              # [B, k, T, 3]

        p = torch.sigmoid(logits[:, :N].float())
        sim = torch.stack([p[..., 0], p[..., 1], torch.sigmoid(ep_logit[:, :N].float()), p[..., 2], p[..., 3]], dim=1)
        out = {
            "trajectory": refined[:, 0],
            "fact_cand": cand,
            "fact_path_logit": path_logit,
            "fact_path_idx": path_idx,
            "fact_pair_logit": pair_logit,
            "fact_speed_logit": speed_logit,
            "fact_speed_idx": speed_idx,
            "fact_logits": logits[:, :N],
            "fact_ep_logit": ep_logit[:, :N],
            "fact_im_logit": im_logit[:, :N],
            "fact_pdms": pdms,
            "fact_final": final,
            "fact_topk_index": top,
            "fact_topk_traj": refined,
            "fact_offset": offset,
            # v2-compatible names for the evaluation / dump tools
            "plan_final_rewards": final,
            "plan_topk_trajectory": refined,
            "plan_topk_reward": final.gather(1, top),
            "plan_topk_index": top,
            "sim_rewards": sim,
        }
        if extra is not None:
            out.update({
                "fact_extra_logits": logits[:, N:],
                "fact_extra_ep_logit": ep_logit[:, N:],
                "fact_extra_im_logit": im_logit[:, N:],
            })
        return out


def _force_include(idx: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Replace the last slot of ``idx [B, K]`` by ``gt [B]`` where it is absent."""
    present = (idx == gt.view(-1, 1)).any(dim=1)
    out = idx.clone()
    out[~present, -1] = gt[~present]
    return out
