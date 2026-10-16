"""Anchor-based planning with PDM-score rewards (PARA-SSR v2), ported from WoTE.

Source: WoTE (github.com/liyingyanUCAS/WoTE), ``navsim/agents/WoTE`` in this
repo -- ``WoTE_model.py`` / ``WoTE_loss.py``.  Everything below follows that code
and its constants; the one part left out is WoTE's latent world model (future
BEV roll-out feeding the reward heads).  Here the reward heads read the anchor
queries after PARA-SSR's own planner layers (BEV + det/motion + map memories),
the role WoTE's ``offset_tf_decoder`` has.

    anchors   256 K-means trajectories [256, 8, 3] (x fwd, y left, heading), fixed
    query_k   = MLP([ego/command feature ; encoder(MLP(anchor_k))])       (WoTE: mlp_planning_vb,
                cluster_encoder, encode_ego_feat_mlp)
    -> PARA-SSR planner layers
    offset_k  = TrajectoryOffsetHead(query_k), heading = tanh * pi          (WoTE offset_head)
    im_k      = softmax_k(reward_head(query_k))                             (WoTE reward_head)
    sim_k,m   = sigmoid(sim_reward_heads[m](query_k)), m = NC, DAC, EP, TTC, C

Losses (``anchor_plan_losses``): WTA L1 on the offset of the anchor nearest the
human trajectory, soft cross-entropy of ``im`` against softmax(-L2 to the human
trajectory), and BCE of ``sim`` against the PDM scores of the 256 anchors
precomputed per token (WoTE's ``formatted_pdm_score_256.npy``), times 5.

Selection (``weighted_reward``): WoTE's
    w0 log im + w1 log NC + w2 log DAC + w3 log(5 TTC + 2 C + 5 EP),  w = [0.1, 0.5, 0.5, 1.0]
argmax over anchors; the output is anchor + offset.  ``plan_topk`` ranked
candidates are returned as well.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kinematics import bezier_xyyaw, bicycle_rollout, clamp_controls, poses_to_controls

SIM_KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort")

# WoTE's label file carries exactly SIM_KEYS, in that order (plan_score_targets.py).
LABEL_COLS = {k: i for i, k in enumerate(SIM_KEYS)}

# The corridor head set.  DAC and NC are gone as separate votes: both describe "may the ego BE
# here", the one question the corridor field answers, and both were measured to be the constraint
# rather than a preference (under a true DAC.NC filter, imitation alone scores 93.58 against
# WoTE's full rule 93.26 on the EPDMS-minus-EC scale).  What is left -- progress, time-to-collision,
# comfort -- is what to prefer AMONG feasible candidates, and those keep their WoTE labels.
# ``feasible`` has no anchor label: feasibility.py computes its target on the EMITTED trajectory.
CORRIDOR_KEYS = ("feasible", "ego_progress", "time_to_collision_within_bound", "comfort")
HEAD_MODES = {"wote": SIM_KEYS, "corridor": CORRIDOR_KEYS}


class AnchorPlanner(nn.Module):
    def __init__(
        self,
        anchor_file: str,
        embed_dims: int = 256,
        fut_ts: int = 8,
        traj_dims: int = 3,
        reward_weights: Sequence[float] = (0.1, 0.5, 0.5, 1.0),
        topk: int = 6,
        kinematic: bool = False,
        dt: float = 0.5,
        heading_from_xy: bool = False,
        select_mode: str = "weighted",
        select_dac_thr: float = 0.5,
        select_gate: float = 20.0,
        head_mode: str = "wote",
        score_sees_offset: bool = False,
    ):
        super().__init__()
        anchors = torch.tensor(np.load(anchor_file), dtype=torch.float32)
        if anchors.ndim != 3 or anchors.shape[1:] != (fut_ts, traj_dims):
            raise ValueError(f"anchors must be [K, {fut_ts}, {traj_dims}], got {tuple(anchors.shape)}")
        # A buffer, not WoTE's frozen Parameter: it must not reach the optimiser
        # and a checkpoint should carry the vocabulary it was trained with.
        self.register_buffer("trajectory_anchors", anchors)
        self.num_anchors = anchors.shape[0]
        self.fut_ts, self.traj_dims = fut_ts, traj_dims
        self.reward_weights = tuple(float(w) for w in reward_weights)
        if len(self.reward_weights) != 4:
            raise ValueError("reward_weights must be (w_im, w_nc, w_dac, w_rest)")
        self.topk = int(topk)
        # kinematic=True: the offset head predicts control corrections (da, domega) per
        # step on top of the anchor's own controls, and the poses are their bicycle
        # rollout (modules/kinematics.py, from TOAD) -- heading consistent with the path.
        self.kinematic = bool(kinematic)
        self.dt = float(dt)
        if self.kinematic and traj_dims != 3:
            raise ValueError("kinematic anchor planner needs (x, y, heading) anchors")
        # heading_from_xy=True (DiffusionDriveV2): the offset head predicts x, y only and
        # the heading is the tangent of the Bezier curve through the predicted points.
        self.heading_from_xy = bool(heading_from_xy)
        if self.heading_from_xy and (self.kinematic or traj_dims != 3):
            raise ValueError("heading_from_xy needs (x, y, heading) anchors and kinematic=False")

        hidden = 128                                          # WoTE SCORE_HEAD_HIDDEN_DIM
        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(fut_ts * traj_dims, hidden), nn.ReLU(), nn.Linear(hidden, embed_dims))
        layer = nn.TransformerEncoderLayer(d_model=embed_dims, nhead=8, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.cluster_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dims, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        self.offset_head = nn.Sequential(                      # WoTE TrajectoryOffsetHead, d_ffn 1024
            nn.Linear(embed_dims, 1024), nn.ReLU(),
            nn.Linear(1024, fut_ts * (2 if (self.kinematic or self.heading_from_xy) else traj_dims)))

        def score_head():
            return nn.Sequential(nn.Linear(embed_dims, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.reward_head = score_head()
        if str(head_mode) not in HEAD_MODES:
            raise ValueError(f"head_mode must be one of {sorted(HEAD_MODES)}, got {head_mode!r}")
        self.head_mode = str(head_mode)
        self.sim_keys = HEAD_MODES[self.head_mode]
        self.sim_reward_heads = nn.ModuleList(score_head() for _ in self.sim_keys)
        # The score heads judge anchor + offset but were only ever shown the query, never the offset:
        # the trajectory they are asked about was invisible to them.  Measured on B3 (epoch 23), if the
        # EP head merely predicted its OWN label exactly the selection would gain +1.3 PDMS (87.80 ->
        # 89.12), so the heads are failing to learn their target, not only learning the wrong one.
        # The offset is added to the query, detached (a score loss must not move the trajectory it
        # judges), through a layer initialised to zero so training starts exactly where v2 did.
        # The imitation head keeps the bare query: its target is the anchor's distance to the human.
        self.score_sees_offset = bool(score_sees_offset)
        if self.score_sees_offset:
            self.offset_embed = nn.Sequential(nn.Linear(fut_ts * traj_dims, hidden), nn.ReLU(),
                                              nn.Linear(hidden, embed_dims))
            nn.init.zeros_(self.offset_embed[-1].weight)
            nn.init.zeros_(self.offset_embed[-1].bias)
        self.select_mode = str(select_mode)
        if self.select_mode not in ("weighted", "corridor_first"):
            raise ValueError(f"select_mode must be 'weighted' or 'corridor_first', got {select_mode!r}")
        self.select_dac_thr = float(select_dac_thr)
        self.select_gate = float(select_gate)

    def queries(self, ego_feat: torch.Tensor, trajectories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ego/command feature [B, 1, C] -> one query per anchor [B, K, C].

        ``trajectories`` [B, K, T, 3] encodes those instead of the fixed anchors (WoTE's
        test-time re-encoding of the refined trajectories, ``WoTE_model.py`` is_eval branch).
        """
        bs = ego_feat.shape[0]
        if trajectories is None:
            flat = self.trajectory_anchors.reshape(1, self.num_anchors, -1).to(ego_feat.dtype).expand(bs, -1, -1)
        else:
            flat = trajectories.reshape(bs, self.num_anchors, -1).to(ego_feat.dtype)
        anchor_feat = self.cluster_encoder(self.mlp_planning_vb(flat))
        return self.encode_ego_feat_mlp(torch.cat([ego_feat.expand(-1, self.num_anchors, -1), anchor_feat], dim=-1))

    def weighted_reward(self, im: torch.Tensor, sim: torch.Tensor):
        """im [B, K] (softmax), sim [B, H, K] (sigmoid, ``self.sim_keys`` order) -> (final, ungated).

        ``head_mode``:

        ``wote``       five heads, WoTE's rule.  ``select_mode`` decides whether drivable-area
                       compliance votes (``weighted``) or gates (``corridor_first``).
        ``corridor``   four heads.  DAC and NC are replaced by ONE ``feasible`` head fed by the
                       merged corridor field, and it always gates -- "may the ego be here" is a
                       constraint, not a term to trade off.  What remains orders the admissible
                       set by progress, time-to-collision and comfort, which is what makes the
                       plan free to be aggressive: the constraint, not timidity, keeps it legal.

        ``ungated`` is the pre-gate score.  The feasibility term weights candidates by it, because
        weighting by the GATED score would starve exactly the candidates the constraint is meant to
        pull back in -- a self-fulfilling verdict.
        """
        eps = 1e-6
        w = self.reward_weights
        if self.head_mode == "corridor":
            feas, ep, ttc, comfort = sim.unbind(dim=1)
            rest = (w[0] * torch.log(im + eps)
                    + w[3] * torch.log(5 * ttc + 2 * comfort + 5 * ep + eps))
            return self._gate(rest, feas)

        nc, dac, ep, ttc, comfort = sim.unbind(dim=1)
        rest = (w[0] * torch.log(im + eps) + w[1] * torch.log(nc + eps)
                + w[3] * torch.log(5 * ttc + 2 * comfort + 5 * ep + eps))
        if self.select_mode == "weighted":
            full = rest + w[2] * torch.log(dac + eps)
            return full, full
        return self._gate(rest, dac)

    def _gate(self, rest: torch.Tensor, admit: torch.Tensor):
        """Push every inadmissible candidate below every admissible one, and order what is left.

        The gate only reorders; nothing downstream differentiates through it (plan_final_rewards
        feeds an argmax, and the feasibility weighting detaches it).  A scene with no admissible
        candidate keeps its ungated order, or the argmax would be arbitrary.
        """
        admissible = admit >= self.select_dac_thr
        any_ok = admissible.any(dim=-1, keepdim=True)
        return rest - self.select_gate * (~admissible & any_ok).to(rest.dtype), rest

    def score(self, h: torch.Tensor, offset: Optional[torch.Tensor] = None):
        """Queries after the planner layers [B, K, C] -> (im [B, K], sim [B, H, K], final, ungated).

        ``offset`` [B, K, T, D] is shown to the sim heads when ``score_sees_offset``; callers without
        one (the evaluation-only rescore of planner_head) score the query alone, as before.
        """
        im = torch.softmax(self.reward_head(h).squeeze(-1).float(), dim=-1)
        hs = h
        if self.score_sees_offset and offset is not None:
            hs = h + self.offset_embed(offset.detach().reshape(*offset.shape[:2], -1).to(h.dtype))
        sim = torch.cat([head(hs) for head in self.sim_reward_heads], dim=-1).permute(0, 2, 1).float().sigmoid()
        final, ungated = self.weighted_reward(im, sim)
        return im, sim, final, ungated

    def forward(self, h: torch.Tensor, init_speed: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Anchor queries after the planner layers [B, K, C] -> predictions.

        ``init_speed`` [B] (current longitudinal speed) is needed when ``kinematic``.
        """
        bs = h.shape[0]
        if self.kinematic:
            if init_speed is None:
                raise ValueError("kinematic anchor planner needs the current speed")
            anchors = self.trajectory_anchors.to(h.dtype).unsqueeze(0).expand(bs, -1, -1, -1)    # [B, K, T, 3]
            v0 = init_speed.to(h.dtype).view(bs, 1).expand(bs, self.num_anchors)
            base = poses_to_controls(anchors, v0, self.dt)                                        # [B, K, T, 2]
            delta = self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, 2)
            poses, _ = bicycle_rollout(clamp_controls(base + delta), v0, self.dt)
            # expressed as an offset from the anchor, so WoTE's losses apply unchanged
            offset = poses - anchors
        elif self.heading_from_xy:
            anchors = self.trajectory_anchors.to(h.dtype).unsqueeze(0).expand(bs, -1, -1, -1)    # [B, K, T, 3]
            xy = anchors[..., :2] + self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, 2)
            poses = bezier_xyyaw(xy, anchors[..., 2])
            offset = poses - anchors          # WoTE's losses apply unchanged; the heading term trains xy
        else:
            offset = self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, self.traj_dims)
            if self.traj_dims == 3:
                offset = torch.cat([offset[..., :2], offset[..., 2:].tanh() * np.pi], dim=-1)
        im, sim, final, ungated = self.score(h, offset)
        all_traj = self.trajectory_anchors.unsqueeze(0).to(offset.dtype) + offset          # [B, K, T, 3]
        k = min(self.topk, self.num_anchors)
        top = final.topk(k, dim=-1).indices                                               # [B, k]
        gather = top[:, :, None, None].expand(-1, -1, self.fut_ts, self.traj_dims)
        top_traj = all_traj.gather(1, gather)
        return {
            "trajectory": top_traj[:, 0],
            "plan_topk_trajectory": top_traj,
            "plan_topk_reward": final.gather(1, top),
            "plan_topk_index": top,
            "plan_final_rewards": final,
            "plan_rank_rewards": ungated,
            "trajectory_offset": offset,
            "im_rewards": im,
            "sim_rewards": sim,
            "trajectory_anchors": self.trajectory_anchors,
        }


def anchor_plan_losses(
    predictions: Dict[str, torch.Tensor],
    gt_trajectory: torch.Tensor,
    sim_reward: torch.Tensor,
    sim_valid: torch.Tensor,
    skip_sim_index: Optional[Sequence[int]] = None,
    label_cols: Optional[Sequence[Optional[int]]] = None,
    offset_hold_radius: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """WoTE's three planning terms.  gt_trajectory [B, T, 3]; sim_reward [B, 5, K]; sim_valid [B].

    ``label_cols`` maps each HEAD to a column of the label tensor, or None for a head that has no
    anchor label because something else supervises it (the corridor ``feasible`` head).  It is how
    a head set that is not WoTE's five stays aligned to WoTE's five-column label file.  Without it
    the heads are assumed to be the label's own columns and ``skip_sim_index`` drops some of them.

    ``offset_hold_radius`` adds ``offset_hold_loss``: the offset regression supervises only the
    winner, so the other K-1 offsets are free, yet their score heads are trained on the ANCHOR's label
    while the car drives anchor + offset.  The hold keeps those offsets within the radius (hinge^2 on
    the per-step xy norm), so the anchor label stays a description of what is emitted.
    """
    anchors = predictions["trajectory_anchors"]
    offset = predictions["trajectory_offset"]
    bs, k = offset.shape[:2]
    gt = gt_trajectory.reshape(bs, 1, -1).to(offset.dtype)
    flat = anchors.reshape(1, k, -1).to(offset.dtype)
    dist = torch.norm(flat - gt, dim=2)                                   # [B, K]
    winner = dist.argmin(dim=1)
    rows = torch.arange(bs, device=offset.device)
    gt_offset = gt[:, 0] - flat[0, winner]
    traj_offset_loss = F.l1_loss(offset.reshape(bs, k, -1)[rows, winner], gt_offset, reduction="mean")

    reward_target = torch.softmax(-dist.float(), dim=-1)
    im = predictions["im_rewards"].clamp(1e-6, 1 - 1e-6)
    im_reward_loss = -(reward_target * im.log()).sum() / bs

    eps = 1e-6
    sim = predictions["sim_rewards"]
    target = sim_reward.to(sim.dtype)
    if label_cols is not None:
        keep_h = [h for h, c in enumerate(label_cols) if c is not None]
        keep_c = [c for c in label_cols if c is not None]
        if len(label_cols) != sim.shape[1]:
            raise ValueError(f"label_cols has {len(label_cols)} entries for {sim.shape[1]} heads")
        sim, target = sim[:, keep_h], target[:, keep_c]
    bce = -(target * (sim + eps).log() + (1 - target) * (1 - sim + eps).log())   # [B, H', K]
    if skip_sim_index is not None:
        # that head is supervised elsewhere (feasibility.py computes its target on the REFINED
        # trajectory).  Leaving the anchor label on would give it two conflicting targets.
        drop = {skip_sim_index} if isinstance(skip_sim_index, int) else set(skip_sim_index)
        keep = [i for i in range(bce.shape[1]) if i not in drop]
        bce = bce[:, keep]
    valid = sim_valid.to(sim.dtype).view(bs, 1, 1)
    sim_reward_loss = (bce * valid).sum() / (valid.sum() * bce.shape[1] * bce.shape[2]).clamp(min=1.0) * 5
    out = {"traj_offset_loss": traj_offset_loss, "im_reward_loss": im_reward_loss,
           "sim_reward_loss": sim_reward_loss}
    if offset_hold_radius is not None:
        mag = offset[..., :2].float().norm(dim=-1)                                    # [B, K, T]
        others = torch.ones(bs, k, device=offset.device)
        others[rows, winner] = 0.0
        excess = F.relu(mag - float(offset_hold_radius)) ** 2
        out["offset_hold_loss"] = (excess * others[..., None]).sum() / (others.sum() * mag.shape[-1]).clamp(min=1.0)
        out["offset_mag_others"] = ((mag.mean(-1) * others).sum() / others.sum().clamp(min=1.0)).detach()
    return out
