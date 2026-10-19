"""Parallel detection + motion head (``ParaDetMotionHead``), mmdet-free.

One head produces both supervisions from a single forward over the shared BEV.
Its final detection and motion latents can also be returned for the planner;
the private decoder has no dependency on the planner or the map head.

Layout follows the nuScenes config: 300 object queries, a 3-layer deformable
DETR decoder with iterative reference-point refinement, then a 1-layer
agent<->agent decoder producing ``fut_mode`` trajectory hypotheses per agent.
There is deliberately no map cross-attention (PARA-Drive Fig. 4 edge (1)).

refiner v5 det-head options (design refiner_v5_design_final.md §3.2, SSR/report/97 §2-2; all GT-supervised, every
one off by default = the head above bit for bit: no module, no parameter, no RNG draw, no output or loss key):

* ``vel_code_weights`` (V1): loss-time weights of the velocity code rows 8, 9.
* ``vel_moving_only`` (V1b): the velocity rows of the box L1 only on matched GT that MOVE (speed >= ``stop_speed``
  and an agent class), so the static majority no longer pulls the regression to 0.  The positive count (avg factor)
  is unchanged.  Needs ``stop_branch``: a static object's velocity is then unsupervised by the L1, and only the
  gate below makes its deployed velocity learnable.
* ``stop_branch`` (V1c): per decoder layer ``Linear(embed, 1)`` on the layer's hidden state -> ``all_stop_logits``
  [L, B, Q].  The head's DEPLOYED velocity is ``v_q = (1 - sigmoid(stop_q)) v_hat_q``: ``all_bbox_preds[..., 8:10]``
  carries v_q (every reader of the box code -- aux records, dumps, the v5 student rows -- gets the gated value; there
  is no ungated path to read by accident), ``all_vel_raw`` [L, B, Q, 2] the raw v_hat that the L1 regresses (a
  hurdle decomposition: the L1 never trains the gate).  Label (matched queries): stop = GT speed < ``stop_speed``
  or a non-agent class; sigmoid focal x ``loss_stop_weight``; init bias -log(99) (gate open, v_q ~ v_hat).  A
  differentiable gate inside the head, not a selection gate.
* ``acc_head`` (ACC aux target of Stage-2 arm B; the critic's mode 7 and any T3 claim stay off): ``_mlp`` on the
  final det hidden state -> ``acc_preds`` [B, Q] (a_lon, m/s^2, along the box's own heading) and ``stop2s_logits``
  [B, Q].  Targets from the GT keyframe speeds of the matched track (ParaSSRTargetBuilder ``gt_kf_speed`` /
  ``gt_kf_present`` [B, A, 5], keyframes k = 0..4 at 0.5 k s, refiner/gt_future rule: non-agents 0):
  a_gt = (|v(1)| - |v(0)|) / 0.5 where both keyframes are annotated (Huber delta ``acc_huber_delta`` x
  ``loss_acc_code_weight`` inside the box term's ``loss_bbox_weight``, final-layer matches, unclipped);
  stop2s = 1 if an annotated keyframe k <= 4 has speed < ``stop_speed``, 0 if all five are annotated and none does,
  no target otherwise (a track that leaves before 2 s without stopping is unknown, not a negative); focal x
  ``loss_stop2s_weight``.  Separate heads, not code_size 11: the 10-D code is a fixed contract of the checkpoints,
  the aux evaluator, the dumps and the iterative refinement.

The new terms are masked sums over all B x Q queries (every new parameter is in the graph each step: DDP-safe) and
reuse the existing avg factors (no new collective).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    distributed_avg_factor,
    hungarian_assign_det,
    normalize_bbox,
    sigmoid_focal_loss,
    weighted_l1_loss,
)
from .ms_deform_attn import CustomMSDeformableAttention
from .transformer_blocks import (
    BaseTransformerLayer,
    MultiheadAttention,
    build_self_attn_decoder,
    inverse_sigmoid,
)

# Detection class index < N_AGENT_CLASSES = nuPlan agent types vehicle / pedestrian / bicycle
# (para_ssr_targets.DET_CLASS_NAMES[:3] = refiner.gt_future.AGENT_CLASSES without ego); every other class is static.
N_AGENT_CLASSES = 3
ACC_N_KF = 5            # ACC head targets: keyframes k = 0..4 (0..2 s)
ACC_KF_DT = 0.5         # keyframe spacing [s] (index based, refiner.gt_future.KF_DT)


def _mlp(in_dim: int, hidden: int, out_dim: int, num_fcs: int = 2) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(num_fcs):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        in_dim = hidden
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _cls_mlp(
    in_dim: int, hidden: int, out_dim: int, num_fcs: int = 2
) -> nn.Sequential:
    """Original SSR classification branch: Linear-LN-ReLU blocks."""
    layers: List[nn.Module] = []
    for _ in range(num_fcs):
        layers.extend(
            [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
            ]
        )
        in_dim = hidden
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _last_valid_fde(
    pred_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-mode FDE at each agent's last annotated future step.

    Both prediction and target tensors contain per-step offsets, so FDE is
    measured only after accumulating them into positions.  Rows without any
    valid future return zero FDE together with ``has_future=False``; callers
    must exclude those rows from trajectory regression and classification.
    """
    pred_positions = pred_offsets.cumsum(dim=-2)
    target_positions = target_offsets.cumsum(dim=-2)
    distances = torch.linalg.vector_norm(
        pred_positions - target_positions[:, None], dim=-1
    )

    valid = valid_mask.bool()
    steps = torch.arange(valid.size(-1), device=valid.device)
    last_valid = torch.where(valid, steps, steps.new_full((), -1)).amax(dim=-1)
    has_future = last_valid >= 0
    safe_last = last_valid.clamp(min=0)
    fde = distances.gather(
        dim=-1,
        index=safe_last[:, None, None].expand(-1, distances.size(1), 1),
    ).squeeze(-1)
    fde = torch.where(has_future[:, None], fde, torch.zeros_like(fde))
    return fde, has_future


class DetectionTransformerDecoder(nn.Module):
    """Deformable DETR decoder with iterative box refinement."""

    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        num_levels: int = 1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            attentions = [
                MultiheadAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    proj_drop=0.0,
                    dropout_layer=attn_dropout,
                ),
                CustomMSDeformableAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    dropout=attn_dropout,
                    batch_first=False,
                ),
            ]
            layers.append(
                BaseTransformerLayer(
                    attentions=attentions,
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels,
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ffn_dropout=ffn_dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: nn.ModuleList,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            # ``output`` is sequence-first; the branches and the reference
            # points are batch-first.
            tmp = reg_branches[lid](output.permute(1, 0, 2))
            new_reference_points = torch.zeros_like(reference_points)
            new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
            new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
            reference_points = new_reference_points.sigmoid().detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)
        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


class ParaDetMotionHead(nn.Module):
    def __init__(
        self,
        num_query: int = 300,
        num_classes: int = 7,
        embed_dims: int = 256,
        bev_h: int = 100,
        bev_w: int = 100,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        code_size: int = 10,
        code_weights: Optional[Sequence[float]] = None,
        num_reg_fcs: int = 2,
        fut_ts: int = 8,
        fut_mode: int = 6,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        use_pe: bool = True,
        sync_cls_avg_factor: bool = True,
        loss_weight: float = 1.0,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        loss_traj_weight: float = 0.2,
        loss_traj_cls_weight: float = 0.2,
        assigner_cls_weight: float = 2.0,
        assigner_reg_weight: float = 0.25,
        vel_code_weights: Optional[Sequence[float]] = None,
        vel_moving_only: bool = False,
        stop_branch: bool = False,
        stop_speed: float = 0.5,
        loss_stop_weight: float = 1.0,
        acc_head: bool = False,
        loss_acc_code_weight: float = 0.5,
        loss_stop2s_weight: float = 0.2,
        acc_huber_delta: float = 1.0,
    ):
        super().__init__()
        # refiner v5 V1 (report 97 §2-2): loss-time (vx, vy) code weights replacing code_weights[8:10] (None = unchanged).
        # A plain attribute, not the code_weights buffer: a checkpoint load (buffers) cannot silently undo it.
        self.vel_code_weights = None if vel_code_weights is None else tuple(float(v) for v in vel_code_weights)
        # refiner v5 V1b / V1c / ACC head (module docstring); plain attributes, the modules are built after _init_weights
        if (vel_moving_only or stop_branch) and code_size < 10:
            raise ValueError(f"det V1b / V1c need the velocity code rows 8, 9 (code_size >= 10), got {code_size}")
        if vel_moving_only and not stop_branch:
            raise ValueError("det V1b (vel_moving_only) needs the V1c stop_branch: static objects would get no velocity "
                             "supervision and an ungated deployed velocity")
        if (vel_moving_only or stop_branch or acc_head) and not float(stop_speed) > 0.0:
            raise ValueError(f"det stop_speed must be > 0 m/s, got {stop_speed}")
        if min(float(loss_stop_weight), float(loss_acc_code_weight), float(loss_stop2s_weight)) < 0.0:
            raise ValueError("det loss_stop_weight / loss_acc_code_weight / loss_stop2s_weight must be >= 0")
        if acc_head and not float(acc_huber_delta) > 0.0:
            raise ValueError(f"det acc_huber_delta must be > 0 m/s^2, got {acc_huber_delta}")
        self.vel_moving_only = bool(vel_moving_only)
        self.stop_speed = float(stop_speed)
        self.loss_stop_weight = float(loss_stop_weight)
        self.loss_acc_code_weight = float(loss_acc_code_weight)
        self.loss_stop2s_weight = float(loss_stop2s_weight)
        self.acc_huber_delta = float(acc_huber_delta)
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = list(pc_range)
        self.code_size = code_size
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.use_pe = use_pe
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_weight = loss_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_traj_weight = loss_traj_weight
        self.loss_traj_cls_weight = loss_traj_cls_weight
        self.assigner_cls_weight = assigner_cls_weight
        self.assigner_reg_weight = assigner_reg_weight

        if code_weights is None:
            code_weights = [1.0] * 8 + [0.2, 0.2]
        self.register_buffer(
            "code_weights", torch.tensor(code_weights[:code_size], dtype=torch.float32)
        )

        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)
        self.reference_points = nn.Linear(embed_dims, 3)

        self.decoder = DetectionTransformerDecoder(
            num_decoder_layers, embed_dims, num_heads, feedforward_channels
        )

        cls_branch = _cls_mlp(embed_dims, embed_dims, num_classes, num_reg_fcs)
        reg_branch = _mlp(embed_dims, embed_dims, code_size, num_reg_fcs)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_decoder_layers)]
        )
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)]
        )

        # agent <-> agent interaction only
        self.motion_decoder = build_self_attn_decoder(
            1,
            embed_dims,
            num_heads,
            feedforward_channels,
            ("cross_attn", "norm", "ffn", "norm"),
            attn_dropout=0.1,
            ffn_dropout=0.1,
        )
        self.motion_mode_query = nn.Embedding(fut_mode, embed_dims)
        self.pos_mlp_sa = nn.Linear(2, embed_dims) if use_pe else None
        self.traj_branch = _mlp(
            embed_dims, embed_dims, fut_ts * 2, num_reg_fcs
        )
        self.traj_cls_branch = _cls_mlp(
            embed_dims, embed_dims, 1, num_reg_fcs
        )

        self._init_weights()

        # refiner v5 V1c / ACC head: built LAST, on a forked RNG stream, so every module above (and everything the
        # caller builds after this head) gets the same initialisation with the options on or off
        self.stop_branches: Optional[nn.ModuleList] = None
        self.acc_head: Optional[nn.Sequential] = None
        if stop_branch or acc_head:
            with torch.random.fork_rng(devices=[]):
                if stop_branch:
                    self.stop_branches = nn.ModuleList(
                        [nn.Linear(embed_dims, 1) for _ in range(num_decoder_layers)]
                    )
                    stop_bias = float(-torch.log(torch.tensor((1 - 0.01) / 0.01)))   # sigmoid = 0.01: gate open
                    for branch in self.stop_branches:
                        nn.init.xavier_uniform_(branch.weight)
                        nn.init.constant_(branch.bias, stop_bias)
                if acc_head:
                    self.acc_head = _mlp(embed_dims, embed_dims, 2, num_reg_fcs)
                    nn.init.constant_(self.acc_head[-1].bias, 0.0)                 # a_lon = 0 (CV) at init

    def _init_weights(self) -> None:
        # Match the BaseModule init hook used by the original heads.  Xavier on
        # the whole decoder overwrites deformable attention's special zero/radial
        # initialization, so restore that module immediately afterwards.
        for decoder in (self.decoder, self.motion_decoder):
            for parameter in decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        for module in self.decoder.modules():
            if isinstance(module, CustomMSDeformableAttention):
                module.init_weights()
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)
        bias_init = float(-torch.log(torch.tensor((1 - 0.01) / 0.01)))
        for branch in self.cls_branches:
            nn.init.constant_(branch[-1].bias, bias_init)

    def forward(
        self, bev_embed: torch.Tensor, return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_embed: ``[bs, bev_h * bev_w, embed_dims]``
            return_hidden: Include attached final decoder latents as
                ``det_hidden [B, Q, C]`` and ``motion_hidden [B, Q, M, C]``.
                Box predictions contain metric XY centres; classification
                predictions contain foreground-only sigmoid logits.
        """
        bs = bev_embed.size(0)
        device = bev_embed.device

        query_embed = self.query_embedding.weight.to(bev_embed.dtype)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        # ``use_pe`` belongs to the motion box-centre PE below.  The original
        # detection decoder consumes the shared BEV directly.
        value = bev_embed

        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], dtype=torch.long, device=device
        )
        level_start_index = torch.tensor([0], dtype=torch.long, device=device)

        inter_states, inter_references = self.decoder(
            query=query.permute(1, 0, 2),
            key=value.permute(1, 0, 2),
            value=value.permute(1, 0, 2),
            query_pos=query_pos.permute(1, 0, 2),
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        outputs_classes, outputs_coords, outputs_coords_bev = [], [], []
        stop_logits, vel_raw = [], []
        for lvl in range(inter_states.shape[0]):
            hidden = inter_states[lvl].permute(1, 0, 2)  # [bs, num_query, C]
            reference = (
                reference_points if lvl == 0 else inter_references[lvl - 1]
            )
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hidden)
            tmp = self.reg_branches[lvl](hidden)
            tmp = tmp.clone()
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # Motion PE uses normalized box centres and deliberately detaches
            # them, as in SSR, so trajectory loss does not update bbox centres.
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # Canonical raw code:
            # [x, y, logW, logL, z, logH, sin(yaw), cos(yaw), vx, vy].
            # Only x/y/z are reference-point-normalised and need conversion to
            # metric coordinates.  Every other channel is already decoder code.
            coord = tmp.clone()
            coord[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            coord[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            if self.stop_branches is not None:
                # refiner v5 V1c: the deployed velocity v_q = (1 - sigmoid(stop_q)) v_hat_q; tmp keeps the raw v_hat
                stop = self.stop_branches[lvl](hidden).squeeze(-1)                # [bs, num_query]
                coord[..., 8:10] = tmp[..., 8:10] * (1.0 - stop.sigmoid())[..., None]
                stop_logits.append(stop)
                vel_raw.append(tmp[..., 8:10])
            outputs_classes.append(outputs_class)
            outputs_coords.append(coord)

        # One token per (agent query, motion mode), q-major/mode-minor.  This is
        # materially different from predicting every mode with one final MLP:
        # the original decoder lets modes interact as separate tokens.
        motion_query = inter_states[-1]  # [Q, B, C]
        mode_query = self.motion_mode_query.weight.to(motion_query.dtype)
        motion_query = (
            motion_query[:, None, :, :] + mode_query[None, :, None, :]
        ).flatten(0, 1)  # [Q*M, B, C]

        if self.pos_mlp_sa is not None:
            motion_pos = self.pos_mlp_sa(outputs_coords_bev[-1])  # [B, Q, C]
            motion_pos = (
                motion_pos.unsqueeze(2)
                .repeat(1, 1, self.fut_mode, 1)
                .flatten(1, 2)
                .permute(1, 0, 2)
            )
        else:
            motion_pos = None

        motion_hs = self.motion_decoder(
            query=motion_query,
            key=motion_query,
            value=motion_query,
            query_pos=motion_pos,
            key_pos=motion_pos,
        )
        motion_hs = motion_hs.permute(1, 0, 2).reshape(
            bs, self.num_query, self.fut_mode, self.embed_dims
        )
        traj = self.traj_branch(motion_hs).reshape(
            bs, self.num_query, self.fut_mode, self.fut_ts, 2
        )
        traj_cls = self.traj_cls_branch(motion_hs).squeeze(-1)

        outputs = {
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "traj_preds": traj,
            "traj_cls_preds": traj_cls,
        }
        if self.stop_branches is not None:              # refiner v5 V1c (all_bbox_preds holds the gated velocity)
            outputs["all_stop_logits"] = torch.stack(stop_logits)      # [L, bs, num_query]
            outputs["all_vel_raw"] = torch.stack(vel_raw)              # [L, bs, num_query, 2] raw v_hat
        if self.acc_head is not None:                   # refiner v5 ACC aux head on the final det hidden state
            acc = self.acc_head(inter_states[-1].permute(1, 0, 2))
            outputs["acc_preds"] = acc[..., 0]                         # [bs, num_query] a_lon [m/s^2]
            outputs["stop2s_logits"] = acc[..., 1]                     # [bs, num_query] logit of 'stops within 2 s'
        if return_hidden:
            outputs["det_hidden"] = inter_states[-1].permute(1, 0, 2)
            outputs["motion_hidden"] = motion_hs
        return outputs

    # ------------------------------------------------------------------ #
    # loss
    # ------------------------------------------------------------------ #
    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_valid: torch.Tensor,
        gt_fut_trajs: Optional[torch.Tensor] = None,
        gt_fut_masks: Optional[torch.Tensor] = None,
        gt_kf_speed: Optional[torch.Tensor] = None,
        gt_kf_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gt_bboxes: ``[bs, max_agents, 9]`` ``(x, y, z, w, l, h, yaw, vx, vy)``
            gt_labels: ``[bs, max_agents]`` class index
            gt_valid: ``[bs, max_agents]`` bool
            gt_fut_trajs: ``[bs, max_agents, fut_ts, 2]`` future offsets
            gt_fut_masks: ``[bs, max_agents, fut_ts]``
            gt_kf_speed / gt_kf_present: ``[bs, max_agents, 5]`` GT keyframe speeds / annotation flags of each slot
                (k = 0..4; ACC head only, required when it is built)
        """
        if self.stop_branches is not None:
            miss = [k for k in ("all_stop_logits", "all_vel_raw") if k not in preds]
            if miss:
                raise KeyError(f"det stop_branch (V1c): predictions lack {miss} (this head's forward outputs)")
        if self.acc_head is not None:
            miss = [k for k in ("acc_preds", "stop2s_logits") if k not in preds]
            if miss:
                raise KeyError(f"det acc_head: predictions lack {miss} (this head's forward outputs)")
            if gt_kf_speed is None or gt_kf_present is None:
                raise KeyError("det_acc_head needs gt_kf_speed / gt_kf_present (ParaSSRTargetBuilder with "
                               "det_acc_head)")
            want = tuple(gt_bboxes.shape[:2]) + (ACC_N_KF,)
            if tuple(gt_kf_speed.shape) != want or tuple(gt_kf_present.shape) != want:
                raise ValueError(f"det acc_head: gt_kf_speed / gt_kf_present must be {want}, got "
                                 f"{tuple(gt_kf_speed.shape)} / {tuple(gt_kf_present.shape)}")
        # Autocast predictions may be FP16/BF16 while cached GT is FP32.
        # Match and reduce losses in at least FP32: this both keeps indexed
        # target assignments type-safe and avoids low-precision focal costs.
        # The casts stay attached to the original decoder predictions.
        loss_dtype = torch.promote_types(preds["all_bbox_preds"].dtype, gt_bboxes.dtype)
        loss_dtype = torch.promote_types(loss_dtype, torch.float32)
        all_cls_scores = preds["all_cls_scores"].to(dtype=loss_dtype)
        all_bbox_preds = preds["all_bbox_preds"].to(dtype=loss_dtype)
        gt_bboxes = gt_bboxes.to(dtype=loss_dtype)
        if gt_fut_trajs is not None:
            gt_fut_trajs = gt_fut_trajs.to(dtype=loss_dtype)
        num_layers = all_cls_scores.size(0)
        device = all_cls_scores.device

        losses: Dict[str, torch.Tensor] = {}
        last_assign: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for lvl in range(num_layers):
            cls_scores = torch.nan_to_num(
                all_cls_scores[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            bbox_preds = torch.nan_to_num(
                all_bbox_preds[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            bs = cls_scores.size(0)

            labels_list, weights_list = [], []
            bbox_targets, bbox_weights = [], []
            stop_targets, stop_weights = [], []
            num_pos_total = 0
            assigns = []
            for b in range(bs):
                valid = gt_valid[b].bool()
                gt_b = gt_bboxes[b][valid]
                gtl_b = gt_labels[b][valid].long()
                gt_norm = normalize_bbox(gt_b) if gt_b.numel() else gt_b.new_zeros((0, self.code_size))

                q_idx, g_idx = hungarian_assign_det(
                    cls_scores[b],
                    bbox_preds[b],
                    gt_norm,
                    gtl_b,
                    cls_weight=self.assigner_cls_weight,
                    reg_weight=self.assigner_reg_weight,
                )
                assigns.append((q_idx, g_idx))
                num_pos_total += len(q_idx)

                labels = torch.full(
                    (self.num_query,), self.num_classes, dtype=torch.long, device=device
                )
                label_w = torch.ones(self.num_query, device=device)
                bt = bbox_preds.new_zeros((self.num_query, self.code_size))
                bw = bbox_preds.new_zeros((self.num_query, self.code_size))
                if len(q_idx):
                    labels[q_idx] = gtl_b[g_idx]
                    bt[q_idx] = gt_norm[g_idx][:, : self.code_size]
                    bw[q_idx] = 1.0
                    if self.vel_moving_only:
                        # refiner v5 V1b: velocity rows only on matched GT that moves (avg factor unchanged)
                        moving = (torch.hypot(gt_b[:, 7], gt_b[:, 8]) >= self.stop_speed) & (gtl_b < N_AGENT_CLASSES)
                        bw[q_idx, 8:10] = bw[q_idx, 8:10] * moving[g_idx].to(bw.dtype)[:, None]
                if self.stop_branches is not None:
                    # refiner v5 V1c stop label of the matched queries: class 0 = stop, 1 (background) = moving
                    st = torch.ones(self.num_query, dtype=torch.long, device=device)
                    sw = bbox_preds.new_zeros(self.num_query)
                    if len(q_idx):
                        is_stop = (torch.hypot(gt_b[:, 7], gt_b[:, 8]) < self.stop_speed) | (gtl_b >= N_AGENT_CLASSES)
                        st[q_idx] = torch.where(is_stop[g_idx], torch.zeros_like(st[q_idx]), torch.ones_like(st[q_idx]))
                        sw[q_idx] = 1.0
                    stop_targets.append(st)
                    stop_weights.append(sw)
                labels_list.append(labels)
                weights_list.append(label_w)
                bbox_targets.append(bt)
                bbox_weights.append(bw)

            if lvl == num_layers - 1:
                last_assign = assigns

            labels = torch.cat(labels_list)
            label_weights = torch.cat(weights_list)
            bbox_targets_t = torch.cat(bbox_targets)
            bbox_weights_t = torch.cat(bbox_weights)

            cls_avg_factor = max(float(num_pos_total), 1.0)
            if self.sync_cls_avg_factor:
                cls_avg_factor = distributed_avg_factor(num_pos_total, cls_scores)
            loss_cls = sigmoid_focal_loss(
                cls_scores.reshape(-1, self.num_classes),
                labels,
                label_weights,
                avg_factor=max(cls_avg_factor, 1.0),
            ) * self.loss_cls_weight

            # Decoder predictions already use the canonical regression code.
            # Encoding them again would log size/position channels a second
            # time and move z away from code index 4.
            if self.stop_branches is not None:
                # refiner v5 V1c: the L1 regresses the RAW velocity v_hat (all_bbox_preds holds the gated one)
                vel_raw = torch.nan_to_num(
                    preds["all_vel_raw"][lvl].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0
                )
                pred_code_all = torch.cat(
                    [bbox_preds[..., :8], vel_raw, bbox_preds[..., 10:]], dim=-1
                ).reshape(-1, self.code_size)
            else:
                pred_code_all = bbox_preds.reshape(-1, self.code_size)
            cw = self.code_weights.to(bbox_weights_t.dtype)
            if self.vel_code_weights is not None:            # refiner v5 V1: sigma-normalised velocity weights
                cw = torch.cat([cw[:8], torch.as_tensor(self.vel_code_weights, dtype=cw.dtype, device=cw.device),
                                cw[10:]])
            bbox_weights_t = bbox_weights_t * cw
            bbox_avg_factor = distributed_avg_factor(num_pos_total, pred_code_all)
            loss_bbox = weighted_l1_loss(
                torch.nan_to_num(pred_code_all),
                torch.nan_to_num(bbox_targets_t),
                bbox_weights_t,
                avg_factor=bbox_avg_factor,
            ) * self.loss_bbox_weight

            suffix = "" if lvl == num_layers - 1 else f"_d{lvl}"
            losses[f"loss_cls{suffix}"] = torch.nan_to_num(loss_cls) * self.loss_weight
            losses[f"loss_bbox{suffix}"] = torch.nan_to_num(loss_bbox) * self.loss_weight
            if self.stop_branches is not None:
                # masked over every B * Q query (each branch stays in the graph with zero GT); the layer's box avg factor
                stop_logit = torch.nan_to_num(
                    preds["all_stop_logits"][lvl].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0
                )
                loss_stop = sigmoid_focal_loss(
                    stop_logit.reshape(-1, 1),
                    torch.cat(stop_targets),
                    torch.cat(stop_weights),
                    avg_factor=bbox_avg_factor,
                ) * self.loss_stop_weight
                losses[f"loss_stop{suffix}"] = torch.nan_to_num(loss_stop) * self.loss_weight

        # ---- motion, on the final layer's matching -----------------------
        # Regression uses matched queries.  Classification follows the original
        # SSR head and uses every B*Q query: unmatched detection queries are the
        # motion background class, while matched queries without any annotated
        # future are the only entries whose classification weight is zero.
        traj_preds = torch.nan_to_num(
            preds["traj_preds"].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        traj_cls_preds = torch.nan_to_num(
            preds["traj_cls_preds"].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        loss_traj_sum = traj_preds.sum() * 0.0
        traj_labels = torch.full(
            traj_cls_preds.shape[:2],
            self.fut_mode,
            dtype=torch.long,
            device=device,
        )
        traj_cls_weights = traj_cls_preds.new_ones(traj_cls_preds.shape[:2])
        num_detection_pos = sum(int(q_idx.numel()) for q_idx, _ in last_assign)

        for b, (q_idx, g_idx) in enumerate(last_assign):
            if len(q_idx) == 0:
                continue
            if gt_fut_trajs is None or gt_fut_masks is None:
                # A matched query without future supervision is neither a mode
                # target nor background; exclude it from motion classification.
                traj_cls_weights[b, q_idx] = 0.0
                continue

            # g_idx indexes the compact, valid-GT list used by Hungarian.
            valid_gt = gt_valid[b].bool()
            gt_traj = gt_fut_trajs[b][valid_gt][g_idx]  # [n, fut_ts, 2]
            gt_mask = gt_fut_masks[b][valid_gt][g_idx]  # [n, fut_ts]
            pred = traj_preds[b][q_idx]                 # [n, mode, fut_ts, 2]
            fde, has_future = _last_valid_fde(pred, gt_traj, gt_mask)
            best = fde.argmin(dim=-1)
            rows = torch.arange(best.numel(), device=device)
            best_pred = pred[rows, best]
            weights = gt_mask[..., None].expand_as(best_pred)

            # Zero-future positives contribute a zero numerator but remain in
            # the original detection-positive denominator.
            loss_traj_sum = loss_traj_sum + weighted_l1_loss(
                best_pred, gt_traj, weights, avg_factor=1.0
            )
            traj_labels[b, q_idx[has_future]] = best[has_future]
            traj_cls_weights[b, q_idx[~has_future]] = 0.0

        motion_avg_factor = distributed_avg_factor(num_detection_pos, traj_preds)
        loss_traj_cls_sum = sigmoid_focal_loss(
            traj_cls_preds.reshape(-1, self.fut_mode),
            traj_labels.reshape(-1),
            traj_cls_weights.reshape(-1),
            avg_factor=1.0,
        )
        losses["loss_traj"] = torch.nan_to_num(
            loss_traj_sum / motion_avg_factor
            * self.loss_traj_weight
            * self.loss_weight
        )
        losses["loss_traj_cls"] = torch.nan_to_num(
            loss_traj_cls_sum / motion_avg_factor
            * self.loss_traj_cls_weight
            * self.loss_weight
        )

        if self.acc_head is not None:
            losses.update(self._acc_losses(preds, gt_kf_speed, gt_kf_present, gt_valid, last_assign,
                                           motion_avg_factor, loss_dtype))
        return losses

    def _acc_losses(
        self,
        preds: Dict[str, torch.Tensor],
        gt_kf_speed: torch.Tensor,
        gt_kf_present: torch.Tensor,
        gt_valid: torch.Tensor,
        last_assign: List[Tuple[torch.Tensor, torch.Tensor]],
        avg_factor: float,
        loss_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """refiner v5 ACC aux head (module docstring): Huber on a_lon and focal on stop2s at the final layer's matches,
        masked over every B * Q query, divided by the final layer's (motion) positive count."""
        acc = torch.nan_to_num(preds["acc_preds"].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0)
        s2 = torch.nan_to_num(preds["stop2s_logits"].to(dtype=loss_dtype), nan=0.0, posinf=0.0, neginf=0.0)
        a_t, a_w, s_w = torch.zeros_like(acc), torch.zeros_like(acc), torch.zeros_like(acc)
        s_t = torch.ones(acc.shape, dtype=torch.long, device=acc.device)       # class 0 = stops, 1 = background
        for b, (q_idx, g_idx) in enumerate(last_assign):
            if len(q_idx) == 0:
                continue
            valid_gt = gt_valid[b].bool()
            sp = gt_kf_speed[b][valid_gt][g_idx].to(dtype=loss_dtype)          # [n, 5] |v| at k = 0..4
            pr = gt_kf_present[b][valid_gt][g_idx].bool()
            a_ok = pr[:, 0] & pr[:, 1]
            a_t[b, q_idx] = torch.where(a_ok, (sp[:, 1] - sp[:, 0]) / ACC_KF_DT, torch.zeros_like(sp[:, 0]))
            a_w[b, q_idx] = a_ok.to(dtype=loss_dtype)
            seen_stop = (pr & (sp < self.stop_speed)).any(dim=1)
            s_t[b, q_idx] = torch.where(seen_stop, torch.zeros_like(s_t[b, q_idx]), torch.ones_like(s_t[b, q_idx]))
            s_w[b, q_idx] = (seen_stop | pr.all(dim=1)).to(dtype=loss_dtype)   # left before 2 s, no stop: unknown
        huber = F.huber_loss(acc, a_t, reduction="none", delta=self.acc_huber_delta)
        loss_acc = (huber * a_w).sum() / max(avg_factor, 1e-6)
        loss_stop2s = sigmoid_focal_loss(s2.reshape(-1, 1), s_t.reshape(-1), s_w.reshape(-1), avg_factor=avg_factor)
        return {
            "loss_acc": torch.nan_to_num(
                loss_acc * self.loss_acc_code_weight * self.loss_bbox_weight * self.loss_weight
            ),
            "loss_stop2s": torch.nan_to_num(loss_stop2s * self.loss_stop2s_weight * self.loss_weight),
        }
