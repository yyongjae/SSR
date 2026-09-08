"""PARA-SSR model: shared BEV encoder with three parallel heads.

Mirrors ``projects/mmdet3d_plugin/SSR/para_ssr.py``:

* history frames build ``prev_bev`` under ``no_grad`` (they exist to give the
  temporal self-attention something to attend to, not to be trained through);
* the current frame's BEV is produced by the planning head, which owns the
  encoder;
* that one ``bev_embed`` fans out to the detection/motion head and the map head.
  Neither feeds the other, nor the planner -- the only coupling is the shared
  feature (PARA-Drive Fig. 5);
* every auxiliary head sees ``bev_embed`` through ``_ScaleGrad``, so its own
  parameters train at full strength while its influence on the shared feature
  is throttled independently.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bevformer import BEVFormerEncoder, SSRPerceptionTransformer
from .modules.det_motion_head import ParaDetMotionHead
from .modules.map_head import ParaMapHead
from .modules.planner_head import ParaSSRPlannerHead


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; scales the gradient flowing backwards.

    Down-weighting an auxiliary *loss* would slow the head itself, which is the
    opposite of what is wanted when the heads are meant to become teachers.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class GridMask(nn.Module):
    """SSR's input augmentation: erase a regular grid of image patches."""

    def __init__(self, ratio: float = 0.5, prob: float = 0.7, mode: int = 1):
        super().__init__()
        if mode not in (0, 1):
            raise ValueError(f"mode must be 0 or 1, got {mode}")
        self.ratio = ratio
        self.prob = prob
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.prob:
            return x
        n, c, h, w = x.shape
        if h <= 2:
            raise ValueError(f"GridMask requires image height > 2, got {h}")

        # Match SSR's GridMask(True, True, rotate=1, ratio=.5, mode=1):
        # sample d over [2, h), build a 1.5x canvas, then centre-crop it.
        hh, ww = int(1.5 * h), int(1.5 * w)
        d = int(torch.randint(2, h, (1,)).item())
        length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = torch.ones((hh, ww), device=x.device, dtype=x.dtype)
        start_h = int(torch.randint(0, d, (1,)).item())
        start_w = int(torch.randint(0, d, (1,)).item())
        for i in range(hh // d):
            start = d * i + start_h
            mask[start : min(start + length, hh), :] = 0
        for i in range(ww // d):
            start = d * i + start_w
            mask[:, start : min(start + length, ww)] = 0
        crop_h, crop_w = (hh - h) // 2, (ww - w) // 2
        mask = mask[crop_h : crop_h + h, crop_w : crop_w + w]
        if self.mode == 1:
            mask = 1 - mask
        return x * mask[None, None]


class SingleLevelFPN(nn.Module):
    """``FPN(in_channels=[2048], out_channels=256, num_outs=1)``, unrolled."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        for m in (self.lateral, self.fpn_conv):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fpn_conv(self.lateral(x))


class ParaSSRModel(nn.Module):
    _BACKBONE_STAGE_MODULES = (
        ("conv1", "bn1", "act1", "maxpool"),
        ("layer1",),
        ("layer2",),
        ("layer3",),
        ("layer4",),
    )

    def __init__(self, config):
        super().__init__()
        self._config = config
        cfg = config
        self._backbone_frozen_stages = int(getattr(cfg, "frozen_stages", -1))
        self._backbone_norm_requires_grad = bool(
            getattr(cfg, "norm_requires_grad", False)
        )
        self._backbone_norm_eval = bool(getattr(cfg, "norm_eval", False))

        self.image_encoder = timm.create_model(
            cfg.image_architecture,
            pretrained=bool(getattr(cfg, "backbone_pretrained", True)),
            features_only=True,
            out_indices=cfg.backbone_out_indices,
        )
        self._apply_backbone_train_policy()
        backbone_channels = self.image_encoder.feature_info.channels()[-1]
        self.img_neck = SingleLevelFPN(backbone_channels, cfg.embed_dims)
        self.grid_mask = GridMask() if cfg.use_grid_mask else None

        encoder = BEVFormerEncoder(
            num_layers=cfg.encoder_num_layers,
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            num_cams=cfg.num_cams,
            pc_range=cfg.pc_range,
            num_points_in_pillar=cfg.encoder_num_points_in_pillar,
            num_points_sca=cfg.encoder_num_points_sca,
            num_levels=cfg.num_feature_levels,
            feedforward_channels=cfg.ffn_channels,
            ffn_dropout=cfg.encoder_ffn_dropout,
            attn_dropout=cfg.encoder_attn_dropout,
        )
        transformer = SSRPerceptionTransformer(
            embed_dims=cfg.embed_dims,
            num_cams=cfg.num_cams,
            num_feature_levels=cfg.num_feature_levels,
            ego_motion_dims=cfg.ego_motion_dims,
            use_ego_motion=cfg.use_ego_motion,
            ego_motion_norm=cfg.ego_motion_norm,
            use_shift=cfg.use_shift,
            use_cams_embeds=cfg.use_cams_embeds,
            encoder=encoder,
        )
        self.pts_bbox_head = ParaSSRPlannerHead(
            transformer=transformer,
            bev_h=cfg.bev_h,
            bev_w=cfg.bev_w,
            embed_dims=cfg.embed_dims,
            pc_range=cfg.pc_range,
            num_scenes=cfg.num_scenes,
            num_reg_fcs=cfg.num_reg_fcs,
            fut_ts=cfg.fut_ts,
            ego_fut_mode=cfg.ego_fut_mode,
            num_navi_cmd=cfg.num_navi_cmd,
            traj_dims=cfg.traj_dims,
            latent_num_layers=cfg.latent_num_layers,
            way_num_layers=cfg.way_num_layers,
            num_heads=cfg.num_heads,
            feedforward_channels=cfg.ffn_channels,
            use_metric_planner=cfg.use_metric_planner,
            num_plan_candidates=cfg.num_plan_candidates,
            plan_anchor_path=cfg.plan_anchor_path,
        )

        if cfg.use_metric_planner:
            from .modules.candidate_planner import CandidateMetricHead
            self.metric_head = CandidateMetricHead(
                cfg.embed_dims, cfg.num_heads, cfg.fut_ts, cfg.metric_detach_bev
            )

        self.det_motion_head = (
            ParaDetMotionHead(
                num_query=cfg.num_query,
                num_classes=cfg.num_det_classes,
                embed_dims=cfg.embed_dims,
                bev_h=cfg.bev_h,
                bev_w=cfg.bev_w,
                pc_range=cfg.pc_range,
                code_size=cfg.det_code_size,
                code_weights=cfg.det_code_weights,
                num_reg_fcs=cfg.num_reg_fcs,
                fut_ts=cfg.fut_ts,
                fut_mode=cfg.fut_mode,
                num_decoder_layers=cfg.det_num_decoder_layers,
                num_heads=cfg.num_heads,
                feedforward_channels=cfg.ffn_channels,
                use_pe=cfg.det_use_pe,
                loss_cls_weight=cfg.loss_cls_weight,
                loss_bbox_weight=cfg.loss_bbox_weight,
                loss_traj_weight=cfg.loss_traj_weight,
                loss_traj_cls_weight=cfg.loss_traj_cls_weight,
            )
            if cfg.use_det_motion_head
            else None
        )

        self.map_head = (
            ParaMapHead(
                map_num_vec=cfg.map_num_vec,
                map_num_pts_per_vec=cfg.map_num_pts_per_vec,
                map_num_classes=cfg.map_num_classes,
                embed_dims=cfg.embed_dims,
                bev_h=cfg.bev_h,
                bev_w=cfg.bev_w,
                # map and shared BEV use the same front-only physical extent
                pc_range=cfg.map_pc_range,
                num_reg_fcs=cfg.num_reg_fcs,
                num_decoder_layers=cfg.map_num_decoder_layers,
                num_heads=cfg.num_heads,
                feedforward_channels=cfg.ffn_channels,
                map_dir_interval=cfg.map_dir_interval,
                loss_map_cls_weight=cfg.loss_map_cls_weight,
                loss_map_pts_weight=cfg.loss_map_pts_weight,
                loss_map_dir_weight=cfg.loss_map_dir_weight,
            )
            if cfg.use_map_head
            else None
        )

        # set by the loss when a GradBalancer is active
        self.aux_grad_scale: Dict[str, float] = {"det": 1.0, "map": 1.0}

    # ------------------------------------------------------------------ #
    def _apply_backbone_train_policy(self) -> None:
        """Apply the original mmdet ResNet freeze and BatchNorm policy."""
        frozen_stages = self._backbone_frozen_stages
        if not -1 <= frozen_stages < len(self._BACKBONE_STAGE_MODULES):
            raise ValueError(
                f"frozen_stages must be in [-1, {len(self._BACKBONE_STAGE_MODULES) - 1}], "
                f"got {frozen_stages}"
            )

        for stage_modules in self._BACKBONE_STAGE_MODULES[: frozen_stages + 1]:
            for module_name in stage_modules:
                module = getattr(self.image_encoder, module_name, None)
                if module is None:
                    raise ValueError(
                        f"image encoder {type(self.image_encoder).__name__} does not expose "
                        f"the ResNet stage module {module_name!r} required by frozen_stages="
                        f"{frozen_stages}"
                    )
                module.eval()
                module.requires_grad_(False)

        for module in self.image_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if self._backbone_norm_eval:
                    module.eval()
                if not self._backbone_norm_requires_grad:
                    module.requires_grad_(False)

    def train(self, mode: bool = True) -> "ParaSSRModel":
        super().train(mode)
        self._apply_backbone_train_policy()
        return self

    # ------------------------------------------------------------------ #
    def extract_img_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """``[B, N, 3, H, W]`` -> ``[[B, N, C, h, w]]``."""
        B, N = img.shape[:2]
        img = img.flatten(0, 1)
        if self.grid_mask is not None:
            img = self.grid_mask(img)
        feat = self.image_encoder(img)[-1]
        feat = self.img_neck(feat)
        _, C, h, w = feat.shape
        return [feat.view(B, N, C, h, w)]

    @torch.no_grad()
    def obtain_history_bev(self, features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Run the encoder over the history frames without building a graph."""
        cams = features["camera_feature"]  # [B, T, N, 3, H, W]
        T = cams.shape[1]
        if T <= 1:
            return None

        was_training = self.training
        self.eval()
        prev_bev = None
        for t in range(T - 1):
            feats = self.extract_img_feat(cams[:, t])
            prev_bev = self.pts_bbox_head(
                feats,
                lidar2img=features["lidar2img"],
                image_hw=features["image_hw"],
                ego_motion=features["ego_motion"][:, t],
                bev_shift=features["bev_shift"][:, t],
                prev_bev=prev_bev,
                only_bev=True,
            )
        if was_training:
            self.train()
        return prev_bev

    # ------------------------------------------------------------------ #
    def forward(
        self, features: Dict[str, torch.Tensor], run_aux: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        cfg = self._config
        if run_aux is None:
            run_aux = self.training or cfg.test_aux_heads

        prev_bev = self.obtain_history_bev(features)

        cams = features["camera_feature"]
        cur_feats = self.extract_img_feat(cams[:, -1])
        outs = self.pts_bbox_head(
            cur_feats,
            lidar2img=features["lidar2img"],
            image_hw=features["image_hw"],
            ego_motion=features["ego_motion"][:, -1],
            bev_shift=features["bev_shift"][:, -1],
            prev_bev=prev_bev,
            cmd=features["command"],
        )

        bev_embed = outs["bev_embed"]
        predictions: Dict[str, torch.Tensor] = {
            "bev_embed": bev_embed,
            "token_attn": outs["token_attn"],
        }
        if cfg.use_metric_planner:
            from .modules.candidate_planner import (
                commanded_candidates, offsets_to_poses, rank_candidates,
            )
            candidates = offsets_to_poses(commanded_candidates(
                outs["candidate_offsets"], features["command"]
            ))
            metric_logits = self.metric_head(
                bev_embed, outs["bev_pos"], candidates, features["status_feature"]
            )
            trajectory, selected, ranks = rank_candidates(
                candidates, outs["candidate_logits"], metric_logits,
                cfg.candidate_score_weight, cfg.metric_score_weight,
            )
            predictions.update({
                "candidate_offsets": outs["candidate_offsets"],
                "candidate_logits": outs["candidate_logits"],
                "plan_anchors": outs["plan_anchors"],
                "trajectory_candidates": candidates,
                "metric_logits": metric_logits,
                "candidate_rank_scores": ranks,
                "selected_candidate": selected,
                "trajectory": trajectory,
            })
        else:
            predictions.update({
                "ego_fut_preds": outs["ego_fut_preds"],
                "trajectory": self.pts_bbox_head.select_trajectory(
                    outs["ego_fut_preds"], features["command"]
                ),
            })

        if run_aux and self.det_motion_head is not None:
            s = self.aux_grad_scale.get("det", 1.0)
            bev_det = bev_embed if s == 1.0 else _ScaleGrad.apply(bev_embed, s)
            predictions.update(self.det_motion_head(bev_det))
        if run_aux and self.map_head is not None:
            s = self.aux_grad_scale.get("map", 1.0)
            bev_map = bev_embed if s == 1.0 else _ScaleGrad.apply(bev_embed, s)
            predictions.update(self.map_head(bev_map))

        return predictions
