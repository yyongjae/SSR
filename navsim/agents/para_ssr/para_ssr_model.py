"""PARA-SSR: shared BEV, independent perception heads, task-memory planner.

* history frames build ``prev_bev`` under ``no_grad`` (they exist to give the
  temporal self-attention something to attend to, not to be trained through);
* the current frame's BEV is produced by the planning head, which owns the
  encoder;
* with task interaction, perception latents feed the planner; without it,
  heads train independently from shared BEV and may be absent for plan-only;
* shared-BEV gradient balancing happens by loss origin in ``ParaSSRLoss``.
  Scaling decoder inputs would incorrectly scale planning gradients that
  travel through those decoders;
* ego velocity/acceleration condition the planning query only; temporal BEV
  alignment warps history with the geometric ``bev_shift`` and relative yaw;
* with ``use_lidar`` (SafeDrive) each frame's point cloud is encoded into a
  LiDAR BEV first, which seeds the BEV queries and feeds the per-layer LiDAR
  cross-attention -- for history frames as well, so ``prev_bev`` is the same
  kind of feature as the current BEV it is aligned with.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bevformer import BEVFormerEncoder, SSRPerceptionTransformer
from .modules.det_motion_head import ParaDetMotionHead
from .modules.lidar_encoder import build_lidar_encoder
from .modules.map_head import ParaMapHead
from .modules.planner_head import ParaSSRPlannerHead


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
        if cfg.use_stl or cfg.plan_num_layers != 3:
            raise ValueError("PARA-SSR requires use_stl=False and plan_num_layers=3")
        if cfg.use_ego_motion:
            raise ValueError(
                "PARA-SSR routes ego status to planning; use_ego_motion must be False "
                "(geometric BEV translation/rotation alignment uses use_shift)"
            )
        self.use_task_interaction = bool(cfg.use_task_interaction)
        if self.use_task_interaction and (not cfg.use_det_motion_head or not cfg.use_map_head):
            raise ValueError("the planner requires both det/motion and map heads")
        if tuple(cfg.map_pc_range) != tuple(cfg.pc_range):
            raise ValueError("shared BEV/detection/map must use one physical ROI")
        active_tasks = ["plan"]
        if cfg.use_det_motion_head:
            active_tasks.extend(("det", "motion"))
        if cfg.use_map_head:
            active_tasks.append("map")
        if any(cfg.task_loss_weight.get(task, 1.0) <= 0 for task in active_tasks):
            raise ValueError("active task supervision must all be enabled; remove an unused head instead")
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
            use_lidar=cfg.use_lidar,
            num_points_lidar=cfg.lidar_attn_points,
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
            use_lidar=cfg.use_lidar,
            use_stl=cfg.use_stl,
            plan_num_layers=cfg.plan_num_layers,
            use_task_interaction=self.use_task_interaction,
        )
        self.lidar_encoder = build_lidar_encoder(cfg) if cfg.use_lidar else None

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

    def lidar_bev(self, features: Dict[str, torch.Tensor], t: int) -> Optional[torch.Tensor]:
        """LiDAR BEV ``[B, C, bev_h, bev_w]`` for queue step ``t``, or ``None``."""
        if self.lidar_encoder is None:
            return None
        if "lidar_points" not in features or "lidar_num_points" not in features:
            raise ValueError(
                "a use_lidar model needs lidar_points/lidar_num_points features; "
                "rebuild the feature cache with use_lidar enabled"
            )
        return self.lidar_encoder(
            features["lidar_points"][:, t], features["lidar_num_points"][:, t]
        )

    @torch.no_grad()
    def obtain_history_bev(self, features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Run the encoder over the history frames without building a graph."""
        cams = features["camera_feature"]  # [B, T, N, 3, H, W]
        T = cams.shape[1]
        if T <= 1:
            return None

        was_training = self.training
        was_cache_enabled = torch.is_autocast_cache_enabled()
        try:
            # History and the trainable current frame share weights and the
            # caller's autocast context. Caching a weight cast under no_grad
            # would let the current frame reuse a detached copy and silently
            # lose encoder gradients. Disable only caching for history;
            # preserve the caller's autocast enabled state and dtype.
            torch.set_autocast_cache_enabled(False)
            self.eval()
            if was_training and self.lidar_encoder is not None:
                # The image backbone uses frozen BN, whereas the from-scratch
                # LiDAR BN must use the same statistics mode for history and
                # current frames (as in SafeDrive).
                self.lidar_encoder.train()
            prev_bev = None
            for t in range(T - 1):
                feats = self.extract_img_feat(cams[:, t])
                prev_bev = self.pts_bbox_head(
                    feats,
                    lidar2img=features["lidar2img"],
                    image_hw=features["image_hw"],
                    ego_motion=None,
                    bev_shift=features["bev_shift"][:, t],
                    bev_yaw=features["ego_motion"][:, t, 2].detach(),
                    prev_bev=prev_bev,
                    only_bev=True,
                    lidar_bev=self.lidar_bev(features, t),
                )
            return prev_bev
        finally:
            torch.set_autocast_cache_enabled(was_cache_enabled)
            self.train(was_training)

    # ------------------------------------------------------------------ #
    def forward(
        self, features: Dict[str, torch.Tensor], run_aux: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        # Interaction requires perception at inference too. In parallel mode,
        # run_aux only requests optional eval perception; training runs all heads.
        cfg = self._config
        if run_aux is None:
            run_aux = self.training or cfg.test_aux_heads

        prev_bev = self.obtain_history_bev(features)

        cams = features["camera_feature"]
        cur_feats = self.extract_img_feat(cams[:, -1])
        bev_embed = self.pts_bbox_head(
            cur_feats,
            lidar2img=features["lidar2img"],
            image_hw=features["image_hw"],
            ego_motion=None,
            bev_shift=features["bev_shift"][:, -1],
            # Read only relative yaw from the existing cache vector. This is
            # geometry for history alignment, not learned status conditioning.
            bev_yaw=features["ego_motion"][:, -1, 2].detach(),
            prev_bev=prev_bev,
            only_bev=True,
            lidar_bev=self.lidar_bev(features, -1),
        )

        run_heads = self.use_task_interaction or self.training or run_aux
        det_out = (
            self.det_motion_head(bev_embed, return_hidden=self.use_task_interaction)
            if run_heads and self.det_motion_head is not None else None
        )
        map_out = (
            self.map_head(bev_embed, return_hidden=self.use_task_interaction)
            if run_heads and self.map_head is not None else None
        )
        outs = self.pts_bbox_head.plan_from_bev(
            bev_embed, features["command"], det_out, map_out,
            # The cached status is command + (vx, vy, ax, ay) in NAVSIM's
            # native ego axes and metric units. Command has its own embedding.
            ego_status=features["status_feature"][:, cfg.num_navi_cmd:],
        )
        predictions: Dict[str, torch.Tensor] = {
            "bev_embed": bev_embed,
            "ego_fut_preds": outs["ego_fut_preds"],
            "trajectory": self.pts_bbox_head.select_trajectory(
                outs["ego_fut_preds"], features["command"]
            ),
        }
        # Training always exposes predictions needed by every supervised loss,
        # even when a caller explicitly passes run_aux=False.
        if self.training or run_aux:
            if det_out is not None:
                predictions.update(det_out)
            if map_out is not None:
                predictions.update(map_out)

        return predictions
