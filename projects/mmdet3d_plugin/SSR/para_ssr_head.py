"""PARA-SSR planning head: SSR's navigation-guided sparse planner, FFP removed.

This is ``SSRHead`` reduced to the path that actually runs at inference:

    BEV encoder -> navigation SE gating -> TokenLearner (16 scene tokens)
    -> latent self-attention decoder -> waypoint cross-attention decoder -> MLP

Everything else is gone:

* the FFP / latent-world-model plumbing (``action_mln``, ``pos_mln``, the
  ``act_query`` output and the ``MLN`` class), and
* the never-called VAD detection/map/motion branches that ``SSRHead`` still
  allocated (``cls_branches``, ``reg_branches``, ``traj_branches``,
  ``map_*_branches``, ``query_embedding``, ``map_*_embedding``, ``ego_query``).
  Those tasks now live in real, separately-built parallel heads.

The forward pass of the retained path is numerically identical to ``SSRHead`` so
planning performance is directly comparable against the original checkpointed
baseline.

Note this head also owns the BEV encoder (``transformer``), keeping SSR's
``only_bev=True`` shortcut used by ``obtain_history_bev``.  It exposes the
resulting ``bev_embed`` so the detector can fan it out to the parallel heads.
"""
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer

from .tokenlearner import TokenLearnerV11


class SELayer(nn.Module):
    """Channel gating of the BEV feature by the navigation-command embedding."""

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


@HEADS.register_module()
class ParaSSRHead(BaseModule):
    """BEV encoder + navigation-guided sparse-token planner.

    Args:
        bev_h, bev_w (int): BEV grid size.
        embed_dims (int): feature width.
        num_scenes (int): number of sparse scene tokens produced by the
            TokenLearner (SSR's bottleneck; 16 in the paper).
        fut_ts (int): planning horizon in 0.5 s steps.
        ego_fut_mode (int): one trajectory branch per navigation command.
        transformer (dict): ``SSRPerceptionTransformer`` config (BEV encoder).
        latent_decoder (dict): self-attention decoder over the scene tokens.
        way_decoder (dict): cross-attention decoder, waypoint queries -> tokens.
    """

    def __init__(self,
                 transformer=None,
                 positional_encoding=dict(
                     type='LearnedPositionalEncoding',
                     num_feats=128,
                     row_num_embed=100,
                     col_num_embed=100),
                 bev_h=100,
                 bev_w=100,
                 embed_dims=256,
                 in_channels=256,
                 pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
                 num_scenes=16,
                 num_reg_fcs=2,
                 fut_ts=6,
                 ego_fut_mode=3,
                 num_navi_cmd=3,
                 latent_decoder=None,
                 way_decoder=None,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.pc_range = list(pc_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_scenes = num_scenes
        self.num_reg_fcs = num_reg_fcs
        self.fut_ts = fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_navi_cmd = num_navi_cmd
        self.fp16_enabled = False

        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. Found ' \
            f'{self.embed_dims} and {num_feats}.'

        self.latent_decoder_cfg = latent_decoder
        self.way_decoder_cfg = way_decoder
        self._init_layers()

        self.loss_plan_reg = build_loss(loss_plan_reg)

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w,
                                          self.embed_dims)

        # navigation command -> channel gate on every BEV cell
        self.navi_embedding = nn.Embedding(self.num_navi_cmd, self.embed_dims)
        self.navi_se = SELayer(self.embed_dims)

        # sparse scene representation
        self.tokenlearner = TokenLearnerV11(self.num_scenes, self.embed_dims * 2)
        self.latent_decoder = build_transformer_layer_sequence(
            self.latent_decoder_cfg)

        # planning: one query per (command, timestep)
        self.way_point = nn.Embedding(self.ego_fut_mode * self.fut_ts,
                                      self.embed_dims * 2)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder_cfg)

        ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(self.embed_dims, self.embed_dims))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(self.embed_dims, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

    def init_weights(self):
        self.transformer.init_weights()
        for decoder in (self.latent_decoder, self.way_decoder):
            if decoder is not None:
                for p in decoder.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                cmd=None,
                **kwargs):
        """
        Args:
            mlvl_feats (list[Tensor]): image features, each ``[B, N, C, H, W]``.
            prev_bev (Tensor | None): previous BEV for temporal self-attention.
            only_bev (bool): return right after the BEV encoder (used by
                ``obtain_history_bev``).
            cmd (Tensor): one-hot navigation command, ``[B, ?, 1, 3]``.
        Returns:
            dict with ``bev_embed`` ``[B, bev_h * bev_w, C]``,
            ``scene_query`` ``[num_scenes, B, C]`` and
            ``ego_fut_preds`` ``[B, ego_fut_mode, fut_ts, 2]``.
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev)
        if only_bev:
            return bev_embed

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        cmd = cmd[0, 0, 0]
        cmd_idx = torch.nonzero(cmd)[0, 0]

        navi_embed = self.navi_embedding.weight[cmd_idx][None, None]
        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)
        learned_latent_query, selected = self.tokenlearner(bev_query)

        learned_latent_query = learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
            query=latent_query,
            key=latent_query,
            value=latent_query,
            query_pos=latent_pos,
            key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(way_point, self.embed_dims, dim=1)
        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)

        way_point = self.way_decoder(
            query=way_point,
            key=latent_query,
            value=latent_query,
            query_pos=wp_pos,
            key_pos=latent_pos)

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2).view(
            bs, self.ego_fut_mode, self.fut_ts, 2)

        return {
            'bev_embed': bev_embed,
            'scene_query': latent_query,
            'token_attn': selected,
            'ego_fut_preds': outputs_ego_trajs,
        }

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, preds_dicts, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
             **kwargs):
        """L1 on the commanded branch only -- identical to ``SSRHead.loss``."""
        ego_fut_preds = preds_dicts['ego_fut_preds']

        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        weight = weight.repeat(1, 1, 1, 2)

        return dict(
            loss_plan_reg=self.loss_plan_reg(ego_fut_preds, ego_fut_gt, weight))
