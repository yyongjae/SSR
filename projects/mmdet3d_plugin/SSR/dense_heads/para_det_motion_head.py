"""Parallel detection + motion-forecasting head for PARA-SSR.

PARA-Drive (CVPR'24, §4) treats tracking and motion prediction as a *single*
module -- "the motion prediction module also uses bounding boxes of tracked
objects and their latent features as inputs, and therefore we consider
tracking-prediction as a whole module ... and have omitted a separate tracking
head".  Its queries "employ self-attention between query features to facilitate
interaction between agents, in addition to applying cross-attention with the BEV
features".

Ported from VAD's ``VADHead`` (detection branch + motion branch + their losses)
with two deliberate departures that follow the paper:

* **No agent<-map cross-attention.**  VAD feeds map-head outputs into motion
  prediction; that is exactly edge (1) of PARA-Drive Fig. 4, which the paper
  removes because "UniAD uses the lane query features from the 1st-stage mapping
  head, which are noisy, therefore removing interaction between lane and motion
  queries improves the performance" (§3.3).  Dropping it also makes this head
  genuinely parallel to the map head.  Consequently ``traj_branch`` consumes
  ``embed_dims`` instead of VAD's ``embed_dims * 2``.
* **Nothing flows to the planner.**  The head only returns predictions and
  losses; the planner sees the BEV feature alone.
"""
import copy

import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


@HEADS.register_module()
class ParaDetMotionHead(BaseModule):
    """Agent queries -> BEV cross-attention -> boxes + multimodal trajectories.

    Args:
        num_query (int): number of agent queries.
        num_classes (int): detection classes (10 for nuScenes).
        embed_dims (int): query/BEV width.
        bev_h, bev_w (int): BEV grid size.
        fut_ts (int): motion horizon in 0.5 s steps.
        fut_mode (int): number of trajectory modes per agent.
        valid_fut_ts (int): steps actually supervised.
        transformer_decoder (dict): ``DetectionTransformerDecoder`` config
            (cross-attends the BEV feature).
        motion_decoder (dict): ``CustomTransformerDecoder`` config used for
            agent<->agent self-attention.
    """

    def __init__(self,
                 num_query=300,
                 num_classes=10,
                 embed_dims=256,
                 bev_h=100,
                 bev_w=100,
                 pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
                 code_size=10,
                 code_weights=None,
                 num_reg_fcs=2,
                 fut_ts=6,
                 fut_mode=6,
                 valid_fut_ts=6,
                 transformer_decoder=None,
                 motion_decoder=None,
                 use_pe=True,
                 motion_det_score=None,
                 sync_cls_avg_factor=True,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj=dict(type='L1Loss', loss_weight=0.2),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.2),
                 loss_weight=1.0,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = list(pc_range)
        self.code_size = code_size
        self.num_reg_fcs = num_reg_fcs
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = valid_fut_ts
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_weight = loss_weight
        self.fp16_enabled = False

        if code_weights is None:
            code_weights = [1.0] * 8 + [0.2, 0.2]
        self.register_buffer('code_weights',
                             torch.tensor(code_weights, dtype=torch.float32),
                             persistent=False)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_out_channels = num_classes if self.use_sigmoid_cls \
            else num_classes + 1
        self.bg_cls_weight = 0
        self.traj_bg_cls_weight = 0
        self.traj_num_cls = 1

        if train_cfg is not None:
            assert 'assigner' in train_cfg, \
                'assigner should be provided when train_cfg is set'
            self.assigner = build_assigner(train_cfg['assigner'])
            self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)
        else:
            self.assigner = None
            self.sampler = None
        self.train_cfg = train_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder) if bbox_coder else None

        self.transformer_decoder_cfg = transformer_decoder
        self.motion_decoder_cfg = motion_decoder
        self._init_layers()

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: embed_dims (not 2 * embed_dims as in VAD) -- no map cross-attn.
        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims, self.embed_dims))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims, self.fut_ts * 2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        assert self.transformer_decoder_cfg is not None, \
            'ParaDetMotionHead requires a transformer_decoder config'
        self.decoder = build_transformer_layer_sequence(
            self.transformer_decoder_cfg)
        num_pred = self.decoder.num_layers

        self.cls_branches = _get_clones(cls_branch, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.traj_branches = _get_clones(traj_branch, 1)
        self.traj_cls_branches = _get_clones(traj_cls_branch, 1)

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        assert self.motion_decoder_cfg is not None, \
            'ParaDetMotionHead requires a motion_decoder config'
        self.motion_decoder = build_transformer_layer_sequence(
            self.motion_decoder_cfg)
        self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)
        if self.use_pe:
            self.pos_mlp_sa = nn.Linear(2, self.embed_dims)

    def init_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.motion_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        if self.use_sigmoid_cls:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, bev_embed, img_metas=None):
        """
        Args:
            bev_embed (Tensor): ``[B, bev_h * bev_w, C]`` shared BEV feature.
        Returns:
            dict with ``all_cls_scores`` ``[L, B, Q, num_classes]``,
            ``all_bbox_preds`` ``[L, B, Q, code_size]``,
            ``all_traj_preds`` ``[1, B, Q, fut_mode, fut_ts * 2]``,
            ``all_traj_cls_scores`` ``[1, B, Q, fut_mode]``.
        """
        bs = bev_embed.size(0)
        dtype = bev_embed.dtype

        query_pos, query = torch.split(
            self.query_embedding.weight.to(dtype), self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference = reference_points

        # decoder expects sequence-first tensors
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        value = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            cls_branches=None,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]],
                                        device=query.device),
            level_start_index=torch.tensor([0], device=query.device))

        hs = inter_states.permute(0, 2, 1, 3)  # [L, B, Q, D]

        outputs_classes, outputs_coords, outputs_coords_bev = [], [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] *
                             (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] *
                             (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] *
                             (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        # ---- motion: agent<->agent self-attention only (no map interaction) ----
        num_agent = outputs_coords_bev[-1].shape[1]
        motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
        mode_query = self.motion_mode_query.weight  # [fut_mode, D]
        motion_query = (motion_query[:, None, :, :] +
                        mode_query[None, :, None, :]).flatten(0, 1)  # [A*M, B, D]

        if self.use_pe:
            motion_pos = self.pos_mlp_sa(outputs_coords_bev[-1])  # [B, A, D]
            motion_pos = motion_pos.unsqueeze(2).repeat(
                1, 1, self.fut_mode, 1).flatten(1, 2).permute(1, 0, 2)
        else:
            motion_pos = None

        if self.motion_det_score is not None:
            max_motion_score = outputs_classes[-1].max(dim=-1)[0]
            invalid_motion_idx = (max_motion_score < self.motion_det_score)
            invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(
                1, 1, self.fut_mode).flatten(1, 2)
        else:
            invalid_motion_idx = None

        motion_hs = self.motion_decoder(
            query=motion_query,
            key=motion_query,
            value=motion_query,
            query_pos=motion_pos,
            key_pos=motion_pos,
            key_padding_mask=invalid_motion_idx)

        motion_hs = motion_hs.permute(1, 0, 2).unflatten(
            dim=1, sizes=(num_agent, self.fut_mode))  # [B, A, M, D]

        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs).squeeze(-1)

        return dict(
            all_cls_scores=torch.stack(outputs_classes),
            all_bbox_preds=torch.stack(outputs_coords),
            all_traj_preds=outputs_traj.unsqueeze(0),
            all_traj_cls_scores=outputs_traj_class.unsqueeze(0))

    # ------------------------------------------------------------------ #
    # targets                                                            #
    # ------------------------------------------------------------------ #
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts * 2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts * 2:self.fut_ts * 3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        traj_targets = torch.zeros((num_bboxes, gt_traj_c),
                                   dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        traj_masks = torch.zeros_like(traj_targets)
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(
            num_gt_bbox, -1)
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                  dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        traj_weights = traj_weights * fut_ts_mask.view(num_bboxes, -1)

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, traj_targets,
                traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_attr_labels_list,
             gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, traj_targets_list, traj_weights_list,
                gt_fut_masks_list, num_total_pos, num_total_neg)

    def get_best_fut_preds(self, traj_preds, traj_targets, gt_fut_masks):
        """Winner-take-all mode selection by final displacement error."""
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)
        dist = torch.linalg.norm(
            cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        return traj_preds[box_idxs, min_mode_idxs, :, :].reshape(
            -1, self.fut_ts * 2)

    def get_traj_cls_target(self, traj_preds, traj_targets, gt_fut_masks,
                            neg_inds):
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)
        dist = torch.linalg.norm(
            cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode
        return traj_labels

    # ------------------------------------------------------------------ #
    # losses                                                             #
    # ------------------------------------------------------------------ #
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None,
                    with_traj=True):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights,
                                 avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # VAD computes the motion losses for every decoder layer's matching and
        # then keeps only the last layer's values; skipping the discarded
        # layers is numerically identical and cheaper.
        if not with_traj:
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_cls = torch.nan_to_num(loss_cls)
                loss_bbox = torch.nan_to_num(loss_bbox)
            return loss_cls, loss_bbox, None, None

        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)
        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))
        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights,
            avg_factor=traj_cls_avg_factor)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_bboxes_list,
             gt_labels_list,
             gt_attr_labels,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports gt_bboxes_ignore=None'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((g.gravity_center, g.tensor[:, 3:]), dim=1).to(device)
            for g in gt_bboxes_list
        ]

        w = self.loss_weight
        loss_dict = dict()
        for lvl in range(num_dec_layers):
            last = lvl == num_dec_layers - 1
            loss_cls, loss_bbox, loss_traj, loss_traj_cls = self.loss_single(
                all_cls_scores[lvl], all_bbox_preds[lvl], all_traj_preds[0],
                all_traj_cls_scores[0], gt_bboxes_list, gt_labels_list,
                gt_attr_labels, None, with_traj=last)
            if last:
                loss_dict['loss_cls'] = w * loss_cls
                loss_dict['loss_bbox'] = w * loss_bbox
                loss_dict['loss_traj'] = w * loss_traj
                loss_dict['loss_traj_cls'] = w * loss_traj_cls
            else:
                loss_dict[f'd{lvl}.loss_cls'] = w * loss_cls
                loss_dict[f'd{lvl}.loss_bbox'] = w * loss_bbox
        return loss_dict

    def forward_train(self, bev_embed, img_metas, gt_bboxes_3d, gt_labels_3d,
                      gt_attr_labels, **kwargs):
        preds = self(bev_embed, img_metas)
        return self.loss(preds, gt_bboxes_3d, gt_labels_3d, gt_attr_labels)

    def forward_test(self, bev_embed, img_metas, rescale=False):
        preds = self(bev_embed, img_metas)
        if self.bbox_coder is None:
            return preds
        return self.get_bboxes(preds, img_metas, rescale=rescale)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        ret_list = []
        for i, preds in enumerate(preds_dicts):
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            ret_list.append([bboxes, preds['scores'], preds['labels'],
                             preds['trajs']])
        return ret_list
