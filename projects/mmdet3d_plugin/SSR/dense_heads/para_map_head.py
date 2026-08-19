"""Parallel online-mapping head for PARA-SSR.

PARA-Drive's mapping module is one of the four modules co-training the shared
BEV feature (CVPR'24, Fig. 5).  The paper itself uses a dense 4-channel semantic
BEV map (Panoptic SegFormer); this port instead reuses VAD's *vectorised* map
representation, which the paper explicitly lists as a legitimate design variant
of the same module (Fig. 1b, §2: "UniAD and OccNet treat online mapping as a
dense prediction task ... whereas VAD opts for vectorized map representations").
The choice is orthogonal to the parallel-vs-sequential axis that PARA-Drive is
actually about, and SSR already ships every piece needed
(``MapDetectionTransformerDecoder``, ``MapHungarianAssigner3D``, ``PtsL1Loss``,
``PtsDirCosLoss``, and ``VectorizedLocalMap`` GT produced for every sample).

Unlike VAD, nothing from this head is fed to motion prediction or to the
planner -- that is edge (1)/(5) in PARA-Drive Fig. 4, both of which the paper
removes.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder

from projects.mmdet3d_plugin.SSR.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.SSR.modules.spatial_cross_attention import (
    MSDeformableAttention3D)
from projects.mmdet3d_plugin.SSR.modules.temporal_self_attention import (
    TemporalSelfAttention)
from projects.mmdet3d_plugin.SSR.utils.map_utils import (
    denormalize_2d_bbox, denormalize_2d_pts, normalize_2d_bbox, normalize_2d_pts)


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


@HEADS.register_module()
class ParaMapHead(BaseModule):
    """Map instance/point queries -> BEV cross-attention -> vectorised map."""

    def __init__(self,
                 map_num_vec=100,
                 map_num_pts_per_vec=20,
                 map_num_pts_per_gt_vec=20,
                 map_num_classes=3,
                 embed_dims=256,
                 bev_h=100,
                 bev_w=100,
                 pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
                 map_code_size=2,
                 map_code_weights=None,
                 map_dir_interval=1,
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v2',
                 num_reg_fcs=2,
                 transformer_decoder=None,
                 sync_cls_avg_factor=True,
                 bbox_coder=None,
                 loss_map_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
                 loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
                 loss_weight=1.0,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_classes = map_num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = list(pc_range)
        self.map_code_size = map_code_size
        self.map_dir_interval = map_dir_interval
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        self.num_reg_fcs = num_reg_fcs
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_weight = loss_weight
        self.fp16_enabled = False

        if map_code_weights is None:
            map_code_weights = [1.0, 1.0, 1.0, 1.0]
        self.register_buffer('map_code_weights',
                             torch.tensor(map_code_weights, dtype=torch.float32),
                             persistent=False)

        self.map_use_sigmoid_cls = loss_map_cls.get('use_sigmoid', False)
        self.map_cls_out_channels = map_num_classes if self.map_use_sigmoid_cls \
            else map_num_classes + 1
        self.map_bg_cls_weight = 0

        if train_cfg is not None:
            assert 'map_assigner' in train_cfg, \
                'map_assigner should be provided when train_cfg is set'
            map_assigner = train_cfg['map_assigner']
            assert loss_map_cls['loss_weight'] == \
                map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be exactly the same.'
            assert loss_map_pts['loss_weight'] == \
                map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be the same.'
            self.map_assigner = build_assigner(map_assigner)
            self.map_sampler = build_sampler(
                dict(type='PseudoSampler'), context=self)
        else:
            self.map_assigner = None
            self.map_sampler = None
        self.train_cfg = train_cfg

        self.map_bbox_coder = build_bbox_coder(bbox_coder) if bbox_coder else None

        self.transformer_decoder_cfg = transformer_decoder
        self._init_layers()

        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)

    def _init_layers(self):
        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        assert self.transformer_decoder_cfg is not None, \
            'ParaMapHead requires a transformer_decoder config'
        self.map_decoder = build_transformer_layer_sequence(
            self.transformer_decoder_cfg)
        map_num_pred = self.map_decoder.num_layers

        self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
        self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)

        # instance_pts factorisation: one embedding per polyline, one per point
        self.map_instance_embedding = nn.Embedding(self.map_num_vec,
                                                   self.embed_dims * 2)
        self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec,
                                              self.embed_dims * 2)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        for p in self.map_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Xavier above overwrites the deformable-attention weights, which must
        # start at zero so that step 0 samples exactly the radial grid held in
        # ``sampling_offsets.bias`` with uniform attention. Re-apply their own
        # initialisation afterwards, exactly as VAD's transformer and
        # ``SSRPerceptionTransformer.init_weights`` do. (The biases survive the
        # loop because they are 1-D, but the weights do not.)
        for m in self.modules():
            if isinstance(m, (MSDeformableAttention3D, TemporalSelfAttention,
                              CustomMSDeformableAttention)):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        if self.map_use_sigmoid_cls:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def map_transform_box(self, pts, y_first=False):
        """Points set -> (cx, cy, w, h) bounding box."""
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                               self.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def forward(self, bev_embed, img_metas=None):
        """
        Args:
            bev_embed (Tensor): ``[B, bev_h * bev_w, C]`` shared BEV feature.
        Returns:
            dict with ``map_all_cls_scores`` ``[L, B, P, num_map_classes]``,
            ``map_all_bbox_preds`` ``[L, B, P, 4]``,
            ``map_all_pts_preds`` ``[L, B, P, npts, 2]``.
        """
        bs = bev_embed.size(0)
        dtype = bev_embed.dtype

        pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
        map_query_embed = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        map_query_pos, map_query = torch.split(
            map_query_embed, self.embed_dims, dim=1)
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        map_reference_points = self.map_reference_points(map_query_pos).sigmoid()
        map_init_reference = map_reference_points

        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        value = bev_embed.permute(1, 0, 2)

        map_inter_states, map_inter_references = self.map_decoder(
            query=map_query,
            key=None,
            value=value,
            query_pos=map_query_pos,
            reference_points=map_reference_points,
            reg_branches=self.map_reg_branches,
            cls_branches=None,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]],
                                        device=map_query.device),
            level_start_index=torch.tensor([0], device=map_query.device))

        map_hs = map_inter_states.permute(0, 2, 1, 3)  # [L, B, Q, D]

        map_outputs_classes, map_outputs_coords, map_outputs_pts_coords = [], [], []
        for lvl in range(map_hs.shape[0]):
            reference = map_init_reference if lvl == 0 \
                else map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs, self.map_num_vec,
                                 self.map_num_pts_per_vec, -1).mean(2))
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        return dict(
            map_all_cls_scores=torch.stack(map_outputs_classes),
            map_all_bbox_preds=torch.stack(map_outputs_coords),
            map_all_pts_preds=torch.stack(map_outputs_pts_coords))

    # ------------------------------------------------------------------ #
    # targets                                                            #
    # ------------------------------------------------------------------ #
    def _map_get_target_single(self,
                               cls_score,
                               bbox_pred,
                               pts_pred,
                               gt_labels,
                               gt_bboxes,
                               gt_shifts_pts,
                               gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(
            bbox_pred, cls_score, pts_pred, gt_bboxes, gt_labels, gt_shifts_pts,
            gt_bboxes_ignore)
        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                                  gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full((num_bboxes, ), self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds,
                                         sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[
            sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights, pts_targets,
                pts_weights, pos_inds, neg_inds)

    def map_get_targets(self,
                        cls_scores_list,
                        bbox_preds_list,
                        pts_preds_list,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self._map_get_target_single, cls_scores_list, bbox_preds_list,
             pts_preds_list, gt_labels_list, gt_bboxes_list, gt_shifts_pts_list,
             gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    # ------------------------------------------------------------------ #
    # losses                                                             #
    # ------------------------------------------------------------------ #
    def map_loss_single(self,
                        cls_scores,
                        bbox_preds,
                        pts_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list, gt_bboxes_list,
            gt_labels_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(cls_scores, labels, label_weights,
                                     avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights
        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(
                pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = \
            denormed_pts_preds[:, self.map_dir_interval:, :] - \
            denormed_pts_preds[:, :-self.map_dir_interval, :]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:, :] - \
            pts_targets[:, :-self.map_dir_interval, :]
        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             map_gt_bboxes_list,
             map_gt_labels_list,
             map_gt_bboxes_ignore=None,
             gt_shifts=None):
        assert map_gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports map_gt_bboxes_ignore=None'

        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        map_gt_bboxes_list = [g.bbox.to(device) for g in map_gt_vecs_list]
        # Reading this attribute draws from the global NumPy RNG for closed
        # polylines -- see materialise_shifts. forward_train does it once and
        # passes the result in; the fallback keeps a direct loss() call working.
        map_gt_shifts_pts_list = (
            gt_shifts if gt_shifts is not None
            else self.materialise_shifts(map_gt_vecs_list, device))

        all_gt_bboxes = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels = [map_gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_ignore = [None for _ in range(num_dec_layers)]

        (losses_cls, losses_bbox, losses_iou, losses_pts,
         losses_dir) = multi_apply(
             self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
             map_all_pts_preds, all_gt_bboxes, all_gt_labels, all_gt_shifts_pts,
             all_ignore)

        w = self.loss_weight
        loss_dict = dict()
        loss_dict['loss_map_cls'] = w * losses_cls[-1]
        loss_dict['loss_map_bbox'] = w * losses_bbox[-1]
        loss_dict['loss_map_iou'] = w * losses_iou[-1]
        loss_dict['loss_map_pts'] = w * losses_pts[-1]
        loss_dict['loss_map_dir'] = w * losses_dir[-1]
        for i in range(num_dec_layers - 1):
            loss_dict[f'd{i}.loss_map_cls'] = w * losses_cls[i]
            loss_dict[f'd{i}.loss_map_bbox'] = w * losses_bbox[i]
            loss_dict[f'd{i}.loss_map_iou'] = w * losses_iou[i]
            loss_dict[f'd{i}.loss_map_pts'] = w * losses_pts[i]
            loss_dict[f'd{i}.loss_map_dir'] = w * losses_dir[i]
        return loss_dict

    _SHIFT_ATTR = {
        'v0': 'shift_fixed_num_sampled_points',
        'v1': 'shift_fixed_num_sampled_points_v1',
        'v2': 'shift_fixed_num_sampled_points_v2',
        'v3': 'shift_fixed_num_sampled_points_v3',
        'v4': 'shift_fixed_num_sampled_points_v4',
    }

    def materialise_shifts(self, map_gt_bboxes_3d, device):
        """Read the shifted GT polylines ONCE per forward.

        `shift_fixed_num_sampled_points_v*` is not a pure accessor. For a
        CLOSED polyline with more vertices than `fixed_num` it enumerates every
        rotation and then picks a subset with `np.random.choice` on the GLOBAL
        NumPy RNG (nuscenes_vad_dataset.py, the `shifts_num > final_shift_num`
        branch). Two consequences, both measured:

          * two reads of the same GT give two different targets, so the metric
            and the loss were being computed against different ground truth;
          * the extra draw shifts the global NumPy stream, which GridMask also
            draws from -- so turning the metric on, or changing its interval,
            changed the augmentation of every later iteration.

        Measured on a closed 41-vertex polyline: loss_map_pts 9.26400375
        without the metric against 9.27320004 with it, and the next
        np.random.rand() 0.0636 against 0.8095.

        A metric is supposed to observe training, not participate in it.
        Materialising once and handing the same tensors to both callers removes
        both effects; saving and restoring the RNG around the metric would fix
        the GridMask side only, and still leave metric and loss scoring
        different targets.
        """
        attr = self._SHIFT_ATTR[self.map_gt_shift_pts_pattern]
        return [getattr(g, attr).to(device) for g in map_gt_bboxes_3d]

    def forward_train(self, bev_embed, img_metas, map_gt_bboxes_3d,
                      map_gt_labels_3d, metrics_out=None, **kwargs):
        preds = self(bev_embed, img_metas)
        # One materialisation, shared. See materialise_shifts.
        gt_shifts = self.materialise_shifts(
            map_gt_bboxes_3d, preds['map_all_cls_scores'].device)
        if metrics_out is not None:
            metrics_out.update(
                self.train_metrics(preds, map_gt_bboxes_3d, map_gt_labels_3d,
                                   gt_shifts=gt_shifts))
        return self.loss(preds, map_gt_bboxes_3d, map_gt_labels_3d,
                         gt_shifts=gt_shifts)

    @torch.no_grad()
    def train_metrics(self, preds, map_gt_bboxes_3d, map_gt_labels_3d,
                      gt_shifts=None):
        """Cheap quality signals for the vectorised map head.

        ``loss_map_pts`` falling is necessary but NOT sufficient: the point
        regression can tighten while the classification head stays wrong, or
        while every query collapses onto the same polyline, and chamfer mAP
        would still be ~0.

        ``map_cls_acc``   classification accuracy on the matched queries.
        ``map_conf_pos``  mean confidence on matched queries; ``map_conf_neg``
                          the same on unmatched ones. Accuracy alone is blind
                          both to a head that stays right about its few matches
                          while every score sinks, and to a flood of confident
                          background false positives.
        ``map_spread_m``  std of the predicted polyline centres across queries,
                          in metres. This is the collapse detector: if every
                          query regresses the same line it goes to zero and
                          nothing else here notices.
        ``map_pts_err_m`` mean EUCLIDEAN per-point error in metres.
        ``map_n_gt``      GT instances per image -- a pure data statistic, kept
                          only as a pipeline sanity check.

        Two warnings, both earned the hard way:

        * ``map_pts_err_m`` is an upper bound on chamfer distance, not a
          chamfer threshold test. It is index-aligned under the shift
          permutation the matcher chose, whereas chamfer takes the nearest
          point.
        * this method runs its OWN Hungarian matching below; ``loss()`` then
          runs one more per decoder layer. There is no extra network forward,
          but there is an extra matching, which is why it is gated behind
          ``aux_metric_log_interval``.

        ``map_pos_frac`` used to live here, reporting ``num_total_pos`` over
        the query count. It is gone because ``MapHungarianAssigner3D`` applies
        no cost threshold: it always returns ``min(n_query, n_gt)`` pairs, so
        that quantity is identically ``sum(n_gt) / (B * n_query)`` no matter
        what the head predicts. It was a data statistic in the costume of a
        model diagnostic. ``map_n_gt`` is the same number, named honestly.
        """
        cls_scores = preds['map_all_cls_scores'][-1]        # [B, Q, C]
        bbox_preds = preds['map_all_bbox_preds'][-1]
        pts_preds = preds['map_all_pts_preds'][-1]
        device = cls_scores.device

        gt_vecs = copy.deepcopy(map_gt_bboxes_3d)
        gt_bboxes = [g.bbox.to(device) for g in gt_vecs]
        # Must be the SAME tensors the loss scores against. Re-reading the
        # attribute would draw a different random shift for closed polylines
        # AND advance the global NumPy stream that GridMask shares.
        if gt_shifts is None:
            gt_shifts = self.materialise_shifts(gt_vecs, device)

        (labels_list, _, _, _, pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = self.map_get_targets(
            [cls_scores[i] for i in range(cls_scores.size(0))],
            [bbox_preds[i] for i in range(bbox_preds.size(0))],
            [pts_preds[i] for i in range(pts_preds.size(0))],
            gt_bboxes, map_gt_labels_3d, gt_shifts)

        labels = torch.cat(labels_list, 0)
        pos = labels < self.map_num_classes
        neg = ~pos
        flat_cls = cls_scores.reshape(-1, self.map_cls_out_channels)
        zero = flat_cls.new_zeros(())
        acc = (flat_cls.argmax(-1)[pos] == labels[pos]).float().mean() \
            if pos.any() else zero

        conf = (flat_cls.sigmoid() if self.map_use_sigmoid_cls
                else flat_cls.softmax(-1)[..., :self.map_num_classes])
        conf = conf.max(-1).values

        # predictions are normalised to (0, 1) over pc_range; scale each axis
        # back to metres before measuring anything so the numbers are readable
        scale = pts_preds.new_tensor([self.pc_range[3] - self.pc_range[0],
                                      self.pc_range[4] - self.pc_range[1]])
        centres = pts_preds.mean(-2)                     # [B, Q, 2]
        spread = (centres.std(dim=1) * scale).mean()

        pts_targets = torch.cat(pts_targets_list, 0)
        # both coordinate channels of the weight carry the same 0/1 mask
        pts_weights = torch.cat(pts_weights_list, 0)[..., 0]
        pts_flat = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        # Euclidean, NOT the mean of |dx| and |dy|: averaging the two axes
        # under-reports a single-axis error by 2x and a 45-degree one by
        # sqrt(2), which is exactly the mistake that made the first map/occ
        # shift controls read as passing.
        delta = (pts_flat - normalize_2d_pts(pts_targets, self.pc_range)) * scale
        err = delta.norm(dim=-1) * pts_weights
        denom = pts_weights.sum().clamp(min=1)
        return {
            'map_cls_acc': acc,
            'map_conf_pos': conf[pos].mean() if pos.any() else zero,
            'map_conf_neg': conf[neg].mean() if neg.any() else zero,
            'map_spread_m': spread,
            'map_pts_err_m': err.sum() / denom,
            'map_n_gt': flat_cls.new_tensor(
                num_total_pos / max(cls_scores.size(0), 1)),
        }

    def forward_test(self, bev_embed, img_metas, rescale=False):
        preds = self(bev_embed, img_metas)
        if self.map_bbox_coder is None:
            return preds
        return self.get_bboxes(preds, img_metas, rescale=rescale)

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.map_bbox_coder.decode(preds_dicts)
        ret_list = []
        for preds in preds_dicts:
            ret_list.append([
                preds['map_bboxes'], preds['map_scores'], preds['map_labels'],
                preds['map_pts']
            ])
        return ret_list
