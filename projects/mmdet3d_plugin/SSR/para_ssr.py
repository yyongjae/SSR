"""PARA-SSR detector: SSR's sparse-token planner with PARA-Drive aux heads.

Structure (PARA-Drive CVPR'24, Fig. 5)::

    images -> ResNet50+FPN -> BEVFormerEncoder(prev_bev) -> bev_embed
                                        |
        +---------------+---------------+---------------+
        |               |               |               |
    planning       det+motion         map          occupancy
    (always on)    (train-only)   (train-only)    (train-only)

No aux head consumes another head's output, and no aux head feeds the planner --
information passes between modules only implicitly, through the shared BEV
feature they co-train.  Because of that the aux heads can be switched off at
inference (``test_aux_heads=False``, the default), which is where PARA-Drive's
2.77x speed-up over UniAD comes from, and each can be set to ``None`` in the
config to reproduce the paper's module-necessity ablation (Table 5).

Relative to ``SSR``, the FFP / latent world model is gone: ``latent_world_model``,
``TokenFuser``, ``loss_bev``, ``obtain_next_bev`` and the future frame in the
temporal queue.  The temporal queue is therefore ``[prev, prev, current]`` and
``queue[-1]`` -- which ``union2one`` uses as the metadata container -- is finally
the *current* frame, so box/map/occupancy GT lines up with the images.
"""
import copy
import inspect

import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS, HEADS, build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; scales the gradient flowing backwards.

    Lets the aux heads train at full strength while limiting how strongly they
    reshape the shared BEV feature. Down-weighting the aux *losses* couples the
    two: it also slows the heads themselves, which matters when the heads are
    meant to be distillation teachers.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


@DETECTORS.register_module()
class ParaSSR(MVXTwoStageDetector):
    """Fully parallel SSR."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 det_motion_head=None,
                 map_head=None,
                 occ_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 task_loss_weight=None,
                 aux_grad_scale=1.0,
                 test_aux_heads=False):
        super(ParaSSR, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head, train_cfg, test_cfg,
            pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.embed_dims = 256

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.planning_metric = None

        # ---- parallel auxiliary heads (all optional) ----
        train_cfg_pts = train_cfg.pts if train_cfg is not None and \
            hasattr(train_cfg, 'pts') else None
        self.det_motion_head = self._build_aux(det_motion_head, train_cfg_pts)
        self.map_head = self._build_aux(map_head, train_cfg_pts)
        self.occ_head = self._build_aux(occ_head, train_cfg_pts)

        self.task_loss_weight = dict(plan=1.0, det=1.0, map=1.0, occ=1.0)
        if task_loss_weight is not None:
            self.task_loss_weight.update(task_loss_weight)
        # <1.0 keeps the aux heads learning normally while reducing how much
        # they pull the shared BEV feature (see _ScaleGrad)
        self.aux_grad_scale = aux_grad_scale
        self.test_aux_heads = test_aux_heads

    @staticmethod
    def _build_aux(cfg, train_cfg_pts):
        if cfg is None:
            return None
        cfg = copy.deepcopy(cfg)
        # Assigners live in train_cfg.pts, alongside the ones SSR already
        # declared. Only the matching heads accept it (the occupancy head is
        # query-free and needs no assigner).
        head_cls = HEADS.get(cfg['type'])
        if head_cls is not None and \
                'train_cfg' in inspect.signature(head_cls.__init__).parameters:
            cfg.setdefault('train_cfg', train_cfg_pts)
        return build_head(cfg)

    @property
    def with_aux_heads(self):
        return any(h is not None for h in
                   (self.det_motion_head, self.map_head, self.occ_head))

    # ------------------------------------------------------------------ #
    # feature extraction (unchanged from SSR)                            #
    # ------------------------------------------------------------------ #
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue,
                                  int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    def obtain_history_bev(self, imgs_queue, img_metas_list, prev_cmd):
        """Roll the BEV forward over the previous frames without gradients."""
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                cmd = prev_cmd[:, i, ...]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev=prev_bev, only_bev=True,
                    cmd=cmd)
            self.train()
            return prev_bev

    # ------------------------------------------------------------------ #
    # training                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_dummy(self, img):
        return self.forward_test(img=img, img_metas=[[None]])

    @force_fp32(apply_to=('img', 'points', 'prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      gt_occ_seg=None,
                      gt_occ_valid=None,
                      **kwargs):
        """One shared BEV forward, then every head in parallel on top of it."""
        len_queue = img.size(1)
        # [prev..., current] -- no future frame any more
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_cmd = ego_fut_cmd[:, :-1, ...]
        ego_fut_cmd = ego_fut_cmd[:, -1, ...]
        ego_fut_trajs = ego_fut_trajs[:, -1, ...]
        ego_fut_masks = ego_fut_masks[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, prev_cmd)
        img_metas = [each[len_queue - 1] for each in img_metas]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        outs = self.pts_bbox_head(
            img_feats, img_metas, prev_bev=prev_bev, cmd=ego_fut_cmd)
        bev_embed = outs['bev_embed']

        losses = dict()
        w = self.task_loss_weight

        plan_losses = self.pts_bbox_head.loss(
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd)
        losses.update(self._scale(plan_losses, w['plan'], prefix=''))

        # --- auxiliary heads: each sees only bev_embed ---
        aux_bev = bev_embed
        if self.aux_grad_scale != 1.0:
            aux_bev = _ScaleGrad.apply(bev_embed, self.aux_grad_scale)

        if self.det_motion_head is not None:
            det_losses = self.det_motion_head.forward_train(
                aux_bev, img_metas, gt_bboxes_3d, gt_labels_3d,
                gt_attr_labels)
            losses.update(self._scale(det_losses, w['det'], prefix='det.'))

        if self.map_head is not None:
            map_losses = self.map_head.forward_train(
                aux_bev, img_metas, map_gt_bboxes_3d, map_gt_labels_3d)
            losses.update(self._scale(map_losses, w['map'], prefix='map.'))

        if self.occ_head is not None:
            assert gt_occ_seg is not None, \
                'occ_head is enabled but gt_occ_seg is missing -- add ' \
                'GenerateSSROccLabels to the pipeline and gt_occ_seg/' \
                'gt_occ_valid to CustomCollect3D keys.'
            occ_losses = self.occ_head.forward_train(
                aux_bev, gt_occ_seg, gt_occ_valid)
            losses.update(self._scale(occ_losses, w['occ'], prefix='occ.'))

        return losses

    @staticmethod
    def _scale(loss_dict, weight, prefix=''):
        return {f'{prefix}{k}': v * weight for k, v in loss_dict.items()}

    # ------------------------------------------------------------------ #
    # inference                                                          #
    # ------------------------------------------------------------------ #
    def forward_test(self,
                     img_metas,
                     gt_bboxes_3d,
                     gt_labels_3d,
                     map_gt_bboxes_3d=None,
                     map_gt_labels_3d=None,
                     img=None,
                     ego_his_trajs=None,
                     ego_fut_trajs=None,
                     ego_fut_cmd=None,
                     ego_lcf_feat=None,
                     gt_attr_labels=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            map_gt_labels_3d=map_gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0] if ego_his_trajs is not None else None,
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0] if ego_lcf_feat is not None else None,
            gt_attr_labels=gt_attr_labels,
            **kwargs)

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.prev_frame_info['prev_angle'] = tmp_angle
        return bbox_results

    def simple_test(self,
                    img_metas,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    map_gt_bboxes_3d=None,
                    map_gt_labels_3d=None,
                    img=None,
                    prev_bev=None,
                    points=None,
                    fut_valid_flag=None,
                    rescale=False,
                    ego_his_trajs=None,
                    ego_fut_trajs=None,
                    ego_fut_cmd=None,
                    ego_lcf_feat=None,
                    gt_attr_labels=None,
                    **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            gt_attr_labels=gt_attr_labels)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        return new_prev_bev, bbox_list

    def simple_test_pts(self,
                        x,
                        img_metas,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        map_gt_bboxes_3d=None,
                        map_gt_labels_3d=None,
                        prev_bev=None,
                        fut_valid_flag=None,
                        rescale=False,
                        ego_fut_trajs=None,
                        ego_fut_cmd=None,
                        gt_attr_labels=None):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  cmd=ego_fut_cmd)

        bbox_results = []
        for i in range(len(outs['ego_fut_preds'])):
            bbox_result = dict()
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        # PARA-Drive: the aux modules are only needed for interpretability /
        # safety checks / display, so they stay off unless asked for.
        if self.test_aux_heads:
            bev_embed = outs['bev_embed']
            if self.det_motion_head is not None:
                bbox_results[0]['det_motion'] = self.det_motion_head.forward_test(
                    bev_embed, img_metas, rescale=rescale)
            if self.map_head is not None:
                bbox_results[0]['map'] = self.map_head.forward_test(
                    bev_embed, img_metas, rescale=rescale)
            if self.occ_head is not None:
                occ_out = self.occ_head.forward_test(bev_embed)
                bbox_results[0]['occ_seg'] = occ_out['occ_seg'].cpu()

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        with torch.no_grad():
            gt_bbox = gt_bboxes_3d[0][0]
            gt_map_bbox = map_gt_bboxes_3d[0]
            gt_map_label = map_gt_labels_3d[0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])

            metric_dict = {}
            assert ego_fut_trajs.shape[0] == 1, \
                'only support batch_size=1 for testing'
            ego_fut_preds = bbox_results[0]['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd_ = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd_)[0, 0]

            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx].cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict.update(
                self.compute_planner_metric_stp3(
                    pred_ego_fut_trajs=ego_fut_pred[None],
                    gt_ego_fut_trajs=ego_fut_trajs[None],
                    gt_agent_boxes=gt_bbox,
                    gt_agent_feats=gt_attr_label.unsqueeze(0),
                    gt_map_boxes=gt_map_bbox,
                    gt_map_labels=gt_map_label,
                    fut_valid_flag=fut_valid_flag))

        return outs['bev_embed'], bbox_results, metric_dict

    # ------------------------------------------------------------------ #
    # planning metric (identical to SSR)                                 #
    # ------------------------------------------------------------------ #
    def compute_planner_metric_stp3(self,
                                    pred_ego_fut_trajs,
                                    gt_ego_fut_trajs,
                                    gt_agent_boxes,
                                    gt_agent_feats,
                                    gt_map_boxes,
                                    gt_map_labels,
                                    fut_valid_flag):
        """Planning L2 / collision -- identical protocol to SSR and VAD."""
        metric_dict = {
            'plan_L2_1s': 0,
            'plan_L2_2s': 0,
            'plan_L2_3s': 0,
            'plan_obj_col_1s': 0,
            'plan_obj_col_2s': 0,
            'plan_obj_col_3s': 0,
            'plan_obj_box_col_1s': 0,
            'plan_obj_box_col_2s': 0,
            'plan_obj_box_col_3s': 0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, segmentation_plus = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time])
                traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time])
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time], occupancy)
                metric_dict['plan_L2_{}s'.format(i + 1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = traj_L2_stp3
                metric_dict['plan_obj_col_stp3_{}s'.format(i + 1)] = obj_coll[-1].item()
                metric_dict['plan_obj_box_col_stp3_{}s'.format(i + 1)] = obj_box_coll[-1].item()
            else:
                metric_dict['plan_L2_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = 0.0

        return metric_dict

    def set_epoch(self, epoch):
        """Called by CustomSetEpochInfoHook."""
        self.epoch = epoch
        for head in (self.pts_bbox_head, self.det_motion_head, self.map_head,
                     self.occ_head):
            if head is not None:
                head.epoch = epoch
