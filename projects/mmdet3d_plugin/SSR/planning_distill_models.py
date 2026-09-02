"""Teacher-adapter pretraining model and SSR-compatible planning decoder."""
import copy

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import DETECTORS, HEADS, build_loss
from mmdet.models.detectors.base import BaseDetector

from .planner.metric_stp3 import PlanningMetric
from .tokenlearner import TokenLearnerV11
from .utils.planning_distill import (
    PlanningBEVAdapter, TeacherFeatureStore, _STORE_CFG_KEYS,
    current_sample_tokens, resize_bev_tokens)


class _NavigationSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = nn.Sigmoid()

    def forward(self, feature, condition):
        condition = self.mlp_expand(self.act(self.mlp_reduce(condition)))
        return feature * self.gate(condition)


@HEADS.register_module()
class SSRPlanningDecoder(BaseModule):
    """The planning portion of :class:`ParaSSRHead`, without a BEV encoder.

    The adapter operates on a compact grid (25x25 by default).  A fixed
    bilinear resize restores SSR's 100x100 grid before the unchanged
    navigation gate, TokenLearner, latent decoder and waypoint decoder.
    """

    def __init__(self,
                 input_size=(25, 25),
                 bev_h=100,
                 bev_w=100,
                 embed_dims=256,
                 num_scenes=16,
                 num_reg_fcs=2,
                 fut_ts=6,
                 ego_fut_mode=3,
                 num_navi_cmd=3,
                 positional_encoding=dict(
                     type='LearnedPositionalEncoding', num_feats=128,
                     row_num_embed=100, col_num_embed=100),
                 latent_decoder=None,
                 way_decoder=None,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.input_size = tuple(input_size)
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.embed_dims = int(embed_dims)
        self.num_scenes = int(num_scenes)
        self.num_reg_fcs = int(num_reg_fcs)
        self.fut_ts = int(fut_ts)
        self.ego_fut_mode = int(ego_fut_mode)
        self.fp16_enabled = False

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        if positional_encoding['num_feats'] * 2 != self.embed_dims:
            raise ValueError('embed_dims must equal 2 * positional num_feats')

        self.navi_embedding = nn.Embedding(num_navi_cmd, self.embed_dims)
        self.navi_se = _NavigationSE(self.embed_dims)
        self.tokenlearner = TokenLearnerV11(
            self.num_scenes, self.embed_dims * 2)
        self.latent_decoder = build_transformer_layer_sequence(latent_decoder)
        self.way_point = nn.Embedding(
            self.ego_fut_mode * self.fut_ts, self.embed_dims * 2)
        self.way_decoder = build_transformer_layer_sequence(way_decoder)

        layers = []
        for _ in range(self.num_reg_fcs):
            layers.extend((Linear(self.embed_dims, self.embed_dims), nn.ReLU()))
        layers.append(Linear(self.embed_dims, 2))
        self.ego_fut_decoder = nn.Sequential(*layers)
        self.loss_plan_reg = build_loss(loss_plan_reg)

    def init_weights(self):
        for decoder in (self.latent_decoder, self.way_decoder):
            for parameter in decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, adapter_tokens, cmd):
        bev_embed = resize_bev_tokens(
            adapter_tokens, self.input_size, (self.bev_h, self.bev_w))
        batch_size, _, _ = bev_embed.shape
        dtype, device = bev_embed.dtype, bev_embed.device

        bev_mask = torch.zeros(
            batch_size, self.bev_h, self.bev_w, dtype=dtype, device=device)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        pos_tokens = bev_pos.flatten(2).permute(0, 2, 1)

        num_commands = self.navi_embedding.num_embeddings
        if cmd is None or cmd.size(0) != batch_size or \
                cmd.numel() != batch_size * num_commands:
            shape = None if cmd is None else tuple(cmd.shape)
            raise ValueError(
                f'expected one {num_commands}-way command per sample, got '
                f'{shape} for batch {batch_size}')
        cmd_index = cmd.reshape(batch_size, num_commands).argmax(dim=-1)
        navi = self.navi_embedding(cmd_index).unsqueeze(1)
        bev_query = torch.cat((self.navi_se(bev_embed, navi), pos_tokens), -1)

        latent, selected = self.tokenlearner(bev_query)
        latent = latent.permute(1, 0, 2)
        latent, latent_pos = torch.split(latent, self.embed_dims, dim=2)
        latent = self.latent_decoder(
            query=latent, key=latent, value=latent,
            query_pos=latent_pos, key_pos=latent_pos)

        waypoint_pos, waypoint = torch.split(
            self.way_point.weight.to(dtype), self.embed_dims, dim=1)
        waypoint_pos = waypoint_pos.unsqueeze(0).expand(
            batch_size, -1, -1).permute(1, 0, 2)
        waypoint = waypoint.unsqueeze(0).expand(
            batch_size, -1, -1).permute(1, 0, 2)
        waypoint = self.way_decoder(
            query=waypoint, key=latent, value=latent,
            query_pos=waypoint_pos, key_pos=latent_pos)

        trajectories = self.ego_fut_decoder(waypoint)
        trajectories = trajectories.permute(1, 0, 2).reshape(
            batch_size, self.ego_fut_mode, self.fut_ts, 2)
        return dict(
            adapter_embed=adapter_tokens,
            bev_embed=bev_embed,
            scene_query=latent,
            token_attn=selected,
            ego_fut_preds=trajectories)

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, preds_dicts, ego_fut_gt, ego_fut_masks, ego_fut_cmd):
        prediction = preds_dicts['ego_fut_preds']
        gt = ego_fut_gt.squeeze(1)
        mask = ego_fut_masks.squeeze(1).squeeze(1)
        command = ego_fut_cmd.squeeze(1).squeeze(1)
        gt = gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        weight = command[..., None, None] * mask[:, None, :, None]
        weight = weight.repeat(1, 1, 1, 2)
        return dict(loss_plan_reg=self.loss_plan_reg(prediction, gt, weight))


class _TeacherAdapterBranch(nn.Module):
    def __init__(self, adapter, planner):
        super().__init__()
        self.adapter = PlanningBEVAdapter(**copy.deepcopy(adapter))
        self.planner = SSRPlanningDecoder(**copy.deepcopy(planner))

    def forward(self, feature, command):
        adapted = self.adapter(feature)
        return self.planner(adapted, command)


@DETECTORS.register_module()
class CachedTeacherAdapterPlanner(BaseDetector):
    """Train planning adapters/heads on frozen, cached teacher BEVs.

    No teacher parameter is part of this model.  A cache entry is the output of
    a frozen teacher under ``torch.no_grad()``, so the only trainable modules
    are exactly ``branches.*.adapter`` and ``branches.*.planner``.
    """

    def __init__(self, feature_root, teachers, test_teacher='ensemble',
                 active_teachers=None, train_cfg=None, test_cfg=None,
                 pretrained=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # MMDetection3D's detector builder always supplies these two keyword
        # arguments, even when the model does not use detector-specific train
        # or test configuration.
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # ``tools/test.py`` clears this conventional detector option before
        # checkpoint loading.  This cache-only model has no separate backbone
        # pretrain source, so only ``None`` is meaningful.
        if pretrained is not None:
            raise ValueError(
                'CachedTeacherAdapterPlanner does not use pretrained=; '
                'load its stage-1 checkpoint through the runner')
        if active_teachers is None:
            active_teachers = list(teachers)
        else:
            active_teachers = list(active_teachers)
        if len(active_teachers) != len(set(active_teachers)):
            raise ValueError(f'duplicate active teachers: {active_teachers}')
        unknown = [name for name in active_teachers if name not in teachers]
        if unknown:
            raise KeyError(f'unknown active teachers: {unknown}')

        self.branches = nn.ModuleDict()
        self.stores = {}
        self.branch_weights = {}
        for name in active_teachers:
            cfg = teachers[name]
            cfg = copy.deepcopy(dict(cfg))
            self.branches[name] = _TeacherAdapterBranch(
                cfg.pop('adapter'), cfg.pop('planner'))
            store_kwargs = {}
            for src, dst in _STORE_CFG_KEYS.items():
                if src in cfg:
                    store_kwargs[dst] = cfg.pop(src)
            self.stores[name] = TeacherFeatureStore(
                feature_root, cfg.pop('cache_name', name), **store_kwargs)
            self.branch_weights[name] = float(cfg.pop('loss_weight', 1.0))
            if cfg:
                raise TypeError(f'unused teacher options for {name}: {cfg}')
        if not self.branches:
            raise ValueError('at least one teacher branch is required')
        if test_teacher != 'ensemble' and test_teacher not in self.branches:
            raise ValueError(f'unknown test_teacher: {test_teacher}')
        self.test_teacher = test_teacher
        self.planning_metric = None
        self.fp16_enabled = False

    def init_weights(self, pretrained=None):
        for branch in self.branches.values():
            branch.planner.init_weights()

    def extract_feat(self, imgs):
        raise RuntimeError('teacher-adapter training consumes cached BEVs')

    @staticmethod
    def _current_targets(ego_fut_trajs, ego_fut_masks, ego_fut_cmd):
        # Temporal training data is [B, queue, ...].  Test data has already
        # removed that dimension before reaching simple_test.
        if ego_fut_trajs.dim() >= 5:
            ego_fut_trajs = ego_fut_trajs[:, -1]
        if ego_fut_masks.dim() >= 5:
            ego_fut_masks = ego_fut_masks[:, -1]
        if ego_fut_cmd.dim() >= 5:
            ego_fut_cmd = ego_fut_cmd[:, -1]
        return ego_fut_trajs, ego_fut_masks, ego_fut_cmd

    def forward_train(self,
                      img_metas,
                      img=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      **kwargs):
        ego_fut_trajs, ego_fut_masks, ego_fut_cmd = self._current_targets(
            ego_fut_trajs, ego_fut_masks, ego_fut_cmd)
        tokens = current_sample_tokens(img_metas)
        parameter = next(self.parameters())
        losses = {}
        for name, branch in self.branches.items():
            feature, valid = self.stores[name].load_batch(
                tokens, parameter.device, parameter.dtype)
            outs = branch(feature, ego_fut_cmd)
            branch_losses = branch.planner.loss(
                outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd)
            weight = self.branch_weights[name]
            losses.update({
                f'{name}.{key}': value * weight
                for key, value in branch_losses.items()
            })
            with torch.no_grad():
                losses[f'adapter_std/{name}'] = \
                    outs['adapter_embed'].float().std()
                losses[f'cache_valid/{name}'] = valid.float().mean()
        return losses

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_test(self,
                     img_metas,
                     img=None,
                     ego_fut_trajs=None,
                     ego_fut_cmd=None,
                     **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError(f'img_metas must be a list, got {type(img_metas)}')
        # Match ParaSSR's test-time augmentation unwrap.
        metas = img_metas[0]
        trajs = ego_fut_trajs[0] if isinstance(ego_fut_trajs, list) \
            else ego_fut_trajs
        command = ego_fut_cmd[0] if isinstance(ego_fut_cmd, list) \
            else ego_fut_cmd
        return self.simple_test(
            img_metas=metas, ego_fut_trajs=trajs, ego_fut_cmd=command,
            **kwargs)

    def simple_test(self,
                    img_metas,
                    ego_fut_trajs,
                    ego_fut_cmd,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    map_gt_bboxes_3d=None,
                    map_gt_labels_3d=None,
                    gt_attr_labels=None,
                    fut_valid_flag=None,
                    **kwargs):
        tokens = current_sample_tokens(img_metas)
        parameter = next(self.parameters())
        predictions = {}
        for name, branch in self.branches.items():
            feature, _ = self.stores[name].load_batch(
                tokens, parameter.device, parameter.dtype)
            predictions[name] = branch(feature, ego_fut_cmd)['ego_fut_preds']
        if self.test_teacher == 'ensemble':
            prediction = torch.stack(list(predictions.values())).mean(0)
        else:
            prediction = predictions[self.test_teacher]

        if prediction.size(0) != 1:
            raise AssertionError('evaluation supports batch_size=1')
        result = dict(
            ego_fut_preds=prediction[0].cpu(),
            ego_fut_cmd=ego_fut_cmd.cpu(),
            sample_token=tokens[0])
        for name, value in predictions.items():
            result[f'ego_fut_preds_{name}'] = value[0].cpu()

        metric = self._planning_metrics(
            result['ego_fut_preds'], ego_fut_trajs, ego_fut_cmd,
            gt_bboxes_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            gt_attr_labels, fut_valid_flag)
        return [dict(pts_bbox=result, metric_results=metric)]

    @torch.no_grad()
    def _planning_metrics(self, prediction, gt_traj, command, gt_boxes,
                          map_boxes, map_labels, gt_attr, valid):
        gt_bbox = gt_boxes[0][0]
        gt_map_bbox = map_boxes[0]
        gt_map_label = map_labels[0].cpu()
        gt_attr_label = gt_attr[0][0].cpu()
        valid = bool(valid[0][0])
        gt_traj = gt_traj[0, 0]
        command_vector = command[0, 0, 0]
        command_index = torch.nonzero(command_vector)[0, 0]
        pred = prediction[command_index].cumsum(dim=-2)
        gt = gt_traj.cumsum(dim=-2)
        return self.compute_planner_metric_stp3(
            pred[None], gt[None], gt_bbox, gt_attr_label.unsqueeze(0),
            gt_map_bbox, gt_map_label, valid)

    def compute_planner_metric_stp3(self, pred_ego_fut_trajs,
                                    gt_ego_fut_trajs, gt_agent_boxes,
                                    gt_agent_feats, gt_map_boxes,
                                    gt_map_labels, fut_valid_flag):
        metric = {'fut_valid_flag': fut_valid_flag}
        for second in range(1, 4):
            for name in ('plan_L2', 'plan_obj_col', 'plan_obj_box_col',
                         'plan_L2_stp3', 'plan_obj_col_stp3',
                         'plan_obj_box_col_stp3'):
                metric[f'{name}_{second}s'] = 0.0
        if not fut_valid_flag:
            return metric
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, _ = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels)
        occupancy = torch.logical_or(segmentation, pedestrian)
        for i in range(3):
            current = (i + 1) * 2
            pred = pred_ego_fut_trajs[0, :current].detach().to(
                gt_ego_fut_trajs.device)
            gt = gt_ego_fut_trajs[0, :current]
            obj_col, box_col = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, :current].detach(),
                gt_ego_fut_trajs[:, :current], occupancy)
            suffix = f'{i + 1}s'
            metric[f'plan_L2_{suffix}'] = \
                self.planning_metric.compute_L2(pred, gt)
            metric[f'plan_L2_stp3_{suffix}'] = \
                self.planning_metric.compute_L2_stp3(pred, gt)
            metric[f'plan_obj_col_{suffix}'] = obj_col.mean().item()
            metric[f'plan_obj_box_col_{suffix}'] = box_col.mean().item()
            metric[f'plan_obj_col_stp3_{suffix}'] = obj_col[-1].item()
            metric[f'plan_obj_box_col_stp3_{suffix}'] = box_col[-1].item()
        return metric

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('test-time augmentation is not supported')

    def set_epoch(self, epoch):
        self.epoch = epoch
