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
from scipy.optimize import linear_sum_assignment
from mmdet.models import DETECTORS, HEADS, build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric
from projects.mmdet3d_plugin.SSR.utils.grad_balance import (
    GradBalancer, all_reduce_mean)

# ``self.CLASSES`` is injected by tools/train.py and tools/test.py from the
# dataset. Fall back to the standard nuScenes detection order so the motion
# metrics still work when a script forgets to set it.
NUSCENES_CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone')


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
                 grad_balance=None,
                 grad_norm_log_interval=0,
                 encoder_grad_log=True,
                 aux_metric_log_interval=0,
                 motion_score_thresh=0.6,
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

        self.task_loss_weight = dict(
            plan=1.0, det=1.0, motion=1.0, map=1.0, occ=1.0)
        if task_loss_weight is not None:
            self.task_loss_weight.update(task_loss_weight)
        # <1.0 keeps the aux heads learning normally while reducing how much
        # they pull the shared BEV feature (see _ScaleGrad)
        self.aux_grad_scale = aux_grad_scale
        # ...or hand the same valve to a controller that re-solves it per task
        # against a target gradient share. aux_grad_scale sets a knob and lets
        # the share fall where it may; grad_balance sets the share and solves
        # for the knob. Mutually exclusive.
        self.grad_balancer = None
        if grad_balance is not None:
            assert aux_grad_scale == 1.0, \
                'grad_balance replaces aux_grad_scale; do not set both'
            self.grad_balancer = GradBalancer(**grad_balance)
        # >0 logs ||d L_task / d bev_embed|| every N iterations. The tasks
        # compete for one shared representation and their gradients differ by
        # orders of magnitude, so the loss values alone do not show which task
        # is actually steering the BEV feature.
        self.grad_norm_log_interval = grad_norm_log_interval
        # Also decompose the gradient at the BEV encoder's parameters, not just
        # at its output. Costs one extra encoder-only backward per task on the
        # measurement iterations (the head-side backward is reused).
        self.encoder_grad_log = encoder_grad_log
        # >0 logs aux-head *quality* (occupancy IoU and positive/negative
        # probability separation) every N iterations. Loss curves alone do not
        # tell a learning head from one that collapsed to a constant.
        self.aux_metric_log_interval = aux_metric_log_interval
        # VAD's ViP3D protocol scores motion on detections above this score.
        self.motion_score_thresh = motion_score_thresh
        self._train_iter = 0
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
        groups = dict()
        w = self.task_loss_weight

        # Declared before the first loss: the planner's per-command breakdown is
        # produced inside pts_bbox_head.loss, on the same schedule as the aux
        # heads' quality metrics.
        aux_metrics = {}
        want_metrics = (self.aux_metric_log_interval > 0 and
                        (self._train_iter + 1) % self.aux_metric_log_interval == 0)
        if want_metrics:
            aux_metrics.update(self._representation_metrics(outs))

        plan_losses = self.pts_bbox_head.loss(
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd,
            metrics_out=aux_metrics if want_metrics else None)
        groups['plan'] = self._scale(plan_losses, w['plan'], prefix='')
        losses.update(groups['plan'])

        # --- auxiliary heads: each sees only bev_embed ---
        # One valve per head, because that is the finest granularity available:
        # detection and motion share a valve since they come out of one forward
        # pass through det_motion_head and therefore one BEV input.
        def _valve(task):
            s = self._task_scale(task)
            return bev_embed if s == 1.0 else _ScaleGrad.apply(bev_embed, s)

        # Only for heads that exist. Building a valve for a disabled head would
        # hang an unused _ScaleGrad node off bev_embed every iteration.
        aux_bev = {t: _valve(t) for t, h in
                   (('det', self.det_motion_head), ('map', self.map_head),
                    ('occ', self.occ_head)) if h is not None}

        # Quality (not loss) probes, gathered while the heads run so no extra
        # forward pass is needed. Must be declared before the first head.

        if self.det_motion_head is not None:
            head_losses = self.det_motion_head.forward_train(
                aux_bev['det'], img_metas, gt_bboxes_3d, gt_labels_3d,
                gt_attr_labels)
            # det and motion are separate TASKS with separate loss weights,
            # even though one forward produces both. `motion=0.0` therefore
            # leaves detection entirely untouched -- and because _scale drops a
            # zero-weighted task rather than multiplying it by zero, the
            # motion-only parameters get no gradient at all, so AdamW's
            # decoupled weight decay does not quietly shrink them either.
            #
            # They still share one grad_balance VALVE: aux_bev['det'] is a
            # single tensor, so there is a single scale on the gradient going
            # back into bev_embed. That is why grad_balance targets name `det`
            # and not `motion`.
            motion_losses = {k: v for k, v in head_losses.items() if 'traj' in k}
            det_losses = {k: v for k, v in head_losses.items()
                          if 'traj' not in k}
            scaled = self._scale(det_losses, w['det'], prefix='det.')
            scaled.update(
                self._scale(motion_losses, w.get('motion', 1.0),
                            prefix='motion.'))
            groups['det'] = scaled
            losses.update(scaled)

        if self.map_head is not None:
            map_losses = self.map_head.forward_train(
                aux_bev['map'], img_metas, map_gt_bboxes_3d, map_gt_labels_3d,
                metrics_out=aux_metrics if want_metrics else None)
            groups['map'] = self._scale(map_losses, w['map'], prefix='map.')
            losses.update(groups['map'])

        if self.occ_head is not None:
            assert gt_occ_seg is not None, \
                'occ_head is enabled but gt_occ_seg is missing -- add ' \
                'GenerateSSROccLabels to the pipeline and gt_occ_seg/' \
                'gt_occ_valid to CustomCollect3D keys.'
            occ_losses = self.occ_head.forward_train(
                aux_bev['occ'], gt_occ_seg, gt_occ_valid,
                metrics_out=aux_metrics if want_metrics else None)
            groups['occ'] = self._scale(occ_losses, w['occ'], prefix='occ.')
            losses.update(groups['occ'])

        # Quality, not loss. A falling loss does not prove a head learned
        # anything -- the v1 dense map head cut its BCE 18% while converging to
        # a constant. These keys avoid the substring "loss" so _parse_losses
        # logs them without optimising them.
        losses.update(aux_metrics)

        self._train_iter += 1
        # One measurement serves both the log and the balancer, so take it if
        # either wants it this iteration. Both conditions are iteration-based
        # and therefore identical on every rank, which _parse_losses's
        # all_reduce over log_vars requires.
        want_gnorm = (self.grad_norm_log_interval > 0 and
                      self._train_iter % self.grad_norm_log_interval == 0)
        want_balance = (self.grad_balancer is not None and
                        self.grad_balancer.should_update(self._train_iter))
        if want_gnorm or want_balance:
            norms, cosines = self._bev_grad_norms(bev_embed, groups)
            if want_balance:
                # Average across ranks BEFORE solving: each rank sees its own
                # batch, and ranks that disagree on the scales are silently
                # optimising different objectives.
                raw = all_reduce_mean({k: float(v) for k, v in norms.items()},
                                      bev_embed.device)
                self.grad_balancer.update(raw)
            if want_gnorm:
                losses.update(self._format_grad_norms(norms))
                losses.update(cosines)
                # The UNSCALED norm, i.e. what each task would push into the
                # BEV with no valve at all. gnorm/* is measured through
                # _ScaleGrad and so already carries the balancer's scale;
                # recalibrating a target by hand needs the raw number, and
                # dividing it back out afterwards is exactly the step that
                # gets forgotten.
                losses.update({
                    f'gnorm_raw/{k}': v / max(self._task_scale(k), 1e-12)
                    for k, v in norms.items()})
                if self.grad_balancer is not None:
                    losses.update({
                        k: bev_embed.new_tensor(v) for k, v
                        in self.grad_balancer.log_dict().items()})

        return losses

    @staticmethod
    @torch.no_grad()
    def _representation_metrics(outs):
        """Has the shared representation itself degenerated?

        Every other quality metric here is per-head. These two watch the thing
        all the heads read, and they catch a failure no loss curve shows.

        ``tok_sim``  mean off-diagonal cosine between SSR's 16 scene tokens.
                     The whole premise of the paper is that those tokens carry
                     DIFFERENT parts of the scene; at 1.0 they are copies of
                     each other and 16 tokens are doing one token's work. v1
                     was already suspicious here (report #01 7 measured token
                     similarity up 63% against the baseline), so it is worth a
                     continuous reading rather than a post-mortem.
        ``bev_std``  spatial standard deviation of ``bev_embed``, averaged over
                     channels. Near zero means the BEV feature is the same
                     everywhere -- the representation-level version of the map
                     head's query collapse.

        Both are a couple of reductions on tensors already in memory, on the
        aux_metric_log_interval schedule.
        """
        out = {}
        q = outs.get('scene_query')
        if q is not None and q.dim() == 3:
            q = q.detach().permute(1, 0, 2)                  # [N,B,C] -> [B,N,C]
            n = q.size(1)
            if n > 1:
                qn = torch.nn.functional.normalize(q.float(), dim=-1)
                sim = torch.bmm(qn, qn.transpose(1, 2))      # [B, N, N]
                off = ~torch.eye(n, dtype=torch.bool, device=sim.device)
                out['tok_sim'] = sim[:, off].mean()
        bev = outs.get('bev_embed')
        if bev is not None and bev.dim() == 3:
            # [B, H*W, C] -- std over the spatial axis, then over channels
            out['bev_std'] = bev.detach().float().std(dim=1).mean()

        # How much of the BEV survives the 16-token bottleneck. TokenLearner
        # returns `selected` [B, n_tokens, H*W], a softmax over BEV cells, so
        # each token is a convex combination of cells and the attention itself
        # says what it kept.
        #
        #   tok_cover  exp(entropy) / (H*W): the effective FRACTION of the BEV
        #              one token averages over. Near 1/(H*W) means each token
        #              is a single cell; near 1 means each token is the global
        #              mean and the 16 of them carry one number's worth of
        #              spatial information.
        #   tok_union  fraction of BEV cells receiving more than uniform
        #              attention from at least one token -- the spatial support
        #              of the whole bottleneck, overlap included.
        #
        # This is the cheap half of "what does compression cost planning": it
        # measures what is discarded, not what that costs in metres of waypoint
        # error. The expensive half needs a dense-BEV planner trained alongside
        # as a reference.
        sel = outs.get('token_attn')
        if sel is not None and sel.dim() == 3:
            p = sel.detach().float()
            hw = p.size(-1)
            ent = -(p * (p + 1e-12).log()).sum(-1)          # [B, n] nats
            out['tok_cover'] = (ent.exp() / hw).mean()
            out['tok_union'] = (p.max(dim=1).values > 1.0 / hw).float().mean()
        return out

    def _task_scale(self, task):
        """The valve currently on this task's path into bev_embed.

        `plan` is never scaled; `motion` shares `det`'s valve because both come
        out of one forward through det_motion_head.
        """
        if task in ('plan', 'motion'):
            task = 'det' if task == 'motion' else task
            if task == 'plan':
                return 1.0
        if self.grad_balancer is not None:
            return self.grad_balancer.scale_for(task)
        return self.aux_grad_scale if task != 'plan' else 1.0

    def _bev_grad_norms(self, bev_embed, groups):
        """``||d L_task / d bev_embed||`` per task, at the shared feature.

        Uses the *weighted* losses, i.e. what actually reaches the trunk, so the
        numbers reflect task_loss_weight and whatever valve is in effect --
        ``autograd.grad`` differentiates through ``_ScaleGrad``, so an aux norm
        here is already multiplied by its scale. Back that out before solving
        for a new scale (GradBalancer.update does).

        Detection and motion come out of one head and are reported separately
        for information: ``motion`` is a SUBSET of ``det``'s path, so the shares
        of {plan, det, map, occ} sum to 1 and motion sits outside that sum.
        """
        groups = dict(groups)
        det = groups.get('det')
        if det is not None:
            motion = {k: v for k, v in det.items() if k.startswith('motion.')}
            if motion:
                groups['motion'] = motion
        grads, enc_grads = {}, {}
        enc_params = self._encoder_params() if self.encoder_grad_log else []
        for name, loss_dict in groups.items():
            terms = [v for v in loss_dict.values()
                     if torch.is_tensor(v) and v.requires_grad]
            if not terms:
                continue
            g = torch.autograd.grad(sum(terms), bev_embed, retain_graph=True,
                                    allow_unused=True)[0]
            grads[name] = None if g is None else g.detach().flatten()
            if enc_params:
                # same reason as in _bev_grad_cosines: unconditional keys
                enc_grads[name] = (
                    bev_embed.new_zeros(1) if g is None
                    else self._encoder_grad(bev_embed, enc_params, g))
        zero = bev_embed.new_zeros(())
        norms = {k: (zero if v is None else v.norm()) for k, v in grads.items()}
        # Each gradient is the size of bev_embed (41 MB at batch 4). Only the
        # tasks that take part in a cosine need to stay resident; motion does
        # not (it is a subset of det's path), so drop it before the pairwise
        # pass rather than hold a copy for nothing.
        grads.pop('motion', None)
        cosines = self._bev_grad_cosines(grads, norms, zero)
        grads.clear()

        # The same decomposition one level deeper: at the encoder's WEIGHTS
        # rather than at the feature it produces. These are not the same
        # question. gshare says who is pushing the shared FEATURE around;
        # enc_gshare says who is reshaping the shared ENCODER, and the two can
        # disagree -- a task whose feature-gradient concentrates on a handful
        # of BEV cells writes a very different weight-gradient than one spread
        # evenly, even at identical feature-gradient norm.
        #
        # enc_gcos is arguably the more meaningful conflict measurement of the
        # two, because it is conflict in the direction the shared trunk's
        # parameters actually move.
        if enc_grads:
            enc_norms = {k: v.norm() for k, v in enc_grads.items()}
            enc_grads.pop('motion', None)
            enc_cos = self._bev_grad_cosines(enc_grads, enc_norms, zero)
            enc_grads.clear()
            cosines.update({k.replace('gcos/', 'enc_gcos/'): v
                            for k, v in enc_cos.items()})
            cosines.update(self._format_grad_norms(enc_norms, prefix='enc_'))
        return norms, cosines

    def _encoder_params(self):
        """The BEV encoder's own parameters, i.e. the shared trunk below
        ``bev_embed``. Everything else in ``pts_bbox_head`` is the planner."""
        head = getattr(self, 'pts_bbox_head', None)
        if head is None:
            return []
        pre = ('transformer.', 'bev_embedding', 'positional_encoding')
        return [p for n, p in head.named_parameters()
                if n.startswith(pre) and p.requires_grad]

    @staticmethod
    def _encoder_grad(bev_embed, enc_params, g_flat):
        """``dL_task/dW_enc`` from the already-computed ``dL_task/d bev_embed``.

        The chain rule, used to avoid paying twice: the head-side backward has
        already run to produce ``g_flat``, so feeding it as ``grad_outputs``
        continues from bev_embed into the encoder and nothing else. Asking for
        ``autograd.grad(loss, enc_params)`` directly would redo the head. The
        backbone is upstream of the encoder, so it is never entered either.
        """
        gs = torch.autograd.grad(
            bev_embed, enc_params, grad_outputs=g_flat.view_as(bev_embed),
            retain_graph=True, allow_unused=True)
        parts = [x.detach().flatten() for x in gs if x is not None]
        return torch.cat(parts) if parts else bev_embed.new_zeros(1)

    @staticmethod
    def _bev_grad_cosines(grads, norms, zero):
        """Pairwise cos(angle) between the tasks' gradients at ``bev_embed``.

        This is the number ``gshare`` structurally cannot produce. gshare is a
        ratio of MAGNITUDES: it says how loudly each task is speaking, never
        whether two of them are saying the same thing. Two tasks can hold 30%
        each and either reinforce or cancel, and the loss curves look identical
        either way.

            cos > 0   the task pushes the shared BEV the same way planning does
                      -- auxiliary supervision is pulling in the same direction
            cos ~ 0   orthogonal; the task uses capacity planning does not touch
            cos < 0   conflict; the task is actively dragging the representation
                      away from what planning wants

        ``gcos/plan-det`` and ``gcos/plan-map`` are therefore the direct
        mechanistic reading of "is the auxiliary supervision helping or hurting
        the planner", which the final L2 can only answer in aggregate and only
        against a control run.

        Motion is excluded: its gradient is a subset of det's path, so the
        cosine between them measures nesting, not conflict.

        Cost is one dot product per pair on tensors already in memory, on the
        same schedule as the norms.
        """
        # Motion is filtered here as well as by the caller. The caller drops it
        # to save memory; this keeps the docstring's guarantee true for anyone
        # who calls the helper directly, which is exactly what the regression
        # test does.
        # The key set must not depend on the data. _parse_losses all-reduces
        # every entry of log_vars in dict order, so a rank that omits one key
        # because its gradient came back None desynchronises the collective --
        # a hang, two ranks deep, days into a run. Emit every pair
        # unconditionally and put a zero in when there is nothing to measure.
        keys = [k for k in grads if k != 'motion']
        out = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ga, gb = grads[a], grads[b]
                if ga is None or gb is None:
                    out[f'gcos/{a}-{b}'] = zero
                    continue
                denom = (norms[a] * norms[b]).clamp(min=1e-12)
                out[f'gcos/{a}-{b}'] = torch.dot(ga, gb) / denom
        return out

    @staticmethod
    def _format_grad_norms(norms, prefix=''):
        """Diagnostic keys. They avoid the substring "loss" so that mmdet's
        _parse_losses logs them without adding them to the optimised total."""
        out = {}
        # motion overlaps det, so it must not enter the normaliser
        total = sum(v for k, v in norms.items() if k != 'motion')
        for name, value in norms.items():
            out[f'{prefix}gnorm/{name}'] = value
            out[f'{prefix}gshare/{name}'] = value / total.clamp(min=1e-12)
        return out

    @staticmethod
    def _scale(loss_dict, weight, prefix=''):
        """Apply a task weight, and treat 0 as OFF rather than as a factor.

        Dropping the terms is not the same as multiplying them by zero, because
        of AdamW. `loss * 0.0` is still a tensor in the graph, so backward gives
        those parameters a ZEROS gradient rather than no gradient -- and AdamW
        skips a parameter only when `p.grad is None`. A zeros gradient still
        goes through

            param.mul_(1 - lr * weight_decay)

        every step, decoupled weight decay being decoupled from the gradient.
        Measured for stage 1 (48 epochs x 3516 iterations, cosine mean LR ~1e-4,
        wd 0.01): exp(-1e-4 * 0.01 * 168768) = 0.845, so the planner-only
        modules would finish stage 1 at 85% of their initial norm despite never
        having been trained.

        The effect is mild -- the eight LayerNorms in the planner renormalise
        their input, so the shrink does not compound through the stack; what is
        left is a 15% smaller initial waypoint scale and slightly flatter
        attention, both recovered early in stage 2. It is also pointless, and
        avoiding it costs one branch. VAD zeroes its planner loss weights and
        therefore has the same artefact; this is a deliberate departure.
        """
        if weight == 0:
            return {}
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
        # Carry the pipeline's occupancy GT through so evaluate_occ() scores the
        # head against exactly the target the loss used. MultiScaleFlipAug3D
        # wraps collected keys in a one-element list.
        def _unwrap(x):
            while isinstance(x, (list, tuple)):
                x = x[0]
            return x

        # Stamp the sample this result came from. Every formatter downstream
        # pairs result[i] with self.data_infos[i]['token'] positionally, and the
        # 2-rank sampler + collect_results_cpu have to agree on that ordering
        # for it to hold. Carrying the token makes the assumption checkable
        # instead of silently scoring predictions against the wrong scene.
        bbox_pts[0]['sample_token'] = _unwrap(img_metas)['sample_idx']

        # `self.occ_head is not None` matters as much as test_aux_heads: the
        # pipeline still rasterises occupancy GT even when the head is disabled
        # (stripping it from a base pipeline list is not something a derived
        # config can do), and without this check that GT is carried through
        # collect_results_cpu for all 6019 samples -- 0.39 GiB -- to be scored
        # against a prediction that was never made.
        gt_occ_seg = kwargs.get('gt_occ_seg', None)
        if gt_occ_seg is not None and self.test_aux_heads and \
                self.occ_head is not None:
            # uint8, not the float32 the format bundle produced: this is held
            # for all 6019 val samples, and [1,T,C,100,100] float32 alone is
            # 1.57 GiB on top of the predictions. The labels are 0/1 and
            # evaluate_occ() calls .bool() on them anyway.
            bbox_pts[0]['gt_occ_seg'] = _unwrap(gt_occ_seg).to(torch.uint8).cpu()
            # Frames past the end of a scene carry no usable GT. The loss masks
            # them; the metric has to as well or those frames are scored as
            # "everything is empty" (8.7% of val future frames are invalid).
            gt_occ_valid = kwargs.get('gt_occ_valid', None)
            if gt_occ_valid is not None:
                bbox_pts[0]['gt_occ_valid'] = \
                    _unwrap(gt_occ_valid).to(torch.bool).cpu()

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        return new_prev_bev, bbox_list

    @staticmethod
    def det_pred2result(bboxes, scores, labels, trajs):
        """Decoded detection+motion predictions -> nuScenes evaluator keys.

        ``output_to_nusc_box`` reads boxes_3d / scores_3d / labels_3d /
        trajs_3d, so these sit at the top level of the per-sample result dict.
        Mirrors VAD's ``VAD.pred2result``.
        """
        return dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            trajs_3d=trajs.cpu())

    @staticmethod
    def assign_pred_to_gt_vip3d(bbox_result, gt_bbox, gt_label,
                                match_dis_thresh=2.0):
        """Hungarian-match predicted to GT boxes by BEV centre distance.

        Ported from VAD (``VAD.assign_pred_to_gt_vip3d``) unchanged: the motion
        metrics are only comparable to the published numbers if the matching
        rule is identical. Static classes are excluded on both sides so a
        parked barrier cannot absorb a match.
        """
        dynamic_list = [0, 1, 3, 4, 6, 7, 8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(
            pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list
                        for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)
        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]
        return matched_bbox_result

    def compute_motion_metric_vip3d(self, gt_bbox, gt_label, gt_attr_label,
                                    pred_bbox, matched_bbox_result,
                                    mapped_class_names, match_dis_thresh=2.0):
        """Per-sample EPA / ADE / FDE / MR counters (VAD's ViP3D protocol).

        Returns raw counters, not ratios -- the dataset sums them over the
        whole split before dividing, which is what makes EPA well defined.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        metric_dict = {f'{met}_{cls}': 0.0
                       for met in motion_metric_names
                       for cls in motion_cls_names}

        veh_list = [0, 1, 3, 4]
        ignore_list = ['construction_vehicle', 'barrier', 'traffic_cone',
                       'motorcycle', 'bicycle']
        fut_ts = self.det_motion_head.fut_ts
        fut_mode = self.det_motion_head.fut_mode

        for i in range(pred_bbox['labels_3d'].shape[0]):
            if pred_bbox['labels_3d'][i] in veh_list:
                pred_bbox['labels_3d'][i] = 0
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_' + box_name] += 1

        for i in range(gt_label.shape[0]):
            if gt_label[i] in veh_list:
                gt_label[i] = 0
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][fut_ts * 2:fut_ts * 3]
            num_valid_ts = int(sum(gt_fut_masks == 1))
            if num_valid_ts == fut_ts:
                metric_dict['gt_' + box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_' + box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:fut_ts * 2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(
                    fut_mode, fut_ts, 2)[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2) \
                    + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2) \
                    + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(
                    gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                metric_dict['ADE_' + box_name] += float(
                    (dist.sum(-1) / num_valid_ts).min())
                if num_valid_ts == fut_ts:
                    fde = float(dist[:, -1].min())
                    metric_dict['cnt_fde_' + box_name] += 1
                    metric_dict['FDE_' + box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_' + box_name] += 1
                    else:
                        metric_dict['MR_' + box_name] += 1
        return metric_dict

    @staticmethod
    def map_pred2result(bboxes, scores, labels, pts):
        """Decoded vector-map predictions -> the keys the evaluator reads.

        ``VADCustomNuScenesDataset.output_to_vecs`` indexes exactly these four
        names, so they have to sit at the top level of the per-sample result
        dict rather than nested under a 'map' key. Mirrors VAD's
        ``VAD.map_pred2result``.
        """
        return dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

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
                det_out = self.det_motion_head.forward_test(
                    bev_embed, img_metas, rescale=rescale)
                if isinstance(det_out, list) and det_out and \
                        isinstance(det_out[0], (list, tuple)):
                    bbox_results[0].update(self.det_pred2result(*det_out[0]))
                else:
                    bbox_results[0]['det_motion'] = det_out
            if self.map_head is not None:
                map_out = self.map_head.forward_test(
                    bev_embed, img_metas, rescale=rescale)
                # A decoded vector head returns [[bboxes, scores, labels, pts]]
                # per sample; flatten it into the keys the dataset's
                # ``output_to_vecs`` reads so chamfer mAP can be computed.
                # Dense heads return a dict and are passed through untouched.
                if isinstance(map_out, list) and map_out and \
                        isinstance(map_out[0], (list, tuple)):
                    bbox_results[0].update(self.map_pred2result(*map_out[0]))
                else:
                    bbox_results[0]['map'] = map_out
            if self.occ_head is not None:
                occ_out = self.occ_head.forward_test(bev_embed)
                # float16, and no occ_seg: these are collected for all 6019 val
                # samples, and [1,7,C,100,100] float32 scores plus an int64
                # occ_seg comes to ~6.3 GiB. The binarised map is derivable from
                # the scores, and the evaluator sweeps thresholds anyway, so
                # only the probabilities are kept -- 0.79 GiB at float16.
                # Budget the GT alongside it: 0.39 GiB at uint8 (see above),
                # 1.18 GiB in total on rank 0 before collect_results_cpu.
                bbox_results[0]['occ_scores'] = occ_out['occ_scores'].half().cpu()

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

            # ViP3D motion counters, only when the detection+motion head ran.
            # Raw counts: the dataset sums them over the split before dividing,
            # which is what makes EPA well defined.
            if 'trajs_3d' in bbox_results[0]:
                # Work on a COPY. compute_motion_metric_vip3d collapses the
                # vehicle classes into `car` in place, so scoring motion on the
                # live result dict would rewrite truck/bus/trailer as car and
                # corrupt the detection evaluation that reads the same dict.
                # VAD deepcopies for exactly this reason (VAD.py:397).
                mot_result = copy.deepcopy(bbox_results[0])
                # VAD scores motion on confident detections only; without this
                # all 300 decoded queries enter the matching and the false
                # positive term of EPA is not comparable to the paper.
                keep = mot_result['scores_3d'] > self.motion_score_thresh
                for k in ('boxes_3d', 'scores_3d', 'labels_3d', 'trajs_3d'):
                    mot_result[k] = mot_result[k][keep]
                matched = self.assign_pred_to_gt_vip3d(
                    mot_result, gt_bbox, gt_labels_3d[0][0])
                metric_dict.update(self.compute_motion_metric_vip3d(
                    gt_bbox, gt_labels_3d[0][0].clone(), gt_attr_label,
                    mot_result, matched,
                    getattr(self, 'CLASSES', None) or NUSCENES_CLASSES))

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
