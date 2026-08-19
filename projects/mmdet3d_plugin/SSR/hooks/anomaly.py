"""Watch the training signal and write down anything that looks wrong.

A 60-epoch run is seven days. The failure modes that matter are not the ones
that crash -- those announce themselves -- but the ones that keep running:

* a loss goes NaN and every downstream number becomes meaningless;
* one head stops moving while the others keep going, and the loss total hides
  it because that head was a small part of it;
* the map head collapses so every query predicts the same polyline (which is
  what v1's dense head did for twelve epochs while its BCE fell 18%);
* the gradient balancer saturates at its clamp, so the shares the config asks
  for are not the shares being applied;
* an evaluation metric goes backwards and nobody notices until the run ends.

This hook checks for each of those and appends what it finds to
``anomalies.jsonl`` in the work dir, one JSON object per event, alongside a
logger warning and a ``anomaly/*`` counter that shows up in wandb and
TensorBoard as a time series.

It never raises and never stops training: a monitor that can kill a seven-day
run over a threshold it got wrong is worse than no monitor.

Register with a priority BELOW EvalHook (NORMAL=50) and ABOVE LoggerHook
(VERY_LOW=90) so that ``after_train_epoch`` sees the evaluation results before
the logger clears them::

    custom_hooks = [dict(type='TrainingAnomalyHook', priority='LOW')]
"""
import json
import os.path as osp
from collections import defaultdict, deque

import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


def _as_float(value):
    """float(value), or None if it is not a number.

    log_vars is whatever the model put there. A monitor that raises on an
    unexpected entry takes down the run it exists to protect.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# metric -> the direction that means "better". Used to call a regression.
_EVAL_DIRECTION = {
    'plan_L2_avg': 'lower',
    'plan_L2_stp3_avg': 'lower',
    'plan_obj_box_col_avg': 'lower',
    'plan_obj_box_col_stp3_avg': 'lower',
    'NuscMap_chamfer/mAP': 'higher',
    'pts_bbox_NuScenes/mAP': 'higher',
    'pts_bbox_NuScenes/NDS': 'higher',
    'EPA_car': 'higher',
    'EPA_pedestrian': 'higher',
    'occ_iou_thr0.5/mean': 'higher',
}


@HOOKS.register_module()
class TrainingAnomalyHook(Hook):
    """Flag NaNs, spikes, stalls, collapses and evaluation regressions.

    Args:
        out_file (str): appended to, under ``runner.work_dir``.
        spike_factor (float): flag an iteration whose total loss exceeds this
            many times the running median. 5x is loose enough that ordinary
            batch-to-batch variation does not trip it.
        window (int): how many recent iterations the median is taken over.
        stall_epochs (int): how many consecutive epochs a loss has to sit still
            before it is called stalled.
        stall_rel (float): "still" means the epoch mean moved less than this
            fraction. 0.005 is well below the drift a learning head shows.
        map_spread_min (float): metres. Below this the map queries have
            collapsed onto each other.
        conf_gap_min (float): ``map_conf_pos - map_conf_neg`` below this means
            the classifier is not separating matched from unmatched queries.
        eval_rel_tol (float): evaluation regressions smaller than this fraction
            are noise, not news.
        clamp (tuple | None): the GradBalancer clamp, for saturation checks.
            Read off the model when left as None.
    """

    def __init__(self,
                 out_file='anomalies.jsonl',
                 spike_factor=5.0,
                 window=200,
                 stall_epochs=3,
                 stall_rel=0.005,
                 map_spread_min=0.5,
                 conf_gap_min=0.02,
                 eval_rel_tol=0.02,
                 clamp=None):
        self.out_file = out_file
        self.spike_factor = spike_factor
        self.stall_epochs = stall_epochs
        self.stall_rel = stall_rel
        self.map_spread_min = map_spread_min
        self.conf_gap_min = conf_gap_min
        self.eval_rel_tol = eval_rel_tol
        self.clamp = clamp

        self._recent_loss = deque(maxlen=window)
        self._epoch = defaultdict(list)      # key -> values seen this epoch
        self._epoch_mean_hist = defaultdict(list)
        self._eval_hist = defaultdict(list)
        self._n_events = 0
        self._path = None

    # ------------------------------------------------------------------ #
    def before_run(self, runner):
        self._path = osp.join(runner.work_dir, self.out_file)
        if self.clamp is None:
            model = getattr(runner.model, 'module', runner.model)
            bal = getattr(model, 'grad_balancer', None)
            if bal is not None:
                self.clamp = bal.clamp

    @master_only
    def _record(self, runner, kind, severity, detail):
        self._n_events += 1
        event = dict(kind=kind, severity=severity,
                     epoch=runner.epoch + 1, iter=runner.iter + 1, **detail)
        try:
            with open(self._path, 'a') as f:
                f.write(json.dumps(event, default=float) + '\n')
        except OSError:
            pass
        line = f'[anomaly/{severity}] {kind}: ' + \
            ' '.join(f'{k}={v}' for k, v in detail.items())
        (runner.logger.error if severity == 'critical'
         else runner.logger.warning)(line)

    # ------------------------------------------------------------------ #
    def after_train_iter(self, runner):
        log_vars = (runner.outputs or {}).get('log_vars') or {}
        n_before = self._n_events
        n_bad = 0

        for key, val in log_vars.items():
            v = _as_float(val)
            if v is None:
                continue
            if not np.isfinite(v):
                n_bad += 1
                # Only the first non-finite key per iteration is worth writing;
                # once one loss is NaN the sum and everything after it follow.
                if n_bad == 1:
                    self._record(runner, 'non_finite', 'critical',
                                 dict(key=key, value=str(v)))
            else:
                self._epoch[key].append(v)

        loss = _as_float(log_vars.get('loss'))
        if loss is not None and np.isfinite(loss):
            if len(self._recent_loss) == self._recent_loss.maxlen:
                med = float(np.median(self._recent_loss))
                if med > 0 and loss > self.spike_factor * med:
                    self._record(runner, 'loss_spike', 'warning',
                                 dict(loss=loss, median=med,
                                      factor=loss / med))
            self._recent_loss.append(loss)

        gn = _as_float(log_vars.get('grad_norm'))
        if gn is not None:
            if not np.isfinite(gn):
                self._record(runner, 'grad_norm_non_finite', 'critical',
                             dict(grad_norm=str(gn)))
            elif gn == 0.0:
                self._record(runner, 'grad_norm_zero', 'warning',
                             dict(note='no gradient reached the parameters'))

        # a time series, so the spikes are visible in wandb next to the losses
        runner.log_buffer.update(
            {'anomaly/new': float(self._n_events - n_before),
             'anomaly/total': float(self._n_events)},
            runner.outputs.get('num_samples', 1))

    # ------------------------------------------------------------------ #
    def after_train_epoch(self, runner):
        means = {k: float(np.mean(v)) for k, v in self._epoch.items() if v}
        self._check_stalls(runner, means)
        self._check_heads(runner, means)
        self._check_balancer(runner, means)
        self._check_eval(runner)
        for k, v in means.items():
            self._epoch_mean_hist[k].append(v)
        self._epoch.clear()

    def _check_stalls(self, runner, means):
        """A loss that has not moved for several epochs is not training."""
        for key, cur in means.items():
            if 'loss' not in key:
                continue
            hist = self._epoch_mean_hist[key]
            if len(hist) < self.stall_epochs:
                continue
            recent = hist[-(self.stall_epochs - 1):] + [cur]
            base = abs(recent[0])
            if base < 1e-8:
                continue        # a term that is genuinely zero (weight 0.0)
            if max(abs(x - recent[0]) for x in recent) / base < self.stall_rel:
                self._record(runner, 'loss_stalled', 'warning',
                             dict(key=key, epochs=self.stall_epochs,
                                  value=cur,
                                  note='epoch mean flat; head may not be '
                                       'learning'))

    def _check_heads(self, runner, means):
        spread = means.get('map_spread_m')
        if spread is not None and spread < self.map_spread_min:
            self._record(runner, 'map_query_collapse', 'critical',
                         dict(map_spread_m=spread,
                              threshold=self.map_spread_min,
                              note='every query predicts the same polyline'))
        pos, neg = means.get('map_conf_pos'), means.get('map_conf_neg')
        if pos is not None and neg is not None:
            gap = pos - neg
            # only meaningful once the head has had a chance to separate them
            # >= 1 means "at least one full epoch has already been logged",
            # i.e. this is epoch 2 or later. _epoch_mean_hist is appended after
            # these checks run, so it lags the current epoch by one.
            if len(self._epoch_mean_hist['map_conf_pos']) >= 1 and \
                    gap < self.conf_gap_min:
                self._record(runner, 'map_no_discrimination', 'warning',
                             dict(conf_pos=pos, conf_neg=neg, gap=gap,
                                  note='matched and unmatched queries score '
                                       'the same'))
        rate = means.get('clip/rate')
        if rate is not None and rate >= 0.999:
            self._record(runner, 'clipping_every_iteration', 'warning',
                         dict(clip_rate=rate,
                              clip_factor=means.get('clip/factor'),
                              note='max_norm is binding on every step'))

    def _check_balancer(self, runner, means):
        if not self.clamp:
            return
        lo, hi = self.clamp
        for key, val in means.items():
            if not key.startswith('gscale/'):
                continue
            if val <= lo * 1.01 or val >= hi * 0.99:
                self._record(runner, 'grad_balance_saturated', 'warning',
                             dict(key=key, value=val, clamp=[lo, hi],
                                  note='the clamp, not the target, is setting '
                                       'this share'))

    def _check_eval(self, runner):
        """Evaluation results, if an EvalHook just wrote any."""
        out = getattr(runner.log_buffer, 'output', {}) or {}
        seen = False
        for key, direction in _EVAL_DIRECTION.items():
            if key not in out:
                continue
            seen = True
            try:
                cur = float(out[key])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(cur):
                self._record(runner, 'eval_non_finite', 'critical',
                             dict(key=key, value=str(cur)))
                continue
            hist = self._eval_hist[key]
            if hist:
                prev = hist[-1]
                worse = (cur > prev * (1 + self.eval_rel_tol)
                         if direction == 'lower'
                         else cur < prev * (1 - self.eval_rel_tol))
                if worse:
                    self._record(runner, 'eval_regression', 'warning',
                                 dict(key=key, previous=prev, current=cur,
                                      better=direction))
            hist.append(cur)
        if seen:
            # aux heads that are still at exactly zero after an evaluation have
            # not learned anything scoreable yet
            for key in ('NuscMap_chamfer/mAP', 'pts_bbox_NuScenes/mAP'):
                if key in out and float(out[key]) == 0.0 and \
                        len(self._eval_hist[key]) >= 2:
                    self._record(runner, 'aux_head_scores_zero', 'critical',
                                 dict(key=key, evaluations=len(
                                     self._eval_hist[key])))

    def after_run(self, runner):
        if self._n_events:
            runner.logger.warning(
                f'{self._n_events} anomal{"y" if self._n_events == 1 else "ies"}'
                f' recorded in {self._path}')
