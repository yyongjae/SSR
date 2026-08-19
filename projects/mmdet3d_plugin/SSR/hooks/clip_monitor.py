"""OptimizerHook that reports what global gradient clipping is actually doing.

``optimizer_config = dict(grad_clip=dict(max_norm=35, ...))`` clips over EVERY
parameter at once, so once the auxiliary heads are attached their gradients
enter the same norm as the planner's and the trunk's. If the clip fires, every
parameter -- planner included -- is scaled by the same factor.

That is not hypothetical here. Reading the logged ``grad_norm`` back out of the
two finished runs:

===========================  ==============  ==========
run                          mean grad_norm  fraction > 35
===========================  ==============  ==========
ssr_noffp_2gpu_b4 (plan only)          0.93         0.0%
para_ssr_8gpu     (v1, aux)           34.07        56.2%
===========================  ==============  ==========

and the PARA run is at 100% from epoch 10 onward. Note that ``aux_grad_scale``
does nothing about this: ``_ScaleGrad`` only touches the gradient flowing into
``bev_embed``, while the norm above is dominated by the aux heads' own
parameter gradients, which it leaves untouched.

The practical size of the effect is small -- a mean norm of 38.5 against
max_norm 35 is a uniform 0.91x, the update direction is unchanged, and AdamW's
``m / sqrt(v)`` absorbs most of a constant rescale. But "small" was an
assumption until it was measured, so this hook measures it:

``clip/rate``     fraction of iterations in the logging window that clipped.
                  Updated every iteration, so the window mean is the true rate
                  (unlike ``gshare/*``, which is only written every
                  ``grad_norm_log_interval`` steps -- see tools/analyze_gshare.py).
``clip/factor``   mean ``min(1, max_norm / grad_norm)`` actually applied.
``pnorm/*``       optional pre-clip parameter-gradient norm per module group,
                  which is what says whether raising ``max_norm`` is safe.
"""
from mmcv.runner.hooks import HOOKS, OptimizerHook

# Defaults live here, not in the signature: mmcv Configs are merged in place and
# a mutable default shared between two built hooks would be a live wire.
_DEFAULT_GROUPS = dict(
    trunk=['img_backbone', 'img_neck'],
    plan=['pts_bbox_head'],
    aux=['det_motion_head', 'map_head', 'occ_head'],
)


@HOOKS.register_module()
class ClipMonitorOptimizerHook(OptimizerHook):
    """``OptimizerHook`` plus clipping telemetry. Optimisation is unchanged.

    Args:
        grad_clip (dict | None): as ``OptimizerHook``.
        group_norm_interval (int): log per-group gradient norms every N
            iterations. 0 disables. This walks every parameter, so keep it
            coarse (the aux metrics use 200).
        groups (dict[str, list[str]] | None): group name -> parameter name
            prefixes. Anything unmatched lands in ``pnorm/other``.
    """

    def __init__(self, grad_clip=None, group_norm_interval=0, groups=None):
        super().__init__(grad_clip=grad_clip)
        self.group_norm_interval = group_norm_interval
        self.groups = dict(_DEFAULT_GROUPS if groups is None else groups)

    def _group_norms(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        sq = {name: 0.0 for name in self.groups}
        sq['other'] = 0.0
        for pname, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            v = float(p.grad.detach().norm()) ** 2
            for gname, prefixes in self.groups.items():
                if pname.startswith(tuple(prefixes)):
                    sq[gname] += v
                    break
            else:
                sq['other'] += v
        return {f'pnorm/{k}': v ** 0.5 for k, v in sq.items()}

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        log = {}
        # every rank hits the same iterations; no collectives are involved
        if self.group_norm_interval > 0 and \
                (runner.iter + 1) % self.group_norm_interval == 0:
            log.update(self._group_norms(runner))

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                gn = float(grad_norm)                  # pre-clip total norm
                max_norm = float(self.grad_clip['max_norm'])
                log['grad_norm'] = gn
                log['clip/rate'] = float(gn > max_norm)
                log['clip/factor'] = min(1.0, max_norm / max(gn, 1e-12))

        if log:
            runner.log_buffer.update(log, runner.outputs['num_samples'])
        runner.optimizer.step()
