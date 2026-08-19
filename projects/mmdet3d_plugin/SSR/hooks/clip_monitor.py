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
# `pts_bbox_head` is NOT the planner. It holds the BEV encoder as well
# (transformer / bev_embedding / positional_encoding, 4.65M parameters), and
# that encoder receives gradient from every task -- so lumping it under `plan`
# reported "BEV encoder + planner" while calling it planning. Split.
_DEFAULT_GROUPS = dict(
    trunk=['img_backbone', 'img_neck'],
    bev=['pts_bbox_head.transformer', 'pts_bbox_head.bev_embedding',
         'pts_bbox_head.positional_encoding'],
    plan=['pts_bbox_head'],          # whatever is left of it: the planner
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

    def _group_of(self, pname):
        """Which group a parameter belongs to. LONGEST prefix wins.

        Not first-match-wins, and the difference is not academic. `plan` is
        `pts_bbox_head`, which is a prefix of every `bev` entry
        (`pts_bbox_head.transformer`, ...), so a first-match rule puts the whole
        BEV encoder into `plan` unless `bev` happens to be earlier in the dict.
        That is exactly the mislabelling this split was introduced to fix, and
        leaving it dependent on dict ordering means one reordering brings it
        back silently. Longest-prefix is order-independent: the more specific
        rule wins because it is more specific.
        """
        best, best_len = 'other', -1
        for gname, prefixes in self.groups.items():
            for pre in prefixes:
                if pname.startswith(pre) and len(pre) > best_len:
                    best, best_len = gname, len(pre)
        return best

    def _group_norms(self, runner):
        """||dL/dW|| per module group: the L2 norm over each group's gradients,
        i.e. sqrt of the summed squared per-parameter norms."""
        model = getattr(runner.model, 'module', runner.model)
        sq = {name: 0.0 for name in self.groups}
        sq['other'] = 0.0
        for pname, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            sq[self._group_of(pname)] += float(p.grad.detach().norm()) ** 2
        out = {f'pnorm/{k}': v ** 0.5 for k, v in sq.items()}
        # The whole point of this decomposition is to say where the clipped
        # norm comes from, so it has to actually reconstruct that norm:
        #
        #     grad_norm^2 == sum_g pnorm_g^2
        #
        # If it does not, some parameter is double-counted or missing and every
        # conclusion drawn from the split is void. `pnorm/total` is logged next
        # to `grad_norm` so the two can be read against each other, and
        # `pnorm/other` is the catch-all that must stay at zero.
        out['pnorm/total'] = sum(sq.values()) ** 0.5
        return out

    def _snapshot(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        return {n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None}

    def _update_ratios(self, runner, before):
        """||dw|| / ||w|| per module group, AFTER the optimiser step.

        The gradient norms above say what the loss asked for; this says what
        the optimiser actually did, and under AdamW those are not proportional.
        Adam divides by a running second moment, so a parameter whose gradient
        is a hundred times another's can still take a similar-sized step -- the
        measurement that pnorm/aux is 12x pnorm/plan therefore says nothing
        about whether the aux heads are being updated 12x harder.

        Read it as: ~1e-3 is the healthy band, below ~1e-5 that group is
        effectively frozen no matter what its gradient looks like, above ~1e-2
        it is taking steps large enough to be unstable.
        """
        model = getattr(runner.model, 'module', runner.model)
        dw = {name: 0.0 for name in self.groups}
        w = {name: 0.0 for name in self.groups}
        dw['other'] = w['other'] = 0.0
        for pname, p in model.named_parameters():
            old = before.get(pname)
            if old is None:
                continue
            gname = self._group_of(pname)
            dw[gname] += float((p.detach() - old).norm()) ** 2
            w[gname] += float(old.norm()) ** 2
        out = {}
        for name in dw:
            if w[name] > 0:
                out[f'uwr/{name}'] = (dw[name] ** 0.5) / (w[name] ** 0.5)
        return out

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        log = {}
        # every rank hits the same iterations; no collectives are involved
        measure = (self.group_norm_interval > 0 and
                   (runner.iter + 1) % self.group_norm_interval == 0)
        before = None
        if measure:
            log.update(self._group_norms(runner))
            # A full parameter copy -- 38M floats, ~152 MB, transient and only
            # on measurement iterations. Against an 18.4 GB training peak that
            # is 0.8%, and it is the only way to get ||dw||: the difference of
            # two norms is not the norm of the difference.
            before = self._snapshot(runner)

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                gn = float(grad_norm)                  # pre-clip total norm
                max_norm = float(self.grad_clip['max_norm'])
                log['grad_norm'] = gn
                log['clip/rate'] = float(gn > max_norm)
                log['clip/factor'] = min(1.0, max_norm / max(gn, 1e-12))

        runner.optimizer.step()
        if before is not None:
            log.update(self._update_ratios(runner, before))
            before.clear()
        if log:
            runner.log_buffer.update(log, runner.outputs['num_samples'])
