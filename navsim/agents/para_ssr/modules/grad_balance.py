"""Closed-loop control of each task's gradient share at the shared BEV feature.

``aux_grad_scale`` sets a KNOB: one constant on the gradient every auxiliary
head pushes back into ``bev_embed``. What share of that gradient each task ends
up with is then whatever it happens to be. Measured on a real run at 0.01, the
planner still only got 4.1% -- the setting did not do what its name suggests,
and there was no way to know without measuring afterwards.

This sets an OUTCOME instead. Given

    target = dict(plan=0.4, det=0.2, map=0.2, occ=0.2)

it periodically measures ``g_k = ||dL_k/d bev_embed||`` for every task, backs
out what that would have been without the current scaling, and solves for the
per-task scales that put the shares where they were asked to be. Taking
planning as the numeraire (its path is never scaled, so ``s_plan = 1``):

    g_k_raw = g_k_measured / s_k
    s_k     = (t_k / t_plan) * (g_plan_raw / g_k_raw)

which gives share_k = t_k / sum(t) by construction, for every k including plan.

WHAT `target` ACTUALLY MEANS, stated precisely because the number invites a
reading it does not support. `plan=0.4` means:

    the rank-averaged share of ||dL_task / d bev_embed||,
    measured at the BEV ACTIVATION, on the measurement iterations.

It is NOT any of these, and the gap is not small:

  * the share of the parameter-gradient update after DDP averaging. Gradients
    are all-reduced across ranks BEFORE the optimiser sees them; the balancer
    averages per-rank NORMS, which is a different operation -- mean of norms is
    not the norm of the mean, and they coincide only if the ranks agree.
  * the share of the actual optimiser step. AdamW divides by a running second
    moment, so a task holding 40% of the gradient does not get 40% of the
    movement. That is what uwr/* measures, and it is not this.
  * contribution accounting for direction. Two tasks at 30% each can reinforce
    or cancel; see gcos/* and the first bullet below.

Read `target` as a knob on one specific, measurable quantity -- not as "task X
gets 40% of the learning".

Three things this does NOT do, all of which matter:

* Equal magnitude is not equal influence. ``sum||g_k|| != ||sum g_k||`` -- two
  tasks pointing in opposite directions cancel in the actual update while both
  count fully here. Balancing magnitudes leaves conflict untouched; that needs
  a direction-based method (PCGrad and friends).
* It amplifies whatever is small. A task that has converged has a small
  gradient and gets scaled UP, which promotes its noise. Hence ``clamp``.
* It only touches the shared-BEV path. The heads' own parameters keep receiving
  unscaled gradients, so head convergence is unaffected -- which is the whole
  point when the heads are meant to become distillation teachers.

Scales are assigned by the originating loss, not by the decoder traversed on
the way back to BEV. Planning now reads detection/motion and map latents, so a
valve at either decoder's BEV input would incorrectly scale planning too.
``balance_shared_gradients`` adds a loss-local correction directly at BEV;
the full planning gradient (direct and via either decoder) stays unscaled.
Detection and motion retain their shared task coefficient. Their losses are
added before measuring: separate norms miss reinforcement or cancellation.

This correction requires FP32 training (FP64 is also supported for numerical
tests). In FP16/BF16, combining large auxiliary and small planning gradients
inside the decoders loses information before the auxiliary VJP is subtracted.
Casting the final loss or BEV boundary to FP32 cannot recover that information.
Unbalanced mixed-precision training and mixed-precision inference are separate
from this controller and do not need the correction.
"""
import math

import torch


def _autocast_enabled(device):
    # torch 2.0 (still supported by requirements_navsim.txt) has separate
    # no-argument CUDA and CPU queries; newer torch accepts a device string.
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:
        if device == "cpu":
            return torch.is_autocast_cpu_enabled()
        return device == "cuda" and torch.is_autocast_enabled()


def require_full_precision_for_balancing(tensors):
    """Reject low-precision forward graphs, even outside an autocast context.

    Callers with predictions must pass the original forward outputs: loss
    routines may already have promoted them, and BEV itself can remain FP32
    while the private decoders ran in BF16. A scalar AMP loss multiplier is
    supported by ``_AddSharedGradient``; low-precision decoder arithmetic is
    not.
    """
    tensors = tuple(value for value in tensors if isinstance(value, torch.Tensor))
    reduced = any(value.dtype in (torch.float16, torch.bfloat16) for value in tensors)
    devices = {value.device.type for value in tensors}
    autocast = any(_autocast_enabled(device) for device in devices)
    if reduced or autocast:
        raise RuntimeError(
            "PARA-SSR shared-gradient balancing requires FP32 forward/backward; "
            "FP16/BF16 cancellation can corrupt the planning gradient. Set "
            "trainer.params.precision=32, or disable balancing with "
            "agent.config.grad_balance_target=null (grad_balance_target=None "
            "in Python) and leave manual aux_grad_scale coefficients at 1.0. "
            "Casting only losses or BEV to FP32 is insufficient."
        )


class _AddSharedGradient(torch.autograd.Function):
    """Keep a loss value and private gradients; correct only its BEV gradient.

    The upstream scalar multiplies BOTH outputs in backward. This is essential
    for Lightning's gradient accumulation and AMP loss scaling. A hook holding
    a fixed correction would silently skip those multipliers. Every invocation
    owns its correction, so several outstanding microbatch graphs can coexist.
    """

    @staticmethod
    def forward(ctx, loss, shared_feature, correction):
        ctx.save_for_backward(correction)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (correction,) = ctx.saved_tensors
        return grad_output, correction * grad_output, None


def balance_shared_gradients(
    total_loss, shared_feature, task_losses, scales, *, measure_norms=False
):
    """Scale task-origin gradients at BEV without scaling private parameters.

    ``task_losses`` contains already weighted losses, with detection and motion
    combined under ``det``. Forward paths must receive the original BEV tensor,
    without branch-level gradient valves. For each auxiliary loss ``L_k`` add
    ``(s_k - 1) * dL_k/dBEV`` at BEV on the final backward; all decoder and
    planner parameters continue to receive their ordinary loss gradients.

    Measurements include every route from a loss to BEV. They report effective
    norms (raw norm times the current task scale), matching ``GradBalancer``'s
    update contract. VJPs stop at the activation and do not populate parameter
    ``.grad`` or invoke DDP parameter reduction. DDP still averages the eventual
    corrected encoder gradients through its ordinary backward hooks.

    Only tasks requiring a correction or a norm measurement incur an extra
    reverse traversal. The detached correction supports ordinary first-order
    training, not higher-order differentiation through gradient balancing.
    """
    norms = {}
    correction = None
    if not torch.is_grad_enabled() or not shared_feature.requires_grad:
        return total_loss, norms
    if any(task != "plan" and float(scales.get(task, 1.0)) != 1.0
           for task in task_losses):
        require_full_precision_for_balancing(
            (shared_feature, total_loss, *task_losses.values())
        )
    for task, loss in task_losses.items():
        # Plan is the numeraire, regardless of the names of branches it uses.
        scale = 1.0 if task == "plan" else float(scales.get(task, 1.0))
        if not measure_norms and scale == 1.0:
            continue
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            continue
        grad = torch.autograd.grad(
            loss, shared_feature, retain_graph=True, allow_unused=True
        )[0]
        if measure_norms:
            norms[task] = 0.0 if grad is None else float(grad.float().norm()) * scale
        if grad is not None and scale != 1.0:
            delta = grad.detach() * (scale - 1.0)
            correction = delta if correction is None else correction + delta
    if correction is not None:
        total_loss = _AddSharedGradient.apply(total_loss, shared_feature, correction)
    return total_loss, norms


class GradBalancer:
    """Per-task scales on the shared-BEV gradient, re-solved periodically.

    Args:
        target (dict[str, float]): desired gradient share per task. Must
            include ``plan`` (the numeraire) with a non-zero value. Values are
            normalised, so ``dict(plan=2, det=1)`` means 2:1.
        interval (int): re-measure and re-solve every N iterations. The ratios
            drift on an epoch timescale, not an iteration one, so this can be
            coarse -- and each update costs one backward per task through the
            heads (not the backbone).
        momentum (float): EMA on the scales. Damps the batch-to-batch jitter in
            the measurement. Two details matter and both were found by watching
            a real run rather than a unit test:

            * the smoothing is GEOMETRIC, ``s <- s^m * s_new^(1-m)``. These
              scales span four orders of magnitude, so averaging them
              arithmetically makes the descent from the initial 1.0 follow
              ``0.9^n`` -- about 70 updates, or four epochs at interval 200, to
              reach 5e-4. Four epochs of the unbalanced regime is most of what
              the controller exists to prevent.
            * the FIRST measurement per task is adopted outright, with no
              smoothing. 1.0 is an arbitrary starting value, not an estimate;
              there is nothing to average it with.
        clamp (tuple[float, float]): bounds on every scale. The lower bound
            stops a dominant task from being silenced entirely; the upper bound
            stops a converged task from having its noise amplified.
        warmup_iters (int): hold every scale at 1.0 for this many iterations.
            Gradients during LR warm-up are not representative -- measured at
            iteration 10 the aux/plan ratio was two orders of magnitude away
            from where it settled by epoch 1.
    """

    NUMERAIRE = 'plan'

    def __init__(self, target, interval=200, momentum=0.9,
                 clamp=(1e-4, 1.0), warmup_iters=500):
        if self.NUMERAIRE not in target or target[self.NUMERAIRE] <= 0:
            raise ValueError(
                f'target must give a non-zero share to {self.NUMERAIRE!r}')
        if any(not math.isfinite(float(v)) or float(v) < 0
               for v in target.values()):
            raise ValueError('target shares must be finite and non-negative')
        total = float(sum(target.values()))
        if not math.isfinite(total) or total <= 0:
            raise ValueError('target shares must have a finite positive sum')
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError(f'interval must be a positive integer, got {interval!r}')
        if not math.isfinite(float(momentum)) or not 0 <= momentum < 1:
            raise ValueError(f'momentum must be in [0, 1), got {momentum!r}')
        if len(clamp) != 2:
            raise ValueError(f'clamp must contain (min, max), got {clamp!r}')
        lo, hi = (float(clamp[0]), float(clamp[1]))
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            raise ValueError(
                f'clamp must be finite and satisfy 0 < min <= max, got {clamp!r}')
        if not isinstance(warmup_iters, int) or warmup_iters < 0:
            raise ValueError(
                f'warmup_iters must be a non-negative integer, got {warmup_iters!r}')
        self.target = {k: float(v) / total for k, v in target.items()}
        self.interval = interval
        self.momentum = float(momentum)
        self.clamp = (lo, hi)
        self.warmup_iters = warmup_iters
        # plan is never scaled; the others start neutral so warm-up runs at the
        # unmodified objective
        self.scale = {k: 1.0 for k in self.target if k != self.NUMERAIRE}
        self._seen = set()          # tasks whose first measurement has landed
        # A target of exactly 0 means OFF, not "very small": the task keeps
        # training its own parameters but contributes nothing to the shared BEV.
        # That is the plan=1.0 control -- SSR's original situation, in which
        # planning necessarily owned 100% of the BEV gradient because no other
        # head existed. Running it as a PARA-SSR arm rather than as SSR-orig
        # holds the parameter count, the optimiser state and (crucially) the
        # global clip norm constant, so only the one variable moves.
        # Set before any measurement, and never revisited: `update` skips
        # tasks whose scale is already 0.
        for k, t in self.target.items():
            if t == 0 and k in self.scale:
                self.scale[k] = 0.0
                self._seen.add(k)

    def scale_for(self, task):
        return self.scale.get(task, 1.0)

    def should_update(self, it):
        return it > self.warmup_iters and it % self.interval == 0

    def update(self, measured):
        """Re-solve the scales from freshly measured gradient norms.

        Args:
            measured (dict[str, float]): ``||dL_k/d bev_embed||`` as measured
                WITH the current scales applied. Tasks absent from ``target``
                are ignored; tasks absent here keep their current scale.
        Returns:
            dict[str, float]: ``gscale/{task}`` for logging.
        """
        raw = {}
        for task, t in self.target.items():
            if task not in measured:
                continue
            s = self.scale_for(task)
            if s <= 0:
                continue          # switched off by a zero target; leave it off
            norm = float(measured[task])
            # One invalid measurement must not poison checkpointed controller
            # state forever. Keep the complete previous update and try again
            # at the next measurement, after the distributed norm reduction.
            if not math.isfinite(norm) or norm < 0:
                return self.log_dict()
            unscaled = norm / s
            if not math.isfinite(unscaled):
                return self.log_dict()
            raw[task] = unscaled        # undo the valve
        base = raw.get(self.NUMERAIRE)
        t_base = self.target[self.NUMERAIRE]
        if not base or base <= 0:
            # planning contributed nothing measurable this batch; leave the
            # scales alone rather than divide by ~0
            return self.log_dict()

        lo, hi = self.clamp
        for task, g in raw.items():
            if task == self.NUMERAIRE or g <= 0:
                continue
            want = (self.target[task] / t_base) * (base / g)
            want = min(max(want, lo), hi)
            if task not in self._seen:
                self.scale[task] = want        # adopt the first estimate as-is
                self._seen.add(task)
                continue
            m = self.momentum
            self.scale[task] = math.exp(
                m * math.log(self.scale[task]) + (1.0 - m) * math.log(want))
        return self.log_dict()

    def log_dict(self):
        return {f'gscale/{k}': v for k, v in self.scale.items()}


def all_reduce_mean(values, device):
    """Average a {name: float} across ranks so every rank solves identically.

    Without this each rank measures its own batch, solves for its own scales,
    and the model silently optimises a different objective on each GPU.
    """
    if not torch.distributed.is_available() or \
            not torch.distributed.is_initialized():
        return dict(values)
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    t /= torch.distributed.get_world_size()
    return {k: float(v) for k, v in zip(keys, t.tolist())}
