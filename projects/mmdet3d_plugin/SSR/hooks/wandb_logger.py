"""A WandbLoggerHook that keeps a fixed set of scalars off the wandb dashboard.

Some loss components carry the substring ``loss`` so ``_parse_losses`` folds
them into the optimised total and logs them, but on wandb they are just noise
-- e.g. the vectorised map head's per-decoder-layer bbox/iou terms. This hook
drops those tags right before the upload. It changes NOTHING about training and
does NOT touch the TextLoggerHook / TensorboardLoggerHook streams, so the values
are still optimised and still visible in the text log for debugging; they are
only withheld from wandb.

The default excludes the six map bbox/iou terms; override ``exclude`` in the
config to change the set. Tags arrive mode-prefixed (``train/<key>``), so
matching is done against the tag both with and without that prefix -- both
``'map.loss_map_iou'`` and ``'train/map.loss_map_iou'`` work.

W&B is telemetry, not part of the optimisation.  A W&B SDK/service failure
must therefore never take a distributed training rank down with it.  By
default this hook is fail-open: the first W&B exception is recorded in the
text log, W&B is disabled for the rest of the process, and training,
TensorBoard and checkpointing continue normally.  Set ``non_fatal=False`` only
when deliberately debugging W&B itself.
"""
from mmcv.runner import HOOKS, WandbLoggerHook
from mmcv.runner.dist_utils import master_only

# map.<layer>.loss_map_{bbox,iou}: final layer + the two aux decoder layers.
DEFAULT_EXCLUDE = (
    'map.loss_map_bbox', 'map.loss_map_iou',
    'map.d0.loss_map_bbox', 'map.d0.loss_map_iou',
    'map.d1.loss_map_bbox', 'map.d1.loss_map_iou',
)


@HOOKS.register_module()
class SSRWandbLoggerHook(WandbLoggerHook):

    def __init__(self, *args, exclude=DEFAULT_EXCLUDE, non_fatal=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude = set(exclude or ())
        self.non_fatal = non_fatal
        self._wandb_disabled = False

    def _handle_failure(self, runner, operation, exc):
        """Disable broken telemetry without hiding the failure.

        Only rank 0 enters the guarded methods below.  Swallowing the
        exception there is essential: if rank 0 leaves an epoch hook while the
        other ranks continue, torch.distributed tears the whole job down.
        """
        if not self.non_fatal:
            raise exc
        if self._wandb_disabled:
            return
        self._wandb_disabled = True
        runner.logger.error(
            'W&B %s failed (%s: %s). Disabling W&B for the rest of this '
            'process; training, TextLogger, TensorBoard and checkpoints will '
            'continue.',
            operation, type(exc).__name__, exc,
            exc_info=(type(exc), exc, exc.__traceback__))

    def _drop(self, tag):
        # get_loggable_tags mode-prefixes keys, e.g. 'train/map.d0.loss_map_iou'
        return tag in self.exclude or tag.split('/', 1)[-1] in self.exclude

    @master_only
    def before_run(self, runner):
        try:
            super().before_run(runner)
        except Exception as exc:
            self._handle_failure(runner, 'initialisation', exc)

    @master_only
    def log(self, runner):
        if self._wandb_disabled:
            return
        try:
            tags = self.get_loggable_tags(runner)
            tags = {k: v for k, v in tags.items() if not self._drop(k)}
            if tags:
                if self.with_step:
                    self.wandb.log(
                        tags, step=self.get_iter(runner), commit=self.commit)
                else:
                    tags['global_step'] = self.get_iter(runner)
                    self.wandb.log(tags, commit=self.commit)
        except Exception as exc:
            self._handle_failure(runner, 'logging', exc)

    @master_only
    def after_run(self, runner):
        if self._wandb_disabled:
            return
        try:
            self.wandb.join()
        except Exception as exc:
            self._handle_failure(runner, 'shutdown', exc)
