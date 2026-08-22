"""CPU regression checks for the fail-open W&B logger.

No W&B run or network connection is created.  A small fake exercises metric
filtering and the exact SDK exception path that ended the epoch-6 run.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

from projects.mmdet3d_plugin.SSR.hooks.wandb_logger import SSRWandbLoggerHook


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append(message % args)


class _Wandb:
    def __init__(self, log_error=None, join_error=None):
        self.log_error = log_error
        self.join_error = join_error
        self.logged = []
        self.join_calls = 0

    def log(self, tags, **kwargs):
        if self.log_error is not None:
            raise self.log_error
        self.logged.append((dict(tags), dict(kwargs)))

    def join(self):
        self.join_calls += 1
        if self.join_error is not None:
            raise self.join_error


def _hook(fake_wandb, non_fatal=True):
    hook = SSRWandbLoggerHook(non_fatal=non_fatal)
    hook.wandb = fake_wandb
    hook.get_iter = lambda runner: 123
    hook.get_loggable_tags = lambda runner: {
        'train/loss': 1.5,
        'train/map.loss_map_iou': 0.0,
    }
    return hook


runner = SimpleNamespace(logger=_Logger())

# Existing filtering behaviour remains intact.
ok_wandb = _Wandb()
ok = _hook(ok_wandb)
ok.log(runner)
assert ok_wandb.logged == [(
    {'train/loss': 1.5}, {'step': 123, 'commit': True})]

# The epoch-6 failure: AssertionError from wandb.log must disable W&B and
# return normally.  Later hooks must not touch the broken service again.
bad_wandb = _Wandb(log_error=AssertionError('asyncio drain failed'))
bad = _hook(bad_wandb)
bad.log(runner)
assert bad._wandb_disabled
assert len(runner.logger.errors) == 1
bad.log(runner)
bad.after_run(runner)
assert bad_wandb.join_calls == 0
assert len(runner.logger.errors) == 1

# Shutdown is telemetry too and must be non-fatal.
join_wandb = _Wandb(join_error=ConnectionResetError('socket closed'))
join = _hook(join_wandb)
join.after_run(runner)
assert join._wandb_disabled
assert join_wandb.join_calls == 1

# A strict opt-in remains available for diagnosing the SDK itself.
strict = _hook(_Wandb(log_error=RuntimeError('strict')), non_fatal=False)
try:
    strict.log(runner)
except RuntimeError as exc:
    assert str(exc) == 'strict'
else:
    raise AssertionError('non_fatal=False did not propagate the W&B error')

print('W&B logger filtering + fail-open checks: ok')
