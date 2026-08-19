"""Does TrainingAnomalyHook actually fire on the things it claims to catch?

A monitor nobody tested is worse than none: it produces silence, and silence
reads as "everything is fine". So feed it each failure mode explicitly and
check that the corresponding event lands in the file.

Exits non-zero on failure.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.SSR.hooks.anomaly import (  # noqa: E402
    TrainingAnomalyHook)


class _Buf:
    def __init__(self):
        self.output = {}

    def update(self, d, n=1):
        pass


class _Log:
    def __init__(self):
        self.lines = []

    def warning(self, m):
        self.lines.append(('warning', m))

    def error(self, m):
        self.lines.append(('error', m))

    info = warning


class _Runner:
    """The three attributes the hook touches, and nothing else."""

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.logger = _Log()
        self.log_buffer = _Buf()
        self.model = object()
        self.epoch = 0
        self.iter = 0
        self.outputs = {}

    def step(self, log_vars):
        self.outputs = dict(log_vars=log_vars, num_samples=8)
        self.iter += 1


def events(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def main():
    tmp = tempfile.mkdtemp()
    fails = []
    try:
        path = os.path.join(tmp, 'anomalies.jsonl')

        # ---------------------------------------------------------- NaN ----
        h = TrainingAnomalyHook(window=10, clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        for _ in range(5):
            r.step({'loss': 3.0, 'grad_norm': 10.0})
            h.after_train_iter(r)
        r.step({'loss': float('nan'), 'map.loss_map_pts': 2.0})
        h.after_train_iter(r)
        kinds = [e['kind'] for e in events(path)]
        print(f'  NaN loss            -> {kinds}')
        fails += [] if 'non_finite' in kinds else ['nan']

        # ------------------------------------------------------- spike ----
        os.remove(path)
        h = TrainingAnomalyHook(window=10, spike_factor=5.0, clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        for _ in range(10):
            r.step({'loss': 2.0, 'grad_norm': 10.0})
            h.after_train_iter(r)
        r.step({'loss': 40.0, 'grad_norm': 10.0})       # 20x the median
        h.after_train_iter(r)
        ev = events(path)
        print(f'  loss 2.0 -> 40.0    -> {[e["kind"] for e in ev]}')
        fails += [] if any(e['kind'] == 'loss_spike' for e in ev) else ['spike']
        # and an ordinary wobble must NOT fire
        os.remove(path)
        r.step({'loss': 5.0, 'grad_norm': 10.0})
        h.after_train_iter(r)
        quiet = not events(path)
        print(f'  loss 2.0 -> 5.0     -> quiet: {quiet}')
        fails += [] if quiet else ['spike-false-positive']

        # ------------------------------------------------------- stall ----
        if os.path.exists(path):
            os.remove(path)
        h = TrainingAnomalyHook(window=10, stall_epochs=3, stall_rel=0.005,
                                clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        for ep in range(4):
            for _ in range(5):
                # map stuck at 4.0; det falling normally
                r.step({'loss': 9.0, 'map.loss_map_pts': 4.0,
                        'det.loss_cls': 5.0 - 0.5 * ep})
                h.after_train_iter(r)
            h.after_train_epoch(r)
            r.epoch += 1
        ev = [e for e in events(path) if e['kind'] == 'loss_stalled']
        stalled = {e['key'] for e in ev}
        print(f'  frozen map loss     -> stalled={sorted(stalled)}')
        fails += [] if 'map.loss_map_pts' in stalled else ['stall']
        fails += [] if 'det.loss_cls' not in stalled else ['stall-false-positive']

        # ---------------------------------------------------- collapse ----
        os.remove(path) if os.path.exists(path) else None
        h = TrainingAnomalyHook(window=10, map_spread_min=0.5,
                                conf_gap_min=0.02, clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        for ep in range(2):
            for _ in range(3):
                r.step({'loss': 5.0, 'map_spread_m': 0.01,
                        'map_conf_pos': 0.30, 'map_conf_neg': 0.299,
                        'clip/rate': 1.0, 'clip/factor': 0.5,
                        'gscale/det': 1e-5})
                h.after_train_iter(r)
            h.after_train_epoch(r)
            r.epoch += 1
        kinds = {e['kind'] for e in events(path)}
        print(f'  collapsed map head  -> {sorted(kinds)}')
        for want in ('map_query_collapse', 'map_no_discrimination',
                     'clipping_every_iteration', 'grad_balance_saturated'):
            fails += [] if want in kinds else [want]

        # -------------------------------------------------- eval regress --
        os.remove(path)
        h = TrainingAnomalyHook(window=10, eval_rel_tol=0.02, clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        r.log_buffer.output = {'plan_L2_stp3_avg': 0.80,
                               'NuscMap_chamfer/mAP': 0.30}
        h.after_train_epoch(r)
        r.epoch += 1
        r.log_buffer.output = {'plan_L2_stp3_avg': 0.95,   # worse (lower=better)
                               'NuscMap_chamfer/mAP': 0.31}
        h.after_train_epoch(r)
        ev = [e for e in events(path) if e['kind'] == 'eval_regression']
        print(f'  L2 0.80 -> 0.95     -> {[e["key"] for e in ev]}')
        fails += [] if any(e['key'] == 'plan_L2_stp3_avg' for e in ev) \
            else ['eval-regression']
        fails += [] if not any(e['key'] == 'NuscMap_chamfer/mAP' for e in ev) \
            else ['eval-false-positive']

        # ------------------------------------------------- never raises ----
        h = TrainingAnomalyHook(clamp=(1e-5, 1.0))
        r = _Runner(tmp)
        h.before_run(r)
        try:
            r.step({'loss': 'not-a-number', 'weird': None})
            h.after_train_iter(r)
            h.after_train_epoch(r)
            print('  garbage log_vars    -> survived')
        except Exception as e:                       # noqa: BLE001
            print(f'  garbage log_vars    -> RAISED {type(e).__name__}: {e}')
            fails += ['robustness']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + ('ALL ANOMALY-HOOK CHECKS PASS' if not fails
                  else f'STILL FAILING: {sorted(set(fails))}'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
