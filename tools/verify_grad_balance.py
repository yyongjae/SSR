"""Does grad_balance actually put each task's BEV-gradient share on target?

Two checks:

1. The solver's algebra, on synthetic norms. Deterministic, instant.
2. The whole loop on real batches: build the model with a target, run a few
   iterations, and watch the MEASURED shares walk to the target. This is the
   one that matters, because it exercises the per-task ``_ScaleGrad`` valves,
   the back-out of the previous scale, and the EMA together.

Exits non-zero on failure.
"""
import argparse
import importlib
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from mmcv import Config  # noqa: E402

importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.SSR.utils.grad_balance import GradBalancer  # noqa

fails = []
TARGET = dict(plan=0.4, det=0.2, map=0.2, occ=0.2)

# ------------------------------------------------------------- 1. algebra --
print('=== solver: shares must land on target from any starting ratio ===')
# raw gradient norms wildly out of balance, the way they actually are
raw = dict(plan=0.010, det=1.200, map=0.400, occ=0.050)
bal = GradBalancer(TARGET, interval=1, momentum=0.0, clamp=(1e-6, 1e6),
                   warmup_iters=0)
for step in range(3):
    measured = {k: raw[k] * bal.scale_for(k) for k in raw}
    bal.update(measured)
measured = {k: raw[k] * bal.scale_for(k) for k in raw}
tot = sum(measured.values())
print(f'  {"task":6s} {"raw":>9s} {"scale":>10s} {"share":>8s} {"target":>8s}')
ok = True
for k in ('plan', 'det', 'map', 'occ'):
    share = measured[k] / tot
    print(f'  {k:6s} {raw[k]:9.4f} {bal.scale_for(k):10.5f} '
          f'{share:8.4f} {TARGET[k]:8.2f}')
    ok &= abs(share - TARGET[k]) < 1e-4
fails += [] if ok else ['solver-shares']

print('\n=== solver: must reach target FAST at the momentum we actually run ===')
# The bug this catches: with arithmetic smoothing the scales crawl from 1.0 as
# 0.9^n -- about 70 updates, four epochs at interval 200, before the controller
# does anything. Only visible at momentum > 0, which the earlier test skipped.
b09 = GradBalancer(TARGET, interval=1, momentum=0.9, clamp=(1e-6, 1e6),
                   warmup_iters=0)
raw2 = dict(plan=0.0007, det=0.5745, map=0.2701, occ=0.1546)   # real iter-0 split
print(f'  {"update":>7s} | {"plan":>8s} {"det":>8s} {"map":>8s} {"occ":>8s}')
for step in range(1, 6):
    b09.update({k: raw2[k] * b09.scale_for(k) for k in raw2})
    cur = {k: raw2[k] * b09.scale_for(k) for k in raw2}
    tot = sum(cur.values())
    print(f'  {step:7d} | ' +
          ' '.join(f'{cur[k]/tot:8.4f}' for k in ('plan', 'det', 'map', 'occ')))
err1 = max(abs(cur[k] / tot - TARGET[k]) for k in TARGET)
print(f'  worst error after 5 updates at momentum 0.9: {err1:.4f}')
fails += [] if err1 < 0.02 else ['momentum-too-slow']

print('\n=== solver: momentum, clamp and warm-up ===')
b2 = GradBalancer(TARGET, interval=10, momentum=0.9, clamp=(1e-3, 1.0),
                  warmup_iters=100)
print(f'  warmup: should_update(50)={b2.should_update(50)} '
      f'(want False)   should_update(110)={b2.should_update(110)} (want True)')
fails += [] if (not b2.should_update(50) and b2.should_update(110)) \
    else ['warmup-gate']
# a task whose gradient has collapsed must not be amplified without bound
b2.update(dict(plan=1.0, det=1e-9, map=1.0, occ=1.0))
for _ in range(200):
    b2.update(dict(plan=1.0, det=1e-9, map=1.0, occ=1.0))
print(f'  collapsed-task scale after 200 updates: {b2.scale_for("det"):.4f} '
      f'(clamped at 1.0)')
fails += [] if b2.scale_for('det') <= 1.0 + 1e-9 else ['clamp']
# planning contributing nothing must be a no-op, not a divide-by-zero
before = dict(b2.scale)
b2.update(dict(plan=0.0, det=1.0, map=1.0, occ=1.0))
fails += [] if b2.scale == before else ['zero-numeraire']
print(f'  zero planning gradient -> scales unchanged: {b2.scale == before}')

# --------------------------------------------------------- 2. real model --
ap = argparse.ArgumentParser()
ap.add_argument('--config',
                default='projects/configs/SSR/PARA_SSR_e2e_12ep.py')
ap.add_argument('--iters', type=int, default=6)
ap.add_argument('--skip-model', action='store_true')
args = ap.parse_args()

if not args.skip_model and torch.cuda.is_available():
    print(f'\n=== real batches: measured share must walk to target '
          f'({args.iters} iterations) ===')
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataloader

    cfg = Config.fromfile(args.config)
    cfg.model.pop('aux_grad_scale', None)          # replaced by grad_balance
    cfg.model.grad_balance = dict(target=TARGET, interval=1, momentum=0.0,
                                  clamp=(1e-6, 1e3), warmup_iters=0)
    cfg.model.grad_norm_log_interval = 1
    cfg.model.aux_metric_log_interval = 0
    ds = build_dataset(cfg.data.train)
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
                          dist=False, shuffle=False, seed=0)
    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'))
    model.init_weights()
    model = model.cuda().train()

    it = iter(dl)
    print(f'  {"iter":>4s} | ' + '  '.join(f'{t:>7s}' for t in TARGET) +
          '   | ' + '  '.join(f's_{t}' for t in ('det', 'map', 'occ')))
    last = None
    for i in range(args.iters):
        batch = next(it)
        kw = {}
        for k, v in batch.items():
            d = v.data[0] if hasattr(v, 'data') else v
            if torch.is_tensor(d):
                d = d.cuda()
            elif isinstance(d, list):
                d = [x.cuda() if torch.is_tensor(x) else x for x in d]
            kw[k] = d
        losses = model.forward_train(**kw)
        share = {t: float(losses[f'gshare/{t}']) for t in TARGET}
        sc = [model.grad_balancer.scale_for(t) for t in ('det', 'map', 'occ')]
        print(f'  {i:4d} | ' + '  '.join(f'{share[t]:7.4f}' for t in TARGET) +
              '   | ' + '  '.join(f'{s:8.2e}' for s in sc))
        last = share
        # free the graph the diagnostic grad() call retained
        sum(v for k, v in losses.items() if 'loss' in k).backward()
        model.zero_grad(set_to_none=True)

    err = max(abs(last[t] - TARGET[t]) for t in TARGET)
    print(f'\n  target {TARGET}')
    print(f'  worst share error after {args.iters} iterations: {err:.4f}')
    # batches differ, so the controller chases a moving target; 0.1 absolute is
    # a loose but meaningful bound against the 0.008/0.42/0.55/0.018 it starts
    # from
    fails += [] if err < 0.10 else ['closed-loop']
else:
    print('\n(skipping the real-model check)')

print('\n' + ('ALL GRAD-BALANCE CHECKS PASS' if not fails
              else f'STILL FAILING: {fails}'))
sys.exit(1 if fails else 0)
