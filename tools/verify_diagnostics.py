"""Do the new diagnostic metrics compute what their names say?

``gcos/*``, ``tok_sim`` and ``bev_std`` are pure tensor reductions, so they can
be checked against constructed inputs with known answers -- no GPU, no data, no
seven-minute dataset load. That matters: the end-to-end run is the only place
these have been exercised so far, and a shared box means a GPU is not always
available to exercise it on.

Exits non-zero on failure.
"""
import importlib
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.SSR.para_ssr import ParaSSR  # noqa: E402

fails = []

# ------------------------------------------------------------------ gcos ---
print('=== gcos: sign and magnitude must match the constructed angle ===')
n = 1000
torch.manual_seed(0)
base = torch.randn(n)
zero = torch.zeros(())
cases = [
    ('identical      ', base, base, 1.0),
    ('opposite       ', base, -base, -1.0),
    ('orthogonal     ', torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 0.0),
    ('60 degrees     ', torch.tensor([1.0, 0.0]),
     torch.tensor([0.5, math.sqrt(3) / 2]), 0.5),
]
for name, a, b, want in cases:
    grads = {'plan': a, 'det': b}
    norms = {'plan': a.norm(), 'det': b.norm()}
    got = float(ParaSSR._bev_grad_cosines(grads, norms, zero)['gcos/plan-det'])
    ok = abs(got - want) < 1e-5
    print(f'  {name} cos = {got:+.5f}  (want {want:+.2f}) {"ok" if ok else "FAIL"}')
    fails += [] if ok else [f'gcos-{name.strip()}']

print('\n=== gcos: motion is excluded, every other pair is present ===')
g = {k: torch.randn(50) for k in ('plan', 'det', 'map', 'motion')}
nm = {k: v.norm() for k, v in g.items()}
keys = set(ParaSSR._bev_grad_cosines(g, nm, zero))
want = {'gcos/plan-det', 'gcos/plan-map', 'gcos/det-map'}
print(f'  got  {sorted(keys)}')
ok = keys == want
fails += [] if ok else ['gcos-pairs']
# a zero-norm task must not produce a divide-by-zero
g2 = {'plan': torch.zeros(50), 'det': torch.randn(50)}
v = float(ParaSSR._bev_grad_cosines(
    g2, {'plan': zero, 'det': g2['det'].norm()}, zero)['gcos/plan-det'])
print(f'  zero-norm task   -> {v:.4f} (finite: {math.isfinite(v)})')
fails += [] if math.isfinite(v) else ['gcos-zero-norm']

# --------------------------------------------------------------- tok_sim ---
print('\n=== tok_sim: 1.0 only when the scene tokens are copies ===')
B, N, C = 2, 16, 256
collapsed = torch.ones(N, B, C)                       # every token identical
m = ParaSSR._representation_metrics({'scene_query': collapsed})
print(f'  identical tokens  tok_sim = {float(m["tok_sim"]):.4f} (want 1.0000)')
fails += [] if abs(float(m['tok_sim']) - 1.0) < 1e-5 else ['tok-collapsed']

torch.manual_seed(0)
diverse = torch.randn(N, B, C)                        # random -> near-orthogonal
m = ParaSSR._representation_metrics({'scene_query': diverse})
sim = float(m['tok_sim'])
print(f'  random tokens     tok_sim = {sim:.4f} (want ~0, |.|<0.1)')
fails += [] if abs(sim) < 0.1 else ['tok-diverse']

# antipodal pairs: half the tokens are the negation of the other half
half = torch.randn(N // 2, B, C)
anti = torch.cat([half, -half], dim=0)
sim = float(ParaSSR._representation_metrics({'scene_query': anti})['tok_sim'])
print(f'  antipodal tokens  tok_sim = {sim:.4f} (want < 0)')
fails += [] if sim < 0 else ['tok-antipodal']

# --------------------------------------------------------------- bev_std ---
print('\n=== bev_std: 0 only when the BEV is spatially constant ===')
flat = torch.ones(2, 10000, 256) * 3.7                # same value every cell
m = ParaSSR._representation_metrics({'bev_embed': flat})
print(f'  constant BEV      bev_std = {float(m["bev_std"]):.6f} (want 0.000000)')
fails += [] if float(m['bev_std']) < 1e-6 else ['bev-constant']

torch.manual_seed(0)
varied = torch.randn(2, 10000, 256) * 2.0             # std 2 per cell
m = ParaSSR._representation_metrics({'bev_embed': varied})
got = float(m['bev_std'])
print(f'  N(0,2) BEV        bev_std = {got:.4f} (want ~2.0)')
fails += [] if abs(got - 2.0) < 0.05 else ['bev-varied']

# missing keys must not raise -- occ_head=None already removes one group
m = ParaSSR._representation_metrics({})
print(f'\n  empty outs        -> {m} (no raise)')
fails += [] if m == {} else ['empty-outs']

print('\n' + ('ALL DIAGNOSTIC CHECKS PASS' if not fails
              else f'STILL FAILING: {sorted(set(fails))}'))
sys.exit(1 if fails else 0)
