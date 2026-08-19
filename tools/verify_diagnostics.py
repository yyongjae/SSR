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

# -------------------------------------------------------- pnorm grouping ---
print('=== pnorm/uwr groups: order-independent, disjoint, complete ===')
import collections                                              # noqa: E402
from mmcv import Config as _Cfg                                 # noqa: E402
from mmdet3d.models import build_model as _build                # noqa: E402
from projects.mmdet3d_plugin.SSR.hooks.clip_monitor import (    # noqa: E402
    ClipMonitorOptimizerHook, _DEFAULT_GROUPS)

_k = _Cfg.fromfile('projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py')
_m = _build(_k.model, train_cfg=_k.get('train_cfg'))


def _totals(groups):
    h = ClipMonitorOptimizerHook(groups=groups)
    t = collections.Counter()
    for n, prm in _m.named_parameters():
        if prm.requires_grad:
            t[h._group_of(n)] += prm.numel()
    return dict(t)


fwd = _totals(_DEFAULT_GROUPS)
rev = _totals(dict(reversed(list(_DEFAULT_GROUPS.items()))))
total = sum(p.numel() for p in _m.parameters() if p.requires_grad)
for g in sorted(fwd):
    print(f'  {g:6s} {fwd[g]:12,d}')
print(f'  order-independent : {fwd == rev}   '
      f'(plan="pts_bbox_head" is a prefix of every bev entry)')
print(f'  disjoint+complete : {sum(fwd.values()) == total}  '
      f'({sum(fwd.values()):,} == {total:,})')
fails += [] if fwd == rev else ['pnorm-group-order']
fails += [] if sum(fwd.values()) == total else ['pnorm-group-coverage']
# and the BEV encoder must NOT be inside plan -- the bug this split fixes
fails += [] if fwd.get('bev', 0) > 4e6 and fwd.get('plan', 0) < 3e6 \
    else ['pnorm-bev-in-plan']

print('\n=== pnorm must reconstruct the norm that clipping actually uses ===')
# The decomposition exists to say WHERE the clipped norm comes from. If the
# parts do not add back up to the whole, every conclusion drawn from it is void.
import torch.nn as _nn                                          # noqa: E402
from torch.nn.utils import clip_grad as _clip                    # noqa: E402


class _Tiny(_nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = _nn.Linear(4, 8)                     # trunk
        self.pts_bbox_head = _nn.Module()
        self.pts_bbox_head.transformer = _nn.Linear(8, 8)        # bev
        self.pts_bbox_head.way_decoder = _nn.Linear(8, 3)        # plan
        self.map_head = _nn.Linear(8, 3)                         # aux
        self.frozen = _nn.Linear(3, 3)                           # requires_grad=False
        for q in self.frozen.parameters():
            q.requires_grad = False


_t = _Tiny()
_h = _t.img_backbone(torch.randn(7, 4))
_b = _t.pts_bbox_head.transformer(_h)
(_t.pts_bbox_head.way_decoder(_b).pow(2).mean()
 + _t.map_head(_b).abs().mean()).backward()


class _R:
    pass


_r = _R()
_r.model = _t
_pn = ClipMonitorOptimizerHook(
    grad_clip=dict(max_norm=1e9, norm_type=2))._group_norms(_r)
_truth = float(_clip.clip_grad_norm_(
    [q for q in _t.parameters() if q.requires_grad and q.grad is not None], 1e9))
print(f'  clip_grad_norm_ {_truth:.8f}   pnorm/total {_pn["pnorm/total"]:.8f}   '
      f'diff {abs(_truth - _pn["pnorm/total"]):.2e}')
print(f'  pnorm/other {_pn["pnorm/other"]:.1f} (must be 0: nothing ungrouped)')
fails += [] if abs(_truth - _pn['pnorm/total']) < 1e-5 else ['pnorm-reconstruct']
fails += [] if _pn['pnorm/other'] == 0.0 else ['pnorm-other-nonzero']
# and the bev group must be non-empty and separate from plan
print(f'  pnorm/bev {_pn["pnorm/bev"]:.6f}  pnorm/plan {_pn["pnorm/plan"]:.6f}  '
      f'(the encoder is its own group, not folded into plan)')
fails += [] if _pn['pnorm/bev'] > 0 and _pn['pnorm/plan'] > 0 else ['pnorm-bev-empty']

# ------------------------------------------------------- encoder gradient ---
print('\n=== _encoder_grad: the chain-rule shortcut must equal the direct grad ===')
# Mimic the real shape: encoder params -> bev_embed -> two heads -> two losses.
torch.manual_seed(0)
W = torch.nn.Parameter(torch.randn(8, 8))          # "encoder weights"
b = torch.nn.Parameter(torch.randn(8))
x = torch.randn(5, 8)
bev = torch.tanh(x @ W + b)                        # "bev_embed"
head_a = torch.nn.Linear(8, 3)
head_b = torch.nn.Linear(8, 3)
loss_a = head_a(bev).pow(2).mean()
loss_b = head_b(bev).abs().mean()

for nm, loss in (('task A', loss_a), ('task B', loss_b)):
    # what the code does: grad at bev_embed, then continue into the encoder
    g_bev = torch.autograd.grad(loss, bev, retain_graph=True)[0]
    via = ParaSSR._encoder_grad(bev, [W, b], g_bev.detach().flatten())
    # what it must equal: straight to the encoder weights
    direct = torch.autograd.grad(loss, [W, b], retain_graph=True)
    direct = torch.cat([d.flatten() for d in direct])
    err = float((via - direct).abs().max())
    ok = err < 1e-6
    print(f'  {nm}: max|shortcut - direct| = {err:.3e}  '
          f'norm {float(via.norm()):.6f} vs {float(direct.norm()):.6f} '
          f'{"ok" if ok else "FAIL"}')
    fails += [] if ok else [f'enc-grad-{nm}']

# and the two tasks must give DIFFERENT encoder gradients, or the metric is
# measuring nothing
ga = ParaSSR._encoder_grad(
    bev, [W, b], torch.autograd.grad(loss_a, bev, retain_graph=True)[0].flatten())
gb = ParaSSR._encoder_grad(
    bev, [W, b], torch.autograd.grad(loss_b, bev, retain_graph=True)[0].flatten())
cos = float(torch.dot(ga, gb) / (ga.norm() * gb.norm()))
print(f'  enc_gcos between the two tasks: {cos:+.4f} (distinct directions)')
fails += [] if abs(cos) < 0.999 else ['enc-grad-identical']

# --------------------------------------------------------- token coverage ---
print('\n=== tok_cover / tok_union: what the 16-token bottleneck keeps ===')
Bt, Nt, HW = 2, 16, 10000
one = torch.full((Bt, Nt, HW), -30.0)
for i in range(Nt):
    one[:, i, i] = 60.0                              # each token = one cell
m = ParaSSR._representation_metrics({'token_attn': one.softmax(-1)})
print(f'  token = 1 cell     cover {float(m["tok_cover"]):.6f} (want {1/HW:.6f})  '
      f'union {float(m["tok_union"]):.4f} (want {Nt/HW:.4f})')
ok = (abs(float(m['tok_cover']) - 1 / HW) < 1e-6 and
      abs(float(m['tok_union']) - Nt / HW) < 1e-6)
fails += [] if ok else ['tok-cover-point']

m = ParaSSR._representation_metrics(
    {'token_attn': torch.zeros(Bt, Nt, HW).softmax(-1)})
print(f'  token = global avg cover {float(m["tok_cover"]):.6f} (want 1.000000)  '
      f'-- 16 tokens carrying one number of spatial information')
fails += [] if abs(float(m['tok_cover']) - 1.0) < 1e-4 else ['tok-cover-uniform']

blk = torch.full((Bt, Nt, HW), -30.0)
for i in range(Nt):
    blk[:, i, i * (HW // Nt):(i + 1) * (HW // Nt)] = 0.0   # disjoint 1/16 each
m = ParaSSR._representation_metrics({'token_attn': blk.softmax(-1)})
print(f'  token = BEV/16     cover {float(m["tok_cover"]):.4f} (want {1/Nt:.4f})  '
      f'union {float(m["tok_union"]):.4f} (want 1.0000)')
ok = (abs(float(m['tok_cover']) - 1 / Nt) < 1e-3 and
      abs(float(m['tok_union']) - 1.0) < 1e-6)
fails += [] if ok else ['tok-cover-blocks']

# ------------------------------------------------------------ per-command ---
print('\n=== plan_err_sum / plan_n: only the commanded branch contributes ===')
from projects.mmdet3d_plugin.SSR.para_ssr_head import ParaSSRHead  # noqa: E402

B, M, T = 4, 3, 6
preds = torch.zeros(B, M, T, 2)
gt = torch.zeros(B, M, T, 2)
# every branch is wrong by a known, different amount
for m, e in enumerate((0.1, 0.2, 0.3)):
    preds[:, m] = e
# sample commands: two straight (idx 2), one right (0), one left (1)
cmd = torch.zeros(B, M)
cmd[0, 2] = cmd[1, 2] = cmd[2, 0] = cmd[3, 1] = 1.0
weight = cmd[..., None, None].repeat(1, 1, T, 2)      # all timesteps valid

out = ParaSSRHead._per_command_error(ParaSSRHead, preds, gt, weight)
n = {k.split('/')[1]: float(v) for k, v in out.items() if k.startswith('plan_n')}
e = {k.split('/')[1]: float(v) for k, v in out.items()
     if k.startswith('plan_err_sum')}
print(f'  counts  {n}   (want right=12, left=12, straight=24 elements)')
ok = n == {'right': T * 2 * 1.0, 'left': T * 2 * 1.0, 'straight': T * 2 * 2.0}
fails += [] if ok else ['cmd-counts']
for name, want in (('right', 0.1), ('left', 0.2), ('straight', 0.3)):
    got = e[name] / max(n[name], 1)
    good = abs(got - want) < 1e-6
    print(f'  {name:9s} sum/n = {got:.4f}  (want {want:.1f}) '
          f'{"ok" if good else "FAIL"}')
    fails += [] if good else [f'cmd-{name}']

# a command absent from the batch must give count 0, not a fake-perfect 0 ratio
cmd2 = torch.zeros(B, M)
cmd2[:, 2] = 1.0                                        # all straight
w2 = cmd2[..., None, None].repeat(1, 1, T, 2)
out2 = ParaSSRHead._per_command_error(ParaSSRHead, preds, gt, w2)
absent = float(out2['plan_n/left'])
print(f'  absent command   plan_n/left = {absent} (want 0.0, and the key exists '
      f'so DDP all_reduce stays consistent)')
fails += [] if absent == 0.0 and 'plan_err_sum/left' in out2 else ['cmd-absent']

# ------------------------------------------------- DDP key-set stability ---
print('\n=== every diagnostic key must be emitted unconditionally ===')
# _parse_losses all-reduces each entry of log_vars in order. A rank that omits
# a key because its gradient came back None desynchronises the collective --
# a hang, two ranks deep, days into a run.
_z = torch.zeros(())
_full = {'plan': torch.randn(10), 'det': torch.randn(10), 'map': torch.randn(10)}
_want = set(ParaSSR._bev_grad_cosines(
    _full, {k: v.norm() for k, v in _full.items()}, _z))
for _label, _g in (('one None', {'plan': torch.randn(10), 'det': None,
                                 'map': torch.randn(10)}),
                   ('all None', {'plan': None, 'det': None, 'map': None})):
    _n = {k: (_z if v is None else v.norm()) for k, v in _g.items()}
    _out = ParaSSR._bev_grad_cosines(_g, _n, _z)
    _same = set(_out) == _want
    _finite = all(bool(torch.isfinite(v).all()) for v in _out.values())
    print(f'  {_label}: same key set {_same}, all finite {_finite}')
    fails += [] if (_same and _finite) else [f'gcos-keys-{_label}']

# ------------------------------------------------ observation vs training ---
print('\n=== map metric logging must not change the loss or the RNG ===')
# shift_fixed_num_sampled_points_v* draws from the GLOBAL NumPy RNG for closed
# polylines with more vertices than fixed_num. Reading it twice (metric, then
# loss) gave the two different targets AND moved the stream GridMask draws from.
import numpy as _np                                              # noqa: E402
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import (  # noqa
    LiDARInstanceLines as _Lines)
from shapely.geometry import LineString as _LS                    # noqa: E402

_head = _m.map_head
_Q, _P, _pc = _head.map_num_vec, _head.map_num_pts_per_vec, _head.pc_range
_L = []
for _i in range(3):                       # 41 vertices > fixed_num, and CLOSED
    _t = _np.linspace(0, 2 * _np.pi, 41)
    _p = _np.stack([-5 + 4 * _i + 3 * _np.cos(_t), 3 * _i + 3 * _np.sin(_t)], 1)
    _p[-1] = _p[0]
    _L.append(_LS(_p))
_gt = _Lines(_L, sample_dist=1, num_samples=_P, padding=False, fixed_num=_P,
             padding_value=-10000, patch_size=(_pc[4] - _pc[1], _pc[3] - _pc[0]))
_lab = [torch.zeros(3, dtype=torch.long)]
_bev = torch.randn(1, _head.bev_h * _head.bev_w, _head.embed_dims)

_np.random.seed(42); torch.manual_seed(3)
_off = _head.forward_train(_bev, [{}], [_gt], _lab, metrics_out=None)
_r_off = _np.random.rand()
_np.random.seed(42); torch.manual_seed(3)
_mo = {}
_on = _head.forward_train(_bev, [{}], [_gt], _lab, metrics_out=_mo)
_r_on = _np.random.rand()
_dl = max(abs(float(_off[k]) - float(_on[k])) for k in _off)
print(f'  loss with vs without the metric: max diff {_dl:.3e}')
print(f'  next np.random.rand(): {_r_off:.8f} vs {_r_on:.8f}  same {_r_off == _r_on}')
print(f'  metric still produced: {sorted(_mo)[:3]}')
fails += [] if _dl == 0.0 else ['metric-changes-loss']
fails += [] if _r_off == _r_on else ['metric-changes-rng']
fails += [] if _mo else ['metric-missing']

print('\n=== motion best mode must use each agent\'s last VALID step ===')
_dm = _m.det_motion_head
_T, _M = _dm.fut_ts, _dm.fut_mode
for _nv, _want_sup in ((3, True), (6, True), (0, False)):
    _tgt = torch.zeros(1, _T, 2); _tgt[0, :, 0] = 1.0
    _mask = torch.zeros(1, _T); _mask[0, :_nv] = 1.0
    _pr = torch.zeros(1, _M, _T, 2)
    _pr[0, 0, :, 0] = 5.0            # mode 0: wrong on the valid window
    _pr[0, 1, :, 0] = 1.0            # mode 1: exact on the valid window
    _best = _dm.get_best_fut_preds(_pr, _tgt, _mask)
    _picked = 0 if torch.allclose(_best[0], _pr[0, 0].reshape(-1)) else 1
    _lbl, _sup = _dm.get_traj_cls_target(
        _pr, _tgt, _mask, torch.zeros(1, dtype=torch.bool))
    print(f'  {_nv}/{_T} valid steps -> mode {_picked}, cls {int(_lbl[0])}, '
          f'supervised {bool(_sup[0])}')
    if _nv > 0:
        fails += [] if _picked == 1 and int(_lbl[0]) == 1 else [f'motion-mode-{_nv}']
    fails += [] if bool(_sup[0]) == _want_sup else [f'motion-sup-{_nv}']

print('\n' + ('ALL DIAGNOSTIC CHECKS PASS' if not fails
              else f'STILL FAILING: {sorted(set(fails))}'))
sys.exit(1 if fails else 0)
