"""Does aux_grad_scale scale ONLY the shared-BEV gradient, or the head too?

Same head, same weights, same input, same loss -- only the scale changes.
"""
import os
import sys, warnings, importlib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import torch
from mmcv import Config
importlib.import_module('projects.mmdet3d_plugin')
from mmdet.models import build_head
from projects.mmdet3d_plugin.SSR.para_ssr import _ScaleGrad

cfg = Config.fromfile('projects/configs/SSR/PARA_SSR_e2e.py')
torch.manual_seed(0)
head = build_head(cfg.model.occ_head)
gt = (torch.rand(1, 7, 1, 100, 100) > 0.85).float()
valid = torch.ones(1, 7, dtype=torch.bool)
bev0 = torch.randn(1, 100 * 100, 256)

def run(scale):
    bev = bev0.clone().requires_grad_(True)
    head.zero_grad(set_to_none=True)
    aux = _ScaleGrad.apply(bev, scale) if scale != 1.0 else bev
    losses = head.forward_train(aux, gt, valid)
    total = sum(losses.values())
    total.backward()
    pg = {n: p.grad.norm().item() for n, p in head.named_parameters()
          if p.grad is not None}
    return total.item(), bev.grad.norm().item(), pg

l1, bev_g1, pg1 = run(1.0)
l2, bev_g2, pg2 = run(0.1)

print('=== aux_grad_scale: 1.0 vs 0.1 (same weights/input) ===')
print(f'loss value          {l1:.6f}  ->  {l2:.6f}   ratio {l2/l1:.4f}')
print(f'||grad @ bev_embed||{bev_g1:.6e}  ->  {bev_g2:.6e}   ratio {bev_g2/bev_g1:.4f}')
print('\nhead parameter gradients:')
for n in sorted(pg1):
    r = pg2[n] / pg1[n] if pg1[n] else float('nan')
    print(f'  {n:28s} {pg1[n]:.6e} -> {pg2[n]:.6e}   ratio {r:.4f}')

ok_bev = abs(bev_g2 / bev_g1 - 0.1) < 1e-4
ok_head = all(abs(pg2[n] / pg1[n] - 1.0) < 1e-5 for n in pg1 if pg1[n])
ok_loss = abs(l2 / l1 - 1.0) < 1e-9
print(f'\nBEV gradient scaled by exactly 0.1 : {ok_bev}')
print(f'head gradients unchanged (ratio 1) : {ok_head}')
print(f'loss value unchanged               : {ok_loss}')
print('\nVERDICT:', 'BEV-only scaling CONFIRMED' if (ok_bev and ok_head and ok_loss)
      else 'NOT what was expected')
