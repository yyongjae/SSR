"""Close the two verification gaps a review flagged.

A. bs=4 FULL train path: forward_train + backward + grad-clip + AdamW step,
   with every head on. Reports peak memory.
B. Proper leakage intervention: perturb ONE sample's *images* and require every
   other sample's output to stay bit-identical. Four identical copies agreeing
   is NOT sufficient evidence -- a sample-0-broadcast bug passes that test.
   Perturbing the images exercises backbone / lidar2img / visibility / prev_bev,
   not just the navigation command.
"""
import os
import sys, warnings, importlib, copy as _copy
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import torch
from mmcv import Config
importlib.import_module('projects.mmdet3d_plugin')
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader

CFG = 'projects/configs/SSR/PARA_SSR_e2e_12ep.py'
BS = 4
cfg = Config.fromfile(CFG)
ds = build_dataset(cfg.data.train)
dl = build_dataloader(ds, samples_per_gpu=BS, workers_per_gpu=2, num_gpus=1,
                      dist=False, shuffle=False, seed=0)
batch = next(iter(dl))

torch.manual_seed(0)
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'))
model.init_weights()
model = model.cuda()

kw = {}
for k, v in batch.items():
    d = v.data[0] if hasattr(v, 'data') else v
    if torch.is_tensor(d):
        d = d.cuda()
    elif isinstance(d, list):
        d = [x.cuda() if torch.is_tensor(x) else x for x in d]
    kw[k] = d

# ------------------------------------------------------------------ A ----
print(f'=== A. bs={BS} full train step (forward + backward + optimizer) ===')
from mmcv.runner import build_optimizer
opt = build_optimizer(model, cfg.optimizer)
torch.cuda.reset_peak_memory_stats()
model.train()
losses = model.forward_train(**_copy.deepcopy(kw))

from mmdet.models.detectors.base import BaseDetector
loss, log_vars = model._parse_losses(losses)
print(f'  loss terms: {len(losses)}   total loss = {float(loss):.4f}')
nonfinite = [k for k, v in losses.items()
             if torch.is_tensor(v) and not torch.isfinite(v).all()]
print(f'  non-finite loss terms: {nonfinite if nonfinite else "none"}')

opt.zero_grad()
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.grad is not None], max_norm=35)
print(f'  grad norm before clip: {float(gn):.3f}')
bad = [n for n, p in model.named_parameters()
       if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f'  params with non-finite grad: {len(bad)}')
before = {n: p.detach().clone() for n, p in model.named_parameters()}
opt.step()
moved = sum(1 for n, p in model.named_parameters()
            if not torch.equal(before[n], p.detach()))
total = sum(1 for _ in model.parameters())
nofinite_after = [n for n, p in model.named_parameters()
                  if not torch.isfinite(p).all()]
print(f'  parameters updated by the step: {moved}/{total}')
print(f'  non-finite parameters after step: {len(nofinite_after)}')
peak = torch.cuda.max_memory_allocated() / 2**30
print(f'  PEAK GPU MEMORY: {peak:.2f} GiB   (49 GiB card)')
okA = not nonfinite and not bad and not nofinite_after and moved > 0

# ------------------------------------------------------------------ B ----
print(f'\n=== B. leakage intervention: corrupt one sample\'s images ===')
model.eval()

def plan(imgs):
    with torch.no_grad():
        prev = model.obtain_history_bev(imgs[:, :-1], _copy.deepcopy(kw['img_metas']),
                                        kw['ego_fut_cmd'][:, :-1])
        model.eval()
        m = [e[imgs.size(1) - 1] for e in _copy.deepcopy(kw['img_metas'])]
        feats = model.extract_feat(img=imgs[:, -1].clone(), img_metas=m)
        return model.pts_bbox_head(feats, m, prev_bev=prev,
                                   cmd=kw['ego_fut_cmd'][:, -1])['ego_fut_preds']

base = plan(kw['img'].clone())
print(f'{"perturbed":>10} | ' + ' | '.join(f'sample {j} delta' for j in range(BS)))
okB = True
for i in range(BS):
    im = kw['img'].clone()
    im[i] = torch.randn_like(im[i]) * 3.0        # destroy sample i's images
    out = plan(im)
    deltas = [(out[j] - base[j]).abs().max().item() for j in range(BS)]
    row = ' | '.join(f'{d:14.3e}' for d in deltas)
    print(f'{i:>10} | {row}')
    if deltas[i] <= 0:
        okB = False
    for j in range(BS):
        if j != i and deltas[j] != 0.0:
            okB = False

print('\n  expected: diagonal > 0 (own images matter), off-diagonal exactly 0')
print(f'  result: {"NO LEAKAGE" if okB else "*** LEAKAGE DETECTED ***"}')

print(f'\nVERDICT  A(full bs=4 step): {"PASS" if okA else "FAIL"}'
      f'   B(no cross-sample leakage): {"PASS" if okB else "FAIL"}')
