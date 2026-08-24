"""E0 -- the gate. Is the real model's BEV -> trajectory path exactly affine?

`tools/lmd/verify_lmd_linearisation.py` proved this on replicas at float64. This
proves it on the actual mmcv modules with the actual trained weights, which is
what every later number depends on. If the residual here is not small, nothing
downstream means anything and the failing op has to be found first.

    python tools/lmd/run_e0.py --ckpt both --samples 3
"""
import argparse

import torch

from lmd_common import build, iter_samples, resolve, val_loader
from lmd_hooks import Recorder, bev_override, linearise


def cmd_index(cmd):
    """Navigation command -> branch index, for either head class."""
    return int(cmd.reshape(-1, 3)[0].argmax())


def plan_from_bev(head, tap, bev):
    """Re-run the head as a pure function of `bev`, using the head's own code."""
    kw = dict(tap['kwargs'])
    kw.pop('only_bev', None)
    with bev_override(head.transformer, bev):
        return head(*tap['args'], **kw)


def analyse(head, tap, bev_ref, rec=None):
    """Returns (traj, per-cell contributions C_i, bias b) for the commanded branch."""
    idx = cmd_index(tap['kwargs']['cmd'])
    bev = bev_ref.detach().clone().requires_grad_(True)

    rec = rec or Recorder()
    with linearise(head, rec):
        outs = plan_from_bev(head, tap, bev)          # records + linearises in one pass
    traj = outs['ego_fut_preds']
    tgt = traj[0, idx].reshape(-1)

    contrib = bev.new_zeros(bev.shape[1], tgt.numel())
    for k in range(tgt.numel()):
        g, = torch.autograd.grad(tgt[k], bev, retain_graph=True)
        contrib[:, k] = (g[0] * bev[0]).sum(-1)

    rec.freeze()
    with linearise(head, rec), torch.no_grad():
        outs0 = plan_from_bev(head, tap, torch.zeros_like(bev))
    bias = outs0['ego_fut_preds'][0, idx].reshape(-1)
    return tgt.detach(), contrib.detach(), bias.detach(), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, choices=['plan_only', 'aux_only', 'staged', 'both'])
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tol', type=float, default=1e-4)
    a = ap.parse_args()

    cfg_path, ckpt_path, desc = resolve(a.ckpt, a.epoch)
    print(f'E0  {a.ckpt}  ({desc})\n    cfg  {cfg_path}\n    ckpt {ckpt_path}\n')
    model, cfg = build(cfg_path, ckpt_path, a.device)
    _, loader = val_loader(cfg)
    head = model.pts_bbox_head

    worst_repro = worst_sum = 0.0
    for i, data, tap in iter_samples(model, loader, a.samples, a.device):
        traj_ref = tap['out']['ego_fut_preds'].detach()
        bev_ref = tap['out']['bev_embed'].detach()
        tgt, contrib, bias, idx = analyse(head, tap, bev_ref)

        ref = traj_ref[0, idx].reshape(-1)
        r_repro = float((tgt - ref).abs().max() / ref.abs().max())
        lhs, rhs = tgt, contrib.sum(0) + bias
        r_sum = float((lhs - rhs).abs().max() / lhs.abs().max())
        worst_repro, worst_sum = max(worst_repro, r_repro), max(worst_sum, r_sum)

        p = contrib.abs().sum(1)
        p = p / p.sum()
        pr = float(p.sum() ** 2 / (p ** 2).sum())
        rho = float(bias.norm() / (bias.norm() + contrib.sum(0).norm()))
        print(f'  sample {i:3d}  cmd={idx}  frozen-vs-real {r_repro:.2e}   '
              f'sum-vs-traj {r_sum:.2e}   PR {pr:7.0f}   rho {rho:.3f}')

    good = worst_repro < a.tol and worst_sum < a.tol
    print(f'\n  worst frozen-vs-real {worst_repro:.2e}   worst sum-vs-traj {worst_sum:.2e}'
          f'   tol {a.tol:.0e}')
    print(f'  {"PASS" if good else "FAIL"} -- {a.ckpt}')
    raise SystemExit(0 if good else 1)


if __name__ == '__main__':
    main()
