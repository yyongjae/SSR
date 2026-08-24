"""E1 + E2 -- planning-centric BEV maps, and what the auxiliary heads read.

Per checkpoint, over N val samples:

  pi[i]   share of the planned trajectory contributed by BEV cell i.
          Sum over cells plus the bias reproduces the trajectory exactly, which
          is what separates this from a saliency map.
  rho     norm of the bias over (bias + signal): the share of the plan that
          comes out the same no matter what the BEV holds. The ego-shortcut,
          measured rather than ablated.
  PR      participation ratio of pi -- how many of the 10,000 cells the planner
          effectively uses.
  PPA     pi mass inside ground-truth boxes over the uniform expectation. Above
          1 means the planner attends to actual objects.
  omega[i] the same decomposition for the detection and map heads, so the two
          can be crossed: cells only the aux heads read are BEV capacity the
          planner never looks at.

    python tools/lmd/run_e1.py --ckpt both --samples 200 --out out/e1_both.npz
"""
import argparse
import os

import numpy as np
import torch

from lmd_common import build, bev_grid_xy, gt_box_mask, iter_samples, resolve, val_loader
from lmd_hooks import Recorder, bev_override, linearise
from run_e0 import analyse, cmd_index, plan_from_bev


def aux_omega(model, bev_ref, head_name, key):
    """Per-cell contribution to an auxiliary head, read the same way as pi.

    The summary is the sum of each query's max-class logit with the argmax frozen
    -- a linear functional of the head's output, so the decomposition still sums
    to it exactly and that identity stays checkable.
    """
    head = getattr(model, head_name, None)
    if head is None:
        return None, None
    bev = bev_ref.detach().clone().requires_grad_(True)
    rec = Recorder()
    with linearise(head, rec):
        outs = head(bev)
        cls = outs[key][-1][0]                       # [Q, num_classes]
        pick = cls.detach().argmax(-1, keepdim=True)
        s = cls.gather(1, pick).sum()
    g, = torch.autograd.grad(s, bev)
    contrib = (g[0] * bev[0]).sum(-1)

    rec.freeze()
    with linearise(head, rec), torch.no_grad():
        outs0 = head(torch.zeros_like(bev))
        b = outs0[key][-1][0].gather(1, pick).sum()
    resid = float((s.detach() - (contrib.sum() + b)).abs() / s.detach().abs().clamp(min=1e-9))
    return contrib.detach().cpu().numpy(), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, choices=['plan_only', 'aux_only', 'staged', 'both'])
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--plan-from', default=None,
                    choices=[None, 'plan_only', 'aux_only', 'staged', 'both'],
                    help="read this checkpoint's BEV with ANOTHER checkpoint's planner. "
                         "Only valid for the stage1/stage2 fork pair -- report #10 section 2.")
    a = ap.parse_args()

    cfg_path, ckpt_path, desc = resolve(a.ckpt, a.epoch)
    print(f'E1  {a.ckpt}  ({desc})\n    ckpt {ckpt_path}')
    model, cfg = build(cfg_path, ckpt_path, a.device)

    reader = model
    if a.plan_from:
        if {a.ckpt, a.plan_from} != {'aux_only', 'staged'}:
            raise SystemExit('--plan-from is only valid for the aux_only/staged fork pair; '
                             'any other combination crosses independently trained runs '
                             '(report #10 section 2)')
        rcfg, rckpt, rdesc = resolve(a.plan_from)
        print(f'    reader {a.plan_from} ({rdesc})  {rckpt}')
        reader, _ = build(rcfg, rckpt, a.device)

    _, loader = val_loader(cfg)
    xs, ys, H, W, pc = bev_grid_xy(cfg)
    print(f'    BEV {H}x{W}  pc_range {pc}\n')

    acc = dict(pi=np.zeros(H * W), omega_det=np.zeros(H * W), omega_map=np.zeros(H * W))
    rows, n_det, n_map = [], 0, 0
    for i, data, tap in iter_samples(model, loader, a.samples, a.device):
        bev_ref = tap['out']['bev_embed'].detach()
        rhead = reader.pts_bbox_head
        tgt, contrib, bias, idx = analyse(rhead, tap, bev_ref)

        r_sum = float((tgt - (contrib.sum(0) + bias)).abs().max() / tgt.abs().max())
        pi = contrib.abs().sum(1)
        pi = (pi / pi.sum()).cpu().numpy()
        pr = float(pi.sum() ** 2 / (pi ** 2).sum())
        rho = float(bias.norm() / (bias.norm() + contrib.sum(0).norm()))

        m = gt_box_mask(data, xs, ys)
        ppa = float(pi[m].sum() / max(m.mean(), 1e-9)) if m is not None and m.any() else np.nan

        od, rd = aux_omega(model, bev_ref, 'det_motion_head', 'all_cls_scores')
        om, rm = aux_omega(model, bev_ref, 'map_head', 'map_all_cls_scores')

        def cosv(u, v):
            if v is None:
                return np.nan
            v = np.abs(v)
            return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

        rows.append((i, idx, r_sum, pr, rho, ppa, cosv(pi, od), cosv(pi, om),
                     rd if rd is not None else np.nan, rm if rm is not None else np.nan))
        acc['pi'] += pi
        if od is not None:
            acc['omega_det'] += np.abs(od) / (np.abs(od).sum() + 1e-12); n_det += 1
        if om is not None:
            acc['omega_map'] += np.abs(om) / (np.abs(om).sum() + 1e-12); n_map += 1
        if i % 25 == 0:
            print(f'  {i:4d}  resid {r_sum:.1e}  PR {pr:7.0f}  rho {rho:.3f}  PPA {ppa:5.2f}')

    R = np.array(rows, dtype=np.float64)
    acc['pi'] /= len(rows)
    if n_det:
        acc['omega_det'] /= n_det
    if n_map:
        acc['omega_map'] /= n_map

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, rows=R, pi=acc['pi'].reshape(H, W),
        omega_det=acc['omega_det'].reshape(H, W), omega_map=acc['omega_map'].reshape(H, W),
        xs=xs.reshape(H, W), ys=ys.reshape(H, W), pc_range=np.array(pc),
        cols=np.array(['sample', 'cmd', 'resid', 'PR', 'rho', 'PPA',
                       'cos_pi_det', 'cos_pi_map', 'resid_det', 'resid_map']))

    def s(j):
        v = R[:, j]
        v = v[~np.isnan(v)]
        return f'{v.mean():.4f} +- {v.std():.4f}' if len(v) else 'n/a'

    print(f'\n  === {a.ckpt} ({desc}), {len(rows)} samples ===')
    print(f'  worst residual (plan)  {np.nanmax(R[:, 2]):.2e}   '
          f'(det) {np.nanmax(R[:, 8]):.2e}   (map) {np.nanmax(R[:, 9]):.2e}')
    print(f'  PR          {s(3)}   / {H * W}')
    print(f'  rho         {s(4)}      <- scene-independent share of the plan')
    print(f'  PPA         {s(5)}      <- >1 means planning mass sits on GT boxes')
    print(f'  cos(pi,det) {s(6)}')
    print(f'  cos(pi,map) {s(7)}')
    print(f'\n  saved {a.out}')


if __name__ == '__main__':
    main()
