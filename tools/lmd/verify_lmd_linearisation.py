"""Is SSR/PARA-SSR's BEV -> trajectory path LMD-linearisable? -- numerical proof.

Prerequisite for everything in `verify_model_delta.py`. LMD (arXiv:2511.00859)
only works if every operation on the path is either (a) already linear, (b)
constant w.r.t. the thing being decomposed, or (c) one of the nonlinearities it
freezes (ReLU/GELU, LayerNorm/GroupNorm, softmax). This checks that claim on
this model's op stack, and checks the consequence that makes the analysis
possible: with the switches frozen, the planned trajectory splits over the
10,000 BEV cells with no residual.

    tau = sum_i C_i + b,     C_i = J_i . bev_i

`C_i` is what defines a planning-centric BEV cell; `b` is the part of the plan
that does not depend on the scene at all.

Modules are replicas of the real ones (see `replica.py`) because mmcv-full is not
installed on the analysis box. Re-run inside the `ssr` env against the real
modules before trusting numbers measured on a checkpoint.

    python3 tools/lmd/verify_lmd_linearisation.py
"""
import argparse

import torch

from lmd_core import SwitchCache, lin_mha
from replica import Pipeline

DT = torch.float64


def rel(a, b):
    return float((a - b).abs().max() / a.abs().max().clamp(min=1e-30))


def ok(name, r, tol=1e-10):
    print(f'  {"PASS" if r < tol else "FAIL":4s}  {name:52s} rel {r:.2e}')
    return r < tol


def run(dev, bev_hw, C=256):
    torch.manual_seed(0)
    HW = bev_hw * bev_hw
    net = Pipeline(C, HW).to(dev).to(DT).eval()
    bev = (torch.randn(1, HW, C, device=dev, dtype=DT) * .5).requires_grad_(True)
    cmd = 2
    passed = []

    # only the reader half: bev_embed -> ego_fut_preds
    READ = ('tokenl', 'latent', 'way', 'mlp')

    def read(b, c):
        st = {'bev': b, 'cmd': cmd}
        for g in READ:
            st = net.step(g, st, c)
        return st['traj']

    c = SwitchCache()
    traj = read(bev, c)
    c.freeze()
    print(f'  bev {tuple(bev.shape)} -> traj {tuple(traj.shape)}, '
          f'commanded branch = mode {cmd}')

    with torch.no_grad():
        L0 = net.latent_decoder[0]
        q = torch.randn(16, 1, C, device=dev, dtype=DT)
        passed.append(ok('frozen MHA == nn.MultiheadAttention',
                         rel(L0.attn(q, q, q, need_weights=False)[0],
                             lin_mha(L0.attn, q, q, q, SwitchCache(), 'x'))))
        f0, f1, f3 = read(bev * 0, c), read(bev * 1, c), read(bev * 3, c)
        passed.append(ok('frozen forward reproduces the real output', rel(f1, traj)))
        passed.append(ok('frozen map is exactly affine in bev',
                         rel(f3 - f0, 3 * (f1 - f0))))
        bias = f0[0, cmd].reshape(-1)

    tgt = traj[0, cmd].reshape(-1)
    contrib = bev.new_zeros(HW, tgt.numel())
    for k in range(tgt.numel()):
        g, = torch.autograd.grad(tgt[k], bev, retain_graph=True)
        contrib[:, k] = (g[0] * bev[0]).sum(-1)
    passed.append(ok('sum_cells C_i + b == trajectory (residual-free)',
                     rel(tgt.detach(), contrib.sum(0) + bias)))

    # the two scalars this buys, on random weights -- shape of the output only
    p = contrib.abs().sum(1)
    p = p / p.sum()
    pr = float(p.sum() ** 2 / (p ** 2).sum())
    print(f'\n  planning support: participation ratio {pr:.0f} / {HW} '
          f'({100 * pr / HW:.1f}% of the grid)')
    print(f'  scene-independence rho = '
          f'{float(bias.norm() / (bias.norm() + contrib.sum(0).norm())):.3f}'
          f'   (share of the plan that ignores the BEV)')
    return all(passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bev', type=int, default=100)
    a = ap.parse_args()
    print(f'LMD linearisation check -- device={a.device} bev={a.bev}x{a.bev} '
          f'dtype=float64\n')
    good = run(a.device, a.bev)
    print(f'\n{"ALL CHECKS PASSED" if good else "SOME CHECKS FAILED"}')
    raise SystemExit(0 if good else 1)
