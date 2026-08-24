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


def analyse(head, tap, bev_ref, scale_check=False):
    """Per-cell contributions C_i and the bias b, for the commanded branch.

    Returns (tgt, contrib, bias, idx, affinity) where `affinity` is the relative
    error of  f(3x) - f(0) == 3*(f(x) - f(0))  under the frozen switches, or None.

    That check matters more than the sum. The sum runs 10,000 float32 terms
    through heavy cancellation, so its relative error is dominated by
    accumulation and says little about whether the linearisation is right. The
    scale identity involves no summation at all: if it holds, the map IS affine,
    and any residual left in the sum is arithmetic.
    """
    idx = cmd_index(tap['kwargs']['cmd'])
    bev = bev_ref.detach().clone().requires_grad_(True)

    rec = Recorder()
    with linearise(head, rec):
        outs = plan_from_bev(head, tap, bev)          # records + linearises in one pass
    traj = outs['ego_fut_preds']
    tgt = traj[0, idx].reshape(-1)

    contrib = bev.new_zeros(bev.shape[1], tgt.numel())
    for k in range(tgt.numel()):
        g, = torch.autograd.grad(tgt[k], bev, retain_graph=True)
        contrib[:, k] = (g[0] * bev[0]).sum(-1)

    rec.freeze()
    affinity = None
    with linearise(head, rec), torch.no_grad():
        def replay(x):
            # every replay forward needs its own pass: the counters are what
            # line each call up with what the recording pass stored
            rec.new_pass()
            return plan_from_bev(head, tap, x)['ego_fut_preds'][0, idx].reshape(-1)

        bias = replay(torch.zeros_like(bev))
        if scale_check:
            f1, f3 = replay(bev), replay(bev * 3.0)
            lhs, rhs = f3 - bias, 3.0 * (f1 - bias)
            # Normalise by the magnitude of the terms being differenced, not by
            # the difference. On a planner whose output is mostly the bias --
            # aux_only, where rho is 0.69 because it trained with plan=0 -- the
            # scene-dependent part is small and dividing by it reports the
            # cancellation inside this check as if it were model error. float64
            # puts both at 2e-15, so what fp32 leaves is arithmetic either way.
            scale = max(float(f3.abs().max()), float(bias.abs().max()),
                        3.0 * float(f1.abs().max()), 1e-30)
            affinity = float((lhs - rhs).abs().max() / scale)
    return tgt.detach(), contrib.detach(), bias.detach(), idx, affinity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, choices=['plan_only', 'aux_only', 'staged', 'both'])
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--fp64', action='store_true',
                    help='diagnostic: rerun the planner path in float64 on one sample. '
                         'If the residuals collapse, what fp32 leaves behind is arithmetic, '
                         'not a nonlinearity that escaped freezing. Mutates the head, so it '
                         'stops after one sample.')
    a = ap.parse_args()

    cfg_path, ckpt_path, desc = resolve(a.ckpt, a.epoch)
    print(f'E0  {a.ckpt}  ({desc})\n    cfg  {cfg_path}\n    ckpt {ckpt_path}\n')
    model, cfg = build(cfg_path, ckpt_path, a.device)
    _, loader = val_loader(cfg)
    head = model.pts_bbox_head

    def to_double(x):
        if torch.is_tensor(x):
            return x.double() if x.is_floating_point() else x
        if isinstance(x, (list, tuple)):
            return type(x)(to_double(v) for v in x)
        return x

    worst_repro = worst_sum = worst_aff = 0.0
    for i, data, tap in iter_samples(model, loader, a.samples, a.device):
        traj_ref = tap['out']['ego_fut_preds'].detach()
        bev_ref = tap['out']['bev_embed'].detach()
        if a.fp64:
            # the planner path is plain torch -- MHA, norms, GELU/ReLU, einsum,
            # Linear -- and bev_override keeps the encoder's custom CUDA ops out
            # of it, so float64 runs. force_fp32 is a passthrough here because
            # the head sets fp16_enabled = False.
            head.double()
            tap = dict(tap, args=to_double(tap['args']),
                       kwargs={k: to_double(v) for k, v in tap['kwargs'].items()})
            bev_ref, traj_ref = bev_ref.double(), traj_ref.double()
        tgt, contrib, bias, idx, aff = analyse(head, tap, bev_ref, scale_check=True)

        ref = traj_ref[0, idx].reshape(-1)
        r_repro = float((tgt - ref).abs().max() / ref.abs().max())
        # sum in float64: the 10,000 float32 terms cancel heavily, and pairwise
        # accumulation error is not evidence about the linearisation
        tot = contrib.double().sum(0)
        lhs, rhs = tgt.double(), tot + bias.double()
        r_sum = float((lhs - rhs).abs().max() / lhs.abs().max())
        # how much cancellation the sum goes through -- the factor by which
        # float32 round-off in the terms is amplified in the total
        cond = float(contrib.double().abs().sum(0).norm() / tot.norm())
        worst_repro = max(worst_repro, r_repro)
        worst_aff = max(worst_aff, aff)
        worst_sum = max(worst_sum, r_sum / max(cond, 1.0))

        p = contrib.abs().sum(1)
        p = p / p.sum()
        pr = float(p.sum() ** 2 / (p ** 2).sum())
        rho = float(bias.norm() / (bias.norm() + tot.norm()))
        print(f'  sample {i:3d}  cmd={idx}  frozen-vs-real {r_repro:.2e}  affine {aff:.2e}  '
              f'sum {r_sum:.2e} (cancel x{cond:.0f} -> {r_sum / max(cond, 1.0):.2e})  '
              f'PR {pr:6.0f}  rho {rho:.3f}')
        if a.fp64:
            print('  (float64 diagnostic -- stopping after one sample)')
            break

    good = worst_repro < a.tol and worst_aff < a.tol and worst_sum < a.tol
    print(f'\n  worst frozen-vs-real {worst_repro:.2e}   worst affine {worst_aff:.2e}'
          f'   worst cancellation-adjusted sum {worst_sum:.2e}   tol {a.tol:.0e}')
    print(f'  {"PASS" if good else "FAIL"} -- {a.ckpt}')
    raise SystemExit(0 if good else 1)


if __name__ == '__main__':
    main()
