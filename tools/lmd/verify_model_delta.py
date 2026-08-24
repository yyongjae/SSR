"""Attribute the aux-vs-no-aux BEV difference, LMD-style. Numerical proof.

The question is "what did the auxiliary tasks do to the BEV feature", which is a
difference between two TRAINED MODELS, not between two input streams. LMD as
published decomposes over additive INPUTS, so it does not apply as written.

What does transfer is the property that makes LMD work: freeze every
nonlinearity at its operating point and the network becomes an exactly affine
map. Then each weight group g is a fixed affine operator, and the difference
between two models telescopes:

    A_L...A_1 - P_L...P_1  =  sum_g  A_L...A_{g+1} (A_g - P_g) P_{g-1}...P_1

so with G_g := "groups 1..g from P, groups g+1..L from A",

    delta_g = G_{g-1} - G_g          and      sum_g delta_g = out_A - out_P

exactly, with no residual. delta_g is the effect on the BEV (or on the planned
trajectory) of switching group g alone from the planning-only model's weights to
the aux-trained model's. Call it Layer-wise MODEL Decomposition: same machinery,
the axis swapped from modality to supervision.

Checks here:
  1. each model's frozen forward reproduces its own output           (LMD holds)
  2. telescoping sums to the model difference, at BEV and at the trajectory
  3. the same run in the reverse order, to expose path dependence
  4. exact Shapley over the 8 groups, which removes the ordering entirely
  5. the signed split of the trajectory change against the GT error direction --
     how much of what aux did to the BEV helped planning and how much hurt it

    python3 tools/lmd/verify_model_delta.py [--shapley]
"""
import argparse
import copy
import itertools
import math

import torch

from lmd_core import SwitchCache
from replica import Pipeline, hybrid

DT = torch.float64


def rel(a, b):
    d = (a - b).abs().max()
    return float(d / a.abs().max().clamp(min=1e-30))


def ok(name, r, tol=1e-9):
    print(f'  {"PASS" if r < tol else "FAIL":4s}  {name:54s} rel {r:.2e}')
    return r < tol


def make_state(dev, HW, FL, C, n_cam, feat_hw, bev_hw):
    g = torch.Generator(device='cpu').manual_seed(7)
    r = lambda *sh: torch.randn(*sh, generator=g, dtype=DT).to(dev)
    idx = [torch.randperm(HW, generator=g)[:HW // 2].sort().values.to(dev)
           for _ in range(n_cam)]
    cnt = torch.zeros(HW, device=dev, dtype=DT)
    for i in idx:
        cnt.index_add_(0, i, torch.ones_like(i, dtype=DT))
    return dict(
        img=[r(1, FL, C) * .3 for _ in range(n_cam)],
        can_bus=r(1, 18) * .3,
        prev=r(1, HW, C) * .3,
        idx=idx, cnt=cnt.clamp(min=1.0),
        shapes=torch.tensor([list(feat_hw)]),
        bshapes=torch.tensor([[bev_hw, bev_hw]]),
        cmd=2)


def build_pair(dev, HW, FL, C, n_cam, drift):
    """Model P (planning-only) and model A (aux-trained), same init + a delta.

    Real runs would be two checkpoints. Here A is P plus a per-group perturbation,
    which is the honest stand-in: it puts the two models in the SAME BASIN, which
    is exactly the condition the analysis needs and which two independently
    trained checkpoints are not guaranteed to satisfy (see report #09 section 7).
    """
    torch.manual_seed(0)
    P = Pipeline(C, HW, FL, n_cam).to(dev).to(DT).eval()
    A = copy.deepcopy(P)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in A.parameters():
            p.add_(torch.randn_like(p) * drift * p.std().clamp(min=1e-6))
    return {'P': P, 'A': A}


def frozen_pair(models, st):
    """One pass per model to cache its switches, then freeze both."""
    caches, outs = {}, {}
    for tag, m in models.items():
        c = SwitchCache()
        with torch.no_grad():
            outs[tag] = m(dict(st), c)
        caches[tag] = c.freeze()
    return caches, outs


def telescope(models, caches, st, order):
    """delta_g for g in `order`; groups already switched stay switched."""
    groups = list(order)
    assign = {g: groups[0] and 'A' for g in Pipeline.GROUPS}
    assign = {g: 'A' for g in Pipeline.GROUPS}
    with torch.no_grad():
        prev = hybrid(models, caches, assign, dict(st))
        deltas = {}
        for g in groups:
            assign = dict(assign); assign[g] = 'P'
            cur = hybrid(models, caches, assign, dict(st))
            deltas[g] = {'bev': prev['bev'] - cur['bev'],
                         'traj': prev['traj'] - cur['traj']}
            prev = cur
    return deltas


def shapley(models, caches, st, key='traj'):
    """Exact Shapley over the 8 groups: 2^8 coalitions, no ordering to pick.

    With the switches frozen the output is multilinear in the group operators, so
    these are the ordinary Shapley values of that function and the efficiency
    axiom (sum == out_A - out_P) is a real check, not a definition.
    """
    G = list(Pipeline.GROUPS)
    n = len(G)
    cache_v = {}

    def v(S):
        k = frozenset(S)
        if k not in cache_v:
            assign = {g: ('A' if g in k else 'P') for g in G}
            with torch.no_grad():
                cache_v[k] = hybrid(models, caches, assign, dict(st))[key].clone()
        return cache_v[k]

    phi = {}
    for g in G:
        rest = [x for x in G if x != g]
        acc = None
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(rest, k):
                d = (v(list(S) + [g]) - v(list(S))) * w
                acc = d if acc is None else acc + d
        phi[g] = acc
    return phi, len(cache_v)


def run(dev, bev_hw, n_cam, feat_hw, drift, do_shapley):
    C = 256
    HW, FL = bev_hw * bev_hw, feat_hw[0] * feat_hw[1]
    models = build_pair(dev, HW, FL, C, n_cam, drift)
    st = make_state(dev, HW, FL, C, n_cam, feat_hw, bev_hw)
    caches, outs = frozen_pair(models, st)
    passed = []

    print(f'\nmodels: P = planning-only, A = aux-trained (weight drift {drift})')
    print(f'groups: {" ".join(Pipeline.GROUPS)}')

    # 1 -- LMD still holds per model
    with torch.no_grad():
        for tag in ('P', 'A'):
            ref = outs[tag]['traj']
            again = hybrid(models, caches, {g: tag for g in Pipeline.GROUPS},
                           dict(st))['traj']
            passed.append(ok(f'frozen forward reproduces model {tag}', rel(ref, again)))

    d_bev = outs['A']['bev'] - outs['P']['bev']
    d_traj = outs['A']['traj'] - outs['P']['traj']
    print(f'\n  |bev_A - bev_P| / |bev_P| = '
          f'{float(d_bev.norm() / outs["P"]["bev"].norm()):.4f}'
          f'    |traj_A - traj_P| = {float(d_traj.norm()):.4f}')

    # 2 -- telescoping, forward order
    print('\nLayer-wise MODEL decomposition')
    fwd = telescope(models, caches, st, Pipeline.GROUPS)
    passed.append(ok('sum_g delta_g == bev_A - bev_P',
                     rel(d_bev, sum(v['bev'] for v in fwd.values()))))
    passed.append(ok('sum_g delta_g == traj_A - traj_P',
                     rel(d_traj, sum(v['traj'] for v in fwd.values()))))

    # 3 -- reverse order, to show the decomposition is exact but path-dependent
    rev = telescope(models, caches, st, tuple(reversed(Pipeline.GROUPS)))
    passed.append(ok('reverse order also sums exactly',
                     rel(d_traj, sum(v['traj'] for v in rev.values()))))
    spread = max(float((fwd[g]['traj'] - rev[g]['traj']).norm()) for g in Pipeline.GROUPS)
    print(f'  NOTE  per-group values differ between the two orders by up to '
          f'{spread:.4f} ({100 * spread / float(d_traj.norm()):.0f}% of |d_traj|)'
          f' -- exact but path-dependent')

    # 4 -- Shapley removes the ordering
    if do_shapley:
        phi, n_eval = shapley(models, caches, st)
        passed.append(ok(f'sum_g shapley_g == traj_A - traj_P ({n_eval} coalitions)',
                         rel(d_traj, sum(phi.values()))))
    else:
        phi = None

    # ---------------------------------------------------------------- readout
    print('\n  where aux rewrote the plan  (share of |delta_traj|, commanded branch)')
    cmd = st['cmd']
    tot = sum(float(v['traj'][0, cmd].norm()) for v in fwd.values())
    rows = [(g, float(fwd[g]['traj'][0, cmd].norm()),
             float(rev[g]['traj'][0, cmd].norm()),
             float(phi[g][0, cmd].norm()) if phi else float('nan'))
            for g in Pipeline.GROUPS]
    print(f'      {"group":8s} {"fwd order":>10s} {"rev order":>10s} {"shapley":>10s}')
    for g, a, b, c in rows:
        sh = f'{100*c/tot:9.1f}%' if phi else '         -'
        print(f'      {g:8s} {100*a/tot:9.1f}% {100*b/tot:9.1f}% {sh}')

    # 6 -- THE ONE THAT ANSWERS THE QUESTION.
    # Split the groups into the ones that BUILD the BEV and the ones that READ
    # it. Hold the reader at A and switch only the builders: the resulting
    # trajectory change is, exactly, the effect of the aux-induced BEV change,
    # and model A's frozen planner is an affine map so it splits per BEV cell.
    print('\n  aux-induced BEV change, routed through model A\'s planner')
    BUILD = ('embed', 'enc0', 'enc1', 'enc2')
    READ = ('tokenl', 'latent', 'way', 'mlp')
    with torch.no_grad():
        st_AA = hybrid(models, caches, {g: 'A' for g in Pipeline.GROUPS}, dict(st))
        st_PA = hybrid(models, caches,
                       {**{g: 'P' for g in BUILD}, **{g: 'A' for g in READ}},
                       dict(st))
    d_traj_bev = (st_AA['traj'] - st_PA['traj'])[0, cmd].reshape(-1)
    passed.append(ok('builder groups alone == BEV-change effect on the plan',
                     rel(d_traj_bev,
                         sum(fwd[g]['traj'] for g in BUILD)[0, cmd].reshape(-1))))

    # per-cell: run A's frozen reader on bev_P, differentiate, multiply by d_bev
    bevP = st_PA['bev'].clone().requires_grad_(True)
    stx = dict(st_PA); stx['bev'] = bevP
    for g in READ:
        stx = models['A'].step(g, stx, caches['A'])
    out = stx['traj'][0, cmd].reshape(-1)
    sig = bevP.new_zeros(bevP.shape[1], out.numel())
    for k in range(out.numel()):
        gr, = torch.autograd.grad(out[k], bevP, retain_graph=True)
        sig[:, k] = (gr[0] * d_bev[0]).sum(-1)
    passed.append(ok('sum_cells J_A,i . d_bev_i == that same change',
                     rel(d_traj_bev, sig.sum(0))))

    sd = sig @ (lambda e: e / e.norm())(
        outs['P']['traj'][0, cmd].reshape(-1) - (
            outs['P']['traj'][0, cmd].reshape(-1) + torch.randn(
                12, device=dev, dtype=DT,
                generator=torch.Generator(device=dev).manual_seed(3)) * .1))
    nh, nk = int((sd < 0).sum()), int((sd > 0).sum())
    print(f'      helped   {nh:6d} cells   {float(sd[sd < 0].sum()):+.5f}')
    print(f'      HURT     {nk:6d} cells   {float(sd[sd > 0].sum()):+.5f}   '
          f'<- "방해되는 BEV"')
    print(f'      net {float(sd.sum()):+.5f};  top 1% of cells carry '
          f'{100 * float(sd.abs().topk(max(1, HW // 100)).values.sum() / sd.abs().sum()):.0f}%'
          f' of the magnitude')

    # 5 -- did the rewrite help or hurt planning?
    gt = outs['P']['traj'][0, cmd].reshape(-1) + torch.randn(
        12, device=dev, dtype=DT, generator=torch.Generator(device=dev).manual_seed(3)) * .1
    eP = outs['P']['traj'][0, cmd].reshape(-1) - gt
    e_hat = eP / eP.norm()
    print(f'\n  signed against the GT error direction of model P '
          f'(|e_P| = {float(eP.norm()):.4f})')
    src = phi if phi else fwd
    tot_signed = 0.0
    for g in Pipeline.GROUPS:
        t = (src[g] if phi else src[g]['traj'])[0, cmd].reshape(-1)
        s = float(t @ e_hat)
        tot_signed += s
        print(f'      {g:8s} {s:+.5f}   {"HURT" if s > 0 else "helped"}')
    lhs = float((outs['A']['traj'][0, cmd].reshape(-1) - gt) @ e_hat)
    passed.append(ok('sum_g signed_g + |e_P| == projected error of A',
                     abs(lhs - (tot_signed + float(eP.norm()))) /
                     max(abs(lhs), 1e-30)))
    return all(passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bev', type=int, default=100)
    ap.add_argument('--cams', type=int, default=6)
    ap.add_argument('--drift', type=float, default=0.05)
    ap.add_argument('--shapley', action='store_true',
                    help='exact Shapley over the 8 groups (256 hybrid forwards)')
    a = ap.parse_args()
    print(f'Layer-wise MODEL decomposition check -- device={a.device} '
          f'bev={a.bev}x{a.bev} cams={a.cams} dtype=float64')
    good = run(a.device, a.bev, a.cams, (24, 40), a.drift, a.shapley)
    print(f'\n{"ALL CHECKS PASSED" if good else "SOME CHECKS FAILED"}')
    raise SystemExit(0 if good else 1)
