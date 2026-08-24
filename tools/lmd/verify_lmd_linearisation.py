"""Does LMD's decomposition actually hold on SSR/PARA-SSR? -- numerical proof.

LMD (arXiv:2511.00859) decomposes a fused BEV feature into per-SENSOR parts.
PARA-SSR is camera-only, so the interesting axes are different, but the machinery
transfers verbatim as long as every operation on the path is either (a) already
linear, (b) constant w.r.t. the thing being decomposed, or (c) one of the
nonlinearities LMD freezes. This script checks that claim end to end.

Three checks, mirroring the three things the analysis needs:

  UP       image features -> bev_embed, decomposed into the 10 ADDITIVE SOURCES
           that BEVFormerEncoder actually mixes: 6 cameras, prev_bev, the learned
           BEV query prior, the can_bus ego signal, and the BEV positional code.
           Read forward (LMD pass 2): few sources, huge output.

  DOWN     bev_embed -> ego_fut_preds, decomposed into the 10,000 BEV CELLS.
           Read adjoint (grad x input on the frozen graph): many sources, 12
           outputs. This is what defines "planning-centric BEV".

  COMPOSE  both at once: source -> BEV -> trajectory. Both stages are affine
           under the same frozen switches, so they compose, and the planned
           trajectory splits exactly over the input sources.

The modules here REPLICATE the op stack of the real code rather than importing
it, because mmcv-full is not installed on this box. Every op and its wiring is
cited against the file it comes from. Run it inside the `ssr` env against the
real modules before trusting the numbers on a checkpoint.

    python3 tools/lmd/verify_lmd_linearisation.py
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmd_core import (SwitchCache, decompose_adjoint, decompose_forward,
                      lin_deform_attn, lin_gelu, lin_groupnorm1, lin_layernorm,
                      lin_mha, lin_relu, lin_softmax)

DT = torch.float64      # so "residual == 0" is a claim and not a rounding artefact


def rel(a, b):
    d = (a - b).abs().max()
    s = a.abs().max().clamp(min=1e-30)
    return float(d), float(d / s)


def ok(name, r, tol=1e-10):
    print(f'  {"PASS" if r < tol else "FAIL":4s}  {name:52s} rel {r:.2e}')
    return r < tol


# ============================================================ DOWN: BEV -> plan
class MlpBlock(nn.Module):                                  # tokenlearner.py:8
    def __init__(s, i, m, o):
        super().__init__(); s.fc1, s.fc2 = nn.Linear(i, m), nn.Linear(m, o)
    def forward(s, x, c, k):
        return s.fc2(lin_gelu(s.fc1(x), c, k + '.gelu'))


class TokenLearnerV11(nn.Module):                           # tokenlearner.py:25
    def __init__(s, n, ch, bott=64):
        super().__init__()
        s.n, s.layer_norm = n, nn.GroupNorm(1, ch, eps=1e-6)
        s.mlp = MlpBlock(ch, bott, n)
    def forward(s, x, c, k='tl'):
        h = lin_groupnorm1(s.layer_norm, x.permute(0, 2, 1), c, k + '.gn').permute(0, 2, 1)
        sel = lin_softmax(s.mlp(h, c, k + '.mlp').view(x.shape[0], -1, s.n).permute(0, 2, 1),
                          -1, c, k + '.sm')
        return torch.einsum('bsi,bic->bsc', sel, x), sel


class DecoderLayer(nn.Module):                              # mmcv BaseTransformerLayer
    """('self_attn'|'cross_attn', 'norm', 'ffn', 'norm') -- SSR_e2e.py:80-105."""
    def __init__(s, d=256, ff=512, h=8):
        super().__init__()
        s.attn = nn.MultiheadAttention(d, h)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.f1, s.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
    def forward(s, q, k, v, qp, kp, c, key):
        x = q + lin_mha(s.attn, q + qp, k + kp, v, c, key + '.attn')   # value gets NO pos
        x = lin_layernorm(s.n1, x, c, key + '.n1')
        x = x + s.f2(lin_relu(s.f1(x), c, key + '.ffn'))
        return lin_layernorm(s.n2, x, c, key + '.n2')


class PlanPath(nn.Module):
    """para_ssr_head.py:186-232 -- bev_embed -> ego_fut_preds."""
    def __init__(s, C=256, HW=10000, n_scenes=16, fut_ts=6, modes=3):
        super().__init__()
        s.C, s.fut_ts, s.modes = C, fut_ts, modes
        s.navi_embedding = nn.Embedding(3, C)
        s.se_reduce, s.se_expand = nn.Linear(C, C), nn.Linear(C, C)
        s.pos_embd = nn.Parameter(torch.randn(1, HW, C) * .02)
        s.tokenlearner = TokenLearnerV11(n_scenes, C * 2)
        s.latent_decoder = nn.ModuleList([DecoderLayer(C, C * 2) for _ in range(3)])
        s.way_decoder = nn.ModuleList([DecoderLayer(C, C * 2)])
        s.way_point = nn.Embedding(modes * fut_ts, C * 2)
        s.ego = nn.ModuleList([nn.Linear(C, C), nn.Linear(C, C), nn.Linear(C, 2)])

    def forward(s, bev, cmd_idx, c):
        B = bev.shape[0]
        # SELayer: the gate is built from the nav embedding ONLY, so w.r.t. bev
        # it is a constant vector -- exactly linear, nothing to freeze.
        navi = s.navi_embedding.weight[cmd_idx][None, None]
        gate = torch.sigmoid(s.se_expand(F.relu(s.se_reduce(navi))))
        x = torch.cat((bev * gate, s.pos_embd.expand(B, -1, -1)), -1)
        tok, sel = s.tokenlearner(x, c)
        lq, lp = torch.split(tok.permute(1, 0, 2), s.C, dim=2)
        for i, L in enumerate(s.latent_decoder):
            lq = L(lq, lq, lq, lp, lp, c, f'lat{i}')
        wpp, wpq = torch.split(s.way_point.weight, s.C, dim=1)
        wpp = wpp.unsqueeze(1).expand(-1, B, -1)
        wpq = wpq.unsqueeze(1).expand(-1, B, -1)
        for i, L in enumerate(s.way_decoder):
            wpq = L(wpq, lq, lq, wpp, lp, c, f'way{i}')
        h = lin_relu(s.ego[0](wpq), c, 'ego.r0')
        h = lin_relu(s.ego[1](h), c, 'ego.r1')
        return s.ego[2](h).permute(1, 0, 2).view(B, s.modes, s.fut_ts, 2), sel


# ============================================================ UP: images -> BEV
class TemporalSelfAttn(nn.Module):                          # temporal_self_attention.py
    def __init__(s, C=256, h=8, npt=4):
        super().__init__()
        s.h, s.npt, s.C = h, npt, C
        s.value_proj, s.output_proj = nn.Linear(C, C), nn.Linear(C, C)
    def forward(s, q, prev, cur, qpos, ref, shapes, c, key):
        # value queue is [prev, cur]; offsets/weights come off the query -> frozen
        B, Q, C = q.shape
        val = s.value_proj(torch.stack([prev, cur], 1).reshape(B * 2, Q, C))
        val = val.reshape(B * 2, Q, s.h, C // s.h)
        loc, w = c.get(key + '.state', lambda: (
            torch.rand(B * 2, Q, s.h, 1, s.npt, 2, device=q.device, dtype=q.dtype),
            torch.rand(B * 2, Q, s.h, 1, s.npt, device=q.device, dtype=q.dtype).softmax(-1)))
        o = lin_deform_attn(val, shapes, loc, w).reshape(B, 2, Q, C).mean(1)
        return q + s.output_proj(o)                                   # residual


class SpatialCrossAttn(nn.Module):                          # spatial_cross_attention.py:120-183
    def __init__(s, C=256, h=8, npt=8, ncam=6):
        super().__init__()
        s.h, s.npt, s.ncam, s.C = h, npt, ncam, C
        s.value_proj, s.output_proj = nn.Linear(C, C), nn.Linear(C, C)
    def forward(s, q, feats, idx, count, shapes, c, key):
        # feats: list of [B, L, C], one per camera. Each camera scatter-ADDS into
        # `slots` (line 175-177), then /count (geometry-only) then output_proj.
        # That additive scatter is why per-camera attribution is exact.
        B, Q, C = q.shape
        slots = torch.zeros_like(q)
        for i in range(s.ncam):
            v = s.value_proj(feats[i]).reshape(B, -1, s.h, C // s.h)
            n = idx[i].numel()
            loc, w = c.get(f'{key}.cam{i}', lambda: (
                torch.rand(B, n, s.h, 1, s.npt, 2, device=q.device, dtype=q.dtype),
                torch.rand(B, n, s.h, 1, s.npt, device=q.device, dtype=q.dtype).softmax(-1)))
            slots = slots.index_add(1, idx[i], lin_deform_attn(v, shapes, loc, w))
        return q + s.output_proj(slots / count[None, :, None])         # residual


class BEVEncoder(nn.Module):
    """encoder.py BEVFormerLayer x3, operation_order
    ('self_attn','norm','cross_attn','norm','ffn','norm').  bev_pos enters only
    through self_attn's q/k (encoder.py:366-367); cross_attn's query_pos is None."""
    def __init__(s, C=256, L=3):
        super().__init__()
        s.tsa = nn.ModuleList([TemporalSelfAttn(C) for _ in range(L)])
        s.sca = nn.ModuleList([SpatialCrossAttn(C) for _ in range(L)])
        s.n = nn.ModuleList([nn.LayerNorm(C) for _ in range(3 * L)])
        s.f1 = nn.ModuleList([nn.Linear(C, C * 2) for _ in range(L)])
        s.f2 = nn.ModuleList([nn.Linear(C * 2, C) for _ in range(L)])

    def forward(s, bev_q, bev_pos, prev_bev, feats, idx, count, shapes, bshapes, c):
        x = bev_q
        for i in range(len(s.tsa)):
            x = s.tsa[i](x + bev_pos, prev_bev, x, bev_pos, None, bshapes, c, f'l{i}.tsa')
            x = lin_layernorm(s.n[3 * i], x, c, f'l{i}.n0')
            x = s.sca[i](x, feats, idx, count, shapes, c, f'l{i}.sca')
            x = lin_layernorm(s.n[3 * i + 1], x, c, f'l{i}.n1')
            x = x + s.f2[i](lin_relu(s.f1[i](x), c, f'l{i}.ffn'))
            x = lin_layernorm(s.n[3 * i + 2], x, c, f'l{i}.n2')
        return x


# ============================================================ checks
def run(dev, bev_hw, n_cam, feat_hw):
    torch.manual_seed(0)
    C, HW = 256, bev_hw * bev_hw
    FL = feat_hw[0] * feat_hw[1]
    shapes = torch.tensor([list(feat_hw)])
    bshapes = torch.tensor([[bev_hw, bev_hw]])
    passed = []

    # ---------------------------------------------------------------- DOWN
    print('\nDOWN  bev_embed -> ego_fut_preds, split over 10,000 BEV cells')
    plan = PlanPath(C, HW).to(dev).to(DT).eval()
    bev = (torch.randn(1, HW, C, device=dev, dtype=DT) * .5).requires_grad_(True)
    cmd = 2
    c = SwitchCache()
    traj, _ = plan(bev, cmd, c)
    c.freeze()

    with torch.no_grad():
        L0 = plan.latent_decoder[0]
        qq = torch.randn(16, 1, C, device=dev, dtype=DT)
        r = rel(L0.attn(qq, qq, qq, need_weights=False)[0],
                lin_mha(L0.attn, qq, qq, qq, SwitchCache(), 'x'))[1]
        passed.append(ok('frozen MHA == nn.MultiheadAttention', r))

        f0, _ = plan(bev * 0, cmd, c)
        f1, _ = plan(bev * 1, cmd, c)
        f3, _ = plan(bev * 3, cmd, c)
        passed.append(ok('frozen forward reproduces the real output', rel(f1, traj)[1]))
        passed.append(ok('frozen map is exactly affine in bev',
                         rel(f3 - f0, 3 * (f1 - f0))[1]))
        bias = f0[0, cmd].reshape(-1)

    tgt = traj[0, cmd].reshape(-1)
    contrib, _ = decompose_adjoint(tgt, bev, bias)
    passed.append(ok('sum_cells C_i + b == trajectory  (residual-free)',
                     rel(tgt.detach(), contrib.sum(0) + bias)[1]))

    # ---------------------------------------------------------------- UP
    print('\nUP    image features -> bev_embed, split over 10 additive sources')
    enc = BEVEncoder(C).to(dev).to(DT).eval()
    idx = [torch.randperm(HW, device=dev)[:HW // 2].sort().values for _ in range(n_cam)]
    cnt = torch.zeros(HW, device=dev, dtype=DT)
    for i in idx:
        cnt.index_add_(0, i, torch.ones_like(i, dtype=DT))
    cnt.clamp_(min=1.0)

    src = {f'cam{i}': torch.randn(1, FL, C, device=dev, dtype=DT) * .3
           for i in range(n_cam)}
    src['prev_bev'] = torch.randn(1, HW, C, device=dev, dtype=DT) * .3
    src['bev_prior'] = torch.randn(1, HW, C, device=dev, dtype=DT) * .3   # nn.Embedding
    src['can_bus'] = torch.randn(1, 1, C, device=dev, dtype=DT) * .3      # broadcast
    src['bev_pos'] = torch.randn(1, HW, C, device=dev, dtype=DT) * .3

    def build(sr):
        # SSR_transformer.py:270 -- bev_queries = prior + can_bus, purely additive
        q = sr['bev_prior'] + sr['can_bus'].expand(-1, HW, -1)
        return enc(q, sr['bev_pos'], sr['prev_bev'],
                   [sr[f'cam{i}'] for i in range(n_cam)],
                   idx, cnt, shapes, bshapes, cu)

    cu = SwitchCache()
    bev_up = build(src)
    cu.freeze()
    parts, b_up = decompose_forward(build, src, cu)
    total = sum(parts.values()) + b_up
    passed.append(ok('sum_sources h^s + b == bev_embed  (residual-free)',
                     rel(bev_up, total)[1]))

    share = {k: float(v.abs().sum()) for k, v in parts.items()}
    share['<bias>'] = float(b_up.abs().sum())
    tot = sum(share.values())
    print('\n  per-source share of |bev_embed| (random weights -- shape of the '
          'output, not a result):')
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f'      {k:11s} {100 * v / tot:5.1f}%')

    # ---------------------------------------------------------------- COMPOSE
    print('\nCOMPOSE  source -> bev -> trajectory')
    cp = SwitchCache()
    traj2, _ = plan(build(src), cmd, cp)
    cp.freeze()
    with torch.no_grad():
        zero = {k: torch.zeros_like(v) for k, v in src.items()}
        b_c = plan(build(zero), cmd, cp)[0][0, cmd].reshape(-1)
        routed = {}
        for name in src:
            only = dict(zero); only[name] = src[name]
            routed[name] = plan(build(only), cmd, cp)[0][0, cmd].reshape(-1) - b_c
    passed.append(ok('sum_sources (routed to plan) + b == trajectory',
                     rel(traj2[0, cmd].reshape(-1).detach(),
                         sum(routed.values()) + b_c)[1]))
    return all(passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bev', type=int, default=100)
    ap.add_argument('--cams', type=int, default=6)
    a = ap.parse_args()
    print(f'LMD linearisation check -- device={a.device} bev={a.bev}x{a.bev} '
          f'cams={a.cams} dtype=float64')
    good = run(a.device, a.bev, a.cams, (24, 40))
    print(f'\n{"ALL CHECKS PASSED" if good else "SOME CHECKS FAILED"}')
    raise SystemExit(0 if good else 1)
