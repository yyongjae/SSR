"""Replicas of PARA-SSR's op stack, wired as an ordered chain of WEIGHT GROUPS.

Every module mirrors the real file it is named after (cited inline) but is built
out of `lmd_core`'s frozen-switch ops, so a forward through this chain is exactly
affine in its activations once the switches are cached.

The chain is exposed as `Pipeline.GROUPS` rather than one monolithic `forward`
because the aux-vs-no-aux analysis needs HYBRID forwards: group g's weights from
model A, the rest from model P.  See `verify_model_delta.py`.

mmcv-full is not installed on the analysis box, hence replicas rather than
imports.  Re-run the checks inside the `ssr` env against the real modules before
trusting any number measured on a checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmd_core import (lin_deform_attn, lin_gelu, lin_groupnorm1, lin_layernorm,
                      lin_mha, lin_relu, lin_softmax)


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


class TemporalSelfAttn(nn.Module):                          # temporal_self_attention.py
    def __init__(s, C=256, h=8, npt=4):
        super().__init__()
        s.h, s.npt, s.C = h, npt, C
        s.value_proj, s.output_proj = nn.Linear(C, C), nn.Linear(C, C)
        # offsets/weights are predicted from the query, exactly as in
        # temporal_self_attention.py:211-216 (query = cat[history, current])
        s.sampling_offsets = nn.Linear(2 * C, h * npt * 2)
        s.attention_weights = nn.Linear(2 * C, h * npt)
    def forward(s, q, prev, cur, shapes, c, key):
        # value queue is [prev, cur] (encoder.py:207); offsets/weights come off
        # the query, so they are cached state rather than recomputed.
        B, Q, C = q.shape
        val = s.value_proj(torch.stack([prev, cur], 1).reshape(B * 2, Q, C))
        val = val.reshape(B * 2, Q, s.h, C // s.h)
        oi = torch.cat([prev, q], -1)
        loc, w = c.get(key + '.state', lambda: (
            torch.sigmoid(s.sampling_offsets(oi)).view(B, Q, s.h, 1, s.npt, 2)
                .repeat(2, 1, 1, 1, 1, 1),
            s.attention_weights(oi).view(B, Q, s.h, 1, s.npt).softmax(-1)
                .repeat(2, 1, 1, 1, 1)))
        o = lin_deform_attn(val, shapes, loc, w).reshape(B, 2, Q, C).mean(1)
        return q + s.output_proj(o)                                   # residual


class SpatialCrossAttn(nn.Module):                          # spatial_cross_attention.py:120
    def __init__(s, C=256, h=8, npt=8, ncam=6):
        super().__init__()
        s.h, s.npt, s.ncam, s.C = h, npt, ncam, C
        s.value_proj, s.output_proj = nn.Linear(C, C), nn.Linear(C, C)
        # spatial_cross_attention.py:345-350
        s.sampling_offsets = nn.Linear(C, h * npt * 2)
        s.attention_weights = nn.Linear(C, h * npt)
    def forward(s, q, feats, idx, count, shapes, c, key):
        # each camera scatter-ADDS into `slots` (:175-177), then /count
        # (geometry-only, :179-182), then output_proj, then + residual (:183).
        B, Q, C = q.shape
        slots = torch.zeros_like(q)
        for i in range(s.ncam):
            v = s.value_proj(feats[i]).reshape(B, -1, s.h, C // s.h)
            qi = q[:, idx[i]]
            n = idx[i].numel()
            loc, w = c.get(f'{key}.cam{i}', lambda: (
                torch.sigmoid(s.sampling_offsets(qi)).view(B, n, s.h, 1, s.npt, 2),
                s.attention_weights(qi).view(B, n, s.h, 1, s.npt).softmax(-1)))
            slots = slots.index_add(1, idx[i], lin_deform_attn(v, shapes, loc, w))
        return q + s.output_proj(slots / count[None, :, None])


class EncoderLayer(nn.Module):
    """encoder.py BEVFormerLayer, operation_order
    ('self_attn','norm','cross_attn','norm','ffn','norm').  bev_pos enters only
    through self_attn's q/k (:366-367); cross_attn's query_pos is None."""
    def __init__(s, C=256):
        super().__init__()
        s.tsa, s.sca = TemporalSelfAttn(C), SpatialCrossAttn(C)
        s.n0, s.n1, s.n2 = nn.LayerNorm(C), nn.LayerNorm(C), nn.LayerNorm(C)
        s.f1, s.f2 = nn.Linear(C, C * 2), nn.Linear(C * 2, C)
    def forward(s, bev, pos, prev, feats, idx, cnt, shapes, bshapes, c, key):
        x = s.tsa(bev + pos, prev, bev, bshapes, c, key + '.tsa')
        x = lin_layernorm(s.n0, x, c, key + '.n0')
        x = s.sca(x, feats, idx, cnt, shapes, c, key + '.sca')
        x = lin_layernorm(s.n1, x, c, key + '.n1')
        x = x + s.f2(lin_relu(s.f1(x), c, key + '.ffn'))
        return lin_layernorm(s.n2, x, c, key + '.n2')


class Pipeline(nn.Module):
    """images -> bev_embed -> ego_fut_preds, as an ordered chain of weight groups.

    Groups are the granularity at which the aux-vs-no-aux weight difference gets
    attributed.  They follow the module boundaries of the real model:

        embed   image stem, the learned BEV query prior + can_bus, bev_pos
        enc0-2  the three BEVFormerLayers of the shared BEV encoder
        tokenl  nav SE gate + TokenLearner (10,000 cells -> 16 scene tokens)
        latent  3-layer self-attention over the scene tokens
        way     1-layer cross-attention, 18 waypoint queries -> tokens
        mlp     ego_fut_decoder
    """
    GROUPS = ('embed', 'enc0', 'enc1', 'enc2', 'tokenl', 'latent', 'way', 'mlp')

    def __init__(s, C=256, HW=10000, FL=960, n_cam=6, n_scenes=16,
                 fut_ts=6, modes=3):
        super().__init__()
        s.C, s.HW, s.n_cam, s.fut_ts, s.modes = C, HW, n_cam, fut_ts, modes
        # --- embed
        s.stem = nn.ModuleList([nn.Linear(C, C) for _ in range(2)])   # stands in for ResNet+FPN
        s.bev_embedding = nn.Embedding(HW, C)
        s.can_bus_mlp = nn.Linear(18, C)
        s.bev_pos = nn.Parameter(torch.randn(1, HW, C) * .02)
        # --- encoder
        s.enc = nn.ModuleList([EncoderLayer(C) for _ in range(3)])
        # --- planner
        s.navi_embedding = nn.Embedding(3, C)
        s.se_reduce, s.se_expand = nn.Linear(C, C), nn.Linear(C, C)
        s.pos_embd = nn.Parameter(torch.randn(1, HW, C) * .02)
        s.tokenlearner = TokenLearnerV11(n_scenes, C * 2)
        s.latent_decoder = nn.ModuleList([DecoderLayer(C, C * 2) for _ in range(3)])
        s.way_decoder = nn.ModuleList([DecoderLayer(C, C * 2)])
        s.way_point = nn.Embedding(modes * fut_ts, C * 2)
        s.ego = nn.ModuleList([nn.Linear(C, C), nn.Linear(C, C), nn.Linear(C, 2)])

    # ---------------------------------------------------------------- groups
    def step(s, name, st, c):
        """Run one weight group.  `st` is the running state dict."""
        if name == 'embed':
            st = dict(st)
            st['feats'] = [lin_relu(s.stem[0](f), c, f'stem{i}') @ s.stem[1].weight.T
                           + s.stem[1].bias for i, f in enumerate(st['img'])]
            # SSR_transformer.py:270 -- bev_queries = prior + can_bus, additive
            st['bev'] = (s.bev_embedding.weight[None]
                         + s.can_bus_mlp(st['can_bus'])[:, None])
            st['pos'] = s.bev_pos.expand(st['bev'].shape[0], -1, -1)
            return st
        if name.startswith('enc'):
            st = dict(st)
            i = int(name[-1])
            st['bev'] = s.enc[i](st['bev'], st['pos'], st['prev'], st['feats'],
                                 st['idx'], st['cnt'], st['shapes'],
                                 st['bshapes'], c, f'l{i}')
            return st
        if name == 'tokenl':
            st = dict(st)
            B = st['bev'].shape[0]
            # SELayer: gate built from the nav embedding ONLY, so it is a
            # constant w.r.t. bev -- already exactly linear, nothing to freeze.
            navi = s.navi_embedding.weight[st['cmd']][None, None]
            gate = torch.sigmoid(s.se_expand(F.relu(s.se_reduce(navi))))
            x = torch.cat((st['bev'] * gate, s.pos_embd.expand(B, -1, -1)), -1)
            tok, st['token_attn'] = s.tokenlearner(x, c)
            st['lq'], st['lp'] = torch.split(tok.permute(1, 0, 2), s.C, dim=2)
            return st
        if name == 'latent':
            st = dict(st)
            lq = st['lq']
            for i, L in enumerate(s.latent_decoder):
                lq = L(lq, lq, lq, st['lp'], st['lp'], c, f'lat{i}')
            st['lq'] = lq
            return st
        if name == 'way':
            st = dict(st)
            B = st['lq'].shape[1]
            wpp, wpq = torch.split(s.way_point.weight, s.C, dim=1)
            wpp = wpp.unsqueeze(1).expand(-1, B, -1)
            wpq = wpq.unsqueeze(1).expand(-1, B, -1)
            for i, L in enumerate(s.way_decoder):
                wpq = L(wpq, st['lq'], st['lq'], wpp, st['lp'], c, f'way{i}')
            st['wq'] = wpq
            return st
        if name == 'mlp':
            st = dict(st)
            h = lin_relu(s.ego[0](st['wq']), c, 'ego.r0')
            h = lin_relu(s.ego[1](h), c, 'ego.r1')
            B = st['wq'].shape[1]
            st['traj'] = s.ego[2](h).permute(1, 0, 2).view(B, s.modes, s.fut_ts, 2)
            return st
        raise KeyError(name)

    def forward(s, st, c):
        for g in s.GROUPS:
            st = s.step(g, st, c)
        return st


def hybrid(models, caches, assign, st):
    """Forward taking each group's weights AND cached switches from assign[g].

    `models` / `caches` map a tag ('A' / 'P') to a Pipeline and its frozen
    SwitchCache.  With every switch already frozen, each group is a fixed affine
    operator, so any assignment is a well-defined composition of affine maps.
    """
    for g in Pipeline.GROUPS:
        tag = assign[g]
        st = models[tag].step(g, st, caches[tag])
    return st
