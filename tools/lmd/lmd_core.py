"""LMD (Layer-Wise Modality Decomposition, arXiv:2511.00859) primitives.

LMD's mechanism in three steps:

  1. FORWARD PASS 1 -- run the network normally and cache the state of every
     nonlinearity: ReLU/GELU gates, LayerNorm/GroupNorm mu & sigma, every
     softmax (self-attention, deformable attention weights) and the deformable
     sampling locations.
  2. LINEARISE -- treat that cached state as a CONSTANT switch. Every layer
     then becomes affine, so f(a + b) = f(a) + f(b) - f(0) holds exactly.
  3. FORWARD PASS 2 -- push each source through the frozen pipeline on its own.

The only thing that makes step 2 true is that the cached state is DETACHED.
Cache values that still carry a graph leak gradient back through mu/sigma and
the softmax, and the decomposition stops summing to the output -- see
`verify_lmd_linearisation.py`, which fails at 4e-1 relative error without the
detach and passes at 1e-15 with it.

Two ways to read the resulting linear map, and this codebase needs both:

  * FORWARD (pass 2): cheap when there are FEW sources and MANY outputs.
    Used upstream -- 10 input sources -> a [10000, 256] BEV.
  * ADJOINT (grad x input on the frozen graph): cheap when there are MANY
    sources and FEW outputs. Used downstream -- 10000 BEV cells -> 12 waypoint
    numbers. On a frozen-switch graph, grad x input IS the exact contribution,
    not a first-order approximation.
"""
import math

import torch
import torch.nn.functional as F


class SwitchCache:
    """Pass-1 state store. Values are detached on the way in; see module docstring."""

    def __init__(self):
        self.d = {}
        self.frozen = False

    def get(self, key, fn):
        if self.frozen:
            return self.d[key]
        v = fn()
        v = tuple(t.detach() for t in v) if isinstance(v, tuple) else v.detach()
        self.d[key] = v
        return v

    def freeze(self):
        self.frozen = True
        return self


# --------------------------------------------------------------- linearised ops
def lin_layernorm(ln, x, cache, key):
    mu, rstd = cache.get(key, lambda: (
        x.mean(-1, keepdim=True),
        1.0 / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)))
    return (x - mu) * rstd * ln.weight + ln.bias


def lin_groupnorm1(gn, x, cache, key):
    """GroupNorm(num_groups=1) over [N, C, L] -- TokenLearner/TokenFuser use this."""
    mu, rstd = cache.get(key, lambda: (
        x.mean(dim=(1, 2), keepdim=True),
        1.0 / torch.sqrt(x.var(dim=(1, 2), unbiased=False, keepdim=True) + gn.eps)))
    return (x - mu) * rstd * gn.weight.view(1, -1, 1) + gn.bias.view(1, -1, 1)


def lin_gelu(x, cache, key):
    """Exact GELU is literally x * Phi(x), so freezing Phi is exact with zero bias."""
    return x * cache.get(key, lambda: 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))))


def lin_relu(x, cache, key):
    return x * cache.get(key, lambda: (x > 0).to(x.dtype))


def lin_softmax(x, dim, cache, key):
    return cache.get(key, lambda: x.softmax(dim=dim))


def lin_mha(mha, q, k, v, cache, key):
    """nn.MultiheadAttention (batch_first=False) with the attention map frozen.

    Verified against torch's own kernel to 6e-16 in verify_lmd_linearisation.py.
    """
    L, N, E = q.shape
    S = k.shape[0]
    H = mha.num_heads
    hd = E // H
    W, b = mha.in_proj_weight, mha.in_proj_bias
    qq = F.linear(q, W[:E], b[:E]).reshape(L, N * H, hd).transpose(0, 1) / math.sqrt(hd)
    kk = F.linear(k, W[E:2 * E], b[E:2 * E]).reshape(S, N * H, hd).transpose(0, 1)
    vv = F.linear(v, W[2 * E:], b[2 * E:]).reshape(S, N * H, hd).transpose(0, 1)
    A = lin_softmax(torch.bmm(qq, kk.transpose(1, 2)), -1, cache, key)   # FROZEN
    o = torch.bmm(A, vv).transpose(0, 1).reshape(L, N, E)
    return F.linear(o, mha.out_proj.weight, mha.out_proj.bias)


def lin_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
    """Deformable attention with locations AND weights already frozen.

    Both come off the query, which is why they are cached rather than recomputed:
    once they are constants, bilinear `grid_sample` plus a weighted sum is linear
    in `value`, which is the image feature. That is what makes per-camera
    attribution exact.
    """
    bs, _, n_head, hd = value.shape
    _, n_q, _, n_lvl, n_pt, _ = sampling_locations.shape
    vlist = value.split([int(h) * int(w) for h, w in spatial_shapes], dim=1)
    grids = 2 * sampling_locations - 1
    out = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        v = (vlist[lvl].flatten(2).transpose(1, 2)
             .reshape(bs * n_head, hd, int(h), int(w)))
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        out.append(F.grid_sample(v, g, mode='bilinear',
                                 padding_mode='zeros', align_corners=False))
    a = attention_weights.transpose(1, 2).reshape(bs * n_head, 1, n_q, n_lvl * n_pt)
    o = (torch.stack(out, -2).flatten(-2) * a).sum(-1)
    return o.view(bs, n_head * hd, n_q).transpose(1, 2)


# --------------------------------------------------------------- readers
def decompose_forward(fn, sources, cache):
    """LMD pass 2. `sources` maps name -> tensor; `fn` takes that dict.

    Returns (parts, bias) with  sum(parts.values()) + bias == fn(sources)  exactly.
    Cheap when |sources| is small and the output is large.
    """
    zeros = {k: torch.zeros_like(v) for k, v in sources.items()}
    with torch.no_grad():
        bias = fn(zeros)
        parts = {}
        for name in sources:
            only = dict(zeros)
            only[name] = sources[name]
            parts[name] = fn(only) - bias
    return parts, bias


def decompose_adjoint(outputs, source, bias_value):
    """Adjoint read of the same linear map: per-element contribution of `source`.

    `outputs` is a 1-D tensor of scalars on the FROZEN graph, `source` the input
    tensor (requires_grad). Returns [numel(source_rows), numel(outputs)] where
    row i is source row i's exact contribution. Cheap when the output is small.
    """
    rows = source.shape[-2]
    contrib = source.new_zeros(rows, outputs.numel())
    for k in range(outputs.numel()):
        g, = torch.autograd.grad(outputs[k], source, retain_graph=True)
        contrib[:, k] = (g.reshape(-1, source.shape[-1])
                         * source.reshape(-1, source.shape[-1])).sum(-1)
    return contrib, bias_value
