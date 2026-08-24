"""LMD linearisation applied to the REAL mmcv/mmdet3d modules.

`lmd_core.py` proves the maths on replicas; this does it to the model that
produced the checkpoints. Same three steps as the paper (arXiv:2511.00859):

  1. forward once, caching the state of every nonlinearity;
  2. treat that state as a constant -- which in PyTorch means DETACHING it, the
     single line the whole method rests on;
  3. the network is now exactly affine, so it decomposes with no residual.

Step 1 and 2 happen in the same pass here: the cached values are detached as
they are recorded, so the graph built by the recording pass is already the
linearised one. A second pass (`Recorder.freeze()` then run with bev=0) gives
the constant term `b`.

What gets frozen, and where it lives in the model:

    nn.LayerNorm        latent/way decoder layers, MLN         mu, sigma
    nn.GroupNorm(1,C)   TokenLearner, TokenFuser               mu, sigma
    nn.GELU             TokenLearner MlpBlock                  Phi(x); exact
                                                               GELU is x*Phi(x)
    nn.ReLU             FFNs, ego_fut_decoder, SELayer, MLN     the 0/1 mask
    nn.MultiheadAttention  latent/way decoders                  attention probs
    TokenLearnerV11     the softmax over 10,000 BEV cells       `selected`
    deformable attn     aux-head decoders, BEV encoder          sampling
                                                                locations and
                                                                attention weights

nn.Sigmoid is deliberately NOT frozen: the only one on the path is SELayer's
navigation gate, which is built from the command embedding alone and is already
a constant w.r.t. the BEV.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.mmdet3d_plugin.SSR.tokenlearner import TokenLearnerV11


# --------------------------------------------------------------------- recorder
class Recorder:
    """Cache of frozen switch state, keyed by (module, call index in this pass).

    Keying per module rather than by a global sequence means the recording and
    replay passes only have to agree on the calls they share, not on the whole
    program -- which matters because the recording pass may run modules (the BEV
    encoder) that the replay pass skips.
    """

    def __init__(self):
        self.store = {}
        self.mode = 'record'
        self._n = {}

    def new_pass(self):
        self._n = {}
        return self

    def freeze(self):
        self.mode = 'replay'
        return self.new_pass()

    def take(self, owner, fn):
        k = id(owner)
        i = self._n.get(k, 0)
        self._n[k] = i + 1
        key = (k, i)
        if self.mode == 'replay':
            if key not in self.store:
                raise KeyError(
                    f'replay asked for {type(owner).__name__} call #{i}, which the '
                    f'recording pass never made -- the two passes ran different code')
            return self.store[key]
        v = fn()
        v = tuple(t.detach() for t in v) if isinstance(v, tuple) else v.detach()
        self.store[key] = v
        return v


# --------------------------------------------------------------------- wrappers
class _Frozen(nn.Module):
    def __init__(self, inner, rec):
        super().__init__()
        self.inner = inner
        self._rec = rec        # leading underscore: not a submodule


class FrozenLayerNorm(_Frozen):
    def forward(self, x):
        ln = self.inner
        mu, rstd = self._rec.take(self, lambda: (
            x.mean(-1, keepdim=True),
            1.0 / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)))
        y = (x - mu) * rstd
        if ln.elementwise_affine:
            y = y * ln.weight + ln.bias
        return y


class FrozenGroupNorm(_Frozen):
    def forward(self, x):
        gn = self.inner
        assert gn.num_groups == 1, 'only GroupNorm(1, C) appears on this path'
        dims = tuple(range(1, x.dim()))
        mu, rstd = self._rec.take(self, lambda: (
            x.mean(dim=dims, keepdim=True),
            1.0 / torch.sqrt(x.var(dim=dims, unbiased=False, keepdim=True) + gn.eps)))
        shape = [1, -1] + [1] * (x.dim() - 2)
        return (x - mu) * rstd * gn.weight.view(shape) + gn.bias.view(shape)


class FrozenGELU(_Frozen):
    def forward(self, x):
        # exact GELU IS x * Phi(x), so this is not an approximation
        import math
        return x * self._rec.take(
            self, lambda: 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))))


class FrozenReLU(_Frozen):
    def forward(self, x):
        return x * self._rec.take(self, lambda: (x > 0).to(x.dtype))


class FrozenMHA(_Frozen):
    """nn.MultiheadAttention with the attention map frozen (batch_first=False)."""

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                need_weights=True, attn_mask=None, **kw):
        import math
        a = self.inner
        assert attn_mask is None and key_padding_mask is None, \
            'masked attention is not on the planner path; add support before using it'
        L, N, E = query.shape
        S = key.shape[0]
        H, hd = a.num_heads, E // a.num_heads
        W, b = a.in_proj_weight, a.in_proj_bias
        q = F.linear(query, W[:E], b[:E]).reshape(L, N * H, hd).transpose(0, 1) / math.sqrt(hd)
        k = F.linear(key, W[E:2 * E], b[E:2 * E]).reshape(S, N * H, hd).transpose(0, 1)
        v = F.linear(value, W[2 * E:], b[2 * E:]).reshape(S, N * H, hd).transpose(0, 1)
        A = self._rec.take(self, lambda: torch.bmm(q, k.transpose(1, 2)).softmax(-1))
        o = torch.bmm(A, v).transpose(0, 1).reshape(L, N, E)
        return F.linear(o, a.out_proj.weight, a.out_proj.bias), None


_SWAP = {nn.LayerNorm: FrozenLayerNorm, nn.GroupNorm: FrozenGroupNorm,
         nn.GELU: FrozenGELU, nn.ReLU: FrozenReLU,
         nn.MultiheadAttention: FrozenMHA}


# ------------------------------------------------------- TokenLearner's softmax
def _frozen_tokenlearner_forward(self, inputs, deterministic=True):
    """tokenlearner.py:34-58 verbatim, with the softmax frozen.

    The `.view(B, num_tokens, -1)` is a raw reshape of a [B, HW, n] tensor, not a
    transpose. That is what the trained model does, so it is what is replicated.
    """
    rec = self._lmd_rec
    if inputs.dim() == 4:
        n, c, h, w = inputs.shape
        inputs = inputs.view(n, c, h * w).permute(0, 2, 1)
    selected = self.mlp(self.layer_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1))
    selected = selected.view(inputs.shape[0], self.num_tokens, -1)
    selected = rec.take(self, lambda: selected.softmax(dim=-1))
    feat = inputs.view(inputs.shape[0], -1, inputs.shape[-1])
    return torch.einsum('bsi,bic->bsc', selected, feat), selected


# ------------------------------------------------------------ deformable attn
class _FrozenDeformFn:
    """Freezes `sampling_locations` and `attention_weights`, keeps `value` live.

    Both are predicted from the query. Once they are constants, the bilinear
    gather plus weighted sum is linear in `value` -- which is the BEV feature for
    the auxiliary heads and the image feature for the encoder. Intercepting here
    covers CustomMSDeformableAttention, MSDeformableAttention3D and
    TemporalSelfAttention at once, since all three funnel through these two
    symbols.
    """

    def __init__(self, rec, tag):
        self._rec, self._tag = rec, tag

    def _state(self, loc, w):
        # one counter per interception point, not per module: the calls happen
        # in a fixed order inside a head's forward, and record/replay run the
        # same forward
        return self._rec.take(self, lambda: (loc, w))

    def apply(self, value, spatial_shapes, level_start_index,
              sampling_locations, attention_weights, im2col_step):
        from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
        loc, w = self._state(sampling_locations, attention_weights)
        return multi_scale_deformable_attn_pytorch(value, spatial_shapes, loc, w)

    def __call__(self, value, spatial_shapes, sampling_locations, attention_weights):
        from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
        loc, w = self._state(sampling_locations, attention_weights)
        return multi_scale_deformable_attn_pytorch(value, spatial_shapes, loc, w)


_DEFORM_HOSTS = ('projects.mmdet3d_plugin.SSR.modules.decoder',
                 'projects.mmdet3d_plugin.SSR.modules.spatial_cross_attention',
                 'projects.mmdet3d_plugin.SSR.modules.temporal_self_attention')
_DEFORM_NAMES = ('MultiScaleDeformableAttnFunction_fp32',
                 'MultiScaleDeformableAttnFunction_fp16',
                 'multi_scale_deformable_attn_pytorch')


# ------------------------------------------------------------------ entry point
@contextlib.contextmanager
def linearise(root, rec, deformable=True):
    """Swap every nonlinearity under `root` for its frozen-switch version."""
    import importlib
    undo = []

    for parent in root.modules():
        for name, child in list(parent.named_children()):
            for cls, wrap in _SWAP.items():
                if type(child) is cls:
                    setattr(parent, name, wrap(child, rec))
                    undo.append((parent, name, child))
                    break

    tl_orig = TokenLearnerV11.forward
    TokenLearnerV11.forward = _frozen_tokenlearner_forward
    for m in root.modules():
        if isinstance(m, TokenLearnerV11):
            m._lmd_rec = rec

    deform_undo = []
    if deformable:
        for host in _DEFORM_HOSTS:
            try:
                mod = importlib.import_module(host)
            except ImportError:
                continue
            for nm in _DEFORM_NAMES:
                if hasattr(mod, nm):
                    deform_undo.append((mod, nm, getattr(mod, nm)))
                    setattr(mod, nm, _FrozenDeformFn(rec, f'{host}.{nm}'))
    try:
        yield rec.new_pass()
    finally:
        for parent, name, child in reversed(undo):
            setattr(parent, name, child)
        TokenLearnerV11.forward = tl_orig
        for mod, nm, orig in deform_undo:
            setattr(mod, nm, orig)


@contextlib.contextmanager
def bev_override(transformer, bev):
    """Make `get_bev_features` hand back `bev`, so the head becomes bev -> traj.

    Uses the head's own forward rather than a copy of it, which is the only way
    to be sure the analysed function is the one the checkpoint was trained as.
    """
    orig = transformer.get_bev_features
    transformer.get_bev_features = lambda *a, **kw: bev
    try:
        yield
    finally:
        transformer.get_bev_features = orig
