"""refiner-v3 forecast generators (docs/refiner_v3/SPEC.md sections 3.6, 4.3).  torch.

Inputs (causal only; pack layout section 3.4, rows section 3.2, metric and unnormalised)
  obj_in      [T, 3, K, 17]  frames f = 0 (t0), 1 (t-0.5 s), 2 (t-1.0 s); K <= K_OBJ = 64 slots per frame
  obj_mask    [T, 3, K] bool valid slots
  frame_valid [T, 3] bool    (a row counts only if obj_mask & frame_valid)
Every generator returns a Forecast of the t0 slots (frame 0) at the FC_T knots 0.5 .. 5.0 s in the t0 N frame.

Forecast fields (per token T, trimmed slot K, mode M)
  mu [T,K,M,10,2] positions, b [T,K,M,10,2] Laplace scales per axis, pi [T,K,M] (sums to 1 over M),
  present [T,K,M,10] (1 = present; 0 only at oracle ABSENT steps), c0 / v0 [T,K,2], h0 / L / W [T,K], group [T,K]
  long (0 vehicle, 1 VRU, 2 static), valid [T,K] bool, log_b / logit_pi (learned only, for the loss), aux (diagnostics).
  Statics (group 2) and padding: mu = c0 at every knot, b = B0, pi = one-hot on mode 0 (so the raster counts them
  once whatever M is; the gap scalars hold statics at c0 in every mode anyway).

Generators
  ca_forecast       torch port of the a3 CA rule (a3_conflict_features.Worker.history + world('CA')) with statics
                    held (section 3.6); M = 1, pi = 1, b = ca_b[group] (fitted on navtrain_train), no gradient.
                    dtype-generic (float64 inputs give the float64 reference used by the tests).
  oracle_forecast   PRIVILEGED (arm d): state FC_OBS or FC_SINGLE (official hold of a one-keyframe track) -> matched
                    GT (x, y), b = B_ORACLE; FC_ABSENT -> present 0;
                    FC_NONE / FC_BEYOND / unmatched / statics -> the CA forecast (x, y, b).  M = 1.
  LearnedForecaster arm a: object encoder (phi + frame embedding), 2 past set-attention layers (t0 query; keys = null,
                    self, every valid past box of frames 1-2; pairwise bias and pairwise value term), 1 social
                    attention layer over valid t0 boxes, M-mode decoder: residual over constant velocity
                    (mu_k = c0 + v0 t_k + d_k), Laplace log-scales clamped to [log B_MIN, log B_MAX], mode logits.
                    fp32 with autocast disabled.  No track ids are needed (set attention).

trim_objects: K' = 1 + max slot index with any True mask over the whole batch (>= 1); slices every frame to K'.
Causality: the generators read obj_in / obj_mask / frame_valid only (+ the privileged keys for the oracle);
TemporalBranch.scene additionally refuses rows whose age contradicts their frame (branch.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (B0, B_MAX, B_MIN, B_ORACLE, COL, D_OBJ, FC_ABSENT, FC_OBS, FC_SINGLE, FC_STEPS, FC_T,
                     FORECASTER_SEED_OFFSET,
                     GROUP_OF_CLASS, N_CLS, N_FRAMES, NORM)

Tensor = torch.Tensor
ACC_CLIP = 6.0            # a3: clip(a, -6, 6)
ACC_MIN_SPEED = 0.5       # a3: a = 0 unless |v0| > 0.5 m/s
ASSOC_GATE = 1.5          # a3: nearest same-group past box within 1.5 m
HIST_FRAMES = (2, 1)      # a3: prefer the 1.0-s window, frame 1 only for objects still without history
LOG_B_MIN, LOG_B_MAX = math.log(B_MIN), math.log(B_MAX)


# ----------------------------------------------------------------------------------------------- container
@dataclass
class Forecast:
    mu: Tensor
    b: Tensor
    pi: Tensor
    present: Tensor
    c0: Tensor
    v0: Tensor
    h0: Tensor
    L: Tensor
    W: Tensor
    group: Tensor
    valid: Tensor
    log_b: Optional[Tensor] = None
    logit_pi: Optional[Tensor] = None
    aux: Optional[Dict[str, Tensor]] = None

    @property
    def n_modes(self) -> int:
        return int(self.mu.shape[2])

    @property
    def agent(self) -> Tensor:
        return self.valid & (self.group < 2)

    def detach(self) -> "Forecast":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.detach()
            elif isinstance(v, dict):
                v = {k: (x.detach() if isinstance(x, torch.Tensor) else x) for k, x in v.items()}
            kw[f.name] = v
        return Forecast(**kw)

    def check(self) -> None:
        """Shape / finiteness assertions (tests and debugging)."""
        T, K, M = self.mu.shape[:3]
        assert self.mu.shape == (T, K, M, FC_STEPS, 2) and self.b.shape == self.mu.shape
        assert self.pi.shape == (T, K, M) and self.present.shape == (T, K, M, FC_STEPS)
        for n in ("c0", "v0"):
            assert getattr(self, n).shape == (T, K, 2)
        for n in ("h0", "L", "W", "group", "valid"):
            assert getattr(self, n).shape == (T, K)
        for n in ("mu", "b", "pi", "present"):
            assert torch.isfinite(getattr(self, n)).all(), n


# ----------------------------------------------------------------------------------------------- helpers
def trim_objects(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """K' = 1 + max slot index with any True mask over the batch (>= 1); slices [..., :K', :]."""
    if obj_in.dim() != 4 or obj_in.shape[1] != N_FRAMES or obj_in.shape[-1] != D_OBJ:
        raise ValueError(f"obj_in must be [T, {N_FRAMES}, K, {D_OBJ}], got {tuple(obj_in.shape)}")
    if obj_mask.shape != obj_in.shape[:3] or frame_valid.shape != obj_in.shape[:2]:
        raise ValueError(f"obj_mask {tuple(obj_mask.shape)} / frame_valid {tuple(frame_valid.shape)} vs obj_in "
                         f"{tuple(obj_in.shape)}")
    any_slot = obj_mask.reshape(-1, obj_mask.shape[-1]).any(0)
    k = int(torch.nonzero(any_slot).max()) + 1 if bool(any_slot.any()) else 1
    return obj_in[:, :, :k], obj_mask[:, :, :k], frame_valid


def effective_mask(obj_mask: Tensor, frame_valid: Tensor) -> Tensor:
    """obj_mask & frame_valid (a slot of an invalid frame never counts)."""
    return obj_mask.bool() & frame_valid.bool()[:, :, None]


def object_fields(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> Dict[str, Tensor]:
    """t0 slot attributes: c0, v0 [T,K,2], h0, L, W [T,K], group [T,K] long, valid [T,K] bool."""
    x0 = obj_in[:, 0]
    valid = effective_mask(obj_mask, frame_valid)[:, 0]
    cls = x0[..., COL["cls0"]:COL["cls0"] + N_CLS].argmax(-1)
    gmap = torch.as_tensor(GROUP_OF_CLASS, dtype=torch.long, device=obj_in.device)
    return dict(c0=x0[..., 0:2], v0=x0[..., COL["vx"]:COL["vy"] + 1],
                h0=torch.atan2(x0[..., COL["sin"]], x0[..., COL["cos"]]), L=x0[..., COL["L"]], W=x0[..., COL["W"]],
                group=gmap[cls], valid=valid)


def _static_pi(T: int, K: int, M: int, like: Tensor) -> Tensor:
    pi = torch.zeros(T, K, M, dtype=like.dtype, device=like.device)
    pi[..., 0] = 1.0
    return pi


def fc_times(like: Tensor) -> Tensor:
    return torch.as_tensor(FC_T, dtype=like.dtype, device=like.device)


# ----------------------------------------------------------------------------------------------- CA (section 3.6)
def ca_history(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> Dict[str, Tensor]:
    """a3 Worker.history on pack rows -> acc [T,K] (longitudinal, m/s^2), has [T,K] bool, n_frames [T] (past frames
    with >= 1 valid box).  Association: predicted past centre c0 - v0 dt_f, nearest past box of the same agent /
    static group (any class within the group) within 1.5 m, agents only, frame 2 first; dt_f from the rows' age."""
    m = effective_mask(obj_mask, frame_valid)
    of = object_fields(obj_in, obj_mask, frame_valid)
    c0, v0 = of["c0"], of["v0"]
    agent0 = of["group"] < 2
    cls_all = obj_in[..., COL["cls0"]:COL["cls0"] + N_CLS].argmax(-1)               # [T,3,K]
    agent_all = torch.as_tensor(GROUP_OF_CLASS, device=obj_in.device)[cls_all] < 2
    sp0 = torch.hypot(v0[..., 0], v0[..., 1])
    u0 = v0 / torch.clamp(sp0, min=1e-6)[..., None]
    acc = torch.zeros_like(sp0)
    has = torch.zeros_like(of["valid"])
    n_frames = torch.zeros(obj_in.shape[0], dtype=torch.long, device=obj_in.device)
    for f in HIST_FRAMES:
        mf = m[:, f]                                                                  # [T,K]
        xf = obj_in[:, f]
        age = torch.where(mf, xf[..., COL["age"]], torch.zeros_like(xf[..., COL["age"]]))
        dt = age.amax(-1)                                                             # [T] (one dt per frame)
        ok_f = mf.any(-1) & (dt > 0)
        n_frames = n_frames + frame_valid[:, f].long()
        dts = torch.where(ok_f, dt, torch.ones_like(dt))
        pred = c0 - v0 * dts[:, None, None]                                           # [T,K,2]
        dx = pred[:, :, None, 0] - xf[:, None, :, 0]
        dy = pred[:, :, None, 1] - xf[:, None, :, 1]
        dd = torch.hypot(dx, dy)                                                      # [T,K,Kp]
        same = agent0[:, :, None] == agent_all[:, f][:, None, :]
        dd = torch.where(same & mf[:, None, :], dd, torch.full_like(dd, float("inf")))
        jj = dd.argmin(-1)
        dmin = torch.gather(dd, -1, jj[..., None])[..., 0]
        ok = (dmin < ASSOC_GATE) & ~has & agent0 & of["valid"] & ok_f[:, None]
        vp = torch.gather(xf[..., COL["vx"]:COL["vy"] + 1], 1, jj[..., None].expand(*jj.shape, 2))
        a = ((v0 - vp) * u0).sum(-1) / dts[:, None]
        a = torch.where(sp0 > ACC_MIN_SPEED, torch.clamp(a, -ACC_CLIP, ACC_CLIP), torch.zeros_like(a))
        acc = torch.where(ok, a, acc)
        has = has | ok
    return {"acc": acc, "has": has, "n_frames": n_frames, "sp0": sp0, "u0": u0}


def ca_positions(c0: Tensor, sp0: Tensor, u0: Tensor, acc: Tensor, t: Tensor) -> Tensor:
    """a3 world('CA'): dist(t) = |v0| te + 0.5 a te^2, te = min(t, tstop), tstop = -|v0| / a if a < -1e-6 else inf;
    c(t) = c0 + u0 dist(t).  c0 [...,2], sp0 / acc [...], t [nt] -> [..., nt, 2]."""
    a = acc[..., None]
    s = sp0[..., None]
    neg = a < -1e-6
    tstop = torch.where(neg, -s / torch.where(neg, a, -torch.ones_like(a)), torch.full_like(a, float("inf")))
    te = torch.minimum(t.to(c0.dtype), tstop)                                         # [..., nt]
    dist = s * te + 0.5 * a * te ** 2
    return c0[..., None, :] + u0[..., None, :] * dist[..., None]


@torch.no_grad()
def ca_forecast(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor, ca_b: Tensor) -> Forecast:
    """Analytic constant-acceleration forecast (M = 1, no grad); statics and padding held at c0 with b = B0."""
    of = object_fields(obj_in, obj_mask, frame_valid)
    h = ca_history(obj_in, obj_mask, frame_valid)
    T, K = of["valid"].shape
    t = fc_times(obj_in)
    pos = ca_positions(of["c0"], h["sp0"], h["u0"], h["acc"], t)                     # [T,K,10,2]
    agent = (of["group"] < 2) & of["valid"]
    held = of["c0"][:, :, None, :].expand_as(pos)
    mu = torch.where(agent[..., None, None], pos, held)
    ca_b = ca_b.to(device=obj_in.device, dtype=obj_in.dtype)
    if ca_b.shape != (2, FC_STEPS):
        raise ValueError(f"ca_b must be [2, {FC_STEPS}], got {tuple(ca_b.shape)}")
    bg = ca_b[of["group"].clamp(max=1)]                                               # [T,K,10]
    bg = torch.where(agent[..., None], bg, torch.full_like(bg, B0))
    return Forecast(mu=mu[:, :, None], b=bg[:, :, None, :, None].expand(T, K, 1, FC_STEPS, 2).contiguous(),
                    pi=torch.ones(T, K, 1, dtype=obj_in.dtype, device=obj_in.device),
                    present=torch.ones(T, K, 1, FC_STEPS, dtype=obj_in.dtype, device=obj_in.device),
                    c0=of["c0"], v0=of["v0"], h0=of["h0"], L=of["L"], W=of["W"], group=of["group"],
                    valid=of["valid"], aux={"acc": h["acc"], "has": h["has"], "n_frames": h["n_frames"]})


@torch.no_grad()
def oracle_forecast(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor, priv_fc_xyh: Tensor, priv_fc_state: Tensor,
                    ca_b: Tensor) -> Forecast:
    """PRIVILEGED (arm d, GTt world): matched GT futures of the t0 boxes, CA elsewhere (M = 1, no grad)."""
    ca = ca_forecast(obj_in, obj_mask, frame_valid, ca_b)
    K = ca.mu.shape[1]
    if priv_fc_xyh.shape[1] < K or priv_fc_state.shape[1] < K:
        raise ValueError(f"privileged targets {tuple(priv_fc_xyh.shape)} shorter than K' = {K}")
    xy = priv_fc_xyh[:, :K, :, :2].to(ca.mu.dtype)
    st = priv_fc_state[:, :K].long()
    agent = ca.agent[..., None]
    obs = ((st == FC_OBS) | (st == FC_SINGLE)) & agent                                # [T,K,10] official rule
    absent = (st == FC_ABSENT) & agent
    mu = torch.where(obs[:, :, None, :, None], xy[:, :, None], ca.mu)
    b = torch.where(obs[:, :, None, :, None], torch.full_like(ca.b, B_ORACLE), ca.b)
    present = torch.where(absent[:, :, None], torch.zeros_like(ca.present), ca.present)
    aux = dict(ca.aux)
    aux.update(n_obs=obs.sum((1, 2)), n_absent=absent.sum((1, 2)))
    return replace(ca, mu=mu, b=b, present=present, aux=aux)


# ----------------------------------------------------------------------------------------------- learned forecaster
def normalise_rows(x: Tensor) -> Tensor:
    """Section 3.2 fixed normalisation: x, y / 32; L / 10; W / 5; vx, vy / 15; cos, sin, score, one-hot, age as is."""
    s = torch.ones(D_OBJ, dtype=x.dtype, device=x.device)
    s[COL["x"]] = s[COL["y"]] = 1.0 / NORM["xy"]
    s[COL["L"]] = 1.0 / NORM["L"]
    s[COL["W"]] = 1.0 / NORM["W"]
    s[COL["vx"]] = s[COL["vy"]] = 1.0 / NORM["v"]
    return x * s


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class _PastLayer(nn.Module):
    """Pre-LN attention of each t0 token i over [null | self i | valid past tokens j], pairwise bias and values."""

    def __init__(self, d: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.bias_mlp = _mlp(6, 32, n_heads)
        self.val_mlp = _mlp(6, 32, d)
        self.null_k = nn.Parameter(torch.randn(d) * 0.02)
        self.null_v = nn.Parameter(torch.randn(d) * 0.02)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = _mlp(d, d_ffn, d)

    def forward(self, e0: Tensor, ep: Tensor, r: Tensor, kmask: Tensor) -> Tensor:
        """e0 [T,K,d] t0 tokens, ep [T,P,d] past tokens, r [T,K,P,6] pair features, kmask [T,P] bool."""
        T, K, d = e0.shape
        P = ep.shape[1]
        H, dh = self.h, self.dh
        sc = 1.0 / math.sqrt(dh)
        q = self.q(self.ln_q(e0)).view(T, K, H, dh)
        hs = self.ln_kv(e0)
        hp = self.ln_kv(ep)
        r_self = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=e0.dtype, device=e0.device)
        k_self = self.k(hs).view(T, K, H, dh)
        v_self = (self.v(hs) + self.val_mlp(r_self)).view(T, K, H, dh)
        k_past = self.k(hp).view(T, P, H, dh)
        v_past = self.v(hp).view(T, P, H, dh)
        s_null = (q * self.null_k.view(H, dh)).sum(-1) * sc                          # [T,K,H]
        s_self = (q * k_self).sum(-1) * sc + self.bias_mlp(r_self)                    # [T,K,H]
        s_past = torch.einsum("tkhd,tphd->tkhp", q, k_past) * sc + self.bias_mlp(r).permute(0, 1, 3, 2)
        s_past = s_past.masked_fill(~kmask[:, None, None, :], float("-inf"))
        a = torch.softmax(torch.cat([s_null[..., None], s_self[..., None], s_past], -1), -1)   # [T,K,H,2+P]
        ap = a[..., 2:]
        v_pair = self.val_mlp(r).view(T, K, P, H, dh)
        out = (a[..., 0, None] * self.null_v.view(H, dh) + a[..., 1, None] * v_self
               + torch.einsum("tkhp,tphd->tkhd", ap, v_past) + torch.einsum("tkhp,tkphd->tkhd", ap, v_pair))
        e0 = e0 + self.o(out.reshape(T, K, d))
        return e0 + self.ff(self.ln_ff(e0))


class _SocialLayer(nn.Module):
    """Pre-LN self-attention over valid t0 tokens (+ self), per-head bias MLP of the relative position / 32."""

    def __init__(self, d: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.bias_mlp = _mlp(2, 32, n_heads)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = _mlp(d, d_ffn, d)

    def forward(self, e: Tensor, xy: Tensor, valid: Tensor) -> Tensor:
        T, K, d = e.shape
        H, dh = self.h, self.dh
        x = self.ln(e)
        q = self.q(x).view(T, K, H, dh)
        k = self.k(x).view(T, K, H, dh)
        v = self.v(x).view(T, K, H, dh)
        rel = (xy[:, None, :, :] - xy[:, :, None, :]) / NORM["xy"]                    # [T,i,j,2] = x_j - x_i
        s = torch.einsum("tihd,tjhd->tihj", q, k) / math.sqrt(dh) + self.bias_mlp(rel).permute(0, 1, 3, 2)
        eye = torch.eye(K, dtype=torch.bool, device=e.device)
        keym = valid[:, None, :] | eye[None]                                          # [T,i,j]
        s = s.masked_fill(~keym[:, :, None, :], float("-inf"))
        a = torch.softmax(s, -1)
        out = torch.einsum("tihj,tjhd->tihd", a, v).reshape(T, K, d)
        e = e + self.o(out)
        return e + self.ff(self.ln_ff(e))


class LearnedForecaster(nn.Module):
    """Arm a forecaster (SPEC section 4.3, pinned).  forward(obj_in, obj_mask, frame_valid) -> Forecast (fp32)."""

    def __init__(self, n_modes: int = 6, d: int = 128, n_heads: int = 4, d_ffn: int = 256,
                 n_past_layers: int = 2, n_social_layers: int = 1, seed: Optional[int] = None):
        super().__init__()
        self.n_modes, self.d = int(n_modes), int(d)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed) + FORECASTER_SEED_OFFSET)
            self.embed = _mlp(D_OBJ, d, d)
            self.frame_emb = nn.Embedding(N_FRAMES, d)
            self.past = nn.ModuleList([_PastLayer(d, n_heads, d_ffn) for _ in range(n_past_layers)])
            self.social = nn.ModuleList([_SocialLayer(d, n_heads, d_ffn) for _ in range(n_social_layers)])
            self.ln_dec = nn.LayerNorm(d)
            self.mode_emb = nn.Parameter(torch.randn(self.n_modes, d) * 0.02)
            self.dec1 = nn.Linear(d, 256)
            self.dec_out = nn.Linear(256, FC_STEPS * 4)
            self.dec_pi = nn.Linear(256, 1)

    @staticmethod
    def pair_features(x0: Tensor, xp: Tensor) -> Tensor:
        """r_ij [T,K,P,6] from metric rows x0 [T,K,17] (t0) and xp [T,P,17] (past)."""
        aj = xp[..., COL["age"]][:, None, :]
        dx = xp[..., 0][:, None, :] - x0[..., 0][:, :, None] + x0[..., COL["vx"]][:, :, None] * aj
        dy = xp[..., 1][:, None, :] - x0[..., 1][:, :, None] + x0[..., COL["vy"]][:, :, None] * aj
        dvx = xp[..., COL["vx"]][:, None, :] - x0[..., COL["vx"]][:, :, None]
        dvy = xp[..., COL["vy"]][:, None, :] - x0[..., COL["vy"]][:, :, None]
        gmap = torch.as_tensor(GROUP_OF_CLASS, device=x0.device)
        ag0 = gmap[x0[..., COL["cls0"]:COL["cls0"] + N_CLS].argmax(-1)] < 2
        agp = gmap[xp[..., COL["cls0"]:COL["cls0"] + N_CLS].argmax(-1)] < 2
        same = (ag0[:, :, None] == agp[:, None, :]).to(x0.dtype)
        return torch.stack([dx / 4.0, dy / 4.0, dvx / 5.0, dvy / 5.0, aj.expand_as(dx), same], -1)

    def forward(self, obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> Forecast:
        with torch.autocast(device_type=obj_in.device.type, enabled=False):
            return self._forward(obj_in.float(), obj_mask.bool(), frame_valid.bool())

    def _forward(self, obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> Forecast:
        T, NF, K, _ = obj_in.shape
        M = self.n_modes
        m = effective_mask(obj_mask, frame_valid)
        x = torch.where(m[..., None], obj_in, torch.zeros_like(obj_in))               # padding rows -> 0
        of = object_fields(x, m, frame_valid)
        e = self.embed(normalise_rows(x)) + self.frame_emb.weight[None, :, None, :]   # [T,3,K,d]
        e0 = e[:, 0]
        ep = e[:, 1:].reshape(T, (NF - 1) * K, self.d)
        xp = x[:, 1:].reshape(T, (NF - 1) * K, D_OBJ)
        kmask = m[:, 1:].reshape(T, (NF - 1) * K)
        r = self.pair_features(x[:, 0], xp)
        for layer in self.past:
            e0 = layer(e0, ep, r, kmask)
        for layer in self.social:
            e0 = layer(e0, x[:, 0, :, 0:2], of["valid"])
        z = self.ln_dec(e0)[:, :, None, :] + self.mode_emb                             # [T,K,M,d]
        h = F.gelu(self.dec1(z))
        out = self.dec_out(h).view(T, K, M, FC_STEPS, 4)
        logit_pi = self.dec_pi(h)[..., 0]                                              # [T,K,M]
        t = fc_times(x)
        cv = of["c0"][:, :, None, None, :] + of["v0"][:, :, None, None, :] * t[:, None]   # [T,K,1,10,2]
        mu_l = cv + out[..., :2]
        log_b_l = torch.clamp(out[..., 2:], LOG_B_MIN, LOG_B_MAX)
        agent = (of["group"] < 2) & of["valid"]
        a5 = agent[:, :, None, None, None]
        held = of["c0"][:, :, None, None, :].expand_as(mu_l)
        mu = torch.where(a5, mu_l, held)
        log_b = torch.where(a5, log_b_l, torch.full_like(log_b_l, math.log(B0)))
        pi = torch.where(agent[..., None], torch.softmax(logit_pi, -1), _static_pi(T, K, M, logit_pi))
        b = torch.where(a5, torch.exp(log_b_l), torch.full_like(log_b_l, B0))
        return Forecast(mu=mu, b=b, pi=pi,
                        present=torch.ones(T, K, M, FC_STEPS, dtype=x.dtype, device=x.device),
                        c0=of["c0"], v0=of["v0"], h0=of["h0"], L=of["L"], W=of["W"], group=of["group"],
                        valid=of["valid"], log_b=log_b, logit_pi=logit_pi)
