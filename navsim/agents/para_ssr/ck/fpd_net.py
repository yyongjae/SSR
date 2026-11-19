# SPDX-License-Identifier: Apache-2.0
# The DiT blocks / DDIM sampler (ResidualDiT) are adapted from FPD-Drive's diffsynth/planning/fpd_planner/refiner.py
# (https://github.com/CountNemoChan/FPD-Drive, Apache-2.0; itself adapted from ChainFlow-VLA, Apache-2.0) and the scorer
# (FPDScorer) from FPD-Drive's transformer_decoder.py / score_module/scorer.py / fpd_drive/model.py (adapted from DrivoR,
# Apache-2.0).  See NOTICE.FPD-Drive next to this file for the upstream attribution.
"""FPD-Drive refiner structure (residual DiT + DrivoR-style candidate scorer) for the CK2 refiner slot (REFINER_ARCH=fpd).

What is FPD-Drive's and what is ours
  ResidualDiT   the 12-block adaLN-Zero DiT of FPD-Drive: 8 pose tokens of ONE candidate (RoPE + q/k RMSNorm
                self-attention), cross-attention to the scene tokens, condition = timestep + ego MLP; epsilon
                prediction; DDIM (T 1000, scaled-linear betas 1e-4..0.02, 'leading' spacing, eta 0).  It never sees the
                proposal it refines: a sampled residual is ADDED to the proposal (raw m / m / rad, no normalisation).
                Verbatim apart from the class names, a batched-timestep velocity() and train_terms().
  FPDScorer     the released student scorer: MLP(24 -> 1024 -> 256) embedding of each (detached) candidate, 4 pre-LN
                blocks (self-attention among the candidates, cross-attention to the scene tokens, MLP x4), the ego
                token added after the blocks, one MLP(256 -> 1024 -> 1) head per score.  Ours: 5 heads in CK_KEYS order
                (no DDC head), dropout 0 (the release uses proj_drop 0.1; 0 keeps the CUDA RNG of v2's own dropout
                untouched), ego token = Linear(8) of SpecDrive's status_feature (FPD: Linear(11) of [pose 0, vel, acc,
                cmd]; the current pose is 0 in the ego frame).
  SceneTokens   ours: FPD-Drive's scene tokens are 64 camera register tokens of a DINOv2 encoder; SpecDrive's scene
                representation is the BEV, so the 64 tokens are the BEV grid adaptive-avg-pooled to 8 x 8, LayerNorm,
                Linear, + a learned position embedding.  The gradient into the BEV is scaled by bev_grad_scale (the CK2
                AdapterSGrid rule).
  FPDRefinerNet the student / teacher module = SceneTokens + ResidualDiT + FPDScorer (shared scene tokens, as FPD-Drive
                shares h between its DiT and scorer).  The same class is the frozen privileged teacher (on a teacher BEV).

Helpers: pool assembly (c = 6 k + v; v 0 = the parent, v 1..5 = DiT draws), token-keyed deterministic noise,
Hungarian target matching (FPD Eq. 5, cost = mean xy distance), the DiT losses (FPD Eq. 6: epsilon MSE + one-step
reconstruction L1) and FPD's multi-target selection (safety pass, best seed, greedy max-min ADE >= 1 m, <= K targets).
No global RNG is consumed anywhere (construction forks the RNG; all noise comes from token-keyed numpy generators).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CK_KEYS

# ------------------------------------------------------------------------------------------------- constants
FPD_DIM = 256
FPD_POSES = 8
FPD_STATE = 3
DIT_LAYERS = 12
DIT_HEADS = 8
DIT_HIDDEN = 1024
DDIM_T = 1000
DDIM_BETA = (1e-4, 0.02)
DDIM_STEPS = 8                     # FPD-Drive paper (App. A / Table 6); the released configs use 12 (Table 12: same)
SCORER_LAYERS = 4
SCORER_HEADS = 1                   # as released (refiner_num_heads 1)
SCORER_FFN = 1024
SCENE_GRID = 8                     # 8 x 8 = 64 scene tokens (FPD-Drive: 64 camera register tokens)
N_DRAWS = 5                        # DiT draws per parent: pool = 16 x (1 parent + 5 draws) = 96 (the base layout)
EGO_DIM = 8
SEED_OFFSET = 52021                # construction RNG offset (CKNet uses 31337 / 7919)
NOISE_STREAM = "fpd.noise"
TRAIN_STREAM = "fpd.train"


# ------------------------------------------------------------------------------------------------- DiT (FPD-Drive)
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)
        self.max_positions = int(max_positions)

    def forward(self, x: torch.Tensor, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(0, length, device=x.device, dtype=self.inv_freq.dtype)
        phase = torch.einsum("i,j->ij", pos, self.inv_freq.to(device=x.device))
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos()[None, None].to(dtype=x.dtype), phase.sin()[None, None].to(dtype=x.dtype)


class _DiTAttention(nn.Module):
    """RMS-normalised q/k; RoPE on self attention."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("DiT dim must be divisible by heads.")
        self.heads, self.head_dim = int(heads), dim // heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.q_norm, self.k_norm = _RMSNorm(self.head_dim), _RMSNorm(self.head_dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                rotary: Optional[_RotaryEmbedding] = None) -> torch.Tensor:
        context = x if context is None else context
        b, nq, _ = x.shape
        nk = context.shape[1]
        q = self.to_q(x).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rotary is not None:
            cos, sin = rotary(x, nq)
            q = q * cos + _rotate_half(q) * sin
            if context is x:
                k = k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        return self.out(y.transpose(1, 2).reshape(b, nq, -1))


class _DiTBlock(nn.Module):
    """adaLN-Zero self-attention, scene cross-attention, FFN."""

    def __init__(self, dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _DiTAttention(dim, heads, dropout)
        self.cross_attn = _DiTAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor,
                rotary: _RotaryEmbedding) -> torch.Tensor:
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(cond).chunk(9, dim=-1)
        x = x + g1[:, None] * self.self_attn(self._mod(self.norm1(x), s1, c1), rotary=rotary)
        h = self._mod(self.norm2(x), s2, c2)
        if context.shape[0] == x.shape[0]:
            ca = self.cross_attn(h, context=context)
        else:
            # one scene context shared by its modes: fold the modes into the query length (cross-attention queries
            # do not interact), so the context K/V is projected once per scene instead of once per mode (exact)
            bm, p, d = h.shape
            b = context.shape[0]
            ca = self.cross_attn(h.reshape(b, (bm // b) * p, d), context=context).reshape(bm, p, d)
        x = x + g2[:, None] * ca
        x = x + g3[:, None] * self.ffn(self._mod(self.norm3(x), s3, c3))
        return x


class ResidualDiT(nn.Module):
    """FPD-Drive's Refiner: residual DDIM over one candidate's poses, conditioned on scene tokens + ego status."""

    def __init__(self, dim: int = FPD_DIM, num_poses: int = FPD_POSES, state_size: int = FPD_STATE,
                 num_layers: int = DIT_LAYERS, num_heads: int = DIT_HEADS, hidden_dim: int = DIT_HIDDEN,
                 dropout: float = 0.0, num_inference_steps: int = DDIM_STEPS, ego_status_dim: int = EGO_DIM,
                 num_train_timesteps: int = DDIM_T, beta_start: float = DDIM_BETA[0],
                 beta_end: float = DDIM_BETA[1]) -> None:
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 2:
            raise ValueError("ResidualDiT requires an even per-head dimension.")
        self.num_poses, self.state_size = int(num_poses), int(state_size)
        self.num_inference_steps = int(num_inference_steps)
        self.num_train_timesteps = int(num_train_timesteps)
        betas = torch.linspace(math.sqrt(float(beta_start)), math.sqrt(float(beta_end)), self.num_train_timesteps,
                               dtype=torch.float32).square()          # 'scaled_linear'
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0), persistent=False)
        self.action_proj = nn.Linear(state_size, dim)
        self.action_time = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_embed = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ego_encoder = nn.Sequential(nn.Linear(ego_status_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, dim))
        self.position = nn.Embedding(self.num_poses, dim)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        self.rotary = _RotaryEmbedding(dim // num_heads, self.num_poses)
        self.blocks = nn.ModuleList([_DiTBlock(dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.velocity_head = nn.Linear(dim, state_size)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
                          / max(half, 1))
        emb = torch.cat(((t.float()[:, None] * freqs[None]).sin(), (t.float()[:, None] * freqs[None]).cos()), dim=-1)
        return F.pad(emb, (0, dim - emb.shape[-1])) if emb.shape[-1] < dim else emb

    def velocity(self, noisy: torch.Tensor, scene_tokens: torch.Tensor, ego_status: torch.Tensor,
                 timesteps: torch.Tensor) -> torch.Tensor:
        """noisy [B, M, P, 3], scene_tokens [B, N, D], ego_status [B, E], timesteps [B] (one per scene, shared by its
        M modes) -> predicted epsilon [B, M, P, 3].  Modes are flattened into the batch: no cross-candidate attention."""
        b, modes, poses, _ = noisy.shape
        if poses != self.num_poses or scene_tokens.ndim != 3:
            raise ValueError("Unexpected trajectory or scene-token shape.")
        dim = scene_tokens.shape[-1]
        cond = self.time_embed(self._sinusoidal(timesteps, dim).to(dtype=scene_tokens.dtype))
        cond = cond + self.ego_encoder(ego_status.to(device=cond.device, dtype=cond.dtype))
        cond = cond[:, None].expand(-1, modes, -1).reshape(b * modes, -1)
        context = scene_tokens            # [B, N, D], shared by the modes (the blocks fold the modes into the queries)
        tflat = timesteps[:, None].expand(-1, modes).reshape(-1)
        x = noisy.reshape(b * modes, poses, self.state_size)
        action = self.action_proj(x.to(dtype=self.action_proj.weight.dtype))
        time = self._sinusoidal(tflat, dim).to(dtype=action.dtype)[:, None].expand(-1, poses, -1)
        x = self.action_time(torch.cat((action, time), dim=-1)) + self.position.weight[:poses].to(action.dtype)[None]
        for block in self.blocks:
            x = block(x, context.to(dtype=x.dtype), cond.to(dtype=x.dtype), self.rotary)
        shift, scale = self.final_ada(cond.to(dtype=x.dtype)).chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + scale[:, None]) + shift[:, None]
        return self.velocity_head(x).reshape(b, modes, poses, self.state_size)

    def ddim_timesteps(self, steps: int, device) -> torch.Tensor:
        if not 0 < steps <= self.num_train_timesteps:
            raise ValueError(f"DDIM steps must be in [1, {self.num_train_timesteps}], got {steps}")
        ratio = self.num_train_timesteps // steps
        return torch.arange(0, steps, device=device, dtype=torch.long).mul(ratio).flip(0)

    @torch.no_grad()
    def sample(self, base: torch.Tensor, scene_tokens: torch.Tensor, ego_status: torch.Tensor,
               noise: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Deterministic DDIM from noise [B, M, P, 3] (the initial residual) -> base + residual [B, M, P, 3]."""
        if noise.shape != base.shape:
            raise ValueError(f"noise {tuple(noise.shape)} != base {tuple(base.shape)}")
        steps = self.num_inference_steps if steps is None else int(steps)
        ratio = self.num_train_timesteps // steps
        residual = noise.to(device=base.device, dtype=base.dtype)
        for timestep in self.ddim_timesteps(steps, residual.device):
            tid = int(timestep.item())
            ts = torch.full((base.shape[0],), tid, device=base.device, dtype=torch.long)
            eps = self.velocity(residual, scene_tokens, ego_status, ts).to(residual.dtype)
            a_t = self.alphas_cumprod[tid].to(device=residual.device, dtype=residual.dtype)
            prev = tid - ratio
            a_prev = (self.alphas_cumprod[prev].to(device=residual.device, dtype=residual.dtype) if prev >= 0
                      else residual.new_tensor(1.0))
            x0 = (residual - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-12)
            residual = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
        return base + residual

    def train_terms(self, residual: torch.Tensor, scene_tokens: torch.Tensor, ego_status: torch.Tensor,
                    t: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FPD-Drive Eq. 2-3: z = sqrt(abar_t) r + sqrt(1 - abar_t) eps -> (eps_hat, r0_hat), both [B, J, P, 3];
        r0_hat = (z - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t) (the one-step reconstruction of the residual)."""
        ab = self.alphas_cumprod.to(residual.device)[t].to(residual.dtype)[:, None, None, None]
        z = ab.sqrt() * residual + (1.0 - ab).sqrt() * eps
        eps_hat = self.velocity(z, scene_tokens, ego_status, t)
        r0 = (z - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt().clamp_min(1e-12)
        return eps_hat, r0


# ------------------------------------------------------------------------------------------------- scorer (FPD-Drive)
class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _ScorerBlock(nn.Module):
    """DrivoR / FPD-Drive Block: pre-LN self-attention among candidates, cross-attention to the scene, MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.self_attn_norm(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.cross_attn_norm_kv(ctx)
        x = x + self.cross_attn(self.cross_attn_norm_q(x), kv, kv, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class FPDScorer(nn.Module):
    """candidates [B, K, 8, 3] (detached), scene tokens [B, N, D], status [B, 8] -> score logits [B, K, 5] (CK_KEYS)."""

    def __init__(self, dim: int = FPD_DIM, ffn: int = SCORER_FFN, layers: int = SCORER_LAYERS,
                 heads: int = SCORER_HEADS, poses: int = FPD_POSES, ego_dim: int = EGO_DIM) -> None:
        super().__init__()
        self.pos_embed = nn.Sequential(nn.Linear(poses * FPD_STATE, ffn), nn.ReLU(), nn.Linear(ffn, dim))
        self.blocks = nn.ModuleList([_ScorerBlock(dim, heads) for _ in range(layers)])
        self.ego = nn.Linear(ego_dim, dim)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(), nn.Linear(ffn, 1))
                                    for _ in CK_KEYS])

    def set_prior(self, prior: Sequence[float], eps: float = 1e-4) -> None:
        """head bias = logit(prior) (CK2's score prior: the label mean of the warm-up mix)."""
        p = np.clip(np.asarray(prior, np.float64), eps, 1.0 - eps)
        with torch.no_grad():
            for h, v in zip(self.heads, np.log(p / (1.0 - p))):
                h[-1].bias.fill_(float(v))

    def forward(self, cand: torch.Tensor, ctx: torch.Tensor, status: torch.Tensor) -> torch.Tensor:
        B, K = cand.shape[:2]
        x = self.pos_embed(cand.detach().float().reshape(B, K, -1))
        for blk in self.blocks:
            x = blk(x, ctx)
        x = x + self.ego(status.float())[:, None]
        return torch.cat([h(x) for h in self.heads], -1).float()


# ------------------------------------------------------------------------------------------------- scene tokens
class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = float(s)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


class SceneTokens(nn.Module):
    """BEV grid [B, C, H, W] -> [B, G*G, D]: adaptive avg pool G x G, LayerNorm, Linear, + learned position."""

    def __init__(self, c_in: int = 256, dim: int = FPD_DIM, grid: int = SCENE_GRID) -> None:
        super().__init__()
        self.grid = int(grid)
        self.norm = nn.LayerNorm(c_in)
        self.proj = nn.Linear(c_in, dim)
        self.pos = nn.Parameter(torch.zeros(self.grid * self.grid, dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.bev_grad_scale = 1.0

    def forward(self, grid_bev: torch.Tensor) -> torch.Tensor:
        x = grid_bev.float()
        if self.bev_grad_scale != 1.0 and x.requires_grad:
            x = _GradScale.apply(x, self.bev_grad_scale)
        x = F.adaptive_avg_pool2d(x, (self.grid, self.grid)).flatten(2).transpose(1, 2)      # [B, G*G, C]
        return self.proj(self.norm(x)) + self.pos[None]


# ------------------------------------------------------------------------------------------------- the module
def ego_dit(status: torch.Tensor) -> torch.Tensor:
    """SpecDrive status_feature [B, 8] = [cmd one-hot 4, vx, vy, ax, ay] -> FPD-Drive's refiner ego order
    [vx, vy, ax, ay, cmd 4] (ego_status[:, -1, 3:] of the release)."""
    s = status.float()
    return torch.cat([s[:, 4:8], s[:, 0:4]], -1)


class FPDRefinerNet(nn.Module):
    """SceneTokens + ResidualDiT + FPDScorer.  Student (agent.ck_student, on the student BEV) or frozen teacher (on a
    privileged teacher BEV).  Built with a forked RNG: no global RNG is consumed."""

    def __init__(self, seed: int = 0, c_in: int = 256, steps: int = DDIM_STEPS, grid: int = SCENE_GRID) -> None:
        super().__init__()
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + SEED_OFFSET)
            self.tok = SceneTokens(c_in, FPD_DIM, grid)
            self.dit = ResidualDiT(num_inference_steps=steps)
            self.scorer = FPDScorer()
        self.init_report: Dict = {}

    def param_counts(self) -> Dict[str, int]:
        return {k: sum(p.numel() for p in m.parameters()) for k, m in (("tok", self.tok), ("dit", self.dit),
                                                                       ("scorer", self.scorer))}

    def scene(self, grid_bev: torch.Tensor) -> torch.Tensor:
        return self.tok(grid_bev)

    def sample_draws(self, parents: torch.Tensor, ctx: torch.Tensor, status: torch.Tensor, noise: torch.Tensor,
                     steps: Optional[int] = None) -> torch.Tensor:
        """parents [B, K, P, 3], noise [B, K, D, P, 3] -> draws [B, K, D, P, 3] = parent + sampled residual (no
        gradient; FPD-Drive: proposals are detached, the DiT is trained only by its own losses)."""
        B, K, Dn = noise.shape[:3]
        base = parents.detach().float()[:, :, None].expand(-1, -1, Dn, -1, -1).reshape(B, K * Dn, *parents.shape[2:])
        out = self.dit.sample(base, ctx.detach().float(), ego_dit(status), noise.reshape(B, K * Dn,
                                                                                         *parents.shape[2:]), steps)
        out[..., 2] = wrap_angle(out[..., 2])
        return out.reshape(B, K, Dn, *parents.shape[2:])

    def score(self, cand: torch.Tensor, ctx: torch.Tensor, status: torch.Tensor) -> torch.Tensor:
        return self.scorer(cand, ctx, status)


# ------------------------------------------------------------------------------------------------- pool / noise
def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def pool96(parents: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    """parents [B, K, P, 3], draws [B, K, D, P, 3] -> [B, K * (1 + D), P, 3] with column c = k * (1 + D) + v (v 0 =
    the parent bitwise, v >= 1 = draw v - 1): the CK2 base layout when K = 16, D = 5."""
    B, K = parents.shape[:2]
    allc = torch.cat([parents.float()[:, :, None], draws.float()], 2)
    return allc.reshape(B, K * allc.shape[2], *parents.shape[2:])


def token_noise(tokens: Sequence[str], shape: Tuple[int, ...], seed: int, epoch: int = -1, stream: str = NOISE_STREAM,
                device=None) -> torch.Tensor:
    """standard normal noise [len(tokens), *shape], a pure function of (token, seed, epoch, stream) (anchor_sampler
    .token_rng: blake2b-keyed PCG64; no global RNG).  epoch -1 = the evaluation noise (FPD-Drive: one fixed noise per
    scene)."""
    from .anchor_sampler import token_rng

    arr = np.stack([token_rng(str(t), int(epoch), int(seed), stream).standard_normal(shape).astype(np.float32)
                    for t in tokens], 0)
    return torch.from_numpy(arr).to(device) if device is not None else torch.from_numpy(arr)


def status_keys(status: torch.Tensor) -> list:
    """a deterministic per-sample key from the status bytes (the evaluation fallback when no token is given)."""
    s = status.detach().float().cpu().numpy()
    return ["status:" + row.tobytes().hex() for row in s]


# ------------------------------------------------------------------------------------------------- DiT training (FPD Eq. 5-6)
def mean_xy_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mean over poses of the xy distance: a [..., P, 3], b [..., P, 3] (broadcast) -> [...]."""
    return (a[..., :2] - b[..., :2]).norm(dim=-1).mean(-1)


def match_targets(parents: torch.Tensor, targets: torch.Tensor, tmask: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                               torch.Tensor]:
    """FPD-Drive Eq. 5: optimal bipartite matching of the J targets to distinct parents, cost = mean xy distance.
    parents [B, K, P, 3], targets [B, J, P, 3], tmask [B, J] -> par_idx int64 [B, J] (-1 unmatched), ok bool [B, J]."""
    from scipy.optimize import linear_sum_assignment

    B, J = targets.shape[:2]
    cost = mean_xy_dist(targets.detach().float()[:, :, None], parents.detach().float()[:, None]).cpu().numpy()
    m = tmask.detach().bool().cpu().numpy()
    idx = np.full((B, J), -1, np.int64)
    for b in range(B):
        rows = np.flatnonzero(m[b])
        if len(rows) == 0:
            continue
        c = cost[b][rows]
        if not np.isfinite(c).all():
            c = np.where(np.isfinite(c), c, 1e9)
        r, k = linear_sum_assignment(c)
        idx[b, rows[r]] = k
    t = torch.from_numpy(idx).to(targets.device)
    return t, t >= 0


def d1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FPD-Drive's d1: mean over poses of |dx| + |dy| + |dpsi| (metres and radians summed; heading wrapped)."""
    dxy = (a[..., :2] - b[..., :2]).abs().sum(-1)
    dpsi = wrap_angle(a[..., 2] - b[..., 2]).abs()
    return (dxy + dpsi).mean(-1)


def dit_loss(net: FPDRefinerNet, parents: torch.Tensor, targets: torch.Tensor, tmask: torch.Tensor,
             ctx: torch.Tensor, status: torch.Tensor, keys: Sequence[str], seed: int, epoch: int, step: int
             ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """FPD-Drive Eq. 5-6 on the matched (target, parent) pairs: residual r = y - sg(b) (heading wrapped); one timestep
    per scene t ~ U{0..T-1} and epsilon ~ N(0, I) from token-keyed generators (stream 'fpd.train', keyed by epoch and
    step); L_diff = MSE(eps_hat, eps), L_traj = d1(b + r0_hat, y).  Returns (L_diff, L_traj, stats); both 0 (with the
    DiT in the graph) when no pair is valid."""
    B, J = targets.shape[:2]
    dev = targets.device
    pidx, ok = match_targets(parents, targets, tmask)
    b = torch.gather(parents.detach().float(), 1, pidx.clamp(min=0)[..., None, None].expand(-1, -1,
                                                                                         *parents.shape[2:]))
    y = targets.detach().float()
    y = torch.where(ok[..., None, None], y, b)          # unmatched / masked rows: r = 0 (never NaN * 0)
    r = y - b
    r = torch.cat([r[..., :2], wrap_angle(r[..., 2:3])], -1)
    from .anchor_sampler import token_rng

    gens = [token_rng(str(k), int(epoch), int(seed), f"{TRAIN_STREAM}|{int(step)}") for k in keys]
    t = torch.as_tensor([int(g.integers(0, net.dit.num_train_timesteps)) for g in gens], dtype=torch.long, device=dev)
    eps = torch.from_numpy(np.stack([g.standard_normal(tuple(r.shape[1:])).astype(np.float32) for g in gens], 0)
                           ).to(dev)
    eps_hat, r0 = net.dit.train_terms(r, ctx, ego_dit(status), t, eps)
    okf = ok.float()
    n = okf.sum().clamp(min=1.0)
    l_diff = (((eps_hat.float() - eps) ** 2).mean((-1, -2)) * okf).sum() / n
    l_traj = (d1(b + r0.float(), y) * okf).sum() / n
    if not bool(ok.any()):
        l_diff = 0.0 * eps_hat.sum()
        l_traj = 0.0 * r0.sum()
    stats = {"fpd/n_pairs": float(ok.sum()), "fpd/res_abs": float((r.abs().mean((-1, -2)) * okf).sum() / n)}
    return l_diff, l_traj, stats


# ------------------------------------------------------------------------------------------------- multi-target (FPD S3)
def expected_pdms(prob: torch.Tensor) -> torch.Tensor:
    """[..., 5] probabilities (CK_KEYS nc, dac, ep, ttc, comfort) -> NC * DAC * (5 EP + 5 TTC + 2 C) / 12."""
    nc, dac, ep, ttc, c = (prob[..., i] for i in range(5))
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * c) / 12.0


def select_targets(bank: torch.Tensor, prob: torch.Tensor, ok: torch.Tensor, k_max: int = 4, min_ade: float = 1.0,
                   thr: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """FPD-Drive Sec. 3.3 target selection on a scored bank: keep candidates that pass the safety check (here the
    privileged teacher's P(NC) >= thr and P(DAC) >= thr; FPD-Drive: PDM-Closed's NC and DAC), seed with the best
    expected PDMS, then greedily add the candidate with the largest minimum ADE (mean xy distance) to the selected set
    while that ADE >= min_ade, up to k_max.  bank [B, M, P, 3], prob [B, M, 5], ok [B, M] -> targets [B, k_max, P, 3],
    tmask [B, k_max]."""
    B, M = bank.shape[:2]
    safe = ok.bool() & (prob[..., 0] >= thr) & (prob[..., 1] >= thr) & torch.isfinite(bank).all(-1).all(-1)
    score = torch.where(safe, expected_pdms(prob.float()), torch.full_like(prob[..., 0], -1.0))
    out = bank.new_zeros((B, k_max) + tuple(bank.shape[2:]))
    tm = torch.zeros(B, k_max, dtype=torch.bool, device=bank.device)
    ade = mean_xy_dist(bank[:, :, None].float(), bank[:, None].float())                    # [B, M, M]
    for b in range(B):
        if not bool(safe[b].any()):
            continue
        sel = [int(score[b].argmax())]
        while len(sel) < k_max:
            dmin = ade[b][:, sel].min(-1).values
            dmin = torch.where(safe[b], dmin, torch.full_like(dmin, -1.0))
            dmin[sel] = -1.0
            j = int(dmin.argmax())
            if float(dmin[j]) < min_ade:
                break
            sel.append(j)
        out[b, :len(sel)] = bank[b, sel]
        tm[b, :len(sel)] = True
    return out, tm


# ------------------------------------------------------------------------------------------------- teacher run format
FPD_TEACHER_TRAINER = "train_fpd_teacher"
FPD_TEACHER_ARMS = {"det": "T", "map": "M"}          # T: det / BEVFusion BEV (kd_bev_0), M: map / ReSMap BEV (kd_bev_1)


def save_fpd_teacher(run_dir, net: FPDRefinerNet, cfg: Dict, tag: str = "last") -> str:
    """run_dir/ckpt_<tag>.pt = {'state_dict', 'config'}; config.json is written by the trainer (same cfg)."""
    from pathlib import Path

    p = Path(run_dir) / f"ckpt_{tag}.pt"
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"state_dict": net.state_dict(), "config": dict(cfg)}, tmp)
    tmp.replace(p)
    return str(p)


def load_fpd_teacher(run_dir, which: str = "last", device="cpu", require_done: bool = True
                     ) -> Tuple[FPDRefinerNet, Dict]:
    """A finished FPD teacher run (tools/ck/fpd/train_fpd_teacher.py): config.json {trainer 'train_fpd_teacher', arm
    'T' | 'M', score_keys, ep_target, seed, steps, grid, ...}, ckpt_<which>.pt, done.json (require_done).  Strict load,
    eval mode, frozen.  init_report['ckpt_sha16'] = the first 16 hex of the checkpoint's sha256."""
    import hashlib
    import json
    from pathlib import Path

    rd = Path(run_dir)
    cj = rd / "config.json"
    if not cj.is_file():
        raise ValueError(f"FPD teacher {rd}: no config.json")
    cfg = json.loads(cj.read_text())
    if cfg.get("trainer") != FPD_TEACHER_TRAINER:
        raise ValueError(f"FPD teacher {rd}: trainer {cfg.get('trainer')!r} != {FPD_TEACHER_TRAINER!r}")
    if require_done and not (rd / "done.json").is_file():
        raise ValueError(f"FPD teacher {rd}: no done.json (training not finished)")
    ck = rd / f"ckpt_{which}.pt"
    blob = ck.read_bytes()
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    net = FPDRefinerNet(seed=int(cfg.get("seed", 0)), steps=int(cfg.get("steps", DDIM_STEPS)),
                        grid=int(cfg.get("grid", SCENE_GRID)))
    net.load_state_dict(sd["state_dict"], strict=True)
    net.requires_grad_(False)
    net.eval()
    net.init_report = {"ckpt": str(ck), "ckpt_sha16": hashlib.sha256(blob).hexdigest()[:16], "arm": cfg.get("arm")}
    return net.to(device), cfg
