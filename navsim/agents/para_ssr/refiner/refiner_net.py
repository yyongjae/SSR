"""M5 refiner network (IMPL_SPEC §3.8): gate + longitudinal / lateral correction controls for the M4 decoder.  torch.

Inputs (token-batched: T tokens x K drafts, draft b = t * K + k)
  bev  : [T, C_in, 50, 100] S-grid scene feature (arm 'T': teacher bev_feature after teacher_to_s_grid) or None ('none')
  tau0 : [T, K, 8, 3] drafts (N frame: x forward, y left, heading CCW, rad; poses at t = 0.5..4.0 s)
  v0, a0 : [T] ego speed |(vx, vy)| [m/s] and longitudinal acceleration ax [m/s^2] at t0
  eds  : [T, 4] ego dynamic state (vx, vy, ax, ay) at t0, ego frame
  cmd  : [T] driving command index, order (left, straight, right, unknown) as PARA-SSR; anything else -> all zeros
Outputs (dict)
  gate_logit [T, K] (p_g = sigmoid; computed from the DETACHED draft-token trunk output, so the gate BCE never
  reaches the trunk), z_lon [T, K, 6] (decoder raw longitudinal controls), w_lat [T, K, 6] (raw lateral controls),
  ud [T*K, 69], geom (corridor.CorridorGeom).
  The lon / lat output layers are zero-initialised, so an untrained net decodes to tau0 exactly (identity).

Architecture (IMPL_SPEC §3.8; ~3.37 M parameters + adapter)
  adapter  (adapters.py; the ONLY arm-dependent module)     -> F [T, 64, 50, 100]
  corridor (corridor.py) X [T*K, 70, 48, 17] -> Conv3x3 70->96, GN(8), GELU; 96->96, GN, GELU; 96->128, GN, GELU
           -> station tokens: [T*K, 48, 128*17] (per station: channel-major, lateral-minor flatten) -> Linear -> 192
  draft token: u_d [69] -> Linear 69->192, GELU, Linear 192->192
  global   (corridor.global_tokens) [T, 200, 64] -> Linear 64->192 + learned positional embedding [200, 192]
  4 pre-LN layers, d = 192, 6 heads, FFN 768 (GELU):  x += SA(LN x) over [draft + 48 stations];
           x += CA(LN x, LN_mem g) to the token's 200 global tokens (queries of the K drafts of a token are attended
           together per token -- exact, cross-attention is per query); x += FFN(LN x).  Final LayerNorm.
  heads    gate: MLP 192->128->1 on sg(x_draft);  lon / lat: MLP [x_draft || mean_j x_station_j] 384->256->6.
  Initialisation: the trunk is built under torch.manual_seed(seed), the adapter afterwards under seed + ADAPTER_SEED_OFFSET,
  so the trunk initialisation is IDENTICAL across arms for the same seed (tests/test_refiner_net.py).  Arms M / TM
  (AMENDMENT 6): adapters.build_adapter(seed=seed + ADAPTER_SEED_OFFSET) -- TM's det branch == the T adapter and TM's
  map branch == the M adapter of the same seed (adapters.py docstring, tests/test_stageT4.py); arm T / none unchanged.
  bev input per arm: T / M [T, 256, 50, 100] (BEVFusion / ReSMap S grid), TM [T, 512, 50, 100] (det then map).

Draft token features u_d (69-d, UD_NAMES, exact order):
   0  v0 / 15                      1  a0 / 4
   2  vx / 15     3  vy / 15       4  ax / 4       5  ay / 4                      (eds, ego frame at t0)
   6..9    command one-hot (left, straight, right, unknown)
  10..41   pose k = 1..8, interleaved per pose: x_k / 40, y_k / 10, cos h_k, sin h_k  (index 10 + 4 (k-1) + {0,1,2,3})
  42..49   segment speed u_k / 15, u_k = (S_{k+1} - S_k) / 0.5 s, k = 0..7 (chord arc length S of the draft path)
  50..57   segment accel (u_k - u_{k-1}) / 0.5 s / 4, u_{-1} = v0, clipped to [-5, 5]
  58..65   smooth-path curvature at pose k = 1..8 / 0.2 [1/m], clipped to [-5, 5] (i.e. |kappa| <= 1 /m, the decoder clip)
  66       S_8 / 40
  67       flag near_stop   = S_8 < 2 m (straight corridor extension)
  68       flag ext_clipped = the constant-curvature corridor extension was clipped by kappa_limit(v_end)
  NaN inputs (missing ego state) are replaced by 0.

Deviation from IMPL_SPEC §3.8 (documented, interface unchanged)
  * The spec lists u_d as "v0, a0, command one-hot(4), 8 x (x/40, y/10, cos h, sin h), per-segment speed/accel/curvature
    (8 each), S_8/40, flags(2)" = 65 values but states 69-d; the 4 missing values are the ego (vx, vy, ax, ay) of the
    design draft (architecture_draft_v0 M5, "u_d is really 69-d" in critic_impl.md issue 14), included here.
  * The two flags are not named in the spec ("flags(2)"); the design draft names "near stop, extrapolation used".  Every
    corridor extrapolates beyond S_8 (S_look > S_8 always), so the second flag is "extension curvature clipped".
  * A final LayerNorm after the last layer (pre-LN transformer) is added; 384 parameters, identical in every arm.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .adapters import OUT_CH, build_adapter
from .corridor import N_GEO, N_GLOBAL, N_LAT, N_ST, corridor_geometry, corridor_tensor, global_tokens
from .decoder import N_FREE
from .geometry import N_POSE, T_POSE

UD_DIM = 69
UD_NAMES = (["v0_15", "a0_4", "vx_15", "vy_15", "ax_4", "ay_4", "cmd_left", "cmd_straight", "cmd_right", "cmd_unknown"]
            + [f"p{k}_{c}" for k in range(1, N_POSE + 1) for c in ("x_40", "y_10", "cos_h", "sin_h")]
            + [f"u{k}_15" for k in range(N_POSE)] + [f"acc{k}_4" for k in range(N_POSE)]
            + [f"kappa{k}_0p2" for k in range(1, N_POSE + 1)] + ["S8_40", "near_stop", "ext_clipped"])
assert len(UD_NAMES) == UD_DIM
N_CMD = 4
ADAPTER_SEED_OFFSET = 7919
FEAT_SCALE_CLIP = 5.0
KAPPA_MAX_FEAT = 0.2      # curvature feature scale [1/m] (~ geometry.KAPPA_MAX = 0.213)


# ----------------------------------------------------------------------------------------------- features
def draft_features(tau0: torch.Tensor, v0: torch.Tensor, a0: torch.Tensor, eds: torch.Tensor, cmd: torch.Tensor,
                   geom) -> torch.Tensor:
    """u_d [B, 69] (order: module docstring / UD_NAMES).  tau0 [B, 8, 3]; v0, a0 [B]; eds [B, 4]; cmd [B] (int);
    geom = corridor_geometry(tau0)."""
    B = tau0.shape[0]
    dt = tau0.dtype
    nz = lambda x: torch.nan_to_num(x.to(dt), nan=0.0, posinf=0.0, neginf=0.0)
    v0, a0, eds = nz(v0).reshape(B), nz(a0).reshape(B), nz(eds).reshape(B, 4)
    cmd = cmd.reshape(B).long()
    ok = (cmd >= 0) & (cmd < N_CMD)
    onehot = torch.zeros(B, N_CMD, dtype=dt, device=tau0.device)
    onehot[ok, cmd[ok]] = 1.0
    pose = torch.stack([tau0[..., 0] / 40.0, tau0[..., 1] / 10.0, torch.cos(tau0[..., 2]), torch.sin(tau0[..., 2])],
                       -1).reshape(B, 4 * N_POSE)
    S = geom.S.to(dt)
    u = (S[:, 1:] - S[:, :-1]) / T_POSE
    acc = (u - torch.cat([v0[:, None], u[:, :-1]], 1)) / T_POSE / 4.0
    kap = geom.kappa_knots.to(dt) / KAPPA_MAX_FEAT
    clip = lambda x: torch.clamp(x, -FEAT_SCALE_CLIP, FEAT_SCALE_CLIP)
    return torch.cat([
        v0[:, None] / 15.0, a0[:, None] / 4.0, eds[:, :2] / 15.0, eds[:, 2:] / 4.0, onehot, pose,
        u / 15.0, clip(acc), clip(kap), S[:, -1:] / 40.0,
        geom.near_stop.to(dt)[:, None], geom.ext_clipped.to(dt)[:, None]], 1)


# ----------------------------------------------------------------------------------------------- blocks
class RefinerLayer(nn.Module):
    """Pre-LN: self-attention over [draft + stations], cross-attention to the token's global tokens, FFN."""

    def __init__(self, d: int, n_heads: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.ln_sa = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_ca = nn.LayerNorm(d)
        self.ln_mem = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, n_tokens: int, n_drafts: int) -> torch.Tensor:
        B, L, d = x.shape
        h = self.ln_sa(x)
        x = x + self.drop(self.sa(h, h, h, need_weights=False)[0])
        q = self.ln_ca(x).reshape(n_tokens, n_drafts * L, d)
        m = self.ln_mem(mem)
        x = x + self.drop(self.ca(q, m, m, need_weights=False)[0].reshape(B, L, d))
        return x + self.drop(self.ffn(self.ln_ff(x)))


def _mlp(d_in: int, d_hid: int, d_out: int, zero_last: bool = False) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
    if zero_last:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


class RefinerNet(nn.Module):
    """Stage-T refiner (see module docstring).  arm in {'T', 'none', 'M', 'TM'}; norm_mean / norm_std [256] =
    BEVFusion stats (arms T, TM); map_norm_mean / map_norm_std [256] = ReSMap stats (arms M, TM)."""

    def __init__(self, arm: str = "T", norm_mean=None, norm_std=None, d_model: int = 192, n_heads: int = 6,
                 d_ffn: int = 768, n_layers: int = 4, dropout: float = 0.0, seed: int = 0, map_norm_mean=None,
                 map_norm_std=None):
        super().__init__()
        self.arm = arm
        self.d_model = d_model
        feat_ch = OUT_CH
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.enc = nn.Sequential(
                nn.Conv2d(feat_ch + N_GEO, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                nn.Conv2d(96, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU())
            self.station = nn.Linear(128 * N_LAT, d_model)
            self.draft_mlp = nn.Sequential(nn.Linear(UD_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.global_proj = nn.Linear(feat_ch, d_model)
            self.global_pos = nn.Parameter(torch.randn(N_GLOBAL, d_model) * 0.02)
            self.layers = nn.ModuleList([RefinerLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model)
            self.gate_head = _mlp(d_model, 128, 1)
            self.lon_head = _mlp(2 * d_model, 256, N_FREE, zero_last=True)
            self.lat_head = _mlp(2 * d_model, 256, N_FREE, zero_last=True)
            torch.manual_seed(seed + ADAPTER_SEED_OFFSET)
            if arm in ("T", "none"):          # exactly as runs 1-3
                self.adapter = build_adapter(arm, norm_mean, norm_std)
            else:
                self.adapter = build_adapter(arm, norm_mean, norm_std, map_norm_mean, map_norm_std,
                                             seed=seed + ADAPTER_SEED_OFFSET)

    @property
    def needs_bev(self) -> bool:
        return bool(getattr(self.adapter, "needs_bev", False))

    def trunk_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("adapter.")]

    def param_counts(self) -> Dict[str, int]:
        adapter = sum(p.numel() for p in self.adapter.parameters())
        trunk = sum(p.numel() for _, p in self.trunk_named_parameters())
        return {"adapter": adapter, "trunk": trunk, "total": adapter + trunk}

    def forward(self, bev: Optional[torch.Tensor], tau0: torch.Tensor, v0: torch.Tensor, a0: torch.Tensor,
                eds: torch.Tensor, cmd: torch.Tensor) -> Dict[str, torch.Tensor]:
        T, K = tau0.shape[:2]
        B = T * K
        dev = tau0.device
        # geometry and draft features in float32, outside autocast (DraftPath solves, exact grid coordinates)
        with torch.autocast(device_type=dev.type, enabled=False):
            flat = tau0.reshape(B, N_POSE, 3).float()
            geom = corridor_geometry(flat)
            rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
            ud = draft_features(flat, rep(v0.float()), rep(a0.float()), rep(eds.float()), rep(cmd), geom)
        feat = self.adapter(bev, n_tokens=T)                                          # [T, 64, 50, 100]
        X = corridor_tensor(feat, geom, T, K)                                         # [B, 70, 48, 17]
        Hc = self.enc(X)                                                              # [B, 128, 48, 17]
        st = self.station(Hc.permute(0, 2, 1, 3).reshape(B, N_ST, -1))               # [B, 48, d]
        x = torch.cat([self.draft_mlp(ud.to(st.dtype))[:, None], st], 1)              # [B, 49, d]
        mem = self.global_proj(global_tokens(feat)) + self.global_pos                 # [T, 200, d]
        for layer in self.layers:
            x = layer(x, mem, T, K)
        x = self.norm_out(x)
        xd = x[:, 0]
        gate = self.gate_head(xd.detach())[:, 0]
        h = torch.cat([xd, x[:, 1:].mean(1)], -1)
        return {"gate_logit": gate.reshape(T, K), "z_lon": self.lon_head(h).reshape(T, K, N_FREE),
                "w_lat": self.lat_head(h).reshape(T, K, N_FREE), "ud": ud, "geom": geom}
